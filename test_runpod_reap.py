"""
test_runpod_reap.py — pure logic of the orphaned-RunPod-pod reaper: matching
open ledger jobs against live pods, and the reap decision. No real network;
run_reap()'s network path is exercised via monkeypatched runpod_launch calls.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
import runpod_reap as reap  # noqa: E402
import runpod_launch as rl  # noqa: E402


# --- find_open_jobs -----------------------------------------------------

def test_find_open_jobs_skips_closed():
    rows = [
        {"job_id": "a", "opened_utc": "t1", "actual_cost": None, "reserved_cost": 1.0},
        {"job_id": "a", "opened_utc": "t1", "actual_cost": 0.5, "reserved_cost": 1.0},  # close row
        {"job_id": "b", "opened_utc": "t2", "actual_cost": None, "reserved_cost": 2.0},
    ]
    open_jobs = reap.find_open_jobs(rows)
    assert [j["job_id"] for j in open_jobs] == ["b"]


# --- match_orphans --------------------------------------------------------

def test_match_orphans_still_billing():
    jobs = [{"job_id": "gv-div4-ks", "opened_utc": "t", "reserved_cost": 3.45, "hourly": 0.69}]
    pods = [{"id": "h7sxegyzvjqwqm", "name": "gv-div4-ks", "desiredStatus": "RUNNING"}]
    matches = reap.match_orphans(jobs, pods)
    assert len(matches) == 1
    assert matches[0]["still_billing"] is True
    assert matches[0]["pod"]["id"] == "h7sxegyzvjqwqm"


def test_match_orphans_pod_already_gone():
    jobs = [{"job_id": "gv-div3-ks", "opened_utc": "t", "reserved_cost": 1.38, "hourly": 0.69}]
    matches = reap.match_orphans(jobs, live_pods=[])
    assert matches[0]["still_billing"] is False
    assert matches[0]["pod"] is None


def test_match_orphans_ignores_unrelated_live_pods():
    jobs = [{"job_id": "gv-div4-ks", "opened_utc": "t", "reserved_cost": 3.45, "hourly": 0.69}]
    pods = [{"id": "xyz", "name": "some-other-job", "desiredStatus": "RUNNING"}]
    matches = reap.match_orphans(jobs, pods)
    assert matches[0]["still_billing"] is False


# --- estimate_cost_for_gone_pod / elapsed_seconds_since -------------------

def test_estimate_cost_caps_at_reserved():
    job = {"reserved_cost": 3.45}
    assert reap.estimate_cost_for_gone_pod(job) == 3.45


def test_elapsed_seconds_since():
    assert reap.elapsed_seconds_since(
        "2026-07-10T00:23:51Z", "2026-07-10T02:37:40Z") == 8029.0


# --- run_reap (network monkeypatched) -------------------------------------

def test_run_reap_awaiting_key(monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    res = reap.run_reap()
    assert res["state"] == "awaiting_key"


def test_run_reap_clean_when_no_open_jobs(monkeypatch, tmp_path):
    monkeypatch.setenv("RUNPOD_API_KEY", "fake-key-for-test")
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text('{"job_id": "x", "opened_utc": "t", "actual_cost": 1.0, "reserved_cost": 1.0}\n')
    res = reap.run_reap(ledger_path=str(ledger))
    assert res["state"] == "clean"


def test_run_reap_apply_terminates_and_closes(monkeypatch, tmp_path):
    monkeypatch.setenv("RUNPOD_API_KEY", "fake-key-for-test")
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        '{"job_id": "gv-div4-ks", "workload": "grid-div4-ks", "gpu": "RTX4090", '
        '"hourly": 0.69, "max_hours": 5.0, "reserved_cost": 3.45, "actual_cost": null, '
        '"opened_utc": "2026-07-10T00:23:51Z", "closed_utc": null, "note": "launch"}\n')

    terminated = []
    monkeypatch.setattr(rl, "list_pods", lambda key: [
        {"id": "h7sxegyzvjqwqm", "name": "gv-div4-ks", "desiredStatus": "RUNNING"}])
    monkeypatch.setattr(rl, "terminate_pod", lambda pod_id, key: terminated.append(pod_id))
    monkeypatch.setattr("runpod_budget._utcstamp", lambda: "2026-07-10T02:37:57Z")

    res = reap.run_reap(apply=True, ledger_path=str(ledger))
    assert terminated == ["h7sxegyzvjqwqm"]
    assert res["acted"][0]["job_id"] == "gv-div4-ks"
    assert res["acted"][0]["actual_cost"] > 0

    rows = reap.rb.load_ledger(str(ledger))
    closed = [r for r in rows if r.get("job_id") == "gv-div4-ks" and r.get("actual_cost") is not None]
    assert len(closed) == 1
    assert closed[0]["closed_utc"] == "2026-07-10T02:37:57Z"


def test_run_reap_apply_closes_already_gone_pod_at_reserved_cost(monkeypatch, tmp_path):
    monkeypatch.setenv("RUNPOD_API_KEY", "fake-key-for-test")
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        '{"job_id": "gv-div3-ks", "workload": "grid-div3-ks", "gpu": "RTX4090", '
        '"hourly": 0.69, "max_hours": 2.0, "reserved_cost": 1.38, "actual_cost": null, '
        '"opened_utc": "2026-07-09T11:31:03Z", "closed_utc": null, "note": "launch"}\n')
    monkeypatch.setattr(rl, "list_pods", lambda key: [])
    monkeypatch.setattr("runpod_budget._utcstamp", lambda: "2026-07-10T02:37:57Z")

    res = reap.run_reap(apply=True, ledger_path=str(ledger))
    assert res["acted"][0]["job_id"] == "gv-div3-ks"
    assert res["acted"][0]["actual_cost"] == 1.38  # capped at reserved, no live measurement possible
