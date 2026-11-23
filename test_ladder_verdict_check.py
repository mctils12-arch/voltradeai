"""
Tests for scripts/ladder_verdict_check.py — the check that a registered gate
verdict on production is recorded in datacore/signal_ladder.json.

The fixtures marked "as read 2026-11-23" are the shapes /api/diag/earnings_qoq
and /api/diag/github_gate2 answered that night, cut to the fields the check
reads. The ladder statuses of the DRIFT cases are the ones the file carried
before that session: the check would have reported both.

Run: python3 -m pytest test_ladder_verdict_check.py -q
"""
import importlib.util
import json
import os

import pytest

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO_ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lvc = _load("ladder_verdict_check")
hc = _load("session_health_check")

QOQ = {"probe": "earnings_qoq", "path": "latest.first_sufficient_run.root_verdict", "gate": 2}
GH = {"probe": "github_gate2", "path": "latest.first_sufficient_run.verdict", "gate": 2}

# as read 2026-11-23
QOQ_PAYLOAD = {"probe": "earnings_qoq", "latest": {
    "root_verdict": {"result": "FAIL", "horizons_passing": ["1"]},
    "first_sufficient_run": {"generated_at": "2026-11-22T23:05:30+00:00", "root_verdict": "FAIL", "pairs": 1844}}}
GH_PAYLOAD = {"probe": "github_gate2", "latest": {
    "verdict": "FAIL",
    "first_sufficient_run": {"generated_at": "2026-11-18T00:27:21.428Z", "verdict": "FAIL", "n_clusters": 15}}}
# a probe before its first sufficient run
GH_WAITING = {"probe": "github_gate2", "latest": {"verdict": "WAITING", "first_sufficient_run": None}}


def root(rid, status, spec):
    return {"id": rid, "name": rid, "status": status, "registered_verdict": spec}


def test_the_two_unread_verdicts_of_2026_11_23_are_drift():
    a = lvc.classify(root("sec_8k_earnings_language", "gate2_pending", QOQ), QOQ_PAYLOAD)
    b = lvc.classify(root("github_org_engineering_momentum", "gate1_pass", GH), GH_PAYLOAD)
    for r in (a, b):
        assert r["outcome"] == lvc.DRIFT
        assert r["verdict"] == "FAIL"
        assert "registered FAIL at gate 2" in r["detail"]
    assert "gate2_pending" in a["detail"] and "gate1_pass" in b["detail"]


def test_a_recorded_fail_agrees_as_gate_fail_or_killed():
    assert lvc.classify(root("x", "gate2_fail", QOQ), QOQ_PAYLOAD)["outcome"] == lvc.AGREES
    assert lvc.classify(root("x", "killed", QOQ), QOQ_PAYLOAD)["outcome"] == lvc.AGREES
    # a fail recorded at the WRONG gate is not a record of this verdict
    assert lvc.classify(root("x", "gate1_fail", QOQ), QOQ_PAYLOAD)["outcome"] == lvc.DRIFT


def test_a_registered_pass_needs_that_gate_passed_or_a_later_gate():
    passed = {"latest": {"first_sufficient_run": {"verdict": "PASS"}}}
    assert lvc.classify(root("x", "gate2_pass", GH), passed)["outcome"] == lvc.AGREES
    assert lvc.classify(root("x", "gate3_pending", GH), passed)["outcome"] == lvc.AGREES
    assert lvc.classify(root("x", "gate3_fail", GH), passed)["outcome"] == lvc.AGREES
    for stale in ("gate2_pending", "gate1_pass", "gate2_fail", "killed", "raw_only"):
        assert lvc.classify(root("x", stale, GH), passed)["outcome"] == lvc.DRIFT, stale


def test_no_verdict_yet_is_not_drift_whatever_the_status():
    for payload in (GH_WAITING, {"latest": None}, {}, {"latest": {"first_sufficient_run": {"verdict": 0.4}}}):
        r = lvc.classify(root("x", "gate1_pass", GH), payload)
        assert r["outcome"] == lvc.NO_VERDICT and r["verdict"] is None


def test_the_registered_verdict_is_read_not_the_latest_rerun():
    # weekly re-runs are informational: a later PASS must not overrule a registered FAIL
    payload = {"latest": {"verdict": "PASS", "first_sufficient_run": {"verdict": "FAIL"}}}
    assert lvc.classify(root("x", "gate2_fail", GH), payload)["outcome"] == lvc.AGREES
    assert lvc.classify(root("x", "gate2_pass", GH), payload)["outcome"] == lvc.DRIFT


def test_an_unanswered_probe_is_unreadable_never_a_guess():
    for payload in (None, "502 Bad Gateway", []):
        r = lvc.classify(root("x", "gate2_pending", QOQ), payload)
        assert r["outcome"] == lvc.UNREADABLE
        assert "not checked" in r["detail"]


@pytest.mark.parametrize("spec", [
    None, "earnings_qoq", {"probe": "earnings_qoq", "path": "a.b"},
    {"probe": "../audit", "path": "a", "gate": 2}, {"probe": "x", "path": "", "gate": 2},
    {"probe": "x", "path": ".a", "gate": 2}, {"probe": "x", "path": "a", "gate": 0},
    {"probe": "x", "path": "a", "gate": 6}, {"probe": "x", "path": "a", "gate": True},
    {"probe": "x", "path": "a", "gate": "2"},
])
def test_a_malformed_block_fails_loud(spec):
    with pytest.raises(ValueError):
        lvc.validate_spec("r", spec)


def test_check_all_asks_each_probe_once_and_keeps_root_order():
    asked = []

    def fetch(probe):
        asked.append(probe)
        return {"earnings_qoq": QOQ_PAYLOAD, "github_gate2": None}[probe]

    out = lvc.check_all(fetch, roots=[
        root("a", "gate2_pending", QOQ), root("b", "gate2_fail", QOQ), root("c", "gate1_pass", GH)])
    assert asked == ["earnings_qoq", "github_gate2"]
    assert [(r["id"], r["outcome"]) for r in out] == [("a", lvc.DRIFT), ("b", lvc.AGREES), ("c", lvc.UNREADABLE)]


def test_summarize_names_every_drift_and_counts_the_rest():
    drift = lvc.classify(root("a", "gate2_pending", QOQ), QOQ_PAYLOAD)
    ok = lvc.classify(root("b", "gate2_fail", QOQ), QOQ_PAYLOAD)
    waiting = lvc.classify(root("c", "gate1_pass", GH), GH_WAITING)
    down = lvc.classify(root("d", "gate1_pass", GH), None)
    needs, line = lvc.summarize([drift, ok, waiting, down])
    assert needs and "earnings_qoq" in line and "gate2_pending" in line
    needs, line = lvc.summarize([ok, waiting, down])
    assert not needs
    assert "3 root(s)" in line and "1 recorded" in line and "1 with no verdict yet" in line and "github_gate2" in line


def test_health_check_warns_on_drift_and_never_alarms():
    drift = lvc.classify(root("a", "gate2_pending", QOQ), QOQ_PAYLOAD)
    ok = lvc.classify(root("b", "gate2_fail", QOQ), QOQ_PAYLOAD)
    f = hc.check_ladder_verdicts([drift, ok])
    assert f["severity"] == hc.WARN and f["label"] == "Ladder verdicts" and f.get("fix")
    assert hc.check_ladder_verdicts([ok])["severity"] == hc.OK
    assert hc.check_ladder_verdicts([])["severity"] == hc.OK
    assert hc.check_ladder_verdicts(None)["severity"] == hc.WARN
    # a probe that did not answer is said, and is not a reason to page anyone
    down = hc.check_ladder_verdicts([lvc.classify(root("d", "gate1_pass", GH), None)])
    assert down["severity"] == hc.OK and "did not answer" in down["detail"]


def test_the_real_ladder_file_blocks_are_valid_and_name_real_probes():
    roots = lvc.load_roots()
    ids = {r["id"] for r in roots}
    assert {"sec_8k_earnings_language", "github_org_engineering_momentum"} <= ids
    for r in roots:
        probe, path, gate = lvc.validate_spec(r["id"], r["registered_verdict"])
        # the probe is one server/*.ts registers under exactly this name
        hits = [f for f in os.listdir(os.path.join(REPO_ROOT, "server"))
                if f.endswith(".ts") and not f.endswith(".test.ts")
                and f'probe: "{probe}"' in open(os.path.join(REPO_ROOT, "server", f), encoding="utf-8").read()]
        assert hits, f"{r['id']}: no server module answers diag probe {probe!r}"
        assert "first_sufficient_run" in path, f"{r['id']}: the path must read the REGISTERED verdict"


def test_the_real_ladder_records_the_two_verdicts_with_their_layer_of_death():
    with open(lvc.LADDER_PATH, encoding="utf-8") as f:
        by_id = {r["id"]: r for r in json.load(f)["roots"]}
    for rid, payload in (("sec_8k_earnings_language", QOQ_PAYLOAD), ("github_org_engineering_momentum", GH_PAYLOAD)):
        r = by_id[rid]
        assert lvc.classify(r, payload)["outcome"] == lvc.AGREES
        assert r["current_gate"] == 2
        assert "LAYER OF DEATH: gate 2 (SIGNAL)" in r["note"]
