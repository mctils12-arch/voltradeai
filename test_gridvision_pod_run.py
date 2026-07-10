"""test_gridvision_pod_run.py — scripts/gridvision_train/pod_run.py battery.

Pure plan-building + injected-runner tests need no network; one END-TO-END test
pushes a real (throwaway, local, /tmp) git repo through the WHOLE plan with real
git commands — no RunPod, no GitHub — to catch shell/argument mistakes in the
git command sequence itself BEFORE it is ever run against a live paid pod.
"""
import importlib.util
import os
import subprocess

import pytest

_ROOT = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location(
    "gridvision_pod_run", os.path.join(_ROOT, "scripts", "gridvision_train", "pod_run.py"))
pod_run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pod_run)


def test_result_branch_name_unique_per_job_and_sanitized():
    assert pod_run.result_branch_name("gv-div4-ks") == "gridvision-pod-result-gv-div4-ks"
    # prior sessions reused ONE branch name across jobs (leftover-branch clutter,
    # experiments.md 2026-07-09 LEFTOVERS) -- different job ids must differ
    assert pod_run.result_branch_name("a") != pod_run.result_branch_name("b")
    # never lets an odd job id break the git command
    assert pod_run.result_branch_name("weird id/with spaces!") == "gridvision-pod-result-weird-id-with-spaces"


def test_find_best_weights_walks_nested_ultralytics_layout(tmp_path):
    nested = tmp_path / "tower_v0" / "weights"
    nested.mkdir(parents=True)
    (nested / "best.pt").write_bytes(b"fake-weights")
    (nested / "last.pt").write_bytes(b"fake-weights-2")
    found = pod_run.find_best_weights(str(tmp_path))
    assert found == str(nested / "best.pt")


def test_find_best_weights_none_when_absent(tmp_path):
    assert pod_run.find_best_weights(str(tmp_path)) is None


def test_build_push_plan_without_weights():
    plan = pod_run.build_push_plan("/repo", "br1", "/tmp/log.txt")
    assert plan["branch"] == "br1"
    assert plan["copy_ops"] == [("/tmp/log.txt", "/repo/gv_train.log")]
    add_cmd = next(c for c in plan["commands"] if len(c) > 3 and c[3] == "add")
    assert "gv_best.pt" not in add_cmd
    push_cmd = plan["commands"][-1]
    assert push_cmd[-3:] == ["origin", "HEAD:br1", "--force"]


def test_build_push_plan_with_weights_and_remote_url():
    plan = pod_run.build_push_plan("/repo", "br2", "/tmp/log.txt",
                                   weights_path="/tmp/best.pt",
                                   remote_url="https://TOKEN@github.com/o/r.git")
    assert ("/tmp/best.pt", "/repo/gv_best.pt") in plan["copy_ops"]
    add_cmd = [c for c in plan["commands"] if len(c) > 3 and c[3] == "add"][0]
    assert "gv_best.pt" in add_cmd and "gv_train.log" in add_cmd
    push_cmd = plan["commands"][-1]
    assert push_cmd[-3:] == ["https://TOKEN@github.com/o/r.git", "HEAD:br2", "--force"]
    # the token-bearing URL is a PUSH ARGUMENT, never folded into the commit
    # message or any other logged string in the plan
    commit_cmd = [c for c in plan["commands"] if len(c) > 3 and c[3] == "commit"][0]
    assert "TOKEN" not in " ".join(commit_cmd)


def test_execute_push_plan_runs_copies_then_commands_in_order():
    calls = []
    plan = {"copy_ops": [("a", "b"), ("c", "d")],
            "commands": [["git", "one"], ["git", "two"]]}
    result = pod_run.execute_push_plan(
        plan, runner=lambda cmd: calls.append(("run", cmd)),
        copier=lambda s, d: calls.append(("copy", s, d)))
    assert result == {"ok": True, "failed_step": None, "error": None}
    assert calls == [("copy", "a", "b"), ("copy", "c", "d"),
                     ("run", ["git", "one"]), ("run", ["git", "two"])]


def test_execute_push_plan_reports_copy_failure_without_raising():
    plan = {"copy_ops": [("missing", "dest")], "commands": [["git", "x"]]}
    ran = []
    result = pod_run.execute_push_plan(
        plan, runner=lambda cmd: ran.append(cmd),
        copier=lambda s, d: (_ for _ in ()).throw(FileNotFoundError("no such file")))
    assert result["ok"] is False and result["failed_step"] == "copy"
    assert ran == []  # never reaches git commands after a copy failure


def test_execute_push_plan_reports_command_failure_and_stops():
    plan = {"copy_ops": [], "commands": [["git", "ok"], ["git", "boom"], ["git", "never"]]}

    def runner(cmd):
        if cmd[-1] == "boom":
            raise RuntimeError("git exited 1")

    ran = []
    result = pod_run.execute_push_plan(
        plan, runner=lambda cmd: (ran.append(cmd), runner(cmd))[-1], copier=lambda s, d: None)
    assert result["ok"] is False and result["failed_step"] == "git boom"
    assert ["git", "never"] not in ran


def test_main_reports_worse_of_train_and_push_failure(tmp_path, monkeypatch):
    # training itself failed (nonzero) -> that code wins even if push "succeeds"
    monkeypatch.setattr(pod_run, "run_training", lambda *a, **k: 7)
    monkeypatch.setattr(pod_run, "find_best_weights", lambda out_dir: None)
    monkeypatch.setattr(pod_run, "execute_push_plan", lambda plan, **k: {"ok": True, "failed_step": None, "error": None})
    rc = pod_run.main(["--job-id", "t1", "--out", str(tmp_path), "--repo-dir", str(tmp_path), "--", "--epochs", "1"])
    assert rc == 7


def test_main_reports_push_failure_when_training_succeeded(tmp_path, monkeypatch):
    monkeypatch.setattr(pod_run, "run_training", lambda *a, **k: 0)
    monkeypatch.setattr(pod_run, "find_best_weights", lambda out_dir: None)
    monkeypatch.setattr(pod_run, "execute_push_plan",
                        lambda plan, **k: {"ok": False, "failed_step": "git push", "error": "boom"})
    rc = pod_run.main(["--job-id", "t2", "--out", str(tmp_path), "--repo-dir", str(tmp_path), "--", "--epochs", "1"])
    assert rc == 3


# ── end-to-end: REAL git, local-only (a bare repo in tmp_path; no network) ──

def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.mark.skipif(subprocess.run(["which", "git"], capture_output=True).returncode != 0,
                    reason="git not available")
def test_end_to_end_push_against_local_bare_remote(tmp_path):
    """Exercises the REAL git command sequence build_push_plan produces, against
    a throwaway local bare repo standing in for GitHub — proves the orphan-
    branch + add + commit + push mechanics are correct BEFORE ever spending a
    GPU dollar running this for real against RunPod."""
    bare = tmp_path / "remote.git"
    bare.mkdir()
    _git(str(bare), "init", "--bare")

    work = tmp_path / "work"
    work.mkdir()
    _git(str(work), "init")
    _git(str(work), "remote", "add", "origin", str(bare))
    (work / "README.md").write_text("seed\n")
    _git(str(work), "add", "README.md")
    _git(str(work), "-c", "user.email=t@t.com", "-c", "user.name=t", "commit", "-m", "seed")

    log_path = tmp_path / "train_stdout.log"
    log_path.write_text("epoch 1/1 map50=0.5\n")
    weights_path = tmp_path / "best.pt"
    weights_path.write_bytes(b"fake-weights")

    plan = pod_run.build_push_plan(str(work), "gridvision-pod-result-e2e",
                                   str(log_path), weights_path=str(weights_path))
    result = pod_run.execute_push_plan(plan)
    assert result == {"ok": True, "failed_step": None, "error": None}

    # verify the pushed branch in the bare "remote" really carries both files
    listing = subprocess.run(
        ["git", "--git-dir", str(bare), "ls-tree", "-r", "--name-only",
         "gridvision-pod-result-e2e"],
        capture_output=True, text=True, check=True).stdout
    assert "gv_train.log" in listing
    assert "gv_best.pt" in listing
    # the orphan branch must NOT carry the original repo's history/files
    assert "README.md" not in listing
