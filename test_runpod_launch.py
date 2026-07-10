"""
test_runpod_launch.py — pure logic of the RunPod launcher: the create-body
(with the in-pod timeout wrapper), the wall-clock watchdog decision, actual-cost
math, and the gate/key gating. No network, no key, deterministic.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
import runpod_launch as rl  # noqa: E402


# --- build_create_body -------------------------------------------------------

def test_body_wraps_command_in_timeout():
    # The whole train_cmd is handed to `timeout` as a single `bash -lc <quoted>`
    # argument so the cap bounds the ENTIRE command (see the chaining test below).
    b = rl.build_create_body("job1", "NVIDIA GeForce RTX 4090",
                             "img:tag", "python train.py", 14400)
    assert b["dockerStartCmd"] == ["/bin/bash", "-lc",
                                   "timeout 14400s bash -lc 'python train.py'"]


def test_cost_cap_bounds_the_whole_chained_command():
    # COST-CAP SAFETY regression: a chained command must be bounded IN FULL, not
    # just its first segment. The bug: `timeout Ns a && b && c` parses as
    # `(timeout Ns a) && b && c`, leaving b/c UNBOUNDED — the in-pod half of the
    # cost cap silently defeated. The fix wraps the chain in `bash -lc <quoted>`
    # so the whole thing is one bounded unit.
    import shlex
    chain = "pip install x && python train.py && upload.sh"
    b = rl.build_create_body("j", "g", "i", chain, 14400)
    wrapped = b["dockerStartCmd"][2]
    toks = shlex.split(wrapped)
    # timeout <cap>s bash -lc "<the ENTIRE chain as ONE token>"
    assert toks[0] == "timeout" and toks[1] == "14400s"
    assert toks[2] == "bash" and toks[3] == "-lc"
    assert toks[4] == chain, "the full && chain must be a single timeout-bounded arg"
    # there must be exactly ONE timeout and no chain operator OUTSIDE the quoted arg
    assert wrapped.count("timeout ") == 1
    assert "&&" not in wrapped[:wrapped.index("bash -lc")], "no && ahead of the bounded unit"


def test_body_core_fields():
    b = rl.build_create_body("job1", "NVIDIA GeForce RTX 4090", "img:tag", "run", 60)
    assert b["gpuTypeIds"] == ["NVIDIA GeForce RTX 4090"]
    assert b["imageName"] == "img:tag"
    assert b["name"] == "job1"
    assert b["volumeMountPath"] == "/workspace"
    assert b["cloudType"] == "COMMUNITY"
    assert b["interruptible"] is True
    assert b["gpuCount"] == 1


def test_body_no_api_key_leaks_in():
    # The body must never carry the key (it rides the Authorization header only).
    b = rl.build_create_body("j", "g", "i", "c", 10, env={"FOO": "bar"})
    flat = str(b)
    assert "Authorization" not in flat and "Bearer" not in flat
    assert b["env"] == {"FOO": "bar"}


def test_body_seconds_are_integers():
    b = rl.build_create_body("j", "g", "i", "c", 3600.9)
    assert "timeout 3600s" in b["dockerStartCmd"][2]


def test_body_env_rides_the_pod_env_not_the_command():
    # A secret the pod needs (e.g. a clone token) goes in the pod env, NOT baked
    # into the (logged) dockerStartCmd string.
    b = rl.build_create_body("j", "g", "i", "python t.py", 60, env={"GH_TOKEN": "s3cr3t"})
    assert b["env"] == {"GH_TOKEN": "s3cr3t"}
    assert "s3cr3t" not in b["dockerStartCmd"][2], "secret must not leak into the command string"


def test_parse_env_pairs():
    assert rl.parse_env_pairs(["K=V"]) == {"K": "V"}
    assert rl.parse_env_pairs(["A=b=c"]) == {"A": "b=c"}, "split on FIRST '=' only"
    assert rl.parse_env_pairs(["X=1", "Y=2"]) == {"X": "1", "Y": "2"}
    assert rl.parse_env_pairs(["noeq", "", "=val", " K =v"]) == {"K": "v"}
    assert rl.parse_env_pairs(None) == {}


# --- watchdog_should_terminate ----------------------------------------------

def test_watchdog_terminates_at_cap():
    should, reason = rl.watchdog_should_terminate(3600, 3600, "RUNNING")
    assert should is True and reason == "cap"


def test_watchdog_terminates_past_cap():
    assert rl.watchdog_should_terminate(3601, 3600, None)[0] is True


def test_watchdog_terminates_on_terminal_status():
    should, reason = rl.watchdog_should_terminate(10, 3600, "EXITED")
    assert should is True and reason == "exited"
    assert rl.watchdog_should_terminate(10, 3600, "terminated")[0] is True  # case-insensitive


def test_watchdog_keeps_running_below_cap_and_active():
    should, reason = rl.watchdog_should_terminate(100, 3600, "RUNNING")
    assert should is False and reason == ""


def test_watchdog_none_status_below_cap_keeps_running():
    # A poll error (status None) must NOT terminate early — wall-clock is authoritative.
    assert rl.watchdog_should_terminate(100, 3600, None)[0] is False


def test_watchdog_terminates_on_result_ready():
    # BUG FOUND 2026-07-10 (gv-div4-ks): a fast finish/crash still reports RUNNING
    # (base image keeps jupyter/ssh alive) -- result_ready must end the watch early
    # instead of idle-billing to the cap.
    should, reason = rl.watchdog_should_terminate(100, 3600, "RUNNING", result_ready=True)
    assert should is True and reason == "result-pushed"


def test_watchdog_exited_status_wins_over_result_ready():
    # a pod that is BOTH naturally exited AND has a result ready reports "exited",
    # not "result-pushed" -- pod_status is checked first.
    should, reason = rl.watchdog_should_terminate(100, 3600, "EXITED", result_ready=True)
    assert should is True and reason == "exited"


def test_watchdog_result_ready_false_by_default():
    assert rl.watchdog_should_terminate(100, 3600, "RUNNING") == (False, "")


# --- remote_branch_exists -----------------------------------------------------

def test_remote_branch_exists_true_when_ls_remote_has_output():
    class FakeResult:
        stdout = "abc123\trefs/heads/gridvision-pod-result-x\n"

    calls = []
    assert rl.remote_branch_exists("gridvision-pod-result-x",
                                   runner=lambda cmd: (calls.append(cmd), FakeResult())[-1]) is True
    assert calls == [["git", "ls-remote", "origin", "refs/heads/gridvision-pod-result-x"]]


def test_remote_branch_exists_false_when_ls_remote_empty():
    class FakeResult:
        stdout = ""

    assert rl.remote_branch_exists("nope", runner=lambda cmd: FakeResult()) is False


def test_remote_branch_exists_false_on_error_never_raises():
    def boom(cmd):
        raise RuntimeError("network down")

    assert rl.remote_branch_exists("x", runner=boom) is False


# --- actual_cost -------------------------------------------------------------

def test_actual_cost():
    assert rl.actual_cost(3600, 0.34) == 0.34
    assert rl.actual_cost(1800, 0.34) == 0.17
    assert rl.actual_cost(0, 1.64) == 0.0


# --- key gating + gate integration ------------------------------------------

def test_run_launch_awaiting_key_without_env(monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    res = rl.run_launch("j", "w", "g", 0.34, 4, "img", "cmd")
    assert res["ok"] is False and res["state"] == "awaiting_key"


def test_run_launch_refuses_unbounded_even_with_key(monkeypatch, tmp_path):
    # Key present but the gate must still refuse an unbounded job — never touches RunPod.
    monkeypatch.setenv("RUNPOD_API_KEY", "x" * 40)
    monkeypatch.setattr(rl.rb, "LEDGER", str(tmp_path / "l.jsonl"))
    res = rl.run_launch("j", "w", "NVIDIA GeForce RTX 4090", 0.34, 0, "img", "cmd")
    assert res["ok"] is False and res["state"] == "refused" and res["reason"] == "unbounded"


def test_run_launch_terminates_early_when_result_branch_appears(monkeypatch, tmp_path):
    # the actual bug this closes: a pod whose training already finished (fast
    # crash or fast success) must not idle-bill all the way to max_hours just
    # because RunPod's status field never reports it as exited.
    monkeypatch.setenv("RUNPOD_API_KEY", "x" * 40)
    monkeypatch.setattr(rl.rb, "LEDGER", str(tmp_path / "l.jsonl"))
    monkeypatch.setattr(rl, "create_pod", lambda body, key: "pod123")
    monkeypatch.setattr(rl, "get_pod", lambda pod_id, key: {"status": "RUNNING"})
    terminated = []
    monkeypatch.setattr(rl, "terminate_pod", lambda pod_id, key: terminated.append(pod_id))
    monkeypatch.setattr(rl.time, "sleep", lambda s: None)  # no real waiting in the test
    monkeypatch.setattr(rl, "remote_branch_exists", lambda branch, **k: branch == "gv-result-x")

    res = rl.run_launch("j", "w", "NVIDIA GeForce RTX 4090", 0.34, 4, "img", "cmd",
                        result_branch="gv-result-x")
    assert res["ok"] is True
    assert terminated == ["pod123"]
    # elapsed stayed near-zero (time.sleep faked to a no-op) -- proves it did NOT
    # spin through the full 4-hour cap before terminating
    assert res["elapsed_s"] < 5


def test_api_key_absent_by_default(monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    assert rl.api_key() is None
    monkeypatch.setenv("RUNPOD_API_KEY", "abc")
    assert rl.api_key() == "abc"


# --- --cloud-type / --non-interruptible CLI wiring ---------------------------
# (added so a long, non-checkpointed run can request a non-interruptible pod —
# a COMMUNITY/spot pod was PREEMPTED mid-run before with no output recoverable,
# research/experiments.md 2026-07-09 gv-div1-1)

def test_cli_launch_defaults_to_community_interruptible():
    a = rl._build_parser().parse_args(
        ["launch", "--job", "j1", "--gpu", "g", "--hourly", "0.34", "--cmd", "x"])
    assert a.cloud_type == "COMMUNITY"
    assert a.non_interruptible is False


def test_cli_launch_can_request_secure_non_interruptible():
    a = rl._build_parser().parse_args(
        ["launch", "--job", "j2", "--gpu", "g", "--hourly", "0.69", "--cmd", "x",
         "--cloud-type", "SECURE", "--non-interruptible"])
    assert a.cloud_type == "SECURE"
    assert a.non_interruptible is True


def test_cmd_launch_forwards_cloud_type_and_interruptible(monkeypatch):
    captured = {}

    def fake_run_launch(job, workload, gpu, hourly, max_hours, image, cmd, env=None,
                        cloud_type="COMMUNITY", interruptible=True,
                        result_branch=None, result_repo_dir=None):
        captured.update(cloud_type=cloud_type, interruptible=interruptible,
                        result_branch=result_branch, result_repo_dir=result_repo_dir)
        return {"ok": True}

    monkeypatch.setattr(rl, "run_launch", fake_run_launch)
    a = rl._build_parser().parse_args(
        ["launch", "--job", "j3", "--gpu", "g", "--hourly", "0.69", "--cmd", "x",
         "--cloud-type", "SECURE", "--non-interruptible",
         "--result-branch", "gv-result-x", "--result-repo-dir", "/repo"])
    try:
        rl._cmd_launch(a)
    except SystemExit:
        pass
    assert captured == {"cloud_type": "SECURE", "interruptible": False,
                        "result_branch": "gv-result-x", "result_repo_dir": "/repo"}
