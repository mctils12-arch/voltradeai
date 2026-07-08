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


def test_api_key_absent_by_default(monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    assert rl.api_key() is None
    monkeypatch.setenv("RUNPOD_API_KEY", "abc")
    assert rl.api_key() == "abc"
