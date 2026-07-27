#!/usr/bin/env python3
"""
runpod_launch.py — Option A launcher: create a RunPod GPU pod, WATCHDOG it to the
cost cap, terminate it, and ledger the ACTUAL spend. Runs from the agent session
(reads RUNPOD_API_KEY from env). Gated: with no key it prints awaiting_key and
does nothing (so it is safe to ship before the key exists).

WHY A WATCHDOG (verified against docs.runpod.io 2026-07-07): RunPod pods have NO
native auto-terminate / TTL field. A pod bills until it is DELETEd or exits. So
`max_runtime_seconds` from the cost-cap gate is NOT a value we hand RunPod — the
cap is enforced two ways, belt-and-suspenders:
  (1) this launcher's wall-clock WATCHDOG — poll GET /pods/{id}; DELETE at the cap
      or when the pod reaches a terminal state; then ledger the real cost; and
  (2) an in-pod `timeout <max_seconds>` wrapper around the training command, so
      the training PROCESS self-kills even if this launcher disconnects.

OPTION A CAVEAT (human chose A, 2026-07-07): the watchdog lives in THIS session.
Keep the session alive for the run. If it dies mid-job, (2) still bounds the
training process, but the idle pod may keep billing until it exits — so prefer a
short --max-hours and watch it. Option B (server-side watchdog in the always-on
Railway process, with boot-recovery) removes that caveat; see research/runpod_ledger.md.

Endpoints (https://rest.runpod.io/v1, Bearer auth — key rides the Authorization
header, NEVER a URL): POST /pods, GET /pods/{id}, DELETE /pods/{id}.

CLI:
  python3 scripts/runpod_launch.py dry-run --gpu "NVIDIA GeForce RTX 4090" \
      --hourly 0.34 --max-hours 4 --workload grid-detector-v0 \
      --image runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04 \
      --cmd "python train.py"          # no key needed; prints gate + create body
  python3 scripts/runpod_launch.py smoke   # cheap nvidia-smi run, proves the path
  python3 scripts/runpod_launch.py launch --job gd-1 --gpu "..." --hourly 0.34 \
      --max-hours 4 --workload grid-detector-v0 --image ... --cmd "python train.py"
"""
import argparse
import json
import os
import shlex
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runpod_budget as rb  # cost-cap gate + ledger (same dir)

API_ROOT = "https://rest.runpod.io/v1"
POLL_SECONDS = 20
# RunPod pod status strings that mean "no longer running / stop watching".
TERMINAL_STATES = {"EXITED", "TERMINATED", "DEAD", "STOPPED", "COMPLETED", "FAILED"}


def api_key():
    """RUNPOD_API_KEY from env, or None. Presence-only — never returned to logs."""
    k = os.environ.get("RUNPOD_API_KEY")
    return k if k else None


def _headers(key):
    # Key rides the Authorization header, never a URL/querystring.
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def build_create_body(
    name,
    gpu,
    image,
    train_cmd,
    max_seconds,
    container_gb=50,
    volume_gb=20,
    cloud_type="COMMUNITY",
    interruptible=True,
    env=None,
):
    """Pure: the POST /pods body. The training command is wrapped in an in-pod
    `timeout <max_seconds>s` so the job self-kills even if the watchdog dies —
    the second half of the cost cap. Never includes the API key.

    COST-CAP SAFETY (repair): the whole train_cmd is handed to `timeout` as a
    SINGLE `bash -lc <quoted>` argument, so `timeout` bounds the ENTIRE command
    tree. Without this, a chained command — `a && b && c` — parses as
    `(timeout Ns a) && b && c`: timeout bounds only `a`, and `b`/`c` run
    UNBOUNDED, silently defeating the in-pod half of the cost cap. Quoting via
    shlex keeps the chain intact as one bounded unit. (The wall-clock watchdog
    DELETE remains the PRIMARY enforcer; this is the belt-and-suspenders backup,
    and it now actually holds for chained commands.)"""
    wrapped = f"timeout {int(max_seconds)}s bash -lc {shlex.quote(train_cmd)}"
    body = {
        "name": name,
        "gpuTypeIds": [gpu],
        "gpuCount": 1,
        "imageName": image,
        "containerDiskInGb": int(container_gb),
        "volumeInGb": int(volume_gb),
        "volumeMountPath": "/workspace",
        "cloudType": cloud_type,
        "interruptible": bool(interruptible),
        "dockerStartCmd": ["/bin/bash", "-lc", wrapped],
    }
    if env:
        body["env"] = dict(env)
    return body


def watchdog_should_terminate(elapsed_s, max_seconds, pod_status, result_ready=False):
    """Pure decision. Returns (should_terminate, reason). Wall-clock cap is the
    primary enforcer (does not depend on parsing RunPod's status field); a
    terminal status ends the watch early on natural completion.

    `result_ready` (BUG FOUND 2026-07-10, gv-div4-ks): the runpod/pytorch base
    image keeps jupyter/ssh alive regardless of the launched command, so a pod
    whose training PROCESS already exited (success OR crash — pod_run.py
    pushes a result either way) still reports RUNNING indefinitely — the
    watchdog idle-bills all the way to the wall-clock cap on every fast
    finish/failure. Prior sessions (div2, research/experiments.md 2026-07-09)
    solved this ad hoc each time by watching for the result branch to appear
    ("reason=result-pushed" in the ledger) and were never committed; this
    finally is. Checked LAST (after pod_status) so a pod that is ALSO
    naturally EXITED still reports "exited", not "result-pushed"."""
    if elapsed_s >= max_seconds:
        return True, "cap"
    if pod_status and str(pod_status).upper() in TERMINAL_STATES:
        return True, "exited"
    if result_ready:
        return True, "result-pushed"
    return False, ""


def actual_cost(elapsed_s, hourly):
    """Pure: real dollars for the elapsed wall-clock at `hourly`."""
    return round(float(hourly) * float(elapsed_s) / 3600.0, 6)


def remote_branch_exists(branch, remote="origin", runner=None, cwd=None):
    """True if `branch` exists on `remote` RIGHT NOW (git ls-remote — works over
    this session's own git proxy, no GitHub token needed). Never raises: a
    network/git hiccup returns False, so the wall-clock cap stays the
    authoritative bound regardless (this is only ever an EARLY-exit signal,
    never a substitute for it). `runner` is injectable for tests."""
    import subprocess
    run = runner or (lambda cmd: subprocess.run(cmd, cwd=cwd, capture_output=True,
                                                 text=True, timeout=30))
    try:
        res = run(["git", "ls-remote", remote, f"refs/heads/{branch}"])
        out = getattr(res, "stdout", "") or ""
        return bool(out.strip())
    except Exception:
        return False


# ---- HTTP (gated behind key presence; network) -----------------------------
def _request(method, url, key, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=_headers(key), method=method)
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read().decode()
        return json.loads(raw) if raw.strip() else {}


def create_pod(body, key):
    """POST /pods -> pod id. Raises on HTTP error."""
    res = _request("POST", f"{API_ROOT}/pods", key, body)
    return res.get("id") or res.get("podId") or res


def get_pod(pod_id, key):
    return _request("GET", f"{API_ROOT}/pods/{pod_id}", key)


def list_pods(key):
    """GET /pods -> list of every pod on the account (all in-flight jobs, any
    job_id). Used by runpod_reap.py to find pods still billing after the
    session that launched them (and was watching the wall-clock watchdog) has
    ended — see the OPTION A CAVEAT in this file's docstring."""
    res = _request("GET", f"{API_ROOT}/pods", key)
    return res if isinstance(res, list) else res.get("pods", [])


def terminate_pod(pod_id, key):
    return _request("DELETE", f"{API_ROOT}/pods/{pod_id}", key)


def _pod_status(pod_obj):
    """Best-effort extraction of a status string (field name confirmed at first
    live smoke test; the wall-clock cap does not depend on this)."""
    if not isinstance(pod_obj, dict):
        return None
    for k in ("desiredStatus", "status", "state", "lastStatus"):
        if pod_obj.get(k):
            return pod_obj[k]
    return None


def run_launch(job_id, workload, gpu, hourly, max_hours, image, train_cmd,
               cloud_type="COMMUNITY", interruptible=True, poll_s=POLL_SECONDS,
               env=None, result_branch=None, result_repo_dir=None):
    """Full lifecycle: gate -> create -> watchdog -> terminate -> ledger. Returns
    a result dict. Refuses (and never touches RunPod) if the gate says no or the
    key is absent.

    `env` (dict) is passed to the pod as RunPod pod env vars — the clean channel
    for a secret the on-pod command needs (e.g. a GH token to clone a private
    repo), so the secret rides the pod env instead of being baked into the
    logged command string. It is NOT the RunPod API key (that stays in the
    Authorization header, never the pod).

    `result_branch` (optional): if the on-pod command pushes its result to a
    git branch (e.g. via pod_run.py), pass that branch name here so the
    watchdog terminates the INSTANT it appears instead of idle-billing to the
    cap. BUG FOUND 2026-07-10 (gv-div4-ks): the base image keeps jupyter/ssh
    alive regardless of the launched command, so a pod whose training exited
    in the first few minutes (success OR crash) still reports RUNNING for the
    full cap — this cost $3.45 for ~8 minutes of real work. Omit to keep the
    old wall-clock-only behavior (e.g. a job with no result-push step)."""
    key = api_key()
    if not key:
        return {"ok": False, "state": "awaiting_key",
                "detail": "RUNPOD_API_KEY absent in this session — start a new session after adding it."}

    rows = rb.load_ledger()
    auth = rb.authorize_job(rows, gpu, hourly, max_hours)
    if not auth["authorized"]:
        return {"ok": False, "state": "refused", "reason": auth["reason"], "detail": auth["detail"]}

    max_seconds = auth["max_runtime_seconds"]
    body = build_create_body(job_id, gpu, image, train_cmd, max_seconds,
                             cloud_type=cloud_type, interruptible=interruptible,
                             env=env)

    # Reserve worst-case in the ledger BEFORE creating the pod.
    rb._append_row({
        "job_id": job_id, "workload": workload, "gpu": gpu, "hourly": hourly,
        "max_hours": max_hours, "reserved_cost": auth["worst_case"], "actual_cost": None,
        "opened_utc": rb._utcstamp(), "closed_utc": None, "note": "launch: creating pod",
    })

    started = time.time()
    pod_id = None
    try:
        pod_id = create_pod(body, key)
        print(f"[launch] pod {pod_id} created; watchdog cap {max_seconds}s (${auth['worst_case']:.2f} worst).")
        while True:
            time.sleep(poll_s)
            elapsed = time.time() - started
            status = None
            try:
                status = _pod_status(get_pod(pod_id, key))
            except Exception as e:  # transient poll error: keep the wall-clock cap authoritative
                print(f"[launch] poll error (continuing on wall-clock): {e}")
            result_ready = bool(result_branch) and remote_branch_exists(result_branch, cwd=result_repo_dir)
            should, reason = watchdog_should_terminate(elapsed, max_seconds, status, result_ready=result_ready)
            print(f"[launch] t={int(elapsed)}s status={status} result_ready={result_ready} -> "
                 f"{'TERMINATE:'+reason if should else 'running'}")
            if should:
                break
    finally:
        if pod_id:
            try:
                terminate_pod(pod_id, key)
                print(f"[launch] pod {pod_id} terminated.")
            except Exception as e:
                print(f"[launch] WARNING: terminate failed for {pod_id}: {e} — CHECK RUNPOD DASHBOARD MANUALLY.")

    elapsed = time.time() - started
    cost = actual_cost(elapsed, hourly)
    rb._append_row({
        "job_id": job_id, "workload": workload, "gpu": gpu, "hourly": hourly,
        "max_hours": max_hours, "reserved_cost": auth["worst_case"], "actual_cost": cost,
        "opened_utc": rb._utcstamp(), "closed_utc": rb._utcstamp(),
        "note": f"closed: elapsed {int(elapsed)}s | closed",
    })
    return {"ok": True, "state": "done", "pod_id": pod_id, "elapsed_s": int(elapsed),
            "actual_cost": cost, "remaining": rb.remaining(rb.load_ledger())}


# ---- CLI -------------------------------------------------------------------
def _cmd_dry_run(a):
    rows = rb.load_ledger()
    auth = rb.authorize_job(rows, a.gpu, a.hourly, a.max_hours)
    print("GATE:", json.dumps(auth, indent=2))
    if auth["authorized"]:
        body = build_create_body("dry-run", a.gpu, a.image, a.cmd, auth["max_runtime_seconds"])
        print("CREATE BODY:", json.dumps(body, indent=2))
    print("KEY:", "present" if api_key() else "ABSENT (start a new session after adding it)")


def _cmd_smoke(a):
    # Cheapest possible end-to-end proof: nvidia-smi in a stock CUDA image, tiny cap.
    res = run_launch("smoke-" + str(int(a.epoch or 0)), "smoke", a.gpu, a.hourly, 0.05,
                     "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
                     "nvidia-smi && sleep 20", poll_s=10)
    print(json.dumps(res, indent=2))
    sys.exit(0 if res.get("ok") else 2)


def parse_env_pairs(pairs):
    """Pure: ['K=V', 'A=b=c'] -> {'K': 'V', 'A': 'b=c'}. Splits on the FIRST '='
    only (values may contain '='). Ignores empty/'='-less entries. Used to turn
    repeated --env flags into the pod env dict."""
    out = {}
    for item in pairs or []:
        if not item or "=" not in item:
            continue
        k, v = item.split("=", 1)
        k = k.strip()
        if k:
            out[k] = v
    return out


def _cmd_launch(a):
    env = parse_env_pairs(getattr(a, "env", None))
    res = run_launch(a.job, a.workload, a.gpu, a.hourly, a.max_hours, a.image, a.cmd,
                     env=env or None, cloud_type=a.cloud_type, interruptible=not a.non_interruptible,
                     result_branch=a.result_branch, result_repo_dir=a.result_repo_dir)
    print(json.dumps(res, indent=2))
    sys.exit(0 if res.get("ok") else 2)


def _build_parser():
    p = argparse.ArgumentParser(description="RunPod pod launcher (gate -> create -> watchdog -> terminate -> ledger)")
    sub = p.add_subparsers(dest="subcmd", required=True)

    def common(sp, with_cmd=True):
        sp.add_argument("--gpu", default="NVIDIA GeForce RTX 4090")
        sp.add_argument("--hourly", type=float, default=0.34)
        sp.add_argument("--max-hours", dest="max_hours", type=float, default=4.0)
        sp.add_argument("--workload", default="grid-detector-v0")
        sp.add_argument("--image", default="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04")
        if with_cmd:
            sp.add_argument("--cmd", default="python train.py")

    common(sub.add_parser("dry-run", help="print gate decision + create body; no key needed"))
    sm = sub.add_parser("smoke", help="cheap nvidia-smi run — proves the launch path end-to-end")
    sm.add_argument("--gpu", default="NVIDIA GeForce RTX 4090")
    sm.add_argument("--hourly", type=float, default=0.34)
    sm.add_argument("--epoch", type=int, default=0, help="pass a timestamp for a unique job id")
    lp = sub.add_parser("launch", help="real run")
    lp.add_argument("--job", required=True)
    lp.add_argument("--env", action="append", metavar="KEY=VAL", default=[],
                    help="pod env var (repeatable) — the clean channel for a secret "
                         "the on-pod --cmd needs (e.g. --env GH_TOKEN=... to clone a "
                         "private repo). NEVER the RunPod API key.")
    lp.add_argument("--cloud-type", dest="cloud_type", default="COMMUNITY", choices=["COMMUNITY", "SECURE"],
                    help="COMMUNITY (default, cheaper, spot-like — can be PREEMPTED mid-run, "
                         "research/experiments.md 2026-07-09 gv-div1-1) or SECURE (non-"
                         "interruptible, costs more) for a long run whose output isn't "
                         "checkpointed and would be lost on preemption")
    lp.add_argument("--non-interruptible", dest="non_interruptible", action="store_true",
                    help="pass with --cloud-type SECURE for a true non-interruptible pod")
    lp.add_argument("--result-branch", dest="result_branch", default=None,
                    help="if the on-pod command pushes a result branch (e.g. pod_run.py's "
                         "gridvision-pod-result-<job>), the watchdog terminates the instant "
                         "it appears instead of idle-billing to the cap (fast finish/crash "
                         "still reports RUNNING on the base image regardless — 2026-07-10 finding)")
    lp.add_argument("--result-repo-dir", dest="result_repo_dir", default=None,
                    help="local repo dir to run 'git ls-remote' from when checking --result-branch "
                         "(defaults to the current working directory)")
    common(lp)
    return p


def main(argv=None):
    a = _build_parser().parse_args(argv)
    {"dry-run": _cmd_dry_run, "smoke": _cmd_smoke, "launch": _cmd_launch}[a.subcmd](a)


if __name__ == "__main__":
    main()
