#!/usr/bin/env python3
"""pod_run.py — GRID VISION on-pod ENTRYPOINT: run train.py, then push whatever
survives (train log always; gv_best.pt if training got that far) to a dedicated
git branch so results outlive the pod's DELETE.

WHY THIS EXISTS (gap this closes): train.py's run_full() already tries a keyless
host (weight_sink.upload_weight) for the weights file, but that host has 503'd
before (v1's WEIGHTS note) and NEITHER it nor train.py captures the actual
training LOG anywhere durable — the launcher (runpod_launch.py) does not fetch
pod logs, so a failed/crashed run leaves NOTHING to diagnose. Prior sessions
(div2/div3, research/experiments.md 2026-07-09) solved this ad hoc with an
un-committed git-push wrapper each time ("gridvision-pod-result" branches) —
this is that pattern, finally COMPILED INTO CODE (CLAUDE.md EDGE DOCTRINE #3)
instead of re-invented (and re-risked) on every GPU dollar spent.

DESIGN: a training FAILURE must still attempt the result push (the log is the
only diagnostic if train.py crashes) — so this never uses `set -e`-style
short-circuiting; train.py runs as a subprocess whose exit code is captured,
not raised. The git plan (branch name, files, commands) is PURE and unit-
tested without touching git or the network; only `main()` executes it, via an
injectable `runner` so the on-pod behavior itself is testable against a local
throwaway repo (no RunPod, no GitHub) before ever spending a GPU dollar on it.

ON-POD USAGE (the whole pod --cmd becomes just this + the clone step):
  cd /workspace/repo && python3 scripts/gridvision_train/pod_run.py \\
      --job-id gv-div4-ks --remote-url "https://${GRIDVISION_GH_PAT}@github.com/mctils12-arch/voltradeai.git" \\
      -- --model yolov8s.pt --aug strong --regions USA_AZ_Tucson,NZ_Dunedin,NZ_Gisborne,NZ_Palmerston_North,NZ_Rotorua,NZ_Tauranga,USA_CT_Hartford,USA_NC_Clyde,USA_NC_Wilmington \\
      --holdout-region USA_KS_Colwich_Maize --epochs 80 --out /workspace/gv_out

Everything after the bare `--` is forwarded verbatim to train.py.
"""
import argparse
import os
import re
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))


def result_branch_name(job_id):
    """job_id -> a safe git branch name, unique per job (prior sessions reused a
    single 'gridvision-pod-result' branch across jobs, leaving stale leftovers
    with no per-job trace — research/experiments.md 2026-07-09 LEFTOVERS note).
    Strips anything not [A-Za-z0-9._-] so an odd job_id can never break the git
    command. Pure."""
    safe = re.sub(r"[^A-Za-z0-9._-]", "-", str(job_id)).strip("-") or "job"
    return f"gridvision-pod-result-{safe}"


def find_best_weights(out_dir):
    """Locate the first best.pt under out_dir (Ultralytics' own layout), or None.
    Thin IO wrapper (not pure), kept tiny and separately overridable for tests."""
    for root, _dirs, files in os.walk(out_dir):
        if "best.pt" in files:
            return os.path.join(root, "best.pt")
    return None


def build_push_plan(repo_dir, branch, log_path, weights_path=None,
                    commit_message=None, remote_url=None, git_user="GRID VISION pod",
                    git_email="gridvision-pod@voltradeai.com"):
    """PURE: the ordered list of argv command lists that push `log_path` (always)
    and `weights_path` (if not None) to an orphan branch `branch` in `repo_dir`.
    Never touches git/network — just builds the plan. `remote_url` carries the
    token (never logged elsewhere); if None, pushes to the existing 'origin'
    remote instead (e.g. for local testing against a throwaway bare repo)."""
    commit_message = commit_message or f"gridvision result: {branch}"
    dest_log = os.path.join(repo_dir, "gv_train.log")
    cmds = [
        ["git", "-C", repo_dir, "config", "user.email", git_email],
        ["git", "-C", repo_dir, "config", "user.name", git_user],
        ["git", "-C", repo_dir, "checkout", "--orphan", branch],
        ["git", "-C", repo_dir, "reset"],  # unstage everything the orphan checkout inherited
    ]
    copy_ops = [(log_path, dest_log)]
    add_paths = ["gv_train.log"]
    if weights_path:
        dest_w = os.path.join(repo_dir, "gv_best.pt")
        copy_ops.append((weights_path, dest_w))
        add_paths.append("gv_best.pt")
    cmds.append(["git", "-C", repo_dir, "add", *add_paths])
    cmds.append(["git", "-C", repo_dir, "commit", "-m", commit_message])
    push_target = remote_url if remote_url else "origin"
    cmds.append(["git", "-C", repo_dir, "push", push_target, f"HEAD:{branch}", "--force"])
    return {"copy_ops": copy_ops, "commands": cmds, "branch": branch}


def _default_runner(cmd):
    """Real execution: subprocess.run, raising on nonzero (the caller decides how
    much of the plan to attempt). Kept as the injectable default so tests never
    touch a real shell."""
    subprocess.run(cmd, check=True)


def execute_push_plan(plan, runner=_default_runner, copier=None):
    """Execute a build_push_plan() result: copy files, then run each git command
    in order. `runner`/`copier` are injectable (tests use fakes; on-pod uses the
    real ones). Returns {"ok": bool, "failed_step": str or None, "error": str or None}
    — never raises, so a push failure can be logged and swallowed rather than
    losing the training result entirely to an uncaught exception."""
    import shutil
    copier = copier or shutil.copyfile
    try:
        for src, dst in plan["copy_ops"]:
            copier(src, dst)
    except Exception as e:
        return {"ok": False, "failed_step": "copy", "error": repr(e)}
    for cmd in plan["commands"]:
        try:
            runner(cmd)
        except Exception as e:
            return {"ok": False, "failed_step": " ".join(cmd), "error": repr(e)}
    return {"ok": True, "failed_step": None, "error": None}


def run_training(train_args, out_dir, log_path):
    """Run train.py as a SEPARATE PROCESS (never import-and-call in-process) so a
    training crash cannot take this wrapper down with it — the whole point of
    this file is to survive a training failure and still push the log. Returns
    the process return code. stdout+stderr both land in log_path, unbuffered."""
    train_py = os.path.join(_HERE, "train.py")
    cmd = [sys.executable, train_py, "--out", out_dir, *train_args]
    with open(log_path, "wb") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return proc.returncode


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--job-id", required=True)
    p.add_argument("--out", default="/workspace/gv_out")
    p.add_argument("--repo-dir", default=_REPO_ROOT)
    p.add_argument("--remote-url", default=None,
                  help="push URL carrying the token (e.g. https://<PAT>@github.com/owner/repo.git); "
                       "defaults to the repo's existing 'origin' if omitted")
    p.add_argument("train_args", nargs=argparse.REMAINDER,
                  help="everything after -- is forwarded to train.py verbatim")
    args = p.parse_args(argv)
    train_args = args.train_args[1:] if args.train_args[:1] == ["--"] else args.train_args

    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "train_stdout.log")
    print(f"[pod_run] training: {train_args}", flush=True)
    train_rc = run_training(train_args, args.out, log_path)
    print(f"[pod_run] train.py exit code: {train_rc}", flush=True)

    weights = find_best_weights(args.out)
    branch = result_branch_name(args.job_id)
    plan = build_push_plan(args.repo_dir, branch, log_path, weights_path=weights,
                           commit_message=f"gridvision result: {args.job_id} (train_rc={train_rc})",
                           remote_url=args.remote_url)
    print(f"[pod_run] pushing result to branch {branch} (weights={'yes' if weights else 'no'})", flush=True)
    result = execute_push_plan(plan)
    print(f"[pod_run] push result: {result}", flush=True)
    # the training result code is what matters upstream (a push failure is logged,
    # not silently swallowed into a false "success"), but never masks a push
    # failure as a training success either — report the WORSE of the two.
    return train_rc if train_rc != 0 else (0 if result["ok"] else 3)


if __name__ == "__main__":
    sys.exit(main())
