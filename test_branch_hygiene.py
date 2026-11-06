"""Branch hygiene: scripts/branch_hygiene.py (plan) + scripts/branch_cleanup_apply.sh
(apply), run weekly by .github/workflows/branch-hygiene.yml (2026-11-06).

The human approved the cleanup on 2026-10-28 (merged branches and data-backup
deleted, unique branches tagged archive/<name> then deleted; main,
celestial-catalog-data, open PRs, tips under 24 h and branches named in code
kept) and asked again on 2026-11-06. These tests pin that policy rule by rule,
then run plan + apply end to end against a throwaway bare repository standing
in for GitHub: a branch that moved after planning is kept, a reused archive tag
name gets a suffix, a tag the remote refuses keeps its branch (and the run goes
red), and nothing changes without --apply.
"""
import importlib.util
import json
import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent
PLANNER = ROOT / "scripts" / "branch_hygiene.py"
APPLY = ROOT / "scripts" / "branch_cleanup_apply.sh"
WORKFLOW = ROOT / ".github" / "workflows" / "branch-hygiene.yml"

_spec = importlib.util.spec_from_file_location("branch_hygiene", PLANNER)
bh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bh)

NOW = 1_762_462_800          # 2026-11-06T21:00:00Z
DAY = 86_400
OLD = NOW - 30 * DAY
TIP = "a" * 40
KEEP = ["main", "celestial-catalog-data", "gv-res-*"]


def classify(name, tip=TIP, tip_time=OLD, **kw):
    args = dict(now=NOW, keep_globs=KEEP, open_heads={}, prs_by_head={}, main_set=set(),
                referenced=set())
    args.update(kw)
    return bh.classify(name, tip, tip_time, **args)


def a_pr(num, state="closed", merged=False, head_sha=TIP):
    return {"number": num, "state": state, "merged": merged, "head_ref": "claude/x",
            "head_sha": head_sha, "head_repo": "o/r"}


# ── the policy, rule by rule ───────────────────────────────────────────────
def test_main_and_keep_globs_are_never_touched():
    assert classify("main") == ("protected", "skip")
    assert classify("celestial-catalog-data") == ("protected", "skip")
    assert classify("gv-res-test-zz") == ("protected", "skip")


def test_open_pr_heads_are_kept_however_old():
    assert classify("claude/x", open_heads={"claude/x": 9}) == ("open-pr#9", "skip")
    assert classify("claude/x", tip_time=NOW - 60, open_heads={"claude/x": 9}) == ("open-pr#9", "skip")


def test_tips_younger_than_24_hours_are_kept():
    assert classify("claude/x", tip_time=NOW - 3600) == ("recent<24h", "skip")
    assert classify("claude/x", tip_time=NOW - DAY - 1) == ("unique(no-pr)", "tag+delete")


def test_a_tip_missing_locally_is_never_acted_on():
    assert classify("claude/x", tip_time=None) == ("unknown(no-local-object)", "skip")


def test_branches_named_in_code_are_kept():
    assert classify("claude/x", referenced={"claude/x"}) == ("referenced-in-code", "skip")


def test_a_branch_at_its_merged_head_is_deleted_without_a_tag():
    prs = {"claude/x": [a_pr(5, merged=True), a_pr(9, merged=True)]}
    assert classify("claude/x", prs_by_head=prs) == ("merged-pr#9", "delete")
    assert classify("claude/x", prs_by_head=prs, main_set={TIP}) == ("merged-pr#9", "delete")


def test_a_tip_already_on_main_is_deleted_without_a_tag():
    assert classify("claude/x", main_set={TIP}) == ("merged-ancestor", "delete")


def test_everything_else_is_archived_before_deletion():
    moved = {"claude/x": [a_pr(5, merged=True, head_sha="b" * 40)]}
    assert classify("claude/x", prs_by_head=moved) == ("unique(#5:merged)", "tag+delete")
    closed = {"claude/x": [a_pr(5, merged=True, head_sha="b" * 40), a_pr(9)]}
    assert classify("claude/x", prs_by_head=closed) == ("unique(#9:closed,#5:merged)", "tag+delete")
    assert classify("claude/x") == ("unique(no-pr)", "tag+delete")


def test_data_backup_is_deleted_only_at_the_tip_the_human_approved():
    approved = bh.APPROVED_DELETE_NOTAG["data-backup"]
    assert approved == "a23d708f3c2cb5110b38ae85157c51830336eb40"  # ops/branch_cleanup_plan_2026-10-28.tsv
    assert classify("data-backup", tip=approved, referenced={"data-backup"}) == ("data-backup", "delete-notag")
    # Moved (the backup job writing again): the ordinary rules apply, and
    # backup_to_github.py names the branch, so it is kept.
    assert classify("data-backup", tip="c" * 40, referenced={"data-backup"}) == ("referenced-in-code", "skip")


def test_only_this_repositorys_pr_heads_count():
    prs = [dict(a_pr(1, state="open"), head_repo="someone/fork"),
           dict(a_pr(2, state="open"), head_ref="claude/y"),
           dict(a_pr(3, merged=True), head_ref="claude/z")]
    open_heads, by_head = bh.index_prs(prs, "o/r")
    assert open_heads == {"claude/y": 2}
    assert set(by_head) == {"claude/y", "claude/z"}


def test_the_committed_keep_file():
    globs = bh.read_keep(str(ROOT / "ops" / "branch_keep.txt"))
    assert {"main", "celestial-catalog-data", "gridvision-pod-result*", "gv-res-*"} <= set(globs)


def test_a_keep_file_without_main_refuses_to_plan(tmp_path):
    bad = tmp_path / "keep.txt"
    bad.write_text("celestial-catalog-data\n")
    with pytest.raises(SystemExit) as exc:
        bh.read_keep(str(bad))
    assert exc.value.code == 2


def test_the_plan_format_is_the_2026_10_28_format():
    first = (ROOT / "ops" / "branch_cleanup_plan_2026-10-28.tsv").read_text().splitlines()[0]
    assert tuple(first.split("\t")) == bh.HEADER


def test_the_planner_never_writes_a_ref():
    code = PLANNER.read_text()
    assert not re.search(r'git\(\s*"(push|update-ref|tag|branch|fetch)"', code)
    assert "gh\", \"api\", \"-X" not in code


# ── end to end, against a throwaway bare "origin" ──────────────────────────
def run(args, cwd, env=None, check=True):
    full_env = {**os.environ, **(env or {})}
    full_env.pop("GITHUB_STEP_SUMMARY", None)  # never write into a real CI summary
    r = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=120, env=full_env)
    if check and r.returncode != 0:
        raise AssertionError(f"{args} -> {r.returncode}\n{r.stdout}\n{r.stderr}")
    return r


def at(when):
    return {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
            "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com",
            "GIT_AUTHOR_DATE": f"@{when} +0000", "GIT_COMMITTER_DATE": f"@{when} +0000"}


def commit(work, label, when, files=None):
    for rel, text in (files or {f"{label.replace('/', '_')}.txt": label}).items():
        p = work / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    run(["git", "add", "-A"], work)
    run(["git", "-c", "commit.gpgsign=false", "commit", "-q", "--no-verify", "-m", label], work, at(when))
    return run(["git", "rev-parse", "HEAD"], work).stdout.strip()


def branch_from(work, name, base, when):
    run(["git", "checkout", "-q", "-b", name, base], work)
    sha = commit(work, name, when)
    run(["git", "checkout", "-q", "main"], work)
    return sha


def remote_refs(origin, kind):
    out = run(["git", "for-each-ref", f"refs/{kind}/", "--format=%(refname) %(objectname)"], origin).stdout
    return {ref.split(f"refs/{kind}/", 1)[1]: sha for ref, sha in (l.split() for l in out.splitlines())}


@pytest.fixture
def repo(tmp_path):
    origin = tmp_path / "origin.git"
    work = tmp_path / "work"
    run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], tmp_path)
    run(["git", "init", "-q", "-b", "main", str(work)], tmp_path)
    run(["git", "remote", "add", "origin", str(origin)], work)
    base = commit(work, "base", OLD, {
        "ops/branch_keep.txt": "main\ncelestial-catalog-data   # mirror\n",
        "src/code.py": 'BRANCH = "claude/named"\n',
        # Session logs, docs and prose never protect a branch; each exclusion
        # is exercised on its own (a non-.md file under research/, an .md file
        # outside research/ and docs/).
        "research/state.json": '{"branch": "claude/unique"}\n',
        "docs/note.txt": "claude/closed\n",
        "src/README.md": "claude/clash\n",
    })
    t = {"base": base}
    t["merged"] = branch_from(work, "claude/merged", base, OLD)
    t["unique"] = branch_from(work, "claude/unique", base, OLD)
    t["closed"] = branch_from(work, "claude/closed", base, OLD)
    t["open"] = branch_from(work, "claude/open", base, OLD)
    t["recent"] = branch_from(work, "claude/recent", base, NOW - 3600)
    t["named"] = branch_from(work, "claude/named", base, OLD)
    t["mirror"] = branch_from(work, "celestial-catalog-data", base, OLD)
    t["moved"] = branch_from(work, "claude/moved", base, OLD)
    t["clash"] = branch_from(work, "claude/clash", base, OLD)
    run(["git", "branch", "claude/ancestor", base], work)
    run(["git", "push", "-q", "origin", "--all"], work)
    # A reused name: archive/claude/clash already holds an older, different tip.
    run(["git", "push", "-q", "origin", f"{base}:refs/tags/archive/claude/clash"], work)
    prs = [
        {"number": 1, "state": "closed", "merged_at": "2026-10-01T00:00:00Z",
         "head": {"ref": "claude/merged", "sha": t["merged"], "repo": {"full_name": "o/r"}}},
        {"number": 2, "state": "closed", "merged_at": None,
         "head": {"ref": "claude/closed", "sha": t["closed"], "repo": {"full_name": "o/r"}}},
        {"number": 3, "state": "open", "merged_at": None,
         "head": {"ref": "claude/open", "sha": t["open"], "repo": {"full_name": "o/r"}}},
        {"number": 4, "state": "closed", "merged_at": "2026-10-02T00:00:00Z",
         "head": {"ref": "claude/moved", "sha": t["moved"], "repo": {"full_name": "o/r"}}},
    ]
    (tmp_path / "prs.json").write_text(json.dumps(prs))
    return {"tmp": tmp_path, "origin": origin, "work": work, "tips": t}


def make_plan(repo):
    plan = repo["tmp"] / "plan.tsv"
    r = run(["python3", str(PLANNER), "--out", str(plan), "--repo", "o/r", "--now", str(NOW),
             "--prs-json", str(repo["tmp"] / "prs.json")], repo["work"])
    rows = {line.split("\t")[0]: line.split("\t")[3:] for line in plan.read_text().splitlines()[1:]}
    return plan, rows, r.stderr


def apply(repo, plan, *flags, check=False):
    return run(["bash", str(APPLY), *flags], repo["work"], {"PLAN": str(plan)}, check=check)


def test_the_plan_from_live_state(repo):
    _, rows, summary = make_plan(repo)
    assert rows == {
        "main": ["protected", "skip"],
        "celestial-catalog-data": ["protected", "skip"],
        "claude/open": ["open-pr#3", "skip"],
        "claude/recent": ["recent<24h", "skip"],
        "claude/named": ["referenced-in-code", "skip"],
        "claude/merged": ["merged-pr#1", "delete"],
        "claude/ancestor": ["merged-ancestor", "delete"],
        "claude/moved": ["merged-pr#4", "delete"],
        "claude/unique": ["unique(no-pr)", "tag+delete"],
        "claude/closed": ["unique(#2:closed)", "tag+delete"],
        "claude/clash": ["unique(no-pr)", "tag+delete"],
    }
    assert "-> delete 3, delete-notag 0, tag+delete 3, skip 5" in summary


def test_dry_run_changes_nothing(repo):
    plan, _, _ = make_plan(repo)
    before = (remote_refs(repo["origin"], "heads"), remote_refs(repo["origin"], "tags"))
    r = apply(repo, plan)
    assert r.returncode == 0, r.stderr
    assert "dry run" in r.stdout
    assert "would tag archive/claude/unique" in r.stdout
    assert (remote_refs(repo["origin"], "heads"), remote_refs(repo["origin"], "tags")) == before


def test_apply_archives_then_deletes_and_keeps_what_moved(repo):
    plan, _, _ = make_plan(repo)
    tips = repo["tips"]
    # Someone pushes to claude/moved after the plan was made.
    run(["git", "checkout", "-q", "claude/moved"], repo["work"])
    moved_now = commit(repo["work"], "late push", NOW - 600)
    run(["git", "push", "-q", "origin", "claude/moved"], repo["work"])
    run(["git", "checkout", "-q", "main"], repo["work"])

    r = apply(repo, plan, "--apply")
    assert r.returncode == 0, r.stdout + r.stderr
    heads = remote_refs(repo["origin"], "heads")
    assert set(heads) == {"main", "celestial-catalog-data", "claude/open", "claude/recent",
                          "claude/named", "claude/moved"}
    assert heads["claude/moved"] == moved_now
    tags = remote_refs(repo["origin"], "tags")
    assert tags["archive/claude/unique"] == tips["unique"]
    assert tags["archive/claude/closed"] == tips["closed"]
    assert tags["archive/claude/clash"] == tips["base"], "an existing archive tag is never moved"
    assert tags[f"archive/claude/clash-{tips['clash'][:12]}"] == tips["clash"]
    assert "archive/claude/merged" not in tags, "merged branches need no archive"
    assert "MOVED (kept): claude/moved" in r.stdout

    again = apply(repo, plan, "--apply")
    assert again.returncode == 0, "a second run finds nothing left to do"
    assert "branches to delete: 0" in again.stdout


def test_a_refused_tag_keeps_its_branch_and_fails_the_run(repo):
    # pre-receive refuses the WHOLE push that carries the bad ref (all or
    # nothing, as a protected-ref rule does), so the batch must be retried ref
    # by ref for the other tags to land.
    hook = repo["origin"] / "hooks" / "pre-receive"
    hook.write_text('#!/bin/sh\nwhile read old new ref; do\n'
                    '  [ "$ref" = "refs/tags/archive/claude/unique" ] && exit 1\n'
                    'done\nexit 0\n')
    hook.chmod(0o755)
    plan, _, _ = make_plan(repo)
    r = apply(repo, plan, "--apply")
    assert r.returncode == 1
    heads = remote_refs(repo["origin"], "heads")
    assert "claude/unique" in heads, "no archive tag, no delete"
    assert "claude/closed" not in heads and "claude/merged" not in heads, \
        "one refused ref must not hold back the rest of its batch"
    assert "KEPT, archive tag did not verify: claude/unique" in r.stdout


def test_a_branch_that_moves_mid_run_is_kept_by_its_lease(repo):
    # The tag push (step 1) lands before the deletes (step 2); this hook moves
    # claude/merged right then, after the executor read the remote. Only the
    # --force-with-lease on the planned tip can still save it.
    base = repo["tips"]["base"]
    hook = repo["origin"] / "hooks" / "post-receive"
    hook.write_text('#!/bin/sh\nwhile read old new ref; do\n'
                    '  [ "$ref" = "refs/tags/archive/claude/unique" ] && '
                    f'git update-ref refs/heads/claude/merged {base}\n'
                    'done\nexit 0\n')
    hook.chmod(0o755)
    plan, _, _ = make_plan(repo)
    r = apply(repo, plan, "--apply")
    assert r.returncode == 0, "a branch the lease kept is not a failure"
    assert remote_refs(repo["origin"], "heads")["claude/merged"] == base
    assert "MOVED during this run (kept by the lease): claude/merged" in r.stdout
    assert "claude/ancestor" not in remote_refs(repo["origin"], "heads")


# ── the workflow ───────────────────────────────────────────────────────────
def load(path):
    data = yaml.safe_load(path.read_text())
    data["on"] = data.pop(True, data.get("on"))  # YAML 1.1 reads `on` as True
    return data


def test_the_workflow_runs_weekly_with_an_explicit_apply():
    wf = load(WORKFLOW)
    assert set(wf["on"]) == {"schedule", "workflow_dispatch"}, "never on push/pull_request"
    (entry,) = wf["on"]["schedule"]
    minute, hour, dom, month, dow = entry["cron"].split()
    assert (dom, month) == ("*", "*") and dow in {"0", "6"}, "weekly, on a weekend"
    assert wf["permissions"] == {"contents": "write", "pull-requests": "read"}
    assert wf["concurrency"] == {"group": "branch-hygiene", "cancel-in-progress": False}
    job = wf["jobs"]["hygiene"]
    assert job["timeout-minutes"] >= 15, "full-history checkout (ciCheckoutBudget.test.ts's floor)"
    checkout = job["steps"][0]
    assert checkout["uses"] == "actions/checkout@v4" and checkout["with"]["fetch-depth"] == 0
    runs = [s.get("run", "").strip() for s in job["steps"]]
    assert runs[1].startswith("python3 scripts/branch_hygiene.py --out ")
    assert runs[2].endswith("bash scripts/branch_cleanup_apply.sh --apply")
    assert job["steps"][1]["env"] == {"GH_TOKEN": "${{ secrets.GITHUB_TOKEN }}"}
