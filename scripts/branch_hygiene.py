#!/usr/bin/env python3
"""scripts/branch_hygiene.py — plan the remote-branch cleanup from LIVE state.

WHY. Every Claude session pushes a claude/<name> branch and nothing ever
deleted a merged one: 1,056 remote branches on 2026-11-06 (986 claude/*). A ref
list that long slows every GitHub integration picker, and CI's `changes` job
fetches all of them (fetch-depth: 0) on every PR. The human approved the
cleanup on 2026-10-28 ("Delete the merged and superseded branches and the
data-backup branch. Tag the unique ones as archive/<name>, then delete them.
Keep celestial-catalog-data. Skip any branch with an open PR.") and asked for
it again on 2026-11-06 ("Run the branch cleanup"). Sessions cannot push other
refs, so the 2026-10-28 plan was never applied; GitHub Actions can
(GITHUB_TOKEN, contents: write). This rebuilds that plan from live state every
week (.github/workflows/branch-hygiene.yml) and scripts/branch_cleanup_apply.sh
executes it. This script only reads: it never writes a ref.

CLASSES, first match wins (the vocabulary of the 2026-10-28 plan):
  protected            main, or a glob in ops/branch_keep.txt          skip
  unknown(...)         the tip commit is not available locally         skip
  open-pr#N            the head of open PR #N                          skip
  recent<24h           tip committed less than 24 h ago                skip
  data-backup          data-backup at its approved 2026-10-28 tip      delete-notag
  referenced-in-code   the name appears in a tracked code file         skip
  merged-pr#N          the tip is exactly the merged head of PR #N     delete
  merged-ancestor      the tip is already part of main                 delete
  unique(...)          anything else: a closed PR, a tip that moved
                       after its merge, or no PR at all                tag+delete
A merged branch loses nothing (main holds the squash and GitHub keeps
refs/pull/N/head); a unique one is archived as the tag archive/<name> first.

USAGE
  python3 scripts/branch_hygiene.py [--out PLAN.tsv] [--repo OWNER/NAME]
                                    [--keep-file FILE] [--now EPOCH] [--prs-json FILE]
Writes the plan TSV (name, tip_sha, tip_date, class, action) to --out (default
stdout) and a summary to stderr. Exit 2 = could not plan; nothing is guessed.
"""
import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone

MAIN = "main"
RECENT_HOURS = 24
HEADER = ("name", "tip_sha", "tip_date", "class", "action")

# The 2026-10-28 plan deleted data-backup outright (the human named it apart
# from the unique branches it asked to archive). Only at the tip the human saw:
# if the disabled backup job ever writes to it again, the tip moves and the
# ordinary rules apply (backup_to_github.py names it, so it is kept).
APPROVED_DELETE_NOTAG = {"data-backup": "a23d708f3c2cb5110b38ae85157c51830336eb40"}

# Where a branch name does NOT count as code depending on that branch: prose,
# session logs, the cleanup lists, and the cleanup tooling itself, all of which
# name the branches they delete.
REFERENCE_EXCLUDES = (
    ":(exclude)research/**",
    ":(exclude)docs/**",
    ":(exclude)ops/**",
    ":(exclude)*.md",
    ":(exclude)scripts/branch_hygiene.py",
    ":(exclude)scripts/branch_cleanup_apply.sh",
    ":(exclude)scripts/github_branch_cleanup.sh",
    ":(exclude)test_branch_hygiene.py",
)


def die(msg):
    print(f"branch_hygiene: {msg}", file=sys.stderr)
    sys.exit(2)


def git(*args, stdin=None):
    return subprocess.run(["git", *args], input=stdin, capture_output=True, text=True)


def remote_heads(remote):
    """{branch: tip sha} as the remote reports it right now."""
    r = git("ls-remote", "--heads", remote)
    if r.returncode != 0:
        die(f"git ls-remote --heads {remote} failed: {r.stderr.strip()}")
    heads = {}
    for line in r.stdout.splitlines():
        sha, ref = line.split("\t", 1)
        if ref.startswith("refs/heads/"):
            heads[ref[len("refs/heads/"):]] = sha
    return heads


def commit_times(shas):
    """{sha: committer unix time} for every sha whose commit is available locally."""
    if not shas:
        return {}
    r = git("cat-file", "--batch-check=%(objectname) %(objecttype)", stdin="\n".join(shas) + "\n")
    present = [line.split()[0] for line in r.stdout.splitlines() if line.endswith(" commit")]
    if not present:
        return {}
    r = git("log", "--no-walk=unsorted", "--format=%H %ct", "--stdin", stdin="\n".join(present) + "\n")
    if r.returncode != 0:
        die(f"git log failed: {r.stderr.strip()}")
    return {sha: int(ct) for sha, ct in (line.split() for line in r.stdout.splitlines() if line.strip())}


def main_commits(main_sha):
    r = git("rev-list", main_sha)
    if r.returncode != 0:
        die(f"git rev-list {main_sha} failed (is the history fetched?): {r.stderr.strip()}")
    return set(r.stdout.split())


def referenced_names(names):
    """The names that appear anywhere in tracked code (see REFERENCE_EXCLUDES)."""
    names = sorted(n for n in names if n)
    if not names:
        return set()
    fd, path = tempfile.mkstemp(prefix="branch-names-", suffix=".txt")
    with os.fdopen(fd, "w") as fh:
        fh.write("\n".join(names) + "\n")
    try:
        r = git("grep", "-I", "-h", "-o", "-F", "-f", path, "--", ".", *REFERENCE_EXCLUDES)
    finally:
        os.unlink(path)
    if r.returncode not in (0, 1):  # 1 = no match
        die(f"git grep failed ({r.returncode}): {r.stderr.strip()}")
    return set(r.stdout.splitlines()) & set(names)


def read_keep(path):
    try:
        lines = open(path, encoding="utf-8").read().splitlines()
    except OSError as exc:
        die(f"keep file {path} unreadable ({exc}); refusing to plan without it")
    globs = [line.split("#", 1)[0].split()[0] for line in lines if line.split("#", 1)[0].strip()]
    if MAIN not in globs:
        die(f"keep file {path} does not list {MAIN}; refusing to plan")
    return globs


def slim(pr):
    head = pr.get("head") or {}
    return {
        "number": pr["number"],
        "state": pr["state"],
        "merged": bool(pr.get("merged_at")),
        "head_ref": head.get("ref"),
        "head_sha": head.get("sha"),
        "head_repo": (head.get("repo") or {}).get("full_name"),
    }


def fetch_prs(repo):
    """Every PR of the repository (REST, explicit pages)."""
    prs, page = [], 1
    while True:
        r = subprocess.run(["gh", "api", f"repos/{repo}/pulls?state=all&per_page=100&page={page}"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            die(f"gh api pulls page {page} failed: {r.stderr.strip()}")
        batch = json.loads(r.stdout)
        if not batch:
            return prs
        prs += [slim(p) for p in batch]
        page += 1
        if page > 1000:
            die("more than 100,000 PRs listed; refusing to trust the pagination")


def index_prs(prs, repo):
    """(open heads {ref: number}, PRs by head ref), this repository's heads only."""
    open_heads, by_head = {}, {}
    for p in prs:
        if p["head_repo"] != repo or not p["head_ref"]:
            continue
        by_head.setdefault(p["head_ref"], []).append(p)
        if p["state"] == "open":
            open_heads[p["head_ref"]] = p["number"]
    return open_heads, by_head


def classify(name, tip, tip_time, *, now, keep_globs, open_heads, prs_by_head,
             main_set, referenced, recent_hours=RECENT_HOURS):
    """(class, action) for one remote branch. First matching rule wins."""
    if name == MAIN or any(fnmatch.fnmatchcase(name, g) for g in keep_globs):
        return "protected", "skip"
    if tip_time is None:
        return "unknown(no-local-object)", "skip"
    if name in open_heads:
        return f"open-pr#{open_heads[name]}", "skip"
    if now - tip_time < recent_hours * 3600:
        return f"recent<{recent_hours}h", "skip"
    if APPROVED_DELETE_NOTAG.get(name) == tip:
        return name, "delete-notag"
    if name in referenced:
        return "referenced-in-code", "skip"
    prs = sorted(prs_by_head.get(name, []), key=lambda p: -p["number"])
    merged_here = [p["number"] for p in prs if p["merged"] and p["head_sha"] == tip]
    if merged_here:
        return f"merged-pr#{merged_here[0]}", "delete"
    if tip in main_set:
        return "merged-ancestor", "delete"
    if prs:
        seen = ",".join(f"#{p['number']}:{'merged' if p['merged'] else p['state']}" for p in prs)
        return f"unique({seen})", "tag+delete"
    return "unique(no-pr)", "tag+delete"


def plan(heads, times, *, now, keep_globs, open_heads, prs_by_head, main_set, referenced):
    rows = []
    for name in sorted(heads):
        tip = heads[name]
        cls, action = classify(name, tip, times.get(tip), now=now, keep_globs=keep_globs,
                               open_heads=open_heads, prs_by_head=prs_by_head,
                               main_set=main_set, referenced=referenced)
        when = times.get(tip)
        stamp = datetime.fromtimestamp(when, timezone.utc).isoformat() if when is not None else ""
        rows.append((name, tip, stamp, cls, action))
    return rows


def family(cls):
    return re.split(r"[#(<]", cls, maxsplit=1)[0]


def summarize(rows):
    actions = Counter(r[4] for r in rows)
    lines = [f"branch-hygiene plan: {len(rows)} remote branches -> "
             + ", ".join(f"{a} {actions[a]}" for a in ("delete", "delete-notag", "tag+delete", "skip"))]
    for action in ("delete", "tag+delete", "skip"):
        fams = Counter(family(r[3]) for r in rows if r[4] == action)
        if fams:
            lines.append(f"  {action}: " + ", ".join(f"{k} {v}" for k, v in sorted(fams.items())))
    for r in rows:
        if r[4] == "skip":
            lines.append(f"  keep {r[0]} ({r[3]})")
    return "\n".join(lines)


def repo_from_remote(remote):
    url = git("remote", "get-url", remote).stdout.strip()
    m = re.search(r"github\.com[:/]([^/]+/[^/]+?)(?:\.git)?/?$", url)
    if not m:
        die(f"cannot tell the GitHub repository from {remote} ({url!r}); pass --repo")
    return m.group(1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--out", default="-", help="plan TSV path (default: stdout)")
    ap.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY"))
    ap.add_argument("--remote", default="origin")
    ap.add_argument("--keep-file", default="ops/branch_keep.txt")
    ap.add_argument("--now", type=int, default=None, help="epoch seconds (tests)")
    ap.add_argument("--prs-json", default=None, help="the PR list as the REST API returns it (tests)")
    args = ap.parse_args(argv)

    top = git("rev-parse", "--show-toplevel")
    if top.returncode != 0:
        die("not inside a git work tree")
    os.chdir(top.stdout.strip())  # the code scan and the keep file are repository-relative

    repo = args.repo or repo_from_remote(args.remote)
    keep_globs = read_keep(args.keep_file)
    heads = remote_heads(args.remote)
    if MAIN not in heads:
        die(f"{args.remote} has no {MAIN} branch; refusing to plan")
    if args.prs_json:
        prs = [slim(p) for p in json.load(open(args.prs_json, encoding="utf-8"))]
    else:
        prs = fetch_prs(repo)
    open_heads, prs_by_head = index_prs(prs, repo)
    times = commit_times(sorted(set(heads.values())))
    main_set = main_commits(heads[MAIN])
    candidates = [n for n in heads if n != MAIN and not any(fnmatch.fnmatchcase(n, g) for g in keep_globs)]
    referenced = referenced_names(candidates)

    rows = plan(heads, times, now=args.now if args.now is not None else int(time.time()),
                keep_globs=keep_globs, open_heads=open_heads, prs_by_head=prs_by_head,
                main_set=main_set, referenced=referenced)
    text = "\n".join("\t".join(r) for r in [HEADER, *rows]) + "\n"
    if args.out == "-":
        sys.stdout.write(text)
    else:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
    report = summarize(rows)
    print(report, file=sys.stderr)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as fh:
            fh.write("### Branch hygiene plan\n```\n" + report + "\n```\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
