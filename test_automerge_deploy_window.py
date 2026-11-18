"""DEPLOY WINDOW for Claude PR merges (scripts/automerge.sh, 2026-11-06).

The human's rule: nothing restarts the bot during market hours; ship after the
4pm ET close. Every merge to main redeploys (and restarts) the bot, and until
this change ci.yml's automerge job merged the moment CI went green, at any hour
(#1631 merged 2026-11-05 16:12:46Z, bot rebooted 16:14:52Z = 11:14 ET).

These tests pin:
  * the time decision (09:29 / 09:30 / 15:59 / 16:00 ET, weekends, EST vs EDT,
    the 15-minute pre-open guard, and failing closed without a tz database);
  * both merge paths, driven against a fake `gh` so nothing real is ever
    called: the CI path defers inside the window and merges outside it; the
    after-close sweep merges ONE green claude/ PR and nothing that is failed,
    cancelled, pending or missing a check;
  * the frozen YAML: ci.yml still parses, and the automerge job's `needs:` and
    `if:` are byte-identical to what they were before this change, so the same
    checks gate a merge.
"""
import json
import os
import re
import shutil
import stat
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
import yaml

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "scripts" / "automerge.sh"
CI = ROOT / ".github" / "workflows" / "ci.yml"
SWEEP = ROOT / ".github" / "workflows" / "automerge-sweep.yml"
NY = ZoneInfo("America/New_York")
DEFERRED_TEXT = "merge deferred to after the 4pm ET close (DEPLOY WINDOW)"


def utc(text):
    """Epoch seconds for an ISO UTC instant like '2026-11-06T14:30'."""
    return int(datetime.fromisoformat(text).replace(tzinfo=timezone.utc).timestamp())


def call_fn(fn, epoch, env=None):
    """Return code of one function from the script, sourced in a fresh bash."""
    r = subprocess.run(
        ["bash", "-c", f'source "{SCRIPT}"; {fn} {epoch}'],
        capture_output=True, text=True, timeout=30,
        env={**os.environ, **(env or {})},
    )
    return r.returncode


# ── the time decision ──────────────────────────────────────────────────────
# Instants are written in UTC on purpose: the offsets come from the calendar
# (EST = UTC-5 from 2026-11-01, EDT = UTC-4 from 2026-03-08), not from the same
# tz machinery the script uses, so a shared tz bug cannot hide here.
MARKET_HOURS_CASES = [
    # Friday 2026-11-06, EST
    ("2026-11-06T14:29", False),  # 09:29 ET
    ("2026-11-06T14:30", True),   # 09:30 ET, the open
    ("2026-11-06T20:59", True),   # 15:59 ET
    ("2026-11-06T21:00", False),  # 16:00 ET, the close
    # Wednesday 2026-07-15, EDT
    ("2026-07-15T13:29", False),  # 09:29 ET
    ("2026-07-15T13:30", True),   # 09:30 ET
    ("2026-07-15T19:59", True),   # 15:59 ET
    ("2026-07-15T20:00", False),  # 16:00 ET
    # The Mondays after each 2026 DST switch: a fixed-offset implementation
    # gets exactly one of these two wrong.
    ("2026-03-09T13:35", True),   # 09:35 EDT (EST arithmetic: 08:35, closed)
    ("2026-11-02T20:30", True),   # 15:30 EST (EDT arithmetic: 16:30, closed)
    # Weekends
    ("2026-11-07T15:00", False),  # Saturday 10:00 ET
    ("2026-07-19T14:00", False),  # Sunday 10:00 ET
]


@pytest.mark.parametrize("instant,expected", MARKET_HOURS_CASES)
def test_in_market_hours_is_the_nyse_regular_session(instant, expected):
    assert (call_fn("in_market_hours", utc(instant)) == 0) is expected, instant


MERGE_WINDOW_CASES = [
    # EST weekday: open until 09:15, closed 09:15-16:00, open again at 16:00
    ("2026-11-06T14:14", True),   # 09:14 ET
    ("2026-11-06T14:15", False),  # 09:15 ET, pre-open guard starts
    ("2026-11-06T14:29", False),  # 09:29 ET: a merge now would restart the bot after the open
    ("2026-11-06T14:30", False),  # 09:30 ET
    ("2026-11-06T20:59", False),  # 15:59 ET
    ("2026-11-06T21:00", True),   # 16:00 ET
    ("2026-11-06T03:00", True),   # 22:00 ET the evening before
    # EDT weekday
    ("2026-07-15T13:14", True),   # 09:14 ET
    ("2026-07-15T13:15", False),  # 09:15 ET
    ("2026-07-15T13:29", False),  # 09:29 ET
    ("2026-07-15T19:59", False),  # 15:59 ET
    ("2026-07-15T20:00", True),   # 16:00 ET
    # Weekend middays
    ("2026-11-07T15:00", True),
    ("2026-07-19T14:00", True),
]


@pytest.mark.parametrize("instant,open_", MERGE_WINDOW_CASES)
def test_merge_window(instant, open_):
    assert (call_fn("merge_window_open", utc(instant)) == 0) is open_, instant


def test_the_pre_open_guard_covers_the_measured_deploy_lag():
    # 100 boots 2026-11-02..06: merge -> boot 164 s median, 675 s max. The
    # guard must exceed the worst case, or a 09:2x merge restarts the bot
    # after the open.
    out = subprocess.run(["bash", "-c", f'source "{SCRIPT}"; echo $PREOPEN_GUARD_MIN'],
                         capture_output=True, text=True, timeout=30).stdout.strip()
    assert int(out) * 60 > 675


def _no_tz_date(dirpath):
    """A `date` that behaves like one on a host WITHOUT a tz database: GNU date
    quietly answers in UTC for a zone it cannot load. Simulated by a wrapper on
    PATH that forces TZ=UTC, because pointing TZDIR at an empty directory only
    works where `date` honors TZDIR (GNU coreutils; not every runner's `date`
    does — the 2026-11-11 CI runner resolved the zone anyway)."""
    real = shutil.which("date")
    assert real, "a real date binary"
    fake = dirpath / "date"
    fake.write_text(f'#!/bin/sh\nTZ=UTC exec "{real}" "$@"\n')
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return dirpath


def test_without_a_tz_database_the_gate_fails_closed(tmp_path):
    # the gate must notice (zone abbreviation is not EST/EDT) instead of
    # computing a wrong window
    env = {"PATH": f"{_no_tz_date(tmp_path)}{os.pathsep}{os.environ['PATH']}"}
    assert call_fn("merge_window_open", utc("2026-11-07T15:00"), env) == 2
    assert call_fn("in_market_hours", utc("2026-11-06T15:00"), env) == 2


# ── both merge paths, against a fake gh ────────────────────────────────────
FAKE_GH = r'''#!/usr/bin/env python3
import json, os, re, sys
from urllib.parse import parse_qs, urlparse
args = sys.argv[1:]
with open(os.environ["FAKE_GH_LOG"], "a") as fh:
    fh.write(json.dumps(args) + "\n")
scen = json.load(open(os.environ["FAKE_GH_SCENARIO"]))
if args[:2] == ["pr", "merge"]:
    url = args[-1]
    rc = scen.get("merge_rc", {}).get(url, 0)
    print(("merged " if rc == 0 else "cannot merge ") + url)
    sys.exit(rc)
if args[:1] == ["api"]:
    method, path, i, rest = "GET", None, 0, args[1:]
    takes = {"-X", "--method", "-f", "-F", "--raw-field", "--field", "-H", "--header", "-q", "--jq", "--input"}
    while i < len(rest):
        a = rest[i]
        if a in ("-X", "--method"):
            method = rest[i + 1]; i += 2; continue
        if a in takes:
            i += 2; continue
        if not a.startswith("-") and path is None:
            path = a
        i += 1
    if method == "POST" and path.endswith("/labels"):
        sys.exit(scen.get("label_rc", 0))
    if method == "POST" and re.search(r"/issues/[0-9]+/comments$", path):
        sys.exit(scen.get("comment_rc", 0))
    if method == "GET" and re.search(r"/issues/[0-9]+/comments", path):
        num = path.split("/issues/")[1].split("/")[0]
        if num in scen.get("comments_rc", {}):
            sys.exit(scen["comments_rc"][num])
        print(json.dumps(scen.get("comments", {}).get(num, [])))
        sys.exit(0)
    if method == "GET" and "/pulls?" in path and scen.get("pulls_rc"):
        sys.exit(scen["pulls_rc"])
    if method == "GET" and "/pulls?" in path:
        page = int(parse_qs(urlparse(path).query).get("page", ["1"])[0])
        print(json.dumps(scen.get("pulls", []) if page == 1 else []))
        sys.exit(0)
    if method == "GET" and re.search(r"/pulls/[0-9]+$", path):
        num = path.rsplit("/", 1)[1]
        print(json.dumps({"number": int(num), "merged": scen.get("merged", {}).get(num, False)}))
        sys.exit(0)
    if method == "GET" and "/check-runs" in path:
        sha = path.split("/commits/")[1].split("/")[0]
        print(json.dumps(scen.get("checks", {}).get(sha, {"total_count": 0, "check_runs": []})))
        sys.exit(0)
print("fake gh: unexpected call " + " ".join(args), file=sys.stderr)
sys.exit(1)
'''

GREEN = {
    "changes": ("completed", "success"),
    "python-tests": ("completed", "skipped"),
    "node-build": ("completed", "success"),
    "docker-build": ("completed", "success"),
    "test": ("completed", "success"),
    "Auto-merge Claude PRs": ("completed", "success"),
}


def checks(**changes):
    """Check-runs JSON for one commit. Keyword names use _ for -; a value of
    None drops that check (it never ran)."""
    runs = dict(GREEN)
    for k, v in changes.items():
        runs[k.replace("_", "-")] = v
    rows = [{"name": n, "status": v[0], "conclusion": v[1], "app": {"slug": "github-actions"}}
            for n, v in runs.items() if v is not None]
    return {"total_count": len(rows), "check_runs": rows}


def pr(num, ref=None, draft=False, repo="o/r"):
    return {
        "number": num, "state": "open", "draft": draft,
        "html_url": f"https://github.com/o/r/pull/{num}",
        "head": {"ref": ref or f"claude/change-{num}", "sha": f"{num:040x}",
                 "repo": {"full_name": repo}},
    }


def sha(num):
    return f"{num:040x}"


def run(args, now, scenario, tmp_path, env=None):
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    gh = bindir / "gh"
    gh.write_text(FAKE_GH)
    gh.chmod(gh.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    (tmp_path / "scenario.json").write_text(json.dumps(scenario))
    log = tmp_path / "gh.log"
    full_env = {**os.environ, **(env or {})}
    full_env.update({
        "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}",
        "FAKE_GH_SCENARIO": str(tmp_path / "scenario.json"),
        "FAKE_GH_LOG": str(log),
        "AUTOMERGE_NOW_EPOCH": str(utc(now)),
        "GITHUB_REPOSITORY": "o/r",
    })
    full_env.pop("GITHUB_STEP_SUMMARY", None)  # never write into a real CI summary
    if "GITHUB_EVENT_PATH" not in (env or {}):
        full_env.pop("GITHUB_EVENT_PATH", None)  # CI's own event is not this test's PR
    r = subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True,
                       timeout=120, env=full_env)
    calls = [json.loads(l) for l in log.read_text().splitlines()] if log.exists() else []
    return r, calls


def merges(calls):
    return [c for c in calls if c[:2] == ["pr", "merge"]]


URL7 = "https://github.com/o/r/pull/7"
INSIDE = "2026-11-06T16:12"   # 11:12 ET Friday: when #1631 merged
AFTER = "2026-11-06T21:07"    # 16:07 ET Friday: the first winter sweep


def test_ci_path_defers_inside_the_window(tmp_path):
    r, calls = run([URL7], INSIDE, {}, tmp_path)
    assert r.returncode == 0, r.stderr
    assert merges(calls) == [], "nothing may merge during market hours"
    assert DEFERRED_TEXT in r.stdout
    assert "::notice title=DEPLOY WINDOW::" in r.stdout
    label_posts = [c for c in calls if c[0] == "api" and "repos/o/r/issues/7/labels" in c]
    assert label_posts, f"the deferral should label the PR: {calls}"
    assert any("labels[]=deploy-window-deferred" in a for a in label_posts[0])


def test_a_label_failure_never_turns_a_deferral_red(tmp_path):
    r, calls = run([URL7], INSIDE, {"label_rc": 1}, tmp_path)
    assert r.returncode == 0
    assert merges(calls) == []
    assert "::warning title=DEPLOY WINDOW::" in r.stdout


def test_ci_path_merges_outside_the_window(tmp_path):
    r, calls = run([URL7], AFTER, {}, tmp_path)
    assert r.returncode == 0, r.stderr
    assert merges(calls) == [["pr", "merge", "--squash", "--delete-branch", URL7]]


@pytest.mark.parametrize("now", ["2026-11-07T15:00", "2026-11-06T14:14"])  # Sat 10:00 ET; Fri 09:14 ET
def test_ci_path_merges_on_a_weekend_and_before_the_guard(tmp_path, now):
    r, calls = run([URL7], now, {}, tmp_path)
    assert r.returncode == 0, r.stderr
    assert merges(calls) == [["pr", "merge", "--squash", "--delete-branch", URL7]], now


def test_a_failed_merge_still_fails_the_job(tmp_path):
    r, _ = run([URL7], AFTER, {"merge_rc": {URL7: 1}}, tmp_path)
    assert r.returncode == 1, "a merge error must stay visible, as before"


def test_ci_path_trusts_the_api_when_gh_errs_after_merging(tmp_path):
    # gh can exit non-zero after the server merged (a later step failed, or
    # the response was lost); the job then reports the merge, not a failure.
    r, _ = run([URL7], AFTER, {"merge_rc": {URL7: 1}, "merged": {"7": True}}, tmp_path)
    assert r.returncode == 0
    assert "#7 is merged, but gh exited 1" in r.stdout


# ── oldest first on the CI path (2026-11-18) ───────────────────────────────
# The sweep's cron ran once a day, 3 h 51 min to 7 h 06 min after the close
# (all 4 runs, 2026-11-11..16), so a PR deferred in market hours was overtaken
# by PRs opened after the close and left in conflict (#1809, 2026-11-17). The
# CI path now merges an older deferred PR before its own.
BOT = "github-actions[bot]"
MARKER = "deploy-window-deferred head="


def record(num, head=None, login=BOT):
    """One PR comment carrying the deferral record for a head commit."""
    return {"user": {"login": login},
            "body": f"DEPLOY WINDOW: deferred.\n<!-- {MARKER}{head or sha(num)} -->"}


def event(tmp_path, num):
    """The pull_request event file GitHub hands the automerge job."""
    path = tmp_path / "event.json"
    path.write_text(json.dumps({"pull_request": {"number": num, "head": {"sha": sha(num)}}}))
    return {"GITHUB_EVENT_PATH": str(path)}


def comment_posts(calls, num):
    return [c for c in calls if c[0] == "api" and "POST" in c and f"repos/o/r/issues/{num}/comments" in c]


def queue_merge(num):
    return ["pr", "merge", "--squash", "--delete-branch", "--match-head-commit", sha(num),
            f"https://github.com/o/r/pull/{num}"]


OWN = ["pr", "merge", "--squash", "--delete-branch", URL7]


def test_a_deferral_records_the_head_commit_the_checks_passed_on(tmp_path):
    r, calls = run([URL7], INSIDE, {}, tmp_path, env=event(tmp_path, 7))
    assert r.returncode == 0, r.stderr
    assert merges(calls) == []
    posts = comment_posts(calls, 7)
    assert len(posts) == 1, calls
    body = next(a for a in posts[0] if a.startswith("body="))
    assert f"{MARKER}{sha(7)}" in body
    assert "::warning" not in r.stdout


def test_a_deferral_without_the_event_or_the_comment_still_defers(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    r, calls = run([URL7], INSIDE, {}, tmp_path / "a")            # no event file
    assert r.returncode == 0 and merges(calls) == [] and comment_posts(calls, 7) == []
    assert "could not write the deferral record" in r.stdout
    r, calls = run([URL7], INSIDE, {"comment_rc": 1}, tmp_path / "b", env=event(tmp_path / "b", 7))
    assert r.returncode == 0 and merges(calls) == []
    assert "could not write the deferral record" in r.stdout


def test_ci_path_merges_an_older_deferred_pr_first_and_not_its_own(tmp_path):
    scen = {"pulls": [pr(5), pr(7)], "comments": {"5": [record(5)]}}
    r, calls = run([URL7], AFTER, scen, tmp_path)
    assert r.returncode == 0, r.stderr
    assert merges(calls) == [queue_merge(5)], "the older deferred PR merges, and only it"
    assert "#7 was NOT merged" in r.stdout
    told = comment_posts(calls, 7)
    assert len(told) == 1 and any("#5" in a and "NOT merged" in a for a in told[0]), calls


def test_the_2026_11_17_orphan_is_merged_by_the_first_run_after_the_close(tmp_path):
    # #1809 went green at 16:57Z (11:57 ET) and was deferred. #1810's run at
    # 21:11Z merged #1810; the sweep had not run; #1809 was left in conflict.
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    url_1810 = "https://github.com/o/r/pull/1810"
    r, calls = run(["https://github.com/o/r/pull/1809"], "2026-11-17T16:57", {}, tmp_path / "a",
                   env=event(tmp_path / "a", 1809))
    assert r.returncode == 0 and merges(calls) == []
    body = next(a for a in comment_posts(calls, 1809)[0] if a.startswith("body="))[len("body="):]
    scen = {"pulls": [pr(1706), pr(1809), pr(1810)],
            "comments": {"1809": [{"user": {"login": BOT}, "body": body}]}}
    r, calls = run([url_1810], "2026-11-17T21:11", scen, tmp_path / "b")
    assert r.returncode == 0, r.stderr
    assert merges(calls) == [queue_merge(1809)]


@pytest.mark.parametrize("scen", [
    {"pulls": [pr(5), pr(7)]},                                                     # no record at all
    {"pulls": [pr(5), pr(7)], "comments": {"5": [record(5, head=sha(4))]}},        # a push since the record
    {"pulls": [pr(5), pr(7)], "comments": {"5": [record(5), record(5, head=sha(4))]}},  # newest record names another commit
    {"pulls": [pr(5), pr(7)], "comments": {"5": [record(5, login="someone")]}},    # not the workflow's bot
    {"pulls": [pr(7), pr(9)], "comments": {"9": [record(9)]}},                     # a NEWER PR never jumps ahead
    {"pulls": [pr(5, ref="feature/x"), pr(7)], "comments": {"5": [record(5)]}},    # not a claude/ branch
    {"pulls": [pr(5, draft=True), pr(7)], "comments": {"5": [record(5)]}},         # a draft
    {"pulls": [pr(5, repo="fork/r"), pr(7)], "comments": {"5": [record(5)]}},      # a fork
    {"pulls_rc": 1},                                                               # the queue cannot be listed
    {"pulls": [pr(5), pr(7)], "comments_rc": {"5": 1}},                            # its comments cannot be read
])
def test_ci_path_merges_its_own_pr_when_no_older_pr_holds_a_valid_record(tmp_path, scen):
    r, calls = run([URL7], AFTER, scen, tmp_path)
    assert r.returncode == 0, r.stderr
    assert merges(calls) == [OWN], scen
    assert comment_posts(calls, 7) == []


def test_ci_path_takes_the_oldest_of_several_deferred_prs_and_only_one(tmp_path):
    scen = {"pulls": [pr(5), pr(3), pr(7)], "comments": {"3": [record(3)], "5": [record(5)]}}
    r, calls = run([URL7], AFTER, scen, tmp_path)
    assert r.returncode == 0, r.stderr
    assert merges(calls) == [queue_merge(3)]


def test_ci_path_falls_back_to_its_own_pr_when_the_older_one_cannot_merge(tmp_path):
    # A deferred PR already in conflict with main: the merge is refused, and
    # this run's own PR merges as it did before the queue existed.
    scen = {"pulls": [pr(5), pr(7)], "comments": {"5": [record(5)]},
            "merge_rc": {"https://github.com/o/r/pull/5": 1}}
    r, calls = run([URL7], AFTER, scen, tmp_path)
    assert r.returncode == 0, r.stderr
    assert merges(calls) == [queue_merge(5), OWN]


def test_ci_path_counts_an_older_pr_gh_misreported_as_merged(tmp_path):
    scen = {"pulls": [pr(5), pr(7)], "comments": {"5": [record(5)]},
            "merge_rc": {"https://github.com/o/r/pull/5": 1}, "merged": {"5": True}}
    r, calls = run([URL7], AFTER, scen, tmp_path)
    assert r.returncode == 0, r.stderr
    assert merges(calls) == [queue_merge(5)], "two merges in one run would be two changes in one deploy"


def test_a_failed_comment_on_the_overtaken_pr_does_not_fail_the_run(tmp_path):
    scen = {"pulls": [pr(5), pr(7)], "comments": {"5": [record(5)]}, "comment_rc": 1}
    r, calls = run([URL7], AFTER, scen, tmp_path)
    assert r.returncode == 0
    assert merges(calls) == [queue_merge(5)]
    assert "could not comment on #7" in r.stdout


def test_the_queue_is_never_read_inside_the_window(tmp_path):
    scen = {"pulls": [pr(5), pr(7)], "comments": {"5": [record(5)]}}
    r, calls = run([URL7], INSIDE, scen, tmp_path)
    assert r.returncode == 0 and merges(calls) == []
    assert not [c for c in calls if any("/pulls?" in a for a in c)], calls


def test_the_automerge_job_can_write_the_record_it_relies_on():
    # The record and the notice are PR comments: the job needs pull-requests
    # write, which it has had since before this change (frozen YAML, read only).
    job = load(CI)["jobs"]["automerge"]
    assert job["permissions"].get("pull-requests") == "write"
    assert "GITHUB_EVENT_PATH" not in CI.read_text(), "the event file is GitHub's own; no workflow may point it elsewhere"


def test_ci_path_fails_closed_without_a_tz_database(tmp_path):
    (tmp_path / "bin").mkdir(exist_ok=True)
    _no_tz_date(tmp_path / "bin")  # run() puts tmp_path/bin first on PATH, beside the fake gh
    r, calls = run([URL7], "2026-11-07T15:00", {}, tmp_path)
    assert r.returncode == 0
    assert merges(calls) == []
    assert "::error title=DEPLOY WINDOW::" in r.stdout


def test_sweep_inside_the_window_makes_no_call_at_all(tmp_path):
    r, calls = run(["--sweep"], INSIDE, {"pulls": [pr(1)], "checks": {sha(1): checks()}}, tmp_path)
    assert r.returncode == 0
    assert calls == []
    assert "merges nothing" in r.stdout


def test_sweep_merges_only_the_oldest_green_claude_pr(tmp_path):
    scenario = {
        "pulls": [pr(12), pr(11), pr(10, ref="feature/not-claude"), pr(9, draft=True),
                  pr(8, repo="someone/fork")],
        "checks": {sha(n): checks() for n in (8, 9, 10, 11, 12)},
    }
    r, calls = run(["--sweep"], AFTER, scenario, tmp_path)
    assert r.returncode == 0, r.stderr
    assert merges(calls) == [["pr", "merge", "--squash", "--delete-branch", "--match-head-commit", sha(11),
                              "https://github.com/o/r/pull/11"]]
    checked = [c[1] for c in calls if c[0] == "api" and "/check-runs" in c[1]]
    assert checked == [f"repos/o/r/commits/{sha(11)}/check-runs?per_page=100&filter=latest"], \
        "drafts, forks and non-claude heads are not even considered; the run stops at its one merge"


def test_sweep_never_merges_a_failed_cancelled_pending_or_missing_check(tmp_path):
    bad = {
        1: checks(test=("completed", "failure")),
        2: checks(docker_build=("completed", "cancelled")),
        3: checks(node_build=("in_progress", None)),
        4: checks(python_tests=None),                       # never reported
        5: checks(changes=("completed", "skipped")),        # automerge's `if:` needs success
        6: checks(test=("completed", "timed_out")),
        7: checks(changes=("queued", None)),
    }
    scenario = {"pulls": [pr(n) for n in [*bad, 20]],
                "checks": {**{sha(n): c for n, c in bad.items()}, sha(20): checks()}}
    r, calls = run(["--sweep"], AFTER, scenario, tmp_path)
    assert r.returncode == 0, r.stderr
    assert merges(calls) == [["pr", "merge", "--squash", "--delete-branch", "--match-head-commit", sha(20),
                              "https://github.com/o/r/pull/20"]]
    for n, reason in [(1, "test: failure"), (2, "docker-build: cancelled"), (3, "node-build: in_progress"),
                      (4, "python-tests: missing"), (5, "changes: skipped"), (6, "test: timed_out"),
                      (7, "changes: queued")]:
        assert re.search(rf"skip #{n} .*{re.escape(reason)}", r.stdout), (n, r.stdout)


def test_sweep_accepts_skipped_and_neutral_heavy_jobs(tmp_path):
    c = checks(python_tests=("completed", "skipped"), node_build=("completed", "neutral"),
               docker_build=("completed", "skipped"), test=("completed", "skipped"))
    r, calls = run(["--sweep"], AFTER, {"pulls": [pr(5)], "checks": {sha(5): c}}, tmp_path)
    assert r.returncode == 0
    assert len(merges(calls)) == 1


def test_sweep_moves_on_when_a_merge_fails_but_still_merges_only_one(tmp_path):
    scenario = {"pulls": [pr(1), pr(2), pr(3)],
                "checks": {sha(n): checks() for n in (1, 2, 3)},
                "merge_rc": {"https://github.com/o/r/pull/1": 1}}
    r, calls = run(["--sweep"], AFTER, scenario, tmp_path)
    assert r.returncode == 0
    assert [c[-1] for c in merges(calls)] == ["https://github.com/o/r/pull/1",
                                             "https://github.com/o/r/pull/2"]
    assert "MERGED #2" in r.stdout


def test_sweep_stops_after_a_merge_gh_misreported(tmp_path):
    # The one-merge-per-run promise rests on the API's answer, not gh's exit
    # code: #1 merged although gh exited 1, so #2 must wait for the next run.
    scenario = {"pulls": [pr(1), pr(2)],
                "checks": {sha(n): checks() for n in (1, 2)},
                "merge_rc": {"https://github.com/o/r/pull/1": 1},
                "merged": {"1": True}}
    r, calls = run(["--sweep"], AFTER, scenario, tmp_path)
    assert r.returncode == 0
    assert [c[-1] for c in merges(calls)] == ["https://github.com/o/r/pull/1"]
    assert "MERGED #1" in r.stdout


def test_sweep_with_nothing_to_merge(tmp_path):
    r, calls = run(["--sweep"], AFTER, {"pulls": []}, tmp_path)
    assert r.returncode == 0
    assert merges(calls) == []
    assert "No PR merged" in r.stdout


def test_sweep_reads_every_page_of_open_prs(tmp_path):
    r, calls = run(["--sweep"], AFTER, {"pulls": [pr(1)], "checks": {sha(1): checks()}}, tmp_path)
    pages = [c[1] for c in calls if c[0] == "api" and "/pulls?" in c[1]]
    assert pages == ["repos/o/r/pulls?state=open&per_page=100&page=1",
                     "repos/o/r/pulls?state=open&per_page=100&page=2"]


def test_both_merge_paths_delete_the_merged_branch(tmp_path):
    # 2026-11-06, human: "turn on GitHub's auto-delete setting". The setting is a
    # human-only toggle, so each merge deletes its own head.
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    _, ci_calls = run([URL7], AFTER, {}, tmp_path / "a")
    _, sweep_calls = run(["--sweep"], AFTER, {"pulls": [pr(3)], "checks": {sha(3): checks()}}, tmp_path / "b")
    for call in merges(ci_calls) + merges(sweep_calls):
        assert "--delete-branch" in call, call
    assert len(merges(ci_calls)) == 1 and len(merges(sweep_calls)) == 1


def test_window_command_exit_codes(tmp_path):
    assert run(["--window"], AFTER, {}, tmp_path)[0].returncode == 0
    assert run(["--window"], INSIDE, {}, tmp_path)[0].returncode == 3


# ── the frozen YAML ────────────────────────────────────────────────────────
# Copied byte for byte from ci.yml as it stood on main at a017a2be (v1.0.1341),
# before this change. The approval scope was the merge STEP only: the checks
# that gate a merge (`needs:` + `if:`) must not move.
GATE_BEFORE = """\
    needs: [changes, python-tests, node-build, docker-build, test]
    if: >
      always() &&
      github.event_name == 'pull_request' &&
      startsWith(github.head_ref, 'claude/') &&
      needs.changes.result == 'success' &&
      needs.python-tests.result != 'failure' && needs.python-tests.result != 'cancelled' &&
      needs.node-build.result != 'failure' && needs.node-build.result != 'cancelled' &&
      needs.docker-build.result != 'failure' && needs.docker-build.result != 'cancelled' &&
      needs.test.result != 'failure' && needs.test.result != 'cancelled'
    runs-on: ubuntu-latest
    permissions:
      contents: write
      pull-requests: write
    steps:
"""


def load(path):
    data = yaml.safe_load(path.read_text())
    # YAML 1.1 reads the bare key `on` as boolean True.
    data["on"] = data.pop(True, data.get("on"))
    return data


def test_ci_yaml_parses_and_the_merge_gate_is_byte_identical():
    raw = CI.read_text()
    job = raw[raw.index("\n  automerge:\n"):]
    assert GATE_BEFORE in job, "automerge's needs:/if:/permissions must be byte-identical to before"
    ci = load(CI)
    am = ci["jobs"]["automerge"]
    assert am["needs"] == ["changes", "python-tests", "node-build", "docker-build", "test"]
    assert am["permissions"] == {"contents": "write", "pull-requests": "write"}
    assert set(ci["on"]) == {"push", "pull_request"} and ci["on"]["push"] == {"branches": ["main"]}


def test_the_merge_step_runs_the_gate_script():
    steps = load(CI)["jobs"]["automerge"]["steps"]
    assert steps[0]["uses"] == "actions/checkout@v4"
    assert steps[0]["with"]["sparse-checkout"] == "scripts/automerge.sh"
    assert steps[-1]["run"].strip() == 'bash scripts/automerge.sh "$PR_URL"'
    assert steps[-1]["env"] == {"PR_URL": "${{ github.event.pull_request.html_url }}",
                                "GH_TOKEN": "${{ secrets.GITHUB_TOKEN }}"}
    assert not any("gh pr merge" in (s.get("run") or "") for s in steps), \
        "the merge must go through scripts/automerge.sh, never straight to gh"


def test_the_sweep_requires_exactly_the_jobs_automerge_needs():
    out = subprocess.run(["bash", "-c", f'source "{SCRIPT}"; echo "$REQUIRED_CHECKS"'],
                         capture_output=True, text=True, timeout=30).stdout
    assert set(json.loads(out)) == set(load(CI)["jobs"]["automerge"]["needs"])


def cron_times(expr):
    """(hour, minute) pairs of a 'M H * * D' cron line (lists and ranges only)."""
    def expand(field):
        vals = []
        for part in field.split(","):
            lo, _, hi = part.partition("-")
            vals += list(range(int(lo), int(hi or lo) + 1))
        return vals
    minute, hour, dom, month, dow = expr.split()
    assert (dom, month, dow) == ("*", "*", "1-5"), "weekday runs only"
    return [(h, m) for h in expand(hour) for m in expand(minute)]


def test_sweep_workflow_runs_after_the_close_in_both_est_and_edt():
    wf = load(SWEEP)
    assert set(wf["on"]) == {"schedule", "workflow_dispatch"}, "never on push/pull_request"
    times = [t for entry in wf["on"]["schedule"] for t in cron_times(entry["cron"])]
    for day in ("2026-11-04", "2026-07-15"):  # a Wednesday in EST, one in EDT
        local = [datetime.fromisoformat(f"{day}T{h:02d}:{m:02d}").replace(tzinfo=timezone.utc)
                 .astimezone(NY) for h, m in times]
        close = datetime.fromisoformat(f"{day}T16:00").replace(tzinfo=NY)
        after = [t for t in local if t >= close]
        assert len(after) >= 2, f"{day}: at least two sweeps after the close, got {after}"
        early = [t for t in after if t < close + timedelta(hours=2)]
        assert len(early) >= 2, f"{day}: at least two sweeps within two hours of the close"


def test_sweep_workflow_permissions_and_entrypoint():
    wf = load(SWEEP)
    assert wf["permissions"] == {"contents": "write", "pull-requests": "write", "checks": "read"}
    assert wf["concurrency"] == {"group": "automerge-sweep", "cancel-in-progress": False}
    steps = wf["jobs"]["sweep"]["steps"]
    assert steps[-1]["run"].strip() == "bash scripts/automerge.sh --sweep"
    assert steps[-1]["env"] == {"GH_TOKEN": "${{ secrets.GITHUB_TOKEN }}"}


def test_no_workflow_sets_the_test_clock():
    for wf in (ROOT / ".github" / "workflows").glob("*.yml"):
        assert "AUTOMERGE_NOW_EPOCH" not in wf.read_text(), \
            f"{wf.name}: the test-only clock override would let a workflow merge at any hour"
