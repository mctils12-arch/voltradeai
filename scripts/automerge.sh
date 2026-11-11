#!/usr/bin/env bash
# scripts/automerge.sh — the DEPLOY WINDOW for Claude PR merges.
#
# THE RULE (human, 2026-11-05): nothing restarts the bot during market hours;
# ship after the 4pm ET close. Railway deploys every merge to main, and every
# deploy restarts the bot, whose shutdown handler cancels every open order
# (server/bot.ts gracefulShutdown). The automerge job in .github/workflows/ci.yml
# used to run `gh pr merge --squash` the moment CI went green, at any hour:
# #1631 merged 2026-11-05 16:12:46Z and the bot booted again at 16:14:52Z
# (11:14 ET); #1632 merged 20:17:57Z and booted at 20:20:01Z (15:20 ET). The gap
# was filed 2026-08-15 and 2026-09-14 (research/wishlist.md "PROCESS GAP FOUND");
# the human approved closing it, frozen-path edit included, on 2026-11-06
# ("fix all of this").
#
# THE WINDOW. Monday-Friday, America/New_York: no merge from 09:15 (the 09:30
# open minus PREOPEN_GUARD_MIN) until the 16:00 close. The guard exists because
# a merge restarts the bot minutes later, not at once: over the 100 boots of
# 2026-11-02..06, merge -> boot took 164 s median, 197 s p90 and 675 s at most,
# so a merge at 09:29 would restart the bot inside the session. Holidays and
# half days count as full market days, which is harmless: the merge waits for
# 16:00. Daylight saving comes from the tz database; if that zone cannot be
# read, the gate fails CLOSED (defers) rather than guess.
#
# USAGE
#   automerge.sh <PR_URL>  ci.yml's automerge job, after every required job has
#                          settled: merge now, or defer (exit 0, logged, labelled).
#   automerge.sh --sweep   .github/workflows/automerge-sweep.yml, after the close:
#                          merge the OLDEST open claude/ PR whose head commit
#                          passed every required check. ONE merge per run, so
#                          each deploy carries one change.
#   automerge.sh --window  print the decision for now; exit 0 open, 3 closed.
#
# BRANCHES. Both paths merge with --delete-branch (human, 2026-11-06: "turn on
# GitHub's auto-delete setting"; that repository setting is a human-only toggle,
# so the merge deletes its own head instead). gh deletes the branch after the
# merge; if only that step fails, the PR is still merged (pr_is_merged), and the
# weekly .github/workflows/branch-hygiene.yml run removes the leftover head.
#
# WHY A SCRIPT. `.github/workflows/` is a FROZEN PATH. As with
# scripts/gated_tests.sh, the YAML gains one `run:` line and the rule lives in a
# mutable, tested file (test_automerge_deploy_window.py).

set -uo pipefail

OPEN_MIN=$((9 * 60 + 30))   # 09:30 ET, NYSE open
CLOSE_MIN=$((16 * 60))      # 16:00 ET, NYSE close
PREOPEN_GUARD_MIN=15        # see THE WINDOW above (max merge -> boot seen: 675 s)

# The jobs ci.yml's automerge job `needs` (it is not one of them). The sweep
# holds a PR to the same bar the automerge job's `if:` does: `changes` must have
# succeeded, every heavy job must have succeeded or been skipped (neutral is
# accepted too); anything failed, cancelled, still running or missing blocks.
# test_automerge_deploy_window.py pins this list to the job's `needs`.
REQUIRED_CHECKS='["changes","python-tests","node-build","docker-build","test"]'

DEFERRED_LABEL="deploy-window-deferred"
DEFERRED_TEXT="merge deferred to after the 4pm ET close (DEPLOY WINDOW)"

# Seconds since the epoch. AUTOMERGE_NOW_EPOCH exists for the tests only; no
# workflow may set it (pinned by test_automerge_deploy_window.py).
now_epoch() { printf '%s\n' "${AUTOMERGE_NOW_EPOCH:-$(date +%s)}"; }

# "<ISO weekday 1-7> <minutes since midnight, New York time>" for an epoch.
# GNU date silently falls back to UTC for an unknown zone, so the zone
# abbreviation is checked: anything but EST/EDT is a hard error (return 2).
et_dow_min() {
  local dow hh mm zone
  read -r dow hh mm zone < <(TZ=America/New_York date -d "@$1" '+%u %H %M %Z' 2>/dev/null) || return 2
  case "$zone" in EST|EDT) ;; *) return 2 ;; esac
  printf '%s %s\n' "$dow" $((10#$hh * 60 + 10#$mm))
}

# 0 when the epoch falls in the NYSE regular session: weekdays 09:30 <= t < 16:00 ET.
in_market_hours() {
  local dow min
  read -r dow min < <(et_dow_min "$1") || return 2
  [ "$dow" -le 5 ] && [ "$min" -ge "$OPEN_MIN" ] && [ "$min" -lt "$CLOSE_MIN" ]
}

# 0 when a merge may happen at this epoch; 1 inside the window; 2 = cannot tell
# (callers treat every non-zero as "do not merge").
merge_window_open() {
  local dow min
  read -r dow min < <(et_dow_min "$1") || return 2
  if [ "$dow" -le 5 ] && [ "$min" -ge $((OPEN_MIN - PREOPEN_GUARD_MIN)) ] && [ "$min" -lt "$CLOSE_MIN" ]; then
    return 1
  fi
  return 0
}

et_stamp() { TZ=America/New_York date -d "@$1" '+%a %Y-%m-%d %H:%M %Z' 2>/dev/null || echo "epoch $1"; }

window_text() {
  printf 'weekdays %02d:%02d-16:00 ET' $(((OPEN_MIN - PREOPEN_GUARD_MIN) / 60)) $(((OPEN_MIN - PREOPEN_GUARD_MIN) % 60))
}

# Reads the check-runs JSON of one commit on stdin; prints "ok" or the reasons
# it is blocked (one line).
checks_verdict() {
  jq -r --argjson req "$REQUIRED_CHECKS" '
    def allowed($n): if $n == "changes" then ["success"] else ["success", "skipped", "neutral"] end;
    (.check_runs // []) as $runs
    | [ $req[] as $n
        | [ $runs[] | select(.name == $n) ] as $m
        | if ($m | length) == 0 then "\($n): missing"
          elif any($m[]; .status != "completed") then "\($n): \([$m[].status] | unique | join(","))"
          elif any($m[]; (.conclusion // "none") as $c | any(allowed($n)[]; . == $c) | not)
            then "\($n): \([$m[] | .conclusion // "none"] | unique | join(","))"
          else empty end ]
    | if length == 0 then "ok" else "blocked (" + join("; ") + ")" end'
}

# Reads open-PR JSON (the REST pulls list) on stdin; prints "number<TAB>head
# sha<TAB>url" for every merge candidate, oldest first: a claude/ head branch in
# this repository (never a fork), not a draft.
sweep_candidates() {
  jq -r --arg repo "$1" '
    [ .[] | select(.state == "open")
          | select((.head.ref // "") | startswith("claude/"))
          | select((.head.repo.full_name // "") == $repo)
          | select(.draft != true) ]
    | sort_by(.number) | .[] | [.number, .head.sha, .html_url] | @tsv'
}

# Every open PR, all pages, as one JSON array.
list_open_prs() {
  local repo="$1" page=1 out pages=()
  while [ "$page" -le 20 ]; do
    out=$(gh api "repos/$repo/pulls?state=open&per_page=100&page=$page") || return 1
    [ "$(printf '%s' "$out" | jq 'length')" = "0" ] && break
    pages+=("$out")
    page=$((page + 1))
  done
  if [ "${#pages[@]}" -eq 0 ]; then echo '[]'; return 0; fi
  printf '%s\n' "${pages[@]}" | jq -s 'add'
}

summary() { [ -n "${GITHUB_STEP_SUMMARY:-}" ] && printf '%s\n' "$@" >> "$GITHUB_STEP_SUMMARY"; return 0; }

# Sets PR_REPO ("owner/name") and PR_NUM from a pull-request URL.
parse_pr_url() {
  [[ "$1" =~ ^https://github\.com/([^/]+/[^/]+)/pull/([0-9]+)/?$ ]] || return 1
  PR_REPO="${BASH_REMATCH[1]}"
  PR_NUM="${BASH_REMATCH[2]}"
}

# 0 when the PR is merged, asked of the API rather than inferred from gh's exit
# code: gh can exit non-zero after the server has already merged (an error in a
# later step, or a lost response), and the sweep's one-merge-per-run promise
# must not rest on that exit code.
pr_is_merged() {
  local merged
  merged=$(gh api "repos/$1/pulls/$2" </dev/null | jq -r '.merged // false') || return 1
  [ "$merged" = "true" ]
}

# Best effort: a failure here never turns a correct deferral into a red job.
label_deferred() {
  parse_pr_url "$1" || return 1
  # Creating the label answers 422 once it exists; only attaching it matters.
  gh api -X POST "repos/$PR_REPO/labels" -f name="$DEFERRED_LABEL" -f color=fbca04 \
    -f description="$DEFERRED_TEXT" >/dev/null 2>&1 || true
  gh api -X POST "repos/$PR_REPO/issues/$PR_NUM/labels" -f "labels[]=$DEFERRED_LABEL" >/dev/null
}

merge_one_pr() {
  local url="$1" now stamp rc
  now=$(now_epoch)
  stamp=$(et_stamp "$now")
  merge_window_open "$now"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    [ "$rc" -eq 2 ] && echo "::error title=DEPLOY WINDOW::cannot read New York time (tz database) — deferring rather than guessing"
    echo "DEPLOY WINDOW: $stamp is inside the no-merge window ($(window_text)) — $DEFERRED_TEXT."
    echo "Nothing was merged. The after-close sweep (.github/workflows/automerge-sweep.yml) merges this PR"
    echo "once the window opens, if every required check is still green on its head commit."
    echo "::notice title=DEPLOY WINDOW::$DEFERRED_TEXT ($stamp)"
    summary "### DEPLOY WINDOW" "$stamp: $DEFERRED_TEXT. The after-close sweep merges it."
    if ! label_deferred "$url"; then
      echo "::warning title=DEPLOY WINDOW::could not add the '$DEFERRED_LABEL' label; the deferral stands"
    fi
    return 0
  fi
  echo "DEPLOY WINDOW open ($stamp): merging $url"
  gh pr merge --squash --delete-branch "$url"
  rc=$?
  [ "$rc" -eq 0 ] && return 0
  if parse_pr_url "$url" && pr_is_merged "$PR_REPO" "$PR_NUM"; then
    echo "::warning title=DEPLOY WINDOW::#$PR_NUM is merged, but gh exited $rc after the merge"
    return 0
  fi
  return "$rc"
}

sweep() {
  local now stamp rc repo prs cands n=0 num sha url checks verdict
  now=$(now_epoch)
  stamp=$(et_stamp "$now")
  merge_window_open "$now"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    [ "$rc" -eq 2 ] && echo "::error title=DEPLOY WINDOW::cannot read New York time (tz database) — merging nothing"
    echo "DEPLOY WINDOW: $stamp is inside the no-merge window ($(window_text)) — the sweep merges nothing this run."
    return 0
  fi
  repo="${GITHUB_REPOSITORY:-}"
  if [ -z "$repo" ]; then
    echo "::error::GITHUB_REPOSITORY is not set; cannot list pull requests" >&2
    return 2
  fi
  if ! prs=$(list_open_prs "$repo"); then
    echo "::error::could not list open pull requests of $repo" >&2
    return 2
  fi
  cands=$(printf '%s' "$prs" | sweep_candidates "$repo")
  echo "DEPLOY WINDOW open ($stamp). Candidates (open, non-draft claude/ PRs, oldest first): $(printf '%s' "$cands" | grep -c . || true)"
  while IFS=$'\t' read -r num sha url; do
    [ -n "$num" ] || continue
    n=$((n + 1))
    # gh's stdin is /dev/null inside this loop: the loop reads the candidates on
    # stdin, and a command that read it would silently eat the rest of the list.
    if ! checks=$(gh api "repos/$repo/commits/$sha/check-runs?per_page=100&filter=latest" </dev/null); then
      echo "skip #$num ($sha): could not read its check runs"
      continue
    fi
    verdict=$(printf '%s' "$checks" | checks_verdict)
    if [ "$verdict" != "ok" ]; then
      echo "skip #$num ($sha): $verdict"
      continue
    fi
    echo "merging #$num at $sha: every required check passed"
    gh pr merge --squash --delete-branch --match-head-commit "$sha" "$url" </dev/null
    rc=$?
    if [ "$rc" -eq 0 ] || pr_is_merged "$repo" "$num"; then
      [ "$rc" -ne 0 ] && echo "::warning title=DEPLOY WINDOW::#$num is merged, but gh exited $rc after the merge"
      echo "MERGED #$num. One merge per run; any other green PR waits for the next sweep."
      summary "### DEPLOY WINDOW sweep" "$stamp: merged #$num ($url) at $sha."
      return 0
    fi
    echo "merge of #$num failed (a conflict or a newer push); trying the next candidate"
  done <<< "$cands"
  echo "No PR merged this run ($n candidate(s) checked)."
  summary "### DEPLOY WINDOW sweep" "$stamp: no PR merged ($n candidate(s) checked)."
  return 0
}

window_cmd() {
  local now rc
  now=$(now_epoch)
  merge_window_open "$now"
  rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "OPEN: $(et_stamp "$now") is outside the no-merge window ($(window_text))"
    return 0
  fi
  if [ "$rc" -eq 2 ]; then
    echo "CLOSED: cannot read New York time (tz database), so no merge is allowed"
    return 3
  fi
  echo "CLOSED: $(et_stamp "$now") is inside the no-merge window ($(window_text))"
  return 3
}

main() {
  case "${1:-}" in
    --sweep) sweep ;;
    --window) window_cmd ;;
    "" | -*)
      echo "usage: $0 <PR_URL> | --sweep | --window" >&2
      return 2
      ;;
    *) merge_one_pr "$1" ;;
  esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
  exit $?
fi
