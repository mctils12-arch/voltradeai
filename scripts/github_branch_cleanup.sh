#!/usr/bin/env bash
# scripts/github_branch_cleanup.sh — prune the remote branches that are just
# sitting on GitHub (human directive 2026-10-21: "delete the ones that aren't
# being utilized"). Every Claude session pushes its own claude/<name> branch
# and the auto-merge workflow squash-merges the PR but never removes the
# branch, so the repo reached 865 branches; a branch list that long slows
# every GitHub-integration picker (Railway's source settings among them).
#
# Policy (dry-run by default; --apply pushes the deletions):
#   keep  main, every branch whose last commit is younger than --days (7),
#         every branch named in --keep-file (open-PR heads, data/result
#         branches such as gv-res-* and data-backup), and the current branch;
#   prune everything else. A pruned branch that had a PR stays recoverable
#         on GitHub as refs/pull/<n>/head; a pruned branch that never had a
#         PR is gone for good — hence the age floor.
# The standing fix for the pile-up is GitHub's own setting: repository
# Settings → General → Pull Requests → "Automatically delete head branches".
set -euo pipefail
DAYS=7; APPLY=0; KEEP_FILE=""; BATCH=100
while [ $# -gt 0 ]; do
  case "$1" in
    --apply) APPLY=1 ;;
    --days) DAYS="$2"; shift ;;
    --keep-file) KEEP_FILE="$2"; shift ;;
    *) echo "usage: $0 [--apply] [--days N] [--keep-file FILE]" >&2; exit 2 ;;
  esac; shift
done
git fetch -q origin --prune '+refs/heads/*:refs/remotes/origin/*'
NOW=$(date +%s); CUR=$(git branch --show-current || true)
LIST=$(mktemp)
git for-each-ref refs/remotes/origin --format='%(refname:short) %(committerdate:unix)' | grep -v '/HEAD$' | while read -r ref ts; do
  b=${ref#origin/}
  [ "$b" = "main" ] && continue
  [ -n "$CUR" ] && [ "$b" = "$CUR" ] && continue
  if [ -n "$KEEP_FILE" ] && grep -qxF "$b" "$KEEP_FILE"; then continue; fi
  age=$(( (NOW - ts) / 86400 ))
  [ "$age" -lt "$DAYS" ] && continue
  echo "$b"
done > "$LIST"
N=$(wc -l < "$LIST")
echo "branches older than $DAYS days and not kept: $N"
if [ "$APPLY" != "1" ]; then echo "dry run — pass --apply to prune them (list: $LIST)"; exit 0; fi
split -l "$BATCH" "$LIST" "$LIST.part_"
for part in "$LIST".part_*; do
  # shellcheck disable=SC2046
  git push -q origin --delete $(cat "$part") && echo "pruned $(wc -l < "$part")"
done
echo "remaining remote branches: $(git ls-remote --heads origin | wc -l)"
