#!/usr/bin/env bash
# scripts/branch_cleanup_apply.sh — apply the human-approved branch cleanup
# (2026-10-28: "Delete the merged and superseded branches and the data-backup
# branch. Tag the unique ones as archive/<name>, then delete them. Keep
# celestial-catalog-data. Skip any branch with an open PR.")
# Agent sessions cannot push refs other than their own branch, so a human runs
# this from any clone with normal push access. Dry-run by default.
#
# Plan: ops/branch_cleanup_plan_2026-10-28.tsv (name, tip_sha, tip_date, class, action)
#   action=delete / delete-notag -> delete the branch
#   action=tag+delete            -> push tag archive/<name> at tip_sha, verify, then delete
#   action=skip                  -> untouched (main, celestial-catalog-data, open PRs,
#                                   tips < 24h old, branches named in code)
# Safety: a branch whose CURRENT remote tip differs from the planned tip_sha is
# skipped (someone pushed after the plan was made) and reported.
set -euo pipefail
PLAN="${PLAN:-ops/branch_cleanup_plan_2026-10-28.tsv}"; APPLY=0; BATCH=50
[ "${1:-}" = "--apply" ] && APPLY=1
git fetch -q origin --prune '+refs/heads/*:refs/remotes/origin/*'
DEL=$(mktemp); TAG=$(mktemp); MOVED=0
tail -n +2 "$PLAN" | while IFS=$'\t' read -r name sha _date _class action; do
  case "$action" in delete|delete-notag|tag+delete) ;; *) continue ;; esac
  cur=$(git rev-parse -q --verify "refs/remotes/origin/$name" || true)
  [ -z "$cur" ] && continue                      # already gone
  if [ "$cur" != "$sha" ]; then echo "MOVED (skipped): $name" >&2; continue; fi
  [ "$action" = "tag+delete" ] && echo "$sha refs/tags/archive/$name" >> "$TAG"
  echo "$name" >> "$DEL"
done
echo "tags to push: $(wc -l < "$TAG"); branches to delete: $(wc -l < "$DEL")"
if [ "$APPLY" -ne 1 ]; then echo "dry run — re-run with --apply"; exit 0; fi
# 1) archive tags first, verified on the remote before any delete
awk '{print $1":"$2}' "$TAG" | xargs -r -n "$BATCH" git push -q origin
while read -r sha ref; do
  git ls-remote --exit-code origin "$ref" >/dev/null || { echo "TAG MISSING $ref — aborting before deletes" >&2; exit 1; }
done < "$TAG"
# 2) deletions
sed 's|^|:refs/heads/|' "$DEL" | xargs -r -n "$BATCH" git push -q origin
echo "done. remaining remote branches: $(git ls-remote --heads origin | wc -l)"
