#!/usr/bin/env bash
# scripts/branch_cleanup_apply.sh — apply the human-approved branch cleanup
# (2026-10-28: "Delete the merged and superseded branches and the data-backup
# branch. Tag the unique ones as archive/<name>, then delete them. Keep
# celestial-catalog-data. Skip any branch with an open PR."; asked again
# 2026-11-06: "Run the branch cleanup"). Dry-run by default.
#
# WHO RUNS IT. Agent sessions cannot push refs other than their own branch, so
# .github/workflows/branch-hygiene.yml runs it every week with GITHUB_TOKEN, on
# a plan scripts/branch_hygiene.py builds from live state seconds earlier. A
# human can still run it from any clone with push access (bash, awk, git only).
#
# Plan: PLAN=<file>, default ops/branch_cleanup_plan_2026-10-28.tsv
# (name, tip_sha, tip_date, class, action):
#   action=delete / delete-notag -> delete the branch
#   action=tag+delete            -> push tag archive/<name> at tip_sha, read it
#                                   back from the remote, then delete the branch
#   action=skip                  -> untouched (main, celestial-catalog-data, open
#                                   PRs, tips < 24h old, branches named in code)
# Safety:
#   * the remote is read fresh (git ls-remote): a branch whose tip is no longer
#     the planned tip_sha is skipped and reported (someone pushed after the plan);
#   * every delete is leased on the planned tip (--force-with-lease), so a push
#     landing between that read and the delete still wins;
#   * a unique branch is deleted only once its archive tag reads back from the
#     remote at the planned sha. If archive/<name> already exists at another sha
#     (a reused name), the tag is archive/<name>-<first 12 of the sha>;
#   * a rejected batch is retried one ref at a time, so one refused ref cannot
#     hold back the other 49.
# Exit 0 = done, or dry run. Exit 1 = some archive tag did not verify or some
# planned delete did not land (all listed). Exit 2 = could not run at all.
set -uo pipefail
PLAN="${PLAN:-ops/branch_cleanup_plan_2026-10-28.tsv}"
APPLY=0
BATCH=50
case "${1:-}" in
  --apply) APPLY=1 ;;
  "") ;;
  *) echo "usage: [PLAN=file] $0 [--apply]" >&2; exit 2 ;;
esac
[ -f "$PLAN" ] || { echo "FAIL: plan '$PLAN' not found" >&2; exit 2; }

WORK=$(mktemp -d "${TMPDIR:-/tmp}/branch-cleanup.XXXXXX") || exit 2
trap 'rm -rf -- "${WORK:?}"' EXIT

read_heads() { git ls-remote --heads origin > "$1"; }
read_tags() { git ls-remote --tags origin 'refs/tags/archive/*' > "$1"; }
read_heads "$WORK/heads" || { echo "FAIL: cannot list origin's branches" >&2; exit 2; }
read_tags "$WORK/tags" || { echo "FAIL: cannot list origin's archive tags" >&2; exit 2; }

# One pass over the plan against the fresh remote state. Writes, one row per line:
#   del    name<TAB>sha          delete, leased on sha (merged, or archived already)
#   tagq   name<TAB>sha<TAB>tag  push this archive tag first; delete once it verifies
#   moved  name<TAB>planned<TAB>current
#   gone   name                  already deleted
: > "$WORK/del"; : > "$WORK/tagq"; : > "$WORK/moved"; : > "$WORK/gone"
awk -F'\t' -v W="$WORK" '
  FILENAME == W "/heads" { cur[substr($2, 12)] = $1; next }
  FILENAME == W "/tags"  { if ($2 !~ /\^\{\}$/) tag[substr($2, 11)] = $1; next }
  FNR == 1 { next }
  $5 != "delete" && $5 != "delete-notag" && $5 != "tag+delete" { next }
  !($1 in cur) { print $1 > (W "/gone"); next }
  cur[$1] != $2 { print $1 "\t" $2 "\t" cur[$1] > (W "/moved"); next }
  $5 != "tag+delete" { print $1 "\t" $2 > (W "/del"); next }
  {
    t = "archive/" $1
    if ((t in tag) && tag[t] != $2) t = t "-" substr($2, 1, 12)
    if ((t in tag) && tag[t] == $2) { print $1 "\t" $2 > (W "/del"); next }
    print $1 "\t" $2 "\t" t > (W "/tagq")
  }' "$WORK/heads" "$WORK/tags" "$PLAN"

lines() { wc -l < "$1" | tr -d ' '; }
n_del=$(lines "$WORK/del"); n_tag=$(lines "$WORK/tagq")
echo "plan: $PLAN"
echo "archive tags to push: $n_tag; branches to delete: $((n_del + n_tag)) ($n_tag after their tag verifies);" \
  "moved since the plan (kept): $(lines "$WORK/moved"); already gone: $(lines "$WORK/gone")"
awk -F'\t' '{ print "  MOVED (kept): " $1 " planned " substr($2, 1, 12) ", now " substr($3, 1, 12) }' "$WORK/moved"

if [ "$APPLY" -ne 1 ]; then
  awk -F'\t' '{ print "  would tag " $3 " at " substr($2, 1, 12) ", then delete " $1 }' "$WORK/tagq"
  awk -F'\t' '{ print "  would delete " $1 " (" substr($2, 1, 12) ")" }' "$WORK/del"
  echo "dry run — re-run with --apply"
  exit 0
fi

# Objects for the tags: a clone that never fetched a branch cannot push a tag at
# its tip. Best effort; a tag that cannot be pushed only keeps its branch.
if [ -s "$WORK/tagq" ]; then
  git fetch -q origin '+refs/heads/*:refs/remotes/origin/*' \
    || echo "WARN: fetch failed; tags for tips missing locally will not verify, and their branches stay" >&2
fi

# push_lines: one `git push` of every argument on stdin (TAB- or newline-separated).
# An array, not xargs: xargs may split a long list between a lease and its
# refspec, and it mangles quotes, which ref names may contain.
push_lines() {
  local args=() a
  while IFS= read -r a; do
    [ -n "$a" ] && args+=("$a")
  done < <(tr '\t' '\n')
  [ "${#args[@]}" -gt 0 ] || return 0
  git push -q origin "${args[@]}"
}

# push_batched FILE: FILE holds one argument group per line (a tag refspec, or a
# lease plus its delete). Pushes BATCH lines at a time; a refused batch is
# retried line by line, so one refused ref cannot hold back the rest.
push_batched() {
  local file="$1" part line
  split -l "$BATCH" "$file" "$WORK/chunk."
  for part in "$WORK"/chunk.*; do
    [ -f "$part" ] || continue
    if ! push_lines < "$part"; then
      while IFS= read -r line; do
        printf '%s\n' "$line" | push_lines || echo "  refused: $line" >&2
      done < "$part"
    fi
    rm -f -- "$part"
  done
}

# 1) archive tags first; a branch is deleted only after its tag reads back.
: > "$WORK/tagfail"
if [ -s "$WORK/tagq" ]; then
  awk -F'\t' '{ print $2 ":refs/tags/" $3 }' "$WORK/tagq" > "$WORK/tagpush"
  push_batched "$WORK/tagpush"
  if read_tags "$WORK/tags2"; then
    awk -F'\t' -v W="$WORK" '
      FILENAME == W "/tags2" { if ($2 !~ /\^\{\}$/) tag[substr($2, 11)] = $1; next }
      (($3 in tag) && tag[$3] == $2) { print $1 "\t" $2 >> (W "/del"); print "  tagged " $3 " at " substr($2, 1, 12); next }
      { print $1 "\t" $2 "\t" $3 > (W "/tagfail") }' "$WORK/tags2" "$WORK/tagq"
  else
    echo "FAIL: cannot re-read archive tags, so no unique branch is deleted" >&2
    cp "$WORK/tagq" "$WORK/tagfail"
  fi
fi

# 2) deletions, each leased on its planned tip.
if [ -s "$WORK/del" ]; then
  awk -F'\t' '{ print "--force-with-lease=refs/heads/" $1 ":" $2 "\t:refs/heads/" $1 }' "$WORK/del" > "$WORK/delpush"
  push_batched "$WORK/delpush"
fi

# 3) verify against the remote, then report. A branch still at its planned tip
# is a delete that did not land (a failure); one at another tip moved while this
# ran and the lease kept it (not a failure).
: > "$WORK/notdeleted"; : > "$WORK/deleted"; : > "$WORK/racemoved"
if read_heads "$WORK/heads2"; then
  awk -F'\t' -v W="$WORK" '
    FILENAME == W "/heads2" { cur[substr($2, 12)] = $1; next }
    !($1 in cur) { print $1 > (W "/deleted"); print "  deleted " $1; next }
    cur[$1] == $2 { print $1 > (W "/notdeleted"); next }
    { print $1 > (W "/racemoved"); print "  MOVED during this run (kept by the lease): " $1 }' "$WORK/heads2" "$WORK/del"
  remaining=$(lines "$WORK/heads2")
else
  echo "FAIL: cannot re-read origin's branches to verify the deletes" >&2
  cut -f1 "$WORK/del" > "$WORK/notdeleted"
  remaining="?"
fi
awk -F'\t' '{ print "  KEPT, archive tag did not verify: " $1 " (" $3 ")" }' "$WORK/tagfail"
awk '{ print "  NOT DELETED (still at the planned tip): " $1 }' "$WORK/notdeleted"
if [ -s "$WORK/tagfail" ]; then
  # GitHub may refuse a GITHUB_TOKEN tag at a commit that changes workflow
  # files. A human's credentials can push it; nothing was lost either way.
  echo "REMEDY: from a full clone with push access, run" \
    "python3 scripts/branch_hygiene.py --out plan.tsv && PLAN=plan.tsv bash scripts/branch_cleanup_apply.sh --apply" >&2
fi

n_tagged=$((n_tag - $(lines "$WORK/tagfail")))
msg="done: tagged $n_tagged, deleted $(lines "$WORK/deleted"); tag not verified $(lines "$WORK/tagfail"); delete not landed $(lines "$WORK/notdeleted"); moved $(( $(lines "$WORK/moved") + $(lines "$WORK/racemoved") )); remote branches now $remaining"
echo "$msg"
[ -n "${GITHUB_STEP_SUMMARY:-}" ] && printf '### Branch cleanup\n%s\n' "$msg" >> "$GITHUB_STEP_SUMMARY"
if [ -s "$WORK/tagfail" ] || [ -s "$WORK/notdeleted" ]; then
  exit 1
fi
exit 0
