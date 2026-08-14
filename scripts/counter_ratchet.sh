#!/usr/bin/env bash
# counter_ratchet.sh — enforce the §4.2 counters (MASTER PROGRAM T1.7).
#
# scripts/program_status.sh has measured 22 counters since #822, but only when
# a human typed the command. Measured is not enforced, and the gap is not
# theoretical: `commented_empty_catch` drifted 112 -> 113 during the 2026-08-14
# session from a concurrent session's merge, and nothing noticed until someone
# happened to run the script. This closes that.
#
# Reads ci/counter_baseline.txt (`<counter> <value> <direction>`) and fails the
# build when any counter moves the wrong way. Same shape as the two gates that
# came before it — the rule lives out here, mutable and tested
# (server/counterRatchet.test.ts), so the FROZEN workflow gains one `run:` line.
#
# tsc_errors / tsc_2304 are deliberately NOT here: scripts/tsc_ratchet.sh
# already gates them in node-build against ci/tsc_baseline.txt, and two pins for
# one number is how they drift apart.
#
# Exit 0 = clean. Exit 1 = a counter regressed. Exit 2 = the check could not
# run, treated as failure — a ratchet that silently stops ratcheting is the
# drift this whole program exists to prevent.

set -uo pipefail
cd "$(dirname "$0")/.."

BASELINE="${COUNTER_BASELINE_FILE:-ci/counter_baseline.txt}"

if [ ! -f "$BASELINE" ]; then
  echo "FAIL: '$BASELINE' not found — refusing to run an unpinned ratchet." >&2
  exit 2
fi

# --no-tsc: the typecheck is already gated by scripts/tsc_ratchet.sh in
# node-build. Running it again here would double the cost for no extra safety.
json=$(bash scripts/program_status.sh --json --no-tsc 2>/dev/null)
if [ -z "$json" ] || ! printf '%s' "$json" | grep -q '"detectors_registered"'; then
  echo "FAIL: program_status.sh produced no usable JSON — cannot verify counters." >&2
  printf '%s\n' "$json" | head -20 >&2
  exit 2
fi

printf '%s' "$json" > /tmp/vt_counters.json

python3 - "$BASELINE" /tmp/vt_counters.json <<'PY'
import json, sys

baseline_path, json_path = sys.argv[1], sys.argv[2]
cur = json.load(open(json_path))

regressions, checked, improved = [], 0, []
for line in open(baseline_path):
    line = line.split('#')[0].strip()
    if not line:
        continue
    parts = line.split()
    if len(parts) != 3:
        print(f"FAIL: malformed baseline line: {line!r}", file=sys.stderr)
        sys.exit(2)
    name, direction = parts[0], parts[2]
    try:
        pinned = int(parts[1])
    except ValueError:
        # Exit 2, not an uncaught traceback: "the pin is unreadable" is a
        # cannot-check, and a cannot-check must never read as checked-and-fine.
        print(f"FAIL: '{name}' has a non-numeric pin: {parts[1]!r}", file=sys.stderr)
        sys.exit(2)
    if direction not in ('non-increasing', 'non-decreasing'):
        print(f"FAIL: '{name}' has no valid direction: {direction!r}", file=sys.stderr)
        sys.exit(2)
    if name not in cur:
        # A counter named in the pin file but absent from the JSON means the
        # two have drifted apart — fail rather than silently skip it, which is
        # how a ratchet quietly stops covering half its surface.
        print(f"FAIL: '{name}' is pinned but program_status.sh no longer reports it.",
              file=sys.stderr)
        sys.exit(2)
    actual = cur[name]
    if not isinstance(actual, int):
        print(f"FAIL: '{name}' is not numeric in the JSON: {actual!r}", file=sys.stderr)
        sys.exit(2)
    checked += 1
    if direction == 'non-increasing' and actual > pinned:
        regressions.append((name, pinned, actual, direction))
    elif direction == 'non-decreasing' and actual < pinned:
        regressions.append((name, pinned, actual, direction))
    elif (direction == 'non-increasing' and actual < pinned) or \
         (direction == 'non-decreasing' and actual > pinned):
        improved.append((name, pinned, actual))

if regressions:
    print("FAIL: counters moved the wrong way.\n", file=sys.stderr)
    for name, pinned, actual, direction in regressions:
        print(f"  {name}: {pinned} -> {actual}  ({direction})", file=sys.stderr)
    print("""
Fix the code, not the pin. These counters exist because prose drifts and
numbers do not; loosening one to make a build green is MASTER PROGRAM stop
condition 2 and §12 calls it reducing counts by suppression.

If the move is genuinely correct — a refactor that consolidates assertions, a
counter whose definition changed — say so explicitly in the PR body and state
the before/after on identical inputs, per MEASUREMENT INTEGRITY.""",
          file=sys.stderr)
    sys.exit(1)

if improved:
    print("Counters IMPROVED — lower the pins in the same PR so the gain is locked in:")
    for name, pinned, actual in improved:
        print(f"  {name}: {pinned} -> {actual}   (edit ci/counter_baseline.txt)")
    print()

print(f"OK: {checked} counters at or better than baseline.")
PY
status=$?
rm -f /tmp/vt_counters.json
exit $status
