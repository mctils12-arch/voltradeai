#!/usr/bin/env bash
# gated_tests.sh — the test gate (MASTER PROGRAM T1.2 / T1.3).
#
# T1.1 got all 368 test files RUNNING in CI, deliberately non-blocking, to
# establish a baseline (research/test_baseline.md). This is the second half:
# every test file is now REQUIRED — its failure turns CI red — except the ones
# explicitly listed in ci/quarantine.txt with a reason and a review date.
#
# WHY A SCRIPT AND NOT YAML. `.github/workflows/` is a FROZEN PATH; the MASTER
# PROGRAM §9 names Track 1 as the specific authorization to touch it. Keeping
# the rule out here means the frozen file gains one `run:` line and every future
# adjustment happens in a mutable, tested file (server/gatedTests.test.ts).
#
# THREE GATES:
#   1. REQUIRED SUITES — server, client, python, minus quarantine. Any failure
#      exits 1. All three run even if an earlier one fails, so one red suite
#      never hides the other two.
#   2. QUARANTINE MAY NOT GROW (T1.3) — the count is pinned in
#      ci/quarantine_max.txt. Adding an entry fails the build unless a human
#      raises the pin. Quarantining the test you just broke is precisely how a
#      gate rots into decoration.
#   3. NO ENTRY MAY AGE PAST 30 DAYS (T1.3) — past its review date the build
#      fails. A quarantine is a promise to come back; this makes it one.
#
# Quarantined tests still RUN, at the end, non-blocking, so the day one starts
# passing is visible rather than silently overdue.
#
# Exit 0 = clean. Exit 1 = a required test failed, or a quarantine rule broke.
# Exit 2 = the gate could not run at all — treated as failure, because a gate
# that silently stops gating is the drift this whole program exists to prevent.

set -uo pipefail
cd "$(dirname "$0")/.."

QUARANTINE_FILE="${QUARANTINE_FILE:-ci/quarantine.txt}"
MAX_FILE="${QUARANTINE_MAX_FILE:-ci/quarantine_max.txt}"
MAX_AGE_DAYS="${QUARANTINE_MAX_AGE_DAYS:-30}"

if [ ! -f "$QUARANTINE_FILE" ]; then
  echo "FAIL: '$QUARANTINE_FILE' not found — cannot tell required from quarantined." >&2
  exit 2
fi

# Entries are the first whitespace-delimited token; `#` lines and blanks skipped.
quarantined() {
  awk '!/^[[:space:]]*(#|$)/ { print $1 }' "$QUARANTINE_FILE"
}

# ── Gate 2: the list may only shrink ────────────────────────────────────────
q_count=$(quarantined | wc -l | tr -d ' ')
q_max=$(awk '!/^[[:space:]]*(#|$)/ { print $1; exit }' "$MAX_FILE" 2>/dev/null || echo "")
if ! [[ "$q_max" =~ ^[0-9]+$ ]]; then
  echo "FAIL: '$MAX_FILE' has no numeric pin — refusing to run an unpinned gate." >&2
  exit 2
fi
if [ "$q_count" -gt "$q_max" ]; then
  echo "FAIL: quarantine grew ${q_max} -> ${q_count}." >&2
  echo "A newly-failing test belongs FIXED, not parked. If the addition is" >&2
  echo "genuinely unresolvable, a human raises the pin in $MAX_FILE and says why." >&2
  quarantined | sed 's/^/  /' >&2
  exit 1
fi

# ── Gate 3: no entry may age out ────────────────────────────────────────────
stale=$(python3 - "$QUARANTINE_FILE" "$MAX_AGE_DAYS" <<'PY'
import datetime, re, sys
path, max_age = sys.argv[1], int(sys.argv[2])
today, bad = datetime.date.today(), []
for line in open(path):
    if not line.strip() or line.lstrip().startswith('#'):
        continue
    entry = line.split()[0]
    m = re.search(r'review by (\d{4}-\d{2}-\d{2})', line)
    if not m:
        bad.append(f"{entry}: no 'review by YYYY-MM-DD' — every quarantine needs an expiry")
        continue
    due = datetime.date.fromisoformat(m.group(1))
    if due < today:
        bad.append(f"{entry}: review date {due} passed {(today - due).days} day(s) ago")
    elif (due - today).days > max_age:
        bad.append(f"{entry}: review date {due} is more than {max_age} days out")
print("\n".join(bad))
PY
)
if [ -n "$stale" ]; then
  echo "FAIL: quarantine entries are overdue or unbounded:" >&2
  printf '%s\n' "$stale" | sed 's/^/  /' >&2
  echo >&2
  echo "A quarantine is a promise to come back. Fix the test, or resolve the" >&2
  echo "underlying gap — do NOT simply push the date out." >&2
  exit 1
fi

# ── Gate 1: run the required suites ─────────────────────────────────────────
is_quarantined() { quarantined | grep -qxF "$1"; }

collect() { # $1 = git pathspec
  local f
  while IFS= read -r f; do
    is_quarantined "$f" || printf '%s\n' "$f"
  done < <(git ls-files "$1")
}

server_files=$(collect 'server/*.test.ts')
client_files=$(collect 'client/**/*.test.ts')

echo "Quarantined (excluded from the gate, run non-blocking at the end): $q_count / pin $q_max"
quarantined | sed 's/^/  - /'
echo

fail=0
run_suite() { # $1 = label, rest = command
  local label="$1"; shift
  echo "── $label ─────────────────────────────────────────────"
  if "$@"; then
    echo "OK: $label"
  else
    echo "FAIL: $label" >&2
    fail=1
  fi
  echo
}

# All three run unconditionally: one red suite must never hide the other two.
# shellcheck disable=SC2086
run_suite "server ($(printf '%s\n' "$server_files" | wc -l | tr -d ' ') files)" \
  npx tsx --test $server_files
# shellcheck disable=SC2086
run_suite "client ($(printf '%s\n' "$client_files" | wc -l | tr -d ' ') files)" \
  npx tsx --test $client_files

# Python has no per-file exclusion need today (nothing python is quarantined);
# --deselect would be added here if that changes.
py_ignores=""
while IFS= read -r f; do
  case "$f" in *.py) py_ignores="$py_ignores --ignore=$f" ;; esac
done < <(quarantined)
# shellcheck disable=SC2086
VOLTRADE_CI=1 run_suite "python" python3 -m pytest -q $py_ignores

# ── Quarantined tests: run, report, never block ─────────────────────────────
if [ "$q_count" -gt 0 ]; then
  echo "── quarantined (non-blocking) ─────────────────────────"
  while IFS= read -r f; do
    case "$f" in
      *.test.ts) npx tsx --test "$f" >/dev/null 2>&1 \
        && echo "  NOW PASSING — remove from $QUARANTINE_FILE: $f" \
        || echo "  still failing (expected): $f" ;;
      *.py) python3 -m pytest -q "$f" >/dev/null 2>&1 \
        && echo "  NOW PASSING — remove from $QUARANTINE_FILE: $f" \
        || echo "  still failing (expected): $f" ;;
    esac
  done < <(quarantined)
  echo
fi

if [ "$fail" -ne 0 ]; then
  echo "GATE FAILED: a required test suite is red." >&2
  echo "Do NOT add the failing file to $QUARANTINE_FILE to get green — gate 2" >&2
  echo "rejects a grown quarantine for exactly that reason." >&2
  exit 1
fi

echo "GATE PASSED: all required suites green; quarantine $q_count/$q_max, none overdue."
