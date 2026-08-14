#!/usr/bin/env bash
# tsc_ratchet.sh — the typecheck gate (MASTER PROGRAM T1.6 / Q5).
#
# Replaces `npx tsc --noEmit || true` in ci.yml's node-build job. That `|| true`
# carried the comment "tighten to hard-fail once existing TS errors are
# cleared" — so the output printed on every CI run for months into a log nobody
# opened, and two live user-facing bugs sat in it (research/tsc_baseline.md §1).
#
# THE LOGIC LIVES HERE, NOT IN ci.yml, ON PURPOSE. `.github/workflows/` is a
# FROZEN PATH; the MASTER PROGRAM is the specific authorization for Track 1 to
# touch it. Keeping the rule in a MUTABLE, testable script means the frozen
# file changes by exactly one line, once, and every future adjustment to the
# gate happens out here under test (server/tscRatchet.test.ts).
#
# TWO GATES, deliberately different in kind:
#   1. TOTAL must not exceed the TOTAL pin in ci/tsc_baseline.txt. Non-
#      increasing (D4: ratchets are downward-only; raising a pin is stop
#      condition 2 and needs a human).
#   2. TS2304 must be ZERO, always, regardless of the pin. "Cannot find name"
#      is the one code that is never config noise and never a false positive —
#      the name is in scope or it is not. Both bugs found by T0.0 were TS2304.
#
# Exit 0 = clean. Exit 1 = regression. Exit 2 = the check could not run, which
# is treated as failure: a ratchet that silently stops ratcheting is the exact
# drift this program exists to prevent.

set -uo pipefail
cd "$(dirname "$0")/.."

BASELINE_FILE="${TSC_BASELINE_FILE:-ci/tsc_baseline.txt}"

if [ ! -f "$BASELINE_FILE" ]; then
  echo "FAIL: baseline file '$BASELINE_FILE' not found — cannot verify the ratchet." >&2
  exit 2
fi

# `TOTAL <n>` plus `<code> <count>` lines; comments and blanks ignored.
pinned_total=$(awk '$1=="TOTAL" {print $2; exit}' "$BASELINE_FILE")
if ! [[ "$pinned_total" =~ ^[0-9]+$ ]]; then
  echo "FAIL: '$BASELINE_FILE' has no valid 'TOTAL <n>' line." >&2
  exit 2
fi

echo "Typechecking (pin: $pinned_total from $BASELINE_FILE) ..."
tsc_out=$(npx tsc --noEmit 2>&1)
tsc_status=$?

# Distinguish "tsc ran and found errors" (normal — it exits non-zero whenever
# any exist) from "tsc could not run at all". Without this, a broken install
# would produce zero matches and sail through as a PASS.
if [ "$tsc_status" -ne 0 ] && ! printf '%s\n' "$tsc_out" | grep -q 'error TS'; then
  echo "FAIL: tsc did not run (exit $tsc_status). Output:" >&2
  printf '%s\n' "$tsc_out" | head -40 >&2
  exit 2
fi

actual_total=$(printf '%s\n' "$tsc_out" | grep -c 'error TS')
actual_2304=$(printf '%s\n' "$tsc_out" | grep -c 'error TS2304')

echo "tsc reported $actual_total error(s); TS2304: $actual_2304"

fail=0

# ── Gate 2 first: TS2304 is always a real bug, so report it above the count. ──
if [ "$actual_2304" -gt 0 ]; then
  fail=1
  echo >&2
  echo "FAIL: $actual_2304 TS2304 'Cannot find name' error(s) — always real bugs." >&2
  echo "An identifier that does not resolve is a ReferenceError at runtime, and" >&2
  echo "inside a try block it kills every statement after it, not just its own" >&2
  echo "line. Fix the scope; never silence this with a placeholder declaration." >&2
  printf '%s\n' "$tsc_out" | grep 'error TS2304' | sed 's/^/  /' >&2
fi

# ── Gate 1: the non-increasing total. ──
if [ "$actual_total" -gt "$pinned_total" ]; then
  fail=1
  echo >&2
  echo "FAIL: typecheck errors rose $pinned_total -> $actual_total." >&2
  echo >&2
  echo "Per-code diff vs $BASELINE_FILE (this tells you WHICH case you are in):" >&2
  # A regression shows up in the codes your diff touches. A wholesale shift
  # across unrelated codes is an environment divergence (a different tsc or
  # @types resolution), which is NOT fixed by raising the pin.
  diff <(awk '$1 ~ /^TS[0-9]+$/ {print $1, $2}' "$BASELINE_FILE" | sort) \
       <(printf '%s\n' "$tsc_out" | grep -oE 'error TS[0-9]+' | sed 's/error //' \
         | sort | uniq -c | awk '{print $2, $1}' | sort) \
    | sed 's/^/  /' >&2 || true
  echo >&2
  echo "  '<' = pinned baseline    '>' = this run" >&2
  echo >&2
  echo "If these are your errors, fix them. If the shift is wholesale and" >&2
  echo "unrelated to your diff, it is an environment divergence — investigate," >&2
  echo "do NOT raise the pin. Raising a pin is MASTER PROGRAM stop condition 2" >&2
  echo "and needs a human." >&2
fi

if [ "$fail" -ne 0 ]; then
  echo >&2
  echo "Full typecheck output:" >&2
  printf '%s\n' "$tsc_out" | grep 'error TS' | sed 's/^/  /' >&2
  exit 1
fi

if [ "$actual_total" -lt "$pinned_total" ]; then
  echo
  echo "The count DROPPED $pinned_total -> $actual_total. Lower the TOTAL in"
  echo "$BASELINE_FILE to $actual_total in this same PR so the gain is locked in —"
  echo "an unlowered pin lets the next change quietly give it back."
fi

echo "OK: $actual_total <= $pinned_total, TS2304 = 0"
