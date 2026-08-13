#!/usr/bin/env bash
# program_status.sh — the MASTER PROGRAM's §4.2 counter set.
#
# WHY THIS EXISTS (MASTER PROGRAM §0.6, §4.2): "progress that cannot be
# imagined." Prose drifts — this repo has already shipped docs claiming work
# that was never built (A1's RunPod bake, A4's "filed above, NOT built"). These
# numbers are measured from the tree on every run, so a claim can be checked
# instead of believed. §4.2 is explicit: "These numbers are the definition of
# progress. Prose contradicting them is wrong."
#
# USAGE
#   scripts/program_status.sh            # human-readable table
#   scripts/program_status.sh --json     # machine-readable, for CI ratchets
#   scripts/program_status.sh --no-tsc   # skip the slow typecheck (~40s)
#
# EXIT CODE is always 0 — this script MEASURES, it does not gate. Ratchets
# that gate the build live in the test suites (Track 1), reading this output.
#
# Every counter prints `name value baseline direction`, where direction is the
# only way the number is allowed to move. A counter moving the wrong way is a
# regression whether or not anyone noticed.

set -uo pipefail
cd "$(dirname "$0")/.."

JSON=0
RUN_TSC=1
for arg in "$@"; do
  case "$arg" in
    --json) JSON=1 ;;
    --no-tsc) RUN_TSC=0 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
  esac
done

# Source trees we own. node_modules/dist/build are never counted.
TS_SRC=(client/src server shared)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

# Count matching lines, tolerating "no matches" (grep exits 1) without tripping
# the pipeline. Every counter goes through this so a zero is a real zero.
count() { grep -rn "$@" 2>/dev/null | wc -l | tr -d ' '; }

py_files() { git ls-files '*.py' 2>/dev/null | grep -v '^node_modules/'; }
ts_files() { git ls-files '*.ts' '*.tsx' 2>/dev/null | grep -v '^node_modules/'; }

# ---------------------------------------------------------------------------
# 1. gated_tests — how many test FILES a CI run actually invokes.
#
# The gap this measures is the program's spine (Track 1): the repo has ~370
# test files and ci.yml names four of them. Tests that never run protect
# nothing, so "we have tests" is not the same claim as "a break turns CI red".
# ---------------------------------------------------------------------------
py_tests=$(py_files | grep -cE '(^|/)test_|_test\.py$' || true)
client_tests=$(ts_files | grep -c '^client/.*\.test\.tsx\?$' || true)
server_tests=$(ts_files | grep -c '^server/.*\.test\.ts$' || true)
tests_total=$(( py_tests + client_tests + server_tests ))

# Parse the python-tests job's pytest invocation out of ci.yml rather than
# hardcoding a list — if someone adds a suite to CI, this counter must notice
# on its own. We take every *.py token on the continuation lines of the
# `python -m pytest` command.
gated_py=0
if [ -f .github/workflows/ci.yml ]; then
  gated_py=$(awk '
    /python -m pytest/ { inblock=1 }
    inblock {
      n = gsub(/[A-Za-z0-9_\/]+\.py/, "&")
      total += n
      if ($0 !~ /\\$/) inblock = 0
    }
    END { print total + 0 }
  ' .github/workflows/ci.yml)
fi
# node-build runs `npm ci`, `tsc --noEmit || true`, `npm run build`. No test
# invocation — verified by the absence of any test script in that job.
gated_node=0
if [ -f .github/workflows/ci.yml ]; then
  gated_node=$(grep -cE 'run:.*(npm (run )?test|tsx --test|vitest|jest)' .github/workflows/ci.yml || true)
fi
gated_tests=$(( gated_py + gated_node ))

# ---------------------------------------------------------------------------
# 2. tsc_errors — the typecheck baseline.
#
# ci.yml runs `npx tsc --noEmit || true`. The `|| true` is why F-A (`altScale`
# undefined in a live readout) survived: the error was printed on every CI run
# for months and nothing read it. This counter is what replaces `|| true` with
# a ratchet (T1.6).
# ---------------------------------------------------------------------------
tsc_errors="skipped"
tsc_2304="skipped"
if [ "$RUN_TSC" = 1 ] && [ -d node_modules/typescript ]; then
  tsc_out=$(npx tsc --noEmit 2>&1 || true)
  tsc_errors=$(printf '%s\n' "$tsc_out" | grep -c 'error TS' || true)
  # TS2304 = "Cannot find name" — an identifier that does not exist at
  # runtime. This is the F-A defect class and it is separated out because it
  # is the only subset that is ALWAYS a real bug, never config noise.
  tsc_2304=$(printf '%s\n' "$tsc_out" | grep -c 'error TS2304' || true)
fi

# ---------------------------------------------------------------------------
# 3. silent_py_handlers — `except:` whose entire body is pass/continue.
#
# A swallowed exception is how a broken pipeline keeps reporting success, which
# CLAUDE.md rates worse than an outage ("a broken pipeline generates poisoned
# learning data"). Counted with -A1 so only genuinely empty handlers match; a
# handler that logs is doing its job and is not counted.
# ---------------------------------------------------------------------------
py_except_total=$(py_files | xargs grep -hnE '^\s*except' 2>/dev/null | wc -l | tr -d ' ')
silent_py=$(py_files | xargs grep -hn -A1 -E '^\s*except' 2>/dev/null \
  | grep -cE '^[0-9]+-\s*(pass|continue)\s*$' || true)
bare_except=$(py_files | xargs grep -hcE '^\s*except\s*:' 2>/dev/null \
  | awk '{s+=$1} END {print s+0}')

# ---------------------------------------------------------------------------
# 4. empty_ts_catch / ts_any — the TypeScript equivalents.
#
# `} catch {}` at datamap.tsx:4365 and 4367 is the second half of why F-A was
# invisible: the type error was ignored by `|| true`, and the ReferenceError it
# predicted was then swallowed at runtime. Both counters are non-increasing.
# ---------------------------------------------------------------------------
empty_ts_catch=$(ts_files | xargs grep -hoE 'catch\s*(\([^)]*\))?\s*\{\s*\}' 2>/dev/null | wc -l | tr -d ' ')
ts_any=$(ts_files | xargs grep -hoE ':\s*any\b' 2>/dev/null | wc -l | tr -d ' ')

# ---------------------------------------------------------------------------
# 5. layers_full_schema — F-E, the mechanism behind "new ideas land half-built".
#
# layersRegistry.test.ts enforces 8 fields and every layer carries exactly
# those 8. renderKind/time/provenance/altitudeRef are on a handful; `lod` is on
# ONE. The schema is exactly as complete as the test forces and not one field
# further — so this counter tracks the five UNENFORCED fields, and Track 4
# makes them required.
# ---------------------------------------------------------------------------
read -r layers_total layers_full layers_lod <<<"$(python3 - <<'PY'
import json, sys
try:
    d = json.load(open('datacore/layers.json'))
except Exception:
    print(0, 0, 0); sys.exit()
ls = d['layers'] if isinstance(d, dict) and 'layers' in d else d
want = ('renderKind', 'time', 'provenance', 'altitudeRef', 'lod')
full = sum(1 for l in ls if all(k in l for k in want))
lod  = sum(1 for l in ls if 'lod' in l)
print(len(ls), full, lod)
PY
)"

# ---------------------------------------------------------------------------
# 6. law_iv_scanned_files — F-B, the exemption in the Law IV audit.
#
# test_audit_critical.py selects layer modules by `basename.endswith("Layer.ts")`.
# That predicate catches five small modules and misses the two heaviest render
# surfaces in the repo: celestialSky.ts (a second WebGL2 context) and
# spaceFrame.ts (4,305 LOC of CPU 2D rasterisation). The rule is real and in
# CI; the hole is the predicate. Track 2 widens it — this counter is how you
# see that happen.
# ---------------------------------------------------------------------------
law_iv_scanned=$(ts_files | grep -c 'Layer\.ts$' || true)
# What a context-acquiring predicate WOULD select — the honest denominator.
law_iv_ctx=$(ts_files | grep -E '^client/src/(lib|render)/' \
  | xargs grep -ln 'getContext(' 2>/dev/null | wc -l | tr -d ' ')

# ---------------------------------------------------------------------------
# 7. order_post_sites — F-F, no execution choke point.
#
# Files (not lines) referencing /v2/orders outside tests. The account is paper
# so this is not urgent, but options_manager.py:46 already records a live bug
# from exactly this scatter: "FIX: was hardcoded — broke paper/live switching".
# Target is 1. Counting FILES, not occurrences, because the choke point is a
# module boundary.
# ---------------------------------------------------------------------------
order_post_sites=$( { py_files; ts_files; } | grep -viE '(test|spec)' \
  | xargs grep -ln '/v2/orders' 2>/dev/null | wc -l | tr -d ' ')

# ---------------------------------------------------------------------------
# 8. design_token_drift — T8.1 / D12.
#
# DESIGN.md carries a canonical token table and client/src/index.css defines
# the tokens. They agree today (D12, verified 2026-08-13). This counter keeps
# them agreeing: index.css is the source of truth, DESIGN.md documents it, and
# a mismatch in either direction is drift. Must stay 0.
# ---------------------------------------------------------------------------
design_token_drift=$(python3 - <<'PY'
import re
def css_tokens(path):
    out = {}
    try:
        src = open(path).read()
    except OSError:
        return out
    # Only the :root block — later scoped overrides are legitimately different.
    m = re.search(r':root\s*\{(.*?)\}', src, re.S)
    if not m:
        return out
    for name, val in re.findall(r'(--[a-z0-9-]+)\s*:\s*([^;]+);', m.group(1)):
        out[name] = re.sub(r'\s+', ' ', val).strip()
    return out

def md_tokens(path):
    out = {}
    try:
        src = open(path).read()
    except OSError:
        return out
    for name, val in re.findall(r'^\|\s*`(--[a-z0-9-]+)`\s*\|\s*`([^`]+)`\s*\|', src, re.M):
        out[name] = re.sub(r'\s+', ' ', val).strip()
    return out

css = css_tokens('client/src/index.css')
md  = md_tokens('DESIGN.md')
drift = 0
for name, val in md.items():
    if name not in css:
        drift += 1          # documented but undefined
    elif css[name] != val:
        drift += 1          # documented with the wrong value
print(drift)
PY
)

# ---------------------------------------------------------------------------
# 9. harness_rules_checked — F-H, how much of DESIGN.md is machine-checked.
#
# DESIGN.md opens "This is not advisory" and then relies on an agent reviewing
# its own screenshots — a human step incompatible with continuous operation, so
# it is where UI drift lands first. Counting distinct failure assertions in the
# visual harness: each Track 8 conversion adds one and this number rises.
# ---------------------------------------------------------------------------
harness_rules_checked=0
if [ -f scripts/visual_check.mjs ]; then
  harness_rules_checked=$(grep -cE '(checks\.)?failures\.push\(' scripts/visual_check.mjs || true)
fi

# ---------------------------------------------------------------------------
# 10. detectors_registered — the §0.7 DETECT duty.
#
# Ratchets only guard what someone already thought to count; they could never
# have found altScale. So every session adds one sweep for a defect class not
# yet counted. MUST INCREASE EACH SESSION — a session that adds no detector has
# not discharged the duty. Read from PROGRAM_STATE.md's DETECTORS table.
# ---------------------------------------------------------------------------
detectors_registered=0
if [ -f research/PROGRAM_STATE.md ]; then
  detectors_registered=$(awk '/^## DETECTORS/{f=1;next} /^## /{f=0} f && /^\| *D[0-9]+ /' \
    research/PROGRAM_STATE.md | wc -l | tr -d ' ')
fi

# ---------------------------------------------------------------------------
# 11. quarantine — Track 1's shrinking pool of known-failing tests.
#
# Does not exist yet (T1.2 creates it). Reported as 0/n-a so the counter is
# live from day one rather than appearing later and hiding its own history.
# ---------------------------------------------------------------------------
quarantine_size=0
quarantine_oldest_days=0
if [ -f ci/quarantine.txt ]; then
  quarantine_size=$(grep -cvE '^\s*(#|$)' ci/quarantine.txt || true)
  quarantine_oldest_days=$(python3 - <<'PY'
import re, datetime, sys
oldest = 0
today = datetime.date.today()
try:
    for line in open('ci/quarantine.txt'):
        if line.strip().startswith('#') or not line.strip():
            continue
        m = re.search(r'(\d{4}-\d{2}-\d{2})', line)
        if m:
            d = datetime.date.fromisoformat(m.group(1))
            oldest = max(oldest, (today - d).days)
except OSError:
    pass
print(oldest)
PY
)
fi

# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------
if [ "$JSON" = 1 ]; then
  cat <<EOF
{
  "gated_tests": $gated_tests,
  "tests_total": $tests_total,
  "tsc_errors": "$tsc_errors",
  "tsc_2304": "$tsc_2304",
  "silent_py_handlers": $silent_py,
  "py_except_total": $py_except_total,
  "bare_except": $bare_except,
  "empty_ts_catch": $empty_ts_catch,
  "ts_any": $ts_any,
  "layers_full_schema": $layers_full,
  "layers_total": $layers_total,
  "layers_with_lod": $layers_lod,
  "law_iv_scanned_files": $law_iv_scanned,
  "law_iv_context_files": $law_iv_ctx,
  "order_post_sites": $order_post_sites,
  "design_token_drift": $design_token_drift,
  "harness_rules_checked": $harness_rules_checked,
  "detectors_registered": $detectors_registered,
  "quarantine_size": $quarantine_size,
  "quarantine_oldest_days": $quarantine_oldest_days
}
EOF
  exit 0
fi

printf '%-24s %-14s %-12s %s\n' COUNTER VALUE BASELINE DIRECTION
printf '%-24s %-14s %-12s %s\n' ------- ----- -------- ---------
printf '%-24s %-14s %-12s %s\n' gated_tests        "$gated_tests/$tests_total" "4/364"  "must increase (>216)"
printf '%-24s %-14s %-12s %s\n' tsc_errors         "$tsc_errors"          "83"     "must decrease"
printf '%-24s %-14s %-12s %s\n' "  of which TS2304" "$tsc_2304"           "5"      "must reach 0 (always real bugs)"
printf '%-24s %-14s %-12s %s\n' silent_py_handlers "$silent_py/$py_except_total" "255/873" "non-increasing"
printf '%-24s %-14s %-12s %s\n' bare_except        "$bare_except"         "3"      "non-increasing"
printf '%-24s %-14s %-12s %s\n' empty_ts_catch     "$empty_ts_catch"      "495"    "non-increasing"
printf '%-24s %-14s %-12s %s\n' ts_any             "$ts_any"              "1252"   "non-increasing"
printf '%-24s %-14s %-12s %s\n' layers_full_schema "$layers_full/$layers_total" "1/238" "non-decreasing"
printf '%-24s %-14s %-12s %s\n' "  layers with lod" "$layers_lod"         "1"      "non-decreasing"
printf '%-24s %-14s %-12s %s\n' law_iv_scanned     "$law_iv_scanned"      "5"      "must reach $law_iv_ctx (ctx-acquiring)"
printf '%-24s %-14s %-12s %s\n' order_post_sites   "$order_post_sites"    "6"      "must reach 1"
printf '%-24s %-14s %-12s %s\n' design_token_drift "$design_token_drift"  "0"      "must stay 0"
printf '%-24s %-14s %-12s %s\n' harness_rules      "$harness_rules_checked" "71"   "non-decreasing"
printf '%-24s %-14s %-12s %s\n' detectors          "$detectors_registered" "0"     "MUST increase each session"
printf '%-24s %-14s %-12s %s\n' quarantine_size    "$quarantine_size"     "0"      "non-increasing"
printf '%-24s %-14s %-12s %s\n' quarantine_oldest  "${quarantine_oldest_days}d" "0d" "fail if >30"

if [ "$tsc_errors" = "skipped" ]; then
  echo
  echo "NOTE: tsc skipped (no node_modules/typescript, or --no-tsc). Run 'npm ci' first."
fi
