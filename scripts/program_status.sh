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
# 1. tests_run_in_ci / tests_gating_merge — two numbers, because they differ.
#
# COUNTER REDEFINED 2026-08-14 (T1.1/Q10). The old `gated_tests` counted test
# files NAMED in ci.yml, which worked while CI listed four files by hand and
# broke the moment T1.1 added a job running whole globs: it reported 6/368 for
# a run that actually executes all 364.
#
# The fix is two counters, not one, because "CI runs it" and "its failure can
# block a merge" are genuinely different facts and T1.1 deliberately created a
# gap between them. Collapsing them either way would lie: report 364 and the
# non-blocking baseline job masquerades as a gate; report 4 and ~3,586 tests
# now running on every PR are invisible. The MASTER PROGRAM's ">216" target is
# about GATING, so tests_gating_merge is the one that must climb.
#
# A job is treated as gating when it is NOT `continue-on-error: true`. Globs are
# resolved against the tree so the count is files, not tokens.
# ---------------------------------------------------------------------------
py_tests=$(py_files | grep -cE '(^|/)test_|_test\.py$' || true)
client_tests=$(ts_files | grep -c '^client/.*\.test\.tsx\?$' || true)
server_tests=$(ts_files | grep -c '^server/.*\.test\.ts$' || true)
tests_total=$(( py_tests + client_tests + server_tests ))

read -r tests_run_in_ci tests_gating_merge <<<"$(python3 - <<'PYEOF'
import re, subprocess, sys
try:
    import yaml
except ImportError:
    print(0, 0); sys.exit()
try:
    ci = yaml.safe_load(open('.github/workflows/ci.yml'))
except Exception:
    print(0, 0); sys.exit()

tracked = subprocess.run(['git', 'ls-files'], capture_output=True, text=True).stdout.split()
py_t  = [f for f in tracked if f.endswith('.py') and re.search(r'(^|/)test_|_test\.py$', f)]
srv_t = [f for f in tracked if re.match(r'server/.*\.test\.ts$', f)]
cli_t = [f for f in tracked if re.match(r'client/.*\.test\.tsx?$', f)]

def files_for(cmd):
    """Which test files a `run:` block actually executes."""
    hit = set()
    # scripts/gated_tests.sh (T1.2) runs EVERY test file except the entries in
    # ci/quarantine.txt. Without this branch the counter reports 0 for the very
    # job that gates the build — the same failure mode as the old `gated_tests`,
    # which counted files NAMED in ci.yml and broke the moment a job ran globs.
    if 'gated_tests.sh' in cmd:
        try:
            q = {l.split()[0] for l in open('ci/quarantine.txt')
                 if l.strip() and not l.lstrip().startswith('#')}
        except OSError:
            q = set()
        hit |= (set(py_t) | set(srv_t) | set(cli_t)) - q
    # `pip install pytest` is not a test run — match an invocation only.
    if re.search(r'python\s+-m\s+pytest|(^|[;&|]\s*)pytest\b', cmd):
        named = re.findall(r'[\w/]+\.py', cmd)
        hit |= set(named) if named else set(py_t)      # bare `pytest` = whole tree
    if 'tsx --test' in cmd or 'node --test' in cmd:
        if 'server/*.test.ts' in cmd:
            hit |= set(srv_t)
        if 'client/**/*.test.ts' in cmd:
            hit |= set(cli_t)
    return hit

run, gating = set(), set()
for job in (ci.get('jobs') or {}).values():
    job_soft = job.get('continue-on-error') is True
    for step in job.get('steps') or []:
        cmd = step.get('run') or ''
        if not cmd:
            continue
        f = files_for(cmd)
        if not f:
            continue
        run |= f
        # A step gates only if neither it nor its job tolerates failure.
        if not job_soft and step.get('continue-on-error') is not True:
            gating |= f
print(len(run), len(gating))
PYEOF
)"
tests_run_in_ci=${tests_run_in_ci:-0}
tests_gating_merge=${tests_gating_merge:-0}

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
#
# RULER CHANGE 2026-08-14 (Q13, MEASUREMENT INTEGRITY). Both counters ran
# `grep` over raw file text, so PROSE ABOUT the pattern counted as the pattern.
# Fifteen matches were documentation, not code: 495 -> 494 and 1251 -> 1237.
# That is the sixth time this session a source-scraping check has been defeated
# by a comment describing what it looks for (L9/L11/L12/L15 and twice in #833).
#
# TWO DESIGN CONSTRAINTS, both learned by getting them wrong first:
#
# 1. EXCLUDE BY LOCATION, never by re-matching cleaned text. Blanking comment
#    text and re-running the regex sends `empty_ts_catch` UP, 495 -> 516:
#    `catch { // why\n}` blanks to `catch {      \n}`, which now matches. Those
#    21 are exactly what D4 `commented_empty_catch` counts, and the two are
#    deliberately DISJOINT. A "cleanup" of this counter would have silently
#    merged them. Matching raw text and discarding matches whose bytes are
#    non-code can only ever REMOVE, so the counters cannot cross.
#
# 2. SCAN PER LINE. The first attempt lexed whole files; the regex literal
#    `/'/g` at server/billing.ts:83 opened a string that never closed, and the
#    next 30 lines were declared non-code — silently excluding four REAL
#    `catch (err: any)` annotations. Per-line scanning bounds a mis-parse to
#    the line that causes it.
#
# Every ambiguity resolves toward COUNTING (backticks untouched, unterminated
# quotes untouched, block-comment interiors only when the line starts with
# `*`). These are non-increasing counters: under-exclusion keeps them honestly
# high, over-exclusion makes the tree look better than it is, and per
# MEASUREMENT INTEGRITY a ruler change in the flattering direction is suspect
# by default. The 15 excluded sites are named individually in the PR.
# ---------------------------------------------------------------------------
read -r empty_ts_catch ts_any <<<"$(python3 - <<'PYEOF'
import os, re, sys
sys.path.insert(0, os.path.join(os.getcwd(), 'scripts'))
from ts_code_only import count_in_tree

print(count_in_tree(re.compile(r'catch\s*(\([^)]*\))?\s*\{\s*\}')),
      count_in_tree(re.compile(r':\s*any\b')))
PYEOF
)"

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
# 9b. D2 — long_try_empty_catch: a `try` spanning >50 lines whose `catch` body
# is empty (a bare comment counts as empty).
#
# This is the multiplier that turned F-A from a cosmetic bug into a functional
# one. `altScale` threw at datamap.tsx:4384, and because the enclosing `try`
# ran 135 lines, the throw did not break one readout — it skipped EVERY
# remaining statement in the block, including GND SPD, VERT SPD and the whole
# follow-aircraft recenter. A short try/catch is a decision; a 135-line one is
# a blast radius.
#
# Deliberately narrow: 3 hits repo-wide, and one of them is the altScale block
# itself. A detector that fires on hundreds of sites gets ignored, so this one
# is tuned to name only the blocks where a single bad line can silently take
# out a whole subsystem. Non-increasing.
# ---------------------------------------------------------------------------
long_try_empty_catch=$(python3 - <<'PY'
import re, subprocess
files = subprocess.run(['git', 'ls-files', '*.ts', '*.tsx'],
                       capture_output=True, text=True).stdout.split()
hits = 0
for f in files:
    if '/node_modules/' in f:
        continue
    try:
        src = open(f).read()
    except OSError:
        continue
    for m in re.finditer(r'\btry\s*\{', src):
        i = m.end() - 1
        depth = 0
        for j in range(i, len(src)):
            if src[j] == '{':
                depth += 1
            elif src[j] == '}':
                depth -= 1
                if depth == 0:
                    break
        else:
            continue                      # unbalanced — skip rather than guess
        if src.count('\n', m.start(), j) <= 50:
            continue
        if re.match(r'\s*catch\s*(\([^)]*\))?\s*\{\s*(/\*.*?\*/|//[^\n]*)?\s*\}',
                    src[j + 1:j + 200], re.S):
            hits += 1
print(hits)
PY
)

# ---------------------------------------------------------------------------
# 9c. D3 — boundary_any: `: any` appearing in a function's PARAMETER LIST or
# RETURN ANNOTATION, as opposed to anywhere inside a body.
#
# Q2 is the argument for this counter. `execAsync(cmd, opts?: any)` was ONE
# `any`, on one parameter, of one wrapper function — and it defeated overload
# resolution on promisify(exec) so thoroughly that 42 downstream `stdout.trim()`
# calls became type errors. That was 51% of the entire tsc baseline, and its
# noise is why nobody ever read the list that had two live bugs in it.
#
# `ts_any` counts all 1250 occurrences and is far too coarse to act on. An
# `any` inside a function body is a local shortcut; an `any` on a boundary is a
# contract the compiler cannot check, and it propagates to every caller. 233
# across 94 files is a list a session can actually work down. Non-increasing.
# ---------------------------------------------------------------------------
boundary_any=$(python3 - <<'PYEOF'
import re, subprocess
files = [f for f in subprocess.run(['git', 'ls-files', '*.ts', '*.tsx'],
                                   capture_output=True, text=True).stdout.split()
         if '/node_modules/' not in f]
# Two shapes cover essentially every declaration style in this tree:
# `function name(params): ret` and `const name = (params): ret =>`.
decl = re.compile(r'\b(?:export\s+)?(?:async\s+)?function\s+\w+\s*(\([^)]*\))\s*(:\s*[^{\n]+)?', re.S)
arrow = re.compile(r'\bconst\s+\w+\s*=\s*(?:async\s*)?(\([^)]*\))\s*(:\s*[^=\n]+)?\s*=>', re.S)
total = 0
for f in files:
    try:
        src = open(f).read()
    except OSError:
        continue
    for pat in (decl, arrow):
        for m in pat.finditer(src):
            sig = (m.group(1) or '') + (m.group(2) or '')
            total += len(re.findall(r':\s*any\b', sig))
print(total)
PYEOF
)

# ---------------------------------------------------------------------------
# 9d. D4 — commented_empty_catch: a `catch` whose body is ONLY a comment.
#
# L3 named the masking pattern as a triple, and this is its third layer. A bare
# `catch {}` at least looks like an oversight; a catch containing
# `/* readouts must never break the tick */` looks CONSIDERED, so the next
# reader accepts it and moves on. That exact comment is what kept F-A invisible
# while it truncated a 135-line tick every frame.
#
# Nothing counted these before: `empty_ts_catch`'s regex requires whitespace
# only, so every comment-bearing swallow sat outside all three existing
# counters. 112 across 46 files, datamap.tsx leading with 19. Non-increasing —
# and unlike the others, the fix is usually to LOG rather than to delete, since
# the comment often records a real reason that simply deserves a log line.
# ---------------------------------------------------------------------------
commented_empty_catch=$(python3 - <<'PYEOF'
import re, subprocess
files = [f for f in subprocess.run(['git', 'ls-files', '*.ts', '*.tsx'],
                                   capture_output=True, text=True).stdout.split()
         if '/node_modules/' not in f]
pat = re.compile(r'catch\s*(?:\([^)]*\))?\s*\{\s*(?://[^\n]*|/\*.*?\*/)\s*\}', re.S)
total = 0
for f in files:
    try:
        total += len(pat.findall(open(f).read()))
    except OSError:
        pass
print(total)
PYEOF
)

# ---------------------------------------------------------------------------
# 9e. D5 — conflicting_const: an exported SCREAMING_CASE constant declared in
# more than one module with DIFFERENT values.
#
# Two modules can each be internally consistent and still disagree with each
# other, and nothing in the type system notices — the names match, so a reader
# skimming either file sees nothing wrong. Found on its first run:
#
#   EARTH_RADIUS_KM      6371 (orbital/geometry.ts) vs 6378.137
#                        (orbital/satDerived.ts) — mean vs WGS84 equatorial, in
#                        the SAME orbital module, ~7km apart in satellite
#                        altitude
#   EARTH_CIRCUMFERENCE_M  2*PI*6371008.8 (glElev.ts) vs 40075016.686 (lod.ts)
#                        — ~45km apart, and Web Mercator requires the latter
#
# Both are accuracy defects in code whose whole premise is real positions, not
# style nits. Filed as Q14 rather than fixed here (one logical change per PR).
# Non-increasing. Constants that legitimately differ belong under distinct
# names, which is what makes the fix and the ratchet agree.
# ---------------------------------------------------------------------------
conflicting_const=$(python3 - <<'PYEOF'
import re, subprocess, collections
files = [f for f in subprocess.run(['git', 'ls-files', '*.ts', '*.tsx'],
                                   capture_output=True, text=True).stdout.split()
         if '/node_modules/' not in f and '.test.' not in f]
vals = collections.defaultdict(set)
pat = re.compile(r'^export const ([A-Z][A-Z0-9_]{2,})\s*(?::[^=]+)?=\s*([^;\n]+);?\s*$', re.M)
for f in files:
    try:
        src = open(f).read()
    except OSError:
        continue
    for m in pat.finditer(src):
        vals[m.group(1)].add(m.group(2).strip())
print(sum(1 for v in vals.values() if len(v) > 1))
PYEOF
)

# ---------------------------------------------------------------------------
# 9f. D6 — undeclared_py_import: a third-party module imported by tracked
# Python code but named in neither requirements.txt nor requirements-dev.txt.
#
# T1.1 is the argument for this one. `test_grid_county_ba.py` imports openpyxl
# transitively, and in an environment lacking it the resulting COLLECTION error
# aborts the ENTIRE pytest run — 1337 passes became `1 skipped, 1 error` in 4
# seconds. One undeclared import takes down the whole suite, not one file.
#
# (That specific case turned out to be openpyxl declared in requirements-dev.txt
# with CI simply never installing that file — fixed in the same PR. This counter
# catches the harder version: imports declared NOWHERE, which fail the moment a
# clean container touches them.)
#
# Stdlib and first-party modules are excluded, and the alias table maps import
# names to distribution names (sklearn->scikit-learn, shapefile->pyshp, ...).
# Baseline 2: laspy (scripts/gridvision_lidar_probe.py) and ultralytics
# (scripts/gridvision_train/train.py), both GRID VISION GPU tooling that only
# runs on a RunPod box. Non-increasing.
# ---------------------------------------------------------------------------
undeclared_py_import=$(python3 - <<'PYEOF'
import ast, os, re, subprocess, sys
files = [f for f in subprocess.run(['git', 'ls-files', '*.py'],
                                   capture_output=True, text=True).stdout.split()
         if '/node_modules/' not in f]
local = {os.path.splitext(os.path.basename(f))[0] for f in files}
local |= {p.split('/')[0] for p in files if '/' in p}
declared = set()
for rq in ('requirements.txt', 'requirements-dev.txt'):
    try:
        for line in open(rq):
            line = line.split('#')[0].strip()
            if line:
                declared.add(re.split(r'[<>=\[!]', line)[0].strip().lower().replace('-', '_'))
    except OSError:
        pass
ALIAS = {'sklearn': 'scikit_learn', 'yaml': 'pyyaml', 'PIL': 'pillow',
         'cv2': 'opencv_python', 'dateutil': 'python_dateutil',
         'bs4': 'beautifulsoup4', 'shapefile': 'pyshp'}
std = getattr(sys, 'stdlib_module_names', set())
missing = set()
for f in files:
    try:
        tree = ast.parse(open(f).read())
    except Exception:
        continue
    for n in ast.walk(tree):
        mods = []
        if isinstance(n, ast.Import):
            mods = [a.name.split('.')[0] for a in n.names]
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            mods = [n.module.split('.')[0]]
        for m in mods:
            if m in std or m in local or m.startswith('_'):
                continue
            if ALIAS.get(m, m).lower().replace('-', '_') not in declared:
                missing.add(m)
print(len(missing))
PYEOF
)

# ---------------------------------------------------------------------------
# 9g. D7 — dead_workflow_env: an env var SET in a workflow and read by NOTHING
# in tracked code.
#
# Found on its first run: `VOLTRADE_CI` is set in two ci.yml jobs, and grep
# finds zero readers repo-wide. The comment beside it claims "Network-dependent
# tests are excluded in CI" — but no test consults the variable, so that
# exclusion rests entirely on the hand-picked four-file list the same job runs,
# not on any mechanism. A env var that looks like a switch and is wired to
# nothing is worse than no switch: it makes the next reader believe a control
# exists. CLAUDE.md's STALENESS AUDIT names this class outright ("dead config,
# unused env var reads").
#
# GitHub's own injected names (GITHUB_TOKEN, GITHUB_OUTPUT, ...) and workflow
# locals passed straight into a `run:` block are excluded. Non-increasing.
# ---------------------------------------------------------------------------
dead_workflow_env=$(python3 - <<'PYEOF'
import re, subprocess
try:
    import yaml
except ImportError:
    print(0); raise SystemExit
setnames = set()
for wf in subprocess.run(['git', 'ls-files', '.github/workflows/*.yml'],
                         capture_output=True, text=True).stdout.split():
    try:
        d = yaml.safe_load(open(wf))
    except Exception:
        continue
    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == 'env' and isinstance(v, dict):
                    setnames.update(v.keys())
                else:
                    walk(v)
        elif isinstance(o, list):
            for x in o:
                walk(x)
    walk(d)
src = ""
for f in subprocess.run(['git', 'ls-files', '*.py', '*.ts', '*.tsx', '*.sh', '*.mjs'],
                        capture_output=True, text=True).stdout.split():
    if '/node_modules/' in f:
        continue
    try:
        raw = open(f).read()
    except OSError:
        continue
    # STRIP COMMENTS FIRST (PROGRAM_STATE.md L11). A source-scraping check
    # cannot tell code from prose about code — and the comment block above,
    # which names VOLTRADE_CI while explaining this very counter, made the
    # detector count itself as a reader and report 0 instead of 1.
    src += "\n".join(l for l in raw.split("\n")
                      if not re.match(r'\s*(#|//|\*|/\*)', l))
GITHUB = {'GITHUB_TOKEN', 'GH_TOKEN', 'GITHUB_OUTPUT', 'EVENT', 'BASE_REF', 'BEFORE', 'PR_URL'}

def is_read(name, text):
    """A READ of the env var, not an assignment.

    FIXED 2026-08-14: this was a bare name grep, and it went 1 -> 0 the moment
    scripts/gated_tests.sh added `VOLTRADE_CI=1 run_suite ...` — an ASSIGNMENT,
    counted as a reader. The counter then reported all-clear while the variable
    was still exactly as decorative as when D7 first found it. Matching the
    actual read syntaxes instead: os.environ / getenv, process.env.X, and $X.
    """
    n = re.escape(name)
    return bool(
        re.search(r'os\.environ(?:\.get)?\s*[\[(]\s*[\'"]' + n, text) or
        re.search(r'getenv\s*\(\s*[\'"]' + n, text) or
        re.search(r'process\.env\.' + n + r'\b', text) or
        re.search(r'process\.env\s*\[\s*[\'"]' + n, text) or
        re.search(r'\$\{?' + n + r'\}?\b', text)
    )

print(sum(1 for n in setnames if n not in GITHUB and not is_read(n, src)))
PYEOF
)

# ---------------------------------------------------------------------------
# 9h. D8 — uncapped_surface: a module that acquires a canvas/WebGL context and
# reads devicePixelRatio WITHOUT clamping to the device tier.
#
# Generalises F-C past the two files it named. datamap.tsx always clamped to
# tier.pixelRatioCap; celestialSky.ts and spaceFrame.ts did not, and nothing
# would have caught the next surface that forgot. Backing-store cost is
# QUADRATIC in the ratio — a 3x phone at an uncapped 3x allocates 9x the pixels
# of a 1x surface, for every raster op the module performs.
#
# Baseline 3 after the celestial fix: DataWorldMap.tsx, bot.tsx, login.tsx
# (filed as Q19 — each needs its own look; a small login canvas is not the same
# call as a full-screen map). Must reach 0.
#
# Comments are stripped before matching (L15) — the comments explaining this
# very fix quote `devicePixelRatio`, and a source scan cannot tell code from
# prose about code.
# ---------------------------------------------------------------------------
uncapped_surface=$(python3 - <<'PYEOF'
import re, subprocess
files = [f for f in subprocess.run(['git', 'ls-files', 'client/src/**/*.ts', 'client/src/**/*.tsx'],
                                   capture_output=True, text=True).stdout.split()
         if '.test.' not in f]
n = 0
for f in files:
    try:
        raw = open(f).read()
    except OSError:
        continue
    code = "\n".join(l for l in raw.split("\n") if not re.match(r'\s*(//|\*|/\*)', l))
    if not re.search(r'getContext\(\s*["\'](webgl2?|2d)', code):
        continue
    if re.search(r'devicePixelRatio', code) and not re.search(r'surfacePixelRatio|pixelRatioCap', code):
        n += 1
print(n)
PYEOF
)

# ---------------------------------------------------------------------------
# 9i. D9 — assertions: total assert statements across every test file.
#
# CLAUDE.md's MUTABLE section is explicit — "Never delete or weaken an existing
# assertion to make your change pass" — and until now NOTHING counted it. A
# file-count check misses the real move: keeping the file and quietly removing
# the assertion inside it that was in the way.
#
# T1.2 makes this urgent rather than theoretical. Now that a red test blocks a
# merge, deleting the assertion is the cheapest way to get green, and the
# quarantine route is already closed (gate 2 rejects a grown list). This closes
# the other door.
#
# NON-DECREASING, unlike most counters here — the only one that must go UP.
# A legitimate refactor that consolidates assertions will trip it; that is
# intended, and the answer is to say so in the PR, not to quietly re-pin.
# Comments stripped first (L15).
# ---------------------------------------------------------------------------
assertions=$(python3 - <<'PYEOF'
import re, subprocess
ts = subprocess.run(['git', 'ls-files', '*.test.ts', '*.test.tsx'],
                    capture_output=True, text=True).stdout.split()
py = [f for f in subprocess.run(['git', 'ls-files', '*.py'],
                                capture_output=True, text=True).stdout.split()
      if re.search(r'(^|/)test_|_test\.py$', f)]
total = 0
for f in ts:
    try:
        raw = open(f).read()
    except OSError:
        continue
    code = "\n".join(l for l in raw.split("\n") if not re.match(r'\s*(//|\*|/\*)', l))
    total += len(re.findall(r'\bassert\s*(?:\.\w+)?\s*\(', code))
for f in py:
    try:
        raw = open(f).read()
    except OSError:
        continue
    code = "\n".join(l for l in raw.split("\n") if not re.match(r'\s*#', l))
    total += len(re.findall(r'\bself\.assert\w+\s*\(|^\s*assert\s', code, re.M))
print(total)
PYEOF
)

# ---------------------------------------------------------------------------
# 9j. D10 — baseline_divergence: this script's PRINTED baseline column
# disagreeing with the pin CI actually enforces in ci/counter_baseline.txt.
#
# Found while re-pinning Q13's two counters: the display column said
# `ts_any ... 1252` where the enforced pin said 1251. That is one number with
# two homes, which is the exact failure ci/counter_baseline.txt's own header
# warns about for tsc ("two pins for one number is how they drift apart") — and
# then this file did it to itself.
#
# It matters because the printed column is what a session READS. On first run
# it found FIVE divergences, including `uncapped_surface 3` (fixed to 0 in
# #837, so the table showed a regression that did not exist) and `detectors 0`
# against a real 9. A status table that misreports the target is worse than no
# table: §4.2 says these numbers are the definition of progress, so a session
# trusts them.
#
# The structural fix is to delete the hardcoded column and read the pin file —
# filed as Q23, not done here (one logical change per PR). Until then this
# counts the drift, non-increasing, so it cannot grow while waiting.
#
# Compares only counters present in BOTH: 10 pins have no display row and 5
# display rows have no pin, which is a naming mismatch filed with Q23.
# ---------------------------------------------------------------------------
baseline_divergence=$(python3 - <<'PYEOF'
import re

disp = {}
for line in open('scripts/program_status.sh'):
    m = re.match(r"printf\s+'[^']*'\s+(\w+)\s+\"\$\{?\w+\}?\"\s+\"(-?\d+)\"", line.strip())
    if m:
        disp[m.group(1)] = int(m.group(2))

pins = {}
try:
    for line in open('ci/counter_baseline.txt'):
        parts = line.split('#')[0].split()
        if len(parts) == 3:
            pins[parts[0]] = int(parts[1])
except OSError:
    pass

print(sum(1 for k, v in disp.items() if k in pins and pins[k] != v))
PYEOF
)

# ---------------------------------------------------------------------------
# 9k. D11 — dup_precise_literal: a HIGH-PRECISION numeric literal restated in
# more than one module. Counts the redundant copies, not the distinct values,
# so it falls when a copy is deleted.
#
# This is the mechanism BEHIND D5. Q14's two colliding constants did not appear
# from nowhere: `6371008.8` was written out longhand in THREE modules
# (glElev.ts, orbital/occlusion.ts, air/trackModel.ts), so there were three
# places to be wrong and nothing tying them together. D5 only notices once two
# EXPORTED names collide — by then the drift already happened. This notices the
# copy.
#
# On its first run it found what Q14 did not: `40075016.686` is also restated in
# cameraRig.ts, and `6378.137` in propagate.ts as a bare `const a`. Neither is
# an exported name, so D5 is blind to both.
#
# >=7 significant digits, with trailing zeros NOT counted as significant — that
# is what separates a measured constant from a round one, and without it
# 86400000 (ms/day, 4 modules) and 1000000 dominate the list with numbers no
# one could get wrong. Comments and string literals excluded via
# scripts/ts_code_only.py (Q13), so prose quoting a constant is not a copy of it.
# ---------------------------------------------------------------------------
dup_precise_literal=$(python3 - <<'PYEOF'
import collections, os, re, subprocess, sys
sys.path.insert(0, os.path.join(os.getcwd(), 'scripts'))
from ts_code_only import blank_source, read_text

files = [f for f in subprocess.run(['git', 'ls-files', '*.ts', '*.tsx'],
                                   capture_output=True, text=True).stdout.split()
         if '/node_modules/' not in f and '.test.' not in f]
NUM = re.compile(r'(?<![\w.])(\d+\.\d+|\d{6,})(?![\w.])')


def significant(tok):
    digits = tok.replace('.', '')
    # Integers: trailing zeros are place-holders, not precision (86400000 -> 3).
    # Decimals: a written trailing zero IS precision, so only strip leading.
    return len(digits.strip('0')) if '.' not in tok else len(digits.lstrip('0'))


seen = collections.defaultdict(set)
for f in files:
    src = read_text(f)
    if src is None:
        continue
    for m in NUM.finditer(blank_source(src)):
        if significant(m.group(1)) >= 7:
            seen[m.group(1)].add(f)
print(sum(len(v) - 1 for v in seen.values() if len(v) > 1))
PYEOF
)

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
  "tests_run_in_ci": $tests_run_in_ci,
  "tests_gating_merge": $tests_gating_merge,
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
  "long_try_empty_catch": $long_try_empty_catch,
  "boundary_any": $boundary_any,
  "commented_empty_catch": $commented_empty_catch,
  "conflicting_const": $conflicting_const,
  "undeclared_py_import": $undeclared_py_import,
  "dead_workflow_env": $dead_workflow_env,
  "uncapped_surface": $uncapped_surface,
  "assertions": $assertions,
  "harness_rules_checked": $harness_rules_checked,
  "baseline_divergence": $baseline_divergence,
  "dup_precise_literal": $dup_precise_literal,
  "detectors_registered": $detectors_registered,
  "quarantine_size": $quarantine_size,
  "quarantine_oldest_days": $quarantine_oldest_days
}
EOF
  exit 0
fi

printf '%-24s %-14s %-12s %s\n' COUNTER VALUE BASELINE DIRECTION
printf '%-24s %-14s %-12s %s\n' ------- ----- -------- ---------
printf '%-24s %-14s %-12s %s\n' tests_run_in_ci    "$tests_run_in_ci/$tests_total" "4/364" "must increase"
printf '%-24s %-14s %-12s %s\n' tests_gating_merge "$tests_gating_merge/$tests_total" "4/364" "must increase (>216)"
printf '%-24s %-14s %-12s %s\n' tsc_errors         "$tsc_errors"          "83"     "must decrease"
printf '%-24s %-14s %-12s %s\n' "  of which TS2304" "$tsc_2304"           "5"      "must reach 0 (always real bugs)"
printf '%-24s %-14s %-12s %s\n' silent_py_handlers "$silent_py/$py_except_total" "255/873" "non-increasing"
printf '%-24s %-14s %-12s %s\n' bare_except        "$bare_except"         "3"      "non-increasing"
printf '%-24s %-14s %-12s %s\n' empty_ts_catch     "$empty_ts_catch"      "494"    "non-increasing"
printf '%-24s %-14s %-12s %s\n' ts_any             "$ts_any"              "1237"   "non-increasing"
printf '%-24s %-14s %-12s %s\n' layers_full_schema "$layers_full/$layers_total" "1/238" "non-decreasing"
printf '%-24s %-14s %-12s %s\n' "  layers with lod" "$layers_lod"         "1"      "non-decreasing"
printf '%-24s %-14s %-12s %s\n' law_iv_scanned     "$law_iv_scanned"      "5"      "must reach $law_iv_ctx (ctx-acquiring)"
printf '%-24s %-14s %-12s %s\n' order_post_sites   "$order_post_sites"    "6"      "must reach 1"
printf '%-24s %-14s %-12s %s\n' design_token_drift "$design_token_drift"  "0"      "must stay 0"
printf '%-24s %-14s %-12s %s\n' long_try_empty_catch "$long_try_empty_catch" "3"     "non-increasing"
printf '%-24s %-14s %-12s %s\n' boundary_any       "$boundary_any"        "233"    "non-increasing"
printf '%-24s %-14s %-12s %s\n' commented_catch    "$commented_empty_catch" "113"   "non-increasing"
printf '%-24s %-14s %-12s %s\n' conflicting_const  "$conflicting_const"   "3"      "non-increasing"
printf '%-24s %-14s %-12s %s\n' undeclared_py_imp  "$undeclared_py_import" "2"     "non-increasing"
printf '%-24s %-14s %-12s %s\n' dead_workflow_env  "$dead_workflow_env"   "1"      "must reach 0"
printf '%-24s %-14s %-12s %s\n' uncapped_surface   "$uncapped_surface"    "3"      "must reach 0"
printf '%-24s %-14s %-12s %s\n' assertions         "$assertions"          "11313"  "NON-DECREASING"
printf '%-24s %-14s %-12s %s\n' harness_rules      "$harness_rules_checked" "71"   "non-decreasing"
printf '%-24s %-14s %-12s %s\n' baseline_diverge   "$baseline_divergence" "2"      "must reach 0"
printf '%-24s %-14s %-12s %s\n' dup_precise_lit    "$dup_precise_literal"  "4"      "non-increasing"
printf '%-24s %-14s %-12s %s\n' detectors          "$detectors_registered" "0"     "MUST increase each session"
printf '%-24s %-14s %-12s %s\n' quarantine_size    "$quarantine_size"     "1"      "non-increasing"
printf '%-24s %-14s %-12s %s\n' quarantine_oldest  "${quarantine_oldest_days}d" "0d" "fail if >30"

if [ "$tsc_errors" = "skipped" ]; then
  echo
  echo "NOTE: tsc skipped (no node_modules/typescript, or --no-tsc). Run 'npm ci' first."
fi
