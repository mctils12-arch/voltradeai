# PROGRAM_STATE.md — the MASTER PROGRAM resume file

**One file. Updated every session. Never replaced.** (MASTER PROGRAM §4.1)

Context dies between sessions; the repo is the only memory. A finding that
exists only in a transcript did not happen. Read this after `CLAUDE.md` and
before the MASTER PROGRAM document, then run `scripts/program_status.sh` and
**trust its numbers over any prose here, including this file's own NUMBERS
block** (§0.2).

---

## NEXT — the single highest-value unclaimed item

**T1.2 — promote the green set to REQUIRED and quarantine the rest.** Q15 is
now FIXED (#830), so the last blocker is Q12 alone. `tests_run_in_ci` is
368/368 but `tests_gating_merge` is still **4/368**.

Build `ci/required.txt` + `ci/quarantine.txt` (reason + date per entry) and make
the required set blocking:
- **Q12** — `gridTiles.test.ts` asserts ≥50 pmtiles and 3 exist, none ever
  committed. The one standing failure. Quarantine with a reason, or resolve via
  A1/A4. This is the only thing that would red the gate on day one.
- **Q18** — decide what `VOLTRADE_CI` is for before promoting the python suite.
  It is set in two ci.yml jobs and **read by nothing repo-wide** (D7 found it),
  so `ci.yml`'s claim that "network-dependent tests are excluded in CI" rests on
  the hand-picked four-file list, not on a mechanism. Either wire it up (tests
  that reach the network skip when it is set) or delete it and the claim. Five
  test files import `requests`/`socket` directly with no guard at all.

Do **not** start Track 2 or 3 before Track 1 lands — §12 names that as a failure
mode by name.

---

## QUEUE

Ordered. Take the highest-priority unclaimed item. Mark `CLAIMED(session-id)`
when you take it, `DONE` with the PR number when it merges.

| # | item | track | state |
|---|---|---|---|
| Q0 | `tsc --noEmit` run + full triage of all 83 errors | T0.0 | **DONE** — `research/tsc_baseline.md`, this session |
| Q1 | `program_status.sh` + `PROGRAM_STATE.md` | T0.1 | **DONE** — this session |
| Q2 | `execAsync`/`execPythonSerialized` opts typed `ExecOptions` + encoding pinned (clears 42 errors) | T0.0/T5 | **DONE** — PR #824 |
| Q3 | `tsconfig.json` gains `"target": "ES2022"` (clears 29) | T0.0 | **DONE** — PR #825 |
| Q4 | Fix the 5 TS2304 real bugs + `tsc_2304` ratchet test | F-A + new | **DONE** — PR #823, session 2 |
| Q5 | T1.6 — replace `\|\| true` with a ratchet on the post-Q2/Q3 count | T1.6 | **DONE** — PR #826, pinned at 12 |
| Q6 | T2.4 — cap DPR in `celestialSky:788` + `spaceFrame:2877` | T2.4 | **TODO** — up to 9× faster moon on a 3× device |
| Q7 | T2.1/T2.2 — widen the Law IV predicate to context-acquiring modules | T2.1 | **TODO** — 5 → 7 files; it will fail, that is the deliverable |
| Q8 | T2.6 — the §2.1 F16 NaN-guard unit test | T2.6 | **TODO** — closes a PR open since 2026-08-12 |
| Q9 | T8.1 — design-token drift check into the harness | T8.1 | **TODO** — measured 0 today, so it starts green |
| Q10 | T1.1 — all three suites into CI non-blocking | T1.1 | **DONE** — PR #829. 368/368 files now RUN in CI; 4/368 gate |
| Q17 | T1.2 — `ci/required.txt` + `ci/quarantine.txt`, make the green set blocking | T1.2 | **TODO** ← next — blocked on Q15 (fix) and Q12 (quarantine or resolve) |
| Q11 | T4.1 — `renderKind` + `lod` required in `layersRegistry.test.ts` | T4.1 | **TODO** — will fail on 237 of 238 layers; that number is the deliverable |
| Q12 | `server/gridTiles.test.ts` asserts ≥50 pmtiles; 3 exist and none were ever committed — decide: build the tiles (A1/A4), or quarantine with a reason | T1.2 | **TODO** — found by running the suite, see L8 |
| Q14 | `EARTH_RADIUS_KM` 6371 vs 6378.137 (both in `client/src/lib/orbital/`) and `EARTH_CIRCUMFERENCE_M` 2πR vs 40075016.686 — pick one per meaning, or rename so the difference is explicit | T2/orbital | **TODO** — found by D5; ~7km in sat altitude, ~45km in a mercator constant. Accuracy defects in code whose premise is real positions |
| Q15 | `server/datacoreArchive.test.ts` rollup tests fail near UTC midnight | T1.2 | **DONE** — PR #830. Fixed, not quarantined: it was a bug in the test's date arithmetic, never in the code under test |
| Q18 | `VOLTRADE_CI` is set in 2 ci.yml jobs and read by NOTHING repo-wide; ci.yml's "network-dependent tests are excluded" claim has no mechanism behind it. 5 test files import requests/socket unguarded | T1.2 | **TODO** — found by D7; blocks promoting the python suite to required |
| Q16 | CI never installed `requirements-dev.txt`, so `test_grid_county_ba.py` could not import openpyxl and the COLLECTION error aborted the whole python suite (1337 passes → `1 skipped, 1 error`) | T1.1 | **DONE** — PR #829. Not fixed by moving openpyxl into requirements.txt: that file feeds the frozen Dockerfile's production image |
| Q13 | `empty_ts_catch` / `ts_any` count comment text — strip comments and string literals before counting | T0.1 | **TODO** — MEASUREMENT INTEGRITY: own PR, must state before/after on identical inputs, see L9 |

**The queue is not empty.** §0.3 condition 5 satisfied.

---

## NUMBERS

`scripts/program_status.sh`, run 2026-08-13 at commit `0d3d7d8`,
`package.json` 1.0.701, after `npm ci`.

```
COUNTER                  VALUE          BASELINE     DIRECTION
tests_run_in_ci          368/368        4/364        must increase
tests_gating_merge       4/368          4/364        must increase (>216)
tsc_errors               12             83           must decrease
  of which TS2304        0              5            AT TARGET — hold at 0
silent_py_handlers       255/873        255/873      non-increasing
bare_except              3              3            non-increasing
empty_ts_catch           495            495          non-increasing
ts_any                   1250           1252         non-increasing
boundary_any             233            233          non-increasing
commented_catch          112            112          non-increasing
conflicting_const        5              5            non-increasing
undeclared_py_imp        2              2            non-increasing
dead_workflow_env        1              1            must reach 0
layers_full_schema       1/238          1/238        non-decreasing
  layers with lod        1              1            non-decreasing
law_iv_scanned           5              5            must reach 7 (ctx-acquiring)
order_post_sites         6              6            must reach 1
design_token_drift       0              0            must stay 0
harness_rules            71             71           non-decreasing
detectors                7              0            MUST increase each session
quarantine_size          0              0            non-increasing
quarantine_oldest        0d             0d           fail if >30
```

**Where these differ from the MASTER PROGRAM's §4.2 table, this block is
right** — §4.2's numbers were measured by hand at audit time and the tree has
moved since (the audit is dated to a 237-layer registry; there are 238 now).
Differences worth naming:

- `layers_full_schema` **1/238**, not 8/237. The audit counted layers carrying
  the four fields `renderKind`/`time`/`provenance`/`altitudeRef` (9 do). The
  counter requires all **five** unenforced fields including `lod` — and `lod`
  is on exactly one layer. 1 is the honest number for "carries the full
  schema"; F-E is worse than stated.
- `silent_py_handlers` **255/873**, not 308/869. Different predicate: this
  counts handlers whose entire body is `pass`/`continue`. A handler that logs
  is doing its job and is not counted.
- `order_post_sites` **6**, not 9. The counter measures **files** referencing
  `/v2/orders` outside tests, because a choke point is a module boundary, not a
  line count. The 6: `server/bot.ts`, `bot_engine.py`, `server/routes.ts`,
  `options_manager.py`, `options_execution.py`, `intraday_shorts.py`.
- `law_iv_context_files` = **7** — the audit's "target 7+" is exactly right.
  Two context-acquiring modules (`celestialSky.ts`, `spaceFrame.ts`) sit
  outside the `*Layer.ts` predicate. F-B confirmed.
- `tests_total` **364**, not 360.

---

## LEARNED — findings that change what future sessions do

**L1 — `tsc_errors` alone is the wrong ratchet metric; `tsc_2304` is the right
one.** Of 83 errors, 29 are manufactured by a missing `target` in
`tsconfig.json` and 42 by one untyped function. 86% from two mechanical causes.
A ratchet pinned at 83 pins 71 phantoms and tempts a future session to clear
them by suppression — which §12 lists as a failure mode. TS2304 is the only
code that is never noise: the name is in scope or it is not.

**L2 — a second bug of the `altScale` class exists, and the full static audit
missed it.** `datamap.tsx:6889` references `e` inside `focusSat(index)`, which
has no such parameter — left behind when the satellite-focus body was extracted
out of `onClick(e)` for SatFinder reuse. Clicking a satellite never stamps
`__vtFeatClaim`, so the curtain/card interaction at 3882 silently takes the
wrong branch. Found by running a tool that was already installed.

**L3 — the masking pattern is a triple, and all three parts are load-bearing.**
`|| true` in `ci.yml` hides the type error; an empty `catch {}` swallows the
`ReferenceError`; and a *reassuring comment* (`/* readouts must never break the
tick */`) makes the swallow look deliberate to the next reader. Any one alone
would have been caught. When adding a detector, prefer ones that fire at the
first layer.

**L4 — `altScale` is worse than "a wrong readout".** The throw at 4384 skips
every remaining statement in the tick, including GND SPD, VERT SPD, and the
whole follow-aircraft recenter block at 4412. A TS2304 inside a long `try`
truncates its block, it does not just break its own line.

**L5 — clear `node_modules/typescript/tsbuildinfo` before any A/B on
`tsconfig.json`.** `"incremental": true` served a stale, unchanged 83 on the
first attempt at the §3 experiment and nearly published a false negative.

**L6 — the visual harness is richer than the audit implies.** F-H says UI rests
on five mechanical checks; `visual_check.mjs` actually carries 71 distinct
failure assertions including legend parity, imagery-date honesty, TTI budgets
and self-see. Track 8 is still right that the *specific* `DESIGN.md` numbered
rules are unconverted — but it is a smaller gap than "five checks" suggests.
Re-scope T8 against the file before planning it.

**L15 — prose defeated one of my own checks for the FOURTH time today, and
the fix is now a rule I apply up front.** D7 (`dead_workflow_env`) reported 0
instead of 1 because the comment block explaining the counter names
`VOLTRADE_CI`, so the detector counted its own documentation as a reader. Same
shape as L9, L11 and L12. Every source-scraping check in
`program_status.sh` now strips comment lines before searching, and that is the
FIRST thing to write, not the fix after the false reading. The general form:
**a checker and its own explanation live in the same file, so the explanation
is part of the corpus unless you exclude it.**

**L13 — "CI runs it" and "its failure blocks a merge" are different numbers,
and T1.1 deliberately created a gap between them.** The old `gated_tests`
counted test files NAMED in `ci.yml`. That worked while CI listed four by hand
and broke the moment T1.1 added a job running whole globs — it reported 6/368
for a run executing all 368. Collapsing the two facts into one number lies in
whichever direction you pick: report 368 and a non-blocking baseline job
masquerades as a gate; report 4 and ~3,586 tests now running on every PR are
invisible. Split into `tests_run_in_ci` (368/368) and `tests_gating_merge`
(4/368), the latter defined as "not `continue-on-error` at either job or step
level". The MASTER PROGRAM's ">216" target is about GATING, so it now points at
the counter that must actually climb. **Note the direction of this measurement
change: it makes the headline number look WORSE (4, not 368), which is the
right sign for a ruler change** — MEASUREMENT INTEGRITY treats a change that
flatters as suspect by default.

**L14 — the suites were never the problem.** ~3,586 tests, **one** standing
failure, and that one asserts the presence of files never committed. The reason
360-odd tests were not gating was never that they were failing — it was that
nobody had run them. Two of the four apparent failures dissolved on
inspection: the python suite is fully green once CI installs the
`requirements-dev.txt` it already declares, and the `datacoreArchive` pair is a
UTC-midnight arithmetic flake (failed 23:55Z, passed 01:00Z, same commit).
Expect the same shape elsewhere in Track 1: measure before assuming decay.

**L12 — my own resume file shipped self-contradictory, and the cause was a
silently-failing edit.** PR #826 left QUEUE saying "Q5 **DONE** — PR #826" and
NEXT saying "do Q5", in the one file whose entire job is telling the next
session what to do. Cause: I update these files with `python3 -c` string and
regex replacements, and **`str.replace()` / `re.sub()` return the input
unchanged when nothing matches** — no error, no output, exit 0. My pattern
ended `touching a frozen path).` while the text ended ``in FROZEN `billing.ts`).``
so it matched nothing and wrote the file back identical. I *did* verify that
commit, but grepped for the queue rows and the new detector, not for the NEXT
section — so the check passed while the edit had not happened.

**Standing rule: every scripted edit to a research/ or config file must assert
its precondition (`assert old in s`) and re-read the specific region it
changed.** Verifying "something changed" is not verifying "this changed". This
is the same class as L9 and L11 — a check that looks like it covers the thing
but doesn't — and it is the third instance today, which makes it a pattern in
how I work rather than three coincidences.

**L11 — prose moved one of my own checks for the SECOND time today.** L9 was
comments containing `` `catch {}` `` inflating `empty_ts_catch`. This time the
comment in `ci.yml` documenting the old `npx tsc --noEmit || true` line tripped
my own assertion that the line is gone. Both are the same shape: **a
source-scraping check cannot tell code from prose about code**, and the most
natural place to explain a rule is right next to the rule, where the check
sees it. Fixed properly here by stripping comment lines before asserting —
narrowing to the assertion's real meaning ("no CI STEP swallows the
typecheck"), not relaxing it. **Standing guidance: any new source-scraping
check should strip comments FIRST**, and Q13 (do this for `empty_ts_catch` /
`ts_any`) is now the second confirmed instance rather than a one-off.

**L10 — the typecheck baseline was 83; it is now 12, and the 12 are all real.**
Three sequenced PRs (Q4/Q2/Q3) took it there: 5 genuine bugs fixed, 42 errors
from one `any` parameter, 29 from an absent `tsconfig` target. **86% of what
looked like a mountain of technical debt was two mechanical causes**, and the
noise is the entire reason nobody had ever read the list — including the two
live bugs inside it. The residual 12 need individual judgment and are itemised
in `tsc_baseline.md` §4; three sit in FROZEN `billing.ts` (Stripe SDK drift
plus a real `toLowerCase`-on-`string[]` crash risk) and must be filed, not
fixed. **Sequence generalises: clear the mechanical causes first, then the
count means something and a ratchet can be trusted.**

**L9 — the text-based counters can be moved by PROSE, and it caught me the
same session I wrote them.** `empty_ts_catch` and `ts_any` grep raw source, so
a COMMENT containing the literal `catch {}` increments the count. Writing this
session's fix, my own explanatory comments (one in `datamap.tsx`, one in the new
ratchet test) pushed `empty_ts_catch` 495 → 497, and `catch (err: any)` in the
ratchet test pushed `ts_any` 1252 → 1253 — in the very PR whose body claimed
`ts_any` was being held. Caught by running `program_status.sh` before finishing,
which is exactly why §0.3 requires it.

Fixed the code, NOT the ruler: reworded both comments, and narrowed the test's
handler off `unknown` instead of typing it `any`. Tuning the counter to ignore
comments would have been a measurement change that makes the numbers look
better, which MEASUREMENT INTEGRITY treats as suspect by default and which
§12 names as "reducing counts by suppression". Queued as **Q13** so it lands as
its own PR with a before/after on identical inputs, per the same rule.

The general lesson for every future session: **run `scripts/program_status.sh`
before you write the PR body, not after** — a claim about a counter is only
worth making if you re-measured it after your last edit.

**L8 — a test in this repo has never passed, and nobody could have known.**
`server/gridTiles.test.ts` asserts `files.length >= 50` over
`client/public/tiles/*.pmtiles`. **Three exist.** `git log --all` finds no
`power_*.pmtiles` ever committed, so the assertion has failed since the day it
was written — invisible because `gated_tests` is 4 and this is not one of them.
It is simultaneously: (a) the single concrete proof of the Track 1 thesis, that
360 ungated tests protect nothing; (b) confirmation of **A4 PHASE 2 item 2**
("US-full power grid, boot-fetch-from-Release — *filed above, NOT built*") — the
test asserts the built state and the build never happened; and (c) the first
entry Track 1's quarantine file will need, since turning the node suite on
blocking without it would red the build on day one. Queued as Q12.

**L7 — CI's `automerge` job merges any `claude/*` branch when no job
*failed*.** A skipped job counts as mergeable. So a docs-only PR merges without
ever running `tsc`, and a green run is not evidence a counter held. Ratchets
must live in a job that the path filter actually triggers for the files they
guard.

---

## DETECTORS — §0.7, one per session, mandatory

The counter set must grow. A session that adds no detector has not discharged
the duty. `detectors_registered` reads this table.

| id | detector | added | baseline | status |
|---|---|---|---|---|
| D1 | `tsc_2304` — identifiers used outside their declaring scope, repo-wide, as a counter split out of `tsc_errors` | 2026-08-13 | 5 | live in `program_status.sh`; **now 0**, pinned by `server/tsc2304Ratchet.test.ts` |
| D2 | `long_try_empty_catch` — a `try` spanning >50 lines whose `catch` body is empty | 2026-08-13 | 3 | live in `program_status.sh` |
| D3 | `boundary_any` — `: any` in a function's parameter list or return annotation (not bodies) | 2026-08-13 | 233 | live in `program_status.sh` |
| D4 | `commented_empty_catch` — a `catch` whose body is ONLY a comment (L3's third layer; counted by nothing before) | 2026-08-13 | 112 | live in `program_status.sh` |
| D5 | `conflicting_const` — an exported SCREAMING_CASE constant declared in 2+ modules with different values | 2026-08-13 | 5 | live in `program_status.sh`; found Q14 on its first run |
| D6 | `undeclared_py_import` — a third-party module imported by tracked Python but named in neither requirements file | 2026-08-14 | 2 | live in `program_status.sh` (`laspy`, `ultralytics` — GRID VISION GPU tooling) |
| D7 | `dead_workflow_env` — an env var SET in a workflow and read by NOTHING in tracked code | 2026-08-14 | 1 | live in `program_status.sh`; found Q18 (`VOLTRADE_CI`) on its first run |

**Seeds not yet taken** (MASTER PROGRAM §0.7, plus new ones from this session):

- `useEffect` with a dependency array omitting a ref it reads
- registry ids in `layers.json` with no server route; routes with no id
- `setInterval` callbacks that can outlive their component
- `any` at a module boundary (params and return types only)
- theme-token literals hardcoded instead of referenced (D11) — *partly covered
  by `design_token_drift`; the off-palette-hex half is not yet built*
- functions taking a parameter that shadows an outer binding of the same name
  (the inverse of D1 — would catch the `focusSat` extraction *before* the
  binding is lost)

---

## SESSION LOG

### 2026-08-13 — session 1. Territory: T-BOT/shared (instrumentation only)

Day One items 0 and 1. No runtime code touched.

- **T0.0** — ran `npx tsc --noEmit` for the first time. 83 errors, all triaged
  into `research/tsc_baseline.md`. Confirmed F-A (`altScale`, 3 sites) and
  found **L2**, a second live bug of the same class the audit missed.
- **T0.1** — built `scripts/program_status.sh` (16 counters, `--json` for CI)
  and this file.
- Verified the missing-`target` hypothesis by experiment rather than assertion:
  83 → 54 with `"target": "ES2022"`, cache cleared. Not applied — own PR (Q3).
- **D12 re-verified independently**: `design_token_drift` = 0. All 13 documented
  tokens in `DESIGN.md` match `client/src/index.css` exactly.
- Detector added: **D1**.
- **STARVED: yes** — Q2 through Q11 are queued and unclaimed.
