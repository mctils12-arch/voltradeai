# PROGRAM_STATE.md — the MASTER PROGRAM resume file

**One file. Updated every session. Never replaced.** (MASTER PROGRAM §4.1)

Context dies between sessions; the repo is the only memory. A finding that
exists only in a transcript did not happen. Read this after `CLAUDE.md` and
before the MASTER PROGRAM document, then run `scripts/program_status.sh` and
**trust its numbers over any prose here, including this file's own NUMBERS
block** (§0.2).

---

## NEXT — the single highest-value unclaimed item

**Q5 / T1.6 — replace `npx tsc --noEmit || true` in `ci.yml` with a hard
ratchet.** Now unblocked and now honest: the baseline is **12**, every one of
them a genuine type issue (list in `research/tsc_baseline.md` §4), with zero
phantoms left to tempt a suppression "fix". `.github/workflows/` is FROZEN —
the MASTER PROGRAM is the specific authorization and must be cited in the PR
body, the way `celestial-catalog-mirror.yml` cites its own approval.
Pin at 12, non-increasing; do NOT pin at 0 (that would force the 12 real fixes
into this PR, and three of them are in FROZEN `billing.ts`).

Do **not** start Track 2 or 3 before Q1–Q5 land — §12 names that as a failure
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
| Q5 | T1.6 — replace `\|\| true` with a ratchet on the post-Q2/Q3 count | T1.6 | **TODO** ← next, UNBLOCKED — pin at 12 |
| Q6 | T2.4 — cap DPR in `celestialSky:788` + `spaceFrame:2877` | T2.4 | **TODO** — up to 9× faster moon on a 3× device |
| Q7 | T2.1/T2.2 — widen the Law IV predicate to context-acquiring modules | T2.1 | **TODO** — 5 → 7 files; it will fail, that is the deliverable |
| Q8 | T2.6 — the §2.1 F16 NaN-guard unit test | T2.6 | **TODO** — closes a PR open since 2026-08-12 |
| Q9 | T8.1 — design-token drift check into the harness | T8.1 | **TODO** — measured 0 today, so it starts green |
| Q10 | T1.1 — all three suites into CI non-blocking + `visual --soft` | T1.1 | **TODO** — **promoted: this is what arms `server/tsc2304Ratchet.test.ts`, which no CI job runs today** |
| Q11 | T4.1 — `renderKind` + `lod` required in `layersRegistry.test.ts` | T4.1 | **TODO** — will fail on 237 of 238 layers; that number is the deliverable |
| Q12 | `server/gridTiles.test.ts` asserts ≥50 pmtiles; 3 exist and none were ever committed — decide: build the tiles (A1/A4), or quarantine with a reason | T1.2 | **TODO** — found by running the suite, see L8 |
| Q13 | `empty_ts_catch` / `ts_any` count comment text — strip comments and string literals before counting | T0.1 | **TODO** — MEASUREMENT INTEGRITY: own PR, must state before/after on identical inputs, see L9 |

**The queue is not empty.** §0.3 condition 5 satisfied.

---

## NUMBERS

`scripts/program_status.sh`, run 2026-08-13 at commit `0d3d7d8`,
`package.json` 1.0.701, after `npm ci`.

```
COUNTER                  VALUE          BASELINE     DIRECTION
gated_tests              4/364          4/364        must increase (>216)
tsc_errors               12             83           must decrease
  of which TS2304        0              5            AT TARGET — hold at 0
silent_py_handlers       255/873        255/873      non-increasing
bare_except              3              3            non-increasing
empty_ts_catch           495            495          non-increasing
ts_any                   1250           1252         non-increasing
boundary_any             233            233          non-increasing
commented_catch          112            112          non-increasing
layers_full_schema       1/238          1/238        non-decreasing
  layers with lod        1              1            non-decreasing
law_iv_scanned           5              5            must reach 7 (ctx-acquiring)
order_post_sites         6              6            must reach 1
design_token_drift       0              0            must stay 0
harness_rules            71             71           non-decreasing
detectors                4              0            MUST increase each session
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

**Seeds not yet taken** (MASTER PROGRAM §0.7, plus new ones from this session):

- `useEffect` with a dependency array omitting a ref it reads
- registry ids in `layers.json` with no server route; routes with no id
- `setInterval` callbacks that can outlive their component
- exported constants duplicated across modules with different values
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
