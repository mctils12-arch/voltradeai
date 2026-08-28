# PROGRAM_STATE.md — the MASTER PROGRAM resume file

**One file. Updated every session. Never replaced.** (MASTER PROGRAM §4.1)

Context dies between sessions; the repo is the only memory. A finding that
exists only in a transcript did not happen. Read this after `CLAUDE.md` and
before the MASTER PROGRAM document, then run `scripts/program_status.sh` and
**trust its numbers over any prose here, including this file's own NUMBERS
block** (§0.2).

---

## NEXT — the single highest-value unclaimed item

**Track 1 is COMPLETE.** Three gates now stand between a bad change and main,
all built the same way — rule in a mutable tested script, pin in a data file,
one `run:` line in the FROZEN workflow:

| gate | pins | added |
|---|---|---|
| `tsc_ratchet.sh` | `ci/tsc_baseline.txt` (12, TS2304 = 0) | #826 |
| `gated_tests.sh` | `ci/quarantine.txt` + `_max.txt` (1) | #832 |
| `counter_ratchet.sh` | `ci/counter_baseline.txt` (22 counters) | #833 |

**Q23 is DONE** (this session, scheduled-routine, PR #842 — awaiting after-hours
merge per the market-hours run instruction). The table's 26
`printf` lines no longer carry a second hardcoded copy of each pin — a
`declare -A PIN` loader reads `ci/counter_baseline.txt` (and `ci/tsc_baseline.txt`
for the two counters deliberately absent from the former) once, and every
BASELINE column cell now interpolates `${PIN[name]}`. This isn't a smaller gap,
it's a closed one: D10's own regex only counts a DISPLAYED baseline that is a
literal quoted digit string in this script's source, and a variable expansion
is no longer one — so `baseline_divergence` is now **0 by construction**, not
by discipline, confirmed live (`2` before this PR, `0` after, on the identical
tree modulo this diff). Also renamed 8 display labels that had drifted from
their pin names (`commented_catch`→`commented_empty_catch`,
`undeclared_py_imp`→`undeclared_py_import`, `harness_rules`→
`harness_rules_checked`, `baseline_diverge`→`baseline_divergence`,
`dup_precise_lit`→`dup_precise_literal`, `detectors`→`detectors_registered`,
`law_iv_scanned`→`law_iv_scanned_files`, `"  layers with lod"`→
`layers_with_lod`) so the printed name and the pin name are now the same
string everywhere. `ci/counter_baseline.txt`'s own `baseline_divergence` pin
lowered 2→0 in the same PR (the direct, sole effect of this change) — the
other three counters `counter_ratchet.sh` reported as "IMPROVED" this session
(`tests_run_in_ci`/`tests_gating_merge` 373→374, `assertions` 11313→11333) are
**not** re-pinned here: they are pre-existing drift from unrelated merges
since the pins were last set, not caused by this PR, and re-pinning them here
would blur attribution (PROMOTION RULE 5). Left for whichever session's change
actually produced them.

**Q24 is DONE** (this session, scheduled-routine — the ellipsoid/sphere frame
mismatch measured in #839). `orbital/geometry.ts` gained
`wgs84GeocentricRadiusKm(latDeg)` — the standard WGS-84 closed-form geocentric
radius at a geodetic latitude (reduces exactly to `EARTH_EQUATORIAL_RADIUS_KM`
at the equator, `EARTH_POLAR_RADIUS_KM` at the poles, both pinned in tests).
`groundFootprintRadiusKm()` (gained a required third `latDeg` param — its only
callers, `footprintCircle()` and the test file, both updated) and
`elevationAzimuthFromSite()` (internal-only change, no signature break — its
one production caller, `siteQuery.ts`'s Starlink coverage feature, is fixed
for free) now add `altKm` to this instead of the flat constant
`EARTH_MEAN_RADIUS_KM`, which removes the dominant (radius) term of the
0.02-0.2% error #839 measured. `EARTH_MEAN_RADIUS_KM` itself is UNCHANGED
(6371, still exported — the fix was never "change the constant", per #839's
own explicit instruction) and simply is no longer what this module adds
`altKm` to.

BYPRODUCT (not scope creep — needed to avoid ADDING a new dup_precise_literal
while fixing this): `wgs84GeocentricRadiusKm()` needs both WGS-84 a and b.
`satDerived.ts` already exported `a` (`EARTH_EQUATORIAL_RADIUS_KM`); gained a
new paired `EARTH_POLAR_RADIUS_KM` export for `b`. `propagate.ts`'s
`eciToGeodetic()` — which D11 had ALREADY flagged as carrying its own
unexported bare `const a = 6378.137` copy (#839's own note: "found ...
6378.137 in propagate.ts as a bare `const a`") — now imports both from
`satDerived.ts` instead of restating either literal. `dup_precise_literal`
4 → **3**, the direct, sole effect of this sub-change (re-pinned in the same
PR, same discipline as baseline_divergence in Q23).

**Q7 is DONE (T2.1 half)** (this session, scheduled-routine — starved since
#839, ~15 consecutive sessions per experiments.md's own STARVED lines).
`test_law_iv_context_modules.py` (new, root) widens the ACTUAL Law IV audit
predicate: `*Layer.ts` UNION any file under the layer dirs exporting a
top-level `mount<Name>()` — the non-MapLibre owned-canvas lifecycle pattern
celestialSky.ts/spaceFrame.ts use in place of a CustomLayerInterface class.
Still a structural naming rule, not a hardcoded list (test asserts the
predicate does NOT also catch lunarSymbols.ts/spaceAssets.ts/moonTiles.ts —
one-shot offscreen texture bakers that call `getContext()` but own no
persistent surface and export no `mount*`). NOT folded into
`test_audit_critical.py`'s existing REQUIRED test in place — that would turn
CI red for a gap this session is documenting, not fully closing. Instead:
**celestialSky.ts is now FULLY FIXED** — gained real `maxFeatures`
(`SKY_BODIES.length` = 9, a fixed catalog, never data-driven) and
`vramBudget` (1MB; derived from its actual two buffers, ~20KB, not chosen to
look reasonable — see the file's own new comment). **spaceFrame.ts remains
the one honest known gap** (4,300+ LOC, half a dozen data-dependent CPU
raster canvases — a real budget number needs its own session reading the
whole module, not an invented one) — the new test is pinned to exactly that
2-line gap (`maxFeatures`/`vramBudget` missing, `dispose()` already present
and passing), not to `[]`, so it starts GREEN today (no quarantine slot
needed) and fails the build the moment anything regresses further or a new
context-acquiring module ships without budgets. `scripts/program_status.sh`'s
`law_iv_scanned_files` counter updated to mirror the same widened predicate
(was counting bare `*Layer.ts` only) — 5 → **7**, re-pinned.

**Q7 is now DONE** (this session, scheduled-routine — the budget-derivation
session this queue asked for). `spaceFrame.ts` gained real `maxFeatures`
(9096, the Yale BSC5 star count) and `vramBudget` (~460 MB, summed from the
module's own owned allocations: the spaceAssets.ts texture manager it creates
and disposes, the two moonTiles.ts tile-manager instances it can hold
resident at once, the shared sprite cache, moon-patch buffers, and the sky/
star canvases — full byte-math in the export's own comment).
`test_law_iv_context_modules.py`'s `expected` list narrowed to `[]` per its
own standing instruction; all 3 of its tests pass. Two-thirds of the total
(302 MB) is the tile managers' worst-case LRU ceiling, which `TILE_CACHE_MAX`
's own comment already calls "generous headroom" over real usage — filed as
its own open_questions.md follow-up (is that ceiling too loose against the
256 MB mobile band?) rather than tightened in this PR.

**Q8 is DONE** (this session, scheduled-routine, 2026-08-23). See the QUEUE
table row below for the full account — a new `flightTrackLayer.test.ts` test
models F16's realistic multi-sample SIGNAL-LOST-AIRBORNE gap (the existing
test only covered a single missing point) and pins hypothesis (a)'s
segment-skip-guard contract for it. This closes out the code side of the F16
workstream that has sat open since 2026-08-12; the live discriminating test
(rotate the globe, watch whether the cut-off moves with the camera) still
needs a human at the map to actually decide (a) vs (b) — not something this
session could run.

**Q9 is DONE** (this session, scheduled-routine — see the QUEUE table row for
the full account: it turned out to already be CI-gated since #833; what
this session actually shipped was correcting the detector's own
mismatched-claim comment, extracting it to `scripts/design_token_drift.py`,
and adding `test_design_token_drift.py`).

**Q11 is DONE** (this session, scheduled-routine, v1.0.800). Re-measured live
before writing anything: the registry has grown to 249 layers since this row
was written (not 238); 9 carry `renderKind`, only `orbital_sats` carries
`lod`, so 248 of 249 lack at least one of the two (the "will fail on 237 of
238" framing is now 248/249 — same shape, updated number). Requiring both
fields on every layer today, the way the file's other REQUIRED tests already
do line-by-line, would mean either (a) backfilling metadata for 248
pre-existing layers in one PR — the exact "big-bang rewrite" A2 forbids
("MIGRATION IS OPPORTUNISTIC ... never a big-bang rewrite ... every layer
TOUCHED afterward migrates in that PR") — or (b) a per-layer assertion that
fails immediately on the first offending layer, which cannot report "N of
M failing" as a single number at all (`assert.ok` inside a loop throws on
the first miss). Built instead: a new REQUIRED test in
`server/layersRegistry.test.ts` (not a separate/quarantined file — it
PASSES today, so it does not need the quarantine mechanism Q12 needed for a
test that genuinely couldn't pass) that computes `missing = layers lacking
renderKind or lod` and pins the exact count (248) with `assert.equal` — a
number that moves in EITHER direction now fails the build, forcing the PR
that migrates a layer (or regresses one) to consciously update the pin, the
same discipline `dup_precise_literal`/`conflicting_const` already use.
A/B-verified live this session (not assumed): temporarily set the pin to
247 and confirmed the test fails with a clear message naming the mismatch
and the first offending layer IDs; restored to 248, green again. This
makes "required" real in the sense Track 4 asked for — a hard, CI-checked
number instead of the previously fully-optional "when present" tests above
it — without a false claim that 248 layers were migrated this session (they
were not; this is instrumentation, not backfill). Full account in
experiments.md.

**NEXT**, then Track 2/3 — the moon.

Prior Q22 (DONE, now merged). A diagnostic
probe plugin (patched `yfinance.Ticker.history` to log the current pytest
node id on every real call, run once before and once after the fix) found the
true call sites — none were "at import time": 4 gated test files reach
`macro_data.get_macro_snapshot()` indirectly WITHOUT mocking `yfinance`
themselves (`test_deep_score_credit_spread_cache.py`,
`test_gridvision_pod_run.py`, `test_tiered_strategy.py`,
`test_voltrade_daemon.py`). Fixed with one session-scoped autouse fixture in
`conftest.py` defaulting `yfinance.Ticker` to an empty history for the whole
suite — reproducing the exact same "yfinance failed → macro_data's documented
default" code path every one of those tests already exercised via a live,
always-failing network call, hermetically instead of over the network. Local
per-test mocks (`test_macro_snapshot_spy_dedup.py`) shadow it safely — verified
by inspection of `unittest.mock.patch.stopall()` semantics (only unwinds
patches started via `.start()`; the fixture uses a `with` block) and by running
that file alongside the 4 previously-offending files together: 39/39 pass.
Same 1348 passed/1 skipped both before and after; wall time 117.99s → 32.94s
(network-timeout latency removed, not a correctness change). See
experiments.md for the full probe transcript.

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
| Q6 | T2.4 — cap DPR in `celestialSky` + `spaceFrame` | T2.4 | **DONE** — PR #831. Bounds memory + fill rate. **NOT a moon speedup** — the audit's 9× claim is false, see L16 |
| Q7 | T2.1/T2.2 — widen the Law IV predicate to context-acquiring modules | T2.1 | **DONE** — this session (scheduled-routine). Predicate widened 5 → 7 in `test_law_iv_context_modules.py`; both celestialSky.ts and spaceFrame.ts now carry real `maxFeatures`/`vramBudget`. `expected` narrowed to `[]`; not yet folded into test_audit_critical.py's REQUIRED `_layer_modules()` predicate (separable structural edit, left as-is) |
| Q8 | T2.6 — the §2.1 F16 NaN-guard unit test | T2.6 | **DONE** — scheduled-routine session, 2026-08-23. New test in `flightTrackLayer.test.ts` models a multi-sample SIGNAL-LOST-AIRBORNE gap (not just the single-point gap the existing test covered); pins F16 hypothesis (a)'s segment-skip-guard contract for that realistic shape. Does NOT itself resolve F16 (a) vs (b) — that still needs the live human "rotate the globe" discriminating test the doc names |
| Q9 | T8.1 — design-token drift check into the harness | T8.1 | **DONE** — this session (scheduled-routine). It was already CI-gated: `design_token_drift` has lived in `ci/counter_baseline.txt` since that file's creation and `counter_ratchet.sh` (Q20) ratchets every line in it generically, so this queue row was stale the moment #833 shipped — nobody had re-checked it. What this session actually found and fixed: the detector's own comment claimed "a mismatch in either direction is drift" but the code only ever walked `md.items()` — a one-directional check, and correctly so (DESIGN.md documents 15 curated author-facing tokens; the other 72 real `:root` custom properties in `client/src/index.css` are shadcn/ui internal plumbing, verified by reading both files directly, not assumed). Extracted the inline heredoc to `scripts/design_token_drift.py` (gate2_stats.py precedent) with the corrected docstring, and added `test_design_token_drift.py` — a live-pair regression pin (0, matching the baseline) plus three synthetic cases that lock in the intentional one-directional semantics so a future "fix" can't quietly turn 72 healthy tokens into false positives. Counter value unchanged (0 before/after on the identical live DESIGN.md/index.css pair) |
| Q10 | T1.1 — all three suites into CI non-blocking | T1.1 | **DONE** — PR #829. 368/368 files now RUN in CI; 4/368 gate |
| Q17 | T1.2/T1.3 — quarantine file + pin, green set BLOCKING, quarantine may only shrink and no entry may age past 30d | T1.2 | **DONE** — PR #832. `tests_gating_merge` 4 → **367/368** |
| Q20 | T1.7 — wire the §4.2 counters into CI as ratchets | T1.7 | **DONE** — PR #833. 22 counters now fail the build on a wrong-direction move |
| Q11 | T4.1 — `renderKind` + `lod` required in `layersRegistry.test.ts` | T4.1 | **DONE** — this session (scheduled-routine), v1.0.800. Registry now 249 layers, 248 missing at least one field; pinned as a hard `assert.equal` ratchet (not a per-layer requirement — that would be the big-bang rewrite A2 forbids), A/B-verified to actually fail on a wrong pin |
| Q12 | ≥50 state+national power pmtiles asserted, 3 exist | T1.2 | **REFRAMED** — PR #834. Split into `gridTilesCoverage.test.ts` (quarantined, review 2026-09-13). Cannot be resolved by committing tiles: `build_power_tiles.sh:53` forbids it at US scale. Real work = the boot-fetch path (A4 PHASE 2 item 2) |
| Q21 | The magic-byte guard in `gridTiles.test.ts` had been DEAD since it was written — the `>=50` assertion ran first and prevented it | T1.2 | **DONE** — PR #834. Split; the guard now passes and gates |
| Q14 | `EARTH_RADIUS_KM` 6371 vs 6378.137 (both in `client/src/lib/orbital/`) and `EARTH_CIRCUMFERENCE_M` 2πR vs 40075016.686 | T2/orbital | **DONE** — PR #839. All FOUR values are CORRECT where they live; the collision was the defect, so renamed not unified. `conflicting_const` 5 → **3**. The "~7km altitude error" framing was wrong — the constant cancels; measured 0.02–0.20%. See L21 |
| Q24 | `propagate.ts` emits geodetic height above the WGS-84 ELLIPSOID; `orbital/geometry.ts` adds it to a 6371 SPHERE | T2/orbital | **DONE** — this session (scheduled-routine). New `wgs84GeocentricRadiusKm(latDeg)`; `groundFootprintRadiusKm`/`elevationAzimuthFromSite` add `altKm` to it instead of the flat `EARTH_MEAN_RADIUS_KM` (which is unchanged, per the instruction not to "fix" this by changing that constant). Byproduct: `dup_precise_literal` 4 → 3 (propagate.ts's own unexported literal copy of `EARTH_EQUATORIAL_RADIUS_KM`, already flagged in #839, now imports instead of restating) |
| Q25 | the visual harness's perf gate is NON-DETERMINISTIC — its thresholds sit inside its own noise band | T-CLIENT | **DONE** — this session (scheduled-routine, v1.0.807). `scripts/visual_check.mjs`'s perf sampler now runs THREE independent measured passes per viewport (after one shared warm-up) and gates on the MEDIAN of the three per-pass medians/p95s, per this row's own instruction ("best-of-N... do NOT simply raise the numbers" — thresholds unchanged: `MEDIAN_GATE`/`350` p95 untouched). Failure messages now include the raw per-pass array so a future noisy-run investigation doesn't need to re-run to see the spread. Verified live on this sandbox: 2 pre-fix runs (single-pass) read 768/1440 medians+p95 of 133/183, 133/233 then 117/167, 133/200 — real run-to-run movement even here; post-fix run's raw per-pass data (`.visual/results.json`) shows the mechanism doing its job, e.g. 768px passes `[117,100,117]` median/`[167,183,167]` p95 collapsing to 117/167 instead of whichever single pass happened to land — one pass no longer decides the verdict. 0 hard failures at 390/768/1440 |
| Q15 | `server/datacoreArchive.test.ts` rollup tests fail near UTC midnight | T1.2 | **DONE** — PR #830. Fixed, not quarantined: it was a bug in the test's date arithmetic, never in the code under test |
| Q18 | `VOLTRADE_CI` set in 2 ci.yml jobs, read by nothing | T1.2 | **DONE** — PR #836. Removed; the comment now names the REAL mechanism (`conftest.py` `collect_ignore`). `dead_workflow_env` 1 → **0** |
| Q22 | `macro_data.py` makes live yfinance calls (DX-Y.NYB, ^TNX, ^VIX) reachable from 4 gated test files that don't mock it — the suite is not hermetic | T1.2 | **DONE** — this session (scheduled-routine). Session-scoped autouse fixture in `conftest.py` defaults `yfinance.Ticker` to empty history repo-wide; the 4 real offending files (found by a diagnostic probe, not by the "at import time" guess) needed no per-file changes. 1348/1348 unchanged, 117.99s → 32.94s |
| Q16 | CI never installed `requirements-dev.txt`, so `test_grid_county_ba.py` could not import openpyxl and the COLLECTION error aborted the whole python suite (1337 passes → `1 skipped, 1 error`) | T1.1 | **DONE** — PR #829. Not fixed by moving openpyxl into requirements.txt: that file feeds the frozen Dockerfile's production image |
| Q19 | 3 uncapped render surfaces (5 sites) | T2 | **DONE** — PR #837. `uncapped_surface` 3 → **0**. My "small login canvas" caveat was BACKWARDS — login's is a full-viewport animated canvas with its own rAF loop, the largest of the three |
| Q13 | `empty_ts_catch` / `ts_any` count comment text — strip comments and string literals before counting | T0.1 | **DONE** — PR #838. 495 → **494**, 1251 → **1237**; all 15 excluded sites named. The directive's "strip comments" was WRONG for `empty_ts_catch`: stripping sends it UP to 516 by merging it into D4. Rule is exclude-by-LOCATION, scan per line (L20) |
| Q23 | `program_status.sh`'s printed baseline column is a second copy of `ci/counter_baseline.txt` and has already drifted on 3 counters | T0.1 | **DONE** — this session (scheduled-routine). `declare -A PIN` loader reads `ci/counter_baseline.txt`/`ci/tsc_baseline.txt`; every BASELINE cell now interpolates the pin. `baseline_divergence` 2 → **0**, structurally (D10's regex can no longer find a literal to compare). 8 display labels renamed to match their pin names |

**The queue is not empty.** §0.3 condition 5 satisfied.

---

## NUMBERS

`scripts/program_status.sh --no-tsc`, run 2026-08-15 at commit `e90a524`
(pre-this-session), `package.json` 1.0.723 (this session). `--no-tsc` because
this sandbox has no `node_modules`/`npx tsc` until `npm ci` runs (done this
session for the test/build gates below); `tsc_errors`/`tsc_2304` are carried
forward unmeasured here from the last on-CI run (`ci/tsc_baseline.txt` TOTAL,
unchanged by this session's diff — confirmed separately via
`bash scripts/tsc_ratchet.sh`, which DOES run tsc: 12, TS2304 = 0, unchanged).

```
COUNTER                  VALUE          BASELINE     DIRECTION
tests_run_in_ci          374/375        374          must increase
tests_gating_merge       374/375        374          must increase (>216)
tsc_errors               (not run)      12           must decrease
  of which TS2304        (not run)      0            AT TARGET — hold at 0
silent_py_handlers       255/874        255          non-increasing
bare_except              3              3            non-increasing
empty_ts_catch           494            494          non-increasing (ruler fixed, Q13)
ts_any                   1237           1237         non-increasing (ruler fixed, Q13)
boundary_any             233            233          non-increasing
commented_empty_catch    113            113          non-increasing (disjoint from empty_ts_catch — L20)
conflicting_const        3              3            non-increasing
undeclared_py_import     2              2            non-increasing
dead_workflow_env        0              0            AT TARGET — hold at 0
uncapped_surface         0              0            AT TARGET — hold at 0
assertions               11354          11354        NON-DECREASING — this session +41 (Q24's tests)
layers_full_schema       1/238          1/238        non-decreasing
layers_with_lod          1              1            non-decreasing
law_iv_scanned_files     5              5            must reach 7 (ctx-acquiring)
order_post_sites         6              6            must reach 1
design_token_drift       0              0            must stay 0
harness_rules_checked    71             71           non-decreasing
baseline_divergence      0              0            must reach 0 — structural since Q23
dup_precise_literal      3              3            non-increasing — this session 4→3 (Q24 byproduct)
orphaned_set_interval    0              0            must stay 0 — NEW this session, D12
detectors_registered     12             12           MUST increase each session — D12 added this session
quarantine_size          1              1            non-increasing (gridTiles/Q12)
quarantine_oldest        1d             0d           fail if >30
```

`tests_run_in_ci`/`tests_gating_merge` (373→374 vs the pin) moved since the
pins were last set, from merges unrelated to this session's diff — not
re-pinned here per PROMOTION RULE 5 (attribution dies when unrelated
improvements are bundled into one PR's pin update). `assertions`
(11313→11354), `dup_precise_literal` (4→3), and `detectors_registered`
(11→12) ARE re-pinned in this PR: all three are this session's own direct
effect (Q24's new tests, Q24's byproduct literal-import cleanup, and D12).

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

**L21 — "two constants disagree" is not the same finding as "one of them is
wrong", and the queue entry conflated them.** Q14 was filed as "~7km of
satellite altitude error" and "Web Mercator REQUIRES the latter". Checked
against ground truth, all FOUR values are correct where they live:

| constant | value | correct because |
|---|---|---|
| `geometry.ts` | 6371 | mean radius, and the module is an explicit spherical-cap model |
| `satDerived.ts` | 6378.137 | WGS-84 equatorial = SGP4's reference radius; apsides are quoted above it |
| `glElev.ts` | 2π×6371008.8 | MapLibre's OWN `earthRadius` — verified at `maplibre-gl-dev.js:36206` and against its live `MercatorCoordinate.fromLngLat(_,1).z` |
| `lod.ts` | 40075016.686 | EPSG:3857 is defined on the equatorial radius |

Unifying either pair would have broken whichever side lost — for `glElev.ts`,
by 0.112% on every projected altitude, silently. **The defect was the shared
NAME**: a physical constant's name reads as a claim about the world, so
`EARTH_RADIUS_KM` imported from the wrong file is undetectable at the call
site. Renaming fixes it and changes no value — proven by A/B'ing the emitted
GLSL, byte-identical at `2.4981121215e-8`.

**The "7km" number was never realized as 7km of error.** In
`groundFootprintRadiusKm` the radius appears in both numerator and denominator,
so it largely cancels: measured 0.02%/0.06% for LEO (550km, 25°/0° masks) and
0.10%/0.20% at GEO, equator vs pole. There IS a real frame mismatch underneath
(propagate emits geodetic-above-ellipsoid, geometry consumes it as
above-sphere) and it is now Q24 — but it is a 0.2% latitude-dependent effect,
not a 7km one. Third queue entry this session whose premise did not survive
reading the code (Q12, Q18, the Q13 directive), which is itself the pattern:
**a filed hypothesis is a lead, not a finding.**

**L20 — the fix the directive asked for would have broken the counter, and
only measuring first revealed it.** Q13 said "strip comments before counting."
For `ts_any` that is right (1251 → 1237). For `empty_ts_catch` it is backwards:
stripping sends it **UP**, 495 → 516, because blanking a comment between `{`
and `}` *creates* `catch { }` — and those 21 are exactly what D4
`commented_empty_catch` counts. Doing what the queue entry said would have
silently merged two counters the program keeps disjoint on purpose, while
looking like a cleanup.

The general rule, now compiled into `scripts/ts_code_only.py`: **exclude by
LOCATION, never by re-matching cleaned text.** Match the raw source and discard
matches whose bytes turn out to be prose. That operation can only subtract, so
counters cannot cross into each other.

Second half, learned the same way: **scan per line.** A file-wide lexer looked
more correct and was more dangerous — the regex literal `/'/g` at
`server/billing.ts:83` opened a quote that never closed, and it declared the
next 30 lines non-code, silently excluding four real `catch (err: any)`
annotations. Telling a regex literal from division needs a parser; bounding the
blast radius to one line does not. Both failures were in the FLATTERING
direction, which MEASUREMENT INTEGRITY says to distrust by default — and both
were invisible until the excluded sites were printed and read one by one.

**L19 — my own scoping caveat was backwards, and reading the files was what
caught it.** Filing Q19 I wrote "each needs its own look — a small login canvas
is not the same call as a full-screen map", implying `login.tsx` was the weak
case. It is the STRONGEST: `CityMatrixCanvas` sizes to
`window.innerWidth × window.innerHeight` and animates under its own rAF loop, so
an uncapped 3× device paints 9× the pixels every frame. `bot.tsx`'s equity chart
and `DataWorldMap`'s offscreen land layer are the element-sized ones. A guess
about relative severity, written into a queue entry, would have been inherited
as fact by whoever took the item — the entry now records the correction.

Also: naming the three files individually would have missed the fourth. The
companion assertion added here closes the whole class — no client module may
size a canvas from raw `devicePixelRatio` — with exactly two allowlisted
readers (`deviceTier.ts`, where the clamp lives, and `datamap.tsx`, which
PRODUCES the tier reading and cannot call the helper without circularity). The
allowlist is itself guarded: a second test asserts `datamap.tsx` still calls
`classifyDevice`, still clamps via `Math.min(dpr, tier.pixelRatioCap)`, and
still publishes `__vtDeviceTier` — an allowlist entry that stops honouring the
rule is worse than no rule, because it looks covered.

**L18 — the claim was true; the mechanism named was fiction.** `ci.yml` said
"Network-dependent tests are excluded in CI" beside `VOLTRADE_CI: "1"`. D7 found
the variable had zero readers, and the obvious readings were both wrong:
- NOT "the claim is false" — the genuinely un-runnable scripts
  (`test_full_system.py`, `test_auto_discovery.py`) really ARE excluded;
- NOT "wire the variable up" — nothing needed wiring, because
  `conftest.py`'s `collect_ignore` already does the job and documents why.

The defect was **attribution**: a real behaviour credited to a control that did
not exist. Fixed by deleting the variable and pointing the comment at
`conftest.py`. Generalises: when a detector says a named mechanism is dead,
check whether the BEHAVIOUR is dead too — often the behaviour is fine and only
the label is wrong, and deleting the behaviour would be the expensive mistake.

Also added the caveat the old note omitted: the suite is NOT hermetic
(`macro_data.py` makes live yfinance calls when imported by tests). Filed as
Q22 rather than papered over — the old comment's confidence is exactly what
kept that unexamined.

**L17 — a failing assertion had killed the guard behind it, and the guard was
the whole point of the file.** `server/gridTiles.test.ts` exists to catch a
REAL shipped defect: v1.0.251, where `power_us.pmtiles` was written as SQLite
and the `pmtiles://` protocol silently rendered nothing (logged in
experiments.md). Its first line asserted `files.length >= 50`. That assertion
has never once been true — so **the magic-byte loop underneath it never ran**,
and the regression guard was dead from the day it was written.

Two general lessons, both cheap to apply:
1. **A cheap precondition placed above an expensive guard can silently disable
   it.** Order matters inside a test, not just between tests. When a test has a
   setup-shaped assertion and a substance-shaped one, they want separate tests
   — otherwise the first failure hides everything after it, exactly as a long
   `try` block hides everything after a throw (L4). Same defect, different
   scale.
2. **A red test is worth reading before it is worth fixing.** The obvious fix
   here — build the tiles — is FORBIDDEN by the repo's own build script
   (`build_power_tiles.sh:53`: "US scale: DO NOT commit — boot-fetch from a
   GitHub Release asset") and by its ODbL share-alike note. The assertion was
   demanding the one location the architecture says these files must never
   occupy. Building 12GB of tiles to satisfy it would have been days of work in
   the wrong direction.

Splitting deleted nothing and weakened nothing — `assertions` went UP 11268 →
11269, which is the objective check on that claim.

**L16 — the MASTER PROGRAM's "up to 9× faster moon" claim is FALSE, and the
Day One table ranks T2.4 on it.** F-C observes correctly that `celestialSky.ts`
and `spaceFrame.ts` size their backing stores from raw `devicePixelRatio` while
`datamap.tsx` clamps to `tier.pixelRatioCap`. F-D then multiplies the two:
*"the raycast runs over the uncapped backing store, so a 3× device does 9× the
raycasts."* **That step does not hold.** Traced this session:

- `spaceFrame.ts` sets `ctx.setTransform(dpr, 0, 0, dpr, 0, 0)`, so every
  drawing coordinate in that file is in **CSS pixels**.
- `drawBodyPatch(..., w, h, ...)` is called with `w, h` from `cssSize()`
  (`canvas.clientWidth`) — CSS pixels again.
- The moon bbox `bw, bh` is derived from those, and
  `patchBufDims(bw, bh, longPx)` clamps the raycast buffer's long side to
  `MOON_PATCH_FULL_LONG_PX = 1100` (settled), `MOON_PATCH_MOVING_LONG_PX = 480`
  (moving), `MOON_PATCH_FAST_LONG_PX = 116` (bootstrap).

So the raycast buffer is bounded by those CONSTANTS, in CSS pixels, and is
**independent of `devicePixelRatio`**. Capping DPR changes it by nothing.

What capping DPR genuinely buys: backing-store MEMORY and the FILL RATE of
every raster op in those files, both quadratic in the ratio. Worth doing, and
done (#831) — but the honest claim is "bounded memory and fill rate on
high-DPR devices", not a moon speedup.

**The moon's actual cost** is a per-pixel CPU ray-sphere intersection
(`renderMoonSurfaceRows`) over up to 1100×1100 ≈ **1.2M rays per settled
patch, on the main thread**, which is why it renders in row bands across
frames. That is Track 3's GPU work (D1 already decided it), and nothing short
of it will make the moon fast. **Do not expect T2.4 to have moved it.**

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
| D8 | `uncapped_surface` — a module acquiring a canvas/WebGL context that reads `devicePixelRatio` without clamping to the device tier | 2026-08-14 | 3 | live in `program_status.sh`; found Q19 on its first run |
| D9 | `assertions` — total assert statements across every test file. **NON-DECREASING** — the only counter that must go up | 2026-08-14 | 11228 | live in `program_status.sh`; enforces CLAUDE.md's "never delete or weaken an existing assertion", which nothing counted before |
| D10 | `baseline_divergence` — this script's PRINTED baseline column disagreeing with the pin CI actually enforces in `ci/counter_baseline.txt` | 2026-08-14 | 5 | live in `program_status.sh`; found `ts_any` 1252-vs-1251 while re-pinning Q13, then 4 more. Down to **2** in the same PR. The last two (`dead_workflow_env`, `uncapped_surface`) are left DELIBERATELY: hand-patching them would zero the counter while the mechanism that lets a second copy drift survives — Q23 removes the copy |
| D11 | `dup_precise_literal` — a high-precision numeric literal (≥7 significant digits, trailing zeros not counted) restated in 2+ modules; counts the redundant COPIES so it falls when one is deleted | 2026-08-14 | 5 | live in `program_status.sh`; the mechanism BEHIND D5 — `6371008.8` was written longhand in 3 modules before anything collided. Found `40075016.686` also in `cameraRig.ts` and `6378.137` in `propagate.ts`, neither of which D5 can see (not exported names). 5 → **4**, then **3** this session (Q24 fixed the `propagate.ts` copy it found) |
| D12 | `orphaned_set_interval` — a `client/src` file calling `setInterval()` with no `clearInterval()` anywhere in it (whole-file, not per-call-site pairing — real scope analysis would need an AST, not a grep) | 2026-08-15 | 0 | live in `program_status.sh`; A/B-verified live (0→1 on an induced probe file, reverted). Baseline 0: every current caller already pairs the two. Seeded by F13 (PROGRAM_STATE.md above) — a `setInterval` with no visible off-switch is the same "mechanism with no visible off-switch" shape D7/D8 exist for |

**Seeds not yet taken** (MASTER PROGRAM §0.7, plus new ones from this session):

- `useEffect` with a dependency array omitting a ref it reads
- registry ids in `layers.json` with no server route; routes with no id —
  **investigated 2026-08-15, correctly NOT built**: layer `id`s in
  `datacore/layers.json` do not follow a consistent `/api/data/<id>` naming
  convention (raw client-only overlays, dedicated tile routes, etc.), so a
  naive id-vs-route-string comparison would false-positive on most of the
  238 layers. Needs real semantic mapping, not a grep — left for a session
  with time to build that mapping properly rather than ship a noisy detector
- `any` at a module boundary (params and return types only)
- theme-token literals hardcoded instead of referenced (D11) — *partly covered
  by `design_token_drift`; the off-palette-hex half is not yet built*
- functions taking a parameter that shadows an outer binding of the same name
  (the inverse of D1 — would catch the `focusSat` extraction *before* the
  binding is lost)

---

## SESSION LOG

### 2026-08-15 — scheduled-routine session. Territory: T-CLIENT (client/src/lib/orbital/**, satDerived.ts) + shared (scripts/program_status.sh, ci/counter_baseline.txt, package.json, research/*), last and minimal

Took the queue's own stated NEXT item, **Q24**, unclaimed since #839
(2026-08-14). Full account in `research/experiments.md`'s tagged entry for
this session; summary here:

- `orbital/geometry.ts` gained `wgs84GeocentricRadiusKm(latDeg)` (WGS-84
  closed-form geocentric radius at a geodetic latitude). `groundFootprintRadiusKm`
  and `elevationAzimuthFromSite` now add `altKm` to this instead of the flat
  `EARTH_MEAN_RADIUS_KM` — removes the dominant term of the 0.02-0.2% error
  #839 measured, without touching `EARTH_MEAN_RADIUS_KM` itself (kept exactly
  as instructed) or any angular formula.
- Byproduct: implementing the fix without adding a NEW `dup_precise_literal`
  instance meant giving WGS-84 a/b one canonical home (`satDerived.ts`,
  paired `EARTH_EQUATORIAL_RADIUS_KM`/new `EARTH_POLAR_RADIUS_KM`) and
  pointing `propagate.ts`'s `eciToGeodetic()` at it instead of its own bare
  literal — which also closed the `propagate.ts` copy D11 had already found
  and left unfixed. `dup_precise_literal` 4 → 3.
- Detector added: **D12** `orphaned_set_interval`. A/B-verified live on an
  induced probe file (0→1, reverted cleanly). Considered and explicitly
  declined the layers.json↔routes.ts seed (would false-positive on most of
  238 layers — no consistent naming convention to grep against; needs real
  semantic mapping, filed as a note instead of shipped as a noisy detector).
- GATES: `bash scripts/gated_tests.sh` GATE PASSED (client 1017/1017, python
  1354/1 after `pip install -r requirements.txt -r requirements-dev.txt`,
  quarantine 1/1 none overdue) · `bash scripts/tsc_ratchet.sh` 12, TS2304 0 ·
  `bash scripts/counter_ratchet.sh` 25 counters OK (re-pinned `assertions`
  11313→11354, `dup_precise_literal` 4→3, `detectors_registered` 11→12 — all
  three this session's own direct effect; left `tests_run_in_ci`/
  `tests_gating_merge` 373-pinned/374-live, pre-existing drift per PROMOTION
  RULE 5) · `npm run build` clean · `npm run visual` at 390/768/1440 (see
  experiments.md for the A/B — this code has ZERO current rendering callers,
  confirmed by grep, so no visual delta was expected or found).
- **STARVED: no** — this was the queue's own stated highest-priority item and
  its own gate closed cleanly this session.

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
