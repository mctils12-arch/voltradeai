# Experiment Log

Append-only. Newest at top. Never rewrite history (CLAUDE.md — MEMORY PROTOCOL).
Each entry: date · change · version tag · backtest result · hypothesis · (later) live-vs-backtest.

## AUDIT REGISTER (maintained in place per the AUDIT CYCLE clause,
CLAUDE.md SESSION BUDGET — this block is updatable state, the only
exception to append-only; the log below it stays append-only)

| audit | cadence | last run |
|---|---|---|
| staleness audit (code/deps/config/expired adapters — DEAD CODE POLICY governs) | 30d | never — due (2026-07-03 OpenSky sweep was targeted, not a full audit) |
| constitutional audit (rules — CONSTITUTIONAL HYGIENE governs) | 30d | 2026-07-03 (first audit; findings approved + applied 2026-07-04) |
| market_calendar year-add (FROZEN PATHS exception governs) | December | 2026 dates present; add 2027 in Dec 2026 |

## 2026-07-04 — [PRODUCT] Everything Graph v1 spec filed (flagship) + R5/R6 roadmap slots — docs PR

- Charter directive items 4-5 executed as design artifacts (no build):
  datacore/EVERYTHING_GRAPH.md specs entity types (company/person/
  facility/vessel; aircraft_operator PLANNED pending tail→operator
  gate 1), relationship types (insider_of, operates, located_at,
  calls_at from port-dwell visits), storage (v1 = pure builder +
  cache per the recompute-derivations doctrine; sqlite only past a
  stated size/latency trigger; never auth.ts's db), the entity_map
  resolution table (shared infrastructure with fusion (b) gate 1),
  and a 3-PR build order. Roadmap: R5 (graph, flagship) + R6
  (signal-strength / data-quality / pipeline-health dashboards, all
  sourced from monitoring we already emit) added to MAP V2 ROADMAP.
- Key design call, recorded: the graph is RAW (relationships as filed/
  registered, with provenance + confidence on every edge) — any
  interpretation on top is SIGNAL-class and ladder-gated; and v1 is a
  materialized view, not a store — losing it loses nothing
  (ARCHIVE-ingredients doctrine).
- Docs-only PR, no version bump.

## 2026-07-04 — [PRODUCT] Port dwell analytics from our own AIS archive (v1.0.60) — fusion directive, highest-immediate-value item

- PRIOR (REASONING STANDARD #10, stated before reading any archive
  output): with the archive only ~1 day old, expected near-zero
  completed calls and mostly in-port-now counts; the v1's value is the
  MACHINERY being correct from day one so history accrues into it —
  medians/anomalies become meaningful in 2-4 weeks. Expected design
  risks: (a) harbor craft polluting call counts (mitigated: median SOG
  <=3kts filter), (b) LA/LB fence overlap double-counting (mitigated:
  nearest-port assignment), (c) coverage gaps inflating dwell
  (mitigated: gaps SPLIT visits — lower-bound property, tested).
- Built: server/portDwell.ts (pure, baseDir-injectable, reuses
  shadowFleet.readVesselTracks); /api/data/portdwell (10-min cache);
  geofences = the 9 imagery-verified port terminals from
  datacore/sites (REFERENCE DATA ACCURACY: only verified coordinates
  may become geofences — these are the only port coordinates in the
  repo that qualify). UI: per-port text labels under port sites +
  panel row with per-port note ("X in port · med Yh"), RAW-labeled,
  filings-&-flows group. Anomaly flags 3x-median, suppressed <10
  completed calls/port.
- Tests: 7 new node tests (visit detection, right-censoring,
  nearest-port overlap, speed filter, anomaly threshold + thin-history
  suppression, gap-splits-visit lower-bound, wiring pins). Gates: node
  56/56, python 114 passed / 1 skipped, harness green 390/768/1440
  with self-see. NEW-LAYER RENDER PROBE (MapLibre silent-expression
  lesson): standalone playwright probe jumped to San Pedro Bay at z9
  and asserted queryRenderedFeatures on portdwell-labels — rendered=1
  with correct label text ("Los Angeles / 4 in port · med 22.5h");
  second label collision-hidden as designed (text-allow-overlap off).
- Downstream chain (#1): dwell stats surface -> port calls accumulate
  into distributions -> gate-2 test plan (open_questions.md PORT DWELL)
  becomes runnable in weeks; zero trading-path impact (datacore
  boundary: no bot_engine/system_config imports; worst failure mode is
  an empty stats endpoint).
- Backtest: not applicable (no trading logic changed); version bumped
  1.0.59 -> 1.0.60 for attribution.

## 2026-07-04 — [PRODUCT] VISION.md installed — platform charter (reconstruction), north-star reading rule

- Charter-installation directive executed. HONESTY NOTE: the human's
  verbatim charter text did not arrive (paste placeholder came through
  unfilled; confirmed absent from the session transcript) — VISION.md
  is a clearly-labeled session reconstruction from the directive's own
  enumeration, with a provenance banner asking the human to supply the
  original for verbatim replacement. Installing a labeled
  reconstruction now beats waiting: the north star exists for tonight's
  routines.
- Reconciliation annotations: every charter item marked
  DONE/IN-PROGRESS/QUEUED/NEW/BLOCKED-BY-ACCESS citing the existing
  mechanism (archive doctrine, ladder, DESIGN.md rules, roadmap slots).
  BLOCKED-BY-ACCESS register: card panels, private fleet telemetry,
  sub-meter counting, mid-ocean satellite AIS (declined), filed flight
  plans, US freight-rail positions.
- Reading rule: STANDING BEHAVIORS gains the VISION.md north-star line
  (approved 2026-07-04); KNOWN STATE carries the existence fact.
  Placement reasoning recorded in the rule text itself (directive said
  KNOWN STATE; same message approved the facts-vs-rules split — rule
  goes to STANDING BEHAVIORS, fact to KNOWN STATE).
- Routine prompts: B4/B3 canonical texts exist only in the routine
  platform — usage_log.md now carries the exact north-star line to
  append, flagged HUMAN ACTION NEEDED.
- Docs-only PR, no version bump.

## 2026-07-04 — [RULE-REVIEW] Approved consolidations applied: AUDIT CYCLE + STANDING BEHAVIORS (docs PR)

- Human approvals received 2026-07-04 for: (1) the AUDIT CYCLE
  consolidation proposal (wishlist.md, filed 2026-07-03) and (2) first
  constitutional audit Findings 1 (STANDING BEHAVIORS section) and 2
  (PERIODIC AUDITS register). Applied exactly as proposed: AUDIT CYCLE
  clause appended to SESSION BUDGET; this register created; DEAD CODE
  POLICY and CONSTITUTIONAL HYGIENE trigger sentences trimmed to point
  here (policy bodies untouched); KNOWN STATE's five standing-behavior
  rule paragraphs moved verbatim into a new STANDING BEHAVIORS section.
- F2 note: the AUDIT CYCLE register IS Finding 2's register in the
  concrete form the AUDIT CYCLE proposal specified ("supersedes the
  first audit's Finding-2 sketch") — one register, in experiments.md,
  not a second one in SESSION BUDGET. Shipping both literally would have
  created the exact redundancy the hygiene rule exists to kill.
- Docs-only PR: no behavior change, no version bump (per docs-PR
  precedent #132/#133), no backtest applicable.

## 2026-07-03 — [PIPELINE] Position archive — MAP V2 ROADMAP R1 (PR 2/3 of the Map v2 directive)

- Session start check: read CLAUDE.md, all of research/, per the [PRODUCT]
  session protocol. Loop-health ratio over the last 8 experiments.md
  entries: 4 REPAIR, 1 RESEARCH, 3 PRODUCT/PIPELINE — below the 7/10
  escalation threshold, no concern. KNOWN BROKEN #3/#4/#5 remain
  un-diagnosed (owner-gated diagnostics access, per wishlist) but per this
  session's brief, product work is not preempted by outstanding repairs
  that don't block it; none of them touch datacore/ or the /data surface.
- Chose this action over the other three options in the brief (new
  UI/enrichment work, a new-root proposal, or API/docs hardening) because
  the prior session's Map v2 directive (PR #105, docs-only "PR 1 of 3")
  explicitly named the position archive as the most time-sensitive R1 item
  — "every day not recorded is unrecoverable proprietary data" — and it is
  the load-bearing prerequisite for R2 (maritime transit analytics, "the
  strongest trading-signal candidate here") and three of the four
  ARCHIVE-ENABLED SIGNAL HYPOTHESES in open_questions.md. Every day this
  stays unbuilt is lost history those hypotheses can never recover.
- PRIOR (before writing any code, REASONING STANDARD #10): expected the
  main design risk to be disk growth on the Railway volume, since aircraft
  responses are capped at 800 records and vessels at 1500 and the /data
  page can poll every 10-30s — recording every poll would be tens of MB/day
  from one bbox alone. Expected the fix to be a time-gated sample interval
  decoupled from poll rate, landing in the 15-100MB/mo combined range
  depending on the interval chosen, plus a retention/rollup scheme so the
  raw-file directory doesn't grow forever.
- Design (server/dataArchive.ts, wired into server/routes.ts): aircraft and
  vessel positions are sampled independent of request/poll frequency via a
  30-min-per-kind throttle (module-level last-write timestamp), recorded as
  compact POSITIONAL ARRAYS (not objects — no repeated field names) into
  one append-only JSONL file per UTC day at
  `${DATA_DIR}/archive/{aircraft,vessels}/YYYY-MM-DD.jsonl`, rounded to
  ~11m lat/lon precision. Every sample also updates a tiny permanent
  rollup JSON (`${kind}_rollup.json`: total samples/records, per-day
  counts, first/last day) BEFORE any pruning, so raw-file retention (90
  days) never loses the count history — only the ability to replay exact
  positions that far back. New route `/api/data/archive/stats` exposes
  this for the wishlist.md volume-watch item without a shell.
  Computed at these parameters: aircraft ≈800 recs/sample × ~35B/rec ×
  48 samples/day ≈ 1.3MB/day ≈ 40MB/mo; vessels ≈1500 recs/sample ×
  ~30B/rec × 48/day ≈ 2.1MB/day ≈ 65MB/mo. Combined ≈105MB/mo — close to
  the prior session's aspirational <100MB/mo estimate; wishlist.md updated
  with the real figures and a note to revisit if live stats run hotter
  (record counts near the per-request caps more often than assumed).
- Downstream chain (REASONING STANDARD #1): recording starts today ->
  every subsequent session gains one more day of position history it can
  never otherwise recover -> R2's geofence transit counters and the
  corporate-jet / tanker-routing / destination-prediction hypotheses all
  become buildable once enough days accumulate -> none of this touches
  live trading (datacore/ boundary rule: zero imports from bot_engine.py /
  system_config.py / strategies/ / server/bot.ts) so it cannot affect
  order flow, sizing, or the kill switch even in a bug scenario worse than
  intended (worst case: excess disk usage on the Railway volume, caught by
  the new stats route and the wishlist watch item, not by any trading
  behavior change).
- Boundary discipline verified: `server/dataArchive.ts` has no imports
  from trading modules; it is pure filesystem I/O over positions already
  fetched by the existing `/api/data/aircraft` and `/api/data/vessels`
  handlers (unchanged upstream fetch/cache logic — this only adds a
  recording side-effect after the existing cache-set line in each).
- Regression tests FIRST (loop-health rule 3):
  `server/dataArchive.test.ts` — 7 new tests covering: UTC day-key
  boundaries, interval throttling (3 calls inside one window write only 2
  samples), compact-record rounding/null handling for both aircraft and
  vessel shapes, JSONL round-trip, and an end-to-end
  mkdtemp-write-read-stats-cleanup test proving the throttle, the per-day
  file append, and `/api/data/archive/stats`'s counts all agree. All 7
  pass; `npm run test:node` is 13/13 (6 pre-existing + 7 new).
- Verified: `npx tsc --noEmit` — identical 45 pre-existing errors with and
  without this change (git-stash A/B diff, same method as the prior
  backtest_v2 session), zero new errors, none in the touched/new files.
  `npm run build` succeeds (client 3 chunks + server bundle, same warning
  profile as before — the large maplibre-gl chunk is pre-existing from the
  map slices, not from this change).
- Version: 1.0.41 (from 1.0.40, read-then-incremented per the prior
  session's attribution-note lesson on the 1.0.36 collision).
- Not done (explicitly out of scope for one logical change): geofence
  transit counters (R2 gate 1) — this PR only builds the raw material R2
  will read; WebGL rendering / viewport-fetching / enrichment (rest of
  R1) — untouched, still open per open_questions.md.
- Rollback trigger: if `/api/data/archive/stats.kinds.*.approxBytesOnDisk`
  trends toward the Railway volume's plan limit faster than the ~105MB/mo
  estimate (e.g. because bbox record counts run near the 800/1500 caps
  far more often live than assumed), lengthen `SAMPLE_INTERVAL_MS` in
  `server/dataArchive.ts` first (cheapest lever) before shortening
  `RETENTION_DAYS` — retention loses history depth, the interval only
  loses temporal resolution.
- STARVED: no — this session's scope (position archive) was fully shipped;
  high-value work remains queued (KNOWN BROKEN #3/#4/#5, counterfactual
  logger, Sentinel-2 gate 1, rest of R1/R2/R3/R4) for future sessions.

## 2026-07-03 — [REPAIR] Extended-hours order gating (KNOWN BROKEN #8)

- Session start check: /api/health all-ok (Alpaca ACTIVE, python bridge ok,
  bot active, equityPeak=108151.39/drawdownPct=0.0% — confirms the
  2026-07-03 equityPeak-persistence fix is holding live across deploys).
  Loop-health ratio: 3 of 4 total experiments.md entries are [REPAIR]
  (framework bootstrapped today, <10 entries exist yet — below the 7/10
  escalation threshold, not a concern). Audit log / trade_feedback still
  unreachable from an autonomous session (owner-gated per KNOWN BROKEN #4)
  so this session worked from KNOWN BROKEN #8, the next actionable
  un-diagnosed item, per SESSION BUDGET (no matured experiment to judge —
  everything else logged today is same-day).
- PRIOR (before reading the order-submission code, REASONING STANDARD #10):
  expected either (a) both stock and options extended-hours paths already
  correctly gated (nothing to do), or (b) options orders missing a time
  gate and firing outside 9:30-4:00 relying on Alpaca to reject them
  (wasted scan cycles per the human's framing). Did NOT expect the actual
  finding — that options were fine and the real gap was on the stock side.
- Finding (READ BEFORE WRITE, this session): `executeTrades()` — the only
  function that ever calls `submit_options_order`/`select_contract` — is
  invoked exclusively `if (isMarketOpen)` (bot.ts:3030); outside market
  hours new trades are queued (`morningQueue`) and executed at the next
  open via `executeMorningQueue()`, gated on `clock.is_open`. Options were
  never actually at risk of an off-hours submission attempt. The
  `options_exit` OrderContext variant is declared but never passed by any
  call site — dead but harmless.
  The real bug: `getOrderParams()`'s extended-hours branch (4am-9:30am,
  4pm-8pm ET) computes wider-buffer limit prices for stock/ETF orders but
  never sets Alpaca's `extended_hours: true`. Per Alpaca's API, a
  day-limit order submitted without that flag outside regular hours is
  simply queued for the NEXT REGULAR session — it does not attempt to
  fill during the extended session it was priced for. This branch is hit
  live by the real-time WS position-exit handler (stop_loss/trailing_stop/
  take_profit — fires on any price tick, not gated to market hours) and by
  the Tier-3 SPY/QQQ floor buy. Net effect: a stop-loss or trailing-stop
  computed during a 4am-9:30am or 4pm-8pm price move would never actually
  attempt to execute until 9:30am the next regular session — silently
  defeating the stop during exactly the window (thin liquidity,
  pre-market gaps) it matters most.
- Downstream chain (REASONING STANDARD #1): adding `extended_hours: true`
  → those day-limit orders become eligible to fill during the pre-market/
  after-hours session Alpaca actually runs → a stop-loss priced at 6am can
  fill near 6am instead of silently waiting until 9:30am → smaller
  realized loss on overnight/pre-market adverse moves that would otherwise
  ride uncapped until the regular open → net effect is MORE stops firing
  during extended hours (intended; this is a bug fix restoring intended
  behavior, not a threshold change) with no change to entry cadence (entry
  orders during extended hours were already funneled through the
  market-hours-gated morning queue in the live-fire paths that matter).
- Change (one logical change, options untouched): extracted
  `getETHour`/`getOrderParams`/`OrderContext` out of `server/bot.ts` into a
  new pure module `server/orderParams.ts` (zero behavior change beyond the
  fix — needed because `bot.ts` has import-time side effects and isn't
  safe to import directly in a test). Added `extended_hours: true` to the
  extended-hours branch for `stop_loss`/`trailing_stop`/`take_profit`/
  `new_entry`. Left the options branch (`options_entry`/`options_exit`)
  untouched — Alpaca has no options extended-hours session, so the flag
  must never be set there.
- Regression test FIRST (loop-health rule 3): `server/orderParams.test.ts`
  (Node's built-in `node:test`, zero new dependencies; `getOrderParams` now
  takes an optional `etHourOverride` param for determinism). Verified by
  temporarily stripping the fix and re-running: 4 of 6 assertions FAILED
  on the pre-fix code (stop_loss/trailing_stop/take_profit/new_entry all
  missing `extended_hours`); all 6 pass post-fix. Added `npm run test:node`
  to package.json to run it (`tsx --test server/*.test.ts`). Note: CI's
  node-build job (`.github/workflows/ci.yml`, FROZEN) does not currently
  invoke this script — only `tsc --noEmit` and `npm run build` run in CI.
  Wiring `test:node` into CI is a follow-up worth a human-approved
  wishlist entry since ci.yml can't be self-edited.
- Verified locally: `npm ci && npx tsc --noEmit` shows zero NEW errors
  (diffed against main via `git stash` — identical pre-existing
  vite/client + tsconfig + Buffer.trim() errors, all unrelated to this
  change, all already non-blocking in CI's `|| true`); `npm run build`
  succeeds (client + server bundle); `npm run test:node` — 6/6 pass.
- Version: 1.0.36 (from 1.0.35).
- Frozen-path judgment call, stated explicitly for the human to override:
  this touches order-body fields inside `server/bot.ts`'s stock/ETF order
  construction. Read the FROZEN PATHS order-submission clause as covering
  the HTTP transport/auth/retry mechanics (the `alpaca()` helper) and
  `options_execution.py`'s `submit_options_order`, not the pre-existing,
  already-mutable `getOrderParams()` order-type/pricing logic (which
  already varies type/limit-price/time-in-force by time of day before this
  change) — neither the `alpaca()` transport function, retries, auth, nor
  `options_execution.py` were touched.
- Rollback trigger: if live audit logs (once accessible) show extended-
  hours orders being rejected by Alpaca (e.g. account not enabled for
  extended-hours trading) rather than filling, revert this commit — the
  pre-fix behavior (queue for regular open) is a safe fallback.

## 2026-07-03 — [RESEARCH] Equity-momentum backtest harness (`bot_backtest.py`)
- Change: added a reproducible backtest of the bot's OWN momentum scoring
  (`strategies/momentum.py`) run as a monthly ETF rotation, vs SPY buy&hold.
  Yahoo daily adj-close, stdlib only, cached under `.bt_cache/`.
- Window: 2016-01 → 2026-04 (matches `backtest_10yr_results.json`).
- Result (CAGR / Sharpe / maxDD):
  - SPY buy & hold ............ 14.13% / 0.83 / -33.7%
  - QQQ buy & hold ............ 18.67% / 0.88 / -35.1%
  - Bot momentum top-3 rot. ... 10.09% / 0.63 / -31.4%
  - **Dual-momentum SPY/QQQ ... 16.31% / 0.90 / -28.6%**  (beats SPY; higher Sharpe, lower DD)
  - 200dma regime filter ...... HURT (cash drag in a bull decade)
- Reference: live options-heavy bot (static artifact) = **0.27% CAGR**.
- Hypothesis: the live bot's drag is **options overtrading** (options avg
  -0.75%/trade over 328+ trades), NOT the core signal. A simple rules-based
  equity/ETF momentum core is a large improvement. Dual-momentum SPY/QQQ is a
  candidate but is IN-SAMPLE over a tech-led decade — needs out-of-sample /
  walk-forward validation before any live change (HONESTY METRIC risk).
- Caveat: equity logic only; the options leg can't be backtested without
  historical option prices (see wishlist).

## 2026-07-03 — [REPAIR] Backtest engine rebuild (backtest_v2.py) — PRIOR STATED BEFORE FIRST RUN
- Change: real engine replacing the backtest.py stub (KNOWN BROKEN #1).
  Daily-bar sim of strategies/momentum.py + mean_reversion.py with
  live-identical regime gating (regime_util.classify_regime_5level;
  vxx_ratio = VXX/30d avg per macro_data.py). No lookahead (signal close[i]
  -> fill open[i+1]), 5 bps/side costs, ATR stops (1.5x, 2-6% clamp),
  2R targets, regime time stops. Alpaca-first data, Yahoo fallback.
  Options/squeeze legs NOT simulated (no historical data — wishlist).
- PRIOR (before first real-data run, REASONING STANDARD #10):
  On SPY, 3yr, strategy=all I expect: (a) LOW trade counts (regime gate —
  NEUTRAL/BEAR/PANIC all block entries, so the engine only trades
  BULL/CAUTION days); (b) momentum Sharpe 0.3-0.8, positive but BELOW SPY
  buy-and-hold (base-rate check #3) because gating keeps it in cash much of
  the time; (c) mean-reversion few trades (RSI<40 rarely coincides with
  BULL/CAUTION) with small positive expectancy; (d) runtime well under the
  120s bot.ts budget. The deliverable is infrastructure honesty
  (promotion rule 3 becomes satisfiable), NOT alpha.
- Result (SPY, all, 3yr; runtime 2s — well under the 120s bot.ts budget):
  momentum: 42 trades, Sharpe 0.777, +3.1% total, maxDD 1.48%, win 64.3%,
  alpha -16.5 vs SPY CAGR 17.2 (buy-and-hold base rate). mean_reversion:
  0 trades. VXX data ok (real vxx_ratio on trade records).
- Prior vs actual: (a) low trade count CONFIRMED; (b) momentum Sharpe 0.777
  inside the 0.3-0.8 prior band, below buy-and-hold as predicted; (c)
  mean-reversion "few trades" was directionally right but reality is ZERO —
  oversold days essentially never coincide with BULL/CAUTION + score>=63;
  worth a rule-cost-audit question, not a code change; (d) runtime confirmed.
- Version: 1.0.34. Ships with offline tests (canned bars) in
  test_audit_critical.py; the backtest_v2 regime-consistency gated test now
  RUNS and PASSES; a new test asserts backtest regime blocking stays
  consistent with live system_config.get_adaptive_params.
- Also found (pre-existing, NOT this change): full-repo `pytest -q` dies at
  collection because test_auto_discovery.py calls sys.exit() at module level;
  excluding it, 7 failures + 1 error in network/keys-dependent files
  (identical with and without this change — verified by stash A/B). CI's
  4-file offline subset is green: 110 passed, 1 skipped.

## 2026-07-03 — [REPAIR] Live-system diagnosis (BOOTSTRAP Phase 2, public surface)
- /api/health all-ok (Alpaca ACTIVE, python bridge ok, bot active); calendar
  correct on the July-3 NYSE holiday. Finding: equityPeak in-memory only ->
  drawdown kill-switch high-water mark resets every deploy (6 today);
  strengthening fix touches frozen kill-switch machinery -> wishlist
  proposal, not edited. Deeper verification of KNOWN BROKEN #3/#4 blocked
  by owner-only auth on all diagnostic routes -> access options proposed in
  wishlist. No code change this entry.

## 2026-07-03 — [REPAIR] Kill-switch peak persistence (KNOWN BROKEN #7, human-approved)
- Change: persist state.equityPeak (max-drawdown kill switch high-water mark)
  to /data/voltrade/voltrade_equity_peak.json (/tmp fallback), restored on
  boot — mirroring the 2026-05-05 equity-curve persistence fix for the same
  bug class. All 4 raise-sites save on the same line; halt logic untouched.
  Version 1.0.35.
- Regression test FIRST (loop-health rule 3): TestKillSwitchPeakPersistence
  in test_audit_critical.py — 3 of 4 assertions PROVEN FAILING on the old
  code (no persistence file, no boot restore, raises without save); the 4th
  is a scope guard asserting the halt condition itself is unchanged. All 4
  pass post-fix.
- PRIOR (before live verification, REASONING STANDARD #10): after deploy,
  the peak file is created on the first account poll / Tier 1 cycle (market
  closed until Jul 6, so expect /api/health equityPeak = 0 until then);
  from Jul 6 onward equityPeak survives restarts (nonzero across deploys)
  and drawdownPct measures from the true historical peak.
- Downstream chain (#1): persisted peak -> drawdown measured from true peak
  -> halt CAN fire after a slow multi-deploy bleed -> possibly more halts in
  genuine drawdowns (intended). ROLLBACK TRIGGER: a spurious DRAWDOWN-KILL
  from a stale peak (e.g. after an intentional capital change) -> revert the
  commit AND delete voltrade_equity_peak.json from the volume.

## 2026-07-03 — [PRODUCT] /data map v1 shipped (PROMPTS.md Section A executed)
- A0 status check -> A1-A3 + STARVED metric installed (PR #98, human
  pre-approved inline). A4 map build shipped in 4 vertical slices:
  #99 map+imagery+datacore boundary (v1.0.36), #100 live aircraft/OpenSky
  (v1.0.37), #101 strategic sites x16 (v1.0.38), vessels scaffold this PR
  (v1.0.39, key-gated: goes live automatically when AISSTREAM_KEY is set).
- The /data tab is LIVE on the site: satellite imagery, live ADS-B
  aircraft (30s-cached boundary proxy, stale-over-error), Cushing/STLD/
  ports reference layer with tradable-exposure metadata, RAW/SIGNAL
  labeling per the surface rules, gate-2 disclosure in the layer panel.
- Boundary discipline: frontend calls /api/data/* only; base imagery tiles
  are the documented scoped exception. datacore/ has zero trading imports.
- Ladder position: everything shipped is RAW (ungated by design). First
  SIGNAL candidate remains Sentinel-2 tank-fill at Cushing — gate 1 ground
  truth = EIA weekly storage. The sites layer just gave it its ground.
- STARVED: no — high-value work remains queued (KNOWN BROKEN #3/#4/#5/#6/#8,
  counterfactual logger, Sentinel-2 gate 1) but this session cleared its
  planned scope. Session end.

## 2026-07-03 — [PRODUCT] /data map UX overhaul + two production bug fixes (v1.0.40)
- Human production review failed the v1 map on basic usability -> full
  redesign per DESIGN.md (installed this session, PR #103): full-bleed at
  every width (fixed inset under 56px desktop nav / 48+64px mobile bars),
  collapsible layer panel (FAB top-right; collapsed by default on phone,
  open on desktop), toggle switches with live status dots + count badges,
  RAW/SIGNAL as info-tooltip, aircraft+sites ON by default (alive at first
  paint), site detail card (side card desktop / bottom sheet phone),
  styled popovers, legend, loading skeleton with 8s failsafe, maplibre
  chrome themed (44px touch controls, dark attribution).
- PRODUCTION BUGS found & fixed in scope: (1) /api/data/layers returned []
  in prod — frozen Dockerfile never copies datacore/ into the image; fixed
  by importing the JSON statically so esbuild bakes it into dist/index.cjs.
  (2) aircraft 502 — OpenSky rejects the Railway egress; fixed with UA
  header + adsb.lol fallback (verified live: 679 aircraft over Cushing).
  (3) maplibre CSS dynamically imported -> never applied in prod build ->
  300px phantom canvas + unpositioned controls; fixed with static CSS
  import + resize-on-ready.
- Harness (rule 6): npm run visual GREEN at 390/768/1440, screenshots
  reviewed against DESIGN.md. Harness itself hardened: SwiftShader WebGL
  flags + full network isolation (external requests aborted).
- Remaining known (site shell, NOT this page; follow-up candidate): desktop
  tab nav clips at 768px; several nav buttons < 44px touch floor.
- STARVED: no.

## 2026-07-03 — [PRODUCT] Map v2 directive: constitution + roadmap installed (PR 1 of 3)
- BUILD-FIRST RULE added to EDGE DOCTRINE (human-approved inline):
  4-step free-alternative assessment before any paid wishlist entry;
  honesty clause. DESIGN.md gains PERFORMANCE BUDGET (10k+ features
  smooth on phone, <3s interactive, stale-beats-spinner, server
  proxies-only) + FEATURE COMPLETENESS CHECKLIST (global scale, failure
  modes, first-load/error/empty, client-side heavy work, honest limits).
- Map v2 roadmap filed in open_questions (R1 this session; R2 maritime
  transit analytics ARCHIVE-FIRST; R3 environmental layers; R4 3D globe
  perf-gated) + archive-enabled signal hypotheses with ladder paths.
- Wishlist: OpenSky free account ($0, HUMAN ACTION), FlightAware/SWIM
  priced entry WITH build-first analysis, archive volume watch.
- ATTRIBUTION NOTE: version collision discovered — the concurrent routine
  session's extended-hours fix (PR #97) and map slice 1 (PR #99) both
  carry code_version 1.0.36 (slice 1's hardcoded-string bump silently
  no-op'd). Impact limited: only #97 affects trading behavior. Rule
  forward: bump by READING the current version and incrementing, never by
  replacing a hardcoded prior value.

## 2026-07-03 — [PRODUCT] Map v2 PR 2/3: global feeds + PERMANENT POSITION ARCHIVE (v1.0.41)
- ARCHIVE EVERYTHING is live: server/datacoreArchive.ts records every
  ingested aircraft/vessel position to the Railway volume from this deploy
  forward. Adaptive thinning (30s near strategic sites / 60s low-altitude /
  5min cruise; 2min near ports / 10min open water / 30min anchored),
  hourly JSONL rotate, gzip after 2h, 7-day rollup into per-entity daily
  track summaries (bbox + coarse polyline), /api/data/archive/stats for
  the volume watch, /api/data/track/:kind/:id serves recent trails.
  13 hermetic node:test cases (thinning ordering, cadence enforcement,
  gzip/rollup lifecycle, round-trip, stats).
- Feeds: GLOBAL coverage. Aircraft: OpenSky primary with OAuth2 support
  (activates when OPENSKY_CLIENT_ID/SECRET land — wishlist), adsb.lol
  fallback with HONEST partial-coverage flag when the viewport needs
  >250nm; per-provider exponential backoff (30s..15min); in-flight dedup
  so all visitors share ONE upstream request per bbox; ?since= dedup
  returns {unchanged:true} instead of re-sending payloads; caps raised to
  5000 (WebGL client lands in PR 3/3). Vessels: aisstream subscription
  widened to global, ShipStaticData captured (shiptype/destination) and
  merged into reads, caps 20k in-memory / 5k per response, coverage
  honesty in the source string (terrestrial AIS = mid-ocean gaps).
- Feed-error diagnosis (the "feed error — retrying" symptom): root cause
  was OpenSky rejecting Railway egress + no backoff, so every 15s poll
  re-failed the primary before falling back. Backoff now pins the
  provider out for 30s..15min after failures, and the fallback serves
  immediately.
- STARVED: no.

## 2026-07-03 — [PRODUCT] Map v2 client half: WebGL rendering + enrichment (with PR #106)
- Rendering: MapLibre WebGL symbol layers (per-marker DOM eliminated).
  Runtime-generated SDF silhouettes (jet/turboprop/piston/helicopter;
  tanker/cargo/small-craft) tinted per-feature, rotated to heading,
  velocity-vector line layers zoom-gated >=6 (halves low-zoom draw load —
  found by the perf harness). Verified at 10,000 aircraft.
- Enrichment: class from ICAO type designator + emitter category (free
  feed fields — adsb.lol 't'/'category'); detail cards with archive
  trails (/api/data/track), "route data unavailable — filed plans are a
  paid source" honesty line, AIS destination shown when broadcast,
  per-layer stale/partial-coverage notes; Escape/keyboard rules kept.
- Harness upgraded to the perf budget: 10k-feature fixture, __vtMap
  pan-driving, TTI + median-frame guards. Calibration journey (logged
  honestly): p95 -> median-after-warmup (upload hitches vs steady-state),
  fixture ?since= support (validates the delta path AND stops measuring
  redundant re-uploads), dsf 1 (features not pixels under software GL).
  Final: PASS x3 widths, median 67-167ms @10k in SwiftShader, TTI <1.7s.
- CI note: #106's first docker-build failed on npm ECONNRESET (registry
  flake — gate correctly held). Frozen ci.yml/Dockerfile can't grow
  retries without human approval; recurrence -> wishlist proposal.
- STARVED: no.

## 2026-07-03 — [PRODUCT] Archive supersession + coordination lesson (merge of #106/#107 work)
- A routine session and this interactive session BOTH built MAP V2 R1's
  position archive concurrently (#107 merged first; this branch carried the
  fuller build). Reconciled to ONE system: server/datacoreArchive.ts stays
  (adaptive thinning per the ARCHIVE EVERYTHING amendment, gzip, 7-day
  rollups, /api/data/track trails needed by the map client); PR #107's
  server/dataArchive.ts + test removed — its uniform 30-min sampling and
  no-compression design didn't meet the amendment. Adopted from #107: the
  measured growth-estimate discipline (~100MB/mo documented in datacore/
  README + module header). #107's experiments entry stays (append-only).
- Version: three-way lesson compounding — #97/#99 collided on 1.0.36, and
  #106/#107 BOTH took 1.0.41. This merge lands as 1.0.42.
- COORDINATION GOTCHA (added to ops list): concurrent sessions must CLAIM a
  roadmap item before building — append a one-line [CLAIMED <date> <PR#>]
  marker to the roadmap entry in open_questions.md in their FIRST commit,
  and check for claims before starting.

## 2026-07-03 — [REPAIR] Aircraft feed resilience: third provider + cause capture (v1.0.43)
- Live incident found during v1.0.42 verification: production aircraft
  layer dead with "both providers backing off" — OpenSky blocked from
  Railway (known) + adsb.lol egress flake ("fetch failed", cause hidden)
  exponentially pinned the only fallback -> zero aircraft for fresh
  bboxes. Both fallbacks verified healthy from outside Railway (adsb.lol
  764, airplanes.live 548 aircraft, identical field shape) -> egress-
  specific, transient.
- Fix: fallback chain now OpenSky -> adsb.lol -> airplanes.live (per-
  provider backoff so one flake can't kill the layer) + fetch-failure
  cause codes captured into the error string (bare "fetch failed" was
  undiagnosable). Feature-completeness checklist Q2 gap, closed.
- Also explains v1.0.42's empty aircraft archive (writes happen on
  successful fetch only); vessels archive + ShipStaticData typing were
  boot-warm-up, re-verified post-deploy below.

## 2026-07-03 — [PRODUCT] Map v2 Part 1 CLOSED — live verification (v1.0.43)
- Production evidence: aircraft GLOBAL alive (782 over Europe via adsb.lol
  with airplanes.live behind it); POSITION ARCHIVE RECORDING (aircraft +
  vessels files, 1.7MB day one — and the vessels file survived the deploy,
  volume persistence working); vessels feed in expected post-deploy
  warm-up (lazy WS connect — filed as KNOWN BROKEN #9 with the one-line
  eager-connect fix for a routine). ShipStaticData typing verification
  pending warm-up.
- Day tally: 20 PRs through the needs-gated automerge; 4 live incidents
  found by verification and fixed same-day (datacore-not-in-image,
  OpenSky egress block, maplibre CSS phantom canvas, both-providers-
  backing-off); 2 new permanent ops rules (claim-before-build,
  version-by-increment); the archive is accumulating the proprietary
  dataset R2 builds on. STARVED: no — queue is deep but this directive's
  scope is fully shipped.

## 2026-07-03 — [RESEARCH] OpenSky creds verification (negative) + aircraft-feed licensing audit
- Human set OPENSKY_CLIENT_ID/SECRET in Railway (wishlist item closed).
  Verification NEGATIVE: 6+ fresh-bbox probes over ~30 min (wider than
  the 15-min max backoff window, so at least one live OpenSky attempt
  was forced) all served from community fallbacks; OpenSky never took a
  request. OpenSky reachable anonymously from a non-Railway network
  (HTTP 200) -> the Railway egress rejection persists even with OAuth
  creds available. Disambiguation (IP block vs. auth-endpoint failure
  vs. env-not-loaded) needs Railway deploy logs — handed to the human
  in the wishlist entry.
- Prior expectation (recorded in the wishlist entry BEFORE the test):
  creds "may restore the primary feed" — rejected; an egress-level
  block is not an authentication problem.
- Licensing audit (triggered by the verification request): OpenSky
  license = non-profit research/education ONLY, and operational REST-API
  integration requires written permission even for non-profits — both
  tripwires fire for us (paid site features + automated integration).
  airplanes.live free API = explicitly non-commercial. adsb.lol = ODbL
  1.0, the only terms-compatible provider for commercial display today.
  Flagged constraint + provider-priority recommendation filed in
  wishlist.md for human decision — ANALYSIS ONLY per instruction, no
  code or priority-order change shipped.
- STARVED: no — usage-calibration loop build queued in same session.

## 2026-07-03 — [RESEARCH] Correction to the aircraft-feed licensing audit (same day)
- Human correction: VolTradeAI has NO paid product — billing code exists
  but nothing is charged; the site is a proof of concept. The earlier
  entry's claim that the commercial tripwire "fires" was wrong on the
  facts as of today.
- Revised assessment (wishlist entry updated in place): adsb.lol (ODbL)
  compatible now and after monetization; airplanes.live compatible NOW
  (non-commercial POC) but flips incompatible the day anything is
  charged; OpenSky's OPERATIONAL-use clause (written agreement for any
  live/automated integration, even non-profit) still applies today —
  and OpenSky is non-functional from Railway anyway, costing a 12s
  timeout per fresh viewport for zero data.
- Constraint reframed as a MONETIZATION TRIPWIRE: re-run provider
  compliance before enabling billing/ads. Still analysis-only; provider
  order unchanged pending human decision.
- STARVED: no.

## 2026-07-03 — [REPAIR] Vessel stream eager boot connect (KNOWN BROKEN #9, v1.0.44)
- PRIOR (stated before implementing): the aisstream websocket only ever
  connects when `ensureVesselStream()` is called, and the only caller was
  the `/api/data/vessels` GET handler — so a fresh deploy with zero
  visitors to the map stays disconnected indefinitely. Expected fix:
  calling the same function once at route-registration time (server boot)
  closes the gap with no other behavior change, since `ensureVesselStream`
  already no-ops if a socket is OPEN/CONNECTING and no-ops if the key is
  unset.
- Downstream chain (REASONING STANDARD #1): eager connect at boot -> the
  websocket is live before any request arrives -> the 60s archive-snapshot
  interval (already running unconditionally) has real positions to record
  from minute one instead of only after a map visit -> position-archive
  continuity across deploys improves (this feeds R2's transit-analytics
  signal, whose value depends on unbroken history) -> no change to sizing,
  scoring, or any trading path (this module has zero imports from bot_engine/
  system_config/strategies, so the datacore boundary is intact).
- Fix: extracted `vesselStreamEnabled(env)` and `bootVesselStream(env,
  connect)` into new pure module `server/vesselStream.ts` (no imports,
  fully unit-testable — avoids importing routes.ts's heavy deps, notably
  auth.ts's top-level sqlite `db` open, into a test). Replaced the three
  independent `process.env.AISSTREAM_KEY` truthy checks (layers status,
  vessels route, and the new boot call) with the single shared predicate,
  removing a duplication-drift risk. Added
  `bootVesselStream(process.env, ensureVesselStream)` immediately after
  `ensureVesselStream`'s definition in `server/routes.ts`.
- Test (loop-health rule 3 — regression test that would have caught the
  break): `server/vesselStream.test.ts`, 4 cases — `vesselStreamEnabled`
  true/false on key presence, and `bootVesselStream` invokes its connect
  callback iff a key is present (dependency-injected, so it directly
  proves the "connect on boot when enabled" wiring without needing a real
  WebSocket or Express app). 17/17 `npm run test:node` pass (13 pre-
  existing + 4 new).
- `npx tsc --noEmit`: identical 46 pre-existing errors with/without this
  change (git-stash A/B diff), zero new errors, none in touched/new files.
  `npm run build`: client + server bundle succeed. No client/ files
  touched, so DESIGN.md visual-verification (promotion rule 6) doesn't
  apply.
- Version 1.0.43 -> 1.0.44 (read-and-increment).
- STARVED: no — KNOWN BROKEN #5 (data-module wiring audit) and #6 (pytest
  collection) remain queued for a future session; SESSION BUDGET caps
  this session at one action.

## 2026-07-03 — [PIPELINE] OpenSky dropped from aircraft chain (human decision, v1.0.45)
- Change: fetchAircraft in server/routes.ts no longer attempts OpenSky —
  chain is now adsb.lol (primary) -> airplanes.live (fallback). OAuth
  helper removed (v1.0.43 git history holds the implementation for
  reinstatement). layers.json + harness fixture attribution updated;
  coverage note no longer promises OpenSky credentials.
- Why: verification showed OpenSky never serves from Railway even with
  creds (egress block), so it contributed only a ~12s dead attempt per
  fresh viewport; AND its operational-use license clause requires a
  written agreement we don't have. Human emailed OpenSky requesting a
  research agreement — reinstate + re-verify Railway connectivity if
  granted (wishlist entry updated with the trigger).
- Expected effect: fresh-bbox latency drops by the dead-attempt time
  (~12s worst case); zero functional coverage change (OpenSky served 0%
  of requests). Downstream chain: fewer request timeouts -> fewer
  in-flight dedup pile-ups behind slow fetches; archive feed cadence
  unchanged (it records on successful fetches only).
- Regression tests: server/aircraftChain.test.ts (4 source-level tests:
  OpenSky fully absent, adsb.lol before airplanes.live, no stale
  coverage note, layers.json attribution matches chain).
- Rollback trigger: OpenSky grants the research agreement (reinstate),
  or both community providers show sustained simultaneous failure in
  the audit log (re-add a third provider — see the provider-redundancy
  research item in open_questions.md).

## 2026-07-03 — [RULE-REVIEW] Monetization tripwire hardened: runtime guard (v1.0.46)
- Constitutional half (human-approved verbatim, this message): CLAUDE.md
  KNOWN STATE gains the MONETIZATION TRIPWIRE standing rule — sessions
  touching billing/pricing/subscriptions/ads/paid-gating must re-run the
  wishlist provider-compliance check before merging.
- Runtime half (new server/providerCompliance.ts): billingActive() =
  BILLING_ENABLED=true OR STRIPE_SECRET_KEY present (billing.ts — frozen,
  read-only — activates on that key, so key presence is the earliest
  observable monetization signal). While airplanes.live (non-commercial
  license) remains in the aircraft chain, activation produces a throttled
  COMPLIANCE-WARNING row in the persistent audit log + a licensing check
  on /api/health that degrades overall status — a dashboard-only
  monetization flip becomes visible to the next DAILY routine's health
  check within hours, no code change needed to detect it.
- Wiring: boot check at datacore route registration + tick per aircraft
  request (both throttled to one warning per 6h window); /api/health
  Check 6 in bot.ts. Datacore boundary intact: the guard lives in the
  serving layer (routes/bot), NOT datacore/ — datacore keeps zero
  knowledge of billing.
- Tests: server/providerCompliance.test.ts (6) — inactive-by-default,
  BILLING_ENABLED trip, STRIPE_SECRET_KEY trip, tick throttling with
  injected clock, non-commercial list pinned to the live chain, and
  wiring pins for /api/health + aircraft path.
- Measurement-integrity note: this changes /api/health's payload shape
  (adds checks.licensing; can newly degrade status). Direction of bias:
  none on trading metrics — it can only ADD a warning state.
- OPS LESSON (2nd occurrence — now a pattern): importing auth.ts's
  top-level sqlite db into any node:test hangs the runner (open handle).
  PR #113 dodged it with pure vesselStream.ts; this PR first hit it,
  then adopted the same pattern (pure module + injected audit writer).
  RULE: server modules that need the db AND unit tests take the db
  dependency by injection; never import ./auth from a tested module.

## 2026-07-03 — [RULE-REVIEW] Constitution batch: fall-through, dead-code, hygiene + first audit
- Applied three human-approved amendments (this message, bookkept in
  wishlist): SESSION BUDGET -> productive fall-through ladder; DEAD CODE
  POLICY; CONSTITUTIONAL HYGIENE (audit files proposals, never
  self-applies; live rule conflicts resolved by GOAL order + filed).
- Queued research filed per the directive: aircraft/vessel provider
  redundancy (chain is two-deep post-OpenSky; vessels single-sourced);
  OpenSky likely-returner tracked with REVIEW-BY 2026-08-17, no adapter
  retained (git-history reinstatement documented).
- Ran the FIRST CONSTITUTIONAL AUDIT (details in wishlist): 2 findings
  proposed for consolidation (rules-in-KNOWN-STATE -> STANDING BEHAVIORS
  section; scattered periodic duties -> one PERIODIC AUDITS register);
  factual drift fixed directly (backtest STUB claims stale since the
  rebuild); interaction checks clean (STARVED, BUILD-FIRST, tripwire).
- Docs-only; no version bump (no runtime behavior change).
- STARVED: no — remaining queue continues in this same session (daily
  usage loop PR + scale-now schedule answer).

## 2026-07-03 — [RULE-REVIEW] Daily usage-calibration mode + scale-now schedule (docs)
- Applied the approved daily-aggressive usage rule (CLAUDE.md KNOWN
  STATE updated in place — supersedes the weekly-only rule from earlier
  today; usage_log.md carries both modes with the ~2026-07-24 revisit).
- voltrade-usage-check routine prompt canonicalized in usage_log.md
  (DAILY 21:30 ET; draft-only Gmail caveat + Notifications-tab
  recommendation embedded so the routine never depends on send).
- SAME-DAY SCALE RECOMMENDATION recorded in usage_log.md per the human's
  "scale now" directive: create product-am 9:00, daily-midday 12:30,
  product-pm 14:00, product-eve 20:00, usage-check 21:30, edge-late
  22:30 — completes the A5 8-run table + daily nudge. Evidence: weekly
  15% (resets ~Jul 5) with 3 routines + the heaviest interactive day;
  throttle trigger stated (5-hour peaks >~80% → drop order).
- STARVED: no — directive fully executed this session (PRs #114 #115
  #116 + this).

## 2026-07-04 — [PIPELINE] SEC EDGAR Form 4 (insider transactions) — first datacore pipeline beyond aircraft/vessels

- Session start: read CLAUDE.md, all of research/, per [PRODUCT] session
  protocol. Loop-health ratio over the last 10 entries: 0 [REPAIR], 3
  [PIPELINE]/[PRODUCT], 3 [RULE-REVIEW], 1 [RESEARCH] — well below the 7/10
  escalation threshold. KNOWN BROKEN #3/#4/#5 remain open (owner-gated
  diagnostics, per wishlist) but don't touch datacore/ or block product
  work, per this session's brief. Prior branch (#117, daily usage-calibration
  docs) was already merged to main — reset claude/quirky-hopper-u5pdl1 onto
  origin/main fresh per the merged-PR restart protocol, no stacking.
- Chose this action (option (a): advance a datacore/ pipeline through gate
  1) over the other three (UI-only work with no new signal, a proposal-only
  new-root writeup, or API/docs hardening) because EDGE DOCTRINE #1 ("BUILD
  DATA, DON'T BUY IT") names SEC EDGAR real-time Form 4 as a standing
  example, `datacore/layers.json` had exactly one non-"live" candidate
  (`tank_fill`, gated on Sentinel-2 satellite image processing — infeasible
  to attempt end-to-end in one session, no image-analysis toolchain
  available here), and no datacore pipeline had shipped since aircraft/
  vessels/archive — Form 4 requires no API key, no image processing, and no
  new paid access, purely free-data processing (the labor-not-ingredient
  edge the doctrine calls out).
- PRIOR (before writing any parser code, REASONING STANDARD #10): expected
  the main design risk to be SEC's fair-access rate limiting (10 req/s) and
  the feed's atom format double-listing each filing (once per filer, once
  per issuer) causing double-counted/double-fetched filings; expected the
  fix to be accession-number dedup plus a small per-request delay, and
  expected a hand-rolled tag-scoped regex extractor to be sufficient given
  no XML parsing library exists in package.json today (matches CLAUDE.md's
  "don't add unneeded abstractions" — Form 4's schema is flat and stable).
  Both priors held; see design below.
- Verified network reachability first (Bash curl, not a guess): SEC EDGAR
  is directly reachable from this environment. Live-fetched two REAL,
  current Form 4 filings (accessions 0001104659-26-080497 — a derivative
  RSU grant, code A — and 0000902664-26-003001 — a 3-reporting-owner
  non-derivative open-market SALE, code S, two transactions) and
  hand-verified every field (issuer, owner relationship flags, transaction
  code/shares/price/shares-owned-after) by reading the raw filed XML myself
  before writing a single assertion. This IS ladder gate 1 (DATA — verified
  against an external truth source): for a filings parser, the filed
  document itself, read directly, is the only ground truth there is to
  check against — there's no separate "official" source above the filing.
- Design: `server/edgarForm4.ts` (pure module, zero imports from trading
  logic — datacore boundary rule) — a dependency-free tag-scoped XML
  extractor (`parseForm4Xml`), a transaction-code classifier
  (`classifyTransactionCode`: P=open_market_buy the informative discretionary
  case, S=open_market_sale, A=award_grant, M=option_exercise, G=gift,
  F=tax_withholding, else other), an atom-feed parser that dedupes by
  accession number (`parseFilingFeed`), an index.json XML-document picker
  (`pickOwnershipXmlName`), a sequential fetch-with-delay batch fetcher
  (`fetchLatestForm4Filings`), and an in-memory cache + 15-min poll loop
  booted eagerly at route registration (`bootForm4Poll`) — same eager-boot
  shape as `vesselStream.ts`, deliberately avoiding KNOWN BROKEN #9's lazy-
  first-request gap. Wired at `/api/data/insider` in `server/routes.ts`,
  kind: raw (as-filed display, no predictive claim — the interpreted
  "clustering predicts returns" question is gate 2, unattempted, logged as
  a new hypothesis in open_questions.md). Registered in
  `datacore/layers.json`. `datacore/README.md` updated to correct its
  stale aspirational `pipelines/` Python layout note (every real pipeline,
  including this one, lives in `server/*.ts` — DEAD CODE POLICY spirit:
  don't let docs claim an unbuilt structure is authoritative).
- Client (MUTABLE rule: new user-visible bot function needs UI in the same
  PR): `client/src/pages/datamap.tsx` — the insider feed has no lat/lon, so
  rather than force it into the maplibre marker machinery (`wireLivePoints`,
  built for geospatial layers), it renders as an inline expandable list
  directly under its row in the existing layer panel (new `.vt-filings-*`
  CSS in `index.css`), scoped to avoid the floating-panel collision risk a
  new independently-positioned overlay would add on phone widths (site-card
  and layer-panel already claim opposite corners / the mobile bottom sheet).
  Defaulted on (`DEFAULT_ON.insider = true`) — matches aircraft/sites, the
  other no-key-required RAW layers.
- Downstream chain (REASONING STANDARD #1): pipeline ships gate-1-verified
  -> `/api/data/insider` serves real filings today -> the feed accumulates
  its own history from this point forward (no live trading impact —
  datacore boundary rule holds, verified zero imports from bot_engine.py /
  system_config.py / strategies/ / server/bot.ts) -> once enough history
  accumulates (or the free SEC bulk full-index is pulled in, unexplored,
  logged in open_questions.md), gate 2 (does insider-buy clustering predict
  forward returns vs. a random-entry base rate, REASONING STANDARD #3) can
  be attempted -> only then would this ever become a SIGNAL eligible for
  the tournament in strategies/, per the ladder — no shortcut taken.
- Regression tests FIRST (loop-health rule 3): `server/edgarForm4.test.ts`
  — 6 tests: the two gate-1 field-by-field fixture checks against the real
  filings above, transaction-code classification for all 7 codes, feed
  dedup against a real (trimmed) atom-feed snippet proving the actual
  filer/issuer double-listing collapses to one entry, and two
  `pickOwnershipXmlName` cases (found / not-found) against a real
  index.json shape. All 6 pass; `npm run test:node` is 33/33 (27
  pre-existing + 6 new) — no existing assertion touched or weakened.
- Verified: `npx tsc --noEmit` — identical 45 pre-existing errors with and
  without this change, zero new errors, none in touched/new files.
  `npm run build` succeeds (same chunk/warning profile as documented).
  `npm run visual` (client/ touched — PROMOTION RULES rule 6): added
  `insider` to the layers + a 2-filing fixture in
  `scripts/visual_check.mjs`; PASS at 390/768/1440 with 0 hard failures
  (touch-target warnings present are pre-existing global-nav elements, not
  from this change) — screenshots reviewed, the new layer row and its
  inline filings list render correctly with GRANT/SELL badges and RAW
  labeling, no overflow or overlay-coverage regression.
- No backtest applies (data pipeline, not a strategy/parameter change) —
  PROMOTION RULES item 3 is N/A here, same as the aircraft/vessel PRs.
- Version: 1.0.47 (from 1.0.46, read-then-incremented).
- Hypothesis: gate 1 stands permanently verified (parser correctness is a
  static property, not a market claim). Gate 2 hypothesis and prior stated
  in research/open_questions.md ("Insider Form 4 clustering as a signal") —
  expect small positive edge in officer/director open-market buys on
  small/mid caps, near-zero on mega-caps, kill if no separation from
  random-entry baseline after >=90 days of accumulated history.

## 2026-07-03 — [PRODUCT] Strategic-sites accuracy audit: 16/16 imagery-verified, 11 corrected (v1.0.48)
- Human reported Port of Charleston mispositioned; full audit ordered.
  Method (now compiled as scripts/site_verify.py + site_candidate_verify.py
  per EDGE DOCTRINE #3): render every stored coordinate on Esri World
  Imagery with a crosshair; the facility must be visibly present.
  DESIGN.md gains the human-approved REFERENCE DATA ACCURACY rule.
- VERIFIED UNCHANGED (5): cushing_enbridge (tanks under crosshair),
  cushing_plains (tank rows), port_la (wharf/cranes, San Pedro),
  port_nynj (Elizabeth container yard), port_savannah (Garden City
  stacks).
- CORRECTED (11) — old -> new (offset, what the old pin actually was):
  - port_charleston (32.921,-79.86) -> (32.8325,-79.8800): ~10km — pin
    was residential Mount Pleasant; now on Wando Welch container yards.
  - cushing_hub (35.985,-96.767) -> (35.9487,-96.7587): ~4.1km — pin was
    downtown Cushing street grid; now mid-tank-farm in the main district.
  - stld_butler (41.428,-84.855) -> (41.3705,-84.9170): ~8.2km — pin was
    farmland; now on the SDI Butler mill (scrap yard + melt shop visible).
  - stld_columbus (33.532,-88.415) -> (33.4473,-88.5768): ~17.9km(!) —
    pin was Columbus MS suburbs; mill is actually at the Golden Triangle
    megasite next to GTR airport.
  - stld_sinton (28.041,-97.56) -> (28.0563,-97.4493): ~11.0km — pin was
    ranch land W of town; mill is NE along the rail line.
  - stld_columbia_city (41.157,-85.488) -> (41.1199,-85.3484): ~12.4km —
    pin was downtown Columbia City; mill is E of town (structural mill +
    rail sidings visible).
  - port_houston (29.681,-94.942) -> (29.6770,-95.0060): ~6.2km — pin
    was open water in Trinity Bay; now on Barbours Cut yard.
  - port_oakland (37.796,-122.279) -> (37.7980,-122.3145): ~3.1km — pin
    was Jack London Square; now on OICT container yard.
  - port_norfolk (36.877,-76.328) -> (36.9155,-76.3275): ~4.3km — pin
    was Lambert's Point COAL pier (wrong facility); now on NIT apron.
  - port_lb (33.754,-118.216) -> (33.7515,-118.2130): ~370m — pin was in
    the channel; now on Pier J stacks.
  - port_seattle (47.582,-122.352) -> (47.5820,-122.3474): ~340m — pin
    was Harbor Island's fuel tank farm; now on T18 container yard.
- Lesson (feeds the geofence future): "researched coordinates" from
  memory/public materials produced town centroids, not facilities — the
  archive's site-proximity thinning (nearAnySite) has been using these
  wrong positions, so near-site full-resolution sampling was mistargeted
  for 6 of 16 sites by >4km. Corrected data fixes that silently.
- Downstream chain: sites layer markers move -> archive adaptive
  thinning now samples the RIGHT areas at full resolution -> future R2
  transit counters + tank-shadow work inherit verified ground truth.

## 2026-07-03 — [PRODUCT] Site category icons: anchor/tanks/factory silhouettes + legend (v1.0.49)
- Map v2.1 SYMBOLS directive: strategic-site markers upgraded from
  generic colored dots to category silhouettes in the existing SDF icon
  system (per-feature icon-color on GPU, upright, dark halo for imagery
  contrast): vt-port anchor, vt-tank 3-cylinder cluster, vt-mill
  factory-with-chimney. Legend now leads with the three shapes (inline
  SVG twins of the canvas shapes); aircraft/vessel color entries kept.
- Promotion rule 6: npm run visual green at 390/768/1440 (0 hard
  failures; pre-existing site-shell nav warnings unchanged). Icon
  legibility self-reviewed via isolated render (sites-only fixture,
  US view, 390px + 1440px): three shapes distinct at phone size,
  correct category colors, legend aligned.
- Perf unchanged: same symbol-layer path as aircraft (16 features is
  noise next to the 10k-aircraft budget); harness perf medians 117ms
  unchanged from v1.0.48 baseline.

## 2026-07-03 — [PRODUCT] US power plants layer: 9,833 plants, fuel icons, clustering (v1.0.50)
- Map v2.1 POWER PLANTS directive. Free data root per EDGE DOCTRINE #1:
  WRI Global Power Plant Database v1.3.0 (CC BY 4.0 — commercial-safe,
  attribution shipped in layers.json + detail card + legend source).
  Compiler: scripts/build_powerplants.py (re-runnable when WRI updates)
  -> datacore/powerplants/us_power_plants.json (9,833 US plants, 762KB
  compact rows; solar 3283 / gas 1852 / hydro 1449 / wind 1139 / oil 876
  / other 879 / coal 297 / nuclear 58).
- Serving: /api/data/powerplants — whole-file, day-cached, static import
  (esbuild bakes it; Dockerfile never copies datacore/). RAW layer, no
  ladder gate needed (no predictive claim); signal hypotheses filed in
  open_questions.md (EIA-930 generation-mix utilization; NRC outage
  adjacency) with full ladder paths.
- Client: maplibre native clustering (clusterMaxZoom 7, radius 50) so
  ~9.8k features stay legible + cheap on phones; unclustered points are
  8 new fuel silhouettes in the SDF system (atom/coal pile/flame/derrick/
  drop/turbine/sun/bolt) with per-fuel tint; cluster click zooms in;
  point click -> detail card with MW + operator + attribution; legend
  gains a plants row; panel row with count badge (Zap icon).
- Tests: server/powerplants.test.ts (3) — dataset scale + row validity +
  US bounds, CC BY attribution present, route/layers.json wiring.
  Harness fixture serves the REAL compiled dataset so the perf window
  now measures 10k aircraft + 9.8k clustered plants together.

## 2026-07-03 — [PRODUCT] M4 mobile performance pass: fill-rate fix from per-layer profile (v1.0.51)
- Method (compiled: scripts/perf_profile.mjs): pan-frame medians at 390px
  under SwiftShader (mid-range-phone proxy), per-layer A/B at two views.
  PRIOR (stated before profiling): expected the new 9.8k-plant layer or
  its clustering to be the top cost.
- PROFILE REJECTED THE PRIOR: at global zoom, base=17.7ms, +plants+sites
  =17.5ms (clustering is FREE — supercluster renders only ~40 blobs),
  +aircraft=33.8ms. The 10k-aircraft symbol layer was the only
  meaningful cost, and it is FILL-RATE bound (software rasterizer +
  phone GPUs pay per drawn pixel; 10k icons at constant 0.5 scale).
  Continental zoom (z4.5) was already smooth for all combos (21ms).
- Fix: zoom-interpolated icon-size on the two 10k-class live layers
  (aircraft 0.32@z2 -> 0.55@z7; vessels 0.30 -> 0.50) — ~60% fewer
  drawn pixels where icons are dense, full size where you can tap them.
- Measured effect: all-layers global-zoom median 32.8 -> 27.3ms in the
  A/B; full harness at 390px median 33ms p95 67ms (previous runs were
  ~117ms class); 768px 83ms; 1440px 117ms — all far inside the 300ms
  budget. Icons re-verified legible at the new global-zoom size.
- Tile prefetch not measurable offline (harness aborts CDN) — noted
  honestly; raster layer cost showed as flat base across combos.

## 2026-07-03 — [PRODUCT] Map v2.1 ops items: delivery channel fixed, tripwire verified, audit-cycle proposal
- GMAIL DRAFT-ONLY FIX (human directive): both calibration routines
  (usage-check daily 21:30, weekly-review Sun 10:00) now deliver via
  their FINAL SESSION OUTPUT — read from the Claude Code Notifications
  tab, the one channel verified to reach the human (routine completions
  land there by platform design). The Gmail-draft step was dropped from
  both canonical prompts as a dead letterbox (connector verified
  draft-only, no send tool; drafts sat unread). usage_log.md prompts +
  CLAUDE.md KNOWN STATE updated. Honest limit: no push-to-phone channel
  is verifiable from a session; if the Notifications tab proves too
  passive in practice, the human should say so and we revisit.
- TRIPWIRE FALSE-ALARM CHECK: production /api/health shows
  checks.licensing = ok -> STRIPE_SECRET_KEY is NOT set in Railway and
  the compliance warning is NOT firing. Condition for changing the
  trigger ("if firing falsely") unmet — guard unchanged. Residual
  behavior documented: setting a Stripe TEST key without charging would
  fire it; that is by design (earliest observable signal) and the
  warning text says exactly what to do.
- AUDIT CYCLE consolidation proposal filed in wishlist.md per the
  hygiene process (three scattered periodic triggers -> one SESSION
  BUDGET clause + register; policies untouched; human decides).

## 2026-07-03 — [PIPELINE] Aircraft chain three deep: adsb.fi third leg (v1.0.52)
- Human directive (multi-modal expansion): self-hosted receivers OFF the
  table (declined; logged in open_questions so no session re-proposes
  hardware); software-only third provider instead. LICENSING FIRST per
  the standing rule: adsb.fi = personal/non-commercial + attribution
  (same class as airplanes.live; MONETIZATION TRIPWIRE list updated to
  cover it); adsb.one rejected (Cloudflare blocks server egress); ADS-B
  Exchange rejected for the free chain (community API non-commercial +
  keyed; commercial tier is paid Enterprise).
- Integration: PROVIDERS gains per-provider response-array key (adsb.fi
  returns "aircraft" where the others return "ac"; URL pattern
  /api/v2/lat/{lat}/lon/{lon}/dist/{nm}); shared-upstream pattern,
  backoff, cause-capture all inherited. layers.json attribution updated.
- INTERNATIONAL COVERAGE VERIFIED through the new leg (the directive's
  requirement): Tokyo 130, Sydney 146, São Paulo 69 aircraft from
  adsb.fi; legs 1-2 verified global in prior sessions (all three are
  worldwide community networks — coverage everywhere feeders exist).
- Tests updated: aircraftChain.test.ts pins the three-deep order
  (ODbL leader first); providerCompliance.test.ts pins that BOTH
  non-commercial legs are in the tripwire list (sync test caught the
  edit requirement immediately).

## 2026-07-04 — [PRODUCT] Live trains layer: Finland + Norway launch, archived, iconed (v1.0.53)
- Multi-modal directive, TRAINS part. Free real-time rail positions with
  clean licenses (checked FIRST): Finland Digitraffic (CC BY 4.0, no
  key, plain JSON) + Norway Entur (NLOD-class open, ET-Client-Name
  header only, GraphQL mode:RAIL) — both verified live before build.
  US freight rail positions are PROPRIETARY (Class I railroads sell
  them; no free source) — stated in layers.json where users read it AND
  pinned by test so no session chases it. Amtrak has no clean official
  free JSON — future source evaluation, not launch.
- Server: /api/data/trains — pure mapping module server/trainsFeed.ts
  (unit-tested with real captured payloads; m/s->km/h for Entur;
  Digitraffic has no bearing -> null -> upright icon), shared 30s cache
  + in-flight dedup + per-source backoff, per-source status in the
  response so the panel labels coverage HONESTLY ("FI 47 · NO 12").
  Positions feed the permanent archive: datacoreArchive gains the
  trains kind end-to-end (2-min fixed cadence; hourly JSONL; gzip;
  rollup; recentTrack -> click-through trail like aircraft/vessels).
- Client: vt-train SDF locomotive (teal #2dd4bf; rotates to bearing
  where published), 30s poll, detail card with speed + per-country
  source attribution + archive trail, legend entry, panel row with
  count badge + per-source note.
- BUG CAUGHT BY SELF-REVIEW (rule 6 render check, would have shipped
  invisible): a ["case",["get","rotate"],...] icon-rotate expression
  silently killed symbol rendering (source had features, image
  registered, zero rendered). Fix: always-numeric bearing property +
  plain ["get","bearing"]. Lesson: maplibre expression rejection is
  SILENT — any new expression-driven layer needs a rendered-count
  assertion in review, not just "layer exists".
- Gates: node 41/41 (5 new trains tests incl. archive round-trip via the
  shared machinery), CI python 114, harness green 390/768/1440.

## 2026-07-04 — [RESEARCH] Ships coverage verified + trucks build-first conclusion + freight proxies
- SHIPS (directive): aisstream subscription confirmed GLOBAL in code
  (BoundingBoxes ±90/±180 — routes.ts) — the honest gap is physical:
  terrestrial AIS sees ~40-60nm offshore, mid-ocean is dark. Satellite
  AIS filed in wishlist as PRICED (quote-only, ~$500+/mo entry class)
  with build-first analysis: dead-reckoned predicted tracks + coastal
  reacquisition cover most port-transit needs free; do not buy unless
  a gated signal needs mid-ocean truth.
- TRUCKS (directive): build-first analysis TERMINATES — individual
  truck positions are private fleet telematics with no public feed at
  any price tier relevant to us; conclusion filed in open_questions so
  no session chases it. Four free freight PROXIES filed with ladder
  paths instead (CBP border waits, PeMS truck-lane volumes, FMCSA
  carrier census, port TEU monthlies) — archive-first, gate-2 gated.
- STARVED: no — directive fully executed across PRs #124 (aircraft
  third leg), #125 (trains layer), and this docs bundle.

## 2026-07-04 — [PRODUCT] Power-plant position accuracy: EIA-860 coords + top-100 imagery-verified (v1.0.54)
- Directive (Map v2.2): human confirmed Hardeeville mispositioned.
  KEY FINDING that shaped the protocol: GPPD and EIA-860 AGREE on
  Hardeeville's wrong position — the registries share self-reported
  geocodes (address/office, not the plant), so registry cross-agreement
  is NOT verification; imagery is the only ground truth. Rule text
  ("imagery or an authoritative source") interpreted accordingly:
  authoritative-source checks fix DISAGREEMENTS, imagery establishes
  VERIFIED.
- Data work: joined all 9,833 GPPD-US plants to EIA-860 2024 by plant
  code (9,557 matched; 276 GPPD-only = mostly retired since 2021).
  67 disagreements >300m (median 1.6km, worst 13.4km — wind-farm
  centroid vs substation ambiguity dominates); EIA 2024 coordinates now
  preferred for ALL matched plants.
- Imagery verification of the TOP 100 BY MW (directive): composite
  4x4 verification sheets (7 sheets, z14 crops with crosshairs) —
  98/100 passed on sheet review; 2 borderline resolved PASS at z15/z16
  close-up (Bath County = crosshair on the pumped-storage intake works;
  Ravenswood = on the station). 100/100 verified; audit artifact
  checked in (datacore/powerplants/imagery_verified.json).
- Product honesty: row format gains a verified flag; detail card says
  "Position imagery-verified." or "Position approximate
  (registry-reported — GPPD/EIA-860)."; the layer panel row notes
  "top 100 by MW imagery-verified · rest approximate".
- Tests: 7-element row validation + top-100-all-verified + audit
  artifact + EIA-860 credited. 42/42 node, 114 python, harness green.

## 2026-07-04 — [PRODUCT] Form 4 full view + filings archive (v1.0.55)
- Map v2.2 FORM 4 UI directive. Server: filings now ARCHIVED
  (COLLECT-EVERYTHING) — every 15-min poll appends new accessions to
  daily JSONL under the datacore archive volume (restart-safe dedup via
  day-file seeding), days >2 gzipped; /api/data/insider/history?days=N
  merges archive + live cache (history accumulates from 2026-07-04).
- Client: #/data/filings full view (hash-driven overlay; deep-linkable;
  back-button works): readable table at 768/1440 (ticker+company,
  insider+role, color-coded BUY/SELL/GRANT/EXERCISE, shares, price,
  computed value, date, SEC-filing link per row), stacked labeled cards
  at 390; filters all/open-market/buys/sells; designed empty/loading/
  error states. Panel keeps the compact list + "Open full view" button.
- TWO BUGS CAUGHT BY SELF-REVIEW: (1) home.tsx's tab-hash sync stomped
  #/data/filings back to #/data on mount — sync now rewrites only when
  the hash ROOT differs (subpaths survive); (2) the harness fixture
  matcher's startsWith let /api/data/insider shadow .../insider/history
  — exact-match-first fix in visual_check.mjs.
- Tests: archive dedup round-trip, gzip-day readback, route + poll-loop
  wiring pins (45/45 node). Harness green all widths; both view states
  (empty + populated) screenshot-reviewed.

## 2026-07-04 — [PRODUCT] Detail-card link-outs + vessel flag states (v1.0.56)
- Map v2.2 SHIP DETAIL CARDS directive. Vessels: flag state now shown,
  computed locally from the MMSI MID prefix (ITU table baked into
  client/src/lib/mmsiFlag.ts — data derived from the AIS message itself,
  no external lookup); dimensions honestly OMITTED (our aisstream
  subscription doesn't carry them reliably); destination/type/speed
  already shown. LINK OUT (never embed — photo copyright): MarineTraffic
  + VesselFinder by MMSI. Aircraft: Planespotters (photos/registry) by
  hex + adsb.lol live-track link. Rendered as pill links in the detail
  card, 32px min height, external target with rel=noreferrer.
- Gates: build, node 45/45, harness green all widths.

## 2026-07-04 — [PRODUCT] Layer panel v2: collapsible groups + imagery-date honesty (v1.0.57)
- Map v2.2 LAYER PANEL directive: with 7+ layers the flat list stopped
  scaling. Panel now groups into Base / Live tracking / Facilities /
  Filings & flows / Signals—coming-soon (signal/planned layers auto-
  route to the last group), with collapsible headers ("2/3 on" counts),
  per-layer info toggles (name click -> description + source), and the
  existing status lines/badges/switches unchanged. 44px touch targets
  on headers and name buttons.
- IMAGERY METADATA honesty: the imagery row now states "capture date
  unavailable (Esri base tiles)" — DESIGN.md gains the human-approved
  standing rule: show "imagery as of [date]" where a source exposes
  capture dates (Sentinel-2 when it lands), say "date unavailable"
  otherwise; no imagery surface may imply currency it cannot prove.
- Verified at 390/768/1440 (rule 6): desktop screenshot shows all four
  groups with honest per-layer notes; phone keeps the collapsed FAB.

## 2026-07-04 — [PIPELINE] Shadow-fleet analytics from our own AIS archive (v1.0.58)
- Map v2.2 SHADOW FLEET directive. server/shadowFleet.ts derives from
  the vessel archive (first payoff of ARCHIVE EVERYTHING beyond trails):
  gap events (silent >6h AND reappeared >100km), identity candidates
  (name under two MMSIs; new-MMSI-near-last-position hull-swap
  heuristic), loitering (>=4h, median <2kts) in 7 public STS zones
  (datacore/shadow_zones.json: Laconian, Ceuta, Malta, Fujairah,
  Singapore OPL, Kerch, Port Said).
- RAW/SIGNAL boundary enforced: the surface shows COUNTS ONLY with the
  coverage-loss caveat attached (panel row in Filings & flows, zone
  breakdown when loitering >0); per-vessel claims stay OFF until gate 1.
  Gate-1 plan filed (open_questions): enrichment of detections for
  publicly documented shadow vessels (OFAC SDN annexes, KSE lists) vs a
  size-matched random tanker cohort — identical coverage loss in both
  cohorts controls the terrestrial-AIS ambiguity. Gate-2 hypothesis:
  zone event rates lead tanker-rate names (FRO/STNG/TNK basket) +
  crude spreads; second-order reason the edge survives: maritime-intel
  vendors sell to compliance desks, not rate traders.
- Tests: 4 hermetic synthetic-archive cases (gap yes/no discrimination,
  both identity heuristics, loiter vs fast-transit, aggregation +
  caveat + wiring pins). 49/49 node; harness green.

## 2026-07-04 — [RESEARCH] Fusion hypotheses filed + collect-everything audit (Map v2.2 close-out)
- Three fusion hypotheses filed in open_questions with pairings,
  testable claims, and gate-1 ground truths (insider x facility at the
  verified SDI mills; generation shifts x operator tickers with the
  registry->ticker mapping as gate-1 work; ship-movement anomalies x
  tanker/retail names with TEU reports as truth). Logged, not built —
  per the directive.
- COLLECT-EVERYTHING verified in code for every layer: aircraft/
  vessels/trains stream to JSONL; Form 4 archives per poll; static
  reference data (plants/sites/zones) is git-versioned by doctrine;
  derived stats (shadowstats) intentionally not archived (recomputable
  from archived ingredients); imagery honestly not archived until the
  Sentinel-2 pipeline lands. Doctrine line added to open_questions.
- Map v2.2 directive fully executed across PRs #127-#131 + this docs
  close-out. STARVED: no.

## 2026-07-04 — [PRODUCT] Map UI v2.3: panel overflow root-caused + self-see harness rule + fullscreen (v1.0.59)
- ROOT CAUSE of the human-reported clipping: .vt-map-controls was
  position:absolute with NO bottom constraint, so the panel's
  max-height:100% resolved against an auto-height wrapper and never
  engaged — the panel grew past the viewport, clipped by the page's
  overflow:hidden, lower rows unreachable. Fix: wrapper now
  top+bottom-constrained (flex column, pointer-events pass-through
  under the panel); panel max-height:100% + existing overflow-y:auto
  now actually scroll.
- Panel restructured per directive: groups beyond the first fold
  (Facilities, Filings & flows, Signals) start COLLAPSED — headers
  visible, one tap to expand; the Form 4 FEED is fully removed from the
  panel (a feed doesn't belong in a layer-toggle sidebar) — the panel
  keeps one "Open filings view" button; the feed lives only in
  #/data/filings where columns wrap (word-break added), never clip.
  Dead panel-feed code + CSS removed per the dead-code policy.
- HARNESS SELF-SEE (approved amendment, DESIGN.md + wishlist): the
  harness now opens the panel via its own control, expands every
  collapsed group, and asserts panel-bottom-in-viewport, scroll-when-
  overflowing, every registry layer has a reachable row, every toggle
  scrollable-into-view and hit-testable (nothing covering it).
  PROVEN AGAINST THE BUG (loop-health rule 3 applied to the harness
  itself): A/B with the old CSS makes the harness FAIL with exactly the
  reported defect ("panel bottom 1084 past viewport 900"); fixed CSS
  passes. The harness gap that let this ship is closed by construction.
- FULLSCREEN MAP MODE: top-left 44px toggle hides the site nav (desktop
  top bar; phone top+bottom bars) for a true full-viewport map;
  sessionStorage-persisted; map.resize() on toggle. Verified
  mechanically at 390: nav display:none, map rect 0..innerHeight,
  persistence flag set; screenshot reviewed.
- Gates: harness green all widths WITH self-see active; node 49/49;
  CI python 114.
