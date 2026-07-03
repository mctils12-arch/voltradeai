# Experiment Log

Append-only. Newest at top. Never rewrite history (CLAUDE.md — MEMORY PROTOCOL).
Each entry: date · change · version tag · backtest result · hypothesis · (later) live-vs-backtest.

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
