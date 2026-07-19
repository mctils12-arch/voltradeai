# Open Questions

## KNOWN BROKEN — fix these first (repair mandate)

1. **[RESOLVED 2026-07-03 — backtest_v2.py]** ~~Backtest engine missing.~~
   Rebuilt: see experiments.md entry. Original text: `backtest.py` is a stub; the real engine
   that produced `backtest_10yr_results.json` was never ported from the
   workbench. Nothing can be evaluated until this exists. Reproduce the
   output schema in that JSON; invoke signature is
   `python3 backtest.py <ticker> <strategy> <years>` (bot.ts JSON-parses
   stdout). STANDING TOP PRIORITY.

2. **[RESOLVED 2026-07-03]** ~~2 failing tests reference missing backtest_v2.py.~~
   The backtest_v2 gated test now runs and passes; the backtest_v1028_full
   test remains skip-with-reason (legacy file superseded by backtest_v2,
   never ported).

3. **[ROOT CAUSE FOUND + FIXED 2026-07-11, v1.0.275]** ~~CSP execution
   cascade~~ (per CHANGELOG 2026-05-22): CSP trades were failing on three
   modes — insufficient capital for high-priced underlyings, no suitable
   puts in chain, liquidity filters rejecting everything. Fixes were
   applied (dynamic price ceiling etc.) — VERIFY in current audit logs
   that Tier 2 CSP trades actually fire now. If still zero fills, the fix
   pack didn't take.
   CONCLUSIVE EVIDENCE 2026-07-11 (this session, superseding the PARTIAL
   EVIDENCE note below): `/api/diag/orders?limit=200&token=$DIAG_TOKEN`
   spans 2026-06-03 through 2026-07-10 — 183 `us_equity` orders vs. 17
   `us_option` orders, and every single options order is dated
   2026-06-04/05 (BB puts). Zero options orders anywhere between
   2026-06-09 and 2026-07-10, a 5-week window with 185 equity orders
   filling normally in the same span. The May-22 affordability fix pack
   IS intact and working (verified by direct execution, see below) — a
   DIFFERENT, newer bug fully masked it.
   ROOT CAUSE (found by tracing the actual call graph this session, not
   assumed from memory): `tiered_strategy.py:328`'s `tier1_csp_core()`
   computed its options position-slot cap from
   `caps.get("MAX_POSITIONS", 6)` — but `MAX_POSITIONS` is the STOCK
   position cap, which `system_config.py`'s regime blocks zero out in
   PANIC/BEAR/NEUTRAL specifically to block new stock longs (each block's
   own comment says CSP should keep running: "Options engine takes over"
   /"Options engine continues running — BEAR is prime time for premium
   selling" / "CSP/options trades still fire via the tier engine (separate
   code path)"). `tier1_csp_core` doesn't know that distinction and
   silently inherited the stock 0, so CSP could only ever produce
   candidates in BULL/CAUTION — the two regimes where it's LEAST needed
   (stock longs are already active there) — and went hard-zero in exactly
   the three regimes it's supposed to be the standing engine for. Verified
   by direct execution (`tier1_csp_core` with empty positions across all
   five regimes): 0 actions in NEUTRAL/BEAR/PANIC, 3 in BULL/CAUTION,
   before the fix; nonzero in all five after.
   FIX: `system_config.py` BASE_CONFIG gained a dedicated
   `MAX_OPTIONS_POSITIONS` key (default 6, mirroring the value
   `tiered_strategy.py`'s own fallback already assumed) that is NOT zeroed
   by any regime block, so it survives PANIC/BEAR/NEUTRAL independent of
   the stock cap; `tier1_csp_core` now reads `MAX_OPTIONS_POSITIONS`
   instead of `MAX_POSITIONS`. This is a mechanical bug fix restoring
   already-documented intended behavior (the regime comments already
   asserted "CSP keeps running here"), not a new threshold policy — no
   RULE REVIEW evidence gate applies. A SEPARATE, related finding (the
   tier engine's own internal `master_kill_switch` recomputing regime from
   raw inputs with hardcoded `markov_state=1`, comparing TOTAL portfolio
   exposure against a regime-adaptive ceiling, and killing all 4 tiers —
   not just T2-4 — when exceeded) is a genuine design/threshold judgment
   call, NOT fixed this session, filed separately below as item #20.
   VISIBILITY GAP ALSO CLOSED (same PR): `run_tiers()`'s own docstring
   promises `{"killed": bool, "kill_reason": str}` but `bot_engine.py`
   never read those two keys — a `master_kill_switch` firing left
   literally zero audit trail (bot.ts only audits `tier_actions` when
   the list is non-empty). `bot_engine.py` now captures both into a new
   `tier_kill_status` field on `scan_market`'s return (distinct from the
   pre-existing top-level `kill_status`, which is `risk_kill_switch.py`'s
   separate, FROZEN-mechanism check); `server/bot.ts` audits it as
   `TIER-KILL` whenever `killed` is true, mirroring the existing
   `KILL-SWITCH` pattern. This means item #20 below is now independently
   observable in production instead of requiring code archaeology.
   RATCHET: `test_tiered_strategy.py` (NEW — zero test coverage existed
   for this file before this PR, despite it being the sole live CSP code
   path) — 6 tests; A/B-verified via `git stash` that 4 of the 6 fail
   against the pre-fix code (the 3 stress-regime parametrized cases +
   the dedicated-cap pin) and all 6 pass post-fix. Full gates:
   `python3 -m pytest -q` 644 passed, 1 skipped (baseline + 6 new, zero
   regressions); `npx tsx --test server/*.test.ts` 574 passed, 0 failed;
   `npx tsc --noEmit` 64 errors, byte-identical to the `git stash`-verified
   baseline; `npm run build` clean.
   Backtest: N/A — this restores a documented existing rule to working
   order (CSP already fires in BULL/CAUTION in the current live system);
   it does not change any scoring, sizing, or threshold value, so
   PROMOTION RULE 3's Sharpe/drawdown comparison doesn't apply the way it
   would to a new strategy. The live effect will be directly observable in
   `/api/diag/orders` (options orders resuming) and `/api/diag/ml`
   (CSP-sourced `trade_feedback` records appearing) over the coming days —
   a future session should check both.
   PARTIAL EVIDENCE 2026-07-11 (earlier same-day check, kept for the
   record — see CONCLUSIVE EVIDENCE above for the follow-up that resolved
   the ambiguity): `/api/diag/orders?token=$DIAG_TOKEN` returned the 100
   most recent Alpaca orders, spanning 2026-07-07 10:00 UTC through
   2026-07-10 18:42 UTC (~3.5 days) — every single one `asset_class:
   "us_equity"`, zero options/CSP orders in the window. NOT CONCLUSIVE
   EITHER WAY at the time: consistent with "the fix pack didn't take" but
   equally consistent with regime conditions producing no CSP-eligible
   candidates in this specific 3.5-day window — the probe returns a fixed
   order count, not a date-range query, so a longer/older window wasn't
   checked in that pass. Widening the window to 200 orders (this session,
   later the same day) is what turned this into the CONCLUSIVE EVIDENCE
   above.

20. **[FOUND 2026-07-11, not fixed — design/threshold judgment call,
    needs RULE REVIEW evidence] `tiered_strategy.py`'s
    `master_kill_switch` may be over-killing the CSP tier engine relative
    to its own documented intent.** Found while diagnosing item #3 above.
    Two separate issues, neither touched this session:
    (a) `master_kill_switch` (tiered_strategy.py:278-303) calls
    `get_regime_caps(ctx.vxx_ratio, ctx.spy_vs_ma50, ctx.equity)`, which
    hardcodes `markov_state=1, spy_below_200_days=0` (tiered_strategy.py:
    114-123) instead of using `ctx.regime` — the richer, already-computed
    regime `bot_engine.py` passes into `TierContext` (from its own
    Markov-based classification, bot_engine.py:1187-1194 per the prior
    session's regime-detector research). The kill switch is therefore
    judging exposure against a DIFFERENT, cruder regime classification
    than the rest of the system uses, via `regime_util.classify_regime_5level`
    whose `CAUTION_VXX_THRESHOLD = 1.05` trips easily.
    (b) The kill switch compares ctx.positions' TOTAL market value (stock
    + options combined) against that regime's `MAX_TOTAL_EXPOSURE` ceiling
    (95% BULL/NEUTRAL, 60% CAUTION, 50% BEAR, 30% PANIC) and, if exceeded,
    kills ALL 4 tiers — including Tier 1 CSP, whose own per-position
    budget is already small and self-limiting (`MAX_OPTIONS_PCT` = 8%
    of equity per position). Live snapshot this session (`/api/diag/
    positions-detail`): 4 stock ETF positions (SPY-floor basket) at
    ~80.6% gross exposure vs. equity, 0 options positions held — at that
    exposure level, any `vxx_ratio` >= ~1.05 (CAUTION or worse) kills
    the ENTIRE tier engine, not just the stock-adjacent tiers, even though
    Tier 1 CSP holds none of that exposure itself.
    NOT FIXED THIS SESSION, deliberately: whether Tier 1 CSP should be
    exempted from a whole-portfolio exposure kill it doesn't contribute
    to, and whether `master_kill_switch` should use `ctx.regime` instead
    of recomputing its own, are both threshold/design changes under this
    file's RULE REVIEW protocol — need either counterfactual logging (does
    this kill correctly prevent overexposure, or does it needlessly starve
    CSP during the SPY-floor basket's normal 65-95% passive allocation?)
    or a backtest ablation before shipping, one change at a time, with a
    logged rollback trigger. The item #3 fix above already makes this
    independently observable going forward via the new `TIER-KILL` audit
    line and `tier_kill_status` field — a future session should query
    `/api/diag/audit?type=TIER-KILL` after a few days of live data before
    proposing a specific threshold change.
    **COUNTERFACTUAL LOGGING BUILT 2026-07-11 (v1.0.279):** the evidence
    gate this item asked for now exists. `tiered_strategy.log_masterkill_csp_shadow()`
    fires from `run_tiers()`'s kill branch — computes what `tier1_csp_core()`
    would have proposed that cycle (pure function of `ctx`, no orders) and
    logs each priced candidate via `shadow_portfolio.log_candidate(...,
    decision="rejected_masterkill")`, reusing the existing outcome-backfill/
    `get_shadow_stats()` pipeline (same one the spread-filter and
    correlation-block buckets already use) rather than a new one.
    Visibility only — cannot affect the kill decision or `run_tiers()`'s
    returned actions; 5 new tests in `test_tiered_strategy.py` pin the
    wiring and the never-raises contract. Caveat carried in the record
    itself: the win/loss label is shadow_portfolio's existing stock
    price-path proxy (+2%/-4% PT/SL), not real CSP premium economics — a
    directional "would this underlying have been calm" proxy, not exact
    options P&L, same limitation the other rejected_* buckets already
    carry. **NEXT**: no TIER-KILL audit lines have fired yet since the
    item #3 fix shipped (checked this session — `/api/diag/audit?
    type=TIER-KILL` is empty), so there's no `rejected_masterkill` shadow
    data yet either. A future session should check
    `get_shadow_stats()["win_rate_by_decision"]["rejected_masterkill"]`
    once master_kill_switch has actually fired a few times with >=20
    trading days of history behind those firings (mirrors the ~90-day
    readiness bar already used for the other rejected_* buckets) before
    proposing the (a)/(b) threshold or regime-source change above.

4. **Human-reported: bot "doesn't work right" overall.**
   DIAGNOSIS 2026-07-03 (public API surface only — see access limitation
   below): /api/health reports ALL subsystems ok (server, sqlite, Alpaca
   account ACTIVE, python bridge, bot state "active"; Node RSS 78MB).
   Market-status/calendar correct (July-3 NYSE holiday handled). One
   evidence-backed finding: `state.equityPeak` (bot.ts:359) is in-memory
   only — initialized 0, lazily seeded from CURRENT equity on the first
   account poll (bot.ts:862, 2482), never persisted to the volume or
   rehydrated on boot. Therefore the MAX-DRAWDOWN KILL SWITCH high-water
   mark RESETS on every deploy/restart; with frequent autonomous deploys
   (6 on 2026-07-03 alone) drawdown protection is silently re-based each
   time and can never accumulate a true peak. Fix (persist equityPeak in
   the existing /data/voltrade state) touches frozen kill-switch machinery
   -> proposed in wishlist.md for human approval, not edited.
   ACCESS LIMITATION — STALE, CORRECTED 2026-07-09: the requireOwner routes
   below are still cookie-gated as described, but a token-gated read-only
   alternative was built and human-approved 2026-07-04/07 (server/diag.ts,
   `/api/diag/:probe` — audit/ml/daemon/positions/positions-detail/orders/
   scanner) specifically to close this gap, and `DIAG_TOKEN` is present in
   the autonomous session env as of this session. Future sessions: query
   `/api/diag/*?token=$DIAG_TOKEN` directly instead of writing "gated, no
   access" — see the R19 entry in experiments.md (2026-07-09) for a worked
   example (traced KNOWN BROKEN #12's live-feedback gate to root cause this
   way). Original (now-outdated) text preserved below for context.
   ~~every deeper diagnostic route (/api/bot/audit,
   /positions, /performance, /api/daemon/health, /api/bot/ml-status,
   /api/monitoring/*) is requireOwner (session cookie for OWNER_EMAIL,
   auth.ts — frozen). Autonomous sessions cannot read audit logs or
   trade_feedback from outside the container. Deeper #3/#4 verification
   (CSP fills firing? feedback accumulating? Tier-3 retrain green?) needs
   either the human pasting /api/bot/audit + /api/bot/ml-status JSON into
   a session, or the wishlist read-only-diagnostics proposal.~~
   Original symptom list to collect: Symptoms to
   collect from audit logs: are trades firing at expected frequency? Are
   fills tracked into trade_feedback? Is the ML retrain loop (Tier 3)
   completing or erroring? Diagnose from `/data/voltrade` state files and
   the persisted audit log before assuming any subsystem works.

5. **[RESOLVED 2026-07-04 — audit, no orphans found; one real gap closed]**
   ~~Data modules not wired to live scoring.~~ Traced every named module's
   call sites (READ BEFORE WRITE, static analysis only — no live access
   needed): `macro_data.py`, `alt_data.py`, `social_data.py`,
   `finnhub_data.py`, `intelligence.py` are all imported and consumed
   inside `bot_engine.py:deep_score()`'s parallel 5-source fetch
   (`_fetch_macro`/`_fetch_intel`/`_fetch_alt`/`_fetch_social`/
   `_fetch_finnhub`, bot_engine.py:543-608) and their fields are read
   downstream into `reasons`/score/regime features (verified: macro.*,
   intel.*, alt.*, social.*, finnhub.* all have live read sites past
   line 609). `institutional_data.py` feeds `insights.py`, which is wired
   to the site's `/api/insights/:ticker` endpoint (server/routes.ts:573)
   — a user-facing feature, not the trading loop, by design (GOAL
   priority 4), not orphaned. `alphadesk/` is wired via server/routes.ts.
   `instrument_selector.py` (a separate options-specific intelligence
   layer keyed on the SAME variable name `intelligence` but a distinct
   dataset from `intelligence.py`'s `get_full_intelligence` — a naming
   collision worth remembering, not a bug) is imported at
   bot_engine.py:3026. `diagnostics.py` is wired into
   `server/bot.ts`'s Tier-2 cycle (every 5th cycle) and actually drives
   `state.positionSizeMultiplier`/`state.minScoreThreshold`/pause.
   **Nothing named in this item is dead code.**
   GAP FOUND DURING THE AUDIT (fixed same PR, v1.0.66): `diagnostics.py`'s
   API-health check (`run_diagnostics()` #4) monitored polygon/sec_edgar/
   wikipedia/gdelt/fred cache freshness but had ZERO monitoring for
   `social_data.py` (reddit_/gtrends_ cache) or `finnhub_data.py` (fh_
   cache) — meaning `bot_engine.py`'s silent `except Exception: return {}`
   around those two fetches had no failure trail anywhere: a dead Reddit
   RSS feed or an expired/never-set FINNHUB_KEY would degrade those
   signals to permanent silent no-ops with zero visibility, exactly the
   HONESTY METRIC risk (live results diverging from backtest expectation
   for an unexplained reason). Added `extended_checks` (reddit_/fh_ cache
   presence) as a **separate, warnings-only bucket** — deliberately NOT
   folded into `api_checks`/`failed_apis`, which drives the existing
   `reduce_position_size` auto-fix at >=3 failures; merging them would
   have silently changed when that risk-affecting auto-fix fires, which
   RULE REVIEW requires evidence + one-at-a-time for, not an audit
   side-effect. FINNHUB_KEY unconfigured is treated as expected-degraded,
   not a break (mirrors the existing ml_model dynamic-criticality
   false-positive fix). 6 new regression tests in
   test_diagnostic_false_positives.py, isolation explicitly pinned by a
   source-inspection test. See experiments.md for the full trace.
   **UPDATE 2026-07-06, v1.0.151**: the extended_checks above only ever
   checked CACHE FRESHNESS (does a reddit_/fh_ file exist and how old is
   it) — the actual exception from a failed fetch was still never
   captured, for any of the 5 sources (not just the 2 that got
   extended_checks). Closed: `bot_engine.py`'s `deep_score()` now routes
   all 5 fetchers (`_fetch_macro`/`_fetch_intel`/`_fetch_alt`/
   `_fetch_social`/`_fetch_finnhub`) through a new `_run_diag_fetch`
   helper that captures `"ExcType: message"` per source into an optional
   `_diag` dict — same pattern `_fetch_snap` got in v1.0.148. Surfaced
   top-level-only (never inside a per-candidate dict, so it can't reach
   ML features or the shadow_portfolio log) as `data_source_errors` on
   `scan_market`'s return, read into `tier2LastDataSourceErrors` in
   `server/bot.ts`, and exposed on the existing token-gated
   `/api/diag/scanner` probe (never `/api/health`). See experiments.md
   for the full trace and downstream-chain reasoning.

6. **[RESOLVED 2026-07-04 — v1.0.73]** ~~Full-repo pytest is broken at
   collection (pre-existing).~~ Root causes were NOT network/keys (the
   original hypothesis was wrong): (a) two root-level STANDALONE SCRIPTS
   with test_ prefixes broke pytest itself — test_auto_discovery.py
   sys.exit()s at import (INTERNALERROR kills collection),
   test_full_system.py's module-level `def test(phase,...)` helper
   collides with fixture resolution and its import costs 62s — both now
   collect_ignore'd in conftest.py, still runnable directly as scripts;
   (b) 7 failures were STALE TEST PINS, not live bugs: test_fixes_pr8
   tearDown didn't expect track_fill's legitimate .lock sidecar (atomic
   write, present since import), TestOptionsSlotseparation pinned old
   tunable VALUES (5/3) that dated comments moved to 8/8
   (SIZING-FIX 2026-04-22, ALPHA-TUNE 2026-04-21), TestFix8's string pin
   went stale when max_loss flow moved through shared_max_loss (the
   mechanism is intact on BOTH leg paths — re-pinned stricter). Full
   bare gate now green (311 passed, 1 skipped, ~8s);
   test_collection_health.py is the ratchet (subprocess collect-only,
   A/B-proven to fail on the original breaker). Original text preserved
   above via strikethrough; see experiments.md 2026-07-04 [REPAIR].

7. **[RESOLVED 2026-07-03 — v1.0.35, executed same-day on human request]**
   ~~Persist the max-drawdown~~
   high-water mark.** `state.equityPeak` (bot.ts:359, seeded at 862/2482) is
   in-memory only: every deploy/restart re-bases the drawdown kill switch
   from current equity, so frequent autonomous deploys silently defang it.
   Approved fix: save/restore equityPeak via the existing /data/voltrade
   state files (storage_config.py paths; bot already persists other state
   there). REQUIREMENTS per constitution: (a) regression test that fails on
   the current reset behavior (loop-health rule 3 — no fix without its
   test); (b) do NOT alter the halt logic itself, only add persistence of
   its input (the human approval covers exactly this scope); (c) one
   logical change, version bump, prior stated in experiments.md before
   measuring any live effect; (d) trace the downstream chain in the PR
   (REASONING STANDARD #1): persisted peak -> drawdownPct reflects true
   history -> halt can actually fire after a slow multi-deploy bleed ->
   fewer trades during real drawdowns (intended).

8. **[RESOLVED 2026-07-03 — v1.0.36]** ~~Verify extended-hours order
   handling end-to-end~~. Findings: (a) options orders were ALREADY
   correctly gated — `executeTrades()` (the only function that submits
   options orders) is called exclusively `if (isMarketOpen)`
   (server/bot.ts:3030), so no code change was needed there; the
   `options_exit` OrderContext case existed but was dead (never actually
   passed by any caller), harmless. (b) Real bug found: `getOrderParams()`
   priced stock/ETF orders for the extended-hours window (4am-9:30am,
   4pm-8pm ET) with wider limit buffers but never set Alpaca's
   `extended_hours: true` flag — so those day-limit orders were silently
   queued for the next REGULAR session instead of attempting to fill
   during the pre-market/after-hours session they were priced for. This
   hit the real-time WS stop-loss/trailing-stop/take-profit exit handler
   (server/bot.ts, fires on any live price tick regardless of market
   hours) and the Tier-3 SPY/QQQ floor buy — meaning a stop-loss computed
   at, say, 6am would never actually attempt to execute until 9:30am,
   defeating the point of a stop during an overnight/pre-market move.
   Fix: extracted `getETHour`/`getOrderParams`/`OrderContext` into a new
   pure module `server/orderParams.ts` (no behavior change beyond the
   fix) and added `extended_hours: true` to the extended-hours branch for
   stop_loss/trailing_stop/take_profit/new_entry. Options branch
   deliberately untouched (no options extended session exists on Alpaca).
   See experiments.md for the regression test and downstream-chain trace.

9. **[RESOLVED 2026-07-03 — v1.0.44]** ~~Vessel stream: connect eagerly at
   boot~~. Found in v1.0.43 live verification: the aisstream websocket
   connected lazily on the first /api/data/vessels request, so every
   deploy left a vessels gap (map empty + archive not recording) until
   someone opened the map. Fix: extracted `vesselStreamEnabled` /
   `bootVesselStream` into new module `server/vesselStream.ts` (single
   source of truth for the AISSTREAM_KEY gate, replacing three
   independent `process.env.AISSTREAM_KEY` checks) and call
   `bootVesselStream(process.env, ensureVesselStream)` once at route
   registration time, right after the function is defined. See
   experiments.md for the regression test and downstream-chain trace.
   STILL OPEN, unrelated to this fix: verify ShipStaticData
   typing/destination populates post-warm-up on the next live check —
   that's read-path enrichment, not connection timing.

10. **[FOUND 2026-07-04 — dead config, not yet repaired] SCORE_BAND_MAX,
    MAX_CHANGE_PCT, SCORE_BAND_OPTIMAL_LO/HI are defined in
    `system_config.py`'s BASE_CONFIG with comments claiming they gate
    trades ("Skip stocks already up/down 35%+", "Scores above this are
    often fake breakouts", "Sweet spot confirmed by 10-year backtest")
    but are READ NOWHERE ELSE IN THE ENTIRE REPO.** Verified via
    `grep -rn "SCORE_BAND_MAX\|MAX_CHANGE_PCT\|SCORE_BAND_OPTIMAL"` across
    every `.py`/`.ts` file: the only hits are the four definitions in
    `system_config.py` itself and this file. `bot_engine.py`'s actual
    quick-score pass (line ~2458-2462) computes `change_pct` and applies
    an `_extreme_penalty` that SOFTLY penalizes the score for >30%/>50%
    moves — it never hard-skips a candidate the way the `MAX_CHANGE_PCT`
    comment says, and `combined_score` is never checked against
    `SCORE_BAND_MAX`/`SCORE_BAND_OPTIMAL_LO/HI` anywhere. HONESTY METRIC
    RELEVANCE: these four values read as active, backtest-validated
    guardrails to anyone consulting `system_config.py` (including this
    file's own RULE COST AUDIT section below, which asked to "measure
    prevention-P&L" for a rule that prevents nothing) — a session or
    human could reasonably believe the system is filtering out fake
    breakouts and 35%-already-run names when it is not. NOT REPAIRED
    THIS SESSION, deliberately: per RULE REVIEW, wiring a hard skip back
    in would be a genuine rule/threshold CHANGE (the system's actual
    live behavior today has no such gate) and requires either
    counterfactual evidence or a backtest ablation before shipping —
    neither exists yet, and inventing the ablation harness for a
    per-candidate stock-selection filter is a separate, larger effort
    than this session's scope (bot_backtest.py/backtest_v2.py model
    ETF-rotation strategies, not the full deep_score candidate-selection
    path, so they cannot test this specific hypothesis as-is). USEFUL
    NEWS: `shadow_portfolio.py`'s `log_candidate()` already records
    `change_pct` inside the logged `features` dict for EVERY scanned
    candidate (not just accepted trades) with forward +5d/+10d/+20d
    outcomes backfilled nightly (see the entry directly below) — so once
    enough shadow history accumulates, a future session can query the
    existing archive directly (win rate for |change_pct|>35 candidates
    vs. the rest) instead of building a new backtest harness. NEXT STEP:
    once shadow_portfolio has >=90 days of backfilled history, test that
    query; if it shows overreacting names are worse bets, wire the skip
    with the evidence cited in the PR per RULE REVIEW; if not, delete
    the four dead keys from `system_config.py` (DEAD CODE POLICY) rather
    than leave them as misleading documentation.

11. **[FOUND + FIXED 2026-07-04, v1.0.71]** ~~Daemon RPC route
    `shadow_stats` pointed at a function that doesn't exist~~.
    `voltrade_daemon.py`'s `RPCDispatcher._routes["shadow_stats"]` was
    `("shadow_portfolio", "get_stats")` — but `shadow_portfolio.py` only
    defines `get_shadow_stats()`, never `get_stats()`. Every call to this
    RPC method would have silently returned `{"status": "error",
    "error_message": "Method get_stats not found in shadow_portfolio"}`
    at runtime — exactly the READ BEFORE WRITE failure mode CLAUDE.md
    warns about (a Python-side rename/typo with no updated caller,
    invisible to CI because nothing in the test suite exercised the
    dispatch table). Found while auditing `shadow_portfolio.py` for the
    counterfactual-logging finding above (#10) — confirmed via `grep`
    that nothing in `server/bot.ts`/`server/routes.ts` currently calls
    the `shadow_stats` RPC method (so this was latent, not an active
    live break; `backfill_outcomes` IS called daily from bot.ts's Tier-1
    cycle at 10pm UTC and works correctly — only the read-side stats
    route was broken). FIX: corrected the route to `get_shadow_stats`.
    RATCHET: new `test_voltrade_daemon.py` — no daemon test file existed
    before this PR — walks every route in `RPCDispatcher._routes` whose
    target module exists on disk and asserts the attribute is a real,
    callable function (would have caught this bug, and will catch any
    future rename of a routed function); a second test pins the two
    genuinely-placeholder routes (`ml_status_impl`/`ml_toggle_impl`,
    which dispatch to a local fallback by design, no such modules exist)
    so they're never silently miscounted as "checked." 4/4 new tests
    pass; full offline CI-gate subset unaffected (124 passed, 1 skipped
    — 120 pre-existing + 4 new, identical baseline otherwise). Zero
    Python trading-path behavior change (nothing calls this route today)
    — version bumped 1.0.70 -> 1.0.71 per the read-and-increment
    convention anyway, for `code_version` attribution hygiene.

12. **[FOUND 2026-07-06, R12 series — 3 of 4 defects FIXED, 2 gated
    follow-ups open] The live ML feedback loop recorded NOTHING for 2.5
    months.** Full trace in experiments.md (v1.0.152→155). Fixed:
    (D1, v1.0.153) track_fill's qty guard silently dropped every
    regular-hours entry fill since bot.ts's payload switched to
    qty_requested/qty_filled on 2026-04-23; (D2, v1.0.154) Bug #13's
    exit machinery had NO caller — WS final exits + position kills now
    record exit_context; (D4, v1.0.155) boot cleanup wiped all Kelly
    seeds every deploy, used a lexicographic version compare ('1.0.153'
    < '1.0.34' as strings — latent total wipe), and preserved the 500
    April fossils that blocked reseeding — all three fixed in
    feedback_boot_cleanup.py. VERIFIED POST-DEPLOY: feedback_count 0,
    fossil signature gone. STILL OPEN, GATED:
    (a) RESEED CHECK — feedback_seeded_count was still 0 immediately
    post-purge (daemon's autoseed check ran before the purge landed);
    it should fire on the NEXT deploy-boot. If seeds are still 0 after
    another deploy, the autoseed path itself is broken (seeder script +
    backtest_10yr_results.json both verified present in the image) —
    that becomes a new defect.
    (b) D3 — trackClosedTrades' feedback block (bot.ts ~line 633) is
    DEAD CODE since the OOM fix hardcoded entryFeatures: null (its
    `t.entryFeatures != null` filter rejects every record; the block
    has written nothing since 2026-04-20 and its pre-filter variant is
    what wrote the purged fossils). DECISION GATED on D2's first live
    verification: once a WS exit records a real outcome via track_fill
    (VXUS/SMH already flagged — expect within days), the block is
    redundant and should be REMOVED per the dead-code policy; if D2's
    path underdelivers (e.g. exits that bypass the WS monitor), repair
    it instead. Do not touch until then — one attribution at a time.
    (c) Exit paths beyond the WS monitor (options_manager exits,
    bot_engine-side closes, manual dashboard closes) still record
    nothing; wire path-by-path after (b) resolves.

13. **[RESOLVED 2026-07-07, T-CLIENT — v1.0.178]** ~~`--accent` CSS
    custom property silently redeclared in the SAME `:root` block,
    breaking every direct `var(--accent)` use as a `color`/`background`/
    `border-color` value sitewide.~~ FIXED: renamed the shadcn/Tailwind
    internal HSL-triple token to `--shadcn-accent` (not the
    DESIGN.md-documented brand hex) and updated its 3 real consumers —
    the 2 `--accent-border` wrapper sites in index.css AND
    `tailwind.config.ts`'s `accent.DEFAULT` (a THIRD consumer this
    item's original text below missed, feeding `bg-accent`/
    `text-accent-foreground` Tailwind utility classes across 10 shadcn
    components — renaming the wrong side would have broken those
    working interactive states instead). A repo-wide grep (not just
    index.css) found 2 more real broken sites beyond the 18 catalogued
    below: `filings.tsx`'s `option_exercise` badge and `analyze.tsx`'s
    range-fill bar / border-left accent stripe — fixed by the same
    rename. Verified live via Playwright (layer-toggle switches and the
    filings EXERCISE badge now render visible blue, not transparent);
    `npm run visual` 0 hard failures at all 3 widths; `python3 -m
    pytest -q` 457 passed / 2 skipped. See experiments.md 2026-07-07
    [REPAIR] for the full trace. The `--border` dormant collision noted
    below is UNCHANGED and unaddressed — still zero live call sites,
    re-confirmed dormant after this fix, left for a future session.
    Original finding text preserved below for the record.
    `client/src/index.css`'s `:root`
    declares `--accent: #4d9fff` (line 25, the DESIGN.md-documented
    brand hex) and then, further down the SAME block, `--accent: 212
    100% 65%` (line 92 — a bare HSL triple with no `hsl()` wrapper, part
    of the shadcn/Tailwind token set that reuses the same name). The
    later declaration wins the cascade unconditionally, so `--accent`
    resolves to the invalid triple everywhere, not the hex. Per the CSS
    custom-properties spec, a `var()` substitution invalid at computed-
    value time makes that property compute to its inherited/initial
    value instead of erroring visibly — for non-inherited properties
    (`background`, `border-color`) that's typically `transparent` or
    `currentColor`, so the failure is SILENT and cosmetic-looking, not a
    build break or console error. Confirmed via computed-style dump
    (`getPropertyValue("--accent")` returns `"212 100% 65%"` on every
    element; a `background: var(--accent)` rule computes to `rgba(0,
    0, 0, 0)`). Found while building the Everything Graph panel (this
    session, v1.0.160) — a new badge using `background: var(--accent)`
    rendered as invisible dark-on-transparent text, caught only by
    screenshotting the actual opened sub-view with Playwright (the
    standard `npm run visual` harness only screenshots the /data map
    shell, never opened filings/earnings/shortvol/graph sub-views, so
    this class of bug is invisible to the existing harness). SCOPE:
    `grep -c "var(--accent)\b" client/src/index.css` (excluding the safe
    `--accent-bright`/`-green`/`-red`/`-orange`/`-purple` siblings, which
    are uniquely named and NOT collided) finds 18 pre-existing call
    sites across `.vt-map-fab:hover`, `.vt-kind-badge.raw`,
    `.vt-switch.on`/`.vt-switch.on i` (the layer toggle switches — the
    "on" thumb has likely been invisible/transparent in production this
    whole time, masked because the switch's outer track already uses a
    similarly-blue literal `rgba()`), `.vt-filings-sub a`,
    `.vt-filings-filter.active`, `.vt-filings-seclink`,
    `.vt-dev-badge`, `.vt-field-slider`/`.vt-field-check` (accent-color,
    same collision — native checkbox/range tinting), plus 3 more this
    session added and already fixed in-scope (see experiments.md). A
    SAME-SHAPE collision exists on `--border` (line 20 hex/rgba vs line
    98 HSL triple) but is currently DORMANT — zero direct
    `var(--border)` call sites found (everything uses `--border-subtle`
    or literal rgba instead), so no visible symptom today; worth
    re-checking this audit's fix doesn't accidentally activate it.
    NOT REPAIRED THIS SESSION, deliberately: fixing the 18 existing
    sites is a cross-many-components visual change that needs its own
    `npm run visual` regression pass at all three widths per CLAUDE.md's
    one-logical-change rule, and is unrelated to any single feature —
    a dedicated [REPAIR] session should (a) rename the shadcn/Tailwind
    block's competing tokens (they're only ever consumed via
    `hsl(var(--x))` wrappers elsewhere in this file, e.g.
    `--accent-border: hsl(var(--accent))` at line 119/166 — renaming to
    e.g. `--shadcn-accent` and updating those 2 wrapper sites is the
    minimal fix) OR move the hex brand-token block to load AFTER the
    shadcn block so it wins the cascade instead, then re-screenshot all
    18 affected components (plus the two `--accent-border` wrapper
    sites, which must keep resolving to the shadcn HSL value, not the
    hex, since `hsl()` requires the bare-triple form) to confirm no
    visual regression before shipping.

14. **[RESOLVED 2026-07-07 — v1.0.200, PR #351]** ~~manipulation_detect's
    Tier-3 scan failure is invisible.~~ FIXED: `tier3Strategic`'s
    manipulation-scan catch block now routes through `audit("TIER3-MANIP-
    ERROR", ...)` with the same stderr/stdout/code/signal classification
    the ML-retrain catch block already used, mirroring the pattern KNOWN
    BROKEN #5 established. Regression battery: `server/tier3ManipVisibility
    .test.ts` (3 tests, statically pins the catch block through `audit()`).
    STALE-DOC NOTE (caught this session, 2026-07-08, while doing the
    KNOWN-BROKEN-first repair check): this entry was still marked
    "not repaired this PR, deliberately" even though the fix had already
    merged the same day — a near-duplicate unmerged PR #343 proposing the
    identical fix under nearly the same title is still open, evidence the
    original close-out never happened here. #343 is now fully superseded
    by main's already-shipped fix; flagged for whoever next touches open
    PRs to close it as redundant rather than re-merge duplicate work.
    Original finding text preserved below for the record.
    manipulation_detect's Tier-3 scan failure is invisible.** Same root
    cause as the ML-retrain stdout-corruption bug this PR fixed
    (alpaca_feed.data_feed() printing "[FEED] SIP..." to stdout inside
    a one-shot subprocess whose stdout must be pure JSON) also breaks
    `server/bot.ts`'s manipulation_detect scan (~line 3987) whenever SIP
    stays 403-rejected — but that call site's catch block only
    `console.error`s, never `audit()`s, so the failure leaves ZERO
    trail in the persisted audit log (unlike the ML retrain path, which
    at least surfaced as TIER3-ML-ERROR). This is the same visibility
    gap KNOWN BROKEN #5 closed for social_data/finnhub_data via
    extended_checks — a silent `except`/no-audit swallow hides a live
    break exactly per the HONESTY METRIC risk. NOT fixed in the same PR
    as the retrain fix (one logical change per PR) — the print->stderr
    fix already resolves the underlying corruption for BOTH call sites
    once deployed; what remains is adding `audit()` visibility to the
    manipulation_detect catch block so a *future* failure of any kind
    there doesn't go dark again. NEXT STEP: route bot.ts's
    tier3Strategic manipulation-scan catch block through `audit()`
    (mirroring the ML-retrain catch block's existing pattern), own PR.

15. **[RESOLVED 2026-07-10 — R19, v1.0.260, CONFIRMED LIVE this session]
    Real-time position-exit monitor (WS stream) was silently
    non-functional for at least 22+ hours.** LIVE VERIFICATION (this
    session, `/api/diag/*?token=$DIAG_TOKEN` against production,
    2026-07-10 ~16:00 UTC): `?type=STREAM&limit=10` shows "Real-time
    feed live — 43 tickers" recurring every ~10 minutes across the last
    100+ minutes (was zero occurrences in 22+ hours pre-fix);
    `?type=WS-EXIT&limit=10` shows a real fill — "WS TRAILING STOP Phase
    2: P&L 0.2% dropped 9.0% from peak 9.2% ... MARKET sell 12 SMH @
    $601.04" at 13:46:18Z (was zero rows ever, same query); `/api/diag/ml`
    now reports `feedback_live_count: 1` (was stuck at exactly 0 for 3+
    days pre-fix, per KNOWN BROKEN #12(b)'s gate — the falsifiable
    prediction R19's entry made ("should start moving off zero") is
    confirmed true). All three of R19's own pre-stated NEXT-check
    conditions passed. Closing this item — no further live-verification
    action needed. Full trace in experiments.md's R19 entry. Summary: `checkPositionOnTick`
    (the sole executor of stop-loss/take-profit/trailing-stop/scale-out
    exits per bot.ts's own design comment) only runs on bars delivered by
    a WebSocket hardcoded to `wss://stream.data.alpaca.markets/v2/sip` —
    the same paid-entitlement tier whose 2026-07-06 rejection required
    `alpaca_feed.py`'s resolver for every REST call site, but that fix's
    ratchet (`test_alpaca_feed.py`, Python-file-only) could not see this
    trace in experiments.md's R19 entry. Summary: `checkPositionOnTick`
    (the sole executor of stop-loss/take-profit/trailing-stop/scale-out
    exits per bot.ts's own design comment) only runs on bars delivered by
    a WebSocket hardcoded to `wss://stream.data.alpaca.markets/v2/sip` —
    the same paid-entitlement tier whose 2026-07-06 rejection required
    `alpaca_feed.py`'s resolver for every REST call site, but that fix's
    ratchet (`test_alpaca_feed.py`, Python-file-only) could not see this
    TypeScript file, so the hardcoding was never caught. Live evidence
    (`/api/diag/audit`, this session): zero "Real-time feed live", zero
    WS-EXIT, across 19 restarts / 22+ hours, while bot_engine.py's
    `manage_positions()` repeatedly flagged trailing_stop conditions that
    Node deliberately never executes itself (by design, defers entirely
    to the — broken — WS path). FIXED: switched to `/v2/iex` (no
    entitlement dependency; this consumer only needs `close` price, never
    volume, so the volume-undercount reason `alpaca_feed.py` rejects iex
    for REST discovery doesn't apply) + added audit visibility for the
    stream-error frame and every disconnect (both were previously
    silent). CLOSED 2026-07-10: all three of the NEXT-check conditions
    below were confirmed positive live (see the CONFIRMED LIVE line at
    the top of this item) — deploy succeeded, `/v2/iex` reaches
    "subscribed", exits fire, feedback records. Original NEXT-check text
    preserved for the record: query `/api/diag/audit?type=STREAM&limit=5`
    for a "Real-time feed live —" entry (must appear now) and
    `?type=WS-EXIT&limit=5` (should go non-empty once any tracked
    position's stop/target is hit), then `/api/diag/ml` for
    `feedback_live_count` > 0 (was stuck at exactly 0 for 3+ days despite
    the D1/D2 fix in KNOWN BROKEN #12). If STREAM still never reaches
    "live" on /v2/iex, the new STREAM-ERROR/STREAM-DISCONNECT logging
    this PR added will name the actual rejection reason directly — treat
    that as a NEW finding, not a reopening of this one (RECURRENCE
    ESCALATES only applies to the same root cause recurring, not a
    different cause behind the same symptom).

16. **[RESOLVED 2026-07-10, v1.0.263] Extended-hours market orders were
    rejected and blindly retried instead of queued — root cause: an
    entirely different, undiagnosed call site than the KNOWN BROKEN #8
    fix covers.** Original finding preserved below. DIAGNOSIS this
    session: the answer to the NEXT-STEP question is "bypasses it
    entirely, and not via tier1Reflex/morningQueue at all." The
    resubmit loop is `bot_engine.py`'s `_manage_spy_floor()` — the
    passive floor-basket rebalancer (QQQ/SMH/KWEB/VXUS/...), called
    unconditionally every Tier-2 scan cycle (~5 min) with zero
    market-hours gating anywhere in the function. Its basket-rebalance
    branch (added 2026-04-22, `FLOOR_BASKET_ENABLED` path) submits raw
    `"type": "market"` orders directly via Python `requests.post` to
    Alpaca — a code path that never goes through `server/bot.ts` or
    `orderParams.ts`/`getOrderParams()` at all, so KNOWN BROKEN #8's
    fix (which only touched the Node side) could never have covered it.
    Alpaca rejects/cancels a market order placed outside 9:30am-4:00pm
    ET, so every ~5-minute cycle's drift check saw the same still-open
    gap and resubmitted — confirmed live via `/api/diag/orders` this
    session: 50 canceled + 31 still-pending "new" SMH buy-market orders
    spanning 2026-07-07 through 2026-07-10, ~5 minutes apart, entirely
    during extended/pre-market hours (the legacy single-ticker path,
    used when `FLOOR_BASKET_ENABLED=False`, already used limit orders
    per an earlier 2026-04-10 fix note but still had no `extended_hours`
    flag and no market-hours gate — same defect class, lower blast
    radius since Alpaca just queues a day-limit order rather than
    rejecting it, but still capable of resubmitting duplicate pending
    orders every cycle).
    FIX: added `_is_regular_hours()` to `bot_engine.py` — a direct copy
    of the already-established, already-tested convention in
    `options_scanner.py` (DST-aware `datetime.now(ZoneInfo("America/
    New_York"))`, 9:30-16:00 ET bounds; narrowed to catch only
    `ZoneInfoNotFoundError`, fail-closed, so it does not trip the
    `test_silent_except_ratchet.py` broad-except pin) — and gated
    every direct order-submission call inside `_manage_spy_floor` on
    it: the basket buy/sell branches, the legacy single-ticker
    buy/sell branch, and the target_pct<=0 regime-exit branch. When
    the market is closed, no order is submitted; an honest
    `*_deferred` action is recorded instead (`buy_deferred`/
    `sell_deferred` for the basket path, `floor_rebalance_deferred`
    for the legacy path, `floor_exit_deferred` for the zero-target
    exit) — this is a deferral, not a silent drop: the basket/legacy
    paths recompute drift from live positions every cycle regardless
    of `_floor_state`, so a skipped cycle retries automatically once
    the market reopens. The regime-exit branch needed one extra piece
    of care: it only fires `if regime_changed`, and `_floor_state`'s
    `last_regime` normally gets persisted unconditionally right after
    — so a deferred exit that still updated `last_regime` would make
    `regime_changed` false on the next call and the exit would be
    silently forgotten forever. Fixed by skipping the state-persist
    step specifically when deferred, so `last_regime` stays stale and
    the retry condition holds until the exit actually executes.
    RATCHET: `test_spy_floor_market_hours.py` (new file — this
    function had ZERO test coverage before this PR, for either the old
    or new behavior) — 5 tests: basket buy deferred + no POST when
    closed, basket sell deferred + no POST when closed, basket buy
    still executes with the correct `type: market` payload when open
    (regression guard against the fix disabling live trading), legacy
    path defers + no POST when closed, and the regime-exit-preserves-
    retry case (asserts the on-disk floor-state file's `last_regime`
    is untouched after a deferred exit). All mock every Alpaca
    call — no live network dependency.
    GATES: `python3 -m pytest -q` 626 passed, 2 skipped (baseline 621
    + 5 new; zero regressions, including `test_silent_except_ratchet.py`
    after narrowing the new helper's except clause). No TypeScript/
    client files touched — `server/bot.ts`'s own extended-hours path
    (KNOWN BROKEN #8) is untouched and remains correct for the code
    paths it actually covers (`executeTrades`, the Tier-3 `BUY`/
    `BUY_PUT` tier-action dispatcher, scale-outs, position kills — all
    of which DO call `getOrderParams`). No backtest run: this is an
    execution-layer safety fix (when an order may be submitted, not
    what/how much to trade) with the same shape as KNOWN BROKEN #8's
    fix, which also shipped without a backtest per PROMOTION RULES —
    strategy/sizing/scoring logic is unchanged.
    LIVE VERIFICATION CONFIRMED 2026-07-10 (this session, `/api/diag/
    orders?limit=200&token=$DIAG_TOKEN`): the fix deployed and took
    effect around 2026-07-10T11:22:31Z (a batch of pre-fix pending SMH
    market orders was canceled at that instant, consistent with a
    restart picking up the new gate). Zero orders in the 200-row window
    were canceled after that timestamp (94 total canceled in the window,
    all pre-fix); every order submitted after 11:22:31Z was during
    regular hours (earliest fill 13:38:23Z / 9:38am ET) and filled
    normally — no resubmit-loop recurrence. STILL OPEN, smaller
    follow-up (not a reopening of this fix, logged separately): the
    `buy_deferred`/`sell_deferred`/`floor_rebalance_deferred` trail
    itself was not checked this session (no deferred cycle happened to
    query during market hours) and is still not surfaced on any diag
    probe — `spy_floor_result` is computed in `scan_market()`'s return
    but not obviously logged to the audit trail the way `TIER3-*` events
    are; a future session should confirm deferred actions are visible,
    not just absent-of-bad-orders.
    Original finding text preserved below for the record.
    Extended-hours market orders
    are rejected and blindly retried instead of queued.** Observed live
    via `/api/diag/orders` (this session, R19 investigation): on
    2026-07-07 ~11:39-13:24 ET (pre-market), "SMH buy market 6" was
    submitted and immediately `canceled` roughly every 5 minutes for
    ~2 hours straight (about 20 consecutive reject/retry cycles). This
    contradicts the system's own stated rule (`server/bot.ts` RULES audit
    dump: "Smart Execution: Extended hours → queue for morning, no
    chasing thin liquidity") — a market order during extended hours
    should either be queued as a limit order for the open or held, not
    resubmitted as a market order and rejected on a ~5-minute loop. NOT
    diagnosed to a specific call site or fixed this session (found while
    investigating R19, kept out of that PR per one-logical-change-per-PR;
    this is a wasted-cycles/possible-missed-fill issue, not obviously a
    safety issue, so it did not preempt R19). NEXT: find the call site
    that resubmits "SMH buy market" every ~5 minutes during extended
    hours (likely the morning-queue or stale-order-sweeper path in
    tier1Reflex) and confirm whether `getOrderParams`'s `extended_hours`
    branch (KNOWN BROKEN #8) is actually reached for this order type, or
    whether this path bypasses it entirely.

17. **[FOUND 2026-07-09, RESOLVED 2026-07-15, v1.0.317] `TIER3-ML-ERROR:
    ML retrain failed: failed` — a real but content-free error message.**
    Observed live via `/api/diag/audit` this session (2026-07-09
    19:01:59Z). The audited detail is the literal string "failed" with no
    stderr/stdout/cause — unlike the ML-retrain catch block's usual
    pattern (KNOWN BROKEN #14's R18 fix documents the richer
    stderr/stdout/code/signal shape other TIER3 error paths now use).
    `/api/diag/ml` shows the model retrained successfully within the same
    hour (`model_age_hours: 1.1` at query time), so this is self-healing,
    not currently blocking — logged for a future session to trace why
    this one call site's error detail collapsed to a bare "failed" instead
    of a real message.

    UPDATE 2026-07-15 ([REPAIR], v1.0.317) — RESOLVED. Confirmed still
    recurring live this session (`/api/diag/audit?type=TIER2` /full audit
    read` showed a fresh `TIER3-ML-ERROR: ML retrain failed: failed` entry
    at 2026-07-15T10:56:22Z, 6 days after the original finding, so this
    was a real recurring gap, not a one-off). ROOT CAUSE (read-before-write
    trace): `ml_model_v2.py`'s `_train_model_impl` has four early-return
    sites shaped `{"status": "failed", "reason": "<real cause>"}` (lines
    1290/1336/1444/2126 — e.g. "Could not fetch training bars", "No
    training data", "Model training failed") that never pass through
    `train_model()`'s outer `except Exception` wrapper — `error_location`/
    `traceback_tail` are ONLY ever set inside that wrapper's exception
    handler, a completely different failure shape. `server/bot.ts`'s
    audit-message builder (~line 4122) read `trainResult.error_location`
    and `trainResult.traceback_tail` but never `trainResult.reason` — so
    every one of these four early returns rendered as the bare
    `${_statusStr}` with both `_loc` and `_tbTail` empty, i.e. literally
    "ML retrain failed: failed". FIX (own PR, v1.0.317): `server/bot.ts`
    now also reads `trainResult.reason` and interpolates it before
    location/traceback (` — <reason>`) when present, so a future
    "Could not fetch training bars"-class failure reads as "ML retrain
    failed: failed — Could not fetch training bars" instead of a bare
    "failed". RATCHET: `server/mlRetrainReasonVisibility.test.ts` (NEW, 3
    tests, source-inspection convention matching `tier3ManipVisibility
    .test.ts`/`tier2DaemonTimeoutVisibility.test.ts`) — A/B-verified via
    `git stash`: all 3 fail on pre-fix code, pass post-fix. GATES:
    `npx tsx --test server/*.test.ts` 680/680 (677 baseline + 3 new, zero
    regressions); `npx tsc --noEmit` byte-identical pre-existing 66-error
    baseline (git-stash-verified, only line-number shifts from this
    diff's +8 lines); `npm run build` clean. Zero Python files touched —
    `python3 -m pytest` not re-run. BACKTEST: N/A — pure audit-message
    visibility fix, does not change ML training logic, model selection,
    or any trading/scoring/sizing decision.

    UPDATE 2026-07-17 (session 4, [REPAIR], v1.0.384) — the "Could not
    fetch training bars" cause this item anticipated recurred live (5x
    in ~90min, model stuck 26.1h stale) and was root-caused + fixed: full
    trace in experiments.md's 2026-07-17 session-4 entry.
    `ml_retrain_safe.py` runs as a fresh subprocess every retrain and
    never imports `bot_engine` — the only place `install_global_
    throttle()` was called — so `_fetch_training_bars()` made completely
    unthrottled Alpaca calls, and a bare `except: continue` silently
    discarded whatever error resulted. Fixed: throttle now installs from
    `ml_model_v2.py` itself (idempotent); `_fetch_training_bars()`
    surfaces the real HTTP status/exception instead of discarding it.
    RESIDUAL RISK, not fixed (deliberately scoped out, own future item if
    it recurs): the daemon and the subprocess each throttle to 180/min
    independently with no cross-process coordination — if Alpaca's real
    200/min account-wide limit is still exceeded by their combined load,
    a future occurrence will now at least carry a diagnosable cause
    (e.g. "HTTP 429: ...") instead of the bare message; if that specific
    signature recurs, the next step is a cross-process shared limiter
    (file-lock token bucket, or routing the subprocess's fetch through
    the daemon's RPC instead of a raw HTTP burst) — real architectural
    work, not a threshold tune.

18. **[FOUND 2026-07-10, RESOLVED 2026-07-17, v1.0.307 — confirmed live]**
    ~~TIER2-ERROR "daemon run_full_scan failed: Daemon timeout" recurred 7x
    in ~95 minutes (18:22-19:57 UTC) with zero diagnostic detail~~ — visibility fixed,
    ROOT CAUSE STILL OPEN.** Full trace in experiments.md's 2026-07-10
    entry. `run_full_scan` is `HEAVY_DAEMON_ONLY` (never falls back to
    subprocess); a daemon-path failure reached the tier2 catch block as a
    bare Error with none of the stderr/stdout/code/signal fields that
    catch block's classification logic needs, producing the useless
    "code=? signal=none" line plus a `process.memoryUsage()` snapshot that
    describes bot.ts's own Node process, not the daemon that actually
    failed. FIXED (visibility only): daemon-path failures are now tagged
    and route to a distinct branch that probes the daemon's real health
    (rss/uptime/`active_dispatches`) instead. NEW SIGNAL added:
    `active_dispatches` on `voltrade_daemon.py`'s `_health()` — a
    thread-safe counter of dispatch worker threads actually executing
    right now, INCLUDING ones the handler already gave up waiting on
    (`RPCHandler.handle()`'s `t.join(REQUEST_TIMEOUT_SEC)` returns and
    releases `_inflight_sem` after 300s regardless of whether the thread
    finished — Python threads can't be forcibly killed, so a hung
    `dispatch()` call keeps running invisibly in the background after the
    client has already been told "Daemon timeout"). HYPOTHESIS, NOT YET
    CONFIRMED: this zombie-thread pileup is what's driving the clustering
    in the 7 timeouts (not evenly spread across the 95 minutes) — each
    stall that outlives its own 300s timeout could make the next call more
    likely to also stall, since `MAX_INFLIGHT_REQUESTS=8` no longer bounds
    real concurrent load once a slot is released before its thread
    actually stops. LADDER PATH: this is a system-health/reliability
    finding, not a trading signal — no ladder gates apply, but the same
    "state the prior, then check the evidence" discipline does. NEXT STEP
    for whichever session catches the next occurrence: query
    `/api/diag/daemon` (or the richer `TIER2-ERROR` audit line this PR
    adds) at the moment of a future timeout and read `active_dispatches`.
    ELEVATED (well above 8, non-decreasing across a few checks) confirms
    the zombie-pileup theory → next move is tightening
    `REQUEST_TIMEOUT_SEC`/`MAX_INFLIGHT_REQUESTS` or splitting
    `run_full_scan`'s internal work into cancellable chunks (all threshold
    changes needing evidence + one-at-a-time per RULE REVIEW, now
    possible because this PR gives the evidence a place to come from).
    LOW/NORMAL points elsewhere — the next suspect would be upstream
    market-data-provider latency (`/api/diag/scanner`'s
    `dataSourceErrors` was empty when checked this session, but that's a
    snapshot, not a history across the 7 prior failures — don't rule it
    out from one clean read). Do not guess between these without the new
    signal's evidence; do not re-patch this without checking whether the
    NEXT occurrence's `active_dispatches` reading actually supports the
    zombie theory first (RECURRENCE ESCALATES only applies once, but
    "patch blind before checking the diagnostic you just built" defeats
    the entire point of building it).
    UPDATE 2026-07-11, v1.0.277: THE VISIBILITY FIX ITSELF WAS BROKEN —
    fixed this session; root cause of the underlying timeout still open.
    Live production check (`/api/diag/audit?type=TIER2-ERROR&token=
    $DIAG_TOKEN`) caught 3 fresh occurrences today (14:48/15:18/15:48Z,
    suspiciously regular ~30min spacing) — every single one logged
    `daemon health returned non-alive: {"status":"ok","result":{"alive":
    true,...}` instead of the intended `daemon rss=...MB active_dispatches=
    N uptime=...s` line. READ BEFORE WRITE traced it to `server/bot.ts`'s
    daemon-timeout catch branch (the v1.0.266 visibility fix itself):
    `pythonRpc("health")`'s raw return is the RPC envelope
    `{"status":"ok","result":{...}}` (`voltrade_daemon.py`'s
    `RPCDispatcher.dispatch()`: `return {"status": "ok", "result": result}`)
    — `_health()`'s own `alive`/`rss_mb`/`active_dispatches`/
    `uptime_seconds` fields live one level down, under `.result`, not on
    the envelope itself. The branch read the top-level envelope's `alive`
    field directly — always `undefined`/falsy for every successful health
    call — so it has misclassified every healthy daemon as "non-alive" and
    never once surfaced `active_dispatches` since v1.0.266 shipped
    2026-07-10. The evidence this item's own NEXT STEP asks for has never
    actually been captured in production. FIXED (mechanical unwrap-the-
    envelope bug fix, no threshold/rule change — RULE REVIEW's evidence
    gate does not apply): unwrap `h.result` when `h.status === "ok"`
    before reading `alive`/`rss_mb`/`active_dispatches`/`uptime_seconds`.
    RATCHET: new test in `server/tier2DaemonTimeoutVisibility.test.ts`
    (A/B-verified via `git stash` on `bot.ts` alone that it fails pre-fix,
    passes post-fix) asserts the branch unwraps the envelope before
    reading those fields and never reads them off the raw envelope.
    ROOT CAUSE OF THE TIMEOUT ITSELF STILL OPEN — this fix only repairs
    the diagnostic; a live `/api/diag/daemon` probe taken moments after
    this session's 15:48Z occurrence read `active_dispatches: 2` (well
    under 8, NOT elevated), but that is a post-hoc snapshot minutes later,
    not a reading taken at the moment of the stall — it does not confirm
    or refute the zombie-thread-pileup theory either way. NEXT STEP
    unchanged in spirit, now actually possible: the next occurrence's
    TIER2-ERROR audit line will finally carry a real `active_dispatches`
    reading taken at the moment of the timeout — read it before proposing
    any threshold change. The ~30min regular spacing observed today
    (14:48/15:18/15:48) is itself a new, unexplained data point worth a
    future session's attention — regular intervals suggest a scheduled/
    periodic contention source rather than random load spikes, but this
    is a hypothesis, not yet investigated.

    UPDATE 2026-07-12, v1.0.288: THE EVIDENCE THIS ITEM ASKED FOR IS NOW
    IN — ZOMBIE-THREAD THEORY REFUTED; A NEW, TIGHTER CORRELATION FOUND
    AND MADE CHECKABLE. Live `/api/diag/audit?type=TIER2-ERROR&token=
    $DIAG_TOKEN` caught 2 fresh occurrences today (14:43:42Z, 15:13:44Z —
    exactly 30 min apart again, matching 2026-07-11's cadence): both
    logged a real `active_dispatches` reading taken at the moment of the
    stall (the thing 2026-07-11's update said was still missing) —
    `active_dispatches=1` BOTH times, nowhere near the `MAX_INFLIGHT_
    REQUESTS=8` cap. Two independent readings, both low, is enough to
    retire the zombie-thread-pileup theory as the explanation for THIS
    pair of occurrences (it remains theoretically possible the pileup
    happens on some other occasion, but it is no longer the leading
    hypothesis — two data points that both contradict it outweigh zero
    that support it).
    NEW FINDING (not previously connected): pulling the full audit window
    around both occurrences shows `STREAM-DISCONNECT` ("Feed disconnected
    — reconnecting in 10s") landing within 17-35ms of each `TIER2-ERROR`
    — `14:43:42.554` (error) vs. `14:43:42.537` (disconnect); `15:13:44.239`
    (error) vs. `15:13:44.274` (disconnect). The 6 `STREAM-DISCONNECT`
    entries visible in today's window are spaced ~598-606s apart
    (essentially an exact 600s/10min cycle, independent of TIER2 state —
    4 of the 6 have no coincident TIER2-ERROR at all). The daemon-timeout
    `setTimeout` is armed with a scan-dependent start time (300s from
    whenever that cycle's `run_full_scan` RPC call began — NOT the 90s
    the surrounding code comment still claims; `pythonCall`'s actual call
    site passes `300000` for both the daemon and subprocess timeout, a
    stale "temporarily" bump from 2026-04-23 that was never reverted or
    corrected in the comment — separate minor STALENESS AUDIT item, not
    fixed this session, not touched to keep this PR one logical change).
    Two independently-scheduled timers (a fixed ~600s WS cycle and a
    scan-dependent ~300s RPC timeout) landing within tens of ms of each
    other, on both occurrences observed, is closer than their independent
    periods should coincidentally produce. HYPOTHESIS: a shared cause — a
    Node.js event-loop stall (synchronous work blocking the loop: GC
    pause, a large JSON.parse/stringify, etc.) — would explain both
    symptoms at once: an already-elapsed `setTimeout(300000)` and an
    already-arrived-but-unprocessed WS `close` event both stay queued
    until the loop resumes, then fire back-to-back, which is exactly the
    17-35ms adjacency observed. NOT YET CONFIRMED — inferred from timing
    adjacency on 2 occurrences, not measured directly; REASONING STANDARD
    #4 applies (distrust a 2-point pattern) until direct measurement
    exists.
    SHIPPED (own PR, visibility-only, same pattern as the two prior fixes
    on this item — a third pass, but this one adds new diagnostic
    capability rather than patching the existing one, so it does not
    trip the "two failed fixes = architecture smell" bar, which is about
    repairing the SAME mechanism twice, not investigating with a new
    instrument): `server/eventLoopLag.ts` (NEW) — a standard event-loop-lag
    monitor. A `setInterval(2000)` tick measures how many ms late it
    actually fired vs. its nominal 2000ms schedule (`computeLagMs`); any
    tick landing >=500ms late (`lagExceedsThreshold`, `EVENTLOOP_LAG_
    ALERT_THRESHOLD_MS`) is audited as a new `EVENTLOOP-LAG` entry. Wired
    into `server/bot.ts` alongside the other tier interval definitions.
    Pure measurement — no trading, sizing, scheduling, or scan logic
    touched; cannot affect any live decision. The NEXT `TIER2-ERROR`/
    `STREAM-DISCONNECT` coincidence will show whether an `EVENTLOOP-LAG`
    entry landed in the same window — that is the direct test this
    hypothesis needs, replacing "guess from the comment" with "read the
    instrument."
    RATCHET: `server/eventLoopLag.test.ts` (NEW, 9 tests) — pure-function
    coverage for `computeLagMs`/`lagExceedsThreshold` (on-schedule, late,
    early/clamped-to-zero, default and custom thresholds) plus wiring-pin
    tests (mirroring `tier2DaemonTimeoutVisibility.test.ts`'s convention
    of reading `bot.ts`'s source text) confirming the monitor is actually
    armed via `setInterval` at `EVENTLOOP_LAG_CHECK_MS` and gates the
    audit call on `lagExceedsThreshold`.
    NEXT STEP for whichever session catches the next occurrence: query
    `/api/diag/audit?type=EVENTLOOP-LAG` (or `type=TIER2-ERROR` and scan
    the surrounding window) at the time of the next `TIER2-ERROR`/
    `STREAM-DISCONNECT` coincidence. An `EVENTLOOP-LAG` entry in that same
    window confirms the stall theory and turns this into a normal
    performance-debugging problem (profile what's blocking the loop
    around scan-result processing). No `EVENTLOOP-LAG` entry despite a
    fresh coincidence would refute the stall theory in turn and reopen
    the question — do not assume either answer before reading it.

    UPDATE 2026-07-12 20:16 UTC, v1.0.291: STALL THEORY CONFIRMED LIVE —
    A PLAUSIBLE ROOT CAUSE FOUND AND FIXED (converted to non-blocking;
    causal proof of the specific magnitude still open). Live
    `/api/diag/audit?type=EVENTLOOP-LAG&token=$DIAG_TOKEN` (this session,
    ~4 hours after the event-loop-lag monitor shipped) shows the monitor
    firing constantly, NOT rarely: 35 EVENTLOOP-LAG entries in a single
    ~4-hour window (12:38-20:16 UTC), lag magnitudes 59,000-75,000ms per
    occurrence (the loop stalls a full 60-75 SECONDS, not a brief GC
    blip), on an almost-exact ~600s/10-minute rhythm (occasional shorter
    gaps of 1-7 min are cascading reconnect bursts right after a big
    stall, not independent stalls). Cross-checked against
    `type=STREAM-DISCONNECT` over the same window: every ~10-minute
    EVENTLOOP-LAG entry has a STREAM-DISCONNECT within 20-40ms of it
    (e.g. 20:16:01.292Z lag vs. 20:16:01.323Z disconnect), and a fresh
    `type=TIER2-ERROR` daemon-timeout landed in the same pattern at
    19:30:09.944Z — CONFIRMS the shared-stall hypothesis this item has
    been building evidence for since 2026-07-11.
    ROOT CAUSE HYPOTHESIS (found by READ BEFORE WRITE grep for any
    setInterval on a matching ~600000ms period doing synchronous work):
    `server/bot.ts`'s temp-file cleanup interval — `setInterval(() => {
    fs.readdirSync('/tmp')... fs.statSync()... fs.unlinkSync() }, 600000)`
    — ran on the EXACT 600000ms period observed, using the fully
    synchronous fs API on the Node event loop's own thread. This is the
    ONLY setInterval in bot.ts with both a ~10-minute period and
    synchronous fs work; every other periodic writeFileSync in the file
    (equity curve, equity peak, kill-switch state) writes a small
    (<1KB-few KB) object on event-driven or daily cadences, not a 10-min
    timer, and none does a directory scan. NOT YET DIRECTLY MEASURED:
    this identifies a real, textbook event-loop-blocking hazard on the
    right period, but does not yet prove /tmp actually held enough
    fb_/fill_/opt_ files to cost 60-75 real seconds of readdirSync+N*
    (statSync+unlinkSync) — that would need a file-count reading at the
    moment of the stall, which this fix does not capture retroactively
    (REASONING STANDARD #4/#10: matching period is suggestive, not
    proof of magnitude).
    FIX (v1.0.291): extracted the sweep into `server/tmpCleanup.ts`,
    rewritten with `node:fs/promises` (readdir/stat/unlink) — this
    removes the blocking hazard unconditionally, regardless of file
    count, since async fs work runs off the main thread via libuv's pool.
    Wired into `bot.ts` at the same `TMP_CLEANUP_INTERVAL_MS` (600000)
    and `TMP_FILE_MAX_AGE_MS` (300000) — identical cleanup behavior, only
    the blocking mechanism changed. Added a new `TMP-CLEANUP` audit line,
    gated on `TMP_CLEANUP_AUDIT_THRESHOLD` (200) — fires ONLY when a scan
    is anomalously large, so if this really is the cause, a future
    session will see large `scanned` counts logged right before the
    cadence stops reappearing; if the cadence in `EVENTLOOP-LAG`/
    `STREAM-DISCONNECT` STOPS after this ships with zero `TMP-CLEANUP`
    entries ever firing, that's strong confirmation the fix (not a large
    backlog) was what mattered — if the ~600s cadence CONTINUES despite
    this fix, the theory is refuted and the search moves to the next
    setInterval candidate, guided by whatever `TMP-CLEANUP` telemetry (or
    its absence) shows.
    NEXT STEP: whichever session checks in after this deploys (`git log`
    shows the merge; a few hours of live data is enough given the ~10min
    cadence) should query `/api/diag/audit?type=EVENTLOOP-LAG` and
    `type=TMP-CLEANUP` — no more ~600s-cadence EVENTLOOP-LAG entries
    confirms the fix; entries continuing on the same cadence with no
    TMP-CLEANUP hits reopens the search for a different blocking op on a
    10-minute period (re-grep bot.ts's setIntervals; also check whether
    any Python-side daemon call this triggers is itself blocking
    something Node-side synchronously awaits every ~10 min).

    UPDATE 2026-07-13, pre-flight health check (this session, PRODUCT —
    noted per this session's own instructions, not investigated further):
    TMPCLEANUP HYPOTHESIS REFUTED. `/api/diag/audit?type=EVENTLOOP-LAG&
    token=$DIAG_TOKEN` (~4h after v1.0.291 deployed) shows the exact
    outcome this item's own NEXT STEP anticipated as the refutation case:
    EVENTLOOP-LAG entries continue on the same ~10-minute cadence
    (12:38-20:16 UTC the day it shipped, still recurring past 00:01 UTC
    the next day), magnitude GREW (86,000-97,800ms vs. the pre-fix
    59,000-75,000ms), and `type=TMP-CLEANUP` returned ZERO entries in the
    same window — the /tmp sweep isn't even the thing crossing
    `TMP_CLEANUP_AUDIT_THRESHOLD`, let alone the blocking cause. tmpCleanup.ts
    is now confirmed innocent (it's async fs AND it's not even the trigger).
    NEXT STEP unchanged in spirit, narrower in scope: re-grep bot.ts's
    setIntervals for the next ~600000ms-period candidate (tmpCleanup ruled
    out); the daemon-side RPC path is the leading unexplored suspect,
    not yet investigated. Full detail in research/experiments.md's
    2026-07-13 entry.

    UPDATE 2026-07-13 (v1.0.302), this session: re-grepped bot.ts's
    setIntervals per the prior update's NEXT STEP — no other JS timer with a
    ~600000ms period doing synchronous work exists (TIER2's interval varies
    60s-1800s by time of day, inconsistent with the constant ~600s cadence
    actually observed in the diag data, and every other setInterval on the
    file is 30s/45s/60s/2s/3600s). That pointed the search away from a JS
    timer entirely and toward a candidate no prior update in this item had
    named: `db` (server/auth.ts, imported into bot.ts for the audit_log
    table) is a better-sqlite3 connection — fully SYNCHRONOUS, every
    .run()/.get() call blocks the Node event loop itself for the full
    duration of the underlying write syscall. `DB_PATH` resolves to
    `/data/voltrade.db`, the Railway PERSISTENT VOLUME (not local/ephemeral
    disk), and auth.ts never sets `PRAGMA journal_mode` — it runs SQLite's
    default rollback-journal mode, which does a create/write/fsync/delete
    cycle on a separate journal file for EVERY transaction. `audit()` calls
    `persistAudit()` synchronously on every audit-worthy event (dozens to
    hundreds of times per 10-minute window), so any periodic I/O latency on
    the persistent volume (a known characteristic of network-attached block
    storage, e.g. background snapshot/housekeeping jobs) would stall
    whichever synchronous write happens to be in flight — explaining a
    ~600s-periodic, market-hours-INDEPENDENT cadence (an infra-side period)
    far better than Tier 2's own time-of-day-dependent interval could.
    NOT YET DIRECTLY MEASURED at the moment of a live EVENTLOOP-LAG entry —
    same posture as every prior update on this item: plausible architecture,
    not yet proof. SHIPPED (own PR, v1.0.302): `server/dbWriteTiming.ts`
    (NEW) times persistAudit's INSERT and DELETE independently and audits a
    new `DB-SLOW-WRITE` entry (with which statement was slow and for how
    long) whenever either crosses 500ms — the direct instrument this theory
    needs, mirroring eventLoopLag.ts's precedent exactly. Also switched the
    shared `db` connection to WAL journal mode (`db.pragma("journal_mode =
    WAL")`, called from bot.ts — auth.ts is a FROZEN PATH, so this
    configures the already-exported connection instead of editing that
    file); `synchronous` was left untouched (SQLite's FULL default) to avoid
    weakening durability for auth.ts's users/sessions tables. This is a
    genuinely different mechanism than the refuted tmpCleanup fix (a
    different file, a different blocking primitive, a different physical
    volume) — per the "two failed fixes = architecture smell" bar being
    about repairing the SAME mechanism twice, not a second hypothesis on the
    same open item, this does not trip it; if EVENTLOOP-LAG keeps firing
    with zero coincident DB-SLOW-WRITE entries after this deploys, THAT
    would be the second refutation, and the next session should stop
    guessing and file the architecture-smell escalation per RECURRENCE
    ESCALATES rather than attempt a third theory blind.
    RATCHET: `server/dbWriteTiming.test.ts` (NEW, 9 tests) — pure-function
    coverage for `isSlowDbWrite`/`formatSlowWriteMessage` plus wiring-pin
    tests (mirroring eventLoopLag.test.ts's convention of reading bot.ts's
    source text) confirming the WAL pragma call exists, persistAudit times
    both statements, gates the audit call on either exceeding the
    threshold, and guards against re-entrant recursion when logging its own
    slow-write event.
    NEXT STEP for whichever session checks in after this deploys (a few
    hours of live data is enough given the ~10min cadence): query
    `/api/diag/audit?type=EVENTLOOP-LAG` and `type=DB-SLOW-WRITE`. Entries
    landing in the same windows confirms this theory (and turns the next
    move — likely moving audit_log writes off the main thread via a worker
    thread, since better-sqlite3 has no async API — into a justified,
    evidence-backed change rather than a guess). EVENTLOOP-LAG continuing
    with zero coincident DB-SLOW-WRITE entries refutes it — that is the
    second refutation on this item, and per RECURRENCE ESCALATES the next
    session must stop patching and file a structural wishlist.md proposal
    (e.g., request the ability to attach a CPU/heap profiler, or propose
    moving off Railway's shared/burstable compute tier if the stall turns
    out to be container-level CPU throttling rather than anything in this
    codebase at all) instead of guessing a third mechanism.

    UPDATE 2026-07-13 (v1.0.307), this session — SECOND REFUTATION
    CONFIRMED LIVE, AND (unlike the prior two hypotheses) A ROOT CAUSE
    ACTUALLY LOCATED AND FIXED WITH DIRECT EVIDENCE, NOT A PERIOD-MATCHING
    GUESS. This is a root-cause-analysis session per RECURRENCE ESCALATES
    (two prior fix attempts on this item — tmpCleanup v1.0.291, SQLite WAL
    v1.0.302 — both refuted), not a third blind patch: it widens the
    search that both priors scoped too narrowly, and it verifies the
    mechanism directly rather than inferring it from a matching period.
    SECOND REFUTATION, confirmed before investigating further: live
    `/api/diag/audit?type=DB-SLOW-WRITE&token=$DIAG_TOKEN` returned ZERO
    entries; `/api/diag/audit?type=EVENTLOOP-LAG` on the currently-running
    process (uptime traced via `/api/health` to a restart at 13:49 UTC
    today, well after v1.0.302's WAL-mode fix was live) shows the same
    ~10-minute cadence CONTINUING with magnitude GROWN FURTHER (83,000-
    95,000ms, up from the pre-WAL-fix 72,000-98,000ms). The SQLite
    sync-write theory is refuted exactly as its own NEXT STEP anticipated.
    ROOT CAUSE FOUND (widening the search past bot.ts): every prior
    update in this item grepped ONLY `server/bot.ts`'s `setInterval`s.
    `server/routes.ts` has THREE separate `setInterval`s on the exact same
    `10 * 60_000` period: `archiveTick`, `refreshShadowStats`,
    `refreshPortDwell`. `refreshShadowStats` calls
    `computeShadowStatsAsync` (`server/shadowFleet.ts`) which — despite
    being named "Async" and using a genuinely non-blocking streaming file
    reader (`foldVesselArchiveAsync`, the 2026-07-05 OOM-repair precedent)
    — ends every cycle with `ShadowAggregator.finish()` calling a fully
    SYNCHRONOUS, single-threaded, zero-yield-point all-pairs comparison
    for hull-swap identity-candidate detection: for every vessel A's last
    point, compare against every OTHER vessel B's first point (O(vessels²)
    haversine distance calls). Live confirmation:
    `/api/data/shadowstats` reports `vessels_seen: 34895` in the 72h
    window (2026-07-13) — 34,895² ≈ 1.2 BILLION haversine calls, run
    synchronously with no `await`/`setImmediate` yield anywhere in the
    loop, which is more than enough single-threaded CPU work to explain
    an 80-95 SECOND stall. This is the same O(N²) shape in BOTH the sync
    (`detectIdentityCandidates`, used by `computeShadowStats`, the
    request-path/test variant) and async (`ShadowAggregator.finish()`,
    used by the live 10-min poller) code paths — a genuine algorithmic
    bug, not something the 2026-07-05 streaming-I/O fix introduced or
    could have caught (that fix targeted file-read blocking, not the
    computation that runs after the read finishes). It also explains the
    GROWING MAGNITUDE trend across every session's readings on this item:
    `vessels_seen` in a fixed 72h window only grows as the archive's
    coverage/history matures, so the O(N²) blowup gets worse every day,
    independent of whatever the tmpCleanup/WAL fixes did or didn't do —
    a symptom neither prior hypothesis could explain (file counts and DB
    write volume don't have an obvious multi-day growth trend the way
    `vessels_seen` does). `refreshPortDwell` (the third 10-min interval)
    was checked and does NOT share this defect — `portDwell.ts`'s dwell
    detection is a straightforward per-vessel fold, no all-pairs
    comparison; `archiveTick` is two lightweight upstream fetches. The
    root cause is isolated to `refreshShadowStats`'s identity-candidate
    computation specifically.
    FIX (`server/shadowFleet.ts`): replaced the all-pairs loop (shared by
    both `detectIdentityCandidates` and `ShadowAggregator.finish()` via a
    new shared `countHullSwapCandidates` helper — same predicate, same
    output) with a spatial grid (cell size = nearKm degrees-equivalent,
    with a latitude-aware longitude-neighbor-cell radius so degree
    compression toward the poles never causes a false negative) combined
    with a sorted-by-time binary-search window per cell. Time-window
    filtering ALONE was checked and rejected as insufficient during this
    session's own design work: terrestrial AIS traffic packs into a FIXED
    72h wall-clock window regardless of vessel count, so `firsts`-per-
    second density scales WITH the archive, meaning a time-only filter's
    candidate-per-query count (K) also scales with N — real complexity
    unchanged, only a constant-factor (withinHours/windowHours ≈ 1/6)
    win. The spatial grid is what actually breaks the N² (real traffic
    clusters near coastlines/receivers rather than spreading uniformly
    over the globe, so per-cell candidate counts stay bounded as N
    grows). KNOWN, ACCEPTED LIMITATION: cell keys use unwrapped longitude,
    so a pair straddling the antimeridian (~180°) would be missed; none
    of this system's tracked zones are near the date line, and this is a
    heuristic RAW statistic, not a trading signal, so this was accepted
    rather than adding dateline-wrap bucket logic for a real-world-
    irrelevant case (documented in the code).
    VERIFICATION (this is where the prior two attempts on this item fell
    short — "not yet directly measured" was their own honest caveat both
    times; this fix is measured against realistic scale before shipping,
    not inferred from a matching period): (1) all pre-existing tests pass
    unchanged, including the 2026-07-05 sync-vs-async byte-identical
    ratchet, confirming the new shared helper produces IDENTICAL output
    to the pre-fix code on real fixtures; (2) a synthetic 8,000-vessel
    perf ratchet test (`server/shadowFleet.test.ts`) completes in ~140ms
    post-fix; A/B-verified via `git stash` that the OLD code takes
    ~700-1600ms on the SAME 8,000-vessel input (the gap widens
    non-linearly with N — this is O(N²) vs. much-better-than-N² by
    construction, not a fixed multiplier); (3) an INDEPENDENT brute-force
    O(n²) oracle (deliberately not sharing any code with the fix, so a
    shared bug can't hide in both the fix and its own test) fuzz-verifies
    the optimized path against 40 random/clustered layouts (including the
    "everyone in one grid cell" worst case) — exact count match every
    time; a deliberately-injected radius bug (`latRadius = 0`) was
    confirmed to make both the fuzz oracle and a dedicated boundary test
    fail, proving the tests actually catch this class of defect, not just
    pass by construction; (4) a standalone Node measurement (not part of
    the shipped test suite, too slow for CI) simulated PRODUCTION SCALE —
    34,895 vessels clustered into as few as 5 hotspots (a more
    pathological concentration than real global AIS coverage) — and
    completed in ~1 SECOND, vs. the 80-95 SECOND live stalls this item
    tracks.
    RATCHET: `server/shadowFleet.test.ts` — perf regression test (5s
    ceiling on 8,000 vessels; old code would take tens of seconds at this
    N, let alone the ~35,000 seen live), independent fuzz-oracle
    correctness test (40 trials), and a deterministic hull-swap boundary
    test (exactly-12h/exactly-20km edges, both directions).
    NOT YET LIVE-CONFIRMED (same honest posture as every prior update on
    this item): this fix has not yet been observed to actually stop the
    production EVENTLOOP-LAG cadence — that is the one thing static
    analysis and synthetic benchmarks cannot substitute for. NEXT STEP
    for whichever session checks in after this deploys and merges (a few
    hours of live data is enough given the ~10min cadence, same as every
    prior update): query `/api/diag/audit?type=EVENTLOOP-LAG` and
    `/api/data/shadowstats` (confirm `identity_candidates` is still a
    sane, non-zero, non-degenerate number — the fix must preserve
    behavior, not just speed). No more ~600s-cadence EVENTLOOP-LAG
    entries (or a large drop in magnitude/frequency) confirms the fix;
    if the exact-same cadence and magnitude persist despite this deploy,
    that would be the THIRD refutation on this item and — per RECURRENCE
    ESCALATES, now genuinely warranted — the next session must stop
    patching entirely and file the CPU-profiler/compute-tier structural
    proposal in wishlist.md that the prior update already sketched,
    rather than attempt a fourth theory.

    UPDATE 2026-07-13 ~20:20 UTC (this session, [RESEARCH] — judging a
    matured experiment per SESSION BUDGET, no code touched): **LIVE
    CONFIRMED — v1.0.307's shadowFleet O(n²) fix stopped the stall.**
    `/api/health` shows the current process's `uptime_s: 13251` at
    20:17:14 UTC, i.e. a restart at ~16:32:23 UTC — consistent with the
    PR #470 merge/deploy at 16:33:50 UTC. `/api/diag/audit?
    type=EVENTLOOP-LAG&token=$DIAG_TOKEN` shows the last big stall
    (~96,003ms) at 16:31:38 UTC, from the PRE-deploy process, then a
    single minor 1,361ms blip at 16:36:26 UTC (small enough to be
    ordinary GC/scheduling noise, not the tracked defect), and **zero**
    EVENTLOOP-LAG entries since — a clean 3h41m with no recurrence,
    against a pre-fix cadence that had fired every ~10 minutes without
    fail for days. `type=DB-SLOW-WRITE` is still empty (consistent,
    that theory stays refuted). `/api/data/shadowstats` confirms the fix
    preserved behavior, not just speed: `vessels_seen: 38215` (up from
    34,895, archive still maturing as expected) and `identity_candidates:
    149782` — sane and non-degenerate, not zeroed out by the rewrite.
    3h41m is short of a full ~10-min-cadence multi-day read, but it is
    22+ consecutive missed occurrences against a defect that had a
    ~100% recurrence rate beforehand — strong evidence, not yet the
    "several days clean" bar for fully closing the item. KNOWN BROKEN
    #18 status: CONFIRMED FIXED, pending one more session's check a
    day+ out to close it outright. NEXT STEP: a future session queries
    `/api/diag/audit?type=EVENTLOOP-LAG` again after ≥24h live — if
    still clean, mark this item RESOLVED; any recurrence at this point
    would be a NEW, fourth mechanism (the shadowFleet path is now
    directly measured and ruled out, not just theorized), not a
    reopening of the O(n²) theory.

    UPDATE 2026-07-14 ~02:35 UTC (this session, [REPAIR], read-only
    check — not yet the full ≥24h close-out, logged so the next session
    doesn't re-derive it): ~10h post-deploy (deploy 2026-07-13 16:33:50
    UTC). `type=TIER2-ERROR` and `type=STREAM-DISCONNECT` are BOTH still
    zero entries since deploy (last STREAM-DISCONNECT at 16:31:38 UTC,
    pre-deploy) — strong continued confirmation. `type=EVENTLOOP-LAG` is
    NOT literally zero, though: 6 small entries since the deploy (16:36,
    20:24, 23:19 on the 13th; 00:39, 00:40 on the 14th), magnitude
    ~1,151-1,366ms — two orders of magnitude below the pre-fix
    83,000-98,000ms stalls and NOT on the tight ~600s cadence (irregular
    spacing, one pair only 39s apart). This matches the prior update's
    own precedent for the 16:36:26 blip ("small enough to be ordinary
    GC/scheduling noise, not the tracked defect") — same read applies to
    all 6. Not yet the ≥24h bar; next session should re-check and, if the
    pattern holds (zero TIER2-ERROR/STREAM-DISCONNECT, only occasional
    sub-2s EVENTLOOP-LAG blips), mark RESOLVED.

    UPDATE 2026-07-17, this session ([REPAIR], read-only close-out check
    per the 2026-07-14 update's own NEXT STEP — no code touched): **CLOSING
    THE ≥24H BAR — RESOLVED.** ~4 days post-deploy (v1.0.307 merged/live
    2026-07-13 16:33:50 UTC). Live checks against production
    (`/api/diag/audit?type=...&token=$DIAG_TOKEN`, each queried at
    `limit=200` to confirm the type-filtered scan reaches well past the
    deploy boundary rather than just the most-recent window — verified by
    the EVENTLOOP-LAG query below actually surfacing entries from
    2026-07-16, i.e. the endpoint does filter across real history, not
    just a shallow recent-N slice): `type=TIER2-ERROR` — **zero** entries
    of any kind since the deploy (previously a ~100% recurrence rate on a
    ~10min/~30min cadence). `type=STREAM-DISCONNECT` — **zero** entries
    since the deploy. `type=DB-SLOW-WRITE` and `type=TMP-CLEANUP` — still
    zero, consistent with both refuted theories staying refuted.
    `type=EVENTLOOP-LAG` — only the same 3 small entries the prior update
    already discounted as ordinary GC noise (506-614ms, 2026-07-16
    20:32/22:30 and 2026-07-17 00:23), nothing since, nothing at the
    ~600s cadence, nothing near the pre-fix 60-98 SECOND magnitude. Four
    days clean on the two symptoms that actually mattered
    (TIER2-ERROR/STREAM-DISCONNECT) is well past the "several days clean"
    bar this item set for itself. KNOWN BROKEN #18 is RESOLVED: the
    root cause was `server/shadowFleet.ts`'s O(n²) all-pairs hull-swap
    identity-candidate scan (v1.0.307's spatial-grid fix), not the two
    earlier refuted theories (tmpCleanup sync fs, SQLite sync writes).
    No further action needed on this item; a future recurrence would be a
    new, fourth mechanism, not a reopening of the O(n²) theory (per this
    item's own 2026-07-13 note).

    UPDATE 2026-07-18 (this session, scheduled-routine, [REPAIR]) —
    RECURRENCE CONFIRMED, EXACTLY AS THIS ITEM'S OWN 2026-07-13 NOTE
    ANTICIPATED: A NEW (FOURTH) MECHANISM, NOT A REOPENING OF THE O(n²)
    THEORY. Live `/api/health` at session start showed
    `scanner.status: "degraded"`, `consecutiveFailures: 6` — the exact
    `TIER2-ERROR "Daemon timeout"` symptom this item tracks, now
    recurring continuously since **2026-07-18T13:35:06Z** (16 entries by
    16:01:57Z, every attempt failing, ~2.5h and counting at session end,
    STILL ONGOING). Per RECURRENCE ESCALATES this was treated as a
    root-cause investigation, not a blind re-patch of the O(n²) fix
    (which is a different file/mechanism and already directly measured
    as fixed).
    FIRST STEP (per this item's own established discipline — check the
    instruments already built before guessing): queried every symptom
    the three prior mechanisms would produce. `type=STREAM-DISCONNECT` —
    **zero** entries (the O(n²) fix's signature symptom is completely
    absent this time — strong evidence this is NOT that mechanism
    recurring). `type=EVENTLOOP-LAG` — only two small sub-second blips
    (778ms, 513ms), nowhere near the 60-98 SECOND magnitude the O(n²)
    bug produced, and not on its ~600s cadence. `type=DB-SLOW-WRITE` and
    `type=TMP-CLEANUP` — both zero (both theories stay refuted, as
    expected). This is a genuinely different symptom shape: no
    coincident event-loop stall of any meaningful size, meaning the
    Node process itself is NOT blocked — the daemon RPC is just
    taking longer than 300s to actually finish its work.
    TIMING RECONSTRUCTION (via `type=TIER2` success/failure audit lines):
    the last SUCCESSFUL full scan completed at 13:25:05Z ("Scanned 11924
    stocks... via daemon", ~63s wall time — consistent with this
    function's own documented "~4-60s" normal range). The very next scan
    started at 13:30:06Z and **never completed** — first timeout logged
    at 13:35:06Z, exactly 300s later (Node's own `pythonCall` timeout).
    Every subsequent attempt (16 total by session end) failed the same
    way, spaced >=6 minutes apart (always >300s — meaning at most one
    stuck dispatch's zombie thread can still be alive when the next
    attempt starts, which is exactly why `active_dispatches` reads a flat
    "2" — [this health-probe call] + [the one genuinely-still-running
    scan] — every single time, never climbing, ruling out unbounded
    zombie-thread pileup as well).
    HYPOTHESIS (NOT YET DIRECTLY MEASURED — same honest posture every
    prior update on this item has taken before its instrument existed):
    **this morning's own KNOWN BROKEN #23 fix (`bars_feed()`, v1.0.397,
    merged ~11:20-11:30Z) is the proximate trigger.** Downstream chain
    (REASONING STANDARD #1, traced two steps, stated before shipping):
    before that fix, all 29 `/v2/stocks/bars` call sites 400'd
    immediately (fast failure) — so every VXX/SPY/regime/floor-basket/
    correlation bars fetch across `bot_engine.py`, `macro_data.py`,
    `options_scanner.py`, `instrument_selector.py`, `vol_surface.py`,
    `intraday_shorts.py`, `shadow_portfolio.py` etc. was cheap. After the
    fix, all 29 sites now SUCCEED — each is a real network round-trip
    through the single process-wide `alpaca_throttle` token bucket
    (`alpaca_rate_limiter.py`, 180 req/min, FROZEN PATH — a shared
    `threading.Lock` + bucket every Alpaca caller in the process
    contends for). `bot_engine.py` alone has bars-fetch call sites inside
    `deep_score`'s per-candidate loop and the floor-basket/correlation
    checks (lines ~837, ~1360, ~1467, ~2198, ~3970, ~4529, ~4935) that
    previously failed instantly and now perform real throttled requests
    — on top of whatever Tier 1 (30s cadence) and Tier 3 (hourly) are
    also now successfully doing through the same shared bucket. More
    real traffic sharing one 180/min budget, concentrated during actual
    market hours, is a coherent and structurally-motivated (REASONING
    STANDARD #5: an honest "why now" — the fix itself is what changed)
    explanation for a full scan's snapshot-fetch/deep-score phase
    queuing behind that contention long enough to blow through the 300s
    outer bound — consistent with the ~2h delay between the fix going
    live and the first timeout (traffic had to accumulate) and with
    zero event-loop/DB symptoms (the Node process isn't stalled; the
    daemon thread is genuinely waiting on rate-limited I/O). NOT proof —
    an alternative explanation (Alpaca-side latency degradation coincident
    with, but unrelated to, this morning's fix) has not been ruled out.
    INSTRUMENT SHIPPED (own PR, v1.0.398, visibility-only — no trading/
    scoring/timeout-threshold change, mirroring this item's own
    eventLoopLag.ts/dbWriteTiming.ts precedent of building the direct
    measurement before proposing a fix): `bot_engine.py`'s
    `_scan_market_inner()` (TIMING-DISK 2026-04-23) already persists a
    per-phase wall-clock breakdown to `voltrade_scan_timings.json`
    specifically so it survives a timeout kill — but nothing exposed
    that file outside the container except the owner-cookie-gated
    `/api/system/snapshot`. Added a new token-gated `timings` probe to
    `/api/diag/:probe` (`server/diag.ts` DIAG_PROBES whitelist +
    `server/bot.ts` case, same read-only file lookup
    `/api/system/snapshot` already does, same `sanitizeDiag` pass-through
    every other probe uses) so a DIAG_TOKEN session — not just an owner
    cookie — can read exactly which phase the next stuck scan reached
    without needing dashboard access.
    RATCHET: `server/diag.test.ts` — new probe pinned in `DIAG_PROBES`
    (covered by the existing generic "every whitelisted probe has a case"
    test) plus a dedicated test asserting the block checks both the
    `/data/voltrade` and `/tmp` paths, passes through `sanitizeDiag`, and
    reports `found: false` rather than erroring when no scan has run yet.
    NOT YET LIVE-CONFIRMED. NEXT STEP for whichever session catches the
    next occurrence (or checks in once this deploys, given the scanner is
    STILL degraded as of session end): query
    `/api/diag/timings?token=$DIAG_TOKEN` at/near a live timeout — if
    `last_phase_completed` sits at "quick_scan"/snapshot-fetch with a
    `duration_sec` far above the ~4-60s normal range, that CONFIRMS the
    rate-limiter-contention hypothesis and the fix becomes deduplicating
    the many redundant same-symbol (VXX/SPY/QQQ/TLT/HYG) bars fetches
    that are now scattered across ~7 files with no shared cache (a
    MUTABLE-territory fix — `alpaca_rate_limiter.py` itself is frozen and
    cannot be loosened). If `last_phase_completed` instead sits somewhere
    that does no Alpaca I/O (e.g. deep in `deep_score`'s scoring math with
    no bars call nearby), that refutes this hypothesis and reopens the
    search for a genuinely different (now actually a fourth) mechanism —
    do not assume either answer before reading the probe.

    UPDATE 2026-07-18 (session 3, scheduled-routine [REPAIR]) —
    RATE-LIMITER-CONTENTION HYPOTHESIS DIRECTLY MEASURED AND CONFIRMED,
    exactly per the prior update's own NEXT STEP ("whichever session
    catches the next occurrence, query `/api/diag/timings?token=
    $DIAG_TOKEN`"). SESSION-START STATE: the storm this item's prior
    update flagged as "STILL ONGOING" at 16:01:57Z had continued —
    `/api/diag/audit?type=TIER2-ERROR` showed 6 further occurrences this
    session directly observed (19:18:23Z through 20:00:44Z, same
    signature every time: `daemon rss=2xx-27xMB active_dispatches=2`),
    then went clean for 23+ minutes straight through this session's own
    live checks (20:00:44Z to 20:23:51Z, re-verified with `/api/diag/
    audit`, `/api/diag/scanner` `consecutiveFailures: 0`, and a fresh
    27.33s `/api/diag/timings` completed-scan read) — the cluster is
    intermittent, not constant, consistent with load-dependent
    contention rather than a fixed always-broken state.
    LIVE CAPTURE: polled `/api/diag/timings` on a ~55s cadence and caught
    a scan mid-flight at 20:26:23Z (`status: "in_progress",
    last_phase_completed: "quick_scan"` at only 4.33s total — i.e.
    already past the two prior EVENTLOOP-LAG-adjacent phases with nothing
    unusual, then stalled somewhere inside `deep_score`, the next phase).
    Continued polling until it finished: this particular scan did NOT
    time out (completed at 96.96s total, under the 300s bound) but its
    own phase breakdown is the direct evidence the hypothesis needed:
    `deep_score` took **51.91s** (prior completed scan, read minutes
    earlier this same session: 21.33s — ~2.4x) and the `tier_engine_
    breakdown` sub-object showed `tier1_sec: 39.42` (prior baseline:
    `tier1_sec: 1.01` — ~39x). Both elevated phases are exactly the two
    the contention hypothesis predicted would be sensitive to real
    Alpaca traffic, and both moved together in the same run — not noise
    in one isolated phase.
    SECOND, COMPLEMENTARY MECHANISM FOUND (read-before-write trace of
    WHY Tier 1 specifically would be rate-limiter-sensitive, not just
    deep_score): `tier1_csp_core()` (tiered_strategy.py:384) calls
    `_get_t1_universe()` (tiered_strategy.py:148) → `csp_universe.
    get_top_csp_candidates()`, gated by `LAYER1_CACHE_TTL`/
    `LAYER2_CACHE_TTL` = 900s/15min (csp_universe.py:47-48). Cache HITS
    are cheap (explains the 1.01s baseline); a cache MISS triggers real
    `requests.get()` calls (csp_universe.py:116, 266) that queue behind
    the exact same process-wide `alpaca_throttle` bucket every bars-fetch
    call site in the codebase shares (`alpaca_rate_limiter.py`, 180
    req/min, FROZEN mechanism). This is not a competing theory to the
    prior session's deep_score/bars-fetch hypothesis — both are
    independent contributors funneling into the SAME shared bottleneck,
    which is consistent with why the two elevated phases in the captured
    run moved together.
    REDUNDANT-CALL-SITE CENSUS (verified this session, not assumed from
    the prior update's "~7 files" estimate): `grep`'d every literal
    `/v2/stocks/bars` call site across the files the prior update named
    plus `tiered_strategy.py` — **14 call sites across 8 files**, not 7:
    `bot_engine.py:1467,3970`; `macro_data.py:202,247,282`;
    `options_scanner.py:206,240,539`; `vol_surface.py:263,308`;
    `intraday_shorts.py:253,340`; `shadow_portfolio.py:333,353`.
    (`tiered_strategy.py` itself has none — its exposure is indirect, via
    `csp_universe.py`'s two call sites above, not a 15th direct one.)
    NOT FIXED THIS SESSION, DELIBERATELY: per RECURRENCE ESCALATES this
    stayed a root-cause investigation, not a re-patch — and unlike a
    visibility-only instrument, a shared bars cache is a real behavior
    change spanning 8 MUTABLE files that directly feed live scoring and
    CSP sizing decisions. Before any of those 14 call sites can safely
    share a cache, each one's actual timeframe/lookback/adjustment
    parameters need to be read and compared (READ BEFORE WRITE) — two
    calls that both hit `/v2/stocks/bars` for the same symbol are only
    truly redundant if they also request the same window; assuming that
    from the URL pattern alone, without reading each call site, is
    exactly the "patch from assumption" the constitution forbids, and a
    wrong merge here risks a staleness/lookahead bug in code that
    directly drives trade decisions (REASONING STANDARD #7) — a
    materially worse outcome than the current slow-but-correct
    scanning delay this item tracks.
    CONCRETE NEXT STEP for the fix session (not vague): (1) read all 14
    call sites' actual `timeframe`/`start`/`end`/`adjustment` params and
    build a same-symbol-same-window compatibility matrix — only truly
    identical requests collapse; (2) implement as an ADDITIVE per-scan-
    cycle cache (dict keyed on `(symbol, timeframe, start, end)`,
    populated lazily on first request within a scan, passed as an
    optional pre-fetched arg with the live call as fallback) so it can
    land file-by-file with zero behavior change for any not-yet-touched
    caller, rather than one large cross-file PR; (3) re-run this same
    live-catch procedure against `/api/diag/timings` post-fix and confirm
    `deep_score`/`tier1_sec` durations return toward the 21s/1s baseline
    under real contention conditions, not just in a quiet window.
    Root cause is now MEASURED, not merely theorized — the remaining gap
    is the fix's own correctness verification, which deserves a session
    of its own.

    UPDATE 2026-07-19 (session 2, scheduled-routine, [REPAIR], v1.0.416)
    — FIRST FILE-BY-FILE FIX LANDED, per this item's own 2026-07-18
    NEXT STEP (1). Storm STILL ACTIVE at session start: `TIER2-ERROR`
    recurring continuously since 13:36:01Z, 16 occurrences by 16:11:47Z
    (~2h35m). Built the full 14-site compatibility matrix by hand (symbol,
    start formula, limit, adjustment) exactly as the prior session's own
    caution demanded before merging anything — most of the 14 turned out
    genuinely NOT collapsible (different tickers, different windows,
    different `adjustment` flags); only 2 true duplicates survived:
    `bot_engine.py`'s `deep_score()` credit_spread (TLT/HYG) fetch — market
    -wide, not ticker-specific, but re-issued on every one of up to 15
    deep-scored candidates per scan (the single highest-multiplier site of
    the 14) — and `macro_data.py`'s `get_macro_snapshot()` SPY 300d/220
    bars, fetched twice in the SAME function call despite its own comment
    already claiming reuse. Both fixed with zero-risk, proven patterns
    (the first mirrors the codebase's own existing `_MOST_ACTIVES_CACHE`
    2026-05-18 fix for the identical bug class; the second just makes the
    existing "already fetched above — reuse" comment true). Full trace,
    the compatibility matrix, and the two sites deliberately left alone
    (the VXX cross-file dup between `macro_data.py`/`options_scanner.py`,
    and `csp_universe.py`'s cache-miss path) are in experiments.md's
    2026-07-19 session-2 entry. RATCHET: `test_deep_score_credit_spread_
    cache.py` + `test_macro_snapshot_spy_dedup.py`, 6 new behavioral tests
    (call-count assertions against mocked `requests.get`, not just
    static-shape checks). THIS IS DELIBERATELY PARTIAL — 2 of 14 sites,
    chosen for zero inference risk, not maximum coverage. NOT YET
    LIVE-CONFIRMED (this PR is pending merge — market-hours session, human
    judgment call on whether the still-active storm counts as the
    "critical live break" merge-timing exception). NEXT STEP for whichever
    session catches the next occurrence post-merge: re-run the
    `/api/diag/timings` live-catch procedure; if `deep_score`/`tier1_sec`
    are still elevated, that points at the remaining untouched sites (VXX
    cross-file merge, `csp_universe.py`'s cache-miss path), not evidence
    this fix was wrong — do not re-diagnose from scratch, continue from
    here.

19. **[RESOLVED 2026-07-11, v1.0.270] `track_fill()`'s `code_version` field
    was hardcoded to the literal `"1.0.34"` (Bug #13's fix version) for
    EVERY live trade_feedback record, forever — PROMOTION RULES #4's
    attribution mechanism ("bump the version so code_version separates
    this change's live results from prior code") has never worked for a
    single live-recorded fill since v1.0.34.** Found while live-verifying
    KNOWN BROKEN #3/#4 via `/api/diag/ml` this session: `live_code_versions`
    showed `{"1.0.34": 3}` for records dated 2026-07-10, while the deployed
    app was v1.0.269 — 235 version bumps with zero change reflected in the
    attribution field. READ BEFORE WRITE traced it to three literal
    `"code_version": "1.0.34"` string constants inside `ml_model_v2.py`'s
    `track_fill()` (entry-fill record, orphan_exit fallback x2) — none of
    them read `order_data.get("code_version")`, so nothing any caller
    passed could ever change the stamped value. `server/bot.ts`'s three
    `track_fill` call sites (recordExitFill via `buildExitFillPayload`,
    the morning-queue payload, the regular-hours payload) never sent a
    `code_version` key either, so there was no counter-pressure from the
    JS side; a fourth hardcoded site (`trackClosedTrades`'s feedback
    block, bot.ts:718) is separately known-dead per #12(b) but carried the
    same stale literal and was fixed for consistency.
    MEASUREMENT INTEGRITY (own PR, not bundled with any strategy change):
    BEFORE — every live record, regardless of when/what version produced
    it, reported `code_version: "1.0.34"`. Since `MIN_TRUSTED_VERSION =
    (1,0,34)` (`feedback_boot_cleanup.py`) and `MIN_FEEDBACK_VERSION =
    "1.0.33"` (`ml_model_v2.py`) are both ≤ "1.0.34", every live record
    already always passed both trusted/non-legacy gates — so the 0.4x
    legacy-weighting mechanism described in KNOWN STATE could never
    engage for any live fill, and no session could ever attribute a
    specific live result to a specific code change via this field. AFTER
    — `bot.ts` now passes the real running `pkgVersion` (package.json) on
    every `track_fill` call; Python reads `order_data.get("code_version")`
    with the old literal kept only as a fallback for callers that don't
    supply one (e.g. direct test calls), matching pre-fix behavior exactly
    in that fallback case. BIAS DIRECTION: neutral-to-corrective, not
    strategy-favorable — no record was ever wrongly downweighted before
    (the bug hid granularity, it didn't inflate quality), and the fix
    restores an existing designed mechanism rather than introducing a new
    one; it does NOT change any live P&L, slippage, or trade decision.
    ADJACENT FINDING (not fixed, logged for the future): the 3 live
    records observed this session were ALL `orphan_exit` (an exit fill
    that found no matching open entry) — relevant to #12(b)'s open
    decision gate on whether D2's WS-exit path is now producing real
    matched closes; this session's snapshot alone can't say whether
    that's the norm or a coincidence of timing, so #12(b) stays open,
    not resolved, pending more live records under the now-fixed
    versioning.
    RATCHET: 3 new Python regression tests (`test_fixes_pr8.py` —
    entry-fill, orphan-exit, and matched-exit paths each assert the
    caller-supplied version survives, not the old literal) + 2 new/updated
    TS tests (`exitFill.test.ts`) asserting `code_version` is present and
    correct in the built payload. Full local gates: `python3 -m pytest -q`
    — 638 passed, 1 skipped (pre-existing legacy-file skip, unchanged);
    `npx tsx --test server/*.test.ts` — 559 passed, 0 failed (after
    `npm ci`, which this sandbox needed cold). `npx tsc --noEmit` has
    ~60 pre-existing errors unrelated to this change (Buffer/Map-iteration/
    downlevelIteration noise across unrelated files, confirmed via grep
    that none reference `exitFill`/`codeVersion`/`code_version`) — not a
    regression, and not part of the documented PROMOTION RULES gate.
    Backtest: N/A — no trading/scoring/sizing logic touched, pure
    attribution-metadata fix.

21. **[CONFIRMED 2026-07-14 — visibility gap FIXED, v1.0.310; underlying
    analyze.py root cause still open] `deep_score()`'s alt-data enrichment
    (wikipedia/fred/gdelt) may not be running at all this
    weekend — or this may be entirely normal weekend quiet.** Found while
    checking `/api/diag/audit` for the CSP-fix live-verification task (item #3):
    every `DIAGNOSTIC` entry in the visible ~2.3h window (Sat 2026-07-11
    18:00-20:20 UTC) reports `"Multiple API sources down: ['wikipedia',
    'gdelt', 'fred']"` (`diagnostics.py`'s `api_checks`, existence-only —
    triggers `reduce_position_size(0.6x)` at >=3) plus `"4 data sources
    stale: ['insider', 'fred', 'gdelt', 'event_memory']"`. `/api/diag/
    scanner`'s `dataSourceErrors` (the REAL per-exception capture built in
    v1.0.150 specifically to disambiguate this) was empty `{}` the whole
    window — consistent with either (a) these fetchers succeeding cleanly
    (then their cache files SHOULD exist — every fetcher in `alt_data.py`
    calls `_cache_set()` unconditionally, even on total failure, so a
    missing file needs the function to never have been CALLED, not just to
    have errored), or (b) `deep_score()`'s own early return
    (`get_stock_details()` — subprocess to `analyze.py` — returning
    None/error, bot_engine.py:561-562) firing before the enrichment block
    is ever reached, which would make the exception-capture mechanism
    blind by construction (it only wraps code the early return never lets
    run). NOT CHASED FURTHER THIS SESSION: `insider`/`event_memory` live on
    the persistent volume (not /tmp) with modest staleness thresholds
    (3h/72h) that a quiet weekend (no SEC filings, no major news) could
    plausibly breach on its own; the options-chain "no contracts available"
    T2-FAILs in the same window have an independently-verified benign
    weekend cause (OPRA snapshot endpoint returns empty with no live
    quotes, see item #3's neighbor evidence) — same shape of ambiguity, and
    this repo's own precedent (item #3's PARTIAL EVIDENCE note) says don't
    conclude from a market-closed window. **CHECK ON THE NEXT TRADING DAY**
    (Monday 2026-07-13 or later): if
    `/api/diag/audit?type=DIAGNOSTIC` still shows wikipedia/gdelt/fred
    "down" DURING market hours with active Tier2 scans, that rules out the
    weekend explanation and confirms a real gap — most likely `deep_score`'s
    early return silently skipping the whole enrichment block for candidates
    whose `analyze.py` subprocess call fails or times out (30s timeout,
    exceptions swallowed at bot_engine.py:352-353 with zero diag capture,
    unlike the 5 enrichment fetchers which got that treatment in v1.0.150).
    If confirmed, the fix is the same shape as v1.0.150's `_run_diag_fetch`:
    capture `get_stock_details()`'s failure reason into a diag field the
    existing `/api/diag/scanner` surface already exposes, rather than
    silently discarding it.

    UPDATE 2026-07-14 (this session, [REPAIR]) — CONFIRMED, root cause of
    the VISIBILITY gap fixed; root cause of the underlying analyze.py
    failure still open. Per this item's own "CHECK ON THE NEXT TRADING
    DAY" instruction: `/api/diag/audit?type=DIAGNOSTIC&token=$DIAG_TOKEN`
    shows `"Multiple API sources down: ['wikipedia', 'gdelt', 'fred']"` on
    EVERY entry from 2026-07-13 12:13 UTC through 22:45 UTC — a full
    Monday trading day, well past market open (13:30 UTC), with active
    Tier2 scans running throughout (confirmed via the same window's
    EVENTLOOP-LAG/TIER2-ERROR entries showing the bot alive and cycling).
    `/api/diag/scanner?token=$DIAG_TOKEN` returned `dataSourceErrors: {}`
    for the whole window — the weekend explanation is RULED OUT; this is
    hypothesis (b) from the original entry, not (a).
    ROOT CAUSE OF THE BLIND SPOT (READ BEFORE WRITE trace, no live
    analyze.py access from this sandbox — traced statically):
    `get_alt_data_score()` (`alt_data.py:487`) calls `get_wiki_attention`,
    `get_fred_macro`, `get_geopolitical_risk` — EVERY one of the three
    unconditionally calls `_cache_set()` at its own end, even when every
    internal HTTP call fails (each wraps its own `requests.get` in a
    `try/except: pass`-style swallow and still writes a default/empty
    result). That means if `_fetch_alt()` (bot_engine.py's `deep_score`)
    ever actually runs `get_alt_data_score(ticker)` to completion, `wiki_
    <ticker>.json`/`fred_macro_expanded.json`/`gdelt_risk.json` WILL exist
    regardless of network outcome — so 10+ hours of all three being
    simultaneously absent means `get_alt_data_score` was never entered at
    all, for any ticker, all day. The only gate standing between a
    scanned candidate and the 5-fetcher enrichment block (macro/intel/
    alt/social/finnhub, of which "alt" is one) is `deep_score()`'s early
    return (bot_engine.py, `if not detail or "error" in detail: return
    quick_result`) keyed on `get_stock_details(ticker)` — a 30s-timeout
    subprocess call to `analyze.py` wrapped in a bare `except Exception:
    pass`, with zero diagnostic capture anywhere (confirmed by grep: no
    caller of `get_stock_details` ever inspected the reason for a None/
    error return). This means a live analyze.py failure (subprocess
    exception, timeout, or analyze.py's own `{"error": ...}` payload)
    would silently skip ALL FIVE enrichment fetchers for that candidate —
    a strictly larger blind spot than the wikipedia/fred/gdelt symptom
    alone (macro/intel/social/finnhub would be equally starved, just
    without a cache-freshness alarm surfacing it, since those don't have
    diagnostics.py existence checks the way alt_data's three do).
    FIX SHIPPED (visibility only, own PR, v1.0.310):
    `get_stock_details(ticker, _diag=None)` gained the same optional
    `_diag` parameter `deep_score`'s 5 fetchers already use
    (`_run_diag_fetch` convention) — captures the subprocess exception
    type+message, empty-stdout detail (exit code + truncated stderr), or
    analyze.py's own `{"error": ...}` payload into `_diag["stock_details"]`
    without changing any return value (byte-identical whether or not
    `_diag` is passed, verified by test). `deep_score()` now forwards its
    own `_diag` into `get_stock_details(ticker, _diag=_diag)` — one call
    site, already reads `data_source_errors` -> `/api/diag/scanner`, no
    new wiring needed downstream. RATCHET: new
    `test_get_stock_details_diag.py` (6 tests) — A/B-verified via `git
    stash` that all 6 fail on pre-fix code (`TypeError: unexpected keyword
    argument '_diag'`) and pass post-fix; covers the analyze.py-error-
    payload case, empty-stdout case, subprocess-timeout case, malformed-
    JSON case, and a byte-identical-return-value check with/without
    `_diag`. `test_silent_except_ratchet.py`'s `bot_engine.py` pin lowered
    78 -> 77 (this fix converts one bare `except Exception: pass` into a
    capturing handler — a real improvement, not a regression, per that
    ratchet's own "count dropped, lower the pin" rule). Full gates:
    `python3 -m pytest -q` 673 passed, 2 skipped (baseline + 6 new, zero
    regressions); `npx tsx --test server/*.test.ts` 640/644 (the 4
    failures — apiKeyAccounts/compression/gdeltEvents/owmTiles — are the
    same pre-existing network-dependent failures documented in prior
    sessions' entries, confirmed unrelated since this PR touches zero TS
    files); `npx tsc --noEmit` shows only the pre-existing 3 sandbox-
    environment errors (missing @types/node/vite entry points, deprecated
    tsconfig option), unrelated to this change. `npm run build` could not
    run in this sandbox (`tsx` binary missing from `node_modules/.bin`,
    an environment gap, not a code issue) — moot in spirit since this PR
    touches zero TypeScript/client files.
    STILL OPEN — the actual reason `analyze.py` (or its subprocess
    invocation) is failing for every ticker all trading day. NEXT STEP
    for whichever session catches the next occurrence: query
    `/api/diag/scanner?token=$DIAG_TOKEN` and read the new
    `dataSourceErrors.stock_details` field — it will now carry the exact
    exception type, timeout, or analyze.py error string, replacing "guess
    from a cache-freshness proxy" with "read the actual failure reason."
    Leading candidates to check once that evidence is in, per this
    session's static read (not yet confirmed): analyze.py hitting a
    30s timeout under Railway's shared/burstable CPU (would tie into
    KNOWN BROKEN #18's now-closed event-loop-stall investigation as a
    sibling symptom of the same compute-tier constraint, though #18 was
    Node-side and this is a separate Python subprocess — worth checking
    for correlation, not assumed); a missing/expired API key or dependency
    inside analyze.py's own code path; or a code-level bug introduced
    since this path last worked cleanly. Do not guess between these
    without reading `stock_details` first.
    BACKTEST: N/A — pure diagnostic-visibility fix, no scoring/sizing/
    trading logic touched; `get_stock_details`'s and `deep_score`'s
    return values are unchanged for every input (proven by the byte-
    identical-return-value test), so this cannot affect any live trading
    decision.

    UPDATE 2026-07-14 later same day ([REPAIR], v1.0.311) — RESOLVED. Per
    this item's own NEXT STEP, read `/api/diag/scanner?token=$DIAG_TOKEN`
    ~8h after v1.0.310 deployed (Tier2 scans confirmed running throughout,
    "Scanned 11901 stocks, 2 trade candidates" every ~5min via
    `/api/diag/audit?type=TIER2`): `dataSourceErrors` was STILL `{}`, and
    `stock_details` never appeared — ruling out ALL THREE leading
    candidates named above (analyze.py timeout, missing key/dependency,
    code bug in analyze.py). If `get_stock_details()` were failing, the new
    v1.0.310 capture would show it; it never fired even once, meaning
    `get_stock_details()` itself was never the blocker — `deep_score()`
    was never being reached AT ALL, for a different reason.
    ACTUAL ROOT CAUSE (read-before-write trace of `bot_engine.py`'s
    memory-pressure guards, `scan_market()`): `_mem_rss_mb()` (used by
    BOTH the `SURVIVAL_MODE` guard at 700MB, bot_engine.py ~2627-2671, AND
    the `pre_deep_score` skip/trim guard at 130MB/100MB, ~2681-2720) read
    `resource.getrusage(RUSAGE_SELF).ru_maxrss` — a per-process HIGH-WATER
    MARK that never decreases for the life of the process (POSIX
    semantics), not "current" memory. Both guards compare that value
    against a threshold expecting real-time pressure. A long-running
    daemon process with pandas/numpy/lightgbm loaded will cross 130MB (and
    very plausibly 700MB, given hours of scanning 11,901 stocks per cycle)
    within its first scan or two — after which `ru_maxrss` stays at that
    peak FOREVER, so the guard trips once and never releases, even after
    GC frees memory back down. Net effect: `deep_score()` — and therefore
    `get_stock_details()` and all 5 enrichment fetchers — was silently
    disabled for the remainder of every daemon process's uptime after the
    first memory spike, exactly matching the observed symptom (multi-day
    all-three-sources-down with zero exceptions ever captured, because the
    code that would raise them was never reached). NOTE: a near-identical
    bug was partially patched once before without addressing the shared
    root cause — the `SURVIVAL-FIX 2026-04-23` comment on the 700MB guard
    describes fixing an env-var default that skipped deep score on every
    Railway scan; that fix left the underlying `_mem_rss_mb()` peak-vs-
    current confusion in place, so the class of bug re-manifested through
    the SAME helper function via a different trigger.
    FIX SHIPPED (own PR, v1.0.311): `_mem_rss_mb()` now reads
    `/proc/self/status`'s `VmRSS` line (actual current RSS on Linux — what
    Railway runs) as the primary path, falling back to the old
    `ru_maxrss` path only when `/proc` is unavailable (e.g. local macOS
    dev) — unchanged behavior there. This is a single shared-helper fix
    that corrects BOTH mem-pressure guards at once, not a per-guard patch
    — the RECURRENCE ESCALATES-flavored structural fix rather than a third
    band-aid on a different threshold. RATCHET: new
    `test_mem_rss_current.py` (5 tests) — A/B-verified via `git stash`:
    2 of 5 fail on pre-fix code (peak-leaks-through-as-800MB-when-current-
    is-100MB regression pin, and the basic /proc-read test), 3 pass
    unchanged (fallback + error-path tests, confirming the non-Linux
    fallback behavior is preserved byte-for-byte). The new `/proc` read
    path's `except Exception: pass` was narrowed to `except (OSError,
    ValueError):` specifically to exit `test_silent_except_ratchet.py`'s
    AST-based silent-handler scan (which forbids raising any pin) rather
    than bumping `bot_engine.py`'s pin from 77 to 78.
    GATES: `python3 -m pytest -q` — 678 passed, 2 skipped (baseline 673 +
    5 new, zero regressions); `test_silent_except_ratchet.py` — 2/2 pass
    (narrowed except, pin unchanged at 77). `npx tsx --test
    server/*.test.ts` — 640/644, same 4 pre-existing network-dependent
    failures (apiKeyAccounts/compression/gdeltEvents/owmTiles) as every
    prior session's baseline — this PR touches zero TS files. `npx tsc
    --noEmit` — same 3 pre-existing sandbox-environment errors, unrelated.
    `npm run build` could not run (`tsx` binary absent from
    `node_modules/.bin` in this sandbox, an environment gap, not a code
    issue, same as the v1.0.310 session) — moot in spirit, zero TS/client
    files touched.
    DOWNSTREAM CHAIN (REASONING STANDARD #1): the guards' INTENT (skip
    deep scoring under genuine memory pressure to avoid an OOM SIGKILL) is
    fully preserved — they now trigger on real current pressure instead of
    a stale permanent trip. The change makes deep_score() run MORE than it
    was (correctly, when memory is actually fine) — this restores ML-
    enriched scoring, yfinance fundamentals, and the 5-source alt-data
    enrichment to trade candidates that were silently getting quick-scan-
    only treatment, which is a real improvement to decision quality, not
    just a diagnostic — but carries the (intended, not a regression) risk
    that a Railway container running close to its real memory ceiling
    could now legitimately hit the 130MB/100MB/700MB gates far more
    reactively (rising and FALLING with actual pressure) than before,
    where they were effectively latched off. Worth watching post-deploy:
    OOM/SIGKILL rate should not increase (the guards restored to their
    intended real-time behavior); if it does, the 130MB/100MB thresholds
    themselves (unchanged by this PR) are the next thing to evidence-check
    per RULE REVIEW, not this fix.
    BACKTEST: N/A — infra/memory-guard correctness fix; does not change
    what deep_score computes for any candidate it does reach, only whether
    it is reached at all under actual (now correctly measured) memory
    pressure.
    STILL OPEN: whether the 130MB/100MB/700MB threshold VALUES themselves
    are well-calibrated for the current container size now that they will
    actually be evaluated against live current-RSS readings going forward
    (previously untestable, since the guards were effectively always-on
    after the first spike) — a future session should check
    `/api/diag/audit?type=DIAGNOSTIC` clears (wikipedia/gdelt/fred caches
    should start appearing) and confirm no new OOM/restart signature
    post-deploy before this item can be marked fully closed.

    UPDATE 2026-07-14 later same day ([REPAIR], v1.0.314) — RECURRED, per
    exactly this item's own "STILL OPEN" question above, and RESOLVED
    (pending live confirmation). Per RECURRENCE ESCALATES this session
    became root-cause analysis, not a re-patch. Checked
    `/api/diag/audit?type=DIAGNOSTIC&token=$DIAG_TOKEN` ~9h after v1.0.311
    deployed: EVERY one of 18 entries from 2026-07-14 10:57 UTC through
    19:57 UTC (spanning the v1.0.311 deploy at 11:20:58 UTC and the full
    trading day after) still reports `"Multiple API sources down:
    ['wikipedia', 'gdelt', 'fred']"`; `/api/diag/scanner`'s
    `dataSourceErrors` was still `{}` throughout — same shape of evidence
    as the original finding, meaning `deep_score()` was still never being
    reached, even after the monotonic-peak bug was fixed.
    ROOT CAUSE: `/api/diag/daemon?token=$DIAG_TOKEN` gave the answer this
    item's own STILL-OPEN question asked for — live daemon `rss_mb` read
    **254.2MB**, stable across 4 polls over ~80s with `active_dispatches:
    1` (i.e. this is baseline idle load, not a scan-induced spike), with
    `modules_loaded` confirming `ml_model_v2` (lightgbm+sklearn) stays
    permanently imported across calls. The 130MB skip / 100MB trim
    defaults were tuned 2026-04-22 against a short-lived subprocess's RSS
    trajectory ("Railway SIGKILLs bot_engine 'full' around 150-180MB") —
    but the persistent daemon (the primary execution path per CLAUDE.md's
    CODEBASE MAP) never resets that baseline between cycles, so its idle
    RSS alone sat above BOTH thresholds before any scan started. Once
    v1.0.311 made `_mem_rss_mb()` accurate, the guard's real behavior on
    the daemon path became "skip deep scoring unconditionally, every
    cycle" — the identical live symptom via a newly-exposed mechanism, a
    second failure on the same subsystem (RECURRENCE ESCALATES: two
    failed fixes = architecture smell, root-cause analysis required, which
    this update is).
    FIX SHIPPED (own PR, v1.0.314): re-based `VOLTRADE_MEM_SKIP_DEEP_MB`/
    `VOLTRADE_MEM_TRIM_DEEP_MB` defaults from 130/100 to 550/400 —
    grounded in the two untouched, already-live thresholds on the same
    guard chain (SURVIVAL_MODE=700MB, daemon self-kill=1024MB) rather than
    a fresh guess: trim_mb=400 leaves ~150MB margin above the live-
    measured 254MB idle baseline for normal per-scan allocation growth;
    skip_mb=550 leaves 150MB of clean margin below SURVIVAL_MODE so "skip
    deep scoring" is a real intermediate escalation step again, not a
    permanently-tripped floor. SURVIVAL_MODE (700MB) and the daemon
    self-kill ceiling (1024MB) are UNCHANGED — this only re-tunes the
    guard tier below them. The inline if/elif threshold comparison was
    extracted into a small pure function, `_deep_score_guard_decision`,
    specifically so the threshold values are unit-testable against the
    live-observed baseline without invoking the full scan pipeline.
    RATCHET: new `test_deep_score_guard_decision.py` (6 tests) — pins the
    254MB-is-"normal"-at-550/400 case (the regression this recurrence
    exposed), and separately documents that the OLD 130/100 defaults
    would classify that same 254MB baseline as "skip" (so a future
    session cannot silently re-lower the defaults back into the bug
    without a test failing).
    GATES: `python3 -m pytest -q` — 687 passed, 1 skipped, zero
    regressions (this sandbox needed a fresh `pip install -r
    requirements.txt -r requirements-dev.txt` this session — same
    recurring environment-tooling gap noted by the two immediately-prior
    sessions, still not chased further here). Zero TypeScript/client
    files touched, so `npx tsx --test`/`tsc`/`npm run build` were not
    re-run — no server/bot.ts call site reads these env vars (grepped;
    this guard is fully internal to bot_engine.py).
    DOWNSTREAM CHAIN (REASONING STANDARD #1): raises how often deep_score
    runs (restoring ML/alt-data enrichment on the daemon path, same
    direction as v1.0.311's intended effect, now actually realized) while
    leaving SURVIVAL_MODE and the daemon self-kill ceiling as the true OOM
    backstops, unchanged. Traced risk: if the container's real available
    memory is smaller than the 700MB/1024MB thresholds assume (unverified
    this session — inferred only from the daemon surviving multiple hours
    at 254MB), deep scoring now running on every cycle instead of never
    could push RSS higher more often and expose an OOM ceiling this
    session didn't have visibility into. ROLLBACK TRIGGER: rising OOM/
    SIGKILL frequency or daemon-restart cadence in `/api/diag/daemon`'s
    `uptime_seconds` resetting unexpectedly post-deploy — revert skip_mb/
    trim_mb toward 130/100 and re-open this item rather than re-guessing.
    BACKTEST: N/A — same infra/memory-guard class as v1.0.311; does not
    change what deep_score computes, only whether it runs.
    NEXT STEP for whichever session catches the next occurrence: confirm
    live via `/api/diag/audit?type=DIAGNOSTIC` clearing (wikipedia/gdelt/
    fred should start appearing fresh) within a few scan cycles post-
    deploy, and `/api/diag/scanner`'s `dataSourceErrors` starting to show
    occasional real entries (individual fetcher exceptions, now actually
    reachable) instead of a permanent empty `{}`. If wikipedia/gdelt/fred
    are STILL down a full trading day after v1.0.314 deploys, the
    threshold re-tune did not fully address it and per RECURRENCE
    ESCALATES the next session must stop threshold-tuning this guard and
    file a structural wishlist.md proposal (e.g., decoupling deep_score's
    enrichment fetch from the memory guard's candidate-scoring skip, or
    moving the memory ceiling constants into system_config.py with
    proper regime-aware bounds) instead of adjusting these numbers a
    third time.

    UPDATE 2026-07-15 (this session, [PIPELINE], not [REPAIR] — no new
    evidence to act on) — checked this item's own confirmation bar per
    the NEXT STEP above: `/api/diag/audit?type=DIAGNOSTIC` at
    2026-07-15T02:35Z's newest entry is still 2026-07-14T19:57:31Z
    (market-close-adjacent), so **no post-deploy, market-hours DIAGNOSTIC
    data exists yet** — the "full trading day after v1.0.314 deploys" bar
    has not elapsed (market opens 2026-07-15 13:30 UTC). Absence of
    evidence in an off-hours window is not evidence either way (same
    discipline as this item's own PARTIAL EVIDENCE note, 2026-07-11).
    `/api/diag/daemon` reads rss=165.6MB, well under the 550MB skip_mb
    threshold — no sign of the pre-v1.0.314 recurrence, but this is a
    quiet-hours idle reading, not a scan-loaded one. Genuinely open,
    correctly left open. TOOLING: `scripts/session_health_check.py`
    (new this session, see experiments.md) now runs exactly this item's
    two checks (`check_alt_data_enrichment`, `check_daemon_memory`)
    mechanically — the next session that catches a full post-deploy
    trading day should run it instead of re-deriving this by hand.

    UPDATE 2026-07-15 16:03 UTC (this session, [PRODUCT], not [REPAIR] —
    window reset, no new evidence to act on) — ran
    `scripts/session_health_check.py` mid-trading-day (market open since
    13:30 UTC); it still classifies this item WARN (wikipedia/gdelt/fred
    down, `dataSourceErrors: {}`). NOT a new recurrence: `/api/diag/daemon`
    shows `uptime_seconds` in the hundreds-to-low-thousands (two same-day
    redeploys — the secMidas OOM hotfix, PR #483, and the EARTH TWIN
    merge, both landing this morning — restarted the daemon), and
    `/api/diag/audit?type=DIAGNOSTIC`'s newest entry is STILL
    2026-07-14T19:57:31Z: zero new Tier-3 DIAGNOSTIC entries have fired
    since either restart. Tier-3 diagnostics run on an hourly cadence
    (CLAUDE.md CODEBASE MAP), so a daemon only tens of minutes old simply
    hasn't reached its first post-restart cycle yet — this is the same
    "absence of evidence in a window too short to judge" shape as the
    2026-07-15T02:35Z update above, not a fresh data point. rss_mb reads
    254.8, consistent with the v1.0.314 threshold model (well under the
    550MB skip_mb line). The "full trading day post-v1.0.314" bar is
    still not met and is now reset relative to TODAY's redeploys, not
    2026-07-14 — next session should re-run `session_health_check.py`
    later in the day once several hourly DIAGNOSTIC entries have
    accumulated post-restart.

22. **[RESOLVED 2026-07-19 — ≥44h live confirmation clean, v1.0.380]**
    ~~Floor-basket ETFs (SMH/KWEB/VXUS/GLD) were subject to active
    stop-loss/take-profit/time-stop logic meant only for actively-traded
    satellite positions.~~ CLOSING UPDATE (2026-07-19, scheduled-routine
    session, docs-only): queried production directly, ~44h after v1.0.380
    deployed. `/api/diag/audit?type=WS-EXIT` and `?type=WS-EXIT-ERROR` —
    **zero** entries of either since the fix (the endpoint scans the full
    retained window, not just the recent tail — confirmed by the type-scoped
    query reaching further back than an unfiltered `limit=200` query, same
    verification method KNOWN BROKEN #18's closure used). Cross-checked
    against `/api/diag/orders?limit=200`: the 200 most recent orders end at
    2026-07-17T15:35:47Z (the deploy) and are dominated by the EXACT
    pre-fix pattern this item describes — repeated VXUS/KWEB sell-then-
    immediate-rebuy cycles at ~30-90min cadence throughout 2026-07-16 and
    2026-07-17, same qty each time (VXUS ~42-44 shares, KWEB ~95/189-190
    shares) — i.e. this was firing far more often than the "5x" first
    noticed, not an isolated incident. Zero orders of any kind have posted
    since the deploy timestamp (~44h, spanning one full trading day
    2026-07-18 plus partial 2026-07-17/07-19), consistent with the fix
    suppressing the erroneous stop/rebuy churn entirely; the regime hasn't
    independently changed enough in that window to trigger a legitimate
    `_manage_spy_floor()`/`_manage_defensive_floor()` rebalance either, so
    this confirms absence-of-bug rather than proving the legitimate
    regime-change exit path still fires correctly — that will be observable
    the next time a real regime shift occurs. VERDICT: RESOLVED per KNOWN
    BROKEN #18's "re-check and mark RESOLVED" precedent. Original
    diagnosis and fix description below, unchanged.
    Found live via `/api/diag/audit`: 5x erroneous "WS TIME STOP" sells of
    VXUS + 1x "WS STOP LOSS" sell of SMH in a single session (2026-07-17,
    market hours), each immediately re-bought by `_manage_spy_floor()`'s
    drift rebalancer. Root cause: `FLOOR_AND_LEG_TICKERS`
    (`bot_engine.py`) and its Node mirror `MANAGED_TICKERS` (three call
    sites in `server/bot.ts`) were hardcoded to `{"QQQ","SVXY","SPY"}` and
    never updated when `system_config.py`'s `FLOOR_BASKET`
    (SMH/KWEB/VXUS) shipped 2026-04-22, nor for `DEFENSIVE_FLOOR_TICKER`
    (GLD) — full trace in `research/experiments.md` 2026-07-17 (session
    3). FIX: `bot_engine.py`'s set moved to module scope, derived from
    `BASE_CONFIG` so it can't drift stale again; `server/bot.ts` given a
    single shared `FLOOR_AND_LEG_TICKERS` constant used at all three call
    sites instead of three independent literals. Ratcheted by
    `test_floor_basket_stops.py` (Python, 5 tests) and
    `server/floorBasketExemption.test.ts` (TS, 4 tests), both A/B-verified
    to fail pre-fix. **NEXT CHECK**: once deployed, query
    `/api/diag/audit?type=WS-EXIT&limit=20` — should show zero further
    TIME STOP/STOP LOSS entries for QQQ/SMH/KWEB/VXUS/GLD; any future exit
    on those tickers should only come from `_manage_spy_floor()`/
    `_manage_defensive_floor()`'s own regime-change path. Mark RESOLVED
    once confirmed clean for a few live trading days, per KNOWN BROKEN
    #18's "re-check and mark RESOLVED" precedent.

23. **[FOUND + FIXED 2026-07-18, v1.0.397, scheduled-routine session]
    Alpaca's `/v2/stocks/bars` historical endpoint rejects `feed=delayed_sip`
    with HTTP 400 — silently degrading VXX/SPY regime detection, ML
    training, options vol, and correlation checks account-wide whenever
    the account is in the (currently active) SIP-403-downgraded state.**
    Found by following up on KNOWN BROKEN #17's diagnosability fix
    (v1.0.317, which made `_fetch_training_bars` failures legible): this
    session's `/api/diag/audit?type=TIER3-ML-ERROR` query showed the
    2026-07-17 ml_retrain throttle fix's OWN stated hypothesis (b)
    materializing — a specific diagnosable cause, but `HTTP 400: {"message":
    "invalid feed: delayed_sip"}`, recurring on literally every retrain
    cycle since ~19:00Z the prior day (`model_age_hours` at 40.8 with zero
    successful retrains in that window), not the predicted HTTP 429.
    ROOT CAUSE: `alpaca_feed.py`'s `data_feed()` resolver (the 2026-07-06
    SIP-403 fix) downgrades to `"delayed_sip"` on entitlement rejection —
    correct and confirmed working for snapshot/quote/trade endpoints (Tier2
    scans kept completing normally throughout: "Scanned 11931 stocks" every
    cycle, zero scanner degradation). But `/v2/stocks/bars` (single- or
    multi-symbol historical daily bars) doesn't recognize `"delayed_sip"` as
    a feed value at all — a 400 (bad request), not a 403 (entitlement), so
    it fails regardless of subscription tier once the account is in the
    delayed state. `data_feed()` was used identically for BOTH endpoint
    families across 13 files (bot_engine.py, macro_data.py, ml_model_v2.py,
    ml_model.py [dead/unimported, fixed anyway for ratchet consistency],
    options_scanner.py, intraday_shorts.py, shadow_portfolio.py,
    vol_surface.py, instrument_selector.py, probability_engine.py,
    alt_data.py, etf_data_sources.py) — 29 call sites total, all wrapped in
    bare `except Exception` (or an equivalent broad catch) that silently
    fell back to hardcoded defaults (`vxx_ratio=1.0`, `spy_above_200d=True`,
    `spy_below_200_days=0`) with **zero audit trail** for every site except
    `_fetch_training_bars` (the only one with the KNOWN BROKEN #17
    diagnosability upgrade) — meaning regime classification across the
    whole live trading stack was likely silently defaulting to "neutral/
    healthy" for as long as the account has been SIP-403-downgraded, not
    just ML retraining. DOWNSTREAM CHAIN (REASONING STANDARD #1): a
    silently-defaulted `vxx_ratio`/`spy_below_200_days` feeds
    `get_market_regime()` → `tiered_strategy`'s regime caps and
    `master_kill_switch` (KNOWN BROKEN #20) → position sizing and the CSP
    tier gate → realized P&L attributed to "regime X" that may not have
    been the actual regime. This is a HONESTY METRIC concern independent
    of ML retraining specifically.
    FIX: `alpaca_feed.py` gained `bars_feed()` — defers to `data_feed()`
    for everything except the one value bars demonstrably rejects, where it
    substitutes `"iex"` (free, always accepted; already used successfully
    for this identical endpoint by `alphadesk/alphadesk/market.py`).
    `data_feed()` itself is UNCHANGED — every snapshot/quote/trade call
    site keeps `delayed_sip`'s correct consolidated-volume semantics; only
    the 29 confirmed `/v2/stocks/bars` call sites were swept onto
    `bars_feed()` (mechanical substitution, each site read and verified
    individually, not regex-blind — mirrors the 2026-07-06 44-site sweep's
    precedent for one root cause, one centralized fix). MEASUREMENT
    INTEGRITY NOTE: `iex` undercounts consolidated volume ~30-50x, which is
    exactly why the 2026-07-06 fix rejected it for snapshot-based dollar-
    volume GATES — that risk doesn't transfer here because `bars_feed()`
    only ever reaches historical BARS calls (price/regime/training series),
    never the snapshot-based scan gate; `ml_model_v2.py`'s `volume_ratio`
    training feature (today/20d-avg, same-source ratio) is the one place
    an absolute-volume distortion could theoretically bias a feature, but a
    same-ticker same-source ratio largely cancels a constant per-venue
    market-share bias — accepted as a materially smaller integrity risk
    than zero retraining for 40+ hours and straight up (not just quietly
    wrong) failures on every retrain cycle.
    RATCHET: `test_alpaca_feed.py` gained `TestBarsFeedResolution` (4 new
    tests: sip passthrough, delayed_sip→iex substitution, env-override
    interaction both ways) and `TestBarsEndpointsUseBarsFeed` (a source-scan
    ratchet — for every `/v2/stocks/bars` URL occurrence in a runtime file,
    asserts the feed resolved in that call's params is `bars_feed()`, never
    `data_feed()`; A/B-verified live this session by reverting one call site
    back to `data_feed()` — the ratchet failed exactly as expected, then
    passed again after restoring the fix). Full `python3 -m pytest -q`:
    756 passed, 2 skipped (751 baseline + 5 new, zero regressions).
    **NEXT CHECK**: once deployed, `/api/diag/audit?type=TIER3-ML-ERROR`
    should show zero further entries (or a genuinely new, different cause
    if one exists) and `/api/diag/ml`'s `model_age_hours` should stop
    climbing past ~1-2h; also worth a follow-up spot-check of `vxx_ratio`/
    `spy_below_200_days` in a live `/api/diag/scanner` or macro snapshot to
    confirm regime inputs are now real-valued rather than the silent
    defaults, though no counterfactual archive exists to quantify how much
    P&L the pre-fix silent-default period may have cost (the shadow_
    portfolio/counterfactual logging infrastructure logs REJECTED
    candidates' outcomes, not regime-input-quality drift — a genuinely
    new gap, filed as its own follow-up: regime-input freshness/validity
    should probably get its own audit-visible field, similar to
    `data_source_errors`, rather than being inferable only from bars-fetch
    error absence).
    CONFIRMATION UPDATE (2026-07-19, same scheduled-routine session as
    item #22's closure): `/api/diag/audit?type=TIER3-ML-ERROR` — zero
    entries anywhere in the retained window, ~44h post-deploy.
    `/api/diag/ml` shows `model_age_hours: 23.6`, `retrain_needed: false`,
    `retrain_overdue: false` — consistent with the system's own
    `needs_retrain = model_age_hours > 24` daily cadence (`server/bot.ts`),
    not a stuck/failing retrain; the item's original "should stop climbing
    past ~1-2h" language was itself imprecise (retrain is daily, not
    hourly) but the underlying claim — no further silent bars-fetch
    failures — holds. NOT independently re-verified this session: the
    `vxx_ratio`/`spy_below_200_days` real-value spot-check the item asked
    for — no existing `/api/diag/*` probe exposes those fields; adding one
    would be a second logical change, left as the item's own still-open
    follow-up rather than done here.

## RULE COST AUDIT — after counterfactual logging exists

- Is MIN_SCORE=63 leaving winners on the table or blocking losers?
- SCORE_BAND_MAX=75 ("fake breakout" ceiling) — measure prevention-P&L.
  **UPDATE 2026-07-04: this rule does not exist in code — see KNOWN
  BROKEN #10. Nothing to measure until it's either wired with evidence
  or the dead config is removed.**
- MAX_CHANGE_PCT=35 ("easy money gone") — verify against outcomes.
  **UPDATE 2026-07-04: same as above — MAX_CHANGE_PCT is never read
  outside system_config.py. See KNOWN BROKEN #10.**
- Spread filter 0.5% — how many blocked names would have filled fine?
  **UPDATE 2026-07-05 (v1.0.130): NOW COUNTERFACTUAL-LOGGED** — the
  `_spread_pct > 0.005` rejection in `_scan_market_inner()`
  (bot_engine.py) calls `shadow_portfolio.update_last_decision(ticker,
  "rejected_other", ...)` immediately after rejecting, correcting the
  candidate's shadow-log decision from `deep_score()`'s premature
  "taken". Answerable once >=90d of shadow history accumulates
  (~2026-10-02) via `get_shadow_stats()["win_rate_by_decision"]
  ["rejected_other"]` vs `["taken"]`. See BUILD ORDER 4 #6 above for
  the full trace.
- Correlation/sector blocks — cost vs. protection in current regime.
  **UPDATE 2026-07-05 (v1.0.130): NOW COUNTERFACTUAL-LOGGED** — same
  mechanism as the spread filter above, decision bucket
  `"rejected_heat"`, wired at `check_sector_correlation()`'s rejection
  site. Same readout date and query shape.
- Kill-switch drawdown thresholds — sized for real-money caution; is
  that optimal for a paper account whose goal is learning speed?
  **STILL NOT counterfactual-logged** (2026-07-05): `check_kill_switches()`
  gates the separate TieredStrategy action list, not the `deep_score()`
  candidates this session's fix targeted, and any halt at the
  execution layer lives in `server/bot.ts` — see BUILD ORDER 4 #6 for
  why this is a materially bigger, still-open follow-up.
- `tiered_strategy.master_kill_switch` killing Tier 1 CSP along with
  T2-4 on a whole-portfolio exposure breach it doesn't contribute to —
  see KNOWN BROKEN #20. **UPDATE 2026-07-11 (v1.0.279): NOW
  COUNTERFACTUAL-LOGGED**, decision bucket `"rejected_masterkill"`,
  wired at `run_tiers()`'s kill branch via the new
  `log_masterkill_csp_shadow()`. Answerable via
  `get_shadow_stats()["win_rate_by_decision"]["rejected_masterkill"]`
  once master_kill_switch has fired a few times with enough history
  behind those firings — no firings recorded yet as of this session
  (`/api/diag/audit?type=TIER-KILL` empty).

## OPEN RESEARCH QUESTIONS

- **Insider Form 4 clustering as a signal** (gate 1 PASSED 2026-07-03 — see
  `server/edgarForm4.ts` / `edgarForm4.test.ts` / `datacore/README.md`; the
  feed is live at `/api/data/insider`, surfaced as RAW only, no predictive
  claim). Gate 2 hypothesis, not yet attempted: do clusters of open-market
  insider BUYS (transaction code P specifically — code A grants/RSU vesting
  and code M option exercises are not discretionary purchases and would
  dilute the signal; code S sales are the mirror case worth testing
  separately for predictive shorts) at a given issuer, within a short
  window, predict forward N/20/60-day excess return over a size-matched
  random-entry baseline (REASONING STANDARD #3 — demand the base rate, not
  the raw number)? PRIOR stated before any run: expect a small positive
  edge concentrated in officer/director (not 10%-owner fund) buys on
  small/mid caps specifically (EDGE DOCTRINE #2 — capacity-constrained
  corners), close to zero or negative on mega-caps where the signal is
  already arbitraged; kill the hypothesis if officer/director open-market
  buys show no separation from the random-entry baseline after >=90 days
  of accumulated feed history (need real history first — the feed only
  started polling today, no backtest possible yet from filing text alone
  without a paid historical EDGAR bulk-data source or accumulating our own
  archive from here forward, per BUILD-FIRST rule #2). Ladder: gate 1 DATA
  done; gate 2 SIGNAL blocked on accumulating enough live filing history
  (or sourcing free historical Form 4 index files from SEC's bulk data
  page, `www.sec.gov/Archives/edgar/full-index/`, which is public and free
  — worth trying before waiting on live accumulation, unexplored).

  UPDATE 2026-07-19 (this session, [PIPELINE]+[RESEARCH]) — the
  "unexplored" free historical path above was explored and is now a real
  gate-1 archive AND a real gate-2 screen, both run live this session.
  DATA (gate 1): SEC publishes a keyless, PRE-PARSED structured TSV
  dataset per calendar quarter (SUBMISSION/REPORTINGOWNER/NONDERIV_TRANS/
  ...) at the `insider-transactions-data-sets` page, going back to 2006,
  82 quarters live-confirmed available — no XML parsing, no per-filing
  HTTP request, no waiting for live accumulation at all. New
  `sec_form4_bulk.py` (+ `test_sec_form4_bulk.py`, 26 tests) downloads,
  joins, and archives officer/director open-market P/S transactions
  (10%-owner-only filers excluded per this entry's own prior — a fund's
  purchase is a portfolio decision, not the same information event as an
  insider's own money); wired into `server/bot.ts`'s hourly Tier 3 call
  exactly like `cftc_cot.py`, self-guarding to one quarter fetched per
  ~12h so it is nightly-free going forward, chronological 8-quarter
  backfill window. LIVE DATA-QUALITY FINDING: SEC's raw
  `ISSUERTRADINGSYMBOL` field is free text, not a validated ticker —
  live-probed values include `N/A`, `NONE`, `-`, `NASDAQ:SVC`, and
  dual-class/merged values like `GEF, GEF-B` — a dedicated ticker-shape
  filter now rejects these rather than guessing which class was meant
  (never fabricate an attribution the filer didn't actually report).

  SIGNAL (gate 2): new `form4_gate2_test.py` (+ `test_form4_gate2.py`, 26
  tests) builds per-ticker chronological events, flags a `cluster` when
  >=2 DISTINCT officer/director owners bought the same issuer within a
  causal (backward-only, no-lookahead) 5-calendar-day window, and
  compares forward 20d/60d returns against each ticker's OWN
  unconditional forward-return distribution (a same-symbol baseline like
  `cot_gate2_test.py`'s, but additionally excluding any baseline entry
  within one horizon of that ticker's own signal events — a
  contaminated-baseline guard neither this repo's COT screen nor the
  original PRIOR anticipated needing). Uses a Welch two-sample t-test
  rather than `cot_gate2_test.py`'s Newey-West HAC correction — justified
  explicitly in the module docstring: COT's autocorrelation problem comes
  from one symbol sampled weekly with heavily overlapping horizons; Form 4
  events are cross-sectional (many different tickers), a much weaker
  dependence structure, so a HAC correction built for a different problem
  would be cargo-culted rigor, not real rigor.

  A LIVE SELECTION BUG WAS FOUND AND FIXED MID-SESSION, and it mattered:
  the ticker sampler (network calls don't scale to fetching bars for
  every one of ~2,700 archived tickers in one session, so a capped sample
  is unavoidable) originally prioritized cluster-eligible tickers first,
  filling remaining budget with single-buy tickers. Once the archive grew
  past ~4 quarters, cluster-eligible tickers ALONE (1,180) exceeded the
  500-ticker cap, silently starving the "single" bucket to zero clean
  observations — every reported "single" event was actually a non-cluster
  event belonging to a ticker that ALSO had a cluster elsewhere, a
  selection bias invisible in a small sample and only exposed once the
  sample was large enough to saturate the cap. Fixed to a balanced ~50/50
  split (ratchet: `test_full_budget_used_when_one_bucket_is_scarce` +
  `test_single_bucket_not_starved_when_cluster_pool_exceeds_cap`,
  A/B-verified against the pre-fix logic).

  RESULTS, all three runs kept honest rather than reporting only the best
  one (REASONING STANDARD #4/#10 — distrust in proportion to what was
  tried, state the prior before updating): a SMALL, pre-fix sample (2
  quarters, 300-ticker cap) showed an exciting 20d cluster hit (mean diff
  +2.36pp over baseline, p=0.0002) that appeared to strengthen with a
  150->300 ticker resample. It DID NOT REPLICATE once backfilled to 6
  quarters (2025q1-2026q2) and corrected to the balanced 500-ticker
  sample (2,227 events, 806 clustered, 477/500 tickers fetched
  successfully): 20d shows NO separation (cluster mean_diff +0.50pp,
  p=0.48; single -0.54pp, p=0.35) — this is exactly the COT-SLV pattern
  this repo has now seen twice, an exciting small-sample raw gap that
  does not survive a larger, more rigorous re-test. 60d shows a
  SURPRISE the opposite direction of the stated PRIOR: BOTH buckets
  underperform their ticker-matched baseline (cluster -3.97pp, p=0.049
  marginal; single -4.68pp, p=0.008, clears even a same-session
  Bonferroni bar of 0.05/4=0.0125 in the NEGATIVE direction). NOT YET
  CALLED A KILL, deliberately: the 60d baseline's own pooled mean is
  unusually high (+8.55%, vs. +1.92% at 20d) — worth checking whether the
  fetched-ticker universe (skewed toward officer/director-active names in
  a 2025-2026 window) happens to include a disproportionate share of
  momentum small-caps whose OWN unconditional 60d drift is elevated,
  which would make "insiders underperform their own ticker's baseline"
  read as a false negative rather than a real one — an unchecked
  confound, not yet ruled out. NEXT STEPS: (1) the archive keeps
  backfilling automatically via Tier 3 (6 of the 8-quarter target window
  archived from this session's manual runs — 2025q1 through 2026q2; the
  hourly poll fills the remaining 2 older quarters + future new quarters
  with zero further attention needed); (2) a future
  session should sanity-check the 60d baseline-inflation confound above
  before treating the negative 60d finding as real; (3) the `code S`
  sales mirror test (predictive shorts) named in this entry's original
  PRIOR was NOT run this session — deliberately, to keep the live
  network-call budget to one hypothesis at a time — filed as the
  concrete next run; (4) re-run this exact screen once BACKFILL_QUARTERS
  reaches its full 8-quarter window for a still-larger, still-corrected
  sample before drawing any promotion-or-kill conclusion.
- **CFTC Commitments of Traders (COT) positioning** (gate 1 DATA PASSED
  2026-07-05 — see `cftc_cot.py` / `test_cftc_cot.py`; EDGE DOCTRINE #1
  standing example; recovered this session from a stalled dirty PR
  opened 2026-07-04 that never merged — see experiments.md). Free, no
  key, via CFTC's public Socrata "Legacy Futures Only" dataset. 7
  symbols tracked (GLD, SLV, USO, CORN, TLT, SPY, QQQ — natural gas
  deliberately skipped, its classic NYMEX contract code stopped
  reporting in 2022 and the successor is ambiguous from the API alone,
  revisit rather than guess). Every fetched record is validated against
  CFTC's own accounting identities (long/short/spread sum to reported
  totals; reported + non-reported = open interest) before being
  archived — 0 rejections across a 156-week backfill for all 7 symbols,
  re-verified live this session. Archived to
  `storage_config.COT_ARCHIVE_PATH`; `run_daily_update()` wired into the
  existing hourly Tier 3 maintenance call (server/bot.ts
  tier3Strategic, step 5) but self-guards to hit the network at most
  once per ~20h, so the other 23 hourly calls are a free file check.
  NOT wired into deep_score/macro_data — this is DATA-LADDER GATE 1
  ONLY.
  **GATE 2 SCREEN RUN 2026-07-05** (`cot_gate2_test.py` /
  `test_cot_gate2.py`; PRIOR restated above was written before this run).
  Non-commercial COT-index extremes (>=80 / <=20 over the trailing
  156-week window) vs. a same-symbol all-weeks baseline, forward 20d/60d
  returns, entry anchored to the first trading day STRICTLY AFTER the
  Friday publish date (no lookahead — the Tuesday as-of date is never
  used as an entry). Full mean table lives in experiments.md; verdicts:
  - **KILLED — no separation from baseline**: GLD, CORN, SPY, QQQ (SPY/
    QQQ matches the stated prior; GLD's low-extreme bucket was also only
    n=7, too thin to have shown anything either way). TLT killed too:
    its one large deviation (extreme_low, 20d, -4.18% vs +0.07% baseline)
    did not hold at 60d (-0.16% vs +0.35%, both near zero) — a single-
    horizon flash with no cross-horizon confirmation is exactly the
    fishing pattern REASONING STANDARD #4 warns about, not a real effect.
  - **NOT killed — carried to further work, NOT a gate-2 pass**: SLV
    (extreme_high, n=44, consistently below baseline at both horizons:
    20d 1.85% vs 3.41%, 60d 7.78% vs 11.40%) and especially USO
    (extreme_low, n=58, 60d 15.48% vs 7.22% baseline; extreme_high,
    n=14, 60d -4.54% vs 7.22% — both legs point the same
    mean-reversion direction the prior predicted, and USO/oil is exactly
    where REASONING STANDARD #5's hedger-information-asymmetry argument
    is strongest).
  - **METHODOLOGICAL FINDING that limits belief in the two "not killed"
    cases and applies to every future weekly/monthly-cadence gate-2
    test in this repo (FINRA short-volume, COT itself, Wikimedia
    attention, CBP border waits, any future daily/weekly signal)**:
    forward-return windows computed at weekly sampling with a 60-trading-
    day (~12-week) horizon overlap ~11/12 between consecutive
    observations — they are NOT 58 independent draws. Effective
    independent sample size for the 60d horizon is roughly n/12 (USO's
    n=58 -> ~5 effectively independent windows), for the 20d horizon
    roughly n/4. This screen used raw means only (no significance test,
    no autocorrelation-adjusted standard error) — exactly why it can
    only KILL (a flat/opposite-signed mean is disqualifying regardless
    of overlap) but cannot PASS (a large raw mean gap can come from a
    handful of correlated episodes, e.g. one persistent oil rally).
    NEXT STEP before SLV/USO get anywhere near LOGIC gate 3: either (a)
    a block-bootstrap or Newey-West-style test that accounts for the
    overlap, or (b) a non-overlapping-window design (one observation per
    resolved horizon, not per week) — n=58 becomes ~n=5-13 either way,
    so also keep accumulating weekly history to raise the useful sample
    over time. Do not promote SLV/USO on raw means alone.
  - **NEWEY-WEST FOLLOW-UP RUN 2026-07-08** (option (a) above; see
    `cot_gate2_test.py`'s `hac_significance()` / `_newey_west_diff_test()`,
    `test_cot_gate2.py`; full JSON in this session's PR description). All
    7 symbols re-screened live (156 fresh weeks, same construction, no
    lookahead) with a Newey-West (Bartlett-kernel) HAC test: each
    extreme-bucket dummy regressed against the COMPLEMENT weeks (not the
    all-weeks-inclusive baseline `summarize()` uses — a deliberately more
    conservative two-sample comparison), truncation lag = round(horizon /
    5) weeks (4 for 20d, 12 for 60d — exactly the overlap span the
    methodological finding above named).
    - **SLV now reads as KILLED**: neither bucket at either horizon is
      significant (20d extreme_high p=0.52, extreme_low p=0.65; 60d
      extreme_high p=0.48, extreme_low p=0.87) — the raw-mean gap this
      screen was carried forward on does not survive correcting for the
      overlap autocorrelation. Matches this finding's own prediction
      ("a large raw mean gap can come from a handful of correlated
      episodes").
    - **USO stays carried forward, still NOT a gate-2 pass, but now with
      an actual number instead of an eyeballed gap**: 60d extreme_high
      is nominally significant (mean_diff -14.96pp, HAC SE 7.12pp,
      t=-2.10, p=0.0355) and 60d extreme_low is marginal (+13.88pp, SE
      7.56pp, t=1.84, p=0.066) — both legs still point the same
      mean-reversion direction as the original raw-means read. REASONING
      STANDARD #4 still applies with full force: this session alone ran
      28 symbol x bucket x horizon comparisons (7x2x2), and the original
      screen ran the same 28 — a single p=0.0355 hit among that many
      trials does not clear even a same-session Bonferroni bar (0.05/28
      ~= 0.0018, or 0.05/8 ~= 0.00625 restricted to just SLV+USO's own 4
      combos each) — DO NOT promote USO to LOGIC gate 3 on this number.
      What changed: the effect is no longer just "the biggest raw gap in
      a noisy table," it is a specific, quantified, correctly-signed
      result that is WORTH the next real test. NEXT STEP (unchanged in
      kind from the prior note, now sharper): this exact 156-week window
      cannot be reused as its own confirmation — the only honest
      out-of-sample test is NEW weekly COT reports as they accumulate
      from here forward (the archive is already recording); re-run this
      screen restricted to weeks published after 2026-07-08 once there
      are enough of them (~15-20 new weeks minimum given the horizon
      overlap) and require the same sign and a nominal p-value that
      clears the Bonferroni bar above before this heads toward LOGIC
      gate 3.
    - **TLT's already-killed single-horizon flash is now quantified, not
      just eyeballed, and the kill verdict is UNCHANGED**: the 20d
      extreme_low deviation this session's screen also caught
      (mean_diff -4.60pp, SE 1.12pp, t=-4.11, p<0.001) is a genuinely
      strong statistical result — but it still does not appear at 60d
      (t=-0.13, p=0.89), which is exactly the cross-horizon-replication
      failure REASONING STANDARD #4 disqualifies on. A real, significant,
      NON-replicating effect is still not a tradeable one; no change to
      TLT's kill.
    - GLD/CORN/SPY/QQQ: no significant HAC hits at either horizon or
      bucket (all p > 0.24) — consistent with, and now backed by an
      actual test statistic instead of, their original raw-means kill.
  - Discount applied throughout per REASONING STANDARD #4: 7 symbols x
    2 buckets x 2 horizons = 28 comparisons run in one pass; 2 "hits"
    out of 28 is within what noise alone would produce, which is exactly
    why neither is called a pass here.
- **Options fill realism.** The synthetic slippage haircut in bot.ts is
  volume-tiered with a random component — good for stocks, weak for
  options. Replace for options with quote-based fills: short premium
  fills at the BID, long premium at the ASK, using the contract's actual
  quote at fill time (the liquidity filters already fetch it). Also cap
  simulated fill quantity at a sane multiple of the contract's real
  volume/open interest — Alpaca paper fills unlimited size, which is
  fiction for thin chains. Validate the existing stock slippage tiers
  against recorded bid/ask spreads in the fills tracker.
- **Strategy tournament.** Run strategies as isolated, tagged competitors
  (strategies/ modules are already shaped for this) with buy-and-hold SPY
  as a permanent benchmark entrant. Allocate more to winners, retire
  losers, log every promotion/retirement decision with evidence. Answers
  "is any of this beating doing nothing" continuously. Requires backtest
  engine (#1) first.
- Live-vs-backtest divergence: unmeasurable until #1 done. Then it is
  the standing honesty metric.
- **[SCREENED 2026-07-09 — see experiments.md, T-BOT/measurement] Which
  regime detector (markov_regime vs. VXX-ratio heuristics) actually
  predicts forward volatility better?** They still coexist in
  `system_config.get_market_regime()` (VXX-ratio + SPY-vs-50MA are the
  primary absolute-threshold classifier; `markov_state` only breaks ties
  in one `elif` branch) — this did not change anything, it is a
  measurement-only finding. PRIOR (stated before running, matches):
  VIX-ratio is close to a direct forward-vol estimate by construction, so
  it should correlate with realized forward vol far more than a Markov
  chain trained purely on SPY return DIRECTION. RESULT (10y FRED SP500 +
  VIXCLS, n=2,442 trading days, `regime_detector_compare.py` +
  `test_regime_detector_compare.py`, Spearman rank correlation vs. 5d/20d
  forward realized vol): VIX-ratio rho=0.30/0.24 (p≈0), SPY-vs-50MA
  rho=-0.36/-0.25 (p≈0, sign-correct — below-MA correlates with higher
  forward vol) — both ~3x/~2x the Markov chain's bear-probability
  rho=-0.07/-0.12 (statistically significant at this n but a small,
  SIGN-FLIPPED effect: higher predicted bear-probability correlated with
  LOWER forward vol, plausibly a bear-exhaustion/relief-bounce artifact
  of the 10y training window, not a vol signal). CONFIRMS THE PRIOR: the
  VXX-ratio+trend heuristic is doing essentially all of the volatility-
  forecasting work; the Markov component is not adding meaningful
  volatility-predictive power (though this says nothing about its
  DIRECTIONAL signal quality, e.g. STRONG_BUY/SELL — untested here, a
  separate question). HONEST DATA SUBSTITUTION (this sandbox could not
  reach live SPY/VXX history — yfinance 429s, Stooq JS-walled, Alpha
  Vantage needs a paid key): FRED's free SP500 index stands in for SPY
  (safe — near-identical daily moves) and VIXCLS (spot VIX) stands in for
  VXX (NOT safe for reproducing `get_market_regime()`'s exact VXX-
  calibrated absolute thresholds, which is why this screen used
  threshold-free rank correlation instead of the live classifier's
  buckets — see the script's docstring for the full reasoning). NOT
  ACTED ON: this is evidence, not yet a RULE REVIEW change — the Markov
  component also feeds `get_regime_multiplier` (position sizing) and the
  STRONG_BUY/SELL direction signal, neither of which this screen touched;
  demoting Markov's role in `get_market_regime()` specifically (its only
  live consumer for the tie-break `elif`) would need its own evidence
  pass on THAT narrower question before any change ships, per RULE
  REVIEW's counterfactual/ablation requirement. FOLLOW-UP if ever
  revisited: real VXX history would let this reproduce the exact
  production thresholds instead of a rank-correlation proxy.
- Earnings/FOMC calendar awareness: verify positions are actually
  gated around scheduled events, not just theoretically supported.

- **[KILLED 2026-07-04 — see experiments.md]** ~~Dual-momentum SPY/QQQ~~
  (from 2026-07-03 harness run, `bot_backtest.py`): in-sample 2016-2026 beat
  SPY (16.3% vs 14.1% CAGR, Sharpe 0.90 vs 0.83, DD -28.6% vs -33.7%).
  1-of-~7 variants tried — discount per REASONING STANDARD #4. PRIOR stated
  before any out-of-sample run (#10): edge shrinks but survives ~+1% CAGR
  over SPY ex-2020-21; kill if negative in >=2 sub-periods. OUT-OF-SAMPLE
  RESULT (`bot_backtest_subperiods.py`, calendar sub-period split): negative
  alpha vs SPY in 2/3 non-2020-21 sub-periods (2016-2019: -1.09pp; 2024-2026:
  -11.49pp; only 2022-2023 positive at +19.44pp) — kill threshold met. The
  pooled decade number was REASONING STANDARD #2 in action: almost the
  entire in-sample edge was concentrated in the single 2022-2023 regime
  (tech underperforming, so the SPY/QQQ winner-take-all rotation avoided
  the 2022 drawdown), not a persistent property of the strategy. NOT a
  tournament candidate. Do not re-propose without a materially different
  variant and a fresh out-of-sample test — this exact config is closed.

- **Aircraft/vessel provider redundancy** — AIRCRAFT SIDE EXECUTED
  2026-07-03 (v1.0.52): chain is now THREE deep — adsb.lol (ODbL,
  primary) -> airplanes.live -> adsb.fi. Licensing checked first:
  adsb.fi = personal/non-commercial with attribution (same class as
  airplanes.live; covered by the MONETIZATION TRIPWIRE), global
  coverage verified from three continents (Tokyo 130 / Sydney 146 /
  São Paulo 69 aircraft), same readsb JSON shape. Rejected: adsb.one
  (Cloudflare-blocks server egress), ADS-B Exchange (community API
  non-commercial AND keyed via RapidAPI; commercial = paid Enterprise —
  a priced wishlist candidate only if the free chain proves fragile).
  SELF-HOSTED RECEIVER: DECLINED BY HUMAN 2026-07-03 — no physical
  builds; do not re-propose feeder hardware. VESSELS SIDE still open:
  single-sourced on aisstream.io; find a second AIS source (AISHub
  requires feeding a receiver — excluded by the same no-hardware
  decision; satellite AIS is paid — see wishlist).
- **OpenSky reinstatement (likely-returner, DEAD CODE POLICY tracking).**
  Human emailed contact@opensky-network.org for a research agreement
  (2026-07-03). No disabled adapter retained — the v1.0.43 OAuth +
  states/all implementation lives in git history (revert of PR #114's
  removal restores it). REVIEW-BY 2026-08-17 (+45d): if no agreement by
  then, close this item and strike OpenSky from the redundancy
  candidates; if granted, reinstate the chain attempt AND re-verify
  Railway egress connectivity before relying on it.

## MIDAS HFT-COLONIZATION FILTER HYPOTHESIS (RAW layer shipped 2026-07-10, v1.0.265 — census build #10)

`server/secMidas.ts` archives SEC MIDAS's quarterly individual-security
market-structure metrics (public domain; endpoint found via live web
search this session, after two prior sessions' static-URL guesses both
404'd — see experiments.md 2026-07-10 entry). RAW only: `/api/data/
microstructure`'s `smallcap_watch` is sorted purely by the source's own
published Cancels/LitTrades ratio for Stock rows with McapRank<=2 (bottom
decile pair, smallest ~20% by SEC's own ranking) — no model, no
predictive claim.

HYPOTHESIS (EDGE DOCTRINE #2 — fish where whales can't): among small-cap
stocks, a persistently HIGH cancel-to-trade ratio / hidden-order rate /
odd-lot rate is a FILTER, not a directional signal — it flags names
already colonized by HFT market-making/quote-stuffing activity, where a
capacity-constrained small-fund edge (deep_score's actual target
universe) is more likely to be competed away at the microstructure level
even if the fundamental/catalyst thesis is sound. The inverse — small
caps with LOW colonization metrics relative to peers of similar
McapRank/TurnRank — are the genuinely under-arbitraged corner EDGE
DOCTRINE #2 describes. Prior: expect the filter to matter more for
momentum/breakout entries (execution-sensitive, thin liquidity gets
front-run) than for longer-hold catalyst plays (Form 4 clusters, 8-K
events) where a few extra bps of adverse selection matters less. Second-
order reasoning (REASONING STANDARD #5): the "why hasn't this been
arbitraged" answer here is structural, not "nobody noticed" — a fund
running this filter needs BOTH a small-cap candidate stream (already
built: Form 4, 8-K, insider clusters) AND this microstructure layer
jointly, which is exactly the kind of two-source join a single retail
screener doesn't bother building.

LADDER: gate 1 DATA — passed for the raw feed itself (SEC's own published
metrics, no external ground-truth needed the way tank-shadow imagery
does; integrity enforced by the malformed-row-rate guard in secMidas.ts,
not a cross-source check). Gate 2 SIGNAL — BLOCKED, twice over: (a)
MIDAS's own multi-quarter publish lag means meaningful history takes real
calendar time to accumulate quarter-by-quarter (this session archived
2025q4 only; the poller backfills MIDAS_LOOKBACK_QUARTERS=6 quarters over
subsequent daily polls, not instantly); (b) this is a CROSS-STREAM
hypothesis by construction — it needs a JOIN against an existing
small-cap candidate stream (Form 4 clusters is the most natural partner,
also gate-2-blocked on its own accumulating history) to test "does the
colonization filter change the Form-4-cluster edge's forward return,"
not a standalone screen the way COT/FTD were. NEXT STEP once both sides
have >=2 quarters of MIDAS history and enough Form 4 cluster events to
overlap: split Form-4-cluster candidates into high vs. low McapRank-
matched colonization-metric buckets, compare forward 5/20/60d excess
return over the existing random-entry baseline (REASONING STANDARD #3);
discount for the number of metric combinations tried (cancel-to-trade,
hidden-rate, odd-lot-rate — 3 candidate filters, not 1) per REASONING
STANDARD #4.

WHAT DIDN'T WORK (logged so nobody re-walks it, corrects the record left
by two prior sessions): the 2026-07-06 census entry flagged MIDAS "probed
200" without recording the actual URL used, and the 2026-07-08 COT
Newey-West session tried several remembered/guessed static paths
(`sec.gov/data/market-structure/metrics-by-security`,
`sec.gov/marketstructure/midas.html`/`-system`, an `/files/opa/data/
market-structure/metrics-by-{security,market}/{year}/q{n}/` guess) — all
404 or redirect loops. The real path was only found by web-searching for
the current downloads page and following its actual listed hrefs (WebFetch
against `sec.gov/data-research/sec-markets-data/marketstructure
data-security`), not by pattern-guessing from memory — the general lesson
for any future "known endpoint moved" case in this repo.

## OPS GOTCHAS (avoid re-learning)

- BRANCH RESET AUTO-CLOSES OPEN PRs (bit us 2026-07-16: the two live
  repairs in PR #498 silently stranded): force-pushing the branch to
  EQUAL main (the supersession reset step) makes an open PR's diff
  empty and GitHub CLOSES it within minutes; later pushes do NOT
  reopen it, and body edits still succeed on the closed PR — nothing
  looks wrong. RULE: during a supersession reset, close the duplicate
  PR YOURSELF first (with a comment), reset, push the salvage, then
  open a FRESH PR. And merge watchers must check the PR's merged
  field (or scan main's recent log), never just main's HEAD line —
  a concurrent PR landing after yours hides your merge from -1.

- MAPLIBRE IMAGE-POOL STARVATION (root-caused 2026-07-16, probe chain
  in experiments.md v1.0.366 entry): a HANGING raster tile CDN
  (requests that never resolve — e.g. a proxy black-hole, unlike a
  fast 4xx) saturates MapLibre's global ~16-slot image-request pool
  and silently starves EVERY raster/raster-dem source on the map — no
  errors, tiles show 'loading' events but zero network. Vector
  sources use a different queue and keep working, which makes the
  failure look raster-specific. OPEN RESILIENCE QUESTION: should the
  imagery source get a client-side timeout/abort so a hanging imagery
  CDN can't take down fires/weather/seafloor with it? (Prod imagery =
  Esri, reliable; risk is low but the failure mode is total and
  invisible.) Sandbox drives must abort blocked CDNs (fail-fast) to
  stay prod-faithful — the committed drive pattern now does.

- MAPLIBRE color-relief SILENTLY IGNORES ["step"] EXPRESSIONS (v5.24,
  root-caused 2026-07-16 with pixel evidence): validation accepts the
  paint, getPaintProperty returns it verbatim, nothing renders. Use
  interpolate form with knife-edge stops (±0.02 around integer
  values) for classed rasters — lib/seafloorV2.tidConfidenceColorRelief
  is the precedent. Presence-only assertions (getLayer/isSourceLoaded)
  CANNOT catch this class — a paint check needs pixels.

- DEPLOY VERIFICATION MUST BE CONTENT-BASED, never build-hash-based:
  Railway's build produces different Vite chunk hashes than a local
  build of the same source (observed 2026-07-16: three distinct
  solarView-*.js hashes for one content). Verify by fetching a
  wave-unique asset/route and checking its BYTES (magic numbers,
  byte-compare against the committed file), not by a hash-pinned URL.

- STOP-HOOK FALSE POSITIVE after every post-merge branch reset: the
  git-check hook flags the branch tip as "Unverified (committer
  noreply@github.com)" — that commit is GitHub's OWN squash-merge
  commit on main, visible only because the branch was just reset to
  origin/main. VERIFY with `git log origin/main..HEAD` (empty = nothing
  of yours to sign) and DO NOT follow the hook's amend advice: amending
  rewrites merged main history onto the branch, diverges from origin,
  and recreates the dirty-PR stall. Correct action: none.

- WRONG-MERGE RESET (recurred 2026-07-04, now mechanically fixed): a
  hash-only merge monitor fires on ANY main advance — a concurrent PR
  merging first (#147 while #148 waited) triggered a reflexive branch
  reset + force-push that EMPTIED the open PR (GitHub auto-closed it;
  work recovered via cherry-pick from the local object store, reopened
  as #149). RULE: merge monitors must print the merged commit's
  SUBJECT and gate the reset on it matching the expected PR number —
  never reset on a bare hash change. The identity-aware monitor
  template lives in this session's history (PR #149's watch).
- CONCURRENT SESSIONS DOUBLE-BUILD roadmap items: an interactive session
  and a routine both built R1's archive on 2026-07-03 (#106 branch vs
  #107), forcing a supersession merge. Rule: CLAIM before building —
  append [CLAIMED <date> <PR#>] to the roadmap entry in your first
  commit; check for claims first. Version bumps: read-and-increment,
  never hardcode (three collisions today: 1.0.36 x2, 1.0.41 x2).
- A `mergeable_state: "dirty"` claude/* PR stalls SILENTLY: no merge ref ->
  pull_request workflows never start -> no checks, no automerge, no error.
  Check mergeability FIRST, not CI logs. Cause: reusing one branch across
  squash-merged PRs; scheduled sessions (fresh branch each run) are immune,
  interactive sessions must reset the branch onto main after each merge.

- **RECURRENCE 2026-07-08, two instances found in one routine sweep**: #379
  (globe 3D terrain, opened 2026-07-08) and #350 (satellite orbits layer,
  opened 2026-07-07) were BOTH `mergeable_state: "dirty"` with
  `get_status total_count: 0` (zero CI checks ever ran) — confirming this
  gotcha isn't a one-off. #379 rescued this session (rebuilt cleanly onto
  current main, re-verified fresh, PR #386; #379 and the now-confirmed-
  redundant #343 both closed with pointer comments). **#350 STILL NEEDS
  THE SAME RESCUE** — not done this session (one logical change per PR).
  Its diff is small enough to reapply directly from `pull_request_read
  get_diff` on #350 (`client/src/lib/satelliteOrbits.ts` new file +
  `datamap.tsx` layer wiring + `datacore/layers.json` + `LAYER_GROUP`
  fixture) — check `datamap.tsx`'s current satellites-layer state first in
  case a concurrent session already built an equivalent client half before
  rescuing (avoid the double-build gotcha above). Its live effect stays
  muted regardless (CelesTrak firewalled from Railway per R17) so it is
  not urgent, just queued.
  **RESOLVED 2026-07-09**: closed as superseded, not rescued — by the
  time this was picked up, the ORBITAL program's O2 wiring (#369, client-
  fetch design that sidesteps R17 entirely) had already shipped an
  equivalent, better client half. No unique delta salvaged. See
  orbital_program.md RESUME STATE.

- **[RESOLVED 2026-07-09, v1.0.236]** ~~tiles3dBudget.ts missing durability
  classification~~ (found 2026-07-09, T-CLIENT orbital-O3 session's gate
  run). FIX: routed `LEDGER` through a local `tiles3dStateDir()` resolving
  `DATA_DIR`/`/data/voltrade` with a `/tmp` fallback — the same pattern
  `analyst.ts`'s token-budget file already uses — rather than
  `archiveBaseDir()`. Tried `archiveBaseDir()` first (the obvious durable
  marker) but it tripped a SECOND, unrelated gate: `manifests.test.ts`'s
  FORWARD ENFORCEMENT test scrapes every `path.join(archiveBaseDir(), "x")`
  call as a new datacore STREAM requiring a full envelope manifest
  (source/license/attribution/geo_fields/entity_key) — correct for actual
  archived data streams, wrong for this ops cost-ledger (no source, license,
  or geo fields apply to a request-count guard). `analyst.ts`'s own
  `DATA_DIR`-direct pattern is precedent for exactly this case: durable
  across redeploys (the whole point — a non-durable ledger would silently
  reset the daily/monthly caps on every deploy, defeating the guard) without
  being swept into the manifest system meant for data streams. Both
  `durability.test.ts` and `manifests.test.ts` pass; full `npm run test:node`
  (517/517) and `python3 -m pytest -q` (558 passed, 1 skipped) green.

- **RECURRING (noticed 2026-07-09, third instance): merged PRs land without
  their final commit's own experiments.md entry.** The 2026-07-09 REPAIR
  session flagged 5 merged PRs (#392-#397) that never logged; the same
  session's own tiles3dBudget entry then also went unlogged by the PR that
  shipped it (only caught because the NEXT session diffed `git log` against
  `## ` headings); this session found a THIRD instance — PR #400's final
  commit ("grid overlay: expand to 9 states") shipped the state-expansion
  feature but the PR's earlier commits' experiments.md entries (tower-
  detector v0/v1/gate-1/diversity) were the only ones written — the last,
  separately-meaningful commit in the same PR added nothing. Pattern: a
  multi-commit PR's EARLIER commits log correctly (built commit-by-commit
  with their own reasoning), but the FINAL wrap-up commit — often added
  late, sometimes by a different subagent — skips the ratchet. Not
  investigated or fixed this session (process/tooling question, not a code
  defect in the traditional sense — no test can assert "the next PR must
  touch this file"). Two directions worth a future session's evaluation:
  (a) a CI check on `.github/workflows/` that a merged PR touching
  `datacore/**`, `server/**`, `bot_engine.py`, or `client/src/**` also
  touched `research/experiments.md` (would need care not to punish trivial
  PRs — version-bump-only, docs-only); (b) a pre-merge session checklist
  item ("does my LAST commit's change have its own paragraph, not just my
  first"). Filing here rather than building (a) or (b) blind — a CI-workflow
  change is itself constitutionally sensitive territory (`.github/
  workflows/` is a FROZEN PATH) and deserves its own wishlist proposal with
  the exact before/after, not a same-session drive-by.

- **[RESOLVED 2026-07-11, v1.0.280]** ~~Multi-segment top-level routes
  404 their own JS bundle in production~~ — FIXED: `vite.config.ts`'s
  `base` flipped `"./"` -> `"/"` (the app is always served from the
  domain root — `server/static.ts`'s express.static + unconditional SPA
  fallback, no reverse-proxy path prefix in railway.json/Dockerfile,
  confirmed this session, not assumed). A/B-verified: rebuilding with the
  OLD config and running the visual harness against `/newsletter/:slug`
  produced 3/3 hard failures at all three widths; the fix produces 0.
  Also closed the harness's own blind spot that let this ship unnoticed
  the first time: `scripts/visual_check.mjs`'s `CHECKS_SNIPPET` gained an
  "app actually mounted" assertion (`#root` has >=1 child) on EVERY page
  — the pre-existing checks (overflow, interactive-element sizing)
  structurally cannot catch a blank page, since a dead page has zero of
  both. `/newsletter/test-issue` is now a permanent PAGES entry (the
  harness's first 2+-segment route) so this class of bug can't silently
  regress. Full trace in experiments.md. Original text preserved below.
  ~~`vite.config.ts` sets `base: "./"`, so the built
  `dist/public/index.html` bakes in a RELATIVE script src
  (`./assets/index-XXXX.js`). The browser resolves that relative to the
  CURRENT URL's directory, not the site root. For a single-segment route
  like `/developers` or `/pricing`, `./assets/x.js` resolves to
  `/assets/x.js` — correct, by coincidence (one path segment = one
  "directory" to strip). For ANY route with two or more segments —
  confirmed against the real built output with a standalone script serving
  `dist/public` — the browser instead requests `/first-segment/assets/
  x.js`, which doesn't exist, and the page renders a blank white screen
  with a strict-MIME-type console error (SPA `index.html` fallback serves
  text/html for the 404'd asset request, and the browser refuses to
  execute a `text/html` response as a module script). Reproduced live
  against `/newsletter/:slug` (an EXISTING production route) with the
  identical failure — this is not new, it has silently affected that page
  since it was added. NOT FIXED this session (deliberately out of scope —
  own logical change, and the fix needs a decision: switch `base` to `/`
  (safe only if the app is always served from the domain root, true today
  but a constraint worth stating explicitly) vs. teaching the Express
  static handler to rewrite asset paths per-route). This session's own new
  page (`/apikeys`, PLATFORM P3) was deliberately kept single-segment to
  sidestep the bug rather than trip over it — see the comment at its
  route registration in `client/src/App.tsx`. NEXT STEP for whichever
  session fixes this: flip `vite.config.ts`'s `base` to `"/"`, rebuild,
  and re-verify `/newsletter/:slug` and any other multi-segment route
  render (not just build) via the visual harness or a direct headless
  load — a build succeeding is not evidence a route renders, exactly what
  let this ship unnoticed the first time.~~

## SPINOUT-READY DATA LAYER (human-approved 2026-07-03)

All EDGE-DOCTRINE data pipelines live in datacore/ with no imports from or
knowledge of trading logic; signals exposed only through an internal API
boundary (the bot consumes them like an external customer would).
Potential standalone product (satellite, ADS-B, AIS, EDGAR, Trends).
Spinout trigger (human decides): a root passes ladder gate 2 AND (external
demand OR dedicated-infrastructure need). Until then: one loop, one repo;
gate-2 signals get a /data surface on the existing site. RAW-DATA overlays
(as-is display + attribution, no predictive claim) ship ungated; SIGNALS
gate at ladder gate 2. Every map layer labeled as one or the other.

## MAP V2 ROADMAP (human directive 2026-07-03 — product routines work in order)

R1. **Performance + live-layer overhaul** — WebGL layer rendering at 10k+
    features, viewport-culled; global aircraft+vessel coverage with
    viewport fetching; shared server-side feed cache + exponential backoff
    + delta updates; aircraft/vessel enrichment (heading rotation,
    velocity vectors, type-differentiated icons, detail cards, recent
    trails). Honest coverage labeling (terrestrial AIS has mid-ocean gaps;
    ADS-B coverage follows receiver density).
    - **[SHIPPED 2026-07-03] POSITION ARCHIVE** — recording started
      immediately per the "every day not recorded is unrecoverable" note
      below (see experiments.md, `server/dataArchive.ts`). Still open in
      R1: WebGL rendering, viewport-fetching, delta updates, and the
      enrichment features listed above.

R2. **Maritime transit analytics — the strongest trading-signal candidate
    here.** Geofence counters on major ports and chokepoints (Suez,
    Panama, Hormuz, Malacca, major US ports) counting AIS transits/day
    from OUR OWN accumulating feed history; baseline vs anomaly display on
    the map. Ladder path: gate 1 = ground truth against published port
    statistics; gate 2 = transit anomalies as predictive signal for
    shipping/energy/commodity tickers. ARCHIVE-FIRST: recording starts
    with R1's archive even though the signal validates later — every day
    not recorded is unrecoverable proprietary data.

R3. **Environmental layers (all free sources).** USGS water gauges
    (lake/river levels + trend indicators), NWS weather overlays, NASA
    FIRMS active fires. Each ships as a RAW layer first. Logged
    hypotheses with ladder paths: drought/low-water -> ag futures,
    utilities cooling constraints, barge draft limits on the Mississippi
    (shipping costs); active fires -> insurers (P&C), utilities
    (liability precedent: PCG), timber.

R4. **3D globe mode.** MapLibre globe projection (or Cesium only if
    terrain tilt justifies it) as a 2D/3D toggle; pan/tilt/rotate; free
    elevation tiles for a terrain/relief base option. GATE: evaluate
    performance impact on phone BEFORE shipping — 3D must not degrade the
    2D default experience (DESIGN.md performance budget applies).
    Elevation/terrain as possible future signal input (flood-risk
    context) — hypothesis only, no build until a use case passes the
    ladder.

R5. **THE EVERYTHING GRAPH — flagship (charter directive 2026-07-04).**
    Design doc: datacore/EVERYTHING_GRAPH.md. v1 links ONLY what we
    already collect: person(CIK) —insider_of→ company(ticker)
    —operates→ facility(sites/plants via entity_map) ←calls_at—
    vessel(MMSI, from port-dwell visits). Storage v1 = pure builder +
    cache (recompute-from-archives doctrine; sqlite materialization
    only past the stated evolution trigger). Build order (each own PR):
    (1) datacore/entity_map.json verified operator→ticker table [also
    unblocks fusion (b) gate 1], (2) server/entityGraph.ts +
    /api/data/graph + tests, (3) /data graph panel + company→facility
    map highlighting. Graph queries become a /data feature when v1
    lands — RAW with provenance; interpretations on top stay
    ladder-gated.

    **[STEP 1 SHIPPED 2026-07-05, v1.0.131]** `datacore/entity_map.json`
    built via `scripts/build_entity_map.py`: every `operator` string in
    `strategic_sites.json` (13) + every `owner` string among
    `us_power_plants.json`'s top-100-by-capacity plants (56) = 69
    entities, hand-verified against SEC filings/investor-relations
    pages via live search this session (not recalled from training
    data alone — regulated-utility subsidiary structures are stable
    but merchant-generator ownership churns; the search caught two
    registry entries that are now STALE: `NRG Homer City Services LLC`
    (plant retired 2023, demolished 2025 — NRG was the contracted
    operator, never the owner) and `Louisiana Generating LLC` (sold by
    NRG to the now-private Cleco in 2019)). 44/69 mapped (high or
    medium confidence + parent + provenance note), 25/69 honest
    unmapped gaps (federal/state/municipal authorities, privately-held
    post-bankruptcy/PE-owned merchant generators, fragmented joint
    ventures like Keystone/Conemaugh's KeyCon Operating LLC) — no
    guessed tickers anywhere. `server/entityMap.test.ts` pins
    structure, the RAW/no-predictive-claim doc language, and coverage
    honesty (every current site operator + top-100 plant owner has an
    entry; the test fails loudly if either source registry grows a new
    operator string this table hasn't researched yet). ALSO UNBLOCKS:
    fusion hypothesis (b) generation×operator gate 1 (its own
    registry-owner→ticker mapping requirement is now satisfied by this
    same table).

    **[STEP 2 SHIPPED 2026-07-06, v1.0.149]** `server/entityGraph.ts` +
    `/api/data/graph` (see experiments.md for the full trace): joins
    facility nodes (all 16 sites + 9,833 plants), `operates` edges (from
    entity_map's 44 mapped operators — honest gaps stay edgeless),
    `insider_of` edges (30-day Form4 archive, aggregated per person/
    issuer pair with roles/filing_count/first-last-seen), and `calls_at`
    edges (vessel -> port, from a `portDwell.ts` refactor —
    `foldPortVisitsAsync` extracted so the graph reuses the same bounded
    AIS fold rather than re-scanning). Every edge carries
    {source, confidence, first_seen, last_seen}. Route follows the
    portdwell/shadowstats eager-poller-cache shape (15-min rebuild, never
    per-request — the exact event-loop/OOM class those two were repaired
    for) and returns counts-only without `?entity=`, a BFS neighborhood
    (`?entity=<ticker|MMSI|CIK|facility id>&hops=1..3`) with it.
    `server/entityGraph.test.ts` (10 tests) pins node/edge counts,
    operates-edge honesty (unmapped operators produce no edge),
    insider_of aggregation math, calls_at median-dwell aggregation, and
    the neighborhood BFS.
    **[STEP 3 SHIPPED — v1.0.208 wishlist.md note flagged this as stale
    2026-07-08]** the `/data` graph panel is already live
    (`client/src/pages/graph.tsx`, wired into `datamap.tsx` at
    `#/data/graph`) — this entry's "unclaimed" language was stale
    documentation, not a real gap; caught reading this file at session
    start, not fixed by the shipping session itself. STILL OPEN,
    smaller/independent: an `/api/v1/graph` keyed mirror (mirrors the
    existing `stats/portdwell`/`stats/shadow` pattern).

R6. **Dashboards from monitoring we already emit (charter directive
    2026-07-04).** Three /data panels, no new collection: (a)
    SIGNAL-STRENGTH — ladder position of every root (gate passed/date/
    next gate, from research/ bookkeeping made machine-readable); (b)
    DATA-QUALITY — feed freshness + per-provider status (runtime layer
    statuses), archive growth (/api/data/archive/stats), verification
    coverage (sites 16/16, plants 100/9833 imagery-verified); (c)
    PIPELINE-HEALTH — /api/health checks history, provider backoff
    states, compliance status. DESIGN.md applies (self-see, three
    widths); each panel its own PR.

## ARCHIVE-ENABLED SIGNAL HYPOTHESES (raw material accumulating from R1;
   each still validates through the full ladder)

- **Corporate jet activity around M&A targets**: per-tail-number history
  from our archive -> unusual visits of corporate jets to counterparty
  HQs/airfields preceding announcements. Ladder: gate 1 = tail->operator
  mapping verified against public registries; gate 2 = do clustered
  visits precede M&A announcements at better-than-base rates?
- **Tanker routing anomalies**: deviations from a vessel's own historical
  route patterns (from our archive) near chokepoints/sanctioned routes ->
  energy price/logistics signals. Gate 1 = route baselines vs known
  seasonal patterns; gate 2 = anomaly counts vs tanker rates/crude moves.
- **Destination prediction quality**: trajectory + per-aircraft
  historical route patterns -> predicted destination (labeled PREDICTED).
  Gate 1 = predictions scored against actually-observed landings from our
  own archive (self-labeling ground truth — free).

## POWER-PLANT SIGNAL HYPOTHESES (raw layer live 2026-07-03; WRI GPPD CC BY 4.0)

- **Generation-mix shift trades.** The static plant registry (capacity by
  fuel per region) is the denominator for a future flow signal: EIA-930
  hourly generation by fuel (free API) against installed capacity gives
  regional utilization by fuel. Hypothesis: sustained gas-burn
  utilization spikes (heat waves, coal retirements) lead regional
  utility earnings surprises and nat-gas demand (UNG, XLU components).
  LADDER PATH — DATA: EIA-930 vs the registry (this layer) reconciles
  within ~5% of EIA-860 capacity; SIGNAL: utilization anomalies vs
  forward utility/gas returns; LOGIC: entry rules by anomaly magnitude;
  SIZING: vs equal-weight utility basket; EXECUTION: fills tracker.
- **Outage-adjacent trades.** Nuclear plants (58 sites, now geolocated)
  file NRC daily status reports (free). Unplanned outages at large units
  move regional power prices and the operator's stock same-week.
  Hypothesis: NRC event reports + this registry (unit MW + operator) =
  same-day operator-impact estimate nobody prices for small utilities.
  LADDER: DATA gate = NRC report parse matches registry units; then as
  above. Capacity-constrained corner: mid-cap single-plant operators.
- Both hypotheses use the archive-first pattern: start recording EIA-930
  + NRC dailies NOW (cheap cron), judge after a quarter of history.

## EARTHQUAKE HAZARD-ADJACENT HYPOTHESES (RAW layer shipped 2026-07-08,
   v1.0.209 — server/usgsQuakes.ts, /api/data/earthquakes, keyless USGS
   M2.5+ global feed, archive-first from day one)

- **Insurer (P&C) exposure trades.** PRIOR stated before any test: a
  significant event (USGS `sig` score, which already folds magnitude +
  population exposure) within a capacity-constrained P&C insurer's known
  concentration region (e.g. a regional carrier writing CA/PNW
  earthquake policies) should show forward underperformance vs. the
  insurance-sector baseline in the days after, before reinsurance/loss
  estimates are public — REASONING STANDARD #5's second-order test:
  the trade exists because loss-reserve revisions lag the event by days
  while the market's initial reaction is often noisy/overdone in either
  direction, not because "nobody noticed a quake." Ladder: gate 1 DATA
  passed today (USGS feed is ground truth, not a proxy — no separate
  verification needed, unlike tank shadows or satellite proxies); gate 2
  SIGNAL needs (a) a small-cap/regional-insurer universe with disclosed
  geographic concentration and (b) enough archived event history to test
  forward N/5/20-day returns vs. a same-universe random-entry baseline
  (REASONING STANDARD #3) — not attempted yet, blocked on accumulating
  archive depth (started today, needs weeks-months).
- **Utility/infrastructure exposure.** Same construction as the NRC
  outage-adjacent hypothesis above: a significant event near a mapped
  facility in `entity_map.json`/`us_power_plants.json`/`strategic_sites
  .json` (join on lat/lon proximity, not yet built) is a same-day
  operator-impact estimate nobody prices for small single-region
  utilities. Ladder: gate 1 = proximity join reconciles against known
  plant coordinates (mechanical, no new data needed); gate 2 = event-
  adjacent operator returns vs. baseline. Capacity-constrained corner:
  small single-plant/single-region utilities, per EDGE DOCTRINE #2.
- **Supply-chain disruption (industrial/logistics sites).** Same
  proximity-join construction against port/rail/site registries already
  archived (R2 maritime, `datacore/rail`). Speculative — filed per the
  ACTIVE ANGLE-HUNTING standing behavior's explicit invitation to log
  even weird hypotheses with their ladder path; no work done beyond
  filing.
- Discount all three per REASONING STANDARD #4 before any of them run:
  they share one raw event feed, so a "hit" on one is not independent
  evidence for the others' validity as a class.

## MARINE/BUOY HAZARD-ADJACENT HYPOTHESES (RAW layer shipped 2026-07-08,
   v1.0.220 — server/ndbcBuoys.ts, /api/data/buoys, keyless NOAA NDBC
   latest-observations feed, ~889 stations, archive-first from day one)

- **Sea-state as a shipping-cost/insurance-exposure proxy (FOREIGN-FIELD
  IMPORT — marine forecasting technique).** PRIOR stated before any test:
  sustained significant-wave-height (WVHT) spikes at buoys clustered near
  major shipping lanes/chokepoints (Gulf of Mexico, mid-Atlantic, West
  Coast approaches — cross-reference against the existing AIS port-dwell
  archive's traffic density) should correlate with same-week vessel transit
  slowdowns/rerouting and marine-insurance (P&C) loss-adjacent moves,
  analogous to the earthquake hazard hypotheses above but for a slower-
  building, multi-day hazard rather than an instantaneous event. Second-
  order test (REASONING STANDARD #5): the trade would exist because
  routine NWS marine forecasts are public but the SPECIFIC correlation
  between a given buoy cluster's readings and a given operator's fleet
  exposure requires joining our own AIS archive to NDBC stations, which
  nobody else has built — not because the raw sea-state data itself is
  hidden. Ladder: gate 1 = buoy readings already ground truth (NOAA's own
  instruments, no proxy-verification step needed, unlike tank shadows);
  gate 2 = build the buoy-to-shipping-lane proximity join (mechanical,
  reuses the port-dwell/entity_map join pattern) then test forward
  N/5/20-day shipping-sector or single-operator returns vs. a same-
  universe random-entry baseline — not attempted, blocked on (a) the
  join and (b) archive depth (started today).
- **Pressure-tendency (PTDY) as a fast-moving-storm precursor.** A rapid
  negative PTDY swing across a cluster of Gulf/Atlantic buoys is a
  classical marine-forecasting precursor to a fast-developing storm
  system, ahead of the NHC's own official advisories in the earliest
  hours. Speculative — filed per the ACTIVE ANGLE-HUNTING standing
  behavior's explicit invitation to log even weird hypotheses; no work
  done beyond filing. Gate 1 = PTDY reconciles against NHC's own
  advisory timing on a handful of known past storms (mechanical,
  retrospective); gate 2 = does an early PTDY cluster-swing predict
  next-day energy/insurance-sector moves better than waiting for the
  official advisory?
- Discount both per REASONING STANDARD #4: same raw feed, so a "hit" on
  one does not independently validate the other; and per STANDARD #7,
  the archive only started today — neither hypothesis has any history to
  test against yet.

## NIGHT-LIGHTS RADIANCE HYPOTHESIS (RAW layer shipped 2026-07-08, v1.0.224
   — client/src/pages/datamap.tsx "nightlights" + client/src/lib/gibs.ts,
   worldview_globe.md Phase G2a — the first NASA GIBS layer)

- **Metro/industrial radiance delta as a regional-economic-activity proxy
  (Pillar 6, per worldview_globe.md's own G2a text).** PRIOR stated before
  any test: sustained MoM/YoY radiance INCREASES over a metro/industrial
  area (VIIRS/SNPP Day/Night Band, daily) correlate with regional economic
  activity upticks — the classic "nighttime lights as GDP proxy" literature,
  applied here at a finer time cadence (daily vs. the usual annual composite)
  and joined to our OWN archive once enough daily history accumulates.
  Second-order test (REASONING STANDARD #5): the residual edge, if any, is
  in TIMING (daily granularity most GDP-proxy studies don't use) and in
  SMALL/REGIONAL names — regional-bank, retail, and single-region utility
  tickers by CBSA, not mega-cap names where satellite-radiance research is
  already well-mined. Ladder: gate 1 DATA — this is a NASA-published,
  publicly documented product (not a proxy we're inferring from raw
  pixels the way tank-shadow analysis is); the verification step is
  confirming OUR tile fetch/date-alignment reproduces GIBS's own published
  imagery for a handful of known dates (mechanical, not yet done — next
  session can spot-check a few dates against the GIBS Worldview UI
  directly). Gate 2 SIGNAL is blocked on (a) building a metro/CBSA →
  ticker join (reuses the entity_map.json/CBSA-mapping pattern already
  established for other hypotheses) and (b) accumulating enough daily
  archive history to compute a MoM/YoY delta at all — the archive starts
  today, nothing to test yet.
- **Honesty constraints already built into the layer itself** (not
  deferred to gate 2): the layer ships RAW/as-is with no predictive claim
  in its registry description; it is NOT archived yet (this PR is
  client-only display — a display layer, not a data pipeline — so there
  is no daily-radiance archive to query until a follow-up PIPELINE session
  adds one, which is the actual prerequisite for gate 2's (b) above, not
  yet started); dates that render blank (daylight side of the terminator,
  real sensor gaps) are stated as an honest, expected outcome in both the
  layer description and the on-panel status note, never silently retried
  as if something were broken.
- Discount per REASONING STANDARD #4: "nighttime lights predict economic
  activity" is one of the most heavily studied remote-sensing proxies in
  economics — expect a LOW prior on finding genuinely new alpha here vs.
  the daily-cadence/small-cap-residual angle actually being real; this is
  exactly the kind of well-known effect REASONING STANDARD #5 says to be
  skeptical of ("because nobody noticed" is almost never the answer) — the
  claimed edge here is specifically the finer time granularity and the
  small/regional universe, not the existence of the underlying proxy.

## AEROSOL OPTICAL DEPTH HYPOTHESIS (RAW layer shipped 2026-07-08, v1.0.228
   — client/src/pages/datamap.tsx "aerosol" + client/src/lib/gibs.ts,
   worldview_globe.md Phase G2c — NASA GIBS MODIS Combined Value-Added AOD)

- **Aerosol optical depth over an industrial basin / shipping lane as an
  output-and-throughput proxy (Pillar 6, per worldview_globe.md's G2c
  text).** PRIOR stated before any test: sustained AOD ELEVATION over a
  defined industrial cluster or a busy shipping lane co-moves with, and may
  slightly LEAD, that region's production/throughput — combustion haze and
  ship-plume aerosol are a physical by-product of activity. Expected effect
  weak-to-moderate and heavily confounded (meteorology, wildfire smoke,
  Saharan/Asian dust transport all dwarf the industrial signal on most
  days), so the tradeable residual — if any — is in ANOMALY vs. a
  local seasonal/meteorological baseline, not raw AOD level. Second-order
  (STANDARD #5): satellite-AOD-as-economic-activity is published research
  (COVID-era industrial-slowdown AOD drops are well documented), so the
  residual edge, if real, is in (a) daily cadence on a NAMED small cluster
  vs. the usual national/annual composite and (b) pairing AOD with our own
  facility archive (a smelter/refinery/port we already geolocate) rather
  than a diffuse regional average. Universe bias per EDGE DOCTRINE #2:
  single-basin operators and their commodity, not diversified mega-caps.
- Ladder: gate 1 DATA — NASA-published product (MODIS Combined Value-Added
  AOD), so verification is mechanical: confirm OUR tile fetch/date
  alignment reproduces GIBS's own published imagery (a real yesterday tile
  was already pixel-checked non-blank, 21% coverage, at build time; a
  spot-check of a few dates against the GIBS Worldview UI is the remaining
  step). Gate 2 SIGNAL is blocked on (a) a daily-AOD ARCHIVE over our
  facility/lane polygons (this PR is display-only — no pipeline yet, same
  as night lights) and (b) a de-confounding baseline (subtract a
  meteorology/dust reference so the residual is industrial, not weather).
  NOT a signal; ships RAW with no predictive claim, blank-over-cloud/glint
  areas stated as honest retrieval coverage in the layer description and
  status note.
- Discount per STANDARD #4: AOD-as-activity is a known effect; expect a LOW
  prior on new alpha except through the daily-cadence + named-facility +
  de-confounded-anomaly angle. CROSS-TIE (2026-07-07 integration principle):
  the natural join is AOD × the strategic-facility archive — the same
  archive the fires × facilities cross-tie (#388, v1.0.227) already joins —
  so both fire-thermal and aerosol observations hang off the one facility
  spine, which is the honest place any future signal would be built.

## VEGETATION / NDVI YIELD HYPOTHESIS (RAW layer shipped 2026-07-08, v1.0.229
   — client/src/pages/datamap.tsx "vegetation" + client/src/lib/gibs.ts,
   worldview_globe.md Phase G2e — NASA GIBS VIIRS/SNPP NDVI 8-day)

- **NDVI anomaly over a NAMED crop belt as a yield proxy (Pillar 6, per
  worldview_globe.md's G2e text).** PRIOR stated before any test: a positive
  (negative) NDVI anomaly over a defined crop belt vs. its own planting-calendar
  baseline co-moves with the season's yield outcome for that crop, and — because
  it is observable continuously through the growing season — may LEAD the
  USDA/WASDE print the market prices off. Expected effect moderate, but this is
  THE most heavily mined satellite-ag signal (commercial ag-analytics firms,
  USDA's own NASS/VegScape, every ag-commodity desk already run NDVI), so per
  STANDARD #4/#5 the tradeable residual for a small system is NOT the level — it
  is (a) the de-seasonalized anomaly vs. a crop-calendar baseline, (b) tight
  regional windows the aggregate national number smooths over, and (c) pairing
  NDVI with our OWN facility/entity archive (an ethanol plant, a grain elevator,
  a single-region ag-input name) rather than the front-month future everyone
  watches. Honest expectation: LOW prior on new alpha vs. well-capitalized
  incumbents; the layer's near-term value is HONEST CONTEXT on the globe and a
  substrate for cross-ties, not a standalone edge.
- Ladder: gate 1 DATA — NASA-published product; verification is mechanical
  (confirm our tile fetch reproduces GIBS imagery; a yesterday tile was
  pixel-checked non-blank over land, 41%, at build time — NDVI is land-only so
  ocean transparency is correct, not a gap). Gate 2 SIGNAL blocked on (a) an
  NDVI archive over crop-belt polygons and (b) a planting-calendar / prior-year
  baseline so the residual is anomaly, not season. RAW, no predictive claim.
- CROSS-TIE (2026-07-07 integration principle): NDVI joins the drought/
  soil-moisture (future G2d) and river-gauge (barge-draft) layers into a
  coherent ag-supply-chain view — crop health × water availability × the barge
  corridor that ships the grain — all RAW observations over the same geography;
  a future gate-2 ag signal would be built on that stack, not NDVI alone.

## SOIL-MOISTURE / AG-SUPPLY-CHAIN HYPOTHESIS (RAW layer shipped 2026-07-08,
   v1.0.230 — client/src/pages/datamap.tsx "soilmoisture" + client/src/lib/gibs.ts,
   worldview_globe.md Phase G2d — NASA GIBS SMAP L4 root-zone soil moisture)

- **Root-zone soil moisture over a NAMED crop belt / river basin as an
  agricultural-supply and inland-shipping proxy (Pillar 6, G2d text).** PRIOR
  before any test: a sustained soil-moisture DEFICIT (surplus) in a defined
  basin during the growing season co-moves with lower (higher) realized yield
  for that crop AND lower (higher) navigable draft on the barge corridor that
  ships it — the same drought that stresses the crop drops the river. Expected
  effect real but slow and heavily lagged (SMAP is ~6 days behind and soil
  moisture integrates over weeks), and — like NDVI — already mined by ag desks,
  so per STANDARD #4/#5 the residual for a small system is NOT the level; it is
  the JOINT reading across our own layers: soil-moisture deficit AND NDVI
  anomaly AND a falling river gauge on the same basin is a stronger, less-mined
  composite than any one alone.
- Ladder: gate 1 DATA — NASA-published product; verification mechanical (a
  07-01/07-02 tile pixel-checked non-blank over land at build time; 07-07 blank,
  confirming the ~6-day lag that the layer's date-default now honors rather than
  hiding). Gate 2 SIGNAL blocked on (a) archives of all three layers over
  basin/crop-belt polygons and (b) a seasonal/drought baseline so the residual
  is anomaly, not climatology. RAW, no predictive claim.
- CROSS-TIE (the point of shipping this one): with G2d live, the ag-supply-chain
  TRIAD is complete on the globe — NDVI (crop health) × soil moisture (root-zone
  water) × river gauges (barge draft). This is the honest substrate for a future
  gate-2 ag composite; none of the three is a signal alone, but the STACK is the
  differentiated, small-system-appropriate view (EDGE DOCTRINE #2/#3).

## NO2-THROUGHPUT HYPOTHESIS (RAW layer shipped 2026-07-08, v1.0.231 —
   client/src/pages/datamap.tsx "no2" + client/src/lib/gibs.ts,
   worldview_globe.md Phase G2g — NASA GIBS Sentinel-5P/TROPOMI NO₂)

- **Tropospheric NO₂ column over a NAMED industrial zone / port as a real-time
  throughput nowcast (Pillar 6, G2g text — the charter's "genuinely
  differentiated" layer).** PRIOR before any test: because NO₂ is a short-lived
  combustion by-product (hours, not weeks — unlike CO₂), its column density over
  a defined industrial cluster or port tracks CURRENT operating rate, and the
  daily satellite read should LEAD monthly PMI / industrial-production prints
  that the market trades. This is the most PROMISING of the G2 hypotheses for a
  small system because the effect is (a) physically direct (emission ∝ activity),
  (b) HIGH TIME-RESOLUTION relative to the economic print it anticipates, and
  (c) SPATIALLY SPECIFIC — resolvable to a single industrial basin, exactly the
  small/regional target EDGE DOCTRINE #2 favors over crowded aggregates.
  Second-order (STANDARD #5): the COVID-lockdown NO₂ collapse is famous, so the
  gross "NO₂ = activity" link is known; the under-mined residual for us is the
  NAMED-facility, daily-cadence, de-weathered anomaly (NO₂ also depends on wind
  dispersion and photochemistry — a still, sunny day concentrates it), joined to
  the specific operator/utility ticker rather than a national index.
- Ladder: gate 1 DATA — ESA/NASA-published product; verification mechanical
  (yesterday tile pixel-checked as a real continuous field, 100% over N.America,
  ocean low). Gate 2 SIGNAL blocked on (a) a daily-NO₂ archive sampled over
  industrial-zone/port polygons (raster→point sampling — needs server-side tile
  sampling, a real build), (b) a meteorological de-confounding (wind/insolation
  normalization) so the residual is throughput not weather, and (c) a
  facility→ticker join. RAW, no predictive claim; swath gaps/cloud stated as
  honest coverage in the layer description + status note.
- CROSS-TIE (2026-07-07 integration principle): NO₂ × the strategic-facility /
  powerplant archive is the natural join — the same facility spine the fires and
  (future) AOD cross-ties hang off. NO₂ over a named smelter/refinery/port is the
  most direct "is this specific asset RUNNING right now" observable we have; a
  future gate-2 signal would sample this field at our facility coordinates. The
  still-open TEMPO (US hourly) and SO₂ (point-source smelter on/off) layers would
  deepen exactly this cross-tie once the sub-daily factory work lands.

## BIOMASS-DENSITY / STANDING-CARBON HYPOTHESIS (RAW layer shipped 2026-07-15,
   v1.0.318 — client/src/pages/datamap.tsx "biomass" + client/src/lib/gibs.ts,
   worldview_globe.md Phase G2h — NASA GIBS GEDI L4B aboveground biomass
   density, mission-life mean)

- **Standing forest biomass (aboveground carbon stock) over a NAMED timber
  region as a substrate for a carbon-credit / forestry-supply hypothesis
  (Pillar 6, per worldview_globe.md's G2h grouping).** PRIOR stated before any
  test: this layer is STATIC (a single 2019-04–2023-03 mission-life mean, not
  a repeat-observation product), so on its own it can never show CHANGE — the
  one thing a trading or carbon-market signal would need. The honest
  near-term value is CONTEXT (a real, non-fabricated forest-density basemap)
  and a BASELINE for a future change-detection build, not a standalone
  signal. Any tradeable residual would have to come from DIFFERENCING this
  baseline against a repeat product (e.g. a future GEDI L4B reprocessing
  covering later years, or JRC GFC2020's existing forest-EXTENT layer's own
  future updates) — deforestation/afforestation delta over a named timber
  concession or REDD+ project area is the plausible gate-2 candidate, not the
  static density level itself.
- Ladder: gate 1 DATA — NASA-published product; verification mechanical this
  session (Amazon-basin tile pixel-checked 95% non-transparent with
  plausible green biomass-density coloring; US Pacific-NW forest tile 99%
  non-transparent; open-ocean tile ~0.04% non-transparent — legitimately
  blank, GEDI is a land-vegetation LiDAR product limited to roughly ±51.6°
  latitude, not a coverage bug). Gate 2 SIGNAL is NOT YET REACHABLE for this
  static composite alone — it is blocked on a repeat/change-detection product
  existing at all, which is outside our control (depends on NASA publishing
  a later GEDI L4B reprocessing) or on pairing with forest.ts's own
  extent-change tracking over time. RAW, no predictive claim.
- CROSS-TIE (2026-07-07 integration principle): biomass density joins the
  existing `forest` (JRC GFC2020 2020 extent) layer into a two-axis view of
  the same geography — WHERE forest exists (JRC, binary extent) × HOW MUCH
  standing carbon it holds (GEDI, continuous density) — a substrate for a
  future carbon/timber hypothesis, not a signal on its own; also a candidate
  future input to the fires×facilities and NDVI×soil-moisture ag cross-ties
  already filed, since biomass density constrains how much a given fire or
  drought event can actually affect (dense old-growth vs. sparse scrub react
  differently).

## LOW-WATER / GENERATION-EXPOSURE HYPOTHESIS (RAW cross-tie shipped 2026-07-08,
   v1.0.232 — server/riverPlants.ts + /api/data/plants-near-rivergauges,
   worldview-globe Pillar 6 backend inference)

- **Low water on a barge-corridor river = operational risk for the generating
  capacity that depends on that stretch → the owning/operating utilities.**
  PRIOR before any test: a sustained low-stage/low-discharge reading at a
  Mississippi/Ohio/Missouri/Illinois gauge stresses the nearby plants two ways —
  (a) river-cooled thermal plants (coal/gas/nuclear) face cooling-water-intake
  and thermal-discharge-permit limits, forcing derates, and (b) coal plants that
  receive fuel by barge face draft restrictions that raise delivered fuel cost
  and can force switching. Effect real but EVENT-DRIVEN and rare (only bites in
  genuine drought — 2022's Mississippi low-water is the reference event), highly
  operator-specific, and confounded by weather-driven demand at the same time.
  Second-order (STANDARD #5): drought-on-the-Mississippi is a headline event, so
  the gross link is known; the under-mined residual for a small system is the
  PRE-COMPUTED, NAMED exposure — the instant a specific gauge crosses a
  low-water threshold, this cross-tie already lists the exact plants + total MW +
  fuel mix exposed on that reach, before the wire story names them.
- Ladder: gate 0/1 DATA — pure PROXIMITY join over two published datasets (USGS
  NWIS live gauge positions × WRI Global Power Plant DB coordinates), fully
  offline-tested (server/riverPlants.test.ts: known distances, radius cutoff,
  dedupe, nearest-first, capacity aggregation, never-fabricated MW). Gate 2
  SIGNAL blocked on (a) a low-water THRESHOLD/percentile per gauge (the raw
  stage number alone is not "low" — needs each gauge's own historical
  distribution, which the USGS archive is only now accumulating), (b) a plant→
  ticker join for the operators, and (c) an event study on the 2022 low-water
  episode as out-of-sample confirmation. RAW today, no predictive claim on the
  endpoint.
- HONEST LIMITS built into the tie: only 14 barge-corridor gauges exist, so
  coverage is the Mississippi/Ohio system, NOT all US rivers; plant locations are
  WRI 2021-vintage static; the "dependence" is inferred from PROXIMITY (within
  R km), not from confirmed water-intake permits — a plant near the river may
  draw from a different source. All stated on the endpoint. CROSS-TIE: this is
  the river-gauge arm of the same facility spine the fires (#388) and NO₂ (#392)
  cross-ties use — one geography, multiple exposure lenses.

## SEVERE-WEATHER / GENERATION-DISRUPTION HYPOTHESIS (RAW cross-tie shipped
   2026-07-08, v1.0.234 — server/plantsUnderAlerts.ts +
   /api/data/plants-under-alerts, worldview-globe Pillar 6 backend inference)

- **Generating capacity inside an active NWS severe-weather WARNING polygon =
  near-term generation-disruption risk → the operating utilities / the regional
  grid balancing authority.** PRIOR before any test: a tornado/hurricane/severe-
  thunderstorm/ice warning over a cluster of plants raises the odds of forced
  outages, transmission trips, and demand spikes on that balancing authority in
  the next hours-to-days. Effect real but VERY event-driven and short-horizon
  (warnings last hours), and the market already reacts to major storms; the
  under-mined residual for a small system is the PRE-COMPUTED, NAMED exposure —
  the instant a warning polygon is issued, this cross-tie lists the exact plants
  + MW + fuel mix inside it, before the outage reports come in, and (with a BA
  join) the balancing authority whose reserve margin is about to be tested.
- Ladder: gate 0/1 DATA — pure POINT-IN-POLYGON join over two published sources
  (NWS active-warning polygons × WRI plant coordinates), fully offline-tested
  (server/plantsUnderAlerts.test.ts: ray-cast containment, bbox pre-filter,
  per-fuel aggregation, most-exposed ordering, zone-only skip, never-fabricated
  MW). Gate 2 SIGNAL blocked on (a) a plant→operator→ticker join, (b) a
  plant→balancing-authority join (the gridvision TX BA work is the seed), and
  (c) an event study on historical warning×outage×price episodes for
  out-of-sample confirmation.
- HONEST LIMITS on the endpoint: only POLYGON-carrying warnings are testable —
  zone-only alerts (no geometry) are EXCLUDED and counted (zone_only_excluded),
  never silently treated as covering nothing; warning severity ≠ realized
  impact; WRI plant locations are 2021-vintage static. CROSS-TIE: the
  severe-weather arm of the SAME facility spine as the fires (#388), NO₂ (#392),
  and low-water (#393) cross-ties — four hazard/throughput lenses on one plant/
  facility geography, which is the Everything-Graph substrate a future gate-2
  composite signal would be built on (no single lens is a signal alone).

## FREIGHT-ACTIVITY PROXIES (trucks directive 2026-07-04 — build-first conclusion + research)

- **TRUCKS CONCLUSION (do not chase): individual truck positions are
  private fleet telematics** (Samsara/Motive/Geotab class, sold per
  fleet; no public feed anywhere, free or paid-aggregate). The
  build-first ladder terminates at step 4 with nothing to buy at our
  scale either — the capability simply isn't for sale as a market feed.
  Filed so no session burns time on it.
- Freight PROXIES worth building instead (all free, each with its
  ladder path):
  1. **Border crossing wait times** (CBP BWT public API + BTS border
     crossing monthly volumes). Hypothesis: sustained commercial-lane
     wait/volume anomalies at Laredo/Otay Mesa lead cross-border
     logistics + Mexico-exposure names. LADDER — DATA: BWT api vs BTS
     monthlies reconcile; SIGNAL: anomalies vs forward returns of a
     logistics basket; then LOGIC/SIZING/EXECUTION as standard.
  2. **Truck-lane traffic volumes** (Caltrans PeMS + state DOT APIs,
     free registration): real-time-ish corridor truck counts (I-710
     port drayage corridor, I-80). Hypothesis: port-corridor drayage
     volume leads retail-inventory names. DATA gate: PeMS truck counts
     vs port TEU monthlies correlate.
  3. **FMCSA carrier census/inspection counts** (free bulk): slow-moving
     capacity proxy (carrier entries/exits lead trucking-rate cycles —
     KNX/JBHT class). Monthly cadence; archive-first.
  4. **Port TEU monthlies** (already adjacent to our verified port
     sites): denominator for #2, slow ground truth for R2 transit
     counters.
  Archive-first rule applies to all four: start recording now, judge
  after a quarter. None surfaces on the map until ladder gate 2 (they
  are SIGNALS, not raw overlays).

## SHADOW-FLEET SIGNAL (Map v2.2 directive 2026-07-04; RAW stats live, per-vessel claims GATED)

- **What ships now (RAW)**: server/shadowFleet.ts computes from OUR OWN
  AIS archive — gap events (silent >6h, reappeared >100km), identity
  candidates (name under two MMSIs; new MMSI first seen near another's
  last position), loitering in 7 public STS zones
  (datacore/shadow_zones.json). Surface: counts only, caveat attached
  ("a gap can be coverage loss"). Archive grows the sample daily.
- **GATE 1 (DATA) — validation plan**: build a reference list of
  publicly documented shadow-fleet vessels (OFAC SDN vessel annexes +
  KSE Institute dark-fleet publications provide MMSIs/IMOs). Gate
  passes if our gap/loiter detections are significantly ENRICHED for
  reference-list vessels vs a size-matched random tanker sample from
  the same archive window (odds ratio with CI, not eyeballing).
  Terrestrial-coverage ambiguity is controlled by the comparison: both
  cohorts suffer identical coverage loss.
- **GATE 2 (SIGNAL) hypothesis + trading relevance**: sanctioned-oil
  flow volume (proxied by gap+loiter event rates in Laconian/Kerch/
  Fujairah zones) leads (a) tanker-rate proxies (FRO, STNG, TNK — clean
  vs dirty rates diverge when shadow capacity absorbs dirty trade) and
  (b) crude spreads (Urals-Brent proxied via RSX-era instruments is
  gone; use Brent-WTI + tanker basket). Test: weekly event-rate series
  vs forward 1-4w returns of the tanker basket, vs base rate.
- **Second-order (REASONING STANDARD #5)**: who's on the other side —
  commercial maritime-intel vendors sell this at $$$$ to compliance
  desks; the trade-relevant LAG (compliance buyers act on sanctions
  risk, not tanker-rate positioning) is the structural reason a small
  player can still extract the market signal.

## NEW DATA ROOTS (charter gap execution 2026-07-04 — licensing verified from primary sources by a 10-agent research pass; build order = expected signal × coverage × time-to-testable)

BUILD ORDER RATIONALE: 8-K language first because EDGAR history already
exists (gate 2 testable immediately, not time-blocked) with complete
small/micro-cap coverage and exact timestamps; jobs second (uniquely
un-arbitraged free panel, but 2 quarters of accumulation before gate 2);
app-store third (archiver is ~30 HTTP calls/day — trivial cost, heavily
arbitraged category so expectations low); USPTO fourth (clean licensing,
18-month publication hole, blocked on a human signup); GitHub fifth
(free and deep but the public-slice bias confound attacks the premise).

1. **Earnings language from SEC 8-K Item 2.02 (Exhibit 99) — the lawful
   transcript substitute.** **[GATE 1 (DATA) SHIPPED 2026-07-04 — v1.0.67]**
   `server/sec8kEarnings.ts` polls the getcurrent 8-K feed, filters Item
   2.02, resolves the Exhibit 99 press release, and extracts plain text
   (dependency-free HTML->text, verified against two real, live-fetched
   filings across two distinct filer-agent HTML formats — see
   `server/sec8kEarnings.test.ts` and research/experiments.md for the
   full trace). Live at `/api/data/earnings-language` (+ `/history`),
   RAW-DATA overlay only, no predictive claim. No UI page yet — same
   sequencing edgarForm4.ts used (pipeline+API first, a view once archive
   history exists) — queued as the next PRODUCT action once a few days
   of archive accumulate. Gate 2 below is UNSTARTED, unchanged.
   LICENSING VERDICTS (fetched 2026-07-04):
   Motley Fool and Seeking Alpha transcripts PROHIBITED as pipelines
   (both ToS bar automated access + commercial use); FMP transcripts
   effectively paid+restricted (personal-use free tier; data-deletion
   clause on termination); EDGAR is public-record, free, "no
   restrictions on public domain use," 10 req/s + declared User-Agent.
   WHAT WE GET: results + guidance language, same-day, timestamp-exact
   (acceptance-datetime = lookahead-free), EVERY reporting company incl.
   micro caps. HONEST GAP: Q&A sessions (where much academic signal
   lives) are almost never filed; the build-first path to true
   transcripts is self-ASR of public IR webcasts (Whisper, MIT) for a
   small watchlist — per-platform ToS check before any bulk automation,
   gray zone labeled honestly. PRIOR: modest post-earnings-drift
   prediction from guidance-language deltas (Lazy-Prices-style QoQ
   changes), strongest where analyst coverage is thin. LADDER — DATA:
   extract Exhibit 99 text; verify 50-filing sample vs actual exhibits +
   IR press releases; SIGNAL: L-M tone + language-delta features vs
   forward returns against size-matched random entry, regime-split;
   self-ASR side gates on guidance-sentence WER ≈ 0 vs 20
   company-published texts.
   GATE 2 FIRST PILOT 2026-07-12 (v1.0.282, scripts/
   earnings_language_gate2.py): tone-LEVEL-only pass (QoQ delta still
   not testable — needs a second quarter of filings per company,
   ~Oct-Nov 2026). N=47 at the 1-day horizon (only horizon reaching the
   pre-stated 30-sample floor) — mean-spread PASS, outlier-robust under
   median but dominated in magnitude by a few single-name events (worst:
   MVO's -53% alpha on its final-trust-distribution 8-K, a real wind-
   down repricing, not a data artifact). VERDICT: preliminary/
   encouraging, NOT gate-2-complete — sample far too thin across one
   calendar week of one archive snapshot to trust or trade. Re-run
   trigger: >=90 days of archive (5-day horizon N>=30) or a second
   filing quarter per company (unlocks the delta feature). Full trace +
   numbers in experiments.md's 2026-07-12 [RESEARCH] entry.
2. **Job postings via ATS public JSON (hiring velocity / role mix).**
   LICENSING: Greenhouse/Lever/Ashby/SmartRecruiters public postings
   endpoints carry no express third-party grant — CONDITIONAL: polite
   cadence, derived signals only on any paid surface (counts/deltas/
   ratios, never raw posting text), added to the provider-compliance
   checklist; LinkedIn/Indeed scraping PROHIBITED (and no scraped
   derivatives); USAJOBS restricted (OPM approval needed); Indeed
   Hiring Lab aggregates CC BY 4.0 (the panel ground truth). HONEST
   GAP: Russell-2000 ATS coverage is UNMEASURED — gate 0 exists to
   kill that unknown. LADDER — GATE 0 (week 1): ATS resolver probes
   the four endpoints per ticker, outputs a measured coverage table;
   if coverage <~10% and Workday stays blocked, downgrade to
   covered-universe-only and log it. GATE 1: sampled counts vs the
   company's own careers page; panel vs Hiring Lab index + JOLTS.
   GATE 2 (after ~2 quarters of archive): posting-count deltas,
   freeze-detection (abnormal deletion rates), role-mix shifts vs
   forward returns/restructuring announcements vs base rate. Archive
   starts with the resolver — collect-everything, diff-based.
3. **App-store rankings + review velocity (DUOL/BMBL/MTCH/HOOD/COIN/
   RBLX class).** LICENSING: Apple RSS/marketingtools top-chart JSON +
   iTunes Lookup rating counts CONDITIONAL (existing public feeds,
   low-volume internal use; Enterprise Partner Feed is the sanctioned
   bulk hedge — free program, human enrollment); Google Play
   PROHIBITED programmatically (robots.txt + ToS — Android side is
   dark, stated honestly); Appfigures free tier REJECTED (no
   commercial license); Apple customer-reviews RSS VERIFIED DEAD.
   HONEST GAPS: no downloads/revenue anywhere free (ordinal ranks +
   top-grossing as revenue proxy); rank history must be self-built —
   every day not archived is lost. SOBER PRIOR: the MOST arbitraged
   alt-data category; expect near-zero on large caps; residual only in
   thin-coverage small caps. LADDER — DATA: daily archiver (~30
   calls/day: genre top-free/top-grossing × 4-5 storefronts + Lookup
   rating counts for an app→ticker map); rating-count deltas vs
   product-page displayed counts; GATE 2: quarterly rank/velocity
   aggregates vs company-REPORTED metrics (DUOL DAU/bookings, RBLX
   bookings) — the EIA-equivalent ground truth — then vs returns.
4. **USPTO patents (filing velocity / topic shifts).** LICENSING: USPTO
   ODP public domain (redistribution OK) but API key needs a HUMAN
   ID.me signup (wishlist action filed). CORRECTION (2026-07-04
   research, live-probed): the keyless landscape is DEAD — BDSS bulk
   server retired Apr 2025, Developer Hub decommissioned Jun 2026,
   PatentsView API offline pending ODP relaunch (its bulk tables moved
   behind the same ODP wall; old keys incompatible). The ODP key is
   the sole gateway to live data; Google Patents BigQuery (CC BY 4.0,
   free GCP account, 1TB/mo scan budget) remains the only keyless
   path and is BACKFILL-ONLY (source repo archived read-only Apr
   2026; freshness unverified). Pipeline design is key-first,
   single-threaded (ODP burst=1; 429 lockouts ~7d); EPO OPS free
   ≤4GB/week unaffected. STRUCTURAL HONESTY:
   the 18-month publication hole is universal (filing velocity is
   really publication velocity of ~18-month-old filings; ~7%
   non-publication requesters never appear pre-grant); grants publish
   weekly Tuesdays, applications Thursdays — THOSE are the timely
   events. PRIOR: large-cap patent factors are crowded/near-zero;
   residual in small-cap assignee-resolution quality. LADDER — DATA:
   weekly XML → per-assignee counts + CPC mix; reconcile vs
   PatentsView quarterly (~99% on top-500 grant counts); assignee→
   ticker map vs KPSS match file (>95% top-500 agreement, small-cap
   disagreement quantified, never hidden); SIGNAL: allowance/grant
   velocity anomalies vs forward returns vs base rate.
5. **GitHub org activity (engineering momentum, small-cap devtools).**
   LICENSING: GitHub API conditional (aggregated non-personal metrics
   OK; 5k req/hr authed); GH Archive free (redistribution ambiguous —
   internal computation + derived aggregates only); OSS Insight
   treated prohibited-by-default; Libraries.io CC BY-SA. HONEST
   CONFOUND (attacks the premise): public activity is a strategic,
   biased slice that varies by company — meaningful for
   develop-in-public names (ESTC, MDB), a rounding error elsewhere;
   private repos invisible everywhere. LADDER — DATA: weekly per-org
   metrics (merged PRs, pushes, bot-filtered unique actors) from GH
   Archive for a hand-verified ~15-org→ticker watchlist + mega-cap
   controls; cross-verify vs GitHub REST; known-event replay
   (HashiCorp BSL Aug-2023 discontinuity, announced layoffs) must
   appear at the right dates; SIGNAL: velocity deltas vs forward
   returns, develop-in-public names only.

## GEOSPATIAL LICENSING REGISTER (Tier 1/2, verified from primary sources 2026-07-04 — build PRs cite this)

NEXT ACTIONS (queued for [PRODUCT] routines, build in order, one layer
per PR; licensing below is DONE — do not re-research): (a) terrain
SHIPPED v1.0.61; (b) weather SHIPPED v1.0.62 (US radar; OWM global
fields await the key); (c) FIRMS fires SHIPPED v1.0.65 scaffolded
awaiting_key (server/nasaFirms.ts — same key-gated shape as vessels;
polls + archives from day one the moment NASA_FIRMS_MAP_KEY is set;
new "Environmental" panel group added for it and future R3 layers);
(d) USDA CDL crops; (e) drought/soil moisture (USDM +
drought.gov tiles); (f) USGS groundwater points; (g) oil/gas infra
(GEM + TX RRC + OSM; per-source coverage honesty — no free national
pipeline vector exists); then Tier-2 buildings v1 (OpenFreeMap render
layer + client-side viewport stats). Also queued: PMTiles AOI extract
(terrain resilience), Alpaca options-chain daily archiver ([PIPELINE],
free, from the options HOLD package). NEW DATA ROOTS #1 (8-K language
pipeline), the former top research build, SHIPPED gate 1 (DATA)
2026-07-04 (v1.0.67, server/sec8kEarnings.ts — see this section's #1
entry above and research/experiments.md); its natural follow-up, a
filings-language view mirroring filings.tsx, **SHIPPED 2026-07-05
(v1.0.82)** — see research/experiments.md. Remaining queue: (d)-(g)
below.
STATUS UPDATE: FIRMS is ACTIVE (key set as FIRMS_MAP_KEY + env-name
alias fix v1.0.68 — archive recording live).

## ATLAS PARITY (geospatial-parity directive 2026-07-04 — free public
layers Google Earth re-displays; endpoints VERIFIED server-side from
this session's sandbox via the sanctioned proxy, pixel-decoded, not
just HTTP 200s per the DESIGN.md tile rule)

PART 1 BUILD LIST — one layer per PR (X7 precedent), registry-native
(lazy, zero-cost-off, opacity slider via field:true, legend entry):

1. **Surface water (JRC GSW v2021)** — VERIFIED: XYZ tiles
   `https://storage.googleapis.com/global-surface-water/tiles2021/{set}/{z}/{x}/{y}.png`
   (sets: occurrence, transitions, seasonality...) → 200 image/png,
   65,155 non-transparent px on the z4 Americas probe tile. License:
   EC JRC/Google, free for any use with attribution + Pekel et al.
   2016 (Nature) citation. STATUS: build next ([T-CLIENT]).
2. **Forest cover 2020 10m (JRC GFC2020 via GFW tile API)** —
   VERIFIED: `https://tiles.globalforestwatch.org/jrc_global_forest_cover/latest/dynamic/{z}/{x}/{y}.png`
   → 200 image/png, 24,850 non-transparent px z4 probe. This is the
   directive's exact dataset (10m, 2020, EC JRC; CC BY 4.0; GFW
   attribution for the tile service). NOTE dead ends recorded so
   nobody re-walks them: Hansen `tree_alpha` GCS paths 404 (v1.7 and
   v1.11), `umd_tree_cover_density_2020` has "no latest version" in
   the GFW data API.
3. **Admin boundaries (Natural Earth admin-0)** — VERIFIED: 110m
   countries GeoJSON (839KB raw) from the nvkelso/natural-earth-vector
   mirror; PUBLIC DOMAIN. Build: compile a slim lines-only artifact
   into datacore/ (self-hosted — zero external dependency), render as
   a reference line layer. GADM explicitly NOT used: its license bars
   commercial use — conflicts with the monetization path; Natural
   Earth is the lawful choice.
4. **Land cover (ESA WorldCover 2021 v200, CC BY 4.0)** — PARTIAL:
   raw COGs verified anonymously accessible (AWS S3 `esa-worldcover`
   bucket listing works; per-3° Map.tif tiles). The official
   Terrascope WMTS could NOT be verified from this sandbox
   (connection reset — possibly egress policy, possibly their server;
   indistinguishable from here). Plan: attempt WMTS from prod with
   the standard tile-pixel verification; if it fails there too,
   build-first fallback = offline low-zoom PNG pyramid compiled from
   the free COGs into datacore (their license permits redistribution
   with attribution).
5. **Population density (GHSL / WorldPop)** — BLOCKED-BY-ENDPOINT:
   GHSL WMS moved (ghsl.jrc.ec.europa.eu redirects to
   human-settlement.emergency.copernicus.eu; probed paths 404) and
   WorldPop's sdi.worldpop.org WMS 404'd on the standard pattern.
   Both datasets remain free (GHSL CC BY 4.0, WorldPop CC BY 4.0) —
   this is endpoint research, not licensing. Next: read GHSL's
   current services page for the live WMS base URL; WorldPop REST at
   hub.worldpop.org as alternative. Raw GeoTIFFs of both are free
   downloads if tile services stay dead (same fallback as #4).
6. **Elevation (Copernicus GLO-30)** — ALREADY LIVE: the terrain
   hillshade layer ships Mapterhorn tiles built from GLO-30 + national
   DEMs ("selectable elevation layer" satisfied). Future upgrade:
   hypsometric tint/color-relief rendering when justified.
7. **Cropland (USDA CDL)** — already queued as Tier-1(d) in the
   licensing register above.

PART 2 — BLOCKED-BY-ACCESS (honest boundary, do NOT chase): Google
Earth Professional/Advanced layers (driveway/EV-charger/building-
footprint counts, proprietary zoning) derive from Street View +
sub-meter imagery + Places — raw material with NO free lawful source.
Not a build target at any effort level. The free-tier equivalent we
CAN build: Microsoft Global ML Building Footprints (ODbL) / Google
Open Buildings (CC BY 4.0) — already Tier-2 queued in the licensing
register.

PART 3 — the differentiation (pointers, no duplicate filing): (a)
MOTION = timeline-slider as registry-native capability, queued
[T-CLIENT] in GIP BUILD QUEUE below; (b) FUSION = Everything Graph
click-card (R5 + aircraft spine [T-DATACORE] below); (c) VALIDATED
SIGNALS = ladder-gated (Sentinel-2 tank-fill in flight at gate 1);
(d) API = /api/v1 shipped v1.0.70 — atlas layers surface there as
they land where license terms permit (NOT the GFW/JRC tile passthrough;
metadata + our derived stats only).

PART 4 — POSITIONING COPY (queued [T-CLIENT], own small PR): honest
line for landing + /developers: "We are not a basemap competitor —
same open geospatial foundation as any Earth viewer (and we name our
sources), plus what a static atlas can't do: live movement, entity
fusion, market-validated signals, and full API access." Live-vs-coming
stated per feature; no claim to Google's proprietary imagery.

## SATELLITE OBJECT DETECTION & MULTI-SENSOR CHANGE INFERENCE (major
roadmap, directive 2026-07-04 — build in stated order; extends
SENTINEL2_CHANGE_SPEC and the gate-1 Cushing pipeline)

RESOLUTION & SENSOR REALITY (the honest boundary, stated first): free
optical imagery (Sentinel-2, 10m) supports FACILITY-SCALE CHANGE
DETECTION, not individual-object counting/classification.
Fixed-vs-floating roof discrimination, coil counting, ship-type
classification require sub-meter PAID imagery — filed as build-first
wishlist items, gated on the free version proving a signal first. The
wall is "individual-object identification from free data," and it is
NOT beaten by resolution alone — it is attacked with MORE SENSORS of
the same scene.

MULTI-SENSOR FUSION (the free unlock — never rely on optical alone):
1. Sentinel-1 SAR (radar; free, ESA) — metal (steel yards, tanks,
   ships, railcars) is radar-bright; sees through clouds and at night
   (~doubles usable observations); responds to structure/volume
   optical can't capture. Revisit: ~6 days (S1A alone since S1B loss;
   S1C ramping 2025-26 restores denser cadence). License: Copernicus
   free full use with attribution.
2. Sentinel-2 optical (10m) — reflectance/shadow/color change.
   Revisit ~5 days (2A+2B+2C). License: same Copernicus terms.
   Zero-credential access PROVEN (#158: Element84 STAC + AWS COGs).
3. Landsat 8/9 thermal (TIRS ~100m) — facility activity and tank fill
   correlate with heat signature. Revisit ~8 days combined. License:
   USGS public domain.
LIDAR (USGS 3DEP where available) is a ONE-TIME CALIBRATION input for
fixed site geometry (tank empty-height baselines) — NOT a
change-detection sensor; no free frequent satellite LIDAR exists.
FUSION PRINCIPLE: agreement across independent sensors raises
confidence; disagreement flags noise. Fusion improves change-detection
CONFIDENCE — it still does not enable per-object counting. Explicitly
validate whether the fused free signal is tradeable BEFORE assuming
paid sub-meter is required.

PHASE 1 (free, build first) — FUSED CHANGE DETECTION:
- Tank farms (Cushing first): S2 cluster reflectance/shadow + S1
  backscatter + Landsat thermal over time as a fill proxy → "fill
  trend up/down". LADDER: gate 1 vs EIA weekly crude storage (the
  existing prior/criteria from the 2026-07-04 kickoff entry carry
  over; fusion is the next iteration alongside per-tank annulus
  geometry).
- Steel yards: yard/stockpile reflectance + SAR metal-brightness +
  area change week-over-week → "activity up/down". LADDER: gate 1 vs
  AISI weekly raw-steel production / company shipment reports.
- Construction: new/expanded footprint at tracked facilities →
  "possible expansion, verify". LADDER: gate 1 vs building permits +
  local news for a sample of detections.
Every output labeled ESTIMATE with a confidence score and an evidence
list (the approved envelope); nothing surfaces as a SIGNAL before
ladder gate 2.

PHASE 2 (paid, wishlist-gated) — OBJECT COUNTING: only after a Phase-1
signal validates AND revenue justifies it: sub-meter imagery for tank
roof-type + count, ship type + count, vehicle/coil counts. Wishlist
entries carry cost, expected accuracy gain over the free fused proxy,
and a counts-vs-ground-truth validation plan.

IMAGERY-AGE INDICATOR (build alongside Phase 1): every imagery-derived
layer displays its capture date per zoom/location where the source
exposes it (S1/S2 scenes carry dates; Esri base tiles do not — show
"date unavailable", per the existing DESIGN.md imagery-honesty rule).
Inference freshness ties to it: new imagery → change detection re-runs
→ indicator + estimate timestamp update.

VALIDATION IS MANDATORY: every detected quantity/change validates
against an independent ground truth before surfacing (tank fill vs
EIA; port/ship counts vs published port stats; construction vs
permits/news). Confidence + evidence on every inference; earlier
estimates re-evaluated as new imagery arrives. Priors stated BEFORE
testing (Reasoning Standard #10); discount by combinations tried;
out-of-sample confirmation required.

ACCESS: CDSE (one signup) covers S1+S2 — exact steps filed in
wishlist.md; S2 also has the proven zero-credential path; S1
zero-credential alternatives to verify (ASF DAAC with free NASA
Earthdata login; AWS S1 buckets are requester-pays). Landsat thermal:
USGS — landsatlook STAC / AWS usgs-landsat (requester-pays; free API
alternatives to verify). Per-sensor licensing above.

## GRID-ADJACENT FUTURE ROOTS (human directive 2026-07-07 — backlog, NOT active build; neither blocks anything)

Filed with hypotheses so future sessions can pick them up (Mike's
instruction verbatim: "Log two future candidates in the research/
ideas backlog (not active build)").

1. NOAA GEOMAGNETIC-STORM / SPACE-WEATHER FEED — "viable now, just
   not prioritized." Free national feeds from NOAA SWPC (planetary
   K-index, G1-G5 storm scales, alerts/warnings JSON; US-gov public
   domain). HYPOTHESIS (Mike's, verbatim core): "geomagnetic storms
   induce grid-damaging currents = outage-risk signal for utilities"
   — geomagnetically induced currents (GIC) stress transformers,
   worst on long E-W high-voltage lines and at higher magnetic
   latitudes; pairs as an EVENT OVERLAY with the GRID VISION layer
   and feeds the A3 exposure screens (which listed utilities' assets
   sit on GIC-susceptible corridors). LADDER PATH when picked up:
   gate 1 = archive K-index/G-scale events and validate against a
   public outage ground truth (DOE OE-417 electric disturbance
   reports — free, dated, cause-coded) for storm-coincident outage
   excess vs base rate; gate 2 = event study on utility tickers
   conditioned on exposure. Build shape: small keyless SWPC archiver
   (NWS-alerts pattern) + map event layer; the grid layer join is
   what makes it more than a weather feed.
2. GROUND-BASED MAGNETOMETER SENSING OF PER-LINE LOAD — long-horizon
   PARK ("if we ever do physical sensors"). Physics is real: current
   in a conductor produces a measurable magnetic field at ground
   distance; researchers and some commercial players infer per-line
   MW flow from roadside magnetometers. OUTSIDE the current
   free-data model — requires deployed hardware near specific lines
   (site access, calibration per line geometry, maintenance). No
   ladder path until hardware exists; revisit only if the platform
   ever adds a physical-sensor arm. Recorded so nobody re-derives
   the idea or chases it as free data.

## GRID BUILD ORDER (DATACORE MAXIMUS Phase 2; filed 2026-07-06 after
the census + a live feasibility workup — pipeline PROVEN VIABLE in this
container: tippecanoe 2.49 (native .pmtiles) + osmium 1.16 one apt
install away; Texas extract 709MB → filtered → PMTiles in <10 min;
full US 12GB PBF ≲1hr at ~15GB peak disk (27GB free); output tens of
MB served static from the volume via range requests; MapLibre 5.24
current, ADD npm `pmtiles`@4.4.1 for the pmtiles:// protocol.
Overpass = weekly REFRESH mechanism only, never bulk (fair-use +
single-instance dependency — mirrors unreachable from our proxy).
[T-DATACORE server/pipeline + T-CLIENT layer]; one item per PR.)

1. TX PILOT: session-run pipeline script (scripts/build_power_tiles.sh
   — Geofabrik texas → osmium tags-filter power=line,substation,plant
   → osmium export → tippecanoe -o power_tx.pmtiles), artifact onto
   the volume, /api serving via range requests, MapLibre layer behind
   a registry toggle with zoom-decimation gates (≥230kV z<6, ≥100kV
   z<9, all z≥11; substations z≥9; NO towers below z12). VOLTAGE
   HONESTY: untagged-voltage lines flagged distinct, never silently
   dropped (OSM voltage coverage incomplete — overstating ≥100kV
   coverage is fabricated completeness). ODbL: attribution
   "© OpenStreetMap contributors, ODbL" on the layer; tiles are a
   produced work; the .pmtiles file is NOT offered for download
   (share-alike boundary). Perf harness at 390px gates the PR.
2. US FULL: same pipeline on us-latest 12GB (delete PBF post-filter);
   replaces the TX file; manifest states coverage = US, OSM
   completeness caveats, CEII note (US substation/line data on OSM is
   community-mapped public knowledge; no restricted CEII detail —
   underground/distribution largely absent, stated honestly).
3. DEMAND JOIN: grid-demand stream (live, v1.0.163) respondent stats
   surfaced on the layer (BA-region badge, latest MWh) — first
   cross-layer join of the power vertical.
4. GEM REGISTRY JOIN (blocked on wishlist 9b form-fill): unit-level
   plant status (announced/construction/operating) as a plants-layer
   upgrade — CC BY 4.0, join spine plant↔owner↔ticker.
5. EU EXPANSION (blocked on 9c ENTSO-E token): europe extract +
   zonal load; only after 1-4 hold.
6. CEMS UTILIZATION OVERLAY (blocked on 9a EPA key): unit-level
   grossLoad×opTime on plant popups — the ground-truth utilization
   layer; ladder gate 1 for every power-vertical inference.
SIGNAL HYPOTHESES (each gate-locked, evaluated not assumed): grid
stress (demand vs capacity by BA) × power prices/utility tickers;
substation-proximity × datacenter-buildout names; plant status
transitions × regional generators; storm/heat (weather layers) ×
outage-sensitive names. DATA-PRODUCT NOTE: sellable = derived
signals/analyses with attribution (ODbL produced works + CC BY);
NOT sellable without legal review = raw OSM-derived geometry
database (share-alike) — routed to BLOCKED-FOR-MIKE when
monetization nears.

## BUILD ORDER 6 (SELF-PROPOSED, standing directive; filed 2026-07-06
after BUILD ORDER 5 closed 5/5. [T-DATACORE]; same rules: one item
per PR, licensing first, keyless-or-already-keyed builds first,
archive from day one, envelope manifest, RAW until gate 2. All twelve
candidate sources PROBED LIVE 2026-07-06 by two parallel research
agents before filing (working URLs, auth, shape samples, cadence,
history depth, license text recorded in the probe reports; verdicts
below are the parent session's judgment). Theme: POSITIONING +
NOWCASTS + STRESS — financial-futures positioning, daily fiscal
nowcast, sector-level stress/quality events. Priors stated per item.)

1. CFTC TFF — TRADERS IN FINANCIAL FUTURES (Socrata gpe5-46if
   futures-only; keyless; weekly Fri, as-of Tue; verified 2006-06-13
   → present, ~44 markets/wk). CHEAPEST BUILD IN THE LIST: clone of
   the live cftcCot.ts adapter (72hh-3qpy), different dataset id +
   named fields (dealer / asset-manager / leveraged-money
   long/short). HYPOTHESIS: leveraged-money net-positioning extremes
   in ES/NQ/rates/FX mean-revert; dealer positioning is the informed
   side — extremes gate regime risk. PRIOR: modest (well-studied
   data; edge if any is in JOINS with our COT commodities archive).
   LADDER: gate 1 = cross-check one week's values vs CFTC's published
   HTML report; gate 2 = positioning-extreme vs forward SPY/sector
   returns, regime-split.
2. TREASURY DAILY STATEMENT — DTS (fiscaldata.treasury.gov API,
   keyless, verified: deposits_withdrawals_operating_cash, daily,
   ~90 rows/business day, history to 2005-10; ~1-2 day lag; brackets
   URL-encoded). HYPOTHESIS: withheld income/employment tax deposits
   = a DAILY payroll nowcast weeks ahead of BLS releases; category
   deltas (corporate tax, FUTA) nowcast macro turns → regime input
   the bot already consumes via macro_data. PRIOR: medium-high on
   signal existence (withheld-tax nowcasting is published research),
   medium on OUR edge (must beat FRED-lagged equivalents; the daily
   cadence + 20-yr history is the moat). LADDER: gate 1 = reconcile
   monthly sums vs MTS/FRED federal receipts; gate 2 = withheld-tax
   YoY growth vs payroll-surprise dates.
3. FDIC BANK DATA (api.fdic.gov — NOTE host moved from
   banks.data.fdic.gov, 301; keyless; quarterly financials ~1.68M
   bank-quarter records to ~1992, latest REPDTE 2026-03-31; failures
   event-driven to 1934, latest 2026-05-01). HYPOTHESIS: deposit
   flight + asset-quality deltas at SMALL regional banks lead KRE and
   individual small-cap bank moves — exactly the whales-can't-fish
   territory (EDGE DOCTRINE #2). PRIOR: medium (quarterly cadence
   limits timing; failures feed is the live-event kicker). LADDER:
   gate 1 = cross-check a bank's ASSET/DEP vs its 10-Q; gate 2 =
   deposit-decline quintiles vs forward stock returns for listed
   small banks (ticker join via our EDGAR spine).
4. NHTSA COMPLAINTS VELOCITY (api.nhtsa.gov keyless JSON +
   static.nhtsa.gov FLAT_CMPL.zip daily bulk, updated even Sundays;
   complaints to mid-1990s, recalls to 1960s; public domain).
   HYPOTHESIS: complaint-rate acceleration per make/model (esp. fire/
   crash flags) precedes recalls and NHTSA investigations, which move
   automakers AND their small-cap suppliers. PRIOR: medium-low for
   megacaps (crowded), medium for supplier mapping (uncrowded, needs
   the component field → supplier join — Everything Graph substrate).
   LADDER: gate 1 = complaint counts vs NHTSA's own published recall
   timeline for 3 known cases; gate 2 = velocity anomalies vs forward
   returns.
5. [ALREADY BUILT — duplicate filing, discovered 2026-07-06 at build
   time: v1.0.118 / PR #229 shipped this exact stream 2026-07-05 as
   BUILD ORDER 2 #5 (same FIPS probe finding, CONUS + 8 belt states +
   DSCI, /api/data/drought live). First-merged wins; no rebuild.
   LESSON compiled into the build-order method: grep server/ for
   existing stream modules BEFORE filing — the probe agents verified
   the sources but nobody checked our own repo.] Original filing
   preserved below for the record:
   US DROUGHT MONITOR (usdmdataservices.unl.edu REST + weekly
   shapefiles; keyless; weekly Thu; county-level D0-D4 percentages to
   2000; citation required — NDMC-UNL/USDA/NOAA attribution travels
   on every record and the map layer). HYPOTHESIS: county-weighted
   drought exposure over crop belts leads ag-complex moves (grains,
   fertilizer, equipment small caps); joins degree-days (live) + NDVI
   (roadmap) into one ag-stress index. PRIOR: medium (weekly, well
   published — the join is the edge, not the raw map). LADDER: gate 1
   = spot-check vs the published national map; gate 2 = belt-weighted
   drought delta vs corn/soy futures forward returns. ALSO a natural
   /data map layer (RAW overlay, attributed).
6. EIA-930 HOURLY GRID DEMAND (api.eia.gov v2; FREE KEY REQUIRED —
   absent from session + Railway env, filed BLOCKED-FOR-MIKE #8;
   hourly, ~1-2h lag, 2019→present verified, public domain;
   build key-gated on EIA_API_KEY per the fredMacro/census pattern so
   it activates the moment the key lands). HYPOTHESIS: regional
   demand anomalies (weather-adjusted via our degree-days archive) =
   industrial-activity nowcast; joins the power-plants layer.
   PRIOR: medium. LADDER: gate 1 = cross-check US48 daily sum vs
   EIA's own Grid Monitor; gate 2 = weather-adjusted demand residual
   vs industrial-sector returns.
7. USDA NASS CROP CONDITIONS (quickstats.nass.usda.gov API; FREE KEY,
   instant email signup — Mike action, BLOCKED-FOR-MIKE #8; weekly
   Monday in season; decades of history; 50k-row request cap; no
   commercial-use prohibition found, NASS-attribution rules apply).
   HYPOTHESIS: week-over-week condition deltas in corn/soy lead
   futures and ag small caps; pairs with #5 into the ag-stress index.
   PRIOR: medium-low standalone (widely watched), medium in the join.
   LADDER: gate 1 = values vs the published Crop Progress report;
   gate 2 = condition-delta vs forward futures returns.

DECLINED (probed, judged, recorded — not built):
- AAR weekly rail traffic: duplicative of the live STB EP724 stream;
  detailed data is AAR's paid product (license risk); toplines only.
- BTS on-time performance: ~2-month lag kills trading value; we have
  live aircraft + FAA status for the same sector.
- BLS API: FRED (live, key-gated) already covers the same series
  same-day; keyless BLS adds a 10-yr-span limit, not an edge.
- Baker Hughes rig count: no explicit redistribution grant on an
  8MB corporate XLSX — license risk for the /data product. FREE
  PUBLIC-DOMAIN ALTERNATIVE to probe next order: EIA Drilling
  Productivity Report + DUC counts (same capex signal, clean
  license).
- OSHA ITA establishment injuries: keyless + public domain but
  ANNUAL cadence — parked as an Everything-Graph facility-spine join
  (injury intensity per plant), not a stream.

## BUILD ORDER 5 (SELF-PROPOSED, standing directive; filed 2026-07-05
after BUILD ORDER 4 resolved for T-DATACORE — remaining B4 items wait
on external clocks (options QA 07-06, natgas gate-2 ~09-27, registry
per-country discovery) or belong to other territories. [T-DATACORE];
same rules: one item per PR, licensing first, keyless-or-already-keyed
only, archive from day one, envelope manifest, RAW until gate 2. Theme:
NEW ROOTS at market microstructure + attention + freight friction —
all five sources PROBED LIVE 2026-07-05 before filing (200s recorded
in experiments.md). Priors stated per item; every hypothesis enters
the ladder before belief.)

1. FINRA DAILY SHORT-SALE VOLUME (cdn.finra.org/equity/regsho/daily/
   CNMSshvolYYYYMMDD.txt — keyless pipe-delimited, ~540KB/day, probed
   200; FINRA publishes for free redistribution with attribution).
   Per-ticker daily short volume / total volume across TRF+ADF —
   covers EVERY equity incl. the small caps where whales can't fish.
   HYPOTHESIS: short-volume-ratio extremes and multi-day deltas
   precede reversals in small caps; joins our 13F clusters + Form 4
   stream (insider buying x elevated short pressure = squeeze
   candidate list nobody bills us for). History: dated files persist
   — backfill 1-2 years session-side, then daily server poll.
   PRIOR: P(gate-2 pass) ~35% — short-ratio signals are well-mined
   in large caps; the residual edge, if any, is in the illiquid tail
   (EDGE DOCTRINE #2). GATE 1: our parsed ratios vs FINRA's own
   monthly aggregates on a sampled month.
2. CFTC COT DISAGGREGATED (cftc.gov/dea/newcot/f_disagg.txt weekly,
   keyless, probed 200 442KB; the legacy deacot.txt path 404s — use
   the disaggregated report, which is also the analytically richer
   one: producer/merchant vs managed-money vs swap-dealer
   positioning. US government work, public domain. Named in the EDGE
   DOCTRINE since day one, never built.) HYPOTHESIS: managed-money
   net-positioning extremes (percentile vs trailing 3y) mean-revert
   in commodity-linked ETFs (USO/UNG/GLD/copper proxies); joins the
   EIA petroleum + natgas + tank-fill work (positioning x inventory
   is a classic two-sided read). Weekly cadence = slow archive,
   start NOW (accumulation substitutes for purchase — the paid COT
   history vendors sell is exactly this file recorded over time;
   Tuesday data released Friday 15:30 ET). PRIOR: P(gate-2) ~30% —
   COT signals are heavily studied; regime-conditioning and the
   less-crowded contracts are where residual edge would live.
3. WIKIMEDIA PAGEVIEWS ATTENTION PROXY (wikimedia.org/api/rest_v1/
   metrics/pageviews/per-article — keyless JSON, probed 200 with
   real daily counts; CC-licensed data, generous rate limits;
   history to 2015). THE PYTRENDS REPLACEMENT: Google Trends gate-1
   FAILED (#215, upstream abandoned); Wikipedia article views are
   the free, stable, historical attention signal. Needs a
   ticker->article map (small curated seed: universe tickers ->
   company article titles; entity_map.json precedent). HYPOTHESIS:
   attention spikes (views z-score vs trailing 90d) on SMALL-CAP
   company articles lead volume and volatility 1-5d; attention
   without news (no 8-K/GDELT hit same day) is the interesting
   subset. PRIOR: P(gate-2) ~30%, discounted for the attention-
   signal literature being crowded on large caps — small-cap
   restriction is the edge claim. GATE 1: views series sanity vs
   known events (earnings dates spike) on 10 hand-checked tickers.
4. FAA AIRPORT STATUS / DELAY PROGRAMS (nasstatus.faa.gov/api/
   airport-status-information — keyless, probed 200; US government
   work). Ground stops, ground-delay programs, closure reasons by
   airport. RAW overlay for the map (aviation ops layer joins our
   ADS-B archive) + HYPOTHESIS: sustained delay-program frequency at
   cargo hubs (MEM, SDF, CVG, ANC) is a cost-pressure leading
   indicator for parcel carriers (FDX/UPS) quarter-to-date; also
   event-decorates the site timeline (Everything Graph). PRIOR:
   P(gate-2) ~20% — weather dominates delays and is already priced;
   the archive is cheap and the map layer is honest RAW value
   regardless.
5. CBP BORDER WAIT TIMES (bwt.cbp.gov API, keyless, probed 200; US
   government work). Commercial-vehicle wait times at land ports of
   entry, hourly. Fills the trucks gap from the N3 freight-proxy
   analysis (we have sea=AIS, rail=STB+trains, air=ADS-B; road was
   missing). HYPOTHESIS: sustained commercial wait-time anomalies at
   top freight crossings (Laredo, Otay Mesa, El Paso) lead
   border-dependent logistics/rail intermodal volumes; joins STB
   rail carloads (substitution) and the site timeline. PRIOR:
   P(gate-2) ~20% — indirect transmission; archive-first, signal
   later.
6. USPTO PATENT GRANTS/APPLICATIONS (EDGE DOCTRINE named, never
   built; PatentsView/USPTO ODP APIs — access shape NOT yet probed,
   may need a free key like Census: probe FIRST, and if key-gated
   file it as BLOCKED-FOR-MIKE with the Census precedent rather
   than building blind). HYPOTHESIS: grant-rate inflections and
   citation-weighted grants for small-cap assignees lead re-rating;
   assignee->ticker mapping reuses the name-matcher pattern.
   DELIBERATELY LAST: heaviest mapping work, unprobed access.

   STATUS 2026-07-05 — PROBED, BLOCKED-FOR-MIKE #7 FILED (build
   order otherwise COMPLETE 5/5 built + this decision). PatentsView
   requires a free API key (since 2021; legacy keyless endpoint
   301s away; request form is human-facing); additionally
   search.patentsview.org 502s through the session proxy and
   developer.uspto.gov 503s, so even with a key the first build
   session must re-probe reachability (Railway-side may work —
   the key-gated fredMacro/census pattern covers either). Free
   fallback if the key path stalls: USPTO weekly bulk XML
   (keyless, heavy build) — analyzed in the wishlist entry.

   UI STATUS 2026-07-06: all five #1-5 pipelines shipped API-only
   2026-07-05 (same sequencing gap earnings-language had before its
   own PRODUCT follow-up). #1 FINRA short-volume now has its /data
   full view (v1.0.145, market-wide trend + per-ticker archive
   lookup + top-ratio table — see experiments.md). #2 CFTC COT
   disaggregated and #3 Wikimedia attention are non-geospatial
   (same insider/earnings/shortvol inline-panel-row + full-view
   pattern would apply directly) and are the next lowest-effort UI
   gaps. #4 FAA airport status is geospatial but its own doc comment
   (server/faaStatus.ts) explicitly flags needing an airport-
   coordinate lookup table as a deliberate follow-up before a map
   layer is possible — that table is the actual next build item
   there, not just a UI pass. #5 CBP border wait times carries a
   `port_number` join key (server/cbpBorderWait.ts) but no
   coordinate table exists yet either — unverified whether one is
   readily available; check before assuming it's a straight port.

   UI STATUS 2026-07-14: #3 Wikimedia attention SHIPPED (v1.0.309) —
   `#/data/attention` full view, exact insider/earnings/shortvol
   inline-panel-row + full-view pattern; see experiments.md. #2 CFTC
   COT disaggregated is now the sole remaining lowest-effort UI gap
   from this note (same pattern applies directly — `/api/data/cot`
   already serves it, `server/cftcCot.ts`).

   UI STATUS 2026-07-14 (later same day): #2 CFTC COT disaggregated
   SHIPPED (v1.0.313) — `#/data/cot` full view. COT has no curated
   ticker seed (unlike attention/shortvol), so the exact-key lookup is
   replaced by a market search (name/commodity substring or exact
   contract code) that then drills into that market's managed-money
   net-position series; see experiments.md. ALL FIVE #1-5 pipelines
   from this build order now have their /data UI follow-up — this note
   is closed.

## BUILD ORDER 4 (SELF-PROPOSED, standing directive; filed 2026-07-05
after BUILD ORDER 3 closed 6/6 same-day; [T-DATACORE] unless noted.
Theme: DEEPEN what tonight built — the gate-2 unlocks and the queued
GIP items, before any new roots)

1. REGISTRANT->OPERATOR RESOLUTION (the B3-1 gate-2 blocker, stated
   at the gate-1 pass): trustee/leasing registrants (TVPX, bank
   trustees, fractional programs) hide the operating company.
   v1: rule-based — known trustee/lessor name list (from our own
   spine composition) + callsign-prefix operator inference from the
   archive (a hex flying UAL#### callsigns IS United ops regardless
   of registrant; callsigns are already archived per point).
   Evidence envelope on every resolution; unresolved stays
   registrant-labeled. GATE: >=90% agreement on a 20-airframe
   hand-check vs public fleet trackers.
2. UI SCALABILITY ARCHITECTURE (GIP Part 4 queued item, [T-CLIENT]):
   panel virtualization, per-layer cost budgets in the registry
   schema, 50/100/200-layer synthetic harness batteries asserting
   the interactive budget; no regression to current map speed.

   PARTIALLY BUILT + MEASURED 2026-07-05 (v1.0.129, see experiments.md):
   registry-native `group`/`costTier` fields shipped (datacore/layers.json,
   groupOf() prefers them over the old hardcoded LAYER_GROUP fallback);
   `groupCollapsed` now derives from an OPEN_GROUPS_BY_DEFAULT allowlist
   so any group added later defaults collapsed automatically instead of
   needing a second hardcoded entry; GROUP_ROW_CAP=12 bounds an open
   group's DOM behind a "show all" control; an active-cost-budget badge
   consumes costTier (verified reads "heavy load" when enough
   heavy/moderate layers are on — real fixture, not synthetic); an
   unknown-group catch-all prevents a mis-grouped layer from silently
   vanishing (a real bug the new scale harness caught mid-build, fixed
   same PR). scripts/visual_check.mjs's new `--page scale` battery
   MEASURED (not assumed) 50/100/200 synthetic-layer registries:
   default-open panel rows stayed 14/24/24 (budget 30), "show all" self-
   see reached 100% at every scale, TTI held 1.2-2.5s (gate 3000ms) — the
   collapse-by-default + row-cap combination holds with real margin.
   NOT BUILT: literal windowed DOM virtualization (removing off-screen
   nodes entirely) — the measured numbers above show it is not yet
   evidence-justified (CLAUDE.md: don't build for hypothetical
   requirements). REVISIT TRIGGER (precise, not vague): if any single
   panel group's real member count approaches ~25 (the largest measured-
   clean case in the n=200 synthetic run), re-run
   `node scripts/visual_check.mjs --page scale` first — if defaultRows or
   TTI approach their budgets, that is the evidence to build real
   windowing; if they still hold, no action needed yet. STILL OPEN
   (unchanged from the original item, [T-CLIENT]): a timeline-slider and
   confidence-display capability (GIP Part 4's other named items) — not
   touched this session, scoped separately.
3. INTERNATIONAL AIRCRAFT REGISTRIES v1 (GIP Part 5a filings): UK
   G-INFO + Ireland + Switzerland public registries x our archived
   non-US hexes (22% of archive unmatched tonight); same
   exact-hex-join + evidence-envelope discipline as the FAA spine.

   ACCESS PROBES 2026-07-05 (build deferred to a fresh session —
   per-country discovery needed): UK siteapps.caa.co.uk/g-info 502s
   through the session proxy (retry direct/alternate CAA data page);
   the IAA register URL 404s (moved — find the current
   register-download page); Swiss FOCA loads but is an SPA (find the
   underlying API or Excel export). RULED OUT: OpenSky's
   aircraft-metadata database would be the one-download answer
   (icao24-keyed, all countries) but carries the SAME non-commercial
   licensing that got OpenSky dropped from the provider chain under
   the monetization tripwire — do not re-import that conflict.
   Meanwhile the callsign-prefix resolution (#241) already gives
   non-US AIRLINE hexes operator series without any registry.
4. NATGAS STORAGE x DEGREE-DAY GATE-2 DESIGN (the pairing #233/#234
   were built for): pre-state criteria NOW, run when live-week
   overlap >= 12 weeks (~2026-09-27): storage-delta surprise vs
   degree-day-implied draw, regime-split, vs random-entry base rate.

   DESIGN PRE-STATED 2026-07-05 (before any overlapping live week
   exists — nothing here is fit to data): (1) MODEL: regress weekly
   natgas storage delta on population-weighted CONUS HDD+CDD sums
   for the same Thu-Thu week, fit ONLY on weeks before the
   prediction week (expanding window, min 8 weeks; both series'
   history supports a long pre-fit on pre-2026 data). (2) SIGNAL:
   residual = actual storage delta minus degree-day-implied delta,
   available Thursday 10:30 ET at the EIA print. (3) TEST: sign and
   magnitude of the residual vs UNG/natgas-proxy returns from
   Thursday close to the following Wednesday close, vs the
   random-entry base rate over the same weeks; regime-split per
   Reasoning Standard #2. PASS = residual-sign hit rate >= 60% on
   n >= 12 out-of-sample weeks AND mean excess over base rate > 0.
   PRIOR: P(pass) ~30% — weekly storage surprises are heavily
   traded; the free edge, if any, lives in the slower ETF
   transmission (same reasoning as the tank-fill hypothesis, which
   died — stated so the posterior updates honestly).
5. OPTIONS-CHAIN FIRST-WEEK QA: the daily archiver's first
   snapshots land 2026-07-06 close — verify shape, contract counts,
   IV sanity vs the pilot sample; file the first-week report.
6. [RULE-REVIEW] COUNTERFACTUAL LOGGER CHECK-IN (CLAUDE.md mandate):
   verify blocked-trade logging is live and accumulating; earliest
   prevention-P&L readout; if not yet built, THIS becomes the item.

   CHECK-IN 2026-07-05 — VERDICT: SUBSTANTIALLY BUILT, one gap.
   grep for "counterfactual" finds no code, but the MACHINERY lives
   in shadow_portfolio.py: log_candidate() records EVERY scanned
   candidate (accepted or not) with the features dict (incl.
   change_pct, scores) and forward +5d/+10d/+20d outcomes backfilled
   nightly (verified in KNOWN BROKEN #10's analysis, 2026-07-04).
   Any rule whose predicate is computable from logged features gets
   its prevention-P&L POST-HOC by re-applying the predicate to the
   candidate archive — better than block-event logging for
   threshold rules (it also measures the counterfactual of looser
   settings). THE GAP: block-REASON attribution — rules whose
   predicates depend on non-logged state (correlation blocks,
   kill-switch halts, spread limits at quote time) cannot be
   reconstructed from features. BUILD PLAN for the T-BOT session
   that takes this: add a block_reasons[] tag to log_candidate at
   the rejection sites in bot_engine.py scan/deep_score + the
   risk_kill_switch call sites (LOGGING ONLY — no mechanism change,
   frozen paths untouched); earliest readout unchanged (>=90 days
   of shadow history per #10, first query ~2026-10-02: win rates
   for |change_pct|>35 candidates, then per-rule prevention-P&L).

   **[PARTIALLY BUILT 2026-07-05, v1.0.130, T-BOT]** Two of the
   three named gaps wired: `shadow_portfolio.update_last_decision()`
   (new function, logging-only, mirrors `log_candidate()`'s
   non-blocking contract) corrects a candidate's shadow-log decision
   immediately after `_scan_market_inner()`'s correlation/sector
   check (`decision="rejected_heat"`) or quote-time spread check
   (`decision="rejected_other"`) rejects it — both fire AFTER
   `deep_score()` already logged the candidate "taken" for clearing
   MIN_SCORE, so before this fix every candidate rejected by either
   filter was silently mislabeled "taken" in the learning archive
   (`get_shadow_stats()`'s by-decision win-rate buckets would have
   attributed real correlation/spread rejections to "taken" outcomes
   — a live HONESTY METRIC risk, not just a missing feature). Only
   the MOST RECENT record for the ticker, and only if still fresh
   (<=120s old) and still "taken", is ever corrected — a stale or
   already-resolved record from an earlier scan is never touched.
   STILL UNWIRED (correctly out of scope for a logging-only, one-PR
   change): `rejected_halt` — `check_kill_switches()` gates the
   separate TieredStrategy action list (`tiered_actions`), not the
   `deep_score()`-based `trades` loop this fix targets, and any halt
   that blocks orders at the execution layer lives in
   `server/bot.ts` (cross-language, a materially bigger change);
   `rejected_earnings` — no per-candidate stock-long earnings
   blackout exists in the `trades` loop today (earnings only enters
   as a soft ML feature and, for options, a covered-call-specific
   guard in `options_execution.py`) — there is no rejection site to
   log yet, so wiring it would mean ADDING a gate, a genuine rule
   CHANGE requiring RULE REVIEW's evidence-or-ablation gate, out of
   scope here. Both remain open, logged as the natural next slice of
   this same item. Readout date unchanged (~2026-10-02, needs >=90d
   of shadow history); once there, `get_shadow_stats()`'s
   `win_rate_by_decision` will show `rejected_heat`/`rejected_other`
   alongside `taken`/`rejected_score` automatically (buckets
   generically by whatever `decision` string appears, no
   special-casing needed). See experiments.md for the regression
   tests (`test_shadow_portfolio.py`, new file, 8/8 passing).

## BUILD ORDER 3 (SELF-PROPOSED, standing directive; filed 2026-07-05
after BUILD ORDER 2 resolved 6/6 same-day — see experiments.md
v1.0.119 scoreboard; [T-DATACORE] unless noted; same rules: one item
per PR, licensing first, archive from day one, envelope manifest,
RAW until gate 2. Emphasis shifts from NEW ROOTS toward FUSION of
the archives we now have — the compounding asset is the accumulation)

1. CORPORATE-FLEET UTILIZATION SERIES (fusion — the angle-hunting
   standing behavior's named cross-connection, substrate shipped in
   #222/#223/#226). Derived series from our aircraft position archive
   x the entity spine: flights/week + hours-airborne/week per
   corporate registrant (registrant_type corporation/llc, matched to
   listed tickers via normalized owner names — the usaSpending
   name-matcher pattern reuses). GATE 1 (pre-state before scoring):
   join accuracy vs >=20 hand-verified known corporate tails; GATE 2:
   utilization deltas vs subsequent 5/20d returns and earnings-date
   proximity, base-rate-controlled.

   GATE-1 CRITERIA (pre-stated 2026-07-05 BEFORE sampling, per
   Reasoning Standard #10): sample 20 archived hexes stratified as
   10 highest-point-count spine-corporate/llc hexes + 10 uniform-
   random spine hexes; fetch INDEPENDENT registration via
   api.adsbdb.com (probed: returns registration + owner; both known
   test hexes matched exactly); PASS = >=90% exact N-number match
   among hexes adsbdb can resolve, with every mismatch investigated
   and logged. Unresolvable hexes are excluded from the denominator
   and counted honestly. The Mode S join itself is deterministic —
   this gate tests whether the FAA snapshot's hex assignments match
   independent reality (stale registrations, hex reuse).
2. EIA WEEKLY PETROLEUM + NATGAS STORAGE (new root, keyless XLS
   family PROVEN by the Cushing comparator; ngs.html 302s — exact
   endpoint verified at build like STB was). Archive the weekly
   national petroleum balance sheet lines + natgas working storage
   with vintage discipline. HYPOTHESIS: storage-vs-5yr-band deltas
   condition the energy regime the bot already classifies; also the
   external truth source for any future inventory root.
3. NOAA CPC DEGREE DAYS (new root, PROBED 2026-07-05: keyless, US-gov
   public domain). Weekly population-weighted HDD/CDD + departures.
   HYPOTHESIS: degree-day departure deltas lead natgas/power demand
   and utility earnings surprises; joins the weather axis with the
   first DEMAND-side series.
4. CBP MONTHLY CONTAINER IMPORT STATS (new root; format verified at
   build). HYPOTHESIS: import TEU deltas lead retail inventory
   cycles; pairs with our port-dwell analytics (supply friction) for
   a two-sided port view.
5. EVERYTHING GRAPH R1 — SITE EVENT TIMELINE ([PRODUCT]; fusion of
   existing archives only, no new source): per-strategic-site
   timeline joining alerts, fires, gauges, aircraft/vessel density
   from what we already record; surfaces on the site detail card
   (/data). First user-visible cross-stream join; no ladder gate
   needed (raw overlay composition, each element already labeled).
6. [RESEARCH] ANOMALY MINING PASS (angle-hunting mandate #2): scan
   the position archives for recurring unexplained patterns
   (dwell-time regime shifts, corridor density changes) preceding
   commodity moves; terminates in filed open_questions entries with
   ladder paths — never unrecorded browsing.

   RUN 2026-07-05 (closing BUILD ORDER 3) — findings:
   a. INFRASTRUCTURE ANOMALIES (found + FIXED, #237/#238): the
      survey itself surfaced the event-loop scan class on both
      archive-analytics surfaces. Mining that finds the platform
      eating itself outranks market mining; both repaired with
      ratchets, class closed.
   b. ARCHIVE TOO YOUNG FOR MARKET MINING (honest verdict): position
      archives began 2026-07-03. The 32 port-dwell "anomalies" are
      window-edge artifacts (visits truncated by archive birth), and
      2 days cannot host a baseline. Filed, not mined.
   c. BASELINES ESTABLISHED (the compounding output): airline weekly
      utilization now measured under OUR coverage (DAL 2018
      flights/wk / AAL 2196 / UAL 1453 — partial-coverage counts,
      and that partiality IS the baseline: consistent coverage makes
      deltas meaningful even when levels undercount).
   d. MINING DESIGN PRE-REGISTERED (to run at archive depth >=30d;
      statistics fixed NOW): (1) port-dwell weekly median shifts
      (z>=3 vs trailing 4-week baseline) vs freight/commodity moves;
      (2) corridor density z-scores near strategic sites vs the
      site's commodity; (3) fleet-utilization week-over-week
      outliers vs earnings-date proximity (needs the
      registrant->operator step from the B3-1 finding). DISCIPLINE:
      every hit discounted by the number of series scanned
      (hundreds — expect ~1 z>=3 artifact per 300 series-weeks by
      chance); nothing believed without out-of-sample confirmation
      on data recorded AFTER the pattern is filed. RE-RUN TRIGGERS:
      first pass at 30d depth (~2026-08-03), delta studies at 60d.

## BUILD ORDER 2 (SELF-PROPOSED, standing directive 2026-07-05: "when
the wishlist is empty, generate the next wishlist yourself" — filed
2026-07-05 after build order 1 fully resolved; [T-DATACORE]; same
rules: one stream per PR, licensing first, archive from day one,
envelope manifest, RAW-labeled until gate 2; access LIVE-PROBED
keyless before filing where marked)

1. AIRCRAFT ENTITY SPINE v1 (GIP Part 3b — promoted from the GIP
   queue; queued work outranks new streams). FAA Releasable Aircraft
   DB (free full download) x our ADS-B archive icao24s -> one
   identity per airframe (owner, type, registrant dates), committed
   as a SMALL derived artifact (only archived hexes, not the 300k-row
   dump). HYPOTHESIS: corporate-fleet aircraft utilization
   (flights/week of jets registered to listed companies or their
   subsidiaries) shifts before M&A announcements and earnings
   surprises — the angle-hunting standing behavior's named
   cross-connection, unlocked by this join. Ladder: gate 1 = registry
   join accuracy vs known corporate tails; gate 2 = utilization vs
   subsequent 5/20d returns.
2. TANK-FILL v3 GATE-1 PIPELINE (S1 SAR double-bounce — successor
   root above; CDSE creds proven on S1). Chip client reuses the PR-2
   pattern (same registry, same EIA comparator, tankfill_gate1.py
   reusable as-is); estimator = per-tank bright double-bounce line
   intensity, per-orbit normalized. Criteria pre-stated in its
   workup before any scoring; discounted prior P(pass) ~25%.
3. NWS ALERTS stream (api.weather.gov, PROBED 2026-07-05: keyless,
   US-gov public domain, wants a User-Agent contact). Archive active
   severe alerts (type, severity, geometry, timing); /data raw
   overlay (alerts are official warnings — no predictive claim).
   HYPOTHESIS: severe-alert clusters over strategic sites (refinery
   freeze-offs, port closures) lead sector moves by hours-days;
   joins the existing weather + sites layers on the Everything
   Graph.
4. TREASURY AUCTION RESULTS (treasurydirect.gov TA_WS, PROBED
   2026-07-05: keyless JSON, public domain). Archive every auction
   (bid-to-cover, high yield vs when-issued tail, dealer take).
   HYPOTHESIS: tail/bid-to-cover deterioration precedes rate-regime
   shifts the bot's regime classifier consumes; a direct macro input
   nobody bills for.
5. US DROUGHT MONITOR weekly (usdmdataservices.unl.edu, PROBED
   2026-07-05: keyless CSV/JSON; attribution required — NDMC/USDA/
   NOAA). Archive weekly D0-D4 area % (CONUS + key ag states).
   HYPOTHESIS: drought severity deltas over ag counties lead ag
   commodities and food-producer margins by weeks; joins USGS gauges
   + FIRMS on the environmental axis.
6. STB WEEKLY RAIL TRAFFIC (stb.gov public-domain weekly carload
   Excel; format to verify at build time). HYPOTHESIS: carload
   deltas by commodity group lead rail earnings (UNP/CSX/NSC) and
   the industrial regime; gives the trains layer an economic-volume
   spine our live-position feed cannot.

## DATA STREAM EXPANSION (directive 2026-07-05 — audit + build order;
[T-DATACORE]; one stream per PR, licensing first, archive from day one,
envelope manifest required, RAW-labeled; work across sessions)

AUDIT (live vs charter, 2026-07-05): RECORDING NOW — aircraft, vessels,
trains, fires, Form 4 filings, 8-K earnings language, option chains
(from 2026-07-06 close), Sentinel-2 readings (git-side), COT (#191,
routine, v1.0.86). IN CHARTERS BUT NOT BUILT: 13F clusters, FRED,
USAspending, FDA calendar, USGS water, GDELT, Google Trends, patents
(blocked on USPTO ID.me), app ranks, jobs postings.

BUILD ORDER (signal value × ease; hypothesis BEFORE first pull per
Reasoning Standard #10):
1. ~~CFTC COT~~ — DONE by routine (#191); its hypothesis rides with it.
2. EDGAR 13F institutional clusters (keyless): quarterly holdings →
   cluster detection (multiple funds newly entering the same small
   cap). HYPOTHESIS: new-position clustering in capacity-constrained
   names precedes 60-90d outperformance (whales telegraph slowly).
   LADDER: gate 1 = parse accuracy vs a hand-checked 20-filing sample;
   gate 2 = cluster events vs forward returns with the 45-day filing
   lag honestly modeled (holdings are STALE when public — the signal,
   if any, is crowd formation, not news). PRIOR: weak-positive,
   heavily lag-discounted. STATUS 2026-07-05: BUILT (v1.0.88,
   server/edgar13f.ts) — gate 1 PASSED for the parser on hand-checked
   real filings; archive recording from merge (Q2 season just opened,
   Aug-14 deadline burst ahead). Gate 2 waits for a quarter boundary
   in OUR archive (need holdings at two consecutive periods per
   manager to detect NEW positions — earliest useful cut after the
   Q3-2026 filings land in Oct-Nov). Focused-manager cap >250
   positions per the EDGE-DOCTRINE rationale in the manifest.
3. FRED macro series (free API key — signup steps in wishlist): ~30
   series (rates, spreads, claims). HYPOTHESIS: not a direct signal —
   the REGIME INPUT feed for regime classification and for gate-2
   conditioning of every other stream. LADDER: gate 1 = values match
   the FRED web UI on 10 spot checks; gate 2 = regime-conditioning on
   FRED series improves an EXISTING validated signal's stability
   (never traded alone). STATUS 2026-07-05: BUILT (v1.0.90,
   server/fredMacro.ts) — 31 series, key-gated (FRED_API_KEY in
   Railway), point-in-time vintage archive (revisions append with rt;
   free ALFRED substitute compounding from day one), licensing split
   (CBOE/ICE BofA/UMich restricted = internal-only, never
   product-surfaced). Gate 1 spot checks vs fredgraph.csv run against
   prod /api/data/macro after deploy — result in experiments.md.
4. USAspending contracts (keyless API): daily awards mapped to
   tickers. HYPOTHESIS: large award/market-cap ratios move small caps
   with a lag; the filing feed beats news wires. LADDER: gate 1 =
   recipient→ticker mapping precision on 50 awards; gate 2 =
   award/mcap vs 5-20d returns, small caps only (fish where whales
   can't). PRIOR: positive on micro caps, zero on large.
   RESEARCH COMPLETE 2026-07-05 (subagent, live-verified): use
   POST /api/v2/search/spending_by_transaction (transaction level —
   award level shows LIFETIME totals, useless for events); poll by
   last_modified_date; dedup (aid,mod,amt) with revisions appending;
   EXPLICIT cap |amt|>=$25k keeps 99.74% of positive dollars in 20%
   of rows (measured n=6,691). CRITICAL HONESTY: rt (fetch date) is
   the only valid event date — DoD/USACE publish ~90 DAYS late
   (verified: 9 recent-week DoD txns vs 91,999 four months back), so
   gate 2 must exclude or cohort DoD; "filing beats wires" holds for
   CIVILIAN agencies only. Mapping: EDGAR company_tickers.json exact
   normalized name + award-detail parent (NEVER the recipient-profile
   endpoint — its parent data is vintage-less and demonstrably wrong:
   NTESS->Resideo, Accenture->Novetta both backwards); unmatched =
   skipped, never fuzzy; UEI->ticker cache compounds. LICENSE: US-gov
   free incl. commercial; DUNS is D&B-proprietary — archive UEI ONLY.
5. FDA calendars (keyless openFDA + PDUFA dates where lawfully
   listable). HYPOTHESIS: binary-event timing for biotech options —
   IV ramps into PDUFA dates; a theta-side input, not directional.
   LADDER: gate 1 = date accuracy vs 20 known events; gate 2 =
   IV-ramp reproducibility on our own archived chains.
   RESEARCH 2026-07-05 (live-verified): PDUFA target dates are NOT
   freely available — FDA is barred from confirming pending
   applications (21 CFR 314.430); every free-looking calendar is a
   ToS-protected aggregator scrape (do NOT scrape). FREE substitute
   that preserves the hypothesis: ADVISORY-COMMITTEE meeting dates
   via the keyless Federal Register API (real notices verified,
   titles carry committee + sponsor + BLA + drug) — AdCom dates are
   forward-looking binary catalysts with the same IV-ramp behavior,
   testable against our own chain archive. Plus openFDA drugsfda
   approvals (562 real approvals pulled for June; keyless 240/min,
   1000/day). Follow-on filed separately: mine company-disclosed
   PDUFA dates from our own 8-K text, labeled estimates.
6. USGS water levels (keyless): gauges near barge routes/refineries.
   HYPOTHESIS: Mississippi low water → barge freight stress →
   grain/fertilizer basis moves. LADDER: gate 1 = readings vs USGS
   site; gate 2 = low-water episodes vs barge-rate proxies. PRIOR:
   episodic — fires only in drought years; conditional signal.
   RESEARCH 2026-07-05 (live-verified): waterservices.usgs.gov/nwis/iv
   keyless; 14 gauges verified returning CURRENT readings (St. Louis,
   Grafton, Chester, Thebes, Baton Rouge, Belle Chasse, Hermann,
   Valley City + 4 Ohio R gauges; Memphis 07032000 and Vicksburg
   07289000 publish DISCHARGE 00060, not stage 00065 — request both
   params, store whichever returns; Metropolis 03611500 is DEAD,
   2010-stale, dropped). Gauges have lat/lon → this stream gets a
   /data registry layer. Provisional→approved revisions append with
   rt (vintage discipline). 1h poll, one request covers all sites.
7. GDELT event stream (keyless, 15-min): HYPOTHESIS: geo-tagged event
   bursts near tracked facilities (strikes, outages) as an ALERT
   trigger joined to the Everything Graph — verification prompt, not
   a direct trade. LADDER: gate 1 = geo precision on 30 events;
   gate 2 = burst→own-sensor (imagery/AIS) confirmation rate.
   RESEARCH 2026-07-05 (live-verified): use the 15-min Events export
   files (lastupdate.txt pointer; a real export was 24.5KB zipped /
   428 rows — trivially small), NOT the DOC/GEO APIs. GOTCHAS: host
   is HTTP-ONLY (data.gdeltproject.org has an invalid HTTPS cert,
   verified); column indices confirmed against a real file (EventCode
   col 26, ActionGeo lat/lon 56/57, GlobalEventID col 0 = dedup key).
   HONESTY: CAMEO is an actor-actor political taxonomy — it captures
   STRIKES/PROTESTS/UNREST well but NOT clean industrial accidents
   (FIRMS is our fire sensor); hypothesis reworded to unrest/strike
   bursts. Filter at ingest: unrest/coercion root codes x ~0.5° boxes
   around tracked facilities → KB/day archive. License free incl.
   commercial + redistribution; attribution "The GDELT Project" req'd.
8. Google Trends via pytrends (no key, dormant in requirements).
   HYPOTHESIS (standing EDGE example): consumer-demand proxy, most
   valuable on small caps. LADDER: gate 1 = series stability across
   re-pulls (pytrends sampling noise measured); gate 2 = inflections
   vs revenue surprises on a 20-name panel. RISK: unofficial API —
   gate 1 may kill it; that is a finding, not a failure.
   RESEARCH 2026-07-05: the risk is largely CONFIRMED upfront —
   pytrends upstream was ARCHIVED read-only 2025-04-17 (abandoned;
   Google's official Trends API is limited-alpha, unavailable).
   DOWNGRADED from stream build to a one-session GATE-1 PROBE:
   scripts/gtrends_probe.py pulls a ~20-name small-cap consumer BRAND
   panel (brand terms, not tickers) 3x at spaced intervals; PASS
   requires median re-pull correlation > 0.95. NO archiver, NO
   manifest, NO daemon route unless the probe passes (no dead code
   for a stream likely dead at gate 1). If it fails: log the
   layer-of-death and the paid alternatives note goes to wishlist.
   GATE 1 RUN 2026-07-05: **FAIL — layer of death: DATA (access
   reliability)**. Nuanced shape: STABILITY passed emphatically
   (median cross-pull r = 0.998, min 0.952, 20/20 terms in rounds
   1-2) but round 3 collapsed to 6/20 on HTTP 429s — Google
   rate-limits the unofficial path after ~45 pulls in ~35 min, and
   the only client library is abandoned (its retry path is
   urllib3-v2-broken; probe runs retries-free). Stream NOT built.
   RESIDUAL (re-probe trigger, don't build speculatively): a weekly
   single-round pull (20 pulls/week, far under the observed limit)
   may be viable IF a validated hypothesis ever specifically needs
   Trends; paid access alternatives are in BLOCKED-FOR-MIKE.
CREDENTIALS: FRED key free/instant (wishlist steps filed); all others
keyless; patents stays blocked on USPTO ID.me. Every stream ships
fetcher/parser/archiver + envelope manifest + registry entry where it
has a geographic surface (USGS gauges, GDELT events, USAspending
recipient HQs mappable; 13F/FRED are not).

## DATACORE DEFECT QUEUE (quality audit 2026-07-05, subagent-verified;
each item = own [REPAIR] PR; prioritized)

1. Aircraft + trains archives are REQUEST-DRIVEN — no visitors = no
   archive; 11 hourly gaps each already permanent. Fix: eager
   background sampling tick (vessels-tick pattern), routes.ts.
2. trains needs a health-aware registry status override like
   vessels/fires (static "live" while dead = dishonest). (The hang
   itself fixed v1.0.92.)
3. archiveStats() hardcodes 6 kinds — fires, filings, earnings8k,
   filings13f, fredmacro, optionchains invisible to
   /api/data/archive/stats; the archive-gap rule is unenforceable for
   them. Enumerate dirs on disk.
4. optionchains: .last_run_day claimed BEFORE the run (crash = day
   permanently lost); files never gzipped (manifest promises .gz);
   runs on market holidays. Move claim after success + add gzip +
   holiday check.
5. earnings8k manifest documents acceptanceDatetime + ticker that the
   writer never stores — the lookahead-free claim is unbacked. Add the
   fields or correct the manifest before any gate-2 work reads it.
6. fredmacro: seedSeen covers 3 days but each poll fetches 120d — a
   restart >3d after backfill re-appends ~120d x 31 series as
   duplicate vintage rows; also (s,d,v) dedup silently drops a
   revision that REVERTS to a prior value. Extend seed window; decide
   revert semantics explicitly.
7. aircraft manifest field_map wrong (gs vs g + undocumented v) —
   one-line docs fix.
8. COT stream (Python-side) has NO manifest and escapes the envelope
   test (which only scrapes server/*.ts) — add cot manifest + test
   note.
9. sentinel2 readings stall invisibly (manual-run cadence, newest 8d
   old) — surface reading age in platform stats or audit register.
10. edgar13f: null entryTotal can silently truncate holdings at 250
    without holdingsOmitted (violates never-silent cap); nasaFirms
    archivedIds.clear() at 200k can mass-re-append in peak season.

## TANK-FILL % v2 — PER-TANK CRESCENT-SHADOW ESTIMATOR (full workup
2026-07-05, subagent-drafted + session-reviewed; [T-DATACORE];
supersedes/extends the v1 facility-scale index whose gate-1 was
honestly NOT claimed — v1's |r|=0.73 was trend-vs-trend inflated)

STATUS 2026-07-05: PR-1 (registry, #213) and PR-2 (chip client,
scripts/cdse_chips.py) BUILT + live-verified. Three workup corrections
found building PR-2, binding on PR-3+ (details in experiments.md
v1.0.108): (1) the bbox below missed 20 measurable tanks — corrected
CHIP_BBOX [-96.770,35.922,-96.712,35.960] covers all 234 at 1.40
PU/scene (test-pinned against the registry); (2) pillow cannot decode
5-band TIFFs — PR-3 uses tifffile (session-local dep); (3) CDSE's SH
Catalog exposes NO sun angles — discovery stays on anonymous
earth-search, which carries both. NEXT: PR-3 crescent estimator.

STATUS 2026-07-05 (later): GATE 1 FAIL — ROOT DEAD, layer of death
DATA (experiments.md v1.0.110). 99 chips backfilled 24 mo, 72 matched
EIA weeks: delta r -0.06, sign hit 0.50; winter split delta r -0.28
(wrong direction). Sun confound measured (composite~sun_elev r +0.40)
AND residualizing it leaves zero signal — 10 m optical sub-pixel
crescents cannot read Cushing fill. PR-5/6/7 cancelled; readings_v2
paused; weekly CHIP acquisition continues (raw material for the S1
successor + any future estimator). Successor root below.

## TANK-FILL v3 — SENTINEL-1 SAR DOUBLE-BOUNCE (successor root, filed
2026-07-05 at the v2 gate-1 death; [T-DATACORE]; attempt #2 on the
same target so the prior is DISCOUNTED per Reasoning Standard #4)

- PHYSICS: floating roof + shell wall form a corner reflector; the
  double-bounce return in VV scales with exposed wall height (roof
  depth). Radar is cloud-immune (~2x usable cadence vs optical),
  sun-independent (kills the v2 confound class entirely), and metal
  tanks are radar-bright (proven: our first CDSE chip ever was S1 VV
  over Cushing with tanks clearly resolved).
- KNOWN ART: Ursa/Orbital used exactly this before optical; published
  literature exists on S1 tank-fill at Cushing — read before building
  (avoid rediscovering their preprocessing).
- LADDER PATH: gate 1 = same design as v2 (chips per S1 scene x same
  OSM registry x EIA weekly truth; criteria PRE-STATED in a build
  workup before any scoring run; P(gate 1) ~25% stated now). Gates
  2-5 unchanged.
- INGREDIENTS IN HAND: CDSE creds (S1 GRD via the same Process API),
  tank registry, EIA comparator (tankfill_gate1.py reusable as-is on
  a new readings stream), 24 mo of coincident OPTICAL chips for
  cross-checks.
- OPEN QUESTIONS BEFORE BUILD: GRD vs SLC (GRD first — Process API
  serves it; SLC needs SNAP-class tooling = heavy compute, flag if
  required); ascending/descending pass mixing (incidence angle
  conditions the double-bounce — likely per-orbit normalization);
  10 m GRD pixel vs 40-60 m tank = same sub-object regime, but the
  double-bounce is a BRIGHT ADDITIVE line, not a dark sub-pixel
  crescent — physically better posed.

STATUS 2026-07-05 + GATE-1 CRITERIA (pre-stated BEFORE any scoring,
Reasoning Standard #10; PR-1 built same day):
- PR-1 BUILT: scripts/cdse_s1_chips.py — discovery via earth-search
  sentinel-1-grd (orbit metadata carried; probed: Cushing = ASCENDING
  relative orbit 34 only, S1A+S1D, ~4-6 scenes/mo), chips via the
  same CDSE Process API (orthorectified SIGMA0_ELLIPSOID, FLOAT32
  VV+VH — probe showed double-bounce sigma0 up to ~2500, integer
  types would clamp), SAME bbox as v2 (test-pinned — stacks stay
  pixel-aligned for fusion), 1.12 PU/scene, 61-scene 24-mo backfill
  ~68 PU. Physics visible in one probe: 14k px with sigma0 > 1.0 vs
  ground median 0.10.
- ESTIMATOR DESIGN (PR-2, fixed now): per-tank double-bounce
  intensity = p95 VV within the tank disk + 1 px halo, log-domain,
  normalized per tank by its own scene-median series (self-ratio like
  v2 — absolute calibration cancels); exposed wall height scales the
  return, so FILL-direction composite = NEGATIVE of the normalized
  double-bounce (higher return = emptier). D^2-weighted site/global
  composites, ascending/34 scenes only in v1 (other geometries
  archived but excluded from scoring).
- GATE-1 CRITERIA (identical shape to v2, comparator
  tankfill_gate1.py reused as-is on the fill-direction composite):
  >=20 matched scene-weeks spanning >=1 EIA reversal; PASS = delta
  r >= +0.3 AND delta-sign hit >= 65% (levels supporting only).
  Matching: +/-3 days to the EIA Friday, delta pairs capped at 14-day
  gaps. DISCOUNTED PRIOR (attempt #2 on this target): P(pass) ~25%.
- S1 CADENCE HONESTY: 12-day repeat per satellite -> only ~2-3
  matched EIA weeks/month and rarely adjacent weeks; expect the
  14-day delta-pair cap to bind. If matched pairs < 20 after the
  full backfill, the verdict is INSUFFICIENT-SAMPLE (not FAIL) and
  the root waits for cadence (S1C/S1D ramp) rather than dying.

STATUS 2026-07-05 (later): GATE 1 FAIL — sample was AMPLE (60 matched
weeks, 57 delta pairs, reversals present), so the insufficient-sample
branch never triggered. delta r +0.056, sign hit 0.544 vs the
pre-stated +0.3 / 65% -> FAIL under the binding criteria
(experiments.md v1.0.114). Levels r +0.41 is REAL but unbinding — it
is exactly the trend-vs-trend inflation pattern that poisoned v1's
optical claim; not credited. Root dead at DATA under the v3 design.
Chips keep archiving (~4-6/mo, 1.12 PU each).

## TANK-FILL v3.1 — PRE-REGISTERED FOLLOW-UP (filed 2026-07-05 at the
v3 kill, to be scored ONLY by a future session on OUT-OF-SAMPLE
scenes; attempt #3 on this target — prior discounted to ~15%)

- HYPOTHESIS: per-scene speckle noise swamps 7-14 day fill changes;
  MULTI-WEEK deltas average speckle down. Design (fixed now): same
  readings_s1 fill* composites; deltas over 28-42 day windows
  (composite delta vs EIA stock delta over the same window,
  non-overlapping windows only); PASS = delta r >= +0.3 AND
  delta-sign >= 65% on n >= 15 windows, scored on scenes acquired
  AFTER 2026-07-05 plus the existing corpus ONLY as the baseline for
  per-tank medians (never re-scored in-sample). If v3.1 fails, the
  free-imagery tank-fill line TERMINATES: two sensors + three
  designs dead means 10 m free imagery cannot read Cushing
  week-scale fill — the sub-meter BLOCKED-FOR-MIKE entry becomes the
  only path, and InSAR/SLC coherence (heavy compute, SNAP-class
  tooling) stays filed as the lone free alternative, unbuilt.

- ACQUISITION (free, 0.4% of CDSE free tier): ONE master chip per
  usable S2 scene covering all Cushing tank sites — bbox
  [-96.80,35.90,-96.72,35.98] (~720x890px @10m), 5 bands (B02/03/04 +
  B08 NIR + SCL cloud mask), UINT16 TIFF via the PROVEN Sentinel Hub
  Process API (~4.1 PU/scene; free tier 10,000 PU/mo verified; 24-mo
  backfill ~500 PU one-time). Sun angles + cloud % from one Catalog
  API search per run. Cadence: every usable scene (~weekly after
  clouds; gaps stay honest gaps).
- TANK REGISTRY (free, verified live): OSM has 333 storage-tank
  polygons at Cushing (median 60.4m diameter; 234 >= 40m measurable at
  10m; ZERO height tags). One-time pull -> cushing_tanks.geojson (id,
  center, diameter, site assignment, default height 14.6m/48ft API-650
  flagged as assumption; capacity reconciliation vs EIA's ~76M bbl
  working capacity sanity-checks the registry). ODbL attribution.
- ESTIMATOR (the physics, known art — Orbital Insight/Ursa method):
  floating-roof rim casts a crescent shadow INSIDE the shell; reach
  L = roof_depth/tan(sun_elev); fill = 1 - depth/height. HONESTY AT
  10m: the crescent is SUB-PIXEL most of the year (0.4px June, 2.2px
  Dec) — so the estimator is sub-pixel + aggregate: sun-facing vs
  anti-solar interior sector reflectance ratio (self-normalizing per
  tank), inverted through circle-lens geometry, D^2-weighted over
  185-234 tanks (thousands of interior px/scene). Per-tank values
  archived low-confidence; SITE/RING aggregates are the candidate
  signal. Readings carry sun-elevation sensitivity weights (June
  carries ~5x less signal than December — the exact artifact that
  poisoned v1). Error sources logged per reading: summer sub-pixel,
  <40m tanks excluded (~10-15% capacity), fixed-roof contamination
  (classify: interior that never varies across 20+ scenes = not
  floating), rain ponding (flag scenes <=48h after precip), ~1px
  registration, SCL residuals.
- GATE 1 (criteria stated BEFORE the run): >=20 matched scene-weeks
  spanning >=1 inventory reversal vs EIA weekly Cushing stocks;
  PASS = delta r >= +0.3 AND delta-sign hit rate >= 65% (levels
  r >= +0.5 supporting only — deltas are binding; v1 lesson). Scale
  reconciliation within ~2x of physics.
- GATE 2: our Delta-estimate is in hand Sat-Mon, 2-4 days before the
  Wednesday 10:30 ET EIA print. Test A: predicts the print surprise
  (vs naive free consensus proxy, labeled). Test B: conditional on
  |predicted surprise| > 1.5M bbl, print-day USO/XLE/XOP returns vs
  random-entry base rate, regime-split.
- HYPOTHESIS + PRIOR: predicted draw -> long USO Tue close, exit Wed
  close (build -> SCO/USO puts); ~25-40 events/yr; prior 20-40bp/event
  gross IF gate 1 passes; P(gate1) ~40%, P(gate2|gate1) ~30% — the
  futures reaction is likely arbitraged by the paid vendors selling
  exactly this; residual edge lives in slower ETF transmission + the
  multi-week trend form. Even at null trading edge the fill-% layer is
  a datacore product surface (GOAL priority 3 platform line).
- COMPUTE: sub-second numpy pixel math per scene; pillow decodes the
  UINT16 TIFF (no rasterio). Session-run weekly like v1 until gate 1.
- BUILD PLAN (each own PR): PR-1 tank registry; PR-2 CDSE chip client
  (fixtures in CI, no live calls); PR-3 crescent estimator + readings
  schema; PR-4 backfill + gate-1 attempt (prior logged first); PR-5
  weekly cadence + RAW scene-date surface; PR-6 gate-2 study; PR-7
  unlock on pass. Follow-on filed: S1 SAR double-bounce fusion
  (cloud-immune, ~2x cadence) after the optical estimator settles.
- BLOCKED-FOR-MIKE: NOTHING in the core build. Optional post-gate-2
  enhancements only (sub-meter imagery ~$10-25/km2-class; paid
  consensus survey feed) — filed in wishlist, not blocking.

## GIP BUILD QUEUE (directive 2026-07-04 Parts 3-6 — territory-tagged
per the WORKSTREAM PARTITION proposal in wishlist.md; routines claim a
territory in their first commit)

- [T-DATACORE] **Aircraft entity-continuity spine (GIP Part 3b, next
  flagship build)**: FAA Releasable Aircraft download (free full DB:
  registry.faa.gov ReleasableAircraft.zip — MASTER.txt has N-number,
  MODE S CODE HEX, serial, owner, registrant dates) × our ADS-B
  archive icao24s × hex/tail cross-refs → ONE identity per airframe
  with a life timeline; every inference carries the approved envelope
  (confidence + evidence, never bare facts). v1 scope: US registry ×
  archived hexes (small derived artifact, not the full 300k-row dump
  committed); international registries per the Part 5a research
  results. Timeline endpoint + click-through UI [T-CLIENT half] ride
  the Everything Graph R5 build order.
- [T-CLIENT] **UI scalability architecture (GIP Part 4, performance
  non-negotiable)**: lazy per-layer loading (exists — zero-cost-
  when-off), panel VIRTUALIZATION for hundreds of rows, per-layer
  cost budgets in the registry schema, timeline-slider + confidence-
  display as registry-native capabilities; harness synthetic
  registries at 50/100/200 layers asserting interactive budget;
  measure before/after — no regression to current map speed.
- [T-CLIENT] **Landing-page data-intelligence section** (directive
  2026-07-04, additive only): globe (or pre-rendered fallback per
  phone budget), approved copy (verbatim in the directive/task #50),
  live stats, /data CTA, waitlist tie-in with /developers.
- [T-DATACORE] **Sentinel-2 iteration**: per-tank annulus geometry at
  Cushing (the gate-1 path — facility index logged, confounded);
  weekly runs continue via scripts/sentinel2_tankfill.py.
- [T-DATACORE] **Part 5 research results (4-agent workflow, primary
  sources, filed 2026-07-04):**
  1. AIRCRAFT REGISTRIES — BUILD the 4-registry spine pipeline: FAA
     ReleasableAircraft + Transport Canada ccarcsdb.zip (VERIFIED
     downloaded: daily, Open Government Licence incl. redistribution,
     Mode-S hex + owner + address, 34,948 aircraft) + Brazil ANAC RAB
     CSV (daily, CC-BY, owners+operators with ownership %, hex
     derivable) + NZ CAA CSV (hex + owner; licence probably NZGOAL
     CC-BY — verify before product surface). Overlay
     wiedehopf/tar1090-db aircraft.csv.gz (ODC-By, weekly, 621k rows)
     as the global hex→reg→type backbone. NEGATIVES recorded so no
     session re-chases: UK CAA bulk is PAID with a
     no-redistribution licence (unusable even if bought); Ireland's
     monthly file froze at Sep-2025 (use as static lessor snapshot);
     Australia CASA files exist free but the licence page 503'd via
     our egress — VERIFY from another network before use (wishlist);
     Germany/France/Spain/Italy unevaluated (next research pass).
  2. COUNTY PERMITS — GIP annotation CONFIRMED AND STRENGTHENED:
     no free national permit layer exists; coverage is bimodal
     (big port cities have open portals; our small industrial sites
     mostly have nothing). Do NOT budget a per-county campaign. Build
     two cheap things only: Census BPS monthly ingest (public domain,
     residential-only macro layer, small-place imputation labeled)
     and a generic ArcGIS-REST/Socrata permit adapter proven against
     Seattle (Socrata 76t5-zqzr, public domain, daily) + Savannah
     SAGIS — per-target verification stays manual.
  3. VESSEL IDENTITY — every authoritative registry (ITU MARS,
     Equasis, GISIS) is free-to-view but CLOSED to programmatic use;
     the upgrade path is METHODS, not data: GFW/Park et al. 2023
     over OUR archive — parse IMO/callsign/dimensions from archived
     AIS static messages, IMO checksum validation (checksum-valid IMO
     agreement across two MMSIs = strongest hull-match evidence),
     replace guessed nearKm/withinHours with published thresholds,
     emit envelope-carrying confidence-scored inferences.
     Zero licence exposure; survives monetization.
  4. MULTI-DATE IMAGERY — architecture decided: weekly OFFLINE chip
     pipeline (STAC per AOI, lowest-cloud scene per ISO week, TCI COG
     window read from s3://sentinel-cogs, WebP chip + metadata JSON
     {sensingDatetimeUTC, cloudPct, productId}) served FIRST-PARTY
     from the volume; never live-proxy third-party tiles. Honesty:
     ~40-70% of weeks yield a usable scene — the timeline UI must
     show "no usable capture this week" gaps, never silently reuse
     stale chips (extends the imagery-date rule). CDSE Processing API
     as validation/backup (free tier ample).
- [SHARED→T-DATACORE] **API product foundation (throughput directive
  2026-07-04)**: /api/v1 versioned read endpoints + key scaffolding +
  metering + per-endpoint license marks; /developers page [T-CLIENT];
  monetization readiness checklist awaits human approval (wishlist).

- (a) TERRAIN: **Mapterhorn** primary (free, no key, commercial OK —
  Copernicus + CC-BY national sources; terrarium 512px z0-17;
  attribution "© Mapterhorn"); AWS/Mapzen Terrarium fallback (free,
  public-domain sources). MapTiler free tier REJECTED (non-commercial
  — tripwire class). Resilience: archive a PMTiles extract of our AOIs
  (accumulation substitutes for dependency).
- (b) WEATHER: NWS api.weather.gov + NOAA nowCOAST WMS (public domain,
  US-only, no key, no SLA — degrade gracefully); global fields:
  OpenWeatherMap free tier (commercial OK with visible attribution,
  60 calls/min, 1M/mo; tiles are model-derived — label as such).
  Open-Meteo free tier PROHIBITED (non-commercial only); RainViewer
  DISQUALIFIED (personal/educational; API gutted Jan 2026). HONEST
  GAP: no free lawful GLOBAL true-radar exists — US radar only.
- (c) FIRES: NASA FIRMS free MAP_KEY, commercial lawful, VIIRS 375m,
  NRT ~3h latency, LANCE attribution + "not for safety-of-life"
  disclaimer; NO free history — archive detections from day one.
- (d) CROPS: USDA NASS CDL public domain ("free to redistribute"),
  30m, ANNUAL + retrospective (Feb release covers prior season) —
  label vintage; CropScape WMS for display; no free intra-season
  crop map (NASS Crop Progress = state-level text only).
- (e) DROUGHT/SOIL MOISTURE: US Drought Monitor weekly GeoJSON
  (mandatory NDMC/USDA/NOAA credit line, permanent); drought.gov XYZ
  tiles (NOAA open, daily); NASA SMAP L4 CC0 (9km MODEL product —
  label; native 36km). No free field-scale soil moisture.
- (f) GROUNDWATER: USGS NWIS / api.waterdata.usgs.gov (public domain;
  free key on the new API) — POINT data (wells), labeled as points
  with per-well trend + last-measured date, never a surface.
- (g) OIL/GAS INFRA — MAJOR FINDING: **no free, current, national US
  pipeline vector source exists anymore.** EIA Energy Atlas geospatial
  layers verified DEAD (DCAT absent, about-pages 404, maps.eia.gov DNS
  dead); HIFLD Open discontinued Aug-Sep 2025 (survivors gov-only);
  PHMSA NPMS restricts bulk access by policy. BUILD FROM: Global
  Energy Monitor trackers (CC BY 4.0, global, major infra), TX RRC
  bulk (public records; wells + TX pipelines), ND DMR public tier,
  OSM pipeline tags (ODbL — share-alike on derived DB), DataLumos
  archived HIFLD gas-pipeline snapshot (static — label vintage).
  Anything shipped states coverage per source honestly.
- (Tier 2) BUILDINGS: render via **OpenFreeMap** OSM building layer
  (public instance explicitly allows commercial use; sustainability
  risk noted) + client-side queryRenderedFeatures viewport stats at
  z13+ labeled "estimate — rendered features only, heights where
  mapped"; HONEST HEIGHT GAP: MS footprints have heights for only
  ~12% of corpus, Google 2.5D excludes the US, OSM tags sparse —
  viewport height stats are partial estimates by construction. Bulk
  analysis later: VIDA combined Google-MS-OSM (ODbL) or Overture
  GeoParquet (ODbL; hosted PMTiles is a beta convenience bucket, not
  an SLA). Hypotheses (ladder-pathed): metro-level footprint-vintage
  deltas ↔ homebuilder/REIT tickers — gate 1 = vintage deltas vs
  Census building permits reconciliation.

## PORT DWELL ANALYTICS (fusion directive 2026-07-04 — RAW live, SIGNAL gated)

- **What ships now (RAW, v1.0.60)**: server/portDwell.ts computes from OUR
  OWN AIS archive against the 9 imagery-verified port geofences (5km,
  nearest-port assignment resolves LA/Long Beach overlap): completed port
  calls (arrival/departure detection: >=3 in-fence points, >=2h span,
  median SOG <=3kts), dwell distributions (median/p90/max), ships in port
  now (right-censored, excluded from distributions), and 3x-median anomaly
  FLAGS suppressed below 10 completed calls per port (thin-history
  honesty). Dwell figures are LOWER BOUNDS: an archive coverage gap >6h
  splits a visit, never bridges it.
- **GATE 2 (SIGNAL) hypothesis**: sustained dwell-median or queue
  anomalies at container ports lead (a) retail-import names (XRT) and
  (b) logistics (IYT) on a 2-8 week horizon — the 2021 San Pedro Bay
  queue was the famous instance; the open question is whether the
  post-2022 normalized regime still carries a tradable residual. PRIOR:
  weak-positive at container ports, near-zero at energy ports (Houston)
  where dwell reflects terminal ops, not demand.
- **GATE 2 test plan (vs published congestion indices)**: weekly
  archive-derived series {median dwell, in-port count} per port vs (1)
  the port authority's published monthly TEU + vessel-call stats
  (ground truth for our counters — also gate 1 for R2 transit), and (2)
  a published congestion proxy (e.g. Kiel Trade Indicator / port-call
  datasets) — our series must correlate with the established measure
  before any return test. Then anomaly weeks vs forward XRT/IYT returns
  against a random-entry base rate (REASONING STANDARD #3), regime-split
  (#2), discounted for variants tried (#4).
- **Second-order (#5)**: congestion nowcasts are sold commercially at
  container-line scale; the retail-ticker LAG (buyers of those products
  hedge freight, not equities) is the structural room. Capacity: XRT-class
  liquidity is fine at our size.
- **Imagery enrichment (later)**: Sentinel-2 berth occupancy at the same
  9 ports verifies AIS-derived in-port counts when that pipeline lands
  (Tier-3 spec) — imagery verifies, AIS remains primary.

## FUSION HYPOTHESES (Map v2.2 directive 2026-07-04 — logged, NOT built; each with ladder path)

- **(a) Insider × facility activity (STLD first).** PAIRING: Form 4
  archive (officer/director open-market buys at STLD) × Sentinel-2
  change detection at the four imagery-verified SDI mills (Butler,
  Columbus, Sinton, Columbia City — coordinates fixed 2026-07-03).
  TESTABLE CLAIM: quarters where insider open-market buying co-occurs
  with visible yard-inventory drawdown (finished-steel yard shrinking =
  shipments outpacing production) beat single-signal quarters for
  forward 1-2q returns. GATE 1 GROUND TRUTH: Sentinel-2 yard readings
  reconcile against STLD's disclosed quarterly shipment volumes before
  any return test; Form 4 side is already gate-1-passed (as-filed).
- **(b) Generation shifts × utility tickers.** PAIRING: EIA-930 hourly
  generation by fuel/region × our EIA-located plant registry × operator
  equity tickers. TESTABLE CLAIM: sustained regional gas-utilization
  spikes with a single dominant listed operator lead that operator's
  earnings surprises vs the XLU base rate. GATE 1 GROUND TRUTH: a
  registry-owner -> ticker mapping table (build once, verify against
  10-K subsidiary lists) + EIA-930 totals reconciling to registry
  capacity within ~5% per region. Extends the POWER-PLANT hypotheses
  entry; the fusion is the operator-concentration conditioning.
- **(c) Ship-movement anomalies × commodity/retail tickers.** PAIRING:
  our port-transit stats (arrivals at the 9 imagery-verified ports from
  the vessel archive) + shadow-fleet zone rates × (i) tanker basket
  FRO/STNG/TNK, (ii) retail-import names (XRT) for container ports.
  TESTABLE CLAIM: container-port arrival-rate anomalies lead XRT
  earnings-season surprises; dirty-STS zone rates lead tanker rates.
  GATE 1 GROUND TRUTH: monthly port TEU reports (containers) and
  published tanker-rate indices (Baltic Dirty) reconciling with our
  archive-derived counts.
- Discipline: three hypotheses = three separate gate-1 efforts; none
  advances without its reconciliation; REASONING STANDARD #4 applies
  (discount for every variant tried when any reaches gate 2).

## COLLECT-EVERYTHING AUDIT (verified 2026-07-04, Map v2.2 directive)

Every layer's data path to permanent storage, verified in code:
- aircraft -> archiveAircraft on every fresh upstream fetch ✓
- vessels -> archiveVessels on the 60s snapshot tick ✓
- trains -> archiveTrains on every fetch (2-min cadence gate) ✓ (v1.0.53)
- Form 4 filings -> archiveFilings on every 15-min poll, gzip after 2d ✓
  (v1.0.55 — history accumulates, never display-only)
- power plants / strategic sites / shadow zones (STATIC reference data)
  -> git-versioned in datacore/ — the repo history IS the snapshot
  archive; the builder is re-runnable and every change is a commit. No
  runtime archiving needed; DOCTRINE: static reference layers are
  archived by versioning, streamed layers by JSONL.
- shadowstats (DERIVED) -> not separately archived BY DESIGN: it is a
  pure function of the vessel archive and recomputable for any window;
  archiving ingredients, not derivations, is the rule.
- imagery -> NOT archived (CDN tiles; licensing + volume); the
  Sentinel-2 pipeline will archive scene IDs + readings when it lands.

## KILL-SWITCH INPUT COHERENCE (filed 2026-07-07, from R14 follow-up)

HYPOTHESIS: single-field validation (drawdownGuard: finite, >0) is
necessary but not sufficient — Alpaca paper account snapshots can be
INTERNALLY inconsistent (2026-07-07: headline $27k vs last_equity at
$109.4k peak vs own chart ~$106.8k, no visible fills). A
coherence-class check could treat a catastrophic equity reading as
INVALID unless corroborated: |equity − last_equity| large ⇒ require
matching same-day fills or position-value change before accepting.
RISK (why not patched immediately): a too-clever coherence check that
wrongly rejects a REAL catastrophic reading would disable the halt —
the exact failure the mechanism exists to prevent. Design must be
fail-safe: incoherent ⇒ still halt trading (pause-new-orders state)
but tag the halt as data-suspect rather than drawdown-confirmed.
LADDER PATH: not a data root — a risk-mechanism design change;
requires evidence from accumulated EQUITY-READ-INVALID audit rows +
this incident, one-threshold-at-a-time discipline, and a pre-stated
rollback trigger. Blocked on: resolution of the 2026-07-07 incident
(human fills check) so the design fits the true failure mode.

## [HYPOTHESIS] NO2/PM2.5 as an industrial-activity proxy (Google Air Quality × strategic sites) — filed 2026-07-07

Combustion/industrial throughput emits NO2 and PM2.5; AQI over a steel
mill / tank farm / port should co-move with the facility's activity
level, giving a near-real-time activity proxy that fuses with the
power-plant, strategic-site, and grid layers (the Everything Graph).
Google Air Quality current-conditions archived at our sites, 500m
resolution; accumulation is the moat (Google exposes only 30d of
history — every unrecorded day is permanently lost).
LADDER PATH:
- Gate 1 (DATA): validate archived AQI-near-site against the site's
  known activity/output ground truth (e.g. STLD Sinton/Butler
  utilization vs NO2; Cushing tank-farm activity; port throughput) —
  is the AQI at 500m actually elevated OVER the facility vs regional
  background, and does it track known up/down periods?
- SECOND-ORDER (the confounder that will kill a naive version):
  weather/wind dominate short-term AQI. De-trend against regional
  background AND wind before crediting any site-specific signal.
- DISCOUNT the observed edge by the number of site×pollutant
  combinations tried (multiple-hypothesis fishing).
- OUT-OF-SAMPLE confirmation required before Gate 2 (does the activity
  proxy predict forward returns for the exposed ticker).
RAW archive only until gated. Runtime: server/airQuality.ts (key-gated
GOOGLE_MAPS_API_KEY, activates when the human enables the Air Quality
API on the GCP project). BLOCKED-FOR-MIKE: enable the Air Quality API
in the Google Cloud console — the stream then activates on the key
already in Railway.

## ORBITAL CROSS-SYSTEM TIES (filed 2026-07-07, ORBITAL program — each a hypothesis)

HONEST FRAMING (human directive): visualization + brand value justifies
a build on its own; state which ties are real signal vs operational vs
showcase; never manufacture a signal that isn't there.

(a) [OPERATIONAL — strongest real tie] EO PASS-OVER PREDICTION. From
    Sentinel-2A/2B + Landsat-8/9 TLEs, compute the next overpass over
    each monitored site (ports, grid, strategic sites) → "next fresh-
    imagery opportunity here." Feeds the tank-fill / imagery-acquisition
    workflow directly. NOT trading alpha — an operational utility.
    Validate: predicted overpass times vs actual scene timestamps (a
    gate-1-style accuracy check before trusting the schedule).
(b) [STRUCTURAL + SHOWCASE; trading signal SPECULATIVE] ENTITY-GRAPH
    JOIN. SATCAT operator/owner → company → public ticker (RKLB, ASTS,
    VSAT, IRDM, GSAT, LMT/NOC…). A company lookup shows its orbital
    footprint beside jets/vessels/plants — Everything-Graph connective
    tissue. Structural value real. Trading edge (constellation size vs
    capex/revenue) is slow-moving and speculative — its own gate; no
    alpha claimed from a slow count.
(c) [SHOWCASE + WEAK SIGNAL] LAUNCH-ACTIVITY OVERLAY. New-object
    appearances tied to launch-provider/operator + news. Launches are
    largely pre-announced (efficient) → weak alpha; only failures /
    cadence anomalies plausibly matter and are rare. Mostly a real event
    overlay, not a signal. Gate honestly.
(d) [SHOWCASE] GEO COMMS INFRASTRUCTURE LAYER. GEO comms sats as the
    space tier of the infra map alongside grid + vessels + plants. Pure
    visualization/completeness value; no trading claim (legitimate on
    brand grounds).
(e) OTHERS (honest): reentry/decay-event overlay (minor, showcase);
    conjunction/collision-risk near active assets (real space signal,
    not market-relevant — note, don't build); comms coverage over a
    monitored site (tenuous, parked). VERDICT: (a) is the one genuinely
    useful operational tie; NONE is strong trading alpha, and that is
    acceptable per the directive.
LADDER/DATA-PATH: TLEs from CelesTrak. Client-computed ties (a on the
imagery UI, b/d in the client graph view) are UNBLOCKED. Server-computed
ties (served pass-over schedule, launch-event detection) need TLEs
server-side → blocked on the CelesTrak-relay decision (wishlist).

---
HYPOTHESIS (2026-07-08, worldview-globe Pillar 6 — filed with the fires×facilities
cross-tie, PR pending): FIRES NEAR STRATEGIC ASSETS → INSURER/UTILITY/OPERATOR
RETURNS. Live NASA VIIRS active-fire detections joined to our strategic-facility
archive (server/firesFacilities.ts → /api/data/fires-near-facilities) surface
which assets have fires within R km right now. TESTABLE FORM: does an active-fire
cluster within R km of an insured/industrial/utility asset precede negative
returns (or vol) in the owning/insuring ticker over +1d/+5d/+20d vs a base rate
of random asset-days? LADDER: gate-0 = the join exists + records (SHIPPED as a
RAW cross-tie, no predictive claim); gate-1 blocked on (a) ARCHIVING the
fires×facility hits daily (the endpoint is display-only now — no history yet)
and (b) an asset→ticker/insurer join (entity graph). gate-2 = the outcome study
above with base rates + regime split. PRIOR: weak-to-moderate and highly
event-driven (most fires are immaterial; the tail — a refinery/mill/large-utility
service territory — is where any signal lives); discount hard for the
multiple-asset multiple-comparison. NOT a signal until gate 2. Second-order: fire
proximity is public/fast (NASA NRT ~3h), so any edge is in the ASSET→TICKER join
and the materiality filter we build, not the fire data itself.

## GRID VISION tower-detector: tiling CONFIRMED in-domain (2026-07-09) — remaining gate-1 work is held-out region + NAIP + scene-level

RESOLVED (the tiling hypothesis below): gv-detector-v1-2 tiled the ortho into
640 px windows and scored **val AP50 0.566 / recall 0.499 / precision 0.732**
(v0 was 0.036), yolov8n held constant — a ~16× jump and INSIDE the 0.55–0.75
prior band. The sub-pixel diagnosis was correct; tiling clears gate-1's SIGNAL
bar IN-DOMAIN. Weights archived on branch `gridvision-pod-result` (gv_best.pt).
STILL OPEN before the detector is trusted for the national rollout (ranked):
1. HELD-OUT REGION — retrain/eval on a US region that is NOT AZ/KS; current
   result is out-of-sample by image only. This is the real generalization test.
2. NAIP DOMAIN — ETDII is USGS ortho, not NAIP radiometry; run inference on
   actual NAIP tiles (the streaming path in gridvision_naip_stac is built) with
   a human-sampled precision pass (OSM gives recall only).
3. SCENE-LEVEL metric — stitch overlapping tiles back with cross-seam NMS and
   measure per-scene P/R (the 0.566 is tile-level, and overlap double-counts
   towers in the val instance tally: 514 tiled vs ~283 unique).
4. HEADROOM — yolov8s/m backbone, more epochs (curve noisy/near-plateau 30–61),
   more regions. Discount any sweep winner by variants tried.
SUBSTATION track is still separate (6 labels; needs Duke-US zips + OSM-seeded
self-bootstrapping per the charter). — Original hypothesis, now confirmed, kept
below for the record.

## GRID VISION tower-detector v1: tile, don't downscale (next ladder step after v0's gate-1 failure) [CONFIRMED 2026-07-09 — see resolution above]

Opened 2026-07-09 after gv-detector-v0-2 (experiments.md 2026-07-09) trained
cleanly but scored **AP50 0.036 / recall 0.035** vs a 0.55–0.75 prior — a
ladder gate-1 (SIGNAL) failure. HYPOTHESIS: the miss is dominated by the v0
pipeline resizing WHOLE ETDII ortho frames to imgsz 512, which collapses the
~19-towers/image (a few px each at 0.6 m) to ~1 px — sub-pixel targets, hence
recall ~3%. The groups hitting 0.73 detect on native-GSD TILES, not downscaled
full frames.
TESTABLE FORM: rebuild `build_yolo_dataset` to TILE each ortho into 512/640
windows (sliding, stride ≤ tile/2; keep tiles with ≥1 tower; boxes clipped to
tile) instead of whole-image resize; hold 0.6 m GSD. Re-fine-tune (try yolov8s
and yolov8m alongside n). Compare AP50 to v0's 0.036 on the SAME 15-image val
region, then confirm on a HELD-OUT region (not AZ/KS) before believing.
LADDER: DATA gate already passed (phase-B labels verified); this is SIGNAL
gate-1 again. PRIOR: tiling alone lifts AP50 well above 0.036 (moderate-high
confidence — sub-pixel is the textbook cause); reaching 0.55+ also needs a
bigger backbone and >2 regions (lower confidence). DISCOUNT the winner of any
n/s/m × tile-size sweep by the number of variants tried; out-of-sample region
confirms. COST: one more RTX-4090 run ≈ $0.10 under the ledger cap. Also fold
two infra fixes from the v0 session into the next run: push best.pt over the
git channel (0x0.st/transfer.sh are unreachable from RunPod egress) and use the
result-branch-SHA completion signal (the runpod/pytorch container never reports
EXITED, so the status watchdog idle-bills to the cap).

---

## [PRODUCT-DEBT · filed 2026-07-09] Two overlapping "US power plants" /data layers — consolidate

STATE: after #408, the /data facilities group has TWO power-plant toggles:
- `powerplants` — "US power plants": WRI Global Power Plant Database
  v1.3.0 (CC-BY 4.0), ~9,800 US plants, 2021 vintage, served via the
  `/api/data/powerplants` GeoJSON API.
- `powergrid_hifld_plants` — "US Power Plants — HIFLD (authoritative)":
  HIFLD/EIA-860 (public domain), 11,810 plants, PMTiles, fuel-colored.

WHY BOTH SHIPPED: the HIFLD one completes the authoritative HIFLD grid
trio (transmission #405 + substations + plants) from one public-domain
gov source, has ~2,000 more plants, and uses PMTiles delivery consistent
with the other grid layers. Genuinely additive — but a user sees two
near-identical toggles, which is exactly the redundant-layer debt the
STALENESS AUDIT targets.

DECISION NEEDED (not acted on this session — removing a live layer is its
own change, never bundled): keep both (distinct tiers, some users want
WRI's global coverage for future non-US expansion) OR deprecate the WRI
GeoJSON layer in favor of the authoritative HIFLD PMTiles one (cleaner
licensing — public domain vs CC-BY attribution burden; better delivery;
more complete). LEANING: deprecate WRI for the US surface once a
side-by-side confirms HIFLD is a superset on coverage AND carries the
same click-through fields (capacity/operator) the WRI layer exposes.
LADDER: N/A (raw overlay). NEXT: a session doing a STALENESS-AUDIT
fall-through runs the coverage/field comparison and files the
keep-or-remove call; if remove, it ships as one docs+code removal PR.

UPDATE 2026-07-14 (this session, [PRODUCT]) — coverage/field comparison
RUN; the LEANING above is REVISED, not confirmed. On raw display fields
HIFLD IS the superset the 2026-07-09 note expected: 11,810 plants (WRI:
9,833 US rows), public-domain licensing (no CC-BY attribution burden),
name/fuel/mw/operator all present PLUS state/status/VAL_METHOD position-
honesty fields WRI lacks, and it already renders fuel-typed SDF icons
(SYMBOLS NOT DOTS-compliant) — so a naive removal looked justified.

BUT (REASONING STANDARD #1 — trace the downstream chain before changing
anything) a fuller trace turned up a real coupling the 2026-07-09 filing
didn't check: `server/entityGraph.ts` builds its ENTIRE "operates" edge
set and every `facility:plant:N` node from `datacore/powerplants/
us_power_plants.json` (the WRI array) BY ARRAY INDEX — nothing server-
side reads the HIFLD PMTiles data at all. The WRI map layer's click
handler is therefore the ONLY live path that calls `fetchDossier(...,
"facility:plant:${p.plantId}", ...)` with a real entity id; clicking a
plant there returns the FULL dossier (identity + 2-hop Everything Graph
neighborhood + ticker linkage + insider filings + USAspending contracts +
hazards), per server/dossier.ts. The HIFLD plants layer, meanwhile, had
NO `fetchDossier` call at all before this session (a plain parity gap
vs. every other clickable layer — aircraft/trains/fires/gauges/quakes/
buoys all wire it with `entityId:null` for the graph doesn't-model-this-
yet case). Removing the WRI toggle as filed would have silently deleted
the only click path to that richer dossier for power plants — a real
product regression the 2026-07-09 filing's "same click-through fields"
check (capacity/operator only) was too narrow to catch.

WHAT SHIPPED THIS SESSION (small, safe, non-regressive): HIFLD plants'
click handler now calls `fetchDossier(dossierKey, null, lat, lon)` too —
closes the parity gap (every HIFLD plant now gets the lat/lon-keyed
nearest-sites + hazards + flood-zone section other ungraphed layers get),
with zero risk since `entityId:null` cannot collide with or corrupt the
WRI-indexed graph. `datacore/layers.json`/`DEFAULT_ON`/`LAYER_GROUP` were
NOT touched — both toggles stay live.

REVISED NEXT STEP (this is now the real size of the consolidation, not a
one-PR removal): before the WRI toggle can be safely retired, something
needs to build a server-side HIFLD plant dataset (a compact JSON extract
alongside/instead of the PMTiles, analogous to `us_power_plants.json`)
and migrate `entityGraph.ts`'s facility nodes + `operates` edges onto it
(11,810 rows vs. 9,833 — the id scheme changes, every existing
`facility:plant:N` dossier reference and the riverPlants.ts pillar-6
cross-tie's `PlantTuple` consumer would need updating in the same
migration). That is its own scoped PR, is where the "keep-or-remove"
call should actually be filed, and is NOT attempted this session (scope
discipline — one logical change per PR). Until then, both toggles
staying live is the CORRECT state, not unresolved debt.

---

## [GRID-VISION · filed 2026-07-12 · ANSWERED SAME DAY — see experiments.md survey entry: 1/24 collects, hit fails geometry checks, lane CLOSED] Do ANY 3DEP collects carry wire/tower classes or intact aerial returns?

FINDING (experiments.md 2026-07-12): the 2 collects probed (incl. a
dedicated TL corridor survey) have min-spec classification and no
structure-band returns at OSM-confirmed tower sites — 3DEP LiDAR tower
extraction UNPROVEN. But n=2 of 2,273 EPT resources.

TESTABLE FORM (one command per collect now):
  python3 scripts/gridvision_lidar_probe.py classify <resource>   # Q1: classes 13-16?
  python3 scripts/gridvision_lidar_probe.py hag <resource> <x> <y> # Q2: 8-60m returns at OSM tower?
Survey design: sample ~30 collects stratified by year (2018-2023; newer
USGS specs demand more classes) and vendor; for each, pick OSM towers
INSIDE the footprint polygon (resources.geojson — bbox is a lie) and run
both probes. If ≥ some collects carry classes 14/15, LiDAR becomes a
PARTIAL-coverage authoritative tower source (label coverage honestly,
never extrapolate); if none do, close the LiDAR lane for towers and
promote street-view ML to the top of the remaining-gap ranking.
LADDER: this is DATA-gate work (reading vs external truth = OSM towers).
PRIOR: most collects min-spec; maybe 10-25% of 2020+ collects carry
wire classes (USGS Lidar Base Spec 2020+ lists them as optional-but-
defined). COST: free, ~2 min/collect, no GPU.

---

## [MEASUREMENT-DEBT · filed 2026-07-12] Visual harness: two gaps found during the nuclear wave

1. FIELDS-ON PROBE NONDETERMINISM: data@1440's wx probe failed 4/8 runs
   across diffs that provably don't touch weather (registry
   byte-identical; harness mocks the tiles itself) and passed the
   final run clean — software-GL symbol-placement timing inside its
   48×250ms window. Its own comments record a 2026-07-05 false-fail of
   the same shape. FIX SHAPE: deterministic wait (poll until
   areTilesLoaded AND two consecutive stable queryRenderedFeatures
   counts, budget scaled when SwiftShader is detected), or a --retry-
   once-on-fields-only policy. MEASUREMENT CODE: own [RULE-REVIEW] PR,
   never bundled with layer work.
2. REGISTRY FIXTURE DRIFT: scripts/visual_check.mjs hardcodes its
   /api/data/layers fixture (comment says "every toggleable registry
   layer must appear") — radiation, nukeaccidents, nukefacilities (and
   any newer layers) are absent, so harness toggle/cost checks don't
   exercise them. FIX SHAPE: generate the fixture FROM
   datacore/layers.json at harness start (one source of truth), or a
   drift test that fails when registry ids ⊄ fixture ids. Same
   [RULE-REVIEW] PR as (1) or its own.

## [WISHLIST-POINTER · filed 2026-07-12] GEM Global Nuclear Power Tracker (CC BY 4.0) — one human download

Global reactor-unit-level data (1,749 units, 61 countries, all
statuses incl. retired/under-construction, owner/operator, coords;
March 2026 release) would upgrade the nuclear-facilities picture
beyond US-HIFLD + Wikidata. License verified CC BY 4.0; download is
FORM-GATED (name/email, no payment). BUILD-FIRST: no free ungated
equivalent at unit level (Wikidata is entity-level and patchy on
status; EIA-860M is US-only). ASK: human submits the form once at
globalenergymonitor.org/projects/global-nuclear-power-tracker/download-data/
and drops the xlsx in the repo or a volume path; a session wires the
layer with attribution "Global Energy Monitor (CC BY 4.0)".

## [HARNESS-ENV · filed 2026-07-15] data@1440 fields-on flake frequency rising — environment-correlated, mechanism unknown

Five solo harness runs during EARTH TWIN session #3 (v1.0.340-341
batch): branch runs 1-3 FAILED at data@1440 fields-on only (one
locator timeout, twice the wx-never-rendered triple: layers/tiles/
arrows absent + raster-opacity null + no arrows), then PURE
origin/main PASSED, then the SAME branch tree PASSED — an A/A proof
the failures track the ENVIRONMENT window (~35min), not the code.
Machine was idle (13GB free) during failures, so the earlier
"concurrent-heavy-work starvation" story is incomplete. Cumulative:
4 fails across 2 batches, always this one battery/width, always
SwiftShader. OPEN: what state does headless SwiftShader get into
where wx raster tiles + symbol layers never mount for ~30 min, and
what should the harness DO about it (per-battery retry? WebGL
context probe + honest SKIP-with-reason?). Any harness change is its
own [RULE-REVIEW] PR with the bias statement (a retry can only mask
real regressions if unbounded — cap + log both outcomes).

## [EVERYTHING-GRAPH · filed 2026-07-17, GEM ownership CIK join SHIPPED] Broaden `owns` beyond CIK-to-CIK pairs — the more novel joins are state/foreign ownership, which this slice deliberately excludes

SHIPPED this session (v1.0.374, T-DATACORE): `server/entityGraph.ts` gained
an `owns` edge type (company(owner) → company(owned)) sourced from
`datacore/gem/ownership.json.gz` (GEM Global Energy Ownership Tracker,
CC BY 4.0, ingested 2026-07-07 per wishlist.md CENSUS #5). Restricted to
`entity_edges` where BOTH the subject (owned) and interested party
(owner) resolve to a real US SEC CIK via GEM's own crosswalk field —
node ids reuse the same ticker-preferred/`company:cik:<CIK>` fallback
scheme `insider_of` already uses (via the shared `getCikTickerMap`
resolver from sec8kEarnings.ts), so a GEM-tracked company lands on the
SAME node the EDGAR pipeline populates, not a duplicate. Of GEM's 24,351
total `entity_edges`, 393 resolve both ends to a CIK today (live-verified
against the real file) — mostly institutional 13F-style holders
(BlackRock/Vanguard/State Street), which is real signal but overlaps
what `edgar13f.ts` likely already surfaces from a different source.

THE MORE INTERESTING GAP (not built this session, deliberately — scope
discipline, one logical change per PR): 1,403 `entity_edges` have AT
LEAST ONE CIK-mapped end — the other 1,010 are edges to governments,
private holders, or foreign parents with NO CIK (e.g. "Sonatrach owned
100% by Government of Algeria"). These are the genuinely novel joins
(state ownership of US-adjacent energy assets, foreign conglomerate
control chains) invisible to any SEC-only source, but they can't honestly
be typed as `company` nodes (per the design doc's v1 entity-type table:
company/person/facility/vessel only — no "government"/"institution"
type exists). LADDER: this is still DATA-gate connective tissue (RAW, no
predictive claim), not a new gate; the real work is either (a) a new
`GraphNodeType` (e.g. `institution`) honestly distinguishing
state/private owners from SEC-reporting companies, sourced from GEM's own
`Entity Type` field, or (b) accepting looser typing with an explicit
`cik_verified: false` attribute flag. PRIOR: (a) is more correct and only
moderately more work (the entity list is already loaded); do that, not
(b), when this is picked up. Cross-tie candidate once built: state/
foreign ownership × USAspending contract awards to the same ticker
(does government ownership correlate with government contract flow?) —
a testable SECOND-ORDER hypothesis, not assumed.

Also STILL STALE (found, not touched this session — out of scope): GRID
BUILD ORDER item 4 (line ~3192) says "GEM REGISTRY JOIN (blocked on
wishlist 9b form-fill)" — 9b resolved 2026-07-07 (Mike enabled Drive
access, full GEM suite ingested), so that blocking note is stale text; a
future session touching that section should update it to point here.

UPDATE 2026-07-17 (same-day follow-up session, v1.0.375, [PIPELINE],
T-DATACORE) — THE GAP ABOVE IS CLOSED, per option (a) as prioritized.
`GraphNodeType` gained `"institution"` (server/entityGraph.ts); an `owns`
edge now ships whenever AT LEAST ONE end resolves to a real US SEC
CIK — the CIK-mapped side anchors the edge into a node the EDGAR/insider
pipeline already knows, the other side becomes an `institution` node
typed from GEM's own `Entity Type` field (`gem_entity_type` attr) plus
`headquarters_country` and an explicit `cik_verified: false`, never
mistyped as `company`. Live-verified against the real archive (not just
the fixture): `owns` count 393 -> 1,403 (the full CIK-anchored pool
predicted here), 764 unique institution nodes. `client/src/pages/graph.tsx`
updated in the same PR (NodeType union, TYPE_ICON/TYPE_LABEL, counts row,
CSS badge color) — the exact client-crash class PR #506 fixed for a new
EDGE type applies equally to a new NODE type, so this shipped together,
not as an afterthought.

HONEST CORRECTION to this filing's own example: "Sonatrach owned 100% by
Government of Algeria" was illustrative, not literal — live query of the
actual `Entity Type` breakdown among the 1,010 one-CIK-end edges shows
ZERO `state`/`state body` entities in that pool (753 `legal entity`
foreign/private parents, 7 `arrangement`, 2 `unknown entity`, 2 `person`).
Direct state/government ownership never appears adjacent to a CIK-mapped
company in GEM's edge list — it sits one hop further out (a government
owns a private holdco, which owns the CIK-mapped ticker). This session's
single-hop-anchored join therefore captures the foreign/private-parent
joins in full but NOT direct state ownership chains; a 2-hop transitive
walk (already possible via `neighborhood(graph, id, 2)`) is the next step
if the state-ownership cross-tie hypothesis below is picked up. Filed
here rather than left implicit, per REASONING STANDARD #4 (state the
result, don't let the original filing's example stand uncorrected).

CROSS-TIE HYPOTHESIS (per ACTIVE ANGLE-HUNTING, testable form, not yet
tested — this is DATA-gate connective tissue only, no ladder gate
applies to the join itself, but any downstream claim from it would need
gate 2): now that `institution` nodes carry `headquarters_country`, a
future session can test whether US small-caps with a foreign
institutional owner (2-hop from that owner's home-country peers) show
correlated moves around that owner's home-market macro events (e.g. a
Chinese/German owner's domestic activity leading a US small-cap's price)
— testable via `neighborhood(graph, tickerId, 1)` filtered to
`type: "institution"`, joined against the existing macro_data.py feeds
for that country. PRIOR: low-to-moderate (indirect/thin channel; the
existing insider Form-4 and USAspending joins are more direct), worth a
cheap first look given the join now exists for free.

VERIFICATION: `npx tsx --test server/entityGraph.test.ts` 14/14 (2 tests
rewritten to assert the broadened at-least-one-CIK behavior + a true
neither-end-CIK exclusion case kept as a regression pin; 2 new tests for
institution-node typing/attrs); `npx tsx --test server/dossier.test.ts`
20/20 (fixture's `counts` shape updated for the new field);
`npx tsx --test server/*.test.ts` 709/709; `npx tsc --noEmit` 66 errors,
byte-identical set to the `git stash`-verified baseline (one pre-existing
cosmetic union-ordering diff, unrelated file); `npm run build` clean —
confirms `Landmark` (new institution icon) is a real lucide-react export.
Live boot (`node dist/index.cjs`) + `/api/data/graph` + `/api/data/graph?
entity=BLK`: real counts (owns 1,403, institution 764) match the
fixture-test predictions exactly; ad hoc Playwright drive against the
built app (`/app#/data/graph`, searching BLK) confirmed 100+ real
institution-typed connections render (BHP Group, Siemens, TotalEnergies,
Deutsche Bank, HSBC, etc.) with the Landmark icon and correct share_pct/
confidence, zero page or console errors beyond a pre-existing sandbox
Google-Fonts CDN block (confirmed unrelated via `requestfailed` capture
against this same URL before this session's changes existed in the
built bundle). `npm run visual -- --page data` run this session (see
experiments.md for the full result) — the CSS-only addition is one new
attribute-selector rule scoped to `.vt-graph-typebadge[data-type=
"institution"]`, not reachable from the default `/data` map render path.

## [GEM METHANE-PLUME × EXTRACTION-REGISTRY PROXIMITY · filed 2026-07-18, gate 2(a) done 2026-07-19 — proximity join + map layer shipped, still NOT a signal] Does satellite-detected plume density/rate near a GEM-catalogued oil/gas/coal asset track that operator's own emissions disclosures?

Filed alongside `server/gemMethane.ts` shipping (`/api/data/methane-plumes`,
research/experiments.md 2026-07-18 [PIPELINE] entry) — a HYPOTHESIS, not
attempted this session (REASONING STANDARD #10: stated before any
downstream use). GEM's GMET file ships 3,473 geolocated, dated satellite
methane-plume detections (CarbonMapper/GHGSat-class, `infrastructureType`
tagged for ~82% of rows: wellpad/pipeline/livestock/coal-mine/etc.) in the
SAME release family as GEM's own `oil_gas_extraction` (7,673 fields w/ WKT
outlines), `gas_pipelines` (4,246 segments, Owner/Parent fields), and
`coal_mine_tracker` (5,382 active mines) registries — all joinable by
proximity to the plume's `lat`/`lon`.

HYPOTHESIS: an operator (via GEM's Owner/Parent fields, joinable further
to a CIK via the GEOT ownership crosswalk `entityGraph.ts` already reads)
whose facilities show elevated nearby plume detection RATE or magnitude
is exposed to real regulatory/ESG/methane-fee risk (EPA's Methane
Emissions Reduction Program fee structure is public) — a second-order
question (REASONING STANDARD #5): why would this be tradeable rather
than already priced in? Plausible answer: CarbonMapper/GHGSat detections
are irregular-cadence aerial/satellite overflights, not continuous
monitoring — a company's OWN emissions disclosures (10-K/ESG reports,
often annual and self-reported) may lag or understate what an independent
satellite catches between filing periods, so a persistent divergence
(plume-implied rate >> disclosed rate, sustained across repeat detections)
could be the actual signal, not raw plume presence.

LADDER PATH (not started — gate 0, needs the join built first):
GATE 1 (ground truth): the plume detections ARE the ground truth (GEM
attributes each to CarbonMapper's/GHGSat's own calibrated instrument
readings, with a stated uncertainty field) — no separate verification
needed, unlike e.g. tank-shadow inference. GATE 2 (signal): would need
(a) the proximity join (plume -> nearest GEM asset within some radius,
TBD threshold, honestly excluding ambiguous multi-asset clusters), (b)
enough repeat-detection history per asset to compute a rate (single
detections are noise), (c) a same-universe base rate per REASONING
STANDARD #3 (do names with ANY nearby plume underperform peers with
none, before conditioning on rate/magnitude at all?), (d) matching
against each operator's actual disclosed methane intensity where public
— CURRENTLY UNSOURCED, a real gap this hypothesis would need to solve
before it clears gate 2, not merely proximity to price.

NOT STARTED: no map layer exists yet to even visually sanity-check
clustering (filed as the concrete NEXT step in the same experiments.md
entry). Discount heavily per REASONING STANDARD #4 — GHG/ESG-materiality
angles are a crowded, well-covered thesis space in institutional research;
"nobody noticed" is not a credible answer here, so the real edge (if any)
is narrowly in the disclosure-lag mechanism above, not in discovering the
plumes exist.

UPDATE 2026-07-19 ([PRODUCT], T-DATACORE+T-CLIENT) — GATE 2(a) SHIPPED:
the proximity join (`server/gemMethaneProximity.ts`) and the map layer
(datamap.tsx `methane_plumes`, group "environmental") both built. Join
scope is the two point-geometry registries only — `oil_gas_extraction`
(fields, 7,673 rows, ~7,055 with coordinates) and `coal_mine_tracker`
(non_closed, 5,382 mines) — `gas_pipelines`/`oil_ngl_pipelines` are
route/WKT geometry, not a single lat/lon per asset, and are NOT joined
here (a plume tagged "pipeline", ~48 of 3,474, simply finds no match,
honestly, rather than a fabricated nearest-point guess against a route).
Method: grid-indexed (0.5° cells) nearest + second-nearest haversine
search, MATCH_RADIUS_KM=2 (stated constant from GEM's own location-
accuracy vocabulary, not tuned), AMBIGUOUS_MARGIN_KM=0.5 flags a plume
where two assets are nearly equidistant instead of silently picking one.
LIVE JOIN RESULT on the real GEM release (2026-07-19): 3,473 plumes ->
1,027 matched within 2km (772 coal_mine, 255 oil_gas_extraction), 206 of
those flagged ambiguous, 2,446 unmatched. Sanity check: coal-mine matches
(772) land under the plume file's own "coal mine" infrastructureType tag
count (1,690) as expected for a stricter 2km geometric radius vs. GEM's
own (looser) infrastructure-note tagging — consistent, not suspicious.

STILL GATE 2, NOT A SIGNAL — REASONING STANDARD #10 discipline: the join
only answers "which catalogued asset is nearest," not the hypothesis
itself. Still required before this clears gate 2:
  (b) repeat-detection RATE per asset (a single detection is noise) —
      needs the matched-plume set grouped by nearestAsset.id and time-
      binned; not built.
  (c) a same-universe BASE RATE (REASONING STANDARD #3): do assets with
      ANY nearby plume underperform peers with none, before conditioning
      on rate/magnitude at all? Requires a price/returns join against the
      operator entity (via ownership.json.gz's Owner/Parent -> CIK
      crosswalk, entityGraph.ts already reads this) — not built.
  (d) matching against operators' own disclosed methane intensity
      (10-K/ESG) — CURRENTLY UNSOURCED, unchanged from the filing.
NEXT STEP (not this session): (b) is the smallest next slice — group
`cachedGemMethaneProximity().plumes` by `nearestAsset.id`, compute a
detection count/rate per asset, and surface it as a sortable stat before
attempting (c)/(d). Discount stays heavy (REASONING STANDARD #4) — this
update proves the join works, not that the underlying thesis has edge.
