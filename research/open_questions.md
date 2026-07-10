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

3. **CSP execution cascade** (per CHANGELOG 2026-05-22): CSP trades were
   failing on three modes — insufficient capital for high-priced
   underlyings, no suitable puts in chain, liquidity filters rejecting
   everything. Fixes were applied (dynamic price ceiling etc.) — VERIFY
   in current audit logs that Tier 2 CSP trades actually fire now. If
   still zero fills, the fix pack didn't take.

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

15. **[PARTIALLY FIXED 2026-07-09, R19, v1.0.260 — root cause; live
    verification still pending] Real-time position-exit monitor (WS
    stream) was silently non-functional for at least 22+ hours.** Full
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
    silent). STILL OPEN: this session could not deploy or observe the
    fix live. NEXT: the first session after this deploys should query
    `/api/diag/audit?type=STREAM&limit=5` for a "Real-time feed live —"
    entry (must appear now) and `?type=WS-EXIT&limit=5` (should go
    non-empty once any tracked position's stop/target is hit), then
    `/api/diag/ml` for `feedback_live_count` > 0 (was stuck at exactly 0
    for 3+ days despite the D1/D2 fix in KNOWN BROKEN #12). If STREAM
    still never reaches "live" on /v2/iex, the new STREAM-ERROR/
    STREAM-DISCONNECT logging this PR added will name the actual
    rejection reason directly — treat that as a NEW finding, not a
    reopening of this one (RECURRENCE ESCALATES only applies to the same
    root cause recurring, not a different cause behind the same symptom).

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
    LIVE VERIFICATION STILL PENDING: this session could not deploy or
    observe the fix live (autonomous sessions don't control Railway
    deploys). NEXT: the first session after this deploys should query
    `/api/diag/orders?token=$DIAG_TOKEN` and confirm no NEW SMH (or any
    other floor-basket ticker) market orders appear during extended
    hours, and that a `buy_deferred`/`sell_deferred`/
    `floor_rebalance_deferred` trail is visible instead (currently
    these actions aren't surfaced on any diag probe — `spy_floor_result`
    is computed in `scan_market()`'s return but not obviously logged
    to the audit trail the way `TIER3-*` events are; if the deferred
    actions turn out to be invisible in production, that's a smaller
    follow-up visibility gap, not a reopening of this fix).
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

17. **[FOUND 2026-07-09, not yet repaired, low priority] `TIER3-ML-ERROR:
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

## OPS GOTCHAS (avoid re-learning)

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
