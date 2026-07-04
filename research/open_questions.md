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
   ACCESS LIMITATION: every deeper diagnostic route (/api/bot/audit,
   /positions, /performance, /api/daemon/health, /api/bot/ml-status,
   /api/monitoring/*) is requireOwner (session cookie for OWNER_EMAIL,
   auth.ts — frozen). Autonomous sessions cannot read audit logs or
   trade_feedback from outside the container. Deeper #3/#4 verification
   (CSP fills firing? feedback accumulating? Tier-3 retrain green?) needs
   either the human pasting /api/bot/audit + /api/bot/ml-status JSON into
   a session, or the wishlist read-only-diagnostics proposal.
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
  (this one DOES exist in code — bot_engine.py:3011, `_spread_pct > 0.005`
  — unlike the two above; not yet counterfactual-logged, since
  `shadow_portfolio.log_candidate()` is only called from inside
  `deep_score()` with decision values `taken`/`rejected_score` — the
  spread/correlation/regime/kill-switch decision buckets the function's
  own docstring anticipates (`rejected_heat`/`rejected_halt`/
  `rejected_earnings`/`rejected_other`) have zero real call sites
  anywhere in the repo. Wiring those in is the natural next
  counterfactual-logging PR — smaller and lower-risk than #10's
  ablation, since it only adds observability, not new trading behavior.)
- Correlation/sector blocks — cost vs. protection in current regime.
- Kill-switch drawdown thresholds — sized for real-money caution; is
  that optimal for a paper account whose goal is learning speed?

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
- Which regime detector (markov_regime vs. VXX-ratio heuristics) actually
  predicts forward volatility better? They currently coexist.
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
entry above and research/experiments.md); its natural follow-up (a
filings-language view once archive history accumulates, mirroring
filings.tsx) is now the next queued PRODUCT item alongside (d)-(g).
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
