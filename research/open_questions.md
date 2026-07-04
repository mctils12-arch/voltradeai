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

5. **Data modules not wired to live scoring.** The repo contains
   `alphadesk/` (EDGAR filings, catalyst detection, LLM filing reader),
   `macro_data.py`, `alt_data.py`, `social_data.py`,
   `institutional_data.py` — audit which of these actually feed
   `deep_score`/tier decisions vs. which are orphaned. Wire or retire.

6. **Full-repo pytest is broken at collection (pre-existing).**
   `test_auto_discovery.py` calls sys.exit() at module level, killing
   collection for everything; excluding it, 7 failures + 1 error remain in
   network/keys-dependent files (test_options_fixes, test_options_v134_fixes,
   test_fixes_pr8/11, test_full_system) — verified identical with and without
   the backtest change. CI's 4-file offline subset is the real gate and is
   green. Fix candidates: convert test_auto_discovery to proper pytest tests;
   mark network suites with a skip-if-no-keys guard.

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

## RULE COST AUDIT — after counterfactual logging exists

- Is MIN_SCORE=63 leaving winners on the table or blocking losers?
- SCORE_BAND_MAX=75 ("fake breakout" ceiling) — measure prevention-P&L.
- MAX_CHANGE_PCT=35 ("easy money gone") — verify against outcomes.
- Spread filter 0.5% — how many blocked names would have filled fine?
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

- **Dual-momentum SPY/QQQ** (from 2026-07-03 harness run, `bot_backtest.py`):
  in-sample 2016-2026 beat SPY (16.3% vs 14.1% CAGR, Sharpe 0.90 vs 0.83,
  DD -28.6% vs -33.7%). 1-of-~7 variants tried — discount per REASONING
  STANDARD #4. PRIOR stated before any out-of-sample run (#10): edge shrinks
  but survives ~+1% CAGR over SPY ex-2020-21; kill if negative in >=2
  sub-periods. Candidate tournament entrant once #1 lands.

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
