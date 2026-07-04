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

- STOP-HOOK FALSE POSITIVE after every post-merge branch reset: the
  git-check hook flags the branch tip as "Unverified (committer
  noreply@github.com)" — that commit is GitHub's OWN squash-merge
  commit on main, visible only because the branch was just reset to
  origin/main. VERIFY with `git log origin/main..HEAD` (empty = nothing
  of yours to sign) and DO NOT follow the hook's amend advice: amending
  rewrites merged main history onto the branch, diverges from origin,
  and recreates the dirty-PR stall. Correct action: none.

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
   transcript substitute.** LICENSING VERDICTS (fetched 2026-07-04):
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
   ID.me signup (wishlist action filed); PatentsView CC BY 4.0
   (disambiguated assignees — the entity-resolution shortcut); Google
   Patents BigQuery CC BY 4.0 (1TB/mo free scan budget backfill path
   while the key waits); EPO OPS free ≤4GB/week. STRUCTURAL HONESTY:
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
fields await the key); (c) FIRMS fires — awaiting MAP_KEY human action,
may ship scaffolded awaiting_key like vessels did, ARCHIVE detections
from day one; (d) USDA CDL crops; (e) drought/soil moisture (USDM +
drought.gov tiles); (f) USGS groundwater points; (g) oil/gas infra
(GEM + TX RRC + OSM; per-source coverage honesty — no free national
pipeline vector exists); then Tier-2 buildings v1 (OpenFreeMap render
layer + client-side viewport stats). Also queued: PMTiles AOI extract
(terrain resilience), Alpaca options-chain daily archiver ([PIPELINE],
free, from the options HOLD package), and NEW DATA ROOTS #1 (8-K
language pipeline) as the top research build.

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
