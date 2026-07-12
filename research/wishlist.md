# Data / Access Wishlist — human reviews weekly

## DATACORE MAXIMUS — program state (standing directive 2026-07-06;
## RESUME HERE — this block is the cross-session handoff, update it
## every session that works the program)

STATUS as of 2026-07-07 ~00:50Z (session claude/new-session-iu72vf):
- PHASE 0: ✅ COMPLETE + LIVE-VERIFIED — /api/data/grid-demand
  serves 9 BA respondents (EIA key active); /api/data/crop-conditions
  serves week 2026-07-05, 10 rows (NASS key active; documented-shape
  build worked first query, no error iteration needed).
- PHASE 1: ✅ COMPLETE — research/data_census.md merged (#294).
  Census #1 (OCC) ALSO BUILT + VERIFIED: /api/data/occ-volume serves
  2026-07-02, 4,547 underlyings, full customer/MM P/C splits
  (v1.0.165, #295); the 2-year-purge archive is recording. Session-run
  follow-up: one-time ~500-call OCC 2-year backfill (script it like
  the FINRA backfill precedent, deep-backfill env-gated).
- PHASE 2 (grid layer): item 1 TX PILOT — PIPELINE EXECUTED AND
  PROVEN this session: texas 709MB → 5.2MB power features (31s) →
  16.2MB PMTiles (25s, maxzoom 12). scripts/build_power_tiles.sh
  COMMITTED (reproduces the artifact in <1 min; US variant
  documented). SERVING DECISION (made, not yet built): TX pilot
  16MB artifact → commit under client/public/tiles/ (bundled to
  dist/public, express static serves range requests; image rule
  satisfied since dist/ ships in the image); US scale (est.
  60-150MB) → NEVER commit — boot-fetch from a GitHub Release asset
  into the volume. ITEM 1 SHIPPED v1.0.166 (see experiments 2026-07-07). Original steps kept for the US-full rerun:
  (a) run scripts/build_power_tiles.sh, copy power_tx.pmtiles to
  client/public/tiles/; (b) npm i pmtiles@4.4.1; (c) datamap layer:
  pmtiles:// protocol registration, line layer colored by voltage
  with 'voltage unknown' class (NEVER drop untagged), zoom gates
  (≥230kV z<6, ≥100kV z<9, all z≥11, substations z≥9), ODbL
  attribution string; (d) npm run build && npm run visual at
  390/768/1440 + screenshot self-review; (e) layers.json registry
  entry labeled RAW with coverage=Texas-pilot honesty.
- PHASE 2 item 1 LIVE-VERIFIED post-deploy: /tiles/power_tx.pmtiles
  answers range requests (206) and the layers registry serves the
  powergrid entry with full coverage-honesty text. Items remaining:
  2 = US-full (boot-fetch-from-Release design, filed above, NOT
  built); 3 = EIA-930 demand join (needs item 2 or region mapping).
- PHASE 3 (imagery): item 3a SHIPPED v1.0.173 — live viewport
  capture-date chip on /data (moveend identify at view centre,
  zoom-level-aware, 'unknown' honesty states, harness ratchet pins
  the chip; two occlusion drafts caught in screenshot self-review).
  Remaining Phase 3: 3b Latest Sentinel-2 cloud-free toggle (CDSE
  quota mandate: scheduled facility chips ONLY, never per-viewport —
  10k req + 10k PU/month); 3c S2 utilization review across asset
  classes; per-layer freshness chips (with Phase 5).
- PHASE 4 (UI): Streams inventory tab SHIPPED v1.0.167 — server
  aggregate /api/data/streams (manifests × archive scan; cache-only
  route; health derived from age vs cadence, raw age always shown;
  size-capped latest-record peek) + #/data/streams overlay
  (mobile-first cards, health filter chips, expandable peek) +
  launcher row atop the layer panel. Remaining Phase 4: cross-stream
  timeline, per-entity dashboards, alerting.
- PHASE 5 (ratchets): inventory-coverage ratchet LANDED with Phase 4
  (streamsInventory.test.ts: every datacore/manifests/*.json must
  appear in the inventory; aggregator enumerates the dir at runtime
  so new streams surface mechanically). streams page registered in
  the visual harness PAGES (perf/layout gate at 390/768/1440).
  Remaining: imagery capture-date requirement (with Phase 3),
  per-layer freshness chips.
- CENSUS BUILD #3 JODI: SHIPPED v1.0.169 — scripts/jodi_oil.py +
  datacore/jodi/primary_stocks.json (350 series, 61k closing-stock
  points, full history 2002+; monthly session-run rebuild ~19th).
  Gate-1 workup filed: reconcile JODI US stock definition vs EIA
  (first look found a definitional gap, honest non-match in
  experiments.md). Secondary (products) file = separate future build.
- R14 PACKAGING REPAIR (v1.0.168): the image ships dist/ only —
  runtime datacore/ disk reads were silently empty on prod (streams
  inventory + sentinel2 freshness). build.ts now stages runtime files
  into dist/datacore/; repoDataPath() resolver + ratchet battery.
  LESSON for every future stream: verify the POSITIVE case on prod,
  not just error-free responses.
- CENSUS BUILD #4a FINRA Query API: SHIPPED v1.0.170 —
  server/finraQuery.ts + /api/data/short-interest
  (consolidatedShortInterest semi-monthly + thresholdList daily;
  contract live-verified: POST-only filters, record-total pagination,
  204-as-empty, count-verified partitions; live end-to-end matched
  the workup exactly: 22,180 records @ 2026-06-15, 17 threshold
  names @ 2026-07-02). Part 2 (ATS venue summaries weekly/monthly/
  blocks, 66k-210k rows/week) = separate build, volume budget first.
  History backfill env-gated FINRA_QUERY_BACKFILL=1 (OFF by default,
  R8 lesson; SI ~80MB gz + threshold ~8MB when Mike wants it).
- SEC FTD workup COMPLETE (subagent, findings in experiments
  2026-07-07): build is next — URL pattern NON-uniform (5 eras since
  2004; index-page scrape is the robust source; 202605a lives at
  /files/data/other/ TODAY), half-month boundary is 1-14/15-EOM (not
  15/16 folklore), 2-line trailer = free checksum, PRICE "." = null,
  QUANTITY is a level not a flow, earliest 2004-03-22, ~1.3MB zip
  per half-month, SEC UA etiquette required (default UA → 403).
- CENSUS BUILD #6 SEC FTD: SHIPPED v1.0.171 — server/secFtd.ts +
  /api/data/ftd (trailer-checksummed, URL-fallback-chain, live
  end-to-end: 58,328 rows @ 202606a exact-match to trailer). The
  SETTLEMENT-STRESS COMPOSITE is now ingredient-complete
  (finrathreshold + secftd + finrashortvol) — computing it is a
  queued [RESEARCH] task with a gate-1 plan (threshold persistence x
  FTD balance delta x short-volume ratio; base rates per REASONING
  STANDARD #3). 2004→present backfill = filed volume-budget decision
  (~370MB gz).
- FINRA PART 2 DESIGN NOTES (for the build session): weeklySummary
  partitions are 66k-210k rows — EXCEEDS finraQuery's sync
  MAX_PAGES=12 (60k cap); use the workup-verified ASYNC path (POST
  async:true → 202 checkStatusLink → poll → presigned S3 CSV, ≤100k
  rows, 12h expiry, 20 async/min) OR raise MAX_PAGES to 45 with the
  count-verify guard (simpler; 210k-row 2021 weeks only matter for
  backfill). Volume: ~4-6MB gz per week ≈ 200-300MB/yr — comparable
  to finrashortvol growth, acceptable; state the estimate in the
  manifest. blocksSummary (192/mo) + monthlySummary (63k/mo) ride
  along trivially in the same module.
- SETTLEMENT-STRESS COMPOSITE [RESEARCH] (queued): ingredients all
  recording (finrathreshold + secftd + finrashortvol). Gate-1 plan:
  threshold-persistence (consecutive days listed) x FTD balance
  DELTA (it's a level — diff it) x short-vol ratio percentile;
  base-rate control = same-universe random entry (REASONING STANDARD
  #3); needs the env-gated backfills for depth — design can precede
  data.
- CENSUS BUILD #7 EU MACRO: SHIPPED v1.0.175 — server/euMacro.ts +
  /api/data/eu-macro (ECB EXR/EST/ILM + Eurostat sts_inpr_m + BBK
  Bund 10y; keyless, per-series attribution verified, vintage-honest;
  live end-to-end matched the workup exactly). Follow-up filed:
  ILM balance-sheet COMPOSITION items (enumerate BS_ITEM codes
  before coding).
- CENSUS #5 GEM: PART 1 SHIPPED v1.0.176 (9b RESOLVED — Mike
  delivered release files): datacore/gem/ = coal mines (250, full
  geometry gz), gas pipelines registry (4,246 rows, NO route coords
  in this variant), gas finance tracker (243 LNG + 531 plants). MORE
  FILES IN MIKE'S GOOGLE DRIVE — connector installed but toggled OFF
  for the chat; Mike enables it, then ingest the rest (extend
  scripts/gem_ingest.py + gem manifest). Follow-ups filed: coal-mine
  map layer from the geojson.gz; parent-company -> ticker join into
  the entity graph; GEOT ownership tracker.
- CENSUS #5 GEM COMPLETE (v1.0.176 part 1 + v1.0.178 part 2): full
  tracker catalog ingested (18 artifacts ~40MB: GEOT ownership graph
  w/ SEC-CIK crosswalk, GIPT 182k power units, GMET methane plumes,
  coal/oil/gas/LNG/steel/cement/chemicals registries; every skip
  stated in suite_skips.json). QUEUED FOLLOW-UPS: (a) pipelines
  PMTiles map layer from the GIS zips (68-72MB geojsons in Mike's
  Drive copy + scratchpad); (b) entity-graph CIK join (GEOT <->
  EDGAR); (c) LNG-carrier IMO join to AIS archive; (d) GMET plumes
  as a dated map/event layer.
- CENSUS BUILD #4 FINRA PART 2: SHIPPED v1.0.208 — server/finraQuery.ts
  extended (not a new file — same module, same contract family) with
  weeklySummary + monthlySummary (composite-key partitions
  [period, tierIdentifier], live-verified 2026-07-08: weekly tiers
  T1/T2/OTCE/NA, monthly tiers NMS/OTCE) + blocksSummary (single-key,
  per-ATS-venue ranks). Live-probed sizing found weeklySummary runs
  66k+ rows for a populated tier — exceeds part 1's MAX_PAGES=12 (60k)
  cap, so a new ATS_MAX_PAGES=50 (250k ceiling) is used for weekly/
  monthly only via a generalized fetchPartitionRowsMulti (part 1's
  fetchPartitionRows now delegates through it, zero behavior change,
  covered by the existing test suite). Went with the "simpler" option
  the design notes flagged as acceptable (raised page cap over the
  async-job S3 path) — current live weeks fit comfortably under 250k;
  the async path stays unbuilt, only relevant to 2021-era ~210k-row
  backfill weeks, which aren't in scope (no backfill built for part 2).
  GRANULARITY FINDING (live-probed, not assumed): each partition mixes
  per-symbol-per-firm/per-symbol/per-firm rows in one response
  (summaryTypeCode field) — only the *_SMBL (no _FIRM suffix) rows are
  ranked into the weekly ATS/OTC-by-symbol and monthly OTC-by-symbol
  leaderboards; the composition[] field on every summary states every
  granularity mixed in, so nothing is silently blended. Empty tier/
  venue checks (e.g. weekly OTCE/NA when only T1 is populated) are
  NOT marked archived — matches part 1's existing "204 isn't a done-
  marker" pattern, so they're honestly re-polled every 6h cycle rather
  than permanently skipped. New route GET /api/data/ats-summary
  (RAW, no predictive claim); 3 new manifests (finraweekly/finramonthly/
  finrablocks) picked up automatically by the streams-inventory ratchet
  (enumerates datacore/manifests/ at runtime — no hardcoded list to
  update). 17 new server tests (composite-key tuples, multi-filter
  AND-not-OR, raised page cap, all three summarizers, end-to-end
  refresh incl. the empty-tier-repoll behavior); node 462/462 total;
  tsc 64 (unchanged baseline); build clean, dist/datacore manifests
  confirmed staged (R14 packaging lesson). No UI page — same
  pipeline+API-first sequencing as sec8kEarnings/finraQuery part 1
  (view once archive history accumulates). NEXT: still open —
  NDBC buoys, SEC MIDAS; EPA CAMD/ENTSO-E on Mike's keys
  (9a/9c); a FINRA part 2 UI view once weeks of archive accumulate;
  the settlement-stress composite [RESEARCH] item (unrelated to part 2,
  ingredients from finrathreshold+secftd+finrashortvol only).
- **USGS EARTHQUAKES: SHIPPED 2026-07-08 (v1.0.209)** —
  server/usgsQuakes.ts + /api/data/earthquakes (RAW, M2.5+, global,
  rolling 24h, keyless US-government-public-domain feed; live-verified
  end-to-end from this session's own `fetch`, not just curl). Every
  event carries USGS's own stable id (no composite key needed); a
  revised `updated` timestamp re-archives the row (append-only) so
  in-review magnitude/location corrections aren't silently frozen at
  their first-seen "automatic" values. MAP LAYER SHIPPED 2026-07-11
  (v1.0.267, T-CLIENT [PRODUCT] session): datamap.tsx "earthquakes"
  layer — magnitude-scaled/colored markers (USGS ShakeMap-style
  green->red bands), click detail card, legend, off by default
  (perf-budget precedent). Hazard-adjacent SIGNAL hypotheses (insurer P&C exposure, utility/
  infra proximity, supply-chain proximity) filed in open_questions.md
  under EARTHQUAKE HAZARD-ADJACENT HYPOTHESES — gate 1 trivially passes
  (USGS is ground truth), gate 2 blocked on archive depth (started
  today) or an unbuilt facility-proximity join. NDBC buoys (grouped
  with quakes in the prior NEXT line) is now the next unclaimed item —
  a single global `latest_obs.txt` fetch (~866 stations, live-verified
  2026-07-08), no per-station requests needed.
- **NDBC BUOYS: SHIPPED 2026-07-08 (v1.0.220)** — server/ndbcBuoys.ts +
  /api/data/buoys (RAW, ~889 reporting stations worldwide, keyless
  US-government-public-domain feed; live-verified end-to-end from
  this session's own `fetch`, not just curl — 889 stations, 553 with
  a valid wave-height reading). Single global whitespace-delimited
  text file (NOT fixed-width despite the aligned header — confirmed
  against the live sample, including 4-char vs 5-char station ids and
  negative lat/lon rows); "MM" missing-sensor tokens map to null,
  never coerced to 0. Archive dedup keyed on (station, observation
  time) rather than a simple id, since every station reappears every
  poll (a snapshot feed, not a discrete-event feed like USGS quakes) —
  a station only re-archives when its own timestamp advances, so
  30-min over-polling against an hourly-reporting station is a cheap
  no-op, not archive bloat. MAP LAYER SHIPPED 2026-07-11 (v1.0.267,
  same T-CLIENT [PRODUCT] session as earthquakes above): datamap.tsx
  "buoys" layer — station markers, click detail card (wave/wind/
  pressure/temp, missing sensors shown as "no data" never zero),
  legend, off by default. Sea-state (WVHT/DPD) and pressure-tendency
  fields are a filed ANGLE-HUNTING candidate (marine-forecasting
  technique import) in open_questions.md — gate 1/2 unattempted.
- **SEC MIDAS: SHIPPED 2026-07-10 (v1.0.265)** — server/secMidas.ts +
  /api/data/microstructure (RAW, census build #10, public domain).
  Endpoint FOUND this session via live web search (WebSearch +
  WebFetch against the current downloads page) — two prior sessions
  (2026-07-06 census entry, 2026-07-08 COT-session fall-through note)
  both guessed static URL patterns from memory and 404'd; the real
  path (`sec.gov/files/opa/data/market-structure/
  metrics-individual-security/individual_security_{year}_q{n}.zip`)
  is only discoverable from the live page, not guessable — see
  experiments.md and open_questions.md's new MIDAS HFT-COLONIZATION
  FILTER HYPOTHESIS section for the full trace. Live-verified: q4_2025
  zip (23.5MB) parses to 533,077 rows across 65 dates, 235,165 Stock +
  297,911 ETF rows; discovered live (not documented anywhere) that
  ETF ranks are QUARTILES (1-4) while Stock ranks are DECILES (1-10) —
  never comparable, kept distinct in the schema and the smallcap_watch
  filter (Stock-only). No official checksum exists for this format
  (unlike secftd's trailer lines); integrity enforced by a >1%
  malformed-row-rate refusal guard instead. MEMORY-BOUNDED poller
  (deviation from the secFtd template, stated in the module docstring):
  probes all unarchived candidate quarters cheaply (404s cost nothing)
  but archives (parses + gzips a ~68MB CSV) at most once per poll call,
  given each quarter is ~9x secFtd's half-month file size. 8 new tests
  (server/secMidas.test.ts); full node suite green (547/547 with
  node_modules freshly installed — 3 pre-existing "failures" from the
  prior session's report turned out to be missing-devDependency
  artifacts of a partial sandbox, not real regressions, resolved by
  `npm install`); tsc error count unchanged (3, pre-existing
  vite/client + baseUrl config warnings, confirmed via git-stash
  A/B); `npm run build` clean, dist/datacore/manifests/secmidas.json
  confirmed staged (R14 packaging lesson respected). Only the newest
  quarter (2025q4) is archived so far — MIDAS_LOOKBACK_QUARTERS=6
  backfills the rest over the next several daily polls automatically;
  full 2013-2025 history (~50 quarters, ~1GB gz est.) is a SEPARATE
  volume-budget decision, not built. Gate 2 SIGNAL is blocked on BOTH
  MIDAS history accumulating AND a join against an existing small-cap
  candidate stream (Form 4 clusters is the natural partner) — this is
  a cross-stream filter hypothesis, not a standalone screen; full
  ladder path in open_questions.md. NEXT unclaimed DATACORE MAXIMUS
  item: EPA CAMD/ENTSO-E, both gated on Mike's keys (see 9a/9c above)
  — nothing else queued in this program as of this session.

## GRID VISION — program state (human directive 2026-07-07; charter =
## research/grid_vision.md, RESUME STATE block at its bottom is the
## authoritative handoff — this entry is the pointer)

STATUS as of 2026-07-07 later same session (claude/new-session-iu72vf):
- Charter installed (research/grid_vision.md): phases A (research
  first) / A2 (products plan — "the products define the spec") /
  B (VERIFY→EXTEND→DISCOVER detection w/ provenance tags) /
  C (national rollout, TX first, per-state honest coverage) /
  D (visualization at the premium standard) / E (ratchets).
- PHASE A COMPLETE + PHASE A2 COMPLETE, same day: all four subagent
  reports filed in research/grid_vision_research.md (Items 1–4 +
  cross-cutting summary); products plan filed in
  research/grid_vision_products.md. Headlines: retraining mandatory
  (no released weights anywhere); Esri ML use forbidden by quoted
  contract clauses → NAIP public-domain substrate via free Planetary
  Computer STAC streaming; honest detection scope = transmission
  towers + substations (not poles); TX corridor-verify CPU-feasible
  ~7 h; first product needs NO ML (grid-stress index from OSM TX +
  EIA-930 + weather, all already archived). The charter's RESUME
  STATE block carries the full build order.
- Strategic update installed same day: "DATA IS THE MOAT, EXPERIENCE
  IS THE DOOR" (VISION.md appended section; CLAUDE.md Amendment 5 —
  GOAL self-proposed-work weighting + PREMIUM EXPERIENCE STANDARD
  standing behavior). Human directive = the approval (HUMAN
  SOVEREIGNTY clause).
- UPDATE 2026-07-07 (PRODUCT session, v1.0.191): gate-2's FAIL-path
  item (2) SHIPPED — descriptive grid-stress dashboard surface,
  #/data/grid-stress. Full trace in experiments.md + grid_vision.md's
  RESUME STATE block. Remaining queue: V3 gate-2 design, Phase B
  training-data prep, Phase B1 VERIFY spec.

## [RESOLVED same day — Mike funded RunPod and put RUNPOD_API_KEY in
## Railway, 2026-07-07. GPU training/sweeps unblocked; first consumer
## = Phase B training-data prep + fine-tune (charter RESUME STATE).]
## Original order kept below for the record.
## BLOCKED-FOR-MIKE — GRID VISION GPU compute (2026-07-07; RunPod
## purchase order per the directive's paid-boundary rule)

WHAT: prepaid GPU rental for the GRID VISION detector. Training is
needed REGARDLESS of scope (Phase A finding: no usable released
weights exist anywhere — we must fine-tune our own on CC-BY labeled
data); full-state discovery sweeps use the same account later.
- SERVICE: RunPod — https://www.runpod.io/ (per-second billing, no
  quota approval, prepaid credit; L4 $0.39/hr, RTX 4090 $0.69/hr,
  prices verified 2026-07-07).
- COST: initial deposit $50 (covers the full training/experiment
  cycle, est. $50–100 total, + a Texas full sweep $5–25); national
  sweeps later ~$100–400 depending on resolution mix.
- SIGNUP: create account at runpod.io → add $50 credit → Settings →
  API Keys → create key.
- CREDENTIAL: put it in Railway as RUNPOD_API_KEY (presence-checked
  only, never echoed). The session that sees it appear activates GPU
  work without being told.
- CAPABILITY UNLOCKED: detector training + repeatable full-state ML
  sweeps at ~$4–15/state/pass (change detection on every NAIP
  refresh, not one-time mapping).
- CPU-FEASIBLE MEANWHILE (no purchase): Phase B1 corridor-verify
  over ERCOT OSM buffers (~7 h @0.6 m streamed free from Planetary
  Computer) and the entire A1→grid-stress-index product path.
- OPEN SUB-DECISION (2026-07-07, non-blocking) — LAUNCH PATH:
  RUNPOD_API_KEY is in Railway but NOT in the agent session
  (presence-checked 0), so a session cannot itself call the RunPod
  API to launch a pod. The cost-cap gate + spend ledger are BUILT and
  tested (scripts/runpod_budget.py, test_runpod_budget.py,
  research/runpod_ledger.md); the actual pod launch must run as a
  server-side / Railway routine that reads the key and calls
  authorize_job() first (refuses any unbounded job, passes the hard
  max_runtime_seconds cap to RunPod) — OR the key is added to the
  session. Either is fine; Mike's call. This gates only the GPU
  launch step, NOT the non-GPU Phase B data-prep (ETDII download +
  OSM-seeded chips), which proceeds now. SATELLITE SPLATTING is
  CANCELLED ($0 GPU) — model research found 0 splat candidates; glTF
  covers it. The whole $50 is reserved for grid-vision.

## BLOCKED-FOR-MIKE — OCC backfill flag (2026-07-07)

9e. OCC 2-YEAR BACKFILL — RESOLVED 2026-07-07: Mike confirmed
    headroom (~2GB of 5GB used) and approved; v1.0.174 flips the
    mechanism DEFAULT-ON (opt-out OCC_DEEP_BACKFILL=0). The backfill
    runs once on the next deploy's boot (~20 min oldest-first walk,
    gz-on-write, done-marker prevents repeats). VERIFY: occvolume dir
    grows toward ~500 day-files; backfill_done.json appears with
    days_fetched; /api/data/archive/stats shows the growth.

## BLOCKED-FOR-MIKE — DATACORE MAXIMUS census additions (2026-07-06;
## each unlocks a census top-10 item; routed around meanwhile)

9a. **EPA CAMD key (api.data.gov, instant, free)** — unlocks census
    #2: unit-level HOURLY power-plant utilization (grossLoad×opTime),
    history to 1995 — the ground-truth source for the whole power
    vertical. Signup: api.data.gov/signup → set EPA_CAMD_API_KEY in
    Railway. Highest-value single key in the census.
9b. **Global Energy Monitor download (form-fill, 2 min)** — name +
    email at globalenergymonitor.org/projects/global-integrated-power-tracker/download-data
    → forward the xlsx link (or the file) — CC BY 4.0, 182k
    facilities with status + coordinates; the join spine for the
    grid layer.
9c. [RESOLVED 2026-07-07 — Mike set ENTSOE_API_KEY in Railway;
    server/euLoad.ts activated same-day per the detect-and-activate
    directive (v1.0.186): actual total load A65/A16 for 8 zones
    (DE_LU/FR/ES/IT/NL/PL/BE/SE), /api/data/eu-load, euload manifest.
    Follow-ups filed: generation mix + day-ahead prices (same token,
    separate builds); 2015+ historical backfill = volume-estimate
    decision first.] Original: ENTSO-E token (free) — register at
    transparency.entsoe.eu → ENTSOE_TOKEN. Unlocks EU hourly
    load/gen/prices.
9d. **OpenAQ key (low priority)** — explore.openaq.org signup →
    OPENAQ_API_KEY; S3 bulk archive exists keyless so this can wait.

## ⚠️ URGENT — READ FIRST (2026-07-06 ~15:45Z, push-notified)

**#9. ALPACA SIP DATA ENTITLEMENT REJECTED — check the Alpaca
dashboard Market Data subscription (2 min).** Since Monday's open
every market-data request with feed=sip returns HTTP 403 (named by
/api/diag/scanner after the v1.0.148 visibility fix): the Tier2 scan
found ZERO new candidates all morning; options scanner, VXX regime
reads, SPY floor, and shadow-portfolio backfills were equally blind.
The trading API and account access are unaffected. Likely causes: an
Algo Trader Plus subscription lapsed/changed, or Alpaca changed free-
tier SIP access. **Self-repair shipped (v1.0.150):** the stack now
probes the entitlement and auto-downgrades to feed=delayed_sip — the
FULL consolidated tape at a 15-minute delay, free on all tiers — so
scanning works again with honest volume numbers (feed=iex was
rejected: ~30-50x volume undercount would poison the $50M floors).
If you restore the paid SIP subscription, the bot upgrades itself
back to real-time within 10 minutes (probe TTL) — no deploy needed.
Decision for you: pay for real-time SIP (Algo Trader Plus, ~$99/mo)
or accept 15-min-delayed candidate discovery (executions still price
live via the trading API either way).

## [RESOLVED — 2026-07-05, confirmed stale 2026-07-06 (two independent
sessions, v1.0.146 and this one)] ~~#8. PROD RESTART LOOP~~

Self-resolved same day: root-caused and fixed in code (v1.0.143, boot
archive folds now stream with bounded state + cgroup-aware Node heap;
see experiments.md) before this got read — no Railway dashboard
action was ever taken on it. Verified twice since: uptime_s climbing
cleanly past 29,000s+ with steady heap/rss, zero restart signature.
No human action needed; leaving this closed-out note so a stale
"if unresolved by Monday the bot cannot trade" alarm doesn't sit at
the top of a weekly review past its truth date.

## FROZEN-FILE AMENDMENT PROPOSAL (durability audit 2026-07-05 —
needs your one-line edit; billing.ts is frozen so sessions may not
touch it)

**server/billing.ts:34 writes its SQLite DB to the container image
dir — wiped on every redeploy — AND it is a different file from the
auth DB whose tables it queries.** Billing is currently dark (no
STRIPE_SECRET_KEY) so nothing has been lost yet, but the moment
billing activates, customer/subscription state would be ephemeral
and the users/sessions UPDATEs would hit an empty database. EXACT
CHANGE (mirrors auth.ts three lines above it):
  before: `const dbPath = path.resolve(process.cwd(), "voltradeai.db");`
  after:  `const dbPath = process.env.DB_PATH || path.resolve(fs.existsSync("/data") ? "/data" : process.cwd(), "voltrade.db");`
(pointing billing at the SAME durable /data/voltrade.db auth already
uses fixes both the persistence and the split-brain table bug; add
`import fs from "fs";` if absent). A ratchet in
server/durability.test.ts pins the current stray signature and says
exactly what to update once you apply this. The monetization
tripwire already forces a review before billing activates — this
proposal is now part of that checklist.

## BLOCKED-FOR-MIKE (standing list per the overnight directive
2026-07-05: items needing paid keys, spend, or a human-only decision.
Logged and ROUTED AROUND — nothing here blocks the free build order.)

1. **Databento options history, ~$740** (2016→present daily-close
   chains) — pilot validated GO (parity-perfect sample); ALSO needs
   your durable-storage decision (~5GB; sessions are ephemeral).
   Worth it: only affordable path to a decade of chain history for
   theta/IV research; our free forward archive covers 2026-07-06 on.
2. **PDUFA target-date calendar** — NOT freely available (FDA legally
   barred from publishing; aggregator calendars are ToS-protected
   scrapes we will not touch). Free substitute SHIPPING instead:
   Federal Register AdCom dates + openFDA approvals; follow-on will
   mine company-disclosed PDUFA dates from our own 8-K archive as
   labeled estimates. Paid option if ever wanted: BPIQ-class
   subscription (~$50-100/mo tier) — low priority given substitutes.
3. **Tank-fill post-gate-2 enhancements (optional, NOT blocking the
   build):** sub-meter tasking/archive for roof-type confirmation
   (Planet/Maxar, ~$10-25/km²-class with minimums) and a paid analyst
   consensus feed for the EIA-surprise definition. Free versions
   proceed; per standing rule these may not even be proposed for
   spend until the free estimator passes gate 2.
4. **Google Trends production-grade access** — pytrends upstream
   abandoned (archived 2025-04); if the free gate-1 probe fails, the
   only reliable paths are paid scrapers (Glimpse/SerpApi-class) or
   Google's invite-only official API. Await probe result first.
5. **FRED_API_KEY in the Claude Code session env** (2 min, free) —
   already live in Railway; adding it to the session env lets
   research sessions pull FRED directly instead of via prod.
8. **TWO FREE KEYS unblock BUILD ORDER 6 #6/#7 (filed 2026-07-06;
   both build key-gated NOW per the fredMacro/census pattern and
   activate the moment keys land in Railway):**
   (a) **EIA_API_KEY** — eia.gov/opendata/register.php, instant,
   free. Unblocks hourly grid demand (EIA-930, 2019→present, public
   domain) — industrial-activity nowcast joining our degree-days +
   power-plants layers. (b) **NASS_API_KEY** —
   quickstats.nass.usda.gov/api key request, instant email, free.
   Unblocks weekly crop conditions (pairs with the keyless Drought
   Monitor stream into an ag-stress index). Free-alternative
   analysis: none needed — these ARE the free tier; the only cost is
   two email signups.
7. **PATENTSVIEW_API_KEY (free request form) — unblocks BUILD ORDER
   5 #6 USPTO patents** (probed 2026-07-05 per the item's
   probe-first instruction): PatentsView has required a free API key
   since 2021 (the legacy keyless api.patentsview.org 301s away);
   the request form at patentsview.org/apis/keyrequest is
   human-facing. ALSO: search.patentsview.org is currently
   502-blocked through the session proxy and developer.uspto.gov
   503s — so after the key, first build session must re-probe
   reachability (server-side from Railway may work where the
   session proxy fails; the key-gated fredMacro/census pattern
   handles either). Worth it: grant-rate inflections +
   citation-weighted grants for small-cap assignees (EDGE DOCTRINE
   named it day one); assignee->ticker mapping reuses the
   name-matcher pattern. Free alternative if the key path stalls:
   USPTO bulk XML weekly files (large but keyless) — materially
   more build work; recommend the key first.
6. **[DONE — VERIFIED LIVE 2026-07-05, same day the human added the
   key]** ~~CENSUS_API_KEY (free, instant email signup) — unblocks
   BUILD ORDER 3 #4 container imports~~ — built as
   server/censusImports.ts (key-gated, fredMacro pattern, v1.0.132,
   PR #249). The key IS in Railway: ~30 min after merge,
   /api/data/imports served 686 live records (April 2026 port-level
   import values with containerized fields populated — the FIRST
   query variant was correct; the anticipated shape-fix-from-logs
   path was never needed). Census's national aggregate row (port
   "-", "TOTAL FOR ALL PORTS") comes through as published — kept.
   Archive now accumulates monthly vintages. NOTE: the key is still
   NOT in the Claude session env (fixed at session start); add it
   there too only if session-side research pulls are ever wanted —
   not required for the pipeline.

- **[DONE 2026-07-05 — key set in Railway, stream #3 built same day]**
  ~~HUMAN ACTION — FRED API key~~ — human set FRED_API_KEY in Railway;
  server/fredMacro.ts (v1.0.90) polls ~31 regime series, archives
  point-in-time vintages, serves /api/data/macro (public-license series
  only; CBOE/ICE BofA/UMich stay internal). NOTE: the key is NOT in the
  Claude Code session env — session-side research pulls will need it
  there too (same env-var screen as the others) or will keep using the
  keyless fredgraph.csv export as ground truth. License: free with
  attribution ("Source: FRED, Federal Reserve Bank of St. Louis");
  terms re-check at monetization switch per the standing rule.

- **[DONE — VERIFIED END-TO-END 2026-07-05]** ~~Copernicus Data Space
  (CDSE) free account~~ — CDSE_CLIENT_ID / CDSE_CLIENT_SECRET live in
  the session environment and PROVEN with the real credentials:
  (1) OAuth client_credentials token issued; (2) OData catalog search
  found a fresh S1D GRDH scene over Cushing (sensed 2026-06-26);
  (3) a REAL 256×256 Sentinel-1 VV SAR chip of the Cushing tank farm
  pulled via the Sentinel Hub Process API on CDSE (61KB PNG,
  radar-bright tanks clearly resolved) — the fused-sensor engine's S1
  leg is UNBLOCKED, and the Process API (server-side AOI windowing,
  free-tier processing units) is the RIGHT primitive for the
  chip-based change-detection design — better than bulk product
  downloads. GOTCHA RECORDED: the bulk zipper endpoint returns
  DAT-ZIP-609 "token audience not allowed" for custom OAuth clients —
  irrelevant to chips; if whole products ever matter, use S3 keys
  from the CDSE dashboard or the cdse-public password grant.
  (Original entry: Copernicus Data Space (CDSE) free account
  (satellite directive 2026-07-04): ONE credential unlocks the
  fused-sensor engine (Sentinel-1 SAR + Sentinel-2).)** Exact steps: (1)
  dataspace.copernicus.eu → "Register" (top right); (2) email +
  password, verify the confirmation email; (3) no approval wait — the
  account is immediately active on the free tier (generous monthly
  download/processing quotas, sufficient for weekly AOI chips); (4)
  create an OAuth client under Account → "OData / API access" (or we
  can use plain username+password S3-compatible access) and set
  CDSE_USER / CDSE_PASSWORD (or client id/secret) in Railway. NOTE:
  Sentinel-2 already works WITHOUT credentials (Element84 STAC + AWS
  COGs, proven in the gate-1 pipeline); CDSE matters for SENTINEL-1
  SAR. Zero-credential S1 fallback we will verify first: ASF DAAC
  (free NASA Earthdata login — also a HUMAN ACTION if preferred:
  urs.earthdata.nasa.gov signup, same shape). Either credential works;
  CDSE is the single-signup option the directive named.
- **⚖ [APPROVED BY HUMAN 2026-07-04 — "ANGLE-HUNTING amendment — ship
  it"; applied to CLAUDE.md STANDING BEHAVIORS same day, dated
  2026-07-04.] AMENDMENT — ACTIVE ANGLE-HUNTING (satellite/EDGE
  directive 2026-07-04).** Applied text (as proposed):

  "- ACTIVE ANGLE-HUNTING (human-approved YYYY-MM-DD): the system does
  not only execute directed roadmaps — it generates its OWN novel
  hypotheses. Every EDGE session not consumed by repair or a
  higher-priority queued item deliberately hunts new angles: (1)
  CROSS-CONNECTIONS — join two or more existing data streams in ways
  not yet tried (insider-buying × facility SAR-activity × port-dwell;
  corporate-fleet aircraft utilization × earnings timing; plant
  thermal-activity × commodity prices) — the Everything Graph is the
  substrate; (2) ANOMALY MINING — scan the archives for unexplained
  recurring patterns, especially ones preceding price moves, and ask
  why; (3) FOREIGN-FIELD IMPORTS — borrow techniques from outside
  finance (epidemiology, ecology, signal processing, aviation
  failure-analysis) and test whether they reveal structure others
  miss; (4) SECOND-ORDER — for any edge found, ask who is on the other
  side and why it has not been arbitraged. FREEDOM + RIGOR:
  unconventional, speculative, even weird hypotheses are explicitly
  wanted — AND every one enters the ROOT VALIDATION LADDER before it
  is believed or traded. Every generated angle is logged in
  open_questions.md with its testable form and ladder path, even the
  speculative ones; priors stated before testing; edges discounted by
  the number of combinations tried; out-of-sample confirmation
  required; a beautiful story never substitutes for validation."
- **Phase-2 PAID sub-meter imagery (satellite directive 2026-07-04 —
  BUILD-FIRST analysis attached; gated on a Phase-1 fused free signal
  passing gate 1 AND revenue justification; do NOT purchase yet):**
  the raw material (sub-meter optical) is genuinely inaccessible free
  — build-first step 4 applies. Candidates: Planet SkySat archive
  (~50cm, historically ~$10+/km², tasking higher; pricing on inquiry),
  Maxar/Vantor 30-50cm archive (historically ~$10–25/km² via
  resellers), Airbus Pléiades 50cm (similar). What paid adds over the
  free fused proxy: individual tank roof-type + count, ship type +
  count, vehicle/coil counts — object identity, not just facility
  change. Validation plan on purchase: counts vs port statistics /
  EIA / operator disclosures on a labeled sample BEFORE any trading
  use. Free version first: the fused S1+S2+thermal change signal must
  prove or fail at gate 1 — if it trades, paid imagery may be
  unnecessary; if it almost-trades, paid imagery is the upgrade with a
  measured accuracy target.

- **⚖ [APPROVED BY HUMAN 2026-07-04 — ALL FOUR: "Constitutional
  amendments approved — ship all four in order 1→2→3→4"] CONSTITUTIONAL
  REPAIR — FOUR AMENDMENT PROPOSALS (directive 2026-07-04; shipping in
  order, each its own docs PR; Fix 2's runtime half additionally ships
  as its own [REPAIR] PR with a regression test).**

  AUDIT SUMMARY: The constitution has one live contradiction and three
  structural gaps. (1) The GOAL/MISSION still defines the paper account
  as the entire mission while VISION.md + GIP.md (both installed at
  human direction) define a data-intelligence platform with the bot as
  ONE consumer — sessions choosing between tending the bot and
  advancing the platform currently get wrong guidance from the highest
  section of the document. (2) Priority 1 demands "trading loop
  running" but defines no alarm — the runtime hook exists (/api/health
  Check 5 in server/bot.ts:1049 already reads killed/active/stopped
  and Check 6 licensing already demonstrates degrade-and-surface) yet
  bot-state never degrades health, which is exactly how the loop sat
  paused without any routine flagging it. (3) Human sovereignty over
  this document is practiced but nowhere stated. (4) The file is
  31.7KB read every session and growing — measured by section, ~5–6K
  chars are narrative history, restated rules, and three audit rules
  living in three places; none of it is normative force.

  ── FIX 1 — MISSION RECONCILIATION (replaces the entire GOAL section;
  priority ORDER, honesty metric, and anti-goals preserved; P3/P4 make
  both compounding lines first-class). EXACT NEW TEXT:

  "## GOAL — the mission and its priority order

  MISSION: Build a continuously-validated intelligence platform on a
  compounding proprietary archive of the physical economy — observed,
  recorded, verified, and compiled by us. The platform has two
  first-class consumers: (a) the trading bot, which turns validated
  signals into long-term compound growth of the paper account, and
  (b) external customers, to whom the same intelligence is sold via
  API and subscriptions (VISION.md and GIP.md name the destination;
  this file governs how everything ships). You are still a factory
  that produces, tests, and retires strategies AND data products
  forever — every edge decays.

  When priorities conflict, the higher number NEVER wins over the lower:

  1. KEEP THE SYSTEM ALIVE. Site up, trading loop running, daemon
     healthy, data flowing, archives recording. A dead system learns
     nothing, and an archive gap never refills.
  2. PROTECT THE INTEGRITY OF LEARNING. Every result attributable to
     a specific change. Every evaluation honest: out-of-sample, no
     lookahead, no leaked data. Every signal ladder-validated before
     it is trusted, traded, or sold. Corrupted learning is worse than
     no learning.
  3. GROW BOTH COMPOUNDING LINES. The account: maximize long-run
     compound growth (log-wealth / Kelly framing), measured as rolling
     performance vs. buy-and-hold SPY — kill switches make sustained
     aggression survivable. The platform: grow validated signals, the
     archive's reach, and the product surface (/data, /api/v1) that
     customers pay for. Neither categorically outranks the other: a
     session facing 'tend the bot vs. advance the platform' weighs
     expected compounding value of each and logs the choice.
  4. EXPAND CAPABILITY. New strategies, signals, data roots
     (wishlist), web research, and the user-facing features that
     surface them.

  HONESTY METRIC, now two-sided: live-vs-backtest divergence for the
  bot; claimed-vs-ground-truth divergence for platform signals. When
  either diverges, the factory is fooling itself — fixing that
  divergence outranks all new research.

  ANTI-GOALS: never optimize for backtest results themselves; never
  trade attribution for speed; never churn changes to look busy; never
  sell or surface a signal the ladder has not validated."

  ── FIX 2 — LIVENESS ALARM. Proposed N: **2 market hours** (Tier-2
  scans run intraday — a paused loop during market hours bleeds
  learning data immediately) plus a **24-hour wall-clock ceiling**
  regardless of session (catches weekend-long halts before Monday).
  (a) CONSTITUTION half — append to Priority 1 (inside the new GOAL
  text above, or the current one if Fix 1 is rejected):

  "LIVENESS ALARM: the trading loop paused, halted, or
  broker-account-unreadable for more than 2 market hours (or 24 hours
  wall-clock) is a TOP-OF-REPORT alarm in every DAILY session and a
  degraded state on /api/health — the loop going dark must be
  surfaced loudly, never discovered by the human on a dashboard."

  (b) RUNTIME half (own [REPAIR] PR, ships with a regression test):
  HOOK CONFIRMED — /api/health Check 5 (server/bot.ts:1049) already
  reports bot status (killed/active/stopped) but never degrades
  overall status on it; Check 6 (licensing/providerCompliance) is the
  existing degrade-and-surface precedent, and DAILY routines already
  read /api/health. Change: persist state.inactiveSince when the loop
  stops being active; when market-open time since inactiveSince
  exceeds 2h (market_calendar.py knows sessions) or wall-clock exceeds
  24h, set checks.status = 'degraded' with detail naming how long the
  loop has been dark. providerCompliance.ts itself is licensing-
  specific and stays untouched.

  ── FIX 3 — SOVEREIGNTY CLAUSE. Placement: FIRST paragraph inside
  "## AUTONOMY AUTHORIZATION", before "You may merge and deploy...".
  EXACT TEXT (verbatim from the directive):

  "HUMAN SOVEREIGNTY: the human may override any rule in this
  constitution at any time; an explicit human instruction outranks any
  provision here. The autonomy granted below is the human's
  delegation, revocable and amendable by the human alone. Nothing in
  this document limits the human — only the autonomous system acting
  without the human."

  ── FIX 4 — BLOAT CONSOLIDATION (no rule loses force — only history,
  narrative, and restatement are cut; all human-approval dates kept in
  compressed form). Measured today: 31,694 bytes. Specific cuts:

  1. STANDING BEHAVIORS (4,169 → ~2,400; −1,750): the USAGE-
     CALIBRATION paragraph carries ~600 chars of Gmail-fix history and
     delivery narrative — compress to the rule + one dated pointer to
     usage_log.md; the VISION/GIP north-star rule carries a ~450-char
     parenthetical explaining a placement decision — one line
     ("placement recorded in experiments.md") suffices; SPINOUT-READY
     and RAW-vs-SIGNALS keep every clause but lose restated rationale
     (~400).
  2. EDGE DOCTRINE (4,369 → ~3,100; −1,250): each BUILD-FIRST step
     restates its precedent twice (inline + parenthetical) — one line
     per precedent; the four-edges preamble keeps all four edges and
     every standing example name, drops repeated framing sentences.
  3. AUDIT-RULE MERGE (DEAD CODE POLICY 769 + CONSTITUTIONAL HYGIENE
     902 + the AUDIT CYCLE paragraph inside SESSION BUDGET ~500 =
     2,171 → ~1,400 as ONE '## AUDITS & DEBT' section; −770): three
     rules currently each restate cadence-register-filing mechanics;
     merged section states the mechanics once and the three audit
     types (staleness / constitutional / calendar-year-add) as a
     table. Every governing clause (exception for likely-returners,
     never-self-apply, GOAL-order conflict rule) survives verbatim.
  4. SESSION BUDGET (2,025 → ~1,500; −520): the fall-through ladder
     keeps all three tiers + hard limits; drops the amendment
     narrative and one restated anti-churn sentence (anti-churn also
     lives in ANTI-GOALS).
  Net: ~31.7K → ~27.3K (−4.3K, ~14%) with Fix 1 (+~400 for the
  platform mission) and Fixes 2+3 (+~700) already included in the
  target — pure prose reduction is ~5.4K. Full before/after text ships
  in the amendment PR for line-by-line review; nothing is applied
  before approval.

- **⚖ [APPROVED BY HUMAN 2026-07-04 — "WORKSTREAM PARTITION amendment
  — ship it"; applied to CLAUDE.md same day as a new section after
  SESSION BUDGET, dated 2026-07-04.] AMENDMENT — WORKSTREAM PARTITION
  (throughput directive 2026-07-04).** Applied text (as proposed):

  "## WORKSTREAM PARTITION (human-approved YYYY-MM-DD)

  Concurrent sessions build in disjoint FILE TERRITORIES; a session
  declares its territory in its first experiments.md entry and stays
  inside it:
  - T-DATACORE: datacore/**, the datacore server modules
    (datacoreArchive, shadowFleet, portDwell, nasaFirms, edgarForm4,
    sec8kEarnings, trainsFeed, owmTiles, future entity/graph modules)
    + their tests, scripts/ pipeline tooling.
  - T-CLIENT: client/src/**, index.css, scripts/visual_check.mjs and
    visual tooling. DESIGN.md rule changes remain amendments.
  - T-BOT: bot_engine.py, ml_model_v2.py, system_config.py,
    strategies/, analyze/insights/instrument_selector/tiered_strategy,
    server/bot.ts outside frozen paths.
  SHARED (any session, serialize + minimize): server/routes.ts,
  datacore/layers.json, package.json, research/*. MERGE-ORDER
  PROTOCOL: (1) shared-file edits are the LAST commit before the PR
  and as small as possible; (2) version = read-and-increment at
  commit time, never planned ahead; (3) research/* conflicts resolve
  keep-both-sides (append-only spirit); (4) merge monitors verify
  WHICH PR merged before any branch reset (ops rule); (5) a
  cross-territory change belongs wholly to the session owning its
  PRIMARY territory — never split one logical change across sessions;
  (6) collisions discovered late follow the supersession precedent:
  first-merged wins, the duplicate salvages its unique delta.
  Within a session, parallel subagents fan out labor (research,
  per-source builds, test generation) while judgment stays in the
  parent — subagent output ships only after the session's own
  read-before-write review."

  RATIONALE: 40 PRs merged 2026-07-04 across concurrent
  sessions/routines with 4 live collisions (2 version, 1 double-build,
  1 wrong-merge reset) — all recovered but each cost a cycle;
  territories prevent the class instead of patching instances.

- **⚖ [PARTIAL APPROVAL BY HUMAN 2026-07-04: items 2 and 3 approved as
  pre-revenue prep (DELIVERED same day: datacore/LICENSING_AUDIT.md +
  datacore/API_TERMS_DRAFT.md); items 1 and 4 explicitly WAIT until the
  human decides to charge.] MONETIZATION READINESS CHECKLIST
  (API-product directive 2026-07-04).**
  1. PROVIDER COMPLIANCE RE-RUN (the tripwire, executed for this
     directive since it touches pricing design): aircraft chain is
     adsb.lol (ODbL — monetization-lawful) primary with
     airplanes.live + adsb.fi fallbacks (both non-commercial).
     AT SWITCH: drop or upgrade both fallbacks; the runtime guard
     (providerCompliance.ts) enforces this if billing activates first.
  2. DATA-LICENSING AUDIT — what we may RESELL vs DISPLAY, per source
     (drafted; verify each at switch): SEC EDGAR + NWS/NOAA + USGS +
     EIA + USDA (US public domain — resellable); NASA FIRMS (open,
     attribution — resellable with credit, safety-of-life disclaimer
     travels); Digitraffic CC BY 4.0 + Entur NLOD (resellable with
     attribution); adsb.lol ODbL (SHARE-ALIKE: an API reselling
     ODbL-derived aircraft data must license that derived database
     ODbL and attribute — compatible with a paid API but constrains
     exclusivity claims; positions endpoints marked accordingly);
     aisstream.io (their ToS on redistribution must be re-read at
     switch — vessel endpoints marked CONDITIONAL); OpenWeatherMap
     tiles (display product — NOT resellable as data; excluded from
     the API); Copernicus (free/open incl. commercial, attribution);
     OUR DERIVED datasets (port dwell, shadow stats, transit counts,
     tank-fill readings, entity timelines) — our own work product over
     mixed inputs; ODbL inputs taint derived DATABASES with
     share-alike, so derived-stat endpoints over aircraft positions
     carry the same mark; AIS-derived stats depend on aisstream terms.
  3. TERMS-OF-SERVICE DRAFT for API customers (attribution
     passthrough, no safety-of-life use, rate/fair-use, data-as-is).
  4. STRIPE WIRING PLAN: BILLING_ENABLED + STRIPE_SECRET_KEY flip,
     key issuance flow bound to billing customer, the runtime
     compliance guard already trips /api/health if flipped early.
  RULE RESTATED: no billing wiring, no pricing enablement, no key
  sales until you approve this checklist item-by-item.

- **Historical options prices** (EOD chains + marks, ~2016→present) to backtest
  the options leg honestly. Without it, only the equity/ETF logic can be
  validated — and the options leg is the suspected main performance drag.
  Candidates: ORATS, CBOE DataShop, historicaloptiondata.com.
  **[DECIDED BY HUMAN 2026-07-04: run the FREE Databento pilot to price
  the historical pull (needs the human's free account + API key —
  signup steps delivered; set DATABENTO_KEY in the session env), and
  START THE FREE ALPACA CHAIN ARCHIVE NOW regardless (queued as its own
  [PIPELINE] PR). No spend until the pilot prices the pull.]**
  - WHAT IT UNLOCKS THAT CURRENT BACKTESTING CANNOT: backtest_v2 is
    equity/ETF OHLCV only — the options leg (CSP selection, convexity
    QQQ puts, options_scanner) is unbacktestable against ANY history:
    no historical chains, no IV, no bid/ask (REASONING STANDARD #6
    makes mid-price options backtests fiction even with marks). The leg
    currently validates only through live paper accumulation.
  - QUEUED WORK DEPENDING ON IT: KNOWN BROKEN #3 (CSP cascade — today
    verifiable live-only); open_questions "Options fill realism"
    (validating the synthetic haircut needs historical quotes); this
    entry's own origin ("suspected main performance drag" — judging
    the suspicion needs history); options entrants in the future
    strategy tournament. Regime honesty: Alpaca's free history starts
    Feb 2024 — all bull tape; a CSP strategy validated only on it is
    regime-blind (STANDARD #2).
  - BUILD-FIRST HALF (free, queue as a [PIPELINE] item): start
    archiving full Alpaca option chains for our universe DAILY now
    (free on paper accounts, feed=indicative — LABELED indicative, not
    NBBO). Forward-only: it can never recover 2016-2023. Every day not
    archiving is history permanently lost.
  - PRICES (verified from vendor pages 2026-07-04):
    ThetaData $40/$80/$160 per mo (Value 4y / Standard 8y / Pro 12y
    history, real NBBO; one-shot: 1-2 months of Pro + bulk download ≈
    $160-320 total for 2014→present — retention-after-cancel terms
    unverified, confirm before relying on this path); ORATS $99/mo
    delayed BUT 20k req/mo makes a 100-underlying 10y pull ~13 months
    of quota — effectively unfit; Polygon $29-199/mo (quotes only on
    upper tiers, short history on lower); Cboe DataShop quote-only
    (sales contact); historicaloptiondata.com ONE-OFF: Level 2 (bid/
    ask + greeks + IV) 24y $1,495, 5y $945; Databento OPRA usage-based
    with $125 free signup credits (1-min NBBO back to 2013-04),
    business-friendly license, exact cost only visible in-portal.
  - RECOMMENDATION (ranked): (1) run the FREE Databento pilot — use
    the $125 credits to price + pull a closing-minute NBBO slice for
    the ~100-underlying CSP universe 2016→present; if the full pull
    quotes under ~$1,500, it is the highest-integrity buy. (2) else
    historicaloptiondata.com L2 24y one-off $1,495 (single EOD
    snapshot quality, all regimes to 2002; confirm internal-business
    use by email). (3) budget option: ThetaData Pro churn ~$160-320
    if retention-after-cancel is confirmed in their terms. Start the
    free Alpaca archive regardless of which (or none) you pick.
  - **PILOT EXECUTED 2026-07-04 (human provided the key; priced live
    via metadata.get_cost — free calls, $0 spent): VERDICT GO.**
    OPRA.PILLAR cbbo-1m (1-min consolidated BBO ≈ NBBO class)
    confirmed back to 2013-04-01 FROM THE API — all regimes including
    2015-08, 2018-Q4, 2020-03, 2022. Measured per-day closing-1-min
    costs: SPY.OPT $0.0129 (largest chain in existence), AAPL.OPT
    $0.0035, F.OPT $0.0019; batching is cost-neutral (3-symbol batch
    priced ≈ sum). Universe extrapolation (5 SPY-class + 15
    AAPL-class + 80 F-class ≈ $0.28/day): 2016→present ≈ **$740**,
    full 2013→present ≈ **$930** — comfortably under the $1,500 line
    even at 2x estimation error. statistics schema (OI/settlement) is
    pricey ($0.36/day for SPY alone) — sample it, don't bulk-pull.
    STAGED PLAN needing NO new money to start: the $125 free credits
    cover ~450 universe-days ≈ pull 2016–2017 first, validate quality
    against known prices, THEN the human green-lights the remaining
    ~$600 for 2018→present. BLOCKED ON: DATABENTO_KEY added to the
    Claude Code session environment (key exists, human has it) + the
    human's go for spend beyond the free credits after the validation
    stage.
  - **QUALITY VALIDATION EXECUTED 2026-07-05 (~$0.30 of credits):
    9 stratified days across 2016–2017 (quarterly + the 2016-01-20
    selloff, Brexit 2016-06-24, election 2016-11-09), 10-underlying
    mix, full closing-window chains (~840k quote rows, ~12k contracts
    per day). RESULTS: ZERO crossed quotes across every row; 13–16%
    zero-bid rows (real deep-OTM market structure, not corruption);
    relative spreads median 2.8–5.6% widening exactly on event days
    (Brexit/election p90 40–47%) — REASONING STANDARD #6's real-cost
    data, not mid-price fiction; put-call parity internal-consistency
    on SPY election day: implied spot 215.5–216.5 across 33 strike
    pairs with textbook American-option drift, and the cleanest pair
    implies 216.50 = SPY's actual close. VERDICT: data quality
    VALIDATED — the ~$600 full-history decision is now purely a
    budget call. ONE ENGINEERING PREREQUISITE before the full pull:
    durable storage (the full slice is ~5GB — too big for git,
    sessions are ephemeral; options: Railway volume via an upload
    path, or confirm Databento's re-download terms so the license
    IS the storage). Deliberately pulled a SAMPLE, not the full
    2016–2017, to avoid burning credits into an ephemeral container.**

- **[APPROVED BY HUMAN 2026-07-03 — queued as next [REPAIR], see open_questions #7]**
  **Persist the max-drawdown high-water mark** (`state.equityPeak`,
  server/bot.ts:359/862/2482): in-memory only today, so every
  deploy/restart re-bases the drawdown kill switch from current equity —
  frequent autonomous deploys silently defang it. Proposal: save/restore
  equityPeak via the existing /data/voltrade state files. Touches frozen
  kill-switch machinery -> needs explicit human approval (this entry).
  Evidence: /api/health shows equityPeak 0 after today's deploys.
- **[APPROVED BY HUMAN 2026-07-04 — option (d), the token-gated
  read-only /api/diag route; token generated and handed to the human
  for Railway + session-env; route ships as its own code PR with the
  sanitizer test.] Read-only diagnostics access for autonomous
  sessions — ANALYSIS DELIVERED 2026-07-04.** WHAT IS GATED TODAY: /api/bot/audit,
  /positions, /performance, /api/daemon/health, /api/bot/ml-status,
  /api/monitoring/* — all requireOwner (session cookie must belong to
  OWNER_EMAIL; auth.ts, frozen). Sessions cannot verify KNOWN BROKEN
  #3/#4 (CSP fills firing? feedback accumulating? retrain green?) from
  outside. FOUR OPTIONS, RISK-ASSESSED:
  (a) STATUS QUO — human pastes JSON on request. Zero new risk; blocks
      routine self-diagnosis; scales badly at 8 runs/day.
  (b) Token path inside auth.ts — touches the FROZEN file; highest
      regression risk in the most sensitive module; no advantage over
      (d); not recommended.
  (c) Nightly sanitized snapshot committed to the repo (repo verified
      PRIVATE 2026-07-04) — zero new attack surface, but up to 24h
      stale, bloats git history permanently, and a future
      repo-visibility change would silently expose all history.
      Viable fallback.
  (d) RECOMMENDED: scoped read-only route in routes.ts (auth.ts
      untouched): GET /api/diag/* gated by a DIAG_TOKEN env var,
      HARD WHITELIST only — audit-log tail, ml-status, daemon health,
      positions SUMMARY (counts/exposure) — plus a sanitizer test
      pinning that responses never contain key-like strings, user
      emails, or env contents. GRANT MECHANICS: you set DIAG_TOKEN in
      Railway AND in the Claude Code environment settings; sessions
      curl the prod endpoint. RISK IF LEAKED: reader sees paper
      positions/P&L/audit entries/ML metrics — strategy-IP disclosure
      on a PAPER account; NO order placement (read-only), NO Alpaca
      keys, NO user data (whitelist excludes the auth db), NO billing.
      Rotation = change the env var. HONESTY NOTE: this deliberately
      routes around the owner gate whose intent auth.ts encodes — which
      is exactly why it ships only on your explicit approval, never as
      an autonomous change.

- **[APPROVED BY HUMAN 2026-07-03 — applied same message]** Constitutional
  amendments A1-A3 + STARVED metric (PROMPTS.md Section A): SPINOUT-READY
  DATA LAYER, RAW-DATA vs SIGNALS surface rules, [PRODUCT] session tag,
  starvation signal in HEALTH OF THE LOOP. Proposal and approval recorded
  here per the amendment rule; applied in the same PR.
- **[DONE BY HUMAN 2026-07-04 — VERIFIED LIVE same day]**
  ~~aisstream.io API key~~ — set in Railway as AISSTREAM_KEY (exact
  name the code reads); prod verified streaming: enabled:true, 1,838
  vessels in a continental-US probe, registry status "live". The
  "off/awaiting key" the human saw was a pre-restart tab — env vars
  read at boot; the v1.0.79 version-skew guard now tells stale tabs to
  reload.

- **[APPROVED BY HUMAN 2026-07-03 — applied same message]** Constitutional
  amendment: PROMOTION RULES gain rule 6 (visual verification) — client/
  PRs must run the DESIGN.md harness at 390/768/1440 and self-review
  screenshots before opening. DESIGN.md + scripts/visual_check.mjs are the
  standard and its enforcement. Proposal+approval recorded here per the
  amendment rule.

- **[APPROVED BY HUMAN 2026-07-03 — applied same message]** Constitutional
  amendment: EDGE DOCTRINE gains the BUILD-FIRST RULE (paid is the last
  resort; 4-step free-alternative assessment; honesty clause; every spend
  proposal must attach the analysis). Also DESIGN.md gains the PERFORMANCE
  BUDGET + FEATURE COMPLETENESS CHECKLIST sections. Bookkept per the
  amendment rule.
- **[DONE BY HUMAN 2026-07-03 — verification NEGATIVE, see below]**
  OpenSky free account ($0): credentials set in Railway as
  OPENSKY_CLIENT_ID / OPENSKY_CLIENT_SECRET. Same-day verification:
  production STILL serves from community fallbacks — 6+ fresh-bbox
  probes spanning ~30 minutes (longer than the 15-min max backoff, so at
  least one live OpenSky attempt was guaranteed inside the window) all
  returned adsb.lol or airplanes.live; OpenSky never served a request.
  The API itself is up (HTTP 200 anonymously from a non-Railway
  network), so the pre-credentials Railway egress rejection appears to
  persist with OAuth. Not distinguishable from outside: (a) IP-level
  block also covering the auth endpoint, (b) states/all rejecting
  Railway even authenticated, or (c) service never restarted after the
  env vars were set. Railway deploy logs disambiguate — look for
  "[datacore] opensky auth:" lines (token fetch failing) around aircraft
  requests; if no restart happened since setting the vars, redeploy once
  and re-check. MOOT UNTIL THE LICENSING DECISION BELOW: the terms
  analysis means OpenSky should not be our primary even if it worked.
- **[APPROVED BY HUMAN 2026-07-03 — applied same message]** Constitutional
  amendment: CLAUDE.md KNOWN STATE gains the USAGE-CALIBRATION LOOP note
  (usage-screenshot readings → research/usage_log.md; 2+ consecutive
  weekly readings <50% with nonzero queue → recommend slot adds; ≥90% →
  recommend drops per the established drop order). usage_log.md carries
  the A5 schedule reference (8-run menu, drop/add order, STARVED valve)
  and the canonical voltrade-weekly-review routine prompt. Gmail
  connector verified draft-only (no send) — weekly email lands in
  Drafts; routine-context availability unverifiable until the first
  Sunday run. Bookkept per the amendment rule.

- **⚠ FLAGGED CONSTRAINT — aircraft-feed licensing (MONETIZATION
  TRIPWIRE, filed 2026-07-03; corrected same day per human: the site is a
  proof of concept with NO paid product today — billing code exists but
  nothing is charged). Analysis only; NO provider or code change made.**
  Provider terms, assessed against the corrected commercial status:
  - **OpenSky Network** (current primary): the license grants use "solely
    for the purpose of non-profit research and non-profit education."
    As a no-revenue POC we are plausibly inside "non-profit research"
    on the commercial clause — but a second, independent clause still
    fires TODAY regardless of revenue: "Use of the REST API in any
    operational capacity — including integration into a live product,
    service, or automated system (even if only internal) — requires a
    previous written agreement, even for non-profit or governmental
    entities." Our bot + site + automated archive are exactly that. So
    OpenSky technically requires a written agreement even for the POC
    (contact@opensky-network.org — plausibly granted free for research).
    The new free account raises rate limits but does not change this.
  - **adsb.lol** (fallback 1): **ODbL 1.0**, "available to everyone" —
    compatible today AND after monetization, with attribution (already
    shown on the map) and share-alike on derivative *databases*. The
    only provider that survives monetization unchanged. Spinout note:
    the position archive is a derivative database — redistribution/sale
    of archive-derived products built on adsb.lol data must carry ODbL
    attribution + share-alike (internal display/signals are fine).
  - **airplanes.live** (fallback 2): free REST API is "Non-Commercial
    Use" (educational, 1 req/s, no SLA) — **compatible with today's
    no-revenue POC**, incompatible the day the site charges anyone
    (commercial access exists via direct arrangement:
    airplanes.live/commercial-use/, RapidAPI "coming soon").
  - **DECIDED BY HUMAN 2026-07-03 (executed same day, v1.0.45):**
    OpenSky dropped from the runtime chain — adsb.lol primary,
    airplanes.live fallback; removes the ~12s dead OpenSky attempt on
    every fresh viewport. The human has emailed
    contact@opensky-network.org requesting a research agreement.
    IF/WHEN GRANTED: reinstate OpenSky in the chain (git history of
    v1.0.43 has the OAuth + states/all implementation to restore) AND
    re-verify Railway connectivity at that time — the egress block is
    independent of the license and may still bite. THE TRIPWIRE stands:
    before enabling billing, ads, or any paid feature, re-run this
    compliance check — at that moment airplanes.live must be dropped or
    upgraded to a commercial arrangement, and adsb.lol becomes the only
    lawful free provider.
  - Sources: opensky-network.org/about/terms-of-use (§1 LICENSE, §3(vi));
    adsb.lol/docs/open-data/api (ODbL 1.0) + adsb.lol privacy-license;
    airplanes.live/api-guide + airplanes.live/commercial-use.

- **FlightAware AeroAPI / FAA SWIM (filed flight plans + routes) — PRICED,
  deferred.** BUILD-FIRST analysis attached per the new rule: (1) raw
  material (filed plans) is NOT freely receivable; (2) accumulation
  substitute BUILT: our own position archive gives track history free;
  (3) inference substitute BUILT: destination PREDICTION from trajectory +
  per-tail route history, labeled predicted, self-scored against observed
  landings; (4) what paid adds over our free version: filed (not
  predicted) routes, ETAs, schedules, pre-departure intent. Price: AeroAPI
  personal tier ~$100/mo class. Recommendation: defer until the predicted
  version's measured accuracy (archive self-scoring) proves insufficient
  for a gated signal.
- **Position-archive volume watch** (standing, LIVE 2026-07-03 — see
  experiments.md): 30-min sample interval per kind, compact positional
  (not object) JSONL records, 90-day raw retention with a permanent
  rollup surviving pruning. Computed estimate at these parameters:
  aircraft ~40MB/mo + vessels ~65MB/mo ≈ **105MB/mo combined** (math in
  `server/dataArchive.ts` header comment) — this is the actual design
  figure, not a guess; revise the interval if the real
  `/api/data/archive/stats` numbers, once the deploy has run for a few
  days, come in materially higher (e.g. from aircraft/vessel counts near
  the 800/1500 per-request caps more often than assumed). FLAG HERE if
  growth trends toward Railway volume plan limits.

- **[APPROVED BY HUMAN 2026-07-03 — applied same message]** Constitutional
  amendments batch: (1) SESSION BUDGET replaced by the PRODUCTIVE
  FALL-THROUGH ladder (queued item -> filed-artifact research -> decision
  request never idles a session; hard limits preserved: own PR/log per
  action, read-before-write, anti-churn, [NO-ACTION] only on empty
  queue); (2) DEAD CODE POLICY (stale code is debt; same-PR removal;
  likely-returner adapters only with zero runtime cost + review-by date +
  open_questions log; 30-day staleness audit as fall-through action);
  (3) CONSTITUTIONAL HYGIENE (monthly rule audit files consolidation
  proposals here, never self-applies; live conflicts resolved by GOAL
  priority order and filed). Bookkept per the amendment rule.

- **⚖ FIRST CONSTITUTIONAL AUDIT (2026-07-03) — [APPROVED BY HUMAN
  2026-07-04: Findings 1 AND 2, shipped as one docs PR same day. F1 =
  STANDING BEHAVIORS section added, rule paragraphs moved verbatim out
  of KNOWN STATE. F2 = delivered via the AUDIT CYCLE register in
  experiments.md (the AUDIT CYCLE proposal's concrete superseding form —
  one register, not two).]**
  - **Finding 1 — rules living in KNOWN STATE.** KNOWN STATE now hosts
    four standing behavior RULES (SPINOUT-READY DATA LAYER, RAW-vs-SIGNAL
    surface rules, USAGE-CALIBRATION LOOP, MONETIZATION TRIPWIRE, plus
    the product-routine mandate). The self-edit rule permits sessions to
    append "factual updates" to KNOWN STATE — rules living there blur
    the facts-vs-rules boundary the amendment lockdown depends on.
    PROPOSAL: add a "STANDING BEHAVIORS (each human-approved, dated)"
    section; MOVE those rule paragraphs there verbatim (zero wording
    change); KNOWN STATE returns to pure facts. Preserves: all rule
    text. Drops: nothing. Resolves: self-edit ambiguity.
  - **Finding 2 — two identical-cadence periodic audits.** DEAD CODE
    POLICY's staleness audit and CONSTITUTIONAL HYGIENE's rule audit
    share trigger (fall-through research tier, 30+ days) but live in
    separate sections; the December market_calendar year-add is a third
    scattered periodic duty. PROPOSAL: one "PERIODIC AUDITS" register
    (subsection of SESSION BUDGET) listing all recurring obligations +
    cadences + last-run dates, each pointing at its governing section.
    Preserves: every audit's content/cadence. Drops: nothing. Resolves:
    scatter — future sessions check one place.
  - Factual drift found and corrected directly in this PR (allowed as
    factual update, not part of the proposal): KNOWN STATE + CODEBASE
    MAP still called backtest.py a STUB hours after the engine was
    rebuilt; both now state the rebuilt reality.
  - Interactions checked, no action needed: STARVED's definition
    survives fall-through unchanged (capacity exhausted with queue
    nonzero); BUILD-FIRST sits correctly as an EDGE DOCTRINE subsection;
    the tripwire rule vs. the FLAGGED CONSTRAINT entry are rule vs.
    decision-record, cross-referenced, not redundant.

- **[APPROVED BY HUMAN 2026-07-03 — applied same message]** USAGE-
  CALIBRATION LOOP switched to DAILY AGGRESSIVE MODE: usage-screenshot
  readings get a SAME-DAY recommendation (headroom → name exact slots to
  add NOW up to the platform cap; near limits → throttle fall-through
  first, then drop order); aggressive-add bias while weekly <50%. New
  voltrade-usage-check routine (DAILY 21:30 ET) — canonical prompt in
  usage_log.md; description carries the ~2026-07-24 revisit note (drop
  back to weekly once readings flatten). Gmail re-verified this session:
  connector remains DRAFT-ONLY (no send tool exists) — daily nudge lands
  in Drafts; the Claude Code Notifications tab is the recommended
  completion signal instead. Bookkept per the amendment rule.

- **⚖ CONSOLIDATION PROPOSAL — AUDIT CYCLE (filed 2026-07-03; [APPROVED
  BY HUMAN 2026-07-04 — applied same day, one docs PR with audit
  Findings 1+2: clause in SESSION BUDGET, register at top of
  experiments.md, both trigger sentences trimmed to pointers]).** Three periodic hygiene duties live in three places:
  (1) DEAD CODE POLICY's staleness sweep ("fall-through reaches the
  research tier and the codebase hasn't had a staleness audit in 30+
  days"); (2) CONSTITUTIONAL HYGIENE's rule audit ("monthly, or as a
  fall-through action when 30+ days since last review"); (3) the
  December market_calendar year-add (FROZEN PATHS exception + KNOWN
  STATE note). PROPOSED AFTER-TEXT — one clause appended to SESSION
  BUDGET, replacing neither policy body (only the scattered TRIGGERS):
  "AUDIT CYCLE: when a session's fall-through reaches the research
  tier, check the audit register at the top of research/experiments.md
  {audit · cadence · last run}: staleness audit (code/deps/config/
  expired adapters — 30d; DEAD CODE POLICY governs), constitutional
  audit (rules — 30d; CONSTITUTIONAL HYGIENE governs), market_calendar
  year-add (December; FROZEN PATHS exception governs). Run the most
  overdue one and update the register." Preserves: every cadence and
  both policy bodies verbatim. Drops: nothing. Resolves: three triggers
  nobody checks in one place; also supersedes the first audit's
  Finding-2 sketch (PERIODIC AUDITS register) with a concrete location.
  If approved: one docs PR adds the clause + the register, and trims
  the two in-place trigger sentences to point at it.

- **Satellite AIS (mid-ocean vessel coverage) — PRICED, deferred; filed
  per the ships directive 2026-07-04.** Verified: our aisstream.io
  subscription is already configured GLOBAL (BoundingBoxes ±90/±180),
  so the coverage gap is physical, not configuration — aisstream
  aggregates TERRESTRIAL receivers, which see ~40-60nm offshore; ships
  mid-ocean go dark between coasts. BUILD-FIRST analysis: (1) the raw
  material (satellite AIS downlink) is inaccessible free — genuinely
  paid class per the EDGE DOCTRINE; (2) accumulation helps at the
  EDGES: our archive records port arrivals/departures + coastal
  transits, which is where R2 transit-analytics value concentrates;
  (3) inference substitute EXISTS for specific questions: a ship that
  left port A heading for port B (destination field) can be
  dead-reckoned mid-ocean and confirmed on coastal reacquisition —
  label as predicted track, never ground truth. (4) Paid adds: true
  mid-ocean positions. Vendors: Spire Maritime, Kpler/exactEarth,
  ORBCOMM — pricing is quote-only, entry commonly $500+/mo class.
  RECOMMENDATION: do not buy unless a specific gated signal needs
  mid-ocean truth (none does today; port-transit signals don't).
  **[DECLINED BY HUMAN 2026-07-04 — entry retained with this revisit
  trigger: reconsider ONLY if a gated signal specifically requires
  open-ocean coverage. Any future proposal must name that gated signal
  and show why coastal reacquisition + dead-reckoned predicted tracks
  (the free inference substitute above) fail it. Port-transit, dwell,
  and shadow-fleet statistics all live in terrestrial-coverage
  waters — none qualifies.]**

- **⚖ PROPOSAL — UNIVERSAL ARCHIVE ENVELOPE (charter directive
  2026-07-04; human approval required; nothing changed yet).** INTENT:
  every archived datum carries {timestamp UTC, source, confidence,
  geo, entity/ticker linkage, sentiment where applicable}. HONEST
  ENGINEERING CONSTRAINT: position archives are compact POSITIONAL
  records by design (~105MB/mo volume budget); repeating constant
  envelope fields on every 2-min position point would ~3x volume for
  zero information (source/confidence are constant per stream).
  PROPOSED TWO-TIER FORM: (1) DATASET-LEVEL manifests —
  datacore/manifests/{kind}.json, one envelope per stream {source,
  license, attribution, schema_version, field_map, confidence_model,
  geo_fields, entity_key (MMSI/icao24/CIK/ticker), started, cadence} —
  covers EXISTING archives retroactively without rewriting append-only
  history (manifests are new files, not edits). (2) DATUM-LEVEL where
  information actually varies per record: t (already UTC epoch
  everywhere), geo (la/lo), entity key (already present) — and
  REQUIRED first-class fields {source, confidence, entity/ticker
  linkage, sentiment where applicable} on ALL NEW pipelines
  (8-K language, jobs, patents, app ranks) from birth. MIGRATION:
  existing JSONL stays byte-stable; readers pick up field_map from
  manifests in a later refactor PR; the Everything Graph's edge
  metadata {source, confidence, first_seen, last_seen} is this same
  envelope applied to derived data. IF APPROVED: PR 1 writes manifests
  for the 5 existing streams + a test (every archive kind must have a
  manifest — enforced).

- **[DONE BY HUMAN 2026-07-04]** ~~OpenWeatherMap free API key~~ — set
  in Railway as OPENWEATHERMAP_KEY (fresh key; OWM activates within
  ~2h). Global temperature + wind field layers wired same day
  (v1.0.63): key stays server-side behind a tile proxy with shared
  cache (60-calls/min budget), "Weather data © OpenWeatherMap"
  attribution, model-derived labeling, and fresh-key-aware status
  (401 = "activating" with retry note, never an error state).
  Verification: prod probe post-deploy; if still "activating" well
  past ~2h from key creation, re-check the key value.
- **[DONE BY HUMAN 2026-07-04]** ~~NASA FIRMS MAP_KEY~~ — key set in
  Railway as `FIRMS_MAP_KEY`. **ENV-NAME MISMATCH CAUGHT SAME DAY**:
  the scaffolded implementation (v1.0.65, `server/nasaFirms.ts` —
  key-gated fetch/parse/archive, `/api/data/fires`, Environmental
  panel group) read `NASA_FIRMS_MAP_KEY`, so the set key would never
  have activated it. Fixed code-side (v1.0.68): `firmsKey()` accepts
  BOTH names — the code adapts to the action already taken, no Railway
  rename needed; regression pinned by test. Detections archive from
  day one of activation (no free history exists upstream).
  DOUBLE-BUILD NOTE (concurrent-sessions gotcha, logged honestly): an
  interactive session built a parallel FIRMS implementation while the
  routine's merged first — the duplicate was abandoned unmerged (PR
  #155 closed), the routine's implementation stands; the interactive
  session's unique salvage is this activation fix + the live-key prod
  verification.
- **HUMAN ACTION — USPTO Open Data Portal API key (STATUS 2026-07-04:
  account created, ODP form submitted, key NOT yet reached — entry
  stays open).** Research completed same day (primary sources: ODP
  FAQ, ID.me help, USPTO notices + live HTTP probes) — CLICK-BY-CLICK
  for next time:
  1. account.uspto.gov — confirm the USPTO.gov account (email + MFA).
     Note: from 2026-08-18 four extra "Open Data Portal" fields in
     account settings (account.uspto.gov/profile) become mandatory.
  2. ID.ME MUST COMPLETE FIRST — the key is gated on it. Path:
     uspto.gov → MyUSPTO → sign in → Profile → "Verify with ID.me" →
     follow prompts → "Allow" on the Authorize screen. Needs gov ID +
     SSN; outside the US it requires a video call. (This is almost
     certainly the "more verification" you hit — the form alone
     doesn't issue keys.)
  3. Then sign in at data.uspto.gov and open "Manage API Key":
     https://data.uspto.gov/apikey (reveal page: /apikey/key-reveal;
     also on the MyODP dashboard). First visit validates + links
     ID.me, then the key is view/copyable. NO approval queue is
     documented — self-serve once ID.me is verified.
  4. Gotchas: ONE key per person (ID.me↔USPTO 1:1, never
     organizational); a duplicate ID.me account blocks linking;
     replacements via data@uspto.gov; use as header `x-api-key`
     against api.uspto.gov. Legacy Developer-Hub and PatentsView keys
     are dead/incompatible — only the ODP key matters.
  KEYLESS-BYPASS VERDICT (probed live 2026-07-04): NO USPTO-native
  keyless start exists anymore — bulkdata.uspto.gov RETIRED (Apr
  2025, host dead), ODP API returns 401 without a key, the ODP web
  bulk directory needs the signed-in account, PatentsView's API is
  OFFLINE pending ODP relaunch (old keys incompatible). The only true
  keyless start is Google Patents Public Datasets on BigQuery (free
  GCP account) for HISTORICAL BACKFILL — its repo was archived
  read-only 2026-04-18, so freshness is unverified; logged as
  backfill-only, not a live feed. The live weekly pipeline (grant +
  application XML, File Wrapper continuity/assignments that BigQuery
  lacks) is designed KEY-FIRST: single-threaded (burst=1 per key —
  parallel calls trigger 429s with ~7-day lockout risk), quotas reset
  Sun 00:00 UTC, and the key is a PER-PERSON credential tied to your
  ID.me — the pipeline config treats it as such.
- **CDSE (Copernicus Data Space) credentials — NOT NEEDED, filed as
  fallback only (Sentinel-2 directive 2026-07-04 asked for exact
  signup steps).** The tank-fill pipeline runs with ZERO credentials
  (Element84 earth-search STAC + AWS Open Data public S3, live-probed;
  scripts/sentinel2_tankfill.py). IF the AWS mirror ever lags or dies,
  the CDSE path: (1) dataspace.copernicus.eu -> "Register" (free,
  email + password, instant — no identity verification); (2) sign in
  -> user menu -> Settings -> generate S3 keys for eodata, OR use the
  OData/STAC APIs with an OAuth token from the same account; (3) set
  CDSE_S3_KEY / CDSE_S3_SECRET in the SESSION environment (not Railway
  — the pipeline is script-side, never runtime). Re-verify their
  free-tier transfer quota at need per the licensing rule.

- **HUMAN ACTION (or non-proxied routine) — Australia CASA data-files
  licence check:** casa.gov.au consistently 503s via our egress proxy;
  the register CSVs are free but the applicable licence may be CC
  BY-NC (non-commercial — would exclude product use). Fetch
  https://www.casa.gov.au/aircraft/aircraft-registration/data-files-registered-aircraft
  ("Downloading and using our data files" conditions) from a normal
  network and paste the licence text into a session. Until verified,
  the aircraft-registry spine ships without Australia.

- **HUMAN ACTION — Apple Performance Partners / Enterprise Partner
  Feed enrollment (free):** sanctioned bulk feed that hedges the
  undocumented Apple RSS endpoints the app-store archiver uses
  (NEW DATA ROOTS #3); its store-linking requirement (App Store
  badges/links on the /data surface) is acceptable and noted.
- **Sensor Tower (app downloads/revenue ESTIMATES) — PRICED, not
  recommended.** ~$6K/yr entry module to ~$42K+/yr realistic.
  BUILD-FIRST: even paid data here is panel-model ESTIMATES, not
  truth; our free archiver (ranks + rating-count velocity + Apple
  top-grossing as revenue proxy) captures the testable core of the
  hypothesis. Revisit only if the free root passes gate 2 AND the
  residual specifically needs download estimates.

- **[APPROVED BY HUMAN 2026-07-04 — applied same message]** DESIGN.md
  amendment: SELF-SEE RULE — "UI changes must verify their own
  rendering: after any change to a panel or overlay, the harness
  screenshots must show ALL registered content reachable (visible or
  behind an on-screen expand control) at all three widths. A component
  that exists in code but can't be reached on screen is a failed
  build." Enforcement shipped in visual_check.mjs (SELF-SEE block) and
  proven against the actual defect by A/B (old CSS -> harness FAILS
  with "panel bottom past viewport"). Bookkept per the amendment rule.

## ANALYST CONSOLE keys (program installed 2026-07-07 by human directive — research/console_charter.md)

- **BLOCKED-FOR-MIKE — ANTHROPIC_API_KEY in Railway (the one that
  matters):** unlocks W6, the LLM analyst pane — the centerpiece of
  the console program. BUILD-FIRST analysis: there is no free
  substitute for LLM inference itself; everything AROUND it (query
  engine, tool protocol, map-command channel, UI, tests) builds
  without the key and the analyst ACTIVATES ON KEY DETECT like every
  other keyed stream. Cost estimate at hobby usage with a small model
  (Haiku-class) for the tool loop and short answers: single-digit
  $/month; heavier use or a bigger model: tens of $/month. Runtime
  half will follow the established key-gated stream pattern (presence
  check only, key never logged/echoed).
- **OPTIONAL — Google Maps Platform key (Photorealistic 3D Tiles,
  free monthly tier, card required):** cinematic city-level 3D for
  W1. NOT a blocker — MapLibre globe projection ships free without
  it. File only if the free globe feels insufficient after W1 lands.
- **OPTIONAL — Windy webcams API key (free tier):** W7 public-camera
  layer candidate alongside state DOT traffic cams (public).
  Licensing research files BEFORE any code per the charter.

- **LICENSING RECORD — CelesTrak GP data (verified 2026-07-07, W2):**
  PROCEED. No terms of use or redistribution restriction exists;
  usage-policy.php frames limits as resource courtesy: "Only download
  the data you need... For GP data, updates are once every 2 hours."
  GP docs FAQ acknowledges commercial users without prohibition
  ("...including those who profit from our efforts..."). Enforcement
  is technical (HTTP 403/firewall for abusers; M2M software must stop
  on any non-200 — implemented in server/satellites.ts). We fetch
  from CelesTrak only (NOT Space-Track — their user agreement does
  not bind us). Attribution carried on every surface though not
  formally required. DECENT-CITIZEN NOTE for the human: CelesTrak is
  a non-profit; a donation would be appropriate if the satellite
  layer ever goes commercial. TIME-SENSITIVE FACT: 5-digit catalog
  numbers exhaust ~2026-07-12; our OMM JSON format is the mandated
  migration (TLE format breaks at 69999).

- **DECISION FOR MIKE — CelesTrak unreachable from Railway (R17
  evidence: connect timeouts on both hosts = IP-range firewall).**
  Recommended: approve a SESSION-RELAY INGEST route — token-gated POST
  (new dedicated INGEST token env var, separate from DIAG_TOKEN whose
  surface stays read-only), payloads validated through the existing
  parseGp path before archiving. Sessions fetch CelesTrak (works from
  session egress, proven) 1-2x/day and relay. Effect: satellite layer
  + orbit-history archive work at daily resolution; zero cost; the
  in-place 6h poller self-heals if Railway's range is ever unblocked.
  Alternative if you prefer no write surface: the satellites layer
  ships browser-fetch-only (display works, no archive accumulation).
  Say "approved: satellite relay" (and add SAT_INGEST_TOKEN to
  Railway + the session env) or pick the alternative.

## GOOGLE MAPS PLATFORM API DECISIONS (recorded 2026-07-07 — human review of the 32-API library)

Map Tiles enabled (GOOGLE_MAPS_API_KEY in Railway). 2025 pricing:
per-API free monthly allotments (Essentials 10k / Pro 5k / Enterprise
1k). I cannot enable APIs from the session — enabling is a GCP console/
IAM action on the human's Google account; the key only CALLS enabled
APIs. So all are human-toggle actions; recommendations:

- **Air Quality API — ENABLE (build shipped v1.0.202, #353).** Pro
  tier 5,000 free/mo. 70+ pollutants incl PM2.5/NO2, 500m, 100+
  countries, current + 30d history + heatmap tiles. Real stream +
  industrial-activity-proxy hypothesis (open_questions). The stream is
  built and waiting; ACTION: enable "Air Quality API" in the console.
- **Solar API — HOLD, do not enable.** Building-level rooftop solar
  potential — a real-estate/quoting tool; no market signal at building
  granularity. No build → idle meter. Revisit only with a hypothesis.
- **Pollen API — HOLD (ideas backlog).** Tree/grass/weed forecast, 65
  countries. Tenuous market link (allergy retail? crop stress?). No
  build; parked, not enabled.
- **Aerial View API — HOLD.** Cinematic address flyover VIDEO. Pure
  visual polish, zero data/signal. Maybe a site-card touch someday;
  not worth a meter now.
- **Weather API (if present) — SKIP.** We already have NOAA/NWS + OWM +
  CPC free; Google's adds nothing.
DISCIPLINE (human rule): don't enable a meter with no build. Only Air
Quality has one. Heatmap-tile client OVERLAY for Air Quality is a
possible LATER map-enhancement PR (the tile URL template is in
server/airQuality.ts heatmapTileUrl) — build only if wanted.

## GOOGLE MAPS PLATFORM — FULL 8-API DECISION (2026-07-07, human enabled most APIs in project be823885)

$300 trial; per-API free monthly allotments. One GOOGLE_MAPS_API_KEY
covers all API-key-based ones (Air Quality, Weather, Solar, Pollen,
Elevation, Datasets REST; 3D Tiles/Aerial View use the same key +
session token). VERDICT: only ONE earns a stream; the rest are
duplicates of free data we already own, Google's processed products
(resell-not-build), or map polish with cost risk. "Enabled but
uncalled = $0" — leave the dormant ones dormant.

- Air Quality — ✅ KEEP (built #353). NO2/PM2.5 combustion/activity
  proxy; 30d history to accumulate; 5k/mo free; budget-guarded to
  ~128 calls/day. **NOTE: prod still reads awaiting_enable/
  PERMISSION_DENIED after the human's enable — check (a) the key
  belongs to project be823885, (b) the key's API restrictions
  allow-list includes Air Quality API.**
- Weather — ❌ DUPLICATE. We have NWS + OWM + NOAA CPC (10yr) free;
  Google history is only 24h (no archive value). 10k/mo free. Dormant.
- Solar — ❌ DISTRACTION. Google's processed product (resell, ToS-
  restricted, anti-moat); STATIC per-building lookup, no tradeable
  time-series; the only legit angle (regional adoption) is dominated
  by free EIA-861 distributed-solar. Not worth any effort. Dormant.
- Pollen — ⏸️ DORMANT (ideas backlog; tenuous market link).
- Aerial View — ⏸️ DORMANT (video, zero data, billed per generation —
  never call at volume).
- Map Tiles (Photorealistic 3D) — ⚠️ ENHANCEMENT, do NOT wire by
  default. Session/tile-billed = THE cost risk. Globe uses free
  MapLibre. Possible future cinematic-city option only.
- Maps Elevation — ❌ DUPLICATE (we have GLO-30 elevation + hillshade
  free, shipped). Dormant.
- Maps Datasets — ❌ NOT A DATA SOURCE (hosting/rendering our own geo
  on Google infra; against build-it/spinout doctrine; SCALE S2 builds
  our own tiling). Dormant.

COST RISK: real risk lives only in 3D Tiles (session-billed), Solar
Data Layers (1k/mo then billed), Aerial View (per-video) — none of
which we call. Only Air Quality calls, budget-guarded → expected
spend $0. RULE HOLDS: only wire what has a budget guard.

GCP BUDGET ALERT (recommended to the human): expected spend is $0, so
the alert is a TRIPWIRE not a budget — set $20/month with alerts at
50% ($10) and 100% ($20) + the trial-credit-consumption alert. First
stray dollar = something mis-wired or a guard failed. Set BEFORE
volume.

## PRODUCT THESIS — the GROUND-TRUTH LAYER FOR AI AGENTS (filed 2026-07-12, human-originated)

CONTEXT: filed at the human's direction after a session comparing us to
Palantir. The human's framing: AI has made *writing software* nearly free,
so undifferentiated software is dead — the durable value is (1) proprietary
data (especially data with time in it, which no model can regenerate at any
budget), and (2) closing the two gaps AI did NOT close — the SPECIFICATION
gap (telling the machine what you want and getting it) and the VERIFICATION
gap (knowing the answer is actually true, not plausible). Palantir's bet is
"we make generic AI specific to your org via the Ontology"; the honest
counter is that its moat is lock-in + accreditation, NOT data — because if
AI makes integration 10x cheaper it makes *replacing* Palantir 10x cheaper
too. Our structurally stronger position: we OWN an archive nobody else
recorded, and cheap software-building is the shovel, not the product.

THE THESIS (a new customer segment, not yet in this file): every company
deploying AI agents hits the verification gap — the agent confabulates
because it has no access to verified physical reality. Sell our archive +
validated signals as an API whose consumer is *someone else's AI agent*:
verified vessel/aircraft/train positions, facility activity, filings-derived
events — each carrying provenance, freshness, and confidence. We are not
competing in the flood of AI apps; we are the thing that makes them stop
lying. This GROWS the addressable buyer pool from "quant funds" to "anyone
whose agent needs to know what is physically true." It is the same /api/v1
product already planned — the AI wave just widens who buys it.

WHY IT FITS THE CONSTITUTION: this is priorities-3/4 platform work and it
leans directly on the two-sided HONESTY METRIC — claimed-vs-ground-truth is
already our machinery. "Every number carries provenance/freshness/confidence"
(PREMIUM EXPERIENCE STANDARD, Amendment 5) stops being internal hygiene and
becomes the selling proposition in a market drowning in AI-generated
plausible-looking data. Nothing here needs new paid access — build-first
holds; the raw material is already archived.

FIRST STEP SHIPPED THIS SESSION (v1.0.284): `/api/v1/agent-tools` —
`agentToolSpec()` in server/apiProduct.ts renders the LIVE API as JSON-Schema
function-calling tool definitions (drop-in for Anthropic tool use / OpenAI
functions / an MCP server), derived from the SAME live endpoint set as
apiMeta() so gated signals can never leak in, each tool naming the
license_marks key(s) of what it returns so provenance travels into the
agent's context. Public docs endpoint (like /meta); the calls behind each
tool still require an x-api-key. apiMeta() now points at it (`agent_tools`).
This is the honest v0 of the surface — real, small, gated-signal-safe.

NEXT STEPS FOR A FUTURE PRODUCT SESSION (each its own PR, none billing):
1. A hosted MCP server (or a published `.well-known/` manifest) so an agent
   can auto-discover and mount the tools — turn agent-tools into an actual
   connectable server, not just a spec document.
2. A `/developers` "Use with your AI agent" section (P2 PREMIUM territory):
   copy-paste the tool spec into Claude/ChatGPT, one live example of an
   agent answering a physical-world question with our provenance cited.
3. As each SIGNAL passes ladder gate 2, expose it as a NEW agent tool with
   its confidence attached — the confidence field IS the product for an
   agent deciding whether to trust the number.

HUMAN DECISION REQUESTED (weekly review): is "ground-truth layer for AI
agents" an official positioning line for the platform (affects how
/developers and /pricing are written, and which signals get prioritized to
gate 2)? If yes, it should be reconciled into VISION.md/GIP.md as a named
customer segment. Until then it lives here as a filed thesis with a shipped
v0, not a silent pivot.
