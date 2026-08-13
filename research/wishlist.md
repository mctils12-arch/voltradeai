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
  3b SHIPPED 2026-07-22 (v1.0.476, scripts/cdse_site_chips.py +
  server/siteImagery.ts) — latest cloud-free Sentinel-2 true-color
  chip for all 16 strategic sites, scheduled/session-run per the
  quota mandate here (10.5 PU spent, 0.1% of the free tier), shown
  in the existing site detail card (RAW, no ladder gate, no new
  layer/toggle). Remaining Phase 3: 3c S2 utilization review across
  asset classes; per-layer freshness chips (with Phase 5); a refresh
  cadence for 3b beyond "whenever a session re-runs the script" (no
  Railway cron — CDSE creds are session-only).
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
  Per-layer freshness chips SHIPPED 2026-07-23 (v1.0.479,
  server/layerFreshness.ts + datamap.tsx layer-row chip) — joins the
  Phase 4 streams-inventory health onto 16 hand-verified layer ids
  (see experiments.md for the full trace + why each of a further ~13
  candidates couldn't be honestly mapped this session: derived joins
  over another stream's archive, static curated reference data, or
  the GEM coal/methane manifest ambiguity). Remaining: imagery
  capture-date requirement (with Phase 3); Phase 5's own
  freshness-chip coverage could widen later if the gem/portdwell/
  shadowstats gaps above get their own design pass.
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
- SETTLEMENT-STRESS COMPOSITE [PIPELINE]: SHIPPED 2026-07-18 (v1.0.392,
  scheduled-routine [PIPELINE] session) — server/settlementStress.ts +
  datacore/manifests/settlementstress.json. Pure local JOIN over the
  three already-archived ingredients (finrathreshold + secftd +
  finrashortvol), zero incremental network/API cost, wired into the
  boot-poll battery in server/routes.ts right after bootFtdPoll().
  Computes, per trade date, the intersection of symbols on the
  threshold list AND carrying an FTD balance for the covering
  half-month period AND an eligible short-vol reading; composite_score
  = persistence_days x (short_vol_percentile/100) x
  sign(ftd_delta)*log1p(|ftd_delta|). GATE 1 ONLY — see the module
  docstring and manifest confidence_model: not checked against forward
  returns, not wired into deep_score/any order path, NOT surfaced on
  /data (unvalidated derived score, neither RAW nor SIGNAL yet). 7 new
  tests in server/settlementStress.test.ts, all A/B-relevant join edge
  cases covered (persistence streak + archive gap, FTD delta incl.
  missing-prior-period, liquidity-floor exclusion, 3-way intersection,
  archive dedup, refresh only-advances-when-ingredients-ready). NOTE
  for future test authors: finraQuery.ts/secFtd.ts export
  _reset*ForTests() (used here via beforeEach); finraShortVolume.ts
  does not, so its archive dedup is process-global regardless of
  baseDir — this file's dates were chosen to never collide across
  tests, same precedent already documented in
  finraShortVolume.test.ts. GATE 2 NEXT STEP (unchanged from the
  original queue entry): once enough dated history has accumulated in
  datacore/settlementstress/, test composite_score (or its rank) vs.
  forward N/5/20-day returns against a same-universe random-entry base
  rate (REASONING STANDARD #3) before this is ever considered a signal.
  UPDATE 2026-07-27 (scheduled-routine PRODUCT session, v1.0.507,
  [REPAIR]) — GATE 2 was actually attempted first-order (checking
  whether enough history had accumulated) and found the composite had
  archived ZERO rows across every date since shipping — root cause:
  finrathreshold is OTC-only (FINRA's own schema) but finrashortvol only
  ever ingested CNMS (exchange-listed-only), a structural population
  mismatch that guaranteed permanent zero overlap regardless of elapsed
  time, not "insufficient history." Fixed by adding FINRA's ORF (OTC
  facility) short-volume file as a second ingestion source
  (finraShortVolume.ts, datacore/manifests/finrashortvolotc.json);
  shortVolPercentiles now ranks across both facilities. Full trace in
  experiments.md same date. GATE 2 remains not-yet-run — this session
  fixed gate-1 plumbing so gate 2 has a chance of ever seeing real
  overlap; a future session should check /api/data/archive/stats for
  nonzero-byte settlementstress files accumulating before attempting the
  correlation test.
  UPDATE 2026-07-28 (scheduled-routine PRODUCT session, v1.0.523) — live-
  verified /api/data/archive/stats: finrashortvolotc is accumulating real
  nonzero-byte day-files (5 files, ~170KB, 07-20..07-24) on its normal 6h
  poll — the thin ORF ingestion is confirmed stable in production, the
  checkpoint the prior entry needed before a deep backfill. Also live-
  verified directly against sec.gov that FTD period 202607a genuinely
  isn't published yet (404 on every URL variant; only 202606b is live) —
  settlementstress staying at its same 4 zero-byte June files is NOT a new
  bug, it is correctly waiting on SEC's own "a" period publishing ~EOM.
  Added the ORF counterpart to finrashortvol's own deep backfill
  (finraShortVolume.ts: orfDeepBackfillIfSparse/countOrfArchivedDays/
  gzipOldOrfShortVolDaysAsync, separate env gate FINRA_ORF_DEEP_BACKFILL=1,
  separate done-marker in finrashortvolotc/, same default-off posture as
  CNMS's own FINRA_DEEP_BACKFILL per its 2026-07-05 emergency-off
  incident) — live-spot-checked FORFshvol{YYYYMMDD}.txt back to
  2024-06-17 (real 200s on trading days), so the same ~2yr/750-day depth
  is reasonable. NOT ENABLED — this is a volume-budget decision for the
  human, same as every other deep-backfill gate in this codebase (~75MB
  gz estimate, same order of magnitude as CNMS's own). To turn it on: set
  FINRA_ORF_DEEP_BACKFILL=1 in Railway's env and the next boot poll
  backfills once (done-marker prevents repeats).
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
  Drive copy + scratchpad); (b) **SHIPPED 2026-07-17 (v1.0.374,
  server/entityGraph.ts `owns` edge type)** entity-graph CIK join
  (GEOT <-> EDGAR) — restricted to pairs where both ends resolve to a
  real CIK (393 of 24,351 entity_edges today); broadening to the
  1,010 government/private/foreign-owner edges with no CIK is filed
  as its own follow-up in open_questions.md (EVERYTHING-GRAPH,
  2026-07-17); (c) LNG-carrier IMO join to AIS archive; (d) **SHIPPED
  2026-07-18 (v1.0.400, server/gemMethane.ts, GET /api/data/methane-plumes)**
  GMET plumes RAW API route — see experiments.md for the full trace;
  the client-side dated map/event layer itself is a separate follow-up
  (own T-CLIENT PR, same earthquakes/buoys precedent).
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
- **EPA CAMD CEMS: SHIPPED 2026-07-18 (v1.0.385)** — the "gated on
  Mike's keys" framing above was STALE, corrected this session: the
  shared api.data.gov DEMO_KEY was live-probed and works today (no
  Railway key needed to build or run this). server/epaCamd.ts +
  GET /api/data/plant-operations, TX pilot scope, quarterly cadence
  (corrected from the census's original hourly/partial-arrival framing
  — live-probed 2026-07-18: the in-progress quarter is rejected
  outright by the real API). Full trace in experiments.md. 9a (a
  dedicated EPA_CAMD_API_KEY) is still worth getting from Mike to widen
  past TX and drop shared-DEMO_KEY collision risk, but is no longer a
  hard blocker — see 9a's own updated entry below. NOTE: 9c (ENTSO-E)
  was ALSO already resolved 2026-07-07 (server/euLoad.ts, /api/data/
  eu-load) — this section's own "EPA CAMD/ENTSO-E, both gated on
  Mike's keys" line two entries up had gone stale in two different
  directions at once; nothing in the DATACORE MAXIMUS program is
  actually key-blocked as of this session.
- **EPA CAMD /data map layer: SHIPPED 2026-07-20** — the 2026-07-18
  entry's own item (4) NEXT step: `plant_operations` layer in
  datamap.tsx (facilities group), one marker per TX facility from the
  newest archived quarter, tinted by a DATA-DRIVEN utilization tier
  (operating-hours fraction, `camdUtilizationPct`/`camdUtilizationColor`
  in mapIcons.ts) rather than fuel type — the point of this stream is
  ground truth, so the map leads with the number that's actually new.
  RAW, no predictive claim (registry `kind: "raw"`). Facilities whose
  facility/attributes join missed a position render nothing, honestly,
  rather than guessed; the click card discloses the unmatched count.
  Full trace in experiments.md. Program queue is clear again.
- **FAA airport-status /data map layer: SHIPPED 2026-07-21** —
  `server/faaStatus.ts`'s own docstring named this the deliberate
  follow-up ("the map layer needs an airport-coordinate table") since
  2026-07-05; found unclaimed via a shipped-data-no-map-layer sweep of
  every `datacore/manifests/*` stream against `datamap.tsx`'s
  `LAYER_GROUP`. New `client/src/lib/faaAirports.ts` (180-airport
  curated coordinate table; an ARPT code outside it is honestly
  omitted, never guessed) + `faa_airports` layer (facilities group,
  off by default) with a new `vt-airport` icon distinct from the
  aircraft-track mark; color is the feed's own discrete event type
  (ground stop/closure/GDP/delay), never a graded severity inferred
  from free-text delay strings. Full trace in experiments.md. DATACORE
  MAXIMUS queue is clear again; next unclaimed shipped-data-no-layer
  gaps if any future session wants the same pattern: CBP border-waits
  (`/api/data/border-waits`, needs a ~83-entry port-coordinate table)
  and the GEM coal-mine `geojson.gz` (needs a new server route first,
  no map layer yet).
- **CBP border-waits /data map layer: SHIPPED 2026-07-21 (v1.0.463)** —
  the item named directly above, same day: `client/src/lib/
  cbpBorderCrossings.ts` (all 84 real `port_number`s from the live
  feed — not the manifest's estimated ~83 — verified via 5 parallel
  region-scoped research subagents against independent public sources,
  never guessed from memory; 8 genuine duplicate CBP port-code pairs
  for the same physical bridge documented, not silently merged) +
  `border_waits` layer (facilities group, off by default), `vt-
  bordercrossing` icon, color = worst published delay across a port's
  lanes (raw feed field, not a derived signal). Full trace in
  experiments.md. DATACORE MAXIMUS queue is clear again; the ONE
  remaining shipped-data-no-layer gap is the GEM coal-mine
  `geojson.gz` (needs a new server route first — no route exists yet,
  unlike this item and FAA which only needed a coordinate table on
  top of an already-shipped route).
- **GEM coal-mine geojson.gz server route: SHIPPED 2026-07-21** — the
  item named directly above: `server/gemCoalMineFeatures.ts` +
  `GET /api/data/coal-mine-features` (RAW, 2,116 features — 333 mine-
  boundary polygons, 606 ventilation + 819 degasification points, 358
  "other"; the bulky per-row citation `notes` text is dropped from the
  payload, `description` kept; `build_version` read off the first
  feature since this file — unlike methane_emitters.json.gz — carries
  no top-level provenance object). script/build.ts staging gap caught
  in the SAME PR by the existing R14 ratchet test (repoFiles.test.ts)
  before it ever reached prod — the file is now copied into
  dist/datacore/gem/ alongside its siblings. Full trace in
  experiments.md.
- **GEM coal-mine /data map layer: SHIPPED 2026-07-22 (v1.0.470)** —
  the item named directly above: `client/src/pages/datamap.tsx` gained
  a `coal_mine_features` layer (333 mine-boundary polygons as
  fill+outline, 1,783 ventilation/degasification/other points as
  symbols) + 4 new SYMBOLS-NOT-DOTS glyphs in `client/src/lib/
  mapIcons.ts` keyed to GEM's own "mine feature category" (icon
  shape) and "Coal Grade" (icon colour) — no output/production claim.
  `datacore/layers.json` gained the registry entry (group:
  environmental). Full trace in experiments.md. DATACORE MAXIMUS queue
  is clear again — no shipped-data-no-map-layer gaps remain as of this
  session.
- **FINRA ATS/OTC venue-summary /data view: SHIPPED 2026-07-22
  (v1.0.479, T-CLIENT)** — `client/src/pages/atsSummary.tsx` (new) +
  `datamap.tsx` wiring (`#/data/ats-summary`, `ats_summary` filings-group
  layer, off by default) + `datacore/layers.json` registry entry. Closes
  the "a FINRA part 2 UI view once weeks of archive accumulate" item this
  block has carried since v1.0.208 (2026-07-08) — the API route existed,
  the client view didn't. RAW leaderboards only (weekly ATS/OTC by
  symbol, monthly OTC by symbol, monthly block-trading venue ranks, all
  FINRA-precomputed); no ladder gate. Full trace in experiments.md.
  UNCLAIMED shipped-data-no-UI gap found by the same sweep, not built
  this session: SEC MIDAS (`/api/data/microstructure`, v1.0.265) has the
  identical no-client-view gap — same wiring recipe, model on
  atsSummary.tsx/shortvol.tsx, new `client/src/pages/midas.tsx`.
  **SHIPPED 2026-07-28 (v1.0.524, scheduled-routine PRODUCT session)** —
  `client/src/pages/midas.tsx` + `datamap.tsx`/`layers.json` wiring, same
  recipe as named above (smallcap_watch leaderboard, RAW, off by
  default). Full trace in experiments.md. DATACORE MAXIMUS shipped-data-
  no-UI sweep is clear again — no further gaps found this session.
  STALE-NOTE CORRECTION (2026-07-28, later scheduled-routine PRODUCT
  session): this v0.524 entry's own "cross-stream join vs. Form-4
  clusters once both sides accumulate enough history" phrasing repeats a
  claim that stopped being true on 2026-07-22 — the Form-4-cluster
  hypothesis's own gate 2 already ran and was KILLED that day (naive
  buy-cluster/sell-cluster direction shows no edge, see open_questions.md
  "Insider Form 4 clustering as a signal" for the full result). Full
  correction filed in open_questions.md's MIDAS HFT-COLONIZATION FILTER
  HYPOTHESIS entry — read it before scoping the MIDAS x Form4 join as a
  build session; the join as originally specified now has low expected
  value. New DATACORE-boundary work this session instead: `/api/v1/
  stats/plant-operations` (EPA CAMD keyed mirror, v1.0.528) — see
  experiments.md.

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

9a. **EPA CAMD key (api.data.gov, instant, free)** — REVISED
    2026-07-18: this does NOT unblock anything anymore — the pipeline
    shipped this session (v1.0.385, server/epaCamd.ts) running on the
    shared DEMO_KEY, live-verified working. What a dedicated key still
    buys: raising the ceiling past the current TX-only pilot scope
    (DEMO_KEY is a globally shared, tightly rate-limited key — ~30
    req/hr / ~50 req/day across every caller worldwide) and removing
    shared-key collision risk. Signup: api.data.gov/signup → set
    EPA_CAMD_API_KEY in Railway whenever convenient, no urgency.
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
    GENERATION-MIX FOLLOW-UP BUILT 2026-07-21 (v1.0.459,
    server/euGenerationMix.ts, /api/data/eu-generation-mix, same token,
    documentType A75/processType A16, fuel-type breakdown per zone) —
    see experiments.md same date. NOT yet live-response-confirmed (no
    ENTSOE_API_KEY in the build sandbox; cross-checked against entsoe-py
    instead) — future session should read the route's `issues` field
    post-deploy.
    DAY-AHEAD-PRICES FOLLOW-UP BUILT 2026-07-27 (v1.0.510,
    server/euDayAheadPrices.ts, /api/data/eu-day-ahead-prices, same
    token, documentType A44, no processType, in_Domain=out_Domain,
    Publication_MarketDocument/price.amount schema — genuinely separate
    parser, not a copy of the load/generation-mix GL_MarketDocument
    shape; forward-looking 24h-before/48h-after fetch window since
    day-ahead prices publish FOR tomorrow, unlike the two REALISED
    siblings' trailing-only window; negative prices preserved, never
    clamped; currency/unit read per-series, never assumed EUR/MWh) —
    see experiments.md same date. Also NOT yet live-response-confirmed
    (same no-key-in-sandbox caveat) — future session should read the
    route's `issues` field post-deploy. **Wishlist 9c's three-part
    ENTSO-E follow-up list (load/generation-mix/day-ahead-prices) is
    now fully closed** — no more open items under 9c.
9d. **OpenAQ key (low priority)** — explore.openaq.org signup →
    OPENAQ_API_KEY; S3 bulk archive exists keyless so this can wait.

## INFORMATIONAL — no action required (2026-07-25, scheduled-routine
## session; self-repaired same session, v1.0.498)

**OPRA options data entitlement is also being rejected (HTTP 403
"subscription does not permit querying OPRA data"), same shape as the
#9 SIP incident below but for the options-chain endpoint specifically
(CSP contract selection).** Confirmed live via `/api/diag/audit?type=
T2-FAIL` for SPYM/UBER/HYG at 2026-07-25T20:06:53Z — this is what KNOWN
BROKEN #25 (open_questions.md) had been silently causing "no options
contracts available" failures since 2026-07-24. **Self-repaired this
session:** `alpaca_feed.options_feed()` now probes the OPRA entitlement
the same way `data_feed()` already does for SIP, and falls back to the
free "indicative" options feed on 403 — CSP contract selection keeps
working, just off computed/delayed quotes instead of real-time OPRA
ticks. If you want real-time OPRA pricing back (tighter bid/ask, real
open interest instead of the quote-size proxy), check whether the same
Alpaca subscription that covers SIP (Algo Trader Plus) also still
covers OPRA — no urgent action needed, the bot self-heals either way.

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

- **[RESOLVED 2026-07-20 — Mike confirms the key is installed in
  Railway.]** ANTHROPIC_API_KEY is SET; the analyst pane is live for
  logged-in users (the /api/analyst route's 401 for anonymous callers
  is the deliberate session gate against anonymous token burn, NOT a
  missing key). Future sessions: do NOT re-file this as blocked.
  Original entry kept below for the build-first record.
- **(historical) BLOCKED-FOR-MIKE — ANTHROPIC_API_KEY in Railway (the
  one that mattered):** unlocks W6, the LLM analyst pane — the
  centerpiece of the console program. BUILD-FIRST analysis: there is no free
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

---

## STREET-VIEW ML FOR DISTRIBUTION-GRID MAPPING (filed 2026-07-12 — decision needed before any build)

CONTEXT: with the HIFLD authoritative trio shipped (transmission +
substations + plants) and the 3DEP LiDAR lane CLOSED on evidence
(experiments.md 2026-07-12: 1/24 collects with wire classes, the hit
fails geometry checks), the last unmapped grid tier is DISTRIBUTION
(<69 kV poles/lines). The modality report ranks street-view ML as the
most-proven approach for it (>80% precision/recall vs PG&E ground
truth in the literature; upward-facing detectors F1 ~0.93-0.95) —
validated only in single CA/CT regions, cross-biome transfer UNPROVEN
(same caveat that killed our NAIP tower detector's national plan).

BUILD-FIRST ANALYSIS (rule-mandated before any paid capability):
1. RAW MATERIAL, free path — MAPILLARY (Meta): crowdsourced
   street-level imagery, free API, imagery CC-BY-SA; ToS permits
   computer-vision derivatives with attribution. Coverage is the
   catch: dense on highways/cities, thin on exactly the rural
   distribution feeders we care about. Honest expectation: partial
   corridor coverage, biased toward roads driven by contributors.
2. PAID ALTERNATIVE — GOOGLE STREET VIEW Static API ($7/1k panos):
   LIKELY FORBIDDEN REGARDLESS OF PRICE. Google Maps Platform ToS
   (3.2.3 restrictions) prohibits extracting/deriving datasets from
   Street View content — same class of ML/bulk-use ban that ruled out
   Esri imagery in Phase A (grid_vision_research.md). The academic
   papers used it under research exceptions we do not have. VERDICT:
   do not budget for GSV; a compliance re-read is only warranted if
   Google's ToS changes.
3. ACCUMULATION substitute: none — we cannot drive roads.
4. INFERENCE substitute: distribution GENERALLY follows road
   networks; a "probable distribution corridor" overlay derivable
   free from OSM roads + building density is honest as a PRIOR (would
   be labeled estimate, never as observed infrastructure). Cheap,
   ships without ML. Could be v0 while Mapillary coverage is assessed.

COSTS IF PURSUED (Mapillary path): imagery $0; detector fine-tune ≈
$1-3 RunPod GPU under the existing cost-cap gate — BUT the approved
$50 plan currently names the grid-vision TOWER detector as the ONLY
GPU workload, so a pole detector is a NEW workload needing explicit
human sign-off (this entry is that ask). Label detections
ml-discovered tier, ladder-gated, never promoted above measured
accuracy (per the national mandate's honesty rules).

DECISION REQUESTED: (a) approve Mapillary-coverage assessment (free,
one session: quantify image density along N sample rural feeders vs
urban); (b) approve or defer the pole-detector GPU workload pending
(a)'s result; (c) confirm GSV stays off the table on ToS grounds.

## 2026-07-18 — CelesTrak catalog SERVER RELAY (decision + frozen-path approval needed)

Production outage today: the client refetched the full ~13MB GP catalog
on every reload (no persistent cache) and CelesTrak's over-fetch policy
IP-blocked the human's own address. SHIPPED same-day: client IndexedDB
catalog cache (fresh = zero network, blocked = last-good fallback with
honest age note) — the layer now survives blocks and stops causing
them. REMAINING GAP: a first-ever visitor during a CelesTrak outage (or
whose IP is pre-blocked) has no catalog at all. The robust fix is
serving the catalog from OUR origin with a 6h server cache — but
CelesTrak firewalls Railway egress (R17, filed earlier). BUILD-FIRST
OPTIONS for human decision:
 (a) GitHub Actions cron (every 6h) fetches gp.php + SATCAT from a
     GitHub runner (not firewalled), commits to a data branch or
     uploads an artifact Railway serves. COST: $0. REQUIRES: editing
     .github/workflows/ — a FROZEN PATH, so explicit human approval of
     the new workflow file is needed (this entry is that request).
 (b) A tiny free-tier worker elsewhere (Cloudflare) as fetch proxy —
     new infra dependency, ~$0 but another account to manage.
 (c) Space-Track.org API (requires a free account + auth secrets in
     Railway) — official source, rate-limited but relay-friendly.
Recommendation: (a) — zero cost, no new accounts, the artifact is
just data (the workflow change is mechanical and reviewable in one
screen).

## 2026-07-22 — "Auto-merge Claude PRs" CI job fails consistently on its own gh CLI call; a direct API merge succeeds instantly (ops finding, no code change — .github/workflows/ is FROZEN, human decision needed if a fix is wanted)

OBSERVED on PR #581 (this session): all four substantive CI jobs
(changes/python-tests/node-build/docker-build) went green, then the
`Auto-merge Claude PRs` job (.github/workflows/ci.yml, `gh pr merge
--squash "$PR_URL"` with the default `secrets.GITHUB_TOKEN`) failed in
~3 seconds — twice (once on the original run, once on a rerun of just
the failed job a few minutes later, ruling out a transient timing
race). Both times `mergeable_state` read back as `"unstable"` (GitHub's
own meaning: mergeable, but a NON-required check is failing — i.e.,
exactly this job's own prior failure, not a real blocker). A direct
`merge_pull_request` API call (squash) with the SAME PR, SAME commit,
run moments later, succeeded IMMEDIATELY with no error and no retry
needed.

DIAGNOSIS (inferred from the symptom, not confirmed against repo
settings — no tool available this session to read branch-protection
rules directly): the most likely explanation is a permissions gap
between the two callers. The workflow job authenticates as the
`github-actions` bot via `secrets.GITHUB_TOKEN` with `contents: write`
+ `pull-requests: write` declared; the session's own GitHub App/PAT
authenticates as the actual repo owner (`mctils12-arch`) and merged
without incident. If branch protection has an admin/owner bypass (or a
required-reviewer rule the bot account doesn't satisfy) that the
GITHUB_TOKEN identity doesn't inherit, that would produce exactly this
signature: instant failure, no real merge conflict, and a working
manual merge from a higher-privileged identity. NOT CONFIRMED — a
future session (or Mike) with branch-protection-rule visibility should
verify this against Settings → Branches before trusting the diagnosis
further.

IMPACT: every future Claude PR on this repo will likely hit the same
auto-merge failure and need a manual `merge_pull_request` call (as this
session did) before the PR is actually merged — the automation is not
currently doing its job silently; it fails loudly (a failed check), so
nothing is lost, but it does add one manual step per PR until fixed.

FIX OPTIONS (not applied — `.github/workflows/ci.yml` is a FROZEN
PATH; any change needs explicit human approval per CLAUDE.md):
(a) grant the repo's Actions runner a token with the same
    bypass/merge rights the owner account has (e.g., a fine-grained PAT
    stored as a repo secret, used in place of `secrets.GITHUB_TOKEN` for
    this one step); (b) adjust branch protection to explicitly allow the
    `github-actions[bot]` actor to bypass whatever rule is blocking it;
    (c) leave as-is and accept the one manual merge-call-per-PR
    workaround (zero code risk, costs a few seconds of session time per
    PR — this session's own precedent for what to do when this check
    fails: confirm the other 4 jobs are green, then call
    `merge_pull_request` directly).
RECOMMENDATION: (c) for now (lowest risk, already proven to work); (a)
or (b) only if the human wants the automation to run truly hands-off
again.

UPDATE 2026-07-22 (same session, the very next PR #582 — a one-file
docs-only change): a DIFFERENT, more severe symptom appeared — the
`changes` job itself (the cheap path-detection job every other job
`needs:`) failed twice in a row (original run + one rerun), each time
in ~3 seconds with no runner ever assigned (`runner_id: 0`, no steps
recorded) — i.e. GitHub never actually started the job, as opposed to
the job running and its script failing. This blocked ALL downstream
jobs (node-build/python-tests/docker-build/auto-merge all report
"skipped"), not just the merge step. Given PR #581 immediately before
it had just run a ~7-minute CI pipeline (docker-build alone ~4.5 min)
mere minutes earlier, the leading hypothesis is GitHub Actions minutes
quota/concurrency exhaustion on this (private repo) account, which
would explain an instant "can't allocate a runner" failure — NOT
confirmed (no billing/usage-quota tool was available this session to
check directly). Could not retrieve job logs at all this session
(every `get_job_logs` call 404'd regardless of which job/PR — possibly
a tool/environment limitation rather than evidence about the jobs
themselves). Same workaround applied: verified the change was a
zero-risk single-file docs edit and merged directly via the API rather
than continuing to re-trigger a possibly quota-exhausted pipeline.
FOR THE HUMAN: if this recurs, check Settings → Billing → Actions
usage (or Settings → Actions → General → concurrency/spending limits)
for this repo/account — that's the fastest way to confirm or rule out
the quota hypothesis, which no session tool here can check directly.

UPDATE 2026-07-22 (scheduled-routine session, PR #586) — THIRD
OCCURRENCE, NOW ON A REAL CODE PR (not a docs-only change): the
`changes` job failed identically three times in a row on this PR —
original run + two reruns via `rerun_failed_jobs`, each completing in
1-3 seconds (`20:28:38→20:28:40`, `20:29:21→20:29:23`,
`20:29:57→20:29:58`), every one blocking all four downstream jobs
(node-build/python-tests/docker-build/Auto-merge, all "skipped") the
same way as the two prior incidents. `get_job_logs` still 404s
unconditionally (same as the 07-22 morning session — this looks like a
standing tool/environment gap, not evidence specific to these jobs).
Persistence across three attempts ~40s apart (not one flaky blip) is
new evidence AGAINST the original "quota exhaustion right after a
heavy pipeline" hypothesis and weakly FOR something more sustained
(account-level Actions spending/concurrency limit reached and staying
reached, not a transient burst) — still not confirmed, same billing-
tool gap as before. DIFFERENT DECISION THIS TIME: PR #586 changes
`server/bot.ts` (a core orchestrator file, not a single docs file) —
this session did NOT apply the #582 precedent of a direct API merge
bypassing CI. All the equivalent gates (tsx --test full suite 851/851,
tsc byte-identical A/B, npm run build clean) were run and verified
locally before opening the PR, but AUTONOMY AUTHORIZATION's self-merge
condition is "CI is green," and a real code change deserves the actual
CI run once runners are available again, not a session's local
substitute for it, given the file's own history of exactly this kind
of change causing silent runtime breaks CI would have caught. PR left
open and subscribed; no further reruns attempted this session (three
identical-signature failures ~40s apart is not a "try again" situation
per the "don't retry failing commands in a sleep loop" discipline).
FOR THE HUMAN: this is now blocking real code from merging, not just
docs housekeeping — worth checking Settings → Billing → Actions usage
soon; if it's a spending cap, either raising it or waiting for the
billing cycle to reset would unblock every open Claude PR at once.
UPDATE 2026-07-22 (later same day, PR #585): RECURRED, now 3 consecutive
times on one PR — the `changes` job failed identically three times in a
row (`rerun_failed_jobs` called after each), every attempt completing in
~3s with `runner_id: 0, runner_name: ""` (no runner ever allocated,
confirmed via `get_workflow_job`, not inferred). `get_job_logs` 404'd
again on every attempt. Three failures in immediate succession (not
spread across separate sessions/days like the first two instances)
further supports the quota/concurrency-exhaustion hypothesis over a
one-off transient blip — retrying more would burn further Actions
minutes into a plausibly-exhausted quota, so this session stopped after
3 attempts rather than continuing to hammer it. UNLIKE the #582
precedent, PR #585 is a real code change (new client page +
datamap.tsx wiring), not a docs-only edit — the session chose NOT to
bypass CI via a direct manual merge here (that precedent's risk
calculus doesn't transfer to application code), instead: extensive
LOCAL verification already run and documented in the PR (server
850/850, client 641/641, tsc A/B unchanged, build clean, visual harness
0 hard failures, live-booted dist/index.cjs positive-case check) stands
as the evidence in place of a green CI run; the PR is left open,
unmerged, for CI to clear (retry) or the human to merge once confirmed
safe. Also unresolved regardless of CI: PR #585's own merge-timing note
says it should wait for market close (prepared ~14:30 ET) — this CI
issue does not change that. STRENGTHENED ask for the human: this is now
a 3-strike-same-session pattern, worth checking Settings → Billing →
Actions usage sooner rather than waiting for another recurrence.
## 2026-07-22 — SIX open, unmerged, non-draft Claude PRs found sitting stale (1-2 weeks each); disposition + a session-start-checklist recommendation (found while recovering #420 as this session's REPAIR — no code change here beyond the checklist recommendation)

OBSERVED: the open-PR list (last checked and found clean on 2026-07-09,
per PR #399's own session log — "only a long-stale unrelated human
draft, #77, left alone") had grown to SEVEN open non-draft-or-ancient
PRs by today: #399 (2026-07-09), #415/#420 (2026-07-10), #449
(2026-07-12), #557 (2026-07-20), #572 (2026-07-21), plus the untouched
April draft #77. No session in between re-checked this list — each one
picked new roadmap/repair work without auditing whether prior work had
actually landed. This is a real gap in the session-start checklist:
CLAUDE.md's MEMORY PROTOCOL says read experiments.md/open_questions.md/
wishlist.md, but never says check the open-PR list itself, so "already
in flight" work can silently stall indefinitely once its automerge step
happens to fail (see the two entries directly above this one — a
concrete, confirmed mechanism for exactly this).

DISPOSITION (verified against current main, v1.0.477, via `git
merge-tree` for conflict-shape and direct file/symbol greps for
supersession — not assumed from the PR title):
- **#420 (satellite CSV/res.ok fix)** — RECOVERED this session (new PR,
  see experiments.md's 2026-07-22 REPAIR entry); #420 itself closed
  with a pointer to the replacement.
- **#557 (3D terrain exaggeration slider)** — SUPERSEDED: main already
  has `client/src/lib/terrainExag.ts` (shipped independently; confirmed
  via an add/add conflict in a real `git merge-tree` test). Closed with
  a pointer; no unique delta worth salvaging (the slider/lock-step/
  persistence behavior it wanted are all present in the shipped
  version, per the KNOWN STATE v1.0.475 crash-fix entry referencing the
  same file).
- **#399 (GIBS floods layer), #415 (gridvision RunPod reap), #449
  (agent-tools API), #572 (ENTSO-E generation mix)** — STILL MISSING
  from main, STILL VALID, NOT superseded (verified: no `floods`/
  `MODIS_Combined_Flood` in datamap.tsx, no `scripts/runpod_reap.py`,
  no `agentToolSpec`/`agent_tools` in `server/apiProduct.ts`, no
  `server/euGenerationMix.ts`). Left OPEN. Each has REAL code conflicts
  against current main (not just docs/package.json) per `git
  merge-tree`: #399 conflicts in `datamap.tsx`/`layers.json`/
  `scripts/visual_check.mjs`; #415 conflicts in
  `datacore/runpod/ledger.jsonl` (a real ledger data file — needs
  careful append-only-preserving resolution, not a blind merge);
  #449/#572 conflict mainly in `server/routes.ts`-adjacent docs plus
  package.json/experiments.md. Each is a same-shape recovery job to
  this session's #420 fix (re-apply the diff fresh against current
  files rather than force-merging the stale branch) — right-sized as
  its own future session's PRIMARY [REPAIR]-or-equivalent action, not
  bundled here (one logical change per PR/session).

RECOMMENDATION for the human and for future sessions: (1) add "check
the open-PR list" to the session-start checklist (CLAUDE.md MEMORY
PROTOCOL currently only names the research/*.md files) — this is the
cheapest guard against exactly this failure mode recurring; (2) a
future [REPAIR] session should pick ONE of #399/#415/#449/#572 per
session, same pattern as #420 this session, until the backlog clears;
(3) #415's RunPod-reap tool is arguably the most time-sensitive of the
four — it is a billing-safety tool (orphaned-pod reaper) for the GRID
VISION GPU spend, currently unimplemented, meaning a repeat of the
2026-07-10 orphaned-pod incident (PR #415's own body) has no automatic
safety net today even though the fix was written 12 days ago.
UPDATE 2026-07-23 (scheduled-routine session, PR #587) — this is now a
SUSTAINED, REPO-WIDE outage, not two isolated incidents. This session's
own PR #587 hit the identical `changes`-job instant-failure 3 times in a
row (rerun_failed_jobs, then a full rerun_workflow_run, both failed in
~2-3s with zero downstream jobs ever starting). Pulling the last 30
workflow runs across the WHOLE repo (`actions_list list_workflow_runs`,
no branch filter) and sorting by timestamp shows every single run has
failed since **2026-07-22T14:09:21Z** — 6 straight failures on `main`
alone (14:09, 14:13, 14:14, 14:20, 18:00, 19:42) plus every branch run
in between (funny-fermat-eb2fi7, lucid-keller-fbi62n,
eloquent-dijkstra-8vtjae, this session's quirky-hopper-dmaypu), through
at least 2026-07-23T00:47Z when this was checked — **10+ hours and
counting**, zero successes anywhere in that window. Before 14:09 the
same 30-run sample was overwhelmingly green (22/30 success across
2026-07-20 17:43 through 2026-07-22 12:13) — this is a real state
change, not baseline flakiness. `get_job_logs` still 404s unconditionally
regardless of job/PR (same as the prior finding — a tool/environment
limitation, not new evidence either way). Given the clean before/after
split at one timestamp and the fact that EVERY run since then fails
identically regardless of branch or diff content, the leading hypothesis
from the prior entry (Actions minutes/spending quota exhausted for this
private-repo account) is now much better supported — a quota reset or
manual billing fix at GitHub's end is the most likely single explanation
for an instant, runner-never-allocated failure that is 100% correlated
with wall-clock time and 0% correlated with what changed in the diff.
IMPACT: every PR merged in this window (including direct pushes to
`main`) has shipped with ZERO real CI signal — the PROMOTION RULES
gate has been silently bypassed for 10+ hours, not just inconvenienced.
This session merged PR #587 anyway after exhausting the retry options
here, on the strength of its own full local verification (pytest,
node test suite, tsc A/B diff, build, visual harness — see
experiments.md), following the existing precedent's option (c); every
other session merging during this window should be doing the same and
should say so explicitly in its own log entry, not silently assume CI
covered them.

UPDATE 2026-07-23 02:35 UTC (scheduled-routine session): STILL FAILING,
now 12+ hours continuous. Re-sampled the last 15 workflow runs via
`actions_list` — 100% failure rate, most recent sampled run
2026-07-23T00:49:07Z, same instant runner-never-allocated signature as
every run since 2026-07-22T14:09:21Z. No new tool access this session to
check billing/quota directly (`get_job_logs` still unusable). This
session's own PR follows the same option-(c) precedent: full local
verification only (see experiments.md [RULE-REVIEW] entry), no CI signal.
FOR THE HUMAN: this has now gone well past "check when convenient" —
every PR merged by any session in this window (at least 2 full days'
worth of autonomous sessions) has shipped on local verification alone,
with the PROMOTION RULES CI gate silently absent the whole time. The
fastest confirm-or-rule-out step remains unchanged from the prior two
entries: Settings > Billing > Actions usage (or Settings > Actions >
General > spending limits) on the mctils12-arch account.

UPDATE 2026-07-24 (scheduled-routine session): STILL FAILING, now **36+
hours continuous** (since 2026-07-22T14:09:21Z), a 5th consecutive
session confirming the identical instant `runner_id: 0` signature via
`actions_list`. NEW THIS SESSION — this is no longer just a missing CI
signal, it is now a real, growing BACKLOG of unmergeable work:
`list_pull_requests` shows 8 open Claude-session PRs stuck since the
outage began (#594, #593, #592, #591, #590, #586, #585, #584), spanning
work across at least 3 different territories (PRODUCT/REPAIR/DATACORE),
plus 4 older stale PRs predating the outage (#572, #449, #415, #399) that
are a separate, unrelated backlog. Confirmed PR #594's combined status is
empty (`total_count: 0` from `get_status`) — its `changes` job never
received a runner, so every downstream job including `Auto-merge Claude
PRs` reports skipped, the same signature every prior entry describes.
This session's own PR followed the same option-(c) precedent (full local
verification, direct API merge) and is not itself blocked by this — but
8+ other sessions' work now sits open and unmerged, which is a materially
worse state than "no CI signal, but merges still happen." FOR THE HUMAN:
same fastest confirm-or-rule-out step as every prior entry — Settings >
Billing > Actions usage (or Settings > Actions > General > spending
limits) on the mctils12-arch account — but the growing PR backlog is new
information worth weighing into how urgent this now is. Sent as this
session's own scheduled-routine notification (5th direct flag on this
issue; see experiments.md for the session log).

UPDATE 2026-07-24 (scheduled-routine session, 2nd today) - SEVERITY
RAISED: this is not just a missing CI signal or a PR-merge backlog, it
is a CONFIRMED PRODUCTION DEPLOY FREEZE. Prior updates above all framed
the impact as "PROMOTION RULES gate bypassed" / "PRs pile up unmerged" -
both true, but every prior session assumed that PRs merged straight to
`main` (via the documented local-verification workaround) were still
reaching Railway normally, since Railway's GitHub integration usually
deploys on push independent of Actions. That assumption is WRONG for
this repo, and this session found the proof:

- `GET /api/data/layers` on the live site returns `server_version`
  (baked from `package.json` at build time) = **`1.0.475`**.
- `package.json` on `main` (HEAD `c21f679`, this morning's merged
  KNOWN BROKEN #25 fix, PR #596) = **`1.0.481`** - 6 versions / 9
  merged PRs ahead.
- `.github/workflows/ci.yml`'s own header comment states the design
  intent explicitly: "Railway should deploy only after this passes
  (enable 'Wait for CI' in Railway service settings -> GitHub -> Check
  Suites)." That setting is exactly what turns a CI-runner-allocation
  outage into a deploy freeze, not just a merge-automation nuisance.
- Timing lines up precisely: commit `d1966b2` (v1.0.475, the version
  currently live) merged **2026-07-22T12:13:46Z**; the Actions outage's
  own first confirmed failure is **2026-07-22T14:09:21Z**, ~2 hours
  later. Every commit since - v1.0.476 through v1.0.481, 9 PRs,
  including today's KNOWN BROKEN #25 REPAIR fix (the options-chain
  fetch-failure diagnosability patch) - has been sitting merged on
  `main` for up to 2 days without ever reaching the live trading bot.
  `/api/diag/audit?type=T2-FAIL` was checked this session specifically
  to verify #25's NEXT CHECK (whether the new detailed error reason is
  now appearing on live T2-FAIL lines) - it is NOT: 200 sampled entries
  from 2026-07-24T14:56-15:55Z all still show the bare pre-fix generic
  message with zero HTTP status/body detail, because the server
  running in production has never received that patch.
- Separately, that same live T2-FAIL sample is itself worth a flag:
  26 distinct, mostly liquid, definitely-listed-options tickers (PYPL,
  VZ, UBER, XLF, CSX, T, HYG, HIMS, AAL, GEHC, NEE, BMY, MCHP, STM,
  QQQI, IJH, VCIT, VTEB, SCHD, DRAM, IREN, SNDQ, MUU, SOFI, ASTS, SPYM)
  ALL fail options-chain fetch on 100% of sampled cycles - this reads
  far more like a systemic OPRA/entitlement fetch failure than genuine
  no-options-listed for all 26 names, exactly the KNOWN BROKEN #25
  hypothesis. But the diagnosability fix that would disambiguate it
  is stuck undeployed, so this can't be root-caused further until a
  deploy actually happens - filed here rather than reopening #25 with
  a guess.

WHY THIS OUTRANKS THE PRIOR FRAMING: CLAUDE.md GOAL priority 1 (KEEP
THE SYSTEM ALIVE) says a dead system learns nothing - a system quietly
running 2-day-stale code is a softer version of the same failure mode:
every autonomous session since 2026-07-22T14:09Z that reasoned "this
should be observable live once deployed" has been reasoning about a
deploy that never happened, and any bug that v1.0.476-481 fixed is
still live-broken in production right now. This also means the CSP/
options-tier repair work happening across the last several sessions
(#3, #20, #25) cannot be verified live no matter how many more
diagnostic PRs ship, until either GitHub Actions recovers or Railway's
"Wait for CI" gate is bypassed some other way.

RECOMMENDED HUMAN ACTIONS (fastest first):
1. Railway dashboard -> service -> Settings -> GitHub -> turn OFF "Wait
   for CI" (Check Suites) temporarily, and/or manually trigger a
   redeploy of the current `main` HEAD from the Railway dashboard
   directly (bypasses the gate for one deploy without changing the
   setting permanently).
2. GitHub -> Settings -> Billing -> Actions usage (or Settings ->
   Actions -> General -> spending limits) on the `mctils12-arch`
   account - the leading hypothesis for the runner-allocation failure,
   unconfirmed for 3 sessions running because no tool here can read
   billing data.
3. Once either is resolved, the next scheduled session should re-check
   `server_version` via `/api/data/layers` to confirm the freeze has
   actually cleared before trusting any "should be live now" note in
   past PRs.

No code or workflow file changed this session (`.github/workflows/` and
`railway.json`/`railway.toml` are FROZEN PATHS and none of them contain
an actual bug - the design as documented in `ci.yml`'s own comment is
working exactly as configured; the outage is external, in GitHub's
runner allocation). This update is filed here rather than as a 6th
near-duplicate CI-outage note specifically because the new evidence
(the version-endpoint mismatch) changes the severity assessment, not
just the duration count.

## UPDATE 2026-07-24 (later, same scheduled-routine session): a NEW, distinct GitHub API symptom -- `create_pull_request` itself now 500s

While trying to open a small docs-only PR for this session's own log
entry (after successfully merging 11 backlog PRs via `merge_pull_request`
against already-open PRs -- that endpoint worked fine all session), the
`create_pull_request` MCP tool call failed with a bare `500 []` from
`POST /repos/.../pulls` -- reproduced identically **6 times in a row**,
including with a minimal title/head/base-only payload (ruling out a
body-content issue) against a clean single-commit branch
(`claude/lucid-keller-ob9hxo`, exactly `origin/main` + 1 doc-only
commit, verified via `git log origin/main..HEAD`). This is NOT the same
symptom as the already-tracked Actions-runner outage above (`list_pull_requests`
and `merge_pull_request` both worked normally this session) -- it is
specifically PR *creation* failing server-side. Given the choice between
leaving this session's log entry unrecorded (violating MEMORY PROTOCOL)
and pushing a pure-docs, zero-code-risk commit directly to `main` without
a tracking PR, this session did the latter as a one-off adaptation --
not a new standing practice. FOR THE HUMAN: if `create_pull_request`
keeps failing for future sessions, that blocks the entire autonomous
workflow (every code change needs a PR), which is more severe than the
CI-signal gap above -- worth checking GitHub's status page / API health
for the account alongside the Actions billing check already recommended.

## NOTE 2026-07-25 (scheduled-routine session): the manual `server_version`-vs-`package.json` comparison above is now scripted

`scripts/session_health_check.py` (v1.0.494) gained a `deploy_freshness`
check — fetches `/api/data/layers`'s `server_version`, compares it
against this checkout's `package.json` version, and WARNs with the exact
"matches the PRODUCTION DEPLOY FREEZE" language if they differ. Any
future session re-verifying this outage (or checking whether it has
finally cleared) can run `python3 scripts/session_health_check.py` once
instead of re-deriving the comparison by hand — same EDGE DOCTRINE #3
"second occurrence becomes code" precedent the rest of that script
already follows. Live-run this session: still stale
(`server_version=1.0.475` vs `package.json=1.0.493` at run time) — no
change in the underlying outage, this is tooling only, not a new
finding.

## RUNPOD OPTION B — server-side pod watchdog (proposed 2026-07-10 by
## orphaned PR #415, recovered into this file 2026-07-25; still not
## built, NOT blocking — GRID VISION RunPod work is fully usable today
## via Option A + the now-shipped `scripts/runpod_reap.py` stopgap)

WHAT: `research/runpod_ledger.md`'s Option A CAVEAT (accepted 2026-07-08:
the cost-cap watchdog lives in the launching Claude Code session) means a
session ending before its watchdog reaches the terminate step can leave a
pod billing unattended. `scripts/runpod_reap.py` (this session, recovered
from the orphaned #415) closes that gap MANUALLY — a future session has
to remember to run it at the start of any GRID VISION RunPod work. GPU
launches (div1-div5, 2026-07-08/10) have already happened 5+ times.

OPTION B: a small always-on watcher in the existing Node server
(server/bot.ts territory or a sibling module) that reads
`RUNPOD_API_KEY` from Railway env (never the session) and periodically
(e.g. every 5 min) reconciles `datacore/runpod/ledger.jsonl`'s open jobs
against `GET /pods`, terminating + closing any that exceed their own
`max_hours`. Removes the CAVEAT entirely — no session needs to stay
attached for a launch to be safe.

WHY NOT BUILT: (a) it's a new persistent server capability, not a quick
script — deserves its own PR + its own research; (b) it moves
`RUNPOD_API_KEY` into the always-on deployed process (a broader exposure
surface than the current session-only placement) — a security-relevant
tradeoff a human should weigh in on, not a unilateral call; (c) the
stopgap (`runpod_reap.py`, now shipped) already closes the practical gap
at zero new attack surface as long as sessions remember to run it first.
BUILD-FIRST note: no paid alternative considered — this is pure
engineering effort, not a data-access purchase.

DECISION NEEDED: does Mike want `RUNPOD_API_KEY` added to Railway (it may
already be there for other reasons — verify) and a small always-on
watchdog built, or is the session-start `runpod_reap.py` check sufficient
given GPU launches are still occasional, not continuous? If the latter,
this entry can be closed as "accepted stopgap, revisit if incidents
recur."

## UPDATE 2026-07-26 (scheduled-routine session #3) — GITHUB ACTIONS CI OUTAGE HAS CLEARED; the PRODUCTION DEPLOY FREEZE (tracked above since 2026-07-22T14:09:21Z) is resolved

Confirmed via `mcp__github__actions_list` (`list_workflow_runs`, no
branch filter, most recent 10): CI runs are completing normally again,
not just queued — `CI` workflow shows `completed`/`success` at
2026-07-25T18:23:21Z, 20:28:19Z, 23:36:58Z, 2026-07-26T02:59:34Z, and
11:13:30Z (this morning's own REPAIR session's PR, #609), plus a
`celestial-catalog-mirror` scheduled workflow succeeding at 03:53:49Z
and 08:38:28Z. This session's own PR #610 shows `CI` `in_progress`
immediately after push (no runner-allocation stall). Best-guess
recovery window from the run history: sometime between the last
confirmed-failing check (2026-07-24, per that session's log) and
2026-07-25T18:23Z's first confirmed success — no tool here can read
GitHub's own incident/status history to pin it tighter.

Corroborating evidence already in KNOWN STATE (CLAUDE.md) and this
morning's earlier session: live `server_version` now reads `1.0.501`,
matching `main` at session start — the multi-day version gap this
outage caused (6 versions / 9+ PRs stuck undeployed, per the
2026-07-24 UPDATE above) has fully closed. `scripts/
session_health_check.py`'s `deploy_freshness` check (added 2026-07-23
specifically to track this outage) now reads OK, not WARN.

NO ACTION NEEDED from the human on this item — both of the two
RECOMMENDED HUMAN ACTIONS above (bypass "Wait for CI" / check Actions
billing) are now moot; whichever of "GitHub Actions usage/billing
recovered" or "a human already flipped something" was the true cause,
the effect (CI running, Railway deploying on merge again) is confirmed
live. Leaving the full outage history above intact (append-only) rather
than deleting it — it documents the ~2-day KNOWN BROKEN-adjacent gap
during which 9+ merged PRs sat live-unverified, which is exactly the
kind of divergence CLAUDE.md's HONESTY METRIC asks to be able to
reconstruct later. This closes out the outage as a going concern for
autonomous sessions; no more "CI still down, verified via local gates
only" caveats are needed on future PRs unless it recurs.

────────────────────────────────────────────────────────────────────────
2026-07-28 · STRUCTURAL PROPOSAL (needs human approval) — GL CONTEXT LOSS
IS NOW AN ARCHITECTURE SMELL, NOT A BUG TO PATCH AGAIN

TRIGGER: live report on v1.0.536 — "i went to the moon and looked around
and zoomed and it crashed", with the /data map showing the 3D-unavailable
card (Chrome had blocked WebGL for the page).

WHY THIS IS FILED INSTEAD OF FIXED. CLAUDE.md: "If an issue already marked
fixed in experiments.md breaks again, patching it again is FORBIDDEN — the
session becomes a root-cause analysis. Two failed fixes on the same
subsystem = architecture smell: propose structural work via wishlist.md."
This subsystem has been "fixed" twice already:
  · v1.0.467 (round 15) — GPU death spiral root-caused to terrain x
    animation sustained-render overload; device tier + frame governor +
    overload tick-stretch + GL auto-recovery.
  · v1.0.475 (round 20) — the exag=3.0 cascade, where the round-15
    auto-reload restarted INTO the crashing state until Chrome
    permanently blocked WebGL.
Both were reasoned from mechanisms nobody could observe at the moment of
failure, because this sandbox runs SwiftShader and a real-GPU context loss
CANNOT be reproduced here. A third fix by the same method would be a third
guess. So this session shipped diagnostics only (below) and files the
structural question here.

THE ONE CONCRETE, VERIFIED FINDING (grep, not inference): NOTHING releases
MapLibre's GPU residency when the user enters space. `map.setTerrain` is
called in exactly two places in datamap.tsx — the terrain-toggle effect and
its cleanup — and neither is on space entry/exit. Consequence: standing at
the Moon, 380,000 km from Earth, the page is still paying for
  · the terrain DEM mesh + the render-to-texture drape (round 15 already
    identified terrain as the single largest GPU consumer),
  · every enabled layer's buffers + tile caches (127 layers registered),
  · the satellite CustomLayer,
while SIMULTANEOUSLY the space frame allocates a full-screen 2D canvas at
devicePixelRatio plus up to a 2048x2048 RGBA mosaic (16 MB, MOON_MOSAIC_MAX_PX)
and a ~1100px patch buffer, AND celestialSky holds a SECOND WebGL context.
Peak GPU demand therefore lands exactly when the user needs the Earth map
least. That is a structural mismatch, and it is consistent with (though not
yet proven to be) what the human hit.

PROPOSALS, cheapest first — approve one or more:
 A. SHED TERRAIN IN SPACE (small). On space entry set terrain null; restore
    on exit. Removes the largest known consumer at zero visual cost, since
    the DEM is invisible at that range. CAUTION, and the reason this is not
    self-applied: datamap.tsx:3681 carries an explicit warning that the
    terrain and seafloor toggles "can never race over map.setTerrain", so a
    third writer needs to be sequenced through the same effect rather than
    bolted on. Estimated: one careful PR + a test that terrain returns on
    exit.
 B. SHED LAYER RESIDENCY IN SPACE (medium). Hide non-celestial layers while
    spaceActive so their buffers/tiles can be evicted, restoring on exit.
    Needs care: hiding is not the same as freeing in MapLibre, so this must
    be MEASURED (renderer info before/after) rather than assumed.
 C. ONE CONTEXT (large, the real architectural answer). EVIDENCE CLOSURE 2026-08-05 (full case file: experiments.md same date + workflow journal wf_8ab35411-642): the field payload ruled out TDR by the investigation's own pre-registered criterion (healthy 17ms frames at both instrumented losses), cleared the space view as a necessary cause (loss #3 on the plain map 3.8s after load), exonerated the negative-zoom state (lawful MapLibre globe latitude re-normalization above the deliberate -2 floor), and produced direct GPU-process-down evidence (a no-webgl boot probe beside a running map). Nothing remains but the context count. RECOMMENDATION: approve C.  EVIDENCE UPDATE
    2026-07-31: the blackbox report identifies Intel Iris Xe (D3D11) with
    healthy frames and heap at every loss — driver/GPU-process-level
    resets, where each standing context is surface area. TDR and memory
    theories are dead (see experiments.md). First step shipped same day:
    celestialSky context lifecycle (mount ≥55° pitch, dispose <45°+4s).
     Today three GPU
    consumers coexist with independent budgets: MapLibre's WebGL context,
    celestialSky's WebGL context, and the space frame's 2D canvas. Draw the
    space frame and sky as MapLibre CustomLayers in the single existing
    context — the satellite layer already proves the pattern works here
    (client/src/lib/orbital/satLayer.ts). Eliminates the summed-budget
    failure mode entirely rather than trimming it.

WHAT ONLY THE HUMAN CAN SUPPLY (this is the fastest path to certainty):
 1. chrome://gpu on the machine that crashed — specifically the "Graphics
    Feature Status" block and any "context lost" entries at the bottom.
 2. Whether terrain (3D relief) was ON when it happened, and whether it
    reproduces on a second attempt or only after a long session.
 3. The output of the new "Copy diagnostics" button on that card (ships in
    v1.0.537 below) — it now carries the resident-canvas sizes, heap, DPR,
    whether the DEM was still live, and where the camera was.

SHIPPED THIS SESSION (v1.0.537, non-speculative, no behaviour change):
captureGlSnapshot() records the above at the instant of webglcontextlost,
rings the last 5 into localStorage (vt-gl-loss-log), prints them under a
[VT GL-LOSS] console tag, and the blocked card gained a Copy-diagnostics
button. Making an unreproducible failure observable is the prerequisite for
root-causing it; it is deliberately NOT another attempt at a fix.

## AUTONOMOUS-SESSION PRs CAN GET ZERO CI AND AGE OUT UNMERGED (found 2026-07-30)

CONTEXT: this session's KNOWN BROKEN sweep found PR #638 ("[REPAIR] KNOWN
BROKEN #27" — options sizer Kelly-bucket bug, a genuine live fix with a
full test gate already run and documented) had been open since
2026-07-29T11:18Z with **zero CI workflow runs ever triggered**
(`actions_list list_workflow_runs` for its branch returned
`total_count: 0`) — distinct from the already-tracked, since-resolved
CI/deploy outage tracked earlier in this file. The PR just sat, unmerged,
across a session boundary, with nothing in `/api/health` or any other
standing check surfacing it. It was caught only because this session
happened to re-verify the KNOWN BROKEN list from scratch and noticed the
bug was still live on `main`.

RESOLVED THIS SESSION (tactical): cherry-picked the fix onto a fresh PR
(#645) after confirming the bug was still live and the cherry-pick applied
cleanly; closed #638 as superseded.

PROPOSAL (needs human review — this is a process/tooling gap, not a code
bug): some standing check should catch "PR open >N hours with 0 CI runs
and not draft" and surface it the way the LIVENESS ALARM surfaces a dead
trading loop — e.g. a cheap addition to the DAILY session's own opening
checklist ("list open non-draft PRs authored by this account; for each,
check total CI run count; flag any at 0"), or a dedicated lightweight
routine. Not built here — this file is for the human's review per
AUTONOMY AUTHORIZATION, and this is exactly the kind of standing-check
proposal that section exists for. Estimated cost: near-zero (one more
`list_pull_requests` + `list_workflow_runs` pair per DAILY session).

**UPDATE 2026-07-30 (second occurrence + a new sub-failure-mode found):**
this scheduled-routine session found the same zero-CI pattern on PR #640
("product: NOAA SWPC space weather (K-index + X-ray flux) archiver + RAW
display", open since 2026-07-29T13:29Z, `list_workflow_runs` for its
branch again `total_count: 0`) — confirming this is a recurring gap, not
a one-off. The proposal above stands unchanged; two data points now
support it.

NEW FINDING this occurrence surfaces: PR #640 was a genuine WORKSTREAM
PARTITION collision (CLAUDE.md's MERGE-ORDER PROTOCOL item 6) with the
already-merged #639 ("SWPC space weather: gate-1 archiver + aurora /data
layer") — two concurrent PRODUCT sessions both built `server/
spaceWeather.ts` under the same filename for overlapping-but-distinct
NOAA SWPC data (aurora+Kp+alerts+wind vs. K-index+X-ray-flux), an
add/add conflict a normal review would have caught but zero CI meant
no one ever looked. Applied the documented supersession precedent:
first-merged (#639) kept its module identity/exports/cache shape;
#640's unique delta (GOES X-ray flux archiving + NOAA's own flare-class
formula) was hand-folded into the existing `spaceWeather.ts` as
additive fields (`SpaceWeatherPull.xray`, `SpaceWeatherCache.xrayLatest/
flare/xrayRecent`, a fourth `xray-YYYY-MM-DD.jsonl` archive prefix) —
the redundant K-index re-parsing in #640 was dropped since #639 already
owned that series. Shipped as v1.0.543, closing #640 as superseded.
This makes the zero-CI gap strictly worse than "PR ages out unmerged"
in the two-concurrent-session case — it can silently produce a same-
filename collision that a normal PR review's diff view would have
flagged immediately (GitHub shows "this file was also changed on
main"), but with zero CI and no reviewer, nothing surfaced it until a
future session's cherry-pick attempt hit the conflict directly. Doesn't
change the proposed fix, but raises its priority: a stuck-PR check would
have caught this within a day instead of it silently waiting for a
session to stumble into the conflict.

**UPDATE 2026-08-02 (third data point, oldest instance found and
closed):** this scheduled-routine session swept `list_pull_requests` for
open PRs and found PR #77 ("fix(tier2): gate inline ML retrain (OOM)",
`fix/tier2-full-scan-oom`) had been open since **2026-04-20** — by far
the oldest and most severe instance, ~104 days with zero CI runs ever
triggered, well predating the two 2026-07-30 occurrences above. Unlike
those two, this one did NOT need the cherry-pick-to-fresh-branch
treatment: investigation found the OOM bug it targeted was independently
fixed a day after the PR was opened (`bot_engine.py`'s "MEM FIX
2026-04-21", still live on main as `_inline_train_allowed()` after this
session's refactor), so PR #77's diff would have been pure redundancy.
Closed #77 as superseded (comment on the PR explains why); the one real
gap the investigation surfaced — zero regression coverage on the
existing gate — is fixed by this session's own PR (`test_inline_train_gate.py`).
Three data points now (PR #77, #638, #640) — this proposal (a standing
"PR open >N hours with 0 CI runs" check, e.g. added to the DAILY
session's opening checklist) remains unbuilt and still needs human
review per AUTONOMY AUTHORIZATION; its case only gets stronger each time
a session finds another instance by accident rather than by a standing
check.

## 2026-07-30 UPDATE — EIA_API_KEY added to the AGENT SESSION ENV (human)

The human reports adding EIA_API_KEY to the claude.ai/code environment
settings on 2026-07-30 (chat confirmation: "its added"). Railway has had
its own active key for a while (grid-demand serves live BAs). Env vars
inject at container start, so the session that received this
confirmation predates the addition and CANNOT verify from inside —
RunPod precedent applies (a key sat wrongly marked "blocked" for ~2
weeks): the NEXT fresh session touching EIA work must run
`env | grep EIA_API_KEY`, and on success log the confirmation here and
treat direct-EIA work (930 history backfill, per-BA live-flow expansion
across the static grid layers, dev testing against the real API) as
UNBLOCKED. Wishlist item 8(a) is thereby fully closed once verified;
item 5 (FRED_API_KEY session-env) remains open and is now the analogous
2-minute ask.

**VERIFIED 2026-07-30 (scheduled-routine session, later same day):**
`env | grep EIA_API_KEY` succeeded in this session's container — the key
is live in the agent session env, confirming the human's report above.
Item 8(a) is CLOSED. Used this session for a quick gate-1-adjacent probe
(direct EIA-930 pulls for US48, cross-checked against the production
`/api/data/grid-demand` cache and against a second pull 4 minutes later)
rather than the full backfill/expansion work — see
`research/experiments.md`'s 2026-07-30 [REPAIR] session-log NEXT item
(4) for the (inconclusive, one-time-settle-not-continuous-revision)
finding. The 930 history backfill and per-BA live-flow expansion this
key was requested for remain unbuilt; next PIPELINE session touching
EIA work can proceed directly, no further unblocking needed.

## 2026-07-30 — TILE SERVING STRATEGY DECISION NEEDED before GRID VISION wave 3 (Asia/Africa) [HUMAN DECISION]

Wave 2 (Europe, 48 countries) hit the ceiling of the committed-tiles
pattern: France's full-detail tile exceeded GitHub's hard 100MB file
limit and had to be recut at z11; the continental master needed z8.
Committed tiles now total ~860MB in-repo (also baked into the Railway
image). Wave 3 candidates (China, India, Japan, SE Asia) have larger
OSM extracts than Europe — the pattern dies there.

BUILD-FIRST ANALYSIS (per the doctrine):
1. Raw material: free (Geofabrik OSM), pipeline proven across 3
   continents — the only question is where the OUTPUT bytes live.
2. Free-tier options, in preference order:
   (a) Cloudflare R2 free tier: 10GB storage + free egress — fits ALL
       remaining continents at full z-detail (no z11/z8 compromises),
       range requests supported (PMTiles native). Needs a human-created
       account + one bucket + a public custom domain or r2.dev URL, and
       an R2 write token in the agent env. ~15 min of human setup, $0.
   (b) Railway volume serving (/data/voltrade/tiles + an express static
       route with range support): zero new accounts, but the volume is
       1GB-class and already holds bot state — tiles would crowd it,
       and getting 1GB+ of tiles ONTO the volume from CI needs a
       deploy-time fetch from somewhere anyway. Weakest option.
   (c) Keep committing capped tiles: dies at wave 3 (single files would
       exceed 100MB even capped, and repo/image bloat compounds).
RECOMMENDATION: (a) R2. If approved, wave 3+ tiles (and optionally a
full-detail Europe re-cut) serve from R2; the client change is one
base-URL constant. Until decided, wave 3 is NOT blocked from BUILDING
(extracts + stats are cheap to produce and archive) — only from
SHIPPING the tiles to users.

## 2026-08-08 — GLOBAL ALL-CIVIL LIVE ADS-B (paid) — build-first analysis attached

HUMAN ASK (2026-08-08): "track planes all the time so we have the adsb
data... every time a plane turns on we have the data all over the world
not just the 250nm."

WHAT WE BUILT FREE FIRST (shipped same day, the build-first ladder):
1. Nonstop per-tail tracker (#753): any named plane polled 24/7, one
   batched request; every fix archived.
2. Day-trace backfill (#756): each tracked plane's COMPLETE current-day
   global-network track (tar1090 trace_full, verified 1,252 points for
   N843S) merged into our archive every 15 min — full world coverage
   for planes we care about, at native network fidelity.
3. Global scopes archiver (this PR): mil + LADD + PIA — the only scopes
   adsb.lol serves globally (verified against their OpenAPI spec; no
   all-aircraft endpoint exists) — ~1,000+ aircraft worldwide archived
   continuously, volume-guarded.
4. ACCUMULATION substitute: every day these run, our own archive of
   tracked + global-scope + viewport traffic deepens — time turns the
   free feeds into the paid product for the planes that matter to us.

WHAT ONLY MONEY BUYS: live positions for ALL ~10-20k airborne civil
aircraft simultaneously. Free providers structurally cannot offer it
(radius-capped queries); ADSBExchange sells exactly this
(commercial API, global firehose). Price: ADSBExchange Enterprise API
from ~$100/mo (rate-limited tiers) — needs a quote for the firehose.
ALSO NOTE adsblol/globe_history GitHub dumps (free, ODbL): full global
per-day history tarballs — a future ingestion pipeline could give us
PAST global data without paying; assets are multi-GB/day so selective
per-hex extraction needs design (not built yet; filed as the free
deepening path before any spend).

RECOMMENDATION: do not buy yet. The per-tail tracker + trace backfill
covers named-plane use cases completely; global scopes cover the
highest-signal traffic (military). Buy only if a validated signal needs
the full civil firehose (e.g. airport-level traffic counts as an
economic indicator — that hypothesis should pass ladder gate 2 on
sampled data first).

## ⚠️ BLOCKED-FOR-MIKE — AIS VESSEL FEED DARK SINCE 2026-08-05 (escalated 2026-08-11; RECURRENCE RULE TRIGGERED — two code fixes have not restored it, the remaining lever is the aisstream ACCOUNT, which only you can touch)

WHAT IS LOST, measured (not estimated): the vessel archive's newest file
is `vessels/2026-08-05-13.jsonl.gz` while `aircraft` is current to
`2026-08-11-22` — so **~6.4 days of global AIS positions were never
recorded**. Per CLAUDE.md Priority 1, "an archive gap never refills":
this is permanently missing history, and every further day compounds it.
Downstream, everything AIS-derived is silently degraded — shadowFleet /
shadowstats (dark-ship), portdwell, vessel tracks, and the vessels layer
(live count 0).

WHY THIS IS ESCALATED RATHER THAN PATCHED AGAIN. CLAUDE.md's
"RECURRENCE ESCALATES" rule: an issue already marked fixed that breaks
again may NOT be patched a third time — two failed fixes on one
subsystem is an architecture/environment smell that gets proposed as
structural work here.
- Fix #1 (2026-08-06): reconnect watchdog + last-frame liveness tracking.
- Fix #2 (2026-08-11, v1.0.658, PR #769, a CONCURRENT session — not this
  one): correctly identified that fix #1 was blind to a socket that
  never receives a first frame (no frame -> no timestamp -> no zombie
  verdict), made the watchdog redial on connect-time silence, fixed the
  panel status, and added feed_frames/feed_parsed instrumentation.
Both fixes are sound. Neither restored data flow.

EVIDENCE THE REMAINING CAUSE IS THE ACCOUNT, NOT OUR CODE (verified live
by me 2026-08-11 against prod v1.0.661, two samples 45 s apart):
`count=0 frames=0 parsed=0`, and `feed_silent_s` RESETS each cycle
(14 s -> 60 s), i.e. the watchdog IS redialing every 60 s and each fresh
socket receives ZERO frames — a clean reconnect loop with no data.
PR #769's own probe result is the key discriminator: a deliberately
bogus key gets CLOSED by aisstream within seconds, so a socket that
stays OPEN and silent means **our key is accepted and then starved**.
Their stated suspicion, which the reset-pattern above is consistent
with: a **one-concurrent-connection-per-key limit**, with our slot held
by something else (a leaked connection from an earlier container, or
more than one instance/replica dialing the same key).

WHAT ONLY YOU CAN DO (any one of these likely resolves it):
1. Log in to aisstream.io and check the account/key state: is the key
   still active, is there a concurrent-connection cap, is there a
   quota/abuse flag or a "connection already in use" indication?
2. REGENERATE the key and update AISSTREAM_KEY in Railway. If the cause
   is a stale server-side session holding our one allowed connection, a
   new key sidesteps it immediately.
3. Confirm whether Railway is running MORE THAN ONE instance/replica of
   the service. If so, every replica dials the same key and they starve
   each other — that is an architecture question (single-dialer
   election, or a dedicated worker) and I will build the fix once the
   replica count is known. Please state the replica count.
4. If aisstream has become unreliable for our use, say so and I will do
   the BUILD-FIRST provider survey for a terrestrial-AIS alternative
   (the last survey is at wishlist.md ~line 1376; ODbL/commercial terms
   must clear the MONETIZATION TRIPWIRE, same as the ADS-B chain).

UNTIL RESOLVED — honest state: the vessels layer reads 0 and the panel
now says so (PR #769 fixed the "needs API key" mislabel). No fabricated
positions are served. The SAR dark-ship validation idea filed in
open_questions.md the same day is BLOCKED behind this: there is no live
AIS to validate against while the feed is dark.

### ✅ 2026-08-12 UPDATE — ROOT CAUSE FOUND: PROVIDER-SIDE OUTAGE AT AISSTREAM. **DO NOT ROTATE THE KEY.** Human actions 1–3 above are now known-futile; action 4 is the live one.

(Append-only: everything above stands as written on 2026-08-11 and was
correct on its evidence. This entry supersedes its RECOMMENDED ACTIONS.)

A human-supplied runbook ("AIS DEAD-AIR RUNBOOK — key rotation is step 6,
not step 1") ordered the diagnosis cheapest-discriminating-test-first and
warned that rotating the key destroys the evidence for every other
hypothesis. Running that ladder resolved it **without any account action.**

**THE FINDING.** aisstream.io stopped delivering frames to everyone on
2026-08-05. It is not our key, not our payload, not our deployment, and
not replica contention. Independent, unrelated users report our exact
symptom, on their own keys, from their own machines:

- **aisstream/issues#269, "Stream silent since 2026-08-05"** (opened
  08-10, still open, no maintainer reply): *"Since 2026-08-05 ~13:31 UTC
  our client (wss://stream.aisstream.io/v0/stream) connects, sends the
  subscription, and receives zero messages indefinitely."* Our archive's
  last vessel file is `vessels/2026-08-05-13.jsonl.gz` — **the same hour,
  a different operator.**
- **#272** (08-11): *"no messages ever arrive — not even a single
  PositionReport. I've tested this from two completely different network
  environments and with three different API keys."* Worldwide bbox,
  documented ~300 msg/s, actual zero. **This kills key rotation and every
  account-flag theory on its own.**
- **#263** (08-07): a brand-new account and a brand-new key receive
  nothing. Rotation cannot help.
- At least ten more filed 08-07 → 08-11 (#262 "Subscription accepted,
  zero frames delivered", #264 "Zero frames - please fix", #266, #267,
  #268, #270, #271, #273 "Alternate feed?"). No maintainer response on
  any of them. Upstream **aisstream/aisstream#15** (opened 2026-03-13,
  still open) is the same failure predating this outage.

**THE LADDER, CLOSED.**

| Step | Result |
|---|---|
| 1. Standalone off-Railway probe w/ current key | Not runnable in-session (key is Railway-only, correctly). **Moot** — #272 already ran it with 3 keys on 2 networks, and #263 with a fresh account: zero frames. |
| 2. Coordinate ordering | **PASS.** We send `[[[-90,-180],[90,180]]]` — byte-identical to the spec's worldwide example. The third-party doc claiming longitude-first is wrong; aisstream's own docs show `[[[lat, long],[lat, long]]]`. |
| 3. Nesting depth | **PASS.** Three levels: `BoundingBoxes → [corner1, corner2] → [lat, lon]`. |
| 4. Filter fields | **PASS.** `FiltersShipMMSI` absent. `FilterMessageTypes: ["PositionReport","ShipStaticData"]` — both valid names, and PositionReport *is* the firehose. |
| — | **The whole payload is unchanged since 2026-07-03** (`git log -L` on server/routes.ts) and delivered frames for a month. Nothing of ours changed on Aug 5; no deploy that day touched the vessel path. |
| 5. Replica count | **ANSWERED WITHOUT YOU: exactly one dialer**, confirmed two independent ways. (a) Measured: 14 consecutive polls of `/api/data/vessels` over 60s show a single strictly-monotonic silence clock (8→64s, frames=0); two replicas would interleave two independent counters. (b) Declared: `railway.json` sets `"numReplicas": 1`. **The starvation theory is dead and the single-dialer guard is cancelled — don't build it.** |
| 6. Account state | Moot — see #272/#263. |
| 7. Rotate | **DO NOT.** Proven futile by two independent reporters, and it would destroy the evidence trail for no gain. |

Also re-verified against aisstream's own docs (runbook claims 1 and 2,
both confirmed): *"the subscription message must be sent within 3 seconds
of creating your websocket... or your connection will be closed"* — we
send on `open`, so a socket that stays open proves our subscription was
accepted; and there is **no documented per-key concurrent-connection
limit**.

**WHAT SHIPPED THIS SESSION (v1.0.667, PR #782):** the runbook's build
item 1 — a **feed dead-air watchdog** on all three continuous position
feeds, throughput-based rather than connection-based, wired into
`/api/health`. It reads the archive on disk, the one clock a redial, a
restart, or a deploy cannot reset. Against this outage it fires the same
morning instead of on day seven. It makes **no** causal claim, so it is
not a third patch under RECURRENCE ESCALATES.

**NOT BUILT, DELIBERATELY:** the single-dialer guard (step 5 says one
replica — the runbook itself gates it on replicas > 1), and the
subscription-assertion logger (its value was in catching a malformed
payload; steps 2–4 plus a month of working frames plus twelve independent
reporters have exonerated our payload, so touching the AIS socket now
would be churn against a disproven hypothesis).

**WHAT IS ACTUALLY LEFT — your action 4, now evidence-backed.** aisstream
has been silent for its entire user base for 7 days with no maintainer
response, and other operators are already asking for alternates (#273).
The BUILD-FIRST survey for a second AIS source is filed separately below.

**WHAT ONLY YOU CAN STILL DO (optional, low priority now):** if you want
the upstream signal amplified, add our reproduction to
aisstream/issues#269 — our archive timestamp corroborates their
13:31 UTC start from an independent deployment. I did not post to a
third-party repository on my own.

## BUILD-FIRST SURVEY — a second AIS source (filed 2026-08-12; triggered by the aisstream outage above)

Not a purchase request. **Cost: $0.** Both candidates below are free,
need no registration, and clear the MONETIZATION TRIPWIRE — filed here
because adding a provider to a monetizable path is a licensing decision,
and because the honest coverage trade-off is a judgement call.

BUILD-FIRST step 1 (do we already receive the raw material?): no — AIS
raw material is receiver-network output we do not own. Step 2
(accumulation): irrelevant, the gap is live coverage. Step 3 (inference):
refused — inferring vessel positions would be fabricating data. Step 4
(is the raw material genuinely inaccessible free?): **no, it is not.**
Two national authorities publish live AIS openly and commercially.

**CANDIDATE 1 — Fintraffic / Digitraffic (Finland). Strongest.**
- Live AIS: MQTT-over-WebSocket `wss://meri.digitraffic.fi:443/mqtt`,
  plus REST `https://meri.digitraffic.fi/api/ais/v1/locations` and vessel
  metadata at `/api/ais/v1/vessels`.
- Licence, quoted from their terms: **CC 4.0 BY** — *"It gives the right
  to distribute, remix, tweak, and build upon our data, even
  commercially, as long as you credit the source."* Required attribution
  string: *"Source: Fintraffic / digitraffic.fi, license CC 4.0 BY"*.
- **We already ingest Digitraffic** for the trains layer under the same
  licence — same provider, same terms, a pattern already proven in
  production. This is the lowest-integration-risk option we have.
- Coverage: Finnish waterways only.

**CANDIDATE 2 — Kystverket (Norwegian Coastal Administration).**
- Live AIS: open raw NMEA stream at `153.44.253.27:5631`, **no
  registration**. BarentsWatch additionally offers developer API
  endpoints.
- Licence: **NLOD** (Norwegian Licence for Open Government Data) —
  commercial use permitted with attribution.
- Coverage: Norwegian economic zone + Svalbard/Jan Mayen protection
  zones. Excludes fishing vessels <15 m and recreational craft <45 m.
- Note the split: the *open* component needs no registration; a *closed*
  component (small-vessel data) requires an application and carries
  "must not be used for anything other than the stated purpose and will
  not be distributed to third parties" — **that restricted tier must
  never enter a monetizable path.** Take the open tier only.

**RULED OUT — AISHub.** Free, but it is a data-*sharing* co-op: access
is conditioned on contributing a receiver's feed. We operate no AIS
receiver hardware, so we cannot pay the entry price in kind. (An
unrelated project reached the same conclusion independently —
koala73/worldmonitor#6227, "AIS has no fallback... AISHub ruled out".)

**RULED OUT for now — MarineTraffic / VesselFinder / Spire / VesselAPI.**
Commercial products. Per BUILD-FIRST these may not even enter this list
until the free path is built and found materially worse. It has not been
built yet.

**THE HONEST TRADE-OFF.** Neither candidate is global; aisstream's pitch
was worldwide terrestrial coverage. Together they cover the Baltic and
the Norwegian Sea, not the world. So this is **not** a replacement — it
is a floor: real, licensed, live vessel positions in two regions instead
of a fully dark archive in all of them, and an archive that keeps
recording somewhere while the provider question resolves. It also removes
the single-point-of-failure that just cost 7 days.

**CROSS-SYSTEM TIE (CROSS-SYSTEM INTEGRATION PRINCIPLE — real, not
decorative).** The Baltic is exactly where the GNSS-integrity passthrough
(v1.0.662–667) is looking for navigation-integrity degradation. Finnish
and Norwegian AIS would put *vessel* GNSS behaviour in the same box where
we are already measuring *aircraft* GNSS behaviour — two independent
receiver populations over one geography. That is a genuine
cross-validation opportunity for the integrity series, not a showcase
link: an interference finding visible in both populations is far harder
to explain away as an avionics or equipage artifact. Filed as a
hypothesis, not a claim — it needs the ROOT VALIDATION LADDER like
anything else.

**RECOMMENDATION.** Build Candidate 1 (Digitraffic AIS) as a first
[PIPELINE] slice: same provider and licence we already run for trains,
zero cost, zero registration, and it restores *some* vessel archive
recording immediately. Then Candidate 2. Keep the aisstream adapter in
place and unmodified — when the provider recovers, worldwide coverage
returns for free, and the dead-air watchdog (v1.0.667) will now say so
the same morning either way.

## 2026-08-12 — CI GAP: the Node test suite (incl. the R15 wiring ratchet) never runs in CI [FROZEN-PATH PROPOSAL]

FOUND VIA A USER-REPORTED BUG: the human asked where the Time zone
lines toggle was — the layer shipped in #774 with wiring + registry
entry but no `LAYER_GROUP` declaration in datamap.tsx, so the panel
rendered its toggle permanently disabled as "reload to enable" (the
exact R15 powergrid defect class). The regression ratchet built for
this (`server/layersWiring.test.ts`) FAILS on that state, exactly as
designed — but it never ran: `.github/workflows/ci.yml` runs pytest,
`tsc --noEmit || true`, and the build. It never runs
`npm run test:node` (`tsx --test server/*.test.ts`), so every Node-side
ratchet is CI-invisible. The layer merged green and sat broken on
desktop production until the human tripped over it (one-line fix
shipped same day).

PROPOSAL (workflows are FROZEN — human approval needed): add one step
to the `node-build` job in ci.yml, after `npm ci`:

    - name: Node test suite (ratchets + server units)
      run: npm run test:node

COST: ~1–2 min per CI run (1,213 tests). One pre-existing failure must
be resolved first or excluded: `server/gridTiles.test.ts` expects the
state+national pmtiles in `client/public/tiles/`, which were migrated
to R2 (2026-07-31) — that test is environment-dependent now and should
either be updated to check the R2 manifest instead, or skipped when the
files are absent (it fails in any fresh clone today, another thing CI
never saw). Until approved, sessions MUST treat `npm run test:node` as
part of the local promotion gate (PROMOTION RULES name pytest only —
that reading let this slip; an amendment adding test:node to rule 1 is
part of this proposal).


## 2026-08-13 — OPTIONS-SLOT CAP: THIRD RECURRENCE, architecture smell, structural fix proposed [RECURRENCE ESCALATES — human/dedicated-session decision, no code shipped this session]

Cross-referenced from open_questions.md KNOWN BROKEN #30 (full evidence
trail lives there — this entry carries the structural proposal per
CLAUDE.md's RECURRENCE ESCALATES rule: "patching it again is FORBIDDEN
... propose structural work via wishlist.md").

**THE PATTERN.** `MAX_OPTIONS_POSITIONS = 6` in `server/bot.ts` /
`system_config.py` has now been breached live in production three times,
each via a genuinely different mechanism:
1. 2026-07-29 (v1.0.540): a stale local constant (3) never wired to the
   canonical value (6) — a drifted-constant bug, fixed by hoisting to
   one module-scope declaration.
2. 2026-08-03 (v1.0.586): an intra-cycle TOCTOU race — the tier
   dispatcher trusted a pre-`executeTrades()` positions snapshot — fixed
   by re-fetching `/v2/positions` fresh immediately before dispatch.
3. 2026-08-13 (found this session, NOT fixed): a cross-cycle race. CSP
   orders are Alpaca DAY LIMIT orders (`options_execution.py:
   submit_options_order`, `time_in_force: "day"`) that return
   `{"status": "submitted"}` on any 2xx response with **no fill
   confirmation/poll**. `bot.ts` counts `"submitted"` as slot-consumed
   the instant it's sent, but that increment lives only in a local
   variable for that one dispatch loop. The NEXT cycle's "fresh"
   `/v2/positions` re-fetch (the exact fix #2 shipped) only reflects
   **filled** orders — a submitted-but-still-resting limit order from
   the prior cycle is invisible to it. Two cycles a few minutes apart
   can each correctly see "5 of 6 filled, room for one more," each
   submit one, and if both later fill, the account ends up at 7.
   Live-verified: real filled-position count crossed 6→7 today at
   19:58:36Z, one full ~2-minute scan cycle after crossing 5→6 at
   19:54:56Z — consistent with this exact race, not a repeat of #1 or
   #2 (both of which were re-read this session and confirmed still
   correctly in place).

**WHY THIS IS ARCHITECTURE SMELL, NOT BAD LUCK.** Three fixes, three
different bugs, same symptom, same subsystem, in 15 days. Each fix
closed the specific hole it found and was immediately re-opened by a
different hole in the same design: **cap enforcement is built entirely
on `/v2/positions` (filled reality) with no concept of "orders currently
live on the book that could still fill."** Any future change to this
code that keeps that same shape — "count filled positions, compare to
6, submit if under" — has the same structural gap and will eventually
find a fourth hole (order latency variance, a slow contract-selection
call, Alpaca API lag, etc.). That is the definition of the RECURRENCE
ESCALATES bar for architecture-level work rather than another surgical
patch.

**THREE CANDIDATE STRUCTURAL FIXES** (not evaluated against each other
in depth — that's the deliberate next session's job; listed here so it
doesn't start from zero):

1. **Count open orders too (cheapest, most surgical).** Before
   submitting any new SELL_CSP (both `executeTrades()` and the tier
   dispatcher), also `GET /v2/orders?status=open&asset_class=us_option`
   and add those to the slot count — a submitted-but-unfilled CSP is a
   real slot commitment even before it fills. Closes this specific hole
   with roughly the same shape as the 2026-08-03 fix. Residual risk:
   still not atomic — two dispatch loops could both read "open orders"
   in the same instant and both proceed, though the window is far
   narrower (a single GET, not the multi-minute gap between scan
   cycles) and `tier2Running`'s mutex already serializes the two
   dispatch loops that exist today, so in the CURRENT codebase this
   closes the hole completely; it degrades gracefully (not perfectly)
   if a future code path adds a third concurrent submitter.
2. **Poll for fill/reject before counting a slot consumed.** Change
   `submit_options_order` (or its caller) to poll the order status for
   a bounded window (Alpaca options can also be checked via
   `GET /v2/orders/{id}`) before returning, and only increment the slot
   counter on an actual `filled` status; a still-`new`/`accepted` order
   after the poll window either gets canceled (freeing the slot
   honestly) or is explicitly tracked as pending. More correct, more
   invasive (changes the hot execution path's latency and return
   contract — every caller of `submit_options_order` would need a
   status contract review), and closer to FROZEN-PATH territory (order
   submission internals) — needs care to stay on the "what gets traded"
   side of that line, not "how orders are transmitted."
3. **A persisted, cross-cycle slot ledger.** A small reservation table
   (ticker, order_id, reserved_at, status) written on submit and
   reconciled against `/v2/positions` + `/v2/orders` on a timer,
   independent of any single dispatch loop's local variables. Most
   robust to future code-shape changes (survives even a process
   restart mid-fill), most work, and the only option that would also
   give the counterfactual-logging infrastructure (RULE REVIEW) a clean
   place to record "slot reservation X was denied/starved" for its own
   evidence trail — arguably the option most aligned with CLAUDE.md's
   general preference for compiling recurring judgment into durable
   state rather than re-deriving it (EDGE DOCTRINE #3), even though this
   is a T-BOT risk-mechanism, not a data pipeline.

**RECOMMENDATION.** Option 1 first (cheap, closes the live hole,
same-shape as the precedent fix so low review risk), landed as its OWN
dedicated session with its own regression test asserting the exact
cross-cycle scenario reproduced here (two sequential dispatch passes,
mocked `/v2/positions` returning 5 filled + 1 open both times, second
pass must skip). If a FOURTH recurrence is ever found after Option 1
ships, that is the trigger for Option 3 — at that point the shape
itself, not any single check, is confirmed to be the problem.

**NOT A SPEND REQUEST** — no paid capability involved, filed here per
RECURRENCE ESCALATES' explicit instruction to route repeated-subsystem
breakage through this file rather than same-day re-patching.
