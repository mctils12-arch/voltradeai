# Experiment Log

Append-only. Newest at top. Never rewrite history (CLAUDE.md — MEMORY PROTOCOL).
Each entry: date · change · version tag · backtest result · hypothesis · (later) live-vs-backtest.

## AUDIT REGISTER (maintained in place per the AUDIT CYCLE clause,
CLAUDE.md SESSION BUDGET — this block is updatable state, the only
exception to append-only; the log below it stays append-only)

| audit | cadence | last run |
|---|---|---|
| staleness audit (code/deps/config/expired adapters — DEAD CODE POLICY governs) | 30d | 2026-07-05 COMPLETE (both sides). Server side: all 23 env reads wired; adapters none expired, next review 2026-08-17. Python side: requirements.txt zero unused; 6 session-run deps undeclared → requirements-dev.txt; VOLTRADE_STATE_DIR dead env write removed; vacuous-pass test sweep 1 low fix, 3 judged acceptable (see log entry). Next full pass due 2026-08-04 |
| constitutional audit (rules — CONSTITUTIONAL HYGIENE governs) | 30d | 2026-07-04 (human-directed CONSTITUTIONAL REPAIR: 4 proposals filed in wishlist.md, awaiting approval) |
| market_calendar year-add (FROZEN PATHS exception governs) | December | 2026 dates present; add 2027 in Dec 2026 |

## 2026-07-08 — [PIPELINE] USGS real-time earthquakes — free-data pipeline, EDGE DOCTRINE #1 (v1.0.209)

TERRITORY: T-DATACORE (server/usgsQuakes.ts, server/usgsQuakes.test.ts,
datacore/manifests/earthquakes.json) + SHARED (server/routes.ts new route
+ import, package.json version, this file, wishlist.md, open_questions.md
— all minimized to the last commit per MERGE-ORDER PROTOCOL).

- SESSION START per MEMORY PROTOCOL: read CLAUDE.md (EDGE DOCTRINE
  emphasized per the task brief), all of research/. LIVE HEALTH CHECK:
  `GET .../api/health` returned 200 — server/db/Alpaca/python-bridge/bot
  all healthy, `bot.liveness.dark:false`, drawdownPct 0.0 — no LIVENESS
  ALARM condition, nothing to surface top-of-report.
- KNOWN BROKEN triage (repair-mandate-first check): every item is
  RESOLVED, gated on human-only live-audit-log access (#3/#4), gated on
  accumulating shadow-portfolio/live-verification evidence not yet ripe
  (#10, #12b/c), or — item #14 — turned out to be ALREADY FIXED in code
  (v1.0.200, PR #351, 2026-07-07) despite open_questions.md still
  reading "not repaired this PR, deliberately". Confirmed via
  `git log -p -- server/tier3ManipVisibility.test.ts` and reading the
  current `tier3Strategic` catch block directly (it already audits
  `TIER3-MANIP-ERROR`) — not a new repair, a stale-documentation bug.
  Fixed the doc in this PR (cheap, high-value: prevents a future session
  from re-diagnosing an already-closed item) and flagged the still-open
  near-duplicate PR #343 as redundant rather than closing it myself
  (GitHub state changes outside this session's explicit scope). Also
  caught and fixed a second stale note while reading R5 (Everything
  Graph): the `/data` graph panel (step 3) has been live since
  `client/src/pages/graph.tsx` shipped, but open_questions.md still
  called it "unclaimed" — a prior session (2026-07-08 FINRA-part-2
  entry) had already flagged this exact staleness without fixing it;
  fixed here. No LIVE CODE repair remained actionable without owner-
  gated audit-log access this session had no path to — so this became
  a primary-axis EDGE session per the task brief, not a REPAIR session.
- PRIMARY ACTION SELECTION: task brief named 6 standing free-data-
  pipeline examples; checked each against research/ before picking —
  Sentinel-2 tank shadows, EDGAR Form4, CFTC COT, USAspending, and FDA
  calendars are ALL ALREADY BUILT (verified via experiments.md search,
  not assumed); Google Trends/pytrends already FAILED gate 1 (upstream
  archived 2025-04, replaced by Wikimedia pageviews). Checked open PRs
  (`mcp__github__list_pull_requests`) to avoid the CONCURRENT-SESSIONS-
  DOUBLE-BUILD failure mode in OPS GOTCHAS — #350 (satellite orbits
  client layer) and #343 (superseded manip-visibility duplicate) don't
  touch this scope. Landed on wishlist.md's DATACORE MAXIMUS NEXT
  pointer: "USGS quakes + NDBC buoys" — unclaimed, well-precedented
  (RAW keyless hazard feed, same shape as nasaFirms.ts), fully outside
  any hot/contested file (no datamap.tsx or routes.ts hotspot beyond a
  minimal SHARED addition). Picked quakes over buoys for this PR to keep
  one logical change (CLAUDE.md promotion rule 5) rather than bundling
  two unrelated domains; buoys is now the explicit next unclaimed item.
- LIVE CONTRACT VERIFICATION (READ BEFORE WRITE for an external API):
  curled `earthquake.usgs.gov/earthquakes/feed/v1.0/summary/*.geojson`
  directly before writing any code. Compared magnitude-threshold feed
  sizes live: significant_day=0, 4.5_day=17, 2.5_day=39, 1.0_day=193,
  all_day=242 (2026-07-08 sample) — picked 2.5_day as USGS's own
  recommended general-use default (below it, dense regional swarms swamp
  a global archive; above it, too sparse). Confirmed the feed's actual
  property/geometry shape (stable `id`, `[lon,lat,depth]` coordinates,
  `updated` timestamp that legitimately revises post-publish) directly
  from a live response, not from memory of "how earthquake APIs usually
  work" (READ BEFORE WRITE — CLAUDE.md is explicit that training
  knowledge does not describe any specific external API either).
- BUILD: server/usgsQuakes.ts mirrors nasaFirms.ts's shape (day-file
  JSONL archive, dedup Set, gzip-after-2-days, in-memory cache + eager
  boot poll) with one deliberate deviation: USGS's own `id` never
  changes but `updated` legitimately does (magnitude/location revised
  during review) — a pure id-dedup would freeze every event at its
  first-seen, often-"automatic"/unreviewed values forever. Added a
  parallel `archivedUpdated` Map so a newer `updated` timestamp for a
  known id re-archives it (append-only, both versions kept — no
  overwrite, matching the archive's forward-only philosophy elsewhere).
  Keyless — no gating needed (simpler than nasaFirms.ts's key-gated
  boot), boots eagerly per KNOWN BROKEN #9's lesson. No map layer this
  PR — pipeline+API-first sequencing, same precedent as sec8kEarnings/
  finraQuery part 1.
- ANGLE-HUNTING (standing behavior, filed not attempted): earthquake
  hazard-adjacency as an insurer/utility/supply-chain proximity signal
  — filed in open_questions.md under EARTHQUAKE HAZARD-ADJACENT
  HYPOTHESES with PRIOR stated (second-order reasoning per STANDARD #5:
  the edge, if real, comes from loss-reserve/reinsurance revision lag,
  not "nobody noticed a quake") and explicit gate-1/gate-2 ladder paths.
  Gate 1 is trivial here (USGS is ground truth, no separate verification
  needed) — gate 2 is honestly blocked on archive depth just started
  today, logged rather than faked.
- TESTS: 11 in usgsQuakes.test.ts — real-feed-shaped fixture (captured
  verbatim from a live GET during this build, ROOT VALIDATION LADDER
  gate-1 discipline matching nasaFirms.test.ts's VIIRS/MODIS fixtures),
  missing-id drop, tsunami-flag numeric/boolean coercion, fetch UA +
  error-status handling, archive dedup + the updated-timestamp re-
  archive path (caught and fixed a test-isolation bug from this
  session's own module-level dedup state colliding across tests before
  landing on unique per-test ids), gzip-after-2-days, idempotent boot.
  LIVE E2E: ran `fetchQuakes()` through this repo's own `tsx` against
  the real USGS URL (not just curl) — 38 live events parsed cleanly,
  confirming Node's own `fetch` + the parser work end-to-end, not just
  against a hand-shaped fixture.
- GATES: `npm install` first (node_modules was effectively empty in this
  sandbox — @types/node missing, `npm run test:node`'s `tsx` binary
  unresolved via npm script though `npx tsx` worked; installed to get an
  accurate tsc baseline rather than trusting a broken toolchain). node
  test:node 473/473 pass (462 baseline + 11 new — the 3 pre-existing
  failures this session first observed in compression/gdeltEvents/
  owmTiles were an artifact of the missing node_modules, not real: they
  pass clean post-install, confirmed via `git stash` A/B on the
  unmodified baseline). tsc 64 (unchanged baseline — one new Set-
  iteration TS2802 error was introduced and fixed in-session by
  switching to `.forEach()`, matching nasaFirms.ts's own established
  workaround for the same tsconfig target constraint). `npm run build`
  succeeds; `dist/datacore/manifests/earthquakes.json` confirmed staged
  (R14 packaging lesson — verify the POSITIVE case, not just error-free
  responses). `python3 -m pytest`: not runnable in this sandbox (pytest
  not installed) — no Python touched, matches PR #350's precedent that
  this doesn't block a client/server-only change.
  Version 1.0.208 -> 1.0.209 (read-and-increment; `git fetch origin main`
  confirmed the tip was unchanged from this session's branch point
  before bumping).
- Backtest: N/A — RAW data-pipeline build, no strategy/measurement code
  touched.

## 2026-07-08 — [PIPELINE] FINRA Query API part 2: ATS venue summaries (census build #4, part 2 of 2) — weeklySummary + monthlySummary + blocksSummary (v1.0.208)

- Territory: T-DATACORE (server/finraQuery.ts extension, server/finraQuery.test.ts,
  3 new datacore/manifests/*.json) + SHARED (server/routes.ts new route,
  package.json version, this file, wishlist.md). [PRODUCT] session per the
  task brief — filed as [PIPELINE] per CLAUDE.md's "[PRODUCT] counts as
  [PIPELINE] for progress-floor/thrash-ratio purposes" and to match part
  1's own tag (v1.0.170 entry below) for the same build.
- SESSION START per MEMORY PROTOCOL: read CLAUDE.md, all of research/.
  Loop-health ratio over the last 10 entries: 2/10 REPAIR — no thrash.
  LIVE HEALTH CHECK: `GET .../api/health` returned 200 before any other
  action — server/db/Alpaca/python-bridge/bot all reported healthy, no
  LIVENESS ALARM condition, nothing to surface top-of-report (a critical
  trading-loop break would not have blocked this PRODUCT session anyway
  per the task brief, but none existed to note).
- PRIMARY ACTION SELECTION: considered building the O2 orbital
  satellite-layer wiring (fully-specified NEXT step in
  research/orbital_program.md's RESUME STATE) but found open, unmerged
  PR #350 already adds a DIFFERENT "satellite orbits" layer to the same
  datamap.tsx file (ANALYST CONSOLE W2 track — server-polled, blocked by
  the CelesTrak/Railway firewall) — building O2's client-fetch GPU layer
  concurrently would create two competing "satellites" layers on one file
  and collide badly on merge (the exact CONCURRENT-SESSIONS-DOUBLE-BUILD
  failure mode in open_questions.md's OPS GOTCHAS). Also checked R5's
  Everything Graph panel (open_questions.md flagged step 3 "unclaimed")
  and found it was ALREADY SHIPPED (client/src/pages/graph.tsx, fully
  wired into datamap.tsx at #/data/graph) — that open_questions.md note
  was stale, not caught by this session's read (worth a housekeeping
  pass: R5 in open_questions.md still reads as step-3-unclaimed and
  should be updated to reflect completion — flagged here for whoever
  next touches R5, not fixed in this PR to keep scope to one logical
  change). Landed on wishlist.md's DATACORE MAXIMUS resume block's
  explicit "NEXT CENSUS BUILDS: FINRA part 2 (design notes above)" —
  fully-specified, unclaimed, builds on an already-shipped module
  (finraQuery.ts part 1), no collision risk with any open PR.
- LIVE CONTRACT VERIFICATION (READ BEFORE WRITE for an external API,
  not just internal code): curled api.finra.org directly from this
  session before writing any code — confirmed weeklySummary and
  monthlySummary are COMPOSITE-KEY partitions ([weekStartDate,
  tierIdentifier] and [monthStartDate, tierIdentifier] respectively),
  which the existing fetchPartitions/fetchPartitionRows primitives
  (built single-key-only for part 1) cannot express — a real gap the
  design notes hadn't flagged. Also live-measured weeklySummary T1's
  record-total at 66,080 for the newest week (exceeds part 1's
  MAX_PAGES=12 / 60k-row cap) and sampled real rows for all three
  datasets to confirm field shapes and the summaryTypeCode granularity
  mix (see BUILD below) — none of this was guessed from the design
  notes' prose alone.
- BUILD:
  - `fetchPartitionTuples()` (new) parallels `fetchPartitions()` but
    keeps the full composite tuple instead of flattening to `tuples[0]`;
    `fetchPartitions()` now delegates to it (zero behavior change for
    part 1's single-key callers, still covered by the existing test).
  - `fetchPartitionRowsMulti()` (new) generalizes pagination + the
    count-verify truncation guard to N EQUAL filters and a configurable
    `maxPages`; `fetchPartitionRows()` (part 1's original signature) now
    delegates through it with a single filter and the original
    `MAX_PAGES=12` default — existing part-1 tests pass unmodified,
    proving no behavior change. New `ATS_MAX_PAGES=50` (250k-row
    ceiling) used only for weeklySummary/monthlySummary.
  - `compositeKey()` joins tuple parts with `"__"` for the archive
    filename/dedup-set key; `seedSeen()`'s restart-reseeding regex
    generalized from a hardcoded `YYYY-MM-DD` pattern to match any
    value before `.jsonl(.gz)?` — needed since composite keys aren't
    dates.
  - Went with the design notes' "simpler" option (raise the page cap)
    over the async-job S3-presigned-CSV path — current live weeks fit
    comfortably under the new 250k ceiling; the notes flagged the async
    path as only mattering for ~210k-row 2021 backfill weeks, and no
    backfill is built for part 2 (matches part 1's own no-boot-backfill
    posture, R8 crash-loop lesson).
  - GRANULARITY HONESTY (live-verified, not assumed): a live full-page
    pull of weeklySummary 2026-06-15/T1 showed SIX summaryTypeCode
    values mixed in one partition (ATS_W_SMBL_FIRM 3263, OTC_W_SMBL_FIRM
    1248, ATS_W_SMBL 194, OTC_W_FIRM 168, OTC_W_SMBL 95, ATS_W_FIRM 32 in
    a 5000-row sample) — FIRM-level and per-symbol-per-firm rows would
    double- or under-count volume if ranked alongside the clean
    per-symbol cross-firm `*_SMBL` rows. `summarizeWeeklyBySymbol`/
    `summarizeMonthlyBySymbol` rank ONLY `*_SMBL` rows and report a
    `composition{}` count of every granularity actually present, so none
    of the excluded rows are silently dropped from view — they're
    accounted for, just not blended into a misleading leaderboard.
    `summarizeAtsBlocks` (blocksSummary) has no such split — one clean
    row per ATS venue/month with FINRA-precomputed ranks — ranked
    directly.
  - Empty tier/venue checks (e.g. weekly OTCE/NA when only T1 has data
    for the newest week — live-confirmed, not assumed) are deliberately
    NOT marked archived, matching part 1's existing 204-is-not-a-
    done-marker pattern for SI/threshold — they're honestly re-polled
    every 6h cycle rather than permanently skipped, since a listed
    partition can still fill in later (part 1's own docstring already
    notes this precedent for monthlySummary 2014-2016).
  - `bootFinraQueryPoll()` now runs `refreshFinraQuery()` then
    `refreshFinraAts()` in the same 6h cycle (no second timer).
  - New route `GET /api/data/ats-summary` (RAW, no predictive claim,
    cache-only request path, 1h Cache-Control) mirrors the existing
    `/api/data/short-interest` shape.
  - 3 new manifests (finraweekly/finramonthly/finrablocks) — picked up
    automatically by the streams-inventory ratchet (enumerates
    `datacore/manifests/` at runtime, R14/Phase-5 precedent) with zero
    code change required; confirmed via a live `streamsInventory.test.ts`
    run.
  - No UI page — same pipeline+API-first sequencing sec8kEarnings and
    finraQuery part 1 both used (a view follows once archive history
    accumulates).
- DOWNSTREAM CHAIN (REASONING STANDARD #1): new archived ATS partitions
  -> `/api/data/ats-summary` exposes RAW weekly/monthly per-symbol
  ATS/OTC volume leaderboards + ATS venue block-trading ranks ->
  discoverable via the streams inventory (#/data/streams) -> feeds the
  still-unstarted settlement-stress composite [RESEARCH] item's
  eventual venue-concentration angle (that composite's ingredients today
  are finrathreshold+secftd+finrashortvol only — this PR adds
  observability, not a new ingredient to that specific hypothesis) ->
  nothing here is claimed as predictive; every response carries
  `kind: "raw"` and an explicit composition/tiers_covered honesty note.
  Nothing in the trading loop (bot_engine.py, server/bot.ts) reads this
  route — zero live trading behavior change.
- GATES: `npx tsc --noEmit` — 64 errors, unchanged baseline (grep-
  confirmed zero hits in finraQuery.ts/finraQuery.test.ts/routes.ts).
  `npm run test:node` — 462/462 passed (445 baseline + 17 new:
  compositeKey, fetchPartitionTuples, fetchPartitionRowsMulti AND-not-OR
  + raised-cap pagination, all three new summarizers incl. the
  FIRM-row-exclusion assertions, end-to-end refreshFinraAts incl. the
  empty-tier-repoll behavior on a second run, transport-failure honesty).
  `npm run build` — clean; `dist/datacore/manifests/` confirmed to
  contain the 3 new files (R14 packaging lesson re-checked, not assumed).
  `python3 -m pytest` — not runnable in this sandbox (no pytest
  install, no network attempted); zero .py files touched, out of scope.
  VISUAL VERIFICATION not applicable — zero client/ files touched.
  Version 1.0.207 -> 1.0.208 (read-and-increment; confirmed against the
  live GitHub API `main` HEAD — c0c8732/1.0.207 — immediately before
  bumping, per the OPS GOTCHAS stale-cache lesson).
- PROMOTION RULES: (1) tests pass; (2) new behavior has new tests (17,
  above); (3) not a strategy/parameter change — RAW reference data, no
  backtest required; (4) version bumped for code_version attribution
  hygiene even though nothing here touches the trading path; (5) one
  logical change (FINRA part 2's three datasets, one module, one route,
  one PR) — the SHARED-file touches (routes.ts route registration,
  package.json version, this research/ log + wishlist.md update) are the
  small SHARED-file tail per WORKSTREAM PARTITION's merge-order protocol,
  not a second logical change; (6) N/A, no client/ files.
- HOUSEKEEPING NOTE (not actioned this PR, filed for the next session
  that touches R5): open_questions.md's Everything Graph section (MAP V2
  ROADMAP R5) still reads "NEXT (unclaimed): step (3) — the `/data`
  graph panel" — that step shipped (client/src/pages/graph.tsx, wired at
  #/data/graph) at some point after that note was written and the note
  was never updated to reflect it. Low-stakes (append-only research docs
  drift is expected) but worth a one-line correction next time R5 is
  touched, to stop a future session from re-discovering "already done"
  the hard way like this session did.
- NEXT: still open per wishlist.md's DATACORE MAXIMUS block — USGS
  quakes + NDBC buoys, SEC MIDAS census builds; EPA CAMD/ENTSO-E on
  Mike's keys (9a/9c); a FINRA part 2 UI view once weeks of archive
  accumulate; the settlement-stress composite [RESEARCH] item (separate,
  unblocked, ingredients already recording).

## 2026-07-07 — [PRODUCT] Grid-stress descriptive dashboard surface — the A1 gate-2 FAIL path product, #/data/grid-stress (v1.0.191)

- Territory: T-CLIENT (client/src/pages/gridstress.tsx, datamap.tsx launcher
  + hash routing, index.css) + T-DATACORE (server/gridStress.ts + test) +
  SHARED (server/routes.ts route, scripts/visual_check.mjs page/fixture,
  package.json version, this file). Executes the NEXT queue item filed by
  the 2026-07-07 gate-2 session (grid_vision_products.md "A1 gate-2
  RESULT": "(2) descriptive stress dashboard surface (honest FAIL-path
  product per the design)").
- WHY THIS SHIP IS HONEST, NOT A BACKDOOR SIGNAL: the v0 index (demand
  percentile x forecast strain x weather extremity) was gate-2 tested
  against two pre-stated outcome definitions and VOIDED on both (0/10 and
  insufficient spot-validation hits — datacore/gridvision/gate2_result.json,
  v1.0.190). The design's own FAIL path, locked BEFORE computation, says
  the index "demotes to a DESCRIPTIVE dashboard surface labeled
  non-predictive" — this PR ships exactly that, nothing more: three raw
  ingredients (same-month demand percentile, forecast strain, NOAA weather
  extremity) plus a plain EQUAL-WEIGHTED composite. Deliberately NOT the
  gate-2 script's fitted weights — those were fit against an outcome
  variable that turned out wrong, so reusing them would smuggle a voided
  claim back in under a new name. `predictive: false` ships on every
  response, never omitted; the client leads with a red honesty banner
  above the numbers, not a footnote (Amendment 5c: "premium presentation
  of wrong numbers is fraud with good typography").
- BUILD: `server/gridStress.ts` streams the ERCO-only griddemand archive
  (readline + gz, same bounded-per-file idiom as shadowFleet.ts's
  foldVesselArchiveAsync — the OOM lesson from that repair applies here:
  state retained is one {peak, strainSum, strainCount} record per archive
  DAY, never per-row) and joins the committed NOAA CPC TX degree-day
  archive (datacore/cpc/degree_days.json, no new collection — same
  datacore-boundary pattern as entityGraph.ts's registry joins). Percentile
  is same-calendar-month, full available history (not gate-2's train/valid
  split — that split is a backtest concern, already run, already voided;
  this is a live descriptive read). WITHHELD, NEVER GUESSED: percentiles
  return null with an honest sample-day count below MIN_SAMPLE_DAYS=5
  same-month peers, and the composite is null unless all three percentiles
  are present. Cache + 6h poll (server/routes.ts `bootGridStressPoll`) —
  a full archive fold is too heavy for the request path; 6h matches how
  slowly the inputs actually change (daily archive files, weekly CPC
  refresh).
- CLIENT: `client/src/pages/gridstress.tsx`, same hash-overlay pattern as
  streams.tsx (#/data/grid-stress), reuses the .vt-filings/.vt-streams
  shell. Launches from a panel-top button next to "Streams inventory"
  (TX-specific, not a spatial layer, so it does not join the map's
  on/off layer list). Three stat tiles with percentile bars + an
  equal-weighted composite row; loading/warming_up/insufficient-history/
  error states all designed, not just the happy path.
- TESTS: 9 new node tests (server/gridStress.test.ts) — ERCO-only
  filtering (other respondents must not leak into the peak), gz vs plain
  day-files read identically, legacy typeless lines treated as demand
  (mirrors gridDemand.ts's own v1/v2 archive-compat rule), pctRank math,
  CPC H+C summation with null-handling, thin-history withholds
  percentiles (never guesses), full end-to-end composite computation.
  node 381/381 (372 baseline + 9 new); tsc 64 (unchanged baseline, zero
  new errors — confirmed via grep against the full tsc log); `npm run
  build` clean. python3 -m pytest not runnable in this container (no
  pytest install available, no network attempted) — out of scope anyway,
  zero .py files touched by this PR.
- VISUAL VERIFICATION (promotion rule 6): `npm run visual` at 390/768/1440
  for the new `gridstress` page (added to scripts/visual_check.mjs PAGES +
  a deterministic /api/data/grid-stress fixture) plus a full-suite run
  (data/streams/gridstress/developers/landing + data-all-off + data-scale)
  — 0 hard failures. Self-reviewed screenshots at all three widths: no
  occlusion, no truncation, honesty banner readable first at 390px, tiles
  reflow to a single column on phone / two-plus-one on tablet / three-wide
  on desktop. One cosmetic bug caught and fixed in this same PR before
  shipping (not a follow-up): percentile ordinals rendered "72th"/"82th" —
  added a proper ordinal() suffix helper, re-screenshotted to confirm
  "72nd"/"82nd"/"61st"/"74th" render correctly.
- NEXT (unclaimed, from the gate-2 session's queue, still open after this
  PR): (1) V3 gate-2 design — subagent research of the complete dated
  public ERCOT conservation/EEA event list 2019-2025 from primary sources,
  fresh design filed before any computation (anti-fishing: this is a new
  design, not a third same-session variant); (2) Phase B training-data
  prep (ETDII download + OSM-seeded chips + NAIP/MPC streaming) toward the
  first RunPod fine-tune; (3) Phase B1 VERIFY spec. Also open, unrelated
  to grid vision: FINRA part 2 (weeklySummary/blocksSummary/
  monthlySummary), USGS quakes + NDBC buoys, SEC MIDAS, EPA CAMD/ENTSO-E
  on Mike's keys (9a/9c) — see wishlist.md DATACORE MAXIMUS resume block.
- VERIFY (pre-stated): post-deploy /api/data/grid-stress serves
  `predictive: false` + a non-null `reading` once the ERCO archive has
  >=5 same-month peer days on file (history_depth_days should already be
  well past that given the GRID_DEMAND_BACKFILL flag has been running on
  prod since 2026-07-07); the #/data panel shows the "Grid stress (TX) —
  descriptive only" launcher and it deep-links/back-buttons correctly.

## 2026-07-07 — [PRODUCT] Power grid map layer, TX pilot — OSM PMTiles vector layer on /data (DATACORE MAXIMUS Phase 2 item 1) (v1.0.166)

- Territory: T-CLIENT (datamap.tsx layer + pmtiles protocol,
  client/public/tiles/power_tx.pmtiles 16MB artifact) + SHARED
  (datacore/layers.json entry, package.json + lock for pmtiles@4.4.1,
  this file).
- BUILD: single 16MB PMTiles on our origin (built by the committed
  scripts/build_power_tiles.sh from Geofabrik Texas — the whole
  pipeline ran in <1 min in-session); pmtiles:// protocol registered
  at map bootstrap (lazy, idempotent); vector source fetched ONLY
  when the toggle is on (zero-cost-when-off). Six style layers with
  the filed zoom gates: HV ≥230kV always; 100-230kV z≥6; <100kV
  z≥11; substation/plant footprints z≥9; and the VOLTAGE-HONESTY
  class — lines with missing OR unparseable voltage (multi-value
  "138000;69000" strings) render dashed-lavender via a to-number
  fallback expression, never hidden. ODbL attribution on the source;
  registry description carries coverage=TEXAS PILOT + no-CEII +
  underground-absent honesty.
- SERVING DECISION (recorded): 16MB rides the repo/image via
  client/public (express static = range requests, image rule
  satisfied since dist/ ships); US-scale (est. 60-150MB) will NOT be
  committed — boot-fetch from a GitHub Release asset into the volume
  is the filed design for item 2.
- Gates: tsc 64; build OK (artifact confirmed in dist/public/tiles);
  visual harness GREEN at 390/768/1440 across all pages (layer
  defaults OFF — screenshots regression-check the panel; 390px data
  view self-reviewed clean); version 1.0.166. VERIFY (pre-stated):
  post-deploy /tiles/power_tx.pmtiles answers range requests (curl
  -r 0-100 → 206) and the /data layer panel shows "Power grid (TX
  pilot)" under facilities; toggling it on a TX viewport renders
  voltage-classed lines.
- NEXT (resume block updated): Phase 4 Streams inventory tab; census
  builds JODI → FINRA cluster → SEC FTD; grid item 2 US-full after
  the boot-fetch design; item 3 demand join.

## 2026-07-07 — [PIPELINE] DATACORE MAXIMUS checkpoint: Phase 0 + census #1 ALL LIVE-VERIFIED same session (docs)

- Pre-stated criteria from v1.0.163/164/165, checked post-deploy:
  /api/data/grid-demand → 9 respondent stats ✓ (EIA_API_KEY active);
  /api/data/crop-conditions → week 2026-07-05, 10 rows ✓
  (NASS_API_KEY active; the documented-shape build worked on the
  FIRST query — the censusImports verbatim-error fallback was never
  needed); /api/data/occ-volume → report 2026-07-02, 4,547
  underlyings, top SPY ~15.0M cleared with full customer/MM P/C
  splits ✓. The OCC rolling-2-year archive is RECORDING — the clock
  on permanent history started today.
- Program scorecard, day one: Phases 0+1 complete; 3 new streams
  built+verified (grid demand, crop conditions, OCC) + census filed
  (~30 sources) + grid build order filed with an in-container-proven
  pipeline. Resume state current in wishlist.md.

## 2026-07-06 — [PIPELINE] OCC daily options volume by trade origin — the census #1 archive-now root + /api/data/occ-volume (v1.0.165)

- Territory: T-DATACORE (server/occVolume.ts + test,
  datacore/manifests/occvolume.json) + SHARED (server/routes.ts
  wiring, open_questions.md GRID BUILD ORDER filed same commit,
  package.json, this file).
- WHY FIRST: the census ranked it #1 — OCC serves a ROLLING 2-YEAR
  window (workup-verified verbatim error body "Report date cannot be
  prior to  2 years"); every unarchived day is permanently lost.
  Keyless, ~700KB gz/day.
- BUILD FROM VERIFIED FACTS (a dedicated workup agent probed
  everything before coding — subagent mandate): 7-field CSV
  (quantity, underlying, symbol=option root, actype C/F/M, porc P/C,
  exchange×18, actdate MM/DD/YYYY, CRLF); ALL errors arrive HTTP 200
  + plain text → classifyBody() (data / no_records /
  not_yet_published / aged_out / unknown), never parsed as data;
  quantity counts EACH CLEARING SIDE → totals halved, C/M put-call
  kept side-consistent (the double-count gotcha is pinned by a test
  with the real AAL 34483/34483 fixture); per-symbol variant's
  trailing comma tolerated; dedup key verified unique across all
  153,181 rows of the probe day. 4h poll over a 5-trading-day window
  (retries not-yet-published, self-heals gaps); gz-on-write;
  _resetOccForTests for module-Set test isolation (alpaca_feed
  precedent — first draft leaked dedup state between tests, caught
  by the battery itself).
- FOLLOW-UPS FILED: one-time ~500-call 2-year backfill = session-run
  task (census); license note — raw resale needs OCC permission,
  gated signals fine.
- ALSO IN THIS COMMIT (docs riding per shared-file batching): GRID
  BUILD ORDER filed in open_questions.md — Phase 2 pipeline PROVEN
  VIABLE by the feasibility workup (tippecanoe 2.49 native pmtiles +
  osmium via apt; TX 709MB proof <10min; US 12GB ≲1hr/15GB peak;
  pmtiles@4.4.1 client dep; voltage-honesty rule: untagged lines
  flagged, never dropped; ODbL produced-works boundary; Overpass =
  refresh only). Items 1-6 with signal hypotheses + blocked
  dependencies (9a/9b/9c).
- PRIOR (restated from census): customer-vs-MM put/call BY TICKER is
  the retired ISEE's successor, uncrowded on small caps. Gate 1 =
  one day's totals vs OCC's published volume page; gate 2 = customer
  P/C extremes vs forward returns, census-breadth discounted.
- Gates: node 326/326 (5 new); tsc 64; build OK; version 1.0.165.
  VERIFY (pre-stated): post-deploy /api/data/occ-volume serves
  report_date 2026-07-02 (or newer once OCC publishes Monday
  overnight) with top-50 underlyings and nonzero customer/MM splits;
  archive gains occvolume/2026-07-02.jsonl.gz (~1MB).

## 2026-07-06 — [RESEARCH] DATACORE MAXIMUS Phase 1: master data census FILED — ~30 sources probed live by 3 parallel agents, 11 buildable ranked, dead-list recorded (docs)

- Territory: SHARED research/* (data_census.md NEW permanent doc,
  wishlist.md program-state block + BLOCKED-FOR-MIKE 9a-9d, this
  file). Subagent mandate applied: parallel research, serial merges,
  durable filed output — every agent finding landed in the census
  doc, nothing lost in-session.
- HEADLINES: (1) OCC daily options volume — keyless, per-ticker
  customer/firm/market-maker put-call, HARD 2-YEAR PURGE window →
  archive-now urgency, next stream build; (2) EPA CAMD CEMS —
  unit-level HOURLY plant utilization to 1995, free key → the
  ground-truth source for the whole power vertical (Mike: 9a);
  (3) Esri World Imagery metadata identify endpoint VERIFIED with
  exact params (Phase 3a unblocked); (4) CDSE free-tier quota math
  done — scheduled facility chips only, never viewport renders;
  (5) ODbL precise read: signals sellable with attribution,
  geometry-database redistribution triggers share-alike; (6) DEAD:
  CME (Akamai-blocked), ISE (retired), NYSE threshold API
  (bot-blocked — FINRA covers), PBOC (scrape-hostile), each with its
  cheapest unlock path stated per NO-ARTIFICIAL-WALLS.
- FAILURE-MODE REGISTER filed (census tail): Overpass single-instance
  dependency real (mirrors unreachable from our proxy), Geofabrik→
  osmium→PMTiles as the grid pattern (~150-400MB global HV
  estimated), mobile 390px jank sources named, DTCC 147MB/day flagged
  for a volume-budget decision before build.
- ANTI-FISHING NOTE: the census files ~30 hypotheses; each will be
  discounted at gate 2 for the breadth of this sweep; nothing is
  believed before its ladder gates.

## 2026-07-06 — [PIPELINE] DATACORE MAXIMUS Phase 0b: NASS crop conditions stream — key-gated archiver + /api/data/crop-conditions (v1.0.164)

- Territory: T-DATACORE (server/cropConditions.ts + test,
  datacore/manifests/cropconditions.json) + SHARED (server/routes.ts
  wiring last + minimal, package.json, this file).
- BUILD: NASS_API_KEY landed in Railway → BUILD ORDER 6 #7 builds
  key-gated. The key is Railway-only (session presence check: absent)
  so the row shape follows the DOCUMENTED QuickStats contract with the
  censusImports precedent applied in full: defensive parse (Value and
  value keys both tolerated, comma-grouped numbers stripped), NASS
  error bodies logged VERBATIM so the next session fixes queries from
  prod logs, LIVE VERIFICATION PENDING post-deploy. Scope v1: NATIONAL
  weekly CONDITION for CORN + SOYBEANS (5 classes as separate rows,
  short_desc verbatim); commodity|week|item dedup; 12h poll.
- PRIOR (restated): medium-low standalone (widely watched), medium in
  the JOIN with the live Drought Monitor stream (belt-weighted
  ag-stress index). Gate 1 = values vs the published Crop Progress
  report; gate 2 = condition-delta vs forward grain futures returns.
- Gates: node 321/321 (5 new); tsc 64; build OK; version 1.0.164.
  VERIFY (pre-stated): post-deploy /api/data/crop-conditions either
  serves the current week's ~10 rows (in-season Monday data exists) OR
  logs a verbatim NASS error naming the query fix — either outcome is
  progress; enabled:false would mean the key is NOT actually in
  Railway (escalate to wishlist).

## 2026-07-06 — [PIPELINE] DATACORE MAXIMUS Phase 0a: EIA-930 grid demand stream — key-gated archiver + /api/data/grid-demand (v1.0.163)

- PROGRAM START: the human installed the DATACORE MAXIMUS standing
  program this session (multi-session; phases 0-5: key activations →
  exhaustive source census → global power-grid layer → imagery
  freshness at all levels → UI expansion → ratchets). Program state
  and resume protocol live in wishlist.md; this entry is Phase 0a.
- Territory: T-DATACORE (server/gridDemand.ts + test,
  datacore/manifests/griddemand.json) + SHARED (server/routes.ts
  wiring last + minimal, package.json, this file).
- BUILD: EIA_API_KEY landed in Railway (Mike's part) → the BUILD
  ORDER 6 #6 stream builds NOW, key-gated on the
  fredMacro/censusImports pattern (enabled() env check, honest
  enabled:false route state, activates wherever the key lands).
  Shape verified live with DEMO_KEY before coding: EIA v2 envelope,
  value as STRING MWh, bracketed params must be URL-encoded (pinned
  by a test alongside a key-never-logged encoding pin). Scope v1:
  demand (type D) for US48 + 8 major BAs, 48h trailing window, 2h
  poll, observation-day day-files (hours land where they belong),
  respondent|period dedup across restarts, gz 3d.
- PRIOR (restated from the build order): medium. Gate 1 = US48 daily
  sum vs EIA's Grid Monitor dashboard; gate 2 = degree-day-adjusted
  demand residual vs industrial-sector returns. Joins: CPC
  degree-days (live), power plants, the Phase-2 grid layer.
- Gates: node 316/316 (5 new); tsc 64; build OK; version 1.0.163.
  VERIFY (pre-stated): post-deploy /api/data/grid-demand serves 9
  respondent stats with same-day latest_period (key is in Railway);
  archive gains griddemand/<obs-date>.jsonl.

## 2026-07-06 — [NO-ACTION] Day close-out: options first-snapshot QA (landed ✓), R13 verdict (FALSE ALARM), BUILD ORDER 6 closed 4 built + 1 pre-existing + 2 key-gated (docs)

- OPTIONS-CHAIN FIRST-SNAPSHOT QA (open_questions item 5, first
  criteria checkable today): optionchains/2026-07-06.jsonl EXISTS on
  the volume — forward history has begun, exactly one trading day
  after the archiver shipped (v1.0.83). Size 1.68MB plain (inside the
  stated 3-5MB/day raw budget); .last_run_day once-per-day marker
  present (restart-double-fire protection engaged). HONEST SCOPE:
  shape/contract-count/IV-sanity checks need a read surface the
  archiver deliberately doesn't expose publicly — DEFERRED to the
  first-week report (~2026-07-10+) via a session-run volume read or a
  small token-gated diag sample. Today's QA verdict: schedule ✓,
  write ✓, size envelope ✓, dedup marker ✓.
- R13 VERDICT — FALSE ALARM (SUPERSESSION NOTE: PR #290, another
  session, reached and merged the same verdict first — this paragraph
  stands as independent confirmation via the same probe): the new
  ?type=SHUTDOWN&limit=200 probe (v1.0.161) returned the full series:
  20 SIGTERMs since last night, EVERY ONE mapping to a merge/deploy
  (6-18 min spacing exactly during the 15-PR burst 16:11→18:11Z,
  hours apart overnight, and the "no deploy in flight" 18:46 restart
  matched #288's merge — the premise was wrong). The intermittent
  health 000s are Railway's container-replace window: the COST OF
  MERGE VELOCITY, not a defect. ~12 deploys in 2 hours ≈ a lot of
  wall-clock inside replace windows. Observation for Mike (wishlist):
  a zero-downtime/overlap healthcheck deploy config would remove the
  blips — railway.json/Dockerfile are FROZEN, so that is his call,
  zero urgency. The probe params remain net value (this diagnosis
  took one query).
- BUILD ORDER 6 CLOSED: #1 TFF (v1.0.156, live: 87 markets, and the
  archive already holds TWO report weeks — the poller caught
  2026-06-30 within hours), #2 DTS (v1.0.157, live: 180 lines, TWO
  statements archived 07-02+07-03), #3 FDIC failures (v1.0.158,
  live: 50 events, newest 2026-05-01), #4 NHTSA (v1.0.159, live: 17
  vehicles with stats — 3 watchlist entries have zero complaints for
  their model year, honest absence), #5 drought = PRE-EXISTING
  (v1.0.118, duplicate filing recorded, grep-own-repo lesson), #6/#7
  key-gated (BLOCKED-FOR-MIKE #8: EIA_API_KEY + NASS_API_KEY, two
  instant free signups). Four new perpetual archives in one day, all
  probed-before-coded, all gate-locked.

## 2026-07-06 — [REPAIR] R13 CLOSED: recurring SIGTERM restarts are deploy-driven, not a liveness bug — STARTUP audit entry compiles the correlation into queryable state (v1.0.162)

- Territory: T-BOT (server/bot.ts).
- SESSION START CHECKS: loop-health ratio over the last 10 tagged entries
  is 4 REPAIR / 10 (R13, R12 close-out, R12-D4, R12-D2) — below the 7+
  thrash trigger, no meta-problem to stop for. /api/health at session
  start: status ok, uptime_s 2858, bot active, liveness.dark false,
  drawdownPct 0.0, scanner consecutiveFailures 0 — Tier2 scans and
  position stop-monitoring both firing normally in the live audit tail
  (TIER2 scans every ~2-15min per market-time cadence, POS-MONITOR-SYNC
  flagging VXUS/SMH/KWEB stops). No open KNOWN BROKEN item was actionable
  without human-gated diagnostics beyond what's below.
- PRIMARY ACTION — closed the R13 investigation opened earlier today
  (v1.0.161, priority-1 liveness concern per GOAL: "loop going dark must
  be surfaced loudly, never discovered by the human on a dashboard").
  Used the v1.0.161 audit-probe upgrade to pull `?type=SHUTDOWN&limit=500`
  (60 entries, 20 distinct SIGTERM events, 2026-07-05T23:19Z through
  2026-07-06T18:46Z) and cross-referenced every timestamp against
  `git log origin/main` merge times (UTC-converted from -04:00). RESULT:
  20/20 sampled SIGTERM restarts land 64-158 seconds after a merge to
  main (Docker build + Railway rollout lag) — including two docs-only
  PRs (#275, #282, #283) which still triggered a redeploy+restart since
  Railway rebuilds on every push to main regardless of diff content. Zero
  restarts were unexplained by a deploy. Two merges 9 minutes apart
  (#270 22:49 EDT, #271 22:59 EDT) produced only ONE restart — consistent
  with Railway coalescing a rapid second push into the in-flight build
  rather than a missed/failed restart. CONCLUSION: today's high restart
  frequency is a direct, mechanical consequence of an unusually high
  autonomous merge cadence (12+ PRs in one day per the BUILD ORDER 6 +
  R12 series above), not container instability, OOM, or a healthcheck
  failure. R13's own REVISED READ ("possibly the SAME recurring cycle
  with deploys coincidental") was correct; this closes it with certainty
  instead of leaving it as an open suspicion for the next session to
  re-litigate.
- COMPILED (EDGE DOCTRINE #3 — never reason the same thing twice): the
  cross-reference above required manually pulling git log and computing
  offsets by hand — exactly the kind of one-off reasoning that should
  become permanent state. `registerBotRoutes()` (server/bot.ts, called
  once per process boot from server/routes.ts:509) now emits
  `audit("STARTUP", "Server boot — code_version <pkgVersion>, pid <pid>")`
  before any route/timer setup, reading `version` from package.json (same
  import pattern already used in server/routes.ts:17 for
  `/api/data/layers`). Future audit queries (`?type=STARTUP`) show
  exactly which version came up after each SIGTERM without needing git
  log at all — the correlation is now a persisted fact, not a
  rederivation. No safety-relevant behavior changed: this is a single
  audit() call, same function/pattern as the existing SHUTDOWN/START/STOP
  entries, no new dependency, no route or scheduling change.
- Downstream chain (REASONING STANDARD #1): STARTUP audit entry exists ->
  next SIGTERM investigation can join SHUTDOWN and STARTUP timestamps
  directly in one probe query -> no need to cross-reference external git
  history under time pressure during a real incident -> faster, more
  reliable triage the next time uptime looks suspicious, which is exactly
  when reasoning quality matters most (mid-incident, human possibly
  asleep).
- Gates: `npm install` (deps weren't present in this session's container),
  `npx tsc` — 64 errors, unchanged from the v1.0.161-stated baseline, none
  on the touched lines; `npm run test:node` — 311/311 pass, 0 fail;
  `npm run build` — client + server bundle both succeed. Python side
  untouched (no pytest available in this container to re-run; server-only
  change, zero Python surface). Version 1.0.162 (read-and-increment after
  #289 took 1.0.161).
- VERIFY (pre-stated): post-deploy, `?type=STARTUP&limit=5` should show
  one entry with `code_version 1.0.162` within ~1-3 minutes of the merge
  landing — confirming the instrumentation fires and the version threads
  through correctly. No behavior change expected in trading/scoring; this
  is observability-only.
## 2026-07-06 — [REPAIR] R13 opened: recurring SIGTERM restarts, health intermittently unresponsive — audit probe gains type/limit params to read the pattern (v1.0.161)

- Territory: T-BOT (server/bot.ts diag audit probe) + SHARED
  (package.json, this file).
- OBSERVATION (liveness priority): at 18:48Z prod uptime was 204s —
  restart at ~18:46 with NO deploy in flight (last merge #287 ~18:05).
  Audit tail shows SIGTERM → graceful shutdown → reboot, i.e. an
  EXTERNAL kill (Railway healthcheck), not a crash. Health curls
  intermittently return 000/timeout right before these windows.
  REVISED READ of today's earlier "deploy churn" unresponsiveness
  (~17:15, ~18:20): possibly the SAME recurring cycle with deploys
  coincidental — the fixed 50-entry audit tail (~5 min of a chatty
  log) could never show the period. Honesty note: today's new stream
  polls (TFF/DTS/FDIC/NHTSA), the 2-min Tier2 cadence, and
  entityGraph are all candidates — MY OWN code is on the suspect
  list.
- INSTRUMENT (this PR): /api/diag/audit accepts ?type= (exact match,
  SQL-parameterized) and ?limit= (cap 500) — the restart pattern
  becomes readable via ?type=SHUTDOWN&limit=200. Same whitelist,
  same sanitizeDiag, limit capped so the probe can't become a heap
  problem itself.
- ALSO THIS SESSION-WINDOW: /api/data/bank-failures VERIFIED live
  exactly as pre-stated (Community Bank and Trust - West Georgia,
  2026-05-01, 50 events); vehicle-complaints was warming_up on first
  check (sweep in progress), re-verify pending; DROUGHT: BUILD ORDER
  6 #5 discovered ALREADY BUILT (v1.0.118, PR #229, BUILD ORDER 2 #5
  — same FIPS finding, CONUS+8 belt states + DSCI). Duplicate filing;
  first-merged wins. LESSON (compiled): a build-order filing MUST
  grep server/ for existing stream modules before proposing items —
  the probe agents checked the SOURCES but nobody checked our own
  repo.
- Gates: tsc 64; build OK; scannerHealth tests 9/9; version 1.0.161 (read-and-increment after #288 took 1.0.160).
  VERIFY (pre-stated): post-deploy ?type=SHUTDOWN&limit=200 returns
  the day's restart timestamps; the inter-restart period tells us
  whether this is periodic (healthcheck cycle) or event-correlated
  (specific poll windows). Diagnosis continues in R13.
## 2026-07-06 — [PRODUCT] Everything Graph — /data panel, R5 build step 3 (v1.0.160)

- Territory: T-CLIENT (client/src/pages/graph.tsx, client/src/pages/datamap.tsx,
  client/src/index.css, scripts/visual_check.mjs) + SHARED (datacore/layers.json,
  datacore/EVERYTHING_GRAPH.md, package.json, this file — last + minimal).
- CONTEXT: R5 (MAP V2 ROADMAP) shipped step 1 (entity_map.json, 2026-07-05)
  and step 2 (server/entityGraph.ts + /api/data/graph, 2026-07-06 earlier
  today) with step 3 explicitly "NEXT (unclaimed)". This is a PRODUCT
  session per the standing directive: build UI/UX surfacing already-shipped
  backend capability, no new data collection.
- BUILD: `client/src/pages/graph.tsx` (GraphView) — same full-view-overlay
  pattern as filings.tsx/earnings.tsx/shortvol.tsx (hash-routed
  `#/data/graph`, opened via a launcher button inside a new "Everything
  Graph" registry layer + panel group). Shows the counts-only summary
  (entities/connections by type) on load, then an entity search (ticker/
  MMSI/CIK/facility id) against `/api/data/graph?entity=&hops=`. Every
  connection row displays the edge's source/confidence/roles/counts and
  is itself clickable — clicking a neighbor re-queries `/api/data/graph`
  with that neighbor's exact node id (`resolveEntityId` already resolves
  raw node ids directly), making this a real graph browser rather than a
  single lookup. SCOPE DECISION: the connections list always shows only
  DIRECT (1-hop) edges touching the searched entity regardless of the
  hops selector, because for hops>1 an edge in the BFS result may not
  touch the center at all — showing it as if direct would misrepresent
  the join. The hops selector instead scales the *reachable-network*
  count line ("N entities, M connections reachable within H hops"),
  which stays honest at any hop depth. Company->facility MAP
  highlighting (design doc's stretch item) is explicitly NOT in this PR —
  filed as the follow-up in datacore/EVERYTHING_GRAPH.md's build plan.
- BUG FOUND + FIXED IN THIS PR'S OWN SCOPE (not a separate PR — a
  one-line-per-site CSS token swap inside the same new component, not a
  behavior/logic change): `client/src/index.css`'s `:root` block declares
  `--accent: #4d9fff` at line 25 and then REDECLARES `--accent: 212 100%
  65%` (a bare HSL triple, no `hsl()` wrapper — the shadcn/Tailwind token
  block) at line 92, later in the same block — the second declaration
  wins the cascade, so `--accent` resolves sitewide to an invalid raw
  triple wherever used directly as a `color`/`background`/`border-color`
  value (18 such existing call sites in index.css, all pre-existing and
  NOT touched by this PR). CSS spec: a var() substitution invalid at
  computed-value time makes the property compute to its inherited/
  initial value instead — for non-inherited properties like
  background/border-color that's usually `transparent`/`currentColor`,
  which is why the badge I first wrote silently rendered dark-text-on-
  transparent (caught by manually screenshotting the actual GraphView
  full-page render with Playwright, since the standard visual harness
  only screenshots the /data map shell, not opened sub-views). FIX
  APPLIED HERE, IN-SCOPE ONLY: my 3 new `var(--accent)` uses
  (`.vt-graph-example`, `.vt-graph-typebadge`, `.vt-graph-conn-row:hover`)
  now use `--accent-bright` (a uniquely-named, non-colliding token) —
  verified via computed-style dump (background now `rgb(124, 196, 255)`,
  not transparent). The SITEWIDE collision (all 18 pre-existing sites,
  plus a same-shape `--border` collision that happens to have zero
  direct `var(--border)` callers today so it's dormant) is NOT fixed
  here — that is a measurement-adjacent, cross-many-components change
  that needs its own visual-regression pass per CLAUDE.md's one-logical-
  change rule, filed as KNOWN BROKEN #13 in open_questions.md for a
  dedicated future session.
- VERIFICATION: `npm run visual --page data` at 390/768/1440 — 0 hard
  failures; the new "Everything Graph" panel row is reachable (self-see),
  shows live status ("42 entities" / "61 connections" against the build
  fixture), and its launcher opens correctly (screenshots reviewed:
  .visual/data-legend-1440.png shows the panel row; full-view rendering
  additionally verified via an ad-hoc Playwright script against a richer
  neighborhood fixture, since the standard harness fixture is counts-only
  by design). node tests: 311/311 pass (no server-side code touched, so
  this is a determinism check, not new coverage — entityGraph.ts's own
  10 tests already cover the API this panel calls). tsc: 64 errors,
  unchanged baseline (confirmed via git-stash A/B; the one datamap.tsx
  line the diff touches was already failing pre-existing for the same
  reason, `graph` merely joins the existing literal union). Build OK.
  Version 1.0.160 (read-and-increment; 1.0.159 landed from a concurrent
  PR mid-session — rebased via `git merge --ff-only` before bumping, per
  OPS GOTCHAS).
- DEPLOY COUPLING: authored during market hours (Mon 2026-07-06, ~14:00
  ET) — PR left for merge after the 16:00 ET close per this session's
  brief, noted in the PR description.

## 2026-07-06 — [PIPELINE] NHTSA complaints watchlist stream (BUILD ORDER 6 #4) — curated-seed archiver + /api/data/vehicle-complaints (v1.0.159)

- Territory: T-DATACORE (server/nhtsaComplaints.ts + test,
  datacore/nhtsa_vehicles.json seed, manifests/nhtsacomplaints.json) +
  SHARED (server/routes.ts wiring last + minimal, package.json, this
  file).
- DESIGN DECISION (recorded): the daily bulk FLAT_CMPL.zip is 84+MB —
  unbootable under the memory/event-loop rules — so the stream polls a
  CURATED ticker-mapped make/model/year watchlist (20 vehicles across
  TSLA/F/GM/RIVN/STLA/TM/HMC/HYMTF/LCID/NSANY/VWAGY, recent model
  years only — complaint VELOCITY needs current product) via the
  keyless per-vehicle API. This is the wikiAttention curated-seed
  pattern applied to NHTSA; the bulk file is the documented
  deep-history follow-up. Free-text summaries deliberately NOT
  archived (counts/flags/dates/components carry the signal; a
  test pins the exclusion).
- Shape probed live before coding: {count, results:[{odiNumber,
  crash, fire, numberOfInjuries, dateComplaintFiled MM/DD/YYYY,
  components,...}]} — dates ISO-normalized; ODI number is the
  event-identity dedup key (across fetches AND restarts). 12h sweep,
  ~20 politeness-spaced calls/cycle, day-file per fetch date, gz 2d,
  eager boot, cache-only route serving per-vehicle stats.
- PRIOR (from the build order, restated): medium-low for megacaps
  (crowded), medium for the supplier-mapping angle via the components
  field (uncrowded — Everything Graph substrate). Gate 1 = complaint
  counts vs NHTSA's published recall timeline for 3 known cases;
  gate 2 = velocity anomalies vs forward returns.
- Gates: node 311/311 (5 new tests incl. watchlist-bounded pin and
  summaries-excluded pin); pytest untouched; tsc 64; build OK;
  version 1.0.159. VERIFY (pre-stated): post-deploy
  /api/data/vehicle-complaints serves 20 vehicle stat rows with
  newest_filed dates in 2026; archive gains
  nhtsacomplaints/<fetch-date>.jsonl with several hundred events on
  first sweep (the API returns each vehicle's recent window).

## 2026-07-06 — [PIPELINE] FDIC bank failures event stream (BUILD ORDER 6 #3 v1) — keyless archiver + /api/data/bank-failures (v1.0.158)

- Territory: T-DATACORE (server/fdicBanks.ts + test,
  datacore/manifests/fdicfailures.json) + SHARED (server/routes.ts
  wiring last + minimal, package.json, this file).
- LIVE VERIFICATIONS of the prior two items (pre-stated criteria,
  checked post-deploy this session): /api/data/tff serves report week
  2026-06-23 with 87 markets, dealer/asset-manager/leveraged-money
  populated ✓; /api/data/dts serves statement 2026-07-02 with 180
  category lines incl. the withheld-tax categories ✓. (Both endpoints
  timed out for ~1 min during the deploy window — same transient
  churn signature as earlier today; self-recovered.)
- BUILD (v1 scope decision recorded): FAILURES ONLY — the event-driven
  live kicker. The quarterly per-bank financials snapshot (deposit-
  flight tracking, ~4.6k banks paginated + once-per-quarter claim) is
  the documented follow-up with its own PR (one logical change).
  Probe findings encoded: HOST IS api.fdic.gov (the documented
  banks.data.fdic.gov 301s — pinned by a test); FAILDATE arrives
  M/D/YYYY (normalized); amounts in $ THOUSANDS; COST (DIF loss) null
  until estimated — null never becomes zero. Event-identity dedup
  (cert|fail_date) within and across restarts (FAA/CBP pattern),
  day-file per fetch date, gz after 2d, eager boot, cache-only route.
- PRIOR (from the build order, restated): medium — quarterly financials
  limit timing; the failures feed is event-driven and small regional
  banks are EDGE DOCTRINE #2 territory. Gate 1 = cross-check a
  failure's assets/deposits vs the FDIC press release; gate 2 =
  failure events vs forward KRE/regional-bank returns.
- Gates: node 306/306 (5 new tests incl. the host pin); pytest
  untouched; tsc 64; build OK; version 1.0.158. VERIFY (pre-stated):
  post-deploy /api/data/bank-failures serves the 50-failure window
  with COMMUNITY BANK AND TRUST - WEST GEORGIA (2026-05-01) newest;
  archive gains fdicfailures/<fetch-date>.jsonl with 50 events on
  first poll.

## 2026-07-06 — [PIPELINE] Treasury Daily Statement stream (BUILD ORDER 6 #2) — keyless FiscalData archiver + /api/data/dts (v1.0.157)

- Territory: T-DATACORE (server/treasuryDts.ts + test,
  datacore/manifests/treasurydts.json) + SHARED (server/routes.ts
  wiring last + minimal, package.json, this file).
- BUILD: Daily Treasury Statement Table II (deposits & withdrawals of
  TGA operating cash) via the keyless FiscalData API. Field names
  probed live 2026-07-06 BEFORE coding: string amounts in $ MILLIONS,
  transaction_catg verbatim (withheld income/employment taxes, FUTA,
  corporate taxes...), and the operational gotcha the probe caught —
  page[size] brackets MUST be URL-encoded or the API 400s (pinned by
  a test). Day-boundary discipline (DESC fetch keeps only the newest
  statement), day-level dedup, gz after 4d, restart cache rebuild
  from archive, eager boot, cache-only request path.
- ROUTE: /api/data/dts (warming_up honesty; RAW label; $-millions and
  lag stated in the note).
- PRIOR (from the build order, restated): medium-high that the
  withheld-tax nowcast SIGNAL exists (published research), medium on
  OUR edge (must beat FRED-lagged equivalents; the daily cadence +
  backfillable 2005+ history is the moat — backfill filed as a
  separate session-run task, not the boot path). Gate 1 = reconcile
  monthly sums vs MTS/FRED federal receipts; gate 2 = withheld-tax
  YoY growth vs payroll-surprise dates. Nothing believed before
  gates.
- Gates: node 301/301 (5 new DTS tests incl. the bracket-encoding
  pin); pytest untouched; tsc 64; build OK; version 1.0.157.
  VERIFY (pre-stated): post-deploy /api/data/dts serves the
  2026-07-02 (or newer) statement with ~90 category lines; archive
  gains treasurydts/<date>.jsonl.

## 2026-07-06 — [PIPELINE] CFTC TFF financial-futures positioning stream (BUILD ORDER 6 #1) — keyless Socrata archiver + /api/data/tff (v1.0.156)

- Territory: T-DATACORE (server/cftcTff.ts + test, datacore/manifests/
  cftctff.json) + SHARED (server/routes.ts wiring last + minimal,
  package.json, this file).
- BUILD: faithful clone of the live cftcCot.ts adapter (the design was
  the point — one verified pattern, second dataset). Dataset
  gpe5-46if (TFF futures-only, ~44 markets/wk, history to 2006).
  Field names verified live 2026-07-06 BEFORE coding (the COT quirk
  lesson): dealer_* keeps the `_all` suffix; asset_mgr_* and
  lev_money_* drop it — FIELD constant encodes them exactly. Week-
  boundary discipline (DESC fetch straddling two weeks keeps only the
  newest), week-level dedup, gz after 9d, restart cache rebuild from
  archive, eager boot poll, cache-only request path (event-loop rule).
- ROUTE: /api/data/tff (warming_up honesty; RAW label; futures-ONLY
  note — the combined variant yw9f-hn96 is a different dataset).
- PRIOR (from the build order, restated): modest — TFF is well-studied
  public data; the edge if any is in JOINS (cross-asset positioning:
  TFF financials × COT commodities × our regime classifier). Gate 1 =
  one week's values vs CFTC's published HTML report; gate 2 =
  leveraged-money extreme percentiles vs forward SPY/sector returns,
  regime-split. Nothing believed or traded before gates.
- Gates: node 296/296 (5 new TFF tests incl. dataset-id pin so the
  disaggregated dataset can never silently swap in + manifest
  enforcement green); pytest 449/1 untouched; tsc 64; build OK;
  version 1.0.156. VERIFY (pre-stated): post-deploy /api/data/tff
  serves the 2026-06-23 (or newer) report week with ~44 markets and
  dealer/asset-manager/leveraged-money fields populated; archive
  gains cftctff/<date>.jsonl.

## 2026-07-06 — [RESEARCH] BUILD ORDER 6 FILED — positioning + nowcasts + stress; 12 sources probed live by parallel agents, 7 selected, 5 declined with reasons (docs)

- Territory: SHARED research/* only (docs PR). Standing-directive
  fall-through: BUILD ORDER 5 closed 5/5 → generate the next order.
- METHOD (BUILD ORDER 5 precedent, scaled): two parallel research
  subagents probed 12 candidate sources LIVE (working URLs, auth,
  response shapes, cadence, history depth, license text — no
  fabrication, HTTP errors reported as-is); judgment and selection
  stayed in the parent session per the WORKSTREAM PARTITION
  subagent rule.
- SELECTED (numbered order + hypothesis + prior + ladder path each,
  in open_questions.md): 1 CFTC TFF financial-futures positioning
  (keyless Socrata clone of the live COT adapter — cheapest build),
  2 Treasury DTS daily statement (keyless; withheld-tax payroll
  nowcast, history to 2005), 3 FDIC bank data (keyless; note the
  HOST MOVED to api.fdic.gov — old host 301s; regional-bank stress,
  EDGE DOCTRINE #2 territory), 4 NHTSA complaints velocity (keyless
  daily bulk; supplier-mapping angle via Everything Graph), 5 US
  Drought Monitor (keyless county-level weekly; attribution
  requirement travels on every record), 6 EIA-930 hourly demand +
  7 USDA NASS crop conditions (both FREE-KEY-gated; keys absent
  from session AND Railway env by presence check — filed
  BLOCKED-FOR-MIKE #8; build key-gated adapters now, fredMacro
  pattern).
- DECLINED with recorded reasons: AAR rail (duplicative of EP724 +
  paid-product license), BTS on-time (2-month lag), BLS API (FRED
  covers same-day), Baker Hughes rigs (no redistribution grant —
  EIA Drilling Productivity Report filed as the public-domain
  alternative to probe next order), OSHA ITA (annual cadence —
  parked as an Everything-Graph facility join, not a stream).
- ANTI-FISHING NOTE (REASONING STANDARD #4): 7 new hypotheses filed
  at once — each gate-2 result will be discounted for the breadth of
  this order; none is believed before its ladder gates.

## 2026-07-06 — [REPAIR] R12 CLOSE-OUT — feedback pipeline repair series verified post-deploy; loop recording again after a 2.5-month blackout (docs)

- SERIES SUMMARY (all merged + deployed same day): v1.0.152 shape
  aggregates named the writer in one deploy → v1.0.153 fixed the entry
  side (qty guard) → v1.0.154 wired the exit side (first-ever
  exit_context callers) → v1.0.155 fixed boot-cleanup correctness
  (seeds/numeric-compare/fossil purge).
- POST-DEPLOY VERIFICATION (v1.0.155, criteria pre-stated in that
  entry): /api/diag/ml → feedback_count 0, live_key_signatures {} —
  the 15-key 1.0.33 fossil signature is GONE (purge verified ✓).
  feedback_seeded_count still 0 (✗ half of the criterion): the
  daemon's autoseed check ran before the purge landed on that boot;
  expected to fire on the NEXT deploy-boot (seeder script +
  backtest_10yr_results.json verified present/tracked). Logged as
  KNOWN BROKEN #12(a) with the escalation rule: still 0 after another
  deploy = the autoseed path itself is a new defect.
- Transient full-site unresponsiveness (~3 min, health timeouts)
  during the #281 deploy window self-recovered (uptime reset observed;
  consistent with prior Railway deploy churn). Watched under the
  liveness rule; no action needed.
- D3 (dead trackClosedTrades feedback block) + remaining exit paths
  filed as KNOWN BROKEN #12(b)/(c), decision gated on D2's first live
  exit verification — VXUS (time_stop) and SMH (trailing_stop) are
  already flagged by bot_engine, so the gate should resolve within
  days.
- SESSION-OPS LESSON (compiled): a merge monitor MUST verify by the
  squash suffix "(#NNN)" — grepping a content string false-positived
  on v1.0.148's commit message, triggered a premature branch reset,
  and auto-closed PR #277 (recovered via reflog + reopen; automerge
  then merged it). The merge-order protocol rule 4 ("verify WHICH PR
  merged before any branch reset") now has a concrete mechanism
  attached: PR-number grep, nothing else.

## 2026-07-06 — [REPAIR] R12-D4: boot cleanup correctness — seeds spared, numeric version floor, April fossils purged (v1.0.155)

- Territory: T-BOT (feedback_boot_cleanup.py + test, server/bot.ts
  snippet, test_fixes_pr11.py re-pin) + SHARED (package.json, this
  file).
- ONE LOGICAL CONCERN — the boot-time trade_feedback cleanup filter —
  fixing its three defects together (they are one filter expression):
  1. SEED WIPE-OUT (D4 proper): seeds carry no code_version; the old
     string compare deleted every Kelly-prior seed on every deploy.
     Seeds (_seed: True) are now kept unconditionally.
  2. LEXICOGRAPHIC TRAP (latent future wipe, found during D4 design):
     '1.0.153' < '1.0.34' as STRINGS — any record stamped a real
     3-digit patch version would have been silently wiped at next
     boot. Compare is now numeric tuples.
  3. FOSSIL PURGE: floor raised to (1,0,34) — the 500 April records
     (pre-Bug-25b writer, code_version 1.0.33, pnl 0, outcome-less)
     are exactly the broken-code artifacts the cleanup exists to
     remove, and they were doing active harm: blocking the daemon's
     <100-record reseed AND matchable by _find_entry_record on ticker
     collision (mislabel risk for D2's exit recording).
- MEASUREMENT INTEGRITY (stated per the rule, this change touches the
  training-data pool): removing the fossils cannot make live
  performance LOOK better — they were already excluded from every
  performance metric (pnl 0 + outcome None is filtered by
  check_model_health and the dashboard). Direction of effect on
  TRAINING: removes 500 records that _build_feedback_training_data
  would default to label=0 (loss) noise; restores ~1326 seeded priors
  the Kelly gate was designed to start from. Named bugs, not tuning.
- MECHANISM: filter extracted from the inline bot.ts snippet into
  feedback_boot_cleanup.py (keep_record/clean_feedback/parse_version,
  strings-only version parsing so corrupt numeric values fail the
  floor) — compile-knowledge-into-code, unit-testable. bot.ts snippet
  now imports it. test_fixes_pr11.py's two source-string pins on the
  OLD inline filter re-pinned BEHAVIORALLY against the module (same
  two protections — ticker filter, version floor — asserted stricter,
  plus a new pin that bot.ts routes through the module; R1 stale-pin
  precedent, no assertion weakened).
- EXPECTED PROD SEQUENCE at next boot: cleanup removes 500 fossils →
  file <100 records → daemon autoseed refills ~1326 _seed records →
  cleanup now SPARES them → Kelly priors restored permanently.
- Gates: pytest 449 passed / 1 skipped (9 new + re-pins); tsc 64;
  version 1.0.155. VERIFY (pre-stated): post-deploy /api/diag/ml shows
  feedback_seeded_count ≈ 1326 (nonzero at minimum) and
  live_key_signatures no longer contains the 15-key 1.0.33 fossil
  signature; check after the next deploy-boot cycle.

## 2026-07-06 — [REPAIR] R12-D2: exit fills finally recorded — Bug #13's exit machinery gets its first callers (WS exits + position kills) (v1.0.154)

- Territory: T-BOT (server/bot.ts, new server/exitFill.ts + test) +
  SHARED (package.json, this file).
- DEFECT (found during R12, confirmed by grep of every track_fill call
  site): ml_model_v2.track_fill has carried exit-fill detection since
  Bug #13's fix (v1.0.34) — exit_context/exit_reason/is_close close
  the matching open entry with real outcome + pnl_pct — but NO bot.ts
  code path ever passed those keys. Both existing call sites are ENTRY
  fills. Even with v1.0.153's entry repair, every record would open
  and never close: live_performance stays zero forever.
- FIX: server/exitFill.ts `buildExitFillPayload` (pure, 4 node tests
  pin the exit-detection contract: exit_context present, qty_filled
  per the v1.0.153 resolution, bot-accounted pnl_pct which wins over
  track_fill's price recomputation, days_held derived NaN-safe) +
  `recordExitFill` in bot.ts (tmp-file + daemon-first pattern copied
  from the entry sites). Wired at the two FULL-exit sites: the WS
  monitor's final exit (stop/trailing/TP/time-stop, remaining qty) and
  the -25% POS-KILL forced liquidation ("position_kill" reason — the
  loop must learn from the worst trades too). Recording runs strictly
  AFTER the frozen order-POST paths; transmission untouched.
- SCALE-OUTS EXCLUDED BY DESIGN: a partial take-profit leaves the
  position open — closing the record on the first scale would label
  the whole trade with a partial P&L. The final WS exit carries the
  bot's cumulative pnl_pct, which is the honest label.
- KNOWN REMAINING GAPS (documented, not silent): (a) exits use the WS
  current price at submit time, not the confirmed Alpaca fill — same
  approximation the entry sites accept; (b) other exit paths (options
  manager exits, bot_engine-side closes, manual dashboard closes)
  still don't record — each nonrecording path leaves its entry open,
  where a LATER same-ticker exit could mislabel it; fossil purge +
  path-by-path wiring are the follow-ups; (c) April fossils
  (outcome=None) are still matchable by _find_entry_record on ticker
  collision — mitigated because exit_context.pnl_pct wins (bot
  accounting, not fossil entry price), fixed properly by the D4/D5
  cleanup PR.
- Gates: tsc 64 (baseline held); exitFill node tests 4/4; pytest 439
  passed / 1 skipped; version 1.0.154. VERIFY (pre-stated): after the
  next WS exit fires on prod, /api/diag/ml live_outcome_breakdown
  shows its first win/loss outcome and live_performance.total_trades
  goes ≥1. VXUS (time_stop) and SMH (trailing_stop) are already
  flagged by bot_engine — candidates within days.

## 2026-07-06 — [REPAIR] R12 ROOT CAUSE FOUND + FIXED: track_fill's qty guard silently dropped every regular-hours entry fill since 2026-04-23 — ML learned nothing live for 2.5 months (v1.0.153)

- Territory: T-BOT (ml_model_v2.py, new test_track_fill_qty.py) +
  SHARED (package.json, this file).
- SHAPE AGGREGATES DELIVERED (v1.0.152 probe, one deploy, prior ~60%
  CONFIRMED): all 500 live records share ONE key signature — exactly
  trackClosedTrades' feedback map (ticker,side,pnl_pct,holding_days,
  strategy,score,rules_score,ml_score,blended_score,won,instrument,
  entry_features,exit_context,timestamp,code_version) — with
  code_version **1.0.33** (the pre-Bug-25b variant, no pnl filter),
  pnl_pct 0 on all 500, entry_features null on all 500, dates
  2026-04-16 → 2026-04-20. They are April fossils; since the file cap
  keeps the NEWEST 500, their survival proves NOTHING has appended to
  trade_feedback since 2026-04-20.
- ROOT CAUSE (reproduced locally, then dated by git archaeology):
  bot.ts's regular-hours fill payload switched to
  qty_requested/qty_filled with NO "qty" key on 2026-04-23 (2479df0,
  the daemon-first change) — track_fill's `if qty <= 0: return` guard
  silently dropped every regular-hours entry fill from that day on.
  The morning-queue payload still carries "qty" but rarely fires.
  Dates align exactly: fossils end 04-20, payload changed 04-23.
  ANOTHER silent-discard instance of the v1.0.148/151 observability
  class (a silent `return`, not a silent `except` — the ratchet's
  lesson generalizes).
- FIX (one logical change): qty resolution accepts qty → qty_filled →
  qty_requested (filled wins over requested for partials; requested
  as fallback covers the FILL-CHECK-WARN path where confirmation
  failed and qty_filled is 0); a genuinely quantity-less payload still
  refuses to write but now LOGS the drop (voltrade.ml logger) instead
  of vanishing.
- RATCHET: test_track_fill_qty.py — 6 tests whose fixtures copy the
  EXACT current bot.ts payload key-sets (R6 lesson): regular-hours
  payload writes; partial fill uses filled qty; zero-filled falls back
  to requested; morning-queue contract unchanged; quantity-less drop
  logs + writes nothing; end-to-end entry→exit_context close (first
  test ever exercising Bug #13's exit machinery with today's entry
  contract).
- DOWNSTREAM CHAIN (REASONING STANDARD #1): entry records resume →
  fills_count/slippage stats resume feeding the fills tracker → the
  April fossils age out of the [-500:] cap as real records append →
  live_performance stays 0 until EXITS also record (defect D2, next
  PR) → only then does trade_feedback training data resume. This PR
  un-blocks the pipeline's first stage; it does NOT claim the loop is
  learning yet.
- REMAINING R12 DEFECTS (each its own PR): D2 no exit-side track_fill
  caller (exit_context never passed — needs bot.ts exit wiring); D3
  trackClosedTrades feedback block dead since the OOM fix hardcoded
  entryFeatures null (its filter rejects everything — decide repair vs
  staleness-removal AFTER D2 verifies); D4 seed wipe-out cycle (seeds
  lack code_version; boot cleanup deletes them every deploy; reseed
  threshold <100 never met while fossils hold 500).
- Gates: pytest 439 passed / 1 skipped (433 + 6 new); tsc untouched;
  version 1.0.153. VERIFY (pre-stated): after the next regular-hours
  entry fill on prod, /api/diag/ml shows fills_count > 0 and a second
  key signature (code_version 1.0.34 shape) in live_key_signatures;
  the fossil count starts shrinking below 500 as appends displace the
  cap. Check on the next trading session.

## 2026-07-06 — [REPAIR] ML feedback loop investigation pt.3 — the 500 mystery records: orphan_exit REFUTED, shape aggregates added to the ml diag probe (v1.0.152)

- Territory: T-BOT (server/bot.ts ml diag probe) + SHARED (package.json,
  this file). Continues the v1.0.146/147 diag-probe thread with the
  answer to its pre-stated question, plus the next instrument.
- FINDING (refutes v1.0.147's hypothesis, exactly as its own pre-stated
  check was designed to): live /api/diag/ml now shows
  `live_outcome_breakdown: {open: 500}` — ZERO orphan_exit outcomes, so
  the orphan_exit fallback shape is NOT what fills the file. Combined
  facts: 500 live records (0 seeded), fills_count=0 (no expected_price),
  total_trades=0 (no usable pnl_pct), all outcome-less.
- WRITER ELIMINATION (READ BEFORE WRITE — every candidate read this
  session): (a) ml_model_v2.track_fill entry path stamps expected_price
  + slippage_pct unconditionally → would make fills_count>0; (b) the
  backtest seeder stamps _seed → seeded_count would be >0; (c)
  bot.ts trackClosedTrades' feedback block filters pnlPct!==0 &&
  !==null before writing → total_trades would be >0; (d) legacy
  ml_model.track_fill writes FILLS_PATH, a different file. Every
  candidate contradicts one observed field — inference from absences
  is exhausted; the record SHAPE must be read from prod.
- TWO CONFIRMED DEFECTS filed regardless of the writer question (fixes
  are separate PRs, one logical change each):
  1. NO EXIT-SIDE track_fill CALLER: bot.ts calls track_fill at exactly
     2 sites (morning queue + regular batch), both ENTRY fills; no code
     path anywhere passes exit_context/exit_reason/is_close, so
     _is_exit_fill can never fire and no entry record can EVER be
     closed by the fill tracker. Bug #13's fix (v1.0.34, exit-detection
     inside track_fill) has had no caller feeding it since it shipped.
  2. SEED WIPE-OUT CYCLE: seed records carry no code_version, and the
     Node boot cleanup keeps only code_version >= '1.0.33' → every seed
     is deleted on every deploy; the daemon reseeds only when the file
     has <100 records, and the 500 mystery records hold it above that
     → Kelly-gate priors are permanently gone (seeded_count=0 live).
- INSTRUMENT SHIPPED (this PR): ml probe now reports field-NAME
  signatures (top 5 key-sets), code_version / session distributions,
  entry_features presence, pnl null/zero counts, and record date range
  — aggregates only, never tickers or prices, same token gate. One
  deploy names the writer definitively.
- PRIOR (stated before reading the result): the 'timestamp' field
  legacy note at bot.ts:2227 suggests trackClosedTrades-shaped records
  (they use 'timestamp', not 'time_filled'), meaning some historical
  variant of that block wrote pnl-less records before the 2026-04-20
  filter existed and the [-500:] cap preserved them forever. Confidence
  ~60%; the shape signature will confirm or refute.
- Gates: tsc 64 (baseline held), pytest untouched (inline python only),
  version 1.0.152. VERIFY (pre-stated): post-deploy /api/diag/ml
  returns the new aggregate fields; the dominant key signature
  identifies the writer; findings and the two defect fixes land as
  follow-up PRs with their own tests.

## 2026-07-06 — [REPAIR] silent-except CLASS ratchet — 329 handlers pinned per-file so the v1.0.148/v1.0.151 generator can only shrink (test-only, no version bump)

- Territory: T-BOT (new test_silent_except_ratchet.py) + SHARED (this
  file). Test-only PR — no runtime behavior change, so no version bump
  (attribution precedent: docs/test-only PRs don't tag).
- WHY NOW: the v1.0.151 session ran the mandatory loop-health check
  (8/10 [REPAIR]) and diagnosed the break-generator as OBSERVABILITY
  DEBT — broad excepts discarding errors with zero trace. It closed the
  deep_score INSTANCE (5 fetchers via _run_diag_fetch). This PR is the
  supersession-salvage delta: freeze the CLASS. An AST audit found
  **329 silent broad-except handlers across 40 runtime modules**
  (bot_engine.py 78, analyze.py 32, ml_model_v2.py 25, ...) — each one
  a pre-built blind spot of the exact shape that hid the SIP-403
  outage behind "empty scan" and degraded deep_score enrichment
  silently.
- WHAT COUNTS (AST, not regex): `except Exception|BaseException|bare`
  whose body is ONLY pass/continue/break/return-nothing-ish (None,
  constant, empty dict/list/tuple). Narrowed types, logging, diag
  capture, re-raise, and non-empty fallbacks all exit the definition —
  so the ratchet pressures toward exactly the house fixes.
- MECHANISM: per-file EXACT pins (not a global total — no smuggling a
  new handler in file B against a cleanup in file A). count>pin fails
  with fix guidance (_run_diag_fetch pattern / log / narrow the type);
  count<pin ALSO fails with "lower the pin in this same commit," so
  pins never go stale and the debt is monotonically shrinking. Raising
  a pin = weakening a test = forbidden. A second test pins the scanner
  itself against 12 classification fixtures so a refactor can't gut
  the ratchet silently. Scope: root runtime *.py + strategies/ +
  alphadesk/ (scripts/ is session-run tooling; test_* excluded).
- Learned while building: ast.parse() accepts `return` at module level
  (return-outside-function is a compile()-time check) — the scanner
  fixture originally assumed unparseable and the test caught it.
- Gates: new test 2/2; full pytest 433 passed / 1 skipped (baseline
  had grown to 433 via other sessions' merged tests; zero failures).
- EXPECTED EFFECT (prior, stated before live data): incident classes
  like "feed silently empty for hours" stop being creatable in new
  code; existing 329 shrink opportunistically as sessions touch those
  files. NOT expected: any change in trading behavior — this is pure
  meta (the ruler for code health, not the code).

## 2026-07-06 — [REPAIR] LOOP-HEALTH TRIGGER (8/10 REPAIR) diagnosed — generator identified as observability debt, closed for deep_score's 5 enrichment fetchers (v1.0.151)

- Territory: T-BOT (bot_engine.py deep_score, server/bot.ts Tier2 diag
  probe) + SHARED (package.json, this file, open_questions.md), shared
  edits kept last per WORKSTREAM PARTITION.
- SESSION-START CHECKS: read CLAUDE.md, this file, open_questions.md,
  wishlist.md in full before any edit (READ BEFORE WRITE).
  **LIVE HEALTH CHECK**: `GET /api/health` on prod —
  `{"status":"ok", "scanner":{"status":"ok","consecutiveFailures":0},
  "bot":{"status":"active","liveness":{"dark":false}}}`, uptime_s 2043,
  RSS 170MB. Confirms v1.0.150's SIP-entitlement fix (this morning) is
  live and its own pre-stated VERIFY criterion (consecutiveFailures reset
  to 0) already passed — no live break, no LIVENESS ALARM, nothing
  urgent blocking normal session budget.
  **LOOP-HEALTH RATIO (mandatory this session per instructions)**: last
  10 experiments.md entries at session start — v1.0.150 REPAIR, v1.0.149
  PRODUCT, v1.0.148 REPAIR, v1.0.147 REPAIR, v1.0.146 REPAIR, v1.0.145
  PRODUCT, v1.0.144 REPAIR, v1.0.143-verify REPAIR, v1.0.143 REPAIR,
  v1.0.142 REPAIR = **8/10 REPAIR**, past the 7+ trigger (v1.0.148/149
  already flagged 9/10 and 8/10 the previous two sessions — this is the
  THIRD consecutive session crossing it, which itself is worth taking
  seriously rather than re-stamping the same "not thrash" verdict from
  memory).
  DIAGNOSIS (HEALTH OF THE LOOP rule 2 — trace each entry to its actual
  cause, don't take the tag at face value): grouped the 8 REPAIR entries
  by root cause, not by tag —
    (A) OOM crash-loop, 2 entries (142 durability audit + 143/143-verify
        the actual fix+verify) — ONE incident, closed, ratcheted,
        verified live (uptime climbed 12x, heap steady). No recurrence.
    (B) /data trail static snapshot, 1 entry (144) — an isolated
        T-CLIENT freshness bug, unrelated to A.
    (C) ML diagnostic-probe build-out, 2 entries (146/147) — closing a
        long-standing KNOWN BROKEN item (deeper audit-log visibility),
        not a regression or re-break.
    (D) Scanner blind spot -> SIP entitlement, 2 entries (148/150) — ONE
        incident's two-step diagnose-then-fix (148 built the visibility,
        150 used it to find and fix the actual cause in one query, as
        148 predicted it would).
  VERDICT: not RECURRENCE-ESCALATES thrash (rule 4) — no issue marked
  fixed broke the same way twice. But a real pattern DOES exist across
  A/D specifically, worth naming as the generator: both incidents ran
  live for a nontrivial window (OOM crash-looping, then a scan blind
  spot "since Monday's open") BEFORE anyone could see them, because the
  relevant subsystem had no error detail surfaced anywhere — only after
  each session built a diagnostic (uptime on /api/health, then
  /api/diag/scanner) did the actual root cause become findable in one
  query instead of an archaeology session. That is a genuine, nameable
  generator: **historical observability debt**, and the last several
  sessions have been paying it down deliberately, not thrashing on it.
  ACTION TAKEN (not just re-logging the ratio): 148's own root-cause
  trace explicitly named this as "the exact same silent-degradation
  shape KNOWN BROKEN #5 already flagged for _fetch_macro/_fetch_intel/
  etc. — same root defect class, different call site" and left it there.
  Verified by reading `bot_engine.py`'s `deep_score()`: all 5 enrichment
  fetchers (`_fetch_macro`/`_fetch_intel`/`_fetch_alt`/`_fetch_social`/
  `_fetch_finnhub`) still had the identical bare `except Exception:
  return {}` with zero detail — the generator was NAMED in 148 but not
  yet CLOSED. Per "fix the generator of breaks, not the next break,"
  this session closes it, rather than starting a new unrelated
  repair/research thread — this IS this session's fall-through choice
  (SESSION BUDGET tier 1: a clearly identified, already-scoped, unclaimed
  item, found live rather than invented).
  ADDITIONAL FINDING while tracing: `diagnostics.py`'s existing
  `api_checks`/`extended_checks` (KNOWN BROKEN #5's 2026-07-04 fix) only
  ever checked CACHE FRESHNESS (does a prefixed file exist / how old is
  it) for these sources — never the actual exception. Cache-fresh but
  silently-erroring-every-call was always possible and would report
  green. Not touched this PR (that's `diagnostics.py`'s own auto-fix
  surface, tied to `reduce_position_size` thresholds — RULE REVIEW
  requires evidence + one-at-a-time for touching that); noted here and
  in open_questions.md as the natural next increment, not bundled in.
- WHAT SHIPPED (visibility only, mirrors v1.0.148's `_snap_diag` pattern
  exactly — zero change to what gets fetched, scored, or traded):
  1. `bot_engine.py`: new standalone `_run_diag_fetch(source_name, fn,
     diag)` — calls `fn()`, on exception records
     `f"{type(e).__name__}: {str(e)[:150]}"` into `diag[source_name]`
     (if `diag` is not None) and re-raises so each `_fetch_*` closure's
     existing `except Exception: return {}`-style fallback fires
     unchanged. Extracted standalone (not inlined per-closure) so it's
     unit-testable without deep_score's network calls or the
     ThreadPoolExecutor — mirrors why `_parse_snapshot_batch` was
     extracted in v1.0.148.
  2. `deep_score(ticker, quick_result, _diag=None)` — new optional
     kwarg, default `None` (fully backward-compatible; the only two
     callers, `_deep_one` and `test_full_system.py`'s direct calls, are
     unaffected). All 5 fetchers now route their body through
     `_run_diag_fetch` with a distinct source key.
  3. `_scan_market_inner`: new local `_source_diag: dict = {}`, passed
     into every `deep_score()` call via `_deep_one`; added to the
     cycle's final return dict as `"data_source_errors": _source_diag`
     — top-level ONLY, deliberately never inside `top_10`/`new_trades`/
     any per-candidate dict, so it can never reach ML features or the
     shadow_portfolio candidate log (unlike `_snap_diag`'s
     `debug_detail`, which only fires on the whole-scan-empty error
     path, this fires on every cycle whether or not deep_score ran into
     trouble — hence keeping it structurally separate from anything
     scored/logged mattered more here).
  4. `server/bot.ts`: new `tier2LastDataSourceErrors: Record<string,
     string>` (mirrors `tier2LastFailureDetail`'s pattern), set from
     `result.data_source_errors` on every successful scan (defaults to
     `{}`, so it self-clears once sources recover — no sticky-forever
     stale entries). Exposed as `dataSourceErrors` on the existing
     token-gated `/api/diag/scanner` probe, alongside
     `lastFailureDetail`. `/api/health` untouched — same discipline as
     v1.0.148 (free-text detail never on the public endpoint).
- DOWNSTREAM CHAIN (REASONING STANDARD #1): new diagnostic-only field on
  `_run_diag_fetch`'s success path returns `fn()`'s value completely
  unchanged -> every `_fetch_*` closure's return value on both success
  and failure is byte-identical to before this PR -> `deep_score`'s
  score computation, `reasons`, and every downstream field are untouched
  -> the only observable effects are (a) a new `data_source_errors` key
  on `scan_market`'s top-level return (Node already treats this as
  opaque JSON), (b) a new `dataSourceErrors` field on the token-gated
  `/api/diag/scanner` probe. Zero change to trade selection, sizing, ML
  features, or the shadow_portfolio log — confirmed by construction
  (the diag dict is a NEW top-level key, never threaded into any
  per-candidate dict).
- Gates: `python3 -m pytest -q` — 428 passed, 2 skipped (422 baseline +
  6 new in `test_deep_score_source_diag.py`; the 2 skips are pre-existing
  and unrelated: `xlrd` not installed in this session's environment,
  and the standing `backtest_v1028_full` skip). `npx tsx --test
  server/*.test.ts` — 287 passed (284 baseline + 3 new in
  `scannerHealth.test.ts`), zero regressions. `npx tsc --noEmit` — same
  pre-existing Buffer/Map-iteration/tsconfig error set documented in
  prior entries (verified none are on or near the changed lines) — zero
  new errors.
- MERGE TIMING NOTE (session ran during market hours, ~12:00 ET):
  visibility-only change to a non-trading-control-flow path (new
  diagnostic field, zero behavior change to scoring/trading), so the
  deploy-coupling risk is low — but per standing preference, this PR
  should wait for a human or routine to merge it AFTER 4:00 PM ET unless
  it's fixing a currently-live critical break, which it is not (health
  is fully green, this is a debt-paydown item, not an active incident).
- STARVED: no. This was the highest-value action available: the
  mandatory loop-health check surfaced a genuine, already-named-but-not-
  closed generator (KNOWN BROKEN #5's exception-visibility gap), scoped
  tightly to one subsystem, with its own regression tests.
- NEXT (open_questions.md KNOWN BROKEN #5 updated in place): (a)
  `diagnostics.py`'s `api_checks`/`extended_checks` still only check
  cache freshness, not real exception detail — a future RULE-REVIEW
  session could fold `data_source_errors` into `get_auto_fix_params()`'s
  problem surface, but that touches the `reduce_position_size`
  auto-fix's trigger conditions and needs its own evidence-backed PR;
  (b) once this deploys, the next time any of the 5 sources actually
  errors (dead Reddit RSS, expired FINNHUB_KEY, etc.) it will name
  itself on `/api/diag/scanner` instead of degrading silently — worth a
  live-verification note in a future session's health check.

## 2026-07-06 — [REPAIR] ⚠️ PRIORITY-1: SIP entitlement rejected (HTTP 403) — scan blind since Monday's open; central feed resolver + delayed_sip fallback across 44 sites (v1.0.150)

- DETECTION CHAIN (v1.0.148's blind-spot fix paid off in ONE query,
  exactly as designed): human directive "make sure the trading bot is
  operating correctly" -> /api/health degraded, scanner
  consecutiveFailures=13 during market hours -> /api/diag/scanner
  (DIAG_TOKEN) -> lastFailureDetail: "HTTP 403". Archaeology-to-
  diagnosis in minutes instead of a session.
- ROOT CAUSE: every market-data request with feed=sip is rejected
  with 403 since Monday's open (SIP entitlement — subscription
  lapse or Alpaca policy change; the trading API and /v2/account are
  unaffected). 44 call sites across 15 modules hardcoded feed=sip,
  so the outage was whole-stack: Tier2 scan (zero new candidates all
  morning), options scanner, VXX regime reads, SPY floor, shadow
  backfills, ML training fetches. Stops/manage_positions kept
  working (trading API); equity peak even rose intraday — the bot
  was safe but BLIND to new opportunities.
- FIX: new alpaca_feed.py — ONE probe, ONE switch, zero per-site
  error wiring. data_feed() probes the SIP entitlement (1-symbol
  snapshot, 10-min TTL): 403 -> process-wide downgrade to
  **delayed_sip** (the FULL consolidated tape at a 15-minute delay,
  free on every tier), 200 -> auto-restore to real-time sip; time-
  outs/5xx are inconclusive and never flap the feed. All 44 sites
  swept onto the resolver (mechanical substitution, every URL site
  verified f-string, imports verified top-level after one
  mis-insertion was caught by compileall).
- MEASUREMENT-INTEGRITY DECISION (why NOT feed=iex): IEX-only prints
  undercount consolidated volume ~30-50x — a silent iex fallback
  would poison every dollar-volume floor ($50M scan gate, options
  liquidity checks) and reproduce the same empty-scan symptom with a
  worse cause. delayed_sip preserves volume semantics; candidate
  DISCOVERY tolerates 15-min staleness; executions price off live
  quotes via the trading API as before. The downgrade is loudly
  logged once per switch and auto-reverts when the entitlement
  returns — no deploy needed either direction.
- RATCHET: test_alpaca_feed.py — 6 resolver tests (403->delayed_sip,
  TTL caching, auto-recovery, inconclusive-probe no-flap, env
  override, iex-never-chosen documented) + a source-scan banning ANY
  hardcoded feed choice from runtime modules (the 44-site dispersion
  is what turned an account-level change into a stack-wide outage).
- BLOCKED-FOR-MIKE #9 (urgent, push-notified): check the Alpaca
  dashboard Market Data subscription — decide between restoring paid
  real-time SIP (~$99/mo Algo Trader Plus) or accepting 15-min
  delayed discovery. The bot self-upgrades within 10 min of restore.
- Gates: pytest 425/1 skipped (419 baseline + 6 new, ZERO regressions
  across the swept stack); compileall clean on all 16 touched
  modules; tsc untouched (no TS changes).
- VERIFY (pre-stated): post-deploy, /api/diag/scanner must show
  consecutiveFailures reset to 0 and /api/health scanner "ok" within
  ~15 min (next Tier2 cycle after the probe downgrades the feed).
- VERIFIED (same day, post-deploy): /api/diag/scanner ->
  `{degraded:false, consecutiveFailures:0, lastFailureDetail:null}`;
  /api/health overall "ok" — server {uptime_s:2304, heap 68MB,
  rss 173MB}, bot "active", alpaca "ok", scanner "ok". The scan is
  discovering again on delayed_sip (honest full-tape volume, 15-min
  delay). Criteria met exactly as pre-stated; incident CLOSED. Only
  open thread is Mike's wishlist #9 subscription decision — the bot
  self-upgrades to real-time sip within 10 min of a restore, no
  deploy needed.

## 2026-07-06 — [PRODUCT] Everything Graph build step 2 — server/entityGraph.ts + /api/data/graph (v1.0.149)

- Territory: T-DATACORE (server/entityGraph.ts, server/entityGraph.test.ts,
  portDwell.ts refactor) + SHARED (server/routes.ts route wiring,
  package.json version bump, this file) — shared edits kept last and
  minimal per WORKSTREAM PARTITION.
- SESSION-START CHECKS: read CLAUDE.md, this file, open_questions.md,
  wishlist.md in full before any edit (READ BEFORE WRITE).
  **LIVENESS/HEALTH CHECK (repair mandate, [PRODUCT] sessions don't
  preempt DAILY repair duty but must not ignore it)**: the v1.0.148 entry
  above logs an ONGOING KNOWN BROKEN item — Tier2 scan failures
  ("Could not fetch market data from Alpaca") still active as of that
  entry's close, with the visibility fix (`/api/health` scanner check +
  `/api/diag/scanner`) merged but not yet confirmed live (Railway hadn't
  redeployed at last check). This session has no Alpaca/Railway access to
  verify further and the item is T-BOT territory — noted here per the
  repair-mandate awareness rule, not acted on; the next T-BOT/DAILY
  session should query `/api/diag/scanner` first.
  **LOOP-HEALTH RATIO**: last 10 tags including this entry — v1.0.149
  PRODUCT (this entry), v1.0.148 REPAIR, v1.0.147 REPAIR, v1.0.146 REPAIR,
  v1.0.145 PRODUCT, v1.0.144 REPAIR, v1.0.143-verify REPAIR, v1.0.143
  REPAIR, v1.0.142 REPAIR, v1.0.141 REPAIR = 8/10 REPAIR, still past the
  7+ trigger. Not re-diagnosing from scratch: the v1.0.148 entry already
  traced 6 of those 9 REPAIR entries to one OOM incident's full lifecycle
  (closed, ratcheted) and the other 2 to one continuing diagnostic thread
  on the live scan-failure item above — nothing changed in that trace
  since. This session's own action is the first non-repair, non-DAILY-
  routine [PRODUCT] work in that window, which is the intended correction
  (SESSION BUDGET's fall-through: pick the highest-value PRIMARY action;
  a [PRODUCT] session's job is exactly to counterweight the recent
  REPAIR-heavy run, not to invent a new repair).
- WHY THIS BECAME THE SESSION: per the [PRODUCT] session mandate, chose
  among (a) advance a datacore pipeline through its next ladder gate, (b)
  build /data UI, (c) propose a new hypothesis, (d) improve datacore's API
  boundary. `datacore/EVERYTHING_GRAPH.md` (the flagship R5 roadmap item,
  charter directive 2026-07-04) names its build plan explicitly: step 1
  (`datacore/entity_map.json`) shipped 2026-07-05 (v1.0.131); step 2
  (`server/entityGraph.ts` + `/api/data/graph` + tests) was logged
  **NEXT (unclaimed)** in open_questions.md R5 — the clearest, already-
  scoped, highest-value product action available this session (option
  (a)/(d): a datacore pipeline advancing to its next stated milestone,
  which also is the platform's API boundary for every fusion hypothesis
  named in the constitution's EDGE DOCTRINE/ACTIVE ANGLE-HUNTING
  sections). No RESEARCH tier needed — this was a queued, unclaimed
  roadmap item (SESSION BUDGET fall-through tier 1).
- WHAT SHIPPED: `server/entityGraph.ts` — a pure, IO-overridable builder
  (mirrors the shadowFleet/portDwell baseDir-injectable pattern) that
  joins four existing sources into one node/edge graph, per the design
  doc's exact v1 spec:
  - **facility nodes**: every `datacore/sites` site (16) + every
    `datacore/powerplants` plant (9,833) — lat/lon carried as an
    intrinsic node attribute rather than a separate `located_at` edge
    (a deliberate, documented deviation from the design doc's literal
    edge table: the entity-type table itself calls facility lat/lon
    "intrinsic" and v1's entity types never define a "geo" node to point
    a `located_at` edge at — inventing one would be undocumented scope
    creep for zero query-answering difference).
  - **operates edges** (company -> facility): from `datacore/entity_map.json`'s
    hand-verified operator/owner -> ticker table (step 1's output) — only
    the 44 mapped entries produce an edge; the 25 honest-unmapped entries
    (municipal authorities, private post-bankruptcy generators, etc.)
    correctly produce none, preserving entity_map's own no-guessed-
    tickers rule at the graph layer.
  - **insider_of edges** (person -> company): from `edgarForm4.readFilingHistory`'s
    30-day archive, aggregated across every filing for a given CIK/issuer
    pair (roles, filing_count, first/last seen, last transaction kind —
    the most recent filing's transaction wins). Issuer identity prefers
    the filed ticker; the rare ticker-less filing falls back to a
    `company:cik:<CIK>` id, explicitly flagged `ticker_known:false` so it
    is never silently conflated with a real ticker node downstream (the
    identity-resolution problem the design doc calls "the hard part" is
    not solved here — this is the honest boundary of what's solvable
    without a second join).
  - **calls_at edges** (vessel -> facility/port): required extracting
    `foldPortVisitsAsync` out of `portDwell.ts`'s `computePortDwellAsync`
    (the online, bounded-memory AIS fold added in the 2026-07-05 OOM
    repair) so the graph builder reuses the exact same archive pass
    instead of re-implementing visit detection — a byte-identical
    refactor, verified by the existing `portDwell.test.ts` suite (8/8
    still passing unchanged). Visits aggregate per vessel/port pair into
    visit_count + last_call + median_dwell_h, the design doc's named
    attributes.
  - Every edge carries `{source, confidence, first_seen, last_seen}` per
    the design doc's envelope-alignment note; `operates` edges use the
    entity_map's `built` date as a constant timestamp (static reference
    data, no observation time dimension); `insider_of`/`calls_at` use
    real filing/AIS timestamps.
  - `/api/data/graph`: same eager-poller-cache shape as
    `/api/data/portdwell`/`/api/data/shadowstats` (15-min interval,
    matching the design doc's stated cache window) — the graph rebuild
    folds a 168h AIS window, so it must NEVER run synchronously
    per-request; that is the exact event-loop/OOM defect class those two
    routes were repaired for in v1.0.125/126/143, and this route is built
    correctly from day one instead of needing its own repair later.
    Without `?entity=`, the route returns COUNTS ONLY (never the full
    ~9.9k-node graph by default); `?entity=<ticker|MMSI|CIK|facility id>&hops=1..3`
    (default 1, capped at 3) returns a BFS neighborhood subgraph.
    `resolveEntityId` accepts a bare ticker/MMSI/CIK or a full node id.
- DOWNSTREAM CHAIN (REASONING STANDARD #1): this is a new read-only
  endpoint over already-archived/registry data -> zero change to any
  existing route's behavior, zero change to scan/scoring/trading control
  flow -> the only new runtime cost is one additional 168h AIS fold every
  15 minutes (same class of cost `/api/data/portdwell` already pays on
  its own 10-min cycle; not a new archive-read pattern, just one more
  consumer of the existing bounded fold) -> the only observable product
  effect is a new, discoverable `/api/data/graph` endpoint; nothing here
  is a SIGNAL (every edge is RAW, provenance-carrying, no predictive
  claim, matching the design doc's ground rule 2) so no ladder gate
  applies to this PR.
- NOT BUILT THIS PR (explicitly out of scope, per "one logical change"):
  the `/data` graph panel UI (design doc build-plan step 3) and an
  `/api/v1/graph` keyed mirror (the existing `stats/portdwell` /
  `stats/shadow` pattern) — both are natural, low-risk fast-follows now
  that the data shape exists, queued below rather than bundled in.
- Gates: `npx tsc --noEmit` — same pre-existing Buffer/Map-iteration/
  tsconfig error set documented in prior entries; zero new errors on
  `server/entityGraph.ts` or the `server/routes.ts` diff (verified by
  filtering the output for both files — empty). `npx tsx --test
  server/*.test.ts` — 284 passed (274 baseline + 10 new in
  `entityGraph.test.ts`), zero regressions, `portDwell.test.ts`'s 8/8
  unchanged post-refactor. `python3 -m pytest -q` — 419 passed, 1 skipped
  (no Python touched this PR; confirms the Python gate is unaffected by a
  TS-only change).
- MERGE TIMING NOTE (per this session's product-session instructions):
  session ran 09:05-ish ET, crossing the 09:30 ET open. This PR touches
  no trading logic and the new route is purely additive/read-only, so
  the deploy-coupling risk is low, but per the standing preference this
  PR should merge outside 09:30-16:00 ET if a human or routine reviews it
  during market hours — otherwise safe to merge whenever CI is green
  (no bot/strategy code path changed).
- STARVED: no. This was a clearly queued, unclaimed, already-scoped
  roadmap item with capacity to spare; the natural fall-through (UI panel
  step 3, /api/v1 mirror) is queued below for the next session rather
  than rushed into this PR.
- NEXT (updates open_questions.md R5): step 3 — `/data` graph panel
  (entity search -> neighborhood card: insiders + recent buys, operated
  facilities on the map, vessels calling at its ports; DESIGN.md self-see
  at three widths) is now unblocked. A `/api/v1/graph` keyed mirror
  (mirrors `stats/portdwell`/`stats/shadow`) is a smaller, independent
  fast-follow. Both queued, neither claimed by this entry beyond noting
  them.

## 2026-07-06 — [REPAIR] Tier2 scan-failure blind spot: silent-except swallowed the real Alpaca error; scan health now visible on /api/health + a new diag probe (v1.0.148)

- Territory: T-BOT (bot_engine.py scan_market, server/bot.ts Tier2 +
  /api/health + diag route, server/diag.ts whitelist) + SHARED
  (package.json, this file), shared edit last per WORKSTREAM PARTITION.
- SESSION-START CHECKS (repair mandate + loop-health, per standing
  protocol): read CLAUDE.md, experiments.md, open_questions.md,
  wishlist.md.
  **LOOP-HEALTH RATIO**: last 10 experiments.md entries by tag —
  v1.0.147 REPAIR, v1.0.146 REPAIR, v1.0.145 PRODUCT, v1.0.144 REPAIR,
  v1.0.143-verify REPAIR, v1.0.143 REPAIR, v1.0.142 REPAIR, v1.0.141
  REPAIR, v1.0.140 REPAIR, v1.0.139 REPAIR = **9/10 REPAIR**, past the
  7+ trigger. Diagnosed WHY before doing anything else (per HEALTH OF
  THE LOOP rule 2): traced each entry to its actual cause rather than
  taking the tag at face value. Verdict — NOT the thrash pattern the
  rule exists to catch: 6 of the 9 (139/140/141/142/143/143-verify)
  are ONE production incident (the 2026-07-05 OOM crash-loop) worked
  through its full lifecycle — mitigate -> alarm/observability ->
  audit -> root-cause -> verify — closed with a real fix, a compiled
  general LESSON ("streaming has two dimensions: event-loop AND heap"),
  and ratchet tests (archiveFoldMemory.test.ts); the event-loop class
  it built on (v1.0.125/126, portdwell/shadowstats) was itself already
  CLASS CLOSED with a stated pattern rule for new surfaces — not a
  RECURRENCE-ESCALATES violation (no issue marked fixed broke the same
  way twice; the memory dimension was a distinct, immediately-diagnosed
  new dimension of the same family). The other 2 (146/147) are one
  continuing diagnostic investigation of a long-standing KNOWN BROKEN
  item, not repeated failed fixes. No structural "generator of breaks"
  found beyond what's already been addressed. Logged here per the rule
  (the check itself is the deliverable when the numeric trigger fires
  but the qualitative read is healthy) — not escalated to a Priority-1
  meta-investigation.
  wishlist.md URGENT item #8 (prod restart-loop, flagged 2026-07-05)
  independently confirmed STALE (matches v1.0.146's same finding, one
  session earlier): live uptime_s 29631s+ this session, zero restart
  signature. Left for a follow-up docs-only close-out (small, does not
  justify its own PR); noting here so it isn't lost.
- WHY THIS BECAME THE SESSION (repair mandate — Priority 1, KEEP THE
  SYSTEM ALIVE, "data flowing"): picked up the v1.0.147 entry's queued
  NEXT step (query `/api/diag/ml` post-redeploy for the orphan_exit
  breakdown) but found the LIVE deployment still serves pre-v1.0.147
  code (uptime 29631s, response has `feedback_seeded_count` from
  v1.0.146 but no `live_outcome_breakdown` from v1.0.147 — Railway
  hasn't picked up the merge yet; nothing actionable server-side, this
  session cannot trigger a redeploy). While confirming that, the
  `/api/diag/audit` tail surfaced a MORE urgent, currently-live issue:
  **TIER2-BACKOFF failure #9 through #20 (at least 12 consecutive,
  ~2h+ span so far), every one "Scan returned error: Could not fetch
  market data from Alpaca (via daemon)"**, while `/api/health` read
  "ok" throughout (bot "active", Alpaca REST check passing via
  `/v2/account`, daemon alive). This is exactly the LIVENESS ALARM's
  blind spot: the loop is nominally active but its Tier2 scan — the
  only thing that finds NEW trade candidates — has not succeeded once
  in the observed window, and nothing surfaces that short of reading
  the token-gated audit tail by hand.
- ROOT-CAUSE TRACE (REASONING STANDARD #1, before touching code):
  `bot_engine.py`'s `_scan_market_inner()` returns `{"error": "Could
  not fetch market data from Alpaca"}` only when `quick_results` is
  empty across the ENTIRE ~11,600-symbol universe. That requires
  `snap_all` (built by `_fetch_snap`, one call per 400-symbol batch to
  `/v2/stocks/snapshots?...&feed=sip`) to be empty for every batch.
  `_fetch_snap` had a bare `except Exception: return {}` AND silently
  skipped any response shape that wasn't per-symbol-keyed (e.g. an
  error body `{"code":..., "message":...}` parses as zero usable
  symbols with no exception at all) — so a genuine API problem (auth,
  SIP feed entitlement, rate limit, malformed request) and a
  legitimately-empty batch were INDISTINGUISHABLE from the caller's
  side, by design. This is the exact silent-degradation shape KNOWN
  BROKEN #5 already flagged for `_fetch_macro`/`_fetch_intel`/etc. —
  same root defect class, different call site. Could not confirm the
  underlying Alpaca-side cause this session (no Alpaca credentials in
  the session env, by design — only DIAG_TOKEN is provisioned); this
  PR fixes the BLIND SPOT (make the real reason loggable) so the next
  live occurrence names its own cause instead of requiring another
  archaeology session.
- FIX (visibility only — zero change to what gets traded, scored, or
  filtered):
  1. `bot_engine.py`: extracted `_fetch_snap`'s parse+filter body into a
     standalone pure function `_parse_snapshot_batch(raw, status_code)`
     (module-level, no network, no globals — unit-testable without
     mocking the whole scan environment). Behavior is byte-identical;
     it now also returns a `detail` string naming WHY a batch came back
     empty (non-200 status, non-dict/error-shaped 200 body, or "0/N
     symbols had a usable dailyBar" for the entitlement/pre-open edge
     case). `_scan_market_inner`'s empty-result error now carries
     `debug_detail` from the last such reason observed across all
     batches.
  2. `server/bot.ts`: `tier2Intelligence` now captures
     `result.debug_detail` (and, in the exception-catch branch, the
     existing stderr/signal/code summary it already computed but never
     retained) into a new `tier2LastFailureDetail` closure var, cleared
     on every scan start and set to null on success. The "Scan returned
     error" audit line now includes it inline.
  3. `/api/health` (UNAUTHENTICATED — no token, no owner cookie) gains a
     `scanner` check: `{status, consecutiveFailures}` only — status
     flips to `degraded` (overall health too) once
     `tier2ConsecutiveFailures >= 6` (`server/scannerHealth.ts`,
     `SCANNER_DEGRADED_FAILURE_THRESHOLD` — chosen to sit at/past the
     scheduler's own 600s backoff cap, so a transient blip that clears
     within its first few retries never trips it). Deliberately NO
     free-text detail on this public endpoint — that's a materially
     bigger disclosure surface than the existing single-line
     `err?.message` on checks 2/3 (subprocess stderr tails can run to
     hundreds of characters). The full `tier2LastFailureDetail` instead
     ships as a NEW whitelisted probe, `/api/diag/scanner`
     (`DIAG_PROBES` extended in `server/diag.ts`), gated + sanitized
     exactly like the existing `ml`/`daemon`/`positions`/`audit`
     probes — same wishlist-approved (d) mechanism, no auth.ts touch.
- DOWNSTREAM CHAIN (REASONING STANDARD #1): new diagnostic fields only
  -> `scan_market`'s control flow, return shape for the SUCCESS path,
  and every trade-selection/scoring code path are byte-identical ->
  the only observable effects are (a) a `debug_detail` key appended to
  an already-error-shaped dict Node already treats as opaque JSON, (b)
  a new `scanner` object in two JSON responses, (c) overall
  `/api/health` status can now go "degraded" when it previously stayed
  "ok" during a scan-failure streak — this is the intended new
  behavior (surfacing loudly, per the LIVENESS ALARM directive's
  spirit) and could trip an external uptime monitor pointed at
  `/api/health` for the first time; that is the point, not a
  regression, and mirrors exactly how Check 5 (bot-state liveness) and
  Check 6 (licensing) already behave.
- Gates: `python3 -m pytest -q` — 416 passed, 2 skipped (410 baseline +
  6 new in `test_snapshot_parse_diag.py`, zero regressions). `npx tsx
  --test server/*.test.ts` — 274 passed (268 baseline + 6 new: 5 in
  `scannerHealth.test.ts` covering the threshold logic, the wiring pin
  on both the public health check and the token-gated probe, and a
  dedicated test asserting `tier2LastFailureDetail` is ABSENT from the
  `/api/health` scanner check specifically; 1 extra assertion folded
  into the diag suite's existing dynamic `DIAG_PROBES` loop covers the
  new `scanner` case). `npx tsc --noEmit`: same pre-existing
  Buffer/Map-iteration/tsconfig error set as documented in the
  v1.0.146/147 entries (verified none are on or near the changed
  lines) — zero new errors.
- STARVED: no. This was the highest-value action found live this
  session (a currently-ongoing Priority-1 blind spot, versus continuing
  the already-well-scoped-for-next-time ml orphan_exit thread, which
  needs a Railway redeploy this session cannot trigger before it can
  proceed further anyway).
- NEXT: once this deploys, the next Tier2 failure (if the underlying
  Alpaca-side cause hasn't cleared on its own — still ongoing as of
  this entry, failure #20 at 11:17Z) will name itself in the audit log
  and via `/api/diag/scanner`, closing this from archaeology to a
  one-query diagnosis. If it turns out to be feed-entitlement-shaped
  (e.g. consistently `HTTP 40x` or "0/N symbols" during the identical
  pre-market window every trading day), that becomes its own filed
  finding once observed — not assumed here. wishlist.md item #8 close-
  out and the v1.0.147 orphan_exit follow-up remain queued for next
  session.

## 2026-07-06 — [REPAIR] diag `ml` probe pt.2 — outcome breakdown, chasing the v1.0.146 finding (v1.0.147)

- Territory: T-BOT (same file/route as v1.0.146, continuing the same
  repair thread within this session — not a new logical change,
  splitting it into its own PR only because it needed a live redeploy
  + re-query between the two code changes to know what to build next).
- LIVE RESULT FROM v1.0.146 (queried immediately after that PR's
  Railway redeploy, uptime_s reset 8133->140 confirmed the new code was
  live): `{"feedback_count":500,"fills_count":0,
  "feedback_seeded_count":0,"feedback_live_count":500,
  "live_performance":{"total_trades":0,"win_rate":0,...}}`.
  **This resolves the ambiguity v1.0.146 set out to resolve, and the
  answer is the concerning one**: all 500 feedback records are LIVE
  (zero seeded — `seed_feedback_from_backtest.py` has apparently never
  run against this deployment, or its output was fully rotated out),
  yet ZERO of them count as a completed trade for win-rate purposes.
- REASONING (per REASONING STANDARD #1, tracing the mechanism before
  touching any code): `check_model_health()`'s live-performance filter
  requires `pnl_pct is not None`; `fills_slippage_stats()` requires
  `expected_price` truthy AND `slippage_pct` present. A normal ENTRY
  record from `track_fill()` (ml_model_v2.py:2428-2444) sets
  `expected_price`/`slippage_pct` unconditionally at creation and only
  gets `pnl_pct` filled in later, on a matching EXIT. So an ordinary
  entry that's still open (`pnl_pct=None`) WOULD still count toward
  `fills_count` (it has `expected_price`/`slippage_pct` from creation)
  — it just wouldn't count toward `total_trades` yet. Getting
  `fills_count=0` simultaneously with `feedback_live_count=500`
  requires records that have NEITHER field — and `track_fill()`'s
  `_is_exit_fill` branch has exactly one such shape: the "orphan exit"
  fallback (ml_model_v2.py:2409-2425, fires when an exit fill finds no
  matching open ENTRY record) — appends `{ticker, side, qty,
  fill_price, time_filled, outcome:"orphan_exit", pnl_pct:None,
  note, code_version}` with no `expected_price`/`slippage_pct` at all.
  HYPOTHESIS (stated before checking, per REASONING STANDARD #10): all
  or nearly all of the 500 live records are `orphan_exit` — i.e. exit
  fills are reaching `track_fill()` but ENTRY fills for those same
  positions never did (or were evicted before the matching exit
  arrived), so the ENTRY/EXIT matching in `_find_entry_record` is
  failing at scale, not just occasionally. Kill condition: if the
  breakdown instead shows most records as `outcome: "open"` (an entry
  that simply hasn't exited yet), that's a different, much less urgent
  story (positions pending, not a matching failure) and changes what
  gets fixed next.
- BUILD: extended the same `ml` diag probe with `live_outcome_breakdown`
  — aggregate counts of live (non-seeded) records grouped by `outcome`
  (`None` normalized to `"open"` in the label only), e.g. `{"open": 3,
  "win": 40, "loss": 55, "orphan_exit": 402}`. Aggregate counts only —
  no ticker, no price, no timestamp — same sensitivity class as
  `live_performance`, well inside the approved whitelist scope.
  Verified locally against a synthetic 4-record fixture covering all
  four shapes (open/orphan_exit/win/seeded) before touching the live
  probe: `fills_count` counted only the entry+win records with
  `expected_price` (2), `check_model_health` counted only the closed,
  non-seeded win record (`total_trades: 1`), and the breakdown showed
  `{"open": 1, "orphan_exit": 2, "win": 1}` — exactly the shape the
  hypothesis predicts finding at scale in prod.
- Gates: `npx tsx --test server/*.test.ts` 268/268 (the 3 sandbox-
  network-flaky failures from the v1.0.146 entry did not reproduce this
  run — transient, not a regression either way); new assertion added to
  the same `server/diag.test.ts` block pinning `live_outcome_breakdown`
  presence; `python3 -m pytest -q` 410 passed/2 skipped (unchanged;
  `diagnostics.py`/`ml_model_v2.py` still untouched, read-only
  consumption again).
- NEXT (same session, after this PR's redeploy): query `/api/diag/ml`
  once more; if `orphan_exit` dominates as hypothesized, that IS the
  KNOWN BROKEN #3/#4 root cause — the actual fix (in
  `ml_model_v2.py`'s `_find_entry_record`/`track_fill`, MUTABLE
  territory, not frozen) becomes its own follow-up [REPAIR] PR with a
  regression test, per the loop-health rule that a repair without a
  test isn't complete; if it shows mostly `open` instead, the write-up
  changes to "positions pending, no matching bug" and this thread
  closes without a code fix.
- STARVED: no.

## 2026-07-06 — [REPAIR] diag `ml` probe distinguishes seeded vs. live feedback + surfaces live win-rate — closing the loop on KNOWN BROKEN #3/#4 (v1.0.146)

- Territory: T-BOT (server/bot.ts diag route; reuses diagnostics.py
  unchanged) + SHARED (package.json, this file) — small, one logical
  change, last commit before the PR per WORKSTREAM PARTITION rule 1.
  Checked `list_pull_requests` first: only PR #77 open (stale draft,
  fix/tier2-full-scan-oom, unrelated, since April) — no collision.
- SESSION-START CHECK (repair mandate + liveness, per the standing
  protocol): read CLAUDE.md, all of research/, then live `/api/health`
  — uptime_s 7191 (~2h, no restart-loop signature), heap/rss steady
  (39/142MB), bot active, drawdownPct 0.0, licensing ok. The URGENT
  wishlist item #8 (prod restarting every ~61s, flagged 2026-07-05,
  "if unresolved by Monday 09:30 ET the bot cannot trade") reads as
  ALREADY RESOLVED — today (2026-07-06, the named Monday) uptime is 2
  hours, not restart-looping. Not touched further this session (no
  code to write for an already-resolved item); left for a docs-only
  wishlist close-out once the live probe's fuller picture below lands.
- WHY THIS BECAME THE SESSION (repair mandate applies): KNOWN BROKEN #3
  ("CSP execution cascade — verify Tier 2 CSP trades actually fire")
  and #4 ("bot doesn't work right — ACCESS LIMITATION: deeper
  diagnostic routes are requireOwner, autonomous sessions cannot read
  audit logs/trade_feedback from outside") were never closed. The
  access limitation itself was already resolved 2026-07-04 (wishlist
  option (d): the token-gated `/api/diag/:probe` route, `server/diag.ts`
  + `server/diag.test.ts`, shipped and live) — but NO session since had
  actually exercised it to answer #3/#4's original questions. Read
  DIAG_TOKEN from this session's own environment (present, set by the
  human per the wishlist grant) and queried prod directly for the first
  time: `positions` (4 open, $59,109 gross), `daemon` (alive, uptime
  7340s, RSS 165/1024MB), `audit` (Tier 3 cycles running normally,
  "ML model fresh (19.1h old) — skipping retrain" three cycles running,
  but ALSO "System health: warning — 1 issues" on every one of the last
  three hourly Tier-3 cycles), `ml` → `{"feedback_count":500,
  "fills_count":0}`.
- THE FINDING (REASONING STANDARD #1 — trace before touching anything):
  `fills_count: 0` out of `feedback_count: 500` is ambiguous from the
  raw probe alone and could mean two very different things: (a) healthy
  — all 500 records are backtest-seeded (`_seed` flag,
  `seed_feedback_from_backtest.py`) and zero live trades have completed
  yet, so there's nothing to alarm on; or (b) broken — live trades ARE
  completing (the recurring "1 issues" Tier-3 warning requires
  `total_trades > 20` in `diagnostics.check_model_health()`'s low-win-
  rate check, which explicitly filters OUT `_seed` records first — so
  if that warning is firing, more than 20 real live trades have
  completed) yet none of them carry `expected_price`/`slippage_pct` —
  which would mean `track_fill()`'s entry-write path (the one that sets
  those two fields, ml_model_v2.py:2432-2434) isn't actually being hit
  for real fills, silently breaking the realistic-P&L/slippage-gap
  metrics `KNOWN BROKEN #3/#4` asked about. The raw probe cannot
  distinguish these — it needed the SAME seeded/corrupt-record filter
  `check_model_health()` already implements (dashboard alpha-audit
  batch 3, 2026-05-03) to separate the two.
- FIX (read-only, additive, no trading-path change): extended the
  `ml` diag probe (bot.ts, wishlist-approved whitelist scope — "ml
  status" already covers exactly this) to also report
  `feedback_seeded_count`, `feedback_live_count`, and
  `check_model_health()`'s `live_performance` dict (`total_trades`,
  `win_rate`, `recent_win_rate_20`, `degradation_detected`) alongside
  `retrain_needed`/`retrain_overdue`. Deliberately REUSED
  `check_model_health()` rather than re-deriving the seeded/corrupt
  filter inline (EDGE DOCTRINE #3 — never re-reason what's already
  compiled into code); zero changes to `diagnostics.py`,
  `ml_model_v2.py`, or any trading-path file — this is strictly a
  read-path addition to an already-approved diagnostic surface.
- DOWNSTREAM CHAIN (REASONING STANDARD #1): one more `diagnostics`
  import + two more dict lookups on the existing `ml` probe's Python
  one-liner -> no new subprocess, no new schedule, no change to what
  Tier 2/Tier 3 actually do -> the ONLY observable effect is more
  fields in an already-token-gated, already-sanitized, already-closed-
  by-default read-only response. `sanitizeDiag` still wraps the whole
  payload (defense-in-depth unchanged); the new fields are counts and
  percentages, strictly less sensitive than the ticker-bearing
  `recentTrades` array the owner-only `/api/bot/performance` route
  already exposes.
- Gates: `npx tsx --test server/*.test.ts` — 254 passed (3 pre-existing
  failures on `compression`/`gdeltEvents`/`owmTiles`, reproduced
  identically on main before this change via `git stash` — sandbox
  network restrictions, not a regression); new test in
  `server/diag.test.ts` pins that the `ml` probe block calls
  `check_model_health` and reports `feedback_seeded_count`/
  `feedback_live_count`/`live_performance` (would fail if a future
  edit silently dropped the enrichment). `npx tsc --noEmit`: same
  pre-existing Buffer/Map-iteration error set as documented in prior
  entries, zero new errors near the changed lines. `python3 -m pytest
  -q`: 410 passed, 2 skipped (installed numpy/pandas/lightgbm/pytest
  fresh into this sandbox, which lacked them; unchanged pass count
  otherwise) — `check_model_health()` itself is untouched, so no new
  Python tests were needed for it; the new consumer is TypeScript-side
  only.
- NOT YET DONE (this PR is the instrumentation; the verdict needs the
  live redeploy): once this merges and Railway redeploys, the NEXT
  query against `/api/diag/ml` will show the real `feedback_seeded_
  count`/`feedback_live_count`/`live_performance.total_trades` split —
  that live readout is what actually closes KNOWN BROKEN #3/#4, not
  this PR by itself. Queuing the follow-up docs-only entry once that
  data is in hand (same session, no context loss expected).
- STARVED: no — this was the highest-value action available (closing a
  dangling repair item using a capability a prior session already built
  but never used), chosen over starting a new data pipeline.

## 2026-07-06 — [PRODUCT] FINRA short-volume gets its /data full view — ticker lookup into the 2024-06-17+ archive + market-wide trend (v1.0.145)

- Territory: crosses T-DATACORE (server/finraShortVolume.ts) + T-CLIENT
  (client/src/pages, index.css, scripts/visual_check.mjs) + SHARED
  (server/routes.ts, datacore/layers.json, package.json, this file) —
  one logical change end to end, not split across sessions, per
  WORKSTREAM PARTITION rule 5. Checked `list_pull_requests` before
  starting: one open PR (#77, unrelated draft, Tier-2 OOM fix, stale
  since April) — no concurrent T-DATACORE/T-CLIENT session to collide
  with, so no serialization needed.
- SESSION-START CHECK (repair mandate + liveness): `/api/health` on prod
  reads uptime_s climbing normally (1382s, no restart-loop signature),
  heap/rss steady (~55/159MB), bot active, liveness dark:false — the
  2026-07-05 OOM crash-loop (v1.0.143) stayed resolved, no LIVENESS ALARM
  to file. KNOWN BROKEN #4/#10 remain correctly deferred (human-audit-log
  access limitation; shadow_portfolio history not yet deep enough) —
  neither blocks product work, per this session's own directive.
- HIGHEST-VALUE PRODUCT ACTION (chosen over starting a new pipeline or a
  new gate-1 effort): BUILD ORDER 5 shipped five datacore pipelines
  2026-07-05 (FINRA short-volume #1, CFTC COT disaggregated #2, Wikimedia
  attention #3, FAA airport status #4, CBP border waits #5) — all five
  are still API-only, zero /data UI, exactly the earnings-language gap
  the 2026-07-05 PRODUCT session closed for 8-K filings. Of the five,
  FINRA short-volume was picked: its gate-1 (parser sum-of-parts
  identity, 100.000% on n=12,240, logged 2026-07-05) already PASSED, its
  data is non-geospatial (fits the proven insider/earnings inline-panel-
  row + full-view pattern with zero new map-coordinate work, unlike
  FAA/CBP which the modules' own doc comments flag as needing an
  airport/crossing coordinate table first), and — the actual find this
  session — the deep backfill that crash-looped prod on 2026-07-05
  (v1.0.138-140) had ALREADY COMPLETED before the emergency default-off:
  prod's `/api/data/archive/stats` confirms 513 day-files, 2024-06-17 to
  present, 80MB gz, sitting unused with zero UI. That's a 2+-year archive
  no paid vendor undercuts (BUILD-FIRST/accumulation-substitutes-for-
  purchase, EDGE DOCTRINE #1) that this session's job was to surface.
- DOWNSTREAM CHAIN (REASONING STANDARD #1), stated before building: a
  new /data/short-volume route reading the archive -> if it recomputed
  ratios by decompressing the full ~12K-row-per-day archive on every
  request, that repeats the EXACT mistake that caused the 2026-07-05 OOM
  incident (materializing a growing archive on a hot path) -> so the
  design splits in two: (a) a NEW tiny append-only trend log
  (`_summary_history.jsonl`, one {date, agg_short_ratio} line per
  trading day, written at POLL time inside `refreshShortVol` exactly
  where `summarize()` already runs — never on the request path) serves
  the market-wide chart; (b) per-symbol lookback reads the day-archive
  directly but is bounded to <=90 trading days (same cap convention as
  insider/earnings history routes) and materializes ONE day at a time,
  discarding it before the next — bounded peak memory regardless of
  lookback depth, on-demand only (not a periodic poll).
- BUILD: `server/finraShortVolume.ts` gains `appendSummaryHistoryEntry`/
  `readSummaryHistory` (the trend log, idempotent per date — a restart
  that rebuilds the cache from the already-archived newest day must not
  double-append) and `listArchivedDates`/`lookupSymbolHistory` (bounded
  per-symbol read, case-insensitive match, honestly OMITS a date with no
  row for the symbol rather than zero-filling — a delisted/untraded day
  is real absence, not a zero). `server/routes.ts`: new
  `GET /api/data/short-volume/history?days=N[&symbol=X]`, symbol mode
  reads the archive, no-symbol mode reads only the trend log.
  `client/src/pages/shortvol.tsx` (new): market-wide ratio stat + a
  self-contained inline SVG sparkline (no charting dependency added) for
  the trend, a ticker-lookup form rendering that symbol's ratio series
  as a table + sparkline, and today's top-30 ratio-mover table — mirrors
  earnings.tsx's page shape (header/attribution, designed empty/loading/
  error states, `.vt-filings-table`/`.vt-earnings-search` reused rather
  than duplicated). `datamap.tsx`/`layers.json`: new non-geospatial
  `shortvol` layer in the "filings" panel group, identical
  poll-for-panel-count + "Open full view" pattern as insider/earnings
  (300s poll — the underlying data is a once/day batch; polling exists
  to refresh the panel badge, not to chase intraday freshness).
  Registry-driven ZERO-COST-WHEN-OFF and self-see/toggle-consistency
  batteries cover the new layer with no hardcoded id list to maintain.
- HONESTY: every response still carries `kind: "raw"` + the FINRA
  attribution and the flow-proxy-not-short-interest caveat already in
  the live route; the client repeats "no predictive claim" in its own
  header — short-ratio-extremes-precede-reversals stays an open gate-2
  hypothesis, nothing here claims otherwise.
- Tests: `server/finraShortVolume.test.ts` +6 (idempotent append/read
  ascending/bounded-days, listArchivedDates ordering + limit, symbol
  lookup honest-gap + case-insensitivity + unknown-symbol-empty,
  refresh-wiring dedup across both the fresh-fetch and restart-rebuild
  branches — the two call sites `appendSummaryHistoryEntry` was added
  to). `npm run test:node`: 267/267 (was 261 pre-PR once `npm install`
  restored this sandbox's missing `node_modules` — the 3 failures seen
  before installing were `Cannot find package 'express'`, an environment
  gap, not a code defect; confirmed clean before crediting this PR's
  count). `npx tsc --noEmit`: identical pre-existing error set (grepped
  for `finraShortVolume|shortvol|routes.ts` — zero hits, zero new
  errors). `npm run build` clean. `npm run visual --page data`: 0 hard
  failures at 390/768/1440; the one new soft warning ("Filings & flows
  4/4 on" clipped-control) is the same below-the-fold static-screenshot
  false positive already on file for this group (earnings-language
  entry, 2026-07-05) — `toggleConsistency` reads "17 layers toggled
  clean" (was 16) and self-see `failures: []`, i.e. the layer that
  actually matters (reachability or a live desync) shows zero. A
  scratch Playwright script (not committed, deleted after use — same
  precedent as the earnings-language PR) screenshotted
  `#/data/short-volume` directly at all three widths and drove the
  ticker-search interaction end to end (typed "ZZTOP", clicked search,
  confirmed the sparkline + table render its 2-day fixture series,
  confirmed the empty/loading/no-symbols-cleared-the-floor states
  render); zero console/page errors at any width.
  Python suite not touched (no `.py` files in this diff); `pytest` is
  not installed in this sandbox to re-verify, noted honestly per the
  precedent set by the 2026-07-05 COT gate-2 and earnings-language
  entries rather than claimed.
- NOT attempted this session (correctly out of scope, SPINOUT-READY /
  RAW-vs-SIGNAL rule): gate 2 (do short-ratio extremes precede reversals
  in small caps) — this PR is a RAW display of the already-gate-1-passed
  pipeline's archive, no predictive claim. FAA airport status and CBP
  border wait times remain the next BUILD ORDER 5 UI gaps, both
  explicitly needing a coordinate table their own doc comments already
  flag as follow-up work — logged as the next queued PRODUCT item below.
- STARVED: no — this was the single highest-value action available
  (an already-validated pipeline's UI gap, chosen over a fresh gate-1
  effort or new pipeline per SESSION BUDGET's ordering), executed start
  to finish in one PR.

## 2026-07-05 — [REPAIR] /data trail was a static snapshot — live 30s refresh + freshness chip + harness ratchet (v1.0.144) [human-directed, T-CLIENT]

- BUG: selecting an aircraft/vessel/train painted the archived trail
  ONCE and never again — the line froze while the entity kept
  moving. FIX: a refresh effect keyed to the open card's
  trailId/trailKind re-pulls /api/data/track every 30s and UPDATES
  the geojson source via setData (no layer churn/flicker); closing
  or switching cards tears the interval down.
- FRESHNESS HONESTY (the directive's second half): the card now
  shows "last position Xs/Xm ago" from the newest archived point's
  timestamp (10s UI ticker between refreshes), with the explicit
  wording that TRAIL GAPS = coverage/sampling, not necessarily
  staleness — the age chip is what distinguishes the two.
- HARNESS RATCHET (all widths PASS; battery runs at 1440): a
  STATEFUL track fixture returns one more point per call; the
  battery clicks a projected fixture aircraft, requires the
  freshness chip, holds the card open across a full 30s interval,
  and requires BOTH a second track pull AND the trail geometry to
  grow. Static-snapshot regressions fail on both counts.
- TESTABILITY LESSON: maplibre's GeoJSONSource internals
  (_data/serialize) were unreliable reads from the bundle — the
  client now exposes window.__vtTrailLen explicitly; ratchets should
  read declared surfaces, not library internals.
- Visual verification per promotion rule: npm run build first
  (stale-bundle lesson), harness green at 390/768/1440 + all-off +
  scale batteries; screenshots self-reviewed at all three widths —
  no layout regressions (the chip renders only inside an open card).

## 2026-07-05 — [REPAIR] ✅ CRASH-LOOP RESOLVED & VERIFIED — uptime 367->728s climbing (12x the death ceiling), heap 54-59MB steady vs the 509MB death point (v1.0.143 post-deploy)

- Pre-stated protocol executed after #267 deployed: /api/health
  uptime_s read 4x over 6 minutes: 367s -> 487s -> 608s -> 728s —
  monotonic, past the 300s bar, 12x the ~61s OOM ceiling that held
  for 2+ hours. heap_used_mb 54-59 steady with one 168MB working
  spike that GC'd back down (bounded folds behaving as designed);
  rss ~162MB. The trading loop can run sustained work again —
  LIVENESS ALARM CLEARED ~13h before Monday's open.
- Full arc for the record: symptom surfaced by attention's slow
  cycle -> R7 self-reporting stats -> restart metronome measured ->
  backfill emergency-off (exonerated it) -> external diagnostics
  exhausted -> human read Railway logs (OOM at ~509MB) -> boot folds
  rewritten to bounded online aggregation + cgroup-aware heap ->
  verified. Detection now permanent: uptime_s + heap/rss on health,
  memory ratchet test, no-slurp source ratchet.

## 2026-07-05 — [REPAIR] ✅ PRIORITY-1 ROOT CAUSE FIXED — Railway logs (via human) showed Node OOM at ~509MB under the 512 cap; boot archive folds now stream with bounded state + cgroup-aware heap (v1.0.143)

- ROOT CAUSE (from Railway logs, human-read — closing the loop on the
  v1.0.141 alarm): "JavaScript heap out of memory" at ~509MB, ~20s
  after startup, under run_with_daemon.sh's hardcoded
  --max-old-space-size=512. Local repro was clean because local
  lacks prod's data volume.
- THE OFFENDER: readVesselTracksAsync — the R4/R5 "streaming" fix
  streamed for the EVENT LOOP but MATERIALIZED every in-window point
  of every vessel into a Map before analytics. The eager boot
  pollers (shadowstats 72h + portdwell 168h) fed it the vessel
  archive, which grew to ~141MB gz (~7M points, >800MB as JS
  objects) — heap crossed the cap the moment boot folds ran. Same
  "works at day-0 size, dies as archives grow" class as the
  event-loop defect, in the MEMORY dimension. The metronomic ~61s
  was OOM-abort + Railway ALWAYS-restart; the healthcheckTimeout=60
  match was coincidence. The FINRA backfill was an amplifier, not
  the cause (loop survived its default-off — posterior updated as
  pre-stated).
- FIX 1 (stream, don't slurp): new foldVesselArchiveAsync hands each
  parsed point to a callback and retains NOTHING. ShadowAggregator
  (online gap/loiter/identity with per-vessel bounded state: last
  point, first point, name sets, in-zone speed runs only) and
  VisitDetector (the exact detectVisits state machine fed
  point-by-point, retaining only the current in-port run) replace
  the materializing path in BOTH async pipelines. The materializing
  readers remain ONLY for tests; the pre-existing async-vs-sync
  deepEqual ratchets pin the online folds output-identical (13/13).
  Shared aggregateVisits tail = one stats engine, no divergence.
- FIX 2 (heap fits the container): run_with_daemon.sh (FROZEN —
  HUMAN-AUTHORIZED this edit) now reads the cgroup memory limit and
  sets --max-old-space-size to 60% of it (floor 512, cap 6144,
  override VOLTRADE_NODE_HEAP_MB), logging the choice at boot.
- DETECTION RATCHETS: /api/health server check now carries
  heap_used_mb + rss_mb next to uptime_s (growth visible BEFORE the
  cap); NEW server/archiveFoldMemory.test.ts — behavioral (folding a
  synthetic 300K-point archive must retain <30MB heap; a slurp
  measures >40MB) + source (no runtime module may CALL the
  materializing readers; async paths in the two modules asserted
  clean).
- VERIFICATION (pre-stated): after deploy, /api/health uptime_s must
  climb past 300s; then heap_used_mb observed at steady state.
- LESSON (standing candidate): "streaming" has TWO dimensions — a
  reader can yield the event loop and still slurp the heap. Archive
  consumers must state BOTH bounds: compute off the request path AND
  memory bounded in archive size. The scan-class rule (v1.0.125)
  should be read as covering both.

## 2026-07-05 — [REPAIR] Durability audit (human-directed) — every write path swept vs the /data volume; 2 strays (1 migrated, 1 wishlisted); loss report; ratchets both runtimes (v1.0.142)

- VOLUME MOUNT VERIFIED: /data (storage_config.py DATA_DIR ->
  /data/voltrade; server archiveBaseDir() ->
  /data/voltrade/datacore_archive; auth DB /data/voltrade.db).
  Prod /api/data/archive/stats confirms real durable data under the
  mount across 26 archive kinds (~377MB total).
- SWEEP RESULT (both runtimes, every fs write in runtime code):
  DURABLE — all 22 datacore stream archives, options-chain archive,
  API metering, waitlist, auth SQLite, equity/liveness/kill-switch/
  ML state (Python via storage_config, mostly atomic tmp+rename).
  EPHEMERAL-BY-DESIGN — /tmp API caches, bot.ts spawn-IPC scratch,
  daemon trace, backtest .bt_cache (all justified, classified in the
  new ratchets). STRAYS — exactly 2:
  (1) intraday_shorts.py:41 resolved its "persistent" short-trade
      log via a bare DATA_DIR env read (unset on Railway) -> /tmp;
      the log reset on EVERY redeploy since the module shipped.
      MIGRATED this PR to storage_config.DATA_DIR (env override
      still honored). Loss: all intraday-short trade history to
      date (the module itself noted "not enough closed trades to
      learn from" — the loss is real but small).
  (2) server/billing.ts:34 (FROZEN) writes its SQLite to the image
      dir — ephemeral AND a different file from the auth DB whose
      users/sessions tables it updates (split-brain). Billing is
      dark (no Stripe key) so zero customer data lost; exact
      one-line amendment filed at the TOP of wishlist.md for the
      human; the Node ratchet pins the stray signature so fixing it
      forces the allowlist to shrink.
- HISTORICAL LOSS REPORT (the honest part): the volume's file
  history for continuously-recording streams begins 2026-07-03
  ~18:00Z (aircraft/vessels raw both start exactly there; trains
  2026-07-04-01; all _tracks rollup dirs empty because no raw file
  has crossed the 7-day threshold yet — first rollup due ~07-10, NOT
  a pruning defect). Everything continuously recorded before that
  moment went to the pre-attach fallback and was lost to redeploys:
  ~4-5 days of aircraft/vessel/train positions from stream launch
  (late June) — a PERMANENT gap in the compounding position archive;
  EDGAR-family archives pre-07-03 (refetchable in principle);
  Python trading state resets pre-attach. Everything since attach is
  durable except the two strays above.
- INCIDENTAL FINDINGS from the same stats read: the FINRA deep
  backfill actually COMPLETED during the crash-loop windows — 513
  day-files back to 2024-06-17, all gz (80MB), marker present. The
  ratcheting-resume design worked under fire; disk footprint is 80MB
  not 750MB, which further weakens the volume-full crash theory
  (loop cause still pending the Railway dashboard read, wishlist
  #8). Also: 6h/30min maintenance timers (rollup, compress) never
  fire while process lives are ~61s — they catch up after the loop
  resolves.
- RATCHETS (no future stream can ship ephemeral): NEW
  server/durability.test.ts — every fs-writing server module must be
  durable-marked (archiveBaseDir//data/DATA_DIR), classified
  tmp-by-design with justification, or on the wishlisted-stray list
  (which is pinned to exactly billing.ts and asserts its signature
  so a fix tightens the ratchet); plus the archiveBaseDir mount
  contract. NEW test_durability.py — intraday_shorts path pinned
  under storage_config.DATA_DIR; no runtime module may bare-env-read
  DATA_DIR with a /tmp default without importing storage_config;
  storage_config mount contract. Both green; full suites green.

## 2026-07-05 — [REPAIR] ⚠️ TOP-OF-REPORT LIVENESS ALARM: prod restart loop CONTINUES after backfill-off — diagnosis blocked on Railway dashboard access; health now exposes uptime (v1.0.141)

- ⚠️ STATE AT FILING (~21:50Z): the container restarts every ~61s and
  has since AT LEAST ~20:00Z (possibly earlier — restarts are
  invisible behind instant-refill caches). Every endpoint answers
  (each 61s life is enough to serve), archives still record in
  bursts, and it is Sunday — but the trading loop CANNOT run
  sustained work. If unresolved by Monday 09:30 ET this is the
  constitutional LIVENESS ALARM. Mike push-notified at ~21:45Z.
- EVIDENCE CHAIN (all external — prod logs unreadable from a
  session): (1) attention last_cycle timestamps = new boot every
  61-63s, metronomic; (2) INDEPENDENT confirmation: COT + FAA cache
  ages always <=15s despite 12h/15min refresh intervals (their
  cache.at only stamps at fill); (3) v1.0.140's backfill default-off
  deployed and the cadence did NOT change -> the deep backfill is
  EXONERATED as the driver (v1.0.140's guards remain correct
  hardening regardless); (4) the prod bundle run LOCALLY under the
  same node --max-old-space-size=512 survives 150s cleanly (2.5x the
  prod death period) with graceful 401 handling -> the trigger is
  ENV-DEPENDENT (prod keys, volume contents, or platform); (5) no
  ~60s timer exists in server code (grep); railway.json has
  healthcheckPath /api/health with healthcheckTimeout 60 and
  restartPolicy ALWAYS — the 61s period matches the healthcheck
  timeout exactly, but external /api/health probes return 200, so
  healthcheck-kill vs container-OOM (node 512MB heap cap + Python
  daemon RSS vs the Railway plan limit) CANNOT be discriminated from
  outside. (6) python-side system-status: 7/7 checks pass.
- ONSET BOUND: restarts confirmed by 20:22Z (attention could not
  fill 24 min after its deploy); earliest possible onset unknown —
  any deploy today could have crossed a boot-time threshold (data
  growth on the volume slowing daemon/boot past 60s is a live
  theory that needs the dashboard to confirm).
- SHIPPED THIS PR (detection, not another blind patch — recurrence
  rule respected): /api/health checks.server.uptime_s. A restart
  loop was INVISIBLE: every probe hit a young-but-alive process
  reading "ok". Repeated low uptime reads ARE the loop signal; the
  DAILY health check must alarm on uptime_s staying <120 across
  reads minutes apart.
- BLOCKED-FOR-MIKE (URGENT, wishlist #8): read the Railway
  dashboard deploy logs + restart reason + memory graph. That
  single read discriminates every remaining theory: healthcheck
  timeout (logs say "healthcheck failed") vs OOM (memory graph
  spikes to plan limit) vs crash (stack trace in deploy logs). If
  healthcheck: likely boot path now exceeds 60s (daemon start +
  data growth) — remedies are healthcheckTimeout bump (frozen
  railway.json -> Mike edits) or boot-path lightening. If OOM:
  container plan limit vs node 512 + daemon RSS — remedies are plan
  bump or heap-cap tuning in run_with_daemon.sh (frozen -> Mike).
  If crash: the stack names the module and the next session fixes
  the actual defect.
- NEXT-SESSION VERIFICATION PROTOCOL: read /api/health uptime_s 3x
  two minutes apart. All <120s = loop persists (work the Mike
  findings); rising past 300s = loop over (then find what changed
  and write the close-out).

## 2026-07-05 — [REPAIR] PRIORITY-1: prod crash-looping ~every 60s after the deep backfill — emergency default-off + gz-on-write (v1.0.140)

- DETECTION CHAIN (the R7 diagnostics paid for themselves within the
  hour): attention's last_cycle timestamps exposed NEW BOOTS at
  20:48, 20:54, 20:55, 21:02, 21:03, 21:05 — a restart roughly every
  minute, with no merges after 20:43. Every other stream hides
  restarts (instant archive-rebuild caches); attention's slow cycle
  made them visible. /api/health reads ok BETWEEN crashes (bot
  active, liveness dark:false — and it is Sunday, markets closed).
- LEADING THEORY (stated as theory — prod logs unreadable from a
  session): v1.0.138's deep backfill writes ~750MB of UN-GZIPPED
  day-files per pass (gzip only ran after a COMPLETE pass, and no
  pass completed) — plausibly filling the Railway volume; a full
  volume makes the bot's periodic state writes (equity curve,
  SQLite, audit) throw ~60s after each boot -> crash -> restart ->
  backfill resumes writing -> loop that does NOT self-heal. The
  original design DID consider memory (one file at a time) and the
  event loop (async gz) but NOT peak disk of an incomplete pass —
  the pre-ship "downstream chain" trace missed the disk dimension.
- EMERGENCY FIX (one PR, three guards): (1) deep backfill now
  DEFAULT-OFF behind FINRA_DEEP_BACKFILL=1 — prod reverts to
  pre-backfill behavior on deploy regardless of which theory is
  right; (2) gz-on-write — when a deep pass runs, each gz-eligible
  day compresses THE MOMENT it lands (~75MB total, no full-pass
  gap); (3) .catch on the boot chain (an unhandled rejection there
  would itself crash the process). SELF-CLEANING: the regular 6h
  refresh already sweeps accumulated plain files to gz on the next
  stable boot — no manual volume surgery needed.
- VERIFICATION PLAN (after this deploys): attention last_cycle
  timestamps must stop advancing across a 5-min watch (restarts
  over), then attention should serve within ~2 min. If restarts
  CONTINUE, the backfill is exonerated and diagnosis moves to what
  else changed (#263 window) — stated now so the posterior updates
  honestly either way.
- RE-ENABLE PATH: verify Railway volume capacity/usage (human can
  read it in the Railway dashboard; or expose a disk-stats surface),
  then set FINRA_DEEP_BACKFILL=1 — gz-on-write caps the pass at
  ~75MB. Backfill remains the right idea; the flooding write pattern
  was the defect.
- RECURRENCE DISCIPLINE: this is the FIRST failure of this subsystem
  (fix #1). If the crash loop persists after this ships, the next
  session's patch attempt is FORBIDDEN — root-cause analysis per
  HEALTH OF THE LOOP rule 4.
- Tests 8/8 (default-off pinned: zero fetches without the env;
  gz-on-write pinned: deep-pass days land compressed); tsc 64
  baseline; pytest 410/1.

## 2026-07-05 — [REPAIR] attention stream warming_up on prod — self-reporting cycle stats shipped, diagnosis reads from the route (v1.0.139)

- SYMPTOM: /api/data/attention stayed warming_up across MULTIPLE
  clean deploys while every sibling stream (short-volume, COT,
  imports, airport-status, border-waits) served — the deploy-churn
  theory was tested and killed by waiting out a full quiet window.
- CONSTRAINT: prod logs are not readable from a session (DIAG token
  not in session env), so the fix is to make the stream
  SELF-REPORTING: fetchAttention now records CycleStats per poll
  (started/finished, ok/err/not_found article counts, last_error as
  "TICKER -> status" — no URLs, nothing sensitive) and the route
  includes last_cycle in the warming payload. The next probe reads
  the diagnosis from prod itself.
- SUSPECTS (to be discriminated by the payload): Wikimedia
  403/429-blocking Railway's egress (last_error would show the
  status), all-timeout cycles (err count = 23 with timeout
  messages), or a not-yet-understood parse mismatch (ok high but
  obs 0). Test pins the all-403 failure mode end-to-end.
- PATTERN NOTE (candidate standing rule if this recurs elsewhere):
  a warming_up that can persist forever is only honest if the route
  can also say WHY — streams whose pollers can fail silently should
  expose last-cycle stats on the warming path. Not yet filed as a
  rule; one data point.
- Tests 6/6 (2 new: cycle-stats accounting on the mixed-404 path,
  all-403 failure capture); tsc 64 baseline; pytest 410/1 (count
  grew on main — other sessions' tests).
- VERIFY (next probe after deploy): /api/data/attention while
  warming should now carry last_cycle; expected resolutions —
  Wikimedia blocks Railway => wishlist/adapter decision; timeouts
  => raise AbortSignal or reduce window; transient => it will have
  simply filled.

## 2026-07-05 — [PIPELINE] FINRA short-volume deep backfill — ~2y history self-fills on prod (v1.0.138)

- TERRITORY: T-DATACORE. Follows gate 1 (passed, entry below):
  dated CNMS files persist for years, so history someone else
  recorded is history we can still capture — accumulation
  substitutes for purchase. Sessions are ephemeral; the honest home
  for ~500 trading days (~75MB gz) is the PROD VOLUME via the stream
  itself: bootShortVolPoll now runs a one-shot 750-calendar-day
  backfill AFTER the normal 7-day refresh (route serves current data
  within seconds; history fills in the background, ~4 min at 300ms
  spacing, one file in memory at a time).
- COMPLETION-MARKER SEMANTICS (a count trigger was designed and
  REJECTED mid-build): a threshold like "skip if >=30 day-files"
  would mistake an INTERRUPTED backfill for a finished one. Instead
  a completed pass writes backfill_done.json (with rt + file count);
  an interrupted pass leaves no marker, so the next boot re-runs and
  date-level dedup resumes it for free. Test pins all three states
  (complete -> marker; marker -> zero refetches; marker removed ->
  idempotent resume).
- EVENT-LOOP GUARD (shadowstats class, caught at design time): the
  post-backfill gzip sweep would gzipSync ~500 x 1.5MB files in one
  synchronous loop — tens of seconds of blocked loop. The poller
  path now uses gzipOldShortVolDaysAsync (yields via setImmediate
  between files); the sync variant remains for tests.
- TEST LESSON (small): archivedDates is module state shared across
  a test file's cases by design (date-level dedup) — fixtures must
  use disjoint dates; the backfill test initially collided with an
  earlier case's 2026-07-02 and read 2/3.
- Tests 8/8; tsc 64 baseline; pytest 397/1. Next on this root:
  verify backfill_done.json on prod after deploy (day_files should
  read ~490-510), then pre-state the gate-2 design before scoring.

## 2026-07-05 — [PIPELINE] FINRA short-volume GATE 1 PASSED — 100.000% sum-of-parts identity, n=12,240 (scripts/finra_gate1.ts, session-run)

- LADDER GATE 1 (DATA layer) for the v1.0.133 stream, run the same
  day it shipped. CRITERIA PRE-STATED before running (in the script
  header, committed): sample day 2026-07-02; truth = FINRA's four
  independent per-facility files (FNYX=N NYSE TRF, FNSQ=Q Nasdaq
  Carteret, FNQC=B Nasdaq Chicago, FORF=O ORF) parsed with the SAME
  parser under test and summed per symbol over exactly the
  facilities each CNMS row's own Market column names; PASS bar
  >=99.9% within 0.01 shares on BOTH short_vol and total_vol;
  unknown market codes reported, never silently skipped. PRIOR:
  ~90% (accounting identity FINRA itself maintains).
- RESULT: checked=12,240 (every CNMS symbol), matched=12,240 —
  100.000%. Zero unknown market codes (the four-file map covers the
  entire consolidated tape), zero mismatches at 0.01-share
  tolerance including the fractional-share rows. Facility coverage
  observed: FNSQ 12,196 / FNYX 7,917 / FNQC 5,541 / FORF 3,048
  symbols.
- MEANING (and its limits): our parse of FINRA's published files is
  faithful — layer 1 verified. This says NOTHING about the signal
  (gate 2: short-ratio extremes vs forward returns needs archive
  depth) and does not upgrade the flow-proxy caveat: this is
  short-marked execution volume, not short interest.
- Next on this root: 1-2y session-side backfill (dated files
  persist), then gate-2 design pre-stated before any scoring.

## 2026-07-05 — [RESEARCH] BUILD ORDER 5 CLOSED — 5/5 buildable items shipped same-day; USPTO probe verdict filed as BLOCKED-FOR-MIKE #7 (docs)

- SCOREBOARD (filed #253 and closed the same day): #1 FINRA short
  volume v1.0.133 (LIVE on prod, 12,240 symbols), #2 CFTC COT
  v1.0.134 (LIVE, 274 markets), #3 Wikimedia attention v1.0.135,
  #4 FAA airport status v1.0.136, #5 CBP border waits v1.0.137.
  Five new always-on archives; every hypothesis gate-locked with
  priors stated at filing; map layers for #4/#5 deferred honestly
  (need coordinate tables — filed as next slices).
- #6 USPTO probe (per its probe-first instruction): PatentsView has
  required a free API key since 2021 (legacy keyless endpoint 301s
  away; the request form is human-facing) — BLOCKED-FOR-MIKE #7
  filed with the free-alternative analysis (weekly bulk XML,
  keyless but heavy). ALSO: search.patentsview.org 502s through the
  session proxy and developer.uspto.gov 503s — first build session
  after the key re-probes reachability (Railway-side may differ;
  the key-gated pattern covers either).
- Probe-discipline lessons this build order (now standing): a
  status-200 probe is not a BODY probe (CBP's RSS path served an
  HTML SPA); named-field sources beat headerless positional files
  even when the flat file probed fine (COT); per-item live probes
  before seeding curated maps (RIOT's renamed article).
- GATE-1 QUEUE created by this build order: FINRA parsed ratios vs
  FINRA monthly aggregates (sampled month); attention views vs
  known events on 10 tickers (~2 weeks of archive); COT archived
  weeks vs CFTC historical annual files (when depth allows).

## 2026-07-05 — [PIPELINE] CBP border wait times stream (BUILD ORDER 5 #5) — change-only archiver + /api/data/border-waits (v1.0.137)

- TERRITORY: T-DATACORE. server/cbpBorderWait.ts polls
  bwt.cbp.gov/api/bwtnew (keyless JSON, 83 land crossings) hourly.
  Completes the five probed-and-buildable BUILD ORDER 5 items; #6
  USPTO stays probe-first for a later session. Fills the ROAD leg of
  the freight-proxy set (sea=AIS, rail=STB, air=ADS-B).
- PROBE LESSON (new standing caution): the RSS path recorded at
  filing (bwtRss/HTML) returned 200 — but its BODY is an HTML SPA.
  A status-only probe is not a body probe; filing-time probes should
  capture the first bytes too. bwtnew is the real JSON feed.
- LOCALE HONESTY: the API localizes STATUS STRINGS by serving region
  — our probe egress received Spanish ("Abierto", "demora"); prod's
  US egress may receive English. Parsing keys ONLY on
  locale-independent fields (numeric delay_minutes/lanes_open, port
  identifiers, structure); localized strings archive VERBATIM,
  never translated or matched (test fixture uses the actual Spanish
  capture). Joins use port_number.
- CHANGE-ONLY ARCHIVE: dedup key = port|crossing|lane|values —
  unchanged waits re-polled hourly write nothing; every published
  change appends (test-pinned). Flattened per lane class
  (commercial standard/FAST + passenger standard); a crossing
  without commercial lanes contributes no commercial rows — real,
  not missing. Null delay = not published, never zero.
- Map layer deferred with the FAA one (needs a port-of-entry
  coordinate table).
- HYPOTHESIS (gate-locked): sustained commercial wait anomalies at
  Laredo/Otay Mesa/El Paso lead border-dependent logistics and rail
  intermodal volumes (joins STB carloads). Prior ~20% (indirect
  transmission) — archive-first, stated at filing.
- Tests 4/4 (live-shape parse with Spanish-capture fixture,
  absent-lane honesty, change-only dedup + delay-change append,
  transport-error keeps last snapshot); manifest battery 3/3; tsc 64
  baseline; pytest 397/1.

## 2026-07-05 — [PIPELINE] FAA airport status stream (BUILD ORDER 5 #4) — ground stops/GDPs/delays/closures + /api/data/airport-status (v1.0.136)

- TERRITORY: T-DATACORE. server/faaStatus.ts polls the keyless FAA
  NAS status XML every 15 min (delay programs are intraday — hourly
  would miss churn; 96 light requests/day). Shape verified live on a
  thunderstorm day (ground stops DCA/BWI, GDPs JFK/LGA/PHL/EWR) —
  the test fixture is trimmed from that real capture.
- PARSER LESSON (new, not in the EDGAR precedent): regex tag
  extraction per edgarForm4.ts, but this document has PREFIX-
  COLLIDING tag names — <Delay> vs <Delay_type> — and `<${tag}[^>]*>`
  matches both. Caught by the battery (delay event came back as DCA
  instead of LGA); fixed with a name-boundary regex
  (`<${tag}(?:\s[^>]*)?>`). Any future regex-XML stream should start
  from the boundary-safe helpers.
- SNAPSHOT ARCHIVE SEMANTICS: the feed is a rolling snapshot with no
  clear/cancel times. Dedup by EVENT IDENTITY (type|airport|reason|
  numbers): a persisting program archives once; changed numbers
  append vintage-style (test: JFK GDP worsening appends exactly 1).
  Durations are therefore LOWER BOUNDS from our capture times —
  stated in the manifest, never inferred. Published delay values stay
  human-readable strings at capture ("2 hours and 30 minutes");
  normalization is analysis-time work so the raw record stays
  faithful.
- EMPTY-NAS HONESTY: zero programs nationwide is a real publishable
  state — cached and served as such (test-pinned); transport errors
  keep the last snapshot instead of faking a clear sky.
- Map layer DEFERRED honestly: the feed has no coordinates; a
  layer needs an airport code->lat/lon table — filed as the natural
  next slice, not half-shipped.
- HYPOTHESIS (gate-locked): sustained delay-program frequency at
  cargo hubs (MEM/SDF/CVG/ANC) as a parcel-carrier cost-pressure
  indicator. Prior ~20% (weather dominates and is priced); the
  archive is cheap honest RAW value regardless.
- Tests 4/4 (live-shape parse all four families + prefix-collision
  regression, empty-NAS + transport-error semantics, event-identity
  dedup + changed-number append, 403/500 fetch semantics); manifest
  battery 3/3; tsc 64 baseline; pytest 397/1.
- PROD VERIFIED same window (B5 running tally): short-volume LIVE
  (12,240 symbols, 2026-07-02, holiday lookback correct), COT LIVE
  (274 markets, report 2026-06-23), attention warming at last probe
  (poller restarts with each deploy; next DAILY session confirms).

## 2026-07-05 — [PIPELINE] Wikimedia pageviews attention stream (BUILD ORDER 5 #3) — curated seed + /api/data/attention (v1.0.135)

- TERRITORY: T-DATACORE. The PYTRENDS REPLACEMENT (gate-1 FAIL #215):
  server/wikiAttention.ts polls the keyless Wikimedia pageviews REST
  API (en.wikipedia, all-access, agent=user — bot traffic excluded at
  the source) for a curated 23-ticker seed, 7-day window, 12h poll.
- CURATION HONESTY: datacore/wiki_articles.json — EVERY pair was
  hand-probed against the live API before inclusion (24 probed, 23
  kept). RIOT dropped at curation: the article was renamed and the
  pageviews API does not follow redirects (both candidate titles
  failed — one 404, one valid-but-no-data). Expansion rule embedded
  in the file: no new pair without a passing probe; true small caps
  often lack articles and that ABSENCE IS DATA. Observed live: quick
  bursts 429 — poller spaces requests >=600ms (test asserts the
  constant respects the observed limit).
- Docker image rule: the seed is a STATIC IMPORT (bundled) — the
  frozen Dockerfile never copies datacore/, so a runtime disk read
  would serve nothing in prod (entity-spine lesson #226).
- PANEL DISCIPLINE: the served day is the newest with a MAJORITY of
  the seed present, so an in-progress publish day (2 of 23 articles)
  never masquerades as the panel (test-pinned). v1 serves RAW daily
  views only — no z-scores or spike labels until the archive holds
  the trailing history to compute them honestly AND gate 1 passes.
- GATE 1 (pre-stated): on 10 hand-checked tickers, views series must
  spike on known event dates (earnings, major announcements) vs the
  surrounding baseline; article-identity errors (wrong company) are
  an automatic fail for that pair. Run after ~2 weeks of archive.
- HYPOTHESIS (gate-locked): attention spikes lead volume/vol 1-5d,
  most interesting on smaller names without same-day news
  (attention-without-news subset). Prior ~30% stated at filing.
- Tests 5/5 (seed bundling + RIOT-absent honesty, API-shape parse,
  one-request-per-article + 404-absence, dedup by view day + 4d gz
  with corrected fixture arithmetic, majority panel-day rule);
  manifest battery 3/3; tsc 64 baseline; pytest 397/1.

## 2026-07-05 — [PIPELINE] CFTC COT disaggregated stream (BUILD ORDER 5 #2) — keyless Socrata archiver + /api/data/cot (v1.0.134)

- TERRITORY: T-DATACORE. server/cftcCot.ts polls the CFTC Public
  Reporting Socrata dataset 72hh-3qpy (disaggregated futures-only,
  ~274 markets/week, Tuesday as-of / Friday ~15:30 ET publish).
- SOURCE CHOICE WORTH REMEMBERING: the build order named the
  f_disagg.txt flat file (probed 200/442KB), but inspection showed
  it is HEADERLESS positional CSV — parsing ~70 columns by position
  is exactly the guess the query-shape honesty rule forbids. The
  Socrata endpoint serves the SAME data keyless with NAMED fields;
  built against that instead. The names have real quirks, verified
  live and encoded in a FIELD constant with a comment each:
  swap__positions_short_all / swap__positions_spread_all carry a
  DOUBLE underscore; several fields drop the _all suffix
  (prod_merc_positions_long, m_money_positions_spread). A test
  fixture mirrors the quirky shape so a silent source rename fails
  loudly.
- WEEK DISCIPLINE: a DESC-ordered fetch can straddle two report
  weeks at the publish boundary — parseCot keeps ONLY the newest
  report_date so vintages never mix in one archive file
  (test-pinned). Week-level dedup; gz after 9 days (a report stays
  plain until superseded); restart rebuilds the cache from the
  newest archived week even with the fetch down (test-pinned).
- Route /api/data/cot serves the poller's cached week (274 rows,
  event-loop rule) with the futures-ONLY caveat and the honest note
  that positioning-extreme signals need trailing history the archive
  is only beginning to accumulate — accumulation substitutes for
  purchase (vendors sell exactly this series recorded over time).
- HYPOTHESIS (gate-locked): managed-money net-positioning extremes
  (percentile vs trailing history) mean-revert in commodity-linked
  ETFs; joins EIA petroleum/natgas + tank-fill work. Prior ~30%
  stated at filing. Gate 1 design when history depth allows:
  archived weeks vs CFTC's own historical annual files on a sampled
  quarter.
- Tests 5/5 (quirky-name fixture, ''->null, week-boundary keep-only-
  newest, week dedup + gz + gz readback, restart-rebuild with fetch
  down); manifest battery 3/3; tsc at the 64 baseline; pytest 397/1.

## 2026-07-05 — [PIPELINE] FINRA daily short-sale volume stream (BUILD ORDER 5 #1) — keyless CNMS archiver + /api/data/short-volume (v1.0.133)

- TERRITORY: T-DATACORE. server/finraShortVolume.ts: keyless daily
  CNMS file (~12.2K symbols/trading day, format verified live —
  pipe header, fractional share counts, bare row-count trailer;
  weekend/holiday URLs 403 = valid not-published). 6h poll with
  7-day lookback newest-first; eager boot (KNOWN BROKEN #9).
- TWO DESIGN DECISIONS WORTH REMEMBERING: (1) DEDUP IS DATE-LEVEL,
  not per-row — the file is atomic and final once published, and
  seeding 12K keys/day x 40d would waste ~50MB in the RSS-capped
  process; if FINRA reposts a corrected file we keep the first
  capture (stated in the manifest, not hidden). (2) TRAILER
  INTEGRITY GATE — the file's own row-count trailer must equal
  parsed rows or the whole file is refused, so a truncated CDN
  download can never poison the archive.
- RESTART HONESTY: on boot with the newest day already on disk, the
  summary cache rebuilds FROM the archive instead of serving
  warming_up until the next publish (test-pinned, no refetch).
- Route serves the poller's cached day summary ONLY (event-loop
  rule): aggregate short ratio + top-30 by ratio with a stated
  500K-share total-volume floor. LABEL HONESTY: this is short-marked
  EXECUTION volume (flow proxy), NOT short interest — route note +
  manifest confidence_model both say so explicitly.
- HYPOTHESIS (gate-locked): small-cap short-ratio extremes/deltas x
  13F+Form4 joins = squeeze-candidate screen. Prior ~35% (stated at
  filing). GATE 1 next: parsed ratios vs FINRA's own monthly
  aggregates on a sampled month; 1-2y session-side backfill after
  gate 1.
- Tests 6/6 (real-format fixture incl. trailer guard + truncation
  refusal, 403-vs-500 semantics, date dedup + gz + gz-readback,
  summary floor/cap honesty, restart-rebuild-no-refetch); manifest
  battery 3/3; pytest 397/1 skip. tsc NOTE: baseline is now 64 on
  main itself (was 63; the +1 is client/src/pages/datamap.tsx:2143
  from another session's merge, verified by stashing my changes) —
  this change adds zero new errors.

## 2026-07-05 — [RESEARCH] BUILD ORDER 5 filed — new roots at microstructure + attention + freight friction, all sources probed live first (docs)

- Standing directive: T-DATACORE queue emptied (BUILD ORDER 4
  resolved for this territory; remainders wait on external clocks or
  belong to T-BOT/T-CLIENT), so the session generated the next build
  order itself. Full text with per-item hypotheses, priors, and gate
  designs in open_questions.md BUILD ORDER 5.
- PROBED BEFORE FILING (all keyless, HTTP status + size recorded):
  FINRA daily short-sale volume CNMS file 200/540KB; CFTC COT
  disaggregated f_disagg.txt 200/442KB (legacy deacot.txt 404 — the
  disaggregated report is the build target); Wikimedia pageviews API
  200 with real daily counts (Nvidia article, June 2026); FAA
  airport-status API 200; CBP border-wait API 200. USPTO deliberately
  filed unprobed-last with a probe-first instruction (may be
  key-gated → Census/BLOCKED-FOR-MIKE precedent, never build blind).
- Order rationale: (1) FINRA short volume first — every-equity
  coverage including the small-cap tail (EDGE DOCTRINE #2), joins
  13F+Form4 for a free squeeze-candidate screen, and dated files
  allow 1-2y session-side backfill; (2) CFTC COT — accumulation
  substitutes for purchase (vendors sell exactly this file recorded
  over time), named in the EDGE DOCTRINE since day one; (3)
  Wikimedia pageviews — the pytrends replacement after its gate-1
  FAIL (#215); (4)/(5) FAA + CBP — cheap honest RAW layers that
  close the air-ops and road-freight gaps; (6) USPTO last.
- Priors stated per item (30-35% for the signal-bearing roots, 20%
  for the friction archives) BEFORE any data is scored, per
  Reasoning Standard #10; every hypothesis enters the ROOT
  VALIDATION LADDER before belief or surfacing.

## 2026-07-05 — [PIPELINE] Census imports VERIFIED LIVE on prod — 686 records, first query variant correct, key confirmed in Railway (docs)

- ~30 min after #249 merged, /api/data/imports served 686 records:
  April 2026 port-level import values with cnt_val/cnt_wgt POPULATED
  — the first QUERY_VARIANT (full containerized set) was correct;
  the anticipated fix-shape-from-prod-logs path was never needed.
  LIVE VERIFICATION PENDING → COMPLETE, ahead of the planned
  first-DAILY-session check.
- Key location resolved: prod answered warming_up (not
  enabled:false) on the very first probe → CENSUS_API_KEY IS in
  Railway. A transient 502 mid-verification was #250's redeploy
  restarting the app, not a stream defect (health 200 immediately
  after; data followed).
- Data note: Census publishes a national aggregate row (port "-",
  "TOTAL FOR ALL PORTS") alongside per-port rows; the header-driven
  parse keeps it as published — consumers filtering to real ports
  should exclude port "-". Kept raw per the as-published discipline.
- Wishlist BLOCKED-FOR-MIKE #6 → DONE-VERIFIED. Ladder status:
  archive accumulating; gate 1 (readings vs a second official
  source) still to be designed before any signal claim.

## 2026-07-05 — [RESEARCH] Staleness audit Python-side pass COMPLETE — deps clean, 6 undeclared session deps filed, 1 dead env write removed, vacuous-pass sweep judged (docs+tests, no runtime change)

- Closes the register's UNSCANNED half (Python deps/config +
  requirements-vs-imports), fanned out to two subagents with judgment
  retained in the parent per WORKSTREAM PARTITION.
- DEPS: requirements.txt has ZERO unused entries (every package
  imported; pytrends stays per its documented re-probe trigger;
  anthropic conditional behind ANTHROPIC_API_KEY in alphadesk).
  Reverse direction found 6 imported-but-undeclared packages — all
  session-run scripts/ or test tooling, none on any runtime path:
  pytest, xlrd, openpyxl, tifffile, rasterio, Pillow. Filed in NEW
  requirements-dev.txt (Dockerfile is frozen and installs
  requirements.txt only — runtime image unchanged by design).
- ENV VARS (Python side): full inventory taken; every var read at a
  live call site EXCEPT VOLTRADE_STATE_DIR — written once in
  test_patches_verification.py ("if storage_config respects it") and
  read NOWHERE. Worse than dead: the comment claimed a tempdir
  sandbox protected production state, but the write was a no-op, so
  peak-equity writes were hitting the real local state dir all along
  (harmless — the assertions are >= ratchets). Removed the dead
  write + false comment + now-unused tempfile import; honest NOTE
  left in place. Two parallel data-dir vars noted (DATA_DIR and
  VOLTRADE_DATA_DIR both live) — mild inconsistency, not debt; no
  action.
- VACUOUS-PASS SWEEP (queued by the R6 lesson): 26 test files
  swept. Read-before-write review DOWNGRADED the subagent's two
  loudest findings: t_vxx_panic_setup (test_full_system.py) asserts
  the hours-gate itself in its outside-hours branch (PASS only on
  None, WARN otherwise) — deliberate two-mode live harness, NOT the
  TestFix7 class; the two test_diagnostic_false_positives.py
  methods are negative-claim tests where an empty match set is the
  correct pass, with a sibling pinning the source-level invariant
  unconditionally. One real (low) fix shipped:
  test_voltrade_daemon.py::test_every_local_route_resolves gained
  the same checked>0 zero-iteration guard its sibling already had.
  No AssertionError-swallowing anywhere.
- No version bump: docs + test files + a dev-only requirements file;
  nothing in the runtime image changes.

## 2026-07-05 — [PIPELINE] Everything Graph R5 step 1 — datacore/entity_map.json (operator→ticker table), unblocks the flagship graph build + fusion (b) gate 1 (v1.0.131) [T-DATACORE]

- [T-DATACORE] Territory: datacore/entity_map.json (new),
  scripts/build_entity_map.py (new), server/entityMap.test.ts (new) —
  per WORKSTREAM PARTITION. SHARED files touched minimally, last:
  package.json (version bump), research/open_questions.md +
  EVERYTHING_GRAPH.md (status update), this entry.
- SESSION START per MEMORY PROTOCOL: read CLAUDE.md, this file (tail),
  open_questions.md (full), wishlist.md (full). Loop-health ratio over
  the last 10 entries: 3 PRODUCT/PIPELINE-class, 2 RULE-REVIEW, 2
  RESEARCH, 2 REPAIR-class among the DATACORE DEFECT QUEUE closure —
  well under the 7/10 REPAIR thrash threshold. `git`/GitHub state
  checked directly (list_commits on `main` via the GitHub API, not
  just local `git fetch`, after the prior session's own note about a
  stale local ref cache): branch head 99921a6 already equals origin
  main's head — this session starts clean, no reset needed. KNOWN
  BROKEN: no unresolved trading-loop-threatening item found in
  open_questions.md's KNOWN BROKEN section (items 1-2, 5-11 resolved;
  item 3 CSP cascade and item 4 general-health remain ACCESS-LIMITED
  verification gaps unchanged from prior sessions, not new breaks, and
  per the task framing a [PRODUCT] session does not preempt DAILY's
  repair duty for them — noted, not chased this session).
- PRIMARY ACTION SELECTION: surveyed BUILD ORDER 4 (items 1-2 shipped
  today already, item 3 international registries deferred pending a
  fresh per-country session, item 4 blocked until ~2026-09-27, item 5
  blocked until 2026-07-06 close, item 6 counterfactual logger already
  extended this same day in v1.0.130) — every BUILD ORDER 4 item was
  either done or genuinely time/access-blocked. Went to the GIP BUILD
  QUEUE / MAP V2 ROADMAP R5 (THE EVERYTHING GRAPH, explicitly named
  "flagship" by the 2026-07-04 charter directive) and found its own
  design doc (datacore/EVERYTHING_GRAPH.md) names an exact, unbuilt,
  fully-specified next step: build-plan item 1,
  `datacore/entity_map.json`. Confirmed via `ls`/`grep` that no such
  file existed yet and that today's earlier aircraft
  registrant→operator work (v1.0.127, `datacore/aircraft/entity_spine.json`)
  is a DIFFERENT table for a different roadmap item (BUILD ORDER 4 #1,
  aircraft tail→operating-airline resolution) — confusingly similar
  name, distinct scope; did not double-build.
- WHAT WAS BUILT: `scripts/build_entity_map.py` reads the exact
  `operator` strings from `datacore/sites/strategic_sites.json` (13,
  all sites) and the exact `owner` strings from the top-100-by-
  capacity_mw plants in `datacore/powerplants/us_power_plants.json`
  (56) — 69 total distinct registry strings, asserts every one has a
  researched entry (fails loudly if a source registry changes and
  introduces a new unresearched string), and emits
  `datacore/entity_map.json` with `{operator, ticker, confidence,
  parent, note}` per REASONING STANDARD #10 discipline stated in the
  design doc ("confidence: exact-name match (high) / alias match
  (medium) / manual research"). RESEARCH METHOD (REFERENCE DATA
  ACCURACY rule): every entry is backed by a live WebSearch this
  session against SEC filings, company investor-relations pages, or
  primary-sourced summaries — not recalled from training data alone.
  This mattered: regulated-utility subsidiary structures (PG&E→PCG,
  Duke Energy Carolinas/Progress/Florida/Indiana→DUK, Southern Company
  subsidiaries→SO, AEP/Entergy/Dominion/PPL/Ameren/Xcel/NiSource/
  FirstEnergy/CMS/DTE/Pinnacle West/IDACORP-class operating companies,
  Evergy Kansas Central→EVRG, AES Indiana→AES) are decades-stable and
  matched prior knowledge, but merchant/IPP generators churn fast and
  TWO SEARCHES CAUGHT STALE ASSUMPTIONS BEFORE THEY SHIPPED: (1)
  `NRG Homer City Services LLC` — the plant retired June 2023 and was
  physically demolished in 2025 (now a gas-fired/data-center
  redevelopment); NRG was the contracted OPERATOR, never the OWNER
  (ownership sat with a hedge-fund vehicle from the plant's bankruptcy
  era) — mapping this to NRG would have been a confidently-wrong
  ticker on a plant that no longer exists. (2) `Louisiana Generating
  LLC` — sold by NRG to Cleco in Feb 2019; Cleco itself has been
  privately held since a 2016 investor buyout, so there is NO current
  public-ticker mapping at all, not even Cleco. Also resolved
  correctly via search rather than guesswork: the Feb-2022 Exelon→
  Constellation spinoff (`Exelon Nuclear`/`Constellation Mystic Power
  LLC`→CEG, not EXC), the 2018 Dynegy→Vistra merger (`Dynegy Midwest
  Generation Inc`→VST), Talen Energy's 2022 bankruptcy-emergence
  relisting (NASDAQ: TLN), and the fragmented, no-single-public-parent
  ownership of South Texas Project (`STP Nuclear Operating Co` —
  Constellation 42% / CPS Energy 42% municipal / Austin Energy 16%
  municipal as of a 2024 transaction) and Keystone/Conemaugh (`KeyCon
  Operating LLC` — multiple PE holders + a ~12-16% Talen minority
  stake). RESULT: 44/69 mapped (34 "high" confidence wholly-owned
  regulated-utility or clean-merger cases, 1 "medium" —
  `PacifiCorp`→BRK.B, two levels removed via Berkshire Hathaway
  Energy, confidence deliberately held down), 25/69 honest unmapped
  gaps — federal agencies (TVA, USACE, Bureau of Reclamation), state/
  municipal port and power authorities (8 of the 13 site operators),
  and privately-held or fragmented-JV merchant generators (GenOn,
  Gavin Power, Helix Ravenswood, LaFrontera, Midland Cogeneration
  Venture, Cardinal Operating, KeyCon, plus the two stale entries
  above). Zero guessed tickers anywhere — every unmapped entry carries
  a one-line reason.
- DOWNSTREAM CHAIN (REASONING STANDARD #1): this table -> unblocks
  EVERYTHING_GRAPH.md build-plan step 2 (`server/entityGraph.ts`, the
  `operates` edge type company→facility) for whichever session claims
  it next -> ALSO unblocks the independently-filed fusion hypothesis
  (b) "Generation shifts × utility tickers" (open_questions.md FUSION
  HYPOTHESES section), whose own gate-1 ground truth explicitly
  required "a registry-owner→ticker mapping table" — this is that
  table, built once and shared, per the design doc's stated intent
  ("removes the join labor, grants no evidential shortcut" — gate 1
  for that hypothesis still requires the separate EIA-930-vs-registry-
  capacity reconciliation, not done here). No trading behavior changed
  (RAW reference data only, no SIGNAL claim, nothing wired into
  deep_score or any live route yet — step 2 is what serves it).
- REGRESSION TESTS: `server/entityMap.test.ts` (new, 5 cases) — every
  entity has required fields + valid confidence tier + no duplicate
  keys; unmapped entities never carry a ticker; the `coverage` block
  in the JSON matches the actual entity list (catches stale
  bookkeeping); COVERAGE HONESTY — every operator string currently in
  `strategic_sites.json` and every top-100-plant owner in
  `us_power_plants.json` has an entry (fails if either source registry
  grows a new unresearched operator, forcing the next session to
  research it rather than silently under-covering); the doc text
  carries the "re-verify" honesty warning and is marked "no predictive
  claim" (RAW not SIGNAL); a fifth test spot-checks that named federal/
  municipal operators are specifically `unmapped`, not silently
  dropped.
- PROMOTION RULES: (1) `npm run test:node` — 214/217 passed; the 3
  failures (`compression.test.ts`, `gdeltEvents.test.ts`,
  `owmTiles.test.ts`) are PRE-EXISTING and unrelated, confirmed via
  `git stash` A/B on the exact same command (209/212 passed on the
  pre-PR commit, identical 3 failures) — this session's 5 new tests
  all pass, no regression. `python3 -m pytest -q` — this sandbox
  started with NO python deps installed at all (`voltrade_daemon.py`
  hard `sys.exit(2)`s if numpy/pandas/requests import fails, which
  crashed pytest's collector entirely); installed `requirements.txt`
  to get a real signal rather than skip the gate, then confirmed via
  the same `git stash` A/B that the result (392 passed, 2 skipped, 2
  pre-existing failures in
  `test_options_v134_fixes.py::TestFix7_EarningsAlwaysIronCondor` —
  the same `KeyError: 'opt_type'` in `options_scanner.py:490` a prior
  session already logged) is identical before and after this PR — this
  PR touches zero files that pytest's suite exercises. (2) new tests
  ship with the new file (rule 2). (3) not a strategy/parameter change
  — no backtest required (this is RAW reference data, not a trading
  rule). (4) version bumped 1.0.130 -> 1.0.131, read-and-increment,
  confirmed against the GitHub API's live `main` HEAD (99921a6,
  identical to this branch's parent) immediately before bumping — the
  prior session's OPS GOTCHA about a stale local `git fetch` cache
  reproduced again this session (a first `git log origin/main` showed
  main 50 commits "behind" after a plain `git fetch`; the GitHub API's
  `list_commits(sha=main)` immediately showed main's HEAD already
  matched local HEAD, i.e. no actual divergence) — worth a permanent
  note: prefer the GitHub API or `git ls-remote` over a bare
  `git fetch && git log origin/main` when this repo's mirror seems
  stale, since the cache artifact has now recurred twice. (5) one
  logical change (one new registry file + its builder + its test);
  research/doc updates and the version bump are the SHARED-file tail
  of the same PR, not a second logical change. (6) VISUAL VERIFICATION
  not applicable — no client/ files touched.
- LIVE HEALTH CHECK (task-mandated, done before the primary action):
  `GET https://voltradeai-production.up.railway.app/api/health`
  returned `status: ok` across the board — server, database, Alpaca
  ACTIVE, python bridge, bot `active`, equityPeak $108,151.39,
  drawdownPct 0.0, `liveness.dark: false`. No LIVENESS ALARM
  condition, nothing to surface top-of-report; unchanged from the
  prior session's reading a few hours earlier.
- NOT IN SCOPE, FLAGGED HONESTLY: step 2 (`server/entityGraph.ts` +
  `/api/data/graph` + tests) and step 3 (the `/data` graph panel) are
  the natural next PRs for whichever session (interactive or routine)
  claims R5 next — this entry's build-plan update in
  EVERYTHING_GRAPH.md and open_questions.md is the claim-before-
  building marker for step 1 only, step 2/3 remain open and unclaimed.

## 2026-07-05 — [RULE-REVIEW] Counterfactual logger: correlation/spread rejections now labeled truthfully in the shadow archive (BUILD ORDER 4 #6 continuation) (v1.0.130) [T-BOT]

- [T-BOT] Territory: bot_engine.py, shadow_portfolio.py (a data module,
  MUTABLE per CLAUDE.md; not one of the T-DATACORE-listed modules but
  squarely inside T-BOT's `server/bot.ts`-adjacent trading-loop scope
  since it's called exclusively from `bot_engine.py`'s scan path), new
  test_shadow_portfolio.py (WORKSTREAM PARTITION). SESSION START per
  MEMORY PROTOCOL: read CLAUDE.md, experiments.md, open_questions.md,
  wishlist.md. Loop-health ratio over the last 10 entries: PRODUCT 3,
  RULE-REVIEW 2, PIPELINE 1, RESEARCH 2, REPAIR 2 — no thrash (2/10
  REPAIR, well under the 7/10 trigger). /api/health checked first: all
  ok, bot active, equityPeak $108,151.39, drawdownPct 0.0, liveness not
  dark — no KNOWN BROKEN item blocked this session, nothing to surface
  top-of-report. `git fetch origin main` confirmed the branch starts
  clean at v1.0.129 (a stale local ref cache briefly showed origin/main
  20+ commits behind; a fresh fetch resolved it — no actual divergence,
  just a cache artifact worth noting for the next session).
- PRIMARY ACTION: BUILD ORDER 4 #6's own build plan (filed this same
  day, see the `be5125e` entry below) named the next concrete, unblocked
  step — "add a block_reasons[] tag ... at the rejection sites in
  bot_engine.py scan/deep_score ... LOGGING ONLY — no mechanism change."
  This was the highest-value fall-through item: BUILD ORDER 4 #4/#5
  wait on future calendar dates (~2026-09-27, 2026-07-06), #3
  (international registries) was already probed-and-deferred today,
  #2 (UI scalability) already shipped today (v1.0.129) — #6's gap was
  the only unblocked, fully-specified queued item left in the roadmap.
  SESSION-BUDGET TIER: "fix a bug seen in audit logs" was not directly
  actionable (no owner-auth audit-log access, per KNOWN BROKEN #4's
  ACCESS LIMITATION, unchanged), so this queued item — fall-through
  tier 1 — was the correct next action, not a downgrade to research.
- HYPOTHESIS STATED BEFORE MEASURING (REASONING STANDARD #10): read
  `bot_engine.py`'s `_scan_market_inner()` (the closure `scan_market()`
  actually delegates candidate-filtering to — NOT `scan_market()`
  itself, which is a thin timeout wrapper; caught this by running the
  new tests against the wrong function first, see BUG FOUND below) top
  to bottom this session. `deep_score()` calls `log_candidate()` with
  decision `"taken"` the moment `combined_score >= MIN_SCORE` — but
  `_scan_market_inner()`'s per-candidate loop applies MORE filters
  AFTER that: cooldown, regime block, correlation/sector
  (`check_sector_correlation`), $50M dollar-volume floor, blocked-
  ticker list, extreme-mover (>50% today) watchlist diversion, and a
  live bid/ask spread check — all of which can still `continue` (skip)
  a candidate `deep_score()` already logged as `"taken"`. PRIOR: ~70%
  chance at least one of these downstream filters was silently
  mislabeling real rejections as "taken" in the shadow archive, since
  nothing in the codebase or `open_questions.md`'s prior audits (KNOWN
  BROKEN #10, the 2026-07-04 `shadow_portfolio.py` audit) had traced
  candidates past `deep_score()`'s own logging call.
  CONFIRMED: correlation/sector blocks and the quote-time spread check
  are exactly this bug — real, frequent rejection paths whose
  candidates were being recorded as "taken" in the learning archive
  before this fix. This is a live HONESTY METRIC risk (GOAL doc): any
  future session running `get_shadow_stats()`'s `win_rate_by_decision`
  would have attributed correlation/spread REJECTIONS' outcomes to the
  "taken" bucket, corrupting exactly the MIN_SCORE RULE COST AUDIT
  question it's meant to answer (mixing "we actually traded this" with
  "we scored it but a downstream filter blocked it" is a different
  population with a different expected win rate).
- BUILT (one logical change, logging-only, mechanisms untouched):
  (1) `shadow_portfolio.update_last_decision(ticker, decision,
  decision_reason, max_age_seconds=120.0)` — new function, mirrors
  `log_candidate()`'s non-blocking contract (any failure swallowed,
  logging must never break the trading loop). Walks the shadow log in
  reverse, finds the MOST RECENT record for the ticker, and corrects
  its `decision`/`decision_reason` ONLY IF that record is still
  `"taken"` (not already resolved by something else) AND still fresh
  (<=120s old, so a stale same-ticker record from an earlier scan can
  never be mislabeled). (2) Two call sites added in
  `_scan_market_inner()`: the `check_sector_correlation()` rejection
  branch now calls `update_last_decision(ticker, "rejected_heat", ...)`
  before its `continue`; the `_spread_pct > 0.005` rejection branch now
  calls `update_last_decision(ticker, "rejected_other", ...)` before
  its `continue`. Both wrapped in their own `try/except Exception:
  pass` so a shadow-logging failure can never affect which candidates
  actually get skipped — the filters' actual trading behavior is
  byte-for-byte unchanged.
- NOT WIRED, deliberately, and why (per the build plan's own scope):
  `rejected_halt` — `check_kill_switches()` (risk_kill_switch.py, a
  FROZEN PATH for mechanisms) gates the separate `TieredStrategy`
  action list (`tiered_actions`), a DIFFERENT code path from the
  `deep_score()`-based `trades` loop this fix targets; wiring it would
  mean tracing kill-switch state into a structurally separate strategy
  engine, a bigger and riskier change than this session's scope.
  `rejected_earnings` — grepped the entire `trades`-loop path: no
  per-candidate stock-long earnings blackout exists today (earnings
  only enters `deep_score()` as a soft ML feature, and separately gates
  covered-call selection in `options_execution.py`'s
  `_check_earnings_guard`, a different candidate population). There is
  no REJECTION SITE to log yet — adding one would mean adding a new
  hard gate, which is a genuine RULE REVIEW-gated behavior change
  (evidence or ablation required), not a logging-only PR. Both left
  open in `open_questions.md`'s BUILD ORDER 4 #6 entry as the next
  slice, correctly scoped separately per PROMOTION RULES rule 5 (one
  logical change per PR).
- BUG FOUND DURING BUILD (the tests caught it, not review): the first
  version of the source-inspection test called
  `inspect.getsource(bot_engine.scan_market)` and got a suspiciously
  short 2,651-char result with neither call site in it — `scan_market()`
  is a thin `def` that sets up a timeout signal handler and delegates
  the real loop to a nested closure, `_scan_market_inner()` (confirmed
  via `grep -n "^def " bot_engine.py` around the relevant line range).
  Fixed by inspecting `_scan_market_inner` instead; both assertions then
  passed. Exactly the READ-BEFORE-WRITE risk CLAUDE.md warns about —
  caught here by a failing test, not by assumption.
- REGRESSION TESTS (new file, `test_shadow_portfolio.py`, written
  BEFORE considering the change complete, per loop-health rule 3):
  6 unit tests on `update_last_decision()` directly (updates a fresh
  "taken" record; no-ops when the most recent record is already
  resolved; no-ops when the record is stale even if max_age_seconds is
  explicit; no-ops on a ticker with no matching record; only ever
  touches the FRESHEST record for a ticker, proven with two seeded
  records for the same ticker at different ages; empty-ticker guard) +
  2 source-inspection tests pinning both `_scan_market_inner()` call
  sites to their exact decision-bucket string and to a defensive
  `except Exception` wrapper (mirrors `test_voltrade_daemon.py`'s
  established pattern of pinning wiring via source inspection). All 8
  pass. Full offline CI-gate subset re-run: `python3 -m unittest
  test_risk_controls test_audit_critical test_diagnostic_false_positives
  test_patches_verification test_voltrade_daemon test_shadow_portfolio`
  — 133 passed, 1 skipped (pre-existing skip, unrelated; identical
  baseline otherwise). `python3 -m py_compile bot_engine.py
  shadow_portfolio.py test_shadow_portfolio.py` clean. NOTE: this
  sandbox lacked numpy/pandas/lightgbm/scikit-learn/requests before
  `pip3 install`, which is why `bot_engine`-importing tests initially
  errored on `ModuleNotFoundError` unrelated to this change — resolved
  by installing the same packages `requirements.txt` already declares;
  not a repo issue.
- Downstream chain (REASONING STANDARD #1): a mislabeled shadow record
  -> `win_rate_by_decision["taken"]` silently pools real trades with
  candidates a downstream filter actually blocked -> the MIN_SCORE RULE
  COST AUDIT question (and any future one relying on the "taken" bucket
  meaning "we actually traded this") reads a biased number without any
  visible error -> fixing the label means `rejected_heat`/
  `rejected_other` become their OWN buckets in `get_shadow_stats()`
  automatically (it groups generically by whatever `decision` string
  appears, confirmed by reading `get_shadow_stats()`'s implementation —
  no code change needed there) -> once >=90d of shadow history
  accumulates (~2026-10-02, unchanged from the prior estimate — this
  PR doesn't touch WHEN backfill runs), the correlation-block and
  spread-filter RULE COST AUDIT questions in `open_questions.md` become
  answerable for the first time. Zero live-trading-behavior change:
  no filter's pass/skip decision changed, only which population its
  rejected candidates get correctly bucketed into for later analysis.
  Version bumped 1.0.129 -> 1.0.130 (read-and-increment, package.json
  is a SHARED file per WORKSTREAM PARTITION — this was the last, small,
  isolated edit in the PR). PROMOTION RULES rule 3 (backtest
  requirement) doesn't apply — no strategy/parameter/threshold changed,
  only which decision string an already-existing rejection logs under.
- MARKET-HOURS NOTE (this run occurs during market hours, per session
  instructions): PR prepared; recommend merge waits until after 4:00 PM
  ET today unless a critical live break is found (none was — this is a
  logging-only correction to the shadow/learning archive, not a
  trading-path fix, so there is no urgency argument for an immediate
  merge).
- STARVED: no — this session's scope (the queued B4-6 item, in full for
  its logging-only-appropriate slice) shipped completely. High-value
  work remains queued for future sessions: `rejected_halt`/
  `rejected_earnings` wiring (this entry, above), KNOWN BROKEN #3
  (CSP execution cascade — still needs owner-auth audit-log
  verification), KNOWN BROKEN #10 (SCORE_BAND_MAX/MAX_CHANGE_PCT
  evidence-or-retire decision, still waiting on shadow history), BUILD
  ORDER 4 #3/#4/#5 (all correctly deferred to their own trigger dates),
  the Python-side staleness-audit sweep the register above still marks
  UNSCANNED, and the CONSTITUTIONAL REPAIR proposals awaiting human
  review in wishlist.md.

## 2026-07-05 — [PRODUCT] UI SCALABILITY ARCHITECTURE — registry-native group/costTier + panel row-cap + 50/100/200-layer synthetic harness (BUILD ORDER 4 #2, GIP Part 4) (v1.0.129) [T-CLIENT, touching datacore/layers.json additively]

- [T-CLIENT] Territory: client/src/**, index.css, scripts/visual_check.mjs
  (WORKSTREAM PARTITION); this PR also touches datacore/layers.json but
  ADDITIVELY only (two new optional metadata fields per layer, no pipeline
  logic changed) — declared here per the partition's cross-territory rule
  (one session, one logical change, not split). SESSION START per MEMORY
  PROTOCOL: read CLAUDE.md, experiments.md, open_questions.md, wishlist.md.
  Loop-health ratio over the last 10 entries: RULE-REVIEW 2, PIPELINE 2,
  RESEARCH 2, REPAIR 2, PRODUCT 2 — no thrash (well under 7/10). /api/health
  checked first: all ok, bot active, equityPeak $108,151.39, drawdownPct
  0.0, liveness not dark — no KNOWN BROKEN item blocked this session, none
  required noting at top-of-report.
- PRIMARY ACTION: BUILD ORDER 4 #2 (self-proposed 2026-07-05, GIP Part 4
  UI SCALABILITY, "IN-PROGRESS" since 2026-07-04) was the next unblocked
  queued product item — #1 (operator resolution) shipped v1.0.127, #3
  (international registries) is blocked on per-country access-page
  discovery, #4 (natgas gate-2) and #5 (options-chain QA) both wait on
  future calendar dates, #6 (counterfactual logger) is T-BOT. This is a
  T-CLIENT item and the only one actually actionable today.
- HYPOTHESIS STATED BEFORE MEASURING (REASONING STANDARD #10): the panel's
  default-open groups (base, live) render ALL members unconditionally and
  PANEL_GROUPS/LAYER_GROUP are hardcoded per-id maps in datamap.tsx — as
  the registry grows toward "hundreds of layers" (GIP Part 4), (a) a large
  default-open group could dump unbounded DOM, and (b) a new layer added
  only to the registry (no client code change) would need a matching
  hardcoded LAYER_GROUP/groupCollapsed entry or it silently mis-groups or
  defaults OPEN. Prior: ~40% chance today's small registry (21 layers, max
  group size 6) already masks a real scaling defect that only shows at
  50-200 layers — worth measuring rather than assuming either way.
- BUILT (one logical change): (1) registry-native `group` + `costTier`
  fields added to every datacore/layers.json entry (schema documented in
  `_doc`) — `groupOf()` in datamap.tsx now prefers `l.group`, falling back
  to the old LAYER_GROUP map only for the visual-harness fixture / a
  registry response from an older deploy mid-rollout; (2) `groupCollapsed`
  init switched from a hardcoded collapsed-name list to a computed
  `!OPEN_GROUPS_BY_DEFAULT.has(id)` (OPEN_GROUPS_BY_DEFAULT = {base, live})
  — IDENTICAL result for today's 6 groups (verified: zero visual diff) but
  any group id introduced later defaults COLLAPSED automatically instead
  of needing a second hardcoded entry remembered; (3) GROUP_ROW_CAP = 12:
  an open group renders at most 12 rows behind a "+N more — show all"
  control — no-op today (max group size 6) but bounds DOM per group at any
  registry size; (4) a `costWeightOf`-summed active-cost-budget badge
  ("moderate load"/"heavy load", silent below weight 15) in the panel
  header — a genuine consumer of costTier, not decorative metadata (the
  STALENESS AUDIT would rightly flag an unused field); (5) an unknown-group
  catch-all ("_more") so a layer whose `group` isn't in PANEL_GROUPS still
  renders instead of silently vanishing from the panel.
- BUG FOUND AND FIXED DURING BUILD (the harness caught it, not review): the
  first version of the "_more" catch-all still showed only ~75% of
  synthetic layers reachable via "show all" — traced to exactly the defect
  the catch-all exists to prevent (layers whose `group` didn't match any
  PANEL_GROUPS id were filtered out of the render entirely, only labeled
  correctly). Fixed by extracting `renderPanelGroup()` and calling it for
  both the named PANEL_GROUPS and the orphan set; re-ran the harness to
  confirm 50/100/200 all reach 100%.
- SEPARATE FINDING, NOT A BUG (verified by direct debugging, see below):
  synthetic layers correctly render their toggle DISABLED ("unwired" guard,
  pre-existing 2026-07-04 open-tab-skew protection) because no real
  map-data fetch/render effect exists for a fabricated `synth_N` id — a
  registry edit alone (adding `group`/`costTier`) was never going to make
  a brand-new interactive layer functional without an actual client
  deploy, and the guard correctly refuses a toggle that would flip and
  paint nothing. Confirms the guard and my registry-native change are
  solving two different problems (grouping/visibility vs. real wiring) —
  worth recording so a future session doesn't re-litigate it.
- MEASURED (scripts/visual_check.mjs, new `--page scale` battery,
  synthetic 50/100/200-layer registries via a per-context Playwright route
  override — the shared FIXTURES/server untouched, no determinism cost to
  any other page): default-open panel rows stayed at 14/24/24 (well under
  the 30-row regression-guard budget) across n=50/100/200 — collapse-by-
  default + GROUP_ROW_CAP hold regardless of registry size, not just
  today's. "Show all" reached 50/50, 100/100, 200/200 layers (100% self-
  see at scale, after the fix above). TTI stayed 1.2-2.5s, under the
  existing 3000ms map-page gate this file already uses elsewhere — no
  regression at scale. The cost-budget badge was separately exercised on
  the REAL 21-layer fixture (toggling all 9 non-default heavy/moderate
  layers, weight 13->35) and correctly read "heavy load" — `.visual/
  results.json`'s `data`/1440 entry: `costBudgetBadge: "heavy load"`,
  `toggleConsistency: "16 layers toggled clean"`.
- CONCLUSION vs PRIOR: hypothesis partially confirmed — the architecture
  DID need the fix (the orphan-group defect was real, not hypothetical),
  but once fixed, today's collapse-by-default + 12-row cap combination
  already holds at 200 layers with real margin (24 rows vs 30 budget) —
  literal windowed DOM virtualization is NOT yet evidence-justified;
  filed as a precise trigger condition in open_questions.md BUILD ORDER 4
  #2 rather than built speculatively (CLAUDE.md: don't design for
  hypothetical requirements) — revisit if any single group's real member
  count approaches ~25 (row cap 12 + one showAll click still renders all
  25, which is the actual measured-safe ceiling per the n=50 case above:
  the largest synthetic group there held ~7 members and passed cleanly;
  extrapolating the n=200 case, up to ~25-member groups measured clean).
- DOWNSTREAM CHAIN (REASONING STANDARD #1): registry-native group/costTier
  -> a future pipeline session can add a datacore layer with correct panel
  placement and a cost estimate by editing layers.json alone (no
  datamap.tsx PR required for KNOWN groups) -> the human's weekly /data
  review keeps working as the registry grows -> BUT the layer stays
  non-interactive ("unwired"/"reload to enable") until a follow-up client
  PR adds its real fetch/render effect, by design — this PR does not
  change that constraint, only removes the panel-placement bottleneck.
- PROMOTION RULES: (1) full test:node suite 223/223 passed (unchanged
  count from before this PR — no test removed/weakened); (2) new tests are
  the scale-harness battery itself (mechanical, in scripts/visual_check.mjs,
  since this repo's client layer has no unit-test framework — DESIGN.md's
  visual harness is the established verification path for client/) plus
  the real-fixture cost-budget exercise; (3) not a strategy/parameter
  change — no backtest required; (4) version bumped 1.0.128 -> 1.0.129
  (read-and-increment at commit time per MERGE-ORDER PROTOCOL, confirmed
  against origin/main immediately before bumping — no new merges since
  this branch was cut); (5) one logical change, own PR; (6) VISUAL
  VERIFICATION: `npm run visual` (soft mode) run at 390/768/1440 for all
  three pages (data/developers/landing) plus the new scale/all-off
  batteries — 0 hard failures; screenshots reviewed
  (.visual/data-1440.png shows the unchanged default panel — no cost
  badge, matches the "silent below weight 15" design; .visual/
  data-scale-200.png shows the 200-layer synthetic registry rendering
  correctly with the "reload to enable" unwired state and a scrollable
  panel). Pre-existing warnings (nav touch-target sizes, "Filings & flows"
  clipped-control note) verified UNCHANGED from the pre-PR baseline via
  `git stash` A/B (not a regression — filed nowhere new, already
  pre-existing per the file's own history).
- NOT IN SCOPE, FLAGGED HONESTLY: `python3 -m pytest -q` was run as a
  sanity check (no Python files touched by this PR) and found 2
  PRE-EXISTING failures unrelated to this change — `test_options_v134_fixes
  .py::TestFix7_EarningsAlwaysIronCondor` (both cases), `KeyError:
  'opt_type'` in `options_scanner.py:490`'s `_find_by_delta` — confirmed
  pre-existing via `git stash` A/B (identical 2 failures on the pre-PR
  commit). T-BOT territory, out of scope for this T-CLIENT PR per
  one-logical-change-per-PR; noting here per the REPAIR MANDATE so a
  T-BOT session doesn't have to rediscover it. `npm run check` (tsc): 63
  pre-existing errors, unchanged count, none touch datamap.tsx (verified
  by grep) — consistent with the prior session's note that tsc is not a
  clean gate in this repo.
- BUILD ORDER 4 #2 STATUS: updated in open_questions.md with the measured
  numbers above — item stays open only for the "revisit if a group
  approaches ~25 members" trigger; not closed as "done forever" since
  that's a real future condition, not a today-problem.

## 2026-07-05 — [RULE-REVIEW] Performance/ml-status/diag slippage stats were reading a dead file — realistic-P&L honesty bug fixed (v1.0.128) [T-BOT]

- [T-BOT] Territory: bot_engine.py/ml_model_v2.py/server/bot.ts outside
  frozen paths, per WORKSTREAM PARTITION. SESSION START per MEMORY
  PROTOCOL: read CLAUDE.md, this file, open_questions.md, wishlist.md.
  Loop-health ratio over the last 10 entries at session start: 2
  [REPAIR], 3 [RESEARCH], 2 [PRODUCT], 3 [PIPELINE], 0 [RULE-REVIEW] —
  well under the 7/10 [REPAIR] thrash threshold, no meta-problem to
  address. DATACORE DEFECT QUEUE (all 10 items + #237/#238) confirmed
  fully closed by prior sessions. /api/health: all checks ok, bot
  active, equityPeak $108,151.39, drawdownPct 0.0, liveness not dark.
- PRIMARY ACTION (SESSION BUDGET tier 1: "fix a bug seen in audit logs"):
  used the human-approved /api/diag route (DIAG_TOKEN, wishlist option
  (d)) to probe live state. `/api/diag/ml` returned
  `{feedback_count: 500, fills_count: 0}` — 500 real trade_feedback
  records but zero "fills." Traced (READ BEFORE WRITE): FILLS_PATH
  (storage_config.py, voltrade_fills.json) has had exactly ONE writer
  ever, ml_model.py's legacy track_fill() — and nothing imports
  ml_model.py anymore (grep confirmed zero call sites repo-wide; it's
  fully orphaned). The LIVE track_fill (ml_model_v2.py, the one bot.ts
  actually calls on every order fill) writes entry-fill
  expected_price/fill_price/slippage_pct straight into
  TRADE_FEEDBACK_PATH instead, and has done so since v1.0.34. Result:
  every route reading FILLS_PATH for slippage/fill-count data
  (`/api/bot/performance`'s realistic-P&L calc, `/api/bot/ml-status`,
  the `/api/diag/ml` probe, plus an unused dead read in
  `/api/bot/export-trades`) has ALWAYS seen an empty list, so
  avgSlippagePct/totalSlippageCost/slippageGapPct/totalFills have been
  silently pinned at zero on every deploy regardless of real trading
  activity.
- HONESTY METRIC RELEVANCE: this is exactly the self-deception CLAUDE.md
  warns about — the performance dashboard's "realistic P&L net of
  slippage" has been reporting IDENTICAL to paper P&L this whole time
  (slippageGapPct always 0), even though real per-trade slippage data
  existed all along, just recorded into a different file than the one
  the dashboard reads. REASONING STANDARD #6 (costs/frictions first)
  was being silently violated by the measurement layer itself, not the
  strategy.
- FIX (own PR, MEASUREMENT-INTEGRITY isolation — no trading behavior
  touched): added `ml_model_v2.fills_slippage_stats(feedback)`, a pure
  function deriving count/avg_slippage_pct/total_slippage_cost directly
  from trade_feedback's entry-fill records (exit-fill updates and
  orphan-exit records lack expected_price/slippage_pct, so they're
  excluded by construction — verified by test). All four `server/bot.ts`
  call sites now use it instead of reading FILLS_PATH; the dead
  export-trades read (loaded, never used in the CSV) was deleted
  outright.
- BEFORE vs AFTER on identical inputs (PROMOTION RULE + MEASUREMENT
  INTEGRITY requirement — see test_fills_slippage_stats.py, 7 cases):
  BEFORE — any non-empty feedback list still yielded
  `{avgSlippagePct: 0, totalSlippageCost: 0, totalFills: 0}` because the
  code looked at a file nothing writes to. AFTER — real entry-fill
  records (expected_price + slippage_pct present) are correctly counted
  and aggregated; e.g. two synthetic fills with slippage_pct 0.05% and
  0.20% now correctly average to 0.125% instead of reporting 0%. DIRECTION
  OF BIAS: this makes realisticPnlPct/slippageGapPct MORE conservative
  (reveals cost drag previously hidden at zero), never more favorable —
  not the "make the strategy look better" pattern MEASUREMENT INTEGRITY
  treats as suspect by default; it is a named bug (two-file split, one
  side orphaned) fixed by reading from where the data actually lives.
  Could not diff against the ACTUAL 500 live records (no diag probe
  exposes raw trade_feedback content by design — the whitelist
  deliberately excludes raw trade data) — stated honestly rather than
  overclaiming a live before/after; the synthetic-fixture test is the
  verification artifact.
- DOWNSTREAM CHAIN (REASONING STANDARD #1): fixed slippage aggregation
  -> `/api/bot/performance`'s realisticPnlPct/slippageGapPct/totalFills
  now reflect true historical fill quality -> the human's next dashboard
  view of "how much is slippage actually costing us" becomes
  trustworthy for the first time since ml_model_v2 replaced ml_model ->
  no change to what the bot trades, sizes, or when it exits (isolation
  requirement satisfied).
- TESTS: `test_fills_slippage_stats.py` (7 cases, python3 -m pytest —
  full suite: 386 passed, 2 skipped, unchanged from before this change).
  `server/fillsSlippageWiring.test.ts` pins that bot.ts never reads
  FILLS_PATH again and that all three live routes call
  fills_slippage_stats (npm run test:node — full suite: 223/223 passed).
  `npm run check` (tsc): 63 pre-existing errors before AND after this
  change (verified via git stash) — none touch edited lines; tsc is not
  a clean gate in this repo today, unrelated to this PR.
- NOT IN SCOPE (flagged, not fixed, per one-logical-change-per-PR):
  `storage_config.FILLS_PATH` itself and the now-fully-orphaned
  `ml_model.py` module (zero import sites repo-wide) are dead code —
  exactly the kind of finding the in-progress STALENESS AUDIT (register
  above, Python-side sweep still marked UNSCANNED) should sweep up and
  delete outright in a future session; noting it here so that session
  doesn't have to rediscover it.

## 2026-07-05 — [REPAIR] TestFix7 wall-clock-dependent options tests — deterministic time + real contract shape (test-only)

- FOUND BY: the Census-stream ship gate (full pytest at 15:20 ET
  Sunday) — first full run ever to land inside 9:30-16:00 ET.
  test_options_v134_fixes.py::TestFix7 (2 tests) failed with
  KeyError: 'opt_type'.
- ROOT CAUSE (two stacked defects): (1) _setup_earnings_iv_crush
  early-returns None outside 9:30-16:00 ET (_is_regular_hours checks
  time-of-day ONLY — fires on weekends too), and the tests' "if
  result is not None" guards made them pass VACUOUSLY outside that
  window — every prior green run never exercised the deep path.
  (2) Inside the window, the mock fixtures predated the current
  _fetch_options_chain contract shape (keys "type"/"expiry", no
  "delta") so _find_by_delta crashed. Production is unaffected —
  real chains carry opt_type/delta (options_scanner.py contract
  build); the stale fixture was test-only debt.
- FIX (test file only, no runtime change → no version bump, same
  convention as docs PRs): patch _is_regular_hours (deep path runs
  at any wall-clock time) + _get_spy_vs_ma50 (was an unmocked live
  fetch inside the patched-feature call's argument list); fixtures
  rebuilt via a _contract() helper matching the exact
  _fetch_options_chain dict; vacuous guards upgraded to hard
  assertIsNotNone + mock_chain.called (RATCHET: a silent
  None-regression now fails loudly instead of skipping every
  assertion). High-IV test additionally pins the wide-wings label.
- Verified: 47/47 in the file and 382 passed / 1 skipped suite-wide,
  run INSIDE the reproducing ET window.
- LESSON (recurring class): "if result is not None: assert..." is a
  vacuous-pass pattern — the test passes forever if the code path
  dies. Same family as the wall-clock dependence: both make a test's
  meaning depend on state outside the fixture. Grep for the pattern
  during the next staleness audit.

## 2026-07-05 — [PIPELINE] Census port imports — key-gated stream built (BUILD ORDER 3 #4 unblocked) (v1.0.132)

- TERRITORY: T-DATACORE. Human message "CENSUS_API_KEY added key"
  unblocked BLOCKED-FOR-MIKE #6 (the only build-order item that was
  key-blocked). Built server/censusImports.ts + tests + manifest +
  /api/data/imports.
- KEY-LOCATION FINDING: the key is NOT in this session's container
  env (presence-only check; container env is fixed at session start —
  exact FRED precedent). So instead of a session-run backfill, the
  stream is SERVER-SIDE key-gated on the fredMacro pattern:
  censusEnabled() = Boolean(CENSUS_API_KEY), poller no-ops keyless,
  route returns {enabled:false, reason} honesty, activates
  automatically on the next deploy IF the key was set in Railway.
  Wishlist #6 corrected (it originally said "session env; Railway not
  needed" — the built design inverts that) and marked
  DONE-PENDING-VERIFICATION.
- QUERY-SHAPE HONESTY: the intltrade imports/porths parameter set
  could not be live-verified without the key (keyless probes 302 to
  missing_key). Mitigations built in: (a) two documented query
  variants tried in order (full containerized set, then GEN_VAL_MO
  fallback); (b) HEADER-DRIVEN parsing — column order never assumed;
  (c) Census's readable error bodies logged verbatim (key never
  logged, never archived — test-asserted) so a wrong shape is fixed
  from prod logs, not guesswork. LIVE VERIFICATION PENDING: first
  DAILY session after deploy checks /api/data/imports.
- Archive: append-only JSONL day-files under censusimports/, dedup
  key port|month|values so FT920 revisions append as new vintages;
  seedSeen 40d; gz after 2d; daily poll (monthly source, ~45d lag —
  off-days are dedup no-ops); eager boot poll (KNOWN BROKEN #9).
  Missing values null, never zero. Public domain, attribution
  "U.S. Census Bureau (USA Trade Online / FT920)".
- HYPOTHESIS (stays gate-locked; RAW display + archive only): port
  import value/containerized-weight deltas lead retail inventory
  cycles; joins with port-dwell analytics for a two-sided port view
  (demand value × supply friction). Ladder work begins only after
  live data lands and gate 1 (readings vs a second official source)
  is designed.
- Tests: censusImports battery 4/4 (header-driven + shuffled-column
  identity, ''→null, key gating, variant fallback with
  key-never-in-records assertion, dedup/vintage/gz lifecycle);
  manifest envelope battery 3/3 with censusimports.json; tsc at the
  63-error baseline.

## 2026-07-05 — [RULE-REVIEW] Counterfactual-logger check-in + natgas gate-2 design pre-stated (BUILD ORDER 4 #4+#6) (docs)

- [RULE-REVIEW] B4-6 VERDICT: the CLAUDE.md counterfactual mandate is
  SUBSTANTIALLY BUILT — shadow_portfolio.log_candidate() records
  every scanned candidate with features + nightly-backfilled
  +5/+10/+20d outcomes, so threshold rules get prevention-P&L
  post-hoc by re-applying predicates to the archive (strictly better
  than block-event logging for those rules). THE GAP: block-reason
  tags for rules whose predicates need non-logged state (correlation
  blocks, halts, quote-time spreads) — build plan filed for a T-BOT
  session (logging-only at rejection sites, mechanisms untouched);
  first readout unchanged at >=90d shadow history (~2026-10-02).
- [RESEARCH] B4-4: natgas-storage x degree-day gate-2 design
  PRE-STATED before any overlapping live week exists (expanding-
  window degree-day-implied draw; residual sign vs UNG Thu->Wed
  returns vs base rate, regime-split; PASS = >=60% sign hit on
  n>=12 out-of-sample weeks + positive excess; prior ~30% with the
  tank-fill posterior stated). Runs ~2026-09-27.
- Territory note: both filed as docs because the remaining B4 build
  items are T-BOT (#6 gap) and T-CLIENT (#2 UI scalability) — this
  session declared T-DATACORE; the partition holds and nothing
  idles (SESSION BUDGET rule 3).

## 2026-07-05 — [PIPELINE] Operator resolution — gate PASSED 98.4% group-aware, n=2,498 (BUILD ORDER 4 #1) (v1.0.127)

- [T-DATACORE] server/operatorResolution.ts: callsign-prefix
  inference (>=2 prefixed observations, >=60% majority, KNOWN-prefix
  table only — 23 ICAO codes data-driven from our archive survey;
  unknown prefixes stay null), trustee-registrant detection (pattern
  from our own spine: Bank of Utah 490 / UMB 328 / Wilmington 301 /
  TVPX 97 airframes), PARENT_GROUP mapping (Envoy/PSA/Piedmont ->
  American; Endeavor -> Delta; SkyWest/Republic deliberately NOT
  mapped — independent companies under capacity agreements). Fleet
  series now aggregates by resolved OPERATOR (works even for non-US
  hexes the spine can't match), falls back to registrant, labels the
  basis per airframe, counts trustee shadows, and carries the listed
  parent group for ticker-level studies.
- GATE (pre-stated >=90% on >=20 airframes; ran on n=2,498 — 125x
  the minimum): raw registrant-agreement 86.1% -> the mismatch class
  was ENTIRELY parent-registered regional jets flying wholly-owned
  subsidiary callsigns, i.e. the resolution being MORE precise than
  the registrant check (the check was circular for exactly the class
  the module exists for). Group-aware agreement: 98.4% -> PASS.
  Residuals verified correct-by-design (SkyWest/Republic capacity
  flying + one JET BLUE/JETBLUE string artifact). Both numbers
  reported; the check mis-specification is the lesson, not hidden.
- DESIGN LESSON (caught by test fixture): occurrence counting beats
  distinct-callsign dedup — one callsign seen twice is two
  observations of the same operator.
- Unlocks fleet-utilization GATE 2 (utilization x earnings at the
  GROUP level) once archive depth suffices.

## 2026-07-05 — [RESEARCH] BUILD ORDER 4 filed — deepen before new roots (docs)

- [T-DATACORE] Theme: the gate-2 unlocks and queued GIP items that
  tonight's builds created, before any new ingestion. Order (full
  hypotheses + gates in open_questions.md): (1) registrant->operator
  resolution — the B3-1 blocker; callsign-prefix inference from our
  own archive is the free key insight (a hex flying UAL#### IS
  United ops regardless of trustee registrant); (2) UI scalability
  architecture (GIP Part 4, [T-CLIENT]); (3) international aircraft
  registries v1 (the 22% of archived hexes FAA can't match);
  (4) natgas-storage x degree-day gate-2 design, criteria pre-stated
  now, runs at >=12 live-overlap weeks (~2026-09-27); (5) options-
  chain first-week QA (first snapshots 2026-07-06 close);
  (6) [RULE-REVIEW] counterfactual-logger check-in per the CLAUDE.md
  mandate.
- One item per PR; read-before-write rigor unchanged for later
  actions per SESSION BUDGET.

## 2026-07-05 — [RESEARCH] Anomaly-mining pass run — BUILD ORDER 3 COMPLETE (docs)

- [T-DATACORE] B3-6 executed per the angle-hunting mandate; full
  findings + pre-registered mining design in open_questions.md
  (BUILD ORDER 3 item 6 RUN annotation). Summary: (a) the pass found
  and fixed the event-loop scan class (#237/#238) — platform-eating-
  itself anomalies outrank market anomalies; (b) honest verdict that
  2-day-old position archives cannot host market mining (the 32
  dwell "anomalies" are archive-birth artifacts); (c) airline
  utilization baselines established under our coverage; (d) mining
  design PRE-REGISTERED with fixed statistics (z>=3 vs 4-week
  baselines, multiple-hypothesis discounting, out-of-sample-only
  belief) and re-run triggers at 30d/60d archive depth (~2026-08-03).
- BUILD ORDER 3 SCOREBOARD: 6/6 resolved — fleet utilization (gate-1
  PASS 20/20), EIA weekly (44yr backfill), CPC degree days (10.5yr),
  CBP imports (routed to BLOCKED-FOR-MIKE: Census key now required),
  Everything Graph R1 (live end-to-end on the site cards), anomaly
  pass (this entry). Two build orders proposed and executed to
  completion in one session; next session proposes BUILD ORDER 4 or
  takes the GIP queue.

## 2026-07-05 — [REPAIR] portdwell: the SAME event-loop defect, 4th site, heavier window (v1.0.126)

- [T-DATACORE] Minutes after #237's prod verify, BOTH analytics
  endpoints timed out again — portdwell's synchronous 168h scan on
  cache-miss was blocking the loop (shadowstats was collateral: its
  new poller answers instantly, but only when the loop is free).
  This confirms the RECURRENCE reading of the class: fixing
  shadowstats alone could never hold while any sibling still
  scanned synchronously.
- FIX: computePortDwellAsync over readVesselTracksAsync; BOTH
  surfaces (/api/data/portdwell + /api/v1/stats/portdwell) serve an
  eager 10-min poller cache (warming_up / 503+Retry-After during
  the first scan; metered honestly on the v1 side). Ratchet appended
  to the portDwell battery: async deepEqual sync on a fixture
  archive (8/8).
- CLASS CLOSED: grep confirms no remaining route calls a synchronous
  archive scan — every archive-derived surface (shadowstats,
  portdwell, fleet utilization, site timeline, hex enumeration,
  platform stats) is now streaming + eager-cached. The experiments
  pattern note from v1.0.125 stands as the standing rule for new
  surfaces.

## 2026-07-05 — [REPAIR] shadowstats blocked the whole event loop — async streaming + eager poller (v1.0.125)

- [T-DATACORE] FOUND BY the B3-6 anomaly pass surveying prod
  analytics: cold /api/data/shadowstats = 90s timeout (Railway 000)
  then 26s warm. ROOT CAUSE: computeShadowStats ran a SYNCHRONOUS
  72h gz archive scan on the request path (gunzipSync + full-file
  parse) — at current archive size (42k vessels seen this week) the
  scan blocked the ENTIRE Node event loop, starving every other
  route, health checks included. Same defect on the /api/v1 paid
  surface (per-request sync scan behind the API key).
- FIX: readVesselTracksAsync (streaming readline, the proven
  fleetUtilization/aircraftEntities pattern — the loop keeps
  breathing) + computeShadowStatsAsync; both routes now serve a
  10-min EAGER poller cache only (warming_up / 503+Retry-After
  while the first scan runs). v1 payload gains kind/source/zones
  fields (additive superset).
- RATCHET: equivalence test — async streaming reader byte-identical
  to the sync scan on a fixture archive incl. the gz path; full
  stats objects deepEqual.
- PATTERN NOTE (third instance tonight: trains inflight, spine
  disk-read, this): the failing class is always "works at day-0
  archive size, dies as archives grow." Every archive-scanning
  surface now uses the streaming+cached pattern; new ones must
  start there.

## 2026-07-05 — [PRODUCT] Everything Graph R1 PR-B: timeline in the site detail card (v1.0.124)

- [T-CLIENT half, primary territory T-DATACORE per the
  cross-territory rule] The /data site detail card now shows "Past 7
  days within 50 km (own archives)": up to 5 cross-stream events
  (alert ⚠ / fire ▲ / gauge ≈ with severity + date) and the traffic
  line (aircraft + vessel archived points over N days). Fetched
  async on card open from /api/data/site-timeline/:id; any failure
  leaves the section absent — the card never degrades. Site geojson
  features now carry the site id (they did not before — the card had
  no way to reference its site).
- Harness 0 hard failures at 390/768/1440; the section is click- and
  data-gated so page-level visuals are unchanged (screenshots
  reviewed; same precedent as the FAA enrichment in #223).
- This closes BUILD ORDER 3 #5. Remaining: #6 anomaly-mining pass
  ([RESEARCH], terminates in filed entries).

## 2026-07-05 — [PRODUCT] Everything Graph R1 PR-A: site event timeline route (BUILD ORDER 3 #5) (v1.0.123)

- [T-DATACORE] First user-visible cross-stream join, composed
  entirely from archives we already record. server/siteTimeline.ts:
  per strategic site, last-7-day NWS alerts + FIRMS fire detections
  + USGS gauge readings within 50 km (event-stream day-files), plus
  our own aircraft/vessel archived-point density per day. ONE scan
  pass computes all sites together, 6h-cached stale-served. Route
  /api/data/site-timeline/:siteId (events capped 12 newest-first —
  stated; zone-only alerts excluded per their stream's honesty;
  absent days absent, never zero; density fairness note: near-site
  traffic archives at full resolution by design, so day-over-day
  comparisons are fair). Battery 4 tests on writer-shaped fixtures.
- BUILD ORDER 3 #4 (CBP container imports) ROUTED AROUND:
  api.census.gov now requires a key on every request (probed —
  porths redirects to missing_key). Free instant signup =
  BLOCKED-FOR-MIKE #6 with build-first analysis (port-authority
  HTML scraping = 2 ports, fragile, materially worse; recommend the
  key). Nothing blocked — the queue moved on per the directive.
- PR-B next: compact timeline section in the /data site detail card
  + harness (client half, own PR).

## 2026-07-05 — [PIPELINE] NOAA CPC degree days (BUILD ORDER 3 #3) (v1.0.122)

- [T-DATACORE] scripts/cpc_degree_days.py: StatesCONUS
  Heating+Cooling daily files per year (keyless, public domain,
  probed 2026-07-05), pipe-delimited parse with format-change refusal
  (missing Region header or non-YYYYMMDD date columns = hard error,
  never guessed), non-numeric cells null never zero, only 2-char
  codes treated as states. Artifact datacore/cpc/degree_days.json
  (0.96 MB), manifest cpcdegreedays.json, battery 3 tests.
- FIRST BUILD CAPTURED: 96 series (48 CONUS states x H/C), 3,837
  daily points each (2016-01-01 .. 2026-07-03), ZERO fetch failures.
  First DEMAND-side weather series in the archive — pairs with the
  supply-side NWS alerts + OWM fields.
- Selection honesty: census-division and fuel-weighted upstream
  variants deliberately not archived in v1 (stated in script +
  manifest).
- Hypothesis (gate-locked): population-weighted degree-day departures
  lead natgas/power demand and utility earnings surprises; joins the
  EIA natgas storage series (#233) naturally — storage delta vs
  degree-day-implied draw is the obvious first gate-2 study.

## 2026-07-05 — [PIPELINE] EIA weekly petroleum + natgas storage (BUILD ORDER 3 #2) (v1.0.121)

- [T-DATACORE] scripts/eia_weekly.py: five keyless hist_xls series
  (US crude ex-SPR / gasoline / distillate / SPR crude stocks +
  lower-48 natgas working storage; all probed 2026-07-05), parsed
  with the proven xlrd comparator pattern. Series titles are READ
  FROM THE SHEETS and stored (title_as_published) — upstream renames
  surface as git diffs, never silent mislabeling; the titles
  confirmed the key naming live (WCSSTUS1 = SPR, as labeled). Gaps
  skipped, never zero; all-series failure refuses to write. Artifact
  datacore/eia/weekly_series.json (0.23 MB), manifest eiaweekly.json.
- FIRST BUILD CAPTURED: 2,283 weekly crude points back to
  1982-08-20; gasoline 1990-; natgas 2010-; all current through
  2026-06-26. 44 years of the energy-regime axis in one pull.
- Cushing deliberately NOT duplicated — the tank-fill comparator
  owns it (one source of truth per series).
- Hypothesis (gate-locked): storage-vs-seasonal-band deltas condition
  the energy regime; also the standing external-truth source for any
  future inventory root.

## 2026-07-05 — [PIPELINE] Fleet utilization v1 + GATE 1 PASS 20/20 (BUILD ORDER 3 #1) (v1.0.120)

- [T-DATACORE] GATE 1 (spine join accuracy) PASSED under criteria
  pre-stated in open_questions.md BEFORE sampling: 20 stratified
  hexes (10 top-count corporate/llc + 10 seeded-random spine hexes)
  vs INDEPENDENT adsbdb registrations -> 20/20 exact N-number
  matches, 0 unresolvable (criteria floor was 90%). First ladder
  gate to PASS tonight after three honest kills.
- FINDING FOR GATE 2 (logged now, before the study): the top
  corporate hexes are dominated by TRUSTEE/LEASING shells (TVPX x3,
  UMB Bank trustee, leasing LLCs) — FAA registrants hide beneficial
  owners, so the utilization x earnings join will need a
  registrant->operator resolution step (or restrict to
  self-registered operators like the airlines). Payload labels
  owners as REGISTRANTS, never "the company flying it".
- server/fleetUtilization.ts: airborne-point sessionization per hex
  (gap > 45 min = new flight; hours = session span sums, LOWER
  BOUNDS under adaptive thinning; ground points excluded), weekly
  Monday buckets (absent weeks stay absent, never zero), per-owner
  aggregation over spine corporations+LLCs; 6h-cached scan;
  /api/data/fleet-utilization (kind: derived, >=2 airframes).
  Battery 4 tests (week bucketing, sessionization ground truth incl.
  a test-fixture bug where the 'second flight' itself exceeded the
  gap — the code was right, the fixture was wrong; spine join with
  ground/non-corporate exclusions; missing-input grace).
- GATE 2 (not attempted): utilization deltas vs 5/20d returns +
  earnings proximity, base-rate-controlled; needs archive depth +
  the registrant->operator step above.

## 2026-07-05 — [RESEARCH] BUILD ORDER 3 filed — fusion-forward (docs)

- [T-DATACORE] Standing directive: queue emptied (Build Order 2 went
  6/6 same-day), so the next order is self-proposed. Emphasis
  deliberately shifts from new roots to FUSION of the archives now in
  hand — the compounding asset is the accumulation, and five of six
  B2 items were ingestion; the graph earns its keep by joining them.
- Order (hypotheses + ladder paths in open_questions.md): (1)
  corporate-fleet utilization series (aircraft archive x entity
  spine — gate-1 criteria to be pre-stated before scoring), (2) EIA
  weekly petroleum + natgas storage (keyless XLS family proven),
  (3) NOAA CPC degree days (probed keyless 2026-07-05), (4) CBP
  container imports (format at build), (5) Everything Graph R1 site
  event timeline ([PRODUCT], existing archives only), (6) [RESEARCH]
  anomaly-mining pass per the angle-hunting mandate.
- Each item its own PR under the established stream rules.

## 2026-07-05 — [PIPELINE] STB EP724 rail stream — BUILD ORDER 2 COMPLETE (v1.0.119)

- [T-DATACORE] #6, the final Build Order 2 item. scripts/stb_rail.py:
  discovers the newest "EP724 Consolidated Data through" workbook on
  stb.gov (format change = hard exit), parses the wide matrix
  (~484 week columns since 2017-03), and writes the compact keyed
  artifact datacore/rail/ep724_carloads.json (0.83 MB: 255 series x
  484 weeks). SELECTION stated in script + manifest, never silent:
  cat 11 Weekly Carloads By 22 Commodity Categories (176 series —
  the volume spine the hypothesis needs), cat 1 System train speed,
  cat 3 cars-on-line; dwell/grain/Chicago metrics deliberately not
  archived in v1. Non-numeric cells -> null, NEVER zero. The source
  republishes full history weekly, so this is a whole-file rebuild —
  nine years of weekly carloads captured in the first build.
- FORMAT VERIFIED AT BUILD (the build-order flag): wide matrix
  confirmed live; openpyxl gotcha encoded (validates by file
  extension); CPKC-merger key discontinuity documented in the
  manifest confidence model.
- LIVE E2E: BNSF coal 18-22k carloads/week latest month — plausible;
  week axis ends 2026-07-01 (current). Battery 5 tests (selection
  filter, null honesty, short-row padding, changed-axis refusal,
  newest-workbook discovery).
- BUILD ORDER 2 SCOREBOARD: 6/6 resolved same-day — entity spine
  (live in prod), tank-fill v3 (honest gate-1 kill + v3.1
  pre-registered), NWS alerts (live), Treasury auctions (live),
  Drought Monitor (live + label-integrity fix), STB rail (this).
  Next session: propose BUILD ORDER 3 per the standing directive, or
  take the GIP queue's next item.

## 2026-07-05 — [PIPELINE] US Drought Monitor stream (BUILD ORDER 2 #5) (v1.0.118)

- [T-DATACORE] server/droughtMonitor.ts: USDM data services (keyless;
  attribution REQUIRED and carried in every payload: "U.S. Drought
  Monitor (NDMC/USDA/NOAA)"). CONUS + 8 ag/water states — states by
  FIPS (probed: abbreviations return empty). Cumulative D0-D4 + the
  USDM's own published DSCI as the one labeled DERIVED field. 24h
  eager poll over a 70-day window, dedup aoi|map_date (maps final on
  publish), per-AOI failure tolerance. Manifest + RAW route
  /api/data/drought.
- LIVE-CAUGHT INTEGRITY BUG before shipping: the aoi=us endpoint
  returns BOTH "CONUS" and "Total" (incl. AK/HI/PR) rows per week —
  the first parse labeled both CONUS, silently mixing two series.
  Fixed: a row is kept only when its OWN label matches the requested
  AOI, never relabeled; regression test pins it. Verified after fix:
  exactly 9 AOIs per map week (was 11 rows mislabeled into 9 keys).
- LIVE E2E: 99 aoi-weeks; 2026-06-30 map: NE DSCI 251 (D2+ 62%!),
  OK 163, CONUS 157, IL 11 — plausible cross-section. Battery 5
  tests (real-fixture normalization + DSCI math, label honesty,
  malformed-row drops, FIPS table pin, dedup+gz).
- Hypothesis (gate-locked): belt drought DELTAS lead ag commodities
  and food-producer margins by weeks; joins USGS gauges + FIRMS on
  the environmental axis.

## 2026-07-05 — [PIPELINE] Treasury auction results stream (BUILD ORDER 2 #4) (v1.0.117)

- [T-DATACORE] server/treasuryAuctions.ts: TreasuryDirect TA_WS
  (keyless, public domain), 6h eager poll over a 30-day window,
  dedup cusip|auction_date (results immutable; reopenings = new
  dates), numeric normalization ('' -> null, never guessed — bills
  carry discount rates, coupons carry yields, each keeps the other
  null). One DERIVED field, labeled: dealer_take = primary dealer
  accepted / competitive accepted. Manifest + RAW route
  /api/data/treasury-auctions.
- HONESTY: the classic tail-vs-when-issued metric needs a paid 1pm
  WI quote — never faked; free stress metrics are bid_to_cover +
  bidder-class shares (manifest states it).
- LIVE E2E: 34 results-complete auctions in the window; dealer takes
  26-38%, plausible. Battery 3 tests (real-fixture normalization,
  pre-result/malformed drops, dedup+reopening+gz).
- Hypothesis (gate-locked): bid-to-cover deterioration + rising
  dealer take precede rate-regime shifts; archive-first, judged
  after depth accumulates.

## 2026-07-05 — [PIPELINE] NWS severe-weather alerts stream + map layer (BUILD ORDER 2 #3) (v1.0.116)

- [T-DATACORE] server/nwsAlerts.ts: api.weather.gov active alerts
  (keyless, public domain, contact User-Agent per policy), 10-min
  poll eager-boot, append-only archive dedup by alert id (CAP
  messages are immutable — updates arrive as new ids), oldest-half
  Set trim (nasaFirms lesson), gz after 2 days. Manifest
  nwsalerts.json. Route /api/data/alerts serves display-simplified
  polygon rings (<=64 pts).
- GEOMETRY HONESTY: zone-coded alerts (geometry:null upstream) are
  archived with null geo and COUNTED (zone_only) in the payload +
  the layer status note ("N zone-coded alerts not drawn") — visible
  cap; zone-polygon resolution filed as follow-up. Archive stores
  centroid+bbox, never full polygons.
- Map layer (T-CLIENT half, same PR per the cross-territory rule):
  fill+line colored by CAP severity (Extreme red / Severe orange /
  Moderate yellow / Minor blue), detail card with not-for-safety-
  of-life note, legend chips, 5-min hidden-gated poll, off by
  default (initial-load budget).
- LIVE E2E: 228 real alerts parsed (4 Extreme / 48 Severe), 40 with
  polygons rendered, 188 zone-only counted; archive wrote 228
  records. Battery: server 3 tests (parse/split, ring decimation
  closure, dedup+gz); harness 0 hard failures at 390/768/1440 —
  first run FAILED on a stale dist bundle (toggle unclickable),
  rebuilt and green; toggle-consistency + legend-parity batteries
  exercised the new layer.
- Hypothesis (gate-locked, from the build order): severe-alert
  clusters over strategic sites lead sector moves by hours-days.
  This PR is display + archive only.

## 2026-07-05 — [REPAIR] Entity spine unreachable in prod — artifact now BUNDLED, not disk-read (v1.0.115)

- [T-DATACORE] Prod verification caught it (the watcher never fired):
  /api/data/aircraft/entity/:hex served entity:null spine_built:false
  through multiple deploys. ROOT CAUSE: loadEntitySpine read
  datacore/aircraft/entity_spine.json from cwd at runtime — works
  locally and in CI (repo = cwd) but the runtime Docker image copies
  dist/, content/, *.py and NEVER datacore/ (frozen Dockerfile), so
  prod could never see the artifact. Every other datacore JSON works
  because it is IMPORTED and bundled into dist — the spine now does
  the same (static import; dist 1.7MB -> 5.7MB, trivial vs the 1GB
  daemon ceiling). Disk-read path retained for tests only.
- RATCHET: new battery test chdirs away from the repo and asserts the
  spine still serves >10k entities — the exact CI blind spot (cwd =
  repo) that let this ship. Would have failed on the old code.
- LESSON ENCODED (module comment + here): server code must NEVER
  disk-read repo files at runtime; bundled import is the only path
  that survives the image. CI green ≠ prod green when the failure is
  in what the image contains — deploy verification is the gate that
  caught this.

## 2026-07-05 — [RESEARCH] Tank-fill v3 GATE 1: FAIL — S1 double-bounce deltas carry no signal; v3.1 pre-registered (v1.0.114)

- [T-DATACORE] PR-2: scripts/tankfill_s1_estimator.py (per-tank
  log10 p95 VV in disk+halo, per-tank series-median self-ratio,
  fill-direction composite, ascending/34 only) + gate-1 via
  tankfill_gate1.py REUSED UNCHANGED. readings_s1.jsonl (61 scenes,
  234/234 tanks median coverage; whole-file rebuild, manifest
  sentinel1readings.json). Battery test_tankfill_s1.py (6,
  synthetic ground truth). Design note stated BEFORE scoring:
  per-tank median normalization cancels exactly in deltas — the
  binding criteria were immune to the normalization choice.
- VERDICT vs pre-stated criteria: n_matched=60 (ample — the
  INSUFFICIENT-SAMPLE branch never triggered), 57 delta pairs,
  reversals present; delta r = +0.056, sign hit = 0.544 -> FAIL.
  Levels r = +0.41: real but UNBINDING and uncredited — it is the
  exact trend-vs-trend inflation pattern that poisoned v1 optical.
- PRIOR vs OUTCOME: stated P(pass) ~25%; outcome in the 75%. Two
  sensors (S2 optical, S1 radar amplitude) and two designs are now
  honestly dead on week-scale Cushing fill from free 10 m imagery.
- v3.1 PRE-REGISTERED (open_questions.md): multi-week deltas
  (28-42d non-overlapping windows) to average down speckle;
  scored ONLY on scenes acquired after 2026-07-05 (out-of-sample by
  time — testing it today on the data that suggested it would be
  fishing); prior discounted to ~15% (attempt #3). If v3.1 fails,
  the free-imagery tank-fill line TERMINATES and sub-meter paid
  imagery (BLOCKED-FOR-MIKE) is the only path.
- Chip acquisition continues (both stacks, pixel-aligned, ~2.5
  PU/scene combined) — raw material compounds regardless.

## 2026-07-05 — [PIPELINE] Tank-fill v3 PR-1: S1 chip client + 24-mo backfill; gate-1 criteria pre-stated (v1.0.113)

- [T-DATACORE] BUILD ORDER 2 #2, first PR of the S1 successor root.
  scripts/cdse_s1_chips.py: discovery via anonymous earth-search
  (sentinel-1-grd; orbit metadata carried — Cushing is ASCENDING
  relative orbit 34 only, S1A+S1D), chips via the CDSE Process API
  (orthorectified SIGMA0_ELLIPSOID, FLOAT32 VV+VH — probed
  double-bounce sigma0 up to ~2500, integer types would clamp), SAME
  bbox as the v2 optical stack, pinned by test_bbox_matches_v2_stack
  so the stacks stay pixel-aligned for fusion. Manifest
  sentinel1chips.json; battery test_cdse_s1_chips.py (7, recorded
  fixture, no network).
- BACKFILL: 61 scenes 2024-07..2026-07, ZERO failures, ~68 PU
  (0.7% of one month's free tier). Spot-check both ends: ground
  median sigma0 0.07-0.10, double-bounce maxima 2100-2535 — the
  physics is present across the whole era.
- CRITERIA + ESTIMATOR DESIGN PRE-STATED in the workup BEFORE any
  scoring (see open_questions.md TANK-FILL v3 STATUS): p95 VV per
  tank disk+halo, log-domain, per-tank self-ratio, fill-direction
  composite = negative normalized double-bounce; gate 1 = same
  criteria as v2 via tankfill_gate1.py reused as-is; matched pairs
  < 20 -> INSUFFICIENT-SAMPLE, not FAIL (12-day repeat may not
  yield 20 adjacent-week deltas). Discounted prior (attempt #2):
  P(pass) ~25%.
- NEXT: PR-2 = estimator + readings stream + gate-1 attempt.

## 2026-07-05 — [PIPELINE] Aircraft entity spine v1 PR-B: artifact SHIPPED + map enrichment (v1.0.112)

- [T-DATACORE] Prod hex list pulled after PR-A deployed (15,248
  distinct archived airframes as of 2026-07-05T08:27Z); spine built
  against the real FAA files: 11,821 matched (78% — the expected US
  share), composition 6,689 corporations / 3,370 LLCs / 935
  individuals / 180 government. Artifact 3.97 MB compact JSON,
  committed at datacore/aircraft/entity_spine.json; manifest
  aircraftspine.json. Sample verified: a6faee -> N549SC, SPARTAN
  EDUCATION LLC, PIPER PA-28-181 (2020) — a flight school owning the
  archive's most-seen airframe, exactly as base rates predict.
- Map enrichment: aircraft detail card now shows "Registered:
  owner · mfr model · year — N-number, FAA registry" via
  /api/data/aircraft/entity/:hex, async after open; non-US hexes show
  nothing extra (never guessed). Harness 0 hard failures at
  390/768/1440; change is click-gated so page visuals are unchanged
  (screenshots reviewed).
- NEXT for this root: gate 1 = join accuracy vs known corporate
  tails; then the utilization x earnings study (BUILD ORDER 2 #1
  hypothesis). Monthly artifact refresh is a session task (manifest
  cadence).

## 2026-07-05 — [PIPELINE] Aircraft entity spine v1 PR-A: hex enumeration + FAA join tooling (v1.0.111)

- [T-DATACORE] BUILD ORDER 2 #1, first of two PRs.
  server/aircraftEntities.ts: distinct archived icao24s streamed
  gz-aware from the aircraft archive (first/last seen, counts,
  callsigns capped at 5, latest type designator; 6h TTL cache,
  stale-served) + spine artifact serving with per-record evidence
  envelopes. Routes: /api/data/aircraft/hexes (the join-key list) and
  /api/data/aircraft/entity/:hex (degrades to entity:null until the
  artifact ships). scripts/build_entity_spine.py: FAA Releasable
  MASTER.txt x ACFTREF.txt join by exact Mode S hex, header-name
  addressing (refuses changed formats), US-gov public domain; ONLY
  archived hexes emitted — never the 300k-row dump.
- GOTCHAS ENCODED: registry.faa.gov 403s non-browser User-Agents
  (probed; browser UA in the script); FAA header cells carry a BOM +
  stray spaces (caught by the verbatim-fixture test).
- E2E SMOKE on the real FAA files (73 MB, 314k rows): 3-hex list ->
  2 matched (SOUTHWEST AIRLINES CO N1801U/737-8 corporate; an LLC's
  M20C), 1 unmatched (non-US, correctly null). Tests: 5 pytest
  (verbatim FAA fixture rows) + 5 node (fold semantics, gz archive
  scan, TTL contract, artifact degradation).
- PR-B after Railway deploys this: pull prod hex list, build + commit
  the spine artifact, live-verify /entity/:hex, then wire the map
  aircraft detail card to it (owner/model/registrant enrichment).
- HYPOTHESIS (unchanged from the build order): corporate-fleet
  utilization x earnings timing; the spine is the join substrate.

## 2026-07-05 — [RESEARCH] BUILD ORDER 2 filed (self-proposed per the standing directive — build order 1 fully resolved) (docs)

- [T-DATACORE] Standing directive 2026-07-05: "when the wishlist is
  empty, generate the next wishlist yourself." Build order 1 is fully
  resolved (streams #1-#8: 6 built, 1 superseded, 1 gate-1 killed;
  trains repair, perf repair, tank-fill workup through its gate-1
  verdict). BUILD ORDER 2 filed in open_questions.md with per-stream
  trading hypotheses and ladder paths; access LIVE-PROBED keyless
  2026-07-05 for the three new externals (api.weather.gov GeoJSON,
  TreasuryDirect TA_WS JSON, Drought Monitor CSV — all responded).
- Order: (1) aircraft entity spine v1 (GIP queue promotion — queued
  work outranks new streams; fleet-utilization x earnings
  hypothesis), (2) tank-fill v3 S1 gate-1 pipeline (successor root,
  discounted prior ~25%), (3) NWS alerts, (4) Treasury auctions,
  (5) Drought Monitor, (6) STB rail traffic.
- Each ships as its own PR under the stream rules (licensing first,
  archive from day one, envelope manifest, RAW until gate 2).

## 2026-07-05 — [RESEARCH] Tank-fill v2 GATE 1: FAIL (layer of death: DATA) — 24-mo backfill + EIA validation (v1.0.110)

- [T-DATACORE] PR-4 of the build plan. Backfill: 99 master chips
  2024-07..2026-07, ZERO failures, ~139 PU total (1.4% of one month's
  free tier). Estimator swept all chips -> 99 readings_v2 lines
  (89 usable scenes, rest cloud-skipped; registration (0,0)
  everywhere — S2 L2A ortho is that good). Gate scorer:
  scripts/tankfill_gate1.py with matching rules + criteria pinned by
  test_tankfill_gate1.py BEFORE the run (prior logged in the
  v1.0.109 entry: P(pass) ~40%).
- VERDICT vs pre-stated criteria (>=20 matched weeks w/ reversal,
  delta r >= +0.3, delta-sign >= 65%): n_matched=72 (ample, reversals
  present), delta r = -0.06, sign hit = 0.50 (n=64), levels r = +0.11
  -> FAIL. Winter/shoulder split (sun_elev < 50, the workup's
  predicted signal carrier): delta r = -0.284 (WRONG direction),
  sign hit 0.42 -> FAIL harder. This is a real fail on a big sample,
  not a thin-sample fail.
- CONFOUND QUANTIFIED: composite vs sun_elev r = +0.40 (+0.0034
  fill/deg) — the tan-zenith inversion does NOT fully remove the sun
  artifact (the exact v1 poison, now measured per-tank). BUT the
  confound is not the whole story: sun-residualized composites
  (post-hoc DIAGNOSTIC, not a gate re-score) still show delta r
  -0.06, sign hit 0.42 — after removing the sun trend there is NO
  residual inventory signal at all. The 10 m optical sub-pixel
  crescent method does not track Cushing inventories. Period.
- LADDER CONSEQUENCES: root DEAD at gate 1 (DATA — reading fails
  external-truth verification vs EIA). Build-plan PR-5/6/7 (weekly
  cadence surface, gate-2 study, unlock) CANCELLED — they presumed a
  pass. Nothing surfaces on /data (gate rule held: no fill% was ever
  shown). readings_v2 production PAUSED — no further appends from a
  falsified estimator. CHIP ACQUISITION CONTINUES weekly (raw
  material is cheap at 1.4 PU/scene, the archive never refills, and
  the S1 successor needs coincident optical pairs).
- SUCCESSOR ROOT FILED (open_questions.md): Sentinel-1 SAR
  double-bounce tank-fill — different physics (roof-wall corner
  reflection scales with roof depth; cloud-immune; ~2x cadence;
  CDSE creds already proven on S1). Discounted prior: this is
  attempt #2 on the same target (Reasoning Standard #4) — P(gate 1)
  ~25%. Criteria to be pre-stated in its own workup before any run.
- HONESTY NOTE: the negative result is the product here — "10 m free
  optical cannot read Cushing fill" is now a COSTED, evidence-backed
  boundary (99 scenes, 72 matched weeks) that no future session
  needs to re-learn, and it sharpens the BLOCKED-FOR-MIKE sub-meter
  imagery entry (that purchase is now the only optical path).

## 2026-07-05 — [PIPELINE] Tank-fill v2 PR-3: crescent estimator BUILT + run on the first real chip (v1.0.109)

- [T-DATACORE] scripts/tankfill_estimator.py: per-tank up-sun vs
  down-sun coverage-weighted B04 ratio (self-normalizing — paint,
  band, atmosphere cancel), inverted through small-s circle-lens
  geometry (f = 4s/(piR); s -> depth via tan(sun_elev)) to fill %;
  SCL per-pixel cloud masking (tank >10% masked skipped, scene >40%
  skipped whole); integer registration vs a reference chip (+/-3 px
  Pearson search); D^2-weighted site aggregates; readings_v2.jsonl
  one line/scene with assumptions (shadow_k=0.35, API-650 heights)
  and per-tank q flags carried on every record. Manifest
  sentinel2v2.json. Tests: test_tankfill_estimator.py (12, fully
  synthetic supersampled rasters with KNOWN crescent geometry —
  depth recovery + ordering, full-tank reads 1.0, registration
  recovery restores measurements, masking, D^2 math, subpixel flag,
  registry load).
- TWO MEASUREMENT DEFECTS CAUGHT BY THE SYNTHETIC BATTERY before any
  real data was trusted: (a) a 1 px edge margin (the "obvious" mixed-
  pixel hygiene) discards exactly the rim pixels carrying the
  crescent — a 0.5 px-reach crescent read ZERO; replaced with
  supersampled coverage WEIGHTS, no margin. (b) assigning whole
  pixels to halves by center puts the proj==0 boundary line in one
  half -> ~5% false darkening on a synthetic FULL tank (fill read
  0.81); fractional half-membership fixed it (full tank reads 1.00).
  Both are exactly the class of silent bias gate-1 would have eaten.
- FIRST REAL READING (June 17 chip, sun_elev 71.1): 234/234 tanks
  measured, reg (0,0); site fill_d2w: enbridge 0.42, hub 0.59,
  plains 0.65, ring 0.58. ALL 234 flagged q=subpixel (June reach
  0.50 px — the v1-poisoning artifact, now labeled per tank, never
  hidden). Winter scenes are the signal carriers; PR-4 backfill will
  weight accordingly.
- PRIOR (stated before gate-1, Reasoning Standard #10): P(gate-1
  PASS) ~40% as filed in the workup; sub-pixel summer readings are
  expected to contribute ~nothing — if gate-1 passes it will be on
  winter/shoulder deltas.
- NEXT: PR-4 = 24-month chip backfill (~140 PU, split across runs)
  + gate-1 attempt vs EIA weekly Cushing stocks (criteria pre-stated
  in the workup: >=20 matched weeks, >=1 reversal, delta r >= +0.3,
  delta-sign >= 65%).

## 2026-07-05 — [PIPELINE] Tank-fill v2 PR-2: CDSE master-chip client BUILT + live-verified (v1.0.108)

- [T-DATACORE] scripts/cdse_chips.py: one 5-band UINT16 GeoTIFF master
  chip per usable Sentinel-2 scene over the Cushing tank registry.
  Discovery via Element84 earth-search (ANONYMOUS — same source as v1;
  dates, cloud %, sun angles); chips via the Sentinel Hub Process API
  on CDSE (credentialed, free tier). Committed metadata:
  datacore/sentinel2/chips_index.jsonl (dedup by scene, per-scene
  est_pu, monthly PU accounting with a hard 50%-of-free-tier refusal);
  chip binaries gitignored (~1.6 MB/scene, regenerable). Manifest:
  datacore/manifests/sentinel2chips.json. Tests: test_cdse_chips.py
  (7, no network; recorded 2026-07-05 earth-search response).
- WORKUP CORRECTION 1 (bbox): the filed workup chip bbox
  [-96.80,35.90,-96.72,35.98] MISSES 20 of 234 measurable tanks
  (registry extends east to lon -96.7149). Corrected CHIP_BBOX
  [-96.770,35.922,-96.712,35.960] covers all measurable tanks with
  >=250 m margin at ~1/3 the pixel area — 1.40 PU/scene vs the
  workup's ~4.1. RATCHET: test_chip_bbox_covers_every_measurable_tank
  pins coverage against the live registry file.
- WORKUP CORRECTION 2 (decode): pillow CANNOT read 5-sample/pixel
  TIFFs ("pillow decodes it" was wrong); tifffile reads (H,W,5) uint16
  directly — session-local dep, documented in the script for PR-3.
- WORKUP CORRECTION 3 (sun angles): the CDSE SH Catalog does NOT
  expose view:sun_elevation/azimuth at all (probed live; fields.include
  returns nothing) — discovery stays on earth-search, which carries
  both angles per scene.
- LIVE E2E (2026-07-05): token OK; 4 usable scenes since 2026-06-01;
  pulled S2B_14SPE_20260617 (cloud 0.08%): 523x420x5 uint16, plausible
  DN ranges (B08 NIR mean 3124), SCL classes {2,4,5,6,7}; index record
  written; 1.4 of 10,000 monthly PU spent.
- NEXT: PR-3 crescent estimator (sun-sector/anti-sun-sector sub-pixel
  ratio over the registry tanks, readings_v2 schema), then PR-4
  backfill + gate-1 attempt (criteria pre-stated in the workup).

## 2026-07-05 — [REPAIR] Audit defects #2/#9/#10 closed: trains health override, sentinel2 staleness surface, silent-cap edges (v1.0.107)

- #2: the layers registry statically claimed trains "live" through the
  entire outage. /api/data/layers now health-overrides trains to
  status "down" (+note) when the cache is >45 min stale or no source
  is ok (the eager tick refreshes every 10 min, so staleness = real
  outage); the client renders "feed down" red and auto-disables the
  toggle — a dead feed never advertises live again.
- #9: sentinel2 readings are git-side/session-run — a stall was
  invisible. platformStats now exposes sentinel2_last_reading +
  age_days so every DAILY check and dashboard sees staleness.
- #10a: edgar13f with a NULL entryTotal that parses exactly to the
  250 cap can no longer ship 250 rows that look complete — treated as
  over-cap (summary-only, holdingsOmitted=true) per the never-silent
  cap doctrine.
- #10b: nasaFirms bounded its dedup memory with a full clear() — in
  peak fire season that forgets the entire ~3h NRT window at once and
  re-appends it as duplicates. Now trims the OLDEST half (insertion
  order), keeping the recent window intact.
- ALL TEN audit defects from the 2026-07-05 quality audit are now
  closed (#209 #210 #211 #212 #214 #216 + this). Harness green; suites
  green.

## 2026-07-05 — [REPAIR] Manifest accuracy: aircraft field_map corrected + COT manifest created (docs)

- Audit defects #7 + #8 (both manifest-accuracy, one logical change:
  the reader contract must match the writers):
- aircraft.json documented a "gs" field that doesn't exist — the
  writer emits g (on-ground flag) and v (speed m/s, rounded). Fixed.
- The COT stream (Python-side, routine #191) had NO manifest and
  escapes the envelope test (which only scrapes server/*.ts writers).
  datacore/manifests/cot.json now documents it honestly: SINGLE keyed
  JSON with atomic replace (an explicit exception to append-only,
  justified by the keyed weekly-history shape), full per-week field
  map from cftc_cot.py _derive_fields, DERIVED fields labeled, the
  3-days-stale-by-design publication lag stated, and a note that
  Python-side archives are documented by convention rather than
  enforced by the TS envelope test.

## 2026-07-05 — [RESEARCH] Stream #8 gate 1: FAIL — Google Trends via pytrends (layer of death: DATA/access)

- PRIOR (filed before the run): gate-1 stability test may kill the
  stream; PASS = median cross-pull r > 0.95 AND >= 80% of the 20-term
  panel ok every round.
- RUN 1 finding: pytrends (upstream ARCHIVED 2025-04) is
  dependency-rotten — with retries enabled it dies on urllib3 v2
  (Retry method_whitelist TypeError) before reaching Google at all.
  A retries-free control pull SUCCEEDED, so run 1 measured our probe
  config, not the source; the probe was corrected and rerun (rigor:
  the first FAIL would have been a false attribution).
- RUN 2 (corrected, 3 rounds x 20 brand terms, 8-min gaps): rounds
  1-2 = 20/20 ok; STABILITY PASSED emphatically — median cross-pull
  r = 0.998, min 0.952, no unstable term. But round 3 collapsed to
  6/20 on HTTP 429s: Google rate-limits the unofficial path after
  ~45 pulls/35min. VERDICT: FAIL on the availability half of the
  pre-stated criterion.
- HONEST SHAPE OF THE DEATH: the DATA is reproducible; the free
  ACCESS is not production-reliable, and the only client library is
  abandoned. NO archiver, NO manifest, NO daemon route built (as
  planned — no dead code for a dead stream). Residual option filed in
  open_questions: an ultra-low-cadence weekly single-round pull
  (20 pulls/week sits far under the observed limit) could be re-probed
  if a hypothesis ever NEEDS Trends; paid alternatives remain in
  BLOCKED-FOR-MIKE. Probe: scripts/gtrends_probe.py (re-runnable).

## 2026-07-05 — [REPAIR] fredmacro vintage dedup: restart-bloat + revert-drop fixed (v1.0.106) [T-DATACORE]

- Audit defect #6, both halves: (a) seedSeen covered 3 days while each
  poll fetches 120 days — a restart >3d after backfill re-appended
  ~120d x 31 series as duplicate vintage rows with fresh rt; (b) the
  (s,d,v) SET dedup silently dropped a revision that REVERTS to a
  previously seen value — but a revert IS a vintage transition.
- FIX: dedup is now LATEST-VALUE per (s,d) — seed the current value
  from up to 130 days of files (oldest->newest so the last write
  wins), append only when the published value CHANGES. Duplicates die;
  every transition including reverts is recorded. Manifest cadence
  line updated to state the semantics.
- RATCHET: honest restart simulation via a real state-reset hook
  (in-memory map cleared, disk seed must carry it) + a
  revise-then-revert sequence test. 8/8 suite green.

## 2026-07-05 — [PIPELINE] Tank-fill v2 PR-1: Cushing tank registry from OSM (v1.0.105) [T-DATACORE]

- First build step of the filed tank-fill v2 workup: 333 storage-tank
  polygons pulled from OSM Overpass -> datacore/sentinel2/
  cushing_tanks.geojson (git-versioned fixed geometry for the
  crescent-shadow estimator). Per tank: center, equivalent-circle
  diameter from polygon area, site assignment (3 tank_farm sites +
  ring), API-650 48ft default height with an explicit provenance flag
  (OSM has zero height tags here — assumptions labeled, never silent),
  measurable_10m flag (234 tanks >= 40m).
- REGISTRATION VALIDATION (the workup's sanity check): computed ring
  shell capacity = 74.9M bbl vs EIA's ~76M bbl published Cushing
  working capacity — the geometry + default height reproduce the known
  ring within ~1.5%. Strong evidence the registry is sound before any
  imagery sampling happens.
- ODbL attribution in the file + pinned by test (counts, provenance
  flags, bbox bounds, capacity plausibility). Builder script
  re-runnable: scripts/build_tank_registry.py.
- Next per the workup: PR-2 CDSE chip client (fixtures in CI), PR-3
  crescent estimator.

## 2026-07-05 — [REPAIR] earnings8k manifest drift: acceptanceDatetime + ticker now actually stored (v1.0.104) [T-DATACORE]

- Audit defect #5 — the honesty-critical one: the manifest documented
  acceptanceDatetime ("lookahead-free event time") and a ticker entity
  key that the writer NEVER stored; gate-2 work reading the manifest
  would have assumed a timestamp that didn't exist (filedAt is a date
  only). Manifests are the reader contract — drift there poisons
  downstream honesty silently.
- FIX (writer side, not a docs downgrade): acceptanceDatetime = the
  getcurrent entry's <updated> timestamp (when this feed made the
  filing publicly visible — the honest "knowable" time, ISO w/offset);
  ticker = EXACT numeric-CIK match vs SEC company_tickers.json (24h
  cached; failed/empty fetches never cached; unlisted filers stay
  null — never guessed). Manifest wording tightened to say precisely
  what the fields are.
- 2 ratchet tests (feed <updated> capture; CIK map exactness) +
  cache-pollution fix found by the tests themselves (a failed tickers
  fetch was being cached 24h). 11/11 suite green.

## 2026-07-05 — [REPAIR] Optionchains: crash-safe day claim, gzip lifecycle, holiday skip (v1.0.103) [T-DATACORE]

- Audit defect #4, three parts: (a) .last_run_day was claimed BEFORE
  the run — any crash/total failure permanently lost that trading day
  (forward-only archive; a lost day can never be re-bought). Now an
  in-memory guard prevents double-fire and the day is claimed AFTER a
  run that wasn't a total failure (shouldClaimDay: empty universe
  claims, partial failure claims + logs, all-failed retries next
  hourly tick). (b) The manifest promised .jsonl(.gz) but nothing
  gzipped the dir (~3-5MB/day raw) — gzipOldChainDays now runs the
  standard 2-day lifecycle after each run. (c) The archiver ran on
  July 3 (full NYSE closure), burning ~120 API calls to archive stale
  quotes as a fresh day — shouldRunNow now skips holidays PARSED from
  market_calendar.py (frozen source of truth read at boot, never
  duplicated — December's year-add flows through automatically; parse
  failure degrades to weekend-only, never a crash).
- 4 new tests incl. parsing the real market_calendar.py (July 3
  present), the exact holiday-evening skip, claim semantics, and the
  gzip round-trip. 10/10 suite green.

## 2026-07-05 — [REPAIR] archiveStats enumerates from disk — archive-gap rule now covers every kind (v1.0.102) [T-DATACORE]

- Audit defect #3: archiveStats() hardcoded six position kinds, so
  fires, filings, earnings8k, filings13f, fredmacro, optionchains and
  every new stream (usaspending, fda, usgswater, gdelt) were INVISIBLE
  to /api/data/archive/stats — a stalled key or dead archiver would be
  discovered by accident, and the archive-gap rule was unenforceable
  for most of the archive.
- FIX: enumerate directories from disk (new streams appear with ZERO
  code change), keep the position kinds explicitly listed so they
  report {files:0} loudly before first write, skip non-files.
- Side benefit: the landing hero's streams_recording count now grows
  automatically as the new stream dirs land on the volume.
- RATCHET: test creates a never-before-seen stream dir and asserts it
  appears in stats without a code change.

## 2026-07-05 — [REPAIR] Eager archive tick: aircraft + trains no longer visitor-dependent (v1.0.101) [T-DATACORE]

- Audit defect #1 (the top finding): aircraft and trains archiving was
  REQUEST-driven — archiveAircraft/archiveTrains run as fetch side
  effects, so no visitors = no archive. 11 hourly gaps each were
  already permanent in the archive's first 36h, and the record was
  visitor-BIASED (only what someone happened to view). Vessels had the
  eager fix (KNOWN BROKEN #9); these two didn't.
- FIX: a 10-minute tick fires one aircraft snapshot rotating across
  four strategic-site regions (Cushing/south-central, US NE corridor,
  LA/Long Beach, Rotterdam ARA — each region ~every 40 min under the
  point-radius API cap) plus one global trains snapshot (which also
  keeps trainsCache warm for visitors). One extra upstream call per
  10 min per feed. Fires once at boot, not after the first interval.
- RATCHET: source-pin test (regions >= 3, both feeds, 10-min cadence,
  boot-fire) so the tick can't be silently de-scoped.

## 2026-07-05 — [PRODUCT] River-gauge /data layer — stream #6's map surface (v1.0.100) [T-CLIENT + shared registry]

- The USGS stream's geographic surface, per the legend same-PR rule:
  vt-gauge registry icon (staff gauge + water wave, SDF), rivergauges
  registry entry (environmental group, OFF by default — reference
  layer, initial-load budget respected), datamap effect (hourly
  hidden-gated refresh matching the server poll; detail card shows
  reading + provisional/approved label + USGS monitoring link;
  discharge-only gauges labeled ft3/s), legend entry gated on enable.
- Harness: registry + /api/data/rivergauges fixtures; toggle-
  consistency (1440 exercises every live layer incl. gauges) + legend
  parity + perf gates all green 10/10.

## 2026-07-05 — [PIPELINE] Stream #7: GDELT facility-event alerts (v1.0.99) [T-DATACORE]

- Built per the verified brief: server/gdeltEvents.ts — 15-min Events
  export files via the lastupdate.txt pointer (HTTP ONLY — the host's
  HTTPS cert is invalid, verified; acknowledged in the manifest),
  fflate unzip, column indices pinned against a real export, CAMEO
  unrest/strike filter (roots 14/17/18/19/20 + 143x) x ~0.5° boxes
  around the 16 strategic sites -> KB-scale archive, dedup by
  GlobalEventID, republished-file skip, 15-min eager poll,
  /api/data/facility-events (48h rolling window), envelope manifest.
- HONESTY encoded: media event MENTIONS with city/ADM-approximate geo
  (never facility-exact); CAMEO cannot see clean industrial accidents
  (FIRMS/AIS are the physical sensors) — hypothesis is unrest/strike
  bursts as verification prompts, gate-2 = burst->own-sensor
  confirmation rate. GDELT attribution required and carried.
- LIVE E2E: real export downloaded/unzipped/parsed; 0 matched events
  this window (holiday news cycle near 16 sites — the tight filter
  working as designed, not a defect).
- 5 tests; offline pytest green. New dep: fflate (pure-JS zip).
- Stream #8 (pytrends) is a PROBE next, not a build — per the
  downgraded plan after upstream abandonment.

## 2026-07-05 — [PIPELINE] Stream #6: USGS river gauges — barge-corridor water levels (v1.0.98) [T-DATACORE]

- Built per the verified brief: server/usgsWater.ts — 14 live-verified
  gauges (Mississippi St. Louis->Belle Chasse + Missouri/Illinois
  tributaries + 4 Ohio R gauges; dead Metropolis gauge excluded),
  BOTH parameter codes requested (Memphis/Vicksburg are
  discharge-only — verified), one request covers all sites, 1h eager
  poll, /api/data/rivergauges route, envelope manifest.
- Vintage discipline: provisional (P) -> approved (A) revisions append
  as new rows with rt. USGS -999999 sentinels dropped. Low-water
  barge-stress SIGNAL stays gate-2-locked — this is RAW readings only.
- LIVE E2E: 26 readings across the 14 sites (several publish both
  params; Ironton shows a negative discharge — real backwater
  behavior, archived as-is per RAW doctrine).
- QUEUED (own [PRODUCT] PR per the legend same-PR rule): the /data
  registry layer + gauge map icons + legend entry — gauges carry
  lat/lon and are map-plottable.
- 5 tests; offline pytest green.

## 2026-07-05 — [PIPELINE] Stream #5: FDA binary events — approvals + AdCom dates (v1.0.97) [T-DATACORE]

- Built per the verified brief: server/fdaEvents.ts — openFDA drugsfda
  approvals (30-day rolling window) + Federal Register FDA advisory-
  committee meeting notices (the FREE forward-looking catalyst that
  preserves the IV-ramp hypothesis; PDUFA dates are legally
  unpublishable, 21 CFR 314.430 — stated in manifest + route, and we
  do NOT scrape aggregator calendars). 6h poll, 2 req/cycle (far under
  keyless caps), /api/data/fda-events, envelope manifest.
- Meeting dates parsed from official notice text with a CONFIDENCE
  LABEL (parsed/unparsed) — an unparsed date stays null, never
  guessed. pub (FR publication date) = when the public could know.
- LIVE E2E caught an honesty bug BEFORE ship: openFDA returns whole
  applications with full submission history, so the naive parse
  emitted 1,619 "approvals" including years-old supplements; the
  window filter drops out-of-window AP submissions -> 106 in-window
  approvals + 40 adcom notices (13 with parsed dates). Regression test
  pins the window.
- LADDER: gate 1 = adcom date accuracy vs 20 known events (runnable
  from the archive + FR links); gate 2 = IV-ramp reproducibility
  around parsed adcom dates on OUR archived option chains (recording
  since 2026-07-06). 9 tests green; offline pytest green.

## 2026-07-05 — [PIPELINE] Stream #4: USAspending contract-awards archiver (v1.0.96) [T-DATACORE]

- Built per the verified brief: server/usaSpending.ts — transaction-
  level search polled by last_modified_date (the publication axis),
  EXPLICIT $25k floor applied client-side via two sorted passes with
  early stop (the API's own amount filter is award-LIFETIME — trap),
  deobligations kept symmetrically, dedup (aid,mod,amt) with FPDS
  corrections appending as vintage rows, 6h eager poll,
  /api/data/contracts route, envelope manifest.
- TICKER MAPPING precision-first: SEC company_tickers.json exact
  normalized-name match (ambiguous normalized names DROPPED — never
  guess), award-detail parent for large unmatched rows (the
  recipient-profile endpoint is banned: vintage-less, provably wrong
  parents), persistent UEI->ticker cache that compounds forever.
  Unmatched rows archive tkr:null and are skipped by consumers.
- HONESTY ENCODED: rt is the only event date (action_date = signature
  date); DoD/USACE publish ~90 days late — manifest + route note carry
  it; gate 2 must cohort/exclude DoD. DUNS never stored (D&B
  proprietary); UEI only.
- LIVE E2E: 308 real txns >= $25k pulled over the 2-day holiday
  window; 23 name/cache-matched (GM resolved by name, then served
  from cache within the same run — the compounding works); 8 queued
  for parent lookups; all 308 archived and deduped on re-run.
- LADDER: gate 1 (recipient->ticker precision on a 50-award
  hand-check) is now RUNNABLE from the archive alone (mm + mname audit
  fields); not yet attempted. Gate 2 blocked on gate 1 + return
  windows. 8 tests green; offline pytest suite green.

## 2026-07-05 — [RESEARCH] Parallel subagent batch: streams 4-8 verified, tank-fill v2 workup, datacore quality audit (docs)

- Throughput directive executed: five subagents ran while the main
  thread shipped repairs. Filings landed in open_questions.md +
  wishlist BLOCKED-FOR-MIKE; summaries:
- USAspending (#4, next build): endpoints live-verified incl. a $900M
  DOE award to a Centrus (LEU) subsidiary as the hypothesis exemplar;
  DoD publishes ~90 DAYS late (gate-2 cohort/exclude); $25k explicit
  cap = 99.74% of dollars in 20% of rows; DUNS is D&B-proprietary —
  UEI only; recipient-profile parent endpoint is vintage-less and
  WRONG (two proven cases) — parents from award-detail only.
- FDA (#5): PDUFA dates not freely publishable (21 CFR 314.430) —
  free substitute preserves the IV-ramp hypothesis: Federal Register
  AdCom meeting notices (verified) + openFDA approvals.
- USGS (#6): 14 gauges live-verified (Memphis/Vicksburg are
  discharge-only; one dead gauge caught); gets a /data layer.
- GDELT (#7): 15-min export files trivially small after
  CAMEO x facility-bbox filter; host is HTTP-only; CAMEO can't see
  industrial accidents — hypothesis reworded to unrest/strikes.
- pytrends (#8): upstream ARCHIVED 2025-04 — downgraded to a gate-1
  probe; no archiver/manifest unless stability passes.
- Tank-fill v2: full crescent-shadow workup filed (sub-pixel aggregate
  estimator, OSM 333-tank registry verified, 0.4% of CDSE free quota,
  delta-based gate-1 criteria fixing the v1 trend inflation, 7-PR
  build plan). BLOCKED-FOR-MIKE: nothing in the core build.
- Quality audit: 10 prioritized defects filed as the DATACORE DEFECT
  QUEUE in open_questions (top: request-driven aircraft/trains
  archiving = permanent gaps; archiveStats blind to 8 of 11 kinds;
  optionchains loses a day on crashed runs).

## 2026-07-05 — [REPAIR] /data map performance 3/3: aircraft low-zoom render decimation (v1.0.95) [T-CLIENT]

- Two-layer split on ONE source: aircraft-sym (full, minzoom 4.5) +
  aircraft-sym-lo (maxzoom 4.5, stable rank-hash filter keeps 35%).
  At the default continent zoom, 10k overlapping icons were pure
  overdraw; zooming past 4.5 shows every aircraft. Rank hashed from
  icao24 so a given aircraft never flickers in/out across refreshes.
  Click/cursor handlers wired to both layers; teardown removes both.
- NO DATA LOSS by construction and by gate: the source always holds
  the full feed — the 2/3 gate's data-richness guard read 10,000
  unique source features while rendering 3,507. This is the profiled
  fill-rate fix (rendering path), not a data cut.
- Measured (harness, SwiftShader): median frame 117 -> 83ms @1440,
  83 -> 67ms @768, 33 -> 17ms @390; p95 200 -> 117ms @1440; sampled
  frames per pan window 66 -> 116 @390 (smoother). 10/10 green under
  the v1.0.94 calibrated gates.
- Harness aircraft samplers now query both layers (mechanical layer-id
  adaptation; gate thresholds untouched).
- Queued: apply the same split (or clustering) to fires BEFORE the
  FIRMS key lands — fires has no low-zoom mitigation and can't be
  harness-verified until its fixture carries data (noted, not built).

## 2026-07-05 — [RULE-REVIEW] /data map performance 2/3: calibrated perf gate in the harness (v1.0.94) [T-CLIENT tooling]

- Measurement change, own PR per MEASUREMENT INTEGRITY. Performance
  regressions now FAIL the harness like visual ones: map-page TTI
  gate 3000ms (observed ceiling ~1.3s), per-width median-frame gates
  120/200/250ms @390/768/1440 (observed 33-117), p95 gate 350ms
  (observed ceiling 200; warn band from 250), all under SwiftShader
  as regression guards — the S24 remains the on-device acceptance
  ruler per DESIGN.md.
- Metric before vs after on identical inputs: same TTI/median/p95
  numbers measured, only thresholds tightened — the change can only
  make builds FAIL more, never look better (bias direction stated).
- DATA-RICHNESS GUARD: at 1440 the aircraft SOURCE must hold >=9500
  unique features (deduped by icao24) regardless of rendered count —
  the enabler for low-zoom decimation (3/3) and the ban on doing it
  by dropping data.
- Deliberately NOT added: a payload-bytes budget — the harness fixture
  server doesn't run the prod compression middleware, so it would
  measure the wrong thing; compression is pinned by the express
  round-trip test in server/compression.test.ts instead.

## 2026-07-05 — [REPAIR] /data map performance 1/3: eliminate redundant network + render work (v1.0.93) [T-CLIENT+server/index.ts]

- Priority-repair directive: map slow to initially load, sluggish with
  all layers on. Constraint honored: NO layer removed, NO data
  richness reduced — engineering only. Profiled first (subagent
  report); this PR ships the waste-elimination tier (biggest win per
  unit risk); the harness perf GATE ships separately next (measurement
  change = own PR), then low-zoom decimation behind that gate.
- SERVER: response compression (compression middleware in
  server/index.ts — Railway's edge does NOT gzip): aircraft snapshot
  ~0.8MB -> ~120KB, powerplants 800KB -> ~200KB; ~70% of initial-load
  bytes cut. Default filter skips the already-compressed wxtile PNGs.
  Pinned by an end-to-end gzip round-trip test.
- CLIENT (datamap.tsx): (1) moveend fetches debounced 400ms — bare
  moveend fired a full fetch + 10k-feature rebuild on EVERY camera
  settle including each wheel step; (2) hidden-tab gating on
  aircraft/vessels/trains/insider/earnings polls + immediate refresh
  on visibilitychange return; (3) setStatus no-op bail — five default
  polls re-rendered the whole page every 15-60s tick with identical
  payloads; (4) insider/earnings panel-count polls 60s -> 300s (they
  render a count; server caches upstream at 15-min); (5) map event
  handlers de-duplicated — click/mouseenter/mouseleave were stacked on
  every toggle cycle (N clicks -> N detail cards + N trail fetches).
- Harness: 10/10 green at 390/768/1440; fields-on battery's mounted
  wait extended to require BOTH arrows AND temp labels placed (fixed a
  1440 placement-timing flake my re-render reduction exposed — labels
  were visibly rendered in the screenshot while the sampler read 0).
- DEPLOY-VERIFY of v1.0.92 (previous entry): prod /api/data/trains
  responds 200 in 0.8s with 126 live trains (FI 86 ok + NO 40 ok) —
  the permanent-hang repair holds in production.

## 2026-07-05 — [REPAIR] /api/data/trains permanent hang: stuck in-flight promise poisoned the route (v1.0.92) [T-DATACORE]

- Production: /api/data/trains returned NOTHING (HTTP 000 at 90s, zero
  bytes) while every other endpoint was healthy; /data showed the
  trains layer erroring. Archive evidence: trains recorded fine
  through the 05:00 UTC hour, dead after — so the feed worked, then
  ONE fetch wedged.
- ROOT CAUSE (architecture, not upstream): the route shares one
  in-flight promise across requests (`.finally()` clears it) — but
  .finally only fires when the promise SETTLES. One fetchTrains stuck
  past its per-source AbortSignal timeouts (upstream/undici pathology)
  → every subsequent request awaited the same dead promise, forever,
  surviving upstream recovery. /api/data/aircraft carried the
  IDENTICAL latent pattern per bbox key.
- Also found while diagnosing: Digitraffic now REQUIRES
  Accept-Encoding: gzip (406 otherwise — verified live). Undici sends
  it by default, but the header is now explicit so a runtime/bundler
  change can't silently 406 the feed.
- FIX (server/routeGuards.ts, pure + unit-tested): (1) raceDeadline —
  no request waits past 15s; falls to stale-beats-spinner; (2) slot
  expiry — an in-flight older than 45s is abandoned and a fresh fetch
  starts (one stuck fetch can no longer poison the future); (3)
  identity-guarded cleanup — a late-settling orphan can't clobber its
  replacement. Applied to trains AND aircraft.
- RATCHET: 7 tests incl. the exact outage shape (stuck promise →
  deadline rejection), orphan-vs-replacement identity, unhandled-
  rejection absorption, and wiring pins (both routes use the guards;
  gzip header present). Deploy-verify: after merge, /api/data/trains
  must respond <15s with either data or stale/503 — never hang.
- Related audit findings (SEPARATE PRs queued): trains/aircraft
  archives are request-driven (gaps never refill — needs eager tick);
  trains lacks a health-aware registry status override.

## 2026-07-05 — [REPAIR] Temp value-labels ate the wind arrows (v1.0.91) [T-CLIENT]

- Production report: with Temperature value-labels ON, wind arrows
  disappeared leaving orphaned kt text. ROOT CAUSE: both sample the
  same server-side grid points; wx-temp-labels sits higher in the
  style, so it wins MapLibre's symbol placement pass, and the arrows
  layer was only half-shielded (icon-allow-overlap true, but its kt
  text still collided and the icon/text pair got split by the
  collision pass — one half surviving without the other).
- FIX (by construction, not tuning): (1) the arrow+kt pair is now ONE
  unit fully outside the collision pass in BOTH directions —
  allow-overlap + ignore-placement on icon AND text, text-optional
  removed — it can never be hidden, never hide others, never be
  separated; density stays bounded by the sampled grid, so opting out
  of declutter is safe. (2) At shared grid points the two label sets
  dodge by OFFSET: temp label anchored bottom at [0,-1.2] (above the
  point), arrow at the point, kt at [0,1.3] (below).
- RATCHET (repair rule 3): the fields-on battery never exercised this
  state — value-labels defaults OFF, so the bug shipped untested. The
  battery now turns the sub-toggle ON and asserts: temp labels placed,
  arrows still placed, the arrows layer's four collision-exemption
  flags intact, and the temp anchor offset intact. Harness green
  390/768/1440; magnified screenshot review confirms 72°F above /
  arrow / 17 kt below at shared points.
- Harness gotcha recorded: .vt-field-controls is a SIBLING of the
  data-vt-layer row — sub-toggle locators anchor on label text, not
  the row selector.

## 2026-07-05 — [PIPELINE] Stream #3 gate 1: PASSED 10/10 (prod vs fredgraph.csv)

- v1.0.90 deployed; prod /api/data/macro serves 28 public series with
  FRED attribution; restricted series (VIX/BAML/UMCSENT) confirmed
  ABSENT from the live payload.
- Gate 1 spot checks — prod latest value vs the FRED web UI's own
  fredgraph.csv export, exact match required: DGS10 4.48, DGS2 4.17,
  T10Y2Y 0.35, SOFR 3.66, ICSA 215000, UNRATE 4.2, CPIAUCSL 333.979,
  WALCL 6724564, DCOILWTICO 71.87, DTWEXBGS 120.8866 — 10/10 MATCH
  across daily/weekly/monthly cadences. Stream #3 gate 1 PASSED;
  vintage archive recording from this deploy onward.

## 2026-07-05 — [PIPELINE] Stream #3: FRED macro regime feed (v1.0.90) [T-DATACORE]

- Human set FRED_API_KEY in Railway (not in the session env — noted in
  wishlist); built stream #3 same day: server/fredMacro.ts — 31
  regime-relevant series (curve, Fed-produced stress indexes, labor,
  inflation, activity, liquidity, WTI/dollar), key-gated 6h poll,
  /api/data/macro route, envelope manifest.
- POINT-IN-TIME VINTAGE ARCHIVE (the build's real asset): FRED revises
  history silently; every observation is archived with rt = as-seen
  date and a revision appends a NEW row (dedup key is (series, date,
  value), never (series, date)). Recording forward turns the free feed
  into the paid ALFRED-style vintage dataset (BUILD-FIRST #2) — and
  protects future regime backtests from lookahead via revised data
  (Reasoning Standard #7): "known on day X" = filter rt <= X.
- LICENSING SPLIT (checked first): CBOE VIX, ICE BofA HY OAS, UMich
  sentiment are third-party copyrighted → license:"restricted",
  archived for internal regime use only, EXCLUDED from the public
  payload by buildMacroPayload (pinned by test). All other 28 series
  are US-gov/Fed-produced. FRED attribution on the route + manifest.
- Gate 1 (values match FRED web UI on 10 spot checks): runs against
  prod /api/data/macro vs the keyless fredgraph.csv export after this
  deploys (API key lives only on Railway) — result appended below when
  checked. Gate 2 (regime-conditioning improves an existing validated
  signal) not attempted; no validated signal exists yet to condition.
- Tests: 6 new (documented-shape parser incl. string values + "."
  missing, series-table license pins, revision-append vintage
  behavior, mocked end-to-end refresh + restricted-exclusion pin, key
  gating, wiring pins). Offline pytest suite green (120 passed). No
  backtest — data pipeline only.

## 2026-07-05 — [PRODUCT] Hero globe: real registry symbols instead of dots (v1.0.89) [T-CLIENT]

- Directive 2026-07-05 (#data-intel hero only): the globe's colored
  dots became REAL vehicle silhouettes from the SAME shared icon
  registry the /data map uses (lib/mapIcons SDF shapes + classifiers,
  lazy-imported with maplibre) — globe and map cannot diverge, and
  future icon improvements land on the globe automatically.
- Aircraft: classifyAircraft(type, category) → jet/prop/heli/generic
  silhouettes, icon-rotate bound to heading. Vessels:
  classifyVessel(shiptype) → tanker/cargo/boat hulls, rotated to COG.
  Sites: SITE_ICON category markers (anchor/tank-rings/factory),
  upright, amber with SDF halo glow.
- MISSING-CLASSIFICATION DEFAULTS: deterministic hash of the track id
  picks from a believable mix (jets-heavy for aircraft, cargo/tanker
  mix for vessels) — stable across refreshes, and real classification
  always wins when the feed carries it. Honest note: shapes for
  unclassified tracks are DISPLAY defaults, not data claims; headings
  keep the map's ?? 0 convention, never fabricated.
- Perf: symbol layers are the SDF path the /data map profiled (M4,
  fill-rate bound); small fixed icon-size (0.32/0.26 air, 0.30/0.24
  sea desktop/phone) + existing phone caps (500/300) keep the budget —
  cap density, never stutter.
- Harness: new landing-globe battery (scroll #data-intel into view,
  wait for placed features, assert symbol layers + icons varied +
  every icon registered via hasImage + icon-rotate bound to heading;
  landing-globe-{w}.png artifacts). Full run green at 390/768/1440,
  0 hard failures. Self-review: harness fixture (10k aircraft in a US
  box) is too dense to judge legibility, so a production-sparsity
  probe (220 aircraft / 140 vessels global) verified all 10 icon
  shapes render and read at both 1440 and 390 — planes/hulls/anchor/
  tank-rings/factory all legible.
- Legend rule note: the hero is a decorative background, not a map
  surface with a legend — the symbols' legend entries live on /data
  where the same registry shapes are already legend-paired (parity
  battery). No new unpaired symbol class was introduced.

## 2026-07-05 — [PIPELINE] Stream #2: EDGAR 13F-HR institutional-holdings archiver (v1.0.88) [T-DATACORE]

- Built stream #2 of the DATA STREAM EXPANSION build order end-to-end:
  server/edgar13f.ts — getcurrent-feed fetcher (reuses Form 4's Atom
  parser), namespace-tolerant primary-doc + information-table parsers,
  append-only JSONL archive under <archive>/filings13f/ (accession
  dedup, old days gzipped), 15-min eager-boot poll, routes
  /api/data/filings13f + /history, envelope manifest.
- LADDER gate 1 (DATA): PASSED for the parser — fixtures are two real
  live-fetched filings (BURKETT 0001762716-26-000003, ATMOS
  0001905162-26-000005), every asserted field hand-checked against the
  filed XML; live end-to-end pull of 4 real filings verified (periods
  normalized, totals match, archive round-trips). Gate 2 (new-position
  clustering vs forward returns, 45-day lag modeled) NOT attempted —
  RAW as-filed records only, no predictive claim.
- FOCUSED-MANAGER CAP (explicit, never silent): filings with >250
  positions archive summary-only (holdings omitted; info table not
  even fetched). This encodes the hypothesis — capacity-constrained
  managers in small caps (EDGE DOCTRINE #2); mega-manager index
  tables would dominate archive bytes with no cluster signal. Stated
  in the manifest (_cap), the API (focused_cap), and pinned by test.
- Data notes for gate 2: value field is FULL USD (post-2023 rule, not
  thousands — manifest documents it); periodOfReport normalized from
  EDGAR's MM-DD-YYYY; amendments (13F-HR/A) flagged via
  submissionType, never merged into originals.
- Tests: 10 new (parsers on real XML, doc-name picker on both real
  directory shapes, cap behavior incl. no-fetch assertion, archive
  round-trip + dedup, wiring pins). Full offline pytest suite green
  (120 passed); manifest sweep green. No backtest — data pipeline
  only, no trading-logic change.
- Timing: Q2-2026 13F season opened this week (deadline Aug 14) —
  every poll from merge onward lands in the heaviest filing window of
  the quarter; a season's small-manager tail accumulates from day one.

## 2026-07-05 — [RESEARCH] Data-stream expansion: audit + 8-stream build order filed, hypotheses before pulls (docs)

- Stream-expansion directive audited and filed (open_questions DATA
  STREAM EXPANSION): RECORDING NOW = aircraft, vessels, trains, fires,
  Form 4, 8-K language, option chains (Monday), Sentinel-2 readings,
  and COT — stream (2) was merged by a concurrent routine (#191,
  v1.0.86) while this session built the hero; supersession honored, no
  duplicate build.
- Build order 2-8 filed with a PRIOR + LADDER PATH per stream BEFORE
  any first pull (Reasoning Standard #10): 13F clusters (45-day lag
  honestly modeled), FRED (regime input, never traded alone),
  USAspending (award/mcap on small caps), FDA calendars (theta-side
  IV-ramp structure vs our own archived chains), USGS water
  (conditional drought-year signal), GDELT (alert trigger joined to
  own sensors, not a trade), pytrends (gate-1 stability test may kill
  it — a finding, not a failure).
- Credentials: FRED free-key steps filed in wishlist (only key needed
  in the whole batch); everything else keyless; patents stays blocked
  on USPTO ID.me.
- Next session builds #2 (13F clusters) end-to-end per the standard
  doctrine: fetcher/parser/archiver + envelope manifest + registry
  where geographic.

## 2026-07-05 — [PRODUCT] Hero refinements: dominant globe, waitlist right, REAL self-updating stats (v1.0.87)

- Territory: T-CLIENT (+ one datacore server module). Hero-refinements
  directive, scope held to #data-intel.
- REAL STATS root-caused and fixed: prod archiveStats() returns
  files/bytes — the hero was summing a `samples` field that only the
  FIXTURE had, hence the production dash; and it only covers position
  kinds, undercounting streams. New /api/data/platform/stats
  (server/platformStats.ts): layers from the live registry;
  streams_recording = live layers + archive dirs not mapped to a layer
  (DIR_LAYER_MAP for filings→insider, earnings8k→earnings; waitlist/
  apiusage operational dirs and *_tracks excluded); observations =
  REAL line counts across the archive, gz-aware streaming, 10-min TTL
  cache with stale-beats-recount. Nothing hardcoded — every number
  grows as the system grows. 3 new tests incl. the phantom-samples
  regression pin.
- GLOBE prominence: zoom 1.05→1.45 (1.15 phone), brighter land
  (#1b3560/#3a67a6), 20° graticule (mission-control read), points
  2.1px aircraft ≤1200 / 1.9px vessels ≤800 with a GRACEFUL PHONE
  DENSITY CAP (500/300 under 640px — cap density, never stutter),
  glowing 4px site markers, legibility shade lightened (right edge
  0.25→0.04) so the sphere is the centerpiece not a curtain.
- LAYOUT: waitlist moved to a bordered card on the RIGHT of the hero
  row; "Open the live map" button REMOVED per directive (its CSS block
  replaced, not orphaned — dead-code policy). Headline, positioning
  copy, and the imagery strip untouched.
- CONCURRENCY NOTE: routine merged #191 (CFTC COT pipeline, v1.0.86)
  mid-build — stream (2) of the data-expansion directive is theirs;
  read-and-increment took this PR to v1.0.87, no collision.
- Gates: node 130/130; harness 0 hard failures ×3 pages ×3 widths +
  all-off; globe probed under software-GL at 390/1440 (sphere with
  graticule/land/points dominant, text legible); screenshots reviewed.

## 2026-07-04 — [REPAIR] /api/diag shipped: token-gated read-only diagnostics (option d, human-approved) (v1.0.81)

- Territory: T-BOT (server/bot.ts route + pure server/diag.ts). Closes
  the session-self-diagnosis gap: KNOWN BROKEN #3/#4 verification
  (fills firing? feedback accumulating? retrain green?) no longer
  needs the human to paste JSON.
- Shape per the approved option (d): GET /api/diag/:probe, HARD
  WHITELIST {audit tail (time/type/message only), ml status
  (model age + fills/feedback counts), daemon health, positions
  SUMMARY (counts + gross/net exposure — never symbols)}. Closed by
  default: no DIAG_TOKEN or <24 chars ⇒ 404; timing-safe token
  compare; every response passes sanitizeDiag (key-like strings, long
  hex/base64, emails → [redacted]) as defense-in-depth over the
  whitelist shaping. auth.ts (frozen) untouched — pinned by test.
- Token generated and handed to the human (Railway + Claude Code
  session env, header x-diag-token). Test file uses a dummy — the real
  value never enters the repo.
- Gates: node 121/121 (4 new: closed-by-default + token check,
  summary-has-no-symbols, sanitizer A/B incl. survivor timestamps,
  wiring pin incl. auth.ts-untouched); build clean; tsc unchanged (61
  pre-existing). Verification plan: once DIAG_TOKEN is set in Railway,
  next session curls /api/diag/ml and logs the first live reading.

## 2026-07-04 — [PRODUCT] Decisions batch executed: monetization 2+3 delivered, options pilot decided, DIAG approved, CDSE/AIS verified (docs)

- Human decision batch (five items) recorded and executed:
  1. Monetization checklist items 2+3 APPROVED as pre-revenue prep —
     DELIVERED: datacore/LICENSING_AUDIT.md (per-source resell-vs-
     display register the API's LICENSE_MARKS derive from; re-verify
     every row at switch) + datacore/API_TERMS_DRAFT.md (customer ToS,
     explicitly DRAFT/not-in-effect). Items 1+4 wait for the charge
     decision.
  2. Options data: FREE Databento pilot chosen (needs the human's
     account + DATABENTO_KEY in session env — steps delivered); free
     Alpaca daily chain archiver queued as its own [PIPELINE] PR —
     starts regardless of the pilot outcome.
  3. DIAG option (d) APPROVED: token generated + handed to the human
     (Railway + session env); /api/diag route ships as its own code PR
     with the sanitizer test.
  4. CDSE: creds set in Railway; NO code read any CDSE var yet —
     canonical names declared CDSE_CLIENT_ID / CDSE_CLIENT_SECRET (S1
     pipeline will read exactly these); told the human sessions also
     need them in the Claude Code env; OAuth endpoint + STAC catalogue
     verified reachable from our egress. Item stays open until an
     authenticated S1 pull succeeds.
  5. AISSTREAM verified LIVE in prod (enabled:true, 1,838 vessels,
     registry "live") — the reported "awaiting key" was a pre-restart
     tab; wishlist entry closed with evidence.
- Bonus verification while probing: the v1.0.80 liveness field is live
  in production health (checks.bot.liveness {dark:false}, bot active).

## 2026-07-05 — [PRODUCT] Hero globe: live rotating 3D Earth behind #data-intel + real facility imagery strip (v1.0.85)

- Territory: T-CLIENT. Hero-globe directive executed within the stated
  scope (ONLY #data-intel; The Bot/pricing/docs untouched — harness
  data+developers shots unchanged).
- GLOBE: MapLibre globe projection (already the /data dependency —
  lazy chunk, no new library; globe.gl/three.js rejected as +600KB;
  Cesium rejected per directive). Land silhouettes from OUR
  self-hosted NE boundaries — ZERO external tiles, so the globe works
  in the harness and never depends on a CDN. REAL data on the sphere:
  live aircraft (blue, ≤400) + vessels (green, ≤300) from world-bbox
  fetches, strategic sites (amber, Cushing/ports/mills) from
  /api/data/sites. Slow auto-rotation (0.02°/frame) that PAUSES
  off-screen, on hidden tabs, and under prefers-reduced-motion;
  aircraft refresh 30s only while visible.
- PERFORMANCE/DEGRADATION: boot is IntersectionObserver-gated
  (rootMargin 400px — zero cost until scrolled near), WebGL-checked,
  interactive:false + pointer-events:none (can never hijack scroll);
  ~700 points vs the proven 10k budget. Any failure (no WebGL, chunk
  load, fetch) leaves the styled dark-space backdrop with the existing
  "Open the live map" CTA — DEVIATION NOTED HONESTLY: the directive's
  named fallback was a "pre-rendered rotating globe"; a CSS-faked
  rotation is itself the jank risk, so the fallback is clean-static +
  live link, per the never-janky principle that motivated the clause.
- IMAGERY STRIP (in-section): three REAL chips pulled by our own
  pipelines this session — Sentinel-1 SAR (Cushing tank farm,
  radar-bright metal), fresh Sentinel-2 true-color of the Cushing
  tank clusters, Sentinel-2 of the Baytown TX refinery complex; all
  honestly captioned + "contains modified Copernicus Sentinel data
  2026" credit; ~290KB total, lazy-loaded.
- Existing headline/copy/stats/CTA/waitlist untouched (text-shade
  gradient added behind them for legibility; heavier on phones —
  text wins over spectacle at 390).
- Gates: harness 0 hard failures ×3 widths ×3 pages + all-off; node
  127/127; globe probed rendering under software-GL at 390+1440
  (sphere, land, points visible; content legible on top); screenshots
  reviewed.

## 2026-07-05 — [PRODUCT] Landing page: DATA INTELLIGENCE section shipped — the oldest unexecuted directive closes (v1.0.84)

- Territory: T-CLIENT. Task #50 (three-part directive PART 3) finally
  executed. STRICTLY ADDITIVE as directed: git numstat on the three
  raw landing files shows insertions only (the two 1-line "deletions"
  are the no-newline-at-EOF artifact — final lines byte-identical).
  New section #data-intel between The Bot and Pricing, using the
  page's existing design system (section.s, prose-grid, accent).
- COPY PROVENANCE (honesty): the full approved copy block did NOT
  survive context compaction — verbatim survivors are the HEADLINE
  ("The physical economy, observed live.") and the atlas-Part-4
  POSITIONING line ("We are not a basemap competitor — ..."), both
  used verbatim. The one supporting paragraph is drafted new and
  FLAGGED in the PR for human review/replacement.
- GLOBE: the landing already ships a D3 canvas globe as its hero — the
  directive's globe requirement is satisfied by the existing one; a
  second globe would be redundant and janky-risk. The new copy points
  at it ("the globe above isn't decoration — it's the product").
- LIVE STATS (live map layers / data streams recording / archived
  observations) from the public /api/data/layers +
  /api/data/archive/stats endpoints; "Open the live map →" CTA to
  /app#/data; EMAIL-ONLY waitlist reusing POST /api/waitlist
  (source "landing"), explicit "no billing" copy — tripwire untouched.
- GRACEFUL DEGRADATION finding: the landing's script (incl. anything
  appended to it) is D3-CDN-gated — landing.tsx returns early on CDN
  failure. The section's wiring therefore lives in landing.tsx itself
  (React side), so stats degrade to em-dash placeholders and the form
  still posts even with the CDN dark. Probed both states headless.
- LANDING NOW UNDER HARNESS: "/" added to visual_check PAGES (it was
  never tested before) — layout/touch/overflow checks ×3 widths ride
  every future client PR. Gates: harness 0 hard failures across
  data+developers+landing+all-off; node 127/127; screenshots reviewed
  (390 + 1440, section + populated stats).

## 2026-07-05 — [RESEARCH] Databento quality VALIDATED (~$0.30) + CDSE Sentinel-1 verified end-to-end (docs)

- DATABENTO (approved validation stage): 9 stratified days across
  2016–2017 incl. the selloff/Brexit/election event days, 10-name mix,
  ~840k closing-window quote rows. Zero crossed quotes anywhere;
  spreads widen on event days exactly as real markets do (median
  2.8–5.6%, event p90 40–47%); put-call parity implies SPY 216.50 on
  election day = the actual close. VERDICT: quality validated; the
  ~$600 full-history go is now purely a budget call, with ONE
  engineering prerequisite filed — durable storage for the ~5GB slice
  (sessions are ephemeral; deliberately sampled instead of burning
  credits into a disposable container). Details in the wishlist entry.
- CDSE (item 3 of the directive): OAuth token issued with the real
  client credentials; catalog search found a fresh S1D GRDH scene over
  Cushing; a REAL 256×256 Sentinel-1 VV chip of the tank farm pulled
  via the Sentinel Hub Process API (radar-bright tanks resolved) —
  the S1 leg of the fused-sensor engine is UNBLOCKED, and Process-API
  chip windowing is the right primitive (61KB vs 1.7GB products).
  Zipper bulk downloads 401 (audience) — recorded with fixes, not
  needed for the chip design. Wishlist item closed with evidence.
- ALSO VERIFIED this session: /api/diag live with DIAG_TOKEN (401
  bare, data with header; ml probe shows model 17.8h old, 500
  feedback records, fills_count 0 — that zero + a TIER3-DIAG
  "warning — 1 issues" line are follow-ups for a diagnostics session).

## 2026-07-05 — [PIPELINE] Options-chain daily archiver LIVE — forward history starts today (v1.0.83)

- Territory: T-DATACORE. Executes the human's "start now — every day
  not archiving is history permanently lost" mandate (options-data
  decision 2026-07-04). server/optionsChainArchive.ts, FIRMS-poller
  pattern: once per trading day after 16:15 ET (ET-aware, once-per-day
  claim persisted so restarts can't double-fire), ≤120 underlyings
  from the CSP-universe cache (spot prices ride along in the cache
  tuples — zero extra API calls for the strike band) + open-position
  symbols, per underlying the Alpaca v1beta1 snapshots endpoint
  (paginated ≤5 pages, 350ms politeness spacing), filtered to exp ≤60
  days and strikes ±20% of spot.
- FEED HONESTY: paper accounts serve feed=indicative — NOT NBBO. The
  label travels on EVERY record ("feed":"indicative"), in the manifest
  license line, and the URL pins feed=indicative. Databento cbbo-1m is
  the ground-truth complement (pilot verdict GO, 2026-07-04).
- Volume budget stated up front: ~3–5MB/day raw JSONL, gzipped by the
  existing archive compressor; envelope manifest
  datacore/manifests/optionchains.json (enforcement test green).
- Gates: node 127/127 (6 new tests: universe cap/dedup/spot-ridealong,
  OCC parse + DTE/band filters, indicative-label-never-dropped,
  day-file + once-per-ET-day scheduling incl. weekend/pre-close cases,
  pagination + HTTP-error surfacing, manifest + routes wiring pins);
  build clean; server-only.
- First real snapshot: next trading day's close (2026-07-06 Mon) after
  this deploys. Verify via /api/data/archive/stats gaining an
  optionchains kind, or /api/diag audit tail.

## 2026-07-04 — [RESEARCH] Databento pilot EXECUTED: options history pull priced at ~$740, verdict GO (docs; $0 spent)

- Human provided the API key; pilot ran same message via free
  metadata.get_cost calls. PRIOR (stated in the wishlist entry before
  pricing): viable if the full pull quotes under ~$1,500. RESULT:
  cbbo-1m confirmed to 2013-04-01 from the API; measured closing-1-min
  per-day costs SPY $0.0129 / AAPL $0.0035 / F $0.0019; batching
  cost-neutral. Universe estimate ~$0.28/day → 2016→present ≈ $740,
  2013→present ≈ $930. Under budget even at 2x error → GO.
- Staged execution filed: $125 free credits cover a 2016–2017
  validation pull ($0 out of pocket); spend beyond credits waits for
  the human's go after quality validation. Env dependency: session
  environment still lacks DATABENTO_KEY (presence-checked this
  session: DATABENTO_KEY, CDSE_CLIENT_ID, CDSE_CLIENT_SECRET,
  DIAG_TOKEN all missing) — human walked through the setting.
- Two-sided honesty note: statistics schema (OI/settlement) priced
  10-100x the quote slices — sampled, never bulk-pulled.

## 2026-07-04 — [RULE-REVIEW] WORKSTREAM PARTITION amendment SHIPPED (docs)

- Human approved ("WORKSTREAM PARTITION amendment — ship it"). Applied
  to CLAUDE.md as a new section after SESSION BUDGET exactly as
  proposed, dated 2026-07-04: T-DATACORE / T-CLIENT / T-BOT
  territories, SHARED serialize-and-minimize list, and the 6-point
  merge-order protocol (shared edits last, read-and-increment
  versioning, keep-both research merges, identity-gated monitors,
  primary-territory ownership, supersession). Sessions declare their
  territory in their first experiments entry from now on.

## 2026-07-04 — [RULE-REVIEW] ACTIVE ANGLE-HUNTING amendment SHIPPED (docs)

- Human approved ("ANGLE-HUNTING amendment — ship it"). Applied to
  STANDING BEHAVIORS exactly as proposed, dated 2026-07-04: every EDGE
  session not consumed by repair or a higher-priority queued item
  hunts novel angles (cross-connections, anomaly mining, foreign-field
  imports, second-order), with the freedom-plus-rigor discipline —
  every angle logged with its testable form and ladder path, priors
  first, multiplicity discounts, out-of-sample required.

## 2026-07-04 — [REPAIR] Liveness alarm RUNTIME half live: dark loop now degrades /api/health (v1.0.80)

- Completes approved Amendment 2: server/liveness.ts (pure,
  node:test-safe) computes NYSE-session overlap (weekdays 9:30–16:00
  ET, DST-correct via Intl; HOLIDAYS deliberately not excluded — the
  alarm errs loud and market_calendar.py stays the single holiday
  source of truth) and the dark verdict (>2 market hours OR >24h
  wall-clock since last seen active).
- Heartbeat persisted like equityPeak (volume + /tmp fallback) so
  DEPLOYS NEVER RESET THE DARKNESS CLOCK — the equityPeak lesson
  applied preemptively. Fresh installs seed now (no instant false
  alarm). Railway's healthcheck polling drives the assessment; a 60s
  interval touch keeps the stamp fresh regardless; zero disk churn
  while dark (same-object return skips writes).
- /api/health Check 5 now carries checks.bot.liveness {dark,
  marketHours, wallHours, detail} and sets overall status=degraded
  when dark → HTTP 503 → every DAILY routine's health read surfaces
  it top-of-report. This closes the exact gap of the incident: bot
  paused, health "ok", human found it on a dashboard.
- Regression tests (5): heartbeat transitions incl. restart-keeps-
  clock, intraday market-hours math, weekend-spans-no-session, the
  2h/24h thresholds A/B (3h dark, 1.5h not; weekend halt trips 24h
  before Monday, Saturday-night check stays quiet), and a bot.ts
  wiring pin (nextLiveness + loopDark + degraded consumption +
  approved constants 2/24).
- Gates: node 117/117; tsc error count identical to main (61 — all
  pre-existing); build clean; server-only (no harness required).

## 2026-07-04 — [RESEARCH] Satellite multi-sensor roadmap filed + angle-hunting amendment proposed (docs)

- Satellite directive filed as the major roadmap in open_questions.md:
  resolution reality stated first (10m = facility-scale change, never
  object counting; the wall is attacked with MORE SENSORS, not more
  optical); S1 SAR + S2 optical + Landsat thermal fusion with
  per-sensor licenses and revisit cadences; LIDAR = one-time
  calibration only; Phase 1 free fused change detection (Cushing tank
  farms vs EIA — extends the existing gate-1 pipeline; steel yards vs
  AISI; construction vs permits/news); Phase 2 paid counting
  wishlist-gated on Phase-1 validation + revenue; imagery-age
  indicator alongside; validation mandatory with priors, multiplicity
  discounts, out-of-sample.
- Wishlist: CDSE signup steps filed (one credential → S1+S2; S2
  already zero-credential; ASF/Earthdata as the S1 fallback to
  verify); Phase-2 paid imagery entries with build-first analysis and
  honest price notes.
- ACTIVE ANGLE-HUNTING amendment PROPOSED in wishlist (exact text,
  NOT self-applied): recurring EDGE-session mandate to generate novel
  hypotheses (cross-connections, anomaly mining, foreign-field
  imports, second-order) with freedom-plus-rigor discipline — every
  angle logged with its ladder path.

## 2026-07-04 — [REPAIR] Toggle state-desync root-caused two ways: remount delta bug (proven+fixed) + open-tab version skew (guarded) (v1.0.79)

- Directive: prod atlas toggles flip the pill but the label stays "off"
  and nothing renders. State model: pill = enabled[id] (local), label =
  runtime[id].status (set ONLY by each layer's effect), map = the
  effect's add/remove — three views that desync exactly when the
  EFFECT IS MISSING for a rendered row.
- NEW RATCHET first (repro before patch): harness TOGGLE-CONSISTENCY
  battery flips EVERY live registry layer (13) and asserts
  pill+label+map move together, both directions. Result: 13/13 CLEAN
  locally — the atlas desync does NOT reproduce in the current code.
  But the sweep EXPOSED a different real bug:
- BUG A (proven + fixed): live-points layers (aircraft/vessels) toggled
  off→on never re-render — teardown kept sinceRef's delta cursor, the
  refetch sent a stale ?since=, the server answered {unchanged:true},
  and the early return skipped addSource/addLayer entirely. In prod
  this self-heals on the next data tick (~10s aircraft) but is
  indefinite for static feeds and deterministic in the harness (the
  fields-on battery failed with "no aircraft rendered" + a -1 indexOf
  masquerading as an ordering violation). Fix: teardown deletes the
  cursor; the unchanged path also drops it when the source isn't
  mounted. A/B: battery red before, green after.
- BUG B (prod vector, guarded): PROD EVIDENCE — deployed bundle hash
  equals this session's local v1.0.78 build (index--vjwcN8a.js,
  Last-Modified 21:55:30Z), registry + /api/data/boundaries current,
  atlas effects present in every bundle ≥v1.0.74. So no single fresh
  deploy shows the symptom. The coherent vector: an OPEN TAB running an
  old bundle remounts the /data page (hash navigation) → registry
  re-fetches (fresh rows incl. atlas ids) → the old bundle has NO
  effects for those ids → pill flips, label stuck "off", no render.
  Structural guard shipped: /api/data/layers now carries
  server_version; the client compares its baked-in build version and
  (a) shows a "reload to enable the newest layers" notice on mismatch,
  (b) renders rows whose id the bundle has no wiring for
  (id ∉ LAYER_GROUP) as DISABLED with "reload to enable" — a
  functional-looking toggle for an unwired layer is now impossible.
- FALSIFIABILITY: if the desync recurs on a FRESH reload of v1.0.79+,
  the skew hypothesis is wrong — per loop-health rule 4 that recurrence
  becomes a root-cause-only session, no re-patching.
- LEGEND half of the directive: SUPERSEDED before it arrived — v1.0.78
  (#173) replaced the dot legend with registry-drawn symbols + the
  both-directions parity battery; prod had it deployed at 21:55:30Z.
  The "generic dots" observation predates that deploy/reload.
- Gates: node 112/112 (new pins: server_version, skew banner, unwired
  guard, cursor clear); harness 0 hard failures ×3 + developers +
  all-off; toggle battery 13/13; fields diffs 46–65.

## 2026-07-04 — [RULE-REVIEW] Amendment 4 SHIPPED: bloat consolidation (docs) — honest shortfall vs estimate

- Fix 4 applied: DEAD CODE POLICY + CONSTITUTIONAL HYGIENE + the AUDIT
  CYCLE paragraph merged into ONE "AUDITS & DEBT" section (2,171 →
  1,662); SESSION BUDGET 2,025 → 1,447; STANDING BEHAVIORS 4,169 →
  2,952 (history moved to experiments.md, every clause kept); EDGE
  DOCTRINE 4,369 → 3,867 (framing cut; every example name, precedent,
  and the full build-first ladder kept). NO rule lost force — each
  compressed section preserves every normative clause, approval date,
  and exception.
- HONEST SHORTFALL: the filed estimate promised ~27.3K; the file is
  31,062 bytes. Two reasons: (a) Amendments 1–3 ADDED ~1,700 chars of
  approved rule text (platform mission, liveness alarm, sovereignty);
  (b) the filed per-section targets (e.g. EDGE DOCTRINE −1,250)
  assumed cutting example detail that turned out to BE the rule's
  force (standing data-source names, precedents) — cutting further
  trades force for length, which the approval explicitly forbade. Net
  prose cut ~4.0K vs the filed ~4.3K; net file −630 from pre-repair
  because the new rules are net-additive. If a harder target matters
  more than the preserved wording, that is a human call — the next
  30d constitutional audit can propose deeper cuts item-by-item.

## 2026-07-04 — [RULE-REVIEW] Amendment 3 SHIPPED: human-sovereignty clause (docs)

- Fix 3 applied exactly as filed: the verbatim HUMAN SOVEREIGNTY
  paragraph placed FIRST inside AUTONOMY AUTHORIZATION, so the
  delegation is read as subordinate to the human before the delegation
  itself is read. No other rule touched.

## 2026-07-04 — [RULE-REVIEW] Amendment 2 SHIPPED: liveness alarm in Priority 1 (docs; runtime half queued as [REPAIR])

- Fix 2 constitution half applied exactly as filed: LIVENESS ALARM
  appended to GOAL Priority 1 — loop paused/halted/broker-unreadable
  for >2 market hours (or 24h wall-clock) = top-of-report alarm in
  every DAILY session + degraded /api/health.
- Runtime half ships next as its own [REPAIR] PR with a regression
  test: persist state.inactiveSince; /api/health Check 5
  (bot.ts:1049) degrades overall status past the thresholds — the
  hook-confirmed gap that let the loop sit paused unflagged.

## 2026-07-04 — [RULE-REVIEW] Amendment 1 SHIPPED: mission reconciled with the charters (docs)

- Human approved all four constitutional-repair amendments ("ship all
  four in order 1→2→3→4"). This PR applies Fix 1 exactly as filed:
  GOAL section replaced — intelligence-platform mission with the bot
  and API customers as the two first-class consumers; priority ORDER
  preserved; P1 gains "archives recording"; P2 gains ladder-validation
  before trust/trade/sale; P3 = GROW BOTH COMPOUNDING LINES with the
  tend-the-bot-vs-advance-the-platform weighing rule; honesty metric
  two-sided; anti-goals extended with never-sell-unvalidated.
- Wishlist entry annotated APPROVED. Fixes 2 (constitution sentence +
  separate runtime [REPAIR] PR), 3 (sovereignty), 4 (bloat) follow in
  order, each its own PR.

## 2026-07-04 — [PRODUCT] Legend v3: real registry symbols, grouped, collapsible, parity-enforced (v1.0.78)

- Legend directive executed. The old legend hand-duplicated three site
  icons as inline SVGs and showed color dots for everything else —
  exactly the divergence-by-construction the directive kills. New:
  mapIcons.ts exports iconDataURL(name,color) which rasterizes THE SAME
  ImageData registerIcons feeds maplibre (SDF tint emulated with
  source-in compositing, cached per name+color) — legend and map share
  one source of truth and cannot diverge.
- Structure: sections mirror the panel groups (Live Tracking /
  Facilities / Environmental / Fields), Title Case labels, entries
  render ONLY while their layer is on, whole block collapses as one
  unit (open desktop / collapsed phone by default; 44px toggle).
  Color-only chips (altitude tints, raster ramps) stay chips — they
  are color MEANINGS, not symbols.
- DESIGN.md rule added VERBATIM as approved: "Every map symbol ships
  with its legend entry in the same PR, drawn from the shared icon
  registry — a symbol on the map without a matching legend entry (or
  vice versa) is a failed build."
- HARNESS parity assertion, both directions, computed from the LIVE
  style (literal icon-image values + ["get",prop] resolved via
  querySourceFeatures) vs legend [data-vt-icon] DOM: (a) every drawn
  icon has an entry, (b) every entry names a registered icon, (c) no
  empty icon renders. Measured 7–8 icons in use / 16 entries per
  width. A/B-PROVEN: a planted bogus entry failed all three ways
  ("map draws 'vt-train' with NO legend entry", "legend claims
  'vt-bogus' but no such icon is registered", "empty icon render").
  Node pin: DESIGN.md rule text verbatim + iconDataURL usage + no
  hand-drawn SVG duplicates inside the legend.
- Harness note: the taller open legend pushed field-layer rows to the
  scroll edge at 1440 in the fields-on battery (Playwright
  actionability timeout) — battery now collapses the legend and
  center-scrolls rows before clicking. New standing artifact:
  data-legend-{w}.png screenshots (legend beside the live map).
- Gates: node 111/111; harness 0 hard failures ×3 + developers +
  all-off; screenshots reviewed (390px legend fully legible).

## 2026-07-04 — [PRODUCT] Positioning copy on /developers (atlas-parity Part 4) (v1.0.77)

- The directive's honest not-a-basemap framing added to the /developers
  hero: same open geospatial foundation as any Earth viewer, every
  source named, differentiation = live movement + entity fusion +
  market-validated signals + API access; explicit "no claim is made to
  proprietary imagery". Live-vs-coming honesty already per-endpoint on
  the page (meta.coming_gated).
- Pinned by test (waitlist.test.ts): "not a basemap competitor" +
  the no-proprietary-imagery disclaimer must stay on the page; the
  monetization-tripwire string pins are untouched.
- The landing-page additive section (task queue: approved copy from
  the three-part directive) will carry the same positioning when
  built — this PR covers the developer-facing half only.
- Gates: node 110/110; harness developers ×3 clean, 390px screenshot
  reviewed; python untouched.

## 2026-07-04 — [PRODUCT] Atlas parity layer 3: country borders (Natural Earth, self-hosted) live (v1.0.76)

- Third build from the ATLAS PARITY filing: Natural Earth 1:110m
  admin-0 compiled into datacore (254KB slim — properties stripped to
  name+iso3, 177 countries) and served by OUR OWN /api/data/boundaries
  route, day-cached. PUBLIC DOMAIN: zero external dependency, zero
  license constraint on resale (GADM was rejected in the filing for
  its non-commercial clause).
- Base panel group, off by default, fetched ONLY on enable (zero-cost-
  when-off); line layer above rasters / below all data symbols; count
  shown as 177 features; HONESTY note: "1:110m generalized — reference,
  not survey-grade" (+ de-facto boundary policy stated in the registry).
- Gates: node 109/109 (new pin: NE attribution + public-domain wording
  + generalized-resolution honesty); harness 0 hard failures ×3 widths
  + developers + all-off (fixture route added); python untouched.
- ATLAS PARITY buildable trio now COMPLETE (water v1.0.74, forest
  v1.0.75, borders v1.0.76). Remaining are the blocked pair (WorldCover
  WMTS prod-verify; GHSL/WorldPop endpoint research) + positioning
  copy + USDA CDL from the Tier-1 register.

## 2026-07-04 — [PRODUCT] Atlas parity layer 2: forest cover 2020 (JRC GFC2020 via GFW) live (v1.0.75)

- Second build from the ATLAS PARITY filing, same shape as layer 1:
  RAW, Environmental group, off by default, field:true opacity slider,
  legend entry gated on enable, STATIC 2020 vintage stated in registry
  + status note (imagery-date rule). Tiles direct from the GFW public
  tile API (jrc_global_forest_cover/latest/dynamic) — zero server
  cost, zero key; CC BY 4.0 with EC JRC attribution, GFW named as the
  tile service. Pixels verified pre-build in the #167 filing (24,850
  non-transparent px on the z4 probe).
- Gates: node 108/108 (new registry pin: JRC attribution + 2020
  vintage + field flag); harness 0 hard failures ×3 widths +
  developers + all-off; python untouched.
- Remaining from the filed order: NE admin boundaries
  (datacore-compiled vector), then the blocked pair (WorldCover WMTS
  prod-verify; GHSL/WorldPop endpoint research).

## 2026-07-04 — [RULE-REVIEW] Constitutional repair: 4 amendment proposals filed (human-directed; NOTHING self-applied)

- Human directive ran the CONSTITUTIONAL HYGIENE process out of cycle
  (register updated). Four proposals filed in wishlist.md with exact
  text / placement / counts, awaiting item-by-item approval; ship
  order after approval: 1 → 2 → 3 → 4, each its own docs PR.
  1. MISSION RECONCILIATION — GOAL still names the paper account as
     the whole mission while VISION.md/GIP.md define the intelligence
     platform with the bot as one consumer (a live contradiction at
     the top of the constitution). Full replacement GOAL text drafted:
     platform mission, both compounding lines first-class in P3,
     two-sided honesty metric, anti-goals extended with "never sell or
     surface an unvalidated signal". Priority ORDER preserved.
  2. LIVENESS ALARM — proposed N = 2 market hours (+24h wall-clock
     ceiling); HOOK CONFIRMED: /api/health Check 5 (bot.ts:1049)
     already reads killed/active/stopped but never degrades overall
     status — the exact gap that let the loop sit paused unflagged;
     Check 6 (licensing) is the degrade precedent to mirror. Runtime
     half specced as its own [REPAIR] PR with regression test.
  3. SOVEREIGNTY CLAUSE — verbatim human text; placement: first
     paragraph inside AUTONOMY AUTHORIZATION.
  4. BLOAT — measured by section (31,694 bytes total): STANDING
     BEHAVIORS 4,169 (−1,750 of history/narrative), EDGE DOCTRINE
     4,369 (−1,250 of restated precedent), three audit rules in three
     places 2,171 (merge to one AUDITS & DEBT section, −770), SESSION
     BUDGET 2,025 (−520). Net target ~27.3K including Fixes 1–3
     additions; NO rule loses force — only words.
- This session may not self-apply any of it (amendments); the audit's
  only self-applied artifact is the register timestamp + this entry.

## 2026-07-04 — [PRODUCT] Atlas parity layer 1: surface water (JRC GSW v2021) live (v1.0.74)

- First build from the ATLAS PARITY filing: JRC Global Surface Water
  occurrence tiles as a RAW layer — Environmental group, off by
  default, field:true (inherits the v1.0.72 opacity slider at 60%),
  legend ramp (rare→seasonal→permanent) gated on the layer being on.
- Zero server cost: tiles direct from the JRC public bucket (like Esri
  imagery) — no proxy, no key, no Railway budget. Zero-cost-when-off
  by the same lazy effect pattern as terrain (source+layer added on
  enable, removed on disable); all-off harness run stays green.
- HONESTY: status note and registry description state the STATIC
  1984–2021 vintage (imagery-date rule) — this shows where water HAS
  occurred, not live conditions. Attribution EC JRC/Google on-map.
- Tile pixels verified server-side pre-build (open_questions ATLAS
  PARITY: 41,840 non-transparent px on the z4 occurrence probe);
  harness can't render external tiles (aborted for determinism) so the
  in-map screenshot proof is panel/registry/self-see — same evidence
  class as terrain/imagery shipped with.
- Gates: node 107/107 (new registry pin: JRC attribution + vintage
  wording + field flag); harness 0 hard failures ×3 widths + all-off;
  python untouched (green per v1.0.73 repair).
- Next in the filed order: forest cover 2020 (JRC GFC2020 via GFW,
  verified), then NE admin boundaries (datacore-compiled).

## 2026-07-04 — [RESEARCH] Atlas parity filed: free-layer endpoints verified, licenses checked, build order set (docs)

- Geospatial-parity directive Part 1/2 filing (open_questions.md ATLAS
  PARITY section). Endpoint verification done SERVER-SIDE with pixel
  decodes (DESIGN.md tile rule — never HTTP 200s): JRC Global Surface
  Water tiles VERIFIED (65k non-transparent px on the z4 probe), JRC
  Global Forest Cover 2020 10m via the GFW public tile API VERIFIED
  (24.8k px) — the directive's exact dataset; Natural Earth admin-0
  GeoJSON VERIFIED (public domain; GADM REJECTED — non-commercial
  license conflicts with the monetization path). WorldCover: COGs free
  on S3 (verified anonymous listing) but the Terrascope WMTS resets
  from this sandbox — prod-side verify or COG-pyramid fallback filed.
  GHSL/WorldPop: WMS endpoints moved/404 — endpoint research filed,
  licensing already clear (both CC BY 4.0). Elevation: already live
  (Mapterhorn = GLO-30). Dead ends recorded (Hansen GCS tile paths,
  umd_tree_cover_density_2020 "no latest version") so no session
  re-walks them.
- Part 2 BLOCKED-BY-ACCESS boundary filed: Google's Street-View-derived
  professional layers have no free lawful raw material — not a build
  target; free building footprints (Microsoft ODbL / Google Open
  Buildings CC BY) remain the Tier-2 path.
- Parts 3/4: differentiation pointers mapped to existing queue items
  (timeline slider, Everything Graph card, ladder-gated signals,
  /api/v1); positioning copy queued as its own small [T-CLIENT] PR.
- Build order chosen: surface water → forest 2020 → NE boundaries,
  one layer per PR (X7 precedent), all registry-native with field:true
  opacity inheritance from v1.0.72.

## 2026-07-04 — [REPAIR] Local pytest gate repaired: collection breakers + stale pins (KNOWN BROKEN #6 RESOLVED, v1.0.73)

- The constitutional gate (`python3 -m pytest -q`, promotion rule 1) has
  been UNRUNNABLE since the repo import — every session either ran the
  CI 4-file whitelist or scoped around it (KNOWN BROKEN #6 filed it,
  hypothesizing network/keys dependence). Root-caused today; the
  hypothesis was WRONG on both counts:
  1. COLLECTION BREAKERS: two root-level standalone SCRIPTS wear test_
     prefixes. test_auto_discovery.py executes its full discovery
     protocol at import and sys.exit()s → pytest INTERNALERROR kills
     collection for the entire repo. test_full_system.py defines a
     module-level `def test(phase, name, fn)` helper that pytest
     collects and fails ("fixture 'phase' not found") — and its import
     alone costs 62s. Fix: conftest.py collect_ignore for both, with
     the policy documented; both remain runnable directly as scripts.
     No assertion was removed — neither file could execute under pytest
     at all.
  2. STALE PINS (7 failures, none a live bug, none network-dependent):
     (a) test_fixes_pr8 TestTrackFillValidation ×3 — tearDown os.rmdir
     failed because track_fill's atomic write leaves feedback.json.lock
     (fcntl thread-safety, in the code since import); tearDown now
     rmtrees. (b) TestOptionsSlotseparation ×3 — pinned tunable VALUES
     (MAX_POSITIONS==5, MAX_OPTIONS_POSITIONS==3) that dated code
     comments legitimately moved to 8/8 (SIZING-FIX 2026-04-22,
     ALPHA-TUNE 2026-04-21); re-anchored to the MECHANISM (separate
     caps exist structurally; full stock book consumes zero options
     slots) with arithmetic against the live constants — pinning
     tunables in tests contradicts RULE REVIEW's tuning authority.
     (c) TestFix8 ×1 — string pin "max_loss=contract.get" went stale
     when the flow moved through shared_max_loss; the mechanism is
     INTACT and improved (single AND multi-leg paths register the same
     max_loss); re-pinned BOTH hops (contract→shared, shared→register),
     stricter than before.
- RATCHET (loop-health rule 3): test_collection_health.py collects the
  whole repo in a subprocess and demands a clean exit — A/B-proven: with
  conftest.py removed it FAILS carrying the original SystemExit
  diagnostics; with it, green. Any future collection breaker fails the
  gate the day it lands.
- Gate after repair: 311 passed, 1 skipped, ~8s (was: INTERNALERROR; or
  with the breaker excluded, 7 failed + 1 error in 74s — 62s of that was
  test_full_system's import). Count reconciled exactly: −1 full_system
  error entry, +4 test_voltrade_daemon (#164, merged after baseline),
  +1 ratchet.
- CORRECTION to today's v1.0.72 entry (learning-integrity): it
  attributed the gate breakage to "routine commit 2479df0 added
  test_auto_discovery.py" — WRONG. 2479df0 is the repo's INITIAL IMPORT
  commit (74k-line squash, authored 2026-04-23); the breakage is
  pre-existing and was already filed as KNOWN BROKEN #6. No routine
  broke the gate today. The v1.0.72 PR body carries the same error;
  corrected here, append-only.

## 2026-07-04 — [PRODUCT] Weather layer upgrade: opacity sliders, wind arrows, temp labels + scale (v1.0.72)

- Directive: make the now-rendering temp/wind fields usable intelligence
  tools — per-layer opacity control (default ~60% so the base map stays
  visible), aviation-style wind vectors rendered HONESTLY at the data's
  real density, temperature value labels (°F/°C) + color-scale legend.
  All RAW display enhancements — no interpretation, no gating needed.
- HONEST SOURCING (the load-bearing finding): OWM's free tile API is a
  raster COLOR FIELD with no vector data in it. Direction/speed/temp
  numbers exist only in the free current-weather POINT API. So vectors
  and labels come from a sampled point grid: ≤40 points per snapped
  viewport bucket (server/weatherGrid.ts), 10-min shared cache, 45/min
  upstream guard under the 60/min free budget. The UI states the real
  spacing ("one observation per ~N km") and never renders arrows denser
  than the samples — no faked barb density. Barbs proper were rejected:
  at 40 points/viewport the pennant/half-tick grammar would imply
  station-level precision we don't have; arrow + kt text is the honest
  form. Static grid that refetches on pan (debounced 600ms), never an
  animation — phone budget over spectacle.
- Registry-native: layers.json field:true flags opt layers into the
  opacity slider (weather radar included); default 60%, sessionStorage-
  persisted, live setPaintProperty updates. Wind arrows an SDF icon
  (mapIcons registry) with OWM's FROM-direction converted (+180°) to
  pointing direction; temp labels precomputed per °F/°C unit; temp
  color ramp added to the legend labeled "approx — amplified for dark
  basemap" (the proxy amplification from v1.0.69 shifts hues).
- snapBbox bug caught by its own test: quantum derived from each
  viewport's raw span gave nearby viewports different buckets, defeating
  the shared cache. Fix: power-of-two quantum ladder + outward
  (floor/ceil) edge snapping so the bucket always covers the viewport.
- HARNESS fields-on battery (new, all 3 widths): toggles temp+wind as a
  user would, then asserts pixel-level rendering (canvas off/on mean
  diff ≥3; measured 46.8/53.6/56.3), aircraft still rendered with
  fields on, rasters BELOW symbols, 60% default applied, arrows placed
  from the sampled grid, and the v2.4 occlusion hit-test re-run with
  fields on. A/B-PROVEN against a real defect it caught during this
  session: enabling weather grew the attribution to 2 lines at 390px
  and covered the zoom-out button ("OCCLUDED by
  maplibregl-ctrl-attrib-inner"); fixed with an attribution max-width
  cap so it wraps inside the right column. Deterministic wx tile
  fixture stands in for the proxy's amplified OUTPUT (the amplification
  itself stays unit-tested against the real captured prod tile).
- Gates: node 106/106; harness green ×3 + developers + all-off
  (fields diffs above); python vs PRISTINE origin/main baseline —
  identical 7 failed + 1 error on main worktree and on my tree, i.e.
  ZERO new failures from this change. Main's local pytest gate is
  BROKEN independently of U1: routine commit 2479df0 added root-level
  test_auto_discovery.py which pytest collects and which sys.exit()s at
  import (INTERNALERROR, kills the whole run; CI stays green only
  because ci.yml whitelists 4 files), and 7 options/track-fill tests +
  test_full_system.py now fail on pristine main. Filed as the next
  [REPAIR] action — not bundled here per one-change-per-PR.
- Live expectation: default-on look unchanged (fields stay off by
  default); when enabled, base map + live layers remain visible at 60%;
  arrow count per viewport ≤40 with kt labels; °F default matches US
  audience, °C one tap away.

## 2026-07-04 — [REPAIR] Daemon RPC route bug fixed (shadow_stats) + counterfactual-logging dead-config audit finding (v1.0.71)

- Session-start protocol followed in order: CLAUDE.md, experiments.md,
  open_questions.md, wishlist.md all read this session. Loop-health ratio
  over the last 10 entries (API product foundation back through KNOWN
  BROKEN #5): 4 REPAIR / 2 RESEARCH / 2 PRODUCT / 2 PIPELINE — well under
  the 7/10 REPAIR-thrash threshold, no meta-problem to address.
  `/api/health` on prod: all-ok (server/database/alpaca ACTIVE/python/
  licensing all "ok", bot active, equityPeak=108151.39, drawdownPct=0.0%
  — the persisted high-water mark still holding, memory nominal at
  163MB RSS). No live break visible from the public surface; deeper
  audit-log/trade_feedback access remains gated behind requireOwner per
  KNOWN BROKEN #4's unchanged ACCESS LIMITATION, so "fix a bug seen in
  audit logs" (SESSION BUDGET tier 1) was not directly actionable. No
  experiment has matured to a judgeable state this session (Insider
  Form-4, port-dwell, and shadow-fleet gate-2 work are all still
  accumulating history; Sentinel-2 explicitly deferred its next check to
  the June-reversal window, not yet reached).
- Per the KNOWN BROKEN #5 precedent (2026-07-04, same session-budget
  bind: no live audit access, nothing matured to judge), fell through to
  a READ-BEFORE-WRITE static audit as the next best "fix a bug" action —
  this time targeting `shadow_portfolio.py`, since CLAUDE.md's RULE
  REVIEW section names counterfactual logging as the standing evidence
  requirement for every open RULE COST AUDIT question, and no session's
  log had ever mentioned whether that infrastructure exists.
- PRIOR (REASONING STANDARD #10, stated before reading `shadow_portfolio.py`):
  expected counterfactual logging to be wholly unbuilt (open_questions.md's
  RULE COST AUDIT section is headed "after counterfactual logging
  exists," implying it doesn't yet).
- FINDING vs. prior — WRONG, in an interesting way: `shadow_portfolio.py`
  (240+ lines, thorough docstring) already implements almost exactly the
  CLAUDE.md RULE REVIEW spec — `log_candidate()` records
  {ticker, timestamp, score, decision, decision_reason, entry_price,
  regime, 34 ML features} for candidates, and `backfill_outcomes()` is
  wired into `server/bot.ts`'s Tier-1 daily cycle (10pm UTC, confirmed
  live in bot.ts:2717-2733) to fill in forward +5d/+10d/+20d hypothetical
  outcomes via real Alpaca bars, using PATH-DEPENDENT labeling that walks
  the bot's actual take-profit/stop-loss rules rather than close-only
  returns. This has apparently been running daily and accumulating data
  without ever being logged in research/ — a documentation gap, not a
  code gap. HOWEVER: `log_candidate()` is only actually CALLED from one
  place (`bot_engine.py` deep_score(), decision values `taken` /
  `rejected_score`) — the four other decision buckets its own docstring
  names (`rejected_heat`/`rejected_halt`/`rejected_earnings`/
  `rejected_other`) have ZERO call sites anywhere in the repo (grepped).
  So today the shadow archive can only ever answer the MIN_SCORE RULE
  COST AUDIT question, not the spread/correlation/regime/kill-switch
  ones — logged as the natural next PR (open_questions.md KNOWN BROKEN
  #10/RULE COST AUDIT update), not built this session (scope: this
  session's action is the audit + the one confirmed bug, not a new
  wiring project across bot_engine.py's many gate points).
- SECOND FINDING (the confirmed, fixed bug): while checking every
  consumer of `shadow_portfolio.py` per READ BEFORE WRITE, found
  `voltrade_daemon.py`'s RPC route table maps
  `"shadow_stats": ("shadow_portfolio", "get_stats")` — but the real
  function is `get_shadow_stats()`; `get_stats` does not exist. Any RPC
  call to `shadow_stats` would silently return a "Method not found"
  error at runtime. Confirmed via grep that nothing in `server/bot.ts`/
  `server/routes.ts` currently calls this RPC method (latent, not an
  active live break) — `backfill_outcomes` (the piece that actually
  writes data) is unaffected and confirmed working via its own bot.ts
  wiring. This is precisely the "Python signature change with an
  un-updated caller fails silently at runtime, not in CI" class READ
  BEFORE WRITE warns about — except here the caller-side name was wrong
  from the start, not a later rename.
- THIRD FINDING (surfaced investigating why the RULE COST AUDIT
  questions read as unanswerable): `system_config.py`'s `SCORE_BAND_MAX`,
  `MAX_CHANGE_PCT`, `SCORE_BAND_OPTIMAL_LO`, `SCORE_BAND_OPTIMAL_HI` are
  read NOWHERE outside `system_config.py` itself (grepped the entire
  repo) — dead config with comments that claim they gate trades
  ("Skip stocks already up/down 35%+", "Scores above this are often fake
  breakouts") when nothing in `bot_engine.py` enforces either as a hard
  block; `bot_engine.py` only applies a soft score PENALTY for extreme
  `change_pct`, never a skip, and never checks `combined_score` against
  `SCORE_BAND_MAX`/`SCORE_BAND_OPTIMAL_LO/HI` anywhere. Full detail,
  honesty-metric relevance, and the deliberate decision NOT to
  unilaterally wire a hard skip back in (that would be a rule/threshold
  CHANGE requiring RULE REVIEW's evidence-or-ablation gate, which
  neither exists nor can be quickly built — bot_backtest.py/backtest_v2.py
  model ETF rotation, not per-candidate stock selection) are in
  open_questions.md KNOWN BROKEN #10.
- FIX SHIPPED (one logical change): `voltrade_daemon.py`'s `shadow_stats`
  route corrected to `get_shadow_stats`. Regression test added FIRST per
  loop-health rule 3: new `test_voltrade_daemon.py` (no daemon test file
  existed before this PR) walks every route in `RPCDispatcher._routes`
  whose target module exists on disk and asserts the attribute resolves
  to a real callable — confirmed FAILING against the pre-fix code (2/4
  tests failed, pinpointing exactly `shadow_stats` -> `get_stats`), then
  confirmed PASSING (4/4) after the one-line fix. This ratchets against
  the entire class of bug (any future route rename), not just this
  instance — the two genuinely-placeholder routes (`ml_status_impl`/
  `ml_toggle_impl`, which have no corresponding .py file by design and
  fall back to local methods) are explicitly pinned as expected-absent
  so they're never silently miscounted as "checked."
- Verified: full offline CI-gate subset + the new file —
  `python3 -m pytest -q test_risk_controls.py test_audit_critical.py
  test_diagnostic_false_positives.py test_patches_verification.py
  test_voltrade_daemon.py` — 124 passed, 1 skipped (120 pre-existing + 4
  new, identical baseline otherwise; KNOWN BROKEN #6's full-repo
  collection issue is pre-existing and untouched). No `.ts`/`.tsx` files
  touched — Node test suite and the visual harness are out of this PR's
  scope (PROMOTION RULES rule 5, one logical change).
- Downstream chain (REASONING STANDARD #1): fixing the route ->
  `shadow_stats` becomes callable the moment any caller (a future
  dashboard, the still-pending DIAG_TOKEN route, or a CLI probe) wires
  it up -> that caller sees real win-rate-by-decision numbers instead of
  a silent error -> the KNOWN BROKEN #10 dead-config finding gives any
  future session the accurate mental model of which RULE COST AUDIT
  questions are actually answerable today (MIN_SCORE, once ~90 days of
  shadow history accumulate) vs. not (SCORE_BAND_MAX/MAX_CHANGE_PCT,
  which govern nothing yet). Zero live-trading-behavior change from this
  PR — nothing in `bot_engine.py`/`system_config.py`/`strategies/`
  changed, and the daemon route was never called by anything live.
  Version bumped 1.0.70 -> 1.0.71 (read-and-increment) per convention,
  though PROMOTION RULES rule 3's backtest requirement doesn't apply
  (no strategy/parameter change).
- STARVED: no — this session's scope (audit + confirmed-bug fix +
  honest documentation of the dead-config finding) shipped in full.
  High-value work remains queued: KNOWN BROKEN #3/#6, wiring the
  remaining `log_candidate()` decision buckets (spread/correlation/
  regime/kill-switch) so the RULE COST AUDIT's other questions become
  answerable, the SCORE_BAND_MAX/MAX_CHANGE_PCT evidence-or-retire
  decision once shadow history or an ablation harness exists, the
  GEOSPATIAL LICENSING REGISTER items (d)-(g), and the GIP BUILD QUEUE.

## 2026-07-04 — [PRODUCT] /developers page + waitlist + pricing designed-not-enabled (v1.0.71)

- API directive part 2: client/src/pages/developers.tsx renders the
  API reference FROM /api/v1/meta (self-documenting — the page cannot
  drift from the deployed API), a live archive-stats sample fetched by
  the page itself, license marks as they travel with responses, curl
  examples, honest coming_gated list, API pricing tiers marked
  "preview — not for sale yet" (numbers TBA; NO buy buttons, NO
  billing anywhere — tripwire test pins that stripe/checkout/billing
  strings cannot appear on the page), and the waitlist form (email
  only). server/waitlist.ts: validated, deduped (restart-surviving
  seed), day-JSONL PII stream with a manifest that states the
  handling contract (never exposed via API, never in git).
- TRIPWIRE: this session touched pricing DESIGN — the compliance
  re-run was executed and recorded in the MONETIZATION READINESS
  CHECKLIST (wishlist.md) before this shipped; billing remains dark.
- Harness upgraded for the multi-page era: PAGES now carry per-page
  config ({route, map}) — map/perf/self-see batteries run on map
  pages; layout + touch-target checks run everywhere. /developers
  green at 390/768/1440 (screenshots reviewed: theme tokens, live
  sample rendering, 44px targets). 4 new node tests (101/101).

## 2026-07-04 — [PRODUCT] API product foundation — /api/v1 over the archives, key scaffolding, metering, license marks (v1.0.70)

- API directive part 1 built pre-revenue, last mile explicitly gated:
  server/apiProduct.ts (pure module — no express/db imports per the
  runner-hang rule): env-seeded keys ONLY (API_PRODUCT_KEYS; no
  issuance flow by design — issuance binds to billing later),
  per-tier sliding-window rate limits (dev/pro/enterprise), usage
  metering to a NEW manifested stream (<archive>/apiusage/, sha256
  key prefixes — raw keys never logged; forward-enforcement test
  covers the manifest automatically), and LICENSE MARKS on every
  response: aircraft-derived = ODbL share-alike; AIS-derived =
  conditional (aisstream ToS re-read at switch); US-gov streams =
  public domain; OWM excluded from the API entirely (display-only
  product). /api/v1: meta (public docs), tracks/:kind/:id,
  stats/portdwell, stats/shadow, stats/archive — the product IS the
  archive, not a live-proxy duplicate. Gated items (tank-fill, entity
  timelines) appear only under coming_gated — meta honesty pinned by
  test.
- 6 new node tests (97/97): key parsing + closed-by-default, limiter
  window behavior, metering hash discipline, license-mark pins, meta
  honesty, route wiring/guard count. No UI in this PR (/developers is
  the next, T-CLIENT-territory change).

## 2026-07-04 — [RESEARCH] Throughput: workstream-partition amendment proposed + velocity metric + GIP queue territory-tagged (docs)

- Throughput directive executed as filed artifacts: (1) WORKSTREAM
  PARTITION amendment proposal in wishlist.md (exact CLAUDE.md text
  for approval — T-DATACORE / T-CLIENT / T-BOT territories, shared-
  file serialization, 6-point merge-order protocol; rationale: 40 PRs
  merged today across concurrent sessions with 4 recovered collisions
  — territories prevent the class). NOT applied — constitutional
  amendments await approval. (2) VELOCITY metric table in
  usage_log.md (PRs merged/day + queue-depth trend; seeded 32
  yesterday / 40 today). (3) GIP BUILD QUEUE in open_questions.md,
  territory-tagged: aircraft continuity spine [T-DATACORE], UI
  scalability + landing section [T-CLIENT], Sentinel-2 per-tank
  iteration [T-DATACORE], API foundation [SHARED→split]. Parallel-
  subagent use is already standing practice (two research workflows
  this session); codified in the proposal's last clause.
- MONETIZATION READINESS CHECKLIST filed in wishlist.md for approval
  (API-product directive): provider-compliance re-run executed for
  this directive (pricing-design touch trips the tripwire — chain is
  adsb.lol primary/lawful + two non-commercial fallbacks that must
  drop at switch), per-source resell-vs-display licensing audit
  drafted (ODbL share-alike marks on aircraft-derived endpoints; OWM
  excluded from API entirely; aisstream CONDITIONAL pending ToS
  re-read), ToS draft + Stripe wiring plan itemized. Rule restated:
  last mile waits for the human's item-by-item go.

## 2026-07-04 — [PRODUCT] Charters installed: verbatim VISION.md + GIP.md companion (GIP directive Parts 0a + 2)

- VISION.md: the human's verbatim charter text received and installed,
  replacing the labeled reconstruction exactly as its provenance
  banner promised; reconciliation annotations kept; a deltas section
  annotates items the verbatim text adds (tick/futures/crypto data,
  news velocity, web traffic, supply chain, embeddings/RL, five more
  dashboard ideas) with honest statuses.
- GIP.md: full verbatim "Expansion of the Global Intelligence
  Platform" text under its own headings + a session-maintained
  reconciliation annex citing real artifacts
  (SENTINEL2_CHANGE_SPEC.md, EVERYTHING_GRAPH.md, shadowFleet.ts,
  datacoreArchive.ts, the licensing register, the approved inference
  envelope). Honest BLOCKED marks per the directive: object/vehicle
  counting blocked at free 10m (change detection is the lawful free
  version); per-county permit verification is per-target, not global;
  bulk maintenance records and manifest-level cargo data have no free
  lawful source.
- Reading rule extended (human-approved in the directive): PRODUCT and
  EDGE sessions read BOTH charters after CLAUDE.md; usage_log.md
  carries the updated routine-prompt line (HUMAN ACTION to paste).
- Part 0c recorded: options data stays HOLD; the free options-chain
  archiver covers go-forward; revisit only when a gated strategy
  needs deep history.

## 2026-07-04 — [REPAIR] Temp/wind v1.0.69 fix VERIFIED — fields render vividly with real prod tiles at all three widths

- Deploy verified: prod wind tile went 0 → 37,811 strong-alpha pixels
  (the amplification is live). PROOF NOTE (environment honesty): this
  sandbox blocks ALL browser egress (even example.com resets — probed
  with and without the agent proxy), so the "screenshots on
  production" proof ran as a faithful MIRROR: 48 REAL production
  tiles (curl-fetched through the sanctioned proxy, amplification
  confirmed per-tile) rendered by the IDENTICAL client build (same
  merged commit as prod) at 390/768/1440. Result: mean off-vs-on
  pixel diff 156.9 / 151.4 / 157.6 (an invisible layer scores ~0.1;
  floor 3) — screenshots in .visual/prod-weather-{w}-{off,on}.png,
  reviewed: temperature gradients and wind structure clearly legible,
  © OpenWeatherMap attribution rendered, basemap survives through the
  230-alpha cap. Both-layers-stacked is the deliberately-tested worst
  case and is close to saturating — if the human prefers a lighter
  blend, raster-opacity 0.85 → ~0.7 is the single tunable, filed as a
  taste knob, not a defect. scripts/verify_weather_prod.mjs gained
  proxy passthrough for environments where browser egress works.
- Key validity + budget re-checked per the directive: prod status
  endpoint "ok"; the 60-calls/min free budget is shared-cache bounded
  (upstream calls = unique tiles per 10-min TTL; the 48-tile world
  sweep is the practical ceiling per window).

## 2026-07-04 — [PIPELINE] Sentinel-2 tank-fill gate-1 kickoff — PRIOR STATED BEFORE FIRST COMPARISON

- PRIOR (REASONING STANDARD #10 — this entry is written BEFORE running
  the backfill comparison, per the directive's explicit instruction):
  for the v1 facility-scale shadow index (dark-pixel fraction in the
  three verified Cushing tank-farm AOIs, scene-relative threshold,
  tan-zenith normalized) vs EIA weekly Cushing crude stocks over
  ~12-16 backfilled scene-weeks, I expect:
  (a) LEVELS correlation r ≈ +0.2 to +0.5 — weak-to-moderate positive.
      Mechanism: fuller tanks → higher floating roofs → SHALLOWER
      roof-well shadows → per-tank shadow area SHRINKS as stocks rise,
      which argues r NEGATIVE — BUT the v1 index is facility-scale
      dark-fraction, dominated by inter-tank ground shadows cast by
      tank WALLS (constant) and contaminated by ponds/asphalt, so the
      per-tank fill signal is a second-order modulation on a noisy
      base. Sign is genuinely uncertain at facility scale; magnitude
      |r| > 0.5 would surprise me.
  (b) WEEK-OVER-WEEK DELTAS r ≈ 0 (noise-dominated at v1 resolution —
      clouds, sun-angle residuals, tile registration).
  (c) KILL/ITERATE CRITERION: this v1 index is NOT expected to pass
      gate 1. Its job is to prove the pipeline (anonymous scene access
      → windowed reads → archived readings → EIA reconciliation runs
      end-to-end) and establish the noise floor. Gate 1 credit
      requires |r| ≥ 0.5 on levels over ≥12 weeks with a sign
      explainable by mechanism — anything less iterates toward
      per-tank annulus geometry (the spec's real design) rather than
      declaring victory or death on the facility-scale proxy.
- Pipeline facts (probed live before this entry): scene access needs
  ZERO credentials (Element84 earth-search STAC + AWS Open Data
  sentinel-cogs public S3; windowed B04 reads verified over the
  Enbridge AOI); EIA ground truth is keyless (public history XLS,
  current through 2026-06-26: 23.0M → 19.7M bbl June drawdown — a
  real live signal in the comparison window). CDSE credentials are
  NOT required; exact CDSE signup steps filed in wishlist.md as a
  fallback only (per the directive's request).
- RESULT (run AFTER the prior above; scripts/sentinel2_tankfill.py,
  36 readings, 12 scene-weeks matched to EIA weeks, 2026-03-19 →
  2026-06-27, archived in datacore/sentinel2/readings.jsonl):
  Pearson r LEVELS = -0.731; DELTAS = -0.225.
  VS PRIOR: sign matches the physical mechanism I named (fuller tanks
  → higher floating roofs → shallower roof-well shadows → less dark
  area) and the magnitude EXCEEDS my |r|<=0.5 expectation — a genuine
  surprise. Deltas ≈ weak, as predicted.
  HONEST VERDICT — GATE 1 NOT CLAIMED, despite technically meeting
  the |r|>=0.5 + mechanism-sign criterion I wrote above: both series
  are strongly TRENDED over this window (stocks fell near-monotonic
  31.5M→19M bbl; the index rose spring→summer, where imperfect
  sun-angle normalization plus surface seasonality push the same
  direction) — 12 points of trend-vs-trend inflate |r| regardless of
  mechanism; the weak deltas correlation is the tell that detrended
  signal is thin. STANDARD #4 applies: one window, one variant, no
  out-of-sample. WHAT WOULD EARN GATE 1: (a) the June stock REVERSAL
  (18957→19666 kbbl) extending into weeks where the index must turn
  DOWN against the seasonal sun trend — the natural experiment is
  already in motion; (b) per-tank annulus geometry (the spec's real
  design) replacing the facility-scale proxy; (c) >=20 weeks spanning
  at least one full reversal, levels AND deltas both mechanism-signed.
  Weekly readings continue via the archived script — every scene is
  now recorded (collect-everything).

## 2026-07-04 — [REPAIR] Temp/wind recurrence ROOT-CAUSED: OWM 1.0 tiles are intrinsically near-invisible on dark basemaps (v1.0.69)

- RECURRENCE (v2.4 touched this surface once — per loop-health rule 4,
  no re-patch: root-cause analysis). MEASURED on production tiles (six
  real tiles pixel-analyzed): temp_new = uniform 76/255 alpha,
  wind_new = 15-53/255 alpha, ZERO pixels above 120/255 in ANY tile —
  OWM Weather Maps 1.0 palettes are pale low-alpha overlays designed
  for LIGHT basemaps. Attenuation chain: intrinsic alpha (0.3/0.1) ×
  client raster-opacity (0.6) × dark satellite background = 3-18%
  effective visibility. "Not rendering" was rendering — invisibly.
- WHY BOTH PRIOR VERIFICATIONS MISSED IT (the actual generator):
  v1.0.63 verified HTTP 200 + content-type + byte size — never pixels;
  v2.4 fixed STATUS/note display — never pixels. Nothing ever asserted
  the layer's pixel CONTRIBUTION. Ratchet: DESIGN.md gains the
  tile-layer pixel-verification lesson;
  scripts/verify_weather_prod.mjs compiles the check (prod layer-off
  vs layer-on canvas screenshots, mean-pixel-diff floor, all three
  widths).
- FIX AT THE GENERATOR: GL raster paint can only reduce opacity below
  a texture's baked-in alpha — the client cannot fix this. The proxy
  we already own now amplifies each tile once per 10-min TTL (pngjs,
  pure JS): alpha ×3.2 (temp) / ×5.5 (wind, from its measured floor),
  capped at 230 so the basemap survives, +1.6× saturation around luma
  for the pale palette; fully-transparent pixels stay transparent (no
  field invented where none exists); transform fail-open (garbage →
  raw buffer served). Client raster-opacity 0.6 → 0.85 (mild blend
  now, not the visibility mechanism). TEST FIXTURE = a real captured
  production wind tile: must exhibit the defect before amplification
  (zero strong pixels — pinned) and read clearly after; alpha cap
  pinned.
- Verification: post-deploy, scripts/verify_weather_prod.mjs against
  voltradeai.com at 390/768/1440 — results land in this log.

## 2026-07-04 — [RESEARCH] USPTO ODP key path + keyless-bypass verdict (docs)

- Human hit a wall between the submitted ODP form and a key; research
  (primary sources + live HTTP probes) resolved it: the key is gated
  on completing ID.ME FIRST (MyUSPTO → Profile → Verify with ID.me),
  then self-serve at data.uspto.gov/apikey — no approval queue
  documented. Click-by-click filed in wishlist.md.
- Landscape finding worth the entry on its own: the ENTIRE keyless
  USPTO ecosystem died 2025-2026 — bulkdata.uspto.gov retired (host
  dead, probed), Developer Hub decommissioned, PatentsView API offline
  pending ODP relaunch with old keys incompatible, ODP web bulk
  directory account-gated since 2026-06-18, ODP API 401 without a key
  (probed). Only keyless start: Google Patents BigQuery — backfill
  only (repo archived read-only 2026-04-18, freshness unverified).
- Design consequences recorded in NEW DATA ROOTS #4: key-first
  pipeline, single-threaded (ODP burst=1, 429 ⇒ ~7-day lockout risk),
  quotas reset Sun 00:00 UTC, key is a per-person credential tied to
  the human's ID.me.

## 2026-07-04 — [REPAIR] FIRMS activation: env-name mismatch fixed (v1.0.68) + duplicate build superseded

- Human set the FIRMS key in Railway as FIRMS_MAP_KEY and asked "tell
  me if the code expects a different env var name" — it did: the
  routine's merged implementation (v1.0.65, server/nasaFirms.ts) reads
  NASA_FIRMS_MAP_KEY, so the layer would have sat awaiting_key forever
  despite the key existing. Fix: firmsKey() accepts BOTH names (code
  adapts to the action already taken; no Railway rename); the
  awaiting_key reason string names both; regression pinned by a test
  asserting FIRMS_MAP_KEY alone enables the module.
- DOUBLE-BUILD INCIDENT (concurrent-sessions gotcha recurrence,
  CLAIM-before-building rule): this interactive session built a
  parallel FIRMS implementation (own module/route/layer/tests) while
  the routine's version was already merged — discovered at
  cherry-pick time via the wishlist conflict. Per the supersession
  precedent the merged implementation stands; the duplicate was
  abandoned unmerged (PR #155 closed, never double-registered
  anything). Salvaged from the duplicate: the activation fix above
  and the invalid-key probe knowledge (FIRMS returns HTTP 400
  "Invalid MAP_KEY." — designed-status material if their impl ever
  needs it). Root cause of the recurrence: the interactive session
  reacted to the human's key message without re-checking wishlist
  claims first — the routine had marked SCAFFOLDED in the entry the
  session was about to edit. Lesson folded into the gotcha entry:
  CLAIM-check applies to human-triggered work too, not just roadmap
  picks.
- Verification plan: prod /api/data/fires probe post-deploy — expect
  enabled:true with detections (FIRMS keys activate immediately).

## 2026-07-04 — [PIPELINE] SEC 8-K Item 2.02 earnings-language pipeline — gate 1 (DATA) passed (v1.0.67)

- Session start: CLAUDE.md + all of research/ read this session (PRODUCT session mandate). Health check: prod `/api/health` was last verified all-ok earlier the same day (equityPeak persisted, drawdownPct 0.0% — see the entry below); no new critical break surfaced this session and KNOWN BROKEN #3/#4 remain blocked on the DIAG_TOKEN access decision (unchanged, not actionable without human approval) — a PRODUCT session does not preempt the DAILY routines' repair duty per the task mandate, so this did not block proceeding with product work.
- Chose the top-priority queued PRODUCT action per wishlist.md's GEOSPATIAL LICENSING REGISTER "NEXT ACTIONS" note ("...and NEW DATA ROOTS #1 (8-K language pipeline) as the top research build") and open_questions.md's NEW DATA ROOTS build-order rationale: 8-K language ranks #1 because EDGAR history already exists (gate 2 testable immediately, not time-blocked like the jobs/patents roots) with complete small/micro-cap coverage and exact timestamps.
- PRIOR (REASONING STANDARD #10, stated before writing any code): expected the getcurrent Atom feed to require opening every 8-K's own filed document just to learn which Items it covers (mirroring Form 4's feed, which needs a second per-filing fetch for owner data); expected exhibit discovery to need a fixed table-column assumption (Type always in the same position).
- FINDING vs prior: wrong on both counts, in the easier direction. SEC's getcurrent `<summary>` field already lists each filing's Item codes inline ("Item 2.02: Results of Operations and Financial Condition") — no extra fetch needed to filter for Item 2.02 before touching a single filing document. And the index.htm exhibit table's column ORDER varies between filer agents: UniFirst's row has "EX-99" in the Description column; MV Oil Trust's has "EXHIBIT 99.1" in Description with "EX-99.1" in Type — both live-fetched and confirmed 2026-07-04. `pickExhibit99Href` therefore matches on ROW CONTENT (any cell matching `/EX-?\s*99/i`) rather than a fixed column index, so both real formats resolve correctly.
- BUILT: `server/sec8kEarnings.ts` (no API key required, same fair-access terms edgarForm4.ts already relies on). Polls the public "getcurrent" 8-K feed, filters to Item 2.02, resolves each filing's Exhibit 99 press release, converts it to plain text via a dependency-free HTML-to-text pass (full numeric + named entity decode table, verified against real decimal entities `&#8211;`/`&#8220;`/`&#8226;`/`&#64;`/`&#38;`/`&mdash;`/`&rsquo;`/`&ldquo;`/`&rdquo;`). CAUGHT WHILE WRITING THE GATE-1 FIXTURE TEST: numeric `&#160;` decodes to a literal non-breaking-space character, not a regular space — left uncaught, every NBSP in a filed exhibit would have shipped as an invisible non-ASCII character in the archived/served text; fixed by normalizing NBSP to a regular space as part of `htmlToText`, verified by an explicit assertion in the test (`!text.includes(" ")`). Text is truncated at 30,000 chars with an honest `truncated`/`textLength` pair (the fact of truncation is never silently dropped). Filings with no EX-99-class exhibit are skipped, not fabricated with empty text — this is the concrete instance of the HONEST GAP already logged in wishlist.md's NEW DATA ROOTS #1 (Q&A sessions are almost never filed as an exhibit; a filing that announces results only in the 8-K body itself now simply contributes nothing to this feed).
- LADDER — gate 1 (DATA) PASSED: two real, live-fetched filings covering two distinct filer-agent HTML formats (UniFirst Corp Q3 FY2026 results, accession 0001628280-26-046349, Workiva-generated lowercase divs/fonts; MV Oil Trust final-distribution announcement, accession 0001104659-26-080431, classic uppercase P/TABLE/FONT with a data table), every extracted fact hand-verified against the actual rendered exhibit (dollar figures, dates, company names, decoded punctuation) — see `server/sec8kEarnings.test.ts`, 9 tests, all passing, including one end-to-end test through the real fetch layer via an injected fake fetch replaying the same real fixtures (no live network in the suite itself, same principle as edgarForm4.test.ts). Gate 2 (does guidance/results language predict forward returns vs. a size-matched random-entry base rate, per REASONING STANDARD #3) is UNSTARTED and unchanged from open_questions.md's existing hypothesis — ships today as a RAW-DATA overlay only (`/api/data/earnings-language`, `/api/data/earnings-language/history`), no predictive claim, per datacore/README.md's RAW-vs-SIGNAL rule.
- NO UI PAGE YET, DELIBERATELY: mirrors the edgarForm4.ts precedent exactly — the pipeline + API shipped alone in PR #118 (v1.0.47), and the full filings.tsx view followed later in PR #128 (v1.0.55) once the archive had accumulated history. Same sequencing here: a page built today would have nothing to show but a warming-up state. Queued as the natural next PRODUCT action once a few days of archive exist.
- COLLECT-EVERYTHING: archive under `<archive>/earnings8k/`, day-file JSONL deduped by accession, gzip after 2 days — identical shape to edgarForm4.ts's filings archive (discrete dated events, not continuous tracks). Every day not archived is unrecoverable: EDGAR's own full-text search index doesn't expose pre-parsed exhibit text, so this archive is the free BUILD-FIRST substitute (rule #2 — accumulation substitutes for purchase) for what a paid earnings-transcript-history vendor sells.
- Downstream chain traced (REASONING STANDARD #1): filed exhibit text -> archived from today forward -> eventually enough history to attempt gate 2 (guidance-language deltas vs. forward returns, regime-split per STANDARD #2) -> if it passes, a strategy-tournament entrant; if not, the archive still cost nothing extra to build and stands as free ground truth for other language-based hypotheses (job postings, patent filings) that need the same extraction machinery. Zero live-trading-behavior change from this PR — RAW display only, not read by `bot_engine.py` or any strategy, so no interaction with the GOAL priority-3/4 tradeoff to trace beyond that.
- Verified: full server test suite 86/86 passing (77 pre-existing + 9 new, `npx tsx --test server/*.test.ts`); `npx tsc --noEmit` shows zero new errors (61 pre-existing across other files, unrelated to this change, confirmed by name-filtering the output for `sec8kEarnings`/`routes.ts`). Python suite untouched — zero `.py` files modified this PR, one logical change (constitution rule 5).

## 2026-07-04 — [REPAIR] KNOWN BROKEN #5 audit: data modules confirmed wired; closed a real silent-failure blind spot in diagnostics.py (v1.0.66)

- Session start check: CLAUDE.md + all of research/ read this session.
  Loop-health ratio over the last 10 entries (NASA FIRMS back through Port
  Dwell): 1 REPAIR, 1 RESEARCH, 8 PRODUCT — well under the 7/10 thrash
  threshold, no meta-problem to address. `/api/health` on prod: all-ok
  (Alpaca ACTIVE, python bridge ok, bot active, equityPeak=108151.39,
  drawdownPct=0.0% — the persisted high-water mark still holding). No
  critical live break to fix. KNOWN BROKEN #4's ACCESS LIMITATION still
  stands (no DIAG_TOKEN route exists — see wishlist HOLD), so "fix a bug
  seen in audit logs" was not actionable; no experiment has matured to a
  judgeable state this session (Insider Form-4 gate 2, port-dwell gate 2,
  and shadow-fleet gate 1 are all still accumulating history). Per SESSION
  BUDGET, fell through to the next tier: start a new (small, evidence-
  gathering) action. Chose KNOWN BROKEN #5 over a fresh web-research
  fall-through because it is (a) squarely REPAIR MANDATE territory — a
  standing, unresolved constitutional TODO — and (b) fully resolvable via
  READ BEFORE WRITE static analysis alone, unlike #3 (CSP cascade), which
  needs live audit-log access this session doesn't have.
- PRIOR (REASONING STANDARD #10, stated before reading any call sites):
  expected to find at least one of `alt_data.py`/`social_data.py`/
  `institutional_data.py` genuinely orphaned (imported nowhere, or
  imported but never actually invoked) given how many alt-data modules
  this repo accumulated over time, and expected the audit itself to be
  the full session's output with nothing to build.
- FINDING: prior was WRONG on the orphan question — grepped every call
  site for `alt_data`, `macro_data`, `social_data`, `institutional_data`,
  `intelligence`, `finnhub_data`, `alphadesk`, `instrument_selector`,
  `diagnostics`, `tiered_strategy`, `analyze`. All are live-consumed:
  macro/alt/social/finnhub/intel are fetched in parallel inside
  `bot_engine.py:deep_score()` (lines 543-608) with every field read
  downstream into scoring (`macro.get(...)`, `intel.get(...)`, etc. —
  verified past line 609, not just imported-and-discarded);
  `institutional_data.py` feeds `insights.py`, wired to the site's
  `/api/insights/:ticker` route (a user-facing feature, correctly
  separate from the trading loop per GOAL priority 4, not a defect);
  `alphadesk/` wired via routes.ts; `instrument_selector.py` imported at
  bot_engine.py:3026 (note for future sessions: its `intelligence`
  parameter is a DIFFERENT, options-specific dataset from
  `intelligence.py`'s `get_full_intelligence` — same name, unrelated
  data, a naming collision that could mislead a future audit);
  `diagnostics.py` wired into `server/bot.ts`'s Tier-2 cycle (every 5th
  cycle) and its output actually sets `state.positionSizeMultiplier` /
  `state.minScoreThreshold` and can trigger a pause. Nothing in this
  KNOWN BROKEN item was dead code — CLOSED, no wire/retire work needed on
  that front. Prior vs actual, stated per REASONING STANDARD #10: I
  expected an orphan and found none; the correct update is to trust the
  evidence over the prior, not to manufacture a finding.
- REAL GAP FOUND (not what I went looking for, but what the audit
  surfaced): `bot_engine.py`'s five parallel data-source fetchers each
  wrap their call in a bare `except Exception: return {}` with **no
  logging anywhere** — a silent failure by design for graceful
  degradation, which is fine for the SCORE (missing data already degrades
  to neutral), but leaves ZERO trail that a source is down. Cross-checked
  against `diagnostics.py`'s existing API-health monitor
  (`run_diagnostics()` section 4, `api_checks`) and found it already
  tracks polygon/sec_edgar/wikipedia/gdelt/fred cache freshness — but had
  never been extended to cover `social_data.py` (reddit_/gtrends_/
  news_multi_ cache prefixes) or `finnhub_data.py` (fh_ prefix), the two
  sources added later than the original five. This is a live, unmonitored
  blind spot directly adjacent to KNOWN BROKEN #3 (CSP cascade) and the
  HONESTY METRIC: if either source silently died, live scoring would
  quietly run on 3-of-5 signal groups indefinitely with no audit-log
  trace and no session able to tell without re-doing this exact grep.
- FIX (one logical change, `diagnostics.py` only): added `extended_checks`
  (reddit_/fh_ cache-file presence) as a **separate, warnings-only
  bucket** in `run_diagnostics()` #4b — explicitly NOT merged into the
  existing `api_checks`/`failed_apis` list, which drives
  `reduce_position_size` at >=3 failures. Downstream chain (REASONING
  STANDARD #1, traced before writing the diff): merging the two new
  checks into `failed_apis` would silently change the count of monitored
  sources feeding an existing risk-affecting auto-fix -> a position-size
  cut could newly fire in situations that previously wouldn't have
  triggered it -> that is a threshold-behavior change, and RULE REVIEW
  requires evidence + one-at-a-time for exactly this class of change,
  which an audit-driven visibility fix does not carry. Keeping the new
  checks in their own warnings-only bucket means: `problems_summary` (and
  thus the `audit("DIAGNOSTIC", ...)` line bot.ts already logs every 5th
  Tier-2 cycle) will now surface "Extended data sources unavailable:
  [...]" if reddit/finnhub go dark, but `position_size_multiplier` and
  `should_pause` are mathematically untouched by this change — zero
  effect on live trading behavior, pure observability gain. FINNHUB_KEY
  unconfigured (empty or the shipped `YOUR_FINNHUB_KEY_HERE` placeholder)
  is treated as expected-degraded, not a break — mirrors the existing
  ml_model dynamic-criticality false-positive fix
  (test_diagnostic_false_positives.py already exists specifically to
  catch this class of bug).
- Regression tests FIRST (loop-health rule 3), added to
  `test_diagnostic_false_positives.py` (the existing, purpose-built home
  for this exact bug class) rather than a new file: 6 new tests —
  reddit+finnhub both flagged when down; no false-positive warning when
  FINNHUB_KEY is unset (with reddit cached); the shipped placeholder key
  treated as unconfigured; healthy when configured+cached; a
  source-inspection test pinning that `"reddit"`/`"finnhub"` never enter
  the `api_checks` dict literal (the isolation guarantee, verified by
  parsing `run_diagnostics`'s own source — this would have caught a
  future session accidentally merging the buckets); and a
  reduce_position_size isolation test. All 27 tests in the file pass;
  full CI-gate subset (`test_risk_controls.py test_audit_critical.py
  test_diagnostic_false_positives.py test_patches_verification.py`) —
  120 passed, 1 skipped, identical baseline to the pre-existing gate
  (KNOWN BROKEN #6, untouched by this PR).
- Verified: no other file touched (diagnostics.py + its test file only);
  no import cycle introduced (`os.environ` read directly, no new import
  of `finnhub_data`/`social_data` into `diagnostics.py`, avoiding any
  coupling to their heavier dependency surface).
- Version 1.0.65 -> 1.0.66 (read-and-increment). Rollback trigger: if the
  new "Extended data sources unavailable" warning fires persistently in
  production for a source that's actually healthy (a cache-prefix
  mismatch this session's static read missed), revert the `extended_checks`
  block — it is fully additive and isolated, so reverting restores
  exactly the pre-PR observability level with no other side effects.
- MARKET-HOURS NOTE: this session's directive stated the run occurs
  during market hours, so per instruction this PR is left UNMERGED and
  states explicitly that merge should wait until after 4:00 PM ET unless
  the change fixes a critical live break (it does not — pure
  observability addition, isolated from every auto-fix threshold, zero
  live-trading behavior change either way). Not self-merged this session
  regardless of the AUTONOMY AUTHORIZATION default, per the run's own
  instruction.
- STARVED: no — this session's scope (the KNOWN BROKEN #5 audit + the
  gap it surfaced) shipped in full. High-value work remains queued:
  KNOWN BROKEN #3 (CSP cascade, needs live audit-log access) and #6
  (pytest collection), the counterfactual logger, R2 maritime transit
  analytics, and the remaining GEOSPATIAL LICENSING REGISTER items.

## 2026-07-04 — [PRODUCT] NASA FIRMS active-fires layer scaffolded (v1.0.65)

- Session start check: read CLAUDE.md, all of research/, KNOWN BROKEN.
  Nothing there blocks product work: #3 (CSP cascade) and #5 (orphaned
  data modules) both need live-only diagnostics this session can't reach
  (KNOWN BROKEN #4's ACCESS LIMITATION, unchanged since 2026-07-04); #6
  (pytest collection) is a pre-existing, already-scoped-around gap. Per
  the task's own instruction ("product sessions do not preempt the DAILY
  routines' repair duty"), proceeded with product work. Loop-health ratio
  over the last 10 entries: well under the 7/10 REPAIR-thrash threshold
  (mostly PRODUCT/RESEARCH) — no meta-problem to address first.
- Chose the concrete next queued item from the GEOSPATIAL LICENSING
  REGISTER's explicit build-order list (open_questions.md): "(c) FIRMS
  fires — awaiting MAP_KEY human action, may ship scaffolded awaiting_key
  like vessels did, ARCHIVE detections from day one." (a) terrain and (b)
  weather were already shipped; no session had claimed (c) yet (checked
  for a [CLAIMED] tag first per the OPS GOTCHAS double-build rule — none
  found). This is squarely ladder-gate-1-adjacent PRODUCT work (a) from
  the task menu: a RAW-DATA overlay ships ungated per the RAW-vs-SIGNAL
  surface rule (as-is detections + attribution, zero predictive claim),
  and the licensing homework was already done in the 2026-07-04 register
  — no re-research needed, matched the build-order rationale exactly.
- BUILT: `server/nasaFirms.ts` — pure fetch/parse/archive/poll module,
  same shape as `edgarForm4.ts` (discrete dated events, not continuous
  tracks, so it reuses that module's day-file-JSONL-with-dedup archive
  pattern rather than `datacoreArchive.ts`'s adaptive-thinning position-
  track pattern). Key-gated exactly like `vesselStream.ts`
  (`NASA_FIRMS_MAP_KEY`) — `bootFirmsPoll()` no-ops entirely without a
  key, so there is zero upstream traffic or archive writes pre-key.
  `parseFirmsCsv` reads FIRMS' area-CSV by column NAME (not fixed index),
  so it handles both the VIIRS (`bright_ti4`, letter confidence l/n/h)
  and MODIS (`brightness`, numeric 0-100 confidence) column layouts
  without knowing in advance which source served a row — both classify
  through `classifyFirmsConfidence` into the same three-bucket scale.
  Dedup identity (`fireDetectionId`) is satellite+rounded-position+
  acquisition-timestamp, because FIRMS has no stable per-row id of its
  own and re-serves the same detections across overlapping day-range
  polls — verified by the archive test (identical set re-archived writes
  zero the second time).
- Wired in `server/routes.ts`: `/api/data/fires` (enabled:false + reason
  when no key, mirroring `/api/data/vessels`'s shape exactly so the
  client's existing awaiting-key handling needs no new cases) and the
  `/api/data/layers` dynamic-status mapping (fires goes `live` the moment
  the key exists, same as vessels).
- Client (`client/src/pages/datamap.tsx`): new "Environmental" panel
  group (collapsed by default, positioned after Facilities — this group
  now also holds the future R3 roadmap layers: USDA CDL crops, drought/
  soil moisture, USGS groundwater, per open_questions.md); a fires
  useEffect following the vessels awaiting-key pattern; a new `vt-fire`
  SDF icon (mapIcons.ts) tinted by confidence bucket
  (`FIRE_CONFIDENCE_COLOR`); detail card states the LANCE "not for
  safety-of-life use" disclaimer on every detection, not just the layer
  description (the licensing register's stated requirement).
- Downstream chain (REASONING STANDARD #1): key set on Railway -> next
  poll (<=30 min) populates the cache -> `/api/data/fires` flips
  enabled:true -> the layer's `awaiting_key` badge clears to `live` with
  no further code change -> every detection from that point forward is
  archived (no free history exists upstream, so this is the only
  archive-from-day-one window that will ever exist for this root) ->
  a future gate-1/gate-2 signal hypothesis (insurer/utility/timber
  exposure near sustained fire activity) has ground truth to validate
  against once enough history accumulates. Zero effect on the trading
  loop today — this module has no import path into `bot_engine.py`,
  `system_config.py`, `strategies/`, or `server/bot.ts` (SPINOUT-READY
  DATA LAYER boundary), and the layer defaults OFF (opt-in), so
  ZERO-COST-WHEN-OFF holds without any special-casing.
- Regression tests: `server/nasaFirms.test.ts`, 12 cases — VIIRS vs MODIS
  column-name parsing, confidence-scale classification (letter + three
  numeric bands + garbage-input default), dedup-id stability, the
  documented URL shape, key-gating (both `firmsEnabled` and
  `bootFirmsPoll`'s no-op-without-a-key path), a non-ok upstream response
  throwing (no silent empty result), and the archive/gzip/history
  round-trip through real temp-directory I/O (mirroring
  `edgarForm4.test.ts`'s dedup-across-poll-overlap test, with distinct
  synthetic lat/lon per test case — the archive's dedup set is module-
  level and content-keyed, so reusing identical detection content across
  test cases would falsely dedup across unrelated temp dirs, exactly the
  trap edgarForm4.test.ts avoids with unique accession numbers per case).
  `server/layersRegistry.test.ts`'s existing schema-invariant test covers
  the new `layers.json` entry automatically (kind/status/source/
  description all present). All 12 new + 65 pre-existing node tests pass
  (`npm run test:node`, 77/77); Python CI-gate suite untouched by this
  PR (no `.py` files touched) — not re-run, per CLAUDE.md's PROMOTION
  RULE 5 scoping (one logical change; this change has zero Python
  surface).
- Visual verification (PROMOTION RULES rule 6): `npm run build` clean;
  `node scripts/visual_check.mjs --page data` — 0 hard failures at
  390/768/1440 plus the zero-cost-when-off pass. Added the new `fires`
  layer + its `/api/data/fires` fixture to the harness's own FIXTURES
  (it was missing for `shadowstats` too, pre-existing gap, out of this
  PR's one-logical-change scope, not fixed here) specifically so the
  SELF-SEE check exercises the brand-new "Environmental" group's
  collapse/expand/reachability at all three widths — this is exactly the
  defect class the Map v2.4 PR fixed (a panel section existing in code
  but unreachable on screen), so proving it mechanically here rather
  than trusting the pattern by inspection. Screenshots reviewed: new
  "ENVIRONMENTAL 0/1 ON" group renders correctly, collapsed, between
  Facilities and Filings & Flows at 1440px; phone view unaffected
  (panel collapsed by default). Pre-existing warnings only (nav touch
  targets, "Signals — coming soon" clipped-below-fold — the check
  function's own comment says elements that scroll into view below the
  fold are expected to warn, not fail; SELF-SEE's reachability assertion
  for every registered layer, including fires, passed with 0 failures).
- No backtest required (PROMOTION RULES rule 3 scopes that to strategy/
  parameter changes) — this PR touches zero files under `bot_engine.py`,
  `system_config.py`, `strategies/`, or Python at all.
- Version 1.0.64 -> 1.0.65 (read-and-increment, checked against the
  OPS GOTCHAS collision history first).
- Merge timing: 2026-07-04 is a Saturday (confirmed via date computation)
  — markets closed all day, well outside the 9:30-16:00 ET deploy-
  coupling window. Safe to merge immediately.
- STARVED: no — this session's scope (ship the queued FIRMS layer)
  shipped in full: server module, tests, route, client layer, registry
  entry, harness fixture, and doc updates. High-value work remains
  queued: KNOWN BROKEN #3/#5/#6, the counterfactual logger, R2 maritime
  transit analytics, the remaining GEOSPATIAL LICENSING REGISTER items
  (d)-(g), and R5/R6 (Everything Graph, dashboards).

## 2026-07-04 — [RESEARCH] Dual-momentum SPY/QQQ judged out-of-sample — KILLED

- Session start check: `/api/health` all-ok (Alpaca ACTIVE, python bridge ok,
  bot active, equityPeak=108151.39/drawdownPct=0.0% — the 2026-07-03
  persistence fix still holding). Loop-health ratio over the last 10
  experiments.md entries: 1 REPAIR, 7 PRODUCT, 2 RESEARCH — below the 7/10
  thrash-escalation threshold. No DIAG_TOKEN route exists yet (still a
  wishlist HOLD pending human decision), so audit-log/trade_feedback
  inspection remains unavailable to autonomous sessions per KNOWN BROKEN #4's
  access limitation — SESSION BUDGET's "fix a bug seen in audit logs" tier
  was not actionable this session. Chose the next tier: judging a matured
  experiment. The Dual-momentum SPY/QQQ candidate (open_questions.md) had a
  PRIOR and an explicit kill rule recorded 2026-07-03, blocked only on the
  backtest engine (#1), which was rebuilt the same day — the out-of-sample
  test was runnable and simply hadn't been executed yet, buried under a run
  of [PRODUCT] map/geospatial sessions. Judging it outranks starting a new
  experiment or researching new ideas per the SESSION BUDGET order, and it
  is squarely GOAL priority 3 (grow the account) work using REASONING
  STANDARD #2/#4 rigor (regime-split, out-of-sample, discount for variants).
- PRIOR (restated from the 2026-07-03 entry, before running anything this
  session, REASONING STANDARD #10): "edge shrinks but survives ~+1% CAGR
  over SPY ex-2020-21; kill if negative in >=2 sub-periods."
- Built `bot_backtest_subperiods.py`: reuses `bot_backtest.py`'s existing
  pure `fetch()`/`backtest()`/`metrics()` (zero duplication) to split the
  same 2016-2026 SPY/QQQ window into four calendar sub-periods (2016-2019,
  2020-2021 isolated as a known outlier confound, 2022-2023, 2024-2026) and
  compare the dual-momentum (top_n=1, winner-take-all SPY-vs-QQQ, no regime
  filter) config against SPY buy-and-hold in each, applying the
  pre-committed kill rule mechanically (`judge()`) rather than eyeballing.
- RESULT: alpha vs SPY was 2016-2019 -1.09pp, 2020-2021 -13.24pp (excluded
  from the kill count per the prior's own "ex-2020-21" framing), 2022-2023
  +19.44pp, 2024-2026 -11.49pp. **2 of 3 counted sub-periods negative ->
  kill threshold met -> VERDICT: KILL.**
- Prior vs actual (REASONING STANDARD #10): prior expected the edge to
  *shrink but survive*; actual is a clean kill. The pooled 2016-2026 in-
  sample number (+2.2pp CAGR alpha) was almost entirely manufactured by the
  single 2022-2023 sub-period (+19.44pp) — a textbook instance of REASONING
  STANDARD #2 ("works overall often means works in the regime that
  dominated the sample"): 2022 was a rare year where a 2-asset SPY/QQQ
  winner-take-all rotation sidesteps a tech-specific drawdown by holding
  SPY; outside that one regime the strategy underperforms simple SPY
  buy-and-hold. This is exactly the failure mode REASONING STANDARD #4
  warns about in a 1-of-~7-variants-tried search.
- Disposition: NOT promoted to the future strategy tournament. Marked
  KILLED in open_questions.md with the full result and an explicit "do not
  re-propose this exact config" note — prevents a future session from
  re-discovering the same pooled-decade number and shipping it on the
  strength of a single dominant regime.
- Regression tests (new behavior, no existing tests to extend since
  `bot_backtest.py` itself has none — REASONING STANDARD-consistent, pure
  functions only, no network mocking needed): `test_bot_backtest_subperiods.py`,
  9 cases covering `split_dates` inclusive-boundary slicing, `slice_data`
  gap handling (must not synthesize missing dates), `run_subperiod`'s
  insufficient-history guard (must return `{}` rather than crash inside the
  252-day momentum lookback), and `judge()`'s kill-threshold arithmetic
  including the 2020-2021 exclusion and empty-sub-period skipping. All 9
  pass (`python3 -m pytest -q test_bot_backtest_subperiods.py`).
- Verified: full existing CI gate still green after adding the new files —
  `python3 -m pytest -q test_risk_controls.py test_audit_critical.py
  test_diagnostic_false_positives.py test_patches_verification.py` — 114
  passed, 1 skipped (identical to the pre-existing baseline; KNOWN BROKEN
  #6's full-repo-collection issue is pre-existing and untouched by this PR).
- No version bump: this is an offline research/judgment script (network
  fetch of SPY/QQQ closes + pure in-memory backtest), same class as the
  original `bot_backtest.py` — it imports nothing from and is imported by
  nothing in `bot_engine.py`/`system_config.py`/`strategies/`/`server/bot.ts`,
  so it cannot affect live trade attribution (PROMOTION RULES rule 4 exists
  to separate live-code changes' `code_version`, which doesn't apply here).
- Downstream chain (REASONING STANDARD #1): killing this candidate now ->
  it never enters the future strategy tournament on the strength of an
  overfit pooled number -> the tournament's baseline-vs-SPY comparisons stay
  honest (HONESTY METRIC) -> no wasted live-paper capital allocation cycles
  spent proving out a strategy that offline evidence already refutes. Zero
  trading-path impact today (nothing in `bot_engine.py`/`system_config.py`
  changed) — the only effect is closing an open research question with
  evidence instead of leaving it to decay as unexamined backlog.
- STARVED: no — this session's scope (judge the matured dual-momentum
  candidate) fully shipped. High-value work remains queued: KNOWN BROKEN
  #3/#5/#6, the counterfactual logger, Sentinel-2 gate 1, R2 maritime
  transit analytics, and the rest of the geospatial roadmap.

## 2026-07-04 — [REPAIR] Ops: wrong-merge reset emptied PR #148 — recovered, monitor pattern hardened (docs)

- Recurrence of the documented "verify WHICH PR merged" gotcha, now
  with a mechanical fix instead of a re-note (loop-health rule 4:
  recurrence escalates). Sequence: #147 (human-created from a branch
  commit) merged while #148 (v2.4) waited; the hash-only monitor
  fired; the reflexive reset force-push emptied #148's branch and
  GitHub auto-closed it. Recovery: cherry-pick from the local object
  store -> reopened as #149 -> merged clean, zero work lost. Ratchet:
  merge monitors now print the merged commit subject and gate "safe
  to reset" on it matching the expected PR (template proven live on
  #149's watch); OPS GOTCHAS entry strengthened from advice to rule.

## 2026-07-04 — [PRODUCT] Map v2.4 — three production defects fixed, each with A/B-proven enforcement (v1.0.64)

- (1) ETERNAL LOADING root cause: the OWM key WAS fine (activation
  verified on prod same day — see entry below) — the defect was
  client-side: statusFor DROPPED status notes on loading rows, so the
  designed "key activating — auto-retrying" note never rendered and
  the human saw a bare spinner for the whole ~2h activation window.
  Fix: loading rows render their notes; every status change is
  timestamped; a 10s-cadence watchdog upgrades any bare loading >30s
  to an explicit retrying note. DESIGN.md gains the approved
  loading-state rule verbatim. A/B PROOF: probe with a HANGING status
  endpoint — bare loading at t+1.5s, designed retrying note attached
  after crossing 30s (+scan cadence). Harness carries an armed
  assertion (any row loading >30s must have a covnote).
- (2) PERFORMANCE: zero-cost-when-off AUDIT result — all 13 layer
  effects already tear down before any fetch/interval when off; no
  violators found. The real load cost was seven default-ON layers
  fetching at mount. Fix: heavy default-on layers (powerplants,
  insider, shadowstats, portdwell, trains) mount DEFERRED after the
  map's first idle (4s failsafe); base map + aircraft + sites win the
  initial contention. NEW HARNESS STEP: all-layers-off run asserts
  ZERO layer-data API calls (mechanical zero-cost proof) + TTI budget
  2500ms — measured 852ms all-off vs 1016-1579ms with the default
  stack. DESIGN.md gains the zero-cost-when-off rule.
- (3) CONTROL OCCLUSION: zoom controls moved bottom-LEFT (the open
  panel AND the legend both live right-side — bottom-right zoom was
  under the LEGEND, a second occluder the directive didn't know
  about). Self-see now hit-tests zoom/fullscreen controls with the
  panel open. A/B PROOF: reverting to bottom-right fails the harness
  at ALL THREE widths with "map control OCCLUDED by <div
  class='vt-legend'>".
- Gates: node 65/65, python 114/1 skipped, harness green x3 + the new
  zero-cost step, screenshots reviewed (zoom bottom-left at 390).
  Version 1.0.63 -> 1.0.64.

## 2026-07-04 — [PRODUCT] OWM v1.0.63 VERIFIED LIVE on prod (follow-up to the entry below)

- Prod probe sequence: deploy live -> /api/data/weather/global/status
  = "activating" (fresh-key state, retry note surfaced, nothing
  marked broken) -> key ACTIVATED within the watch window -> status
  "ok" and a real temperature tile served through the proxy
  (200 image/png 78KB, /api/data/wxtile/temp_new/2/1/1). Global
  temp/wind fields are live end-to-end; the activation-aware flow
  behaved exactly as designed on a real fresh key.

## 2026-07-04 — [PRODUCT] OWM global temp/wind fields — key-proxied tiles, activation-aware status (v1.0.63)

- Human set OPENWEATHERMAP_KEY in Railway (fresh key, ~2h activation
  on OWM's side). Wired the Tier-1(b) global half: two RAW layers
  (Temperature/Wind, model-derived labeling pinned by test) served
  through OUR tile proxy /api/data/wxtile/... — key never reaches the
  client, and the shared TTL cache bounds upstream calls to
  unique-tiles-per-10min across ALL visitors (free tier is 60
  calls/min; client-direct tiles would blow it on one panning user).
  Zoom capped at 7 (fields are smooth; bounds the cache universe).
- FRESH-KEY RULE implemented exactly as directed: upstream 401/403
  classifies as "activating" -> layer shows LOADING with the note
  "key set — OpenWeatherMap activates fresh keys within ~2h;
  auto-retrying" and re-probes every 10 min; a 5-min negative cache
  stops us hammering OWM meanwhile. Never marked error for a
  fresh-key delay; the note itself says when to re-check the key.
- Tests (5 new): tile validation (allowlist/zoom ceiling/range/
  traversal-shaped input), URL builder key-encoding, the
  401->activating classification with ~2h note, TTL cache expiry +
  bounded eviction, wiring/registry pins (attribution + model-derived
  honesty). OWM's 401-for-inactive-key behavior confirmed live by
  curl (invalid-key probe -> 401), so the activating path is the
  real upstream behavior, not an assumption.
- Verification plan: prod probe of /api/data/weather/global/status
  after the deploy; "activating" expected if the key is under ~2h
  old — recorded as such, not as a failure.

## 2026-07-04 — [PRODUCT] Session close-out: charter + geospatial directive — 10 PRs, queue handed to routines. STARVED.

- Directive execution summary (#136-#145): approved consolidations
  applied (AUDIT CYCLE register + STANDING BEHAVIORS); VISION.md
  installed (labeled reconstruction — verbatim charter still needed
  from the human); wishlist decisions recorded (satellite-AIS
  declined, options HOLD package with verified prices, diagnostics
  EXPLAIN); five new data roots + geospatial licensing register +
  universal-envelope proposal filed (10-agent primary-source research
  pass); Everything Graph v1 spec (flagship, R5) + R6 dashboards;
  Tier-3 Sentinel-2 spec; BUILDS: port dwell analytics v1.0.60
  (directive's highest-value item), terrain hillshade v1.0.61,
  US weather radar v1.0.62.
- STARVED: yes — high-value work remains queued at close (Tier-1
  c-g, Tier-2 buildings, 8-K pipeline, options-chain archiver), all
  with licensing pre-cleared and next-actions filed for the
  [PRODUCT]/[PIPELINE] routines. This is capacity starvation, not
  queue exhaustion.
- Awaiting human: verbatim charter paste; universal-envelope
  approval; options purchase pick (or explicit "stay held"); DIAG
  route approval; four free key/signup actions (OWM, FIRMS, USPTO
  ID.me, Apple EPF); north-star line appended to B3/B4 routine
  prompts (usage_log.md has the exact text).

## 2026-07-04 — [PRODUCT] Tier-1(b): US weather radar layer — NOAA nowCOAST WMS (v1.0.62)

- Built against the licensing register: NOAA nowCOAST base-reflectivity
  mosaic (public domain, no key). COVERAGE HONESTY carried in the
  registry entry, status note, and a registry TEST: US + territories
  only — no free lawful global radar exists (RainViewer
  personal/educational-only + API gutted Jan 2026; Open-Meteo free
  tier non-commercial). Global temp/wind path = OpenWeatherMap free
  key (commercial-OK w/ attribution) — HUMAN ACTION filed; the OWM
  code path is NOT scaffolded (dead-code policy: build when the key
  exists). RAW, default OFF, rendered under all data layers, tiles
  refresh on a 5-min bucket via source.setTiles.
- VERIFICATION (browser egress died mid-session — ops gotcha below):
  GetCapabilities 200; the exact GetMap tile-template URL returns
  200 image/png 5.5KB via curl; CORS verified open
  (access-control-allow-origin: * with an Origin header present);
  in-map source spec printed by probe carries exactly that template;
  layer attaches beneath data layers. In-browser tile-load
  confirmation was NOT possible: headless Chromium lost ALL external
  egress mid-session (even Esri base tiles fail "Failed to fetch
  (0)"; curl fine both direct and via proxy; explicit
  --proxy-server no help; proxy status logs no failures — requests
  never leave the browser). The raster-tiles mechanism used is
  identical to the proven imagery base layer, so residual risk is
  low and confined to runtime tile delivery, which prod will show.
- OPS GOTCHA (avoid re-learning): headless-browser external egress in
  this remote env can die mid-session while curl keeps working; when
  a probe needs tile-load confirmation and the browser is dark,
  curl-verify the exact tile URL + CORS headers instead — that
  covers everything except in-browser compositing.
- Gates: node 60/60 (new registry pin: weather description MUST state
  the US-only limit), python 114/1 skipped, harness green x3 with
  self-see (weather row reachable). Version 1.0.61 -> 1.0.62.

## 2026-07-04 — [PRODUCT] Tier-1(a): terrain hillshade layer — Mapterhorn DEM (v1.0.61)

- First geospatial Tier-1 layer, built against the licensing register
  (Mapterhorn: commercial-OK, terrarium 512px z0-17, attribution via
  TileJSON — verified live this session, TileJSON declares encoding +
  © Mapterhorn). MapTiler free tier rejected per register
  (non-commercial). RAW layer, default OFF (imagery base already
  carries relief; hillshade is the opt-in accent), inserted beneath
  the lowest data layer (symbol/circle/LINE — first probe run caught
  hillshade sitting above the aircraft velocity-vector line layer;
  finder widened, re-verified).
- Also wires the raster-dem source that R4's 3D terrain toggle will
  reuse (setTerrain on the same source) — R4's terrain-source
  prerequisite is now done.
- New test file layersRegistry.test.ts: registry-wide invariants
  (every layer carries kind/status/source/description; SIGNAL layers
  may never be status live — the RAW-vs-SIGNAL rule is now
  machine-enforced) + Mapterhorn attribution pin. This ratchets ALL
  future layers, not just terrain.
- RENDER PROBE (new-layer lesson): toggled via the real panel switch,
  jumped to the Rockies at z9 — source attached, layer ordered under
  data layers, isSourceLoaded=true (DEM webp tiles actually fetched).
  Gates: node 59/59, python 114/1 skipped, harness green x3 with
  self-see. Version 1.0.60 -> 1.0.61.
- Queued follow-up (register note): archive a PMTiles extract of our
  AOIs for provider-disappearance resilience — accumulation
  substitutes for dependency.

## 2026-07-04 — [RESEARCH] Five new data roots filed + geospatial licensing register + universal-envelope proposal (docs PR)

- Charter gap execution items 2-3: a 10-agent research workflow
  (primary-source license verification — every verdict from the
  provider's own ToS/license page, fetched 2026-07-04; ~507k subagent
  tokens, 247 tool uses) produced: (1) five NEW DATA ROOTS entries in
  open_questions.md (8-K earnings language, ATS job postings,
  app-store rankings, USPTO patents, GitHub org activity), each with
  licensing verdicts, honest gaps, priors, and concrete ladder paths;
  build order stated with rationale (8-K first — EDGAR history exists,
  gate 2 not time-blocked). (2) GEOSPATIAL LICENSING REGISTER for
  Tier-1 layers a-g + Tier-2 buildings — notable findings: Open-Meteo
  and RainViewer free tiers unlawful for us; MapTiler free tier
  non-commercial (rejected); NO free current national US pipeline
  vector exists anymore (EIA Atlas dead, HIFLD Open discontinued
  2025) — layer (g) builds from GEM/TX-RRC/OSM with per-source
  coverage honesty. (3) UNIVERSAL ARCHIVE ENVELOPE proposal in
  wishlist (two-tier: dataset manifests retroactively + datum-level
  fields on new pipelines; avoids 3x-ing position-archive volume) —
  awaiting human approval. (4) Two free HUMAN ACTION items filed
  (USPTO ID.me key; Apple EPF enrollment) + Sensor Tower priced
  not-recommended entry with build-first analysis.
- Key licensing kills recorded so no session re-chases them: Motley
  Fool + Seeking Alpha transcripts (ToS bar pipelines), LinkedIn/
  Indeed scraping, Google Play programmatic charts, Appfigures free
  tier, USAJOBS derived use, Apple reviews RSS (verified dead).

## 2026-07-04 — [RESEARCH] Wishlist decisions recorded + options/diagnostics decision packages (docs PR)

- Human decisions 2026-07-04 recorded in wishlist.md: satellite-AIS
  DECLINED (entry retained; revisit trigger = a gated signal that
  specifically needs open-ocean coverage); historical options data
  HOLD (full decision package delivered: unlocks, dependencies,
  vendor prices verified same-day from vendor pages, ranked
  recommendation — Databento $125-credit pilot first, then
  historicaloptiondata.com L2 $1,495 one-off, then ThetaData Pro
  churn ~$160-320 if retention confirmed; free Alpaca chain archiving
  queued regardless); read-only diagnostics EXPLAIN delivered (four
  options risk-assessed; recommended = DIAG_TOKEN-gated whitelist
  route in routes.ts, auth.ts untouched, sanitizer test pinned;
  ships only on explicit approval).
- Sources: options prices from a 10-agent primary-source research
  workflow this session (vendor pricing pages fetched 2026-07-04);
  repo verified private (unauthenticated GitHub 404) for the
  snapshot-option risk assessment.

## 2026-07-04 — [PRODUCT] Tier-3 spec filed: Sentinel-2 facility change detection (docs PR)

- Geospatial directive Tier 3 executed as a spec (build queued for a
  [PIPELINE] session): datacore/SENTINEL2_CHANGE_SPEC.md — weekly
  facility-scale activity indices (yard occupancy, tank shadows, berth
  occupancy) from free Copernicus Sentinel-2 L2A via CDSE windowed COG
  reads; "activity index up/down at facility X," NEVER object counts
  (10m honesty). Generalizes the Cushing tank-shadow idea + the
  imagery-verified facility footprints into one system; port berth
  masks cross-verify the port-dwell AIS counts.
- Ladder paths per facility class stated (tank farms vs EIA Cushing
  weekly; steel yards vs STLD disclosed shipments; ports vs our own
  AIS counts). Nothing surfaces before gate 2 except RAW scene
  metadata (imagery-date honesty rule).
- Sub-meter paid imagery (actual counting) is EXPLICITLY GATED: may
  not enter wishlist with quotes until the free 10m version passes
  gate 2 on any facility class — priced classes noted in the spec
  (archive ~$10-25/km² minimums; monitoring $1000s/mo).
- Docs-only PR, no version bump.

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

## 2026-07-05 — [PRODUCT] Earnings-language full view (8-K Item 2.02) — v1.0.82
- Territory: T-CLIENT (client/src/pages, index.css, scripts/visual_check.mjs)
  + minimal SHARED touches (datacore/layers.json registry entry,
  package.json version bump, this file) per WORKSTREAM PARTITION.
- Context: NEW DATA ROOTS #1's pipeline (server/sec8kEarnings.ts, gate 1
  DATA passed 2026-07-04, v1.0.67) shipped API-only with no UI, same
  sequencing edgarForm4.ts used before filings.tsx landed. This was the
  explicitly queued next PRODUCT item in open_questions.md's GEOSPATIAL
  LICENSING REGISTER section. Highest-value action this session: give
  the already-validated (gate 1) pipeline its user-facing surface,
  mirroring the Form 4 pattern exactly (panel row -> full hash-routed
  view) rather than starting a new pipeline or a new gate-1 effort.
- Build: new `earnings` entry in datacore/layers.json (RAW, live,
  honest gate-2-open language carried into the description — no
  predictive claim, mirrors sec8kEarnings.ts's own doc comment).
  client/src/pages/earnings.tsx — #/data/earnings full view: card list
  (not a table — this data is prose, not rows) reading
  /api/data/earnings-language/history, company-name filter, item-code
  tags, excerpt/expand-full-release toggle (480-char clamp), link-outs
  to both the SEC filing index and the actual exhibit (never embedded —
  same link-out-only rule as vessel/aircraft photo links). datamap.tsx
  wired identically to the insider pattern: LAYER_GROUP -> filings,
  DEFAULT_ON true, its own polling effect against
  /api/data/earnings-language (not the /history route — mirrors
  insider's live-cache-first pattern), panel-row "Open earnings
  language view" button, hash listener alongside the existing
  filingsOpen state.
- SELF-REVIEW CAUGHT (rule 6, before opening the PR): my first pass
  reused `.vt-filings-seclink` (a 32px icon-only button in the existing
  Form 4 view) for the new Exhibit/Filing link-outs, which render as
  icon+text pairs here — a scratch harness run (temporary copy of
  visual_check.mjs pointed at #/data/earnings, deleted after use, not
  committed) caught both links under the 44px touch-target minimum at
  390/768. Fixed with a dedicated `.vt-earnings-linkbtn` class (44px
  min-height) instead of overloading the shared class, and bumped the
  read-full-release button to 44px too (untriggered by the fixture's
  short sample text, but real releases run long — fixed proactively
  rather than waiting to be caught live).
- Gates: `npm run visual` green at 390/768/1440 (0 hard failures;
  pre-existing site-shell nav warnings unchanged; the one new soft
  warning, "Filings & flows 3/3 on" clipped-control, is the same
  below-the-fold false-positive class as the pre-existing Planner/Taxes
  nav warnings — the self-see battery, which actually scrolls and
  verifies reachability, shows the new layer with 0 self-see failures
  and "14 layers toggled clean" in toggle-consistency); ZERO-COST-WHEN-
  OFF unaffected (earnings gates on `enabled.earnings` like every other
  layer); node test:node 121/121; tsc --noEmit shows only pre-existing
  unrelated errors (verified none touch datamap.tsx or earnings.tsx).
  Python suite not touched (no .py files in this diff) — pytest is not
  installed in this sandbox to re-verify, noted honestly rather than
  claimed.
- Downstream chain (REASONING STANDARD #1): new layer row -> one more
  default-on poll (60s interval, same cadence as insider) -> the
  ZERO-COST-WHEN-OFF gate proves this is skipped entirely when the
  layer is off, so no new baseline cost for users who don't want it;
  when on, a second small JSON poll alongside insider's — negligible
  next to the 10k-aircraft budget already measured in this harness.
- Not attempted this session (correctly out of scope): gate 2 (does
  guidance language predict forward returns) — this PR is a RAW display
  of the pipeline's existing gate-1-passed output only, per the
  SPINOUT-READY / RAW-vs-SIGNAL rule.
- STARVED: no — this was a fully-specified queued item, executed start
  to finish in one PR.

## 2026-07-05 — [PIPELINE] CFTC Commitments of Traders (COT) — rescued from a stalled dirty PR and merged (v1.0.86)

- Territory: T-BOT (new top-level Python module wired into
  server/bot.ts's tier3Strategic; storage_config.py path additions).
- SESSION-START CHECK (Repair Mandate + system health): full offline
  Python gate re-verified green (328 passed, 1 skipped, after this PR's
  17 new tests — was 311/1 before). No KNOWN BROKEN item required
  [REPAIR] this session; #10 (dead SCORE_BAND config) stays correctly
  deferred pending shadow_portfolio history, per its own entry.
- FINDING (the actual highest-value action this session): PR #134
  ("CFTC Commitments of Traders — free positioning-data pipeline,
  v1.0.58") was opened 2026-07-04 from a since-abandoned branch
  (claude/dazzling-planck-64joy2), fully built (`cftc_cot.py`,
  `test_cftc_cot.py`, 17/17 offline tests, live-verified 156/156-week
  backfill across all 7 symbols, 0 validation rejections), but never
  merged — `mergeable_state: "dirty"`, `total_count: 0` check runs, main
  never advanced past its pre-PR base. This is a live instance of the
  OPS GOTCHA already on file ("a dirty claude/* PR stalls SILENTLY: no
  merge ref -> pull_request workflows never start -> no checks, no
  automerge, no error"). Consequence: an entire validated EDGE DOCTRINE
  #1 pipeline sat invisible for a full day — zero references anywhere
  in this file or open_questions.md on main (`grep -rn cftc` across
  both confirmed zero hits before this entry), meaning any session that
  read research/ before this one would have had no idea the work
  existed and could have duplicated it from scratch.
- WHY REBUILD-FROM-DIFF INSTEAD OF A GIT MERGE/CHERRY-PICK: the stale
  branch's base predates ~50 merged PRs (package.json version 1.0.57 vs
  current 1.0.85; research/experiments.md and open_questions.md have
  been rewritten under it many times over) — a cherry-pick would
  conflict on every touched shared file. Per EDGE DOCTRINE #3 (COMPILE
  KNOWLEDGE INTO CODE — never re-reason what's already been reasoned),
  the code itself (`cftc_cot.py`, `test_cftc_cot.py`, the
  storage_config.py path additions, the tier3Strategic wiring diff) was
  reused byte-for-byte from the stale PR's diff; only the
  version-dependent surroundings (package.json bump, this log, the
  open_questions.md entry, the bot.ts insertion point) were re-applied
  fresh against current main.
- RE-VERIFIED FRESH (did not just trust the year-old PR description):
  `python3 -m pytest -q test_cftc_cot.py` 17/17 pass; CI's 4-file
  offline subset 120 passed/1 skipped (unchanged baseline); full bare
  `pytest -q` 328 passed/1 skipped (was 311/1 — net +17, zero
  regressions); `npx tsc --noEmit` diffed before/after the bot.ts change
  line-for-line: exactly one new error, `cotOut.trim()` on a `Buffer`
  return type, the identical pre-existing pattern already present at
  every other `execPythonSerialized(...).stdout.trim()` call site in
  this same function (12 such errors already existed; now 13) — no new
  error *category*; `npm run build` clean. LIVE-VERIFIED against the
  real CFTC Socrata API from this session's sandbox: 156/156 weeks
  backfilled for all 7 symbols (GLD/SLV/USO/CORN/TLT/SPY/QQQ), 0
  accounting-identity rejections; second immediate call confirmed the
  20h staleness guard returns `{"status": "skipped"}` with zero network
  calls. Local test-run archive/checkpoint files removed after
  verification (not part of the repo; would only ever live on the
  Railway volume).
- WIRED AT GATE 1 ONLY: `server/bot.ts` `tier3Strategic()` step 5 calls
  `run_daily_update()` every hourly cycle; the module's own guard makes
  23 of 24 calls a free file-mtime check. Deliberately NOT wired into
  `deep_score`/`macro_data` — gate 2 (does COT-index positioning predict
  forward returns vs. a random-entry baseline) is unstarted, logged with
  a stated prior and kill criteria in open_questions.md.
- DOWNSTREAM CHAIN (REASONING STANDARD #1): one more hourly subprocess
  call in tier3Strategic -> guarded to a cheap file check on 23/24 calls
  -> zero effect on deep_score/scoring/sizing/position count (nothing
  reads COT data yet) -> zero live-trading behavior change from this PR.
  The only observable effect until gate 2 ships is a growing archive
  file on the Railway volume.
- PR #134 (the stalled original) is being closed as superseded by this
  session's PR, which carries its full delta forward — no unique work
  from #134 is lost. Recorded here so the supersession is traceable from
  the log, not just the PR close comment.
- GATE 2 NOT ATTEMPTED THIS SESSION — same prior and kill criteria as
  the original 2026-07-04 build (real edge expected on the commodity
  contracts, weak-to-none expected on SPY/QQQ given the legacy report's
  weaker financial-futures classification); ready to run next session
  now that the backtest engine already exists and 156 weeks are
  archived from day one.
- STARVED: no — recovering already-validated, already-tested work that
  was about to be silently lost was higher expected value than starting
  a brand-new pipeline from zero this session.

## 2026-07-05 — [RESEARCH] CFTC COT gate 2 (SIGNAL) screen: 2 of 7 symbols not killed, 5 killed; overlapping-window caution filed (docs + cot_gate2_test.py)

- Territory: none of T-DATACORE/T-CLIENT/T-BOT touched — this session
  ran a pure statistical screen (`cot_gate2_test.py`, no imports of
  bot_engine.py/system_config.py/server code) against the COT gate-1
  archive (`cftc_cot.py`, merged the same day by a concurrent session
  this session found and did not duplicate — see below). No version
  bump: nothing on the live trading or server path changed, matching
  the repo's existing "docs(research)"/verification-PR convention
  (e.g. #252) rather than PROMOTION RULES #4, which exists for
  `code_version` attribution of LIVE-behavior changes only.
- CONCURRENCY CHECK (before picking work): a second, concurrent session
  (session_01BAU2qmw3UYfhQ6sG9MjfMv) was actively merging BUILD ORDER 5
  items #1-4 (FINRA short-volume #254, CFTC COT disaggregated #255,
  Wikimedia attention #256, FAA airport status #257 — all merged
  2026-07-05 19:54-20:16 UTC) and had PR #258 (CBP border waits, B5 #5)
  open at 20:18 UTC, discovered via `list_pull_requests` before any
  local edit was made. Rather than duplicate that T-DATACORE queue or
  race it on shared files, this session picked the next item the
  SESSION BUDGET priority order actually ranks higher than "start a new
  pipeline": judging the COT gate-2 hypothesis already on file with its
  prior pre-stated (open_questions.md), using data that was fully
  backfillable NOW (no forward-accumulation wait, unlike EDGAR Form4).
  Branch `claude/funny-fermat-ic1oms` was stale relative to `main` (it
  predated #253-257); reset to `origin/main` before starting, per the
  merged-branch restart rule, rather than rebasing old history that was
  already superseded.
- PRIOR (restated from open_questions.md, written before this run):
  expect a small real edge on the commodity contracts (GLD/SLV/USO/
  CORN) and little-to-no edge on SPY/QQQ.
- METHOD: `cot_gate2_test.py` (new; `test_cot_gate2.py` 12/12 pure-
  function unit tests, no network). For each of the 7 tracked symbols,
  fetched the full 156-week COT history live from CFTC's Socrata API
  and daily price bars via `backtest_v2.fetch_bars` (Yahoo path — no
  Alpaca keys in this sandbox); entry anchored to the first trading day
  STRICTLY AFTER the Friday publish date (report_date + 3d), never the
  Tuesday as-of date (no-lookahead, REASONING STANDARD #7). Bucketed by
  the already-computed non-commercial COT index: extreme_high >=80,
  extreme_low <=20, vs. the same-symbol all-weeks baseline, at 20d and
  60d forward horizons.
- RESULTS (mean forward return %, baseline vs. extreme buckets; n =
  weeks in bucket with a resolved forward return):
  | Symbol | 20d base (n) | 20d high (n) | 20d low (n) | 60d base (n) | 60d high (n) | 60d low (n) |
  |---|---|---|---|---|---|---|
  | GLD | 2.31 (152) | 1.94 (52) | 2.45 (7) | 7.70 (144) | 7.02 (52) | 5.97 (7) |
  | SLV | 3.41 (152) | 1.85 (44) | 1.98 (6) | 11.40 (144) | 7.78 (44) | 9.99 (6) |
  | USO | 2.16 (152) | 2.24 (14) | 1.57 (58) | 7.22 (144) | -4.54 (14) | 15.48 (58) |
  | CORN | -0.81 (152) | -0.38 (29) | -0.84 (41) | -1.83 (144) | 0.42 (25) | -2.30 (41) |
  | TLT | 0.07 (152) | 1.06 (42) | -4.18 (11) | 0.36 (144) | 0.89 (42) | -0.16 (10) |
  | SPY | 1.51 (152) | 0.94 (26) | 0.84 (14) | 4.84 (144) | 6.24 (26) | 3.00 (14) |
  | QQQ | 1.93 (152) | 2.14 (43) | 0.44 (15) | 6.00 (144) | 4.20 (43) | 2.99 (10) |
- VERDICT (kill criterion: no separation from baseline): **KILLED** —
  GLD, CORN, SPY, QQQ (SPY/QQQ exactly as the prior expected; GLD's
  low-extreme bucket was n=7, too thin to have shown anything either
  direction), and TLT (its one large deviation, extreme_low 20d -4.18
  vs +0.07 baseline, evaporated at 60d, -0.16 vs +0.35 — a single-
  horizon flash, not a cross-horizon-confirmed effect — REASONING
  STANDARD #4 in action). **NOT KILLED, carried forward, NOT a gate-2
  pass**: SLV (extreme_high consistently below baseline both horizons,
  n=44) and USO (both legs point the same mean-reversion direction the
  prior predicted, n=58 on the low side, 60d gap more than double
  baseline — the largest effect in the table, and oil is exactly where
  REASONING STANDARD #5's hedger-information-asymmetry argument is
  strongest).
- METHODOLOGICAL FINDING (compiled into open_questions.md so it applies
  to every future weekly-cadence gate-2 test, not just COT): weekly
  sampling with a 60-trading-day forward horizon means consecutive
  observations overlap ~11/12 — USO's raw n=58 is roughly 5 effectively
  independent windows once autocorrelation is accounted for. This
  screen used raw means only, which is enough to KILL (a flat or
  wrong-signed mean is disqualifying regardless of overlap) but NOT
  enough to PASS (a large raw gap can come from a handful of correlated
  episodes — one persistent oil rally, say). SLV/USO stay open pending
  either a block-bootstrap/overlap-adjusted significance test or a
  non-overlapping-window redesign, not promoted on this result alone.
  Discount stated per REASONING STANDARD #4: 28 symbol x bucket x
  horizon comparisons run in one pass; 2 surviving out of 28 is within
  what pure noise would produce.
- Gates: `python3 -m pytest -q test_cot_gate2.py test_cftc_cot.py` 30/30
  pass (this sandbox lacks numpy/lightgbm/the full requirements set, so
  the repo-wide bare `pytest -q` could not be re-verified end-to-end
  this session — noted honestly, matching the precedent in the
  2026-07-05 earnings-language entry, rather than claimed; `cot_gate2_test.py`
  and its test module import only `cftc_cot`/`backtest_v2`/stdlib, so
  this gap does not affect the tested surface). Live-verified against
  the real CFTC Socrata API and Yahoo Finance from this sandbox — 156/
  156 weeks and 0 rejected COT records for all 7 symbols this run too
  (matches the concurrent session's contemporaneous verification of the
  same archive).
- STARVED: no — this was a fully-specified queued gate-2 judgment,
  executed start to finish, that avoided colliding with the concurrent
  session's T-DATACORE build-order work.

## 2026-07-07 — [PRODUCT] Streams inventory tab (DATACORE MAXIMUS Phase 4 first item + Phase 5 coverage ratchet) — v1.0.167

- FIRST: v1.0.166 (power-grid TX pilot, #298) VERIFY CONFIRMED post-deploy
  exactly as pre-stated: `curl -r 0-100 voltradeai.com/tiles/power_tx.pmtiles`
  → 206, and /api/data/layers serves the powergrid entry with the full
  coverage-honesty description. (First registry probe during the deploy
  replace-window returned the old container's registry — resolved on
  re-poll; the R13 lesson about replace-windows applies to verification
  probes too.)
- TERRITORY: T-CLIENT + T-DATACORE server aggregate (one logical change:
  a user-visible surface with its API, per CLAUDE.md client-rule).
- WHAT: one call = the whole archive census. server/streamsInventory.ts
  joins the STATIC manifest envelope (datacore/manifests/*.json — source,
  license, attribution, started, cadence, confidence_model-as-hypothesis)
  with DYNAMIC disk facts (files, bytes, newest-file age, records in
  newest file, latest-record peek). /api/data/streams serves a 5-min
  cache only (event-loop rule; eager boot). #/data/streams overlay
  (hash pattern like filings/graph): mobile-first stacked cards, health
  filter chips with counts, text filter, expandable detail + JSON peek;
  launcher row atop the layer panel ("nothing ships invisible").
- DESIGN DECISIONS (traced): (1) disk-scan over per-module status joins —
  the subagent route-map showed 12 manifested streams have NO server
  route (session-run Python writers: railep724, eiaweekly, cpcdegreedays,
  sentinel*, legacy cot) and would be invisible to any status-function
  join; the archive directory is the one surface EVERY stream touches.
  (2) records counted in the NEWEST file only (free — the peek already
  reads it); full-archive record counts stay platformStats' job (10-min
  TTL) — duplicating that scan every 5 min would double volume reads for
  a number files/bytes already proxy. (3) HEALTH is DERIVED (age vs
  cadence-keyword threshold, generous defaults) and labeled as such; raw
  age always shown so the derivation can be second-guessed; missing dir
  = "no-data" row, never hidden. (4) peek capped at 8MB compressed with
  an honest skip note; torn tail (partial write) flagged, never parsed
  around.
- RATCHETS (Phase 5, landed with the feature): streamsInventory.test.ts
  INVENTORY-COVERAGE test — every datacore/manifests/*.json must appear
  in the inventory output (the aggregator enumerates the dir at runtime,
  so a new manifested stream surfaces on /data mechanically; a stream
  missing from the inventory is a failing build). streams page
  registered in visual_check.mjs PAGES → layout battery gates it at
  390/768/1440 from now on. CJS-bundle gotcha compiled: import.meta.url
  does not survive the dist/index.cjs esbuild — runtime repo-file access
  uses process.cwd() (platformStats precedent).
- GATES: tsc 64 (baseline, unchanged); node tests 331/331 (326 + 5 new);
  npm run build OK (power_tx.pmtiles confirmed still bundled);
  npm run visual PASS 390/768/1440 including the NEW streams page
  (fixture exercises all four health states); 390px streams screenshot
  self-reviewed against DESIGN.md. Version 1.0.166 → 1.0.167.
- VERIFY (pre-stated): post-deploy /api/data/streams returns count=41
  (or current manifest census) with real archive facts — occvolume
  "live" with a real peek, key-gated-but-active streams (griddemand,
  cropconditions) showing files>0, session-run Python streams honestly
  "stale"/"no-data" (their writers never ran on Railway — that is the
  honest reading, not a defect); /data layer panel shows the "Streams
  inventory" launcher; #/data/streams renders the cards on a phone
  viewport.
- STARVED: no — queue continues (census builds JODI → FINRA Query
  cluster → SEC FTD; Phase 3 imagery freshness; grid item 2 US-full).

## 2026-07-07 — [REPAIR] R14: runtime datacore reads silently empty on prod — image ships dist/ only — v1.0.168

- FOUND BY the v1.0.167 pre-stated VERIFY: post-deploy /api/data/streams
  returned {count:0, streams:[]} with a non-null scan time — the scan ran
  and found NO manifests. ROOT CAUSE (traced to the packaging layer, not
  the code): the frozen Dockerfile ships dist/, content/, *.py,
  strategies/, alphadesk/ — the repo datacore/ tree is NOT in the runtime
  image, so every server-side runtime DISK read of datacore/ files
  resolves to nothing on Railway. Statically-imported JSON (layers.json,
  entity_spine.json, us_power_plants.json, …) is unaffected — esbuild
  inlines it at build time, which is why 20+ surfaces work fine and this
  class stayed invisible.
- PRE-EXISTING INSTANCE PROVEN, not just mine: /api/data/platform/stats
  serves sentinel2_last_reading:null on prod — the audit-defect-#9
  "repair" (D2, 2026-07-05) shipped a cwd read of
  datacore/sentinel2/readings.jsonl that has NEVER worked in production;
  null was a designed value so nothing looked broken. Honest correction
  to that entry: the fix worked locally and its prod path was never
  live-verified for the non-null case. The verify-the-positive-case
  lesson is the real ratchet here.
- FIX (class, not instance; Dockerfile untouched — frozen):
  (1) script/build.ts (not frozen) stages the runtime-read set into
  dist/datacore/ — manifests (172KB) + sentinel2/readings.jsonl (11KB);
  NOT the whole datacore/ (254MB, sentinel2 chip corpus).
  (2) server/repoFiles.ts repoDataPath(): working-tree path when present
  (dev/CI), dist/ fallback (prod image), direct path returned when both
  missing so callers surface the miss. platformStats + streamsInventory
  now resolve through it.
  (3) NEVER SILENTLY EMPTY: buildStreamsInventory payload carries
  manifest_dir_found + an explicit "packaging defect" note when the dir
  is missing — an empty inventory can no longer masquerade as an empty
  archive.
- RATCHETS: repoFiles.test.ts — resolver fallback battery (3 layouts) +
  source pin on script/build.ts's copy step (its removal would regress
  prod with no other failing signal) + never-silently-empty payload
  assertions. 6 new tests.
- GATES: node 337/337 (331 + 6); build verified staging 42 manifests +
  readings.jsonl into dist/datacore/; tsc 64 baseline; client untouched
  (no harness run needed). Version 1.0.167 → 1.0.168.
- VERIFY (pre-stated): post-deploy /api/data/streams returns count=41
  with manifest_dir_found:true and real archive facts (occvolume live
  with peek), AND /api/data/platform/stats serves a NON-NULL
  sentinel2_last_reading (2026-06-27 per the repo readings file) for the
  first time in production.

## 2026-07-07 — [PIPELINE] JODI oil closing stocks — census build #3 (session-run, git-artifact) — v1.0.169

- TERRITORY: T-DATACORE (scripts/ + datacore/). Census rank #3 (#1 OCC
  built; #2 EPA CAMD key-blocked 9a).
- SOURCE probed + FULL-FILE SCANNED before coding (the shape-probe rule
  earned its keep again): world_Primary_CSV.zip = 23.3MB zip / 6,872,400
  rows / 118 REF_AREAs / dense 10-flow x 4-product x 5-unit grid /
  latest 2026-04 (the documented ~2-month lag). Header probe alone
  MISSED a third not-available marker: 'N/A' (29,736 occurrences in the
  filtered subset) — found because the parser counts every unparsed row
  instead of dropping silently; markers are '-', 'N/A' (not available),
  'x' (not applicable) — skipped, never zeroed (0.0 is a real level).
- BUILD (eiaweekly precedent — monthly source, session-run Python,
  whole-file-rebuild git artifact; ZIP handling + 283MB CSV are the
  wrong cost shape for the Node poller): scripts/jodi_oil.py filters
  FLOW=CLOSTLV (closing stock LEVELS) x UNIT=KBBL x 4 primary products
  x full history -> datacore/jodi/primary_stocks.json (1.4MB, 350
  series, ~61k points, assessment codes preserved verbatim). Refuses to
  write on zero series; header change raises loudly.
- GATE-1 FIRST LOOK (honest non-match, filed): JODI US CRUDEOIL 2026-04
  = 705,195 kbbl vs our own EIA artifact — commercial 459,495 /
  commercial+SPR 857,419. A DEFINITIONAL gap (lease stocks, in-transit,
  SPR treatment), not a data error; reconciling JODI's stock definition
  against EIA monthlies is the gate-1 workup before any signal use.
  Saudi 139,967 kbbl is in the plausible historical band. HYPOTHESIS
  stays gate-locked: non-OECD stock builds (SA/AE/IN) invisible in EIA
  weeklies -> Brent structure; 2-month lag deters fast money.
- TESTS: test_jodi_oil.py 7/7 (marker honesty incl. real-zero-stays,
  filter correctness, sort order, header-change alarm, refuse-empty,
  committed-artifact coherence). Full pytest 456 passed 1 skipped.
- VERIFY (pre-stated): next monthly rerun (~2026-07-19 publication)
  appends 2026-05 points as a clean git diff; the streams inventory
  shows the jodi manifest (count 42) with volume-side no-data honesty
  (session-run writer — expected, not a defect).

## 2026-07-07 — VERIFY CONFIRMATIONS: v1.0.167/168/169 all live (appended for the record; rides with the next PR)

- R14 (v1.0.168) VERIFIED ON PROD exactly as pre-stated:
  /api/data/streams now serves the full census — count=42 (41 + jodi,
  the v1.0.169 deploy landed during verification), manifest_dir_found:
  true, occvolume "live" with a real 153,181-record newest-file peek,
  griddemand + cropconditions "live" with files, railep724/eiaweekly
  honestly "no-data" (session-run writers — expected). AND
  /api/data/platform/stats serves sentinel2_last_reading=2026-06-27
  (age 10d) — NON-NULL IN PRODUCTION FOR THE FIRST TIME since the D2
  surface shipped 2026-07-05. The 10-day age now visible is the
  staleness signal that repair was built to surface; a sentinel2
  pipeline rerun is a queued follow-up, and the age being loud is the
  feature working.
- v1.0.167 streams tab: end-to-end live (was blocked only by the R14
  packaging defect its own pre-stated verify caught — the
  verify-the-positive-case discipline paid for itself same-day).
- v1.0.169 JODI: manifest in the prod inventory (count 42), artifact
  in git; next monthly rerun ~2026-07-19.

## 2026-07-07 — [PIPELINE] FINRA Query API part 1: short interest + threshold list (census build #4a) — v1.0.170

- TERRITORY: T-DATACORE. Census rank #4. Contract LIVE-VERIFIED by
  subagent workup before any code (probe-first rule): POST-only filters
  (GET params SILENTLY IGNORED — verified), Accept JSON (GET CSV default
  is unquoted and corrupts on issueName commas — 704/3000 sampled rows
  broke), record-total header = count primitive, HTTP 204 = empty (not
  200), limit>5000 silently clamped, partitions endpoint newest-first,
  listed-but-empty partitions exist (monthlySummary 2014-16), unordered
  pagination. Every quirk is in the module header + manifests.
- BUILD: server/finraQuery.ts — partition-diff archiver (newest N per
  6h cycle), count-verified fetch (a partition is archived ONLY when
  rows === record-total; unordered pagination makes a partial read
  unusable — the CNMS trailer-guard analog), SI gz-on-write (~1MB/
  partition), threshold plain (~17 rows/day), env-gated full backfill
  (FINRA_QUERY_BACKFILL=1, OFF — R8 lesson; SI 204 partitions ~80MB gz,
  threshold 2,635 partitions ~8MB). Route /api/data/short-interest
  (cache-only): SI leaderboards + threshold list.
- LIVE END-TO-END from session before commit: settlement 2026-06-15 =
  22,180 records (matches workup count exactly — pagination verified),
  threshold 2026-07-02 = 17 names. The live run CAUGHT a data artifact:
  changePercent +259,847,500% on a near-zero previous position (NVRI) —
  leaderboards now floor BOTH ratio endpoints (SI_PREV_FLOOR added,
  test pinned with the live number).
- SCOPE: part 1 of 2 — ATS venue summaries (weekly/monthly/blocks,
  66k-210k rows/week) are a separate build with their own volume
  budget. SEC FTD workup ALSO complete (subagent, filed for the next
  build): URL pattern is NON-uniform (202605a lives at
  /files/data/other/ TODAY; 5 pattern eras since 2004; index-page
  scrape is the robust source), half-month boundary is 1-14/15-EOM not
  1-15/16-31, two-line trailer = free checksum, PRICE "." = null,
  QUANTITY is a level not a flow, earliest data 2004-03-22.
- GATES: node 346/346 (337 + 9 new); tsc 64 baseline; manifests
  envelope green (44 manifests). Version 1.0.169 → 1.0.170.
- VERIFY (pre-stated): post-deploy /api/data/short-interest serves
  settlement_date 2026-06-15 with si_records=22180 and threshold
  trade_date 2026-07-02 count=17 (or newer if FINRA publishes);
  /api/data/streams count=44 with both new streams "live" after first
  poll. Next SI publish (~Jul 9-10 for the 2026-06-30 settlement)
  should appear within one 6h poll of dissemination.

## 2026-07-07 — [PIPELINE] SEC fails-to-deliver stream (census build #6) — v1.0.171

- TERRITORY: T-DATACORE. Census rank #6, the resale-safe public-domain
  source. Contract from the subagent workup (filed in the v1.0.170
  entry); build verified LIVE end-to-end before commit: 3 real half-
  month zips fetched+parsed+archived — 202606a parsed 58,328 rows
  matching the file's own trailer checksum EXACTLY, and 202605a landed
  via the /files/data/other/ fallback (the nonstandard-path case the
  workup flagged works in practice).
- BUILD: server/secFtd.ts — dependency-free single-member ZIP reader
  (central-directory walk, stored+deflate; build-don't-buy applies to
  dependencies too), trailer-checksummed parser (BOTH trailer lines
  enforced — count and share-quantity; mismatch refuses the file),
  verified URL fallback chain, half-month period math (1-14/'a',
  15-EOM/'b' — the workup's boundary correction, pinned in tests incl.
  the 14th/15th edge), period-level dedup gz-on-write archive, 12h
  poll. Route /api/data/ftd (cache-only) with the level-not-flow note
  and stated quantity floor.
- HONESTY NOTES: PRICE "." → null never zero; CINS letter-prefix
  cusips preserved; QUANTITY documented as a BALANCE (level) —
  composite math must diff it, not sum it. Raw FTD spikes stated as
  maximally crowded in manifest + route note — this stream exists for
  the settlement-stress composite (threshold persistence x FTD x short
  volume), which is now fully ingredient-complete (finrathreshold +
  secftd + finrashortvol all recording). The composite itself is a
  queued [RESEARCH] task with its own gate-1 plan, not part of this PR.
- GATES: node 354/354 (346 + 8 new); tsc 64 baseline (one Set-iteration
  regression caught and fixed pre-commit); manifests envelope green (45
  manifests). Version 1.0.170 → 1.0.171.
- VERIFY (pre-stated): post-deploy /api/data/ftd serves period 202606a,
  rows 58328, newest_date 2026-06-12 (or newer once 202606b publishes
  ~Jul 15 — within one 12h poll); /api/data/streams count=45 with
  secftd live after first poll.

## 2026-07-07 — [PIPELINE] OCC deep backfill mechanism — rolling-window rescue, env-gated (v1.0.172)

- TERRITORY: T-DATACORE. The census-#1 follow-up: OCC purges data off
  the BACK of its rolling 2-year window DAILY — every day not captured
  is permanently lost, and only ~5 days around 2026-07-02 are archived
  so far. Session-run capture cannot reach the Railway volume, so the
  mechanism is server-side and env-gated (FINRA deepBackfillIfSparse
  precedent, R8 crash-loop lesson): OCC_DEEP_BACKFILL=1 opt-in,
  done-marker single pass, runs only after current data is up.
- DESIGN POINTS: walk is OLDEST-FIRST (the purge eats the back edge —
  rescue it before the front); "Report date cannot be prior to  2
  years" (aged_out) is an honest no-op; gz-on-write via the existing
  archiveOccDay so nothing accumulates plain mid-pass; 1.5s politeness
  spacing (~5MB/request, ~500 requests ≈ 15-20 min); window and
  spacing injectable so the test awaits a real 10-day pass instead of
  detaching a 730-day walk into the test process.
- VOLUME BUDGET (the Mike decision, filed in wishlist): full capture ≈
  500 trading days x ~1MB gz ≈ +500MB; prod archive measured 0.25GB
  total today (vessels 102MB, finrashortvol 82MB the largest). Flag
  stays OFF until Mike confirms volume capacity.
- GATES: node 355/355 (1 new test: gated-off no-op, oldest-first
  order, aged_out handling, done-marker idempotence); tsc 64. Version
  1.0.171 → 1.0.172.
- VERIFY (pre-stated): flag unset → prod behavior unchanged (only the
  existing 5-day refresh). When Mike sets OCC_DEEP_BACKFILL=1: one
  boot logs "deep backfill: walking 730 calendar days", the occvolume
  dir grows toward ~500 day-files, backfill_done.json appears with
  days_fetched, and the flag can then be removed.

## 2026-07-07 — [PRODUCT] Phase 3a: live imagery capture-date readout on /data — v1.0.173

- TERRITORY: T-CLIENT. NOT a DESIGN.md amendment: the human-approved
  imagery-honesty rule (2026-07-04) already REQUIRES dates "where
  available" — they became available when the census verified the
  World_Imagery identify endpoint (§3 #9), so the old "capture date
  unavailable (Esri base tiles)" note was overtaken by facts and this
  change is rule-REQUIRED.
- BUILD: debounced moveend identify at the VIEW CENTRE (CORS verified
  Access-Control-Allow-Origin:* live); picks the metadata level
  spanning the current zoom; chip states: "imagery at centre:
  YYYY-MM-DD · SOURCE" / "capture date unknown at this zoom" (verified
  real case: low-zoom TerraColor NextGen carries no DATE) / "capture
  date unknown" on transport failure — never fabricated, never
  stale-implying. Esri-terms reading applied: recency check displayed
  ON the imagery it describes; client-side only, nothing archived, no
  API route (census licensing note: internal recency checks fine, no
  redistribution).
- SELF-REVIEW CAUGHT TWO OCCLUSIONS the harness didn't (the reason the
  screenshot rule exists): draft 1 covered the zoom-out button (Z3
  class); draft 2 clipped the © Esri attribution — a LICENSING surface.
  Final position (left 52px / bottom 32px) clears both; offsets
  documented in the CSS with the reason.
- RATCHET (Phase 5): harness now FAILS any map-page run where the
  imagery base is on and [data-testid=imagery-date] is absent or not in
  a designed state — external Esri calls abort in the harness, so the
  honest unknown state is what CI pins forever.
- GATES: tsc 64 baseline; build OK; visual harness PASS 390/768/1440
  full run + data-page re-runs after each occlusion fix; 390px
  screenshot self-reviewed (chip beside zoom column, attribution fully
  readable). Version 1.0.172 → 1.0.173.
- VERIFY (pre-stated): on prod /data, panning to Cushing at z≥12 shows
  "imagery at centre: 2025-10-21 · Vantor" (the live-probed value;
  date will drift as Esri updates tiles — drift is the feature);
  zooming out past z11 flips to "capture date unknown at this zoom".

## 2026-07-07 — [PIPELINE] OCC backfill APPROVED + default-on; v1.0.173 verify confirmed — v1.0.174

- HUMAN DIRECTIVE (2026-07-07): volume headroom confirmed (~2GB of 5GB
  used; backfill adds ~500MB gz) and OCC_DEEP_BACKFILL enablement
  instructed. Railway env is not writable from a session, so the
  mechanism flips to DEFAULT-ON in code with an explicit opt-out
  (OCC_DEEP_BACKFILL=0) — the R8 caution was about UNAPPROVED
  backfills; this one is approved, single-pass (done-marker), and
  gz-on-write. Wishlist 9e closed as RESOLVED.
- ALSO RECORDED: v1.0.173 (imagery capture-date chip) VERIFY CONFIRMED
  on prod — the deployed bundle carries the chip states (bundle-string
  probe), completing verification of every release this session
  (v1.0.166→173, PRs #298-#302, #304-#308 all merged + verified).
- GATES: node occVolume battery 6/6 (opt-out case now pinned); tsc 64
  baseline. Version 1.0.173 → 1.0.174.
- VERIFY (pre-stated): first boot after deploy logs "deep backfill:
  walking 730 calendar days"; within ~30 min the occvolume dir holds
  hundreds of day-files and backfill_done.json exists; archive stats
  totalBytes grows ~+500MB and stabilizes; subsequent boots skip via
  the marker.

## 2026-07-07 — [PIPELINE] European macro cluster — ECB + Eurostat + Bundesbank regime feed (census build #7) — v1.0.175

- TERRITORY: T-DATACORE. Census section-2 items 2-4 built as ONE keyless
  stream per the workup's recommendation (fredMacro pattern; cadences
  span 08:00 CET €STR → Tuesday-PM ILM → monthly Eurostat, which only a
  daemon poll records vintage-honestly). All three contracts
  LIVE-VERIFIED by the re-run workup agent (the first run was killed by
  an accidental stop; Mike confirmed unintentional and it was re-run —
  its one surviving finding, the ECB license doc, was corrected by the
  re-run: the live document is "Policy regarding the reuse of ESCB
  statistics", attribution string "Source: ECB statistics.").
- BUILD: server/euMacro.ts — 5 curated series (EUR/USD, €STR, weekly
  Eurosystem balance-sheet total, EA20 industrial production, 10y
  Bund), fredMacro vintage mechanism verbatim (latest-value per (s,d);
  revisions AND reverts append), per-series attribution (all three
  licenses verified commercial-OK w/ attribution). Route
  /api/data/eu-macro (cache-only).
- SOURCE QUIRKS encoded from the workup: ECB column positions vary per
  dataflow → header-name parse only; ECB 4xx bodies can be HTML → never
  parse error bodies; ILM ISO-week periods → normalized to the
  statement's Friday reference date (raw week kept in wk; isoWeekFriday
  tested incl. the 2020-W53 spillover — a wrong test assertion using
  nonexistent 2021-W53 was itself caught and fixed); Eurostat value{}
  is SPARSE + wrong-dimension = SILENT 200 with empty value{} → treated
  as fetch failure, alarmed, never cached as truth; Bundesbank values
  are STRINGS with null placeholder rows → parseFloat + skip, never
  zeroed.
- LIVE END-TO-END from session pre-commit: all five series match the
  workup exactly — EUR/USD 1.1415 @ 2026-07-06, €STR 2.183 @ 07-03,
  balance sheet 6,117,260 EUR-millions @ W26→2026-06-26, EA20 indprod
  98.3 @ 2026-04, Bund 2.98 @ 07-06.
- ALSO CONFIRMED THIS BATCH: the v1.0.174 OCC deep backfill is RUNNING
  on prod — occvolume 3 → 61+ day-files and climbing during this
  build's gates; the poll continues toward the ~500-file target.
- GATES: node 362/362 (7 new); tsc 64 baseline; manifests envelope
  green (46 manifests). Version 1.0.174 → 1.0.175.
- VERIFY (pre-stated): post-deploy /api/data/eu-macro serves all 5
  series with non-null latest (values matching the live probe or
  newer); /api/data/streams count=46 with eumacro live after first
  poll; next Tuesday's ILM print (~2026-07-08 PM CET) appends a new
  W27 point within one 6h cycle.

## 2026-07-07 — [PIPELINE] GEM asset registry ingest, part 1 — census #5 unblocked by human delivery (v1.0.176)

- UNLOCK: Mike completed GEM's download form (wishlist 9b) and uploaded
  three release files; he says MORE files are in his Google Drive — the
  Drive connector is installed for the org but toggled OFF for this
  chat (enabledInChat:false), so the remainder waits on him enabling it.
  This PR preserves + processes the delivered three (uploads are
  container-ephemeral — processed before anything else).
- INGEST (scripts/gem_ingest.py, session-run per release; license
  verified verbatim CC BY 4.0 from the files' own Copyright sheets):
  (1) Coal Mine Boundaries & Methane Sources v1.0.2 → coal_mines.json
  (250-mine registry: owners/parent/grade + bbox + centroid labeled
  "bbox midpoint — not a surveyed point") + coal_mine_features
  .geojson.gz (full 2,116-feature master — future map layer +
  satellite-join geometry); (2) GGIT Gas Pipelines Nov 2025 →
  gas_pipelines.json (4,246 segments, all 53 fields; THIS VARIANT HAS
  NO ROUTE COORDINATES — honesty note in provenance; RouteAccuracy/
  RouteType reference GEM's separate geometry product); (3) Gas
  Finance Tracker Dec 2025 → gas_finance.json (243 LNG-terminal + 531
  gas-plant finance rows).
- PARSE HONESTY: xlsx max_row inflation diagnosed before trusting
  counts (LNG sheet reports 973 rows; 729 are trailing formatted-empty
  — verified by full scan; 243 real data rows); nulls dropped per row
  never zero-filled; datetimes ISO-ified; zero-rows refusal on every
  artifact.
- HYPOTHESES filed in the manifest (gate-locked): mine boundaries x
  FIRMS/S1/S2 activity proxies; Parent Company / ParentEntityIDs =
  equity join spine into the entity graph; pipeline FID/status
  capex-cycle features; financier networks.
- GATES: pytest 460 passed 1 skipped (4 new); node 362/362 (envelope
  green, 47 manifests). Version 1.0.175 → 1.0.176.
- VERIFY (pre-stated): artifacts in git with coherent spot-checks
  (Appin M0005 centroid in NSW; Double E operating; counts pinned in
  tests); /api/data/streams count=47 with gem manifested after deploy;
  next GEM release re-runs the script and diffs cleanly.

## 2026-07-07 — [REPAIR] R15: powergrid toggle permanently "reload to enable" — wired-list omission + double ratchet — v1.0.177

- REPORTED BY MIKE: the Power grid (TX pilot) toggle stuck in "reload
  to enable" even in fresh incognito. ROOT CAUSE (one grep): datamap's
  unwired detection (`!(l.id in LAYER_GROUP)`) treats LAYER_GROUP as
  the client-wired declaration — v1.0.166 shipped powergrid's registry
  entry AND render wiring but no LAYER_GROUP entry, so the CURRENT
  bundle declared its own layer unknown. "Reload to enable" that no
  reload can fix — the exact class Mike named. Registry-vs-map diff
  confirmed powergrid was the only affected id.
- WHY THE HARNESS MISSED IT (double gap): the toggle-consistency
  battery iterates the harness's hardcoded layers FIXTURE, which also
  lacked powergrid — the battery never clicked the one toggle that was
  broken. The v1.0.166 visual run was honestly green over an
  incomplete universe (a silent-cap lesson applied to test fixtures).
- FIX + RATCHETS: (1) powergrid → LAYER_GROUP (facilities);
  (2) RATCHET A, server/layersWiring.test.ts: every non-signal/
  non-planned id in datacore/layers.json MUST appear in datamap's
  LAYER_GROUP — a registry entry without the wired declaration now
  fails CI, making the permanent-unenableable state unrepresentable;
  (3) RATCHET B: powergrid added to the harness layers fixture with a
  comment requiring every toggleable registry layer to appear there —
  the battery now clicks it every run (and did, green, exercising the
  pmtiles source path).
- GATES: node 363/363 (1 new ratchet test); tsc 64; build OK; visual
  harness data-page PASS 390/768/1440 with powergrid exercised.
  Version 1.0.176 → 1.0.177.
- VERIFY (pre-stated): post-deploy, a fresh /data load shows the
  "Power grid (TX pilot)" toggle ENABLED (no reload note); toggling it
  on renders the TX voltage-classed lines; Mike confirms from his
  phone (the original report path).
## 2026-07-07 — [REPAIR] `--accent` CSS collision fixed sitewide — T-CLIENT (v1.0.178)

- TERRITORY: T-CLIENT (client/src/index.css, tailwind.config.ts).
- REPAIR MANDATE: session start checked open_questions.md KNOWN BROKEN
  first per the repair mandate. Item #13 (found 2026-07-04, v1.0.160)
  was still unrepaired: `:root` declares `--accent` twice — the
  DESIGN.md-documented brand hex (`#4d9fff`, line 25) and, further down
  the same block, a bare shadcn/Tailwind HSL triple (`212 100% 65%`,
  line 92) — the later declaration wins the cascade, so every direct
  `var(--accent)` consumer (expects a color) silently got an invalid
  triple instead.
- PRIOR STATED BEFORE INVESTIGATING THE FIX: expected the correct fix
  to be "rename the shadcn token, since only 2 wrapper sites in
  index.css consume it" (the shape #13 itself proposed). Investigation
  proved that prior wrong before touching anything: `tailwind.config.ts`
  (`accent: { DEFAULT: "hsl(var(--accent))" }`) is a THIRD consumer,
  feeding real Tailwind `bg-accent`/`text-accent-foreground` utility
  classes used across 10 shadcn components (select, dropdown-menu,
  context-menu, menubar, dialog, calendar, toggle, command,
  navigation-menu, sidebar) — dropdown/menu hover and selected-item
  states. Renaming the shadcn token without also touching
  tailwind.config.ts would have broken those live, currently-working
  interactive states across the whole UI component library — a much
  larger regression than the bug being fixed. DOWNSTREAM CHAIN traced
  before editing: rename target -> which token feeds Tailwind's color
  system -> which utility classes compile from it -> which real,
  currently-functioning components consume those classes. Because the
  brand hex is the one DESIGN.md documents as `--accent` and the
  shadcn token is internal scaffolding never referenced outside
  index.css/tailwind.config.ts, the correct minimal fix renames the
  shadcn side, not the documented brand token — same conclusion as
  #13's plan, reached for the right reason this time (tailwind.config.ts
  is in scope, not just the 2 index.css wrapper sites).
- FIX: renamed the shadcn HSL-triple declaration (both the `:root` copy
  and the `.light, .dark` duplicate) from `--accent` to
  `--shadcn-accent`; updated its two consumers — `--accent-border: hsl(var(--accent))`
  (2 sites, index.css:119/166) and `tailwind.config.ts:40`'s
  `accent.DEFAULT`. `--accent-foreground` was untouched (never
  collided). `--accent` in `:root` now has exactly one declaration
  (the hex) — verified via `grep -n "^\s*--accent:" client/src/index.css`.
- SCOPE CORRECTION vs #13's catalogue: the audit that filed #13 only
  grepped `client/src/index.css` for `var(--accent)` (18 sites). This
  session found 2 more real, currently-broken sites outside that file
  by grepping the whole repo: `client/src/pages/filings.tsx:32`
  (`option_exercise` badge color) and `client/src/pages/analyze.tsx:1392,1477`
  (a range-fill bar background and a border-left accent stripe) — both
  were rendering transparent/invisible for the same reason. All fixed
  by the same single-token-rename fix, no separate change needed.
- DORMANT COLLISION LEFT ALONE (per #13, re-confirmed): `--border` has
  the identical hex-vs-HSL-triple collision (index.css:20 vs :98) but
  zero direct `var(--border)` call sites exist repo-wide — this PR
  does not touch it, and re-grepped after the fix to confirm it's still
  dormant (no new sites appeared). Left for a future session/audit
  rather than bundled in, per the one-logical-change rule.
- VERIFIED LIVE (not just grep): `npm install` (node_modules was
  missing in this container), `npm run build`, then `npm run dev` +
  Playwright against localhost:5000. `getComputedStyle` confirmed
  `--accent` = `#4d9fff`, `--shadcn-accent` = `212 100% 65%`,
  `--accent-border` = `hsl(212 100% 65%)` (a valid color, previously
  would have been `hsl(212 100% 65%)` too by coincidence since the
  triple already won there — the bug was invisible on that one
  property and only visible on the 20 raw `var(--accent)` sites).
  Screenshotted `/app#data`: the `.vt-switch.on` layer toggles (imagery,
  aircraft, trains) now render a visible blue thumb/track instead of
  transparent. Screenshotted the filings full view
  (`/app#/data/filings`): the `EXERCISE` (option_exercise) badge now
  renders the correct blue instead of invisible text.
- GATES: `npm install` + `npm run build` succeeded; `npm run visual`
  (all 3 canonical widths + streams/developers/landing/scale batteries)
  — 0 hard failures, only pre-existing touch-target/clipped-control
  warnings unrelated to this change; `python3 -m pytest -q` — 457
  passed, 2 skipped (installed pytest + requirements.txt in this fresh
  container first; baseline unaffected by this change, which touches
  no Python). No new test added: this is a CSS/Tailwind-config token
  rename with no new runtime behavior to unit-test, and the existing
  visual harness (screenshots + hard-failure assertions) is the
  regression net for this class of change per CLAUDE.md's VISUAL
  VERIFICATION rule. Version 1.0.176 -> 1.0.177 was claimed by a
  concurrent PR (#312, powergrid toggle fix) that merged to main first;
  rebased onto it and re-incremented per the read-and-increment
  convention: 1.0.177 -> 1.0.178.
- STILL OPEN (unchanged from #13): the dormant `--border` collision,
  and re-screenshotting the map's other sub-views (earnings, shortvol,
  graph, powerplants) for any further undiscovered `var(--accent)`
  sites the whole-repo grep in this session didn't need to reach
  (filings.tsx/analyze.tsx were caught because the grep was repo-wide,
  not per-view) — no further sites found this session via
  `grep -rn "var(--accent)"` across `client/src` (20 total: 18 in
  index.css + filings.tsx + analyze.tsx, all now fixed).

## 2026-07-07 — [PIPELINE] GEM suite ingest part 2 — full tracker catalog via Drive (v1.0.179; was drafted v1.0.178 — #313 from a concurrent session took that version first, re-incremented per merge-order protocol)

- DELIVERY: gem-data.zip (224MB, 34 members) pulled from Mike's Drive
  after he enabled the connector + link-sharing (the connector's
  inline-base64 download can't carry 224MB — direct-download URL used).
  A structure-probe subagent mapped every sheet contract first (header-
  row exceptions, Excel-truncated + trailing-space sheet names,
  combined-Coordinates columns, float year headers, GEOT's
  inconsistent rollup headers) — the ingester encodes all of them.
- KEY VERIFICATIONS from the probe: GIPT (182,428 power units) is the
  EXACT union of the 8 standalone power trackers (per-type row counts
  match to the row) → standalones skipped, GIPT ingested as a STATED
  12-field projection (4.2MB gz; full 52-col detail in the raw zip;
  KNOWN DROP: GCPT-only CO2 columns). Portal Energetico = LatAm
  regional duplicate → skipped. GEOT rollup sheets = recomputable
  transitive closure → skipped (ingredients kept: 26,250 entity nodes
  + 24,351 entity edges + 49,391 asset edges, 3.8MB gz).
- THE JOINS THIS UNLOCKS (hypotheses in manifest, gate-locked):
  GEOT entities carry an SEC-CIK/LEI/PermID crosswalk — the DIRECT
  bridge between our EDGAR pipeline and every GEM asset; LNG carriers
  key on IMO → joins our AIS archive; GMET ships 3,474 DATED satellite
  methane plumes (events, joinable to mines/fields); coal-finance =
  6,556 financier->unit edges.
- ARTIFACTS: 18 files, ~36MB (gz over 4MB plain; nulls dropped; every
  skip decision recorded in suite_skips.json — stated, never silent).
  GIS route zips (68-72MB geojsons) NOT ingested — filed as the
  pipelines PMTiles map-layer build.
- GATES: pytest 461 passed 1 skipped (suite coherence battery incl.
  full 8-type census of power_units, CIK-survival check, header-row-2
  contract); node 363/363. Version 1.0.178 → 1.0.179.
- VERIFY (pre-stated): artifacts load + counts pinned in tests; next
  GEM release re-runs both scripts; the entity-graph CIK join and the
  pipelines map layer are the queued follow-ups.

## 2026-07-07 — [RESEARCH] GRID VISION program installed + strategic update "DATA IS THE MOAT, EXPERIENCE IS THE DOOR" (Amendment 5) — docs PR, no version bump

- Territory: T-GRIDVISION declared (new — research/grid_vision*.md,
  future grid-detection modules/scripts, the /data powergrid surfaces
  it extends); this PR touches docs + SHARED research/* only.
- HUMAN DIRECTIVES (both received this session, installed same day —
  human instruction = approval per HUMAN SOVEREIGNTY):
  1. GRID VISION — ML-assisted complete US power-grid mapping ("OSM
     as the base, ML imagery detection as the gap-filler and
     verifier, expanding to full national coverage state by state").
     Charter filed at research/grid_vision.md (phases A research-first
     / A2 products-define-the-spec / B verify→extend→discover with
     provenance tags / C state-by-state rollout with honest coverage /
     D premium visualization / E ratchets; paid boundary →
     BLOCKED-FOR-MIKE purchase orders). Wishlist carries the pointer
     block; charter's RESUME STATE is the authoritative handoff.
  2. STRATEGIC UPDATE — appended to VISION.md as a dated
     provenance-bannered section (verbatim quotes preserved); CLAUDE.md
     Amendment 5 = GOAL self-proposed-work weighting (archives >
     validated signals > licensed data products > premium experience >
     general SaaS) + PREMIUM EXPERIENCE STANDARD standing behavior
     (design system, perceived performance, visible
     freshness/provenance/confidence, polish passes as real work,
     390px always, "would a paying data customer screenshot this and
     trust it?"). Correctness > polish stated explicitly.
- PHASE A LAUNCHED before the docs work (subagent mandate): four
  parallel research agents in flight — GV-A1 (Apr-2025 T&F
  detection-pipeline paper + code adaptability), GV-A2 (labeled
  datasets: Duke T&D imagery, PLAD, TTPLA, OSM-as-weak-labels),
  GV-A3 (methods: RetinaNet tower+routing, HOT/OSM human-in-loop,
  SAR verification, shadow-height), GV-A4 (imagery inventory
  Esri/NAIP/S2 incl. bulk-license reality + CPU-vs-GPU compute
  assessment w/ possible purchase order). Reports → assemble
  research/grid_vision_research.md next; NOTHING BUILDS until it and
  the A2 products plan file.
- No version bump: docs/charter only — zero runtime behavior change,
  code_version attribution unaffected (checkpoint-entry precedent).
- VERIFY (pre-stated): the four agent reports land and are filed
  verbatim-quoted with VERIFIED/REPORTED labeling; grid_vision.md
  RESUME STATE updated; Phase B spec cites the products plan, not the
  reverse.

## 2026-07-07 — [RESEARCH] GRID VISION Phase A + A2 FILED — four-agent research doc + products plan; RunPod purchase order (docs PR, no version bump)

- Territory: T-GRIDVISION (research/grid_vision*.md) + SHARED
  (research/wishlist.md, this file).
- PHASE A COMPLETE same day it launched: all four parallel subagents
  returned; reports filed near-verbatim with VERIFIED/REPORTED
  labeling in research/grid_vision_research.md (Items 1-4 +
  cross-cutting summary). Pre-stated verify criterion from the
  install entry — met.
- DECISIVE FINDINGS (full detail in the research doc):
  (1) The Apr-2025 T&F pipeline paper = ADAPTABLE-WITH-WORK: MIT
  code, honest recipe, but NO released weights and NO training
  annotations (despite its own Data Availability statement — checked
  branch-by-branch); retraining is mandatory. Tower AP50 ~73% @0.3m
  is the bar. (2) Esri World Imagery ML/bulk use is FORBIDDEN by
  three quoted contract clauses (E204 §3.2(c)/§3.3(h)/§3.3(b), E300
  fn.96) — hard wall, routed around: NAIP public domain via free
  Planetary Computer STAC streaming (verified live, 2010-2023).
  Display basemap + identify capture-date (our shipped v1.0.173 use)
  stays legitimate. (3) Honest detection scope: transmission towers
  + substations; distribution poles are sub-detectable at NAIP GSD.
  (4) Eval = two-layer benchmark: Duke-US CC-BY ground truth
  (verified downloadable) + OSM-corridor recall with human-sampled
  precision — OSM is recall-only evidence, never "accuracy". (5) The
  graph/topology step is every published method's weak link (F1
  ~0.63); OSM-as-base sidesteps it. (6) WB/DevSeed hybrid is the
  proven operating pattern (33x speedup; their mosaic-boundary
  artifact -> our per-mosaic-source evaluation ratchet). (7) SAR @10m
  = verification-only gate-1 experiment; shadow+Sundial = height->
  voltage-class estimator (time-of-day recoverable from shadow
  azimuth). (8) Compute is cheap: TX corridor-verify ~7h CPU;
  training $50-100; national GPU re-scans $100-400.
- PHASE A2 FILED (research/grid_vision_products.md): provenance tags
  double as IP classes (ODbL vs ours vs produced-works); indices/
  alerts are the primary commercial surface; build order puts NO ML
  on the critical path to the first product (grid-stress index v0 =
  OSM TX + EIA-930 + weather, all already archived). Priors stated
  per REASONING STANDARD #10 before any gate runs.
- BLOCKED-FOR-MIKE filed: RunPod $50 deposit, RUNPOD_API_KEY in
  Railway (training needs GPU regardless; sweeps later). CPU path
  continues meanwhile.
- Ops note: prod /api/health timed out twice during the #315 Railway
  redeploy cutover, then returned 200/ok with bot active — transient
  deploy window, no liveness alarm.
- VERIFY (pre-stated for the next GRID VISION session): (a) A1
  grid-stress region-attribution build cites the products plan
  sections it implements; (b) gate-1 = mapped ERCOT corridor
  capacity vs published ratings, criteria stated BEFORE the join
  runs; (c) any detector metric reported per-mosaic-source and
  against the two-layer benchmark, never pooled-only.

## 2026-07-07 — [PIPELINE] GRID VISION A1 gate-1 PASS — TX grid capacity registry from OSM extract vs published ERCOT anchor (v1.0.180)

- Territory: T-GRIDVISION (scripts/grid_capacity_tx.py,
  test_grid_capacity.py, datacore/gridvision/) + SHARED
  (datacore/manifests/gridvision.json, package.json, research/*).
  Products-plan build-order item 1 (research/grid_vision_products.md
  A1) — the grid-stress signal's DATA-layer gate.
- METHOD: fresh Geofabrik texas-latest (709MB, downloaded this
  session) -> osmium power filter (5.2MB) -> 111MB GeoJSONSeq ->
  scripts/grid_capacity_tx.py (2.5s, dependency-free): line-km +
  circuit-km per voltage class, honest circuit convention stated in
  the artifact (multi-voltage way = 1 circuit/voltage; circuits=N
  multiplies; untagged -> 'unknown', never guessed).
- PRIOR (stated in the script header BEFORE the comparison ran):
  expected >=69kV circuit-km at 60-110% of the ERCOT 88,514 km
  anchor, 345kV closest to complete; PASS bands (a) [44250,115000]
  km, (b) unknown share <35%, (c) 345kV+ >=8000 km.
- RESULT: PASS all three — 104,928 circuit-km >=69kV (118.5% of
  anchor), unknown share 3.84%, 345kV+ 31,559 km. OBSERVED vs PRIOR:
  landed ABOVE my 60-110% band — the prior underestimated OSM TX
  completeness; consistent with the whole-state extract exceeding
  the ERCOT-region-only anchor (El Paso/WECC + Panhandle-SPP +
  East-TX-MISO included). Voltage census cross-validates structure:
  138kV 41,780 km (ERCOT subtransmission) > 345kV 27,164 km
  (backbone) > 69kV 20,919 km; 500/230/115kV small = non-ERCOT
  edges. 12,330 substations, 1,412 plants.
- HONESTY CAVEAT (in manifest, not buried): the ERCOT figure may be
  route-miles vs our circuit-km — the double-circuit share separates
  the two conventions; gate-2 refines against EIA/HIFLD per-line
  data before the stress index is trusted. Gate-1 is a plausibility
  band, and it holds under either reading.
- GATES: pytest 466 passed 1 skipped (5 new: tag conventions,
  circuit accounting, haversine, artifact coherence w/ recomputable
  gate-1 checks so the verdict cannot drift from data); node fail 0;
  tsc 64 baseline. Version 1.0.179 -> 1.0.180.
- VERIFY (pre-stated): streams inventory on prod shows the
  gridvision manifest (no-data volume side expected, session-run
  writer, gem precedent); next A1 step = region attribution
  (feature-level BA/ISO+county assignment) feeding the EIA-930 join,
  with gate-2 criteria stated before the index is computed.

## 2026-07-07 — [PIPELINE] GRID VISION A1 step 2 — TX county->BA crosswalk from EIA-861, EIA-930-joinable (v1.0.181)

- Territory: T-GRIDVISION (scripts/grid_county_ba.py,
  test_grid_county_ba.py, datacore/gridvision/tx_county_ba.json,
  research doc Item 5) + SHARED (gridvision manifest, package.json,
  research/*). Products-plan A1 continuation: the join key between
  the capacity registry (v1.0.180) and per-BA EIA-930 demand.
- SOURCE CHAIN (GV-A5 workup, filed as research doc Item 5, all
  public domain): EIA-861 2024 final — utility->BA lives in THREE
  files (Sales_Ult_Cust hdr row 3 + Short_Form hdr row 1 +
  Delivery_Companies hdr row 3 — the big ERCOT TDUs are ONLY in the
  third); county link via Service_Territory Counties_States; BA
  codes verified identical to EIA-930's (no crosswalk). HIFLD Open
  discontinued 2025-08-26 (archived 2022 layer is US-Gov-Work
  license-clean but stale); EIA Atlas BA polygons mid-migration
  (re-check ~late July — becomes the multi-BA arbiter when back).
- VERIFICATION BY CONVERGENCE: the workup agent computed the join
  independently; my build reproduced it EXACTLY (254/254 counties;
  ERCO 218 / SWPP 78 / MISO 36 / EPE 3 / PNM 2; 80 multi-BA) —
  after one investigated divergence: first run showed EPE=18 because
  an any-state union smeared Rio Grande EC's NM-row EPE across its
  18 TX counties whose TX filing says ERCO. Fix = STATE-PREFERRED
  lookup (TX-row codes win; any-state fallback only for utilities
  with no TX row — the Farmers-NM case). Second finding during test
  writing: my geographic intuitions were wrong twice (El Paso county
  IS multi-BA {EPE,ERCO} via Rio Grande; Harris carries MISO via
  Entergy Texas — verified in the join, real) — tests now pin
  data-verified facts, not prettier assumptions.
- HONESTY: artifact emits BA SETS + multi_ba flags; NO primary BA
  fabricated (861 cannot weight multi-BA counties); provenance
  states RETAIL service =/= grid topology on every surface.
- GATES: pytest 468 passed 1 skipped (+2 batteries); node 363 fail
  0; tsc 64 baseline. Version 1.0.180 -> 1.0.181. Also this session:
  v1.0.180 VERIFY confirmed on prod (streams inventory count 48,
  gridvision envelope live).
- VERIFY (pre-stated): next A1 step consumes this artifact for
  feature->county->BA capacity aggregates (point-in-polygon over the
  Census cartographic counties, Item 5.1); annual refresh ~October
  when f861-2025 finalizes.

## 2026-07-07 — [PIPELINE] GRID VISION A1 step 3 — per-county/per-BA capacity distribution, conservation-exact (v1.0.182)

- Territory: T-GRIDVISION (scripts/grid_ba_capacity.py,
  test_grid_ba_capacity.py, datacore/gridvision/tx_ba_capacity.json)
  + SHARED (gridvision manifest, requirements-dev.txt pyshp,
  package.json, research/*). Completes the geographic half of the
  grid-stress ingredient chain: capacity now sits in the same
  region space as EIA-930 demand.
- METHOD: segment-midpoint point-in-polygon (even-odd ray casting,
  holes/multipart handled) over Census cartographic county
  boundaries, with last-hit caching (lines are spatially contiguous
  -> most segments resolve without a ray cast; 31,220 ways in 16s).
  BA rollup via tx_county_ba.json: single-BA county km -> exclusive;
  multi-BA county km -> AMBIGUOUS pool keyed by BA set ("ERCO|SWPP"),
  never split by an invented ratio; out-of-state segments counted.
- PRE-STATED EXPECTATIONS (in the script header before first run) vs
  OBSERVED: (1) conservation within 0.5% -> EXACT (108,515.0 ==
  108,515.0; same input + same convention, and the test battery
  RECOMPUTES it from committed pieces so it cannot drift); (2)
  345kV+ ambiguous share 15-35% (CREZ Panhandle seam) -> 31.3%; (3)
  out-of-state <5% -> 2.7%. All three held.
- RESULT HIGHLIGHTS: ERCO-exclusive 345kV+ = 19,206.8 circuit-km
  (the ERCOT backbone attribution); SWPP-exclusive 1,472; MISO 167;
  ambiguous pools exactly the expected seams (EPE|ERCO, ERCO|MISO,
  ERCO|MISO|SWPP, ERCO|SWPP, MISO|SWPP, PNM|SWPP); all 254 counties
  carry capacity; zero unknown-county km (shapefile names resolve
  the crosswalk 1:1).
- GATES: pytest 471 passed 1 skipped (+3: geometry incl.
  hole/multipart, cache correctness, conservation recompute); node
  363 fail 0; tsc 64. Version 1.0.181 -> 1.0.182. pyshp declared in
  requirements-dev.txt (session-run only, imported inside the
  loader so CI never needs it).
- VERIFY (pre-stated): the gate-2 stress-index design consumes
  ba_exclusive + ambiguous pools with the ambiguity carried into the
  index's stated uncertainty, not resolved by assumption; arbiter =
  EIA Atlas BA polygons when their migration completes (~late July).

## 2026-07-07 — [RESEARCH] GRID VISION A1 gate-2 DESIGN filed — stress index v0 criteria pre-stated before computation (docs, no version bump)

- Territory: T-GRIDVISION (research/grid_vision_products.md appended
  design section, charter resume) + SHARED (this file).
- DESIGN (full text in the products plan): v0 index = per-BA demand
  percentile x forecast strain (realized vs EIA-930 day-ahead DF) x
  CPC degree-day extremity. THE HONESTY CONSTRAINT SHAPED IT:
  circuit-km is not MW, so v0 carries NO capacity denominator — the
  grid map spatializes exposure (which counties/corridors/facilities
  hang off a stressed BA) instead of pretending to a rating we don't
  have. Capacity-normalized version waits on a real per-line MW
  ratings join.
- GATE-2 CRITERIA (pre-stated, dated): operationalized stress hours
  (top-decile same-month demand + >=3% forecast exceedance),
  spot-validated against a hand-collected ERCOT conservation-appeal
  list (validation, never fitting); D-morning index predicts D+1;
  fit 2015-2022 / validate 2023-2025 (current-vintage caveat stated
  — live archive accumulates true vintages via rt); seasonal
  same-month base-rate control; PASS = >=1.5x out-of-sample lift
  stable across all three validation summers; FAIL = descriptive
  dashboard demotion, labeled non-predictive. Kill date 2026-08-15.
- PREREQS QUEUED (each own PR): gridDemand DF polling; env-gated
  EIA-930 historical backfill (server-side, OCC pattern — EIA key
  never leaves Railway); then the computation, results filed
  whatever they say.
- VERIFY (pre-stated): the gate-2 computation session must quote
  this design's criteria verbatim and may not adjust them after
  seeing data; weight-fitting confined to the training split.

## 2026-07-07 — [PIPELINE] GRID VISION gate-2 prereq 1 — EIA-930 day-ahead forecast (DF) rides the gridDemand poll (v1.0.183)

- Territory: T-DATACORE (server/gridDemand.ts + test) + SHARED
  (griddemand manifest, package.json, this file). First prereq from
  the filed gate-2 design (#320): forecast strain = realized D minus
  DF per hour.
- DESIGN (read-before-write findings encoded): the archive's dedup
  identity was respondent|period — a second series would COLLIDE
  with demand rows. Key extended to respondent|period|type with
  LEGACY COMPAT: pre-v2 archived lines carry no type field (the
  facet forced D) and hash as D, so the existing archive seeds
  correctly and no demand hour re-archives. DF rides the SAME API
  call via a second type facet; length doubled (two series x the
  same 48h window) — call count unchanged (9/cycle). Stats additive:
  latest_forecast_mwh beside latest_mwh; hours_in_window still
  counts demand rows only (meaning unchanged for consumers).
- TEST NOTES: legacy-compat test placed FIRST in the file
  deliberately — the seed pass fires once per process (comment in
  test); one fixture collision with the pre-existing battery via the
  process-global seen set found and avoided with distinct
  respondent/period. The url assertion updated for the INTENTIONAL
  length change (48->96) and strengthened (both facets pinned) — no
  assertion weakened.
- GATES: node 364 pass 0 fail (gridDemand battery 6/6 incl. new
  legacy-seed + DF-stats tests); pytest 471 passed 1 skipped; tsc 64
  baseline. Version 1.0.182 -> 1.0.183.
- VERIFY (pre-stated): post-deploy, /api/data/grid-demand stats grow
  latest_forecast_mwh (non-null once EIA publishes DF for the
  window; DF exists for BAs not for US48 total — if US48 stays null
  that is the source's shape, not a defect: check ERCO first);
  archived day-files start carrying type:"DF" rows alongside
  typeless legacy + type:"D" rows. Next prereq: env-gated EIA-930
  historical backfill (server-side, OCC pattern).

## 2026-07-07 — [PIPELINE] GRID VISION gate-2 prereq 2 — EIA-930 historical backfill mechanism (v1.0.184); RUNPOD_API_KEY landed

- Territory: T-DATACORE (server/gridDemand.ts + test) + SHARED
  (griddemand manifest, package.json, research/*).
- BACKFILL (OCC deep-backfill pattern): one-time 2019->now walk of
  D+DF for all 9 respondents — year-windowed asc pagination (5000/
  page, EIA v2 max; ~270 calls at 1.5s spacing ≈ 7 min), oldest-
  first, done-marker single pass, gz-at-end (nothing sits plain for
  the 3-day cycle). Env-gated GRID_DEMAND_BACKFILL=1, opt-in OFF per
  R8; Mike asked to flip it (he is responsive today — the OCC
  default-on precedent required explicit approval first).
- HEAP PROTECTION (new failure mode found in design): a full
  backfill creates ~1.2M dedup keys; the seed pass would reload them
  into the 512MB Node heap on every restart (R13 territory). Fixes:
  (a) seed window bounded to 120d (the live poll fetches 48h — huge
  margin); (b) completed backfill years pruned from the in-memory
  set as the pass walks. Re-run contract stated in the marker file
  (delete marker AND day-files together).
- HISTORY CORRECTION (honest amendment to the gate-2 design): this
  endpoint serves ~2019+, not 2015 as the design sketch assumed —
  split becomes fit 2019-2022 / validate 2023-2025 (same three
  validation summers; 2020 COVID anomaly sits in training, noted).
- RUNPOD_API_KEY LANDED (Mike, same session): wishlist purchase
  order marked RESOLVED; GPU training/sweeps unblocked. First
  consumer = Phase B training-data prep (ETDII + OSM-seeded chips +
  NAIP/MPC streaming) then the first fine-tune, next GRID VISION
  build arc.
- GATES: node 365 fail 0 (battery 7/7 — opt-in gate, oldest-first,
  pagination, marker single-shot, seed-window bound); pytest 471
  passed 1 skipped; tsc 64. Version 1.0.183 -> 1.0.184.
- VERIFY (pre-stated): once Mike sets GRID_DEMAND_BACKFILL=1 —
  backfill_done.json appears in the archive dir with rows_archived
  ~1.2M-scale and calls ~270-scale; day-files back to 2019-01-01
  gz'd; memory flat after the pass (no R13-style restarts); the
  gate-2 computation then has its history.

## 2026-07-07 — [REPAIR] grid-demand forecast readout showed partial future-hour aggregates — same-hour DF fix + surface caveat (v1.0.185)

- Territory: T-DATACORE (server/gridDemand.ts + test) + SHARED
  (server/routes.ts note line, package.json, this file).
- FOUND BY THE PRE-STATED VERIFY (v1.0.183 post-deploy): per-BA DF
  values sane (ERCO 65,948 D vs 65,250 DF — real +1.1% strain), but
  US48 showed DF=74,297 against D=544,125. SOURCE-PROBED before
  fixing (DEMO_KEY): DF is day-ahead, so its NEWEST rows sit in
  FUTURE hours where the US48 aggregate is only partially reported
  (T+2h = 74k -> back-fills to 550k+ as BAs file). Picking "newest
  DF" was the defect; premium-standard-wise, a misleading number
  beats no number for badness.
- FIX: latest_forecast_mwh = DF at the SAME hour as latest_mwh (null
  when unpublished — honest); route note states the semantics AND
  the US48 progressive-back-fill caveat. Test adds a future-hour
  partial DF row that must never be surfaced. ARCHIVE UNAFFECTED:
  all DF rows still archive (gate-2 wants the full series; the
  analysis aligns hours itself) — only the readout changed.
- Also observed in verify: CISO absent from one cycle's stats
  (transient single-call failure; poll retries 2h, archive dedups).
  WATCH: if CISO is persistently absent next session, investigate
  the dual-facet call against CISO specifically.
- GATES: node 365 fail 0 (battery 7/7); tsc 64. Version 1.0.184 ->
  1.0.185. (Measurement-adjacent but a READOUT-only change: archive
  and gate-2 inputs untouched; direction of bias removed, not
  introduced.)
- VERIFY (pre-stated): post-deploy US48 latest_forecast_mwh is
  either null or same-scale as latest_mwh; per-BA values unchanged
  in scale; CISO present within two cycles.

## 2026-07-07 — [RESEARCH] GRID VISION amendment installed (DATA STRATEGY & ACCURACY HONESTY) + two backlog roots filed (docs, no version bump)

- Territory: T-GRIDVISION (charter amendment) + SHARED
  (open_questions.md, this file).
- AMENDMENT (human directive, key language verbatim in the charter):
  (1) self-bootstrapping training data with a strict gate — only
  high-confidence OSM-corroborated detections enter the training
  set, never the model's own uncertain guesses; training-set size +
  composition logged per iteration; close the US-NAIP and
  substation gaps first; (2) human-resolved model-vs-OSM
  disagreements become labeled examples ("corrections are training
  data, not just fixes"); (3) per-state precision/recall published
  in coverage manifests AND map layer info — never a single
  national number; disclosed error rate is a product feature; (4)
  accuracy-gated promotion from 'ml-discovered' with PRE-STATED
  bars; each retraining round reports per-state accuracy movement.
  Phase B/C/D/E specs absorb these as requirements.
- BACKLOG ROOTS (open_questions.md, explicitly NOT active build):
  NOAA SWPC space weather (GIC outage-risk hypothesis; gate-1
  ground truth candidate = DOE OE-417 disturbance reports; pairs
  with the grid layer + A3 exposure screens) and ground-magnetometer
  per-line load sensing (physically real, needs deployed hardware —
  parked as the "if we ever do physical sensors" note).
- VERIFY (pre-stated): the Phase B training PR must cite amendment
  items 1-2 in its design (bootstrap gate + corrections pipeline);
  the first Phase C state PR must ship the per-state accuracy
  numbers in manifest + layer info per item 3, with the promotion
  bar stated BEFORE measurement per item 4.

## 2026-07-07 — [NO-ACTION] v1.0.185 post-deploy verify record (session close; docs only)

- /api/data/grid-demand VERIFIED live post-#324: count 9 (CISO
  returned — the watch item was a transient single-cycle miss, as
  hypothesized); same-hour semantics serving with the note. Per-BA
  strain ratios sane (ERCO 0.989, MISO 0.990, SWPP 1.087, CISO
  1.000). OBSERVED SOURCE SHAPES for the gate-2 session to know:
  (a) FPL/ISNE/NYIS/PJM show DF=null at the latest D hour — these
  BAs' past-hour DF rows are not in the trailing window; the readout
  correctly refuses to guess. Gate-2 works from the ARCHIVE (all DF
  rows land) and must record per-BA DF coverage before computing
  strain. (b) US48 same-hour DF remains a partial aggregate at the
  leading edge (0.626 x demand at T05, matching the direct source
  probe) — surfaced as-is WITH the printed caveat; the pre-stated
  "null or same-scale" criterion was imprecise, the honest state is
  same-hour + caveat, recorded here rather than papered over.
- Brief root-endpoint timeouts during the #324 cutover, recovered
  (200 in 0.5s on retry); /api/health stayed 200. No liveness alarm.
- SESSION CLOSE (claude/new-session-iu72vf, 2026-07-07): 11 PRs
  merged this session's GRID VISION arc (#315-#324 + the earlier
  #312/#314 fixes); versions 1.0.180-185; charter + 2 amendments +
  research doc + products plan installed; A1 chain complete with
  gate-1 PASS; gate-2 prereqs shipped (backfill awaits Mike's
  GRID_DEMAND_BACKFILL=1); RUNPOD_API_KEY active. NOT STARVED —
  high-value work queued: gate-2 computation (on flag), Phase B
  training-data arc (amendment-governed), B1 stress-index surface.

## 2026-07-07 — [PIPELINE] ENTSO-E EU load stream — wishlist 9c activated same-day on key landing (v1.0.186)

- Territory: T-DATACORE (server/euLoad.ts + test, euload manifest) +
  SHARED (server/routes.ts, package.json, research/*). Standing
  detect-and-activate directive: Mike set ENTSOE_API_KEY -> built and
  shipped the same hour.
- CONTRACT VERIFIED BEFORE CODE (docs pages 400/403'd — routed
  around): (a) live keyless probe of web-api.tp.entsoe.eu returned
  the Acknowledgement_MarketDocument with Reason 999 "Authentication
  failed." (the error shape, pinned in tests); (b) params + codes +
  all 8 EIC zone codes verified from the ecosystem-canonical
  entsoe-py mapping/client source (A65/A16, outBiddingZone_Domain,
  periodStart/End YYYYMMDDHHMM); (c) GL_MarketDocument structure
  (Period > timeInterval/resolution/Point(position, quantity))
  from its parser source.
- BUILD (gridDemand pattern): 8 country-level zones, 48h trailing
  window, 2h poll (8 spaced calls, limit is 400/min); regex-XML
  parse (edgarForm4 precedent); points stored AS PUBLISHED at
  zone-native resolution (PT15M Germany vs PT60M others — dedup key
  zone|ts|resolution; absent positions stay absent, never
  interpolated); acked zones absent from stats, never zero-filled;
  token accepted under BOTH names (ENTSOE_API_KEY as set,
  ENTSOE_TOKEN as the wishlist proposed) and never logged; seed
  window 120d (heap precedent).
- HYPOTHESIS (gate-locked, manifest): EU zonal load vs seasonal norm
  = demand side of European energy regime features; joins eumacro
  (census #7) for industrial-production nowcast residuals; the
  EIA-930 analog for any future GRID VISION Europe work.
- GATES: node 371 fail 0 (battery 6/6: key gate both names, url
  contract, ack detect, PT60M/PT15M position math incl. gap
  honesty, cross-resolution dedup, sweep stats); pytest 471 passed
  1 skipped; tsc 64. Version 1.0.185 -> 1.0.186.
- VERIFY (pre-stated): post-deploy /api/data/eu-load serves 8 zones
  with sane MW scales (FR daytime ~45-60GW, DE_LU ~50-70GW, PT15M
  resolution on DE_LU specifically) within 2 poll cycles; streams
  inventory count grows to 50 with the euload envelope. If a zone
  acks persistently, the ack text is in the logs by design.

## 2026-07-07 — [REPAIR] alpaca_feed SIP-transition logs corrupt subprocess stdout JSON — ML retrain failing 3/3 hourly cycles (v1.0.187, PR #327)

- Territory: T-BOT (alpaca_feed.py is a shared data-plane module; the
  break manifests through server/bot.ts's tier3Strategic retrain path
  and ml_model_v2.py) + SHARED (package.json, this file).
- SESSION-START HEALTH CHECK found this, not a queued item: /api/health
  ok, equityPeak correctly persisted (109432.59, drawdownPct 0.0,
  liveness.dark:false — the v1.0.35/v1.0.36 fixes both holding).
  /api/diag/ml and /api/diag/audit (token-gated, human-approved
  2026-07-04) showed model_age_hours climbing 27.2->27.3 with
  feedback_live_count still 0 (all 1326 feedback rows seeded — expected
  per KNOWN BROKEN #12's still-open reseed-check gate, not new). Pulled
  the last 200 audit entries: 3/3 TIER3-ML-ERROR entries (09:26, 10:26,
  10:55 — every hourly Tier-3 cycle in the window) with identical text:
  `ML retrain failed (code=? signal=none): Unexpected token 'F',
  "[FEED] SIP "... is not valid JSON`.
- ROOT CAUSE (READ BEFORE WRITE trace): `alpaca_feed.data_feed()`
  (v1.0.150's SIP-403 self-repair) prints `[FEED] SIP entitlement
  rejected...` / `...restored...` straight to STDOUT on any feed-state
  transition. `ml_retrain_safe.py` is invoked by bot.ts as
  `execPythonSerialized("python3 ml_retrain_safe.py")` — a FRESH
  subprocess every hour, so alpaca_feed's module-level `_state`
  (`probed_at: 0.0`) always re-probes on first use. Since SIP is
  currently 403-rejected in prod, this branch fires on literally every
  invocation, and the print lands inside the exact stdout blob bot.ts
  does `JSON.parse(trainOut.trim())` on — corrupting it every time.
  SECOND FINDING, same root cause, worse blast radius: tier3Strategic's
  manipulation_detect scan (bot.ts ~3987) shares the identical
  subprocess-JSON-stdout contract and calls into the same data_feed()
  path — its catch block only `console.error`s (never `audit()`s), so
  this was ALSO silently failing every hour with zero trail in the
  persisted audit log. Not touched this PR (would be a second logical
  change) — flagged in open_questions.md below.
- FIX: both prints -> `file=sys.stderr` (stderr is diagnostic-only by
  every caller's existing contract; Railway still captures it in logs).
  Zero behavior change to the feed-selection logic itself — pure I/O
  channel fix.
- RATCHET: `test_state_transition_logs_never_touch_stdout` in
  test_alpaca_feed.py captures stdout via `contextlib.redirect_stdout`
  across both the 403-downgrade and the restore transition, asserts
  empty. A/B-VERIFIED: stashed the fix, test failed with the exact
  production error string reproduced locally
  (`AssertionError: '[FEED] SIP entitlement rejected...' != ''`);
  restored the fix, test passes. This is not a hypothetical ratchet —
  it reproduces the live incident byte-for-byte.
- DOWNSTREAM CHAIN (REASONING STANDARD #1): fixed retrain -> model
  actually refreshes hourly again -> ML features/predictions stop
  silently drifting stale -> once live feedback accumulates beyond the
  current 0 (all 1326 rows are seeded/backtest, per KNOWN BROKEN #12)
  retrain quality directly affects scoring honesty (HONESTY METRIC).
  No trading logic, threshold, or scoring changed — PROMOTION RULE 3's
  backtest requirement does not apply (data-plane logging plumbing
  only, not a strategy/parameter change).
- GATES: `python3 -m pytest -q` 472 passed, 1 skipped (baseline 471 +
  1 new). No .ts/.js files touched — node/tsc gates and the visual
  harness don't apply (this session's environment also has no
  node_modules installed, unrelated to this fix's scope: zero JS/TS
  changed). Version 1.0.186 -> 1.0.187.
- VERIFY (pre-stated): post-deploy, /api/diag/audit should show
  TIER3-ML-ERROR entries stop recurring at the next hourly Tier-3 cycle
  (or, if SIP is still 403 by then, a genuine retrain success/failure
  with real Python error text instead of the stdout-corruption
  artifact); model_age_hours in /api/diag/ml should drop after a
  successful retrain instead of monotonically climbing.
- FOLLOW-UP FILED (open_questions.md): manipulation_detect's tier3
  scan failure path (bot.ts ~3998) logs to console.error only, never
  audit() — same visibility gap the diagnostics extended_checks fix
  (KNOWN BROKEN #5) closed for social_data/finnhub_data. A future
  session should route it through audit() the same way, its own PR
  since it's a distinct (if related) fix.
## 2026-07-07 — [PIPELINE] euload v1.0.186 verify: LIVE with 2 anomalies -> per-zone observability shipped (v1.0.188; drafted v1.0.187 — #327 from a concurrent session took that version first, re-incremented per merge-order protocol)

- Territory: T-DATACORE (server/euLoad.ts + test) + SHARED
  (server/routes.ts note line, package.json, this file).
- VERIFY RESULT (pre-stated criteria vs observed, ~15 min
  post-deploy): stream LIVE — 7 zones at plausible scales (FR 52.5GW
  in the predicted 45-60 band, ES 35.1, IT 41.1, PL 20.4, SE 14.3,
  BE 11.4; ~190 pts/zone in the 48h window); euload in the streams
  inventory. All zones publish PT15M (not PT60M-mostly as I sketched
  — consistent with the EU's 2025 15-minute MTU harmonization;
  stored as published, so no code impact).
- TWO ANOMALIES, surfaced not shelved:
  (1) DE_LU ABSENT — the one zone the verify specifically predicted.
  Its ack/error text existed only in prod logs (invisible from
  outside). (2) NL latest 2.4GW — physically wrong for Dutch total
  load (~10-15GW expected); could be TenneT under-reporting (a known
  TP data-quality pattern) or a partial leading-edge point — the
  route couldn't distinguish because it showed only the latest
  point.
- FIX (one logical change — route self-diagnosis, R7 precedent):
  per-zone `issues` map on the route (last sweep outcome: http
  status+ack text / ack reason / exception / empty-document) and
  window_min/max/mean_mw per zone (series shape exposes whether NL
  is low across the window or only at the edge). Both lessons pinned
  in tests. No archive/measurement change.
- GATES: node 371 fail 0; tsc 64. Version 1.0.187 -> 1.0.188 (post-rebase).
- VERIFY (pre-stated): next cycle post-deploy — if DE_LU still
  absent, `issues.DE_LU` names the reason from outside (then
  diagnose: suspect either an ack like 'No matching data' needing a
  different domain code, or a timeout on the largest document); NL's
  window mean/min/max classifies its 2.4GW as series-wide
  under-reporting (-> manifest data-quality note, keep archiving as
  published) vs leading-edge partial (-> same-hour-class fix like
  the US48 DF case).

## 2026-07-07 — [REPAIR] euload archive froze partial leading-edge values — value-aware vintage dedup + gz merge (v1.0.189)

- Territory: T-DATACORE (server/euLoad.ts + test, euload manifest) +
  SHARED (package.json, this file).
- FOUND BY THE v1.0.188 OBSERVABILITY FIELDS ON THEIR FIRST CYCLE
  (the ratchet paying for itself same-day): (1) DE_LU absence =
  TRANSIENT — back at sane 63.4GW; this cycle ES timed out instead
  and issues.ES named it from outside; the 48h window + dedup
  self-heal single-cycle misses, no action. (2) NL window
  min/mean/max = 2.4/8.2/11.9GW and IT latest 19.3 vs max 44.9 —
  DIAGNOSIS: partial leading-edge publication that revises upward
  within hours (TenneT/Terna pattern), NOT series-wide
  under-reporting.
- THE REAL DEFECT the diagnosis exposed: dedup key zone|ts|res
  meant the REVISED (correct) value could never re-archive — the
  archive would keep the first partial number forever. Silent-
  corruption class.
- FIX: value-aware event identity (zone|ts|res|mw) — revisions
  APPEND as vintage rows (fredMacro precedent; consumers take last
  per zone|ts|res), exact re-publications still dedup; plus
  gzipOldLoadDays now MERGES with an existing .gz instead of
  overwriting (a late revision after gz would have dropped the gz'd
  rows — second silent-loss edge found while fixing the first).
  Both pinned in tests. Route/stats unchanged (each sweep already
  serves current-best).
- GATES: node 372 fail 0 (battery 7/7 incl. revision-append +
  gz-merge); tsc 64. Version 1.0.188 -> 1.0.189.
- VERIFY (pre-stated): after a few cycles, NL's archived vintages
  for a given leading-edge ts show the upward revision sequence
  (2.x -> ~8-12GW), and window_mean stabilizes; day-file growth
  stays modest (revisions are a small multiple of the leading edge
  only).

## 2026-07-07 — [RESEARCH] GRID-STRESS GATE-2 COMPUTED: NOT PASSED — both outcome operationalizations voided on their own rules; index demoted to descriptive (v1.0.190)

- Territory: T-GRIDVISION (scripts/grid_stress_gate2.py,
  test_grid_stress_gate2.py, datacore/gridvision/gate2_result.json,
  products plan result section, charter resume) + SHARED
  (package.json, this file). Computed the same hour Mike flipped
  GRID_DEMAND_BACKFILL=1 (backfill itself verified running on prod,
  separate report below).
- DATA (session-side, keyless — no prod dependency): EIA-930 bulk
  six-month BALANCE files 2019-2026 (16 files, ~690MB, D + DF
  columns) + the committed CPC TX degree-day artifact. ERCO scope
  per the locked design.
- PROCESS DISCIPLINE HELD: criteria quoted verbatim from the filed
  design; operationalization v1 written in the script header BEFORE
  its run; v2 pre-stated after v1's void with its own header block;
  fitting confined to 2019-2022; thresholds frozen before
  validation; TWO variants total, discount logged.
- V1 (the design's forecast-exceedance stress definition): VOID by
  its own spot-validation rule — 0/10 documented ERCOT emergency
  days detected. Physical diagnosis VERIFIED in data before any
  revision: famous event days max strain +0.4%..+1.3% (heat waves
  are well-forecast) and Uri realized ran 29-32% BELOW forecast
  (shed load caps metered demand). The design's outcome variable
  measured demand surprise; Texas stress is supply-side scarcity
  under well-forecast demand.
- V2 (top-2% same-month pooled daily peaks): ALSO VOID — suspected
  code bug first, ruled out by direct inspection; the real cause is
  ~5-6%/yr ERCOT demand growth making pooled 2019-2022 percentiles
  blind to early-year emergencies (top-5 Aug training peaks all from
  2022; Uri's EEA day 4th behind the Feb-2022 cold snap; the demand
  extreme was Feb 14, a day before the declaration date).
- FOR THE RECORD (void, explicitly not claims): v1 lift 1.455
  (entirely 2024-carried; 2023/2025 hit 0), v2 lift 1.554
  (2023-carried 3.17; 2024 = 0.70) — BOTH independently fail the
  pre-stated no-single-summer-carry clause. The stability test
  caught regime-carry twice; it earned its keep.
- STOPPING RULE APPLIED: a third same-session outcome variant would
  be fitting the ruler to the event list (REASONING STANDARD #4 /
  anti-fishing). V3 = fresh design in a later session: complete
  public ERCOT conservation/EEA event list 2019-2025 from primary
  sources (subagent research) as ground truth + growth-aware
  extremes (same month x same year or detrended); criteria re-filed
  before computation; discount compounds at variant 3.
- CONSEQUENCE (the design's FAIL path, honored): the grid-stress
  index ships as a DESCRIPTIVE dashboard surface labeled
  non-predictive; NO predictive, tradable, or sellable claim. The
  A1 ingredient chain (capacity registry, county-BA join, per-BA
  distribution, D/DF archive) remains valid gate-1 infrastructure —
  the SIGNAL layer is what did not validate.
- GATES: pytest 474 passed 1 skipped (+3 battery: recomputable
  verdicts, both diagnoses pinned); node 372 fail 0; tsc 64.
  Version 1.0.189 -> 1.0.190.
- VERIFY (pre-stated): the v3 design session must cite this entry's
  stopping rule and produce its event list from primary sources
  BEFORE re-deriving any outcome; the descriptive surface must carry
  the non-predictive label per Amendment 5c.

## 2026-07-07 — [NO-ACTION] EIA-930 historical backfill VERIFIED COMPLETE on prod (v1.0.184 pre-stated criteria all met; docs only)

- Final state (13:07Z, ~40 min after Mike set GRID_DEMAND_BACKFILL=1):
  2,927 day-files spanning 2019-01-01 -> 2026-07-08 — the full walk.
  Archive bytes DROPPED 59MB (mid-pass plain) -> 17.7MB: the
  gz-at-end ran, which only happens at the end of a completed pass —
  done-marker confirmed by behavior. Compressed size landed inside
  the predicted 15-25MB band; peak transient stayed ~130MB against
  ~2.5GB headroom (volume check done BEFORE the pass per Mike's
  instruction; no durability risk at any point).
- Health through the pass: /api/health 200 in 0.24s and
  /api/data/grid-demand 200 in 0.68s immediately after completion —
  no R13-style memory/liveness trouble; the seed-window bound +
  per-year prune protections held (or were never stressed).
- The archive now holds ~7.5 years of hourly demand AND day-ahead
  forecast for all 9 respondents — permanent raw material for the
  gate-2 v3 design (which will use it with growth-aware extremes per
  the stopping-rule entry above) and any future demand work.

## 2026-07-07 — [REPAIR] PRIORITY-1: kill switch fired at market open on a garbage equity read — validated-drawdown guard at both sites (v1.0.192; drafted v1.0.191 — #332 from a concurrent session took that version first, re-incremented per merge-order protocol)

- Territory: T-BOT (server/bot.ts kill sites, server/drawdownGuard.ts
  + battery) + SHARED (test_audit_critical.py ratchet update,
  package.json, this file).
- INCIDENT (found when Mike asked "what happened to the bot"):
  /api/health showed bot status "killed" at 14:22Z (10:22 ET, market
  OPEN) with equityPeak $109,432.59 and drawdownPct 0.0 — killed
  with the account AT its peak. killSwitch is memory-only, so the
  kill fired inside the current container's 53-min uptime, i.e.
  9:29-10:22 ET. A real -10% round-trip inside an hour on this
  account is implausible.
- ROOT CAUSE (read, not guessed): both kill sites computed drawdown
  from a single unvalidated Alpaca /v2/account read. The Tier-1 site
  was `parseFloat(acct.equity || "100000")` — the || catches only
  empty/undefined, so a transient "0" sails through and computes as
  -100% drawdown -> kill; worse, an absent field FABRICATED $100k
  equity. The account-route site parsed with no validation at all.
  The persisted audit row (DRAWDOWN-KILL, owner-gated dashboard log)
  carries the triggering dollar figure for confirmation.
- FIX: server/drawdownGuard.ts evaluateDrawdown() — the ONLY
  filtered class is impossible reads (non-finite or <= 0): no kill,
  no peak update, audited as EQUITY-READ-INVALID at both sites.
  Every credible read kills exactly as before at <= -10% (pinned:
  exact-threshold, just-above, catastrophic-but-possible lows still
  kill). The $100k fabrication is gone. MECHANISM PRESERVED — this
  is input validation in T-BOT code, not a threshold or mechanism
  change (risk_kill_switch.py untouched).
- RATCHETS: TestKillSwitchPeakPersistence updated to track the
  guarded form — intent unchanged (same-line peak persistence; halt
  comparison verbatim, now pinned inside the guard) and STRENGTHENED
  (both sites must evaluate through the guard; invalid reads must be
  audited, never silent; the guard may filter nothing beyond the
  impossible class). New node battery drawdownGuard.test.ts pins
  both directions: garbage never kills / credible catastrophe always
  kills.
- RECOVERY: killSwitch is memory-only — this PR's deploy clears it
  and the loop resumes; the kill cancelled open orders only
  (positions untouched), so no state repair needed. Loop dark well
  under the 2-market-hour liveness alarm threshold.
- GATES: pytest 475 passed 1 skipped; node 375 fail 0; tsc 64.
  Version 1.0.191 -> 1.0.192 (post-cherry-pick onto #332's main).
- RECURRENCE EVIDENCE (found during the branch-reset recovery): the
  bot showed "killed" AGAIN after #332's fresh deploy — killSwitch is
  memory-only, so the garbage equity read is RE-FIRING on/after each
  boot, not a one-off transient. This elevates the fix from
  hardening to active outage repair; post-deploy the
  EQUITY-READ-INVALID audit rows will record each bad read as
  evidence for any upstream (Alpaca account API) follow-up.
- VERIFY (pre-stated): post-deploy /api/health bot status returns
  "active" with equityPeak intact ($109,432.59); any future garbage
  read appears as EQUITY-READ-INVALID in the audit log WITHOUT a
  kill; a recurrence of a peak-equity kill = RCA per the recurrence
  rule, not a re-patch.

## 2026-07-07 — [REPAIR] R14 follow-up: Alpaca account-data incoherence incident (docs-only)

Filed as the evidence record promised to the human. No code change; no
version bump (docs-only PR precedent #331).

TIMELINE (ET, 2026-07-07):
- ~09:29 boot → ~10:20 first DRAWDOWN-KILL: garbage-class equity read
  while the account sat at peak (pre-fix pathway; R14 root cause).
- 10:47 (#334 deploy verified): bot active, peak $109,432.59, dd 0.0.
- ~11:13 (15:13Z): bot killed AGAIN — this time on a CREDIBLE-class
  reading (~$27k: finite, positive). The guard is behaving exactly as
  pinned in drawdownGuard.test.ts (credible catastrophe must kill).
- 11:12 human's Alpaca dashboard screenshot, three-way
  self-contradictory: headline portfolio $27,124.81 (−74.92%, daily
  −$81,026.58) vs cash $5,583.78 + buying power $82,649.99 vs Alpaca's
  OWN 1D chart rendering ~$106.5k–$109.25k ending ≈$106.8k. Positions
  panel "No open positions" (Options filter active — not proof of an
  empty book).
- 15:21Z /api/health: alpaca ACTIVE, bot killed (latched, memory-only),
  equityPeak $109,432.59, drawdownPct-vs-last_equity 0.0 — Alpaca's own
  last_equity field sits AT peak, contradicting the −75% headline a
  third way.

ASSESSMENT: the kill mechanism and the new guard both worked as
designed. The incoherence is upstream — Alpaca's paper-account
snapshot disagrees with itself (headline vs chart vs last_equity). An
$81k intraday loss with no visible fills and last_equity at peak is
not a coherent account state. REASONING STANDARD #9 (believe live)
does not resolve this: "live" is self-contradictory, so NEITHER number
is trusted and the safe state is killed.

DECISION: bot stays KILLED until (a) the human's Alpaca Orders→Filled
check shows no morning fills (→ corruption confirmed; safe to resume)
or (b) Alpaca's numbers become self-consistent again. Reactivating on
the assumption of corruption is forbidden by priority order (a wrong
resume risks trading a genuinely damaged book).

TRADE-WINDOW BOUND (for the fills check): orders could only have been
placed ~09:29–~10:20 and ~10:47–~11:13 ET. Each kill cancelled open
orders (DELETE /v2/orders); neither liquidated positions
(VOLTRADE_LIQUIDATE_ON_KILL unset).

FOLLOW-UP QUESTION filed in open_questions.md: coherence-class guard
(cross-check equity against last_equity + positions + fills before
accepting a catastrophic reading). Per the recurrence rule this is an
RCA/design question, not a same-session patch.

## 2026-07-07 — [REPAIR] R15: session trade visibility — diag "orders" + "positions-detail" probes (v1.0.193)

TERRITORY: T-BOT (server/bot.ts diag route, server/diag.ts). Human-directed:
"you need to see the trades somehow so you can learn ask questions."

- PROBLEM: every fill/order/position-detail surface was owner-gated, and
  the human-approved diag whitelist (2026-07-04) was deliberately
  aggregate-only (counts + exposure, never symbols). Sessions could not
  trace actual trades — the learning loop's judgment half was blind to
  the very thing it judges. Today's incident made it concrete: the
  session could not tell WHICH of 4 positions collapsed −$81k overnight.
- AUTHORIZATION: direct human instruction (2026-07-07, quoted above)
  widens the whitelist to per-order and per-position detail. Recorded in
  diag.ts at the DIAG_PROBES declaration. Posture otherwise unchanged:
  token-gated (min 24 chars, timing-safe), READ-ONLY, sanitizer applied,
  no keys/user data/order placement. auth.ts (frozen) untouched.
- CHANGE: diag.ts gains orderRow()/positionRow() whitelist shapers
  (numerics null-not-NaN; client_order_id and unknown fields dropped);
  bot.ts diag switch gains case "orders" (Alpaca /v2/orders, status/
  limit<=200/after params) and case "positions-detail" (per-position
  incl. lastday_price vs current_price — the mark-move forensics
  readout). The original "positions" summary probe is NOT widened; a
  ratchet asserts it stays aggregate-only.
- RATCHETS: 4 new diag.test.ts tests — shaping whitelists, null-not-NaN,
  wiring (probes exist, shaped, sanitized, limit capped), summary-probe
  non-widening. DIAG_PROBES loop in the existing wiring test enforces
  the route cases automatically.
- GATES: node 388 pass 0 fail; tsc 64 (baseline); pytest at commit.
  Version 1.0.192 -> 1.0.193.
- FOLLOW-ON (same day, after deploy): use positions-detail to name the
  collapsed holding in the Alpaca incoherence incident (R14 follow-up,
  PR #335) and answer "what were the trades placed today" with actual
  order rows instead of window bounds.

## 2026-07-07 — [REPAIR] R14/R15 forensics: the −$81k is an Alpaca account-state discontinuity, NOT trading losses (probes v1.0.193, first use)

First production use of the R15 orders/positions-detail probes, same
hour they deployed. Findings, all from Alpaca's own order/position API
via the token-gated diag surface:

- CORRECTION of the PR #335 record: "no trades today" was WRONG. The
  audit log does not visibly record order placements (only Smart
  Execution boot banners) — the earlier inference from audit types was
  an inference from absence. Ground truth from /v2/orders: the bot
  FILLED 4 buys today (SMH 6@580.525 13:34Z; QQQ 17@709.99, KWEB
  95@25.64, VXUS 43@84.93 all ~14:10Z; ~$21.6k total) plus ~57
  canceled unfilled SMH limit orders (pre-market queue churn,
  stale-order sweeper working as designed). Follow-up: order
  placements should write an audit row (small T-BOT item, queued).
- THE DECISIVE FACT: ZERO sell orders exist since at least 2026-07-01.
  Yesterday (07-06) the bot bought QQQ 13@725.27 + VXUS 151@86.06
  (~$22.4k). Those shares are GONE today — current positions are
  exactly and only today's 4 buys (avg_entry prices match today's
  fills to the cent), all healthy (combined unrealized −$127).
  Positions cannot leave an account without sells, expirations, or
  transfers; none exist in the order history.
- CASH MATH: cash now $5,583.78 + today's buys $21.6k ⇒ cash at
  today's open ≈ $27.2k. Equity at yesterday's close (Alpaca's own
  last_equity) ≈ $109.4k. Between Monday 15:31Z (last fill) and
  Tuesday 13:29Z (boot), positions vanished and cash was re-set to
  ~$27.2k with NO orders. This is an Alpaca-side paper-account state
  discontinuity (manual dashboard reset, or their paper platform
  losing/replacing account state) — nothing in our system did it, and
  the frozen order-transmission paths could not have (any sell would
  appear in Alpaca's own order list).
- BOT BEHAVIOR THROUGH IT: correct at every step. It sized today's
  buys to the actual ~$27k account, the guard killed on the credible
  ~$27.1k readings vs the $109,432.59 persisted peak (both kills
  14:19Z/15:07Z, both post-fill), kills canceled open orders only.
  Memory-only killSwitch means each deploy boots active and re-kills
  on the next tier-1 read — by design, safe while the peak/account
  mismatch persists.
- DECISION FOR THE HUMAN (filed, not self-applied): if the account was
  deliberately reset, equityPeak must be re-based to the new account
  (delete voltrade_equity_peak.json state or set peak to current
  equity) and the bot resumes on the ~$27k base; if NOT deliberate,
  this is an Alpaca support case and the bot stays down. Re-basing the
  peak is an account-truth decision, not a session's.

## 2026-07-07 — [REPAIR] R16: kill-switch latch persisted across deploys (v1.0.194)

TERRITORY: T-BOT. Evidence: today's incident timeline — the latch was
memory-only, so every merge-triggered redeploy booted the bot ACTIVE;
in one such window it traded for ~50 minutes ($21.6k of buys) on an
account whose state was in dispute, before the tier-1 read re-killed
it. The R14 follow-up entry earlier today called memory-only
"by design, safe" — today's order forensics (R15 probes) SUPERSEDE that
judgment with evidence: the boot-to-first-tier-1-read window is a real
trading window, not a formality.

- CHANGE: killSwitch latch persisted exactly like equityPeak (KNOWN
  BROKEN #7 pattern): voltrade_kill_switch.json on the volume with
  /tmp fallback, restored on boot, saved (with reason) on all three
  transitions — both drawdown kill sites and the owner /api/bot/kill
  toggle. The owner toggle remains the ONLY clear path; a deploy can
  no longer un-kill the bot. MECHANISM PRESERVED: halt logic, guard,
  thresholds all untouched — this persists the latch's STATE only.
- RATCHET: test_audit_critical.py gains
  test_kill_switch_latch_is_persisted — persistence file + boot
  restore + same-adjacent save on every latch transition (2-line
  window), both directions of the owner toggle.
- GATES: pytest 476 passed 1 skipped (new test included); tsc 64
  baseline; node drawdownGuard+diag batteries pass. Version
  1.0.193 -> 1.0.194.
- OPERATIONAL NOTE: after this deploys, the bot will kill once more on
  the next tier-1 read (~$27k vs $109k peak) and then STAY killed
  through all future deploys until the human either toggles it off
  (dashboard kill-switch button) after re-basing the peak, or Alpaca's
  account state is resolved. This makes the promised "bot stays down"
  actually true.

## 2026-07-07 — [PRODUCT] W4: ANALYST CONSOLE query engine — /api/data/query (v1.0.195)

TERRITORY: T-DATACORE (queryEngine module) + routes.ts (SHARED,
smallest-possible edit, last commit before PR). First build of the
ANALYST CONSOLE program (research/console_charter.md, human directive
2026-07-07).

- WHAT: server/queryEngine.ts — cross-layer geo-temporal query: point
  + radius (cap 250km) + day window (cap = 7d raw retention) + layer
  set over the six archives that exist (aircraft/vessels/trains
  hourly position files; fires/alerts/gauges daily event files).
  Per-layer provenance labels, freshness = newest MATCHED timestamp
  (null when nothing matched), byDay with absent-days-absent,
  topEntities (cap 10), events (cap 50, newest first),
  rejected_layers surfaced, every cap stated in the envelope, honesty
  note on adaptive-thinning lower bounds. LRU cache (5 min TTL, 50
  entries, 0.05-deg coord rounding). GET /api/data/query, kind:"raw"
  (raw-vs-signals rule), Cache-Control 300s.
- HOW: built by a worktree-isolated subagent against a written spec;
  session performed the read-before-write review line-by-line before
  integration (partition rule: subagent output ships only after the
  session's own review). Layouts mirrored from the writers
  (datacoreArchive.ts, nasaFirms.ts, nwsAlerts.ts, usgsWater.ts), not
  guessed; positions stream via readline+gunzip (R4/R5 event-loop
  lesson applied); record-level day guard beyond file-name filtering;
  zone-only alerts excluded (geometry honesty, matches siteTimeline).
- TESTS: server/queryEngine.test.ts — 11 tests: radius/window
  inclusion-exclusion, absent-days-absent, topEntities ranking+cap,
  event ordering+cap, unknown-layer rejection surfaced, cap clamping
  visible in result.query, gz day files, cache TTL/reset via
  injectable nowMs, empty-archive zero-results-no-throw.
- GATES: node full battery green (includes the 11 new); tsc 64
  baseline; pytest 476 passed 1 skipped. Version 1.0.194 -> 1.0.195.
- FOLLOW-UP (charter RESUME STATE): concurrent-scan gate if the
  public endpoint sees real traffic; W2 satellites next (licensing
  check first), then W1 globe mode.

## 2026-07-07 — [REPAIR] R14 incident update: Alpaca equity reading recovered to ~peak; bot active again BY THE RULES (observation, docs-only)

Observed ~16:08-16:15Z, via /api/health + the R15 probes:
- No DRAWDOWN-KILL since 15:07Z across multiple active windows (~65+
  min) — after a day of kills firing within seconds-to-minutes of any
  active window. Tier-1 runs ~30s; the bot surviving 5+ min active
  means equity now reads >= ~90% of the $109,432.59 peak.
- Positions UNCHANGED (same 4 morning buys, ~$21.6k, upl ~ -$3 net).
  Therefore cash must now read ~ $87k for equity ~ $109k —
  consistent with Alpaca RECONCILING the vanished positions by
  crediting their value back as cash (unverifiable from the diag
  surface, which has no cash field; the human's dashboard check
  settles it).
- No new orders since 14:10Z (bot active but scans yield no
  candidates).
- STATE HONESTY: the earlier "bot stays killed" decision presumed the
  kill condition persisted. It cleared on its own (dd ~ 0), so the
  bot is ACTIVE — correct under the mechanism's rules (the halt is a
  drawdown halt, and there is no drawdown). The kill LATCH (v1.0.194)
  never engaged because no kill has fired since it deployed; it will
  latch the next real kill. If the human wants the bot down
  regardless, the dashboard kill switch is owner-only by design —
  the session cannot and will not force it down absent a kill
  condition.
- PENDING for closure: human confirms dashboard shows ~ $109k equity
  / ~ $87k cash (reconciliation theory) — then the incident closes as
  "Alpaca paper-account state corruption, ~5h duration, reconciled
  upstream; zero real trading losses; guard + latch + probes shipped
  as permanent hardening." A recurrence enters as a NEW incident
  under the recurrence rule.

## 2026-07-07 — [PIPELINE] W2 server half: CelesTrak satellite GP stream — /api/data/satellites (v1.0.196)

TERRITORY: T-DATACORE (satellites module + manifest) + routes.ts
(SHARED, minimal edit). ANALYST CONSOLE program, second build.

- LICENSING GATE FIRST (charter rule): verdict PROCEED with quotes,
  filed in wishlist.md. Courtesy limits are the binding terms and are
  ENFORCED IN CODE: 6h/group cadence (3x CelesTrak's 2h update
  cycle), 10s group stagger, non-200 recorded as an issue and never
  retried until the next sweep (their M2M rule).
- WHAT: keyless stream per the euLoad pattern — groups stations/
  starlink/gps-ops/geo (live index enumeration proved the charter's
  "active-geosynchronous" name wrong; geo substituted, pinned in a
  test). OMM JSON format, live-probed (ISS record quoted in the PR) —
  deliberately NOT TLE: CelesTrak's site-wide notice says 5-digit
  catalog numbers exhaust ~2026-07-12 and TLE cannot represent
  6-digit IDs; a TLE build would have broken within days. Archive =
  orbit HISTORY (dedup NORAD_CAT_ID|EPOCH; same-epoch re-fetch
  appends nothing, every epoch advance accumulates), day-file per UTC
  fetch day per group, gz after 3d merge-not-overwrite, 30d seed
  window (starlink ~10k keys/day bounds the heap). Route is
  cache-only (event-loop rule), warming_up honesty, per-group issues
  surfaced, attribution on every response.
- HOW: worktree-isolated subagent against a written spec; session
  read the module line-by-line before integration (partition rule).
- TESTS: 10 new in satellites.test.ts (dedup across polls, day-file
  append + gz-merge, latest-per-satellite, warming_up, seeded dedup
  on simulated restart, manifest envelope, unknown-group rejection,
  geo-substitution pin); manifests.test.ts passes unchanged
  (FORWARD-ENFORCEMENT dir-scrape covers the new archive dir).
- GATES: node 409 pass 0 fail; tsc 64 baseline; pytest 476 passed 1
  skipped. Version 1.0.195 -> 1.0.196.
- NEXT (charter RESUME STATE): W1 globe mode + client satellite layer
  (satellite.js propagation; field-projection param first — starlink
  payload is a few MB), then W3 time scrubber.

## 2026-07-07 — [PRODUCT] W1: 3D globe mode default on the /data map (v1.0.197)

TERRITORY: T-CLIENT (datamap.tsx, index.css, visual_check.mjs).
ANALYST CONSOLE program, third build — the first visible one.

- WHAT: MapLibre v5 NATIVE globe projection (zero new dependencies) is
  now the /data map default; a [data-vt-globe] toggle (stacked under
  the fullscreen control, same 44px family) flips globe/flat,
  persisted in localStorage (a lasting preference, unlike the
  per-session fullscreen flag — deviation from existing
  sessionStorage toggles, accepted deliberately). Projection is baked
  into the bootstrap style so the FIRST paint is already the
  preferred mode. Degradation: runtime without setProjection (or a
  throwing call) -> mercator + disabled toggle with the reason in its
  title — never broken, never silent. Zero-cost-when-off: with the
  flat pref no projection API work happens at all.
- EVIDENCE (harness, 0 hard failures, all pages x 390/768/1440):
  data 390 TTI 2061ms (<3000), median frame 33ms, p95 67ms; all-off
  zero-cost TTI 1023-1086ms with 0 disallowed API calls; 19 layers
  toggled clean IN GLOBE MODE; fields (temp/wind rasters) render on
  the sphere; hillshade verified by synthetic-DEM pixel test
  (meanDiff 24.05 globe vs 24.12 mercator). MOBILE DEFAULT = GLOBE on
  A/B evidence: 390px globe median 33/33/33ms vs flat 17/33/33ms,
  overlapping p95 — no perf case for a flat default.
- RATCHETS (harness STRENGTHENED, nothing weakened): new GLOBE MODE
  battery (default-is-globe assertion, globe->flat->globe round-trip,
  localStorage persistence, aria-pressed sync, both-mode screenshots
  at z1.3 every width); [data-vt-globe] added to BOTH occlusion
  hit-tests (self-see + fields-on).
- REVIEW: built by a worktree-isolated subagent; session re-ran the
  full harness on the integrated tree and reviewed the screenshots
  (390px curved-field globe shot vs axis-aligned flat shot, same
  camera — unmistakable; controls/attribution/nav all clear).
- GATES: harness 0 hard failures; node 409 pass; tsc 64 baseline;
  pytest 476 passed 1 skipped. Version 1.0.196 -> 1.0.197.

## 2026-07-07 — [REPAIR] R17: satellites stream unreachable from Railway — host fallback + real error causes (v1.0.198)

TERRITORY: T-DATACORE (satellites module only). Evidence: two
consecutive prod boots failed every group sweep with connection-level
errors (stations/gps-ops "fetch failed", starlink 60s abort) while the
identical fetch works off-cloud (the W2 build agent's live probe) and
every OTHER stream's egress works — celestrak.org specifically is
unreachable/throttled from Railway's ranges, consistent with
CelesTrak's documented firewalling of abusive datacenter IPs.

- CHANGE: (1) fetchFailureDetail() surfaces undici's buried
  error.cause (ENOTFOUND/ETIMEDOUT/TLS) so the issues map names the
  REAL failure, not "fetch failed"; (2) transport-level failures try
  the alternate official host (celestrak.com) ONCE — the M2M courtesy
  rule governs server RESPONSES, and a connection that never reached
  them returned none; an HTTP non-200 still stops the sweep
  immediately with no alternate-host attempt (pinned by test);
  (3) fetch timeout 60s -> 120s (starlink is multi-MB; the 60s abort
  in the evidence may be a slow-but-working pipe).
- RATCHETS: 2 new tests — fallback-once + non-200-never-falls-back;
  both-hosts-fail surfaces per-host causes verbatim.
- NEXT EVIDENCE POINT: this PR's deploy runs a fresh eager sweep. If
  .com works, done. If both hosts fail, the issues map now names the
  cause and the fallback decision (session-side fetch? mirror?) gets
  made on real data — filed as the follow-up path, not built blind.
- GATES: node battery green (+2), tsc 64 baseline, pytest 476 passed
  1 skipped. Version 1.0.197 -> 1.0.198.

## 2026-07-07 — [REPAIR] R17 verdict: CelesTrak firewalls Railway's IP range — evidence final, route-around options filed (docs-only)

R17's instrumentation (v1.0.198) named the cause on its first boot:
UND_ERR_CONNECT_TIMEOUT on BOTH celestrak.org and celestrak.com, every
group, two sweeps — TCP connect never completes. That is an IP-range
firewall dropping SYNs (not DNS: would be ENOTFOUND; not TLS; not a
rate-limit: no HTTP response ever). Off-cloud the same fetch works.
CONCLUSION: CelesTrak blocks Railway's datacenter egress wholesale.
The stream stays in its honest warming/issues state — no fabricated
data, the route says exactly why it is empty.

OPTIONS (filed in wishlist.md for the human where approval is needed;
no blind builds):
A. SESSION-RELAY INGEST (recommended; needs human approval because it
   adds a WRITE surface): sessions CAN reach CelesTrak (proven). Add a
   token-gated POST ingest route (its own INGEST token, NOT the
   read-only diag surface — that whitelist stays read-only) that
   validates payloads through the existing parseGp/archiveGp path.
   Session routines fetch 1-2x/day and relay. Orbit history
   accumulates at daily rather than 6h resolution — honest and
   labeled. Zero cost.
B. BROWSER-SIDE FETCH for the future client layer only: visitors'
   browsers fetch gp.php directly (their IPs, not Railway's). No
   archive accumulation, and pushes load to CelesTrak per-visitor —
   worse citizenship, display-only. Not recommended alone.
C. RAILWAY STATIC OUTBOUND IP: plan feature; still a datacenter IP,
   likely still blocked. Low odds, only worth trying if free on the
   current plan.
D. Third-party TLE mirrors (KeepTrack etc.): unofficial, staleness and
   licensing unverified — would need its own licensing gate first.

DECISION RULE: A ships when approved; the 6h server poller stays in
place at zero marginal cost (if Railway's range ever gets unblocked or
the egress changes, the stream self-heals and the relay becomes
redundant — staleness audit will catch that).

## 2026-07-07 — [PRODUCT] W6 server half: the LLM Analyst tool-loop — POST /api/analyst (v1.0.199)

TERRITORY: T-DATACORE (analyst module) + routes.ts (SHARED, minimal:
one import + one session-gated route). ANALYST CONSOLE centerpiece,
server half; the chat pane is the next client PR.

- WHAT: key-gated Anthropic Messages tool-use loop answering
  natural-language questions STRICTLY from our own data tools — 7
  tools, all cache/archive reads (query_window, satellites,
  nws_alerts, grid_stress, eu_load, site_timeline, map_command).
  Port-dwell and shadow-fleet tools deliberately EXCLUDED: their
  exported functions trigger multi-day archive scans per call — the
  exact event-loop defect R4/R5 repaired by moving to pollers.
- COST DISCIPLINE (human directive: cheapest model): ANALYST_MODEL
  default claude-haiku-4-5, 1024 output tokens/turn, ~1 cent per
  question expected. Budgets enforced in CODE and stated in every
  envelope: 8 tool calls + 4 LLM round-trips per question with
  honest force-close notes; ANALYST_DAILY_TOKENS (default 500k)
  day-keyed and PERSISTED to the volume so deploys never reset
  spend; budget_exhausted is a first-class honest state; 2
  concurrent questions max (429 beyond).
- HONESTY MACHINERY: system prompt forbids memory facts, requires
  per-figure tool citations, forbids predictions/trading advice,
  mandates reporting warming_up/awaiting_key/truncation; grid_stress
  carries predictive:false + gate-2-failed status VERBATIM into the
  model context AND its schema description; tool results slim()ed
  with every truncation stated first; empty model output yields an
  explicit no-answer, never an invention.
- SECURITY: session auth required (_checkSession — anonymous
  visitors cannot burn the token budget; evidence-based choice, only
  cheap capture POSTs are public); ANTHROPIC_API_KEY never logged,
  every outgoing envelope deep-scrubbed against it (transport errors
  can quote headers); transport errors carry status + API error text
  only. AWAITING_KEY honesty: activates on key detect.
- TESTS: 14 in analyst.test.ts — awaiting_key, happy path, tool-call
  cap force-close, round-trip cap, map_command validation (bad lat /
  unknown layer as is_error to the model, never a throw), daily
  budget exhaust + persistence across module reset + day rollover
  (injectable nowMs), slim() truncation notes, key-scrub even on
  hostile transport errors, 429 concurrency, question validation.
- GATES: node 425 pass 0 fail; tsc 64 baseline; pytest 476 passed 1
  skipped. Version 1.0.198 -> 1.0.199. Built by worktree subagent;
  session line-by-line review before integration (partition rule).

## 2026-07-07 — [REPAIR] R18: manipulation-scan catch block now audits failures — KNOWN BROKEN #14 (v1.0.200, PR #351)

TERRITORY: T-BOT (server/bot.ts, tier3Strategic only). Loop-health
check at session start: last 10 tagged entries were 6 REPAIR / 3
PRODUCT / 1 PIPELINE — under the 7-REPAIR thrash threshold, no
meta-problem to address. System health at session start:
/api/health all green (bot active, drawdownPct 0.0, no liveness
alarm, scanner ok, licensing ok) — no fire to fight.

- WHAT: open_questions.md KNOWN BROKEN #14 (filed same-day as PR
  #327's ML-retrain stdout-corruption fix, deliberately left
  unrepaired there per one-logical-change-per-PR): tier3Strategic's
  manipulation-detection scan catch block (`server/bot.ts`, ~line
  4077) only `console.error`'d on failure — the same stdout-
  corruption failure class the ML-retrain path already surfaces via
  `TIER3-ML-ERROR`, but with zero trail in the persisted audit log.
  A live scan failure was invisible to any session working outside
  the container (no owner-only audit route access). Same visibility
  gap KNOWN BROKEN #5 closed for social_data/finnhub_data via
  extended_checks.
- FIX: mirrored the ML-retrain catch block's audit-with-cause pattern
  exactly — prefer stderr (Python traceback), fall back to stdout
  (structured JSON error payloads), report exit code + kill signal.
  New audit action `TIER3-MANIP-ERROR`, kept distinct from
  `TIER3-ML-ERROR`. `console.error` retained (additive, not a
  replacement). No scoring/sizing/order logic touched — pure
  failure-visibility widening; REASONING STANDARD #1 downstream
  trace: the only second-order effect is a new sibling audit action
  appearing alongside `MANIPULATION`/`TIER3-ML-ERROR` if/when the
  scan errors — future audit-log consumers should expect it.
- RATCHET: new server/tier3ManipVisibility.test.ts (3 tests, static
  source-assertion pattern matching scannerHealth.test.ts/
  diag.test.ts) — pins the catch block calls
  `audit("TIER3-MANIP-ERROR", ...)`, carries stderr/stdout/code/
  signal cause rather than a bare error string, and stays a distinct
  action from TIER3-ML-ERROR (regression guard against a future edit
  merging the two failure classes).
- GATES: node test:node 428/428 pass (425 baseline + 3 new); tsc 64
  pre-existing errors unchanged (re-verified against documented
  baseline after a fresh `npm ci`); python3 -m pytest 473 passed / 2
  skipped (env had drifted — numpy/pandas/requests/openpyxl were
  missing from this container and reinstalled from requirements.txt
  to get a real baseline; no Python file touched by this change, so
  this count is informational, not caused by the PR). Version
  1.0.199 -> 1.0.200. PR #351, branch claude/funny-fermat-jmcz41,
  subscribed to PR activity.
- SESSION BUDGET: this was the single highest-value primary action
  (a small, well-scoped, already-diagnosed known-broken repair beats
  starting new research given system health was otherwise green and
  the loop-health ratio was fine). Falling through to open_questions
  queue items or new research was not reached this session — PR
  review/CI babysitting is the remaining capacity sink per the
  subscription now active on #351.
## 2026-07-07 — [PRODUCT] W6 client: the Analyst chat pane on /data (v1.0.200)

TERRITORY: T-CLIENT (AnalystPane.tsx NEW, datamap.tsx, index.css,
visual_check.mjs). ANALYST CONSOLE centerpiece — client half; the
server tool-loop shipped v1.0.199 (#349).

- WHAT: chat pane opened from a third top-left map control
  ([data-vt-analyst], 44px control family). React.lazy chunk — a
  closed pane loads ZERO analyst code (zero-cost-when-off) and never
  polls (single fetch on explicit send). 390px bottom sheet
  (vt-sheet-in, clears the mobile tab bar) / >=640px side panel
  (left:64px, width-capped to clear an open layers panel at 768).
- HONESTY MACHINERY SURFACED: all 8 server response states rendered
  distinctly — awaiting_key ("activates when the key is added — no
  key, no fake answers"), budget_exhausted (spent/limit + reset
  time), 429 busy (retry hint), 401 (sign-in link into the existing
  /login?next= flow, pricing.tsx precedent), 400, 502 ("no answer is
  invented in its place"), network, success. SUCCESS = answer text +
  SOURCE CHIPS (tool + relative freshness, params in tooltip) +
  collapsible "how I got this" tool-trace (per-call ok/error dots) +
  tokens/spent-today/daily-limit/model footline + note/transport_error
  force-close banners.
- MAP COMMANDS EXECUTE LIVE via runAnalystMapCommand: fly_to through
  the map ref; toggle_layer through the SAME `enabled` state the
  layer panel switches use (datamap.tsx:199/2290) — no parallel state,
  and it mirrors the panel's honesty guards (unknown-layer,
  non-live-status, R15 unwired-mid-deploy all return an honest chat
  note instead of flipping a dead switch). Executed commands render
  as "→ flew to 35.94, -96.74" / "→ turned on Severe weather alerts".
- RATCHETS (harness STRENGTHENED, nothing weakened): [data-vt-analyst]
  added to BOTH occlusion hit-tests; new ANALYST battery (button
  exists → opens → panel fully inside viewport [self-see] → input+send
  reachable → close → asserts panel gone AND zero POST /api/analyst
  fired). A SwiftShader capture race on the pane's 0.18s entry
  animation was fixed at capture time only (450ms settle +
  animations:disabled on that one screenshot; page untouched, no
  assertion touched).
- GATES: visual harness 0 hard failures (all pages x 390/768/1440;
  data 390 TTI 1728ms, all-off 1003ms with 0 disallowed calls); node
  425 pass; tsc 64 baseline; pytest unaffected. Version 1.0.199 ->
  1.0.200. Built by worktree subagent (2 sessions — one resumed after
  a transient API cutoff); session re-built the client, re-ran the
  full harness, and reviewed the screenshots before integration.
- MILESTONE: the ANALYST CONSOLE front-end is COMPLETE end-to-end
  (chat + globe + query spine). It activates the moment
  ANTHROPIC_API_KEY lands in Railway — no further deploy needed.

## 2026-07-07 — [PIPELINE] Google Air Quality stream — /api/data/air-quality + analyst tool (v1.0.202)

TERRITORY: T-DATACORE (airQuality module + manifest) + routes.ts &
analyst.ts (SHARED, minimal edits). New EDGE-DOCTRINE data root, human
directive (Google Maps Platform API review): Air Quality is the one
data-bearing Google API worth a real stream now.

- WHAT: server/airQuality.ts — Google Air Quality API current
  conditions archived at our ~16 strategic sites (universal AQI + US
  EPA AQI + PM2.5 + NO2 at 500m). Key-gated on GOOGLE_MAPS_API_KEY
  (already in Railway) with a SECOND honest state awaiting_enable
  (key present but the Air Quality API not yet enabled on the GCP
  project → 403 SERVICE_DISABLED → breaks the cycle, archives nothing,
  activates automatically on enable). FREE-TIER BUDGET GUARD: 5,000
  calls/mo free; 16 sites x 3h poll = 128/day; DAILY_CALL_BUDGET=150,
  rotating-subset self-limit if the site list grows past 18/cycle,
  stated in the route envelope — never silently exceeds, never
  fabricates. Dedup siteId|dateTime; day-file per UTC day; gz-merge
  after 3d; key scrubbed from every record/issue/log (it rides the
  URL query per Google's contract). Route GET /api/data/air-quality
  cache-only; analyst tool "air_quality" wired (7th data tool).
- WHY / HYPOTHESIS (filed open_questions.md, gate-locked): NO2/PM2.5
  over an industrial site is a combustion/activity proxy fusing with
  power-plant/site/grid layers. Gate 1 = AQI-near-site vs the site's
  known output, de-trended against regional background + wind (the
  confounder), discounted for combinations tried, out-of-sample before
  gate 2. RAW archive only until gated — accumulation is the moat
  (Google exposes only 30d history).
- BUILD-FIRST NOTE: of the Google Maps Platform environment APIs, only
  Air Quality earns a stream now; Solar/Pollen/Aerial View have no
  market-signal build and were NOT enabled (idle-meter discipline).
- HOW: worktree subagent (fixtures only, NO live call — the API is
  likely disabled); session read the module line-by-line and wired the
  SHARED route + analyst tool itself (partition rule).
- TESTS: 10 in airQuality.test.ts (key gate/zero-calls, url+body+
  heatmap contract, parse extract, awaiting_enable, happy-path archive,
  dedup, day-file+gz-merge, budget guard rotating-subset, key-scrub,
  manifest). manifests.test.ts + analyst.test.ts pass with the new
  tool.
- GATES: node 438 pass 0 fail; tsc 64 baseline; pytest 476 passed 1
  skipped. Version 1.0.201 -> 1.0.202.
- BLOCKED-FOR-MIKE: enable the Air Quality API in the Google Cloud
  console; the stream then lights up on the existing key.

## 2026-07-07 — [PRODUCT] SCALE S1(a): viewport-bounded serving helper + aircraft proof (v1.0.203)

TERRITORY: T-DATACORE (viewport module) + routes.ts (SHARED, minimal).
First slice of the SCALE program (research/scale_program.md), answering
"faster with everything on, more data, no lost detail/latency".

- FINDING (investigate-first): the aircraft route already accepts a
  viewport but only to derive a center+radius CIRCLE for the upstream
  ADS-B query — the circle circumscribes the on-screen rectangle, so it
  OVER-returns corner aircraft; nothing viewport-filters the SERVED
  payload. No serve-time zoom decimation exists (P-PERF's decimation is
  client-side render styling; archive "adaptive thinning" is storage).
  So S1(a) correctly targets aircraft.
- WHAT: server/viewport.ts — pure, reusable serve-time filter.
  parseBbox (validates ranges, rejects degenerate/garbage, never
  throws), inBbox (boundary-inclusive, antimeridian wrap handled),
  filterByViewport (single-pass O(n)), and applyViewport(payload,
  bboxStr, arrayKey, getLatLon) — the route-facing helper. Aircraft
  route: optional ?bbox= filters served aircraft to the exact viewport
  before serving; absent/invalid bbox = byte-for-byte unchanged; no
  silent caps (viewport_filtered + count_before_viewport +
  count_dropped_offscreen stated). Serve-time only; archive untouched.
- REVIEW NOTE (my own, on the subagent's draft): the subagent placed
  the filter as a ~20-line handler PREAMBLE, which pushed `raceDeadline(`
  toward the routeGuards.test.ts 3000-char guard window (~59 char
  margin — it did NOT weaken the test, correctly). I REFACTORED to
  applyViewport() in viewport.ts and wrapped the 3 response sites
  inline, so the handler body does not grow, raceDeadline stays put,
  and the guard margin is fully restored — AND the helper is now
  directly reusable for the S1 layer sweep. No brittle-guard change
  needed.
- PERF PROOF (viewport.test.ts): fixed 50-point in-bbox cluster + N
  off-screen points; total 1,050 vs 100,050 (95x more data) both return
  50, byte-identical serialized payload; accessor called exactly
  features.length times (O(n), no quadratic). Rendered payload stays
  FLAT as total grows — the program's whole claim, asserted as numbers.
  Plus backward-compat (null/invalid bbox = same object reference),
  parseBbox validation, antimeridian, delta-envelope pass-through.
- FOLLOW-UP (T-CLIENT, own PR): client sends the current map bounds as
  &bbox= on the aircraft fetch (datamap.tsx:~1194) — one line; until
  then this server capability is inert (backward-compatible). Then the
  S1 sweep applies applyViewport to vessels/trains/etc.
- GATES: node 454 pass 0 fail (incl. routeGuards guard); tsc 64
  baseline; pytest 476 passed 1 skipped. Version 1.0.202 -> 1.0.203.

## 2026-07-07 [PIPELINE]+[PRODUCT] — full-program parallel wave (T-DATACORE + T-CLIENT), 5 PRs

Territory: RunPod tooling (scripts/, research/), GRID VISION (scripts/gridvision_*,
datacore/gridvision/, research/), ORBITAL client libs (client/src/lib/orbital/,
datacore/orbital/, research/). Parallelism actually run: 4-agent orbital fan-out
(disjoint lib files) + 1 grid-vision agent; parent serial-merged every PR through
CI. Width bounded by the serial merge gate + the datamap.tsx single-writer hotspot
(O2 wiring is the one serial client job) — stated to the human, not a cap.

- #360 (v1.0.204) [runpod cost-cap gate] scripts/runpod_budget.py — pure, tested
  ($50 balance, append-only JSONL ledger; authorize_job refuses UNBOUNDED / bad
  rate / over-$5-floor and returns the hard max_runtime_seconds the launcher must
  hand RunPod). 16 tests. PLAN: grid-vision detector is the ONLY GPU workload;
  satellite splatting CANCELLED $0 (model research found 0 splat candidates from
  free imagery, glTF covers ~90% at 1/30th size). PRIOR held: the honest free-vs-
  paid analysis flipped the earlier "1-3 marquee splats" scope to zero.
- #361 (v1.0.205) [orbital foundation] client/src/lib/orbital/ tle+propagate+
  geometry+entityJoin (+operators.json, orbital_models.md). propagate.ts = INLINE
  SGP4, 0 KB, validated 0.000e+0 km ECI vs satellite.js@7 + Spacetrack Report #3
  vector — deviated from the "use satellite.js" charter wording, justified
  (hermetic, 0-dep, machine-precision). Deep-space returns null (never faked).
- #362 (v1.0.206) [grid-vision Phase B] scripts/gridvision_* + labels_manifest +
  research/grid_vision_phaseb.md. CC-BY licenses verified LIVE (ETDII/Duke
  figshare CC-BY-4.0; NAIP USDA public domain — STAC field's "proprietary"
  placeholder documented). VERIFIED US composition 74 imgs -> 1408 towers / 6
  substations => v0 is a TOWER detector (substation underrep confirmed, not
  papered over). First GPU job pre-validated vs #360: RTX 4090, max_hours=4,
  worst-case $1.36, authorized.
- #363 (v1.0.207) [orbital render-lib] satWorker+satLayer(CustomLayerInterface)+
  satBuffer. 68 orbital tests, tsc 64. satBuffer split out to keep the SGP4 kernel
  off the main bundle (good call by the builder). classCode=-1 sentinel for
  deep-space/invalid; no silent decimation; getCounts() honesty.
- MERGE MECHANICS: render-lib committed on top of grid-vision then `git rebase
  --onto origin/main <gridvision-sha>` after #362 merged — clean single-commit
  replay, no bundling, preserved the agent's output in git immediately (recycle
  safety) rather than leaving it untracked.
- NEXT (own PRs): O2 datamap.tsx wiring (recipe in orbital_program.md RESUME
  STATE; real build + visual harness; watch Vite worker bundling + SwiftShader
  WebGL2) -> O3 detail panel -> O7 coverage tools. Grid-vision: RunPod fine-tune
  BLOCKED-FOR-MIKE on launch path; build_power_tiles.sh needs power=tower;
  Duke-US zips for substations. Backtest: N/A (no strategy/measurement change).
