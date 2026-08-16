#!/usr/bin/env node
/**
 * visual_check.mjs — the DESIGN.md enforcement harness.
 *
 * Renders key pages headless at the three canonical widths (390 / 768 /
 * 1440), saves screenshots to .visual/, and runs mechanical layout checks.
 * Deterministic: serves the built client (dist/public) with ALL /api/*
 * routes mocked from fixtures — no live backend, no external data APIs
 * (map base tiles still load from the tile CDN; that's the documented
 * scoped exception and only affects pixels, not checks).
 *
 * Usage:
 *   npm run build              # produce dist/public first
 *   node scripts/visual_check.mjs [--soft] [--page data]
 *
 * --soft: report failures but exit 0 (baseline mode).
 * Per CLAUDE.md promotion rule 6, PRs touching client/ must include this
 * run and review the screenshots against DESIGN.md before opening.
 */
import { createServer } from "http";
import { readFileSync, existsSync, mkdirSync, writeFileSync } from "fs";
import path from "path";
import { fileURLToPath } from "url";
import pngjs from "pngjs";
const { PNG } = pngjs;

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const DIST = path.join(ROOT, "dist", "public");
const OUT = path.join(ROOT, ".visual");
const SOFT = process.argv.includes("--soft");
// DEVICE ENVELOPE D1 (scale_program.md, human directive 2026-08-07):
// `--demand` runs the per-layer demand matrix instead of the visual
// battery — each layer measured in isolation at emulated CPU tiers,
// ledger written to .visual/demand_profile.json (diffable across
// versions). Measurement tool, not a gate: it never fails the build.
const DEMAND = process.argv.includes("--demand");

const WIDTHS = [
  { w: 390, h: 844, label: "phone", touch: true },
  { w: 768, h: 1024, label: "tablet", touch: true },
  { w: 1440, h: 900, label: "desktop", touch: false },
];

// Pages under test: name -> app hash route
// Pages under test: map pages get the full map/perf/self-see battery;
// non-map pages get layout + interaction checks only.
const PAGES = {
  data: { route: "/app#/data", map: true },
  // Streams inventory overlay (Phase 4, 2026-07-06) — a page-wide surface,
  // registered here per the Phase 5 ratchet: every new view passes the
  // harness at 390/768/1440 before it ships. Non-map battery (the overlay
  // covers the map).
  streams: { route: "/app#/data/streams", map: false },
  // Data quality dashboard (MAP V2 ROADMAP R6(b), 2026-07-30) — same Phase 5
  // ratchet rule as streams above: every new view passes the harness at
  // 390/768/1440 before it ships.
  quality: { route: "/app#/data/quality", map: false },
  // Signal-strength dashboard (MAP V2 ROADMAP R6(a), 2026-07-30) — same
  // Phase 5 ratchet rule as streams/quality above.
  signals: { route: "/app#/data/signals", map: false },
  // Pipeline-health dashboard (MAP V2 ROADMAP R6(c), 2026-07-31) — same
  // Phase 5 ratchet rule as streams/quality/signals above; closes the R6
  // dashboard trio.
  pipelinehealth: { route: "/app#/data/pipeline-health", map: false },
  // Grid-stress descriptive reading overlay (GRID VISION A1 gate-2 FAIL
  // path, 2026-07-07) — same Phase 5 ratchet rule as streams above.
  gridstress: { route: "/app#/data/grid-stress", map: false },
  // Methane repeat-detection hotspots (gate-2(b) of the GEM METHANE-PLUME
  // × EXTRACTION-REGISTRY PROXIMITY hypothesis, 2026-07-20) — same Phase 5
  // ratchet rule as streams/gridstress above.
  methanehotspots: { route: "/app#/data/methane-hotspots", map: false },
  // Crop conditions — USDA NASS weekly ratings (2026-08-04) — same Phase 5
  // ratchet rule as streams/gridstress/methanehotspots above.
  cropconditions: { route: "/app#/data/crop-conditions", map: false },
  // App Store rankings — consumer-app watchlist (2026-08-05) — same
  // Phase 5 ratchet rule as streams/gridstress/cropconditions above.
  appstorerankings: { route: "/app#/data/appstore-rankings", map: false },
  // GitHub org engineering-momentum watchlist (2026-08-05) — same
  // Phase 5 ratchet rule as streams/gridstress/appstorerankings above.
  githubactivity: { route: "/app#/data/github-activity", map: false },
  // Cboe VIX term structure (2026-08-07) — same Phase 5 ratchet rule as
  // streams/gridstress/githubactivity above.
  vixtermstructure: { route: "/app#/data/vix-term-structure", map: false },
  // European macro cluster — ECB/Eurostat/Bundesbank regime input
  // (2026-08-08) — same Phase 5 ratchet rule as streams/vixtermstructure
  // above.
  eumacro: { route: "/app#/data/eu-macro", map: false },
  // Institutional 13F-HR holdings (2026-08-08) — same Phase 5 ratchet rule
  // as streams/vixtermstructure/eumacro above.
  filings13f: { route: "/app#/data/filings13f", map: false },
  // US macro regime cluster — FRED, 28 public series (2026-08-08) — same
  // Phase 5 ratchet rule as streams/eumacro/filings13f above.
  fredmacro: { route: "/app#/data/fred-macro", map: false },
  // NRC daily reactor status — sortable per-plant table (2026-08-09,
  // closes the "sec_form4_bulk_archive / nrc_outage_reports still without
  // a standalone /data page" queue item — NRC was the one still genuinely
  // missing) — same Phase 5 ratchet rule as streams/fredmacro above.
  nrcreactorstatus: { route: "/app#/data/nrc-reactor-status", map: false },
  // GNSS integrity anomaly signal — first live gate2_pass SIGNAL detail
  // page (2026-08-16) — same Phase 5 ratchet rule as streams/nrcreactorstatus
  // above.
  gnssintegrity: { route: "/app#/data/gnss-integrity", map: false },
  // EU power markets — ENTSO-E load/generation-mix/day-ahead-price, 8
  // bidding zones (2026-08-16) — same Phase 5 ratchet rule as
  // streams/gnssintegrity above.
  eupower: { route: "/app#/data/eu-power", map: false },
  // SEC CNS fails-to-deliver, half-month files (2026-08-16) — same Phase 5
  // ratchet rule as streams/eupower above.
  secftd: { route: "/app#/data/ftd", map: false },
  // CFTC Traders in Financial Futures — same Phase 5 ratchet rule as
  // streams/secftd above.
  tff: { route: "/app#/data/tff", map: false },
  developers: { route: "/developers", map: false },
  // Self-serve preview key management (PLATFORM P3, 2026-07-11) — same
  // Phase 5 ratchet rule as streams/gridstress above. /api/auth/me's
  // fixture defaults to authenticated:false, so this exercises the
  // logged-out "log in to manage keys" state — the most common visitor.
  apikeys: { route: "/apikeys", map: false },
  landing: { route: "/", map: false },
  // MULTI-SEGMENT ROUTE REGRESSION (found 2026-07-11 PLATFORM P3 session,
  // fixed this PR): vite.config.ts's base:"./" baked a relative script src
  // that only resolved correctly for single-segment routes; any 2+-segment
  // route (this one included) rendered a blank white screen in production
  // with no pageerror. Kept in PAGES permanently as the ratchet for that
  // class of bug — see the "app actually mounted" check in CHECKS_SNIPPET,
  // which is what actually catches it (a blank page has no overflow and no
  // interactive elements, so the pre-existing checks alone passed it clean).
  newsletterissue: { route: "/newsletter/test-issue", map: false },
};
const only = process.argv.includes("--page")
  ? process.argv[process.argv.indexOf("--page") + 1]
  : null;

// ── deterministic API fixtures ─────────────────────────────────────────────
const FIXTURES = {
  "/api/auth/me": { authenticated: false },
  "/api/data/layers": {
    layers: [
      { id: "imagery", name: "Satellite imagery", kind: "raw", status: "live", group: "base", costTier: "light", source: "Esri World Imagery", description: "Base imagery." },
      { id: "terrain", name: "Terrain (hillshade)", kind: "raw", status: "live", group: "base", costTier: "moderate", source: "Mapterhorn (© Mapterhorn)", description: "Global hillshade, off by default." },
      { id: "seafloor", name: "Seafloor (drain the ocean)", kind: "raw", status: "live", field: true, group: "base", costTier: "moderate", source: "NOAA ETOPO1 via Terrain Tiles on AWS", description: "Depth-tinted ocean-floor relief overlay, off by default." },
      { id: "weather", name: "Weather radar (US)", kind: "raw", status: "live", field: true, group: "base", costTier: "heavy", source: "NOAA nowCOAST (public domain)", description: "US radar mosaic, off by default." },
      { id: "weather_temp", name: "Temperature (global)", kind: "raw", status: "live", field: true, group: "base", costTier: "heavy", source: "OpenWeatherMap (© OpenWeatherMap)", description: "Global temp field, off by default." },
      { id: "weather_wind", name: "Wind (global)", kind: "raw", status: "live", field: true, group: "base", costTier: "heavy", source: "OpenWeatherMap (© OpenWeatherMap)", description: "Global wind field, off by default." },
      { id: "aircraft", name: "Live aircraft (ADS-B)", kind: "raw", status: "live", group: "live", costTier: "heavy", source: "adsb.lol/airplanes.live", description: "Live aircraft." },
      { id: "vessels", name: "Live vessels (AIS)", kind: "raw", status: "awaiting_key", group: "live", costTier: "moderate", source: "aisstream.io", description: "Needs AISSTREAM_KEY." },
      { id: "trains", name: "Live trains (rail)", kind: "raw", status: "live", group: "live", costTier: "light", source: "Digitraffic FI + Entur NO", description: "FI+NO launch coverage." },
      { id: "sites", name: "Strategic sites", kind: "raw", status: "live", group: "facilities", costTier: "light", source: "datacore/sites", description: "Reference sites." },
      { id: "powerplants", name: "US power plants", kind: "raw", status: "live", group: "facilities", costTier: "moderate", source: "WRI GPPD (CC BY 4.0)", description: "US plants by fuel." },
      // [REPAIR R15 2026-07-07] powergrid was absent here, so the
      // toggle-consistency battery never clicked it — and the missing
      // LAYER_GROUP entry (permanent "reload to enable") shipped unseen.
      // Every toggleable registry layer must appear in this fixture;
      // the wiring ratchet (layersWiring.test.ts) pins the source side.
      { id: "powergrid", name: "Power grid (TX pilot)", kind: "raw", status: "live", group: "facilities", costTier: "moderate", source: "OpenStreetMap power features (© OpenStreetMap contributors, ODbL)", description: "TX pilot vector tiles." },
      // [REPAIR 2026-07-25] These 9 were shipped end-to-end (client symbol
      // layer + server route) across several prior sessions but never added
      // here — so the self-see/toggle-consistency/legend-parity batteries
      // above have never actually exercised them (filed as open debt across
      // four session log entries before this fix). Every toggleable
      // registry layer must appear in this fixture; see the powergrid note
      // above for the precedent this repeats.
      { id: "nukefacilities", name: "Nuclear facilities (fuel cycle)", kind: "raw", status: "live", group: "facilities", costTier: "light", source: "Wikidata (CC0 1.0), curated", description: "67 fuel-cycle/production facilities, building-trefoil symbols by category." },
      { id: "faa_airports", name: "FAA airport status", kind: "raw", status: "live", group: "facilities", costTier: "light", source: "FAA National Airspace System Status (public domain)", description: "Ground stops/delays/closures, rolling snapshot." },
      // [REPAIR, found alongside this session's NRC dashboard page]: this
      // toggleable registry layer (datacore/layers.json) was missing from
      // this fixture, same class of gap the R15 (2026-07-07) and
      // 2026-07-25 fixes closed for 9 other layers — the toggle-
      // consistency/self-see/legend-parity batteries never exercised it.
      { id: "nrc_reactor_status", name: "US nuclear reactor status (NRC, daily)", kind: "raw", status: "live", group: "facilities", costTier: "light", source: "U.S. NRC daily Power Reactor Status Reports (nrc.gov, public domain, keyless)", description: "Percent-of-rated-thermal-power per operating unit, joined onto the nuclear-plant registry." },
      { id: "border_waits", name: "CBP land-border wait times", kind: "raw", status: "live", group: "facilities", costTier: "light", source: "CBP Border Wait Times (public domain)", description: "Hourly wait times at US land ports of entry." },
      { id: "nucleartests", name: "Nuclear tests 1945–1998 (time machine)", kind: "raw", status: "live", group: "hazards", costTier: "light", source: "SIPRI / Johnston archive nuclear explosions catalog", description: "2,027 located tests, emplacement symbols, year-slider scrub." },
      { id: "radiation", name: "Ambient radiation (gamma)", kind: "raw", status: "live", group: "hazards", costTier: "light", source: "BfS · Health Canada · STUK/FMI · EPA RadNet", description: "Observed gamma dose-rate monitors, four national networks." },
      { id: "nukeaccidents", name: "Nuclear accidents & incidents", kind: "raw", status: "live", group: "hazards", costTier: "light", source: "Wikidata (CC0 1.0), curated", description: "46 accident/incident sites 1949–2024, INES-tinted." },
      { id: "pfas", name: "PFAS drinking-water detections (EPA UCMR 5)", kind: "raw", status: "live", group: "hazards", costTier: "moderate", source: "U.S. EPA UCMR 5, public domain", description: "3,417 water systems with a detected PFAS compound." },
      { id: "cancerrates", name: "County cancer rates (NCI State Cancer Profiles)", kind: "raw", status: "live", field: true, group: "hazards", costTier: "moderate", source: "National Cancer Institute, State Cancer Profiles, public domain", description: "County-level incidence/mortality choropleth, 3,143 US counties." },
      { id: "methane_plumes", name: "Methane plumes (GEM GMET)", kind: "raw", status: "live", group: "environmental", costTier: "moderate", source: "Global Energy Monitor GMET (CC BY 4.0)", description: "Satellite methane-plume detections, nearest-asset match." },
      { id: "coal_mine_features", name: "Coal mine boundaries & infrastructure (GEM)", kind: "raw", status: "live", group: "environmental", costTier: "light", source: "Global Energy Monitor (CC BY 4.0)", description: "Mine boundary polygons + point infrastructure features." },
      // freshness (Phase 5, three of the five fixture health states so the
      // visual harness actually exercises the chip's color/label variants):
      { id: "insider", name: "Insider transactions (Form 4)", kind: "raw", status: "live", group: "filings", costTier: "light", source: "SEC EDGAR", description: "Recent Form 4 filings as filed.", freshness: { stream: "filings", health: "live", age_hours: 0.4, health_note: "newest file 0.4h old" } },
      { id: "earnings", name: "Earnings language (8-K)", kind: "raw", status: "live", group: "filings", costTier: "light", source: "SEC EDGAR", description: "As-filed 8-K Item 2.02 results/guidance releases." },
      { id: "shortvol", name: "Short-sale volume (FINRA)", kind: "raw", status: "live", group: "filings", costTier: "light", source: "FINRA Reg SHO", description: "Daily consolidated short-marked execution volume per symbol — a flow proxy, not short interest.", freshness: { stream: "finrashortvol", health: "stale", age_hours: 411.2, health_note: "newest file 411.2h old — exceeds cadence-derived threshold" } },
      { id: "attention", name: "Attention proxy (Wikipedia pageviews)", kind: "raw", status: "live", group: "filings", costTier: "light", source: "Wikimedia pageviews API", description: "Daily article pageviews for a curated ticker seed — an attention proxy, not a signal." },
      { id: "cot", name: "Commitments of Traders (CFTC, disaggregated)", kind: "raw", status: "live", group: "filings", costTier: "light", source: "CFTC Public Reporting Socrata API", description: "Weekly futures-only positioning by trader category — a positioning proxy, not a signal.", freshness: { stream: "cftccot", health: "recent", age_hours: 122.9, health_note: "newest file 122.9h old (within cadence)" } },
      { id: "tff", name: "Traders in Financial Futures (CFTC, futures-only)", kind: "raw", status: "live", group: "filings", costTier: "light", source: "CFTC Public Reporting Socrata API, futures-only", description: "Weekly financial-futures positioning by trader category — a positioning proxy, not a signal." },
      { id: "portdwell", name: "Port dwell (arrivals/departures)", kind: "raw", status: "live", group: "filings", costTier: "light", source: "Own AIS archive + verified port geofences", description: "Per-port dwell stats; lower bounds; anomaly SIGNAL gate-2 locked." },
      { id: "secftd", name: "Fails-to-deliver (SEC CNS)", kind: "raw", status: "live", group: "filings", costTier: "light", source: "SEC CNS fails-to-deliver, half-month files (public domain, no API key required)", description: "Trailer-checksummed aggregate net fail balances per settlement date — a level, not a daily flow; raw spikes alone are a crowded signal." },
      { id: "graph", name: "Everything Graph", kind: "raw", status: "live", group: "graph", costTier: "light", source: "Own join over Form 4 + entity_map + AIS port-dwell archive", description: "Entity search across insiders, facilities, and vessels. RAW join with provenance, no predictive claim." },
      { id: "fires", name: "Active fires (VIIRS)", kind: "raw", status: "awaiting_key", group: "environmental", costTier: "moderate", source: "NASA FIRMS / LANCE", description: "Needs NASA_FIRMS_MAP_KEY." },
      { id: "nightlights", name: "Night lights radiance (GIBS)", kind: "raw", status: "live", field: true, group: "environmental", costTier: "moderate", source: "NASA GIBS/ESDIS — VIIRS/SNPP Day/Night Band", description: "Daily radiance imagery, dated (defaults to yesterday)." },
      { id: "aerosol", name: "Aerosol optical depth (GIBS)", kind: "raw", status: "live", field: true, group: "environmental", costTier: "moderate", source: "NASA GIBS/ESDIS — MODIS Combined Value-Added AOD", description: "Daily aerosol optical depth, dated (defaults to yesterday)." },
      { id: "vegetation", name: "Vegetation health (NDVI, GIBS)", kind: "raw", status: "live", field: true, group: "environmental", costTier: "moderate", source: "NASA GIBS/ESDIS — VIIRS/SNPP NDVI 8-day", description: "Vegetation greenness (NDVI) 8-day composite, dated (defaults to yesterday)." },
      { id: "soilmoisture", name: "Soil moisture (SMAP, GIBS)", kind: "raw", status: "live", field: true, group: "environmental", costTier: "moderate", source: "NASA GIBS/ESDIS — SMAP L4 root-zone", description: "Root-zone soil moisture, dated (~6-day lag, defaults ~7 days back)." },
      { id: "no2", name: "Nitrogen dioxide (TROPOMI, GIBS)", kind: "raw", status: "live", field: true, group: "environmental", costTier: "moderate", source: "NASA GIBS/ESDIS — Sentinel-5P/TROPOMI NO₂", description: "Tropospheric NO₂ column, dated (defaults to yesterday)." },
      { id: "floods", name: "Flood / water extent (MODIS, GIBS)", kind: "raw", status: "live", field: true, group: "environmental", costTier: "moderate", source: "NASA GIBS/ESDIS — MODIS Terra+Aqua 3-day flood composite", description: "Rolling 3-day flood/water extent composite, dated (defaults to yesterday)." },
      { id: "biomass", name: "Aboveground biomass density (GEDI, GIBS)", kind: "raw", status: "live", field: true, group: "environmental", costTier: "moderate", source: "NASA GIBS/ESDIS — GEDI L4B aboveground biomass density", description: "Static mission-life mean (2019-04 to 2023-03), no scrubber." },
      { id: "rivergauges", name: "River gauges (barge corridor)", kind: "raw", status: "live", group: "environmental", costTier: "light", source: "USGS NWIS (public domain)", description: "Live stage/discharge at 14 barge-corridor gauges." },
      { id: "alerts", name: "Severe weather alerts (NWS)", kind: "raw", status: "live", group: "environmental", costTier: "moderate", source: "National Weather Service (public domain)", description: "Active NWS warnings/watches, colored by severity." },
      { id: "spaceweather", name: "Space weather (NOAA SWPC)", kind: "raw", status: "live", group: "environmental", costTier: "moderate", source: "NOAA Space Weather Prediction Center (public domain)", description: "Aurora oval (OVATION model FORECAST) + observed Kp / R-S-G scales / solar wind." },
      { id: "earthquakes", name: "Earthquakes (USGS)", kind: "raw", status: "live", group: "environmental", costTier: "light", source: "USGS Earthquake Hazards Program (public domain)", description: "Real-time M2.5+ seismic events, sized/colored by magnitude." },
      { id: "volcanoes", name: "Volcano alert levels (USGS/GVP)", kind: "raw", status: "live", group: "environmental", costTier: "light", source: "USGS Volcano Hazards Program + Global Volcanism Program, Smithsonian Institution", description: "Currently-elevated volcanoes, USGS alert level, coordinates joined live from GVP." },
      { id: "buoys", name: "Ocean buoys (NDBC)", kind: "raw", status: "live", group: "environmental", costTier: "light", source: "NOAA National Data Buoy Center (public domain)", description: "Latest wave/wind/pressure readings, ~889 stations worldwide." },
      { id: "surfacewater", name: "Surface water (1984–2021)", kind: "raw", status: "live", field: true, group: "environmental", costTier: "moderate", source: "EC JRC/Google GSW v2021", description: "Static water occurrence, off by default." },
      { id: "forest", name: "Forest cover (2020)", kind: "raw", status: "live", field: true, group: "environmental", costTier: "moderate", source: "EC JRC GFC2020 via GFW", description: "Static forest extent, off by default." },
      { id: "boundaries", name: "Country borders", kind: "raw", status: "live", group: "base", costTier: "light", source: "Natural Earth 1:110m (public domain)", description: "Reference borders, off by default." },
      { id: "tank_fill", name: "Tank-fill % (Sentinel-2)", kind: "signal", status: "planned", group: "signals", costTier: "light", source: "Copernicus", description: "Gate-2 locked." },
    ],
  },
  // W3 TIME SCRUBBER (server/queryEngine.ts querySnapshot) — a small,
  // deterministic replay payload so the panel's status line + rendered
  // points are real content, not the generic "{}" fallback.
  "/api/data/snapshot": {
    kind: "raw", layer: "aircraft", mode: "position",
    requested_at: "2026-07-08T14:00:00.000Z", bucket_at: "2026-07-08T14:00:00.000Z",
    data: [
      { id: "fx1", lat: 36.1, lon: -97.0 },
      { id: "fx2", lat: 35.8, lon: -96.6 },
      { id: "fx3", lat: 36.4, lon: -97.3 },
    ],
    count: 3, count_before_viewport: 3, count_dropped_offscreen: 0,
    viewport_filtered: true, capped: false,
    freshness: "2026-07-08T14:52:00.000Z",
    provenance: "own position archive (fixture)",
    window: { min_iso: "2026-07-01T14:00:00.000Z", max_iso: "2026-07-08T14:00:00.000Z", days: 7 },
    note: "fixture snapshot for the visual harness",
  },
  // 10,000 synthetic aircraft — the DESIGN.md performance budget says map
  // interactions stay smooth at 10k+ features; the fixture proves the
  // rendering path at that scale (deterministic pseudo-random spread).
  "/api/data/aircraft": (() => {
    const aircraft = [];
    let seed = 42;
    const rnd = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
    const types = ["B738", "A320", "C172", "AT76", "EC35", null];
    for (let i = 0; i < 10000; i++) {
      aircraft.push({
        icao24: "fx" + i.toString(16), callsign: "TST" + i, origin_country: "US",
        lon: -125 + rnd() * 58, lat: 25 + rnd() * 24,
        altitude_m: i % 17 === 0 ? null : Math.round(rnd() * 12000),
        on_ground: i % 23 === 0,
        velocity_ms: Math.round(rnd() * 250),
        heading: Math.round(rnd() * 359),
        type: types[i % types.length], category: null,
      });
    }
    return { source: "fixture", kind: "raw", time: 1, count: aircraft.length, aircraft };
  })(),
  "/api/data/vessels": { enabled: false, reason: "AISSTREAM_KEY not set (fixture)", vessels: [] },
  "/api/data/fires": { enabled: false, kind: "raw", reason: "NASA_FIRMS_MAP_KEY not set (fixture)", fires: [] },
  "/api/data/alerts": {
    kind: "raw", source: "National Weather Service (fixture)", time: 1, count: 2, zone_only: 3,
    alerts: [
      { id: "urn:oid:fixture.tornado", event: "Tornado Warning", severity: "Extreme", area: "Payne, OK", ends: "2026-07-05T13:00:00-05:00",
        rings: [[[-97.2, 35.8], [-96.6, 35.8], [-96.6, 36.2], [-97.2, 36.2], [-97.2, 35.8]]] },
      { id: "urn:oid:fixture.flood", event: "Flood Watch", severity: "Moderate", area: "St. Louis Metro, MO", ends: null,
        rings: [[[-90.6, 38.4], [-89.9, 38.4], [-89.9, 38.9], [-90.6, 38.9], [-90.6, 38.4]]] },
    ],
  },
  "/api/data/spaceweather": {
    kind: "raw", source: "NOAA SWPC (fixture)", time: 1,
    note: "fixture: observed Kp/scales/wind + OVATION forecast cells",
    kp_recent: [
      { t: "2026-07-29T06:00:00", kp: 1.67, a: 6, n: 8, rt: "2026-07-29" },
      { t: "2026-07-29T09:00:00", kp: 3.33, a: 18, n: 8, rt: "2026-07-29" },
    ],
    scales: {
      current: { kind: "current", date: "2026-07-29", time: "12:37:00", r: "0", s: "0", g: "1", rMinorProb: null, rMajorProb: null, sProb: null },
      forecast: [{ kind: "forecast", date: "2026-07-30", time: "00:00:00", r: null, s: null, g: "0", rMinorProb: 30, rMajorProb: 1, sProb: 1 }],
    },
    wind: { speedKms: 329, speedAt: "2026-07-29T12:32:00Z", btNt: 4, bzNt: 1, magAt: "2026-07-29T12:34:00Z" },
    aurora: {
      obs: "2026-07-29T12:30:00Z", forecast: "2026-07-29T13:51:00Z", max: 42, aggDeg: 2, minVal: 2,
      cells: [[-160, 62, 12], [-158, 62, 25], [-156, 64, 42], [-100, 60, 8], [20, 68, 15], [-150, -72, 30]],
    },
    alerts_recent: [
      { id: "K05W:2026-07-27 02:11:00.000", productId: "K05W", issued: "2026-07-27 02:11:00.000", code: "WARK05", serial: 2101, title: "WARNING: Geomagnetic K-index of 5 expected", message: "fixture", rt: "2026-07-29" },
    ],
  },
  "/api/data/rivergauges": {
    kind: "raw", source: "USGS Water Services (fixture)", time: 1, count: 2,
    gauges: [
      { site: "07010000", name: "Mississippi River at St. Louis, MO", param: "00065", d: "2026-07-05T00:15:00.000-05:00", v: 15.13, q: "P", lat: 38.62889, lon: -90.17972 },
      { site: "07032000", name: "Mississippi River at Memphis, TN", param: "00060", d: "2026-07-05T00:15:00.000-06:00", v: 512000, q: "P", lat: 35.12278, lon: -90.0775 },
    ],
  },
  "/api/data/earthquakes": {
    kind: "raw", source: "USGS Earthquake Hazards Program (fixture)", time: 1, count: 2,
    quakes: [
      { id: "fx0001", mag: 4.6, place: "12 km SW of Fixture City, CA", lat: 34.02, lon: -118.5, depth: 8.2, time: 1, updated: 1, tsunami: false, sig: 312, net: "ci", magType: "ml", type: "earthquake", status: "reviewed", url: "https://earthquake.usgs.gov/earthquakes/eventpage/fx0001" },
      { id: "fx0002", mag: 6.3, place: "Offshore Fixture Trench", lat: 38.3, lon: -123.1, depth: 22.0, time: 1, updated: 1, tsunami: true, sig: 820, net: "us", magType: "mww", type: "earthquake", status: "reviewed", url: "https://earthquake.usgs.gov/earthquakes/eventpage/fx0002" },
    ],
  },
  "/api/data/volcanoes": {
    kind: "raw", source: "USGS Volcano Hazards Program + Global Volcanism Program, Smithsonian Institution (fixture)", time: 1, count: 2,
    volcanoes: [
      { vnum: "fx0001", name: "Fixture Sitkin", obs: "avo", obsFullname: "Alaska Volcano Observatory", alertLevel: "WATCH", colorCode: "ORANGE", noticeId: "fx-notice-1", noticeUrl: "https://volcanoes.usgs.gov/hans-public/notice/fx-notice-1", sentUtc: "2026-08-01 19:17:44", sentUnixtime: 1, lat: 52.076, lon: -176.13, elevationM: 1740, country: "United States", rt: "2026-08-02" },
      { vnum: "fx0002", name: "Fixture Kilauea", obs: "hvo", obsFullname: "Hawaiian Volcano Observatory", alertLevel: "ADVISORY", colorCode: "YELLOW", noticeId: "fx-notice-2", noticeUrl: "https://volcanoes.usgs.gov/hans-public/notice/fx-notice-2", sentUtc: "2026-08-01 19:07:02", sentUnixtime: 2, lat: 19.421, lon: -155.287, elevationM: 1222, country: "United States", rt: "2026-08-02" },
    ],
  },
  "/api/data/buoys": {
    kind: "raw", source: "NOAA National Data Buoy Center (fixture)", time: 1, count: 2,
    buoys: [
      { station: "46042", lat: 36.79, lon: -122.4, time: 1, windDir: 300, windSpeed: 7.2, gust: 9.1, waveHeight: 2.1, dominantPeriod: 11, avgPeriod: 8.4, waveDir: 285, pressure: 1015.2, pressureTendency: -0.4, airTemp: 14.3, waterTemp: 13.8, dewpoint: 10.1, visibility: null, tide: null },
      { station: "44013", lat: 42.35, lon: -70.65, time: 1, windDir: 210, windSpeed: 5.1, gust: 6.8, waveHeight: null, dominantPeriod: null, avgPeriod: null, waveDir: null, pressure: 1009.8, pressureTendency: 0.2, airTemp: 18.6, waterTemp: 17.9, dewpoint: 15.0, visibility: null, tide: null },
    ],
  },
  "/api/data/sites": {
    kind: "raw",
    categories: {
      tank_farm: { label: "Crude storage", color: "#fbb24c" },
      steel_mill: { label: "Steel mills", color: "#ff5a6e" },
      port: { label: "Ports", color: "#4ade80" },
    },
    sites: [
      { id: "cushing", name: "Cushing Oil Hub", category: "tank_farm", lat: 35.985, lon: -96.767, operator: "Multiple", relevance: "WTI delivery point; Sentinel-2 tank-fill signal ground (EIA benchmark).", note: "" },
      { id: "port_la", name: "Port of Los Angeles", category: "port", lat: 33.74, lon: -118.272, operator: "POLA", relevance: "#1 US container port.", note: "" },
    ],
  },
  "/api/data/insider": {
    kind: "raw", source: "SEC EDGAR (Form 4)", time: 1, count: 2,
    filings: [
      {
        issuerName: "CYPHERPUNK TECHNOLOGIES INC.", issuerTradingSymbol: "CYPH",
        owners: [{ cik: "1645967", name: "Richard Christian M", isDirector: true, isOfficer: false, isTenPercentOwner: false, officerTitle: null }],
        transactions: [{ table: "derivative", kind: "award_grant", shares: 75000, pricePerShare: 0, transactionDate: "2026-07-01" }],
      },
      {
        issuerName: "STRATUS PROPERTIES INC", issuerTradingSymbol: "STRS",
        owners: [{ cik: "1317904", name: "Oasis Management Co Ltd.", isDirector: false, isOfficer: false, isTenPercentOwner: true, officerTitle: null }],
        transactions: [{ table: "nonDerivative", kind: "open_market_sale", shares: 10000, pricePerShare: 28.9, transactionDate: "2026-06-30" }],
      },
    ],
  },
  "/api/health": { status: "ok", checks: {} },
  "/api/data/boundaries": {
    kind: "raw", source: "Natural Earth 1:110m admin-0 (public domain, fixture)",
    type: "FeatureCollection",
    features: [
      { type: "Feature", properties: { name: "United States of America", iso3: "USA" },
        geometry: { type: "LineString", coordinates: [[-125, 49], [-66, 49], [-66, 25], [-125, 25], [-125, 49]] } },
    ],
  },
  "/api/data/weather/global/status": { status: "ok", note: "fixture: key active" },
  "/api/data/platform/stats": { layers_live: 18, layers_total: 19, streams_recording: 20, observations: 1284550, observations_as_of: "2026-07-05T00:00:00Z" },
  "/api/data/weather/grid": {
    kind: "raw", source: "OpenWeatherMap current-weather point samples (fixture)",
    note: "sampled grid — one observation per ~310 km; arrows/labels never denser than the data",
    spacing_km: 310, sampled: 6,
    points: [
      { la: 30, lo: -110, tc: 31.5, wd: 200, ws: 6.2 }, { la: 30, lo: -95, tc: 29.0, wd: 180, ws: 4.0 },
      { la: 30, lo: -80, tc: 27.4, wd: 150, ws: 5.1 }, { la: 42, lo: -110, tc: 22.1, wd: 250, ws: 8.8 },
      { la: 42, lo: -95, tc: 24.7, wd: 230, ws: 7.3 }, { la: 42, lo: -80, tc: 21.9, wd: 270, ws: 9.9 },
    ],
  },
  "/api/v1/meta": {
    version: "v1",
    auth: "x-api-key header (or ?api_key=). Keys are invite-only during the preview — join the waitlist on /developers.",
    endpoints: [
      { path: "/api/v1/tracks/:kind/:id", params: "kind; id; ?hours<=168", desc: "Recent position track from our archive." },
      { path: "/api/v1/stats/portdwell", params: "-", desc: "Per-port dwell statistics.", preview: "/api/data/portdwell" },
      { path: "/api/v1/stats/shadow", params: "-", desc: "Dark-ship RAW statistics.", preview: "/api/data/shadowstats" },
      { path: "/api/v1/stats/archive", params: "-", desc: "Archive growth metadata.", preview: "/api/data/archive/stats" },
      { path: "/api/v1/graph", params: "?entity=<ticker|MMSI|CIK|facility id>&hops<=3 (omit entity for counts-only)", desc: "Everything Graph v1 — entity neighborhood join.", preview: "/api/data/graph" },
      { path: "/api/v1/meta", params: "-", desc: "This document.", preview: "/api/v1/meta" },
    ],
    coming_gated: ["entity timelines (Graph v1)", "tank-fill readings (gate 2)"],
    limits: {
      dev: { perMinute: 60, perDay: 10000 },
      pro: { perMinute: 600, perDay: 100000 },
      enterprise: { perMinute: 3000, perDay: 1000000 },
    },
    license_marks: {
      "tracks/aircraft": { license: "ODbL 1.0 share-alike (fixture)", attribution: "adsb.lol", resell: "share-alike" },
      "stats/archive": { license: "VolTradeAI operational metadata", attribution: "VolTradeAI", resell: "ok" },
    },
    disclaimer: "Data as-is; not for safety-of-life use (fixture).",
  },
  "/api/data/archive/stats": { kinds: { aircraft: { days: 2, samples: 96 }, vessels: { days: 2, samples: 96 } }, totalBytes: 12582912 },
  // Backs the newsletterissue PAGES entry (multi-segment route regression,
  // see vite.config.ts's base:"/" fix) — matched via startsWith prefix, so
  // any /api/newsletter/issues/:slug request gets this single-issue shape.
  "/api/newsletter/issues": {
    slug: "test-issue", title: "Fixture Issue", subtitle: "Harness fixture",
    date: "2026-07-11", excerpt: "Fixture excerpt.", html: "<p>Fixture body.</p>",
  },
  "/api/data/portdwell": {
    kind: "raw", source: "Derived from our own AIS position archive (fixture)",
    window_hours: 168, vessels_seen: 240, visits_completed: 23, in_port_now: 7, anomaly_count: 1,
    ports: [
      { id: "port_la", name: "Port of Los Angeles", lat: 33.74, lon: -118.272, visits_completed: 12, unique_vessels: 11, in_port_now: 4, dwell_median_h: 22.5, dwell_p90_h: 41, dwell_max_h: 78, anomaly_count: 1, anomaly_examples: [{ mmsi: "366999707", name: "FIXTURE STAR", dwell_h: 78, median_h: 22.5 }] },
      { id: "port_lb", name: "Port of Long Beach", lat: 33.7515, lon: -118.213, visits_completed: 8, unique_vessels: 8, in_port_now: 2, dwell_median_h: 19, dwell_p90_h: 30, dwell_max_h: 33, anomaly_count: 0, anomaly_examples: [] },
      { id: "port_savannah", name: "Port of Savannah (Garden City)", lat: 32.129, lon: -81.144, visits_completed: 3, unique_vessels: 3, in_port_now: 1, dwell_median_h: 15, dwell_p90_h: 18, dwell_max_h: 18, anomaly_count: 0, anomaly_examples: [] },
    ],
    caveat: "RAW statistics; dwell figures are lower bounds; anomaly flags suppressed on thin history (fixture).",
  },
  "/api/data/graph": {
    kind: "raw", built_at: 1,
    counts: { nodes: 42, edges: 65, company: 9, person: 14, facility: 16, vessel: 3, insider_of: 18, operates: 12, calls_at: 31, owns: 4 },
    caveat: "RAW graph — every edge asserts a relationship with provenance (fixture).",
    note: "pass ?entity=<ticker|MMSI|CIK|facility id>&hops=1 for a neighborhood query",
  },
  // W5 ENTITY DOSSIER — click-a-feature card enrichment. Non-empty so any
  // page load that happens to click an entity exercises the real render
  // path (companyNode/insider/contracts/nearest_sites all populated), not
  // just an empty-state early-return.
  "/api/data/dossier": {
    kind: "raw", built_at: 1,
    query: { entity: "cushing_hub", lat: 35.94, lon: -96.75, hops: 2 },
    identity: { id: "facility:site:cushing_hub", type: "facility", label: "Cushing Oil Hub", attrs: { lat: 35.94, lon: -96.75 } },
    graph: {
      nodes: [{ id: "company:ENB", type: "company", label: "ENB", attrs: {} }],
      edges: [
        { type: "operates", from: "company:ENB", to: "facility:site:cushing_hub", confidence: "high", attrs: {} },
        { type: "insider_of", from: "person:0000123", to: "company:ENB", confidence: "high", attrs: { visit_count: 0 } },
      ],
    },
    contracts: [{ r: "Enbridge Inc", tkr: "ENB", amt: 1250000, ag: "Department of Energy", rt: "2026-07-01" }],
    contracts_capped: false,
    nearest_sites: [{ id: "la_port", name: "Port of LA", category: "port", km: 1500 }],
    caveat: "RAW composition — fixture.",
  },
  // Grid-stress descriptive reading (GRID VISION A1 gate-2 FAIL path,
  // 2026-07-07) — descriptive-only, predictive:false always present.
  "/api/data/grid-stress": {
    kind: "descriptive",
    source: "VoltradeAI derived composite (EIA-930 ERCOT demand/forecast + NOAA CPC TX degree days) (fixture)",
    attribution: "EIA-930 Hourly Electric Grid Monitor; NOAA Climate Prediction Center",
    predictive: false,
    validation_status: "GATE 2 NOT PASSED (2026-07-07) — two outcome designs voided on their own spot-validation rules (fixture)",
    note: "Descriptive-only TX/ERCOT reading; equal-weighted composite, deliberately not the voided gate-2 fitted weights (fixture).",
    time: 1, history_depth_days: 412,
    reading: {
      date: "2026-07-06", respondent: "ERCO",
      demand_peak_mw: 68500, demand_percentile: 82, demand_sample_days: 41,
      forecast_strain_pct: 1.4, strain_percentile: 61, strain_sample_days: 41,
      weather_degree_days: 27, weather_percentile: 74, weather_sample_days: 62,
      composite_percentile: 72,
    },
  },
  // Crop conditions — USDA NASS weekly ratings (2026-08-04, RAW display).
  "/api/data/crop-conditions": {
    kind: "raw",
    source: "USDA NASS QuickStats — weekly crop condition ratings (US government data) (fixture)",
    attribution: "USDA National Agricultural Statistics Service (QuickStats)",
    time: "2026-08-03T00:00:00.000Z",
    latest_week: "2026-08-02",
    count: 10,
    note: "national weekly CONDITION ratings for corn + soybeans (fixture)",
    rows: [
      { commodity: "CORN", week_ending: "2026-08-02", item: "CORN - CONDITION, MEASURED IN PCT VERY POOR", pct: 4, rt: "2026-08-03" },
      { commodity: "CORN", week_ending: "2026-08-02", item: "CORN - CONDITION, MEASURED IN PCT POOR", pct: 10, rt: "2026-08-03" },
      { commodity: "CORN", week_ending: "2026-08-02", item: "CORN - CONDITION, MEASURED IN PCT FAIR", pct: 25, rt: "2026-08-03" },
      { commodity: "CORN", week_ending: "2026-08-02", item: "CORN - CONDITION, MEASURED IN PCT GOOD", pct: 48, rt: "2026-08-03" },
      { commodity: "CORN", week_ending: "2026-08-02", item: "CORN - CONDITION, MEASURED IN PCT EXCELLENT", pct: 13, rt: "2026-08-03" },
      { commodity: "SOYBEANS", week_ending: "2026-08-02", item: "SOYBEANS - CONDITION, MEASURED IN PCT VERY POOR", pct: 2, rt: "2026-08-03" },
      { commodity: "SOYBEANS", week_ending: "2026-08-02", item: "SOYBEANS - CONDITION, MEASURED IN PCT POOR", pct: 7, rt: "2026-08-03" },
      { commodity: "SOYBEANS", week_ending: "2026-08-02", item: "SOYBEANS - CONDITION, MEASURED IN PCT FAIR", pct: 28, rt: "2026-08-03" },
      { commodity: "SOYBEANS", week_ending: "2026-08-02", item: "SOYBEANS - CONDITION, MEASURED IN PCT GOOD", pct: 52, rt: "2026-08-03" },
      { commodity: "SOYBEANS", week_ending: "2026-08-02", item: "SOYBEANS - CONDITION, MEASURED IN PCT EXCELLENT", pct: 11, rt: "2026-08-03" },
    ],
  },
  // App Store rankings — consumer-app watchlist (2026-08-05, RAW display).
  "/api/data/appstore-rankings": {
    kind: "raw",
    source: "Apple marketingtools RSS top-free/top-grossing charts (US/GB/CA) + iTunes Lookup rating counts (US) (fixture)",
    attribution: "Apple Inc.",
    time: 1785928979602,
    count: 6,
    note: "GATE 1 (DATA) only — no signal validated yet (fixture).",
    records: [
      { t: "rank", ticker: "DUOL", company: "Duolingo, Inc.", appId: "570060128", storefront: "us", chart: "top-free", rank: 65, rt: "2026-08-05" },
      { t: "rank", ticker: "DUOL", company: "Duolingo, Inc.", appId: "570060128", storefront: "us", chart: "top-grossing", rank: 22, rt: "2026-08-05" },
      { t: "rank", ticker: "BMBL", company: "Bumble Inc.", appId: "930441707", storefront: "us", chart: "top-free", rank: null, rt: "2026-08-05" },
      { t: "rank", ticker: "BMBL", company: "Bumble Inc.", appId: "930441707", storefront: "us", chart: "top-grossing", rank: null, rt: "2026-08-05" },
      { t: "rating", ticker: "DUOL", company: "Duolingo, Inc.", appId: "570060128", avgRating: 4.72, ratingCount: 5357702, version: "7.133.0", rt: "2026-08-05" },
      { t: "rating", ticker: "BMBL", company: "Bumble Inc.", appId: "930441707", avgRating: 3.9, ratingCount: 412300, version: "6.2.0", rt: "2026-08-05" },
    ],
  },
  // GitHub org engineering-momentum — develop-in-public watchlist
  // (2026-08-05, RAW display).
  "/api/data/github-activity": {
    kind: "raw",
    source: "GitHub REST Search API (search/issues, search/commits) — keyless (fixture)",
    attribution: "GitHub, Inc. (public repository activity, aggregated)",
    time: 1785928979602,
    count: 4,
    note: "GATE 1 (DATA) only — no signal validated yet (fixture).",
    records: [
      { t: "org_week", key: "2026-07-27|mongodb", ticker: "MDB", org: "mongodb", weekStart: "2026-07-27", weekEnd: "2026-08-02", mergedPRs: 41, commits: 118, uniqueActorsSample: 37, actorSampleCapped: true, rt: "2026-08-03" },
      { t: "org_week", key: "2026-07-27|PagerDuty", ticker: "PD", org: "PagerDuty", weekStart: "2026-07-27", weekEnd: "2026-08-02", mergedPRs: 12, commits: 29, uniqueActorsSample: 9, actorSampleCapped: false, rt: "2026-08-03" },
      { t: "org_week", key: "2026-07-27|jfrog", ticker: "FROG", org: "jfrog", weekStart: "2026-07-27", weekEnd: "2026-08-02", mergedPRs: 6, commits: null, uniqueActorsSample: null, actorSampleCapped: false, rt: "2026-08-03" },
      { t: "org_week", key: "2026-07-20|mongodb", ticker: "MDB", org: "mongodb", weekStart: "2026-07-20", weekEnd: "2026-07-26", mergedPRs: 35, commits: 102, uniqueActorsSample: 33, actorSampleCapped: true, rt: "2026-07-27" },
    ],
  },
  // Cboe VIX term structure (2026-08-07, RAW display).
  "/api/data/vix-term-structure": {
    kind: "raw",
    source: "Cboe Global Markets — VIX1D/VIX9D/VIX/VIX3M/VIX6M/VVIX daily close (fixture)",
    attribution: "Cboe Global Markets volatility indices",
    time: 1786102149033,
    latest: {
      date: "2026-08-06", vix1d: 12.55, vix9d: 12.66, vix: 15.15, vix3m: 18.69, vix6m: 20.91, vvix: 88.72,
      vix_vix3m_ratio: 0.8106, vix9d_vix_ratio: 0.8356,
    },
    recent: [
      { date: "2026-08-05", vix1d: 13.02, vix9d: 13.4, vix: 15.81, vix3m: 19.02, vix6m: 21.2, vvix: 89.5, vix_vix3m_ratio: 0.8313, vix9d_vix_ratio: 0.8477 },
      { date: "2026-08-06", vix1d: 12.55, vix9d: 12.66, vix: 15.15, vix3m: 18.69, vix6m: 20.91, vvix: 88.72, vix_vix3m_ratio: 0.8106, vix9d_vix_ratio: 0.8356 },
    ],
    note: "GATE 1 (DATA) only — ratios are plain arithmetic, no predictive claim (fixture).",
  },
  "/api/data/eu-macro": {
    kind: "raw",
    source: "European macro cluster — ECB Data Portal + Eurostat + Deutsche Bundesbank (fixture)",
    attribution: "per-series (each series carries its required attribution string)",
    time: 1786102149033,
    note: "REGIME INPUT feed (never a direct signal). Sources revise in place (fixture).",
    series: [
      { key: "ECB_EURUSD", source: "ecb", label: "EUR/USD reference rate", cadence: "daily", unit: "USD", attribution: "Source: ECB statistics.",
        latest: { d: "2026-08-06", v: 1.0852 }, prev: { d: "2026-08-05", v: 1.0839 },
        history: [{ d: "2026-08-04", v: 1.0821 }, { d: "2026-08-05", v: 1.0839 }, { d: "2026-08-06", v: 1.0852 }] },
      { key: "ECB_ESTR", source: "ecb", label: "€STR (euro short-term rate)", cadence: "daily", unit: "%", attribution: "Source: ECB statistics.",
        latest: { d: "2026-08-06", v: 2.9 }, prev: { d: "2026-08-05", v: 2.9 },
        history: [{ d: "2026-08-04", v: 2.9 }, { d: "2026-08-05", v: 2.9 }, { d: "2026-08-06", v: 2.9 }] },
      { key: "ECB_BS_TOTAL", source: "ecb", label: "Eurosystem balance sheet (total assets)", cadence: "weekly", unit: "EUR millions", attribution: "Source: ECB statistics.",
        latest: { d: "2026-08-01", v: 6412000 }, prev: { d: "2026-07-25", v: 6398000 },
        history: [{ d: "2026-07-18", v: 6385000 }, { d: "2026-07-25", v: 6398000 }, { d: "2026-08-01", v: 6412000 }] },
      { key: "EU_INDPROD", source: "eurostat", label: "EA20 industrial production (SCA, 2021=100)", cadence: "monthly", unit: "index", attribution: "Source: Eurostat, sts_inpr_m",
        latest: { d: "2026-06-01", v: 101.3 }, prev: { d: "2026-05-01", v: 100.9 },
        history: [{ d: "2026-04-01", v: 100.4 }, { d: "2026-05-01", v: 100.9 }, { d: "2026-06-01", v: 101.3 }] },
      { key: "DE_BUND10Y", source: "bbk", label: "10Y Bund yield (Svensson, listed Federal securities)", cadence: "daily", unit: "%", attribution: "Source: Deutsche Bundesbank",
        latest: { d: "2026-08-06", v: 2.42 }, prev: { d: "2026-08-05", v: 2.4 },
        history: [{ d: "2026-08-04", v: 2.38 }, { d: "2026-08-05", v: 2.4 }, { d: "2026-08-06", v: 2.42 }] },
    ],
  },
  // EU power markets — ENTSO-E load/generation-mix/day-ahead-price, 8
  // bidding zones (2026-08-16, RAW display, euPower.tsx). Fixture covers 2
  // zones so the harness exercises the tile grid, the genmix zone picker,
  // and a negative-price point.
  "/api/data/eu-load": {
    kind: "raw",
    source: "ENTSO-E Transparency Platform (actual total load, A65/A16) (fixture)",
    attribution: "ENTSO-E Transparency Platform",
    time: "2026-08-16T12:00",
    count: 2,
    note: "realised total load in MW per bidding zone, ~1-2h publication lag (fixture).",
    zones: [
      { zone: "DE_LU", latest_ts: "2026-08-16T11:00", latest_mw: 58210, resolution: "PT15M", points_in_window: 192,
        window_min_mw: 51200, window_max_mw: 61840, window_mean_mw: 56430 },
      { zone: "FR", latest_ts: "2026-08-16T11:00", latest_mw: 47120, resolution: "PT15M", points_in_window: 192,
        window_min_mw: 41300, window_max_mw: 52900, window_mean_mw: 46810 },
    ],
    issues: {},
  },
  "/api/data/eu-generation-mix": {
    kind: "raw",
    source: "ENTSO-E Transparency Platform (actual generation per type, A75/A16) (fixture)",
    attribution: "ENTSO-E Transparency Platform",
    time: "2026-08-16T12:00",
    count: 4,
    note: "realised generation in MW per bidding zone x fuel/technology type (fixture).",
    zones: [
      { zone: "DE_LU", psr: "B16", psr_name: "Solar", latest_ts: "2026-08-16T11:00", latest_mw: 18400, resolution: "PT15M", points_in_window: 192,
        window_min_mw: 0, window_max_mw: 24100, window_mean_mw: 9800 },
      { zone: "DE_LU", psr: "B19", psr_name: "Wind Onshore", latest_ts: "2026-08-16T11:00", latest_mw: 12100, resolution: "PT15M", points_in_window: 192,
        window_min_mw: 3200, window_max_mw: 19800, window_mean_mw: 11400 },
      { zone: "FR", psr: "B14", psr_name: "Nuclear", latest_ts: "2026-08-16T11:00", latest_mw: 31200, resolution: "PT15M", points_in_window: 192,
        window_min_mw: 29800, window_max_mw: 32100, window_mean_mw: 30900 },
      { zone: "FR", psr: "B12", psr_name: "Hydro Water Reservoir", latest_ts: "2026-08-16T11:00", latest_mw: 4100, resolution: "PT15M", points_in_window: 192,
        window_min_mw: 1800, window_max_mw: 6200, window_mean_mw: 3900 },
    ],
    issues: {},
  },
  "/api/data/eu-day-ahead-prices": {
    kind: "raw",
    source: "ENTSO-E Transparency Platform (day-ahead auction clearing price, A44) (fixture)",
    attribution: "ENTSO-E Transparency Platform",
    time: "2026-08-16T12:00",
    count: 2,
    note: "day-ahead auction clearing price per bidding zone, currency/unit as published; negative prices are real (fixture).",
    zones: [
      { zone: "DE_LU", latest_ts: "2026-08-16T13:00", latest_price: 62.4, currency: "EUR", unit: "MWH", resolution: "PT60M",
        points_in_window: 72, window_min_price: -8.2, window_max_price: 118.5, window_mean_price: 54.1, negative_price_points: 3 },
      { zone: "FR", latest_ts: "2026-08-16T13:00", latest_price: 58.9, currency: "EUR", unit: "MWH", resolution: "PT60M",
        points_in_window: 72, window_min_price: 12.1, window_max_price: 104.2, window_mean_price: 51.7, negative_price_points: 0 },
    ],
    issues: {},
  },
  // SEC CNS fails-to-deliver, half-month files (2026-08-16, RAW display,
  // secFtd.tsx). Fixture covers 2 top-fail rows so the harness exercises
  // the leaderboard table and the null-price ("." source value) branch.
  "/api/data/ftd": {
    kind: "raw",
    source: "SEC CNS fails-to-deliver, half-month files (public domain) (fixture)",
    attribution: "U.S. Securities and Exchange Commission (CNS fails-to-deliver)",
    time: "2026-08-16T12:00:00.000Z",
    note: "aggregate net fail BALANCES per settlement date (a level, not a daily flow); published on a 2.5-4.5 week lag (fixture).",
    summary: {
      period: "202607b",
      settlement_dates: 11,
      newest_date: "2026-07-31",
      rows: 58328,
      top_fails: [
        { symbol: "JUCY", name: "FIXTURE JUICY CORP", qty: 4823110, price: 3.42 },
        { symbol: "SNDQ", name: "FIXTURE SANDQUEST INC", qty: 2110540, price: null },
      ],
      qty_floor: 100000,
      top_cap: 15,
    },
  },
  // US macro regime cluster — FRED, 28 public series (2026-08-08, RAW
  // display). Fixture covers one series per category so the harness
  // exercises every fmtVal unit branch (%, index, claims, thousands,
  // $M/$B, $/bbl) and the category-header grouping.
  "/api/data/macro": {
    kind: "raw",
    source: "FRED API — Source: FRED, Federal Reserve Bank of St. Louis (fixture)",
    attribution: "Source: FRED, Federal Reserve Bank of St. Louis",
    time: 1786207476120,
    note: "Values as currently published by FRED (revisions overwrite here; our archive keeps every vintage as-seen).",
    series: [
      { id: "DGS10", label: "10-Year Treasury", cadence: "daily", unit: "%",
        latest: { d: "2026-08-06", v: 4.69 }, prev: { d: "2026-08-05", v: 4.63 },
        history: [{ d: "2026-08-04", v: 4.61 }, { d: "2026-08-05", v: 4.63 }, { d: "2026-08-06", v: 4.69 }] },
      { id: "STLFSI4", label: "St. Louis Fed Stress Index", cadence: "weekly", unit: "index",
        latest: { d: "2026-07-31", v: -0.5063 }, prev: { d: "2026-07-24", v: -0.481 },
        history: [{ d: "2026-07-17", v: -0.45 }, { d: "2026-07-24", v: -0.481 }, { d: "2026-07-31", v: -0.5063 }] },
      { id: "ICSA", label: "Initial Jobless Claims", cadence: "weekly", unit: "claims",
        latest: { d: "2026-08-01", v: 199000 }, prev: { d: "2026-07-25", v: 203000 },
        history: [{ d: "2026-07-18", v: 208000 }, { d: "2026-07-25", v: 203000 }, { d: "2026-08-01", v: 199000 }] },
      { id: "CPIAUCSL", label: "CPI (All Items)", cadence: "monthly", unit: "index",
        latest: { d: "2026-06-01", v: 332.568 }, prev: { d: "2026-05-01", v: 331.9 },
        history: [{ d: "2026-04-01", v: 331.2 }, { d: "2026-05-01", v: 331.9 }, { d: "2026-06-01", v: 332.568 }] },
      { id: "HOUST", label: "Housing Starts", cadence: "monthly", unit: "thousands",
        latest: { d: "2026-06-01", v: 1427 }, prev: { d: "2026-05-01", v: 1389 },
        history: [{ d: "2026-04-01", v: 1352 }, { d: "2026-05-01", v: 1389 }, { d: "2026-06-01", v: 1427 }] },
      { id: "M2SL", label: "M2 Money Stock", cadence: "monthly", unit: "$B",
        latest: { d: "2026-06-01", v: 23155.2 }, prev: { d: "2026-05-01", v: 23081.4 },
        history: [{ d: "2026-04-01", v: 23002.1 }, { d: "2026-05-01", v: 23081.4 }, { d: "2026-06-01", v: 23155.2 }] },
      { id: "WALCL", label: "Fed Balance Sheet", cadence: "weekly", unit: "$M",
        latest: { d: "2026-08-05", v: 6748567 }, prev: { d: "2026-07-29", v: 6739210 },
        history: [{ d: "2026-07-22", v: 6731500 }, { d: "2026-07-29", v: 6739210 }, { d: "2026-08-05", v: 6748567 }] },
      { id: "DCOILWTICO", label: "WTI Crude Spot", cadence: "daily", unit: "$/bbl",
        latest: { d: "2026-08-03", v: 81.96 }, prev: { d: "2026-07-31", v: 80.4 },
        history: [{ d: "2026-07-29", v: 79.8 }, { d: "2026-07-31", v: 80.4 }, { d: "2026-08-03", v: 81.96 }] },
    ],
  },
  // Institutional 13F-HR holdings (2026-08-08, RAW display). Two filings:
  // one focused (holdings present) and one over the FOCUSED_MAX_HOLDINGS
  // cap (summary-only) so the harness exercises both card states.
  "/api/data/filings13f/history": {
    kind: "raw",
    source: "SEC EDGAR (13F-HR) — accumulated archive (began 2026-07-05) (fixture)",
    days: 45,
    count: 2,
    focused_cap: 250,
    filings: [
      {
        accession: "0001234567-26-000123", filedAt: "2026-08-05T00:00:00.000Z", cik: "0001234567",
        indexUrl: "https://www.sec.gov/Archives/edgar/data/1234567/000123456726000123/",
        managerCik: "0001234567", managerName: "Example Capital Management LLC",
        periodOfReport: "2026-06-30", submissionType: "13F-HR",
        entryTotal: 42, valueTotal: 812500000, otherManagersCount: 0,
        holdingsOmitted: false,
        holdings: [
          { issuer: "EXAMPLE SMALLCAP CORP", titleOfClass: "COM", cusip: "123456789", value: 45000000, shares: 900000, sharesType: "SH", putCall: null, discretion: "SOLE" },
          { issuer: "SAMPLE MICROCAP INC", titleOfClass: "COM", cusip: "987654321", value: 22000000, shares: 500000, sharesType: "SH", putCall: null, discretion: "SOLE" },
        ],
      },
      {
        accession: "0009876543-26-000456", filedAt: "2026-08-04T00:00:00.000Z", cik: "0009876543",
        indexUrl: "https://www.sec.gov/Archives/edgar/data/9876543/000987654326000456/",
        managerCik: "0009876543", managerName: "Mega Institutional Advisors Inc",
        periodOfReport: "2026-06-30", submissionType: "13F-HR",
        entryTotal: 3800, valueTotal: 412000000000, otherManagersCount: 2,
        holdingsOmitted: true, holdings: null,
      },
    ],
  },
  // Data quality dashboard (MAP V2 ROADMAP R6(b), 2026-07-30) — one entry
  // per health bucket + a multi-kind archive so the bar-chart rendering
  // path is exercised at 390px.
  "/api/data/quality-dashboard": {
    generated_at: "2026-07-30T00:00:00.000Z",
    streams_warming_up: false,
    streams: { total: 42, live: 30, recent: 8, stale: 3, no_data: 1 },
    archive: {
      total_bytes: 4831838208,
      total_files: 5120,
      kinds: [
        { kind: "aircraft", files: 2100, bytes: 2147483648 },
        { kind: "vessels", files: 1800, bytes: 1073741824 },
        { kind: "filings", files: 620, bytes: 536870912 },
        { kind: "fires", files: 300, bytes: 268435456 },
        { kind: "trains", files: 210, bytes: 134217728 },
        { kind: "earthquakes", files: 90, bytes: 67108864 },
      ],
    },
    verification: {
      sites: { verified: 16, total: 16, method: "Esri World Imagery crosshair check, scripts/site_verify.py (2026-07-03) (fixture)" },
      ports: { verified: 9, total: 9, method: "subset of the imagery-verified strategic sites (category = port) (fixture)" },
      plants: { verified: 100, total: 9833, method: "top-100-by-MW plants imagery-verified against Esri World Imagery (fixture)" },
    },
  },
  // Signal-strength dashboard (MAP V2 ROADMAP R6(a), 2026-07-30) — one root
  // per status class so the funnel bars, category chips, and badge colors
  // all render at 390px.
  "/api/data/signal-ladder": {
    generated_at: "2026-07-30T00:00:00.000Z",
    compiled: "2026-07-30",
    sources: ["research/experiments.md (fixture)", "research/open_questions.md (fixture)"],
    roots: [
      { id: "fx_raw", name: "Fixture raw overlay", category: "environmental", status: "raw_only", current_gate: 0, last_update_date: "2026-07-08", note: "Display-only overlay, no predictive claim (fixture).", source_ref: "experiments.md:1 (fixture)" },
      { id: "fx_g1p", name: "Fixture gate-1 pending", category: "macro", status: "gate1_pending", current_gate: 0, last_update_date: "2026-07-06", note: "Archiver shipped, gate-1 criteria stated, no run result yet (fixture).", source_ref: "experiments.md:2 (fixture)" },
      { id: "fx_g1pass", name: "Fixture gate-1 pass", category: "macro", status: "gate1_pass", current_gate: 1, last_update_date: "2026-07-05", note: "Prod values exact-matched the published export (fixture).", source_ref: "experiments.md:3 (fixture)" },
      { id: "fx_g1fail", name: "Fixture gate-1 fail", category: "commodities", status: "gate1_fail", current_gate: 1, last_update_date: "2026-07-05", note: "Both sensor designs dead at gate 1 (fixture).", source_ref: "experiments.md:4 (fixture)" },
      { id: "fx_g2p", name: "Fixture gate-2 pending", category: "government_spending", status: "gate2_pending", current_gate: 2, last_update_date: "2026-07-26", note: "Gate 1 passed, first gate-2 run inconclusive (fixture).", source_ref: "experiments.md:5 (fixture)" },
      { id: "fx_killed", name: "Fixture killed root", category: "insider_trading", status: "killed", current_gate: 2, last_update_date: "2026-07-22", note: "Gate 2 kill in both directions (fixture).", source_ref: "experiments.md:6 (fixture)" },
      // gate2_pass + detail_route (2026-08-16) — exercises the "view live
      // signal" generic link (signalLadder.tsx), first real use being
      // gnss_integrity_adsb's #/data/gnss-integrity page.
      { id: "fx_g2pass_detail", name: "Fixture gate-2 pass with live detail page", category: "geopolitical_intelligence", status: "gate2_pass", current_gate: 2, last_update_date: "2026-08-16", note: "Gate 2 pass with a dedicated live signal page (fixture).", source_ref: "experiments.md:7 (fixture)", detail_route: "#/data/gnss-integrity" },
    ],
    summary: {
      total: 7,
      by_status: { raw_only: 1, gate1_pending: 1, gate1_pass: 1, gate1_fail: 1, gate2_pending: 1, killed: 1, gate2_pass: 1 },
      by_category: { environmental: 1, macro: 2, commodities: 1, government_spending: 1, insider_trading: 1, geopolitical_intelligence: 1 },
      gate_counts: [{ gate: 0, count: 2 }, { gate: 1, count: 2 }, { gate: 2, count: 3 }, { gate: 3, count: 0 }, { gate: 4, count: 0 }, { gate: 5, count: 0 }],
      killed_count: 1,
      raw_only_count: 1,
      furthest_gate_reached: 2,
    },
  },
  // Pipeline-health dashboard (MAP V2 ROADMAP R6(c), 2026-07-31) — a mixed
  // 24h timeline (mostly ok, a few degraded bars) plus non-zero per-check
  // degraded counts so both the uptime tiles and the "what's degrading"
  // bar-chart rendering paths are exercised at 390px.
  "/api/data/pipeline-health-dashboard": {
    generated_at: "2026-07-31T00:00:00.000Z",
    warming_up: false,
    window_24h: {
      window_hours: 24, sample_count: 288, uptime_pct: 97.6,
      current: {
        t: 1783200000, status: "ok", database_ok: true, alpaca_ok: true, python_ok: true,
        scanner_ok: true, scanner_consecutive_failures: 0, licensing_ok: true,
        bot_status: "active", bot_liveness_dark: false,
      },
      degraded_counts: { database: 0, alpaca: 2, python: 0, scanner: 5, licensing: 0, bot_liveness_dark: 0 },
      timeline: Array.from({ length: 48 }, (_, i) => ({ t: 1783200000 - (47 - i) * 1800, status: i === 10 || i === 11 || i === 30 ? "degraded" : "ok" })),
    },
    window_7d: {
      window_hours: 168, sample_count: 2016, uptime_pct: 99.1,
      current: {
        t: 1783200000, status: "ok", database_ok: true, alpaca_ok: true, python_ok: true,
        scanner_ok: true, scanner_consecutive_failures: 0, licensing_ok: true,
        bot_status: "active", bot_liveness_dark: false,
      },
      degraded_counts: { database: 0, alpaca: 3, python: 0, scanner: 15, licensing: 0, bot_liveness_dark: 1 },
      timeline: Array.from({ length: 100 }, (_, i) => ({ t: 1783200000 - (99 - i) * 6048, status: i % 33 === 0 ? "degraded" : "ok" })),
    },
  },
  // Streams inventory (Phase 4) — one row per health state so the card
  // battery exercises every chip/peek/no-data rendering path at 390px.
  "/api/data/streams": {
    time: 1, count: 4,
    streams: [
      {
        stream: "occvolume", source: "OCC volume-query (marketdata.theocc.com) — keyless CSV (fixture)", attribution: "The Options Clearing Corporation (OCC) daily volume",
        license: "OCC informational use", started: "2026-07-06", cadence: "daily source, published overnight ET; 4h poll",
        hypothesis: "customer-vs-market-maker put/call by ticker (gate-locked)", storage: "one gzipped JSONL per report date", written_by: "server/occVolume.ts",
        files: 3, bytes: 3145728, newest_file: "2026-07-02.jsonl.gz", age_hours: 6.2, records_newest_file: 153181,
        latest_peek: '{"date":"2026-07-02","underlying":"SPY","symbol":"SPY","actype":"M","porc":"P","exchange":"CBOE","qty":120}', peek_note: null,
        health: "live", health_note: "newest file 6.2h old",
      },
      {
        stream: "cropconditions", source: "USDA NASS QuickStats weekly crop condition ratings (fixture)", attribution: "USDA National Agricultural Statistics Service",
        license: "US government data", started: "2026-07-06", cadence: "weekly, Mondays 4pm ET",
        hypothesis: "condition-change vs grain price moves (gate 1 pending)", storage: "one JSONL per observation week", written_by: "server/cropConditions.ts",
        files: 1, bytes: 4096, newest_file: "2026-07-05.jsonl", age_hours: 96.0, records_newest_file: 10,
        latest_peek: '{"week":"2026-07-05","commodity":"CORN","item":"GOOD","value":41}', peek_note: null,
        health: "recent", health_note: "newest file 96.0h old (within cadence ~240h)",
      },
      {
        stream: "railep724", source: "STB EP724 weekly rail performance (fixture)", attribution: "Surface Transportation Board",
        license: "US government data", started: "2026-07-05", cadence: "weekly, session-run pipeline",
        hypothesis: "carload trends lead reported volumes", storage: "weekly JSONL artifacts", written_by: "scripts/stb_rail.py (session-run)",
        files: 2, bytes: 51200, newest_file: "2026-06-20.jsonl", age_hours: 384.0, records_newest_file: 88,
        latest_peek: null, peek_note: "peek skipped: newest file 9.1MB > 8MB cap",
        health: "stale", health_note: "newest file 384.0h old — exceeds cadence-derived threshold ~240h",
      },
      {
        stream: "griddemand", source: "EIA-930 hourly demand by balancing authority (fixture)", attribution: "U.S. Energy Information Administration",
        license: "US government data", started: "2026-07-06", cadence: "hourly source, 2h poll",
        hypothesis: "regional demand anomalies vs utility/industrial names", storage: "one JSONL per observation day", written_by: "server/gridDemand.ts",
        files: 0, bytes: 0, newest_file: null, age_hours: null, records_newest_file: null,
        latest_peek: null, peek_note: "no files",
        health: "no-data", health_note: "no archive files yet — key-gated, warming up, or first write pending",
      },
    ],
  },
  "/api/data/earnings-language": {
    kind: "raw", source: "SEC EDGAR (8-K Item 2.02 / Exhibit 99)", time: 1, count: 1,
    filings: [
      {
        accession: "0001-26-9", cik: "9", companyName: "FIXTURE INDUSTRIES INC",
        filedAt: "2026-07-05", itemCodes: ["2.02"],
        indexUrl: "https://www.sec.gov/x/9/", exhibitUrl: "https://www.sec.gov/x/9/ex99.htm",
        text: "Fixture Industries today reported second-quarter results ahead of guidance.", textLength: 76, truncated: false,
      },
    ],
  },
  "/api/data/earnings-language/history": {
    kind: "raw", source: "SEC EDGAR (8-K Item 2.02 / Exhibit 99) — accumulated archive", days: 30, count: 2,
    filings: [
      {
        accession: "0001-26-9", cik: "9", companyName: "FIXTURE INDUSTRIES INC",
        filedAt: "2026-07-05", itemCodes: ["2.02"],
        indexUrl: "https://www.sec.gov/x/9/", exhibitUrl: "https://www.sec.gov/x/9/ex99.htm",
        text: "Fixture Industries today reported second-quarter results ahead of guidance.", textLength: 76, truncated: false,
      },
      {
        accession: "0001-26-10", cik: "10", companyName: "SAMPLE HOLDINGS CORP",
        filedAt: "2026-07-04", itemCodes: ["2.02", "9.01"],
        indexUrl: "https://www.sec.gov/x/10/", exhibitUrl: "https://www.sec.gov/x/10/ex99.htm",
        text: "Sample Holdings Corp announced full-year guidance and a share buyback program.", textLength: 79, truncated: false,
      },
    ],
  },
  "/api/data/short-volume": {
    kind: "raw", source: "FINRA Reg SHO daily short sale volume (CNMS consolidated file)", time: 1,
    summary: {
      date: "2026-07-02", symbols: 12240, agg_short_ratio: 0.4633,
      top_ratio: [
        { symbol: "JUCY", short_ratio: 0.9964, total_vol: 911633, market: "Q" },
        { symbol: "AGGH", short_ratio: 0.9909, total_vol: 1968769, market: "Q,N" },
      ],
      floor_total_vol: 500000, top_cap: 30,
    },
  },
  "/api/data/short-volume/history": {
    kind: "raw", source: "FINRA Reg SHO daily short sale volume (CNMS consolidated file)", days: 60,
    today: {
      date: "2026-07-02", symbols: 12240, agg_short_ratio: 0.4633,
      top_ratio: [{ symbol: "JUCY", short_ratio: 0.9964, total_vol: 911633, market: "Q" }],
      floor_total_vol: 500000, top_cap: 30,
    },
    trend: [
      { date: "2026-07-01", symbols: 12200, agg_short_ratio: 0.4610 },
      { date: "2026-07-02", symbols: 12240, agg_short_ratio: 0.4633 },
    ],
  },
  // [REPAIR 2026-07-25] the 9 endpoint fixtures below back the 9 layer
  // descriptors added above — real response shapes copied from server/
  // routes.ts's actual handlers, one point each (enough to exercise the
  // symbol layer + legend-parity + detail card, not a rendering-scale test).
  "/api/data/pfas": {
    kind: "raw", predictive: false, source: "U.S. EPA UCMR 5 (fixture)",
    note: "fixture", geocode_source: "EPA service-area centroid", built_at: "2026-07-01T00:00:00Z",
    totals: { systems: 1 }, health: { suspect: 0, freshness: "public domain" },
    systems: [
      { pwsid: "FX0000001", name: "Fixture Water System", lat: 39.1, lon: -94.6, n_analytes_detected: 2,
        population_served: 12000, detections: [{ contaminant: "PFOA", max_value: 12.4, units: "ng/L", n_detections: 3, last_detected: "2024-11-01" }] },
    ],
  },
  "/api/data/cancer-rates": {
    kind: "raw", predictive: false, source: "NCI State Cancer Profiles (fixture)",
    caveat: "COUNTY-LEVEL AGGREGATE STATISTIC ONLY (fixture)", built_at: "2026-08-11T00:00:00Z",
    gate1: { passed: true }, county_count: 2, health: { suspect: 0, freshness: "public domain" },
    geojson: {
      type: "FeatureCollection",
      features: [
        { type: "Feature", id: "20177", properties: { fips: "20177", county: "Fixture County", state: "Kansas", has_data: true, incidence_rate: 450.0, incidence_trend: "stable", mortality_rate: 145.0, mortality_trend: "falling" },
          geometry: { type: "Polygon", coordinates: [[[-95.8, 39.0], [-95.6, 39.0], [-95.6, 39.2], [-95.8, 39.2], [-95.8, 39.0]]] } },
        { type: "Feature", id: "20179", properties: { fips: "20179", county: "No-Data Fixture County", state: "Kansas", has_data: false, incidence_rate: null, incidence_trend: null, mortality_rate: null, mortality_trend: null },
          geometry: { type: "Polygon", coordinates: [[[-95.6, 39.0], [-95.4, 39.0], [-95.4, 39.2], [-95.6, 39.2], [-95.6, 39.0]]] } },
      ],
    },
  },
  "/api/data/radiation": {
    kind: "raw", predictive: false, source: "BfS · Health Canada · STUK/FMI · EPA RadNet (fixture)",
    time: 1, networks: { "bfs-de": 1, "radnet-us": 1 },
    health: { source: "fixture" },
    stations: [
      { name: "Fixture DE Station", network: "bfs-de", lat: 50.1, lon: 8.7, value: 0.12, unit: "uSv/h", time: "2026-07-25T00:00:00Z", approx: false },
      { name: "Fixture City, KS", network: "radnet-us", lat: 39.0, lon: -95.7, value: 14.2, unit: "cpm", time: "2026-07-25T00:00:00Z", approx: true, rkm: 6 },
    ],
  },
  "/api/data/nukefacilities": {
    kind: "raw", predictive: false, source: "Wikidata (CC0 1.0), curated (fixture)", count: 1,
    facilities: [{ n: "Fixture Enrichment Plant", cat: "Enrichment plant", country: "United States", qid: "Q0", lat: 43.5, lon: -112.0 }],
  },
  "/api/data/nucleartests": {
    kind: "raw", predictive: false, source: "SIPRI / Johnston archive (fixture)", count: 1, quarantined: 0,
    tests: [{ n: "FIXTURE SHOT", c: "USA", y: 1962, d: "1962-07-09", kt: 1.4, t: "AIRDROP", p: "WE", r: "Fixture Test Site", lat: 37.2, lon: -116.0 }],
  },
  "/api/data/nukeaccidents": {
    kind: "raw", predictive: false, source: "Wikidata (CC0 1.0), curated (fixture)", count: 1,
    events: [{ n: "Fixture Reactor Incident", d: "1979-03-28", loc: "Fixture, PA", ines: 5, qid: "Q0", lat: 40.2, lon: -76.7 }],
  },
  "/api/data/methane-plumes": {
    kind: "raw", predictive: false, source: "Global Energy Monitor GMET (fixture)",
    count: 1, matchedCount: 1, ambiguousCount: 0,
    plumes: [{
      name: "Fixture plume detection", infrastructureType: "Oil and gas", country: "United States",
      observedAt: "2026-06-01T00:00:00Z", provider: "GHGSat", instrument: "GHGSat-C", emissionsKgHr: 340.5,
      lat: 31.9, lon: -102.1,
      nearestAsset: { kind: "oil_gas_extraction", id: "A1", name: "Fixture Field", distanceKm: 0.4, operator: "Fixture Operator LLC" },
    }],
  },
  "/api/data/coal-mine-features": {
    kind: "raw", predictive: false, source: "Global Energy Monitor (fixture)", count: 1, release: "fixture",
    features: [{
      mineName: "Fixture Mine", category: "ventilation", coalGrade: "thermal", country: "United States",
      owners: "Fixture Coal Co", mineId: "FX1", dataSourceDate: "2025-01-01",
      geometry: { type: "Point", coordinates: [-81.5, 37.8] },
    }],
  },
  "/api/data/airport-status": {
    kind: "raw", source: "FAA National Airspace System Status (fixture)", time: 1, update_time: "2026-07-25T00:00:00Z", count: 1,
    events: [{ airport: "ATL", type: "Ground Delay Program", reason: "Volume", avg: "31 minutes", max: "1 hour 5 minutes", min: "16 minutes", trend: "Increasing", direction: "Arrival", end_time: "18:00 UTC", reopen: null, update_time: "2026-07-25T00:00:00Z" }],
  },
  "/api/data/border-waits": {
    kind: "raw", source: "CBP Border Wait Times (fixture)", time: 1, count: 1,
    waits: [{ port_number: "010401", border: "Canada", lane: "POV", status: "Open", delay_min: 15, lanes_open: 2, update_time: "2026-07-25T00:00:00Z" }],
  },
  // Methane hotspots (gate-2(b), 2026-07-20) — descriptive stat, not a
  // signal (predictive:false; note states the honest gate-2(c)/(d) gap).
  "/api/data/methane-plumes/asset-stats": {
    kind: "raw", predictive: false,
    source: "Global Energy Monitor — Methane Emitters Tracker (GMET) × Oil & Gas Extraction Tracker / Global Coal Mine Tracker, CC BY 4.0",
    note: "Repeat satellite methane-plume detections grouped by nearest catalogued GEM asset (within 2km, ambiguous matches excluded). "
      + "A count/rate of repeat detections is NOT an emissions signal on its own — no base rate or disclosed-intensity comparison exists yet (fixture).",
    count: 2,
    assetStats: [
      {
        assetId: "A1", kind: "coal_mine", name: "Sample Mine", operator: null, owner: "Sample Coal Co", parent: "Sample Holdings",
        detectionCount: 5, firstObservedAt: "2025-01-03T00:00:00Z", lastObservedAt: "2026-06-30T00:00:00Z",
        spanDays: 543, ratePerYear: 3.36, meanEmissionsKgHr: 812.4,
      },
      {
        assetId: "A2", kind: "oil_gas_extraction", name: "Sample Field", operator: "Sample Operator LLC", owner: null, parent: null,
        detectionCount: 1, firstObservedAt: "2026-04-11T00:00:00Z", lastObservedAt: "2026-04-11T00:00:00Z",
        spanDays: null, ratePerYear: null, meanEmissionsKgHr: 214.0,
      },
    ],
  },
  "/api/data/attention": {
    kind: "raw", source: "Wikimedia pageviews API (en.wikipedia, all-access, agent=user)", time: 1,
    date: "2026-07-02", seed_size: 23, count: 2,
    tickers: [
      { ticker: "NVDA", article: "Nvidia", views: 10883 },
      { ticker: "GME", article: "GameStop", views: 4211 },
    ],
  },
  "/api/data/attention/history": {
    kind: "raw", source: "Wikimedia pageviews API (en.wikipedia, all-access, agent=user)", days: 60,
    today: {
      date: "2026-07-02", seed_size: 23, count: 2,
      tickers: [
        { ticker: "NVDA", article: "Nvidia", views: 10883 },
        { ticker: "GME", article: "GameStop", views: 4211 },
      ],
    },
    trend: [
      { date: "2026-07-01", tickers: 23, total_views: 98210 },
      { date: "2026-07-02", tickers: 23, total_views: 101422 },
    ],
  },
  // CFTC Traders in Financial Futures, futures-only (2026-08-16, RAW
  // display, tff.tsx). Fixture covers 2 markets so the harness exercises
  // the ranked table's lev-money/dealer-net columns.
  "/api/data/tff": {
    kind: "raw", source: "CFTC Traders in Financial Futures, futures-only (public domain) (fixture)",
    attribution: "CFTC Traders in Financial Futures", time: "2026-08-16T12:00:00.000Z",
    report_date: "2026-08-11", count: 2,
    note: "weekly financial-futures positioning by trader category (fixture)",
    markets: [
      { report_date: "2026-08-11", market: "E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE", code: "13874A", commodity: "STOCK INDICES",
        open_interest: 2450000, dealer_long: 210000, dealer_short: 340000, asset_mgr_long: 620000, asset_mgr_short: 410000,
        lev_money_long: 380000, lev_money_short: 710000, other_rept_long: 150000, other_rept_short: 120000 },
      { report_date: "2026-08-11", market: "10-YEAR U.S. TREASURY NOTES - CHICAGO BOARD OF TRADE", code: "043602", commodity: "TREASURY",
        open_interest: 3980000, dealer_long: 540000, dealer_short: 610000, asset_mgr_long: 890000, asset_mgr_short: 720000,
        lev_money_long: 1150000, lev_money_short: 940000, other_rept_long: 260000, other_rept_short: 210000 },
    ],
  },
  "/api/data/cot": {
    kind: "raw", source: "CFTC Commitments of Traders, disaggregated futures-only (public domain)", time: 1,
    report_date: "2026-06-23", count: 2,
    markets: [
      { code: "067651", market: "CRUDE OIL - NYMEX", commodity: "CRUDE OIL", open_interest: 428305, m_money_long: 68457, m_money_short: 138890 },
      { code: "088691", market: "GOLD - COMMODITY EXCHANGE INC.", commodity: "GOLD", open_interest: 512300, m_money_long: 91200, m_money_short: 40100 },
    ],
  },
  "/api/data/cot/history": {
    kind: "raw", source: "CFTC Commitments of Traders, disaggregated futures-only (public domain)", weeks: 52,
    today: {
      report_date: "2026-06-23",
      rows: [
        { code: "067651", market: "CRUDE OIL - NYMEX", commodity: "CRUDE OIL", open_interest: 428305, m_money_long: 68457, m_money_short: 138890 },
      ],
    },
    trend: [
      { report_date: "2026-06-16", markets: 274, total_open_interest: 18320000 },
      { report_date: "2026-06-23", markets: 274, total_open_interest: 18455000 },
    ],
  },
  "/api/data/insider/history": {
    kind: "raw", source: "SEC EDGAR (Form 4) — accumulated archive", days: 30, count: 2,
    filings: [
      {
        accession: "0001-26-1", filedAt: "2026-07-03", indexUrl: "https://www.sec.gov/x/1/",
        issuerName: "CYPHERPUNK TECHNOLOGIES INC.", issuerTradingSymbol: "CYPH",
        owners: [{ cik: "1", name: "Richard Christian M", isDirector: true, isOfficer: false, isTenPercentOwner: false, officerTitle: null }],
        transactions: [{ table: "derivative", kind: "award_grant", shares: 75000, pricePerShare: 0, transactionDate: "2026-07-01" }],
      },
      {
        accession: "0001-26-2", filedAt: "2026-07-02", indexUrl: "https://www.sec.gov/x/2/",
        issuerName: "STRATUS PROPERTIES INC", issuerTradingSymbol: "STRS",
        owners: [{ cik: "2", name: "Oasis Management Co Ltd.", isDirector: false, isOfficer: false, isTenPercentOwner: true, officerTitle: null }],
        transactions: [{ table: "nonDerivative", kind: "open_market_sale", shares: 10000, pricePerShare: 28.9, transactionDate: "2026-06-30" }],
      },
    ],
  },
  "/api/data/trains": {
    kind: "raw", source: "Digitraffic Finland (CC BY 4.0) + Entur Norway (NLOD)",
    time: 1, coverage: "FI + NO (launch)", count: 3,
    sources: [
      { key: "digitraffic", country: "FI", status: "ok", count: 2 },
      { key: "entur", country: "NO", status: "ok", count: 1 },
    ],
    trains: [
      { id: "FI-62-2026-07-04", country: "FI", lat: 63.7632, lon: 27.3121, speed_kmh: 139, bearing: null, label: "Train 62", ts: 1 },
      { id: "FI-104-2026-07-04", country: "FI", lat: 62.5036, lon: 29.8569, speed_kmh: 129, bearing: null, label: "Train 104", ts: 1 },
      { id: "NO-71-12", country: "NO", lat: 59.741, lon: 10.2016, speed_kmh: 108, bearing: 45, label: "FLY1", ts: 1 },
    ],
  },
  // The real compiled dataset (repo file, deterministic): ~9.8k plants
  // exercises the clustering path at full production scale in the perf
  // window, on top of the 10k-aircraft budget check.
  "/api/data/powerplants": {
    kind: "raw",
    ...JSON.parse(readFileSync(path.join(ROOT, "datacore", "powerplants", "us_power_plants.json"), "utf8")),
  },
  // NRC daily reactor status full view (2026-08-09) — one plant per status
  // tier (full/reduced/outage/unknown) so the filter chips, worst-first
  // sort, and per-unit expand-on-click all get exercised at 390px; a
  // multi-unit plant (Beaver Valley) exercises the units-breakdown table.
  "/api/data/nrc-reactor-status": {
    kind: "raw",
    source: "U.S. NRC Power Reactor Status Reports (public domain, US government data) (fixture)",
    attribution: "U.S. Nuclear Regulatory Commission (NRC)",
    time: 1, date: "2026-08-09", count: 5,
    note: "daily percent-of-rated-thermal-power per operating unit (NRC unit granularity, not plant); outage-adjacent signals stay gate-locked until ladder validation",
    rows: [],
    plantCount: 4,
    plants: [
      { idx: 1, name: "Beaver Valley", lat: 40.622, lon: -80.434, mw: 1836, owner: "Energy Harbor",
        units: [{ unit: "Beaver Valley 1", power: 100 }, { unit: "Beaver Valley 2", power: 100 }],
        avgPower: 100, status: "full" },
      { idx: 2, name: "Palo Verde", lat: 33.3881, lon: -112.8646, mw: 3937, owner: "Arizona Public Service",
        units: [{ unit: "Palo Verde 1", power: 55 }],
        avgPower: 55, status: "reduced" },
      { idx: 3, name: "Diablo Canyon", lat: 35.2116, lon: -120.8551, mw: 2256, owner: "PG&E",
        units: [{ unit: "Diablo Canyon 1", power: 0 }],
        avgPower: 0, status: "outage" },
      { idx: 4, name: "Waterford 3", lat: 29.9942, lon: -90.4696, mw: 1250, owner: "Entergy",
        units: [{ unit: "Waterford 3", power: null }],
        avgPower: null, status: "unknown" },
    ],
  },
  // GNSS integrity anomaly signal (2026-08-16) — the first live gate2_pass
  // SIGNAL detail page. Fixture mirrors the real shape from
  // server/gnssIntegritySignal.ts's computeGnssIntegritySignal(), with one
  // band of each elevated/not-elevated state so both table badge colors
  // render at every canonical width.
  "/api/data/gnss-integrity-signal": {
    kind: "signal", root_id: "gnss_integrity_adsb", generated_at: "2026-08-16T00:00:00.000Z",
    gate: { current_gate: 2, status: "gate2_pass" }, verdict: "PASS",
    bands: [
      { band: "cruise", candidate_k: 15, candidate_n: 414, control_rate: 0.00147, expected_under_null: 0.61, p_value: 0.000001, elevated: true, expected_to_elevate: true },
      { band: "mid", candidate_k: 12, candidate_n: 277, control_rate: 0.00272, expected_under_null: 0.75, p_value: 0.000001, elevated: true, expected_to_elevate: true },
      { band: "low", candidate_k: 7, candidate_n: 903, control_rate: 0.02214, expected_under_null: 20.0, p_value: 0.46, elevated: false, expected_to_elevate: false },
    ],
    region: {
      candidate_label: "Baltic corridor (Gdańsk–Bornholm–Gotland)", candidate_bbox: [53, 60, 17, 24],
      control_label: "NY + Paris control region", control_bbox: [35, 55, -80, 10],
    },
    freshness: {
      writer_live_since: "2026-08-11",
      candidate: { days_read: ["2026-08-11", "2026-08-12", "2026-08-13", "2026-08-14"], days_missing: [], rows_scanned: 1613, truncated: false },
      control: { days_read: ["2026-08-11", "2026-08-12", "2026-08-13", "2026-08-14"], days_missing: [], rows_scanned: 32944, truncated: false },
    },
    methodology_note: "Broadcast-origin ADS-B rows only. One-tailed exact binomial test per altitude band (fixture text).",
    caveats: [
      "GATE 1 is PARTIAL, not a full pass — DTU Space's Bornholm RF station corroborates the phenomenon/region, not the exact sample dates (fixture text).",
      "Small, growing sample — the archive has carried integrity fields only since 2026-08-11 (fixture text).",
      "Not tradeable — this is GATE 2 statistical discrimination, not a trading signal (fixture text).",
    ],
    license: { source: "adsb.lol (ODbL 1.0) + adsb.fi/airplanes.live (non-commercial fallbacks)", note: "Fixture license note." },
  },
};

const MIME = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".png": "image/png", ".svg": "image/svg+xml", ".json": "application/json", ".woff2": "font/woff2", ".ico": "image/x-icon" };

// Deterministic weather tile standing in for the proxy's OUTPUT (alpha-
// amplified, per owmTiles.ts): a colored gradient at the strength production
// serves. The amplification itself is unit-tested against a real captured
// prod tile; this fixture exercises the CLIENT half — mount, blend, opacity.
const WX_TILE_PNG = (() => {
  const png = new PNG({ width: 256, height: 256 });
  for (let y = 0; y < 256; y++) {
    for (let x = 0; x < 256; x++) {
      const i = (y * 256 + x) * 4;
      png.data[i] = Math.round(40 + x * 0.8);
      png.data[i + 1] = 60;
      png.data[i + 2] = Math.round(200 - y * 0.6);
      png.data[i + 3] = 190; // amplified-output strength (cap 230 upstream)
    }
  }
  return PNG.sync.write(png);
})();

// Mean per-channel pixel difference between two PNG buffers — the same
// metric verify_weather_prod.mjs uses for the prod pixel proof.
function pngMeanDiff(a, b) {
  const A = PNG.sync.read(a), B = PNG.sync.read(b);
  if (A.width !== B.width || A.height !== B.height) return 255;
  let sum = 0; const n = A.width * A.height;
  for (let i = 0; i < n * 4; i += 4) {
    sum += Math.abs(A.data[i] - B.data[i]) + Math.abs(A.data[i + 1] - B.data[i + 1]) + Math.abs(A.data[i + 2] - B.data[i + 2]);
  }
  return sum / (n * 3);
}

// Trail-refresh ratchet state: counts /api/data/track/* hits so the battery
// can assert the popup re-pulls while open (module scope — batteries read it).
let trackCalls = 0;

function startServer() {
  return new Promise((resolve) => {
    const srv = createServer((req, res) => {
      const [u, qs] = (req.url || "/").split("?");
      if (u.startsWith("/api/data/wxtile/")) {
        res.writeHead(200, { "content-type": "image/png" });
        return res.end(WX_TILE_PNG);
      }
      // TIME MACHINE v2 T-2 window fixture (2026-08-13): checked BEFORE the
      // generic FIXTURES prefix match below, which would otherwise treat
      // "/api/data/aircraft/window" as a sub-path of the "/api/data/aircraft"
      // key (u.startsWith(k + "/")) and hand TimeScrubber.tsx the raw
      // /api/data/aircraft point-list shape instead of a WindowResult — no
      // `.hexes`, no `.to`, a real crash the first live run of this harness
      // caught (`w.hexes is not iterable`). A dedicated small window payload
      // exercises the real hex/points/coverage shape the client reads.
      if (u.startsWith("/api/data/aircraft/window")) {
        const params = new URLSearchParams(qs || "");
        const to = parseInt(params.get("to") || "0", 10) || Math.floor(Date.now() / 1000);
        const from = parseInt(params.get("from") || String(to - 86400), 10) || to - 86400;
        const step = parseInt(params.get("step") || "300", 10) || 300;
        let seed = 7;
        const rnd = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
        const hexes = [];
        for (let i = 0; i < 40; i++) {
          const baseLat = 25 + rnd() * 24, baseLon = -125 + rnd() * 58;
          const n = 5 + Math.floor(rnd() * 8);
          const points = [];
          for (let k = 0; k < n; k++) {
            const t = from + Math.floor(((to - from) * k) / Math.max(1, n - 1));
            points.push([t, baseLat + k * 0.05, baseLon + k * 0.05, Math.round(rnd() * 11000)]);
          }
          hexes.push({ i: "fx" + i.toString(16), rg: "N" + i + "FX", c: "TST" + i, ty: "B738", points, raw_count: points.length, truncated: false });
        }
        res.writeHead(200, { "content-type": "application/json" });
        return res.end(JSON.stringify({
          kind: "aircraft", from, to, zoom: 5, step_sec: step,
          hexes, hexes_seen: hexes.length,
          total_points: hexes.reduce((s, h) => s + h.points.length, 0),
          coverage: { requested_from: from, scanned_from: from, complete: true, files_scanned: 1 },
        }));
      }
      // STATEFUL track fixture (trail-refresh ratchet, [REPAIR 2026-07-05]):
      // each call returns one MORE point so a live-refreshing trail visibly
      // grows across the popup's 30s interval; a static-snapshot regression
      // shows identical coordinates on every read.
      // 2026-07-20 (flight-track handoff): points now carry `al` altitudes
      // (climb profile) so the curtain + altitude line + ALTITUDE/TIME
      // profile panel actually render under the harness — an altitude-less
      // fixture exercised none of the 3D track path.
      if (u.startsWith("/api/data/track/")) {
        trackCalls++;
        const nowSec = Math.floor(Date.now() / 1000);
        const points = [];
        for (let k = 0; k <= 2 + trackCalls; k++) {
          points.push({
            t: nowSec - (2 + trackCalls - k) * 60,
            la: 36 + k * 0.05, lo: -97 + k * 0.05,
            al: 1500 + k * 400, // steady climb — exercises the ramp min→max
          });
        }
        res.writeHead(200, { "content-type": "application/json" });
        return res.end(JSON.stringify({ kind: "aircraft", id: u.split("/").pop(), points, count: points.length }));
      }
      // /developers live-query widget (graph entity/hops "try it" input):
      // exercise the real ?entity= neighborhood path instead of always
      // falling through to the counts-only FIXTURES["/api/data/graph"]
      // shape, so the harness screenshot actually shows the widget's real
      // result state, not just its idle/counts state.
      if (u === "/api/data/graph" && (qs || "").includes("entity=")) {
        res.writeHead(200, { "content-type": "application/json" });
        return res.end(JSON.stringify({
          kind: "raw", built_at: 1, entity: "company:STLD", hops: 1,
          caveat: "RAW graph — every edge asserts a relationship with provenance (fixture).",
          nodes: [
            { id: "company:STLD", type: "company", label: "STLD", attrs: { parent: "Fixture Steel Co.", tickers: ["STLD"] } },
            { id: "facility:site:fixture_mill", type: "facility", label: "Fixture Steel Mill", attrs: { kind: "site", category: "steel_mill", lat: 41.37, lon: -84.9, operator: "Fixture Steel Co. (STLD)" } },
          ],
          edges: [{ type: "operates", from: "company:STLD", to: "facility:site:fixture_mill", confidence: "high", attrs: {} }],
        }));
      }
      // exact match wins before prefix match — otherwise /api/data/insider
      // shadows /api/data/insider/history
      const fx = Object.keys(FIXTURES).find((k) => u === k) ||
                 Object.keys(FIXTURES).find((k) => u.startsWith(k + "/"));
      if (fx) {
        res.writeHead(200, { "content-type": "application/json" });
        // Exercise the real ?since= delta path: an unchanged snapshot returns
        // {unchanged:true} with no payload — the client must skip setData.
        // (This also makes the perf window measure PAN smoothness, not
        // redundant 10k-feature re-uploads.)
        if (fx === "/api/data/aircraft" && (qs || "").includes("since=1")) {
          return res.end(JSON.stringify({ unchanged: true, time: 1, count: FIXTURES[fx].count }));
        }
        return res.end(JSON.stringify(FIXTURES[fx]));
      }
      if (u.startsWith("/api/")) {
        res.writeHead(200, { "content-type": "application/json" });
        return res.end("{}");
      }
      let fp = path.join(DIST, u === "/" || u === "/app" ? "index.html" : u);
      if (!existsSync(fp)) fp = path.join(DIST, "index.html"); // SPA fallback
      try {
        const body = readFileSync(fp);
        res.writeHead(200, { "content-type": MIME[path.extname(fp)] || "application/octet-stream" });
        res.end(body);
      } catch {
        res.writeHead(404); res.end("nf");
      }
    });
    srv.listen(0, "127.0.0.1", () => resolve(srv));
  });
}

// ── mechanical checks (run in-page) ────────────────────────────────────────
const CHECKS_SNIPPET = (width, touch, mapPage = true) => `(() => {
  const out = { failures: [], warnings: [], info: {} };
  const vw = window.innerWidth, vh = window.innerHeight;

  // 0. the app actually mounted (found missing 2026-07-11: a blank white
  // screen — e.g. from the vite base:"./" multi-segment routing bug — has
  // no horizontal overflow and no interactive elements, so every check
  // below this one passed clean on a totally dead page. #root is React's
  // mount point (client/index.html); a script that never executed leaves
  // it permanently empty.
  const root = document.getElementById('root');
  if (!root || root.childElementCount === 0) {
    out.failures.push("app did not mount: #root has no children (blank page — script failed to load/execute)");
  } else if (root.querySelector('[data-vt-boot]')) {
    // 2026-07-20: index.html now ships an inline boot splash inside #root
    // (slow-connection fix), so "has children" no longer proves the app ran —
    // the splash still present at check time means the bundle never executed.
    out.failures.push("app did not mount: boot splash still present (script failed to load/execute)");
  }

  // 1. no horizontal overflow caused by the page
  if (document.documentElement.scrollWidth > vw + 1) {
    out.failures.push("horizontal overflow: scrollWidth " + document.documentElement.scrollWidth + " > viewport " + vw);
  }

  // 2. map root fills its viewport region (marker: [data-vt-map]) — map pages only
  const map = document.querySelector('[data-vt-map]');
  if (!${mapPage}) {
    // non-map page: no map assertions
  } else if (!map) {
    out.failures.push("[data-vt-map] marker missing — map root not found");
  } else {
    const r = map.getBoundingClientRect();
    out.info.map = { x: r.x, y: r.y, w: r.width, h: r.height };
    if (r.width < vw * 0.98) out.failures.push("map width " + Math.round(r.width) + " < 98% of viewport " + vw);
    if (Math.abs(vh - r.bottom) > 70) out.failures.push("map bottom " + Math.round(r.bottom) + " leaves >70px gap to viewport " + vh + " (letterboxing)");
    if (r.height < vh * 0.6) out.failures.push("map height " + Math.round(r.height) + " < 60% of viewport " + vh);
    // 3. no permanent overlay covering >40% of the map
    const mapArea = r.width * r.height;
    document.querySelectorAll('body *').forEach((el) => {
      if (el === map || map.contains(el) && el.tagName === 'CANVAS') return;
      const s = getComputedStyle(el);
      if ((s.position === 'absolute' || s.position === 'fixed') && s.visibility !== 'hidden' && s.display !== 'none') {
        const b = el.getBoundingClientRect();
        const ix = Math.max(0, Math.min(b.right, r.right) - Math.max(b.left, r.left));
        const iy = Math.max(0, Math.min(b.bottom, r.bottom) - Math.max(b.top, r.top));
        const cover = (ix * iy) / mapArea;
        const bg = s.backgroundColor;
        const opaque = bg && bg !== 'rgba(0, 0, 0, 0)' && bg !== 'transparent';
        if (cover > 0.4 && opaque && !el.closest('canvas')) {
          out.failures.push("overlay covers " + Math.round(cover * 100) + "% of map: <" + el.tagName.toLowerCase() + " class='" + (el.className || "").toString().slice(0, 40) + "'>");
        }
      }
    });
  }

  // 4. interactive elements: visible, unclipped, touch-sized on phone
  const MIN = ${touch} ? 44 : 24;
  document.querySelectorAll('button, [role="button"], input[type="checkbox"], a[href]').forEach((el) => {
    const b = el.getBoundingClientRect();
    if (b.width === 0 || b.height === 0) return; // hidden is fine
    const s = getComputedStyle(el);
    if (s.visibility === 'hidden' || s.display === 'none') return;
    if (b.right > vw + 1 || b.bottom > vh + 1 || b.left < -1 || b.top < -1) {
      // allow elements that scroll with content below the fold on non-immersive pages
      if (b.top < vh) out.warnings.push("clipped control: " + (el.getAttribute('aria-label') || el.textContent || el.tagName).toString().trim().slice(0, 30));
    }
    if (${touch} && b.width > 0 && (b.width < MIN || b.height < MIN)) {
      const label = (el.getAttribute('aria-label') || el.textContent || el.tagName).toString().trim().slice(0, 30);
      // checkbox inputs may be visually replaced by a larger styled parent
      const p = el.parentElement ? el.parentElement.getBoundingClientRect() : b;
      if (Math.max(p.width, b.width) < MIN || Math.max(p.height, b.height) < MIN) {
        out.warnings.push("touch target < ${44}px: '" + label + "' (" + Math.round(b.width) + "x" + Math.round(b.height) + ")");
      }
    }
  });
  return out;
})()`;

// ── main ───────────────────────────────────────────────────────────────────
// ── DEVICE ENVELOPE D1: per-layer demand matrix (scale_program.md) ─────────
// Boots the map with ALL layers off (the harness's own vt-layers-all-off
// escape), samples a baseline window, then toggles each measured layer ON
// alone, samples, toggles it back OFF — at each emulated CPU tier. Fixture
// data only (deterministic, diffable); SwiftShader numbers are relative
// regression signals, never on-device budgets.
const DEMAND_TIERS = [
  { key: "full", cpu: 1 },
  { key: "throttled4x", cpu: 4 }, // the weak-laptop stand-in (work-PC class)
];
const DEMAND_LAYERS = [
  // default-on set + the cost-budget battery's real toggleables — every id
  // verified wired via clickSwitchVerified; fixture-empty layers still
  // measure (renderedFeatures says how much data backed the number)
  "aircraft", "places", "powerplants", "trains", "sites",
  "terrain", "weather", "weather_temp", "weather_wind",
  "rivergauges", "alerts", "surfacewater", "forest", "boundaries",
];
const DEMAND_SAMPLE_MS = 3000;

async function demandSample(page) {
  return page.evaluate(async (windowMs) => {
    const map = window.__vtMap;
    const heap = () => {
      try { return performance.memory ? performance.memory.usedJSHeapSize / 1048576 : null; }
      catch { return null; }
    };
    const netKB = () => {
      try {
        return performance.getEntriesByType("resource")
          .reduce((s, e) => s + (e.transferSize || 0), 0) / 1024;
      } catch { return null; }
    };
    let longTasks = 0;
    let obs = null;
    try {
      obs = new PerformanceObserver((l) => { longTasks += l.getEntries().length; });
      obs.observe({ entryTypes: ["longtask"] });
    } catch {}
    // camera wiggle so the window measures RENDER demand, not an idle map
    try { map?.easeTo({ center: [-97.6, 38.4], zoom: map.getZoom(), duration: windowMs * 0.9 }); } catch {}
    const deltas = [];
    let last = performance.now();
    const t0 = last;
    await new Promise((res) => {
      const tick = (t) => {
        deltas.push(t - last);
        last = t;
        if (t - t0 >= windowMs) return res();
        requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    });
    try { obs?.disconnect(); } catch {}
    deltas.sort((a, b) => a - b);
    const q = (p) => Math.round(deltas[Math.min(deltas.length - 1, Math.floor(deltas.length * p))] * 10) / 10;
    let rendered = null;
    try { rendered = map ? map.queryRenderedFeatures().length : null; } catch {}
    return {
      medianMs: deltas.length ? q(0.5) : null,
      p95Ms: deltas.length ? q(0.95) : null,
      heapMB: heap(), netKB: netKB(), longTasks, renderedFeatures: rendered,
    };
  }, DEMAND_SAMPLE_MS);
}

async function bootDemandPage(browser, port, cpu) {
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
  const page = await ctx.newPage();
  await page.route("**/*", (route) => {
    const u = route.request().url();
    if (u.startsWith(`http://127.0.0.1:${port}`)) return route.continue();
    return route.abort();
  });
  // all-off boot: the baseline is the empty map, same escape the
  // zero-cost battery uses
  await page.addInitScript(() => { try { sessionStorage.setItem("vt-layers-all-off", "1"); } catch {} });
  const cdp = await ctx.newCDPSession(page);
  await cdp.send("Emulation.setCPUThrottlingRate", { rate: cpu });
  await page.goto(`http://127.0.0.1:${port}/app#/data`, { waitUntil: "load", timeout: 30000 });
  for (let i = 0; i < 60; i++) {
    if (await page.evaluate(() => !document.querySelector(".vt-map-skeleton"))) break;
    await page.waitForTimeout(250);
  }
  await page.waitForTimeout(2000);
  await page.click(".vt-map-fab", { timeout: 2000 }).catch(() => {});
  await page.waitForTimeout(300);
  for (let round = 0; round < 6; round++) {
    const btn = page.locator('.vt-layer-group-head[aria-expanded="false"]').first();
    if (!(await btn.count())) break;
    await btn.click().catch(() => {});
    await page.waitForTimeout(100);
  }
  const flip = async (id, on) => {
    const sw = page.locator(`[data-vt-layer="${id}"] [role="switch"]`).first();
    for (let attempt = 0; attempt < 3; attempt++) {
      const cur = await sw.getAttribute("aria-checked").catch(() => null);
      if (cur === String(on)) return true;
      await page.evaluate((lid) => {
        document.querySelector(`[data-vt-layer="${lid}"]`)?.scrollIntoView({ block: "center" });
      }, id).catch(() => {});
      await sw.click({ timeout: 3000 }).catch(() => {});
      for (let i = 0; i < 8; i++) {
        await page.waitForTimeout(150);
        if ((await sw.getAttribute("aria-checked").catch(() => null)) === String(on)) return true;
      }
    }
    return false;
  };
  return { ctx, page, flip };
}

async function runDemandMatrix(browser, port) {
  const { demandCell, buildLedger } = await import("./demandProfile.mjs");
  const version = JSON.parse(readFileSync(path.join(ROOT, "package.json"), "utf-8")).version;
  const cells = [];
  const skipped = [];
  for (const tier of DEMAND_TIERS) {
    // SELF-HEALING (first live run 2026-08-08: SwiftShader's renderer died
    // 4 layers in — the matrix meets the weak-machine failure mode it
    // measures): the browser context is rebooted on death and the sweep
    // continues from the next layer; the death itself is recorded per
    // layer, which is demand data in its own right.
    let sess = await bootDemandPage(browser, port, tier.cpu);
    let baseline = await demandSample(sess.page);
    console.log(`[demand] tier=${tier.key} baseline median=${baseline.medianMs}ms p95=${baseline.p95Ms}ms heap=${baseline.heapMB?.toFixed?.(0) ?? "n/a"}MB`);
    for (const id of DEMAND_LAYERS) {
      try {
        if (!(await sess.flip(id, true))) {
          skipped.push(`${id}@${tier.key}: toggle never landed`);
          console.log(`[demand] SKIP ${id} @ ${tier.key} — toggle not wired/visible`);
          continue;
        }
        await sess.page.waitForTimeout(1500); // settle: fetch + first paint
        const s = await demandSample(sess.page);
        cells.push(demandCell(id, tier.key, baseline, s));
        console.log(`[demand] ${id} @ ${tier.key}: median=${s.medianMs}ms (base ${baseline.medianMs}) p95=${s.p95Ms}ms heapΔ=${s.heapMB != null && baseline.heapMB != null ? (s.heapMB - baseline.heapMB).toFixed(1) : "n/a"}MB feats=${s.renderedFeatures}`);
        if (!(await sess.flip(id, false))) skipped.push(`${id}@${tier.key}: restore-to-off failed — later cells may be contaminated`);
        await sess.page.waitForTimeout(600);
      } catch (e) {
        const msg = String(e?.message || e).split("\n")[0];
        skipped.push(`${id}@${tier.key}: browser died mid-cycle (${msg}) — rebooted, continued`);
        console.log(`[demand] DIED at ${id} @ ${tier.key}: ${msg} — rebooting context`);
        try { await sess.ctx.close(); } catch {}
        sess = await bootDemandPage(browser, port, tier.cpu);
        baseline = await demandSample(sess.page);
        console.log(`[demand] tier=${tier.key} re-baseline median=${baseline.medianMs}ms heap=${baseline.heapMB?.toFixed?.(0) ?? "n/a"}MB`);
      }
    }
    try { await sess.ctx.close(); } catch {}
  }
  const ledger = buildLedger(version, cells, {
    layers: DEMAND_LAYERS, skipped,
    sampleSeconds: DEMAND_SAMPLE_MS / 1000,
    tiers: DEMAND_TIERS.map((t) => t.key),
    renderer: "SwiftShader (headless) — relative regression signal only",
  });
  mkdirSync(OUT, { recursive: true });
  writeFileSync(path.join(OUT, "demand_profile.json"), JSON.stringify(ledger, null, 1));
  console.log(`\n[demand] ledger written: .visual/demand_profile.json (${cells.length} cells, ${skipped.length} skipped)`);
  if (skipped.length) console.log("[demand] bounded coverage — skipped: " + skipped.join("; "));
  const top = ledger.cells.slice(0, 5).map((c) => `${c.layer}@${c.tier} +${(c.medianMs - c.baselineMedianMs).toFixed(1)}ms`).join(", ");
  console.log("[demand] most demanding: " + top);
}

async function main() {
  if (!existsSync(DIST)) {
    console.error("dist/public missing — run `npm run build` first");
    process.exit(1);
  }
  const { chromium } = await import("playwright");
  const exePath = "/opt/pw-browsers/chromium";
  // Software WebGL (SwiftShader) — maplibre-gl requires a GL context, and
  // headless containers have no GPU. Without these flags the map never
  // fires "load" and every screenshot is a skeleton.
  const GL_ARGS = [
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
    "--ignore-gpu-blocklist",
  ];
  let browser;
  try {
    browser = await chromium.launch({ headless: true, args: GL_ARGS });
  } catch {
    browser = await chromium.launch({
      headless: true,
      executablePath: existsSync(exePath) ? exePath : undefined,
      args: GL_ARGS,
    });
  }
  const srv = await startServer();
  const port = srv.address().port;
  mkdirSync(OUT, { recursive: true });

  if (DEMAND) {
    // measurement mode: matrix only, never gates, exit 0 unless it crashed
    try { await runDemandMatrix(browser, port); }
    finally { await browser.close(); srv.close(); }
    return;
  }

  const results = [];
  for (const [name, cfg] of Object.entries(PAGES)) {
    const route = cfg.route;
    if (only && only !== name) continue;
    for (const vp of WIDTHS) {
      const ctx = await browser.newContext({
        viewport: { width: vp.w, height: vp.h },
        hasTouch: vp.touch,
        isMobile: vp.touch && vp.w < 640,
        // dsf 1: SwiftShader rasterizes in software — dsf 2 quadruples the
        // pixel surface at 1440 and turns the perf guard into a pixel-count
        // test instead of a feature-count test. Real-DPR acceptance is the
        // human's S24 (real GPU).
        deviceScaleFactor: 1,
      });
      const page = await ctx.newPage();
      // Full determinism: only the fixture server is reachable. External
      // requests (tile CDN, fonts) abort instantly — layouts and overlays are
      // what we verify, and aborted tiles let maplibre settle immediately
      // instead of hanging on proxy connection-resets.
      await page.route("**/*", (route) => {
        const u = route.request().url();
        if (u.startsWith(`http://127.0.0.1:${port}`)) return route.continue();
        return route.abort();
      });
      const errors = [];
      page.on("pageerror", (e) => errors.push("pageerror: " + e.message));
      const t0 = Date.now();
      await page.goto(`http://127.0.0.1:${port}${route}`, { waitUntil: "load", timeout: 30000 });
      // TTI: skeleton gone = page interactive (DESIGN.md: <3s on device;
      // headless SwiftShader is far slower — hard regression guard at 12s).
      let tti = null;
      for (let i = 0; i < 60; i++) {
        const gone = await page.evaluate(() => !document.querySelector(".vt-map-skeleton"));
        if (gone) { tti = Date.now() - t0; break; }
        await page.waitForTimeout(250);
      }
      await page.waitForTimeout(2500); // let overlay layers mount
      // PERF BUDGET: drive pans through the __vtMap hook while sampling rAF
      // frame deltas. Software-GL thresholds are regression guards, not the
      // on-device budget (that's DESIGN.md's number).
      const perf = !cfg.map ? {} : await page.evaluate(async () => {
        const map = window.__vtMap;
        if (!map) return { error: "__vtMap hook missing" };
        const c = map.getCenter();
        const runPans = async (record) => {
          const deltas = [];
          let last = performance.now();
          let sampling = true;
          const tick = (t) => { deltas.push(t - last); last = t; if (sampling) requestAnimationFrame(tick); };
          if (record) requestAnimationFrame(tick);
          for (const [dx, dy] of [[8, 3], [-14, -5], [10, 4], [-4, -2]]) {
            map.easeTo({ center: [c.lng + dx, c.lat + dy], duration: 600 });
            await new Promise(r => setTimeout(r, 650));
          }
          sampling = false;
          await new Promise(r => setTimeout(r, 60));
          return deltas.filter(d => d > 0).sort((a, b) => a - b);
        };
        await runPans(false);            // warm-up: first-pan upload hitches
        const sorted = await runPans(true);   // measured window (warm)
        const q = (f) => sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * f))] || 0;
        let rendered = 0;
        // decimation split (perf 3/3): full layer above z4.5, rank-filtered
        // twin below — sample whichever is active at the current zoom
        try { rendered = map.queryRenderedFeatures({ layers: ["aircraft-sym", "aircraft-sym-lo"].filter((l) => map.getLayer(l)) }).length; } catch {}
        return { frames: sorted.length, median: Math.round(q(0.5)), p95: Math.round(q(0.95)),
                 max: Math.round(sorted[sorted.length - 1] || 0), renderedAircraft: rendered };
      });
      const shot = path.join(OUT, `${name}-${vp.w}.png`);
      await page.screenshot({ path: shot });
      const checks = await page.evaluate(CHECKS_SNIPPET(vp.w, vp.touch, cfg.map));
      checks.failures.push(...errors);

      // ── IMAGERY CAPTURE-DATE RATCHET (Phase 3a/5, 2026-07-07): the
      // DESIGN.md imagery-honesty rule requires the capture-date chip on
      // the map whenever the imagery base is on. External Esri calls
      // abort in the harness, so the chip must show a designed honest
      // state ("checking…"/"unknown") — its ABSENCE is the failure.
      if (cfg.map) {
        try {
          const chip = await page.waitForSelector('[data-testid="imagery-date"]', { timeout: 5000 }).catch(() => null);
          if (!chip) {
            checks.failures.push("imagery-date: combined status bar absent with imagery base on (data-testid=imagery-date) — DESIGN.md imagery-honesty rule");
          } else {
            // combined bottom status bar (2026-07-22): "<scale> · z<zoom> ·
            // <ISO date | date n/a | …>" — must carry a scale unit, a zoom,
            // and a designed capture-date state (never a fabricated date).
            const txt = (await chip.textContent()) || "";
            if (!/\b(mi|km|ft|m)\b/.test(txt) || !/z\d/.test(txt) || !/\d{4}-\d{2}-\d{2}|date n\/a|…/.test(txt)) {
              checks.failures.push(`imagery-date: status bar not a designed state (scale·zoom·date): '${txt.slice(0, 60)}'`);
            }
          }
        } catch (e) {
          checks.failures.push("imagery-date: driver error — " + (e?.message || e));
        }
      }

      // ── W1 GLOBE MODE (console charter, 2026-07-07): the /data map
      // defaults to MapLibre's native globe projection; [data-vt-globe]
      // flips to flat (mercator) and back, persisting the choice in
      // localStorage. Asserts the default, the round-trip, and the
      // persisted preference; screenshots both modes at a low zoom where
      // the sphere silhouette is unmistakable (PR evidence — a black
      // square or a flat rectangle in the "globe" shot is a review fail).
      try {
        if (!cfg.map) throw { skip: true };
        const globeBtn = page.locator("[data-vt-globe]");
        if (!(await globeBtn.count())) {
          checks.failures.push("globe: [data-vt-globe] projection toggle missing");
          throw { skip: true };
        }
        const projOf = () => page.evaluate(() => {
          try { return (window.__vtMap.getProjection?.() || {}).type || "mercator"; } catch { return "probe-error"; }
        });
        const p0 = await projOf();
        if (p0 !== "globe") checks.failures.push(`globe: default projection '${p0}' != 'globe' (charter default)`);
        // low-zoom evidence shots: same camera for globe and flat
        const cam = await page.evaluate(() => {
          const m = window.__vtMap; const c = m.getCenter();
          return { lng: c.lng, lat: c.lat, zoom: m.getZoom() };
        });
        await page.evaluate(() => window.__vtMap.jumpTo({ center: [-96.77, 30], zoom: 1.3 }));
        await page.waitForTimeout(900);
        await page.screenshot({ path: path.join(OUT, `${name}-globe-${vp.w}.png`) });
        await globeBtn.click();
        await page.waitForTimeout(700);
        const p1 = await projOf();
        const pref1 = await page.evaluate(() => { try { return localStorage.getItem("vt-map-globe"); } catch { return null; } });
        if (p1 !== "mercator") checks.failures.push(`globe: after toggle projection '${p1}' != 'mercator'`);
        if (pref1 !== "0") checks.failures.push(`globe: flat choice not persisted (vt-map-globe='${pref1}')`);
        const pressed1 = await globeBtn.getAttribute("aria-pressed");
        if (pressed1 !== "false") checks.failures.push(`globe: toggle aria-pressed '${pressed1}' desynced from flat state`);
        await page.screenshot({ path: path.join(OUT, `${name}-flat-${vp.w}.png`) });
        await globeBtn.click();
        await page.waitForTimeout(700);
        const p2 = await projOf();
        if (p2 !== "globe") checks.failures.push(`globe: toggle back gave '${p2}' != 'globe'`);
        // restore the default camera for the batteries that follow
        await page.evaluate((c) => window.__vtMap.jumpTo({ center: [c.lng, c.lat], zoom: c.zoom }), cam);
        await page.waitForTimeout(500);
      } catch (e) {
        if (!e?.skip) checks.failures.push("globe: driver error — " + (e?.message || e));
      }
      // ── W6 ANALYST PANE (console charter, 2026-07-07): [data-vt-analyst]
      // opens the chat panel; the panel must sit fully inside the viewport
      // at every width (SELF-SEE rule — no overflow past the viewport
      // bottom), its input + send control must be reachable and
      // hit-testable, and close must work. At side-panel widths (>=640)
      // the open panel may not occlude any map control. HONESTY
      // CONSTRAINT: this battery NEVER sends a question — the pane fires
      // POST /api/analyst only on an explicit user send, and the battery
      // asserts exactly zero analyst calls (no faked API responses).
      let analystPosts = 0;
      const onAnalystReq = (r) => {
        try { if (new URL(r.url()).pathname === "/api/analyst") analystPosts++; } catch {}
      };
      try {
        if (!cfg.map) throw { skip: true };
        page.on("request", onAnalystReq);
        const aBtn = page.locator("[data-vt-analyst]");
        if (!(await aBtn.count())) {
          checks.failures.push("analyst: [data-vt-analyst] control missing");
          throw { skip: true };
        }
        await aBtn.click();
        // the pane is a lazy chunk: wait past the Suspense fallback for the
        // real input, not just the panel shell
        const aPanel = await page.waitForSelector("[data-vt-analyst-panel]", { timeout: 8000 }).catch(() => null);
        const aInput = await page.waitForSelector("[data-vt-analyst-input]", { timeout: 8000 }).catch(() => null);
        if (!aPanel) {
          checks.failures.push("analyst: panel did not open from its own control");
        } else {
          if (!aInput) checks.failures.push("analyst: input never appeared (lazy chunk failed to load?)");
          // settle the pane's 0.18s entry animation + one composited paint
          // BEFORE geometry checks and the evidence screenshot (2026-07-07
          // flake: at 1440 the DOM checks passed but the capture composited
          // before the panel's first stable paint — an empty-map screenshot
          // with a green battery is worthless as PR evidence)
          await page.waitForTimeout(450);
          const aChecks = await page.evaluate(() => {
            const fails = [];
            const p = document.querySelector("[data-vt-analyst-panel]");
            if (!p) return ["analyst: panel vanished mid-check"];
            const r = p.getBoundingClientRect();
            if (r.bottom > innerHeight + 1) fails.push(`analyst: panel bottom ${Math.round(r.bottom)} past viewport ${innerHeight} — SELF-SEE`);
            if (r.top < -1 || r.left < -1 || r.right > innerWidth + 1) {
              fails.push(`analyst: panel outside viewport (l=${Math.round(r.left)} t=${Math.round(r.top)} r=${Math.round(r.right)})`);
            }
            for (const [label, sel] of [["input", "[data-vt-analyst-input]"], ["send", "[data-vt-analyst-send]"]]) {
              const el = document.querySelector(sel);
              if (!el) { fails.push(`analyst: ${label} control missing`); continue; }
              const b = el.getBoundingClientRect();
              if (b.width < 4 || b.height < 4) { fails.push(`analyst: ${label} has no size`); continue; }
              if (b.bottom > innerHeight + 1 || b.top < -1) { fails.push(`analyst: ${label} not inside the viewport`); continue; }
              const hit = document.elementFromPoint(b.left + b.width / 2, b.top + b.height / 2);
              if (hit && !el.contains(hit) && hit !== el && !hit.contains(el)) {
                fails.push(`analyst: ${label} covered by <${hit.tagName.toLowerCase()} class='${String(hit.className).slice(0, 30)}'>`);
              }
            }
            // side-panel widths: the open pane may not occlude map controls
            // (phone is a bottom sheet — transient, closable, same accepted
            // behavior as the detail-card sheet over the zoom cluster)
            if (innerWidth >= 640) {
              for (const sel of [...(((e) => e && e.getBoundingClientRect().width > 0)(document.querySelector("[data-vt-nav-zoomin]")) ? ["[data-vt-nav-zoomin]", "[data-vt-nav-zoomout]", "[data-vt-nav-compass]"] : ["[data-vt-nav-fab]"]), "[data-vt-fullscreen]", "[data-vt-globe]", "[data-vt-analyst]", "[data-vt-timescrub]"]) {
                const el = document.querySelector(sel);
                if (!el) { fails.push(`analyst: map control ${sel} missing with pane open`); continue; }
                const b = el.getBoundingClientRect();
                const hit = document.elementFromPoint(b.left + b.width / 2, b.top + b.height / 2);
                if (hit && !el.contains(hit) && hit !== el && !hit.contains(el)) {
                  fails.push(`analyst: map control ${sel} OCCLUDED by <${hit.tagName.toLowerCase()} class='${String(hit.className).slice(0, 30)}'> with pane open`);
                }
              }
            }
            return fails;
          });
          checks.failures.push(...aChecks);
          // input is editable — fill/verify/clear, never submit
          if (aInput) {
            await page.fill("[data-vt-analyst-input]", "harness reachability probe");
            const v = await page.inputValue("[data-vt-analyst-input]");
            if (v !== "harness reachability probe") checks.failures.push("analyst: input not editable");
            await page.fill("[data-vt-analyst-input]", "");
          }
          // animations:"disabled" fast-forwards the entry fade at capture —
          // under software GL the composited frame otherwise races the
          // animation (2026-07-07: blank/ghosted panel shots with green
          // DOM checks). Capture-time only; the page itself is untouched.
          await page.screenshot({ path: path.join(OUT, `${name}-analyst-${vp.w}.png`), animations: "disabled" });
          // close via the panel's own close control, then re-assert gone
          await page.click('[data-vt-analyst-panel] [aria-label="Close analyst"]', { timeout: 2000 })
            .catch(() => checks.failures.push("analyst: close control unclickable"));
          await page.waitForTimeout(250);
          if (await page.locator("[data-vt-analyst-panel]").count()) {
            checks.failures.push("analyst: panel still open after its close control");
          }
        }
        if (analystPosts > 0) {
          checks.failures.push(`analyst: battery fired ${analystPosts} POST /api/analyst — the battery must never send a question`);
        }
      } catch (e) {
        if (!e?.skip) checks.failures.push("analyst: driver error — " + (e?.message || e));
      } finally {
        page.off("request", onAnalystReq);
      }
      // ── W3 TIME SCRUBBER (console charter): [data-vt-timescrub] opens the
      // replay panel. Unlike the analyst pane, opening this panel is EXPECTED
      // to fire a read-only GET on open — the battery asserts at least one
      // fires, catching a silent-fetch regression, the mirror image of the
      // analyst's never-fires assertion. TIME MACHINE v2 T-2 (2026-08-13):
      // the default layer (aircraft) now fetches ONE window
      // (/api/data/aircraft/window) instead of a per-instant snapshot; every
      // other layer is unchanged, so both paths count.
      let snapshotGets = 0;
      const onSnapshotReq = (r) => {
        try {
          const p = new URL(r.url()).pathname;
          if (p === "/api/data/snapshot" || p === "/api/data/aircraft/window") snapshotGets++;
        } catch {}
      };
      try {
        if (!cfg.map) throw { skip: true };
        page.on("request", onSnapshotReq);
        const tsBtn = page.locator("[data-vt-timescrub]");
        if (!(await tsBtn.count())) {
          checks.failures.push("timescrub: [data-vt-timescrub] control missing");
          throw { skip: true };
        }
        await tsBtn.click();
        const tsPanel = await page.waitForSelector("[data-vt-timescrub-panel]", { timeout: 8000 }).catch(() => null);
        const tsSlider = await page.waitForSelector("[data-vt-timescrub-slider]", { timeout: 8000 }).catch(() => null);
        if (!tsPanel) {
          checks.failures.push("timescrub: panel did not open from its own control");
        } else {
          if (!tsSlider) checks.failures.push("timescrub: slider never appeared (lazy chunk failed to load?)");
          await page.waitForTimeout(600); // entry animation + the initial snapshot fetch to land
          const tsChecks = await page.evaluate(() => {
            const fails = [];
            const p = document.querySelector("[data-vt-timescrub-panel]");
            if (!p) return ["timescrub: panel vanished mid-check"];
            const r = p.getBoundingClientRect();
            if (r.bottom > innerHeight + 1) fails.push(`timescrub: panel bottom ${Math.round(r.bottom)} past viewport ${innerHeight} — SELF-SEE`);
            if (r.top < -1 || r.left < -1 || r.right > innerWidth + 1) {
              fails.push(`timescrub: panel outside viewport (l=${Math.round(r.left)} t=${Math.round(r.top)} r=${Math.round(r.right)})`);
            }
            for (const [label, sel] of [["layer select", "[data-vt-timescrub-layer]"], ["slider", "[data-vt-timescrub-slider]"], ["play", "[data-vt-timescrub-play]"]]) {
              const el = document.querySelector(sel);
              if (!el) { fails.push(`timescrub: ${label} control missing`); continue; }
              const b = el.getBoundingClientRect();
              if (b.width < 4 || b.height < 4) { fails.push(`timescrub: ${label} has no size`); continue; }
              if (b.bottom > innerHeight + 1 || b.top < -1) { fails.push(`timescrub: ${label} not inside the viewport`); continue; }
              const hit = document.elementFromPoint(b.left + b.width / 2, b.top + b.height / 2);
              if (hit && !el.contains(hit) && hit !== el && !hit.contains(el)) {
                fails.push(`timescrub: ${label} covered by <${hit.tagName.toLowerCase()} class='${String(hit.className).slice(0, 30)}'>`);
              }
            }
            if (innerWidth >= 640) {
              for (const sel of [...(((e) => e && e.getBoundingClientRect().width > 0)(document.querySelector("[data-vt-nav-zoomin]")) ? ["[data-vt-nav-zoomin]", "[data-vt-nav-zoomout]", "[data-vt-nav-compass]"] : ["[data-vt-nav-fab]"]), "[data-vt-fullscreen]", "[data-vt-globe]", "[data-vt-analyst]", "[data-vt-timescrub]"]) {
                const el = document.querySelector(sel);
                if (!el) { fails.push(`timescrub: map control ${sel} missing with panel open`); continue; }
                const b = el.getBoundingClientRect();
                const hit = document.elementFromPoint(b.left + b.width / 2, b.top + b.height / 2);
                if (hit && !el.contains(hit) && hit !== el && !hit.contains(el)) {
                  fails.push(`timescrub: map control ${sel} OCCLUDED by <${hit.tagName.toLowerCase()} class='${String(hit.className).slice(0, 30)}'> with panel open`);
                }
              }
            }
            return fails;
          });
          checks.failures.push(...tsChecks);
          await page.screenshot({ path: path.join(OUT, `${name}-timescrub-${vp.w}.png`), animations: "disabled" });
          await page.click('[data-vt-timescrub-panel] [aria-label="Close time machine"]', { timeout: 2000 })
            .catch(() => checks.failures.push("timescrub: close control unclickable"));
          await page.waitForTimeout(250);
          if (await page.locator("[data-vt-timescrub-panel]").count()) {
            checks.failures.push("timescrub: panel still open after its close control");
          }
        }
        if (snapshotGets < 1) {
          checks.failures.push("timescrub: opening the panel fired zero GET /api/data/snapshot or /api/data/aircraft/window — initial fetch regressed");
        }
      } catch (e) {
        if (!e?.skip) checks.failures.push("timescrub: driver error — " + (e?.message || e));
      } finally {
        page.off("request", onSnapshotReq);
      }
      // ── SELF-SEE (DESIGN.md, human-approved 2026-07-04): after any panel/
      // overlay change, ALL registered content must be reachable — visible or
      // behind an on-screen expand control. The 2026-07-04 defect: the panel
      // grew past the viewport with lower rows unreachable while this harness
      // passed. These assertions check what the human actually checks.
      try {
        if (!cfg.map) throw { skip: true };
        // open the panel via its own on-screen control (as a user would)
        await page.click(".vt-map-fab", { timeout: 1500 }).catch(() => {});
        await page.waitForTimeout(250);
        // expand every collapsed group via its visible header
        for (let round = 0; round < 6; round++) {
          const btn = page.locator('.vt-layer-group-head[aria-expanded="false"]').first();
          if (!(await btn.count())) break;
          await btn.click().catch(() => {});
          await page.waitForTimeout(120);
        }
        // GROUP_ROW_CAP progressive disclosure: rows past a group's cap sit
        // behind a visible "+N more — show all" button by DESIGN — expand
        // them all so "reachable" means what the UX means. (First crossed
        // 2026-07-15: biomass made environmental 13 > cap 12, hiding the
        // forest row from this check while the app was working correctly.)
        for (let round = 0; round < 8; round++) {
          const more = page.locator(".vt-layer-showmore").first();
          if (!(await more.count())) break;
          await more.click().catch(() => {});
          await page.waitForTimeout(120);
        }
        const layerIds = FIXTURES["/api/data/layers"].layers.map((l) => l.id);
        const selfSee = await page.evaluate((ids) => {
          const fails = [];
          const panel = document.querySelector(".vt-layer-panel");
          if (!panel) { fails.push("self-see: layer panel not rendered after opening"); return fails; }
          const pr = panel.getBoundingClientRect();
          if (pr.bottom > innerHeight + 2) fails.push(`self-see: panel bottom ${Math.round(pr.bottom)} past viewport ${innerHeight} — content unreachable`);
          if (panel.scrollHeight > panel.clientHeight + 4) {
            const oy = getComputedStyle(panel).overflowY;
            if (!/(auto|scroll)/.test(oy)) fails.push("self-see: panel overflows without internal scrolling");
          }
          for (const id of ids) {
            const row = panel.querySelector(`[data-vt-layer="${id}"]`);
            if (!row) { fails.push(`self-see: registered layer '${id}' has no reachable panel row`); continue; }
            row.scrollIntoView({ block: "center" });
            const toggle = row.querySelector('[role="switch"]');
            if (!toggle) { fails.push(`self-see: layer '${id}' has no toggle`); continue; }
            const tr = toggle.getBoundingClientRect();
            if (tr.right > innerWidth + 1 || tr.left < -1) fails.push(`self-see: '${id}' toggle off-screen horizontally at ${Math.round(tr.left)}..${Math.round(tr.right)}`);
            if (tr.bottom > innerHeight + 1 || tr.top < -1) fails.push(`self-see: '${id}' toggle not scrollable into viewport`);
            const hit = document.elementFromPoint(tr.left + tr.width / 2, tr.top + tr.height / 2);
            if (hit && !toggle.contains(hit) && hit !== toggle && !hit.contains(toggle)) {
              fails.push(`self-see: '${id}' toggle covered by <${hit.tagName.toLowerCase()} class='${String(hit.className).slice(0, 30)}'>`);
            }
          }
          // v2.4 CONTROL OCCLUSION: with the panel OPEN, no map control may
          // sit under it (the production defect: zoom buttons covered).
          // [data-vt-globe] added W1, [data-vt-analyst] added W6 (console
          // charter): both are registered map controls like the others.
          for (const sel of [...(((e) => e && e.getBoundingClientRect().width > 0)(document.querySelector("[data-vt-nav-zoomin]")) ? ["[data-vt-nav-zoomin]", "[data-vt-nav-zoomout]", "[data-vt-nav-compass]"] : ["[data-vt-nav-fab]"]), "[data-vt-fullscreen]", "[data-vt-globe]", "[data-vt-analyst]", "[data-vt-timescrub]"]) {
            const el = document.querySelector(sel);
            if (!el) { fails.push(`self-see: map control ${sel} missing`); continue; }
            const r = el.getBoundingClientRect();
            if (r.width < 4 || r.height < 4) { fails.push(`self-see: map control ${sel} has no size`); continue; }
            const hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
            if (hit && !el.contains(hit) && hit !== el && !hit.contains(el)) {
              fails.push(`self-see: map control ${sel} OCCLUDED by <${hit.tagName.toLowerCase()} class='${String(hit.className).slice(0, 30)}'> with panel open`);
            }
          }
          // v2.4 ETERNAL-SPINNER rule (armed): any row loading >30s must
          // carry a designed note (retrying / activating / awaiting).
          for (const row of panel.querySelectorAll('[data-vt-rt="loading"]')) {
            const since = Number(row.getAttribute("data-vt-since") || 0);
            if (since && Date.now() - since > 30_000 && !row.querySelector(".vt-layer-covnote")) {
              fails.push(`self-see: '${row.getAttribute("data-vt-layer")}' bare loading spinner >30s — eternal-spinner rule`);
            }
          }
          return fails;
        }, layerIds);
        checks.failures.push(...selfSee);
        // ── SELF-SEE × ANALYST COMBO (2026-07-25 follow-up to the phone-width
        // FAB/panel occlusion fix above, filed as NEXT item (1) in that same
        // session's log): index.css collapses the nav cluster down to the
        // same round `.vt-nav-fab` the phone uses whenever the analyst pane
        // is open, at ANY width <=1279px (`.vt-map-page[data-vt-analyst-
        // open="true"] .vt-nav-fab`) — not just the phone breakpoint the CSS
        // fix above was scoped to. If the layer panel is ALSO open at a mid
        // width (640-1279px, i.e. our 768 canonical width), the FAB COULD sit
        // over the panel's own last row exactly like the already-fixed phone
        // case — no test opened both panels together until now, so this is
        // now a permanent ratchet against that regressing (verified this
        // session with a manual geometry probe that the FAB and panel really
        // do share the same screen rectangle at 768px; the check below found
        // no ACTUAL row occluded — see index.css's updated comment on the
        // 639px rule for the full finding — but it stays wired so a future
        // content/layout change that does cause an overlap gets caught).
        if (vp.w > 639 && vp.w <= 1279) {
          const aBtnCombo = page.locator("[data-vt-analyst]");
          if (await aBtnCombo.count()) {
            await aBtnCombo.click().catch(() => {});
            const aPanelCombo = await page.waitForSelector("[data-vt-analyst-panel]", { timeout: 4000 }).catch(() => null);
            if (aPanelCombo) {
              await page.waitForTimeout(300);
              const comboFails = await page.evaluate((ids) => {
                const fails = [];
                const panel = document.querySelector(".vt-layer-panel");
                if (!panel) return fails;
                for (const id of ids) {
                  const row = panel.querySelector(`[data-vt-layer="${id}"]`);
                  if (!row) continue;
                  row.scrollIntoView({ block: "center" });
                  const toggle = row.querySelector('[role="switch"]');
                  if (!toggle) continue;
                  const tr = toggle.getBoundingClientRect();
                  const hit = document.elementFromPoint(tr.left + tr.width / 2, tr.top + tr.height / 2);
                  if (hit && !toggle.contains(hit) && hit !== toggle && !hit.contains(toggle)) {
                    fails.push(`self-see: '${id}' toggle covered by <${hit.tagName.toLowerCase()} class='${String(hit.className).slice(0, 30)}'> with the analyst pane also open`);
                  }
                }
                return fails;
              }, layerIds);
              checks.failures.push(...comboFails);
              await page.click('[data-vt-analyst-panel] [aria-label="Close analyst"]', { timeout: 2000 }).catch(() => {});
              await page.waitForTimeout(200);
            }
          }
        }
        // ── LEGEND PARITY (DESIGN.md legend rule, human-approved 2026-07-04):
        // every icon the live style draws must have a legend entry, and every
        // legend entry must name an icon registered on the map. Both
        // directions, computed from the LIVE style + source features — not
        // from the legend's own code.
        try {
          await page.click('.vt-legend-head[aria-expanded="false"]', { timeout: 800 }).catch(() => {});
          await page.waitForTimeout(200);
          const parity = await page.evaluate(() => {
            const fails = [];
            const m = window.__vtMap;
            if (!m) return ["legend-parity: __vtMap hook missing"];
            const used = new Set();
            for (const l of (m.getStyle().layers || [])) {
              if (l.type !== "symbol") continue;
              const ii = (l.layout || {})["icon-image"];
              if (!ii) continue;
              if (typeof ii === "string") { used.add(ii); continue; }
              if (Array.isArray(ii) && ii[0] === "get") {
                const prop = ii[1];
                try {
                  for (const f of m.querySourceFeatures(l.source)) {
                    const v = f && f.properties && f.properties[prop];
                    if (v) used.add(String(v));
                  }
                } catch {}
              }
            }
            const legend = new Set(
              [...document.querySelectorAll("[data-vt-icon]")].map((e) => e.getAttribute("data-vt-icon")),
            );
            for (const name of used) {
              if (!legend.has(name)) fails.push(`legend-parity: map draws '${name}' with NO legend entry — failed build per the legend rule`);
            }
            for (const name of legend) {
              if (!m.hasImage || !m.hasImage(name)) fails.push(`legend-parity: legend claims '${name}' but no such icon is registered on the map`);
            }
            for (const img of document.querySelectorAll("[data-vt-icon] img")) {
              if (!img.getAttribute("src")) fails.push("legend-parity: legend entry with empty icon render");
            }
            return fails.length ? fails : [`legend-parity-ok:${used.size} used / ${legend.size} entries`];
          });
          const ok = parity.find((p) => String(p).startsWith("legend-parity-ok:"));
          if (ok) checks.info.legendParity = ok;
          else checks.failures.push(...parity);
          // PR evidence: the legend itself, scrolled into view beside the map
          await page.evaluate(() => document.querySelector("[data-vt-legend]")?.scrollIntoView({ block: "center" }));
          await page.waitForTimeout(200);
          await page.screenshot({ path: path.join(OUT, `${name}-legend-${vp.w}.png`) });
        } catch (e) {
          checks.failures.push("legend-parity: driver error — " + (e?.message || e));
        }
        // restore collapsed-by-default state for the phone screenshot honesty
        if (vp.touch) await page.click('.vt-layer-panel [aria-label="Collapse layers panel"]').catch(() => {});
      } catch (e) {
        if (!e?.skip) checks.failures.push("self-see: driver error — " + (e?.message || e));
      }
      // VERIFIED SWITCH CLICK ([MEASUREMENT-DEBT 2026-07-20] fix): the
      // batteries below restore page state via toggle clicks that were
      // fire-and-forget (`.catch(() => {})`) — a missed click left terrain
      // enabled + the camera auto-tilted, and every later check inherited a
      // pitched viewport (the ~50% data@1440 trail flake; instrumented run
      // pinned rowChecked=true after the "off" sweep = the click never
      // landed). This clicks, VERIFIES aria-checked actually flipped, and
      // retries (re-scrolling) up to 2 more times. Assertions are untouched
      // — this fixes the driver's hand, not the ruler's scale.
      const clickSwitchVerified = async (id, wantChecked) => {
        const sw = page.locator(`[data-vt-layer="${id}"] [role="switch"]`).first();
        for (let attempt = 0; attempt < 3; attempt++) {
          const cur = await sw.getAttribute("aria-checked").catch(() => null);
          if (cur === String(wantChecked)) return true;
          await page.evaluate((lid) => {
            document.querySelector(`[data-vt-layer="${lid}"]`)?.scrollIntoView({ block: "center" });
          }, id).catch(() => {});
          await page.waitForTimeout(80);
          await sw.click({ timeout: 4000 }).catch(() => {});
          // poll briefly for the pill to reflect the click
          for (let i = 0; i < 8; i++) {
            await page.waitForTimeout(150);
            const now = await sw.getAttribute("aria-checked").catch(() => null);
            if (now === String(wantChecked)) return true;
          }
        }
        return false;
      };
      // ── TOGGLE CONSISTENCY (state-desync repair 2026-07-04): for EVERY
      // toggleable registry layer, flipping the switch must move pill,
      // label, and actual map state TOGETHER — pill ON with a label still
      // reading "off" (the production defect) is a failed build. Runs at
      // 1440 only: it exercises state wiring, not layout.
      try {
        if (!cfg.map || vp.w !== 1440) throw { skip: true };
        await page.click(".vt-map-fab", { timeout: 1500 }).catch(() => {});
        await page.waitForTimeout(250);
        for (let round = 0; round < 6; round++) {
          const btn = page.locator('.vt-layer-group-head[aria-expanded="false"]').first();
          if (!(await btn.count())) break;
          await btn.click().catch(() => {});
          await page.waitForTimeout(120);
        }
        await page.click('.vt-legend-head[aria-expanded="true"]', { timeout: 800 }).catch(() => {});
        const toggleables = FIXTURES["/api/data/layers"].layers
          .filter((l) => l.status === "live" && l.id !== "imagery") // imagery is the base — stays on
          .map((l) => l.id);
        const desyncs = [];
        for (const id of toggleables) {
          const sw = page.locator(`[data-vt-layer="${id}"] [role="switch"]`).first();
          const before = await sw.getAttribute("aria-checked");
          await page.evaluate((lid) => {
            document.querySelector(`[data-vt-layer="${lid}"]`)?.scrollIntoView({ block: "center" });
          }, id);
          await page.waitForTimeout(100);
          await sw.click({ timeout: 4000 }).catch(() => desyncs.push(`toggle-consistency: '${id}' switch UNCLICKABLE`));
          // wait for the layer's effect to settle: label must leave "off"
          // (loading/active/error all acceptable — but never stuck at off)
          let state = null;
          for (let i = 0; i < 30; i++) {
            state = await page.evaluate((lid) => {
              const row = document.querySelector(`[data-vt-layer="${lid}"]`);
              const swEl = row?.querySelector('[role="switch"]');
              return {
                pill: swEl?.getAttribute("aria-checked"),
                rt: row?.getAttribute("data-vt-rt"),
                label: row?.querySelector(".vt-layer-status")?.textContent?.trim() || "",
              };
            }, id);
            if (state && state.rt && state.rt !== "off" && state.rt !== "none") break;
            await page.waitForTimeout(200);
          }
          if (!state || state.pill !== (before === "true" ? "false" : "true")) {
            desyncs.push(`toggle-consistency: '${id}' pill did not flip (before=${before}, after=${state?.pill})`);
          } else if (state.pill === "true" && (state.rt === "off" || state.rt === "none")) {
            desyncs.push(`toggle-consistency: '${id}' pill ON but runtime '${state.rt}' / label '${state.label}' — the production desync`);
          }
          // flip back to leave the page in its default state — VERIFIED
          // (a silent miss here poisoned every later check; see helper)
          if (!(await clickSwitchVerified(id, before === "true"))) {
            desyncs.push(`toggle-consistency: '${id}' flip-back never landed — page left in non-default state`);
          }
          await page.waitForTimeout(150);
        }
        checks.failures.push(...desyncs);
        if (!desyncs.length) checks.info.toggleConsistency = `${toggleables.length} layers toggled clean`;
        if (vp.touch) {} // (1440 only — no panel-state restore needed)
      } catch (e) {
        if (!e?.skip) checks.failures.push("toggle-consistency: driver error — " + (e?.message || e));
      }
      // ── COST-BUDGET BADGE (BUILD ORDER 4 #2, GIP Part 4 cost-budget item):
      // DEFAULT_ON's registry-native costTier sum is 13 ("light" — badge
      // silent, proven by every other capture in this file never showing
      // it). Turning on every remaining real toggleable layer (terrain +
      // both weather fields + rivergauges/alerts/surfacewater/forest/
      // boundaries — genuinely wired ids, unlike the scale battery's
      // synthetic ones) adds 22 more for a 35 total, crossing the >26
      // "heavy" threshold — proves the badge is an actually-wired consumer
      // of the registry field, not decorative dead code.
      try {
        if (!cfg.map || vp.w !== 1440) throw { skip: true };
        await page.click(".vt-map-fab", { timeout: 1500 }).catch(() => {});
        await page.waitForTimeout(200);
        for (let round = 0; round < 6; round++) {
          const btn = page.locator('.vt-layer-group-head[aria-expanded="false"]').first();
          if (!(await btn.count())) break;
          await btn.click().catch(() => {});
          await page.waitForTimeout(100);
        }
        const extraToggle = ["terrain", "weather", "weather_temp", "weather_wind", "rivergauges", "alerts", "surfacewater", "forest", "boundaries"];
        for (const id of extraToggle) {
          await clickSwitchVerified(id, true);
          await page.waitForTimeout(60);
        }
        await page.waitForTimeout(200);
        const costNote = await page.evaluate(() => document.querySelector(".vt-cost-note")?.textContent?.trim() || null);
        checks.info.costBudgetBadge = costNote;
        if (costNote !== "heavy load") {
          checks.failures.push(`cost-budget: expected 'heavy load' badge with ${extraToggle.length} extra layers on (weight 35), got '${costNote}' — cost-budget consumer not wired or threshold drifted`);
        }
        // ── DRAPE-ORDER RATCHET (terrain-lag root cause, probe-proven
        // 2026-07-20): with terrain ON (it is, in this battery), a custom GL
        // layer buried under a draped layer (fill/line/raster/background/
        // hillshade) splits MapLibre's RTT stack and silently defeats the
        // terrain texture cache — 588ms/frame buried vs 180ms floated in the
        // same scene (scratchpad mechanism probe). lib/drapeOrder.ts guards
        // this at runtime; this check makes a regression build-failing.
        const buried = await page.evaluate(() => {
          const DRAPED = new Set(["background", "fill", "line", "raster", "hillshade", "color-relief"]);
          const map = window.__vtMap;
          const order = map?.style?._order || [];
          const layers = map?.style?._layers || {};
          const seq = order.map((id) => ({ id, type: layers[id]?.type }));
          const out = [];
          for (let i = 0; i < seq.length; i++) {
            if (seq[i].type !== "custom") continue;
            if (seq.slice(i + 1).some((l) => DRAPED.has(l.type))) out.push(seq[i].id);
          }
          return out;
        });
        checks.info.buriedCustomLayers = buried.join(",") || "none";
        if (buried.length) {
          checks.failures.push(`drape-order: custom layer(s) buried under draped layers with terrain on: ${buried.join(", ")} — RTT stack split regresses the 2026-07-20 terrain-lag fix (lib/drapeOrder.ts)`);
        }
        // ── TRUE-ALTITUDE DATUM RATCHET (human 2026-07-21 round 16: "the
        // plane shoots way up visually in the sky i dont want that"): the
        // aircraft altScale is pinned to 1 in EVERY mesh state — the
        // exaggeration lifts the terrain only; the displayAlt hook clamps
        // planes/curtain above the exaggerated mesh instead. Any non-1
        // altScale with terrain on is the planes-launched-into-the-sky bug.
        const datum = await page.evaluate(() => ({
          exag: window.__vtMap?.getTerrain?.()?.exaggeration ?? null,
          altScale: window.__vtAir?.getAltScale?.() ?? null,
        }));
        checks.info.terrainDatum = `exag=${datum.exag} altScale=${datum.altScale}`;
        if (datum.altScale != null && Math.abs(datum.altScale - 1) > 1e-6) {
          checks.failures.push(`true-altitude datum: aircraft altScale ${datum.altScale} must stay 1 with terrain on (exaggeration ${datum.exag} lifts terrain, never aircraft — 2026-07-21 round-16 contract)`);
        }
        // flip back off to restore default state for the remaining checks/
        // screenshots — VERIFIED per switch: a missed terrain off-click left
        // the auto-tilted camera pitched for every later check (the trail
        // flake's root cause, instrumented 2026-07-20)
        for (const id of extraToggle) {
          if (!(await clickSwitchVerified(id, false))) {
            checks.failures.push(`cost-budget: '${id}' restore-to-off never landed — page left in non-default state`);
          }
        }
        await page.waitForTimeout(400);
        // one-datum, off side: terrain off must return the planes to 1×
        // (the stale-exaggeration half of the 2026-07-21 report)
        const altOff = await page.evaluate(() =>
          window.__vtMap?.getTerrain?.() ? "terrain-still-on" : (window.__vtAir?.getAltScale?.() ?? null));
        if (typeof altOff === "number" && Math.abs(altOff - 1) > 1e-6) {
          checks.failures.push(`one-datum: aircraft altScale stuck at ${altOff} after terrain off (must return to 1)`);
        }
        await page.waitForTimeout(150);
      } catch (e) {
        if (!e?.skip) checks.failures.push("cost-budget: driver error — " + (e?.message || e));
      }
      // ── U1 FIELDS-ON VISIBILITY (weather-upgrade directive 2026-07-04):
      // with temp + wind toggled ON at the DEFAULT opacity, the fields must
      // visibly render (canvas pixel diff) while the base map and
      // live-tracking layers stay visible: aircraft still rendered, rasters
      // BELOW symbols, opacity at the 60% default, arrows from the sampled
      // grid present. Pixel proof, never HTTP 200s (DESIGN.md tile rule).
      try {
        if (!cfg.map) throw { skip: true };
        // clean OFF capture: collapse the panel so both captures share chrome
        await page.click('.vt-layer-panel [aria-label="Collapse layers panel"]', { timeout: 1200 }).catch(() => {});
        await page.waitForTimeout(400);
        const canvas = page.locator("[data-vt-map] canvas").first();
        const offShot = await canvas.screenshot();
        // toggle the two field layers as a user would (panel -> group -> switch)
        await page.click(".vt-map-fab", { timeout: 1500 }).catch(() => {});
        await page.waitForTimeout(250);
        for (let round = 0; round < 6; round++) {
          const btn = page.locator('.vt-layer-group-head[aria-expanded="false"]').first();
          if (!(await btn.count())) break;
          await btn.click().catch(() => {});
          await page.waitForTimeout(120);
        }
        // collapse the legend so the taller v3 block can't push the target
        // rows under the sticky panel head, then center each row before
        // clicking (edge-scrolled switches fail Playwright actionability)
        await page.click('.vt-legend-head[aria-expanded="true"]', { timeout: 800 }).catch(() => {});
        await page.waitForTimeout(150);
        for (const id of ["weather_temp", "weather_wind"]) {
          await page.evaluate((lid) => {
            document.querySelector(`[data-vt-layer="${lid}"]`)?.scrollIntoView({ block: "center" });
          }, id);
          await page.waitForTimeout(120);
          await page.locator(`[data-vt-layer="${id}"] [role="switch"]`).first().click({ timeout: 4000 });
          await page.waitForTimeout(150);
        }
        // temp VALUE LABELS sub-toggle ON too — the 2026-07-05 production
        // bug lived exactly in this state (labels + arrows together) and
        // the battery never exercised it: labels-off was the default.
        // (.vt-field-controls is a SIBLING of the data-vt-layer row, so the
        // locator anchors on the checkbox's own label text.)
        const valueLabels = page.locator(".vt-field-check", { hasText: "value labels" }).first();
        await valueLabels.evaluate((el) => el.scrollIntoView({ block: "center" }));
        await page.waitForTimeout(120);
        await valueLabels.click({ timeout: 4000 });
        await page.waitForTimeout(150);
        // wait until tiles are loaded AND arrow symbols are actually PLACED
        // (getLayer succeeds before the SDF icon rasterizes — querying placed
        // features is the only honest "it renders" signal)
        let mounted = false;
        for (let i = 0; i < 48; i++) {
          mounted = await page.evaluate(() => {
            const m = window.__vtMap;
            if (!(m && m.getLayer("wx-temp_new") && m.getLayer("wx-wind_new") &&
                  m.getLayer("wx-wind-arrows") && m.areTilesLoaded())) return false;
            // wait for BOTH label sets to actually place — sampling after
            // only the arrows placed raced ahead of temp-label placement at
            // 1440 (flaky false-fail seen 2026-07-05; labels were visibly
            // rendered in the screenshot taken a second later)
            try {
              return m.queryRenderedFeatures({ layers: ["wx-wind-arrows"] }).length > 0 &&
                     m.queryRenderedFeatures({ layers: ["wx-temp-labels"] }).length > 0;
            } catch { return false; }
          });
          if (mounted) break;
          await page.waitForTimeout(250);
        }
        if (!mounted) checks.failures.push("fields-on: wx layers/tiles/arrows never rendered (status probe, tile fixture, or arrow grid broken)");
        await page.waitForTimeout(400); // settle one render pass
        const fieldChecks = await page.evaluate(() => {
          const fails = [];
          const m = window.__vtMap;
          if (!m) return ["fields-on: __vtMap hook missing"];
          for (const l of ["wx-temp_new", "wx-wind_new"]) {
            const op = m.getLayer(l) ? m.getPaintProperty(l, "raster-opacity") : null;
            if (op !== 0.6) fails.push(`fields-on: ${l} raster-opacity ${op} != 0.6 default — registry default broken`);
          }
          const order = (m.getStyle().layers || []).map((l) => l.id);
          if (order.indexOf("wx-temp_new") > order.indexOf("aircraft-sym"))
            fails.push("fields-on: temp raster ABOVE aircraft symbols — live tracking obscured");
          let aircraft = 0, arrows = 0, tempLabels = 0;
          try { aircraft = m.queryRenderedFeatures({ layers: ["aircraft-sym", "aircraft-sym-lo"].filter((l) => m.getLayer(l)) }).length; } catch {}
          try { arrows = m.queryRenderedFeatures({ layers: ["wx-wind-arrows"] }).length; } catch {}
          try { tempLabels = m.queryRenderedFeatures({ layers: ["wx-temp-labels"] }).length; } catch {}
          if (!aircraft) fails.push("fields-on: no aircraft rendered with fields on — live tracking not visible");
          if (!arrows) fails.push("fields-on: no wind arrows rendered from the sampled grid");
          // REPAIR ratchet 2026-07-05: temp value-labels ON must NOT eat the
          // wind arrows (production bug: collision pass split the arrow/kt
          // pair, leaving orphaned kt text). Both label sets coexist, and
          // the by-construction guarantees are pinned: the arrows layer sits
          // fully outside the collision pass (both directions), and temp
          // labels are offset above the shared grid point.
          if (!tempLabels) fails.push("fields-on: no temp value-labels rendered with the sub-toggle on");
          if (arrows && tempLabels) {
            try {
              const lo = (p) => m.getLayoutProperty("wx-wind-arrows", p);
              if (lo("icon-ignore-placement") !== true || lo("text-ignore-placement") !== true ||
                  lo("icon-allow-overlap") !== true || lo("text-allow-overlap") !== true)
                fails.push("fields-on: wx-wind-arrows re-entered the collision pass — arrow/kt pair can be split again");
              const anchor = m.getLayoutProperty("wx-temp-labels", "text-anchor");
              if (anchor !== "bottom")
                fails.push(`fields-on: wx-temp-labels anchor '${anchor}' != 'bottom' — label no longer dodges the arrow at shared grid points`);
            } catch (e) { fails.push("fields-on: collision-pin probe failed — " + (e?.message || e)); }
          }
          // v2.4 occlusion rule re-checked WITH fields on: enabling a layer
          // grows the attribution strip — it may not spread under controls
          // (the 390px defect this caught: 2-line attribution over zoom-out).
          for (const sel of [...(((e) => e && e.getBoundingClientRect().width > 0)(document.querySelector("[data-vt-nav-zoomin]")) ? ["[data-vt-nav-zoomin]", "[data-vt-nav-zoomout]", "[data-vt-nav-compass]"] : ["[data-vt-nav-fab]"]), "[data-vt-fullscreen]", "[data-vt-globe]", "[data-vt-analyst]", "[data-vt-timescrub]"]) {
            const el = document.querySelector(sel);
            if (!el) { fails.push(`fields-on: map control ${sel} missing`); continue; }
            const r = el.getBoundingClientRect();
            const hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
            if (hit && !el.contains(hit) && hit !== el && !hit.contains(el)) {
              fails.push(`fields-on: map control ${sel} OCCLUDED by <${hit.tagName.toLowerCase()} class='${String(hit.className).slice(0, 40)}'> with fields on`);
            }
          }
          return fails;
        });
        checks.failures.push(...fieldChecks);
        // ON capture with identical chrome, then the pixel proof
        await page.click('.vt-layer-panel [aria-label="Collapse layers panel"]', { timeout: 1200 }).catch(() => {});
        await page.waitForTimeout(400);
        const onShot = await canvas.screenshot();
        const meanDiff = pngMeanDiff(offShot, onShot);
        checks.info.fieldsMeanDiff = Math.round(meanDiff * 10) / 10;
        if (meanDiff < 3) checks.failures.push(`fields-on: canvas mean diff ${meanDiff.toFixed(2)} < 3 — fields not visibly rendering at default opacity`);
        // PR evidence: full page with fields on (panel open on desktop,
        // collapsed on touch — same honesty convention as the main shots)
        if (!vp.touch) await page.click(".vt-map-fab", { timeout: 1200 }).catch(() => {});
        await page.waitForTimeout(300);
        await page.screenshot({ path: path.join(OUT, `${name}-fields-${vp.w}.png`) });
      } catch (e) {
        if (!e?.skip) checks.failures.push("fields-on: driver error — " + (e?.message || e));
      }
      // ── LANDING GLOBE SYMBOLS (directive 2026-07-05): the hero globe
      // renders REAL registry silhouettes, not dots — symbol layers fed by
      // the shared icon registry, heading-rotated, every icon actually
      // registered on the map (hasImage), aircraft shapes varied. Placed
      // features are the honest "it renders" signal (same lesson as
      // fields-on). Screenshot with the section in view is the PR evidence.
      try {
        if (name !== "landing") throw { skip: true };
        await page.evaluate(() => document.getElementById("data-intel")?.scrollIntoView({ block: "center" }));
        let up = false;
        for (let i = 0; i < 80; i++) {
          up = await page.evaluate(() => {
            const m = window.__vtGlobe;
            if (!(m && m.getLayer && m.getLayer("di-air-p"))) return false;
            try { return m.queryRenderedFeatures({ layers: ["di-air-p"] }).length > 0; } catch { return false; }
          });
          if (up) break;
          await page.waitForTimeout(250);
        }
        if (!up) {
          checks.failures.push("landing-globe: globe never booted or no aircraft symbols placed");
        } else {
          const g = await page.evaluate(() => {
            const fails = [];
            const m = window.__vtGlobe;
            const style = m.getStyle().layers || [];
            for (const id of ["di-air-p", "di-sites-p"]) {
              const l = style.find((x) => x.id === id);
              if (!l) { fails.push(`landing-globe: layer ${id} missing`); continue; }
              if (l.type !== "symbol") fails.push(`landing-globe: ${id} is ${l.type}, not symbol — dots regressed`);
            }
            const iconsOf = (src) => {
              try { return [...new Set(m.querySourceFeatures(src).map((f) => f.properties.icon))]; } catch { return []; }
            };
            const air = iconsOf("di-air");
            if (air.length < 2) fails.push("landing-globe: aircraft icons not varied: " + air.join(","));
            for (const src of ["di-air", "di-sites"]) {
              const missing = iconsOf(src).filter((i) => !i || !m.hasImage(i));
              if (missing.length) fails.push(`landing-globe: ${src} icons not in shared registry: ` + missing.join(","));
            }
            try {
              const rot = JSON.stringify(m.getLayoutProperty("di-air-p", "icon-rotate"));
              if (!rot.includes("heading")) fails.push("landing-globe: icon-rotate not bound to heading: " + rot);
            } catch { fails.push("landing-globe: icon-rotate unreadable"); }
            return fails;
          });
          checks.failures.push(...g);
          await page.waitForTimeout(400); // settle a render pass for the shot
          await page.screenshot({ path: path.join(OUT, `${name}-globe-${vp.w}.png`) });
        }
      } catch (e) {
        if (!e?.skip) checks.failures.push("landing-globe: driver error — " + (e?.message || e));
      }
      // Perf budget (headless regression guards; on-device budget in DESIGN.md)
      // CALIBRATED GATE ([RULE-REVIEW] 2026-07-05, perf repair 2/3):
      // performance regressions now FAIL the build like visual ones.
      // Thresholds are ~2x the worst numbers observed across recent green
      // runs under SwiftShader (data TTI 893-1306ms; medians 33-117ms by
      // width; p95 50-183ms) — regression guards, not the on-device budget
      // (DESIGN.md owns that; the S24 is the acceptance device). Direction
      // of bias: this change can only make builds FAIL more, never look
      // better (measurement-integrity note).
      if (tti == null) checks.failures.push("TTI: skeleton never cleared (>15s)");
      else if (cfg.map && tti > 3000) checks.failures.push(`TTI ${tti}ms > 3000ms map-page gate (observed ceiling ~1.3s)`);
      else if (tti > 12000) checks.failures.push(`TTI ${tti}ms > 12s headless guard`);
      if (cfg.map && perf.error) checks.failures.push("perf: " + perf.error);
      else if (cfg.map) {
        const MEDIAN_GATE = { 390: 120, 768: 200, 1440: 250 };
        const medGate = MEDIAN_GATE[vp.w] || 300;
        if (perf.median > medGate) checks.failures.push(`perf: median frame ${perf.median}ms > ${medGate}ms gate @${vp.w} (steady-state jank at 10k features)`);
        if (perf.p95 > 350) checks.failures.push(`perf: p95 frame ${perf.p95}ms > 350ms gate (observed ceiling 183ms)`);
        if (perf.p95 > 250) checks.warnings.push(`perf: p95 frame ${perf.p95}ms (upload-hitch spikes)`);
        if (!perf.renderedAircraft) checks.warnings.push("perf: no aircraft features rendered in viewport sample");
        // DATA-RICHNESS GUARD (enables low-zoom decimation without data
        // loss, and forbids the cheat): the SOURCE must hold the full
        // fixture regardless of how many icons the renderer draws.
        // Deduped by icao24 — querySourceFeatures returns per-tile copies.
        if (vp.w === 1440) {
          const srcCount = await page.evaluate(() => {
            try {
              const m = window.__vtMap;
              return new Set(m.querySourceFeatures("aircraft").map((f) => f.properties.icao24)).size;
            } catch { return -1; }
          });
          checks.info.aircraftSourceCount = srcCount;
          if (srcCount >= 0 && srcCount < 9500) {
            checks.failures.push(`data-richness: aircraft source holds ${srcCount} < 9500 unique features — decimation must trim RENDERING, never DATA`);
          }

          // TRAIL LIVE-REFRESH RATCHET ([REPAIR 2026-07-05]): the trail was a
          // static snapshot — fetched once at selection, never extended while
          // the popup stayed open. Select an aircraft, require the freshness
          // chip, then hold the popup open across the 30s refresh interval and
          // require (a) a second /api/data/track pull and (b) the trail
          // geometry to GROW (the stateful fixture adds a point per call).
          const spot = await page.evaluate(() => {
            try {
              const m = window.__vtMap;
              const rect = m.getContainer().getBoundingClientRect();
              for (const f of m.querySourceFeatures("aircraft")) {
                const [lo, la] = f.geometry.coordinates;
                const px = m.project([lo, la]);
                if (px.x > 260 && px.x < rect.width - 260 && px.y > 140 && px.y < rect.height - 140) {
                  return { x: rect.left + px.x, y: rect.top + px.y };
                }
              }
            } catch {}
            return null;
          });
          if (!spot) {
            checks.failures.push("trail: no clickable aircraft feature found in viewport");
          } else {
            const callsAtSelect = trackCalls;
            await page.mouse.click(spot.x, spot.y);
            const chip = await page.waitForSelector('[data-testid="trail-freshness"]', { timeout: 8000 }).catch(() => null);
            if (!chip) {
              checks.failures.push("trail: freshness chip absent after selecting an aircraft (data-testid=trail-freshness)");
            } else {
              const chipText = await chip.textContent();
              if (!/last position/.test(chipText || "")) {
                checks.failures.push(`trail: freshness chip lacks 'last position' age text: "${chipText}"`);
              }
              const lenBefore = await page.evaluate(() => {
                try { return typeof window.__vtTrailLen === "number" ? window.__vtTrailLen : -1; } catch { return -1; }
              });
              await page.waitForTimeout(32_000); // one full refresh interval
              const lenAfter = await page.evaluate(() => {
                try { return typeof window.__vtTrailLen === "number" ? window.__vtTrailLen : -1; } catch { return -1; }
              });
              if (trackCalls <= callsAtSelect + 1) {
                checks.failures.push(`trail: no live re-pull while popup open (track calls ${callsAtSelect} -> ${trackCalls}; expected >= ${callsAtSelect + 2})`);
              }
              if (!(lenAfter > lenBefore) || lenBefore < 0) {
                checks.failures.push(`trail: geometry did not extend across the refresh interval (${lenBefore} -> ${lenAfter} points)`);
              }
              checks.info.trail = { callsAtSelect, callsAfter: trackCalls, lenBefore, lenAfter };
              await page.keyboard.press("Escape").catch(() => {});
            }
          }
        }
      }
      results.push({ page: name, width: vp.w, label: vp.label, screenshot: shot, tti, perf, ...checks });
      await ctx.close();
    }
  }
  // ── v2.4 ZERO-COST-WHEN-OFF + interactive budget (all layers off) ────────
  // With every layer toggled off, the page must (a) make NO layer-data API
  // calls (registry + auth are the only allowed /api hits) and (b) go
  // interactive fast — the regression guard for "the site got slower".
  if (!only || only === "data") {
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
    const page = await ctx.newPage();
    await page.addInitScript(() => sessionStorage.setItem("vt-layers-all-off", "1"));
    const disallowed = [];
    const ALLOWED = new Set(["/api/data/layers", "/api/auth/me", "/api/health"]);
    page.on("request", (r) => {
      try {
        const u = new URL(r.url());
        if (u.pathname.startsWith("/api/") && !ALLOWED.has(u.pathname)) disallowed.push(u.pathname);
      } catch {}
    });
    await page.route("**/*", (route) => {
      const u = route.request().url();
      if (u.startsWith(`http://127.0.0.1:${port}`)) return route.continue();
      return route.abort();
    });
    const t0 = Date.now();
    await page.goto(`http://127.0.0.1:${port}${PAGES.data.route}`, { waitUntil: "load", timeout: 30000 });
    let ttiOff = null;
    for (let i = 0; i < 60; i++) {
      const gone = await page.evaluate(() => !document.querySelector(".vt-map-skeleton"));
      if (gone) { ttiOff = Date.now() - t0; break; }
      await page.waitForTimeout(100);
    }
    await page.waitForTimeout(4500); // deferred-mount window: violations would fire here
    const failures = [];
    if (ttiOff == null) failures.push("all-off: skeleton never cleared");
    else if (ttiOff > 2500) failures.push(`all-off TTI ${ttiOff}ms > 2500ms budget (headless)`);
    if (disallowed.length) failures.push(`ZERO-COST-WHEN-OFF violated: layer-data calls with all layers off: ${[...new Set(disallowed)].join(", ")}`);
    results.push({ page: "data-all-off", width: 1440, label: "zero-cost", screenshot: "-", tti: ttiOff,
                   failures, warnings: [], info: { disallowed: disallowed.length } });
    await ctx.close();
  }

  // ── LAYER-SCALE SYNTHETIC HARNESS (BUILD ORDER 4 #2, GIP Part 4 UI
  // SCALABILITY): datamap.tsx now reads registry-native `group`/`costTier`
  // and caps each open panel group at GROUP_ROW_CAP (12) rows behind a
  // "show all" control, with any group not explicitly opened-by-default
  // starting collapsed. This battery generates 50/100/200-layer synthetic
  // registries (round-robin across a mix of real + brand-new group ids and
  // all three cost tiers) via a per-context route override — the shared
  // FIXTURES/server are untouched, so every other page's determinism is
  // unaffected — and MEASURES whether that architecture actually holds real
  // numbers rather than assuming it does. No silent cap: results are
  // logged for every scale, not just the pass/fail edges.
  if (!only || only === "data" || only === "scale") {
    const SCALE_GROUPS = ["base", "live", "facilities", "environmental", "filings", "signals", "synth_new_a", "synth_new_b"];
    const SCALE_TIERS = ["light", "moderate", "heavy"];
    const syntheticLayers = (n) => {
      const out = [];
      for (let i = 0; i < n; i++) {
        const g = SCALE_GROUPS[i % SCALE_GROUPS.length];
        out.push({
          id: `synth_${i}`, name: `Synthetic layer ${i}`,
          kind: g === "signals" ? "signal" : "raw",
          status: g === "signals" ? "planned" : "live",
          group: g, costTier: SCALE_TIERS[i % SCALE_TIERS.length],
          source: "layer-scale synthetic harness fixture",
          description: `Synthetic scale-test layer #${i} in group ${g}.`,
        });
      }
      return out;
    };
    const scaleResults = [];
    for (const n of [50, 100, 200]) {
      const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
      const page = await ctx.newPage();
      await page.route("**/*", (route) => {
        const u = route.request().url();
        if (u.startsWith(`http://127.0.0.1:${port}`)) return route.continue();
        return route.abort();
      });
      // registered AFTER the general handler above so it wins (Playwright
      // matches most-recently-registered route first) — overrides only the
      // registry response for this context, nothing else in the app.
      await page.route("**/api/data/layers", (route) => route.fulfill({
        contentType: "application/json",
        body: JSON.stringify({ layers: syntheticLayers(n), server_version: "scale-harness" }),
      }));
      const t0 = Date.now();
      await page.goto(`http://127.0.0.1:${port}${PAGES.data.route}`, { waitUntil: "load", timeout: 30000 });
      let tti = null;
      for (let i = 0; i < 60; i++) {
        const gone = await page.evaluate(() => !document.querySelector(".vt-map-skeleton"));
        if (gone) { tti = Date.now() - t0; break; }
        await page.waitForTimeout(100);
      }
      await page.waitForTimeout(800);
      const openT0 = Date.now();
      await page.click(".vt-map-fab", { timeout: 1500 }).catch(() => {});
      await page.waitForTimeout(200);
      const openMs = Date.now() - openT0;
      const defaultDom = await page.evaluate(() => {
        const panel = document.querySelector(".vt-layer-panel");
        return { rows: panel ? panel.querySelectorAll("[data-vt-layer]").length : -1,
                 totalNodes: panel ? panel.querySelectorAll("*").length : -1 };
      });
      // expand every group (as the existing self-see battery does) and use
      // "show all" on each capped group — the real worst case a user causes,
      // not a hypothetical one — then re-measure.
      for (let round = 0; round < SCALE_GROUPS.length + 2; round++) {
        const btn = page.locator('.vt-layer-group-head[aria-expanded="false"]').first();
        if (!(await btn.count())) break;
        await btn.click().catch(() => {});
        await page.waitForTimeout(80);
      }
      for (let round = 0; round < SCALE_GROUPS.length + 2; round++) {
        const more = page.locator(".vt-layer-showmore").first();
        if (!(await more.count())) break;
        await more.click().catch(() => {});
        await page.waitForTimeout(80);
      }
      const expandedAllDom = await page.evaluate(() => {
        const panel = document.querySelector(".vt-layer-panel");
        return panel ? panel.querySelectorAll("[data-vt-layer]").length : -1;
      });
      // NOTE on toggle state: these synthetic rows correctly render DISABLED
      // ("unwired" guard, datamap.tsx groupOf/unwired) — they have a
      // registry-native `group`/`costTier` but no real map-data fetch/render
      // effect exists for a `synth_N` id (that requires an actual client
      // deploy, not a registry edit alone), so the open-tab-skew guard
      // correctly refuses to render a functional-looking toggle that would
      // flip and paint nothing. This battery measures DOM/reachability at
      // scale, not the cost-badge (that's exercised below with the real
      // fixture's genuinely wired layers, where toggling isn't blocked).
      scaleResults.push({ n, tti, openMs, defaultRows: defaultDom.rows, defaultTotalNodes: defaultDom.totalNodes,
                          expandedAllRows: expandedAllDom });
      if (n === 200) await page.screenshot({ path: path.join(OUT, "data-scale-200.png") });
      await ctx.close();
    }
    console.log("\n[SCALE HARNESS] synthetic layer-count battery (BUILD ORDER 4 #2):");
    for (const r of scaleResults) {
      console.log(`  n=${r.n}: tti=${r.tti}ms panelOpen=${r.openMs}ms defaultRows=${r.defaultRows} defaultDomNodes=${r.defaultTotalNodes} expandedAllRows=${r.expandedAllRows}`);
    }
    // BUDGET ASSERTIONS: the default-open view must stay bounded regardless
    // of total registry size (proves collapse-by-default + GROUP_ROW_CAP
    // actually hold the DOM down, not just today's small registry), "show
    // all" must still reach every synthetic layer (self-see holds at scale,
    // not just for the 21-layer fixture), and TTI must not regress past the
    // existing 3000ms map-page gate used elsewhere in this file.
    const scaleFailures = [];
    for (const r of scaleResults) {
      if (r.tti == null) scaleFailures.push(`scale n=${r.n}: TTI never settled`);
      else if (r.tti > 3000) scaleFailures.push(`scale n=${r.n}: TTI ${r.tti}ms > 3000ms map-page gate at synthetic scale`);
      if (r.defaultRows > 30) scaleFailures.push(`scale n=${r.n}: default-open panel rendered ${r.defaultRows} rows > 30 budget — collapse-by-default/GROUP_ROW_CAP not holding`);
      if (r.expandedAllRows !== r.n) scaleFailures.push(`scale n=${r.n}: "show all" only reached ${r.expandedAllRows}/${r.n} layers — self-see broken at scale`);
    }
    results.push({ page: "data-scale", width: 1440, label: "layer-scale", screenshot: path.join(OUT, "data-scale-200.png"),
                   tti: null, perf: {}, failures: scaleFailures, warnings: [], info: { scaleResults } });
  }

  await browser.close();
  srv.close();

  let hard = 0;
  for (const r of results) {
    const status = r.failures.length ? "FAIL" : "PASS";
    if (r.failures.length) hard += r.failures.length;
    console.log(`\n[${status}] ${r.page} @ ${r.width}px (${r.label}) -> ${path.relative(ROOT, r.screenshot)}`);
    r.failures.forEach((f) => console.log(`  ✗ ${f}`));
    r.warnings.forEach((w) => console.log(`  ⚠ ${w}`));
    if (r.info.map) console.log(`  map: ${Math.round(r.info.map.w)}x${Math.round(r.info.map.h)} at y=${Math.round(r.info.map.y)}`);
    if (r.tti != null) console.log(`  tti: ${r.tti}ms | perf: median ${r.perf?.median}ms p95 ${r.perf?.p95}ms over ${r.perf?.frames} frames | rendered ${r.perf?.renderedAircraft ?? "?"} aircraft`);
  }
  writeFileSync(path.join(OUT, "results.json"), JSON.stringify(results, null, 2));
  console.log(`\n${hard} hard failure(s). Results: .visual/results.json`);
  process.exit(hard && !SOFT ? 1 : 0);
}

main().catch((e) => { console.error(e); process.exit(1); });
