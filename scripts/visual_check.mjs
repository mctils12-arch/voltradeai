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
  developers: { route: "/developers", map: false },
  landing: { route: "/", map: false },
};
const only = process.argv.includes("--page")
  ? process.argv[process.argv.indexOf("--page") + 1]
  : null;

// ── deterministic API fixtures ─────────────────────────────────────────────
const FIXTURES = {
  "/api/auth/me": { authenticated: false },
  "/api/data/layers": {
    layers: [
      { id: "imagery", name: "Satellite imagery", kind: "raw", status: "live", source: "Esri World Imagery", description: "Base imagery." },
      { id: "terrain", name: "Terrain (hillshade)", kind: "raw", status: "live", source: "Mapterhorn (© Mapterhorn)", description: "Global hillshade, off by default." },
      { id: "weather", name: "Weather radar (US)", kind: "raw", status: "live", field: true, source: "NOAA nowCOAST (public domain)", description: "US radar mosaic, off by default." },
      { id: "weather_temp", name: "Temperature (global)", kind: "raw", status: "live", field: true, source: "OpenWeatherMap (© OpenWeatherMap)", description: "Global temp field, off by default." },
      { id: "weather_wind", name: "Wind (global)", kind: "raw", status: "live", field: true, source: "OpenWeatherMap (© OpenWeatherMap)", description: "Global wind field, off by default." },
      { id: "aircraft", name: "Live aircraft (ADS-B)", kind: "raw", status: "live", source: "adsb.lol/airplanes.live", description: "Live aircraft." },
      { id: "vessels", name: "Live vessels (AIS)", kind: "raw", status: "awaiting_key", source: "aisstream.io", description: "Needs AISSTREAM_KEY." },
      { id: "trains", name: "Live trains (rail)", kind: "raw", status: "live", source: "Digitraffic FI + Entur NO", description: "FI+NO launch coverage." },
      { id: "sites", name: "Strategic sites", kind: "raw", status: "live", source: "datacore/sites", description: "Reference sites." },
      { id: "powerplants", name: "US power plants", kind: "raw", status: "live", source: "WRI GPPD (CC BY 4.0)", description: "US plants by fuel." },
      { id: "insider", name: "Insider transactions (Form 4)", kind: "raw", status: "live", source: "SEC EDGAR", description: "Recent Form 4 filings as filed." },
      { id: "earnings", name: "Earnings language (8-K)", kind: "raw", status: "live", source: "SEC EDGAR", description: "As-filed 8-K Item 2.02 results/guidance releases." },
      { id: "portdwell", name: "Port dwell (arrivals/departures)", kind: "raw", status: "live", source: "Own AIS archive + verified port geofences", description: "Per-port dwell stats; lower bounds; anomaly SIGNAL gate-2 locked." },
      { id: "fires", name: "Active fires (VIIRS)", kind: "raw", status: "awaiting_key", source: "NASA FIRMS / LANCE", description: "Needs NASA_FIRMS_MAP_KEY." },
      { id: "rivergauges", name: "River gauges (barge corridor)", kind: "raw", status: "live", source: "USGS NWIS (public domain)", description: "Live stage/discharge at 14 barge-corridor gauges." },
      { id: "surfacewater", name: "Surface water (1984–2021)", kind: "raw", status: "live", field: true, source: "EC JRC/Google GSW v2021", description: "Static water occurrence, off by default." },
      { id: "forest", name: "Forest cover (2020)", kind: "raw", status: "live", field: true, source: "EC JRC GFC2020 via GFW", description: "Static forest extent, off by default." },
      { id: "boundaries", name: "Country borders", kind: "raw", status: "live", source: "Natural Earth 1:110m (public domain)", description: "Reference borders, off by default." },
      { id: "tank_fill", name: "Tank-fill % (Sentinel-2)", kind: "signal", status: "planned", source: "Copernicus", description: "Gate-2 locked." },
    ],
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
  "/api/data/rivergauges": {
    kind: "raw", source: "USGS Water Services (fixture)", time: 1, count: 2,
    gauges: [
      { site: "07010000", name: "Mississippi River at St. Louis, MO", param: "00065", d: "2026-07-05T00:15:00.000-05:00", v: 15.13, q: "P", lat: 38.62889, lon: -90.17972 },
      { site: "07032000", name: "Mississippi River at Memphis, TN", param: "00060", d: "2026-07-05T00:15:00.000-06:00", v: 512000, q: "P", lat: 35.12278, lon: -90.0775 },
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
      { path: "/api/v1/stats/portdwell", params: "-", desc: "Per-port dwell statistics." },
      { path: "/api/v1/stats/shadow", params: "-", desc: "Dark-ship RAW statistics." },
      { path: "/api/v1/stats/archive", params: "-", desc: "Archive growth metadata." },
      { path: "/api/v1/meta", params: "-", desc: "This document." },
    ],
    coming_gated: ["entity timelines (Graph v1)", "tank-fill readings (gate 2)"],
    limits: { dev: { perMinute: 60, perDay: 10000 } },
    license_marks: {
      "tracks/aircraft": { license: "ODbL 1.0 share-alike (fixture)", attribution: "adsb.lol", resell: "share-alike" },
      "stats/archive": { license: "VolTradeAI operational metadata", attribution: "VolTradeAI", resell: "ok" },
    },
    disclaimer: "Data as-is; not for safety-of-life use (fixture).",
  },
  "/api/data/archive/stats": { kinds: { aircraft: { days: 2, samples: 96 }, vessels: { days: 2, samples: 96 } }, totalBytes: 12582912 },
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

function startServer() {
  return new Promise((resolve) => {
    const srv = createServer((req, res) => {
      const [u, qs] = (req.url || "/").split("?");
      if (u.startsWith("/api/data/wxtile/")) {
        res.writeHead(200, { "content-type": "image/png" });
        return res.end(WX_TILE_PNG);
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
          for (const sel of [".maplibregl-ctrl-zoom-in", ".maplibregl-ctrl-zoom-out", "[data-vt-fullscreen]"]) {
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
          // flip back to leave the page in its default state
          await sw.click({ timeout: 4000 }).catch(() => {});
          await page.waitForTimeout(150);
        }
        checks.failures.push(...desyncs);
        if (!desyncs.length) checks.info.toggleConsistency = `${toggleables.length} layers toggled clean`;
        if (vp.touch) {} // (1440 only — no panel-state restore needed)
      } catch (e) {
        if (!e?.skip) checks.failures.push("toggle-consistency: driver error — " + (e?.message || e));
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
          for (const sel of [".maplibregl-ctrl-zoom-in", ".maplibregl-ctrl-zoom-out", "[data-vt-fullscreen]"]) {
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
