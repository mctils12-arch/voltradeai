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
const PAGES = { data: "/app#/data" };
const only = process.argv.includes("--page")
  ? process.argv[process.argv.indexOf("--page") + 1]
  : null;

// ── deterministic API fixtures ─────────────────────────────────────────────
const FIXTURES = {
  "/api/auth/me": { authenticated: false },
  "/api/data/layers": {
    layers: [
      { id: "imagery", name: "Satellite imagery", kind: "raw", status: "live", source: "Esri World Imagery", description: "Base imagery." },
      { id: "aircraft", name: "Live aircraft (ADS-B)", kind: "raw", status: "live", source: "adsb.lol/airplanes.live", description: "Live aircraft." },
      { id: "vessels", name: "Live vessels (AIS)", kind: "raw", status: "awaiting_key", source: "aisstream.io", description: "Needs AISSTREAM_KEY." },
      { id: "trains", name: "Live trains (rail)", kind: "raw", status: "live", source: "Digitraffic FI + Entur NO", description: "FI+NO launch coverage." },
      { id: "sites", name: "Strategic sites", kind: "raw", status: "live", source: "datacore/sites", description: "Reference sites." },
      { id: "powerplants", name: "US power plants", kind: "raw", status: "live", source: "WRI GPPD (CC BY 4.0)", description: "US plants by fuel." },
      { id: "insider", name: "Insider transactions (Form 4)", kind: "raw", status: "live", source: "SEC EDGAR", description: "Recent Form 4 filings as filed." },
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

function startServer() {
  return new Promise((resolve) => {
    const srv = createServer((req, res) => {
      const [u, qs] = (req.url || "/").split("?");
      const fx = Object.keys(FIXTURES).find((k) => u === k || u.startsWith(k + "/"));
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
const CHECKS_SNIPPET = (width, touch) => `(() => {
  const out = { failures: [], warnings: [], info: {} };
  const vw = window.innerWidth, vh = window.innerHeight;

  // 1. no horizontal overflow caused by the page
  if (document.documentElement.scrollWidth > vw + 1) {
    out.failures.push("horizontal overflow: scrollWidth " + document.documentElement.scrollWidth + " > viewport " + vw);
  }

  // 2. map root fills its viewport region (marker: [data-vt-map])
  const map = document.querySelector('[data-vt-map]');
  if (!map) {
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
  for (const [name, route] of Object.entries(PAGES)) {
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
      const perf = await page.evaluate(async () => {
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
        try { rendered = map.queryRenderedFeatures({ layers: ["aircraft-sym"] }).length; } catch {}
        return { frames: sorted.length, median: Math.round(q(0.5)), p95: Math.round(q(0.95)),
                 max: Math.round(sorted[sorted.length - 1] || 0), renderedAircraft: rendered };
      });
      const shot = path.join(OUT, `${name}-${vp.w}.png`);
      await page.screenshot({ path: shot });
      const checks = await page.evaluate(CHECKS_SNIPPET(vp.w, vp.touch));
      checks.failures.push(...errors);
      // Perf budget (headless regression guards; on-device budget in DESIGN.md)
      if (tti == null) checks.failures.push("TTI: skeleton never cleared (>15s)");
      else if (tti > 12000) checks.failures.push(`TTI ${tti}ms > 12s headless guard`);
      if (perf.error) checks.failures.push("perf: " + perf.error);
      else {
        // Software-GL (SwiftShader) is 10-50x slower than any real GPU —
        // REGRESSION GUARDS calibrated to headless, not the on-device budget
        // (DESIGN.md owns that; the S24 is the acceptance device). MEDIAN =
        // steady-state smoothness after a warm-up window; p95 spikes are
        // data-upload hitches, warned not failed under software rasterization.
        if (perf.median > 300) checks.failures.push(`perf: median frame ${perf.median}ms > 300ms headless guard (steady-state jank) @10k features`);
        if (perf.p95 > 700) checks.warnings.push(`perf: p95 frame ${perf.p95}ms (upload-hitch spikes)`);
        if (!perf.renderedAircraft) checks.warnings.push("perf: no aircraft features rendered in viewport sample");
      }
      results.push({ page: name, width: vp.w, label: vp.label, screenshot: shot, tti, perf, ...checks });
      await ctx.close();
    }
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
