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
      { id: "aircraft", name: "Live aircraft (ADS-B)", kind: "raw", status: "live", source: "OpenSky/adsb.lol", description: "Live aircraft." },
      { id: "vessels", name: "Live vessels (AIS)", kind: "raw", status: "awaiting_key", source: "aisstream.io", description: "Needs AISSTREAM_KEY." },
      { id: "sites", name: "Strategic sites", kind: "raw", status: "live", source: "datacore/sites", description: "Reference sites." },
      { id: "tank_fill", name: "Tank-fill % (Sentinel-2)", kind: "signal", status: "planned", source: "Copernicus", description: "Gate-2 locked." },
    ],
  },
  "/api/data/aircraft": {
    source: "fixture", kind: "raw", time: 0, count: 3,
    aircraft: [
      { icao24: "t1", callsign: "TEST1", origin_country: "US", lon: -96.7, lat: 36.2, altitude_m: 10000, on_ground: false, velocity_ms: 230, heading: 90 },
      { icao24: "t2", callsign: "TEST2", origin_country: "US", lon: -97.4, lat: 35.5, altitude_m: 2000, on_ground: false, velocity_ms: 120, heading: 180 },
      { icao24: "t3", callsign: "TEST3", origin_country: "US", lon: -96.0, lat: 35.9, altitude_m: null, on_ground: true, velocity_ms: 5, heading: 0 },
    ],
  },
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
  "/api/health": { status: "ok", checks: {} },
};

const MIME = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".png": "image/png", ".svg": "image/svg+xml", ".json": "application/json", ".woff2": "font/woff2", ".ico": "image/x-icon" };

function startServer() {
  return new Promise((resolve) => {
    const srv = createServer((req, res) => {
      const u = (req.url || "/").split("?")[0];
      const fx = Object.keys(FIXTURES).find((k) => u === k || u.startsWith(k + "/"));
      if (fx) {
        res.writeHead(200, { "content-type": "application/json" });
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
  let browser;
  try {
    browser = await chromium.launch({ headless: true });
  } catch {
    browser = await chromium.launch({
      headless: true,
      executablePath: existsSync(exePath) ? exePath : undefined,
      args: ["--use-angle=swiftshader"],
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
        deviceScaleFactor: 2,
      });
      const page = await ctx.newPage();
      const errors = [];
      page.on("pageerror", (e) => errors.push("pageerror: " + e.message));
      await page.goto(`http://127.0.0.1:${port}${route}`, { waitUntil: "load", timeout: 30000 });
      await page.waitForTimeout(6000); // map init + fixture layers
      const shot = path.join(OUT, `${name}-${vp.w}.png`);
      await page.screenshot({ path: shot });
      const checks = await page.evaluate(CHECKS_SNIPPET(vp.w, vp.touch));
      checks.failures.push(...errors);
      results.push({ page: name, width: vp.w, label: vp.label, screenshot: shot, ...checks });
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
  }
  writeFileSync(path.join(OUT, "results.json"), JSON.stringify(results, null, 2));
  console.log(`\n${hard} hard failure(s). Results: .visual/results.json`);
  process.exit(hard && !SOFT ? 1 : 0);
}

main().catch((e) => { console.error(e); process.exit(1); });
