#!/usr/bin/env node
// M4 per-layer profile: pan-frame medians at 390px (SwiftShader ~ mid-range
// phone proxy) with layer combinations, to find the worst offender.
import { createServer } from "http";
import { readFileSync, existsSync } from "fs";
import path from "path";
import { chromium } from "playwright";

const ROOT = "/home/user/voltradeai";
const DIST = path.join(ROOT, "dist", "public");

// 10k deterministic aircraft (same as harness)
const aircraft = [];
let seed = 42;
const rnd = () => (seed = (seed * 1103515245 + 12345) % 2 ** 31) / 2 ** 31;
for (let i = 0; i < 10000; i++) {
  aircraft.push({
    icao24: i.toString(16).padStart(6, "0"), callsign: `TST${i}`, origin_country: "",
    lon: -170 + rnd() * 340, lat: -60 + rnd() * 130,
    altitude_m: Math.round(rnd() * 12000), on_ground: false,
    velocity_ms: Math.round(rnd() * 250), heading: Math.round(rnd() * 360),
    type: ["B738", "A320", "C172", "AT76", "EC35"][i % 5], category: "A3",
  });
}
const plants = JSON.parse(readFileSync(path.join(ROOT, "datacore/powerplants/us_power_plants.json"), "utf8"));
const sites = JSON.parse(readFileSync(path.join(ROOT, "datacore/sites/strategic_sites.json"), "utf8"));

const scenarios = {
  all:        { aircraft: true, powerplants: true, sites: true },
  no_aircraft:{ aircraft: false, powerplants: true, sites: true },
  no_plants:  { aircraft: true, powerplants: false, sites: true },
  only_base:  { aircraft: false, powerplants: false, sites: false },
};

const FIX = (on) => ({
  "/api/auth/me": { authenticated: false },
  "/api/data/layers": { layers: [
    { id: "imagery", name: "Satellite imagery", kind: "raw", status: "live", source: "Esri", description: "." },
    { id: "aircraft", name: "Live aircraft (ADS-B)", kind: "raw", status: "live", source: "adsb.lol", description: "." },
    { id: "sites", name: "Strategic sites", kind: "raw", status: "live", source: "datacore", description: "." },
    { id: "powerplants", name: "US power plants", kind: "raw", status: "live", source: "WRI", description: "." },
  ]},
  "/api/data/aircraft": on.aircraft ? { kind: "raw", time: 1, count: aircraft.length, coverage: "full", aircraft } : { kind: "raw", time: 1, count: 0, coverage: "full", aircraft: [] },
  "/api/data/powerplants": on.powerplants ? { kind: "raw", ...plants } : { kind: "raw", source: plants.source, fuels: plants.fuels, count: 0, plants: [] },
  "/api/data/sites": on.sites ? { kind: "raw", ...sites } : { kind: "raw", categories: {}, sites: [] },
  "/api/data/insider": { kind: "raw", filings: [], count: 0 },
  "/api/health": { status: "ok", checks: {} },
});
const MIME = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".json": "application/json", ".woff2": "font/woff2" };

let FIXTURES = FIX(scenarios.all);
const srv = await new Promise((resolve) => {
  const s = createServer((req, res) => {
    const [u, qs] = (req.url || "/").split("?");
    const fx = Object.keys(FIXTURES).find((k) => u === k);
    if (fx) {
      res.writeHead(200, { "content-type": "application/json" });
      if (fx === "/api/data/aircraft" && (qs || "").includes("since=1"))
        return res.end(JSON.stringify({ unchanged: true, time: 1, count: FIXTURES[fx].count }));
      return res.end(JSON.stringify(FIXTURES[fx]));
    }
    if (u.startsWith("/api/")) { res.writeHead(200, { "content-type": "application/json" }); return res.end("{}"); }
    let fp = path.join(DIST, u === "/" || u === "/app" ? "index.html" : u);
    if (!existsSync(fp)) fp = path.join(DIST, "index.html");
    try { const b = readFileSync(fp); res.writeHead(200, { "content-type": MIME[path.extname(fp)] || "application/octet-stream" }); res.end(b); }
    catch { res.writeHead(404); res.end(); }
  });
  s.listen(0, "127.0.0.1", () => resolve(s));
});
const port = srv.address().port;

const browser = await chromium.launch({
  executablePath: "/opt/pw-browsers/chromium",
  args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist", "--no-sandbox"],
});

for (const [name, on] of Object.entries(scenarios)) {
  FIXTURES = FIX(on);
  const page = await browser.newPage({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 1 });
  await page.route(/^https?:\/\/(?!127\.0\.0\.1|localhost).*/, (r) => r.abort());
  await page.goto(`http://127.0.0.1:${port}/app#/data`, { waitUntil: "domcontentloaded" });
  await page.waitForFunction("!!window.__vtMap", { timeout: 20000 }).catch(() => {});
  await page.waitForTimeout(3500);
  // measured pan window at a mid zoom where everything is visible
  const frames = await page.evaluate(async () => {
    const map = window.__vtMap;
    if (!map) return null;
    map.jumpTo({ center: [-40, 30], zoom: 2.2 });
    await new Promise((r) => setTimeout(r, 1200));
    // warm-up pan
    map.panBy([60, 0], { duration: 300 }); await new Promise((r) => setTimeout(r, 500));
    const times = [];
    let last = performance.now();
    const onRender = () => { const now = performance.now(); times.push(now - last); last = now; };
    map.on("render", onRender);
    for (const [dx, dy] of [[120, 0], [-120, 40], [0, -80], [-100, 40], [100, 0]]) {
      map.panBy([dx, dy], { duration: 350 });
      await new Promise((r) => setTimeout(r, 550));
    }
    map.off("render", onRender);
    const f = times.filter((t) => t > 1);
    f.sort((a, b) => a - b);
    return { n: f.length, median: f[Math.floor(f.length / 2)], p95: f[Math.floor(f.length * 0.95)] };
  });
  console.log(name.padEnd(12), JSON.stringify(frames));
  await page.close();
}
await browser.close();
srv.close();
