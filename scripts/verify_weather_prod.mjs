#!/usr/bin/env node
/**
 * verify_weather_prod.mjs — proves the OWM temp/wind fields VISIBLY render
 * on production (DESIGN.md tile-layer pixel verification).
 *
 * For each canonical width: loads voltradeai.com/app#/data, waits for the
 * map, screenshots with fields OFF, toggles temperature+wind ON via their
 * real panel switches, waits for tiles, screenshots again, and asserts the
 * mean absolute pixel difference over the map canvas exceeds a floor —
 * an invisible layer produces ~0 diff (the 2026-07-04 defect); a rendering
 * field produces a large one.
 *
 * Usage: node scripts/verify_weather_prod.mjs [--base https://voltradeai.com]
 * Saves .visual/prod-weather-{w}-{off|on}.png; exits nonzero on failure.
 */
import { chromium } from "playwright";
import { mkdirSync } from "fs";
import path from "path";
import { fileURLToPath } from "url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const OUT = path.join(ROOT, ".visual");
const BASE = process.argv.includes("--base")
  ? process.argv[process.argv.indexOf("--base") + 1]
  : "https://voltradeai.com";
const WIDTHS = [
  { w: 390, h: 844, touch: true },
  { w: 768, h: 1024, touch: true },
  { w: 1440, h: 900, touch: false },
];
const MIN_MEAN_DIFF = 3; // 0-255 scale, over the whole canvas — invisible ≈ 0.1

mkdirSync(OUT, { recursive: true });
const browser = await chromium.launch({ args: ["--use-gl=swiftshader"] });
let failures = 0;

for (const vp of WIDTHS) {
  const ctx = await browser.newContext({ viewport: { width: vp.w, height: vp.h }, hasTouch: vp.touch, deviceScaleFactor: 1 });
  const page = await ctx.newPage();
  try {
    await page.goto(`${BASE}/app#/data`, { waitUntil: "load", timeout: 45000 });
    await page.waitForFunction("!!window.__vtMap", { timeout: 30000 });
    await page.waitForTimeout(4000); // base layers settle
    // zoom to a continental view where fields are meaningful
    await page.evaluate(() => new Promise((r) => {
      const m = window.__vtMap; m.once("idle", r);
      m.jumpTo({ center: [-96, 38], zoom: 4 });
    }));
    await page.waitForTimeout(1500);
    const grab = async (name) => {
      const p = path.join(OUT, `prod-weather-${vp.w}-${name}.png`);
      const el = await page.$(".vt-map-canvas canvas");
      await el.screenshot({ path: p });
      return p;
    };
    const offShot = await grab("off");
    // open panel (phone: via FAB), expand Base group if collapsed, toggle both fields
    await page.click(".vt-map-fab", { timeout: 2000 }).catch(() => {});
    await page.waitForTimeout(300);
    for (let i = 0; i < 4; i++) {
      const btn = page.locator('.vt-layer-group-head[aria-expanded="false"]').first();
      if (!(await btn.count())) break;
      await btn.click().catch(() => {});
      await page.waitForTimeout(150);
    }
    for (const id of ["weather_temp", "weather_wind"]) {
      await page.click(`[data-vt-layer="${id}"] [role="switch"]`, { timeout: 5000 });
      await page.waitForTimeout(400);
    }
    // wait for tiles to fetch + render
    await page.waitForTimeout(9000);
    await page.click('.vt-layer-panel [aria-label="Collapse layers panel"]').catch(() => {});
    await page.waitForTimeout(600);
    const onShot = await grab("on");

    // mean absolute pixel diff via in-page canvas comparison
    const diff = await page.evaluate(async ([a, b]) => {
      const load = (src) => new Promise((res) => { const i = new Image(); i.onload = () => res(i); i.src = src; });
      const [ia, ib] = await Promise.all([load(a), load(b)]);
      const c = document.createElement("canvas");
      c.width = ia.width; c.height = ia.height;
      const g = c.getContext("2d");
      g.drawImage(ia, 0, 0);
      const da = g.getImageData(0, 0, c.width, c.height).data;
      g.clearRect(0, 0, c.width, c.height);
      g.drawImage(ib, 0, 0);
      const db = g.getImageData(0, 0, c.width, c.height).data;
      let sum = 0, n = 0;
      for (let i = 0; i < da.length; i += 4) {
        sum += Math.abs(da[i] - db[i]) + Math.abs(da[i + 1] - db[i + 1]) + Math.abs(da[i + 2] - db[i + 2]);
        n += 3;
      }
      return sum / n;
    }, await Promise.all([offShot, onShot].map(async (p) => {
      const { readFileSync } = await import("fs");
      return "data:image/png;base64," + readFileSync(p).toString("base64");
    })));

    const ok = diff >= MIN_MEAN_DIFF;
    console.log(`[${ok ? "PASS" : "FAIL"}] ${vp.w}px — mean pixel diff ${diff.toFixed(2)} (floor ${MIN_MEAN_DIFF}) — ${path.relative(ROOT, onShot)}`);
    if (!ok) failures++;
  } catch (e) {
    console.log(`[FAIL] ${vp.w}px — driver error: ${e?.message || e}`);
    failures++;
  }
  await ctx.close();
}
await browser.close();
process.exit(failures ? 1 : 0);
