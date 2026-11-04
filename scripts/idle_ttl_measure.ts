/**
 * idle_ttl_measure.ts — how much heap each A3-managed structure holds, what a
 * cold load costs, and what the FIRST request after an idle drop costs
 * (memory audit 2026-11-03, fix A3; server/idleTtl.ts).
 *
 *   node --expose-gc --import tsx scripts/idle_ttl_measure.ts
 *
 * Per dataset: gc → load (cold_ms) → gc → held_mb → forced idle drop via the
 * real sweep (sweepIdleTtl far-future) → gc → released_mb → load again
 * (reload_ms = the latency the first dashboard request after a drop pays) →
 * drop. Rows marked "not in A3" are lazy tables measured for the record only
 * (poller-hot: the TTL would never fire). Then one synthetic entry of the two
 * biggest map caches (a zoom-9 window read; a 12k-row world aircraft body)
 * and the Bloom cache at its cap. heapUsed deltas are after 3× gc; the Bloom
 * filters are Uint8Array backing stores and show up in arrayBuffers (RSS),
 * not heapUsed.
 */
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import express from "express";
import type { AddressInfo } from "node:net";
import { performance } from "node:perf_hooks";
import { sweepIdleTtl } from "../server/idleTtl";

declare const global: { gc?: () => void };
const gc = () => { for (let i = 0; i < 3; i++) global.gc?.(); };
const MB = (n: number) => (n / (1 << 20)).toFixed(1);
const heap = () => { gc(); const m = process.memoryUsage(); return { h: m.heapUsed, ab: m.arrayBuffers }; };
const FAR = () => Date.now() + 3600_000;

interface Row { name: string; cold_ms: number; held_mb: string; released_mb: string; reload_ms: number | string; ab_mb: string }
const rows: Row[] = [];
let keep: unknown = null; // retains the loaded value between gc points

function measure(name: string, load: () => unknown, release: (() => void) | null) {
  keep = null;
  const b0 = heap();
  const t0 = performance.now();
  keep = load();
  const cold = performance.now() - t0;
  const b1 = heap();
  let released: number | string = "n/a", reload: number | string = "n/a";
  if (release) {
    release(); keep = null;
    const b2 = heap();
    released = MB(b1.h - b2.h);
    const t1 = performance.now();
    keep = load();
    reload = Math.round(performance.now() - t1);
    release(); keep = null; gc();
  }
  rows.push({ name, cold_ms: Math.round(cold), held_mb: MB(b1.h - b0.h), released_mb: released, reload_ms: reload, ab_mb: MB(b1.ab - b0.ab) });
}

async function main() {
  const gemChem = await import("../server/gemChemicals");
  const gemCmf = await import("../server/gemCoalMineFeatures");
  const gemCt = await import("../server/gemCoalTerminals");
  const gemIom = await import("../server/gemIronOreMines");
  const gemIsp = await import("../server/gemIronSteelPlants");
  const gemLng = await import("../server/gemLngCarriers");
  const gemMeth = await import("../server/gemMethane");
  const gemAssets = await import("../server/gemMethaneAssets");
  const gemProx = await import("../server/gemMethaneProximity");
  const gemSrm = await import("../server/gemSteelRawMaterials");
  const airports = await import("../server/airportsIndex");
  const atc = await import("../server/atcFrequencies");
  const navFixes = await import("../server/navFixes");
  const navAirways = await import("../server/navAirways");

  const drop = () => { sweepIdleTtl(FAR()); };
  measure("gemChemicals", gemChem.cachedGemChemicals, drop);
  measure("gemCoalMineFeatures", gemCmf.cachedGemCoalMineFeatures, drop);
  measure("gemCoalTerminals", gemCt.cachedGemCoalTerminals, drop);
  measure("gemIronOreMines", gemIom.cachedGemIronOreMines, drop);
  measure("gemIronSteelPlants(+units)", gemIsp.cachedGemIronSteelPlants, drop);
  measure("gemLngShipyards", gemLng.cachedGemLngShipyards, drop);
  measure("gemMethane(plumes)", gemMeth.cachedGemMethane, drop);
  measure("gemMethaneAssets", gemAssets.cachedGemAssets, drop);
  measure("gemMethaneProximity(all 3 inputs)", gemProx.cachedGemMethaneProximity, drop);
  measure("gemSteelRawMaterials", gemSrm.cachedGemSteelRawMaterials, drop);
  measure("steelRawMaterialsGeo(+balances)", gemSrm.cachedSteelRawMaterialsGeoJSON, drop);
  measure("airportsIndex", () => airports.loadAirports(), drop);
  measure("atcFrequencies [intent-bus-hot, not in A3]", () => atc.loadAtcData(), atc._resetAtcData);
  measure("navFixes [SWIM-hot, not in A3]", () => navFixes.loadNavFixes(), null);
  measure("navAirways [SWIM-hot, not in A3]", () => navAirways.loadNavAirways(), null);

  // bloom: 2000 × 32 KB Uint8Array (the cap) — backing stores are off-heap (arrayBuffers)
  measure("bloomCache@cap (2000×32KB, synthetic)", () => { const a: Uint8Array[] = []; for (let i = 0; i < 2000; i++) a.push(new Uint8Array(1 << 15)); return a; }, () => undefined);

  // one windowCache entry: 300 hexes × 200 pts in one hour, zoom 9 (no decimation), default caps
  const { archiveAircraftAt } = await import("../server/datacoreArchive");
  const { readWindow } = await import("../server/aircraftWindow");
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-measure-"));
  const T0 = Math.floor(Date.parse("2026-08-10T12:00:00Z") / 1000);
  const fixes: Array<Record<string, unknown>> = [];
  for (let h = 0; h < 300; h++) {
    const hex = h.toString(16).padStart(6, "a");
    for (let k = 0; k < 200; k++) {
      fixes.push({ tSec: T0 + k * 15, icao24: hex, registration: `N${hex}`, type: "B738", callsign: `C${hex}`,
        lat: 40 + (h % 50) * 0.05 + k * 0.001, lon: -100 + Math.floor(h / 50) * 0.5 + k * 0.001, altitude_m: 9000 + k, on_ground: false,
        velocity_ms: null, heading: null, category: null });
    }
  }
  archiveAircraftAt(fixes as unknown as Parameters<typeof archiveAircraftAt>[0], base);
  let winRes: unknown = null;
  const w0 = heap(); const tw = performance.now();
  winRes = await readWindow({ bbox: { w: -110, s: 35, e: -90, n: 45 }, fromSec: T0, toSec: T0 + 3600, zoom: 9, baseDir: base, closeApproaches: false });
  const wms = Math.round(performance.now() - tw); const w1 = heap();
  const r = winRes as { total_points: number; hexes: unknown[] };
  rows.push({ name: `windowCache entry (${r.hexes.length} hexes, ${r.total_points} pts)`, cold_ms: wms, held_mb: MB(w1.h - w0.h), released_mb: "n/a", reload_ms: wms, ab_mb: "0" });
  rows.push({ name: "  -> its JSON body bytes", cold_ms: 0, held_mb: MB(JSON.stringify(winRes).length), released_mb: "n/a", reload_ms: "n/a", ab_mb: "0" });
  winRes = null;
  fs.rmSync(base, { recursive: true, force: true });

  // one bodyCache entry: world payload at ~12k rows through the real route
  const { registerGlobalAircraftRoutes } = await import("../server/globalAircraft");
  const { publishFixes } = await import("../server/aircraftFixBus");
  const app = express();
  const archiveDir = fs.mkdtempSync(path.join(os.tmpdir(), "vt-measure-g-"));
  const h = registerGlobalAircraftRoutes(app, { env: { GLOBAL_SWEEP_ENABLED: "0", GLOBAL_SWEEP_ARCHIVE: "0" } as NodeJS.ProcessEnv,
    fetchImpl: (async () => { throw new Error("no net"); }) as unknown as typeof fetch, sites: [], baseDir: archiveDir, readFree: () => 50 * (1 << 30) });
  const srv = app.listen(0);
  try {
    const now = Date.now();
    const acs: Array<Record<string, unknown>> = [];
    for (let i = 0; i < 12000; i++) {
      const hex = i.toString(16).padStart(6, "0");
      acs.push({ icao24: hex, callsign: `CS${hex}`, lat: -60 + (i % 1200) * 0.1, lon: -180 + Math.floor(i / 1200) * 30, altitude_m: 10000 + i % 3000, on_ground: false,
        velocity_ms: 230, heading: 45, type: "B738", category: "A3", registration: `N${hex}`, seen_pos: 1 });
    }
    publishFixes({ provider: "adsblol", origin: "sweep-type", aircraft: acs as unknown as Parameters<typeof publishFixes>[0]["aircraft"], fetchedAt: now, upstreamNowMs: now });
    const port = (srv.address() as AddressInfo).port;
    const tb = performance.now();
    const body = await (await fetch(`http://127.0.0.1:${port}/api/data/aircraft/global`)).text();
    const bms = Math.round(performance.now() - tb);
    rows.push({ name: "bodyCache entry (12k-row world body, string bytes)", cold_ms: bms, held_mb: MB(body.length), released_mb: "n/a", reload_ms: bms, ab_mb: "0" });
  } finally { h.stop(); srv.close(); fs.rmSync(archiveDir, { recursive: true, force: true }); }

  console.table(rows);
}
main().then(() => process.exit(0)).catch((e: unknown) => { console.error(e); process.exit(1); });
