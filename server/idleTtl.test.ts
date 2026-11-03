/**
 * idleTtl.test.ts — idle TTL for the dashboard caches and lazy datasets
 * (memory audit 2026-11-03, fix A3). Pure-module tests with an injected clock,
 * then the contract that matters: cold → warm → forced idle drop → reload gives
 * BYTE-IDENTICAL results for every structure class A3 touches — a routes.ts
 * cache pattern against the real readWindow, the world-aircraft body through
 * the real Express route, the trips Bloom cache, the airports grid and two GEM
 * datasets (one of them a two-level join). Finally a source ratchet pinning
 * that every A3 structure stays wired through idleTtl.ts.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import express from "express";
import type { AddressInfo } from "node:net";
import {
  IDLE_TTL_FLAG, IDLE_TTL_DEFAULT_MIN, IDLE_TTL_MIN_MIN, IDLE_TTL_MAX_MIN, IDLE_SWEEP_INTERVAL_MS,
  resolveIdleTtl, idleTtlMs, idleTtlMap, lazyDataset, idleRelease, sweepIdleTtl, idleTtlStats,
  _resetIdleTtlForTests,
} from "./idleTtl";
import { readHeapGauges } from "./heapDiag";
import { fullTrackAsync, _resetBloomCache, bloomCacheSize } from "./aircraftTrips";
import { loadAirports, nearestAirport, airportByIdent, _resetAirports } from "./airportsIndex";
import { cachedGemChemicals, _resetGemChemicalsCacheForTests } from "./gemChemicals";
import {
  cachedSteelRawMaterialsGeoJSON, _resetSteelRawMaterialsGeoCacheForTests, _resetGemSteelRawMaterialsCacheForTests,
} from "./gemSteelRawMaterials";
import { readWindow, type WindowResult } from "./aircraftWindow";
import { archiveAircraftAt, type AircraftPoint } from "./datacoreArchive";
import { registerGlobalAircraftRoutes } from "./globalAircraft";
import { publishFixes } from "./aircraftFixBus";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FAR_FUTURE = () => Date.now() + 3600_000; // > IDLE_TTL_MAX_MIN, so any flag value in range drops

// ── the flag ────────────────────────────────────────────────────────────────
test("flag: unset/empty = 12 min (default ON); 0 = disabled; 10..15 honored; anything else clamped or default", () => {
  assert.equal(IDLE_TTL_DEFAULT_MIN, 12);
  assert.equal(IDLE_TTL_MIN_MIN, 10);
  assert.equal(IDLE_TTL_MAX_MIN, 15);
  assert.equal(IDLE_SWEEP_INTERVAL_MS, 60_000);
  assert.deepEqual(resolveIdleTtl({}), { ttlMs: 12 * 60_000, source: "default", raw: undefined });
  assert.equal(resolveIdleTtl({ [IDLE_TTL_FLAG]: "" }).source, "default");
  assert.equal(resolveIdleTtl({ [IDLE_TTL_FLAG]: "   " }).ttlMs, 12 * 60_000);
  assert.deepEqual(resolveIdleTtl({ [IDLE_TTL_FLAG]: "0" }), { ttlMs: 0, source: "disabled", raw: "0" });
  assert.deepEqual(resolveIdleTtl({ [IDLE_TTL_FLAG]: "10" }), { ttlMs: 600_000, source: "env", raw: "10" });
  assert.deepEqual(resolveIdleTtl({ [IDLE_TTL_FLAG]: "15" }), { ttlMs: 900_000, source: "env", raw: "15" });
  assert.equal(resolveIdleTtl({ [IDLE_TTL_FLAG]: "12.5" }).ttlMs, 750_000);
  assert.deepEqual(resolveIdleTtl({ [IDLE_TTL_FLAG]: "5" }), { ttlMs: 600_000, source: "clamped", raw: "5" });
  assert.deepEqual(resolveIdleTtl({ [IDLE_TTL_FLAG]: "30" }), { ttlMs: 900_000, source: "clamped", raw: "30" });
  assert.equal(resolveIdleTtl({ [IDLE_TTL_FLAG]: "-3" }).ttlMs, 600_000, "negative clamps to the floor, never disables");
  assert.deepEqual(resolveIdleTtl({ [IDLE_TTL_FLAG]: "abc" }), { ttlMs: 12 * 60_000, source: "invalid", raw: "abc" });
  assert.equal(resolveIdleTtl({ [IDLE_TTL_FLAG]: "1e400" }).source, "invalid", "Infinity is not a TTL");
});

test("flag: idleTtlMs() reads the environment once per process (reset re-reads)", () => {
  const saved = process.env[IDLE_TTL_FLAG];
  try {
    _resetIdleTtlForTests();
    process.env[IDLE_TTL_FLAG] = "0";
    assert.equal(idleTtlMs(), 0);
    process.env[IDLE_TTL_FLAG] = "11";
    assert.equal(idleTtlMs(), 0, "memoized: a later env change is not seen until reset");
    _resetIdleTtlForTests();
    assert.equal(idleTtlMs(), 660_000);
  } finally {
    if (saved === undefined) delete process.env[IDLE_TTL_FLAG]; else process.env[IDLE_TTL_FLAG] = saved;
    _resetIdleTtlForTests();
  }
});

// ── idleTtlMap ──────────────────────────────────────────────────────────────
test("idleTtlMap: Map surface and insertion order are the Map's own; the callers' eviction idioms see the same picture", () => {
  let t = 1_000_000;
  const now = () => t;
  const m = idleTtlMap<string, { at: number }>("test.map.surface", { ttlMs: 10_000, now });
  m.set("a", { at: 1 }); t += 10;
  m.set("b", { at: 2 }); t += 10;
  m.set("c", { at: 3 });
  assert.equal(m.size, 3);
  assert.deepEqual(Array.from(m.keys()), ["a", "b", "c"]);
  assert.deepEqual(Array.from(m.values()), [{ at: 1 }, { at: 2 }, { at: 3 }]);
  assert.deepEqual(Array.from(m.entries()), [["a", { at: 1 }], ["b", { at: 2 }], ["c", { at: 3 }]]);
  assert.deepEqual(Array.from(m).map(([k]) => k), ["a", "b", "c"], "iterable like a Map");
  m.set("a", { at: 9 });
  assert.deepEqual(Array.from(m.keys()), ["a", "b", "c"], "set on an existing key keeps its position (Map semantics)");
  assert.equal(m.keys().next().value, "a", "the FIFO eviction idiom (windowCache / bloomCache)");
  const oldest = Array.from(m.entries()).sort((x, y) => x[1].at - y[1].at)[0];
  assert.equal(oldest[0], "b", "the oldest-by-timestamp idiom (aircraftCache / vesselSnapCache / bodyCache)");
  assert.equal(m.has("a"), true);
  assert.equal(m.has("zz"), false);
  assert.equal(m.peek("zz"), undefined);
  assert.equal(m.delete("c"), true);
  assert.equal(m.size, 2);
  m.clear();
  assert.equal(m.size, 0);
  m.dispose();
});

test("idleTtlMap: a sweep drops entries unused for the TTL, keeps touched ones, counts drops; TTL 0 never drops", () => {
  let t = 0;
  const now = () => t;
  const m = idleTtlMap<string, number>("test.map.sweep", { ttlMs: 10_000, now });
  m.set("a", 1); m.set("b", 2); m.set("c", 3);
  t = 5_000;
  assert.equal(m.get("b"), 2, "get records a use");
  t = 10_500; // a, c idle 10.5s; b idle 5.5s
  assert.equal(m.sweep(t), 2);
  assert.deepEqual(Array.from(m.keys()), ["b"]);
  assert.deepEqual(m.stats(), { entries: 1, ttl_s: 10, last_used_s: 6, drops: 2 });
  t = 20_000;
  assert.equal(m.sweep(t), 1);
  assert.equal(m.size, 0);
  assert.equal(m.stats().drops, 3);
  m.dispose();

  // has()/peek() are pure reads — they do not keep an entry alive
  const h = idleTtlMap<string, number>("test.map.has", { ttlMs: 10_000, now });
  t = 0; h.set("a", 1);
  t = 20_000;
  assert.equal(h.has("a"), true);
  assert.equal(h.peek("a"), 1);
  assert.equal(h.sweep(t), 1, "has/peek did not touch");
  h.dispose();

  const d = idleTtlMap<string, number>("test.map.disabled", { ttlMs: 0, now });
  d.set("x", 1);
  t = 1e12;
  assert.equal(d.sweep(t), 0, "TTL 0 = today's behavior: nothing ever drops");
  assert.equal(d.get("x"), 1);
  assert.equal(d.stats().ttl_s, 0);
  d.dispose();
});

test("idleTtlMap: on-access sweeps run at most once per IDLE_SWEEP_INTERVAL_MS, and the entry being accessed survives them", () => {
  let t = 0;
  const now = () => t;
  const m = idleTtlMap<string, number>("test.map.access", { ttlMs: 10_000, now });
  m.set("a", 1);
  t = 30_000;
  m.set("b", 2); // 30s since the last sweep: no sweep yet, a (idle 30s) still here
  assert.equal(m.has("a"), true);
  t = 61_000;
  assert.equal(m.get("a"), 1, "the access touches a BEFORE the sweep, so a survives its own request");
  assert.equal(m.has("a"), true);
  assert.equal(m.has("b"), false, "b (idle 31s) was swept by the access-triggered sweep");
  t = 122_000;
  m.set("c", 3);
  assert.equal(m.has("a"), false, "a idle 61s: swept on the next access past the interval");
  assert.equal(m.has("c"), true);
  m.dispose();
});

// ── lazyDataset ─────────────────────────────────────────────────────────────
test("lazyDataset: one load while in use (same reference); released after the TTL idle; the next get() reloads through the same loader", () => {
  let t = 0;
  const now = () => t;
  let loads = 0;
  const ds = lazyDataset("test.ds", () => { loads++; return { rows: [1, 2, 3], tag: "x" }; }, { ttlMs: 10_000, now });
  assert.equal(ds.isLoaded(), false);
  assert.equal(ds.peek(), undefined);
  const a = ds.get();
  t = 9_999;
  const b = ds.get();
  assert.equal(b, a, "same reference while in use");
  assert.equal(loads, 1);
  t = 19_998;
  assert.equal(ds.sweep(t), 0, "used 9.999s ago: not idle yet");
  t = 19_999;
  assert.equal(ds.sweep(t), 1, "idle for exactly the TTL: released");
  assert.equal(ds.isLoaded(), false);
  assert.equal(ds.peek(), undefined);
  const c = ds.get();
  assert.equal(loads, 2, "reloaded through the same loader");
  assert.notEqual(c, a);
  assert.deepEqual(c, a);
  assert.deepEqual(ds.stats(), { entries: 1, ttl_s: 10, last_used_s: 0, drops: 1, loaded: true, loads: 2 });
  ds.reset();
  assert.equal(ds.isLoaded(), false);
  ds.dispose();

  // a loader that degrades to null is cached like any value (the modules' `cached = null` contract)
  let nullLoads = 0;
  const nd = lazyDataset<{ x: number } | null>("test.ds.null", () => { nullLoads++; return null; }, { ttlMs: 10_000, now });
  assert.equal(nd.get(), null);
  assert.equal(nd.get(), null);
  assert.equal(nullLoads, 1, "null is held, not re-run per call");
  assert.equal(nd.isLoaded(), true);
  t += 10_001;
  assert.equal(nd.sweep(t), 1);
  assert.equal(nd.get(), null);
  assert.equal(nullLoads, 2, "after the idle release the file is tried again");
  nd.dispose();

  // a throwing loader leaves the dataset unloaded (today: `cached` stayed undefined)
  const bad = lazyDataset<number>("test.ds.bad", () => { throw new Error("boom"); }, { ttlMs: 10_000, now });
  assert.throws(() => bad.get(), /boom/);
  assert.equal(bad.isLoaded(), false);
  bad.dispose();

  const off = lazyDataset("test.ds.off", () => ({ v: 1 }), { ttlMs: 0, now });
  off.get();
  t = 1e12;
  assert.equal(off.sweep(t), 0, "TTL 0: never released");
  assert.equal(off.isLoaded(), true);
  off.dispose();
});

// ── idleRelease ─────────────────────────────────────────────────────────────
test("idleRelease: release() fires once the held state has been idle for the TTL; a no-op while not held", () => {
  let t = 0;
  const now = () => t;
  let held: number[] | null = null;
  let releases = 0;
  const r = idleRelease("test.rel", () => { held = null; releases++; }, () => held !== null, { ttlMs: 10_000, now });
  held = [1, 2];
  r.touch();
  t = 9_000;
  assert.equal(r.sweep(t), 0);
  assert.deepEqual(r.stats(), { entries: 1, ttl_s: 10, last_used_s: 9, drops: 0, loaded: true });
  t = 10_000;
  assert.equal(r.sweep(t), 1);
  assert.equal(held, null);
  assert.equal(releases, 1);
  t = 1e9;
  assert.equal(r.sweep(t), 0, "nothing held: nothing to release");
  assert.equal(releases, 1);
  assert.equal(r.stats().loaded, false);
  // touch() records the use before any access-triggered sweep, so a use after a long idle is never self-defeating
  held = [3];
  r.touch();
  assert.equal(held !== null, true);
  r.dispose();
});

// ── registry + the aggregate gauge ──────────────────────────────────────────
test("registry: sweepIdleTtl(now) reaches every registered structure; stats are keyed by name; replace-by-name; the idleTtl heap gauge lists them", () => {
  const a = idleTtlMap<string, number>("test.reg.a");
  const b = lazyDataset("test.reg.b", () => ({ v: 1 }));
  a.set("k", 1);
  b.get();
  const stats = idleTtlStats();
  assert.equal(stats["test.reg.a"].entries, 1);
  assert.equal(stats["test.reg.b"].loaded, true);
  assert.deepEqual(Object.keys(stats), Object.keys(stats).slice().sort(), "sorted by name");
  assert.ok(sweepIdleTtl(FAR_FUTURE()) >= 2, "a far-future sweep drops both");
  assert.equal(a.size, 0);
  assert.equal(b.isLoaded(), false);
  const a2 = idleTtlMap<string, number>("test.reg.a");
  a2.set("x", 1); a2.set("y", 2);
  assert.equal(idleTtlStats()["test.reg.a"].entries, 2, "the newest registrant under a name is the live one");
  const gauge = readHeapGauges()["idleTtl"] as { enabled: boolean; ttl_min: number; flag: string; structures: Record<string, { entries: number }> };
  assert.equal(gauge.flag, IDLE_TTL_FLAG);
  assert.equal(typeof gauge.enabled, "boolean");
  assert.equal(gauge.structures["test.reg.a"].entries, 2);
  a2.dispose(); b.dispose(); a.dispose();
  assert.equal(idleTtlStats()["test.reg.a"], undefined, "dispose unregisters");
});

// ── cold → warm → forced idle drop → reload: identical results ──────────────
const T0 = Math.floor(Date.parse("2026-08-10T12:00:00Z") / 1000);
const pt = (hex: string, tSec: number, lat: number, lon: number, altM: number): AircraftPoint & { tSec: number } => ({
  tSec, icao24: hex, registration: `N${hex}`, type: "B738", callsign: `C${hex}`, lat, lon, altitude_m: altM, on_ground: false,
  velocity_ms: null, heading: null, category: null,
} as unknown as AircraftPoint & { tSec: number });

test("routes windowCache pattern: cold → warm → after idle drop, byte-identical window reads from the archive fixture", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-idle-win-"));
  try {
    const fixes: Array<AircraftPoint & { tSec: number }> = [];
    for (let h = 0; h < 12; h++) {
      const hex = h.toString(16).padStart(6, "a");
      for (let k = 0; k < 40; k++) fixes.push(pt(hex, T0 + k * 30, 40 + h * 0.1 + k * 0.001, -100 + k * 0.002, 9000 + k));
    }
    archiveAircraftAt(fixes, base);
    // the exact statements the /api/data/aircraft/window handler runs around its cache
    const windowCache = idleTtlMap<string, { at: number; data: WindowResult }>("test.windowCache");
    const key = ["aircraft", "-110.00", "35.00", "-90.00", "45.00", Math.floor(T0 / 60), Math.floor((T0 + 3600) / 60), 9, "auto", "dflt"].join("|");
    const read = async (): Promise<{ data: WindowResult; cached: boolean }> => {
      const hit = windowCache.get(key);
      if (hit && Date.now() - hit.at < 30_000) return { data: hit.data, cached: true };
      const data = await readWindow({ kind: "aircraft", bbox: { w: -110, s: 35, e: -90, n: 45 }, fromSec: T0, toSec: T0 + 3600, zoom: 9, baseDir: base, closeApproaches: false });
      windowCache.set(key, { at: Date.now(), data });
      if (windowCache.size > 32) {
        const oldest = windowCache.keys().next().value;
        if (oldest !== undefined) windowCache.delete(oldest);
      }
      return { data, cached: false };
    };
    const cold = await read();
    assert.equal(cold.cached, false);
    assert.equal(cold.data.hexes.length, 12);
    assert.equal(cold.data.total_points, 480);
    const warm = await read();
    assert.equal(warm.cached, true);
    assert.equal(warm.data, cold.data, "warm = the cached object");
    assert.ok(sweepIdleTtl(FAR_FUTURE()) >= 1);
    assert.equal(windowCache.size, 0, "the forced idle drop emptied the cache");
    const reload = await read();
    assert.equal(reload.cached, false, "the handler took its ordinary cold path");
    assert.notEqual(reload.data, cold.data);
    assert.deepEqual(reload.data, cold.data);
    assert.equal(JSON.stringify(reload.data), JSON.stringify(cold.data), "byte-identical response body");
    assert.equal(windowCache.stats().drops, 1);
    windowCache.dispose();
  } finally {
    fs.rmSync(base, { recursive: true, force: true });
  }
});

type Fix = AircraftPoint & { seen_at_ms?: number | null };
const ac = (hex: string, lat: number, lon: number): Fix => ({
  icao24: hex, callsign: `C${hex}`, lat, lon, altitude_m: 10000, on_ground: false,
  velocity_ms: 230, heading: 45, type: "B738", category: "A3", registration: `N${hex}`, seen_pos: 1,
} as unknown as Fix);

test("globalAircraft bodyCache: cold → warm → after idle drop, identical bodies through the real /api/data/aircraft/global route", async () => {
  const archiveDir = fs.mkdtempSync(path.join(os.tmpdir(), "vt-idle-global-"));
  const app = express();
  const T = Date.now();
  const h = registerGlobalAircraftRoutes(app, {
    env: { GLOBAL_SWEEP_ENABLED: "0", GLOBAL_SWEEP_ARCHIVE: "0" } as NodeJS.ProcessEnv,
    fetchImpl: (async () => { throw new Error("no network in tests"); }) as unknown as typeof fetch,
    sites: [], baseDir: archiveDir, now: () => T, readFree: () => 50 * 1_073_741_824,
  });
  const srv = app.listen(0);
  try {
    publishFixes({ provider: "adsblol", origin: "sweep-type", aircraft: [ac("aa0001", 40, -75), ac("aa0002", 51, 0), ac("aa0003", -33, 151)], fetchedAt: T, upstreamNowMs: T });
    const url = `http://127.0.0.1:${(srv.address() as AddressInfo).port}/api/data/aircraft/global`;
    const cold = await (await fetch(url)).text();
    assert.equal(JSON.parse(cold).count, 3);
    const warm = await (await fetch(url)).text();
    assert.equal(warm, cold);
    assert.equal(idleTtlStats()["globalAircraft.bodyCache"].entries, 1);
    assert.ok(sweepIdleTtl(FAR_FUTURE()) >= 1);
    assert.equal(idleTtlStats()["globalAircraft.bodyCache"].entries, 0, "the forced idle drop emptied the body cache");
    const reload = await (await fetch(url)).text();
    assert.equal(reload, cold, "byte-identical body after the reload");
    assert.equal(idleTtlStats()["globalAircraft.bodyCache"].drops, 1);
    assert.equal(idleTtlStats()["globalAircraft.bodyCache"].entries, 1, "re-cached by the ordinary miss path");
  } finally {
    h.stop();
    srv.close();
    fs.rmSync(archiveDir, { recursive: true, force: true });
  }
});

test("aircraftTrips bloomCache: cold → warm → after idle drop, identical tracks; filters rebuilt from the closed hour files", async () => {
  _resetBloomCache();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-idle-bloom-"));
  try {
    const dir = path.join(base, "aircraft");
    fs.mkdirSync(dir, { recursive: true });
    const t0 = Math.floor(Date.parse("2026-08-05T10:00:00Z") / 1000);
    const wr = (hh: string, rows: object[]) =>
      fs.writeFileSync(path.join(dir, `2026-08-05-${hh}.jsonl`), rows.map((r) => JSON.stringify(r)).join("\n") + "\n");
    wr("10", [{ t: t0, i: "abe872", la: 1, lo: 1, al: 100 }, { t: t0 + 5, i: "111111", la: 2, lo: 2 }]);
    wr("11", [{ t: t0 + 3600, i: "222222", la: 3, lo: 3 }, { t: t0 + 3700, i: "abe872", la: 1.5, lo: 1.5, al: 200 }]);
    const cold = await fullTrackAsync("aircraft", "abe872", base);
    assert.equal(cold.length, 2);
    assert.equal(bloomCacheSize(), 2, "both closed hours got a filter");
    const warm = await fullTrackAsync("aircraft", "abe872", base);
    assert.deepEqual(warm, cold);
    assert.ok(sweepIdleTtl(FAR_FUTURE()) >= 2);
    assert.equal(bloomCacheSize(), 0, "the forced idle drop emptied the Bloom cache");
    const reload = await fullTrackAsync("aircraft", "abe872", base);
    assert.deepEqual(reload, cold, "a Bloom filter has no false negatives: identical track with or without the filters");
    assert.equal(bloomCacheSize(), 2, "rebuilt by the next scan exactly as after a deploy");
    assert.equal(idleTtlStats()["aircraftTrips.bloomCache"].drops, 2);
    const gauge = readHeapGauges()["aircraftTrips.bloomCache"] as { entries: number; bytes: number; cap: number };
    assert.deepEqual(gauge, { entries: 2, bytes: 2 * 32768, cap: 2000 }, "the A0 gauge line is untouched and still reads the live size");
  } finally {
    fs.rmSync(base, { recursive: true, force: true });
    _resetBloomCache();
  }
});

test("airportsIndex: cold → warm → after idle release, identical lookups; the gauge reads loaded:false in between", () => {
  _resetAirports();
  const q = [40.6413, -73.7781] as const;
  const cold = nearestAirport(q[0], q[1], 6, ["L", "M", "S", "W"]);
  assert.equal(cold?.id, "KJFK");
  const coldSfo = airportByIdent("KSFO");
  assert.ok(coldSfo);
  const loaded = () => (readHeapGauges()["airportsIndex"] as { loaded: boolean; entries: number }).loaded;
  assert.equal(loaded(), true);
  const warm = nearestAirport(q[0], q[1], 6, ["L", "M", "S", "W"]);
  assert.deepEqual(warm, cold);
  assert.ok(sweepIdleTtl(FAR_FUTURE()) >= 1);
  assert.equal(loaded(), false, "released: the 22 MB grid is gone until the next lookup");
  assert.equal(idleTtlStats()["airportsIndex"].loaded, false);
  const reload = nearestAirport(q[0], q[1], 6, ["L", "M", "S", "W"]);
  assert.deepEqual(reload, cold, "reloaded through loadAirports(): identical match");
  assert.deepEqual(airportByIdent("KSFO"), coldSfo);
  assert.equal(loaded(), true);
  assert.ok(loadAirports() > 60_000, "loadAirports() on a loaded grid is a touch, not a reload");
  assert.equal(idleTtlStats()["airportsIndex"].drops, 1);
  // the touch of a lookup after a long idle never defeats itself: use → sweep order is touch-first
  const again = airportByIdent("KSFO");
  assert.deepEqual(again, coldSfo);
  _resetAirports();
});

test("GEM datasets: cold → warm (same reference) → after idle release → reload deep-equal, including the two-level choropleth join", () => {
  _resetGemChemicalsCacheForTests();
  _resetGemSteelRawMaterialsCacheForTests();
  _resetSteelRawMaterialsGeoCacheForTests();
  try {
    const cold = cachedGemChemicals();
    assert.ok(cold && cold.plants.length > 0, "real repo fixture loads");
    const warm = cachedGemChemicals();
    assert.equal(warm, cold, "same object while in use (the modules' existing contract)");
    const geo0 = cachedSteelRawMaterialsGeoJSON();
    assert.ok(geo0 && geo0.matched > 0);
    assert.equal(idleTtlStats()["gemChemicals"].loaded, true);
    assert.equal(idleTtlStats()["gemSteelRawMaterials"].loaded, true);
    assert.equal(idleTtlStats()["gemSteelRawMaterialsGeo"].loaded, true);
    assert.ok(sweepIdleTtl(FAR_FUTURE()) >= 3);
    assert.equal(idleTtlStats()["gemChemicals"].loaded, false);
    assert.equal(idleTtlStats()["gemSteelRawMaterials"].loaded, false);
    assert.equal(idleTtlStats()["gemSteelRawMaterialsGeo"].loaded, false);
    const reload = cachedGemChemicals();
    assert.notEqual(reload, cold, "a fresh parse");
    assert.deepEqual(reload, cold);
    assert.equal(JSON.stringify(reload), JSON.stringify(cold), "byte-identical payload");
    assert.equal(idleTtlStats()["gemChemicals"].loads, 2);
    const geo1 = cachedSteelRawMaterialsGeoJSON();
    assert.notEqual(geo1, geo0);
    assert.equal(JSON.stringify(geo1), JSON.stringify(geo0), "the join reloads its own input and lands on the same bytes");
    assert.equal(idleTtlStats()["gemSteelRawMaterials"].loads, 2, "the input dataset was reloaded by the join");
  } finally {
    _resetGemChemicalsCacheForTests();
    _resetGemSteelRawMaterialsCacheForTests();
    _resetSteelRawMaterialsGeoCacheForTests();
  }
});

// ── source ratchet: every A3 structure stays wired through idleTtl.ts ───────
test("source: the dashboard caches and lazy datasets the audit named are wired through idleTtl.ts (and the A0 gauge lines are untouched)", () => {
  const src = (f: string) => fs.readFileSync(path.join(HERE, f), "utf8");
  const expect: Array<[string, RegExp]> = [
    ["routes.ts", /const aircraftCache = idleTtlMap<[^\n]*\("routes\.aircraftCache"\)/],
    ["routes.ts", /const vesselSnapCache = idleTtlMap<[^\n]*\("routes\.vesselSnapCache"\)/],
    ["routes.ts", /const windowCache = idleTtlMap<[^\n]*\("routes\.windowCache"\)/],
    ["routes.ts", /trackMap\("routes\.windowCache", windowCache, 32\); \/\/ memory audit 2026-11-03 A0/],
    ["globalAircraft.ts", /const bodyCache = idleTtlMap<[^\n]*\("globalAircraft\.bodyCache"\)/],
    ["globalAircraft.ts", /trackMap\("globalAircraft\.bodyCache", bodyCache, 24\); \/\/ memory audit 2026-11-03 A0/],
    ["aircraftTrips.ts", /const bloomCache = idleTtlMap<string, Uint8Array>\("aircraftTrips\.bloomCache"\)/],
    ["aircraftTrips.ts", /registerHeapGauge\("aircraftTrips\.bloomCache", \(\) => \(\{ entries: bloomCache\.size/],
    ["airportsIndex.ts", /idleRelease\("airportsIndex",/],
    ["airportsIndex.ts", /registerHeapGauge\("airportsIndex", \(\) => \(\{ entries: loadedCount/],
    ["gemChemicals.ts", /lazyDataset<[^\n]*\("gemChemicals"/],
    ["gemCoalMineFeatures.ts", /lazyDataset<[^\n]*\("gemCoalMineFeatures"/],
    ["gemCoalTerminals.ts", /lazyDataset<[^\n]*\("gemCoalTerminals"/],
    ["gemIronOreMines.ts", /lazyDataset<[^\n]*\("gemIronOreMines"/],
    ["gemIronSteelPlants.ts", /lazyDataset<[^\n]*\("gemIronSteelPlants"/],
    ["gemLngCarriers.ts", /lazyDataset<[^\n]*\("gemLngShipyards"/],
    ["gemMethane.ts", /lazyDataset<[^\n]*\("gemMethane"/],
    ["gemMethaneAssets.ts", /lazyDataset<[^\n]*\("gemMethaneAssets"/],
    ["gemMethaneProximity.ts", /lazyDataset<[^\n]*\("gemMethaneProximity"/],
    ["gemSteelRawMaterials.ts", /lazyDataset<[^\n]*\("gemSteelRawMaterials"/],
    ["gemSteelRawMaterials.ts", /lazyDataset<[^\n]*\("gemSteelRawMaterialsGeo"/],
  ];
  for (const [file, re] of expect) assert.match(src(file), re, `${file} must match ${re}`);
  // no module-level `let cached` singleton is left behind in the GEM family
  for (const f of fs.readdirSync(HERE).filter((n) => /^gem.*\.ts$/.test(n) && !n.endsWith(".test.ts"))) {
    assert.doesNotMatch(src(f), /^let cached/m, `${f} still holds an un-managed module singleton`);
  }
});
