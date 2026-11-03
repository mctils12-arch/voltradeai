// [A2 2026-11-03] Incremental port-dwell fold — exactness proofs and the
// shadow-compare harness (memory fix A2: fold each closed hour once, combine
// per-hour partials each cycle; VOLTRADE_PORTDWELL_FOLD=full|shadow|incremental).
//
// Every exactness test compares THREE computations on the same archive:
// the sync materializing reference (computePortDwell / detectVisits,
// tests-only), the original online fold (mode=full), and the incremental
// fold (cold cache, then warm cache). The fixtures carry the shapes that
// make a naive per-hour merge wrong: calls spanning many hour files, a
// call whose per-hour speed medians disagree with the whole-call median,
// a call that both starts and ends inside one hour, a coverage gap that
// splits a call, a port switch, an in-progress (right-censored) call
// through the open hour, a vessel seen in exactly one hour, duplicate
// lines, a stale duplicate in the wrong hour, out-of-order timestamps, the
// cutoff-straddling oldest hour and the still-open current hour.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import {
  computePortDwell, computePortDwellAsync, computePortDwellAsyncTimed, foldPortVisitsAsync,
  portDwellFoldStats, _resetPortDwellFoldForTests, _portDwellFoldCacheForTests, PORTDWELL_FOLD_ENV,
  type PortDef, type PortDwellStats, type PortVisit,
} from "./portDwell";
import { listVesselWindowFiles } from "./vesselArchiveFold";

const NOW = Date.UTC(2026, 6, 4, 12, 30, 0); // 12:30Z — hours <= 11 closed, hour 12 open
const t = (hoursAgo: number, from = NOW) => Math.floor((from - hoursAgo * 3600_000) / 1000);
const hourFile = (tSec: number) => { const d = new Date(tSec * 1000).toISOString(); return `${d.slice(0, 10)}-${d.slice(11, 13)}.jsonl`; };

const LA: PortDef = { id: "port_la", name: "Port of Los Angeles", lat: 33.74, lon: -118.272, radius_km: 5 };
const LB: PortDef = { id: "port_lb", name: "Port of Long Beach", lat: 33.7515, lon: -118.213, radius_km: 5 };
const PORTS = [LA, LB];

type Line = Record<string, unknown> & { t: number; _file?: string };

function writeArchive(base: string, points: Line[], gzEvery = 0): string[] {
  const dir = path.join(base, "vessels");
  fs.mkdirSync(dir, { recursive: true });
  const byFile = new Map<string, string[]>();
  for (const p of points) {
    const { _file, ...rest } = p;
    const f = _file ?? hourFile(p.t);
    if (!byFile.has(f)) byFile.set(f, []);
    byFile.get(f)!.push(JSON.stringify(rest));
  }
  const names: string[] = [];
  let k = 0;
  for (const [f, lines] of byFile) {
    const body = lines.join("\n") + "\n";
    if (gzEvery && ++k % gzEvery === 0) { fs.writeFileSync(path.join(dir, f + ".gz"), zlib.gzipSync(body)); names.push(f + ".gz"); }
    else { fs.writeFileSync(path.join(dir, f), body); names.push(f); }
  }
  return names;
}

/** In-fence points for one vessel from `fromH` hours ago down to `toH`, every `stepH`. */
const moored = (mmsi: string, name: string | undefined, port: PortDef, fromH: number, toH: number, stepH = 1, v: number | ((i: number) => number) = 0.2): Line[] => {
  const pts: Line[] = [];
  let i = 0;
  for (let h = fromH; h >= toH - 1e-9; h -= stepH, i++) {
    pts.push({ t: t(h), i: mmsi, c: name, la: port.lat + 0.005, lo: port.lon + 0.005, v: typeof v === "function" ? v(i) : v });
  }
  return pts;
};

async function withMode<T>(mode: string, fn: () => Promise<T>): Promise<T> {
  const prev = process.env[PORTDWELL_FOLD_ENV];
  process.env[PORTDWELL_FOLD_ENV] = mode;
  try { return await fn(); } finally {
    if (prev === undefined) delete process.env[PORTDWELL_FOLD_ENV]; else process.env[PORTDWELL_FOLD_ENV] = prev;
  }
}

function adversarialFixture(): Line[] {
  return [
    // 20h call spanning 21 hour files, completed 20h ago
    ...moored("101", "BOXSHIP", LA, 40, 20),
    // in progress through the open hour (last point 12 min ago): right-censored
    ...moored("102", "STAYER", LA, 30, 0.2, 0.5),
    // MEDIAN TRAP A: 29 fixes at 0.5 kt across two hour files (4-min cadence), then 10 at
    // 5 kt in the next hour, one continuous 2.6h run. Its last hour alone reads "fast";
    // the whole run's median is 0.5 -> a call.
    ...moored("103", "SLOWTHENFAST", LB, 11.95, 10.08, 1 / 15, 0.5),
    ...moored("103", "SLOWTHENFAST", LB, 9.95, 9.35, 1 / 15, 5),
    // MEDIAN TRAP B: 10 fixes at 0.5 then 29 at 5 across the following hours, one 2.9h run:
    // its first hour alone reads "moored"; the whole run's median is 5 -> NOT a call.
    ...moored("104", "FASTLATER", LB, 13.95, 13.35, 1 / 15, 0.5),
    ...moored("104", "FASTLATER", LB, 12.95, 11.08, 1 / 15, 5),
    // a call that starts AND ends inside one hour (1.5h, 4 fixes): below minDwell -> dropped
    ...moored("105", "QUICK", LA, 6.9, 6.3, 0.2),
    // coverage gap > 6h inside a call splits it into two completed calls
    ...moored("106", "GAPPY", LA, 60, 50),
    ...moored("106", "GAPPY", LA, 42, 32),
    // port switch LA -> LB on consecutive hourly fixes
    ...moored("107", "SWITCHER", LA, 50, 44),
    ...moored("107", "SWITCHER", LB, 43, 36),
    // harbor tug: 10h in fence, median 7 kt -> never a call
    ...moored("108", "TUG", LA, 30, 20, 1, 7),
    // off-fence transit vessel, one hour only
    { t: t(5.4), i: "109", c: "PASSER", la: LA.lat + 0.3, lo: LA.lon + 0.3, v: 12 },
    { t: t(5.2), i: "109", c: "PASSER", la: LA.lat + 0.31, lo: LA.lon + 0.31, v: 12 },
    // out-of-order inside one hour: 07:54, 07:06, 07:30 in file order, then 08:36 — in LA
    { t: t(4.6), i: "110", c: "SHUFFLE", la: LA.lat, lo: LA.lon, v: 0.1 },
    { t: t(5.4), i: "110", c: "SHUFFLE", la: LA.lat, lo: LA.lon, v: 0.1 },
    { t: t(5.0), i: "110", c: "SHUFFLE", la: LA.lat, lo: LA.lon, v: 0.1 },
    { t: t(3.9), i: "110", c: "SHUFFLE", la: LA.lat, lo: LA.lon, v: 0.1 },
    // name appears only on the LAST fix (name = last.c || first.c)
    ...moored("111", undefined, LB, 26, 21),
    { t: t(20), i: "111", c: "LATE NAME", la: LB.lat + 0.005, lo: LB.lon + 0.005, v: 0.2 },
    // duplicate lines inside one hour (points count doubles for that hour)
    { t: t(15.5), i: "112", c: "DOUBLE", la: LA.lat, lo: LA.lon, v: 0.2 },
    { t: t(15.5), i: "112", c: "DOUBLE", la: LA.lat, lo: LA.lon, v: 0.2 },
    ...moored("112", "DOUBLE", LA, 15, 12),
    // the cutoff-straddling oldest hour (168h window from 12:30 -> cutoff 12:30 a week ago):
    // a call ending just after the cutoff whose first fix is just before it (excluded)
    { t: t(168.3), i: "113", c: "EDGE", la: LA.lat, lo: LA.lon, v: 0.1 },
    ...moored("113", "EDGE", LA, 167.9, 165),
    // leaves the fence for one fix in the middle of a call (splits it, both halves short)
    ...moored("114", "DIPPER", LB, 9, 7, 0.5),
    { t: t(6.75), i: "114", c: "DIPPER", la: 0, lo: 0, v: 9 },
    ...moored("114", "DIPPER", LB, 6.5, 4.5, 0.5),
  ];
}

/** A stale duplicate of a 15:30 fix written into the 16:xx file — a point
 *  outside its file's hour. The port-dwell partial's composition does not
 *  depend on the hour invariant, so this file stays cacheable AND the
 *  result stays identical. */
function staleDuplicate(): Line {
  return { t: t(15.5), i: "112", c: "DOUBLE", la: LA.lat, lo: LA.lon, v: 0.2, _file: hourFile(t(14.5)) };
}

async function threeWay(base: string, ports: PortDef[], windowHours: number, now: number):
    Promise<{ sync: PortDwellStats; full: PortDwellStats; cold: PortDwellStats; warm: PortDwellStats }> {
  const sync = computePortDwell(ports, windowHours, base, now);
  const full = await withMode("full", () => computePortDwellAsync(ports, windowHours, base, now));
  const cold = await withMode("incremental", () => computePortDwellAsync(ports, windowHours, base, now));
  const warm = await withMode("incremental", () => computePortDwellAsync(ports, windowHours, base, now));
  return { sync, full, cold, warm };
}

const visitRows = (r: { visitsByPort: Map<string, PortVisit[]> }) =>
  Array.from(r.visitsByPort.entries()).map(([port, vs]) => [port, vs] as const);

test("EXACT: incremental fold (cold and warm cache) == original online fold == sync reference on the adversarial fixture, incl. gz files — down to the raw visit lists", async () => {
  _resetPortDwellFoldForTests();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-dwell-inc-"));
  writeArchive(base, adversarialFixture(), 3);
  const { sync, full, cold, warm } = await threeWay(base, PORTS, 168, NOW);
  assert.deepEqual(full, sync, "mode=full must still equal the sync reference (untouched behavior)");
  assert.deepEqual(cold, full, "cold incremental (every closed hour built this cycle, merged via partials) must equal full");
  assert.deepEqual(warm, full, "warm incremental (closed hours from cache) must equal full");
  // Raw visits (before aggregation), in order — stricter than the stats.
  const fullVisits = await withMode("full", () => foldPortVisitsAsync(PORTS, 168, base, NOW));
  const incVisits = await withMode("incremental", () => foldPortVisitsAsync(PORTS, 168, base, NOW));
  assert.deepEqual(visitRows(incVisits), visitRows(fullVisits));
  assert.equal(incVisits.vesselsSeen, fullVisits.vesselsSeen);
  assert.equal(incVisits.pointsScanned, fullVisits.pointsScanned, "pointsScanned counts the points the partials stood in for");
  // The fixture really exercises what it claims to.
  const la = full.ports.find((p) => p.id === "port_la")!, lb = full.ports.find((p) => p.id === "port_lb")!;
  assert.equal(la.in_port_now, 1, "STAYER is right-censored through the open hour");
  const mmsis = (vs: PortVisit[]) => vs.map((v) => v.mmsi);
  assert.ok(mmsis(fullVisits.visitsByPort.get("port_lb")!).includes("103"), "median trap A: whole-run median 0.5 -> a call");
  assert.ok(!mmsis(fullVisits.visitsByPort.get("port_lb")!).includes("104"), "median trap B: whole-run median 5 -> not a call");
  assert.ok(!mmsis(fullVisits.visitsByPort.get("port_la")!).includes("105"), "a 1.5h in-hour run is below minDwell");
  assert.equal(mmsis(fullVisits.visitsByPort.get("port_la")!).filter((m) => m === "106").length, 2, "the coverage gap splits GAPPY");
  assert.equal(fullVisits.visitsByPort.get("port_lb")!.find((v) => v.mmsi === "111")!.name, "LATE NAME");
  assert.equal(fullVisits.visitsByPort.get("port_la")!.find((v) => v.mmsi === "113")!.firstSeen, t(167.9), "the pre-cutoff fix is excluded");
  assert.ok(lb.visits_completed >= 3 && la.visits_completed >= 5);
  const s = portDwellFoldStats();
  assert.equal(s.mode, "incremental");
  const listing = listVesselWindowFiles(168, base, NOW);
  const cacheable = listing.files.filter((f) => f.cacheable).length;
  assert.ok(cacheable >= 40, `fixture must span many closed hours (got ${cacheable})`);
  assert.equal(s.filesCachedLastRun, cacheable, "warm cycle served every closed hour from cache");
  assert.equal(s.filesStreamedLastRun, listing.files.length - cacheable, "warm cycle streamed only the open hour and the cutoff-straddling hour");
  assert.equal(s.partialsCached, cacheable);
  assert.equal(s.fallbacks, 0);
  assert.equal(s.uncacheableFiles, 0);
  assert.ok(s.partialBytesEstimate > 0 && s.lastIncrementalMs != null);
});

test("EXACT with a stale duplicate in the wrong hour: the port-dwell composition needs no hour invariant — file stays cacheable, result identical", async () => {
  _resetPortDwellFoldForTests();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-dwell-inc-stale-"));
  writeArchive(base, [...adversarialFixture(), staleDuplicate()]);
  const { sync, full, cold, warm } = await threeWay(base, PORTS, 168, NOW);
  assert.deepEqual(full, sync);
  assert.deepEqual(cold, full);
  assert.deepEqual(warm, full);
  const s = portDwellFoldStats();
  assert.equal(s.uncacheableFiles, 0);
  assert.equal(s.fallbacks, 0);
});

test("EXACT across a sliding 168h window: as `now` advances hour by hour the cache is reused, evicted beyond the window, and every cycle equals a fresh full fold", async () => {
  _resetPortDwellFoldForTests();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-dwell-inc-slide-"));
  const pts: Line[] = [];
  // 12 vessels with staggered multi-day calls (arrivals/departures every few hours), a tug, a transit lane
  for (let v = 0; v < 12; v++) {
    const port = v % 2 ? LA : LB;
    for (let call = 0; call < 4; call++) {
      const start = 190 - call * 45 - v * 3;
      pts.push(...moored(`50${v}`, `SHIP ${v}`, port, start, Math.max(0.3, start - 18 - v), 0.5, (i) => (i % 5 === 0 ? 1.5 : 0.2)));
    }
  }
  pts.push(...moored("599", "TUG", LA, 190, 0.5, 0.25, 6));
  for (let h = 190; h >= 0; h -= 2) pts.push({ t: t(h), i: "600", c: "LANE", la: LA.lat + 0.2, lo: LA.lon + 0.2, v: 11 });
  writeArchive(base, pts, 2);
  const WINDOW = 168;
  let maxCached = 0;
  for (let step = 0; step <= 8; step++) {
    const now = NOW + step * 3600_000 + (step % 2) * 1_800_000;
    const full = await withMode("full", () => computePortDwellAsync(PORTS, WINDOW, base, now));
    const inc = await withMode("incremental", () => computePortDwellAsync(PORTS, WINDOW, base, now));
    assert.deepEqual(inc, full, `step ${step}: incremental must equal full at now=${new Date(now).toISOString()}`);
    assert.deepEqual(inc, computePortDwell(PORTS, WINDOW, base, now), `step ${step}: and the sync reference`);
    const s = portDwellFoldStats();
    maxCached = Math.max(maxCached, s.partialsCached);
    assert.ok(s.partialsCached <= WINDOW + 2, `cache cap: ${s.partialsCached} partials > ${WINDOW + 2}`);
    const cacheable = listVesselWindowFiles(WINDOW, base, now).files.filter((f) => f.cacheable).length;
    assert.equal(s.partialsCached, cacheable, `step ${step}: exactly the window's closed hours are cached (old ones evicted)`);
    assert.equal(s.fallbacks, 0);
  }
  assert.ok(maxCached >= 150, `the fixture must have exercised a nearly-full cache (max ${maxCached})`);
});

test("SHADOW MODE (the default): serves the full fold, compares against the incremental one every cycle; a poisoned partial is caught as FOLD-MISMATCH portdwell and the full result is still served", async () => {
  _resetPortDwellFoldForTests();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-dwell-inc-shadow-"));
  writeArchive(base, adversarialFixture(), 2);
  const full = await withMode("full", () => computePortDwellAsync(PORTS, 168, base, NOW));
  const errors: string[] = [];
  const origErr = console.error;
  console.error = (...a: unknown[]) => { errors.push(a.map(String).join(" ")); };
  try {
    const prev = process.env[PORTDWELL_FOLD_ENV];
    delete process.env[PORTDWELL_FOLD_ENV]; // unset == shadow, the default for this PR
    try {
      assert.deepEqual(await computePortDwellAsync(PORTS, 168, base, NOW), full);
      assert.deepEqual(await computePortDwellAsync(PORTS, 168, base, NOW), full);
    } finally { if (prev === undefined) delete process.env[PORTDWELL_FOLD_ENV]; else process.env[PORTDWELL_FOLD_ENV] = prev; }
    let s = portDwellFoldStats();
    assert.equal(s.mode, "shadow");
    assert.equal(s.matches, 2);
    assert.equal(s.mismatches, 0);
    assert.equal(errors.length, 0);
    assert.ok(s.lastFullMs != null && s.lastIncrementalMs != null);

    // Poison: drop the STAYER's in-progress run from one cached hour.
    const cache = _portDwellFoldCacheForTests(base, PORTS);
    const victim = listVesselWindowFiles(168, base, NOW).files.find((f) => f.cacheable && f.stampSec > t(10))!;
    const partial = cache.peek(victim.name)!;
    const stayerId = Array.from(partial.inPort.keys()).find((id) => cache.interner.str(id) === "102");
    assert.ok(stayerId !== undefined, "the chosen hour holds STAYER's run");
    partial.inPort.delete(stayerId!);
    const r3 = await withMode("shadow", () => computePortDwellAsync(PORTS, 168, base, NOW));
    assert.deepEqual(r3, full, "shadow mode SERVES THE FULL FOLD even when the incremental one disagrees");
    s = portDwellFoldStats();
    assert.equal(s.mismatches, 1);
    assert.ok(s.lastMismatchAt != null);
    assert.equal(errors.length, 1);
    assert.match(errors[0], /^FOLD-MISMATCH portdwell /);
    assert.match(errors[0], /visitsByPort\["port_la"\]/, "the diff summary points at the port whose visits differ");

    // In incremental mode the poison would reach the served result — which
    // is exactly why the switch waits for 24h of zero mismatches.
    const r4 = await withMode("incremental", () => computePortDwellAsync(PORTS, 168, base, NOW));
    assert.notDeepEqual(r4, full);
  } finally { console.error = origErr; }
});

test("FULL MODE: the original path, untouched — equals the sync reference and never builds a partial", async () => {
  _resetPortDwellFoldForTests();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-dwell-inc-full-"));
  writeArchive(base, adversarialFixture());
  const r = await withMode("full", () => computePortDwellAsync(PORTS, 168, base, NOW));
  assert.deepEqual(r, computePortDwell(PORTS, 168, base, NOW));
  const s = portDwellFoldStats();
  assert.equal(s.mode, "full");
  assert.equal(s.partialsCached, 0);
  assert.equal(s.partialBytesEstimate, 0);
  assert.ok(s.lastFullMs != null && s.lastRunAt != null);
});

test("DUPLICATE FILES: the same closed hour as .jsonl AND .jsonl.gz (the compressOldHours window) composes exactly — no fallback needed, result identical to the original fold's own double read", async () => {
  _resetPortDwellFoldForTests();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-dwell-inc-dup-"));
  const names = writeArchive(base, adversarialFixture());
  const victim = names.find((n) => n === hourFile(t(30)))!; // STAYER's and BOXSHIP's calls both run through this hour
  const fp = path.join(base, "vessels", victim);
  fs.writeFileSync(fp + ".gz", zlib.gzipSync(fs.readFileSync(fp)));
  const { sync, full, cold, warm } = await threeWay(base, PORTS, 168, NOW);
  assert.deepEqual(full, sync);
  assert.deepEqual(cold, full);
  assert.deepEqual(warm, full);
  const s = portDwellFoldStats();
  assert.equal(s.fallbacks, 0, "port-dwell partials compose exactly for any point sequence, duplicates included");
  assert.equal(s.compareSkipped, 0);
});

test("computePortDwellAsyncTimed in incremental mode: identical stats and the same pointsScanned as the full fold (partials carry their point counts)", async () => {
  _resetPortDwellFoldForTests();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-dwell-inc-timed-"));
  writeArchive(base, adversarialFixture());
  const full = await withMode("full", () => computePortDwellAsyncTimed(PORTS, 168, base, NOW));
  await withMode("incremental", () => computePortDwellAsyncTimed(PORTS, 168, base, NOW)); // cold
  const inc = await withMode("incremental", () => computePortDwellAsyncTimed(PORTS, 168, base, NOW));
  const strip = (r: typeof full) => { const { elapsedMs, ...rest } = r; void elapsedMs; return rest; };
  assert.deepEqual(strip(inc), strip(full));
  assert.equal(inc.pointsScanned, adversarialFixture().length - 1, "every in-window point counted, the pre-cutoff straddler excluded");
});

test("MEASURE (7-day synthetic archive, 168h window): incremental warm cycle is a small fraction of the full fold; partial cache is a small fraction of the raw archive; outputs identical", async () => {
  _resetPortDwellFoldForTests();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-dwell-inc-7d-"));
  const dir = path.join(base, "vessels");
  fs.mkdirSync(dir, { recursive: true });
  // 400 vessels, ~10 fixes per vessel-hour; ~10% moored inside a fence for
  // days at a time (full-resolution archive cadence near a site), the rest
  // transiting outside or through the fence at speed.
  const HOURS = 169, VESSELS = 400, PTS = 10;
  let rawBytes = 0;
  for (let h = HOURS; h >= 0; h--) {
    const lines: string[] = [];
    const hourStart = Math.floor((NOW - h * 3600_000) / 1000);
    const aligned = hourStart - (hourStart % 3600);
    for (let v = 0; v < VESSELS; v++) {
      const moor = v % 10 === 0 && Math.floor(h / 30) % 2 === (v % 20 === 0 ? 0 : 1); // days-long calls, staggered
      for (let k = 0; k < PTS; k++) {
        const ts = aligned + k * Math.floor(3600 / PTS) + (v % 29);
        if (ts > Math.floor(NOW / 1000)) continue;
        const port = v % 20 === 0 ? LA : LB;
        lines.push(JSON.stringify(moor
          ? { i: `V${v}`, t: ts, c: `VESSEL ${v}`, la: port.lat + 0.004 + (v % 7) * 0.0005, lo: port.lon + 0.004, v: (k % 3) * 0.1 }
          : { i: `V${v}`, t: ts, c: `VESSEL ${v}`, la: 33.5 + (v % 50) * 0.01 + Math.sin(h / 3 + v) * 0.2, lo: -118.6 + (v % 50) * 0.01 + Math.cos(h / 5 + v) * 0.2, v: 6 + (v + k) % 8 }));
      }
    }
    const body = lines.join("\n") + "\n";
    rawBytes += body.length;
    const name = hourFile(aligned);
    if (h % 2 === 0) fs.writeFileSync(path.join(dir, name + ".gz"), zlib.gzipSync(body));
    else fs.writeFileSync(path.join(dir, name), body);
  }
  const time = async (fn: () => Promise<PortDwellStats>) => { const s = performance.now(); const r = await fn(); return { r, ms: performance.now() - s }; };
  const full = await time(() => withMode("full", () => computePortDwellAsync(PORTS, 168, base, NOW)));
  const heapBefore = process.memoryUsage().heapUsed;
  const cold = await time(() => withMode("incremental", () => computePortDwellAsync(PORTS, 168, base, NOW)));
  const heapAfterCold = process.memoryUsage().heapUsed;
  const warm = await time(() => withMode("incremental", () => computePortDwellAsync(PORTS, 168, base, NOW)));
  const warm2 = await time(() => withMode("incremental", () => computePortDwellAsync(PORTS, 168, base, NOW)));
  assert.deepEqual(cold.r, full.r);
  assert.deepEqual(warm.r, full.r);
  assert.deepEqual(warm2.r, full.r);
  const fullVisits = await withMode("full", () => foldPortVisitsAsync(PORTS, 168, base, NOW));
  const incVisits = await withMode("incremental", () => foldPortVisitsAsync(PORTS, 168, base, NOW));
  assert.deepEqual(visitRows(incVisits), visitRows(fullVisits));
  assert.ok(fullVisits.pointsScanned > 500_000 && fullVisits.vesselsSeen === VESSELS, `fixture scale: ${fullVisits.pointsScanned} points`);
  assert.ok(full.r.visits_completed + full.r.in_port_now >= 100, `fixture must hold many calls (${full.r.visits_completed} + ${full.r.in_port_now})`);
  // What the pre-A2 detector retained for these calls: one {t,v,c} triple per in-fence point.
  let inFencePoints = 0;
  fullVisits.visitsByPort.forEach((vs) => { for (const v of vs) inFencePoints += v.points; });
  const s = portDwellFoldStats();
  const bestWarm = Math.min(warm.ms, warm2.ms);
  console.log(`[A2 measure portdwell 168h] points=${fullVisits.pointsScanned} rawJsonlBytes=${rawBytes} ` +
    `fullMs=${full.ms.toFixed(0)} coldIncMs=${cold.ms.toFixed(0)} warmIncMs=${bestWarm.toFixed(0)} ` +
    `partials=${s.partialsCached} partialBytesEstimate=${s.partialBytesEstimate} ` +
    `heapDeltaAfterColdMB=${((heapAfterCold - heapBefore) / 1048576).toFixed(1)} filesStreamedWarm=${s.filesStreamedLastRun} ` +
    `calls=${full.r.visits_completed}+${full.r.in_port_now} inFencePointsNoLongerRetained=${inFencePoints}`);
  assert.ok(bestWarm < full.ms / 3, `warm incremental ${bestWarm.toFixed(0)}ms should be well under a third of the full fold ${full.ms.toFixed(0)}ms`);
  assert.ok(s.partialBytesEstimate < rawBytes / 4, `partials ${s.partialBytesEstimate}B must be far smaller than the raw window ${rawBytes}B`);
  assert.equal(s.filesStreamedLastRun, 2);
  assert.equal(s.fallbacks, 0);
  fs.rmSync(base, { recursive: true, force: true });
});
