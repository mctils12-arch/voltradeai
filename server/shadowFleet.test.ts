// Dark-ship analytics — synthetic-archive tests (hermetic tmpdir; the module
// reads the same JSONL format datacoreArchive writes).
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  readVesselTracks, detectGapEvents, detectIdentityCandidates,
  detectLoitering, detectLoiteringMmsis, computeShadowStats, ShadowZone,
  ShadowAggregator, foldVesselArchiveAsync, TANKER_SHIP_TYPE_MIN, TANKER_SHIP_TYPE_MAX,
} from "./shadowFleet";

const here = path.dirname(fileURLToPath(import.meta.url));
const NOW = Date.UTC(2026, 6, 4, 12, 0, 0);

function writeArchive(base: string, points: Array<Record<string, any>>) {
  const dir = path.join(base, "vessels");
  fs.mkdirSync(dir, { recursive: true });
  const byHour = new Map<string, string[]>();
  for (const p of points) {
    const d = new Date(p.t * 1000).toISOString();
    const f = `${d.slice(0, 10)}-${d.slice(11, 13)}.jsonl`;
    if (!byHour.has(f)) byHour.set(f, []);
    byHour.get(f)!.push(JSON.stringify(p));
  }
  for (const [f, lines] of byHour) fs.writeFileSync(path.join(dir, f), lines.join("\n") + "\n");
}

const t = (hoursAgo: number) => Math.floor((NOW - hoursAgo * 3600_000) / 1000);

test("gap events: silent >6h AND reappeared >100km triggers; short/near gaps don't", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-shadow-"));
  writeArchive(base, [
    // GAPPER: dark 10h, reappears ~400km away
    { t: t(20), i: "111000111", c: "GAPPER", la: 36.0, lo: 20.0, v: 12 },
    { t: t(10), i: "111000111", c: "GAPPER", la: 36.0, lo: 24.5, v: 12 },
    // STEADY: 10h between points but only ~20km apart (slow transit / thinning)
    { t: t(20), i: "222000222", c: "STEADY", la: 40.0, lo: 5.0, v: 3 },
    { t: t(10), i: "222000222", c: "STEADY", la: 40.15, lo: 5.1, v: 3 },
  ]);
  const tracks = readVesselTracks(72, base, NOW);
  assert.equal(tracks.size, 2);
  const gaps = detectGapEvents(tracks);
  assert.equal(gaps.length, 1);
  assert.equal(gaps[0].mmsi, "111000111");
  assert.ok(gaps[0].gapHours >= 9.9 && gaps[0].distanceKm > 300);
});

test("identity candidates: name under two MMSIs + hull-swap proximity heuristic", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-shadow-id-"));
  writeArchive(base, [
    // same NAME on two MMSIs
    { t: t(30), i: "333000333", c: "TWIN STAR", la: 10, lo: 10, v: 5 },
    { t: t(29), i: "444000444", c: "TWIN STAR", la: 30, lo: 30, v: 5 },
    // hull swap: A last seen, B first seen 3h later 5km away
    { t: t(20), i: "555000555", c: "OLD ID", la: 36.5, lo: 22.7, v: 0 },
    { t: t(17), i: "666000666", c: "NEW ID", la: 36.53, lo: 22.72, v: 0 },
  ]);
  const tracks = readVesselTracks(72, base, NOW);
  const n = detectIdentityCandidates(tracks);
  assert.ok(n >= 2, `expected >=2 candidates (name reuse + hull swap), got ${n}`);
});

test("loitering: sustained slow presence inside a zone counts once per vessel", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-shadow-loiter-"));
  const zone: ShadowZone = { id: "laconian_gulf", name: "Laconian Gulf", lat: 36.55, lon: 22.75, radius_km: 45 };
  writeArchive(base, [
    { t: t(9), i: "777000777", c: "IDLER", la: 36.5, lo: 22.7, v: 0.4 },
    { t: t(7), i: "777000777", c: "IDLER", la: 36.52, lo: 22.72, v: 0.6 },
    { t: t(5), i: "777000777", c: "IDLER", la: 36.51, lo: 22.71, v: 0.2 },
    { t: t(3), i: "777000777", c: "IDLER", la: 36.5, lo: 22.7, v: 0.5 },
    // fast transiter through the same zone — must NOT count
    { t: t(6), i: "888000888", c: "TRANSIT", la: 36.4, lo: 22.6, v: 14 },
    { t: t(5), i: "888000888", c: "TRANSIT", la: 36.6, lo: 22.9, v: 14 },
    { t: t(4), i: "888000888", c: "TRANSIT", la: 36.8, lo: 23.2, v: 14 },
  ]);
  const tracks = readVesselTracks(72, base, NOW);
  const loiter = detectLoitering(tracks, [zone]);
  assert.equal(loiter.laconian_gulf, 1);
});

test("detectLoiteringMmsis: same predicate as detectLoitering, MMSI identities instead of zone counts", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-shadow-loiter-mmsi-"));
  const zoneA: ShadowZone = { id: "laconian_gulf", name: "Laconian Gulf", lat: 36.55, lon: 22.75, radius_km: 45 };
  const zoneB: ShadowZone = { id: "fujairah", name: "Fujairah", lat: 25.12, lon: 56.34, radius_km: 45 };
  writeArchive(base, [
    // loiters in BOTH zones (>=3 points, >=4h span, each) — must appear
    // once, not twice, in the MMSI set
    { t: t(20), i: "111000111", c: "DUAL", la: 36.5, lo: 22.7, v: 0.4 },
    { t: t(17), i: "111000111", c: "DUAL", la: 36.52, lo: 22.72, v: 0.6 },
    { t: t(15), i: "111000111", c: "DUAL", la: 36.51, lo: 22.71, v: 0.2 },
    { t: t(9), i: "111000111", c: "DUAL", la: 25.12, lo: 56.34, v: 0.2 },
    { t: t(6), i: "111000111", c: "DUAL", la: 25.13, lo: 56.35, v: 0.5 },
    { t: t(3), i: "111000111", c: "DUAL", la: 25.12, lo: 56.34, v: 0.3 },
    // fast transiter — must NOT appear
    { t: t(6.5), i: "888000888", c: "TRANSIT", la: 36.4, lo: 22.6, v: 14 },
    { t: t(5.5), i: "888000888", c: "TRANSIT", la: 36.6, lo: 22.9, v: 14 },
    { t: t(4.5), i: "888000888", c: "TRANSIT", la: 36.8, lo: 23.2, v: 14 },
  ]);
  const tracks = readVesselTracks(72, base, NOW);
  const mmsis = detectLoiteringMmsis(tracks, [zoneA, zoneB]);
  assert.deepEqual([...mmsis], ["111000111"]);
  // cross-check against detectLoitering's own counts on the identical input:
  // one vessel loitering in 2 zones must show up as 1 in EACH zone's count
  // (detectLoitering counts per zone) while detectLoiteringMmsis reports it
  // once overall — both are reductions of the same shared predicate.
  const counts = detectLoitering(tracks, [zoneA, zoneB]);
  assert.equal(counts.laconian_gulf, 1);
  assert.equal(counts.fujairah, 1);
});

test("computeShadowStats aggregates with the honest caveat; wiring pinned", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-shadow-agg-"));
  writeArchive(base, [
    { t: t(20), i: "111000111", c: "GAPPER", la: 36.0, lo: 20.0, v: 12 },
    { t: t(10), i: "111000111", c: "GAPPER", la: 36.0, lo: 24.5, v: 12 },
  ]);
  const s = computeShadowStats([], 72, base, NOW);
  assert.equal(s.vessels_seen, 1);
  assert.equal(s.gap_events, 1);
  assert.ok(s.caveat.includes("coverage loss"), "caveat must state the coverage-loss ambiguity");
  const routes = fs.readFileSync(path.join(here, "routes.ts"), "utf8");
  assert.ok(routes.includes("/api/data/shadowstats"), "route missing");
  const layers = JSON.parse(fs.readFileSync(path.join(here, "..", "datacore", "layers.json"), "utf8"));
  const l = layers.layers.find((x: any) => x.id === "shadowstats");
  assert.ok(l && l.kind === "raw", "layers.json entry missing/not raw");
  assert.ok(l.description.includes("coverage loss"), "user-facing description must carry the caveat");
  const zones = JSON.parse(fs.readFileSync(path.join(here, "..", "datacore", "shadow_zones.json"), "utf8"));
  assert.ok(zones.zones.length >= 5);
});

test("ShadowAggregator.gate1Inputs (2026-09-01, GATE 1 support): bounded-memory fold reports the same gap/loiter MMSIs as the materializing detectors, plus the tanker pool", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-shadow-gate1-"));
  const zone: ShadowZone = { id: "laconian_gulf", name: "Laconian Gulf", lat: 36.55, lon: 22.75, radius_km: 45 };
  writeArchive(base, [
    // tanker that gaps (dark 10h, reappears ~400km away)
    { t: t(20), i: "111000111", c: "GAPPER", la: 36.0, lo: 20.0, v: 12, st: 80 },
    { t: t(10), i: "111000111", c: "GAPPER", la: 36.0, lo: 24.5, v: 12, st: 80 },
    // tanker that loiters in the zone
    { t: t(9), i: "222000222", c: "IDLER", la: 36.5, lo: 22.7, v: 0.4, st: 89 },
    { t: t(7), i: "222000222", c: "IDLER", la: 36.52, lo: 22.72, v: 0.6, st: 89 },
    { t: t(5), i: "222000222", c: "IDLER", la: 36.51, lo: 22.71, v: 0.2, st: 89 },
    // cargo ship (NOT a tanker) that also loiters — must be in loiterMmsis
    // but NOT in tankerPool
    { t: t(9), i: "333000333", c: "CARGO", la: 36.4, lo: 22.6, v: 0.3, st: 70 },
    { t: t(7), i: "333000333", c: "CARGO", la: 36.42, lo: 22.62, v: 0.5, st: 70 },
    { t: t(5), i: "333000333", c: "CARGO", la: 36.41, lo: 22.61, v: 0.2, st: 70 },
    // quiet tanker — in the tanker pool, no gap or loiter candidate
    { t: t(6), i: "444000444", c: "QUIET", la: 50.0, lo: 50.0, v: 10, st: 84 },
  ]);
  const agg = new ShadowAggregator([zone]);
  await foldVesselArchiveAsync(72, (mmsi, p) => agg.push(mmsi, p), base, NOW);
  const inputs = agg.gate1Inputs();
  assert.deepEqual(inputs.gapMmsis.sort(), ["111000111"]);
  assert.deepEqual(inputs.loiterMmsis.sort(), ["222000222", "333000333"]);
  assert.deepEqual(inputs.tankerPool.sort(), ["111000111", "222000222", "444000444"]);

  // cross-check against the materializing detectors on the identical
  // archive — the online fold and the sync scan must agree (same
  // discipline as the existing RATCHET test pinning readVesselTracksAsync
  // against readVesselTracks).
  const tracks = readVesselTracks(72, base, NOW);
  assert.deepEqual(new Set(detectGapEvents(tracks).map((e) => e.mmsi)), new Set(inputs.gapMmsis));
  assert.deepEqual(detectLoiteringMmsis(tracks, [zone]), new Set(inputs.loiterMmsis));

  assert.equal(TANKER_SHIP_TYPE_MIN, 80);
  assert.equal(TANKER_SHIP_TYPE_MAX, 89);
});

test("RATCHET [REPAIR 2026-07-05]: async streaming reader is byte-identical to the sync scan (incl. gz)", async () => {
  // The sync scan ran ON THE REQUEST PATH and blocked the whole event loop
  // 26-90s at prod archive size (000/502 on cold hits). Routes now use the
  // async variant via a poller; equivalence keeps both honest.
  const zlib = await import("node:zlib");
  const { readVesselTracksAsync, computeShadowStatsAsync } = await import("./shadowFleet");
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-shadow-"));
  writeArchive(base, [
    { t: t(20), i: "111000111", c: "GAPPER", la: 36.0, lo: 20.0, v: 12 },
    { t: t(10), i: "111000111", c: "GAPPER", la: 36.0, lo: 24.5, v: 12 },
    { t: t(5), i: "333000333", c: "LOITER", la: 35.0, lo: 18.0, v: 0.4 },
  ]);
  // gzip one hour-file to cover the compressed path
  const dir = path.join(base, "vessels");
  const plain = fs.readdirSync(dir)[0];
  fs.writeFileSync(path.join(dir, `${plain}.gz`), zlib.gzipSync(fs.readFileSync(path.join(dir, plain))));
  fs.unlinkSync(path.join(dir, plain));
  const sync = readVesselTracks(72, base, NOW);
  const asy = await readVesselTracksAsync(72, base, NOW);
  assert.deepEqual(
    Array.from(asy.entries()).sort(),
    Array.from(sync.entries()).sort(),
    "async streaming reader must return exactly what the sync reader returns");
  const zones: ShadowZone[] = [{ id: "z1", name: "Zone", lat: 35.0, lon: 18.0, radius_km: 100 }];
  const a = await computeShadowStatsAsync(zones, 72, base, NOW);
  const s = computeShadowStats(zones, 72, base, NOW);
  assert.deepEqual(a, s);
});

// REPAIR (found 2026-07-22, same root cause + fix as datacoreArchive.ts's
// streamJsonlLines): readline.Interface re-emits a piped-in stream's error
// on ITSELF too, independent of the stream.on("error", ...) guard here —
// unlistened, a truncated/corrupt .gz crashed the WHOLE PROCESS. Both
// readVesselTracksAsync and foldVesselArchiveAsync carry an independent copy
// of the same pattern (own PR fix in shadowFleet.ts) — covered together
// here. See datacoreArchive.test.ts for the full writeup + minimal repro.
test("readVesselTracksAsync + foldVesselArchiveAsync resolve (never crash the process) on a truncated/corrupt gzip file", async () => {
  const zlib = await import("node:zlib");
  const { readVesselTracksAsync, foldVesselArchiveAsync } = await import("./shadowFleet");
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-shadow-trunc-"));
  const dir = path.join(base, "vessels");
  fs.mkdirSync(dir, { recursive: true });
  const good = zlib.gzipSync(JSON.stringify({ t: t(5), i: "111000111", la: 35.0, lo: 18.0, v: 5 }) + "\n");
  const d = new Date(t(5) * 1000).toISOString();
  fs.writeFileSync(path.join(dir, `${d.slice(0, 10)}-${d.slice(11, 13)}.jsonl.gz`), good.subarray(0, good.length - 4));
  await readVesselTracksAsync(72, base, NOW);
  await foldVesselArchiveAsync(72, () => {}, base, NOW);
});

// REPAIR (found 2026-08-19, blocking the 2026-08-18 portdwell_window probe's
// own stated purpose — reconciling a HISTORICAL window against a port
// authority's published stats): all three archive readers filtered files
// and points by a LOWER bound only (`stamp/p.t >= now - window`), never an
// UPPER bound at `now` itself. That is invisible against a `now` equal to
// real wall-clock time (nothing in the archive is ever "in the future"),
// which is why every prior test here passed — but a historical query
// (`nowMs` set to a PAST moment) against a live, still-growing archive
// pulled in every file written between that past moment and the real
// present. Concretely, this made `visits_completed` read 0 and
// `in_port_now` read ~vessel-count for ANY `portdwell_window` call with an
// `end` in the past against production: a vessel's archive track keeps
// extending past the requested `end` (today's data is always there), so
// every visit looked "ongoing" relative to the requested `now`, never
// completed. Live-reproduced against production before this fix (see
// research/experiments.md's 2026-08-19 entry for the raw probe output).
test("REPAIR 2026-08-19: a `nowMs` in the past excludes points/files written AFTER it, not just before the window", async () => {
  const { readVesselTracksAsync, foldVesselArchiveAsync } = await import("./shadowFleet");
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-shadow-pastnow-"));
  // FUTURE (relative to the queried past `now` below): the archive keeps
  // recording after the query point, exactly like a live production archive.
  const future = Math.floor(NOW / 1000) + 10 * 3600; // 10h after NOW
  writeArchive(base, [
    { t: t(20), i: "111000111", c: "INWINDOW", la: 36.0, lo: 20.0, v: 12 },
    { t: future, i: "111000111", c: "INWINDOW", la: 36.0, lo: 20.5, v: 12 },
    { t: future + 3600, i: "999000999", c: "FUTUREONLY", la: 10.0, lo: 10.0, v: 5 },
  ]);
  const sync = readVesselTracks(72, base, NOW);
  assert.equal(sync.get("111000111")!.length, 1, "sync reader must drop the point written after `now`");
  assert.equal(sync.has("999000999"), false, "sync reader must ignore a vessel seen only after `now`");

  const asy = await readVesselTracksAsync(72, base, NOW);
  assert.deepEqual(Array.from(asy.entries()).sort(), Array.from(sync.entries()).sort(),
    "async reader must match the sync reader on a past-`now` query too");

  const folded: Array<[string, number]> = [];
  const counts = new Map<string, number>();
  await foldVesselArchiveAsync(72, (mmsi) => counts.set(mmsi, (counts.get(mmsi) ?? 0) + 1), base, NOW);
  assert.equal(counts.get("111000111"), 1, "fold must drop the point written after `now`");
  assert.equal(counts.has("999000999"), false, "fold must ignore a vessel seen only after `now`");
});

test("REPAIR 2026-08-19: a port call that only exists AFTER the queried `end` must not leak in as a false in_port_now — this is the live production symptom (visits_completed:0, in_port_now≈all vessels) for any portdwell_window query with a past `end`", async () => {
  const { computePortDwellAsync } = await import("./portDwell");
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-dwell-pastnow-"));
  const LA = { id: "port_la", name: "Port of Los Angeles", lat: 33.74, lon: -118.272, radius_km: 5 };
  const queriedEnd = NOW - 480 * 3600_000; // the historical `end` this query asks for (ms, matching computePortDwellAsync's `nowMs`)
  // COMPLETED: a real 20h call that finished BEFORE `end` — must count.
  const callStart = t(520), callEnd = t(500); // both before queriedEnd
  // FUTURE-ONLY: a different vessel's call that starts AND ends AFTER
  // `end` — as of `end` this hadn't happened yet, so the query must not
  // see it at all (not completed, not in_port_now).
  const futureStart = t(460), futureEnd = t(450); // both after queriedEnd
  const pts: Array<Record<string, any>> = [];
  for (let ts = callStart; ts <= callEnd; ts += 3600) {
    pts.push({ t: ts, i: "555000555", c: "BOXSHIP", la: LA.lat + 0.005, lo: LA.lon + 0.005, v: 0.2 });
  }
  for (let ts = futureStart; ts <= futureEnd; ts += 3600) {
    pts.push({ t: ts, i: "666000666", c: "LATEARRIVAL", la: LA.lat + 0.005, lo: LA.lon + 0.005, v: 0.2 });
  }
  writeArchive(base, pts);
  const s = await computePortDwellAsync([LA], 168, base, queriedEnd);
  const la = s.ports.find((p) => p.id === "port_la")!;
  assert.equal(la.visits_completed, 1, "the pre-`end` call must count as completed");
  assert.equal(la.in_port_now, 0,
    "the post-`end` call must not leak in as a false 'still in port' reading — it hadn't happened yet as of `end`");
  assert.equal(la.unique_vessels, 1, "the future-only vessel must be entirely invisible to a query ending before it arrived");
  assert.ok(Math.abs((la.dwell_median_h ?? 0) - 20) < 0.2, `median ${la.dwell_median_h} ≠ ~20h`);
});

test("RATCHET [PERF REPAIR 2026-07-13, KNOWN BROKEN #18 root cause]: hull-swap detection stays fast as vessel count grows — was O(vessels^2), the actual cause of the recurring 10-minute EVENTLOOP-LAG stalls", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-shadow-perf-"));
  // 8,000 distinct vessels, each with exactly one point, spread evenly
  // across the 72h window and across the globe — no two are within
  // nearKm/withinHours of each other, so the true identity_candidates
  // count is 0 (pure performance probe, not a correctness fixture).
  // The naive O(N^2) form does N*(N-1) ~= 64M haversine calls on this
  // input; at real prod scale (34,895 vessels, 2026-07-13) the same shape
  // did ~1.2 BILLION calls synchronously with zero yield points, which is
  // what actually produced the 60-95s+ EVENTLOOP-LAG entries this item
  // tracks — this test pins that the fix (sort + binary-search window
  // instead of all-pairs) keeps runtime near-linear.
  const N = 8000;
  // Direct (non-modular) grid placement: row/col indexing (not `i % period`)
  // so no two vessels can ever land on the same cell via periodic wraparound
  // (an earlier draft of this test used `i % 160`/`(i*7) % 340`, whose
  // lcm(160,340)=2720 silently reintroduced ~7,840 close pairs within
  // N=8000 — caught by this test itself failing its own n===0 assertion).
  // Grid spacing (~1.9 deg lat x ~3.8 deg lon) keeps every pair >20km apart
  // even at the highest latitude used (worst case ~36km at row 89).
  const gridSize = Math.ceil(Math.sqrt(N)); // 90
  const points: Array<Record<string, any>> = [];
  const stepSec = Math.floor((71 * 3600) / N); // ~31s apart, spans ~71h
  for (let i = 0; i < N; i++) {
    const row = Math.floor(i / gridSize), col = i % gridSize;
    points.push({
      t: t(71) + i * stepSec,
      i: `V${String(i).padStart(9, "0")}`,
      c: `SHIP${i}`,
      la: -85 + row * (170 / gridSize),
      lo: -170 + col * (340 / gridSize),
      v: 5,
    });
  }
  writeArchive(base, points);
  const tracks = readVesselTracks(72, base, NOW);
  assert.equal(tracks.size, N);
  const start = performance.now();
  const n = detectIdentityCandidates(tracks);
  const elapsedMs = performance.now() - start;
  assert.equal(n, 0, "widely-scattered single-point vessels must not match each other");
  assert.ok(elapsedMs < 5000,
    `detectIdentityCandidates(${N} vessels) took ${elapsedMs.toFixed(0)}ms — ` +
    `an O(vessels^2) regression would take tens of seconds here (this is the ` +
    `exact shape that blocked the event loop in production)`);
});

/** Independent oracle — deliberately naive O(vessels^2) reimplementation,
 *  NOT sharing any code with shadowFleet.ts's grid+time-window fix, used
 *  only to fuzz-verify the optimized path's correctness. Sharing an
 *  implementation between "the fix" and "the test that proves the fix is
 *  right" would let the same bug hide in both. */
function bruteForceHullSwapCount(pts: Array<{ mmsi: string; t: number; la: number; lo: number }>,
                                 nearKm: number, withinHours: number): number {
  const kmBetween = (aLat: number, aLon: number, bLat: number, bLon: number) => {
    const R = 6371, dLat = ((bLat - aLat) * Math.PI) / 180, dLon = ((bLon - aLon) * Math.PI) / 180;
    const s = Math.sin(dLat / 2) ** 2 +
      Math.cos((aLat * Math.PI) / 180) * Math.cos((bLat * Math.PI) / 180) * Math.sin(dLon / 2) ** 2;
    return 2 * R * Math.asin(Math.sqrt(s));
  };
  let count = 0;
  for (const a of pts) {
    for (const b of pts) {
      if (a.mmsi === b.mmsi) continue;
      const dtH = (b.t - a.t) / 3600;
      if (dtH <= 0 || dtH > withinHours) continue;
      if (kmBetween(a.la, a.lo, b.la, b.lo) <= nearKm) count++;
    }
  }
  return count;
}

test("FUZZ oracle: grid+time-window hull-swap count matches independent brute-force O(n^2) reference across random/clustered layouts", () => {
  let seed = 42;
  const rand = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };

  for (let trial = 0; trial < 40; trial++) {
    const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-shadow-fuzz-"));
    const N = 40 + Math.floor(rand() * 60); // small enough for O(n^2) oracle to run instantly
    const numClusters = 1 + Math.floor(rand() * 6); // includes the "everyone in one spot" worst case
    const clusters = Array.from({ length: numClusters }, () => ({
      la: -70 + rand() * 140, lo: -175 + rand() * 350,
    }));
    // Scatter radius (~0.5 deg, comparable to the grid's ~0.18-deg cell
    // size for nearKm=20) deliberately puts many pairs right around cell
    // boundaries — the case a wrong neighbor-radius would silently drop.
    const points: Array<Record<string, any>> = [];
    for (let i = 0; i < N; i++) {
      const c = clusters[i % numClusters];
      points.push({
        t: t(rand() * 71), // NOW-relative (not epoch-1970-relative) — must fall inside readVesselTracks's window cutoff
        i: `F${trial}${String(i).padStart(6, "0")}`,
        c: null, // isolate the hull-swap heuristic from the name-reuse heuristic
        la: Math.max(-89, Math.min(89, c.la + (rand() - 0.5) * 0.5)),
        lo: Math.max(-179, Math.min(179, c.lo + (rand() - 0.5) * 0.5)),
        v: 5,
      });
    }
    writeArchive(base, points);
    const tracks = readVesselTracks(72, base, NOW);
    const actual = detectIdentityCandidates(tracks);
    const oracle = bruteForceHullSwapCount(
      [...tracks.entries()].map(([mmsi, pts]) => ({ mmsi, t: pts[0].t, la: pts[0].la, lo: pts[0].lo })),
      20, 12,
    );
    assert.equal(actual, oracle,
      `trial ${trial}: N=${N} clusters=${numClusters} — optimized count (${actual}) must match brute-force oracle (${oracle})`);
  }
});

// [2026-08-29] `st` (AIS ship-type code) now threads through the Pt shape —
// queued by the immediately-prior 2026-08-29 shadow-fleet-gate-1 session's
// own NEXT note as step (1) toward a tanker-only universe (ship-type codes
// 80-89), which gate 1's enrichment test needs but cannot build without this.
// `archiveVessels` in datacoreArchive.ts writes `st: p.shiptype ?? undefined`
// on the real archive line (confirmed by reading that file this session) —
// this fixture matches that field name exactly, not a guessed one.
test("ship-type (`st`) threads through all three readers when the archive line carries it, and stays undefined when it doesn't", async () => {
  const { readVesselTracksAsync, foldVesselArchiveAsync } = await import("./shadowFleet");
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-shadow-shiptype-"));
  writeArchive(base, [
    // TANKER: real AIS tanker-range code (80 = tanker, all types)
    { t: t(10), i: "111000111", c: "TANKER", la: 10.0, lo: 10.0, v: 8, st: 80 },
    // CARGO: a non-tanker code, must NOT be confused with the tanker above
    { t: t(9), i: "222000222", c: "CARGO", la: 20.0, lo: 20.0, v: 8, st: 70 },
    // NOTYPE: archive line omits `st` entirely (older record / unbroadcast) —
    // must come back `undefined`, never `0` or a guessed value.
    { t: t(8), i: "333000333", c: "NOTYPE", la: 30.0, lo: 30.0, v: 8 },
  ]);

  const sync = readVesselTracks(72, base, NOW);
  assert.equal(sync.get("111000111")![0].st, 80, "sync reader must carry the tanker's ship-type code");
  assert.equal(sync.get("222000222")![0].st, 70, "sync reader must carry the cargo ship's own, different code");
  assert.equal(sync.get("333000333")![0].st, undefined, "sync reader must leave a missing `st` as undefined, never a guessed 0");

  const asy = await readVesselTracksAsync(72, base, NOW);
  assert.deepEqual(Array.from(asy.entries()).sort(), Array.from(sync.entries()).sort(),
    "async streaming reader must carry `st` identically to the sync reader");

  const folded = new Map<string, number | undefined>();
  await foldVesselArchiveAsync(72, (mmsi, p) => folded.set(mmsi, p.st), base, NOW);
  assert.equal(folded.get("111000111"), 80, "fold callback must receive the tanker's ship-type code");
  assert.equal(folded.get("222000222"), 70, "fold callback must receive the cargo ship's own code");
  assert.equal(folded.get("333000333"), undefined, "fold callback must receive undefined, not a guessed 0, when the archive omits `st`");
});

test("hull-swap boundary: candidate exactly at withinHours/nearKm counts; just past either edge doesn't", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-shadow-boundary-"));
  writeArchive(base, [
    { t: t(30), i: "100000001", c: "A-EDGE-IN", la: 10.0, lo: 10.0, v: 5 },
    // exactly 12h later, ~19.9km away (inside both edges)
    { t: t(18), i: "200000002", c: "B-EDGE-IN", la: 10.179, lo: 10.0, v: 5 },
    { t: t(50), i: "300000003", c: "A-EDGE-OUT", la: 20.0, lo: 20.0, v: 5 },
    // 12h + 1s later — just past withinHours
    { t: t(50) + (12 * 3600 + 1), i: "400000004", c: "B-EDGE-OUT", la: 20.001, lo: 20.0, v: 5 },
  ]);
  const tracks = readVesselTracks(72, base, NOW);
  const lasts: Array<{ mmsi: string; t: number; la: number; lo: number }> = [];
  const firsts: Array<{ mmsi: string; t: number; la: number; lo: number }> = [];
  for (const [mmsi, pts] of tracks) {
    firsts.push({ mmsi, t: pts[0].t, la: pts[0].la, lo: pts[0].lo });
    lasts.push({ mmsi, t: pts[pts.length - 1].t, la: pts[pts.length - 1].la, lo: pts[pts.length - 1].lo });
  }
  const n = detectIdentityCandidates(tracks);
  assert.equal(n, 1, "only the within-bounds pair should count; the just-past-12h pair must not");
});
