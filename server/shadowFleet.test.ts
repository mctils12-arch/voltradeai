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
  detectLoitering, computeShadowStats, ShadowZone,
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
