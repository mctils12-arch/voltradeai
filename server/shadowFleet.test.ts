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
  countHullSwapCandidates, TimedPoint,
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

// Standalone haversine (deliberately re-implemented, not imported, so the
// reference below doesn't share a bug with the code under test).
function refKm(aLat: number, aLon: number, bLat: number, bLon: number): number {
  const R = 6371, dLat = ((bLat - aLat) * Math.PI) / 180, dLon = ((bLon - aLon) * Math.PI) / 180;
  const s = Math.sin(dLat / 2) ** 2 +
    Math.cos((aLat * Math.PI) / 180) * Math.cos((bLat * Math.PI) / 180) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(s));
}

// Brute-force reference for countHullSwapCandidates's exact original
// semantics (dtH<=0||dtH>withinHours excluded, self-pairs excluded) — the
// spec this item's optimized sliding-window implementation must match.
function refHullSwapCount(lastPts: TimedPoint[], firstPts: TimedPoint[],
                          nearKm: number, withinHours: number): number {
  let count = 0;
  for (const A of lastPts) {
    for (const B of firstPts) {
      if (A.mmsi === B.mmsi) continue;
      const dtH = (B.t - A.t) / 3600;
      if (dtH <= 0 || dtH > withinHours) continue;
      if (refKm(A.la, A.lo, B.la, B.lo) <= nearKm) count++;
    }
  }
  return count;
}

// Deterministic PRNG (mulberry32) so a failing seed is reproducible.
function mulberry32(seed: number) {
  return () => {
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

test("RATCHET [REPAIR 2026-07-13, KNOWN BROKEN #18 root cause]: countHullSwapCandidates matches brute-force all-pairs count on randomized input", () => {
  const rnd = mulberry32(42);
  const n = 300;
  const lastPts: TimedPoint[] = [], firstPts: TimedPoint[] = [];
  for (let i = 0; i < n; i++) {
    // Half the fixture is a tight cluster (guarantees real matches so the
    // grid's 3x3 neighborhood logic is actually exercised, not just its
    // empty-cell fast path); half is spread across a realistic global
    // range (exercises cell-boundary edges generally).
    if (i % 2 === 0) {
      lastPts.push({ mmsi: `L${i}`, t: Math.floor(rnd() * 72 * 3600), la: 36.5 + (rnd() - 0.5) * 0.3, lo: 22.7 + (rnd() - 0.5) * 0.3 });
      firstPts.push({ mmsi: `L${i % 60}`, t: Math.floor(rnd() * 72 * 3600), la: 36.5 + (rnd() - 0.5) * 0.3, lo: 22.7 + (rnd() - 0.5) * 0.3 });
    } else {
      lastPts.push({ mmsi: `L${i}`, t: Math.floor(rnd() * 72 * 3600), la: rnd() * 60 - 30, lo: rnd() * 120 - 60 });
      firstPts.push({ mmsi: `L${i % 60}`, t: Math.floor(rnd() * 72 * 3600), la: rnd() * 60 - 30, lo: rnd() * 120 - 60 });
    }
  }
  const got = countHullSwapCandidates(lastPts, firstPts, 20, 12);
  const want = refHullSwapCount(lastPts, firstPts, 20, 12);
  assert.equal(got, want, `optimized grid count (${got}) must equal brute-force all-pairs count (${want})`);
  assert.ok(want > 0, "test fixture must actually exercise some matching pairs, not vacuously pass");
});

test("RATCHET [REPAIR 2026-07-13, KNOWN BROKEN #18 root cause]: antimeridian-crossing pair is still found (dateline wraparound)", () => {
  // 179.95E and -179.95E (=180.05E) are ~9km apart across the antimeridian,
  // but on opposite sides of the naive lonBucket range without wraparound.
  const lastPts: TimedPoint[] = [{ mmsi: "A", t: 1000, la: 10, lo: 179.95 }];
  const firstPts: TimedPoint[] = [{ mmsi: "B", t: 1000 + 3600, la: 10, lo: -179.95 }];
  const got = countHullSwapCandidates(lastPts, firstPts, 20, 12);
  const want = refHullSwapCount(lastPts, firstPts, 20, 12);
  assert.equal(got, 1, "a true ~9km match across the antimeridian must be found");
  assert.equal(got, want, "must match the brute-force reference too");
});

test("RATCHET [REPAIR 2026-07-13, KNOWN BROKEN #18 root cause]: countHullSwapCandidates stays fast at production-scale vessel counts (would time out at the old O(n^2) cost)", () => {
  const rnd = mulberry32(7);
  const n = 35493; // matches the production vessels_seen reading that measured the pre-fix stall
  const lastPts: TimedPoint[] = [], firstPts: TimedPoint[] = [];
  for (let i = 0; i < n; i++) {
    lastPts.push({ mmsi: `M${i}`, t: Math.floor(rnd() * 72 * 3600), la: rnd() * 60 - 30, lo: rnd() * 120 - 60 });
    firstPts.push({ mmsi: `M${i}`, t: Math.floor(rnd() * 72 * 3600), la: rnd() * 60 - 30, lo: rnd() * 120 - 60 });
  }
  const start = Date.now();
  const count = countHullSwapCandidates(lastPts, firstPts, 20, 12);
  const elapsedMs = Date.now() - start;
  // The old O(n^2) Map-based scan cost ~1.26B lookups at this exact n
  // (~85-98s measured live in production, KNOWN BROKEN #18). The grid-based
  // version must stay near-instant regardless of n, since it only ever
  // examines pairs that are both time-window- and spatially-adjacent.
  assert.ok(elapsedMs < 1000, `expected well under 1s at n=${n}, took ${elapsedMs}ms — regression toward O(n^2) suspected`);
  assert.ok(count >= 0);
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
