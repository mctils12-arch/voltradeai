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
