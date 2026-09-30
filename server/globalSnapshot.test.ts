// FLIGHT PROGRAM B1 — global snapshot: freshest-wins merge with lawful
// tie-break, provenance-safe identity carry-over, eviction, hard cap,
// filters and the compact row encoding.
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  GlobalSnapshot, toSnapRow, encodeRow, rowInBBox, ROW_FIELDS,
  SNAPSHOT_CAP, SNAPSHOT_EVICT_MS,
} from "./globalSnapshot";
import type { FixBatch } from "./aircraftFixBus";
import type { AircraftPoint } from "./datacoreArchive";

const T = 1_800_000_000_000;
type Fix = AircraftPoint & { seen_at_ms?: number | null };
const ac = (hex: string, o: Partial<Fix> = {}): Fix => ({
  icao24: hex, callsign: "TST1", lat: 40, lon: -75, altitude_m: 10668, on_ground: false,
  velocity_ms: 231, heading: 90, type: "B738", category: "A3", registration: "N1", seen_pos: 0, ...o,
});
const batch = (provider: string, aircraft: Fix[], fetchedAt = T, o: Partial<FixBatch> = {}): FixBatch =>
  ({ provider, origin: "sweep-type", aircraft, fetchedAt, ...o });

test("toSnapRow: metric pipeline row -> ft/kt, seenAt = upstream clock minus seen_pos", () => {
  const r = toSnapRow(ac("ABC123", { seen_pos: 3.5 }), batch("adsblol", [], T, { upstreamNowMs: T + 1000 }))!;
  assert.equal(r.hex, "abc123", "hex lowercased");
  assert.equal(r.altFt, 35000, "10668 m -> 35000 ft");
  assert.equal(r.gsKt, 449, "231 m/s -> 449 kt with the pipeline's own 0.5144 factor");
  assert.equal(r.seenAt, T + 1000 - 3500);
  assert.equal(r.src, "adsblol");
  const g = toSnapRow(ac("g1", { on_ground: true, altitude_m: null }), batch("adsblol", []))!;
  assert.equal(g.altFt, null);
  assert.equal(g.gnd, true);
  assert.equal(toSnapRow({ ...ac("x"), lat: Number.NaN }, batch("adsblol", [])), null, "no position -> no row");
  const os = toSnapRow({ ...ac("os1"), seen_at_ms: T - 7000 }, batch("opensky", []))!;
  assert.equal(os.seenAt, T - 7000, "per-row fix time (OpenSky time_position) wins over the batch clock");
});

test("merge: freshest fix wins; an older fix never overwrites a newer one", () => {
  const s = new GlobalSnapshot();
  s.ingest(batch("adsblol", [ac("a1", { lat: 41 })], T), T);
  s.ingest(batch("opensky", [ac("a1", { lat: 42 })], T - 30_000), T);
  assert.equal(s.get("a1")!.lat, 41, "older opensky fix ignored");
  s.ingest(batch("opensky", [ac("a1", { lat: 43 })], T + 20_000), T + 20_000);
  assert.equal(s.get("a1")!.lat, 43, "newer fix from any provider wins");
  assert.equal(s.get("a1")!.src, "opensky");
});

test("merge: near-tie prefers adsb.lol (lawful), identity never leaks into a lawful row", () => {
  const s = new GlobalSnapshot();
  s.ingest(batch("opensky", [ac("t1", { type: null, registration: null, category: "A5" })], T), T);
  s.ingest(batch("adsblol", [ac("t1", { category: null, type: "A388" })], T + 500), T + 500);
  const r = s.get("t1")!;
  assert.equal(r.src, "adsblol", "tie -> ODbL provider");
  assert.equal(r.cat, null, "an opensky category is NOT carried into an adsblol row");
  // the reverse: an opensky fix may inherit identity learned from adsb.lol
  s.ingest(batch("opensky", [ac("t1", { type: null, registration: null })], T + 60_000), T + 60_000);
  const r2 = s.get("t1")!;
  assert.equal(r2.src, "opensky");
  assert.equal(r2.type, "A388", "static airframe identity carried from the lawful fix");
});

test("eviction: rows older than 10 min are dropped; too-old fixes are never admitted", () => {
  const s = new GlobalSnapshot();
  s.ingest(batch("adsblol", [ac("old")], T), T);
  s.ingest(batch("adsblol", [ac("new")], T + 9 * 60_000), T + 9 * 60_000);
  assert.equal(s.evict(T + SNAPSHOT_EVICT_MS + 1), 1);
  assert.equal(s.size, 1);
  assert.equal(s.get("new")!.hex, "new");
  assert.equal(s.ingest(batch("adsblol", [ac("stale")], T), T + SNAPSHOT_EVICT_MS + 5_000), 0);
  assert.equal(SNAPSHOT_EVICT_MS, 600_000);
});

test("hard cap: oldest fixes dropped first, drop counted (no silent caps)", () => {
  assert.equal(SNAPSHOT_CAP, 25_000);
  const s = new GlobalSnapshot({ cap: 100 });
  const rows = Array.from({ length: 150 }, (_, i) => ac(`h${i}`, { seen_pos: 150 - i })); // h0 oldest
  s.ingest(batch("adsblol", rows, T), T);
  assert.equal(s.size, 100);
  assert.equal(s.droppedAtCap, 50);
  assert.equal(s.get("h0"), undefined, "oldest dropped");
  assert.ok(s.get("h149"), "newest kept");
});

test("filters: bbox (incl. antimeridian), lawful-only, since", () => {
  const s = new GlobalSnapshot();
  s.ingest(batch("adsblol", [ac("us", { lat: 40, lon: -75 }), ac("fj", { lat: -17, lon: 178 })], T), T);
  s.ingest(batch("opensky", [ac("eu", { lat: 50, lon: 8 }), ac("sa", { lat: -17, lon: -179 })], T + 5_000), T + 5_000);
  assert.deepEqual(s.rows({ bbox: { lamin: 30, lamax: 60, lomin: -80, lomax: 10 } }).map((r) => r.hex).sort(), ["eu", "us"]);
  assert.deepEqual(s.rows({ bbox: { lamin: -20, lamax: -10, lomin: 170, lomax: -170 } }).map((r) => r.hex).sort(), ["fj", "sa"],
    "antimeridian-crossing bbox");
  assert.deepEqual(s.rows({ lawfulOnly: true }).map((r) => r.hex).sort(), ["fj", "us"], "ODbL-only subset is a provable filter");
  assert.deepEqual(s.rows({ sinceMs: T + 1 }).map((r) => r.hex).sort(), ["eu", "sa"]);
  assert.equal(rowInBBox({ lat: 0, lon: 0 }, { lamin: -1, lamax: 1, lomin: -1, lomax: 1 }), true);
});

test("changedSinceMs: rows by SERVER ingest time (>=), not fix time; never on the wire", () => {
  const s = new GlobalSnapshot();
  s.ingest(batch("adsblol", [ac("a")], T), T);
  s.ingest(batch("adsblol", [ac("b", { seen_pos: 50 })], T + 10_000), T + 10_000);
  assert.deepEqual(s.rows({ changedSinceMs: T + 10_000 }).map((r) => r.hex), ["b"], ">= keeps same-ms ingests");
  assert.deepEqual(s.rows({ changedSinceMs: T + 1 }).map((r) => r.hex), ["b"]);
  // an older fix that loses the freshest-wins merge does not count as a change
  s.ingest(batch("adsblol", [ac("a")], T - 60_000), T + 20_000);
  assert.deepEqual(s.rows({ changedSinceMs: T + 20_000 }).map((r) => r.hex), []);
  assert.equal(encodeRow(s.get("a")!).length, ROW_FIELDS.length, "ingestAt is internal");
});

test("encodeRow follows ROW_FIELDS exactly (the wire contract)", () => {
  const r = toSnapRow(ac("e1"), batch("adsblol", [], T))!;
  const enc = encodeRow(r);
  assert.equal(enc.length, ROW_FIELDS.length);
  const obj = Object.fromEntries(ROW_FIELDS.map((f, i) => [f, enc[i]]));
  assert.equal(obj.hex, "e1");
  assert.equal(obj.lon, -75);
  assert.equal(obj.lat, 40);
  assert.equal(obj.altFt, 35000);
  assert.equal(obj.trk, 90);
  assert.equal(obj.callsign, "TST1");
  assert.equal(obj.type, "B738");
  assert.equal(obj.seenAt, T);
  assert.equal(obj.gnd, 0);
  assert.equal(obj.src, "adsblol");
  assert.deepEqual(ROW_FIELDS.slice(0, 9), ["hex", "lon", "lat", "altFt", "gsKt", "trk", "callsign", "type", "seenAt"],
    "the brief's contract fields lead, in order");
});

test("summary + typeCounts feed the coverage block and the sweep's type promotion", () => {
  const s = new GlobalSnapshot();
  s.ingest(batch("adsblol", [ac("a", { type: "ZZ1" }), ac("b", { type: "ZZ1" }), ac("c", { type: null })], T), T);
  assert.equal(s.typeCounts().get("ZZ1"), 2);
  const sum = s.summary(T + 1000);
  assert.equal(sum.rows, 3);
  assert.equal(sum.rows_by_source.adsblol, 3);
  assert.equal(sum.rows_fresh_2m, 3);
});
