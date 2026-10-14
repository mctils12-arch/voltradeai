// AIRCRAFT INTENT B (2026-10-14) — intent fields in the hourly aircraft
// archive (schema_version 4): delta-encoded, write only on change, keyframe
// per aircraft per hour file, null = "no longer sent" (never 0), x absent =
// no intent information; plus the permanent daily rollup (E2-E4
// ingredients) folded once per complete UTC day.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import {
  parseReadsbIntent, encodeIntentState, decodeIntentState, intentArchiveBlock, navModesToBits, navModesFromBits,
  IntentArchiveEncoder, IntentArchiveDecoder, INTENT_ARCHIVE_KEYS, INTENT_FIELDS,
} from "./aircraftIntent";
import { archiveAircraft, archiveAircraftAt, intentArchiveStats, _resetIntentArchiveForTests } from "./datacoreArchive";
import { mapPointAircraft } from "./aircraftTiling";
import {
  foldIntentDays, loadIntentDaily, intentDailyInventory, IntentDayAccumulator, bandOf, GRID_COLS, QNH_MAX_ALT_FT,
  INTENT_ARCHIVE_SINCE_DAY,
} from "./aircraftIntentDaily";

const EU = {
  hex: "4ca7b6", type: "adsb_icao", flight: "RYR1AB", alt_baro: 37000, alt_geom: 37625, gs: 452, track: 91,
  ias: 258, tas: 456, mach: 0.788, wd: 270, ws: 50, oat: -53, tat: -26, roll: 0, track_rate: 0.03,
  mag_heading: 88.6, true_heading: 90.1, baro_rate: 0, geom_rate: -32, squawk: "2341", emergency: "none",
  nav_qnh: 1013.6, nav_altitude_mcp: 37008, nav_altitude_fms: 37000, nav_heading: 87.9,
  nav_modes: ["autopilot", "vnav", "lnav", "tcas"], lat: 51.1, lon: 5.2, seen: 0.2, seen_pos: 0.3,
};
const US_LOW = {
  hex: "a1b2c3", flight: "AAL123", alt_baro: 11000, alt_geom: 11475, gs: 280, track: 45, baro_rate: -1216,
  nav_qnh: 1008.2, nav_altitude_mcp: 10016, nav_modes: [], lat: 35.5, lon: -80.6, seen: 1, seen_pos: 1,
};
const GA = { hex: "abc123", alt_baro: 3500, gs: 120, track: 10, lat: 35.2, lon: -80.9, seen: 0.5, seen_pos: 0.5 };

const tmp = () => fs.mkdtempSync(path.join(os.tmpdir(), "vt-intent-"));
const readHour = (base: string, hourIso: string): Record<string, unknown>[] => {
  const fp = path.join(base, "aircraft", `${hourIso.replace("T", "-")}.jsonl`);
  return fs.readFileSync(fp, "utf8").trim().split("\n").map((l) => JSON.parse(l));
};

test("encode/decode round-trip is exact; nav_modes [] survives as bitmask 0; one short key per field", () => {
  for (const raw of [EU, US_LOW, GA]) {
    const v = parseReadsbIntent(raw).v;
    assert.deepEqual(decodeIntentState(encodeIntentState(v)), v);
  }
  assert.equal(encodeIntentState(parseReadsbIntent(US_LOW).v).nm, 0, "[] -> 0 (bits sent, none engaged)");
  assert.deepEqual(navModesFromBits(navModesToBits(["lnav", "autopilot"])), ["autopilot", "lnav"]);
  const keys = Object.values(INTENT_ARCHIVE_KEYS);
  assert.equal(new Set(keys).size, INTENT_FIELDS.length, "keys unique");
  // never collide with the existing schema v1-v3 line keys
  const lineKeys = ["t", "i", "c", "la", "lo", "al", "g", "v", "h", "ty", "ca", "rg", "ni", "np", "nv", "si", "st", "rc", "gv", "sd", "nb", "av", "pt", "ml", "tb", "kla", "klo", "kb", "sp", "pv", "x"];
  assert.ok(!lineKeys.includes("k"), "k is reserved for the keyframe flag inside x");
  const ra = parseReadsbIntent({ acas_ra: { advisory: "Climb", RAT: 0, MTE: 1, threat_id_hex: "abcdef", unix_timestamp: 1792000000 } }).v;
  assert.deepEqual(decodeIntentState(encodeIntentState(ra)).acas_ra, ra.acas_ra);
});

test("block: keyframe with no prior state; delta carries only changes; null clears; {} = unchanged; nothing to say -> null", () => {
  const a = encodeIntentState(parseReadsbIntent(US_LOW).v);
  assert.deepEqual(intentArchiveBlock(a, null), { k: 1, ...a });
  assert.deepEqual(intentArchiveBlock(a, a), {}, "unchanged");
  const b = encodeIntentState(parseReadsbIntent({ ...US_LOW, nav_altitude_mcp: 8000, nav_qnh: undefined }).v);
  assert.deepEqual(intentArchiveBlock(b, a), { na: 8000, nq: null }, "changed value + a field no longer sent (null, never 0)");
  assert.equal(intentArchiveBlock({}, null), null, "an aircraft that sends nothing writes no block");
});

test("encoder: keyframe per aircraft per hour file; decoder rebuilds the exact state at every line", () => {
  const enc = new IntentArchiveEncoder();
  const seq = [
    { ...US_LOW },
    { ...US_LOW },                                            // unchanged
    { ...US_LOW, nav_altitude_mcp: 8000, baro_rate: -1500 },  // new clearance
    { ...US_LOW, nav_altitude_mcp: 8000, baro_rate: -1500, nav_qnh: undefined },
  ];
  const lines = seq.map((raw) => ({ i: "a1b2c3", x: enc.block("a1b2c3", parseReadsbIntent(raw).v, "2026-10-14T21") }));
  assert.equal(lines[0].x!.k, 1);
  assert.deepEqual(lines[1].x, {});
  assert.deepEqual(lines[2].x, { na: 8000, br: -1500 });
  assert.deepEqual(lines[3].x, { nq: null });
  const next = enc.block("a1b2c3", parseReadsbIntent(seq[3]).v, "2026-10-14T22");
  assert.equal(next!.k, 1, "a new hour file starts with a keyframe");
  assert.equal(enc.block("abc123", parseReadsbIntent(GA).v, "2026-10-14T21")!.k, 1, "GA sends baro altitude: keyframe");
  assert.equal(enc.block("ffffff", parseReadsbIntent({ lat: 1 }).v, "2026-10-14T21"), null);
  const dec = new IntentArchiveDecoder();
  lines.forEach((l, i) => assert.deepEqual(dec.feed(l), parseReadsbIntent(seq[i]).v, `line ${i}`));
  assert.equal(new IntentArchiveDecoder().feed({ i: "a1b2c3", x: { na: 1 } }), null, "delta without keyframe: unknown, never guessed");
  assert.equal(dec.feed({ i: "a1b2c3" }), null, "x absent: no intent information");
  const s = enc.stats();
  assert.equal(s.keyframes, 3); assert.equal(s.deltas, 2); assert.equal(s.unchanged, 1);
});

test("archiveAircraft writes x (schema v4): keyframe then deltas; GA without intent fields keyframes alt only; backfill lines carry none", () => {
  _resetIntentArchiveForTests();
  const base = tmp();
  const h0 = Date.parse("2026-10-14T21:00:00Z");
  const batch = (raws: object[]) => mapPointAircraft({ ac: raws }, "ac", "adsblol");
  archiveAircraft(batch([EU, US_LOW]), [], base, h0 + 60_000, 0);
  archiveAircraft(batch([EU, { ...US_LOW, nav_altitude_mcp: 8000 }]), [], base, h0 + 120_000, 0);
  archiveAircraft(batch([{ ...EU, ws: 55 }]), [], base, h0 + 180_000, 0);
  archiveAircraftAt([{ icao24: "4ca7b6", lat: 51, lon: 5, altitude_m: 11000, on_ground: false, velocity_ms: 230, heading: 90, tSec: (h0 + 30_000) / 1000 }], base);
  archiveAircraft(batch([GA]), [], base, h0 + 200_000, 0);
  const rows = readHour(base, "2026-10-14T21");
  const eu = rows.filter((r) => r.i === "4ca7b6");
  const us = rows.filter((r) => r.i === "a1b2c3");
  const kf = eu[0].x as Record<string, unknown>;
  assert.equal(kf.k, 1);
  assert.equal(kf.nm, 1 | 2 | 16 | 32, "autopilot+vnav+lnav+tcas");
  assert.deepEqual(eu[1].x, {}, "unchanged since the last written line");
  assert.deepEqual(eu[2].x, { ws: 55 });
  assert.equal(eu[3].x, undefined, "trace backfill line: no intent information");
  assert.deepEqual(us[1].x, { na: 8000 });
  const dec = new IntentArchiveDecoder();
  const states = eu.map((r) => dec.feed(r));
  assert.equal(states[2]!.ws, 55);
  assert.equal(states[2]!.nav_altitude_mcp, 37008, "carried forward from the keyframe");
  assert.equal(states[3], null);
  const ga = rows.find((r) => r.i === "abc123")!;
  assert.deepEqual(ga.x, { k: 1, ab: 3500 });
  // existing schema fields untouched
  assert.equal(eu[0].al, Math.round(37000 * 0.3048));
  assert.equal(eu[0].pv, "adsblol");
  assert.ok(intentArchiveStats().bytes > 0);
});

test("band edges and the QNH ceiling", () => {
  assert.equal(bandOf(0), 0); assert.equal(bandOf(4_999), 0); assert.equal(bandOf(5_000), 1);
  assert.equal(bandOf(37_000), 7); assert.equal(bandOf(60_000), 8);
  assert.equal(QNH_MAX_ALT_FT, 18_000);
});

test("day accumulator: vector-mean wind, temperature, QNH below FL180 only, geo-baro offset, rate dispersion, field coverage", () => {
  const acc = new IntentDayAccumulator();
  const enc = new IntentArchiveEncoder();
  const mk = (raw: Record<string, unknown> & { hex: string; lat: number; lon: number }) => ({ i: raw.hex, la: raw.lat, lo: raw.lon, x: enc.block(raw.hex, parseReadsbIntent(raw).v, "H") });
  acc.addHour(21, [
    mk(EU),                                     // FL370, wind 270/50, oat -53
    mk({ ...EU, hex: "4ca7b7", wd: 90, ws: 50, oat: -55, alt_geom: 37725 }), // opposite wind, same cell
    mk(US_LOW),                                 // 11,000 ft: QNH counts
    { i: "4ca7b8", la: 51.1, lo: 5.2, g: true, x: { k: 1, nq: 1013 } },       // on the ground: excluded
  ]);
  const d = acc.result("2026-10-14", 0);
  const col = (r: number[], c: string) => r[GRID_COLS.indexOf(c as never)];
  const fl370 = d.grid.rows.find((r) => r[1] === 50 && r[2] === 4 && r[3] === 7)!;
  assert.equal(col(fl370, "n"), 2);
  assert.equal(col(fl370, "wn"), 2);
  assert.ok(Math.abs(col(fl370, "wu")) < 0.05 && Math.abs(col(fl370, "wv")) < 0.05, "270/50 + 090/50 cancel: a vector mean, not a scalar one");
  assert.equal(col(fl370, "ts"), -108);
  assert.equal(col(fl370, "qn"), 0, "QNH at FL370 is standard pressure, not a pressure-field sample");
  assert.equal(col(fl370, "gs"), 625 + 725);
  const low = d.grid.rows.find((r) => r[1] === 34 && r[2] === -82)!;
  assert.equal(col(low, "b"), 2);
  assert.equal(col(low, "qn"), 1); assert.equal(col(low, "qs"), 1008.2);
  assert.equal(col(low, "rs"), -1216);
  assert.equal(col(low, "wn"), 0, "US traffic without EHS: no wind sample, nothing filled");
  assert.equal(d.airborne_intent_lines, 3);
  assert.equal(d.aircraft, 4);
  assert.equal(d.mode_bit_aircraft, 3);
  assert.equal(d.field_pct.nav_altitude_mcp, 100);
  assert.equal(d.field_pct.wd, 66.7);
  // single eastward-blowing wind: from 270 at 50 kt -> u = +50, v = 0
  const one = new IntentDayAccumulator();
  one.addHour(0, [{ i: "1", la: 10, lo: 10, x: { k: 1, ab: 30000, wd: 270, ws: 50 } }]);
  const r = one.result("d", 0).grid.rows[0];
  assert.equal(col(r, "wu"), 50); assert.equal(col(r, "wv"), 0);
});

test("foldIntentDays: complete days only, once (idempotent), gz + plain hours, plain wins mid-compression, permanent file", async () => {
  _resetIntentArchiveForTests();
  const base = tmp();
  // later than every other test here: the writer's per-aircraft thinning clock is module-level
  const d0 = Date.parse("2026-10-20T00:00:00Z");
  const batch = (raws: object[]) => mapPointAircraft({ ac: raws }, "ac", "adsblol");
  archiveAircraft(batch([EU, US_LOW, GA]), [], base, d0 + 5 * 3600_000, 0);
  archiveAircraft(batch([{ ...EU, oat: -50 }]), [], base, d0 + 5 * 3600_000 + 60_000, 0);
  archiveAircraft(batch([EU]), [], base, d0 + 9 * 3600_000, 0);
  const dir = path.join(base, "aircraft");
  // hour 05 compressed; hour 09 mid-compression (plain complete + partial gz)
  const h05 = path.join(dir, "2026-10-20-05.jsonl");
  fs.writeFileSync(h05 + ".gz", zlib.gzipSync(fs.readFileSync(h05)));
  fs.unlinkSync(h05);
  const h09 = path.join(dir, "2026-10-20-09.jsonl");
  fs.writeFileSync(h09 + ".gz", zlib.gzipSync(Buffer.from("{\"i\":\"4ca7b6\",\"x\":{\"k\"")));
  // a later, incomplete day
  archiveAircraft(batch([EU]), [], base, d0 + 86_400_000 + 3600_000, 0);

  const early = await foldIntentDays(base, d0 + 86_400_000 + 3600_000);
  assert.deepEqual(early.daysFolded, [], "day ended 1 h ago: its last hour may still grow");
  const r = await foldIntentDays(base, d0 + 86_400_000 + 3 * 3600_000);
  assert.deepEqual(r.daysFolded, ["2026-10-20"]);
  assert.ok(r.bytes > 0);
  const d = loadIntentDaily("2026-10-20", base)!;
  assert.equal(d.hours_present, 2);
  assert.equal(d.intent_lines, 5, "3 + 1 in hour 05, 1 in hour 09 (the partial gz is not double-read)");
  assert.equal(d.corrupt_lines, 0);
  assert.equal(d.airborne_intent_lines, 5);
  assert.equal(d.aircraft, 3);
  const again = await foldIntentDays(base, d0 + 86_400_000 + 4 * 3600_000);
  assert.deepEqual(again.daysFolded, [], "idempotent: a folded day is never refolded");
  assert.equal(intentDailyInventory(base).days, 1);
  assert.equal(intentDailyInventory(base).latest, "2026-10-20");
  // the oat change (-53 -> -50) reached the rollup through the delta
  const fl370 = d.grid.rows.find((x) => x[0] === 5 && x[3] === 7)!;
  assert.equal(fl370[GRID_COLS.indexOf("tn")], 2);
  assert.equal(fl370[GRID_COLS.indexOf("ts")], -103);
});

test("offload backstop: foldAircraftBeforeEvict folds an unfolded intent day before its hour files can leave the disk", async () => {
  _resetIntentArchiveForTests();
  const base = tmp();
  const d0 = Date.parse("2026-10-25T00:00:00Z");
  archiveAircraft(mapPointAircraft({ ac: [EU] }, "ac", "adsblol"), [], base, d0 + 3600_000, 0);
  const { foldAircraftBeforeEvict } = await import("./archiveOffload");
  const now = d0 + 30 * 86_400_000;
  await foldAircraftBeforeEvict(base, now - 22 * 86_400_000, now);
  assert.equal(loadIntentDaily("2026-10-25", base)?.intent_lines, 1);
});

test("fold floor: days before the intent block existed are never read or folded (no empty rollups)", async () => {
  const base = tmp();
  const dir = path.join(base, "aircraft");
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, "2026-10-01-05.jsonl"), JSON.stringify({ t: 1, i: "4ca7b6", la: 51, lo: 5, al: 11000 }) + "\n");
  const r = await foldIntentDays(base, Date.parse("2026-10-30T00:00:00Z"));
  assert.deepEqual(r.daysFolded, []);
  assert.equal(INTENT_ARCHIVE_SINCE_DAY, "2026-10-14");
  assert.equal(intentDailyInventory(base).days, 0);
});
