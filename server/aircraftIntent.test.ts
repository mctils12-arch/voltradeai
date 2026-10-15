// AIRCRAFT INTENT A (2026-10-14) — the one intent/derived-field parser:
// payloads with the fields PRESENT, PARTIAL, ABSENT and STALE; null never 0;
// nav_modes [] vs null; garbage rejected AND counted; capability tracking;
// freshest-wins; OpenSky's honest subset; the mapped aircraft payload stays
// byte-identical (intent rides beside the row, never on it).
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  parseReadsbIntent, parseOpenSkyIntent, parseAcasRa, IntentStore, intentOf, presentMask,
  INTENT_FIELDS, INTENT_UNITS, INTENT_FRESH_MS, INTENT_TTL_MS,
} from "./aircraftIntent";
import { mapPointAircraft } from "./aircraftTiling";
import { mapOpenSkyStates } from "./openskyGlobal";
import type { FixBatch } from "./aircraftFixBus";

const T0 = 1_792_000_000_000;

// A European airliner as adsb.lol serves it (EHS-equipped: every field).
const FULL = {
  hex: "4ca7b6", type: "adsb_icao", flight: "RYR1AB  ", r: "EI-DCL", t: "B738",
  alt_baro: 37000, alt_geom: 37625, gs: 452.3, ias: 258, tas: 456, mach: 0.788,
  wd: 265, ws: 32, oat: -53, tat: -26, track: 91.2, track_rate: 0.03, roll: 0,
  mag_heading: 88.6, true_heading: 90.1, baro_rate: 0, geom_rate: -32,
  squawk: "2341", emergency: "none", category: "A3",
  nav_qnh: 1013.6, nav_altitude_mcp: 37008, nav_altitude_fms: 37000, nav_heading: 87.9,
  nav_modes: ["autopilot", "vnav", "lnav", "tcas"],
  lat: 51.1, lon: 5.2, nic: 8, rc: 186, seen_pos: 0.3, version: 2, seen: 0.2,
};
// A US airliner: selected altitude + QNH only, no mode bits, no EHS state.
const PARTIAL = {
  hex: "a1b2c3", type: "adsb_icao", flight: "AAL123", t: "A321", alt_baro: 11000, alt_geom: 11475,
  gs: 280, track: 45, baro_rate: -1216, squawk: "4521", emergency: "none",
  nav_qnh: 1012.8, nav_altitude_mcp: 10016, lat: 35.5, lon: -80.6, seen: 1.5,
};
// A GA aircraft: position + baro altitude only.
const ABSENT = { hex: "abc123", type: "adsb_icao", alt_baro: 3500, gs: 120, track: 10, lat: 35.2, lon: -80.9, seen: 0.5 };

test("PRESENT: every intent/derived field parses with its unit; 0s survive as real values", () => {
  const p = parseReadsbIntent(FULL);
  assert.deepEqual(p.rejected, []);
  assert.deepEqual(p.v.nav_modes, ["autopilot", "vnav", "lnav", "tcas"]);
  assert.equal(p.v.nav_altitude_mcp, 37008);
  assert.equal(p.v.nav_altitude_fms, 37000);
  assert.equal(p.v.nav_heading, 87.9);
  assert.equal(p.v.nav_qnh, 1013.6);
  assert.equal(p.v.wd, 265); assert.equal(p.v.ws, 32);
  assert.equal(p.v.oat, -53); assert.equal(p.v.tat, -26);
  assert.equal(p.v.ias, 258); assert.equal(p.v.tas, 456); assert.equal(p.v.mach, 0.788);
  assert.equal(p.v.roll, 0, "wings level is a reported 0, not null");
  assert.equal(p.v.baro_rate, 0, "level flight is a reported 0, not null");
  assert.equal(p.v.geom_rate, -32);
  assert.equal(p.v.alt_geom, 37625); assert.equal(p.v.alt_baro, 37000);
  assert.equal(p.v.track_rate, 0.03);
  assert.equal(p.v.mag_heading, 88.6); assert.equal(p.v.true_heading, 90.1);
  assert.equal(p.v.squawk, "2341"); assert.equal(p.v.emergency, "none");
  assert.equal(p.v.acas_ra, null, "no RA sent");
  assert.equal(p.seen_s, 0.2);
  assert.equal(p.on_ground, false);
  // units live in one table and cover every field
  assert.deepEqual(Object.keys(INTENT_UNITS).sort(), [...INTENT_FIELDS].sort());
  assert.equal(INTENT_UNITS.nav_altitude_mcp, "ft");
  assert.equal(INTENT_UNITS.ws, "kt");
  assert.equal(INTENT_UNITS.nav_qnh, "hPa");
  assert.equal(INTENT_UNITS.oat, "degC");
  assert.equal(INTENT_UNITS.nav_heading, "deg");
});

test("PARTIAL: only what the aircraft sends is set; everything else is null, never 0", () => {
  const p = parseReadsbIntent(PARTIAL);
  assert.equal(p.v.nav_altitude_mcp, 10016);
  assert.equal(p.v.nav_qnh, 1012.8);
  assert.equal(p.v.baro_rate, -1216);
  for (const f of ["nav_modes", "nav_heading", "nav_altitude_fms", "wd", "ws", "oat", "tat", "ias", "tas", "mach", "roll", "track_rate", "mag_heading", "true_heading", "geom_rate", "acas_ra"] as const) {
    assert.equal(p.v[f], null, `${f} absent -> null`);
  }
});

test("ABSENT: no intent fields -> all null (mode bits null, not an empty set)", () => {
  const p = parseReadsbIntent(ABSENT);
  for (const f of INTENT_FIELDS) {
    if (f === "alt_baro") continue;
    assert.equal(p.v[f], null, `${f}`);
  }
  assert.equal(p.v.alt_baro, 3500);
  assert.equal(presentMask(p.v).toString(2).split("1").length - 1, 1, "only alt_baro present");
  assert.deepEqual(parseReadsbIntent(null).v.nav_modes, null);
  assert.deepEqual(parseReadsbIntent({ alt_baro: "ground" }).on_ground, true);
  assert.equal(parseReadsbIntent({ alt_baro: "ground" }).v.alt_baro, null, "on the ground: no baro altitude, not 0");
});

test("nav_modes: [] (bits sent, none engaged) is distinct from null; unknown modes dropped and counted", () => {
  assert.deepEqual(parseReadsbIntent({ nav_modes: [] }).v.nav_modes, []);
  const mixed = parseReadsbIntent({ nav_modes: ["Autopilot", "lnav", "warp"] });
  assert.deepEqual(mixed.v.nav_modes, ["autopilot", "lnav"], "known modes kept, case-normalized");
  assert.deepEqual(mixed.rejected, ["nav_modes"], "the unknown string is counted");
  const junk = parseReadsbIntent({ nav_modes: ["warp"] });
  assert.equal(junk.v.nav_modes, null, "never invented into a known mode");
  assert.equal(parseReadsbIntent({ nav_modes: "autopilot" }).v.nav_modes, null);
});

test("garbage: wrong type / out of range -> null and counted (schema-drift safe)", () => {
  const p = parseReadsbIntent({
    nav_qnh: 0, nav_heading: 400, mach: "0.8", oat: -300, ws: -5, nav_altitude_mcp: Number.NaN,
    squawk: "7800", emergency: "panic", acas_ra: "RA!", roll: 12.5,
  });
  for (const f of ["nav_qnh", "nav_heading", "mach", "oat", "ws", "nav_altitude_mcp", "squawk", "emergency", "acas_ra"] as const) {
    assert.equal(p.v[f], null, f);
    assert.ok(p.rejected.includes(f), `${f} counted`);
  }
  assert.equal(p.v.roll, 12.5, "a valid neighbour is untouched");
});

test("ACAS RA: readsb acas_ra object -> compact typed RA", () => {
  const ra = parseAcasRa({
    utc: "2026-10-14 21:00:00.1", unix_timestamp: 1792000000.1, df_type: 16,
    advisory: "Climb", advisory_complement: "", RAT: "0", MTE: "1", threat_id_hex: "A1B2C3",
  });
  assert.deepEqual(ra, {
    advisory: "Climb", advisory_complement: null, rat: false, mte: true, threat_hex: "a1b2c3", at_ms: 1792000000100,
  });
  assert.equal(parseAcasRa([1, 2]), null);
  assert.equal(parseReadsbIntent({ acas_ra: { advisory: "Descend" } }).v.acas_ra?.advisory, "Descend");
});

test("STALE: a value is current only within 30 s of its observation; stale fields are listed with age, never served", () => {
  const s = new IntentStore();
  s.observe("A1B2C3", parseReadsbIntent(PARTIAL), T0, "adsblol");
  const fresh = s.view("a1b2c3", T0 + INTENT_FRESH_MS - 1000)!;
  assert.equal(fresh.values.nav_altitude_mcp, 10016);
  assert.deepEqual(fresh.stale, []);
  assert.equal(fresh.age_s.nav_altitude_mcp, 29);
  assert.equal(fresh.source.nav_altitude_mcp, "adsblol");
  const stale = s.view("a1b2c3", T0 + INTENT_FRESH_MS + 1000)!;
  assert.deepEqual(stale.values, {}, "nothing served as current");
  assert.ok(stale.stale.includes("nav_altitude_mcp") && stale.stale.includes("nav_qnh"));
  assert.equal(stale.age_s.nav_altitude_mcp, 31);
  assert.equal(stale.fresh_s, 30);
  assert.equal(s.view("ffffff", T0), null, "never seen -> null");
});

test("capability: mode bits sent once keep mode_bits true; a field dropping out goes stale on its own clock", () => {
  const s = new IntentStore();
  s.observe("4ca7b6", parseReadsbIntent(FULL), T0, "adsblol");
  // 20 s later the same aircraft arrives without mode bits (another receiver)
  const noModes = { ...FULL, nav_modes: undefined };
  s.observe("4ca7b6", parseReadsbIntent(noModes), T0 + 20_000, "adsbfi");
  const v = s.view("4ca7b6", T0 + 40_000)!;
  assert.equal(v.capability.mode_bits, true, "it HAS sent mode bits");
  assert.ok(v.stale.includes("nav_modes"), "the last mode bits are 40 s old -> stale");
  assert.equal(v.values.nav_altitude_mcp, 37008, "refreshed 20 s ago -> fresh");
  assert.equal(v.source.nav_altitude_mcp, "adsbfi");
  assert.equal(v.provider, "adsbfi");
  assert.equal(v.capability.samples, 2);
  const ga = new IntentStore();
  ga.observe("abc123", parseReadsbIntent(ABSENT), T0, "adsblol");
  assert.equal(ga.view("abc123", T0)!.capability.mode_bits, false, "never sent mode bits");
  assert.deepEqual(ga.view("abc123", T0)!.capability.fields_ever, ["alt_baro"]);
});

test("freshest wins: a late batch never overwrites a newer value", () => {
  const s = new IntentStore();
  s.observe("a1b2c3", parseReadsbIntent({ ...PARTIAL, nav_altitude_mcp: 8000 }), T0 + 10_000, "adsblol");
  s.observe("a1b2c3", parseReadsbIntent({ ...PARTIAL, nav_altitude_mcp: 10016 }), T0, "airplaneslive");
  const v = s.view("a1b2c3", T0 + 10_000)!;
  assert.equal(v.values.nav_altitude_mcp, 8000);
  assert.equal(v.provider, "adsblol");
});

test("mapPointAircraft: intent attached beside the row; the serialized payload is byte-identical", () => {
  const rows = mapPointAircraft({ ac: [FULL, PARTIAL, ABSENT] }, "ac", "adsblol");
  assert.equal(rows.length, 3);
  for (const r of rows) assert.ok(!("intent" in r), "never a property on the row");
  assert.ok(!JSON.stringify(rows).includes("nav_"), "payload unchanged");
  assert.equal(intentOf(rows[0])!.v.nav_altitude_fms, 37000);
  assert.equal(intentOf(rows[1])!.v.nav_altitude_mcp, 10016);
  assert.equal(intentOf(rows[2])!.v.nav_altitude_mcp, null);
});

test("OpenSky: baro/geometric altitude (m -> ft) + squawk only; no mode bits invented", () => {
  const sv = ["a1b2c3", "AAL123  ", "United States", T0 / 1000 - 3, T0 / 1000 - 1, -80.6, 35.5, 3352.8, false, 144, 45, -6.2, null, 3497.6, "4521", false, 0, 4];
  const p = parseOpenSkyIntent(sv);
  assert.equal(p.v.alt_baro, 11000);
  assert.equal(p.v.alt_geom, 11475);
  assert.equal(p.v.squawk, "4521");
  assert.equal(p.v.nav_modes, null);
  assert.equal(p.v.nav_altitude_mcp, null);
  const ground = parseOpenSkyIntent([...sv.slice(0, 8), true, ...sv.slice(9)]);
  assert.equal(ground.v.alt_baro, null);
  const rows = mapOpenSkyStates({ states: [sv] });
  assert.equal(intentOf(rows[0])!.v.squawk, "4521");
});

const batch = (aircraft: unknown[], o: Partial<FixBatch> = {}): FixBatch =>
  ({ provider: "adsblol", origin: "sweep-disc", aircraft, fetchedAt: T0, ...o }) as FixBatch;

test("ingest from the fix bus: fix time = receive time minus the feed's `seen`; OpenSky uses its own fix time", () => {
  const s = new IntentStore();
  const n = s.ingest(batch(mapPointAircraft({ ac: [PARTIAL] }, "ac", "adsblol")), T0);
  assert.equal(n, 1);
  assert.equal(s.view("a1b2c3", T0)!.last_seen_ms, T0 - 1500);
  const sv = ["c0ffee", "ACA1", "Canada", T0 / 1000 - 12, T0 / 1000 - 2, -75, 45, 10000, false, 230, 90, 0, null, 10100, "2000", false, 0, 4];
  s.ingest(batch(mapOpenSkyStates({ states: [sv] }), { provider: "opensky", origin: "opensky" }), T0);
  assert.equal(s.view("c0ffee", T0)!.last_seen_ms, T0 - 12_000);
  assert.equal(s.ingest(batch([{ icao24: "dddddd", lat: 1, lon: 1 }]), T0), 0, "rows without a parse are skipped");
});

test("coverage: airborne aircraft only, latest vs ever, per provider, rejected counts", () => {
  const s = new IntentStore();
  s.observe("4ca7b6", parseReadsbIntent(FULL), T0, "adsblol");
  s.observe("a1b2c3", parseReadsbIntent(PARTIAL), T0, "adsblol");
  s.observe("abc123", parseReadsbIntent(ABSENT), T0, "adsbfi");
  s.observe("e0e0e0", parseReadsbIntent({ alt_baro: "ground", nav_qnh: 1010 }), T0, "adsblol");
  s.observe("bad001", parseReadsbIntent({ alt_baro: 9000, nav_qnh: 5 }), T0 - 10 * 60_000, "adsblol");
  const c = s.coverage(T0);
  assert.equal(c.aircraft, 3, "ground and out-of-window aircraft excluded");
  assert.equal(c.latest_pct.nav_altitude_mcp, 66.7);
  assert.equal(c.latest_pct.nav_modes, 33.3);
  assert.equal(c.latest_pct.wd, 33.3);
  assert.equal(c.latest_pct.alt_baro, 100);
  assert.equal(c.mode_bit_capable_pct, 33.3);
  assert.equal(c.by_provider.adsblol.aircraft, 2);
  assert.equal(c.by_provider.adsbfi.latest_pct.nav_qnh, 0);
  assert.equal(c.rejected.nav_qnh, 1);
  assert.equal(c.tracked, 5);
  assert.match(c.note, /never 0/);
});

test("bounds: TTL eviction and the hard hex cap (oldest-seen first)", () => {
  const s = new IntentStore({ maxHexes: 2 });
  s.observe("000001", parseReadsbIntent(PARTIAL), T0, "adsblol");
  s.observe("000002", parseReadsbIntent(PARTIAL), T0 + 1, "adsblol");
  s.observe("000001", parseReadsbIntent(PARTIAL), T0 + 2, "adsblol"); // refresh -> newest
  s.observe("000003", parseReadsbIntent(PARTIAL), T0 + 3, "adsblol");
  assert.equal(s.size, 2);
  assert.equal(s.view("000002", T0), null, "least recently observed evicted");
  assert.ok(s.view("000001", T0));
  assert.equal(s.evict(T0 + INTENT_TTL_MS + 10), 2);
  assert.equal(s.size, 0);
});

// ── INTENT C1: heading-mode inference "vectored (inferred)" ──────────────
import { HeadingModeTracker, HDG_MATCH_DEG, VECTOR_HOLD_MS } from "./aircraftIntent";

const HDG = (o: Record<string, unknown>) => parseReadsbIntent({ nav_modes: ["autopilot", "vnav"], nav_heading: 270, ...o }).v;

test("C1 vectored: autopilot on + LNAV off + tracking the selected heading within 5° for 20 s → active (labeled inferred)", () => {
  const h = new HeadingModeTracker();
  h.update(HDG({ mag_heading: 272 }), T0, null, null);
  assert.equal(h.view(T0).reason, "holding_too_short");
  h.update(HDG({ mag_heading: 268.5 }), T0 + 10_000, null, null);
  assert.equal(h.view(T0 + 10_000).active, false);
  h.update(HDG({ mag_heading: 271 }), T0 + VECTOR_HOLD_MS, null, null);
  const v = h.view(T0 + VECTOR_HOLD_MS);
  assert.equal(v.active, true);
  assert.equal(v.reason, "vectored");
  assert.equal(v.since_ms, T0);
  assert.equal(v.heading_source, "mag_heading");
  assert.equal(v.label, "vectored (inferred)");
  assert.equal(HDG_MATCH_DEG, 5);
});

test("C1 vectored: LNAV engaged, autopilot off, a heading off by more than 5° each break the hold", () => {
  const h = new HeadingModeTracker();
  h.update(HDG({ mag_heading: 270 }), T0, null, null);
  h.update(HDG({ mag_heading: 270, nav_modes: ["autopilot", "lnav"] }), T0 + 25_000, null, null);
  assert.equal(h.view(T0 + 25_000).reason, "lnav_engaged");
  h.update(HDG({ mag_heading: 270, nav_modes: ["vnav"] }), T0 + 26_000, null, null);
  assert.equal(h.view(T0 + 26_000).reason, "autopilot_off");
  h.update(HDG({ mag_heading: 278 }), T0 + 27_000, null, null);
  const v = h.view(T0 + 27_000);
  assert.equal(v.reason, "heading_differs");
  assert.equal(v.deviation_deg, 8);
  assert.equal(v.since_ms, null);
});

test("C1 vectored: falls back to true heading / ground track minus the local magnetic variation; nothing without a reference", () => {
  const h = new HeadingModeTracker();
  // true 262 with variation 8°W (-8): magnetic 270
  h.update(HDG({ true_heading: 262 }), T0, null, -8);
  h.update(HDG({ true_heading: 262 }), T0 + 21_000, null, -8);
  assert.equal(h.view(T0 + 21_000).active, true);
  assert.equal(h.view(T0 + 21_000).heading_source, "true_heading");
  const t = new HeadingModeTracker();
  t.update(HDG({}), T0, 263, -8);           // track 263 → magnetic 271
  t.update(HDG({}), T0 + 21_000, 264, -8);
  assert.equal(t.view(T0 + 21_000).heading_source, "track");
  assert.equal(t.view(T0 + 21_000).active, true);
  const n = new HeadingModeTracker();
  n.update(HDG({}), T0, 270, null);         // track but no variation known
  assert.equal(n.view(T0).reason, "no_heading_reference");
  // wrap-around: selected 002 vs magnetic 358 = 4°
  const w = new HeadingModeTracker();
  w.update(HDG({ nav_heading: 2, mag_heading: 358 }), T0, null, null);
  assert.equal(w.view(T0).deviation_deg, 4);
});

test("C1 vectored: a gap longer than the freshness window restarts the hold; with no update for 30 s it goes stale", () => {
  const h = new HeadingModeTracker();
  h.update(HDG({ mag_heading: 270 }), T0, null, null);
  h.update(HDG({ mag_heading: 270 }), T0 + 40_000, null, null); // 40 s gap: continuity unknown
  assert.equal(h.view(T0 + 40_000).since_ms, T0 + 40_000);
  assert.equal(h.view(T0 + 40_000).active, false);
  assert.equal(h.view(T0 + 40_000 + INTENT_FRESH_MS + 1).reason, "stale");
  // a fix without mode bits is no evidence either way: the hold is untouched
  h.update(parseReadsbIntent({ mag_heading: 300 }).v, T0 + 45_000, null, null);
  assert.equal(h.view(T0 + 45_000).since_ms, T0 + 40_000);
});

test("C1 store: the view carries the inference (null for aircraft that never sent mode bits); track + position feed the variation lookup", () => {
  const calls: Array<[number, number]> = [];
  const s = new IntentStore({ magVar: (la, lo) => { calls.push([la, lo]); return -8; } });
  const rows = (t: number) => mapPointAircraft({ ac: [
    { hex: "a1b2c3", alt_baro: 9000, track: 263, nav_modes: ["autopilot"], nav_heading: 270, lat: 35.5, lon: -80.6, seen: 0 },
    { hex: "abc123", alt_baro: 3500, track: 10, lat: 35.2, lon: -80.9, seen: 0 },
  ] }, "ac", "adsblol");
  s.ingest({ provider: "adsblol", origin: "fastlane", aircraft: rows(T0), fetchedAt: T0 } as never, T0);
  s.ingest({ provider: "adsblol", origin: "fastlane", aircraft: rows(T0 + 21_000), fetchedAt: T0 + 21_000 } as never, T0 + 21_000);
  const v = s.view("a1b2c3", T0 + 21_000)!;
  assert.equal(v.inferred.vectored!.active, true);
  assert.equal(v.inferred.vectored!.heading_source, "track");
  assert.deepEqual(calls[0], [35.5, -80.6]);
  assert.equal(s.view("abc123", T0)!.inferred.vectored, null, "never sent mode bits: no inference at all");
});

test("C1 variation lookup: the committed FAA NASR table gives Charlotte ~8°W; nothing outside NASR coverage", async () => {
  const { nasrMagVarLookup } = await import("./aircraftIntent");
  const mv = nasrMagVarLookup();
  const clt = mv(35.2, -80.9)!;
  assert.ok(clt <= -5 && clt >= -10, `KCLT area variation ${clt}`);
  assert.equal(mv(51.1, 5.2), null, "Belgium: no NASR airport within 200 nm");
});
