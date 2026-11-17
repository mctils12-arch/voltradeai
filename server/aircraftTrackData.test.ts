// Aircraft track + fleet data for the app (owner request 2026-11-16):
//   A. the archive records geometric altitude and the broadcast vertical rate
//      on every line (in the v4 x block where it already is, in flat v5 keys
//      where it is not), old lines still read, and the selected-aircraft lane's
//      fixes are archived once each at their own time under the 1 GiB floor;
//   B. GET /api/data/aircraft/track/:hex — fix times, labelled altitudes,
//      broadcast vs derived vertical rate, listed gaps, stated decimation;
//   C. GET /api/data/aircraft/fleet — tail -> hex without guessing, flights
//      from the archive, live state on top;
//   and the ONE-line routes.ts wiring.
import { test } from "node:test";
import assert from "node:assert/strict";
import express from "express";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import { fileURLToPath } from "node:url";
import type { AddressInfo } from "node:net";
import {
  archiveAircraft, archiveAircraftAt, aircraftKinematicFields, _resetIntentArchiveForTests,
  type AircraftPoint,
} from "./datacoreArchive";
import { mapPointAircraft } from "./aircraftTiling";
import { IntentArchiveDecoder } from "./aircraftIntent";
import { fullTrackAsync, scanArchiveRows, _resetBloomCache } from "./aircraftTrips";
import {
  registerGlobalAircraftRoutes, aircraftLiveLookup, fastlaneArchiveAllowed,
  FASTLANE_MAX_FIX_AGE_MS, FASTLANE_ARCHIVE_MAX_LINES_PER_HOUR, SWEEP_ARCHIVE_MIN_FREE_BYTES,
  type AircraftLiveLookup,
} from "./globalAircraft";
import { MIN_FREE_BYTES } from "./globalScopes";
import type { SnapRow } from "./globalSnapshot";
import type { TrackedPlane } from "./trackedPlanes";
import { TrackedPlaneLog } from "./trackedPlaneLog";
import {
  registerAircraftTrackDataRoutes, fixFromLine, cleanFixes, buildTrack, bestAltitude, liveFix,
  parseTrackQuery, parseFleetQuery, resolveTail, faaHexForReg, scanArchiveForRegs, readHexFixes,
  readFleetHistories, archiveReach, fleetLive, mergeTrackedLogFixes, readTrackedLogFixes,
  TRACK_FIELDS, TRACK_UNITS, TRACK_GAP_S, TRACK_MAX_WINDOW_S, TRACK_MAX_POINTS, DERIVED_VR_MAX_DT_S,
  FLEET_MAX_AIRCRAFT, LIVE_MAX_AGE_S,
  type TrackFix, type TrackRow,
} from "./aircraftTrackData";

const here = path.dirname(fileURLToPath(import.meta.url));
const GiB = 1_073_741_824;
/** 2026-11-16 12:30:10 UTC — a whole second, mid-hour */
const T0 = Date.parse("2026-11-16T12:30:10Z");
const tmp = (tag: string): string => fs.mkdtempSync(path.join(os.tmpdir(), `vt-trackdata-${tag}-`));

type Raw = Record<string, unknown>;
/** one readsb row as adsb.lol sends it */
const readsb = (hex: string, o: Raw = {}): Raw => ({
  hex, flight: "TEST1  ", r: "N843S", t: "FA7X", lat: 36.1, lon: -83.9, alt_baro: 34000, alt_geom: 32075,
  gs: 459, track: 268.4, baro_rate: 1280, seen_pos: 0.4, seen: 0.1, ...o,
});
const mapped = (...rows: Raw[]): AircraftPoint[] => mapPointAircraft({ ac: rows }, "ac", "adsblol") as AircraftPoint[];

function readLines(base: string): Raw[] {
  const dir = path.join(base, "aircraft");
  if (!fs.existsSync(dir)) return [];
  const out: Raw[] = [];
  for (const f of fs.readdirSync(dir).sort()) {
    const buf = fs.readFileSync(path.join(dir, f));
    const text = f.endsWith(".gz") ? zlib.gunzipSync(buf).toString() : buf.toString();
    for (const l of text.split("\n")) if (l.trim()) out.push(JSON.parse(l) as Raw);
  }
  return out;
}
const writeHour = (base: string, name: string, rows: Raw[]): void => {
  const dir = path.join(base, "aircraft");
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, name), rows.map((r) => JSON.stringify(r)).join("\n") + "\n");
};
const byField = (row: TrackRow): Record<string, unknown> => Object.fromEntries(TRACK_FIELDS.map((f, i) => [f, row[i]]));

const fix = (t: number, o: Partial<TrackFix> = {}): TrackFix => ({
  t, t_src: "receive_minus_age", la: 40 + t / 1e6, lo: -75, al: null, alt_geom_m: null, alt_baro_m: null, g: false,
  gs: 200, trk: 90, vr: null, vr_kind: null, roll: null, c: null, rg: null, pv: "adsblol", pt: "adsb_icao", ...o,
});

// ── A. archive completeness ─────────────────────────────────────────────────
test("A1/A2 live writer: geometric altitude and the broadcast rates are ALREADY in the x block, so no flat copy is written", () => {
  const base = tmp("live");
  _resetIntentArchiveForTests();
  const n = archiveAircraft(mapped(readsb("a10001")), [], base, T0, 1);
  assert.equal(n, 1);
  const [line] = readLines(base);
  const x = line.x as Raw;
  assert.equal(x.k, 1, "first line of the aircraft in the hour file is a keyframe");
  assert.equal(x.ag, 32075, "geometric altitude, ft as broadcast");
  assert.equal(x.ab, 34000);
  assert.equal(x.br, 1280, "barometric rate, ft/min as broadcast");
  for (const k of ["ag", "vr", "vs", "ak", "ft"]) assert.ok(!(k in line), `flat ${k} would duplicate the x block`);
  // read back through the reader the track endpoint uses
  const f = fixFromLine(line, new IntentArchiveDecoder().feed(line));
  assert.ok(f);
  assert.equal(f.alt_baro_m, 10363, "34000 ft x 0.3048 = 10363.2 -> 10363 m (the line's al)");
  assert.equal(f.alt_geom_m, 9776, "32075 ft x 0.3048 = 9776.46 -> 9776 m");
  assert.equal(f.vr, 6.5, "1280 ft/min x 0.00508 = 6.5024 -> 6.5 m/s");
  assert.equal(f.vr_kind, "barometric");
  assert.equal(f.gs, 236, "459 kt x 0.5144 = 236.1 -> 236 m/s");
  assert.equal(f.trk, 268);
  assert.deepEqual(bestAltitude(f), { alt_m: 9776, alt_kind: "geometric" });
});

test("A1/A2 own-time writer (no x block): flat ag / vr / vs / ak / ft are written and read back", () => {
  const base = tmp("own");
  const [p] = mapped(readsb("a10002"));
  assert.equal(archiveAircraftAt([{ ...p, tSec: T0 / 1000 - 3 }], base), 1);
  const [line] = readLines(base);
  assert.ok(!("x" in line));
  assert.equal(line.t, T0 / 1000 - 3);
  assert.equal(line.ft, 1, "t is the fix's own time");
  assert.equal(line.ag, 9776);
  assert.equal(line.ak, "b", "al is the barometric altitude");
  assert.equal(line.al, 10363);
  assert.equal(line.vr, 6.5);
  assert.equal(line.vs, "b");
  const f = fixFromLine(line, null);
  assert.ok(f);
  assert.equal(f.t, T0 / 1000 - 3);
  assert.equal(f.t_src, "fix", "sp is NOT subtracted from an own-time line");
  assert.equal(f.alt_geom_m, 9776);
  assert.equal(f.alt_baro_m, 10363);
  assert.equal(f.vr, 6.5);
  assert.equal(f.vr_kind, "barometric");
});

test("A2 a fix with NO barometric altitude: al is the geometric fallback and says so; the rate is the geometric one", () => {
  const base = tmp("geom");
  const [p] = mapped(readsb("a10003", { alt_baro: undefined, baro_rate: undefined, geom_rate: -640 }));
  archiveAircraftAt([{ ...p, tSec: T0 / 1000 }], base);
  const [line] = readLines(base);
  assert.equal(line.al, 9776);
  assert.equal(line.ak, "g");
  assert.equal(line.vr, -3.3, "-640 ft/min x 0.00508 = -3.2512 -> -3.3 m/s");
  assert.equal(line.vs, "g");
  const f = fixFromLine(line, null);
  assert.ok(f);
  assert.equal(f.alt_baro_m, null, "never reported as barometric");
  assert.deepEqual(bestAltitude(f), { alt_m: 9776, alt_kind: "geometric" });
  assert.equal(f.vr_kind, "geometric");
});

test("A1 null is not zero: a broadcast 0 ft/min is written as 0; a row with no rate writes no rate", () => {
  const [level] = mapped(readsb("a10004", { baro_rate: 0 }));
  assert.equal(aircraftKinematicFields(level, false).vr, 0);
  const [silent] = mapped(readsb("a10005", { baro_rate: undefined }));
  assert.ok(!("vr" in aircraftKinematicFields(silent, false)));
  // a row with no parsed intent (an OpenSky state, a trace point): the rate is
  // kept, its source only when the caller says which
  const plain: AircraftPoint = { icao24: "a10006", lat: 1, lon: 2, altitude_m: 3000, on_ground: false, velocity_ms: 100, heading: 10, vertical_rate_ms: -2.54 };
  assert.deepEqual(aircraftKinematicFields(plain, false), { vr: -2.5 });
  assert.deepEqual(aircraftKinematicFields({ ...plain, vertical_rate_source: "geom", alt_geom_m: 3050.4 }, false), { ag: 3050, vr: -2.5, vs: "g" });
});

test("A old-schema lines still read: v1 (no sp), v3 (sp), and the existing reader still reads v5 lines", async () => {
  const v1 = { t: 1794787373, i: "a10007", la: 40, lo: -75, al: 3000, v: 120, h: 90 };
  const f1 = fixFromLine(v1, null);
  assert.ok(f1);
  assert.equal(f1.t, 1794787373);
  assert.equal(f1.t_src, "unadjusted", "no position age recorded: the stamp is kept and labelled");
  assert.deepEqual(bestAltitude(f1), { alt_m: 3000, alt_kind: "unrecorded" });
  assert.equal(f1.vr, null);
  const v3 = { ...v1, sp: 12.4, pv: "adsblol", pt: "adsb_icao" };
  const f3 = fixFromLine(v3, null);
  assert.ok(f3);
  assert.equal(f3.t, 1794787361, "1794787373 - 12.4 = 1794787360.6 -> 1794787361");
  assert.equal(f3.t_src, "receive_minus_age");
  assert.equal(fixFromLine({ t: "x", i: "a", la: 1, lo: 2 }, null), null);
  // the existing trips/track reader is unchanged by the new keys
  const base = tmp("compat");
  const [p] = mapped(readsb("a10008"));
  archiveAircraftAt([{ ...p, tSec: T0 / 1000 }], base);
  const got = await fullTrackAsync("aircraft", "a10008", base, {});
  assert.deepEqual(got, [{ t: T0 / 1000, la: 36.1, lo: -83.9, al: 10363, c: "TEST1", g: false }]);
});

// ── A3. selected-aircraft lane ──────────────────────────────────────────────
interface FastCounters {
  enabled: boolean; paused: boolean; min_free_bytes: number; max_lines_per_hour: number; rule: string;
  fixes_offered_total: number; lines_archived_total: number; skipped_same_fix_total: number;
  skipped_no_fix_time_total: number; skipped_stale_total: number; skipped_future_total: number; skipped_hour_cap_total: number;
  skipped_low_disk_total: number; skipped_disabled_total: number;
}
interface GlobalBody { coverage: { archive: { fixes_skipped_low_disk_total: number; lines_archived_total: number; fastlane: FastCounters } } }

async function withGlobal(
  fn: (ctx: { base: string; url: string; h: ReturnType<typeof registerGlobalAircraftRoutes>; clock: { t: number }; fast: () => Promise<FastCounters>; global: () => Promise<GlobalBody> }) => Promise<void>,
  opts: { free?: number | null; env?: Record<string, string> } = {},
): Promise<void> {
  const base = tmp("fast");
  const clock = { t: T0 };
  const app = express();
  const h = registerGlobalAircraftRoutes(app, {
    env: { GLOBAL_SWEEP_ENABLED: "0", ...(opts.env || {}) } as NodeJS.ProcessEnv,
    fetchImpl: (async () => { throw new Error("no network in tests"); }) as unknown as typeof fetch,
    sites: [], baseDir: base, now: () => clock.t,
    readFree: () => (opts.free === undefined ? 50 * GiB : opts.free),
  });
  const srv = app.listen(0);
  const url = `http://127.0.0.1:${(srv.address() as AddressInfo).port}`;
  const global = async () => (await (await fetch(`${url}/api/data/aircraft/global`)).json()) as GlobalBody;
  try {
    await fn({ base, url, h, clock, global, fast: async () => (await global()).coverage.archive.fastlane });
  } finally {
    h.stop();
    srv.close();
  }
}
const lane = (aircraft: AircraftPoint[], nowMs: number, upstreamNowMs: number | null = nowMs - 200) =>
  ({ provider: "adsblol", origin: "fastlane", aircraft, fetchedAt: nowMs, upstreamNowMs });

test("A3 a lane fix is archived ONCE at its own time; the same upstream fix polled again is never re-stamped", async () => {
  await withGlobal(async ({ base, h, clock, fast }) => {
    // poll 1 at T0: provider clock T0-200 ms, position 2.5 s old -> the fix is T0-2700 ms
    h.onBatch(lane(mapped(readsb("ab8c8e", { seen_pos: 2.5 })), clock.t));
    let lines = readLines(base);
    assert.equal(lines.length, 1);
    assert.equal(lines[0].t, T0 / 1000 - 3, "floor((T0 - 2700 ms) / 1000) = T0/1000 - 3: the fix's time, not ours (T0/1000)");
    assert.equal(lines[0].ft, 1);
    assert.equal(lines[0].pv, "adsblol");
    assert.equal(lines[0].vr, 6.5, "the broadcast vertical rate rides on the line");
    assert.equal(lines[0].ag, 9776);
    // poll 2, 2.5 s later: upstream still holds the SAME fix (now 5.0 s old)
    clock.t = T0 + 2500;
    h.onBatch(lane(mapped(readsb("ab8c8e", { seen_pos: 5.0 })), clock.t));
    // poll 3: same fix again, its age wobbling by 0.1 s
    clock.t = T0 + 5000;
    h.onBatch(lane(mapped(readsb("ab8c8e", { seen_pos: 7.6 })), clock.t));
    assert.equal(readLines(base).length, 1, "an unchanged upstream fix is not written again");
    // poll 4: a NEW fix (0.4 s old at provider clock T0+7300) -> T0+6900 ms
    clock.t = T0 + 7500;
    h.onBatch(lane(mapped(readsb("ab8c8e", { seen_pos: 0.4, lat: 36.11 })), clock.t));
    lines = readLines(base);
    assert.equal(lines.length, 2);
    assert.equal(lines[1].t, T0 / 1000 + 6, "floor((T0 + 7500 - 200 - 400) / 1000) = T0/1000 + 6");
    // an OLDER fix arriving late never goes in behind a newer one
    h.onBatch(lane(mapped(readsb("ab8c8e", { seen_pos: 9 })), clock.t));
    assert.equal(readLines(base).length, 2);
    const c = await fast();
    assert.equal(c.enabled, true);
    assert.equal(c.fixes_offered_total, 5);
    assert.equal(c.lines_archived_total, 2);
    assert.equal(c.skipped_same_fix_total, 3);
    assert.equal(c.min_free_bytes, MIN_FREE_BYTES, "the shared 1 GiB floor, not the sweep's 2 GiB gate");
    assert.match(c.rule, /own time/);
  });
});

test("A3 no position age -> not written (its own time is unknown); a fix over 10 min old or dated ahead of our clock -> not written; all counted", async () => {
  await withGlobal(async ({ base, h, clock, fast }) => {
    h.onBatch(lane(mapped(readsb("a20001", { seen_pos: undefined })), clock.t));
    h.onBatch(lane(mapped(readsb("a20002", { seen_pos: 601 })), clock.t));
    // a provider clock 3 h fast must not write into a future hour file
    h.onBatch(lane(mapped(readsb("a20003", { seen_pos: 1 })), clock.t, clock.t + 3 * 3_600_000));
    assert.equal(readLines(base).length, 0);
    const c = await fast();
    assert.equal(c.skipped_no_fix_time_total, 1);
    assert.equal(c.skipped_stale_total, 1);
    assert.equal(c.skipped_future_total, 1);
    assert.equal(c.lines_archived_total, 0);
  });
  // why 10 min: hour files are gzipped 2 h after their hour STARTS; an own-time
  // write into an already-gzipped hour would be compressed over it. A fix under
  // 10 min old lands in a file under 70 min old.
  assert.ok(FASTLANE_MAX_FIX_AGE_MS + 3_600_000 < 2 * 3_600_000);
});

// CHANGED ASSERTION 2026-11-17 (owner's decision: the bulk world log is off unless GLOBAL_SWEEP_ARCHIVE is set).
// before: the first half ran with GLOBAL_SWEEP_ARCHIVE unset (unset meant on) and expected the sweep fix skipped by its 2 GiB gate.
// after:  the same assertions, with GLOBAL_SWEEP_ARCHIVE=1 passed explicitly, so the sweep is still stopped by ITS gate (not by the off switch).
test("A3 disk floor: written between 1 and 2 GiB free (where the sweep is paused); skipped under 1 GiB and when unreadable", async () => {
  assert.equal(fastlaneArchiveAllowed(MIN_FREE_BYTES), true);
  assert.equal(fastlaneArchiveAllowed(MIN_FREE_BYTES - 1), false);
  assert.equal(fastlaneArchiveAllowed(null), false, "fails closed");
  assert.ok(MIN_FREE_BYTES < SWEEP_ARCHIVE_MIN_FREE_BYTES);
  // today's volume: 2.04 GB = 1.9 GiB free -> sweep paused, lane still recorded
  await withGlobal(async ({ base, h, clock, global }) => {
    h.onBatch({ provider: "adsblol", origin: "sweep-type", aircraft: mapped(readsb("a30001")), fetchedAt: clock.t });
    h.onBatch(lane(mapped(readsb("a30002")), clock.t));
    const lines = readLines(base);
    assert.deepEqual(lines.map((l) => l.i), ["a30002"]);
    const a = (await global()).coverage.archive;
    assert.equal(a.fixes_skipped_low_disk_total, 1, "the sweep fix was skipped by ITS gate");
    assert.equal(a.fastlane.lines_archived_total, 1);
    assert.equal(a.fastlane.paused, false);
  }, { free: Math.round(1.9 * GiB), env: { GLOBAL_SWEEP_ARCHIVE: "1" } });
  for (const free of [Math.round(0.9 * GiB), null]) {
    await withGlobal(async ({ base, h, clock, fast }) => {
      h.onBatch(lane(mapped(readsb("a30003")), clock.t));
      assert.equal(readLines(base).length, 0);
      const c = await fast();
      assert.equal(c.skipped_low_disk_total, 1);
      assert.equal(c.paused, true);
    }, { free });
  }
});

test("A3 bounded: AIRCRAFT_FASTLANE_ARCHIVE=0 writes nothing; the hourly cap holds and is counted; other origins are untouched", async () => {
  await withGlobal(async ({ base, h, clock, fast }) => {
    h.onBatch(lane(mapped(readsb("a40001")), clock.t));
    assert.equal(readLines(base).length, 0);
    const c = await fast();
    assert.equal(c.enabled, false);
    assert.equal(c.skipped_disabled_total, 1);
  }, { env: { AIRCRAFT_FASTLANE_ARCHIVE: "0" } });
  await withGlobal(async ({ base, h, clock, fast }) => {
    // 3,610 distinct fixes inside one clock hour: 10 aircraft, a new fix each
    // second for 361 s, every fix 2 s old when it arrives
    const start = Date.parse("2026-11-16T13:00:00Z");
    const ten = Array.from({ length: 10 }, (_, k) => `a4100${k}`);
    for (let r = 0; r <= FASTLANE_ARCHIVE_MAX_LINES_PER_HOUR / 10; r++) {
      clock.t = start + r * 1000 + 2000;
      h.onBatch(lane(mapped(...ten.map((hex) => readsb(hex, { seen_pos: 0 }))), clock.t, start + r * 1000));
    }
    const c = await fast();
    assert.equal(c.lines_archived_total, FASTLANE_ARCHIVE_MAX_LINES_PER_HOUR);
    assert.equal(c.skipped_hour_cap_total, 10);
    assert.equal(c.max_lines_per_hour, FASTLANE_ARCHIVE_MAX_LINES_PER_HOUR);
    assert.equal(readLines(base).length, FASTLANE_ARCHIVE_MAX_LINES_PER_HOUR);
    // viewport / tracked / scopes fixes are archived by their own paths, never here
    const before = readLines(base).length;
    for (const origin of ["viewport", "tracked", "scopes"]) {
      h.onBatch({ provider: "adsblol", origin, aircraft: mapped(readsb("a40003")), fetchedAt: clock.t });
    }
    assert.equal(readLines(base).length, before);
  });
});

// ── B. track assembly ───────────────────────────────────────────────────────
test("B vertical rate: broadcast is passed through; derived is (alt2 - alt1) / dt with its spacing; never across kinds, the ground or a long step", () => {
  const fixes: TrackFix[] = [
    fix(1000, { alt_baro_m: 3000, al: 3000 }),
    fix(1060, { alt_baro_m: 3120, al: 3120 }),                      // derived: (3120 - 3000) / 60 = 2.0
    fix(1090, { alt_baro_m: 3150, al: 3150, vr: 6.5, vr_kind: "barometric" }), // broadcast wins
    fix(1120, { alt_geom_m: 3300 }),                                  // geometric after barometric: not derived
    fix(1150, { alt_geom_m: 3255 }),                                  // derived geometric: (3255 - 3300) / 30 = -1.5
    fix(1150 + DERIVED_VR_MAX_DT_S + 1, { alt_geom_m: 3000 }),        // 121 s later: too far apart
    fix(1300, { al: 500 }),                                           // unlabelled altitude...
    fix(1330, { al: 440 }),                                           // ...derived, kind not recorded: (440 - 500) / 30 = -2.0
    fix(1360, { g: true }),                                           // on the ground: no rate
  ];
  const b = buildTrack(fixes, { max: 100 });
  const rows = b.rows.map(byField);
  assert.deepEqual(rows.map((r) => [r.vr_ms, r.vr_source, r.vr_kind, r.vr_dt_s]), [
    [null, null, null, null],
    [2, "derived", "barometric", 60],
    [6.5, "broadcast", "barometric", null],
    [null, null, null, null],
    [-1.5, "derived", "geometric", 30],
    [null, null, null, null],
    [null, null, null, null],
    [-2, "derived", null, 30],
    [null, null, null, null],
  ]);
  assert.deepEqual(b.counts.vr_broadcast, 1);
  assert.deepEqual(b.counts.vr_derived, 3);
  assert.deepEqual(b.counts.vr_none, 5);
  assert.deepEqual(rows.map((r) => r.alt_kind), ["barometric", "barometric", "barometric", "geometric", "geometric", "geometric", "unrecorded", "unrecorded", null]);
  assert.equal(rows[0].alt_m, 3000);
  assert.equal(rows[8].on_ground, true);
});

test("B gaps are listed and never bridged: no point is invented inside one, and no rate is derived across one", () => {
  const fixes = [
    fix(1000, { alt_baro_m: 1000 }), fix(1075, { alt_baro_m: 1100 }),
    fix(1075 + TRACK_GAP_S + 1, { alt_baro_m: 5000 }), // 301 s of silence
    fix(1451, { alt_baro_m: 5100 }),
    fix(1451 + TRACK_GAP_S, { alt_baro_m: 5100 }),     // exactly 300 s: not a gap
  ];
  const b = buildTrack(fixes, { max: 100 });
  assert.deepEqual(b.gaps, [{ from_t: 1075, to_t: 1376, seconds: 301 }]);
  assert.equal(b.rows.length, 5, "the fixes, and only the fixes");
  assert.equal(b.rows.filter((r) => r[0] > 1075 && r[0] < 1376).length, 0, "nothing inside the gap");
  const after = byField(b.rows[2]);
  assert.equal(after.vr_ms, null, "no rate across the gap ((5000-1000)/301 would be a fiction)");
});

test("B decimation is stated: every k-th fix plus both sides of every gap, first and last; rates are computed before thinning", () => {
  // 1,000 fixes 10 s apart climbing 20 m a fix (2.0 m/s), with one 400 s gap after fix 499
  const fixes: TrackFix[] = [];
  for (let k = 0; k < 1000; k++) fixes.push(fix(10_000 + k * 10 + (k >= 500 ? 400 : 0), { alt_baro_m: 1000 + k * 20 }));
  const b = buildTrack(fixes, { max: 100 });
  assert.equal(b.decimation.applied, true);
  assert.equal(b.decimation.points_total, 1000);
  // must-keep = {first, last, 499, 500} = 4 -> budget 96 -> stride ceil(1000 / 96) = 11
  assert.equal(b.decimation.stride, 11);
  assert.equal(b.decimation.gap_endpoints_kept, true);
  assert.ok(b.rows.length <= 100);
  assert.equal(b.decimation.points_returned, b.rows.length);
  assert.match(b.decimation.method, /every 11th fix/);
  const ts = b.rows.map((r) => r[0]);
  assert.equal(ts[0], 10_000);
  assert.equal(ts[ts.length - 1], 10_000 + 999 * 10 + 400);
  assert.deepEqual(b.gaps, [{ from_t: 14_990, to_t: 15_400, seconds: 410 }]);
  assert.ok(ts.includes(14_990) && ts.includes(15_400), "both ends of the listed gap are in the thinned rows");
  // a kept row's derived rate is the 10 s rate of the full series, not a rate across the thinned spacing
  const mid = byField(b.rows[5]);
  assert.equal(mid.vr_ms, 2, "20 m / 10 s = 2.0 m/s");
  assert.equal(mid.vr_dt_s, 10);
  // no thinning when it fits
  const all = buildTrack(fixes, { max: 1000 });
  assert.equal(all.decimation.applied, false);
  assert.equal(all.rows.length, 1000);
  // more gap ends than max: said so, not hidden
  const gappy = buildTrack(Array.from({ length: 50 }, (_, k) => fix(k * 1000)), { max: 10 });
  assert.equal(gappy.decimation.gap_endpoints_kept, false);
  assert.equal(gappy.gaps.length, 49);
  assert.ok(gappy.rows.length <= 10);
});

test("B one physical fix stamped twice collapses: same second keeps the better line; identical position within 2 s is one fix", () => {
  const own = fix(5000, { t_src: "fix", al: 3000, la: 40, lo: -75 });
  const restamped = fix(5001, { al: 3000, la: 40, lo: -75 });          // receive stamp minus age lands 1 s off
  const sameSecondNoAlt = fix(5000, { al: null, la: 40, lo: -75 });
  const next = fix(5005, { al: 3010, la: 40.01, lo: -75 });
  const c = cleanFixes([next, restamped, sameSecondNoAlt, own], (f) => (f.al != null ? 4 : 0) + (f.t_src === "fix" ? 2 : 0));
  assert.deepEqual(c.fixes.map((f) => [f.t, f.t_src, f.al]), [[5000, "fix", 3000], [5005, "receive_minus_age", 3010]]);
  assert.equal(c.duplicates_dropped, 2);
});

test("B input validation: hex, from/to, the 48 h window and max", () => {
  const now = 1_794_787_373;
  const ok = (q: Record<string, unknown>) => { const r = parseTrackQuery(q, now); assert.ok(r.ok, JSON.stringify(r)); return r.value; };
  const bad = (q: Record<string, unknown>) => { const r = parseTrackQuery(q, now); assert.ok(!r.ok, JSON.stringify(q)); return r.error; };
  assert.deepEqual(ok({}), { from: now - 86_400, to: now, max: 2000, defaulted: true });
  assert.deepEqual(ok({ from: String(now - 3600) }), { from: now - 3600, to: now, max: 2000, defaulted: false });
  assert.deepEqual(ok({ from: "1000", to: "5000", max: "50" }), { from: 1000, to: 5000, max: 50, defaulted: false });
  assert.match(bad({ to: "5000" }), /to needs from/);
  assert.match(bad({ from: "5000", to: "1000" }), /after from/);
  assert.match(bad({ from: "abc" }), /whole epoch seconds/);
  assert.match(bad({ from: "-5" }), /whole epoch seconds/);
  assert.match(bad({ from: "1e9" }), /whole epoch seconds/);
  assert.match(bad({ from: ["1", "2"] }), /whole epoch seconds/);
  assert.match(bad({ from: String(now - TRACK_MAX_WINDOW_S - 1) }), /limited to 48 hours/);
  assert.match(bad({ max: "1" }), /max must be/);
  assert.match(bad({ max: String(TRACK_MAX_POINTS + 1) }), /max must be/);
  assert.match(bad({ max: "2.5" }), /max must be/);
});

// ── C. fleet ────────────────────────────────────────────────────────────────
test("C tail -> hex: tracked registry, then a live row, then the FAA registry; a disagreement is reported; nothing is guessed", () => {
  const tracked: TrackedPlane[] = [{ reg: "N843S", added_at: 1, hex: "AB8C8E" }, { reg: "N1NEW", added_at: 2, hex: null }];
  const none = () => null;
  assert.deepEqual(resolveTail("N843S", { tracked, liveByReg: none, faaByReg: none }).hex, "ab8c8e");
  assert.equal(resolveTail("N843S", { tracked, liveByReg: none, faaByReg: none }).resolved_by, "tracked_registry");
  const liveOnly = resolveTail("N1NEW", { tracked, liveByReg: () => "a11111", faaByReg: none });
  assert.deepEqual([liveOnly.hex, liveOnly.resolved_by], ["a11111", "live_feed"]);
  const faa = resolveTail("N1A", { tracked, liveByReg: none, faaByReg: () => "a00002" });
  assert.deepEqual([faa.hex, faa.resolved_by], ["a00002", "faa_registry"]);
  assert.match(faa.basis, /FAA Releasable Aircraft Registry/);
  // two sources, two answers: the first by order is used and the other is shown
  const split = resolveTail("N843S", { tracked, liveByReg: () => "ab8c8e", faaByReg: () => "a99999" });
  assert.equal(split.hex, "ab8c8e");
  assert.deepEqual(split.conflicts, [{ hex: "a99999", by: "faa_registry" }]);
  // no source: null, never a made-up hex; a malformed hex from a source is ignored
  assert.deepEqual(resolveTail("N404", { tracked, liveByReg: none, faaByReg: none }), { hex: null, resolved_by: null, basis: "" });
  assert.equal(resolveTail("N404", { tracked, liveByReg: () => "zzzzzz", faaByReg: none }).hex, null);
  // the committed FAA spine, reversed (N1A is a00002 in datacore/aircraft/entity_spine.json)
  const spine = { a00002: { n_number: "N1A" }, a00019: { n_number: "N1AY" } };
  assert.equal(faaHexForReg("N1A", spine), "a00002");
  assert.equal(faaHexForReg("N404", spine), null);
  assert.equal(faaHexForReg("N1A", null), null);
});

test("C tail -> hex from the archive's rg field: newest hour files only, bounded, newest fix wins", async () => {
  const base = tmp("regscan");
  writeHour(base, "2026-11-14-05.jsonl", [{ t: 100, i: "a0old1", la: 1, lo: 2, rg: "N777OLD" }]);
  writeHour(base, "2026-11-15-05.jsonl", [{ t: 200, i: "a55555", la: 1, lo: 2, rg: "N555AB" }, { t: 150, i: "a5bad0", la: 1, lo: 2, rg: "N555AB" }]);
  writeHour(base, "2026-11-16-05.jsonl", [{ t: 300, i: "a66666", la: 1, lo: 2, rg: "n666cd" }, { t: 301, i: "a66666", la: 1, lo: 2 }]);
  const all = await scanArchiveForRegs(new Set(["N555AB", "N666CD", "N777OLD", "N404"]), base);
  assert.deepEqual(all.found.get("N555AB"), { hex: "a55555", t: 200 }, "two hexes carried it: the newest fix wins");
  assert.deepEqual(all.found.get("N666CD"), { hex: "a66666", t: 300 }, "registration compared in normalized upper case");
  assert.equal(all.found.has("N777OLD"), false, "its line's id (a0old1) is not a hex: rejected, never returned");
  assert.equal(all.found.has("N404"), false);
  assert.equal(all.files_scanned, 3);
  // bounded: with a 2-file budget the oldest hour is never opened
  writeHour(base, "2026-11-14-05.jsonl", [{ t: 100, i: "a77777", la: 1, lo: 2, rg: "N777OLD" }]);
  const two = await scanArchiveForRegs(new Set(["N777OLD"]), base, { maxFiles: 2 });
  assert.equal(two.found.size, 0);
  assert.equal(two.files_scanned, 2);
  assert.equal(two.oldest_hour_t, Date.parse("2026-11-15T05:00:00Z") / 1000, "how far back it looked is reported");
  const three = await scanArchiveForRegs(new Set(["N777OLD"]), base, { maxFiles: 3 });
  assert.equal(three.found.get("N777OLD")?.hex, "a77777");
  // stops as soon as every asked tail is found
  const early = await scanArchiveForRegs(new Set(["N666CD"]), base);
  assert.equal(early.files_scanned, 1);
});

test("C input validation and caps: tails / hexes strictly parsed, duplicates folded, at most 20 aircraft", () => {
  const ok = (q: Record<string, unknown>) => { const r = parseFleetQuery(q); assert.ok(r.ok, JSON.stringify(r)); return r.value; };
  const bad = (q: Record<string, unknown>) => { const r = parseFleetQuery(q); assert.ok(!r.ok, JSON.stringify(q)); return r.error; };
  assert.deepEqual(ok({}), { asks: [], explicit: false });
  assert.deepEqual(ok({ tails: "n843s, N123AB,N843S", hexes: "AB8C8E" }).asks,
    [{ tail: "N843S", hex: null }, { tail: "N123AB", hex: null }, { tail: null, hex: "ab8c8e" }]);
  assert.match(bad({ tails: "N843S,../etc/passwd" }), /not a tail number/);
  assert.match(bad({ tails: "N843S;DROP" }), /not a tail number/);
  assert.match(bad({ hexes: "xyz123" }), /not a tail number or a 6-digit hex/);
  assert.match(bad({ hexes: "ab8c8" }), /6-digit hex/);
  assert.match(bad({ tails: ["N1", "N2"] }), /ONE comma-separated value/);
  const many = Array.from({ length: FLEET_MAX_AIRCRAFT + 1 }, (_, k) => `N${100 + k}`).join(",");
  assert.match(bad({ tails: many }), /at most 20 aircraft/);
  assert.equal(ok({ tails: Array.from({ length: FLEET_MAX_AIRCRAFT }, (_, k) => `N${100 + k}`).join(",") }).asks.length, 20);
});

test("C the multi-aircraft scan is ONE pass and returns what the single-aircraft reader returns", async () => {
  const base = tmp("multi");
  const rows: Raw[] = [];
  for (let k = 0; k < 40; k++) {
    rows.push({ t: 1_794_700_000 + k * 60, i: k % 2 ? "a0000b" : "a0000a", la: 40 + k / 100, lo: -75, al: 1000 + k });
    rows.push({ t: 1_794_700_000 + k * 60, i: "a0000c", la: 10, lo: 10 });
  }
  writeHour(base, "2026-11-15-13.jsonl", rows.slice(0, 40));
  fs.writeFileSync(path.join(base, "aircraft", "2026-11-15-14.jsonl.gz"), zlib.gzipSync(rows.slice(40).map((r) => JSON.stringify(r)).join("\n") + "\n"));
  _resetBloomCache();
  const seen: Record<string, number[]> = { a0000a: [], a0000b: [] };
  const files = new Set<string>();
  const st = await scanArchiveRows("aircraft", ["a0000a", "a0000b"], base, {}, (r, f) => { seen[String(r.i)].push(r.t as number); files.add(f); });
  assert.equal(st.files_in_window, 2);
  assert.equal(st.files_read, 2);
  assert.equal(files.size, 2);
  for (const hex of ["a0000a", "a0000b"]) {
    const single = await fullTrackAsync("aircraft", hex, base, {});
    assert.deepEqual(seen[hex], single.map((p) => p.t), `${hex}: same fixes as the single-aircraft reader`);
    assert.equal(single.length, 20);
  }
  // second pass: the per-file id filter (built by the first) skips files none of the ids is in
  let calls = 0;
  const st2 = await scanArchiveRows("aircraft", ["affff1", "affff2"], base, {}, () => { calls++; });
  assert.equal(st2.files_skipped_by_filter, 2);
  assert.equal(st2.files_read, 0);
  assert.equal(calls, 0);
  // flights for several aircraft in one pass, each cleaned and split on its own
  const fleet = await readFleetHistories(["a0000a", "a0000b", "a0dead"], base);
  assert.equal(fleet.byHex.get("a0000a")?.archive.fixes, 20);
  assert.equal(fleet.byHex.get("a0000b")?.archive.first_fix_t, 1_794_700_060);
  assert.deepEqual(fleet.byHex.get("a0dead")?.archive, { fixes: 0, first_fix_t: null, last_fix_t: null, fixes_dropped_over_cap: 0, duplicates_dropped: 0, registrations_seen: [] });
  // the per-aircraft cap keeps the NEWEST fixes and counts what it dropped
  const capped = await readFleetHistories(["a0000a"], base, undefined, { perHex: 8 });
  const a = capped.byHex.get("a0000a")!.archive;
  assert.equal(a.last_fix_t, 1_794_700_000 + 38 * 60, "the newest fix survives");
  assert.ok(a.fixes <= 8 && a.fixes >= 6);
  assert.equal(a.fixes + a.fixes_dropped_over_cap, 20, "every dropped fix is counted");
});

test("C live on top: a fresh fix, a last known position, and 'no signal now' are three different answers", () => {
  const row: SnapRow = { hex: "ab8c8e", lon: -83.9, lat: 36.1, altFt: 34000, gsKt: 459, trk: 268.4, callsign: "N843S", type: "FA7X", cat: "A3", gnd: false, reg: "N843S", src: "adsblol", via: "tracked", seenAt: T0 - 5_000 };
  const fresh = fleetLive({ row, intent: null }, T0);
  assert.equal(fresh.live?.age_s, 5);
  assert.equal(fresh.live?.fresh, true);
  assert.equal(fresh.live?.alt_m, 10363);
  assert.equal(fresh.live?.gs_ms, 236);
  assert.equal(fresh.live?.seen_t, T0 / 1000 - 5);
  const old = fleetLive({ row: { ...row, seenAt: T0 - 300_000 }, intent: null }, T0);
  assert.equal(old.live?.fresh, false);
  assert.match(old.live_note, /last known position, 300 s old/);
  const gone = fleetLive({ row: { ...row, seenAt: T0 - (LIVE_MAX_AGE_S + 1) * 1000 }, intent: null }, T0);
  assert.equal(gone.live, null);
  assert.match(gone.live_note, /no signal now/);
  assert.match(fleetLive(null, T0).live_note, /no signal now/);
  // the live fix as a track point: time is the fix's, altitude is not called geometric without a broadcast geometric value
  const lf = liveFix({ row, intent: null }, T0);
  assert.equal(lf.t, T0 / 1000 - 5);
  assert.equal(lf.t_src, "fix");
  assert.deepEqual(bestAltitude(lf), { alt_m: 10363, alt_kind: "unrecorded" });
  assert.equal(lf.vr, null);
});

// ── end to end on a real Express app ────────────────────────────────────────
interface TrackBody {
  hex: string; kind: string; count: number; fields: string[]; units: Record<string, string>; rows: TrackRow[];
  range: { from: number; to: number; defaulted_to_last_24h: boolean };
  live: { appended: boolean; seen_t: number | null; age_s: number | null; reason: string };
  gaps: { threshold_s: number; count: number; listed: Array<{ from_t: number; to_t: number; seconds: number }>; rule: string };
  decimation: { applied: boolean; points_total: number; points_returned: number; stride: number | null };
  coverage: {
    first_fix_t: number | null; last_fix_t: number | null; points_in_window: number; vr_broadcast: number; vr_derived: number;
    alt_geometric: number; alt_unrecorded: number; note?: string;
    archive: { tier_read: string; local_oldest_hour_t: number | null; raw_retention_days: number; cold_tier_configured: boolean; note: string };
  };
  attitude: { pitch: string; bank: string; heading: string }; honesty: string; source: string; error?: string;
}
interface FleetBody {
  count: number; asked: string; error?: string; honesty: string;
  aircraft: Array<{
    tail: string | null; hex: string | null; resolved_by: string | null; resolution: { basis: string };
    tracked: { reg: string } | null; flight_count: number; flights_listed: number;
    flights: Array<{ start_t: number; end_t: number; fixes: number; max_alt_m: number | null; quality: string; from_airport: { id: string } | null; to_airport: { id: string } | null; is_flight: boolean }>;
    archive: { fixes: number; first_fix_t: number | null; last_fix_t: number | null; fixes_dropped_over_cap: number } | null;
    live: { age_s: number; fresh: boolean; lat: number } | null; live_note: string;
  }>;
  coverage: { archive: { local_oldest_hour_t: number | null; raw_retention_days: number; note: string }; registration_scan: { files_scanned: number } | null };
}

/** one flight in the archive: ground, climb, cruise, descent, ground — written
 *  by the REAL writer (receive-time lines with an x block). One hex per call:
 *  the writer's thinning clock is per aircraft and process-wide. */
function writeFlight(base: string, startMs: number, hex: string): { first: number; last: number } {
  _resetIntentArchiveForTests();
  const prof: Array<[number, number | "ground", number]> = [ // [minute, alt_baro ft, baro_rate ft/min]
    [0, "ground", 0], [2, 3000, 1500], [4, 6000, 1500], [6, 9000, 1500], [8, 9000, 0], [10, 9000, 0],
    [12, 6000, -1500], [14, 3000, -1500], [16, "ground", 0],
  ];
  let first = 0, last = 0;
  prof.forEach(([min, alt, rate], k) => {
    const nowMs = startMs + min * 60_000;
    const ground = alt === "ground";
    const row = readsb(hex, {
      lat: 36 + k * 0.05, lon: -84 + k * 0.05, alt_baro: alt, alt_geom: ground ? undefined : (alt as number) + 250,
      baro_rate: ground ? undefined : rate, gs: ground ? 10 : 300, seen_pos: 1.2,
    });
    assert.equal(archiveAircraft(mapped(row), [], base, nowMs, 1), 1);
    const t = Math.round(nowMs / 1000 - 1.2); // the fix is 1.2 s older than our stamp
    if (!first) first = t;
    last = t;
  });
  return { first, last };
}

async function withTrackApp(
  fn: (ctx: { url: string; base: string; clock: { t: number }; live: { state: Map<string, SnapRow> } }) => Promise<void>,
  o: { tracked?: TrackedPlane[]; faa?: Record<string, string> } = {},
): Promise<void> {
  const base = tmp("e2e");
  const clock = { t: T0 };
  const live = { state: new Map<string, SnapRow>() };
  const lookup: AircraftLiveLookup = {
    byHex: (hex) => { const row = live.state.get(hex); return row ? { row, intent: null } : null; },
    byReg: (reg) => Array.from(live.state.values()).find((r) => r.reg === reg) ?? null,
  };
  const app = express();
  registerAircraftTrackDataRoutes(app, {
    baseDir: base, now: () => clock.t, env: {} as NodeJS.ProcessEnv, live: () => lookup,
    tracked: () => o.tracked ?? [],
    nearest: (la) => (la < 36.01 ? { id: "KTYS", n: "McGhee Tyson", dist_km: 1.2, el: 300 } : la > 36.39 ? { id: "KTRI", n: "Tri-Cities", dist_km: 2, el: 460 } : null),
    faaByReg: (reg) => o.faa?.[reg] ?? null,
  });
  const srv = app.listen(0);
  try {
    await fn({ url: `http://127.0.0.1:${(srv.address() as AddressInfo).port}`, base, clock, live });
  } finally {
    srv.close();
  }
}

test("GET /api/data/aircraft/track/:hex — fix times, labelled altitudes, broadcast rates, live fix on top, honest coverage", async () => {
  await withTrackApp(async ({ url, base, clock, live }) => {
    const start = T0 - 40 * 60_000;
    const fl = writeFlight(base, start, "ab8c8e");
    const r = await fetch(`${url}/api/data/aircraft/track/AB8C8E`);
    assert.equal(r.status, 200);
    assert.match(r.headers.get("cache-control") || "", /max-age=10/);
    const d = (await r.json()) as TrackBody;
    assert.equal(d.hex, "ab8c8e");
    assert.equal(d.kind, "raw");
    assert.deepEqual(d.fields, [...TRACK_FIELDS]);
    for (const f of TRACK_FIELDS) assert.ok(TRACK_UNITS[f] && d.units[f], `units for ${f}`);
    assert.equal(d.range.defaulted_to_last_24h, true);
    assert.equal(d.count, 9);
    const rows = d.rows.map(byField);
    assert.equal(rows[0].t, fl.first, "stamp minus the 1.2 s position age");
    assert.equal(rows[0].t_src, "receive_minus_age");
    assert.equal(rows[0].on_ground, true);
    assert.equal(rows[0].alt_m, null);
    // minute 2: 3000 ft baro, 3250 ft geometric, +1500 ft/min
    assert.equal(rows[1].alt_baro_m, 914, "3000 ft = 914.4 m");
    assert.equal(rows[1].alt_geom_m, 991, "3250 ft = 990.6 m");
    assert.equal(rows[1].alt_m, 991);
    assert.equal(rows[1].alt_kind, "geometric");
    assert.equal(rows[1].vr_ms, 7.6, "1500 ft/min x 0.00508 = 7.62 -> 7.6");
    assert.equal(rows[1].vr_source, "broadcast");
    assert.equal(rows[1].vr_kind, "barometric");
    assert.equal(rows[1].gs_ms, 154, "300 kt x 0.5144 = 154.3");
    assert.equal(rows[1].track_deg, 268);
    assert.equal(rows[1].src, "adsblol");
    assert.equal(rows[1].origin, "archive");
    assert.equal(rows[4].vr_ms, 0, "level flight is a broadcast 0, not a missing value");
    assert.equal(rows[6].vr_ms, -7.6);
    assert.equal(d.coverage.first_fix_t, fl.first);
    assert.equal(d.coverage.last_fix_t, fl.last);
    assert.equal(d.coverage.vr_broadcast, 7);
    assert.equal(d.coverage.alt_geometric, 7);
    assert.equal(d.gaps.count, 0);
    assert.equal(d.gaps.threshold_s, TRACK_GAP_S);
    assert.equal(d.decimation.applied, false);
    assert.equal(d.live.appended, false);
    assert.match(d.live.reason, /no signal now/);
    // what the data is NOT
    assert.match(d.attitude.pitch, /NOT in the data/);
    assert.match(d.attitude.bank, /ONLY where roll_deg is not null/);
    assert.match(d.attitude.heading, /direction of travel over the ground/);
    assert.match(d.units.alt_geom_m, /WGS84 ellipsoid/);
    assert.match(d.units.alt_geom_m, /NOT above mean sea level/);
    assert.match(d.units.alt_baro_m, /not corrected/);
    assert.match(d.units.track_deg, /NOT the nose heading/);
    assert.ok(!/true altitude/i.test(JSON.stringify({ u: d.units, h: d.honesty })), "nothing is called true altitude");
    assert.match(d.honesty, /Nothing is interpolated/);
    // how far back: the oldest local hour file and the retention, stated
    assert.equal(d.coverage.archive.tier_read, "local");
    assert.equal(d.coverage.archive.local_oldest_hour_t, Math.floor(start / 3_600_000) * 3600);
    assert.equal(d.coverage.archive.raw_retention_days, 30);
    assert.equal(d.coverage.archive.cold_tier_configured, false);
    assert.match(d.coverage.archive.note, /local disk only/);

    // the live fix on top (a new fix, 4 s old), then thinning to max=5 keeps first + last
    clock.t = T0 + 60_000; // past the 30 s archive-part cache
    live.state.set("ab8c8e", { hex: "ab8c8e", lon: -83.5, lat: 36.5, altFt: null, gsKt: 8, trk: 90, callsign: "N843S", type: "FA7X", cat: "A3", gnd: true, reg: "N843S", src: "adsblol", via: "fastlane", seenAt: clock.t - 4_000 });
    const d2 = (await (await fetch(`${url}/api/data/aircraft/track/ab8c8e?max=5`)).json()) as TrackBody;
    assert.equal(d2.live.appended, true);
    assert.equal(d2.live.age_s, 4);
    assert.equal(d2.decimation.applied, true);
    assert.equal(d2.decimation.points_total, 10);
    assert.ok(d2.rows.length <= 5);
    const lastRow = byField(d2.rows[d2.rows.length - 1]);
    assert.equal(lastRow.origin, "live");
    assert.equal(lastRow.t, clock.t / 1000 - 4);
    assert.equal(lastRow.t_src, "fix");
    assert.equal(byField(d2.rows[0]).t, fl.first);
    // the wait since the last archived fix is a listed gap, not a drawn line
    assert.equal(d2.gaps.count, 1);
    assert.deepEqual(d2.gaps.listed[0], { from_t: fl.last, to_t: clock.t / 1000 - 4, seconds: clock.t / 1000 - 4 - fl.last });

    // an exact window: only the climb (minutes 2-6)
    const from = Math.floor(start / 1000) + 100, to = Math.floor(start / 1000) + 370;
    const d3 = (await (await fetch(`${url}/api/data/aircraft/track/ab8c8e?from=${from}&to=${to}`)).json()) as TrackBody;
    assert.equal(d3.count, 3);
    assert.deepEqual(d3.rows.map((row) => byField(row).alt_baro_m), [914, 1829, 2743]);
    assert.equal(d3.live.appended, false);
    assert.match(d3.live.reason, /outside the requested window/);

    // nothing held: an empty track that says so, never a 404 dressed as data
    const d4 = (await (await fetch(`${url}/api/data/aircraft/track/a00bad`)).json()) as TrackBody;
    assert.equal(d4.count, 0);
    assert.equal(d4.coverage.first_fix_t, null);
    assert.match(d4.coverage.note || "", /no fix for this aircraft in this window/);

    // strict input
    for (const bad of ["/track/xyz", "/track/ab8c8e?from=abc", "/track/ab8c8e?from=10&to=5", "/track/ab8c8e?max=999999", `/track/ab8c8e?from=${Math.floor(T0 / 1000) - TRACK_MAX_WINDOW_S - 3600}`]) {
      const br = await fetch(`${url}/api/data/aircraft${bad}`);
      assert.equal(br.status, 400, bad);
      assert.ok(((await br.json()) as TrackBody).error);
    }
  });
});

test("GET /api/data/aircraft/track/:hex — a derived rate on lines that carry none is flagged with its spacing (old-schema archive)", async () => {
  await withTrackApp(async ({ url, base }) => {
    const t = T0 / 1000 - 1800;
    writeHour(base, "2026-11-16-12.jsonl", [
      { t, i: "a0c0de", la: 40, lo: -75, al: 3000, v: 120, h: 90 },
      { t: t + 60, i: "a0c0de", la: 40.05, lo: -75, al: 3120, v: 120, h: 90 },
      { t: t + 60 + 400, i: "a0c0de", la: 40.4, lo: -75, al: 3900, v: 120, h: 91 },
    ]);
    const d = (await (await fetch(`${url}/api/data/aircraft/track/a0c0de`)).json()) as TrackBody;
    const rows = d.rows.map(byField);
    assert.deepEqual(rows.map((r) => [r.vr_ms, r.vr_source, r.vr_dt_s, r.alt_kind, r.t_src]), [
      [null, null, null, "unrecorded", "unadjusted"],
      [2, "derived", 60, "unrecorded", "unadjusted"], // (3120 - 3000) / 60 = 2.0 m/s
      [null, null, null, "unrecorded", "unadjusted"], // 400 s later: a gap, nothing derived
    ]);
    assert.equal(d.coverage.vr_derived, 1);
    assert.equal(d.coverage.alt_unrecorded, 3);
    assert.deepEqual(d.gaps.listed, [{ from_t: t + 60, to_t: t + 460, seconds: 400 }]);
  });
});

test("GET /api/data/aircraft/fleet — flights per tail from the archive, live on top, an unresolved tail is null with the reason", async () => {
  const tracked: TrackedPlane[] = [{ reg: "N843S", added_at: T0 - 86_400_000, hex: "ab8c8f", last_seen: T0 - 3_600_000 }];
  await withTrackApp(async ({ url, base, live }) => {
    const start = T0 - 5 * 3_600_000;
    const fl = writeFlight(base, start, "ab8c8f");
    // a second aircraft known only by the registration on its archived fixes
    writeHour(base, "2026-11-16-10.jsonl", [
      { t: T0 / 1000 - 9000, i: "a5a5a5", la: 30, lo: -90, al: 2000, rg: "N55RG" },
      { t: T0 / 1000 - 8940, i: "a5a5a5", la: 30.1, lo: -90, al: 2600, rg: "N55RG" },
      { t: T0 / 1000 - 8880, i: "a5a5a5", la: 30.2, lo: -90, al: 3200, rg: "N55RG" },
    ]);
    live.state.set("ab8c8f", { hex: "ab8c8f", lon: -83.5, lat: 36.5, altFt: 12000, gsKt: 250, trk: 45, callsign: "N843S", type: "FA7X", cat: "A3", gnd: false, reg: "N843S", src: "adsblol", via: "tracked", seenAt: T0 - 20_000 });
    const r = await fetch(`${url}/api/data/aircraft/fleet?tails=N843S,N55RG,N1A,N404XX&hexes=A0DEAD`);
    assert.equal(r.status, 200);
    assert.match(r.headers.get("cache-control") || "", /max-age=30/);
    const d = (await r.json()) as FleetBody;
    assert.equal(d.count, 5);
    const by = Object.fromEntries(d.aircraft.map((a) => [a.tail ?? a.hex, a]));
    // the tracked tail: its learned hex, one complete flight between two named airports, live on top
    const a = by.N843S;
    assert.equal(a.hex, "ab8c8f");
    assert.equal(a.resolved_by, "tracked_registry");
    assert.equal(a.tracked?.reg, "N843S");
    assert.equal(a.flight_count, 1);
    const f = a.flights[0];
    assert.equal(f.start_t, fl.first, "the first fix's own time (stamp minus 1.2 s)");
    assert.equal(f.end_t, fl.last);
    assert.equal(f.fixes, 9);
    assert.equal(f.max_alt_m, 2743, "9000 ft = 2743.2 m (barometric, the archive's al)");
    assert.equal(f.quality, "complete");
    assert.equal(f.is_flight, true);
    assert.equal(f.from_airport?.id, "KTYS");
    assert.equal(f.to_airport?.id, "KTRI");
    assert.equal(a.archive?.first_fix_t, fl.first, "the first fix we hold for this aircraft");
    assert.equal(a.archive?.fixes, 9);
    assert.equal(a.live?.age_s, 20);
    assert.equal(a.live?.fresh, true);
    // resolved from the archive's registration field
    assert.equal(by.N55RG.hex, "a5a5a5");
    assert.equal(by.N55RG.resolved_by, "archive_registration");
    assert.equal(by.N55RG.flight_count, 1);
    assert.equal(by.N55RG.live, null);
    assert.match(by.N55RG.live_note, /no signal now/);
    // resolved by the FAA registry pairing, but our archive holds nothing for it: zero flights, said plainly
    assert.equal(by.N1A.hex, "a00002");
    assert.equal(by.N1A.resolved_by, "faa_registry");
    assert.equal(by.N1A.flight_count, 0);
    assert.deepEqual(by.N1A.flights, []);
    assert.equal(by.N1A.archive?.first_fix_t, null);
    // never guessed
    assert.equal(by.N404XX.hex, null);
    assert.equal(by.N404XX.resolved_by, null);
    assert.match(by.N404XX.resolution.basis, /^unresolved: not in the tracked list/);
    assert.equal(by.N404XX.archive, null);
    assert.match(by.N404XX.live_note, /no hex resolved/);
    // a hex given directly
    assert.equal(by.a0dead.resolved_by, "given");
    assert.equal(by.a0dead.tail, null);
    // how far back, and that the reach is the local raw window only
    assert.equal(d.coverage.archive.raw_retention_days, 30);
    assert.equal(d.coverage.archive.local_oldest_hour_t, Math.floor(start / 3_600_000) * 3600);
    assert.ok(d.coverage.registration_scan && d.coverage.registration_scan.files_scanned >= 1);
    assert.match(d.honesty, /no FAA flight-plan or SWIM field/);
    // a flight's start/end feed the track endpoint and return that flight's fixes
    const tr = (await (await fetch(`${url}/api/data/aircraft/track/ab8c8f?from=${f.start_t}&to=${f.end_t}`)).json()) as TrackBody;
    assert.equal(tr.count, 9);

    // no tails given: the server's tracked list is the fleet
    const d2 = (await (await fetch(`${url}/api/data/aircraft/fleet`)).json()) as FleetBody;
    assert.equal(d2.count, 1);
    assert.equal(d2.aircraft[0].tail, "N843S");
    assert.match(d2.asked, /tracked list/);

    // strict input and the cap
    const many = Array.from({ length: FLEET_MAX_AIRCRAFT + 1 }, (_, k) => `N${100 + k}`).join(",");
    for (const bad of ["?tails=N843S,%3Cscript%3E", "?hexes=nothex", `?tails=${many}`, "?tails=N1&tails=N2"]) {
      const br = await fetch(`${url}/api/data/aircraft/fleet${bad}`);
      assert.equal(br.status, 400, bad);
      assert.ok(((await br.json()) as FleetBody).error);
    }
  }, { tracked, faa: { N1A: "a00002" } });
});

test("how far back: the oldest local hour file is reported; with the cold tier on, the note says older days are not read here", () => {
  const base = tmp("reach");
  assert.equal(archiveReach(base, {} as NodeJS.ProcessEnv).local_oldest_hour_t, null, "no archive yet: null, not a date");
  writeHour(base, "2026-10-25-07.jsonl", []);
  writeHour(base, "2026-11-16-12.jsonl", []);
  const local = archiveReach(base, {} as NodeJS.ProcessEnv);
  assert.equal(local.local_oldest_hour_t, Date.parse("2026-10-25T07:00:00Z") / 1000);
  assert.equal(local.raw_retention_days, 30);
  assert.equal(local.local_floor_days, 30);
  assert.ok(!/cold tier/.test(local.note));
  const cold = archiveReach(base, { R2_ACCOUNT_ID: "abcdef0123456789", R2_ACCESS_KEY_ID: "k", R2_SECRET_ACCESS_KEY: "s", R2_ARCHIVE_BUCKET: "vt-archive" } as unknown as NodeJS.ProcessEnv);
  assert.equal(cold.cold_tier_configured, true);
  assert.equal(cold.local_floor_days, 22);
  assert.match(cold.note, /cold tier \(R2\), which this endpoint does NOT read/);
});

test("the live lookup the endpoints read is the registered worldwide snapshot (and is cleared on stop)", async () => {
  await withGlobal(async ({ h, clock }) => {
    h.onBatch({ provider: "adsblol", origin: "tracked", aircraft: mapped(readsb("ab8c8e", { seen_pos: 1 })), fetchedAt: clock.t, upstreamNowMs: clock.t });
    const lk = aircraftLiveLookup();
    assert.ok(lk);
    const st = lk.byHex("ab8c8e", clock.t);
    assert.equal(st?.row.seenAt, clock.t - 1000);
    assert.equal(lk.byReg("N843S")?.hex, "ab8c8e");
    assert.equal(lk.byReg("N000ZZ"), null);
    // the intent store's fresh values ride on the live fix: geometric altitude and the broadcast rate
    const lf = liveFix(st!, clock.t);
    assert.equal(lf.alt_geom_m, 9776);
    assert.equal(lf.alt_baro_m, 10363);
    assert.equal(lf.vr, 6.5);
    assert.equal(lf.vr_kind, "barometric");
  });
  assert.equal(aircraftLiveLookup(), null);
});

test("readHexFixes: a receive-stamped line whose fix falls inside the window is found even when its stamp is past the window's end", async () => {
  const base = tmp("skew");
  writeHour(base, "2026-11-16-12.jsonl", [
    { t: 1_794_832_300, i: "a0aaaa", la: 1, lo: 2, al: 100, sp: 40 },   // fix at ...260: inside [..200, ..270]
    { t: 1_794_832_320, i: "a0aaaa", la: 1.1, lo: 2, al: 100, sp: 5 },  // fix at ...315: outside
  ]);
  const r = await readHexFixes(["a0aaaa"], base, { fromSec: 1_794_832_200, toSec: 1_794_832_270, decodeIntent: true, capPerHex: 100 });
  assert.deepEqual(r.byHex.get("a0aaaa")?.fixes.map((f) => f.t), [1_794_832_260]);
});

test("routes.ts wiring: exactly one registration line + import; the manifest names the v5 keys", () => {
  const src = fs.readFileSync(path.join(here, "routes.ts"), "utf8");
  assert.equal(src.split("registerAircraftTrackDataRoutes(").length - 1, 1);
  assert.ok(src.includes('import { registerAircraftTrackDataRoutes } from "./aircraftTrackData"'));
  const manifest = JSON.parse(fs.readFileSync(path.join(here, "..", "datacore", "manifests", "aircraft.json"), "utf8")) as { schema_version: number; field_map: Record<string, string> };
  assert.equal(manifest.schema_version, 5);
  for (const k of ["ag", "vr", "vs", "ak", "ft", "x"]) assert.ok(manifest.field_map[k], `field_map.${k}`);
});

// ── the tracked tails' permanent log, read back (owner's instruction 2026-11-17) ──
/** the flight of writeFlight, written by the REAL permanent-log writer at `startMs` */
function logFlight(base: string, startMs: number, hex: string): { first: number; last: number } {
  const clock = { t: startMs };
  const log = new TrackedPlaneLog({
    baseDir: base, now: () => clock.t, env: {} as NodeJS.ProcessEnv, readFree: () => 50 * GiB,
    tracked: () => [{ reg: "N843S", added_at: startMs - 1, hex }],
  });
  const prof: Array<[number, number | "ground", number]> = [
    [0, "ground", 0], [2, 3000, 1500], [4, 6000, 1500], [6, 9000, 1500], [8, 9000, 0], [10, 9000, 0],
    [12, 6000, -1500], [14, 3000, -1500], [16, "ground", 0],
  ];
  let first = 0, last = 0;
  prof.forEach(([min, alt, rate], k) => {
    clock.t = startMs + min * 60_000;
    const ground = alt === "ground";
    const row = readsb(hex, {
      lat: 36 + k * 0.05, lon: -84 + k * 0.05, alt_baro: alt, alt_geom: ground ? undefined : (alt as number) + 250,
      baro_rate: ground ? undefined : rate, gs: ground ? 10 : 300, seen_pos: 1.2,
    });
    assert.equal(log.ingest({ provider: "adsblol", origin: "tracked", aircraft: mapped(row), fetchedAt: clock.t, upstreamNowMs: clock.t }), 1);
    const t = Math.floor(clock.t / 1000 - 1.2);
    if (!first) first = t;
    last = t;
  });
  return { first, last };
}

interface TrackedLogBlock {
  held: boolean; months: number; bytes: number; first_fix_t: number | null; last_fix_t: number | null;
  read_for_this_window: boolean; why: string; fixes_in_window_from_log: number; note: string;
  months_read: { months_held: number; months_in_window: number; files_read: number } | null;
}
const logBlock = (d: TrackBody): TrackedLogBlock => (d.coverage as unknown as { tracked_log: TrackedLogBlock }).tracked_log;

test("GET /track/:hex reads a tracked tail's permanent log for a window older than the local hour files, and labels those points", async () => {
  await withTrackApp(async ({ url, base }) => {
    const oldStart = T0 - 40 * 86_400_000; // 2026-10-07: past the 22-day local window and the 30-day raw window
    const old = logFlight(base, oldStart, "ab8c90");
    const recent = writeFlight(base, T0 - 40 * 60_000, "ab8c90"); // today's flight, in the shared hour files only
    assert.ok(fs.existsSync(path.join(base, "aircraft_tracked", "AB8C90", "2026-10.jsonl")));

    // the old flight: served from the permanent log, every point labelled
    const d = (await (await fetch(`${url}/api/data/aircraft/track/ab8c90?from=${old.first - 60}&to=${old.last + 60}`)).json()) as TrackBody;
    assert.equal(d.count, 9);
    const rows = d.rows.map(byField);
    assert.ok(rows.every((r) => r.origin === "tracked_log"), "the source of each older fix is stated");
    assert.ok(rows.every((r) => r.t_src === "fix" && r.src === "adsblol"));
    assert.equal(rows[0].t, old.first);
    assert.equal(rows[8].t, old.last);
    assert.equal(rows[0].on_ground, true);
    assert.deepEqual(rows.slice(1, 4).map((r) => r.alt_baro_m), [914, 1829, 2743]);
    assert.deepEqual(rows.slice(1, 4).map((r) => r.alt_geom_m), [991, 1905, 2819], "geometric altitude from the flat keys");
    assert.equal(rows[1].vr_ms, 7.6);
    assert.equal(rows[1].vr_source, "broadcast");
    assert.match(d.units.origin, /tracked_log/);
    const b = logBlock(d);
    assert.equal(b.held, true);
    assert.equal(b.months, 1);
    assert.equal(b.bytes, fs.statSync(path.join(base, "aircraft_tracked", "AB8C90", "2026-10.jsonl")).size);
    assert.equal(b.first_fix_t, old.first);
    assert.equal(b.last_fix_t, old.last);
    assert.equal(b.read_for_this_window, true);
    assert.equal(b.fixes_in_window_from_log, 9);
    assert.deepEqual(b.months_read, { months_held: 1, months_in_window: 1, files_read: 1, files_failed: 0, lines_unparsed: 0 });
    assert.match(b.note, /capped in bytes \(oldest month deleted first\)/);
    assert.match(d.coverage.archive.note, /tracked list has a permanent log of its own/);

    // a window inside what the hour files hold: the log is not opened, nothing changes
    const d2 = (await (await fetch(`${url}/api/data/aircraft/track/ab8c90?from=${recent.first}&to=${recent.last}`)).json()) as TrackBody;
    assert.equal(d2.count, 9);
    assert.ok(d2.rows.map(byField).every((r) => r.origin === "archive"));
    assert.equal(logBlock(d2).read_for_this_window, false);
    assert.match(logBlock(d2).why, /starts inside what the local hour files hold/);
    assert.equal(logBlock(d2).held, true, "what the log holds is stated either way");
    assert.equal(logBlock(d2).fixes_in_window_from_log, 0);

    // a window that straddles the oldest hour file: the log gives ONLY what is older than it
    const oldestHour = d.coverage.archive.local_oldest_hour_t as number;
    const before = logFlight(base, oldestHour * 1000 - 30 * 60_000, "ab8c90"); // a flight that ended before the oldest hour file began
    logFlight(base, T0 - 40 * 60_000 + 500, "ab8c90"); // and the log holds today's flight too (as it would live), half a second later
    const d5 = (await (await fetch(`${url}/api/data/aircraft/track/ab8c90?from=${before.first}&to=${recent.last}`)).json()) as TrackBody;
    const r5 = d5.rows.map(byField);
    assert.deepEqual(r5.map((r) => r.origin), [...Array(9).fill("tracked_log"), ...Array(9).fill("archive")]);
    assert.deepEqual(r5.slice(9).map((r) => r.t), d2.rows.map((row) => byField(row).t), "inside the hour files' reach the answer is what it was without the log");
    assert.equal(logBlock(d5).fixes_in_window_from_log, 9);

    // an aircraft that was never tracked: no log, said so, and an old window stays empty
    const d3 = (await (await fetch(`${url}/api/data/aircraft/track/a00bad?from=${old.first - 60}&to=${old.last + 60}`)).json()) as TrackBody;
    assert.equal(d3.count, 0);
    assert.equal(logBlock(d3).held, false);
    assert.equal(logBlock(d3).read_for_this_window, false);
    assert.match(logBlock(d3).why, /no permanent log is held for this aircraft/);
  });
});

test("merge: where the hour files and the permanent log hold the same fix it is served once; the cap keeps the newest", async () => {
  const hour = [fix(100, { al: 1000 }), fix(160, { al: 1100 })];
  const logged: TrackFix[] = [
    { ...fix(40, { al: 900, t_src: "fix" }), store: "tracked_log" },   // only in the log
    { ...fix(100, { al: 1000, t_src: "fix" }), store: "tracked_log" }, // the same second as an hour-file line
    { ...fix(161, { al: 1100, t_src: "fix", la: hour[1].la }), store: "tracked_log" }, // the same fix stamped 1 s apart
  ];
  const m = mergeTrackedLogFixes(hour, logged, 100);
  assert.deepEqual(m.fixes.map((f) => f.t), [40, 100, 160]);
  assert.equal(m.duplicates_dropped, 2);
  assert.equal(m.fixes[0].store, "tracked_log");
  const built = buildTrack(m.fixes, { max: 100 });
  assert.equal(byField(built.rows[0]).origin, "tracked_log");
  assert.equal(byField(built.rows[2]).origin, "archive");
  const capped = mergeTrackedLogFixes(hour, logged, 2);
  assert.deepEqual(capped.fixes.map((f) => f.t), [100, 160], "the newest are kept");
  assert.equal(capped.dropped_over_cap, 1);
  // the log reader: window on the fix's own time, newest kept at the cap, every one labelled
  const base = tmp("logread");
  const fl = logFlight(base, T0 - 40 * 86_400_000, "ab8c8e");
  const all = await readTrackedLogFixes(base, "ab8c8e", { fromSec: fl.first, toSec: fl.last, cap: 100 });
  assert.equal(all.fixes.length, 9);
  assert.ok(all.fixes.every((f) => f.store === "tracked_log" && f.t_src === "fix"));
  const part = await readTrackedLogFixes(base, "ab8c8e", { fromSec: fl.first + 100, toSec: fl.first + 370, cap: 100 });
  assert.deepEqual(part.fixes.map((f) => f.alt_baro_m), [914, 1829, 2743]);
  const few = await readTrackedLogFixes(base, "ab8c8e", { fromSec: fl.first, toSec: fl.last, cap: 3 });
  assert.equal(few.fixes[few.fixes.length - 1].t, fl.last, "the newest fix survives the cap");
  assert.equal(few.fixes.length + few.dropped_over_cap, 9, "every dropped fix is counted");
});

test("GET /fleet lists a tracked tail's flights from before the local hour files, out of its permanent log", async () => {
  await withTrackApp(async ({ url, base }) => {
    const old = logFlight(base, T0 - 40 * 86_400_000, "ab8c91");
    const recent = writeFlight(base, T0 - 40 * 60_000, "ab8c91");
    // the log also holds today's flight (as it would live): only its part older than the hour files is read
    logFlight(base, T0 - 40 * 60_000, "ab8c91");
    const d = (await (await fetch(`${url}/api/data/aircraft/fleet`)).json()) as FleetBody;
    const a = d.aircraft[0];
    assert.equal(a.hex, "ab8c91");
    assert.equal(a.flight_count, 2, "the flight of 40 days ago and today's");
    assert.deepEqual(a.flights.map((f) => f.start_t).sort((x, y) => x - y), [old.first, recent.first]);
    assert.equal(a.archive?.first_fix_t, old.first);
    assert.equal(a.archive?.fixes, 18, "9 + 9: today's flight is not counted twice");
    assert.equal((a as unknown as { tracked_log_lines: number }).tracked_log_lines, 9);
    assert.match((d.coverage as unknown as { tracked_log: string }).tracked_log, /flights older than the local hour files come from its permanent log/);
  }, { tracked: [{ reg: "N843S", added_at: T0 - 50 * 86_400_000, hex: "ab8c91" }] });
  // with no hour files at all the whole log is read
  const base = tmp("fleetlog");
  const fl = logFlight(base, T0 - 3 * 86_400_000, "ab8c8e");
  const r = await readFleetHistories(["ab8c8e", "a0dead"], base);
  assert.equal(r.byHex.get("ab8c8e")?.archive.fixes, 9);
  assert.equal(r.byHex.get("ab8c8e")?.archive.first_fix_t, fl.first);
  assert.equal(r.byHex.get("ab8c8e")?.tracked_log_lines, 9);
  assert.equal(r.byHex.get("a0dead")?.tracked_log_lines, 0);
});
