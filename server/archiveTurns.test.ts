// FLIGHT PROGRAM B1 — record-on-change refinement of the aircraft archive
// thinning: inside the cadence interval a fix is still KEPT when it turned
// > 5° or changed altitude > 500 ft (or flipped on-ground) since the last
// kept fix, so replays show real turns. Straight-and-level stays thinned.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  archiveAircraft, aircraftChangedSinceKept, recentTrack,
  TURN_KEEP_DEG, ALT_KEEP_M, CHANGE_MIN_GAP_MS,
  type AircraftPoint,
} from "./datacoreArchive";

const tmp = () => fs.mkdtempSync(path.join(os.tmpdir(), "vt-turns-"));
const fix = (icao: string, o: Partial<AircraftPoint> = {}): AircraftPoint => ({
  icao24: icao, callsign: "TRN1", lat: 45, lon: -30, altitude_m: 11000,
  on_ground: false, velocity_ms: 240, heading: 90, ...o,
});
// "now"-anchored (recentTrack reads a trailing window relative to its now)
const T0 = Math.floor(Date.now() / 3600_000) * 3600_000 + 60_000;

test("pure rule: >5° track (wrap-aware), >500 ft altitude, ground flip", () => {
  const prev = { h: 90, al: 11000, g: false };
  assert.equal(TURN_KEEP_DEG, 5);
  assert.equal(ALT_KEEP_M, 152.4);
  assert.equal(aircraftChangedSinceKept(undefined, fix("x")), false, "no prior -> cadence decides");
  assert.equal(aircraftChangedSinceKept(prev, fix("x", { heading: 94 })), false);
  assert.equal(aircraftChangedSinceKept(prev, fix("x", { heading: 96 })), true);
  assert.equal(aircraftChangedSinceKept({ ...prev, h: 358 }, fix("x", { heading: 2 })), false, "358->2 is 4°, not 356°");
  assert.equal(aircraftChangedSinceKept({ ...prev, h: 355 }, fix("x", { heading: 5 })), true, "355->5 is 10°");
  assert.equal(aircraftChangedSinceKept(prev, fix("x", { altitude_m: 11000 + 150 })), false);
  assert.equal(aircraftChangedSinceKept(prev, fix("x", { altitude_m: 11000 - 160 })), true);
  assert.equal(aircraftChangedSinceKept(prev, fix("x", { on_ground: true, altitude_m: null })), true);
  assert.equal(aircraftChangedSinceKept(prev, fix("x", { heading: null, altitude_m: null })), false, "missing inputs never force a write");
});

test("a turn inside the 75s cruise interval is KEPT; straight flight inside it is still thinned", () => {
  const base = tmp();
  assert.equal(archiveAircraft([fix("turn01", { heading: 90 })], [], base, T0), 1);
  assert.equal(archiveAircraft([fix("turn01", { heading: 91, lon: -29.9 })], [], base, T0 + 30_000), 0, "straight: thinned");
  assert.equal(archiveAircraft([fix("turn01", { heading: 120, lon: -29.8 })], [], base, T0 + 45_000), 1, "30° turn: kept");
  assert.equal(archiveAircraft([fix("turn01", { heading: 150, lon: -29.7 })], [], base, T0 + 47_000), 0,
    `still turning but inside the ${CHANGE_MIN_GAP_MS}ms floor — overlapping discs can't double-write`);
  assert.equal(archiveAircraft([fix("turn01", { heading: 150, lon: -29.7 })], [], base, T0 + 51_000), 1, "past the floor: kept");
  // the reference is the LAST KEPT fix: small steps that accumulate past 5° still trip it
  assert.equal(archiveAircraft([fix("turn01", { heading: 153 })], [], base, T0 + 60_000), 0);
  assert.equal(archiveAircraft([fix("turn01", { heading: 156 })], [], base, T0 + 70_000), 1, "3°+3° since the last kept fix = 6°");
  const tr = recentTrack("aircraft", "turn01", base, T0 + 71_000);
  assert.equal(tr.length, 4, "the replayable track carries the turn");
  const dir = path.join(base, "aircraft");
  const lines = fs.readdirSync(dir).flatMap((f) => fs.readFileSync(path.join(dir, f), "utf8").split("\n").filter(Boolean))
    .map((l) => JSON.parse(l)).filter((r) => r.i === "turn01");
  assert.deepEqual(lines.map((r) => r.h), [90, 120, 150, 156]);
});

test("a climb > 500 ft inside the interval is kept; a takeoff (ground flip) is kept", () => {
  const base = tmp();
  assert.equal(archiveAircraft([fix("climb1", { altitude_m: 3000 })], [], base, T0), 1);
  assert.equal(archiveAircraft([fix("climb1", { altitude_m: 3100 })], [], base, T0 + 20_000), 0, "100 m: thinned");
  assert.equal(archiveAircraft([fix("climb1", { altitude_m: 3200 })], [], base, T0 + 40_000), 1, "200 m since last kept: kept");
  assert.equal(archiveAircraft([fix("gnd001", { on_ground: true, altitude_m: null, heading: null })], [], base, T0), 1);
  assert.equal(archiveAircraft([fix("gnd001", { on_ground: true, altitude_m: null, heading: null })], [], base, T0 + 60_000), 0,
    "ground cadence is 5 min");
  assert.equal(archiveAircraft([fix("gnd001", { altitude_m: 150, heading: 270 })], [], base, T0 + 70_000), 1, "takeoff moment kept");
});
