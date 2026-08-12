import { test } from "node:test";
import assert from "node:assert/strict";
import {
  computeTzCrossings, offsetMinAt, parseGmtOffset, sideLabel, zoneOf,
  CONFIRM_SAMPLES,
} from "./tzCrossings";
import type { TrackSample } from "./trackModel";

const S = (t: number, lat: number, lon: number): TrackSample =>
  ({ t, lat, lon, altM: 10000, gap: false, held: false });

// 2026-08-12 ~15:00Z — summer, so US zones sit at their DST offsets
const T0 = Date.UTC(2026, 7, 12, 15, 0, 0) / 1000;

test("parseGmtOffset handles whole hours, half hours and GMT itself", () => {
  assert.equal(parseGmtOffset("GMT-4"), -240);
  assert.equal(parseGmtOffset("GMT+5:30"), 330);
  assert.equal(parseGmtOffset("GMT"), 0);
  assert.equal(parseGmtOffset("EDT"), null);
});

test("offsetMinAt is DST-aware: New York is −240 in August, −300 in January", () => {
  assert.equal(offsetMinAt("America/New_York", T0), -240);
  const jan = Date.UTC(2026, 0, 12, 15, 0, 0) / 1000;
  assert.equal(offsetMinAt("America/New_York", jan), -300);
});

test("zoneOf returns real zones over land and nautical zones at sea", () => {
  assert.equal(zoneOf(40.7, -74.0), "America/New_York");
  assert.equal(zoneOf(41.85, -87.65), "America/Chicago");
  const sea = zoneOf(30, -140); // mid-Pacific
  assert.ok(sea == null || /^Etc\//.test(sea), String(sea));
});

test("an NYC→Chicago track yields exactly one crossing, Eastern→Central", () => {
  // straight-line samples across Indiana into Illinois
  // (northwest Indiana stays Eastern almost to the Illinois line, so the
  // offset flips near the END of the leg — the arrival dwell below gives
  // the confirmation window room)
  const pts: TrackSample[] = [];
  for (let k = 0; k <= 20; k++) {
    const f = k / 20;
    pts.push(S(T0 + k * 300, 40.7 + (41.85 - 40.7) * f, -74.0 + (-87.65 + 74.0) * f));
  }
  pts.push(S(T0 + 21 * 300, 41.85, -87.65));
  pts.push(S(T0 + 22 * 300, 41.85, -87.65));
  const x = computeTzCrossings(pts);
  assert.equal(x.length, 1);
  assert.equal(x[0].fromOffsetMin, -240);
  assert.equal(x[0].toOffsetMin, -300);
  assert.match(x[0].label, /EDT → CDT/);
  assert.ok(x[0].t > T0 && x[0].idx > 0 && x[0].idx < pts.length - 1);
});

test("same-offset zone renames are NOT crossings (the clock never moved)", () => {
  // Phoenix (America/Phoenix, no DST) → Denver (America/Denver, MDT) in
  // WINTER: both UTC−7 → no crossing. In summer they differ → crossing.
  const jan = Date.UTC(2026, 0, 12, 18, 0, 0) / 1000;
  const winter: TrackSample[] = [];
  const summer: TrackSample[] = [];
  for (let k = 0; k <= 10; k++) {
    const f = k / 10;
    const lat = 33.45 + (39.74 - 33.45) * f;
    const lon = -112.07 + (-104.99 + 112.07) * f;
    winter.push(S(jan + k * 300, lat, lon));
    summer.push(S(T0 + k * 300, lat, lon));
  }
  assert.equal(computeTzCrossings(winter).length, 0);
  assert.equal(computeTzCrossings(summer).length, 1);
});

test("border jitter is confirmed away: a single flapped sample commits nothing", () => {
  const ny = { lat: 40.7, lon: -74.0 };
  const chi = { lat: 41.85, lon: -87.65 };
  const pts = [
    S(T0, ny.lat, ny.lon),
    S(T0 + 60, ny.lat, ny.lon),
    S(T0 + 120, chi.lat, chi.lon), // one-sample flap
    S(T0 + 180, ny.lat, ny.lon),
    S(T0 + 240, ny.lat, ny.lon),
  ];
  assert.ok(CONFIRM_SAMPLES >= 2);
  assert.equal(computeTzCrossings(pts).length, 0);
});

test("a crossing right at the track end still commits (ends inside the new zone)", () => {
  const pts = [
    S(T0, 40.7, -74.0),
    S(T0 + 60, 40.7, -74.0),
    S(T0 + 120, 41.85, -87.65), // last sample is the new zone
  ];
  const x = computeTzCrossings(pts);
  assert.equal(x.length, 1);
  assert.equal(x[0].idx, 2);
});

test("sideLabel prefers real abbreviations and falls back to UTC offsets", () => {
  assert.equal(sideLabel("America/New_York", T0, -240), "EDT");
  // Indian half-hour zone has no letter abbreviation in en-US Intl
  const ist = sideLabel("Asia/Kolkata", T0, 330);
  assert.ok(ist === "IST" || ist === "UTC+5:30", ist);
});
