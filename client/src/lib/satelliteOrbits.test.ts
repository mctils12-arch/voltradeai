// satelliteOrbits — ANALYST CONSOLE W2 client half. Pure-function battery
// (no DOM/React dependency, same testing posture as mapIcons.ts would get):
// TLE checksum algorithm, ommToTle field widths/round-trip against a REAL
// CelesTrak-published OMM+TLE pair (not a hand-built fixture — the strongest
// available ground truth), period math, and the display helpers.
//
// Run via `npm run test:node` (server/*.test.ts client/src/lib/*.test.ts —
// see package.json; there is no separate client test runner in this repo).
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  SAT_GROUPS, SAT_GROUP_LABEL, SATELLITE_GROUP_COLOR, SATELLITE_ATTRIBUTION,
  tleChecksum, ommToTle, periodMinutesFromMeanMotion, formatElementAge,
  celestrakSatcatUrl, type OmmLike,
} from "./satelliteOrbits";

// Live-fetched pair (2026-07-07) for the SAME element set: CelesTrak's own
// FORMAT=json and FORMAT=TLE outputs for NORAD 25544 at the identical EPOCH.
// This is real ground truth, not a fixture we invented.
const ISS_OMM: OmmLike = {
  OBJECT_NAME: "ISS (ZARYA)", OBJECT_ID: "1998-067A",
  EPOCH: "2026-07-07T12:12:01.986912",
  MEAN_MOTION: 15.48933372, ECCENTRICITY: 0.0006687, INCLINATION: 51.6304,
  RA_OF_ASC_NODE: 199.5144, ARG_OF_PERICENTER: 267.6545, MEAN_ANOMALY: 92.3678,
  EPHEMERIS_TYPE: 0, CLASSIFICATION_TYPE: "U", NORAD_CAT_ID: 25544,
  ELEMENT_SET_NO: 999, REV_AT_EPOCH: 57490, BSTAR: 0.00011369,
  MEAN_MOTION_DOT: 5.806e-5, MEAN_MOTION_DDOT: 0,
};
const ISS_TLE_LINE1 = "1 25544U 98067A   26188.50835633  .00005806  00000+0  11369-3 0  9999";
const ISS_TLE_LINE2 = "2 25544  51.6304 199.5144 0006687 267.6545  92.3678 15.48933372574901";

test("tleChecksum: matches the real published ISS TLE checksum digits", () => {
  assert.equal(tleChecksum(ISS_TLE_LINE1.slice(0, 68)), Number(ISS_TLE_LINE1[68]));
  assert.equal(tleChecksum(ISS_TLE_LINE2.slice(0, 68)), Number(ISS_TLE_LINE2[68]));
  // '-' counts as 1, letters/spaces/'.'/'+' count as 0 (spec rule, not just digits)
  assert.equal(tleChecksum("1--"), 3);
  assert.equal(tleChecksum("abc 123"), 6);
});

test("ommToTle: round-trips EXACTLY against a live CelesTrak OMM+TLE pair (ISS, 2026-07-07)", () => {
  const [line1, line2] = ommToTle(ISS_OMM);
  assert.equal(line1, ISS_TLE_LINE1, "line1 must match CelesTrak's own TLE output byte-for-byte");
  assert.equal(line2, ISS_TLE_LINE2, "line2 must match CelesTrak's own TLE output byte-for-byte");
  assert.equal(line1.length, 69, "TLE line1 is a fixed 69-column record");
  assert.equal(line2.length, 69, "TLE line2 is a fixed 69-column record");
});

test("ommToTle: every generated line carries a self-consistent checksum, real records or synthetic edges", () => {
  const cases: OmmLike[] = [
    ISS_OMM,
    // negative first-derivative and BSTAR, small NORAD id (padding), short
    // OBJECT_ID piece, near-zero angles — the edges the real ISS record
    // doesn't exercise.
    {
      OBJECT_NAME: "TEST SAT", OBJECT_ID: "2020-001B",
      EPOCH: "2026-01-01T00:00:00.000000",
      MEAN_MOTION: 1.00273791, ECCENTRICITY: 0.00021, INCLINATION: 0.05,
      RA_OF_ASC_NODE: 5.1234, ARG_OF_PERICENTER: 359.9999, MEAN_ANOMALY: 0.0001,
      EPHEMERIS_TYPE: 0, CLASSIFICATION_TYPE: "U", NORAD_CAT_ID: 123,
      ELEMENT_SET_NO: 5, REV_AT_EPOCH: 12, BSTAR: -0.0000123,
      MEAN_MOTION_DOT: -0.00000012, MEAN_MOTION_DDOT: 0,
    },
    // missing OBJECT_ID/classification/element-set/rev fields entirely —
    // these are never read by twoline2satrec, so ommToTle must still
    // produce a syntactically valid, checksummed line.
    {
      EPOCH: "2026-12-31T23:59:59.5",
      MEAN_MOTION: 14.2, ECCENTRICITY: 0.01, INCLINATION: 98.7,
      RA_OF_ASC_NODE: 300.1, ARG_OF_PERICENTER: 10.5, MEAN_ANOMALY: 200.2,
      NORAD_CAT_ID: 99999, BSTAR: 0, MEAN_MOTION_DOT: 0, MEAN_MOTION_DDOT: 0.000012,
    },
  ];
  for (const rec of cases) {
    const [line1, line2] = ommToTle(rec);
    assert.equal(line1.length, 69, `line1 length for NORAD ${rec.NORAD_CAT_ID}`);
    assert.equal(line2.length, 69, `line2 length for NORAD ${rec.NORAD_CAT_ID}`);
    assert.equal(tleChecksum(line1.slice(0, 68)), Number(line1[68]),
      `line1 checksum digit for NORAD ${rec.NORAD_CAT_ID}`);
    assert.equal(tleChecksum(line2.slice(0, 68)), Number(line2[68]),
      `line2 checksum digit for NORAD ${rec.NORAD_CAT_ID}`);
    assert.equal(line1[0], "1");
    assert.equal(line2[0], "2");
  }
});

test("periodMinutesFromMeanMotion: ISS mean motion -> ~93 minute period", () => {
  const period = periodMinutesFromMeanMotion(15.48933372);
  assert.ok(Math.abs(period - 92.977) < 0.01, `expected ~92.98 min, got ${period}`);
  // GPS operational satellites: ~2 revs/day -> ~12h period
  assert.ok(Math.abs(periodMinutesFromMeanMotion(2) - 720) < 0.001);
  assert.ok(Number.isNaN(periodMinutesFromMeanMotion(0)), "non-positive mean motion is not a period");
  assert.ok(Number.isNaN(periodMinutesFromMeanMotion(NaN)));
});

test("formatElementAge: honesty-chip thresholds match the file's other freshness formatting", () => {
  const now = Date.parse("2026-07-07T12:00:00Z");
  assert.equal(formatElementAge("2026-07-07T11:59:30Z", now), "30s ago");
  assert.equal(formatElementAge("2026-07-07T11:00:00Z", now), "60m ago");
  assert.equal(formatElementAge("2026-07-07T06:00:00Z", now), "6h ago");
  assert.equal(formatElementAge("2026-07-05T12:00:00Z", now), "2d ago");
  assert.equal(formatElementAge("not a date", now), "epoch unknown");
});

test("celestrakSatcatUrl: encodes the object name into the live public lookup page", () => {
  assert.equal(celestrakSatcatUrl("ISS (ZARYA)"),
    "https://celestrak.org/satcat/table-satcat.php?NAME=ISS%20(ZARYA)");
});

test("group registry stays in sync with server/satellites.ts GROUPS (hand-mirrored constant)", () => {
  assert.deepEqual([...SAT_GROUPS], ["stations", "starlink", "gps-ops", "geo"]);
  for (const g of SAT_GROUPS) {
    assert.ok(SAT_GROUP_LABEL[g], `missing label for ${g}`);
    assert.match(SATELLITE_GROUP_COLOR[g], /^#[0-9a-f]{6}$/i, `missing/invalid color for ${g}`);
  }
  assert.match(SATELLITE_ATTRIBUTION, /CelesTrak/);
});
