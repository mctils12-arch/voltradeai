// gasFlareCandidates.ts unit tests — persistent-hotspot gas-flare candidate
// detector built over the NASA FIRMS archive (research/open_questions.md
// "GAS FLARE CANDIDATES"). Every case uses synthetic detections — no
// archive/network access, matching this module's own pure-function contract.
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  gridKey, findGasFlareCandidates, candidatesByRegion,
  type FlareCandidateInput, GRID_DEG,
} from "./gasFlareCandidates";

function det(
  lat: number, lon: number, acq_date: string,
  daynight: "D" | "N" | null, frp: number | null = 10,
): FlareCandidateInput {
  return { lat, lon, acq_date, daynight, frp };
}

test("gridKey: points within one grid cell collapse to the same key", () => {
  const a = gridKey(29.7604, -95.3698);
  const b = gridKey(29.7604 + GRID_DEG * 0.2, -95.3698 - GRID_DEG * 0.2);
  assert.equal(a, b);
});

test("gridKey: points a full cell apart get different keys", () => {
  const a = gridKey(29.7604, -95.3698);
  const b = gridKey(29.7604 + GRID_DEG * 3, -95.3698);
  assert.notEqual(a, b);
});

test("findGasFlareCandidates: a site lit 5/6 nights passes the default bar", () => {
  const dets: FlareCandidateInput[] = [];
  for (const day of ["2026-08-01", "2026-08-02", "2026-08-03", "2026-08-04", "2026-08-06"]) {
    dets.push(det(31.85, -102.35, day, "N"));
  }
  // one other detection anywhere establishes the 6th night in the window
  dets.push(det(40.0, -100.0, "2026-08-05", "N"));
  const out = findGasFlareCandidates(dets);
  const flare = out.find((c) => c.nightsActive === 5);
  assert.ok(flare, "expected the 5-night site to qualify");
  assert.equal(flare!.nightsInWindow, 6);
  assert.ok(Math.abs(flare!.persistence - 5 / 6) < 1e-9);
});

test("findGasFlareCandidates: a site lit only 1 night fails minNights even at 100% persistence", () => {
  const dets = [det(10, 10, "2026-08-01", "N")];
  const out = findGasFlareCandidates(dets);
  assert.equal(out.length, 0);
});

test("findGasFlareCandidates: daytime-only detections never qualify (nighttime-only by design)", () => {
  const dets: FlareCandidateInput[] = [];
  for (const day of ["2026-08-01", "2026-08-02", "2026-08-03", "2026-08-04"]) {
    dets.push(det(31.85, -102.35, day, "D"));
  }
  const out = findGasFlareCandidates(dets);
  assert.equal(out.length, 0);
});

test("findGasFlareCandidates: a moving/spreading wildfire (new cell every night) never accumulates persistence at one site", () => {
  const dets: FlareCandidateInput[] = [];
  const days = ["2026-08-01", "2026-08-02", "2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06"];
  days.forEach((day, i) => dets.push(det(31.85 + i * 0.2, -102.35 + i * 0.2, day, "N")));
  const out = findGasFlareCandidates(dets);
  assert.equal(out.length, 0, "each night lands in a distinct grid cell, so no site reaches minNights");
});

test("findGasFlareCandidates: below-bar persistence is excluded even with enough total nights active", () => {
  // active 3 of 10 nights in the window = 0.30, under the default 0.5 bar
  const dets: FlareCandidateInput[] = [];
  for (const day of ["2026-08-01", "2026-08-02", "2026-08-03"]) dets.push(det(1, 1, day, "N"));
  for (const day of ["2026-08-04", "2026-08-05", "2026-08-06", "2026-08-07", "2026-08-08", "2026-08-09", "2026-08-10"]) {
    dets.push(det(50, 50, day, "N")); // other sites establish the remaining window nights
  }
  const out = findGasFlareCandidates(dets);
  assert.ok(!out.some((c) => c.siteKey === gridKey(1, 1)));
});

test("findGasFlareCandidates: centroid is the mean of the site's own detections, not the last one seen", () => {
  const dets = [
    det(30.0, -100.0, "2026-08-01", "N"),
    det(30.001, -100.001, "2026-08-02", "N"),
    det(29.999, -99.999, "2026-08-03", "N"),
  ];
  const out = findGasFlareCandidates(dets, { minNights: 3, minPersistence: 0 });
  assert.equal(out.length, 1);
  assert.ok(Math.abs(out[0].lat - 30.0) < 1e-6);
  assert.ok(Math.abs(out[0].lon - (-100.0)) < 1e-6);
});

test("findGasFlareCandidates: meanFrp ignores null-FRP detections rather than treating them as zero", () => {
  const dets = [
    det(5, 5, "2026-08-01", "N", 20),
    det(5, 5, "2026-08-02", "N", null),
    det(5, 5, "2026-08-03", "N", 40),
  ];
  const out = findGasFlareCandidates(dets, { minNights: 3, minPersistence: 0 });
  assert.equal(out.length, 1);
  assert.equal(out[0].meanFrp, 30); // (20+40)/2, not (20+0+40)/3
});

test("findGasFlareCandidates: firstSeen/lastSeen bracket the site's own detection dates", () => {
  const dets = [
    det(2, 2, "2026-08-05", "N"),
    det(2, 2, "2026-08-01", "N"),
    det(2, 2, "2026-08-03", "N"),
  ];
  const out = findGasFlareCandidates(dets, { minNights: 3, minPersistence: 0 });
  assert.equal(out[0].firstSeen, "2026-08-01");
  assert.equal(out[0].lastSeen, "2026-08-05");
});

test("findGasFlareCandidates: sorted by persistence descending, tie-broken by mean FRP descending", () => {
  const dets: FlareCandidateInput[] = [];
  // site A: 3/3 nights, low FRP
  for (const day of ["2026-08-01", "2026-08-02", "2026-08-03"]) dets.push(det(1, 1, day, "N", 5));
  // site B: 3/3 nights, high FRP
  for (const day of ["2026-08-01", "2026-08-02", "2026-08-03"]) dets.push(det(2, 2, day, "N", 500));
  const out = findGasFlareCandidates(dets, { minNights: 3, minPersistence: 0 });
  assert.equal(out.length, 2);
  assert.equal(out[0].siteKey, gridKey(2, 2));
  assert.equal(out[1].siteKey, gridKey(1, 1));
});

test("findGasFlareCandidates: empty input returns an empty list, not a throw", () => {
  assert.deepEqual(findGasFlareCandidates([]), []);
});

test("findGasFlareCandidates: custom minNights/minPersistence thresholds are honored", () => {
  const dets: FlareCandidateInput[] = [];
  for (const day of ["2026-08-01", "2026-08-02"]) dets.push(det(3, 3, day, "N"));
  assert.equal(findGasFlareCandidates(dets, { minNights: 2, minPersistence: 0.5 }).length, 1);
  assert.equal(findGasFlareCandidates(dets, { minNights: 5, minPersistence: 0.5 }).length, 0);
});

test("candidatesByRegion: sums candidate counts per region via the injected join, dropping unmatched ones", () => {
  const dets: FlareCandidateInput[] = [];
  for (const day of ["2026-08-01", "2026-08-02", "2026-08-03"]) {
    dets.push(det(31.85, -102.35, day, "N")); // Texas-ish
    dets.push(det(51.5, -0.12, day, "N"));    // London-ish
  }
  const candidates = findGasFlareCandidates(dets, { minNights: 3, minPersistence: 0 });
  assert.equal(candidates.length, 2);
  const regionOf = (lat: number, lon: number): string | null => {
    if (lat > 25 && lat < 35 && lon < -90) return "US";
    return null; // everything else unmapped in this synthetic stub
  };
  const byRegion = candidatesByRegion(candidates, regionOf);
  assert.deepEqual(byRegion, { US: 1 });
});
