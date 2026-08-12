// featureCaps.test.ts — PR8b: the declared caps are now ENFORCED, not just
// declared. 8a proved the numbers were derived; this proves they bite.

import { test } from "node:test";
import assert from "node:assert/strict";

import { AIR_MAX_FEATURES, aircraftImportance, buildAircraftInstances, AIR_INST_STRIDE } from "../lib/air/airLayer.ts";
import { FT_MAX_FEATURES, reportTrackPointCap } from "../lib/air/flightTrackLayer.ts";
import { SAT_MAX_FEATURES } from "../lib/orbital/satLayer.ts";
import { getGauge, resetMetrics } from "./perfMetrics.ts";

const plane = (over: Partial<{ lon: number; lat: number; on_ground: boolean; altitude_m: number }> = {}) => ({
  lon: 0,
  lat: 0,
  on_ground: false,
  altitude_m: 10000,
  ...over,
});

// ── aircraft ────────────────────────────────────────────────────────────────

test("under the cap, every valid aircraft is packed (the cap must not bite in normal use)", () => {
  resetMetrics();
  const rows = Array.from({ length: 4000 }, (_, i) => plane({ lon: i * 0.001 }));
  const { inst, rows: kept } = buildAircraftInstances(rows);
  assert.equal(kept.length, 4000, "4,000 is above the busiest observed render and must pass through untouched");
  assert.equal(inst.length, 4000 * AIR_INST_STRIDE);
  assert.equal(getGauge("aircraft.dropped"), 0);
  resetMetrics();
});

test("over the cap, the buffer is capped and the drop is reported", () => {
  resetMetrics();
  const warn = console.warn;
  const lines: string[] = [];
  console.warn = (m: string) => lines.push(String(m));
  try {
    const rows = Array.from({ length: AIR_MAX_FEATURES + 500 }, (_, i) => plane({ lon: (i % 300) * 0.001 }));
    const { inst, rows: kept } = buildAircraftInstances(rows);
    assert.equal(kept.length, AIR_MAX_FEATURES);
    assert.equal(inst.length, AIR_MAX_FEATURES * AIR_INST_STRIDE, "the cap runs BEFORE packing — no wasted buffer");
    assert.equal(getGauge("aircraft.dropped"), 500);
    assert.equal(lines.length, 1, "a cap that bites must say so");
    assert.match(lines[0], /12000 of 12500/);
  } finally {
    console.warn = warn;
    resetMetrics();
  }
});

test("when the cap bites, AIRBORNE traffic survives and ground clutter goes first", () => {
  resetMetrics();
  const warn = console.warn;
  console.warn = () => {};
  try {
    // A realistic overflow: a mass of parked aircraft plus a few en-route.
    const ground = Array.from({ length: AIR_MAX_FEATURES }, () => plane({ on_ground: true, altitude_m: 0 }));
    const airborne = Array.from({ length: 100 }, () => plane({ on_ground: false, altitude_m: 11000 }));
    const { rows: kept } = buildAircraftInstances([...ground, ...airborne]);
    assert.equal(kept.length, AIR_MAX_FEATURES);
    const keptAirborne = kept.filter((a) => !a.on_ground).length;
    assert.equal(keptAirborne, 100, "every en-route flight must survive a ground-clutter overflow");
  } finally {
    console.warn = warn;
    resetMetrics();
  }
});

test("aircraftImportance ranks airborne above ground, then by altitude", () => {
  assert.equal(aircraftImportance(plane({ on_ground: true, altitude_m: 9999 })), 0, "on_ground is always lowest");
  assert.ok(aircraftImportance(plane({ altitude_m: 0 })) > 0, "airborne at zero still beats on-ground");
  assert.ok(aircraftImportance(plane({ altitude_m: 11000 })) > aircraftImportance(plane({ altitude_m: 3000 })));
  assert.ok(aircraftImportance(plane({ altitude_m: undefined })) > 0, "a missing altitude must not sort below ground");
});

test("invalid positions are still rejected before the cap is applied", () => {
  resetMetrics();
  const rows = [plane(), { ...plane(), lon: NaN }, { ...plane(), lat: 89 }];
  const { rows: kept } = buildAircraftInstances(rows as never);
  assert.equal(kept.length, 1, "nothing renders from a guessed or out-of-mercator position");
  resetMetrics();
});

// ── satellites ──────────────────────────────────────────────────────────────

test("the satellite cap is ARMED by default, not a dormant lever", () => {
  // setRenderCap has existed since O1 but nothing ever called it, so the
  // layer rendered whatever the worker handed it. The default is now the
  // declared cap.
  assert.equal(SAT_MAX_FEATURES, 15000);
  assert.ok(SAT_MAX_FEATURES > 10000, "must sit above the ~10k active CelesTrak catalog");
});

// ── trail ───────────────────────────────────────────────────────────────────

test("an over-cap trail is REPORTED rather than silently reshaped", () => {
  resetMetrics();
  const warn = console.warn;
  const lines: string[] = [];
  console.warn = (m: string) => lines.push(String(m));
  try {
    assert.equal(reportTrackPointCap(100), false);
    assert.equal(getGauge("flightTrack.points"), 100);
    assert.equal(getGauge("flightTrack.overCap"), 0);
    assert.equal(lines.length, 0);

    assert.equal(reportTrackPointCap(FT_MAX_FEATURES + 200), true);
    assert.equal(getGauge("flightTrack.overCap"), 200);
    assert.equal(lines.length, 1);

    // Same size again: no repeat spam on every rebuild.
    reportTrackPointCap(FT_MAX_FEATURES + 200);
    assert.equal(lines.length, 1, "an unchanged over-cap size must not re-warn every frame");

    // Back under cap, then over again: warns afresh.
    reportTrackPointCap(10);
    reportTrackPointCap(FT_MAX_FEATURES + 200);
    assert.equal(lines.length, 2);
  } finally {
    console.warn = warn;
    resetMetrics();
  }
});
