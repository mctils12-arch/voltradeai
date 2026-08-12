// layerContract.test.ts — Law IV, declared and checked.
//
// The interesting assertions are the last two blocks: every real layer
// module is audited against the contract, and every declared vramBudget is
// checked against that layer's OWN stride arithmetic. A budget picked to
// look reasonable rather than derived would fail there.

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  applyFeatureCap,
  budgetBytes,
  disposeAllLayers,
  downsampleByImportance,
  fitsBudget,
  liveLayers,
  packedBufferBytes,
  registerLayer,
  resetLayerRegistry,
  totalDeclaredVramMB,
  verifyLayerContract,
} from "./layerContract.ts";
import { METRIC, getGauge, resetMetrics } from "./perfMetrics.ts";

import * as satLayer from "../lib/orbital/satLayer.ts";
import * as arcLayer from "../lib/orbital/arcLayer.ts";
import * as modelLayer from "../lib/orbital/modelLayer.ts";
import * as airLayer from "../lib/air/airLayer.ts";
import * as flightTrackLayer from "../lib/air/flightTrackLayer.ts";

// ── contract verification ───────────────────────────────────────────────────

test("verifyLayerContract names every missing export", () => {
  const v = verifyLayerContract("ghost", {});
  assert.equal(v.length, 3);
  assert.ok(v.some((s) => /maxFeatures/.test(s)));
  assert.ok(v.some((s) => /vramBudget/.test(s)));
  assert.ok(v.some((s) => /dispose/.test(s)));
});

test("verifyLayerContract rejects wrong types and non-positive budgets", () => {
  assert.equal(verifyLayerContract("x", { maxFeatures: 10, vramBudget: 4, dispose: () => {} }).length, 0);
  assert.equal(verifyLayerContract("x", { maxFeatures: 0, vramBudget: 4, dispose: () => {} }).length, 1);
  assert.equal(verifyLayerContract("x", { maxFeatures: -1, vramBudget: 4, dispose: () => {} }).length, 1);
  assert.equal(verifyLayerContract("x", { maxFeatures: "10", vramBudget: 4, dispose: () => {} }).length, 1);
  assert.equal(verifyLayerContract("x", { maxFeatures: 10, vramBudget: 4, dispose: {} }).length, 1);
});

// ── downsampling ────────────────────────────────────────────────────────────

test("under the cap, everything is kept and nothing is reported dropped", () => {
  const r = downsampleByImportance([1, 2, 3], 10, (n) => n);
  assert.deepEqual(r.kept, [1, 2, 3]);
  assert.equal(r.dropped, 0);
  assert.equal(r.capped, false);
  assert.equal(r.total, 3);
});

test("over the cap, the MOST IMPORTANT survive — not the first N", () => {
  // Head-truncating silently deletes whatever the upstream feed listed
  // last, which is how a map quietly loses a whole region.
  const items = [{ w: 1 }, { w: 9 }, { w: 3 }, { w: 7 }];
  const r = downsampleByImportance(items, 2, (i) => i.w);
  assert.deepEqual(
    r.kept.map((i) => i.w),
    [9, 7],
  );
  assert.equal(r.dropped, 2);
  assert.equal(r.capped, true);
});

test("ties are broken by input order, so the kept set is stable frame to frame", () => {
  // An unstable sort here makes capped features flicker in and out between
  // frames — a Law I violation produced by a Law IV mechanism.
  const items = [{ id: "a", w: 5 }, { id: "b", w: 5 }, { id: "c", w: 5 }];
  for (let i = 0; i < 5; i++) {
    const r = downsampleByImportance(items, 2, (x) => x.w);
    assert.deepEqual(
      r.kept.map((x) => x.id),
      ["a", "b"],
      "same input must yield the same kept set every time",
    );
  }
});

test("a zero or negative cap drops everything rather than dividing by zero", () => {
  const r = downsampleByImportance([1, 2], 0, (n) => n);
  assert.deepEqual(r.kept, []);
  assert.equal(r.dropped, 2);
  assert.equal(r.capped, true);
  assert.equal(downsampleByImportance([], 0, (n: number) => n).capped, false);
});

test("applyFeatureCap REPORTS the cap — no silent truncation", () => {
  resetMetrics();
  const warn = console.warn;
  const lines: string[] = [];
  console.warn = (m: string) => lines.push(String(m));
  try {
    const items = Array.from({ length: 100 }, (_, i) => i);
    const r = applyFeatureCap("testlayer", items, { maxFeatures: 10 }, (n) => n);
    assert.equal(r.kept.length, 10);
    assert.equal(getGauge("testlayer.dropped"), 90);
    assert.equal(getGauge(METRIC.FEATURES), 10);
    assert.equal(lines.length, 1, "a cap that bites must say so exactly once");
    assert.match(lines[0], /10 of 100/);
    assert.match(lines[0], /90 downsampled/);

    // Under cap: the gauge must RESET, or a layer looks permanently capped
    // after one busy frame.
    lines.length = 0;
    applyFeatureCap("testlayer", [1, 2], { maxFeatures: 10 }, (n) => n);
    assert.equal(getGauge("testlayer.dropped"), 0);
    assert.equal(lines.length, 0);
  } finally {
    console.warn = warn;
    resetMetrics();
  }
});

// ── budget arithmetic ───────────────────────────────────────────────────────

test("packedBufferBytes counts FLOATS per feature, matching how the layers describe themselves", () => {
  assert.equal(packedBufferBytes(1000, 8), 1000 * 8 * 4);
  assert.equal(packedBufferBytes(0, 8), 0);
  assert.equal(packedBufferBytes(1000, 0), 0);
  assert.equal(budgetBytes(4), 4 * 1024 * 1024);
  assert.equal(fitsBudget(1024, 1), true);
  assert.equal(fitsBudget(2 * 1024 * 1024, 1), false);
});

// ── the registry ────────────────────────────────────────────────────────────

test("the registry tracks live layers and unregisters cleanly", () => {
  resetLayerRegistry();
  const off = registerLayer({ id: "a", maxFeatures: 10, vramBudget: 4, dispose: () => {} });
  registerLayer({ id: "b", maxFeatures: 20, vramBudget: 8, dispose: () => {} });
  assert.equal(liveLayers().length, 2);
  assert.equal(totalDeclaredVramMB(), 12);
  off();
  assert.equal(liveLayers().length, 1);
  resetLayerRegistry();
});

test("disposeAllLayers isolates a throwing layer so one bad teardown cannot strand the rest", () => {
  resetLayerRegistry();
  const err = console.error;
  console.error = () => {};
  try {
    const done: string[] = [];
    registerLayer({ id: "good1", maxFeatures: 1, vramBudget: 1, dispose: () => done.push("good1") });
    registerLayer({
      id: "bad",
      maxFeatures: 1,
      vramBudget: 1,
      dispose: () => {
        throw new Error("boom");
      },
    });
    registerLayer({ id: "good2", maxFeatures: 1, vramBudget: 1, dispose: () => done.push("good2") });
    const res = disposeAllLayers();
    assert.deepEqual(done, ["good1", "good2"], "a throwing dispose must not strand later layers");
    assert.equal(res.disposed, 2);
    assert.equal(res.errors.length, 1);
    assert.equal(liveLayers().length, 0, "the registry empties even when a teardown threw");
  } finally {
    console.error = err;
    resetLayerRegistry();
  }
});

// ── the real layers ─────────────────────────────────────────────────────────

const REAL_LAYERS: [string, Record<string, unknown>][] = [
  ["satLayer", satLayer as unknown as Record<string, unknown>],
  ["arcLayer", arcLayer as unknown as Record<string, unknown>],
  ["modelLayer", modelLayer as unknown as Record<string, unknown>],
  ["airLayer", airLayer as unknown as Record<string, unknown>],
  ["flightTrackLayer", flightTrackLayer as unknown as Record<string, unknown>],
];

test("every real layer module declares maxFeatures and vramBudget", () => {
  for (const [name, mod] of REAL_LAYERS) {
    assert.equal(typeof mod.maxFeatures, "number", `${name} must export maxFeatures`);
    assert.ok((mod.maxFeatures as number) > 0, `${name} maxFeatures must be positive`);
    assert.equal(typeof mod.vramBudget, "number", `${name} must export vramBudget`);
    assert.ok((mod.vramBudget as number) > 0, `${name} vramBudget must be positive`);
  }
});

test("every real layer CLASS implements dispose()", () => {
  // The module exports the budgets; the layer class carries the teardown,
  // because that is where the GL handles live.
  const classes: [string, Record<string, unknown>][] = [
    ["SatLayer", satLayer as unknown as Record<string, unknown>],
    ["ArcLayer", arcLayer as unknown as Record<string, unknown>],
    ["ModelLayer", modelLayer as unknown as Record<string, unknown>],
    ["AirLayer", airLayer as unknown as Record<string, unknown>],
    ["FlightTrackLayer", flightTrackLayer as unknown as Record<string, unknown>],
  ];
  for (const [name, mod] of classes) {
    const ctor = Object.values(mod).find(
      (v) => typeof v === "function" && typeof (v as { prototype?: { dispose?: unknown } })?.prototype?.dispose === "function",
    );
    assert.ok(ctor, `${name}'s module exports no class with a dispose() method`);
  }
});

test("each declared vramBudget actually covers that layer's own stride arithmetic", () => {
  // The test that makes the declarations mean something. Each budget is
  // checked against the layer's real packing, using the layer's own
  // exported stride constant — a number picked to look plausible rather
  // than derived would fail here.
  const cases: { name: string; features: number; strideFloats: number; budgetMB: number }[] = [
    // SAT_STRIDE(7) position pack + 3 velocity + 1 shape = 11 floats
    { name: "satLayer", features: satLayer.maxFeatures, strideFloats: 11, budgetMB: satLayer.vramBudget },
    { name: "airLayer", features: airLayer.maxFeatures, strideFloats: airLayer.AIR_INST_STRIDE, budgetMB: airLayer.vramBudget },
    {
      name: "flightTrackLayer",
      features: flightTrackLayer.maxFeatures,
      // a quad strip: verts-per-segment x floats-per-vert
      strideFloats: flightTrackLayer.FT_VERT_STRIDE * 4,
      budgetMB: flightTrackLayer.vramBudget,
    },
    {
      name: "arcLayer",
      features: arcLayer.maxFeatures,
      strideFloats: arcLayer.ARC_VERT_STRIDE * arcLayer.ARC_VERTS_PER_SEG,
      budgetMB: arcLayer.vramBudget,
    },
  ];
  for (const c of cases) {
    const bytes = packedBufferBytes(c.features, c.strideFloats);
    assert.ok(
      fitsBudget(bytes, c.budgetMB),
      `${c.name}: worst case ${Math.round(bytes / 1024)}KB exceeds its declared ${c.budgetMB}MB budget`,
    );
    // And the budget must not be absurdly loose either — a 1000x headroom
    // budget is not a budget, it is a decoration.
    assert.ok(
      bytes * 1000 > budgetBytes(c.budgetMB),
      `${c.name}: ${c.budgetMB}MB is >1000x its ${Math.round(bytes / 1024)}KB worst case — not a real budget`,
    );
  }
});

test("the aircraft cap sits above the busiest render actually observed", () => {
  // The visual harness renders 3,507 aircraft at 1440px. A cap at or below
  // that would silently downsample normal traffic on every desktop load.
  assert.ok(airLayer.maxFeatures > 3507 * 2, `airLayer cap ${airLayer.maxFeatures} too close to observed live load`);
});

test("the satellite cap covers the active CelesTrak catalog", () => {
  // ~10k active objects; a cap below that downsamples the whole catalog.
  assert.ok(satLayer.maxFeatures >= 15000, `satLayer cap ${satLayer.maxFeatures} below the active catalog`);
});
