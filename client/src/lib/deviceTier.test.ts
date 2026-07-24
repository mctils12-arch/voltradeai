import { test } from "node:test";
import assert from "node:assert/strict";
import {
  classifyDevice, govInit, govStep, median,
  setOverloaded, isOverloaded, overloadFromState,
  GOV_OVERLOAD_MS, GOV_OVERLOAD_HOLD_MS, GOV_CALM_MS, GOV_CALM_HOLD_MS, GOV_COOLDOWN_MS, GOV_STRETCH_GRACE_MS,
} from "./deviceTier";

const RES_HOLD = GOV_OVERLOAD_HOLD_MS + GOV_STRETCH_GRACE_MS; // resolution waits out the tick-stretch grace

// ── classifyDevice ──────────────────────────────────────────────────────────

test("software renderers are minimal tier, ratio capped at 1", () => {
  for (const r of [
    "Google SwiftShader",
    "ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (Subzero)...)",
    "llvmpipe (LLVM 15.0.7, 256 bits)",
    "Microsoft Basic Render Driver",
  ]) {
    const t = classifyDevice({ renderer: r, devicePixelRatio: 2 });
    assert.equal(t.tier, "minimal", r);
    assert.equal(t.pixelRatioCap, 1, r);
    assert.ok(t.reasons.length >= 1);
  }
});

test("old integrated GPUs are reduced tier, capped at 1.5", () => {
  const t = classifyDevice({ renderer: "ANGLE (Intel, Intel(R) HD Graphics 520 Direct3D11)", devicePixelRatio: 2 });
  assert.equal(t.tier, "reduced");
  assert.equal(t.pixelRatioCap, 1.5);
});

test("low memory or very few cores reduce the tier", () => {
  const lowMem = classifyDevice({ renderer: "NVIDIA GeForce RTX 3060", deviceMemoryGB: 4, devicePixelRatio: 2 });
  assert.equal(lowMem.tier, "reduced");
  const lowCores = classifyDevice({ renderer: "NVIDIA GeForce RTX 3060", cores: 2, devicePixelRatio: 2 });
  assert.equal(lowCores.tier, "reduced");
});

test("capable hardware is full tier, capped at 2 even on 3x displays", () => {
  const t = classifyDevice({ renderer: "Apple M2", deviceMemoryGB: 16, cores: 8, devicePixelRatio: 3 });
  assert.equal(t.tier, "full");
  assert.equal(t.pixelRatioCap, 2);
  assert.equal(t.reasons.length, 0);
});

test("reduced cap never exceeds the actual device ratio", () => {
  const t = classifyDevice({ renderer: "Intel(R) UHD Graphics 400", devicePixelRatio: 1 });
  assert.equal(t.pixelRatioCap, 1);
});

test("modern UHD 620+ integrated graphics is NOT flagged weak", () => {
  const t = classifyDevice({ renderer: "ANGLE (Intel, Intel(R) UHD Graphics 620 Direct3D11)", devicePixelRatio: 2 });
  assert.equal(t.tier, "full");
});

// ── governor ────────────────────────────────────────────────────────────────

test("sustained overload steps the ratio down one ladder notch, with a note", () => {
  let st = govInit(2, 0);
  // overload begins at t=20s (past cooldown), must hold for HOLD_MS
  let d = govStep(st, 200, 20_000);
  assert.equal(d.apply, undefined); // just started
  d = govStep(d.state, 200, 20_000 + RES_HOLD);
  assert.equal(d.apply, 1.75);
  assert.match(d.note ?? "", /Render resolution lowered/);
  assert.match(d.note ?? "", /all layers and data stay on/);
});

test("a momentary spike does NOT step down (hold requirement)", () => {
  let st = govInit(2, 0);
  let d = govStep(st, 200, 20_000);
  d = govStep(d.state, 20, 21_000); // recovered before the hold elapsed
  d = govStep(d.state, 200, 22_000); // spike again — overload clock restarts
  d = govStep(d.state, 200, 22_000 + RES_HOLD - 1_000);
  assert.equal(d.apply, undefined);
});

test("steps are rate-limited by the cooldown", () => {
  let st = govInit(2, 0);
  let d = govStep(st, 200, 20_000);
  d = govStep(d.state, 200, 20_000 + RES_HOLD); // -> 1.75
  assert.equal(d.apply, 1.75);
  const stepAt = 20_000 + RES_HOLD;
  // still overloaded immediately after: no second step inside the cooldown
  d = govStep(d.state, 200, stepAt + 1_000);
  d = govStep(d.state, 200, stepAt + RES_HOLD);
  assert.equal(d.apply, undefined);
  // overload persisted through the cooldown: next notch lands once the
  // full resolution hold has renewed past the cooldown
  d = govStep(d.state, 200, stepAt + 1_000 + RES_HOLD);
  assert.equal(d.apply, 1.5);
});

test("ratio floors at 1 — no step below the ladder", () => {
  let st = govInit(1, 0);
  let d = govStep(st, 500, 20_000);
  d = govStep(d.state, 500, 20_000 + RES_HOLD);
  assert.equal(d.apply, undefined);
  assert.equal(d.state.ratio, 1);
});

test("sustained calm steps back up but never above the device ceiling", () => {
  // classified ceiling 1.5 (reduced tier); driven down to 1, then calm
  let st = govInit(1.5, 0);
  let d = govStep(st, 200, 20_000);
  d = govStep(d.state, 200, 20_000 + RES_HOLD); // -> 1.25
  assert.equal(d.apply, 1.25);
  const t0 = 20_000 + RES_HOLD + GOV_COOLDOWN_MS + 1;
  d = govStep(d.state, 10, t0);
  d = govStep(d.state, 10, t0 + GOV_CALM_HOLD_MS); // -> back up one notch
  assert.equal(d.apply, 1.5);
  assert.match(d.note ?? "", /restored/);
  // more calm: already at ceiling, no further step
  const t1 = t0 + GOV_CALM_HOLD_MS + GOV_COOLDOWN_MS + 1;
  d = govStep(d.state, 10, t1);
  d = govStep(d.state, 10, t1 + GOV_CALM_HOLD_MS);
  assert.equal(d.apply, undefined);
  assert.equal(d.state.ratio, 1.5);
});

test("between-band readings reset both trend clocks", () => {
  let st = govInit(2, 0);
  let d = govStep(st, 200, 20_000);
  d = govStep(d.state, 60, 21_000); // between calm and overload
  assert.equal(d.state.overloadSince, null);
  assert.equal(d.state.calmSince, null);
});

test("median helper", () => {
  assert.equal(median([]), 0);
  assert.equal(median([3]), 3);
  assert.equal(median([1, 9, 3]), 3);
});

// ── overload flag ───────────────────────────────────────────────────────────

test("overloadFromState: on after overload hold, off after 10s calm, sticky between", () => {
  let st = govInit(1, 0); // at the pixel floor — the flag is the only lever left
  let d = govStep(st, 200, 20_000);
  assert.equal(overloadFromState(d.state, 20_000, false), false); // just began
  d = govStep(d.state, 200, 20_000 + GOV_OVERLOAD_HOLD_MS);
  assert.equal(overloadFromState(d.state, 20_000 + GOV_OVERLOAD_HOLD_MS, false), true);
  // between bands: sticky (keeps previous)
  d = govStep(d.state, 60, 40_000);
  assert.equal(overloadFromState(d.state, 40_000, true), true);
  // calm begins: still sticky until 10s of calm
  d = govStep(d.state, 10, 50_000);
  assert.equal(overloadFromState(d.state, 55_000, true), true);
  d = govStep(d.state, 10, 60_001);
  assert.equal(overloadFromState(d.state, 60_001, true), false);
});

test("setOverloaded/isOverloaded module store round-trips", () => {
  assert.equal(isOverloaded(), false);
  setOverloaded(true);
  assert.equal(isOverloaded(), true);
  setOverloaded(false);
  assert.equal(isOverloaded(), false);
});

test("lever order: tick-stretch flag engages before any resolution step (round-16 quality contract)", () => {
  let st = govInit(2, 0);
  let d = govStep(st, 200, 20_000);
  const atFlag = 20_000 + GOV_OVERLOAD_HOLD_MS;
  d = govStep(d.state, 200, atFlag);
  assert.equal(overloadFromState(d.state, atFlag, false), true, "smoothness lever on at the hold");
  assert.equal(d.apply, undefined, "resolution untouched while the stretch grace runs");
  d = govStep(d.state, 200, atFlag + GOV_STRETCH_GRACE_MS - 1_000);
  assert.equal(d.apply, undefined, "still untouched just inside the grace");
  d = govStep(d.state, 200, atFlag + GOV_STRETCH_GRACE_MS);
  assert.equal(d.apply, 1.75, "pixels sacrificed only after the grace expires");
});
