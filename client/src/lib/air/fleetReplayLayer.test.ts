import { test } from "node:test";
import assert from "node:assert/strict";
import * as fleetLayer from "./fleetReplayLayer.ts";
import { verifyLayerContract, fitsBudget, budgetBytes } from "../../render/layerContract.ts";
import { FLEET_BUDGETS, FLEET_STRIDE } from "./fleetModel.ts";

test("Law IV contract: maxFeatures, vramBudget, a class with dispose()", () => {
  const LayerCls = fleetLayer.FleetReplayLayer;
  const inst = new LayerCls();
  assert.deepEqual(verifyLayerContract("fleetReplayLayer", {
    maxFeatures: fleetLayer.maxFeatures, vramBudget: fleetLayer.vramBudget, dispose: () => inst.dispose(),
  }), []);
  assert.equal(typeof LayerCls.prototype.dispose, "function");
  inst.dispose();
  inst.dispose(); // idempotent
});

test("the declared vramBudget covers the full-tier worst case and is not decoration", () => {
  const bytes = fleetLayer.fleetWorstCaseBytes(FLEET_BUDGETS.full);
  assert.ok(fitsBudget(bytes, fleetLayer.vramBudget), `${(bytes / 1048576).toFixed(1)} MB > ${fleetLayer.vramBudget} MB`);
  assert.ok(bytes * 4 > budgetBytes(fleetLayer.vramBudget), "budget within 4x of the real worst case");
  // lower tiers stay well inside it (mobile ~256 MB total budget)
  assert.ok(fleetLayer.fleetWorstCaseBytes(FLEET_BUDGETS.reduced) < bytes / 2);
  assert.ok(fleetLayer.fleetWorstCaseBytes(FLEET_BUDGETS.minimal) < 12 * 1048576);
  assert.equal(fleetLayer.maxFeatures, 2000, "matches the route's max=2000 bound");
});

test("shaders: the playhead cuts the future, the trail fades, the far side culls", () => {
  const vs = fleetLayer.FLEET_VERT_SRC("/*prelude*/", "#define GLOBE");
  assert.match(vs, /in float a_t;/);
  assert.match(vs, /in float a_slot;/);
  assert.match(vs, /0\.998001/, "globe far-side cull (occlusion radius²)");
  assert.match(fleetLayer.FLEET_FRAG_SRC, /if \(v_t > u_now \+ 0\.001\) discard;/, "future never drawn");
  assert.match(fleetLayer.FLEET_FRAG_SRC, /smoothstep\(u_trail \* 0\.6, u_trail, age\)/);
  const hs = fleetLayer.FLEET_HEAD_VERT_SRC("", "");
  assert.match(hs, /in vec4 a_inst;/, "heads are instanced");
  assert.equal(fleetLayer.headGlyph().length, 12, "chevron = 2 triangles");
});

test("setters are cheap CPU state until render; counts are exposed for the harness", () => {
  const l = new fleetLayer.FleetReplayLayer();
  l.setFull(new Float32Array(FLEET_STRIDE * 8));
  l.setThin(new Float32Array(FLEET_STRIDE * 4));
  l.setHeads(new Float32Array(8 * 3), 3);
  assert.deepEqual(l.getCounts(), { full: 8, thin: 4, conn: 0, heads: 3 });
  l.setFull(null);
  l.setHeads(null, 0);
  assert.deepEqual(l.getCounts(), { full: 0, thin: 4, conn: 0, heads: 0 });
  assert.equal(l.projectToScreen(0.5, 0.5, 0, 100, 100), null, "no frame drawn yet → no projection");
  l.dispose();
  assert.deepEqual(l.getCounts(), { full: 0, thin: 0, conn: 0, heads: 0 });
});

test("the highlight color is DESIGN.md --accent-red (#ff5a6e)", () => {
  const [r, g, b] = fleetLayer.FLEET_WARN_RGBA;
  assert.deepEqual([Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)], [0xff, 0x5a, 0x6e]);
});
