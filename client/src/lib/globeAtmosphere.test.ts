// Globe atmosphere presets — EARTH TWIN E2 quality pass. Pins: every preset
// key exists in the INSTALLED style spec's sky definition (imported v8
// reference, not memorized docs), the limb preset is the zoom-interpolated
// atmosphere-blend the v5.24 globe shader actually consumes, and the
// combined preset never lets one regime's fields clobber the other's.
import { test } from "node:test";
import assert from "node:assert/strict";
import { v8 } from "@maplibre/maplibre-gl-style-spec";
import type { SkySpecification } from "maplibre-gl";
import {
  GLOBE_LIMB_SKY,
  TERRAIN_HORIZON_SKY,
  DEFAULT_SKY,
  SOFTWARE_GL_SKY,
  LIMB_VISIBLE_ZOOM_MAX,
  isSoftwareRenderer,
  skyForRenderer,
} from "./globeAtmosphere.js";

const SPEC_SKY_KEYS = new Set(Object.keys((v8 as unknown as { sky: object }).sky));

// Compile-time: presets must satisfy the installed SkySpecification type.
const _typecheck: SkySpecification[] = [GLOBE_LIMB_SKY, TERRAIN_HORIZON_SKY, DEFAULT_SKY];
void _typecheck;

test("every preset key exists in the installed maplibre v8 sky spec — no guessed fields", () => {
  for (const [name, preset] of Object.entries({ GLOBE_LIMB_SKY, TERRAIN_HORIZON_SKY, DEFAULT_SKY })) {
    for (const k of Object.keys(preset)) {
      assert.ok(SPEC_SKY_KEYS.has(k), `${name}: "${k}" must exist in v8.sky of the installed style spec`);
    }
  }
});

test("limb preset: zoom-interpolated atmosphere-blend (the only field the globe limb shader reads)", () => {
  // Verified against dist/maplibre-gl.js: the limb pass uniforms are sun pos,
  // globe pos/radius and u_atmosphere_blend only — colors are hardcoded
  // Rayleigh coefficients, so the limb preset must not pretend otherwise.
  assert.deepEqual(Object.keys(GLOBE_LIMB_SKY), ["atmosphere-blend"],
    "sky/horizon/fog colors do not affect the globe limb in v5.24 — keep the preset honest");
  const expr = GLOBE_LIMB_SKY["atmosphere-blend"] as unknown[];
  assert.ok(Array.isArray(expr), "spec: 'It is best to interpolate this expression when using globe projection'");
  assert.deepEqual(expr.slice(0, 3), ["interpolate", ["linear"], ["zoom"]]);
  const stops = expr.slice(3);
  assert.ok(stops.length >= 4 && stops.length % 2 === 0, "stop/value pairs");
  for (let i = 0; i < stops.length; i += 2) {
    const z = stops[i] as number;
    const blend = stops[i + 1] as number;
    assert.ok(blend >= 0 && blend <= 1, "atmosphere-blend is 0..1");
    if (i > 0) assert.ok(z > (stops[i - 2] as number), "zoom stops strictly ascend");
  }
  assert.equal(stops[1], 1, "full physical limb glow on the hero globe");
  assert.equal(stops[stops.length - 2], LIMB_VISIBLE_ZOOM_MAX, "documented visibility band matches the ramp");
  assert.equal(stops[stops.length - 1], 0, "fully off past the globe→mercator morph (pass is skipped at 0)");
});

test("terrain-horizon preset: colors + blends only, all within their spec ranges", () => {
  const p = TERRAIN_HORIZON_SKY as Record<string, unknown>;
  for (const key of ["sky-color", "horizon-color", "fog-color"]) {
    assert.ok(/^#[0-9a-f]{6}$/i.test(String(p[key])), `${key}: solid hex color`);
  }
  for (const key of ["sky-horizon-blend", "horizon-fog-blend", "fog-ground-blend"]) {
    const v = p[key] as number;
    assert.ok(typeof v === "number" && v >= 0 && v <= 1, `${key}: 0..1 per the spec`);
  }
  assert.ok(!("atmosphere-blend" in p), "limb intensity belongs to the limb preset");
});

test("combined DEFAULT_SKY: exact union, no field lost or overwritten (disjoint shader passes)", () => {
  const union = { ...GLOBE_LIMB_SKY, ...TERRAIN_HORIZON_SKY };
  assert.deepEqual(DEFAULT_SKY, union);
  const limbKeys = Object.keys(GLOBE_LIMB_SKY);
  const horizonKeys = Object.keys(TERRAIN_HORIZON_SKY);
  for (const k of limbKeys) assert.ok(!horizonKeys.includes(k),
    "presets must stay disjoint — one setSky call may never silently drop a regime's field");
  assert.equal(Object.keys(DEFAULT_SKY).length, limbKeys.length + horizonKeys.length);
});

test("software-GL tier: atmosphere pass EXPLICITLY off (spec default is 0.8 — omission is not off)", () => {
  assert.equal(SOFTWARE_GL_SKY["atmosphere-blend"], 0,
    "the ~156ms/frame scattering pass must be pinned off, not left to the 0.8 default");
  // still a full sky otherwise: the cheap horizon/fog gradient stays
  for (const k of Object.keys(TERRAIN_HORIZON_SKY)) {
    assert.deepEqual((SOFTWARE_GL_SKY as Record<string, unknown>)[k],
      (TERRAIN_HORIZON_SKY as Record<string, unknown>)[k], `${k} preserved for software GL`);
  }
  for (const k of Object.keys(SOFTWARE_GL_SKY)) {
    assert.ok(SPEC_SKY_KEYS.has(k), `SOFTWARE_GL_SKY: "${k}" exists in the installed spec`);
  }
});

test("renderer detection: known software rasterizers only — hardware and unknown keep the full sky", () => {
  for (const soft of [
    "ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (Subzero) (0x0000C0DE)), SwiftShader driver)",
    "llvmpipe (LLVM 15.0.7, 256 bits)",
    "softpipe",
    "Microsoft Basic Render Driver",
  ]) {
    assert.equal(isSoftwareRenderer(soft), true, soft);
    assert.deepEqual(skyForRenderer(soft), SOFTWARE_GL_SKY);
  }
  for (const hard of [
    "ANGLE (NVIDIA, NVIDIA GeForce RTX 4070 (0x00002786) Direct3D11 vs_5_0 ps_5_0, D3D11)",
    "Apple M2",
    "Mali-G78",
    "AMD Radeon Pro 5500M OpenGL Engine",
    "", null, undefined,
  ]) {
    assert.equal(isSoftwareRenderer(hard as any), false, String(hard));
    assert.deepEqual(skyForRenderer(hard as any), DEFAULT_SKY);
  }
});
