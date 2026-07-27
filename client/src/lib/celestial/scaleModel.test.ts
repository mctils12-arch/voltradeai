// scaleModel — the B2 scale-system contract, pinned. THE core rail: the
// layout may compress; the numbers never lie — compression is pure render
// layout, inputs are never mutated, c=0/s=1 is the exact identity (the TRUE
// SCALE preset must render bit-identically to the B1 frame).
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  AU_M,
  compressDistance,
  compressExponent,
  applyDistanceCompression,
  renderedDiscPx,
  clampScaleState,
  sizeSliderToMult,
  multToSizeSlider,
  isTrueScale,
  readScalePref,
  SCALE_PRESET_TRUE,
  SCALE_PRESET_VISIBLE,
  SIZE_MULT_MIN,
  SIZE_MULT_MAX,
  SUN_SIZE_MULT_CAP,
  SIZE_APPARENT_CAP_PX,
  type ScaleBodyIn,
} from "./scaleModel.js";
import { solarSystemState, BODY_RADIUS_M, AU_M as AU_M_EPH } from "./solarSystem.js";

const close = (a: number, b: number, tol: number, msg: string): void =>
  assert.ok(Math.abs(a - b) <= tol, `${msg}: ${a} vs ${b} (tol ${tol})`);

test("AU constant matches the ephemeris' (duplicated to keep the panel bundle lean)", () => {
  assert.equal(AU_M, AU_M_EPH);
});

// ── distance compression ────────────────────────────────────────────────────

test("c=0 is the EXACT identity — the TRUE preset carries zero rounding", () => {
  for (const r of [1, 384_400_000, 0.5 * AU_M, AU_M, 5.2 * AU_M, 30.1 * AU_M, 7e12]) {
    assert.equal(compressDistance(r, 0), r, `identity at r=${r}`);
  }
  assert.equal(compressDistance(0, 0), 0);
  assert.equal(compressDistance(0, 1), 0);
});

test("monotone + order-preserving at every c (the ordering guarantee)", () => {
  const rs = [1e6, 384_400_000, 0.31 * AU_M, 0.387 * AU_M, 0.72 * AU_M, AU_M,
    1.52 * AU_M, 5.2 * AU_M, 9.5 * AU_M, 19.2 * AU_M, 30.1 * AU_M, 7e12];
  for (const c of [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]) {
    for (let i = 1; i < rs.length; i++) {
      assert.ok(
        compressDistance(rs[i - 1], c) < compressDistance(rs[i], c),
        `strictly increasing at c=${c}: r=${rs[i - 1]}→${rs[i]}`,
      );
    }
  }
});

test("relative-spacing sense: equal distance ratios map to equal compressed ratios", () => {
  // ln r' = p·ln r + const ⇒ f(2r)/f(r) is the same 2^p everywhere
  for (const c of [0.3, 0.7, 1]) {
    const p = compressExponent(c);
    for (const r of [0.2 * AU_M, AU_M, 4 * AU_M, 20 * AU_M]) {
      close(
        compressDistance(2 * r, c) / compressDistance(r, c),
        Math.pow(2, p),
        1e-12 * Math.pow(2, p),
        `ratio preserved at c=${c}, r=${r / AU_M} AU`,
      );
    }
  }
});

test("continuity in c: small slider steps move the layout smoothly", () => {
  const r = 30.1 * AU_M;
  let prev = compressDistance(r, 0);
  for (let c = 0.01; c <= 1.0001; c += 0.01) {
    const cur = compressDistance(r, c);
    assert.ok(cur < prev, "compression strictly deepens with c");
    assert.ok(prev / cur < 1.09, `no jump: step ratio ${(prev / cur).toFixed(4)} at c=${c.toFixed(2)}`);
    prev = cur;
  }
});

test("c=1 fits Mercury→Neptune in one comfortable view (the VISIBLE promise)", () => {
  const state = solarSystemState(Date.UTC(2026, 6, 18));
  const planets = state.filter((b) => !["sun", "earth", "moon"].includes(b.id));
  const compressed = planets.map((b) => ({ id: b.id, au: compressDistance(b.rSunAu * AU_M, 1) / AU_M }));
  for (const p of compressed) {
    assert.ok(p.au > 0.5 && p.au < 5.5, `${p.id} compressed to ${p.au.toFixed(2)} AU (in [0.5, 5.5])`);
  }
  const rs = compressed.map((p) => p.au);
  const span = Math.max(...rs) / Math.min(...rs);
  assert.ok(span < 8, `span ratio ${span.toFixed(1)} < 8 (true is ~78)`);
  // whole system frames from ~15 AU: atan(4.6/15) ≈ 17° < the 18.4° half-fov
  assert.ok(Math.max(...rs) < 15 * Math.tan((18.4 * Math.PI) / 180), "fits the fov from 15 AU");
  // true ordering that day survives compression
  const trueOrder = [...planets].sort((a, b) => a.rSunAu - b.rSunAu).map((b) => b.id);
  const compOrder = [...compressed].sort((a, b) => a.au - b.au).map((p) => p.id);
  assert.deepEqual(compOrder, trueOrder, "radial ordering preserved");
});

test("applyDistanceCompression: hierarchy, in-place out, inputs never mutated", () => {
  const t = Date.UTC(2026, 6, 18);
  const state = solarSystemState(t);
  const byId = Object.fromEntries(state.map((b) => [b.id, b]));
  const bodies: ScaleBodyIn[] = state.map((b) => ({
    id: b.id,
    parentId: b.id === "moon" ? "earth" : null,
    pos: { x: b.helioAu.x * AU_M, y: b.helioAu.y * AU_M, z: b.helioAu.z * AU_M },
  }));
  // structuredClone, not JSON: Earth's ecliptic z is a literal -0 and JSON
  // round-trips -0 to +0, which strict deepEqual (correctly) distinguishes
  const snapshot = structuredClone(bodies.map((b) => b.pos));
  // preallocated out record is reused in place (no per-frame allocation)
  const out: Record<string, { x: number; y: number; z: number }> = {};
  for (const b of bodies) out[b.id] = { x: 0, y: 0, z: 0 };
  const refs = Object.values(out);
  const res = applyDistanceCompression(bodies, 1, out);
  assert.equal(res, out, "returns the provided out record");
  Object.values(out).forEach((v, i) => assert.equal(v, refs[i], "vectors mutated in place, not replaced"));
  // inputs untouched — the layout may compress; the SOURCE never does
  assert.deepEqual(bodies.map((b) => b.pos), snapshot, "true positions never mutated");
  // sun stays at the origin
  close(Math.hypot(out.sun.x, out.sun.y, out.sun.z), 0, 1e-9, "sun at origin");
  // planets: compressed radius matches the radial map, direction preserved
  for (const id of ["mercury", "venus", "mars", "jupiter", "neptune"]) {
    const trueR = byId[id].rSunAu * AU_M;
    const r = Math.hypot(out[id].x, out[id].y, out[id].z);
    close(r, compressDistance(trueR, 1), 1, `${id} radial map`);
    const dot = out[id].x * byId[id].helioAu.x + out[id].y * byId[id].helioAu.y + out[id].z * byId[id].helioAu.z;
    close(dot / (r * byId[id].rSunAu), 1, 1e-12, `${id} direction preserved`);
  }
  // THE MOON RIDES EARTH AT THE TRUE OFFSET (the B1 lunar payoff survives
  // every c — Earth still subtends its real ~1.9° from the Moon)
  const moonOff = {
    x: out.moon.x - out.earth.x,
    y: out.moon.y - out.earth.y,
    z: out.moon.z - out.earth.z,
  };
  const trueOff = {
    x: (byId.moon.helioAu.x - byId.earth.helioAu.x) * AU_M,
    y: (byId.moon.helioAu.y - byId.earth.helioAu.y) * AU_M,
    z: (byId.moon.helioAu.z - byId.earth.helioAu.z) * AU_M,
  };
  close(moonOff.x, trueOff.x, 1e-3, "moon offset x true");
  close(moonOff.y, trueOff.y, 1e-3, "moon offset y true");
  close(moonOff.z, trueOff.z, 1e-3, "moon offset z true");
  // c=0 through the full pipeline: exact identity
  const outT = applyDistanceCompression(bodies, 0);
  for (const b of bodies) {
    assert.equal(outT[b.id].x, b.pos.x, `${b.id} identity x at c=0`);
    assert.equal(outT[b.id].y, b.pos.y, `${b.id} identity y at c=0`);
    assert.equal(outT[b.id].z, b.pos.z, `${b.id} identity z at c=0`);
  }
});

test("orphan parentId falls back to the heliocentric rule (nothing is dropped)", () => {
  const bodies: ScaleBodyIn[] = [
    { id: "x", parentId: "missing", pos: { x: 2 * AU_M, y: 0, z: 0 } },
  ];
  const out = applyDistanceCompression(bodies, 1);
  close(Math.hypot(out.x.x, out.x.y, out.x.z), compressDistance(2 * AU_M, 1), 1, "orphan compressed radially");
});

// ── body size ───────────────────────────────────────────────────────────────

test("renderedDiscPx: s=1 identity, monotone, cap, close-range truth", () => {
  for (const d of [0.1, 1, 5, 39.9, 40, 100, 612]) {
    for (const rel of [0.27, 1, 11.21]) {
      assert.equal(renderedDiscPx(d, 1, rel, false, false), d, `identity at s=1, disc=${d}, rel=${rel}`);
    }
  }
  // the reference response curve (2026-07-18 reconciliation): the effective
  // multiplier is m^0.78 · rel^(−0.22), floored at 1, then capped
  close(renderedDiscPx(1, 10, 1, false, false), Math.pow(10, 0.78), 1e-9, "rel=1 → m^0.78");
  close(
    renderedDiscPx(1, 10, 11.21, false, false),
    Math.pow(10, 0.78) * Math.pow(11.21, -0.22),
    1e-9,
    "big bodies get proportionally less enlargement (rel^−0.22)",
  );
  assert.ok(
    renderedDiscPx(1, 10, 0.27, false, false) > renderedDiscPx(1, 10, 1, false, false),
    "small bodies get proportionally more",
  );
  // never shrinks below true (mEff floored at 1): Jupiter at s barely >1
  assert.equal(renderedDiscPx(7, 1.5, 11.21, false, false), 7, "mEff floor: never below true size");
  // cap: max slider drives any sub-cap disc to the cap, never past it
  assert.equal(renderedDiscPx(1, SIZE_MULT_MAX, 1, false, false), SIZE_APPARENT_CAP_PX);
  assert.equal(renderedDiscPx(30, 648, 1, false, false), SIZE_APPARENT_CAP_PX);
  // TRUE disc at/above the cap renders TRUE — fly-to framing (~612 px) is
  // exact at any s; so is the seam's 51 px Earth for a non-anchor twin
  assert.equal(renderedDiscPx(612, SIZE_MULT_MAX, 1, false, false), 612);
  assert.equal(renderedDiscPx(51.4, SIZE_MULT_MAX, 1, false, false), 51.4);
  // continuity at the cap boundary
  close(
    renderedDiscPx(39.999, SIZE_MULT_MAX, 1, false, false),
    renderedDiscPx(40.001, SIZE_MULT_MAX, 1, false, false),
    0.01,
    "continuous at cap",
  );
  // monotone non-increasing apparent size as the camera pulls away (disc
  // shrinks with distance; rendered must never grow while zooming out)
  let prev = Infinity;
  for (let d = 200; d > 0.01; d *= 0.9) {
    const r = renderedDiscPx(d, 648, 1, false, false);
    assert.ok(r <= prev + 1e-9, `apparent size monotone (disc ${d.toFixed(2)})`);
    prev = r;
  }
});

test("map-anchor body is NEVER size-scaled — the seam contract survives any s", () => {
  for (const s of [1, 648, SIZE_MULT_MAX]) {
    assert.equal(renderedDiscPx(51.36, s, 1, true, false), 51.36, `anchor true at s=${s}`);
    assert.equal(renderedDiscPx(3.8, s, 1, true, false), 3.8, `anchor true small disc at s=${s}`);
  }
});

test("a body that RIDES the anchor (the Moon) is true-scale like Earth — fixes Moon>Earth AND the cap-pinned zoom", () => {
  // parentIsMapAnchor = true (6th arg): render TRUE at any s, exactly like the
  // anchor. Identical true disc far and near means the far impostor sprite
  // tracks the close true-geometry sphere — no cap-pinned "stuck then snap".
  for (const s of [1, 648, SIZE_MULT_MAX]) {
    assert.equal(renderedDiscPx(2.5, s, 0.273, false, false, true), 2.5, `anchor-rider true at s=${s}`);
    assert.equal(renderedDiscPx(38, s, 0.273, false, false, true), 38, `anchor-rider true near cap at s=${s}`);
  }
  // Earth (anchor) and its Moon (anchor-rider) at the SAME range keep their
  // REAL size ratio: Moon = 0.273 × Earth, not the boost's ~200×.
  const earthTrue = 10;
  const moonTrue = earthTrue * 0.273;
  const earth = renderedDiscPx(earthTrue, 648, 1, true, false); // anchor
  const moon = renderedDiscPx(moonTrue, 648, 0.273, false, false, true); // rider
  assert.ok(moon < earth, `Moon (${moon}) smaller than Earth (${earth})`);
  close(moon / earth, 0.273, 1e-9, "Earth/Moon ratio stays real");
  // guard against silent regression: WITHOUT the exemption the same Moon
  // balloons to the cap and dwarfs the never-scaled Earth — the reported bug.
  const moonBoosted = renderedDiscPx(moonTrue, 648, 0.273, false, false, false);
  assert.ok(moonBoosted > earth, `un-exempt Moon (${moonBoosted}) WOULD dwarf Earth (${earth})`);
});

test("SUN CAP: max exaggeration never swallows the inner system", () => {
  // world-space: 20 R☉ stays under a third of Mercury's TRUE perihelion —
  // the smallest compressed perihelion over all c (sub-AU distances only
  // move OUTWARD under compression)
  const sunCapM = SUN_SIZE_MULT_CAP * 695_700_000;
  const mercPeriM = 0.307 * AU_M;
  for (const c of [0, 0.5, 1]) {
    assert.ok(
      sunCapM < 0.35 * compressDistance(mercPeriM, c),
      `20 R☉ < 0.35× Mercury perihelion at c=${c}`,
    );
  }
  // screen-space at the VISIBLE framing: camera 15 AU out, 900 px viewport.
  // Sun's true disc ≈ 0.84 px; at max s the sun renders ≤ min(20×, cap)
  // (through the response curve the Sun's rel=109 pulls its mEff to ~3.7)
  const k = 450 / Math.tan((36.87 / 2) * Math.PI / 180);
  const sunTrueDisc = 2 * k * Math.tan(Math.asin(695_700_000 / (15 * AU_M)));
  const sunRendered = renderedDiscPx(sunTrueDisc, SIZE_MULT_MAX, 109.2, false, true);
  assert.ok(sunRendered <= Math.min(SUN_SIZE_MULT_CAP * sunTrueDisc, SIZE_APPARENT_CAP_PX) + 1e-9,
    `sun rendered ${sunRendered.toFixed(1)}px respects both caps`);
  // Mercury's compressed orbit (0.65 AU) projects ~58 px from the Sun at
  // that framing — the capped Sun radius stays well inside it
  const mercSepPx = (k * compressDistance(0.387 * AU_M, 1)) / (15 * AU_M);
  assert.ok(sunRendered / 2 < 0.5 * mercSepPx,
    `sun radius ${(sunRendered / 2).toFixed(1)}px < half Mercury's ${mercSepPx.toFixed(1)}px separation`);
});

test("satellites of NON-anchor parents still enlarge (Jovian moons etc.; Earth's Moon is exempt, tested above)", () => {
  // A rel-0.273 satellite whose parent is NOT the map anchor (parentIsMapAnchor
  // defaults false) still rides the slider toward the cap like any body — only
  // the anchor's own satellite (Earth's Moon) is true-scaled.
  const mEff = Math.pow(SIZE_MULT_MAX, 0.78) * Math.pow(0.273, -0.22);
  const d = renderedDiscPx(0.001, SIZE_MULT_MAX, 0.273, false, false);
  close(d, Math.min(0.001 * mEff, SIZE_APPARENT_CAP_PX), 1e-9, "moon follows the response curve");
  assert.ok(d > 0.5, `visible at max slider (${d.toFixed(2)}px)`);
});

// ── state, sliders, presets, store ──────────────────────────────────────────

test("clampScaleState: bounds + garbage fall back to the VISIBLE default", () => {
  assert.deepEqual(clampScaleState({ c: -0.2, s: 0.5 }), { c: 0, s: SIZE_MULT_MIN });
  assert.deepEqual(clampScaleState({ c: 7, s: 1e9 }), { c: 1, s: SIZE_MULT_MAX });
  assert.deepEqual(clampScaleState({ c: Number.NaN, s: undefined as unknown as number }),
    { c: SCALE_PRESET_VISIBLE.c, s: SCALE_PRESET_VISIBLE.s });
  assert.deepEqual(clampScaleState(null), { c: SCALE_PRESET_VISIBLE.c, s: SCALE_PRESET_VISIBLE.s });
});

test("presets: TRUE is the exact identity state; VISIBLE is the startup default", () => {
  assert.deepEqual({ ...SCALE_PRESET_TRUE }, { c: 0, s: 1 });
  assert.ok(isTrueScale(SCALE_PRESET_TRUE));
  assert.ok(!isTrueScale(SCALE_PRESET_VISIBLE));
  assert.equal(SCALE_PRESET_VISIBLE.c, 1, "VISIBLE fully compressed (all planets in view)");
  assert.ok(SCALE_PRESET_VISIBLE.s >= 100 && SCALE_PRESET_VISIBLE.s <= 1000, "VISIBLE size in the readable band");
  // no localStorage in node → readScalePref returns the startup default
  assert.deepEqual(readScalePref(), { ...SCALE_PRESET_VISIBLE });
});

test("size slider: log mapping, round-trips, endpoints", () => {
  assert.equal(sizeSliderToMult(0), 1);
  assert.equal(sizeSliderToMult(100), SIZE_MULT_MAX);
  assert.equal(multToSizeSlider(1), 0);
  assert.equal(multToSizeSlider(SIZE_MULT_MAX), 100);
  // the reference's confirmed default stop: slider 76 → ×647 (the human's
  // screenshot read ×648 — same stop, one rounding hair), and the VISIBLE
  // preset lands back on 76 exactly
  assert.equal(sizeSliderToMult(76), 647);
  assert.equal(multToSizeSlider(SCALE_PRESET_VISIBLE.s), 76);
  for (const v of [25, 50, 76, 90]) {
    const s = sizeSliderToMult(v);
    assert.ok(Math.abs(multToSizeSlider(s) - v) <= 1, `round trip at v=${v} (s=${s})`);
  }
  // low-end integer quantization (×2 spans several slider stops on a
  // 1..5000 log scale): round trip stays within 3 stops
  assert.ok(Math.abs(multToSizeSlider(sizeSliderToMult(10)) - 10) <= 3, "low-end round trip ≤3");
  // monotone
  let prev = 0;
  for (let v = 0; v <= 100; v += 5) {
    const s = sizeSliderToMult(v);
    assert.ok(s >= prev, `slider monotone at ${v}`);
    prev = s;
  }
});

test("radii sanity: the bodies this model scales are the registry's real radii", () => {
  // guards against a future registry radius change silently invalidating
  // the sun-cap analysis above
  assert.equal(BODY_RADIUS_M.sun, 695_700_000);
  assert.equal(BODY_RADIUS_M.mercury, 2_439_700);
});
