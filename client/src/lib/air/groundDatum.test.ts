import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { holdGroundZ, resolveGroundDisplayZ } from "./groundDatum";

// ── the field defect (repair 2026-08-07) ────────────────────────────────────
// Phone screenshots: the blue ground trace floated above valleys / cut
// through ridges. Cause: the curtain base preferred the fixed-zoom DEM
// sample over the RENDERED mesh, so at coarse far-LOD the trace rode a
// surface that wasn't the one on screen; the moving tail used the opposite
// preference, so the datum also jumped at the tail seam.

test("mesh reading wins whenever the mesh has a tile — it IS the displayed surface", () => {
  // coarse far-LOD mesh says 210m where the z10 DEM sample says 340m: the
  // trace must ride the 210m the user actually sees, not float at 340m
  assert.deepEqual(resolveGroundDisplayZ(210, 340, 1.5), { g: 210, source: "mesh" });
});

test("below-sea-level terrain is a real mesh reading, not treated as unloaded", () => {
  assert.deepEqual(resolveGroundDisplayZ(-52, -60, 1), { g: -52, source: "mesh" });
});

test("mesh 0 (tile not loaded) falls back to own DEM × exaggeration — the round-17 sea-level-plunge fix survives", () => {
  assert.deepEqual(resolveGroundDisplayZ(0, 300, 1.5), { g: 450, source: "dem" });
});

test("true sea level: both sources agree near 0, no visible seam either way", () => {
  const r = resolveGroundDisplayZ(0, 0, 2);
  assert.equal(r.g, 0);
  assert.equal(r.source, "dem"); // DEM answered 0 — an honest reading, not pending
});

test("DEM tile still in flight -> 0 now, pending so the retry repaint refines", () => {
  assert.deepEqual(resolveGroundDisplayZ(0, null, 1.5), { g: 0, source: "pending" });
});

test("non-finite mesh readings never poison the base", () => {
  assert.deepEqual(resolveGroundDisplayZ(NaN, 120, 2), { g: 240, source: "dem" });
});

// ── source ratchet (repo convention) ────────────────────────────────────────
// Track build, live tail, replay marker and tag must all resolve the display
// ground through ONE rule — a refactor reintroducing a raw mesh-only or
// DEM-first read at any of those sites recreates the seam/float.

test("datamap routes every curtain/tail/marker ground read through the resolver", () => {
  const src = readFileSync(new URL("../../pages/datamap.tsx", import.meta.url), "utf-8");
  assert.match(src, /resolveGroundDisplayZ\(groundDisplayAt\(map, s\.lon, s\.lat\), gDem, altScale\)/,
    "track build loop lost the mesh-first resolver");
  for (const site of [
    // the tail now passes a last-known-good fallback (2026-08-12) — the call
    // must still go through groundZAt, with or without that argument
    /toGroundZ: terrainOn \? groundZAt\(map, lo, la[,)]/,
    /groundZ: terrOn \? groundZAt\(map, lon, lat[,)]/,
    /const gZ = terrainOn \? groundZAt\(map, lon, lat[,)]/,
  ]) {
    assert.match(src, site, `a tail/marker/tag ground read bypasses groundZAt: ${site}`);
  }
  // STRENGTHENED (2026-08-12): the tail must not merely call groundZAt, it
  // must hand it a fallback. Without one, a camera move that flips the
  // aircraft's position to `pending` collapses the anchor to 0 and the plane
  // at the end of the track disappears.
  assert.match(src, /toGroundZ: terrainOn \? groundZAt\(map, lo, la, st\.groundZ\[li\]\)/,
    "the flight tail must pass a last-known-good ground fallback (Law I/V)");
  assert.match(src, /holdGroundZ\(r, fallbackG\)/,
    "groundZAt must route the pending case through holdGroundZ");
});

// ── holdGroundZ — the camera-coupled ground collapse (2026-08-12) ──────────
// Live symptom: "if I move around and do different camera angles it will
// fail and the plane at the end disappears". Root cause: a pure camera move
// flips a position from mesh-known to pending, and pending carried g:0, so
// the tail's ground anchor snapped from real terrain height to sea level.

test("holdGroundZ passes real measurements straight through", () => {
  assert.deepEqual(holdGroundZ({ g: 250, source: "mesh" }, 999), { g: 250, held: false });
  assert.deepEqual(holdGroundZ({ g: 180, source: "dem" }, 999), { g: 180, held: false });
});

test("holdGroundZ HOLDS the last-known ground instead of rendering a fabricated zero", () => {
  // The fix. Without it this returns 0 and the tail quad degenerates.
  const r = holdGroundZ({ g: 0, source: "pending" }, 250);
  assert.equal(r.g, 250);
  assert.equal(r.held, true);
});

test("holdGroundZ falls back to the pending zero only when there is nothing to hold", () => {
  // First paint of a brand-new track has no previous value; 0 is then the
  // honest best available, and the next tick corrects it.
  assert.deepEqual(holdGroundZ({ g: 0, source: "pending" }, undefined), { g: 0, held: false });
  assert.deepEqual(holdGroundZ({ g: 0, source: "pending" }, null), { g: 0, held: false });
  assert.deepEqual(holdGroundZ({ g: 0, source: "pending" }, NaN), { g: 0, held: false });
});

test("holdGroundZ can hold a legitimate zero (sea level is a real elevation)", () => {
  // An aircraft over the ocean has ground 0 for real. Holding it must not be
  // mistaken for the pending case.
  assert.deepEqual(holdGroundZ({ g: 0, source: "mesh" }, 250), { g: 0, held: false });
});

test("a camera move that flips mesh->pending no longer moves the anchor", () => {
  // The exact regression: same world position, two camera states.
  const cameraOnMesh = resolveGroundDisplayZ(250, null, 1);
  const cameraOffMesh = resolveGroundDisplayZ(0, null, 1); // mesh unloaded, DEM not yet fetched
  assert.equal(cameraOnMesh.source, "mesh");
  assert.equal(cameraOffMesh.source, "pending");
  const held = holdGroundZ(cameraOffMesh, cameraOnMesh.g).g;
  assert.equal(held, cameraOnMesh.g, "ground anchor must not move because the camera did");
});
