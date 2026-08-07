import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { resolveGroundDisplayZ } from "./groundDatum";

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
    /toGroundZ: terrainOn \? groundZAt\(map, lo, la\)/,
    /groundZ: terrOn \? groundZAt\(map, lon, lat\)/,
    /const gZ = terrainOn \? groundZAt\(map, lon, lat\)/,
  ]) {
    assert.match(src, site, `a tail/marker/tag ground read bypasses groundZAt: ${site}`);
  }
});
