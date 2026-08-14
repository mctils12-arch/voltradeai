// Q14 — four Earth constants, two names, and why unifying them would have
// been the wrong fix.
//
// D5 `conflicting_const` found two SCREAMING_CASE constants exported from more
// than one module with DIFFERENT values:
//
//   EARTH_RADIUS_KM        6371            orbital/geometry.ts
//                          6378.137        orbital/satDerived.ts
//   EARTH_CIRCUMFERENCE_M  2π × 6371008.8  glElev.ts
//                          40075016.686    lod.ts
//
// The obvious reading — "one of each pair is wrong, pick the right one" — is
// FALSE. All four values are correct where they live:
//
//   * a spherical footprint cap uses the MEAN radius;
//   * SGP4 apsides are quoted above the WGS-84 EQUATORIAL radius;
//   * altitude → mercator-z must match MAPLIBRE's own constant (6371008.8),
//     verified in its shipped source and pinned against its public API below;
//   * tile resolution is EPSG:3857, defined on the equatorial radius.
//
// So the defect was never a number. It was that two files in the same
// directory exported the same name meaning different things, and a physical
// constant's name reads as a claim about the world — an importer reaching for
// `EARTH_RADIUS_KM` has no way to know which Earth they got. The fix is
// renaming, which changes no value and therefore cannot regress anything (the
// emitted GLSL constant is byte-identical, 2.4981121215e-8, before and after).
//
// These tests exist so the collision cannot come back under the old names.
// Run: npx tsx --test client/src/lib/earthConstants.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";

import { EARTH_MEAN_RADIUS_KM } from "./orbital/geometry.js";
import { EARTH_EQUATORIAL_RADIUS_KM } from "./orbital/satDerived.js";
import { MAPLIBRE_WORLD_CIRCUMFERENCE_M } from "./glElev.js";
import { GLOBE_RADIUS_M } from "./orbital/occlusion.js";
import { WEB_MERCATOR_CIRCUMFERENCE_M, METERS_PER_PIXEL_Z0 } from "./lod.js";

type MercatorCtor = {
  fromLngLat(
    lngLat: { lng: number; lat: number },
    altitude: number,
  ): { x: number; y: number; z: number };
};

const repoRoot = path.resolve(import.meta.dirname, "..", "..", "..");

test("each constant is the documented datum, and they are NOT equal", () => {
  assert.equal(EARTH_MEAN_RADIUS_KM, 6371, "mean radius, spherical cap model");
  assert.equal(EARTH_EQUATORIAL_RADIUS_KM, 6378.137, "WGS-84 equatorial / SGP4 reference");
  assert.notEqual(
    EARTH_MEAN_RADIUS_KM,
    EARTH_EQUATORIAL_RADIUS_KM,
    "collapsing these to one value would break whichever model lost",
  );
  assert.equal(WEB_MERCATOR_CIRCUMFERENCE_M, 40075016.686, "EPSG:3857, 2π × 6378137");
  assert.ok(
    Math.abs(WEB_MERCATOR_CIRCUMFERENCE_M - 2 * Math.PI * 6378137) < 0.01,
    "the Web Mercator figure really is the EQUATORIAL circumference",
  );
});

test("the maplibre world circumference equals MapLibre's OWN constant, live", async () => {
  // Not a copied number: MercatorCoordinate.fromLngLat(_, 1).z at the equator
  // is 1 / circumference, so MapLibre states its own value here. A MapLibre
  // upgrade that moves the constant fails this test instead of silently
  // skewing every projected altitude by whatever it moved.
  // Typed structurally, not `: any`. The namespace shape differs between the
  // ESM and CJS entry points (node resolves the bundle onto `.default`), which
  // the published types do not describe — but reaching for `any` to paper over
  // that would push the `ts_any` ratchet up by one, and the ratchet is right:
  // the shape IS knowable, it just is not declared.
  type MercatorNS = {
    MercatorCoordinate?: MercatorCtor;
    default?: { MercatorCoordinate?: MercatorCtor };
  };
  const ml = (await import("maplibre-gl")) as unknown as MercatorNS;
  const MercatorCoordinate = ml.MercatorCoordinate ?? ml.default?.MercatorCoordinate;
  assert.ok(MercatorCoordinate, "MercatorCoordinate export resolved");

  const implied = 1 / MercatorCoordinate.fromLngLat({ lng: 0, lat: 0 }, 1).z;
  assert.ok(
    Math.abs(MAPLIBRE_WORLD_CIRCUMFERENCE_M - implied) < 1e-6,
    `ours ${MAPLIBRE_WORLD_CIRCUMFERENCE_M} vs MapLibre's ${implied}`,
  );

  // And it is NOT the Web Mercator figure. This is the assertion that would
  // have caught someone "unifying" the two: 0.112% on every altitude.
  const skew = Math.abs(WEB_MERCATOR_CIRCUMFERENCE_M - implied) / implied;
  assert.ok(skew > 1e-3, "the two circumferences are genuinely different data");
  assert.notEqual(MAPLIBRE_WORLD_CIRCUMFERENCE_M, WEB_MERCATOR_CIRCUMFERENCE_M);
});

test("glElev derives from GLOBE_RADIUS_M rather than restating the literal", () => {
  // 6371008.8 appeared three times before Q14. Deriving means there is one
  // place to be wrong, not three places to drift.
  assert.equal(MAPLIBRE_WORLD_CIRCUMFERENCE_M, 2 * Math.PI * GLOBE_RADIUS_M);
  const src = fs.readFileSync(
    path.join(repoRoot, "client/src/lib/glElev.ts"),
    "utf8",
  );
  const code = src
    .split("\n")
    .filter((l) => !/^\s*(\/\/|\*|\/\*)/.test(l))
    .join("\n");
  assert.ok(
    !/=\s*2\s*\*\s*Math\.PI\s*\*\s*6371008\.8/.test(code),
    "the literal is back — derive from GLOBE_RADIUS_M instead",
  );
});

test("METERS_PER_PIXEL_Z0 still follows its own circumference", () => {
  assert.ok(Math.abs(METERS_PER_PIXEL_Z0 - 78271.51696) < 0.001);
  assert.equal(METERS_PER_PIXEL_Z0, WEB_MERCATOR_CIRCUMFERENCE_M / 512);
});

test("the colliding names are gone and cannot be re-exported", () => {
  // D11. Asserted on the CONSTRUCT (an `export const` declaration), not on the
  // string — prose naming the old constant must not trip or satisfy this
  // (L15/L18, which cost two false readings and one false PASS this session).
  const files = execFileSync("git", ["ls-files", "client/src/**/*.ts", "client/src/**/*.tsx"], {
    cwd: repoRoot,
    encoding: "utf8",
  })
    .split("\n")
    .filter(Boolean);

  // NOT VACUOUS. If the glob ever stops matching, `offenders` is trivially
  // empty and this guard passes while checking nothing — precisely how
  // gridTiles.test.ts's magic-byte loop sat dead from the day it was written
  // (#835). Assert the scan reached the four files Q14 actually renamed.
  for (const must of [
    "client/src/lib/orbital/geometry.ts",
    "client/src/lib/orbital/satDerived.ts",
    "client/src/lib/glElev.ts",
    "client/src/lib/lod.ts",
  ]) {
    assert.ok(files.includes(must), `the scan did not reach ${must}`);
  }

  const offenders: string[] = [];
  for (const f of files) {
    const src = fs.readFileSync(path.join(repoRoot, f), "utf8");
    const code = src
      .split("\n")
      .filter((l) => !/^\s*(\/\/|\*|\/\*)/.test(l))
      .join("\n");
    for (const m of code.matchAll(
      /^export const (EARTH_RADIUS_KM|EARTH_CIRCUMFERENCE_M)\b/gm,
    )) {
      offenders.push(`${f}: ${m[1]}`);
    }
  }
  assert.deepEqual(
    offenders,
    [],
    `ambiguous Earth-constant name re-exported — say WHICH Earth:\n  ${offenders.join("\n  ")}`,
  );
});
