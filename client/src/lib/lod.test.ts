// LOD director math — EARTH TWIN A1 (research/earth_twin_program.md).
// Pure tests: no MapLibre, no DOM. The formula constants are pinned against
// the installed maplibre-gl 5.24.0 implementation (512px worlds →
// 78271.51696 m/px at z0 equator; getCameraAltitude = cos(pitch)·
// (0.5·height/tan(fov/2))·metersPerPixel + centerElevation).
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  METERS_PER_PIXEL_Z0,
  metersPerPixel,
  cameraAltitudeMeters,
  cameraAltitudeKmFromMap,
  zoomForCameraAltitudeKm,
  lodOpacity,
  type LodEnvelope,
  type MapLike,
} from "./lod.js";

test("metersPerPixel: 512px-world constant at z0 equator, halves per zoom, cos(lat) scaled", () => {
  assert.ok(Math.abs(METERS_PER_PIXEL_Z0 - 78271.51696) < 0.001,
    "z0 equator constant must be earth circumference / 512 (MapLibre worlds are 512px tiles, NOT 256)");
  assert.ok(Math.abs(metersPerPixel(0, 0) - METERS_PER_PIXEL_Z0) < 1e-9);
  assert.ok(Math.abs(metersPerPixel(0, 1) - METERS_PER_PIXEL_Z0 / 2) < 1e-9, "one zoom level halves the scale");
  assert.ok(Math.abs(metersPerPixel(60, 4) - (METERS_PER_PIXEL_Z0 * 0.5) / 16) < 1e-6, "cos(60°)=0.5 latitude factor");
  // poles clamp instead of hitting cos(90) = 0 exactly (no zero/NaN altitude)
  assert.ok(metersPerPixel(90, 4) > 0);
});

test("cameraAltitudeMeters: matches MapLibre's formula shape and responds to each input correctly", () => {
  const base = { zoom: 10, latDeg: 0, canvasHeightPx: 800 };
  // hand-computed: 0.5*800/tan(18.435°) * 78271.51696/1024 ≈ 1200.13 px * 76.437 m/px
  const expected = ((0.5 * 800) / Math.tan((36.87 * Math.PI) / 360)) * (METERS_PER_PIXEL_Z0 / 1024);
  assert.ok(Math.abs(cameraAltitudeMeters(base) - expected) < 1e-6);
  assert.ok(cameraAltitudeMeters({ ...base, zoom: 11 }) < cameraAltitudeMeters(base),
    "zooming in lowers the camera");
  assert.ok(cameraAltitudeMeters({ ...base, latDeg: 60 }) < cameraAltitudeMeters(base),
    "same zoom at high latitude = smaller ground scale = lower camera");
  assert.ok(cameraAltitudeMeters({ ...base, pitchDeg: 60 }) < cameraAltitudeMeters(base),
    "pitching keeps the same camera distance so altitude drops by cos(pitch) — MapLibre's own formula");
  assert.equal(
    cameraAltitudeMeters({ ...base, centerElevationM: 1000 }) - cameraAltitudeMeters(base),
    1000, "terrain elevation adds through");
});

test("cameraAltitudeKmFromMap: prefers transform.getCameraAltitude, falls back to the pure formula, fails open on total breakage", () => {
  const viaTransform: MapLike = {
    transform: { getCameraAltitude: () => 250_000 },
    getZoom: () => { throw new Error("must not be called when transform path works"); },
    getCenter: () => ({ lat: 0 }),
    getCanvas: () => ({ height: 800 }),
  };
  assert.equal(cameraAltitudeKmFromMap(viaTransform), 250);

  const viaFallback: MapLike = {
    transform: {},
    getZoom: () => 10,
    getCenter: () => ({ lat: 0 }),
    getCanvas: () => ({ height: 800 }),
  };
  const km = cameraAltitudeKmFromMap(viaFallback);
  assert.ok(km != null && Math.abs(km - cameraAltitudeMeters({ zoom: 10, latDeg: 0, canvasHeightPx: 800 }) / 1000) < 1e-9);

  const broken: MapLike = {
    transform: { getCameraAltitude: () => { throw new Error("boom"); } },
    getZoom: () => { throw new Error("boom"); },
    getCenter: () => ({ lat: 0 }),
    getCanvas: () => ({ height: 800 }),
  };
  assert.equal(cameraAltitudeKmFromMap(broken), null, "both paths broken → null (caller fails open)");

  const nonFinite: MapLike = {
    transform: { getCameraAltitude: () => NaN },
    getZoom: () => 10,
    getCenter: () => ({ lat: 0 }),
    getCanvas: () => ({ height: 800 }),
  };
  assert.ok(cameraAltitudeKmFromMap(nonFinite) != null, "NaN from transform falls through to the formula");
});

test("lodOpacity: camMin envelope — hidden below, linear fade band, visible above", () => {
  const env: LodEnvelope = { camMinKm: 100, fadeBandKm: 150 };
  assert.equal(lodOpacity(env, 50), 0, "below camMin = hidden");
  assert.equal(lodOpacity(env, 100), 0, "at camMin = still hidden (bound is exclusive upward)");
  assert.ok(Math.abs(lodOpacity(env, 175) - 0.5) < 1e-9, "midpoint of the fade band = 0.5");
  assert.equal(lodOpacity(env, 250), 1, "top of the fade band = fully visible");
  assert.equal(lodOpacity(env, 10_000), 1, "far above = fully visible");
});

test("lodOpacity: camMax envelope and two-sided envelopes compose by min()", () => {
  const maxOnly: LodEnvelope = { camMaxKm: 1000, fadeBandKm: 200 };
  assert.equal(lodOpacity(maxOnly, 1500), 0);
  assert.ok(Math.abs(lodOpacity(maxOnly, 900) - 0.5) < 1e-9);
  assert.equal(lodOpacity(maxOnly, 700), 1);

  const band: LodEnvelope = { camMinKm: 100, camMaxKm: 1000, fadeBandKm: 100 };
  assert.equal(lodOpacity(band, 50), 0);
  assert.equal(lodOpacity(band, 500), 1, "inside the band, past both fades");
  assert.ok(lodOpacity(band, 150) < 1 && lodOpacity(band, 150) > 0, "inside the lower fade");
  assert.ok(lodOpacity(band, 950) < 1 && lodOpacity(band, 950) > 0, "inside the upper fade");
});

test("lodOpacity: hard step without fadeBand; honesty fail-open rules", () => {
  const hard: LodEnvelope = { camMinKm: 100 };
  assert.equal(lodOpacity(hard, 99.999), 0);
  assert.equal(lodOpacity(hard, 100.001), 1, "no fade band = step function");

  assert.equal(lodOpacity(undefined, 5), 1, "no envelope = always visible");
  assert.equal(lodOpacity({}, 5), 1, "empty envelope gates nothing");
  assert.equal(lodOpacity(hard, null), 1, "unmeasurable camera = FAIL OPEN, never hide data on broken math");
  assert.equal(lodOpacity(hard, Number.NaN), 1, "NaN camera = fail open");
});

test("zoomForCameraAltitudeKm inverts cameraAltitudeMeters (craft framing never dives inside the orbit shell)", () => {
  for (const altKm of [900, 2400, 35786 * 2.3, 120000]) {
    for (const lat of [0, 38, -62]) {
      const zoom = zoomForCameraAltitudeKm(altKm, lat, 900);
      const back = cameraAltitudeMeters({ zoom, latDeg: lat, canvasHeightPx: 900 }) / 1000;
      assert.ok(Math.abs(back - altKm) / altKm < 0.01,
        `round-trip alt ${altKm}km lat ${lat}: zoom ${zoom.toFixed(2)} -> ${back.toFixed(0)}km`);
    }
  }
});
