import { test } from "node:test";
import assert from "node:assert/strict";
import {
  cameraQueryBBox, cameraEye, groundFootprint, horizonDistM, metersPerPixel, bboxContains, bboxAreaDeg2,
  needsRefetch, bboxParam, wrapLon, CameraTargetTracker, REPLAY_CAMERA, WORLD_BBOX,
  type CameraPose,
} from "./replayCamera.ts";

const VP = { widthPx: 1000, heightPx: 800 };
const pose = (o: Partial<CameraPose> = {}): CameraPose =>
  ({ lon: -100, lat: 40, zoom: 10, pitch: 0, bearing: 0, ...o });

test("top-down view: the footprint matches MapLibre's meters-per-pixel", () => {
  const q = cameraQueryBBox(pose(), VP);
  assert.equal(q.world, false);
  assert.equal(q.clamped, false);
  const mpp = metersPerPixel(40, 10);
  const halfHkm = (400 * mpp) / 1000;
  const dLat = (q.visible.n - 40);
  assert.ok(Math.abs(dLat - halfHkm / 111.195) / (halfHkm / 111.195) < 0.02, `dLat ${dLat}`);
  assert.ok(Math.abs((40 - q.visible.s) - dLat) < 1e-6, "symmetric north/south at pitch 0");
  const halfWdeg = (500 * mpp) / (111195 * Math.cos(40 * Math.PI / 180));
  assert.ok(Math.abs((q.visible.e + 100) - halfWdeg) / halfWdeg < 0.02);
  // margin ring: 25% of each span on each side
  assert.ok(Math.abs((q.query.n - q.visible.n) - 0.25 * (q.visible.n - q.visible.s)) < 1e-6);
  assert.ok(bboxContains(q.query, q.visible));
});

test("pitched view extends toward the horizon: far field >> near field", () => {
  const q = cameraQueryBBox(pose({ pitch: 60 }), VP);
  const far = q.visible.n - 40, near = 40 - q.visible.s;
  assert.ok(far > 2 * near, `far ${far} near ${near}`);
  assert.equal(q.clamped, false, "at 60° the top edge still hits the ground inside the clamp");
});

test("pitched past the horizon: rays above it clamp at min(horizon, MAX_DIST_KM)", () => {
  const p = pose({ pitch: 80, zoom: 12 });
  const fp = groundFootprint(p, VP);
  const top = fp.points.slice(0, 3);
  assert.ok(top.every((pt) => pt.clamped), "top-edge rays are above the horizon");
  assert.ok(fp.clampDistM <= REPLAY_CAMERA.MAX_DIST_KM * 1000 + 1e-6);
  assert.ok(fp.clampDistM <= horizonDistM(fp.cameraHeightM) + 1e-6);
  const q = cameraQueryBBox(p, VP);
  assert.equal(q.clamped, true);
  const farKm = (q.visible.n - 40) * 111.195;
  assert.ok(Number.isFinite(farKm) && farKm > 0);
  assert.ok(farKm <= fp.clampDistM / 1000 + 1, `far extent ${farKm} km within the clamp`);
  // a low camera's horizon is the binding clamp, not MAX_DIST_KM
  const low = groundFootprint(pose({ pitch: 85, zoom: 16 }), VP);
  assert.ok(low.clampDistM < REPLAY_CAMERA.MAX_DIST_KM * 1000, `low camera clamps at its horizon (${low.clampDistM})`);
  assert.ok(Math.abs(low.clampDistM - horizonDistM(low.cameraHeightM)) < 1e-6);
});

test("bearing rotates the far field: bearing 90 looks east", () => {
  const q = cameraQueryBBox(pose({ pitch: 60, bearing: 90 }), VP);
  const east = q.visible.e + 100, west = -100 - q.visible.w;
  assert.ok(east > 2 * west, `east ${east} west ${west}`);
});

test("cameraEye: straight down = above the center; pitched = behind it", () => {
  const top = cameraEye(pose(), VP);
  assert.ok(Math.abs(top.lat - 40) < 1e-9 && Math.abs(top.lon + 100) < 1e-9);
  assert.ok(top.heightM > 0);
  const tilted = cameraEye(pose({ pitch: 60 }), VP);
  assert.ok(tilted.lat < 40, "bearing 0: the eye sits south of the center");
  assert.ok(Math.abs(tilted.heightM - top.heightM * 0.5) / top.heightM < 1e-6);
});

test("world zoom answers the whole globe", () => {
  const q = cameraQueryBBox(pose({ zoom: 1.5 }), VP);
  assert.equal(q.world, true);
  assert.deepEqual(q.query, WORLD_BBOX);
});

test("antimeridian view produces a seam box that contains its own footprint", () => {
  const q = cameraQueryBBox(pose({ lon: 179.95, lat: 0, zoom: 8 }), VP);
  assert.ok(q.query.w > q.query.e, `seam box w=${q.query.w} e=${q.query.e}`);
  assert.ok(bboxContains(q.query, q.visible));
  assert.ok(bboxContains(q.query, { w: 179.9, s: -0.1, e: -179.95, n: 0.1 }));
  assert.equal(wrapLon(190), -170);
  assert.equal(wrapLon(-190), 170);
  assert.equal(bboxParam({ w: 1, s: 2, e: 3, n: 4 }), "1.0000,2.0000,3.0000,4.0000");
});

test("needsRefetch: inside the margin ring = no refetch; leaving it or zooming into a capped read = refetch", () => {
  const base = cameraQueryBBox(pose(), VP);
  assert.equal(needsRefetch(null, base), true);
  const fetched = { query: base.query, capped: false };
  // small pan: still inside the padded box
  const nudged = cameraQueryBBox(pose({ lon: -100 + 0.05 }), VP);
  assert.equal(needsRefetch(fetched, nudged), false);
  // big pan: out
  const panned = cameraQueryBBox(pose({ lon: -99 }), VP);
  assert.equal(needsRefetch(fetched, panned), true);
  // zoom in 2 levels: contained, but a CAPPED response must re-read narrower
  const zoomed = cameraQueryBBox(pose({ zoom: 12 }), VP);
  assert.equal(needsRefetch(fetched, zoomed), false);
  assert.equal(needsRefetch({ ...fetched, capped: true }, zoomed), true);
  assert.ok(bboxAreaDeg2(zoomed.visible) < bboxAreaDeg2(base.visible));
});

test("CameraTargetTracker: no target mid-gesture; settled pose after SETTLE_MS; explicit destination immediately", () => {
  const tr = new CameraTargetTracker(250);
  assert.equal(tr.update(pose(), 0), null);
  assert.equal(tr.update(pose({ lon: -99.9 }), 100), null, "moving → unknown destination");
  assert.equal(tr.update(pose({ lon: -99.8 }), 200), null);
  assert.equal(tr.update(pose({ lon: -99.8 }), 300), null, "only 100 ms still");
  const settled = tr.update(pose({ lon: -99.8 }), 460);
  assert.ok(settled && Math.abs(settled.lon + 99.8) < 1e-9);
  // our own flight: the destination is the target while the camera is mid-flight
  tr.setExplicitTarget(pose({ lon: -90, zoom: 11 }), 500, 1200);
  const mid = tr.update(pose({ lon: -95 }), 600);
  assert.ok(mid && mid.lon === -90 && mid.zoom === 11);
  // after the hold, back to settle semantics
  assert.equal(tr.update(pose({ lon: -90.01 }), 1800), null);
  assert.ok(tr.update(pose({ lon: -90.01 }), 2100));
});
