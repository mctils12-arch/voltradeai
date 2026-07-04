// Sampled weather grid — pure-module tests.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  snapBbox, bboxKey, gridPoints, spacingKm, owmPointUrl, parseOwmPoint,
  GRID_MAX_POINTS,
} from "./weatherGrid";

const here = path.dirname(fileURLToPath(import.meta.url));

test("grid density is CAPPED — arrows can never be denser than the sample budget", () => {
  const b = snapBbox(25, -125, 49, -66);
  const pts = gridPoints(b);
  assert.ok(pts.length <= GRID_MAX_POINTS, `${pts.length} > cap`);
  assert.ok(pts.length >= 9, "grid too sparse to be useful");
  for (const p of pts) {
    assert.ok(p.la > b.s && p.la < b.n && p.lo > b.w && p.lo < b.e, "point outside bbox");
  }
});

test("bbox snapping: nearby viewports share one cache bucket", () => {
  const a = snapBbox(25.1, -125.2, 49.3, -66.1);
  const b = snapBbox(25.9, -124.4, 48.8, -66.9);
  assert.equal(bboxKey(a), bboxKey(b), "nearby viewports must share the bucket (shared upstream burst)");
});

test("honesty label: spacing reflects real sampling, scales with area", () => {
  const wide = spacingKm({ s: 25, w: -125, n: 49, e: -66 }, 40);
  const narrow = spacingKm({ s: 33, w: -120, n: 36, e: -116 }, 40);
  assert.ok(wide > narrow, "wider view must report coarser sampling");
  assert.ok(wide > 100, `continental sampling should be >100km (got ${wide})`);
});

test("point parse: missing fields stay null (UI skips, never invents); key encoded in URL", () => {
  const s = parseOwmPoint(35.5, -97.2, { main: { temp: 31.42 }, wind: { deg: 187, speed: 6.28 } });
  assert.deepEqual(s, { la: 35.5, lo: -97.2, tc: 31.4, wd: 187, ws: 6.3 });
  const missing = parseOwmPoint(1, 2, {});
  assert.equal(missing.tc, null);
  assert.equal(missing.wd, null);
  assert.ok(owmPointUrl(1, 2, "k&x").includes("k%26x"));
});

test("wiring pinned: grid route registered; field flags in the registry; honesty note reaches responses", () => {
  const routes = fs.readFileSync(path.join(here, "routes.ts"), "utf8");
  assert.ok(routes.includes('"/api/data/weather/grid"'), "grid route missing");
  assert.ok(routes.includes("never denser than the data"), "sampling-honesty note must ship in responses");
  const layers = JSON.parse(fs.readFileSync(path.join(here, "..", "datacore", "layers.json"), "utf8"));
  for (const id of ["weather", "weather_temp", "weather_wind"]) {
    const l = layers.layers.find((x: any) => x.id === id);
    assert.equal(l.field, true, `${id} must carry the registry field flag (opacity control inheritance)`);
  }
});
