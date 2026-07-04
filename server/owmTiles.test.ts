// OpenWeatherMap tile proxy — pure-module tests (no network).
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  validateWxTile, owmTileUrl, classifyOwmStatus, owmStatusNote,
  makeTileCache, MAX_WX_ZOOM,
} from "./owmTiles";

const here = path.dirname(fileURLToPath(import.meta.url));

test("tile validation: layer allowlist, zoom ceiling, x/y within 2^z", () => {
  assert.ok(validateWxTile("temp_new", 3, 4, 5));
  assert.ok(validateWxTile("wind_new", 0, 0, 0));
  assert.ok(!validateWxTile("precipitation_new", 3, 0, 0), "layer not on the allowlist");
  assert.ok(!validateWxTile("../../etc/passwd", 3, 0, 0), "traversal-shaped layer rejected");
  assert.ok(!validateWxTile("temp_new", MAX_WX_ZOOM + 1, 0, 0), "zoom above ceiling rejected");
  assert.ok(!validateWxTile("temp_new", 2, 4, 0), "x out of range for z=2");
  assert.ok(!validateWxTile("temp_new", 2, 1.5 as any, 0), "non-integer rejected");
});

test("URL builder targets OWM with the key appended and encoded", () => {
  const u = owmTileUrl("temp_new", 2, 1, 3, "abc&def");
  assert.ok(u.startsWith("https://tile.openweathermap.org/map/temp_new/2/1/3.png?appid="));
  assert.ok(u.includes("abc%26def"), "key must be URL-encoded");
});

test("fresh-key honesty: 401/403 classify as 'activating' (retry), never 'error'; missing key = awaiting_key", () => {
  assert.equal(classifyOwmStatus(null, false), "awaiting_key");
  assert.equal(classifyOwmStatus(200, true), "ok");
  assert.equal(classifyOwmStatus(401, true), "activating");
  assert.equal(classifyOwmStatus(403, true), "activating");
  assert.equal(classifyOwmStatus(500, true), "error");
  assert.ok(owmStatusNote("activating").includes("~2h"), "retry note must state the ~2h activation window");
  assert.ok(owmStatusNote("ok").includes("© OpenWeatherMap"), "attribution required by license");
});

test("tile cache: TTL expiry and bounded eviction", () => {
  const c = makeTileCache<string>(2);
  const t0 = 1_000_000;
  c.set("a", "A", 100, t0);
  assert.equal(c.get("a", t0 + 50), "A");
  assert.equal(c.get("a", t0 + 150), undefined, "expired after TTL");
  c.set("x", "X", 1000, t0);
  c.set("y", "Y", 1000, t0);
  c.set("z", "Z", 1000, t0); // evicts oldest
  assert.ok(c.size() <= 2, "cache stays bounded");
});

test("wiring pinned: proxy + status routes registered; registry entries carry OWM attribution", () => {
  const routes = fs.readFileSync(path.join(here, "routes.ts"), "utf8");
  assert.ok(routes.includes("/api/data/wxtile/:layer/:z/:x/:y"), "tile proxy route missing");
  assert.ok(routes.includes("/api/data/weather/global/status"), "status route missing");
  const layers = JSON.parse(fs.readFileSync(path.join(here, "..", "datacore", "layers.json"), "utf8"));
  for (const id of ["weather_temp", "weather_wind"]) {
    const l = layers.layers.find((x: any) => x.id === id);
    assert.ok(l && l.kind === "raw", `${id} entry missing/not raw`);
    assert.ok(l.source.includes("OpenWeatherMap"), `${id} must carry OWM attribution`);
    assert.ok(/model-derived/.test(l.description), `${id} must state model-derived honesty`);
  }
});
