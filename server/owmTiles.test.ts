// OpenWeatherMap tile proxy — pure-module tests (no network).
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  validateWxTile, owmTileUrl, classifyOwmStatus, owmStatusNote,
  makeTileCache, MAX_WX_ZOOM, amplifyWeatherTile, WX_ALPHA_CAP,
} from "./owmTiles";
import { PNG } from "pngjs";

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

test("alpha amplification: the measured root cause is actually fixed (real prod wind tile)", () => {
  // Fixture = a real production wind_new tile captured 2026-07-04 during the
  // root-cause analysis: avg alpha 15/255, ZERO pixels above 120/255 —
  // invisible over a dark basemap. After amplification the visible fraction
  // must be dominated by mid/high-alpha pixels, capped below full opacity.
  const raw = fs.readFileSync(path.join(here, "__fixtures__", "owm_wind_tile.png"));
  const before = PNG.sync.read(raw);
  let beforeStrong = 0, beforeNonzero = 0;
  for (let i = 3; i < before.data.length; i += 4) {
    if (before.data[i] > 0) beforeNonzero++;
    if (before.data[i] > 120) beforeStrong++;
  }
  assert.equal(beforeStrong, 0, "fixture must exhibit the defect (no strong pixels)");
  assert.ok(beforeNonzero > 0, "fixture must have SOME field content");

  const out = amplifyWeatherTile(raw, "wind_new");
  const after = PNG.sync.read(out);
  let afterStrong = 0, maxA = 0;
  for (let i = 3; i < after.data.length; i += 4) {
    if (after.data[i] > 120) afterStrong++;
    if (after.data[i] > maxA) maxA = after.data[i];
  }
  assert.ok(afterStrong > beforeNonzero * 0.1,
    `amplification must create strong pixels (got ${afterStrong} of ${beforeNonzero} nonzero)`);
  assert.ok(maxA <= WX_ALPHA_CAP, `alpha capped at ${WX_ALPHA_CAP} so the basemap stays visible (max ${maxA})`);
  // fully-transparent pixels stay transparent — no field invented where none exists
  assert.equal(after.data[3] > 0, before.data[3] > 0);
});

test("amplification is fail-open: garbage input throws (route falls back to the raw buffer)", () => {
  assert.throws(() => amplifyWeatherTile(Buffer.from("not a png"), "temp_new"));
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
