import { test } from "node:test";
import assert from "node:assert/strict";
import { gebcoWmsTileUrl, GEBCO_WMS_BASE, GEBCO_TID_LAYER } from "./gebco";

test("gebcoWmsTileUrl: builds a WMS GetMap template with the bbox token intact", () => {
  const url = gebcoWmsTileUrl(GEBCO_TID_LAYER);
  assert.ok(url.startsWith(GEBCO_WMS_BASE + "?"), "must hit the GEBCO WMS base endpoint");
  assert.ok(url.includes("service=WMS"), "must request WMS");
  assert.ok(url.includes("request=GetMap"), "must request GetMap");
  assert.ok(url.includes(`layers=${GEBCO_TID_LAYER}`), "must name the TID confidence layer");
  assert.ok(url.includes("crs=EPSG%3A3857") || url.includes("crs=EPSG:3857"), "must request web-mercator");
  assert.ok(url.endsWith("&bbox={bbox-epsg-3857}"), "must end with the literal MapLibre WMS bbox token");
});

test("gebcoWmsTileUrl: any layer name is substitutable (pure template builder)", () => {
  const url = gebcoWmsTileUrl("GEBCO_2024");
  assert.ok(url.includes("layers=GEBCO_2024&") || url.includes("layers=GEBCO_2024&"));
  assert.ok(!url.includes("GEBCO_2024_3"), "must not silently hardcode the TID layer for a different request");
});
