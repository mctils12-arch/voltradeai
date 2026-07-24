// layerFreshness tests — Phase 5 per-layer freshness chip join.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { attachLayerFreshness, LAYER_TO_STREAM } from "./layerFreshness";
import datacoreLayers from "../datacore/layers.json";

const here = path.dirname(fileURLToPath(import.meta.url));

function stream(name: string, health: "live" | "recent" | "stale" | "no-data", age_hours: number | null) {
  return { stream: name, health, age_hours, health_note: `${name} note` };
}

test("attaches freshness only for a mapped layer with a matching stream in the cache", () => {
  const layers = [{ id: "insider", name: "Insider" }, { id: "imagery", name: "Imagery" }];
  const out = attachLayerFreshness(layers, [stream("filings", "live", 0.4)]);
  assert.equal(out[0].freshness?.health, "live");
  assert.equal(out[0].freshness?.age_hours, 0.4);
  assert.equal(out[0].freshness?.stream, "filings");
  assert.equal("freshness" in out[1], false, "unmapped layer (imagery) gets no fabricated field");
});

test("mapped layer with no matching stream in the cache yet gets no freshness field", () => {
  const layers = [{ id: "insider", name: "Insider" }];
  const out = attachLayerFreshness(layers, []); // cache empty / warming up
  assert.equal("freshness" in out[0], false);
});

test("never mutates the input layers array", () => {
  const layers = [{ id: "cot", name: "COT" }];
  const out = attachLayerFreshness(layers, [stream("cftccot", "recent", 122.9)]);
  assert.equal("freshness" in layers[0], false, "original object untouched");
  assert.notEqual(out[0], layers[0], "a new object is returned for a joined layer");
});

test("cot maps to the server-side cftccot manifest, not the Python-side cot manifest", () => {
  assert.equal(LAYER_TO_STREAM.cot, "cftccot");
});

test("every LAYER_TO_STREAM value is a real datacore/manifests/*.json stream name (no typos)", () => {
  const manifestDir = path.join(here, "..", "datacore", "manifests");
  const real = new Set(fs.readdirSync(manifestDir).filter((f) => f.endsWith(".json")).map((f) => f.replace(/\.json$/, "")));
  for (const [layerId, streamName] of Object.entries(LAYER_TO_STREAM)) {
    assert.ok(real.has(streamName), `LAYER_TO_STREAM.${layerId} -> "${streamName}" has no matching manifest file`);
  }
});

test("every LAYER_TO_STREAM key is a real layer id in datacore/layers.json (no stale entries)", () => {
  const ids = new Set((datacoreLayers as any).layers.map((l: any) => l.id));
  for (const layerId of Object.keys(LAYER_TO_STREAM)) {
    assert.ok(ids.has(layerId), `LAYER_TO_STREAM key "${layerId}" is not a real layer id`);
  }
});
