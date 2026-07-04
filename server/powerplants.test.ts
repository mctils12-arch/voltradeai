// US power plants layer (RAW, WRI GPPD CC BY 4.0) — data integrity + wiring.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const dataPath = path.join(here, "..", "datacore", "powerplants", "us_power_plants.json");
const data = JSON.parse(fs.readFileSync(dataPath, "utf8"));

test("compiled dataset is US-scale with valid compact rows", () => {
  assert.ok(data.count > 9000, `expected ~9.8k US plants, got ${data.count}`);
  assert.equal(data.count, data.plants.length);
  const fuels = new Set(data.fuels);
  for (const p of data.plants) {
    const [name, mw, fuel, _owner, lat, lon] = p;
    assert.ok(typeof name === "string" && name.length > 0);
    assert.ok(typeof mw === "number" && mw >= 0);
    assert.ok(fuels.has(fuel), `unknown fuel code ${fuel}`);
    // US bounds incl. AK/HI + territories on both sides of the date line
    // (GPPD counts Guam/N. Marianas as USA — lon ~144-146 E)
    assert.ok(lat > 12 && lat < 72, `lat out of range: ${lat} (${name})`);
    assert.ok((lon > -180 && lon < -60) || (lon > 140 && lon <= 180), `lon out of range: ${lon} (${name})`);
  }
});

test("attribution ships with the data (CC BY 4.0 requires it)", () => {
  assert.ok(String(data.source).includes("WRI"));
  assert.ok(String(data.source).includes("CC BY 4.0"));
});

test("route + layer registry wired", () => {
  const routes = fs.readFileSync(path.join(here, "routes.ts"), "utf8");
  assert.ok(routes.includes("/api/data/powerplants"), "route missing");
  assert.ok(routes.includes("us_power_plants.json"), "static import missing (Dockerfile never copies datacore/ — esbuild must bake it)");
  const layers = JSON.parse(fs.readFileSync(path.join(here, "..", "datacore", "layers.json"), "utf8"));
  const pp = layers.layers.find((l: any) => l.id === "powerplants");
  assert.ok(pp, "layers.json entry missing");
  assert.equal(pp.kind, "raw");
  assert.ok(pp.source.includes("CC BY 4.0"), "layer attribution missing");
});
