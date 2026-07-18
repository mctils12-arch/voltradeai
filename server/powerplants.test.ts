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
    const [name, mw, fuel, _owner, lat, lon, verified] = p;
    assert.ok(typeof name === "string" && name.length > 0);
    assert.ok(typeof mw === "number" && mw >= 0);
    assert.ok(fuels.has(fuel), `unknown fuel code ${fuel}`);
    // US bounds incl. AK/HI + territories on both sides of the date line
    // (GPPD counts Guam/N. Marianas as USA — lon ~144-146 E)
    assert.ok(lat > 12 && lat < 72, `lat out of range: ${lat} (${name})`);
    assert.ok((lon > -180 && lon < -60) || (lon > 140 && lon <= 180), `lon out of range: ${lon} (${name})`);
    assert.ok(verified === 0 || verified === 1, `verified flag must be 0/1 (${name})`);
  }
});

test("position provenance: top plants imagery-verified, audit artifact present", () => {
  assert.equal(data.verified_count, 100, "top-100 verification count");
  // every one of the 100 largest plants carries the verified flag
  const top100 = [...data.plants].sort((a: any, b: any) => b[1] - a[1]).slice(0, 100);
  assert.ok(top100.every((p: any) => p[6] === 1), "a top-100-by-MW plant lost its verified flag");
  const audit = JSON.parse(fs.readFileSync(path.join(here, "..", "datacore", "powerplants", "imagery_verified.json"), "utf8"));
  assert.equal(audit.ids.length, 100);
  assert.ok(String(data.source).includes("EIA-860"), "EIA-860 must be credited as coordinate source");
});

test("OSM position overrides applied (position audit 2026-07-18)", () => {
  // 4 wind plants were 5-22km from their real farms (registry-chain error;
  // research/position_audit_2026-07-18.md). The overrides file is the
  // provenance record; the shipped rows must carry the corrected coords —
  // this catches a rebuild that silently drops position_overrides.json.
  const ov = JSON.parse(fs.readFileSync(
    path.join(here, "..", "datacore", "powerplants", "position_overrides.json"), "utf8"));
  assert.ok(ov.overrides.length >= 4, "override records missing");
  for (const o of ov.overrides) {
    assert.ok(o.gppd_idnr && o.name && o.from?.length === 2 && o.to?.length === 2 && o.evidence,
      `override record incomplete: ${o.name}`);
    const row = data.plants.find((p: any) => p[0] === o.name);
    assert.ok(row, `override target not in dataset: ${o.name}`);
    assert.equal(row[4], Math.round(o.to[0] * 1e4) / 1e4, `${o.name}: lat not overridden`);
    assert.equal(row[5], Math.round(o.to[1] * 1e4) / 1e4, `${o.name}: lon not overridden`);
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
