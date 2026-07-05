import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const here = path.dirname(fileURLToPath(import.meta.url));
const REG = path.join(here, "..", "datacore", "sentinel2", "cushing_tanks.geojson");

// Tank-fill v2 PR-1 (workup in research/open_questions.md): the fixed tank
// geometry the crescent-shadow estimator samples against. Pins hold the
// registry to the live-verified shape (333 polygons, 234 measurable at
// 10m) and to the licensing/provenance honesty rules.
test("cushing tank registry: counts, licensing, and height provenance", () => {
  const reg = JSON.parse(fs.readFileSync(REG, "utf8"));
  assert.ok(reg.license.includes("OpenStreetMap contributors"), "ODbL attribution required");
  assert.ok(reg.count >= 300, `expected >=300 tanks, got ${reg.count}`);
  assert.ok(reg.measurable_count >= 200, `expected >=200 measurable (>=40m) tanks, got ${reg.measurable_count}`);
  assert.equal(reg.features.length, reg.count);
  const sites = new Set(["cushing_hub", "cushing_enbridge", "cushing_plains", "ring"]);
  for (const f of reg.features) {
    const p = f.properties;
    assert.ok(p.diameter_m > 3, `tank ${p.id} has implausible diameter`);
    assert.ok(sites.has(p.site), `tank ${p.id} has unknown site '${p.site}'`);
    assert.equal(p.height_src, "assumed_api650_48ft",
      "every default height must carry its provenance flag — assumptions are labeled, never silent");
    assert.equal(p.measurable_10m, p.diameter_m >= 40, `tank ${p.id} measurable flag inconsistent`);
    const [lon, lat] = f.geometry.coordinates;
    assert.ok(lat > 35.85 && lat < 36.01 && lon > -96.85 && lon < -96.67, `tank ${p.id} outside the Cushing bbox`);
  }
  // capacity sanity (the workup's registration test): sum of pi*(d/2)^2*h
  // across the ring should land near EIA's ~76M bbl working capacity
  const m3 = reg.features.reduce((s: number, f: any) =>
    s + Math.PI * (f.properties.diameter_m / 2) ** 2 * f.properties.height_m, 0);
  const mbbl = (m3 * 6.2898) / 1e6;
  assert.ok(mbbl > 40 && mbbl < 120, `ring shell capacity ${mbbl.toFixed(0)}M bbl implausible vs EIA ~76M bbl working capacity`);
});
