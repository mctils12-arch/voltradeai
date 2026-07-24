// SYMBOLS NOT DOTS ratchet (CLAUDE.md standing behavior, 2026-07-12) for
// the two hazard layers that predated the directive: superfund-pts and
// wv-pts rendered as bare "circle" layers until this fix (filed as
// REMAINING DEBT in research/location_context_engine.md, 2026-07-15).
// Source-inspection style, matching server/layersWiring.test.ts's pattern
// — a real headless-map render isn't available in this test environment
// (mapIcons.ts's SDF shapes need a DOM canvas), so this pins the wiring
// contract instead: neither layer may regress to a plain circle, both
// must reference a registered SDF icon, and both must carry a legend
// entry (the directive's other half — a symbol with no legend entry is
// still a directive violation).
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const datamapSrc = fs.readFileSync(path.join(here, "datamap.tsx"), "utf8");
const mapIconsSrc = fs.readFileSync(path.join(here, "..", "lib", "mapIcons.ts"), "utf8");

test("mapIcons.ts registers the vt-superfund and vt-outfall SDF shapes", () => {
  assert.match(mapIconsSrc, /"vt-superfund":\s*\(\)\s*=>\s*draw\(/,
    "vt-superfund shape missing from the mapIcons.ts shapes registry");
  assert.match(mapIconsSrc, /"vt-outfall":\s*\(\)\s*=>\s*draw\(/,
    "vt-outfall shape missing from the mapIcons.ts shapes registry");
});

test("RATCHET: superfund-pts is a symbol layer using vt-superfund, never a bare circle", () => {
  const block = datamapSrc.slice(datamapSrc.indexOf('id: "superfund-pts"'), datamapSrc.indexOf('id: "superfund-pts"') + 600);
  assert.match(block, /type:\s*"symbol"/, "superfund-pts must be a symbol layer (SYMBOLS NOT DOTS)");
  assert.doesNotMatch(block, /type:\s*"circle"/, "superfund-pts regressed to a bare circle layer");
  assert.match(block, /"icon-image":\s*"vt-superfund"/, "superfund-pts must draw the vt-superfund icon");
});

test("RATCHET: wv-pts is a symbol layer using vt-outfall, never a bare circle", () => {
  const block = datamapSrc.slice(datamapSrc.indexOf('id: "wv-pts"'), datamapSrc.indexOf('id: "wv-pts"') + 600);
  assert.match(block, /type:\s*"symbol"/, "wv-pts must be a symbol layer (SYMBOLS NOT DOTS)");
  assert.doesNotMatch(block, /type:\s*"circle"/, "wv-pts regressed to a bare circle layer");
  assert.match(block, /"icon-image":\s*"vt-outfall"/, "wv-pts must draw the vt-outfall icon");
});

test("RATCHET: both hazard layers have a legend section (symbol directive's other half)", () => {
  assert.match(datamapSrc, /icon="vt-superfund"/, "superfund has no LegendIcon entry");
  assert.match(datamapSrc, /icon="vt-outfall"/, "waterviolators has no LegendIcon entry");
  assert.match(datamapSrc, /enabled\.superfund\s*&&[\s\S]{0,80}vt-legend-sec-head/, "superfund legend section not gated on enabled.superfund");
  assert.match(datamapSrc, /enabled\.waterviolators\s*&&[\s\S]{0,80}vt-legend-sec-head/, "waterviolators legend section not gated on enabled.waterviolators");
});
