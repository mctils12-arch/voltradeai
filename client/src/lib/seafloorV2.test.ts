// GEBCO v2 seafloor / TID confidence — EARTH TWIN E2 v2. Pins the verbatim
// official decode table (never collapsed, never invented), the one-source-of-
// truth contract between the map expression and the legend, and the
// transparent-never-guessed handling of no-data and undocumented values.
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  TID_DECODE,
  TID_GRID_NODATA,
  TID_NO_DATA,
  GEBCO_ATTRIBUTION,
  GEBCO_NOT_FOR_NAVIGATION,
  TID_CONFIDENCE,
  tidGroupOf,
  terrariumDecode,
  tidCodeRuns,
  tidConfidenceColorRelief,
  tidConfidenceLegend,
  SEAFLOOR_V2_REGIONS,
} from "./seafloorV2.js";

/** Evaluate a MapLibre step expression at a scalar input. */
function evalStep(expr: unknown[], v: number): string {
  assert.deepEqual(expr.slice(0, 2), ["step", ["elevation"]]);
  let out = expr[2] as string;
  for (let i = 3; i < expr.length; i += 2) {
    if (v >= (expr[i] as number)) out = expr[i + 1] as string;
    else break;
  }
  return out;
}

test("decode table is EXACTLY the documented GEBCO_2026 set — no invented or collapsed codes", () => {
  const codes = [...TID_DECODE.keys()].sort((a, b) => a - b);
  const documented = [0, 10, 11, 12, 13, 14, 15, 16, 17, 40, 41, 42, 43, 44, 45, 46, 47, 48, 70, 71, 72];
  assert.deepEqual(codes, documented, "codes must mirror the official documentation, nothing else");
  for (const [c, info] of TID_DECODE) {
    const expected = c === 0 ? "land" : c <= 17 ? "direct" : c <= 48 ? "indirect" : "unknown";
    assert.equal(info.group, expected, `code ${c}: GEBCO's own grouping, never re-grouped`);
    assert.ok(info.definition.length > 3, `code ${c}: verbatim definition present`);
  }
  assert.equal(TID_DECODE.get(11)?.definition, "Multibeam - depth value collected by a multibeam echo-sounder");
  assert.equal(TID_GRID_NODATA, 127, "GEBCO's documented grid nodata value");
});

test("attribution and not-for-navigation honesty ride with the module", () => {
  assert.ok(GEBCO_ATTRIBUTION.includes("GEBCO Bathymetric Compilation Group"));
  assert.ok(GEBCO_ATTRIBUTION.includes("doi:10.5285/4f68d5c7-45eb-f999-e063-7086abc036fa"));
  assert.ok(/NOT for navigation/i.test(GEBCO_NOT_FOR_NAVIGATION));
});

test("tidGroupOf: documented codes decode, everything else is null (never guessed)", () => {
  assert.equal(tidGroupOf(0), "land");
  assert.equal(tidGroupOf(11), "direct");
  assert.equal(tidGroupOf(40), "indirect");
  assert.equal(tidGroupOf(71), "unknown");
  assert.equal(tidGroupOf(TID_NO_DATA), null);
  assert.equal(tidGroupOf(25), null, "gap values are not a class");
  assert.equal(tidGroupOf(99), null, "undocumented codes are not a class");
});

test("terrarium decode pins (shared encoding with the pipeline and v1 layer)", () => {
  assert.equal(terrariumDecode(128, 0, 0), 0, "sea level anchor");
  assert.equal(terrariumDecode(128, 11, 0), 11, "TID code 11 rides as elevation 11");
  assert.equal(terrariumDecode(127, 255, 0), TID_NO_DATA, "no-data marker decodes to -1");
  assert.equal(terrariumDecode(85, 77, 0), -10931, "hadal depth (real Challenger Deep cell value)");
});

test("code runs derive from the table: 0 | 10-17 | 40-48 | 70-72", () => {
  assert.deepEqual(tidCodeRuns(), [
    { start: 0, end: 0, group: "land" },
    { start: 10, end: 17, group: "direct" },
    { start: 40, end: 48, group: "indirect" },
    { start: 70, end: 72, group: "unknown" },
  ]);
});

test("confidence expression: every documented code gets its group color; gaps, no-data, and beyond-range are transparent", () => {
  const expr = tidConfidenceColorRelief();
  const transparent = "rgba(0,0,0,0)";
  for (const [c, info] of TID_DECODE) {
    assert.equal(evalStep(expr, c), TID_CONFIDENCE[info.group].color, `code ${c}`);
  }
  assert.equal(evalStep(expr, TID_NO_DATA), transparent, "out-of-coverage marker");
  assert.equal(evalStep(expr, -1000), transparent);
  assert.equal(evalStep(expr, 5), transparent, "gap 1..9");
  assert.equal(evalStep(expr, 25), transparent, "gap 18..39 (cross-group interpolation lands here, never a wrong class)");
  assert.equal(evalStep(expr, 60), transparent, "gap 49..69");
  assert.equal(evalStep(expr, 100), transparent, "beyond the last run");
  // land is transparent: the overlay never paints a reading over land
  assert.equal(evalStep(expr, 0), "rgba(0,0,0,0)");
  // stops strictly ascend (MapLibre step requirement)
  for (let i = 5; i < expr.length; i += 2) {
    assert.ok((expr[i] as number) > (expr[i - 2] as number), "step stops must strictly ascend");
  }
});

test("legend renders from the SAME table + colors as the map expression (one source of truth)", () => {
  const legend = tidConfidenceLegend();
  assert.deepEqual(legend.map((r) => r.group), ["direct", "indirect", "unknown"]);
  const expr = tidConfidenceColorRelief();
  for (const row of legend) {
    assert.equal(row.color, TID_CONFIDENCE[row.group].color, "legend color = shared color");
    assert.ok(row.label.length > 2 && row.meaning.length > 10, "label + meaning from the shared table");
    // the color the legend shows is exactly what the map paints for that group
    const someCode = [...TID_DECODE.entries()].find(([, i]) => i.group === row.group)![0];
    assert.equal(evalStep(expr, someCode), row.color, "legend/map parity");
  }
  assert.ok(!legend.some((r) => r.group === "land"), "land is transparent on the map, so it is absent from the legend");
});

test("region descriptors: committed mariana assets, honest coverage bbox, terrarium 256", () => {
  const m = SEAFLOOR_V2_REGIONS.find((r) => r.name === "mariana");
  assert.ok(m, "mariana region registered");
  assert.equal(m!.demUrl, "/tiles/seafloor_gebco_mariana.pmtiles");
  assert.equal(m!.tidUrl, "/tiles/seafloor_tid_mariana.pmtiles");
  assert.equal(m!.provenanceUrl, "/tiles/seafloor_gebco_mariana.provenance.json");
  assert.equal(m!.tileSize, 256);
  assert.equal(m!.encoding, "terrarium");
  assert.equal(m!.minzoom, 0);
  assert.equal(m!.maxzoom, 9);
  const [w, s, e, n] = m!.bbox;
  assert.ok(w === 138 && s === 7 && Math.round(e) === 146 && Math.round(n) === 15,
    "bbox states the TRUE fetched extent — regional subset, not global");
});
