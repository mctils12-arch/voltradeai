// jodiOil.ts — pure-function view over the static JODI archive. Mirrors the
// synthetic-fixture style of test_jodi_oil.py / test_jodi_eia_reconcile.py
// (no network, no live archive dependency for the logic tests) plus a
// real-file schema check (the militaryInstallations.test.ts precedent).
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { jodiOilStocksView, JODI_AREA_NAMES, JODI_PRODUCT, JODI_GATE_NOTE } from "./jodiOil";

const here = path.dirname(fileURLToPath(import.meta.url));

function fixture(overrides: Partial<Parameters<typeof jodiOilStocksView>[0]> = {}) {
  return {
    source: "JODI Oil World Primary database",
    attribution: "JODI Oil World Database (Joint Organisations Data Initiative)",
    latest_period: "2026-04",
    series_count: 4,
    series: {
      "US|TOTCRUDE": { points: [["2026-02", 1000, "3"], ["2026-03", 1050, "3"]], n: 2, first: "2026-02", last: "2026-03" },
      "SA|TOTCRUDE": { points: [["2026-03", 5000, "3"]], n: 1, first: "2026-03", last: "2026-03" },
      "US|CRUDEOIL": { points: [["2026-03", 400, "3"]], n: 1, first: "2026-03", last: "2026-03" }, // wrong product — excluded
      "ZZ|TOTCRUDE": { points: [], n: 0, first: "", last: "" }, // empty series — excluded
      ...overrides,
    },
  } as any;
}

test("only TOTCRUDE series are surfaced; other products and empty series are dropped", () => {
  const v = jodiOilStocksView(fixture());
  const areas = v.rows.map((r) => r.area);
  assert.deepEqual(areas.sort(), ["SA", "US"]);
});

test("rows sort by latest level descending", () => {
  const v = jodiOilStocksView(fixture());
  assert.equal(v.rows[0].area, "SA"); // 5000 > 1050
  assert.equal(v.rows[1].area, "US");
});

test("delta is computed against the series' own prior point, not a fixed calendar offset", () => {
  const v = jodiOilStocksView(fixture());
  const us = v.rows.find((r) => r.area === "US")!;
  assert.equal(us.period, "2026-03");
  assert.equal(us.levelKbbl, 1050);
  assert.equal(us.priorPeriod, "2026-02");
  assert.equal(us.priorLevelKbbl, 1000);
  assert.equal(us.deltaKbbl, 50);
  assert.equal(us.deltaPct, 5);

  const sa = v.rows.find((r) => r.area === "SA")!;
  assert.equal(sa.priorPeriod, null, "single-point series has no prior");
  assert.equal(sa.deltaKbbl, null);
});

test("never zero-fills a missing prior point — a single-point series reports null deltas, not 0", () => {
  const v = jodiOilStocksView(fixture());
  const sa = v.rows.find((r) => r.area === "SA")!;
  assert.equal(sa.deltaKbbl, null);
  assert.equal(sa.deltaPct, null);
});

test("area names resolve from JODI_AREA_NAMES; falls back to the raw code for an unmapped area", () => {
  const v = jodiOilStocksView(fixture({ "XX|TOTCRUDE": { points: [["2026-01", 1, "3"]], n: 1, first: "2026-01", last: "2026-01" } }));
  const us = v.rows.find((r) => r.area === "US")!;
  assert.equal(us.name, "United States");
  const xx = v.rows.find((r) => r.area === "XX")!;
  assert.equal(xx.name, "XX", "unmapped code falls back to the code itself, never a fabricated name");
});

test("honesty envelope: kind raw, predictive false, gate-2-kill note present verbatim", () => {
  const v = jodiOilStocksView(fixture());
  assert.equal(v.kind, "raw");
  assert.equal(v.predictive, false);
  assert.equal(v.product, JODI_PRODUCT);
  assert.equal(v.note, JODI_GATE_NOTE);
  assert.match(v.note, /GATE 2 \(signal\) KILLED/, "must not omit the kill status");
  assert.match(v.note, /no predictive claim/i);
});

test("archive-level fields pass through unmodified", () => {
  const v = jodiOilStocksView(fixture());
  assert.equal(v.archiveLatestPeriod, "2026-04");
  assert.equal(v.seriesCount, 4);
  assert.equal(v.countriesReporting, 2);
  assert.equal(v.source, "JODI Oil World Primary database");
  assert.equal(v.attribution, "JODI Oil World Database (Joint Organisations Data Initiative)");
});

test("JODI_AREA_NAMES has no duplicate names across distinct codes (a real name-collision would be a data bug)", () => {
  const names = Object.values(JODI_AREA_NAMES);
  assert.equal(new Set(names).size, names.length);
});

test("real archive file: every REF_AREA code JODI ships resolves to a mapped name (no silent code fallback in production)", () => {
  const doc = JSON.parse(fs.readFileSync(path.join(here, "..", "datacore", "jodi", "primary_stocks.json"), "utf8"));
  const areas = new Set(Object.keys(doc.series).map((k: string) => k.split("|")[0]));
  const missing = Array.from(areas).filter((a) => !(a in JODI_AREA_NAMES));
  assert.deepEqual(missing, [], `unmapped area code(s) in the real archive: ${missing.join(", ")}`);
});

test("real archive file: jodiOilStocksView runs against the live import without throwing and returns a non-trivial view", () => {
  const v = jodiOilStocksView();
  assert.equal(v.kind, "raw");
  assert.ok(v.rows.length > 10, "expected many TOTCRUDE-reporting countries in the real archive");
  assert.ok(v.rows.every((r) => typeof r.period === "string" && /^\d{4}-\d{2}$/.test(r.period)));
});
