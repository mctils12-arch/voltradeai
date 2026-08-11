// cdcCancer.test.ts — NCI State Cancer Profiles county cancer-rate hazard
// layer (Location Context Engine hazard layer #5). Pure-function + geometry-
// join + injected-fetch coverage, same convention as pfas.test.ts/
// femaFlood.test.ts.
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  countyFromRecord, parseCountyRates, countyGeometry, joinGeometryAndRates,
  fccQueryUrl, parseFccResponse, fetchCountyAt, countyFipsAt, clearCountyCache,
  cancerRateFor, latestCancerRates,
} from "./cdcCancer";

// ── countyFromRecord / parseCountyRates ─────────────────────────────────

test("countyFromRecord: a normal record passes through with all fields", () => {
  const { county, issues } = countyFromRecord({
    fips: "01001", county: "Autauga County", state: "Alabama",
    incidence_rate: 459.8, incidence_ci: [437.4, 483.1], incidence_avg_annual_count: 330,
    incidence_trend: "stable", incidence_suppressed: false,
    mortality_rate: 153.8, mortality_ci: [141.2, 167.2], mortality_avg_annual_count: 114,
    mortality_trend: "falling", mortality_suppressed: false,
  });
  assert.deepEqual(issues, []);
  assert.equal(county!.fips, "01001");
  assert.equal(county!.incidence_rate, 459.8);
});

test("countyFromRecord: a suppressed county (null rates) is CLEAN, not quarantined — null is a legitimate value here", () => {
  const { county, issues } = countyFromRecord({
    fips: "48269", county: "King County", state: "Texas",
    incidence_rate: null, incidence_ci: null, incidence_avg_annual_count: null,
    incidence_trend: null, incidence_suppressed: true,
    mortality_rate: null, incidence_avg_annual_count_note: "3 or fewer",
  });
  assert.deepEqual(issues, []);
  assert.equal(county!.incidence_rate, null);
  assert.equal(county!.incidence_suppressed, true);
});

test("countyFromRecord: bad FIPS format is quarantined", () => {
  const { county, issues } = countyFromRecord({ fips: "9999", county: "Bad Row" });
  assert.equal(county, undefined);
  assert.ok(issues.some((i) => i.field === "fips"));
});

test("countyFromRecord: an out-of-bounds rate is quarantined (impossible-value guard, not a plausibility filter)", () => {
  const { county, issues } = countyFromRecord({ fips: "01001", county: "X", incidence_rate: 99999 });
  assert.equal(county, undefined);
  assert.ok(issues.some((i) => i.field === "incidence_rate"));
});

test("parseCountyRates: quarantines bad records, keeps clean ones, never throws on a mixed artifact", () => {
  const { counties, suspect } = parseCountyRates({
    counties: [
      { fips: "01001", county: "Good County", incidence_rate: 400 },
      { fips: "bad", county: "Bad County" },
    ],
  });
  assert.equal(counties.length, 1);
  assert.equal(suspect, 1);
});

// ── geometry join ────────────────────────────────────────────────────────

const FAKE_TOPO = {
  type: "Topology",
  objects: {
    counties: {
      type: "GeometryCollection",
      geometries: [
        { type: "Polygon", id: "01001", properties: { name: "Autauga" }, arcs: [[0]] },
        { type: "Polygon", id: "99999", properties: { name: "NoDataCounty" }, arcs: [[1]] },
      ],
    },
  },
  arcs: [[[0, 0], [1, 0], [0, 1], [0, 0]], [[2, 2], [3, 2], [2, 3], [2, 2]]],
};

test("countyGeometry: converts topojson counties object into a GeoJSON FeatureCollection keyed by FIPS id", () => {
  const geo = countyGeometry(FAKE_TOPO);
  assert.equal(geo.type, "FeatureCollection");
  assert.equal(geo.features.length, 2);
  assert.equal((geo.features[0] as any).id, "01001");
});

test("joinGeometryAndRates: a matched county carries its rate properties, has_data:true", () => {
  const geo = countyGeometry(FAKE_TOPO);
  const joined = joinGeometryAndRates(geo, [
    { fips: "01001", county: "Autauga County", state: "Alabama", incidence_rate: 459.8, incidence_ci: null,
      incidence_avg_annual_count: null, incidence_trend: "stable", incidence_suppressed: false,
      mortality_rate: 153.8, mortality_ci: null, mortality_avg_annual_count: null, mortality_trend: "falling", mortality_suppressed: false },
  ]);
  const f = joined.features.find((x: any) => x.properties.fips === "01001")!;
  assert.equal((f.properties as any).has_data, true);
  assert.equal((f.properties as any).incidence_rate, 459.8);
});

test("joinGeometryAndRates: an unmatched geometry is KEPT (never dropped — a hole would read as 'zero risk'), has_data:false, rates null", () => {
  const geo = countyGeometry(FAKE_TOPO);
  const joined = joinGeometryAndRates(geo, []);
  const f = joined.features.find((x: any) => x.properties.fips === "99999")!;
  assert.ok(f, "unmatched county geometry must still be present");
  assert.equal((f.properties as any).has_data, false);
  assert.equal((f.properties as any).incidence_rate, null);
  assert.equal((f.properties as any).county, "NoDataCounty"); // falls back to the topology's own name property
});

// ── FCC point-in-county lookup ──────────────────────────────────────────

test("fccQueryUrl: lat/lon pass through as query params", () => {
  const url = fccQueryUrl(29.76, -95.37);
  assert.ok(url.includes("lat=29.76"));
  assert.ok(url.includes("lon=-95.37"));
});

test("parseFccResponse: a real block result decodes to county FIPS/name/state", () => {
  const out = parseFccResponse({ results: [{ county_fips: "48201", county_name: "Harris County", state_code: "TX" }] });
  assert.equal(out.fips, "48201");
  assert.equal(out.county_name, "Harris County");
  assert.equal(out.ready, true);
});

test("parseFccResponse: zero results is an honest 'no coverage', not a failure", () => {
  const out = parseFccResponse({ results: [] });
  assert.equal(out.fips, null);
  assert.equal(out.ready, true);
});

test("parseFccResponse: a malformed county_fips is quarantined to no-coverage, never guessed", () => {
  const out = parseFccResponse({ results: [{ county_fips: "bad", county_name: "X" }] });
  assert.equal(out.fips, null);
  assert.equal(out.ready, true);
});

test("fetchCountyAt: a real response joins the rate table by the resolved FIPS", async () => {
  const mockFetch = async () => ({
    ok: true, status: 200,
    text: async () => JSON.stringify({ results: [{ county_fips: "01001", county_name: "Autauga County", state_code: "AL" }] }),
  });
  const out = await fetchCountyAt(32.5, -86.6, mockFetch as any);
  assert.equal(out.fips, "01001");
  // rate may be null in this test process (real artifact loaded at module
  // scope) or a real record — either way the field must exist and never throw
  assert.ok("rate" in out);
});

test("fetchCountyAt: HTTP error degrades to ready:false, never throws", async () => {
  const mockFetch = async () => ({ ok: false, status: 503, text: async () => "" });
  const out = await fetchCountyAt(32.5, -86.6, mockFetch as any);
  assert.equal(out.ready, false);
});

test("fetchCountyAt: invalid coordinates never reach the network", async () => {
  let called = false;
  const mockFetch = async () => { called = true; return { ok: true, status: 200, text: async () => "{}" }; };
  const out = await fetchCountyAt(999, 999, mockFetch as any);
  assert.equal(called, false);
  assert.equal(out.ready, false);
});

test("countyFipsAt: caches a resolved result — second call within TTL never hits fetch again", async () => {
  clearCountyCache();
  let calls = 0;
  const mockFetch = async () => { calls++; return { ok: true, status: 200, text: async () => JSON.stringify({ results: [{ county_fips: "01001", county_name: "Autauga County", state_code: "AL" }] }) }; };
  const a = await countyFipsAt(32.5, -86.6, mockFetch as any, 1000);
  const b = await countyFipsAt(32.5, -86.6, mockFetch as any, 2000);
  assert.equal(calls, 1);
  assert.deepEqual(a, b);
});

test("countyFipsAt: a failed lookup is NOT cached — the next call retries instead of freezing an outage", async () => {
  clearCountyCache();
  let calls = 0;
  const mockFetch = async () => { calls++; return { ok: false, status: 500, text: async () => "" }; };
  const a = await countyFipsAt(33.0, -87.0, mockFetch as any, 0);
  const b = await countyFipsAt(33.0, -87.0, mockFetch as any, 100);
  assert.equal(calls, 2);
  assert.equal(a.ready, false);
  assert.equal(b.ready, false);
});

// ── real committed artifact (module-scope load) ─────────────────────────

test("latestCancerRates: the real committed artifact loads, GATE 1 passed, geometry joined for a known county", () => {
  const { counties, geo, meta } = latestCancerRates();
  assert.ok(counties.length > 3000, "expected ~3143 counties from the real artifact");
  assert.equal(meta.gate1.passed, true, "GATE 1 (national row vs CDC's own published reference) must pass in the committed artifact");
  assert.ok(geo.features.length > 3000);
  assert.ok(/county/i.test(meta.caveat), "ecological-fallacy caveat must be present");
});

test("cancerRateFor: a known FIPS resolves to a real record from the committed artifact", () => {
  const rec = cancerRateFor("01001"); // Autauga County, AL
  assert.ok(rec, "Autauga County FIPS 01001 should be present in the real artifact");
  assert.equal(rec!.county, "Autauga County");
});

test("cancerRateFor: null/unknown fips never throws", () => {
  assert.equal(cancerRateFor(null), null);
  assert.equal(cancerRateFor("00000"), null); // national row is excluded from the county table by design
});
