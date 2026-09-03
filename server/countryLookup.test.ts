// countryLookup — point -> country reverse lookup over the vendored Natural
// Earth admin0 boundaries. Tests: known-city containment across Polygon and
// MultiPolygon geometries, open-ocean/no-match, the enclave (hole-less outer
// ring) resolution order, and the bbox-param/name helpers.
import { test } from "node:test";
import assert from "node:assert/strict";
import { countryOf, countryBboxParam, countryBboxParams, countryName } from "./countryLookup.ts";

test("countryOf: known cities resolve to their ISO3 code", () => {
  assert.equal(countryOf(55.7558, 37.6173), "RUS"); // Moscow
  assert.equal(countryOf(33.3152, 44.3661), "IRQ"); // Baghdad
  assert.equal(countryOf(35.6892, 51.389), "IRN"); // Tehran
  assert.equal(countryOf(6.5244, 3.3792), "NGA"); // Lagos
  assert.equal(countryOf(31.9686, -99.9018), "USA"); // central Texas
});

test("countryOf: a MultiPolygon country (archipelago/exclave-bearing) resolves", () => {
  // USA includes non-contiguous MultiPolygon parts in most 110m datasets;
  // the contiguous-US point above already exercises the geometry either
  // way. Indonesia is unambiguously MultiPolygon-shaped everywhere.
  assert.equal(countryOf(-6.2088, 106.8456), "IDN"); // Jakarta
});

test("countryOf: open ocean has no country match", () => {
  assert.equal(countryOf(0, -30), null); // mid-Atlantic
  assert.equal(countryOf(-60, 0), null); // Southern Ocean
});

test("countryOf: an enclave resolves to itself, not its surrounding country", () => {
  // Lesotho is fully enclosed by South Africa; the smallest-bbox-first scan
  // order in loadCountries() exists specifically for this case.
  assert.equal(countryOf(-29.6, 28.2), "LSO");
  // A South African point far from the enclave still resolves correctly.
  assert.equal(countryOf(-33.9249, 18.4241), "ZAF"); // Cape Town
});

test("countryBboxParam: known country returns the archive-probe bbox format", () => {
  const p = countryBboxParam("IRQ");
  assert.ok(p, "expected a bbox string for IRQ");
  const parts = p!.split(",").map(Number);
  assert.equal(parts.length, 4);
  const [lamin, lamax, lomin, lomax] = parts;
  assert.ok(lamin < lamax && lomin < lomax);
  // Baghdad must fall inside its own country's bbox.
  assert.ok(33.3152 >= lamin && 33.3152 <= lamax);
  assert.ok(44.3661 >= lomin && 44.3661 <= lomax);
});

test("countryBboxParam / countryName: unknown iso3 returns null", () => {
  assert.equal(countryBboxParam("ZZZ"), null);
  assert.equal(countryName("ZZZ"), null);
});

test("countryName: known iso3 returns the dataset's own display name", () => {
  assert.equal(countryName("nga"), "Nigeria"); // case-insensitive
  assert.equal(countryName("USA"), "United States of America");
});

test("countryBboxParams: a non-crossing country returns exactly one bbox, matching countryBboxParam", () => {
  const params = countryBboxParams("IRQ");
  assert.ok(params);
  assert.equal(params!.length, 1);
  assert.equal(params![0], countryBboxParam("IRQ"));
});

test("countryBboxParams: Russia (dateline-crossing) splits into two tight bboxes, not one global-spanning box", () => {
  const naive = countryBboxParam("RUS")!.split(",").map(Number);
  const [, , naiveLomin, naiveLomax] = naive;
  // The naive single bbox spans nearly the whole globe's longitude range —
  // this is the defect countryBboxParams() exists to avoid.
  assert.ok(naiveLomax - naiveLomin > 180, "naive RUS bbox should span >180deg of longitude");

  const params = countryBboxParams("RUS");
  assert.ok(params);
  assert.equal(params!.length, 2);
  for (const p of params!) {
    const [lamin, lamax, lomin, lomax] = p.split(",").map(Number);
    assert.ok(lamin < lamax);
    assert.ok(lomin < lomax);
    // Neither split piece should itself span the whole globe.
    assert.ok(lomax - lomin < 180, `split RUS bbox ${p} should not itself span >=180deg`);
  }
  // Moscow falls inside exactly one of the two pieces.
  const moscow = { lat: 55.7558, lon: 37.6173 };
  const hits = params!.filter((p) => {
    const [lamin, lamax, lomin, lomax] = p.split(",").map(Number);
    return moscow.lat >= lamin && moscow.lat <= lamax && moscow.lon >= lomin && moscow.lon <= lomax;
  });
  assert.equal(hits.length, 1);
  // The Chukotka exclave (near -180..-169.9 lon) falls inside the other piece.
  const chukotka = { lat: 66.0, lon: -175.0 };
  const chukotkaHits = params!.filter((p) => {
    const [lamin, lamax, lomin, lomax] = p.split(",").map(Number);
    return chukotka.lat >= lamin && chukotka.lat <= lamax && chukotka.lon >= lomin && chukotka.lon <= lomax;
  });
  assert.equal(chukotkaHits.length, 1);
});

test("countryBboxParams: unknown iso3 returns null", () => {
  assert.equal(countryBboxParams("ZZZ"), null);
});
