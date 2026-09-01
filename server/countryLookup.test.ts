// countryLookup — point -> country reverse lookup over the vendored Natural
// Earth admin0 boundaries. Tests: known-city containment across Polygon and
// MultiPolygon geometries, open-ocean/no-match, the enclave (hole-less outer
// ring) resolution order, and the bbox-param/name helpers.
import { test } from "node:test";
import assert from "node:assert/strict";
import { countryOf, countryBboxParam, countryName } from "./countryLookup.ts";

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
