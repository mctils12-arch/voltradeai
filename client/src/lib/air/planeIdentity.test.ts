import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { countryFromIcao24, countryFromRegistration, typeInfo } from "./planeIdentity";

// ── hex-allocation decode (ICAO Annex 10 blocks) ────────────────────────────

test("countryFromIcao24: majors decode from their allocation blocks", () => {
  assert.equal(countryFromIcao24("abe872"), "United States"); // the field aircraft (N8667D)
  assert.equal(countryFromIcao24("A00001"), "United States");
  assert.equal(countryFromIcao24("c01234"), "Canada");
  assert.equal(countryFromIcao24("400abc"), "United Kingdom");
  assert.equal(countryFromIcao24("3c6789"), "Germany");
  assert.equal(countryFromIcao24("781234"), "China");
  assert.equal(countryFromIcao24("7c0001"), "Australia");
  assert.equal(countryFromIcao24("840123"), "Japan");
  assert.equal(countryFromIcao24("e48abc"), "Brazil");
  assert.equal(countryFromIcao24("155555"), "Russia");
});

test("countryFromIcao24: outside catalogued blocks is an honest null, never a guess", () => {
  assert.equal(countryFromIcao24("900001"), null); // reserved block
  assert.equal(countryFromIcao24(""), null);
  assert.equal(countryFromIcao24("zzzzzz"), null);
  assert.equal(countryFromIcao24("1000000"), null); // > 24 bits
  assert.equal(countryFromIcao24(null), null);
});

// ── registration-prefix decode ──────────────────────────────────────────────

test("countryFromRegistration: nationality marks, longest prefix wins", () => {
  assert.equal(countryFromRegistration("N8667D"), "United States");
  assert.equal(countryFromRegistration("G-EZTL"), "United Kingdom");
  assert.equal(countryFromRegistration("D-AIMA"), "Germany");
  assert.equal(countryFromRegistration("C-FABC"), "Canada");
  assert.equal(countryFromRegistration("JA801A"), "Japan");
  assert.equal(countryFromRegistration("VH-OQA"), "Australia");
  assert.equal(countryFromRegistration("PR-GTA"), "Brazil");
  assert.equal(countryFromRegistration("A9C-KA"), "Bahrain"); // longest-match over A-prefixes
  assert.equal(countryFromRegistration("a6-edc"), "United Arab Emirates"); // case-insensitive
});

test("countryFromRegistration: unknown marks stay null", () => {
  assert.equal(countryFromRegistration("Z9-XYZ"), null);
  assert.equal(countryFromRegistration(""), null);
  assert.equal(countryFromRegistration(null), null);
});

// ── type designator → manufacturer/model ────────────────────────────────────

test("typeInfo: catalogued designators name manufacturer and model", () => {
  assert.deepEqual(typeInfo("B738"), { mfr: "Boeing", model: "737-800", label: "Boeing 737-800" });
  assert.deepEqual(typeInfo("A21N"), { mfr: "Airbus", model: "A321neo", label: "Airbus A321neo" });
  assert.deepEqual(typeInfo("FA8X"), { mfr: "Dassault", model: "Falcon 8X", label: "Dassault Falcon 8X" });
  assert.deepEqual(typeInfo("C172"), { mfr: "Cessna", model: "172 Skyhawk", label: "Cessna 172 Skyhawk" });
  assert.equal(typeInfo("b738").label, "Boeing 737-800"); // case-insensitive
});

test("typeInfo: uncatalogued codes show the raw code — never a fabricated name", () => {
  assert.deepEqual(typeInfo("ZZZZ"), { mfr: null, model: null, label: "ZZZZ" });
  assert.deepEqual(typeInfo(""), { mfr: null, model: null, label: "" });
  assert.deepEqual(typeInfo(null), { mfr: null, model: null, label: "" });
});

// ── source ratchets (identity repair 2026-08-08) ────────────────────────────
// `r` in the readsb schema is the REGISTRATION; it was mislabeled
// origin_country from 2026-07-03 and rendered as "Country" on the flight
// card. The mapping and both client read sites must stay on the honest name,
// and the archive line must keep carrying rg (schema v2).

test("mapPointAircraft maps r -> registration, never origin_country", async () => {
  const src = readFileSync(new URL("../../../../server/aircraftTiling.ts", import.meta.url), "utf-8");
  assert.match(src, /registration: a\.r \|\| ""/, "registration mapping lost");
  assert.doesNotMatch(src, /origin_country: a\.r/, "the r-as-country mislabel came back");
});

test("archive line carries rg (schema v2) and the manifest documents it", async () => {
  const arch = readFileSync(new URL("../../../../server/datacoreArchive.ts", import.meta.url), "utf-8");
  assert.match(arch, /rg: p\.registration \|\| undefined/, "registration dropped from the archive line");
  const manifest = JSON.parse(readFileSync(new URL("../../../../datacore/manifests/aircraft.json", import.meta.url), "utf-8"));
  assert.equal(manifest.schema_version, 2);
  assert.ok(manifest.field_map.rg, "manifest field_map must document rg");
});
