// Submarine cables — STATIC REFERENCE GEOGRAPHY (filed 2026-08-11,
// research/open_questions.md "Bilawal-derived build candidates" item 4).
// Pins the schema, the honesty rules (OSM-tag-only classification, disclosed
// simplification, ODbL credit, the documented rejection of NOAA
// MarineCadastre), and the registry entry — same pattern as
// militaryInstallations.test.ts.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const doc = JSON.parse(fs.readFileSync(path.join(here, "..", "datacore", "submarine_cables.json"), "utf8"));
const registry = JSON.parse(fs.readFileSync(path.join(here, "..", "datacore", "layers.json"), "utf8"));

const CATEGORY_ENUM = new Set(["telecom", "power", "mixed", "other", "unclassified"]);

test("every cable carries the full schema, a valid category, and a LineString geometry", () => {
  assert.ok(Array.isArray(doc.cables) && doc.cables.length > 0);
  assert.equal(doc.count, doc.cables.length);
  for (const f of doc.cables) {
    for (const k of ["id", "category", "disused", "length_km", "geometry", "source_url"]) {
      assert.ok(k in f, `missing ${k} on cable ${f.id}`);
    }
    assert.ok(CATEGORY_ENUM.has(f.category), `bad category ${f.category} on cable ${f.id}`);
    assert.equal(typeof f.disused, "boolean");
    assert.equal(typeof f.length_km, "number");
    assert.ok(f.length_km >= 0, `negative length on cable ${f.id}`);
    assert.equal(f.geometry?.type, "LineString", `cable ${f.id} must be a LineString`);
    assert.ok(Array.isArray(f.geometry.coordinates) && f.geometry.coordinates.length >= 2,
      `cable ${f.id} geometry needs at least 2 points`);
    assert.ok(typeof f.source_url === "string" && /^https:\/\/www\.openstreetmap\.org\/way\//.test(f.source_url),
      `cable ${f.id} missing citable OSM source_url`);
  }
});

test("provenance: ODbL attribution, corrected total/coverage figures, NOAA MarineCadastre documented as rejected", () => {
  const p = doc.provenance;
  assert.ok(/OpenStreetMap contributors/.test(p.attribution), "ODbL credit required");
  assert.ok(/ODbL/.test(p.license));
  assert.equal(typeof p.total_length_km, "number");
  assert.ok(p.total_length_km > 0, "corrected total length must be populated, never hardcoded to the old undercount");
  assert.equal(typeof p.telegeography_benchmark_km, "number");
  assert.equal(typeof p.coverage_pct_of_telegeography_benchmark, "number");
  assert.ok(/^\d{4}-\d{2}-\d{2}$/.test(p.retrieved_date), "retrieval date recorded");
  assert.ok(p.rejected_source && /NOAA MarineCadastre/.test(p.rejected_source.name),
    "the NOAA MarineCadastre rejection must stay documented in the artifact, not just in research notes");
  assert.ok(/NASCA/.test(p.rejected_source.reason), "rejection reason must name the real blocker (NASCA lineage), not a vague note");
  // honesty on any Overpass regions that never came back
  assert.equal(typeof p.fetch_gaps?.failed_leaf_regions, "number");
});

test("registry entry: facilities group, raw kind, DEFAULT OFF (field:false), ODbL credit, Europe/NE-Atlantic coverage caveat", () => {
  const l = registry.layers.find((x: any) => x.id === "submarine_cables");
  assert.ok(l, "submarine_cables layer missing from registry");
  assert.equal(l.kind, "raw");
  assert.equal(l.status, "live");
  assert.equal(l.group, "facilities");
  assert.equal(l.field, false, "heavy static layer must default off");
  assert.ok(/OpenStreetMap contributors/.test(l.source), "ODbL credit in source");
  assert.ok(/NE Atlantic/.test(l.description), "coverage-concentration caveat must be disclosed, not just the raw geometry");
});

test("default-off wiring: submarine_cables is not force-enabled anywhere in the client", () => {
  const page = fs.readFileSync(path.join(here, "..", "client", "src", "pages", "datamap.tsx"), "utf8");
  assert.ok(!/submarine_cables["']?\s*:\s*true/.test(page), "must not be defaulted on");
  assert.ok(page.includes('"/api/data/submarine_cables"'), "client must fetch the registered route");
});
