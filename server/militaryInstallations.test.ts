// Military installations — STATIC REFERENCE GEOGRAPHY (human-specced
// 2026-07-17). Pins the schema, the hard honesty rules (published
// coordinates only, installation-level, ODbL credit), the registry entry
// (default OFF, noCrossTies, provenance banner verbatim), and — the human's
// standing rule — that this layer is NEVER wired to live-tracking correlation.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const doc = JSON.parse(fs.readFileSync(path.join(here, "..", "datacore", "military_installations.json"), "utf8"));
const registry = JSON.parse(fs.readFileSync(path.join(here, "..", "datacore", "layers.json"), "utf8"));

const TYPE_ENUM = new Set(["air", "naval", "army", "joint", "logistics", "other"]);
const REQUIRED = ["name", "operator_nation", "branch", "type", "status", "geometry", "source", "source_url", "source_retrieved_date"];

test("every installation carries the full schema, a valid type, and PUBLISHED geometry (never inferred)", () => {
  assert.ok(Array.isArray(doc.installations) && doc.installations.length > 0);
  assert.equal(doc.count, doc.installations.length);
  for (const f of doc.installations) {
    for (const k of REQUIRED) assert.ok(k in f, `missing ${k} on "${f.name}"`);
    assert.ok(TYPE_ENUM.has(f.type), `bad type ${f.type} on "${f.name}"`);
    assert.ok(["active", "inactive"].includes(f.status), `bad status ${f.status}`);
    // never a feature without geometry — the drop-not-infer rule
    assert.ok(f.geometry && (f.geometry.type === "Polygon" || f.geometry.type === "MultiPolygon" || f.geometry.type === "Point"),
      `"${f.name}" has no valid published geometry`);
    // every entry is citable
    assert.ok(typeof f.source_url === "string" && /^https?:\/\//.test(f.source_url), `"${f.name}" missing citable source_url`);
  }
});

test("installation-level granularity — every feature is NAMED (no unnamed sub-facilities)", () => {
  for (const f of doc.installations) {
    assert.ok(typeof f.name === "string" && f.name.trim().length > 0, "a feature has no name");
  }
});

test("provenance: verbatim banner, ODbL attribution, no_cross_ties, per-source counts + retrieval date", () => {
  const p = doc.provenance;
  assert.equal(p.banner,
    "Officially published installation locations. Sources: US DoD open data, OpenStreetMap contributors, " +
    "and cited government publications. Reference geography only — current as of retrieval date; not operational information.");
  assert.ok(/OpenStreetMap contributors/.test(p.attribution), "ODbL credit required");
  assert.equal(p.no_cross_ties, true, "static reference geography — must be flagged no_cross_ties");
  assert.ok(p.sources && p.sources.osm && typeof p.sources.osm.count === "number", "per-source OSM count");
  assert.ok(/^\d{4}-\d{2}-\d{2}$/.test(p.retrieved_date), "retrieval date recorded");
});

test("registry entry: facilities group, live, DEFAULT OFF, noCrossTies, banner text, ODbL credit", () => {
  const l = registry.layers.find((x: any) => x.id === "military_installations");
  assert.ok(l, "military_installations layer missing from registry");
  assert.equal(l.kind, "raw");
  assert.equal(l.status, "live");
  assert.equal(l.group, "facilities");
  assert.equal(l.noCrossTies, true, "human's standing rule: never correlated with live tracking");
  assert.ok(/OpenStreetMap contributors/.test(l.source), "ODbL credit in source");
  assert.ok(/Reference geography only/.test(l.description), "provenance banner text in description");
  assert.ok(/not operational information/.test(l.description), "banner honesty rail present");
});

test("NO CROSS-TIES enforcement: the client never joins this layer to live tracking (human standing rule)", () => {
  const page = fs.readFileSync(path.join(here, "..", "client", "src", "pages", "datamap.tsx"), "utf8");
  // the military effect must not build a timeline/traffic-density block for its detail card
  const effectStart = page.indexOf("Military installations (RAW; STATIC REFERENCE");
  assert.ok(effectStart > -1, "military installations effect present");
  const effectEnd = page.indexOf("}, [enabled.military_installations", effectStart);
  const effect = page.slice(effectStart, effectEnd);
  assert.ok(!/timeline:/.test(effect), "the military card must NOT build a timeline (no live-tracking correlation)");
  assert.ok(!/trailKind|trailId/.test(effect), "the military card must NOT attach a live trail");
});

test("default-off wiring: military_installations is NOT in the default-on set", () => {
  const page = fs.readFileSync(path.join(here, "..", "client", "src", "pages", "datamap.tsx"), "utf8");
  // ALL_OFF is the base; DEFAULT_ON only flips a small named set on. Ensure our id
  // is not force-enabled anywhere.
  assert.ok(!/military_installations["']?\s*:\s*true/.test(page), "must not be defaulted on");
});
