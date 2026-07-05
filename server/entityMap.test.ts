// Everything Graph R5 build-plan step 1 (datacore/EVERYTHING_GRAPH.md) —
// datacore/entity_map.json structure + coverage-honesty regression tests.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.join(here, "..");
const mapPath = path.join(root, "datacore", "entity_map.json");
const map = JSON.parse(fs.readFileSync(mapPath, "utf8"));

const sites = JSON.parse(fs.readFileSync(path.join(root, "datacore", "sites", "strategic_sites.json"), "utf8"));
const plants = JSON.parse(fs.readFileSync(path.join(root, "datacore", "powerplants", "us_power_plants.json"), "utf8")).plants;

test("every entity has required fields and a valid confidence tier", () => {
  assert.ok(Array.isArray(map.entities) && map.entities.length > 0);
  const seen = new Set<string>();
  for (const e of map.entities) {
    assert.ok(typeof e.operator === "string" && e.operator.length > 0);
    assert.ok(!seen.has(e.operator), `duplicate operator key: ${e.operator}`);
    seen.add(e.operator);
    assert.ok(["high", "medium", "unmapped"].includes(e.confidence), `bad confidence: ${e.confidence}`);
    assert.ok(typeof e.note === "string" && e.note.length > 10, `missing provenance note for ${e.operator}`);
    if (e.confidence === "unmapped") {
      assert.equal(e.ticker, null, `unmapped entity must not carry a guessed ticker: ${e.operator}`);
    } else {
      assert.ok(typeof e.ticker === "string" && e.ticker.length > 0, `mapped entity missing ticker: ${e.operator}`);
      assert.ok(typeof e.parent === "string" && e.parent.length > 0, `mapped entity missing parent: ${e.operator}`);
    }
  }
});

test("coverage block matches the actual entity list (no stale bookkeeping)", () => {
  const mapped = map.entities.filter((e: any) => e.confidence !== "unmapped").length;
  assert.equal(map.coverage.total, map.entities.length);
  assert.equal(map.coverage.mapped, mapped);
  assert.equal(map.coverage.unmapped, map.entities.length - mapped);
});

test("coverage honesty: every current site operator + top-100 plant owner has an entry", () => {
  const keys = new Set(map.entities.map((e: any) => e.operator));
  for (const s of sites.sites) {
    if (s.operator) assert.ok(keys.has(s.operator), `strategic_sites.json operator not in entity_map: ${s.operator}`);
  }
  const top100 = [...plants].sort((a: any, b: any) => (b[1] || 0) - (a[1] || 0)).slice(0, 100);
  for (const p of top100) {
    const owner = p[3];
    if (owner) assert.ok(keys.has(owner), `top-100 plant owner not in entity_map: ${owner}`);
  }
});

test("attribution + reasoning-standard-9 honesty language present", () => {
  assert.ok(String(map._doc).includes("re-verify"), "doc must warn that corporate structures change");
  assert.ok(String(map._doc).toLowerCase().includes("no predictive claim"), "must be marked RAW, not SIGNAL");
});

test("no guessed tickers for government/municipal/federal operators", () => {
  const federalOrMuni = ["Tennessee Valley Authority", "U S Bureau of Reclamation", "New York Power Authority", "PANYNJ", "POLB"];
  for (const key of federalOrMuni) {
    const e = map.entities.find((x: any) => x.operator === key);
    assert.ok(e, `expected ${key} to be present`);
    assert.equal(e.confidence, "unmapped", `${key} should be an honest gap, not a guessed ticker`);
  }
});
