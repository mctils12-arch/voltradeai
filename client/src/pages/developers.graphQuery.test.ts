// /developers live-query widget for the Everything Graph endpoint (closes
// the "docs-explorer live-query widget for the graph endpoint" NEXT STEP
// filed 2026-07-23 alongside the /api/v1/graph PR). Source-inspection style,
// matching datamap.dossierRadius.test.ts's established pattern — a real
// headless component render isn't available in this test environment, so
// this pins the wiring contract: entity-param detection is data-driven (not
// hardcoded to one path), the "Run live example" button actually queries the
// typed entity/hops against the free preview mirror, and the curl sample
// mirrors the same query.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const src = fs.readFileSync(path.join(here, "developers.tsx"), "utf8");

test("hasEntityParam is detected off the live meta doc's own params string, not hardcoded to /api/v1/graph", () => {
  assert.match(src, /ENTITY_PARAM_RE\s*=\s*\/\\bentity=\//, "entity-param detection should be a regex over e.params");
  assert.match(src, /hasEntityParam\s*=\s*\(e: MetaEndpoint\)\s*=>\s*ENTITY_PARAM_RE\.test\(e\.params\)/);
  assert.doesNotMatch(src, /e\.path\s*===\s*["']\/api\/v1\/graph["']/, "must not hardcode the graph path — future param'd endpoints should get the same treatment");
});

test("graphPreviewUrl builds ?entity=&hops= against the given preview base, defaulting a blank entity to DEFAULT_ENTITY", () => {
  const fnStart = src.indexOf("function graphPreviewUrl");
  assert.ok(fnStart >= 0, "graphPreviewUrl not found");
  const block = src.slice(fnStart, fnStart + 400);
  assert.match(block, /q\.entity\.trim\(\)\s*\|\|\s*DEFAULT_ENTITY/, "blank entity input must fall back to DEFAULT_ENTITY, never send an empty entity=");
  assert.match(block, /encodeURIComponent\(entity\)/, "entity must be URI-encoded");
  assert.match(block, /encodeURIComponent\(hops\)/, "hops must be URI-encoded");
});

test("RATCHET: runEndpoint queries graphPreviewUrl(e.preview, ...) for entity-param endpoints instead of the bare preview URL", () => {
  const fnStart = src.indexOf("const runEndpoint = async");
  assert.ok(fnStart >= 0, "runEndpoint not found");
  const block = src.slice(fnStart, fnStart + 500);
  assert.match(block, /hasEntityParam\(e\)/, "runEndpoint must branch on hasEntityParam");
  assert.match(block, /graphPreviewUrl\(e\.preview,\s*graphQueries\[e\.path\]/, "queryable endpoints must build their fetch URL from the live graphQueries state");
});

test("RATCHET: the entity/hops inputs are rendered only for queryable endpoints and write into graphQueries keyed by path", () => {
  const idx = src.indexOf("vt-dev-query-row");
  assert.ok(idx >= 0, "vt-dev-query-row control not found");
  const block = src.slice(idx, idx + 1400);
  assert.match(block, /setGraphQueries\(\(qs\)\s*=>\s*\(\{\s*\.\.\.qs,\s*\[e\.path\]:\s*\{\s*entity:\s*ev\.target\.value/, "the entity input must update graphQueries[e.path].entity");
  assert.match(block, /setGraphQueries\(\(qs\)\s*=>\s*\(\{\s*\.\.\.qs,\s*\[e\.path\]:\s*\{\s*entity:\s*query\.entity,\s*hops:\s*ev\.target\.value/, "the hops select must update graphQueries[e.path].hops");
  const gateIdx = src.lastIndexOf("queryable &&", idx);
  assert.ok(gateIdx >= 0 && idx - gateIdx < 200, "the query row must be gated on `queryable` so non-param endpoints don't show it");
});

test("RATCHET: the curl example for a queryable endpoint includes the live entity/hops query string, not just the bare path", () => {
  const fnStart = src.indexOf("function curlFor");
  assert.ok(fnStart >= 0, "curlFor not found");
  const block = src.slice(fnStart, fnStart + 600);
  assert.match(block, /hasEntityParam\(e\)/);
  assert.match(block, /examplePath \+= `\?entity=\$\{encodeURIComponent\(q\.entity\.trim\(\) \|\| DEFAULT_ENTITY\)\}&hops=/);
});

test("freshness label falls back to built_at when generated_at is absent (the graph response's own field name)", () => {
  assert.match(src, /freshnessLabel\(run\.data\?\.generated_at\s*\?\?\s*run\.data\?\.built_at\)/);
});

test("RATCHET: freshnessLabel accepts a numeric epoch-ms timestamp (built_at's real shape), not just an ISO string", () => {
  const fnStart = src.indexOf("function freshnessLabel");
  assert.ok(fnStart >= 0, "freshnessLabel not found");
  const block = src.slice(fnStart, fnStart + 500);
  assert.match(block, /typeof iso === "number" \? iso : NaN/, "a numeric built_at must be used directly, not Date.parse'd into NaN");
});
