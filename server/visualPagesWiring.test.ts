// visualPagesWiring — [PRODUCT 2026-09-10] ratchet: a /data full-overlay
// view can never ship permanently unverified by the visual harness.
//
// The defect class: datamap.tsx wires each full overlay view
// (`const [xOpen, setXOpen] = useState(() => window.location.hash ===
// "#/data/y")`, per that file's own "Full X view — same overlay pattern"
// convention) independently of scripts/visual_check.mjs's PAGES map, which
// enumerates the routes the 390/768/1440 harness actually renders
// (PROMOTION RULE 6). Nothing tied the two together, so a page can ship,
// pass every other gate, and simply never be screenshotted or checked —
// found live this session: 8 routes (filings, earnings, short-volume, cot,
// graph, ats-summary, midas, dtcc-swaps) had sat unwired since as early as
// the first /data build, dtcc-swaps for three weeks. This test makes that
// unrepresentable, the same way layersWiring.test.ts's RATCHET does for
// LAYER_GROUP: every hash-route useState initializer in datamap.tsx must
// have a matching route (any key name) in visual_check.mjs's PAGES map.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));

test("RATCHET: every datamap.tsx hash-route full overlay view is registered in visual_check.mjs PAGES (no permanent visual-harness blind spot)", () => {
  const datamapSrc = fs.readFileSync(
    path.join(here, "..", "client", "src", "pages", "datamap.tsx"), "utf8");
  const hashRoutes = new Set(
    Array.from(datamapSrc.matchAll(/useState\(\(\) => window\.location\.hash === "(#\/[a-zA-Z0-9/_-]+)"\)/g))
      .map((m) => m[1]));
  assert.ok(hashRoutes.size >= 40, `datamap.tsx hash-route scrape looks broken (found ${hashRoutes.size} routes) — if the useState pattern changed, update this scrape AND the visual harness convention together`);

  const visualSrc = fs.readFileSync(path.join(here, "..", "scripts", "visual_check.mjs"), "utf8");
  const pagesMatch = visualSrc.match(/const PAGES = \{([\s\S]*?)\n\};/);
  assert.ok(pagesMatch, "PAGES map not found in scripts/visual_check.mjs — if it was renamed, update this scrape together with it");
  const pagesRouteStrings = Array.from(pagesMatch![1].matchAll(/route:\s*"([^"]+)"/g)).map((m) => m[1]);
  const registeredHashRoutes = new Set(
    pagesRouteStrings.filter((r) => r.includes("#")).map((r) => "#" + r.split("#", 2)[1]));
  assert.ok(registeredHashRoutes.size >= 30, `PAGES scrape looks broken (found ${registeredHashRoutes.size} hash routes)`);

  const unverified = Array.from(hashRoutes).filter((r) => !registeredHashRoutes.has(r)).sort();
  assert.deepEqual(unverified, [],
    `these datamap.tsx full overlay views have no scripts/visual_check.mjs PAGES entry, so the visual harness NEVER renders or checks them at any width: ${unverified.join(", ")} — ` +
    "add a PAGES entry (and a FIXTURES entry for its /api/data/* route if it needs realistic data to render) in the same PR that adds the hash-route useState");
});
