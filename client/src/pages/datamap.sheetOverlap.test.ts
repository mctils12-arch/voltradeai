// Standing UI Law (no popup covers another element), 2026-09-30: on phones
// the floating Legend (z 18) and the north-lock FAB sat ON TOP of the detail
// bottom sheet (z 11), covering its chips and body. They must step aside
// while a (non-minimized) sheet is up. Probe evidence: 390px legend/FAB
// visible -> hidden (collapsed + expanded sheet) -> visible after close;
// 768/1440 unaffected with 0px card/legend overlap.
// Run: npx tsx --test client/src/pages/datamap.sheetOverlap.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const css = readFileSync(join(here, "..", "index.css"), "utf8");

function phoneSheetBlock(): string {
  // the phone media block that turns .vt-site-card into a bottom sheet
  const start = css.indexOf(".vt-card-handle { display: none; }");
  assert.ok(start >= 0, "bottom-sheet section not found");
  const open = css.indexOf("@media (max-width: 767px) {", start);
  assert.ok(open >= 0, "phone media block not found");
  let depth = 0;
  for (let i = css.indexOf("{", open); i < css.length; i++) {
    if (css[i] === "{") depth++;
    else if (css[i] === "}" && --depth === 0) return css.slice(open, i + 1);
  }
  throw new Error("unterminated media block");
}

test("phone: legend and north-lock FAB hide while a detail sheet is up", () => {
  const block = phoneSheetBlock();
  assert.match(block, /:has\(\.vt-site-card:not\(\.vt-site-card-min\)\)\s+\.vt-legend-float/);
  assert.match(block, /:has\(\.vt-site-card:not\(\.vt-site-card-min\)\)\s+\.vt-nav-fab\s*\{\s*display:\s*none/);
});

test("the stacking that caused the bug is still what the rule guards against", () => {
  // if either z-index changes so the sheet sits above them, this rule can be revisited
  assert.match(css, /\.vt-legend-float \{[^}]*z-index: 18/);
  assert.match(css, /\.vt-site-card \{[^}]*z-index: 11/);
});
