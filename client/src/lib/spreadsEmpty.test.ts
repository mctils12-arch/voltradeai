// Run: npx tsx --test client/src/lib/spreadsEmpty.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { spreadsEmptyMessage } from "./spreadsEmpty.ts";

const here = dirname(fileURLToPath(import.meta.url));

test("no live quotes: says why (market hours / thin options) and that nothing stale is ranked", () => {
  const m = spreadsEmptyMessage("aapl", false);
  assert.match(m.body, /AAPL/);
  assert.match(m.body, /market hours/);
  assert.match(m.body, /never from stale/);
});

test("live quotes but no buildable spread gets its own explanation", () => {
  const m = spreadsEmptyMessage("XYZ", true);
  assert.match(m.body, /has live option quotes/);
  assert.doesNotMatch(m.body, /market hours \(9:30/);
});

test("the Analyze page renders the empty state instead of hiding the section", () => {
  // regression: the section used to be wrapped in `top_spreads.length > 0 && (...)`
  // with no else-branch, so an empty result silently removed it
  const src = readFileSync(join(here, "..", "pages", "analyze.tsx"), "utf8");
  assert.match(src, /spreadsEmptyMessage\(/);
  assert.match(src, /data-testid="spreads-empty"/);
});
