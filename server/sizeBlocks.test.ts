// REPAIR 2026-10-16: the silent sizing block becomes a visible SIZE-BLOCK audit row.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { formatSizeBlocks, SIZE_BLOCK_AUDIT_CAP } from "./sizeBlocks";

test("formats one row per block with ticker, bucket, reason and score", () => {
  const rows = formatSizeBlocks([{ ticker: "amd", strategy: "stocks", reason: "Negative expectancy for strategy 'stocks' (WR=28%); refusing to size", score: 74.25 }]);
  assert.deepEqual(rows, ["AMD [stocks]: Negative expectancy for strategy 'stocks' (WR=28%); refusing to size (score 74.3)"]);
});

test("caps the rows and says how many more were blocked", () => {
  const blocks = Array.from({ length: SIZE_BLOCK_AUDIT_CAP + 3 }, (_, i) => ({ ticker: `T${i}`, reason: "r" }));
  const rows = formatSizeBlocks(blocks);
  assert.equal(rows.length, SIZE_BLOCK_AUDIT_CAP + 1);
  assert.match(rows[rows.length - 1], /3 more candidates blocked/);
});

test("tolerates missing or malformed input", () => {
  assert.deepEqual(formatSizeBlocks(undefined), []);
  assert.deepEqual(formatSizeBlocks([null, 3, { ticker: "X" }]), ["X: blocked"]);
});

test("wiring pinned: tier2 audits SIZE-BLOCK rows from result.sizing_blocks right after the scan summary", () => {
  const here = path.dirname(fileURLToPath(import.meta.url));
  const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
  assert.ok(bot.includes('from "./sizeBlocks"'), "bot.ts must import the formatter");
  const scanned = bot.indexOf("trade candidates (via ${callResult.via})`);");
  const wiring = bot.indexOf('audit("SIZE-BLOCK", ');
  assert.ok(scanned > 0 && wiring > scanned, "SIZE-BLOCK audit must follow the Scanned summary");
  assert.ok(bot.includes("formatSizeBlocks(result.sizing_blocks)"), "must read result.sizing_blocks from the scan result");
});
