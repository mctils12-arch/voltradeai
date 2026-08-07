// REPAIR 2026-08-07 (scheduled-routine session, KNOWN BROKEN #12b): the
// `trackClosedTrades()` feedback-write block (the OLD path that raced
// track_fill for writing ML training records) was permanently dead code,
// confirmed two ways this session. Static: `tradeResults` has exactly one
// write site (the `.unshift()` a few lines above this block), and it
// hardcodes `entryFeatures: null` unconditionally — the block's own filter
// (`t.entryFeatures != null`) can therefore never pass a single record, by
// construction, independent of live data. Live: `/api/diag/ml` now shows
// `live_outcome_breakdown` with real win/open/loss buckets (2/3/4) tagged
// `session:"regular"` — proof the OTHER writer (track_fill via
// entryFill.ts/morningFillPayload/regular-hours fillPayload) is the one
// actually producing labeled feedback, exactly the D2-verification KNOWN
// BROKEN #12b was gated on ("once a WS exit records a real outcome via
// track_fill, the block is redundant and should be REMOVED per the dead
// code policy"). Removed the block entirely rather than leave a permanently
// unreachable filter as misleading documentation (STALENESS AUDIT / DEAD
// CODE POLICY). This test pins the removal so a future edit can't
// reintroduce the same dead branch (e.g. by copy-pasting this block back in
// while "fixing" something else) without the source scan below noticing.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");

function trackClosedTradesBody(): string {
  const start = bot.indexOf("async function trackClosedTrades()");
  assert.ok(start > 0, "trackClosedTrades() not found in bot.ts");
  const end = bot.indexOf("\nlet lastWeightAdjustLog", start);
  assert.ok(end > start, "trackClosedTrades() did not end where expected");
  return bot.slice(start, end);
}

test("trackClosedTrades no longer contains the dead entryFeatures-gated feedback write", () => {
  const body = trackClosedTradesBody();
  assert.ok(
    !body.includes("entryFeatures != null"),
    "the permanently-false filter (entryFeatures is hardcoded null upstream) must stay removed",
  );
  assert.ok(
    !body.includes("TRADE_FEEDBACK_PATH"),
    "trackClosedTrades must not write TRADE_FEEDBACK_PATH itself — that's track_fill's job",
  );
});

test("trackClosedTrades still adjusts strategy weights (only the dead feedback write was removed)", () => {
  const body = trackClosedTradesBody();
  assert.ok(body.includes("adjustStrategyWeights();"), "the live strategy-weight adjustment must be untouched");
});

test("tradeResults still has exactly one write site and it still hardcodes entryFeatures null", () => {
  // Guards the PREMISE of the removal above: if a second write site is ever
  // added, or the hardcoded null is ever lifted, the dead-code reasoning no
  // longer holds and this repair's rationale needs re-examining before any
  // future session trusts this test file's own header comment.
  const writeSites = bot.match(/tradeResults\.(push|unshift)\(/g) || [];
  assert.equal(writeSites.length, 1, "expected exactly one tradeResults write site");
  const start = bot.indexOf("tradeResults.unshift({");
  const end = bot.indexOf("});", start);
  const pushBlock = bot.slice(start, end);
  assert.ok(pushBlock.includes("entryFeatures: null"), "the single write site must still hardcode entryFeatures: null");
});
