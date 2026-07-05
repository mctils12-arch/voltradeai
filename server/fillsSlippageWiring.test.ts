// Regression test for the FILLS_PATH measurement bug (2026-07-05,
// [RULE-REVIEW]). storage_config.FILLS_PATH (voltrade_fills.json) has had
// no writer since the legacy ml_model.py.track_fill() was retired —
// ml_model_v2.track_fill() (the live one) writes entry-fill
// expected_price/fill_price/slippage_pct straight into the trade_feedback
// store instead. Every route that still read FILLS_PATH for slippage/fill
// counts (performance dashboard, ml-status, diag ml probe) silently saw an
// empty list forever, zeroing "realistic P&L" / slippage-gap / fill-count
// numbers regardless of real trading activity. Fix: derive those stats
// from trade_feedback via ml_model_v2.fills_slippage_stats(). This test
// pins the wiring so a future edit can't reintroduce a FILLS_PATH read on
// these routes without failing CI.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");

test("performance/ml-status/diag routes derive slippage from trade_feedback, not the dead FILLS_PATH", () => {
  assert.ok(
    !bot.includes("FILLS_PATH"),
    "bot.ts must not read storage_config.FILLS_PATH anywhere — nothing has written that file " +
      "since ml_model.py's track_fill was retired; ml_model_v2.fills_slippage_stats(feedback) is the live read path"
  );
  const occurrences = bot.split("fills_slippage_stats").length - 1;
  assert.ok(
    occurrences >= 3,
    `expected fills_slippage_stats to be wired into /api/bot/performance, /api/bot/ml-status, and /api/diag/ml probe (found ${occurrences} references)`
  );
});

test("ml_model_v2.py exports fills_slippage_stats with the expected shape", () => {
  const mlModel = fs.readFileSync(path.join(here, "..", "ml_model_v2.py"), "utf8");
  assert.ok(mlModel.includes("def fills_slippage_stats"), "fills_slippage_stats function missing from ml_model_v2.py");
  assert.ok(mlModel.includes("'count'") || mlModel.includes('"count"'), "fills_slippage_stats must return a count field");
});
