// KNOWN BROKEN #17 (open_questions.md, found 2026-07-09): the TIER3-ML-ERROR
// audit line collapsed to the content-free "ML retrain failed: failed"
// whenever ml_model_v2.py's _train_model_impl returned one of its plain
// {"status": "failed", "reason": "..."} early-return shapes (e.g. "Could not
// fetch training bars", "No training data", "Model training failed") —
// those never carry error_location/traceback_tail, which only come from
// train_model()'s outer exception wrapper. bot.ts read error_location and
// traceback_tail but never trainResult.reason, silently dropping the one
// field that explained the failure. This pins the fix: the audited message
// must surface `reason` when present, and must not regress the existing
// error_location/traceback_tail handling for the other failure shape.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");

function mlRetrainErrorBlock(): string {
  const anchorIdx = bot.indexOf('_statusStr.startsWith("error:") || _statusStr.startsWith("failed")');
  assert.ok(anchorIdx > 0, "ML retrain error-status branch anchor not found in bot.ts");
  const blockStart = bot.lastIndexOf("const _statusStr", anchorIdx);
  const blockEnd = bot.indexOf("} else {", anchorIdx);
  assert.ok(blockStart > 0 && blockEnd > blockStart, "ML retrain error-status block not found");
  return bot.slice(blockStart, blockEnd);
}

test("wiring pinned: TIER3-ML-ERROR reads trainResult.reason, not just error_location/traceback_tail", () => {
  const block = mlRetrainErrorBlock();
  assert.ok(block.includes("trainResult.reason"), "must read the reason field ml_model_v2.py's plain-failed shape carries");
  assert.ok(block.includes("trainResult.error_location"), "must still read error_location (regression baseline, other failure shape)");
  assert.ok(block.includes("trainResult.traceback_tail"), "must still read traceback_tail (regression baseline, other failure shape)");
});

test("wiring pinned: the audited message template includes the reason variable before location/traceback", () => {
  const block = mlRetrainErrorBlock();
  const auditCallIdx = block.indexOf('audit("TIER3-ML-ERROR"');
  assert.ok(auditCallIdx > 0, "TIER3-ML-ERROR audit call not found in block");
  const auditCall = block.slice(auditCallIdx);
  assert.ok(auditCall.includes("${_reason}"), "audited template must interpolate the reason");
  assert.ok(
    auditCall.indexOf("${_reason}") < auditCall.indexOf("${_loc}"),
    "reason should render before location/traceback so the real cause reads first",
  );
});

test("regression: a bare {status:'failed'} with no reason still renders without throwing (no reason field present)", () => {
  // Static guard: _reason must be built from a truthy-check (trainResult.reason ? ... : "")
  // so the template never renders "undefined" when reason is absent.
  const block = mlRetrainErrorBlock();
  const reasonLine = block.split("\n").find((l) => l.includes("const _reason"));
  assert.ok(reasonLine, "_reason assignment line not found");
  assert.ok(reasonLine!.includes("?") && reasonLine!.includes(':'), "_reason must be a guarded ternary, not a raw interpolation");
});
