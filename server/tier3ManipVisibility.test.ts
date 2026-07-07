// R18 (KNOWN BROKEN #14, open_questions.md — PR #327 follow-up): the
// tier3Strategic manipulation-detection scan's catch block only
// console.error'd, leaving zero trail in the persisted audit log whenever
// the scan failed — the same stdout-corruption failure class the ML-retrain
// path already surfaces via TIER3-ML-ERROR. This pins that the catch block
// now routes through audit() the same way, so a future failure of any kind
// there does not go dark again.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");

function manipulationCatchBlock(): string {
  const scanStart = bot.indexOf("from manipulation_detect import scan_for_manipulation");
  assert.ok(scanStart > 0, "manipulation_detect scan call site not found in bot.ts");
  const catchStart = bot.indexOf("} catch (err: any) {", scanStart);
  assert.ok(catchStart > scanStart, "manipulation scan catch block not found");
  const catchEnd = bot.indexOf("\n    }", catchStart);
  assert.ok(catchEnd > catchStart, "manipulation scan catch block not closed as expected");
  return bot.slice(catchStart, catchEnd);
}

test("wiring pinned: manipulation-scan catch block records a persisted audit entry, not just console.error", () => {
  const block = manipulationCatchBlock();
  assert.ok(block.includes('audit("TIER3-MANIP-ERROR"'), "manipulation scan failures must be audited, mirroring TIER3-ML-ERROR");
  assert.ok(block.includes("console.error"), "console diagnostics stay too — audit() is additive, not a replacement");
});

test("wiring pinned: the audited detail carries cause, not a bare 'Command failed' string", () => {
  const block = manipulationCatchBlock();
  assert.ok(block.includes("err?.stderr"), "must prefer stderr (Python traceback) like the ML-retrain catch block");
  assert.ok(block.includes("err?.stdout"), "must fall back to stdout (structured JSON error payloads)");
  assert.ok(block.includes("err?.code") && block.includes("err?.signal"), "must surface exit code and kill signal for OOM/timeout diagnosis");
});

test("TIER3-MANIP-ERROR is a distinct audit action from TIER3-ML-ERROR", () => {
  // Guards against a future edit accidentally merging the two failure
  // classes back into one indistinguishable audit action.
  assert.ok(bot.includes('audit("TIER3-ML-ERROR"'), "ML-retrain audit action must still exist (regression baseline)");
  assert.ok(bot.includes('audit("TIER3-MANIP-ERROR"'));
});
