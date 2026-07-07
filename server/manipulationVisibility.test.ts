// KNOWN BROKEN #14 (2026-07-07, PR #327) — the manipulation-detect Tier-3
// scan's catch block only console.error'd, so a scan failure (same
// stdout-corruption class the ML-retrain fix addressed in v1.0.187) left
// zero trail in the persisted audit log. This is the regression test that
// would have caught the gap: it pins that the catch block now routes
// through audit(), mirroring the ML-retrain catch block immediately above
// it in tier3Strategic().
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));

function readBot(): string {
  return fs.readFileSync(path.join(here, "bot.ts"), "utf8");
}

test("manipulation-detect scan failure is audited, not just console.error'd", () => {
  const bot = readBot();
  const callStart = bot.indexOf("from manipulation_detect import scan_for_manipulation");
  assert.ok(callStart > 0, "manipulation_detect call site not found — code moved, re-check this test");
  // Slice from the call site to the next top-level "// 3." step comment
  // (the diagnostics step that follows manipulation detection in
  // tier3Strategic) so we only inspect this call's try/catch.
  const nextStepStart = bot.indexOf("// 3. Run full diagnostics", callStart);
  assert.ok(nextStepStart > callStart, "next tier3Strategic step not found — code moved, re-check this test");
  const block = bot.slice(callStart, nextStepStart);

  const catchStart = block.indexOf("} catch (err: any) {");
  assert.ok(catchStart > 0, "manipulation scan catch block not found");
  const catchBlock = block.slice(catchStart);

  assert.ok(catchBlock.includes('audit("TIER3-MANIP-ERROR"'), "catch block must audit() the failure, not just console.error it");
  assert.ok(catchBlock.includes("console.error"), "console.error should stay too, for local/dev visibility");
});

test("TIER3-MANIP-ERROR is a distinct audit type from TIER3-ML-ERROR", () => {
  const bot = readBot();
  assert.ok(bot.includes('"TIER3-MANIP-ERROR"'), "new audit type missing");
  assert.ok(bot.includes('"TIER3-ML-ERROR"'), "sibling ML-retrain audit type should still exist (sanity check for this test file)");
});
