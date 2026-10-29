// 2026-10-27 (human Phase 4 "Fix the model") — wiring pins for the v3
// CANDIDATE model's nightly job (ml_model_v3.run_job) and its diag probe.
// The candidate is SHADOW ONLY. bot.ts can't be imported, so these read the
// source; the kill-cap test runs server/pythonExec.ts's planPythonExec (the
// lane/kill decision bot.ts's execPythonSerialized uses) on the job's real
// command, because a
// <=150-char command without a heavy import is "trivial": no mutex and a 10s
// SIGKILL instead of 90s.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { isTrivialPython, planPythonExec } from "./pythonExec";
import { DIAG_PROBES } from "./diag";

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
const v3py = fs.readFileSync(path.join(here, "..", "ml_model_v3.py"), "utf8");

function matchBrace(src: string, open: number): number {
  assert.equal(src[open], "{");
  let depth = 0;
  for (let i = open; i < src.length; i++) {
    const c = src[i];
    if (c === '"' || c === "'" || c === "`") {
      for (i++; i < src.length && src[i] !== c; i++) if (src[i] === "\\") i++;
      continue;
    }
    if (c === "{") depth++;
    else if (c === "}" && --depth === 0) return i + 1;
  }
  throw new Error("unbalanced braces");
}

function functionSource(signature: string): string {
  const start = bot.indexOf(signature);
  assert.ok(start >= 0, `${signature} not found in bot.ts`);
  return bot.slice(start, matchBrace(bot, bot.indexOf("{", start + signature.length)));
}

function jobCall(): { cmd: string; timeout: number } {
  const block = functionSource("async function checkMlV3CandidateJob()");
  const m = block.match(/execPythonSerialized\(\s*`([^`]*)`,\s*\{\s*timeout:\s*(\d+)\s*\}\s*\)/);
  assert.ok(m, "the job must call execPythonSerialized(`<literal command>`, { timeout: N })");
  assert.ok(!m![1].includes("${"), "the command must be a plain literal");
  return { cmd: m![1], timeout: Number(m![2]) };
}

test("v3 job: real kill cap is 90s (mutex held), and the Python budget fits under it", () => {
  // Since the 2026-10-27 timeouts repair the lane/kill decision lives in
  // server/pythonExec.ts (planPythonExec), which bot.ts's execPythonSerialized
  // calls — run it on the job's real command and requested timeout.
  const { cmd, timeout } = jobCall();
  assert.equal(isTrivialPython(cmd, timeout), false, "a trivial command is SIGKILLed at 10s");
  const plan = planPythonExec(cmd, timeout);
  assert.equal(plan.lane, "mutex");
  assert.equal(plan.timeoutMs, 90000);
  assert.ok(bot.includes("planPythonExec("), "bot.ts execPythonSerialized must route through planPythonExec");
  const budget = Number((cmd.match(/time_budget_s=(\d+)/) || [])[1]);
  assert.ok(budget > 0 && budget + 20 <= 90, `time_budget_s=${budget} must leave >= 20s for imports + final save`);
  assert.ok(cmd.includes("from ml_model_v3 import run_job"));
});

test("v3 job: unconditional 5-minute interval, bounded attempts, audited both ways", () => {
  const call = bot.indexOf("checkMlV3CandidateJob().catch(");
  assert.ok(call > 0);
  const line = bot.slice(bot.lastIndexOf("\n", call), bot.indexOf("\n", call));
  assert.ok(line.includes("setInterval") && line.includes("5 * 60 * 1000"));
  const block = functionSource("async function checkMlV3CandidateJob()");
  assert.ok(block.includes("ML_V3_MAX_ATTEMPTS_PER_NIGHT"));
  assert.ok(block.includes('"ML-V3-JOB"') && block.includes('"ML-V3-JOB-ERROR"'));
});

test("v3 candidate stays out of the live blend", () => {
  // the job never writes v2's artifact, and deep_score only LOGS v3 keys
  assert.ok(!/^\s*(import ml_model_v2|from ml_model_v2 import)/m.test(v3py), "v3 must not touch v2's module/artifact");
  assert.ok(v3py.includes('CANDIDATE_MODEL_PATH = os.path.join(V3_DIR, "voltrade_ml_v3_candidate.pkl")'));
  assert.ok(v3py.includes('"promoted": False'));
  const engine = fs.readFileSync(path.join(here, "..", "bot_engine.py"), "utf8");
  const uses = engine.split("\n").filter((l) => l.includes("v3_win_probability") || l.includes("_v3["));
  assert.equal(uses.length, 0, "bot_engine must not read the v3 probability (it rides in _ml_trace only)");
});

test("ml_v3 diag probe: whitelisted, routed, read-only", () => {
  assert.ok((DIAG_PROBES as readonly string[]).includes("ml_v3"));
  const block = bot.slice(bot.indexOf('case "ml_v3": {'), bot.indexOf('case "shadow_bands": {'));
  assert.ok(block.includes("sanitizeDiag"));
  assert.ok(!block.includes("execPython") && !block.includes("writeFileSync"));
  assert.ok(block.includes("ml_v3_eval.json") && block.includes("job_state.json"));
});
