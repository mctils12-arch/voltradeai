// REPAIR 2026-10-27 (trading-engine audit D1): Python subprocess lane/timeout
// selection and process-group kill. Pins:
//   1. classification: a caller asking for more than the trivial cap is never
//      "trivial" (the retrain asking 900 s was SIGKILLed at 10 s),
//   2. the kill timer is the caller's own timeout, bounded by the lane ceiling
//      (the mutex ceiling unchanged at 90 s),
//   3. the retrain's explicit budget, and that bot.ts runs it in the long-job
//      lane (not behind the mutex, not awaited by Tier 3),
//   4. a timeout kills the WHOLE process group: no orphaned grandchild —
//      demonstrated against plain child_process.exec, which leaves one.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { exec } from "node:child_process";
import { promisify } from "node:util";
import { fileURLToPath } from "node:url";
import {
  execGroupKill, isTrivialPython, planPythonExec, ML_RETRAIN_TIMEOUT_MS, PY_MUTEX_MAX_TIMEOUT_MS,
  PY_TRIVIAL_TIMEOUT_MS, type GroupExecError,
} from "./pythonExec";

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");

// Real command strings from bot.ts (the audit §1.2 table), with what they ask for.
const RETRAIN = "python3 ml_retrain_safe.py";
const BACKFILL = `python3 -c "from shadow_portfolio import backfill_outcomes; import json; print(json.dumps(backfill_outcomes(3000)))"`;
const SHORTS = 'python3 -c "from intraday_shorts import close_open_shorts; import json; print(json.dumps(close_open_shorts()))"';
const FORM4 = `python3 -c "\nfrom sec_form4_bulk import run_update\nimport json\nprint(json.dumps(run_update()))\n"`;
const PING = `python3 -c "print('ok')"`;

test("classification: asking for more than the trivial cap is never trivial", () => {
  const table: Array<[string, number | undefined, boolean]> = [
    [RETRAIN, 900000, false],     // the live bug: 26 chars, asked 900 s, was trivial
    [RETRAIN, undefined, true],   // text-only classification unchanged
    [BACKFILL, 120000, false],
    [SHORTS, 30000, false],
    [FORM4, 120000, false],
    [PING, 5000, true],           // real pings stay trivial
    [PING, PY_TRIVIAL_TIMEOUT_MS, true],
    [PING, PY_TRIVIAL_TIMEOUT_MS + 1, false],
    [PING, 0, true],              // 0 = "no timeout named", as before
    ["python3 -c \"import numpy; print(1)\"", 1000, false], // heavy import, as before
    [`python3 -c "${"x".repeat(200)}"`, 1000, false],       // > 150 chars, as before
  ];
  for (const [cmd, t, want] of table) assert.equal(isTrivialPython(cmd, t), want, `${cmd.slice(0, 50)} @ ${t}`);
});

test("planPythonExec: the kill timer is the caller's own, bounded by the lane ceiling", () => {
  const table: Array<[string, number | undefined, string, number, number]> = [
    // [cmd, requested, lane, timeoutMs, mutexWaitMs]
    [PING, 5000, "trivial", 5000, 0],
    [PING, undefined, "trivial", 10000, 0],
    [SHORTS, 30000, "mutex", 30000, 30000],            // was trivial @ 10 s
    [BACKFILL, 120000, "mutex", 90000, 120000],        // was trivial @ 10 s; mutex ceiling holds
    [RETRAIN, 900000, "mutex", 90000, 900000],         // never via this lane in bot.ts — see the long-job lane
    ["python3 -c \"" + "y".repeat(200) + "\"", undefined, "mutex", 90000, 30000], // old defaults kept
    ["python3 -c \"" + "y".repeat(200) + "\"", 15000, "mutex", 15000, 15000],
  ];
  for (const [cmd, req, lane, timeoutMs, waitMs] of table) {
    assert.deepEqual(planPythonExec(cmd, req), { lane, timeoutMs, mutexWaitMs: waitMs }, `${cmd.slice(0, 40)} @ ${req}`);
  }
  assert.equal(PY_MUTEX_MAX_TIMEOUT_MS, 90000, "the mutex hold ceiling is deliberately unchanged");
});

test("retrain budget: explicit, well above the trivial/mutex caps, still bounded", () => {
  assert.ok(ML_RETRAIN_TIMEOUT_MS >= 15 * 60_000 && ML_RETRAIN_TIMEOUT_MS <= 30 * 60_000, String(ML_RETRAIN_TIMEOUT_MS));
});

test("bot.ts wiring: execPythonSerialized plans through planPythonExec and kills via the group-kill exec", () => {
  assert.ok(bot.includes('from "./pythonExec"'), "bot.ts imports the tested module");
  const start = bot.indexOf("async function execPythonSerialized(cmd: string, opts?: ExecOptions)");
  assert.ok(start > 0);
  const body = bot.slice(start, bot.indexOf("\n}\n", start));
  assert.ok(body.includes("planPythonExec(cmd, opts?.timeout)"), "the lane + timer come from the pure planner");
  assert.ok(!body.includes("Math.min(hardTimeout, 10000)"), "no silent 10 s override of the caller's timeout");
  assert.ok(!/function isTrivialPython/.test(bot), "one classifier, in server/pythonExec.ts");
  assert.ok(/const _execRaw = execGroupKill;/.test(bot), "every bot.ts Python subprocess kill hits the whole process group");
});

test("bot.ts wiring: the ML retrain runs in the long-job lane with its explicit budget, off the Tier-3 path", () => {
  const retrainCalls = bot.split("ml_retrain_safe.py").length - 1;
  assert.ok(retrainCalls >= 1);
  assert.ok(!/execPythonSerialized\(\s*[`"']python3 ml_retrain_safe\.py/.test(bot), "the retrain must not go through the mutex lane");
  assert.ok(/runLongPythonJob\("ml-retrain", "python3 ml_retrain_safe\.py", ML_RETRAIN_TIMEOUT_MS\)/.test(bot));
  const tier3 = bot.slice(bot.indexOf("async function tier3Strategic()"), bot.indexOf("// 2. Manipulation detection scan"));
  assert.ok(/\n\s+startMlRetrain\("tier3/.test(tier3), "Tier 3 starts the retrain without awaiting it");
  assert.ok(!/await\s+startMlRetrain/.test(bot), "nobody awaits the 20-minute retrain on a scheduler path");
  const daily = bot.slice(bot.indexOf("async function runDailyMaintenance(etDate: string)"), bot.indexOf("// Nightly auto-backup"));
  assert.ok(/\n\s+startMlRetrain\("daily/.test(daily), "the daily branch starts the same single-flight retrain");
  const lane = bot.slice(bot.indexOf("async function runLongPythonJob("), bot.indexOf("function startMlRetrain("));
  assert.ok(lane.includes("longPythonJobsInFlight.has(key)"), "single-flight per job");
  assert.ok(!lane.includes("pythonRunning"), "the long lane never takes the global mutex");
});

// ── process-group kill, against real processes ───────────────────────────────

/** True while `pid` is a live, non-zombie process. */
function alive(pid: number): boolean {
  try {
    const stat = fs.readFileSync(`/proc/${pid}/stat`, "utf8");
    return stat.slice(stat.lastIndexOf(")") + 2)[0] !== "Z";
  } catch {
    return false;
  }
}

async function waitFor(pred: () => boolean, ms: number): Promise<boolean> {
  const end = Date.now() + ms;
  while (Date.now() < end) {
    if (pred()) return true;
    await new Promise((r) => setTimeout(r, 25));
  }
  return pred();
}

/** A shell whose grandchild (`sleep`) is not exec'd in place and records its pid. */
function orphanMaker(): { cmd: string; pidFile: string } {
  const pidFile = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "pyexec-")), "grandchild.pid");
  return { cmd: `sleep 30 & echo $! > ${pidFile}; wait`, pidFile };
}

test("before (plain exec): a timeout SIGKILLs the shell only and the grandchild survives as an orphan", async () => {
  if (process.platform !== "linux") return;
  const { cmd, pidFile } = orphanMaker();
  await assert.rejects(promisify(exec)(cmd, { timeout: 300, killSignal: "SIGKILL" }));
  assert.ok(await waitFor(() => fs.existsSync(pidFile) && fs.readFileSync(pidFile, "utf8").trim() !== "", 2000));
  const pid = Number(fs.readFileSync(pidFile, "utf8").trim());
  try {
    assert.equal(alive(pid), true, "this is the defect: the grandchild outlives the kill");
  } finally {
    try { process.kill(pid, "SIGKILL"); } catch (e: unknown) { assert.ok(e); }
  }
});

test("after (execGroupKill): a timeout kills the whole group, exec's error shape preserved", async () => {
  if (process.platform !== "linux") return;
  const { cmd, pidFile } = orphanMaker();
  const err = await execGroupKill(cmd, { timeout: 300, killSignal: "SIGKILL" }).then(
    () => assert.fail("must reject"), (e: GroupExecError) => e);
  assert.equal(err.killed, true);
  assert.equal(err.signal, "SIGKILL");
  assert.equal(err.code, null, "same as exec on a signal kill (bot.ts renders it as code=?)");
  assert.ok(err.message.startsWith(`Command failed: ${cmd}`));
  const pid = Number(fs.readFileSync(pidFile, "utf8").trim());
  assert.ok(pid > 0);
  assert.ok(await waitFor(() => !alive(pid), 2000), "no orphan: the grandchild died with its group");
});

test("execGroupKill: success, non-zero exit, stderr, env, maxBuffer — exec-compatible", async () => {
  const ok = await execGroupKill(`printf 'héllo'; printf 'warn' 1>&2`, { encoding: "utf8" });
  assert.deepEqual(ok, { stdout: "héllo", stderr: "warn" });
  const env = await execGroupKill('printf "$PYEXEC_T"', { env: { ...process.env, PYEXEC_T: "v1" } });
  assert.equal(env.stdout, "v1");
  const fail = await execGroupKill("echo out; echo boom 1>&2; exit 3").then(
    () => assert.fail("must reject"), (e: GroupExecError) => e);
  assert.equal(fail.code, 3);
  assert.equal(fail.killed, false);
  assert.equal(fail.signal, null);
  assert.equal(fail.stdout, "out\n");
  assert.equal(fail.stderr, "boom\n");
  assert.ok(fail.message.includes("boom"));
  const big = await execGroupKill("yes x | head -c 100000", { maxBuffer: 1000 }).then(
    () => assert.fail("must reject"), (e: GroupExecError) => e);
  assert.equal(big.code, "ERR_CHILD_PROCESS_STDIO_MAXBUFFER", "bot.ts's TIER2 classifier keys on this code");
});
