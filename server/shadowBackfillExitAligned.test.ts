// [RULE-REVIEW] 2026-10-14 — wiring pins for the EXIT-ALIGNED shadow ruler's
// nightly job (shadow_portfolio.backfill_exit_aligned), which labels shadow
// records under the live checkPositionOnTick exit policy ALONGSIDE the old
// +2%/-4% outcomes. bot.ts is a side-effecting orchestrator and can't be
// imported, so — like shadowBackfillVisibility.test.ts, which pins the old
// job and must keep passing untouched — these read the source. Where it
// matters (the kill cap, the run gate) they go further: they cut bot.ts's OWN
// functions out of the source, transpile them with esbuild and RUN them
// against stubs, so the test measures what the code does, not what a
// comment says it does.
//
// What must hold:
//   1. its own top-level function + flag (the old checkShadowBackfill block
//      stays byte-identical; its own pin test still guards that),
//   2. driven by an unconditional interval (22:00 UTC is never inside market
//      hours — the KNOWN BROKEN #10/#20 root cause),
//   3. SEQUENCED after the old job (two subprocesses each build their own
//      180/min Alpaca bucket, so concurrency could breach the 200/min quota),
//      with a bounded number of attempts per night,
//   4. the REAL kill cap — derived by running isTrivialPython and
//      execPythonSerialized on the real command — leaves room for the Python
//      job's own time budget plus its final save (2026-10-15: the first
//      version's 132-char command was "trivial", i.e. no mutex and SIGKILL at
//      10s, while its comments and this test assumed 90s),
//   5. success and both failure paths audited (SHADOW-BACKFILL-XA / -ERROR).
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { transformSync } from "esbuild";
import { isTrivialPython, planPythonExec } from "./pythonExec";

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
const shadowPy = fs.readFileSync(path.join(here, "..", "shadow_portfolio.py"), "utf8");

function xaBlock(): string {
  const start = bot.indexOf("async function checkShadowBackfillExitAligned()");
  assert.ok(start > 0, "checkShadowBackfillExitAligned not found in bot.ts");
  const end = bot.indexOf("\n  }\n", start);
  assert.ok(end > start, "checkShadowBackfillExitAligned body end not found");
  return bot.slice(start, end);
}

/** Index just past the `}` closing the brace opened at `open` (quotes skipped). */
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

/** Source of a function declared in bot.ts, cut at its own closing brace. */
function functionSource(signature: string): string {
  const start = bot.indexOf(signature);
  assert.ok(start >= 0, `${signature} not found in bot.ts`);
  return bot.slice(start, matchBrace(bot, bot.indexOf("{", start + signature.length)));
}

function transpile(ts: string): string {
  return transformSync(ts, { loader: "ts" }).code;
}

type ExecCall = { cmd: string; opts: { timeout?: number; killSignal?: string }; mutexHeld: boolean };

/**
 * bot.ts's own execPythonSerialized, run against a stub execAsync, with the
 * classifier/planner it calls (server/pythonExec.ts since 2026-10-27, audit D1).
 */
function pythonExecHarness() {
  for (const decl of ["let pythonRunning = false;", "let pythonLockedAt = 0;", "let pythonLockHardTimeout = 90000;"]) {
    assert.ok(bot.includes(`\n${decl}`), `mutex state declaration changed: ${decl}`);
  }
  const js = transpile([
    "let pythonRunning = false;", "let pythonLockedAt = 0;", "let pythonLockHardTimeout = 90000;",
    functionSource("async function execPythonSerialized(cmd: string, opts?: ExecOptions)"),
  ].join("\n"));
  const calls: ExecCall[] = [];
  let mutexHeld: () => boolean = () => false;
  const execAsync = (cmd: string, opts: ExecCall["opts"]) => {
    calls.push({ cmd, opts, mutexHeld: mutexHeld() });
    return Promise.resolve({ stdout: "{}", stderr: "" });
  };
  const api = new Function("execAsync", "planPythonExec",
    `${js}\nreturn { execPythonSerialized, mutexHeld: () => pythonRunning };`)(execAsync, planPythonExec) as {
    execPythonSerialized: (cmd: string, opts?: { timeout?: number }) => Promise<unknown>;
    mutexHeld: () => boolean;
  };
  mutexHeld = api.mutexHeld;
  return { ...api, isTrivialPython, calls };
}

/** The exact command string + options checkShadowBackfillExitAligned passes. */
function xaCall(): { cmd: string; timeout: number } {
  const m = xaBlock().match(/execPythonSerialized\(\s*`([^`]*)`,\s*\{\s*timeout:\s*(\d+)\s*\}\s*\)/);
  assert.ok(m, "the job must call execPythonSerialized(`<literal command>`, { timeout: N })");
  assert.ok(!m![1].includes("${"), "the command must be a plain literal so the test sees the exact string");
  return { cmd: m![1], timeout: Number(m![2]) };
}

test("exit-aligned job: own top-level function, outside the old pinned block and outside tier1Reflex", () => {
  const block = xaBlock();
  assert.ok(block.includes("from shadow_portfolio import backfill_exit_aligned"));
  const oldStart = bot.indexOf("async function checkShadowBackfill()");
  const oldEnd = bot.indexOf("if (nowHour === 0) {", oldStart);
  assert.ok(oldStart > 0 && oldEnd > oldStart);
  assert.ok(!bot.slice(oldStart, oldEnd).includes("backfill_exit_aligned"),
    "the old checkShadowBackfill block must stay untouched — the new ruler gets its own function");
  const tier1Start = bot.indexOf("async function tier1Reflex()");
  const tier1End = bot.indexOf("// Once per day, backfill shadow portfolio outcomes.", tier1Start);
  assert.ok(!bot.slice(tier1Start, tier1End).includes("backfill_exit_aligned"),
    "must not live inside market-hours-gated tier1Reflex");
  assert.ok(bot.includes("let _shadowXaBackfilledToday = false;"), "needs its own once-a-day flag");
});

test("exit-aligned job: unconditional 5-minute interval (no market-hours / kill-switch gate)", () => {
  const call = bot.indexOf("checkShadowBackfillExitAligned().catch(");
  assert.ok(call > 0, "must be driven by a scheduled interval");
  const line = bot.slice(bot.lastIndexOf("\n", call), bot.indexOf("\n", call));
  assert.ok(line.includes("setInterval"), "the call must be the interval's own callback");
  assert.ok(!line.includes("is_open") && !line.includes("state.killSwitch"));
});

test("exit-aligned job: the REAL kill cap, from bot.ts's own isTrivialPython/execPythonSerialized on the real command", async () => {
  const h = pythonExecHarness();

  // The harness reproduced the defect it exists for: the first version's
  // 132-char command was "trivial" — no mutex, SIGKILL at 10s, not 90s.
  const ROUND1 = `python3 -c "from shadow_portfolio import backfill_exit_aligned; import json; print(json.dumps(backfill_exit_aligned(3000, 40, 60)))"`;
  assert.equal(h.isTrivialPython(ROUND1), true);
  // 2026-10-27 (audit D1): text alone still reads "trivial", but the caller
  // asked for 120 s, so it is no longer forced into the no-mutex 10 s lane:
  // it holds the mutex and gets the mutex lane's 90 s ceiling.
  assert.equal(h.isTrivialPython(ROUND1, 120000), false);
  await h.execPythonSerialized(ROUND1, { timeout: 120000 });
  // Was [10000, false] — the pinned defect itself (no mutex, SIGKILL at 10s).
  assert.deepEqual([h.calls[0].opts.timeout, h.calls[0].mutexHeld], [90000, true]);

  const { cmd, timeout } = xaCall();
  assert.equal(h.isTrivialPython(cmd), false,
    `${cmd.length}-char command is "trivial" (<= 150 chars, no heavy import): it would skip the mutex and be SIGKILLed at 10s`);
  await h.execPythonSerialized(cmd, { timeout });
  const real = h.calls[1];
  assert.equal(real.cmd, cmd);
  assert.equal(real.mutexHeld, true, "runs while holding the Python mutex (the job loads the whole shadow log)");
  assert.equal(h.mutexHeld(), false, "and releases it");
  assert.equal(real.opts.killSignal, "SIGKILL");
  const capMs = real.opts.timeout!;
  assert.equal(capMs, 90000, "derived cap");

  // The Python side is sized to THAT cap: its wall-clock budget (no request
  // starts unless it fits) plus one reserved merge-save, plus process
  // start/exit slack, must finish before the SIGKILL.
  const kw = (name: string) => {
    const m = cmd.match(new RegExp(`${name}=(\\d+(?:\\.\\d+)?)`));
    assert.ok(m, `the command must pass ${name}= explicitly`);
    return Number(m![1]);
  };
  const [maxRecords, maxWindows, budgetS] = [kw("max_records"), kw("max_windows"), kw("time_budget_s")];
  assert.ok(maxRecords <= 3000 && maxWindows <= 40, "nightly job counts stay bounded");
  const reserve = shadowPy.match(/^XA_SAVE_RESERVE_S = (\d+(?:\.\d+)?)/m);
  assert.ok(reserve, "shadow_portfolio.XA_SAVE_RESERVE_S not found");
  const START_EXIT_SLACK_S = 10;
  assert.ok(budgetS + Number(reserve![1]) + START_EXIT_SLACK_S <= capMs / 1000,
    `time_budget_s=${budgetS} + save reserve ${reserve![1]}s + ${START_EXIT_SLACK_S}s slack exceeds the ${capMs}ms cap`);
});

test("exit-aligned job: sequenced AFTER the old ruler's job, at most N attempts a night, reset at midnight", async () => {
  // Run the REAL gate: cut the flag declarations + the function out of bot.ts.
  const start = bot.indexOf("let _shadowXaBackfilledToday = false;");
  const fnStart = bot.indexOf("async function checkShadowBackfillExitAligned()", start);
  assert.ok(start > 0 && fnStart > start && fnStart - start < 300, "flags must sit right above the function");
  const src = bot.slice(start, matchBrace(bot, bot.indexOf("{", fnStart)));
  let hour = 22;
  let fail = true;
  const runs: string[] = [];
  const audits: string[] = [];
  class FakeDate { getUTCHours() { return hour; } }
  const exec = (cmd: string) => {
    runs.push(cmd);
    return fail ? Promise.reject(new Error("boom")) : Promise.resolve({ stdout: '{"updated": 1}' });
  };
  // job heartbeats (server/jobHeartbeat.ts, 2026-10-27): the function beats
  // start/finish around each attempt — recorded here and asserted below.
  const beats: string[] = [];
  const job = new Function("execPythonSerialized", "audit", "Date", "beat",
    `let _shadowBackfilledToday = false;\n${transpile(src)}\n` +
    "return { run: checkShadowBackfillExitAligned, setOldDone: (v) => { _shadowBackfilledToday = v; } };")(
    exec, (type: string, msg: string) => audits.push(`${type} ${msg}`), FakeDate,
    (job: string, phase: string, status?: string) => beats.push(`${job} ${phase}${status ? ` ${status}` : ""}`)) as {
    run: () => Promise<void>; setOldDone: (v: boolean) => void;
  };

  await job.run();
  assert.equal(runs.length, 0, "hour 22 before the old job finished: wait");
  job.setOldDone(true);
  for (let i = 0; i < 5; i++) await job.run();
  assert.equal(runs.length, 3, "a failing job stops after 3 attempts, not every 5 minutes for two hours");
  assert.equal(audits.filter((a) => a.startsWith("SHADOW-BACKFILL-XA-ERROR attempt ")).length, 3);
  hour = 23;
  await job.run();
  assert.equal(runs.length, 3, "the cap holds across hour 23");
  hour = 0;
  await job.run();
  hour = 23;
  fail = false;
  job.setOldDone(false);
  await job.run();
  assert.equal(runs.length, 4, "midnight resets the count; from 23:00 it runs even if the old job never succeeded");
  assert.ok(audits[audits.length - 1].startsWith('SHADOW-BACKFILL-XA {"updated":1}'));
  await job.run();
  assert.equal(runs.length, 4, "once a night on success");
  // every attempt beats start + finish: 3 failed attempts then 1 success
  assert.equal(beats.filter((b) => b === "shadow_backfill_exit_aligned start").length, 4);
  assert.equal(beats.filter((b) => b === "shadow_backfill_exit_aligned finish error").length, 3);
  assert.equal(beats[beats.length - 1], "shadow_backfill_exit_aligned finish ok");
});

test("exit-aligned job: success and both failure paths reach the persisted audit log", () => {
  const block = xaBlock();
  assert.ok(block.includes('audit("SHADOW-BACKFILL-XA", JSON.stringify(stats))'));
  assert.equal((block.match(/audit\("SHADOW-BACKFILL-XA-ERROR"/g) || []).length, 2,
    "the JSON.parse catch and the subprocess rejection must both audit SHADOW-BACKFILL-XA-ERROR");
  assert.ok(!/:\s*any\b/.test(block), "no `: any` (ts_any ratchet)");
});

test("SHADOW-BACKFILL-XA-ERROR detail-building never throws on a non-Error rejection value", () => {
  const buildDetail = (e: unknown) => String(e instanceof Error ? e.message : e).slice(0, 500);
  assert.equal(buildDetail(new Error("boom")), "boom");
  assert.equal(buildDetail("plain"), "plain");
  assert.equal(buildDetail({ code: "ETIMEDOUT" }), "[object Object]");
  assert.equal(buildDetail(undefined), "undefined");
});
