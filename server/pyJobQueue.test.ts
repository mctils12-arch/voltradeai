// A6 HEAVY PYTHON JOB QUEUE (2026-11-03, memory audit fix A6, second half).
// server/pyJobQueue.ts is the pure half; this pins it — two heavy jobs never
// overlap, FIFO order, a job outside the queue is never delayed, stale holders
// are released, a bounded wait never lets a successor overtake the running job
// — and pins bot.ts's wiring: the retrain and every overnight research phase
// take the queue, Tier 1 and the Tier-2 scan never do.
//
// Ordering in these tests is driven by explicit GATES (a job holds the queue
// until the test releases it), never by one sleep being shorter than another:
// under CI/container load a 10 ms sleep can take 100 ms, and a test that
// assumed otherwise flaked once in a loaded full-suite run.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  PyJobQueue, PyJobWaitTimeout, SERIALIZE_PY_JOBS_ENV, PY_JOB_STALE_GRACE_MS, serializePyJobs,
} from "./pyJobQueue";

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

/** A promise the test resolves by hand, so a fake job holds the queue exactly as long as the test wants. */
function gate(): { wait: Promise<void>; open: () => void } {
  let open!: () => void;
  const wait = new Promise<void>((r) => { open = r; });
  return { wait, open };
}

/** Yield to the event loop until `cond` holds (or ~2 s pass) — never assumes how long a tick takes. */
async function until(cond: () => boolean, what: string): Promise<void> {
  const deadline = Date.now() + 2000;
  while (!cond()) {
    if (Date.now() > deadline) assert.fail(`timed out waiting for: ${what}`);
    await sleep(1);
  }
}

function slice(start: string, end: string, from = bot): string {
  const a = from.indexOf(start);
  assert.ok(a >= 0, `anchor not found: ${start.slice(0, 70)}`);
  const b = from.indexOf(end, a + start.length);
  assert.ok(b > a, `end anchor not found after start: ${end.slice(0, 70)}`);
  return from.slice(a, b);
}

/** Fake heavy jobs: each records its start/end order and how many ran at once. */
function makeJobs() {
  let active = 0;
  let maxActive = 0;
  let seq = 0;
  const started: string[] = [];
  const log: Array<{ name: string; startSeq: number; endSeq: number }> = [];
  const run = (name: string, hold: Promise<void> | number) => async () => {
    active++;
    maxActive = Math.max(maxActive, active);
    started.push(name);
    const startSeq = seq++;
    if (typeof hold === "number") await sleep(hold); else await hold;
    active--;
    log.push({ name, startSeq, endSeq: seq++ });
    return name;
  };
  return { run, log, started, maxActive: () => maxActive, active: () => active };
}

// ── flag ────────────────────────────────────────────────────────────────────
test("VOLTRADE_SERIALIZE_PY_JOBS defaults ON; 0/false/off/no turn it off", () => {
  assert.equal(serializePyJobs({}), true);
  for (const off of ["0", "false", "OFF", " no "]) assert.equal(serializePyJobs({ [SERIALIZE_PY_JOBS_ENV]: off }), false, off);
  for (const on of ["1", "true", "yes", "junk"]) assert.equal(serializePyJobs({ [SERIALIZE_PY_JOBS_ENV]: on }), true, on);
  assert.equal(PY_JOB_STALE_GRACE_MS, 10_000, "same grace as the Python mutex's stale threshold");
});

// ── the two jobs that used to overlap never do ───────────────────────────────
test("two heavy jobs never overlap: the retrain starts only after the research phase releases, and reports its wait", async () => {
  const waits: Array<[string, number, string]> = [];
  const q = new PyJobQueue(true, { onWait: (n, w, b) => waits.push([n, w, b]) });
  const { run, log, started, maxActive, active } = makeJobs();
  const researchGate = gate();

  const research = q.run("overnight-evening-scan", 1000, run("research", researchGate.wait));
  await until(() => q.running === "overnight-evening-scan", "research to take the queue");
  const retrain = q.run("ml-retrain", 1000, run("retrain", 5));
  assert.equal(q.waiting, 1, "the retrain is queued");
  await sleep(30);
  assert.deepEqual(started, ["research"], "the retrain has not started while research holds the queue");
  assert.equal(active(), 1);
  assert.equal(q.running, "overnight-evening-scan");
  researchGate.open();
  const [r1, r2] = await Promise.all([research, retrain]);

  assert.equal(r1.result, "research");
  assert.equal(r2.result, "retrain");
  assert.equal(maxActive(), 1, "never two heavy jobs at once");
  assert.deepEqual(started, ["research", "retrain"]);
  assert.ok(log[1].startSeq > log[0].endSeq, "the retrain started after research ended");
  assert.ok(r2.waitedMs >= 25, `the retrain waited at least the 30 ms the gate was held, got ${r2.waitedMs}`);
  assert.equal(waits.length, 1, "exactly one PY-JOB-QUEUED report, for the job that waited (the first job never waited)");
  assert.equal(waits[0][0], "ml-retrain");
  assert.equal(waits[0][2], "overnight-evening-scan", "...naming what it waited behind");
  assert.ok(waits[0][1] >= 25);
  assert.equal(q.running, null);
  assert.equal(q.waiting, 0);
});

test("FIFO: three jobs run strictly in arrival order, one at a time", async () => {
  const q = new PyJobQueue(true);
  const { run, log, started, maxActive } = makeJobs();
  const aGate = gate();
  const all = Promise.all([
    q.run("a", 1000, run("a", aGate.wait)),
    q.run("b", 1000, run("b", 1)),
    q.run("c", 1000, run("c", 1)),
  ]);
  await until(() => q.running === "a", "a to take the queue");
  assert.equal(q.waiting, 2);
  assert.deepEqual(started, ["a"]);
  aGate.open();
  await all;
  assert.deepEqual(started, ["a", "b", "c"]);
  assert.equal(maxActive(), 1);
  for (let i = 1; i < log.length; i++) assert.ok(log[i].startSeq > log[i - 1].endSeq, `${log[i].name} overlapped ${log[i - 1].name}`);
});

test("a call outside the queue (Tier 1's sync, a Tier-2 scan) is not delayed by a running or queued research job", async () => {
  const q = new PyJobQueue(true);
  const { run } = makeJobs();
  const researchGate = gate();
  const research = q.run("overnight-backtest", 1000, run("research", researchGate.wait));
  const retrain = q.run("ml-retrain", 1000, run("retrain", 1));
  await until(() => q.running === "overnight-backtest", "research to take the queue");
  assert.equal(q.waiting, 1);
  // Tier 1 never calls q.run — its work is a plain await, and it finishes while both heavy jobs are still pending.
  let tier1Done = false;
  await (async () => { await sleep(1); tier1Done = true; })();
  assert.equal(tier1Done, true);
  assert.equal(q.running, "overnight-backtest", "research is still holding the queue — Tier 1 did not wait for it");
  assert.equal(q.waiting, 1, "...and the retrain is still queued behind it");
  researchGate.open();
  await Promise.all([research, retrain]);
});

test("a job that throws releases the queue; the next job still runs", async () => {
  const q = new PyJobQueue(true);
  const { run } = makeJobs();
  const failing = q.run("overnight-earnings", 1000, async () => { await sleep(1); throw new Error("python exploded"); });
  const next = q.run("ml-retrain", 1000, run("retrain", 1));
  await assert.rejects(failing, /python exploded/);
  assert.equal((await next).result, "retrain");
  assert.equal(q.running, null);
});

test("a holder still there past budget + grace is force-released so the retrain is never wedged behind a leak", async () => {
  const stale: Array<[string, number]> = [];
  const { run, started } = makeJobs();
  const q = new PyJobQueue(true, {
    staleGraceMs: 10,
    onStaleRelease: (n, held, budget) => { stale.push([n, budget]); started.push(`stale-release:${n}`); },
  });
  const leakGate = gate();
  const leaked = q.run("overnight-premarket-scan", 30, run("leaked", leakGate.wait));
  await until(() => q.running === "overnight-premarket-scan", "the leaking job to take the queue");
  const retrain = q.run("ml-retrain", 1000, run("retrain", 1));
  const r = await retrain; // the leaked job is still held open: the only way the retrain ran is the stale release
  assert.equal(r.result, "retrain");
  assert.deepEqual(started, ["leaked", "stale-release:overnight-premarket-scan", "retrain"],
    "the retrain started only after the stale release, never beside the still-open leaked job");
  assert.ok(r.waitedMs > 0, `the retrain did wait for the release, got ${r.waitedMs}`);
  assert.deepEqual(stale, [["overnight-premarket-scan", 30]]);
  assert.equal(q.running, null, "the stale holder no longer owns the queue");
  leakGate.open();
  assert.equal((await leaked).result, "leaked", "the leaked job's own result still comes back to its caller");
  assert.equal(q.running, null);
});

test("maxWaitMs: a research phase gives up (PyJobWaitTimeout) without running, and jobs behind it never overtake the running one", async () => {
  const q = new PyJobQueue(true);
  const { run, started, maxActive } = makeJobs();
  const retrainGate = gate();
  let phaseRan = false;
  const retrain = q.run("ml-retrain", 1000, run("retrain", retrainGate.wait));
  await until(() => q.running === "ml-retrain", "the retrain to take the queue");
  const phase = q.run("overnight-evening-scan", 1000, async () => { phaseRan = true; }, { maxWaitMs: 20 });
  const later = q.run("overnight-backtest", 1000, run("backtest", 1));
  const err = await phase.catch((e: unknown) => e);
  assert.ok(err instanceof PyJobWaitTimeout, `expected PyJobWaitTimeout, got ${String(err)}`);
  assert.equal(err.job, "overnight-evening-scan");
  assert.equal(err.behind, "ml-retrain");
  assert.ok(err.waitedMs >= 15, `waited the bound, got ${err.waitedMs}`);
  assert.equal(phaseRan, false, "the skipped phase never ran");
  assert.equal(q.running, "ml-retrain", "giving up did not disturb the running job");
  await sleep(10);
  assert.deepEqual(started, ["retrain"], "the job queued behind the quitter is still waiting for the retrain");
  retrainGate.open();
  await Promise.all([retrain, later]);
  assert.deepEqual(started, ["retrain", "backtest"]);
  assert.equal(maxActive(), 1, "the job queued behind the quitter never overtook the retrain");
});

test("maxWaitMs on an idle queue never throws; a non-waiting job reports no wait", async () => {
  const waits: string[] = [];
  const q = new PyJobQueue(true, { onWait: (n) => waits.push(n) });
  const r = await q.run("overnight-earnings", 1000, async () => "ran", { maxWaitMs: 1 });
  assert.equal(r.result, "ran");
  assert.deepEqual(waits, [], "an idle queue reports no wait");
});

test("VOLTRADE_SERIALIZE_PY_JOBS=0: run() calls the job directly — jobs overlap as before, nothing is reported", async () => {
  const waits: string[] = [];
  const q = new PyJobQueue(false, { onWait: (n) => waits.push(n) });
  const { run, maxActive, active } = makeJobs();
  const aGate = gate();
  const bGate = gate();
  const a = q.run("a", 1000, run("a", aGate.wait));
  const b = q.run("b", 1000, run("b", bGate.wait));
  await until(() => active() === 2, "both jobs to be running at once");
  aGate.open();
  bGate.open();
  await Promise.all([a, b]);
  assert.equal(maxActive(), 2, "the flag off restores today's overlap");
  assert.deepEqual(waits, []);
  assert.equal(q.enabled, false);
});

// ── bot.ts wiring ───────────────────────────────────────────────────────────
test("bot.ts: one heavy-job queue, built from the flag, auditing PY-JOB-QUEUED <name> waited_ms=<n>", () => {
  assert.ok(/import \{[^}]*PyJobQueue[^}]*\} from "\.\/pyJobQueue";/.test(bot));
  assert.equal(bot.split("new PyJobQueue(").length - 1, 1, "exactly one queue");
  // end marker moved 2026-11-06: the OVERNIGHT_PHASE_* constants that followed the declaration left with the overnight engine
  const decl = slice("const heavyPyJobs = new PyJobQueue(serializePyJobs(), {", "async function runLongPythonJob(");
  assert.ok(decl.includes('audit("PY-JOB-QUEUED", `${name} waited_ms=${waitedMs} behind=${behind}`)'));
  assert.ok(decl.includes('audit("PY-JOB-STALE"'));
});

test("bot.ts: the ML retrain takes the queue around the unchanged long lane, start beat at the actual start, second trigger skipped not queued", () => {
  const lane = slice("async function runLongPythonJob(", "// The ML retrain, shared by Tier 3");
  assert.ok(!lane.includes("heavyPyJobs"), "the generic long lane stays as it was; the caller takes the queue");
  assert.ok(lane.includes("if (longPythonJobsInFlight.has(key)) return { skipped: true };"), "single-flight per key is unchanged");
  const retrain = slice("let mlRetrainPending = false;", "// ─── Temp File Cleanup");
  const queued = retrain.indexOf('await heavyPyJobs.run("ml-retrain", ML_RETRAIN_TIMEOUT_MS, async () => {');
  const start = retrain.indexOf('beat("ml_retrain", "start");');
  const lanecall = retrain.indexOf('return runLongPythonJob("ml-retrain", "python3 ml_retrain_safe.py", ML_RETRAIN_TIMEOUT_MS);');
  assert.ok(queued > 0 && start > queued && lanecall > start, "queue -> start beat -> the pinned three-argument long-lane call, in that order");
  assert.equal(retrain.split('beat("ml_retrain", "start")').length - 1, 1, "the start beat fires in exactly one place, inside the queued callback");
  assert.ok(retrain.indexOf("if (mlRetrainPending) {") < queued, "a retrain already queued or running skips the new trigger before it could queue");
  assert.ok(retrain.includes('beat("ml_retrain", "finish", "skipped", "already running or queued")'), "a skip is a one-shot finish beat");
  assert.ok(retrain.includes("mlRetrainPending = false;"), "the pending flag is cleared in finally");
  assert.ok(retrain.includes("queued ${Math.round(waitedMs / 1000)}s"), "the completion line carries the queue wait");
});

// 2026-11-06: this test pinned how each overnight research phase took the
// queue. The engine was removed (unreachable: tier2Intelligence only ran it
// at 20:00-04:00 ET, but scheduleTier2 only calls tier2Intelligence inside
// 04:00-20:00 ET or while the market is open — rule call "overnight research
// window off", decided keep-off). It now pins that nothing of it is left and
// that the queue's remaining entries are the retrain and the forecast jobs,
// each with its own bounded wait (the property the old test guarded).
test("bot.ts: the unreachable overnight research engine is gone; the queue holds only the retrain and the forecast jobs", () => {
  assert.ok(!bot.includes("async function runOvernightResearch(") && !bot.includes("runOvernightResearch("), "no overnight engine and no caller (only history comments may name it)");
  assert.ok(!/OVERNIGHT_PHASE_/.test(bot), "no overnight phase budget left behind");
  for (const phase of ["overnight-evening-scan", "overnight-backtest", "overnight-earnings", "overnight-premarket-scan"]) {
    assert.ok(!bot.includes(`"${phase}"`), `${phase} is gone`);
  }
  assert.ok(!bot.includes("extreme_movers_today.json"), "nothing reads the overnight watchlist file any more");
  const runs = [...bot.matchAll(/heavyPyJobs\.run\("([a-z0-9-]+)"/g)].map((m) => m[1]).sort();
  assert.deepEqual(runs, ["forecast-generate", "forecast-score", "ml-retrain"]);
  assert.equal(bot.split("heavyPyJobs.run(").length - 1, 3, "no other (name-variable) queue entry");
  // the forecast entries keep the bounded wait + audited skip the overnight phases had
  assert.equal(bot.split("if (!(e instanceof PyJobWaitTimeout)) throw e;").length - 1, 2);
  assert.equal(bot.split('audit("PY-JOB-SKIPPED"').length - 1, 2);
});

test("bot.ts: Tier 1 and the Tier-2 scan never enter the heavy-job queue", () => {
  const protectiveFn = slice("async function tier1ProtectiveSync()", "async function tier1HaltedExits(");
  const tier1Reflex = slice("async function tier1Reflex() {", "// Once per day, backfill shadow portfolio outcomes.");
  // (end marker moved 2026-11-06: the overnight call that followed the scan's try/catch was removed; its removal note sits there now)
  const tier2Scan = slice('require("path").resolve(process.cwd(), "bot_engine.py")', '// REMOVED 2026-11-06 (rule call "overnight research window off"');
  for (const [label, text] of [["tier1ProtectiveSync", protectiveFn], ["tier1Reflex", tier1Reflex], ["tier2 scan block", tier2Scan]] as const) {
    assert.ok(!text.includes("heavyPyJobs"), `${label} must not take the heavy-job queue`);
    assert.ok(!text.includes("researchPhase("), `${label} must not be a research phase`);
  }
  assert.ok(protectiveFn.includes("from bot_engine import manage_positions"), "Tier 1's sync is unchanged by this branch");
});
