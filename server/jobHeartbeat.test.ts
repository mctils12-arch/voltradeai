// Job heartbeats + watchdog evaluation (Phase 3 "no silent failures",
// human-directed 2026-10-27). server/jobHeartbeat.ts is the server side the
// external watchdog routine polls through /api/diag/heartbeats.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  JOB_REGISTRY, evaluate, evaluateJob, applyBeat, applyNoteError, closeInterruptedRuns, parseHeartbeatFile,
  createHeartbeatStore, readHeartbeatFile, withJobBeat, healthJobsBlock, classifyJobError, windowState, zonedInstant,
  MAX_RUNS_PER_JOB, MAX_DETAIL_CHARS, type JobRun, type JobSpec, type MarketCalendar, type HeartbeatFile,
} from "./jobHeartbeat";
import { parseMarketHolidays } from "./optionsChainArchive";
import { SERVING_CHECKS, healthHttpCode } from "./healthGate";
import { DIAG_PROBES } from "./diag";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const read = (rel: string) => fs.readFileSync(path.join(HERE, "..", rel), "utf8");

const CAL: MarketCalendar = { holidays: new Set(["2026-11-26"]), halfDays: new Set(["2026-11-27"]) };
const spec = (id: string): JobSpec => {
  const s = JOB_REGISTRY.find((j) => j.id === id);
  assert.ok(s, `registry has ${id}`);
  return s;
};
const T = (iso: string) => Date.parse(iso);
const SEC = 1000, MIN = 60_000, HOUR = 3_600_000;
const LONG_AGO_BOOT = T("2026-10-01T00:00:00Z");
const ctx = { calendar: CAL, bootAtMs: LONG_AGO_BOOT, env: { ALPACA_KEY: "k", ALPACA_SECRET: "s" } };
const ok = (startedAt: number, dur = 5 * SEC): JobRun => ({ startedAt, finishedAt: startedAt + dur, status: "ok" });

// Tue 2026-10-27 is a trading day (EDT, UTC-4): 10:00 ET = 14:00Z
const TUE_10ET = T("2026-10-27T14:00:00Z");

test("tier1 (market hours): ok / late / missing / errored / stuck on a trading day", () => {
  const t1 = spec("tier1_reflex");
  assert.equal(evaluateJob(t1, [ok(TUE_10ET - 30 * SEC)], TUE_10ET, ctx).status, "ok");
  const late = evaluateJob(t1, [ok(TUE_10ET - 10 * MIN)], TUE_10ET, ctx);
  assert.equal(late.status, "late");
  assert.match(late.reason, /cadence \+ grace/);
  assert.equal(evaluateJob(t1, [], TUE_10ET, ctx).status, "missing");
  const err = evaluateJob(t1, [{ startedAt: TUE_10ET - 20 * SEC, finishedAt: TUE_10ET - 10 * SEC, status: "error", detail: "drawdown check: boom" }], TUE_10ET, ctx);
  assert.equal(err.status, "errored");
  assert.match(err.reason, /boom/);
  const stuck = evaluateJob(t1, [{ startedAt: TUE_10ET - 6 * MIN }], TUE_10ET, ctx);
  assert.equal(stuck.status, "late", "a run open past maxRunMs is stuck");
  assert.ok(stuck.running);
  // a skipped tick (gated) still proves the timer fires
  assert.equal(evaluateJob(t1, [{ startedAt: TUE_10ET - 40 * SEC, finishedAt: TUE_10ET - 40 * SEC, status: "skipped", detail: "bot inactive" }], TUE_10ET, ctx).status, "ok");
  // an older error followed by an ok run is not "errored"
  assert.equal(evaluateJob(t1, [{ startedAt: TUE_10ET - 90 * SEC, finishedAt: TUE_10ET - 80 * SEC, status: "error" }, ok(TUE_10ET - 30 * SEC)], TUE_10ET, ctx).status, "ok");
});

test("tier1 window: weekend, holiday, half-day close and the open anchor read not_due / ok, never late", () => {
  const t1 = spec("tier1_reflex");
  const yesterdayClose = T("2026-10-26T19:59:00Z");
  assert.equal(evaluateJob(t1, [ok(yesterdayClose)], T("2026-10-31T14:00:00Z"), ctx).status, "not_due", "Saturday");
  assert.equal(evaluateJob(t1, [], T("2026-11-26T15:00:00Z"), ctx).status, "not_due", "Thanksgiving (holiday from the calendar)");
  assert.equal(evaluateJob(t1, [], T("2026-11-27T19:00:00Z"), ctx).status, "not_due", "14:00 ET on a 13:00 half day");
  assert.equal(evaluateJob(t1, [], T("2026-11-27T17:00:00Z"), ctx).status, "missing", "12:00 ET on the half day is inside the session");
  assert.equal(evaluateJob(t1, [], T("2026-10-27T12:00:00Z"), ctx).status, "not_due", "08:00 ET pre-open");
  // the window opening anchors lateness: 09:31 with the last run yesterday is fine, 09:40 is late
  assert.equal(evaluateJob(t1, [ok(yesterdayClose)], T("2026-10-27T13:31:00Z"), ctx).status, "ok");
  assert.equal(evaluateJob(t1, [ok(yesterdayClose)], T("2026-10-27T13:40:00Z"), ctx).status, "late");
});

test("boot warm-up: a job with no beat since a fresh restart is not late inside its first cadence (capped)", () => {
  const t1 = spec("tier1_reflex");
  const fresh = { ...ctx, bootAtMs: TUE_10ET - 60 * SEC };
  assert.equal(evaluateJob(t1, [ok(TUE_10ET - 10 * MIN)], TUE_10ET, fresh).status, "not_due");
  // the cap: a 24h-cadence job is NOT excused for 30h just because the process restarted
  const ml = spec("ml_retrain");
  const bootedHourAgo = { ...ctx, bootAtMs: TUE_10ET - HOUR };
  assert.equal(evaluateJob(ml, [ok(TUE_10ET - 40 * HOUR)], TUE_10ET, bootedHourAgo).status, "late");
  assert.equal(evaluateJob(ml, [ok(TUE_10ET - 20 * HOUR)], TUE_10ET, bootedHourAgo).status, "ok");
});

test("overnight window wraps midnight (20:00-04:00 ET)", () => {
  const ov = spec("overnight_research");
  const d = windowState({ kind: "et_range", startMin: 20 * 60, endMin: 4 * 60 }, T("2026-10-28T06:00:00Z"), CAL); // 02:00 ET
  assert.equal(d.active, true);
  assert.equal(d.openedAt, T("2026-10-28T00:00:00Z"), "opened 20:00 ET the previous evening");
  assert.equal(evaluateJob(ov, [], T("2026-10-27T16:00:00Z"), ctx).status, "not_due", "12:00 ET");
  assert.equal(evaluateJob(ov, [], T("2026-10-28T00:30:00Z"), ctx).status, "ok", "20:30 ET: inside the first cadence after the window opened");
  assert.equal(evaluateJob(ov, [], T("2026-10-28T02:00:00Z"), ctx).status, "missing", "22:00 ET: never ran this window");
  assert.equal(evaluateJob(ov, [ok(T("2026-10-28T00:05:00Z"))], T("2026-10-28T06:00:00Z"), ctx).status, "late", "02:00 ET, last run 20:05 ET");
});

test("daily, trading days (options chain archive, due 16:15 ET, deadline 18:30 ET)", () => {
  const oc = spec("options_chain_archive");
  const day = "2026-10-27";
  const due = zonedInstant(day, 16 * 60 + 15, "America/New_York");
  assert.equal(due, T("2026-10-27T20:15:00Z"));
  const prev = [ok(T("2026-10-26T20:20:00Z"))];
  assert.equal(evaluateJob(oc, prev, T("2026-10-27T19:00:00Z"), ctx).status, "not_due", "15:00 ET, before due");
  assert.equal(evaluateJob(oc, prev, T("2026-10-27T21:30:00Z"), ctx).status, "not_due", "17:30 ET, inside grace");
  assert.equal(evaluateJob(oc, prev, T("2026-10-27T23:00:00Z"), ctx).status, "late", "19:00 ET, deadline passed");
  assert.equal(evaluateJob(oc, [], T("2026-10-27T23:00:00Z"), ctx).status, "missing");
  assert.equal(evaluateJob(oc, [...prev, ok(T("2026-10-27T20:20:00Z"))], T("2026-10-27T23:00:00Z"), ctx).status, "ok");
  const failed = [...prev, { startedAt: T("2026-10-27T20:20:00Z"), finishedAt: T("2026-10-27T20:25:00Z"), status: "error" as const, detail: "total failure 40/40" }];
  assert.equal(evaluateJob(oc, failed, T("2026-10-27T20:30:00Z"), ctx).status, "errored", "an error since due is errored even before the deadline");
  assert.equal(evaluateJob(oc, [...failed, ok(T("2026-10-27T21:20:00Z"))], T("2026-10-27T23:00:00Z"), ctx).status, "ok", "the hourly retry recovered the day");
  assert.equal(evaluateJob(oc, prev, T("2026-10-31T23:00:00Z"), ctx).status, "not_due", "Saturday");
  assert.equal(evaluateJob(oc, prev, T("2026-11-26T23:00:00Z"), ctx).status, "not_due", "Thanksgiving");
  const off = evaluateJob(oc, [], T("2026-10-27T23:00:00Z"), { ...ctx, env: {} });
  assert.equal(off.status, "not_due");
  assert.match(off.reason, /disabled/);
});

test("daily UTC (shadow backfill, hour 22, deadline 23:00 UTC) runs on weekends too", () => {
  const sb = spec("shadow_backfill");
  const prev = [ok(T("2026-10-30T22:00:00Z"))];
  assert.equal(evaluateJob(sb, prev, T("2026-10-31T21:00:00Z"), ctx).status, "not_due");
  assert.equal(evaluateJob(sb, prev, T("2026-10-31T23:30:00Z"), ctx).status, "late", "Saturday counts: the job is unconditional");
  assert.equal(evaluateJob(sb, [...prev, ok(T("2026-10-31T22:05:00Z"))], T("2026-10-31T23:30:00Z"), ctx).status, "ok");
});

test("bisection flags mark Tier 2 / Tier 3 / retrain disabled (not_due), never late", () => {
  const env = { ...ctx.env, VOLTRADE_DISABLE_TIER2: "1", VOLTRADE_DISABLE_TIER3: "1" };
  for (const id of ["tier2_scan", "tier3_strategic", "ml_retrain"]) {
    const r = evaluateJob(spec(id), [], TUE_10ET, { ...ctx, env });
    assert.equal(r.status, "not_due", id);
    assert.match(r.reason, /VOLTRADE_DISABLE_TIER/);
  }
});

test("evaluate(): overall verdict, failing list, counts, unregistered jobs; health block is names+counts only", () => {
  const file: HeartbeatFile = { version: 1, jobs: {
    tier1_reflex: [ok(TUE_10ET - 30 * SEC)],
    tier2_scan: [{ startedAt: TUE_10ET - 2 * MIN, finishedAt: TUE_10ET - MIN, status: "error", detail: "secret-ish detail" }],
    mystery_job: [ok(TUE_10ET)],
  } };
  const r = evaluate(TUE_10ET, file, JOB_REGISTRY, ctx);
  assert.equal(r.verdict, "degraded");
  assert.ok(r.failing.includes("tier2_scan"));
  assert.ok(!r.failing.includes("tier1_reflex"));
  assert.deepEqual(r.unregistered, ["mystery_job"]);
  assert.equal(r.jobs.length, JOB_REGISTRY.length);
  assert.equal(Object.values(r.counts).reduce((a, b) => a + b, 0), JOB_REGISTRY.length);
  const h = healthJobsBlock(r);
  assert.equal(h.status, "degraded");
  assert.equal(h.gates_top_level_status, false);
  assert.ok(!JSON.stringify(h).includes("secret-ish"), "no free-text detail on the unauthenticated route");
  const allOk = evaluate(TUE_10ET, { version: 1, jobs: {} }, [spec("tier1_reflex")], { ...ctx, bootAtMs: TUE_10ET - SEC });
  assert.equal(allOk.verdict, "ok");
});

test("applyBeat: start/finish, one-shot skipped, noted error wins, overlap closes the old run, bounded", () => {
  let runs = applyBeat(undefined, "start", 1000);
  assert.deepEqual(runs, [{ startedAt: 1000 }]);
  runs = applyBeat(runs, "finish", 2000, "ok");
  assert.deepEqual(runs, [{ startedAt: 1000, finishedAt: 2000, status: "ok" }]);
  runs = applyBeat(runs, "finish", 3000, "skipped", "market closed");
  assert.deepEqual(runs[1], { startedAt: 3000, finishedAt: 3000, status: "skipped", detail: "market closed" });
  runs = applyBeat(runs, "start", 4000);
  runs = applyNoteError(runs, 4500, "stop-state sync: ECONNRESET");
  runs = applyBeat(runs, "finish", 5000, "ok");
  assert.equal(runs[2].status, "error", "a swallowed step error makes the run an error");
  assert.match(String(runs[2].detail), /ECONNRESET/);
  runs = applyBeat(runs, "start", 6000);
  runs = applyBeat(runs, "start", 7000);
  assert.equal(runs[3].status, "error");
  assert.equal(runs[3].finishedAt, 7000);
  assert.equal(runs[4].finishedAt, undefined);
  let many: JobRun[] = [];
  for (let i = 0; i < 50; i++) many = applyBeat(many, "finish", i, "ok", "x".repeat(1000));
  assert.equal(many.length, MAX_RUNS_PER_JOB);
  assert.equal(many[many.length - 1].startedAt, 49, "keeps the newest");
  assert.equal(String(many[0].detail).length, MAX_DETAIL_CHARS);
  assert.deepEqual(applyNoteError([], 9, "boom"), [{ startedAt: 9, finishedAt: 9, status: "error", detail: "boom" }]);
});

test("persistence: round trip, interrupted runs close as errors on load, corrupt file -> empty", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "hb-"));
  const file = path.join(dir, "nested", "hb.json");
  let now = 1_000;
  const a = createHeartbeatStore(file, () => now);
  a.beat("tier1_reflex", "start"); now = 2_000;
  a.beat("tier1_reflex", "finish", "ok");
  a.beat("tier3_strategic", "start"); // process "dies" here
  assert.equal(a.lastWriteError(), null);
  const onDisk = JSON.parse(fs.readFileSync(file, "utf8"));
  assert.equal(onDisk.version, 1);
  assert.equal(typeof onDisk.savedAt, "string");
  const b = createHeartbeatStore(file, () => 3_000);
  const snap = b.snapshot();
  assert.deepEqual(snap.jobs.tier1_reflex, [{ startedAt: 1000, finishedAt: 2000, status: "ok" }]);
  assert.equal(snap.jobs.tier3_strategic[0].status, "error");
  assert.match(String(snap.jobs.tier3_strategic[0].detail), /interrupted/);
  fs.writeFileSync(file, "{ torn");
  assert.deepEqual(readHeartbeatFile(file), { version: 1, jobs: {} });
  assert.deepEqual(parseHeartbeatFile({ jobs: { x: [{ startedAt: "nope" }, { startedAt: 5, status: "weird" }] } }), { version: 1, jobs: { x: [{ startedAt: 5 }] } });
  assert.deepEqual(closeInterruptedRuns({ version: 1, jobs: { y: [{ startedAt: 7, finishedAt: 8, status: "ok" }] } }).jobs.y, [{ startedAt: 7, finishedAt: 8, status: "ok" }]);
  fs.rmSync(dir, { recursive: true, force: true });
});

test("persistence is atomic: tmp + rename, no tmp left behind, a failed rename leaves the old file intact, beat never throws", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "hb-atomic-"));
  const file = path.join(dir, "hb.json");
  const s = createHeartbeatStore(file, () => 1);
  s.beat("position_sync", "finish", "ok");
  const good = fs.readFileSync(file, "utf8");
  assert.deepEqual(fs.readdirSync(dir), ["hb.json"], "no tmp file survives a write");
  const realRename = fs.renameSync;
  const origError = console.error;
  try {
    (fs as { renameSync: typeof fs.renameSync }).renameSync = () => { throw new Error("EIO simulated"); };
    console.error = () => {};
    assert.doesNotThrow(() => s.beat("position_sync", "finish", "error", "after"));
  } finally {
    (fs as { renameSync: typeof fs.renameSync }).renameSync = realRename;
    console.error = origError;
  }
  assert.equal(fs.readFileSync(file, "utf8"), good, "the reader still sees the last complete file");
  assert.deepEqual(fs.readdirSync(dir), ["hb.json"], "the failed tmp is cleaned up");
  assert.match(String(s.lastWriteError()), /EIO/);
  fs.rmSync(dir, { recursive: true, force: true });
});

test("withJobBeat passes the result and the exception through unchanged", async () => {
  const prev = process.env.DATA_DIR;
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "hb-with-"));
  process.env.DATA_DIR = dir;
  try {
    assert.equal(await withJobBeat("unit_ok", async () => 42), 42);
    await assert.rejects(withJobBeat("unit_err", async () => { throw new Error("kaput"); }), /kaput/);
  } finally {
    if (prev === undefined) delete process.env.DATA_DIR; else process.env.DATA_DIR = prev;
  }
  // the default store resolves its dir once; whichever dir it picked, the beats are in memory
  fs.rmSync(dir, { recursive: true, force: true });
});

// ── wiring pins ─────────────────────────────────────────────────────────────

test("every registered job has start AND finish beats in its source file; every beat names a registered job", () => {
  for (const s of JOB_REGISTRY) {
    const src = read(s.source);
    const viaWrapper = src.includes(`withJobBeat("${s.id}"`);
    assert.ok(viaWrapper || src.includes(`beat("${s.id}", "start")`), `${s.id}: no start beat in ${s.source}`);
    assert.ok(viaWrapper || src.includes(`beat("${s.id}", "finish"`), `${s.id}: no finish beat in ${s.source}`);
  }
  const ids = new Set(JOB_REGISTRY.map((s) => s.id));
  for (const f of ["server/bot.ts", "server/optionsChainArchive.ts"]) {
    const src = read(f);
    for (const m of src.matchAll(/(?:\bbeat|noteJobError|withJobBeat)\("([a-z0-9_]+)"/g)) {
      assert.ok(ids.has(m[1]), `${f}: beat for unregistered job "${m[1]}" (typo, or add it to JOB_REGISTRY)`);
    }
  }
});

test("bot.ts wiring: Tier 1 tick beats in a finally, Tier 2 finish follows the back-off failure flag, Tier 3 both call sites wrapped", () => {
  const bot = read("server/bot.ts");
  const t1 = bot.slice(bot.indexOf("// TIER 1: Reflex (every 45 seconds)"), bot.indexOf("// TIER 2: Intelligence"));
  assert.ok(t1.indexOf('beat("tier1_reflex", "start")') < t1.indexOf("await tier1Reflex();"));
  assert.match(t1, /\} finally \{[\s\S]*beat\("tier1_reflex", "finish"/);
  const t2 = bot.slice(bot.indexOf("async function scheduleTier2()"), bot.indexOf("// Start Tier 2 with adaptive scheduling"));
  assert.ok(t2.indexOf('beat("tier2_scan", "start")') < t2.indexOf("await tier2Intelligence("));
  assert.match(t2, /const failed = schedulerThrew \|\| tier2LastScanFailed;\s*if \(t2Started\) \{\s*beat\("tier2_scan", "finish", failed \? "error" : "ok"/);
  assert.ok(bot.includes('await withJobBeat("tier3_strategic", tier3Strategic);'), "hourly Tier 3");
  assert.ok(bot.includes('withJobBeat("tier3_strategic", tier3Strategic).catch('), "startup Tier 3");
  assert.ok(bot.includes('pythonRpc("health", {}, 10000)'), "daemon liveness probe uses the read-only health RPC");
});

test("diag probe `heartbeats` is whitelisted, wired, sanitized and read-only", () => {
  assert.ok((DIAG_PROBES as readonly string[]).includes("heartbeats"));
  const bot = read("server/bot.ts");
  const start = bot.indexOf('case "heartbeats": {');
  assert.ok(start > 0, "bot.ts case missing");
  const block = bot.slice(start, bot.indexOf("default:", start));
  assert.ok(block.includes("jobHeartbeatReport()"));
  assert.ok(block.includes("sanitizeDiag("));
  assert.ok(!/alpaca\(|execPython|pythonRpc|\.beat\(|beat\(/.test(block), "the probe only reads");
});

test("/api/health carries the jobs block but it is informational only: no status flip, HTTP code still from healthGate", () => {
  const bot = read("server/bot.ts");
  const start = bot.indexOf('app.get("/api/health"');
  const handler = bot.slice(start, bot.indexOf("app.get(", start + 10));
  const jobsAt = handler.indexOf("checks.checks.jobs = healthJobsBlock(");
  assert.ok(jobsAt > 0, "jobs block missing from /api/health");
  const jobsBlock = handler.slice(jobsAt - 600, handler.indexOf("// Check 6:", jobsAt));
  assert.ok(!jobsBlock.includes('checks.status = "degraded"'), "the jobs block must never flip the top-level status");
  assert.deepEqual([...SERVING_CHECKS], ["server", "database"], "SERVING_CHECKS unchanged");
  const payload = { status: "degraded", checks: { server: { status: "ok" }, database: { status: "ok" }, jobs: { status: "degraded" } } };
  assert.equal(healthHttpCode(payload), 200, "a degraded jobs block never fails the deploy gate");
});

test("calendar: market_calendar.py half days parse through the existing helper (holidays unchanged)", () => {
  const py = read("market_calendar.py");
  const half = parseMarketHolidays(py, "HALF_DAYS");
  assert.ok(half.has("2026-11-27") && half.has("2026-12-24"));
  const hol = parseMarketHolidays(py);
  assert.ok(hol.has("2026-11-26") && !hol.has("2026-11-27"), "default still reads MARKET_HOLIDAYS only");
});

test("classifyJobError: closed enum, health block names the cause but never the text", () => {
  assert.equal(classifyJobError("exceeded the 180-minute budget — process group killed"), "timeout");
  assert.equal(classifyJobError("RSS 950 MB > budget 900 MB after 400 series"), "memory_budget");
  assert.equal(classifyJobError("unparseable summary: Traceback (JSON)"), "unparseable_output");
  assert.equal(classifyJobError("code=1 signal=none: boom"), "nonzero_exit");
  assert.equal(classifyJobError("interrupted: the process exited before this run finished"), "interrupted");
  assert.equal(classifyJobError(undefined), "other");
  const file: HeartbeatFile = { version: 1, jobs: {
    tier2_scan: [{ startedAt: TUE_10ET - 2 * MIN, finishedAt: TUE_10ET - MIN, status: "error", detail: "RSS 950 MB > budget 900 MB /secret/path" }],
  } };
  const h = healthJobsBlock(evaluate(TUE_10ET, file, JOB_REGISTRY, ctx));
  assert.equal(h.error_class["tier2_scan"], "memory_budget");
  assert.ok(!JSON.stringify(h).includes("/secret/path"));
});
