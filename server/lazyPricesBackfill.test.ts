// lazyPricesBackfill — nightly job + read-only status for lazy_prices_backfill.py
// (the Lazy Prices gate-2 INPUT). Run: npx tsx --test server/lazyPricesBackfill.test.ts
import { test, mock } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  lazyPricesBackfillFlag, lazyPricesBackfillCmd, lazyPricesBackfillPairsDir, lazyPricesBackfillTmpArchive, inLazyPricesBackfillWindow,
  lazyPricesBackfillNightOf, lazyPricesBackfillNightEndMs, lazyPricesBackfillBudgetSeconds, lazyPricesBackfillTimeoutMs,
  decideLazyPricesBackfill, parseLazyPricesBackfillSummary, runLazyPricesBackfill, lazyPricesBackfillStatus, bootLazyPricesBackfill,
  LAZY_PRICES_BACKFILL_TIMEOUT_MS, LAZY_PRICES_BACKFILL_BUDGET_S, LAZY_PRICES_BACKFILL_MIN_FREE_BYTES,
  LAZY_PRICES_BACKFILL_KILL_GRACE_MS, LAZY_PRICES_BACKFILL_NIGHT_END_MIN, LAZY_PRICES_BACKFILL_MIN_SLICE_S,
  LAZY_PRICES_BACKFILL_TICK_MS, LAZY_PRICES_BACKFILL_FIRST_TICK_MS,
} from "./lazyPricesBackfill";
import { lazyPricesPairsDir, lazyPricesStatus, LAZY_PRICES_WINDOW_START_MIN, inLazyPricesWindow } from "./lazyPrices";
import { WALKFORWARD_ENV_CAPS, type WalkforwardExec } from "./walkforward";
import { JOB_REGISTRY } from "./jobHeartbeat";
import { sanitizeDiag } from "./diag";

const here = path.dirname(fileURLToPath(import.meta.url));
// 2026-11-10 is a Tuesday; 02:30Z (Nov 11) = 21:30 ET on Nov 10 (EST)
const TUE_IN = Date.parse("2026-11-11T02:30:00Z");
const base = { enabled: true, lastAttemptDay: "", freeBytes: 5e9 as number | null };

test("flag: env present wins; registry enabledWhen agrees", () => {
  const job = JOB_REGISTRY.find((j) => j.id === "lazy_prices_backfill_nightly");
  assert.ok(job && job.kind === "interval" && job.source === "server/lazyPricesBackfill.ts");
  assert.ok(job!.maxRunMs! >= LAZY_PRICES_BACKFILL_TIMEOUT_MS, "registry maxRun must cover the hard kill");
  for (const [v, on] of [[undefined, true], ["1", true], ["0", false], ["off", false], ["", false]] as const) {
    const env = v === undefined ? {} : { VOLTRADE_LAZY_PRICES_BACKFILL: v };
    assert.equal(lazyPricesBackfillFlag(env).on, on);
    assert.equal(job!.enabledWhen!(env), on);
  }
});

test("window: the night opens 21:00 ET any day and a slice may start until 03:10 ET; nothing starts in the daytime or at 20:30 ET", () => {
  assert.equal(inLazyPricesBackfillWindow(TUE_IN), true);
  assert.equal(inLazyPricesBackfillWindow(Date.parse("2026-11-11T01:30:00Z")), false);   // 20:30 ET
  assert.equal(inLazyPricesBackfillWindow(Date.parse("2026-11-11T04:30:00Z")), true);    // 23:30 ET — the 2026-11-10 night's lost case
  assert.equal(inLazyPricesBackfillWindow(Date.parse("2026-11-11T06:00:00Z")), true);    // 01:00 ET
  assert.equal(inLazyPricesBackfillWindow(Date.parse("2026-11-11T08:09:00Z")), true);    // 03:09 ET: a 21-min slice still fits
  assert.equal(inLazyPricesBackfillWindow(Date.parse("2026-11-11T08:10:00Z")), false);   // 03:10 ET: 20 min left is not a slice
  assert.equal(inLazyPricesBackfillWindow(Date.parse("2026-11-11T08:30:00Z")), false);   // 03:30 ET
  assert.equal(inLazyPricesBackfillWindow(Date.parse("2026-11-11T18:00:00Z")), false);   // 13:00 ET
  assert.equal(inLazyPricesBackfillWindow(Date.parse("2026-11-15T02:30:00Z")), true);    // Saturday night too
  // the Sunday 16:00-18:00 ET weekly gate-2 run never coincides with a backfill start
  const sun16 = Date.parse("2026-11-15T21:00:00Z");
  assert.equal(inLazyPricesWindow(sun16), true);
  assert.equal(inLazyPricesBackfillWindow(sun16), false);
  assert.ok(LAZY_PRICES_WINDOW_START_MIN < 21 * 60);
});

test("night: the small hours belong to the night that opened the evening before; its end is 03:30 ET the next morning", () => {
  assert.equal(lazyPricesBackfillNightOf(TUE_IN), "2026-11-10");                              // 21:30 ET Nov 10
  assert.equal(lazyPricesBackfillNightOf(Date.parse("2026-11-11T06:00:00Z")), "2026-11-10");  // 01:00 ET Nov 11 -> still Nov 10's night
  assert.equal(lazyPricesBackfillNightOf(Date.parse("2026-11-12T02:05:00Z")), "2026-11-11");  // 21:05 ET Nov 11
  const end = lazyPricesBackfillNightEndMs(TUE_IN);
  assert.equal(end, Date.parse("2026-11-11T08:30:00Z"));                                      // 03:30 EST Nov 11
  assert.equal(lazyPricesBackfillNightEndMs(Date.parse("2026-11-11T06:00:00Z")), end, "same night, same end");
  // the night DST ends (2026-11-01, 02:00 EDT -> 01:00 EST): 21:00 EDT Oct 31 is 01:00Z; 03:30 EST Nov 1 is 08:30Z
  assert.equal(lazyPricesBackfillNightEndMs(Date.parse("2026-11-01T01:30:00Z")), Date.parse("2026-11-01T08:30:00Z"));
  assert.equal(lazyPricesBackfillNightOf(Date.parse("2026-11-01T06:30:00Z")), "2026-10-31");  // 01:30 EST Nov 1, after the repeat hour
});

test("budget: the 3.5 h cap early in the night, what is left of it later; the kill is budget + 30 min and never past 4 h; everything ends before the 04:00 ET daily branch", () => {
  assert.equal(lazyPricesBackfillBudgetSeconds(TUE_IN), LAZY_PRICES_BACKFILL_BUDGET_S);                      // 21:30 ET: 6 h left, capped
  assert.equal(lazyPricesBackfillBudgetSeconds(Date.parse("2026-11-11T05:00:00Z")), LAZY_PRICES_BACKFILL_BUDGET_S); // 00:00 ET: exactly 3.5 h left
  assert.equal(lazyPricesBackfillBudgetSeconds(Date.parse("2026-11-11T06:00:00Z")), 2.5 * 3600);              // 01:00 ET
  assert.equal(lazyPricesBackfillBudgetSeconds(Date.parse("2026-11-11T08:10:00Z")), LAZY_PRICES_BACKFILL_MIN_SLICE_S); // 03:10 ET: the floor, not started
  assert.equal(lazyPricesBackfillBudgetSeconds(Date.parse("2026-11-11T09:00:00Z")), 0);                       // 04:00 ET: nothing left (never negative)
  assert.equal(lazyPricesBackfillTimeoutMs(2.5 * 3600), 2.5 * 3600_000 + LAZY_PRICES_BACKFILL_KILL_GRACE_MS);
  assert.equal(lazyPricesBackfillTimeoutMs(), LAZY_PRICES_BACKFILL_TIMEOUT_MS);
  assert.equal(lazyPricesBackfillTimeoutMs(10 * 3600), LAZY_PRICES_BACKFILL_TIMEOUT_MS, "never past the 4 h maximum");
  assert.equal(LAZY_PRICES_BACKFILL_TIMEOUT_MS, 4 * 60 * 60_000);
  assert.ok(LAZY_PRICES_BACKFILL_BUDGET_S * 1000 < LAZY_PRICES_BACKFILL_TIMEOUT_MS, "Python budget must end before the kill");
  // latest budget end 03:30 ET + 30 min grace = 04:00 ET, the daily branch — never after it
  assert.ok((LAZY_PRICES_BACKFILL_NIGHT_END_MIN + LAZY_PRICES_BACKFILL_KILL_GRACE_MS / 60_000) <= 4 * 60);
  // the longest slice from the night's opening ends 00:30 ET, well inside the night
  assert.ok(21 * 60 + LAZY_PRICES_BACKFILL_BUDGET_S / 60 <= 24 * 60 + LAZY_PRICES_BACKFILL_NIGHT_END_MIN);
});

test("decide: idle outside/disabled/already-attempted; skip on low disk; otherwise run (no freshness skip)", () => {
  assert.equal(decideLazyPricesBackfill(TUE_IN, { ...base, enabled: false }).action, "idle");
  assert.equal(decideLazyPricesBackfill(Date.parse("2026-11-11T01:30:00Z"), base).action, "idle");
  const run = decideLazyPricesBackfill(TUE_IN, base);
  assert.equal(run.action, "run");
  assert.equal((run as { day: string }).day, "2026-11-10");                 // the ET night, not the UTC day
  assert.equal((run as { budgetS: number }).budgetS, LAZY_PRICES_BACKFILL_BUDGET_S);
  assert.equal(decideLazyPricesBackfill(TUE_IN, { ...base, lastAttemptDay: "2026-11-10" }).action, "idle");
  // the restart case: a fresh process (lastAttemptDay "") at 01:00 ET re-attempts the SAME night with what is left of it
  const late = decideLazyPricesBackfill(Date.parse("2026-11-11T06:00:00Z"), base);
  assert.equal(late.action, "run");
  assert.equal((late as { day: string }).day, "2026-11-10");
  assert.equal((late as { budgetS: number }).budgetS, 2.5 * 3600);
  assert.match(late.reason, /150 min of budget until 03:30 ET/);
  // ...but a process that already attempted that night does not run twice in it
  assert.equal(decideLazyPricesBackfill(Date.parse("2026-11-11T06:00:00Z"), { ...base, lastAttemptDay: "2026-11-10" }).action, "idle");
  assert.equal(decideLazyPricesBackfill(Date.parse("2026-11-11T08:10:00Z"), base).action, "idle", "03:10 ET: too little night left");
  assert.equal(decideLazyPricesBackfill(TUE_IN, { ...base, freeBytes: LAZY_PRICES_BACKFILL_MIN_FREE_BYTES - 1 }).action, "skip");
  assert.equal(decideLazyPricesBackfill(TUE_IN, { ...base, freeBytes: 1_020_000_000 }).action, "run", "today's live free space (~1.02 GB) must not block");
  assert.equal(decideLazyPricesBackfill(TUE_IN, { ...base, freeBytes: null }).action, "run");
});

test("run: own long-job lane — budget flag, SIGKILL group, thread caps, parsed summaries, version never interpolated unchecked", async () => {
  const calls: Array<{ cmd: string; opts: Record<string, unknown> }> = [];
  const ok: WalkforwardExec = async (cmd, opts) => {
    calls.push({ cmd, opts: opts as Record<string, unknown> });
    return { stdout: 'noise\n{"status":"ok","phase":"score","processed":1200,"cursor_day":"2016-03-01","cursor_closed":true,"remaining_in_plan":48000,"plan_filings":51000,"stop_reason":"budget","counts":{"rows":900},"elapsed_s":12600.2,"peak_rss_mb":410}\n', stderr: "" };
  };
  const r = await runLazyPricesBackfill({ PATH: "/usr/bin" }, ok);
  assert.equal(r.status, "ok");
  assert.match(r.detail, /scored 1200 filings \(900 rows total\) to 2016-03-01; stop: budget; 48000 left/);
  assert.match(calls[0].cmd, /^python3 lazy_prices_backfill\.py --run --nice 19 --budget-seconds \d+ --max-rss-mb \d+ --code-version \S+ --pairs-dir \S+ --tmp-archive \S+$/);
  assert.equal(calls[0].opts.timeout, LAZY_PRICES_BACKFILL_TIMEOUT_MS);
  assert.equal(calls[0].opts.killSignal, "SIGKILL");
  for (const [k, v] of Object.entries(WALKFORWARD_ENV_CAPS)) assert.equal((calls[0].opts.env as Record<string, string>)[k], v);
  const killed: WalkforwardExec = async () => { throw Object.assign(new Error("x"), { killed: true }); };
  assert.match((await runLazyPricesBackfill({}, killed)).detail, /exceeded the 240-minute hard budget — process group killed/);
  // a late slice: the Python gets the clamped budget and the kill follows it by the grace, not the 4 h maximum
  calls.length = 0;
  await runLazyPricesBackfill({ PATH: "/usr/bin" }, ok, 2.5 * 3600);
  assert.match(calls[0].cmd, /--budget-seconds 9000 /);
  assert.equal(calls[0].opts.timeout, 2.5 * 3600_000 + LAZY_PRICES_BACKFILL_KILL_GRACE_MS);
  assert.match((await runLazyPricesBackfill({}, killed, 2.5 * 3600)).detail, /exceeded the 180-minute hard budget/);
  assert.match(lazyPricesBackfillCmd("1.0.1", {}, 1234.9), /--budget-seconds 1234 /, "whole seconds on the command line");
  assert.match(lazyPricesBackfillCmd("1.0.1", {}, 99 * 3600), new RegExp(`--budget-seconds ${LAZY_PRICES_BACKFILL_BUDGET_S} `), "never above the cap");
  assert.equal(parseLazyPricesBackfillSummary('{"status":"skipped","reason":"only 200 MB free on the volume (< 268 MB)"}').status, "skipped");
  const plan = parseLazyPricesBackfillSummary('{"status":"ok","phase":"plan","plan_complete":false,"plan_ciks_listed":4000,"plan_ciks_total":9000,"plan_filings":21000,"elapsed_s":12600}');
  assert.equal(plan.status, "ok");
  assert.match(plan.detail, /plan phase: 4000\/9000 CIKs/);
  assert.equal(parseLazyPricesBackfillSummary("Traceback").status, "error");
  assert.equal(parseLazyPricesBackfillSummary('{"status":"error","reason":"ValueError: x"}').detail, "ValueError: x");
  assert.equal(lazyPricesBackfillCmd("1.0.9; rm -rf /"), lazyPricesBackfillCmd("unknown"));
});

test("scratch dirs: container tmp by default (never the volume), pairs cache shared with the weekly runner, safe overrides only", () => {
  const p = lazyPricesBackfillPairsDir({});
  const t = lazyPricesBackfillTmpArchive({});
  assert.ok(!p.startsWith("/data") && !t.startsWith("/data"), `${p} ${t}`);
  assert.equal(p, lazyPricesPairsDir({}), "one pairs cache for both jobs — the zips are downloaded once");
  assert.notEqual(p, t);
  assert.equal(lazyPricesBackfillTmpArchive({ VOLTRADE_LAZY_PRICES_TMP_ARCHIVE: "/scratch/t" }), "/scratch/t");
  assert.equal(lazyPricesBackfillTmpArchive({ VOLTRADE_LAZY_PRICES_TMP_ARCHIVE: "/x; rm -rf /" }), t, "unsafe override ignored");
  assert.match(lazyPricesBackfillCmd("1.0.1", {}), new RegExp(`--tmp-archive ${t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}$`));
});

test("status: headline fields only, embedded in the lazy_prices probe, survives sanitizeDiag; empty dir => nulls", () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "lpb-"));
  try {
    const root = path.join(tmp, "lazy_prices", "backfill");
    const empty = lazyPricesBackfillStatus({ DATA_DIR: tmp }, root);
    assert.equal(empty.last_run, null);
    assert.equal(empty.plan, null);
    assert.equal(empty.rows_months, 0);
    fs.mkdirSync(path.join(root, "rows"), { recursive: true });
    fs.writeFileSync(path.join(root, "rows", "2016-03.jsonl.gz"), "");
    fs.writeFileSync(path.join(root, "plan_meta.json"), JSON.stringify({ complete: true, filings: 51000, done_ciks: ["1", "2"], completed_at: "2026-11-11T02:40:00+00:00", list_errors: 3 }));
    fs.writeFileSync(path.join(root, "status.json"), JSON.stringify({
      status: "ok", phase: "score", finished_at: "2026-11-11T06:10:00+00:00", code_version: "1.0.1346", processed: 1200, cursor_day: "2016-03-01",
      cursor_closed: true, remaining_in_plan: 48000, plan_filings: 51000, stop_reason: "budget", counts: { rows: 900, failed: 2 }, io_noise: {},
      http: { requests: 2600, retries: 1, errors: 0 }, elapsed_s: 12600.2, peak_rss_mb: 410, secret_extra: "0000320193",
    }));
    const st = lazyPricesBackfillStatus({ DATA_DIR: tmp }, root);
    const last = st.last_run as Record<string, any>;
    assert.equal(last.cursor_day, "2016-03-01");
    assert.equal(last.counts.rows, 900);
    assert.equal((st.plan as Record<string, any>).ciks, 2);
    assert.equal(st.rows_months, 1);
    assert.ok(!JSON.stringify(st).includes("0000320193"));
    assert.deepEqual(sanitizeDiag(st), st);
    const probe = lazyPricesStatus({ DATA_DIR: tmp }, path.join(tmp, "lazy_prices"), path.join(tmp, "pp"));
    assert.deepEqual(probe.backfill, st);
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});

test("wiring: booted from routes.ts, no mutex/daemon/order path, never ticks on boot, heartbeat start+finish", () => {
  assert.ok(fs.readFileSync(path.join(here, "routes.ts"), "utf8").includes("bootLazyPricesBackfill();"));
  const src = fs.readFileSync(path.join(here, "lazyPricesBackfill.ts"), "utf8");
  const boot = src.slice(src.indexOf("export function bootLazyPricesBackfill"));
  assert.ok(!/^\s*(void\s+)?tick\(\);/m.test(boot));
  assert.ok(boot.includes("setInterval(") && boot.includes(".unref"));
  assert.ok(!/execPythonSerialized|pythonRunning|pythonRpc/.test(src));
  assert.ok(!src.includes("/orders") && !/method:\s*"(POST|PUT|PATCH|DELETE)"/.test(src));
  assert.ok(src.includes('beat("lazy_prices_backfill_nightly", "start")') && src.includes('beat("lazy_prices_backfill_nightly", "finish"'));
  // the tick passes the decision's clamped budget to the run (a late restart must not get the full 3.5 h)
  assert.ok(boot.includes("runLazyPricesBackfill(env, exec, d.budgetS)"));
  assert.ok(boot.includes("exec: WalkforwardExec = execGroupKill"), "the long-job lane is the default; the seam is for tests");
  // the first tick is a short timeout after boot, then the interval — never one interval after boot (three nights lost to that)
  assert.ok(boot.includes("setTimeout(() => { void tick(); }, LAZY_PRICES_BACKFILL_FIRST_TICK_MS)"));
  assert.ok(LAZY_PRICES_BACKFILL_FIRST_TICK_MS >= 60_000, "let boot settle (archive loads, heartbeat restore) before a nice-19 Python starts");
  assert.ok(LAZY_PRICES_BACKFILL_FIRST_TICK_MS <= 5 * 60_000, "inside the 11-29 min container lifetimes measured on the night of 11-11/12");
  assert.ok(LAZY_PRICES_BACKFILL_FIRST_TICK_MS < LAZY_PRICES_BACKFILL_TICK_MS);
  // the attempt marker stays in process memory on purpose: a redeploy's fresh process re-attempts the night
  assert.ok(/let lastAttemptDay = "";/.test(boot) && !/lastAttemptDay.*(readFileSync|writeFileSync)/.test(boot));
});

test("boot: a container that lives only minutes still starts the night — the first tick lands FIRST_TICK_MS after boot (not one 30-min interval later), one run at a time, once per night", async () => {
  // Would have failed on the pre-2026-11-12 code: with the first tick one interval after boot, a 25-minute-lived
  // container never ran the night's slice, and 25 such containers lost the night of 11-11/12 entirely.
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "lpb-boot-"));
  const calls: string[] = [];
  let release!: () => void;
  const exec: WalkforwardExec = async (cmd) => {
    calls.push(cmd);
    await new Promise<void>((r) => { release = r; });
    return { stdout: '{"status":"ok","phase":"score","processed":48,"cursor_day":"2015-02-03","cursor_closed":false,"remaining_in_plan":71000,"plan_filings":71747,"stop_reason":"budget","counts":{"rows":30},"elapsed_s":12600,"peak_rss_mb":400}\n', stderr: "" };
  };
  mock.timers.enable({ apis: ["setTimeout", "setInterval", "Date"], now: TUE_IN });   // boot at 21:30 ET, inside the night
  try {
    bootLazyPricesBackfill({ DATA_DIR: tmp, PATH: "/usr/bin" }, exec);
    assert.equal(calls.length, 0, "never ticks synchronously on boot");
    mock.timers.tick(LAZY_PRICES_BACKFILL_FIRST_TICK_MS - 1);
    assert.equal(calls.length, 0, "not before the first-tick delay");
    mock.timers.tick(1);
    assert.equal(calls.length, 1, "the slice starts FIRST_TICK_MS after boot");
    assert.match(calls[0], new RegExp(`--budget-seconds ${LAZY_PRICES_BACKFILL_BUDGET_S} `), "21:32 ET: the full 3.5 h");
    mock.timers.tick(LAZY_PRICES_BACKFILL_TICK_MS);
    assert.equal(calls.length, 1, "the interval tick while a run is in flight starts nothing");
    release();
    await new Promise<void>((r) => setImmediate(r));          // setImmediate is not mocked: lets the run's finish beat land
    mock.timers.tick(LAZY_PRICES_BACKFILL_TICK_MS);
    assert.equal(calls.length, 1, "the same night is not attempted twice by one process");
    // the second boot in one process is a no-op (module guard), so a test cannot observe a second timer chain
    bootLazyPricesBackfill({ DATA_DIR: tmp, PATH: "/usr/bin" }, exec);
    mock.timers.tick(LAZY_PRICES_BACKFILL_FIRST_TICK_MS);
    assert.equal(calls.length, 1);
  } finally {
    mock.timers.reset();
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});
