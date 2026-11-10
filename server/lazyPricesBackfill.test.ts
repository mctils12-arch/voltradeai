// lazyPricesBackfill — nightly job + read-only status for lazy_prices_backfill.py
// (the Lazy Prices gate-2 INPUT). Run: npx tsx --test server/lazyPricesBackfill.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  lazyPricesBackfillFlag, lazyPricesBackfillCmd, lazyPricesBackfillPairsDir, lazyPricesBackfillTmpArchive, inLazyPricesBackfillWindow,
  decideLazyPricesBackfill, parseLazyPricesBackfillSummary, runLazyPricesBackfill, lazyPricesBackfillStatus,
  LAZY_PRICES_BACKFILL_TIMEOUT_MS, LAZY_PRICES_BACKFILL_BUDGET_S, LAZY_PRICES_BACKFILL_MIN_FREE_BYTES,
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

test("window is 21:00-23:00 ET any day; the 4h kill ends before the 04:00 ET daily branch; disjoint from the Sunday gate-2 slot", () => {
  assert.equal(inLazyPricesBackfillWindow(TUE_IN), true);
  assert.equal(inLazyPricesBackfillWindow(Date.parse("2026-11-11T01:30:00Z")), false);   // 20:30 ET
  assert.equal(inLazyPricesBackfillWindow(Date.parse("2026-11-11T04:30:00Z")), false);   // 23:30 ET
  assert.equal(inLazyPricesBackfillWindow(Date.parse("2026-11-15T02:30:00Z")), true);    // Saturday night too
  // latest start 22:59 ET + 4 h hard kill = 02:59 ET, before the 04:00 ET branch
  assert.ok(23 * 60 + LAZY_PRICES_BACKFILL_TIMEOUT_MS / 60_000 <= 28 * 60);
  assert.ok(LAZY_PRICES_BACKFILL_BUDGET_S * 1000 < LAZY_PRICES_BACKFILL_TIMEOUT_MS, "Python budget must end before the kill");
  // the Sunday 16:00-18:00 ET weekly gate-2 run never coincides with a backfill start
  const sun16 = Date.parse("2026-11-15T21:00:00Z");
  assert.equal(inLazyPricesWindow(sun16), true);
  assert.equal(inLazyPricesBackfillWindow(sun16), false);
  assert.ok(LAZY_PRICES_WINDOW_START_MIN < 21 * 60);
});

test("decide: idle outside/disabled/already-attempted; skip on low disk; otherwise run (no freshness skip)", () => {
  assert.equal(decideLazyPricesBackfill(TUE_IN, { ...base, enabled: false }).action, "idle");
  assert.equal(decideLazyPricesBackfill(Date.parse("2026-11-11T01:30:00Z"), base).action, "idle");
  const run = decideLazyPricesBackfill(TUE_IN, base);
  assert.equal(run.action, "run");
  assert.equal((run as { day: string }).day, "2026-11-10");                 // the ET day, not the UTC one
  assert.equal(decideLazyPricesBackfill(TUE_IN, { ...base, lastAttemptDay: "2026-11-10" }).action, "idle");
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
  assert.match((await runLazyPricesBackfill({}, killed)).detail, /hard budget — process group killed/);
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
});
