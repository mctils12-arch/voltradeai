// filings13fClusters — nightly job + read-only status for filings13f_clusters.py (repo root):
// the 13F new-position cluster detector (ladder root sec_edgar_13f_institutional_clustering).
// Run: npx tsx --test server/filings13fClusters.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  filings13fClustersFlag, filings13fClustersCmd, inFilings13fClustersWindow, decideFilings13fClusters,
  parseFilings13fClustersSummary, runFilings13fClusters, filings13fClustersStatus, filings13fClustersLastFinishedAt,
  FILINGS13F_CLUSTERS_JOB, FILINGS13F_CLUSTERS_TIMEOUT_MS, FILINGS13F_CLUSTERS_MIN_FREE_BYTES,
  FILINGS13F_CLUSTERS_WINDOW_START_MIN, FILINGS13F_CLUSTERS_WINDOW_END_MIN,
} from "./filings13fClusters";
import { WALKFORWARD_ENV_CAPS, type WalkforwardExec } from "./walkforward";
import { JOB_REGISTRY } from "./jobHeartbeat";
import { sanitizeDiag, DIAG_PROBES } from "./diag";

const here = path.dirname(fileURLToPath(import.meta.url));
// 2026-11-23 10:20Z = 05:20 ET (EST), a Monday
const IN = Date.parse("2026-11-23T10:20:00Z");
const base = { enabled: true, lastFinishedAt: null as string | null, lastAttemptDay: "", freeBytes: 5e9 as number | null };

test("flag: env present wins; the registry's enabledWhen agrees", () => {
  const job = JOB_REGISTRY.find((j) => j.id === FILINGS13F_CLUSTERS_JOB);
  assert.ok(job && job.kind === "daily" && job.source === "server/filings13fClusters.ts");
  assert.ok(job!.maxRunMs >= FILINGS13F_CLUSTERS_TIMEOUT_MS, "registry maxRunMs must cover the job budget");
  if (job!.kind === "daily") {
    assert.equal(job!.tz, "America/New_York");
    assert.equal(job!.dueMin, FILINGS13F_CLUSTERS_WINDOW_START_MIN, "due when the window opens");
    assert.equal(job!.tradingDaysOnly, false, "filings arrive on weekdays; the archive's backfill runs every night");
    // the window's last tick plus the budget lands inside due + grace
    assert.ok(job!.graceMs >= (FILINGS13F_CLUSTERS_WINDOW_END_MIN - FILINGS13F_CLUSTERS_WINDOW_START_MIN) * 60_000);
  }
  for (const [v, on] of [[undefined, true], ["1", true], ["0", false], ["off", false], ["", false]] as const) {
    const env = v === undefined ? {} : { VOLTRADE_13F_CLUSTERS: v };
    assert.equal(filings13fClustersFlag(env).on, on);
    assert.equal(job!.enabledWhen!(env), on);
  }
});

test("window is 05:00-06:00 ET on any day: after the XBRL read, before the 13F completeness pass, never in market hours", () => {
  assert.equal(inFilings13fClustersWindow(IN), true);
  assert.equal(inFilings13fClustersWindow(Date.parse("2026-11-22T10:20:00Z")), true, "Sunday too");
  assert.equal(inFilings13fClustersWindow(Date.parse("2026-11-23T09:59:00Z")), false);  // 04:59 ET
  assert.equal(inFilings13fClustersWindow(Date.parse("2026-11-23T11:00:00Z")), false);  // 06:00 ET
  assert.equal(inFilings13fClustersWindow(Date.parse("2026-07-06T09:20:00Z")), true, "05:20 EDT in summer");
  assert.ok(FILINGS13F_CLUSTERS_WINDOW_END_MIN * 60_000 + FILINGS13F_CLUSTERS_TIMEOUT_MS <= (9 * 60 + 30) * 60_000, "done before the open");
});

test("decide: one attempt per ET day, also across a restart and after an error; skip on low disk", () => {
  assert.equal(decideFilings13fClusters(IN, { ...base, enabled: false }).action, "idle");
  assert.equal(decideFilings13fClusters(Date.parse("2026-11-23T15:00:00Z"), base).action, "idle");
  const run = decideFilings13fClusters(IN, base);
  assert.equal(run.action, "run");
  assert.equal((run as { day: string }).day, "2026-11-23");
  assert.equal(decideFilings13fClusters(IN, { ...base, lastAttemptDay: "2026-11-23" }).action, "idle");
  // a new container 10 minutes after the day's run: last_run.json says it already happened
  assert.equal(decideFilings13fClusters(IN, { ...base, lastFinishedAt: "2026-11-23T10:05:00+00:00" }).action, "idle");
  // yesterday's run does not block today's
  assert.equal(decideFilings13fClusters(IN, { ...base, lastFinishedAt: "2026-11-22T10:05:00+00:00" }).action, "run");
  // 03:00Z is still the previous ET day
  assert.equal(decideFilings13fClusters(IN, { ...base, lastFinishedAt: "2026-11-23T03:00:00+00:00" }).action, "run");
  assert.equal(decideFilings13fClusters(IN, { ...base, lastFinishedAt: "not a date" }).action, "run");
  assert.equal(decideFilings13fClusters(IN, { ...base, freeBytes: FILINGS13F_CLUSTERS_MIN_FREE_BYTES - 1 }).action, "skip");
  assert.equal(decideFilings13fClusters(IN, { ...base, freeBytes: 1_000_000_000 }).action, "run", "typical live free space must not block");
  assert.equal(decideFilings13fClusters(IN, { ...base, freeBytes: null }).action, "run");
});

test("run: the long-job lane — budget, SIGKILL group, thread caps, parsed summary, version never interpolated unchecked", async () => {
  const calls: Array<{ cmd: string; opts: Record<string, unknown> }> = [];
  const ok: WalkforwardExec = async (cmd, opts) => {
    calls.push({ cmd, opts: opts as Record<string, unknown> });
    return { stdout: 'noise\n{"status":"ok","pairs":1,"full_filings":9120,"clusters":2400,"thin_clusters":310,"first_seen_appended":12,"elapsed_s":41.2,"peak_rss_mb":288}\n', stderr: "" };
  };
  const r = await runFilings13fClusters({ PATH: "/usr/bin" }, ok);
  assert.equal(r.status, "ok");
  assert.match(r.detail, /9120 filings with tables, 1 quarter pair\(s\), 2400 clusters \(310 thinly held\), 12 new to the record/);
  assert.match(calls[0].cmd, /^python3 filings13f_clusters\.py --run --nice 19 --max-rss-mb \d+ --code-version \S+$/);
  assert.equal(calls[0].opts.timeout, FILINGS13F_CLUSTERS_TIMEOUT_MS);
  assert.equal(calls[0].opts.killSignal, "SIGKILL");
  for (const [k, v] of Object.entries(WALKFORWARD_ENV_CAPS)) assert.equal((calls[0].opts.env as Record<string, string>)[k], v);
  const killed: WalkforwardExec = async () => { throw Object.assign(new Error("x"), { killed: true }); };
  assert.match((await runFilings13fClusters({}, killed)).detail, /budget — process group killed/);
  // exit code 1 with the Python's own error line on stdout: its reason is what the heartbeat shows
  const failed: WalkforwardExec = async () => { throw Object.assign(new Error("exit 1"), { code: 1, stdout: '{"status":"error","reason":"MemoryError: peak RSS 800 MB"}\n' }); };
  assert.equal((await runFilings13fClusters({}, failed)).detail, "MemoryError: peak RSS 800 MB");
  assert.equal(parseFilings13fClustersSummary('{"status":"skipped","reason":"no filings13f rows found"}').status, "skipped");
  assert.equal(parseFilings13fClustersSummary("Traceback").status, "error");
  assert.equal(filings13fClustersCmd("1.0.9; rm -rf /"), filings13fClustersCmd("unknown"));
});

test("status: whitelisted blocks only, survives sanitizeDiag, states that nothing is claimed; empty dir => nulls", () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "f13c-"));
  try {
    const root = path.join(tmp, "filings13f_clusters");
    const empty = filings13fClustersStatus({ DATA_DIR: tmp }, root);
    assert.equal(empty.latest, null);
    assert.equal(empty.last_run, null);
    assert.match(String(empty.claim), /^none:/);
    assert.equal(filings13fClustersLastFinishedAt(root), null);
    fs.mkdirSync(root);
    fs.writeFileSync(path.join(root, "latest.json"), JSON.stringify({
      schema: "filings13f_clusters/1", generated_at: "2026-11-23T10:21:00+00:00", code_version: "1.0.1",
      archive: { day_files: 127, rows: 14000, full_filings: 9120 }, periods_full_filings: { "2026-06-30": 4400, "2026-09-30": 4600 },
      pairs: [{ period: "2026-09-30", base_period: "2026-06-30", clusters: 2400, clusters_by_class: { thin: 310, no_prior: 90, broad: 2000 } }],
      newest_pair_top_thin: [{ cusip: "88888A101", issuer: "THIN CO INC", new_entrants: 5, prior_holders: 2, event_date: "2026-11-13" }],
      first_seen_appended: 12, definitions: { min_new_entrants: 3, thin_prior_max: 10 }, elapsed_s: 41.2, peak_rss_mb: 288,
      secret_extra: "x",
    }));
    fs.writeFileSync(path.join(root, "last_run.json"), JSON.stringify({ status: "ok", finished_at: "2026-11-23T10:21:00+00:00" }));
    const st = filings13fClustersStatus({ DATA_DIR: tmp }, root);
    const latest = st.latest as Record<string, any>;
    assert.equal(latest.pairs[0].clusters_by_class.thin, 310);
    assert.equal(latest.newest_pair_top_thin[0].issuer, "THIN CO INC");
    assert.equal(latest.definitions.min_new_entrants, 3);
    assert.ok(!JSON.stringify(st).includes("secret_extra"), "only whitelisted blocks are served");
    assert.deepEqual(sanitizeDiag(st), st);
    assert.equal(filings13fClustersLastFinishedAt(root), "2026-11-23T10:21:00+00:00");
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});

test("wiring: whitelisted probe, read-only case in bot.ts, booted from routes.ts, no daemon/order/price path, never ticks on boot", () => {
  assert.ok((DIAG_PROBES as readonly string[]).includes("filings13f_clusters"));
  const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
  const i = bot.indexOf('case "filings13f_clusters": {');
  assert.ok(i > 0);
  const block = bot.slice(i, bot.indexOf("}", bot.indexOf("return res.json", i)) + 1);
  assert.ok(block.includes("sanitizeDiag(filings13fClustersStatus())"));
  assert.ok(!/execPython|pythonRpc|beat\(|runFilings13fClusters/.test(block));
  assert.ok(fs.readFileSync(path.join(here, "routes.ts"), "utf8").includes("bootFilings13fClusters();"));
  const src = fs.readFileSync(path.join(here, "filings13fClusters.ts"), "utf8");
  const boot = src.slice(src.indexOf("export function bootFilings13fClusters"));
  assert.ok(!/^\s*(void\s+)?tick\(\);/m.test(boot));
  assert.ok(boot.includes("setInterval(") && boot.includes(".unref"));
  assert.ok(!/execPythonSerialized|pythonRunning|pythonRpc/.test(src));
  assert.ok(!src.includes("/orders") && !/method:\s*"(POST|PUT|PATCH|DELETE)"/.test(src));
  assert.ok(boot.includes('beat("filings13f_clusters_nightly", "start")') && boot.includes('beat("filings13f_clusters_nightly", "finish"'));
  // the Python the job runs is at the repo root (the image ships root *.py), pins the registered
  // definitions, and reads no price: the event list must exist before any return is looked at
  const py = fs.readFileSync(path.join(here, "..", "filings13f_clusters.py"), "utf8");
  assert.ok(py.includes("MIN_NEW_ENTRANTS = 3") && py.includes("THIN_PRIOR_MAX = 10"));
  assert.ok(py.includes("first_seen.jsonl"));
  assert.ok(!/fetch_bars|backtest_v2|alpaca/i.test(py), "the detector reads no price");
});
