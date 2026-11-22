// eventStudy — weekly job + read-only status for event_reaction_panel.py (the
// pre-registered event-reaction study on the delisting-inclusive panel).
// Run: npx tsx --test server/eventStudy.test.ts
import { test, mock } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  eventStudyFlag, eventStudyCmd, eventStudyCacheDir, inEventStudyWindow, decideEventStudy, busyPanelReader,
  parseEventStudySummary, runEventStudy, eventStudyStatus, studyCodeHash,
  EVENT_STUDY_TIMEOUT_MS, EVENT_STUDY_STALE_DAYS, EVENT_STUDY_MIN_FREE_BYTES, EVENT_STUDY_WINDOW_START_MIN,
  EVENT_STUDY_WINDOW_END_MIN, EVENT_STUDY_FILES, EVENT_STUDY_SEC_RATE, PANEL_READER_JOBS,
  nextEventStudyRetry, EVENT_STUDY_SEC_GAP_REASON, EVENT_STUDY_MAX_ATTEMPTS, EVENT_STUDY_RETRY_WAIT_MS, type EventStudyRetry,
  eventStudyCachePlan, currentEventStudyCachePlan, dirUsage, bootEventStudy, EVENT_STUDY_CACHE_MIN_FREE_BYTES,
  EVENT_STUDY_CACHE_MAX_BYTES, EVENT_STUDY_CACHE_SUBDIR, EVENT_STUDY_FIRST_TICK_MS, EVENT_STUDY_TICK_MS,
} from "./eventStudy";
import { WALKFORWARD_ENV_CAPS, WF_WINDOW_START_MIN, type WalkforwardExec } from "./walkforward";
import { WINDOW_START_MIN as PANEL_BUILD_START_MIN } from "./pricePanel";
import { JOB_REGISTRY, heartbeatStore, type HeartbeatFile } from "./jobHeartbeat";
import { sanitizeDiag, DIAG_PROBES } from "./diag";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");
// 2026-11-07 is a Saturday; EST (UTC-5) after the Nov 1 fall-back: 12:00Z = 07:00 ET
const SAT_09_ET = Date.parse("2026-11-07T14:00:00Z");
const DAY = 86_400_000;
const base = {
  enabled: true, panelBuildId: "b20261029", latest: null, lastAttemptWeek: "", studyCode: "aaaa000011112222",
  freeBytes: 5e9 as number | null, busy: null as string | null,
};

test("flag: env present wins; registry enabledWhen agrees; the run budget fits inside the stuck-run bound", () => {
  const job = JOB_REGISTRY.find((j) => j.id === "event_study_weekly");
  assert.ok(job && job.kind === "interval" && job.source === "server/eventStudy.ts");
  for (const [v, on] of [[undefined, true], ["1", true], ["0", false], ["off", false], ["", false]] as const) {
    const env = v === undefined ? {} : { VOLTRADE_EVENT_STUDY: v };
    assert.equal(eventStudyFlag(env).on, on);
    assert.equal(job!.enabledWhen!(env), on);
  }
  assert.ok(job!.maxRunMs > EVENT_STUDY_TIMEOUT_MS);
});

test("window: Saturday 07:00-15:00 ET only; the budget ends before the 20:15 ET panel build; starts after the walk-forward's slot", () => {
  assert.equal(inEventStudyWindow(SAT_09_ET), true);
  assert.equal(inEventStudyWindow(Date.parse("2026-11-07T12:00:00Z")), true);    // 07:00 ET
  assert.equal(inEventStudyWindow(Date.parse("2026-11-07T11:59:00Z")), false);   // 06:59 ET (the walk-forward's slot)
  assert.equal(inEventStudyWindow(Date.parse("2026-11-07T19:59:00Z")), true);    // 14:59 ET
  assert.equal(inEventStudyWindow(Date.parse("2026-11-07T20:00:00Z")), false);   // 15:00 ET
  assert.equal(inEventStudyWindow(Date.parse("2026-11-08T15:00:00Z")), false);   // Sunday (Edge P2's day)
  assert.equal(inEventStudyWindow(Date.parse("2026-11-06T15:00:00Z")), false);   // Friday (market hours)
  assert.equal(inEventStudyWindow(Date.parse("2026-07-11T11:30:00Z")), true);    // Saturday 07:30 EDT
  assert.ok(EVENT_STUDY_WINDOW_END_MIN + EVENT_STUDY_TIMEOUT_MS / 60_000 <= PANEL_BUILD_START_MIN, "a run that starts by the window end must finish before the panel build window opens");
  assert.ok(EVENT_STUDY_WINDOW_START_MIN > WF_WINDOW_START_MIN);
  assert.ok(EVENT_STUDY_SEC_RATE > 0 && EVENT_STUDY_SEC_RATE < 10, "under SEC's 10 req/s fair-access ceiling");
});

test("decide: idle outside/disabled/attempted/busy (busy spends nothing); skip without panel or on low disk; fresh keyed on build + STUDY code", () => {
  assert.equal(decideEventStudy(SAT_09_ET, { ...base, enabled: false }).action, "idle");
  assert.equal(decideEventStudy(Date.parse("2026-11-08T15:00:00Z"), base).action, "idle");
  const run = decideEventStudy(SAT_09_ET, base);
  assert.equal(run.action, "run");
  const week = (run as { week: string }).week;
  assert.equal(decideEventStudy(SAT_09_ET, { ...base, lastAttemptWeek: week }).action, "idle");
  const busy = decideEventStudy(SAT_09_ET, { ...base, busy: "walkforward_weekly" });
  assert.equal(busy.action, "idle");
  assert.match(busy.reason, /walkforward_weekly/);
  assert.ok(!("week" in busy), "waiting on another reader must not spend the week's attempt");
  assert.equal(decideEventStudy(SAT_09_ET, { ...base, panelBuildId: null }).action, "skip");
  const low = decideEventStudy(SAT_09_ET, { ...base, freeBytes: EVENT_STUDY_MIN_FREE_BYTES - 1 });
  assert.equal(low.action, "skip");
  assert.match(low.reason, /result files/);
  assert.equal(decideEventStudy(SAT_09_ET, { ...base, freeBytes: 1_184_000_000 }).action, "run", "the live volume's ~1.18 GB free must not block");
  assert.equal(decideEventStudy(SAT_09_ET, { ...base, freeBytes: null }).action, "run");
  const fresh = { build_id: "b20261029", study_code: base.studyCode, generated_at: new Date(SAT_09_ET - 3 * DAY).toISOString() };
  assert.equal(decideEventStudy(SAT_09_ET, { ...base, latest: fresh }).action, "skip", "an unrelated deploy (same study code) does not repeat ~42k SEC requests");
  assert.equal(decideEventStudy(SAT_09_ET, { ...base, latest: { ...fresh, build_id: "b20261001" } }).action, "run");
  assert.equal(decideEventStudy(SAT_09_ET, { ...base, latest: { ...fresh, study_code: "bbbb000011112222" } }).action, "run");
  const stale = { ...fresh, generated_at: new Date(SAT_09_ET - (EVENT_STUDY_STALE_DAYS + 1) * DAY).toISOString() };
  assert.equal(decideEventStudy(SAT_09_ET, { ...base, latest: stale }).action, "run");
});

test("busy: an open run of a panel reader blocks; a finished one or a stuck one (older than its maxRunMs) does not", () => {
  const now = SAT_09_ET;
  const snap = (jobs: HeartbeatFile["jobs"]): HeartbeatFile => ({ version: 1, jobs });
  assert.equal(busyPanelReader(snap({}), now), null);
  assert.equal(busyPanelReader(snap({ walkforward_weekly: [{ startedAt: now - 60_000 }] }), now), "walkforward_weekly");
  assert.equal(busyPanelReader(snap({ edge_p2_weekly: [{ startedAt: now - 9 * DAY, finishedAt: now - 9 * DAY }, { startedAt: now - 1000 }] }), now), "edge_p2_weekly");
  assert.equal(busyPanelReader(snap({ walkforward_weekly: [{ startedAt: now - 60_000, finishedAt: now - 1000, status: "ok" }] }), now), null);
  const wf = JOB_REGISTRY.find((j) => j.id === "walkforward_weekly")!;
  assert.equal(busyPanelReader(snap({ walkforward_weekly: [{ startedAt: now - wf.maxRunMs - 1 }] }), now), null, "a stuck open beat must not block forever");
  assert.equal(busyPanelReader(snap({ dtcc_swaps: [{ startedAt: now - 1000 }] }), now), null);
  for (const id of PANEL_READER_JOBS) assert.ok(JOB_REGISTRY.some((j) => j.id === id), id);
});

test("run: own long-job lane — budget, SIGKILL group, thread caps, parsed summary, version and paths never interpolated unchecked", async () => {
  const calls: Array<{ cmd: string; opts: Record<string, unknown> }> = [];
  const ok: WalkforwardExec = async (cmd, opts) => {
    calls.push({ cmd, opts: opts as Record<string, unknown> });
    return { stdout: '[log]\n{"status":"ok","build_id":"b20261029","as_of":"2026-10-27","events":61234,"elapsed_s":8400.2,"peak_rss_mb":610,"verdicts":{"E1_insider_cluster":"PRICED IN FAST / NO DRIFT"}}\n', stderr: "" };
  };
  const r = await runEventStudy({ PATH: "/usr/bin" }, ok);
  assert.equal(r.status, "ok");
  assert.match(r.detail, /61234 events/);
  assert.match(r.detail, /E1 PRICED IN FAST/);
  assert.match(calls[0].cmd, /^python3 event_reaction_panel\.py --run --nice 19 --max-rss-mb \d+ --rate 5 --workers \d+ --code-version \S+ --cache-dir \S+$/);
  assert.equal(calls[0].opts.timeout, EVENT_STUDY_TIMEOUT_MS);
  assert.equal(calls[0].opts.killSignal, "SIGKILL");
  for (const [k, v] of Object.entries(WALKFORWARD_ENV_CAPS)) assert.equal((calls[0].opts.env as Record<string, string>)[k], v);
  const killed: WalkforwardExec = async () => { throw Object.assign(new Error("x"), { killed: true }); };
  assert.match((await runEventStudy({}, killed)).detail, /budget — process group killed/);
  const failed: WalkforwardExec = async () => { throw Object.assign(new Error("x"), { code: 1, stdout: '{"status":"error","reason":"memory budget: RSS 1300 MB"}' }); };
  assert.deepEqual(await runEventStudy({}, failed), { status: "error", detail: "memory budget: RSS 1300 MB" });
  assert.equal(parseEventStudySummary('{"status":"insufficient","reason":"SEC insider data sets missing"}').status, "insufficient");
  assert.equal(parseEventStudySummary('{"status":"skipped","reason":"no complete price-panel build"}').status, "skipped");
  assert.equal(parseEventStudySummary("Traceback (most recent call last)").status, "error");
  assert.equal(eventStudyCmd("1.0.9; rm -rf /"), eventStudyCmd("unknown"));
});

test("cache dir: with the environment alone it is container tmp, safe override only (since 2026-11-22 the job asks eventStudyCachePlan, which may choose the volume)", () => {
  const d = eventStudyCacheDir({});
  assert.ok(!d.startsWith("/data"), d);
  assert.ok(eventStudyCmd("1.0.1", {}).endsWith(`--cache-dir ${d}`));
  assert.equal(eventStudyCacheDir({ VOLTRADE_EVENT_STUDY_CACHE_DIR: "/scratch/es" }), "/scratch/es");
  assert.equal(eventStudyCacheDir({ VOLTRADE_EVENT_STUDY_CACHE_DIR: "/x; rm -rf /" }), d, "unsafe override ignored");
});

test("study code hash: the same digest as event_reaction_panel.study_code_hash (pinned on one fixture); both files exist", () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "esh-"));
  try {
    fs.writeFileSync(path.join(tmp, "event_reaction.py"), "alpha\n");
    fs.writeFileSync(path.join(tmp, "event_reaction_panel.py"), "beta\n");
    assert.equal(studyCodeHash(tmp), "686200ca135543a4");   // test_event_reaction_panel.py pins the same value
    fs.rmSync(path.join(tmp, "event_reaction_panel.py"));
    assert.equal(studyCodeHash(tmp), "missing:event_reaction_panel.py");
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
  assert.deepEqual([...EVENT_STUDY_FILES], ["event_reaction.py", "event_reaction_panel.py"]);
  const py = fs.readFileSync(path.join(root, "event_reaction_panel.py"), "utf8");
  assert.ok(py.includes('STUDY_FILES = ("event_reaction.py", "event_reaction_panel.py")'));
  for (const flag of ["--run", "--nice", "--max-rss-mb", "--rate", "--workers", "--code-version", "--cache-dir"]) {
    assert.ok(py.includes(`ap.add_argument("${flag}"`), `the Python CLI defines ${flag}`);
  }
  assert.match(studyCodeHash(root), /^[0-9a-f]{16}$/);
});

function latestFixture(): Record<string, unknown> {
  const st = (n: number, mean: number, t: number) => ({ n, mean, se: Math.abs(mean / t), t, g: 40, median: mean / 2, hit: 0.5 });
  const row = (verdict?: string) => ({
    "0_1": st(900, 0.021, 9.1), "2_5": st(900, 0.002, 1.1), "2_20": st(900, 0.004, 1.3), "2_60": st(880, 0.01, 1.2), "0_20": st(900, 0.025, 6),
    raw: { "0_1": st(900, 0.02, 8) }, halves: { "2016_2020": st(400, 0.006, 1.2), "2021_on": st(500, 0.002, 0.4) },
    regime: { bull: { "0_1": st(700, 0.02, 8) } }, spread_median: 0.007, late_share: null, secret_ticker: "QZXA",
    ...(verdict ? { verdict, verdict_core: verdict } : {}),
  });
  const block = { n: 3600, t_bar: 3.02, verdict: "PRICED IN FAST / NO DRIFT", verdict_core: "PRICED IN FAST / NO DRIFT",
    table: { pooled: row(), small: row("fail"), mid: row("fail"), large: row("fail") } };
  const summary = Object.fromEntries(["E1_insider_cluster", "E2_item_1_01", "E3_item_4_02", "E4_ceo_cfo_exit"].map((t) => [t, block]));
  return {
    schema: "event_study_panel/v1", status: "ok", generated_at: "2026-11-07T16:10:00Z", code_version: "1.0.1342", study_code: "aaaa000011112222",
    as_of: "2026-10-27", panel: { build_id: "b20261029", as_of: "2026-10-27" }, preregistration: "research/event_reaction_study.md §1",
    bars: { t: 2.87, t_e1: 3.02, cells: 12 }, delisting_assumption: { merger_or_rename: 0, other: -0.3 }, prior: "Stated 2026-11-06, before any panel run.",
    deviations: ["PD1 price source", "PD2 universe"], caveats: ["c"], meta: { measured_by_type: { E1_insider_cluster: 3600 }, sec_cache_bytes: 21_000_000 },
    verdicts: { E1_insider_cluster: "PRICED IN FAST / NO DRIFT" }, summary,
    survivor_subset: { note: "n", verdicts: { E1_insider_cluster: "PRICED IN FAST / NO DRIFT" }, summary },
    elapsed_s: 8400.2, peak_rss_mb: 610.3, records: [{ ticker: "QZXA", car: 0.1 }],
  };
}

test("status: verdicts + headline cells only (no ticker/row passes the whitelist), survives sanitizeDiag; empty dir => nulls", () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "es-"));
  try {
    const st0 = eventStudyStatus({ DATA_DIR: tmp }, path.join(tmp, "event_study"), path.join(tmp, "pp"), root);
    assert.equal(st0.latest, null);
    assert.equal(st0.last_run, null);
    assert.equal(st0.probe, "event_study");
    fs.mkdirSync(path.join(tmp, "event_study"));
    fs.writeFileSync(path.join(tmp, "event_study", "latest.json"), JSON.stringify(latestFixture()));
    fs.writeFileSync(path.join(tmp, "event_study", "last_run.json"), JSON.stringify({ status: "ok", finished_at: "2026-11-07T16:10:00Z" }));
    const st = eventStudyStatus({ DATA_DIR: tmp }, path.join(tmp, "event_study"), path.join(tmp, "pp"), root);
    const latest = st.latest as Record<string, any>;
    assert.equal(latest.verdicts.E1_insider_cluster, "PRICED IN FAST / NO DRIFT");
    assert.equal(latest.cells.E1_insider_cluster.t_bar, 3.02);
    const small = latest.cells.E1_insider_cluster.small;
    assert.deepEqual(small.car_2_20, { n: 900, mean: 0.004, t: 1.3, median: 0.002, hit: 0.5 });
    assert.deepEqual(small.car_2_20_2016_2020, { n: 400, mean: 0.006, t: 1.2 });
    assert.deepEqual(small.car_2_20_2021_on, { n: 500, mean: 0.002, t: 0.4 });
    assert.equal(small.verdict, "fail");
    assert.equal(latest.survivor_subset.cells.E3_item_4_02.pooled.car_0_1.mean, 0.021);
    assert.equal(latest.prior, "Stated 2026-11-06, before any panel run.");
    assert.deepEqual(latest.deviations, ["PD1 price source", "PD2 universe"]);
    const text = JSON.stringify(st);
    assert.ok(!text.includes("QZXA") && !text.includes("secret_ticker") && !text.includes("records"), "whitelist only");
    assert.deepEqual(sanitizeDiag(st), st, "fits the diag depth/size caps unchanged");
    assert.equal((st.last_run as Record<string, unknown>).status, "ok");
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});

test("wiring: whitelisted probe, read-only case in bot.ts, booted from routes.ts, no mutex/daemon/order path, never ticks on boot", () => {
  assert.ok((DIAG_PROBES as readonly string[]).includes("event_study"));
  const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
  const i = bot.indexOf('case "event_study": {');
  assert.ok(i > 0);
  const block = bot.slice(i, bot.indexOf("}", bot.indexOf("return res.json", i)) + 1);
  assert.ok(block.includes("sanitizeDiag(eventStudyStatus())"));
  assert.ok(!/execPython|pythonRpc|beat\(|runEventStudy|req\.query/.test(block));
  assert.ok(fs.readFileSync(path.join(here, "routes.ts"), "utf8").includes("bootEventStudy();"));
  const src = fs.readFileSync(path.join(here, "eventStudy.ts"), "utf8");
  const boot = src.slice(src.indexOf("export function bootEventStudy"));
  assert.ok(!/^\s*(void\s+)?tick\(\);/m.test(boot));
  assert.ok(boot.includes("setInterval(") && boot.includes(".unref"));
  assert.ok(boot.includes("busyPanelReader(heartbeatStore().snapshot()"));
  assert.ok(!/execPythonSerialized|pythonRunning|pythonRpc/.test(src));
  assert.ok(!src.includes("/orders") && !/method:\s*"(POST|PUT|PATCH|DELETE)"/.test(src));
  assert.ok(src.includes('beat("event_study_weekly", "start")') && src.includes('beat("event_study_weekly", "finish"'));
});

// 2026-11-14: the first live run ended "insufficient: SEC insider data sets missing for [six quarters]" ten
// minutes into the Saturday window, and the week's one attempt was spent. The six download and parse cleanly.
test("a run that lacked SEC downloads does not spend the week: up to 3 attempts, an hour apart, inside the window", () => {
  const first = decideEventStudy(SAT_09_ET, base);
  assert.equal(first.action, "run");
  const week = (first as { week: string }).week;
  const gap = { status: "insufficient" as const, detail: `${EVENT_STUDY_SEC_GAP_REASON} 6 quarter(s): 2014q2 (HTTP 403)` };
  const done = SAT_09_ET + 10 * 60_000;
  const r1 = nextEventStudyRetry(null, week, gap, done);
  assert.deepEqual(r1, { week, attempts: 1, notBeforeMs: done + EVENT_STUDY_RETRY_WAIT_MS });
  const attempted = { ...base, lastAttemptWeek: week };
  // before the wait is over: idle, and the reason says why
  const waiting = decideEventStudy(done + 30 * 60_000, { ...attempted, retry: r1 });
  assert.equal(waiting.action, "idle");
  assert.match(waiting.reason, /attempt 1 of 3 lacked SEC downloads/);
  // after it: the same week runs again
  const again = decideEventStudy(done + EVENT_STUDY_RETRY_WAIT_MS, { ...attempted, retry: r1 });
  assert.equal(again.action, "run");
  assert.equal((again as { week: string }).week, week);
  // a retry still waits for another panel reader and still respects the window
  assert.equal(decideEventStudy(done + EVENT_STUDY_RETRY_WAIT_MS, { ...attempted, retry: r1, busy: "walkforward_weekly" }).action, "idle");
  assert.equal(decideEventStudy(Date.parse("2026-11-07T20:00:00Z"), { ...attempted, retry: r1 }).action, "idle", "15:00 ET: the window is over");
  // the second gap allows a third attempt; the third spends the week
  const r2 = nextEventStudyRetry(r1, week, gap, done + 2 * EVENT_STUDY_RETRY_WAIT_MS);
  assert.equal(r2?.attempts, 2);
  assert.equal(nextEventStudyRetry(r2, week, gap, done + 4 * EVENT_STUDY_RETRY_WAIT_MS), null);
  assert.equal(EVENT_STUDY_MAX_ATTEMPTS, 3);
  assert.equal(decideEventStudy(done + 5 * EVENT_STUDY_RETRY_WAIT_MS, { ...attempted, retry: null }).action, "idle");
  // a record from another week never reopens this one, and starts its own count
  const stale: EventStudyRetry = { week: "2026-W44", attempts: 2, notBeforeMs: 0 };
  assert.equal(decideEventStudy(SAT_09_ET, { ...attempted, retry: stale }).action, "idle");
  assert.equal(nextEventStudyRetry(stale, week, gap, done)?.attempts, 1);
});

test("only the SEC-download gap is retried: every other outcome spends the week", () => {
  const week = "2026-W45";
  for (const out of [
    { status: "ok" as const, detail: "build b20261029" },
    { status: "error" as const, detail: "memory budget: RSS 1300 MB" },
    { status: "skipped" as const, detail: "no complete price-panel build (current.json absent)" },
    { status: "insufficient" as const, detail: "SEC fetch errors: submissions 2/37" },
    { status: "insufficient" as const, detail: "panel has no usable SPY series (trading calendar)" },
    { status: "error" as const, detail: `${EVENT_STUDY_SEC_GAP_REASON} 1 quarter(s)` },
  ]) assert.equal(nextEventStudyRetry(null, week, out, SAT_09_ET), null, out.detail);
});

test("the tick keeps the retry record: cleared before each attempt, set from the run's outcome, named in the heartbeat", () => {
  const src = fs.readFileSync(path.join(here, "eventStudy.ts"), "utf8");
  const tick = src.slice(src.indexOf("const tick = async () => {"), src.indexOf("setInterval(() => { void tick(); }"));
  assert.match(tick, /busy: busyPanelReader\(heartbeatStore\(\)\.snapshot\(\), now\), retry,\n/);
  assert.match(tick, /lastAttemptWeek = d\.week;\n\s*const before = retry;\n\s*retry = null;/);
  assert.match(tick, /const out = await runEventStudy\(env\);\n\s*retry = nextEventStudyRetry\(before, d\.week, out, Date\.now\(\)\);/);
  assert.match(tick, /attempt \$\{retry\.attempts\} of \$\{EVENT_STUDY_MAX_ATTEMPTS\}/);
});

// 2026-11-21: seven deploys between 13:49Z and 20:57Z, the longest gap 2 h 21 min. One run is about 2.5 h of SEC
// requests and its cache was in the container tmp dir, so every replaced container started from zero: the third
// Saturday without a result. The Python checkpoints every stage into the cache; the cache has to outlive the container.
const VOL_ROOT = "/data/voltrade/event_study";
const ROOMY = 10 * EVENT_STUDY_CACHE_MIN_FREE_BYTES;

test("SEC cache plan: on the volume while it has room, so a replaced container resumes; container tmp otherwise", () => {
  const vol = eventStudyCachePlan({}, { root: VOL_ROOT, freeBytes: ROOMY, volumeCacheBytes: 0, tmpDir: "/tmp" });
  assert.deepEqual({ dir: vol.dir, where: vol.where }, { dir: `${VOL_ROOT}/${EVENT_STUDY_CACHE_SUBDIR}`, where: "volume" });
  // a cache already there keeps being used (that is the resume)
  assert.equal(eventStudyCachePlan({}, { root: VOL_ROOT, freeBytes: ROOMY, volumeCacheBytes: 21_000_000, tmpDir: "/tmp" }).where, "volume");
  // exactly at the floor is allowed, one byte under is not
  assert.equal(eventStudyCachePlan({}, { root: VOL_ROOT, freeBytes: EVENT_STUDY_CACHE_MIN_FREE_BYTES, volumeCacheBytes: 0, tmpDir: "/tmp" }).where, "volume");
  const low = eventStudyCachePlan({}, { root: VOL_ROOT, freeBytes: EVENT_STUDY_CACHE_MIN_FREE_BYTES - 1, volumeCacheBytes: 0, tmpDir: "/tmp" });
  assert.deepEqual({ dir: low.dir, where: low.where }, { dir: "/tmp/voltrade_event_study", where: "tmp" });
  assert.match(low.reason, /MB free on the volume/);
  assert.equal(eventStudyCachePlan({}, { root: VOL_ROOT, freeBytes: null, volumeCacheBytes: 0, tmpDir: "/tmp" }).where, "tmp", "unreadable free space never writes to the volume");
  const big = eventStudyCachePlan({}, { root: VOL_ROOT, freeBytes: ROOMY, volumeCacheBytes: EVENT_STUDY_CACHE_MAX_BYTES + 1, tmpDir: "/tmp" });
  assert.equal(big.where, "tmp");
  assert.match(big.reason, /above the \d+ MB cap/);
  assert.equal(eventStudyCachePlan({}, { root: VOL_ROOT, freeBytes: ROOMY, volumeCacheBytes: EVENT_STUDY_CACHE_MAX_BYTES, tmpDir: "/tmp" }).where, "volume");
  // no volume (local, CI): the root is in the tmp dir, so the "volume" would be tmp anyway
  assert.equal(eventStudyCachePlan({}, { root: "/tmp/event_study", freeBytes: ROOMY, volumeCacheBytes: 0, tmpDir: "/tmp" }).where, "tmp");
  // the env override still wins, and an unsafe one is still ignored
  assert.deepEqual(eventStudyCachePlan({ VOLTRADE_EVENT_STUDY_CACHE_DIR: "/scratch/es" }, { root: VOL_ROOT, freeBytes: 0, volumeCacheBytes: 0, tmpDir: "/tmp" }),
    { dir: "/scratch/es", where: "override", reason: "VOLTRADE_EVENT_STUDY_CACHE_DIR" });
  assert.equal(eventStudyCachePlan({ VOLTRADE_EVENT_STUDY_CACHE_DIR: "/x; rm -rf /" }, { root: VOL_ROOT, freeBytes: ROOMY, volumeCacheBytes: 0, tmpDir: "/tmp" }).where, "volume");
  assert.equal(eventStudyCachePlan({}, { root: "/data/vol trade/event_study", freeBytes: ROOMY, volumeCacheBytes: 0, tmpDir: "/tmp" }).where, "tmp", "a path that is not plain never reaches the shell");
});

test("SEC cache limits: the floor is the bot's 1 GiB headroom plus 256 MiB; the cap is far above the estimate and far under the floor's margin of error", () => {
  const gs = fs.readFileSync(path.join(here, "globalScopes.ts"), "utf8");
  const m = /export const MIN_FREE_BYTES = ([0-9_]+);/.exec(gs);
  assert.ok(m, "globalScopes.MIN_FREE_BYTES is a literal");
  assert.equal(EVENT_STUDY_CACHE_MIN_FREE_BYTES, Number(m![1].replace(/_/g, "")) + 256 * 1024 * 1024);
  assert.ok(EVENT_STUDY_CACHE_MAX_BYTES >= 4 * 25_000_000, "the 15-25 MB estimate may be wrong by 4x without losing the resume");
  assert.ok(EVENT_STUDY_CACHE_MAX_BYTES <= 256 * 1024 * 1024, "a full cache cannot take the volume under the bot's headroom");
});

test("run: the plan's directory is the one the Python is given; a volume cache above the cap is deleted before the run", async () => {
  const cmds: string[] = [];
  const ok: WalkforwardExec = async (cmd) => { cmds.push(cmd); return { stdout: '{"status":"skipped","reason":"x"}\n', stderr: "" }; };
  await runEventStudy({ PATH: "/usr/bin" }, ok, eventStudyCachePlan({}, { root: VOL_ROOT, freeBytes: ROOMY, volumeCacheBytes: 0, tmpDir: "/tmp" }));
  assert.ok(cmds[0].endsWith(`--cache-dir ${VOL_ROOT}/${EVENT_STUDY_CACHE_SUBDIR}`), cmds[0]);
  assert.equal(eventStudyCmd("1.0.1", {}, "/x; rm -rf /"), eventStudyCmd("1.0.1", {}, "/tmp/voltrade_event_study"), "an unsafe directory never reaches the shell");
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "es-cap-"));
  try {
    const cache = path.join(tmp, "event_study", EVENT_STUDY_CACHE_SUBDIR, "b20261029-c1", "form345");
    fs.mkdirSync(cache, { recursive: true });
    fs.writeFileSync(path.join(cache, "2016q1.pairs.json.gz"), "abc");
    fs.writeFileSync(path.join(cache, "..", "listing.json.gz"), "12345");
    assert.deepEqual(dirUsage(path.join(tmp, "event_study", EVENT_STUDY_CACHE_SUBDIR)), { bytes: 8, files: 2 });
    assert.deepEqual(dirUsage(path.join(tmp, "absent")), { bytes: 0, files: 0 });
    const cur = currentEventStudyCachePlan({ DATA_DIR: tmp });
    assert.equal(cur.volumeCacheBytes, 8);
    assert.equal(cur.where, "tmp", "a root in the tmp dir is not a volume");
    // under the cap: nothing is deleted
    await runEventStudy({ DATA_DIR: tmp }, ok, { dir: "/tmp/voltrade_event_study", where: "tmp", reason: "low", volumeCacheBytes: 8 });
    assert.ok(fs.existsSync(path.join(cache, "2016q1.pairs.json.gz")));
    // over the cap: deleted, and the run still starts (on tmp)
    await runEventStudy({ DATA_DIR: tmp }, ok, { dir: "/tmp/voltrade_event_study", where: "tmp", reason: "over the cap", volumeCacheBytes: EVENT_STUDY_CACHE_MAX_BYTES + 1 });
    assert.ok(!fs.existsSync(path.join(tmp, "event_study", EVENT_STUDY_CACHE_SUBDIR)));
    assert.ok(cmds[cmds.length - 1].endsWith("--cache-dir /tmp/voltrade_event_study"));
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});

test("status: the probe says where the SEC cache is, whether it survives a deploy, and how large it is", () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "es-st-"));
  try {
    const es = path.join(tmp, "event_study");
    const st = eventStudyStatus({ DATA_DIR: tmp }, es, path.join(tmp, "pp"), root);
    const c = st.sec_cache as Record<string, unknown>;
    assert.equal(c.where, "tmp");
    assert.equal(c.survives_a_deploy, false);
    assert.equal(typeof c.bytes, "number");
    assert.equal(c.volume_min_free_mb, Math.round(EVENT_STUDY_CACHE_MIN_FREE_BYTES / 1e6));
    const over = path.join(tmp, "kept");
    fs.mkdirSync(over);
    fs.writeFileSync(path.join(over, "a.json.gz"), "1234567");
    const st2 = eventStudyStatus({ DATA_DIR: tmp, VOLTRADE_EVENT_STUDY_CACHE_DIR: over }, es, path.join(tmp, "pp"), root);
    assert.deepEqual({ ...(st2.sec_cache as Record<string, unknown>), volume_min_free_mb: 0, max_mb: 0 },
      { where: "override", reason: "VOLTRADE_EVENT_STUDY_CACHE_DIR", survives_a_deploy: true, bytes: 7, files: 1, volume_min_free_mb: 0, max_mb: 0 });
    assert.deepEqual(sanitizeDiag(st2), st2, "fits the diag depth/size caps unchanged");
    assert.match(String(st.schedule), /first tick 2 min after boot/);
    assert.ok(!JSON.stringify(st2.sec_cache).includes(over), "no path in the probe");
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});

test("boot: the first tick lands FIRST_TICK_MS after boot, not one 30-min interval later; nothing runs synchronously", () => {
  // Would have failed before 2026-11-22: the first tick was one EVENT_STUDY_TICK_MS after boot, so a container
  // replaced inside 30 minutes never looked at the week. Observed without Python: with no price-panel build in
  // DATA_DIR the tick's decision is "skip", which it records as a finished heartbeat run.
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "es-boot-"));
  const runs = () => (heartbeatStore().snapshot().jobs.event_study_weekly ?? []).length;
  assert.ok(EVENT_STUDY_FIRST_TICK_MS >= 60_000 && EVENT_STUDY_FIRST_TICK_MS <= 5 * 60_000 && EVENT_STUDY_FIRST_TICK_MS < EVENT_STUDY_TICK_MS);
  mock.timers.enable({ apis: ["setTimeout", "setInterval", "Date"], now: SAT_09_ET });
  try {
    const before = runs();
    bootEventStudy({ DATA_DIR: tmp, PATH: "/usr/bin" });
    assert.equal(runs(), before, "never ticks synchronously on boot");
    mock.timers.tick(EVENT_STUDY_FIRST_TICK_MS - 1);
    assert.equal(runs(), before, "not before the first-tick delay");
    mock.timers.tick(1);
    assert.equal(runs(), before + 1, "the week is looked at FIRST_TICK_MS after boot");
    const last = heartbeatStore().snapshot().jobs.event_study_weekly!.slice(-1)[0];
    assert.match(String(last.detail), /no complete price-panel build yet/);
    mock.timers.tick(EVENT_STUDY_TICK_MS);
    assert.equal(runs(), before + 1, "one process does not spend the week twice");
  } finally {
    mock.timers.reset();
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});
