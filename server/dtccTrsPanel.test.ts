// dtccTrsPanel — the job that runs dtcc_trs_panel_job.py (repo root) so the event
// panel of the registered DTCC equity-swap gate-2 test exists on the server
// (research/open_questions.md (aq1), 2026-11-26).
// Run: npx tsx --test server/dtccTrsPanel.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  dtccTrsPanelCmd, dtccTrsPanelFlag, dtccTrsPanelRoot, dtccTrsPanelLastRun, dtccTrsPanelNight,
  decideDtccTrsPanel, parseDtccTrsPanelSummary, runDtccTrsPanel, describeEventsCheck,
  DTCC_TRS_PANEL_JOB, DTCC_TRS_PANEL_TIMEOUT_MS, DTCC_TRS_PANEL_BUDGET_S, DTCC_TRS_PANEL_TICK_MS,
  DTCC_TRS_PANEL_MIN_FREE_BYTES, DTCC_TRS_PANEL_WINDOW_START_MIN, DTCC_TRS_PANEL_WINDOW_END_MIN,
} from "./dtccTrsPanel";
import { WALKFORWARD_ENV_CAPS, type WalkforwardExec } from "./walkforward";
import { JOB_REGISTRY, MAX_DETAIL_CHARS } from "./jobHeartbeat";

const here = path.dirname(fileURLToPath(import.meta.url));
const at = (iso: string) => Date.parse(iso);
// 2026-11-27 is a Friday (EST = UTC-5)
const FRI_0300 = at("2026-11-27T08:00:00Z");     // inside Thursday's night
const FRI_0400 = at("2026-11-27T09:00:00Z");
const FRI_1100 = at("2026-11-27T16:00:00Z");
const FRI_1959 = at("2026-11-28T00:59:00Z");
const FRI_2010 = at("2026-11-28T01:10:00Z");
const SAT_0100 = at("2026-11-28T06:00:00Z");     // still Friday's night
const THU_NIGHT = "2026-11-27T01:00:00.000Z";    // Thursday 20:00 ET
const FRI_NIGHT = "2026-11-28T01:00:00.000Z";
const base = { enabled: true, last: null as { finishedAt: string | null; backlog: number | null; read: number | null } | null, lastNightlyAttemptNight: "", freeBytes: 5e9 as number | null };

// The Python's own last stdout line, from a session machine on 2026-11-26.
// CATCH_UP: a run stopped by its day limit with days still owed.
// WHOLE: the run that found every vendor day held and wrote the event list.
const CATCH_UP = "{\"schema\": \"dtcc_trs_panel_job/1\", \"code_version\": \"unknown\", \"wanted\": 732, \"read\": 3, \"unavailable_new\": 0, \"not_published\": 0, \"errors\": 0, \"pending\": 729, \"backlog\": 725, \"stopped\": \"day limit\", \"days_ok\": 3, \"first_day\": \"2024-11-24\", \"last_day\": \"2024-11-26\", \"days_unavailable\": 0, \"stat_unreadable\": 0, \"oldest_unavailable\": null, \"newest_unavailable\": null, \"events\": null, \"status\": \"ok\", \"bytes\": 89384, \"finished_at\": \"2026-11-26T00:21:06+00:00\", \"elapsed_s\": 28.4, \"peak_rss_mb\": 31.6}";
const WHOLE = "{\"schema\": \"dtcc_trs_panel_job/1\", \"code_version\": \"unknown\", \"wanted\": 732, \"read\": 1, \"unavailable_new\": 0, \"not_published\": 0, \"errors\": 0, \"pending\": 0, \"backlog\": 0, \"stopped\": null, \"days_ok\": 732, \"first_day\": \"2024-11-24\", \"last_day\": \"2026-11-25\", \"days_unavailable\": 0, \"stat_unreadable\": 0, \"oldest_unavailable\": null, \"newest_unavailable\": null, \"events\": {\"events\": 27915, \"first_testable\": \"2025-02-25\", \"last_testable\": \"2026-11-25\", \"registered\": {\"through\": \"2026-11-24\", \"registered\": 27869, \"here\": 27869, \"match\": true, \"months_differing\": []}}, \"status\": \"ok\", \"bytes\": 130818629, \"finished_at\": \"2026-11-26T00:54:25+00:00\", \"elapsed_s\": 53.0, \"peak_rss_mb\": 229.2}";

test("registry: a daily job due when the window opens, its grace running to midnight; its budget fits a tick and the registry's run limit", () => {
  const job = JOB_REGISTRY.find((j) => j.id === DTCC_TRS_PANEL_JOB);
  assert.ok(job && job.kind === "daily" && job.source === "server/dtccTrsPanel.ts");
  assert.ok(job!.maxRunMs >= DTCC_TRS_PANEL_TIMEOUT_MS, "registry maxRunMs must cover the job budget");
  if (job!.kind === "daily") {
    assert.equal(job!.tz, "America/New_York");
    assert.equal(job!.dueMin, DTCC_TRS_PANEL_WINDOW_START_MIN);
    assert.equal(job!.dueMin * 60_000 + job!.graceMs, 24 * 60 * 60_000);
    assert.equal(job!.tradingDaysOnly, false);
  }
  // a run killed at its limit ends before the next tick, so two never overlap
  assert.ok(DTCC_TRS_PANEL_TIMEOUT_MS < DTCC_TRS_PANEL_TICK_MS);
  assert.ok(DTCC_TRS_PANEL_BUDGET_S * 1000 < DTCC_TRS_PANEL_TIMEOUT_MS);
  assert.equal(job!.enabledWhen?.({}), true);
  assert.equal(job!.enabledWhen?.({ VOLTRADE_DTCC_TRS_PANEL: "0" }), false);
  assert.equal(dtccTrsPanelFlag({}), true);
  assert.equal(dtccTrsPanelFlag({ VOLTRADE_DTCC_TRS_PANEL: "off" }), false);
});

test("the command names a root-level file and passes the budget; a hostile version string is not passed through", () => {
  const cmd = dtccTrsPanelCmd("1.0.1560");
  assert.equal(cmd, "python3 dtcc_trs_panel_job.py --run --nice 19 --max-rss-mb 700 --budget-s 600 --code-version 1.0.1560");
  assert.ok(fs.existsSync(path.join(here, "..", "dtcc_trs_panel_job.py")));
  assert.ok(fs.existsSync(path.join(here, "..", "dtcc_trs_panel.py")));
  assert.ok(dtccTrsPanelCmd("1; rm -rf /").endsWith("--code-version unknown"));
  assert.equal(dtccTrsPanelRoot({ DATA_DIR: "/x" }), path.join("/x", "dtcc_trs_panel"));
});

test("the window is 20:00 to 04:00 ET every day, the hours the Tier-2 scan does not run; a night is named by its start", () => {
  assert.equal(DTCC_TRS_PANEL_WINDOW_START_MIN, 20 * 60);
  assert.equal(DTCC_TRS_PANEL_WINDOW_END_MIN, 4 * 60);
  const tier2 = JOB_REGISTRY.find((j) => j.id === "tier2_scan");
  assert.ok(tier2 && tier2.kind === "interval" && tier2.window.kind === "et_range");
  if (tier2!.kind === "interval" && tier2!.window.kind === "et_range") {
    assert.equal(tier2!.window.startMin, DTCC_TRS_PANEL_WINDOW_END_MIN, "the job's window ends where the scan's begins");
    assert.equal(tier2!.window.endMin, DTCC_TRS_PANEL_WINDOW_START_MIN, "the job's window begins where the scan's ends");
  }
  assert.equal(dtccTrsPanelNight(FRI_0300), THU_NIGHT);
  assert.equal(dtccTrsPanelNight(at("2026-11-27T08:59:00Z")), THU_NIGHT);   // Fri 03:59
  assert.equal(dtccTrsPanelNight(FRI_0400), null);
  assert.equal(dtccTrsPanelNight(FRI_1100), null);
  assert.equal(dtccTrsPanelNight(FRI_1959), null);
  assert.equal(dtccTrsPanelNight(at("2026-11-28T01:00:00Z")), FRI_NIGHT);   // Fri 20:00
  assert.equal(dtccTrsPanelNight(SAT_0100), FRI_NIGHT);
  // summer time: 2026-07-06 20:00 EDT = 2026-07-07 00:00Z; and the night the clocks change (2026-11-01)
  assert.equal(dtccTrsPanelNight(at("2026-07-07T03:00:00Z")), "2026-07-07T00:00:00.000Z");
  assert.equal(dtccTrsPanelNight(at("2026-11-01T08:30:00Z")), "2026-11-01T00:00:00.000Z");  // 03:30 EST, night began 20:00 EDT
});

test("decide: nothing outside the window; the first run and every catch-up run start on any tick inside it", () => {
  assert.deepEqual(decideDtccTrsPanel(FRI_0300, { ...base, enabled: false }), { action: "idle", reason: "VOLTRADE_DTCC_TRS_PANEL off" });
  for (const t of [FRI_0400, FRI_1100, FRI_1959]) assert.deepEqual(decideDtccTrsPanel(t, base), { action: "idle", reason: "outside 20:00-04:00 ET" });
  assert.deepEqual(decideDtccTrsPanel(FRI_0300, base), { action: "run", mode: "catch-up", night: THU_NIGHT, reason: "no run yet" });
  assert.deepEqual(decideDtccTrsPanel(SAT_0100, base), { action: "run", mode: "catch-up", night: FRI_NIGHT, reason: "no run yet" });
  const owed = { finishedAt: "2026-11-27T07:40:00+00:00", backlog: 612, read: 41 };
  assert.deepEqual(decideDtccTrsPanel(FRI_0300, { ...base, last: owed }), { action: "run", mode: "catch-up", night: THU_NIGHT, reason: "612 vendor days owed" });
  assert.equal(decideDtccTrsPanel(FRI_1100, { ...base, last: owed }).action, "idle");
  assert.equal(decideDtccTrsPanel(FRI_2010, { ...base, last: owed }).action, "run");
});

test("decide: a catch-up that read nothing, or ended in error tonight, is asked once a night", () => {
  // read nothing on Thursday's night at 02:40 ET: that night has had its run
  const stalled = { finishedAt: "2026-11-27T07:40:00+00:00", backlog: 612, read: 0 };
  assert.deepEqual(decideDtccTrsPanel(FRI_0300, { ...base, last: stalled }), { action: "idle", reason: "already ran tonight (before a restart)" });
  assert.deepEqual(decideDtccTrsPanel(FRI_2010, { ...base, last: stalled }),
    { action: "run", mode: "nightly", night: FRI_NIGHT, reason: "last finished 2026-11-27T07:40:00+00:00" });
  // no last_run.json at all (the Python died before writing one) and an error this night
  assert.deepEqual(decideDtccTrsPanel(FRI_0300, { ...base, catchUpFailedNight: THU_NIGHT }),
    { action: "run", mode: "nightly", night: THU_NIGHT, reason: "no run yet" });
  assert.deepEqual(decideDtccTrsPanel(FRI_0300, { ...base, catchUpFailedNight: THU_NIGHT, lastNightlyAttemptNight: THU_NIGHT }),
    { action: "idle", reason: "already attempted tonight" });
  // the next night it may catch up again
  assert.equal(decideDtccTrsPanel(FRI_2010, { ...base, catchUpFailedNight: THU_NIGHT, lastNightlyAttemptNight: THU_NIGHT }).action, "run");
});

test("decide: a whole panel is run once a night, and not again after a restart", () => {
  const whole = { finishedAt: "2026-11-27T02:00:00+00:00", backlog: 0, read: 1 };   // Thursday 21:00 ET
  assert.deepEqual(decideDtccTrsPanel(FRI_0300, { ...base, last: whole }), { action: "idle", reason: "already ran tonight (before a restart)" });
  assert.deepEqual(decideDtccTrsPanel(FRI_2010, { ...base, last: whole }),
    { action: "run", mode: "nightly", night: FRI_NIGHT, reason: "last finished 2026-11-27T02:00:00+00:00" });
  assert.deepEqual(decideDtccTrsPanel(FRI_2010, { ...base, last: whole, lastNightlyAttemptNight: FRI_NIGHT }),
    { action: "idle", reason: "already attempted tonight" });
  const ranTonight = { finishedAt: "2026-11-28T01:05:00+00:00", backlog: 0, read: 1 };   // Friday 20:05 ET
  assert.equal(decideDtccTrsPanel(SAT_0100, { ...base, last: ranTonight }).action, "idle");
  // a weekend night is a night like any other (Saturday's and Sunday's files exist)
  assert.equal(decideDtccTrsPanel(at("2026-11-29T01:30:00Z"), { ...base, last: ranTonight }).action, "run");
  // a run missed at 20:00 (the server was down) is made later the same night
  assert.equal(decideDtccTrsPanel(SAT_0100, { ...base, last: whole }).action, "run");
});

test("decide: a nearly full volume stops both lanes; the nightly lane says so on the job row", () => {
  const low = DTCC_TRS_PANEL_MIN_FREE_BYTES - 1;
  assert.deepEqual(decideDtccTrsPanel(FRI_0300, { ...base, freeBytes: low }), { action: "skip", mode: "nightly", night: THU_NIGHT, reason: "only 299 MB free on the volume (< 300 MB)" });
  assert.deepEqual(decideDtccTrsPanel(FRI_0300, { ...base, freeBytes: low, lastNightlyAttemptNight: THU_NIGHT }), { action: "idle", reason: "already attempted tonight" });
  // free space that cannot be read does not stop the job
  assert.equal(decideDtccTrsPanel(FRI_0300, { ...base, freeBytes: null }).action, "run");
});

test("last_run.json: the three fields the scheduler reads; absent or broken reads as no run", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "dtcc-trs-panel-"));
  try {
    assert.equal(dtccTrsPanelLastRun(dir), null);
    fs.writeFileSync(path.join(dir, "last_run.json"), WHOLE);
    assert.deepEqual(dtccTrsPanelLastRun(dir), { finishedAt: JSON.parse(WHOLE).finished_at, backlog: 0, read: JSON.parse(WHOLE).read });
    fs.writeFileSync(path.join(dir, "last_run.json"), "[1,2]");
    assert.equal(dtccTrsPanelLastRun(dir), null);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test("summary: the Python's real lines become a job row that fits; days owed and the event check are said", () => {
  const c = parseDtccTrsPanelSummary(`noise\n${CATCH_UP}\n`);
  assert.equal(c.status, "ok");
  assert.match(c.detail, /^read 3; holds 3 days 2024-11-24 to 2024-11-26, \d+ owed \(/);
  assert.ok(!c.detail.includes("events"));
  const w = parseDtccTrsPanelSummary(WHOLE);
  assert.equal(w.status, "ok");
  assert.match(w.detail, /^read \d+; holds 73\d days 2024-11-24 to 2026-11-2\d; events \d+, registered list reproduced \(27869 to 2026-11-24\) \(/);
  for (const o of [c, w]) assert.ok(`catch-up: ${o.detail}`.length <= MAX_DETAIL_CHARS, `${o.detail.length} chars: ${o.detail}`);
});

test("summary: an unread day and an unequal event list come first and loud; a run with no counts is an error", () => {
  const j = JSON.parse(WHOLE);
  const err = parseDtccTrsPanelSummary(JSON.stringify({ ...j, status: "error", errors: 3, last_error: "2025-03-04: URLError: timed out", backlog: 600 }));
  assert.equal(err.status, "error");
  assert.match(err.detail, /^3 day\(s\) unread \(2025-03-04: URLError: timed out\); read /);
  assert.match(err.detail, /, 600 owed/);
  const reg = { through: "2026-11-24", registered: 27869, here: 27871, match: false, months_differing: [{ month: "2025-02", here: 457, registered: 455 }] };
  assert.equal(describeEventsCheck({ events: 27990, registered: reg }),
    "; events 27990, REGISTERED LIST NOT REPRODUCED: 27871 here vs 27869 to 2026-11-24, 1 month(s) differ");
  assert.equal(describeEventsCheck({}), "");
  const cold = parseDtccTrsPanelSummary(JSON.stringify({ ...j, days_unavailable: 2, oldest_unavailable: "2024-11-24" }));
  assert.match(cold.detail, /, 2 refused by the vendor \(oldest 2024-11-24\)/);
  assert.deepEqual(parseDtccTrsPanelSummary(JSON.stringify({ status: "error", reason: "OSError: volume gone" })), { status: "error", detail: "OSError: volume gone" });
  assert.equal(parseDtccTrsPanelSummary("Traceback (most recent call last):").status, "error");
  assert.equal(parseDtccTrsPanelSummary(JSON.stringify({ status: "ok" })).status, "error");
});

test("run: the long-job lane's caps and budget; exit 1 with a summary is that summary; a timeout says the days are kept", async () => {
  let seen: { cmd: string; opts: Record<string, unknown> } | null = null;
  const ok: WalkforwardExec = async (cmd, opts) => { seen = { cmd, opts: opts as unknown as Record<string, unknown> }; return { stdout: WHOLE, stderr: "" }; };
  const out = await runDtccTrsPanel({ PATH: "/usr/bin" }, ok);
  assert.equal(out.status, "ok");
  assert.ok(seen!.cmd.startsWith("python3 dtcc_trs_panel_job.py --run --nice 19"));
  assert.equal(seen!.opts.timeout, DTCC_TRS_PANEL_TIMEOUT_MS);
  assert.equal(seen!.opts.killSignal, "SIGKILL");
  for (const [k, v] of Object.entries(WALKFORWARD_ENV_CAPS)) assert.equal((seen!.opts.env as Record<string, string>)[k], v);
  const j = JSON.parse(WHOLE);
  const exit1: WalkforwardExec = async () => { throw Object.assign(new Error("exit 1"), { code: 1, stdout: JSON.stringify({ ...j, status: "error", errors: 1, last_error: "2025-03-04: BadZipFile" }), stderr: "" }); };
  const e1 = await runDtccTrsPanel({}, exit1);
  assert.equal(e1.status, "error");
  assert.match(e1.detail, /^1 day\(s\) unread \(2025-03-04: BadZipFile\)/);
  const killed: WalkforwardExec = async () => { throw Object.assign(new Error("killed"), { killed: true, signal: "SIGKILL" }); };
  assert.deepEqual(await runDtccTrsPanel({}, killed), { status: "error", detail: "exceeded the 19-minute budget — process group killed (days already written are kept)" });
  const crashed: WalkforwardExec = async () => { throw Object.assign(new Error("boom"), { code: 2, stderr: "python3: can't open file" }); };
  assert.equal((await runDtccTrsPanel({}, crashed)).detail, "code=2 signal=none: python3: can't open file");
});
