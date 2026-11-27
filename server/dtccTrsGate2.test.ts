// dtccTrsGate2 — the weekly job that runs dtcc_trs_gate2.py, the registered
// gate-2 test of the DTCC equity-swap root (research/open_questions.md
// 2026-11-25; the runner is (as2) step 2, 2026-11-27), and its diag probe.
// Run: npx tsx --test server/dtccTrsGate2.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  dtccTrsGate2Cmd, dtccTrsGate2Flag, dtccTrsGate2Final, dtccTrsGate2Status, decideDtccTrsGate2, inDtccTrsGate2Window,
  parseDtccTrsGate2Summary, runDtccTrsGate2,
  DTCC_TRS_GATE2_JOB, DTCC_TRS_GATE2_TIMEOUT_MS, DTCC_TRS_GATE2_WINDOW_END_MIN, DTCC_TRS_GATE2_WINDOW_START_MIN,
  DTCC_TRS_GATE2_MIN_FREE_BYTES, DTCC_TRS_GATE2_WAITS_FOR, DTCC_TRS_GATE2_MAX_RSS_MB,
} from "./dtccTrsGate2";
import { DTCC_TRS_PANEL_JOB, DTCC_TRS_PANEL_WINDOW_START_MIN } from "./dtccTrsPanel";
import { busyPanelReader, PANEL_READER_JOBS } from "./eventStudy";
import { WALKFORWARD_ENV_CAPS, type WalkforwardExec } from "./walkforward";
import { JOB_REGISTRY, type HeartbeatFile, type JobRun } from "./jobHeartbeat";
import { sanitizeDiag, DIAG_PROBES } from "./diag";

const here = path.dirname(fileURLToPath(import.meta.url));
const at = (iso: string) => Date.parse(iso);
// 2026-11-29 is a Sunday (EST = UTC-5)
const SUN_1810 = at("2026-11-29T23:10:00Z");
const SUN_1759 = at("2026-11-29T22:59:00Z");
const SUN_1914 = at("2026-11-30T00:14:00Z");
const SUN_1915 = at("2026-11-30T00:15:00Z");
const SAT_1810 = at("2026-11-28T23:10:00Z");
const MON_1810 = at("2026-11-30T23:10:00Z");
const base = { enabled: true, panelBuildId: "b20261129" as string | null, final: null as string | null, lastAttemptWeek: "", freeBytes: 5e9 as number | null, busy: null as string | null };

test("registry and flag: a weekly interval job whose run limit covers the budget; env present wins", () => {
  const job = JOB_REGISTRY.find((j) => j.id === DTCC_TRS_GATE2_JOB);
  assert.ok(job && job.kind === "interval" && job.source === "server/dtccTrsGate2.ts");
  assert.ok(job!.maxRunMs >= DTCC_TRS_GATE2_TIMEOUT_MS);
  for (const [v, on] of [[undefined, true], ["1", true], ["0", false], ["off", false], ["", false]] as const) {
    const env = v === undefined ? {} : { VOLTRADE_DTCC_TRS_GATE2: v };
    assert.equal(dtccTrsGate2Flag(env).on, on);
    assert.equal(job!.enabledWhen!(env), on);
  }
});

test("window: Sunday 18:00-19:15 ET only, and the last start plus the kill ends before the nightly panel job may rewrite the events", () => {
  assert.equal(inDtccTrsGate2Window(SUN_1810), true);
  assert.equal(inDtccTrsGate2Window(SUN_1914), true);
  for (const t of [SUN_1759, SUN_1915, SAT_1810, MON_1810]) assert.equal(inDtccTrsGate2Window(t), false);
  const lastEndMin = DTCC_TRS_GATE2_WINDOW_END_MIN + DTCC_TRS_GATE2_TIMEOUT_MS / 60_000;
  assert.ok(lastEndMin <= DTCC_TRS_PANEL_WINDOW_START_MIN, `a run started at the window's end is killed by ${lastEndMin} min, after the 20:00 ET panel job`);
  assert.ok(DTCC_TRS_GATE2_WINDOW_START_MIN >= 16 * 60, "never inside regular market hours");
});

test("decide: one attempt a week; a verdict of record is never looked at again; waiting does not spend the week", () => {
  assert.equal(decideDtccTrsGate2(SUN_1810, { ...base, enabled: false }).action, "idle");
  assert.equal(decideDtccTrsGate2(SAT_1810, base).action, "idle");
  const run = decideDtccTrsGate2(SUN_1810, base);
  assert.equal(run.action, "run");
  const week = (run as { week: string }).week;
  assert.equal(decideDtccTrsGate2(SUN_1810, { ...base, lastAttemptWeek: week }).action, "idle");
  for (const final of ["KILLED", "PASS"]) {
    const d = decideDtccTrsGate2(SUN_1810, { ...base, final });
    assert.equal(d.action, "skip");
    assert.match(d.reason, new RegExp(`verdict of record: ${final}`));
  }
  // a verdict of record wins over everything else: no run even when nothing is in the way
  assert.equal(decideDtccTrsGate2(SUN_1810, { ...base, final: "KILLED", busy: "price_panel_build" }).action, "skip");
  const waiting = decideDtccTrsGate2(SUN_1810, { ...base, busy: "lazy_prices_weekly" });
  assert.equal(waiting.action, "idle");
  assert.match(waiting.reason, /waiting for lazy_prices_weekly/);
  assert.equal(decideDtccTrsGate2(SUN_1810, { ...base, panelBuildId: null }).action, "skip");
  assert.equal(decideDtccTrsGate2(SUN_1810, { ...base, freeBytes: DTCC_TRS_GATE2_MIN_FREE_BYTES - 1 }).action, "skip");
  assert.equal(decideDtccTrsGate2(SUN_1810, { ...base, freeBytes: null }).action, "run");
});

test("it waits for every panel reader, the price build, the two Sunday-evening runs and the job that writes its events", () => {
  for (const id of [...PANEL_READER_JOBS, "event_study_weekly", "lazy_prices_weekly", "earnings_qoq_weekly", DTCC_TRS_PANEL_JOB]) {
    assert.ok(DTCC_TRS_GATE2_WAITS_FOR.includes(id), id);
    assert.ok(JOB_REGISTRY.some((j) => j.id === id), `${id} is a registered job`);
  }
  assert.ok(!DTCC_TRS_GATE2_WAITS_FOR.includes(DTCC_TRS_GATE2_JOB));
  const now = SUN_1810;
  const snap = (jobs: Record<string, Partial<JobRun>[]>): HeartbeatFile => ({ version: 1, jobs: jobs as Record<string, JobRun[]> } as HeartbeatFile);
  assert.equal(busyPanelReader(snap({ lazy_prices_weekly: [{ startedAt: now - 60_000 }] }), now, DTCC_TRS_GATE2_WAITS_FOR), "lazy_prices_weekly");
  // the default list is unchanged for the event study, which does not wait on that job
  assert.equal(busyPanelReader(snap({ lazy_prices_weekly: [{ startedAt: now - 60_000 }] }), now), null);
  assert.equal(busyPanelReader(snap({ lazy_prices_weekly: [{ startedAt: now - 60_000, finishedAt: now - 1000, status: "ok" }] }), now, DTCC_TRS_GATE2_WAITS_FOR), null);
});

test("command: a root-level file the image ships, niced, with the memory budget and a safe version", () => {
  const cmd = dtccTrsGate2Cmd("1.0.1571");
  assert.equal(cmd, `python3 dtcc_trs_gate2.py --run --nice 19 --max-rss-mb ${DTCC_TRS_GATE2_MAX_RSS_MB} --code-version 1.0.1571`);
  assert.match(dtccTrsGate2Cmd("1.0; rm -rf /"), /--code-version unknown$/);
  assert.ok(fs.existsSync(path.join(here, "..", "dtcc_trs_gate2.py")), "the script is at the repo root (the image copies root *.py only)");
});

// The Python's own last stdout lines (test_dtcc_trs_gate2.py runs produce these shapes).
const LOOK = "{\"status\":\"ok\",\"look\":1,\"verdict\":\"KILLED\",\"final\":\"KILLED\",\"tested\":4120,\"mean_pct\":-0.8123,\"t\":-2.41,\"build_id\":\"b20261129\",\"elapsed_s\":212.4,\"peak_rss_mb\":612.0}";
const INCONCLUSIVE = "{\"status\":\"ok\",\"look\":1,\"verdict\":\"INCONCLUSIVE\",\"final\":null,\"tested\":4120,\"mean_pct\":0.31,\"t\":1.7,\"build_id\":\"b20261129\",\"elapsed_s\":212.4,\"peak_rss_mb\":612.0}";
const NOT_ARMED = "{\"status\":\"not_armed\",\"reason\":\"priced share 0.41 (needs 0.8), median comparison names 212 (needs 30): no statistic computed, no look taken\",\"build_id\":\"b20261129\",\"tested\":null,\"elapsed_s\":90.1,\"peak_rss_mb\":400.0}";

test("summary: a look reads ok with its verdict; a run that took no look never reads ok; anything else is an error", () => {
  const ok = parseDtccTrsGate2Summary(`noise\n${LOOK}\n`);
  assert.equal(ok.status, "ok");
  assert.match(ok.detail, /^look 1: KILLED, 4120 tested events, statistic -0\.8123% t -2\.41, build b20261129/);
  assert.match(parseDtccTrsGate2Summary(INCONCLUSIVE).detail, /^look 1: INCONCLUSIVE \(not final\)/);
  const na = parseDtccTrsGate2Summary(NOT_ARMED);
  assert.equal(na.status, "insufficient");
  assert.match(na.detail, /^NOT ARMED, no look taken: priced share 0\.41/);
  assert.equal(parseDtccTrsGate2Summary("{\"status\":\"insufficient\",\"reason\":\"212 tested events, under 300\"}").status, "insufficient");
  assert.equal(parseDtccTrsGate2Summary("{\"status\":\"skipped\",\"reason\":\"verdict of record: KILLED\"}").status, "skipped");
  assert.equal(parseDtccTrsGate2Summary("{\"status\":\"error\",\"reason\":\"memory budget: 950 MB > 900 MB\"}").status, "error");
  assert.equal(parseDtccTrsGate2Summary("Traceback (most recent call last):").status, "error");
  assert.equal(parseDtccTrsGate2Summary("").status, "error");
});

test("run: the long-job lane's caps and budget; a kill and a crash read error and never throw", async () => {
  let seen: { cmd: string; opts: Record<string, unknown> } | null = null;
  const good: WalkforwardExec = async (cmd, opts) => { seen = { cmd, opts: opts as unknown as Record<string, unknown> }; return { stdout: LOOK, stderr: "" }; };
  const out = await runDtccTrsGate2({ PATH: "/usr/bin" }, good);
  assert.equal(out.status, "ok");
  assert.match(seen!.cmd, /^python3 dtcc_trs_gate2\.py --run --nice 19 /);
  assert.equal(seen!.opts.timeout, DTCC_TRS_GATE2_TIMEOUT_MS);
  assert.equal(seen!.opts.killSignal, "SIGKILL");
  for (const [k, v] of Object.entries(WALKFORWARD_ENV_CAPS)) assert.equal((seen!.opts.env as Record<string, string>)[k], v);
  const killed: WalkforwardExec = async () => { throw Object.assign(new Error("killed"), { killed: true, signal: "SIGKILL" }); };
  const k = await runDtccTrsGate2({}, killed);
  assert.equal(k.status, "error");
  assert.match(k.detail, /40-minute budget/);
  const crash: WalkforwardExec = async () => { throw Object.assign(new Error("exit 1"), { code: 1, stdout: "{\"status\":\"error\",\"reason\":\"KeyError: 'cusip'\"}", stderr: "" }); };
  assert.equal((await runDtccTrsGate2({}, crash)).detail, "KeyError: 'cusip'");
  const dead: WalkforwardExec = async () => { throw Object.assign(new Error("spawn failed"), { code: "ENOENT", stderr: "python3: not found" }); };
  assert.match((await runDtccTrsGate2({}, dead)).detail, /code=ENOENT.*python3: not found/);
});

test("probe: registered, token-gated list; before any run it answers with no look; after a look it returns the looks of record and names no security", () => {
  assert.ok((DIAG_PROBES as readonly string[]).includes("dtcc_trs_gate2"));
  const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
  assert.match(bot, /case "dtcc_trs_gate2": \{[\s\S]{0,400}sanitizeDiag\(dtccTrsGate2Status\(\)\)/);
  const routes = fs.readFileSync(path.join(here, "routes.ts"), "utf8");
  assert.match(routes, /\n  bootDtccTrsGate2\(\);\n/);
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "dtcc-gate2-"));
  try {
    const root = path.join(dir, "dtcc_trs_gate2");
    const empty = dtccTrsGate2Status({ DATA_DIR: dir }, root, path.join(dir, "price_panel"));
    assert.deepEqual(empty.registered, { looks: [], final: null, ladder_verdict: null, first_sufficient_run: null });
    assert.equal(empty.latest, null);
    assert.equal(dtccTrsGate2Final(root), null);
    fs.mkdirSync(root, { recursive: true });
    const look = { look: 1, at: "2026-11-29T23:20:00+00:00", verdict: "KILLED", tested: 4120, pooled: { n: 4120, mean_pct: -0.8123, t: -2.41 } };
    fs.writeFileSync(path.join(root, "registered.json"), JSON.stringify({ schema: "dtcc_trs_gate2/1", looks: [look], final: "KILLED", ladder_verdict: "FAIL" }));
    fs.writeFileSync(path.join(root, "latest.json"), JSON.stringify({ status: "ok", verdict: "KILLED", counts: { events_tested: 4120 } }));
    assert.equal(dtccTrsGate2Final(root), "KILLED");
    const s = sanitizeDiag(dtccTrsGate2Status({ DATA_DIR: dir }, root, path.join(dir, "price_panel"))) as Record<string, unknown>;
    const reg = s.registered as { looks: unknown[]; final: string; ladder_verdict: string; first_sufficient_run: Record<string, unknown> };
    assert.equal(reg.final, "KILLED");
    assert.equal(reg.ladder_verdict, "FAIL");
    assert.equal(reg.looks.length, 1);
    assert.equal(reg.first_sufficient_run.verdict, "KILLED");
    assert.equal(reg.first_sufficient_run.ladder_verdict_of_record, "FAIL");
    // an INCONCLUSIVE first look is a look of record with no ladder verdict yet
    fs.writeFileSync(path.join(root, "registered.json"), JSON.stringify({ looks: [{ ...look, verdict: "INCONCLUSIVE" }], final: null, ladder_verdict: null }));
    const open = dtccTrsGate2Status({ DATA_DIR: dir }, root, path.join(dir, "price_panel")).registered as { first_sufficient_run: Record<string, unknown> };
    assert.equal(open.first_sufficient_run.verdict, "INCONCLUSIVE");
    assert.equal(open.first_sufficient_run.ladder_verdict_of_record, null);
    assert.equal(dtccTrsGate2Final(root), null);
    assert.match(String(s.registration), /One look/);
    // the Python that writes these files is held to "no CUSIP, ticker or price" by its own test; the status adds none
    assert.ok(!/cusip"\s*:/i.test(JSON.stringify(s.latest)) && !/"ticker"\s*:/i.test(JSON.stringify(s)));
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test("the ladder row names this probe and the path scripts/ladder_verdict_check.py reads", () => {
  const ladder = JSON.parse(fs.readFileSync(path.join(here, "..", "datacore", "signal_ladder.json"), "utf8")) as { roots: Array<Record<string, unknown>> };
  const row = ladder.roots.find((r) => r.id === "dtcc_sbsdr_equity_swaps");
  assert.deepEqual(row?.registered_verdict, { probe: "dtcc_trs_gate2", path: "registered.first_sufficient_run.ladder_verdict_of_record", gate: 2 });
  assert.equal(row?.status, "gate2_pending", "a runner is not a verdict: the status changes only when a session reads a look");
});
