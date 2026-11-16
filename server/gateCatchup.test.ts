// gateCatchup — the one rule for repeating a weekly gate job whose registered
// Sunday run died on plumbing (open_questions.md 2026-11-16; KNOWN BROKEN #53/#54).
// Run: npx tsx --test server/gateCatchup.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { catchupDecision, gateCatchupFlag, CATCHUP_WEEKDAYS } from "./gateCatchup";
import { decideGithubGate2, GITHUB_GATE2_WINDOW_START_MIN, GITHUB_GATE2_WINDOW_END_MIN } from "./githubGate2";
import type { JobRun } from "./jobHeartbeat";

const here = path.dirname(fileURLToPath(import.meta.url));
const W = { windowStartMin: 19 * 60, windowEndMin: 20 * 60 };
// The real case: github_gate2_weekly's first registered run, Sun 2026-11-15
// 19:26:45 ET (00:26:45Z Mon), ended 83 ms later "alpaca HTTP 400".
const SUN_ERR: JobRun = { startedAt: Date.parse("2026-11-16T00:26:45Z"), finishedAt: Date.parse("2026-11-16T00:26:45.083Z"), status: "error", detail: "error: alpaca HTTP 400" };
const MON_IN = Date.parse("2026-11-17T00:26:00Z");   // Mon 11-16 19:26 ET
const TUE_IN = Date.parse("2026-11-18T00:26:00Z");   // Tue 11-17 19:26 ET
const WED_IN = Date.parse("2026-11-19T00:26:00Z");   // Wed 11-18 19:26 ET
const inp = (runs: JobRun[] | undefined, latestGeneratedAt: string | null = null, enabled = true) => ({ enabled, runs, latestGeneratedAt, ...W });

test("REGRESSION 2026-11-15: a Sunday run that errored with no result is repeated Monday in the same hour", () => {
  const d = catchupDecision(MON_IN, inp([SUN_ERR]));
  assert.equal(d.eligible, true);
  assert.equal(d.eligible && d.registeredSunday, "2026-11-15");
  assert.equal(d.eligible && d.etDate, "2026-11-16");
  assert.match(d.reason, /catch-up for the 2026-11-15 run: 1 attempt\(s\) since, all errored, no result written/);
});

test("only Monday and Tuesday, only inside the job's own clock window", () => {
  assert.deepEqual(Object.keys(CATCHUP_WEEKDAYS), ["Mon", "Tue"]);
  assert.equal(catchupDecision(TUE_IN, inp([SUN_ERR])).eligible, true, "Tuesday is the second and last day");
  assert.equal(catchupDecision(WED_IN, inp([SUN_ERR])).eligible, false, "Wednesday: wait for Sunday");
  assert.equal(catchupDecision(Date.parse("2026-11-16T23:30:00Z"), inp([SUN_ERR])).eligible, false, "Mon 18:30 ET is before the window");
  assert.equal(catchupDecision(Date.parse("2026-11-17T01:00:00Z"), inp([SUN_ERR])).eligible, false, "Mon 20:00 ET: the window is half-open");
  assert.equal(catchupDecision(Date.parse("2026-11-16T00:40:00Z"), inp([SUN_ERR])).eligible, false, "Sunday itself is the registered run, not a catch-up");
});

test("one look stays one look: a verdict, a skip, an open run or a written result never retries", () => {
  const sunOk: JobRun = { ...SUN_ERR, status: "ok", detail: "ok: verdict FAIL" };
  const sunSkip: JobRun = { ...SUN_ERR, status: "skipped", detail: "fresh" };
  const sunOpen: JobRun = { startedAt: SUN_ERR.startedAt };
  for (const [name, runs] of [["ok", [sunOk]], ["skipped", [sunSkip]], ["still open", [sunOpen]], ["error then ok", [SUN_ERR, { ...sunOk, startedAt: SUN_ERR.startedAt + 1000 }]]] as const) {
    const d = catchupDecision(MON_IN, inp([...runs]));
    assert.equal(d.eligible, false, name);
    assert.match(d.reason, /one look stays one look/, name);
  }
  // errored AFTER writing a result (generated at or after the Sunday window opened): the look is spent
  const spent = catchupDecision(MON_IN, inp([SUN_ERR], "2026-11-16T00:26:45.000Z"));
  assert.equal(spent.eligible, false);
  assert.match(spent.reason, /the look is spent/);
  // a result from the Sunday BEFORE is not this week's look
  assert.equal(catchupDecision(MON_IN, inp([SUN_ERR], "2026-11-09T00:26:45.000Z")).eligible, true);
  assert.equal(catchupDecision(MON_IN, inp([SUN_ERR], "not a date")).eligible, true, "an unreadable stamp is not a result");
});

test("nothing to catch up when the job never started since the Sunday window opened (never before a registered date)", () => {
  assert.equal(catchupDecision(MON_IN, inp(undefined)).eligible, false);
  assert.equal(catchupDecision(MON_IN, inp([])).eligible, false);
  const lastWeek: JobRun = { ...SUN_ERR, startedAt: SUN_ERR.startedAt - 7 * 86_400_000, finishedAt: SUN_ERR.startedAt - 7 * 86_400_000 + 80 };
  const d = catchupDecision(MON_IN, inp([lastWeek]));
  assert.equal(d.eligible, false, "an error from the Sunday before is not this week's attempt");
  assert.match(d.reason, /no attempt since the 2026-11-15 window opened/);
  // an error earlier on the same Sunday, before the window opened, is not an attempt at the registered run either
  const sunEarly: JobRun = { ...SUN_ERR, startedAt: Date.parse("2026-11-15T20:00:00Z"), finishedAt: Date.parse("2026-11-15T20:00:01Z") };
  assert.equal(catchupDecision(MON_IN, inp([sunEarly])).eligible, false);
});

test("at most one catch-up attempt per ET day, and the evidence survives a deploy (it is the persisted runs)", () => {
  const monErr: JobRun = { startedAt: MON_IN, finishedAt: MON_IN + 90, status: "error", detail: "error: alpaca HTTP 500" };
  const later = MON_IN + 30 * 60_000;                       // Mon 19:56 ET, a fresh process with empty memory
  const d = catchupDecision(later, inp([SUN_ERR, monErr]));
  assert.equal(d.eligible, false);
  assert.match(d.reason, /already attempted today \(2026-11-16\)/);
  const tue = catchupDecision(TUE_IN, inp([SUN_ERR, monErr]));
  assert.equal(tue.eligible, true, "Tuesday gets the second and last attempt");
  assert.match(tue.reason, /2 attempt\(s\) since/);
  const tueErr: JobRun = { startedAt: TUE_IN, finishedAt: TUE_IN + 90, status: "error" };
  assert.equal(catchupDecision(TUE_IN + 20 * 60_000, inp([SUN_ERR, monErr, tueErr])).eligible, false);
  // a run interrupted by a deploy is closed as "error" on load (closeInterruptedRuns) and counts the same way
  const interrupted: JobRun = { startedAt: SUN_ERR.startedAt, finishedAt: SUN_ERR.startedAt, status: "error", detail: "interrupted: the process exited before this run finished" };
  assert.equal(catchupDecision(MON_IN, inp([interrupted])).eligible, true);
});

test("off switch: VOLTRADE_GATE_CATCHUP=0 restores Sunday-only", () => {
  for (const [v, on] of [[undefined, true], ["1", true], ["0", false], ["off", false], ["", false]] as const) {
    assert.equal(gateCatchupFlag(v === undefined ? {} : { VOLTRADE_GATE_CATCHUP: v }).on, on);
  }
  const d = catchupDecision(MON_IN, inp([SUN_ERR], null, false));
  assert.equal(d.eligible, false);
  assert.match(d.reason, /VOLTRADE_GATE_CATCHUP off/);
});

test("the Sunday window's DST edge: the week the clocks change still finds its Sunday", () => {
  // Sun 2027-03-14 is the US spring-forward day; 19:26 EDT = 23:26Z. Mon 19:26 EDT = 23:26Z Mon.
  const sun: JobRun = { startedAt: Date.parse("2027-03-14T23:26:00Z"), finishedAt: Date.parse("2027-03-14T23:26:01Z"), status: "error" };
  const d = catchupDecision(Date.parse("2027-03-15T23:26:00Z"), inp([sun]));
  assert.equal(d.eligible && d.registeredSunday, "2027-03-14");
});

test("github_gate2_weekly opts in: Monday catch-up runs, with the reason in the decision; without the runs it is Sunday-only", () => {
  const base = { enabled: true, latest: null, lastAttemptWeek: "", codeVersion: "1.0.1", freeBytes: 5e9 as number | null };
  assert.equal(GITHUB_GATE2_WINDOW_START_MIN, W.windowStartMin);
  assert.equal(GITHUB_GATE2_WINDOW_END_MIN, W.windowEndMin);
  assert.equal(decideGithubGate2(MON_IN, base).action, "idle", "no catchup input = the old Sunday-only behaviour");
  const run = decideGithubGate2(MON_IN, { ...base, catchup: { enabled: true, runs: [SUN_ERR] } });
  assert.equal(run.action, "run");
  assert.match(run.reason, /^catch-up for the 2026-11-15 run: .*no result yet$/);
  assert.equal((run as { week: string }).week, "catchup:2026-11-16");
  // the in-memory guard: the same process does not start it twice that day
  assert.equal(decideGithubGate2(MON_IN + 60_000, { ...base, lastAttemptWeek: "catchup:2026-11-16", catchup: { enabled: true, runs: [SUN_ERR] } }).action, "idle");
  // settled Sunday -> idle on Monday, and the reason says why
  const idle = decideGithubGate2(MON_IN, { ...base, catchup: { enabled: true, runs: [{ ...SUN_ERR, status: "ok" }] } });
  assert.equal(idle.action, "idle");
  assert.match(idle.reason, /no catch-up: .*one look stays one look/);
  assert.equal(decideGithubGate2(MON_IN, { ...base, catchup: { enabled: false, runs: [SUN_ERR] } }).action, "idle", "switch off");
  // the job's own gates still apply to a catch-up tick
  assert.equal(decideGithubGate2(MON_IN, { ...base, enabled: false, catchup: { enabled: true, runs: [SUN_ERR] } }).action, "idle");
  assert.equal(decideGithubGate2(MON_IN, { ...base, freeBytes: 1, catchup: { enabled: true, runs: [SUN_ERR] } }).action, "skip");
  // Sunday is unchanged by the new input
  const SUN_IN = Date.parse("2026-11-23T00:30:00Z");
  const sunday = decideGithubGate2(SUN_IN, { ...base, catchup: { enabled: true, runs: [SUN_ERR] } });
  assert.equal(sunday.action, "run");
  assert.equal(sunday.reason, "no result yet");
});

test("RATCHET: who opts in is written down — the boot tick feeds the persisted runs; the heavy jobs do not import the rule", () => {
  const read = (f: string) => fs.readFileSync(path.join(here, f), "utf8");
  const gh = read("githubGate2.ts");
  assert.match(gh, /catchup: \{ enabled: gateCatchupFlag\(env\)\.on, runs: heartbeatStore\(\)\.snapshot\(\)\.jobs\.github_gate2_weekly \}/);
  // earnings_qoq (~1,000 per-symbol Alpaca requests), lazy_prices and edge_p2 (1.5 GB, hours) were
  // scheduled for an idle Sunday lane; opting one in needs a shown-free weekday lane and its own PR.
  for (const f of ["earningsQoq.ts", "lazyPrices.ts", "edgeP2.ts"]) {
    assert.doesNotMatch(read(f), /gateCatchup/, `${f} must not opt in without a lane decision (update this test in that PR)`);
  }
});
