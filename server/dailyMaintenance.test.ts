// REPAIR 2026-10-27 (trading-engine audit D2): the once-a-day "4am branch"
// (daily counter resets, ML retrain, nightly backup) was nested inside
// runOvernightResearch, whose caller only runs when ET < 4 or ET >= 20 and
// whose scheduler only enters between 04:00 and 20:00 ET — so
// `Math.floor(etHour) === 4` could never be true. These tests pin:
//   1. the pure decision table (window edges, DST, ET-not-UTC dates),
//   2. exactly one run per ET day across 5-minute polls AND process restarts
//      (the claim is persisted before work starts; memory guards a failed write),
//   3. the bot.ts wiring: own unconditional interval, the resets live in
//      runDailyMaintenance, and the dead branch is gone from runOvernightResearch.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  claimDailyMaintenance, etClock, latestEtDate, loadLastRunEtDate, parseDailyMaintenanceRecord,
  saveLastRunEtDate, shouldRunDailyMaintenance, DAILY_MAINTENANCE_CHECK_MS,
} from "./dailyMaintenance";

const here = path.dirname(fileURLToPath(import.meta.url));
const ms = (iso: string) => Date.parse(iso);

function tmpPaths(): string[] {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "daily-maint-"));
  return [path.join(dir, "vol", "voltrade_daily_maintenance.json"), path.join(dir, "tmp", "voltrade_daily_maintenance.json")];
}

test("etClock: ET date and hour, DST-correct, ET calendar date (not UTC)", () => {
  assert.deepEqual(etClock(ms("2026-10-27T08:00:00Z")), { etDate: "2026-10-27", etHour: 4 }); // EDT, UTC-4
  assert.deepEqual(etClock(ms("2026-12-15T09:00:00Z")), { etDate: "2026-12-15", etHour: 4 }); // EST, UTC-5
  assert.deepEqual(etClock(ms("2026-10-28T02:00:00Z")), { etDate: "2026-10-27", etHour: 22 }); // UTC next day, ET same day
  assert.deepEqual(etClock(ms("2026-10-27T04:00:00Z")), { etDate: "2026-10-27", etHour: 0 }); // midnight ET is hour 0, not 24
  assert.equal(etClock(ms("2026-10-27T13:29:00Z")).etHour, 9 + 29 / 60);
});

test("shouldRunDailyMaintenance: decision table", () => {
  const cases: Array<[string, string | null, boolean, string]> = [
    // [now (UTC), lastRunEtDate, run?, reason]
    ["2026-10-27T07:59:00Z", null, false, "before-window"],         // 03:59 ET
    ["2026-10-27T08:00:00Z", null, true, "due"],                    // 04:00 ET, first boot ever
    ["2026-10-27T08:00:00Z", "2026-10-26", true, "due"],            // 04:00 ET, ran yesterday
    ["2026-10-27T08:05:00Z", "2026-10-27", false, "already-ran-today"],
    ["2026-10-27T13:29:00Z", "2026-10-26", true, "due"],            // 09:29 ET: restart mid-window still catches up
    ["2026-10-27T13:29:00Z", "2026-10-27", false, "already-ran-today"],
    ["2026-10-27T13:30:00Z", "2026-10-26", false, "after-window"],  // 09:30 ET: market open, missed day is skipped
    ["2026-10-27T20:00:00Z", null, false, "after-window"],          // 16:00 ET
    ["2026-10-28T03:30:00Z", "2026-10-26", false, "after-window"],  // 23:30 ET on 10-27 (UTC already 10-28)
    ["2026-10-28T05:00:00Z", "2026-10-27", false, "before-window"], // 01:00 ET on 10-28
    ["2026-12-15T08:59:00Z", null, false, "before-window"],         // 03:59 EST (would be 04:59 under EDT)
    ["2026-12-15T09:00:00Z", "2026-12-14", true, "due"],            // 04:00 EST
    ["2026-03-08T08:00:00Z", "2026-03-07", true, "due"],            // DST start day, 04:00 EDT
    ["2026-11-01T09:00:00Z", "2026-10-31", true, "due"],            // DST end day, 04:00 EST
    ["2026-10-31T08:00:00Z", "2026-10-30", true, "due"],            // Saturday: daily, weekends included
  ];
  for (const [now, last, run, reason] of cases) {
    const d = shouldRunDailyMaintenance(ms(now), last);
    assert.equal(d.run, run, `${now} last=${last}: run`);
    assert.equal(d.reason, reason, `${now} last=${last}: reason`);
  }
});

test("record parsing + latestEtDate: invalid records are ignored, newest valid date wins", () => {
  assert.equal(parseDailyMaintenanceRecord({ lastRunEtDate: "2026-10-27" }), "2026-10-27");
  for (const bad of [null, 5, "2026-10-27", {}, { lastRunEtDate: 20261027 }, { lastRunEtDate: "10/27/2026" }, { lastRunEtDate: "" }]) {
    assert.equal(parseDailyMaintenanceRecord(bad), null, JSON.stringify(bad));
  }
  assert.equal(latestEtDate(null, null), null);
  assert.equal(latestEtDate("2026-10-27", null), "2026-10-27");
  assert.equal(latestEtDate(null, "2026-10-27"), "2026-10-27");
  assert.equal(latestEtDate("2026-10-26", "2026-10-27"), "2026-10-27");
  assert.equal(latestEtDate("2026-10-27", "2026-09-30"), "2026-10-27");
});

test("persistence: save then load round-trips; the newer of the two paths wins; corrupt files are ignored", () => {
  const [vol, tmp] = tmpPaths();
  assert.equal(loadLastRunEtDate([vol, tmp]), null, "no files yet");
  assert.equal(saveLastRunEtDate("2026-10-26", ms("2026-10-26T08:00:00Z"), [vol, tmp]), vol, "volume path first");
  assert.equal(loadLastRunEtDate([vol, tmp]), "2026-10-26");
  // A day the volume write failed over to /tmp: the /tmp record is newer and must win.
  fs.mkdirSync(path.dirname(tmp), { recursive: true });
  fs.writeFileSync(tmp, JSON.stringify({ lastRunEtDate: "2026-10-27" }));
  assert.equal(loadLastRunEtDate([vol, tmp]), "2026-10-27");
  fs.writeFileSync(vol, "{not json");
  assert.equal(loadLastRunEtDate([vol, tmp]), "2026-10-27", "a corrupt volume file does not hide the valid /tmp one");
  // Every path unwritable (parent is a regular file): null, not a throw.
  const blocker = path.join(path.dirname(path.dirname(vol)), "blocker");
  fs.writeFileSync(blocker, "x");
  assert.equal(saveLastRunEtDate("2026-10-28", 0, [path.join(blocker, "a.json"), path.join(blocker, "b.json")]), null);
});

/**
 * Drive claimDailyMaintenance exactly as bot.ts's poll does (5-minute ticks,
 * an in-memory last-run that a restart wipes) and count the runs per ET date.
 */
function simulate(fromIso: string, toIso: string, restartsAt: string[], paths: string[]) {
  const restarts = new Set(restartsAt.map(ms));
  const runs: Record<string, number> = {};
  let mem: string | null = null;
  for (let t = ms(fromIso); t <= ms(toIso); t += DAILY_MAINTENANCE_CHECK_MS) {
    if (restarts.has(t)) mem = null; // process restart: memory gone, disk survives
    const { decision } = claimDailyMaintenance(t, mem, paths);
    if (decision.run) {
      mem = decision.etDate;
      runs[decision.etDate] = (runs[decision.etDate] || 0) + 1;
    }
  }
  return runs;
}

test("exactly once per ET day over three days of polls, with restarts before, inside and after the window", () => {
  const runs = simulate("2026-10-27T00:00:00Z", "2026-10-30T03:55:00Z", [
    "2026-10-27T07:55:00Z", // 03:55 ET: restart just before the window
    "2026-10-27T08:10:00Z", // 04:10 ET: restart right after the run — must NOT repeat
    "2026-10-27T12:00:00Z", // 08:00 ET: another deploy mid-window
    "2026-10-28T18:00:00Z", // mid-day deploy
    "2026-10-29T13:00:00Z", // 09:00 ET deploy (after that day's run)
  ], tmpPaths());
  assert.deepEqual(runs, { "2026-10-27": 1, "2026-10-28": 1, "2026-10-29": 1 });
});

test("a process that was down at 04:00 and boots mid-window still runs that day, once", () => {
  const paths = tmpPaths();
  saveLastRunEtDate("2026-10-26", ms("2026-10-26T08:00:00Z"), paths);
  // Down from 03:00 to 07:45 ET; polls only resume after the boot.
  const runs = simulate("2026-10-27T11:45:00Z", "2026-10-27T23:00:00Z", ["2026-10-27T11:45:00Z"], paths);
  assert.deepEqual(runs, { "2026-10-27": 1 });
});

test("unwritable state paths: memory still holds the day within one process", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "daily-maint-ro-"));
  const blocker = path.join(dir, "blocker");
  fs.writeFileSync(blocker, "x");
  const paths = [path.join(blocker, "a.json"), path.join(blocker, "b.json")];
  const first = claimDailyMaintenance(ms("2026-10-27T08:00:00Z"), null, paths);
  assert.equal(first.decision.run, true);
  assert.equal(first.persistedTo, null, "bot.ts audits DAILY-MAINT-WARN on this");
  assert.deepEqual(simulate("2026-10-27T07:00:00Z", "2026-10-27T14:00:00Z", [], paths), { "2026-10-27": 1 });
});

// ── bot.ts wiring ─────────────────────────────────────────────────────────────
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");

function fnBody(signature: string): string {
  const start = bot.indexOf(signature);
  assert.ok(start >= 0, `${signature} not found in bot.ts`);
  const next = bot.indexOf("\n  async function ", start + signature.length);
  const nextComment = bot.indexOf("\n  // ──", start + signature.length);
  const end = Math.min(...[next, nextComment].filter((i) => i > 0));
  return bot.slice(start, end);
}

test("wiring: the dead 4am branch is gone from runOvernightResearch", () => {
  const research = fnBody("async function runOvernightResearch(etHour: number)");
  assert.ok(!research.includes("Math.floor(etHour) === 4"), "the unreachable hour-4 branch must not live here");
  assert.ok(!research.includes("state.morningQueueExecuted = false"), "the daily reset must not live here");
});

test("wiring: runDailyMaintenance holds every daily reset, the retrain and the backup", () => {
  const daily = fnBody("async function runDailyMaintenance(etDate: string)");
  for (const reset of [
    "(state as any).dailyBlockedTickers = [];", "state.consecutiveStopLosses = 0;",
    "state.morningQueueExecuted = false;", "orderAttemptCounts = {};", "sweptCspAt = {};",
  ]) assert.ok(daily.includes(reset), `missing daily reset: ${reset}`);
  assert.ok(daily.includes("ml_retrain_safe.py"), "the daily retrain runs here");
  assert.ok(daily.includes('audit("BACKUP"'), "the nightly backup runs here");
  // grep-level guarantee there is no other (dead) copy of the reset left behind.
  assert.equal(bot.split("state.morningQueueExecuted = false;").length - 1, 1, "exactly one morningQueueExecuted reset");
});

test("wiring: checkDailyMaintenance claims through the tested helper before running, and is driven unconditionally", () => {
  const check = fnBody("async function checkDailyMaintenance()");
  const claimAt = check.indexOf("claimDailyMaintenance(Date.now(), dailyMaintenanceLastRunEtDateMem)");
  const runAt = check.indexOf("await runDailyMaintenance(decision.etDate)");
  assert.ok(claimAt > 0 && runAt > claimAt, "the claim is persisted before the work starts");
  assert.ok(check.includes("if (dailyMaintenanceRunning) return;"), "no overlapping runs");
  assert.ok(check.includes('audit("DAILY-MAINT-WARN"'), "a failed claim write is audited, not silent");
  const poll = bot.indexOf("setInterval(() => { checkDailyMaintenance().catch(");
  assert.ok(poll > 0, "own interval");
  const line = bot.slice(bot.lastIndexOf("\n", poll), bot.indexOf("\n", poll + 1));
  assert.ok(line.includes("DAILY_MAINTENANCE_CHECK_MS"));
  assert.ok(!line.includes("is_open") && !line.includes("killSwitch") && !line.includes("state.active"),
    "no market-hours / kill-switch / active gate on the bookkeeping branch");
  assert.ok(bot.includes("setTimeout(() => { checkDailyMaintenance().catch("), "boot-time check for a process starting inside the window");
});
