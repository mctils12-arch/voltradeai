// LIVENESS ALARM runtime half (Amendment 2) — the regression test that
// would have caught the production incident: the bot sat PAUSED and
// /api/health stayed "ok". Pure-module tests + a wiring pin on bot.ts.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  nextLiveness, marketHoursBetween, loopDark, shouldSendLivenessReminder, parseLivenessFile,
  LOOP_DARK_MARKET_HOURS, LOOP_DARK_WALLCLOCK_HOURS, LIVENESS_REMINDER_INTERVAL_HOURS,
  LIVENESS_SUSTAINED_ACTIVE_MINUTES, LIVENESS_RUN_GAP_MINUTES, type LivenessFile,
} from "./liveness";

const here = path.dirname(fileURLToPath(import.meta.url));

// Fixed timestamps (July 2026, EDT = UTC-4). Wed 2026-07-08.
const WED_10ET = Date.UTC(2026, 6, 8, 14, 0);   // Wed 10:00 ET — market open
const WED_1130ET = Date.UTC(2026, 6, 8, 15, 30);
const WED_13ET = Date.UTC(2026, 6, 8, 17, 0);
const FRI_1630ET = Date.UTC(2026, 6, 10, 20, 30); // Fri after close
const MON_09ET = Date.UTC(2026, 6, 13, 13, 0);    // Mon 9:00 ET — pre-open

test("heartbeat: active stamps now; fresh boot seeds now (no false alarm); inactive keeps the last stamp across restarts", () => {
  const now = WED_10ET;
  assert.equal(nextLiveness(null, true, now).lastActiveAt, now);
  assert.equal(nextLiveness(null, false, now).lastActiveAt, now, "fresh install seeds, never alarms instantly");
  const prev = { lastActiveAt: WED_10ET };
  assert.equal(nextLiveness(prev, false, WED_13ET), prev, "restart must NOT reset the darkness clock");
});

test("market hours: intraday counts, weekend does not", () => {
  const intraday = marketHoursBetween(WED_10ET, WED_13ET);
  assert.ok(Math.abs(intraday - 3) < 0.2, `Wed 10:00→13:00 ET should be ~3 market hours (got ${intraday})`);
  const weekend = marketHoursBetween(FRI_1630ET, MON_09ET);
  assert.ok(weekend < 0.1, `Fri 16:30 → Mon 9:00 ET spans no session (got ${weekend})`);
});

test("alarm: >2 market hours dark degrades; 1.5h does not; active never does", () => {
  const pausedAt = { lastActiveAt: WED_10ET };
  const dark = loopDark(pausedAt, false, WED_13ET); // 3 market hours
  assert.equal(dark.dark, true);
  assert.ok(dark.detail.includes("LIVENESS ALARM"), "detail must name the alarm");
  const ok = loopDark(pausedAt, false, WED_1130ET); // 1.5 market hours
  assert.equal(ok.dark, false);
  assert.equal(loopDark(pausedAt, true, WED_13ET).dark, false, "active loop is never dark");
});

test("alarm: weekend-long halt trips the 24h wall-clock ceiling before Monday", () => {
  const pausedFri = { lastActiveAt: FRI_1630ET };
  const monday = loopDark(pausedFri, false, MON_09ET);
  assert.ok(monday.wallHours > LOOP_DARK_WALLCLOCK_HOURS);
  assert.equal(monday.dark, true, "weekend halts must surface before the Monday open");
  // but a Saturday check ~14h in stays quiet — no market hours, under 24h wall
  const satNight = loopDark(pausedFri, false, FRI_1630ET + 14 * 3_600_000);
  assert.equal(satNight.dark, false);
});

test("wiring pinned: /api/health consumes the assessment and degrades on dark", () => {
  const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
  assert.ok(bot.includes("nextLiveness"), "bot.ts must maintain the liveness heartbeat");
  assert.ok(bot.includes("loopDark"), "bot.ts health check must assess loop darkness");
  assert.ok(/lv\.dark.*\n?.*degraded|degraded.*lv\.dark/.test(bot) || bot.includes('if (lv.dark) checks.status = "degraded"'),
    "a dark loop must degrade overall /api/health status");
  assert.equal(LOOP_DARK_MARKET_HOURS, 2, "approved threshold: 2 market hours");
  assert.equal(LOOP_DARK_WALLCLOCK_HOURS, 24, "approved ceiling: 24h wall-clock");
});

// KNOWN BROKEN #42/#43 (2026-09-23): the one-time drawdown-kill trip email
// never re-fires while the loop stays dark, and a real incident sat 13
// days before a session noticed nobody had re-notified. These pin the
// re-escalation gate that closes that hole.
test("reminder: never fires while active or not dark", () => {
  assert.equal(shouldSendLivenessReminder(false, null, WED_13ET), false);
  assert.equal(shouldSendLivenessReminder(false, WED_10ET, WED_13ET), false, "a stale reminder stamp must not force a fire once resolved");
});

test("reminder: fires immediately the first time a dark loop has no reminder on record", () => {
  assert.equal(shouldSendLivenessReminder(true, null, WED_13ET), true);
  assert.equal(shouldSendLivenessReminder(true, undefined, WED_13ET), true);
  assert.equal(shouldSendLivenessReminder(true, NaN, WED_13ET), true, "a corrupt persisted stamp must not permanently suppress reminders");
});

test("reminder: gated to once per LIVENESS_REMINDER_INTERVAL_HOURS, not on every tick", () => {
  const firstReminderAt = WED_13ET;
  const justUnder = firstReminderAt + (LIVENESS_REMINDER_INTERVAL_HOURS - 0.01) * 3_600_000;
  const atInterval = firstReminderAt + LIVENESS_REMINDER_INTERVAL_HOURS * 3_600_000;
  const wellPast = firstReminderAt + (LIVENESS_REMINDER_INTERVAL_HOURS + 5) * 3_600_000;
  assert.equal(shouldSendLivenessReminder(true, firstReminderAt, justUnder), false, "must not re-fire before the interval elapses");
  assert.equal(shouldSendLivenessReminder(true, firstReminderAt, atInterval), true, "must fire exactly at the interval boundary");
  assert.equal(shouldSendLivenessReminder(true, firstReminderAt, wellPast), true, "must fire well past the interval too — no upper cap");
});

test("wiring pinned: bot.ts sends a periodic dark-loop reminder, gated by shouldSendLivenessReminder", () => {
  const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
  assert.ok(bot.includes("shouldSendLivenessReminder"), "bot.ts must consult the re-escalation gate");
  assert.ok(bot.includes("lastReminderAt"), "bot.ts must persist the reminder timestamp so it survives restarts");
  assert.ok(/shouldSendLivenessReminder\([\s\S]{0,600}sendEmailAlert/.test(bot),
    "the gate's true branch must actually send an email, not just compute a flag");
});

// ── SUSTAINED ACTIVE (2026-11-12): a toggle that re-latches at once leaves the dark clock alone ──
const MIN = 60_000;
/** Run the minute tick over a list of [minuteOffset, active] steps from `start`. */
function runTicks(start: LivenessFile | null, t0: number, steps: Array<[number, boolean]>): LivenessFile {
  let lv = start;
  for (const [minute, active] of steps) lv = nextLiveness(lv, active, t0 + minute * MIN);
  return lv as LivenessFile;
}

test("sustained active: the 2026-11-05 case — two toggles that re-latch within a tick do not move the dark clock", () => {
  const darkSince = WED_10ET;                       // the loop went dark here
  const t0 = WED_10ET + 90 * MIN;                   // 1.5 market hours later the owner toggles
  const afterFirst = runTicks({ lastActiveAt: darkSince }, t0, [[0, true], [1, false]]);
  assert.equal(afterFirst.lastActiveAt, darkSince, "one active tick stamps nothing");
  assert.equal(afterFirst.activeSince, undefined, "the short run ended");
  const afterSecond = runTicks(afterFirst, t0, [[6, true], [7, false]]);
  assert.equal(afterSecond.lastActiveAt, darkSince);
  // 2h12m after darkSince the alarm fires as if the toggles never happened
  const lv = loopDark(afterSecond, false, darkSince + 132 * MIN);
  assert.equal(lv.dark, true);
  assert.match(lv.detail, new RegExp(new Date(darkSince).toISOString().replace(/[.]/g, "\\.")));
});

test("sustained active: lastActiveAt is stamped once a run is LIVENESS_SUSTAINED_ACTIVE_MINUTES long, then every tick", () => {
  assert.equal(LIVENESS_SUSTAINED_ACTIVE_MINUTES, 10);
  const prev: LivenessFile = { lastActiveAt: WED_10ET, lastReminderAt: WED_10ET + 5 * MIN };
  const t0 = WED_1130ET;
  const steps: Array<[number, boolean]> = Array.from({ length: 10 }, (_, i) => [i, true]);
  const nine = runTicks(prev, t0, steps);            // minutes 0..9: nine minutes of run
  assert.equal(nine.lastActiveAt, WED_10ET, "not yet sustained");
  assert.equal(nine.lastReminderAt, WED_10ET + 5 * MIN, "the reminder record is kept until the run counts");
  assert.equal(nine.activeSince, t0);
  const ten = nextLiveness(nine, true, t0 + 10 * MIN);
  assert.equal(ten.lastActiveAt, t0 + 10 * MIN, "sustained: stamped");
  assert.equal(ten.lastReminderAt, undefined, "an active loop carries no reminder record (as before)");
  const eleven = nextLiveness(ten, true, t0 + 11 * MIN);
  assert.equal(eleven.lastActiveAt, t0 + 11 * MIN);
  assert.equal(eleven.activeSince, t0);
});

test("sustained active: a run survives a short restart gap; a long gap starts a new run", () => {
  assert.equal(LIVENESS_RUN_GAP_MINUTES, 5);
  const t0 = WED_10ET;
  const running: LivenessFile = { lastActiveAt: t0 + 60 * MIN, activeSince: t0, activeSeenAt: t0 + 60 * MIN };
  const afterDeploy = nextLiveness(running, true, t0 + 64 * MIN);      // 4-minute gap: a deploy
  assert.equal(afterDeploy.activeSince, t0);
  assert.equal(afterDeploy.lastActiveAt, t0 + 64 * MIN, "still sustained: stamped at once");
  const afterOutage = nextLiveness(running, true, t0 + 90 * MIN);      // 30-minute gap: the process was down
  assert.equal(afterOutage.activeSince, t0 + 90 * MIN, "a new run");
  assert.equal(afterOutage.lastActiveAt, t0 + 60 * MIN, "not stamped until the new run is sustained");
  const clockBack = nextLiveness(running, true, t0 + 30 * MIN);        // the clock went backwards
  assert.equal(clockBack.activeSince, t0 + 30 * MIN);
});

test("sustained active: an older file (no run stamps) and an inactive loop behave as before", () => {
  const old: LivenessFile = { lastActiveAt: WED_10ET };
  assert.equal(nextLiveness(old, false, WED_13ET), old, "same object: no disk churn while dark");
  const first = nextLiveness(old, true, WED_13ET);
  assert.deepEqual(first, { lastActiveAt: WED_10ET, activeSince: WED_13ET, activeSeenAt: WED_13ET });
  const ended = nextLiveness(first, false, WED_13ET + MIN);
  assert.deepEqual(ended, { lastActiveAt: WED_10ET }, "the run's stamps are dropped once, then the object is stable");
  assert.equal(nextLiveness(ended, false, WED_13ET + 2 * MIN), ended);
  assert.deepEqual(nextLiveness(null, true, WED_10ET), { lastActiveAt: WED_10ET, activeSince: WED_10ET, activeSeenAt: WED_10ET });
});

test("parseLivenessFile keeps the finite stamps and rejects a body without lastActiveAt", () => {
  assert.deepEqual(parseLivenessFile({ lastActiveAt: 5, savedAt: "x", lastReminderAt: 6, activeSince: 1, activeSeenAt: 4 }),
    { lastActiveAt: 5, lastReminderAt: 6, activeSince: 1, activeSeenAt: 4 });
  assert.deepEqual(parseLivenessFile({ lastActiveAt: 5, lastReminderAt: "6", activeSince: null }), { lastActiveAt: 5 });
  for (const bad of [null, undefined, "x", 5, {}, { lastActiveAt: "5" }, { lastActiveAt: NaN }]) assert.equal(parseLivenessFile(bad), null);
});

test("wiring pinned: bot.ts loads the liveness file through parseLivenessFile (the run survives a restart)", () => {
  const src = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
  const load = src.slice(src.indexOf("function loadLiveness()"), src.indexOf("function saveLiveness("));
  assert.match(load, /const parsed = parseLivenessFile\(JSON\.parse\(fs\.readFileSync\(p, "utf8"\)\)\);\s*if \(parsed\) return parsed;/);
  assert.match(src, /fs\.writeFileSync\(p, JSON\.stringify\(\{ \.\.\.v, savedAt: new Date\(\)\.toISOString\(\) \}\)\);/, "saved whole, so the run stamps persist");
});
