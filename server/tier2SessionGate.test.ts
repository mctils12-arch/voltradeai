// Tier-2 closed-day gate (2026-11-15). REGRESSION: on Sunday 2026-11-15 the
// audit log showed a full 12,148-stock scan every ~3.5 min, and 6,434 of the
// shadow log's 20,000 records had been written on days with no session —
// scheduleTier2's window was `clock.is_open || 04:00 <= ET hour < 20:00`,
// with no trading-day test.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  tier2ScanDecision, closedDayScansEnabled, closedDayAuditLine,
  OUTSIDE_WINDOW_WHY, CLOSED_DAY_SCANS_ENV, type Tier2GateInput,
} from "./tier2SessionGate";

const CAL = { holidays: new Set(["2026-11-26"]), halfDays: new Set(["2026-11-27"]) };
const base: Tier2GateInput = {
  clockIsOpen: false, etHour: 11, etDate: "2026-11-13", calendar: CAL,
  hasScanResult: true, closedDayScans: false,
};
const d = (o: Partial<Tier2GateInput>) => tier2ScanDecision({ ...base, ...o });

test("REGRESSION: a Sunday inside 04:00-20:00 ET with a scan already held does not scan", () => {
  const r = d({ etDate: "2026-11-15", etHour: 11.2 });
  assert.equal(r.scan, false);
  assert.equal(r.closedDay, true);
  assert.match(String(r.why), /no session on 2026-11-15/);
});

test("Saturday and a full-closure holiday are closed days too", () => {
  assert.equal(d({ etDate: "2026-11-14" }).scan, false);
  const h = d({ etDate: "2026-11-26" }); // Thanksgiving, a Thursday
  assert.equal(h.scan, false);
  assert.equal(h.closedDay, true);
});

test("a closed-day container with no scan result runs ONE scan (the picks pages need a result)", () => {
  const first = d({ etDate: "2026-11-15", hasScanResult: false });
  assert.deepEqual(first, { scan: true, closedDay: true, why: null });
  assert.equal(d({ etDate: "2026-11-15", hasScanResult: true }).scan, false);
});

test("weekday behaviour is unchanged: pre-market, regular hours and after-hours all scan", () => {
  for (const etHour of [4, 9.4, 11, 15.9, 16.5, 19.99]) {
    assert.deepEqual(d({ etDate: "2026-11-13", etHour }), { scan: true, closedDay: false, why: null }, `etHour ${etHour}`);
  }
});

test("a half day is a session day", () => {
  assert.equal(d({ etDate: "2026-11-27", etHour: 15 }).scan, true);
});

test("outside 04:00-20:00 ET with the market closed nothing scans, with the scheduler's original text", () => {
  for (const etHour of [0, 3.99, 20, 23.5]) {
    const r = d({ etDate: "2026-11-13", etHour });
    assert.deepEqual(r, { scan: false, closedDay: false, why: OUTSIDE_WINDOW_WHY });
  }
  // ...and a closed day outside the window reports the window, not the day
  assert.equal(d({ etDate: "2026-11-15", etHour: 22 }).why, OUTSIDE_WINDOW_WHY);
});

test("the broker clock reading OPEN always scans — the calendar table never stops a live session", () => {
  assert.deepEqual(d({ clockIsOpen: true, etDate: "2026-11-15", etHour: 11 }), { scan: true, closedDay: false, why: null });
  assert.equal(d({ clockIsOpen: true, etDate: "2026-11-26", etHour: 2 }).scan, true);
});

test("an unreadable date is treated as a session day (the rule only removes what it can place on a closed day)", () => {
  assert.equal(d({ etDate: "" }).scan, true);
  assert.equal(d({ etDate: "not-a-date" }).scan, true);
});

test("off switch: VOLTRADE_TIER2_CLOSED_DAY_SCANS=1 restores closed-day scans", () => {
  assert.equal(closedDayScansEnabled({}), false);
  assert.equal(closedDayScansEnabled({ [CLOSED_DAY_SCANS_ENV]: "0" }), false);
  assert.equal(closedDayScansEnabled({ [CLOSED_DAY_SCANS_ENV]: "1" }), true);
  assert.equal(closedDayScansEnabled({ [CLOSED_DAY_SCANS_ENV]: " TRUE " }), true);
  assert.deepEqual(d({ etDate: "2026-11-15", closedDayScans: true }), { scan: true, closedDay: true, why: null });
});

test("the audit line names the day, the scan shown and the off switch", () => {
  const line = closedDayAuditLine("2026-11-15", "2026-11-13T19:45:02Z");
  assert.match(line, /2026-11-15/);
  assert.match(line, /2026-11-13T19:45:02Z/);
  assert.match(line, new RegExp(CLOSED_DAY_SCANS_ENV));
  assert.match(closedDayAuditLine("2026-11-15", null), /one scan was run/);
});

test("RATCHET: scheduleTier2 decides through tier2ScanDecision, not an inline window test", () => {
  const here = path.dirname(fileURLToPath(import.meta.url));
  const src = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
  const start = src.indexOf("async function scheduleTier2()");
  assert.ok(start > 0, "scheduleTier2 not found");
  const body = src.slice(start, src.indexOf("// Start Tier 2 with adaptive scheduling", start));
  assert.match(body, /tier2ScanDecision\(/);
  assert.match(body, /closedDayScansEnabled\(\)/);
  assert.doesNotMatch(body, /clock\.is_open \|\| \(etH >= 4 && etH < 20\)/, "the weekday-blind window test is back");
  // a skipped closed-day tick must beat `skipped` (the heartbeat stays fed)
  assert.match(body, /beat\("tier2_scan", "finish", "skipped", t2Gate\.why/);
});
