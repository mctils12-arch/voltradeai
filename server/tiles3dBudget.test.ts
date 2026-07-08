// Google 3D Tiles root-request cost guard — same safety discipline as the RunPod
// budget: authorize before spending, refuse past the cap so it can never run away.
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  utcDay, utcMonth, countForDay, countForMonth, authorizeRoot, makeRootRow,
  budgetStatus, DAILY_ROOT_CAP, MONTHLY_ROOT_CAP,
} from "./tiles3dBudget.ts";

const T = (s: string) => Date.parse(s);

test("utcDay/utcMonth derive UTC calendar buckets", () => {
  assert.equal(utcDay(T("2026-07-08T23:59:00Z")), "2026-07-08");
  assert.equal(utcMonth(T("2026-07-08T23:59:00Z")), "2026-07");
});

test("counts filter by day and month", () => {
  const rows = [
    makeRootRow(T("2026-07-08T01:00:00Z")),
    makeRootRow(T("2026-07-08T02:00:00Z")),
    makeRootRow(T("2026-07-07T02:00:00Z")),
    makeRootRow(T("2026-06-30T02:00:00Z")),
  ];
  assert.equal(countForDay(rows, "2026-07-08"), 2);
  assert.equal(countForDay(rows, "2026-07-07"), 1);
  assert.equal(countForMonth(rows, "2026-07"), 3);
  assert.equal(countForMonth(rows, "2026-06"), 1);
});

test("authorizes under the daily cap", () => {
  const now = T("2026-07-08T12:00:00Z");
  const rows = [makeRootRow(T("2026-07-08T01:00:00Z"))];
  const d = authorizeRoot(rows, now);
  assert.equal(d.authorized, true);
  assert.equal(d.today, 1);
  assert.equal(d.daily_remaining, DAILY_ROOT_CAP - 1);
});

test("REFUSES at the daily cap (remount/reload runaway guard)", () => {
  const now = T("2026-07-08T12:00:00Z");
  const rows = Array.from({ length: DAILY_ROOT_CAP }, () => makeRootRow(T("2026-07-08T03:00:00Z")));
  const d = authorizeRoot(rows, now);
  assert.equal(d.authorized, false);
  assert.equal(d.reason, "daily_cap");
  assert.equal(d.daily_remaining, 0);
});

test("REFUSES at the monthly free-tier ceiling (never silently roll to paid)", () => {
  const now = T("2026-07-20T12:00:00Z");
  // spread across days so no single day hits the daily cap, but the month does
  const rows: any[] = [];
  for (let i = 0; i < MONTHLY_ROOT_CAP; i++) {
    const day = String((i % 28) + 1).padStart(2, "0");
    rows.push(makeRootRow(T(`2026-07-${day}T0${i % 6}:00:00Z`)));
  }
  const d = authorizeRoot(rows, now, { dailyCap: 100000 }); // disable daily so monthly is the binding cap
  assert.equal(d.authorized, false);
  assert.equal(d.reason, "monthly_cap");
});

test("new UTC day resets the daily count", () => {
  const rows = Array.from({ length: DAILY_ROOT_CAP }, () => makeRootRow(T("2026-07-08T03:00:00Z")));
  // next day: yesterday's cap-full does not block today
  const d = authorizeRoot(rows, T("2026-07-09T00:05:00Z"));
  assert.equal(d.authorized, true);
  assert.equal(d.today, 0);
});

test("budgetStatus mirrors the decision", () => {
  const s = budgetStatus([makeRootRow(T("2026-07-08T01:00:00Z"))], T("2026-07-08T12:00:00Z"));
  assert.equal(s.daily_cap, DAILY_ROOT_CAP);
  assert.equal(s.today, 1);
  assert.equal(s.would_authorize, true);
});
