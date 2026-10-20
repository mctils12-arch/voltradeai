import { test } from "node:test";
import assert from "node:assert/strict";
import { fmtSigned, fmtPct, fmtMoney, fmtCompact, fmtAge, fmtCountdown, edgeStats, fmtOcc, spyTodayPct, MINUS } from "./format";

test("signed numbers print a true minus and never rely on colour", () => {
  assert.equal(fmtPct(-1.2), `${MINUS}1.20%`);
  assert.equal(fmtPct(0.5), "+0.50%");
  assert.equal(fmtPct(0), "0.00%");
  assert.equal(fmtSigned(null), "—");
  assert.equal(fmtMoney(-1247.3), `${MINUS}$1,247`);
  assert.equal(fmtMoney(100172.07, true), "$100,172.07");
  assert.equal(fmtCompact(82570), "$82.6k");
  assert.equal(fmtCompact(-1900), `${MINUS}$1,900`);
});

test("ages and countdowns are short", () => {
  assert.equal(fmtAge(3200), "3s");
  assert.equal(fmtAge(4 * 60_000), "4m");
  assert.equal(fmtAge(9.7 * 3600_000), "10h");
  assert.equal(fmtAge(3 * 86400_000), "3d");
  assert.equal(fmtAge(null), "—");
  assert.equal(fmtCountdown(2 * 3600_000 + 2 * 60_000), "2h02m");
  assert.equal(fmtCountdown(65_000), "1m05s");
});

test("edge stats reproduce the audit's live numbers", () => {
  // live 2026-10-15: WR 27.8 %, avg win +9.08, avg loss −5.16
  const e = edgeStats(27.8, 9.08, -5.16)!;
  assert.equal(e.payoff!.toFixed(2), "1.76");
  assert.equal((e.breakevenWr * 100).toFixed(1), "36.2");
  assert.equal(e.expectancy.toFixed(2), "-1.20");
  assert.equal(edgeStats(null, 1, 1), null);
  assert.equal(edgeStats(0.5, 0, 0), null);
});

test("OCC symbols read as root, strike, right and expiry", () => {
  const yy = String(new Date().getUTCFullYear()).slice(2);
  assert.equal(fmtOcc(`CRWV${yy}1120P00055000`), "CRWV 55P 20-Nov");
  assert.equal(fmtOcc(`TEM${yy}1120P00070000`), "TEM 70P 20-Nov");
  assert.equal(fmtOcc("QQQ"), "QQQ");
  assert.equal(fmtOcc(null), "—");
});

test("SPY today is shown only for the ET session day of its bar", () => {
  const bar = { date: "2026-10-20", pct: 0.25 };
  assert.equal(spyTodayPct(bar, Date.parse("2026-10-20T20:30:00Z")), 0.25, "16:30 ET same day");
  assert.equal(spyTodayPct(bar, Date.parse("2026-10-21T02:00:00Z")), 0.25, "22:00 ET is still the 20th in New York");
  assert.equal(spyTodayPct(bar, Date.parse("2026-10-21T14:00:00Z")), null, "next day: yesterday's move is not today's");
  assert.equal(spyTodayPct(null, Date.parse("2026-10-20T20:30:00Z")), null);
});
