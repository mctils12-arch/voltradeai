import { test } from "node:test";
import assert from "node:assert/strict";
import { firesCheckLines, sharePct, type FiresCheck, type FiresCheckDay } from "./archiveChecks";

const NOW = Date.parse("2026-11-21T08:40:00Z");
const day = (d: string, status: FiresCheckDay["status"], up: number, held: number, added: number): FiresCheckDay => ({ day: d, status, upstream_rows: up, held_before: held, added, attempts: 1 });

test("a share under one percent keeps two decimals; a missing share is a dash, never 0%", () => {
  assert.equal(sharePct(0.0037), "0.37%");
  assert.equal(sharePct(0.12), "12.0%");
  assert.equal(sharePct(null), "—");
});

test("before the first run, and switched off, are two different sentences; no payload draws nothing", () => {
  assert.deepEqual(firesCheckLines(null, NOW), []);
  assert.deepEqual(firesCheckLines({ enabled: false }, NOW), ["The daily check against NASA's record by date is off on this server."]);
  assert.deepEqual(firesCheckLines({ enabled: true, last_run_at: null }, NOW), ["The daily check against NASA's record by date has not run yet (it runs once a day from 06:00 UTC)."]);
});

test("the full reading: what the live recording missed, the refilled days as NASA's later record, the days that stay empty, the failures", () => {
  const s: FiresCheck = {
    enabled: true, last_run_at: "2026-11-21T06:30:00Z",
    live_poll_missed: { days: 1, upstream_rows: 75878, added: 281, share: 0.0037 },
    days: [
      day("2026-11-20", "complete", 75878, 75597, 281),
      day("2026-11-19", "refilled", 70100, 0, 70100),
      day("2026-11-03", "refilled", 64000, 0, 64000),
      day("2026-10-28", "no_rows", 0, 0, 0),
      day("2026-10-27", "no_rows", 0, 0, 0),
      day("2026-11-10", "failed", 0, 0, 0),
    ],
  };
  assert.deepEqual(firesCheckLines(s, NOW), [
    "Checked against NASA's record by date 2 h 10 min ago.",
    "Over 1 finished day the live recording had missed 281 of the 75,878 detections NASA holds for those dates (0.37%); they were added.",
    "2 days with no recording (2026-11-03 to 2026-11-19) were refilled afterwards from NASA's answer for the date: 134,100 detections. They are NASA's record as of the refill, not a live recording.",
    "2 days (2026-10-27 to 2026-10-28) stay empty: NASA returned nothing for the date from any satellite.",
    "1 day (2026-11-10) could not be read from NASA: the request failed.",
  ]);
});

test("nothing missed is said as a count, and no compared day is said plainly", () => {
  const held: FiresCheck = { enabled: true, last_run_at: "2026-11-21T08:39:40Z", live_poll_missed: { days: 3, upstream_rows: 210000, added: 0, share: 0 }, days: [] };
  assert.deepEqual(firesCheckLines(held, NOW), [
    "Checked against NASA's record by date under a minute ago.",
    "Over 3 finished days the live recording held every one of the 210,000 detections NASA holds for those dates.",
  ]);
  const none: FiresCheck = { enabled: true, last_run_at: "2026-11-21T08:00:00Z", live_poll_missed: { days: 0, upstream_rows: 0, added: 0, share: null }, days: [day("2026-11-03", "refilled", 5, 0, 5)] };
  assert.equal(firesCheckLines(none, NOW)[1], "No finished day has been compared yet.");
  assert.match(firesCheckLines(none, NOW)[2], /^1 day with no recording \(2026-11-03\) was refilled/);
});
