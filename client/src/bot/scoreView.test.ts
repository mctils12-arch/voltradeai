import { test } from "node:test";
import assert from "node:assert/strict";
import { visibleRows, fmtDay, beatLine, cashFlowNote, SCORE_VISIBLE_DAYS } from "./scoreView";
import type { ScoreRow } from "./useBotData";

const row = (k: number): ScoreRow => ({ date: `2026-10-${String(k + 1).padStart(2, "0")}`, prevDate: "", sessions: 1, botPct: 0, spyPct: 0, diffPp: 0, botCumPct: 0, spyCumPct: 0, cumDiffPp: 0, provisional: false });

test("about a month of rows until 'show all'", () => {
  const rows = Array.from({ length: 45 }, (_, k) => row(k % 28));
  assert.equal(visibleRows(rows, false).length, SCORE_VISIBLE_DAYS);
  assert.equal(visibleRows(rows, true).length, 45);
  assert.equal(visibleRows(rows.slice(0, 3), false).length, 3);
});

test("day labels and the beat line", () => {
  assert.equal(fmtDay("2026-10-27"), "Oct 27");
  assert.equal(fmtDay("2026-05-06", 2026), "May 6");
  assert.equal(fmtDay("2025-12-31", 2026), "Dec 31 2025");
  assert.equal(fmtDay("bad"), "—");
  assert.equal(beatLine({ from: "", to: "", botCumPct: 0, spyCumPct: 0, cumDiffPp: 0, daysBeat: 48, daysCompared: 120, gapRows: 0, botMaxDdPct: 0, spyMaxDdPct: 0 }), "48 of 120 days · 40%");
  assert.equal(beatLine({ from: "", to: "", botCumPct: 0, spyCumPct: 0, cumDiffPp: 0, daysBeat: 1, daysCompared: 1, gapRows: 0, botMaxDdPct: 0, spyMaxDdPct: 0 }), "1 of 1 day · 100%");
});

test("cash-flow caveat: unknown until the ledger loads, clean at zero, loud otherwise — never silent", () => {
  assert.equal(cashFlowNote(null).tone, "unknown");
  const clean = cashFlowNote({ journals: 0, transfers: 0, since: "2026-05-07T00:00:00.000Z" });
  assert.equal(clean.tone, "ok");
  assert.match(clean.text, /since May 7/);
  const moved = cashFlowNote({ journals: 5000, transfers: 0, since: null });
  assert.equal(moved.tone, "warn");
  assert.match(moved.text, /\$5,000.*NOT adjusted/);
});
