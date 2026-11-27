// Run: npx tsx --test client/src/lib/dtccNotional.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import { dtccNotionalText, dtccRankingLine, dtccCapMarksLine, type DtccCapMarksSummary } from "./dtccNotional";

test("a cell: a number with its currency, a capped value with the source's +, a blank that is not a zero", () => {
  assert.equal(dtccNotionalText(220_000_000, "USD"), "USD 220,000,000");
  assert.equal(dtccNotionalText(220_000_000, "USD", false), "USD 220,000,000");
  assert.equal(dtccNotionalText(250_000_000, "USD", true), "USD 250,000,000+");
  assert.equal(dtccNotionalText(null, "USD"), "— (not reported)");
  assert.equal(dtccNotionalText(null, "USD", true), "— (not reported)");
});

test("no counts from the server: no line", () => {
  assert.equal(dtccRankingLine(undefined, 250_000_000), null);
  assert.equal(dtccRankingLine(null), null);
});

test("the file of 2026-11-26 as the vendor published it: every row is accounted for", () => {
  // 13,376 US-underlier rows: 10,936 USD at or under the cap, 17 capped,
  // 2,422 in other currencies ranked nowhere, 3 blank. (The 700,000,000 MXN
  // row is one of the 2,422, not a placeholder: the cap is a dollar figure.)
  const line = dtccRankingLine({ usd: 10_934, usd_capped: 17, other_currency: 2_422, over_cap_unmarked: 0, blank: 3 }, 250_000_000)!;
  assert.equal(line,
    "Ranked in US dollars: 10,951 USD events in this file · 17 at or above the source's cap, written 250,000,000+, listed first"
    + " · not in the list: 2,422 in other currencies (no exchange rate is applied) · 3 with no notional reported");
});

test("a placeholder day: the unmarked value above the cap is named and counted", () => {
  const line = dtccRankingLine({ usd: 5, usd_capped: 0, other_currency: 0, over_cap_unmarked: 1, blank: 0 }, 250_000_000)!;
  assert.match(line, /not in the list: 1 unmarked value above the cap \(a reporter's placeholder\)/);
  assert.doesNotMatch(line, /listed first|other currencies|no notional/);
});

test("nothing left out: the line says only what was ranked", () => {
  assert.equal(dtccRankingLine({ usd: 1, usd_capped: 0, other_currency: 0, over_cap_unmarked: 0, blank: 0 }),
    "Ranked in US dollars: 1 USD event in this file");
});

// ── the cap-mark record (open_questions.md (bc1), 2026-11-27) ───────────────
const marks: DtccCapMarksSummary = { enabled: true, days_read: 61, recorded: 61, owed: 0, lost: 0, lost_days: [], capped_events: 1_480 };

test("cap marks: no counts, or no file read: no line", () => {
  assert.equal(dtccCapMarksLine(undefined), null);
  assert.equal(dtccCapMarksLine(null), null);
  assert.equal(dtccCapMarksLine({ ...marks, days_read: 0, recorded: 0 }), null);
});

test("cap marks: 'all' is said only when nothing is owed and nothing was lost", () => {
  assert.equal(dtccCapMarksLine(marks),
    "Events archived before 2026-11-27 hold a blank where the source wrote its cap mark: the mark is on record for all 61 files read (1,480 capped events)");
  assert.doesNotMatch(dtccCapMarksLine({ ...marks, recorded: 60, owed: 1 })!, /for all/);
  assert.doesNotMatch(dtccCapMarksLine({ ...marks, recorded: 60, lost: 1, lost_days: ["2024-11-25"] })!, /for all/);
});

test("cap marks: the state on the day this shipped: files owed, two lost and named", () => {
  const line = dtccCapMarksLine({ enabled: true, days_read: 62, recorded: 5, owed: 55, lost: 2, lost_days: ["2024-11-24", "2024-11-25"], capped_events: 482 })!;
  assert.equal(line,
    "Events archived before 2026-11-27 hold a blank where the source wrote its cap mark: the mark is on record for 5 of 62 files read (482 capped events)"
    + " · 55 files still to read again (two after each poll, oldest first)"
    + " · 2 files left the vendor before the mark was read (2024-11-24, 2024-11-25): on those days a blank is either no notional or a capped one");
});

test("cap marks: one lost file, one capped event, the re-read switched off", () => {
  const line = dtccCapMarksLine({ enabled: false, days_read: 3, recorded: 1, owed: 1, lost: 1, lost_days: ["2024-11-25"], capped_events: 1 })!;
  assert.match(line, /1 of 3 files read \(1 capped event\)/);
  assert.match(line, /1 file not read again \(the re-read is switched off\)/);
  assert.match(line, /1 file left the vendor before the mark was read \(2024-11-25\): on that day a blank/);
});

test("cap marks: a long list of lost days is counted, not printed", () => {
  const lost_days = ["2024-11-20", "2024-11-21", "2024-11-22", "2024-11-23", "2024-11-24"];
  assert.match(dtccCapMarksLine({ ...marks, recorded: 56, lost: 5, lost_days })!, /5 files left the vendor before the mark was read: on those days/);
});
