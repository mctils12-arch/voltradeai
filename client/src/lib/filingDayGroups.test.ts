// Filed-day groups on the Form 4 page ([PRODUCT] 2026-11-18, item (l)).
// Run: npx tsx --test client/src/lib/filingDayGroups.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "node:url";
import { DAY_ROWS_FIRST, DAY_ROWS_STEP, dayRowsWords, defaultOpenDay, groupByFiledDay, isDayOpen, rowsShown, showMoreLabel } from "./filingDayGroups";
import type { FiledDayStatus } from "./filingCoverage";

type Row = { filedAt: string | null; id: number };
const dayOf = (r: Row): string => (typeof r.filedAt === "string" ? r.filedAt.slice(0, 10) : "");
const rowsOn = (day: string, n: number, from = 0): Row[] => Array.from({ length: n }, (_, i) => ({ filedAt: `${day}T21:00:00.000Z`, id: from + i }));

// production, read 2026-11-18 18:10Z (/api/data/insider/history?filed_from=2026-10-20): the filed days the
// page loaded and the transaction rows their filings held
const LIVE_DAYS: FiledDayStatus[] = [
  { day: "2026-11-18", filings: 94, edgarIndex: null, complete: null },
  { day: "2026-11-17", filings: 1047, edgarIndex: 1047, complete: true },
  { day: "2026-11-16", filings: 624, edgarIndex: 624, complete: true },
];
const LIVE_ROWS: Row[] = [...rowsOn("2026-11-18", 264), ...rowsOn("2026-11-17", 2800, 1000), ...rowsOn("2026-11-16", 1333, 5000)];

test("REGRESSION 2026-11-18: the live page's three filed days are all on the page (the 500-row cap ended inside the second)", () => {
  // what the page did: the first 500 rows of everything loaded
  const capped = LIVE_ROWS.slice(0, 500);
  assert.deepEqual([...new Set(capped.map(dayOf))], ["2026-11-18", "2026-11-17"], "the old cap never reached 2026-11-16");
  assert.equal(capped.filter((r) => dayOf(r) === "2026-11-17").length, 236, "and showed 236 of the second day's 2,800 rows");
  // now: one group per loaded filed day, holding every row of it
  const groups = groupByFiledDay(LIVE_ROWS, dayOf, LIVE_DAYS);
  assert.deepEqual(groups.map((g) => [g.day, g.rows.length]), [["2026-11-18", 264], ["2026-11-17", 2800], ["2026-11-16", 1333]]);
  assert.equal(groups.reduce((n, g) => n + g.rows.length, 0), LIVE_ROWS.length, "no row is dropped");
  assert.equal(groups[1].status?.edgarIndex, 1047);
  // before anything is opened: the newest day's first page only
  const open = defaultOpenDay(groups);
  assert.equal(open, "2026-11-18");
  const onScreen = groups.reduce((n, g) => n + (isDayOpen(new Map(), g.day, open) ? rowsShown(g.rows.length, 0) : 0), 0);
  assert.equal(onScreen, DAY_ROWS_FIRST);
});

test("a 1,000-filing day costs a phone at most the first page of rows before anything is opened", () => {
  // item (l)'s check: under 20 screens at 390 px. A card measured 218 px on 2026-11-18 (109,000 px for 500).
  const groups = groupByFiledDay(rowsOn("2026-11-17", 2800), dayOf, [LIVE_DAYS[1]]);
  const px = rowsShown(groups[0].rows.length, 0) * 218;
  assert.ok(px < 20 * 844, `${px} px is 20 phone screens or more`);
});

test("a listed day with no row left keeps its line; a day the server did not list still gets one", () => {
  const groups = groupByFiledDay(rowsOn("2026-11-16", 2), dayOf, LIVE_DAYS);
  assert.deepEqual(groups.map((g) => [g.day, g.rows.length]), [["2026-11-18", 0], ["2026-11-17", 0], ["2026-11-16", 2]]);
  assert.equal(defaultOpenDay(groups), "2026-11-16", "the default is the newest day that holds a row");
  const unlisted = groupByFiledDay([...rowsOn("2026-11-13", 1), { filedAt: null, id: 9 }], dayOf, LIVE_DAYS.slice(0, 1));
  assert.deepEqual(unlisted.map((g) => [g.day, g.rows.length, g.status === null]), [["2026-11-18", 0, false], ["2026-11-13", 1, true], ["", 1, true]]);
  assert.equal(defaultOpenDay(groupByFiledDay([], dayOf, LIVE_DAYS)), null);
});

test("groups are newest first whatever order the rows and the day list arrive in, and keep row order inside a day", () => {
  const rows = [...rowsOn("2026-11-13", 2, 10), ...rowsOn("2026-11-17", 2, 20), ...rowsOn("2026-11-13", 1, 30)];
  const groups = groupByFiledDay(rows, dayOf, [LIVE_DAYS[2], LIVE_DAYS[0]]);
  assert.deepEqual(groups.map((g) => g.day), ["2026-11-18", "2026-11-17", "2026-11-16", "2026-11-13"]);
  assert.deepEqual(groups[3].rows.map((r) => r.id), [10, 11, 30]);
});

test("a day is open by the visitor's choice when there is one, else only the default day is", () => {
  const none = new Map<string, boolean>();
  assert.equal(isDayOpen(none, "2026-11-18", "2026-11-18"), true);
  assert.equal(isDayOpen(none, "2026-11-17", "2026-11-18"), false);
  assert.equal(isDayOpen(none, "2026-11-17", null), false);
  const chosen = new Map([["2026-11-18", false], ["2026-11-16", true]]);
  assert.equal(isDayOpen(chosen, "2026-11-18", "2026-11-18"), false, "closing the default day holds");
  assert.equal(isDayOpen(chosen, "2026-11-16", "2026-11-18"), true);
});

test("an open day shows its rows a page at a time and the control says how many are left", () => {
  assert.equal(rowsShown(2800, 0), 25);
  assert.equal(rowsShown(2800, 1), 225);
  assert.equal(rowsShown(2800, 14), 2800, "fourteen uses reach the end of the largest day read live");
  assert.equal(rowsShown(7, 0), 7);
  assert.equal(rowsShown(0, 3), 0);
  assert.equal(rowsShown(100, -2), 25);
  assert.equal(showMoreLabel(25, 2800), "Show 200 more · 25 of 2,800 rows shown");
  assert.equal(showMoreLabel(25, 37), "Show the last 12 · 25 of 37 rows shown");
  assert.equal(showMoreLabel(25, 25 + DAY_ROWS_STEP), "Show the last 200 · 25 of 225 rows shown");
  assert.equal(showMoreLabel(2800, 2800), null);
  assert.equal(showMoreLabel(7, 7), null);
});

test("a day line states its rows, and says so when a filter matched none", () => {
  assert.equal(dayRowsWords(2800, false), "2,800 rows");
  assert.equal(dayRowsWords(1, false), "1 row");
  assert.equal(dayRowsWords(0, false), "no transaction rows");
  assert.equal(dayRowsWords(312, true), "312 matching rows");
  assert.equal(dayRowsWords(1, true), "1 matching row");
  assert.equal(dayRowsWords(0, true), "no rows match the filter");
});

test("RATCHET: the Form 4 page groups every loaded row by filed day and caps only an archive-day page", () => {
  const here = path.dirname(fileURLToPath(import.meta.url));
  const src = fs.readFileSync(path.join(here, "..", "pages", "filings.tsx"), "utf8");
  assert.match(src, /groupByFiledDay\(visible, filedDayOf, filedDays\)/, "day groups must be built from every loaded row, not a capped slice");
  assert.match(src, /!byFiledDay && visible\.length > MAX_ROWS/, "the 'showing first N' cap is for archive-day pages only");
  assert.match(src, /aria-expanded=\{open\}/, "a day line must say whether it is open");
  assert.match(src, /showMoreLabel\(/, "an open day must offer the rest of its rows");
  assert.doesNotMatch(src, /narrow with a filter to see the rest[^]*narrow with a filter to see the rest/, "one cap message only");
});
