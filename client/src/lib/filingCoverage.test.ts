// What a filing-history page holds, in words ([PRODUCT] 2026-11-15).
// Run: npx tsx --test client/src/lib/filingCoverage.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "node:url";
import { readCoverage, extendCoverage, appendOlderFilings, coverageWords, loadOlderLabel, spanDays,
  readFiledDays, appendFiledDays, filedDayState, filedDayWords, filedDaysSummary, filedDayFirstQuery } from "./filingCoverage";

// production, read 2026-11-15 ~18:20Z (rows dropped): the two pages this module exists for
const FORM4_LIVE = { kind: "raw", days: 30, window: { from: "2026-10-17", to: "2026-11-15" }, covered: { from: "2026-11-10", to: "2026-11-15" },
  truncated: true, next: "?from=2026-10-17&to=2026-11-09", max_filings: 2000, days_scanned: 6, archive_start: "2026-07-04", count: 1864 };
const F13_LIVE = { kind: "raw", days: 45, window: { from: "2026-10-02", to: "2026-11-15" }, covered: { from: "2026-11-14", to: "2026-11-15" },
  truncated: true, next: "?from=2026-10-02&to=2026-11-13", max_filings: 500, days_scanned: 2, archive_start: "2026-07-05", count: 160 };

test("spanDays counts inclusive UTC days and refuses anything that is not two ordered days", () => {
  assert.equal(spanDays("2026-10-17", "2026-11-15"), 30);
  assert.equal(spanDays("2026-11-15", "2026-11-15"), 1);
  assert.equal(spanDays("2026-11-15", "2026-11-14"), 0);
  assert.equal(spanDays("nope", "2026-11-14"), 0);
});

test("REGRESSION: the live Form 4 page is six of thirty days, and the words say so (it was labelled 'last 30 days')", () => {
  const c = readCoverage(FORM4_LIVE)!;
  assert.equal(c.windowDays, 30);
  assert.equal(c.coveredDays, 6);
  assert.equal(c.truncated, true);
  assert.equal(coverageWords(c), "2026-11-10 → 2026-11-15 · 6 of the 30 days asked for — older days not loaded");
  assert.equal(loadOlderLabel(c, false), "Load older days (24 days before 2026-11-10 not loaded)");
  assert.equal(loadOlderLabel(c, true), "Loading older days…");
});

test("REGRESSION: the live 13F page is two of forty-five days", () => {
  const c = readCoverage(F13_LIVE)!;
  assert.equal(coverageWords(c), "2026-11-14 → 2026-11-15 · 2 of the 45 days asked for — older days not loaded");
});

test("a complete page says 'all N days' and offers nothing older", () => {
  const c = readCoverage({ window: { from: "2026-10-17", to: "2026-11-15" }, covered: { from: "2026-10-17", to: "2026-11-15" }, truncated: false, next: null })!;
  assert.equal(coverageWords(c), "2026-10-17 → 2026-11-15 · all 30 days");
  assert.equal(loadOlderLabel(c, false), null);
  const one = readCoverage({ window: { from: "2026-11-15", to: "2026-11-15" }, covered: { from: "2026-11-15", to: "2026-11-15" }, truncated: false })!;
  assert.equal(coverageWords(one), "2026-11-15 · 1 day");
});

test("a payload that states no coverage yields none: the page claims no window instead of guessing", () => {
  assert.equal(readCoverage(null), null);
  assert.equal(readCoverage({ days: 30, count: 2 }), null); // a server older than the contract
  assert.equal(readCoverage({ window: { from: "2026-11-15", to: "2026-11-01" }, covered: { from: "2026-11-15", to: "2026-11-01" } }), null);
  assert.equal(readCoverage({ window: { from: "x", to: "y" }, covered: { from: "x", to: "y" } }), null);
  assert.equal(coverageWords(null), "");
  assert.equal(loadOlderLabel(null, false), null);
});

test("truncated with no way to continue still reads as truncated, and renders no button", () => {
  const c = readCoverage({ ...FORM4_LIVE, next: null })!;
  assert.equal(c.truncated, true);
  assert.match(coverageWords(c), /6 of the 30 days asked for/);
  assert.equal(loadOlderLabel(c, false), null);
  // a `next` that is not a query string is not followed
  assert.equal(readCoverage({ ...FORM4_LIVE, next: "https://example.com/x" })!.next, null);
});

test("extendCoverage: an older page that continues the run moves covered.from back and keeps walking", () => {
  const c = readCoverage(FORM4_LIVE)!;
  const older = { window: { from: "2026-10-17", to: "2026-11-09" }, covered: { from: "2026-11-03", to: "2026-11-09" }, truncated: true, next: "?from=2026-10-17&to=2026-11-02" };
  const e = extendCoverage(c, older);
  assert.deepEqual(e.covered, { from: "2026-11-03", to: "2026-11-15" });
  assert.equal(e.coveredDays, 13);
  assert.equal(e.windowDays, 30);
  assert.equal(e.truncated, true);
  assert.equal(e.next, "?from=2026-10-17&to=2026-11-02");
  // the last page reaches the window's first day: complete, nothing older
  const last = extendCoverage(e, { window: { from: "2026-10-17", to: "2026-11-02" }, covered: { from: "2026-10-17", to: "2026-11-02" }, truncated: false, next: null });
  assert.equal(last.truncated, false);
  assert.equal(last.next, null);
  assert.equal(coverageWords(last), "2026-10-17 → 2026-11-15 · all 30 days");
});

test("extendCoverage: a page that does not continue the run is never presented as loaded — the walk stops", () => {
  const c = readCoverage(FORM4_LIVE)!;
  // a gap (ends two days before the loaded run), a page with no coverage, a page reaching before the window
  for (const older of [
    { window: { from: "2026-10-17", to: "2026-11-08" }, covered: { from: "2026-11-01", to: "2026-11-08" }, truncated: true, next: "?from=2026-10-17&to=2026-10-31" },
    { days: 30, count: 0 },
    null,
    { window: { from: "2026-09-01", to: "2026-11-09" }, covered: { from: "2026-09-01", to: "2026-11-09" }, truncated: false, next: null },
  ]) {
    const e = extendCoverage(c, older as any);
    assert.deepEqual(e.covered, c.covered);
    assert.equal(e.truncated, true, "still says older days are not loaded");
    assert.equal(e.next, null);
    assert.equal(loadOlderLabel(e, false), null);
  }
});

test("appendOlderFilings keeps one row per accession, loaded copy first", () => {
  const have = [{ accession: "a", v: 1 }, { accession: "b", v: 1 }];
  const out = appendOlderFilings(have, [{ accession: "b", v: 2 }, { accession: "c", v: 2 }, { accession: "c", v: 3 }, null as any]);
  assert.deepEqual(out, [{ accession: "a", v: 1 }, { accession: "b", v: 1 }, { accession: "c", v: 2 }]);
  assert.deepEqual(appendOlderFilings(have, null), have);
  assert.notEqual(appendOlderFilings(have, []), have, "returns a new array");
});

// RATCHET: a /data page that fetches a `*-history` filing route must read its coverage through this module —
// a page that renders `days`/`count` alone is the defect this file was written for.
test("RATCHET: every client page fetching a filing-history route reads coverage through filingCoverage", () => {
  const here = path.dirname(fileURLToPath(import.meta.url));
  const pages = path.join(here, "..", "pages");
  const ROUTES = /\/api\/data\/(insider|filings13f|earnings-language)\/history/;
  const users = fs.readdirSync(pages).filter((f) => f.endsWith(".tsx") && ROUTES.test(fs.readFileSync(path.join(pages, f), "utf8")));
  assert.deepEqual(users.sort(), ["earnings.tsx", "edgar13f.tsx", "filings.tsx"], "the set of pages reading these routes changed — update this ratchet with it");
  for (const f of users) {
    const src = fs.readFileSync(path.join(pages, f), "utf8");
    assert.match(src, /from "@\/lib\/filingCoverage"|from "\.\.\/lib\/filingCoverage"/, `${f} must import filingCoverage`);
    assert.match(src, /coverageWords\(/, `${f} must render coverageWords()`);
    assert.doesNotMatch(src, /last \{[^}]*days[^}]*\} days/, `${f} still prints "last N days" from the echoed days field`);
  }
});

// REGRESSION (read live 2026-11-16 13:20Z): /api/data/earnings-language/history?days=30 answered a 30-day
// window with seven archive days loaded and 442 releases; the earnings page header read "442 releases, last 30 days".
test("REGRESSION 2026-11-16: the earnings-language history payload reads as 7 of the 30 days asked for", () => {
  const live = {
    days: 30, window: { from: "2026-10-18", to: "2026-11-16" }, covered: { from: "2026-11-10", to: "2026-11-16" },
    truncated: true, next: "?from=2026-10-18&to=2026-11-09", count: 442, archive_start: "2026-07-04",
  };
  const c = readCoverage(live);
  assert.ok(c);
  assert.equal(coverageWords(c), "2026-11-10 → 2026-11-16 · 7 of the 30 days asked for — older days not loaded");
  assert.equal(loadOlderLabel(c, false), "Load older days (23 days before 2026-11-10 not loaded)");
  // the older page the server's `next` returns continues the run: the walk goes on, no gap is claimed
  const older = { window: { from: "2026-10-18", to: "2026-11-09" }, covered: { from: "2026-10-30", to: "2026-11-09" }, truncated: true, next: "?from=2026-10-18&to=2026-10-29" };
  const after = extendCoverage(c!, older);
  assert.equal(coverageWords(after), "2026-10-30 → 2026-11-16 · 18 of the 30 days asked for — older days not loaded");
  assert.equal(after.next, "?from=2026-10-18&to=2026-10-29");
});

// ── Form 4 by filed day ([PRODUCT] 2026-11-18) ───────────────────────────────

// production, read 2026-11-18 13:25Z (rows dropped): /api/data/insider/history?filed_from=2026-10-20
const FILED_LIVE = { kind: "raw", window_by: "filed_day", days: null, window: { from: "2026-10-20", to: "2026-11-18" },
  covered: { from: "2026-11-16", to: "2026-11-18" }, truncated: true, next: "?filed_from=2026-10-20&filed_to=2026-11-13",
  max_filings: 2000, days_scanned: 3, archive_start: "2026-07-04", omitted: 312, count: 1693, complete_days: 2,
  filed_days: [
    { day: "2026-11-18", filings: 22, omitted: 0, edgar_index: null, complete: null, checked_at: null },
    { day: "2026-11-17", filings: 1047, omitted: 310, edgar_index: 1047, complete: true, checked_at: "2026-11-18T07:15:41.235Z" },
    { day: "2026-11-16", filings: 624, omitted: 2, edgar_index: 624, complete: true, checked_at: "2026-11-18T07:15:41.235Z" },
  ] };
// the page `next` leads to: it ends on the Friday, two days before the loaded run starts
const FILED_OLDER = { window_by: "filed_day", window: { from: "2026-10-20", to: "2026-11-13" }, covered: { from: "2026-11-10", to: "2026-11-13" },
  truncated: true, next: "?filed_from=2026-10-20&filed_to=2026-11-09",
  filed_days: [
    { day: "2026-11-13", filings: 675, omitted: 189, edgar_index: 675, complete: true },
    { day: "2026-11-12", filings: 841, omitted: 0, edgar_index: 841, complete: true },
    { day: "2026-11-10", filings: 555, omitted: 0, edgar_index: 555, complete: true },
  ] };

test("a filed-day page is read as one, and an archive-day page still is not", () => {
  const c = readCoverage(FILED_LIVE)!;
  assert.equal(c.by, "filed_day");
  assert.equal(coverageWords(c), "2026-11-16 → 2026-11-18 · 3 of the 30 days asked for — older days not loaded");
  assert.equal(readCoverage(FORM4_LIVE)!.by, "archive_day");
  assert.equal(readCoverage({ ...FORM4_LIVE, window_by: "archive_day" })!.by, "archive_day");
});

test("REGRESSION (live 2026-11-18): the walk crosses a weekend — Monday's page continues at Friday's, which the archive-day rule refused", () => {
  const c = readCoverage(FILED_LIVE)!;
  const e = extendCoverage(c, FILED_OLDER);
  assert.deepEqual(e.covered, { from: "2026-11-10", to: "2026-11-18" });
  assert.equal(e.by, "filed_day");
  assert.equal(e.truncated, true);
  assert.equal(e.next, "?filed_from=2026-10-20&filed_to=2026-11-09");
  assert.equal(coverageWords(e), "2026-11-10 → 2026-11-18 · 9 of the 30 days asked for — older days not loaded");
  // the same two pages read as archive days must still stop: 11-13 is not the day before 11-16
  const asArchive = extendCoverage({ ...c, by: "archive_day" }, { ...FILED_OLDER, window_by: undefined });
  assert.deepEqual(asArchive.covered, c.covered);
  assert.equal(asArchive.next, null);
  // the last page reaches the window's first day
  const last = extendCoverage(e, { window_by: "filed_day", window: { from: "2026-10-20", to: "2026-11-09" }, covered: { from: "2026-10-20", to: "2026-11-09" }, truncated: false, next: null });
  assert.equal(coverageWords(last), "2026-10-20 → 2026-11-18 · all 30 days");
  assert.equal(loadOlderLabel(last, false), null);
});

test("a filed-day walk stops on a page that is not the one `next` asked for", () => {
  const c = readCoverage(FILED_LIVE)!;
  for (const older of [
    { ...FILED_OLDER, covered: { from: "2026-11-10", to: "2026-11-12" } },            // ends a day short of the day asked for
    { ...FILED_OLDER, window_by: "archive_day" },                                       // answered by archive day
    { ...FILED_OLDER, window_by: undefined },                                           // an older server ignoring filed_from
    { ...FILED_OLDER, covered: { from: "2026-10-01", to: "2026-11-13" } },            // reaches before the window
    null,
  ]) {
    const e = extendCoverage(c, older as any);
    assert.deepEqual(e.covered, c.covered);
    assert.equal(e.truncated, true);
    assert.equal(e.next, null);
  }
  // a `next` with no filed_to, or one that does not go back, is not followed
  for (const next of ["?filed_from=2026-10-20", "?filed_from=2026-10-20&filed_to=2026-11-16", "?filed_from=2026-10-20&filed_to=2026-11-17"]) {
    const e = extendCoverage({ ...c, next }, { ...FILED_OLDER, covered: { from: "2026-11-10", to: next.slice(-10) } });
    assert.deepEqual(e.covered, c.covered, next);
    assert.equal(e.next, null, next);
  }
});

test("each filed day says how it compares with EDGAR's index: complete, partial, or an unchecked sample", () => {
  const days = readFiledDays(FILED_LIVE);
  assert.deepEqual(days.map((d) => d.day), ["2026-11-18", "2026-11-17", "2026-11-16"]);
  assert.deepEqual(days.map(filedDayState), ["sample", "complete", "complete"]);
  assert.equal(filedDayWords(days[0]), "sample · not yet checked against EDGAR's index");
  assert.equal(filedDayWords(days[1]), "complete · all 1,047 on EDGAR's index");
  // 2026-11-06 as the refill left it on 2026-11-18: 512 held of 515 listed (3 fetches failed)
  const partial = readFiledDays({ filed_days: [{ day: "2026-11-06", filings: 512, edgar_index: 515, complete: false }] })[0];
  assert.equal(filedDayState(partial), "partial");
  assert.equal(filedDayWords(partial), "partial · 512 of the 515 on EDGAR's index");
  assert.equal(filedDaysSummary(days), "2 of 3 filed days complete against EDGAR's index · 1 not yet checked");
  assert.equal(filedDaysSummary([...days, partial]), "2 of 4 filed days complete against EDGAR's index · 1 partial · 1 not yet checked");
  assert.equal(filedDaysSummary(days.slice(1)), "all 2 filed days are complete against EDGAR's index");
  assert.equal(filedDaysSummary(days.slice(1, 2)), "the 1 filed day is complete against EDGAR's index");
  assert.equal(filedDaysSummary([]), "");
});

test("a day is never called complete without EDGAR's count, and malformed entries are dropped", () => {
  const days = readFiledDays({ filed_days: [
    { day: "2026-11-17", filings: 10, edgar_index: null, complete: true },   // "all N" needs the N
    { day: "2026-11-16", filings: 10, edgar_index: 12 },                     // counted but no verdict
    { day: "2026-11-16", filings: 99, edgar_index: 99, complete: true },     // a second entry for a day: the first stands
    { day: "nope", filings: 1 }, { day: "2026-11-15" }, { day: "2026-11-14", filings: -1 }, null, "x",
  ] });
  assert.deepEqual(days, [
    { day: "2026-11-17", filings: 10, edgarIndex: null, complete: null },
    { day: "2026-11-16", filings: 10, edgarIndex: null, complete: null },
  ]);
  assert.deepEqual(days.map(filedDayState), ["sample", "sample"]);
  assert.deepEqual(readFiledDays(FORM4_LIVE), []);
  assert.deepEqual(readFiledDays(null), []);
  assert.deepEqual(readFiledDays({ filed_days: "x" }), []);
});

test("appendFiledDays keeps the loaded entry for a day and stays newest first", () => {
  const have = readFiledDays(FILED_LIVE);
  const out = appendFiledDays(have, readFiledDays({ filed_days: [...FILED_OLDER.filed_days, { day: "2026-11-16", filings: 1, edgar_index: 624, complete: false }] }));
  assert.deepEqual(out.map((d) => d.day), ["2026-11-18", "2026-11-17", "2026-11-16", "2026-11-13", "2026-11-12", "2026-11-10"]);
  assert.equal(out[2].filings, 624);
});

test("the first query asks for N filed days ending today and leaves the last day to the server", () => {
  assert.equal(filedDayFirstQuery(Date.parse("2026-11-18T13:25:00Z"), 30), "?filed_from=2026-10-20");
  assert.equal(filedDayFirstQuery(Date.parse("2026-11-18T23:59:59Z"), 1), "?filed_from=2026-11-18");
  assert.equal(filedDayFirstQuery(Date.parse("2026-03-01T00:00:00Z"), 30), "?filed_from=2026-01-31");
});

test("RATCHET: the Form 4 page asks by filed day and prints each day's completeness", () => {
  const here = path.dirname(fileURLToPath(import.meta.url));
  const src = fs.readFileSync(path.join(here, "..", "pages", "filings.tsx"), "utf8");
  assert.match(src, /filedDayFirstQuery\(/, "the first page must be a filed-day window");
  assert.doesNotMatch(src, /"\?days=\d+"/, "the archive-day lookback is back on the Form 4 page");
  assert.match(src, /filedDayWords\(/, "each filed day must state its completeness");
  assert.match(src, /filedDaysSummary\(/, "the header must count the complete filed days");
  assert.match(src, /UTC archive days/, "a page answered by archive day must still be labelled as that");
});
