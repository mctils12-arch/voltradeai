import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import type { FilingHistoryArchive } from "./filingHistoryPage";
import {
  parseFiledDayQuery, buildFiledDayPage, refreshFiledDayIndex, filedDaysInArchiveDay, filedDayIndexScans,
  _resetFiledDayIndexForTests,
} from "./filedDayHistory";
import { form4History, form4FiledDayStatus, isForm4Record } from "./edgarForm4FiledHistory";
import type { Form4Filing } from "./edgarForm4";
import type { DayLedger, LedgerDay } from "./edgarForm4Completeness";

// A filing archive read by the day a filing was FILED (server/filedDayHistory.ts)
// and the Form 4 routes' reader over it (server/edgarForm4FiledHistory.ts).
// The regression these pin was measured on production 2026-11-17: the
// archive-day window 2026-11-13..11-15 returned 411 of the 675 Form 4s EDGAR
// lists for Friday 2026-11-13 and 189 rows that are not Form 4s; the other 262
// had been recovered by the nightly completeness pass into archive day 11-17.

interface Row { accession: string; filedAt: string | null; issuerCik?: string | null; tag?: string }
const NOW = Date.UTC(2026, 10, 17, 18, 0, 0); // 2026-11-17T18:00Z
const row = (accession: string, filedAt: string | null, issuerCik: string | null = "0000320193", tag?: string): Row =>
  ({ accession, filedAt, issuerCik, ...(tag ? { tag } : {}) });
const many = (prefix: string, n: number, filedAt: string, issuerCik: string | null = "0000320193"): Row[] =>
  Array.from({ length: n }, (_, i) => row(`${prefix}-${i}`, filedAt, issuerCik));

function archiveAt(dir: string, over: Partial<FilingHistoryArchive<Row>> = {}): FilingHistoryArchive<Row> {
  return { dir, archiveStart: "2026-07-04", maxFilings: 2000, maxDays: 90, defaultDays: 30, maxSpanDays: 400, ...over };
}
function writeDay(dir: string, day: string, rows: Row[], gz = false): void {
  fs.mkdirSync(dir, { recursive: true });
  const text = rows.map((r) => JSON.stringify(r)).join("\n") + "\n";
  if (gz) fs.writeFileSync(path.join(dir, `${day}.jsonl.gz`), zlib.gzipSync(text));
  else fs.writeFileSync(path.join(dir, `${day}.jsonl`), text);
}
const tmpBase = (tag: string) => fs.mkdtempSync(path.join(os.tmpdir(), `vt-filed-${tag}-`));
const record = (r: Row) => Boolean(r.issuerCik);

/** The production shape of 2026-11-13..11-17: the poll's sample of Friday in
 *  Friday's and Saturday's files beside non-Form-4 shells and other days'
 *  rows; the rest of Friday recovered into Tuesday's file. */
function seedNovember(dir: string): void {
  writeDay(dir, "2026-11-13", [
    ...many("p13", 300, "2026-11-13"), ...many("shell13", 189, "2026-11-13", null),
    ...many("p12", 81, "2026-11-12"), ...many("p10", 7, "2026-11-10"),
  ], true);
  writeDay(dir, "2026-11-14", many("q13", 111, "2026-11-13"), true);
  writeDay(dir, "2026-11-17", [
    ...many("b13", 262, "2026-11-13"), ...many("b12", 391, "2026-11-12"), ...many("b16", 532, "2026-11-16"),
    ...many("p16", 73, "2026-11-16"), ...many("p17", 101, "2026-11-17"), ...many("shell17", 169, "2026-11-17", null),
  ]);
}

test("parseFiledDayQuery: null unless filed_from or filed_to is a valid day (the caller then serves the archive-day window); filed_to alone = the default span ending there; filed_from alone runs to today; to clamps to today; from > to collapses; the span is bounded by raising from; no clamp at the archive start; limit capped", () => {
  const a = archiveAt("/nonexistent");
  assert.equal(parseFiledDayQuery({}, a, NOW), null);
  assert.equal(parseFiledDayQuery({ from: "2026-11-01", to: "2026-11-10", days: "5" }, a, NOW), null, "from/to/days are the archive-day window");
  assert.equal(parseFiledDayQuery({ filed_from: "banana", filed_to: "2026-13-40" }, a, NOW), null, "an invalid day is not a filed-day request");
  assert.deepEqual(parseFiledDayQuery({ filed_from: "2026-11-10", filed_to: "2026-11-13" }, a, NOW), { from: "2026-11-10", to: "2026-11-13", maxFilings: 2000 });
  assert.deepEqual(parseFiledDayQuery({ filed_to: "2026-11-13" }, a, NOW), { from: "2026-10-15", to: "2026-11-13", maxFilings: 2000 });
  assert.deepEqual(parseFiledDayQuery({ filed_from: "2026-11-10" }, a, NOW), { from: "2026-11-10", to: "2026-11-17", maxFilings: 2000 });
  assert.equal(parseFiledDayQuery({ filed_from: "2026-11-10", filed_to: "2027-01-01" }, a, NOW)!.to, "2026-11-17", "to is clamped to today");
  const swapped = parseFiledDayQuery({ filed_from: "2026-11-13", filed_to: "2026-11-10" }, a, NOW)!;
  assert.equal(swapped.from, swapped.to);
  assert.equal(parseFiledDayQuery({ filed_from: "2020-01-01", filed_to: "2026-11-13" }, a, NOW)!.from, "2025-10-10", "400-day span, and a day before the archive start is allowed");
  assert.equal(parseFiledDayQuery({ filed_from: ["2026-11-12", "2026-11-01"], filed_to: "2026-11-13" }, a, NOW)!.from, "2026-11-12", "a repeated param takes its first value");
  assert.equal(parseFiledDayQuery({ filed_to: "2026-11-13", limit: "99999" }, a, NOW)!.maxFilings, 2000);
  assert.equal(parseFiledDayQuery({ filed_to: "2026-11-13", limit: "0" }, a, NOW)!.maxFilings, 1);
  assert.equal(parseFiledDayQuery({ filed_to: "2026-11-13", limit: "50" }, a, NOW)!.maxFilings, 50);
});

test("REGRESSION (production 2026-11-17): the filed-day window returns every Form 4 the archive holds for Friday 2026-11-13 — the poll's 411 and the 262 recovered into archive day 11-17 — and leaves out, counted, the 189 rows that are not Form 4s; the archive-day window over the same days holds 411", async () => {
  _resetFiledDayIndexForTests();
  const dir = path.join(tmpBase("nov"), "filings");
  seedNovember(dir);
  const a = archiveAt(dir);
  const page = await buildFiledDayPage(parseFiledDayQuery({ filed_from: "2026-11-13", filed_to: "2026-11-13" }, a, NOW)!, a, { isRecord: record });
  assert.equal(page.window_by, "filed_day");
  assert.equal(page.count, 673);
  assert.equal(page.filings.length, 673);
  assert.ok(page.filings.every((f) => f.filedAt === "2026-11-13" && f.issuerCik), "only rows filed that day, only records");
  assert.deepEqual(page.filed_days, [{ day: "2026-11-13", filings: 673, omitted: 189 }]);
  assert.equal(page.omitted, 189);
  assert.equal(page.archive_days_read, 3, "the filed day's rows sit in three day files");
  assert.deepEqual([page.truncated, page.next, page.days, page.days_scanned], [false, null, null, 1]);
  assert.deepEqual(page.covered, { from: "2026-11-13", to: "2026-11-13" });
  // the archive-day reader over the same three days, for the record of what it cannot see
  const { buildFilingHistoryPage } = await import("./filingHistoryPage");
  const byArchiveDay = buildFilingHistoryPage({ from: "2026-11-13", to: "2026-11-15" }, a, null, NOW);
  assert.equal(byArchiveDay.filings.filter((f) => f.filedAt === "2026-11-13" && f.issuerCik).length, 411);
});

test("a filed-day window walks filed days newest first, lists each with its counts, and a day with no row in the archive is simply absent (the window still says what was asked)", async () => {
  _resetFiledDayIndexForTests();
  const dir = path.join(tmpBase("walk"), "filings");
  seedNovember(dir);
  const a = archiveAt(dir);
  const page = await buildFiledDayPage(parseFiledDayQuery({ filed_from: "2026-11-09", filed_to: "2026-11-17" }, a, NOW)!, a, { isRecord: record });
  assert.deepEqual(page.filed_days, [
    { day: "2026-11-17", filings: 101, omitted: 169 },
    { day: "2026-11-16", filings: 605, omitted: 0 },
    { day: "2026-11-13", filings: 673, omitted: 189 },
    { day: "2026-11-12", filings: 472, omitted: 0 },
    { day: "2026-11-10", filings: 7, omitted: 0 },
  ]);
  assert.equal(page.count, 101 + 605 + 673 + 472 + 7);
  assert.equal(page.omitted, 358);
  assert.deepEqual(page.window, { from: "2026-11-09", to: "2026-11-17" });
  assert.deepEqual(page.covered, { from: "2026-11-09", to: "2026-11-17" }, "not truncated: the whole window was read");
  assert.deepEqual(page.filings.slice(0, 2).map((f) => f.filedAt), ["2026-11-17", "2026-11-17"], "newest filed day first");
  assert.equal(page.filings[page.count - 1].filedAt, "2026-11-10");
});

test("paging: whole FILED days are atomic — the page stops before the filed day that would exceed the cap, `next` continues with neither a gap nor a repeat, and the newest filed day is returned whole even when it alone exceeds the cap", async () => {
  _resetFiledDayIndexForTests();
  const dir = path.join(tmpBase("page"), "filings");
  seedNovember(dir);
  const a = archiveAt(dir);
  const first = await buildFiledDayPage(parseFiledDayQuery({ filed_from: "2026-11-09", filed_to: "2026-11-17", limit: "800" }, a, NOW)!, a, { isRecord: record });
  assert.deepEqual(first.filed_days.map((d) => d.day), ["2026-11-17", "2026-11-16"], "101 + 605 fits 800; 673 more does not");
  assert.deepEqual([first.truncated, first.next], [true, "?filed_from=2026-11-09&filed_to=2026-11-13&limit=800"]);
  assert.deepEqual(first.covered, { from: "2026-11-16", to: "2026-11-17" });
  const seen = new Set(first.filings.map((f) => f.accession));
  let next = first.next;
  let pages = 1;
  while (next) {
    const q = Object.fromEntries(new URLSearchParams(next.slice(1)));
    const p = await buildFiledDayPage(parseFiledDayQuery(q, a, NOW)!, a, { isRecord: record });
    for (const f of p.filings) { assert.ok(!seen.has(f.accession), `${f.accession} served twice`); seen.add(f.accession); }
    next = p.next;
    pages++;
    assert.ok(pages < 10, "the walk ends");
  }
  assert.equal(seen.size, 101 + 605 + 673 + 472 + 7, "the walk returns every record in the window once");
  assert.equal(pages, 3, "[17,16] [13] [12,10] at a cap of 800: 673 alone, then 472 + 7");
  const tiny = await buildFiledDayPage(parseFiledDayQuery({ filed_from: "2026-11-09", filed_to: "2026-11-16", limit: "10" }, a, NOW)!, a, { isRecord: record });
  assert.deepEqual([tiny.count, tiny.truncated, tiny.next], [605, true, "?filed_from=2026-11-09&filed_to=2026-11-13&limit=10"], "the page always makes progress");
  const full = await buildFiledDayPage(parseFiledDayQuery({ filed_from: "2026-11-09", filed_to: "2026-11-17" }, a, NOW)!, a, { isRecord: record });
  assert.equal(full.next, null, "the default cap needs no limit in next, and nothing is truncated here");
});

test("one row per accession across day files: a filing the restart re-appended is served once, and a record row is kept over a non-record row with the same accession whichever file comes first", async () => {
  _resetFiledDayIndexForTests();
  const dir = path.join(tmpBase("dup"), "filings");
  writeDay(dir, "2026-09-15", [row("A", "2026-09-15", "1", "first"), row("A", "2026-09-15", "1", "again"), row("S", "2026-09-15", null), row("T", "2026-09-15", "7", "poll")], true);
  writeDay(dir, "2026-09-16", [row("A", "2026-09-15", "1", "next-day"), row("T", "2026-09-15", null)], true);
  writeDay(dir, "2026-11-18", [row("S", "2026-09-15", "9", "recovered"), row("B", "2026-09-15", "2")]);
  const a = archiveAt(dir);
  const page = await buildFiledDayPage(parseFiledDayQuery({ filed_from: "2026-09-15", filed_to: "2026-09-15" }, a, NOW)!, a, { isRecord: record });
  assert.deepEqual(page.filings.map((f) => `${f.accession}:${f.tag ?? ""}`).sort(), ["A:first", "B:", "S:recovered", "T:poll"]);
  assert.deepEqual(page.filed_days, [{ day: "2026-09-15", filings: 4, omitted: 0 }], "S and T each have a record row: nothing is omitted");
  const noFilter = await buildFiledDayPage(parseFiledDayQuery({ filed_from: "2026-09-15", filed_to: "2026-09-15" }, a, NOW)!, a);
  assert.equal(noFilter.count, 4, "without isRecord every row is a record: still one per accession");
});

test("rows with no filedAt, torn lines and an archive `project` are handled: undated rows are never in a filed-day page, torn lines in the files read are counted once per file, the projection is applied", async () => {
  _resetFiledDayIndexForTests();
  const dir = path.join(tmpBase("odd"), "filings");
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, "2026-10-01.jsonl"),
    [JSON.stringify(row("X", "2026-10-01")), JSON.stringify(row("U", null)), '{"accession":"TORN","filedAt":"2026-10-0', JSON.stringify(row("Y", "2026-09-30"))].join("\n") + "\n");
  const a = archiveAt(dir, { project: (r) => ({ ...r, tag: "projected" }) });
  const page = await buildFiledDayPage(parseFiledDayQuery({ filed_from: "2026-09-30", filed_to: "2026-10-01" }, a, NOW)!, a, { isRecord: record });
  assert.deepEqual(page.filings.map((f) => f.accession), ["X", "Y"]);
  assert.ok(page.filings.every((f) => f.tag === "projected"));
  assert.equal(page.malformed_lines, 1, "one torn line, in a file read for two filed days, counted once");
  assert.equal(page.archive_days_read, 1);
});

test("the index reads a day file once and again only when it changes: an unchanged archive costs no file read on the next request, an appended file is re-read alone, a gzipped copy replacing the plain file is picked up, a deleted day leaves the index", async () => {
  _resetFiledDayIndexForTests();
  const dir = path.join(tmpBase("idx"), "filings");
  writeDay(dir, "2026-11-10", many("a", 3, "2026-11-10"), true);
  writeDay(dir, "2026-11-11", many("b", 3, "2026-11-11"));
  let idx = await refreshFiledDayIndex(dir);
  assert.deepEqual(Array.from(idx.keys()).sort(), ["2026-11-10", "2026-11-11"]);
  assert.equal(filedDayIndexScans(dir), 2);
  await refreshFiledDayIndex(dir);
  assert.equal(filedDayIndexScans(dir), 2, "nothing changed: nothing read");
  fs.appendFileSync(path.join(dir, "2026-11-11.jsonl"), JSON.stringify(row("late", "2026-07-06")) + "\n");
  idx = await refreshFiledDayIndex(dir);
  assert.equal(filedDayIndexScans(dir), 3, "only the appended day is read again");
  assert.deepEqual(Array.from(idx.get("2026-11-11")!.filedDays).sort(), ["2026-07-06", "2026-11-11"]);
  const a = archiveAt(dir);
  const july = await buildFiledDayPage(parseFiledDayQuery({ filed_from: "2026-07-06", filed_to: "2026-07-06" }, a, NOW)!, a);
  assert.deepEqual(july.filings.map((f) => f.accession), ["late"], "a filing recovered months later is found under its filed day");
  // the day's plain file is gzipped (what gzipOldFilingDays does)
  const plain = path.join(dir, "2026-11-11.jsonl");
  fs.writeFileSync(`${plain}.gz`, zlib.gzipSync(fs.readFileSync(plain)));
  fs.unlinkSync(plain);
  await refreshFiledDayIndex(dir);
  assert.equal(filedDayIndexScans(dir), 4);
  assert.deepEqual(Array.from(filedDaysInArchiveDay(dir, "2026-11-11")).sort(), ["2026-07-06", "2026-11-11"]);
  fs.unlinkSync(path.join(dir, "2026-11-10.jsonl.gz"));
  idx = await refreshFiledDayIndex(dir);
  assert.deepEqual(Array.from(idx.keys()), ["2026-11-11"]);
  // two requests at once share one refresh
  _resetFiledDayIndexForTests();
  await Promise.all([refreshFiledDayIndex(dir), refreshFiledDayIndex(dir), refreshFiledDayIndex(dir)]);
  assert.equal(filedDayIndexScans(dir), 1, "one day file, read once for three concurrent callers");
});

test("an archive directory that does not exist yet is an empty page, not an error", async () => {
  _resetFiledDayIndexForTests();
  const a = archiveAt(path.join(tmpBase("none"), "filings"));
  const page = await buildFiledDayPage(parseFiledDayQuery({ filed_to: "2026-11-13" }, a, NOW)!, a);
  assert.deepEqual([page.count, page.truncated, page.days_scanned, page.archive_days_read], [0, false, 0, 0]);
  assert.deepEqual(page.filed_days, []);
});

// ── the Form 4 routes' reader ────────────────────────────────────────────────

const f4 = (accession: string, filedAt: string, issuerCik: string | null = "0000320193"): Form4Filing => ({
  accession, filedAt, cik: "1", indexUrl: `https://www.sec.gov/${accession}`,
  issuerCik, issuerName: issuerCik ? "Apple Inc." : null, issuerTradingSymbol: issuerCik ? "AAPL" : null,
  periodOfReport: null, owners: [], transactions: [],
});
const ledgerDay = (day: string, over: Partial<LedgerDay>): LedgerDay => ({
  day, status: "checked", index_form4: 0, held: 0, unparsed: 0, failed: 0, deferred: 0,
  complete: false, settled: false, attempts: 1, checked_at: "2026-11-17T07:19:42.516Z", ...over,
});

function seedForm4(base: string): void {
  const dir = path.join(base, "filings");
  writeDay(dir, "2026-11-13", [
    ...Array.from({ length: 411 }, (_, i) => f4(`p13-${i}`, "2026-11-13")),
    ...Array.from({ length: 189 }, (_, i) => f4(`s13-${i}`, "2026-11-13", null)),
  ] as unknown as Row[], true);
  writeDay(dir, "2026-11-17", [
    ...Array.from({ length: 262 }, (_, i) => ({ ...f4(`b13-${i}`, "2026-11-13"), form: "4", backfilledAt: "2026-11-17T07:10:00.000Z" })),
    ...Array.from({ length: 624 }, (_, i) => f4(`x16-${i}`, "2026-11-16")),
    ...Array.from({ length: 40 }, (_, i) => f4(`p09-${i}`, "2026-09-15")),
  ] as unknown as Row[]);
}

test("form4History: filed_from/filed_to serve the filed-day page with each day's EDGAR index count and completeness from the pass's ledger — complete when the ledger says so, short when it says so, null when the day was never checked; without them the archive-day page is served exactly as before, marked window_by archive_day", async () => {
  _resetFiledDayIndexForTests();
  const base = tmpBase("f4");
  seedForm4(base);
  const ledger: DayLedger = {
    archive_start: "2026-07-04", updated_at: "2026-11-17T07:19:42.516Z",
    days: {
      "2026-11-13": ledgerDay("2026-11-13", { index_form4: 675, held: 673, failed: 2 }),
      "2026-11-16": ledgerDay("2026-11-16", { index_form4: 624, held: 624, complete: true, settled: true }),
      "2026-11-11": ledgerDay("2026-11-11", { status: "no_index" }),
    },
  };
  const filed = await form4History({ filed_from: "2026-09-01", filed_to: "2026-11-16" }, base, NOW, [], ledger);
  assert.equal(filed.window_by, "filed_day");
  if (filed.window_by !== "filed_day") return;
  assert.deepEqual(filed.filed_days, [
    { day: "2026-11-16", filings: 624, omitted: 0, edgar_index: 624, complete: true, checked_at: "2026-11-17T07:19:42.516Z" },
    { day: "2026-11-13", filings: 673, omitted: 189, edgar_index: 675, complete: false, checked_at: "2026-11-17T07:19:42.516Z" },
    { day: "2026-09-15", filings: 40, omitted: 0, edgar_index: null, complete: null, checked_at: null },
  ]);
  assert.equal(filed.complete_days, 1);
  assert.equal(filed.count, 624 + 673 + 40);
  assert.equal(filed.filings.filter((f) => f.backfilledAt).length, 262, "recovered rows keep their backfilledAt");
  assert.equal(Object.keys(filed)[Object.keys(filed).length - 1], "filings", "the rows come last, after everything that describes them");
  // for a completed day the filed-day window returns exactly EDGAR's count (the entry's own check)
  const complete = await form4History({ filed_from: "2026-11-16", filed_to: "2026-11-16" }, base, NOW, [], ledger);
  assert.ok(complete.window_by === "filed_day" && complete.count === complete.filed_days[0].edgar_index);

  const byArchive = await form4History({ from: "2026-11-13", to: "2026-11-15" }, base, NOW, [], ledger);
  assert.equal(byArchive.window_by, "archive_day");
  assert.equal(byArchive.count, 600, "the archive-day page is unchanged: every row as written, shells included");
  assert.ok(!("filed_days" in byArchive));
  const legacy = await form4History({ days: "1" }, base, NOW, [f4("LIVE", "2026-11-17")], ledger);
  assert.deepEqual([legacy.window_by, legacy.days, legacy.count], ["archive_day", 1, 262 + 624 + 40 + 1], "legacy ?days= and its live merge are untouched");
  const bad = await form4History({ filed_from: "soon", from: "2026-11-13", to: "2026-11-15" }, base, NOW, [], ledger);
  assert.equal(bad.window_by, "archive_day", "an invalid filed day is not a filed-day request, and the page says which window it used");
});

test("form4FiledDayStatus / isForm4Record: a holiday or errored ledger entry is not a checked day; a row is a record only when the document named an issuer", () => {
  const day = { day: "2026-11-11", filings: 3, omitted: 0 };
  const ledger: DayLedger = { archive_start: "2026-07-04", updated_at: "", days: {
    "2026-11-11": ledgerDay("2026-11-11", { status: "no_index" }),
    "2026-11-12": ledgerDay("2026-11-12", { status: "error", index_form4: 0 }),
  } };
  assert.deepEqual(form4FiledDayStatus(day, ledger), { ...day, edgar_index: null, complete: null, checked_at: null });
  assert.equal(form4FiledDayStatus({ ...day, day: "2026-11-12" }, ledger).complete, null);
  assert.equal(form4FiledDayStatus({ ...day, day: "2026-07-06" }, ledger).edgar_index, null, "a day the ledger has never seen");
  assert.equal(isForm4Record(f4("a", "2026-11-13")), true);
  assert.equal(isForm4Record(f4("b", "2026-11-13", null)), false);
});
