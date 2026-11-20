import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import { filings13fHistory, filings13fFiledDayStatus } from "./edgar13fFiledHistory";
import { FOCUSED_MAX_HOLDINGS, type Filing13F, type Holding13F } from "./edgar13f";
import type { Ledger13F, Ledger13FDay } from "./edgar13fCompleteness";
import { _resetFiledDayIndexForTests } from "./filedDayHistory";

// The 13F routes' reader by the day a filing was FILED
// (server/edgar13fFiledHistory.ts over server/filedDayHistory.ts). The
// regression this pins was measured on production 2026-11-20 00:20Z:
// `/api/data/filings13f/history?days=2` returned 640 filings, 600 of them
// filed on the 2026-11-16 deadline and recovered into archive day 2026-11-19.

const NOW = Date.UTC(2026, 10, 20, 0, 20, 0); // 2026-11-20T00:20Z
const holding = (i: number, n: number): Holding13F =>
  ({ issuer: `ISSUER ${i}`, titleOfClass: "COM", cusip: String(i).padStart(9, "0"), value: n - i, shares: 1, sharesType: "SH", putCall: null, discretion: "SOLE" } as Holding13F);
const f13 = (accession: string, filedAt: string | null, holdings = 0, extra: Partial<Filing13F> = {}): Filing13F => ({
  managerCik: "0001000001", managerName: "Small Cap Partners LP", periodOfReport: "2026-09-30", submissionType: "13F-HR",
  entryTotal: holdings, valueTotal: 1_000_000, otherManagersCount: 0,
  accession, filedAt, cik: "0001000001", indexUrl: `https://www.sec.gov/${accession}`,
  holdings: Array.from({ length: holdings }, (_, i) => holding(i, holdings)), holdingsOmitted: false, ...extra,
});
const many = (prefix: string, n: number, filedAt: string, holdings = 0, extra: Partial<Filing13F> = {}): Filing13F[] =>
  Array.from({ length: n }, (_, i) => f13(`${prefix}-${i}`, filedAt, holdings, extra));
function writeDay(base: string, day: string, rows: Filing13F[], gz = false): void {
  const dir = path.join(base, "filings13f");
  fs.mkdirSync(dir, { recursive: true });
  const text = rows.map((r) => JSON.stringify(r)).join("\n") + "\n";
  if (gz) fs.writeFileSync(path.join(dir, `${day}.jsonl.gz`), zlib.gzipSync(text));
  else fs.writeFileSync(path.join(dir, `${day}.jsonl`), text);
}
const tmpBase = () => fs.mkdtempSync(path.join(os.tmpdir(), "vt-13f-filed-"));
const CHECKED = "2026-11-19T11:27:47.470Z";
const ledgerDay = (over: Partial<Ledger13FDay>): Ledger13FDay => ({
  status: "checked", index_13f: 0, held: 0, unparsed: 0, failed: 0, deferred: 0,
  attempts: 1, complete: false, settled: false, checked_at: CHECKED, ...over,
});

/** Production's shape on 2026-11-20: the poll's sample of the deadline in
 *  the deadline's own file (feed timestamps), 600 more recovered three days
 *  later (index dates), quiet days whole. */
function seed(base: string): void {
  writeDay(base, "2026-11-16", [
    ...many("p16", 462, "2026-11-16T16:05:11-05:00", 30),
    ...many("p13", 20, "2026-11-13T17:01:00-05:00"),
  ], true);
  writeDay(base, "2026-11-17", many("p17", 95, "2026-11-17T10:00:00-05:00"));
  writeDay(base, "2026-11-19", [
    ...many("b16", 600, "2026-11-16", 0, { form: "13F-HR", backfilledAt: "2026-11-19T11:25:00.000Z" }),
    f13("p16-0", "2026-11-16", 30, { form: "13F-HR", backfilledAt: "2026-11-19T11:25:01.000Z" }), // a second copy of a polled filing
    ...many("p19", 32, "2026-11-19T09:00:00-05:00"),
    f13("nodate", null),
  ]);
}
const LEDGER: Ledger13F = { days: {
  "2026-11-17": ledgerDay({ index_13f: 95, held: 95, complete: true, settled: true }),
  "2026-11-16": ledgerDay({ index_13f: 1712, held: 1062, deferred: 650 }),
  "2026-11-11": ledgerDay({ status: "no_index" }),
} };

test("filings13fHistory: REGRESSION (production 2026-11-20) — filed_from/filed_to return the deadline day's polled AND recovered filings as one filed day, each day with EDGAR's index count and the ledger's completeness; a day the pass never checked reads null, not complete", async () => {
  _resetFiledDayIndexForTests();
  const base = tmpBase();
  seed(base);
  const page = await filings13fHistory({ filed_from: "2026-11-10", filed_to: "2026-11-19", limit: "500" }, 25, base, NOW, [], LEDGER);
  assert.equal(page.window_by, "filed_day");
  if (page.window_by !== "filed_day") return;
  // newest first; 11-19 (32) + 11-17 (95) fit under 500, the deadline's 1,062 would not
  assert.deepEqual(page.filed_days, [
    { day: "2026-11-19", filings: 32, omitted: 0, edgar_index: null, complete: null, checked_at: null },
    { day: "2026-11-17", filings: 95, omitted: 0, edgar_index: 95, complete: true, checked_at: CHECKED },
  ]);
  assert.deepEqual([page.count, page.truncated, page.next, page.complete_days], [127, true, "?filed_from=2026-11-10&filed_to=2026-11-16", 1]);
  assert.deepEqual(page.covered, { from: "2026-11-17", to: "2026-11-19" });
  assert.equal(Object.keys(page)[Object.keys(page).length - 1], "filings", "the rows come last, after everything that describes them");

  // the continuation: the deadline day whole (over the cap, because a filed day is atomic), then 11-13
  const next = await filings13fHistory({ filed_from: "2026-11-10", filed_to: "2026-11-16", limit: "500" }, 25, base, NOW, [], LEDGER);
  if (next.window_by !== "filed_day") return assert.fail("filed-day page expected");
  assert.deepEqual(next.filed_days[0], { day: "2026-11-16", filings: 1062, omitted: 0, edgar_index: 1712, complete: false, checked_at: CHECKED },
    "462 polled + 600 recovered, the twice-archived accession counted once; EDGAR lists 1,712 and the page says the day is short");
  assert.equal(next.count, 1062, "the first filed day of a page is returned whole even over the cap");
  assert.deepEqual([next.truncated, next.next], [true, "?filed_from=2026-11-10&filed_to=2026-11-13"]);
  assert.equal(next.filings.filter((f) => f.backfilledAt).length, 600, "recovered rows keep backfilledAt; the polled copy of a twice-archived filing is the one served");
  assert.equal(next.filings.filter((f) => f.accession === "p16-0").length, 1);
  assert.ok(next.filings.every((f) => f.filedAt?.slice(0, 10) === "2026-11-16"));
  assert.ok(!next.filings.some((f) => f.accession === "nodate"), "a row with no filedAt is in no filed day");

  const last = await filings13fHistory({ filed_from: "2026-11-10", filed_to: "2026-11-13" }, 25, base, NOW, [], LEDGER);
  if (last.window_by !== "filed_day") return assert.fail("filed-day page expected");
  assert.deepEqual([last.count, last.truncated, last.next, last.complete_days], [20, false, null, 0]);
  assert.deepEqual(last.covered, { from: "2026-11-10", to: "2026-11-13" });
});

test("filings13fHistory: the per-route holdings trim applies to a filed-day page (25 on the RAW route, the full as-filed table on the keyed mirror)", async () => {
  _resetFiledDayIndexForTests();
  const base = tmpBase();
  seed(base);
  const q = { filed_from: "2026-11-16", filed_to: "2026-11-16" };
  const raw = await filings13fHistory(q, 25, base, NOW, [], LEDGER);
  const keyed = await filings13fHistory(q, FOCUSED_MAX_HOLDINGS, base, NOW, [], LEDGER);
  const polled = (p: typeof raw) => p.filings.filter((f) => f.accession.startsWith("p16-"));
  assert.equal(polled(raw).length, 462);
  assert.ok(polled(raw).every((f) => f.holdings?.length === 25), "RAW display trim");
  assert.ok(polled(keyed).every((f) => f.holdings?.length === 30), "keyed mirror: every stored position");
});

test("filings13fHistory: without a valid filed day the archive-day page is served exactly as before, marked window_by archive_day (live merge included)", async () => {
  _resetFiledDayIndexForTests();
  const base = tmpBase();
  seed(base);
  const byArchive = await filings13fHistory({ from: "2026-11-19", to: "2026-11-19" }, 25, base, NOW, [], LEDGER);
  assert.equal(byArchive.window_by, "archive_day");
  assert.equal(byArchive.count, 634, "archive day 11-19 as written: 601 deadline filings recovered that night + 32 + the undated row — the reading the filed-day window exists to replace");
  assert.ok(!("filed_days" in byArchive));
  const legacy = await filings13fHistory({ days: "1" }, 25, base, NOW, [f13("LIVE", "2026-11-19T19:00:00-05:00")], LEDGER);
  assert.deepEqual([legacy.window_by, legacy.days], ["archive_day", 1]);
  assert.ok(legacy.filings.some((f) => f.accession === "LIVE"), "the live poll is still merged on a page that ends today");
  const bad = await filings13fHistory({ filed_from: "soon", filed_to: "2026-13-40", days: "1" }, 25, base, NOW, [], LEDGER);
  assert.equal(bad.window_by, "archive_day", "an invalid filed day is not a filed-day request, and the page says which window it used");
});

test("filings13fFiledDayStatus: a day EDGAR had no index for, or one the ledger does not hold, is not a checked day; an unreadable or absent ledger file reads as nothing checked", async () => {
  const day = { day: "2026-11-11", filings: 3, omitted: 0 };
  assert.deepEqual(filings13fFiledDayStatus(day, LEDGER), { ...day, edgar_index: null, complete: null, checked_at: null });
  assert.deepEqual(filings13fFiledDayStatus({ ...day, day: "2026-08-14" }, LEDGER), { day: "2026-08-14", filings: 3, omitted: 0, edgar_index: null, complete: null, checked_at: null });
  assert.deepEqual(filings13fFiledDayStatus({ ...day, day: "2026-11-17" }, LEDGER).complete, true);
  // default ledger: read from the heartbeat dir, which holds none here
  _resetFiledDayIndexForTests();
  const base = tmpBase();
  seed(base);
  const prior = process.env.DATA_DIR;
  process.env.DATA_DIR = tmpBase();
  try {
    const page = await filings13fHistory({ filed_from: "2026-11-17", filed_to: "2026-11-17" }, 25, base, NOW, []);
    if (page.window_by !== "filed_day") return assert.fail("filed-day page expected");
    assert.deepEqual([page.filed_days[0].filings, page.filed_days[0].complete, page.complete_days], [95, null, 0]);
  } finally {
    if (prior === undefined) delete process.env.DATA_DIR; else process.env.DATA_DIR = prior;
  }
});
