import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import {
  parseFilingHistoryQuery, readFilingHistoryWindow, readFilingArchiveDay, filingHistoryNextQuery,
  buildFilingHistoryPage, type FilingHistoryArchive,
} from "./filingHistoryPage";

// The generic contract behind every per-UTC-day filing archive page
// (server/filingHistoryPage.ts). These tests mirror the earnings-language
// reader's own suite (sec8kEarnings.test.ts, v1.0.1350) so the two contracts
// cannot drift apart: whole days atomic, soft cap by at most one day, an
// honest cursor, the live merge only when the page ends today.

interface Row { accession: string; filedAt: string | null; ticker: string }
const NOW_11_11 = Date.UTC(2026, 10, 11, 13, 0, 0); // 2026-11-11T13:00Z
const mk = (acc: string, filedAt: string, ticker = "TST"): Row => ({ accession: acc, filedAt, ticker });

function archiveAt(dir: string, over: Partial<FilingHistoryArchive<Row>> = {}): FilingHistoryArchive<Row> {
  return { dir, archiveStart: "2026-07-04", maxFilings: 500, maxDays: 90, defaultDays: 30, maxSpanDays: 400, ...over };
}

function seedDays(dir: string, days: string[], perDay: number, gzDays: string[] = []): void {
  fs.mkdirSync(dir, { recursive: true });
  for (const day of days) {
    const lines = Array.from({ length: perDay }, (_, i) => JSON.stringify(mk(`${day}-${i}`, `${day}T1${i % 10}:00:00`))).join("\n") + "\n";
    if (gzDays.includes(day)) fs.writeFileSync(path.join(dir, `${day}.jsonl.gz`), zlib.gzipSync(lines));
    else fs.writeFileSync(path.join(dir, `${day}.jsonl`), lines);
  }
}

const tmp = (tag: string) => path.join(fs.mkdtempSync(path.join(os.tmpdir(), `vt-fhp-${tag}-`)), "filings");

test("parseFilingHistoryQuery: legacy ?days= (capped at the archive's maxDays, default defaultDays, garbage -> default) means [today-(N-1), today]; from/to wins and nulls `days`; to clamps to today; from clamps to the archive start; from>to collapses; span bounded by raising from; limit capped at maxFilings", () => {
  const a = archiveAt("/nonexistent");
  const w = parseFilingHistoryQuery({ days: "7" }, a, NOW_11_11);
  assert.deepEqual([w.from, w.to, w.days, w.maxFilings], ["2026-11-05", "2026-11-11", 7, 500]);
  assert.equal(parseFilingHistoryQuery({ days: "200" }, a, NOW_11_11).days, 90, "the 90-day legacy cap is unchanged");
  assert.equal(parseFilingHistoryQuery({}, a, NOW_11_11).days, 30);
  assert.equal(parseFilingHistoryQuery({ days: "banana" }, a, NOW_11_11).days, 30);
  assert.equal(parseFilingHistoryQuery({ days: ["5", "9"] }, a, NOW_11_11).days, 5, "a repeated param takes its first value");
  const ft = parseFilingHistoryQuery({ from: "2026-09-01", to: "2026-09-30" }, a, NOW_11_11);
  assert.deepEqual([ft.from, ft.to, ft.days], ["2026-09-01", "2026-09-30", null], "from/to wins; `days` is null (not echoed as a fake lookback)");
  assert.equal(parseFilingHistoryQuery({ from: "2026-09-01", to: "2027-01-01" }, a, NOW_11_11).to, "2026-11-11", "to is clamped to today");
  assert.equal(parseFilingHistoryQuery({ from: "2020-01-01", to: "2026-07-10" }, a, NOW_11_11).from, "2026-07-04", "nothing exists before the archive start");
  const swapped = parseFilingHistoryQuery({ from: "2026-10-20", to: "2026-10-01" }, a, NOW_11_11);
  assert.equal(swapped.from, swapped.to, "from > to collapses to a one-day window rather than throwing");
  const span = parseFilingHistoryQuery({ from: "2026-07-04", to: "2028-06-01" }, a, Date.UTC(2028, 5, 1));
  assert.equal(Math.round((Date.parse(span.to) - Date.parse(span.from)) / 86400_000) + 1, 400, "span bounded by raising from");
  assert.equal(parseFilingHistoryQuery({ to: "2026-11-10" }, a, NOW_11_11).from, "2026-10-12", "to alone => defaultDays ending at to");
  assert.equal(parseFilingHistoryQuery({ limit: "99999" }, a, NOW_11_11).maxFilings, 500, "limit is capped at the archive's maxFilings");
  assert.equal(parseFilingHistoryQuery({ limit: "0" }, a, NOW_11_11).maxFilings, 1);
  assert.equal(parseFilingHistoryQuery({ from: "2026-13-40" }, a, NOW_11_11).from, "2026-10-13", "an invalid day is ignored, not thrown: falls back to the 30-day default");
  // a differently-configured archive is honoured, not hardcoded
  const b = archiveAt("/nonexistent", { maxDays: 120, defaultDays: 10, maxFilings: 2000, archiveStart: "2026-07-05" });
  assert.deepEqual([parseFilingHistoryQuery({ days: "500" }, b, NOW_11_11).days, parseFilingHistoryQuery({}, b, NOW_11_11).days, parseFilingHistoryQuery({}, b, NOW_11_11).maxFilings], [120, 10, 2000]);
  assert.equal(parseFilingHistoryQuery({ from: "2026-07-04" }, b, NOW_11_11).from, "2026-07-05");
});

test("readFilingHistoryWindow: whole archive days are atomic, the page stops BEFORE the day that would exceed the cap, and truncated/covered/next_to say exactly what was read", () => {
  const dir = tmp("win");
  // 6 archive days x 40 filings = 240; cap 100 => page 1 holds 11-10 + 11-09 (80), stops before 11-06
  seedDays(dir, ["2026-11-03", "2026-11-04", "2026-11-05", "2026-11-06", "2026-11-09", "2026-11-10"], 40);
  const a = archiveAt(dir);
  const win = parseFilingHistoryQuery({ from: "2026-11-01", to: "2026-11-10", limit: "100" }, a, NOW_11_11);
  const page = readFilingHistoryWindow(win, a);
  assert.equal(page.filings.length, 80, "80 not 100 — a day is never split");
  assert.equal(page.truncated, true);
  assert.deepEqual(page.covered, { from: "2026-11-07", to: "2026-11-10" },
    "covered names the days read in full — including the empty weekend days 11-07/11-08, which WERE read (an empty day is covered, not skipped)");
  assert.equal(page.next_to, "2026-11-06", "the first day NOT read is the cursor");
  assert.deepEqual(page.window, { from: "2026-11-01", to: "2026-11-10" }, "the requested window is echoed, not silently rewritten");
  assert.equal(page.days_scanned, 4);
  assert.equal(filingHistoryNextQuery(page, a), "?from=2026-11-01&to=2026-11-06&limit=100");
  assert.ok(String(page.filings[0].filedAt) >= String(page.filings[79].filedAt), "newest first");
  // the default cap is not echoed into the cursor
  const dflt = readFilingHistoryWindow(parseFilingHistoryQuery({ from: "2026-11-01", to: "2026-11-10" }, archiveAt(dir, { maxFilings: 100 }), NOW_11_11), archiveAt(dir, { maxFilings: 100 }));
  assert.equal(filingHistoryNextQuery(dflt, archiveAt(dir, { maxFilings: 100 })), "?from=2026-11-01&to=2026-11-06");
});

test("readFilingHistoryWindow: walking the cursor to the end covers every filing exactly once (no gaps, no duplicates) and the last page is not truncated; gzipped days read like plain ones", () => {
  const dir = tmp("walk");
  const days = ["2026-10-26", "2026-10-27", "2026-10-28", "2026-10-29", "2026-10-30", "2026-11-02", "2026-11-03", "2026-11-04"];
  seedDays(dir, days, 30, ["2026-10-26", "2026-10-27", "2026-10-28"]);
  const a = archiveAt(dir);
  const seen = new Map<string, number>();
  let query: Record<string, unknown> = { from: "2026-10-25", to: "2026-11-05", limit: "70" };
  let pages = 0;
  for (;;) {
    const page = readFilingHistoryWindow(parseFilingHistoryQuery(query, a, NOW_11_11), a);
    pages++;
    for (const f of page.filings) seen.set(f.accession, (seen.get(f.accession) ?? 0) + 1);
    if (!page.truncated) { assert.equal(page.next_to, null); break; }
    const next = filingHistoryNextQuery(page, a)!;
    assert.match(next, /^\?from=2026-10-25&to=2026-\d\d-\d\d&limit=70$/);
    query = Object.fromEntries(new URLSearchParams(next.slice(1)));
    assert.ok(pages < 20, "the walk must terminate");
  }
  assert.equal(seen.size, days.length * 30, "every filing seen");
  assert.ok([...seen.values()].every((n) => n === 1), "no filing seen twice");
  assert.equal(pages, 4, "240 filings / 70-cap with 30-row days = 2 days per page = 4 pages");
});

test("readFilingHistoryWindow: the cap is soft by at most ONE day — a newest day larger than the cap is still returned whole so the page always makes progress; an empty window is honest (0 filings, not truncated, covered = the window); duplicates within and across days collapse and the cap counts UNIQUE rows; torn lines are counted, never dropped silently", () => {
  const dir = tmp("soft");
  seedDays(dir, ["2026-11-09", "2026-11-10"], 120);
  const a = archiveAt(dir);
  const page = readFilingHistoryWindow(parseFilingHistoryQuery({ from: "2026-11-01", to: "2026-11-10", limit: "50" }, a, NOW_11_11), a);
  assert.deepEqual([page.filings.length, page.truncated, page.next_to, page.covered], [120, true, "2026-11-09", { from: "2026-11-10", to: "2026-11-10" }]);
  const empty = readFilingHistoryWindow(parseFilingHistoryQuery({ from: "2026-08-01", to: "2026-08-05" }, a, NOW_11_11), a);
  assert.deepEqual([empty.filings.length, empty.truncated, empty.next_to, empty.days_scanned, empty.covered], [0, false, null, 5, { from: "2026-08-01", to: "2026-08-05" }]);
  // dedup: the same accession appended twice on one day (restart re-seed gap) and again on the next day
  const dir2 = tmp("dup");
  fs.mkdirSync(dir2, { recursive: true });
  fs.writeFileSync(path.join(dir2, "2026-11-10.jsonl"), [mk("X-1", "2026-11-10"), mk("X-1", "2026-11-10"), mk("X-2", "2026-11-10"), mk("X-3", "2026-11-10")].map((r) => JSON.stringify(r)).join("\n") + "\n");
  fs.writeFileSync(path.join(dir2, "2026-11-09.jsonl"), [mk("X-3", "2026-11-09"), mk("X-4", "2026-11-09")].map((r) => JSON.stringify(r)).join("\n") + "\n{\"torn\n" + JSON.stringify({ noAccession: true }) + "\n");
  const b = archiveAt(dir2);
  const capped = readFilingHistoryWindow(parseFilingHistoryQuery({ from: "2026-11-09", to: "2026-11-10", limit: "3" }, b, NOW_11_11), b);
  assert.deepEqual([capped.filings.length, capped.truncated], [3, true],
    "3 UNIQUE rows across 4 lines fill a limit of 3 (the cap is measured on deduped rows, not raw lines); 11-09 adds only X-4 (X-3 was already served) -> 3+1 > 3 -> truncated at the day boundary");
  assert.deepEqual([capped.next_to, capped.malformed_lines], ["2026-11-09", 0], "the unread day's torn lines are not counted — only covered days are");
  const full = readFilingHistoryWindow(parseFilingHistoryQuery({ from: "2026-11-09", to: "2026-11-10" }, b, NOW_11_11), b);
  assert.deepEqual(full.filings.map((f) => f.accession).sort(), ["X-1", "X-2", "X-3", "X-4"]);
  assert.equal(full.malformed_lines, 2, "the torn line and the accession-less object are both counted");
  assert.equal(readFilingArchiveDay(dir2, "2026-01-01").rows.length, 0, "a missing day is an empty day");
});

test("readFilingHistoryWindow: `project` is applied per row before dedup/cap (13F's trimHoldings use) and must see every archived row", () => {
  const dir = tmp("proj");
  seedDays(dir, ["2026-11-10"], 3);
  const a = archiveAt(dir, { project: (r) => ({ ...r, ticker: r.ticker.toLowerCase() }) });
  const page = readFilingHistoryWindow(parseFilingHistoryQuery({ days: "2" }, a, NOW_11_11), a);
  assert.deepEqual(page.filings.map((f) => f.ticker), ["tst", "tst", "tst"]);
});

test("buildFilingHistoryPage: the live cache is merged on top only when the page ends TODAY, deduped by accession and projected; a past window never gains today's rows; `count` is the true page count and `days` echoes only a legacy lookback", () => {
  const dir = tmp("live");
  seedDays(dir, ["2026-11-10", "2026-11-11"], 2);
  const a = archiveAt(dir, { project: (r) => ({ ...r, ticker: r.ticker.toUpperCase() }) });
  const live = [mk("2026-11-11-1", "2026-11-11T13:00:00", "dup"), mk("LIVE-NEW", "2026-11-11T12:59:00", "live")];
  const today = buildFilingHistoryPage({ days: "5" }, a, live, NOW_11_11);
  assert.equal(today.count, 5, "4 archived + 1 live (the live duplicate of an archived accession is dropped)");
  assert.equal(today.filings[0].accession, "LIVE-NEW", "live rows lead");
  assert.equal(today.filings[0].ticker, "LIVE", "live rows pass through the same projection");
  assert.deepEqual([today.days, today.truncated, today.next, today.archive_start, today.window], [5, false, null, "2026-07-04", { from: "2026-11-07", to: "2026-11-11" }]);
  const past = buildFilingHistoryPage({ from: "2026-11-01", to: "2026-11-10" }, a, live, NOW_11_11);
  assert.equal(past.count, 2, "a page ending yesterday does not gain today's live rows");
  assert.equal(past.days, null);
  assert.ok(!past.filings.some((f) => f.accession === "LIVE-NEW"));
  const noLive = buildFilingHistoryPage({ days: "5" }, a, null, NOW_11_11);
  assert.equal(noLive.count, 4);
  // the response carries every field of the contract
  for (const k of ["days", "window", "covered", "truncated", "next", "max_filings", "days_scanned", "archive_start", "malformed_lines", "count", "filings"]) {
    assert.ok(k in today, `response must carry ${k}`);
  }
});

test("REGRESSION (live 2026-11-11): a season-heavy archive asked for `days=90` is answered with the cap's worth of WHOLE days, truncated=true, covered naming only the days read, and next continuing from the first unread day — never 2,000 rows labelled 90 days with no marker", () => {
  const dir = tmp("season");
  // production shape scaled 1/10: 11-10 614, 11-09 465, 11-06 448, 11-05 371, 11-04 300, 11-03 250 rows at a 2,000 cap
  const counts: Record<string, number> = { "2026-11-10": 61, "2026-11-09": 47, "2026-11-06": 45, "2026-11-05": 37, "2026-11-04": 30, "2026-11-03": 25 };
  for (const [day, n] of Object.entries(counts)) seedDays(dir, [day], n);
  const a = archiveAt(dir, { maxFilings: 200 });
  const page = buildFilingHistoryPage({ days: "90" }, a, null, NOW_11_11);
  assert.equal(page.days, 90, "the consumer's own parameter is echoed...");
  assert.equal(page.truncated, true, "...but the page SAYS it did not cover it");
  assert.deepEqual(page.covered, { from: "2026-11-05", to: "2026-11-11" }, "61+47+45+37 = 190 fit; 11-04's 30 would make 220 > 200");
  assert.equal(page.count, 190);
  assert.equal(page.next, "?from=2026-08-14&to=2026-11-04", "the cursor continues from the first unread day down to the window's own start");
  // and the walk actually reaches the oldest archived day
  const rest = buildFilingHistoryPage(Object.fromEntries(new URLSearchParams(page.next!.slice(1))), a, null, NOW_11_11);
  assert.deepEqual([rest.count, rest.truncated, rest.covered.from], [55, false, "2026-08-14"]);
});
