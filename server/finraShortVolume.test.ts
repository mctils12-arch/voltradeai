// FINRA short-volume battery (BUILD ORDER 5 #1): real-format parse with
// trailer integrity, holiday no-op, date-level dedup, summary honesty,
// restart-rebuild from archive.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  parseShortVol, fetchShortVolDay, archiveShortVolDay, gzipOldShortVolDays,
  gzipOldShortVolDaysAsync, summarize, refreshShortVol, latestShortVol,
  readArchivedDay, deepBackfillIfSparse, countArchivedDays,
  TOP_CAP, FLOOR_TOTAL_VOL, DEEP_BACKFILL_DAYS,
  appendSummaryHistoryEntry, readSummaryHistory, listArchivedDates, lookupSymbolHistory,
  fetchOrfShortVolDay, archiveOrfShortVolDay, readOrfArchivedDay, refreshOrfShortVol,
} from "./finraShortVolume";

// Mirrors the live file verified 2026-07-05: pipe header, fractional
// share counts, comma-joined market codes, bare row-count trailer.
const FILE = [
  "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market",
  "20260702|AA|2551427.692872|67707.208540|4489378.153460|B,Q,N",
  "20260702|ZZTOP|900000|0|1000000|Q",
  "20260702|TINY|301|0|671.228446|Q",
  "3",
].join("\n");

test("parseShortVol: real format, ISO dates, fractional floats, trailer ok", () => {
  const rows = parseShortVol(FILE, "2026-07-05");
  assert.equal(rows.length, 3);
  assert.equal(rows[0].date, "2026-07-02");
  assert.equal(rows[0].symbol, "AA");
  assert.ok(Math.abs(rows[0].short_vol - 2551427.692872) < 1e-6);
  assert.equal(rows[0].market, "B,Q,N");
  assert.equal(rows[1].short_exempt_vol, 0);
});

test("parseShortVol: trailer mismatch refuses the file (truncation guard)", () => {
  const truncated = FILE.split("\n").slice(0, 3).concat(["3"]).join("\n"); // says 3, has 2
  assert.deepEqual(parseShortVol(truncated, "x"), []);
  assert.deepEqual(parseShortVol("", "x"), []);
  assert.deepEqual(parseShortVol("Nope|Header\n1|2", "x"), [], "unknown header = empty");
});

test("fetchShortVolDay: 403 is a valid non-trading day ([]), 500 is a transport error (null)", async () => {
  const mk = (status: number, body = "") => async () =>
    ({ ok: status === 200, status, text: async () => body });
  assert.deepEqual(await fetchShortVolDay("20260704", mk(403) as any), []);
  assert.equal(await fetchShortVolDay("20260702", mk(500) as any), null);
  const rows = await fetchShortVolDay("20260702", mk(200, FILE) as any);
  assert.equal(rows!.length, 3);
});

test("archive: date-level dedup + gz lifecycle", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "finra-"));
  const rows = parseShortVol(FILE, "2026-07-05");
  assert.equal(archiveShortVolDay(rows, base), 3);
  assert.equal(archiveShortVolDay(rows, base), 0, "same trade date never re-archives");
  assert.equal(readArchivedDay("2026-07-02", base).length, 3);
  assert.equal(gzipOldShortVolDays(base, Date.parse("2026-07-05T12:00:00Z")), 1);
  const day = path.join(base, "finrashortvol", "2026-07-02.jsonl");
  assert.ok(!fs.existsSync(day) && fs.existsSync(`${day}.gz`));
  assert.equal(readArchivedDay("2026-07-02", base).length, 3, "gz day still readable");
});

test("summarize: agg ratio + stated floor/cap honesty", () => {
  const rows = parseShortVol(FILE, "2026-07-05");
  const s = summarize(rows)!;
  assert.equal(s.date, "2026-07-02");
  assert.equal(s.symbols, 3);
  // TINY (671 total) is below the stated floor and must be excluded
  assert.ok(s.top_ratio.every((t) => t.symbol !== "TINY"));
  assert.equal(s.floor_total_vol, FLOOR_TOTAL_VOL);
  assert.equal(s.top_cap, TOP_CAP);
  // ZZTOP 0.9 ratio sorts above AA (~0.568)
  assert.equal(s.top_ratio[0].symbol, "ZZTOP");
  assert.ok(s.agg_short_ratio! > 0 && s.agg_short_ratio! < 1);
  assert.equal(summarize([]), null);
});

test("deep backfill: marker completes it, interruption resumes, done-marker short-circuits", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "finra-bf-"));
  const now = Date.parse("2026-07-05T12:00:00Z");
  assert.ok(DEEP_BACKFILL_DAYS >= 700, "target ~2 years of trading days");
  // serve files for 3 specific dates; everything else 403 (non-trading).
  // Dates deliberately unused by other tests: archivedDates is module
  // state shared across this file's tests (date-level dedup by design).
  const SERVED = new Set(["20260615", "20260616", "20260617"]);
  let calls = 0;
  const fake = async (url: string) => {
    calls++;
    const m = url.match(/CNMSshvol(\d{8})\.txt/);
    if (!m || !SERVED.has(m[1])) return { ok: false, status: 403, text: async () => "" };
    const day = m[1];
    return { ok: true, status: 200, text: async () =>
      `Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n${day}|AA|100|0|200|Q\n1` };
  };
  const ON = { FINRA_DEEP_BACKFILL: "1" } as any;
  // EMERGENCY DEFAULT-OFF (crash-loop postmortem): keyless env = no-op
  await deepBackfillIfSparse(fake as any, now, base, {} as any);
  assert.equal(calls, 0, "default-off: not a single fetch without the env opt-in");
  await deepBackfillIfSparse(fake as any, now, base, ON);
  assert.equal(countArchivedDays(base), 3, "all served days archived");
  assert.ok(fs.existsSync(path.join(base, "finrashortvol", "backfill_done.json")), "completion marker written");
  // gz-on-write: backfilled days older than 2d land compressed, never plain
  assert.ok(fs.existsSync(path.join(base, "finrashortvol", "2026-06-15.jsonl.gz")),
    "deep-pass day is gzipped IMMEDIATELY (volume-flood guard)");
  assert.ok(!fs.existsSync(path.join(base, "finrashortvol", "2026-06-15.jsonl")));
  const callsAfterDone = calls;
  await deepBackfillIfSparse(fake as any, now, base, ON);
  assert.equal(calls, callsAfterDone, "done marker short-circuits — zero refetches");
  // interruption semantics: remove the marker -> re-run resumes via dedup
  fs.unlinkSync(path.join(base, "finrashortvol", "backfill_done.json"));
  await deepBackfillIfSparse(fake as any, now, base, ON);
  assert.equal(countArchivedDays(base), 3, "resume run added nothing new (dedup)");
  assert.ok(fs.existsSync(path.join(base, "finrashortvol", "backfill_done.json")), "marker rewritten");
});

test("gzip async variant matches sync semantics", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "finra-gz-"));
  const rows = parseShortVol(FILE.replace(/20260702/g, "20260628"), "2026-07-05");
  archiveShortVolDay(rows, base);
  assert.equal(await gzipOldShortVolDaysAsync(base, Date.parse("2026-07-05T12:00:00Z")), 1);
  assert.ok(fs.existsSync(path.join(base, "finrashortvol", "2026-06-28.jsonl.gz")));
});

test("summary history: append is idempotent per date, reads ascending, bounded to `days`", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "finra-trend-"));
  appendSummaryHistoryEntry({ date: "2026-07-01", symbols: 100, agg_short_ratio: 0.40 } as any, base);
  appendSummaryHistoryEntry({ date: "2026-07-02", symbols: 101, agg_short_ratio: 0.41 } as any, base);
  appendSummaryHistoryEntry({ date: "2026-07-01", symbols: 999, agg_short_ratio: 0.99 } as any, base);
  const all = readSummaryHistory(30, base);
  assert.equal(all.length, 2, "duplicate date did not append a second line");
  assert.deepEqual(all.map((r) => r.date), ["2026-07-01", "2026-07-02"], "ascending by date");
  assert.equal(all[0].agg_short_ratio, 0.40, "first-written value kept, not overwritten");
  assert.equal(readSummaryHistory(1, base).length, 1, "days cap slices to the most recent N");
  assert.deepEqual(readSummaryHistory(5, path.join(base, "nonexistent")), [], "missing file = empty, not a throw");
});

test("listArchivedDates + lookupSymbolHistory: newest-first listing, honest gap on absent symbol-day, bounded by days", () => {
  // NOTE: archivedDates dedup is keyed by date ONLY, shared module state
  // across every test in this file regardless of baseDir (see the deep
  // backfill test's comment above) — these dates must be unused elsewhere.
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "finra-sym-"));
  const mk = (day: string, rows: string) =>
    `Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n${rows}\n${rows.split("\n").length}`;
  archiveShortVolDay(parseShortVol(mk("20260501", `20260501|ZZTOP|400|0|1000|Q`), "x"), base);
  archiveShortVolDay(parseShortVol(mk("20260502", `20260502|AA|100|0|200|Q`), "x"), base); // no ZZTOP this day
  archiveShortVolDay(parseShortVol(mk("20260503", `20260503|zztop|900|0|1000|Q`), "x"), base); // lowercase symbol
  assert.deepEqual(listArchivedDates(base, 90), ["2026-05-03", "2026-05-02", "2026-05-01"], "newest first");
  assert.deepEqual(listArchivedDates(base, 2), ["2026-05-03", "2026-05-02"], "limit respected");
  const series = lookupSymbolHistory("ZZTOP", 90, base);
  assert.deepEqual(series.map((r) => r.date), ["2026-05-01", "2026-05-03"], "05-02 honestly omitted, not zero-filled");
  assert.equal(series[0].short_ratio, 0.4);
  assert.equal(series[1].short_ratio, 0.9, "case-insensitive symbol match");
  assert.deepEqual(lookupSymbolHistory("NOPE", 90, base), []);
});

test("refresh: wires the market-wide trend log (bootstrap + restart both append exactly once)", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "finra-trendwire-"));
  const fake = async (url: string) => {
    const m = url.match(/CNMSshvol(\d{8})\.txt/);
    if (m && m[1] === "20260706") {
      return { ok: true, status: 200, text: async () =>
        "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n20260706|AA|100|0|200|Q\n1" };
    }
    return { ok: false, status: 403, text: async () => "" };
  };
  await refreshShortVol(fake as any, Date.parse("2026-07-06T23:00:00Z"), 1, base);
  assert.deepEqual(readSummaryHistory(30, base).map((r) => r.date), ["2026-07-06"]);
  // restart path (day already archived) must not double-append the same date
  await refreshShortVol(async () => ({ ok: false, status: 403, text: async () => "" }) as any,
    Date.parse("2026-07-06T23:30:00Z"), 1, base);
  assert.equal(readSummaryHistory(30, base).length, 1, "restart rebuild-from-archive did not duplicate the trend entry");
});

test("refresh: restart with the newest day already archived rebuilds cache from disk, no refetch", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "finra-"));
  archiveShortVolDay(parseShortVol(FILE.replace(/20260702/g, "20260705"), "2026-07-05")
    .map((r) => ({ ...r })), base);
  const fetched: string[] = [];
  const fake = async (url: string) => {
    fetched.push(url);
    return { ok: false, status: 403, text: async () => "" };
  };
  await refreshShortVol(fake as any, Date.parse("2026-07-05T23:00:00Z"), 3, base);
  const hit = latestShortVol();
  assert.ok(hit, "cache rebuilt from the archived day");
  assert.equal(hit!.summary.date, "2026-07-05");
  assert.ok(!fetched.some((u) => u.includes("20260705")), "archived day was not refetched");
});

// ── ORF (OTC facility) — added 2026-07-27 to fix settlementStress.ts's
// permanent-zero-overlap bug (finrathreshold is OTC-only; CNMS is
// exchange-listed-only). Same format as CNMS (parseShortVol handles
// both); this battery covers the ORF-specific wiring only. Same dedup
// caveat as CNMS above: orfArchivedDates has no reset export, so every
// date used across THIS WHOLE FILE for archiveOrfShortVolDay/
// refreshOrfShortVol must be globally unique regardless of baseDir. ────

const ORF_FILE = [
  "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market",
  "20260624|CHLSY|27404|0|60932|O", // real values, live-verified 2026-07-27
  "1",
].join("\n");

test("fetchOrfShortVolDay: same 403/404/500 contract as CNMS, reuses parseShortVol", async () => {
  const mk = (status: number, body = "") => async () =>
    ({ ok: status === 200, status, text: async () => body });
  assert.deepEqual(await fetchOrfShortVolDay("20260628", mk(403) as any), []);
  assert.equal(await fetchOrfShortVolDay("20260624", mk(500) as any), null);
  const rows = await fetchOrfShortVolDay("20260624", mk(200, ORF_FILE) as any);
  assert.equal(rows!.length, 1);
  assert.equal(rows![0].symbol, "CHLSY");
  assert.equal(rows![0].market, "O");
});

test("archiveOrfShortVolDay + readOrfArchivedDay: date-level dedup, separate namespace from CNMS", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "finra-orf-"));
  const rows = parseShortVol(ORF_FILE, "2026-07-27");
  assert.equal(archiveOrfShortVolDay(rows, base), 1);
  assert.equal(archiveOrfShortVolDay(rows, base), 0, "same trade date never re-archives");
  assert.deepEqual(readOrfArchivedDay("2026-06-24", base).map((r) => r.symbol), ["CHLSY"]);

  // Same trade date can be archived independently in CNMS's own dir —
  // the two facilities must never collide (this is the whole point of
  // the fix: they cover disjoint symbol populations, stored separately).
  assert.equal(archiveShortVolDay(
    [{ date: "2026-06-24", symbol: "AA", short_vol: 100, short_exempt_vol: 0, total_vol: 200, market: "Q", rt: "r" }] as ShortVolRow[],
    base,
  ), 1);
  assert.deepEqual(readArchivedDay("2026-06-24", base).map((r) => r.symbol), ["AA"]);
  assert.deepEqual(readOrfArchivedDay("2026-06-24", base).map((r) => r.symbol), ["CHLSY"], "CNMS write did not touch the ORF archive");
});

test("refreshOrfShortVol: fetches only unarchived days in the lookback window, archives newest-first", async () => {
  // Date must be globally unique across this whole file — orfArchivedDates
  // has no reset export, same documented constraint as CNMS's archivedDates
  // (see finraShortVolume.test.ts's own precedent in settlementStress.test.ts).
  const day = "2026-05-11";
  const orfFileForDay = ORF_FILE.replace(/20260624/, day.replace(/-/g, ""));
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "finra-orf-refresh-"));
  const fetched: string[] = [];
  const fake = async (url: string) => {
    fetched.push(url);
    const m = url.match(/FORFshvol(\d{8})\.txt/);
    if (m && m[1] === day.replace(/-/g, "")) return { ok: true, status: 200, text: async () => orfFileForDay };
    return { ok: false, status: 403, text: async () => "" };
  };
  await refreshOrfShortVol(fake as any, Date.parse(`${day}T23:00:00Z`), 2, base);
  assert.deepEqual(readOrfArchivedDay(day, base).map((r) => r.symbol), ["CHLSY"]);
  assert.ok(fetched.some((u) => u.includes(`FORFshvol${day.replace(/-/g, "")}.txt`)), "hit the ORF URL, not CNMS");

  // A second pass over the same window must not refetch the already-archived day.
  const fetched2: string[] = [];
  await refreshOrfShortVol(async (url: string) => { fetched2.push(url); return { ok: false, status: 403, text: async () => "" }; },
    Date.parse(`${day}T23:30:00Z`), 2, base);
  assert.ok(!fetched2.some((u) => u.includes(day.replace(/-/g, ""))), "already-archived day is skipped, not refetched");
});
