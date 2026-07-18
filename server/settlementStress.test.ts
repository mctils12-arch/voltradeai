// SETTLEMENT-STRESS COMPOSITE battery: join correctness (persistence,
// FTD delta, short-vol percentile), gate-1-only composite scoring, and
// archive dedup/refresh behavior. Fixtures use the real ingredient
// archivers (archivePartition/archiveShortVolDay/archiveFtdPeriod) so the
// join is tested against the actual on-disk contract, not a hand-rolled
// approximation of it.
import { test, beforeEach } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { archivePartition, _resetFinraQueryForTests } from "./finraQuery";
import { archiveShortVolDay, type ShortVolRow } from "./finraShortVolume";
import { archiveFtdPeriod, _resetFtdForTests, type FtdRow } from "./secFtd";
import {
  thresholdPersistence, periodForDate, prevPeriod, ftdDeltas,
  shortVolPercentiles, computeComposite, archiveComposite, isCompositeArchived,
  refreshSettlementStress, listThresholdDates, _resetSettlementStressForTests,
} from "./settlementStress";

// finrathreshold/secftd/settlementstress dedup state is reset between
// tests (reset helpers exist for those three). finrashortvol has no reset
// export, so — same precedent as finraShortVolume.test.ts's own comment —
// every date passed to archiveShortVolDay across this whole file must be
// globally unique, regardless of baseDir.
beforeEach(() => {
  _resetFinraQueryForTests();
  _resetFtdForTests();
  _resetSettlementStressForTests();
});

function tmpBase(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), "settlestress-"));
}

function thresholdRow(symbol: string, name: string, tradeDate: string) {
  return { tradeDate, issueSymbolIdentifier: symbol, issueName: name, marketCategoryDescription: "OTC" };
}

test("periodForDate / prevPeriod: half-month boundary + month/year rollover", () => {
  assert.equal(periodForDate("2026-07-14"), "202607a");
  assert.equal(periodForDate("2026-07-15"), "202607b");
  assert.equal(periodForDate("2026-07-01"), "202607a");
  assert.equal(periodForDate("2026-07-31"), "202607b");
  assert.equal(prevPeriod("202607b"), "202607a");
  assert.equal(prevPeriod("202607a"), "202606b");
  assert.equal(prevPeriod("202601a"), "202512b");
});

test("thresholdPersistence: counts a consecutive-archived-day streak, breaks on gap", () => {
  const base = tmpBase();
  archivePartition("finrathreshold", "2026-07-01", [thresholdRow("AAA", "Alpha Co", "2026-07-01")], "2026-07-01", base);
  archivePartition("finrathreshold", "2026-07-02", [thresholdRow("AAA", "Alpha Co", "2026-07-02")], "2026-07-02", base);
  archivePartition("finrathreshold", "2026-07-03", [
    thresholdRow("AAA", "Alpha Co", "2026-07-03"),
    thresholdRow("BBB", "Beta Co", "2026-07-03"),
  ], "2026-07-03", base);

  assert.deepEqual(listThresholdDates(base), ["2026-07-01", "2026-07-02", "2026-07-03"]);
  const p = thresholdPersistence("2026-07-03", base);
  assert.equal(p.get("AAA")?.persistence_days, 3);
  assert.equal(p.get("AAA")?.name, "Alpha Co");
  assert.equal(p.get("BBB")?.persistence_days, 1);
  assert.equal(thresholdPersistence("2026-07-04", base).size, 0, "no data archived for that date");
});

test("ftdDeltas: newest-date-per-period balance, delta vs. prior period, missing prior = delta from 0", () => {
  const base = tmpBase();
  const prevRows: FtdRow[] = [
    { date: "2026-06-20", cusip: "X1", symbol: "AAA", qty: 100_000, name: "Alpha Co", price: 5, rt: "r" },
  ];
  const curRows: FtdRow[] = [
    { date: "2026-07-01", cusip: "X1", symbol: "AAA", qty: 130_000, name: "Alpha Co", price: 5, rt: "r" },
    { date: "2026-07-10", cusip: "X1", symbol: "AAA", qty: 150_000, name: "Alpha Co", price: 5, rt: "r" }, // newest wins
    { date: "2026-07-05", cusip: "X2", symbol: "CCC", qty: 40_000, name: "Gamma Co", price: 2, rt: "r" },
  ];
  archiveFtdPeriod("202606b", prevRows, base);
  archiveFtdPeriod("202607a", curRows, base);

  const d = ftdDeltas("2026-07-05", base); // any date in 202607a resolves to the same period
  assert.equal(d.get("AAA")?.qty, 150_000);
  assert.equal(d.get("AAA")?.delta, 50_000);
  assert.equal(d.get("CCC")?.qty, 40_000);
  assert.equal(d.get("CCC")?.delta, 40_000, "no prior-period record = delta from a 0 balance");
  assert.equal(ftdDeltas("2026-08-01", base).size, 0, "period not archived yet = empty, never interpolated");
});

test("shortVolPercentiles: liquidity floor excludes illiquid rows, ranks ascending ratio", () => {
  const base = tmpBase();
  const rows: ShortVolRow[] = [
    { date: "2026-07-03", symbol: "AAA", short_vol: 100_000, short_exempt_vol: 0, total_vol: 1_000_000, market: "Q", rt: "r" },
    { date: "2026-07-03", symbol: "BBB", short_vol: 800_000, short_exempt_vol: 0, total_vol: 1_000_000, market: "Q", rt: "r" },
    { date: "2026-07-03", symbol: "TINY", short_vol: 10, short_exempt_vol: 0, total_vol: 100, market: "Q", rt: "r" }, // below floor
  ];
  archiveShortVolDay(rows, base);
  const pct = shortVolPercentiles("2026-07-03", base);
  assert.equal(pct.size, 2, "illiquid row excluded");
  assert.equal(pct.get("BBB")?.percentile, 100, "highest ratio = 100th percentile");
  assert.equal(pct.get("AAA")?.percentile, 0, "lowest ratio = 0th percentile");
  assert.ok(!pct.has("TINY"));
});

test("computeComposite: gate-1 join keeps only the 3-way intersection; empty when any ingredient missing", () => {
  const base = tmpBase();
  const date = "2026-07-10";
  archivePartition("finrathreshold", date, [
    thresholdRow("AAA", "Alpha Co", date),
    thresholdRow("ZZZ", "Zeta Co", date), // on threshold list but missing from the other two archives
  ], date, base);
  archiveShortVolDay([
    { date, symbol: "AAA", short_vol: 500_000, short_exempt_vol: 0, total_vol: 1_000_000, market: "Q", rt: date },
  ] as ShortVolRow[], base);
  archiveFtdPeriod(periodForDate(date), [
    { date, cusip: "X1", symbol: "AAA", qty: 200_000, name: "Alpha Co", price: 5, rt: date },
  ] as FtdRow[], base);

  const rows = computeComposite(date, base);
  assert.equal(rows.length, 1, "only the symbol present in all three archives is scored");
  assert.equal(rows[0].symbol, "AAA");
  assert.equal(rows[0].persistence_days, 1);
  assert.equal(rows[0].ftd_delta, 200_000);
  assert.ok(rows[0].composite_score > 0, "positive FTD delta -> positive composite sign");

  assert.deepEqual(computeComposite("2026-01-01", base), [], "no threshold data for that date");
});

test("archiveComposite + isCompositeArchived: date-level dedup, empty result still recorded honestly", () => {
  const base = tmpBase();
  assert.equal(isCompositeArchived("2026-07-10", base), false);
  assert.equal(archiveComposite("2026-07-10", [], base), 0);
  assert.equal(isCompositeArchived("2026-07-10", base), true, "an empty-overlap day is still a recorded finding");
  assert.equal(archiveComposite("2026-07-10", [{ date: "2026-07-10" } as any], base), 0, "re-archiving a done date is a no-op");
});

test("refreshSettlementStress: only advances dates whose short-vol + FTD ingredients are already archived; idempotent", () => {
  const base = tmpBase();
  const d1 = "2026-08-10", d2 = "2026-08-11"; // distinct from computeComposite test's shortvol date (no reset for that namespace)
  archivePartition("finrathreshold", d1, [thresholdRow("AAA", "Alpha Co", d1)], d1, base);
  archivePartition("finrathreshold", d2, [thresholdRow("AAA", "Alpha Co", d2)], d2, base);
  archiveShortVolDay([{ date: d1, symbol: "AAA", short_vol: 500_000, short_exempt_vol: 0, total_vol: 1_000_000, market: "Q", rt: d1 }] as ShortVolRow[], base);
  archiveFtdPeriod(periodForDate(d1), [{ date: d1, cusip: "X1", symbol: "AAA", qty: 200_000, name: "Alpha Co", price: 5, rt: d1 }] as FtdRow[], base);
  // d2's short-vol/FTD ingredients are deliberately NOT archived yet.

  const first = refreshSettlementStress(base);
  assert.deepEqual(first, [d1], "d2 skipped — its ingredients haven't landed yet");
  assert.equal(isCompositeArchived(d1, base), true);
  assert.equal(isCompositeArchived(d2, base), false);

  const second = refreshSettlementStress(base);
  assert.deepEqual(second, [], "d1 already archived — never recomputed");
});
