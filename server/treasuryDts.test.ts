// Treasury DTS battery (BUILD ORDER 6 #2): FiscalData envelope parse,
// day boundary discipline, "null"-string honesty, day-level dedup,
// bracket-encoding pin, restart rebuild from archive.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  parseDts, fetchLatestDts, archiveDtsDay, gzipOldDtsDays,
  refreshDts, latestDts, readArchivedDtsDay, DTS_FETCH_LIMIT,
  sumTgaDepositsExDebt, PUBLIC_DEBT_CASH_ISSUES_CATEGORY, DtsRow,
} from "./treasuryDts";

// Mirrors the live FiscalData shape verified 2026-07-06 (amounts are
// STRINGS in $ millions; transaction_catg_desc is the literal string
// "null" on current-era rows).
const ROW = (date: string, ttype: string, catg: string, today: string) => ({
  record_date: date,
  account_type: "Treasury General Account (TGA)",
  transaction_type: ttype,
  transaction_catg: catg,
  transaction_catg_desc: "null",
  transaction_today_amt: today,
  transaction_mtd_amt: "19",
  transaction_fytd_amt: "3360",
  table_nbr: "II",
  table_nm: "Deposits and Withdrawals of Operating Cash",
  src_line_nbr: "1",
});

const ENVELOPE = (rows: any[]) => ({ data: rows, meta: { count: rows.length } });

test("parseDts: fields land, string amounts become numbers, ''/'null' stay null", () => {
  const rows = parseDts(ENVELOPE([
    ROW("2026-07-02", "Deposits", "Taxes - Withheld Individual/FICA", "12873"),
    ROW("2026-07-02", "Withdrawals", "Dept of Defense (DoD) - misc", ""),
  ]), "2026-07-06");
  assert.equal(rows.length, 2);
  assert.equal(rows[0].record_date, "2026-07-02");
  assert.equal(rows[0].transaction_type, "Deposits");
  assert.equal(rows[0].category, "Taxes - Withheld Individual/FICA");
  assert.equal(rows[0].today_amt, 12873);
  assert.equal(rows[1].today_amt, null, "empty string is null, never zero");
  assert.deepEqual(parseDts(ENVELOPE([]), "x"), []);
  assert.deepEqual(parseDts(null, "x"), []);
  assert.deepEqual(parseDts({ data: [{ no_fields: 1 }] }, "x"), []);
});

test("parseDts: a DESC batch straddling two business days keeps ONLY the newest", () => {
  const rows = parseDts(ENVELOPE([
    ROW("2026-07-02", "Deposits", "NEW DAY CATG", "1"),
    ROW("2026-07-01", "Deposits", "OLD DAY CATG", "9"),
    ROW("2026-07-02", "Withdrawals", "NEW DAY CATG 2", "3"),
  ]), "2026-07-06");
  assert.equal(rows.length, 2);
  assert.ok(rows.every((r) => r.record_date === "2026-07-02"), "statements never mix in one file");
});

test("fetchLatestDts: non-200 returns []; url has ENCODED brackets + DESC sort", async () => {
  const urls: string[] = [];
  const bad = async (url: string) => { urls.push(url); return { ok: false, status: 500, text: async () => "" }; };
  assert.deepEqual(await fetchLatestDts(bad as any), []);
  assert.ok(urls[0].includes(`page%5Bsize%5D=${DTS_FETCH_LIMIT}`),
    "raw [] 400s on this API — the encoding is load-bearing");
  assert.ok(!urls[0].includes("page[size]"));
  assert.ok(urls[0].includes("sort=-record_date"));
});

test("archive: day-level dedup + gz after margin + gz readback", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "dts-"));
  const rows = parseDts(ENVELOPE([ROW("2026-07-02", "Deposits", "Taxes - Withheld Individual/FICA", "12873")]), "2026-07-06");
  assert.equal(archiveDtsDay(rows, base), 1);
  assert.equal(archiveDtsDay(rows, base), 0, "same statement day never re-archives");
  assert.equal(gzipOldDtsDays(base, Date.parse("2026-07-04T00:00:00Z")), 0, "within 4d stays plain");
  assert.equal(gzipOldDtsDays(base, Date.parse("2026-07-07T00:00:00Z")), 1);
  assert.equal(readArchivedDtsDay("2026-07-02", base).length, 1, "gz day still readable");
});

test("refresh: restart with fetch down rebuilds cache from the newest archived day", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "dts-"));
  archiveDtsDay(parseDts(ENVELOPE([ROW("2026-07-03", "Deposits", "RESTART CATG", "5")]), "2026-07-06"), base);
  const dead = async () => ({ ok: false, status: 503, text: async () => "" });
  await refreshDts(dead as any, Date.parse("2026-07-06T12:00:00Z"), base);
  const hit = latestDts();
  assert.ok(hit, "cache rebuilt from archive despite fetch being down");
  assert.equal(hit!.record_date, "2026-07-03");
  assert.equal(hit!.rows[0].category, "RESTART CATG");
});

// sumTgaDepositsExDebt (GATE 1 — scripts/treasury_dts_gate1.ts)

const dtsRow = (over: Partial<DtsRow>): DtsRow => ({
  record_date: "2026-07-31",
  account_type: "Treasury General Account (TGA)",
  transaction_type: "Deposits",
  category: "Taxes - Withheld Individual/FICA",
  today_amt: 1,
  mtd_amt: 100,
  fytd_amt: 1000,
  src_line: 1,
  rt: "2026-07-31",
  ...over,
});

test("sumTgaDepositsExDebt: excludes the API's own Total-Deposits subtotal row (avoids the 2x double-count)", () => {
  const rows = [
    dtsRow({ category: "Taxes - Withheld Individual/FICA", mtd_amt: 279940 }),
    dtsRow({ category: "Taxes - Corporate Income", mtd_amt: 19206 }),
    dtsRow({ account_type: "Treasury General Account Total Deposits", category: "null" as any, mtd_amt: 299146 }),
  ];
  assert.equal(sumTgaDepositsExDebt(rows), 279940 + 19206, "the subtotal row must not be added on top of its own parts");
});

test("sumTgaDepositsExDebt: excludes Public Debt Cash Issues (financing, not a receipt)", () => {
  const rows = [
    dtsRow({ category: "Taxes - Withheld Individual/FICA", mtd_amt: 279940 }),
    dtsRow({ category: PUBLIC_DEBT_CASH_ISSUES_CATEGORY, mtd_amt: 3022505 }),
  ];
  assert.equal(sumTgaDepositsExDebt(rows), 279940);
});

test("sumTgaDepositsExDebt: ignores Withdrawals rows and treats a null mtd_amt as 0, never NaN", () => {
  const rows = [
    dtsRow({ category: "Taxes - Corporate Income", mtd_amt: 19206 }),
    dtsRow({ transaction_type: "Withdrawals", category: "SSA - Benefits Payments", mtd_amt: 124463 }),
    dtsRow({ category: "Unclassified - Deposits", mtd_amt: null }),
  ];
  assert.equal(sumTgaDepositsExDebt(rows), 19206);
});

test("sumTgaDepositsExDebt: empty input sums to 0", () => {
  assert.equal(sumTgaDepositsExDebt([]), 0);
});
