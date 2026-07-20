// FDIC bank-financials quarterly-snapshot battery (BUILD ORDER 6 #3 v2):
// nested envelope parse, REPDTE normalization, paginated fetch (stable
// CERT sort, page-limit cap, offset stepping to total), immutable
// once-per-quarter archive dedup, host + fields pin.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  parseFinancials, normalizeRepdte, fetchLatestRepdte, fetchFinancialsSnapshot,
  snapshotArchived, archiveSnapshot, readArchivedCount, readArchivedSnapshot, refreshFinancials,
  latestFinancialsMeta, FINANCIALS_PAGE_LIMIT,
} from "./fdicBankFinancials";

// Mirrors the live FDIC envelope verified 2026-07-20: rows nested under
// data[].data, REPDTE as YYYYMMDD, amounts in $ thousands.
const WRAP = (cert: number, name: string, repdte: string, extra: Record<string, any> = {}) => ({
  data: {
    CERT: cert, NAME: name, REPDTE: repdte, STALP: "WI", CITY: "MARKESAN",
    ASSET: 263059, DEP: 221684, DEPDOM: 221684, DEPINS: 188440, DEPUNINS: 34085,
    EQ: 25651, NETINC: 239, ROA: 0.36, ROE: 3.71, NPERFV: 1.41,
    ...extra,
  },
  score: 1,
});

const ENVELOPE = (rows: any[], total?: number) => ({ meta: { total: total ?? rows.length }, data: rows });

test("normalizeRepdte: YYYYMMDD -> ISO; garbage -> null", () => {
  assert.equal(normalizeRepdte("20260331"), "2026-03-31");
  assert.equal(normalizeRepdte("20251231"), "2025-12-31");
  assert.equal(normalizeRepdte("2026-03-31"), null, "already-ISO is not this API's shape");
  assert.equal(normalizeRepdte(""), null);
  assert.equal(normalizeRepdte(null), null);
});

test("parseFinancials: nested envelope lands; the run-risk + asset-quality fields survive", () => {
  const rows = parseFinancials(ENVELOPE([
    WRAP(10004, "ERGO BANK", "20260331"),
    WRAP(10011, "WOODFORD STATE BANK", "20260331", { DEPUNINS: null, NPERFV: null }),
  ]), "2026-07-20");
  assert.equal(rows.length, 2);
  assert.equal(rows[0].repdte, "2026-03-31");
  assert.equal(rows[0].cert, 10004);
  assert.equal(rows[0].deposits_uninsured_k, 34085);
  assert.equal(rows[0].nonperforming_pct, 1.41);
  assert.equal(rows[1].deposits_uninsured_k, null, "absent field stays null, never coerced to 0");
  assert.deepEqual(parseFinancials(ENVELOPE([]), "x"), []);
  assert.deepEqual(parseFinancials(null, "x"), []);
  assert.deepEqual(parseFinancials(ENVELOPE([{ data: { NAME: "NO REPDTE" } }]), "x"), []);
});

test("fetchLatestRepdte: url pins api.fdic.gov host + DESC sort; non-200 -> null", async () => {
  const urls: string[] = [];
  const ok = async (url: string) => {
    urls.push(url);
    return { ok: true, status: 200, text: async () => JSON.stringify(ENVELOPE([WRAP(1, "X", "20260331")])) };
  };
  assert.equal(await fetchLatestRepdte(ok as any), "20260331");
  assert.ok(urls[0].startsWith("https://api.fdic.gov/banks/financials"),
    "host is api.fdic.gov — banks.data.fdic.gov 301s (same probe finding as fdicfailures)");
  assert.ok(urls[0].includes("sort_order=DESC") && urls[0].includes("sort_by=REPDTE"));

  const bad = async () => ({ ok: false, status: 500, text: async () => "" });
  assert.equal(await fetchLatestRepdte(bad as any), null);
});

test("fetchFinancialsSnapshot: pages until offset >= total; stable CERT sort + explicit fields on every page", async () => {
  const urls: string[] = [];
  const page1 = Array.from({ length: FINANCIALS_PAGE_LIMIT }, (_, i) => WRAP(i + 1, `BANK ${i + 1}`, "20260331"));
  const page2 = [WRAP(FINANCIALS_PAGE_LIMIT + 1, "LAST BANK", "20260331")];
  const total = page1.length + page2.length;
  const fetchImpl = async (url: string) => {
    urls.push(url);
    const body = url.includes(`offset=${FINANCIALS_PAGE_LIMIT}`) ? page2 : page1;
    return { ok: true, status: 200, text: async () => JSON.stringify(ENVELOPE(body, total)) };
  };
  const rows = await fetchFinancialsSnapshot("20260331", fetchImpl as any, Date.parse("2026-07-20T00:00:00Z"));
  assert.equal(rows.length, total);
  assert.equal(rows[rows.length - 1].name, "LAST BANK");
  assert.equal(urls.length, 2, "stops paginating once offset reaches total");
  for (const u of urls) {
    assert.ok(u.includes("sort_by=CERT") && u.includes("sort_order=ASC"));
    assert.ok(u.includes("REPDTE%3A20260331") || u.includes("REPDTE:20260331"));
    assert.ok(u.includes("NPERFV"), "explicit fields list travels on every page");
  }
});

test("fetchFinancialsSnapshot: a page short of the limit stops pagination even if total looks bigger", async () => {
  const short = [WRAP(1, "ONLY BANK", "20260331")];
  const fetchImpl = async () => ({ ok: true, status: 200, text: async () => JSON.stringify(ENVELOPE(short, 9999)) });
  const rows = await fetchFinancialsSnapshot("20260331", fetchImpl as any);
  assert.equal(rows.length, 1, "short page is itself the stop signal, independent of a stale/wrong total");
});

test("archive: quarter dedup — already-archived REPDTE is never re-fetched; immutable, gzipped immediately", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "fdicfin-"));
  const rows = parseFinancials(ENVELOPE([WRAP(1, "BANK A", "20260331"), WRAP(2, "BANK B", "20260331")]), "2026-07-20");
  assert.equal(snapshotArchived("2026-03-31", base), false);
  assert.equal(archiveSnapshot("2026-03-31", rows, base), 2);
  assert.equal(snapshotArchived("2026-03-31", base), true);
  assert.ok(fs.existsSync(path.join(base, "fdicbankfinancials", "2026-03-31.jsonl.gz")),
    "gzipped immediately — a published quarter never gets appended to again");
  assert.equal(readArchivedCount("2026-03-31", base), 2);
  assert.equal(archiveSnapshot("2026-03-31", rows, base), 2, "archiveSnapshot itself is idempotent-safe to call again");
  const readBack = readArchivedSnapshot("2026-03-31", base);
  assert.equal(readBack.length, 2);
  assert.deepEqual(readBack[0], rows[0], "round-trips the exact typed record, not the raw FDIC envelope");
  assert.deepEqual(readArchivedSnapshot("1999-01-01", base), [], "absent quarter -> empty, not a throw");
});

test("refreshFinancials: new quarter fetches + archives once; repeat call for the same quarter skips the paginated pull", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "fdicfin-"));
  let snapshotCalls = 0;
  const fetchImpl = async (url: string) => {
    if (url.includes("sort_by=REPDTE")) {
      return { ok: true, status: 200, text: async () => JSON.stringify(ENVELOPE([WRAP(1, "X", "20260331")])) };
    }
    snapshotCalls++;
    return {
      ok: true, status: 200,
      text: async () => JSON.stringify(ENVELOPE([WRAP(1, "BANK A", "20260331"), WRAP(2, "BANK B", "20260331")], 2)),
    };
  };
  await refreshFinancials(fetchImpl as any, Date.parse("2026-07-20T00:00:00Z"), base);
  assert.equal(snapshotCalls, 1);
  assert.equal(latestFinancialsMeta()?.repdte, "2026-03-31");
  assert.equal(latestFinancialsMeta()?.count, 2);

  await refreshFinancials(fetchImpl as any, Date.parse("2026-07-21T00:00:00Z"), base);
  assert.equal(snapshotCalls, 1, "same REPDTE already on disk -> no second paginated pull");
});
