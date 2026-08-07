// Cboe VIX term-structure battery (EDGE DOCTRINE build-first pipeline,
// scheduled-routine session 2026-08-07). Fixtures are VERBATIM row shapes
// from the live 2026-08-07 workup (DATE,OPEN,HIGH,LOW,CLOSE for the OHLC
// tenors; DATE,VVIX for the close-only series).
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import {
  normalizeMdY, parseOhlcCsv, parseCloseOnlyCsv, mergeTenors, seriesUrl,
  archiveCboeVixDays, isDateArchived, refreshCboeVix, latestCboeVix,
  gzipPastYearFiles, _resetCboeVixForTests, CBOE_VIX_FLOOR_DATE,
  OHLC_TENORS, CLOSE_ONLY_TENORS,
} from "./cboeVix";

const VIX_CSV = [
  "DATE,OPEN,HIGH,LOW,CLOSE",
  "08/03/2026,16.030000,16.300000,15.540000,15.860000",
  "08/04/2026,15.760000,16.650000,15.510000,16.500000",
  "08/05/2026,16.150000,18.430000,15.480000,15.810000",
].join("\r\n") + "\r\n";

const VVIX_CSV = [
  "DATE,VVIX",
  "08/03/2026,89.500000",
  "08/04/2026,91.200000",
  "08/05/2026,90.430000",
].join("\r\n") + "\r\n";

test("normalizeMdY: MM/DD/YYYY -> YYYY-MM-DD, garbage -> null", () => {
  assert.equal(normalizeMdY("08/06/2026"), "2026-08-06");
  assert.equal(normalizeMdY("2026-08-06"), null);
  assert.equal(normalizeMdY("garbage"), null);
});

test("parseOhlcCsv: verbatim VIX shape, CLOSE column extracted", () => {
  const m = parseOhlcCsv(VIX_CSV);
  assert.equal(m.size, 3);
  assert.equal(m.get("2026-08-03"), 15.86);
  assert.equal(m.get("2026-08-04"), 16.5);
  assert.equal(m.get("2026-08-05"), 15.81);
  assert.deepEqual(parseOhlcCsv("garbage"), new Map(), "missing DATE, header -> empty, never throws");
});

test("parseCloseOnlyCsv: verbatim VVIX shape (2 columns)", () => {
  const m = parseCloseOnlyCsv(VVIX_CSV);
  assert.equal(m.size, 3);
  assert.equal(m.get("2026-08-05"), 90.43);
});

test("mergeTenors: floors at CBOE_VIX_FLOOR_DATE, skips partial days, computes ratios", () => {
  const mk = (rows: [string, number][]) => new Map(rows);
  const series = {
    VIX1D: mk([["2022-05-13", 20], ["2026-08-05", 12.55]]), // pre-floor day present here...
    VIX9D: mk([["2026-08-05", 13.79]]),
    VIX: mk([["2026-08-05", 15.81]]),
    VIX3M: mk([["2026-08-05", 18.95]]),
    VIX6M: mk([["2026-08-05", 21.06]]),
    VVIX: mk([["2026-08-05", 90.43]]),
  };
  const days = mergeTenors(series as any);
  // 2022-05-13 dropped: it's before the floor AND only VIX1D has it (partial)
  assert.equal(days.length, 1);
  assert.equal(days[0].date, "2026-08-05");
  assert.equal(days[0].vix9d_vix_ratio, Math.round((13.79 / 15.81) * 10000) / 10000);
  assert.equal(days[0].vix_vix3m_ratio, Math.round((15.81 / 18.95) * 10000) / 10000);
  assert.ok(days[0].vix_vix3m_ratio! < 1, "normal contango day: vix < vix3m");
});

test("mergeTenors: a tenor missing for a date drops that day entirely (never fabricates)", () => {
  const mk = (rows: [string, number][]) => new Map(rows);
  const series = {
    VIX1D: mk([["2026-08-05", 12.55]]),
    VIX9D: mk([["2026-08-05", 13.79]]),
    VIX: mk([["2026-08-05", 15.81]]),
    VIX3M: mk([]), // missing this tenor for the date
    VIX6M: mk([["2026-08-05", 21.06]]),
    VVIX: mk([["2026-08-05", 90.43]]),
  };
  assert.deepEqual(mergeTenors(series as any), []);
});

test("seriesUrl: keyless cdn.cboe.com history CSV pattern", () => {
  assert.equal(seriesUrl("VIX"), "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv");
  assert.equal(seriesUrl("VIX9D"), "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX9D_History.csv");
});

test("OHLC_TENORS/CLOSE_ONLY_TENORS cover exactly the six probed series", () => {
  assert.deepEqual([...OHLC_TENORS], ["VIX1D", "VIX9D", "VIX", "VIX3M", "VIX6M"]);
  assert.deepEqual([...CLOSE_ONLY_TENORS], ["VVIX"]);
});

async function withTmpDir(fn: (dir: string) => void | Promise<void>) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "cboevix-test-"));
  try { await fn(dir); } finally { fs.rmSync(dir, { recursive: true, force: true }); }
}

test("archiveCboeVixDays + isDateArchived: day-level dedup round-trips through disk", async () => {
  await withTmpDir((dir) => {
    _resetCboeVixForTests();
    const days = [
      { date: "2026-08-05", vix1d: 12.55, vix9d: 13.79, vix: 15.81, vix3m: 18.95, vix6m: 21.06, vvix: 90.43,
        vix_vix3m_ratio: 0.8344, vix9d_vix_ratio: 0.8722 },
    ];
    assert.equal(isDateArchived("2026-08-05", dir), false);
    const n = archiveCboeVixDays(days, dir);
    assert.equal(n, 1);
    assert.equal(isDateArchived("2026-08-05", dir), true);
    // re-archiving the same day is a no-op
    assert.equal(archiveCboeVixDays(days, dir), 0);

    const written = fs.readFileSync(path.join(dir, "cboevix", "2026.jsonl"), "utf8").trim();
    assert.deepEqual(JSON.parse(written), days[0]);

    // a fresh in-process instance must re-seed from disk, not assume empty
    _resetCboeVixForTests();
    assert.equal(isDateArchived("2026-08-05", dir), true);
  });
});

test("gzipPastYearFiles: gzips a prior year, leaves the current year alone", async () => {
  await withTmpDir((dir) => {
    _resetCboeVixForTests();
    const vixDir = path.join(dir, "cboevix");
    fs.mkdirSync(vixDir, { recursive: true });
    fs.writeFileSync(path.join(vixDir, "2025.jsonl"), '{"date":"2025-06-01"}\n');
    fs.writeFileSync(path.join(vixDir, "2026.jsonl"), '{"date":"2026-08-05"}\n');
    const now = Date.UTC(2026, 7, 7);
    const n = gzipPastYearFiles(dir, now);
    assert.equal(n, 1);
    assert.ok(fs.existsSync(path.join(vixDir, "2025.jsonl.gz")));
    assert.ok(!fs.existsSync(path.join(vixDir, "2025.jsonl")));
    assert.ok(fs.existsSync(path.join(vixDir, "2026.jsonl")), "current year stays plain");
  });
});

test("archiveCboeVixDays: appending to a past year already gzipped reopens it (no data loss)", async () => {
  await withTmpDir((dir) => {
    _resetCboeVixForTests();
    const vixDir = path.join(dir, "cboevix");
    fs.mkdirSync(vixDir, { recursive: true });
    fs.writeFileSync(path.join(vixDir, "2025.jsonl.gz"), zlib.gzipSync('{"date":"2025-06-01"}\n'));
    _resetCboeVixForTests();
    const late = [{ date: "2025-06-02", vix1d: 1, vix9d: 1, vix: 1, vix3m: 1, vix6m: 1, vvix: 1,
                     vix_vix3m_ratio: 1, vix9d_vix_ratio: 1 }];
    archiveCboeVixDays(late, dir, Date.UTC(2026, 7, 7));
    const text = fs.readFileSync(path.join(vixDir, "2025.jsonl"), "utf8");
    assert.match(text, /2025-06-01/, "prior content preserved after reopen");
    assert.match(text, /2025-06-02/, "new day appended");
    assert.ok(!fs.existsSync(path.join(vixDir, "2025.jsonl.gz")), "stale gz removed on reopen");
  });
});

test("refreshCboeVix + latestCboeVix: fetch -> archive -> cache, using a fake fetch", async () => {
  await withTmpDir(async (dir) => {
    _resetCboeVixForTests();
    const csvFor: Record<string, string> = {
      VIX1D: "DATE,OPEN,HIGH,LOW,CLOSE\r\n08/05/2026,14.05,14.62,11.31,12.55\r\n",
      VIX9D: "DATE,OPEN,HIGH,LOW,CLOSE\r\n08/05/2026,16.21,17.05,13.37,13.79\r\n",
      VIX: VIX_CSV,
      VIX3M: "DATE,OPEN,HIGH,LOW,CLOSE\r\n08/05/2026,19.57,19.72,18.72,18.95\r\n",
      VIX6M: "DATE,OPEN,HIGH,LOW,CLOSE\r\n08/05/2026,21.42,21.52,20.91,21.06\r\n",
      VVIX: VVIX_CSV,
    };
    const fakeFetch = async (url: string) => {
      const ticker = Object.keys(csvFor).find((t) => url.includes(`/${t}_History.csv`))!;
      return { ok: true, status: 200, text: async () => csvFor[ticker] };
    };
    await refreshCboeVix(fakeFetch as any, dir, Date.UTC(2026, 7, 7));
    const hit = latestCboeVix();
    assert.ok(hit);
    assert.equal(hit!.latest.date, "2026-08-05");
    assert.equal(hit!.latest.vix, 15.81);
    assert.ok(hit!.recent.length >= 1);
    assert.equal(isDateArchived("2026-08-05", dir), true);
  });
});

test("refreshCboeVix: an HTTP failure on one tenor degrades to no-op for that tenor, never throws", async () => {
  await withTmpDir(async (dir) => {
    _resetCboeVixForTests();
    const fakeFetch = async (url: string) => {
      if (url.includes("VIX3M")) return { ok: false, status: 500, text: async () => "" };
      return { ok: true, status: 200, text: async () => VIX_CSV };
    };
    await assert.doesNotReject(refreshCboeVix(fakeFetch as any, dir, Date.UTC(2026, 7, 7)));
    // VIX3M never returned data -> no fully-populated day -> nothing archived
    assert.equal(isDateArchived("2026-08-05", dir), false);
  });
});
