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
  summarize, refreshShortVol, latestShortVol, readArchivedDay,
  TOP_CAP, FLOOR_TOTAL_VOL,
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
