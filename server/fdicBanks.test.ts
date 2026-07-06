// FDIC bank-failures battery (BUILD ORDER 6 #3 v1): nested envelope
// parse, M/D/YYYY date normalization, event-identity dedup across
// fetches AND restarts, host + fields pin.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  parseFailures, normalizeFailDate, fetchRecentFailures,
  archiveNewFailures, gzipOldFailureDays, refreshFailures,
  latestFailures, FAILURES_FETCH_LIMIT,
} from "./fdicBanks";

// Mirrors the live FDIC envelope verified 2026-07-06: rows nested under
// data[].data, FAILDATE as M/D/YYYY, amounts in $ thousands, COST null
// until the DIF loss is estimated.
const WRAP = (name: string, cert: number, faildate: string, cost: number | null) => ({
  data: {
    NAME: name, CERT: cert, FAILDATE: faildate,
    CITYST: "LAGRANGE, GA", PSTALP: "GA", CHCLASS: "NM",
    RESTYPE: "FAILURE", QBFASSET: 305716, QBFDEP: 296420,
    COST: cost, SAVR: "DIF", ID: "4115",
  },
  score: 1,
});

const ENVELOPE = (rows: any[]) => ({ meta: { total: rows.length }, data: rows });

test("normalizeFailDate: M/D/YYYY -> ISO; garbage -> null", () => {
  assert.equal(normalizeFailDate("5/1/2026"), "2026-05-01");
  assert.equal(normalizeFailDate("12/31/2025"), "2025-12-31");
  assert.equal(normalizeFailDate("2026-05-01"), null, "already-ISO is not this API's shape");
  assert.equal(normalizeFailDate(""), null);
  assert.equal(normalizeFailDate(null), null);
});

test("parseFailures: nested envelope lands; COST null preserved (not zero)", () => {
  const rows = parseFailures(ENVELOPE([
    WRAP("COMMUNITY BANK AND TRUST - WEST GEORGIA", 25796, "5/1/2026", null),
    WRAP("PULASKI SAVINGS BANK", 57488, "1/30/2026", 19647),
  ]), "2026-07-06");
  assert.equal(rows.length, 2);
  assert.equal(rows[0].fail_date, "2026-05-01");
  assert.equal(rows[0].cert, 25796);
  assert.equal(rows[0].assets_k, 305716);
  assert.equal(rows[0].cost_k, null, "unestimated DIF loss stays null, never zero");
  assert.equal(rows[1].cost_k, 19647);
  assert.deepEqual(parseFailures(ENVELOPE([]), "x"), []);
  assert.deepEqual(parseFailures(null, "x"), []);
  assert.deepEqual(parseFailures(ENVELOPE([{ data: { NAME: "NO DATE" } }]), "x"), []);
});

test("fetchRecentFailures: non-200 -> []; url pins api.fdic.gov host + fields + DESC", async () => {
  const urls: string[] = [];
  const bad = async (url: string) => { urls.push(url); return { ok: false, status: 500, text: async () => "" }; };
  assert.deepEqual(await fetchRecentFailures(bad as any), []);
  assert.ok(urls[0].startsWith("https://api.fdic.gov/banks/failures"),
    "host is api.fdic.gov — banks.data.fdic.gov 301s (probe finding 2026-07-06)");
  assert.ok(urls[0].includes(`limit=${FAILURES_FETCH_LIMIT}`) && urls[0].includes("sort_order=DESC"));
  assert.ok(urls[0].includes("FAILDATE"), "explicit fields list travels on the request");
});

test("archive: event-identity dedup within and across fetches; restart re-seeds from disk", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "fdic-"));
  const now = Date.parse("2026-07-06T12:00:00Z");
  const batch = parseFailures(ENVELOPE([
    WRAP("BANK A", 111, "5/1/2026", null),
    WRAP("BANK B", 222, "1/30/2026", 19647),
  ]), "2026-07-06");
  assert.equal(archiveNewFailures(batch, base, now), 2);
  assert.equal(archiveNewFailures(batch, base, now), 0, "same events never re-archive");
  const withNew = [...batch, ...parseFailures(ENVELOPE([WRAP("BANK C", 333, "7/3/2026", null)]), "2026-07-06")];
  assert.equal(archiveNewFailures(withNew, base, now), 1, "only the genuinely new event lands");
  const dayFile = path.join(base, "fdicfailures", "2026-07-06.jsonl");
  assert.equal(fs.readFileSync(dayFile, "utf8").trim().split("\n").length, 3);
});

test("gz after 2d + refresh caches recent failures", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "fdic-"));
  const t0 = Date.parse("2026-07-01T12:00:00Z");
  archiveNewFailures(parseFailures(ENVELOPE([WRAP("OLD BANK", 999, "6/30/2026", null)]), "2026-07-01"), base, t0);
  assert.equal(gzipOldFailureDays(base, Date.parse("2026-07-05T00:00:00Z")), 1);
  const ok = async () => ({
    ok: true, status: 200,
    text: async () => JSON.stringify(ENVELOPE([WRAP("FRESH BANK", 444, "7/2/2026", null)])),
  });
  await refreshFailures(ok as any, Date.parse("2026-07-06T12:00:00Z"), base);
  const hit = latestFailures();
  assert.ok(hit);
  assert.equal(hit!.failures[0].name, "FRESH BANK");
});
