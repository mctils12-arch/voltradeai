// OCC volume battery (DATACORE MAXIMUS census #1, archive-now root).
// Fixtures are VERBATIM rows and error bodies from the 2026-07-06 live
// workup — incl. the CRLF endings, the per-symbol trailing comma, the
// double-space "prior to  2 years" body, and the clearing-side
// double-count (AAL C and M rows both carry 34483).
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  classifyBody, parseOcc, normalizeActDate, occUrl, fetchOccDay,
  archiveOccDay, isDayArchived, aggregateOcc, refreshOcc, latestOcc,
  occDeepBackfillEnabled, occDeepBackfillIfEnabled,
  _resetOccForTests,
} from "./occVolume";

const HEADER = "quantity,underlying,symbol,actype,porc,exchange,actdate";
const DAY_CSV = [
  HEADER,
  "34483,AAL,1AAL,C,C,ISE,07/02/2026",
  "34483,AAL,1AAL,M,C,ISE,07/02/2026",
  "1168,BE,1BE,C,C,CBOE,07/02/2026",
  "1168,BE,1BE,F,C,CBOE,07/02/2026",
  "14814,BE,1BE,C,P,CBOE,07/02/2026",
].join("\r\n") + "\r\n";

test("classifyBody: all three verbatim error bodies + data + unknown", () => {
  assert.equal(classifyBody(DAY_CSV), "data");
  assert.equal(classifyBody("No record(s) found"), "no_records");
  assert.equal(classifyBody("Report date cannot be greater than 07/02/2026"), "not_yet_published");
  assert.equal(classifyBody("Report date cannot be prior to  2 years"), "aged_out", "double space verbatim");
  assert.equal(classifyBody("<html>maintenance</html>"), "unknown_error");
});

test("parseOcc: CRLF handled, 7 typed fields, per-symbol trailing comma tolerated", () => {
  const rows = parseOcc(DAY_CSV);
  assert.equal(rows.length, 5);
  assert.deepEqual(rows[0], { date: "2026-07-02", underlying: "AAL", symbol: "1AAL",
                              actype: "C", porc: "C", exchange: "ISE", qty: 34483 });
  // per-symbol variant shape (trailing comma)
  const perSym = parseOcc(HEADER + "\r\n1610,SPY,2SPY,C,C,CBOE,07/02/2026,\r\n");
  assert.equal(perSym.length, 1);
  assert.equal(perSym[0].qty, 1610);
  assert.equal(normalizeActDate("07/02/2026"), "2026-07-02");
  assert.equal(normalizeActDate("2026-07-02"), null);
  assert.deepEqual(parseOcc("No record(s) found"), [], "error bodies never parse as data");
});

test("aggregate: clearing-side double-count halved for totals; C/M put-call kept side-consistent", () => {
  const { top, underlyings } = aggregateOcc(parseOcc(DAY_CSV));
  assert.equal(underlyings, 2);
  const aal = top.find((s) => s.underlying === "AAL")!;
  assert.equal(aal.total_cleared, 34483, "C 34483 + M 34483 = 68966 raw -> halved");
  assert.equal(aal.cust_call, 34483);
  assert.equal(aal.mm_call, 34483);
  const be = top.find((s) => s.underlying === "BE")!;
  assert.equal(be.cust_put, 14814);
  assert.equal(be.cust_call, 1168);
});

test("archive: day-level dedup, gz-on-write, restart re-seed from disk", () => {
  _resetOccForTests();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "occ-"));
  const rows = parseOcc(DAY_CSV);
  assert.equal(archiveOccDay(rows, base), 5);
  assert.equal(archiveOccDay(rows, base), 0, "a report date is final — never re-archives");
  assert.ok(fs.existsSync(path.join(base, "occvolume", "2026-07-02.jsonl.gz")), "gz on write (~1MB/day vs 18MB plain)");
  assert.ok(isDayArchived("2026-07-02", base));
});

test("refresh: not-yet-published and weekend bodies are honest no-ops; data day lands + caches", async () => {
  _resetOccForTests();  // the module-level dedup Set spans the process by design
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "occ-"));
  const bodies: Record<string, string> = {
    "20260706": "Report date cannot be greater than 07/02/2026",
    "20260703": "No record(s) found",                     // holiday
    "20260702": DAY_CSV,
    "20260701": "No record(s) found",
    "20260630": "No record(s) found",
  };
  const calls: string[] = [];
  const fake = async (url: string) => {
    const ymd = url.match(/reportDate=(\d{8})/)![1];
    calls.push(ymd);
    return { ok: true, status: 200, text: async () => bodies[ymd] ?? "No record(s) found" };
  };
  await refreshOcc(fake as any, Date.parse("2026-07-06T23:00:00Z"), base);
  assert.ok(calls.includes("20260702"));
  assert.ok(isDayArchived("2026-07-02", base));
  const hit = latestOcc();
  assert.ok(hit);
  assert.equal(hit!.report_date, "2026-07-02");
  assert.equal(hit!.underlyings, 2);
  assert.ok(occUrl("20260702").includes("productKind=ALL"), "only ALL is accepted (probed)");
});

test("deep backfill: default-on (2026-07-07 approval) with opt-out, walks oldest-first, honors aged_out, writes done-marker exactly once", async () => {
  _resetOccForTests();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "occ-"));
  // DEFAULT-ON since 2026-07-07 (human-approved, volume headroom
  // confirmed ~2GB/5GB); OCC_DEEP_BACKFILL=0 is the explicit opt-out.
  const neverCalled: string[] = [];
  await occDeepBackfillIfEnabled((async (url: string) => {
    neverCalled.push(url);
    return { ok: true, status: 200, text: async () => "No record(s) found" };
  }) as any, Date.parse("2026-07-06T23:00:00Z"), base, { OCC_DEEP_BACKFILL: "0" } as any, 10, 0);
  assert.equal(neverCalled.length, 0, "OCC_DEEP_BACKFILL=0 must be a no-op");
  assert.equal(occDeepBackfillEnabled({} as any), true, "default-on per 2026-07-07 approval");
  assert.equal(occDeepBackfillEnabled({ OCC_DEEP_BACKFILL: "0" } as any), false, "opt-out honored");
  assert.equal(occDeepBackfillEnabled({ OCC_DEEP_BACKFILL: "1" } as any), true);

  // ON: 10-calendar-day window (injectable — prod default stays 730),
  // zero spacing for the test. Oldest days aged_out, one data day, rest empty.
  const ymds: string[] = [];
  const fake = async (url: string) => {
    const ymd = url.match(/reportDate=(\d{8})/)![1];
    ymds.push(ymd);
    if (ymd === "20260702") return { ok: true, status: 200, text: async () => DAY_CSV };
    if (ymd <= "20260628") return { ok: true, status: 200, text: async () => "Report date cannot be prior to  2 years" };
    return { ok: true, status: 200, text: async () => "No record(s) found" };
  };
  const NOW = Date.parse("2026-07-06T23:00:00Z");
  const env = { OCC_DEEP_BACKFILL: "1" } as any;
  await occDeepBackfillIfEnabled(fake as any, NOW, base, env, 10, 0);
  assert.ok(ymds.length >= 6, "weekday walk over 10 calendar days");
  assert.ok(ymds[0] < ymds[ymds.length - 1], "walk is OLDEST-FIRST — the purge eats the back edge");
  assert.ok(isDayArchived("2026-07-02", base), "the data day landed");
  const marker = path.join(base, "occvolume", "backfill_done.json");
  assert.ok(fs.existsSync(marker), "done-marker written");
  // second call: marker short-circuits — zero fetches
  const before = ymds.length;
  await occDeepBackfillIfEnabled(fake as any, NOW, base, env, 10, 0);
  assert.equal(ymds.length, before, "done-marker prevents a second pass");
});
