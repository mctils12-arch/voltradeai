// The nightly 13F completeness pass (server/edgar13fCompleteness.ts).
//
// MEASURED 2026-11-19 against EDGAR's daily form index
// (scripts/edgar13f_index_check.py): the archive held every 13F-HR of the
// eight quiet filing days of 2026-11-02..11-17, and 462 of the 1,712 filed on
// the deadline day, Monday 2026-11-16 (27%). The index lines below are copied
// from the real form.20261116.idx, trailing blanks included.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import os from "os";
import path from "path";
import zlib from "zlib";
import { fileURLToPath } from "url";
import {
  FILINGS13F_COMPLETENESS_JOB,
  FILINGS13F_COMPLETENESS_MAX_ATTEMPTS,
  FILINGS13F_COMPLETENESS_REFUSAL_STOP,
  filings13fCompletenessEnabled,
  filings13fCompletenessStatus,
  held13FAccessions,
  in13FCompletenessWindow,
  ledger13FSummary,
  parseDailyFormIndex13F,
  read13FLedger,
  run13FCompleteness,
  run13FNight,
  weekdaysBack,
  type Ledger13F,
} from "./edgar13fCompleteness";
import { type CompletenessFetch } from "./edgarForm4Completeness";
import { FOCUSED_MAX_HOLDINGS } from "./edgar13f";
import { JOB_REGISTRY } from "./jobHeartbeat";
import { DIAG_PROBES, sanitizeDiag } from "./diag";

const here = path.dirname(fileURLToPath(import.meta.url));

const REAL_LINES = [
  "10-Q             Zeta Example Corp                                             1890225     20261116    edgar/data/1890225/0001683168-26-008661.txt                                                ",
  "13F-HR           &PARTNERS                                                     107136      20261116    edgar/data/107136/0001214659-26-014210.txt                                                ",
  "13F-HR           1015 Capital Partners, LLC                                    2133991     20261116    edgar/data/2133991/0001376474-26-000734.txt                                                ",
  "13F-HR           11 Capital Partners LP                                        1801172     20261116    edgar/data/1801172/0000902664-26-004569.txt                                                ",
  "13F-HR/A         ADAR1 Capital Management, LLC                                 1940272     20261116    edgar/data/1940272/0001940272-26-000034.txt                                                ",
  "13F-HR/A         AMERIFLEX GROUP, INC.                                         2057637     20261116    edgar/data/2057637/0002057637-26-000006.txt                                                ",
  "13F-NT           140 Summer Partners Master Fund LP                            1844084     20261116    edgar/data/1844084/0000919574-26-007847.txt                                                ",
  "13F-NT           1922 Investments GP LLC                                       1584304     20261116    edgar/data/1584304/0001104659-26-127551.txt                                                ",
];
const INDEX_1116 = ["Form Type   Company Name      CIK      Date Filed  File Name", "-----", ...REAL_LINES].join("\n") + "\n";
const HR = ["0001214659-26-014210", "0001376474-26-000734", "0000902664-26-004569", "0001940272-26-000034", "0002057637-26-000006"];

const primaryDoc = (entries: number) => `<?xml version="1.0"?>
<edgarSubmission><headerData><submissionType>13F-HR</submissionType>
<filerInfo><filer><credentials><cik>0000107136</cik></credentials></filer><periodOfReport>09-30-2026</periodOfReport></filerInfo></headerData>
<formData><coverPage><filingManager><name>Test Manager &amp; Co</name></filingManager></coverPage>
<summaryPage><otherIncludedManagersCount>0</otherIncludedManagersCount><tableEntryTotal>${entries}</tableEntryTotal><tableValueTotal>1500000</tableValueTotal></summaryPage></formData></edgarSubmission>`;
const infoTable = `<informationTable><ns1:infoTable xmlns:ns1="x"><ns1:nameOfIssuer>SMALL CAP INC</ns1:nameOfIssuer><ns1:titleOfClass>COM</ns1:titleOfClass><ns1:cusip>123456789</ns1:cusip><ns1:value>1500000</ns1:value><ns1:shrsOrPrnAmt><ns1:sshPrnamt>1000</ns1:sshPrnamt><ns1:sshPrnamtType>SH</ns1:sshPrnamtType></ns1:shrsOrPrnAmt><ns1:investmentDiscretion>SOLE</ns1:investmentDiscretion></ns1:infoTable></informationTable>`;

const resp = (status: number, body = "") => ({ ok: status >= 200 && status < 300, status, text: async () => body });

/** A stand-in for EDGAR: daily indexes by day (403 for any other day, as
 *  EDGAR answers), and every filing directory answering index.json, a
 *  primary doc and an information table. Records every URL asked. */
function edgarLike(
  indexes: Record<string, string>,
  opts: { filingStatus?: (acc: string) => number; noPrimary?: Set<string>; entries?: (acc: string) => number; indexStatus?: (day: string) => number } = {},
) {
  const asked: string[] = [];
  const impl: CompletenessFetch = async (url) => {
    asked.push(url);
    const idx = /daily-index\/\d{4}\/QTR\d\/form\.(\d{4})(\d{2})(\d{2})\.idx$/.exec(url);
    if (idx) {
      const day = `${idx[1]}-${idx[2]}-${idx[3]}`;
      const forced = opts.indexStatus?.(day);
      if (forced) return resp(forced);
      return indexes[day] === undefined ? resp(403, "no such file") : resp(200, indexes[day]);
    }
    const m = /\/Archives\/edgar\/data\/\d+\/(\d{18})\/(.+)$/.exec(url);
    if (!m) return resp(404);
    const acc = `${m[1].slice(0, 10)}-${m[1].slice(10, 12)}-${m[1].slice(12)}`;
    const status = opts.filingStatus?.(acc) ?? 200;
    if (status !== 200) return resp(status);
    if (m[2] === "index.json") {
      const item = opts.noPrimary?.has(acc) ? [{ name: `${acc}.txt` }] : [{ name: "primary_doc.xml" }, { name: "infotable.xml" }];
      return resp(200, JSON.stringify({ directory: { item } }));
    }
    return resp(200, m[2] === "primary_doc.xml" ? primaryDoc(opts.entries?.(acc) ?? 1) : infoTable);
  };
  return { impl, asked };
}

function tmpArchive(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), "13f-completeness-"));
}
function writeDay(base: string, day: string, rows: object[], gz = false): void {
  const dir = path.join(base, "filings13f");
  fs.mkdirSync(dir, { recursive: true });
  const text = rows.map((r) => JSON.stringify(r)).join("\n") + "\n";
  if (gz) fs.writeFileSync(path.join(dir, `${day}.jsonl.gz`), zlib.gzipSync(text));
  else fs.writeFileSync(path.join(dir, `${day}.jsonl`), text);
}
function archiveRows(base: string): Array<Record<string, any>> {
  const dir = path.join(base, "filings13f");
  const out: Array<Record<string, any>> = [];
  for (const f of fs.readdirSync(dir).sort()) {
    const buf = fs.readFileSync(path.join(dir, f));
    const text = f.endsWith(".gz") ? zlib.gunzipSync(buf).toString("utf8") : buf.toString("utf8");
    for (const line of text.split("\n")) if (line) out.push(JSON.parse(line));
  }
  return out;
}
/** A plain appender shaped like archive13FFilings, so a test owns its own
 *  dedup state (the real writer keeps a process-wide seen set). */
const appendTo = (rows: object[], baseDir?: string, nowMs?: number): number => {
  const dir = path.join(baseDir as string, "filings13f");
  fs.mkdirSync(dir, { recursive: true });
  fs.appendFileSync(path.join(dir, `${new Date(nowMs as number).toISOString().slice(0, 10)}.jsonl`), rows.map((r) => JSON.stringify(r)).join("\n") + "\n");
  return rows.length;
};

const T = (iso: string) => Date.parse(iso);
/** Tuesday 2026-11-17 07:00 ET: yesterday is the deadline Monday. */
const TUE = T("2026-11-17T12:00:00Z");
const base = (over: Partial<Parameters<typeof run13FCompleteness>[0]> & { fetch: CompletenessFetch; baseDir: string }) => ({
  ledger: { days: {} } as Ledger13F, now: () => TUE, delayMs: 0, archiveStart: "2026-11-16", append: appendTo, ...over,
});

test("parseDailyFormIndex13F keeps exactly 13F-HR and 13F-HR/A, one row per accession", () => {
  const rows = parseDailyFormIndex13F(INDEX_1116 + REAL_LINES[1] + "\n");
  assert.deepEqual(rows.map((r) => r.accession), HR);
  assert.deepEqual(rows.map((r) => r.form), ["13F-HR", "13F-HR", "13F-HR", "13F-HR/A", "13F-HR/A"]);
  assert.deepEqual(rows[0], { accession: HR[0], form: "13F-HR", cik: "107136", filedAt: "2026-11-16" });
  assert.equal(parseDailyFormIndex13F("").length, 0);
});

test("weekdaysBack: yesterday back to the archive's first day, newest first, no weekend, never today", () => {
  assert.deepEqual(weekdaysBack(TUE, "2026-11-10"), ["2026-11-16", "2026-11-13", "2026-11-12", "2026-11-11", "2026-11-10"]);
  // 23:30 ET Monday is still Monday in New York though Tuesday in UTC
  assert.equal(weekdaysBack(T("2026-11-17T04:30:00Z"), "2026-11-10")[0], "2026-11-13");
});

test("window and off switch", () => {
  assert.equal(in13FCompletenessWindow(T("2026-11-17T11:00:00Z")), true);   // 06:00 ET
  assert.equal(in13FCompletenessWindow(T("2026-11-17T10:59:00Z")), false);  // 05:59 ET: the Form 4 pass's window
  assert.equal(in13FCompletenessWindow(T("2026-11-17T14:00:00Z")), false);  // 09:00 ET
  assert.equal(filings13fCompletenessEnabled({}), true);
  assert.equal(filings13fCompletenessEnabled({ VOLTRADE_13F_COMPLETENESS: "0" }), false);
});

test("held13FAccessions reads plain and gzipped day files", async () => {
  const dir = tmpArchive();
  writeDay(dir, "2026-11-16", [{ accession: HR[0], holdings: [] }]);
  writeDay(dir, "2026-11-13", [{ accession: HR[1] }, { managerName: "no accession" }], true);
  assert.deepEqual([...(await held13FAccessions(path.join(dir, "filings13f")))].sort(), [HR[0], HR[1]].sort());
  assert.equal((await held13FAccessions(path.join(dir, "nope"))).size, 0);
});

test("REGRESSION deadline Monday 2026-11-16: the filings the poll missed are fetched and appended as recovered rows", async () => {
  const dir = tmpArchive();
  writeDay(dir, "2026-11-16", [{ accession: HR[0], filedAt: "2026-11-16" }]);
  const { impl, asked } = edgarLike({ "2026-11-16": INDEX_1116 });
  const ledger: Ledger13F = { days: {} };
  const run = await run13FCompleteness(base({ fetch: impl, baseDir: dir, ledger }));
  assert.equal(run.status, "ok");
  assert.deepEqual(run.totals, { index_13f: 5, held_before: 1, backfilled: 4, unparsed: 0, failed: 0, deferred: 0 });
  const rows = archiveRows(dir);
  assert.deepEqual(rows.map((r) => r.accession).sort(), [...HR].sort());
  const rec = rows.find((r) => r.accession === HR[3])!;
  assert.equal(rec.form, "13F-HR/A");
  assert.equal(rec.filedAt, "2026-11-16");
  assert.equal(rec.backfilledAt, new Date(TUE).toISOString());
  assert.equal(rec.cik, "0001940272");
  assert.equal(rec.indexUrl, "https://www.sec.gov/Archives/edgar/data/1940272/000194027226000034/0001940272-26-000034-index.htm");
  assert.equal(rec.managerName, "Test Manager & Co");
  assert.equal(rec.periodOfReport, "2026-09-30");
  assert.equal(rec.holdings.length, 1);
  assert.equal(rec.holdingsOmitted, false);
  // the held filing and the 13F-NT notices are never fetched
  assert.ok(!asked.some((u) => u.includes("000121465926014210")));
  assert.ok(!asked.some((u) => u.includes("000091957426007847")));
  assert.deepEqual(ledger.days["2026-11-16"], {
    status: "checked", index_13f: 5, held: 5, unparsed: 0, failed: 0, deferred: 0, attempts: 1,
    complete: true, settled: true, checked_at: new Date(TUE).toISOString(),
  });
  // a settled day is not read again
  const again = edgarLike({ "2026-11-16": INDEX_1116 });
  const run2 = await run13FCompleteness(base({ fetch: again.impl, baseDir: dir, ledger }));
  assert.equal(again.asked.length, 0);
  assert.equal(run2.days_read, 0);
  assert.equal(run2.status, "ok");
});

test("the focused-manager cap applies to recovered rows: over 250 entries is summary-only, table not fetched", async () => {
  const dir = tmpArchive();
  const { impl, asked } = edgarLike({ "2026-11-16": INDEX_1116 }, { entries: (acc) => (acc === HR[1] ? FOCUSED_MAX_HOLDINGS + 1 : 1) });
  await run13FCompleteness(base({ fetch: impl, baseDir: dir }));
  const big = archiveRows(dir).find((r) => r.accession === HR[1])!;
  assert.equal(big.holdingsOmitted, true);
  assert.equal(big.holdings, null);
  assert.ok(!asked.some((u) => u.includes("000137647426000734/infotable.xml")));
});

test("the per-run cap defers the rest; the day stays unsettled and the next night finishes it", async () => {
  const dir = tmpArchive();
  const { impl } = edgarLike({ "2026-11-16": INDEX_1116 });
  const ledger: Ledger13F = { days: {} };
  const run = await run13FCompleteness(base({ fetch: impl, baseDir: dir, ledger, maxPerRun: 2 }));
  assert.equal(run.status, "partial");
  assert.equal(run.capped, true);
  assert.deepEqual([run.totals.backfilled, run.totals.deferred], [2, 3]);
  assert.equal(ledger.days["2026-11-16"].settled, false);
  assert.equal(ledger.days["2026-11-16"].complete, false);
  const run2 = await run13FCompleteness(base({ fetch: impl, baseDir: dir, ledger }));
  assert.deepEqual([run2.totals.held_before, run2.totals.backfilled], [2, 3]);
  assert.equal(ledger.days["2026-11-16"].complete, true);
  assert.equal(archiveRows(dir).length, 5);
});

test("a failed fetch is retried on later nights, then the day is settled short with its failures stated; no primary doc is unparsed", async () => {
  const dir = tmpArchive();
  const { impl } = edgarLike({ "2026-11-16": INDEX_1116 }, { filingStatus: (a) => (a === HR[2] ? 500 : 200), noPrimary: new Set([HR[4]]) });
  const ledger: Ledger13F = { days: {} };
  for (let night = 1; night <= FILINGS13F_COMPLETENESS_MAX_ATTEMPTS; night++) {
    const run = await run13FCompleteness(base({ fetch: impl, baseDir: dir, ledger }));
    assert.equal(run.status, "partial");
    const d = ledger.days["2026-11-16"];
    assert.deepEqual([d.attempts, d.held, d.failed, d.unparsed, d.complete], [night, 3, 1, 1, false]);
    assert.equal(d.settled, night === FILINGS13F_COMPLETENESS_MAX_ATTEMPTS);
  }
  assert.equal(archiveRows(dir).length, 3);
  const s = ledger13FSummary(ledger, TUE, "2026-11-16") as any;
  assert.deepEqual([s.short_days, s.complete_days, s.complete_back_to], [1, 0, null]);
  assert.deepEqual(s.short, [{ day: "2026-11-16", index_13f: 5, held: 3 }]);
});

test("EDGAR refusing filings in a row stops the pass and reads as an error", async () => {
  const dir = tmpArchive();
  const many = Array.from({ length: 12 }, (_, i) =>
    `13F-HR           Manager ${i}                                                    ${1000 + i}      20261116    edgar/data/${1000 + i}/0000001000-26-${String(i).padStart(6, "0")}.txt   `).join("\n");
  const { impl, asked } = edgarLike({ "2026-11-16": many }, { filingStatus: () => 403 });
  const ledger: Ledger13F = { days: {} };
  const run = await run13FCompleteness(base({ fetch: impl, baseDir: dir, ledger }));
  assert.equal(run.status, "error");
  assert.equal(run.refused, true);
  assert.equal(asked.length, 1 + FILINGS13F_COMPLETENESS_REFUSAL_STOP);
  assert.equal(ledger.days["2026-11-16"].settled, false);
});

test("no index: one weekday is a holiday (settled only once it is 3 days old); three in a row is a refusal and records nothing", async () => {
  const dir = tmpArchive();
  // Veterans Day 2026-11-11 has no index; read on Tuesday 11-17 it is 6 days old
  const { impl } = edgarLike({ "2026-11-16": INDEX_1116, "2026-11-13": "", "2026-11-12": "", "2026-11-10": "" });
  const ledger: Ledger13F = { days: {} };
  const run = await run13FCompleteness(base({ fetch: impl, baseDir: dir, ledger, archiveStart: "2026-11-10" }));
  assert.equal(run.status, "ok");
  assert.deepEqual([ledger.days["2026-11-11"].status, ledger.days["2026-11-11"].settled], ["no_index", true]);
  assert.equal(ledger.days["2026-11-13"].complete, true); // an index with no 13F-HR is a complete day
  const s = ledger13FSummary(ledger, TUE, "2026-11-10") as any;
  assert.deepEqual([s.weekdays, s.complete_days, s.closed_days, s.not_read_days, s.complete_back_to], [5, 4, 1, 0, "2026-11-10"]);

  // yesterday's index not published yet: recorded, not settled, read again
  const early: Ledger13F = { days: {} };
  await run13FCompleteness(base({ fetch: edgarLike({}).impl, baseDir: dir, ledger: early }));
  assert.deepEqual([early.days["2026-11-16"].status, early.days["2026-11-16"].settled], ["no_index", false]);

  // every index read answers 403: EDGAR is refusing this client
  const refusedLedger: Ledger13F = { days: {} };
  const refusedRun = await run13FCompleteness(base({ fetch: edgarLike({}).impl, baseDir: dir, ledger: refusedLedger, archiveStart: "2026-11-02" }));
  assert.equal(refusedRun.status, "error");
  assert.equal(refusedRun.refused, true);
  assert.deepEqual(refusedLedger.days, {});
});

test("an index read that fails for another reason is an error, not 'no index'", async () => {
  const dir = tmpArchive();
  const ledger: Ledger13F = { days: {} };
  const run = await run13FCompleteness(base({ fetch: edgarLike({}, { indexStatus: () => 500 }).impl, baseDir: dir, ledger }));
  assert.equal(run.status, "error");
  assert.deepEqual(ledger.days, {});
});

test("a night uses the poll's own writer, persists ledger and state, and the probe serves them as counts", async () => {
  const dir = tmpArchive();
  const hb = fs.mkdtempSync(path.join(os.tmpdir(), "13f-hb-"));
  const env = { DATA_DIR: hb };
  const { impl } = edgarLike({ "2026-11-16": INDEX_1116 });
  const { run, failed } = await run13FNight({ fetch: impl, now: () => TUE, delayMs: 0, archiveStart: "2026-11-16", baseDir: dir }, env);
  assert.equal(failed, false);
  assert.equal(run.totals.backfilled, 5);
  assert.equal(archiveRows(dir).length, 5);
  assert.ok(fs.existsSync(path.join(dir, "filings13f", "2026-11-17.jsonl")), "rows reach <archive>/filings13f/ under the night's day");
  assert.equal(read13FLedger(path.join(hb, "voltrade_13f_day_ledger.json")).days["2026-11-16"].complete, true);
  const status = filings13fCompletenessStatus(env) as any;
  assert.equal(status.latest.totals.backfilled, 5);
  assert.equal(status.enabled, true);
  assert.deepEqual(sanitizeDiag(status), status);
  assert.ok(!JSON.stringify(status).includes(HR[0]), "no accession in the probe");
});

test("wiring: registered job, booted beside the poll, probe served and scrub-clean, manifest documents the marker", () => {
  const spec = JOB_REGISTRY.find((j) => j.id === FILINGS13F_COMPLETENESS_JOB);
  assert.ok(spec && spec.kind === "daily");
  assert.equal(spec.source, "server/edgar13fCompleteness.ts");
  assert.equal(spec.enabledWhen?.({ VOLTRADE_13F_COMPLETENESS: "0" }), false);
  assert.equal(spec.enabledWhen?.({}), true);
  const routes = fs.readFileSync(path.join(here, "routes.ts"), "utf8");
  assert.ok(routes.includes("boot13FCompleteness();"));
  const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
  assert.ok(bot.includes('case "filings13f_completeness"'));
  assert.ok((DIAG_PROBES as readonly string[]).includes("filings13f_completeness"));
  const status = filings13fCompletenessStatus({});
  assert.equal(status.probe, "filings13f_completeness");
  assert.deepEqual(sanitizeDiag(status), status);
  const manifest = JSON.parse(fs.readFileSync(path.join(here, "..", "datacore", "manifests", "filings13f.json"), "utf8"));
  assert.ok(manifest.field_map.backfilledAt, "the manifest documents the recovered-row marker");
  assert.ok(manifest.field_map.form);
});
