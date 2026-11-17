// The nightly Form 4 completeness pass (server/edgarForm4Completeness.ts).
//
// MEASURED 2026-11-17 against EDGAR's daily form index for Friday 2026-11-13:
// the index lists 663 distinct Form 4 accessions and 12 Form 4/A; the archive
// held 404 and 7. Of the archive's 600 rows filed that day, 189 were 424B2 /
// 424B3 / 485BPOS / 497 filings (the live feed's `type=4` is a prefix match),
// stored as rows with no issuer. Monday 2026-11-16 held 39 rows after a
// refused page size. The index lines below are copied from the real
// form.20261113.idx, trailing blanks included.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import os from "os";
import path from "path";
import zlib from "zlib";
import { fileURLToPath } from "url";
import {
  FORM4_COMPLETENESS_JOB,
  FORM4_COMPLETENESS_REFUSAL_STOP,
  candidateDays,
  dailyFormIndexUrl,
  filingDirUrl,
  form4CompletenessEnabled,
  form4CompletenessStatus,
  heldParsedAccessions,
  inCompletenessWindow,
  parseDailyFormIndex,
  readCompletenessState,
  runForm4Completeness,
  writeCompletenessState,
  type CompletenessFetch,
} from "./edgarForm4Completeness";
import { JOB_REGISTRY } from "./jobHeartbeat";
import { DIAG_PROBES, sanitizeDiag } from "./diag";

const here = path.dirname(fileURLToPath(import.meta.url));

const INDEX_HEAD = [
  "Description:           Daily Index of EDGAR Dissemination Feed by Form Type",
  "Last Data Received:    Nov 13, 2026",
  "Comments:              webmaster@sec.gov",
  "Anonymous FTP:         ftp://ftp.sec.gov/edgar/",
  " ",
  "Form Type   Company Name                                                  CIK",
  "      Date Filed  File Name",
  "---------------------------------------------------------------------------------------------------------------------------------------------",
];
const REAL_LINES = [
  "1-A POS          Doroni Aerospace, Inc.                                        1890225     20261113    edgar/data/1890225/0001683168-26-008661.txt                                                ",
  "4                101 HUNTINGTON HOLDINGS LLC                                   1208476     20261113    edgar/data/1208476/0001193125-26-461179.txt                             ",
  "4                101 Huntington Holdings Subsidiary, LLC                       2076441     20261113    edgar/data/2076441/0001193125-26-461179.txt                             ",
  "4                10x Genomics, Inc.                                            1770787     20261113    edgar/data/1770787/0001610717-26-000453.txt                             ",
  "4                ADVANCED MICRO DEVICES INC                                    2488        20261113    edgar/data/2488/0000002488-26-000193.txt                                ",
  "4                Denzel Nora                                                   1418351     20261113    edgar/data/1418351/0000002488-26-000193.txt                             ",
  "4/A              Alpine Global Management, LLC                                 1581655     20261113    edgar/data/1581655/0000899140-26-001311.txt                             ",
  "4/A              Alpine Partners (BVI), L.P.                                   1845320     20261113    edgar/data/1845320/0000899140-26-001311.txt                             ",
  "424B2            BANK OF AMERICA CORP /DE/                                     70858       20261113    edgar/data/70858/0001918704-26-035521.txt                               ",
  "425              BCP Investment Corp                                           1372807     20261113    edgar/data/1372807/0001193125-26-458870.txt                             ",
  "SCHEDULE 13D     Coursera, Inc.                                                1651562     20261113    edgar/data/1651562/0001072613-26-000890.txt                             ",
];
const INDEX_1113 = [...INDEX_HEAD, ...REAL_LINES].join("\n") + "\n";
const FORM4_ACCESSIONS = ["0001193125-26-461179", "0001610717-26-000453", "0000002488-26-000193", "0000899140-26-001311"];

const ownershipXml = (symbol: string) => `<?xml version="1.0"?>
<ownershipDocument>
  <documentType>4</documentType>
  <periodOfReport>2026-11-11</periodOfReport>
  <issuer><issuerCik>0000002488</issuerCik><issuerName>ISSUER ${symbol}</issuerName><issuerTradingSymbol>${symbol}</issuerTradingSymbol></issuer>
  <reportingOwner>
    <reportingOwnerId><rptOwnerCik>0001418351</rptOwnerCik><rptOwnerName>Owner One</rptOwnerName></reportingOwnerId>
    <reportingOwnerRelationship><isDirector>1</isDirector><isOfficer>0</isOfficer><isTenPercentOwner>0</isTenPercentOwner></reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable><nonDerivativeTransaction>
    <securityTitle><value>Common Stock</value></securityTitle>
    <transactionDate><value>2026-11-11</value></transactionDate>
    <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
    <transactionAmounts>
      <transactionShares><value>1000</value></transactionShares>
      <transactionPricePerShare><value>10.5</value></transactionPricePerShare>
      <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
    </transactionAmounts>
  </nonDerivativeTransaction></nonDerivativeTable>
</ownershipDocument>`;

const resp = (status: number, body = "") => ({ ok: status >= 200 && status < 300, status, text: async () => body });

/** A stand-in for EDGAR: daily indexes by day, and every filing directory
 *  answering index.json + one ownership XML. Records every URL asked. */
function edgarLike(indexes: Record<string, string>, opts: { filingStatus?: (accession: string) => number; emptyDoc?: Set<string> } = {}) {
  const asked: string[] = [];
  const impl: CompletenessFetch = async (url) => {
    asked.push(url);
    const idx = /daily-index\/\d{4}\/QTR\d\/form\.(\d{4})(\d{2})(\d{2})\.idx$/.exec(url);
    if (idx) {
      const day = `${idx[1]}-${idx[2]}-${idx[3]}`;
      return indexes[day] === undefined ? resp(403, "no such file") : resp(200, indexes[day]);
    }
    const dirMatch = /\/Archives\/edgar\/data\/\d+\/(\d{18})\/(.+)$/.exec(url);
    if (!dirMatch) return resp(404);
    const acc = `${dirMatch[1].slice(0, 10)}-${dirMatch[1].slice(10, 12)}-${dirMatch[1].slice(12)}`;
    const status = opts.filingStatus?.(acc) ?? 200;
    if (status !== 200) return resp(status);
    if (dirMatch[2] === "index.json") {
      return resp(200, JSON.stringify({ directory: { item: [{ name: `${acc}.txt` }, { name: "wk-form4_1.xml" }] } }));
    }
    return resp(200, opts.emptyDoc?.has(acc) ? "<xml><notOwnership/></xml>" : ownershipXml("TST"));
  };
  return { impl, asked };
}

function tmpArchive(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), "form4-completeness-"));
}
function writeDay(base: string, day: string, rows: object[], gz = false): void {
  const dir = path.join(base, "filings");
  fs.mkdirSync(dir, { recursive: true });
  const text = rows.map((r) => JSON.stringify(r)).join("\n") + "\n";
  if (gz) fs.writeFileSync(path.join(dir, `${day}.jsonl.gz`), zlib.gzipSync(text));
  else fs.writeFileSync(path.join(dir, `${day}.jsonl`), text);
}
function archiveRows(base: string): Array<Record<string, unknown>> {
  const dir = path.join(base, "filings");
  const out: Array<Record<string, unknown>> = [];
  for (const f of fs.readdirSync(dir).sort()) {
    const buf = fs.readFileSync(path.join(dir, f));
    const text = f.endsWith(".gz") ? zlib.gunzipSync(buf).toString("utf8") : buf.toString("utf8");
    for (const line of text.split("\n")) if (line) out.push(JSON.parse(line));
  }
  return out;
}
/** A plain appender shaped like archiveFilings, so a test owns its own dedup
 *  state (archiveFilings keeps a process-wide seen set). */
const appendTo = (rows: object[], baseDir?: string, nowMs?: number): number => {
  const dir = path.join(baseDir as string, "filings");
  fs.mkdirSync(dir, { recursive: true });
  fs.appendFileSync(path.join(dir, `${new Date(nowMs as number).toISOString().slice(0, 10)}.jsonl`), rows.map((r) => JSON.stringify(r)).join("\n") + "\n");
  return rows.length;
};

const T = (iso: string) => Date.parse(iso);
// Tuesday 2026-11-17 03:00 ET — the night after Monday 11-16.
const NIGHT = T("2026-11-17T08:00:00Z");

test("parseDailyFormIndex keeps exactly form types 4 and 4/A, one row per accession", () => {
  const got = parseDailyFormIndex(INDEX_1113);
  assert.deepEqual(got.map((f) => f.accession), FORM4_ACCESSIONS);
  assert.deepEqual(got.map((f) => f.form), ["4", "4", "4", "4/A"]);
  assert.ok(got.every((f) => f.filedAt === "2026-11-13"));
  // the first filer listed is the directory used
  assert.equal(got[0].cik, "1208476");
  assert.equal(got[2].cik, "2488");
  // prefix look-alikes are not Form 4: this is what the live feed gets wrong
  for (const notForm4 of ["0001918704-26-035521", "0001193125-26-458870", "0001072613-26-000890", "0001683168-26-008661"]) {
    assert.ok(!got.some((f) => f.accession === notForm4), notForm4);
  }
  assert.deepEqual(parseDailyFormIndex(""), []);
  assert.deepEqual(parseDailyFormIndex(INDEX_HEAD.join("\n")), []);
});

test("dailyFormIndexUrl and filingDirUrl build EDGAR's own paths", () => {
  assert.equal(dailyFormIndexUrl("2026-11-13"), "https://www.sec.gov/Archives/edgar/daily-index/2026/QTR4/form.20261113.idx");
  assert.equal(dailyFormIndexUrl("2026-01-02"), "https://www.sec.gov/Archives/edgar/daily-index/2026/QTR1/form.20260102.idx");
  assert.equal(dailyFormIndexUrl("2026-06-30"), "https://www.sec.gov/Archives/edgar/daily-index/2026/QTR2/form.20260630.idx");
  assert.equal(dailyFormIndexUrl("2026-07-01"), "https://www.sec.gov/Archives/edgar/daily-index/2026/QTR3/form.20260701.idx");
  // measured live 2026-11-17: this directory's index.json lists wk-form4_1794604616.xml
  assert.equal(filingDirUrl({ accession: "0000002488-26-000193", cik: "2488" }), "https://www.sec.gov/Archives/edgar/data/2488/000000248826000193/");
});

test("candidateDays: the weekdays of the last 7 ET days, oldest first, never today", () => {
  assert.deepEqual(candidateDays(NIGHT), ["2026-11-10", "2026-11-11", "2026-11-12", "2026-11-13", "2026-11-16"]);
  // 23:30 ET on Monday is still Monday in ET although it is Tuesday in UTC
  assert.deepEqual(candidateDays(T("2026-11-17T04:30:00Z")).at(-1), "2026-11-13");
  assert.deepEqual(candidateDays(NIGHT, 1), ["2026-11-16"]);
});

test("window and off switch", () => {
  assert.equal(inCompletenessWindow(T("2026-11-17T06:59:00Z")), false); // 01:59 ET
  assert.equal(inCompletenessWindow(T("2026-11-17T07:00:00Z")), true); // 02:00 ET
  assert.equal(inCompletenessWindow(T("2026-11-17T10:59:00Z")), true); // 05:59 ET
  assert.equal(inCompletenessWindow(T("2026-11-17T11:00:00Z")), false); // 06:00 ET
  assert.equal(inCompletenessWindow(T("2026-11-16T15:00:00Z")), false); // market hours
  assert.equal(form4CompletenessEnabled({}), true);
  assert.equal(form4CompletenessEnabled({ VOLTRADE_FORM4_COMPLETENESS: "0" }), false);
  assert.equal(form4CompletenessEnabled({ VOLTRADE_FORM4_COMPLETENESS: "on" }), true);
});

test("heldParsedAccessions reads plain and gzipped day files; an issuer-less shell is not held", () => {
  const base = tmpArchive();
  writeDay(base, "2026-11-13", [{ accession: "A-1", issuerCik: "0000000001" }, { accession: "SHELL", issuerCik: null }], true);
  writeDay(base, "2026-11-14", [{ accession: "A-2", issuerCik: "0000000002" }]);
  fs.appendFileSync(path.join(base, "filings", "2026-11-14.jsonl"), '{"accession":"TORN","issuerCik":"1"\n');
  writeDay(base, "2026-11-12", [{ accession: "BEFORE", issuerCik: "0000000003" }]);
  const held = heldParsedAccessions(path.join(base, "filings"), "2026-11-13", "2026-11-17");
  assert.deepEqual([...held].sort(), ["A-1", "A-2"]);
});

test("a pass fetches only the Form 4s the archive lacks and appends them marked as recovered", async () => {
  const base = tmpArchive();
  // held live: one Form 4, plus the 424B2 the live poll stored as a shell
  writeDay(base, "2026-11-13", [
    { accession: "0001610717-26-000453", issuerCik: "0001770787", filedAt: "2026-11-13" },
    { accession: "0001918704-26-035521", issuerCik: null, filedAt: "2026-11-13" },
  ], true);
  const edgar = edgarLike({ "2026-11-13": INDEX_1113 });
  const run = await runForm4Completeness({ fetch: edgar.impl, now: () => NIGHT, baseDir: base, delayMs: 0, append: appendTo });

  const day = run.days.find((d) => d.day === "2026-11-13")!;
  assert.deepEqual(
    { status: day.status, index: day.index_form4, held: day.held_before, backfilled: day.backfilled, complete: day.complete },
    { status: "checked", index: 4, held: 1, backfilled: 3, complete: true },
  );
  assert.equal(run.status, "ok");
  assert.equal(run.totals.backfilled, 3);
  // the other four weekdays have no index here: not an error
  assert.deepEqual(run.days.filter((d) => d.day !== "2026-11-13").map((d) => d.status), ["no_index", "no_index", "no_index", "no_index"]);

  const rows = archiveRows(base);
  const recovered = rows.filter((r) => r.backfilledAt);
  assert.deepEqual(recovered.map((r) => r.accession).sort(), ["0000002488-26-000193", "0000899140-26-001311", "0001193125-26-461179"]);
  for (const r of recovered) {
    assert.equal(r.filedAt, "2026-11-13");
    assert.equal(r.backfilledAt, "2026-11-17T08:00:00.000Z");
    assert.equal(r.issuerTradingSymbol, "TST");
    assert.match(String(r.cik), /^\d{10}$/);
    assert.ok(String(r.indexUrl).endsWith(`${r.accession}-index.htm`));
  }
  assert.equal(recovered.find((r) => r.accession === "0000899140-26-001311")!.form, "4/A");
  assert.equal(recovered.find((r) => r.accession === "0000002488-26-000193")!.form, "4");
  // recovered rows land in the day file of the night they were recovered: nothing is rewritten
  assert.ok(fs.existsSync(path.join(base, "filings", "2026-11-17.jsonl")));
  assert.equal(zlib.gunzipSync(fs.readFileSync(path.join(base, "filings", "2026-11-13.jsonl.gz"))).toString("utf8").split("\n").filter(Boolean).length, 2);
  // the held filing and the non-Form-4 filings were never requested
  for (const never of ["000161071726000453", "000191870426035521", "000119312526458870", "000107261326000890"]) {
    assert.ok(!edgar.asked.some((u) => u.includes(never)), never);
  }

  // every archive Form 4 accession for the day now equals the index's list
  const held = heldParsedAccessions(path.join(base, "filings"), "2026-11-13", "2026-11-17");
  assert.deepEqual(FORM4_ACCESSIONS.filter((a) => !held.has(a)), []);

  // IDEMPOTENT: the next night asks for no filing and appends nothing
  const again = edgarLike({ "2026-11-13": INDEX_1113 });
  const run2 = await runForm4Completeness({ fetch: again.impl, now: () => NIGHT + 86_400_000, baseDir: base, delayMs: 0, append: appendTo });
  assert.equal(run2.totals.backfilled, 0);
  assert.equal(run2.days.find((d) => d.day === "2026-11-13")!.held_before, 4);
  assert.ok(!again.asked.some((u) => u.includes("/Archives/edgar/data/")));
  assert.equal(archiveRows(base).length, rows.length);
});

test("REGRESSION Monday 2026-11-16: a day the poll mostly missed is refilled to the index's count", async () => {
  const base = tmpArchive();
  const lines: string[] = [];
  const all: string[] = [];
  for (let i = 0; i < 120; i++) {
    const acc = `0009999999-26-${String(i).padStart(6, "0")}`;
    all.push(acc);
    lines.push(`4                Issuer ${i}`.padEnd(79) + "1234567     20261116    " + `edgar/data/1234567/${acc}.txt   `);
    lines.push(`4                Owner ${i}`.padEnd(79) + "7654321     20261116    " + `edgar/data/7654321/${acc}.txt   `);
  }
  const index = [...INDEX_HEAD, ...lines].join("\n");
  // the poll stopped at about 07:40 ET: 21 of the day's filings are held
  writeDay(base, "2026-11-16", all.slice(0, 21).map((accession) => ({ accession, issuerCik: "0001234567", filedAt: "2026-11-16" })));
  const edgar = edgarLike({ "2026-11-16": index });
  const run = await runForm4Completeness({ fetch: edgar.impl, now: () => NIGHT, baseDir: base, delayMs: 0, append: appendTo });
  const day = run.days.find((d) => d.day === "2026-11-16")!;
  assert.deepEqual({ index: day.index_form4, held: day.held_before, backfilled: day.backfilled, complete: day.complete }, { index: 120, held: 21, backfilled: 99, complete: true });
  const held = heldParsedAccessions(path.join(base, "filings"), "2026-11-16", "2026-11-17");
  assert.equal(all.filter((a) => held.has(a)).length, 120);
  assert.equal(new Set(archiveRows(base).map((r) => r.accession)).size, 120, "no accession written twice");
});

test("the per-run cap defers the rest to the next night, oldest day first", async () => {
  const base = tmpArchive();
  const edgar = edgarLike({ "2026-11-13": INDEX_1113 });
  const run = await runForm4Completeness({ fetch: edgar.impl, now: () => NIGHT, baseDir: base, delayMs: 0, maxPerRun: 3, append: appendTo });
  const day = run.days.find((d) => d.day === "2026-11-13")!;
  assert.deepEqual({ backfilled: day.backfilled, deferred: day.deferred, complete: day.complete }, { backfilled: 3, deferred: 1, complete: false });
  assert.equal(run.capped, true);
  assert.equal(run.status, "partial");
  const next = await runForm4Completeness({ fetch: edgar.impl, now: () => NIGHT + 86_400_000, baseDir: base, delayMs: 0, maxPerRun: 3, append: appendTo });
  assert.equal(next.days.find((d) => d.day === "2026-11-13")!.complete, true);
  assert.equal(next.status, "ok");
});

test("a document with no issuer is counted unparsed and not written; a failed fetch is retried the next night", async () => {
  const base = tmpArchive();
  let failAmd = true;
  const edgar = edgarLike({ "2026-11-13": INDEX_1113 }, {
    emptyDoc: new Set(["0001610717-26-000453"]),
    filingStatus: (acc) => (acc === "0000002488-26-000193" && failAmd ? 503 : 200),
  });
  const run = await runForm4Completeness({ fetch: edgar.impl, now: () => NIGHT, baseDir: base, delayMs: 0, append: appendTo });
  const day = run.days.find((d) => d.day === "2026-11-13")!;
  assert.deepEqual({ backfilled: day.backfilled, unparsed: day.unparsed, failed: day.failed, complete: day.complete }, { backfilled: 2, unparsed: 1, failed: 1, complete: false });
  assert.equal(run.status, "partial");
  assert.equal(run.refused, false);
  assert.ok(!archiveRows(base).some((r) => r.accession === "0001610717-26-000453"));
  failAmd = false;
  const next = await runForm4Completeness({ fetch: edgar.impl, now: () => NIGHT + 86_400_000, baseDir: base, delayMs: 0, append: appendTo });
  assert.equal(next.days.find((d) => d.day === "2026-11-13")!.backfilled, 1);
});

test("EDGAR refusing filings in a row stops the pass and reads as an error", async () => {
  const base = tmpArchive();
  const lines: string[] = [];
  for (let i = 0; i < 20; i++) lines.push(`4                Issuer ${i}`.padEnd(79) + "1234567     20261113    " + `edgar/data/1234567/0009999999-26-${String(i).padStart(6, "0")}.txt`);
  const edgar = edgarLike({ "2026-11-13": [...INDEX_HEAD, ...lines].join("\n"), "2026-11-16": INDEX_1113 }, { filingStatus: () => 429 });
  const run = await runForm4Completeness({ fetch: edgar.impl, now: () => NIGHT, baseDir: base, delayMs: 0, append: appendTo });
  assert.equal(run.refused, true);
  assert.equal(run.status, "error");
  const d13 = run.days.find((d) => d.day === "2026-11-13")!;
  assert.equal(d13.failed, FORM4_COMPLETENESS_REFUSAL_STOP);
  assert.equal(d13.deferred, 20 - FORM4_COMPLETENESS_REFUSAL_STOP);
  // nothing of the later day was asked for once EDGAR refused
  const d16 = run.days.find((d) => d.day === "2026-11-16")!;
  assert.deepEqual({ failed: d16.failed, deferred: d16.deferred }, { failed: 0, deferred: 4 });
  assert.equal(edgar.asked.filter((u) => u.includes("/Archives/edgar/data/")).length, FORM4_COMPLETENESS_REFUSAL_STOP);
});

test("an index read that fails for another reason is an error for that day, not 'no index'", async () => {
  const base = tmpArchive();
  const impl: CompletenessFetch = async () => resp(500);
  const run = await runForm4Completeness({ fetch: impl, now: () => NIGHT, baseDir: base, delayMs: 0, append: appendTo });
  assert.ok(run.days.every((d) => d.status === "error"));
  assert.equal(run.status, "error");
});

test("the default writer is the poll's own archiveFilings (rows reach <archive>/filings/)", async () => {
  const base = tmpArchive();
  const edgar = edgarLike({ "2026-11-13": INDEX_1113 });
  const run = await runForm4Completeness({ fetch: edgar.impl, now: () => NIGHT, baseDir: base, delayMs: 0 });
  assert.equal(run.totals.backfilled, 4);
  assert.equal(archiveRows(base).length, 4);
});

test("status probe: counts only, state survives a restart, passes the diag scrub unchanged", () => {
  const dir = tmpArchive();
  const env = { DATA_DIR: dir };
  assert.equal((form4CompletenessStatus(env) as { latest: unknown }).latest, null);
  const run = {
    started_at: "2026-11-17T08:00:00.000Z", finished_at: "2026-11-17T08:20:00.000Z", status: "ok" as const, detail: "1 day(s) checked",
    days: [{ day: "2026-11-13", status: "checked" as const, index_form4: 675, held_before: 411, backfilled: 264, unparsed: 0, failed: 0, deferred: 0, complete: true }],
    totals: { index_form4: 675, held_before: 411, backfilled: 264, unparsed: 0, failed: 0, deferred: 0 }, capped: false, refused: false,
  };
  writeCompletenessState(run, path.join(dir, "voltrade_form4_completeness.json"));
  assert.deepEqual(readCompletenessState(path.join(dir, "voltrade_form4_completeness.json")), run);
  const status = form4CompletenessStatus(env);
  assert.deepEqual((status as { latest: unknown }).latest, run);
  assert.deepEqual(sanitizeDiag(status), status);
  assert.ok(!/\d{10}-\d{2}-\d{6}/.test(JSON.stringify(status)), "no accession in the probe");
  assert.ok((DIAG_PROBES as readonly string[]).includes("form4_completeness"));
});

test("wiring: registered job, booted beside the poll, probe served, flag named once per place", () => {
  const spec = JOB_REGISTRY.find((j) => j.id === FORM4_COMPLETENESS_JOB);
  assert.ok(spec && spec.kind === "daily");
  assert.equal(spec.source, "server/edgarForm4Completeness.ts");
  assert.equal(spec.enabledWhen?.({ VOLTRADE_FORM4_COMPLETENESS: "0" }), false);
  assert.equal(spec.enabledWhen?.({}), true);
  const routes = fs.readFileSync(path.join(here, "routes.ts"), "utf8");
  assert.ok(routes.includes("bootForm4Completeness();"));
  const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
  assert.ok(bot.includes('case "form4_completeness"'));
  const manifest = JSON.parse(fs.readFileSync(path.join(here, "..", "datacore", "manifests", "filings.json"), "utf8"));
  assert.ok(manifest.field_map.backfilledAt, "the manifest documents the recovered-row marker");
  assert.ok(manifest.field_map.form);
});
