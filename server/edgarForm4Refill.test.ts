// The older-day refill of the Form 4 archive and its per-day ledger
// (server/edgarForm4Completeness.ts, runForm4Refill).
//
// MEASURED 2026-11-17 against EDGAR's daily form index: the archive held 576
// of the 1,227 Form 4s filed 2026-07-06 (47%) and 63 of the 511 filed
// 2026-09-15 (12%; production was down 09-10..09-16). The nightly pass
// reached 7 days back, so those days could never be completed. The index for
// Columbus Day 2026-10-12 answers 403 (EDGAR closed).
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import os from "os";
import path from "path";
import zlib from "zlib";
import {
  FORM4_REFILL_MAX_ATTEMPTS,
  FORM4_REFILL_NO_INDEX_STOP,
  form4CompletenessStatus,
  form4RefillEnabled,
  heldParsedAccessionsAll,
  ledgerSummary,
  noteLedgerDay,
  noteRecentLedgerDay,
  olderDays,
  readDayLedger,
  runForm4Night,
  runForm4Refill,
  writeDayLedger,
  type CompletenessFetch,
  type DayLedger,
} from "./edgarForm4Completeness";
import { FORM4_ARCHIVE_START } from "./edgarForm4";
import { sanitizeDiag } from "./diag";

const T = (iso: string) => Date.parse(iso);
// Tuesday 2026-11-17 03:00 ET.
const NIGHT = T("2026-11-17T08:00:00Z");
// Later runs keep the same ET date: a day later the older range itself grows
// by one weekday (the day that just left the recent pass's reach).
const LATER = 60_000;

const accOf = (day: string, i: number) => `000${day.slice(5, 7)}${day.slice(8, 10)}999-26-${String(i).padStart(6, "0")}`;
/** A daily index in EDGAR's own column layout with `n` Form 4s (two filer
 *  lines each) and one 424B2 look-alike. */
function indexFor(day: string, n: number): string {
  const ymd = day.replace(/-/g, "");
  const lines = ["Form Type   Company Name                                                  CIK", "---"];
  for (let i = 0; i < n; i++) {
    const acc = accOf(day, i);
    lines.push(`4                Issuer ${i}`.padEnd(79) + `1234567     ${ymd}    edgar/data/1234567/${acc}.txt   `);
    lines.push(`4                Owner ${i}`.padEnd(79) + `7654321     ${ymd}    edgar/data/7654321/${acc}.txt   `);
  }
  lines.push("424B2            BANK OF AMERICA CORP /DE/".padEnd(79) + `70858       ${ymd}    edgar/data/70858/0001918704-26-035521.txt   `);
  return lines.join("\n") + "\n";
}

const OWNERSHIP = `<?xml version="1.0"?><ownershipDocument><documentType>4</documentType><periodOfReport>2026-07-01</periodOfReport>
<issuer><issuerCik>0001234567</issuerCik><issuerName>ISSUER</issuerName><issuerTradingSymbol>TST</issuerTradingSymbol></issuer>
<reportingOwner><reportingOwnerId><rptOwnerCik>0007654321</rptOwnerCik><rptOwnerName>Owner</rptOwnerName></reportingOwnerId></reportingOwner>
</ownershipDocument>`;

const resp = (status: number, body = "") => ({ ok: status >= 200 && status < 300, status, text: async () => body });

function edgarLike(indexes: Record<string, string>, opts: { filingStatus?: (acc: string) => number; indexStatus?: (day: string) => number } = {}) {
  const asked: string[] = [];
  const impl: CompletenessFetch = async (url) => {
    asked.push(url);
    const idx = /form\.(\d{4})(\d{2})(\d{2})\.idx$/.exec(url);
    if (idx) {
      const day = `${idx[1]}-${idx[2]}-${idx[3]}`;
      const forced = opts.indexStatus?.(day);
      if (forced && forced !== 200) return resp(forced);
      return indexes[day] === undefined ? resp(403, "no such file") : resp(200, indexes[day]);
    }
    const m = /\/Archives\/edgar\/data\/\d+\/(\d{18})\/(.+)$/.exec(url);
    if (!m) return resp(404);
    const acc = `${m[1].slice(0, 10)}-${m[1].slice(10, 12)}-${m[1].slice(12)}`;
    const status = opts.filingStatus?.(acc) ?? 200;
    if (status !== 200) return resp(status);
    if (m[2] === "index.json") return resp(200, JSON.stringify({ directory: { item: [{ name: "wk-form4_1.xml" }] } }));
    return resp(200, OWNERSHIP);
  };
  const indexReads = () => asked.filter((u) => u.endsWith(".idx")).length;
  const filingFetches = () => asked.filter((u) => u.endsWith("index.json")).length;
  return { impl, asked, indexReads, filingFetches };
}

function tmp(): string { return fs.mkdtempSync(path.join(os.tmpdir(), "form4-refill-")); }
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
const appendTo = (rows: object[], baseDir?: string, nowMs?: number): number => {
  const dir = path.join(baseDir as string, "filings");
  fs.mkdirSync(dir, { recursive: true });
  fs.appendFileSync(path.join(dir, `${new Date(nowMs as number).toISOString().slice(0, 10)}.jsonl`), rows.map((r) => JSON.stringify(r)).join("\n") + "\n");
  return rows.length;
};
const freshLedger = (archiveStart = FORM4_ARCHIVE_START): DayLedger => ({ archive_start: archiveStart, updated_at: "", days: {} });
const held = (day: string, i: number) => ({ accession: accOf(day, i), issuerCik: "0001234567", filedAt: day });

test("olderDays: weekdays from the archive's first day to the edge of the 7-day reach, newest first", () => {
  const days = olderDays(NIGHT);
  // the recent pass covers 11-10..11-16; the first older weekday is Monday 11-09
  assert.equal(days[0], "2026-11-09");
  assert.equal(days.at(-1), "2026-07-06", "2026-07-04 is a Saturday: the first weekday is Monday 07-06");
  assert.equal(days.length, 91);
  assert.ok(days.every((d) => ![0, 6].includes(new Date(`${d}T00:00:00Z`).getUTCDay())));
  assert.ok(!days.includes("2026-11-10"), "no day is in both passes");
  assert.deepEqual(olderDays(NIGHT, "2026-11-05"), ["2026-11-09", "2026-11-06", "2026-11-05"]);
  assert.deepEqual(olderDays(NIGHT, "2026-11-12"), []);
});

test("REGRESSION 2026-09-15 (63 of 511 held): an older day is refilled to the index's count and settled", async () => {
  const base = tmp();
  const day = "2026-09-15";
  // the poll's rows sit in the filed day's own (gzipped) file and the next day's
  writeDay(base, day, Array.from({ length: 40 }, (_, i) => held(day, i)), true);
  writeDay(base, "2026-09-16", Array.from({ length: 23 }, (_, i) => held(day, 40 + i)), true);
  const edgar = edgarLike({ [day]: indexFor(day, 511) });
  const ledger = freshLedger(day);
  const run = await runForm4Refill({ fetch: edgar.impl, now: () => NIGHT, baseDir: base, delayMs: 0, append: appendTo, ledger, archiveStart: day, lookbackDays: 61 });
  const rec = run.days.find((d) => d.day === day)!;
  assert.deepEqual({ index: rec.index_form4, held: rec.held_before, backfilled: rec.backfilled, complete: rec.complete }, { index: 511, held: 63, backfilled: 448, complete: true });
  assert.equal(edgar.filingFetches(), 448, "only the missing filings are requested");
  const rows = archiveRows(base);
  assert.equal(new Set(rows.map((r) => r.accession)).size, 511);
  assert.equal(rows.length, 511, "no accession written twice");
  const recovered = rows.filter((r) => r.backfilledAt);
  assert.equal(recovered.length, 448);
  assert.ok(recovered.every((r) => r.filedAt === day && r.form === "4" && r.backfilledAt === "2026-11-17T08:00:00.000Z"));
  // append-only: the recovered rows are in the night's own day file
  assert.ok(fs.existsSync(path.join(base, "filings", "2026-11-17.jsonl")));
  assert.equal(zlib.gunzipSync(fs.readFileSync(path.join(base, "filings", `${day}.jsonl.gz`))).toString("utf8").trim().split("\n").length, 40);
  assert.deepEqual(
    { ...ledger.days[day], checked_at: undefined },
    { day, status: "checked", index_form4: 511, held: 511, unparsed: 0, failed: 0, deferred: 0, complete: true, settled: true, attempts: 1, checked_at: undefined },
  );
});

test("a settled day is never read again: the second night asks EDGAR for nothing and reads as finished", async () => {
  const base = tmp();
  const edgar = edgarLike({ "2026-11-09": indexFor("2026-11-09", 5), "2026-11-06": indexFor("2026-11-06", 3) });
  const ledger = freshLedger("2026-11-06");
  const first = await runForm4Refill({ fetch: edgar.impl, now: () => NIGHT, baseDir: base, delayMs: 0, append: appendTo, ledger });
  assert.equal(first.status, "finished");
  assert.equal(first.totals.backfilled, 8);
  const before = edgar.asked.length;
  const second = await runForm4Refill({ fetch: edgar.impl, now: () => NIGHT + LATER, baseDir: base, delayMs: 0, append: appendTo, ledger });
  assert.equal(edgar.asked.length, before, "no index and no filing is requested for settled days");
  assert.deepEqual({ status: second.status, days: second.days.length, remaining: second.remaining_days }, { status: "finished", days: 0, remaining: 0 });
  assert.equal(archiveRows(base).length, 8);
});

test("the per-run cap stops the night; the next night continues with what is left, newest day first", async () => {
  const base = tmp();
  const edgar = edgarLike({ "2026-11-09": indexFor("2026-11-09", 6), "2026-11-06": indexFor("2026-11-06", 6), "2026-11-05": indexFor("2026-11-05", 6) });
  const ledger = freshLedger("2026-11-05");
  const opts = { fetch: edgar.impl, baseDir: base, delayMs: 0, append: appendTo, ledger, maxPerRun: 8 };
  const n1 = await runForm4Refill({ ...opts, now: () => NIGHT });
  assert.deepEqual(n1.days.map((d) => [d.day, d.backfilled, d.deferred]), [["2026-11-09", 6, 0], ["2026-11-06", 2, 4]]);
  assert.deepEqual({ status: n1.status, capped: n1.capped, remaining: n1.remaining_days }, { status: "partial", capped: true, remaining: 2 });
  assert.equal(ledger.days["2026-11-06"].settled, false);
  assert.equal(ledger.days["2026-11-05"], undefined, "a day the cap never reached is not in the ledger");
  const n2 = await runForm4Refill({ ...opts, now: () => NIGHT + LATER });
  assert.deepEqual(n2.days.map((d) => [d.day, d.held_before, d.backfilled]), [["2026-11-06", 2, 4], ["2026-11-05", 0, 4]]);
  const n3 = await runForm4Refill({ ...opts, now: () => NIGHT + 2 * LATER });
  assert.equal(n3.status, "finished");
  const rows = archiveRows(base);
  assert.equal(rows.length, 18);
  assert.equal(new Set(rows.map((r) => r.accession)).size, 18, "no filing fetched or written twice across nights");
  assert.equal(edgar.filingFetches(), 18);
});

test("a filing that keeps failing is retried for three nights, then the day is settled short with the failure stated", async () => {
  const base = tmp();
  const day = "2026-11-09";
  const bad = accOf(day, 2);
  const edgar = edgarLike({ [day]: indexFor(day, 4) }, { filingStatus: (acc) => (acc === bad ? 404 : 200) });
  const ledger = freshLedger(day);
  const opts = { fetch: edgar.impl, baseDir: base, delayMs: 0, append: appendTo, ledger };
  for (let n = 0; n < FORM4_REFILL_MAX_ATTEMPTS; n++) {
    const run = await runForm4Refill({ ...opts, now: () => NIGHT + n * LATER });
    assert.equal(run.days[0].failed, 1);
    assert.equal(ledger.days[day].settled, n === FORM4_REFILL_MAX_ATTEMPTS - 1);
  }
  assert.deepEqual(
    { held: ledger.days[day].held, index: ledger.days[day].index_form4, complete: ledger.days[day].complete, failed: ledger.days[day].failed, attempts: ledger.days[day].attempts },
    { held: 3, index: 4, complete: false, failed: 1, attempts: 3 },
  );
  const before = edgar.asked.length;
  const after = await runForm4Refill({ ...opts, now: () => NIGHT + 3 * LATER });
  assert.equal(edgar.asked.length, before);
  assert.equal(after.status, "finished");
  const summary = ledgerSummary(ledger, NIGHT) as Record<string, unknown>;
  assert.equal(summary.settled_short_days, 1, "a settled day that is not complete is counted, never called complete");
  assert.equal(summary.complete_days, 0);
});

test("a lone day without an index is a holiday (Columbus Day 2026-10-12 answers 403) and does not stop the walk", async () => {
  const base = tmp();
  const edgar = edgarLike({ "2026-10-13": indexFor("2026-10-13", 2), "2026-10-09": indexFor("2026-10-09", 2) });
  const ledger = freshLedger("2026-10-09");
  // a clock on Thursday 10-22 puts 10-13 at the top of the older days
  const run = await runForm4Refill({ fetch: edgar.impl, now: () => T("2026-10-22T08:00:00Z"), baseDir: base, delayMs: 0, append: appendTo, ledger });
  assert.deepEqual(run.days.map((d) => [d.day, d.status]), [["2026-10-14", "no_index"], ["2026-10-13", "checked"], ["2026-10-12", "no_index"], ["2026-10-09", "checked"]]);
  assert.equal(run.refused, false);
  assert.equal(run.totals.backfilled, 4);
  assert.deepEqual({ status: ledger.days["2026-10-12"].status, settled: ledger.days["2026-10-12"].settled, attempts: ledger.days["2026-10-12"].attempts }, { status: "no_index", settled: false, attempts: 1 });
});

test("several weekdays in a row without an index is EDGAR refusing: the run is an error and no day is recorded as a holiday", async () => {
  const base = tmp();
  const edgar = edgarLike({}, { indexStatus: () => 403 });
  const ledger = freshLedger();
  const run = await runForm4Refill({ fetch: edgar.impl, now: () => NIGHT, baseDir: base, delayMs: 0, append: appendTo, ledger });
  assert.deepEqual({ status: run.status, refused: run.refused }, { status: "error", refused: true });
  assert.equal(edgar.indexReads(), FORM4_REFILL_NO_INDEX_STOP, "the walk stops instead of asking for 91 indexes");
  assert.deepEqual(ledger.days, {});
});

test("filings refused in a row (403/429) stop the refill; what was appended stays and the day is not settled", async () => {
  const base = tmp();
  const day = "2026-11-09";
  let n = 0;
  const edgar = edgarLike({ [day]: indexFor(day, 30) }, { filingStatus: () => (++n > 4 ? 429 : 200) });
  const ledger = freshLedger(day);
  const run = await runForm4Refill({ fetch: edgar.impl, now: () => NIGHT, baseDir: base, delayMs: 0, append: appendTo, ledger });
  assert.deepEqual({ status: run.status, refused: run.refused }, { status: "error", refused: true });
  assert.equal(archiveRows(base).length, 2, "two filings (four requests) were served before the refusals");
  assert.equal(ledger.days[day].settled, false);
  assert.ok(ledger.days[day].deferred > 0);
});

test("an index read that fails for another reason spends no retry and settles nothing", async () => {
  const base = tmp();
  const edgar = edgarLike({ "2026-11-06": indexFor("2026-11-06", 1) }, { indexStatus: (d) => (d === "2026-11-09" ? 503 : 200) });
  const ledger = freshLedger("2026-11-06");
  const run = await runForm4Refill({ fetch: edgar.impl, now: () => NIGHT, baseDir: base, delayMs: 0, append: appendTo, ledger });
  assert.deepEqual(run.days.map((d) => d.status), ["error", "checked"]);
  assert.equal(run.status, "partial");
  assert.equal(ledger.days["2026-11-09"], undefined);
  assert.equal(run.remaining_days, 1);
});

test("heldParsedAccessionsAll unions every day file; a row recovered on a later night counts, an issuer-less shell does not", async () => {
  const base = tmp();
  writeDay(base, "2026-07-06", [held("2026-07-06", 0), { accession: "0000000000-26-000001", issuerCik: null }], true);
  writeDay(base, "2026-11-17", [{ ...held("2026-07-06", 1), backfilledAt: "2026-11-17T08:00:00.000Z" }]);
  fs.writeFileSync(path.join(base, "filings", "notes.txt"), "not a day file");
  const got = await heldParsedAccessionsAll(path.join(base, "filings"));
  assert.deepEqual(Array.from(got).sort(), [accOf("2026-07-06", 0), accOf("2026-07-06", 1)]);
  assert.equal((await heldParsedAccessionsAll(path.join(base, "nowhere"))).size, 0);
});

test("ledger notes: a recent day settles only when complete and spends no retry; an older day does", () => {
  const ledger = freshLedger();
  const short = { day: "2026-11-12", status: "checked" as const, index_form4: 841, held_before: 449, backfilled: 391, unparsed: 0, failed: 1, deferred: 0, complete: false };
  // measured: the first live night left 2026-11-12 one filing short
  assert.deepEqual({ ...noteRecentLedgerDay(ledger, short, NIGHT), checked_at: "" }, { day: "2026-11-12", status: "checked", index_form4: 841, held: 840, unparsed: 0, failed: 1, deferred: 0, complete: false, settled: false, attempts: 0, checked_at: "" });
  assert.equal(noteRecentLedgerDay(ledger, { ...short, backfilled: 392, failed: 0, complete: true }, NIGHT).settled, true);
  assert.equal(noteRecentLedgerDay(ledger, { ...short, status: "no_index", index_form4: 0, held_before: 0, backfilled: 0, failed: 0 }, NIGHT).settled, false);
  // older: an unparsed document is tried once and the day is settled short
  const unparsed = noteLedgerDay(ledger, { ...short, day: "2026-08-03", failed: 0, unparsed: 1 }, NIGHT);
  assert.deepEqual({ settled: unparsed.settled, complete: unparsed.complete, attempts: unparsed.attempts }, { settled: true, complete: false, attempts: 1 });
  // older: a deferred remainder is never settled, however many nights
  for (let i = 0; i < 5; i++) noteLedgerDay(ledger, { ...short, day: "2026-08-04", failed: 0, deferred: 10 }, NIGHT);
  assert.equal(ledger.days["2026-08-04"].settled, false);
  // older: a holiday settles after the retries
  for (let i = 0; i < FORM4_REFILL_MAX_ATTEMPTS; i++) noteLedgerDay(ledger, { ...short, day: "2026-09-07", status: "no_index", index_form4: 0, held_before: 0, backfilled: 0, failed: 0 }, NIGHT);
  assert.equal(ledger.days["2026-09-07"].settled, true);
});

test("ledger file: survives a restart; an unreadable file reads as empty (days are re-read, never assumed complete)", () => {
  const dir = tmp();
  const file = path.join(dir, "voltrade_form4_day_ledger.json");
  assert.deepEqual(readDayLedger(file).days, {});
  const ledger = freshLedger();
  noteLedgerDay(ledger, { day: "2026-11-09", status: "checked", index_form4: 5, held_before: 2, backfilled: 3, unparsed: 0, failed: 0, deferred: 0, complete: true }, NIGHT);
  writeDayLedger(ledger, file);
  assert.deepEqual(readDayLedger(file), ledger);
  fs.writeFileSync(file, "{torn");
  assert.deepEqual(readDayLedger(file).days, {});
});

test("ledgerSummary: complete_back_to is the oldest day of the unbroken complete run from yesterday", () => {
  const ledger = freshLedger();
  const ok = (day: string) => noteLedgerDay(ledger, { day, status: "checked", index_form4: 10, held_before: 4, backfilled: 6, unparsed: 0, failed: 0, deferred: 0, complete: true }, NIGHT);
  for (const d of ["2026-11-16", "2026-11-13", "2026-11-12", "2026-11-10", "2026-11-09", "2026-11-05"]) ok(d);
  // Veterans Day: closed, settled after the retries
  for (let i = 0; i < 3; i++) noteLedgerDay(ledger, { day: "2026-11-11", status: "no_index", index_form4: 0, held_before: 0, backfilled: 0, unparsed: 0, failed: 0, deferred: 0, complete: false }, NIGHT);
  const s = ledgerSummary(ledger, NIGHT) as Record<string, unknown>;
  // 11-06 is not in the ledger, so the run stops at 11-09 although 11-05 is complete
  assert.equal(s.complete_back_to, "2026-11-09");
  assert.deepEqual({ weekdays: s.weekdays, complete: s.complete_days, closed: s.closed_days, open: s.not_settled_days, index: s.index_form4_checked, held: s.held_checked }, { weekdays: 96, complete: 6, closed: 1, open: 89, index: 60, held: 60 });
  assert.equal((ledgerSummary(freshLedger(), NIGHT) as Record<string, unknown>).complete_back_to, null);
});

test("one night: recent days first, then older days; the ledger and the probe carry both; VOLTRADE_FORM4_REFILL=0 turns only the older days off", async () => {
  assert.equal(form4RefillEnabled({}), true);
  assert.equal(form4RefillEnabled({ VOLTRADE_FORM4_REFILL: "0" }), false);
  const base = tmp();
  const dataDir = tmp();
  const env = { DATA_DIR: dataDir };
  const indexes = { "2026-11-16": indexFor("2026-11-16", 3), "2026-11-09": indexFor("2026-11-09", 2), "2026-11-06": indexFor("2026-11-06", 2) };
  const edgar = edgarLike(indexes);
  const off = await runForm4Night({ fetch: edgar.impl, now: () => NIGHT, baseDir: base, delayMs: 0, append: appendTo }, { ...env, VOLTRADE_FORM4_REFILL: "0" });
  assert.equal(off.refill, null);
  assert.equal(off.run.totals.backfilled, 3);
  assert.ok(!edgar.asked.some((u) => u.includes("form.20261109.idx")), "no older index is read with the refill off");
  const on = await runForm4Night({ fetch: edgar.impl, now: () => NIGHT, baseDir: base, delayMs: 0, append: appendTo, refillMaxPerRun: 3 }, env);
  assert.equal(on.run.totals.backfilled, 0, "the recent day is already held");
  assert.deepEqual({ backfilled: on.refill?.totals.backfilled, capped: on.refill?.capped }, { backfilled: 3, capped: true });
  assert.equal(on.failed, false);
  assert.match(on.detail, /day\(s\) checked.* \| older days: 2 read, backfilled 3/);
  const ledger = readDayLedger(path.join(dataDir, "voltrade_form4_day_ledger.json"));
  assert.equal(ledger.days["2026-11-16"].complete, true);
  assert.equal(ledger.days["2026-11-09"].settled, true);
  assert.equal(ledger.days["2026-11-06"].deferred, 1);
  const status = form4CompletenessStatus(env) as { refill: { enabled: boolean; ledger: Record<string, unknown>; latest: { totals: { backfilled: number } } } };
  assert.equal(status.refill.enabled, true);
  assert.equal(status.refill.latest.totals.backfilled, 3);
  assert.equal(status.refill.ledger.complete_days, 2);
  assert.deepEqual(sanitizeDiag(status), status);
  assert.ok(!/\d{10}-\d{2}-\d{6}/.test(JSON.stringify(status)), "no accession in the probe");
});

test("a refused recent pass does not start the older days", async () => {
  const base = tmp();
  const env = { DATA_DIR: tmp() };
  const edgar = edgarLike({ "2026-11-16": indexFor("2026-11-16", 20), "2026-11-09": indexFor("2026-11-09", 2) }, { filingStatus: () => 403 });
  const night = await runForm4Night({ fetch: edgar.impl, now: () => NIGHT, baseDir: base, delayMs: 0, append: appendTo }, env);
  assert.equal(night.run.refused, true);
  assert.equal(night.refill, null);
  assert.equal(night.failed, true);
  assert.ok(!edgar.asked.some((u) => u.includes("form.20261109.idx")));
});

test("a day leaving the 7-day reach: complete in the ledger is not read again; left one filing short is picked up by the older days", async () => {
  const base = tmp();
  const env = { DATA_DIR: tmp() };
  const bad = accOf("2026-11-10", 1);
  let serveBad = false;
  const edgar = edgarLike(
    { "2026-11-10": indexFor("2026-11-10", 3), "2026-11-12": indexFor("2026-11-12", 2) },
    { filingStatus: (acc) => (acc === bad && !serveBad ? 500 : 200) },
  );
  // Tuesday night: both days are inside the recent pass; 11-10 ends one short
  const n1 = await runForm4Night({ fetch: edgar.impl, now: () => NIGHT, baseDir: base, delayMs: 0, append: appendTo }, { ...env, VOLTRADE_FORM4_REFILL: "0" });
  assert.equal(n1.run.days.find((d) => d.day === "2026-11-10")?.failed, 1);
  // Thursday night 11-19: 11-10 and 11-11 have left the reach, 11-12 has not
  serveBad = true;
  const thursday = T("2026-11-19T08:00:00Z");
  const before = edgar.asked.length;
  const ledger = readDayLedger(path.join(env.DATA_DIR, "voltrade_form4_day_ledger.json"));
  const refill = await runForm4Refill({ fetch: edgar.impl, now: () => thursday, baseDir: base, delayMs: 0, append: appendTo, ledger, archiveStart: "2026-11-10" });
  const asked = edgar.asked.slice(before);
  assert.ok(asked.some((u) => u.includes("form.20261110.idx")));
  assert.ok(!asked.some((u) => u.includes("form.20261112.idx")), "11-12 is still the recent pass's day");
  assert.deepEqual(refill.days.filter((d) => d.status === "checked").map((d) => [d.day, d.held_before, d.backfilled, d.complete]), [["2026-11-10", 2, 1, true]]);
  assert.equal(ledger.days["2026-11-10"].settled, true);
});
