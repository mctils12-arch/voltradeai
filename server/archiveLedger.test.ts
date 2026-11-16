/**
 * archiveLedger.test.ts — the EDGAR archive writers never re-append a filing
 * an archive day already holds. Regression cases are the days measured live
 * on 2026-11-16 (research/open_questions.md): the 13F Sunday 2026-11-08
 * (40 rows, all already in the 2026-11-06 day file) and the Labor Day triple.
 */
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import { fileURLToPath } from "node:url";
import {
  AccessionLedger, LEDGER_LOOKBACK_DAYS, accessionOfLine, readDayAccessions,
} from "./archiveLedger";
import { archive13FFilings } from "./edgar13f";
import { archiveFilings } from "./edgarForm4";
import { archiveEarnings8Ks } from "./sec8kEarnings";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const at = (day: string, hour = 12): number => Date.parse(`${day}T${String(hour).padStart(2, "0")}:00:00Z`);
const row = (accession: string, filedAt: string | null, extra: object = {}): any => ({ accession, filedAt, cik: "1", indexUrl: "https://www.sec.gov/x", ...extra });
const tmp = (tag: string): string => fs.mkdtempSync(path.join(os.tmpdir(), `vt-ledger-${tag}-`));

/** Writes an archive day file the way an EARLIER process left it. */
function writeDay(dir: string, day: string, rows: any[], gz = false): void {
  fs.mkdirSync(dir, { recursive: true });
  const body = rows.map((r) => JSON.stringify(r)).join("\n") + "\n";
  if (gz) fs.writeFileSync(path.join(dir, `${day}.jsonl.gz`), zlib.gzipSync(body));
  else fs.writeFileSync(path.join(dir, `${day}.jsonl`), body);
}
const linesOf = (dir: string, day: string): string[] => {
  try { return fs.readFileSync(path.join(dir, `${day}.jsonl`), "utf8").split("\n").filter(Boolean); } catch { return []; }
};

test("the 2026-11-08 13F restart: 40 rows filed and archived 11-06 (gzipped since) are still on the feed's page two days later — a fresh process appends none of them", () => {
  const dir = tmp("1108");
  const forty = Array.from({ length: 40 }, (_, i) => row(`0001000000-26-${String(i).padStart(6, "0")}`, "2026-11-06"));
  writeDay(dir, "2026-11-06", forty, true);
  const restarted = new AccessionLedger();
  assert.equal(restarted.filterFresh(dir, forty, at("2026-11-08")).length, 0);
  assert.equal(restarted.stats().repeats_refused, 40);
  // the pre-fix rule (today + yesterday only) would have called all 40 fresh
  assert.equal(readDayAccessions(dir, "2026-11-08").size + readDayAccessions(dir, "2026-11-07").size, 0);
});

test("the Labor Day triple: rows already in 08-29, 08-31 and 09-02 are refused on 09-07; the three filings that really are new are kept, in feed order", () => {
  const dir = tmp("laborday");
  const old = Array.from({ length: 40 }, (_, i) => row(`LD-${i}`, "2026-08-28"));
  writeDay(dir, "2026-08-29", old, true);
  writeDay(dir, "2026-08-31", old, true);
  writeDay(dir, "2026-09-02", old, true);
  const fresh3 = [row("LD-NEW-1", "2026-09-04"), row("LD-NEW-2", "2026-09-04"), row("LD-NEW-3", "2026-09-07")];
  const feed = [fresh3[0], ...old.slice(0, 20), fresh3[1], ...old.slice(20), fresh3[2]];
  const out = new AccessionLedger().filterFresh(dir, feed, at("2026-09-07"));
  assert.deepEqual(out.map((r) => r.accession), ["LD-NEW-1", "LD-NEW-2", "LD-NEW-3"]);
});

test("after a 3-day outage: a filing filed 11-10 but first archived late (11-13) is found by a process restarted on 11-15 — the check walks from filedAt forward, not back from today by a fixed two days", () => {
  const dir = tmp("outage");
  writeDay(dir, "2026-11-13", [row("LATE-1", "2026-11-10"), row("LATE-2", "2026-11-10")], true);
  const led = new AccessionLedger();
  assert.deepEqual(led.filterFresh(dir, [row("LATE-1", "2026-11-10"), row("LATE-2", "2026-11-10"), row("NEW-1", "2026-11-15")], at("2026-11-15")).map((r) => r.accession), ["NEW-1"]);
});

test("a genuinely new filing is appended once; the same accession twice in one batch is one row; rows with no accession are dropped", () => {
  const dir = tmp("new");
  const led = new AccessionLedger();
  const out = led.filterFresh(dir, [row("N-1", "2026-11-16"), row("N-1", "2026-11-16"), row("", "2026-11-16"), row("N-2", null)], at("2026-11-16"));
  assert.deepEqual(out.map((r) => r.accession), ["N-1", "N-2"]);
});

test("only the day files a candidate's own filedAt makes relevant are read, each once per process", () => {
  const dir = tmp("lazy");
  for (let d = 1; d <= 16; d++) writeDay(dir, `2026-11-${String(d).padStart(2, "0")}`, [row(`D-${d}`, `2026-11-${String(d).padStart(2, "0")}`)], d < 15);
  const led = new AccessionLedger();
  const now = at("2026-11-16");
  assert.equal(led.filterFresh(dir, [row("X-1", "2026-11-16"), row("X-2", "2026-11-16")], now).length, 2);
  assert.equal(led.stats().day_reads, 1, "candidates filed today read today's file only");
  assert.equal(led.filterFresh(dir, [row("D-13", "2026-11-13"), row("X-3", "2026-11-13")], now).length, 1);
  assert.equal(led.stats().day_reads, 4, "11-15, 11-14 and 11-13 read once more; 11-16 not re-read");
  led.filterFresh(dir, [row("X-3", "2026-11-13"), row("D-14", "2026-11-13")], now);
  assert.equal(led.stats().day_reads, 4, "nothing is read twice");
});

test("today's file appended by ANOTHER process (two containers overlapping during a deploy) is re-read before the next filter", () => {
  const dir = tmp("overlap");
  const now = at("2026-11-16");
  const led = new AccessionLedger();
  assert.equal(led.filterFresh(dir, [row("O-1", "2026-11-16")], now).length, 1);
  const body = JSON.stringify(row("O-1", "2026-11-16")) + "\n";
  fs.mkdirSync(dir, { recursive: true });
  fs.appendFileSync(path.join(dir, "2026-11-16.jsonl"), body);
  led.recordAppended(dir, [row("O-1", "2026-11-16")], now, Buffer.byteLength(body));
  // the other container archives O-2 into the same day file
  fs.appendFileSync(path.join(dir, "2026-11-16.jsonl"), JSON.stringify(row("O-2", "2026-11-16")) + "\n");
  assert.deepEqual(led.filterFresh(dir, [row("O-1", "2026-11-16"), row("O-2", "2026-11-16"), row("O-3", "2026-11-16")], now + 60_000).map((r) => r.accession), ["O-3"]);
  assert.equal(led.stats().today_rereads, 1);
  // and an append of ours that lands together with a foreign one is noticed too
  const mine = JSON.stringify(row("O-3", "2026-11-16")) + "\n";
  fs.appendFileSync(path.join(dir, "2026-11-16.jsonl"), mine + JSON.stringify(row("O-4", "2026-11-16")) + "\n");
  led.recordAppended(dir, [row("O-3", "2026-11-16")], now + 60_000, Buffer.byteLength(mine));
  assert.equal(led.filterFresh(dir, [row("O-4", "2026-11-16")], now + 120_000).length, 0);
});

test("a candidate filed before the lookback window (the 2020/2021 releases the feed served on 2026-11-15) is counted, refused if the window holds it, and still archived if it does not — the cap is an I/O bound, not an age policy", () => {
  const dir = tmp("old");
  writeDay(dir, "2026-11-15", [row("OLD-2020", "2020-07-09")]);
  const led = new AccessionLedger();
  const out = led.filterFresh(dir, [row("OLD-2020", "2020-07-09"), row("OLD-2021", "2021-02-02")], at("2026-11-16"));
  assert.deepEqual(out.map((r) => r.accession), ["OLD-2021"]);
  assert.equal(led.stats().beyond_lookback, 2);
  assert.ok(led.stats().day_reads <= LEDGER_LOOKBACK_DAYS + 1, "one bounded window scan, shared by both candidates");
});

test("no clear(): a process that runs for weeks drops day sets outside the window and still refuses a repeat inside it (the disk is the truth, a dropped day is read again)", () => {
  const dir = tmp("long");
  const led = new AccessionLedger(3);
  for (let d = 1; d <= 12; d++) {
    const day = `2026-10-${String(d).padStart(2, "0")}`;
    const rows = Array.from({ length: 50 }, (_, i) => row(`L-${d}-${i}`, day));
    const fresh = led.filterFresh(dir, rows, at(day));
    assert.equal(fresh.length, 50);
    const body = fresh.map((r) => JSON.stringify(r)).join("\n") + "\n";
    fs.mkdirSync(dir, { recursive: true });
    fs.appendFileSync(path.join(dir, `${day}.jsonl`), body);
    led.recordAppended(dir, fresh, at(day), Buffer.byteLength(body));
  }
  assert.ok(led.stats().days_loaded <= 4 && led.size <= 200, `bounded: ${JSON.stringify(led.stats())} size ${led.size}`);
  assert.equal(led.filterFresh(dir, [row("L-10-7", "2026-10-10"), row("L-12-0", "2026-10-12")], at("2026-10-12", 18)).length, 0);
});

test("day files: plain and gzipped are both read, blank and corrupt lines are skipped, and an escaped \"accession\" inside a text field is never taken for the row's own", () => {
  const dir = tmp("lines");
  fs.mkdirSync(dir, { recursive: true });
  const tricky = JSON.stringify({ text: 'he said "accession":"FAKE-1" twice', accession: "REAL-1", filedAt: "2026-11-10" });
  fs.writeFileSync(path.join(dir, "2026-11-10.jsonl"), [tricky, "", "{not json", JSON.stringify({ accession: 'ESC-"1' })].join("\n") + "\n");
  fs.writeFileSync(path.join(dir, "2026-11-10.jsonl.gz"), zlib.gzipSync(JSON.stringify(row("GZ-1", "2026-11-10")) + "\n"));
  assert.deepEqual(Array.from(readDayAccessions(dir, "2026-11-10")).sort(), ['ESC-"1', "GZ-1", "REAL-1"]);
  assert.equal(accessionOfLine('{"accession":""}'), null);
  assert.equal(readDayAccessions(dir, "2026-11-11").size, 0, "a missing day is empty, not an error");
});

test("through the real writers: a filing an earlier process archived two or more days ago is not appended again (Form 4, 13F, 8-K), and a new one is", () => {
  const cases: Array<[string, string, (rows: any[], base: string, now: number) => number]> = [
    ["filings", "F4", archiveFilings as any],
    ["filings13f", "13F", archive13FFilings as any],
    ["earnings8k", "8K", archiveEarnings8Ks as any],
  ];
  for (const [sub, tag, write] of cases) {
    const base = tmp(`writer-${tag}`);
    const dir = path.join(base, sub);
    const held = [row(`W-${tag}-1`, "2026-11-06"), row(`W-${tag}-2`, "2026-11-06")];
    writeDay(dir, "2026-11-06", held, true);
    const now = at("2026-11-08");
    assert.equal(write(held, base, now), 0, `${tag}: nothing re-appended`);
    assert.equal(fs.existsSync(path.join(dir, "2026-11-08.jsonl")), false, `${tag}: no day file created for repeats`);
    assert.equal(write([...held, row(`W-${tag}-3`, "2026-11-08")], base, now + 1000), 1, `${tag}: the new filing is appended`);
    assert.equal(write([row(`W-${tag}-3`, "2026-11-08")], base, now + 2000), 0);
    assert.deepEqual(linesOf(dir, "2026-11-08").map((l) => JSON.parse(l).accession), [`W-${tag}-3`]);
    assert.deepEqual(readDayAccessions(dir, "2026-11-06").size, 2, `${tag}: the older day file is untouched`);
  }
});

test("ratchet: the three EDGAR writers keep no two-day seed and no clear() — each asks the shared ledger", () => {
  for (const f of ["edgarForm4.ts", "edgar13f.ts", "sec8kEarnings.ts"]) {
    const src = fs.readFileSync(path.join(HERE, f), "utf8");
    assert.ok(!/function seedSeen/.test(src), `${f}: seedSeen is gone`);
    assert.ok(!/archivedAccessions\.clear\(\)/.test(src), `${f}: no clear()`);
    assert.ok(/archivedAccessions\.filterFresh\(dir, filings, now\)/.test(src), `${f}: filters through the ledger`);
    assert.ok(/archivedAccessions\.recordAppended\(/.test(src), `${f}: records its own append`);
  }
});
