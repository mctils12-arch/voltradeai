// filingSeen.test.ts — [REPAIR] 2026-11-16: the Form 4, 13F and 8-K archive
// writers ask the day files whether a filing is already archived, from the
// filing's own filed date forward. Before, each rebuilt its seen set after a
// restart from today's and yesterday's files only, so every restart
// re-appended whatever was still on EDGAR's feed page and two or more days
// old (measured live 2026-11-16: 776 of 8,446 13F rows over 60 days).
// The cases below are the measured ones (research/open_questions.md
// 2026-11-16). A "restart" is the writer's seen state dropped while the files
// stay, which is what a new container sees.
// Run: npx tsx --test server/filingSeen.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import { accessionsInBuffer, createFilingSeen, FILING_SEEN_LOOKBACK_DAYS } from "./filingSeen";
import { archive13FFilings, gzipOld13FDays, _filings13fSeenForTests, type Filing13F } from "./edgar13f";
import { archiveFilings, gzipOldFilingDays, _filingsSeenForTests, type Form4Filing } from "./edgarForm4";
import { archiveEarnings8Ks, _earnings8kSeenForTests, type Earnings8K } from "./sec8kEarnings";

const HERE = path.dirname(new URL(import.meta.url).pathname);
const tmp = (p: string) => fs.mkdtempSync(path.join(os.tmpdir(), p));
const at = (day: string, hour = 12) => Date.parse(`${day}T${String(hour).padStart(2, "0")}:00:00Z`);

const mk13F = (acc: string, filedAt: string): Filing13F => ({
  managerCik: "0001762716", managerName: "TEST MANAGER LLC",
  periodOfReport: "2026-09-30", submissionType: "13F-HR",
  entryTotal: 1, valueTotal: 1000, otherManagersCount: 0,
  accession: acc, filedAt, cik: "0001762716",
  indexUrl: `https://www.sec.gov/x/${acc}/`,
  holdings: [{
    issuer: 'SAYS "accession":"FAKE-IN-A-NAME"', titleOfClass: "COM", cusip: "000000000",
    value: 1, shares: 1, sharesType: "SH", putCall: null, discretion: "SOLE",
  }],
  holdingsOmitted: false,
});
const mkForm4 = (acc: string, filedAt: string): Form4Filing => ({
  accession: acc, filedAt, cik: "1", indexUrl: `https://www.sec.gov/x/${acc}/`,
  issuerCik: "2", issuerName: "TEST CO", issuerTradingSymbol: "TST",
  periodOfReport: filedAt, owners: [], transactions: [],
});
const mk8k = (acc: string, filedAt: string): Earnings8K => ({
  accession: acc, cik: "1", companyName: "TEST CO", filedAt,
  acceptanceDatetime: filedAt, ticker: "TST", itemCodes: ["2.02"],
  indexUrl: `https://www.sec.gov/x/${acc}/`, exhibitUrl: `https://www.sec.gov/x/${acc}/ex99.htm`,
  text: "results text", textLength: 12, truncated: false,
});

/** Lines in every day file (plain or gzipped) of an archive directory. */
function rowsOnDisk(dir: string): Array<{ day: string; accession: string }> {
  const out: Array<{ day: string; accession: string }> = [];
  for (const f of fs.readdirSync(dir).sort()) {
    const raw = fs.readFileSync(path.join(dir, f));
    const text = (f.endsWith(".gz") ? zlib.gunzipSync(raw) : raw).toString("utf8");
    for (const line of text.split("\n")) if (line) out.push({ day: f.slice(0, 10), accession: JSON.parse(line).accession });
  }
  return out;
}
const quiet = () => {
  const lines: string[] = [];
  const e = console.error;
  console.error = (...a: unknown[]) => { lines.push(a.map(String).join(" ")); };
  return { lines, restore: () => { console.error = e; } };
};

// ── the byte scan ───────────────────────────────────────────────────────────
test("accessionsInBuffer reads each row's accession and never one quoted inside a string value", () => {
  const rows = [mk13F("0001762716-26-000001", "2026-11-06"), mk13F("0001762716-26-000002", "2026-11-06")];
  const buf = Buffer.from(rows.map((r) => JSON.stringify(r)).join("\n") + "\n");
  const got = new Set<string>();
  accessionsInBuffer(buf, got);
  assert.deepEqual(Array.from(got).sort(), ["0001762716-26-000001", "0001762716-26-000002"],
    "the holding whose NAME contains the key text is escaped on disk and must not be read as an accession");
  const empty = new Set<string>();
  accessionsInBuffer(Buffer.from(""), empty);
  accessionsInBuffer(Buffer.from('{"accession":"UNTERMINATED'), empty);
  accessionsInBuffer(Buffer.from('{"accession":""}'), empty);
  assert.equal(empty.size, 0, "empty, unterminated and blank values add nothing and do not throw");
});

test("each writer's row holds `accession` once, at the top level (what the byte scan relies on)", () => {
  for (const row of [mk13F("A", "2026-11-06"), mkForm4("A", "2026-11-06"), mk8k("A", "2026-11-06")]) {
    const s = JSON.stringify(row);
    assert.equal(s.split('"accession":"').length - 1, 1, `one accession key per row: ${s.slice(0, 60)}`);
  }
  // the row interfaces declare it only as a top-level string field
  for (const f of ["edgar13f.ts", "edgarForm4.ts", "sec8kEarnings.ts"]) {
    const src = fs.readFileSync(path.join(HERE, f), "utf8");
    const decls = src.match(/^[ \t]*accession\??:[^\n;,]*;[ \t]*$/gm) || [];
    assert.ok(decls.length >= 1, `${f} declares accession`);
    for (const d of decls) assert.match(d, /accession:\s*string;/, `${f}: accession is a plain top-level string (${d.trim()})`);
  }
});

// ── the measured cases, through the real writers ────────────────────────────
test("REGRESSION 13F 2026-11-08: a restart two days after 40 filings were archived appends none of them again", () => {
  const base = tmp("vt-seen-13f-");
  const dir = path.join(base, "filings13f");
  const page = Array.from({ length: 40 }, (_, i) => mk13F(`0001000000-26-${String(i).padStart(6, "0")}`, "2026-11-06"));
  assert.equal(archive13FFilings(page, base, at("2026-11-06", 21)), 40);
  // Sunday 11-08: a new container, the 11-06 file gzipped, the same 40 still on the feed's page
  _filings13fSeenForTests._resetForTests();
  gzipOld13FDays(base, at("2026-11-08", 14));
  assert.equal(archive13FFilings(page, base, at("2026-11-08", 14)), 0, "all 40 are already in the 2026-11-06 day file");
  assert.equal(fs.existsSync(path.join(dir, "2026-11-08.jsonl")), false, "no Sunday day file is created for rows already held");
  assert.equal(rowsOnDisk(dir).length, 40);
  const poll = _filings13fSeenForTests.stats().last_poll!;
  assert.deepEqual([poll.candidates, poll.fresh, poll.held_older_day, poll.held_recent], [40, 0, 40, 0],
    "the status says these were held by a day the old two-day check could not see");
});

test("REGRESSION 13F Labor Day: rows first archived 08-29 are not appended again on 08-31, 09-02 or 09-07 across three restarts", () => {
  const base = tmp("vt-seen-labor-");
  const dir = path.join(base, "filings13f");
  const page = Array.from({ length: 40 }, (_, i) => mk13F(`0002000000-26-${String(i).padStart(6, "0")}`, "2026-08-28"));
  assert.equal(archive13FFilings(page, base, at("2026-08-29", 2)), 40);
  for (const day of ["2026-08-31", "2026-09-02", "2026-09-07"]) {
    _filings13fSeenForTests._resetForTests();
    gzipOld13FDays(base, at(day));
    assert.equal(archive13FFilings(page, base, at(day)), 0, `${day}: nothing re-appended`);
  }
  assert.deepEqual(Array.from(new Set(rowsOnDisk(dir).map((r) => r.day))), ["2026-08-29"], "one archive day holds them, once");
});

test("a poll after a 3-day outage: rows first archived days after they were filed are still found after the next restart", () => {
  const base = tmp("vt-seen-outage-");
  const late = [mk13F("0003000000-26-000001", "2026-11-03"), mk13F("0003000000-26-000002", "2026-11-03")];
  assert.equal(archive13FFilings(late, base, at("2026-11-06")), 2, "first seen three days after filing: archived under 11-06");
  _filings13fSeenForTests._resetForTests();
  assert.equal(archive13FFilings(late, base, at("2026-11-09")), 0, "the walk from the filed date forward reaches the 11-06 file");
});

test("a genuinely new filing is still appended, next to held ones, before and after a restart", () => {
  const base = tmp("vt-seen-new-");
  const dir = path.join(base, "filings13f");
  const held = mk13F("0004000000-26-000001", "2026-11-04");
  assert.equal(archive13FFilings([held], base, at("2026-11-04")), 1);
  _filings13fSeenForTests._resetForTests();
  const fresh = mk13F("0004000000-26-000002", "2026-11-09");
  assert.equal(archive13FFilings([held, fresh], base, at("2026-11-09")), 1);
  assert.equal(archive13FFilings([held, fresh], base, at("2026-11-09", 13)), 0, "same process, next poll");
  _filings13fSeenForTests._resetForTests();
  assert.equal(archive13FFilings([held, fresh], base, at("2026-11-09", 14)), 0, "after another restart the same day");
  assert.deepEqual(rowsOnDisk(dir), [
    { day: "2026-11-04", accession: "0004000000-26-000001" },
    { day: "2026-11-09", accession: "0004000000-26-000002" },
  ]);
  // a row with no accession is dropped, as the writers always did
  assert.equal(archive13FFilings([{ ...fresh, accession: "" }], base, at("2026-11-09", 15)), 0);
});

test("Form 4 and 8-K writers answer from the day files too (restart two days later appends nothing)", () => {
  const b4 = tmp("vt-seen-f4-");
  const f4 = [mkForm4("0005000000-26-000001", "2026-10-27"), mkForm4("0005000000-26-000002", "2026-10-27")];
  assert.equal(archiveFilings(f4, b4, at("2026-10-27")), 2);
  _filingsSeenForTests._resetForTests();
  gzipOldFilingDays(b4, at("2026-10-30"));
  assert.equal(archiveFilings(f4, b4, at("2026-10-30")), 0);
  assert.equal(rowsOnDisk(path.join(b4, "filings")).length, 2);

  const b8 = tmp("vt-seen-8k-");
  const e8 = [mk8k("0006000000-26-000001", "2026-11-10")];
  assert.equal(archiveEarnings8Ks(e8, b8, at("2026-11-10")), 1);
  _earnings8kSeenForTests._resetForTests();
  assert.equal(archiveEarnings8Ks(e8, b8, at("2026-11-13")), 0);
  assert.equal(rowsOnDisk(path.join(b8, "earnings8k")).length, 1);
});

test("a release filed years before the window (the 2020/2021 rows of 2026-11-15) is archived once, counted, and not repeated by a restart", () => {
  const base = tmp("vt-seen-old-");
  const old = [mk8k("0001018724-20-000021", "2020-07-09"), { ...mk8k("0007000000-26-000009", "2026-11-15"), filedAt: null }];
  assert.equal(archiveEarnings8Ks(old, base, at("2026-11-15")), 2, "refusing very old feed entries is a separate open rule; this one only stops repeats");
  assert.equal(_earnings8kSeenForTests.stats().last_poll!.before_lookback, 2, "an old filed date and a missing one are both counted");
  _earnings8kSeenForTests._resetForTests();
  assert.equal(archiveEarnings8Ks(old, base, at("2026-11-17")), 0, "held inside the window: not appended again");
  assert.equal(rowsOnDisk(path.join(base, "earnings8k")).length, 2);
});

// ── the helper's own bounds ─────────────────────────────────────────────────
test("nothing is forgotten at volume: 60,000 accessions in one day stay held (the old set cleared itself at 50,000)", () => {
  const dir = tmp("vt-seen-vol-");
  const seen = createFilingSeen();
  const now = at("2026-11-16");
  const ids = Array.from({ length: 60_000 }, (_, i) => `0008000000-26-${String(i).padStart(6, "0")}`);
  fs.writeFileSync(path.join(dir, "2026-11-16.jsonl"), ids.map((a) => JSON.stringify({ accession: a, filedAt: "2026-11-16" })).join("\n") + "\n");
  seen.noteAppended(dir, ids, now);
  assert.equal(seen.size, 60_000);
  const again = [ids[0], ids[49_999], ids[59_999]].map((a) => ({ accession: a, filedAt: "2026-11-16" }));
  assert.equal(seen.selectFresh(dir, again, now).length, 0, "first, 50,000th and last are all still held");
});

test("memory is bounded by the window, not by clearing: day sets older than the lookback are dropped, files are read once", () => {
  const dir = tmp("vt-seen-evict-");
  const seen = createFilingSeen(5);
  for (const day of ["2026-11-01", "2026-11-02", "2026-11-03"]) {
    fs.writeFileSync(path.join(dir, `${day}.jsonl`), JSON.stringify({ accession: `E-${day}`, filedAt: day }) + "\n");
  }
  const cand = [{ accession: "E-2026-11-01", filedAt: "2026-11-01" }];
  assert.equal(seen.selectFresh(dir, cand, at("2026-11-04")).length, 0, "found in its own day file");
  const loaded = seen.stats().day_sets;
  assert.ok(loaded >= 4, "the walk loaded 10-31..11-04");
  // the file changes behind the cache: a day already read is not read again this process
  fs.unlinkSync(path.join(dir, "2026-11-01.jsonl"));
  assert.equal(seen.selectFresh(dir, cand, at("2026-11-04", 13)).length, 0, "answered from the day set already loaded");
  // far later: every earlier day set is outside the 5-day window and is dropped
  seen.selectFresh(dir, [{ accession: "E-new", filedAt: "2026-11-30" }], at("2026-11-30"));
  assert.ok(seen.stats().day_sets <= 5, `day sets stay within the window (${seen.stats().day_sets})`);
  assert.equal(FILING_SEEN_LOOKBACK_DAYS, 21, "the production window: twice the longest measured stay on a feed page (9 days) plus margin");
});

test("an unreadable day file is counted and treated as empty: the worst case is a repeat row, never a filing left out", () => {
  const dir = tmp("vt-seen-bad-");
  fs.writeFileSync(path.join(dir, "2026-11-10.jsonl.gz"), Buffer.from("this is not gzip"));
  const seen = createFilingSeen();
  const q = quiet();
  let fresh: Array<{ accession: string; filedAt: string }>;
  try {
    fresh = seen.selectFresh(dir, [{ accession: "B-1", filedAt: "2026-11-10" }], at("2026-11-12"));
  } finally { q.restore(); }
  assert.equal(fresh.length, 1, "kept as fresh");
  assert.equal(seen.stats().unreadable_day_files, 1);
  assert.ok(q.lines.some((l) => l.includes("filingSeen") && l.includes("2026-11-10")), "and said so in the log");
});

// ── source ratchet ──────────────────────────────────────────────────────────
test("the three writers share the one check: no two-day seed, no clear(), gauge kept", () => {
  for (const [f, gauge] of [
    ["edgar13f.ts", "edgar13f.archivedAccessions"],
    ["edgarForm4.ts", "edgarForm4.archivedAccessions"],
    ["sec8kEarnings.ts", "sec8kEarnings.archivedAccessions"],
  ]) {
    const src = fs.readFileSync(path.join(HERE, f), "utf8");
    assert.ok(src.includes("createFilingSeen()"), `${f} uses filingSeen.ts`);
    assert.ok(src.includes("archiveSeen.selectFresh(dir, filings, now)"), `${f} selects through it`);
    assert.ok(!/function seedSeen\b/.test(src), `${f} has no private two-day seed`);
    assert.ok(!/Accessions\.clear\(\)/.test(src), `${f} never clears its seen state`);
    assert.ok(src.includes(`trackSet("${gauge}", archiveSeen)`), `${f} keeps its /api/health gauge`);
  }
});
