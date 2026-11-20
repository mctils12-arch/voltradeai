import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import os from "os";
import path from "path";
import zlib from "zlib";
import {
  runFiresCompleteness, firesCandidateDays, firesCompletenessDue, heldFireDay, readFiresLedger, writeFiresLedger,
  firesCompletenessEnabled, firesCompletenessStatus,
  FIRES_REFILL_LOOKBACK_DAYS, FIRES_LOCAL_WINDOW_DAYS, FIRES_DAY_MAX_ATTEMPTS, FIRES_COMPLETENESS_STATE_FILENAME,
  type FiresLedger,
} from "./firesCompleteness";
import { firmsUrl } from "./nasaFirms";

// NASA's column layout (the NOAA-20 file of 2026-11-20).
const HEADER = "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,confidence,version,bright_ti5,frp,daynight\n";
const row = (lat: number, day: string, time: string, sat = "N20") => `${lat},-21.78175,297.71,0.59,0.53,${day},${time},${sat},nominal,2.0NRT,269.68,2.24,N\n`;
const csv = (...rows: string[]) => HEADER + rows.join("");
const NOW = Date.parse("2026-11-21T06:30:00Z");
const emptyLedger = (): FiresLedger => ({ version: 1, last_run_day: null, last_run_at: null, last_run_detail: null, days: {} });
const tmpDir = () => fs.mkdtempSync(path.join(os.tmpdir(), "vt-fires-complete-"));
const lines = (fp: string) => (fp.endsWith(".gz") ? zlib.gunzipSync(fs.readFileSync(fp)).toString("utf8") : fs.readFileSync(fp, "utf8")).trim().split("\n").map((l) => JSON.parse(l));

/** NASA by (source, date): a CSV body, an HTTP status, or absent (header only). */
function fakeNasa(answers: Record<string, string | number>) {
  const asked: string[] = [];
  const impl = async (url: string) => {
    const m = url.match(/\/(VIIRS_[A-Z0-9]+_NRT)\/[^/]+\/1\/(\d{4}-\d{2}-\d{2})$/);
    assert.ok(m, `a dated one-day request, got ${url}`);
    const key = `${m![1]} ${m![2]}`;
    asked.push(key);
    const a = answers[key];
    if (typeof a === "number") return { ok: false, status: a, text: async () => "" };
    return { ok: true, status: 200, text: async () => a ?? HEADER };
  };
  return { impl, asked };
}
const noSleep = async () => {};

test("the dated URL is NASA's documented form, and a malformed date is refused before any request", () => {
  assert.match(firmsUrl("K", "VIIRS_NOAA20_NRT", "-180,-90,180,90", 1, "2026-11-03"), /\/api\/area\/csv\/K\/VIIRS_NOAA20_NRT\/-180,-90,180,90\/1\/2026-11-03$/);
  assert.throws(() => firmsUrl("K", "VIIRS_NOAA20_NRT", "-180,-90,180,90", 1, "11/03/2026"), /YYYY-MM-DD/);
});

test("the lookback stays inside the fires stream's local window: older than that, a missing local file may be off-site", () => {
  assert.ok(FIRES_REFILL_LOOKBACK_DAYS < FIRES_LOCAL_WINDOW_DAYS, `${FIRES_REFILL_LOOKBACK_DAYS} < ${FIRES_LOCAL_WINDOW_DAYS}`);
});

test("THE GAP: days with no file are refilled oldest first from NOAA-20's dated answer, gzipped; a day that has a file is not asked for", async () => {
  const dir = tmpDir();
  // the archive as it stood: Suomi NPP days up to 11-02, then nothing
  fs.writeFileSync(path.join(dir, "2026-11-02.jsonl.gz"), zlib.gzipSync(JSON.stringify({ id: "N:1", satellite: "N", acq_date: "2026-11-02" }) + "\n"));
  fs.writeFileSync(path.join(dir, "2026-11-20.jsonl"), JSON.stringify({ id: "x", satellite: "N20", acq_date: "2026-11-20" }) + "\n");
  const ledger = emptyLedger();
  // only three days are served in this test; every other missing day answers header-only
  const nasa = fakeNasa({
    "VIIRS_NOAA20_NRT 2026-11-03": csv(row(1, "2026-11-03", "0100"), row(2, "2026-11-03", "0200")),
    "VIIRS_NOAA20_NRT 2026-11-04": csv(row(3, "2026-11-04", "0100"), row(9, "2026-11-05", "0001")),
    "VIIRS_NOAA21_NRT 2026-11-05": csv(row(4, "2026-11-05", "0100", "N21")),
  });
  const days = firesCandidateDays(dir, ledger, NOW);
  assert.equal(days[0], "2026-11-20", "yesterday first");
  assert.equal(days[1], "2026-10-27", "then the oldest day in reach with no file");
  assert.ok(!days.includes("2026-11-02"), "a day with a file is not a candidate");
  assert.ok(!days.includes("2026-11-21"), "today is the live poll's");

  // the ledger is handed to `persist` after every day: a pass cut short by a restart keeps what it finished
  const saved: number[] = [];
  const run = await runFiresCompleteness({ mapKey: "K", fetch: nasa.impl, dir, now: () => NOW, sleep: noSleep, ledger, maxDays: 99, persist: (l) => saved.push(Object.keys(l.days).length) });
  assert.equal(saved.length, run.days_asked);
  assert.deepEqual(saved.slice(0, 3), [1, 2, 3]);
  assert.equal(run.status, "ok");
  assert.equal(run.refilled_days, 3);
  const d3 = lines(path.join(dir, "2026-11-03.jsonl.gz"));
  assert.equal(d3.length, 2);
  assert.equal(d3[0].satellite, "N20");
  assert.equal(lines(path.join(dir, "2026-11-04.jsonl.gz")).length, 1, "a row of another date in NASA's answer is not written into this day");
  assert.equal(lines(path.join(dir, "2026-11-05.jsonl.gz"))[0].satellite, "N21", "NOAA-20 had none for the date, so NOAA-21 was asked");
  assert.equal(ledger.days["2026-11-03"].status, "refilled");
  assert.equal(ledger.days["2026-11-03"].added, 2);
  assert.ok(ledger.days["2026-11-03"].refilled_at);
  // a day NASA no longer serves: all three sources asked, nothing written, said so
  assert.equal(ledger.days["2026-10-27"].status, "no_rows");
  assert.equal(nasa.asked.filter((a) => a.endsWith("2026-10-27")).length, 3);
  assert.ok(!fs.existsSync(path.join(dir, "2026-10-27.jsonl.gz")));
  assert.equal(lines(path.join(dir, "2026-11-02.jsonl.gz")).length, 1, "the gzipped day is untouched");
  assert.equal(ledger.last_run_day, "2026-11-21");
});

test("yesterday: rows NASA has by date that the live poll did not archive are appended, once, and counted", async () => {
  const dir = tmpDir();
  const y = "2026-11-20";
  // the live poll archived two rows; NASA's dated answer has those two and one published after midnight
  const first = csv(row(1, y, "0100"), row(2, y, "1200"));
  const { parseFirmsCsv } = await import("./nasaFirms");
  fs.writeFileSync(path.join(dir, `${y}.jsonl`), parseFirmsCsv(first).map((r) => JSON.stringify(r)).join("\n") + "\n");
  // every older day in reach has a file, so only yesterday is a candidate
  for (let b = 2; b <= FIRES_REFILL_LOOKBACK_DAYS; b++) fs.writeFileSync(path.join(dir, `${new Date(NOW - b * 86400_000).toISOString().slice(0, 10)}.jsonl.gz`), zlib.gzipSync("\n"));
  const ledger = emptyLedger();
  const nasa = fakeNasa({ [`VIIRS_NOAA20_NRT ${y}`]: csv(row(1, y, "0100"), row(2, y, "1200"), row(3, y, "2350")) });
  const run = await runFiresCompleteness({ mapKey: "K", fetch: nasa.impl, dir, now: () => NOW, sleep: noSleep, ledger });
  assert.deepEqual(nasa.asked, [`VIIRS_NOAA20_NRT ${y}`], "the day's own satellite, one request");
  assert.equal(run.rows_added, 1);
  const held = lines(path.join(dir, `${y}.jsonl`));
  assert.equal(held.length, 3);
  assert.equal(held[2].acq_time, "2350");
  assert.deepEqual({ ...ledger.days[y], at: "" }, { status: "complete", source: "VIIRS_NOAA20_NRT", upstream_rows: 3, held_before: 2, added: 1, attempts: 1, error: null, at: "" });
  // settled: a second run asks nothing and adds nothing
  const again = await runFiresCompleteness({ mapKey: "K", fetch: nasa.impl, dir, now: () => NOW, sleep: noSleep, ledger });
  assert.equal(again.days_asked, 0);
  assert.equal(lines(path.join(dir, `${y}.jsonl`)).length, 3);
});

test("a gzipped day file is never rewritten, even when it is yesterday's", () => {
  const dir = tmpDir();
  fs.writeFileSync(path.join(dir, "2026-11-20.jsonl.gz"), zlib.gzipSync(JSON.stringify({ id: "a", satellite: "N20" }) + "\n"));
  assert.ok(!firesCandidateDays(dir, emptyLedger(), NOW).includes("2026-11-20"));
  assert.equal(heldFireDay(dir, "2026-11-20").file, "gz");
  // a line cut short by a killed write is counted, not held and not fatal
  fs.writeFileSync(path.join(dir, "2026-11-19.jsonl"), JSON.stringify({ id: "b", satellite: "N20" }) + "\n" + '{"id":"c","sat');
  const torn = heldFireDay(dir, "2026-11-19");
  assert.deepEqual([torn.file, torn.ids.size, torn.unreadable_lines], ["plain", 1, 1]);
});

test("NASA refusing: each day is recorded failed with the error, the run stops after three in a row, and a day is given up after its attempts", async () => {
  const dir = tmpDir();
  const ledger = emptyLedger();
  const nasa = { asked: [] as string[], impl: async (url: string) => { nasa.asked.push(url); return { ok: false, status: 400, text: async () => "" }; } };
  const run = await runFiresCompleteness({ mapKey: "K", fetch: nasa.impl, dir, now: () => NOW, sleep: noSleep, ledger });
  assert.equal(run.status, "error");
  assert.equal(run.stopped_early, true);
  assert.equal(run.days_asked, 3);
  assert.match(ledger.days["2026-11-20"].error || "", /VIIRS_NOAA20_NRT: FIRMS 400; VIIRS_NOAA21_NRT: FIRMS 400; VIIRS_SNPP_NRT: FIRMS 400/);
  assert.equal(fs.readdirSync(dir).length, 0, "nothing is written on a failure");
  for (let i = 1; i < FIRES_DAY_MAX_ATTEMPTS; i++) await runFiresCompleteness({ mapKey: "K", fetch: nasa.impl, dir, now: () => NOW, sleep: noSleep, ledger });
  assert.equal(ledger.days["2026-11-20"].attempts, FIRES_DAY_MAX_ATTEMPTS);
  assert.ok(!firesCandidateDays(dir, ledger, NOW).includes("2026-11-20"), "given up, and the ledger says failed");
  assert.equal(ledger.days["2026-11-20"].status, "failed");
});

test("an error that quotes the request URL never puts the key in the ledger", async () => {
  const dir = tmpDir();
  const ledger = emptyLedger();
  const KEY = "0123456789abcdef0123456789abcdef";
  const impl = async (url: string) => { throw new Error(`request to ${url} failed`); };
  await runFiresCompleteness({ mapKey: KEY, fetch: impl as never, dir, now: () => NOW, sleep: noSleep, ledger, maxDays: 1 });
  assert.ok(!JSON.stringify(ledger).includes(KEY));
  assert.match(ledger.days["2026-11-20"].error || "", /\[key\]/);
});

test("one pass per UTC day, from 06:00 UTC; the ledger survives a restart", () => {
  const l = emptyLedger();
  assert.equal(firesCompletenessDue(l, Date.parse("2026-11-21T05:59:00Z")), false);
  assert.equal(firesCompletenessDue(l, Date.parse("2026-11-21T06:00:00Z")), true);
  l.last_run_day = "2026-11-21";
  assert.equal(firesCompletenessDue(l, Date.parse("2026-11-21T21:00:00Z")), false);
  assert.equal(firesCompletenessDue(l, Date.parse("2026-11-22T06:01:00Z")), true);
  const base = tmpDir();
  const file = path.join(base, FIRES_COMPLETENESS_STATE_FILENAME);
  writeFiresLedger(l, file);
  assert.equal(readFiresLedger(file).last_run_day, "2026-11-21");
  fs.writeFileSync(file, "{not json");
  const orig = console.error; console.error = () => {};
  try { assert.deepEqual(readFiresLedger(file).days, {}); } finally { console.error = orig; }
});

test("off without a key or with the switch off; the status counts what the live poll missed", () => {
  assert.equal(firesCompletenessEnabled({}), false);
  assert.equal(firesCompletenessEnabled({ FIRMS_MAP_KEY: "k" }), true);
  assert.equal(firesCompletenessEnabled({ FIRMS_MAP_KEY: "k", VOLTRADE_FIRES_COMPLETENESS: "0" }), false);
  const base = tmpDir();
  const l = emptyLedger();
  l.days["2026-11-20"] = { status: "complete", source: "VIIRS_NOAA20_NRT", upstream_rows: 1000, held_before: 880, added: 120, attempts: 1, error: null, at: "2026-11-21T06:30:00.000Z" };
  l.days["2026-11-03"] = { status: "refilled", source: "VIIRS_NOAA20_NRT", upstream_rows: 70000, held_before: 0, added: 70000, attempts: 1, error: null, at: "2026-11-21T06:31:00.000Z", refilled_at: "2026-11-21T06:31:00.000Z" };
  l.days["2026-10-27"] = { status: "no_rows", source: null, upstream_rows: 0, held_before: 0, added: 0, attempts: 3, error: null, at: "2026-11-21T06:32:00.000Z" };
  writeFiresLedger(l, path.join(base, FIRES_COMPLETENESS_STATE_FILENAME));
  const s = firesCompletenessStatus({ DATA_DIR: base, FIRMS_MAP_KEY: "k" }) as any;
  assert.equal(s.refilled_days, 1);
  assert.equal(s.no_rows_days, 1);
  assert.deepEqual(s.live_poll_missed, { days: 1, upstream_rows: 1000, added: 120, share: 0.12 }, "a refilled day is not a reading of what the live poll missed");
  assert.equal(s.days[0].day, "2026-11-20", "newest first");
  assert.ok(!JSON.stringify(s).includes('"k"'), "the key is not served");
});
