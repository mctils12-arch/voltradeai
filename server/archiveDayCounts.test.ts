// archive_day_counts battery ([PIPELINE] 2026-11-26, open_questions.md (av2)).
// KNOWN BROKEN #62 let a late row replace a compressed archive day; the repair
// stops new loss and this probe counts what the day files hold, so the days
// already lost can be named and read again from the source.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import { fileURLToPath } from "node:url";
import {
  archiveDayCounts, foldDay, suspects, specFor, COUNT_STREAMS, MAX_WINDOW_DAYS, DEFAULT_WINDOW_DAYS, LOCAL_MEDIAN_SPAN,
  type DayCount,
} from "./archiveDayCounts";
import { archiveGeneration, _resetGridGenerationForTests, type GenerationObs } from "./gridGeneration";
import { archiveDemand, _resetGridDemandForTests } from "./gridDemand";

const here = path.dirname(fileURLToPath(import.meta.url));
const tmpDir = (tag: string) => fs.mkdtempSync(path.join(os.tmpdir(), `daycounts-${tag}-`));
const hour = (day: string, h: number) => `${day}T${String(h).padStart(2, "0")}`;
const dayPlus = (day: string, n: number) => new Date(Date.parse(`${day}T00:00:00Z`) + n * 86400_000).toISOString().slice(0, 10);

/** One respondent's whole day of generation: 24 hours of two fuel types. */
function genDay(day: string, respondent: string, hours = 24): GenerationObs[] {
  const out: GenerationObs[] = [];
  for (let h = 0; h < hours; h++) {
    out.push({ period: hour(day, h), respondent, fueltype: "NG", mwh: 100 + h, rt: day });
    out.push({ period: hour(day, h), respondent, fueltype: "SUN", mwh: 5, rt: day });
  }
  return out;
}
const lines = (rows: object[]) => rows.map((r) => JSON.stringify(r)).join("\n") + "\n";
const writeGz = (dir: string, day: string, rows: object[]) => fs.writeFileSync(path.join(dir, `${day}.jsonl.gz`), zlib.gzipSync(lines(rows)));

type Summary = {
  days_held: number; first_day: string; last_day: string;
  window: Record<string, number | string | null>;
  suspects: { day: string; rows: number; groups: number; fewer_groups_than_both_neighbours: boolean; under_half_the_median_rows: boolean }[];
  by_day?: DayCount[]; scan?: string; error?: string;
};

test("the keys are the archive modules' own: rows the two EIA modules write count as that many distinct keys", async () => {
  const base = tmpDir("keys");
  try {
    _resetGridGenerationForTests();
    _resetGridDemandForTests();
    const day = "2026-11-20";
    const gen = [...genDay(day, "PJM"), ...genDay(day, "ERCO", 20)];
    assert.equal(archiveGeneration(gen, base), gen.length);
    const dem = [];
    for (let h = 0; h < 24; h++) {
      dem.push({ period: hour(day, h), respondent: "PJM", type: "D" as const, mwh: 90_000, rt: day });
      dem.push({ period: hour(day, h), respondent: "PJM", type: "DF" as const, mwh: 91_000, rt: day });
    }
    assert.equal(archiveDemand(dem, base), dem.length);
    const g = await archiveDayCounts({ stream: "gridgeneration", day }, base) as unknown as DayCount & { by_group: { group: string; rows: number; slots: number; series: number }[] };
    assert.equal(g.rows, gen.length);
    assert.equal(g.keys, gen.length, "every row the module wrote is its own key");
    assert.equal(g.repeated_rows, 0);
    assert.equal(g.groups, 2);
    assert.equal(g.slots, 24);
    assert.equal(g.groups_short, 1, "ERCO holds 20 of 24 hours");
    assert.equal(g.slots_min, 20);
    assert.deepEqual(g.by_group, [
      { group: "ERCO", rows: 40, slots: 20, series: 2, null_values: 0 },
      { group: "PJM", rows: 48, slots: 24, series: 2, null_values: 0 },
    ]);
    const d = await archiveDayCounts({ stream: "griddemand", day }, base) as unknown as DayCount;
    assert.equal(d.rows, 48);
    assert.equal(d.keys, 48, "demand and forecast of the same hour are two keys");
    assert.equal(d.groups_short, 0);
  } finally {
    _resetGridGenerationForTests();
    _resetGridDemandForTests();
    fs.rmSync(base, { recursive: true, force: true });
  }
});

test("a day's fold: repeated keys, null values, lines that are not rows; a legacy demand row with no type is demand", () => {
  const spec = specFor("griddemand")!;
  const text = [
    JSON.stringify({ period: "2026-11-20T00", respondent: "NW", mwh: 10 }),             // legacy: no type
    JSON.stringify({ period: "2026-11-20T00", respondent: "NW", type: "D", mwh: 10 }),  // the same key
    JSON.stringify({ period: "2026-11-20T01", respondent: "NW", type: "D", mwh: null }),
    "{cut",
    JSON.stringify({ respondent: "NW", mwh: 3 }),                                       // no hour
    JSON.stringify([1, 2]),
    "",
  ].join("\n");
  const f = foldDay(spec, "2026-11-20", [text], { plain: true, gz: false, unreadable: null });
  assert.equal(f.count.rows, 3);
  assert.equal(f.count.keys, 2);
  assert.equal(f.count.repeated_rows, 1);
  assert.equal(f.count.null_values, 1);
  assert.equal(f.count.bad_lines, 3);
  assert.equal(f.count.groups_short, 1);
  // page views: one row a ticker a day, no slots
  const wiki = foldDay(specFor("wikiattention")!, "2026-11-20", [lines([
    { date: "2026-11-20", ticker: "AAPL", article: "Apple_Inc.", views: 100, rt: "2026-11-21" },
    { date: "2026-11-20", ticker: "MSFT", article: "Microsoft", views: 90, rt: "2026-11-21" },
  ])], { plain: false, gz: true, unreadable: null });
  assert.equal(wiki.count.groups, 2);
  assert.equal(wiki.count.keys, 2);
  assert.equal(wiki.count.slots, 0);
  assert.equal(wiki.count.groups_short, 0);
  assert.equal(wiki.count.slots_min, null);
});

test("#62's damage as it was written: a compressed day holding only one respondent's late hours is named, with both signatures", async () => {
  const base = tmpDir("overwrite");
  const dir = path.join(base, "gridgeneration");
  fs.mkdirSync(dir);
  try {
    const three = (day: string) => [...genDay(day, "PJM"), ...genDay(day, "ERCO"), ...genDay(day, "MISO")];
    const days = ["2026-11-10", "2026-11-11", "2026-11-12", "2026-11-13", "2026-11-14"];
    for (const d of days) writeGz(dir, d, three(d));
    // what the unrepaired compressor left: the day's .gz is the late file alone
    writeGz(dir, "2026-11-12", genDay("2026-11-12", "ERCO", 4));
    // a day still filling (plain, thin) is never a suspect
    fs.writeFileSync(path.join(dir, "2026-11-15.jsonl"), lines(genDay("2026-11-15", "PJM", 3)));
    const before = fs.readdirSync(dir).sort();
    const s = await archiveDayCounts({ stream: "gridgeneration" }, base) as unknown as Summary;
    assert.equal(s.days_held, 6);
    assert.equal(s.window.days, 6);
    assert.equal(s.window.compressed, 5);
    assert.equal(s.window.suspect_days, 1);
    assert.deepEqual(s.suspects, [
      { day: "2026-11-12", rows: 8, groups: 1, fewer_groups_than_both_neighbours: true, under_half_the_median_rows: true },
    ]);
    assert.equal(s.window.compressed_days_with_a_group_short_of_a_whole_day, 1);
    assert.equal(s.by_day!.length, 6);
    assert.deepEqual(fs.readdirSync(dir).sort(), before, "the probe writes nothing");
    // the answer carries counts and codes, never a reading
    assert.ok(!/"mwh"/.test(JSON.stringify(s)));
  } finally {
    fs.rmSync(base, { recursive: true, force: true });
  }
});

test("two damaged days in a row hide each other from the neighbour signature; the row signature names both", () => {
  const mk = (day: string, rows: number, groups: number, gz = true): DayCount => ({
    day, plain: !gz, gz, unreadable: null, rows, bad_lines: 0, keys: rows, repeated_rows: 0, null_values: 0,
    groups, slots: 24, groups_short: 0, slots_min: 24, slots_max: 24,
  });
  const days = [mk("2026-11-01", 4000, 12), mk("2026-11-02", 4000, 12), mk("2026-11-03", 64, 1), mk("2026-11-04", 32, 1),
    mk("2026-11-05", 4000, 12), mk("2026-11-06", 4000, 12)];
  const r = suspects(days);
  assert.deepEqual(r.suspects.map((x) => [x.day, x.fewer_groups_than_both_neighbours, x.under_half_the_median_rows]), [
    ["2026-11-03", false, true], ["2026-11-04", false, true],
  ]);
  assert.equal(r.median_rows, 4000);
  // a gap in the calendar is not a neighbour: the neighbour signature needs the day before AND the day after
  const gap = suspects([mk("2026-11-01", 4000, 12), mk("2026-11-03", 3000, 11), mk("2026-11-04", 4000, 12)]);
  assert.deepEqual(gap.suspects, []);
  // an unreadable day is neither a suspect nor a neighbour
  const bad = { ...mk("2026-11-02", 0, 0), unreadable: "does not gunzip" };
  assert.deepEqual(suspects([mk("2026-11-01", 4000, 12), bad, mk("2026-11-03", 4000, 12)]).suspects, []);
});

test("the row signature compares a day with the days around it, so an early year with fewer series is not a suspect", () => {
  const mk = (i: number, rows: number): DayCount => ({
    day: dayPlus("2025-01-01", i), plain: false, gz: true, unreadable: null, rows, bad_lines: 0, keys: rows, repeated_rows: 0,
    null_values: 0, groups: 12, slots: 24, groups_short: 0, slots_min: 24, slots_max: 24,
  });
  // 60 days at 288 rows (demand only), then 60 days at 576 (demand and forecast), then 60 more at 1,200
  const days = [...Array(60).keys()].map((i) => mk(i, 288))
    .concat([...Array(60).keys()].map((i) => mk(60 + i, 576)))
    .concat([...Array(60).keys()].map((i) => mk(120 + i, 1200)));
  assert.equal(suspects(days).suspects.length, 0, "288 is under half of 1,200 and is still not flagged against it");
  days[30] = mk(30, 100);
  assert.deepEqual(suspects(days).suspects.map((x) => x.day), [dayPlus("2025-01-01", 30)]);
  assert.ok(LOCAL_MEDIAN_SPAN < 60);
});

test("late rows waiting beside a compressed day are counted with it; a .gz that does not read is said, not thrown", async () => {
  const base = tmpDir("beside");
  const dir = path.join(base, "gridgeneration");
  fs.mkdirSync(dir);
  try {
    writeGz(dir, "2026-11-10", genDay("2026-11-10", "PJM"));
    fs.writeFileSync(path.join(dir, "2026-11-10.jsonl"), lines(genDay("2026-11-10", "ERCO", 2)));
    fs.writeFileSync(path.join(dir, "2026-11-11.jsonl.gz"), "not gzip");
    const s = await archiveDayCounts({ stream: "gridgeneration" }, base) as unknown as Summary;
    const [a, b] = s.by_day!;
    assert.deepEqual([a.plain, a.gz, a.rows, a.groups], [true, true, 52, 2]);
    assert.equal(s.window.late_rows_waiting_beside_a_compressed_day, 1);
    assert.match(String(b.unreadable), /^the \.gz does not read: /);
    assert.equal(b.rows, 0);
    assert.equal(s.window.unreadable, 1);
    assert.equal(s.suspects.length, 0, "an unreadable day is reported as unreadable, not as a suspect");
  } finally {
    fs.rmSync(base, { recursive: true, force: true });
  }
});

test("windows: the newest days by default, an older window with ?to, at most 60 with per-day rows, the whole archive with ?scan=all", async () => {
  const base = tmpDir("window");
  const dir = path.join(base, "wikiattention");
  fs.mkdirSync(dir);
  try {
    const first = "2026-06-01";
    for (let i = 0; i < 100; i++) {
      if (i === 50) continue; // one calendar day with no file
      const d = dayPlus(first, i);
      const tickers = i === 20 ? ["AAPL"] : ["AAPL", "MSFT", "NVDA", "XOM"];
      writeGz(dir, d, tickers.map((t) => ({ date: d, ticker: t, article: t, views: 10, rt: d })));
    }
    const def = await archiveDayCounts({ stream: "wikiattention" }, base) as unknown as Summary;
    assert.equal(def.days_held, 99);
    assert.equal(def.window.days, DEFAULT_WINDOW_DAYS);
    assert.equal(def.window.last, dayPlus(first, 99));
    assert.equal(def.window.suspect_days, 0);
    const capped = await archiveDayCounts({ stream: "wikiattention", days: "500" }, base) as unknown as Summary;
    assert.equal(capped.window.days, MAX_WINDOW_DAYS);
    assert.equal(capped.by_day!.length, MAX_WINDOW_DAYS);
    assert.equal(capped.window.calendar_days_with_no_file, 1);
    const older = await archiveDayCounts({ stream: "wikiattention", to: dayPlus(first, 29), days: "30" }, base) as unknown as Summary;
    assert.equal(older.window.first, first);
    assert.deepEqual(older.suspects.map((x) => x.day), [dayPlus(first, 20)]);
    const all = await archiveDayCounts({ stream: "wikiattention", scan: "all" }, base) as unknown as Summary;
    assert.equal(all.window.days, 99);
    assert.equal(all.scan, "all");
    assert.equal(all.by_day, undefined, "the whole-archive answer carries suspects and totals only");
    assert.deepEqual(all.suspects.map((x) => x.day), [dayPlus(first, 20)]);
    assert.equal(all.window.rows, 98 * 4 + 1);
  } finally {
    fs.rmSync(base, { recursive: true, force: true });
  }
});

test("what it refuses and what it says with nothing to read", async () => {
  const base = tmpDir("empty");
  try {
    const list = await archiveDayCounts({}, base) as { streams: { stream: string; days_held: number; first_day: string | null }[]; compress_since_boot: object };
    assert.deepEqual(list.streams.map((s) => s.stream), ["gridgeneration", "griddemand", "wikiattention"]);
    assert.ok(list.streams.every((s) => s.days_held === 0 && s.first_day === null));
    assert.deepEqual(Object.keys(list.compress_since_boot).sort(), ["compressed", "left_beside", "merged"]);
    assert.match(String((await archiveDayCounts({ stream: "../waitlist" }, base)).error), /^unknown stream; one of: gridgeneration, griddemand, wikiattention$/);
    assert.match(String((await archiveDayCounts({ stream: "waitlist" }, base)).error), /^unknown stream/);
    assert.equal((await archiveDayCounts({ stream: "griddemand", day: "../../etc" }, base)).error, "day must be YYYY-MM-DD");
    assert.equal((await archiveDayCounts({ stream: "griddemand", to: "yesterday" }, base)).error, "to must be YYYY-MM-DD");
    assert.equal((await archiveDayCounts({ stream: "griddemand", day: "2026-11-20" }, base)).held, false);
    const none = await archiveDayCounts({ stream: "griddemand" }, base) as unknown as Summary;
    assert.equal(none.days_held, 0);
    assert.equal(none.window.days, 0);
    assert.deepEqual(none.suspects, []);
  } finally {
    fs.rmSync(base, { recursive: true, force: true });
  }
});

test("the streams counted are the three #62 exposed; the probe is in the probe list and the diag switch", async () => {
  assert.deepEqual(COUNT_STREAMS.map((s) => s.stream), ["gridgeneration", "griddemand", "wikiattention"]);
  const { DIAG_PROBES } = await import("./diag");
  assert.ok((DIAG_PROBES as readonly string[]).includes("archive_day_counts"));
  const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
  assert.match(bot, /case "archive_day_counts": \{[\s\S]{0,600}sanitizeDiag\(await archiveDayCounts\(/);
  // read-only by construction: the module opens nothing for writing
  const src = fs.readFileSync(path.join(here, "archiveDayCounts.ts"), "utf8");
  assert.ok(!/writeFile|appendFile|unlink|rename|rmSync|mkdir|createWriteStream/.test(src.replace(/\/\*[\s\S]*?\*\//g, "")));
});
