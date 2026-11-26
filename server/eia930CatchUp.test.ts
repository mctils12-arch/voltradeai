// eia930CatchUp battery: the rule that decides whether one page of the newest
// EIA-930 rows reached the archive, and what the two pollers (gridDemand.ts,
// gridGeneration.ts) do when it did not. [REPAIR 2026-11-26]: EIA stopped
// publishing for about 36 hours; a poller that reads one page loses every
// hour beyond that page when the missing hours are published at once.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import { readBackToArchive, oldestHourUnseen, CATCH_UP_MAX_EXTRA_PAGES } from "./eia930CatchUp";
import {
  refreshGeneration, latestGeneration, latestGenerationPoll, generationUrl, fetchGeneration,
  RESPONDENTS as GEN_RESPONDENTS, _resetGridGenerationForTests,
} from "./gridGeneration";
import {
  refreshDemand, latestDemandPoll, demandUrl, RESPONDENTS as DEMAND_RESPONDENTS,
  HOURS_PER_FETCH as DEMAND_HOURS, _resetGridDemandForTests,
} from "./gridDemand";

type Row = { period: string; k: string };
const hour = (n: number) => new Date(Date.UTC(2026, 10, 1) + n * 3600_000).toISOString().slice(0, 13);

/** A source of `hours` hourly rows (hour 0 oldest), served newest first in
 *  pages of `len`. */
function source(hours: number, len: number, failAtOffset = -1) {
  const all: Row[] = [];
  for (let h = hours - 1; h >= 0; h--) all.push({ period: hour(h), k: `r|${hour(h)}` });
  const offsets: number[] = [];
  return {
    offsets,
    fetchPage: async (offset: number) => {
      offsets.push(offset);
      if (offset === failAtOffset) return null;
      const rows = all.slice(offset, offset + len);
      return { rows, rawCount: rows.length };
    },
  };
}

test("oldestHourUnseen: only the page's oldest hour is asked, and one unseen row there is enough", () => {
  const rows = [{ period: "2026-11-02T05", k: "a" }, { period: "2026-11-02T04", k: "b" }, { period: "2026-11-02T04", k: "c" }];
  assert.equal(oldestHourUnseen(rows, () => true), false);
  assert.equal(oldestHourUnseen(rows, (r) => r.k !== "a"), false, "an unseen NEWER hour is ordinary new data");
  assert.equal(oldestHourUnseen(rows, (r) => r.k !== "c"), true);
  assert.equal(oldestHourUnseen([], () => false), false);
});

test("normal poll: the first page reaches the archive, so exactly one request is made", async () => {
  const src = source(100, 10);
  const seen = new Set(Array.from({ length: 97 }, (_, h) => `r|${hour(h)}`)); // hours 0..96 archived, 97..99 new
  const r = await readBackToArchive<Row>({ fetchPage: src.fetchPage, pageLength: 10, isSeen: (o) => seen.has(o.k), hasHistory: true });
  assert.deepEqual(src.offsets, [0]);
  assert.equal(r.outcome, "none");
  assert.equal(r.first.length, 10);
  assert.equal(r.older.length, 0);
});

test("after a stall: pages are read until one's oldest hour is archived, and every missing hour is returned", async () => {
  const src = source(100, 10);
  const seen = new Set(Array.from({ length: 65 }, (_, h) => `r|${hour(h)}`)); // hours 0..64 archived, 65..99 published at once
  const r = await readBackToArchive<Row>({ fetchPage: src.fetchPage, pageLength: 10, isSeen: (o) => seen.has(o.k), hasHistory: true });
  assert.deepEqual(src.offsets, [0, 10, 20, 30]);
  assert.equal(r.outcome, "joined");
  assert.equal(r.extraPages, 3);
  const got = new Set([...r.first, ...r.older].map((o) => o.period));
  for (let h = 65; h < 100; h++) assert.ok(got.has(hour(h)), `hour ${h} was read`);
  assert.equal(r.oldestRead, hour(60));
});

test("a respondent with no archived row is never read back (a first run keeps one page)", async () => {
  const src = source(100, 10);
  const r = await readBackToArchive<Row>({ fetchPage: src.fetchPage, pageLength: 10, isSeen: () => false, hasHistory: false });
  assert.deepEqual(src.offsets, [0]);
  assert.equal(r.outcome, "none");
});

test("a short page means nothing older exists: no further request", async () => {
  const src = source(14, 10);
  const r = await readBackToArchive<Row>({ fetchPage: src.fetchPage, pageLength: 10, isSeen: () => false, hasHistory: true });
  assert.deepEqual(src.offsets, [0, 10]);
  assert.equal(r.outcome, "exhausted");
  assert.equal(r.first.length + r.older.length, 14);
});

test("a page that cannot be read is 'failed', and the limit is 'capped' with the oldest hour read", async () => {
  const bad = source(100, 10, 20);
  const f = await readBackToArchive<Row>({ fetchPage: bad.fetchPage, pageLength: 10, isSeen: () => false, hasHistory: true });
  assert.equal(f.outcome, "failed");
  assert.equal(f.extraPages, 1);
  const long = source(1000, 10);
  const c = await readBackToArchive<Row>({ fetchPage: long.fetchPage, pageLength: 10, isSeen: () => false, hasHistory: true });
  assert.equal(c.outcome, "capped");
  assert.equal(long.offsets.length, 1 + CATCH_UP_MAX_EXTRA_PAGES);
  assert.equal(c.oldestRead, hour(1000 - 10 * (1 + CATCH_UP_MAX_EXTRA_PAGES)));
  const first = source(100, 10, 0);
  assert.equal((await readBackToArchive<Row>({ fetchPage: first.fetchPage, pageLength: 10, isSeen: () => false, hasHistory: true })).outcome, "first_failed");
});

test("urls: the ordinary request is unchanged; a later page adds only its offset", () => {
  assert.ok(!generationUrl("PJM", "k").includes("offset"));
  assert.equal(generationUrl("PJM", "k", 960), generationUrl("PJM", "k") + "&offset=960");
  assert.ok(!demandUrl("PJM", "k").includes("offset"));
  assert.equal(demandUrl("PJM", "k", 96), demandUrl("PJM", "k") + "&offset=96");
});

// ── The two pollers, against a fake EIA that holds `published` hours ─────────

/** Every archived row, plain and gzipped day files alike (the pollers gzip a
 *  day file once it is three days old). */
interface ArchivedRow { respondent: string; period: string; type?: string; fueltype?: string }
const archivedRows = (dir: string): ArchivedRow[] => fs.readdirSync(dir)
  .filter((f) => /\.jsonl(\.gz)?$/.test(f))
  .flatMap((f) => {
    const buf = fs.readFileSync(path.join(dir, f));
    return (f.endsWith(".gz") ? zlib.gunzipSync(buf) : buf).toString("utf8").split("\n").filter(Boolean);
  })
  .map((l) => JSON.parse(l) as ArchivedRow);
const archivedHours = (dir: string, respondent: string) =>
  new Set(archivedRows(dir).filter((o) => o.respondent === respondent).map((o) => o.period));
const param = (url: string, name: string) => new URL(url).searchParams.get(name);

/** Fake EIA: every respondent has hours 0..published-1; `rowsFor` gives the
 *  rows of one hour. Serves `length` rows from `offset`, newest first. */
function fakeEia(state: { published: number; fail?: (resp: string, offset: number) => boolean },
                 rowsFor: (resp: string, period: string) => unknown[]) {
  const calls: Array<{ resp: string; offset: number }> = [];
  const impl = async (url: string) => {
    const resp = new URL(url).searchParams.get("facets[respondent][]") || "X";
    const offset = Number(param(url, "offset") || 0);
    const length = Number(param(url, "length"));
    calls.push({ resp, offset });
    if (state.fail?.(resp, offset)) return { ok: false, status: 504, text: async () => "" };
    const all: unknown[] = [];
    for (let h = state.published - 1; h >= 0; h--) all.push(...rowsFor(resp, hour(h)));
    return { ok: true, status: 200, text: async () => JSON.stringify({ response: { data: all.slice(offset, offset + length) } }) };
  };
  return { impl, calls };
}

const GEN_FUELS = Array.from({ length: 16 }, (_, i) => `F${i}`); // US48's 16 fuel types: 60 hours a page
const genRows = (resp: string, period: string) =>
  GEN_FUELS.map((f) => ({ period, respondent: resp, fueltype: f, value: "10" }));

test("generation: EIA silent for 5 days then publishing at once leaves no missing hour, where one page would have lost 60", async () => {
  _resetGridGenerationForTests();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "eia930-gen-"));
  const dir = path.join(base, "gridgeneration");
  const state = { published: 100 };
  const eia = fakeEia(state, genRows);
  const env = { EIA_API_KEY: "k" } as any;
  await refreshGeneration(eia.impl as any, env, Date.now(), base, 0);
  assert.equal(eia.calls.length, GEN_RESPONDENTS.length, "first run: one page per respondent, no read-back");
  assert.equal(archivedHours(dir, "US48").size, 60);

  eia.calls.length = 0;
  await refreshGeneration(eia.impl as any, env, Date.now(), base, 0);
  assert.equal(eia.calls.length, GEN_RESPONDENTS.length, "nothing new: still one request per respondent");
  assert.deepEqual(latestGenerationPoll()!.catch_up, []);
  assert.equal(latestGenerationPoll()!.rows_archived, 0);

  // the control: what the one-page reader does with the same catch-up
  state.published = 220; // 120 new hours appear at once
  const onePage = await fetchGeneration(eia.impl as any, env, Date.now(), 0);
  const onePageHours = new Set(onePage.filter((o) => o.respondent === "US48").map((o) => o.period));
  let lost = 0;
  for (let h = 100; h < 220; h++) if (!onePageHours.has(hour(h))) lost++;
  assert.equal(lost, 60, "one page of 960 rows holds 60 of the 120 new US48 hours");

  eia.calls.length = 0;
  await refreshGeneration(eia.impl as any, env, Date.now(), base, 0);
  const got = archivedHours(dir, "US48");
  for (let h = 40; h < 220; h++) assert.ok(got.has(hour(h)), `US48 hour ${h} is archived`);
  const note = latestGenerationPoll()!.catch_up.find((c) => c.respondent === "US48")!;
  assert.equal(note.outcome, "joined");
  assert.equal(note.extra_pages, 2);
  assert.equal(latestGenerationPoll()!.newest_period, hour(219));
  assert.equal(latestGenerationPoll()!.rows_archived, 120 * 16 * GEN_RESPONDENTS.length);
  // the cache describes the newest page only, as before
  assert.equal(latestGeneration()!.stats.find((s) => s.respondent === "US48")!.hours_in_window, 60);
  // no row is written twice
  const rows = archivedRows(dir);
  assert.equal(new Set(rows.map((o) => `${o.respondent}|${o.period}|${o.fueltype}`)).size, rows.length);
});

test("generation: a read-back that fails archives nothing for that respondent, so the next poll still sees the hole and fills it", async () => {
  _resetGridGenerationForTests();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "eia930-genfail-"));
  const dir = path.join(base, "gridgeneration");
  const state: { published: number; fail?: (resp: string, offset: number) => boolean } = { published: 100 };
  const eia = fakeEia(state, genRows);
  const env = { EIA_API_KEY: "k" } as any;
  await refreshGeneration(eia.impl as any, env, Date.now(), base, 0);

  state.published = 220;
  state.fail = (resp, offset) => resp === "US48" && offset > 0;
  await refreshGeneration(eia.impl as any, env, Date.now(), base, 0);
  assert.equal(archivedHours(dir, "US48").size, 60, "US48: nothing written while its hole cannot be read");
  assert.ok(archivedHours(dir, "PJM").has(hour(219)), "the other respondents are archived as usual");
  assert.equal(latestGenerationPoll()!.catch_up.find((c) => c.respondent === "US48")!.outcome, "failed");
  assert.equal(latestGeneration()!.stats.find((s) => s.respondent === "US48")!.latest_period, hour(219),
    "the live cache still shows the newest hour");

  state.fail = undefined;
  await refreshGeneration(eia.impl as any, env, Date.now(), base, 0);
  const got = archivedHours(dir, "US48");
  for (let h = 40; h < 220; h++) assert.ok(got.has(hour(h)), `US48 hour ${h} is archived after the retry`);
});

test("generation: a restart seeds from the day files before the first request, so the read-back still happens", async () => {
  _resetGridGenerationForTests();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "eia930-genboot-"));
  const dir = path.join(base, "gridgeneration");
  const state = { published: 100 };
  const eia = fakeEia(state, genRows);
  const env = { EIA_API_KEY: "k" } as any;
  await refreshGeneration(eia.impl as any, env, Date.now(), base, 0);
  _resetGridGenerationForTests(); // the process restarts: memory is empty, the files remain
  state.published = 220;
  await refreshGeneration(eia.impl as any, env, Date.now(), base, 0);
  const got = archivedHours(dir, "US48");
  for (let h = 40; h < 220; h++) assert.ok(got.has(hour(h)), `US48 hour ${h} is archived`);
});

const demandRows = (resp: string, period: string) => [
  { period, respondent: resp, type: "D", value: "100" },
  { period, respondent: resp, type: "DF", value: "101" },
];

test("demand: a stall longer than the 48-hour page is read back whole, forecast rows included", async () => {
  _resetGridDemandForTests();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "eia930-dem-"));
  const dir = path.join(base, "griddemand");
  const state = { published: 100 };
  const eia = fakeEia(state, demandRows);
  const env = { EIA_API_KEY: "k" } as any;
  await refreshDemand(eia.impl as any, env, Date.now(), base, 0);
  assert.equal(eia.calls.length, DEMAND_RESPONDENTS.length);
  assert.equal(archivedHours(dir, "NW").size, DEMAND_HOURS);

  state.published = 230; // 130 hours published at once
  eia.calls.length = 0;
  await refreshDemand(eia.impl as any, env, Date.now(), base, 0);
  const got = archivedHours(dir, "NW");
  for (let h = 100 - DEMAND_HOURS; h < 230; h++) assert.ok(got.has(hour(h)), `NW hour ${h} is archived`);
  const note = latestDemandPoll()!.catch_up.find((c) => c.respondent === "NW")!;
  assert.equal(note.outcome, "joined");
  assert.equal(note.extra_pages, 2);
  assert.equal(latestDemandPoll()!.rows_archived, 130 * 2 * DEMAND_RESPONDENTS.length);
  assert.equal(eia.calls.length, 3 * DEMAND_RESPONDENTS.length);
});

test("demand: an hour whose forecast is archived but whose demand is not does not count as reached", async () => {
  _resetGridDemandForTests();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "eia930-demdf-"));
  const dir = path.join(base, "griddemand");
  // before the stall EIA had demand to hour 99 and the day-ahead forecast to hour 199
  let demandTo = 100;
  const impl = async (url: string) => {
    const resp = new URL(url).searchParams.get("facets[respondent][]") || "X";
    const offset = Number(param(url, "offset") || 0);
    const length = Number(param(url, "length"));
    const all: unknown[] = [];
    for (let h = 199; h >= 0; h--) {
      if (h < demandTo) all.push({ period: hour(h), respondent: resp, type: "D", value: "100" });
      all.push({ period: hour(h), respondent: resp, type: "DF", value: "101" });
    }
    return { ok: true, status: 200, text: async () => JSON.stringify({ response: { data: all.slice(offset, offset + length) } }) };
  };
  const env = { EIA_API_KEY: "k" } as any;
  await refreshDemand(impl as any, env, Date.now(), base, 0);
  await refreshDemand(impl as any, env, Date.now(), base, 0);
  await refreshDemand(impl as any, env, Date.now(), base, 0); // forecast rows now archived well behind the newest page
  demandTo = 200; // the missing demand hours are published at once
  await refreshDemand(impl as any, env, Date.now(), base, 0);
  const demandHours = new Set(archivedRows(dir).filter((o) => o.respondent === "NW" && o.type === "D").map((o) => o.period));
  let missing = 0;
  for (let h = 100; h < 200; h++) if (!demandHours.has(hour(h))) missing++;
  assert.equal(missing, 0, "every demand hour published during the stall is archived");
});

test("generation: when EIA publishes between two pages of a read-back the pages overlap, and each row is still written once", async () => {
  _resetGridGenerationForTests();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "eia930-genmove-"));
  const dir = path.join(base, "gridgeneration");
  const state = { published: 100 };
  const eia = fakeEia(state, genRows);
  // two more hours appear after each respondent's first page of a read-back:
  // every later offset now points 32 rows newer than it did
  const moving = async (url: string) => {
    const res = await eia.impl(url);
    if (state.published >= 220 && Number(param(url, "offset") || 0) === 0) state.published += 2;
    return res;
  };
  const env: NodeJS.ProcessEnv = { EIA_API_KEY: "k" };
  await refreshGeneration(moving, env, Date.now(), base, 0);
  state.published = 220;
  await refreshGeneration(moving, env, Date.now(), base, 0);
  const rows = archivedRows(dir);
  assert.equal(new Set(rows.map((o) => `${o.respondent}|${o.period}|${o.fueltype}`)).size, rows.length, "no row is written twice");
  const got = archivedHours(dir, "US48");
  for (let h = 40; h < 220; h++) assert.ok(got.has(hour(h)), `US48 hour ${h} is archived`);
});

test("demand: a row the quality gate refuses at the edge of the page is not a missing hour, so the poll stays one request", async () => {
  _resetGridDemandForTests();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "eia930-demq-"));
  const dir = path.join(base, "griddemand");
  const state = { published: 100 };
  // the oldest hour of the page (hour 52) carries a negative demand, which the gate refuses on every poll
  const eia = fakeEia(state, (resp, period) => [
    { period, respondent: resp, type: "D", value: period === hour(100 - DEMAND_HOURS) ? "-5" : "100" },
    { period, respondent: resp, type: "DF", value: "101" },
  ]);
  const env: NodeJS.ProcessEnv = { EIA_API_KEY: "k" };
  await refreshDemand(eia.impl, env, Date.now(), base, 0);
  eia.calls.length = 0;
  await refreshDemand(eia.impl, env, Date.now(), base, 0);
  assert.equal(eia.calls.length, DEMAND_RESPONDENTS.length, "one request per respondent");
  assert.deepEqual(latestDemandPoll()!.catch_up, []);
  assert.ok(!archivedRows(dir).some((o) => o.period === hour(100 - DEMAND_HOURS) && o.type === "D"), "the refused row is not archived");
});
