// Wikimedia attention battery (BUILD ORDER 5 #3): API-shape parse,
// polite spacing + 404-absence honesty, dedup by view day, majority
// panel-day selection, gz lifecycle.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import {
  parsePageviews, fetchAttention, archiveAttention, gzipOldAttentionDays,
  pickLatestCompleteDay, lastAttentionCycle, ARTICLES, REQUEST_SPACING_MS,
  listArchivedDates, readArchivedDay, lookupTickerHistory, readAggregateHistory,
} from "./wikiAttention";

const ITEMS = (article: string, days: Array<[string, number]>) => ({
  items: days.map(([ts, views]) => ({
    project: "en.wikipedia", article, granularity: "daily",
    timestamp: `${ts}00`, access: "all-access", agent: "user", views,
  })),
});

test("seed map: bundled statically, non-empty, RIOT honestly absent", () => {
  assert.ok(Object.keys(ARTICLES).length >= 20);
  assert.equal(ARTICLES.NVDA, "Nvidia");
  assert.equal(ARTICLES.RIOT, undefined, "RIOT dropped at curation (renamed article)");
});

test("parsePageviews: documented items[] shape, malformed items dropped", () => {
  const obs = parsePageviews(ITEMS("Nvidia", [["20260701", 10883], ["20260702", 11200]]),
                             "NVDA", "Nvidia", "2026-07-05");
  assert.equal(obs.length, 2);
  assert.deepEqual([obs[0].date, obs[0].ticker, obs[0].views], ["2026-07-01", "NVDA", 10883]);
  assert.deepEqual(parsePageviews(null, "X", "X", "x"), []);
  assert.deepEqual(parsePageviews({ items: [{ timestamp: "bad", views: 5 }] }, "X", "X", "x"), []);
});

test("fetchAttention: hits every seed article once, skips 404 silently (absence is data)", async () => {
  const urls: string[] = [];
  const fake = async (url: string) => {
    urls.push(url);
    if (url.includes("Oklo")) return { ok: false, status: 404, text: async () => "" };
    return { ok: true, status: 200,
             text: async () => JSON.stringify(ITEMS("A", [["20260704", 100]])) };
  };
  const obs = await fetchAttention(fake as any, Date.parse("2026-07-05T12:00:00Z"), 7, 0);
  assert.equal(urls.length, Object.keys(ARTICLES).length, "one request per seed article");
  assert.equal(obs.length, Object.keys(ARTICLES).length - 1, "404 article contributes nothing");
  assert.ok(REQUEST_SPACING_MS >= 500, "spacing respects the observed 429 limit");
  const cyc = lastAttentionCycle()!;
  assert.equal(cyc.ok_articles, Object.keys(ARTICLES).length - 1);
  assert.equal(cyc.not_found, 1);
  assert.equal(cyc.err_articles, 0);
  assert.ok(cyc.finished, "cycle stats close out");
});

test("cycle stats capture the failure mode when every request errors (R7 diagnosability)", async () => {
  const blocked = async () => ({ ok: false, status: 403, text: async () => "" });
  const obs = await fetchAttention(blocked as any, Date.parse("2026-07-05T12:00:00Z"), 7, 0);
  assert.equal(obs.length, 0);
  const cyc = lastAttentionCycle()!;
  assert.equal(cyc.err_articles, Object.keys(ARTICLES).length);
  assert.match(cyc.last_error!, /-> 403$/, "route-visible last_error names the status");
  assert.equal(cyc.obs, 0);
});

test("archive: dedup by date|ticker, day-files by VIEW date, gz at 4d", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "wiki-"));
  const now = Date.parse("2026-07-05T12:00:00Z");
  const obs = [
    ...parsePageviews(ITEMS("Nvidia", [["20260630", 10000], ["20260702", 10883]]), "NVDA", "Nvidia", "2026-07-05"),
    ...parsePageviews(ITEMS("GameStop", [["20260702", 4000]]), "GME", "GameStop", "2026-07-05"),
  ];
  assert.equal(archiveAttention(obs, base, now), 3);
  assert.equal(archiveAttention(obs, base, now), 0, "same day|ticker never re-archives");
  assert.ok(fs.existsSync(path.join(base, "wikiattention", "2026-06-30.jsonl")));
  assert.ok(fs.existsSync(path.join(base, "wikiattention", "2026-07-02.jsonl")));
  assert.equal(gzipOldAttentionDays(base, now), 1,
    "06-30 (5.5d) gzips; 07-02 (3.5d) stays plain under the 4d rule");
  assert.ok(fs.existsSync(path.join(base, "wikiattention", "2026-06-30.jsonl.gz")));
});

test("pickLatestCompleteDay: an in-progress publish day never masquerades as the panel", () => {
  const full = Object.keys(ARTICLES).map((t) =>
    ({ date: "2026-07-03", ticker: t, article: ARTICLES[t], views: 100, rt: "x" }));
  const partial = [
    { date: "2026-07-04", ticker: "NVDA", article: "Nvidia", views: 500, rt: "x" },
    { date: "2026-07-04", ticker: "GME", article: "GameStop", views: 400, rt: "x" },
  ];
  const day = pickLatestCompleteDay([...full, ...partial])!;
  assert.equal(day.date, "2026-07-03", "majority rule skips the 2-article partial day");
  assert.equal(day.tickers.length, Object.keys(ARTICLES).length);
  assert.equal(pickLatestCompleteDay([]), null);
});

// History reads (#/data/attention full view). Day-files written directly
// to disk (not via archiveAttention) — archivedKeys dedup is module-level
// state shared across this whole test file regardless of baseDir, same
// caveat finraShortVolume.test.ts notes for its own archivedDates set.
test("listArchivedDates + readArchivedDay: newest-first, plain and gz both readable", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "wiki-hist-"));
  const dir = path.join(base, "wikiattention");
  fs.mkdirSync(dir, { recursive: true });
  const row = (date: string, ticker: string, views: number) =>
    JSON.stringify({ date, ticker, article: ARTICLES[ticker] || ticker, views, rt: date });
  fs.writeFileSync(path.join(dir, "2026-05-01.jsonl"), `${row("2026-05-01", "NVDA", 1000)}\n${row("2026-05-01", "GME", 200)}\n`);
  fs.writeFileSync(path.join(dir, "2026-05-02.jsonl.gz"), zlib.gzipSync(`${row("2026-05-02", "NVDA", 1100)}\n`));
  assert.deepEqual(listArchivedDates(base, 90), ["2026-05-02", "2026-05-01"], "newest first");
  assert.deepEqual(listArchivedDates(base, 1), ["2026-05-02"], "limit respected");
  assert.equal(readArchivedDay("2026-05-01", base).length, 2);
  assert.equal(readArchivedDay("2026-05-02", base).length, 1, "gz day readable");
  assert.deepEqual(readArchivedDay("2026-05-03", base), [], "missing day = empty, not a throw");
});

test("lookupTickerHistory: ascending by date, honest gap on absent ticker-day, case-insensitive", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "wiki-tick-"));
  const dir = path.join(base, "wikiattention");
  fs.mkdirSync(dir, { recursive: true });
  const row = (date: string, ticker: string, views: number) =>
    JSON.stringify({ date, ticker, article: ARTICLES[ticker] || ticker, views, rt: date });
  fs.writeFileSync(path.join(dir, "2026-05-10.jsonl"), `${row("2026-05-10", "nvda", 500)}\n`);
  fs.writeFileSync(path.join(dir, "2026-05-11.jsonl"), `${row("2026-05-11", "GME", 400)}\n`); // no NVDA this day
  fs.writeFileSync(path.join(dir, "2026-05-12.jsonl"), `${row("2026-05-12", "NVDA", 700)}\n`);
  const series = lookupTickerHistory("NVDA", 90, base);
  assert.deepEqual(series.map((r) => r.date), ["2026-05-10", "2026-05-12"], "05-11 honestly omitted, not zero-filled");
  assert.equal(series[0].views, 500, "case-insensitive ticker match");
  assert.equal(series[1].views, 700);
  assert.deepEqual(lookupTickerHistory("NOPE", 90, base), []);
});

test("readAggregateHistory: seed-total views + ticker count per archived day, ascending by date", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "wiki-agg-"));
  const dir = path.join(base, "wikiattention");
  fs.mkdirSync(dir, { recursive: true });
  const row = (date: string, ticker: string, views: number) =>
    JSON.stringify({ date, ticker, article: ARTICLES[ticker] || ticker, views, rt: date });
  fs.writeFileSync(path.join(dir, "2026-05-20.jsonl"), `${row("2026-05-20", "NVDA", 500)}\n${row("2026-05-20", "GME", 300)}\n`);
  fs.writeFileSync(path.join(dir, "2026-05-21.jsonl"), `${row("2026-05-21", "NVDA", 600)}\n`);
  const trend = readAggregateHistory(90, base);
  assert.deepEqual(trend.map((t) => t.date), ["2026-05-20", "2026-05-21"], "ascending by date");
  assert.equal(trend[0].tickers, 2);
  assert.equal(trend[0].total_views, 800);
  assert.equal(trend[1].tickers, 1);
  assert.equal(trend[1].total_views, 600);
  assert.deepEqual(readAggregateHistory(5, path.join(base, "nonexistent")), [], "missing dir = empty, not a throw");
});
