// Wikimedia attention battery (BUILD ORDER 5 #3): API-shape parse,
// polite spacing + 404-absence honesty, dedup by view day, majority
// panel-day selection, gz lifecycle.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  parsePageviews, fetchAttention, archiveAttention, gzipOldAttentionDays,
  pickLatestCompleteDay, lastAttentionCycle, ARTICLES, REQUEST_SPACING_MS,
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
