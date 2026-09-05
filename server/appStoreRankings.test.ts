import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import os from "os";
import path from "path";
import { fileURLToPath } from "url";
import {
  parseChart,
  parseLookup,
  fetchAppStoreSnapshot,
  archiveAppStoreRecords,
  listArchivedAppStoreDates,
  readArchivedAppStoreDay,
  lookupAppStoreTickerHistory,
  readAppStoreAggregateHistory,
  WATCHLIST,
} from "./appStoreRankings";

const here = path.dirname(fileURLToPath(import.meta.url));

const TEST_WATCHLIST = [
  { ticker: "DUOL", company: "Duolingo, Inc.", appId: "570060128", appName: "Duolingo" },
  { ticker: "BMBL", company: "Bumble Inc.", appId: "930441707", appName: "Bumble" },
];

// Real-shape marketingtools RSS response (verified live 2026-08-01):
// feed.results[] ordered by rank, each carrying a string "id".
const CHART_JSON = {
  feed: {
    results: [
      { id: "407558537", name: "Capital One Mobile" },
      { id: "570060128", name: "Duolingo: Language Lessons" }, // rank 2
      { id: "999999999", name: "Some Other App" },
    ],
  },
};

// Real-shape iTunes Lookup response (verified live 2026-08-01).
const LOOKUP_JSON = {
  results: [
    { trackId: 570060128, trackName: "Duolingo: Language Lessons", averageUserRating: 4.72504, userRatingCount: 5357702, version: "7.133.0" },
  ],
};

test("parseChart: ranks found watchlist apps, records null (not fabricated) for the rest", () => {
  const rows = parseChart(CHART_JSON, "us", "top-free", TEST_WATCHLIST, "2026-08-01");
  assert.equal(rows.length, 2, "one row per watchlist app, present or not");
  const duol = rows.find((r) => r.ticker === "DUOL")!;
  assert.equal(duol.rank, 2);
  assert.equal(duol.company, "Duolingo, Inc.");
  assert.equal(duol.key, "2026-08-01|us|top-free|DUOL");
  const bmbl = rows.find((r) => r.ticker === "BMBL")!;
  assert.equal(bmbl.rank, null, "app absent from this chart must be null, never a fabricated worst rank");
});

test("parseLookup: rating fields mapped; missing watchlist apps get null fields, never dropped", () => {
  const rows = parseLookup(LOOKUP_JSON, TEST_WATCHLIST, "2026-08-01");
  assert.equal(rows.length, 2, "row emitted for every watchlist app even when Lookup omits it");
  const duol = rows.find((r) => r.ticker === "DUOL")!;
  assert.equal(duol.avgRating, 4.72504);
  assert.equal(duol.ratingCount, 5357702);
  assert.equal(duol.version, "7.133.0");
  assert.equal(duol.company, "Duolingo, Inc.");
  assert.equal(duol.key, "2026-08-01|DUOL");
  const bmbl = rows.find((r) => r.ticker === "BMBL")!;
  assert.equal(bmbl.avgRating, null);
  assert.equal(bmbl.ratingCount, null);
});

test("fetchAppStoreSnapshot: one dead storefront/chart never drops the rest of the cycle", async () => {
  let calls = 0;
  const impl = async (url: string) => {
    calls++;
    if (url.includes("/gb/apps/top-free/")) return { ok: false, status: 500, text: async () => "" };
    if (url.includes("itunes.apple.com/lookup")) return { ok: true, status: 200, text: async () => JSON.stringify(LOOKUP_JSON) };
    return { ok: true, status: 200, text: async () => JSON.stringify(CHART_JSON) };
  };
  const rows = await fetchAppStoreSnapshot(impl as any, Date.parse("2026-08-01T12:00:00Z"));
  // 3 storefronts x 2 charts = 6 chart calls (one fails) + 1 lookup call = 7
  assert.equal(calls, 7);
  const rankRows = rows.filter((r) => r.t === "rank");
  const ratingRows = rows.filter((r) => r.t === "rating");
  // 5 surviving chart fetches x WATCHLIST.length rows each
  assert.equal(rankRows.length, 5 * WATCHLIST.length);
  assert.equal(ratingRows.length, WATCHLIST.length);
});

test("archive round-trip with dedup by key", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "vtappstore-"));
  const t0 = Date.parse("2026-08-01T12:00:00Z");
  const rows = parseChart(CHART_JSON, "us", "top-free", TEST_WATCHLIST, "2026-08-01");
  assert.equal(archiveAppStoreRecords(rows, dir, t0), 2);
  assert.equal(archiveAppStoreRecords(rows, dir, t0), 0, "same keys never archive twice");
  fs.rmSync(dir, { recursive: true, force: true });
});

// NOTE: archiveAppStoreRecords' dedup Set is module-level (shared across
// every test in this process, regardless of tmpdir), so each test below
// uses dates no other test in this file touches — reusing "2026-08-01"
// a second time would silently no-op the write (see the "archive
// round-trip" test just above, which already claims that date).

test("history: listArchivedAppStoreDates + readArchivedAppStoreDay round-trip across two days", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "vtappstore-"));
  const day1 = parseChart(CHART_JSON, "us", "top-free", TEST_WATCHLIST, "2027-01-04");
  const day2 = parseChart(CHART_JSON, "us", "top-free", TEST_WATCHLIST, "2027-01-05");
  archiveAppStoreRecords(day1, dir, Date.parse("2027-01-04T12:00:00Z"));
  archiveAppStoreRecords(day2, dir, Date.parse("2027-01-05T12:00:00Z"));
  const dates = listArchivedAppStoreDates(dir, 90);
  assert.deepEqual(dates, ["2027-01-05", "2027-01-04"], "newest first");
  const rows = readArchivedAppStoreDay("2027-01-04", dir);
  assert.equal(rows.length, 2);
  fs.rmSync(dir, { recursive: true, force: true });
});

test("lookupAppStoreTickerHistory: one ticker's rank + rating series, ascending, other tickers excluded, unfetched days never zero-filled", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "vtappstore-"));
  const ranks1 = parseChart(CHART_JSON, "us", "top-free", TEST_WATCHLIST, "2027-02-01");
  const ratings1 = parseLookup(LOOKUP_JSON, TEST_WATCHLIST, "2027-02-01");
  archiveAppStoreRecords([...ranks1, ...ratings1], dir, Date.parse("2027-02-01T12:00:00Z"));
  const ranks2 = parseChart(CHART_JSON, "us", "top-free", TEST_WATCHLIST, "2027-02-02");
  archiveAppStoreRecords(ranks2, dir, Date.parse("2027-02-02T12:00:00Z")); // no ratings fetched this day
  const series = lookupAppStoreTickerHistory("duol", 10, dir);
  assert.equal(series.length, 2, "ascending across both archived days");
  assert.deepEqual(series.map((p) => p.date), ["2027-02-01", "2027-02-02"]);
  assert.equal(series[0].ranks.length, 1);
  assert.equal(series[0].ranks[0].rank, 2);
  assert.ok(series[0].rating, "2027-02-01 has a rating row");
  assert.equal(series[0].rating!.ratingCount, 5357702);
  assert.equal(series[1].rating, null, "2027-02-02 never fetched ratings — honestly null, not carried over");
  const bmbl = lookupAppStoreTickerHistory("bmbl", 10, dir);
  assert.equal(bmbl[0].ranks[0].rank, null, "BMBL was outside the top 100 in the fixture chart");
  fs.rmSync(dir, { recursive: true, force: true });
});

test("readAppStoreAggregateHistory: ranked-slot ratio + summed rating counts per day, ascending", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "vtappstore-"));
  const ranks1 = parseChart(CHART_JSON, "us", "top-free", TEST_WATCHLIST, "2027-03-01");
  const ratings1 = parseLookup(LOOKUP_JSON, TEST_WATCHLIST, "2027-03-01");
  archiveAppStoreRecords([...ranks1, ...ratings1], dir, Date.parse("2027-03-01T12:00:00Z"));
  const trend = readAppStoreAggregateHistory(10, dir);
  assert.equal(trend.length, 1);
  assert.equal(trend[0].date, "2027-03-01");
  assert.equal(trend[0].total_slots, 2, "both watchlist apps fetched for this one chart");
  assert.equal(trend[0].ranked_slots, 1, "only DUOL landed a non-null rank; BMBL was outside the top 100");
  assert.equal(trend[0].total_rating_count, 5357702, "BMBL's missing rating row contributes 0, not a crash");
  fs.rmSync(dir, { recursive: true, force: true });
});

test("WATCHLIST entries are well-formed and unique", () => {
  assert.ok(WATCHLIST.length >= 10, "watchlist should be a real panel, not a stub");
  const tickers = new Set(WATCHLIST.map((a) => a.ticker));
  assert.equal(tickers.size, WATCHLIST.length, "no duplicate tickers");
  const appIds = new Set(WATCHLIST.map((a) => a.appId));
  assert.equal(appIds.size, WATCHLIST.length, "no duplicate app ids");
  for (const a of WATCHLIST) {
    assert.ok(/^\d+$/.test(a.appId), `${a.ticker} appId must be numeric`);
  }
});

test("routes.ts boots the App Store poll and registers /api/data/appstore-rankings; manifest states the honesty gaps", () => {
  const routes = fs.readFileSync(path.join(here, "routes.ts"), "utf8");
  assert.ok(routes.includes("bootAppStorePoll()"), "App Store poll must boot eagerly");
  assert.ok(routes.includes('"/api/data/appstore-rankings"'), "route registered");
  const manifest = JSON.parse(fs.readFileSync(path.join(here, "..", "datacore", "manifests", "appstore.json"), "utf8"));
  assert.equal(manifest.stream, "appstore");
  assert.ok(String(manifest.license).includes("Google Play"), "Android exclusion must be stated");
  assert.ok(String(manifest.field_map.rank).includes("never fabricated"), "rank:null honesty must be stated");
  assert.ok(String(manifest.confidence_model).includes("MOST arbitraged"), "sober prior must be stated in the manifest");
});
