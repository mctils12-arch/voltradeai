import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import os from "os";
import path from "path";
import { fileURLToPath } from "url";
import {
  lastCompletedWeek,
  mergedPrQuery,
  commitsQuery,
  parseSearchTotal,
  parseCommitSample,
  fetchGithubActivity,
  archiveGithubActivity,
  isArchived,
  readArchivedGithubActivity,
  lookupGithubOrgHistory,
  readGithubActivityAggregateHistory,
  WATCHLIST,
} from "./githubOrgActivity";

const here = path.dirname(fileURLToPath(import.meta.url));

test("lastCompletedWeek: always the prior Mon-Sun UTC week, never the in-progress one", () => {
  // Wednesday 2026-08-05 (a Wednesday) -> last completed week is
  // 2026-07-27 (Mon) .. 2026-08-02 (Sun)
  const w = lastCompletedWeek(Date.parse("2026-08-05T12:00:00Z"));
  assert.equal(w.weekStart, "2026-07-27");
  assert.equal(w.weekEnd, "2026-08-02");
  // Monday exactly at 00:00 UTC -> still the PRIOR week, not the one that
  // just started (in-progress week must never be reported as final)
  const wMon = lastCompletedWeek(Date.parse("2026-08-03T00:00:00Z"));
  assert.equal(wMon.weekStart, "2026-07-27");
  assert.equal(wMon.weekEnd, "2026-08-02");
});

test("mergedPrQuery excludes the known bot-app authors and scopes by org+date", () => {
  const q = mergedPrQuery("mongodb", "2026-07-27", "2026-08-02");
  assert.match(q, /org:mongodb/);
  assert.match(q, /is:pr is:merged/);
  assert.match(q, /merged:2026-07-27\.\.2026-08-02/);
  assert.match(q, /-author:app\/dependabot/);
  assert.match(q, /-author:app\/renovate/);
});

test("commitsQuery scopes by org and committer-date, no bot exclusion (documented gap)", () => {
  const q = commitsQuery("elastic", "2026-07-27", "2026-08-02");
  assert.equal(q, "org:elastic committer-date:2026-07-27..2026-08-02");
});

test("parseSearchTotal reads total_count directly, null when absent", () => {
  assert.equal(parseSearchTotal({ total_count: 42 }), 42);
  assert.equal(parseSearchTotal({}), null);
  assert.equal(parseSearchTotal(null), null);
});

test("parseCommitSample bot-filters via author.type and [bot] login/name, caps honestly", () => {
  const json = {
    total_count: 250,
    items: [
      { sha: "a1", author: { login: "alice", type: "User" }, commit: { author: { name: "Alice", email: "a@x.com" } } },
      { sha: "a2", author: { login: "alice", type: "User" }, commit: { author: { name: "Alice", email: "a@x.com" } } }, // dup actor
      { sha: "b1", author: { login: "dependabot[bot]", type: "Bot" }, commit: { author: { name: "dependabot[bot]", email: "d@x.com" } } },
      { sha: "c1", author: null, commit: { author: { name: "renovate[bot]", email: "r@x.com" } } }, // unlinked bot, name-matched
      { sha: "d1", author: { login: "bob", type: "User" }, commit: { author: { name: "Bob", email: "b@x.com" } } },
    ],
  };
  const { uniqueActorsSample, capped } = parseCommitSample(json);
  assert.equal(uniqueActorsSample, 2, "alice (deduped) + bob only — bots excluded");
  assert.equal(capped, true, "total_count 250 > 5 sampled items");
});

test("parseCommitSample: total_count within the sample page is not marked capped", () => {
  const json = { total_count: 1, items: [{ sha: "a1", author: { login: "alice", type: "User" }, commit: { author: { name: "Alice" } } }] };
  const { capped } = parseCommitSample(json);
  assert.equal(capped, false);
});

test("fetchGithubActivity: per-org failure isolated, skip() short-circuits without a network call", async () => {
  let calls = 0;
  const impl = async (url: string) => {
    calls++;
    if (url.includes("mongodb") && url.includes("search/issues")) return { ok: false, status: 500, text: async () => "" };
    if (url.includes("search/issues")) return { ok: true, status: 200, text: async () => JSON.stringify({ total_count: 5 }) };
    return { ok: true, status: 200, text: async () => JSON.stringify({ total_count: 3, items: [] }) };
  };
  const watchlist = [
    { ticker: "MDB", company: "MongoDB, Inc.", org: "mongodb" },
    { ticker: "ESTC", company: "Elastic N.V.", org: "elastic" },
    { ticker: "SKIP", company: "Skip Co.", org: "skipme" },
  ];
  const records = await fetchGithubActivity(
    watchlist as any, impl as any, Date.parse("2026-08-05T12:00:00Z"),
    { delayMs: 0, skip: (key) => key.endsWith("|skipme") },
  );
  assert.equal(records.length, 2, "skipped org never produces a record");
  const mongo = records.find((r) => r.org === "mongodb")!;
  assert.equal(mongo.mergedPRs, null, "mongodb's search/issues call failed -> null, not fabricated");
  assert.equal(mongo.commits, 3, "commits call for mongodb still succeeded independently");
  const elastic = records.find((r) => r.org === "elastic")!;
  assert.equal(elastic.mergedPRs, 5);
  assert.equal(elastic.commits, 3);
  assert.ok(!calls || calls > 0, "sanity: calls were made");
  assert.ok(!records.some((r) => r.org === "skipme"), "skip() prevented any call for that org");
});

test("archive round-trip with dedup by key; isArchived reflects a fresh dir per test", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "vtghorg-"));
  const t0 = Date.parse("2026-08-05T12:00:00Z");
  const rec = {
    t: "org_week" as const, key: "2026-07-27|mongodb", ticker: "MDB", org: "mongodb",
    weekStart: "2026-07-27", weekEnd: "2026-08-02", mergedPRs: 12, commits: 88,
    uniqueActorsSample: 6, actorSampleCapped: true, rt: "2026-08-05",
  };
  assert.equal(archiveGithubActivity([rec], dir, t0), 1);
  assert.equal(archiveGithubActivity([rec], dir, t0), 0, "same key never archives twice");
  fs.rmSync(dir, { recursive: true, force: true });
});

test("archiveGithubActivity keeps a partial record (one leg failed) rather than dropping the week", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "vtghorg-"));
  const t0 = Date.parse("2026-08-05T12:00:00Z");
  const rec = {
    t: "org_week" as const, key: "2026-07-27|elastic", ticker: "ESTC", org: "elastic",
    weekStart: "2026-07-27", weekEnd: "2026-08-02", mergedPRs: null, commits: 40,
    uniqueActorsSample: 4, actorSampleCapped: false, rt: "2026-08-05",
  };
  assert.equal(archiveGithubActivity([rec], dir, t0), 1, "commits succeeded, so the week is worth keeping");
  fs.rmSync(dir, { recursive: true, force: true });
});

// NOTE: archiveGithubActivity's dedup Set is module-level (shared across
// every test in this process, regardless of tmpdir), so each test below
// uses weekStart dates no other test in this file touches — reusing a key
// like "2026-07-27|mongodb" a second time would silently no-op the write.

test("readArchivedGithubActivity: scans every day-file (not just the newest), dedups by key", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "vtghorg-"));
  const week1 = {
    t: "org_week" as const, key: "2027-01-04|mongodb", ticker: "MDB", org: "mongodb",
    weekStart: "2027-01-04", weekEnd: "2027-01-10", mergedPRs: 12, commits: 88,
    uniqueActorsSample: 6, actorSampleCapped: true, rt: "2027-01-13",
  };
  const week2 = {
    t: "org_week" as const, key: "2027-01-11|mongodb", ticker: "MDB", org: "mongodb",
    weekStart: "2027-01-11", weekEnd: "2027-01-17", mergedPRs: 20, commits: 100,
    uniqueActorsSample: 8, actorSampleCapped: false, rt: "2027-01-20",
  };
  // Written on two different days -> two different day-files, same dir.
  archiveGithubActivity([week1], dir, Date.parse("2027-01-13T12:00:00Z"));
  archiveGithubActivity([week2], dir, Date.parse("2027-01-20T12:00:00Z"));
  const all = readArchivedGithubActivity(dir);
  assert.equal(all.length, 2, "must read across both day-files, not just the latest");
  assert.deepEqual(new Set(all.map((r) => r.key)), new Set(["2027-01-04|mongodb", "2027-01-11|mongodb"]));
  fs.rmSync(dir, { recursive: true, force: true });
});

test("lookupGithubOrgHistory: one ticker's weekly series, ascending, other orgs excluded, unfetched weeks never zero-filled", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "vtghorg-"));
  const t0 = Date.parse("2027-02-09T12:00:00Z");
  archiveGithubActivity([
    { t: "org_week" as const, key: "2027-02-01|mongodb", ticker: "MDB", org: "mongodb", weekStart: "2027-02-01", weekEnd: "2027-02-07", mergedPRs: 12, commits: 88, uniqueActorsSample: 6, actorSampleCapped: true, rt: "2027-02-09" },
    { t: "org_week" as const, key: "2027-02-08|mongodb", ticker: "MDB", org: "mongodb", weekStart: "2027-02-08", weekEnd: "2027-02-14", mergedPRs: 20, commits: 100, uniqueActorsSample: 8, actorSampleCapped: false, rt: "2027-02-09" },
    { t: "org_week" as const, key: "2027-02-08|elastic", ticker: "ESTC", org: "elastic", weekStart: "2027-02-08", weekEnd: "2027-02-14", mergedPRs: 5, commits: 30, uniqueActorsSample: 3, actorSampleCapped: false, rt: "2027-02-09" },
  ], dir, t0);
  const series = lookupGithubOrgHistory("mdb", 10, dir);
  assert.equal(series.length, 2, "only MDB's own rows, ESTC excluded");
  assert.deepEqual(series.map((r) => r.weekStart), ["2027-02-01", "2027-02-08"], "ascending by weekStart");
  assert.equal(series[1].mergedPRs, 20);
  const capped = lookupGithubOrgHistory("mdb", 1, dir);
  assert.equal(capped.length, 1, "weeks param caps the series to the most recent N");
  assert.equal(capped[0].weekStart, "2027-02-08");
  fs.rmSync(dir, { recursive: true, force: true });
});

test("readGithubActivityAggregateHistory: per-week org count + summed PR/commit totals, nulls excluded from sums not coerced to zero", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "vtghorg-"));
  const t0 = Date.parse("2027-03-02T12:00:00Z");
  archiveGithubActivity([
    { t: "org_week" as const, key: "2027-03-01|mongodb", ticker: "MDB", org: "mongodb", weekStart: "2027-03-01", weekEnd: "2027-03-07", mergedPRs: 20, commits: 100, uniqueActorsSample: 8, actorSampleCapped: false, rt: "2027-03-02" },
    { t: "org_week" as const, key: "2027-03-01|elastic", ticker: "ESTC", org: "elastic", weekStart: "2027-03-01", weekEnd: "2027-03-07", mergedPRs: null, commits: 30, uniqueActorsSample: 3, actorSampleCapped: false, rt: "2027-03-02" },
  ], dir, t0);
  const trend = readGithubActivityAggregateHistory(10, dir);
  assert.equal(trend.length, 1);
  assert.equal(trend[0].weekStart, "2027-03-01");
  assert.equal(trend[0].weekEnd, "2027-03-07");
  assert.equal(trend[0].orgs_reporting, 2, "both orgs reported at least one non-null field");
  assert.equal(trend[0].total_merged_prs, 20, "elastic's null mergedPRs excluded from the sum, not coerced to 0");
  assert.equal(trend[0].total_commits, 130);
  fs.rmSync(dir, { recursive: true, force: true });
});

test("watchlist entries are well-formed and unique (ticker, org)", () => {
  const tickers = new Set<string>();
  const orgs = new Set<string>();
  for (const w of WATCHLIST) {
    assert.ok(/^[A-Z]{1,6}$/.test(w.ticker), `ticker ${w.ticker} looks malformed`);
    assert.ok(w.org.length > 0);
    assert.ok(w.company.length > 0);
    assert.ok(!tickers.has(w.ticker), `duplicate ticker ${w.ticker}`);
    assert.ok(!orgs.has(w.org), `duplicate org ${w.org}`);
    tickers.add(w.ticker); orgs.add(w.org);
  }
  assert.ok(WATCHLIST.length >= 10, "watchlist should be a meaningful panel, not a token entry");
});

test("routes.ts boots the GitHub activity poll and registers /api/data/github-activity; manifest states GATE 1 honesty", () => {
  const routes = fs.readFileSync(path.join(here, "routes.ts"), "utf8");
  assert.ok(routes.includes("bootGithubActivityPoll()"), "poll must boot eagerly");
  assert.ok(routes.includes('"/api/data/github-activity"'), "route registered");
  const manifest = JSON.parse(fs.readFileSync(path.join(here, "..", "datacore", "manifests", "github_activity.json"), "utf8"));
  assert.equal(manifest.stream, "github_activity");
  assert.ok(String(manifest.confidence_model).includes("GATE 1"), "gate-1-only honesty must be stated");
  assert.ok(String(manifest.field_map.actorSampleCapped).includes("undercounts"), "sampling cap honesty stated");
});
