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
