/**
 * Hermetic tests for siteReport.ts (owner signup/visit report + the
 * counts-only aggregate behind /api/diag/signups). In-memory SQLite with
 * auth.ts's own column set, temp archive dirs, fixed clock.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import Database from "better-sqlite3";
import { buildSiteReport, aggregateSiteReport, AGGREGATE_DAY_CAP, siteReportNotes, VISIT_REPORT_DAYS, TOP_PAGES_DAYS } from "./siteReport";
import type { VisitHistorySnapshot } from "./visitHistory";
import type { AnalyticsSnapshot } from "./requestLog";

const here = path.dirname(fileURLToPath(import.meta.url));
const NOW = Date.parse("2026-10-13T12:00:00Z");
const ANALYTICS: AnalyticsSnapshot = {
  totalPageViews: 42, uniquesToday: 7,
  uniquesByDay: { "2026-10-12": 11, "2026-10-13": 7 },
  topRoutes: [{ route: "/", count: 30 }, { route: "/developers", count: 12 }],
  ringSize: 0, ringCap: 5000,
};

/** auth.ts's users/sessions columns (incl. its billing ALTERs) + apiKeyAccounts.ts's api_keys. */
function makeDb(withApiKeys = true) {
  const db = new Database(":memory:");
  db.exec(`
    CREATE TABLE users (id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT UNIQUE NOT NULL,
      password_hash TEXT NOT NULL, role TEXT DEFAULT 'user', created_at TEXT NOT NULL,
      tier TEXT DEFAULT 'free', stripe_customer_id TEXT, stripe_subscription_id TEXT,
      subscription_status TEXT, subscription_period_end INTEGER);
    CREATE TABLE sessions (token TEXT PRIMARY KEY, user_id INTEGER NOT NULL, created_at INTEGER NOT NULL);
  `);
  if (withApiKeys) {
    db.exec(`CREATE TABLE api_keys (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL,
      key_hash TEXT NOT NULL UNIQUE, label TEXT NOT NULL, created_at INTEGER NOT NULL,
      last_used_at INTEGER, revoked_at INTEGER)`);
  }
  const add = db.prepare("INSERT INTO users (email, password_hash, role, created_at, tier, subscription_status) VALUES (?, 'x', ?, ?, ?, ?)");
  add.run("owner@example.com", "owner", "2026-04-01T00:00:00.000Z", "free", null);
  add.run("ada@example.com", "user", "2026-08-02T09:30:00.000Z", "free", null);
  add.run("grace@example.org", "user", "2026-09-14T17:05:00.000Z", "pro", "active");
  return db;
}

function makeArchive(lines: string[] | null) {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-sitereport-"));
  if (lines) {
    fs.mkdirSync(path.join(base, "waitlist"));
    fs.writeFileSync(path.join(base, "waitlist", "2026-07-05.jsonl"), lines.join("\n") + "\n");
  }
  return base;
}

test("buildSiteReport: accounts in join order, owner flagged, key use + 24h sign-in, waitlist parsed", () => {
  const db = makeDb();
  db.prepare("INSERT INTO api_keys (user_id, key_hash, label, created_at, last_used_at, revoked_at) VALUES (?, ?, ?, ?, ?, ?)")
    .run(2, "h1", "dev", NOW - 5 * 86_400_000, NOW - 3_600_000, null);
  db.prepare("INSERT INTO api_keys (user_id, key_hash, label, created_at, last_used_at, revoked_at) VALUES (?, ?, ?, ?, ?, ?)")
    .run(2, "h2", "old", NOW - 9 * 86_400_000, null, NOW - 8 * 86_400_000);
  db.prepare("INSERT INTO sessions (token, user_id, created_at) VALUES (?, ?, ?)").run("s-new", 3, NOW - 3_600_000);
  db.prepare("INSERT INTO sessions (token, user_id, created_at) VALUES (?, ?, ?)").run("s-old", 2, NOW - 2 * 86_400_000);
  const base = makeArchive([
    '{"t":1783239600,"e":"late@example.net","s":"landing"}',
    '{"t":1783236000,"e":"early@example.net"}',
    '{"t":17517',                                        // torn last write
  ]);
  const r = buildSiteReport(db, { archiveBase: base, analytics: ANALYTICS, processStartMs: NOW - 7_200_000, now: NOW });

  assert.deepEqual(r.accounts.map((a) => a.email), ["owner@example.com", "ada@example.com", "grace@example.org"]);
  assert.deepEqual(r.accounts.map((a) => a.isOwner), [true, false, false]);
  const ada = r.accounts[1], grace = r.accounts[2];
  assert.equal(ada.activeApiKeys, 1, "revoked keys don't count as active");
  assert.equal(ada.apiKeyLastUsedAt, NOW - 3_600_000);
  assert.equal(ada.signedInLast24h, false, "a 2-day-old session is not 'last 24h'");
  assert.equal(grace.signedInLast24h, true);
  assert.equal(grace.tier, "pro");
  assert.equal(grace.subscriptionStatus, "active");

  assert.deepEqual(r.waitlist, [
    { at: 1783236000_000, email: "early@example.net", source: "developers" },
    { at: 1783239600_000, email: "late@example.net", source: "landing" },
  ], "sorted oldest-first, torn line skipped, missing source defaults to the /developers form");

  assert.equal(r.visits.since, new Date(NOW - 7_200_000).toISOString());
  assert.equal(r.visits.totalPageViews, 42);
  assert.deepEqual(r.visits.topPages, ANALYTICS.topRoutes);
  assert.equal(r.visits.source, "memory", "no ledger running -> the in-memory counters");
  assert.deepEqual(r.notes, siteReportNotes("memory"));
  fs.rmSync(base, { recursive: true, force: true });
});

test("buildSiteReport: no api_keys table and no waitlist dir read as none, not an error", () => {
  const db = makeDb(false);
  const base = makeArchive(null);
  const r = buildSiteReport(db, { archiveBase: base, analytics: ANALYTICS, processStartMs: NOW, now: NOW });
  assert.equal(r.accounts.length, 3);
  assert.ok(r.accounts.every((a) => a.activeApiKeys === 0 && a.apiKeyLastUsedAt === null));
  assert.deepEqual(r.waitlist, []);
  fs.rmSync(base, { recursive: true, force: true });
});

test("aggregateSiteReport: counts exclude the seeded owner; by day/tier/source", () => {
  const db = makeDb();
  db.prepare("INSERT INTO sessions (token, user_id, created_at) VALUES (?, ?, ?)").run("s1", 3, NOW - 60_000);
  const base = makeArchive(['{"t":1783236000,"e":"a@example.net","s":"developers"}', '{"t":1783239600,"e":"b@example.net","s":"landing"}']);
  const agg = aggregateSiteReport(buildSiteReport(db, { archiveBase: base, analytics: ANALYTICS, processStartMs: NOW, now: NOW }));
  assert.equal(agg.accounts.signups, 2, "owner row is not a signup");
  assert.deepEqual(agg.accounts.byDay, { "2026-08-02": 1, "2026-09-14": 1 });
  assert.deepEqual(agg.accounts.byTier, { free: 1, pro: 1 });
  assert.equal(agg.accounts.signedInLast24h, 1);
  assert.equal(agg.accounts.latestSignupDay, "2026-09-14");
  assert.equal(agg.waitlist.total, 2);
  assert.deepEqual(agg.waitlist.bySource, { developers: 1, landing: 1 });
  assert.equal(agg.waitlist.latestDay, "2026-07-05");
  fs.rmSync(base, { recursive: true, force: true });
});

test("aggregateSiteReport: user-supplied waitlist sources like 'constructor' / '__proto__' count as plain keys", () => {
  const db = makeDb();
  const base = makeArchive([
    '{"t":1783236000,"e":"a@example.net","s":"constructor"}',
    '{"t":1783236001,"e":"b@example.net","s":"__proto__"}',
    '{"t":1783236002,"e":"c@example.net","s":"constructor"}',
  ]);
  const agg = aggregateSiteReport(buildSiteReport(db, { archiveBase: base, analytics: ANALYTICS, processStartMs: NOW, now: NOW }));
  const sources = JSON.parse(JSON.stringify(agg.waitlist.bySource));
  assert.equal(sources.constructor, 2, "not Object.prototype.constructor + 1");
  assert.equal(sources["__proto__"], 1, "not swallowed by the __proto__ setter");
  assert.equal(Object.getPrototypeOf({}).polluted, undefined);
  fs.rmSync(base, { recursive: true, force: true });
});

test("aggregateSiteReport: carries NO email, id or per-person field (the diag token's 'no user data' posture)", () => {
  const db = makeDb();
  const base = makeArchive(['{"t":1783236000,"e":"user@münchen.de","s":"developers"}']);
  const full = buildSiteReport(db, { archiveBase: base, analytics: ANALYTICS, processStartMs: NOW, now: NOW });
  const json = JSON.stringify(aggregateSiteReport(full));
  for (const p of [...full.accounts.map((a) => a.email), ...full.waitlist.map((w) => w.email)]) {
    assert.ok(!json.includes(p), `aggregate leaked ${p}`);
  }
  assert.ok(!json.includes("@"), "no email-shaped string at all");
  assert.ok(!/"(email|id|user_id|joinedAt|apiKeyLastUsedAt)"/.test(json), "no per-person keys");
  fs.rmSync(base, { recursive: true, force: true });
});

test("aggregateSiteReport: per-day maps keep the NEWEST days when history outgrows the cap", () => {
  const db = makeDb();
  const add = db.prepare("INSERT INTO users (email, password_hash, created_at) VALUES (?, 'x', ?)");
  const start = Date.parse("2026-01-01T00:00:00Z");
  for (let i = 0; i < AGGREGATE_DAY_CAP + 30; i++) add.run(`u${i}@example.com`, new Date(start + i * 86_400_000).toISOString());
  const base = makeArchive(null);
  const agg = aggregateSiteReport(buildSiteReport(db, { archiveBase: base, analytics: ANALYTICS, processStartMs: NOW, now: NOW }));
  const days = Object.keys(agg.accounts.byDay);
  assert.equal(days.length, AGGREGATE_DAY_CAP);
  assert.ok(days.includes(agg.accounts.latestSignupDay!), "the newest day survives the cap");
  assert.equal(agg.accounts.signups, AGGREGATE_DAY_CAP + 30 + 2, "totals stay all-time");
  fs.rmSync(base, { recursive: true, force: true });
});

test("wiring: emails only behind the owner login; the diag probe returns the aggregate", () => {
  const routes = fs.readFileSync(path.join(here, "routes.ts"), "utf8");
  assert.ok(routes.includes('app.get("/api/admin/site-report", requireOwner,'), "owner report must sit behind auth.ts requireOwner");
  const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
  const start = bot.indexOf('case "signups"');
  const end = Math.min(...[bot.indexOf("\n        case ", start + 10), bot.indexOf("default:", start)].filter((i) => i > 0));
  assert.ok(start > 0 && end > start, "signups probe block not found");
  const block = bot.slice(start, end);
  assert.ok(block.includes("aggregateSiteReport(") && block.includes("sanitizeDiag("), "probe must return the sanitized aggregate");
  assert.ok(!/res\.json\(\s*(sanitizeDiag\(\s*)?(liveSiteReport|buildSiteReport)\(/.test(block), "probe must never return the full report");
});

test("visits come from the persisted ledger when it runs: window, today, browser subset, top pages over the newest days", () => {
  const db = makeDb();
  const base = makeArchive(null);
  const days: VisitHistorySnapshot["days"] = {};
  const start = Date.parse("2026-09-01T12:00:00Z");
  for (let i = 0; i < VISIT_REPORT_DAYS + 13; i++) {        // 2026-09-01 .. 2026-10-13
    const d = new Date(start + i * 86_400_000).toISOString().slice(0, 10);
    days[d] = { views: 10, uniques: 4, browserViews: 6, browserUniques: 3, routes: { "/": 7, [`/only-${d}`]: 3 } };
  }
  days["2026-10-13"] = { views: 50, uniques: 20, browserViews: 30, browserUniques: 12, routes: { "/": 40, "/developers": 10 }, uniquesCapped: true };
  const history: VisitHistorySnapshot = { recordingSince: "2026-09-01T11:59:00.000Z", days };
  const r = buildSiteReport(db, { archiveBase: base, analytics: ANALYTICS, processStartMs: NOW, now: NOW, history });
  const v = r.visits;
  assert.equal(v.source, "history");
  assert.equal(v.since, history.recordingSince, "the window is the ledger's, not this process's");
  assert.equal(v.uniquesToday, 20);
  assert.equal(v.browserUniquesToday, 12);
  assert.equal(v.totalPageViews, 10 * (VISIT_REPORT_DAYS + 12) + 50, "all recorded days");
  assert.equal(Object.keys(v.uniquesByDay).length, VISIT_REPORT_DAYS);
  assert.equal(v.days!.length, VISIT_REPORT_DAYS);
  assert.equal(v.days![v.days!.length - 1].day, "2026-10-13");
  assert.equal(v.days![v.days!.length - 1].capped, true);
  assert.equal(v.topPages[0].route, "/");
  assert.equal(v.topPages[0].count, 7 * (TOP_PAGES_DAYS - 1) + 40, "summed over the newest TOP_PAGES_DAYS days only");
  assert.ok(!v.topPages.some((p) => p.route === "/only-2026-09-01"), "older days don't reach top pages");
  assert.deepEqual(r.notes, siteReportNotes("history"));
  const agg = aggregateSiteReport(r);
  assert.equal(agg.visits.source, "history");
  assert.ok(!JSON.stringify(agg).includes("@"));
  fs.rmSync(base, { recursive: true, force: true });
});
