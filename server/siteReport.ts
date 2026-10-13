/**
 * siteReport.ts — who signed up, and how many people visit
 * (human-directed 2026-10-13: "get back to me on who's visit[ed] and
 * signed up").
 *
 * ONE builder, two tiers, so the owner view and the agent view can never
 * disagree on a number:
 *   - buildSiteReport(): the full owner report — account emails, waitlist
 *     emails, per-account API-key use. Served ONLY behind the owner login
 *     (GET /api/admin/site-report, auth.ts requireOwner).
 *   - aggregateSiteReport(): counts only — per-day signups, waitlist by
 *     source, key holders, visit counters. No email, no id, no hash. This
 *     is all the token-gated /api/diag/signups probe returns, so an
 *     autonomous session can report signup/visit trends without ever
 *     holding user data (diag.ts's "no user data" posture holds).
 *
 * Visits come from requestLog.ts's IP-hashed, in-memory counters. They
 * cover only the time since this process started (every deploy resets
 * them), count full page loads (not in-app navigation) and include
 * crawlers; visitors are anonymous by design. The report carries that as
 * `visits.since` + `notes` rather than passing the numbers off as
 * all-time traffic.
 *
 * Pure core: db, archive base, analytics snapshot and clock are injected
 * (never imports auth.ts — its module load hangs the test runner; see
 * apiKeyAccounts.ts). liveSiteReport() wires the production inputs.
 */
import fs from "fs";
import path from "path";
import type { Database as DatabaseType } from "better-sqlite3";
import { analyticsSnapshot, type AnalyticsSnapshot } from "./requestLog";
import { archiveBaseDir } from "./datacoreArchive";
import { visitHistorySnapshot, type VisitHistorySnapshot } from "./visitHistory";

export interface AccountEntry {
  email: string;
  /** users.created_at as stored (ISO 8601). */
  joinedAt: string;
  role: string;
  tier: string;
  subscriptionStatus: string | null;
  /** role === "owner" — auth.ts seeds this account at boot; it is not a signup. */
  isOwner: boolean;
  activeApiKeys: number;
  /** epoch ms of the newest use of any of this account's keys. */
  apiKeyLastUsedAt: number | null;
  /** holds a session created in the last 24h (auth.ts sessions expire at 24h). */
  signedInLast24h: boolean;
}

export interface WaitlistEntry {
  /** epoch ms */
  at: number;
  email: string;
  source: string;
}

export interface VisitDay {
  day: string;
  views: number;
  uniques: number;
  /** page loads / visitors whose user agent classifies as a browser —
   *  declared bots, headless browsers and scripts excluded. */
  browserViews: number;
  browserUniques: number;
  /** the day hit visitHistory's unique cap: "at least" this many. */
  capped: boolean;
}

export interface VisitSummary {
  /** "history": visitHistory.ts's persisted daily ledger (survives
   *  deploys). "memory": requestLog's counters since this process started
   *  (the fallback when the ledger isn't running). */
  source: "history" | "memory";
  /** ISO start of the window the numbers cover. */
  since: string;
  totalPageViews: number;
  uniquesToday: number;
  /** newest VISIT_REPORT_DAYS days */
  uniquesByDay: Record<string, number>;
  /** history: summed over the newest TOP_PAGES_DAYS days; memory: since restart. */
  topPages: Array<{ route: string; count: number }>;
  /** history only — newest VISIT_REPORT_DAYS days, oldest first. */
  days?: VisitDay[];
  browserUniquesToday?: number;
}

export interface SiteReport {
  generatedAt: string;
  accounts: AccountEntry[];
  waitlist: WaitlistEntry[];
  visits: VisitSummary;
  notes: string[];
}

export interface SiteReportAggregate {
  generatedAt: string;
  accounts: {
    /** every non-owner account (the owner row is seeded, not a signup). */
    signups: number;
    byDay: Record<string, number>;
    byTier: Record<string, number>;
    withActiveApiKeys: number;
    signedInLast24h: number;
    latestSignupDay: string | null;
  };
  waitlist: {
    total: number;
    byDay: Record<string, number>;
    bySource: Record<string, number>;
    latestDay: string | null;
  };
  visits: VisitSummary;
  notes: string[];
}

const DAY_MS = 86_400_000;

/** Tally maps have no prototype: waitlist `source` is user-supplied (POST
 *  /api/waitlist), so a key like "constructor" or "__proto__" must count as
 *  a plain key, not resolve to Object.prototype. */
const tally = (): Record<string, number> => Object.create(null);
function bump(m: Record<string, number>, k: string) { m[k] = (m[k] ?? 0) + 1; }

export const VISIT_REPORT_DAYS = 30;
export const TOP_PAGES_DAYS = 7;

export function siteReportNotes(source: VisitSummary["source"]): string[] {
  return [
    source === "history"
      ? "Visits come from a daily ledger that survives deploys, recording since `visits.since` (nothing earlier exists). Each day counts full page loads, not in-app navigation; the browser figures exclude declared bots, headless browsers and scripts."
      : "Visits are in-memory counters since `visits.since` (every deploy resets them); they count full page loads, not in-app navigation, and include crawlers.",
    "Visitors are anonymous by design: IPs are salted and hashed before they are counted, so no visit can be tied to a person.",
    "The owner account is created automatically at startup and is not counted as a signup.",
  ];
}

function visitsFromHistory(h: VisitHistorySnapshot, now: number): VisitSummary {
  const keys = Object.keys(h.days).sort();
  const recent = keys.slice(-VISIT_REPORT_DAYS);
  const today = h.days[new Date(now).toISOString().slice(0, 10)];
  const routes = tally();
  for (const d of keys.slice(-TOP_PAGES_DAYS)) {
    for (const [r, n] of Object.entries(h.days[d].routes)) routes[r] = (routes[r] ?? 0) + n;
  }
  return {
    source: "history",
    since: h.recordingSince,
    totalPageViews: keys.reduce((sum, d) => sum + h.days[d].views, 0),
    uniquesToday: today?.uniques ?? 0,
    browserUniquesToday: today?.browserUniques ?? 0,
    uniquesByDay: Object.fromEntries(recent.map((d) => [d, h.days[d].uniques])),
    topPages: Object.entries(routes).sort((a, b) => b[1] - a[1]).slice(0, 20).map(([route, count]) => ({ route, count })),
    days: recent.map((d) => ({
      day: d, views: h.days[d].views, uniques: h.days[d].uniques,
      browserViews: h.days[d].browserViews, browserUniques: h.days[d].browserUniques,
      capped: !!h.days[d].uniquesCapped,
    })),
  };
}

function readWaitlist(archiveBase: string): WaitlistEntry[] {
  const dir = path.join(archiveBase, "waitlist");
  if (!fs.existsSync(dir)) return []; // nobody has joined yet
  const out: WaitlistEntry[] = [];
  for (const f of fs.readdirSync(dir).filter((n) => n.endsWith(".jsonl")).sort()) {
    for (const line of fs.readFileSync(path.join(dir, f), "utf8").split("\n")) {
      if (!line.trim()) continue;
      let row: { t?: unknown; e?: unknown; s?: unknown };
      // A torn last line (crash mid-append) is skipped, never fatal.
      try { row = JSON.parse(line); } catch { continue; }
      if (typeof row?.e !== "string" || typeof row.t !== "number" || !Number.isFinite(row.t)) continue;
      out.push({ at: row.t * 1000, email: row.e, source: typeof row.s === "string" && row.s ? row.s : "developers" });
    }
  }
  return out.sort((a, b) => a.at - b.at);
}

function tableExists(db: DatabaseType, name: string): boolean {
  return !!db.prepare("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?").get(name);
}

/** Per-user API-key use. The api_keys table only exists once
 *  apiKeyAccounts.ts's store has been created, so absence reads as none. */
function apiKeyUse(db: DatabaseType): Map<number, { active: number; lastUsed: number | null }> {
  const out = new Map<number, { active: number; lastUsed: number | null }>();
  if (!tableExists(db, "api_keys")) return out;
  const rows = db.prepare(
    "SELECT user_id, SUM(CASE WHEN revoked_at IS NULL THEN 1 ELSE 0 END) AS active, MAX(last_used_at) AS last_used FROM api_keys GROUP BY user_id",
  ).all() as Array<{ user_id: number; active: number | null; last_used: number | null }>;
  for (const r of rows) out.set(r.user_id, { active: Number(r.active) || 0, lastUsed: r.last_used ?? null });
  return out;
}

function recentSessionUsers(db: DatabaseType, now: number): Set<number> {
  if (!tableExists(db, "sessions")) return new Set();
  const rows = db.prepare("SELECT DISTINCT user_id FROM sessions WHERE created_at > ?").all(now - DAY_MS) as Array<{ user_id: number }>;
  return new Set(rows.map((r) => r.user_id));
}

export function buildSiteReport(db: DatabaseType, opts: {
  archiveBase: string;
  analytics: AnalyticsSnapshot;
  processStartMs: number;
  now: number;
  /** visitHistory.ts's ledger when it is running; null -> in-memory counters. */
  history?: VisitHistorySnapshot | null;
}): SiteReport {
  const users = db.prepare(
    "SELECT id, email, role, tier, subscription_status, created_at FROM users ORDER BY created_at, id",
  ).all() as Array<{ id: number; email: string; role: string | null; tier: string | null; subscription_status: string | null; created_at: string }>;
  const keys = apiKeyUse(db);
  const recent = recentSessionUsers(db, opts.now);
  const visits: VisitSummary = opts.history
    ? visitsFromHistory(opts.history, opts.now)
    : {
        source: "memory",
        since: new Date(opts.processStartMs).toISOString(),
        totalPageViews: opts.analytics.totalPageViews,
        uniquesToday: opts.analytics.uniquesToday,
        uniquesByDay: opts.analytics.uniquesByDay,
        topPages: opts.analytics.topRoutes,
      };
  const accounts: AccountEntry[] = users.map((u) => ({
    email: u.email,
    joinedAt: u.created_at,
    role: u.role || "user",
    tier: u.tier || "free",
    subscriptionStatus: u.subscription_status || null,
    isOwner: u.role === "owner",
    activeApiKeys: keys.get(u.id)?.active ?? 0,
    apiKeyLastUsedAt: keys.get(u.id)?.lastUsed ?? null,
    signedInLast24h: recent.has(u.id),
  }));
  return {
    generatedAt: new Date(opts.now).toISOString(),
    accounts,
    waitlist: readWaitlist(opts.archiveBase),
    visits,
    notes: siteReportNotes(visits.source),
  };
}


/** Per-day maps keep the newest AGGREGATE_DAY_CAP days (totals stay
 *  all-time): sanitizeDiag keeps an object's FIRST 100 keys, which for an
 *  oldest-first day map would silently drop the newest days. */
export const AGGREGATE_DAY_CAP = 90;

function newestDays(m: Record<string, number>): Record<string, number> {
  const out: Record<string, number> = {};
  for (const d of Object.keys(m).sort().slice(-AGGREGATE_DAY_CAP)) out[d] = m[d];
  return out;
}

/** Counts only — the per-person fields never leave this function. */
export function aggregateSiteReport(r: SiteReport): SiteReportAggregate {
  const signups = r.accounts.filter((a) => !a.isOwner);
  const byDay = tally();
  const byTier = tally();
  for (const a of signups) {
    bump(byDay, String(a.joinedAt).slice(0, 10));
    bump(byTier, a.tier);
  }
  const wlByDay = tally();
  const wlBySource = tally();
  for (const w of r.waitlist) {
    bump(wlByDay, new Date(w.at).toISOString().slice(0, 10));
    bump(wlBySource, w.source);
  }
  const lastKey = (m: Record<string, number>) => Object.keys(m).sort().pop() ?? null;
  return {
    generatedAt: r.generatedAt,
    accounts: {
      signups: signups.length,
      byDay: newestDays(byDay),
      byTier: { ...byTier },
      withActiveApiKeys: signups.filter((a) => a.activeApiKeys > 0).length,
      signedInLast24h: signups.filter((a) => a.signedInLast24h).length,
      latestSignupDay: lastKey(byDay),
    },
    waitlist: { total: r.waitlist.length, byDay: newestDays(wlByDay), bySource: { ...wlBySource }, latestDay: lastKey(wlByDay) },
    visits: r.visits,
    notes: r.notes,
  };
}

/** Production wiring shared by the owner route and the diag probe. */
export function liveSiteReport(db: DatabaseType, now = Date.now()): SiteReport {
  return buildSiteReport(db, {
    archiveBase: archiveBaseDir(),
    analytics: analyticsSnapshot(now),
    processStartMs: now - Math.round(process.uptime() * 1000),
    now,
    history: visitHistorySnapshot(),
  });
}
