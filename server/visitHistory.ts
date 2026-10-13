/**
 * visitHistory.ts — daily visit counts that survive restarts
 * (human-directed 2026-10-13, follow-up to siteReport.ts: "get back to me
 * on who's visit[ed]").
 *
 * requestLog.ts's page-view counters live in memory and reset on every
 * deploy — several a day — so the site had no visit history at all. This
 * keeps ONE small record per UTC day on the volume (site_visits.json):
 * page loads, unique visitors, the same two for browser-class user agents
 * (requestLog.classifyUa === "browser": declared bots, headless browsers
 * and scripts excluded — the closest available proxy for people), and a
 * capped per-page tally. requestLog's own counters are untouched; this is
 * an additional ledger fed by the same page-view decision.
 *
 * PRIVACY: unique visitors are counted with a DAILY salt — sha256(salt|ip)
 * truncated to 16 hex — which is persisted only next to that same day's
 * hashes, so a mid-day restart doesn't count everyone twice. At UTC
 * rollover the salt and hashes are discarded and only counts remain: no
 * visitor can be followed from one day to the next, and past days hold no
 * identifier at all. Raw IPs are never stored.
 *
 * DURABILITY: writes are atomic (tmp + rename) and batched — flushed every
 * FLUSH_MS while dirty and on SIGTERM — so a crash loses at most one
 * interval of counts, never the file. An unreadable file is set aside
 * (renamed, logged), never silently overwritten.
 *
 * Inert until startVisitHistory() runs (server/index.ts at boot), so tests
 * and scripts that mount requestLog never touch the disk.
 */
import crypto from "crypto";
import fs from "fs";
import path from "path";

/** Days of counts retained (each is a few hundred bytes). */
export const VISIT_HISTORY_DAYS = 400;
/** Per-day page tally cap; further pages count under "(other)" (the SPA
 *  answers any path, so a random-path flood must not grow the file). */
export const DAY_ROUTE_CAP = 100;
/** Per-day unique cap; past it the day reads "at least N" (uniquesCapped). */
export const DAY_UNIQUES_CAP = 50_000;
const FLUSH_MS = 60_000;

export interface DayVisits {
  views: number;
  uniques: number;
  browserViews: number;
  browserUniques: number;
  routes: Record<string, number>;
  uniquesCapped?: boolean;
}

interface TodayState { day: string; salt: string; hashes: string[]; browserHashes: string[] }
interface FileState { version: 1; startedAt: string; days: Record<string, DayVisits>; today: TodayState | null }

export interface VisitHistorySnapshot {
  /** when this ledger began recording (ISO) — nothing exists before it. */
  recordingSince: string;
  days: Record<string, DayVisits>;
}

const dayOf = (t: number) => new Date(t).toISOString().slice(0, 10);
const own = (o: object, k: string) => Object.prototype.hasOwnProperty.call(o, k);
/** Request paths are attacker-chosen; these would hit Object.prototype. */
const PROTO_KEYS = new Set(["__proto__", "constructor", "prototype"]);

function freshState(now: number): FileState {
  return { version: 1, startedAt: new Date(now).toISOString(), days: {}, today: null };
}

function load(filePath: string, now: number): FileState {
  if (!fs.existsSync(filePath)) return freshState(now);
  try {
    const s = JSON.parse(fs.readFileSync(filePath, "utf8"));
    if (s?.version === 1 && s.days && typeof s.days === "object" && typeof s.startedAt === "string") {
      return { version: 1, startedAt: s.startedAt, days: s.days, today: s.today ?? null };
    }
    throw new Error("unrecognised shape");
  } catch (e) {
    const aside = `${filePath}.unreadable-${now}`;
    console.error(`[visitHistory] ${filePath} unreadable (${e instanceof Error ? e.message : String(e)}); moved to ${aside}, starting a new ledger`);
    try { fs.renameSync(filePath, aside); } catch (re) { console.error("[visitHistory] could not move it aside:", re instanceof Error ? re.message : String(re)); }
    return freshState(now);
  }
}

export function createVisitHistory(filePath: string, nowFn: () => number = Date.now) {
  const state = load(filePath, nowFn());
  let hashes = new Set<string>();
  let browserHashes = new Set<string>();
  let dirty = false;
  if (state.today && state.today.day === dayOf(nowFn())) {
    hashes = new Set(state.today.hashes);
    browserHashes = new Set(state.today.browserHashes);
  } else if (state.today) {
    state.today = null; // a past day's salt/hashes: drop them at the next flush
    dirty = true;
  }

  function rollTo(day: string) {
    if (state.today?.day === day) return;
    state.today = { day, salt: crypto.randomBytes(16).toString("hex"), hashes: [], browserHashes: [] };
    hashes = new Set();
    browserHashes = new Set();
    const keep = Object.keys(state.days).sort().slice(-VISIT_HISTORY_DAYS);
    for (const d of Object.keys(state.days)) if (!keep.includes(d)) delete state.days[d];
  }

  /** One page load. `ip` is used only to derive today's salted hash. */
  function record(ip: string | null | undefined, uaClass: string, route: string, t: number = nowFn()) {
    const day = dayOf(t);
    rollTo(day);
    const d = (state.days[day] ??= { views: 0, uniques: 0, browserViews: 0, browserUniques: 0, routes: {} });
    const browser = uaClass === "browser";
    d.views++;
    if (browser) d.browserViews++;
    const r = PROTO_KEYS.has(route) ? "(other)" : route;
    const key = own(d.routes, r) || Object.keys(d.routes).length < DAY_ROUTE_CAP ? r : "(other)";
    d.routes[key] = (own(d.routes, key) ? d.routes[key] : 0) + 1;
    const h = crypto.createHash("sha256").update(`${state.today!.salt}|${ip || "anon"}`).digest("hex").slice(0, 16);
    if (!hashes.has(h)) {
      if (hashes.size < DAY_UNIQUES_CAP) { hashes.add(h); d.uniques++; } else d.uniquesCapped = true;
    }
    if (browser && !browserHashes.has(h)) {
      if (browserHashes.size < DAY_UNIQUES_CAP) { browserHashes.add(h); d.browserUniques++; } else d.uniquesCapped = true;
    }
    dirty = true;
  }

  /** Atomic write when dirty. Failures are logged and retried next flush. */
  function flush(): boolean {
    if (!dirty) return false;
    if (state.today) state.today = { ...state.today, hashes: [...hashes], browserHashes: [...browserHashes] };
    const tmp = `${filePath}.tmp`;
    try {
      fs.mkdirSync(path.dirname(filePath), { recursive: true });
      fs.writeFileSync(tmp, JSON.stringify(state));
      fs.renameSync(tmp, filePath);
      dirty = false;
      return true;
    } catch (e) {
      console.error("[visitHistory] flush failed (will retry):", e instanceof Error ? e.message : String(e));
      return false;
    }
  }

  function snapshot(): VisitHistorySnapshot {
    return { recordingSince: state.startedAt, days: JSON.parse(JSON.stringify(state.days)) };
  }

  return { record, flush, snapshot };
}

export type VisitHistory = ReturnType<typeof createVisitHistory>;

export function defaultVisitHistoryPath(env: NodeJS.ProcessEnv = process.env): string {
  const dataDir = env.DATA_DIR || (fs.existsSync("/data") ? "/data/voltrade" : "/tmp");
  return path.join(dataDir, "site_visits.json");
}

let live: VisitHistory | null = null;

/** Boot hook (server/index.ts). Idempotent. */
export function startVisitHistory(filePath: string = defaultVisitHistoryPath()): VisitHistory {
  if (live) return live;
  live = createVisitHistory(filePath);
  const timer = setInterval(() => live?.flush(), FLUSH_MS);
  timer.unref?.();
  // bot.ts's SIGTERM handler exits after cancelling orders; this listener
  // runs synchronously before that exit, so a redeploy loses nothing.
  process.on("SIGTERM", () => live?.flush());
  return live;
}

/** requestLog.ts calls this for every counted page view; no-op until started. */
export function recordDailyVisit(ip: string | null | undefined, uaClass: string, route: string, t?: number): void {
  live?.record(ip, uaClass, route, t);
}

export function visitHistorySnapshot(): VisitHistorySnapshot | null {
  return live ? live.snapshot() : null;
}

/** test hook — point the module at a temp-file ledger (or none). */
export function _useVisitHistory(h: VisitHistory | null) { live = h; }
