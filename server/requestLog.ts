// Request logging + first-party analytics — W6 hardening sprint (2026-07-18).
//
// Structured, privacy-respecting, in-memory. NO third-party analytics, NO
// cookies, NO raw IPs anywhere. Raw client IPs are one-way hashed (salted
// SHA-256, truncated) before they touch the ring buffer or the page-view
// counters, so a dump of this module's state can never re-identify a
// visitor. Mounted in server/index.ts BEFORE routes.
//
// Two products:
//   1. A capped ring buffer of the last N request records (route, status,
//      latency, hashed IP, UA class) for operational visibility.
//   2. A minimal page-view counter: total views, per-day unique hashed IPs,
//      and a top-routes tally — surfaced behind an admin token.

import crypto from "crypto";
import type { Request, Response, NextFunction } from "express";

// Salt for IP hashing. A per-process random salt by default means hashes are
// NOT stable across restarts (extra privacy); set REQUEST_LOG_IP_SALT to a
// fixed value only if you need stable day-over-day unique counts.
const IP_SALT =
  process.env.REQUEST_LOG_IP_SALT || crypto.randomBytes(16).toString("hex");

/** One-way hash of a client IP. Never store or log the raw IP. */
export function hashIp(ip: string | undefined | null): string {
  if (!ip) return "anon";
  return crypto
    .createHash("sha256")
    .update(IP_SALT + "|" + ip)
    .digest("hex")
    .slice(0, 16);
}

export type UaClass =
  | "browser"
  | "bot"
  | "headless"
  | "cli"
  | "empty"
  | "other";

// Known headless / automation UA signatures (lowercased substrings).
const HEADLESS_UA = [
  "headlesschrome",
  "phantomjs",
  "electron",
  "puppeteer",
  "playwright",
  "selenium",
  "webdriver",
  "cypress",
];
const BOT_UA = [
  "bot",
  "crawler",
  "spider",
  "scrape",
  "slurp",
  "bingpreview",
  "facebookexternalhit",
  "semrush",
  "ahrefs",
  "mj12",
  "dotbot",
];
const CLI_UA = ["curl", "wget", "python-requests", "httpie", "go-http-client", "java/", "okhttp", "axios/", "node-fetch", "libwww-perl"];

/** Classify a User-Agent string into a coarse, privacy-safe bucket. */
export function classifyUa(ua: string | undefined | null): UaClass {
  if (!ua || ua.trim() === "") return "empty";
  const s = ua.toLowerCase();
  if (HEADLESS_UA.some((h) => s.includes(h))) return "headless";
  if (BOT_UA.some((b) => s.includes(b))) return "bot";
  if (CLI_UA.some((c) => s.includes(c))) return "cli";
  if (s.includes("mozilla/") || s.includes("safari") || s.includes("chrome") || s.includes("firefox"))
    return "browser";
  return "other";
}

export interface RequestRecord {
  t: number; // epoch ms
  method: string;
  route: string;
  status: number;
  latencyMs: number;
  ipHash: string;
  uaClass: UaClass;
}

// ── Capped ring buffer ──────────────────────────────────────────────────────
const RING_CAP = Math.max(
  100,
  parseInt(process.env.REQUEST_LOG_CAP || "5000", 10) || 5000,
);
const ring: RequestRecord[] = [];

function pushRecord(rec: RequestRecord) {
  ring.push(rec);
  if (ring.length > RING_CAP) ring.splice(0, ring.length - RING_CAP);
}

export function recentRequests(limit = 200): RequestRecord[] {
  return ring.slice(-Math.max(1, Math.min(limit, ring.length)));
}

// ── Page-view analytics ─────────────────────────────────────────────────────
let totalPageViews = 0;
const routeCounts = new Map<string, number>();
// day (YYYY-MM-DD) -> set of hashed IPs seen that day.
const uniquesByDay = new Map<string, Set<string>>();
// cap retained days so the map cannot grow unbounded.
const DAY_CAP = 90;

function dayKey(t: number): string {
  return new Date(t).toISOString().slice(0, 10);
}

/** True for a request that should count as a "page view" (an HTML nav, not an
 *  asset or API call). Heuristic: GET, not /api, not a file with an
 *  extension, not a tile. */
export function isPageView(method: string, route: string): boolean {
  if (method !== "GET") return false;
  if (route.startsWith("/api")) return false;
  if (route.startsWith("/tiles")) return false;
  if (route.startsWith("/assets")) return false;
  if (route.startsWith("/robots.txt")) return false;
  // has a file extension (e.g. .js, .png, .css, .ico) → asset, not a page.
  const last = route.split("/").pop() || "";
  if (last.includes(".")) return false;
  return true;
}

function recordPageView(route: string, ipHash: string, t: number) {
  totalPageViews++;
  routeCounts.set(route, (routeCounts.get(route) || 0) + 1);
  const dk = dayKey(t);
  let set = uniquesByDay.get(dk);
  if (!set) {
    set = new Set();
    uniquesByDay.set(dk, set);
    // Evict oldest days beyond the cap.
    if (uniquesByDay.size > DAY_CAP) {
      const oldest = Array.from(uniquesByDay.keys()).sort()[0];
      uniquesByDay.delete(oldest);
    }
  }
  set.add(ipHash);
}

export interface AnalyticsSnapshot {
  totalPageViews: number;
  uniquesToday: number;
  uniquesByDay: Record<string, number>;
  topRoutes: Array<{ route: string; count: number }>;
  ringSize: number;
  ringCap: number;
}

export function analyticsSnapshot(now = Date.now()): AnalyticsSnapshot {
  const uByDay: Record<string, number> = {};
  uniquesByDay.forEach((set, d) => { uByDay[d] = set.size; });
  const topRoutes = Array.from(routeCounts.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, 20)
    .map(([route, count]) => ({ route, count }));
  return {
    totalPageViews,
    uniquesToday: uniquesByDay.get(dayKey(now))?.size || 0,
    uniquesByDay: uByDay,
    topRoutes,
    ringSize: ring.length,
    ringCap: RING_CAP,
  };
}

/** Test/ops helper: wipe all in-memory state. */
export function _resetRequestLog() {
  ring.length = 0;
  totalPageViews = 0;
  routeCounts.clear();
  uniquesByDay.clear();
}

/**
 * Express middleware. Captures a structured record on response finish.
 * Mount BEFORE routes in index.ts.
 */
export function requestLog() {
  return function requestLogMw(req: Request, res: Response, next: NextFunction) {
    const start = Date.now();
    const ipHash = hashIp(req.ip || req.socket?.remoteAddress || undefined);
    const uaClass = classifyUa(req.headers["user-agent"]);
    res.on("finish", () => {
      const rec: RequestRecord = {
        t: Date.now(),
        method: req.method,
        // Use the matched route path when available, else the raw path, to
        // avoid unbounded cardinality from query strings / path params.
        route: (req.route?.path && typeof req.route.path === "string"
          ? req.baseUrl + req.route.path
          : req.path) || req.path,
        status: res.statusCode,
        latencyMs: Date.now() - start,
        ipHash,
        uaClass,
      };
      pushRecord(rec);
      if (isPageView(rec.method, rec.route) && rec.status < 400) {
        recordPageView(rec.route, ipHash, rec.t);
      }
    });
    next();
  };
}
