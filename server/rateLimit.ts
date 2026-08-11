// Rate limiting + auth-event hardening — W6 hardening sprint (2026-07-18).
//
// Two limiters, both in-memory sliding-window, both mounted in
// server/index.ts BEFORE routes and WITHOUT touching auth.ts:
//
//   1. A GENEROUS global limiter keyed by (hashed IP, route class). The
//      numbers below are set so no interactive human — even one power-panning
//      the map, which fires hundreds of tile requests — ever trips them; they
//      exist only to cap runaway automation.
//
//   2. A STRICT limiter mounted specifically on the auth POST paths
//      (/api/auth/login, /api/auth/register) with incremental backoff/lockout
//      and a CONSTANT generic error message (never leaks whether an account
//      exists or whether the caller is locked out).
//
// Auth events (success / failure / lockout) are audit-logged. auth.ts is
// frozen, so we observe the auth POST's RESPONSE STATUS on finish (200 =
// success, 401/400 = failure, 429 = lockout) rather than reading into
// auth.ts. IPs are hashed before logging (privacy). If a persistent audit
// writer is injected via setAuthAuditWriter it is used; otherwise events land
// in a capped in-memory append structure surfaced at the stats endpoint.

import type { Request, Response, NextFunction } from "express";
import { hashIp } from "./requestLog";

// ── Route classes & generous global limits ─────────────────────────────────
// Window is 60s. Limits are per (ipHash, class). Rationale for generosity:
// a single map interaction session can burst ~hundreds of /tiles and /api
// requests; we set caps ~5-10x above any realistic human peak.
export interface RouteClassLimit {
  windowMs: number;
  max: number;
}
export const GLOBAL_LIMITS: Record<string, RouteClassLimit> = {
  // Map tiles / imagery proxy: panning + zooming fires many in parallel.
  tiles: { windowMs: 60_000, max: 3000 },
  // API/data endpoints: dashboards poll; /data map hydrates many layers.
  api: { windowMs: 60_000, max: 1200 },
  // AUTH ISOLATION (field lockout 2026-08-11: the owner's login 429'd —
  // "Too many requests" ON THE SIGN-IN PAGE): auth actions previously
  // shared the api bucket, so the same household IP's busy map tabs
  // (phone + two PCs poll /api hard) starved the login POST. Auth gets
  // its OWN small lane: 30/min is far beyond any human's clicking and
  // far below useful brute force — and createStrictAuthLimiter (wired in
  // index.ts) still owns real attempt-counting and lockouts on top.
  auth: { windowMs: 60_000, max: 30 },
  // Everything else (pages, static assets).
  default: { windowMs: 60_000, max: 2000 },
};

export function routeClass(path: string): keyof typeof GLOBAL_LIMITS {
  if (path.startsWith("/api/auth/")) return "auth";
  if (path.startsWith("/tiles")) return "tiles";
  if (path.startsWith("/api")) return "api";
  return "default";
}

// ── Sliding-window store ────────────────────────────────────────────────────
// key -> ascending array of hit timestamps within the active window.
type WindowStore = Map<string, number[]>;

function hit(store: WindowStore, key: string, now: number, windowMs: number): number {
  let arr = store.get(key);
  if (!arr) {
    arr = [];
    store.set(key, arr);
  }
  const cutoff = now - windowMs;
  // Drop expired timestamps (arr is ascending).
  let i = 0;
  while (i < arr.length && arr[i] < cutoff) i++;
  if (i > 0) arr.splice(0, i);
  arr.push(now);
  return arr.length;
}

// Periodically sweep empty/expired buckets so the maps cannot grow unbounded.
function sweep(store: WindowStore, now: number, windowMs: number) {
  const cutoff = now - windowMs;
  store.forEach((arr, k) => {
    while (arr.length && arr[0] < cutoff) arr.shift();
    if (arr.length === 0) store.delete(k);
  });
}

const globalStore: WindowStore = new Map();
let lastSweep = 0;
// Hard cap (parent-review fix 2026-07-18): under the pre-existing
// `trust proxy: true` (Bug 36), req.ip is client-influenced via
// X-Forwarded-For — an attacker rotating spoofed XFF values creates one
// bucket per fake identity between sweeps. Cap + oldest-first eviction
// keeps that a nuisance, not a memory attack. (Identity rotation also
// evades per-IP limits themselves — documented in antiScraping.ts; the
// robust quota layer is per API key on /api/v1.)
const GLOBAL_STORE_CAP = 100_000;

function capStore(store: Map<string, unknown>, cap: number) {
  if (store.size <= cap) return;
  const drop = store.size - Math.floor(cap * 0.9);
  let i = 0;
  for (const k of store.keys()) {
    if (i++ >= drop) break;
    store.delete(k);
  }
}

/**
 * Generous global limiter. Mount with app.use(globalRateLimit()).
 */
export function globalRateLimit(
  nowFn: () => number = Date.now,
  limits: Record<string, RouteClassLimit> = GLOBAL_LIMITS,
) {
  return function globalRateLimitMw(req: Request, res: Response, next: NextFunction) {
    const now = nowFn();
    if (now - lastSweep > 120_000) {
      lastSweep = now;
      sweep(globalStore, now, 60_000);
    }
    capStore(globalStore, GLOBAL_STORE_CAP);
    const cls = routeClass(req.path);
    const limit = limits[cls] || limits.default;
    const key = cls + "|" + hashIp(req.ip || req.socket?.remoteAddress || undefined);
    const count = hit(globalStore, key, now, limit.windowMs);
    if (count > limit.max) {
      res.setHeader("Retry-After", Math.ceil(limit.windowMs / 1000).toString());
      return res.status(429).json({ error: "Too many requests. Please slow down." });
    }
    next();
  };
}

// ── Auth event audit log ────────────────────────────────────────────────────
export type AuthEventKind = "success" | "failure" | "lockout" | "ratelimited";
export interface AuthEvent {
  t: number;
  kind: AuthEventKind;
  path: string;
  ipHash: string;
}
const AUTH_EVENT_CAP = 2000;
const authEvents: AuthEvent[] = [];

type AuditWriter = (type: string, message: string) => void;
let authAuditWriter: AuditWriter | null = null;
/** Optional: wire the persistent audit_log writer (same pattern as
 *  providerCompliance.setComplianceAuditWriter). Injected from index.ts/
 *  routes.ts if desired; unset by default → in-memory only. */
export function setAuthAuditWriter(w: AuditWriter | null) {
  authAuditWriter = w;
}

function recordAuthEvent(kind: AuthEventKind, path: string, ipHash: string, now: number) {
  authEvents.push({ t: now, kind, path, ipHash });
  if (authEvents.length > AUTH_EVENT_CAP) authEvents.splice(0, authEvents.length - AUTH_EVENT_CAP);
  try {
    authAuditWriter?.("AUTH-" + kind.toUpperCase(), `${path} ip=${ipHash}`);
  } catch {}
}

export function recentAuthEvents(limit = 200): AuthEvent[] {
  return authEvents.slice(-Math.max(1, Math.min(limit, authEvents.length)));
}

export interface AuthEventSummary {
  total: number;
  success: number;
  failure: number;
  lockout: number;
  ratelimited: number;
}
export function authEventSummary(): AuthEventSummary {
  const s: AuthEventSummary = { total: authEvents.length, success: 0, failure: 0, lockout: 0, ratelimited: 0 };
  for (const e of authEvents) s[e.kind]++;
  return s;
}

// ── Strict auth limiter with incremental backoff / lockout ──────────────────
export interface StrictAuthConfig {
  windowMs: number; // sliding window for counting attempts
  maxAttempts: number; // attempts within window before lockout
  baseLockMs: number; // first lockout duration
  maxLockMs: number; // cap on backed-off lockout
}
export const STRICT_AUTH_DEFAULTS: StrictAuthConfig = {
  windowMs: 15 * 60_000, // 15 minutes
  maxAttempts: 10, // 10 auth POSTs per IP per 15 min before lockout
  baseLockMs: 60_000, // 1 minute, doubling each subsequent lockout…
  maxLockMs: 60 * 60_000, // …capped at 1 hour
};

interface LockState {
  lockedUntil: number;
  violations: number; // consecutive lockouts → backoff exponent
}

const CONSTANT_AUTH_ERROR = "Too many attempts. Please try again later.";

export interface StrictAuthLimiter {
  middleware: (req: Request, res: Response, next: NextFunction) => void;
  _reset: () => void;
}

/**
 * Build the strict auth limiter. Mount its `.middleware` on each auth POST
 * path in index.ts, e.g.
 *   const strict = createStrictAuthLimiter();
 *   app.use("/api/auth/login", strict.middleware);
 *   app.use("/api/auth/register", strict.middleware);
 *
 * On every auth request it (a) enforces the sliding-window + backoff lockout
 * with a CONSTANT generic 429 message, and (b) records the auth outcome from
 * the response status on finish.
 */
export function createStrictAuthLimiter(
  cfg: StrictAuthConfig = STRICT_AUTH_DEFAULTS,
  nowFn: () => number = Date.now,
): StrictAuthLimiter {
  const attempts: WindowStore = new Map();
  const locks = new Map<string, LockState>();
  // Expired-state sweep (parent-review fix 2026-07-18): neither map evicted,
  // so every IP that ever touched an auth path stayed in memory forever.
  // Attempts drop once their window is empty; lock records are kept for 24h
  // past expiry so repeat offenders still escalate the backoff exponent.
  const LOCK_MEMORY_MS = 24 * 60 * 60_000;
  let lastAuthSweep = 0;

  function middleware(req: Request, res: Response, next: NextFunction) {
    const now = nowFn();
    if (now - lastAuthSweep > 600_000) {
      lastAuthSweep = now;
      sweep(attempts, now, cfg.windowMs);
      locks.forEach((l, k) => {
        if (l.lockedUntil + LOCK_MEMORY_MS < now) locks.delete(k);
      });
      capStore(attempts, 20_000);
      capStore(locks, 20_000);
    }
    const ipHash = hashIp(req.ip || req.socket?.remoteAddress || undefined);

    // 1) Currently locked out?
    const lock = locks.get(ipHash);
    if (lock && lock.lockedUntil > now) {
      const retryAfter = Math.ceil((lock.lockedUntil - now) / 1000);
      res.setHeader("Retry-After", retryAfter.toString());
      recordAuthEvent("lockout", req.path, ipHash, now);
      return res.status(429).json({ error: CONSTANT_AUTH_ERROR });
    }

    // 2) Count this attempt in the sliding window.
    const count = hit(attempts, ipHash, now, cfg.windowMs);
    if (count > cfg.maxAttempts) {
      // Trip a lockout with incremental (exponential) backoff.
      const violations = (lock?.violations || 0) + 1;
      const dur = Math.min(cfg.baseLockMs * Math.pow(2, violations - 1), cfg.maxLockMs);
      locks.set(ipHash, { lockedUntil: now + dur, violations });
      res.setHeader("Retry-After", Math.ceil(dur / 1000).toString());
      recordAuthEvent("ratelimited", req.path, ipHash, now);
      return res.status(429).json({ error: CONSTANT_AUTH_ERROR });
    }

    // 3) Allowed — observe the outcome to audit-log success/failure.
    res.on("finish", () => {
      const t = nowFn();
      if (res.statusCode >= 200 && res.statusCode < 300) {
        // Success → clear the attempt counter and any prior violations.
        attempts.delete(ipHash);
        locks.delete(ipHash);
        recordAuthEvent("success", req.path, ipHash, t);
      } else if (res.statusCode === 429) {
        // A downstream limiter (e.g. auth.ts's own) locked it out.
        recordAuthEvent("lockout", req.path, ipHash, t);
      } else if (res.statusCode >= 400) {
        recordAuthEvent("failure", req.path, ipHash, t);
      }
    });
    next();
  }

  return {
    middleware,
    _reset() {
      attempts.clear();
      locks.clear();
    },
  };
}

/** Test helper: wipe global limiter + auth-event state. */
export function _resetRateLimit() {
  globalStore.clear();
  authEvents.length = 0;
  authAuditWriter = null;
  lastSweep = 0;
}
