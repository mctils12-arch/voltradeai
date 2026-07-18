// Anti-scraping — W6 hardening sprint (2026-07-18).
//
// A behavioural scorer + response ladder that raises the cost of bulk
// harvesting our /data and /api surface without inconveniencing humans.
// Mounted globally in server/index.ts BEFORE routes. In-memory, per hashed
// IP, self-expiring — no dependency on auth.ts or the DB.
//
// Signals scored per IP:
//   • velocity     — request rate in a short rolling window
//   • breadth      — number of DISTINCT endpoints hit in a short window
//                    (a human reads a page; a scraper sweeps the catalogue)
//   • headless UA  — HeadlessChrome/puppeteer/playwright/… signatures
//   • abnormal hdr — missing User-Agent or missing Accept (typical of naive
//                    scripts; real browsers always send both)
//   • honeypot     — any hit on a trap route → immediate maximum score
//
// Response ladder (per IP, escalating with accumulated score):
//   log  →  429 throttle  →  403 block (offender persisted with TTL)
// Offenders are held in an in-memory set with a TTL; while blocked every
// request short-circuits to 403.

import type { Request, Response, NextFunction, Express } from "express";
import { hashIp, classifyUa } from "./requestLog";

// ── Honeypot trap routes ────────────────────────────────────────────────────
// Disallowed in robots.txt and linked invisibly by the client (the client
// wiring is the parent's job). No legitimate human/browser ever requests
// these — a hit is a strong scraper tell.
export const HONEYPOT_PATHS = [
  "/api/_internal/export",
  "/api/_internal/backup",
  "/api/v1/_bulk",
];

// ── Tunables ────────────────────────────────────────────────────────────────
// Scoped to the /api data surface only (see `scoped()` below): tiles, fonts,
// and static assets are NEVER evaluated, so heavy map panning cannot trip
// this. Thresholds are set ABOVE a legitimate /data map hydration — which
// fetches dozens of distinct layer endpoints in one burst — so a real user is
// score 0; only a sustained catalogue sweep, a headless client, or a honeypot
// hit escalates.
export const AS_CONFIG = {
  windowMs: 10_000, // rolling window for velocity + breadth
  velocitySoft: 150, // /api hits/window above which velocity starts scoring
  velocityHard: 500, // /api hits/window that alone is very suspicious
  breadthSoft: 80, // distinct /api endpoints/window above which breadth scores
  breadthHard: 200, // (a full map load hits ~40-60 distinct endpoints once)
  throttleScore: 100, // ≥ this → 429 throttle
  blockScore: 200, // ≥ this → 403 block + persist offender
  honeypotScore: 1000, // honeypot hit → instant block
  scoreHalfLifeMs: 30_000, // suspicion decays (halves) every 30s of quiet
  blockTtlMs: 15 * 60_000, // offender block persists 15 min, then re-evaluated
  endpointCap: 400, // cap distinct endpoints tracked per IP
} as const;

/** Anti-scraping only evaluates the data surface. Tiles/assets/pages are
 *  exempt so a tile-heavy map never scores as a scraper. Honeypots live under
 *  /api so they are always in scope. */
export function scoped(path: string): boolean {
  return path.startsWith("/api");
}

export interface RequestFeatures {
  uaClass: ReturnType<typeof classifyUa>;
  missingUa: boolean;
  missingAccept: boolean;
  velocity: number; // hits in window (including this one)
  breadth: number; // distinct endpoints in window
  honeypot: boolean;
}

/**
 * Pure instantaneous score for a request's features. Stateless: velocity and
 * breadth are passed in already-computed. Exposed for unit testing the
 * scoring logic in isolation.
 */
export function scoreFeatures(f: RequestFeatures, cfg = AS_CONFIG): number {
  if (f.honeypot) return cfg.honeypotScore;
  let s = 0;
  // Velocity: ramp between soft and hard thresholds.
  if (f.velocity > cfg.velocitySoft) {
    s += Math.min(80, ((f.velocity - cfg.velocitySoft) / Math.max(1, cfg.velocityHard - cfg.velocitySoft)) * 80);
  }
  // Breadth: sweeping many distinct endpoints fast is the scraper signature.
  if (f.breadth > cfg.breadthSoft) {
    s += Math.min(80, ((f.breadth - cfg.breadthSoft) / Math.max(1, cfg.breadthHard - cfg.breadthSoft)) * 80);
  }
  if (f.uaClass === "headless") s += 60;
  if (f.uaClass === "bot") s += 25;
  if (f.uaClass === "cli") s += 20;
  if (f.missingUa) s += 40;
  if (f.missingAccept) s += 20;
  return s;
}

// ── Per-IP state ────────────────────────────────────────────────────────────
interface IpRec {
  hits: number[]; // timestamps within window (velocity)
  endpoints: Map<string, number>; // endpoint -> last-seen ts (breadth)
  score: number; // accumulated, decaying suspicion
  lastSeen: number;
  blockedUntil: number; // offender TTL
  honeypotHits: number;
  throttled: number; // count of 429s served
  blocked: number; // count of 403s served
}

const ipStore = new Map<string, IpRec>();

function getRec(ipHash: string, now: number): IpRec {
  let r = ipStore.get(ipHash);
  if (!r) {
    r = { hits: [], endpoints: new Map(), score: 0, lastSeen: now, blockedUntil: 0, honeypotHits: 0, throttled: 0, blocked: 0 };
    ipStore.set(ipHash, r);
  }
  return r;
}

function decayScore(r: IpRec, now: number) {
  const dt = now - r.lastSeen;
  if (dt > 0 && r.score > 0) {
    r.score *= Math.pow(0.5, dt / AS_CONFIG.scoreHalfLifeMs);
    if (r.score < 0.5) r.score = 0;
  }
}

function isHoneypot(path: string): boolean {
  return HONEYPOT_PATHS.some((h) => path === h || path.startsWith(h + "/"));
}

export type Verdict = "allow" | "throttle" | "block";

export interface EvalResult {
  verdict: Verdict;
  score: number;
  ipHash: string;
  features: RequestFeatures;
}

/**
 * Core stateful evaluation for one request. Updates the IP record and returns
 * the ladder verdict. Separated from Express so tests can drive it directly.
 */
export function evaluate(req: { path: string; headers: Record<string, any>; ip?: string }, now: number): EvalResult {
  const ipHash = hashIp(req.ip || undefined);
  const r = getRec(ipHash, now);
  decayScore(r, now);

  const path = req.path;
  const honeypot = isHoneypot(path);

  // Already an active offender → keep blocking (re-arm TTL on continued abuse).
  if (r.blockedUntil > now) {
    r.blocked++;
    r.lastSeen = now;
    if (honeypot) r.honeypotHits++;
    return {
      verdict: "block",
      score: r.score,
      ipHash,
      features: { uaClass: "other", missingUa: false, missingAccept: false, velocity: 0, breadth: 0, honeypot },
    };
  }

  // Update velocity window.
  const cutoff = now - AS_CONFIG.windowMs;
  r.hits.push(now);
  let i = 0;
  while (i < r.hits.length && r.hits[i] < cutoff) i++;
  if (i > 0) r.hits.splice(0, i);

  // Update breadth (distinct endpoints in window).
  r.endpoints.set(path, now);
  r.endpoints.forEach((ts, ep) => {
    if (ts < cutoff) r.endpoints.delete(ep);
  });
  if (r.endpoints.size > AS_CONFIG.endpointCap) {
    // Drop oldest to cap memory.
    const oldest = Array.from(r.endpoints.entries()).sort((a, b) => a[1] - b[1])[0];
    if (oldest) r.endpoints.delete(oldest[0]);
  }

  const ua = req.headers["user-agent"];
  const features: RequestFeatures = {
    uaClass: classifyUa(ua),
    missingUa: !ua || String(ua).trim() === "",
    missingAccept: !req.headers["accept"],
    velocity: r.hits.length,
    breadth: r.endpoints.size,
    honeypot,
  };

  if (honeypot) r.honeypotHits++;

  const instant = scoreFeatures(features);
  r.score += instant;
  r.lastSeen = now;

  let verdict: Verdict = "allow";
  if (r.score >= AS_CONFIG.blockScore || honeypot) {
    verdict = "block";
    r.blockedUntil = now + AS_CONFIG.blockTtlMs;
    r.blocked++;
  } else if (r.score >= AS_CONFIG.throttleScore) {
    verdict = "throttle";
    r.throttled++;
  }

  return { verdict, score: r.score, ipHash, features };
}

// ── Express middleware ──────────────────────────────────────────────────────
export function antiScraping(nowFn: () => number = Date.now) {
  return function antiScrapingMw(req: Request, res: Response, next: NextFunction) {
    if (!scoped(req.path)) return next();
    const now = nowFn();
    const { verdict } = evaluate(req as any, now);
    if (verdict === "block") {
      res.setHeader("Retry-After", Math.ceil(AS_CONFIG.blockTtlMs / 1000).toString());
      return res.status(403).json({ error: "Forbidden." });
    }
    if (verdict === "throttle") {
      res.setHeader("Retry-After", "10");
      return res.status(429).json({ error: "Too many requests. Please slow down." });
    }
    next();
  };
}

/**
 * Register decoy honeypot endpoints. The global antiScraping middleware
 * already traps these paths (it runs first), but registering real handlers
 * guarantees the traps exist and respond even if middleware order changes.
 */
export function registerHoneypots(app: Express, nowFn: () => number = Date.now) {
  for (const p of HONEYPOT_PATHS) {
    app.all(p, (req: Request, res: Response) => {
      // Mark the offender in case the global mw did not (defensive).
      evaluate(req as any, nowFn());
      res.status(403).json({ error: "Forbidden." });
    });
  }
}

// ── Stats / introspection ───────────────────────────────────────────────────
export interface AntiScrapingSnapshot {
  trackedIps: number;
  activeOffenders: number;
  totalHoneypotHits: number;
  totalThrottled: number;
  totalBlocked: number;
  honeypotPaths: string[];
  offenders: Array<{ ipHash: string; score: number; blockedForMs: number; honeypotHits: number }>;
}

export function antiScrapingSnapshot(now = Date.now()): AntiScrapingSnapshot {
  let activeOffenders = 0;
  let totalHoneypotHits = 0;
  let totalThrottled = 0;
  let totalBlocked = 0;
  const offenders: AntiScrapingSnapshot["offenders"] = [];
  ipStore.forEach((r, ipHash) => {
    totalHoneypotHits += r.honeypotHits;
    totalThrottled += r.throttled;
    totalBlocked += r.blocked;
    if (r.blockedUntil > now) {
      activeOffenders++;
      offenders.push({ ipHash, score: Math.round(r.score), blockedForMs: r.blockedUntil - now, honeypotHits: r.honeypotHits });
    }
  });
  offenders.sort((a, b) => b.blockedForMs - a.blockedForMs);
  return {
    trackedIps: ipStore.size,
    activeOffenders,
    totalHoneypotHits,
    totalThrottled,
    totalBlocked,
    honeypotPaths: HONEYPOT_PATHS,
    offenders: offenders.slice(0, 50),
  };
}

export function _resetAntiScraping() {
  ipStore.clear();
}
