// Admin stats endpoints — W6 hardening sprint (2026-07-18).
//
// Read-only operational counters for the security + analytics subsystems,
// gated by a shared-secret admin token (env ADMIN_STATS_TOKEN). Mounted in
// index.ts BEFORE routes. Behaviour:
//   • ADMIN_STATS_TOKEN unset  → endpoints return 404 (invisible, no oracle)
//   • token set, wrong/missing → 401
//   • token set, correct       → 200 with the snapshot
// Token may be supplied as header `x-admin-token: <t>` or query `?token=<t>`.
//
// Endpoints:
//   GET /api/_admin/security-stats  — anti-scraping + auth-event summary
//   GET /api/_admin/analytics       — page views / uniques / top routes

import crypto from "crypto";
import type { Request, Response, Express } from "express";
import { antiScrapingSnapshot } from "./antiScraping";
import { authEventSummary, recentAuthEvents } from "./rateLimit";
import { analyticsSnapshot, recentRequests } from "./requestLog";

function presentedToken(req: Request): string {
  const h = req.headers["x-admin-token"];
  if (typeof h === "string" && h) return h;
  const q = (req.query?.token as string) || "";
  return q;
}

/** Constant-time compare so the token can't be timing-guessed. */
function tokenOk(presented: string, expected: string): boolean {
  const a = Buffer.from(presented);
  const b = Buffer.from(expected);
  if (a.length !== b.length) return false;
  return crypto.timingSafeEqual(a, b);
}

/**
 * Guard: returns true if the request is authorized. Otherwise it has already
 * sent the appropriate 404/401 response.
 */
export function adminGuard(req: Request, res: Response, env: NodeJS.ProcessEnv = process.env): boolean {
  const expected = env.ADMIN_STATS_TOKEN || "";
  if (!expected) {
    // Feature disabled — behave as if the route does not exist.
    res.status(404).json({ error: "Not found" });
    return false;
  }
  const presented = presentedToken(req);
  if (!presented || !tokenOk(presented, expected)) {
    res.status(401).json({ error: "Unauthorized" });
    return false;
  }
  return true;
}

export function registerAdminStats(app: Express) {
  app.get("/api/_admin/security-stats", (req: Request, res: Response) => {
    if (!adminGuard(req, res)) return;
    res.json({
      antiScraping: antiScrapingSnapshot(),
      authEvents: authEventSummary(),
      recentAuthEvents: recentAuthEvents(50),
    });
  });

  app.get("/api/_admin/analytics", (req: Request, res: Response) => {
    if (!adminGuard(req, res)) return;
    res.json({
      analytics: analyticsSnapshot(),
      recentRequests: recentRequests(100),
    });
  });
}
