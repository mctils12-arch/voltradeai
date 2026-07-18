import express, { type Request, Response, NextFunction } from "express";
import compression from "compression";
import { registerRoutes } from "./routes";
import { registerBillingRoutes } from "./billing";
import { registerNewsletterRoutes } from "./newsletter";
import { serveStatic } from "./static";
import { createServer } from "http";
// ── W6 security-hardening middleware (2026-07-18) ──────────────────────────
// All mounted BELOW, BEFORE routes register. None edit auth.ts/billing.ts.
import { securityHeaders } from "./securityHeaders";
import { requestLog } from "./requestLog";
import { globalRateLimit, createStrictAuthLimiter } from "./rateLimit";
import { antiScraping, registerHoneypots } from "./antiScraping";
import { registerRobots } from "./robots";
import { registerAdminStats } from "./adminStats";
import { registerTerms } from "./terms";

const app = express();
app.set("trust proxy", true); // Bug 36: Railway runs behind a reverse proxy

// ── SECURITY HARDENING — mounted FIRST, before routes (W6, 2026-07-18) ──────
// Order matters: (1) headers on every response, (2) structured request log
// captures all traffic, (3) generous global rate limit, (4) anti-scraping
// scorer on the /api surface + honeypot traps, (5) robots.txt, admin stats,
// and ToS content served ahead of the SPA catch-all. The strict auth limiter
// is mounted on the real login/register paths (from routes.ts/auth.ts) without
// touching auth.ts. See server/compression.test.ts precedent + the mount-order
// pin in server/securityMiddleware.test.ts.
app.use(securityHeaders());
app.use(requestLog());
app.use(globalRateLimit());
app.use(antiScraping());
const strictAuth = createStrictAuthLimiter();
app.use("/api/auth/login", strictAuth.middleware);
app.use("/api/auth/register", strictAuth.middleware);
registerRobots(app);
registerHoneypots(app);
registerAdminStats(app);
registerTerms(app);
// [REPAIR 2026-07-05, /data load perf] Response compression. Railway's edge
// does NOT gzip for us — an aircraft snapshot was ~0.8MB raw (~120KB
// gzipped) and the powerplants static JSON 800KB (~200KB), refetched by
// every fresh visitor. The default filter skips already-compressed types
// (the wxtile PNG proxy stays untouched). Response-side only, so the
// Stripe webhook's raw-body parsing below is unaffected.
app.use(compression());
const httpServer = createServer(app);

declare module "http" {
  interface IncomingMessage {
    rawBody: unknown;
  }
}

// CRITICAL: Stripe webhook is registered FIRST, before express.json(),
// because Stripe signature verification needs the raw request body.
// registerBillingRoutes registers the webhook with its own raw-body parser.
registerBillingRoutes(app);

app.use(
  express.json({ limit: "1mb" }), // Bug 27: body size limit
);

app.use(express.urlencoded({ extended: false }));

// Newsletter archive routes (markdown-backed)
registerNewsletterRoutes(app);

export function log(message: string, source = "express") {
  const formattedTime = new Date().toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
    hour12: true,
  });

  console.log(`${formattedTime} [${source}] ${message}`);
}

app.use((req, res, next) => {
  const start = Date.now();
  const path = req.path;
  let capturedJsonResponse: string | undefined = undefined;

  const originalResJson = res.json;
  res.json = function (bodyJson, ...args) {
    // OOM fix: truncate immediately instead of keeping full response object in closure
    capturedJsonResponse = JSON.stringify(bodyJson).slice(0, 500);
    return originalResJson.apply(res, [bodyJson, ...args]);
  };

  res.on("finish", () => {
    const duration = Date.now() - start;
    if (path.startsWith("/api")) {
      let logLine = `${req.method} ${path} ${res.statusCode} in ${duration}ms`;
      if (capturedJsonResponse) {
        logLine += ` :: ${capturedJsonResponse}${capturedJsonResponse.length >= 500 ? "..." : ""}`;
      }

      log(logLine);
    }
  });

  next();
});

(async () => {
  await registerRoutes(httpServer, app);

  app.use((err: any, _req: Request, res: Response, next: NextFunction) => {
    const status = err.status || err.statusCode || 500;
    const message = err.message || "Internal Server Error";

    console.error("Internal Server Error:", err);

    if (res.headersSent) {
      return next(err);
    }

    return res.status(status).json({ message });
  });

  // importantly only setup vite in development and after
  // setting up all the other routes so the catch-all route
  // doesn't interfere with the other routes
  if (process.env.NODE_ENV === "production") {
    serveStatic(app);
  } else {
    const { setupVite } = await import("./vite");
    await setupVite(httpServer, app);
  }

  // ALWAYS serve the app on the port specified in the environment variable PORT
  // Other ports are firewalled. Default to 5000 if not specified.
  // this serves both the API and the client.
  // It is the only port that is not firewalled.
  const port = parseInt(process.env.PORT || "5000", 10);
  httpServer.listen(
    {
      port,
      host: "0.0.0.0",
    },
    () => {
      log(`serving on port ${port}`);
    },
  );
})();
