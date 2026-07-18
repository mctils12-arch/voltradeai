// Security headers — W6 hardening sprint (2026-07-18).
//
// Hand-rolled (NO helmet dependency): the header set below is small,
// explicit, and self-documenting, and avoids pulling a transitive dep tree
// into a frozen-adjacent boot path. Mounted in server/index.ts BEFORE any
// route registration so every response — API, static, catch-all — carries
// the headers.
//
// The CSP is the delicate part: this is a MapLibre GL app that pulls raster
// and vector tiles, fonts, worker bundles, and topojson from a wide set of
// third-party hosts. Shipping an ENFORCING CSP blind would blank the map.
// So we ship Content-Security-Policy-Report-Only by DEFAULT — browsers
// evaluate the policy and report violations but never block — and flip to
// the enforcing Content-Security-Policy header only when SECURITY_CSP_ENFORCE
// is explicitly set truthy. This lets us watch report traffic, tighten the
// source list, and enforce with confidence later.

import type { Request, Response, NextFunction } from "express";

// ── Enumerated external hosts the client legitimately reaches ───────────────
// Derived by grepping client/src for external hosts (2026-07-18). Grouped by
// the CSP directive they belong to. img-src and connect-src additionally get
// a blanket `https:` because map imagery/data is fetched from dozens of
// government + tile hosts (NOAA sub-domains, ESRI ArcGIS tile servers, NASA
// GIBS/FIRMS, GEBCO/EMODnet WMS, S3/GCS tile buckets, CelesTrak, adsb.lol,
// SEC/FINRA/CFTC data endpoints…) that change over time — enumerating every
// one would be brittle and would silently break the map when a new gov feed
// is added. Report-only-first plus a broad https: for passive fetches is the
// documented, deliberate trade-off.
export const CSP_SOURCES = {
  // Script CDNs actually referenced by the client bundle.
  script: [
    "'self'",
    "'unsafe-inline'", // Vite injects inline module bootstrap; MapLibre inline
    "'unsafe-eval'", // MapLibre GL / pmtiles use eval/wasm-compile paths
    "blob:", // bundled Web Workers (GP/SATCAT propagation, satWorker) load as blob:
    "https://cdnjs.cloudflare.com", // d3 (landing.tsx)
    "https://cdn.jsdelivr.net", // world-atlas topojson (DataWorldMap.tsx)
  ],
  worker: ["'self'", "blob:"], // Vite worker bundling emits blob: workers
  style: [
    "'self'",
    "'unsafe-inline'", // shadcn/inline styles + MapLibre injected styles
    "https://fonts.googleapis.com", // Geist font stylesheet (index.html)
  ],
  font: [
    "'self'",
    "data:", // MapLibre glyph/data URIs
    "https://fonts.gstatic.com", // Geist font files
  ],
  // Passive fetches: tiles, imagery, topojson, blobs. Broad https: is
  // intentional (see header comment) — img also allows data:/blob: for
  // SDF icon sprites and canvas-generated marker images.
  img: ["'self'", "data:", "blob:", "https:"],
  // Active fetches: tile JSON, GP/SATCAT feeds, gov data APIs, SSE, sockets.
  connect: ["'self'", "https:", "wss:"],
  // beehiiv newsletter embed is the only third-party frame target.
  frame: ["'self'", "https://voltradeai.beehiiv.com"],
} as const;

/** Build the CSP policy string from CSP_SOURCES. */
export function buildCsp(): string {
  const s = CSP_SOURCES;
  return [
    `default-src 'self'`,
    `script-src ${s.script.join(" ")}`,
    `worker-src ${s.worker.join(" ")}`,
    `style-src ${s.style.join(" ")}`,
    `font-src ${s.font.join(" ")}`,
    `img-src ${s.img.join(" ")}`,
    `connect-src ${s.connect.join(" ")}`,
    `frame-src ${s.frame.join(" ")}`,
    `object-src 'none'`,
    `base-uri 'self'`,
    `form-action 'self'`,
    `frame-ancestors 'none'`,
  ].join("; ");
}

export function cspEnforceEnabled(env: NodeJS.ProcessEnv = process.env): boolean {
  const v = (env.SECURITY_CSP_ENFORCE || "").toLowerCase();
  return v === "1" || v === "true" || v === "yes" || v === "on";
}

/** True when the request arrived over HTTPS (behind Railway's proxy this is
 *  carried by x-forwarded-proto; `trust proxy` makes req.secure honor it). */
function isHttps(req: Request): boolean {
  if (req.secure) return true;
  const xfp = req.headers["x-forwarded-proto"];
  const proto = Array.isArray(xfp) ? xfp[0] : xfp;
  return typeof proto === "string" && proto.split(",")[0].trim() === "https";
}

/**
 * Express middleware setting the security header set. Pure (env-injectable)
 * for tests. Mount BEFORE routes in index.ts.
 */
export function securityHeaders(env: NodeJS.ProcessEnv = process.env) {
  const enforce = cspEnforceEnabled(env);
  const csp = buildCsp();
  return function securityHeadersMw(req: Request, res: Response, next: NextFunction) {
    // CSP: report-only by default, enforcing only when explicitly enabled.
    res.setHeader(
      enforce ? "Content-Security-Policy" : "Content-Security-Policy-Report-Only",
      csp,
    );
    res.setHeader("X-Content-Type-Options", "nosniff");
    res.setHeader("X-Frame-Options", "DENY");
    res.setHeader("Referrer-Policy", "strict-origin-when-cross-origin");
    res.setHeader("X-DNS-Prefetch-Control", "off");
    res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
    // HSTS only over HTTPS — never assert it on plaintext (would be ignored
    // by spec, but avoid emitting a misleading header on local http).
    if (isHttps(req)) {
      res.setHeader(
        "Strict-Transport-Security",
        "max-age=31536000; includeSubDomains",
      );
    }
    next();
  };
}
