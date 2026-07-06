/**
 * diag.ts — token-gated READ-ONLY diagnostics (wishlist option (d),
 * human-approved 2026-07-04). Lets autonomous sessions verify KNOWN
 * BROKEN items (fills firing? feedback accumulating? retrain green?)
 * without the owner cookie. auth.ts is deliberately untouched — this
 * routes around the owner gate ONLY with the human's explicit approval
 * and only for a HARD-WHITELISTED, sanitized read surface.
 *
 * Security posture: absent/short DIAG_TOKEN ⇒ the route 404s (closed by
 * default); token compare is timing-safe; every response passes
 * sanitizeDiag as defense-in-depth (whitelist shaping is the primary
 * control). Worst leak case: paper-account strategy telemetry — no
 * order placement, no Alpaca keys, no user data, no billing. Rotation =
 * change the env var.
 *
 * Pure module (node:test safe).
 */
import crypto from "crypto";

export const DIAG_PROBES = ["audit", "ml", "daemon", "positions", "scanner"] as const;
export type DiagProbe = (typeof DIAG_PROBES)[number];

const MIN_TOKEN_LEN = 24;

export function diagEnabled(env: Record<string, string | undefined>): boolean {
  return (env.DIAG_TOKEN || "").length >= MIN_TOKEN_LEN;
}

export function checkDiagToken(
  env: Record<string, string | undefined>, provided: string,
): boolean {
  const expected = env.DIAG_TOKEN || "";
  if (expected.length < MIN_TOKEN_LEN || !provided) return false;
  const a = crypto.createHash("sha256").update(provided).digest();
  const b = crypto.createHash("sha256").update(expected).digest();
  return crypto.timingSafeEqual(a, b);
}

/** Positions collapse to counts + exposure — never symbols, never
 *  per-position P&L (the approved whitelist says SUMMARY). */
export function positionsSummary(positions: any[]): {
  count: number; stocks: number; options: number;
  grossExposure: number; netExposure: number;
} {
  let stocks = 0, options = 0, gross = 0, net = 0;
  for (const p of Array.isArray(positions) ? positions : []) {
    const isOption = String(p?.symbol || "").length > 8 || p?.asset_class === "us_option";
    if (isOption) options++; else stocks++;
    const mv = parseFloat(p?.market_value) || 0;
    gross += Math.abs(mv);
    net += mv;
  }
  return {
    count: stocks + options, stocks, options,
    grossExposure: Math.round(gross), netExposure: Math.round(net),
  };
}

const SECRETISH = [
  /(?:key|token|secret|password|bearer|sk|pk)[-_ :="']{0,3}[A-Za-z0-9+/_-]{16,}/gi,
  /\b[a-f0-9]{32,}\b/gi,                       // bare long hex
  /\b[A-Za-z0-9+/]{40,}={0,2}\b/g,             // bare long base64
  /[\w.+-]+@[\w-]+\.[\w.]+/g,                  // emails
];

/** Defense-in-depth scrub over already-whitelisted payloads. */
export function sanitizeDiag<T>(v: T, depth = 0): T {
  if (depth > 8) return "[depth]" as any;
  if (typeof v === "string") {
    let s: string = v;
    for (const re of SECRETISH) s = s.replace(re, "[redacted]");
    return s.slice(0, 2000) as any;
  }
  if (Array.isArray(v)) return v.slice(0, 200).map((x) => sanitizeDiag(x, depth + 1)) as any;
  if (v && typeof v === "object") {
    const out: any = {};
    for (const [k, val] of Object.entries(v as any).slice(0, 100)) {
      out[k] = sanitizeDiag(val, depth + 1);
    }
    return out;
  }
  return v;
}
