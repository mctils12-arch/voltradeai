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

export const DIAG_PROBES = [
  "audit", "ml", "daemon", "positions", "scanner",
  // WHITELIST WIDENED 2026-07-07 by direct human instruction ("you need
  // to see the trades somehow so you can learn"): sessions get per-order
  // and per-position detail — symbols, quantities, prices. Same posture
  // otherwise: token-gated, READ-ONLY, sanitized; no keys, no user data,
  // no order placement. Paper-account book contents are the accepted
  // (and now explicitly authorized) exposure.
  "orders", "positions-detail",
  // ADDED 2026-07-18 (scheduled-routine session, REPAIR — KNOWN BROKEN #18
  // recurrence investigation): scan_market's daemon-thread path already
  // persists per-phase wall-clock timings to survive a 300s daemon-timeout
  // kill (TIMING-DISK 2026-04-23, bot_engine.py), but nothing exposed that
  // file outside the container. Read-only passthrough of the last scan's
  // phase breakdown — no trading/order data, same posture as the other
  // probes.
  "timings",
  // ADDED 2026-07-26 (scheduled-routine PRODUCT session): sessions doing
  // ladder-gate-2 statistical work (e.g. USAspending award/mcap ratio vs.
  // forward returns) had no way to read a datacore stream's historical
  // JSONL archive from outside the Railway volume — every other read
  // surface (/api/data/*) serves only an in-memory recent-cache window.
  // Read-only, token-gated, one-day-per-call passthrough of an existing
  // archive directory; see readArchiveDay in datacoreArchive.ts and the
  // "archive" case in bot.ts for the stream/day validation.
  "archive",
] as const;
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

const num = (v: any): number | null => {
  const n = parseFloat(v);
  return Number.isFinite(n) ? n : null;
};

/** One Alpaca order, shaped for the "orders" probe (2026-07-07
 *  authorization above). Whitelist shaping: only these fields, ever —
 *  client_order_id is deliberately dropped (can carry long opaque
 *  strings the sanitizer would mangle; the server-assigned UUID id is
 *  enough to reference an order). */
export function orderRow(o: any): Record<string, unknown> {
  return {
    id: String(o?.id || ""),
    symbol: String(o?.symbol || ""),
    asset_class: String(o?.asset_class || ""),
    side: String(o?.side || ""),
    type: String(o?.type || ""),
    qty: num(o?.qty),
    notional: num(o?.notional),
    filled_qty: num(o?.filled_qty),
    filled_avg_price: num(o?.filled_avg_price),
    limit_price: num(o?.limit_price),
    status: String(o?.status || ""),
    submitted_at: String(o?.submitted_at || ""),
    filled_at: o?.filled_at ? String(o.filled_at) : null,
    canceled_at: o?.canceled_at ? String(o.canceled_at) : null,
  };
}

/** One Alpaca position, shaped for the "positions-detail" probe
 *  (2026-07-07 authorization above). lastday_price vs current_price is
 *  the mark-move readout — it separates "the holding collapsed" from
 *  "the holding was sold/vanished" in incident forensics. */
export function positionRow(p: any): Record<string, unknown> {
  return {
    symbol: String(p?.symbol || ""),
    asset_class: String(p?.asset_class || ""),
    side: String(p?.side || ""),
    qty: num(p?.qty),
    avg_entry_price: num(p?.avg_entry_price),
    current_price: num(p?.current_price),
    lastday_price: num(p?.lastday_price),
    change_today: num(p?.change_today),
    market_value: num(p?.market_value),
    cost_basis: num(p?.cost_basis),
    unrealized_pl: num(p?.unrealized_pl),
    unrealized_plpc: num(p?.unrealized_plpc),
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
