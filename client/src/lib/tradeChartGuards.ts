/** Crash-guards for the bot trade chart (repair 2026-08-06, full-code-review
 *  finding, adversarially verified): Alpaca returns null pnl/price fields for
 *  options legs without quotes and for halted names; the server's
 *  parseFloat → NaN serializes back to null, and unguarded .toFixed in
 *  TradeChart took down the WHOLE bot dashboard via the app-root
 *  ErrorBoundary. Pure module so node:test can pin the contract without a
 *  DOM (deviceTier/cameraRig precedent). */

/** Finite number or null — the single gate every numeric read passes. */
export function finiteOrNull(v: unknown): number | null {
  if (v == null || v === "") return null; // Number("") is 0 — an empty field is unknown, not zero
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

/** Display helper: em-dash for unknown — never a fabricated 0.00. */
export const fmt = (v: number | null, d = 2): string =>
  v == null ? "—" : v.toFixed(d);

/** The numeric fields a position card renders; everything else passes through. */
export const SANITIZED_FIELDS = [
  "pnl", "pnlPct", "rMultiple", "entryPrice",
  "stopPrice", "takeProfitPrice", "currentPrice", "highestPnl",
] as const;

export type SanitizedPosition<T> = Omit<T, (typeof SANITIZED_FIELDS)[number]> &
  Record<(typeof SANITIZED_FIELDS)[number], number | null>;

/** One sanitation pass over a raw API position — every numeric display read
 *  and every createPriceLine call sees this view, never the raw payload. */
export function sanitizePosition<T extends object>(raw: T): SanitizedPosition<T> {
  const src = raw as Record<string, unknown>;
  const out: Record<string, unknown> = { ...src };
  for (const k of SANITIZED_FIELDS) out[k] = finiteOrNull(src[k]);
  return out as SanitizedPosition<T>;
}
