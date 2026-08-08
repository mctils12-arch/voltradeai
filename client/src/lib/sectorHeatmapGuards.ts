/** Crash/honesty guard for SectorHeatmap (repair — 2026-08-06 full-code-
 *  review finding "SectorHeatmap fabricated -100%", adversarially verified,
 *  queued unfixed for several sessions). Before Alpaca prints a sector
 *  ETF's first trade of the session, `dailyBar` can be present but missing
 *  `c` (or the whole snapshot can omit it) while `prevDailyBar.c` still
 *  holds yesterday's real close. The old inline math (`bar.c || 0` then
 *  `(c - pc) / pc`) turned that into a computed -100% change — a real
 *  sector tile rendered as "crashed to zero" when the true state was
 *  "no reading yet". Pure module so node:test can pin the contract without
 *  a DOM (tradeChartGuards/deviceTier precedent). */

export interface DailyBarLike {
  c?: number;
}

/** Percent change from prevDailyBar.c to dailyBar.c, or null when dailyBar
 *  has no real current price yet — the caller must render "no data" for
 *  that sector, never a fabricated -100%. Missing/zero prevDailyBar.c
 *  falls back to the current price (0% — "no baseline available"), same
 *  as the pre-existing behavior for that case. */
export function sectorChangePct(
  dailyBar: DailyBarLike | undefined,
  prevDailyBar: DailyBarLike | undefined,
): number | null {
  const c = dailyBar?.c;
  if (typeof c !== "number" || !Number.isFinite(c) || c <= 0) return null;
  const rawPc = prevDailyBar?.c;
  const pc = typeof rawPc === "number" && Number.isFinite(rawPc) && rawPc > 0 ? rawPc : c;
  return Math.round(((c - pc) / pc) * 100 * 100) / 100;
}
