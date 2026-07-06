/**
 * scannerHealth.ts — Tier2 scan-failure visibility (REPAIR 2026-07-06):
 * a bot whose /api/health reads "active" can still be functionally blind
 * to new trade candidates if Tier2's market-data scan has failed every
 * cycle for hours. The LIVENESS ALARM (liveness.ts) only catches the loop
 * being paused/halted/killed — it does not catch "running, but its scan
 * keeps erroring on the same root cause and finds nothing new."
 *
 * FOUND LIVE 2026-07-06: 10+ consecutive TIER2-BACKOFF failures over
 * ~1.5h, every one "Could not fetch market data from Alpaca", while
 * /api/health read clean the entire time (see research/experiments.md).
 * The threshold below (6 consecutive failures) is the point where the
 * scheduler's exponential back-off has already maxed out at its 600s
 * cap (server/bot.ts scheduleTier2) — by then a blip has had several
 * chances to clear on its own, so persistence past it is a real signal,
 * not noise.
 *
 * Pure module (node:test safe).
 */

export const SCANNER_DEGRADED_FAILURE_THRESHOLD = 6;

export function scannerDegraded(consecutiveFailures: number): boolean {
  return consecutiveFailures >= SCANNER_DEGRADED_FAILURE_THRESHOLD;
}
