/**
 * drawdownGuard.ts — validated drawdown evaluation for the max-drawdown
 * kill switch ([REPAIR] 2026-07-07).
 *
 * LIVE INCIDENT: the kill switch fired within ~53 min of a market-open
 * container boot while account equity sat AT its all-time peak (health:
 * status "killed", drawdownPct 0.0, equityPeak $109,432). A real -10%
 * round-trip inside an hour on this paper account is implausible; the
 * trigger sites computed drawdown directly from a single Alpaca
 * /v2/account read with no validation — a transient zero/garbage equity
 * value computes as ~-100% and kills the loop. One site even fabricated
 * `parseFloat(acct.equity || "100000")`, which lets the string "0"
 * through while inventing equity when the field is absent.
 *
 * THE MECHANISM IS PRESERVED, deliberately: any CREDIBLE equity read at
 * or below maxDrawdownPct still kills exactly as before. This module
 * only refuses to kill (or move the peak) on reads that are impossible
 * for a funded account — non-finite or <= 0 — and surfaces them to the
 * caller for auditing. Ambiguous-but-possible low reads still kill
 * (safe direction; catastrophe must never be filtered as "bad data").
 */

export interface DrawdownEval {
  valid: boolean;                // credible equity read
  equity: number | null;        // parsed equity when valid
  newPeak: number;              // peak after this read (unchanged if invalid)
  drawdownPct: number | null;   // vs newPeak, when valid
  kill: boolean;                // fire the kill switch
}

export function evaluateDrawdown(
  equityRaw: unknown,
  peak: number,
  maxDrawdownPct: number,
): DrawdownEval {
  const equity =
    typeof equityRaw === "number" ? equityRaw : parseFloat(String(equityRaw ?? ""));
  if (!Number.isFinite(equity) || equity <= 0) {
    return { valid: false, equity: null, newPeak: peak, drawdownPct: null, kill: false };
  }
  const newPeak = peak <= 0 ? equity : Math.max(peak, equity);
  const drawdownPct = ((equity - newPeak) / newPeak) * 100;
  return { valid: true, equity, newPeak, drawdownPct, kill: drawdownPct <= maxDrawdownPct };
}

export interface DrawdownStatus {
  current_pct: number;
  kill_threshold_pct: number;
  proximity_pct: number;
  status: "OK" | "WARNING" | "CRITICAL";
}

/**
 * Monitoring-dashboard view of a validated drawdown reading ([REPAIR]
 * 2026-08-09 — /api/monitoring/overview previously derived this from a
 * `current_dd_pct` field `get_kill_switch_status()` never returns, so `dd`
 * silently defaulted to 0 every call and this always reported "OK" /100%
 * proximity regardless of the real number; the proximity formula was also
 * inverted (0% drawdown computed as 100% proximity to the kill threshold).
 * Now takes the SAME validated equity/peak the live Tier-1 kill switch
 * actually acts on (evaluateDrawdown above), so a real breach is visible
 * here too.
 */
export function drawdownStatus(drawdownPct: number, maxDrawdownPct: number): DrawdownStatus {
  const proximityRaw = maxDrawdownPct < 0 ? (drawdownPct / maxDrawdownPct) * 100 : 0;
  const proximity_pct = Math.max(0, Math.min(100, proximityRaw));
  const status: DrawdownStatus["status"] =
    drawdownPct <= maxDrawdownPct * 0.75 ? "CRITICAL" :
    drawdownPct <= maxDrawdownPct * 0.5 ? "WARNING" : "OK";
  return { current_pct: drawdownPct, kill_threshold_pct: maxDrawdownPct, proximity_pct, status };
}
