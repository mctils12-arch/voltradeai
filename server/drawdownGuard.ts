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
