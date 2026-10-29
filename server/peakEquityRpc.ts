// /api/monitoring/drawdown reads risk_kill_switch.get_peak_equity through
// pythonCall. The two paths answer in different shapes: the daemon returns
// the function's raw float, the subprocess fallback prints {"peak": float}.
// The route read `.peak` only, got undefined from the daemon, fell back to the
// current equity, and so reported a drawdown of ~0% whenever the daemon was up
// (audit §1.x D3, fixed 2026-10-27). Accept both shapes; anything else
// (including get_peak_equity's 0.0 "no file yet") is null, so the caller can
// say the peak is unknown instead of inventing one.
export function peakEquityFromRpc(result: unknown): number | null {
  let v: unknown = result;
  if (v && typeof v === "object" && "peak" in (v as Record<string, unknown>)) {
    v = (v as Record<string, unknown>).peak;
  }
  const n = typeof v === "number" ? v : typeof v === "string" ? Number(v) : NaN;
  return Number.isFinite(n) && n > 0 ? n : null;
}
