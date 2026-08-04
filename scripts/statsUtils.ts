/**
 * statsUtils.ts — small reusable statistical helpers for datacore GATE 2
 * (SIGNAL) research scripts. Pure functions, no I/O, never imported by any
 * runtime path.
 *
 * Compiled out of the occ_options_volume 2026-08-03 gate-2 kill: a naive
 * pooled Welch t-test over (day,ticker) rows overstates significance when
 * the same tickers recur across sample days and every name in one day's
 * bucket shares that day's market-wide move (Reasoning Standard #4 — see
 * research/open_questions.md's OCC reversal-candidate entry). The fix here
 * is the "collapse to cluster means" approach: average within each cluster
 * first, then test across cluster-level means, so the CLUSTER (e.g. the
 * report day) is the unit of independence, not the raw row. Recommended
 * over analytic cluster-robust SE formulas specifically when the cluster
 * count is small (Cameron & Miller 2015, "A Practitioner's Guide to
 * Cluster-Robust Inference" — few-cluster analytic SEs are known to
 * under-reject).
 */

export interface ClusterTTestResult {
  n: number;
  mean: number;
  std: number;
  se: number;
  t: number;
  df: number;
}

// One-sample t-test on a set of per-cluster values (e.g. one CALL-PUT
// spread per report day). n = number of clusters, NOT number of raw rows.
export function clusterMeanTTest(clusterValues: number[]): ClusterTTestResult {
  const n = clusterValues.length;
  const mean = n > 0 ? clusterValues.reduce((a, b) => a + b, 0) / n : NaN;
  const variance = n > 1
    ? clusterValues.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1)
    : 0;
  const std = Math.sqrt(variance);
  const se = n > 0 ? std / Math.sqrt(n) : NaN;
  const t = se === 0 || !Number.isFinite(se) ? NaN : mean / se;
  return { n, mean, std, se, t, df: Math.max(0, n - 1) };
}

// Two-tailed alpha=0.05 critical value of the t-distribution, df 1-30
// (standard table). Beyond df=30 the normal approximation (1.96) is close
// enough for this use — gate-2 research samples here never exceed ~30
// clusters (weekly/day-level sampling bounds network cost).
const T_CRIT_005: Record<number, number> = {
  1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
  8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
  15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080,
  22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048,
  29: 2.045, 30: 2.042,
};

export function tCrit005(df: number): number {
  if (df <= 0) return NaN;
  if (df > 30) return 1.96;
  return T_CRIT_005[Math.round(df)] ?? 1.96;
}

// Whether a cluster-mean t-test result clears the correct df-adjusted
// critical value (NOT the flat |t|>2 heuristic used by the pooled tests —
// with few clusters the correct critical value is materially higher, e.g.
// 2.131 at df=15, so a naive |t|>2 bar can falsely "pass" a result that
// doesn't actually clear 5% significance at this sample size).
export function survivesAtCrit005(result: ClusterTTestResult): boolean {
  const crit = tCrit005(result.df);
  return Number.isFinite(result.t) && Math.abs(result.t) > crit;
}
