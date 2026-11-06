/**
 * earningsHistory.ts — the Tier-3 refresh of the earnings-surprise cache
 * (2026-11-06; human refocus spec §3 step 5 "earnings history (beats last 3
 * quarters)" and §7 "log every signal"; the logic is earnings_history.py).
 *
 * bot.ts Tier 3 runs EARNINGS_HISTORY_REFRESH_CMD hourly in the Python mutex
 * lane (execPythonSerialized, the flows step's pattern): it fetches Finnhub's
 * EPS actual vs estimate for the names deep_score asked about in the last 24 h,
 * paced under the free tier's limit, into the cache deep_score reads with no
 * network. LOGGING ONLY: nothing that scores, sizes or gates reads it.
 *
 * earningsRefreshAuditMessage turns the job's summary into at most one audit
 * row: EARNINGS-HISTORY when it fetched something, EARNINGS-HISTORY-ERROR when
 * every fetch failed, the run stopped on a Finnhub refusal, the job failed, or
 * (once, on the change) no FINNHUB_KEY is set. An idle or switched-off run
 * writes nothing. Pure (node:test safe).
 */

export const EARNINGS_HISTORY_REFRESH_TIMEOUT_MS = 60_000; // the job stops starting calls at 40 s (earnings_history.run_budget_s)
export const EARNINGS_HISTORY_REFRESH_CMD =
  `python3 -c "import json; from earnings_history import refresh_wanted; print(json.dumps(refresh_wanted(), default=str))"`;

export interface EarningsAuditRow { type: "EARNINGS-HISTORY" | "EARNINGS-HISTORY-ERROR"; message: string }

function isObj(x: unknown): x is Record<string, unknown> {
  return !!x && typeof x === "object" && !Array.isArray(x);
}
function count(x: unknown): number {
  return typeof x === "number" && Number.isFinite(x) && x > 0 ? Math.trunc(x) : 0;
}
function str(x: unknown, n = 200): string {
  return x === null || x === undefined ? "" : String(x).replace(/\s+/g, " ").trim().slice(0, n);
}
const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;

/** One audit row for one Tier-3 refresh (see the header), or null. */
export function earningsRefreshAuditMessage(result: unknown): EarningsAuditRow | null {
  const r = isObj(result) ? result : {};
  const status = str(r.status, 20);
  const firstError = str(r.first_error);
  if (status === "error") {
    return { type: "EARNINGS-HISTORY-ERROR", message: `earnings history refresh failed: ${firstError || "unknown error"}` };
  }
  if (status === "no_key") {
    if (r.status_changed !== true) return null;
    return { type: "EARNINGS-HISTORY-ERROR", message: `earnings history: FINNHUB_KEY is not set, so no earnings surprises are fetched; ${plural(count(r.wanted), "deep-scored name")} log the reason no_api_key` };
  }
  if (status !== "done") return null;
  const ok = count(r.ok), empty = count(r.empty), failed = count(r.failed), left = count(r.left);
  const stopped = str(r.stopped, 120);
  if (ok + empty + failed === 0 && !stopped) return null;   // nothing attempted (max_fetch_per_run 0)
  const parts = [`fetched ${plural(ok + empty + failed, "name")} from Finnhub`];
  const detail = [ok ? `${ok} with reported quarters` : "", empty ? `${empty} with none on record` : "",
    failed ? `${failed} failed${firstError ? ` (${firstError.slice(0, 140)})` : ""}` : ""].filter(Boolean);
  if (detail.length) parts[0] += ` (${detail.join(", ")})`;
  if (stopped) parts.push(`stopped early: ${stopped}${!failed && firstError ? ` (${firstError.slice(0, 140)})` : ""}`);
  if (left) parts.push(`${plural(left, "name")} still queued for the next hourly run`);
  const type = ok + empty === 0 ? "EARNINGS-HISTORY-ERROR" : "EARNINGS-HISTORY";
  return { type, message: `earnings history: ${parts.join("; ")}`.slice(0, 900) };
}
