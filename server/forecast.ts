// forecast.ts — the Node half of the weekly forecast screen (2026-10-27,
// SHADOW; the model and the store are forecast.py). Pure helpers so the
// scheduling decision, the universe and the exact python3 -c command strings
// are testable without booting bot.ts; bot.ts does the I/O.
//
// Nothing here reaches order flow: the job reads Alpaca positions and the
// last scan's top_10, writes a temp payload, and runs forecast.py.

/** Python commands longer than this, or importing a heavy lib, take the
 *  mutex path with the 90 s hard kill; shorter ones are "trivial" and are
 *  SIGKILLed at 10 s (bot.ts isTrivialPython). Both forecast jobs fetch bars,
 *  so their commands must stay above it — server/forecast.test.ts pins that. */
export const TRIVIAL_PYTHON_MAX_CHARS = 150;
export const FORECAST_JOB_TIMEOUT_MS = 90_000;
/** Python-side wall budget: below the 90 s kill, leaving room for the fit + write. */
export const FORECAST_TIME_BUDGET_S = 60;
export const FORECAST_GENERATE_EVERY_MS = 60 * 60_000;
/** Score once per UTC day from this hour (after the US close in EDT and EST). */
export const FORECAST_SCORE_UTC_HOUR = 22;
export const FORECAST_MAX_CANDIDATES = 10;
export const FORECAST_MAX_TICKERS = 30;

export type ForecastGroup = "index" | "position" | "candidate";
export interface ForecastTicker { symbol: string; group: ForecastGroup }

/** The underlying of an OCC option symbol (AAPL261218C00200000 → AAPL), else null. */
export function occUnderlying(sym: string): string | null {
  const m = /^([A-Z]{1,6})\d{6}[CP]\d{8}$/.exec(String(sym || "").replace(/\s+/g, "").toUpperCase());
  return m ? m[1] : null;
}

const plainSymbol = (s: unknown): string | null => {
  const v = String(s ?? "").trim().toUpperCase();
  return /^[A-Z][A-Z.]{0,5}$/.test(v) ? v : null;
};

/** SPY, then every open position (an option counts as its underlying), then
 *  the last scan's top candidates — deduped with that priority, capped. */
export function forecastUniverse(
  positions: ReadonlyArray<{ symbol?: unknown }> | null | undefined,
  topCandidates: ReadonlyArray<{ ticker?: unknown }> | null | undefined,
  maxCandidates = FORECAST_MAX_CANDIDATES, cap = FORECAST_MAX_TICKERS,
): ForecastTicker[] {
  const out = new Map<string, ForecastGroup>([["SPY", "index"]]);
  for (const p of positions ?? []) {
    const raw = String(p?.symbol ?? "");
    const sym = occUnderlying(raw) ?? plainSymbol(raw);
    if (sym && !out.has(sym)) out.set(sym, "position");
  }
  let n = 0;
  for (const c of topCandidates ?? []) {
    if (n >= maxCandidates) break;
    const sym = plainSymbol(c?.ticker);
    if (!sym) continue;
    n += 1;
    if (!out.has(sym)) out.set(sym, "candidate");
  }
  return [...out.entries()].slice(0, cap).map(([symbol, group]) => ({ symbol, group }));
}

/** The latest IV30 per underlying from chainIv rows of ONE day (null iv30 dropped). */
export function ivMapFrom(rows: ReadonlyArray<{ u: string; iv30: number | null }>, symbols: ReadonlyArray<string>): Record<string, number> {
  const want = new Set(symbols);
  const out: Record<string, number> = {};
  for (const r of rows) if (want.has(r.u) && typeof r.iv30 === "number" && Number.isFinite(r.iv30) && r.iv30 > 0) out[r.u] = r.iv30;
  return out;
}

export interface ForecastPayload {
  tickers: ForecastTicker[];
  iv: Record<string, number>;
  iv_day: string | null;
  candidates_available: boolean;
  candidates_scan_at: string | null;
  time_budget_s: number;
}

/** A temp path that is safe on a shell command line: fixed dir, digits only. */
export function forecastPayloadPath(nowMs: number, dir = "/tmp"): string {
  return `${dir}/vt_forecast_${Math.floor(nowMs)}.json`;
}

export function forecastGenerateCmd(payloadPath: string): string {
  if (!/^\/tmp\/vt_forecast_\d+\.json$/.test(payloadPath)) throw new Error("forecast payload path must be /tmp/vt_forecast_<digits>.json");
  return `python3 -c "import json, os, sys; from forecast import generate_weekly; p = sys.argv[1]; d = json.load(open(p)); os.remove(p); print(json.dumps(generate_weekly(**d), default=str))" ${payloadPath}`;
}

export function forecastScoreCmd(timeBudgetS = FORECAST_TIME_BUDGET_S): string {
  // explicit keyword arguments: deliberately longer than the trivial cutoff (see above)
  return `python3 -c "import json; from forecast import score_matured; print(json.dumps(score_matured(time_budget_s=${Number(timeBudgetS)}, now=None, fetch=None, path=None, scores_path=None), default=str))"`;
}

export const FORECAST_REPORT_CMD = `python3 -c "import json; from forecast import report; print(json.dumps(report(), default=str))"`;

/** Generate: at most once per FORECAST_GENERATE_EVERY_MS (forecast.py skips
 *  cheaply, without any network call, when this week's batch exists). */
export function shouldAttemptGenerate(nowMs: number, lastAttemptMs: number | null): boolean {
  return lastAttemptMs === null || nowMs - lastAttemptMs >= FORECAST_GENERATE_EVERY_MS;
}

/** Score: once per UTC day, from FORECAST_SCORE_UTC_HOUR on. */
export function shouldScore(nowMs: number, lastScoredUtcDay: string | null): boolean {
  const d = new Date(nowMs);
  return d.getUTCHours() >= FORECAST_SCORE_UTC_HOUR && lastScoredUtcDay !== d.toISOString().slice(0, 10);
}
