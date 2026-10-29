// Weekly forecast screen (2026-10-27, SHADOW) — the shape of GET /api/forecast
// (forecast.py report()) and the page's pure helpers: formatting, freshness,
// and the calibration chart's geometry. Kept out of the page so they are tested.

export interface CalBin { lo: number; hi: number; n: number; mean_p: number | null; freq: number | null }
export interface ForecastScore {
  id: string; status: "resolved" | "no_price_data"; resolved_date?: string; realized_close?: number;
  realized_ret_pct?: number; outcome_up?: number; hit?: number; in_range?: number; brier?: number;
}
export interface ForecastCall {
  id: string; week_key: string; symbol: string; group: "index" | "position" | "candidate";
  made_at: string; base_date: string; horizon_end: string; spot: number; p_up: number;
  direction: "up" | "down"; p_call: number; range_lo: number; range_hi: number; band_coverage: number;
  sigma_h: number; vol_source: "chain_iv30" | "realized_20d" | string; vol_annual: number | null;
  model_version: string; score: ForecastScore | null;
}
export interface Accuracy {
  n: number; thin: boolean; hits: number; hit_rate: number | null; brier: number | null; brier_coinflip: number;
  brier_skill_vs_coinflip: number | null; in_range: number; in_range_rate: number | null; in_range_target: number; up_rate: number | null;
}
export interface Validation {
  split_date: string; n_train: number; n_test: number; train_base_rate: number; test_base_rate: number;
  brier: number; brier_climatology: number; brier_skill: number | null; log_loss: number; hit_rate: number; calibration: CalBin[];
}
export interface LatestBatch {
  week_key: string; made_at: string; base_date: string; horizon_end: string; on_time: boolean; model_version: string;
  model: { coef: number[]; features: string[]; n_samples: number; first_date: string; last_date: string; validation: Validation };
  train_tickers: string[]; n_forecasts: number; skipped: Array<{ symbol: string; reason: string }>;
  candidates_available: boolean; candidates_scan_at: string | null; iv_day: string | null; iv_names: number;
  forecasts: ForecastCall[];
}
export interface WeekRow {
  week_key: string; made_at: string; base_date: string; horizon_end: string; on_time: boolean; model_version: string;
  n: number; resolved: number; hits: number; hit_rate: number | null; brier: number | null; in_range_rate: number | null;
}
export interface ForecastReport {
  generated_at: string; model_version: string; experimental: boolean; shadow: boolean; horizon_days: number; band_coverage: number;
  latest: LatestBatch | null;
  accuracy: Accuracy & { on_time: Accuracy; by_group: Record<string, Accuracy>; by_vol_source: Record<string, Accuracy> };
  calibration: { edges: number[]; bins: CalBin[] };
  weeks: WeekRow[]; open: number; method: string; caveats: string[];
}

export const GROUP_LABEL: Record<string, string> = { index: "Index", position: "Position", candidate: "Candidate" };
export const VOL_LABEL: Record<string, string> = { chain_iv30: "chain IV30", realized_20d: "20-day realized vol" };

export const pct = (x: number | null | undefined, digits = 0): string =>
  typeof x === "number" && Number.isFinite(x) ? `${(x * 100).toFixed(digits)}%` : "—";

export function usd(x: number | null | undefined): string {
  if (typeof x !== "number" || !Number.isFinite(x)) return "—";
  const d = Math.abs(x) >= 1000 ? 0 : 2;
  return `$${x.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d })}`;
}

/** Half-width of the band as a percent of spot, symmetric in log space (the upper side). */
export function bandPct(c: Pick<ForecastCall, "spot" | "range_lo" | "range_hi">): { down: number; up: number } {
  return { down: (c.range_lo / c.spot - 1) * 100, up: (c.range_hi / c.spot - 1) * 100 };
}

/** "3m", "5h", "2d" — the age of a timestamp at `nowMs`; "—" when unparseable. */
export function ageText(iso: string | null | undefined, nowMs: number): string {
  const t = iso ? Date.parse(iso) : NaN;
  if (!Number.isFinite(t)) return "—";
  const s = Math.max(0, Math.round((nowMs - t) / 1000));
  if (s < 90) return `${s}s`;
  if (s < 90 * 60) return `${Math.round(s / 60)}m`;
  if (s < 36 * 3600) return `${Math.round(s / 3600)}h`;
  return `${Math.round(s / 86400)}d`;
}

/** A call's state for its badge: the score once resolved, else open until its horizon. */
export function callState(c: ForecastCall): "hit" | "miss" | "open" | "no-data" {
  if (!c.score) return "open";
  if (c.score.status !== "resolved") return "no-data";
  return c.score.hit ? "hit" : "miss";
}

/** Short, honest reading of a Brier skill score (vs a reference forecast). */
export function skillText(skill: number | null | undefined): string {
  if (typeof skill !== "number" || !Number.isFinite(skill)) return "—";
  const v = `${skill >= 0 ? "+" : ""}${(skill * 100).toFixed(1)}%`;
  return skill > 0.005 ? `${v} better` : skill < -0.005 ? `${v} worse` : `${v} ≈ no skill`;
}

// ── calibration chart geometry ──────────────────────────────────────────────
export interface CalChart {
  w: number; h: number; pad: { l: number; r: number; t: number; b: number };
  xDomain: [number, number]; yDomain: [number, number];
  x: (p: number) => number; y: (f: number) => number;
  /** the y = x reference, clipped to the x domain */
  diagonal: { x1: number; y1: number; x2: number; y2: number };
  xTicks: number[]; yTicks: number[];
}

/** Predicted P(up) sits near the base rate, so the x axis zooms to the
 *  populated range (never narrower than [0.3, 0.7]); y is always 0–100%. */
export function calChart(w: number, h: number, series: CalBin[][]): CalChart {
  const pad = { l: 40, r: 14, t: 12, b: 34 };
  const ps = series.flat().filter((b) => b.n > 0 && typeof b.mean_p === "number").map((b) => b.mean_p as number);
  const lo = Math.max(0, Math.min(0.3, ...ps.map((p) => Math.floor((p - 0.02) * 20) / 20)));
  const hi = Math.min(1, Math.max(0.7, ...ps.map((p) => Math.ceil((p + 0.02) * 20) / 20)));
  const x = (p: number) => pad.l + ((p - lo) / (hi - lo)) * (w - pad.l - pad.r);
  const y = (f: number) => h - pad.b - f * (h - pad.t - pad.b);
  const xTicks: number[] = [];
  for (let t = Math.ceil(lo * 10) / 10; t <= hi + 1e-9; t += 0.1) xTicks.push(Math.round(t * 10) / 10);
  return { w, h, pad, xDomain: [lo, hi], yDomain: [0, 1], x, y, diagonal: { x1: x(lo), y1: y(lo), x2: x(hi), y2: y(hi) }, xTicks, yTicks: [0, 0.25, 0.5, 0.75, 1] };
}

/** Marker radius by count (area ∝ n), 4–10 px. */
export function dotRadius(n: number, maxN: number): number {
  if (n <= 0 || maxN <= 0) return 0;
  return 4 + 6 * Math.sqrt(n / maxN);
}

// ── cockpit KPI strip (2026-10-29, human-asked: the page fills the screen like
// the AI Engine) ──────────────────────────────────────────────────────────────
/** One tile of a cockpit KPI strip: eyebrow, hero value, one mono line. */
export interface Kpi { label: string; value: string; line: string; tone?: "g" | "r" | "o" | "t3" }

/** "Oct 30" from a YYYY-MM-DD; the raw date stays in the tile's line. */
export function shortDate(ymd: string | null | undefined): string {
  if (!ymd || !/^\d{4}-\d{2}-\d{2}$/.test(ymd)) return "—";
  return new Date(`${ymd}T12:00:00Z`).toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" });
}

/** The forecast strip: calls this week, direction hit rate, Brier skill vs a
 *  coin flip, in-range rate and when the open calls are scored — "—" wherever
 *  the payload has nothing yet. Tones only compare two numbers the page already
 *  shows (hit rate vs the base rate, skill vs zero) and stay off while thin. */
export function forecastKpis(rep: ForecastReport | null | undefined): Kpi[] {
  const latest = rep?.latest ?? null;
  const a = rep?.accuracy ?? null;
  const g = { index: 0, position: 0, candidate: 0 };
  for (const c of latest?.forecasts ?? []) if (c.group in g) g[c.group as keyof typeof g] += 1;
  const thin = a?.thin ? " · thin" : "";
  const resolved = !!a && a.n > 0;
  const hitTone = resolved && !a.thin && typeof a.hit_rate === "number" && typeof a.up_rate === "number"
    ? (a.hit_rate > a.up_rate ? "g" : a.hit_rate < a.up_rate ? "r" : undefined) : undefined;
  const skill = a?.brier_skill_vs_coinflip;
  const skillTone = resolved && !a.thin && typeof skill === "number" ? (skill > 0.005 ? "g" : skill < -0.005 ? "r" : undefined) : undefined;
  return [
    { label: "Calls this week", value: latest ? String(latest.n_forecasts) : "—", line: latest ? `${g.index} index · ${g.position} positions · ${g.candidate} candidates` : "no weekly batch yet" },
    { label: "Direction hit rate", value: resolved ? pct(a.hit_rate, 1) : "—", line: resolved ? `${a.hits} of ${a.n} resolved · base rate ${pct(a.up_rate, 1)}${thin}` : "no resolved calls yet", tone: hitTone },
    { label: "Brier skill vs coin flip", value: resolved && typeof skill === "number" ? `${skill >= 0 ? "+" : ""}${(skill * 100).toFixed(1)}%` : "—", line: resolved ? `Brier ${a.brier ?? "—"} · coin flip 0.25 · ${skillText(skill)}${thin}` : "lower Brier is better · coin flip 0.25", tone: skillTone },
    { label: "Inside the 68% range", value: resolved ? pct(a.in_range_rate, 1) : "—", line: resolved ? `target ${pct(a.in_range_target, 0)} · ${a.in_range} of ${a.n}${thin}` : "no resolved calls yet" },
    { label: "Next scoring", value: latest ? shortDate(latest.horizon_end) : "—", line: latest ? `after the ${latest.horizon_end} close · ${rep?.open ?? 0} open` : "scored after the horizon's last close" },
  ];
}
