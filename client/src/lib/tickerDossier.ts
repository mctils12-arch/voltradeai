// Ticker dossier (2026-10-29, human-asked) — the page's pure half. The server's
// /api/ticker-dossier/:symbol (ticker_dossier.py) builds the layers; this
// module turns them into what the page draws: the range band geometry, the
// trade-plan ladder, the score-decile match against the bot's live score, the
// position-size helper and the honest one-liners. No DOM, no fetch; pinned by
// tickerDossier.test.ts. Every value here is a price or a percentage — no
// distance/speed/temperature, so the units preference does not apply.

export type LayerStatus = "live" | "partial" | "unavailable";
export interface LayerMeta { status: LayerStatus; source: string; as_of: string | null; reason: string | null }
export interface Bands { lo68: number; hi68: number; lo90: number; hi90: number; move68_pct: number; move90_pct: number }
export interface Backtest { h: number; n: number; in68: number; in90: number; rate68: number; rate90: number; target68: number; target90: number; overlapping: boolean }
export interface RangeH { h: number; model: string | null; fallback_reason: string | null; sigma_pct: number | null; bands: Bands | null; backtest: Backtest | null }
export interface Trend { label: string; sma20: number | null; sma50: number | null; sma200: number | null; vs_sma50_pct: number | null; vs_sma200_pct: number | null; ret_5d_pct: number | null; ret_20d_pct: number | null; ret_60d_pct: number | null; ret_250d_pct: number | null }
export interface StockLayer extends LayerMeta {
  price?: number; price_day?: string; price_is_partial_session?: boolean; base_close?: number; base_date?: string; trend?: Trend;
  realized_vol_20d_pct?: number | null; realized_vol_60d_pct?: number | null; high_52w?: number; low_52w?: number; vs_spy_20d?: number | null; vs_spy_60d?: number | null;
  ranges?: Record<string, RangeH>; expected_move?: { sigma_1d_pct: number; median_abs_move_pct: number; ratio_vs_median: number | null; sigma_pctile_1y: number | null } | null; range_method?: string;
}
export interface MarketLayer extends LayerMeta { composite?: string; legacy_label?: string; exposure?: number; exposure_continuous?: number; trend?: string; fast?: string; vol?: string; breadth?: string; confidence?: number; days_in_regime?: number; panic?: boolean; stress?: Record<string, number | boolean | null>; allowed?: string[]; spy?: { close: number; trend: Trend } }
export interface SectorLayer extends LayerMeta { key?: string; label?: string; etf?: string; tried?: string[]; trend?: Trend; rs_vs_spy_20d?: number | null; rs_vs_spy_60d?: number | null; stock_vs_sector_20d?: number | null; stock_vs_sector_60d?: number | null; stock_vs_spy_20d?: number | null; stock_vs_spy_60d?: number | null }
export interface OptionsLayer extends LayerMeta {
  contracts?: number; atm_iv?: number | null; atm_iv_30d?: number; front?: { exp_date: string; days_out: number }; implied_move_1d_pct?: number; implied_move_5d_pct?: number;
  skew_put_minus_call_pts?: number | null; skew_strikes?: string; model_move_5d_pct?: number; model_move_1d_pct?: number; implied_vs_model_5d?: number;
  earnings?: { status: string; date?: string; hour?: string; reason?: string; source?: string } | null;
}
/** One measured event type (research/event_reaction_study.md): history, never a prediction. */
export interface EventHistoryRow { label: string; n: number; car01_pct: number; t01: number; drift: string; conditional?: string }
export interface EventHistory extends EventHistoryRow { key: string; study_path: string; study_date: string; study_window: string; study_caveat: string }
export interface EventStudy { path: string; date: string; window: string; caveat: string }
export interface EventItem { kind: "filing" | "news"; form?: string; title: string; time: string; date: string; source: string; tier: number; url?: string | null; material?: boolean; items?: Array<{ item: string; label: string }>; history?: EventHistory | null }
export interface EventsLayer extends LayerMeta { items?: EventItem[]; form4_count?: number; window_days?: number; history_note?: string; event_history?: Record<string, EventHistoryRow>; event_study?: EventStudy }
export interface Touch { p_target_first: number; p_stop_first: number; p_neither: number; paths: number }
export interface Plan { entry_ref: number; horizon_sessions: number; stop: number; target: number; stretch_target: number; risk_per_share: number; reward_per_share: number; reward_risk: number | null; stretch_reward_risk: number | null; stop_pct: number; target_pct: number; stretch_pct: number; touch: Touch; touch_stretch: Touch; rules: Record<string, string> }
export interface DecileStats { n: number; pos_rate: number | null; mean_pct: number | null; rule_win_rate: number | null }
export interface Decile { decile: number; lo: number; hi: number; count: number; d5: DecileStats; d20: DecileStats; decisions: Record<string, number> }
export interface BotTrack { n: number; buckets: Decile[]; base: { d5: DecileStats; d20: DecileStats } | null; thin: boolean; source?: string }
export interface Direction { status: "call" | "none"; statement: string; source?: string; model_version?: string; model_accuracy?: { n: number; thin: boolean; hits: number; hit_rate: number | null; brier: number | null; brier_skill_vs_coinflip: number | null; in_range_rate: number | null }; call?: { direction: string; p_up: number; base_date: string; horizon_end: string; made_at?: string } | null; calibration_bin?: { lo: number; hi: number; n: number; mean_p: number | null; freq: number | null } | null; ticker_resolved?: number; ticker_hits?: number }
export interface LedgerSide { range: Record<string, { n: number; in68: number; in90: number; rate68: number | null; rate90: number | null }>; plan: { n: number; target_first: number; stop_first: number; neither: number; ambiguous: number; target_first_rate: number | null; stop_first_rate: number | null; neither_rate: number | null; ambiguous_rate: number | null } }
export interface Ledger { all: LedgerSide; ticker: LedgerSide | null; open: number; min_n: number }
export interface Guards { flags: Record<string, number>; reasons: Record<string, string>; would_block: string[]; composite: string | null; as_of: string | null; source?: string }
export interface InsiderSummary { windowDays: number; filings: number; buys: number; sells: number; buyShares: number; sellShares: number; rows: Array<{ filedAt: string | null; owner: string; role: string; kind: string; shares: number | null; price: number | null; date: string | null; url: string }>; source: string; note: string }
export interface Dossier {
  symbol: string; generated_at: string; errors: Record<string, string>; timings?: Record<string, number>;
  market?: MarketLayer; sector?: SectorLayer; stock?: StockLayer; options?: OptionsLayer; events?: EventsLayer;
  direction?: Direction; bot_track?: BotTrack | null; bot_guards?: Guards; plan?: Plan | null; ledger?: Ledger; insiders?: InsiderSummary | null;
  credibility_tiers?: Record<string, string>; cached?: boolean; cachedAgeS?: number; via?: string;
}

const fin = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
export const fmtPx = (v: number | null | undefined) => (fin(v) ? (Math.abs(v) >= 1000 ? `$${Math.round(v).toLocaleString("en-US")}` : `$${v.toFixed(2)}`) : "—");
export const fmtPct = (v: number | null | undefined, d = 1, signed = false) => (fin(v) ? `${signed && v > 0 ? "+" : ""}${v.toFixed(d)}%` : "—");
export const fmtRate = (v: number | null | undefined) => (fin(v) ? `${Math.round(v * 100)}%` : "—");

/** The symbol box: a stock symbol or an OCC code (looked up by its underlying). */
export function parseSymbol(raw: string): string | null {
  const s = raw.trim().toUpperCase().replace(/\s+/g, "");
  const m = /^([A-Z]{1,6})(\d{6}[CP]\d{8})?$/.exec(s);
  return m ? m[1] : null;
}

/** #/dossier/RKLB -> "RKLB" (null when the hash names no symbol). */
export function symbolFromHash(hash: string): string | null {
  const parts = hash.replace(/^#\/?/, "").split("/");
  return parts[0] === "dossier" && parts[1] ? parseSymbol(decodeURIComponent(parts[1])) : null;
}

/** Where each level sits on the band's 0–100 axis (the 90% band plus a margin; the live price is clamped and flagged). */
export function bandGeometry(b: Bands, base: number, price?: number | null) {
  const pad = (b.hi90 - b.lo90) * 0.08;
  const lo = b.lo90 - pad, hi = b.hi90 + pad, span = hi - lo || 1;
  const at = (v: number) => Math.max(0, Math.min(100, ((v - lo) / span) * 100));
  const out = { lo90: at(b.lo90), lo68: at(b.lo68), base: at(base), hi68: at(b.hi68), hi90: at(b.hi90), price: null as number | null, priceOutside: null as "below" | "above" | null };
  if (fin(price)) {
    out.price = at(price);
    out.priceOutside = price < b.lo90 ? "below" : price > b.hi90 ? "above" : null;
  }
  return out;
}

/** The trade-plan ladder: stop, entry, target, stretch on one axis (0 = stop side). */
export function ladderGeometry(p: Plan) {
  const lo = p.stop - (p.stretch_target - p.stop) * 0.06, hi = p.stretch_target + (p.stretch_target - p.stop) * 0.06, span = hi - lo || 1;
  const at = (v: number) => ((v - lo) / span) * 100;
  return { stop: at(p.stop), entry: at(p.entry_ref), target: at(p.target), stretch: at(p.stretch_target) };
}

/** Shares for a dollar risk: risk$ / (entry − stop), floored; null when the inputs cannot size. */
export function positionSize(riskDollars: number, entry: number, stop: number): { shares: number; notional: number; lossAtStop: number } | null {
  if (!fin(riskDollars) || !fin(entry) || !fin(stop) || riskDollars <= 0 || entry <= stop || stop <= 0) return null;
  const shares = Math.floor(riskDollars / (entry - stop));
  if (shares < 1) return { shares: 0, notional: 0, lossAtStop: 0 };
  return { shares, notional: shares * entry, lossAtStop: shares * (entry - stop) };
}

/** The decile whose score range holds the bot's live score. */
export function decileFor(track: BotTrack | null | undefined, score: number | null | undefined): Decile | null {
  if (!track || !track.buckets?.length || !fin(score)) return null;
  const b = track.buckets;
  if (score < b[0].lo) return b[0];
  for (const d of b) if (score >= d.lo && score < d.hi) return d;
  return b[b.length - 1];
}

/** "n=212: positive after 5 sessions 54% (all candidates 51%), mean +0.4%" — never a bare rate. */
export function decileLine(d: Decile | null, base: BotTrack["base"] | undefined, key: "d5" | "d20" = "d5"): string {
  if (!d) return "no bot score for this ticker yet";
  const s = d[key];
  const days = key === "d5" ? 5 : 20;
  if (!s.n) return `score ${d.lo.toFixed(0)}–${d.hi.toFixed(0)}: no settled +${days}-session outcomes yet`;
  const b = base?.[key];
  const thin = s.n < 30 ? " · thin sample — anecdote, not evidence" : "";
  return `n=${s.n}: up after ${days} sessions ${fmtRate(s.pos_rate)}${b && fin(b.pos_rate) ? ` (all scored candidates ${fmtRate(b.pos_rate)})` : ""}, mean ${fmtPct(s.mean_pct, 2, true)}${thin}`;
}

export function coverageLine(bt: Backtest | null | undefined): string {
  if (!bt) return "not enough history to backtest the band";
  return `backtest, last ${bt.n} sessions: 68% band held ${fmtRate(bt.rate68)} · 90% band held ${fmtRate(bt.rate90)}${bt.overlapping ? " (overlapping windows)" : ""}`;
}

/** The live ledger's line for one horizon: never a rate without its n. */
export function ledgerRangeLine(l: Ledger | undefined, h: "1" | "5", scope: "ticker" | "all" = "all"): string {
  const side = scope === "ticker" ? l?.ticker : l?.all;
  const r = side?.range?.[h];
  if (!r || !r.n) return scope === "ticker" ? "no scored bands for this ticker yet" : "no scored bands yet — the ledger fills nightly as horizons pass";
  const thin = r.n < (l?.min_n ?? 30) ? " · thin" : "";
  return `68% band contained the close ${r.in68}/${r.n} times (${fmtRate(r.rate68)}) · 90%: ${r.in90}/${r.n} (${fmtRate(r.rate90)})${thin}`;
}

export function ledgerPlanLine(l: Ledger | undefined): string {
  const p = l?.all?.plan;
  if (!p || !p.n) return "no plans scored yet — every served plan is scored after its 5 sessions";
  return `plans like this so far: target first ${fmtRate(p.target_first_rate)}, stop first ${fmtRate(p.stop_first_rate)}, neither ${fmtRate(p.neither_rate)}${p.ambiguous ? `, same-day both ${fmtRate(p.ambiguous_rate)}` : ""} · n=${p.n}${p.n < (l?.min_n ?? 30) ? " (thin)" : ""}`;
}

export function directionHeadline(d: Direction | undefined): { word: string; tone: "g" | "r" | "t" } {
  if (!d || d.status !== "call" || !d.call) return { word: "no validated direction", tone: "t" };
  const up = d.call.direction === "up";
  const p = up ? d.call.p_up : 1 - d.call.p_up;
  return { word: `${up ? "up" : "down"} ${Math.round(p * 100)}%`, tone: up ? "g" : "r" };
}

export function trackRecordLine(d: Direction | undefined): string {
  const a = d?.model_accuracy;
  if (!a || !a.n) return "the model has no resolved calls yet — its probability is untested";
  return `model record: ${a.hits}/${a.n} calls right (${fmtRate(a.hit_rate)})${fin(a.brier_skill_vs_coinflip) ? ` · Brier skill vs coin flip ${a.brier_skill_vs_coinflip >= 0 ? "+" : ""}${a.brier_skill_vs_coinflip.toFixed(2)}` : ""}${a.thin ? " · thin" : ""}`;
}

export const TIER_SHORT: Record<number, string> = { 1: "primary filing", 2: "wire / major outlet", 3: "aggregator", 4: "unverified" };

/** The layers that could not run, in order, for the honesty strip. */
export function layerGaps(d: Dossier | undefined): string[] {
  if (!d) return [];
  const out: string[] = [];
  for (const k of ["market", "sector", "stock", "options", "events"] as const) {
    const l = d[k] as LayerMeta | undefined;
    if (l && l.status !== "live" && l.reason) out.push(`${k}: ${l.reason}`);
  }
  return out;
}

export function ageText(iso: string | null | undefined, nowMs: number): string {
  const t = iso ? Date.parse(iso.length === 10 ? `${iso}T21:00:00Z` : iso) : NaN;
  if (!Number.isFinite(t)) return "—";
  const s = Math.max(0, Math.round((nowMs - t) / 1000));
  if (s < 90) return `${s}s ago`;
  if (s < 90 * 60) return `${Math.round(s / 60)}m ago`;
  if (s < 36 * 3600) return `${Math.round(s / 3600)}h ago`;
  return `${Math.round(s / 86400)}d ago`;
}

export const DOSSIER_RECENT_KEY = "vt-dossier-recent";
export const DOSSIER_RECENT_MAX = 8;
export function pushRecent(list: string[], sym: string): string[] {
  return [sym, ...list.filter((s) => s !== sym)].slice(0, DOSSIER_RECENT_MAX);
}

// ── measured event-reaction history (2026-10-29, research/event_reaction_study.md) ──
// No event type passed gate 2: every measured type was priced in on day 0–1.
// The page therefore shows the measured day-0–1 average as HISTORY with its n
// and t, and never a "not yet priced in" flag. Unmeasured types get no line.
export function historyLine(h: EventHistoryRow | null | undefined, study: { window: string; path: string; date: string } | null | undefined): string | null {
  if (!h || !study) return null;
  const sign = h.car01_pct > 0 ? "+" : "";
  const t = `t ${h.t01 > 0 ? "+" : ""}${h.t01.toFixed(1)}`;
  return `Historically (${study.window}, n=${h.n.toLocaleString("en-US")}): after ${h.label}, stocks moved ${sign}${h.car01_pct.toFixed(2)}% vs SPY on average over the filing day and the next (${t}); ${h.drift}.${h.conditional ? ` This ${h.conditional}.` : ""} Source: ${study.path}, ${study.date}.`;
}

/** The study's insider-cluster definition on the archive rows: ≥2 distinct officers/directors (not 10%-only holders) with open-market buys within 10 days. */
export function insiderCluster(rows: InsiderSummary["rows"] | null | undefined): boolean {
  const buys = (rows ?? []).filter((r) => r.kind === "buy" && r.role !== "10% owner" && (r.date || r.filedAt));
  for (const a of buys) {
    const ta = Date.parse(String(a.date || a.filedAt).slice(0, 10));
    const owners = new Set(buys.filter((b) => { const tb = Date.parse(String(b.date || b.filedAt).slice(0, 10)); return Number.isFinite(ta) && Number.isFinite(tb) && Math.abs(tb - ta) <= 10 * 86_400_000; }).map((b) => b.owner));
    if (owners.size >= 2) return true;
  }
  return false;
}

// ── cockpit KPI strip (2026-10-29, human-asked: the page fills the screen like
// the AI Engine): price & 5d, next-session ±%, next-5 ±%, options-implied vs
// model, regime, bot score vs min. Every value is one the page shows below. ────
export interface DossierKpi { label: string; value: string; sub?: string; subTone?: "g" | "r"; line: string; tone?: "g" | "r" | "t3" }

export function dossierKpis(d: Dossier, bot: { score: number | null; minScore: number | null | undefined; verdict: string | null | undefined; busy: boolean }): DossierKpi[] {
  const st = d.stock;
  const t5 = st?.trend?.ret_5d_pct;
  const range = (h: "1" | "5", label: string): DossierKpi => {
    const r = st?.ranges?.[h];
    const b = r?.bands;
    return b
      ? { label, value: `±${b.move68_pct.toFixed(1)}%`, line: `68% ${fmtPx(b.lo68)}–${fmtPx(b.hi68)} · 90% ±${b.move90_pct.toFixed(1)}% · ${r?.model === "har" ? "HAR-RV" : r?.model === "ewma" ? "EWMA" : "—"}` }
      : { label, value: "—", line: r?.fallback_reason ?? "no range — not enough price history" };
  };
  const o = d.options;
  const ivm = o?.implied_vs_model_5d, imp5 = o?.implied_move_5d_pct, mod5 = o?.model_move_5d_pct;
  const m = d.market;
  const vix = m?.stress?.vix, expo = m?.exposure;
  const score = bot.score;
  const min = fin(bot.minScore) ? bot.minScore : null;
  return [
    { label: "Price · 5 sessions", value: fmtPx(st?.price), sub: fin(t5) ? `${fmtPct(t5, 1, true)} 5d` : undefined, subTone: fin(t5) ? (t5 > 0 ? "g" : t5 < 0 ? "r" : undefined) : undefined,
      line: `last close ${fmtPx(st?.base_close)}${st?.base_date ? ` (${st.base_date})` : ""}${st?.price_is_partial_session ? " · session in progress" : ""}` },
    range("1", "Next session"),
    range("5", "Next 5 sessions"),
    { label: "Options-implied vs model", value: fin(ivm) ? `${ivm.toFixed(2)}×` : "—",
      line: fin(imp5) && fin(mod5) ? `implied ±${fmtPct(imp5)} · model ±${fmtPct(mod5)} · 5 sessions` : (o?.reason ?? "options unavailable") },
    { label: "Regime", value: m?.status === "live" ? (m.composite ?? "—") : "—",
      line: m?.status === "live" ? `exposure ${fin(expo) ? `${Math.round(expo * 100)}%` : "—"} · ${m.days_in_regime ?? "—"} days · VIX ${fin(vix) ? vix.toFixed(1) : "—"}` : (m?.reason ?? "market layer unavailable") },
    { label: "Bot score vs min", value: score !== null ? score.toFixed(1) : bot.busy ? "…" : "—", sub: min !== null ? `/ ${min}` : undefined,
      line: score !== null ? (bot.verdict ?? "undecided") : bot.busy ? "scoring through the scan path…" : "no bot score (the look-up could not score this name)",
      tone: score !== null && min !== null ? (score >= min ? "g" : "r") : score === null ? "t3" : undefined },
  ];
}
