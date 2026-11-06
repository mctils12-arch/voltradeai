// decisionJournal.ts — the DECISION JOURNAL, Node half (2026-11-06; human
// refocus spec §7 "log every signal, decision, and outcome", §8 per-pick
// plain-English cards). decision_journal.py documents the row shape and owns
// the Python producers (scan_market's gates, the tier engine's CSP refusals
// and master kill, options_manager's exits, the volatility-managed floor).
// This module appends the SAME rows to the SAME file for the decisions made in
// server/bot.ts — executeTrades' stock / ETF / option entries and its gates,
// the tier dispatcher's SELL_CSP / BUY / BUY_PUT outcomes, the stock exits
// (WS exits, scale-outs, POS-KILL, TIME-EXIT) — and serves the journal:
// GET /api/bot/decisions?ticker=&limit= and the "decisions" diag probe.
//
//   file: DATA_DIR/voltrade_decisions.jsonl (+ .1, the previous generation),
//         and the ONE-TIME backfill DATA_DIR/voltrade_decisions_backfill.jsonl
//         (server/decisionBackfill.ts; every row there carries backfilled:
//         true and is never appended to again).
//   action: "buy" = opened a position (a sold cash-secured put is an entry
//         too; `side` holds the order side), "sell" = closed one, "skip" = a
//         candidate not taken, "rotate" = a roll, "scale_in" / "scale_out" =
//         the floor or a partial exit.
//
// LOGGING ONLY: no order, size or gate reads this; every writer swallows and
// logs its own errors. The off switch is the env var VOLTRADE_DECISION_JOURNAL
// (any off value) — mirroring system_config DECISION_JOURNAL.enabled, whose
// default (on) and size cap test_decision_journal.py pins against this file.
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { attachCounterfactualOutcomes, contractLegs, wouldBeFor } from "./instrumentCounterfactual";

export const DECISIONS_FILENAME = "voltrade_decisions.jsonl";
export const DECISIONS_BACKFILL_FILENAME = "voltrade_decisions_backfill.jsonl";
/** mirrors system_config BASE_CONFIG["DECISION_JOURNAL"]["max_bytes"] (test_decision_journal.py pins the parity) */
export const DECISION_JOURNAL_MAX_BYTES = 5_000_000;
export const DECISION_JOURNAL_ENV = "VOLTRADE_DECISION_JOURNAL";
export const DECISION_ROW_MAX_BYTES = 8_000;
export const DECISIONS_DEFAULT_LIMIT = 50;
export const DECISIONS_MAX_LIMIT = 500;
const READ_CHUNK_BYTES = 65_536;
const OFF_VALUES = new Set(["", "0", "off", "false", "no", "none"]);

export const DECISION_ACTIONS = ["buy", "sell", "skip", "rotate", "scale_in", "scale_out"] as const;
export type DecisionAction = (typeof DECISION_ACTIONS)[number];
export const STAGE_KEYS = ["regime", "calendar", "flows", "sector", "selection", "options", "tax"] as const;
export type StageKey = (typeof STAGE_KEYS)[number];
type Obj = Record<string, unknown>;

/** `would_be` (2026-11-06, server/instrumentCounterfactual.ts): on an entry the
 *  instrument selector decided, the would-be entry of the instrument passed over
 *  across the stock/options line, plus its settled outcomes when read. */
export interface RejectedAlt { instrument: string; reason: string; would_be?: Obj }
export interface DecisionRow {
  v: number; id: string; ts: string; ticker: string; action: DecisionAction;
  instrument: string; chosen: string; rejected: RejectedAlt[];
  stages: Record<string, Obj | null>;
  sizing: Obj | null; summary: string;
  source?: string; signal?: string; side?: string; symbol?: string;
  qty?: number; price?: number; notional?: number; order?: Obj; mode?: string;
  rule?: string; pnl_pct?: number; dedupe?: string; backfilled?: boolean; truncated?: boolean;
}
const OPTIONAL_KEYS = ["source", "signal", "side", "symbol", "qty", "price", "notional", "order", "mode",
  "rule", "pnl_pct", "dedupe", "backfilled", "truncated"] as const;

// ── small helpers ────────────────────────────────────────────────────────────
function rec(x: unknown): Obj {
  return x && typeof x === "object" && !Array.isArray(x) ? (x as Obj) : {};
}
function num(x: unknown): number | null {
  if (x === null || x === undefined || x === "" || typeof x === "boolean") return null;
  const v = typeof x === "number" ? x : parseFloat(String(x));
  return Number.isFinite(v) ? v : null;
}
function text(x: unknown, n = 400): string {
  const s = x === null || x === undefined ? "" : String(x);
  return s.replace(/\s+/g, " ").trim().slice(0, n);
}
function sentence(x: unknown, n = 400): string {
  const s = text(x, n);
  return !s || /[.!?]$/.test(s) ? s : `${s}.`;
}
const r1 = (x: number) => Math.round(x * 10) / 10;
const r2 = (x: number) => Math.round(x * 100) / 100;

export function money(x: unknown): string {
  const v = num(x);
  if (v === null) return "?";
  const a = Math.abs(v);
  const s = a >= 100 ? a.toLocaleString("en-US", { maximumFractionDigits: 0 }) : a.toFixed(2);
  return `${v < 0 ? "-" : ""}$${s}`;
}

/** A per-share / per-contract price: always cents ("$150.20"). */
export function priceText(x: unknown): string {
  const v = num(x);
  return v === null ? "?" : `${v < 0 ? "-" : ""}$${Math.abs(v).toFixed(2)}`;
}

/** JSON-safe copy: non-finite numbers -> null, bounded depth / width. */
function clean(x: unknown, depth = 0): unknown {
  if (depth > 6) return null;
  if (x === null || x === undefined) return null;
  if (typeof x === "number") return Number.isFinite(x) ? x : null;
  if (typeof x === "string" || typeof x === "boolean") return x;
  if (Array.isArray(x)) return x.slice(0, 40).map((v) => clean(v, depth + 1));
  if (typeof x === "object") {
    const out: Obj = {};
    for (const [k, v] of Object.entries(x as Obj).slice(0, 60)) out[k.slice(0, 60)] = clean(v, depth + 1);
    return out;
  }
  return text(x, 200);
}

const OCC_RE = /^([A-Z.]{1,6})(\d{2})(\d{2})(\d{2})([CP])(\d{8})$/;
/** "TLT261219P00085000" -> "TLT $85 put expiring 2026-12-19"; anything else as given. */
export function describeOcc(symbol: unknown): string {
  const s = String(symbol ?? "");
  const m = OCC_RE.exec(s);
  if (!m) return s;
  return `${m[1]} $${parseInt(m[6], 10) / 1000} ${m[5] === "C" ? "call" : "put"} expiring 20${m[2]}-${m[3]}-${m[4]}`;
}
export function occUnderlying(symbol: unknown): string {
  const m = OCC_RE.exec(String(symbol ?? ""));
  return m ? m[1] : String(symbol ?? "").toUpperCase();
}

// ── the switch, the files ────────────────────────────────────────────────────
export function journalEnabled(env: Record<string, string | undefined> = process.env): boolean {
  const v = env[DECISION_JOURNAL_ENV];
  return v === undefined ? true : !OFF_VALUES.has(String(v).trim().toLowerCase());
}
export function decisionsDir(env: Record<string, string | undefined> = process.env): string {
  return env.DATA_DIR || (fs.existsSync("/data") ? "/data/voltrade" : "/tmp");
}

// ── plain-English vocabulary (decision_journal.py RULE_TEXT is the twin) ────
export const RULE_TEXT: Readonly<Record<string, string>> = {
  rejected_held: "it is already held",
  rejected_stock_slots: "every stock slot was taken",
  rejected_cooldown: "it was stopped out recently (re-entry cooldown)",
  rejected_regime_gate: "new stock buys are off in this market regime",
  rejected_score: "its score was below the bar",
  rejected_heat: "its sector already holds the maximum number of positions",
  rejected_spread: "the bid-ask spread was too wide",
  rejected_other: "a pre-trade check refused it",
  rejected_dollar_volume: "it trades too little dollar volume",
  rejected_blocked_ticker: "it is on the blocked list",
  rejected_extreme_mover: "it already moved more than 50% today",
  rejected_bearish_stock_signal: "the signal was bearish and stock shorting is off",
  rejected_sizing: "position sizing gave it no size",
  rejected_sizing_error: "position sizing failed on it",
  rejected_zero_shares: "position sizing rounded to zero shares",
  rejected_volume_share: "the position would be too large a share of its daily volume",
  rejected_options_slots: "every options slot was taken",
  rejected_node_score: "its score was below the server's threshold",
  rejected_daily_block: "it is blocked for the day after repeated stop-losses",
  rejected_size_cap: "the position was above the per-position size ceiling",
  rejected_exposure_cap: "total exposure would have gone over the cap",
  rejected_corr_cap: "its sector would be over-concentrated",
  rejected_stock_halted: "trading in it was halted",
  rejected_max_retry: "too many order attempts on it today",
  rejected_dup_order: "an order for it was already pending",
  rejected_buying_power: "there was not enough buying power",
  rejected_killswitch: "the kill switch was on",
  rejected_options_nocontract: "no suitable option contract was found",
  rejected_options_error: "the options engine failed on it",
  rejected_csp_resubmit: "its last put order was just cancelled unfilled or is still working",
  rejected_masterkill: "the tier engine's master kill switch blocked all new entries",
  rejected_capital: "there was not enough capital to secure the put",
  rejected_order: "the broker refused the order",
  shadow_mode_repeat: "it was already shadow-entered today",
};
export function ruleText(decision: unknown): string {
  const d = String(decision ?? "");
  if (RULE_TEXT[d]) return RULE_TEXT[d];
  if (d.startsWith("rejected_regime_")) return `the regime guard '${d.slice("rejected_regime_".length)}' blocked it`;
  return d.replace(/^rejected_/, "").replace(/_/g, " ").trim() || "a gate blocked it";
}

/** Stock exit types (bot.ts WS-EXIT / POS-KILL / TIME-EXIT), in plain English. */
export const STOCK_EXIT_TEXT: Readonly<Record<string, string>> = {
  stop_loss: "the stop-loss was hit",
  trailing_stop: "the trailing stop was hit",
  breakeven_stop: "the breakeven stop was hit",
  take_profit: "the profit target was hit",
  time_stop: "the time stop fired (held too long without working)",
  etf_max_hold: "a leveraged ETF reached its maximum holding period",
  position_kill: "the position-loss backstop (POS-KILL) forced the exit",
  time_exit: "it was stale (held over 14 days with little P&L)",
  scale_out: "a profit-taking scale-out",
};
export function stockExitText(t: unknown): string {
  const k = String(t ?? "");
  return STOCK_EXIT_TEXT[k] || k.replace(/_/g, " ").trim() || "an exit rule fired";
}

// ── stages (decision_journal.py's twins) ────────────────────────────────────
export const SUBSCORE_WEIGHTS: Readonly<Record<string, number>> = { momentum: 0.25, mean_reversion: 0.20, vrp: 0.25, squeeze: 0.15, volume: 0.15 };
export const SUBSCORE_ADDITIVE = ["vol_premium", "macro", "news", "adx", "obv", "intel", "alt", "social", "fundamental", "recommendation", "contrarian", "rel_strength"] as const;
export const SUBSCORE_LABELS: Readonly<Record<string, string>> = {
  momentum: "momentum", mean_reversion: "mean reversion", vrp: "volatility risk premium", squeeze: "short squeeze",
  volume: "volume", vol_premium: "volatility premium bonus", macro: "macro backdrop", news: "news",
  adx: "trend strength (ADX)", obv: "on-balance volume", intel: "insider and analyst intelligence",
  alt: "alternative data", social: "social sentiment", fundamental: "fundamentals",
  recommendation: "analyst recommendation", contrarian: "contrarian signal", rel_strength: "relative strength vs SPY",
  insider_mspr: "insider buying (MSPR)",
};
export interface Driver { name: string; label: string; points: number }

/** Each term's points in deep_score's RULES score vs a neutral 50 (weighted
 *  components) or as added (adjustments), largest |points| first. */
export function scoreDrivers(breakdown: unknown): Driver[] {
  const bd = rec(breakdown);
  const out: Driver[] = [];
  for (const [k, w] of Object.entries(SUBSCORE_WEIGHTS)) {
    const v = num(bd[k]);
    if (v !== null) out.push({ name: k, label: SUBSCORE_LABELS[k], points: r1((v - 50) * w) });
  }
  for (const k of SUBSCORE_ADDITIVE) {
    const v = num(bd[k]);
    if (v !== null && v !== 0) out.push({ name: k, label: SUBSCORE_LABELS[k], points: r1(v) });
  }
  return out.sort((a, b) => Math.abs(b.points) - Math.abs(a.points) || a.name.localeCompare(b.name));
}
export function dominantDriver(drivers: readonly Driver[]): Driver | null {
  const pos = drivers.filter((d) => d.points > 0);
  if (pos.length) return pos.reduce((a, b) => (b.points > a.points ? b : a));
  return drivers[0] ?? null;
}

/** The regime risk budget's compact view (scan_market's `risk_budget`,
 *  regime_engine.risk_budget_view) as a stage carries it; null without one or on its error. */
export function budgetView(b: unknown): Obj | null {
  const o = rec(b);
  if (o.error || num(o.budget) === null) return null;
  return {
    budget: num(o.budget), band: text(o.band, 40) || null, summary: text(o.summary, 400) || null,
    drivers: (Array.isArray(o.drivers) ? o.drivers : []).slice(0, 6).map((d) => text(d, 160)),
    applied: o.applied === true, stale: o.stale === true, as_of: text(o.as_of, 40) || null,
  };
}

/** The regime stage. With the regime risk budget, its ready-made plain-English
 *  `summary` is the stage's reason; else the label + VXX read (decision_journal.py's twin). */
export function regimeStage(label: unknown, gateRegime: unknown, vxxRatio: unknown, reason?: string, budget?: unknown): Obj | null {
  const lbl = text(label, 40) || null;
  const gate = text(gateRegime, 40) || null;
  const vxx = num(vxxRatio);
  const rb = budgetView(budget);
  if (!lbl && !gate && vxx === null && !reason && !rb) return null;
  const shown = gate || lbl;
  let t = shown ? `${shown} market` : "market regime";
  if (vxx !== null) t += ` (VXX ${vxx.toFixed(2)}x its 30-day average)`;
  if (lbl && gate && lbl !== gate) t += `; the scorer's regime read was ${lbl}`;
  if (reason) t += `; ${text(reason, 200)}`;
  const out: Obj = { label: shown, scorer_label: lbl, gate_regime: gate, vxx_ratio: vxx, reason: t };
  if (rb) {
    out.risk_budget = rb;
    if (rb.summary) out.reason = `${rb.summary}${shown || vxx !== null ? ` (${t})` : ""}`;
  }
  return out;
}

/** The institutional-flow stage: the trade's / floor result's `flows` block
 *  (flows_model: sector, etf, pressure, multiplier, applied, reason), kept as-is. */
export function flowsStage(flows: unknown): Obj | null {
  const o = rec(flows);
  return Object.keys(o).length ? (clean(o) as Obj) : null;
}
function sectorStage(sector: unknown): Obj | null {
  const s = text(sector, 60);
  return s && s.toLowerCase() !== "unknown" ? { name: s, reason: null } : null;
}
/** A real regime label on a trade dict (the options scanner's marker is not one). */
function tradeRegimeLabel(t: Obj): string | null {
  const r = text(t.regime_at_entry, 40);
  return r && r !== "OPTIONS_SCANNER" && r !== "UNKNOWN" ? r : null;
}

export function selectionStage(t: Obj): Obj | null {
  const score = num(t.score ?? t.deep_score);
  const drivers = scoreDrivers(t.score_breakdown);
  const bd = rec(t.score_breakdown);
  const subs: Obj = {};
  for (const k of [...Object.keys(SUBSCORE_WEIGHTS), ...SUBSCORE_ADDITIVE, "insider_mspr", "rules_raw"]) {
    const v = num(bd[k]);
    if (v !== null) subs[k] = r1(v);
  }
  if (score === null && !drivers.length) return null;
  const reasons = Array.isArray(t.reasons) ? t.reasons.slice(0, 6).map((r) => text(r, 160)).filter(Boolean) : [];
  return {
    score: score === null ? null : r1(score), threshold: num(t.score_threshold),
    rules_score: num(t.rules_score), ml_score: num(t.ml_score_raw),
    subscores: Object.keys(subs).length ? subs : null,
    drivers: drivers.slice(0, 5), dominant: dominantDriver(drivers), reasons,
    recommendation: text(t.recommendation, 40) || null, side: text(t.side, 12) || null,
  };
}

/** What produced a trade dict, in a few words (the attribution's "signal"). */
export function signalForTrade(t: Obj): string {
  if (text(t.regime_at_entry) === "OPTIONS_SCANNER") return `options scanner: ${text(t.setup || t.options_strategy || "setup", 60)}`;
  if (text(t.instrument_strategy) === "covered_call" || text(t.options_strategy) === "covered_call") return "covered call on held shares";
  const dom = dominantDriver(scoreDrivers(t.score_breakdown));
  const base = text(t.trade_type) === "options" ? "deep_score scan (options)" : "deep_score scan";
  return dom && dom.points > 0 ? `${base}, ${dom.label}-led` : base;
}

/** instrument_selector's per-instrument scores -> the alternatives NOT chosen and why. */
export function rejectedAlternatives(t: Obj, chosen: string): RejectedAlt[] {
  const scores = rec(t.instrument_scores);
  const key = chosen === "option" ? "options" : chosen;
  const chosenScore = num(rec(scores[key]).score);
  const out: RejectedAlt[] = [];
  const names: Record<string, string> = { stock: "stock", etf: "etf", options: "option" };
  const reasoning = text(t.instrument_reasoning, 600);
  for (const k of ["stock", "etf", "options"]) {
    if (k === key) continue;
    const s = scores[k];
    if (s === null || s === undefined) {
      if (!Object.keys(scores).length) continue;            // no selector ran (scanner / tier trades)
      if (k === "etf") out.push({ instrument: "etf", reason: "no leveraged ETF exists for it in this direction" });
      if (k === "options") {
        const ro = /Options ruled out: ([^|]+)/.exec(reasoning)?.[1];
        out.push({ instrument: "option", reason: ro ? `ruled out: ${text(ro, 160)}` : "options were not scored (no options data or no viable strategy)" });
      }
      continue;
    }
    const o = rec(s);
    const sc = num(o.score);
    const why = text(o.reasoning, 240);
    const ruled = /\[RULED OUT: ([^\]]+)\]/.exec(why)?.[1];
    const vs = sc !== null && chosenScore !== null ? `scored ${sc} vs ${chosenScore} for the ${chosen}` : "scored lower";
    out.push({ instrument: names[k], reason: ruled ? `ruled out: ${text(ruled, 200)}` : `${vs} (${why.replace(/\s*\[RULED OUT[^\]]*\]/, "")})` });
  }
  return out;
}

function extraStages(t: Obj): Record<string, Obj> {
  const out: Record<string, Obj> = {};
  for (const [k, v] of Object.entries(rec(t.decision_stages))) if (v && typeof v === "object" && !Array.isArray(v)) out[k] = v as Obj;
  return out;
}

// ── normalize / write / read ─────────────────────────────────────────────────
export function newDecisionId(): string { return crypto.randomBytes(8).toString("hex"); }

/** The contract shape, or null when the input cannot be a decision row. */
export function normalizeDecision(raw: unknown): DecisionRow | null {
  const r = rec(raw);
  const action = text(r.action, 20).toLowerCase() as DecisionAction;
  const ticker = text(r.ticker, 24).toUpperCase();
  if (!DECISION_ACTIONS.includes(action) || !ticker) return null;
  const stIn = rec(r.stages);
  const stages: Record<string, Obj | null> = {};
  for (const k of STAGE_KEYS) { const v = stIn[k]; stages[k] = v && typeof v === "object" && !Array.isArray(v) ? (clean(v) as Obj) : null; }
  for (const [k, v] of Object.entries(stIn)) if (!(k in stages) && v && typeof v === "object" && !Array.isArray(v)) stages[k.slice(0, 30)] = clean(v) as Obj;
  const rejected: RejectedAlt[] = (Array.isArray(r.rejected) ? r.rejected : [])
    .map(rec).filter((x) => text(x.instrument)).slice(0, 6)
    .map((x) => {
      const alt: RejectedAlt = { instrument: text(x.instrument, 40), reason: text(x.reason, 300) };
      if (Object.keys(rec(x.would_be)).length) alt.would_be = clean(x.would_be) as Obj;
      return alt;
    });
  const out: DecisionRow = {
    v: num(r.v) ?? 1, id: text(r.id, 40) || newDecisionId(), ts: text(r.ts, 40) || new Date().toISOString(),
    ticker, action, instrument: text(r.instrument || "stock", 40),
    chosen: text(r.chosen, 80) || (action === "skip" ? "none" : ""), rejected, stages,
    sizing: r.sizing && typeof r.sizing === "object" && !Array.isArray(r.sizing) ? (clean(r.sizing) as Obj) : null,
    summary: text(r.summary, 900),
  };
  const o = out as unknown as Obj;
  for (const k of OPTIONAL_KEYS) {
    const v = r[k];
    if (v === "" || (Array.isArray(v) && !v.length)) continue;
    const c = clean(v);
    if (c !== null) o[k] = c;
  }
  return out;
}

function fit(row: DecisionRow): string {
  let s = JSON.stringify(row);
  if (Buffer.byteLength(s) <= DECISION_ROW_MAX_BYTES) return s;
  const sel = row.stages.selection ? { ...row.stages.selection } : null;
  if (sel) { delete sel.reasons; delete sel.subscores; }
  const cut: DecisionRow = {
    ...row, summary: row.summary.slice(0, 400), truncated: true,
    rejected: row.rejected.slice(0, 4).map((x) => ({ instrument: x.instrument, reason: x.reason.slice(0, 160), ...(x.would_be ? { would_be: x.would_be } : {}) })),
    stages: { ...row.stages, selection: sel },
  };
  s = JSON.stringify(cut);
  if (Buffer.byteLength(s) <= DECISION_ROW_MAX_BYTES) return s;
  const bare: DecisionRow = { ...cut, stages: Object.fromEntries(STAGE_KEYS.map((k) => [k, null])), sizing: null };
  delete bare.order;
  return JSON.stringify(bare);
}

/** The ET calendar date (YYYY-MM-DD) skips are deduplicated on. */
export function etDay(ms: number): string {
  return new Intl.DateTimeFormat("en-CA", { timeZone: "America/New_York", year: "numeric", month: "2-digit", day: "2-digit" }).format(new Date(ms));
}
/** "skip|<ticker>|<rule>|<ET day>" — the same key format decision_journal.py writes. */
export function skipDedupeKey(ticker: string, rule: string, nowMs: number): string {
  return `skip|${ticker}|${rule}|${etDay(nowMs)}`;
}
const seenToday: { day: string | null; keys: Set<string> } = { day: null, keys: new Set() };
/** True the first time `key` (which ends in its ET day) is offered in this process. */
export function firstTimeToday(key: string): boolean {
  const day = key.slice(key.lastIndexOf("|") + 1);
  if (seenToday.day !== day) { seenToday.day = day; seenToday.keys = new Set(); }
  if (seenToday.keys.has(key)) return false;
  seenToday.keys.add(key);
  return true;
}
/** Tests: forget what was seen. */
export function resetDecisionDedupe(): void { seenToday.day = null; seenToday.keys = new Set(); }

/** Append rows (ONE write, so a Python append never lands inside a line),
 *  rotating to <file>.1 above maxBytes. A row with a `dedupe` key (a skip) is
 *  written once per key — once per ticker, gate and ET day. Returns rows
 *  written; never throws. */
export function appendDecisions(dir: string, rows: readonly unknown[], maxBytes = DECISION_JOURNAL_MAX_BYTES,
  env: Record<string, string | undefined> = process.env): number {
  if (!journalEnabled(env) || !rows.length) return 0;
  const lines: string[] = [];
  for (const r of rows) {
    const n = normalizeDecision(r);
    if (!n || (n.dedupe && !firstTimeToday(n.dedupe))) continue;
    lines.push(fit(n) + "\n");
  }
  if (!lines.length) return 0;
  const p = path.join(dir, DECISIONS_FILENAME);
  try {
    if (fs.existsSync(p) && fs.statSync(p).size > maxBytes) fs.renameSync(p, p + ".1");
    fs.appendFileSync(p, lines.join(""));
    return lines.length;
  } catch (err) {
    console.error("[decision-journal] rows not written (no decision changed):", err instanceof Error ? err.message : err);
    return 0;
  }
}

export interface DecisionQuery { ticker?: string | null; limit?: number; actions?: readonly string[] | null }

function matches(row: DecisionRow, q: DecisionQuery): boolean {
  if (q.actions && q.actions.length && !q.actions.includes(row.action)) return false;
  if (q.ticker) {
    const t = q.ticker.toUpperCase();
    if (row.ticker !== t && String(row.symbol ?? "").toUpperCase() !== t) return false;
  }
  return true;
}

function parseLine(line: string): DecisionRow | null {
  const s = line.trim();
  if (!s) return null;
  try { return normalizeDecision(JSON.parse(s)); } catch { return null; }
}

/** Newest-first rows of one JSONL file read BACKWARDS in chunks until `limit`
 *  matches: an unfiltered query reads only the tail; a filtered one keeps one
 *  chunk in memory at a time. Lines are split on the newline BYTE, so a
 *  multi-byte character across a chunk edge is never cut. */
export function scanFileBackwards(file: string, q: DecisionQuery, limit: number, out: DecisionRow[]): void {
  let fd: number;
  try { fd = fs.openSync(file, "r"); } catch { return; }
  try {
    let pos = fs.fstatSync(fd).size;
    let carry = Buffer.alloc(0);
    const take = (bytes: Buffer) => {
      const row = parseLine(bytes.toString("utf8"));
      if (row && matches(row, q)) out.push(row);
    };
    while (pos > 0 && out.length < limit) {
      const start = Math.max(0, pos - READ_CHUNK_BYTES);
      const buf = Buffer.alloc(pos - start);
      fs.readSync(fd, buf, 0, buf.length, start);
      pos = start;
      const data = Buffer.concat([buf, carry]);
      let end = data.length;
      for (let i = data.length - 1; i >= 0 && out.length < limit; i--) {
        if (data[i] !== 0x0a) continue;
        if (end > i + 1) take(data.subarray(i + 1, end));
        end = i;
      }
      carry = Buffer.from(data.subarray(0, end));
    }
    if (pos === 0 && carry.length && out.length < limit) take(carry);
  } finally {
    fs.closeSync(fd);
  }
}

/** Newest-first decisions: the live journal (current, then .1) and the
 *  one-time backfill, filtered by ticker (underlying or ordered symbol) /
 *  actions, sorted by ts, at most `limit`. */
export function readDecisions(dir: string, q: DecisionQuery = {}): DecisionRow[] {
  const limit = Math.min(Math.max(Math.floor(q.limit ?? DECISIONS_DEFAULT_LIMIT) || DECISIONS_DEFAULT_LIMIT, 1), Number.MAX_SAFE_INTEGER);
  const p = path.join(dir, DECISIONS_FILENAME);
  const live: DecisionRow[] = [];
  scanFileBackwards(p, q, limit, live);
  if (live.length < limit) scanFileBackwards(p + ".1", q, limit - live.length, live);
  const back: DecisionRow[] = [];
  scanFileBackwards(path.join(dir, DECISIONS_BACKFILL_FILENAME), q, limit, back);
  const ms = (r: DecisionRow) => { const t = Date.parse(r.ts); return Number.isFinite(t) ? t : 0; };
  // 2026-11-06: an entry's passed-over instrument gains its settled outcomes (instrumentCounterfactual.ts)
  return attachCounterfactualOutcomes([...live, ...back].sort((a, b) => ms(b) - ms(a)).slice(0, limit), dir);
}

/** ?ticker=&limit= -> a DecisionQuery for the API (limit 1..DECISIONS_MAX_LIMIT). */
export function decisionsQuery(raw: Record<string, unknown>): DecisionQuery {
  const t = text(raw.ticker, 24).toUpperCase().replace(/[^A-Z0-9.]/g, "");
  const n = parseInt(String(raw.limit ?? ""), 10);
  return { ticker: t || null, limit: Math.min(Math.max(Number.isFinite(n) ? n : DECISIONS_DEFAULT_LIMIT, 1), DECISIONS_MAX_LIMIT) };
}

// ── Node producers ───────────────────────────────────────────────────────────
// Each exported builder is GUARDED: it returns null (logged) instead of
// throwing, because bot.ts calls them inside order paths — a journal bug must
// never stand between a decision and its order or its bookkeeping.
function guarded<A extends unknown[]>(name: string, fn: (...a: A) => DecisionRow | null): (...a: A) => DecisionRow | null {
  return (...a: A) => {
    try { return fn(...a); } catch (err) {
      console.error(`[decision-journal] ${name} failed (no decision changed):`, err instanceof Error ? err.message : err);
      return null;
    }
  };
}
export interface EntryOutcome {
  instrument: "stock" | "etf" | "option";
  /** the strategy acted on: buy_stock, buy_etf_bull, sell_cash_secured_put ... */
  chosen: string;
  side: string;
  /** what was ordered (stock / ETF ticker / OCC); the trade's ticker stays the underlying */
  symbol: string;
  qty: number | null;
  price: number | null;
  notional: number | null;
  status: string;
  orderId?: string | null;
  mode: "live" | "shadow";
  source: string;
  equity?: number | null;
  contract?: unknown;
  detail?: string | null;
}

/** The stock was bought although the selector chose the ETF / options path:
 *  that path declined or failed at execution (its own audit row says why). */
function fallbackAlternatives(t: Obj, o: EntryOutcome): RejectedAlt[] {
  if (o.instrument !== "stock") return [];
  const out: RejectedAlt[] = [];
  const etf = text(t.instrument_ticker, 12).toUpperCase();
  if (text(t.instrument) === "etf" && etf && etf !== text(t.ticker, 24).toUpperCase()) {
    out.push({ instrument: "etf", reason: `the instrument selector chose the leveraged ETF ${etf}, but its order could not be planned or placed (the ETF-FALLBACK audit row says why), so the stock was bought` });
  }
  if (t.use_options === true && text(t.options_strategy) && text(t.options_strategy) !== "stock") {
    out.push({ instrument: "option", reason: `the ${text(t.options_strategy, 40).replace(/_/g, " ")} path declined at execution or its order failed (the OPTIONS-SKIP / OPTIONS-FALLBACK audit row says why), so the stock was bought` });
  }
  return out;
}

function pctOfEquity(notional: number | null, equity: number | null | undefined): string {
  return notional !== null && equity && equity > 0 ? `, ${r1((notional / equity) * 100)}% of equity` : "";
}

function whyLine(t: Obj, stages: Record<string, Obj | null>): string {
  const sel = stages.selection;
  const parts: string[] = [];
  const score = num(sel?.score);
  const th = num(sel?.threshold);
  const reg = text(rec(stages.regime).label);
  if (score !== null) parts.push(`score ${score}${th !== null ? ` cleared the ${th} needed` : ""}${reg ? ` in a ${reg} market` : ""}`);
  const dom = rec(sel?.dominant);
  if (num(dom.points) !== null && (num(dom.points) as number) > 0) parts.push(`biggest driver ${text(dom.label)} (+${num(dom.points)} pts)`);
  const reasons = Array.isArray(sel?.reasons) ? (sel?.reasons as unknown[]) : [];
  if (reasons.length) parts.push(text(reasons[0], 140));
  else if (text(t.instrument_reasoning)) parts.push(text(t.instrument_reasoning, 160));
  return parts.join("; ");
}

function optionStage(t: Obj, o: EntryOutcome): Obj | null {
  if (o.instrument !== "option") return null;
  const c = rec(o.contract);
  const occ = text(c.occ_symbol || (OCC_RE.test(o.symbol) ? o.symbol : ""), 30);
  const legs = contractLegs(c, o.side);
  return clean({
    strategy: o.chosen || null, setup: text(t.setup, 60) || null, occ_symbol: occ || null,
    // 2026-11-06: every leg of the structure (instrument_counterfactual.py prices them)
    legs: legs.length ? legs : (occ ? [{ symbol: occ, side: o.side }] : null),
    contract: occ ? describeOcc(occ) : null, side: o.side, qty: o.qty,
    limit_price: num(c.limit_price) ?? o.price, credit: num(c.net_credit ?? c.credit),
    max_loss: num(c.max_loss), cash_required: num(c.cash_required), delta: num(c.delta),
    edge_pct: num(t.options_edge_pct), reason: text(t.options_reasoning || t.instrument_reasoning, 300) || null,
  }) as Obj;
}

/** An entry executeTrades placed (or shadow-logged) for a scan trade dict. */
function buildEntryDecision(trade: unknown, o: EntryOutcome, nowMs = Date.now()): DecisionRow | null {
  const t = rec(trade);
  const ticker = text(t.ticker, 24).toUpperCase();
  const stages: Record<string, Obj | null> = {
    regime: regimeStage(tradeRegimeLabel(t), t.scan_regime, null, undefined, t.risk_budget),
    calendar: null, flows: flowsStage(t.flows), sector: sectorStage(t.sector),
    selection: selectionStage(t), options: optionStage(t, o), tax: null, ...extraStages(t),
  };
  const fallback = fallbackAlternatives(t, o);
  const rejected = [...fallback, ...rejectedAlternatives(t, o.instrument).filter((r) => !fallback.some((f) => f.instrument === r.instrument))];
  // INSTRUMENT COUNTERFACTUAL 2026-11-06: the passed-over instrument's would-be entry
  const wb = wouldBeFor(t, o);
  if (wb) {
    const i = rejected.findIndex((r) => r.instrument === wb.instrument);
    if (i >= 0) rejected[i] = { ...rejected[i], would_be: wb.would_be };
    else rejected.push({ instrument: wb.instrument, reason: `the ${wb.instrument} alternative was not scored by the instrument selector`, would_be: wb.would_be });
  }
  const shadow = o.mode === "shadow" ? "Shadow mode, no order placed: " : "";
  let what: string;
  if (o.instrument === "option") {
    const occ = text(rec(o.contract).occ_symbol || o.symbol, 30);
    const verb = o.side === "sell" ? "sold" : "bought";
    what = `opened ${o.chosen ? `a ${o.chosen.replace(/_/g, " ")}` : "an options position"} on ${ticker}: ${verb} ${o.qty ?? 1} ${OCC_RE.test(occ) ? describeOcc(occ) : "contract(s)"}${o.price !== null ? ` at a ${priceText(o.price)} limit` : ""}`;
  } else if (o.instrument === "etf") {
    what = `bought ${o.qty ?? "?"} ${o.symbol} (a leveraged ETF on ${ticker}) at ~${priceText(o.price)} (${money(o.notional)}${pctOfEquity(o.notional, o.equity)})`;
  } else {
    what = `bought ${o.qty ?? "?"} ${ticker} at ~${priceText(o.price)} (${money(o.notional)}${pctOfEquity(o.notional, o.equity)})`;
  }
  const why = whyLine(t, stages);
  const instr = rejected.length ? ` Instead of: ${rejected.map((r) => `${r.instrument} (${r.reason})`).join("; ")}.` : "";
  const sizing = text(t.sizing_reasoning, 220);
  const status = o.mode === "live" ? ` Order ${o.status}${o.detail ? `: ${text(o.detail, 120)}` : ""}.` : "";
  const summary = `${shadow}${what.charAt(0).toUpperCase()}${what.slice(1)}.${why ? ` Why: ${why}.` : ""}${instr}${sizing ? ` Sizing: ${sentence(sizing, 220)}` : ""}${status}`;
  return normalizeDecision({
    ts: new Date(nowMs).toISOString(), ticker, action: "buy", instrument: o.instrument, chosen: o.chosen,
    rejected, stages, summary,
    sizing: { qty: o.qty, price: o.price, notional: o.notional, pct_equity: o.notional !== null && o.equity ? r2((o.notional / o.equity) * 100) : null,
      reasoning: sizing || null, scalars: rec(t.sizing_scalars) },
    source: o.source, signal: signalForTrade(t), side: o.side, symbol: o.symbol, qty: o.qty, price: o.price,
    notional: o.notional, order: { status: o.status, id: o.orderId ?? null }, mode: o.mode,
  });
}

/** A trade dict executeTrades' own gates refused (its gateRejections). */
function buildSkipDecision(trade: unknown, decision: string, reason: string, source = "bot.ts executeTrades", nowMs = Date.now()): DecisionRow | null {
  const t = rec(trade);
  const ticker = text(t.ticker, 24).toUpperCase();
  if (!ticker) return null;
  const option = text(t.trade_type) === "options" || (t.use_options === true && text(t.options_strategy) !== "stock");
  const proposed = option ? `an options trade (${text(t.options_strategy || t.instrument_strategy || "options", 40)})` : "a stock buy";
  const stages: Record<string, Obj | null> = {
    regime: regimeStage(tradeRegimeLabel(t), t.scan_regime, null, undefined, t.risk_budget), calendar: null,
    flows: flowsStage(t.flows), sector: sectorStage(t.sector), selection: selectionStage(t), options: null, tax: null, ...extraStages(t),
  };
  const r = text(reason, 220);
  return normalizeDecision({
    ts: new Date(nowMs).toISOString(), ticker, action: "skip", instrument: option ? "option" : "stock", chosen: "none",
    rejected: [{ instrument: option ? "option" : "stock", reason: `${proposed} was refused because ${ruleText(decision)}${r ? `: ${r}` : ""}` }],
    stages, summary: `Skipped ${ticker}: ${ruleText(decision)}${r ? ` (${r})` : ""}.`,
    source, signal: signalForTrade(t), rule: decision, dedupe: skipDedupeKey(ticker, decision, nowMs),
  });
}

/** A tier dispatcher outcome: SELL_CSP / BUY_PUT (options_execution's result
 *  dict) or the Tier-3 BUY (pass `buy`). Submitted / shadowed -> an entry;
 *  anything else -> a skip with the reason the broker or selector gave. */
function buildTierDecision(action: unknown, result: unknown, buy?: { qty: number; price: number } | null, nowMs = Date.now()): DecisionRow | null {
  const a = rec(action);
  const r = rec(result);
  const ticker = text(a.ticker, 24).toUpperCase();
  const kind = text(a.action, 20);
  if (!ticker || !kind) return null;
  const meta = rec(a.metadata);
  const status = text(r.status, 30) || (buy ? "submitted" : "unknown");
  const shadowed = status === "shadowed";
  const ok = shadowed || ["submitted", "filled", "accepted", "pending_new"].includes(status);
  const isStock = kind === "BUY";
  const chosen = kind === "SELL_CSP" ? "sell_cash_secured_put" : kind === "BUY_PUT" ? "buy_put (tail hedge)" : "buy_stock (tier 3 trend)";
  const signal = `tier ${num(a.tier) ?? "?"} ${kind === "SELL_CSP" ? "CSP harvest" : kind === "BUY_PUT" ? "tail hedge" : "trend capture"}`;
  const regime = regimeStage(meta.regime_label, null, meta.vxx_ratio, undefined, a.risk_budget);
  const occ = text(r.occ_symbol, 30);
  const qty = isStock ? (buy?.qty ?? null) : num(r.qty) ?? 1;
  const price = isStock ? (buy?.price ?? null) : num(r.limit_price);
  const cash = num(r.cash_required);
  const options = isStock ? null : clean({
    strategy: chosen, occ_symbol: occ || null, contract: occ ? describeOcc(occ) : null, side: text(r.side, 8) || (kind === "SELL_CSP" ? "sell" : "buy"),
    qty, limit_price: price, cash_required: cash, target_delta: num(meta.target_delta),
    target_dte: [num(meta.target_dte_min), num(meta.target_dte_max)], reason: text(a.reason, 300),
  }) as Obj;
  const stages: Record<string, Obj | null> = { regime, calendar: null, flows: null, sector: null, selection: null, options, tax: null };
  const why = text(a.reason, 260);
  if (!ok) {
    const detail = text(r.reason || r.detail || r.error || status, 220);
    return normalizeDecision({
      ts: new Date(nowMs).toISOString(), ticker, action: "skip", instrument: isStock ? "stock" : "option", chosen: "none",
      rejected: [{ instrument: isStock ? "stock" : "option", reason: `${chosen.replace(/_/g, " ")} did not go through: ${detail}` }],
      stages, summary: `Skipped ${ticker}: the tier ${num(a.tier) ?? "?"} ${kind} could not be placed (${detail}). It was proposed because: ${why || "no reason given"}.`,
      source: "bot.ts tier dispatcher", signal, rule: "rejected_order", dedupe: skipDedupeKey(ticker, `rejected_order:${kind}`, nowMs),
    });
  }
  const verb = isStock ? `bought ${qty ?? "?"} ${ticker} at ~${priceText(price)}`
    : `${kind === "SELL_CSP" ? "sold" : "bought"} ${qty ?? 1} ${occ ? describeOcc(occ) : `${ticker} option(s)`}${price !== null ? ` at a ${priceText(price)} limit` : ""}${cash ? ` (cash-secured: ${money(cash)} set aside)` : ""}`;
  return normalizeDecision({
    ts: new Date(nowMs).toISOString(), ticker, action: "buy", instrument: isStock ? "stock" : "option", chosen,
    rejected: [], stages,
    summary: `${shadowed ? "Shadow mode, no order placed: " : ""}${verb.charAt(0).toUpperCase()}${verb.slice(1)}. Why: ${why || signal}.${shadowed ? "" : ` Order ${status}.`}`,
    sizing: { qty, price, notional: isStock && qty !== null && price !== null ? r2(qty * price) : cash, size_pct: num(a.size_pct) },
    source: "bot.ts tier dispatcher", signal, side: isStock ? "buy" : (text(r.side, 8) || (kind === "SELL_CSP" ? "sell" : "buy")),
    symbol: isStock ? ticker : occ || ticker, qty, price, notional: cash ?? (isStock && qty !== null && price !== null ? r2(qty * price) : null),
    order: { status, id: text(r.order_id, 60) || null }, mode: shadowed ? "shadow" : "live",
  });
}

export interface ExitInput {
  ticker: string; exitType: string; reason: string; side: string; qty: number; price: number | null;
  pnlPct: number | null; partial?: boolean; entryDate?: string | null; orderId?: string | null;
  status?: string; source: string; instrument?: "stock" | "etf" | "option";
}
/** A stock / ETF exit bot.ts placed (WS exit, scale-out, POS-KILL, TIME-EXIT). */
function buildExitDecision(i: ExitInput, nowMs = Date.now()): DecisionRow | null {
  const ticker = text(i.ticker, 30).toUpperCase();
  if (!ticker) return null;
  const option = i.instrument === "option" || OCC_RE.test(ticker);
  const under = option ? occUnderlying(ticker) : ticker;
  const what = option ? describeOcc(ticker) : `${i.qty} ${ticker}${i.partial ? " (part of the position)" : ""}`;
  const pnl = i.pnlPct !== null && Number.isFinite(i.pnlPct) ? ` The position was ${i.pnlPct >= 0 ? "up" : "down"} ${Math.abs(r1(i.pnlPct))}% at the decision.` : "";
  const held = i.entryDate && Number.isFinite(Date.parse(i.entryDate)) ? ` Held since ${String(i.entryDate).slice(0, 10)}.` : "";
  return normalizeDecision({
    ts: new Date(nowMs).toISOString(), ticker: under, action: i.partial ? "scale_out" : "sell",
    instrument: option ? "option" : (i.instrument || "stock"), chosen: i.exitType, rejected: [],
    stages: { options: option ? { contract: describeOcc(ticker), occ_symbol: ticker, rule: i.exitType } : null },
    summary: `${i.side === "buy" ? "Covered" : "Sold"} ${what}${i.price !== null ? ` at ~${priceText(i.price)}` : ""}: ${stockExitText(i.exitType)}.${pnl}${held} ${sentence(i.reason, 300)}${i.status ? ` Order ${i.status}.` : ""}`.trim(),
    source: i.source, signal: `exit: ${stockExitText(i.exitType)}`, side: i.side, symbol: ticker, qty: i.qty,
    price: i.price, pnl_pct: i.pnlPct, order: { status: i.status || "decided", id: i.orderId ?? null }, mode: "live",
  });
}

/** An entry executeTrades placed (or shadow-logged) for a scan trade dict. */
export const entryDecision = guarded("entryDecision", buildEntryDecision);
/** A trade dict executeTrades' own gates refused (its gateRejections). */
export const skipDecision = guarded("skipDecision", buildSkipDecision);
/** A tier dispatcher outcome (SELL_CSP / BUY_PUT result dict, or the Tier-3 BUY with `buy`). */
export const tierDecision = guarded("tierDecision", buildTierDecision);
/** A stock / ETF / option exit bot.ts placed (WS exit, scale-out, POS-KILL, TIME-EXIT). */
export const exitDecision = guarded("exitDecision", buildExitDecision);
