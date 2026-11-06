// instrumentCounterfactual.ts — the INSTRUMENT COUNTERFACTUAL, Node half
// (2026-11-06; human refocus spec §3 step 7: "instrument choice: stock vs
// options vs both; log why the rejected instrument lost").
// instrument_counterfactual.py documents the ledger, its math and its honesty
// rules, and owns the nightly outcome job. This module:
//   - wouldBeFor(): the would-be entry of the instrument an ENTERED trade the
//     instrument selector decided passed over across the stock/options line —
//     the option alternative of a stock / ETF buy (instrument_selector.
//     option_alternative, carried on the scan trade as `option_alternative`)
//     or the stock alternative of an options entry (the scan's price, size,
//     stop and target). server/decisionJournal.ts puts it on the entry row's
//     `rejected` entry as `would_be`.
//   - appendInstrumentCounterfactuals(): one `decision` row per such entry row
//     to DATA_DIR/voltrade_instrument_counterfactual.jsonl (bot.ts
//     journalDecisions — the same moment the journal row is written).
//   - readInstrumentCounterfactuals() / foldCounterfactuals(): each decision
//     joined to the job's proxy and outcome rows (decisionAttribution.ts
//     aggregates them as byInstrumentChoice).
//   - attachCounterfactualOutcomes(): the settled outcomes on the journal
//     row's would_be when the journal is read.
// LOGGING ONLY: no order, size or gate reads any of this; every writer
// swallows and logs its own errors. Off switch: the env var
// VOLTRADE_INSTRUMENT_COUNTERFACTUAL (any off value), mirroring system_config
// INSTRUMENT_COUNTERFACTUAL.enabled (default on); test_instrument_counterfactual.py
// pins the parity. decisionJournal.ts imports this module, so this one takes
// only TYPES from it (no runtime import cycle).
import fs from "node:fs";
import path from "node:path";
import type { DecisionRow } from "./decisionJournal";

export const INSTRUMENT_CF_FILENAME = "voltrade_instrument_counterfactual.jsonl";
export const INSTRUMENT_CF_ENV = "VOLTRADE_INSTRUMENT_COUNTERFACTUAL";
/** mirrors system_config INSTRUMENT_COUNTERFACTUAL.max_bytes */
export const INSTRUMENT_CF_MAX_BYTES = 5_000_000;
export const INSTRUMENT_CF_ROW_MAX_BYTES = 4_000;
/** mirror system_config INSTRUMENT_COUNTERFACTUAL.proxy_dte_min / proxy_dte_max (the ATM proxy's expiry window) */
export const PROXY_DTE_MIN = 30;
export const PROXY_DTE_MAX = 45;
export const INSTRUMENT_CF_HORIZONS = ["+1d", "+5d", "+20d"] as const;
export type CfHorizon = (typeof INSTRUMENT_CF_HORIZONS)[number];
const OFF_VALUES = new Set(["", "0", "off", "false", "no", "none"]);
const OCC_RE = /^([A-Z.]{1,6})(\d{2})(\d{2})(\d{2})([CP])(\d{8})$/;
type Obj = Record<string, unknown>;

// ── small helpers (decisionJournal.ts's twins; no runtime import from it) ────
function rec(x: unknown): Obj {
  return x && typeof x === "object" && !Array.isArray(x) ? (x as Obj) : {};
}
function num(x: unknown): number | null {
  if (x === null || x === undefined || x === "" || typeof x === "boolean") return null;
  const v = typeof x === "number" ? x : parseFloat(String(x));
  return Number.isFinite(v) ? v : null;
}
function text(x: unknown, n = 300): string {
  const s = x === null || x === undefined ? "" : String(x);
  return s.replace(/\s+/g, " ").trim().slice(0, n);
}
function clean(x: unknown, depth = 0): unknown {
  if (depth > 6 || x === null || x === undefined) return null;
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
function priceText(x: unknown): string {
  const v = num(x);
  return v === null ? "?" : `$${v.toFixed(2)}`;
}
function describeContract(symbol: unknown): string {
  const m = OCC_RE.exec(String(symbol ?? ""));
  return m ? `${m[1]} $${parseInt(m[6], 10) / 1000} ${m[5] === "C" ? "call" : "put"} expiring 20${m[2]}-${m[3]}-${m[4]}` : String(symbol ?? "");
}
const ET_DATE = new Intl.DateTimeFormat("en-CA", { timeZone: "America/New_York", year: "numeric", month: "2-digit", day: "2-digit" });
/** The ET calendar date (YYYY-MM-DD) of an instant: a decision's entry day. */
export function etDate(ms: number): string {
  return ET_DATE.format(new Date(ms));
}

export function cfEnabled(env: Record<string, string | undefined> = process.env): boolean {
  const v = env[INSTRUMENT_CF_ENV];
  return v === undefined ? true : !OFF_VALUES.has(String(v).trim().toLowerCase());
}

// ── the legs of an options entry ─────────────────────────────────────────────
const LEG_KEYS: ReadonlyArray<readonly [string, "buy" | "sell" | null]> = [
  ["occ_symbol", null], ["short_leg", "sell"], ["long_leg", "buy"], ["short_call", "sell"], ["long_call", "buy"],
  ["short_put", "sell"], ["long_put", "buy"], ["call_leg", null], ["put_leg", null],
];
/** The legs ({symbol, side}) of options_execution's contract dict: a short_* key
 *  is sold, a long_* key bought, a single occ_symbol / straddle leg takes the
 *  contract's side (else `fallbackSide`). OCC symbols only, each once. */
export function contractLegs(contract: unknown, fallbackSide: string): Array<{ symbol: string; side: string }> {
  const c = rec(contract);
  const side = text(c.side, 8).toLowerCase() || text(fallbackSide, 8).toLowerCase() || "buy";
  const out: Array<{ symbol: string; side: string }> = [];
  for (const [k, s] of LEG_KEYS) {
    const sym = text(c[k], 30).toUpperCase();
    if (OCC_RE.test(sym) && !out.some((l) => l.symbol === sym)) out.push({ symbol: sym, side: s ?? side });
  }
  return out;
}

// ── the would-be leg (the journal's rejected[].would_be) ─────────────────────
export interface WouldBe { instrument: "option" | "stock"; would_be: Obj }

/** The instrument an entered trade passed over across the stock/options line,
 *  as its would-be entry; null when the instrument selector did not decide
 *  the trade (the options scanner's setups, the tier dispatcher) or the entry
 *  is neither shares nor an option. */
export function wouldBeFor(trade: unknown, entered: { instrument: string }): WouldBe | null {
  const t = rec(trade);
  const scores = rec(t.instrument_scores);
  const ticker = text(t.ticker, 24).toUpperCase();
  if (!ticker || !Object.keys(scores).length || text(t.regime_at_entry) === "OPTIONS_SCANNER") return null;
  const bearish = ["sell", "short"].includes(text(t.side, 8).toLowerCase());
  if (entered.instrument === "stock" || entered.instrument === "etf") {
    const py = rec(t.option_alternative);
    const opt = rec(scores.options);
    const strategy = text(opt.strategy, 60);
    const ruled = /Options ruled out: ([^|]+)/.exec(text(t.instrument_reasoning, 600))?.[1];
    const base: Obj = Object.keys(py).length ? (clean(py) as Obj) : {
      instrument: "option",
      strategy: strategy && !["none", "stock"].includes(strategy) ? strategy : null,
      selector_score: num(opt.score), scored: Object.keys(opt).length > 0,
      chosen_by_selector: text(t.instrument) === "options" || t.use_options === true,
      selector_reason: ruled ? `options were ruled out: ${text(ruled, 200)}` : Object.keys(opt).length ? text(opt.reasoning, 240) : "options were not scored",
      contract: null,
      contract_reason: "the selector's option block was not on the trade, so the ATM proxy is resolved and priced afterwards from Alpaca's contract list and daily option bars",
      source: "the scan trade (no option_alternative from the selector)",
    };
    const right = bearish ? "P" : "C";
    const proxy: Obj = { kind: bearish ? "atm_put" : "atm_call", right, side: "buy", dte_min: PROXY_DTE_MIN, dte_max: PROXY_DTE_MAX,
      underlying: ticker, underlying_price: num(t.price), ...rec(base.proxy) };
    if (num(proxy.underlying_price) === null) proxy.underlying_price = num(t.price);
    return { instrument: "option", would_be: { ...base, instrument: "option", proxy } };
  }
  if (entered.instrument === "option") {
    return { instrument: "stock", would_be: {
      instrument: "stock", symbol: ticker, side: bearish ? "sell" : "buy", entry_price: num(t.price), qty: num(t.shares),
      notional: num(t.position_value), stop: num(t.stop_loss), target: num(t.take_profit), selector_score: num(rec(scores.stock).score),
      source: "the scan's stock plan (price, size, stop and target on the trade)",
      ...(bearish ? { note: "stock shorting is off: a short-stock alternative is hypothetical" } : {}),
    } };
  }
  return null;
}

// ── the ledger's decision row ────────────────────────────────────────────────
function cfSummary(ticker: string, entryDate: string, chosen: Obj, rejected: Obj): string {
  const inst = text(chosen.instrument);
  const strat = text(chosen.strategy, 60).replace(/_/g, " ");
  const what = inst === "option"
    ? `opened ${strat ? `a ${strat}` : "an options position"}${OCC_RE.test(text(chosen.symbol, 30)) ? ` (${describeContract(chosen.symbol)})` : ""}`
    : `bought ${num(chosen.qty) ?? "?"} ${text(chosen.symbol, 12)} at ~${priceText(chosen.price)}`;
  const head = `Chose ${inst === "option" ? "options" : inst === "etf" ? "a leveraged ETF" : "stock"} over ${inst === "option" ? "the stock" : "options"} for ${ticker}: ${what}.`;
  const why = text(rejected.reason, 240);
  let leg: string;
  if (text(rejected.instrument) === "stock") {
    const stop = num(rejected.stop), target = num(rejected.target);
    leg = `The stock leg: ${num(rejected.qty) ?? "?"} ${ticker} at ~${priceText(rejected.entry_price)}${stop !== null ? `, stop ${priceText(stop)}` : ""}${target !== null ? `, target ${priceText(target)}` : ""}.`;
  } else {
    const p = rec(rejected.proxy);
    const own = text(rejected.strategy, 60).replace(/_/g, " ");
    const score = num(rejected.selector_score);
    const idea = own ? ` The selector's own options idea was a ${own}${score !== null ? ` (score ${score})` : ""}; it is named, not priced.` : "";
    leg = text(rec(rejected.contract).symbol)
      ? `The option leg is the selector's own ATM contract ${text(rec(rejected.contract).symbol, 30)}.`
      : `The option leg is a proxy: a bought ATM ${text(p.right) === "P" ? "put" : "call"} expiring ${num(p.dte_min) ?? PROXY_DTE_MIN}-${num(p.dte_max) ?? PROXY_DTE_MAX} days after ${entryDate}, resolved and priced by the nightly job.${idea}`;
  }
  return text(`${head}${why ? ` Why not ${inst === "option" ? "the stock" : "options"}: ${why}.` : ""} ${leg}`, 900);
}

/** The ledger's `decision` row for an entry journal row whose rejected
 *  alternative carries a would_be; null for any other row. */
export function counterfactualRow(row: DecisionRow | null | undefined): Obj | null {
  if (!row || row.action !== "buy" || !Array.isArray(row.rejected)) return null;
  const alt = row.rejected.find((r) => Object.keys(rec(r.would_be)).length > 0);
  const ms = Date.parse(row.ts);
  const inst = row.instrument;
  if (!alt || !Number.isFinite(ms) || (inst !== "stock" && inst !== "etf" && inst !== "option")) return null;
  const opt = rec(row.stages?.options);
  const chosen: Obj = {
    instrument: inst, symbol: text(row.symbol || row.ticker, 30).toUpperCase(), side: text(row.side, 8).toLowerCase() || "buy",
    strategy: text(row.chosen, 80) || null, price: num(row.price), qty: num(row.qty), notional: num(row.notional),
  };
  if (inst === "option") {
    const legs = (Array.isArray(opt.legs) ? opt.legs : []).map(rec)
      .map((l) => ({ symbol: text(l.symbol, 30).toUpperCase(), side: text(l.side, 8).toLowerCase() || "buy" }))
      .filter((l) => OCC_RE.test(l.symbol));
    chosen.legs = legs.length ? legs : OCC_RE.test(text(chosen.symbol)) ? [{ symbol: chosen.symbol, side: chosen.side }] : [];
    chosen.cash_required = num(opt.cash_required);
    chosen.max_loss = num(opt.max_loss);
    chosen.credit = num(opt.credit);
  }
  const rejected: Obj = { ...rec(clean(alt.would_be)), instrument: alt.instrument, reason: alt.reason };
  const entryDate = etDate(ms);
  return {
    v: 1, kind: "decision", id: row.id, ts: row.ts, entry_date: entryDate, ticker: row.ticker, mode: row.mode ?? "live",
    source: row.source ?? null, choice: inst === "option" ? "option_over_stock" : inst === "etf" ? "etf_over_option" : "stock_over_option",
    chosen, rejected, summary: cfSummary(row.ticker, entryDate, chosen, rejected),
  };
}

function fitCf(row: Obj): string {
  let s = JSON.stringify(row);
  if (Buffer.byteLength(s) <= INSTRUMENT_CF_ROW_MAX_BYTES) return s;
  const rj = rec(row.rejected);
  const cut: Obj = { ...row, summary: text(row.summary, 300), truncated: true,
    rejected: { ...rj, reason: text(rj.reason, 160), selector_reason: text(rj.selector_reason, 160), contract_reason: text(rj.contract_reason, 120) } };
  s = JSON.stringify(cut);
  if (Buffer.byteLength(s) <= INSTRUMENT_CF_ROW_MAX_BYTES) return s;
  return JSON.stringify({ ...cut, summary: text(row.summary, 120), chosen: { ...rec(row.chosen), legs: (Array.isArray(rec(row.chosen).legs) ? (rec(row.chosen).legs as unknown[]) : []).slice(0, 4) } });
}

/** Append one `decision` row per entry row that carries a would_be (ONE write,
 *  so the Python job's append never lands inside a line), rotating to <file>.1
 *  above maxBytes. Returns rows written; never throws. */
export function appendInstrumentCounterfactuals(dir: string, rows: ReadonlyArray<DecisionRow | null | undefined>,
  maxBytes = INSTRUMENT_CF_MAX_BYTES, env: Record<string, string | undefined> = process.env): number {
  try {
    if (!cfEnabled(env) || !rows.length) return 0;
    const lines: string[] = [];
    for (const r of rows) {
      const c = counterfactualRow(r);
      if (c) lines.push(fitCf(c) + "\n");
    }
    if (!lines.length) return 0;
    const p = path.join(dir, INSTRUMENT_CF_FILENAME);
    if (fs.existsSync(p) && fs.statSync(p).size > maxBytes) fs.renameSync(p, p + ".1");
    fs.appendFileSync(p, lines.join(""));
    return lines.length;
  } catch (err) {
    console.error("[instrument-counterfactual] rows not written (no decision changed):", err instanceof Error ? err.message : err);
    return 0;
  }
}

// ── reading: decisions joined to the job's proxy and outcome rows ────────────
export interface CfLeg { priced: boolean; pct: number | null; basis: string | null; reason: string | null }
export interface CfOutcome {
  horizon: CfHorizon; endDate: string | null; chosen: CfLeg; rejected: CfLeg;
  diffPp: number | null; winner: string | null; line: string; unpriced: boolean; proxy: boolean;
}
export interface InstrumentChoice {
  id: string; ts: string; ms: number; entryDate: string; ticker: string; mode: string; source: string | null;
  /** stock_over_option | etf_over_option | option_over_stock */
  choice: string;
  chosen: Obj; rejected: Obj; summary: string;
  /** the listed contract the ATM proxy resolved to (null: none yet, or it did not resolve — see proxyReason) */
  proxy: Obj | null; proxyReason: string | null;
  outcomes: Partial<Record<CfHorizon, CfOutcome>>;
}

function legOf(x: unknown): CfLeg {
  const o = rec(x);
  const pct = num(o.pct);
  return { priced: o.priced === true && pct !== null, pct, basis: text(o.basis, 20) || null, reason: text(o.reason, 220) || null };
}
function isHorizon(h: string): h is CfHorizon {
  return (INSTRUMENT_CF_HORIZONS as readonly string[]).includes(h);
}

/** Decisions (oldest first) with their outcomes and proxy, from rows of any
 *  kind in any order (the decision and its outcomes may sit in different
 *  generations). The first row of each kind per id wins. */
export function foldCounterfactuals(rows: readonly unknown[]): InstrumentChoice[] {
  const decisions = new Map<string, InstrumentChoice>();
  const outcomes = new Map<string, Partial<Record<CfHorizon, CfOutcome>>>();
  const proxies = new Map<string, Obj>();
  for (const raw of rows) {
    const r = rec(raw);
    const kind = text(r.kind, 20);
    if (kind === "decision") {
      const id = text(r.id, 40);
      const ms = Date.parse(text(r.ts, 40));
      if (!id || !Number.isFinite(ms) || decisions.has(id)) continue;
      const chosen = rec(r.chosen);
      decisions.set(id, {
        id, ts: text(r.ts, 40), ms, entryDate: text(r.entry_date, 10) || etDate(ms), ticker: text(r.ticker, 24).toUpperCase(),
        mode: text(r.mode, 12) || "live", source: text(r.source, 80) || null, choice: text(r.choice, 40),
        chosen, rejected: rec(r.rejected), summary: text(r.summary, 900), proxy: null, proxyReason: null, outcomes: {},
      });
    } else if (kind === "outcome") {
      const id = text(r.cf_id, 40);
      const h = text(r.horizon, 6);
      if (!id || !isHorizon(h)) continue;
      const m = outcomes.get(id) ?? {};
      if (!m[h]) {
        m[h] = { horizon: h, endDate: text(r.end_date, 10) || null, chosen: legOf(r.chosen), rejected: legOf(r.rejected),
          diffPp: num(r.diff_pp), winner: text(r.winner, 10) || null, line: text(r.line, 600), unpriced: r.unpriced === true, proxy: r.proxy === true };
      }
      outcomes.set(id, m);
    } else if (kind === "proxy") {
      const id = text(r.cf_id, 40);
      if (id && !proxies.has(id)) proxies.set(id, r);
    }
  }
  for (const [id, d] of decisions) {
    d.outcomes = outcomes.get(id) ?? {};
    const p = proxies.get(id);
    if (p) {
      const c = rec(p.contract);
      d.proxy = Object.keys(c).length ? (clean(c) as Obj) : null;
      d.proxyReason = text(p.reason, 240) || null;
    }
  }
  return [...decisions.values()].sort((a, b) => a.ms - b.ms);
}

const readCache: { key: string; choices: InstrumentChoice[] } = { key: "", choices: [] };
function statKey(f: string): string {
  try {
    const s = fs.statSync(f);
    return `${s.size}:${s.mtimeMs}`;
  } catch (err) {
    return (err as NodeJS.ErrnoException)?.code === "ENOENT" ? "-" : `err:${(err as Error)?.message ?? "?"}`;
  }
}

/** Every recorded decision with its outcomes (both generations), re-read only
 *  when either file changed. Malformed lines are skipped (counted in the log). */
export function readInstrumentCounterfactuals(dir: string): InstrumentChoice[] {
  const p = path.join(dir, INSTRUMENT_CF_FILENAME);
  const key = `${p}|${statKey(p + ".1")}|${statKey(p)}`;
  if (readCache.key === key) return readCache.choices;
  const rows: unknown[] = [];
  let malformed = 0;
  for (const f of [p + ".1", p]) {
    let body = "";
    try {
      body = fs.readFileSync(f, "utf8");
    } catch (err) {
      if ((err as NodeJS.ErrnoException)?.code !== "ENOENT") console.error("[instrument-counterfactual] unreadable:", f, err instanceof Error ? err.message : err);
      continue;
    }
    for (const line of body.split("\n")) {
      const s = line.trim();
      if (!s) continue;
      try { rows.push(JSON.parse(s)); } catch { malformed += 1; }
    }
  }
  if (malformed) console.error(`[instrument-counterfactual] ${malformed} malformed line(s) skipped`);
  readCache.key = key;
  readCache.choices = foldCounterfactuals(rows);
  return readCache.choices;
}

/** The journal read path: each entry row whose rejected alternative carries a
 *  would_be gains its settled outcomes ({"+1d": {line, chosen_pct, ...}}) and
 *  the resolved proxy contract. Rows are the reader's own fresh copies. */
export function attachCounterfactualOutcomes<R extends DecisionRow>(rows: R[], dir: string): R[] {
  if (!rows.some((r) => Array.isArray(r.rejected) && r.rejected.some((x) => x.would_be))) return rows;
  const byId = new Map(readInstrumentCounterfactuals(dir).map((c) => [c.id, c] as const));
  for (const r of rows) {
    const c = byId.get(r.id);
    if (!c) continue;
    for (const x of r.rejected) {
      if (!x.would_be) continue;
      const outcomes: Obj = {};
      for (const h of INSTRUMENT_CF_HORIZONS) {
        const o = c.outcomes[h];
        if (o) outcomes[h] = { line: o.line, chosen_pct: o.chosen.pct, rejected_pct: o.rejected.pct, diff_pp: o.diffPp, winner: o.winner, end_date: o.endDate };
      }
      x.would_be = { ...x.would_be, outcomes, ...(c.proxy ? { proxy_contract: c.proxy } : {}), ...(c.proxyReason ? { proxy_reason: c.proxyReason } : {}) };
    }
  }
  return rows;
}
