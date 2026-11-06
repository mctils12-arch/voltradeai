// decisionBackfill.ts — ONE-TIME backfill of the decision journal from the
// history that already exists (2026-11-06; human: "backfill once from
// existing history so the panel is not empty on day one — label backfilled
// rows"). Pure builder + a once-only writer; bot.ts runs it after the sleeve
// ledger is warm.
//
// SOURCES, all already on hand:
//   * the sleeve ledger's closed-lot walk (sleeveLedger.realizedEvents) and
//     the fills it walked — when each position opened, at what price, and,
//     for options, which legs opened together (the structure is INFERRED from
//     those legs and labelled as inferred);
//   * the audit log (30-day retention) — why: the TRADE / ETF-TRADE /
//     INSTRUMENT / OPTIONS-TRADE / tier (T1–T4) rows at the entry, the
//     WS-EXIT / WS-SCALE-OUT / POS-KILL / TIME-EXIT / OPTIONS-EXIT rows at the
//     exit, parsed from the messages bot.ts writes (formats pinned in
//     decisionBackfill.test.ts).
// WHAT IS WRITTEN: an entry ("buy") row per position opened inside the audit
// window or still open now (one row per option structure: the legs that
// opened in the same second on the same underlying), and a "sell" row per
// close inside the audit window whose reason the audit log holds. The floor
// is left out (its P&L is attributed by symbol; it has no "pick"). Nothing is
// guessed: no matching audit row = the row says the reason was not recorded.
//
// The rows go to DATA_DIR/voltrade_decisions_backfill.jsonl, written once
// (its existence is the marker) — never into the live journal, so the live
// file stays in time order. Every row carries backfilled: true and a summary
// that says so.
import fs from "node:fs";
import path from "node:path";
import type { RealizedEvent } from "./sleeveLedger";
import type { FillRow, PositionRow } from "./killSwitchAutoResume";
import { AUDIT_QUERY_MAX_LIMIT } from "./auditRetention";
import { DECISIONS_BACKFILL_FILENAME, describeOcc, money, normalizeDecision, occUnderlying, priceText, regimeStage, type DecisionRow } from "./decisionJournal";

/** Audit types the backfill reads (bot.ts queries each, newest first, ≤ AUDIT_QUERY_MAX_LIMIT). */
export const BACKFILL_AUDIT_TYPES = ["TRADE", "ETF-TRADE", "INSTRUMENT", "OPTIONS-TRADE", "OPTIONS-SCANNER-EXEC",
  "T1", "T2", "T3", "T4", "WS-EXIT", "WS-SCALE-OUT", "POS-KILL", "TIME-EXIT", "OPTIONS-EXIT", "OPTIONS-ROLL"] as const;
/** The audit log's retention (server/auditRetention.ts AUDIT_RETENTION_DAYS): entries older than this have no reason to find. */
export const BACKFILL_WINDOW_DAYS = 30;
/** An audit row explains an entry / exit up to this long before the fill (a DAY limit fills the same session). */
export const BACKFILL_MATCH_MS = 24 * 3_600_000;
/** Legs whose opening fills are this close on one underlying are one structure. */
export const STRUCTURE_GROUP_MS = 2_000;
/** Rows read per audit type (the audit query's own cap; newest first). */
export const BACKFILL_AUDIT_LIMIT = AUDIT_QUERY_MAX_LIMIT;
/** bot.ts runs the backfill this long after boot (the sleeve ledger warms first)... */
export const DECISION_BACKFILL_DELAY_MS = 10 * 60_000;
/** ...and, while the ledger or positions are unreadable, retries this often, at most this many times. */
export const DECISION_BACKFILL_RETRY_MS = 30 * 60_000;
export const DECISION_BACKFILL_MAX_ATTEMPTS = 4;

const OCC_RE = /^([A-Z.]{1,6})(\d{6})([CP])(\d{8})$/;
export interface AuditLite { time: string; type: string; message: string }

// ── option structures ────────────────────────────────────────────────────────
export interface Leg { symbol: string; short: boolean }
/** The structure a set of legs opened together forms, from the legs alone. */
export function inferOptionStructure(legs: readonly Leg[]): string {
  const p = legs.map((l) => ({ m: OCC_RE.exec(l.symbol), short: l.short })).filter((x) => x.m)
    .map((x) => ({ exp: x.m![2], type: x.m![3], strike: parseInt(x.m![4], 10) / 1000, short: x.short }));
  if (!p.length) return "option";
  const puts = p.filter((x) => x.type === "P"), calls = p.filter((x) => x.type === "C");
  const sp = puts.filter((x) => x.short), sc = calls.filter((x) => x.short);
  if (p.length === 1) {
    const x = p[0];
    return x.short ? (x.type === "P" ? "short put (cash-secured)" : "short call") : (x.type === "P" ? "long put" : "long call");
  }
  const exps = new Set(p.map((x) => x.exp));
  if (p.length === 2 && exps.size === 2 && (puts.length === 2 || calls.length === 2)) {
    return p[0].strike === p[1].strike ? "calendar spread" : "diagonal spread";
  }
  if (p.length === 4 && puts.length === 2 && calls.length === 2 && sp.length === 1 && sc.length === 1) return "iron condor";
  if (p.length === 2 && puts.length === 2 && sp.length === 1) {
    const lp = puts.find((x) => !x.short)!;
    return sp[0].strike > lp.strike ? "put credit spread" : "put debit spread";
  }
  if (p.length === 2 && calls.length === 2 && sc.length === 1) {
    const lc = calls.find((x) => !x.short)!;
    return sc[0].strike < lc.strike ? "call credit spread" : "call debit spread";
  }
  if (p.length === 2 && puts.length === 1 && calls.length === 1) {
    const same = puts[0].strike === calls[0].strike;
    if (puts[0].short && calls[0].short) return same ? "short straddle" : "short strangle";
    if (!puts[0].short && !calls[0].short) return same ? "long straddle" : "long strangle";
    return "risk reversal";
  }
  return `${p.length}-leg structure`;
}

// ── when did each position open? ─────────────────────────────────────────────
export interface Opening { symbol: string; t: number; qty: number; price: number | null; short: boolean; open: boolean }

/** Openings of the lots the closed-lot walk saw (keyed by symbol + open
 *  time, closed quantities summed) plus the still-open positions' opening
 *  fills (the last flat -> non-flat crossing, walked like realizedEvents). */
export function openings(events: readonly RealizedEvent[], fills: readonly FillRow[], positions: readonly PositionRow[]): Opening[] {
  const byKey = new Map<string, Opening>();
  for (const e of events) {
    if (e.openedAt === null) continue;
    const key = `${e.symbol}|${e.openedAt}`;
    const mult = OCC_RE.test(e.symbol) ? 100 : 1;
    const o = byKey.get(key);
    if (o) { o.qty += e.qty; continue; }
    byKey.set(key, { symbol: e.symbol, t: e.openedAt, qty: e.qty, price: e.qty > 0 ? e.basis / (e.qty * mult) : null, short: e.closeDir === 1, open: false });
  }
  const bySymbol = new Map<string, FillRow[]>();
  for (const f of fills) { const a = bySymbol.get(f.symbol); if (a) a.push(f); else bySymbol.set(f.symbol, [f]); }
  for (const p of positions) {
    const list = (bySymbol.get(p.symbol) ?? []).slice().sort((a, b) => a.t - b.t);
    let pos = p.qty - list.reduce((s, f) => s + f.dir * f.qty, 0);
    let openedAt: number | null = null;
    let price: number | null = null;
    for (const f of list) {
      const before = pos;
      pos += f.dir * f.qty;
      if (Math.abs(pos) < 1e-9) pos = 0;
      if ((before === 0 || Math.sign(before) !== Math.sign(pos)) && pos !== 0) { openedAt = f.t; price = f.price; }
    }
    if (openedAt === null) continue;
    const key = `${p.symbol}|${openedAt}`;
    if (!byKey.has(key)) byKey.set(key, { symbol: p.symbol, t: openedAt, qty: Math.abs(p.qty), price: price ?? p.avgEntry, short: p.qty < 0, open: true });
    else byKey.get(key)!.open = true;
  }
  return [...byKey.values()].sort((a, b) => a.t - b.t);
}

// ── audit message parsers (the formats bot.ts writes) ────────────────────────
export interface EntryWhy { kind: string; ticker: string; t: number; score: number | null; reason: string; strategy?: string; occ?: string; regime?: string; vxx?: number | null; rejected?: Array<{ instrument: string; reason: string }> }

/** TRADE: "LIMIT BUY 12 AMD @ ~$150.2 | Score: 71.3 | Rules: 68 ML: 62 Blend: 71.3 | $1802 (1.9%) | Sizing: ..." */
export function parseTradeAudit(msg: string): { ticker: string; score: number | null; sizing: string | null; rules: string | null } | null {
  const m = /^\w+ (?:BUY|SELL) \d+(?:\.\d+)? ([A-Z.]{1,10}) @ ~\$[\d.]+ \| Score: ([\d.]+)(?: \| Rules: (\S+) ML: (\S+) Blend: \S+)? \| \$[\d.]+ \([\d.]+%\)(?: \| Sizing: (.*))?$/.exec(msg);
  return m ? { ticker: m[1], score: parseFloat(m[2]), rules: m[3] ? `rules ${m[3]}, ML ${m[4]}` : null, sizing: m[5] ?? null } : null;
}
/** INSTRUMENT: "AMD → STOCK chosen | scores: stock=63 etf=41 options=N/A | reasoning" */
export function parseInstrumentAudit(msg: string): { ticker: string; chosen: string; scores: Record<string, string>; reasoning: string } | null {
  const m = /^([A-Z.]{1,10}) → (\w+) chosen \| scores: stock=(\S+) etf=(\S+) options=(\S+) \| ?(.*)$/.exec(msg);
  return m ? { ticker: m[1], chosen: m[2].toLowerCase(), scores: { stock: m[3], etf: m[4], options: m[5] }, reasoning: m[6] } : null;
}
/** T1/T2: "SELL_CSP TLT | T1 CSP harvest (regime=BULL, vxx=0.95, DD=-3.1%, size=85%)"; T3: "BUY 10 SPY @ ~$580.10 | reason"; T4: "BUY_PUT SPY hedge | reason" */
export function parseTierAudit(type: string, msg: string): { ticker: string; reason: string; kind: string } | null {
  let m = /^(SELL_CSP|BUY_PUT) ([A-Z.]{1,10})(?: hedge)? \| (.*)$/.exec(msg);
  if (m) return { kind: m[1], ticker: m[2], reason: m[3] };
  m = /^BUY \d+ ([A-Z.]{1,10}) @ ~\$[\d.]+ \| (.*)$/.exec(msg);
  return m && type === "T3" ? { kind: "BUY", ticker: m[1], reason: m[2] } : null;
}
/** OPTIONS-TRADE: "SELL_CASH_SECURED_PUT TLT | TLT261219P00085000 | reasoning"; OPTIONS-SCANNER-EXEC: "TLT | strategy=x | setup=y | result=z" */
export function parseOptionsAudit(type: string, msg: string): { ticker: string; strategy: string; occ?: string; setup?: string; reason: string; ok: boolean } | null {
  if (type === "OPTIONS-TRADE") {
    const m = /^(\S*) ([A-Z.]{1,10}) \| (\S*) \| ?(.*)$/.exec(msg);
    return m ? { ticker: m[2], strategy: m[1].toLowerCase(), occ: m[3] || undefined, reason: m[4], ok: true } : null;
  }
  const m = /^([A-Z.]{1,10}) \| strategy=(\S*) \| setup=(\S*) \| result=(\S*)$/.exec(msg);
  return m ? { ticker: m[1], strategy: m[2], setup: m[3], reason: `options scanner setup ${m[3]}`, ok: ["submitted", "filled", "accepted", "pending_new"].includes(m[4]) } : null;
}
/** The exit rows -> {ticker, type, reason}. */
export function parseExitAudit(type: string, msg: string): { ticker: string; exitType: string; reason: string } | null {
  let m: RegExpExecArray | null;
  switch (type) {
    case "WS-EXIT":
      m = /^(.*) \| \w+ (?:sell|buy) \d+(?:\.\d+)? ([A-Z.]{1,10}) @ \$[\d.]+$/.exec(msg);
      if (!m) return null;
      return { ticker: m[2], reason: m[1], exitType: /TAKE PROFIT/i.test(m[1]) ? "take_profit" : /TIME STOP/i.test(m[1]) ? "time_stop" : /ETF MAX HOLD/i.test(m[1]) ? "etf_max_hold" : /TRAIL/i.test(m[1]) ? "trailing_stop" : "stop_loss" };
    case "WS-SCALE-OUT":
      m = /^(.*) \| \w+ (?:sell|buy) \d+ of \d+ ([A-Z.]{1,10}) @ \$[\d.]+/.exec(msg);
      return m ? { ticker: m[2], reason: m[1], exitType: "scale_out" } : null;
    case "POS-KILL":
      m = /^([A-Z0-9.]{1,25}): (position down .*)$/.exec(msg);
      return m ? { ticker: m[1], reason: m[2], exitType: "position_kill" } : null;
    case "TIME-EXIT":
      m = /^([A-Z.]{1,10}): (.*)$/.exec(msg);
      return m ? { ticker: m[1], reason: m[2], exitType: "time_exit" } : null;
    case "OPTIONS-EXIT":
    case "OPTIONS-ROLL":
      // a deferred close ("order deferred") sent nothing: it is not an exit
      m = /^([A-Z.]{1,10}) ([^:]+): (.*)$/.exec(msg);
      return m && !/^… /.test(msg) && !/ · order deferred\b/.test(msg)
        ? { ticker: m[1], reason: m[3].split(" · ")[0], exitType: type === "OPTIONS-ROLL" ? "roll" : "options_exit" } : null;
    default:
      return null;
  }
}

function regimeFromReason(reason: string): { regime: string | null; vxx: number | null } {
  const r = /regime=([A-Z_]+)/.exec(reason);
  const v = /vxx=([\d.]+)/.exec(reason);
  return { regime: r ? r[1] : null, vxx: v ? parseFloat(v[1]) : null };
}

/** Every entry explanation the audit rows carry, by underlying ticker. */
export function entryReasons(rows: readonly AuditLite[]): Map<string, EntryWhy[]> {
  const out = new Map<string, EntryWhy[]>();
  const add = (w: EntryWhy) => { const a = out.get(w.ticker); if (a) a.push(w); else out.set(w.ticker, [w]); };
  const instr = new Map<string, Array<{ t: number; v: NonNullable<ReturnType<typeof parseInstrumentAudit>> }>>();
  for (const r of rows) {
    if (r.type !== "INSTRUMENT") continue;
    const v = parseInstrumentAudit(r.message); const t = Date.parse(r.time);
    if (v && Number.isFinite(t)) { const a = instr.get(v.ticker) ?? []; a.push({ t, v }); instr.set(v.ticker, a); }
  }
  for (const r of rows) {
    const t = Date.parse(r.time);
    if (!Number.isFinite(t)) continue;
    if (r.type === "TRADE") {
      const v = parseTradeAudit(r.message);
      if (!v) continue;
      const iv = (instr.get(v.ticker) ?? []).filter((x) => x.t <= t && t - x.t < 120_000).sort((a, b) => b.t - a.t)[0]?.v;
      const rejected = iv ? Object.entries(iv.scores).filter(([k]) => k !== iv.chosen)
        .map(([k, s]) => ({ instrument: k === "options" ? "option" : k, reason: s === "N/A" ? "not scored" : `scored ${s} vs ${iv.scores[iv.chosen] ?? "?"} for the ${iv.chosen}` })) : undefined;
      add({ kind: "stock", ticker: v.ticker, t, score: v.score, reason: [v.rules, v.sizing ? `sizing: ${v.sizing}` : null, iv?.reasoning].filter(Boolean).join("; "), rejected });
    } else if (r.type === "ETF-TRADE") {
      const m = /^\w+ \d+ ([A-Z.]{1,10}) @ ~\$[\d.]+ \(\$[\d.]+, leveraged ETF for ([A-Z.]{1,10})[^)]*\) \| Score: ([\d.]+) \| ?(.*)$/.exec(r.message);
      if (m) add({ kind: "etf", ticker: m[2], t, score: parseFloat(m[3]), reason: `bought ${m[1]} (leveraged ETF): ${m[4]}`, strategy: m[1] });
    } else if (r.type === "OPTIONS-TRADE" || r.type === "OPTIONS-SCANNER-EXEC") {
      const v = parseOptionsAudit(r.type, r.message);
      // the scanner path writes OPTIONS-SCANNER-EXEC (with its setup) AND an OPTIONS-TRADE twin
      // ("Scanner direct execution"): the twin adds nothing, so it is skipped
      if (v && v.ok && !(r.type === "OPTIONS-TRADE" && /^Scanner direct execution/.test(v.reason))) {
        add({ kind: r.type === "OPTIONS-TRADE" ? "options" : "options scanner", ticker: v.ticker, t, score: null, reason: v.reason, strategy: v.setup ? `${v.strategy} (${v.setup})` : v.strategy, occ: v.occ });
      }
    } else if (r.type === "T1" || r.type === "T2" || r.type === "T3" || r.type === "T4") {
      const v = parseTierAudit(r.type, r.message);
      if (v) { const rg = regimeFromReason(v.reason); add({ kind: `tier ${r.type.slice(1)} ${v.kind}`, ticker: v.ticker, t, score: null, reason: v.reason, strategy: v.kind, regime: rg.regime ?? undefined, vxx: rg.vxx }); }
    }
  }
  return out;
}

/** Every exit explanation, by underlying ticker. */
export function exitReasons(rows: readonly AuditLite[]): Map<string, Array<{ t: number; exitType: string; reason: string; type: string }>> {
  const out = new Map<string, Array<{ t: number; exitType: string; reason: string; type: string }>>();
  for (const r of rows) {
    const v = parseExitAudit(r.type, r.message);
    const t = Date.parse(r.time);
    if (!v || !Number.isFinite(t)) continue;
    const k = occUnderlying(v.ticker);
    const a = out.get(k) ?? [];
    a.push({ t, exitType: v.exitType, reason: v.reason, type: r.type });
    out.set(k, a);
  }
  return out;
}

/** The signal label a live row of the same path carries (server/decisionJournal.ts
 *  signalForTrade / tierDecision), so a backfilled lot lands in the same bucket. */
export function backfillSignal(w: EntryWhy): string {
  if (w.kind.startsWith("tier ")) {
    const [, tier, kind] = w.kind.split(" ");
    return `tier ${tier} ${kind === "SELL_CSP" ? "CSP harvest" : kind === "BUY_PUT" ? "tail hedge" : "trend capture"}`;
  }
  if (w.kind === "options scanner") return `options scanner: ${w.strategy?.match(/\(([^)]+)\)$/)?.[1] ?? w.strategy ?? "setup"}`;
  if (w.kind === "options") return "deep_score scan (options)";
  return "deep_score scan";
}

function latestBefore<T extends { t: number }>(list: readonly T[] | undefined, t: number): T | null {
  let best: T | null = null;
  for (const x of list ?? []) if (x.t <= t + 60_000 && t - x.t <= BACKFILL_MATCH_MS && (!best || x.t > best.t)) best = x;
  return best;
}

export interface BackfillInputs {
  events: readonly RealizedEvent[]; fills: readonly FillRow[]; positions: readonly PositionRow[];
  audit: readonly AuditLite[]; floorTickers: ReadonlySet<string>; nowMs: number;
}
export interface BackfillResult { rows: DecisionRow[]; entries: number; matchedEntries: number; exits: number }

const BACKFILL_NOTE = "Backfilled on the journal's first run from the broker's fill history";

/** Pure: the rows the one-time backfill writes (see the header). */
export function buildBackfill(i: BackfillInputs): BackfillResult {
  const since = i.nowMs - BACKFILL_WINDOW_DAYS * 86_400_000;
  const why = entryReasons(i.audit);
  const exits = exitReasons(i.audit);
  const rows: DecisionRow[] = [];
  const push = (r: DecisionRow | null) => { if (r) rows.push(r); };
  let entries = 0, matchedEntries = 0, exitRows = 0;
  // entries: one row per stock lot / option structure
  const opens = openings(i.events, i.fills, i.positions).filter((o) => !i.floorTickers.has(o.symbol.toUpperCase()) && (o.open || o.t >= since));
  const groups = new Map<string, Opening[]>();
  for (const o of opens) {
    const opt = OCC_RE.test(o.symbol);
    const key = opt ? `${occUnderlying(o.symbol)}|${Math.round(o.t / STRUCTURE_GROUP_MS)}` : `${o.symbol}|${o.t}`;
    const g = groups.get(key); if (g) g.push(o); else groups.set(key, [o]);
  }
  for (const legs of groups.values()) {
    const first = legs[0];
    const opt = OCC_RE.test(first.symbol);
    const ticker = occUnderlying(first.symbol);
    const t = Math.min(...legs.map((l) => l.t));
    const w = latestBefore(why.get(ticker), t);
    const structure = opt ? inferOptionStructure(legs.map((l) => ({ symbol: l.symbol, short: l.short }))) : null;
    const what = opt
      ? `${structure} on ${ticker} (structure inferred from the legs that opened together): ${legs.map((l) => `${l.short ? "sold" : "bought"} ${l.qty} ${describeOcc(l.symbol)} at ${priceText(l.price)}`).join("; ")}`
      : `${first.short ? "sold short" : "bought"} ${first.qty} ${first.symbol} at ${priceText(first.price)}`;
    const reason = w ? `${w.kind}${w.score !== null ? ` (score ${w.score})` : ""}: ${w.reason}` : "the reason was not recorded (no matching audit row in the 30-day window)";
    if (w) matchedEntries++;
    entries++;
    push(normalizeDecision({
      ts: new Date(t).toISOString(), ticker, action: "buy",
      instrument: opt ? "option" : (w?.kind === "etf" ? "etf" : "stock"),
      chosen: opt ? (w?.strategy || structure || "option") : (w?.kind === "etf" ? `buy_etf (${w.strategy})` : "buy_stock"),
      rejected: w?.rejected ?? [],
      stages: {
        regime: w?.regime || w?.vxx ? regimeStage(w?.regime, null, w?.vxx ?? null) : null,
        selection: w && w.score !== null ? { score: w.score, threshold: null, drivers: [], dominant: null, reasons: [w.reason].filter(Boolean) } : null,
        options: opt ? { strategy: structure, inferred: true, legs: legs.map((l) => ({ symbol: l.symbol, side: l.short ? "sell" : "buy", qty: l.qty, price: l.price })) } : null,
      },
      sizing: { qty: opt ? null : first.qty, price: opt ? null : first.price, notional: opt ? null : (first.price !== null ? Math.round(first.qty * first.price * 100) / 100 : null) },
      summary: `${BACKFILL_NOTE}: ${what.charAt(0).toUpperCase()}${what.slice(1)}. Why: ${reason}.${legs.some((l) => l.open) ? " Still open." : ""}`,
      source: "backfill.lots+audit", signal: w ? backfillSignal(w) : "not recorded (before the decision journal)",
      side: opt ? undefined : (first.short ? "sell" : "buy"), symbol: opt ? (legs.length === 1 ? first.symbol : undefined) : first.symbol,
      backfilled: true,
    }));
  }
  // exits: a close inside the audit window whose reason the audit log holds
  const seen = new Set<string>();
  for (const e of i.events) {
    if (e.t < since || i.floorTickers.has(e.symbol.toUpperCase())) continue;
    const ticker = occUnderlying(e.symbol);
    const x = latestBefore(exits.get(ticker), e.t);
    if (!x) continue;
    const opt = OCC_RE.test(e.symbol);
    const key = `${e.symbol}|${e.t}`;
    if (seen.has(key)) continue;
    seen.add(key);
    exitRows++;
    const mult = opt ? 100 : 1;
    const exitPx = e.qty > 0 ? e.basis / (e.qty * mult) + (e.pnl / (e.qty * mult)) * (e.closeDir === -1 ? 1 : -1) : null;
    push(normalizeDecision({
      ts: new Date(e.t).toISOString(), ticker, action: x.exitType === "scale_out" ? "scale_out" : x.exitType === "roll" ? "rotate" : "sell",
      instrument: opt ? "option" : "stock", chosen: x.exitType, rejected: [],
      stages: { options: opt ? { contract: describeOcc(e.symbol), occ_symbol: e.symbol } : null },
      summary: `${BACKFILL_NOTE}: closed ${opt ? describeOcc(e.symbol) : `${e.qty} ${e.symbol}`}${exitPx !== null ? ` at ${priceText(exitPx)}` : ""} for ${money(e.pnl)}${opt ? " on this leg" : ""}. Why (audit ${x.type}): ${x.reason}.`,
      source: "backfill.lots+audit", signal: `exit: ${x.exitType.replace(/_/g, " ")}`, symbol: e.symbol,
      side: e.closeDir === -1 ? "sell" : "buy", qty: e.qty, price: exitPx, backfilled: true,
    }));
  }
  rows.sort((a, b) => Date.parse(a.ts) - Date.parse(b.ts));
  return { rows, entries, matchedEntries, exits: exitRows };
}

/** Has the one-time backfill run in `dir`? (the file's existence is the marker) */
export function backfillDone(dir: string): boolean {
  return fs.existsSync(path.join(dir, DECISIONS_BACKFILL_FILENAME));
}

/** Write the backfill file ONCE (tmp + rename). Returns rows written, or null
 *  when it already existed or the write failed (logged). */
export function writeBackfill(dir: string, rows: readonly DecisionRow[]): number | null {
  const p = path.join(dir, DECISIONS_BACKFILL_FILENAME);
  if (fs.existsSync(p)) return null;
  try {
    const tmp = `${p}.tmp`;
    fs.writeFileSync(tmp, rows.map((r) => JSON.stringify(r)).join("\n") + (rows.length ? "\n" : ""));
    fs.renameSync(tmp, p);
    return rows.length;
  } catch (err) {
    console.error("[decision-backfill] not written:", err instanceof Error ? err.message : err);
    return null;
  }
}

/** One audit line for the run. */
export function backfillAuditLine(r: BackfillResult, written: number | null): string {
  return written === null
    ? "decision journal backfill not written (already present or the write failed)"
    : `decision journal backfill: ${written} rows (${r.entries} entries, ${r.matchedEntries} with an audit reason; ${r.exits} exits with an audit reason) — labelled backfilled`;
}
