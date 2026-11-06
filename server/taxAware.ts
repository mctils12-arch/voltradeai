// taxAware.ts — GET /api/bot/tax (refocus spec 2026-11-05 §6; [RULE-REVIEW]
// 2026-11-06). The tax filter itself is Python (tax_filter.py): before every
// new entry it compares the trade's after-tax expected return with holding
// SPY for the same days after long-term tax, skips the ones that do not beat
// it by a small margin, and appends every decision to
// DATA_DIR/voltrade_tax_decisions.jsonl. This module only READS that log and
// shapes the route body (the UI contract in the build preamble):
//   {rates:{shortTerm, longTerm}, decisions:[{ts, ticker, side, allowed,
//    expectedPct, afterTaxPct, spyAfterTaxPct, holdingDays, reason, ...}],
//    harvest:[...], ltApproaching:[...]}
// plus a summary over the decisions, the live broker expectancy per sleeve
// next to the sizer's estimate (the filter prices trades with the Kelly
// sizer's stats; when live disagrees, the page must say so) and notes.
// Every number is an ESTIMATE at assumed rates; the paper account pays no tax.
// Pure except readTaxDecisions (one bounded file read).
import fs from "node:fs";
import path from "node:path";
import { DEFAULT_TAX_RATES, taxLedger } from "./taxLedger";
import type { EdgeLedger, EdgeRow } from "./edgeLedger";
import type { FillRow, PositionRow } from "./killSwitchAutoResume";
import type { RealizedEvent } from "./sleeveLedger";

export const TAX_DECISIONS_FILE = "voltrade_tax_decisions.jsonl";
export const TAX_DECISIONS_DEFAULT_LIMIT = 100;
export const TAX_DECISIONS_MAX_LIMIT = 500;
/** Bytes read from the end of the log — the newest rows (the writer rotates at 4 MB). */
export const TAX_DECISIONS_TAIL_BYTES = 1024 * 1024;

/** One decision as the route serves it. The first nine fields are the UI contract. */
export interface TaxDecision {
  ts: string;
  ticker: string;
  side: string;
  allowed: boolean;
  expectedPct: number | null;
  afterTaxPct: number | null;
  spyAfterTaxPct: number | null;
  holdingDays: number | null;
  reason: string;
  kind: string;
  instrument: string | null;
  source: string | null;
  /** the verdict before the off switch: false = the filter would skip it */
  wouldAllow: boolean;
  /** false while TAX_FILTER.ENABLED is off — every row is then allowed */
  enforced: boolean;
  /** false = the filter could not price it (no measured stats, a hedge, an error) */
  priced: boolean;
  costPct: number | null;
  edgePp: number | null;
  marginPp: number | null;
  bucket: string | null;
  strategy: string | null;
}

/** One losing position in the latest harvest review. The first five fields are the UI contract;
 *  action: "harvested" | "harvest" | "recommend" | "failed" | "none". washSaleUntil: the last day a buy-back would wash the loss. */
export interface HarvestRow { ticker: string; unrealized: number; replacement: string | null; washSaleUntil: string | null; reason: string; action?: string; taxSaving?: number | null }
export interface LtApproachingRow { ticker: string; daysToLongTerm: number; gain: number; reason: string }

export interface LiveExpectancy { n: number; winRatePct: number | null; expectancyPct: number | null; lastCloseAt: string | null }

export interface TaxBody {
  rates: { shortTerm: number; longTerm: number };
  decisions: TaxDecision[];
  harvest: HarvestRow[];
  ltApproaching: LtApproachingRow[];
  summary: { rows: number; allowed: number; skipped: number; wouldSkip: number; notPriced: number; since: string | null; enforced: boolean | null };
  /** the Python assumption the newest row was computed with (SPY's annual return, %) */
  spyAssumptionAnnualPct: number | null;
  /** the broker's own expectancy per closed lot (server/edgeLedger.ts), beside the sizer's estimate */
  liveExpectancy: { options: LiveExpectancy; stocks: LiveExpectancy; basis: string } | null;
  notes: string[];
}

const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const str = (v: unknown): string | null => (typeof v === "string" && v ? v : null);

/** One log line -> a route row; null for a torn or foreign line. */
export function taxDecisionRow(raw: unknown): TaxDecision | null {
  if (!raw || typeof raw !== "object") return null;
  const r = raw as Record<string, unknown>;
  const ts = str(r.ts), ticker = str(r.ticker);
  if (!ts || !ticker || typeof r.allowed !== "boolean") return null;
  return {
    ts, ticker, side: str(r.side) ?? "buy", allowed: r.allowed,
    expectedPct: num(r.expectedPct), afterTaxPct: num(r.afterTaxPct), spyAfterTaxPct: num(r.spyAfterTaxPct),
    holdingDays: num(r.holdingDays), reason: str(r.reason) ?? "",
    kind: str(r.kind) ?? "entry", instrument: str(r.instrument), source: str(r.source),
    wouldAllow: typeof r.wouldAllow === "boolean" ? r.wouldAllow : r.allowed,
    enforced: r.enforced !== false, priced: r.priced !== false,
    costPct: num(r.costPct), edgePp: num(r.edgePp), marginPp: num(r.marginPp),
    bucket: str(r.bucket), strategy: str(r.strategy),
  };
}

/** Parse JSONL text (oldest first) into rows, newest first, at most `limit`. */
export function parseTaxDecisions(text: string, limit: number): { rows: TaxDecision[]; spyAssumption: number | null } {
  const lines = text.split("\n");
  const rows: TaxDecision[] = [];
  let spyAssumption: number | null = null;
  for (let i = lines.length - 1; i >= 0 && rows.length < limit; i--) {
    const line = lines[i].trim();
    if (!line) continue;
    let parsed: unknown;
    try { parsed = JSON.parse(line); } catch { continue; } // a torn tail line or the cut first line of the window
    const row = taxDecisionRow(parsed);
    if (!row) continue;
    if (spyAssumption === null) spyAssumption = num((parsed as Record<string, unknown>).spyAssumptionAnnualPct);
    rows.push(row);
  }
  return { rows, spyAssumption };
}

/** ?limit -> 1..TAX_DECISIONS_MAX_LIMIT, default TAX_DECISIONS_DEFAULT_LIMIT. */
export function taxLimit(q: unknown): number {
  const n = Math.floor(Number(q));
  return Number.isFinite(n) && n >= 1 ? Math.min(n, TAX_DECISIONS_MAX_LIMIT) : TAX_DECISIONS_DEFAULT_LIMIT;
}

/** The newest rows of the decision log in `dir` (the last TAX_DECISIONS_TAIL_BYTES only). */
export function readTaxDecisions(dir: string, limit: number): { rows: TaxDecision[]; spyAssumption: number | null; error: string | null } {
  const p = path.join(dir, TAX_DECISIONS_FILE);
  let fd: number | null = null;
  try {
    fd = fs.openSync(p, "r");
    const size = fs.fstatSync(fd).size;
    const len = Math.min(size, TAX_DECISIONS_TAIL_BYTES);
    const buf = Buffer.alloc(len);
    fs.readSync(fd, buf, 0, len, size - len);
    return { ...parseTaxDecisions(buf.toString("utf8"), limit), error: null };
  } catch (err) {
    const code = (err as NodeJS.ErrnoException)?.code;
    // no file yet = no decision made yet; anything else is reported, not hidden
    return { rows: [], spyAssumption: null, error: code === "ENOENT" ? null : String(err instanceof Error ? err.message : err).slice(0, 200) };
  } finally {
    if (fd !== null) fs.closeSync(fd);
  }
}

function liveRow(r: EdgeRow | undefined): LiveExpectancy {
  return { n: r?.n ?? 0, winRatePct: r?.winRate ?? null, expectancyPct: r?.expectancyPct ?? null, lastCloseAt: r?.lastCloseAt ?? null };
}

export interface TaxBodyInput {
  rows: TaxDecision[];
  spyAssumption: number | null;
  readError: string | null;
  edge: EdgeLedger | null;
  harvest?: HarvestRow[];
  ltApproaching?: LtApproachingRow[];
  extraNotes?: string[];
}

/** Pure: the route body. */
export function taxBody(input: TaxBodyInput): TaxBody {
  const { rows } = input;
  const skipped = rows.filter((r) => !r.allowed).length;
  const notes = [
    "estimates at assumed rates (24 % short-term, 15 % long-term federal, no state, no NIIT); the paper account pays no tax",
    "each entry's expected return is the Kelly sizer's own per-trade expectancy for its bucket (win rate x average win - loss rate x average loss) over the mean holding period of the sample it was measured on; the composite score is not used (the shadow study found it anti-predictive)",
    "SPY leg: an assumed long-run return (TAX_FILTER.SPY_EXPECTED_RETURN_ANNUAL_PCT, not a forecast) for the same days, taxed at the long-term rate because buy-and-hold defers its tax",
    "costs: the repo's round-trip model is subtracted on top of stats that are already net of their own trades' costs, so the filter errs strict",
    "rows are one per ticker, instrument and verdict per day; skipped stock entries also appear as SIZE-BLOCK audit rows, skipped option opens as T-FAIL / OPTIONS-SKIP rows",
  ];
  if (input.edge) notes.push("liveExpectancy: the broker's realized expectancy per closed lot (% of the premium for options, % of cost for stocks) — when it disagrees with the sizer's estimate the filter still uses the sizer's; that disagreement is the sizer's open question");
  if (input.readError) notes.push(`decision log unreadable: ${input.readError}`);
  if (!rows.length && !input.readError) notes.push("no tax decision logged yet");
  for (const n of input.extraNotes ?? []) notes.push(n);
  return {
    rates: { shortTerm: DEFAULT_TAX_RATES.shortTerm, longTerm: DEFAULT_TAX_RATES.longTerm },
    decisions: rows,
    harvest: input.harvest ?? [],
    ltApproaching: input.ltApproaching ?? [],
    summary: {
      rows: rows.length, allowed: rows.length - skipped, skipped,
      wouldSkip: rows.filter((r) => !r.wouldAllow).length,
      notPriced: rows.filter((r) => !r.priced).length,
      since: rows.length ? rows[rows.length - 1].ts : null,
      enforced: rows.length ? rows[0].enforced : null,
    },
    spyAssumptionAnnualPct: input.spyAssumption,
    liveExpectancy: input.edge ? { options: liveRow(input.edge.options), stocks: liveRow(input.edge.stocks), basis: "per closed lot at the broker (server/edgeLedger.ts)" } : null,
    notes,
  };
}

// ── holding winners past one year (2026-11-06) ──────────────────────────────
// Python's manage_positions decides (tax_filter.lt_deferral: inside 45 days of
// the long-term date, a winner's tax saving vs its expected give-back) and
// writes the verdict into voltrade_stop_state.json (tax_defer_until = the
// first long-term day, tax_defer_reason). bot.ts reads it on every position
// sync and holds ONLY the discretionary profit-taking exits — WS scale-out,
// WS take-profit, WS time stop — never a stop, a trailing or breakeven stop,
// POS-KILL, the 14-day TIME-EXIT or any risk exit. One decision, made once,
// obeyed by both layers.

export interface TaxHold { until: string; reason: string }

/** The hold a stop-state entry carries, when it still applies `now` (UTC date before `until`). */
export function taxHoldOf(ps: unknown, now: Date): TaxHold | null {
  if (!ps || typeof ps !== "object") return null;
  const r = ps as Record<string, unknown>;
  const until = typeof r.tax_defer_until === "string" ? r.tax_defer_until.slice(0, 10) : "";
  if (!/^\d{4}-\d{2}-\d{2}$/.test(until)) return null;
  if (now.toISOString().slice(0, 10) >= until) return null; // long-term from that day: nothing left to wait for
  return { until, reason: typeof r.tax_defer_reason === "string" && r.tax_defer_reason ? r.tax_defer_reason : `tax hold until ${until}` };
}

/** One TAX-DEFER audit row per ticker and exit kind per UTC day (the WS tick and the sync both run often). */
export class TaxDeferAudit {
  private readonly seen = new Map<string, string>();
  due(ticker: string, kind: string, now: Date = new Date()): boolean {
    const day = now.toISOString().slice(0, 10);
    const key = `${ticker.toUpperCase()}|${kind}`;
    if (this.seen.get(key) === day) return false;
    for (const [k, d] of Array.from(this.seen)) if (d !== day) this.seen.delete(k);
    this.seen.set(key, day);
    return true;
  }
}

export function taxDeferLine(ticker: string, exit: string, hold: TaxHold): string {
  return `${ticker.toUpperCase()}: ${exit} held for tax until ${hold.until} (stops still run) — ${hold.reason}`.slice(0, 400);
}

/** tax_filter.py TAX_FILTER.LT_DEFER_WINDOW_DAYS (system_config) — MUST mirror it; taxAware.test.ts pins the parity. */
export const TAX_LT_WINDOW_DAYS = 45;
const DAY_MS = 86_400_000;

export interface OpenLot { qty: number; price: number; t: number }

/** FIFO open long lots of one symbol from its fills (any order): buys add a
 *  lot, sells consume the oldest first (the broker's default lot relief). A
 *  history that does not reach back to the position's first buy leaves fewer
 *  lot shares than the position holds; ltApproachingRows then reports only
 *  the lots it can date, never a guessed one. */
export function fifoOpenLots(fills: readonly FillRow[]): OpenLot[] {
  const lots: OpenLot[] = [];
  for (const f of [...fills].sort((a, b) => a.t - b.t)) {
    if (f.dir === 1) { lots.push({ qty: f.qty, price: f.price, t: f.t }); continue; }
    let left = f.qty;
    while (left > 1e-9 && lots.length) {
      const take = Math.min(left, lots[0].qty);
      lots[0].qty -= take;
      left -= take;
      if (lots[0].qty <= 1e-9) lots.shift();
    }
  }
  return lots;
}

/** Calendar days held by UTC date and the first long-term date (held MORE than one year: day 366). */
export function lotTerm(t: number, now: Date): { daysHeld: number; daysToLongTerm: number; longTermDate: string } {
  const open = new Date(t);
  const d0 = Date.UTC(open.getUTCFullYear(), open.getUTCMonth(), open.getUTCDate());
  const d1 = Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate());
  const daysHeld = Math.round((d1 - d0) / DAY_MS);
  return { daysHeld, daysToLongTerm: 366 - daysHeld, longTermDate: new Date(d0 + 366 * DAY_MS).toISOString().slice(0, 10) };
}

const usd = (x: number) => "$" + Math.round(x).toLocaleString("en-US");

/** Pure: every open long stock/ETF lot within `windowDays` of its long-term
 *  date, from the fill history the sleeve ledger walked and the live
 *  positions (the floor's included — informational: its rebalancing trims are
 *  risk exits and never held). `holdOf(ticker)`: the stop state's tax hold of
 *  a position the exit logic manages, null otherwise. */
export function ltApproachingRows(fills: readonly FillRow[], positions: readonly PositionRow[], now: Date,
  holdOf: (ticker: string) => TaxHold | null, windowDays = TAX_LT_WINDOW_DAYS,
  rates: { shortTerm: number; longTerm: number } = DEFAULT_TAX_RATES): LtApproachingRow[] {
  const bySymbol = new Map<string, FillRow[]>();
  for (const f of fills) { const a = bySymbol.get(f.symbol); if (a) a.push(f); else bySymbol.set(f.symbol, [f]); }
  const st = Math.round(rates.shortTerm * 100), lt = Math.round(rates.longTerm * 100);
  const out: LtApproachingRow[] = [];
  for (const p of positions) {
    if (p.option || !(p.qty > 0) || p.avgEntry === null || p.unrealizedPl === null) continue;
    const priceNow = p.avgEntry + p.unrealizedPl / p.qty;
    const lots = fifoOpenLots(bySymbol.get(p.symbol) ?? []);
    // the broker holds p.qty: keep the NEWEST p.qty lot shares (a surplus means the walk missed a sale)
    let excess = lots.reduce((s, l) => s + l.qty, 0) - p.qty;
    while (excess > 1e-9 && lots.length) {
      const take = Math.min(excess, lots[0].qty);
      lots[0].qty -= take;
      excess -= take;
      if (lots[0].qty <= 1e-9) lots.shift();
    }
    const hold = holdOf(p.symbol);
    for (const lot of lots) {
      const term = lotTerm(lot.t, now);
      if (term.daysToLongTerm <= 0 || term.daysToLongTerm > windowDays) continue;
      const gain = Math.round((priceNow - lot.price) * lot.qty * 100) / 100;
      const days = `${term.daysToLongTerm} day${term.daysToLongTerm === 1 ? "" : "s"}`;
      const lotText = `${lot.qty} sh bought ${new Date(lot.t).toISOString().slice(0, 10)} at ${"$" + lot.price.toFixed(2)}`;
      const reason = gain > 0
        ? `${days} to long-term (${term.longTermDate}) for ${lotText}: selling it now would be taxed at ${st}% instead of ${lt}%, about ${usd(gain * (rates.shortTerm - rates.longTerm))} more tax on the ${usd(gain)} gain`
          + (hold ? `; its profit-taking exits are held until ${hold.until} (stops still run)` : "")
        : `${days} to long-term (${term.longTermDate}) for ${lotText}, at a ${usd(-gain)} loss: sold before then it stays short-term and offsets gains taxed at ${st}%`;
      out.push({ ticker: p.symbol, daysToLongTerm: term.daysToLongTerm, gain, reason });
    }
  }
  return out.sort((a, b) => a.daysToLongTerm - b.daysToLongTerm);
}

// ── tax-loss harvesting (2026-11-06) ────────────────────────────────────────
// The harvest review and its swaps are Python (tax_filter.harvest_step, once
// per scan). bot.ts supplies the one input only it has, this calendar year's
// realized gains from the sleeve ledger's fill walk (written to
// DATA_DIR/voltrade_tax_realized.json on every refresh, kept fresh by Tier 3),
// serves the latest review, and audits every swap the scan made.

export const TAX_REALIZED_FILE = "voltrade_tax_realized.json";
export const TAX_HARVEST_REVIEW_FILE = "voltrade_tax_harvest_review.json";
/** Tier 3 refreshes the sleeve ledger when it is older than this, so the snapshot stays under tax_filter's 48 h bar. */
export const TAX_REALIZED_REFRESH_MS = 6 * 3_600_000;
const WASH_LOOKBACK_MS = 31 * 86_400_000;

export interface TaxRealizedSnapshot {
  schema: "tax_realized/v1";
  asOf: string;
  year: number;
  /** server/taxLedger.ts netting for the calendar year to date (wash-sale disallowances added back) */
  shortNet: number; longNet: number; shortAfterNetting: number; longAfterNetting: number; carryforward: number;
  washDisallowed: number;
  estimatedTax: number;
  /** symbol -> newest close at a loss in the last 31 days (a replacement bought now would wash it) */
  recentLossSales: Record<string, string>;
  /** symbol -> newest buy fill in the last 31 days */
  recentBuys: Record<string, string>;
  source: string;
}

/** Pure: the snapshot tax_filter.harvest_step reads. */
export function taxRealizedSnapshot(events: readonly RealizedEvent[], fills: readonly FillRow[], now: Date): TaxRealizedSnapshot {
  const year = now.getUTCFullYear();
  const ledger = taxLedger(events, fills, { windowStartMs: Date.UTC(year, 0, 1) });
  const cut = now.getTime() - WASH_LOOKBACK_MS;
  const newest = (into: Record<string, string>, symbol: string, t: number) => {
    const iso = new Date(t).toISOString();
    if (!into[symbol] || iso > into[symbol]) into[symbol] = iso;
  };
  const recentLossSales: Record<string, string> = {};
  for (const e of events) if (e.pnl < 0 && e.t >= cut) newest(recentLossSales, e.symbol, e.t);
  const recentBuys: Record<string, string> = {};
  for (const f of fills) if (f.dir === 1 && f.t >= cut) newest(recentBuys, f.symbol, f.t);
  const n = ledger.netting;
  return {
    schema: "tax_realized/v1", asOf: now.toISOString(), year,
    shortNet: n.shortNet, longNet: n.longNet, shortAfterNetting: n.shortAfterNetting, longAfterNetting: n.longAfterNetting,
    carryforward: n.carryforward, washDisallowed: ledger.washSales.disallowed, estimatedTax: ledger.estimatedTax.total,
    recentLossSales, recentBuys,
    source: "server/taxLedger.ts over the sleeve ledger's realized events, calendar year to date",
  };
}

/** Atomic write; returns the error text instead of throwing (a snapshot must never fail a ledger refresh). */
export function writeTaxRealized(dir: string, snap: TaxRealizedSnapshot): string | null {
  const p = path.join(dir, TAX_REALIZED_FILE);
  try {
    fs.writeFileSync(`${p}.tmp`, JSON.stringify(snap));
    fs.renameSync(`${p}.tmp`, p);
    return null;
  } catch (err) {
    return String(err instanceof Error ? err.message : err).slice(0, 200);
  }
}

export interface HarvestReview { asOf: string | null; enabled: boolean | null; realizedWhy: string | null; harvest: HarvestRow[] }

/** The latest review tax_filter wrote; null when there is none (or it is unreadable — the reason is returned). */
export function readHarvestReview(dir: string): { review: HarvestReview | null; error: string | null } {
  let raw: unknown;
  try {
    raw = JSON.parse(fs.readFileSync(path.join(dir, TAX_HARVEST_REVIEW_FILE), "utf8"));
  } catch (err) {
    const code = (err as NodeJS.ErrnoException)?.code;
    return { review: null, error: code === "ENOENT" ? null : String(err instanceof Error ? err.message : err).slice(0, 200) };
  }
  const r = (raw && typeof raw === "object" ? raw : {}) as Record<string, unknown>;
  const rows = Array.isArray(r.harvest) ? r.harvest : [];
  const harvest: HarvestRow[] = [];
  for (const x of rows) {
    if (!x || typeof x !== "object") continue;
    const o = x as Record<string, unknown>;
    const ticker = str(o.ticker);
    const unrealized = num(o.unrealized);
    if (!ticker || unrealized === null) continue;
    harvest.push({ ticker, unrealized, replacement: str(o.replacement), washSaleUntil: str(o.washSaleUntil), reason: str(o.reason) ?? "",
      action: str(o.action) ?? "none", taxSaving: num(o.taxSaving) });
  }
  return { review: { asOf: str(r.asOf), enabled: typeof r.enabled === "boolean" ? r.enabled : null, realizedWhy: str(r.realizedWhy), harvest }, error: null };
}

/** Audit rows for the swaps one scan made (scan_market's `tax_harvest`): one per swap, plus the step's error. */
export function taxHarvestAuditRows(h: unknown): Array<{ type: string; message: string }> {
  if (!h || typeof h !== "object") return [];
  const o = h as { actions?: unknown; error?: unknown };
  const out: Array<{ type: string; message: string }> = [];
  for (const a of Array.isArray(o.actions) ? o.actions : []) {
    if (!a || typeof a !== "object") continue;
    const x = a as { ticker?: unknown; replacement?: unknown; sold?: { ok?: unknown; error?: unknown } | null; bought?: { ok?: unknown; error?: unknown } | null; error?: unknown };
    const t = String(x.ticker ?? "?"), r = String(x.replacement ?? "?");
    if (x.sold && x.sold.ok) {
      const bought = x.bought && x.bought.ok ? `bought ${r} with the proceeds` : `${r} NOT bought (${String(x.bought?.error ?? "no order")}), proceeds in cash`;
      out.push({ type: "TAX-HARVEST", message: `${t}: sold for a tax-loss harvest; ${bought}; ${t} is blocked from re-entry for 30 days (wash-sale window)` });
    } else {
      out.push({ type: "TAX-HARVEST-FAIL", message: `${t}: harvest sale refused (${String(x.sold?.error ?? x.error ?? "unknown")}); nothing bought` });
    }
  }
  if (o.error) out.push({ type: "TAX-HARVEST-ERROR", message: String(o.error).slice(0, 200) });
  return out;
}
