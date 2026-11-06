// decisionAttribution.ts — "why are we losing money" (2026-11-06; human
// refocus spec §8 "a clear panel: loss attribution by signal, sector,
// instrument, timing"; §7 "which signals add return, which are noise").
// GET /api/bot/attribution?days= and the "attribution" diag probe serve
// attributionReport(): the window's P&L split four ways (plus execution), each
// bucket with a plain-English reason.
//
// THE P&L (no new ruler — the sleeve ledger's): realized = the closed-lot
// walk of the broker's fill history (sleeveLedger.realizedEvents, average
// cost, the same walk /api/bot/sleeves and the "lots" probe serve) for every
// lot CLOSED inside the window; unrealized = the current unrealized P&L of
// each open position (since its entry — the mark at the window's start is not
// on record). Each grouping adds up to the same total.
//
// THE LABELS, per lot:
//   signal     — what opened it: the decision journal's entry row (or its
//                one-time backfill) matched on symbol / underlying and open
//                time; the floor is "floor rebalance"; no match = "not
//                recorded" (never guessed).
//   sector     — the entry row's sector, else the sector cache bot_engine
//                keeps (DATA_DIR/voltrade_sector_cache.json), else what an
//                index / commodity ETF holds, else "Unknown". Options: the
//                underlying's.
//   instrument — stocks / leveraged ETFs / the floor / options by strategy:
//                the entry row's strategy, else the structure INFERRED from
//                the legs that opened together (labelled inferred).
//   timing     — entry and exit time of day (ET: the first 30 minutes, the
//                middle, the last 30 minutes, outside regular hours) and hold
//                length (sleeveLedger.HOLD_BUCKETS).
//   execution  — option fills against the mid recorded at submission: the
//                daily line's own optionFillsInWindow + fillsSlippage over the
//                option fill ledger (no second slippage formula here). Fills
//                with no recorded mid are counted as unattributed; their
//                slippage stays inside the other buckets.
// Pure (node:test safe); bot.ts gathers the inputs and caches the body.
import fs from "node:fs";
import path from "node:path";
import { holdBucket, occRoot, sleeveOf, type HoldBucket, type RealizedEvent, type Sleeve } from "./sleeveLedger";
import type { FillRow, PositionRow } from "./killSwitchAutoResume";
import { fillsSlippage, optionFillsInWindow, type LedgerRow, type OptionFill } from "./dailyLine";
import { money, type DecisionRow } from "./decisionJournal";
import { inferOptionStructure, openings, STRUCTURE_GROUP_MS } from "./decisionBackfill";

export const ATTRIBUTION_DEFAULT_DAYS = 30;
export const ATTRIBUTION_MAX_DAYS = 365;
/** bot.ts recomputes a window's body at most this often (each compute reads the orders of the window). */
export const ATTRIBUTION_CACHE_MS = 10 * 60_000;
/** A journal entry explains a lot that opened up to this long after it (a DAY limit fills the same session; this covers a weekend). */
export const ATTRIBUTION_ENTRY_MATCH_MS = 3 * 86_400_000;
/** ...and an exit row explains a close up to this long after it. */
export const ATTRIBUTION_EXIT_MATCH_MS = 86_400_000;
/** A broker order fill and an activity fill of the same contract this close in time are the same fill. */
export const ATTRIBUTION_FILL_MATCH_MS = 120_000;
/** Journal rows read per computation (the whole journal at its size cap is ~10k rows). */
export const ATTRIBUTION_MAX_DECISIONS = 50_000;
export const ATTRIBUTION_WORST_TRADES = 10;
export const SECTOR_CACHE_FILENAME = "voltrade_sector_cache.json";
export const NOT_RECORDED_SIGNAL = "not recorded (before the decision journal)";

const DAY = 86_400_000;
const r2 = (x: number) => Math.round(x * 100) / 100;

/** What an index / commodity / sector ETF holds — a fact about the fund, used when no sector is on record. */
export const ETF_SECTORS: Readonly<Record<string, string>> = {
  SPY: "Index: S&P 500", SSO: "Index: S&P 500", UPRO: "Index: S&P 500", SPXL: "Index: S&P 500", VOO: "Index: S&P 500", IVV: "Index: S&P 500",
  QQQ: "Index: Nasdaq-100", QLD: "Index: Nasdaq-100", TQQQ: "Index: Nasdaq-100",
  IWM: "Index: Russell 2000", TNA: "Index: Russell 2000", VTI: "Index: US total market",
  GLD: "Gold", IAU: "Gold", SLV: "Silver", TLT: "Treasuries", IEF: "Treasuries", SCHP: "Treasuries (TIPS)",
  VXUS: "International equity", EFA: "International equity", EEM: "Emerging markets", KWEB: "China internet",
  SMH: "Semiconductors", SOXX: "Semiconductors", SOXL: "Semiconductors", SVXY: "Volatility", VXX: "Volatility", UVXY: "Volatility",
  XLK: "Technology", XLF: "Financials", XLE: "Energy", XLV: "Health Care", XLP: "Consumer Staples", XLU: "Utilities",
  XLY: "Consumer Discretionary", XLI: "Industrials", XLB: "Materials", XLRE: "Real Estate", XLC: "Communication Services",
};

/** The journal's option strategy names, as the panel says them. */
const STRATEGY_NAMES: Readonly<Record<string, string>> = {
  sell_cash_secured_put: "cash-secured puts", csp: "cash-secured puts", csp_core: "cash-secured puts",
  covered_call: "covered calls", iron_condor: "iron condors", qqq_iron_condor: "iron condors",
  bull_put_credit_spread: "put credit spreads", bear_call_credit_spread: "call credit spreads",
  bull_call_spread: "call debit spreads", bear_put_spread: "put debit spreads",
  buy_call: "long calls", buy_put: "long puts", "buy_put (tail hedge)": "tail-hedge puts",
  short_straddle: "short straddles", buy_straddle: "long straddles", calendar_spread: "calendar spreads", diagonal_spread: "diagonal spreads",
};
const INFERRED_NAMES: Readonly<Record<string, string>> = {
  "short put (cash-secured)": "cash-secured puts", "short call": "short calls", "long put": "long puts", "long call": "long calls",
  "put credit spread": "put credit spreads", "put debit spread": "put debit spreads", "call credit spread": "call credit spreads",
  "call debit spread": "call debit spreads", "iron condor": "iron condors", "short straddle": "short straddles",
  "short strangle": "short strangles", "long straddle": "long straddles", "long strangle": "long strangles",
  "calendar spread": "calendar spreads", "diagonal spread": "diagonal spreads", "risk reversal": "risk reversals",
};

// ── time of day (ET) ─────────────────────────────────────────────────────────
export const TIME_WINDOWS = ["the first 30 minutes (09:30–10:00 ET)", "the middle of the day (10:00–15:30 ET)",
  "the last 30 minutes (15:30–16:00 ET)", "outside regular hours", "unknown"] as const;
export type TimeWindow = (typeof TIME_WINDOWS)[number];
const ET_PARTS = new Intl.DateTimeFormat("en-US", { timeZone: "America/New_York", hour: "2-digit", minute: "2-digit", hourCycle: "h23", weekday: "short" });
export function timeWindow(ms: number | null): TimeWindow {
  if (ms === null || !Number.isFinite(ms)) return "unknown";
  const parts = Object.fromEntries(ET_PARTS.formatToParts(new Date(ms)).map((p) => [p.type, p.value]));
  if (parts.weekday === "Sat" || parts.weekday === "Sun") return "outside regular hours";
  const m = parseInt(parts.hour, 10) * 60 + parseInt(parts.minute, 10);
  if (m < 9 * 60 + 30 || m >= 16 * 60) return "outside regular hours";
  if (m < 10 * 60) return TIME_WINDOWS[0];
  if (m < 15 * 60 + 30) return TIME_WINDOWS[1];
  return TIME_WINDOWS[2];
}
const HOLD_TEXT: Readonly<Record<HoldBucket, string>> = {
  "<1h": "held under an hour", "<1d": "held under a day", "1–7d": "held 1–7 days", "8–30d": "held 8–30 days",
  ">30d": "held over 30 days", unknown: "held for an unknown time (opened before the fill history)",
};

// ── inputs ───────────────────────────────────────────────────────────────────
export interface AttributionInputs {
  nowMs: number;
  days: number;
  /** the sleeve ledger's closed-lot walk (whole history) and the fills it walked */
  events: readonly RealizedEvent[];
  fills: readonly FillRow[];
  positions: readonly PositionRow[];
  /** journal rows (live + backfill), any order */
  decisions: readonly DecisionRow[];
  /** Alpaca's nested orders of the window (dailyLineOrders); null when unreadable */
  orders: unknown[] | null;
  ordersError?: string | null;
  /** the option fill ledger (dailyLine.readOptionFillLedger) */
  ledger: { rows: readonly LedgerRow[]; malformed: number };
  floorTickers: ReadonlySet<string>;
  /** bot_engine's sector cache (ticker -> sector) */
  sectors: Readonly<Record<string, string>>;
}

export interface Lot {
  symbol: string; root: string; option: boolean; sleeve: Sleeve;
  openedAt: number | null; closedAt: number | null; qty: number; pnl: number; open: boolean;
  instrument: string; signal: string; sector: string; regime: string; exitRule: string | null;
  entryWindow: TimeWindow; exitWindow: TimeWindow; hold: HoldBucket;
  /** the option strategy was inferred from the legs (no recorded strategy) */
  inferred: boolean;
}

export interface Bucket {
  /** the bucket's name under the grouping's own key (signal / sector / instrument / timing / execution) */
  [key: string]: unknown;
  pnl: number; realized: number; unrealized: number; n: number; open: number; wins: number; losses: number; reason: string;
}

export interface AttributionReport {
  window: { days: number; from: string; to: string; label: string };
  total: number; realized: number; unrealized: number; lots: number; openPositions: number;
  bySignal: Bucket[]; bySector: Bucket[]; byInstrument: Bucket[]; byTiming: Bucket[]; byExecution: Bucket[];
  worstTrades: Array<{ symbol: string; root: string; pnl: number; open: boolean; signal: string; instrument: string; closedAt: string | null }>;
  notes: string[];
  sources: Record<string, unknown>;
}

export function attributionDays(raw: unknown): number {
  const n = parseInt(String(raw ?? ""), 10);
  return Math.min(Math.max(Number.isFinite(n) ? n : ATTRIBUTION_DEFAULT_DAYS, 1), ATTRIBUTION_MAX_DAYS);
}

/** bot_engine's ticker -> sector cache, or {} (missing / unreadable). */
export function readSectorCache(dir: string): Record<string, string> {
  try {
    const raw: unknown = JSON.parse(fs.readFileSync(path.join(dir, SECTOR_CACHE_FILENAME), "utf8"));
    const out: Record<string, string> = {};
    if (raw && typeof raw === "object" && !Array.isArray(raw)) {
      for (const [k, v] of Object.entries(raw as Record<string, unknown>)) if (typeof v === "string" && v) out[k.toUpperCase()] = v;
    }
    return out;
  } catch (err) {
    if ((err as NodeJS.ErrnoException)?.code !== "ENOENT") console.error("[attribution] sector cache unreadable:", err instanceof Error ? err.message : err);
    return {};
  }
}

// ── the journal, indexed ─────────────────────────────────────────────────────
const ENTRY_ACTIONS = new Set(["buy", "scale_in", "rotate"]);
const EXIT_ACTIONS = new Set(["sell", "scale_out", "rotate"]);
interface Indexed { ts: number; row: DecisionRow }
function index(decisions: readonly DecisionRow[], actions: ReadonlySet<string>): { bySymbol: Map<string, Indexed[]>; byTicker: Map<string, Indexed[]> } {
  const bySymbol = new Map<string, Indexed[]>();
  const byTicker = new Map<string, Indexed[]>();
  const add = (m: Map<string, Indexed[]>, k: string, x: Indexed) => { const a = m.get(k); if (a) a.push(x); else m.set(k, [x]); };
  for (const row of decisions) {
    if (!actions.has(row.action)) continue;
    const ts = Date.parse(row.ts);
    if (!Number.isFinite(ts)) continue;
    const x = { ts, row };
    if (row.symbol) add(bySymbol, String(row.symbol).toUpperCase(), x);
    add(byTicker, row.ticker.toUpperCase(), x);
  }
  return { bySymbol, byTicker };
}
function newestIn(list: readonly Indexed[] | undefined, lo: number, hi: number, ok: (r: DecisionRow) => boolean): DecisionRow | null {
  let best: Indexed | null = null;
  for (const x of list ?? []) if (x.ts >= lo && x.ts <= hi && ok(x.row) && (!best || x.ts > best.ts)) best = x;
  return best ? best.row : null;
}
function str(x: unknown): string { return typeof x === "string" ? x : x === null || x === undefined ? "" : String(x); }
function stageField(row: DecisionRow | null, stage: string, key: string): string {
  const st = row?.stages?.[stage];
  return st && typeof st === "object" ? str((st as Record<string, unknown>)[key]) : "";
}

/** The strategy name and whether it was inferred: the journal row's strategy, else the
 *  legs' structure (one name either way, so a recorded and an inferred condor share a bucket). */
function optionStrategyName(row: DecisionRow | null, inferred: string | null): { name: string; inferred: boolean } {
  const chosen = str(row?.chosen).trim();
  const fromRow = STRATEGY_NAMES[chosen] || STRATEGY_NAMES[chosen.split(" (")[0]] || STRATEGY_NAMES[stageField(row, "options", "strategy")]
    || INFERRED_NAMES[stageField(row, "options", "strategy")];
  if (fromRow) return { name: fromRow, inferred: false };
  if (inferred && inferred !== "option") return { name: INFERRED_NAMES[inferred] ?? inferred, inferred: true };
  return { name: "strategy not recorded", inferred: false };
}

// ── the lots ─────────────────────────────────────────────────────────────────
export function buildLots(i: AttributionInputs): Lot[] {
  const fromMs = i.nowMs - i.days * DAY;
  const entries = index(i.decisions, ENTRY_ACTIONS);
  const exits = index(i.decisions, EXIT_ACTIONS);
  // option structures: the legs that opened together, across the whole history + what is open now
  const legs = new Map<string, Array<{ symbol: string; short: boolean }>>();
  const groupKey = (root: string, t: number) => `${root}|${Math.round(t / STRUCTURE_GROUP_MS)}`;
  const addLeg = (symbol: string, t: number | null, short: boolean) => {
    if (t === null || sleeveOf(symbol, i.floorTickers) !== "options") return;
    const k = groupKey(occRoot(symbol).toUpperCase(), t);
    const a = legs.get(k) ?? [];
    if (!a.some((l) => l.symbol === symbol)) a.push({ symbol, short });
    legs.set(k, a);
  };
  for (const e of i.events) addLeg(e.symbol, e.openedAt, e.closeDir === 1);
  const opened = openings([], i.fills, i.positions);
  for (const o of opened) addLeg(o.symbol, o.t, o.short);
  const openTime = new Map(opened.map((o) => [o.symbol, o.t] as const));

  const label = (symbol: string, openedAt: number | null, closedAt: number | null): Omit<Lot, "pnl" | "qty" | "open" | "closedAt" | "openedAt" | "symbol"> => {
    let inferredStrategy = false;
    const option = sleeveOf(symbol, i.floorTickers) === "options";
    const sleeve = sleeveOf(symbol, i.floorTickers);
    const root = occRoot(symbol).toUpperCase();
    const lo = (openedAt ?? 0) - ATTRIBUTION_ENTRY_MATCH_MS, hi = (openedAt ?? 0) + 5 * 60_000;
    const entry = openedAt === null ? null
      : newestIn(entries.bySymbol.get(symbol.toUpperCase()), lo, hi, () => true)
        ?? newestIn(entries.byTicker.get(root), lo, hi, (r) => (r.instrument === "option") === option && (option || !r.symbol || r.symbol === symbol));
    const exit = closedAt === null ? null
      : newestIn(exits.bySymbol.get(symbol.toUpperCase()), closedAt - ATTRIBUTION_EXIT_MATCH_MS, closedAt + 5 * 60_000, () => true)
        ?? newestIn(exits.byTicker.get(root), closedAt - ATTRIBUTION_EXIT_MATCH_MS, closedAt + 5 * 60_000, (r) => (r.instrument === "option") === option);
    let instrument: string;
    if (sleeve === "floor") instrument = "floor";
    else if (option) {
      const strat = optionStrategyName(entry, openedAt === null ? null : inferOptionStructure(legs.get(groupKey(root, openedAt)) ?? [{ symbol, short: false }]));
      instrument = `options: ${strat.name}`;
      inferredStrategy = strat.inferred;
    }
    else instrument = entry?.instrument === "etf" ? "leveraged ETFs" : "stocks";
    const signal = sleeve === "floor" ? "floor rebalance" : str(entry?.signal) || NOT_RECORDED_SIGNAL;
    const sector = stageField(entry, "sector", "name") || i.sectors[root] || ETF_SECTORS[root] || "Unknown";
    const regime = stageField(entry, "regime", "label") || "unknown";
    return { root, option, sleeve, instrument, signal, sector, regime, exitRule: exit ? str(exit.chosen) || null : null, inferred: inferredStrategy,
      entryWindow: timeWindow(openedAt), exitWindow: closedAt === null ? "unknown" : timeWindow(closedAt),
      hold: holdBucket(openedAt, closedAt ?? i.nowMs) };
  };
  const lots: Lot[] = [];
  for (const e of i.events) {
    if (e.t < fromMs || e.t > i.nowMs) continue;
    lots.push({ symbol: e.symbol, openedAt: e.openedAt, closedAt: e.t, qty: e.qty, pnl: e.pnl, open: false, ...label(e.symbol, e.openedAt, e.t) });
  }
  for (const p of i.positions) {
    if (p.unrealizedPl === null) continue;
    const t = openTime.get(p.symbol) ?? null;
    lots.push({ symbol: p.symbol, openedAt: t, closedAt: null, qty: Math.abs(p.qty), pnl: p.unrealizedPl, open: true, ...label(p.symbol, t, null) });
  }
  return lots;
}

// ── execution: the daily line's fills vs mid, per lot ────────────────────────
interface FillCost { fill: OptionFill; lotIdx: number[] }
/** Each window option fill, joined to the lot(s) it opened or closed (same contract, within ATTRIBUTION_FILL_MATCH_MS). */
function fillsByLot(lots: readonly Lot[], fills: readonly OptionFill[]): FillCost[] {
  const bySymbol = new Map<string, number[]>();
  lots.forEach((l, idx) => { if (l.option) { const a = bySymbol.get(l.symbol) ?? []; a.push(idx); bySymbol.set(l.symbol, a); } });
  return fills.map((f) => {
    const t = Date.parse(str(f.filled_at));
    const near = (x: number | null) => x !== null && Number.isFinite(t) && Math.abs(x - t) <= ATTRIBUTION_FILL_MATCH_MS;
    return { fill: f, lotIdx: (bySymbol.get(f.symbol) ?? []).filter((idx) => near(lots[idx].openedAt) || near(lots[idx].closedAt)) };
  });
}

// ── buckets + plain-English reasons ──────────────────────────────────────────
type Dim = "signal" | "sector" | "instrument" | "regime" | "entryWindow" | "exitWindow" | "hold" | "exitRule";
const DIM_PHRASE: Record<Dim, (v: string) => string> = {
  signal: (v) => `trades from ${v}`,
  sector: (v) => `${v} names`,
  instrument: (v) => (v === "floor" ? "the floor" : v),
  regime: (v) => `entries in a ${v} market`,
  entryWindow: (v) => `entries in ${v}`,
  exitWindow: (v) => `exits in ${v}`,
  hold: (v) => `positions ${HOLD_TEXT[v as HoldBucket] ?? v}`,
  exitRule: (v) => `exits by the ${v.replace(/_/g, " ")} rule`,
};
function dimValue(l: Lot, d: Dim): string | null {
  const v = d === "exitRule" ? l.exitRule : String(l[d]);
  if (!v || v === "unknown" || v === "Unknown" || v === NOT_RECORDED_SIGNAL) return null;
  if (d === "exitWindow" && l.open) return null;
  return v;
}

function sumOf(lots: readonly Lot[]) {
  let realized = 0, unrealized = 0, n = 0, open = 0, wins = 0, losses = 0;
  for (const l of lots) {
    if (l.open) { unrealized += l.pnl; open++; continue; }
    realized += l.pnl; n++;
    if (l.pnl > 0) wins++; else if (l.pnl < 0) losses++;
  }
  return { pnl: r2(realized + unrealized), realized: r2(realized), unrealized: r2(unrealized), n, open, wins, losses };
}

/** The sub-group (along another dimension) holding the largest share of a
 *  bucket's P&L, in plain English. Dimensions are tried in this order, so on
 *  a tie the time-of-day and regime reads win. */
const DRIVER_DIMS: readonly Dim[] = ["exitWindow", "regime", "hold", "exitRule", "instrument", "signal", "sector", "entryWindow"];
export function driverPhrase(lots: readonly Lot[], total: number, skip: readonly Dim[], wholeSkip: readonly Dim[] = []): string {
  if (Math.abs(total) < 0.005 || lots.length < 2) return "";
  let best: { phrase: string; share: number; pnl: number; n: number } | null = null;
  for (const d of DRIVER_DIMS) {
    if (skip.includes(d)) continue;
    const groups = new Map<string, Lot[]>();
    for (const l of lots) { const v = dimValue(l, d); if (v) { const a = groups.get(v) ?? []; a.push(l); groups.set(v, a); } }
    for (const [v, g] of groups) {
      // a category every lot of the bucket shares only restates the bucket (floor <-> floor rebalance)
      if (wholeSkip.includes(d) && g.length === lots.length) continue;
      const p = g.reduce((s, l) => s + l.pnl, 0);
      const share = p / total;
      if (share > 0.3 && (!best || share > best.share + 1e-9)) best = { phrase: DIM_PHRASE[d](v), share, pnl: p, n: g.length };
    }
  }
  if (!best) return "";
  const what = `${money(best.pnl)} across ${best.n} ${best.n === 1 ? "lot" : "lots"}`;
  if (best.share > 1.0005) return `; ${best.phrase} alone came to ${what}, more than the net (other lots offset part of it)`;
  return `; ${best.share >= 0.5 ? "mostly" : "the largest part"} from ${best.phrase}: ${what} (${Math.round(best.share * 100)}% of it)`;
}

/** One bucket's sentence: what it made or lost, realized vs open, its main driver, its fill cost. */
export function reasonFor(label: string, lots: readonly Lot[], skip: readonly Dim[], execution: string, wholeSkip: readonly Dim[] = []): string {
  const s = sumOf(lots);
  const verb = s.pnl < 0 ? "lost" : s.pnl > 0 ? "made" : "broke even on";
  const trades = `${s.n} closed ${s.n === 1 ? "trade" : "trades"} (${s.wins} won, ${s.losses} lost)`;
  const opens = `${s.open} open ${s.open === 1 ? "position" : "positions"}`;
  let head: string;
  if (s.n && s.open) head = `${label} ${verb} ${money(Math.abs(s.pnl))}: ${money(s.realized)} realized across ${trades} and ${money(s.unrealized)} unrealized on ${opens}`;
  else if (s.n) head = `${label} ${verb} ${money(Math.abs(s.pnl))} across ${trades}`;
  else head = `${label} ${s.pnl < 0 ? "is down" : s.pnl > 0 ? "is up" : "is flat at"} ${money(Math.abs(s.pnl))} unrealized on ${opens}`;
  const inferred = lots.filter((l) => l.inferred).length;
  const note = inferred ? ` (the strategy of ${inferred} of these ${lots.length} lots was inferred from the legs that opened together)` : "";
  return `${head}${driverPhrase(lots, s.pnl, skip, wholeSkip)}${execution}${note}.`;
}

/** Group the lots `include` admits by `valueOf`; idxs stay indexes into the FULL list (the fill join keys on them). */
function group(lots: readonly Lot[], key: string, dim: Dim | null, valueOf: (l: Lot) => string, execOf: (idx: number[]) => string,
  extra: Record<string, unknown> = {}, include: (l: Lot) => boolean = () => true): Bucket[] {
  const groups = new Map<string, number[]>();
  lots.forEach((l, idx) => {
    if (!include(l)) return;
    const v = valueOf(l);
    const a = groups.get(v) ?? [];
    a.push(idx);
    groups.set(v, a);
  });
  const out: Bucket[] = [];
  for (const [v, idxs] of groups) {
    const g = idxs.map((x) => lots[x]);
    const wholeSkip: Dim[] = key === "signal" ? ["instrument"] : key === "instrument" ? ["signal"] : [];
    out.push({ [key]: v, ...extra, ...sumOf(g), reason: reasonFor(capitalize(bucketLabel(key, v)), g, dim ? [dim] : [], execOf(idxs), wholeSkip) });
  }
  return out.sort((a, b) => a.pnl - b.pnl || String(a[key]).localeCompare(String(b[key])));
}
function bucketLabel(key: string, v: string): string {
  if (key === "signal" && v === NOT_RECORDED_SIGNAL) return "trades whose entry decision was not recorded (opened before the decision journal)";
  if (key === "signal" && v === "floor rebalance") return "the floor's rebalancing trades";
  if (key === "signal") return `trades from ${v}`;
  if (key === "sector") return `${v} positions`;
  return v;
}
function capitalize(s: string): string { return s ? s.charAt(0).toUpperCase() + s.slice(1) : s; }

// ── the report ───────────────────────────────────────────────────────────────
export function attributionReport(i: AttributionInputs): AttributionReport {
  const fromMs = i.nowMs - i.days * DAY;
  const lots = buildLots(i);
  // execution (the daily line's math over the window's option fills)
  const win = i.orders ? optionFillsInWindow(i.orders, fromMs, i.nowMs) : { fills: [] as OptionFill[], mleg_no_leg_fills: 0 };
  const joined = fillsByLot(lots, win.fills);
  const fillsOfLots = (idxs: number[]): OptionFill[] => {
    const set = new Set(idxs);
    return joined.filter((j) => j.lotIdx.some((x) => set.has(x))).map((j) => j.fill);
  };
  const execOf = (idxs: number[]): string => {
    const f = fillsOfLots(idxs);
    if (!f.length) return "";
    const b = fillsSlippage(f, i.ledger);
    if (!b.attributed) return "";
    return `; fills versus the mid at submission ${b.cost >= 0 ? "cost" : "saved"} ${money(Math.abs(b.cost))} of it (${b.attributed} of ${b.fills} option fills priced)`;
  };
  const execution: Bucket[] = [];
  if (i.orders) {
    const byWin = new Map<TimeWindow, OptionFill[]>();
    for (const f of win.fills) { const w = timeWindow(Date.parse(str(f.filled_at))); const a = byWin.get(w) ?? []; a.push(f); byWin.set(w, a); }
    for (const [w, f] of byWin) {
      const b = fillsSlippage(f, i.ledger);
      const worst = b.worst[0];
      execution.push({
        execution: `option fills in ${w}`, pnl: b.pnl, realized: b.pnl, unrealized: 0, n: b.attributed, open: 0, wins: 0, losses: 0,
        fills: b.fills, unattributed: b.unattributed.count, cost: b.cost,
        reason: b.attributed
          ? `${b.attributed} of ${b.fills} option fills in ${w} were priced against the mid recorded at submission: they ${b.cost >= 0 ? "cost" : "saved"} ${money(Math.abs(b.cost))}${worst ? ` (worst: ${worst.symbol}, ${worst.side} ${worst.qty} at ${worst.fill} vs a ${worst.mid} mid, ${money(worst.cost)})` : ""}.${b.unattributed.count ? ` ${b.unattributed.count} had no recorded mid; their slippage stays inside the other buckets.` : ""}`
          : `${b.fills} option fills in ${w}, none with a recorded mid (the option fill ledger records the mid from the daily line's release on), so their slippage stays inside the other buckets.`,
      });
    }
    if (win.mleg_no_leg_fills) {
      execution.push({ execution: "multi-leg orders without per-leg fills", pnl: 0, realized: 0, unrealized: 0, n: 0, open: 0, wins: 0, losses: 0,
        fills: win.mleg_no_leg_fills, unattributed: win.mleg_no_leg_fills, cost: 0,
        reason: `${win.mleg_no_leg_fills} multi-leg orders filled with no per-leg fill prices; their fills are not split by a guessed sign convention.` });
    }
    execution.sort((a, b) => a.pnl - b.pnl);
  }

  const timing: Bucket[] = [
    ...group(lots, "timing", "exitWindow", (l) => (l.open ? "still open" : `exits in ${l.exitWindow}`), execOf, { dimension: "exit time" }),
    ...group(lots, "timing", "entryWindow", (l) => `entries in ${l.entryWindow}`, execOf, { dimension: "entry time" }),
    ...group(lots, "timing", "hold", (l) => `positions ${HOLD_TEXT[l.hold]}`, execOf, { dimension: "hold" }),
  ];
  const tot = sumOf(lots);
  const worstTrades = [...lots].sort((a, b) => a.pnl - b.pnl).slice(0, ATTRIBUTION_WORST_TRADES).filter((l) => l.pnl < 0)
    .map((l) => ({ symbol: l.symbol, root: l.root, pnl: r2(l.pnl), open: l.open, signal: l.signal, instrument: l.instrument,
      closedAt: l.closedAt === null ? null : new Date(l.closedAt).toISOString() }));
  const live = i.decisions.filter((d) => !d.backfilled && ENTRY_ACTIONS.has(d.action));
  const firstLive = live.reduce((m, d) => Math.min(m, Date.parse(d.ts) || Infinity), Infinity);
  const unmatched = lots.filter((l) => l.sleeve !== "floor" && l.signal === NOT_RECORDED_SIGNAL).length;
  const notes = [
    `P&L: realized is the broker fill history's closed-lot walk (average cost — the same walk as the sleeve ledger) for lots closed in ${i.days === 1 ? "the last day" : `the last ${i.days} days`}; unrealized is each open position's P&L since its entry. Every grouping adds up to the same total.`,
    "Timing has three groupings (exit time — open positions under \"still open\" — entry time, hold length); each one adds up to the total on its own.",
    `Signals come from the decision journal${Number.isFinite(firstLive) ? ` (live since ${new Date(firstLive).toISOString().slice(0, 10)})` : ""} and its one-time backfill (labelled backfilled); ${unmatched} of ${lots.length} lots have no recorded entry decision and show as "${NOT_RECORDED_SIGNAL}" — never guessed.`,
    "An option strategy that was not recorded is inferred from the legs that opened together, and labelled so.",
    i.orders
      ? `Execution compares each option fill with the mid recorded when its order was sent (the daily line's option fill ledger: ${i.ledger.rows.length} rows); a fill with no recorded mid is counted as unattributed and its slippage stays inside the other groupings.`
      : `Execution is unavailable for this window: the broker's orders could not be read (${i.ordersError || "unknown error"}).`,
    "Sector for an option is its underlying's; an index or commodity ETF is named by what it holds. Paper account: no fees, no dividends.",
  ];
  return {
    window: { days: i.days, from: new Date(fromMs).toISOString(), to: new Date(i.nowMs).toISOString(), label: i.days === 1 ? "the last day" : `the last ${i.days} days` },
    total: tot.pnl, realized: tot.realized, unrealized: tot.unrealized, lots: tot.n, openPositions: tot.open,
    bySignal: group(lots, "signal", "signal", (l) => l.signal, execOf),
    bySector: group(lots, "sector", "sector", (l) => l.sector, execOf),
    byInstrument: group(lots, "instrument", "instrument", (l) => l.instrument, execOf),
    byTiming: timing,
    byExecution: execution,
    worstTrades,
    notes,
    sources: {
      events: i.events.length, positions: i.positions.length, decisions: i.decisions.length,
      backfilledDecisions: i.decisions.filter((d) => d.backfilled).length, unmatchedLots: unmatched,
      orders: i.orders ? i.orders.length : null, optionFills: win.fills.length, ledgerRows: i.ledger.rows.length,
      generatedAt: new Date(i.nowMs).toISOString(),
    },
  };
}

