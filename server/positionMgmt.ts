// server/positionMgmt.ts — the Node half of position management (2026-11-06,
// refocus spec section 4; the Python half is position_mgmt.py, which owns every
// number: this file only spends and re-checks what Python computed).
//
// 1. CASH RESERVE. parseReserveSnapshot() reads position_mgmt.snapshot_rpc's
//    result (the daemon route "cash_reserve_snapshot"). A ReserveBudget spends
//    that headroom across one entry loop: fitShares() trims or blocks a stock /
//    ETF entry to whole shares (the same rule as position_mgmt.fit_entry),
//    fitOpen() blocks an option open once nothing is left, spend() records what
//    was placed. An off reserve (enabled=false) or an unreadable one gates
//    nothing — the same fail-open rule as the other pre-trade checks.
// 2. CAPITAL ROTATION. parseRotationPlan() reads the scan's rotation_plan
//    (position_mgmt.plan_rotations); rotationPreflight() re-checks a planned
//    swap against the live book right before server/bot.ts
//    runUpgradeCandidates places it.
// 3. appendPositionMgmtLog(): the same JSONL file position_mgmt.log_decision
//    writes (DATA_DIR/voltrade_position_mgmt.jsonl), one object per line.
//
// Pure except appendPositionMgmtLog (one append).

import fs from "node:fs";
import path from "node:path";

export const POSITION_MGMT_LOG_FILE = "voltrade_position_mgmt.jsonl";
export const POSITION_MGMT_LOG_MAX_BYTES = 5_000_000;
/** Regular-hours window a rotation may trade in (ET hours): after the opening
 *  spreads settle, before the closing minutes — the floor's own window. */
export const ROTATION_WINDOW_ET: readonly [number, number] = [9.75, 15.9];

const num = (v: unknown, d = 0): number => {
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? n : d;
};
const str = (v: unknown): string => (v === null || v === undefined ? "" : String(v));

// ── 1. Cash reserve ────────────────────────────────────────────────────────

export interface ReserveSnapshot {
  enabled: boolean;
  regime: string;
  reservePct: number;
  reserveDollars: number;
  headroomDollars: number;
  floorCommitted: number;
  satelliteCommitted: number;
  minEntryPct: number;
  /** what set the reserve: the regime label's cash share, or the regime engine's risk budget */
  reserveReason: string;
  riskBudget: number | null;
  summary: string;
}

/** position_mgmt.satellite_headroom's dict, or null when it is not one. */
export function parseReserveSnapshot(raw: unknown): ReserveSnapshot | null {
  if (!raw || typeof raw !== "object") return null;
  const r = raw as Record<string, unknown>;
  if (typeof r.enabled !== "boolean" || !Number.isFinite(Number(r.headroom_dollars))) return null;
  return {
    enabled: r.enabled,
    regime: str(r.regime) || "NEUTRAL",
    reservePct: num(r.reserve_pct),
    reserveDollars: num(r.reserve_dollars),
    headroomDollars: Math.max(0, num(r.headroom_dollars)),
    floorCommitted: num(r.floor_committed),
    satelliteCommitted: num(r.satellite_committed),
    minEntryPct: num(r.min_entry_pct, 0.01),
    reserveReason: str(r.reserve_reason),
    riskBudget: r.risk_budget === null || r.risk_budget === undefined ? null : num(r.risk_budget),
    summary: str(r.summary),
  };
}

export type ReserveAction = "allow" | "trim" | "block";
export interface ReserveFit { action: ReserveAction; qty: number; value: number; reason: string }

const usd = (v: number): string => `$${Math.round(v).toLocaleString("en-US")}`;

export class ReserveBudget {
  private left: number;
  constructor(readonly snapshot: ReserveSnapshot | null, readonly equity: number) {
    this.left = snapshot ? snapshot.headroomDollars : Number.POSITIVE_INFINITY;
  }
  /** true while the reserve gates entries (a snapshot was read and the switch is on) */
  get active(): boolean { return !!this.snapshot && this.snapshot.enabled; }
  get remaining(): number { return this.active ? Math.max(0, this.left) : Number.POSITIVE_INFINITY; }

  /** A whole-share entry of `qty` at `price`: allowed, cut to the room left, or blocked when
   *  the cut would leave less than minEntryPct of equity (position_mgmt.fit_entry's rule). */
  fitShares(price: number, qty: number): ReserveFit {
    const value = qty * price;
    if (!this.active) return { action: "allow", qty, value, reason: "cash reserve not applied" };
    const room = this.remaining;
    if (value <= room + 0.005) return { action: "allow", qty, value, reason: `fits inside the cash reserve (${usd(room)} of room)` };
    if (room <= 0 || !(price > 0)) return { action: "block", qty: 0, value: 0, reason: "the cash reserve leaves no room for a new satellite position" };
    const cut = Math.floor(room / price);
    const minValue = Math.max(0, this.equity) * (this.snapshot ? this.snapshot.minEntryPct : 0.01);
    if (cut < 1 || cut * price < minValue) {
      return { action: "block", qty: 0, value: 0, reason: `only ${usd(room)} of room is left above the cash reserve, below the ${usd(Math.max(minValue, price))} smallest worthwhile position` };
    }
    return { action: "trim", qty: cut, value: cut * price, reason: `cut from ${usd(value)} to ${usd(cut * price)} so the cash reserve stays whole (${usd(room)} of room)` };
  }

  /** An option open whose cost is only known after contract selection: blocked once no room
   *  is left; otherwise allowed, with `capFor()` as the most it may commit. */
  fitOpen(): ReserveFit {
    if (!this.active) return { action: "allow", qty: 0, value: 0, reason: "cash reserve not applied" };
    return this.remaining > 0
      ? { action: "allow", qty: 0, value: this.remaining, reason: `up to ${usd(this.remaining)} of room above the cash reserve` }
      : { action: "block", qty: 0, value: 0, reason: "the cash reserve leaves no room for a new options position" };
  }

  /** The cash a sell-to-open may commit: the broker's figure, capped by the reserve's room. */
  capFor(brokerCash: number): number {
    return this.active ? Math.min(brokerCash, this.remaining) : brokerCash;
  }

  spend(value: number): void {
    if (this.active && Number.isFinite(value) && value > 0) this.left -= value;
  }

  /** Give back dollars the snapshot already counted as pending — a scale-in tranche
   *  spending its own reservation (position_mgmt counts unbought tranches as pending). */
  release(value: number): void {
    if (this.active && Number.isFinite(value) && value > 0) this.left += value;
  }
}

/** The fields of a scan trade the reserve reads and, on a trim, rewrites. */
export interface ReserveTradeLike {
  ticker?: string; price?: number; shares?: number; position_value?: number;
  instrument?: string; instrument_ticker?: string;
  etf_shares?: number | null; etf_price?: number | null; etf_position_value?: number | null;
}

/** Fit one executeTrades candidate: an option open is checked for room only (its cost is
 *  known after contract selection — the caller spends it then); a stock entry is fitted on
 *  its shares, an ETF substitution on the ETF's own shares and price (the order that is
 *  placed), with the stock fallback's share count cut to the same dollars. Spends the
 *  allowed or trimmed value. */
export function fitTradeToReserve(budget: ReserveBudget, trade: ReserveTradeLike, isOptions: boolean): ReserveFit {
  if (isOptions) return budget.fitOpen();
  const etf = trade.instrument === "etf" && !!trade.instrument_ticker
    && str(trade.instrument_ticker).toUpperCase() !== str(trade.ticker).toUpperCase()
    && num(trade.etf_shares) >= 1 && num(trade.etf_price) > 0;
  const fit = etf
    ? budget.fitShares(num(trade.etf_price), Math.floor(num(trade.etf_shares)))
    : budget.fitShares(num(trade.price), Math.floor(num(trade.shares)));
  if (fit.action === "trim") {
    if (etf) {
      trade.etf_shares = fit.qty;
      trade.etf_position_value = fit.value;
      const p = num(trade.price);
      trade.shares = p > 0 ? Math.floor(fit.value / p) : 0;
    } else {
      trade.shares = fit.qty;
      trade.position_value = fit.value;
    }
  }
  if (fit.action !== "block") budget.spend(fit.value);
  return fit;
}

/** One plain-English audit line for a trim or block. */
export function reserveAuditLine(ticker: string, fit: ReserveFit, snap: ReserveSnapshot | null): string {
  if (snap && snap.reserveReason) return `${ticker}: ${fit.reason} — ${snap.reserveReason}`;
  const pct = snap ? Math.round(snap.reservePct * 100) : 0;
  return `${ticker}: ${fit.reason} — the ${snap ? snap.regime : "?"} market keeps ${pct}% of equity in cash`;
}

// ── 2. Capital rotation ────────────────────────────────────────────────────

export interface RotationRow {
  sell: { ticker: string; qty: number; marketValue: number; pnlPct: number; state: string };
  buy: { ticker: string; underlying: string; price: number; score: number; instrument: string; value: number };
  summary: string;
  gainPct: number;
  evHoldPct: number;
  evCandidatePct: number;
  taxNow: number;
  costDollars: number;
}

/** The scan's rotation_plan.rotations, or [] (absent plan, error, malformed rows dropped). */
export function parseRotationPlan(raw: unknown): RotationRow[] {
  const rows = raw && typeof raw === "object" ? (raw as { rotations?: unknown }).rotations : null;
  if (!Array.isArray(rows)) return [];
  const out: RotationRow[] = [];
  for (const r of rows) {
    if (!r || typeof r !== "object") continue;
    const row = r as Record<string, unknown>;
    const s = (row.sell || {}) as Record<string, unknown>;
    const b = (row.buy || {}) as Record<string, unknown>;
    const sell = { ticker: str(s.ticker).toUpperCase(), qty: num(s.qty), marketValue: num(s.market_value), pnlPct: num(s.pnl_pct), state: str(s.state) };
    const buy = { ticker: str(b.ticker).toUpperCase(), underlying: str(b.underlying || b.ticker).toUpperCase(), price: num(b.price), score: num(b.score), instrument: str(b.instrument) || "stock", value: num(b.value) };
    if (!sell.ticker || !buy.ticker || sell.ticker === buy.ticker) continue;
    out.push({ sell, buy, summary: str(row.summary), gainPct: num(row.gain_pct), evHoldPct: num(row.ev_hold_pct), evCandidatePct: num(row.ev_candidate_pct), taxNow: num(row.tax_now), costDollars: num(row.cost_dollars) });
  }
  return out;
}

export interface BookPosition { symbol?: string; qty?: string | number; side?: string; market_value?: string | number; asset_class?: string; unrealized_plpc?: string | number }
export interface BookOrder { symbol?: string }
export interface PreflightInput {
  positions: readonly BookPosition[];
  openOrders: readonly BookOrder[];
  floorTickers: ReadonlySet<string>;
  /** the WS monitor's view of the sell ticker (pendingExit = an exit order is working) */
  monitored?: { pendingExit?: boolean } | undefined;
  exiting: boolean;
  etHour: number;
  marketOpen: boolean;
}
export interface Preflight { ok: boolean; reason: string; sellQty: number; sellValue: number; buyQty: number; buyValue: number; pnlPct: number }

const OCC = /^[A-Z][A-Z0-9.]{0,5}\d{6}[CP]\d{8}$/;

/** Is a planned swap still safe to place against the live book? */
export function rotationPreflight(row: RotationRow, input: PreflightInput): Preflight {
  const fail = (reason: string): Preflight => ({ ok: false, reason, sellQty: 0, sellValue: 0, buyQty: 0, buyValue: 0, pnlPct: 0 });
  const [from, to] = ROTATION_WINDOW_ET;
  if (!input.marketOpen || !(input.etHour >= from && input.etHour < to)) return fail("outside the 09:45-15:54 ET rotation window");
  const s = row.sell.ticker, b = row.buy.ticker;
  if (input.floorTickers.has(s) || OCC.test(s)) return fail(`${s} is a floor or options position, never rotated`);
  const held = input.positions.find((p) => str(p.symbol).toUpperCase() === s);
  const qty = held ? num(held.qty) : 0;
  if (!held || qty <= 0 || str(held.side || "long") !== "long" || str(held.asset_class || "us_equity") !== "us_equity") return fail(`${s} is no longer held as a long stock position`);
  if (input.monitored?.pendingExit || input.exiting) return fail(`${s} already has an exit working`);
  if (input.openOrders.some((o) => str(o.symbol).toUpperCase() === s)) return fail(`${s} has an order working at the broker`);
  if (input.positions.some((p) => str(p.symbol).toUpperCase() === b)) return fail(`${b} is already held`);
  if (input.openOrders.some((o) => str(o.symbol).toUpperCase() === b)) return fail(`${b} already has an order working`);
  if (!(row.buy.price > 0)) return fail(`no price for ${b}`);
  const sellValue = Math.abs(num(held.market_value));
  const buyValue = Math.min(row.buy.value > 0 ? row.buy.value : sellValue, sellValue);
  const buyQty = Math.floor(buyValue / row.buy.price);
  if (buyQty < 1) return fail(`${b} at $${row.buy.price.toFixed(2)} is more than the ${usd(buyValue)} the sale frees`);
  return { ok: true, reason: "ok", sellQty: qty, sellValue, buyQty, buyValue, pnlPct: num(held.unrealized_plpc) * 100 };
}

// ── 3. Decision log ────────────────────────────────────────────────────────

export function positionMgmtLogPath(env: Record<string, string | undefined> = process.env): string {
  const dir = env.DATA_DIR || (fs.existsSync("/data") ? "/data/voltrade" : "/tmp");
  return path.join(dir, POSITION_MGMT_LOG_FILE);
}

const dedupSeen = new Map<string, number>();

/** Append one row ({ts, kind, action, ticker, summary, ...}) to the shared JSONL. Returns
 *  false when deduplicated (same key inside dedupMs) or the write failed (logged). */
export function appendPositionMgmtLog(
  row: { kind: string; action: string; ticker: string; summary: string } & Record<string, unknown>,
  opts: { dedupKey?: string; dedupMs?: number; file?: string; nowMs?: number } = {},
): boolean {
  const now = opts.nowMs ?? Date.now();
  if (opts.dedupKey && opts.dedupMs) {
    const last = dedupSeen.get(opts.dedupKey);
    if (last !== undefined && now - last < opts.dedupMs) return false;
    dedupSeen.set(opts.dedupKey, now);
    if (dedupSeen.size > 5000) for (const k of Array.from(dedupSeen.keys()).slice(0, 2500)) dedupSeen.delete(k);
  }
  const file = opts.file ?? positionMgmtLogPath();
  try {
    if (fs.existsSync(file) && fs.statSync(file).size > POSITION_MGMT_LOG_MAX_BYTES) fs.renameSync(file, `${file}.1`);
    fs.appendFileSync(file, JSON.stringify({ ts: new Date(now).toISOString(), ...row, ticker: row.ticker.toUpperCase() }) + "\n");
    return true;
  } catch (e: unknown) {
    console.warn(`[position-mgmt] log not written (${row.kind}/${row.action} ${row.ticker}): ${e instanceof Error ? e.message : String(e)}`);
    return false;
  }
}
