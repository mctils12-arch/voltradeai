// openingCounterfactual.ts — what waiting out the 09:30–09:45 ET opening window
// cost or saved, per exit and per day.
//
// MEASUREMENT ONLY (2026-11-12; research/open_questions.md, "opening-window
// exits: the counterfactual to log"). options_manager.py sends no option exit
// before 09:45 ET. Its rollback trigger compares what each held exit got after
// 09:45 with the quote at the moment a rule first fired on it. Two records
// make that readable, and this joins them once a day inside the daily line:
//   - the FIRST-FIRED file (options_manager._note_opening_first_fired): per ET
//     day, one record per exit the window held (a contract, or a group's legs
//     joined by ","): the rule, the time, and per leg the side its close takes,
//     the quantity and the quote (bid / ask / mid) at that moment;
//   - the option FILL LEDGER: every row written for a close of that contract
//     later the same day carries the record ("first_fired").
//
//   first fired   exits the window held that day
//   sent          a close order for it was accepted later that day: a ledger
//                 row carrying the record or, when there is none, a row of
//                 another order path (the Node kill ladder, a manual close)
//                 for the same contract and side after the first fire
//   never sent    none was. The rule no longer held after 09:45, which is what
//                 the window is for. `refused` of them had a close the broker
//                 refused after the window: not the rule's doing.
//   waiting cost  per sent exit, over the legs sent:
//                   sum of s x (after − first) x qty x 100,
//                 s = +1 for a buy-to-close and −1 for a sell-to-close, in
//                 dollars and as a % of sum(first x qty x 100).
//                 POSITIVE = WAITING COST MONEY: the later order paid more to
//                 buy back, or received less on a sale, than the first-fired
//                 reference. Negative = waiting saved money.
//                 `after` is the average fill of the exit's orders (each
//                 followed through its replacements), or, with no fill, the
//                 last limit sent. `first` is the first-fired MID (vs_mid) or
//                 the first-fired TOUCH, the ask for a buy and the bid for a
//                 sell (vs_touch).
//
// HOW TO READ IT. The first-fired quote is a QUOTE in the widest-spread minutes
// of the day, not a fill. An order sent then would have paid somewhere between
// its mid and its touch, so the two comparisons bracket the answer:
//   vs_mid    flatters the EARLY exit: it assumes a fill at the mid, and so
//             overstates what waiting cost;
//   vs_touch  flatters WAITING: it assumes the early order paid the whole
//             quoted spread (live 2026-10-29: the XLF put's limit at a quoted
//             5.00 ask filled at 1.61).
// Waiting clearly cost money only where vs_touch is positive too, and clearly
// saved money only where vs_mid is negative too. Exits never sent have no
// price and are only counted, so the cost over the sent exits alone leaves
// out the exits the window spared. An exit priced at a limit that never
// filled is counted apart (by_limit).
// Nothing is inferred: an exit with a leg that has no fill and no limit is
// sent but not priced, and one with no two-sided first-fired quote has no
// vs_mid. One exit is one contract (or group) per day, so `first_fired` can be
// smaller than the block count next to it, which counts (contract, rule).
import { zonedParts } from "./jobHeartbeat";

const ET = "America/New_York";
export const OPENING_FIRST_FIRED_FILENAME = "voltrade_option_opening_first_fired.json";
/** Exits listed on one line (a day's are a handful; the rest are counted). */
export const OPENING_COUNTERFACTUAL_MAX_EXITS = 40;
/** An order replaced more often than this is not followed further. */
const REPLACE_HOPS = 12;
export const WAITING_COST_SIGN = "waiting cost > 0: the later order paid more (buy-to-close) or received less (sell-to-close) than the first-fired reference; < 0: waiting saved money";

const r2 = (x: number) => Math.round(x * 100) / 100;
const r4 = (x: number) => Math.round(x * 10_000) / 10_000;

function rec(x: unknown): Record<string, unknown> {
  return x && typeof x === "object" && !Array.isArray(x) ? (x as Record<string, unknown>) : {};
}

function num(x: unknown): number | null {
  if (x === null || x === undefined || x === "") return null;
  const v = typeof x === "number" ? x : parseFloat(String(x));
  return Number.isFinite(v) ? v : null;
}

function str(x: unknown): string {
  return typeof x === "string" ? x : x === null || x === undefined ? "" : String(x);
}

function msOf(x: unknown): number | null {
  const t = Date.parse(str(x));
  return Number.isFinite(t) ? t : null;
}

function etDay(x: unknown): string | null {
  const t = msOf(x);
  return t === null ? null : zonedParts(t, ET).date;
}

/** What a fill-ledger row carries for a close the window held earlier that day. */
export interface FirstFiredLink { key: string; at: string; rule: string; kind: string; bid: number | null; ask: number | null; mid: number | null }

export function parseFirstFiredLink(x: unknown): FirstFiredLink | null {
  const o = rec(x);
  const key = str(o.key), at = str(o.at);
  if (!key || !at) return null;
  return { key, at, rule: str(o.rule), kind: str(o.kind), bid: num(o.bid), ask: num(o.ask), mid: num(o.mid) };
}

/** The fields of a fill-ledger row read here (dailyLine's LedgerRow has them). */
export interface CounterfactualRow { order_id: string; symbol: string; side: string; qty: number | null; ts?: string; limit?: number | null; first_fired?: FirstFiredLink }

/** The three optional fields dailyLine.parseLedger adds to a row, each only when the stored row has it (a
 *  null limit and a row with no record add nothing). The eight fields every other reader uses are untouched. */
export function ledgerRowExtras(o: Record<string, unknown>): Pick<CounterfactualRow, "ts" | "limit" | "first_fired"> {
  const out: Pick<CounterfactualRow, "ts" | "limit" | "first_fired"> = {};
  const ts = str(o.ts), limit = num(o.limit), ff = parseFirstFiredLink(o.first_fired);
  if (ts) out.ts = ts;
  if (limit !== null) out.limit = limit;
  if (ff) out.first_fired = ff;
  return out;
}

export interface OpeningExit {
  /** the contract, or a group's legs joined by "," */
  key: string;
  kind: string;
  /** the rule that fired first */
  rule: string;
  first_fired_at: string | null;
  /** scans inside the window on which a rule fired for it */
  fires: number | null;
  legs: number;
  sent: boolean;
  /** closes the broker refused after the window */
  refused: number;
  /** "link": a ledger row carries the record; "match": only rows of another order path were found */
  via: "link" | "match" | null;
  sent_at: string | null;
  /** minutes from the first fire to the first order */
  delay_min: number | null;
  orders: number;
  /** what `after` is: the fills, or (no fill) the last limit sent */
  priced_by: "fill" | "limit" | null;
  /** a single leg only: the side its close takes, the first-fired quote, the price after */
  side: string | null;
  first_bid: number | null;
  first_ask: number | null;
  first_mid: number | null;
  after: number | null;
  /** waiting cost against the first-fired mid (> 0 = waiting cost money) */
  cost_usd: number | null;
  cost_pct: number | null;
  /** waiting cost against the first-fired touch */
  touch_cost_usd: number | null;
  touch_cost_pct: number | null;
  /** why a sent exit is not priced, or what a never-sent one met */
  note: string | null;
}

export interface WaitingCost { priced: number; mean_pct: number | null; median_pct: number | null; usd: number | null }

export interface OpeningCounterfactual {
  first_fired: number;
  sent: number;
  never_sent: number;
  /** of never_sent: a close was refused by the broker after the window */
  refused: number;
  /** sent exits by what they are priced at; unpriced = neither a fill nor a limit */
  by_fill: number;
  by_limit: number;
  unpriced: number;
  vs_mid: WaitingCost;
  vs_touch: WaitingCost;
  median_delay_min: number | null;
  sign: string;
  exits: OpeningExit[];
  exits_omitted: number;
  reason: string | null;
}

interface Leg { symbol: string; side: string; qty: number | null; bid: number | null; ask: number | null; mid: number | null }
interface Held { key: string; kind: string; rule: string; at: string | null; atMs: number | null; fires: number | null; refused: number; legs: Leg[]; rows: CounterfactualRow[]; linked: boolean }

function mean(xs: readonly number[]): number | null {
  return xs.length ? xs.reduce((s, x) => s + x, 0) / xs.length : null;
}

function median(xs: readonly number[]): number | null {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

const rounded = (x: number | null) => (x === null ? null : r2(x));

/** The order and every order that replaced it (a ladder step is a replace: a new id). */
function chain(byId: ReadonlyMap<string, Record<string, unknown>>, id: string): Array<Record<string, unknown>> {
  const out: Array<Record<string, unknown>> = [];
  let cur = id;
  for (let hop = 0; hop <= REPLACE_HOPS && cur; hop++) {
    const o = byId.get(cur);
    if (!o) break;
    out.push(o);
    cur = str(o.replaced_by);
  }
  return out;
}

/** What this order filled of `symbol` (an mleg order carries its legs' fills). */
function fillOf(o: Record<string, unknown>, symbol: string): { qty: number; price: number } | null {
  const legs = Array.isArray(o.legs) ? o.legs.map(rec) : [];
  const src = legs.length ? legs.find((l) => str(l.symbol) === symbol) : str(o.symbol) === symbol ? o : undefined;
  if (!src) return null;
  const qty = num(src.filled_qty), price = num(src.filled_avg_price);
  return qty !== null && qty > 0 && price !== null && price > 0 ? { qty, price } : null;
}

function priceExit(e: Held, byId: ReadonlyMap<string, Record<string, unknown>> | null): OpeningExit {
  const one = e.kind !== "group" && e.legs.length === 1 ? e.legs[0] : null;
  const out: OpeningExit = {
    key: e.key, kind: e.kind, rule: e.rule, first_fired_at: e.at, fires: e.fires, legs: e.legs.length,
    sent: e.rows.length > 0, refused: e.refused, via: null, sent_at: null, delay_min: null, orders: 0, priced_by: null,
    side: one ? one.side || null : null, first_bid: one ? one.bid : null, first_ask: one ? one.ask : null, first_mid: one ? one.mid : null,
    after: null, cost_usd: null, cost_pct: null, touch_cost_usd: null, touch_cost_pct: null, note: null,
  };
  if (!out.sent) {
    if (e.refused > 0) out.note = `${e.refused} close(s) refused by the broker after the window`;
    return out;
  }
  out.via = e.linked ? "link" : "match";
  out.orders = new Set(e.rows.map((r) => r.order_id)).size;
  const first = e.rows.map((r) => msOf(r.ts)).filter((t): t is number => t !== null).sort((a, b) => a - b)[0];
  if (first !== undefined) {
    out.sent_at = new Date(first).toISOString();
    if (e.atMs !== null) out.delay_min = Math.round(((first - e.atMs) / 60_000) * 10) / 10;
  }
  let costMid = 0, baseMid = 0, costTouch = 0, baseTouch = 0;
  let okAfter = true, okMid = true, okTouch = true, anyLimit = false;
  for (const symbol of Array.from(new Set(e.rows.map((r) => r.symbol)))) {
    const rows = e.rows.filter((r) => r.symbol === symbol);
    const leg = e.legs.find((l) => l.symbol === symbol);
    const side = (leg?.side || rows[0].side).toLowerCase();
    const s = side === "buy" ? 1 : side === "sell" ? -1 : 0;
    const fills = new Map<string, { qty: number; price: number }>();
    if (byId) {
      for (const r of rows) {
        for (const o of chain(byId, r.order_id)) {
          const f = fillOf(o, symbol);
          if (f) fills.set(str(o.id), f);
        }
      }
    }
    let qty = 0, notional = 0;
    for (const f of fills.values()) { qty += f.qty; notional += f.price * f.qty; }
    let after: number | null = qty > 0 ? notional / qty : null;
    if (after === null) {
      const resting = [...rows].reverse().find((r) => (r.limit ?? 0) > 0 && (r.qty ?? 0) > 0);
      if (resting) { after = resting.limit ?? null; qty = resting.qty ?? 0; anyLimit = true; }
    }
    if (after === null || s === 0) { okAfter = false; continue; }
    if (one) out.after = r4(after);
    const mid = leg?.mid ?? null;
    if (mid !== null && mid > 0) { costMid += s * (after - mid) * qty * 100; baseMid += mid * qty * 100; } else okMid = false;
    const touch = (side === "buy" ? leg?.ask : leg?.bid) ?? null;
    if (touch !== null && touch > 0) { costTouch += s * (after - touch) * qty * 100; baseTouch += touch * qty * 100; } else okTouch = false;
  }
  if (!okAfter) {
    out.note = byId ? "an order of it has no fill and no limit recorded" : "no limit recorded, and the day's orders are unreadable";
    return out;
  }
  out.priced_by = anyLimit ? "limit" : "fill";
  if (okMid && baseMid > 0) { out.cost_usd = r2(costMid); out.cost_pct = r2((costMid / baseMid) * 100); }
  else out.note = "no two-sided first-fired quote on a leg sent";
  if (okTouch && baseTouch > 0) { out.touch_cost_usd = r2(costTouch); out.touch_cost_pct = r2((costTouch / baseTouch) * 100); }
  return out;
}

/** The day's block, or null when no exit first fired in the window that day: no record in the first-fired file
 *  and no ledger row carrying one (so a quiet day's line is the line it always was). `orders` is the daily line's
 *  own list (nested: an mleg parent carries its legs), null when it could not be read. */
export function openingCounterfactual(doc: unknown, day: string, rows: readonly CounterfactualRow[], orders: readonly unknown[] | null): OpeningCounterfactual | null {
  const days = rec(doc).days;
  const fileOk = !!days && typeof days === "object" && !Array.isArray(days);
  const linked = rows.filter((r) => r.first_fired && etDay(r.first_fired.at) === day);
  const held = new Map<string, Held>();
  for (const [key, raw] of Object.entries(rec(rec(days)[day]))) {
    if (!raw || typeof raw !== "object" || Array.isArray(raw)) continue;      // not a record
    const o = rec(raw);
    const at = str(o.first_fired_at) || null;
    const legs: Leg[] = (Array.isArray(o.legs) ? o.legs : []).map(rec).filter((l) => str(l.symbol))
      .map((l) => ({ symbol: str(l.symbol), side: str(l.side).toLowerCase(), qty: num(l.qty), bid: num(l.bid), ask: num(l.ask), mid: num(l.mid) }));
    held.set(key, { key, kind: str(o.kind) || (legs.length > 1 ? "group" : "single_leg"), rule: str(o.rule), at, atMs: msOf(at),
      fires: num(o.fires), refused: num(o.refused) ?? 0, legs, rows: [], linked: false });
  }
  for (const r of linked) {
    const ff = r.first_fired;
    if (!ff) continue;
    let e = held.get(ff.key);
    if (!e) {
      // a row carries a record the file no longer holds: the exit is rebuilt from the row
      e = { key: ff.key, kind: ff.kind || (ff.key.includes(",") ? "group" : "single_leg"), rule: ff.rule, at: ff.at, atMs: msOf(ff.at),
        fires: null, refused: 0, legs: [], rows: [], linked: false };
      held.set(ff.key, e);
    }
    if (!e.legs.some((l) => l.symbol === r.symbol)) e.legs.push({ symbol: r.symbol, side: r.side.toLowerCase(), qty: r.qty, bid: ff.bid, ask: ff.ask, mid: ff.mid });
    e.rows.push(r);
    e.linked = true;
  }
  if (!held.size) return null;
  // a close sent by another order path writes its row without the record: the same contract, the side the close
  // takes, after the first fire and on the same ET day. Only for an exit no row carries the record of: where the
  // manager's own orders are known, they are the exit.
  for (const e of held.values()) {
    if (e.atMs === null || e.linked) continue;
    for (const r of rows) {
      if (r.first_fired) continue;
      const t = msOf(r.ts);
      if (t === null || t <= e.atMs || zonedParts(t, ET).date !== day) continue;
      if (e.legs.some((l) => l.symbol === r.symbol && l.side === r.side.toLowerCase())) e.rows.push(r);
    }
  }
  let byId: Map<string, Record<string, unknown>> | null = null;
  if (orders) {
    byId = new Map();
    for (const raw of orders) {
      const o = rec(raw);
      if (str(o.id)) byId.set(str(o.id), o);
    }
  }
  const index = byId;
  const exits = Array.from(held.values()).map((e) => priceExit(e, index))
    .sort((a, b) => (a.first_fired_at ?? "").localeCompare(b.first_fired_at ?? "") || a.key.localeCompare(b.key));
  const sent = exits.filter((x) => x.sent);
  const cost = (pct: (x: OpeningExit) => number | null, usd: (x: OpeningExit) => number | null): WaitingCost => {
    const priced = sent.filter((x) => pct(x) !== null);
    const pcts = priced.map((x) => pct(x) ?? 0);
    return { priced: priced.length, mean_pct: rounded(mean(pcts)), median_pct: rounded(median(pcts)),
      usd: priced.length ? r2(priced.reduce((s, x) => s + (usd(x) ?? 0), 0)) : null };
  };
  const reasons: string[] = [];
  if (!fileOk) reasons.push("the first-fired file is unreadable: exits never sent are not counted");
  if (!orders) reasons.push("the day's orders are unreadable: no fill is joined, a sent exit is priced at its last limit");
  const delays = sent.map((x) => x.delay_min).filter((d): d is number => d !== null);
  const medDelay = median(delays);
  return {
    first_fired: exits.length, sent: sent.length, never_sent: exits.length - sent.length,
    refused: exits.filter((x) => !x.sent && x.refused > 0).length,
    by_fill: sent.filter((x) => x.priced_by === "fill").length, by_limit: sent.filter((x) => x.priced_by === "limit").length,
    unpriced: sent.filter((x) => x.priced_by === null).length,
    vs_mid: cost((x) => x.cost_pct, (x) => x.cost_usd), vs_touch: cost((x) => x.touch_cost_pct, (x) => x.touch_cost_usd),
    median_delay_min: medDelay === null ? null : Math.round(medDelay * 10) / 10,
    sign: WAITING_COST_SIGN,
    exits: exits.slice(0, OPENING_COUNTERFACTUAL_MAX_EXITS), exits_omitted: Math.max(0, exits.length - OPENING_COUNTERFACTUAL_MAX_EXITS),
    reason: reasons.length ? reasons.join("; ") : null,
  };
}

const signed = (w: WaitingCost) => (w.median_pct === null ? "n/a" : `${w.median_pct >= 0 ? "+" : ""}${w.median_pct.toFixed(2)}%`);

/** The daily line's audit clause for the block ("" without one). The cost is the median over the priced exits,
 *  signed: + = waiting cost money. */
export function openingCounterfactualText(c: OpeningCounterfactual | null | undefined): string {
  if (!c || !(c.first_fired > 0)) return "";
  const never = `${c.never_sent} never${c.refused > 0 ? ` (${c.refused} refused)` : ""}`;
  const cost = c.vs_mid.priced || c.vs_touch.priced
    ? `waiting cost median ${signed(c.vs_mid)} vs mid (${c.vs_mid.priced} priced), ${signed(c.vs_touch)} vs touch (${c.vs_touch.priced} priced)`
    : "waiting cost n/a";
  return `opening-window exits: ${c.first_fired} first fired, ${c.sent} sent, ${never}; ${cost}`;
}
