// Bot page — the owner's four actions ([PRODUCT] 2026-11-25; owner: "i like
// the owner buttons"): run a scan now, reset the circuit breaker, close a
// position, place a manual order. Each has had an owner-only route for
// months (POST /api/bot/run-now, /circuit-breaker/reset, /close, /trade) and
// no control on the page.
//
// This file decides, from data the page already holds, what each control
// SAYS before it acts: the state it will change, the exact consequence, the
// reasons the server would refuse (said before the click, not after), and
// what the answer means. Nothing here sends anything. Pure; pinned by
// ownerActionsView.test.ts. The consequences are worded from the routes'
// own code in server/bot.ts and from server/manualOrderCheck.ts.
import { fmtClockET, fmtMoney, fmtOcc } from "./format";
import { optionManagement } from "../../../shared/positionLevels";

const fin = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const str = (v: unknown): string | null => (typeof v === "string" && v.trim() ? v.trim() : null);
const obj = (v: unknown): Record<string, unknown> | null => (v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : null);
const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;
/** HH:MM in New York */
const hm = (iso: string) => fmtClockET(iso).slice(0, 5);

// ── circuit breaker ─────────────────────────────────────────────────────────
/** the loop trips the breaker at this many stock stop-losses in a row (server/bot.ts; pinned by the test) */
export const BREAKER_STOPS = 3;
export interface BreakerView { known: boolean; active: boolean; until: string | null; stops: number | null; sentence: string; canReset: boolean }
/** /api/bot/status → the breaker's state in one sentence. */
export function breakerView(status: unknown): BreakerView {
  const s = obj(status);
  const known = !!s && (typeof s.circuitBreakerActive === "boolean" || fin(s.consecutiveStopLosses));
  if (!s || !known) return { known: false, active: false, until: null, stops: null, sentence: "Circuit breaker: state not reported by this server", canReset: false };
  const active = s.circuitBreakerActive === true;
  const until = str(s.circuitBreakerUntil);
  const stops = fin(s.consecutiveStopLosses) ? s.consecutiveStopLosses : null;
  const streak = stops === null ? null : stops > 0 ? `${plural(stops, "stop-loss", "stop-losses")} in a row (it trips at ${BREAKER_STOPS})` : "no stop-loss streak";
  const sentence = active
    ? `Circuit breaker ON${until ? ` until ${hm(until)} ET` : ""}: new entries are paused${streak ? ` · ${streak}` : ""}`
    : `Circuit breaker off${streak ? ` · ${streak}` : ""}`;
  return { known: true, active, until, stops, sentence, canReset: active || (stops ?? 0) > 0 };
}
export const BREAKER_RESET_CONFIRM = "Ends the pause now and sets the stop-loss streak back to 0. New entries can resume on the loop's next pass. Exits were never paused.";

// ── scan now ────────────────────────────────────────────────────────────────
/** what "scan now" does, by market state (null = not known) */
export function scanConfirm(marketOpen: boolean | null): string {
  const base = "Starts a full scan now, outside its schedule. It scores the same universe and can place orders exactly as a scheduled scan does.";
  if (marketOpen === false) return `${base} The market is closed: the order gate refuses orders until it opens.`;
  return marketOpen === null ? `${base} Market state not known.` : base;
}
/** the route answers {message}; a second click while one runs says so */
export function scanResult(body: unknown): string {
  const m = str(obj(body)?.message);
  if (!m) return "The server answered without a message.";
  return /already running/i.test(m) ? "A scan is already running: nothing new was started." : "Scan started. Its rows appear in the timeline as it runs.";
}

// ── close a position ────────────────────────────────────────────────────────
export interface CloseChoice {
  symbol: string;
  /** "NVDA · 12 sh · $2,174" / "AAL 11.5P 31-Dec · 1 ct short · $135" */
  label: string;
  isOption: boolean; qty: number;
  warnings: string[];
}
const isOpt = (p: Record<string, unknown>) => p.isOption === true || String(p.asset_class ?? "").toLowerCase().includes("option");
/** /api/bot/positions → the positions that can be closed, with what closing each one means. `baseSymbols` are the
 *  base holding's tickers (the floor-tracking report's config), any case. */
export function closeChoices(positions: unknown, baseSymbols: ReadonlyArray<string | null | undefined> = []): CloseChoice[] {
  if (!Array.isArray(positions)) return [];
  const base = new Set(baseSymbols.filter((s): s is string => !!s).map((s) => s.toUpperCase()));
  const rows = positions.map(obj).filter((p): p is Record<string, unknown> => !!p && !!str(p.ticker) && fin(p.qty) && p.qty !== 0);
  // legs of one multi-leg strategy on one underlying are managed as a unit (the Book's own grouping)
  const groupKey = (p: Record<string, unknown>) => { const m = obj(p.optionMeta); return `${str(m?.underlyingTicker) ?? ""}_${str(m?.strategy) ?? ""}`; };
  const legs = new Map<string, number>();
  for (const p of rows) if (isOpt(p) && optionManagement(str(obj(p.optionMeta)?.strategy)) === "group") legs.set(groupKey(p), (legs.get(groupKey(p)) ?? 0) + 1);
  return rows.map((p) => {
    const symbol = (str(p.ticker) as string).toUpperCase(), qty = p.qty as number, option = isOpt(p);
    const mv = fin(p.marketValue) ? Math.abs(p.marketValue) : null;
    const size = option ? `${plural(Math.abs(qty), "ct", "ct")} ${qty < 0 ? "short" : "long"}` : `${Math.abs(qty)} sh${qty < 0 ? " short" : ""}`;
    const warnings: string[] = [];
    if (base.has(symbol)) warnings.push("Part of the base holding: the bot buys it back at its next rebalance.");
    if (option && qty < 0) warnings.push("A short option: closing buys it back at the market price.");
    if (option && (legs.get(groupKey(p)) ?? 0) > 1) warnings.push(`One of ${legs.get(groupKey(p))} legs the bot manages together: closing it alone leaves the rest open.`);
    if (option) warnings.push("Option quotes here are indicative: a market order can fill far from the price shown.");
    return { symbol, label: `${option ? fmtOcc(symbol) : symbol} · ${size}${mv !== null ? ` · ${fmtMoney(mv)}` : ""}`, isOption: option, qty, warnings };
  });
}
/** the sentence above the confirm button */
export function closeConfirm(c: CloseChoice, marketOpen: boolean | null): string {
  const closed = marketOpen === false ? (c.isOption ? " The market is closed: the order gate refuses it until the open." : " The market is closed: the order gate refuses it, except a held stock inside an extended session.") : "";
  return `Closes ${c.label}: one market order for the whole position, sent now.${closed}`;
}

// ── manual order ────────────────────────────────────────────────────────────
/** the server's position-size ceiling for a manual buy (MAX_POSITION_SIZE in server/bot.ts; pinned by the test) */
export const MANUAL_POSITION_LIMIT = 0.2;
export const STOCK_SYMBOL_RE = /^[A-Z][A-Z.]{0,5}$/;
export interface OrderDraft { ticker: string; side: "buy" | "sell"; qty: string }
export interface OrderContext {
  /** the instrument's last price and when; null = none read */
  price: number | null; priceAt?: string | null;
  /** shares held now (0 = none; null = positions not loaded) */
  heldQty: number | null;
  equity: number | null; killSwitch: boolean; marketOpen: boolean | null;
}
export interface OrderReview {
  ok: boolean;
  /** why the order cannot be sent as written (each is a refusal the server would give, or an unusable field) */
  problems: string[];
  /** true but worth knowing before confirming */
  notes: string[];
  symbol: string; side: "buy" | "sell"; qty: number | null; value: number | null;
  /** "BUY 10 AAPL at market · about $2,314 at the last trade ($231.42) · 2.7% of the account" */
  sentence: string;
}
/** Review a draft against what the page knows. Stocks and ETFs, market orders, whole shares: what the route can send. */
export function reviewOrder(d: OrderDraft, ctx: OrderContext): OrderReview {
  const symbol = d.ticker.trim().toUpperCase();
  const problems: string[] = [], notes: string[] = [];
  const q = /^\d+$/.test(d.qty.trim()) ? Number(d.qty.trim()) : NaN;
  const qty = Number.isFinite(q) && q > 0 ? q : null;
  if (!symbol) problems.push("Enter a ticker.");
  else if (!STOCK_SYMBOL_RE.test(symbol)) problems.push("Stocks and ETFs only here: a ticker of up to six letters. Options are not placed from this form.");
  if (qty === null) problems.push("Quantity must be a whole number of shares, 1 or more.");
  if (ctx.killSwitch) problems.push("The kill switch is latched: the server refuses every manual order until it is cleared. Closing a position still works.");
  const value = qty !== null && ctx.price !== null && ctx.price > 0 ? qty * ctx.price : null;
  if (d.side === "sell") {
    if (ctx.heldQty === null) problems.push("Positions are not loaded, so the holding cannot be checked.");
    else if (ctx.heldQty <= 0) problems.push(`You hold no ${symbol || "shares of it"}: this route does not sell short.`);
    else if (qty !== null && qty > ctx.heldQty) problems.push(`You hold ${ctx.heldQty} ${symbol}: a sell cannot exceed it.`);
    if (ctx.marketOpen === false) notes.push("The market is closed: the order gate refuses the sale unless an extended session is open.");
  } else {
    if (symbol && STOCK_SYMBOL_RE.test(symbol) && ctx.price === null) problems.push(`No last price for ${symbol}: the server's size check refuses a buy it cannot price.`);
    if (value !== null && ctx.equity !== null && ctx.equity > 0 && value / ctx.equity > MANUAL_POSITION_LIMIT) {
      problems.push(`About ${Math.round((value / ctx.equity) * 100)}% of the account: the server refuses a manual buy over ${MANUAL_POSITION_LIMIT * 100}%.`);
    }
    if (ctx.marketOpen === false) problems.push("The market is closed: the order gate refuses a buy until it opens.");
  }
  if (ctx.marketOpen === null) notes.push("Market state not known: the server decides when the order arrives.");
  notes.push("A market order fills at the next available price, which can differ from the last trade.");
  const share = value !== null && ctx.equity !== null && ctx.equity > 0 ? ` · ${((value / ctx.equity) * 100).toFixed(1)}% of the account` : "";
  const priced = value !== null ? ` · about ${fmtMoney(value)} at the last trade ($${(ctx.price as number).toFixed(2)}${ctx.priceAt ? `, ${hm(ctx.priceAt)} ET` : ""})${share}` : ctx.price === null ? " · no last price" : "";
  return {
    ok: problems.length === 0, problems, notes, symbol, side: d.side, qty, value,
    sentence: `${d.side.toUpperCase()} ${qty ?? "?"} ${symbol || "?"} at market${priced}`,
  };
}
/** POST /api/bot/trade's answer (the broker's order object) in one sentence. */
export function orderResult(body: unknown): string {
  const o = obj(body);
  const id = str(o?.id), status = str(o?.status);
  if (!o || (!id && !status)) return "The server answered, but not with an order: check the timeline and the working orders.";
  return `Order sent${status ? `: the broker says "${status.replace(/_/g, " ")}"` : ""}${id ? ` · id ${id.slice(0, 8)}` : ""}. It appears under working orders until it fills.`;
}
/** the last price from GET /api/bot/bars/:ticker?timeframe=1Min&limit=1 ({bars:[{time, close}]}); null when there is none */
export function lastPriceFromBars(body: unknown): { price: number; at: string } | null {
  const bars = obj(body)?.bars;
  const b = Array.isArray(bars) && bars.length ? obj(bars[bars.length - 1]) : null;
  if (!b || !fin(b.close) || b.close <= 0 || !fin(b.time)) return null;
  return { price: b.close, at: new Date(b.time * 1000).toISOString() };
}

/** The block's one line, closed: the breaker's state and what is on offer. */
export function ownerActionsSentence(b: BreakerView, positions: number | null): string {
  return `${b.sentence} · scan now, close ${positions === null ? "a position" : `one of ${plural(positions, "position")}`}, or place an order: each asks before it acts`;
}

/** A failed action in the server's own words. apiRequest raises "<status>: <body>"; a JSON body's `error` is the reason. */
export function refusalWords(err: unknown): string {
  const raw = String(err instanceof Error ? err.message : err ?? "").trim();
  const m = /^(\d{3}):\s*([\s\S]*)$/.exec(raw);
  let reason = m ? m[2].trim() : raw;
  try { const j = JSON.parse(reason) as { error?: unknown; message?: unknown }; if (typeof j?.error === "string") reason = j.error; else if (typeof j?.message === "string") reason = j.message; } catch { /* not JSON: the text is the reason */ }
  reason = reason.replace(/\s+/g, " ").slice(0, 220) || "no reason given";
  if (/^timeout:/i.test(raw)) return `No answer from the server in time. Check the timeline before trying again: the action may have gone through.`;
  return m && m[1].startsWith("4") ? `Not done. The server refused: ${reason}` : `Not done. ${m ? `Server error ${m[1]}: ` : ""}${reason}`;
}
