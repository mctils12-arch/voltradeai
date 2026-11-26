// Bot page — the owner's actions block under the Issues panel ([PRODUCT]
// 2026-11-25; owner: "i like the owner buttons"). Four actions that had an
// owner-only route and no control: run a scan now, reset the circuit breaker,
// close a position, place a manual order. It sits with the page's other
// actions (the Issues panel already carries resume / stop / kill) and uses
// the same two-step pattern: a button opens a confirm box that states the
// consequence; only the confirm button sends.
//
// Every sentence is built in ownerActionsView.ts (pure, tested): what the
// action will do, and each reason the server would refuse, BEFORE the click.
// The server's answer is shown in its own words. After any action the page's
// status, timeline, positions and working orders are re-read. Nothing is sent
// on open, on typing, or on review; a manual order is a market order for
// whole shares of a stock or ETF, which is all the route can send.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { apiRequest, queryClient } from "@/lib/queryClient";
import type { BotAccount, BotPosition, BotStatus, MarketStatus } from "./useBotData";
import { Fold } from "./selfCheck";
import {
  breakerView, BREAKER_RESET_CONFIRM, scanConfirm, scanResult, closeChoices, closeConfirm, reviewOrder, orderResult, lastPriceFromBars,
  ownerActionsSentence, refusalWords, STOCK_SYMBOL_RE, type OrderDraft,
} from "./ownerActionsView";

type Kind = "scan" | "breaker" | "close" | "order";
type Outcome = { ok: boolean; text: string };
const REFRESH = ["/api/bot/status", "/api/bot/audit", "/api/bot/positions", "/api/orders/open", "/api/bot/account"] as const;
/** a mutation is never cut off mid-flight by the client (a retried POST could act twice): no timeout, no retry */
async function send(method: string, url: string, body?: unknown): Promise<unknown> {
  const r = await apiRequest(method, url, body);
  return r.json().catch(() => ({}));
}

export function OwnerActions({ status, positions, account, market, baseSymbols }: {
  status: BotStatus | undefined; positions: BotPosition[] | undefined; account: BotAccount | undefined; market: MarketStatus | undefined;
  baseSymbols: ReadonlyArray<string | null | undefined>;
}) {
  const breaker = useMemo(() => breakerView(status), [status]);
  const choices = useMemo(() => closeChoices(positions, baseSymbols), [positions, baseSymbols]);
  const marketOpen = market ? market.isOpen : null;
  const [confirm, setConfirm] = useState<Kind | null>(null);
  const [busy, setBusy] = useState<Kind | null>(null);
  const [out, setOut] = useState<Partial<Record<Kind, Outcome>>>({});
  const [closeSym, setCloseSym] = useState("");
  const [draft, setDraft] = useState<OrderDraft>({ ticker: "", side: "buy", qty: "" });
  const [quote, setQuote] = useState<{ symbol: string; price: number | null; at: string | null } | null>(null);
  const [quoting, setQuoting] = useState(false);
  const quoteAc = useRef<AbortController | null>(null);
  useEffect(() => () => quoteAc.current?.abort(), []);

  const run = useCallback(async (kind: Kind, fn: () => Promise<Outcome>) => {
    setBusy(kind); setConfirm(null);
    let o: Outcome;
    try { o = await fn(); } catch (e: unknown) { o = { ok: false, text: refusalWords(e) }; }
    setOut((p) => ({ ...p, [kind]: o }));
    setBusy(null);
    for (const k of REFRESH) void queryClient.invalidateQueries({ queryKey: [k] });
  }, []);

  const chosen = choices.find((c) => c.symbol === closeSym) ?? null;
  const symbol = draft.ticker.trim().toUpperCase();
  const held = positions ? (positions.find((p) => p.ticker?.toUpperCase() === symbol && !p.isOption)?.qty ?? 0) : null;
  const review = reviewOrder(draft, {
    price: quote && quote.symbol === symbol ? quote.price : null, priceAt: quote && quote.symbol === symbol ? quote.at : null,
    heldQty: held, equity: typeof account?.equity === "number" ? account.equity : null, killSwitch: !!status?.killSwitch, marketOpen,
  });
  // REVIEW reads the last price (one abortable GET) and opens the confirm box; it sends no order
  const reviewNow = useCallback(async () => {
    setOut((p) => ({ ...p, order: undefined }));
    if (!STOCK_SYMBOL_RE.test(symbol)) { setQuote({ symbol, price: null, at: null }); setConfirm("order"); return; }
    quoteAc.current?.abort();
    const ac = new AbortController(); quoteAc.current = ac;
    setQuoting(true);
    try {
      const r = await fetch(`/api/bot/bars/${encodeURIComponent(symbol)}?timeframe=1Min&limit=1`, { credentials: "include", signal: ac.signal });
      const last = r.ok ? lastPriceFromBars(await r.json()) : null;
      if (!ac.signal.aborted) setQuote({ symbol, price: last?.price ?? null, at: last?.at ?? null });
    } catch { if (!ac.signal.aborted) setQuote({ symbol, price: null, at: null }); }
    if (!ac.signal.aborted) { setQuoting(false); setConfirm("order"); }
  }, [symbol]);

  const result = (k: Kind) => (out[k] ? <div className={`vt-oa-out ${out[k]!.ok ? "vt-bot-g" : "vt-bot-o"}`} role="status" data-oa-out={k} data-ok={out[k]!.ok ? "true" : "false"}>{out[k]!.text}</div> : null);
  const box = (k: Kind, words: string, label: string, onYes: () => void, extra?: JSX.Element | null, can = true) => (
    <div className="vt-bot-confirm" data-oa-confirm={k}>
      <div className="vt-bot-t2">{words}</div>
      {extra}
      <div className="vt-bot-row">
        <button type="button" data-primary="true" disabled={busy !== null || !can} onClick={onYes}>{label}</button>
        <button type="button" onClick={() => setConfirm(null)}>cancel</button>
      </div>
    </div>
  );

  return (
    <Fold id="actions" title="Owner actions" help="ownerActions" cls={breaker.active ? "vt-bot-o" : "vt-bot-t2"} summary={ownerActionsSentence(breaker, positions ? choices.length : null)}>
      <div className="vt-oa" data-oa="scan">
        <div className="vt-oa-h"><b>Scan now</b><span className="vt-bot-t3">an unscheduled full scan</span></div>
        {confirm === "scan"
          ? box("scan", scanConfirm(marketOpen), "start the scan", () => void run("scan", async () => ({ ok: true, text: scanResult(await send("POST", "/api/bot/run-now")) })))
          : <button type="button" className="vt-oa-btn" data-oa-open="scan" disabled={busy !== null} onClick={() => setConfirm("scan")}>{busy === "scan" ? "starting…" : "scan now…"}</button>}
        {result("scan")}
      </div>

      <div className="vt-oa" data-oa="breaker">
        <div className="vt-oa-h"><b>Circuit breaker</b><span className={breaker.active ? "vt-bot-o" : "vt-bot-t3"}>{breaker.sentence.replace(/^Circuit breaker:? ?/, "")}</span></div>
        {confirm === "breaker"
          ? box("breaker", BREAKER_RESET_CONFIRM, "reset the breaker", () => void run("breaker", async () => { await send("POST", "/api/bot/circuit-breaker/reset"); return { ok: true, text: "Breaker reset: the pause is over and the streak is 0." }; }))
          : <button type="button" className="vt-oa-btn" data-oa-open="breaker" disabled={busy !== null || !breaker.canReset} title={breaker.canReset ? undefined : "nothing to reset: the breaker is off and the streak is 0"} onClick={() => setConfirm("breaker")}>{busy === "breaker" ? "resetting…" : "reset…"}</button>}
        {!breaker.canReset && breaker.known ? <div className="vt-bot-t3 vt-oa-note">Nothing to reset.</div> : null}
        {result("breaker")}
      </div>

      <div className="vt-oa" data-oa="close">
        <div className="vt-oa-h"><b>Close a position</b><span className="vt-bot-t3">{positions ? (choices.length ? `${choices.length} held` : "none held") : "positions not loaded"}</span></div>
        <div className="vt-oa-row">
          <select aria-label="Position to close" value={closeSym} disabled={busy !== null || !choices.length} onChange={(e) => { setCloseSym(e.target.value); setConfirm(null); setOut((p) => ({ ...p, close: undefined })); }}>
            <option value="">choose a position</option>
            {choices.map((c) => <option key={c.symbol} value={c.symbol}>{c.label}</option>)}
          </select>
          {confirm !== "close" ? <button type="button" className="vt-oa-btn" data-oa-open="close" disabled={busy !== null || !chosen} onClick={() => setConfirm("close")}>{busy === "close" ? "closing…" : "close…"}</button> : null}
        </div>
        {confirm === "close" && chosen
          ? box("close", closeConfirm(chosen, marketOpen), `close ${chosen.isOption ? "the contract" : chosen.symbol}`,
              () => void run("close", async () => { await send("POST", "/api/bot/close", { ticker: chosen.symbol }); setCloseSym(""); return { ok: true, text: `Close order sent for ${chosen.label.split(" · ")[0]}. The Book updates when it fills.` }; }),
              chosen.warnings.length ? <ul className="vt-oa-warn" data-oa-warnings>{chosen.warnings.map((w, i) => <li key={i}>{w}</li>)}</ul> : null)
          : null}
        {result("close")}
      </div>

      <div className="vt-oa" data-oa="order">
        <div className="vt-oa-h"><b>Place an order</b><span className="vt-bot-t3">stocks and ETFs · market order · whole shares</span></div>
        <div className="vt-oa-row vt-oa-form">
          <select aria-label="Side" value={draft.side} disabled={busy !== null} onChange={(e) => { setDraft((d) => ({ ...d, side: e.target.value === "sell" ? "sell" : "buy" })); setConfirm(null); }}>
            <option value="buy">buy</option><option value="sell">sell (held shares)</option>
          </select>
          <input aria-label="Quantity" inputMode="numeric" placeholder="qty" value={draft.qty} disabled={busy !== null} onChange={(e) => { setDraft((d) => ({ ...d, qty: e.target.value })); setConfirm(null); }} />
          <input aria-label="Ticker" placeholder="ticker" autoCapitalize="characters" autoComplete="off" spellCheck={false} value={draft.ticker} disabled={busy !== null} onChange={(e) => { setDraft((d) => ({ ...d, ticker: e.target.value.toUpperCase() })); setConfirm(null); }} />
          {confirm !== "order" ? <button type="button" className="vt-oa-btn" data-oa-open="order" disabled={busy !== null || quoting || (!draft.ticker.trim() && !draft.qty.trim())} onClick={() => void reviewNow()}>{quoting ? "reading the price…" : busy === "order" ? "sending…" : "review…"}</button> : null}
        </div>
        {confirm === "order"
          ? box("order", review.sentence, review.ok ? `${review.side} ${review.qty} ${review.symbol}` : "cannot send",
              () => void run("order", async () => {
                const body = await send("POST", "/api/bot/trade", { ticker: review.symbol, side: review.side, qty: review.qty, type: "market" });
                setDraft({ ticker: "", side: "buy", qty: "" }); setQuote(null);
                return { ok: true, text: orderResult(body) };
              }),
              <>
                {review.problems.length ? <ul className="vt-oa-warn" data-oa-problems>{review.problems.map((w, i) => <li key={i}>{w}</li>)}</ul> : null}
                {review.ok ? <ul className="vt-oa-notes" data-oa-notes>{review.notes.map((w, i) => <li key={i}>{w}</li>)}</ul> : null}
              </>, review.ok)
          : null}
        {result("order")}
      </div>
      <div className="vt-sc-src">each action goes through the bot's existing owner-only route and the server's own checks · nothing is sent until its confirm button is pressed · every action is written to the timeline</div>
    </Fold>
  );
}
