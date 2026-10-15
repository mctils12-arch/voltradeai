# F — EXECUTION PATH (signal → order → broker → fill) & ALL PERSISTED STATE

Auditor F. Repo `/home/user/voltradeai` at HEAD `8998328` (2026-10-15). Read-only audit; no repo file edited.
Every claim is `file:line` against that HEAD. "VERIFY-LIVE" marks a code-reading conclusion that should be
confirmed against production (`/api/diag/*`) before it is shown to users as fact.

---------------------------------------------------------------------------------------------------

## 0. TL;DR — the twelve facts a cockpit designer must know

1. **No join key exists between a decision and a broker order.** No order anywhere sets `client_order_id`
   (repo-wide grep: zero writers; `server/diag.ts:336` deliberately drops it). Alpaca `order.id` is kept only in
   the in-memory `openOrders[]` (`server/bot.ts:3754`) and a few audit-line strings. Any decision↔order↔fill
   link today is a fuzzy `ticker + time` join.
2. **`trade_feedback.fill_price` is NOT the broker fill.** For regular-hours stock entries it is
   `alpaca filled_avg_price × (1 ± random synthetic haircut)` (`server/bot.ts:5659-5668`); for morning-queue
   entries it is the *queued signal price* × haircut, never the broker fill (`server/bot.ts:4130-4138`); for
   ETF entries it is the *underlying stock's* scan price (`server/bot.ts:5409-5413`). The real Alpaca fill
   (`alpaca_fill_price`) and the drawn haircut (`slippage_applied_pct`) are sent to Python but **dropped** by
   `track_fill` (`ml_model_v2.py:2502-2527` fixed key set).
3. **The "volume-tiered" haircut is not volume-tiered in production.** Trade candidates never carry a
   `volume` key (`bot_engine.py:3554-3600`), so `pending.trade.volume || 1000000` always yields 1,000,000,
   which fails the `> 1000000` test → every regular entry draws from the illiquid tier **0.12–0.25 %**
   (`server/bot.ts:5660-5665`); morning-queue always draws **0.15–0.25 %** (`server/bot.ts:4131-4135`).
   Every live feedback record's `volume` is therefore `1000000` (VERIFY-LIVE).
4. **Unfilled entries become phantom "filled" entry records.** Fill confirmation is ONE poll 2 s after the
   batch (`server/bot.ts:5642-5650`); an unfilled limit returns `filled_qty "0"` / `filled_avg_price null`, and
   the payload falls back to `qty_filled: filledQty || requestedQty` and `trade.price` (`:5649`, `:5674`). The
   record is never corrected if the order later fills at another price, partially fills, or is swept/cancelled.
   (KNOWN BROKEN #35 fixed this for **exits** and `register_options_entry` fixed it for **options** entries;
   the stock-entry side is not fixed and not documented in research/ — NEW FINDING.)
5. **Decision-time quote/spread is fetched and thrown away.** `bot_engine.py:3416-3425` reads the latest
   bid/ask for the 0.5 % spread filter; bid/ask/mid/spread_pct are not attached to the candidate. The "signal
   price" that IS kept (`trade.price`, persisted as `expected_price`) is the Alpaca snapshot **`dailyBar.c`**
   at scan time (`bot_engine.py:2826, 2847`), not a quote mid, last-trade, or ask.
6. **Exit P&L label is decision-time, not fill-time.** WS exits record `exit_context.pnl_pct` computed from
   the 1-min IEX bar close at submit (`server/bot.ts:6735-6737`, `7132-7135`); `track_fill` lets that override
   the fill-based P&L (`ml_model_v2.py:2447-2452`). `exit_price` IS the confirmed fill (post-#35) — so exit
   slippage is back-derivable only by reversing `pnl_pct`. Options exits record the Alpaca *mark* at
   order **submission**, before any fill (`options_manager.py:1075-1077` etc.).
7. **Fill detection is polling only.** No `trade_updates` websocket anywhere (repo-wide grep). Paths: one
   2 s poll (stock entries), `sweepStaleOrders()` every 45 s **only during market hours, only while
   active/not-killed/not-circuit-broken** (`server/bot.ts:7602-7616`), and options entries resolved when the
   OCC symbol appears in `/v2/positions` during `manage_options_positions()` (`options_manager.py:1031-1054`).
8. **Several order sources record nothing at all**: Tier-dispatcher SELL_CSP / BUY_PUT (the main CSP path —
   no `register_options_entry`, no feedback), Tier-3 BUY, TIME-EXIT, CC-UNWIND, LIQUIDATE-ALL, manual
   `/api/bot/trade` and `/api/bot/close`, scale-outs, options rolls, and every Python-side order (floor
   rebalances, defensive GLD, third leg, convexity puts, intraday-short covers). Python-side order actions are
   not even audited by Node — they live only in the in-memory `lastScanResult` (overwritten each scan).
9. **The audit log is short-lived.** SQLite `audit_log` keeps the last **2000** rows (`server/bot.ts:1170`);
   at the observed ~50 rows/5 min (`server/bot.ts:2434-2436` comment) that is roughly 3–4 hours of history.
10. **The dashboard's performance numbers are probably the fallback path.** The Python branch of
    `/api/bot/performance` raises `TypeError` on any feedback record with `pnl_pct: None` (every open entry and
    every orphan_exit) at `server/bot.ts:1691`; it falls back to "today's Alpaca closed orders" with
    `avgSlippagePct: 0` (`:1758-1898`). `/api/bot/export-trades` raises on the same records
    (`round(None, 2)`, `:1945`) → HTTP 500. Reproduced in isolation (§5). VERIFY-LIVE.
11. **Mode is a literal.** `/api/bot/account` and `/api/bot/status` return `mode: "paper"` as a hardcoded
    string (`server/bot.ts:1263`, `:2317`). Node trades through a hardcoded paper URL (`:319`); Python trades
    through `ALPACA_BASE_URL` env (default paper). A live env value would split-brain the system while the UI
    still says "paper".
12. **Every deploy cancels every open order.** `gracefulShutdown` on SIGTERM → `DELETE /v2/orders`
    (`server/bot.ts:821-840`); Railway sends SIGTERM on each merge. In-memory `openOrders`, `morningQueue`,
    `monitoredPositions`, notifications, circuit-breaker state, and `lastScanResult` also die with the process.

---------------------------------------------------------------------------------------------------

## 1. EXECUTION PATH

### 1.0 Every order-transmission site (FROZEN per CLAUDE.md — listed, not to be modified)

Node — all go through `alpaca()` (`server/bot.ts:324-346`: `fetch(${ALPACA_BASE}${path})`, 15 s abort,
throws on non-2xx, **calls `r.json()` on every 2xx — so a 204 No Content (single-order DELETE) throws**).

| # | Site | file:line | Method / body |
|---|------|-----------|---------------|
| N1 | Tier-2 stock entry | bot.ts:5590-5595 | POST /v2/orders `{symbol, qty, side, ...getOrderParams(trade.price)}` |
| N2 | Tier-2 ETF entry | bot.ts:5375-5381 | POST /v2/orders `{symbol: etfTicker, qty, side:"buy", ...getOrderParams(trade.price)}` |
| N3 | Morning queue | bot.ts:4098-4106 | POST /v2/orders `{..., ...getOrderParams(trade.price)}` |
| N4 | Tier-3 BUY | bot.ts:4882-4890 | POST /v2/orders limit via getOrderParams(currentPrice) |
| N5 | LIQUIDATE-ALL (kill ≤ -25 % + env) | bot.ts:3962-3968 | POST market DAY |
| N6 | TIME-EXIT | bot.ts:6397-6403 | POST market DAY |
| N7 | POS-KILL | bot.ts:6489-6495 | POST `getOrderParams(current,'stop_loss')` |
| N8 | CC-UNWIND before scale-out | bot.ts:6835-6844 | POST market DAY (buy back short calls) |
| N9 | WS scale-out | bot.ts:6862-6870 | POST `getOrderParams(currentPrice,'take_profit')` |
| N10 | CC-UNWIND before exit | bot.ts:7049-7058 | POST market DAY |
| N11 | WS full exit | bot.ts:7096-7104 | POST `getOrderParams(currentPrice, stop_loss/trailing_stop/take_profit)` |
| N12 | Manual `/api/bot/trade` | bot.ts:3224-3233 | POST `{type (default "market"), time_in_force:"day"}` |
| N13 | Manual `/api/bot/close` | bot.ts:3266 | DELETE /v2/positions/{sym} (Alpaca close-position) |
| C1 | Cancel-all | bot.ts:830 (shutdown), 1248 (account-route DD kill), 2380 (owner kill), 3943 (tier-1 DD kill) | DELETE /v2/orders |
| C2 | Cancel one | bot.ts:3824 (sweep), 3894 (replaceIfBetter) | DELETE /v2/orders/{id} |

Python (each `requests.post(f"{ALPACA_BASE_URL}/v2/orders")`):

| # | Site | file:line | Order |
|---|------|-----------|-------|
| P1 | `submit_options_order` single-leg | options_execution.py:2286-2298 | limit, DAY, `limit_price=contract.limit_price` |
| P2 | `_submit_mleg_order` | options_execution.py:2369-2374 | `order_class:"mleg"`, limit = abs(net), DAY |
| P3 | `_submit_close_order` (all options exits + both roll legs) | options_manager.py:313-325 | limit, DAY |
| P4 | `_close_strategy_mleg` | options_manager.py:538-544 | **market** mleg |
| P5 | third leg SVXY buy / exits | bot_engine.py:4309, 4340 | limit ×1.001 / market |
| P6 | SQQQ cleanup | bot_engine.py:4398 | market |
| P7 | convexity overlay roll-close / open | bot_engine.py:4536, 4663 | market `sell_to_close` / limit at mid `buy_to_open` |
| P8 | `_unwind_covered_calls` | bot_engine.py:4736 | market |
| P9 | SPY/QQQ floor basket + floor ticker | bot_engine.py:5111, 5147, 5205, 5315, 5342 | market or `_order_type`/`_limit_price` |
| P10 | defensive floor (GLD) | bot_engine.py:5493, 5563, 5586 | market / limit at last_px |
| P11 | intraday shorts open / EOD cover | intraday_shorts.py:415, 514 | market |

`placeOptionsOrder()` (`server/bot.ts:1065-1132`) has **no callers** (grep); the comment at `bot.ts:4778`
claiming SELL_CSP goes "via placeOptionsOrder" is wrong — SELL_CSP runs inline Python (§1.3d).

### 1.1 Order-parameter construction — `getOrderParams()` (`server/orderParams.ts:19-76`)

Regular hours = `getETHour()` in [9.5, 16.0) (clock-based, holiday-blind; `orderParams.ts:2-7, 24-25`).

| context | regular hours | extended hours |
|---|---|---|
| `new_entry` (default) | **limit** @ round2(price × **1.001**), DAY | limit @ round2(price × **1.005**), DAY, `extended_hours:true` |
| `stop_loss` / `trailing_stop` | **market**, DAY | limit @ round2(price × **0.995**), DAY, `extended_hours:true` |
| `take_profit` | limit @ round2(price), DAY (no buffer) | limit @ round2(price), DAY, `extended_hours:true` |
| `options_entry/exit` (unused by callers) | limit @ round2(price), DAY | same |

Doc disagreements: comment says "Limit at **ask** + 0.1 %" (`orderParams.ts:47`) — callers pass `trade.price`
(snapshot `dailyBar.c`), never an ask. Morning-queue comment says "Market during regular hours"
(`bot.ts:4104`) and its audit line prints `MARKET … @ market` (`bot.ts:4108`) — the order is actually a limit.

### 1.2 Stock entry — Tier-2 `executeTrades(trades, equity)` (`server/bot.ts:5220-5705`)

Caller: `tier2Intelligence` → `executeTrades(result.new_trades, equity)` only when `isMarketOpen`
(`bot.ts:4661-4662`); otherwise candidates are pushed onto the in-memory `morningQueue` (`:4663-4671`).

**Candidate shape** (built at `bot_engine.py:3554-3600`): `action, side, trade_type, ticker, shares, price,
score, reasons, recommendation, rec_reasoning, stop_loss, take_profit, position_value, sizing_reasoning,
sizing_scalars, momentum_score, mean_reversion_score, vrp_score, squeeze_score, volume_score, ewma_rv,
garch_rv, rsi, vrp, instrument, instrument_strategy, instrument_reasoning, instrument_ticker,
instrument_scores, use_options, options_strategy, options_reasoning, options_edge_pct, rules_score,
ml_score_raw, entry_features, entry_date, regime_at_entry`. **Absent**: `volume`, `dollar_volume`, quote
bid/ask, spread, price timestamp, sizing `costs` (computed at `position_sizing.py:1123`, dropped).
`price` = quick-scan `round(dailyBar.c, 2)` from `/v2/stocks/snapshots?feed=data_feed()`
(`bot_engine.py:2784, 2826, 2847`).

Step-by-step:

| Step | file:line | What happens | Recorded where | Lost |
|---|---|---|---|---|
| Snapshot | 5221-5254 | GET /v2/positions; count stock slots, options slots (+ open sell-to-open option orders, `:5243-5248`); `totalDeployed` excl. `FLOOR_AND_LEG_TICKERS` (`:393`) | memory | — |
| Gates (in order) | 5257-5356 | killSwitch break; options slot cap 6 (`OPTIONS-SLOT-FULL`); stock `MAX_POSITIONS`=8 break; `score < state.minScoreThreshold` (`SKIP`); already held; `_consumed`; `dailyBlockedTickers` (`SKIP`); `position_value > equity×0.20×positionSizeMultiplier` (`SIZE-CAP`); `totalDeployed + position_value > equity×0.30` (`HEAT-CAP`, **break**); sector corr via Python `check_correlation_pre_trade(max 0.40)` (`CORR-CAP`, fail-open); halt check (`HALT-SKIP`, fail-open); `orderAttemptCounts ≥ 10` (`MAX-RETRY`); any open order on ticker (`DUP-SKIP`); `position_value > buying_power` → `replaceIfBetter()` | audit_log row per rejection (text only) | Node-side rejections are **not** written to the shadow/counterfactual log (only Python-side rejections are, `shadow_portfolio.update_last_decision`) |
| Sizing | — | No sizing call in Node: trusts `trade.shares` / `trade.position_value` from `position_sizing.calculate_position` (Python, `bot_engine.py:3484-3500`) | `sizing_reasoning` text only in audit `TRADE` line (150 chars) | `sizing.costs`, `stop_loss`, `take_profit` (the "ideal" stop/target) are never enforced or persisted |
| Attempt count | 5359 | `orderAttemptCounts[ticker]++` | memory (reset 4 am ET, `:6207`) | — |
| Instrument audit | 5363-5367 | `INSTRUMENT` audit with stock/etf/options scores | audit_log | — |
| Short block | 5582-5586 | side sell/short → forced "buy" (`SHORT-BLOCKED`) | audit_log | — |
| Order build | 5589 | `getOrderParams(trade.price)` → regular-hours **limit @ price×1.001, DAY** | — | decision price observation time |
| **Raw POST (frozen)** | 5590-5595 | `alpaca("/v2/orders", POST)` | Alpaca | — |
| Audit | 5599-5602 | `TRADE`: `LIMIT BUY {qty} {ticker} @ ~${trade.price} | Score … | Rules/ML/Blend | $value (pct%) | Sizing: …` + notification | audit_log (500-char cap) + in-memory notifications | order id not in the audit text |
| Monitor | 5605 | `addPositionToMonitor(ticker,"long",trade.price,qty)` **at submit, before fill** | memory | — |
| Track order | 5608-5631 | push `{orderId, trade, side, qty}` to `pendingOrderIds`; push `TrackedOrder{orderId,ticker,score,placedAt,side,qty,limitPrice}` to `openOrders` | memory only | lost on restart |
| Pacing | 5637 | 500 ms sleep per trade | — | — |
| Fill check | 5641-5650 | single `sleep(2000)` then GET /v2/orders/{id} per order | — | any fill after 2 s |
| Partial/reject | 5653-5657 | `PARTIAL-FILL` / `ORDER-REJECTED` audit | audit_log | — |
| Synthetic haircut | 5659-5668 | see §2.1 | — | drawn value |
| Fill payload | 5670-5685 | `{ticker, order_type:"market"(**wrong — order is limit**), side, qty_requested, qty_filled: filledQty‖requestedQty, partial_fill, expected_price: trade.price, alpaca_fill_price, fill_price: realisticFillPrice, slippage_applied_pct, order_status, time_placed: now (≈ submit+2 s), session:"regular", volume: 1000000, score, instrument, entry_features, code_version}` | tmp JSON `/tmp/fill_r_{ticker}_{ms}.json` → `track_fill` (daemon first, subprocess fallback, 5 s) | `order_type, qty_requested, partial_fill, alpaca_fill_price, slippage_applied_pct, order_status, instrument` all **discarded** by `track_fill` (§1.9) |
| Failure | 5699-5702 | GET failure → `FILL-CHECK-WARN` and **no record at all** | audit_log | the fill |

### 1.3 Other entry paths

**(a) ETF entry** (`bot.ts:5370-5433`): `etfShares = max(1, floor(trade.shares/2))`; order priced with
`getOrderParams(trade.price)` where `trade.price` is the **underlying** stock price, not the ETF's
(`instrument_selector.py:1523` returns the ETF ticker but no ETF price). Entry record via
`buildEntryFillPayload` (`server/entryFill.ts:52-67`) at **submit time**: `expected_price = fill_price =
trade.price` (underlying), `volume: trade.volume ?? 1000000`, `instrument:"etf"`, no fill check at all.
Monitor registered with the underlying's price as the ETF's entry (`bot.ts:5427`) until the next sync (≤60 s)
replaces it with Alpaca `avg_entry_price` (`bot.ts:6372`). NEW FINDING (VERIFY-LIVE): if ETF price ≪
underlying, the buy limit is far above market (fills, but the feedback entry price is wrong by the ratio);
if ETF price > underlying the limit never fills (swept at 12 min) but the phantom entry record remains; a
1-min bar arriving before the sync computes P&L vs the underlying price.

**(b) Options — scanner path** (`trade_type==="options" && regime_at_entry==="OPTIONS_SCANNER"`,
`bot.ts:5440-5526`): inline Python `select_contract()` → `submit_options_order()` (P1/P2) →
`register_options_entry()` per leg with `entry_price = leg_price or contract.limit_price` → writes
`voltrade_options_state.json` (`bot.ts:5500-5519`). Node keeps only `optExec.order.status`; **not** pushed
to `openOrders` (never swept), monitored via the **underlying** ticker with `trade.price`/`trade.shares`
(`bot.ts:5558`) — but `checkPositionOnTick` returns early only for `positionType !== 'stock'`, and
`addPositionToMonitor` never sets `positionType`, so the stock exit logic *can* evaluate that underlying
ticker (VERIFY-LIVE; the 60 s sync removes it if the underlying is not actually held as stock).

**(c) Options — stock→options path** (`bot.ts:5528-5550`): `pythonCall("evaluate_and_execute")`
(`options_execution.py:2506-2626`) → same submit + `register_options_entry`.

**(d) Tier dispatcher SELL_CSP** (`bot.ts:4780-4865`) — the main CSP path: gates = market open
(`tierDispatchGate.ts:40-48`), option slot cap incl. resting sell-to-open orders (`:4695-4707`), swept-CSP
60-min cooldown / existing resting order (`:4763-4774`, `cspSweepCooldown.ts`). Inline Python
`select_contract(ticker,"sell_cash_secured_put",0,equity,cash_available=options_buying_power)` →
`submit_options_order`. Contract: 30–50 DTE (`CSP_ENTRY_MIN_DTE`, `options_execution.py:107`), delta
≈ -0.30, **limit = mid − 0.30×(mid−bid)** (`_optimized_limit_price`, `options_execution.py:724-761`),
qty ≤ 2. Node keeps `{order_id, side, qty, limit_price}` in `openOrders` with `cspUnderlying`
(`bot.ts:4847-4858`); audit `T{tier}`: `SELL_CSP {ticker} | {reason}` — **no strike, premium, order id,
bid/ask**. **No `register_options_entry` → no feedback entry, ever** (manage_options_positions later
initializes state on first sight with `strategy:"unknown"`, `entry_timestamp: now`,
`options_manager.py:1011-1023`). Capital-starved rejects are logged to shadow as `rejected_capital`
(`bot.ts:4811-4823`).

**(e) Tier-4 BUY_PUT** (`bot.ts:4919-4970`): same pattern, `buy_put`, limit = mid + 0.30×(ask−mid); no
feedback.

**(f) Tier-3 BUY** (`bot.ts:4869-4916`): price from `alpaca('/v2/stocks/{t}/snapshot')` — `alpaca()` targets
`paper-api.alpaca.markets`, which does not serve market-data snapshots (those live on
`data.alpaca.markets`), so `.catch(()=>null)` → price 0 → `T3-SKIP` every time. NEW FINDING, VERIFY-LIVE
(count `T3-SKIP` vs `T3` in `/api/diag/audit?type=`).

**(g) Morning queue** (`executeMorningQueue`, `bot.ts:3990-4177`): runs on the first market-open tier-1
cycle when `morningQueue.length>0` (`:4237-4241`). Drops items queued > 16 h (`MORNING-STALE`); gap check
vs delayed-SIP `dailyBar.c || latestTrade.p` (`:4055`): > 5 % skip (`MORNING-GAP`), > 2 % recompute
shares only (`MORNING-ADJUST`) — **limit price still derives from the stale queued `trade.price`**
(`:4104`). POST (N3), push to `openOrders` with Alpaca's returned `limit_price` (`:4117-4127`). Entry
record **at submit, never checks the fill**: `fill_price = trade.price × (1 + mSlip/100)`,
`time_placed = queuedAt` (the evening queue time), `session:"morning_queue"`, no `entry_features`
(`:4130-4161`). Monitor registered at submit (`:4166`). The queue is in-memory: a deploy overnight empties it.

**(h) Manual** `/api/bot/trade` (`bot.ts:3189-3259`): `checkTradeAllowed()` uses
`currentPrice = last_equity/100` "rough estimate" (`:3198`) — the 20 % size check is meaningless. Tracked in
`openOrders` only if `type !== "market"`. No feedback, no monitor registration (the 60 s sync picks it up).
`/api/bot/close` (`:3262-3274`): Alpaca close-position DELETE, `removePositionFromMonitor`; no exit record
(the matching feedback entry, if any, stays `outcome: null` forever).

### 1.4 Fill detection (what exists)

| Mechanism | Covers | Cadence / gating | file:line |
|---|---|---|---|
| One poll after 2 s | Tier-2 stock entries | once per executeTrades batch | bot.ts:5641-5650 |
| `sweepStaleOrders()` reconciliation | every `openOrders` entry (entries & exits) | tier-1, every 45 s, **RTH only**, skipped while killed / inactive / circuit-breaker | bot.ts:3809-3875, 7602-7616 |
| `resolvePendingExitFill()` | exits carrying `pendingExitFill` (WS full exit, POS-KILL) | inside the sweep | bot.ts:3790-3807 |
| Position appearance | options entries registered via `register_options_entry` (non-multi-leg) | each scan's `manage_options_positions` | options_manager.py:1031-1054 |
| None | morning queue, ETF, tier dispatcher options, Tier-3, manual, all Python-side orders, options exits | — | — |

No `trade_updates` / `account_updates` stream exists. The only market-data WS is
`wss://stream.data.alpaca.markets/v2/iex` 1-min **bars** (`bot.ts:7248`), used for exits (§1.6).

### 1.5 Position-monitor registration

`monitoredPositions: Record<string, MonitoredPosition>` (in-memory, `bot.ts:6305-6336`). Fields:
`ticker, side, entryPrice, qty, phase, initialRiskPct, highestPnl, currentStopPct, tpPct, entryDate,
lastCheckedPrice, originalQty, remainingQty, scalesCompleted, breakevenActive, regime, atrPct,
highestPrice, pendingExit?, positionType?, creditReceived?, maxLoss?, legSymbols?, strategyLabel?`.

- `addPositionToMonitor` (`:6658-6688`): at submit; defaults `initialRiskPct=currentStopPct=4.0`,
  `tpPct=6.0`, `atrPct=2.0`, `regime:"NEUTRAL"`, `entryDate: today(UTC)`; subscribes ticker to WS bars.
- `syncMonitoredPositions` (`:6345-6653`): every 60 s while active (`:7227-7230`, **not** RTH-gated) and in
  tier-1; rebuilds every non-floor Alpaca position: `entryPrice = avg_entry_price`, stop/TP from
  `voltrade_stop_state.json` (written by `bot_engine.manage_positions`), `remainingQty` capped at live qty
  (`POS-QTY-CORRECTED`), removes tickers no longer held. Also runs TIME-EXIT (§1.6) and POS-KILL.
- Python `manage_positions()` (`bot_engine.py:1975-2337`) runs every tier-1 cycle (`bot.ts:4212-4231`)
  only to refresh stop_state; its actions are NOT executed (logged as `POS-MONITOR-SYNC`).
  NOTE: Python's rules (60-min min-hold, 4-day short-hold stop suppression, regime stop scalars,
  `bot_engine.py:2104-2110, 2251-2275`) are **not** applied by the Node WS exit engine that actually trades.

### 1.6 Exit paths

| Exit | Trigger (decision price) | Order | Tracked in `openOrders` | Fill recorded |
|---|---|---|---|---|
| WS scale-out 1/3 at 1R, 2/3 at 1.5/2/3R by regime | IEX 1-min bar close `item.c` (`bot.ts:7286-7296`) | `take_profit` limit @ bar close (N9) | yes, `isExit`, no `pendingExitFill` | **never** (by design — `exitFill.ts:14-17`) |
| WS stop / trailing / breakeven / TP / time-stop (7 d & |pnl|<2 %) | bar close vs `pos.entryPrice` | stop → market (RTH); TP → limit @ close; time-stop → market | yes + `pendingExitFill{ticker, exitSide, pnlPct, exitReason, entryDate, codeVersion}` (`:7111-7136`) | on confirmed `filled` in sweep, at `filled_avg_price`/`filled_qty` |
| POS-KILL (stock ≤ -25 %, option short ≤ -150 %, option long ≤ -75 %) | Alpaca `unrealized_plpc` in sync | `stop_loss` params (market RTH / limit ×0.995 ext) | yes + `pendingExitFill{…, exitReason:"position_kill"}` | same as above |
| TIME-EXIT (>14 d, pnl −5..+2 %) | Alpaca position | market | no | no. Reads `pos.created_at‖submitted_at` (`:6390`) — Alpaca Position objects document neither field → **likely never fires** (NEW, VERIFY-LIVE) |
| LIQUIDATE-ALL | DD kill ≤ -25 % and `VOLTRADE_LIQUIDATE_ON_KILL=true` | market | no | no |
| CC-UNWIND | before scale-out/exit | market buy-to-close | no | no |
| options_manager (DTE≤5, assignment, 21-DTE, 50 % profit, 2× credit loss, 50 % bought loss, gamma) | Alpaca position `current_price` + option snapshot | limit at mid / ask / bid (`options_manager.py:1069-1302`) | no (Python) | `_record_options_exit_feedback` **at submit**: `fill_price = expected_price = current_price` (mark), `pnl_pct = unrealized_pl / (avg_entry×qty×100)` (`options_manager.py:106-150`). A close that does not fill is re-recorded on a later cycle → duplicate orphan_exit records |
| options rolls | as above | two `_submit_close_order` calls | no | none |
| multi-leg groups | `_manage_strategy_group` | market mleg | no | none (KNOWN BROKEN #12(c)) |
| Python floor / defensive / third-leg / convexity / shorts | regime & drift | P5–P11 | no | none |

Exit-side frictions to show in a cockpit: after a stop/trailing exit, `voltrade_stop_cooldown.json` is
written and the circuit breaker counters move (`bot.ts:7150-7212`); `pendingExit=true` blocks re-evaluation
until the sweep clears it (`:7217`, `:3831`).

### 1.7 `track_fill` — the only fill/closed-trade recorder (`ml_model_v2.py:2355-2566`)

- Validation: ticker required; `qty = qty ‖ qty_filled ‖ qty_requested`, ≤0 → logged DROP (`:2382-2392`).
- `slippage = |fill_price − expected| / expected × 100` — **absolute, sign lost** (`:2394-2396`).
- **Exit detection** `_is_exit_fill`: `exit_context` or `exit_reason` or `is_close` (`:2313-2330`).
- **Entry → APPEND** record (`:2502-2555`), exact keys:
  `ticker, side, qty, expected_price, fill_price, slippage_pct, volume, session, time_placed,
  time_filled (= Python now unless passed — no caller passes it on entries), score, entry_features,
  outcome: null, pnl_pct: null, code_version, config_fingerprint, config_snapshot`.
- **Exit → mutate matching entry** found by `_find_entry_record` = newest record with same ticker and
  `outcome is None` (`:2333-2352`, ticker-only match). Adds `outcome (win/flat/loss), pnl_pct (bot-reported
  exit_context.pnl_pct wins over price math), exit_price (= exit fill_price), exit_time, exit_reason,
  days_held, exit_context`. Calls `risk_kill_switch.record_trade_outcome(pnl_pct)` (consecutive-loss counter).
- No match → APPEND `{ticker, side, qty, fill_price, time_filled, outcome:"orphan_exit", pnl_pct:null,
  exit_reason, note, code_version}` (`:2472-2499`).
- Rotation: keep last **5000** records (`:2557-2559`). Write: `_atomic_write_feedback` = mkstemp + `os.replace`
  under `flock(FEEDBACK_PATH+".lock")` (`:2275-2310`) — but the **read** happens before the flock, guarded
  only by an in-process `threading.Lock` (`:2398-2406`), so daemon vs subprocess vs boot-cleanup writers can
  lose updates. Return value ignored; all exceptions swallowed at `debug` level (`:2563-2566`).
- Node side: fire-and-forget `pythonCall(...).catch(()=>{})` at every site — **no audit line on failure**.

Consequence: the "closed-trade record" is the entry record mutated in place. Its P&L is the final
tranche's decision-time P&L only (scale-out tranches never recorded), entry price may be synthetic, and
cross-instrument collisions are possible (an options exit on underlying `X` can close a stock entry record
for `X`, since both match on ticker).

### 1.8 Other "closed trade" views (all derived, none durable)

- `tradeResults[]` (in-memory, ≤200, `bot.ts:554`, filled by `trackClosedTrades` `:851-937` from the last 50
  Alpaca closed orders): `entryPrice = exitPrice = filled_avg_price` ("simplified", `:863-864`), then
  `pnlPct` back-filled from **average** buy vs **average** sell per symbol over the last 100 rows
  (`:893-930`); `pnl` is never set (stays 0), `strategy:"auto"`, `instrument:"stock"` for options too.
- `adjustStrategyWeights()` (`:994-1019`): win rate = `t.pnl > 0` → always 0 % → every cycle shifts
  `vrp→0.40, momentum→0.15, volume→0.10` and writes a `LEARN` audit every 30 min. `strategyWeights` (`:843`)
  is **read by nothing** (grep) — the LEARN lines describe a no-op. NEW FINDING.
- `/api/bot/history` (`:1411-1429`): Alpaca `/v2/orders?status=closed&limit=50` (includes canceled) mapped to
  `{ticker, side, qty, price, status, submittedAt, filledAt, type}`.
- `/api/trades/today`, `/api/trades/history`, `/api/orders/open`, `/api/positions` (`server/routes.ts:6845,
  7012, 6947, 6968`): Alpaca passthroughs with FIFO pairing; dollar P&L `(exit−entry)×qty` ignores the
  ×100 option multiplier and treats sell-to-open as a close.

---------------------------------------------------------------------------------------------------

## 2. SLIPPAGE / COST MODEL

### 2.1 The "synthetic volume-tiered slippage haircut" — exact code

Regular-hours entries (`server/bot.ts:5660-5668`):
```
vol = pending.trade.volume || 1000000
slippagePct = 0.03                                  // unreachable default
if (vol > 20_000_000) slippagePct = 0.02 + U[0,1)*0.03   // 0.02–0.05 %
else if (vol > 5_000_000) slippagePct = 0.05 + U*0.05    // 0.05–0.10 %
else if (vol > 1_000_000) slippagePct = 0.08 + U*0.07    // 0.08–0.15 %
else slippagePct = 0.12 + U*0.13                          // 0.12–0.25 %
realisticFillPrice = round2(filledAvgPrice * (1 + dir*slippagePct/100)), dir = +1 buy / -1 sell
```
Morning queue (`server/bot.ts:4131-4138`): `>20M: 0.05–0.10 %`, `>5M: 0.08–0.15 %`, else `0.15–0.25 %`;
applied to **`trade.price`** (queued signal price), not any fill.
Because candidates carry no `volume` (§1.2), the effective live tiers are always the last branch:
**0.12–0.25 % (regular), 0.15–0.25 % (morning)**. Uses `Math.random()` — not reproducible, draw not persisted.

Mirrors: `backtest_v2.py:63-71` claims to mirror these tiers (per-side 0.035/0.075/0.115/0.185 %, plus a
half-tick floor `:88-99`) using real trailing-20-day volume. So backtest charges by real volume while live
charges the illiquid tier → a built-in live-vs-backtest cost divergence for liquid names. Neither tier set is
calibrated against measured fills. `position_sizing._estimate_costs` (`position_sizing.py:817-872`) is a
third, different volume-share model (0.02–0.30 %, + $0.65/contract options fee) used only inside sizing.

### 2.2 Is real slippage measured?

**No — not end-to-end, not persisted as such.**
- Decision reference price stored: `expected_price = trade.price` = snapshot **`dailyBar.c`** at scan time
  (`bot_engine.py:2847`). Not a quote mid, not last trade, not the ask the comments claim. The scan itself can
  take 55–300 s (`bot.ts:4400-4412`) before Node submits; the observation timestamp is not stored.
- Fill price stored: `fill_price` (synthetic-adjusted); the true broker `filled_avg_price` is sent as
  `alpaca_fill_price` and dropped. `slippage_pct = |fill_price − expected| / expected` therefore mixes
  (decision→fill drift) + (random haircut), unsigned.
- Exits: no exit decision price stored (only inside audit text `@ $X` and implicitly in `exit_context.pnl_pct`).
- Options: CSP limit derived from bid/ask/mid inside `_optimized_limit_price`; the contract dict returned
  (`options_execution.py:1074-1092`) carries `limit_price, premium_received, cash_required, delta, gamma,
  theta, iv, days_to_expiry` but **not bid/ask/mid**; Node persists none of it.
- `fills_slippage_stats(feedback)` (`ml_model_v2.py:2627-2650`) = mean of `slippage_pct` over records with
  `expected_price>0`; `total_slippage_cost = Σ expected×qty×slip%`. `/api/bot/performance`
  "realistic P&L" = `total_pnl − avg_slippage_pct × 2 × N` (`bot.ts:1726`).
- The only true broker execution ledger available is Alpaca itself: orders (`filled_avg_price`,
  `filled_qty`, `submitted_at`, `filled_at`, `limit_price`) and `/v2/account/activities/FILL` (used only by
  auto-resume, `bot.ts:7449`).

### 2.3 Spread at decision time

Fetched (`bot_engine.py:3416-3425`, `/v2/stocks/{t}/quotes/latest`), used for the 0.5 % reject rule (rejects
logged to shadow as `rejected_other`), **not persisted** for accepted trades. Options: `_is_liquid` spread
≤ 10 % of mid (`options_execution.py:695-699`), not persisted.

### 2.4 Fees / commissions

None recorded anywhere in execution state. Equity commissions $0 (`position_sizing.py:46`); options
"$0.65/contract fee on live (free on paper)" (`options_execution.py:14`) — modeled only in
`_estimate_costs` for sizing; nothing in trade_feedback, audit, or equity curve. No regulatory (SEC/TAF) fee
model. On paper the realized fee bucket is legitimately ~0, but nothing records that it was checked.

---------------------------------------------------------------------------------------------------

## 3. STATE INVENTORY

### 3.1 DATA_DIR resolution

- Python: `storage_config.py:10-15` → `/data/voltrade` if `os.path.isdir("/data")` else `/tmp`; inline
  fallbacks repeat the same rule (e.g. `risk_kill_switch.py:41`, `options_manager.py:40`). No module checks
  that `/data` is an actual volume mount.
- Node: hardcoded `"/data/voltrade/<file>"` with `"/tmp/<file>"` fallback for each persisted file
  (`bot.ts:561-562, 604-605, 647-648, 729-730, 963-964`); writers `mkdirSync('/data/voltrade',{recursive})`
  then write; fall to `/tmp` only on a write **error** (e.g. disk full — `durability.test.ts` notes a
  "nearly-full /data volume"); readers try `/data` first. A /tmp fallback write is lost on the next deploy and
  the stale `/data` copy wins at boot. Failures go to `console.error` only — never audited, never in
  `/api/health`.
- Ad-hoc checks differ: `fs.existsSync('/data/voltrade')` (`bot.ts:1302,1330`) vs `fs.existsSync('/data')`
  (`:6930`) vs Python `os.path.isdir('/data')`.
- SQLite: `server/auth.ts:12-16` → `process.env.DB_PATH || (exists('/data') ? '/data' : cwd) + '/voltrade.db'`
  — i.e. **`/data/voltrade.db`, outside `/data/voltrade/`**. Billing uses a separate `cwd/voltradeai.db`
  (`billing.ts:34`, ephemeral, wishlisted stray).

### 3.2 Inventory (bot-relevant)

Atomic = tmp+rename. "Ephemeral" = memory or container FS.

| State | Path | Writer | Reader | Shape (example) | Retention | Atomic |
|---|---|---|---|---|---|---|
| Live positions (truth) | Alpaca `/v2/positions` | broker | everything | Alpaca Position | — | — |
| Position monitor mirror | memory `monitoredPositions` | bot.ts:6607, 6660 | WS exits | §1.5 | rebuilt every 60 s | — |
| Evolving stop state | `voltrade_stop_state.json` | bot_engine.py:2183-2200, 2331-2336; bot.ts scale-out persist :6908-6941 (Python + Node fallback) | bot.ts:6353, 1286; bot_engine | `{TICKER:{initial_risk_pct, highest_pnl, highest_price, phase, current_stop_pct(raw ATR%), r_multiple, entry_date, entry_timestamp(first-seen), days_held, original_qty, remaining_qty, scales_completed, breakeven_active, regime, [short_hold_suppression]}}` | non-held tickers pruned each manage_positions | **no** (3 writers) |
| Options position state | `voltrade_options_state.json` | options_manager.py:173-179, 1346-1412; bot.ts:5506-5518 inline | options_manager; `/api/bot/positions` | `{OCC:{entry_price, entry_delta, entry_date, entry_timestamp, initial_credit, max_profit_target, highest_value, strategy, setup, ticker, side, qty, max_loss, [pending_entry_feedback], current_delta/gamma/theta/price, dte, last_checked}, UNDERLYING:{strategy, setup, options_strategy}}` | stale OCCs pruned each cycle | no |
| Trade feedback (entries + closed trades + orphans) | `voltrade_trade_feedback.json` (+ `.lock`) | `track_fill` (ml_model_v2.py:2355); boot cleanup (bot.ts ~7877, `open(w)` no lock); `_clean_poisoned_feedback_on_startup` (ml_model_v2.py:998-1050); seeder (`_seed:true`) | perf/export/tier-pnl/diag ml, ML training | §1.7 | last 5000 | mostly (track_fill only) |
| Fills (legacy) | `voltrade_fills.json` | **none** (dead) | imported by position_sizing.py:29 | — | — | — |
| Equity curve | `voltrade_equity_curve.json` | `saveEquityCurve` bot.ts:583-596 via `recordDailyEquity` :1021-1041 | `/api/bot/performance` (`equityCurve`), `/api/diag/equity_curve` | array newest-first `[{date:"2026-10-15"(UTC), value: portfolio_value, pnl: equity−last_equity}]` | 365 rows (see bug §5) | no (writeFileSync) |
| Equity peak (Node DD kill) | `voltrade_equity_peak.json` | bot.ts:622-633 | :607-620 boot | `{equityPeak, savedAt}` | overwrite | no |
| Node kill-switch latch | `voltrade_kill_switch.json` | bot.ts:675-694 | :658-673 boot | `{killSwitch, reason:"drawdown-kill (tier1)"‖"drawdown-kill (account route)"‖"owner toggle ON/OFF", savedAt, lastAutoResumeAt?}` | overwrite | no |
| Liveness heartbeat | `voltrade_liveness.json` | bot.ts:751-762 (60 s while active) | :732-749, `/api/health` | `{lastActiveAt(ms), lastReminderAt?, savedAt}` | overwrite | no |
| Learning cutover | `voltrade_deploy_timestamp.json` | bot.ts:965-991 (first boot only) | adjustStrategyWeights | `{deployTimestamp}` | permanent | no |
| bot_engine DD halt | `voltrade_portfolio_dd.json` | bot_engine.py:551-561, 695-706; paper_resume_sync | scan_market, paper_resume_sync | `{peak_equity, last_equity, last_updated, halted, halt_reason, halt_started_at, halt_confirmed, halt_trip_ts, halt_trip_equity, last_anomaly_release}` | overwrite | yes |
| risk_kill_switch state | `voltrade_killswitch.json`, `voltrade_kill_history.json`, touch-file `voltrade_MANUAL_KILL` | risk_kill_switch.py:112-124, 131-144 | tier engine, status endpoints | `{killed, kill_reason, killed_at, peak_equity, consecutive_losses, last_loss_ts, daily_loss_date, daily_loss_pct}` | overwrite | state yes / history no |
| Shared tier/risk peak | `voltrade_peak_equity.json` | tiered_strategy.update_peak_equity, risk_kill_switch.set_peak_equity (max-ratchet) | tiered_strategy T2 gate / master kill | `{peak_equity, last_updated}` | ratchet only | yes |
| Floor regime state | `voltrade_floor_state.json`, `voltrade_defensive_floor_state.json` | bot_engine.py:4868, 5450 | floors, `/api/system/snapshot` | `{floor_regime, floor_regime_pending, floor_regime_pending_count, floor_regime_last_obs, …}` | overwrite | — |
| Stop cooldown | `voltrade_stop_cooldown.json` | bot.ts:7152-7162; bot_engine.py:2281 | bot_engine.py:3267 | `{TICKER: epoch_s}` | 24 h | no |
| Shadow / counterfactual | `voltrade_shadow_candidates.json`, `voltrade_shadow_stats.json` | shadow_portfolio.log_candidate / update_last_decision / backfill_outcomes | diag shadow, ML | `{ticker, timestamp, score, decision:"taken"‖"rejected_score"‖"rejected_heat"‖"rejected_masterkill"‖"rejected_capital"‖"rejected_other", decision_reason, entry_price, vxx_ratio, regime_label, side, action_label, trade_type, features{}, outcomes{"+5d","+10d","+20d"}, code_version:"1.0.34-shadow"}` | 20,000 | yes (flock) |
| Intraday shorts log | `voltrade_intraday_shorts.json` | intraday_shorts.py | close_open_shorts | `{trades:[{symbol, shares, entry_price, exit_price, pnl_pct, pnl_dollar, closed_at, status}]}` | — | — |
| Scan timings | `voltrade_scan_timings.json` | bot_engine (progressive) | snapshot, TIER2-ERROR detail | `{status, phases[{phase,duration_sec}], last_phase_completed, last_phase_at, scan_started}` | per scan | — |
| Audit log | SQLite `/data/voltrade.db` table `audit_log(id INTEGER PK AUTOINCREMENT, time TEXT, type TEXT, message TEXT)` (bot.ts:1138-1143); WAL (`:1153-1158`) | `persistAudit` :1160-1180 (message ≤500 chars); routes.ts:906 | `/api/bot/audit` (100), `/api/diag/audit?type=&limit≤500`, SSE `/api/bot/stream` (live, all fields `{time,type,action,detail,message}`) | — | **last 2000 rows**, pruned on every insert | SQLite |
| Other SQLite tables on the same DB (not bot) | `users, sessions, password_resets` (auth.ts:20-51), `watchlists` (routes.ts:607), `api_keys` (apiKeyAccounts.ts:97) | — | — | — | — | — |
| `shared/schema.ts`, `server/storage.ts` | no schema, `NoopStorage`; drizzle only listed in `script/build.ts:12-13` | — | — | — | — | — |
| Last scan cache | memory `lastScanResult` (bot.ts:4182, 4478) | tier2 | `/api/bot/last-scan` | full scan_market return incl. `new_trades, top_10, position_actions, third_leg, convexity_overlay, options_management, spy_floor, defensive_floor, tier_actions, timestamp…` (bot_engine.py:4006-4060) | overwritten each scan; lost on restart | ephemeral |
| Signals | memory `signals` | tier2 | `/api/bot/signals` | `{ticker, action, reason, confidence, timestamp, type, rank?, factors?…}` | 30 rows, 1 h | ephemeral |
| Notifications | memory (bot.ts:527-533) | `notify()` | `/api/bot/notifications` | `{time, type, message, read}` | 100 | ephemeral |
| Open-order tracker | memory `openOrders` (bot.ts:3754) | entry/exit sites | sweep, replace, dup guards | `TrackedOrder{orderId, ticker, score, placedAt, side, qty, limitPrice, isExit?, pendingExitFill?, cspUnderlying?}` | until resolved | ephemeral |
| Morning queue | memory (bot.ts:3716) | tier2 off-hours | executeMorningQueue | candidate + `queuedAt` | cleared at open | ephemeral |
| Strategy weights | memory (bot.ts:843-849) | adjustStrategyWeights | **nobody** | `{momentum, mean_reversion, vrp, squeeze, volume}` | — | ephemeral |
| Bot `state` | memory (bot.ts:485-499) | many | status endpoints | `{active, killSwitch, dailyPnL, positionSizeMultiplier, minScoreThreshold, consecutiveStopLosses, circuitBreakerUntil, equityPeak, maxDrawdownPct:-10, …}` + ad-hoc `dailyBlockedTickers, lastStopTimes, recentStopTickers, streamSignalQueue` | circuit breaker & counters reset on deploy | ephemeral |
| Backups | `${DATA_DIR}/backups/backup_YYYYmmdd_HHMMSS/` (*.json/*.pkl/*.db in DATA_DIR only — **misses `/data/voltrade.db`**) | 4 am ET (bot.ts:6224-6250) and POST `/api/bot/backup` (:1962-1999) | — | copies | last 5 | — |
| Off-site backup | git branch `data-backup` of `mctils12-arch/voltradeai` | backup_to_github.py:40-138 (4 am, needs `GITHUB_BACKUP_TOKEN`) | — | DATA_DIR *.json/*.pkl ≤50 MB + `voltrade.db` from parent dir (:84-94); status → `backup_status.json` | git history | — |

Equity-curve cadence: `recordDailyEquity()` is called only from `tier3Strategic` (`bot.ts:5870`), hourly,
and the Tier-3 interval returns early when `!state.active` (`:7737-7745`) — **no equity rows are recorded
while the kill switch is latched or the bot is stopped**. Rows are keyed by **UTC** date (`:1024`), updated in
place within the UTC day; weekends produce rows too when active.

---------------------------------------------------------------------------------------------------

## 4. PAPER vs LIVE

- Node: `const ALPACA_BASE = "https://paper-api.alpaca.markets"` hardcoded (`server/bot.ts:319`) — every Node
  order, cancel, account and position call is paper regardless of env.
- Python: `ALPACA_BASE_URL` env, default paper — `bot_engine.py:185`, `options_execution.py:61`,
  `options_manager.py:46`, `intraday_shorts.py:52`, `tiered_strategy.py:996`, `position_sizing.py:791`;
  `options_scanner.py:165` hardcoded paper; `csp_universe.py:180` reads a **different** env var
  `ALPACA_TRADING_URL`.
- routes.ts dashboard passthroughs: `ALPACA_BASE_URL` env default paper (`server/routes.ts:600`).
- Same `ALPACA_KEY`/`ALPACA_SECRET` for all; market data `data.alpaca.markets`; WS `stream.data.alpaca.markets/v2/iex`.
- Exposure of mode: `/api/bot/account` → `mode:"paper"` literal (`bot.ts:1263`); `/api/bot/status` →
  `mode:"paper"` literal (`:2317`). The only computed check is `allPaperAlpacaUrls([ALPACA_BASE,
  env||ALPACA_BASE])` (`server/killSwitchAutoResume.ts:100-118`) used for auto-resume eligibility
  (`bot.ts:7498, 7544`; surfaced indirectly as `autoResume.eligible/reason`) and `paper_resume_sync._is_paper`.
- Split-brain risk: a live `ALPACA_BASE_URL` would send Python orders (CSPs, options exits, floors, hedges)
  live while Node stock entries/exits and the kill-switch cancel-all stay on paper, and the UI keeps saying
  "paper". A cockpit badge should be derived (e.g., from `allPaperAlpacaUrls`), not echoed from the literal.

---------------------------------------------------------------------------------------------------

## 5. HEALTHY VALUES / FAILURE SIGNATURES

| Condition | Healthy | Failure signature | Surfaced? |
|---|---|---|---|
| Stale entry/exit limits | swept at `STALE_ORDER_MINUTES=12` (bot.ts:3768) within one 45 s tier-1 tick → ≤ ~12.75 min open | Alpaca open orders older than 13 min during RTH; or any age outside RTH / while killed / circuit-broken / after a restart (tracker empty) | `SWEEP` audit lines (pre-fix: 0 SWEEP vs 51 cancels, bot.ts:3835-3846) |
| Single-order cancel | `alpaca()` throws on 204 (bot.ts:338-342) | sweep handles it (`confirmCancelAfterDeleteError`); **`replaceIfBetter` does not**: cancel succeeds, function returns false (`:3893-3901`) → new trade skipped, weak order already gone (NEW, VERIFY-LIVE: `REPLACE` lines are never followed by a `TRADE` for the new ticker) | no |
| Entry recorded but never filled | none | feedback entry with `outcome:null` and no Alpaca fill near `time_placed` (phantom) | no |
| Fill not matched to entry | orphan_exit ≈ 0 | `outcome:"orphan_exit"`; was 225/270 (83 %) live on 2026-09-02 (`ml_model_v2.py:2415-2418`) | `/api/diag/ml` `live_outcome_breakdown`, `live_options_outcome_breakdown`; Python `ORPHAN EXIT` warning (log only) |
| Feedback write failure | — | `track_fill` swallows everything (`ml_model_v2.py:2563-2566`), Node `.catch(()=>{})`; subprocess fallback dies on "Python mutex timeout" after 5 s when a scan holds the mutex (`bot.ts:240-264`) | **no** — only indirect (`feedback_live_count` stops growing) |
| Synthetic tier collapse | `volume` varies | every live record `volume == 1000000` | no |
| Equity curve gap | one row per UTC day | missing days while killed/stopped; after 365 rows `unshift`+`shift` drops the **newest** row (`bot.ts:1034-1035`) so the curve freezes (diag comment `:2593` says oldest is pruned — disagreement; triggers ~May 2027) | `/api/diag/equity_curve` |
| /tmp fallback | `/data/voltrade` writes succeed | `console.error("[equity-curve]/[equity-peak]/[kill-switch]/[liveness] could not save …")` | **no** (stdout only) |
| Perf endpoint path | Python branch | response has no `seededTrades` key and `avgSlippagePct:0` ⇒ fell back to today-only Alpaca stats (TypeError at bot.ts:1691 on `pnl_pct: None`) | no |
| CSV export | 200 CSV | HTTP 500 `type NoneType doesn't define __round__` (bot.ts:1945); even when it works, live rows have blank `Date`(`timestamp`), `Strategy`, `Holding Days` and `Won=No` (keys never written) | yes (500) |
| Learning loop | — | `LEARN` "Win rate 0%…" every 30 min (always, `pnl` never set) — no effect | misleading audit |
| WS exit feed | `STREAM` "Real-time feed live" | `STREAM-ERROR`, `STREAM-DISCONNECT`; IEX bars only arrive on minutes with IEX prints → illiquid names get no exit checks | audit |
| Daemon | `via:"daemon"` | `TIER2-ERROR … daemon …`, subprocess fallbacks | audit, `/api/daemon/health` |
| Audit DB | INSERT/DELETE fast | `DB-SLOW-WRITE` | audit |
| Equity read | valid | `EQUITY-READ-INVALID` | audit |
| Kill / DD | — | `DRAWDOWN-KILL`, `DD-HALT`, `TIER-KILL`, `KILL-SWITCH`, `/api/health checks.bot.liveness.dark` | health + audit + email |

Reproduction of the two TypeErrors (scratch Python, same logic as bot.ts:1689-1691 and :1945):
`perf: TypeError '>' not supported between instances of 'NoneType' and 'int'`;
`export: TypeError type NoneType doesn't define __round__ method`.

---------------------------------------------------------------------------------------------------

## 6. GAPS FOR A COCKPIT — ideal-vs-real attribution

"Ideal trade" candidates available today: `trade.price` (dailyBar.c), `stop_loss`/`take_profit` from sizing
(never enforced or stored), `score`/`rules_score`/`ml_score_raw`, `position_value`. Persisted durably only
as feedback `expected_price`, `score`, `entry_features`, `qty`; everything else lives in the 2000-row audit
log text or the in-memory last scan.

| Bucket | Computable today? | From | Missing / blocking |
|---|---|---|---|
| Entry slippage (decision → fill), stocks | **Partially** | feedback `expected_price` (dailyBar.c) vs Alpaca order `filled_avg_price`, joined by ticker+time | no order_id/client_order_id on the record; `fill_price` is synthetic; `alpaca_fill_price` dropped; reference is a daily-bar close, not a quote mid; morning/ETF records have no real fill at all |
| Entry slippage, options | **No** (dispatcher CSPs) / partial (scanner path: `entry_price` = limit, then Alpaca avg) | Alpaca orders only | no decision bid/ask/mid; no feedback entry for dispatcher CSPs |
| Spread / half-spread cost at decision | **No** | — | `bot_engine.py:3418-3425` quote discarded; options bid/ask not in contract dict |
| Limit-price buffer (paid vs signal) | Yes (stocks) | `limit = round2(price×1.001)` reconstructable; Alpaca `limit_price` | — |
| Exit slippage (WS/POS-KILL) | **Back-derivable only** | confirmed `exit_price` vs `entryPrice×(1+pnl_pct/100)` — but the entry used there is Alpaca `avg_entry_price`, not the record's `fill_price` | exit decision price & timestamp not stored; scale-out legs not recorded; TIME-EXIT/CC-UNWIND/manual exits not recorded |
| Exit slippage (options) | **No** | — | recorded at submit with the mark; no fill confirmation |
| Fees / commissions | Trivially 0 on paper; **not recorded** | — | no fee field; options $0.65/contract only in sizing model |
| Synthetic haircut (model cost) | **No** | — | `slippage_applied_pct` dropped; random |
| Timing — decision → submit | **No** (durably) | scan `timestamp` (memory) + audit `TRADE` time | price observation time not stored; scan may run 55–300 s |
| Timing — submit → fill | Via Alpaca only | `submitted_at`/`filled_at` | feedback `time_placed` = submit+2 s (or queue time for morning); `time_filled` = Python now() |
| Fill completeness (partial/unfilled) | Via Alpaca only | `filled_qty`, `status` | record stores requested qty when unfilled at 2 s |
| Opportunity cost of rejections | Python-side rejections only | shadow log `decision` + outcomes | Node rejections (SKIP/SIZE-CAP/HEAT-CAP/CORR-CAP/HALT-SKIP/MAX-RETRY/DUP-SKIP/OPTIONS-SLOT-FULL/CSP-SKIP) are audit-text only; shadow `taken` ≠ order placed |
| Orders lost to deploys | Partially | audit `SHUTDOWN` + Alpaca canceled orders at that time | — |
| Python-side orders (floors, hedges, rolls) | Alpaca only | `lastScanResult.{spy_floor,defensive_floor,third_leg,convexity_overlay,options_management}.actions` (memory, latest scan only) | not audited, not persisted |

Concrete fields a cockpit would need (to be added through normal PRs, not by this audit):
decision `{signal_price, signal_price_source, signal_price_ts, quote_bid, quote_ask, quote_ts, spread_pct,
intended_stop, intended_target, scan_ts}`; submit `{client_order_id, alpaca_order_id, submit_ts, order_type,
limit_price, time_in_force, extended_hours}`; fill `{broker_filled_avg_price, broker_filled_qty,
broker_filled_at, synthetic_slip_pct (separate), fees}`; exit `{exit_decision_price, exit_decision_ts,
exit_order_id, per-tranche scale-out fills}`; options `{bid, ask, mid, premium}` at both ends.

---------------------------------------------------------------------------------------------------

## 7. DOC / COMMENT DISAGREEMENTS (code wins)

1. `orderParams.ts:47` "Limit at ask + 0.1 %" — input is snapshot `dailyBar.c`, not ask.
2. `bot.ts:4104` "Market during regular hours" / audit `MARKET … @ market` (`:4108`) — order is a DAY limit.
3. `bot.ts:5671` and `:4141` payload `order_type:"market"` — orders are limits (field dropped anyway).
4. `bot.ts:4778` "use the Python options_execution path via placeOptionsOrder" — `placeOptionsOrder` has no callers.
5. `bot.ts:2328-2330` `/api/bot/status` `maxPositionPct: 8, maxExposurePct: 95, dailyLossLimitPct: 5` —
   Node enforces `MAX_POSITION_SIZE 0.20`, `MAX_TOTAL_EXPOSURE 0.30`, `DAILY_LOSS_LIMIT -3` (`:350-352`).
6. `bot.ts:2593` "oldest-day pruned past 365" — code prunes the newest (`:1034-1035`).
7. `ml_model_v2.py:2270-2272` "fcntl file locking … prevent corruption" — read is outside the flock.
8. `server/exitFill.ts:24` "fillPrice … Best-known execution price at submit time (WS current price)" —
   since #35 the value passed is the confirmed fill (`bot.ts:3795-3799`).
9. `bot.ts:6583-6586` "nothing ever deletes a closed ticker's entry from voltrade_stop_state.json" —
   `bot_engine.py:2329-2331` does prune non-held tickers each manage_positions run.
10. `backtest_v2.py:63-65` "mirrors server/bot.ts's fill-tracking tiers" — live always lands in the illiquid tier.
11. `/api/bot/positions` shows `stopPrice`/`takeProfitPrice` from a formula (`bot.ts:1347-1353`) that is not
    the WS engine's stop logic (`:6952-7008`) — the displayed stop is not the enforced one.
12. Client calls `/api/bot/clock` (`client/src/pages/bot.tsx:824-826`); no such server route exists.

## 8. NEW FINDINGS NOT FOUND IN research/ (each VERIFY-LIVE before acting)

- Stock-entry phantom/synthetic records from the single 2 s poll (§1.2, §0.4).
- Synthetic haircut tier collapse due to missing `volume` on candidates (§2.1).
- ETF entry priced and recorded with the underlying's price (§1.3a).
- Tier-3 BUY price fetch hits the trading host → permanent `T3-SKIP` (§1.3f).
- TIME-EXIT reads Position fields Alpaca does not return (§1.6).
- `replaceIfBetter` treats a successful 204 cancel as failure (§5).
- `adjustStrategyWeights` always sees 0 % win rate and writes weights nothing reads (§1.8).
- `/api/bot/performance` Python branch and `/api/bot/export-trades` crash on `pnl_pct: None` (§5).
- Equity curve not recorded while killed/stopped; freezes after 365 rows (§3.2, §5).
- Tier-dispatcher CSPs (main options path) never create a feedback entry; options exits recorded at
  submit with the mark, re-recorded if the close doesn't fill (§1.3d, §1.6).
- Nightly local backup misses `/data/voltrade.db` (the audit log) — only the GitHub backup copies it (§3.2).
