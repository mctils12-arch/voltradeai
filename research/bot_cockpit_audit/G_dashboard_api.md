# G — Current Bot Dashboard (client) + every endpoint a cockpit could read

Auditor G, repo `/home/user/voltradeai` @ `8998328` (package.json v1.0.1084), audited 2026-10-15.
Scope: what the code DOES today. Nothing in the repo was edited. All paths are relative to the repo root;
`file:line` cites the code that backs each claim.

Vocabulary used below (use these names verbatim):
- **AI Engine tab**: the owner-only tab id `"bot"` labelled "AI Engine" in `client/src/pages/home.tsx`.
- **BotDashboard**: the default export of `client/src/pages/bot.tsx` (the page to be replaced).
- Panel names are the on-screen titles: **Header**, **CalendarBanner**, **Account Overview**, **PERFORMANCE**
  (contains **ML MODEL**), **Open Positions** (`EnhancedPositions`), **TODAY'S TRADES** + **OPEN ORDERS**
  (`TradingActivity`), **TRADE HISTORY** (`TradeHistoryPanel`), **Position Rules**, **AI Trade Signals**,
  **LIVE POSITIONS** (`TradeCharts`), **Security Controls**, **Activity Log**.

---

## 0. Headline findings (read these first)

1. **The page is a narrow centered column, not full-bleed.** BotDashboard renders inside
   `<main className="page-container tab-page">` (`home.tsx:502`), and `.page-container` is
   `max-width: 920px; margin: 0 auto` (`client/src/index.css:583-588`). BotDashboard's own
   `maxWidth: "1100px"` (`bot.tsx:1481`) is therefore moot: effective content width at 1440px is
   ~824px (920 − 2×32px container padding − 2×16px BotDashboard padding). ~13 stacked cards, one
   long vertical scroll. Full-bleed precedent exists: `.vt-map-page` is `position: fixed; top: 56px;
   left/right/bottom: 0` (mobile `top: 48px; bottom: 64px`) (`index.css:2265-2274`).
2. **The SSE stream is dead weight on the client.** `GET /api/bot/stream` exists (`server/bot.ts:1212-1220`)
   but **no client file opens an EventSource** (grep of `client/src` for `EventSource|/api/bot/stream`
   = 0 hits). Everything is polling.
3. **`/api/bot/clock` does not exist.** `useMarketClock()` queries it every 60s (`bot.tsx:822-838`), but no
   server route defines it (grep of `server/` = 0). In production the SPA fallback answers it with
   `index.html` 200 (`server/static.ts:42-47`), `r.json()` throws, the query errors (not retried — not a
   transient error, `queryClient.ts:50-55`), `isOpen` falls back to `false` → **the "AFTER HOURS" badge
   and every per-position "AH" badge are ALWAYS shown, even during market hours** (`bot.tsx:887-898,
   1048, 1144-1155`). The real clock exists at `GET /api/bot/market-status` (`bot.ts:3575-3587`, unused).
4. **The equity curve is drawn time-reversed.** Server keeps `equityCurve` NEWEST-FIRST (`unshift`,
   `bot.ts:1029-1035`); the canvas plots index 0 at the left (`bot.tsx:392`) and labels dates in array
   order (`bot.tsx:434-444`) → newest day on the left. (Latent server bug at the same site: once length
   >365, `shift()` removes index 0 — the day just added — so the curve would freeze; ~163 days today.)
5. **"Today:" row on every stock position is dead.** BotDashboard reads `p.change_today` and
   `p.lastday_price` (`bot.tsx:1035, 1041`), but `/api/bot/positions` never returns them (mapped object
   `bot.ts:1382-1401`) → always "+0.00%", "Prev Close" never shown, AH note never shown.
6. **Options values are off by 100× in several places.** Client recomputes `costBasis = qty*entry` and
   `marketValue = qty*current` with no contract multiplier (`bot.tsx:1039-1040`), ignoring the server's
   correct `marketValue`. `/api/trades/history` and `/api/trades/today` compute option `pnl =
   (exit-entry)*qty` with no ×100 (`routes.ts:7070-7071`, `routes.ts:6918, 6926`), and the history
   pairing assumes buy-then-sell, so a CSP (sell-to-open, buy-to-close) shows as a "standalone sell" with
   `pnl: 0` (`routes.ts:7082-7093`) and the closing buy is dropped.
7. **Displayed risk limits are hardcoded, and one is wrong.** `/api/bot/status` returns literal
   `maxPositionPct: 8, maxExposurePct: 95, dailyLossLimitPct: 5` (`bot.ts:2328-2330`). The Node loop
   enforces `DAILY_LOSS_LIMIT = -3` (`bot.ts:350`, `state.dailyLossLimit` `bot.ts:489`) and
   `risk_kill_switch.py:52` also uses −3%, yet the Security Controls card prints "−5%"
   (`bot.tsx:1873`). Four different drawdown thresholds coexist: Node kill −10% (`bot.ts:498`),
   bot_engine scan halt 18% (`system_config.py:308`), risk_kill_switch −20% (`risk_kill_switch.py:51`),
   `/api/monitoring/drawdown` hardcoded −20 with its own bands (`bot.ts:3537-3541`).
8. **PERFORMANCE silently switches definitions.** Primary path: Python over `trade_feedback` —
   `totalPnlPct` = SUM of per-trade % (`bot.ts:1696-1698`). If Python fails (mutex wait >10s, error) or
   yields 0 trades, it falls back to Alpaca where `totalPnlPct` = TODAY's account % change, open
   positions are counted as "trades", and best/worst include unrealized (`bot.ts:1758-1897`). Nothing in
   the response says which path produced it.
9. **No control has a confirmation.** Kill Switch (cancels ALL open orders at Alpaca + email), Close,
   Close All Legs, Pause all fire on one click (`bot.tsx:1453-1460, 966, 1158, 1556, 1857`). No mutation
   has `onError` → failures are silent.
10. **Errors render as zeros.** Account cards show `$0.00` while loading or on error
    (`bot.tsx:1571-1590`); status badge shows "PAUSED" while loading/erroring (`bot.tsx:1473-1498`);
    TradingActivity converts any failure (including 401) into "No trades today" (`TradingActivity.tsx:231-232`).
    There is **no data-age / freshness display anywhere** on the page (violates PREMIUM EXPERIENCE (c)).
11. **The LIVENESS ALARM (CLAUDE.md priority 1) is not on the bot page at all.** It lives only in public
    `GET /api/health` → `checks.bot.liveness` (`bot.ts:1518-1520`).
12. **Auth surface is inconsistent.** `server/bot.ts` routes are `requireOwner`, but the bot-data routes in
    `server/routes.ts` (`/api/trades/today`, `/api/orders/open`, `/api/trades/history`, `/api/positions`,
    `/api/ml/toggle`, `/api/ml/retrain`, `/api/shorts/dashboard`) only check "any logged-in session"
    (`routes.ts:576-585`), and `/api/ml/status`, `/api/bot/system-status`, `/api/bot/market-status`,
    `/api/health` have **no auth**. Any free account can toggle the bot's ML flag via `POST /api/ml/toggle`.

---

## 1. CURRENT PAGE

### 1.1 Route and auth

- SPA routes: `/app` and `/app/:rest*` → `AppDashboard` (`client/src/App.tsx:162-163`), which renders
  `Home` (`App.tsx:121`). Tabs are hash-routed: the bot page is **`/app#/bot`** (hash sync `home.tsx:232-241`,
  initial tab from hash `home.tsx:63-70`).
- Tab definition: `{ id: "bot", label: "AI Engine", requiresAuth: true, requiresOwner: true }` (`home.tsx:37`).
  Desktop: a standalone "AI Engine" button outside the Trade/Data menus, rendered only if `isOwner`
  (`home.tsx:423-432`). Mobile: owner-only tabs are filtered into the "More" sheet (not one of the four
  primary tabs) (`home.tsx:44, 552-575`).
- Client gate: `authenticated && isOwner ? <BotDashboard/> : <designed "restricted"/"Sign in" copy>`
  (`home.tsx:527-547`). `isOwner` comes from `GET /api/auth/me` with `staleTime: 60000, retry: false`, no
  refetch interval (`App.tsx:79-90`). After login, `handleLoginSuccess` defaults to the bot tab
  (`home.tsx:294`).
- Who is "owner": `requireOwner` (`server/auth.ts:96-120`) loads the session cookie, enforces 24h session
  expiry, then requires `user.email === OWNER_EMAIL` (env `OWNER_EMAIL`, defaulting to the repo owner's
  address, `auth.ts:10`). Non-owner → 403 `{error:"Owner only", message:"Automated trading is currently
  restricted to the platform owner."}`; no/expired session → 401. `/api/auth/me` returns `isOwner`
  (`auth.ts:301-331`).
- Session expiry while the page is open: `/api/auth/me` is not re-polled (`App.tsx:85`, global
  `refetchOnWindowFocus: false` `queryClient.ts:74`), so BotDashboard stays mounted and every owner
  query starts failing 401 → panels show zeros/empty (see 1.4). No "session expired" state.
- Desktop info icon links `/#bot` (landing section) (`home.tsx:437`); the mobile one links `/bot`
  (`home.tsx:480`), which is not a route → landing catch-all at the top, not the `#bot` section.

### 1.2 Layout

- Shell: fixed 56px desktop top nav (`index.css:304-319`); mobile (≤639px) 48px top bar + 64px bottom tab
  bar (`index.css:2030-2075`), page padding switches accordingly (`index.css:2196-2201`).
- Container: `.page-container` max-width 920px centered, `padding-top: calc(56px + clamp(1.5rem,4vw,2.5rem))`
  (`index.css:583-588`); `.tab-page { min-height: calc(100dvh - 56px) }` (`index.css:589-591`).
- BotDashboard root: `<div style={{ maxWidth: "1100px", margin: "0 auto", padding: "24px 16px" }}>`
  (`bot.tsx:1481`) → constrained by the 920px parent. Vertical stack of cards, `marginBottom: 20px`
  each; whole page scrolls. Inner scroll regions: Trade History (max 300px, `bot.tsx:1314`), Today's
  Trades (300px, `TradingActivity.tsx:272`), Open Orders (250px, `TradingActivity.tsx:340`), Activity Log
  (600px, `bot.tsx:1990`), Notifications popover (400px, `bot.tsx:207`). Tables have `minWidth`
  420/560/780px and scroll horizontally inside their card at 390px.
- Styling: all inline `style={{}}` objects; shared `card`/`label`/`bigNum` constants (`bot.tsx:77-79`).
  No CSS classes, no theme tokens, no shadcn primitives.

### 1.3 Data-fetch defaults (`client/src/lib/queryClient.ts`)

- Default `queryFn` = `fetch(queryKey.join("/"), {credentials:"include"})`; 401 → `null`
  (`queryClient.ts:29-44`, `:72`). BotDashboard mostly passes explicit `queryFn`s using `apiRequest`, which
  THROWS on any non-2xx (`queryClient.ts:5-26`).
- `refetchInterval: false`, `refetchOnWindowFocus: false`, `staleTime: 60_000` by default
  (`queryClient.ts:73-75`). React Query v5 (`package.json:45`): interval polling pauses in hidden tabs
  (`refetchIntervalInBackground` default false). The raw `setInterval`s in `TradingActivity` and
  `TradeChart` do NOT pause.
- Retry: up to 4 retries ONLY for 502/503/504/network errors, 3s→6s→12s→24s (`queryClient.ts:50-67`);
  4xx/500 are not retried. React Query keeps the last successful `data` on error → **silently stale**.
- Mutations: `retry: false` (`queryClient.ts:80`).

### 1.4 Panels, in render order

| # | Panel (on-screen title) | Code | Shows | Fed by (endpoint → fields) | Refresh | Loading / empty / error behaviour |
|---|---|---|---|---|---|---|
| 1 | **Header** | `bot.tsx:1484-1561` | Title "AI Trading Engine"; state badge KILLED/ACTIVE/PAUSED; "PAPER TRADING" badge (static); NotificationBell; 📥 Snapshot; Start/Pause; Kill Switch/Unkill | `/api/bot/status` → `killSwitch`, `active`; `/api/bot/notifications` | status 5s (`:1409-1413`); notifs 15s (`:1438-1442`) | Status undefined (loading/error) → badge reads **PAUSED** and Start button shows (`:1473-1474, 1546`). Silent. |
| 1a | NotificationBell popover | `bot.tsx:141-257` | Up to 100 in-memory notifications, emoji by type, unread dot, "Mark all read" | `/api/bot/notifications` (array of `{time,type,message,read}`) | 15s | Empty → "No notifications yet" (`:227-230`). Opening the bell **auto-marks all read** (`:178-181`). Types `critical`/`exit`/`research`/`upgrade` fall to the blue "info" style (`:159-164`). |
| 2 | **CalendarBanner** | `bot.tsx:260-288` | Tomorrow's market status note; upcoming closures list | `/api/bot/calendar` → `tomorrow.{status,note}`, `holidays[]` | once per mount, `staleTime 600000`, no interval (`:261-265`) | Loading/error/no `tomorrow` → renders **null** (`:267`). Server computes "tomorrow" in UTC (`bot.ts:7829-7831`) → after 8pm ET it describes the day after tomorrow; holiday names are a 2026-only table (`bot.ts:7795-7807`). |
| 3 | **Account Overview** (4 cards) | `bot.tsx:1567-1593` | Portfolio Value, Cash Available, Today's P&L ($ and %), Buying Power | `/api/bot/account` → `portfolioValue`, `cash`, `dailyPnL`, `dailyPnLPct`, `buyingPower` | 10s (`:1399-1403`) | Undefined → **`$0.00`, `+0.00 (+0.00%)`** (`:1471-1472, 1571-1590`). Error → last good value kept silently. Today's P&L number has no `$` sign (`:1583`). |
| 4 | **PERFORMANCE** (collapsible, default open) | `bot.tsx:291-740`, placed `:1596` | Metric grid: Total Trades, Total P&L (% and $ if ≠0), Win Rate (+bar), Profit Factor (∞ = "Wins only"), Avg Win, Avg Loss, Drawdown "From peak"; Best/Worst trade; canvas Equity Curve with dashed PEAK line; "Recent Trades (Last 10)" table (Ticker/Side/P&L%/Strategy/Date) | `/api/bot/performance` → `totalTrades, winRate, profitFactor, avgGain, avgLoss, totalPnlPct, totalPnlDollar, currentDrawdown ?? maxDrawdown, bestTrade, worstTrade, equityCurve[], recentTrades[] (or trades[])` (`:296-313`) | 60s (`:1424-1428`); canvas redraws on data change + window resize (`:447-460`) | Undefined → all zeros, "—", canvas text "Equity curve builds as trades complete" when <2 points (`:348-355`); empty trades → "No completed trades yet…" (`:729-733`). Server error → 200 `{error,totalTrades:0,equityCurve}` (`bot.ts:1918-1920`) → zeros, error not shown. Equity curve **time-reversed** (finding 4). Canvas uses `surfacePixelRatio()` (`:331`). Colours: equity line `#00ffd5`, peak amber. |
| 4a | **ML MODEL** (nested INSIDE PERFORMANCE's open block, `bot.tsx:735`) | `bot.tsx:745-819` | ON/OFF badge, Enable/Disable ML button, explainer copy, Status (Trained/No Model), Accuracy, Samples, Model Age (amber >48h) | `/api/ml/status` → `enabled, last_accuracy, last_samples, model_age_hours, model_exists` | 30s (`:746-749`), default queryFn | Loading → "..." for Status only. Server failure returns 200 fallback `{model_exists:false, enabled:false, error}` (`routes.ts:7134-7141`) → shows "No Model"/OFF. Collapsing PERFORMANCE unmounts it (stops polling). |
| 5 | **Open Positions** (`EnhancedPositions`) | `bot.tsx:840-1273`, placed `:1599` | "AFTER HOURS" badge (always — finding 3); count. Multi-leg option groups (bull_put_credit_spread / iron_condor / qqq_iron_condor / bear_call_credit_spread grouped by `optionMeta.underlyingTicker:strategy`, `:855-878`): underlying, strategy badge, legs, DTE→expiry, legs list, P&L, % of Max Profit, Max Profit, Max Loss, **Close All Legs**. Single positions: ticker/underlying, LONG/SHORT, STOCK SHORT badge, CALL/PUT, strategy label, qty shares/contracts, price, strike, DTE, AH badge, **Close**; Entry / Prev Close / Now; Cost / Value / P&L ($,%) / % Max Profit (short options); "Today:" %↔$ toggle | `/api/bot/positions` → `ticker, qty, side, entryPrice, currentPrice, pnl, pnlPct, asset_class, optionMeta{underlyingTicker,expiry,strike,optionType,strategy,setup,daysToExpiry}`; clock from nonexistent `/api/bot/clock` | positions 15s (`:1404-1408`); clock 60s | Empty or error → "No open positions. The bot will generate signals and trade when activated." (`:1266-1270`) — **an error looks like an empty book**. Dead "Today" row (finding 5); options Cost/Value ×100 wrong (finding 6). |
| 6 | **TODAY'S TRADES** | `TradingActivity.tsx:262-327`, placed `bot.tsx:1602` | Time, Symbol, Side (inverse-ETF buy shown as SHORT), Type badge (Iron Condor/Put/Call/Option/Short ETF/ETF/Short/Long), Qty, Entry, Exit ("OPEN"), P&L $, P&L % | `/api/trades/today` → `trades[]` = raw Alpaca filled orders + `entry_price, exit_price, pnl, pnl_pct` | own `setInterval` 30s (`:241-245`), raw `fetch` | Any failure → `{trades:[]}` → "No trades today" (`:231-232, 269-270`). Option P&L $ understated 100× (finding 6). "Iron Condor" label is assigned to ANY bracket/oco/oto order (`:31-33`). |
| 7 | **OPEN ORDERS** | `TradingActivity.tsx:330-388` | Time, Symbol, Side, Qty, Order Type, Limit Price, Status badge | `/api/orders/open` → `orders[]` (raw Alpaca open orders) | 30s, same timer | Failure → "No open orders". |
| 8 | **TRADE HISTORY** | `bot.tsx:1276-1395`, placed `:1605` | Symbol (OCC → underlying, purple), Strategy tag, Side, Qty, Entry, Exit, P&L $, P&L %, Date/Time | `/api/trades/history` → `trades[{symbol,side,shares,entryPrice,exitPrice,pnl,pnlPct,filledAt}]` (`marketOpen` ignored) | 60s (`:1277-1285`) | **Designed** loading ("loading…"), error ("Failed to load trade history."), empty ("No closed trades yet") (`:1297-1311`). `t.strategy` is never sent by the server → tag is always "Options"/"Stock" (`:1340-1342`). Only last 50 filled orders paired FIFO (`routes.ts:7019`). |
| 9 | **Position Rules** (conditional) | `bot.tsx:1619-1703` | Size-tier label; Max Positions, Sector Cap, Single Name %, Liquidity Cap %, Min Position $K; "Next tier unlocks at $X" | `/api/bot/last-scan` → `size_tier.{label,max_positions,max_sector_positions,max_single_pct,max_volume_share_pct,min_position_dollar,next_tier_at}` | 60s (`:1433-1437`) | Hidden entirely when no scan yet / no `size_tier.label`. The rest of the (large) last-scan payload is ignored. |
| 10 | **AI Trade Signals** | `bot.tsx:1706-1834` | Count; per signal: rank badge #n, ticker, action pill, sector / ⚠ sector-clash, time; reason; factor mini-grid (Mom/MR/VRP/Squz/Vol + "Strong n/5"); "Confidence" bar | `/api/bot/signals` → `[{ticker,action,reason,confidence,timestamp,type, rank?,factors?,factor_strength?,sector_clash?,sector?,tiebreak_score?}]` | 30s (`:1419-1423`) | Empty/error → a lone grey Zap icon, **no text** (`:1829-1833`) — undesigned. Time shown as `toLocaleTimeString()` only (no date). "Confidence" is the capped composite score (`Math.min(95, pick.score)`, `bot.ts:4496`) or a heuristic/default 75 for tier actions (`bot.ts:4547`), not a calibrated probability. |
| 11 | **LIVE POSITIONS** (`TradeCharts`) | `TradeChart.tsx:327-355`, placed `bot.tsx:1837-1839` | One lightweight-charts candlestick+volume chart (320px) per position, with Entry/Stop/Target/Now price lines, Phase & R-multiple, days held, timeframe buttons (1m/5m/15m/1h/1D) + refresh, footer "qty shares · $value · P&L · Peak %" | `/api/bot/positions` (same array) + `GET /api/bot/bars/:ticker?timeframe&limit` per chart | Labelled "Auto-refreshes every 30s" (`:347`) but see note | Empty → "No open positions — charts will appear…" (`:328-338`); per chart "Loading chart..." overlay and red error text (`:300-309`). **Note:** `sanitizePosition()` returns a new object every render (`client/src/lib/tradeChartGuards.ts:31-36`), so `loadBars`'s deps change every render (`TradeChart.tsx:151`) and the `[loadBars]` effects re-fetch bars and reset the 30s timer on EVERY BotDashboard re-render (`:224-232`) — each time flashing the opaque loading overlay. Option legs get `{bars: []}` (`bot.ts:3282-3285`) → an empty chart with no message; stop/target lines for options are stock-style formulas. |
| 12 | **Security Controls** | `bot.tsx:1842-1935` | Kill Switch toggle (a clickable `div`, `:1853-1857`), Daily Loss Limit "−5%", Max Position Size "8%", Max Exposure "95%"; Circuit Breaker banner + Reset (only when active); badges: Paper Trading Mode, "{n} Audit Entries", "API keys encrypted" (static), "{n} unread alerts" | `/api/bot/status` → `killSwitch, dailyLossLimitPct, maxPositionPct, maxExposurePct, circuitBreakerActive, circuitBreakerUntil, consecutiveStopLosses, auditLogCount` | 5s | Loading → hardcoded fallbacks `5 / 8 / 95` and "OFF" (`:1873-1887`). Server values themselves are hardcoded (finding 7). `auditLogCount` saturates at 2000 (SQLite trim `bot.ts:1170`). |
| 13 | **Activity Log** | `bot.tsx:1938-2003` | Newest-first 100 audit entries: time, action (blue), detail; "Copy last 50" (clipboard + textarea fallback) | `/api/bot/audit` → `[{time,action,type,detail,message}]` | 15s (`:1414-1418`) | Empty/error → "No activity yet. Start the bot to begin." (`:2000`). |

**Not imported by bot.tsx:** `SectorHeatmap` (`client/src/components/SectorHeatmap.tsx`) is used only by
`scanner.tsx`. Its fetch is one-shot (no polling) from `GET /api/market/sectors` (`routes.ts:6762-6774`,
Alpaca snapshots of 11 SPDR sector ETFs), with designed loading/error/empty+Retry states
(`SectorHeatmap.tsx:83-142`). Reusable in a cockpit as-is (it is the one bot-adjacent component that
already has designed states), but its colours are off-token.

**No hooks from `client/src/hooks/*` are used by bot.tsx** (`use-mobile.tsx`, `use-toast.ts` exist; bot.tsx
imports neither, `bot.tsx:1-12`).

### 1.5 Polling load per open BotDashboard tab (approx., steady state)

| Endpoint | Interval | Server cost per call |
|---|---|---|
| `/api/bot/status` | 5s | in-memory + `SELECT COUNT(*)` |
| `/api/bot/account` | 10s | 1 Alpaca `/v2/account` (+ possible drawdown-kill side effect, §3) |
| `/api/bot/positions` | 15s | 1 Alpaca `/v2/positions` + **2 python3 subprocesses** under the global Python mutex |
| `/api/bot/audit`, `/api/bot/notifications` | 15s | SQLite / in-memory |
| `/api/bot/signals` | 30s | in-memory |
| `/api/ml/status` | 30s | 1 python3 subprocess (no mutex) |
| `/api/trades/today` + `/api/orders/open` | 30s (setInterval, runs in background tabs) | 2 + 1 Alpaca |
| `/api/bot/performance` | 60s | 1 python3 (imports numpy via ml_model_v2) under mutex; fallback 3 Alpaca |
| `/api/bot/last-scan` | 60s | in-memory (large payload) |
| `/api/trades/history` | 60s | 2 Alpaca (`orders` + `clock`) |
| `/api/bot/clock` | 60s | 404-equivalent (SPA HTML) |
| `/api/bot/bars/:t` ×N positions | every re-render (see panel 11) | 1 Alpaca data API each |

≈18 Alpaca trading-API calls/min/tab before bars (Alpaca paper limit is 200/min/account and the bot's
own loop shares it), ~10 Python subprocess spawns/min/tab. No server-side cache on any of these.

---

## 2. CONTROLS (everything the page can DO)

| Control (on-screen) | Where | Request | Server effect | Guard / feedback |
|---|---|---|---|---|
| **Start** | `bot.tsx:1547-1549` | `POST /api/bot/start` | `startBotActivity("")`: refuses if kill switch on (400), sets `state.active=true`, audit START, notify, starts WebSocket stream after 2s; clears `ownerStopped` (`bot.ts:2340-2360`) | Disabled when killed. No confirm, no error UI. Invalidates status+audit. |
| **Pause** | `bot.tsx:1551-1553` | `POST /api/bot/stop` | `ownerStopped=true`, `state.active=false`, stops stream, audit STOP (`bot.ts:2362-2370`). Auto-resume will never re-activate an owner-stopped bot. In-memory only: **a deploy boots active** (`bot.ts:486`, comment `bot.ts:2337-2339`). | No confirm. |
| **Kill Switch / Unkill** (header button AND Security Controls toggle `div`) | `bot.tsx:1556-1558`, `:1853-1857` | `POST /api/bot/kill` (TOGGLE, no body) | ON: persists kill file, emails alert, `active=false`, audit "KILL SWITCH ON", **DELETE `/v2/orders` (cancels every open order)**. OFF: audit only — does NOT re-activate (`bot.ts:2372-2387`) | No confirm. The toggle is a `div` (no keyboard/role). Unkill leaves state PAUSED; user must press Start. |
| **Close** (per position) | `bot.tsx:1156-1166` | `POST /api/bot/close {ticker}` | `DELETE /v2/positions/{TICKER}` (market liquidation), removes from position monitor, audit CLOSE (`bot.ts:3262-3275`) | No confirm; failure silent. |
| **Close All Legs** (multi-leg group) | `bot.tsx:965-974` | N × `POST /api/bot/close` fired concurrently (`legs.forEach`) | Each leg liquidated independently (no combo order) — a partial failure leaves a naked leg | No confirm; no aggregate result. |
| **Reset Circuit Breaker** (visible only when active) | `bot.tsx:1907-1916` | `POST /api/bot/circuit-breaker/reset` | zeroes `circuitBreakerUntil` and `consecutiveStopLosses`, audit (`bot.ts:2389-2396`) | No confirm. |
| **Enable ML / Disable ML** | `bot.tsx:780-788` | `POST /api/ml/toggle {enabled}` | runs `python3 ml_toggle.py enable|disable` → writes `DATA_DIR/ml_toggle.json` (`routes.ts:7145-7160`, `ml_toggle.py:17-33`). **No reader in the trading path**: grep of `bot_engine.py`, `ml_model_v2.py`, `system_config.py`, `server/bot.ts` for `ml_toggle`/`ML_ENABLED` = 0; only `ml_status.py` and the daemon's status/toggle impls read it (`voltrade_daemon.py:271-272, 346-351, 499`). | Auth: any logged-in user, not owner. Effectively a cosmetic flag. |
| **Mark all read** / opening the bell | `bot.tsx:178-181, 220-224` | `POST /api/bot/notifications/read` | flips `read=true` on the in-memory list (`bot.ts:1439-1442`) | Automatic on bell open. |
| **📥 Snapshot** | `bot.tsx:1514-1544` | `GET /api/system/snapshot` (raw fetch, downloads JSON) | heavy read: daemon RPC, 500 audit rows, 2 Alpaca calls, one ~20s Python subprocess importing ml_model_v2/risk_kill_switch/system_config/macro_data/stress_index/probability_engine/csp_universe (`bot.ts:2006-2243`) | Errors via `alert()`. |
| **Copy last 50** | `bot.tsx:1942-1987` | none (clipboard) | — | "Copied!"/"Copy failed" text. |
| Today %/$ toggle | `bot.tsx:1235-1248` | none | — | (field is dead, finding 5) |
| Chart timeframe 1m/5m/15m/1h/1D + ↻ | `TradeChart.tsx:277-295` | `GET /api/bot/bars/:ticker` | read | — |
| Tooltips (`Tip`) | `bot.tsx:43-74` | none | hover-only (`onMouseEnter`) → unreachable on touch | — |

**Acting endpoints that exist but the page never calls** (relevant to "read-only vs can act"):
`POST /api/bot/trade` (manual market/limit order; blocks new shorts; `bot.ts:3189-3260`),
`POST /api/bot/run-now` (forces a Tier-2 scan, respects `tier2Running` lock; `bot.ts:7956-7966`),
`POST /api/bot/refresh-signals` (runs `analyze.py` serially for 10 hard-coded mega-caps; `bot.ts:3673-3678`),
`POST /api/bot/backtest {ticker,strategy,years}` (≤120s Python; **body values are interpolated unescaped
inside double quotes of a shell command**, `bot.ts:3681-3697` — `$(…)` would execute; owner-only, but a
cockpit form must not pass free text through), `POST /api/bot/backup` (copies DATA_DIR json/pkl/db, keeps
5; `bot.ts:1962-2000`), `POST /api/ml/retrain` (any logged-in user; ≤300s, auto-enables the toggle on
success; `routes.ts:7162-…`), `POST /api/system/verify-change` (read-only string search despite POST;
`bot.ts:2286-2309`).

Hidden side effect on a READ: **`GET /api/bot/account` can trip the max-drawdown kill switch** — it runs
`evaluateDrawdown` and, on a valid read below −10% from peak, sets `killSwitch`, emails, notifies and
cancels all orders (`bot.ts:1235-1250`). A cockpit that polls it is not purely observational.

---

## 3. ENDPOINT CATALOG

Legend: **Used** = called by bot.tsx or a component it renders. Auth: `owner` = `requireOwner`;
`session` = any logged-in user via `_checkSession` (`routes.ts:576-585`); `none` = public.
"Alpaca" = live HTTP to `https://paper-api.alpaca.markets` via `alpaca()` (`bot.ts:324-345`, 15s abort, no
cache, no rate limiter) or routes.ts raw `fetch` with `alpacaHeaders` (`routes.ts:600, 6704`).
"Python (mutex)" = `execPythonSerialized` — one global lock for non-trivial scripts, waits up to the call's
`timeout` then throws (`bot.ts:239-273`; trivial = <150 chars and no heavy import, `bot.ts:233-237`).
"pythonCall" = daemon RPC with subprocess fallback (`bot.ts:151`).

### 3.1 server/bot.ts

| Method + path | Auth | file:line | Data source | Response shape | Cost / latency | Used |
|---|---|---|---|---|---|---|
| GET `/api/bot/stream` | owner | bot.ts:1212-1220 | in-memory `sseClients` | `text/event-stream`; see §4 | holds connection | **No** |
| GET `/api/bot/account` | owner | bot.ts:1223-1268 | Alpaca `/v2/account` | `{accountNumber, cash, portfolioValue, buyingPower, equity, lastEquity, dailyPnL, dailyPnLPct, status, tradingBlocked, mode:"paper"}`; 500 `{error}` | 1 Alpaca; **may trip drawdown kill (side effect)** | Yes (10s) |
| GET `/api/bot/positions` | owner | bot.ts:1271-1407 | Alpaca `/v2/positions` + `voltrade_stop_state.json` + `voltrade_options_state.json` (each via a python3 one-liner, fs fallback) | array of `{ticker, qty, side (display side, inverse ETFs flipped), entryPrice, currentPrice, marketValue, pnl, pnlPct (×100), stopPrice, takeProfitPrice|null, phase, rMultiple, highestPnl, daysHeld, asset_class, isOption, optionMeta|null{underlyingTicker, expiry, strike, optionType CALL|PUT, strategy, setup, daysToExpiry}}`; non-array Alpaca → `[]`; 500 `{error}` | 1 Alpaca + 2 Python (mutex, 5s wait each, then fs fallback). `stopPrice`/`takeProfitPrice` are **synthesized** from `current_stop_pct` (default 2.5%) — phase≥2 trails from current price, TP = entry×(1+1.5×atr%) (`:1347-1353`), not broker orders. No `change_today`/`lastday_price`. | Yes (15s) |
| GET `/api/bot/history` | owner | bot.ts:1411-1429 | Alpaca `/v2/orders?status=closed&limit=50` | array `{ticker, side, qty, price, status, submittedAt, filledAt, type}` (includes canceled/rejected) | 1 Alpaca | **No** |
| GET `/api/bot/notifications` | owner | bot.ts:1435-1437 | in-memory (≤100, lost on deploy, `bot.ts:527-533`) | array `{time, type, message, read}`; types emitted: alert, critical, earnings, exit, research, system, trade, upgrade | trivial | Yes (15s) |
| POST `/api/bot/notifications/read` | owner | bot.ts:1439-1442 | in-memory | `{ok:true}` | trivial | Yes |
| GET `/api/health` | **none** | bot.ts:1445-1657 | Node, SQLite, Alpaca `/v2/account`, trivial python ping, liveness file, scanner counter, feed dead-air, licensing, crash stats, memory/cgroup | `{timestamp, status:"ok"|"degraded", checks:{server{status,uptime_s,heap_used_mb,rss_mb}, database, alpaca{status,account_status}, python, bot{status:"killed"|"active"|"stopped", equityPeak, drawdownPct (string or "N/A"), liveness{dark, marketHours?, wallHours?, detail?}, autoResume{eligible,outcome,classification,basis,reason,consecutiveOk,required,lastEvaluation,nextCheck,lastResume}}, scanner{status,consecutiveFailures}, feeds{status,dead,feeds,detail?,gates_top_level_status:false}, licensing, process, memory{heapUsedMB…pressure}}, serving{ok,gates,failing,note}}`. HTTP 503 only if server/database fail (`server/healthGate.ts:73-85`) | 1 Alpaca + ping; also records the pipeline-health time series | Not by bot.tsx (used by `pipelineHealthDashboard.tsx`) |
| GET `/api/bot/performance` | owner | bot.ts:1660-1921 | Python (mutex, 10s) over `TRADE_FEEDBACK_PATH` excluding `_seed` records; fallback Alpaca orders (today, 200) + positions + account | `{totalTrades, totalFills, seededTrades?, winRate, avgGain, avgLoss, totalPnlPct, totalPnlDollar? (fallback only), profitFactor (number or "inf"), byStrategy{strat:{wins,losses,total_pnl}}, recentTrades[] (primary: raw feedback records newest 20 incl. entry_features; fallback: {ticker,side,pnl_pct,timestamp,strategy}), realisticPnlPct, avgSlippagePct, totalSlippageCost, slippageGapPct, bestTrade{ticker,pnlPct}, worstTrade, equityCurve[{date,value,pnl}] NEWEST-FIRST, equityPeak, currentDrawdown}`; on error 200 `{error,totalTrades:0,equityCurve}` | Python ~0.5-1s+ (numpy import) or up to 10s mutex wait; fallback 3 Alpaca. **Two incompatible definitions** of totalPnlPct (finding 8). | Yes (60s) |
| GET `/api/bot/export-trades` | owner | bot.ts:1924-1959 | Python (mutex) over trade_feedback (INCLUDING seeded records) | CSV: Date, Ticker, Side, P&L %, Score, Strategy, Holding Days, Won | ~0.3s | **No** |
| POST `/api/bot/backup` | owner | bot.ts:1962-2000 | Python (mutex, 30s) | `{backed_up, files[], path, total_backups}` | disk copy | **No** |
| GET `/api/system/snapshot` | owner | bot.ts:2006-2243 | daemon RPC `health`, SQLite audit (500), Alpaca positions+account, big Python script (pythonCall, 20s), timing + daemon trace files | `{snapshot_version:"1.1", generated_at, git_head_expected, server{…}, bot_state{active,kill_switch,daily_pnl,daily_loss_limit,position_size_multiplier,min_score_threshold,max_drawdown_pct,equity_peak,consecutive_stop_losses,circuit_breaker_until}, daemon{…}, audit_log{total_entries,recent_errors[],recent_backoffs,last_scan_success,last_scan_failure}, scan_stats{success,error,sigkill,last_24h}, positions{count,deployment_pct,by_ticker,equity,cash,buying_power}, kill_switches, ml{feedback counts, model_age_hours,…}, config_files{base_config,current_adaptive,macro{regime,vxx_ratio,spy_vs_ma50,vix,data_quality}}, stress_index, floor_state, tier_engine{tiers_enabled}, probability_engine, last_scan_timings?, daemon_trace?}`; `Content-Disposition: attachment` | heaviest endpoint (seconds) | Yes (button) |
| GET `/api/system/health-check` | owner | bot.ts:2245-2284 | SQLite audit (200), process RSS, daemon RPC | `{timestamp, overall:"critical"|"warning"|"healthy", issues_count, warnings_count, issues[{severity,category,detail}], warnings[]}` (categories: scan_failures, no_successful_scans, stale_scans, memory_pressure (RSS>450MB crit / >380 warn — stale thresholds vs 1GB daemon budget), daemon_down) | cheap + 1 RPC | **No** |
| POST `/api/system/verify-change` | owner | bot.ts:2286-2309 | reads 13 source files | `{verified[], missing[], checked_at}` | cheap | **No** |
| GET `/api/bot/status` | owner | bot.ts:2311-2336 | in-memory `state` + SQLite count | `{active, killSwitch, dailyLossLimit (−3), auditLogCount (≤2000), mode:"paper", equityPeak, maxDrawdownPct (−10), autoResume{eligible, latchReason, outcome, classification, basis, reason, consecutiveOk, required, lastEvaluation, nextCheck, numbers|null{equity,peak,drawdownPct,resumeAbovePct,latchedMinutes,marketHoursSinceLatch,rebaseAfterMarketHours,reportedDrop,unrealizedPl,realizedPl,explainedLoss,explainedRatio}, lastResume|null}, unreadNotifications, circuitBreakerActive, circuitBreakerUntil|null, consecutiveStopLosses, maxPositionPct: 8 (literal), maxExposurePct: 95 (literal), dailyLossLimitPct: 5 (literal)}` | trivial | Yes (5s) |
| POST `/api/bot/start` | owner | bot.ts:2354-2360 | in-memory | `{ok,active:true}` / 400 `{error}` | — | Yes |
| POST `/api/bot/stop` | owner | bot.ts:2362-2370 | in-memory | `{ok,active:false}` | — | Yes |
| POST `/api/bot/kill` | owner | bot.ts:2372-2387 | state + kill file + Alpaca DELETE `/v2/orders` + email | `{ok, killSwitch}` | 1 Alpaca when turning ON | Yes |
| POST `/api/bot/circuit-breaker/reset` | owner | bot.ts:2389-2396 | in-memory | `{ok, circuitBreakerUntil:0, consecutiveStopLosses:0}` | — | Yes (conditional) |
| GET `/api/bot/audit` | owner | bot.ts:2399-2402 | SQLite `audit_log` newest 100 (table trimmed to 2000, `bot.ts:1170`) | array `{time, action, type, detail, message}` (action==type, detail==message, ≤500 chars) | cheap | Yes (15s) |
| GET `/api/diag/:probe` | `x-diag-token` (404 unless DIAG_TOKEN set) | bot.ts:2426-… | many | sanitized probe payloads: audit, ml, daemon, positions (summary), account, equity_curve, reconstruct_pnl, orders, **scanner** (degraded, consecutiveFailures, lastFailureDetail…), timings, archive, shadow, shadow_bands, … | — | **No** (not browser-reachable without token) |
| POST `/api/bot/trade` | owner | bot.ts:3189-3260 | Alpaca | Alpaca order object / 400 `{error}` | 2-3 Alpaca | **No** |
| POST `/api/bot/close` | owner | bot.ts:3262-3275 | Alpaca DELETE position | `{ok:true}` / 500 `{error}` | 1 Alpaca | Yes |
| GET `/api/bot/bars/:ticker` | owner | bot.ts:3277-3326 | `data.alpaca.markets/v2/stocks/{T}/bars?feed=iex&adjustment=split` (raw fetch, **no timeout, no `r.ok` check** → API error = `bars:[]` 200) | `{bars[{time(unix s),open,high,low,close,volume}], ticker, timeframe}`; OCC symbol → `{bars:[], ticker, timeframe:"1Day", isOptionSymbol:true}` | 1 data-API call; IEX-only prints (thin for small caps) | Yes (TradeChart) |
| GET `/api/daemon/health` | owner | bot.ts:3328-3343 | daemon RPC `health` | `{alive:true, uptime_seconds, rss_mb, max_rss_mb, modules_loaded[], active_dispatches, active_dispatch_detail, layer2_prefetch, layer1_stats, surface_score_stats, pid}` (`voltrade_daemon.py:442-461`) or `{alive:false, reason}` | ~3ms RPC | **No** |
| GET `/api/monitoring/overview` | owner | bot.ts:3347-3402 | daemon RPC, pythonCall `risk_status`, Alpaca account, pythonCall `ml_status` | `{timestamp, daemon{alive,…}, kill_switches{peak_equity, consecutive_losses, consecutive_loss_limit, daily_pnl_pct, daily_loss_limit_pct, portfolio_dd_kill_pct, vxx_regime_kill_threshold, min_free_bp} (default {active:0,warnings:[]}), memory{rss_mb}, drawdown{current_pct, kill_threshold_pct (−10), proximity_pct, status OK|WARNING|CRITICAL|UNKNOWN}, ml{…ml_status shape}}` (`risk_kill_switch.py:708-734`, `server/drawdownGuard.ts:100-107`) | 1 Alpaca + up to 2 Python | **No** |
| GET `/api/monitoring/tier-pnl?days=30` | owner | bot.ts:3416-3518 | pythonCall over trade_feedback | `{days, tiers:{csp_options|etf|stocks|vrp|other:{trade_count (live), total_pnl_pct, avg_pnl_pct, win_rate, seeded{count,total_pnl_pct,win_rate}}}}` | Python | **No** |
| GET `/api/monitoring/drawdown` | owner | bot.ts:3521-3546 | Alpaca account + Python `risk_kill_switch.get_peak_equity` (a DIFFERENT peak file from Node's `state.equityPeak`) | `{equity, peak, drawdown_pct, kill_threshold_pct: -20 (literal), status KILLED|CRITICAL|WARNING|NOTICE|OK}` | 1 Alpaca + Python | **No** |
| GET `/api/monitoring/caches` | owner | bot.ts:3549-3572 | pythonCall glob `/tmp/voltrade_*.json` | `{caches[{path,size_kb}]}` (top 20) | Python | **No** |
| GET `/api/bot/market-status` | **none** | bot.ts:3575-3587 | Alpaca `/v2/clock` | `{isOpen, nextOpen, nextClose, timestamp}` | 1 Alpaca | **No** (the clock bot.tsx needs) |
| GET `/api/bot/signals` | owner | bot.ts:3668-3670 | in-memory `signals` (≤30) written by Tier-2 scan (`bot.ts:4481-4560`) and by `generateSignals` (`bot.ts:3596-3665`) | array `{ticker, action, reason, confidence, timestamp, type buy|sell|neutral|stop, rank?, factors?{momentum,mean_reversion,vrp,squeeze,volume}, factor_strength?, sector_clash?, sector?, tiebreak_score?}`. 1h expiry is pruned ONLY at the end of a successful Tier-2 scan (`bot.ts:4556-4559`) — scans that error return earlier (`bot.ts:4440-4446`), so **during a scanner outage old signals keep showing as "active"** | trivial | Yes (30s) |
| POST `/api/bot/refresh-signals` | owner | bot.ts:3673-3678 | Python (mutex, 30s each × 10 tickers serially) | `{ok, count}` | up to minutes | **No** |
| POST `/api/bot/backtest` | owner | bot.ts:3681-3697 | Python `backtest.py` (mutex, 120s) | backtest_v2 response (`results[]` with strategy/sharpe/totalReturn/maxDrawdown…; error `{ok:false,status:"error",message,request,results:[]}`, `backtest.py:39-52`) | ≤120s; shell-interpolated body | **No** |
| GET `/api/bot/last-scan` | owner | bot.ts:7772-7774 | in-memory `lastScanResult` (null after every deploy until first scan; `bot.ts:4182, 4478, 6024, 6187`) | `{message}` before first scan, else scan_market's return (`bot_engine.py:4005-4072`): `{timestamp (naive local ISO), scanned, filtered, deep_analyzed, portfolio_value, cash, current_positions, slots_available, options_slots_available, new_trades[], options_trades_added, position_actions[], upgrade_candidates[], top_10[{ticker,price,score,change_pct,side,action_label,reasons[≤3],rank,tiebreak_score,factor_strength,sector_clash,sector,factors{…}}], third_leg, convexity_overlay, intraday_shorts, options_management, spy_floor, defensive_floor, tier_actions[{tier,action,ticker,strategy,size_pct,reason,metadata}], tier_stats, kill_status, kill_suppressed_trades, tier_kill_status, scan_phase_times, size_tier{…}, data_source_errors}`; halted scans add `{halted, halt_reason, peak_equity, current_equity, dd_pct, dd_event?}` | trivial (large JSON) | Yes (60s, only `size_tier` read) |
| GET `/api/bot/calendar` | owner | bot.ts:7777-7853 | Alpaca `/v2/calendar` (30 days) | `{tomorrow{status open|weekend|holiday|early_close, note, date}, holidays[{date,name}], earlyCloses[{date,close}], tradingDaysNext30}` | 1 Alpaca | Yes (once) |
| POST `/api/bot/run-now` | owner | bot.ts:7956-7966 | Tier-2 scan | `{message}` immediately; scan runs async | heavy | **No** |
| GET `/api/bot/diagnostics` | owner | bot.ts:7969-7984 | Python (mutex, 15s) `diagnostics.run_diagnostics` + `get_auto_fix_params` | `{report{timestamp, overall_status healthy|warning|degraded, problems[], warnings[], auto_fixes[], recommendations[]}, auto_fix{…}}` or `{error}` (200) | Python | **No** |
| GET `/api/bot/ml-status` | owner | bot.ts:7988-8021 | Python (mutex, 10s) | `{model_exists, retrain_schedule, retrain_threshold_hours, model_age_hours?, needs_retrain?, total_fills, learned_weights?}` or `{error}` (200) | Python | **No** |

### 3.2 server/routes.ts (bot-adjacent)

| Method + path | Auth | file:line | Data source | Response shape | Used |
|---|---|---|---|---|---|
| GET `/api/bot/system-status` | **none** | routes.ts:6575-6595 | `python3 system_status.py` (15s, no mutex) | `{generated_at, schema_version:"system-status-v1", overall ok|degraded|broken, summary{total,passed,failed,warned}, checks{size_tier_module, adaptive_params_overlay, size_portfolio_respects_tier, sector_correlation_override, top_10_enrichment, tie_breaking_present, scan_caches, …: {status pass|fail|warn, message, details}}}` | **No** |
| GET `/api/market/sectors` | none | routes.ts:6762-6774 | Alpaca data snapshots (11 sector ETFs) | raw Alpaca snapshot map | No (scanner.tsx's SectorHeatmap) |
| GET `/api/trades/today` | session | routes.ts:6845-6944 | Alpaca orders (closed, since 4am ET, 200) + positions | `{trades:[…raw Alpaca order fields, entry_price, exit_price, pnl, pnl_pct]}`; errors `{error, trades:[]}` | Yes (TradingActivity, 30s) |
| GET `/api/orders/open` | session | routes.ts:6947-6965 | Alpaca `/v2/orders?status=open&limit=200` | `{orders:[raw Alpaca orders]}` | Yes (TradingActivity, 30s) |
| GET `/api/positions` | session | routes.ts:6968-6986 | Alpaca positions | `{positions:[raw]}` | **No** |
| GET `/api/shorts/dashboard` | session | routes.ts:6989-7010 | python `intraday_shorts.get_dashboard_data()` | `{enabled,total_trades,open_trades,win_rate,avg_pnl_pct,total_pnl_pct,total_pnl_dollar,recent_trades[],strategy_status,…}` (fabricated-zero fallback with `error` on failure) | **No** |
| GET `/api/trades/history` | session | routes.ts:7012-7120 | Alpaca `/v2/orders?status=filled&limit=50` + `/v2/clock` | `{trades:[{symbol, side:"SELL", shares, entryPrice, exitPrice, pnl, pnlPct, filledAt}], marketOpen}` — FIFO buy→sell pairing; unmatched sell → `pnl:0`; no options multiplier | Yes (60s) |
| GET `/api/ml/status` | **none** | routes.ts:7127-7143 | `python3 ml_status.py` (no mutex) | `{model_exists, model_age_hours, enabled, last_accuracy, last_samples, last_features, last_status, last_error, last_traceback, last_steps, last_train_time, contributes_to_cagr, note}` (`ml_status.py:36-52`) | Yes (30s) |
| POST `/api/ml/toggle` | session | routes.ts:7145-7160 | `python3 ml_toggle.py` | `{enabled, status:"ok"}` | Yes |
| POST `/api/ml/retrain` | session | routes.ts:7162-… | `python3 ml_retrain_safe.py` (300s, background) | `{status:"started", message}` | **No** |
| GET `/api/data/pipeline-health-dashboard` | none | routes.ts:1750-1760 | JSONL written by `/api/health` (`server/pipelineHealthHistory.ts:85`) | `{generated_at, warming_up, window_24h, window_7d}`, each window built from snapshots `{t, status, database_ok, alpaca_ok, python_ok, scanner_ok, scanner_consecutive_failures, licensing_ok, bot_status, bot_liveness_dark, heap_used_mb, rss_mb}` (`pipelineHealthHistory.ts:39-52`) incl. `degraded_counts` | No (datamap overlay) |
| GET `/api/auth/me` | none | server/auth.ts:301-331 | SQLite | `{authenticated, email, role, isOwner, tier, subscriptionStatus, subscriptionPeriodEnd}` | Yes (App.tsx) |

**Called by the page but not defined anywhere:** `GET /api/bot/clock` (`bot.tsx:824-826`).

**Unused by any UI** (exists, owner-reachable): `/api/bot/stream`, `/api/bot/history`, `/api/bot/export-trades`,
`/api/bot/backup`, `/api/system/health-check`, `/api/system/verify-change`, `/api/bot/trade`,
`/api/daemon/health`, `/api/monitoring/overview`, `/api/monitoring/tier-pnl`, `/api/monitoring/drawdown`,
`/api/monitoring/caches`, `/api/bot/market-status`, `/api/bot/refresh-signals`, `/api/bot/backtest`,
`/api/bot/run-now`, `/api/bot/diagnostics`, `/api/bot/ml-status`, `/api/bot/system-status`,
`/api/ml/retrain`, `/api/positions`, `/api/shorts/dashboard`.

---

## 4. SSE STREAM — `GET /api/bot/stream`

- Handler: `writeHead(200, {Content-Type: text/event-stream, Cache-Control: no-cache, Connection:
  keep-alive})`, add `res` to `sseClients`, remove on `req.close` (`bot.ts:1212-1220`). No initial event,
  no `retry:` field, no `event:` names, no `id:`, **no heartbeat/comment pings**.
- `broadcastSSE(data)` writes `data: ${JSON.stringify(data)}\n\n` to every client (`bot.ts:504-508`). It has
  exactly **one caller: `audit()`** (`bot.ts:510-517`). So the stream is exactly the audit log, live.
- Payload of every event: `{ time: ISO, type: <ACTION>, action: <ACTION>, detail: <string>, message:
  <string> }` (`bot.ts:511`). (Persisted copy is truncated to 500 chars; the SSE copy is not.)
- `type` values = the first argument of the 253 literal `audit("…")` calls in bot.ts plus
  `'MANIPULATION'` (`bot.ts:5810`) and the crash handlers' `FATAL-REJECTION` / `FATAL-EXCEPTION`
  (`server/crashVisibility.ts:54-55`, wired `bot.ts:1209`). Grouped:
  - Lifecycle/system: STARTUP, SHUTDOWN, SHUTDOWN-ERROR, SYSTEM, RULES, SCHEDULE, EXECUTION, START, STOP, UPGRADE, UPGRADE-ERROR, TMP-CLEANUP, DB-SLOW-WRITE, EVENTLOOP-LAG, EMAIL, BACKUP, BACKUP-WARN, BACKUP-ERROR, FATAL-REJECTION, FATAL-EXCEPTION
  - Kill/halts/risk: KILL SWITCH ON, KILL SWITCH OFF, KILL-SWITCH, KILL-WARN, KILL SWITCH AUTO-RESUME, KILL-AUTO-RESUME-EVAL, DRAWDOWN-KILL, DD-HALT, TIER-KILL, CIRCUIT-BREAKER, CIRCUIT-BREAKER-RESET, DIAGNOSTIC, DIAGNOSTIC-PAUSE, WEEKLY-LIMIT, WEEKLY-WARNING, HALT-SKIP, HALT-CHECK-WARN, EQUITY-READ-INVALID, HEAT-CAP, CORR-CAP, SIZE-CAP, BLOCKED, SHORT-BLOCKED, TICKER-BLOCKED, MANIPULATION
  - Tier scheduler: TIER2, TIER2-ERROR, TIER2-FAIL, TIER2-BACKOFF, TIER2-LIMIT, TIER2-DISABLED, TIER3, TIER3-ERROR, TIER3-DIAG, TIER3-DISABLED, TIER3-ML-ERROR, TIER3-MANIP-ERROR, TIER3-COT, TIER3-FORM4BULK, TIER3-PORTDWELL, TIERS, TIER-DISPATCH-ERR, TIER-DEFER, T, T3, T3-SKIP, T3-FAIL, T4, T4-FAIL, T4-ERR
  - Trading/orders: TRADE, TRADE-ERROR, CLOSE, SIGNAL, REFRESH, QUEUE, REPLACE, SWEEP, SKIP, DUP-SKIP, MAX-RETRY, PARTIAL-FILL, ORDER-REJECTED, FILL-CHECK-WARN, INSTRUMENT, ETF-TRADE, ETF-FALLBACK, TIME-EXIT, TIME-EXIT-ERROR, LIQUIDATE, LIQUIDATE-ALL, LIQUIDATE-FAIL, LIQUIDATE-ERROR, SHORTS, SHORTS-ERROR
  - Morning queue: MORNING, MORNING-TRADE, MORNING-STALE, MORNING-SKIP, MORNING-GAP, MORNING-ERROR, MORNING-BLOCKED, MORNING-ADJUST
  - Options: OPTIONS, OPTIONS-TRADE, OPTIONS-EXEC, OPTIONS-SKIP, OPTIONS-ERROR, OPTIONS-FALLBACK, OPTIONS-SLOT-FULL, OPTIONS-SCANNER-EXEC, CSP-SKIP, CC-UNWIND, CC-UNWIND-WARN, CC-UNWIND-ERROR
  - Position monitor / websocket exits: POS-MONITOR, POS-MONITOR-SYNC, POS-WARN, POS-QTY-CORRECTED, POS-KILL, POS-KILL-SKIP, POS-KILL-WARN, POS-KILL-ERROR, POS-KILL-OPTION-HOLD, WS-EXIT, WS-EXIT-WARN, WS-EXIT-ERROR, WS-SCALE-OUT, WS-SCALE-OUT-WARN, WS-SCALE-OUT-ERROR, STREAM, STREAM-ERROR, STREAM-SIGNAL, STREAM-PRIORITY, STREAM-DISCONNECT
  - Learning/research: LEARN, LEARN-ERROR, RETRAIN, RETRAIN-ERROR, RESEARCH, RESEARCH-ERROR, BACKTEST, SHADOW-BACKFILL, SHADOW-BACKFILL-ERROR
- Nothing else is pushed: no position, account, P&L, signal-object, or status-change events (state
  changes only appear as the audit TEXT of START/STOP/KILL…).
- **Client reconnection behaviour: none — there is no client.** A cockpit must build it (EventSource
  auto-reconnects, but there is no `id:`/Last-Event-ID replay, so missed events must be back-filled from
  `GET /api/bot/audit`).
- **Risk to verify before relying on it:** `compression()` is mounted globally (`server/index.ts:48`) and the
  SSE handler never calls `res.flush()` (grep: 0). The `compression` package documents that SSE needs
  `res.flush()`; text/* responses are normally compressible, so events may be held in the gzip buffer.
  Unverified at runtime here (node_modules not installed). Railway/edge idle timeouts with no heartbeat are
  a second risk.

---

## 5. DESIGN CONSTRAINTS a replacement page must honor

From **DESIGN.md** (binding for any `client/` PR):
1. Works at **390 / 768 / 1440**; the human reviews on a Galaxy S24 ("if it fails on the phone, it failed").
2. **Full-viewport immersive pages**: 100% width, full height below the nav, no letterboxing, no page-level
   scroll caused by the page. Desktop nav 56px fixed; mobile (≤639px) 48px top + **64px bottom tab bar —
   content must end above it**; prefer `100dvh`. Precedent class: `.vt-map-page` (`index.css:2265-2274`) and
   the `body.vt-map-fullscreen` mode that hides navs (`index.css:3048-3054`, toggled `datamap.tsx:3123-3128`).
3. No permanent overlays covering content; controls collapse (phone: collapsed by default).
4. Alive on first load — live content with zero clicks.
5. **Touch targets ≥ 44×44px on phone**; desktop hover/tooltips/keyboard; Escape closes popovers. (Current
   bot.tsx fails this: Close ≈24px tall, timeframe buttons ≈20px, kill toggle 44×24, hover-only tooltips.)
6. **Theme tokens only** — `--bg-primary #050a13`, `--bg-secondary #0a1628`, `--bg-card rgba(15,29,51,.6)`,
   `--bg-card-hover`, `--text-primary #eef3fb`, `--text-secondary #b3c2d8`, `--text-tertiary #6680a0`,
   `--accent #4d9fff`, `--accent-bright #7cc4ff`, `--accent-green #4ade80`, `--accent-red #ff5a6e`,
   `--accent-orange #fbb24c`, `--accent-purple #c084fc`, `--font-body Geist`, `--font-mono Geist Mono`.
   (Current page uses three off-token palettes: iOS `#30d158/#ff453a/#ffd60a/#bf5af2/#64d2ff` + `#c8d6e5/
   #4a5c70/#a1a1a6` + 'JetBrains Mono' in bot.tsx; GitHub-dark `#0d1117/#161b22/#21262d/#8b949e/#ff4444/
   #00ffd5` in TradeChart.tsx.)
7. Hover/active/loading states on every interactive element; **designed empty / error / awaiting states**.
8. No loading state >30s without resolving to a designed status ("eternal spinners are a failed build").
9. Live data degrades to **last-known state with a visible timestamp**, never an empty panel.
10. Feature checklist (first load / error / empty / honest limits / heavy work client-side).
11. Self-see rule: every registered panel/control must be reachable on screen at all three widths.

From **CLAUDE.md**: PREMIUM EXPERIENCE STANDARD — (b) skeletons, progressive loading, zero jank; (c) **every
number visibly carries freshness, provenance, and confidence**; (e) mobile-flawless at 390; (f) "would a
paying data customer screenshot this and trust it?"; correctness beats polish. UNITS PREFERENCE applies
only to distance/length/speed/temperature (none on the bot page today). If the cockpit draws on/around a
sphere or uses map layers, the RENDERING & MOTION LAW applies (one rAF loop via `client/src/render/
frameCore.ts`, no visual updates on map events, `dispose()`/`maxFeatures`/`vramBudget`). PROMOTION RULE 6:
run `npm run build && npm run visual` at 390/768/1440, review screenshots, describe them in the PR; bump
`package.json` version; one logical change per PR; T-CLIENT territory (`client/src/**`, `index.css`,
`scripts/visual_check.mjs`); `server/routes.ts` and `package.json` are SHARED (last commit, minimal).

**Mechanical gates a replacement will trip (plan for these in the same PR):**
- `scripts/visual_check.mjs` has **no bot page** in `PAGES` (`visual_check.mjs:48-201`), and the fixture map
  is GLOBAL with `"/api/auth/me": { authenticated: false }` (`visual_check.mjs:247`); unknown `/api/*`
  return `{}` (`visual_check.mjs:2194-2197`). There is no per-page fixture override: a cockpit PAGES entry
  needs owner-authenticated fixtures (or a special-case like the graph `?entity=` branch at
  `visual_check.mjs:2167-2180`) plus fixtures for every endpoint it reads. Widths: 390×844 touch,
  768×1024 touch, 1440×900 (`visual_check.mjs:39-43`). Checks: no horizontal overflow, ≥44px targets at
  390, controls visible/unclipped, "app actually mounted".
- **Exact-pinned counters** (tests assert `==`, not `<=`): `hardcoded_palette_hex == 402`
  (`test_hardcoded_palette_hex.py`, last test; bot.tsx contributes **41** of the 402, TradingActivity 5,
  TradeChart 0, home.tsx 29); `ts_any == 1247` and `empty_ts_catch == 493` (`test_ts_code_only.py:149-180`).
  Deleting/rewriting bot.tsx changes these → the PR must lower the pins in `ci/counter_baseline.txt` AND the
  literals in those tests (lowering is a tightening; `counter_ratchet.sh` enforces non-increasing).
- `uncapped_surface == 0`: any canvas/WebGL must size via `surfacePixelRatio()` from `@/lib/deviceTier`
  (enforced by `client/src/lib/deviceTier.test.ts` "no client module sizes a canvas from raw
  devicePixelRatio"; bot.tsx already complies at `:331`).
- `orphaned_set_interval == 0`: any file calling `setInterval` must also call `clearInterval`
  (`scripts/program_status.sh:777-…`).
- `design_token_drift == 0` (`test_design_token_drift.py`): if a new token is DOCUMENTED in DESIGN.md it must
  exist with the same value in `index.css :root`.

**Reusable pieces in `client/src/render/`** (all framework-free TS):
- `frameCore.ts` — singleton `frameCore()` → `FrameLoop` with `register(fn, PRIORITY.RENDER, {label})`,
  `start/stop/dispose`, `stats()` (p95 frame time), `Spring`, `lerpTowards`; constants `FRAME/SPRING/PRIORITY`
  are "law" (changing = amendment). Hidden-tab pause built in.
- `perfMetrics.ts` — `setGauge/addGauge/bump/getGauge/snapshot` + `METRIC` names; `perfHud.ts` —
  `perfEnabled()` (`?perf=1`), `mountPerfHud(loop)`, `sparkline()`.
- `freshness.ts` — `freshnessLevel()` → fresh|stale|dead|unknown (defaults 15 min / 60 min),
  `formatDataAge(ms)`, `FreshnessTracker`, `reportChainFallthrough()` — directly useful for PREMIUM (c).
- `layerContract.ts`, `tileCore.ts`, `zoomInput.ts`, `mapBaseConfig.ts` — map/layer specific.
- shadcn primitives available in `client/src/components/ui/` incl. `alert-dialog` (no page uses it yet —
  the natural confirm for Kill/Close), `skeleton`, `tabs`, `sheet`, `drawer`, `tooltip`, `scroll-area`,
  `resizable`, `chart` (recharts 2.15). `lightweight-charts` 5.1 is already a dependency.

---

## 6. GAPS

### 6.1 Exposed by the API today, not shown by the page

- **Liveness alarm & loop health** (CLAUDE.md priority 1): `/api/health` `checks.bot.liveness{dark,
  marketHours, wallHours, detail}`, `checks.scanner.consecutiveFailures`, `checks.feeds`, `checks.process`
  crash counts, `checks.memory.pressure`; 24h/7d history via `/api/data/pipeline-health-dashboard`.
- **Kill-switch auto-resume**: `/api/bot/status.autoResume` (latchReason, outcome, classification, basis,
  reason, consecutiveOk/required, nextCheck, `numbers{equity, peak, drawdownPct, explainedLoss…}`,
  lastResume), plus `equityPeak`, `maxDrawdownPct`.
- **Honesty/slippage metrics** (the HONESTY METRIC's raw inputs): `/api/bot/performance.realisticPnlPct,
  avgSlippagePct, totalSlippageCost, slippageGapPct, totalFills, seededTrades, byStrategy`.
- **Per-strategy live vs seeded P&L**: `/api/monitoring/tier-pnl`.
- **Everything in the last scan** except `size_tier`: scan timestamp & counts (scanned/filtered/
  deep_analyzed), slots available (stock/options), top_10 with price/score/change, new_trades,
  position_actions, upgrade_candidates, tier_actions + tier_stats, floors (spy_floor/defensive_floor/
  third_leg), convexity overlay, options_management, intraday_shorts, **kill_status / tier_kill_status /
  kill_suppressed_trades**, halted + dd fields, `scan_phase_times`, `data_source_errors`.
- **Python-side risk state**: `/api/monitoring/overview.kill_switches` thresholds + consecutive losses +
  daily P&L; drawdown proximity; daemon health (`/api/daemon/health`); self-diagnostics
  (`/api/bot/diagnostics`), `/api/system/health-check`, `/api/bot/system-status` self-tests.
- **ML detail**: `/api/ml/status.last_status/last_error/last_train_time/last_features/contributes_to_cagr`;
  `/api/bot/ml-status.learned_weights/needs_retrain`.
- Account: `equity`, `lastEquity`, `status`, `tradingBlocked` (`/api/bot/account`). Positions: `rMultiple`,
  `phase`, `highestPnl`, `daysHeld`, server `marketValue` (only partially used, in charts).
- Real market clock (`/api/bot/market-status`), early closes (`/api/bot/calendar.earlyCloses`), closed-order
  history incl. rejects (`/api/bot/history`), CSV export, live audit push (`/api/bot/stream`).

### 6.2 Not exposed by any endpoint (a cockpit would need new server work)

- **Benchmark vs SPY** (GOAL 3's stated yardstick: rolling performance vs buy-and-hold SPY) — computed nowhere
  in these endpoints.
- **Live-vs-backtest divergence** (HONESTY METRIC) — no endpoint.
- **Counterfactual rule P&L** (RULE REVIEW's blocked-trade outcomes) — no endpoint.
- **Current regime** as a lightweight read — only buried in `/api/system/snapshot`
  (`config_files.macro.regime`, ~20s call).
- **Tier-2 scan health detail** for the owner (last success time, last failure reason) — only via
  token-gated `/api/diag/scanner`; owner browser gets only the failure count.
- **WebSocket stream state** (`streamConnected`, `bot.ts:6299`), bot-tracked resting orders (`openOrders`),
  `strategyWeights` (`bot.ts:843`) and in-memory `tradeResults` (`bot.ts:554`) — never serialized.
- **Kill reason/timestamp** as a first-class field (only indirectly via `autoResume.latchReason`).
- **As-of timestamps** on account/positions/status payloads (no `asOf` field) — the client must stamp
  receive time itself to show data age.
- A single consistent **drawdown** definition and the **actually enforced** risk limits (status returns
  literals).
- Per-trade realized P&L that handles options (×100) and short-first round trips (CSPs) correctly.
