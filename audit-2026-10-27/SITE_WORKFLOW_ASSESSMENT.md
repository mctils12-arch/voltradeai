# VolTradeAI site workflow and functionality assessment — 2026-10-29

Scope: the `/app` shell as deployed right now (live `server_version` 1.0.1232 == `main`), its five top-nav items, who can reach what, what is duplicated, what exists on the server with no screen, and a recommended target structure that carries the AI Engine cockpit's design language across the site. Nothing here is built yet; this is the decision document.

---

## 1. What the nav is today

| Nav item | Who sees it | What it opens | Files |
|---|---|---|---|
| **Trade** (dropdown) | everyone | *Analyze*: Options & Volatility, Smart Money, Structure, ETF Builder. *Tools*: Scanner, Watchlist, Research, News, Planner, Taxes | `pages/analyze.tsx` (2,602 lines), `InsightsView.tsx`, `ETFBuilderView.tsx`, `scanner.tsx`, `watchlist.tsx`, `research.tsx`, `news.tsx`, `planner.tsx`, `taxes.tsx` |
| **Data** (dropdown) | everyone | Live Map, Streams, Everything Graph, Developers & API, My API Keys, **Signals (disabled, "soon")** | `pages/datamap.tsx` (16,628 lines) + 50 `#/data/*` overlays |
| **Forecast** | owner only | weekly SHADOW direction/range calls for SPY, open positions, scan candidates, with calibration history. Added 2026-10-27. | `pages/forecast.tsx` (298) → `/api/forecast` |
| **Brain** | owner only | every input/rule/model/sleeve with live / shadow / off status, short- vs long-term evidence, best combinations. Added 2026-10-29. | `pages/brain.tsx` (317) → `/api/bot/brain` |
| **AI Engine** | owner only | the bot cockpit: 11 panels (Edge, Stages, Issues, Now, Watch, Model, Chart, Book, Orders, Today, Daily vs SPY), hold / pin / float / zoom / pop-out, FLIP springs on `frameCore`. | `client/src/bot/*` (35 files) |
| *(no nav entry)* **Dossier** | owner only | ticker page: Market → Sector → Stock → Options → Events → trade plan, plus the bot's own score and gates. Opened only from the AI Engine look-up box or `#/dossier/SYM`. Added 2026-10-29. | `pages/tickerDossier.tsx` (377) → `/api/ticker-dossier/:symbol` |

Outside the shell: `/`, `/login`, `/pricing`, `/about`, `/newsletter`, `/careers`, `/contact`, `/developers`, `/apikeys`, `/admin`. Mobile bottom bar shows Analyze, Research, News, Data and a "more" sheet.

So a visitor sees **2** top-level items; you see **5** plus a sixth page with no entry. Three of your five (Forecast, Brain, AI Engine) plus Dossier are four separate pages for one persona, each with its own layout, three of them added in the last 48 hours by the autonomous sessions.

---

## 2. Access matrix (verified against the live site, anonymous)

| Surface | Anonymous | Signed-in user | Owner | Note |
|---|---|---|---|---|
| Analyze, Smart Money, Structure, ETF Builder | yes (200) | yes | yes | `/api/analyze/SPY` 4.8 s, `/api/insights/SPY` 9.6 s, `/api/etf/SPY` 8.7 s: each request spawns Python |
| Research | yes (200) | yes | yes | `/api/research/SPY` **13.5 s** (`python3 -m alphadesk` per request) |
| Scanner, News | yes | yes | yes | sub-second, cached |
| Watchlist | yes (empty) | persisted | persisted | the only thing login buys a normal user, besides API keys |
| Data map, streams, graph, 50 overlays | yes | yes | yes | |
| Developers / API keys / `/api/v1/*` | docs yes; keys need login | yes | yes | 44 `/api/v1/data/*` routes are API-only by design |
| Signals | **disabled** | disabled | disabled | the ladder page `#/data/signals` and `/api/v1/signals/*` exist |
| Forecast, Brain, AI Engine, Dossier | 401 | "restricted to the account owner" | yes | |
| Admin (`/admin`) | 401 | no | yes | site report |
| Pricing / checkout | page yes | checkout | | billing code is frozen; the billing SQLite file lives in an ephemeral path (see audit REPORT.md) |

Two navigation dead-ends: a bare `https://voltradeai.com/data` renders the **landing page** (no redirect to `/app#/data`); and the Dossier has no nav entry at all.

---

## 3. Things that exist but have no screen

The server exposes 302 `/api` routes; the client calls about 200. The gap is almost entirely operator functionality you can only reach with curl:

**Bot controls with no button** — `/api/bot/run-now`, `/close`, `/trade`, `/circuit-breaker`, `/backtest`, `/export-trades`, `/history`, `/signals`, `/refresh-signals`, `/ml-status`, `/diagnostics`, `/backup`, `/system-status`, `/notifications`, `/regime`, `/stream` (the SSE audit feed; the cockpit polls instead), `/api/ml/toggle`, `/api/ml/retrain`.

**Monitoring with no screen** — `/api/monitoring/{overview,drawdown,tier-pnl,caches}`, `/api/shorts/dashboard`, `/api/positions`, `/api/trades/history`, `/api/market-snapshot`, `/api/system/{snapshot,health-check,verify-change}`, `/api/daemon/health`, every `/api/diag/*` probe the sessions rely on (`benchmark`, `heartbeats`, `ml_v3`, `regime`…), `/api/debug/*`.

**Data world with no screen** — `/api/data/query` (the W4 query engine), `/api/data/satellites`, `/api/data/swim/*`, `/api/data/archive/*` (offload status, the thing the volume-capacity wishlist item is about).

The cockpit itself uses only 17 endpoints: status, account, positions, audit, performance, market-status, calendar, insight, last-scan, benchmark, sleeves, scoreboard, lookup, lookup-ledger, bars, start/stop/kill.

---

## 4. Duplicates and overlaps

| # | Same job, several places | Where |
|---|---|---|
| 1 | **Three scanners.** Analyze's "Scan" button (`/api/scan`: a Node-side tier-1 / full-universe cache in `routes.ts`), the Scanner tab (`/api/market/scanner`: Alpaca momentum & unusual activity), and the bot's Python `scan_market` whose last run is the cockpit's Watch panel. | `routes.ts:6822`, `routes.ts:6936`, `bot_engine.py` |
| 2 | **Ticker analysis three ways.** Analyze (`analyze.py`: IV/HV, VRP, GEX regime, IV rank, options flow, insider activity, price targets), Research (`alphadesk`: verdict, conviction, composite score, tax flags), Dossier (market → sector → stock → options → events, calibrated ranges, the bot's score). | `analyze.tsx`, `research.tsx`, `tickerDossier.tsx` |
| 3 | **Insider activity twice.** Analyze's Insider Activity card (`intelligence.py` / Finnhub) vs `#/data/insider` (EDGAR Form 4 archive, `server/edgarForm4.ts`). Two pipelines, two UIs, different numbers possible. | |
| 4 | **Smart Money / Structure** (`insights.py`: float, short, flow) vs the Data overlays short-volume, short-interest, FTD, filings13f. | |
| 5 | **"Watch" means two things.** Trade → Watchlist (your tickers) vs AI Engine → Watch (the bot's candidates). | |
| 6 | **Three world maps.** Analyze's `DataWorldMap.tsx` (736 lines, a topojson canvas of "data source cities" with arc lines; decorative, draws no data), the landing-page MapLibre globe, and the real `/data` globe. The one in your screenshot is the decorative one. | |
| 7 | **Bot performance twice.** Cockpit Edge strip + Daily vs SPY vs `/api/monitoring/*` (no UI). | |
| 8 | **Tax logic twice.** Taxes tab (`/api/tax/estimate`) and Research's `--bracket/--state/--ltcg` flags. | |
| 9 | **Planning twice.** Planner (`/api/plan`, position & risk) vs the Dossier's trade plan (stop/target, simulated odds). | |
| 10 | **Operator pages ×4.** Forecast, Brain, Dossier and the cockpit each fetch their own data and lay out their own page; none reuses the cockpit's `Panel` kit. | |

---

## 5. Why the bot page feels modern and Analyze does not

The cockpit (`research/bot_cockpit_design_brief_v2.md`) has four properties the rest of the app lacks:

1. **Full-bleed grid with a state machine**: the centre column shows what the bot is doing now; panels are tiered by viewport width; motion runs on `frameCore` springs (Law I).
2. **One panel chrome**: title, status word, meta, data age, hold/pin/float/zoom/pop-out; honest empty states ("not recorded").
3. **Cache-first data**: 17 cheap endpoints polled on a schedule; nothing waits on Python.
4. **Everything carries age and provenance** (the PREMIUM EXPERIENCE STANDARD).

Analyze is the opposite: one 2,602-line page, a 736-line decorative canvas behind it, tab-local layout, and a 5–13 s Python subprocess per click. The old map you like is the decorative canvas; it can stay as the ambient backdrop, but the data has to come from caches.

---

## 6. Recommended target structure

Two personas, three nav items for everyone, one operator surface for you.

### 6.1 Nav

| Today | Proposed | Contents |
|---|---|---|
| Trade (10 leaves) | **Markets** | one **Ticker page** (the Dossier layout, for everyone, with the bot-only layers hidden for non-owners) + one **Screener** + Watchlist + ETF Builder |
| Data (6 leaves) | **Data** | unchanged, plus Signals switched on (the page and API exist) |
| Forecast, Brain, AI Engine, Dossier (4 pages) | **Engine** (owner) | the cockpit shell with Forecast, Brain and Dossier as zones/sub-tabs inside it, and the hidden controls exposed as panel actions |
| /pricing, /developers, /about … | unchanged | marketing + API routes |

### 6.2 The Ticker page (replaces Analyze, Research, Smart Money, Structure, Planner, Taxes)

Top-down, the way the Dossier already reads: **Market → Sector → Stock → Options & Volatility → Smart Money → Structure → Events → Plan**. Options & Volatility is the hero layer and keeps the IV/HV cone, VRP, GEX, IV rank and spreads from Analyze. The decorative globe stays as the backdrop. Research's verdict becomes one card. Planner and Taxes become the Plan layer (one calculator, one tax model). Owner-only layers: the bot's score and gates, the plan ledger. Every card gets the cockpit chrome: age, source, "not recorded".

Rule: the page reads **cached/archived JSON** (nightly dossier, the options chain archive, the EDGAR archive). A per-click Python run exists only behind an explicit "refresh" button, with its age shown.

### 6.3 The Screener (replaces Scanner, Analyze's Scan button, and the cockpit Watch's public half)

Two tabs: **Market** (today's `/api/market/scanner`) and **Engine** (the bot's last scan, owner-only, the same data the Watch panel shows). Watchlist tickers are a filter on both and a one-click way into the Ticker page.

### 6.4 The Engine shell (owner)

Keep the cockpit exactly as it is and make it the home for the other three pages: Forecast and Brain as zones (or a second tab row inside the shell), Dossier opening from the look-up box as it does now. Add the missing actions as panel buttons: run now, close position, circuit breaker, ML retrain/toggle, backtest, export trades, and surface `/api/monitoring/*`, `/api/bot/regime`, `/api/bot/diagnostics` and the `/api/diag/*` probes as read-only panels instead of curl.

### 6.5 Shared kit

Move `Panel`, `KpiStrip`, the status words, `fmtAge`, and the tier/grid helpers out of `client/src/bot/` into `client/src/components/` so Markets pages use the same pieces. This also retires the four age formatters and per-page `fmtPct/fmtMoney` copies the audit found.

### 6.6 What gets retired once the above ships

`analyze.tsx` (2,602), `research.tsx`, `planner.tsx`, `taxes.tsx`, `InsightsView.tsx` (folded), `DataWorldMap.tsx` (or kept as a pure backdrop component), the Node-side `/api/scan` cache in `routes.ts`, and the per-request Python spawns for analyze/insights/research once their data comes from archives.

---

## 7. Sequencing (each its own PR, visual harness at 390/768/1440)

| Step | PR | Risk |
|---|---|---|
| S1 | Extract the cockpit panel kit to `client/src/components/` (no visual change) | low |
| S2 | Ticker page for everyone: Dossier layout + Analyze's Options & Volatility as its hero layer + the globe backdrop; non-owner hides bot layers | medium: needs the nightly dossier cache for every requested symbol, not just the bot's |
| S3 | Screener: Scanner + Engine last-scan + Watchlist filter | low |
| S4 | Engine shell absorbs Forecast, Brain, Dossier; hidden controls become actions/panels | medium |
| S5 | Switch Signals on in the Data menu; `/data` redirect | low |
| S6 | Retire the replaced pages and endpoints | low once S2–S4 are live |

---

## 8. Decisions needed from you

1. Do visitors get the Ticker page (Dossier minus bot layers), or does the public keep today's Analyze while only you get the new page?
2. Keep ETF Builder, Planner and Taxes as standalone tools, or fold Planner/Taxes into the Plan layer as proposed?
3. Forecast and Brain: zones inside the Engine shell, or keep them as separate top-level pages?
4. Switch the Signals menu on now (the page exists) or leave it "soon" until a second signal passes gate 2?
5. Which hidden bot controls do you actually want as buttons (run now, close, circuit breaker, ML retrain/toggle, backtest, export)? Each one is a live-account action and gets a confirm step.

---

## 9. Note on Phase 4 of the cleanup

While this audit was running, another session filed a human-approved branch cleanup: `ops/branch_cleanup_plan_2026-10-28.tsv` + `scripts/branch_cleanup_apply.sh` (805 merged branches + `data-backup` deleted, 86 unique ones tagged `archive/<name>` then deleted, 25 skipped). The agent proxy refuses branch deletes and tag pushes, so **the branch half of Phase 4 has to be run by you from a local clone**: `bash scripts/branch_cleanup_apply.sh` (dry run) then `--apply`. The plan lists this audit's branch `ccr-5f668b84-lz6g69` for tag+delete; the script skips it because its tip has moved, but merge or keep it deliberately. The dead-code half of Phase 4 (REPORT.md §6 step 3) is still mine to do on your go.
