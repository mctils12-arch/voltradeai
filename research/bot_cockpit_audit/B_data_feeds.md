# B — Data feeds into the trading-bot decision path (ground truth)

Repo: `/home/user/voltradeai` at `8998328` (package.json `version` 1.0.1084), audited 2026-10-15.
Scope: every input that reaches a trading decision (entry, sizing, exit, floor/overlay, tier engine, ML score).
Out of scope: datacore /data map layers. None of them reach the bot (see §4.9).
Method: I read the code and traced each consumer back to its fetch. Every claim cites `file:line`. Where a
claim depends on live upstream behavior, I probed from the sandbox (not from Railway) and labeled it **[probe]**, or
labeled it **[verify]** when it could not be checked here. Live production readings captured by a sibling auditor
(digest in `research/bot_cockpit_audit/live_snapshot_2026-10-15.md`, ~2026-10-15T03:12Z) are labeled **[live]**.

Status legend used below:
- **LIVE**: wired into the decision path and does real work.
- **LIVE-DEAD**: wired in, but in practice always returns a constant or neutral value (see the reason given).
- **ORPHAN**: defined but never reached from the decision path.
- **OBSERVE**: runs on a bot schedule but only writes to the audit log or an archive. It never changes a decision.

---

## 0. Top findings (read this first)

1. **Most feed failures are invisible by construction.** Nearly every fetcher catches its own exception and
   returns a neutral-looking dict. So `_run_diag_fetch` (`bot_engine.py:756`) and the `.result(timeout=15)` capture
   (`bot_engine.py:875-901`) only ever see import errors and timeouts. Live `dataSourceErrors` is `{}` **[live]
   scanner.json**, even though, for example, the USPTO and EDGAR-insider sub-feeds cannot work (items 3 and 4).
2. **Negative caching defeats every cache-file health check.** `alt_data`, `social_data`, `finnhub_data`,
   `intelligence.get_insider_activity` and `macro_data` all write their cache file *even when the fetch failed*
   (fabricated neutral payload, or `{"error":…}` for Finnhub). `diagnostics.py` judges "API up" by **file
   existence/mtime** (`diagnostics.py:86-94`, `389-395`, `446-453`). A dead upstream therefore reads as healthy.
3. **`intelligence.get_insider_activity` EDGAR branch is dead.** It filters on `_source.form_type`
   (`intelligence.py:260-261`), but the live EFTS response has `form` and `root_forms` and no `form_type`
   **[probe: 200 OK, keys listed in §3.E2]**. Every hit is skipped, so EDGAR always contributes 0 buys and 0 sells.
4. **USPTO patent feed is dead.** `developer.uspto.gov/ibd-api/...` (`alt_data.py:442`) 301-redirects to an HTML
   page (`data.uspto.gov/home`) **[probe]**. `resp.json()` raises, the exception is swallowed, and `signal=0` is
   cached for 24h.
5. **Several deep_score/ML inputs read keys their producer never emits. These are constants at inference, not data:**
   - `intel["vxx_ratio"]` → always 1.0. This feeds both ML `vxx_ratio` and ml_score's regime-model selector, so the
     **bull regime model can never be selected**.
   - `intel["earnings_surprise"]` → 0.
   - `vol_metrics["ewma_vol"]` → 2.
   - `quick_result["high_52w"]` → `price_vs_52w_high` = 0.
   - `macro["market_breadth_ma50"]` → never set; falls to the SPY proxy.
   - `macro["spy_change_pct"]` → so `idiosyncratic_ret` = raw change_pct.
   - `edge["short_float_pct"]` and `edge["days_to_cover"]` → squeeze strategy always gets 0/0.
   - Markov regime is called with `_spy_rets = []`, so `markov_state` ≡ 1.
   Full table in §5.
6. **Analyze.py sentiment fabricates a *bullish* reading on failure.** When the Reddit JSON call is non-200
   (live: **403** [probe]), `reddit_buzz` defaults to 0.5 (`analyze.py:937-939`), which normalizes to the
   **maximum** reddit component of 100 (`analyze.py:971`). A total failure (StockTwits/Reddit/pytrends) yields
   `sentiment.score` = 60 ("Bullish"). That adds +10 to the squeeze score via `strategies/squeeze.py`.
7. **Macro fallbacks are silent and directional.** A failed VIX fetch gives VIX=20.0 (`macro_data.py:103,106`),
   which sizes every stock trade at 0.80x (`position_sizing.py:595-600`). A failed sector-ETF snapshot gives all
   sectors 0.0, so `market_regime` is set to `"risk_off"` (`macro_data.py:184-188`), which costs −5 on every deep
   score. A failed VXX/SPY fetch gives the NEUTRAL regime, so `MAX_POSITIONS` = 0 (`system_config.py:504`) and there
   are **zero new scanner stock entries**. `data_quality` can only ever be `good`/`degraded` ("poor" is
   unreachable; §3.D1). It is read by only one decision consumer (the floor-regime debounce).
8. **ML "fallback" is blended as if it were the model.** `ml_score` returns `model_type:"fallback_rules"` when the
   model file is >24h old or missing (`ml_model_v2.py:1590,1712`). deep_score only excludes the literal string
   `"fallback"` (`bot_engine.py:1745`), so the rules fallback is blended at the ML weight and labeled "ML model: …".
9. **The Alpaca feed tier is downgraded in production, and that state is never surfaced.** The code comments
   record SIP 403 and OPRA 403 (`server/bot.ts:3297-3300`, `alpaca_feed.py:166-176`). In that state:
   - snapshots and quotes are `delayed_sip` (15 min stale);
   - **all daily bars are `iex`** (`alpaca_feed.py:115-137`);
   - options are `indicative`.
   `alpaca_feed.feed_status()` and `options_feed_status()` exist but are **never called** (grep: zero call sites),
   despite the docstring saying "scan results carry data_feed".
10. **The trading universe silently drops every symbol ending in "W"** (`bot_engine.py:2413`, meant to drop
   warrants). That removes LOW, NOW, SNOW, CDW, DOW, EW, TROW, … from the 12,138-symbol scan, except when one
   re-enters as a top mover.
11. **The CONVEXITY overlay runs every scan even though config says disabled.**
    `BASE_CONFIG["CONVEXITY_OVERLAY"]["enabled"]=False` (`system_config.py:236-242`) is never read. Step 9b calls
    `_run_convexity_overlay` unconditionally (`bot_engine.py:3846-3851`). It fetches QQQ, option contracts and an
    `indicative` quote. On a quote failure it prices the put at a fabricated 0.7% of QQQ (`bot_engine.py:4641`) and
    POSTs an order with `side:"buy_to_open"` (`bot_engine.py:4653-4663`). **[verify]**: that side value is likely
    rejected. The sibling capture shows 0 QQQ option orders in the last 100.
12. **The stream-signal queue is dropped, and the WS stream state is unexposed.** The WS STREAM-SIGNAL queue is
    audited and then cleared and never passed to Python (`server/bot.ts:4381-4387`, `run_full_scan` args `{}`).
    `streamConnected` and last-bar time are not exposed by any API.
13. **No datacore signal reaches the bot.** No Python decision module or bot.ts trading path calls `/api/v1`,
    `/api/data`, or reads a datacore signal. CLAUDE.md's "bot is customer zero" is aspirational today (§4.9).

---

## 1. Who calls what, and when (scheduler → Python entry points)

All in `server/bot.ts` unless noted.

| Loop | Cadence / gate | Python entry point | Transport |
|---|---|---|---|
| Tier 1 `tier1Reflex()` (`:4191`) | every 45s (`:7601-7616`), market-open only (`/v2/clock` `is_open`, `:7610`); skipped while kill latched | `bot_engine.manage_positions()` via inline `python3 -c` (`:4213-4219`) — result used only to refresh `voltrade_stop_state.json`, its actions are NOT executed | subprocess (execPythonSerialized, 15s) |
| Tier 1 sync | `syncMonitoredPositions()` (`:6345`) every 60s (`:7227`) + Tier 1 step 3 | inline reader of `voltrade_stop_state.json` | subprocess |
| Tier 2 `tier2Intelligence()` (`:4335`) | adaptive `getTier2Interval()` (`:7618-7634`): 9:30–10:30 ET 60s; 10:30–12 3m; 12–14 7m; 14–16 2m; 16–20 15m; 4–9:30 5m; first run +10s (`:7734`). Gate `isAnyWindow = clock.is_open \|\| 4≤ET<20` (`:7679-7685`) → **no Tier 2 scans 20:00–04:00 ET**. Backoff 60→600s on failure (`:7709-7726`) | `bot_engine.scan_market` as daemon RPC `run_full_scan` (`:4406-4412`, 300s); `diagnostics.get_auto_fix_params` every 5th cycle (`:4357-4377`); `diagnostics.check_weekly_loss` (`:4625`); `intraday_shorts.close_open_shorts` after 15:50 ET (`:4645-4660`) | daemon (scan is daemon-only: `HEAVY_DAEMON_ONLY`, `:162-164`) |
| Tier 3 `tier3Strategic()` (`:5709`) | hourly (`:7737-7756`) + once 30s after boot (`:7768`) | ML retrain check → `ml_retrain_safe.py` if model >24h (`:5714-5739`); `manipulation_detect.scan_for_manipulation` (`:5802`); `diagnostics.run_diagnostics` (`:5835`); `cftc_cot.run_daily_update` (`:5879`); `sec_form4_bulk.run_update` (`:5900`); port-dwell capture (Node, datacore) | subprocess |
| WS stream `startStreaming()` (`:7232`) | persistent; reconnect 10s after close (`:7358`) | none (Node only) | `wss://stream.data.alpaca.markets/v2/iex` (`:7248`) |
| Overnight research `runOvernightResearch()` (`:6010`) | only reachable from inside tier2Intelligence when `!isMarketOpen && (ET≥20 \|\| ET<4)` (`:5208-5210`) — but Tier 2 itself never runs in that window (gate above) → **effectively unreachable** except a scan that starts before 20:00 and finishes after | `bot_engine.py full`, `analyze.quick_scan`, etc. | subprocess |
| On-demand | `/api/bot/refresh-signals` (`:3673`) | `analyze.py <T> --mode=scan` ×10 tickers (`generateSignals`, `:3596`) — UI signals list only | subprocess |

Python modules imported by the daemon stay resident (`voltrade_daemon.py:245-260`). Their in-memory caches
(yfinance fundamentals, most-actives, credit spread, vol surface, option chains, the options-scanner regime,
`position_sizing._earnings_cache`) persist across scans. Tier-1 subprocesses start cold every 45s.

---

## 2. The scan pipeline and the feed each phase consumes

`scan_market()` (`bot_engine.py:2461`) → `_scan_market_inner()` (`:2571`). The phase names below are the exact
`_timing_log` labels persisted to `voltrade_scan_timings.json`.

| Phase label | Feeds consumed (details in §3) |
|---|---|
| (pre) DD halt check | Alpaca `/v2/account` (`:2687`), `macro_data.get_macro_snapshot()` (`:2693`) |
| `universe_load` | `_get_full_universe()` assets (A2), screener movers (A3), most-actives fallback (A7) |
| `snapshot_fetch` | Alpaca `/v2/stocks/snapshots` 400-symbol batches, `data_feed()` (A4) |
| `quick_scan` | snapshot tuples only (no fetch) |
| `deep_score` | per candidate (cap 5, 3 if RSS ≥400MB, skip if ≥550MB; 35s total wall cap `:3049`): analyze.py subprocess (B), 5 parallel enrichers macro/intel/alt/social/finnhub (D–H), yfinance fundamentals subprocess (C), Alpaca 400d bars (A5), ML inputs (most-actives, TLT/HYG, VXX) |
| `step6a_trade_loop_and_covered_calls` | Alpaca account/positions (`:3176-3197`), macro snapshot again (`:3209`), per-candidate `/quotes/latest` spread check (A6), `instrument_selector`, `position_sizing.calculate_position` (K) |
| `step6_trade_loop_and_options` | `options_scanner.get_options_trades` (J1) |
| `step7_manage_positions` | positions, macro, `_get_atr` bars |
| `step8b_options_manager` | options quotes (`options_manager.py:198,392`) |
| `step9_third_leg` | account, positions, open orders, VXX bars, SVXY snapshot (L1) |
| `step9b_convexity_overlay` | account, positions, orders, QQQ snapshot, option contracts, indicative quote (L2) |
| `step10_spy_floor` / `step10b_defensive_floor` | macro → floor regime (L3), floor-ticker bars + basket snapshots (L4) |
| `tier_engine_*` | account, positions, `get_portfolio_margin_status()` account, csp_universe Layer 1/2 (J3), options chains |

**[live] timings.json:** `deep_score` 35.67s (at the 35s cap), `tier_engine_tier1` 57.04s with
`csp_layer2_prefetch {cache_hit:false, completed:106, total:150, budget_exceeded:true}`. Total 99.12s.

---

## 3. Feed catalog

### A. Alpaca market data and trading API

**A1. Feed resolver**: `alpaca_feed.data_feed()` (`alpaca_feed.py:72`), `bars_feed()` (`:113`), `options_feed()` (`:181`)
- Upstream probes:
  - `GET data.alpaca.markets/v2/stocks/snapshots?symbols=SPY&feed=sip` (`:56-68`).
  - `GET /v1beta1/options/snapshots/SPY?feed=opra` (`:152-162`).
- Auth: `ALPACA_KEY`, `ALPACA_SECRET`. Overrides `ALPACA_DATA_FEED` / `ALPACA_OPTIONS_FEED` force a value and skip probing.
- Cadence/cache: in-process state, re-probed at most every `PROBE_TTL_S=600` (`:40`). Every fresh subprocess re-probes.
- Behavior:
  - 403 → `delayed_sip` (stocks) or `indicative` (options).
  - 200 → upgrade back.
  - 0 / 5xx / 429 → keep the current feed.
  - `bars_feed()` maps `delayed_sip` → `iex` (`:136-137`).
- Recorded: one stderr line per switch only (`:99-103`). `feed_status()`/`options_feed_status()` (`:139`, `:226`)
  return `{feed, degraded, downgraded_since}` but **have zero callers**.
- Production state (from code comments, not observed): stock snapshots/quotes/trades = `delayed_sip`, bars = `iex`,
  options = `indicative`, WS stream = IEX. **[verify]** via a feed_status read once exposed.
- Consequence:
  - Every daily-bar consumer gets **IEX-only volume** (~2–3% of consolidated). Ratios such as `volume_ratio` and
    OBV direction survive. Absolute thresholds do not: ML training drops rows with `volumes[idx] < 300_000`
    (`ml_model_v2.py:386`), and the alt_data FTD turnover ratio is affected (§3.F5).
  - analyze.py's own bars call sends **no `feed` param** (`analyze.py:94-99`). It gets Alpaca's account default.
    **[verify]** whether that is iex.

**A2. Tradable universe**: `bot_engine._get_full_universe()` (`:2390`)
- Upstream: `GET {ALPACA_BASE_URL}/v2/assets?status=active&asset_class=us_equity` (`:2405-2408`), Alpaca trading API.
- Cache: `DATA_DIR/universe_cache.json`, 24h by file mtime (`:2396-2404`).
- Filter (`:2410-2416`): `tradable` and not containing "." and not ending in "W" and length ≤5. The "W" rule also
  drops real common stocks (LOW, NOW, SNOW, CDW, DOW, EW, TROW, …).
- Failure: logs a warning and returns `[]`. The caller then falls back to most-actives top 100 (`:2747-2754`). If
  that also fails, the scan returns `{"error":"Could not fetch market universe"}` (`:2754`).
- Also reused by csp_universe Layer 1 (`csp_universe.py:327`) and the options-scanner candidates (`options_scanner.py:1338`).
- Healthy value: `scanned` ≈ 12,138 **[live] TIER2 audit "Scanned 12138 stocks"**.

**A3. Top movers ("freshness boost")**: `GET /v1beta1/screener/stocks/movers?top=50` (`bot_engine.py:2729-2737`)
- Merged into the universe. Failure is silently skipped. Per scan.

**A4. Scan snapshots**: `_fetch_snap` (`bot_engine.py:2781-2797`), parser `_parse_snapshot_batch` (`:2525`)
- Request: `GET /v2/stocks/snapshots?symbols=<400>&feed=data_feed()`, sequential, timeout 15s, `alpaca_throttle`.
- Produces per-symbol `(c,o,v,pc)` from `dailyBar` and `prevDailyBar`. These are the quick-score fields `price`,
  `close`, `open`, `prev_close`, `volume`, `dollar_volume`, `change_pct`, `quick_score`, `above_vwap` (= c>o, not a
  VWAP), `range_pct`=0, `vwap_dist`=0 (`:2848-2867`).
- Failure: per batch it records `_snap_diag["detail"]` (HTTP status, non-dict body, or "0/N usable dailyBar").
  Only if *every* batch is empty does the scan return `{"error":"Could not fetch market data from Alpaca",
  "debug_detail":…}` (`:2872-2877`). Node surfaces that as `tier2LastFailureDetail` (`server/bot.ts:4440-4444`).
- Staleness: under `delayed_sip` the bars are 15 min old. Pre-market and early-open scans see the previous
  session's `dailyBar` (inference **[verify]**), yet Tier 2 runs every 60s from 9:30 to 10:30.

**A5. Daily bars (many consumers, all `bars_feed()`)**

| Consumer | file:line | Window | Cache | Failure |
|---|---|---|---|---|
| deep_score ADX/OBV/momentum/MA200/ATR | `bot_engine.py:983-1045` | 400d, limit 300 | none | `except: pass` → `adx_value=None`, `obv_signal=0`, `_deep_closes=[]` → `momentum_score=50`, `change_5d`=today's change. Not recorded |
| deep_score VXX fallback for vol_of_vol | `:1506-1517` | 20d | none | → `_vol_of_vol=2.0` |
| deep_score credit spread TLT/HYG | `:1615-1640` | 35d, limit 30 | 60s `_CREDIT_SPREAD_CACHE` | `except: pass` → 0.0 |
| `_get_atr` (manage_positions) | `:2341-2387` | 60d | none | None → `atr_pct=2.0` (`:2044`) |
| macro SPY 300d / VXX 45d | `macro_data.py:205-238,247-279` | — | 300s file | defaults (§D1) |
| options_scanner VXX/SPY regime | `options_scanner.py:204-275` | 45d / 80d | 60s in-memory | 1.0 / 1.0 |
| options_scanner IV rank | `options_scanner.py:558-565` | 400d | per scan via csp cache | None |
| floor trend (QQQ MA50/200) | `bot_engine.py:5022-5032`, defensive `:5427` | `limit=210`, **no `start`** | none | trend override skipped |
| third-leg VXX trend | `bot_engine.py:4284-4289` | `limit=6`, **no `start`** | none | `vxx_declining=False` |
| instrument_selector ETF volume | `instrument_selector.py:1255-1268` | `limit=1`, no start | none | fail-open `True` |
| ML training | `ml_model_v2.py:278-294` | fast_mode 730d, 10 symbols/call, `limit=10000`, no pagination | none | `last_error` string returned |

**[verify]**: Alpaca v2 bars without `start` default to the beginning of the current day. If so, the QQQ
death-cross override and the third-leg VXX-declining test receive ≤1 bar and silently never fire.

**A6. Spread check**: `GET /v2/stocks/{T}/quotes/latest?feed=data_feed()` (`bot_engine.py:3416-3436`)
- Rejects a candidate if `(ask−bid)/mid > 0.005` and logs `shadow_portfolio.update_last_decision(…,"rejected_other")`.
- Failure → **fail-open** (trade allowed). Under `delayed_sip` the quote is 15 min old.

**A7. Most-actives**: `GET /v1beta1/screener/stocks/most-actives?by=volume&top=100` (`bot_engine.py:1546-1552`)
- Used for the ML `cross_sec_rank` (60s cache).
- Reads `s.get("change")`. Per Alpaca's documented most-actives schema (`symbol`, `volume`, `trade_count`) there is
  no `change` field **[verify]**. If so, fewer than 10 changes are collected and the result is the constant
  **0.5**. Even if present, `change` would be dollars compared against `change_pct` (percent).
- Also used as the universe fallback (`:2750`), the ML training seed (`ml_model_v2.py:244`), and manipulation_detect (`:30`).

**A8. Alpaca news `/v1beta1/news`**: no throttle object (the global monkey-patch still gates it), no cache except social.

| Caller | file:line | Params | Output | Failure |
|---|---|---|---|---|
| `macro_data.get_news_sentiment` | `macro_data.py:402-470` | limit 10 | `sentiment_score` (−100..100), `sentiment_label`, `headline_count`, `top_headline`, `positive_count`, `negative_count` → deep_score `news_adjustment` (`bot_engine.py:1212-1225`) | returns all-zero neutral (`:424-426`). **Indistinguishable from "no news"** |
| `intelligence.classify_news` | `intelligence.py:120-128` | limit 15 | `events`, `score`, `trap_warning`, `trap_reason`, `headline_count`, … | `{"events":[],"score":0,"trap_warning":False,"headline_count":0}` |
| `intelligence.get_insider_activity` (2nd half) | `intelligence.py:285-299` | limit 5 | +buy_count/+sell_count from "insider…buy/sell" headlines | swallowed |
| `intelligence.check_earnings_pattern` | `intelligence.py:375-396` | limit 5 | `latest_earnings` | swallowed |
| `social_data.get_news_multi_source` | `social_data.py:331-343` | limit 10 | headlines | swallowed (1h cache) |

The docstring on `get_news_sentiment` says "Polygon news". The code uses Alpaca (`macro_data.py:404,419`).

**A9. Options snapshots (OPRA → indicative)**

| Caller | file:line | Feed | Cache |
|---|---|---|---|
| `options_scanner._fetch_options_chain` | `:283-360` | `options_feed()` | 180s in-memory per (ticker, dte window) |
| `options_scanner._fetch_iv_rank` ATM IV | `:592-622` | hardcoded `"indicative"` | (via csp per-scan cache) |
| `vol_surface._fetch_options_chain` | `vol_surface.py:187-247` | `options_feed()` | surface 3600s in-memory |
| `options_execution._fetch_option_chain` | `options_execution.py:571-600` | `options_feed()` | — |
| `options_manager` quotes | `options_manager.py:198,392` | `options_feed()` | — |
| convexity overlay quote | `bot_engine.py:4631-4633` | hardcoded `"indicative"` | — |

The `options_feed()` docstring (`alpaca_feed.py:177-180`) says options_scanner, options_manager and vol_surface
"still hardcode feed=opra". **Stale**: all three now call `options_feed()`.

**A10. Real-time WS stream (Node)**: `startStreaming()` (`server/bot.ts:7232-7366`)
- Subscription: IEX 1-min bars for `STREAM_TICKERS` (43 symbols, `:6287-6293`) plus held positions
  (`positionSubscribedTickers`).
- Consumers:
  - `checkPositionOnTick(ticker, close)` (`:6720`, invoked `:7294`) → real-time stop, take-profit and trailing exits.
  - The STREAM-SIGNAL detector (vol ≥2.5x, |bar| ≥0.8%, |mom5| ≥0.3%) → `state.streamSignalQueue` (`:7313-7334`).
    The queue is only audited and cleared in tier2 (`:4381-4387`) and **never sent to Python**.
- Recorded: audit `STREAM` ("Real-time feed live — N tickers"), `STREAM-ERROR`, `STREAM-DISCONNECT` (`:7268-7356`).
  **[live]** "Real-time feed live — 43 tickers" ×3 after restarts, one disconnect.
- Not recorded: last-bar time, bars per minute, per-ticker last tick. `streamConnected` (`:6299`) is not in any API response.
- IEX-only prints: thin names get sparse ticks, so their stop checks run rarely. **[inference]**

**A11. Trading API (Node + Python)**

| Read | Where | Use | Failure |
|---|---|---|---|
| `/v2/account` | Node `alpaca()` (`server/bot.ts:324-345`, 15s abort): tier2 health (`:4342`), daily-loss (`:4562-4620`, guarded by `evaluateDailyPnl` → `EQUITY-READ-INVALID` audit), drawdown kill (`:3922-3934`, `evaluateDrawdown`); Python `get_alpaca_account` (`bot_engine.py:504`) for DD halt + sizing (`:3176-3182` falls back to `portfolio_value=100000`, `cash=100000` **fabricated**); third leg / convexity `equity` default 100_000 (`:4256`, `:4455`) | sizing, halts | Node: guarded; Python: fabricated 100k |
| `/v2/positions` | Node sync (`:6347`), Python `get_alpaca_positions` | exits, slots, sector checks, POS-KILL, TIME-EXIT | Python scan: `[]` on error (`:3196-3199`) |
| `/v2/clock` | Tier1/2 gating (`:7610`, `:7679`) | schedule | throws → tier error |
| `/v2/assets/{T}` | `position_sizing.check_halt_status` (`:783-808`), daemon `check_halt` (`server/bot.ts:5318`) | exchange-halt gate | **fail-open** `{"halted":False,"status":"unknown"}` |
| options buying power | `acct.options_buying_power` → fallback `acct.cash` (`server/bot.ts:4585-4586`) | CSP budget | — |
| `multiplier` | `tiered_strategy.get_portfolio_margin_status` (`:991-1019`) | PM gate (≥4) | `False` |
| morning gap check | `snapshot?feed=delayed_sip` hardcoded (`server/bot.ts:4050-4072`) | skip if overnight gap >5% | swallowed (order proceeds). It reads `dailyBar.c` first. At the open under delayed_sip that is probably **yesterday's close** (same as `trade.price`), so the guard ≈ never fires **[verify]** |

### B. `analyze.py` subprocess: `get_stock_details()` (`bot_engine.py:467-501`)

- Invocation: `python3 analyze.py <T> --mode=scan`, 30s timeout. **`--mode=scan` is ignored**: `__main__` only reads
  `argv[1]` (`analyze.py:2643-2662`), so it runs the full `analyze_ticker()` (`:1930`) for every deep candidate.
- Upstreams inside `analyze_ticker`:
  - B1 price history: `_fetch_alpaca_bars(T, 365)` (`:80-122`, no `feed` param, `adjustment=split`, end = today
    00:00Z). Falls back to yfinance `history(1y)` with an 8s timeout (`:1941-1957`). The source is recorded in
    `result["data_source"]` = `"alpaca"|"yfinance"` (`:2421`). **deep_score drops this field.**
  - B2 fundamentals: yfinance `ticker.info` (15s timeout, skipped for `_KNOWN_ETFS`) (`:1965-1973`) → `valuation`,
    `fundamentals` (incl. `short_pct_float`, `earnings_date`).
  - B3 options: yfinance `ticker.options` plus up to 8 `option_chain(exp)` and one extra chain for GEX
    (`:1990-2097`) → `atm_iv`, `vol_surface`, `skew`, `vrp`.
    - **No expirations → returns `{"error":…,"no_options":True}` (`:1998-1999`).** deep_score then returns
      `quick_result` unchanged (`bot_engine.py:793-795`) and records `_diag["stock_details"]="analyze.py error: …
      has no options market"` (a normal condition logged as an error).
    - Quick score maxes at 45+10+5=60, below MIN_SCORE 63, so **non-optionable tickers can never enter via the scanner**.
  - B4 realized vol: `rv10/20/30 = realized_vol(...) or 20.0` (`:1978-1980`), a fabricated 20%. `vrp` = `atm_iv − rv20`,
    or `None` if no plausible IV (`compute_vrp`, `:559-573`). deep_score maps None→0.
  - B5 `get_sentiment` (`:878-1002`):
    - StockTwits `api.stocktwits.com/api/2/streams/symbol/{T}.json` (8s) — **[probe] 200**.
    - Reddit `www.reddit.com/search.json` (8s) — **[probe] 403**.
    - pytrends 30-day.
    - All fail → `score` = 0.4·50 + 0.3·100 + 0.3·33.3 = **60** (`reddit_buzz` default 0.5 → component 100).
    - Output keys: `score`, `signal`, `bull_pct`, `reddit_buzz`, `trend_spike`, `contrarian_flag`.
  - B6 `get_earnings_intelligence` (yfinance earnings history/calendar), `compute_insider_net` (yfinance insider
    transactions), relative strength vs SPY (Alpaca 30d → yfinance fallback).
- deep_score consumes only: `vrp`, `recommendation`, `sentiment.score`, `sentiment.reddit_buzz`,
  `sentiment.contrarian_flag`, `edge_factors` (`squeeze_score`, `relative_strength`), `vol_metrics` (`rsi_14`,
  `volume_ratio_5d`, `ewma_rv`, `garch_rv`, `rv20`), and `valuation.sector` (→ `voltrade_sector_cache.json`, `:1814-1815`).
- Failure capture (`get_stock_details`):
  - error payload → `_diag["stock_details"]="analyze.py error: …"`
  - empty stdout → `"analyze.py empty stdout (exit=…, stderr=…)"`
  - exception or timeout → `"TimeoutExpired: …"`
  - Inner yfinance/StockTwits/Reddit failures are **not** captured; they become defaults.

### C. yfinance fundamentals subprocess in deep_score (`bot_engine.py:905-962`)

- Runs `python3 -c` with yfinance `info` + `recommendations` and a 5s hard kill (`:950-953`).
- Cache: in-memory `_yf_cache` for 300s, max 50 entries (`:156-159`), written only on success.
- Lands in `fundamental_adjustment` (`:1371-1415`): `recommendation`, `target_mean`, `current_price`,
  `forward_pe`, `ev_ebitda`, `num_analysts`.
- Failure: `except (TimeoutExpired, Exception): pass` → `{}` → adjustment 0. **Not recorded.**

### D. `macro_data.get_macro_snapshot()` (`macro_data.py:76-399`)

**D1. Snapshot**
- Callers (per scan): DD-halt regime (`bot_engine.py:2693`), deep_score `_fetch_macro` (`:806`), the slots/regime
  block (`:3208-3213`), `manage_positions` (`:1999`), stress_index (×4 components), instrument_selector (`:1335`),
  `position_sizing` stop scalar (`:1071`), `/api/system/snapshot`.
- Cache: `/tmp/voltrade_macro_cache.json` (**ephemeral**, not DATA_DIR), TTL 300s (`:22-23`). A hit only requires
  `"vix" in cached` (`:89`), which is always true. **It always saves, even when every field is a default** (`:398`).
- Upstreams, failure values, and where each value lands:

| Field(s) | Upstream | Failure value (silent) | Lands in |
|---|---|---|---|
| `vix`, `vix_change`, `vix_regime` | yfinance `^VIX` 5d (`:95-106`) | `vix=20.0`, `vix_change=0` → `vix_regime="high"` (20 ≥ 20) | deep_score VIX adj (`bot_engine.py:1181-1189`, only extreme/low act); `position_sizing._regime_scalar` (20.0 → **0.80x**); stress `spy_realized_vol` (→30) |
| `treasury_10y`, `yield_trend` | yfinance `^TNX` (`:118-133`) | 4.25 / "flat" | not used by decisions (display) |
| `dollar_index` | yfinance `DX-Y.NYB` (`:136-145`) | 104.0 | not used by decisions |
| `sector_momentum{Technology,Financials,Energy,Healthcare,Industrials,Communications,Consumer Discretionary,Consumer Staples,Utilities,Real Estate,Materials}` (% daily) | Alpaca snapshots of 11 SPDR ETFs, `data_feed()` (`:148-169`) | every sector 0.0 (exception, or a 403 body parses to c=0) | deep_score sector tailwind ±8 via `SECTOR_MAP` (`bot_engine.py:1192-1201`; only ~50 hard-mapped tickers, others "Other"→0); stress `sector_rotation` |
| `strongest_sector`, `weakest_sector`, `market_regime` (`risk_on`/`risk_off`/`neutral`) | derived (`:174-190`) | all-zero sectors → green 0/11 < 0.3 → **`risk_off`** | deep_score −5 "Market regime: risk-off" (`:1204-1209`) |
| `spy_below_200_days`, `spy_vs_ma200`, `spy_above_200d` | Alpaca SPY 300d bars (`:205-238`) | 0 / 1.0 / True | regime classifier (`regime_util.classify_regime_5level`), stress `breadth` |
| `vxx_ratio`, `vxx_latest`, `vxx_avg30`, `vxx_closes` | Alpaca VXX 45d bars (`:247-279`) | `vxx_ratio=1.0`, `vxx_closes=[]`/absent | regime (→NEUTRAL), slots (`MAX_POSITIONS` 0 in NEUTRAL), cooldowns, floors, third leg, ML `vol_of_vol`, `put_call_ratio` (`vxx_latest/20`, default 25→1.25), instrument_selector market IVR |
| `spy_vs_ma50`, `spy_ma50`, `spy_close`, `spy_closes_60d` | reuse of the SPY 300d bars (`:281-320`) | 1.0 / `[]` | regime, ML `spy_vs_ma50`, market_breadth proxy, stress |
| `regime` | `system_config.get_market_regime` (→`regime_util`) (`:325-346`) | defaults → `"NEUTRAL"` | instrument_selector, position_sizing `STOP_SCALAR_<regime>` |
| `data_quality`, `data_quality_score`, `data_quality_failures` | derived (`:351-396`) | see below | `_floor_macro_vxx_is_fallback` only (`bot_engine.py:4785-4800`); `/api/system/snapshot` `ml.macro.data_quality` |

- **`data_quality` is written twice; the second write wins** (`:360-363` then `:390-394`). `data_quality_reason`
  from the first block can survive stale.
- Score: VIX always passes (default 20 > 0), spy_ma always passes (defaults set), sectors always pass (11 keys,
  zeros included). The only reachable states are 100 → `"good"` and 70 → `"degraded"` (VXX failure).
  `data_quality_failures` ⊆ `["vxx_ratio"]`. **"poor" is unreachable.**
- Healthy ranges: `vix` 12–35; `vxx_ratio` 0.8–1.4 (BULL ≤0.95, CAUTION ≥1.05, BEAR ≥1.15, PANIC ≥1.40 per
  `regime_util.py:43-50`); `spy_vs_ma50` 0.94–1.06; sector daily % about ±3; `spy_below_200_days` 0 in normal tape.

**D2. Keys consumers read that `get_macro_snapshot` never sets**
- `market_breadth_ma50` (`bot_engine.py:1658`)
- `spy_returns_5d` (`:1455`, unused)
- `spy_change_pct` (`:1494`)
- `qqq_ma50`, `qqq_ma200`, `qqq_price` (`:5015-5017`, `:5422-5424`) → forces the floor's own QQQ bars fetch (A5).

### E. `intelligence.get_full_intelligence()` (`intelligence.py:527-563`), via deep_score `_fetch_intel` (`bot_engine.py:812-817`)

Returns `intel_score`, `news`, `insider`, `earnings_pattern`, `trap_warning`, `trap_reason`, `summary`. **It does not
return `vxx_ratio` or `earnings_surprise`**, though deep_score reads both (§5).

- **E1 `classify_news`** (Alpaca news, A8)
  - No cache read. Writes `EVENT_MEMORY_PATH` (`DATA_DIR/voltrade_event_memory.json`) only when ≥1 article was
    fetched (`:206`). That makes its mtime a real liveness signal for Alpaca news.
  - Lands in `intel_adjustment = clamp(intel_score*0.3, ±15)`; a trap caps at −10 (`bot_engine.py:1263-1275`).
- **E2 `get_insider_activity`** (`:224-327`)
  - EDGAR FTS `efts.sec.gov/LATEST/search-index?q="T"&forms=4&dateRange=custom&startdt=<30d>` with
    `User-Agent: VolTradeAI research@voltradeai.com`.
  - **Dead filter**: it requires `_source.form_type`. **[probe] 2026-10-15**: HTTP 200, 1361 hits for AAPL. `_source`
    keys are `adsh, biz_locations, biz_states, ciks, display_names, file_date, file_description, file_num,
    file_type, film_num, form, inc_states, items, period_ending, root_forms, schema_version, sequence, sics, xsl`.
    There is no `form_type`, so every hit is `continue`d.
  - Even if fixed, the buy/sell words are searched in `display_names` + `file_description` ("FORM 4 SUBMISSION"),
    which carry no transaction direction.
  - So only the Alpaca-news headline heuristic moves `net_direction`.
  - Cache: `INSIDER_CACHE_PATH` (`DATA_DIR/voltrade_insider_cache.json`), key `T_YYYYMMDDHH` (hourly), written
    unconditionally (`:316-325`).
  - Output: `net_direction` (strong_buying…strong_selling), `buy_count`, `sell_count`, `insider_signal` (−100..100),
    `recent_transactions`.
- **E3 `check_earnings_pattern`**
  - Reads `EARNINGS_MEMORY_PATH`, but `record_earnings_outcome()` has **no caller** (grep). So `sell_the_news_risk`
    is always False and `pattern` is always "unknown".
- Failure: every sub-fetch catches internally. `_diag["intel"]` is only set on an import error or the 15s timeout.

### F. `alt_data.get_alt_data_score()` (`alt_data.py:519-558`), via deep_score `_fetch_alt` (`bot_engine.py:819-824`)

Sub-feeds are called **sequentially** inside the 15s budget. All caches live in `/tmp/voltrade_alt_cache/<key>.json`
(mtime TTL). **Every sub-feed writes its cache on failure.** Output keys: `alt_score` (clamped ±20 in deep_score,
`bot_engine.py:1311-1313`), `wiki`, `short_interest`, `congressional`, `patent`, `ftd`, `fred_macro`,
`geopolitical`, `fred_adjustment`.

| # | Sub-feed | Upstream | TTL / key | Failure → | Status |
|---|---|---|---|---|---|
| F1 | `get_wiki_attention` (`:62-119`) | Wikimedia pageviews REST (UA set) **[probe] 200** | 4h `wiki_<T>` | `signal 0`, `spike_ratio 1.0`. Only the 20 tickers in `TICKER_TO_WIKI` are ever fetched; all others are cached as no-data immediately | LIVE (20 tickers only) |
| F2 | `get_fred_macro` (`:136-192`) | `fred.stlouisfed.org/graph/fredgraph.csv?id=…` ×7 concurrent (UNRATE, UMCSENT, ICSA, BAMLC0A0CM, MICH, T10Y2Y, FEDFUNDS) **[probe] 200** | 6h `fred_macro_expanded` (an empty `{}` is not served from cache because it is falsy, but the file exists → diag "fresh") | missing keys → no derived flags | LIVE; also feeds stress_index |
| F3 | `get_geopolitical_risk` (`:207-280`) | GDELT doc API ×5 queries, **3 concurrent** | 2h `gdelt_risk` | `[]` per query → `risk_level "low"`, `risk_score 0`. **[probe] GDELT returns 429 "Please limit requests to one every 5 seconds"** in plain text, so `.json()` raises. Concurrent requests virtually guarantee rate-limit hits and a systematic bias toward "low" | LIVE-DEGRADED (likely) |
| F4 | `get_short_interest` (`:285-359`) | Alpaca 30d bars (`bars_feed`) (a volume-pattern proxy, not real SI) | 4h `short_<T>` | `short_pressure "unknown"`, `signal 0` | LIVE (proxy) |
| F5 | `get_congressional_signal` (`:364-410`) | EDGAR FTS ×2 using `hits.total.value` counts (this part works with the real schema) | 12h `congress_<T>` | `signal 0` | LIVE (misnamed: Form 4 filing-count spike, not congress) |
| F6 | `get_patent_activity` (`:425-459`) | `developer.uspto.gov/ibd-api/v1/...` **[probe] 301 → data.uspto.gov/home (HTML)** | 24h `patent_<T>` | `signal 0` | **LIVE-DEAD** (API retired) |
| F7 | `get_ftd_signal` (`:464-512`) | Alpaca 15d bars (`bars[:5]` = the **oldest** five) + Polygon `/v3/reference/tickers` (`POLYGON_KEY`) | 6h `ftd_<T>` | `signal 0` (no key → 401 → no shares outstanding). With IEX volume the turnover ratio understates ~30–50x | likely LIVE-DEAD **[verify POLYGON_KEY]** |

FRED-derived flags: `yield_curve_inverted` (T10Y2Y<0) −5, `recession_signal`, `labor_market_weak` (ICSA>300k) −3,
`credit_stress` (BAMLC0A0CM>4) −8 (`:181-189`, `:543-550`). Healthy: T10Y2Y −1…+1.5; BAMLC0A0CM ~0.8–2.0;
ICSA ~200–260k.

### G. `social_data.get_social_intelligence()` (`social_data.py:415-466`), via deep_score `_fetch_social`

- **G1 Reddit RSS** (`:79-210`)
  - Source: `r/{wallstreetbets,stocks,investing,options,StockMarket}/search.rss`.
  - Non-200 is silently skipped. **[probe] 200 from the sandbox**; Railway IPs are untested **[verify]**.
  - Cache: 2h `reddit_<T>`, written on total failure too.
  - Output: `total_mentions`, `bullish_pct`, `bearish_pct`, `sentiment_score`, `buzz_level`, `top_posts`,
    `wsb_mentions`, `signal`.
- **G2 Google Trends**: **disabled.** It is a hardcoded zero dict (`:428-429`). `get_google_trends` (`:215-293`) is
  ORPHAN, so `confidence:"high"` is unreachable (it needs 3 sources).
- **G3 `get_news_multi_source`** (`:303-410`)
  - Sources: Alpaca news plus 3 RSS feeds. **[probe]** reutersagency 301, marketwatch 301, cnbc 200.
  - Cache: 1h `news_multi_<T>`.
- Output: `combined_signal` (±15), `confidence` (`high`/`medium`/`low`), `reddit`, `google_trends`, `news_multi`,
  `sources_active` → deep_score `social_adjustment` (halved if confidence low) (`bot_engine.py:1342-1367`).

### H. Finnhub (`FINNHUB_KEY`)

- **H1** `finnhub_data.get_insider_sentiment` (MSPR, 6h `fh_insider_<T>`) and `get_recommendation_trends`
  (12h `fh_reco_<T>`), via deep_score `_fetch_finnhub` (`bot_engine.py:833-847`)
  - Output: `insider_mspr`, `insider_signal`, `analyst_consensus`, `analyst_buy_count`.
  - Lands: MSPR >30 → +5, <−30 → −5; consensus buy → +3. This is **inside the `if intel:` block**
    (`bot_engine.py:1283-1294`), so it is ignored whenever intel is empty.
  - Failure: `_finnhub_get` returns `{"error":…}` with no raise (`finnhub_data.py:105-132`). It is merged with the
    defaults and **cached 6–12h** (`:170-171`, `:430-431`). deep_score sees mspr 0 and consensus "hold"; nothing reaches `_diag`.
  - 55/min token bucket (`:96`).
- **H2** `options_scanner._fetch_earnings_calendar_with_status` (`options_scanner.py:643-674`),
  `finnhub.io/api/v1/calendar/earnings`
  - Used by `scan_options` (21d) and csp_universe Layer 2 (30d).
  - **No status check**: a 401 `{"error":…}` body gives `.get("earningsCalendar",[])`=[], which returns
    **`({}, True)`**. That is "lookup OK, nobody reporting", exactly the conflation the `_with_status` variant was
    created to prevent.
- **H3** ML training `_fetch_earnings_surprises` (`ml_model_v2.py:297-310`): top 30 tickers, swallowed → `{}`.
- Also `/api/debug/finnhub-insider/:ticker` (`server/routes.ts:6461`) for manual raw inspection.

### I. Derived / no-fetch regime inputs

- **I1** `markov_regime.get_regime()` (`markov_regime.py:225-230`)
  - deep_score passes `_spy_rets = []` (`bot_engine.py:1454-1458`). `len<3` returns state 1 with fixed probs
    (`:138-139`).
  - `vxx_ratio` comes from `intel` (always 1.0).
  - Therefore `markov_state`≡1, `regime_label` depends only on `spy_vs_ma50`, and `regime_score` ∈ [43.5, 68.5].
  - Also stored as `quick_result["regime_label"]` and passed to shadow logging.
- **I2** `regime_util` (`regime_util.py:57-122`): pure thresholds. `system_config.get_market_regime` delegates to it
  (`system_config.py:404-411`).
- **I3** `options_scanner.scan_options` uses its **own inline regime** (`options_scanner.py:1513-1522`: PANIC needs
  vxx≥1.30 **and** spy<0.94; BULL vxx≤0.90), fed by its own VXX/SPY fetch. This disagrees with `regime_util`
  (PANIC ≥1.40; BULL ≤0.95 ∧ spy≥0.98 ∧ above-200d). One scan can therefore carry two different regimes.
- **I4** Tier engine `TierContext.regime` = `_regime_ctx…` only `if '_regime_ctx' in locals()`
  (`bot_engine.py:3920`). `_regime_ctx` is a deep_score local and never exists in `_scan_market_inner`, so the
  tier engine always gets **"NEUTRAL"**. vxx and spy are passed separately.
- **I5** `stress_index.compute_stress_index()` (`stress_index.py:154-219`), **LIVE** via
  `position_sizing.calculate_position` (`position_sizing.py:1029-1047`)
  - Cache: `DATA_DIR/voltrade_stress_cache.json`, 600s.
  - Components: `yield_curve`/`credit_spread` (FRED via alt_data, None→50), `vix_term_structure` (macro vxx),
    `spy_realized_vol` (macro vix), `breadth` (macro spy_vs_ma50/ma200), `sector_rotation` (macro sectors). Each has
    `except→50`.
  - If **all upstreams fail**: 50·.15+50·.25+35·.20+30·.15+25·.15+35·.10 = **38.75 → "elevated" → 0.75x size**,
    a fabricated de-risk.
  - Output keys: `index`, `level`, `action_suggestion`, `components{…}`, `weights`, `timestamp`, `_cached_at`.
  - `position_sizing` records `s_stress`, `_stress_level`, and `"unavailable"` on exception.

### J. Options-side feeds

- **J1** `options_scanner.get_options_trades` → `scan_options` (`options_scanner.py:1467`)
  - Returns early outside 9:30–16:00 ET (`:1487-1496`).
  - Feeds: own VXX/SPY regime (A5), SPY `vol_surface.get_vrp`/`get_surface_score`, Finnhub earnings calendar (H2),
    `_get_options_candidates()`.
  - `_get_options_candidates()` is a **third full-universe snapshot pass**: 50-symbol batches, 4 workers
    (`:1361-1377`), 60s candidate cache.
  - Per-ticker work: OPRA chains (A9), `_fetch_iv_rank` (A5+A9), `_fetch_price` snapshots (`:688-704`).
- **J2** `vol_surface` (`vol_surface.py`)
  - Fetches: chain (`:187-247`, 4s/page × 10 pages), spot (`/v2/stocks/trades/latest` then bars fallback,
    `:250-292`, **0.0** on total failure), historical bars (`:295-335`).
  - In-memory caches `_surface_cache` (3600s) and `_sabr_cache` (3600s).
  - Scan-scoped cost stats feed the daemon health (§7).
- **J3** `csp_universe` (via `tiered_strategy._get_t1_universe`, `tiered_strategy.py:148-180`)
  - Layer 1 (`csp_universe.py:198-320`): a **second full-universe snapshot pass** (500-symbol batches, 2 workers,
    `:323-367`) plus account equity (`:163-196`, 15-min in-memory cache).
    - Cache `DATA_DIR/voltrade_csp_universe_cache.json` 900s.
    - Empty snap → anchors only, at price 0.
  - Layer 2 (`:612-810`): Finnhub calendar (H2), IV rank + vol surface prefetch for the top 150 (45s budget), then
    baselines for the rest (`iv_rank_s=40`, `vrp_s=40`, `skew_s=50`).
    - Cache `DATA_DIR/voltrade_csp_scores_cache.json` 900s. `_load_layer2_cache` ignores the TTL (`:825`).
  - Under 17 tickers → `T1_TICKERS_FALLBACK` (`tiered_strategy.py:142-146`).
- **J4** `options_execution._fetch_option_chain` (`options_execution.py:571-600+`): options_feed chain; failures
  surface as `T2-FAIL` audits (KNOWN BROKEN #25 lineage).

### K. Sizing feeds: `position_sizing.calculate_position`

- **Earnings date**: yfinance `calendar` (`position_sizing.py:637-690`).
  - In-memory cache 12h, success only.
  - **Fail-closed**: lookup failure → 0.85x (`:720-723`). Also used by `options_execution._check_earnings_guard`
    (`options_execution.py:115-127`, which fails open to True only on import error).
- **VIX regime scalar**: macro `vix` / `vix_regime` (`:968-971`, `575-607`).
- **Stress multiplier**: §I5.
- **Stop scalar**: macro `regime` (`:1068-1075`).
- **Liquidity**: snapshot `volume`.
- **Heat**: positions.

### L. Floors and overlays

- **L1 Third leg** `_run_third_leg` (`bot_engine.py:4218`)
  - Inputs: macro regime inputs, `/v2/account` (equity default 100k, `:4256`), `/v2/positions`, open orders, VXX bars
    (no start, A5), SVXY snapshot `latestTrade.p` (delayed_sip).
- **L2 Convexity overlay** `_run_convexity_overlay` (`bot_engine.py:4414-4716`)
  - **Not gated by `CONVEXITY_OVERLAY.enabled=False`**.
  - Inputs: macro, account, positions, open orders, QQQ snapshot, option contract discovery, `indicative` quote
    (fabricated 0.7%·QQQ mid on failure, `:4639-4641`).
- **L3 Floor regime** `_resolve_floor_regime` (`bot_engine.py:4825-4945`)
  - The **only consumer that detects a macro fallback**: `_floor_macro_vxx_is_fallback` (`:4785-4800`).
  - Debounce keyed on the macro snapshot `timestamp` (`:4803-4822`); `FLOOR_REGIME_CONFIRM_CYCLES` (default 3).
  - Returns `regime`, `raw_regime`, `confirmed_by` (`bootstrap`/`unchanged`/`escalation`/`held_vxx_fallback`/
    `pending_repeat_observation`/`pending`/`confirmed`), `pending`, `pending_count`, `vxx_fallback`.
  - State persisted in `DATA_DIR/voltrade_floor_state.json`. Exposed on scan result `spy_floor.raw_regime` /
    `spy_floor.floor_regime_confirmed_by` (`:4977-4978`) and on `/api/system/snapshot` `floor_state`.
- **L4 SPY/defensive floor trend override**: QQQ bars `limit=210` without `start` (`:5022-5032`, `:5427-5431`)
  **[verify]**; basket snapshots (`:5095`, `:5132`, `:5278`).

### M. ML feeds

- **Inference**: `ml_score` (`ml_model_v2.py:1573-1686`)
  - Reads the model file `DATA_DIR/voltrade_ml_v2.pkl`, mtime-cached. Age >`ML_MAX_AGE_HOURS` (24,
    `system_config.py:285`) → `_rule_based_fallback` → `model_type:"fallback_rules"`.
  - Regime-model selection uses the `vxx_ratio` feature (constant 1.0) and `spy_vs_ma50` → only the `neutral`/`bear`
    models are reachable.
  - Blend weights come from `BLEND_TRACKER_PATH` (`DATA_DIR/voltrade_blend_tracker.json`, written by
    `diagnostics.run_diagnostics`, `:480-500`).
- **Training** (Tier 3, `ml_retrain_safe.py` → `train_model(fast_mode=True)`):
  - `_fetch_training_bars` (A5/A7): most-actives top 50 plus 30 core tickers, capped at 100.
  - `_fetch_earnings_surprises` (H3).
  - Both report an error only when nothing came back (`last_error`); the result surfaces as `TIER3-ML-ERROR` audit.
- **ORPHAN**:
  - `_fetch_historical_news_sentiment` (`:315`): no caller.
  - `_fetch_insider_signal` (`:348`): no caller, and it reads `insider["signal"]`, a key `get_insider_activity`
    never returns (the real key is `insider_signal`).

### N. Node-only decision inputs

- **Diagnostics auto-fix** (every 5th Tier 2): `state.positionSizeMultiplier` and `state.minScoreThreshold`
  (`server/bot.ts:4365-4366`) gate trades (`:5269`, `:5282`). They are driven by cache-file health (§7), and so
  inherit the negative-caching blind spot.
- **Weekly loss** `diagnostics.check_weekly_loss`, fed a single-point history (`:4625-4631`).

---

## 4. Orphaned or off-decision-path modules (proof)

| Module | Callers outside itself (grep) | Verdict |
|---|---|---|
| `institutional_data.py` | `insights.py:151` only; `insights.py` is used only by `server/routes.ts` | ORPHAN from the bot (UI insights only) |
| `etf_data_sources.py` | `etf_analyzer.py` only, called by `server/routes.ts:6427` | ORPHAN from the bot (ETF builder UI) |
| `intraday_shorts.py` | `close_open_shorts` at 15:50 ET (`server/bot.ts:4645-4660`), dashboard route (`routes.ts:6995`); scan step 8 hard-disabled (`bot_engine.py:3816-3819`) | OBSERVE/cleanup only |
| `manipulation_detect.py` | `server/bot.ts:5802` hourly → `MANIPULATION` audits only; `is_ticker_flagged` has no caller | OBSERVE (**[live]** 25 MANIPULATION lines in the last 200 audits) |
| `cftc_cot.py` | Tier 3 `run_daily_update` (`server/bot.ts:5879`), 20h self-guard (`cftc_cot.py:58`), archive `DATA_DIR/voltrade_cot_archive.json`; docstring "Nothing here is wired into deep_score/tier decisions" (`:12-13`) | OBSERVE (gate-1 archive) |
| `sec_form4_bulk.py` | Tier 3 `run_update` (`server/bot.ts:5900`), 12h guard, `DATA_DIR/form4_bulk/` | OBSERVE (gate-1 archive) |
| `alphadesk/` | only `server/routes.ts:767-772,6374,6400` (`python3 -m alphadesk …`) | ORPHAN from the bot |
| `probability_engine.py` | only `/api/system/snapshot` (`server/bot.ts:2171`) | display only |
| datacore / `/api/v1` | no Python decision module and no bot.ts trade path consumes datacore; bot.ts uses datacore only for port-dwell capture (`:5932`) and diag probes | **no datacore signal reaches the bot** |
| `macro_data.get_sector_for_ticker` (Polygon) | no caller | ORPHAN |
| `social_data.get_google_trends` | disabled at the call site | ORPHAN |
| `intelligence.record_earnings_outcome`, `update_event_outcomes`, `get_event_learning_stats` | no caller | ORPHAN (earnings_pattern is permanently empty) |
| `options_scanner._get_next_earnings_date` | **does not exist**; deep_score's import fails (`bot_engine.py:1597`) → swallowed | dead reference |
| daemon RPC `macro_snapshot` (`voltrade_daemon.py:292`) | never called from Node | ORPHAN RPC |
| `POLYGON_KEY` reads in `bot_engine.py:182`, `intelligence.py:23`, `social_data.py:20` | unused in those files | dead env reads |
| WS `streamSignalQueue` | audited `STREAM-PRIORITY` then cleared (`server/bot.ts:4381-4387`) | ORPHAN signal |

---

## 5. Contract mismatches (consumer reads a key the producer never emits)

| Consumer (file:line) | Key read | Producer | Effective value at runtime |
|---|---|---|---|
| deep_score ML + markov (`bot_engine.py:1456`, `:1703`) | `intel["vxx_ratio"]` | `get_full_intelligence` (`intelligence.py:557-565`) | **1.0 always**; ml_score never picks the `bull` model |
| deep_score ML (`:1710`), `_intel_score_feat` (`:1582`) | `intel["earnings_surprise"]` | same | 0 |
| deep_score ML (`:1695`, `:1725`) | `vol_metrics["ewma_vol"]` | analyze emits `ewma_rv`, not `ewma_vol` (`analyze.py:2147-2152`) | 2 (and trend_strength divides by 2) |
| deep_score ML (`:1464`, `:1697`) | `quick_result["high_52w"]` | quick_results (`:2848-2867`) | `price_vs_52w_high` = 0 |
| deep_score ML (`:1696`) | `quick_result["range_pct"]` | hardcoded 0 (`:2864`) | 0 |
| deep_score squeeze (`:1131-1132`) | `edge["short_float_pct"]`, `edge["days_to_cover"]` | analyze `edge_factors` has neither (`analyze.py:2350-2402`; short % lives at `fundamentals.short_pct_float`) | 0 / 0 → squeeze score only from sentiment and volume |
| squeeze (`strategies/squeeze.py:28-29`) | `reddit_buzz ≥ 5/8` | analyze `reddit_buzz` is a 0–1 fraction | never fires |
| deep_score ML (`:1658`) | `macro["market_breadth_ma50"]` | not produced | SPY/MA50 proxy always |
| deep_score ML (`:1494`) | `macro["spy_change_pct"]` | not produced | `idiosyncratic_ret` = change_pct |
| deep_score ML (`:1546-1557`) | most-actives `change` | Alpaca most-actives (`symbol`, `volume`, `trade_count`) **[verify]** | `cross_sec_rank` = 0.5 |
| deep_score markov (`:1454`) | `_spy_rets = []` | hardcoded | `markov_state` = 1 |
| deep_score (`:1597`) | `options_scanner._get_next_earnings_date` | does not exist | ImportError swallowed (feature forced 0.5 anyway, `:1735`) |
| `_fetch_insider_signal` (`ml_model_v2.py:354`) | `insider["signal"]` | key is `insider_signal` | 0 (and the function is orphaned) |
| tier engine (`bot_engine.py:3920`) | `_regime_ctx` local | deep_score-scoped | `"NEUTRAL"` always |
| floors (`:5015-5017`) | `macro["qqq_ma50"/"qqq_ma200"/"qqq_price"]` | not produced | triggers its own bars fetch |
| deep_score ML gate (`:1745`) | `model_type != "fallback"` | fallback emits `"fallback_rules"` | rules fallback blended as ML |

### 5a. ML feature provenance at inference (34 keys of `ml_features`, `bot_engine.py:1687-1740`)

| Feature | Real source | Status |
|---|---|---|
| `momentum_1m`, `momentum_3m` | deep 400d Alpaca bars (iex) | live (0 if bars fail) |
| `rsi_14` | analyze `vol_metrics.rsi_14` | live (default 50) |
| `volume_ratio` | analyze `volume_ratio_5d` | live (default 1.0) |
| `vwap_position` | snapshot c > o | proxy |
| `adx` | deep bars | live (default 20) |
| `ewma_vol` | missing key | **const 2** |
| `range_pct` | hardcoded | **const 0** |
| `price_vs_52w_high` | missing key | **const 0** |
| `vrp` | analyze (yfinance IV − rv20) | live (None→0) |
| `iv_rank_proxy`, `iv_rank_stock` | HV-rank of deep closes | live (default 50) |
| `atr_pct` | deep bars | live (default 3.0) |
| `vxx_ratio` | `intel` missing key | **const 1.0** |
| `spy_vs_ma50` | macro | live (default 1.0) |
| `markov_state` | empty returns | **const 1** |
| `regime_score` | markov(const) + vxx(const) + spy_ma | function of spy_ma only |
| `sector_momentum` | macro × SECTOR_MAP (~50 tickers) | 0 for most candidates |
| `cross_sec_rank` | most-actives `change` | likely **const 0.5** [verify] |
| `earnings_surprise` | missing key | **const 0** |
| `put_call_proxy` | derived from vrp | live |
| `vol_of_vol` | macro `vxx_closes` / own VXX fetch | live (default 2.0) |
| `frac_diff_price` | deep closes | live |
| `idiosyncratic_ret` | change_pct − 0 | = change_pct |
| `change_pct_today` | snapshot | live (15-min delayed) |
| `above_ma10` | deep closes | live (default 1.0) |
| `trend_strength` | \|change\| / const 2 | degenerate |
| `intel_score`, `news_sentiment`, `insider_signal` | hardcoded 0.0 (`:1728-1730`) | const by design |
| `days_to_earnings` | hardcoded 0.5 (`:1735`) | const by design |
| `credit_spread` | TLT/HYG bars (60s cache) | live (default 0.0) |
| `market_breadth` | SPY/MA50 proxy | proxy |
| `put_call_ratio` | macro `vxx_latest`/20 | live (default 1.25) |

---

## 6. Silent fallbacks that fabricate values (consolidated)

| Feed failure | Fabricated value | Direction of effect | Recorded? |
|---|---|---|---|
| yfinance VIX | `vix=20.0` → `vix_regime="high"` | sizing ×0.80; stress +30 component | no |
| Sector ETF snapshot | all sectors 0.0 → `market_regime="risk_off"` | −5 every deep score | no |
| VXX bars | `vxx_ratio=1.0` | regime NEUTRAL → scanner stock slots 0 (fail-closed) | `data_quality="degraded"`, `data_quality_failures=["vxx_ratio"]` (floor only) |
| SPY bars | 1.0 / 0 / True | regime NEUTRAL; breadth stress 25 | no |
| analyze sentiment (Reddit non-200) | `reddit_buzz=0.5` → component 100 | bullish (+squeeze) | no |
| analyze realized vol | `rv*=20.0` | vrp distorted | no |
| analyze no option chain | error → no deep score | candidate cannot trade | `_diag["stock_details"]` |
| yfinance fundamentals | `{}` | 0 adj | no |
| News (Alpaca) | score 0 | neutral | no |
| EDGAR insider | 0/0 always (schema bug) | neutral | no |
| GDELT 429 | `risk_level="low"` cached 2h | understates risk | no |
| USPTO | 0 cached 24h | neutral | no |
| FRED | missing flags | removes −5/−3/−8 penalties; stress comps 50 | no |
| Finnhub | `{…default, error}` cached 6–12h | neutral | error text inside the cache file only |
| Finnhub calendar 401 | `({}, True)` | "no earnings soon" → CSP earnings factor treated as clean | no |
| Earnings date (yfinance, sizing) | 0.85x | fail-closed (good) | no |
| Quote spread | allow trade | fail-open | no |
| Halt check | not halted | fail-open | no |
| Account (Python scan) | equity/cash 100,000 | sizing on a fake equity | debug log only |
| Convexity put quote | 0.7%·QQQ limit price | order at a fabricated price | log.debug |
| stress_index all-fail | 38.75 "elevated" | ×0.75 sizing | no |
| ML model stale | rules fallback blended as ML | score changes | reason string says "ML model:" |
| Bars without `start` (floors, third leg) | ≤1 bar [verify] | trend override and SVXY entry never fire | no |

---

## 7. Feed health already exposed

| Surface | Code | Fields (exact) | What it really measures |
|---|---|---|---|
| deep_score per-source error map | `_source_diag` (`bot_engine.py:3021-3029`) → scan `data_source_errors` (`:4075`) → `tier2LastDataSourceErrors` (`server/bot.ts:4475-4476`) → **`GET /api/diag/scanner`** (`:2678-2692`, token-gated) | `probe`, `degraded`, `consecutiveFailures`, `lastFailureDetail`, `dataSourceErrors{macro,intel,alt,social,finnhub,stock_details}` | Only import errors, 15s timeouts, and analyze.py failures. It has no ticker, no count, no timestamp, and is last-writer-wins across candidates. It is not updated on a failed scan (stale), and it is `{}` when deep scoring was skipped by the memory guard. **[live] `{}`**. Tests: `test_deep_score_source_diag.py`, `test_deep_score_timeout_diag_capture.py`, `test_get_stock_details_diag.py`, `server/scannerHealth.test.ts` |
| Snapshot failure reason | `_parse_snapshot_batch` → `debug_detail` → `tier2LastFailureDetail` | `lastFailureDetail` | Only when every batch is empty. Test `test_snapshot_parse_diag.py` |
| Public scanner bit | `/api/health` `checks.scanner` (`server/bot.ts:1555-1559`) | `status`, consecutive count | scan pass/fail, not data |
| `/api/health` alpaca | `:1475-1480` | `checks.alpaca.status`, `account_status` | trading API account read only; **not** market-data entitlement |
| `/api/health` feeds | `observeFeedDeadAir` (`server/feedDeadAir.ts:62`) | `checks.feeds{status,dead,feeds,detail}` | **datacore aircraft/vessels/trains only.** The design (archive-throughput, not connection state) is the right model to copy |
| Scan phase timings | `_timing_log` → `DATA_DIR/voltrade_scan_timings.json` → **`/api/diag/timings`** (`:2694-2720`) and `/api/system/snapshot` | `scan_started`, `phases[{phase,duration_sec,total_sec}]`, `status`, `last_phase_completed`, `last_phase_at`, `total_sec`, `tier_engine_breakdown.tier_timings{tier1_sec, csp_layer2_prefetch{…}, …}` | phase latency, not data freshness |
| Daemon health | `voltrade_daemon._health` (`:442-461`) → **`/api/diag/daemon`** (`server/bot.ts:2549-2557`) | `alive`, `uptime_seconds`, `rss_mb`, `max_rss_mb`, `modules_loaded`, `active_dispatches`, `active_dispatch_detail[{method,elapsed_sec}]`, `layer2_prefetch{cache_hit,completed,total,elapsed_sec,budget_exceeded,checked_at,age_sec}`, `layer1_stats{cache_hit,universe_size,elapsed_sec,checked_at,age_sec}`, `surface_score_stats{calls,cache_misses,cumulative_elapsed_sec,skipped_budget,checked_at,age_sec}`, `pid` | CSP universe and vol-surface cost/coverage. **[live]** all `{}` right after a restart |
| Cache inventory | daemon `_cache_inventory` (`voltrade_daemon.py:467-488`) → **`/api/monitoring/caches`** (`server/bot.ts:3549-3572`, owner) | `caches[{path,size_kb,age_seconds}]` (top 20 by size; the subprocess fallback omits `age_seconds`) | file age, which negative caching makes unreliable as a health signal |
| Diagnostics | `diagnostics.run_diagnostics` (Tier 3 hourly → `TIER3-DIAG` audit, `server/bot.ts:5835-5874`); `get_auto_fix_params` (Tier 2 every 5th → `DIAGNOSTIC` audit) | `EXPECTED_CACHE_FRESHNESS` (`diagnostics.py:86-94`: macro 1h, insider 2h, fred 8h, gdelt 4h, ml_model 168h, event_memory 48h, trade_feedback 168h; stale = >1.5× max); `api_checks` (`:389-395`: "polygon"=macro cache file (a misnomer), "sec_edgar"=insider cache, wikipedia/gdelt/fred files); `extended_checks` (`:446-453`: reddit_*, fh_* files); report persisted to `DATA_DIR/voltrade_health_log.json` (last 100) | file existence and mtime. **Defeated by negative caching**. ≥3 "down" → 0.6x sizing; `tier2_ran_since_boot` grace. **[live]** TIER3-DIAG shows only the ML win-rate issue, no API warnings |
| Macro quality | `get_macro_snapshot` → `/api/system/snapshot` `ml.macro{regime,vxx_ratio,spy_vs_ma50,vix,data_quality}` (`server/bot.ts:2156-2162`) | as listed | VXX only (§D1) |
| Stress index | `/api/system/snapshot` `stress_index{index,level,action_suggestion,components{yield_curve,credit_spread,vix_term_structure,spy_realized_vol,breadth,sector_rotation},weights,timestamp}` | as listed | component values, with no flag for "this is the 50 default" |
| Floor regime | scan `spy_floor{raw_regime,floor_regime_confirmed_by,regime,target_pct,trend_override}`; snapshot `floor_state` | as listed | the only explicit `vxx_fallback` signal (`held_vxx_fallback`) |
| WS stream | audit `STREAM`, `STREAM-ERROR`, `STREAM-DISCONNECT`, `STREAM-SIGNAL`, `STREAM-PRIORITY` | audit text | connect/disconnect events only |
| ML retrain | audit `TIER3`, `TIER3-ML-ERROR` with `last_error` (`ml_model_v2.py:1319-1323`) | audit text | training-bar fetch failure |
| COT / Form4 | audit `TIER3-COT`, `TIER3-FORM4BULK`; datacore `streamsInventory.ts` reads the COT archive (`server/streamsInventory.ts:144`) | audit, /data Streams tab | archive freshness |
| Equity reads | audit `EQUITY-READ-INVALID` (`server/bot.ts:4611-4612`, `3930-3931`) | audit text | account field validity |
| Registry / collection tests | `test_data_stream_registry_check.py` (build-state registry), `test_collection_health.py` (pytest collection) | — | **not** runtime feed health |

---

## 8. Disagreements (docs or comments vs code)

1. `alpaca_feed.py` module docstring: "scan results carry data_feed so the audit trail shows which tape produced
   them". **False**: `feed_status()` has no callers and the scan result has no feed field.
2. `alpaca_feed.options_feed()` docstring (`:177-180`): scanner/manager/surface "still hardcode feed=opra".
   **Stale**: they use `options_feed()`.
3. `macro_data.get_news_sentiment` docstring "Polygon news headlines" (`:404`). **It is Alpaca news.**
4. `diagnostics.py:390` labels the macro cache check `"polygon"`. Macro uses yfinance and Alpaca.
5. `bot_engine.get_stock_details` docstring: "analyze.py --mode=scan". The flag is ignored and a full
   analysis runs.
6. `social_data` docstring ranks Google Trends as the strongest predictor (35%). **It is hardcoded off.**
7. `system_config.CONVEXITY_OVERLAY.enabled=False` ("Disabled 2026-04-17"). **The code never reads it**, and the
   overlay runs every scan.
8. `voltrade_daemon.py:203-210` says vol_surface fetches "are NOT gated by alpaca_throttle". Inside the daemon,
   `bot_engine` import calls `install_global_throttle()` (`bot_engine.py:63-68`), which monkey-patches
   `requests.get` for all `alpaca.markets` URLs (`alpaca_rate_limiter.py:134-171`). So they **are** gated. Many call
   sites additionally call `alpaca_throttle.acquire()` explicitly, which **double-spends tokens**:
   `bot_engine.py:2783-2784`, `1509-1510`, `1547-1548`, `1623-1624`; `macro_data.py:156-157`, `254-255`;
   `options_scanner.py` passim.
9. `deep_score` comment "all 6 sources run simultaneously" (`:800`). The pool has 5 workers. analyze.py and
   yfinance run serially before and after it.
10. `markov_regime` docstring describes order-2 Markov on SPY returns. **Production always feeds an empty list.**
11. `/api/bot/status` hardcodes `maxPositionPct: 8`, `dailyLossLimitPct: 5` (`server/bot.ts:2328-2330`). The code
    uses −3% daily loss (`:350`) and regime-adaptive sizing (display drift; not a feed issue, but cockpit-visible).
12. CLAUDE.md SPINOUT-READY: "the bot consumes them the way an external API customer would". **No consumption
    exists today.**

---

## 9. Gaps for a cockpit feed-health strip

What a strip needs that **is not recorded anywhere today**:

1. **Per-feed last success time vs last attempt time.** No fetcher records success separately from attempt. Cache
   mtime is written on failure, so it cannot stand in for last success.
2. **Upstream status code per feed** (HTTP 200/401/403/429/301). Every module discards it except the snapshot
   parser, `_fetch_training_bars.last_error`, and the feed probes.
3. **Is the value real or a default?** No payload carries a provenance flag. A "default substituted" flag is needed
   on at least: macro `vix`, `sector_momentum`, `vxx_ratio`, `spy_*`; sentiment `reddit_buzz`; `rv20`; FRED series;
   stress components; `insider`; `alt` sub-signals; Finnhub.
4. **Current Alpaca tier**: `{data_feed, bars_feed, options_feed, degraded, downgraded_since}`. Read
   `alpaca_feed.feed_status()` and `options_feed_status()` and add them to the scan result or `/api/diag/daemon`.
   Currently nothing reads them.
5. **WS stream throughput**: `streamConnected`, last bar ts, bars per minute, subscribed ticker count, last error.
   Same rationale as `feedDeadAir.ts`: connection-up ≠ data-flowing.
6. **Error counts with time windows** (per feed: n_ok / n_fail over the last N scans). `_source_diag` keeps only
   the last string per key, with no ticker or time.
7. **Per-candidate enrichment coverage**: for each deep-scored ticker, which of analyze, macro, intel, alt, social,
   finnhub and yfinance contributed real data. Plus `analyze.data_source` (alpaca vs yfinance), which is dropped
   today.
8. **ML provenance**: `model_type` (`lightgbm_<regime>`/`lightgbm_global`/`fallback_rules`), `regime_used`, and
   model age per scored candidate. None reach the scan result. Only `ml_only_score` does.
9. **Deep-score coverage**: `deep_analyzed` vs the cap, whether the 35s cap was hit (only a log line, `:3050-3053`),
   and memory-guard decisions (skip/trim are stderr only).
10. **Data age of the snapshot tape** (delayed_sip = 15 min), shown next to every price-derived number. Also the
    staleness of `dailyBar` during pre-market and early-open scans.
11. **Rate-limit pressure**: the `alpaca_throttle` bucket level and wait time; GDELT/EDGAR/Finnhub 429 counts.
12. **Universe completeness**: universe size, the count dropped by each filter (the ".", "W", length rules), cache age.
13. **Negative-cache honesty**: cache writers should store `{ok:false, error, fetched_at}`, and diagnostics should
    check `ok`, not existence.

Suggested shape (**proposal only, does not exist**): one `feed_health` dict per scan, keyed by feed id, each
entry `{last_attempt_at, last_ok_at, last_status, last_error, ok_count, fail_count, default_substituted: bool,
age_sec, source_tier}`. It would be emitted on the scan result next to `data_source_errors` and mirrored on
`/api/diag/scanner`. Natural feed ids:

```
alpaca.assets
alpaca.movers
alpaca.snapshots
alpaca.bars
alpaca.quotes
alpaca.news
alpaca.options
alpaca.ws_iex
alpaca.account
yf.vix
yf.tnx
yf.dxy
yf.fundamentals
yf.options_chain
yf.calendar
analyze.subprocess
stocktwits
reddit.json
reddit.rss
pytrends
edgar.fts
wikimedia
fred
gdelt
uspto
polygon.ref
finnhub.insider
finnhub.reco
finnhub.calendar
rss.news
ml.model_file
```

---

## 10. Verification notes

- **[probe]** results come from the Claude sandbox egress proxy on 2026-10-15 ~03:18Z, not from Railway:
  - EFTS 200 (schema above)
  - Wikimedia 200
  - FRED 200
  - GDELT 429 (plain-text rate-limit body)
  - Reddit RSS 200
  - Reddit search.json 403
  - StockTwits 200
  - USPTO 301 → HTML
  - CNBC RSS 200
  - Reuters agency / MarketWatch RSS 301
- No Alpaca, Finnhub or Polygon keys exist in this sandbox. Items marked **[verify]** need one live call or a
  production read:
  - Alpaca bars without `start`
  - the most-actives schema
  - whether Alpaca accepts `side:"buy_to_open"`
  - the analyze.py default bars feed
  - delayed_sip `dailyBar` at the open
  - Railway's reachability of Reddit, StockTwits and GDELT
  - whether `POLYGON_KEY` and `FINNHUB_KEY` are set in production
- **[live]** data is from the sibling capture in
  `research/bot_cockpit_audit/live_snapshot_2026-10-15.md (digest; raw probes not committed) `
  (`scanner.json`, `daemon.json`, `timings.json`, `audit.json`, `types/*.json`, `orders.json`).
