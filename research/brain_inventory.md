# Phase 0: Brain inventory (2026-10-27)

What the trading bot reads, what decides, and what actually runs. Written for the
"VolTradeAI Master Prompt" plan (human, 2026-10-27). Docs only: nothing here
changes behaviour.

**Sources.** The read-only trading-engine audit of 2026-10-27 (main @ v1.0.1168,
`f9892473`; cited as "audit §N"; the file lives in the session scratchpad, so
the code citations here are the durable reference), plus re-reads of the
current code this session. Line numbers are approximate (`~`). Search by
function name when they drift.

**Re-verified in code this session** (not just copied from the audit):
the 34-name `FEATURE_COLS` list (`ml_model_v2.py:140-186`); the serve-time
`ml_features` dict (`bot_engine.py` `deep_score` ~1731-1782), including the
constants `intel_score`/`news_sentiment`/`insider_signal`=0.0 and
`days_to_earnings`=0.5; `ewma_vol` read from a key analyze.py never emits
(analyze emits `ewma_rv`); `high_52w` never set on quick results (~1508);
`range_pct` hard-coded 0 (~2990); `spy_change_pct` and `market_breadth_ma50`
never emitted by `macro_data.py`; `_regime_ctx` scoped to `deep_score`, so
the tier engine gets "NEUTRAL" (~4140); the tie-break reads
`current_positions` (~3231) before it is assigned (~3336); the deep_score
side logic (~1852-1875) and the sell-to-buy rewrite (~3602-3604);
`position_sizing.py:1004` reads `peak_equity` from the peak file;
`backtest_v2.DEFAULT_EXIT_MODEL = "legacy"` (:169); Google Trends disabled
(`social_data.py:428`); `diagnostics.check_weekly_loss` returns "continue"
with fewer than 2 rows (:44); `runOvernightResearch` is only called at
ET ≥ 20 or < 4 (`server/bot.ts` ~5691), while Tier 2 only runs at 4 ≤ ET < 20
or when the market is open (~8232), and the 4am branch (~6747) is therefore
unreachable; `strategyWeights` is written but never read (`server/bot.ts`
~859, ~1021-1031); `FLOOR_AND_LEG_TICKERS` (~409); Node limits
`DAILY_LOSS_LIMIT=-3`, `MAX_POSITION_SIZE=0.20`, `MAX_TOTAL_EXPOSURE=0.30`,
`MAX_POSITIONS=8`, `maxDrawdownPct=-10` (~361-364, ~514).

**Column legend.**
- *Consumer*: rules score / ML / sizing / regime / exits / gate / none.
- *Status*: **live** (real values reach a decision), **constant** (a fixed or
  default value reaches the decision), **dead** (not read, or unreachable),
  **broken** (reaches a decision with wrong, stale or mis-joined values).
- *Action*: keep / fix / replace / remove. "(shadow)" means the change alters
  trading, so under the master-plan rule it ships shadow-logged first.
- "N#" = audit §6.8 conflict number. "KB #" = `research/open_questions.md`
  KNOWN BROKEN. "D#" = audit §1.x defect.

---

## 1. Inputs

### 1.1 Price, volume and account data

| Input | Source / provider | Code path | Consumer | Status | Evidence | Action |
|---|---|---|---|---|---|---|
| Tradable universe | Alpaca `/v2/assets?status=active` + screener movers top 50 | `bot_engine._get_full_universe` ~2471; `_scan_market_inner` ~2852-2883 | gate (scan universe) | live | audit §3.1 stage 1 | keep (no delisted names, so any training on it has survivorship bias, audit §2.0) |
| Stock snapshots (~12k/scan) | Alpaca `/v2/stocks/snapshots`, `data_feed()` = `delayed_sip` (15 min late; SIP real-time 403 since 2026-07-06) | `_scan_market_inner` ~2887-2934, `_parse_snapshot_batch` ~2615 | quick score, price/volume floor, `change_pct_today`, `vwap_position` | live (15-min delayed) | audit §2.1, §3.1 stage 2 | keep; paid real-time feed is a human decision |
| Quick score | derived from snapshots | `_scan_market_inner` ~2969-3009 | gate (top 5 to deep score) | live; direction-blind | audit §3.1 stage 4 | fix (shadow): rank losers and winners separately |
| Daily bars, 400 days | Alpaca `/v2/stocks/{t}/bars`, feed `iex` | `deep_score` ~1013-1077 | rules score (momentum 12-1, ADX, OBV), ML (`momentum_*`, `atr_pct`, `frac_diff_price`, `above_ma10`, `iv_rank_*`) | live | audit §4.1, §4.2 | keep |
| Latest quote (spread check) | Alpaca `/v2/stocks/{t}/quotes/latest` | `_scan_market_inner` ~3610-3631 | gate (spread > 0.5%) | live; fail-open on error | audit §6.2 | keep |
| 1-minute bars of held tickers | Alpaca WebSocket `v2/iex` | `server/bot.ts` `startStreaming` ~7793 → `checkPositionOnTick` ~7268 | exits | live (IEX single venue) | audit §1.3, §6.4 | keep |
| Stream spike signals | same WebSocket | `server/bot.ts` ~7860-7890 → `streamSignalQueue` | none (`run_full_scan` is called with `{}`) | dead | audit §6.2 last row | remove, or wire it in (shadow) |
| analyze.py per-ticker bundle (RSI, `volume_ratio_5d`, VRP, `ewma_rv`, RV20, edge factors, recommendation, sentiment) | analyze.py subprocess: Alpaca bars, then yfinance; yfinance option chains | `bot_engine.get_stock_details` ~475 → `analyze.analyze_ticker` ~1930 | rules score, ML | live; `--mode=scan` ignored, so the full analysis runs per ticker | audit §1.2, D9 | fix (honour scan mode; move into the daemon) |
| Fundamentals (rating, target upside, fwd P/E, EV/EBITDA) | yfinance `.info`, `.recommendations` (unkeyed, no SLA) | `deep_score` ~938-996 (5 s subprocess), applied ~1403-1446 | rules score (about −18 to +16) | live; never ladder-validated | audit §4.2 | remove from live score (shadow) until gate 2 |
| Earnings date | yfinance `.calendar` (12 h cache) | `position_sizing._fetch_earnings_date_with_status` ~637, `_earnings_scalar` ~694 | sizing, options earnings guard | live; fail-open | audit §2.1 yfinance row, §6.2 | fix (fail closed; one earnings source) |
| Earnings calendar (CSP gate) | Finnhub `/calendar/earnings` | `options_scanner._fetch_earnings_calendar_with_status` ~641 → `csp_universe._layer2_score` | gate (CSP earnings ≤ 7 d) | live; bypasses the Finnhub rate limiter; KB #45 path (c) unconfirmed | audit §2.1 Finnhub row | fix (route through `_finnhub_get`) |
| TLT / HYG bars | Alpaca bars | `deep_score` ~1660-1686 | ML `credit_spread` | live | audit §4.1 | keep |
| Most-actives "change" field | Alpaca `/v1beta1/screener/stocks/most-actives` | `deep_score` ~1583-1602 | ML `cross_sec_rank` | constant (probably 0.5: items likely carry no `change`; not verified live) | audit §4.1 | replace (rank against the scan's own snapshot changes) |
| Account equity, last_equity, buying power | Alpaca `/v2/account` | `bot_engine.get_alpaca_account` ~512; `server/bot.ts` `tier2Intelligence`, `checkDrawdownKillSwitch` | sizing, halts | live | audit §1.3 | keep |
| Positions | Alpaca `/v2/positions` | `bot_engine.get_alpaca_positions`; `server/bot.ts` `syncMonitoredPositions` ~6906 | gates (slots, held), exits, halts | live | audit §1.3 | keep |
| Trade feedback (win rate / avg win / avg loss per bucket) | `voltrade_trade_feedback.json` (written by `ml_model_v2.track_fill`) | `position_sizing._kelly_fraction` ~153 | sizing (Kelly), ML training | broken: 86% orphan exits, 362/398 null `pnl_pct`, fill price has a synthetic haircut | audit §7.1, §7.2 item 4 | fix (record real fill; link exits to entries) |
| Equity peak (four separate stores) | Node `voltrade_equity_peak.json {equityPeak}`; `voltrade_portfolio_dd.json`; `voltrade_peak_equity.json`; `voltrade_killswitch.json` | `server/bot.ts` `saveEquityPeak` ~638; `bot_engine.update_equity_peak` ~584; `risk_kill_switch`; `tiered_strategy` | halts, sizing | broken: sizing reads key `peak_equity` from the Node file (`position_sizing.py:1004`), so its DD scalar never applies | audit N13, §6.6 | fix (one peak store; read the right key) |
| Blend weights | `voltrade_blend_tracker.json` (written by `diagnostics.run_diagnostics` ~476) | `deep_score` ~1810-1832 | ML blend weight (~0.72 rules / 0.28 ML live) | live | audit §3.4 item 2 | keep; re-derive after the model is fixed |
| Score threshold / size multiplier | `diagnostics.get_auto_fix_params` every 5th Tier-2 cycle | `server/bot.ts` ~4800 → `state.minScoreThreshold`, `positionSizeMultiplier` | gate, sizing | live (overrides Python thresholds, N12) | audit §1.3, N12 | fix (one threshold source) |

### 1.2 News and social

| Input | Source / provider | Code path | Consumer | Status | Evidence | Action |
|---|---|---|---|---|---|---|
| Headline sentiment | Alpaca `/v1beta1/news` (keyword scoring) | `macro_data.get_news_sentiment` ~402 → `deep_score` ~1244-1256 | rules score (+10/+5/−6/−12) | live; unvalidated | audit §4.2, §9.4 | remove from live score (shadow) until gate 2 |
| News classification (intel) | Alpaca news (limit 15) | `intelligence.classify_news` ~115 → `get_full_intelligence` ~527 → `deep_score` ~1295-1340 | rules score (part of intel ±15; trap ≤ −10; sell-the-news −5) | live; unvalidated | audit §9.4 | remove from live score (shadow) until gate 2 |
| Multi-source news RSS | Reuters / CNBC / MarketWatch RSS | `social_data.get_news_multi_source` ~306 | rules score (social ±15, 25% weight) | live; unvalidated | audit §2.1, §4.2 | remove (shadow) until gate 2 |
| Reddit sentiment / buzz | Reddit RSS (unofficial) | `social_data.get_reddit_sentiment` ~76 | rules score (social ±15; squeeze input) | live; unvalidated | audit §4.2 | remove (shadow) until gate 2 |
| Google Trends | pytrends | `social_data.get_google_trends` ~215 | rules score (35% of social) | constant: disabled, signal 0 (`social_data.py:428`) | code | remove the 35% weight slot (or re-enable with a real source) |
| analyze sentiment (StockTwits, Reddit search, pytrends) | unofficial, unkeyed | `analyze.get_sentiment` ~878-950 | rules score (contrarian +10/+8/+4; squeeze) | live; source liveness unverified; unvalidated | audit §2.1 | remove (shadow) until gate 2 |
| ML `news_sentiment` | — | `ml_features` ~1772 | ML | constant 0.0 | code | see §1.6 |

### 1.3 Insider and institutional

| Input | Source / provider | Code path | Consumer | Status | Evidence | Action |
|---|---|---|---|---|---|---|
| Form 4 filing counts | SEC EDGAR full-text search (hit counts only, not parsed buys/sells) | `intelligence.get_insider_activity` ~224 | rules score (inside intel ±15) | live; unvalidated; no direction parsing | audit §2.1 EDGAR row | replace with the parsed `sec_form4_bulk` archive once it passes gate 2 |
| Insider MSPR | Finnhub `/stock/insider-sentiment` | `finnhub_data.get_insider_sentiment` ~139 → `deep_score` | rules score (±5, only inside `if intel:`) | live; unvalidated | audit §4.2 | remove (shadow) until gate 2 |
| Analyst recommendation trend | Finnhub `/stock/recommendation` | `finnhub_data.get_recommendation_trends` ~406 | rules score (+3) | live; unvalidated | audit §4.2 | remove (shadow) until gate 2 |
| Form 3/4/5 bulk archive | SEC bulk | `sec_form4_bulk.run_update` (Tier 3) | none (archive, gate 1) | live as archive | audit §2.1 | keep (research input) |
| 13F holdings | SEC | `institutional_data.get_institutional_score` ~512 | none (UI `insights.py` only) | dead for trading | audit §1.5, §9.4 | keep (UI) |
| Congressional trades | SEC EDGAR search | `alt_data.get_congressional_signal` ~364 | rules score (alt ±20) | live; unvalidated | audit §2.1 | remove (shadow) until gate 2 |
| ML `insider_signal` | — | `ml_features` ~1773 | ML | constant 0.0 | code | see §1.6 |

### 1.4 Alternative data

| Input | Source / provider | Code path | Consumer | Status | Evidence | Action |
|---|---|---|---|---|---|---|
| Wikipedia attention | Wikimedia pageviews | `alt_data.get_wiki_attention` ~67 | rules score (alt ±20) | live; unvalidated | audit §9.4 | remove (shadow); the datacore wiki-attention stream is the validation path |
| Short-interest proxy | volume pattern (Alpaca bars), not real SI | `alt_data.get_short_interest` ~285 | rules score (alt) | live; a proxy labelled "short interest" | code docstring ~287-290 | replace (FINRA short volume archive exists in datacore) or remove |
| Patent activity | USPTO `developer.uspto.gov` | `alt_data.get_patent_activity` ~423 | rules score (alt) | unknown liveness (endpoint unverified) | audit §2.1 | remove |
| FTD signal | Polygon `/v3/reference/tickers` (shares outstanding) + FTD | `alt_data.get_ftd_signal` ~464 | rules score (alt) | live if `POLYGON_KEY` is set; prod env-var name unknown (`POLYGON_KEY` vs `POLYGON_API_KEY`) | audit §2.0 | fix env split, then validate (shadow) |
| Geopolitical risk | GDELT DOC API (5 titles) | `alt_data.get_geopolitical_risk` ~205 | rules score (alt; same value for every ticker) | live; global, so it shifts every candidate equally | code ~536 | remove from per-ticker score |
| FRED flags (yield curve, labour, credit) | FRED keyless CSV | `alt_data.get_fred_macro` ~134 | rules score (−5/−3/−8, global), stress index | live | code ~545-552 | move to regime/sizing only (remove from per-ticker score) |
| Alt composite | sum of the above | `alt_data.get_alt_data_score` ~517 → `deep_score` ~1343-1371 | rules score (clamp ±20) | live; unvalidated | audit §9.4 | remove (shadow) until gate 2 |
| Datacore streams (ADS-B, AIS, weather, grid, CFTC, FINRA, CBOE, OCC, EDGAR, etc.) | many | `server/*` datacore modules | none (archive + diag probes only) | dead for trading | audit §2.3 | keep as archives; enter only through the ladder |
| CFTC COT | CFTC Socrata | `cftc_cot.run_daily_update` (Tier 3) | none | live as archive | audit §2.1 | keep |
| AlphaDesk | EDGAR / catalysts | `python -m alphadesk` via `/api/research/:ticker` | none (UI) | dead for trading | audit §1.5 | keep (UI) |
| Polygon grouped daily | Polygon | `ml_model.py` (v1, never imported) | none | dead | audit §3.1, §9.3 | remove `ml_model.py` |

### 1.5 Macro and regime inputs

| Input | Source / provider | Code path | Consumer | Status | Evidence | Action |
|---|---|---|---|---|---|---|
| VIX, VIX change | yfinance `^VIX` | `macro_data.get_macro_snapshot` ~96-118 | rules score (`vix_regime` ±5/−10), sizing (regime scalar) | live; silently defaults to 20.0 on fetch failure | code ~103, ~106 | fix (flag the default; prefer the VXX ratio already fetched) |
| 10y yield, dollar index | yfinance `^TNX`, `DX-Y.NYB` | `get_macro_snapshot` ~96-139 | `market_regime` risk on/off (rules ±3-5) | live | audit §2.1 | keep |
| VXX ratio (VXX / 30d avg), VXX closes | Alpaca bars | `macro_data` ~210-300; `regime_inputs.regime_inputs` | regime (all classifiers), ML (`vxx_ratio`, `vol_of_vol`, `put_call_ratio`), instrument IVR | live (ML copy was pinned at 1.0 before 2026-10-16) | audit §4.1 | keep |
| SPY vs MA50, SPY above 200d, days below 200d | Alpaca bars | `macro_data` ~210-237 | regime, ML (`spy_vs_ma50`, `regime_score`) | live; but `get_adaptive_params` cannot take `spy_above_200d` and the slot call omits `spy_below_200_days` (N10) | audit §6.5, N10 | fix (one regime call per scan) |
| SPY 5-day returns (Markov) | Alpaca bars | `markov_regime.get_regime` ~229 via `deep_score` ~1502 | ML (`markov_state`), regime size multiplier | live since 2026-10-16 | audit §4.1 | keep |
| Sector ETF day % | Alpaca snapshots | `macro_data` ~151-171 | rules score (±8), ML `sector_momentum` | live; static `SECTOR_MAP`, "Other" gives 0 | audit §4.1 | fix (real sector lookup) |
| `spy_change_pct` | not produced | `deep_score` ~1538 | ML `idiosyncratic_ret` | constant (missing key gives 0, so the feature equals `change_pct_today`) | code | fix (emit it in `macro_data`) |
| `market_breadth_ma50` | not produced | `deep_score` ~1702 | ML `market_breadth` | constant source; falls back to a function of `spy_vs_ma50` | code | fix or replace |
| `macro["regime"]` | not produced | `position_sizing` ~1073 | sizing regime scalar | constant (always 1.0) | audit §6.5 | fix |
| Stress index (yield curve, credit, VIX term, SPY RV, breadth, sector rotation) | FRED keyless + market data | `stress_index.get_stress_multiplier` ~223 | sizing | live; unvalidated | audit §6.3 | keep (shadow-log its effect) |
| Floor regime with 3-cycle dwell | macro snapshot | `bot_engine._resolve_floor_regime` ~5047 | floor/defensive targets | live | audit §6.4 | keep |

### 1.6 Options data

| Input | Source / provider | Code path | Consumer | Status | Evidence | Action |
|---|---|---|---|---|---|---|
| Contract chain for entries | Alpaca `/v1beta1/options/snapshots`, `indicative` feed (OPRA 403 since ~2026-07-25); limit 100, one page | `options_execution._fetch_option_chain` ~566-717 | contract selection, liquidity gate | broken: "open interest" is a `max(bid_size, ask_size)×10` proxy; indicative quotes are not NBBO; single page can miss strikes | audit §2.4 | fix (paginate; label the proxy; real OI needs a feed decision) |
| Scanner chain | same endpoint, limit 1000, ≤ 5 pages, 180 s cache | `options_scanner._fetch_options_chain` ~283 | options scanner setups | live; same OI proxy; moneyness-table delta proxy when greeks missing | audit §2.4 | fix (label proxies) |
| IV rank (scanner/CSP) | same endpoint, hard-coded `indicative`, limit 100, no strike filter | `options_scanner._fetch_iv_rank` ~532 | CSP universe (25% of composite), setups | broken: may find no ATM strike and silently fall back to HV rank | audit §2.4 | fix (strike filter; log fallback) |
| Position greeks/quotes for exits | same endpoint `?symbols=` | `options_manager._get_option_snapshot` ~182 | options exits (gamma, delta, profit) | live; `symbols=` behaviour unverified; root parse breaks on adjusted roots (e.g. `IONQ1`) | audit §2.4 | fix |
| Vol surface / skew | same endpoint, ≤ 2,000 contracts | `vol_surface.build_surface` ~829 | CSP universe (skew 10%) | live | audit §2.4 | keep |
| yfinance chains (ATM IV, VRP, skew, GEX, put/call OI) | yfinance `.option_chain` ≤ 8 expiries | `analyze.analyze_ticker` ~1994-2107 | rules score (VRP weight 0.25), side ("SELL OPTIONS"), ML `vrp` | live; a second, separate IV source from the one used to trade | audit §2.4, §9.5 IV row | replace (one IV definition) |
| Per-stock "IV rank" | 30-day **HV** rank from bars | `bot_engine._compute_stock_iv_rank` ~20 | ML `iv_rank_proxy`, `iv_rank_stock` | live but misnamed (HV, not IV) | audit §4.1, §9.5 | replace or rename |
| Market IVR | VXX ratio mapped to 20-85 | `instrument_selector` ~1358-1374 | instrument choice (sell premium if IVR > 60) | live; a proxy | audit §6.1 | keep (label as proxy) |
| QQQ put quote | `/v1beta1/options/quotes/latest`, indicative; 0.7%-of-QQQ estimate if missing | `_run_convexity_overlay` ~4636 | convexity overlay | dead (overlay disabled) | audit §2.4 | keep disabled |
| Daily chain archive | Alpaca indicative snapshots, ≤ 120 underlyings, since 2026-07-05 | `server/optionsChainArchive.ts` | none (research probes `chain_iv30`, `chain_csp_entry`) | broken: day label shifted one session (Node 20 hour "24"), SPY chains truncated at ~39 DTE, no OI, no gamma, no volume, no spot | audit §2.4 defects 1-3 | fix (Phase 6) |
| OPRA real-time options | Alpaca | `alpaca_feed.options_feed` ~173 | — | dead (HTTP 403) | audit §2.4 | human decision (paid) |

### 1.7 The 34 ML features (`ml_model_v2.FEATURE_COLS`, serve values from `deep_score` `ml_features` ~1731-1782)

All consumers are **ML**. Production currently trains only on shadow records, which carry the serve-time values (audit §5.6), so "constant at serve" means "constant in training too". Train/serve definition gaps matter whenever generic rows dominate.

| # | Feature | Serve source (code path) | Status | Evidence | Action |
|---|---|---|---|---|---|
| 1 | momentum_1m | Alpaca 400d bars, 20-bar return (~1093-1106) | live | audit §4.1 | keep |
| 2 | momentum_3m | 62-bar return (~1725-1728) | live | audit §4.1 | keep |
| 3 | rsi_14 | `analyze.compute_rsi` (train uses a different simple-average RSI) | live, definition differs | audit §4.1 | fix (one RSI) |
| 4 | price_vs_52w_high | `quick_result["high_52w"]`, never set (~1508) | constant (0) | code | fix (compute from the 400d bars already fetched) |
| 5 | volume_ratio | analyze `volume_ratio_5d` = 5d / 90d (train: today / 20d) | live, definition differs | audit §4.1 | fix |
| 6 | ewma_vol | `vol_metrics.get("ewma_vol", 2)`; analyze emits `ewma_rv` (~1739) | constant (2) | code | fix (read `ewma_rv`) |
| 7 | atr_pct | Wilder ATR14 / price (~1533) | live | audit §4.1 | keep |
| 8 | vrp | analyze per-stock ATM IV − RV20 (train: market-wide VXX proxy) | live, different quantity | audit §4.1, §9.2 | fix (one definition both sides) |
| 9 | iv_rank_proxy | `_compute_stock_iv_rank` = HV rank (train: VXX 52w rank) | live, different quantity, misnamed | audit §4.1 | replace |
| 10 | vwap_position | snapshot close > open (not VWAP; 15-min delayed) | live, misnamed | audit §4.1 | keep (rename) |
| 11 | adx | Wilder DX of last bar (not ADX; train: simplified ratio) | live, definition differs | audit §4.1 | fix |
| 12 | range_pct | quick result hard-coded 0 (~2990) | constant (0) | code | fix (snapshot high − low) |
| 13 | vxx_ratio | `regime_inputs` | live (was 1.0 on all shadow rows before 2026-10-16) | audit §4.1 | keep |
| 14 | spy_vs_ma50 | `regime_inputs` | live | audit §4.1 | keep |
| 15 | markov_state | `regime_inputs` | live | audit §4.1 | keep |
| 16 | regime_score | `regime_inputs` | live | audit §4.1 | keep |
| 17 | sector_momentum | sector ETF 1-day % via static `SECTOR_MAP` (train: stock 5d − SPY 5d) | live, different quantity | audit §4.1 | fix |
| 18 | cross_sec_rank | most-actives `change` field (~1583-1602) | constant (probably 0.5) | audit §4.1 (not verified live) | replace |
| 19 | earnings_surprise | `intel.get("earnings_surprise")`, never emitted | constant (0) at serve; **lookahead leak** in generic training (latest surprise applied to all history) | audit §2.1 side finding, §4.1 | remove, or rebuild point-in-time |
| 20 | put_call_proxy | rule on per-stock `vrp` (not a put/call ratio) | live, misnamed | audit §4.1 | replace (real P/C from chain archive after gate 2) |
| 21 | vol_of_vol | std of 10 VXX daily returns | live | audit §4.1 | keep |
| 22 | frac_diff_price | `_frac_diff` of last 20 closes | live | audit §4.1 | keep |
| 23 | idiosyncratic_ret | change % − `macro["spy_change_pct"]` (missing, so = change %) (~1538) | broken (duplicates #24) | code | fix |
| 24 | change_pct_today | snapshot change (partial day; train: full day) | live, definition differs | audit §4.1 | keep (document) |
| 25 | above_ma10 | last close > mean(last 10) | live | audit §4.1 | keep |
| 26 | trend_strength | \|chg\| / `ewma_vol` (constant 2) | broken (degraded) | code ~1762 | fix (follows #6) |
| 27 | intel_score | hard-coded 0.0 (~1772) | constant | code | replace or remove |
| 28 | news_sentiment | hard-coded 0.0 | constant | code | replace or remove |
| 29 | insider_signal | hard-coded 0.0 | constant | code | replace or remove |
| 30 | iv_rank_stock | same HV rank as #9 | live, duplicate | audit §4.1 | remove or replace |
| 31 | days_to_earnings | hard-coded 0.5; real value computed ~1636-1655 and discarded | constant | code | fix (point-in-time in training too) |
| 32 | credit_spread | TLT 21d − HYG 21d | live | audit §4.1 | keep |
| 33 | market_breadth | falls back to 0.5 + (spy_vs_ma50 − 1)·6 | live, proxy of #14 | code ~1699-1712 | replace (real breadth from the scan's snapshots) |
| 34 | put_call_ratio | VXX latest / 20 (not a P/C ratio; non-stationary level) | live, misnamed | audit §4.1 | replace |

Tally: 9 constant at serve (#4, 6, 12, 18, 19, 27, 28, 29, 31); 2 broken (#23, 26); 12 live but with a different definition, meaning or name from training/label (#3, 5, 8, 9, 10, 11, 17, 20, 24, 30, 33, 34); 11 live and roughly aligned (#1, 2, 7, 13, 14, 15, 16, 21, 22, 25, 32). No feature comes from datacore or any EDGE-DOCTRINE pipeline (audit §4.1).

---

## 2. Rules and models that influence a trade

"Executes?" means: does it change what is actually ordered, blocked, sized or closed today.

### 2.1 Universe, scan and score

| Rule / model | Where | Executes? | Conflicts | Action |
|---|---|---|---|---|
| Price ≥ $5, volume ≥ 500k (100k at 9:30-9:59) | `bot_engine._scan_market_inner` ~2946-2968 (module consts, not `system_config`) | yes | N21 (open-hour floor inverts `OPEN_HOUR_MIN_VOLUME` 1M) | fix (read `system_config`; pick one policy) |
| Quick-score rank, deep-score cap 5 (3 / 0 under memory guard) | ~2969-3146 | yes | direction-blind | fix (shadow) |
| Tie-break (dollar volume, factor count, sector clash) | ~3219-3311 | **no**: NameError on `current_positions` (read ~3231, assigned ~3336), caught and logged | — | fix |
| Rules composite: momentum 0.25, mean reversion 0.20, VRP 0.25, squeeze 0.15, volume 0.15 | `deep_score` ~1259-1266; strategy funcs in `strategies/` | yes | §6.9 double counting with ML; squeeze counted 3-4×; VRP raises a stock-buy score then diverts to options | fix (decide rules vs ML per input; shadow) |
| EWMA-premium bonus (+8/+5) | `deep_score` ~1192-1220 | yes | double counts `vrp`/`ewma_vol` | remove (shadow) |
| Macro adjustments: VIX regime ±5/−10, sector ±8, risk on/off −5/+3 | `deep_score` ~1213-1241 | yes | double counts ML regime features (§6.9) | fix (shadow) |
| News adjustment +10/+5/−6/−12 | ~1244-1256 | yes | unvalidated (§9.4) | remove (shadow) |
| ADX +6/+3/−4, OBV ±5, relative strength +6 | ~1269-1281, analyze edge factors | yes | ADX duplicates ML `adx` | keep pending ablation |
| Intel adjustment ±15 (trap ≤ −10, sell-the-news −5) + Finnhub MSPR ±5, analyst +3 | ~1295-1340 | yes | unvalidated; insider counted twice | remove (shadow) |
| Alt adjustment ±20 | ~1343-1371 | yes | unvalidated; includes global terms (GDELT, FRED) | remove (shadow) |
| Social adjustment ±15 (halved at low confidence) | ~1374-1400 | yes | unvalidated; 35% Trends slot is 0 | remove (shadow) |
| yfinance fundamentals about −18..+16 | ~1403-1446 | yes | unvalidated | remove (shadow) |
| "AI recommends" +8 if `"BUY" in action` | ~1449-1453 | yes, **direction-blind**: analyze's bearish "BUY PUTS" raises a long score; "+6 SELL OPTIONS" branch never matches ("SELL PREMIUM") | §6.9 | fix |
| Contrarian flags +10/+8/+4, "Squeeze Watch" +10 | ~1455-1470 | yes | squeeze multi-count | remove (shadow) |
| Combined clamp 0-100 → `rules_only_score` | ~1476 | yes | — | keep |
| Combined data adjustments (intel+alt+social+news+fundamentals) | above | yes: can move a score by ~±55 points on 0-100 | audit §9.4: none passed gate 2; may explain inverted score quintiles (§7.3) | remove (shadow) as a block |

### 2.2 ML model and blend

| Rule / model | Where | Executes? | Conflicts | Action |
|---|---|---|---|---|
| LightGBM global model + isotonic calibrator | `ml_model_v2._train_single_lgbm` ~1197, `_train_model_impl` ~1287, `ml_score` ~1573 | yes (28-40% of final score) | validation not out-of-sample (ticker-major rows, early stopping on test fold; audit §5.5); trains only on shadow rows with +2%/−4%/10d labels that match no live exit (§5.6, KB #48); 58% accuracy is not evidence | fix (Phase 4) |
| Regime models (bull/bear/neutral) | `_train_model_impl` | partially: only "neutral" can train (all shadow rows tagged neutral) | §5.6 item 3 | fix |
| GB fallback if prob spread < 0.08 | `_train_single_lgbm` | rarely; chosen on test output | selection bias | fix |
| Rules fallback if model > 24 h old | `ml_score`, `ML_MAX_AGE_HOURS` | yes when retrain fails | retrain is SIGKILLed at 10 s (D1) | fix (Phase 3) |
| Blend `final = rules·rw + ml·mw` (0.6/0.4 default, ~0.72/0.28 live) | `deep_score` ~1805-1836 | yes | weights never fit for the overlap | fix after Phase 4 |
| ML-uncertain branch: \|ml−50\| < 8 → ml=50 at full weight | `deep_score` ~1825-1832 | yes: pulls a 70 rules score to ~62, below MIN_SCORE | N21 | fix (drop ML term instead of blending 50) |
| `ml_toggle.json` "enabled" | `/api/ml/status` | no (cosmetic; not read by `ml_score`) | — | fix or remove |
| `meta_score` / `train_meta_model` | `ml_model_v2` | no callers | — | remove |
| Options ML (`options_ml_score`) | `ml_model_v2` ~2056, `options_scanner` ~834 | yes in scanner; trained on synthetic VXX/SPY data, only via `bot_engine.py train` | — | fix or remove (validate) |
| `strategyWeights` "LEARN" loop | `server/bot.ts` ~859, `adjustStrategyWeights` ~1010 | no: written and audited, never read; audit text misleads | — | remove |

### 2.3 Side and instrument selection

| Rule / model | Where | Executes? | Conflicts | Action |
|---|---|---|---|---|
| Side: score > 65 & VRP > 5 → "SELL OPTIONS"; score > 65 & RSI > 70 → "SELL"; else BUY | `deep_score` ~1852-1875 | yes | N2, N3, N4 | fix |
| Sell/short (non-options) rewritten to BUY | `_scan_market_inner` ~3602-3604 | yes: RSI-overbought "SELL" is **bought** | N3 | fix (drop, don't flip) |
| Morning-queue short block skips `trade_type=="options"` | `server/bot.ts` `executeMorningQueue` ~4404 | yes: an off-hours "SELL OPTIONS" can POST a **stock sell (naked short)** and monitor it as long (inferred; check Alpaca history) | N2 | fix (Phase 2, top) |
| `instrument_selector.select_instrument` (stock vs 2x ETF vs options) | `instrument_selector.py` ~1277-1530 | partially: choice used, its sizing and stop config discarded | N4, N5, N6 | fix |
| Options eligibility (score ≥ 65, options < 16%, RTH) | `instrument_selector` ~1430, `_score_options` ~750 | yes | — | keep |
| Sell-premium trigger (SELL OPTIONS or VRP > 5 or IVR > 60) → CSP / bull put spread | `_score_options` ~773-856 | yes, then re-decided in Node | N4 | fix |
| QQQ Monday iron condor | `instrument_selector` ~828 | yes as a choice | — | keep (validate) |
| Covered call choice | `instrument_selector` ~834 | no: held tickers skipped earlier (~3425); wrong regime field | N4 | remove or fix |
| Buy calls/puts, debit spreads | `instrument_selector` ~861-888 | disabled in selector (but `bear_put_spread` reachable via N4) | N4 | keep disabled; close the N4 hole |
| Second options decision `should_use_options` (VRP > 2; side=sell → `bear_put_spread`) | `options_execution.should_use_options` ~251 via `evaluate_and_execute` ~2548 | yes: overrides the selector; declined options trades fall back to **stock BUY** counted as options slots (bypass stock-slot cap) | N4 | fix (one decision point) |
| Covered-call sweep | `bot_engine` ~3835-3930 | no: always OPTIONS-SKIP | N4 | remove or fix |
| Scanner-direct options (`regime_at_entry=="OPTIONS_SCANNER"`) | `server/bot.ts` `executeTrades` ~5977-6053 | yes; `select_contract` called with no `trade` | N4, N8 | fix |
| 2x ETF substitution | `instrument_selector._score_etf` ~546; Node buy `server/bot.ts` ~5907-5970 | yes, but buys `floor(shares/2)` of the ETF at the **underlying's** price; TNA/FAS are 3x though 3x is blocked; `ETF_MAX_HOLD_DAYS=5` has no exit; PANIC sub-rule reads a missing field | N6, N21 | fix (Phase 2) |
| Leveraged/inverse exclusion | `csp_universe` ~273; `options_scanner` ~1403 | yes for options only | N6 | keep; extend to stocks |
| Intraday stock shorts | `bot_engine` ~4036 | disabled (EOD cover still runs) | — | keep disabled |

### 2.4 Entry gates

| Rule / model | Where | Executes? | Conflicts | Action |
|---|---|---|---|---|
| Held-ticker skip | `_scan_market_inner` ~3425 | yes; stock ticker vs OCC symbols, so cannot see an existing short put | N19 | fix |
| Regime-aware stop cooldown (1.5-4 h) | ~3447-3480; file written by Node ~7713 and Python ~2362 | yes | Python writes cooldown without an order | fix (one writer) |
| Regime gate: no stock longs in BEAR/PANIC | ~3483 | yes | N10 (regime computed differently from slots) | keep; fix N10 |
| Adaptive MIN_SCORE 63/63/67/75 | ~3490-3507; `system_config.get_adaptive_params` ~441 | partially: Node floor 65 overrides | N12 | fix (one threshold) |
| Stock slots (`MAX_POSITIONS` adaptive; size tier overrides CAUTION 4→8; floor ETFs count) | ~3367-3410; `system_config` ~567 | yes | N10, N11 | fix |
| Sector / beta cap (`check_sector_correlation`) | `bot_engine` ~1962 | yes; fail-open | §6.9 (sector modelled 4×) | keep; consolidate |
| Dollar volume ≥ $50M (hard-coded) | ~3557 | yes | not in `system_config` | fix (move to config) |
| Blocked tickers + UBER | ~3570 | yes; bypassed by ETF substitution | N6 | fix |
| Extreme mover > +50% → overnight watchlist | ~3581 | partially: overnight research unreachable (D2) | D2 | fix |
| Spread > 0.5% | ~3610 | yes; fail-open; `MAX_SPREAD_PCT` config unread | — | keep; read config |
| Sizing block (halt, negative Kelly, 0 shares) | ~3672 | yes | — | keep |
| Volume-share cap (3% at $100k) | ~3731 | yes | — | keep |
| Kill-switch entry suppression | ~4128-4209, `suppress_entries_on_kill` ~2523 | yes (reads `killed` only; `blocks_entries`/`tier_blocks` ignored) | §6.6 | fix (honour the fields) |
| Node: options slot cap 6 | `server/bot.ts` `executeTrades` | yes; Python uses 8 and counts QQQ puts | N11 | fix (one constant) |
| Node: score < `minScoreThreshold` (≥ 65) | ~5795 | yes; rejects all 60-64.9 including scanner options admitted at 60 | N12 | fix |
| Node: daily ticker block (3 stops) | ~5803 | yes | — | keep |
| Node: size ceiling 20% × diag multiplier | ~5810 | yes | — | keep |
| Node: active heat cap 30% (excludes floor/leg) | ~5818 | yes | — | keep |
| Node: sector > 40% (`check_correlation_pre_trade`) | ~5830; `risk_kill_switch` ~577 | yes; fail-open | §6.9 | keep |
| Node: halt / ≥ 10 attempts / duplicate order / buying power + `replaceIfBetter` | ~5848-5893 | partially: attempt counts never reset except on restart (D2) | D2, KB #46 (`replaceIfBetter` 204 parse) | fix |
| Node: daily loss ≤ −3% | ~5061-5072 | yes, but after Python already placed floor/leg/options orders this scan | N9 | fix (scope) |
| Node: weekly loss −8%/−4% | `diagnostics.check_weekly_loss` ~38 | no: called with 1-row history | — | fix or remove |
| Off-hours deferral → morning queue | ~5112 | yes | N2 | keep; fix queue |
| Morning queue: no score/regime/heat/sector/earnings/options check; slots = all positions | `executeMorningQueue` ~4333-4520 | yes | N2, N11 | fix (re-run Node gates) |
| Stream priority queue | ~4815 | no | — | remove or wire (shadow) |

### 2.5 Sizing

| Rule / model | Where | Executes? | Conflicts | Action |
|---|---|---|---|---|
| Third-Kelly per bucket, clamp 1-8% | `position_sizing._kelly_fraction` ~153, `calculate_position` ~878 | yes; inputs from broken feedback (86% orphans) | §7.2 | fix inputs |
| Scalars: vol × confidence × regime × earnings × time-of-day × heat × liquidity | `calculate_position` ~967-993 | partially: regime scalar always 1.0 (`macro["regime"]` missing) | §6.5 | fix |
| Drawdown scalar 0.8/0.6/0.4 | ~995-1026 | no: wrong key `peak_equity` vs Node `equityPeak` | N13 | fix (Phase 2) |
| Stress multiplier | ~1028 | yes | unvalidated | keep (log) |
| Absolute clamp 8% per stock | `ABSOLUTE_MAX_POSITION_PCT` ~42 | yes; regime `MAX_POSITION_PCT` never applied to stocks | — | keep; delete dead config |
| Sizer entry stop / TP (2.5×ATR, TP 6-18%) | ~1066-1115 | no (computed, shipped on trade, unused) | N5 | remove |
| Options size (csp Kelly × scalars × 0.5, 1-8%) | `options_execution._dynamic_options_size` ~177 | partially: runs on default score 70 with no trade context | N8 | fix |
| CSP budget: max(size, 2%), **stretch to 20% of equity**, qty ≤ 2 | `_select_sell_put` ~897-1060 | yes | N8 (vs 8% cap; three affordability ceilings) | fix (Phase 2/6) |
| T1 VXX/DD size scalars | `tiered_strategy` ~486-500 | no: `size_pct` never passed to `select_contract` | N8 | fix |
| Scanner base size 6/7/8% | `options_scanner` ~1873 | no: Node passes no trade | N8 | fix |
| ETF size | Node `floor(shares/2)` | yes, wrong units | N6 | fix |
| Node ceilings 20% / 30% / 8 positions | `server/bot.ts` ~361-364 | yes | D7 (status API misreports) | keep |

### 2.6 Exits: the six stock exit models (N5, KB #48)

| Rule / model | Where | Executes? | Conflicts | Action |
|---|---|---|---|---|
| (1) **Live**: phase-1 stop clamp(2×ATR, 1.5-8%), scale-outs 1/3 at 1R and 3R/2R/1.5R, trails, breakeven, TP, 7-day flat time stop | `server/bot.ts` `checkPositionOnTick` ~7268 | **yes, the only stock exit executor** | KB #47 (adopts phantom Python scale state via max/min) | keep as the one model |
| (2) Python `manage_positions` (60-min hold, 4-day grace, −20% floor, regime scalar, 10% clamp) | `bot_engine.py` ~2045-2420 | no orders; but writes stop state and **phantom scale-outs** Node adopts | KB #47 | fix (stop writing scale state) then remove rules |
| (3) `system_config` exit keys (`ATR_STOP_MULTIPLIER`, `MIN/MAX_STOP_PCT`, `ATR_TP_MULTIPLIER`, `MIN/MAX_TP_PCT`, `TIME_STOP_DAYS`) | `system_config.py` ~268-274 | no readers except legacy backtest | KB #48 | remove |
| (4) `backtest_v2` legacy exits (default) | `backtest_v2.py:169` `DEFAULT_EXIT_MODEL="legacy"` | yes for PROMOTION RULE 3 backtests: they score exits the bot does not run | KB #48 | fix (default to `live`) |
| (5) Sizer stop/TP | `position_sizing` ~1066-1115 | no | N5 | remove |
| (6) Selector `_build_stop_config` | `instrument_selector` ~1165 | no | N5 | remove |
| Mirrors: `live_exit_policy.py`, `shadow_portfolio` `XA_*`, `shared/positionLevels.ts` (draws model 2 on the Bot page) | — | display / labelling | N5 | consolidate to one mirror of (1) |
| POS-KILL stock ≤ −25%; option backstop short ≤ −150%, long ≤ −75% | `server/bot.ts` `syncMonitoredPositions` ~6979-7101 | yes | KB #35 guard present | keep |
| POS-WARN ≤ −15% | ~6980 | audit only | — | keep |
| TIME-EXIT > 14 d and −5..+2% | ~6943-6972 | probably no (reads `created_at`, absent on Alpaca positions) | KB #48 | fix or remove |
| Floor/leg exemption (QQQ, SVXY, SPY, SMH, KWEB, VXUS, GLD skip all stops incl. POS-KILL) | `FLOOR_AND_LEG_TICKERS` ~409 | yes | N14 | fix (SVXY needs a stop) |
| Circuit breaker (3 stops → halt/1 h/ticker block) | ~7726-7770 | partially: day-halt branch uses `dailyPnL` only updated by a UI route | N15 | fix |
| Covered-call unwind before sale | ~7378-7417 | yes (blocks the exit if buy-back fails) | — | keep |

### 2.7 Options entries and exits

| Rule / model | Where | Executes? | Conflicts | Action |
|---|---|---|---|---|
| CSP universe layer 1 (price $10..equity×25%/95, vol ≥ 1M, $vol ≥ $20M, no leveraged) | `csp_universe._layer1_hard_gates` ~203 | yes | N8 (one of three affordability ceilings) | fix |
| CSP universe layer 2 (IVR .25, VRP .20, liquidity .15, skew .10, earnings .10, history .10, stability .10; earnings ≤ 7 d or lookup failure excluded) | `_layer2_score` ~618 | yes; IVR may be HV fallback; only top 150 fully scored | KB #45, N7 | fix (validate composite) |
| Options scanner setups: earnings IV-crush condor, VXX panic put sale, high-IV premium sale | `options_scanner.get_options_trades` ~1771, `_setup_*` ~720-1297 | yes; half-day / long-weekend skip | N7 (earnings condor vs CSP earnings exclusion; executes at ≥ 30 DTE not the priced `days+14`) | fix |
| CSP normal-market setup | `_setup_csp_normal_market` ~1124 | yes | — | keep (validate) |
| Entry DTE floor 30 (CSP and multi-leg) | `options_execution` ~105-148, ~432 | yes | F3 fixed; N16 same class | keep |
| Contract liquidity: OI ≥ 200, vol ≥ 10, spread ≤ 10% | `_is_liquid` ~719 | yes, on the OI proxy | §2.4 | fix (label / real OI) |
| Off-hours tier defer (F6) | `server/tierDispatchGate.ts` | yes for Node dispatch only | N17 | keep; extend to Python |
| Options slot cap at dispatch (6) | `server/bot.ts` ~5146-5226 | yes | KB #30 (3 recurrences) | keep |
| CSP re-submit brake (F7 + KB #46) | `server/cspSweepCooldown.ts` | yes | KB #46 partially fixed (cancel-all on deploy, fail-open orders fetch, no `openOrders` rehydrate) | fix |
| Options exits Check 0-7: 60-min hold, ≤ 5 DTE close, \|Δ\| > 0.8 roll/close, 21 DTE roll/close, 50% profit, 2× credit / −50% loss, gamma > 0.08 & DTE ≤ 30 close, delta drift warn | `options_manager.manage_options_positions` ~855 | yes (Python POSTs) | N16 (gamma not price-normalised, applies to longs, can close fresh CSPs); N17 (no hours/weekday gate; state dropped even if close never fills) | fix (Phase 6 / Phase 2 for N17) |
| Strategy-group exits (condor 50% of max loss, VRP-flip) | `_manage_strategy_group` ~571 | yes | — | keep |
| T1 exit metadata (`take_profit_pct`, `stop_loss_mult`, `force_close_dte`) | `tiered_strategy` ~185 | no | — | remove |

### 2.8 Floors, third leg, convexity

| Rule / model | Where | Executes? | Conflicts | Action |
|---|---|---|---|---|
| Floor basket (QQQ .50, SMH .15, KWEB .10, VXUS .15, cash .10) at BULL 70% / NEUTRAL 90% / CAUTION 35% / BEAR-PANIC 0; 3% drift rebalance; 200d and death-cross caps | `bot_engine._manage_spy_floor` ~5170 | yes (Python POSTs); the only positive sleeve | N18 (BEAR/PANIC sells only QQQ; SMH/KWEB/VXUS orphaned and push invested % over caps → master kill), N17 (hour-only RTH check, weekends), N19 | fix (Phase 2) |
| Floor regime dwell (F2) | `_resolve_floor_regime` ~5047 | yes | — | keep |
| Defensive GLD floor 8/8/15/30/40% | `_manage_defensive_floor` ~5592 | yes | N17 | keep; add hours gate |
| Third leg: SVXY up to 40% of equity when VXX 1.05-1.25 and falling | `_run_third_leg` ~4440 (`LEG3_VRP_PCT` 0.40, `system_config.py:216`) | yes; no stop, no POS-KILL, no hours gate, no dwell | N14, N17, N20 | fix (Phase 2: size cap + stop) |
| Third-leg exit in NEUTRAL/BULL (SVXY, ITA, XOM, LMT) | ~4555 | yes; no exit during a vol spike | N14, N20 | fix |
| QQQ put convexity overlay | `_run_convexity_overlay` ~4636; `CONVEXITY_OVERLAY.enabled=False` (`system_config.py:242`) | no (disabled) | N19 | keep disabled |

### 2.9 Tier engine (T1-T4)

| Rule / model | Where | Executes? | Conflicts | Action |
|---|---|---|---|---|
| Tier context regime | `bot_engine` ~4140 | always "NEUTRAL" (`_regime_ctx` lives in `deep_score`) | N1 | fix |
| Master kill: DD −20%, daily −3%, invested ≥ regime cap (95/95/60/50/30%) incl. floor | `tiered_strategy.master_kill_switch` ~279 | yes; kills all tiers incl. T1 | KB #20 (premise wrong per N1), N18 | fix |
| T1 CSP core: ranked `csp_universe`, OCC de-dupe, affordability 8%/95, max 3/scan, 6 slots | `tier1_csp_core` ~466 → Node SELL_CSP dispatch | yes | N8 (size scalars, DTE/delta/IVR metadata ignored), N19, half-day skip not applied | fix |
| T1 IV-rank floor 40 | `T1_MIN_IV_RANK` ~181 | no (metadata only) | — | fix or remove |
| T2 leverage multiplier | `tier2_leverage_multiplier` ~604 | no (`size_pct` never passed) | N8 | remove or wire (shadow) |
| T3 trend long SPY/QQQ | `tier3_trend_capture` ~682 | no (requires BULL; gets NEUTRAL); `exit_on_ma_break` not implemented; would clash with floor | N1, N19 | keep off; fix N1 before any revival (shadow) |
| T4 SPY tail hedge | `tier4_tail_hedge` ~754; `TIERS_ENABLED[4]=False` | no (disabled) | N19 | keep disabled |
| Exposure cap | `_enforce_exposure_cap` ~800 | yes | — | keep |
| `get_regime_caps` (forces `spy_below_200_days=0`, `markov_state=1`) | `tiered_strategy` ~103 | yes | N10 | fix |

### 2.10 Regime detectors

| Rule / model | Where | Executes? | Conflicts | Action |
|---|---|---|---|---|
| Canonical 5-level (PANIC VXX ≥ 1.40; BEAR ≥ 1.15 or SPY/MA50 < 0.94 or below 200d; CAUTION ≥ 1.05; BULL ≤ 0.95 & above 200d; else NEUTRAL) | `regime_util.classify_regime_5level` ~74 via `system_config.get_market_regime` ~386 | yes (scan gate) | — | keep as the one classifier |
| Legacy fallback (unread `REGIME_*` keys) | `system_config` ~414 | only on import failure | — | remove |
| Inline fallback (PANIC 1.30) | `bot_engine` ~3467 | only on exception | stale threshold | fix |
| `get_adaptive_params` regime (no 200d inputs) | `system_config` ~441 | yes (slots, MIN_SCORE) | N10 | fix |
| Tier caps regime | `get_regime_caps` | yes | N10 | fix |
| Tier context regime | `bot_engine` ~4140 | constant NEUTRAL | N1 | fix |
| Markov state | `markov_regime.get_regime` | yes (ML, size multiplier) | — | keep |
| Node exit regime class (current regime) | `server/bot.ts` ~7314 | yes (scale-out 2, trail width) | — | keep |
| Instrument market IVR from VXX | `instrument_selector` ~1358 | yes | — | keep |
| Sizing regime | `position_sizing` ~1073 | constant 1.0 | §6.5 | fix |
| Kill-switch VXX regimes (≥ 1.40 kills T2-T4; resume < 1.20) | `risk_kill_switch` ~64, ~271 | yes | — | keep |
| Shadow labeler regime (no 200d) | `shadow_portfolio` | labelling only | 86% NEUTRAL_BULL | fix (Phase 5) |

### 2.11 Halts and kill switches (eight mechanisms, four peak stores)

| Rule / model | Where | Executes? | Conflicts | Action |
|---|---|---|---|---|
| Node drawdown kill −10% (cancel all, latch) | `server/bot.ts` `checkDrawdownKillSwitch` ~4265 | yes; **also stops every exit** (WS, sync, POS-KILL, Tier 2 options exits) | N15 | fix (exits keep running while killed) |
| Liquidate on kill (−25%, env flag) | same ~4294 | only if `VOLTRADE_LIQUIDATE_ON_KILL=true` | — | human decision |
| Paper auto-resume | `server/killSwitchAutoResume.ts` | yes | — | keep |
| Python DD halt 18% (confirm ≥ 9% after 60 s) | `bot_engine.update_equity_peak` ~584, gate ~2832 | yes; returns before options exits, floors, GLD rotation | N9 | fix (gate entries only) |
| `risk_kill_switch.check_kill_switches` (manual, DD −20%, 5 losses/24 h) | `risk_kill_switch.py` ~154 (FROZEN mechanism) | partially: `tier_blocks` / `blocks_entries` ignored by caller | §6.6 | fix caller only |
| Tier master kill | see 2.9 | yes | KB #20 | fix |
| Node daily loss −3% | ~5061 | yes, after Python orders | N9 | fix |
| Node circuit breaker (3 stops) | ~7743 | yes; skips Tier 1, which suspends the −10% check | N15 | fix |
| `check_position_risk` −25% | `risk_kill_switch` ~509 | no caller (dangerous if wired) | F4 | keep unwired |
| Weekly loss −8% | `diagnostics` ~38 | no | — | fix or remove |
| Status/RULES display (5% daily, 15% stop, 25% TP) | `/api/bot/status` ~2578; startup RULES audit | display only, wrong | D7 | fix |
| Kill thresholds counterfactual logging | — | not logged | §9.2 | fix (Phase 5) |

### 2.12 Learning and measurement loops that feed the brain

| Rule / model | Where | Executes? | Conflicts | Action |
|---|---|---|---|---|
| Hourly retrain (`python3 ml_retrain_safe.py`, asks 900 s) | `server/bot.ts` `tier3Strategic` ~6293 | broken: `isTrivialPython` (~244) → SIGKILL at 10 s; the orphan child probably finishes | D1 | fix (Phase 3) |
| 4am retrain, daily reset, nightly backup | `runOvernightResearch` 4am branch ~6747 | no: unreachable | D2 | fix (Phase 3) |
| Nightly shadow backfill (`backfill_outcomes(3000)`) | ~4630 | broken: 10 s kill | D1 | fix |
| Exit-aligned shadow backfill | `checkShadowBackfillExitAligned` | yes (command lengthened on purpose) | — | keep |
| `track_fill` slippage | `ml_model_v2.track_fill` ~2355; Node haircut ~6213 | broken: fill price is synthetic; `abs()` counts favourable fills as cost | §7.1 | fix (Phase 5, [RULE-REVIEW]) |
| Shadow candidate log | `shadow_portfolio.log_candidate` ~270 | yes; one record per scan (duplicates), 20k cap rotating | §7.3 | fix (dedupe; deploy `shadow_bands_dedup`) |
| Daily equity curve | `recordDailyEquity` ~1037 (Tier 3, UTC date) | yes; weekend rows with stale pnl | §7.1 | fix |

---

## 3. Summary

### Working correctly
- Live stock exit engine `checkPositionOnTick` is the single executor and matches its backtest mirror (`server/bot.ts` ~7268; `live_exit_policy.py`).
- POS-KILL with option backstops and duplicate guard (F4, KB #35; `server/bot.ts` ~6979-7101).
- Canonical 5-level regime classifier (`regime_util.py` ~74).
- Floor regime dwell F2 (`bot_engine._resolve_floor_regime` ~5047); floor basket is the only positive sleeve (+$2.75k, audit §7.3).
- CSP entry DTE floor and multi-leg DTE floor (F3; `options_execution.py` ~105-148).
- Off-hours tier dispatch defer F6 (`server/tierDispatchGate.ts`) and options slot cap at dispatch.
- Node entry ceilings: 20% size, 30% heat, sector 40%, attempt and duplicate guards (`server/bot.ts` ~5795-5893).
- Regime ML inputs `vxx_ratio`/`spy_vs_ma50`/`markov_state`/`regime_score` aligned since 2026-10-16 (`regime_inputs.py`).
- Exit-aligned shadow backfill and labeler (`shadow_portfolio.backfill_exit_aligned` ~1808).
- Paper kill-switch auto-resume (`server/killSwitchAutoResume.ts`).

### Broken
- ML validation is not out-of-sample; early stopping on the test fold; earnings-surprise lookahead (audit §5.5; `ml_model_v2.py` ~711, ~1225, ~297).
- 9 of 34 ML features constant at serve; 2 broken; 12 mis-defined (§1.7 above; `bot_engine.py` ~1731-1782).
- Model trains only on shadow rows with a +2%/−4%/10d label no live exit uses (audit §5.6; KB #48).
- Retrain, shadow backfill and insight jobs SIGKILLed at 10 s (D1; `server/bot.ts` `isTrivialPython` ~244).
- 4am daily reset/retrain/backup unreachable; attempt counts and morning-queue flag only reset on restart (D2; `server/bot.ts` ~5691, ~6747, ~8232).
- Off-hours "SELL OPTIONS" can become a naked stock short via the morning queue (N2; `server/bot.ts` ~4404, inferred).
- RSI-overbought "SELL" is executed as a BUY (N3; `bot_engine.py` ~1872, ~3602).
- Second options decision turns sells into `bear_put_spread` and failed options into uncapped stock buys (N4; `options_execution.should_use_options` ~251).
- ETF substitution mis-sized, mis-priced, 3x allowed, no max hold (N6; `server/bot.ts` ~5907).
- Tier/scanner option sizing never reaches `select_contract`; CSP collateral stretches to 20% (N8).
- Tier context regime always NEUTRAL, so T3 is dead and KB #20's premise is wrong (N1; `bot_engine.py` ~4140).
- Tie-break never runs (NameError; `bot_engine.py` ~3231 vs ~3336).
- Drawdown sizing scalar never applies (N13; `position_sizing.py:1004`).
- SVXY up to 40% of equity with no stop (N14; `system_config.py:216`, `server/bot.ts` ~409).
- Node drawdown kill also stops all exits (N15).
- Python DD halt freezes de-risking; Node daily halt runs after Python orders (N9).
- Python order paths have no hours/weekday gate; options state dropped on unfilled closes (N17).
- BEAR/PANIC floor exit sells only QQQ (N18).
- Fill slippage is synthetic; 86% of feedback records are orphan exits (audit §7.1-7.2).
- Options chain archive: day label shifted, chains truncated, no OI/gamma/volume/spot; live OI is a quote-size proxy (audit §2.4).
- Backtests default to the dead "legacy" exit model (`backtest_v2.py:169`).
- `/api/orders/open` shape mismatch hides working orders in the cockpit (D4).

### Should change (works as coded, but is the wrong design)
- Remove the unvalidated intel/alt/social/news/fundamentals adjustments (up to ~±55 points) from the live score; shadow-log them through the ladder (audit §9.4; `bot_engine.py` ~1244-1446).
- Stop double counting: each input in rules **or** ML, not both; squeeze counted 3-4× (audit §6.9).
- Make the "AI recommends" boost direction-aware (`bot_engine.py` ~1451).
- One exit model, one regime call per scan, one score threshold, one options decision point, one peak store, one slot constant (N5, N10, N12, N4, N13, N11).
- Make the quick-score rank direction-aware and fix cross-sectional rank (`bot_engine.py` ~2969, ~1583).
- Gamma exit should be price-normalised and side-aware (N16; `options_manager.py` ~1282).
- Resolve the earnings contradiction (CSP excludes ≤ 7 d; scanner sells condors 0-7 d) (N7).
- Default backtests to `exit_model="live"`; add an options simulator before any options promotion (KB #48; audit §10.3).
- Add live Sharpe/Sortino/beta and rolling-vs-SPY, regime-split, capital-weighted stats (audit §7.2).
- Delete dead config (36 of 117 `system_config` keys unread), dead modules (`ml_model.py`, `strategyWeights`, `meta_score`, covered-call sweep) (audit §9.1-9.3).
- Fix the Polygon env-var split (`POLYGON_KEY` vs `POLYGON_API_KEY`) (audit §2.0).

---

## 4. Fix order (mapped to master-plan phases)

Ordered within each phase by expected damage avoided. "Live" = bug/reliability fix ships live; "shadow" = trading-behaviour change ships shadow-logged first.

**Phase 2: execution bugs (what is ordered differs from what was decided)**
1. N2 morning-queue naked stock short on "SELL OPTIONS" — live (check Alpaca history first, audit §6.10 item 1).
2. N14 SVXY: cap leg size and give it a stop / POS-KILL coverage — live (risk bug); size change logged per RULE REVIEW.
3. N15 drawdown kill must not stop exits; circuit breaker must not skip the drawdown check — live.
4. N9 Python DD halt gates entries only; Node daily halt scope — live.
5. N17 hours/weekday/holiday gate on Python order paths; keep options state until the close fills — live.
6. N18 BEAR/PANIC floor exit sells every basket member — live.
7. N3 RSI "SELL" → drop instead of BUY — live (bug: inverted direction).
8. N4 single options decision point; failed options do not fall back to uncapped stock — live.
9. N6 ETF substitution sizing/pricing, 3x block, max hold — live.
10. N8 pass `size_pct`/trade context to `select_contract`; remove the 20% CSP stretch — live (cap is a bug vs `MAX_OPTIONS_PCT`).
11. N13 drawdown sizing scalar key; N1 tier context regime; tie-break NameError; N11 slot counting; KB #47 phantom scale-out; KB #46 remaining items — live.
12. D4 orders panel shape; D7 status numbers — live.

**Phase 3: heartbeats (jobs that must run, and must be seen running)**
1. D1 `isTrivialPython` 10 s kill on retrain, shadow backfill, insight — live.
2. D2 4am daily reset / retrain / backup unreachable — live.
3. D5/D6 unregistered RPCs; D3 drawdown monitor reads float as object — live.
4. Weekly-loss check fed 1 row; circuit-breaker `dailyPnL` stale; TIME-EXIT reads absent fields — fix or remove.
5. Feed-status probe (SIP/OPRA entitlement), audit retention (~3 h), equity-curve weekend rows, stream queue dead — live.

**Phase 4: model**
1. Sort by date; purge by label window; early stop on a validation split, not the test fold; report AUC/Brier/logloss — [RULE-REVIEW].
2. Remove the `earnings_surprise` lookahead; fix or remove the 9 constant and 2 broken features; one definition per feature in train and serve (§1.7).
3. One label that matches the live exit (exit-aligned), deduplicated shadow rows, real regime tags.
4. ML-uncertain branch (N21); then re-fit blend weights; remove `ml_toggle`/`meta_score`/`strategyWeights` dead code.

**Phase 5: brain / validation**
1. Deploy the deduplicated shadow band probe; confirm or refute the inverted score quintiles (audit §7.3).
2. Move intel/alt/social/news/fundamentals adjustments to shadow; run each through ladder gate 2 (shadow).
3. Resolve §6.9 double counting and the direction-blind boosts (shadow).
4. Measurement: real fill slippage, live Sharpe/Sortino/beta, rolling-vs-SPY, regime-split, capital-weighted trade stats, counterfactual logging for kill thresholds — each its own [RULE-REVIEW].
5. One exit model (KB #48 step 3); backtests default to `live` exits; one regime classifier per scan (N10); one score threshold (N12).

**Phase 6: options and edges**
1. Chain archive: day-label fix, pagination/truncation, add gamma/volume/spot; label the OI proxy (audit §2.4, §10.2).
2. IV-rank fetch strike filter; one IV definition across analyze / vol_surface / scanner / HV rank (audit §9.5).
3. Gamma exit normalisation (N16); earnings rule contradiction and condor DTE (N7).
4. Options simulator for backtests; then review the options sleeve (−$9.2k realized) setup by setup.
5. Derived options features (GEX, P/C, skew, term, expected move) via datacore → gate 2 → shadow, per audit §10.3.

**Needs a human decision:** paid real-time SIP / OPRA / Databento history (audit §2.4, wishlist); whether the options sleeve keeps trading live while Phases 2-6 run; `VOLTRADE_LIQUIDATE_ON_KILL`; which Polygon env var prod sets.
