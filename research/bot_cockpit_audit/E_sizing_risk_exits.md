# AUDIT E: Sizing, Risk Limits / Kill Switches, Stops & Exits, Hedges / Floors

Ground truth as of repo HEAD `5a8ab6d` (2026-10-15). I read the code only. Nothing here is checked against the live account. Where I infer a runtime behavior from code paths rather than seeing it in live data, I label it **(code-path inference)**. Every claim cites `file:line`.

Vocabulary in `monospace` is the exact identifier, audit-type string or JSON key in the code. Use those names verbatim in the cockpit.

---

## 0. The facts that matter most (read first)

1. **Three different executors fire exits, and the Python "exit engine" is not one of them.**
   - `bot_engine.manage_positions()` computes stops, scale-outs, take-profit and time stops (`bot_engine.py:1964-2338`). `tier1Reflex()` runs it every ~45 s and throws the actions away. Its only use of them is an audit line: `POS-MONITOR-SYNC "bot_engine flagged X (type) — WS monitor should handle"` (`server/bot.ts:4210-4229`). Tier-2 shows the same `position_actions` only as UI "signals" (`server/bot.ts:4523-4532`).
   - The stock and ETF exits that actually fire are in Node: `checkPositionOnTick()` (`server/bot.ts:6720-7222`), driven by 1-minute IEX bars (`server/bot.ts:7286-7297`).
   - `syncMonitoredPositions()` fires two exits: `POS-KILL` and `TIME-EXIT` (`server/bot.ts:6345-6650`).
   - Option exits fire inside Python, in `options_manager.manage_options_positions()`. It submits its own orders (`options_manager.py:855-1343`).
   - Floors, the third leg and the convexity overlay submit their own orders, also inside Python (`bot_engine.py:4218-5600`).
2. **Scale-outs in thirds DO execute live**, from Node `checkPositionOnTick` (`WS-SCALE-OUT`, `server/bot.ts:6765-6950`). There are four problems with them:
   - No scale-out leg is ever written to `trade_feedback`. This is by design (`server/exitFill.ts:14-17`).
   - No leg's fill is confirmed. The order is registered without `pendingExitFill` (`server/bot.ts:6876-6886`).
   - The leg counters advance at order submission, not at fill.
   - **(code-path inference)** Python `manage_positions` writes *phantom* scale-outs into the shared `voltrade_stop_state.json`: it advances `scales_completed` and lowers `remaining_qty` without placing any order (`bot_engine.py:2083-2109`, `2181-2203`, `2334`). The Node monitor then adopts those values (`server/bot.ts:6579-6603`), so a real scale-out leg can be skipped while the position is booked as "scaled".
3. **Six independent halt mechanisms run off three different peak-equity files.**

   | Mechanism | Threshold | Peak file |
   |---|---|---|
   | Node drawdown kill | −10% | `voltrade_equity_peak.json` |
   | Python DD halt | −18% | `voltrade_portfolio_dd.json` |
   | `risk_kill_switch` | −20% | `voltrade_peak_equity.json` + its own state |
   | `tiered_strategy.master_kill_switch` | −20% | `voltrade_peak_equity.json` |
   | Node Tier-2 daily loss | −3% | — |
   | Node circuit breaker | — | — |

   The `risk_kill_switch` daily-loss check (−3%) and its VXX regime kill are warnings only: `blocks_entries` and `tier_blocks` have no consumer anywhere.
4. **When the Node kill switch is latched, nothing manages positions.** No WS exits, no `POS-KILL`, no Tier-2 scan, so no option exits, floor rotation or hedges. All open orders, exit orders included, are cancelled when it trips (`server/bot.ts:3943`, `6724`, `7227-7229`, `7602-7606`).
5. **When the Python DD halt fires, `scan_market()` returns early** (`bot_engine.py:2703-2716`). That freezes `options_manager`, both floors, the third leg and the convexity overlay. Node WS stock exits keep running.
6. **Sizing reality: the per-bucket Kelly is effectively hardcoded.**
   - Live `trade_feedback` records carry no `strategy` key, so live outcomes never reach the per-bucket stats (`position_sizing.py:342-354`; record shape at `ml_model_v2.py:2502-2523`).
   - With the shipped priors, the `"stocks"` bucket Kelly is ≤ 0, so `calculate_position` blocks every candidate routed to `"stocks"` (`position_sizing.py:192-208`, `278`, `950-954`). Computed: stocks = 0, vrp = 6.67%, etf = 4.91%, csp_options = 8% (capped).
   - In practice only candidates whose top factor score is `vrp_score`, or known leveraged-ETF tickers, get a size (`position_sizing.py:1289-1302`).
7. **Several sizing and risk inputs are dead.**
   - The drawdown sizing scalar never fires. Python reads key `peak_equity` from `voltrade_equity_peak.json`, but Node writes key `equityPeak` (`position_sizing.py:1000-1004` vs `server/bot.ts:604-628`).
   - The stress-index "crisis" multiplier of 0.0 is clamped back up to 1% (`position_sizing.py:1050`; `stress_index.py:225`).
   - The weekly loss limit never fires: it is called with a one-point history (`server/bot.ts:4628`; `diagnostics.py:44-45`).
   - The sector heat scalar never fires: Alpaca positions have no `sector` key (`position_sizing.py:559-566`).
   - `STOP_SCALAR_*`, `POSITION_HARD_STOP_PCT`, the 1-hour minimum hold and the 4-day short-hold grace exist only in the advisory Python path.
8. **Tier-1 and Tier-2 CSP `size_pct` is computed and then ignored.**
   - The dispatcher sends `size_pct` into the payload but calls `select_contract(ticker, strategy, 0, equity, cash_available=…)` without it (`server/bot.ts:4780-4800`).
   - The real CSP size comes from `_select_sell_put`: budget = `max(equity×_dynamic_options_size, 2% equity)`, capped by options buying power. It can stretch to 20% of equity for the smallest strike. Quantity is capped at **2 contracts** (`options_execution.py:922-1071`).
9. **`CONVEXITY_OVERLAY["enabled"]=False` is never read.**
   - `_run_convexity_overlay()` runs on every scan regardless of the flag (`system_config.py:242`; `bot_engine.py:3846-3850`, `4414-4715`).
   - **(likely)** Its orders use `side: "buy_to_open"` / `"sell_to_close"`. Those are not values of Alpaca's `side` enum (`buy|sell`), so they are probably rejected. The action is still logged with `order_id: "?"` (`bot_engine.py:4531`, `4657`, `4677`).
10. **Floor basket hole (code-path inference).** When the floor target becomes 0 (BEAR/PANIC), the basket branch is skipped because it requires `target_pct > 0`. The legacy branch then sells only `FLOOR_TICKER` (QQQ) (`bot_engine.py:5067`, `5178-5192`). SMH, KWEB and VXUS are left with no exit owner: they are excluded from every stop path.

---

## 1. SIZING

### 1.1 Stock / ETF scanner trades: `position_sizing.calculate_position()` (`position_sizing.py:878-1193`)

**Caller.** `bot_engine._scan_market_inner()` loops over `deep_scored` and calls `calculate_position(trade={**stock, "score": final_score, "side": side}, equity=portfolio_value, current_positions=…, macro=_macro)` (`bot_engine.py:3478-3497`). The trade passed in does **not** include the `instrument_selector` decision, because that decision is made just before sizing but not merged in (`bot_engine.py:3440-3475`). `size_portfolio()` (`position_sizing.py:1317-1406`) is **not used** by the scan.

**Pre-checks, in order.**
1. `check_halt_status(ticker)` (an Alpaca `/v2/assets` call) blocks with `"Stock is halted (status: …)"` (`position_sizing.py:919-921`).
2. `price <= 0` blocks with `"Invalid price"`.
3. `equity <= 0` blocks with `"No equity available"` (`923-927`).

**Formula.**
```
kelly_base = _kelly_fraction(bucket.win_rate, bucket.avg_win, bucket.avg_loss)       # 931-944
   f* = (WR·avg_win − (1−WR)·avg_loss)/avg_win ; ≤0 → 0 (BLOCK) ; f*/3 clamped [0.01, 0.08]  # 153-208
   avg_win ≤ 0 → 0 (BLOCK); avg_loss ≤ 0 → 0.005 (starter)                             # 192-195
   kelly_divisor default 3.0 (BASE_CONFIG KELLY_DIVISOR=3.0 is NOT read)               # 154, system_config.py:113
kelly_base ≤ 0 → _blocked("Negative expectancy for strategy '<bucket>' (WR=..); refusing to size")  # 950-954

final_pct = kelly_base × s_volatility × s_confidence × s_regime × s_earnings × s_time × s_heat × s_liquidity  # 986-993
final_pct ×= dd_scalar   (DEAD, see below)                                              # 998-1026
final_pct ×= stress_mult (only if < 1.0)                                                # 1031-1047
final_pct  = clamp(final_pct, 0.01, 0.08)  ← ABSOLUTE_MIN/MAX_POSITION_PCT              # 1050, 42-43
shares = floor(equity × final_pct / price); 0 → _blocked("Position too small …")         # 1054-1059
```

**Bucket (`_infer_strategy`, `position_sizing.py:1221-1310`).** Rules are applied in this order:
1. `options_strategy` contains csp, covered_call, iron_condor, bull_put_credit, short_straddle, … → `csp_options`.
2. `instrument == "etf"` or `chosen == "etf"` → `etf`. The scanner never sets either field at sizing time.
3. Ticker is in the `_KNOWN_LETFS` set (`1270-1285`) → `etf`.
4. Otherwise the dominant of the five factor scores decides: `vrp` → `vrp`, anything else → `stocks`.
5. No scores and `instrument == "options"` → `csp_options`.
6. Default → `stocks`.

**Bucket statistics (`_get_historical_stats`, `211-356`).**
- The hardcoded priors are used unless `trade_feedback` holds at least 10 *closed* records (numeric `pnl_pct`, `309-311`).
- Per-bucket overrides happen only for records whose `strategy` key equals the bucket and only when there are at least 5 of them (`342-354`). Live `track_fill` records never carry a `strategy` key (`ml_model_v2.py:2502-2523`). Only seeder records do (`seed_feedback_from_backtest.py:137`).
- Result: live outcomes never update per-bucket Kelly.

| Bucket | Prior (WR, avg_win, avg_loss) | Computed Kelly base |
|---|---|---|
| stocks | 0.49, 4.19, 4.25 | **0 → BLOCKED** |
| etf | 0.558, 3.82, 3.55 | 4.91% |
| vrp | 0.60, 2.0, 2.0 | 6.67% |
| csp_options | 0.716, 0.69, 0.46 | 8.00% (cap) |

Priors are at `position_sizing.py:267-288`. The code comment at `273-277` says the stocks prior "gives nearly-zero Kelly which … will round to a small starter size". **That comment is wrong: the value is −0.027, so the candidate is blocked.**

**Scalars (name → key in `scalars` → rule).**

- `volatility` = `_volatility_scalar(atr_pct, garch_rv)` (`401-430`).
  - Formula: `2.0 / max(daily_atr%, 0.5)`, multiplied by 0.80 when GARCH > 1.5×ATR, then clamped to [0.25, 1.20].
  - The input is `ewma_rv` (annualized ×100). It is normalized to a daily percentage by magnitude (`363-398`): ≤0.6 is treated as an annual fraction, ≤12 as a daily percentage, anything larger as an annual percentage divided by √252.
- `confidence` = `_confidence_scalar(score, ml_confidence)` (`437-462`).
  - Formula: `0.6 + (score−65)/35×0.55`, clamped to [0.5, 1.15].
  - Multiplied by 1.10 if `ml_confidence > 0.75`, by 0.75 if `ml_confidence < 0.45`.
  - Final clamp [0.4, 1.25].
- `regime` = `_regime_scalar(vix, vix_regime)` (`575-606`).

  | VIX | Scalar |
  |---|---|
  | < 15 | 1.10 |
  | 15–20 | 1.0 |
  | 20–25 | 0.80 |
  | 25–30 | 0.65 |
  | 30–40 | 0.50 |
  | ≥ 40 | 0.30 |

  The docstring at `579-583` lists different bands (20–30 = 0.75). That disagrees with the code.
- `earnings` = `_earnings_scalar(ticker)` (`694-741`).

  | Next earnings | Scalar |
  |---|---|
  | > 7 days, or already passed | 1.0 |
  | 5–7 days | 0.85 |
  | 2–4 days | 0.60 |
  | 0–1 days | 0.35 |
  | Lookup failed or date unparseable | 0.85 (fail-closed) |

  The source is yfinance, cached for 12 h.
- `time_of_day` = `_time_scalar()` (`748-776`). ET 9:30–10:00 = 0.80; 10:00–15:30 = 1.0; 15:30–16:00 = 0.70; anything else = 0.50.
- `portfolio_heat` = `_portfolio_heat_scalar` (`518-568`). It is `count_scalar × deploy_scalar × sector_scalar`.
  - `count_scalar = max(0.4, 1 − 0.10 × n_active)`. Only `FLOOR_TICKER` (QQQ) is excluded (`535-536`). **SMH, KWEB, VXUS and GLD count as active**, so a fully deployed floor costs about ×0.6 on every new trade (code-path inference).
  - `deploy_scalar`: deployed > 90% → 0.50; > 80% → 0.70; > 70% → 0.85.
  - `sector_scalar`: 0.75 / 0.50 for 1 / 2+ positions in the same sector. **It never fires** because Alpaca positions carry no `sector` key.
- `liquidity` = `_liquidity_scalar(volume, price, preliminary_value)` (`469-504`).
  - Volume impact: > 2% of dollar volume → 0.25; > 1% → 0.50; > 0.5% → 0.75.
  - Otherwise by share volume: < 500k → 0.50; < 1M → 0.70; < 5M → 0.85; else 1.0.
  - Missing data → 0.5.
- `drawdown_scalar` / `drawdown_pct` (`995-1026`). **DEAD.**
  - It reads `DATA_DIR/voltrade_equity_peak.json` and takes `.get("peak_equity", equity)`.
  - Node writes that file as `{equityPeak, savedAt}` (`server/bot.ts:604-628`), so `_peak == equity` and the scalar is always 1.0.
  - Designed bands: DD ≥ 5% → 0.8; ≥ 10% → 0.6; ≥ 15% → 0.4.
- `stress` / `stress_level` / `stress_value` (`1028-1047`; `stress_index.py:187-197`, `223-226`).

  | Stress index | Level | Multiplier |
  |---|---|---|
  | < 30 | calm | 1.0 |
  | ≥ 30 | elevated | 0.75 |
  | ≥ 55 | stressed | 0.5 |
  | ≥ 75 | crisis | 0.0 |

  **The crisis value of 0.0 is clamped up to 1% by `1050`**, so the documented `"halt_new"` action does not halt.
- `combined` = the final clamped `final_pct`.

**Entry stop and target computed by the sizer (`1064-1115`). None of these values is used by anything that executes (§2.8).**

| `trade_type` | Stop distance | Target distance |
|---|---|---|
| stock | `clamp(daily_ATR × 2.5 × STOP_SCALAR_<regime>, 5%, 12%)` | `clamp(daily_ATR × 4, 6%, 18%)` |
| etf | `clamp(daily_ATR × 1.5 × scalar, 4%, 9%)` | `clamp(daily_ATR × 4, 6%, 18%)` |
| other | `clamp(×1.5 × scalar, 1.5%, 8%)` | `clamp(×3, 4%, 12%)` |
| no volatility data | 6% | 10% |

- `STOP_SCALAR_*` values: BULL 0.90, NEUTRAL 1.00, CAUTION 1.10, BEAR 1.20, PANIC 0.70 (`system_config.py:377-381`).
- The sizer comment at `1093-1096` says the ETF target is ×2.5 (14% cap). The code applies ×4 / 18% to both stock and ETF; the ×2.5 value exists only in `instrument_selector._build_stop_config` (`instrument_selector.py:1206-1212`).

**Costs (`_estimate_costs`, `817-871`).**
- Commission: 0 per share for stock (`DEFAULT_COMMISSION_PER_SHARE`); $0.65 per contract for options.
- Slippage % per side, by share-volume share or liquidity: 0.30 / 0.15 / 0.08 / 0.02 / 0.04 / 0.06 / 0.10. Total cost counts the round trip.

**Output keys (`1173-1193`).**
- Top level: `ticker, shares, price, side, position_value, position_pct, stop_loss, take_profit, stop_distance_pct, tp_distance_pct, risk_per_share, risk_total, max_loss_pct, costs{commission_entry, commission_exit, estimated_slippage_pct, total_cost_pct, total_cost_dollars}, scalars{…as above…}, reasoning, blocked, block_reason, hist_stats{win_rate, avg_win, avg_loss, total_trades}`.
- `_blocked()` returns the same keys with zeros, plus `blocked: True` and `block_reason` (`1196-1218`).

### 1.2 What the scanner keeps and what Node enforces

**Scan-side gates around sizing (`bot_engine.py`).**
- Stock slots = adaptive `MAX_POSITIONS` − `num_positions` (`3214-3229`).
  - `num_positions` counts **all** equity positions, the five floor ETFs included (`3192-3196`).
  - In active regimes the size tier sets `MAX_POSITIONS` (`system_config.py:567-572`). The CAUTION value of 4 is therefore overridden by the tier value (`system_config.py:509` vs `572`).
  - PANIC, BEAR and NEUTRAL give 0 stock slots.
- Stop cooldown by regime: PANIC 4 h, BEAR 3 h, CAUTION 2.5 h, BULL 1.5 h, NEUTRAL 2 h (`3266-3298`). `STOP_COOLDOWN_SECONDS=7200` is never read.
- Hard block of stock longs in BEAR/PANIC (`3301-3302`).
- Score floor `MIN_SCORE` by regime: BULL 63, CAUTION 67, BEAR/PANIC 75 (`3305-3318`).
- Sector and beta correlation (`check_sector_correlation`, `1881-1961`):
  - Same sector count ≥ tier `max_sector_positions` (2 under $500k) → block.
  - New beta ≥ 1.5 when 3 or more existing high-beta names are held → block.
  - Average beta > 1.6 → block.
  - The beta map is hardcoded.
- Volume-share cap: `position_value / dollar_volume > MAX_VOLUME_SHARE_PCT` from the tier (5% / 4% / 3% / 2.5% …) → `VOL_SHARE_BLOCK` (`3519-3551`).

**Trade dict sent to Node (`bot_engine.py:3554-3597`).** Keys:
`action, side, trade_type, ticker, shares, price, score, reasons, recommendation, rec_reasoning, stop_loss, take_profit, position_value, sizing_reasoning, sizing_scalars, momentum_score, mean_reversion_score, vrp_score, squeeze_score, volume_score, ewma_rv, garch_rv, rsi, vrp, instrument, instrument_strategy, instrument_reasoning, instrument_ticker, instrument_scores, use_options, options_strategy, options_reasoning, options_edge_pct, rules_score, ml_score_raw, entry_features, entry_date, regime_at_entry`.

**Node `executeTrades()` gates (`server/bot.ts:5220-5705`), in order.**
1. Options slot cap: 6, counting open positions plus resting sell-to-open orders (`5258-5263`, `361`).
2. Stock slot cap `MAX_POSITIONS = 8`. This counts **every** `us_equity` position, floor included (`5226-5228`, `5264-5266`, `353`).
3. `trade.score < state.minScoreThreshold` → `SKIP`. The threshold defaults to 65 from `diagnostics.get_auto_fix_params` (`5267-5270`; `diagnostics.py:527`). This overrides the Python 63, and it skips options-scanner trades scored 60–64.
4. Already held, `_consumed`, or `dailyBlockedTickers` → skip.
5. `SIZE-CAP`: `position_value > equity × 0.20 × state.positionSizeMultiplier` (`5282-5286`). This is a **block, not a resize**. `positionSizeMultiplier` (0.5–1.0) comes from diagnostics (`4366`).
6. `HEAT-CAP`: non-floor `totalDeployed + position_value > equity × 0.30` → **`break`**, which drops all remaining trades in the batch (`5289-5292`, `352`).
7. `CORR-CAP`: `risk_kill_switch.check_correlation_pre_trade(ticker, positions, 0.40)`. The coarse 5-sector map (`risk_kill_switch.py:558-620`) assumes the new position is 5% of the book. It fails open.
8. `HALT-SKIP` (exchange halt).
9. `MAX-RETRY`: 10 attempts per ticker per day.
10. `DUP-SKIP` if an open order exists.
11. Buying power, falling back to `replaceIfBetter` (cancels an entry order that scores at least 10 points lower).

**Order sent.** `qty = floor(trade.shares)`. The order params are a DAY limit at ask + 0.1% during regular hours, or ask + 0.5% with `extended_hours` outside them (`server/orderParams.ts:36-62`). **`stop_loss`, `take_profit` and `sizing_scalars` are never sent or persisted.** The `TRADE` audit line carries only the first 150 characters of `sizing_reasoning` (`5599-5600`).

**ETF instrument path (`server/bot.ts:5370-5430`).**
- `etfShares = max(1, floor(trade.shares / 2))`. This halves the **share count of the underlying stock** and applies it to a differently priced ETF, so dollar exposure has no relation to the sized `position_value`.
- The entry `track_fill` uses `fillPrice: trade.price`, which is the **underlying's** price (`5411`).
- `addPositionToMonitor` also seeds `entryPrice` with the underlying's price (`5427`). **(code-path inference)** Until the next sync (45–60 s) corrects `entryPrice` from Alpaca's `avg_entry_price`, any WS bar would compute a nonsense P&L.

**Morning queue (`server/bot.ts:3960-4181`).**
- Slots count **all** positions.
- `position_value > equity × 0.20 × multiplier` → `MORNING-BLOCKED`.
- An overnight gap > 5% → `MORNING-GAP` skip; a gap > 2% → shares recomputed.

### 1.3 Caps: enforced vs inert

| Cap | Value | Where | Enforced? |
|---|---|---|---|
| Per-stock % | 8% (`ABSOLUTE_MAX_POSITION_PCT`) | `position_sizing.py:42,1050` | YES, by clamp |
| Per-stock floor | 1% (`ABSOLUTE_MIN_POSITION_PCT`) | `position_sizing.py:43` | YES (and overrides the stress halt) |
| Regime `MAX_POSITION_PCT` (BULL 0.15 / CAUTION 0.10 / BEAR 0.08 / PANIC 0.06, tier min) | — | `system_config.py:460-575` | **NOT read by calculate_position**. Only T3 uses it (`tiered_strategy.py:647`) |
| Tier `min_position_dollar` | $1k–$300k | `position_sizing.py:77-86` | **Not enforced anywhere** |
| Node per-trade ceiling | 20% × diagnostics multiplier | `server/bot.ts:351,5282` | YES (block) |
| Node active exposure | 30% of equity, floor excluded | `server/bot.ts:352,5289` | YES (break) |
| Tier-engine total exposure | `MAX_TOTAL_EXPOSURE`: PANIC 0.30, BEAR 0.50, CAUTION 0.60, BULL/NEUTRAL 0.95 | `system_config.py:463-512`; `tiered_strategy.py:296-302, 741-783` | YES for tier actions (master kill + proportional scaler) |
| Stock positions | Tier `max_positions` (6 at $50–100k, 8 at $100–250k); regime 0 in PANIC/BEAR/NEUTRAL | `system_config.py:567-572` | YES (Python), plus Node hard 8 |
| Options positions | 6 | `system_config.py:110`; `server/bot.ts:361` | YES |
| Options per trade | 8% (`MAX_OPTIONS_PCT_CEILING`) | `options_execution.py:82,196` | YES |
| Options total | 16% (`MAX_TOTAL_OPTIONS_PCT`) | `options_execution.py:83,269` | Only in `should_use_options` |
| `MAX_TOTAL_CAPITAL_PCT`, `MIN_POSITION_PCT` (0.02), `KELLY_DIVISOR`, `OPTIONS_SCALE` | — | `system_config.py:84-113` | **Inert** (no reader) |
| Display values `maxPositionPct: 8`, `maxExposurePct: 95`, `dailyLossLimitPct: 5` in `/api/bot/status` | — | `server/bot.ts:2327-2331` | **Hardcoded display only.** They differ from what is enforced (Node daily −3%, active 30%) |

### 1.4 Options sizing

**`_dynamic_options_size(trade, equity, positions, macro)` (`options_execution.py:138-198`).**
- Uses the Kelly of the `csp_options` bucket (8% with priors) × the same scalars as stocks × `OPTIONS_LEVERAGE_DISCOUNT = 0.50`, clamped to [1%, 8%].
- The scanner path and the tier path call it with no trade context, effectively `{"ticker", "price", "score": 70}` with volume 0, so the liquidity scalar is 0.5. Result ≈ **1.0–1.4%** of equity (code-path inference).

**Contract quantity caps per strategy.**

| Strategy | Quantity rule | Location |
|---|---|---|
| CSP | `min(budget / (strike × 100), 2)` | `options_execution.py:1065-1071` |
| Bull-put credit spread | ≤ 5 | `1181-1185` |
| Bull spread | ≤ 5 | `1554-1557` |
| Calendar / diagonal | ≤ 3 | `1678`, `1761` |
| Bear spread | ≤ 5 | `1842` |
| Short straddle | ≤ 2 | `2047` |
| Condor | ≤ 3 | `2188` |
| Buy call / buy put | `floor(equity × size_pct / (limit × 100))`, minimum 1 | `791-797` |
| Covered call | `shares_held // 100` | — |

**CSP budget (`_select_sell_put`, `855-1092`).**
- `affordable_budget = max(equity × size_pct, equity × 0.02)`, then `min(…, cash_available)` (`922-938`).
- **Stretch:** if no strike fits, the smallest OTM strike is allowed up to `equity × 0.20`, capped by cash (`974-993`).
- Strike choice is by delta closest to −0.30; without delta data, the strike closest to 95% of price (`1031-1048`).
- Limit price is `_optimized_limit_price(…, "sell")`.

**Tier engine (`tiered_strategy.py`).**
- **T1 CSP** (`407-538`):
  - `size_pct = MAX_OPTIONS_PCT (0.08) × size_scalar`.
  - `size_scalar` by VXX ratio: ≥ 1.40 → 0.30; ≥ 1.20 → 0.60; ≥ 1.05 → 0.85; else 1.0. Then ×0.50 if DD ≤ −15%, ×0.75 if DD ≤ −10% (DD measured against `voltrade_peak_equity.json`).
  - An affordability pre-filter skips underlyings priced above `equity × size_pct / 95`.
  - At most 3 actions per scan, within free options slots.
  - **The dispatcher drops `size_pct` before `select_contract`** (`server/bot.ts:4780-4800`). The tier size is used only for the affordability pre-filter and as display.
- **T2 leverage** (`545-617`): requires portfolio margin (Alpaca `multiplier ≥ 4`, `989-1017`) and DD > −8%. It multiplies T1 `size_pct` by up to 3× within headroom, capped at 2×`MAX_OPTIONS_PCT`. Because `size_pct` is ignored downstream, this has **no real effect on contract count**.
- **T3 trend** (`623-689`): BUY SPY and QQQ, `min(levered, 2 × MAX_POSITION_PCT)` each, total budget ≤ 40%. It requires the raw regime BULL, VXX ≤ 0.95 and SPY/MA50 ≥ 1.01. The floor QQQ counts toward `held_t3` (`656-664`), so T3 seldom fires. Node computes shares as `equity × size_pct / price` (`server/bot.ts:4855-4870`). **There is no exit owner for these** (§2.6).
- **T4 tail hedge:** `TIERS_ENABLED[4] = False` (`tiered_strategy.py:82-87`).
- **Master allocator** `_enforce_exposure_cap` (`741-783`): scales T1–T3 sizes proportionally into headroom.

### 1.5 Floor and hedge sizing

Formulas are in §4.5.

---

## 2. STOPS AND EXITS

### 2.0 Who fires what

| Exit family | Evaluated in | Executes? | Cadence |
|---|---|---|---|
| Stock/ETF scale-outs, phase stops, breakeven, take-profit, 7-day time stop | **Node `checkPositionOnTick`** | **YES** | Each 1-min IEX bar with v > 0, per held ticker (`server/bot.ts:7286-7297`) |
| Same rules, plus regime stop scalar, −20% hard stop, 1-h min hold, 4-day short-hold grace | Python `manage_positions` | **NO** (advisory; mutates state files) | `tier1Reflex` every 45 s, market hours, subprocess with 15 s timeout (`server/bot.ts:4210-4217`); also inside each scan (`bot_engine.py:3809-3812`) |
| `POS-KILL` −25% stock / −150% short option / −75% long option; `POS-WARN` −15% | Node `syncMonitoredPositions` | YES | `tier1Reflex` (45 s, market hours) **and** a 60-s `setInterval` at any hour (`server/bot.ts:4234`, `7227-7230`) |
| `TIME-EXIT` 14 days | Node `syncMonitoredPositions` | Probably never fires (§2.4) | Same |
| Option exits and rolls | Python `options_manager` | YES (own orders) | Each Tier-2 scan, 4:00–20:00 ET, **skipped while the Python DD halt is active** |
| Floor, defensive floor, third leg, convexity, SQQQ cleanup | Python `bot_engine` | YES | Each Tier-2 scan, skipped while DD-halted |
| `LIQUIDATE-ALL` | Node `checkDrawdownKillSwitch` | YES, only if `VOLTRADE_LIQUIDATE_ON_KILL=true` and DD ≤ −25% | Tier-1 |
| `UPGRADE` (sell a laggard for a better pick) | `runUpgradeCandidates` | **DEAD**: defined, never called (`server/bot.ts:5953`) | — |

**Global gates on Node WS exits** (`server/bot.ts:6721-6729`): no exit if the ticker is not monitored, is in `exitingTickers`, has `pos.pendingExit`, if `!state.active || state.killSwitch`, or if `positionType` is not `stock`. The circuit breaker does **not** gate WS exits.

**Monitor population.**
- `syncMonitoredPositions` covers every Alpaca position **except** `FLOOR_AND_LEG_TICKERS` = {QQQ, SVXY, SPY, SMH, KWEB, VXUS, GLD} (`server/bot.ts:393`, `6366`). Option symbols are monitored but never receive bars (not subscribed, `6624-6631`).
- `addPositionToMonitor(ticker, side, entryPrice, qty)` seeds `initialRiskPct = 4.0`, `currentStopPct = 4.0`, `tpPct = 6.0`, `atrPct = 2.0`, `regime = "NEUTRAL"` until the next sync (`6658-6690`).

**R-unit definition (shared).**
- `initial_risk_pct = 2 × ATR%`, set at the position's **first observation** by Python and persisted (`bot_engine.py:2069`, `2182`).
- `ATR% = Wilder ATR(14) / current × 100` from daily Alpaca bars (`bot_engine.py:2341-2386`, `2042-2043`). If ATR is unavailable, it falls back to 2.0, giving 1R = 4%.
- **1R is not the sizer's stop distance** (2.5 × daily ATR clamped 5–12%).
- `r_multiple = pnl% / initial_risk_pct`; `peak_r = highest_pnl / initial_risk_pct`.

**Node state source.** On each sync, `syncMonitoredPositions` overwrites from `voltrade_stop_state.json` (`server/bot.ts:6553-6614`):
- `atrPct = ps.current_stop_pct || 2.0`. That field holds raw ATR%, not a stop.
- `phase = ps.phase || derived`.
- `highestPnl = max(ps.highest_pnl, pnlNow)` and `highestPrice = max(ps.highest_price, current)`. Peaks Node observes from bars between syncs are **lost** and replaced by Python's 45-s Alpaca samples.
- `regime = ps.regime` (Python's raw `get_market_regime` label, not debounced).
- `scalesCompleted = max(memory, file)`, `remainingQty = min(memory, file, live qty)`, `breakevenActive = memory || file`.
- `pendingExit` is preserved.

### 2.1 Node `checkPositionOnTick`: rule order, thresholds and records (`server/bot.ts:6720-7222`)

Inputs per bar: `pnlPct` against `pos.entryPrice` (Alpaca `avg_entry_price` after sync). `highestPnl` and `highestPrice` ratchet. `phase` ratchets up on `peakR` thresholds 1, 2 and 3 (`6740-6750`). `isBullish = regime ∈ {BULL, NEUTRAL_BULL}` (NEUTRAL_BULL is never produced). `isBearish = regime ∈ {BEAR, PANIC, CAUTION}`.

1. **SCALE-OUT** (evaluated first; returns early when fired) (`6765-6950`).
   - Condition: `scalesCompleted < 2 && remainingQty > 1`, with `thirdQty = floor(originalQty / 3) ≥ 1`.
   - **Leg 1:** `scalesCompleted == 0 && rMultiple ≥ 1.0`.
   - **Leg 2:** `scalesCompleted == 1 && rMultiple ≥ scaleOut2R`, where `scaleOut2R` = 3.0 bull, 1.5 bear/caution, 2.0 neutral.
   - Duplicate guards: an in-memory `openOrders` exit for the ticker, or an open sell order on Alpaca → `WS-SCALE-OUT "skipping …"`.
   - CC unwind first if fewer than 100 shares would remain. A failed unwind → `CC-UNWIND-ERROR` and the leg is blocked.
   - Order: qty `thirdQty`, `getOrderParams(currentPrice, 'take_profit')` = **DAY LIMIT at the bar close** (`extended_hours` outside regular hours) (`server/orderParams.ts:41-45, 67-70`).
   - Registered in `openOrders` with `isExit: true` and **no `pendingExitFill`**.
   - Then, at submission and not at fill: `scalesCompleted++`, `remainingQty -= thirdQty`, `pos.qty = remainingQty`; after leg 1, `breakevenActive = true`.
   - Persists `scales_completed / remaining_qty / original_qty / breakeven_active` to the stop-state file (Python one-liner, Node `fs` fallback).
   - Audit `WS-SCALE-OUT "SCALE-OUT 1/3 at +x% (yR) | LIMIT sell q of N TICKER @ $p (r remaining)"`. Leg 2 is labelled `"SCALE-OUT 2/3 …"`. Plus `notify("exit")`.
2. **Stop level** (`6952-6966`).

   | Phase | Stop % |
   |---|---|
   | 4 | `atr × trailAtrMult` (bull 3, bear 1.5, neutral 2), measured from **highest price** |
   | 3 | `max(1.0, atr × 1.0)`, P&L points down from peak P&L |
   | 2 | `max(1.5, atr × 1.5)`, same |
   | 1 | `pos.currentStopPct` = `max(1.5, min(atr × 2.0, 8.0))` from sync (`6570`); 4.0 if just added |

   **No `STOP_SCALAR_*`, no minimum hold and no short-hold grace apply here.**
3. **Stop trigger** (`6972-7008`).
   - Phase 4: `(highestPrice − p) / highestPrice × 100 ≥ stopPct` → `trailing_stop`, reason `"WS TRAILING STOP Phase 4: …"`.
   - Phase 2–3: `highestPnl − pnlPct ≥ stopPct` → `trailing_stop`, reason `"WS TRAILING STOP Phase n: …"`.
   - **Breakeven:** phase ≥ 2, `breakevenActive` and `pnlPct < 0` → `trailing_stop`, reason `"WS BREAKEVEN STOP: …"`. It triggers below entry, so the market fill will be a small loss, not exactly breakeven.
   - Phase 1: `pnlPct ≤ −stopPct` → `stop_loss`, reason `"WS STOP LOSS: …"`.
4. **Take-profit** (`7010-7015`): `pnlPct ≥ tpPct && phase < 3 && scalesCompleted ≥ 2`, where `tpPct = max(4, min(3 × atr, 15))` and 999 once phase ≥ 3 (`6572`).
   - In **BULL**, leg 2 needs 3R, which means phase 4, so TP never fires.
   - In **NEUTRAL**, leg 2 needs 2R, which means phase 3, so TP never fires.
   - In **CAUTION/BEAR/PANIC**, leg 2 needs 1.5R (about 3 × ATR, which is about `tpPct`), so TP usually fires right after leg 2 (code-path inference).
5. **Time stop** (`7017-7026`): calendar days since `pos.entryDate ≥ 7` and `|pnlPct| < 2%` → `time_stop`, reason `"WS TIME STOP: held since …"`. The config `TIME_STOP_DAYS` (10, or 3/5/10 by regime) is **not read**.
6. **Fire** (`7028-7222`).
   - CC unwind first: buy back short calls at market. On failure → `CC-UNWIND-ERROR`, exit blocked.
   - Skip if any open sell order exists (`WS-EXIT "skipping — N open …"`).
   - **Quantity = `pos.remainingQty`**, not the live quantity.
   - Order context: `stop_loss` or `trailing_stop` → **MARKET** during regular hours, otherwise LIMIT at bar × 0.995 with `extended_hours`. `take_profit` → LIMIT at the bar close. `time_stop` uses context `stop_loss`, so market (`server/orderParams.ts:36-45, 62-70`).
   - Registered with `pendingExitFill {ticker, exitSide, pnlPct, exitReason: stop_loss|trailing_stop|take_profit|time_stop, entryDate, codeVersion}` (`7132-7135`).
   - Audit `WS-EXIT "<reason> (final r/N shares, k prior scale-outs) | MARKET|LIMIT sell q TICKER @ $bar"`.
   - Stop exits write `voltrade_stop_cooldown.json[ticker] = now` and drive the circuit breaker (§4.1).
   - `pos.pendingExit = true`.

**What reaches `trade_feedback` (full exits only).**
- `sweepStaleOrders` (Tier-1 every 45 s, market hours only) calls `resolvePendingExitFill`. It records only when Alpaca reports `status == "filled"`, at the confirmed `filled_avg_price` and `filled_qty` (`server/bot.ts:3790-3807`, `3866-3874`).
- Payload from `buildExitFillPayload` (`server/exitFill.ts:37-75`): `{ticker, side, qty_filled, fill_price, expected_price (= fill_price!), time_filled (= resolution time, not Alpaca filled_at), session: "regular" (always), exit_reason, exit_context{pnl_pct (TRIGGER-time pnl), exit_reason, days_held}, code_version}`.
- `track_fill` updates the most recent open entry record **matched by ticker only** (`ml_model_v2.py:2333-2352`). It sets `outcome`, `pnl_pct` (**the bot-reported trigger pnl wins over the fill-derived pnl**, `2447-2456`), `exit_price` (fill), `exit_time`, `exit_reason`, `days_held` and `exit_context`. It then calls `risk_kill_switch.record_trade_outcome(pnl_pct)` (`2458-2459`).
- When the exit order is cancelled by the 12-minute sweep, the success branch clears `pendingExit` (`3831-3833`) without calling `resolvePendingExitFill` (`3822-3833`).

### 2.2 Python `manage_positions`: advisory rules and how they differ from Node

This function **never executes**, but it **writes `voltrade_stop_state.json`** (Node reads it) and **`voltrade_stop_cooldown.json`** (`bot_engine.py:2279-2287`). So a Python "stop" that never fired still blocks re-entry.

| Rule | Python (`bot_engine.py`) | Node (live) |
|---|---|---|
| Regime source | Raw `get_market_regime` each run (`1995-2007`) | `ps.regime` from the Python file |
| Leg size | `max(1, original_qty // 3)` (`2083`) | `floor(originalQty / 3)`, needs ≥ 1 |
| Leg 1 / leg 2 thresholds | 1R / 3R bull, 1.5R bear+caution, 2R neutral (`2014`, `2086-2109`) | Same |
| Phase-1 stop | `max(1.5, min(atr × 2 × STOP_SCALAR_<regime>, 10))` (`2132-2135`) | `max(1.5, min(atr × 2, 8))`, **no scalar** |
| Phase 2/3/4 | × `STOP_SCALAR` (`2120-2131`) | No scalar |
| −20% hard stop (`POSITION_HARD_STOP_PCT`, phase 1) | Yes (`2158-2169`) | **No** (only `POS-KILL` −25%) |
| 1-h minimum hold (except loss ≤ −10%) | Yes (`2048-2060`, `2222-2225`) | **No** |
| 4-day short-hold grace (phase 1 suppressed unless ≤ −15%, hard stop or breakeven) | Yes (`2240-2251`; ties to `instrument_selector` `min_hold_days: 4`, `instrument_selector.py:1229-1233`) | **No** |
| Time stop | 7 days and \|pnl\| < 2 (`2174-2180`) | Same |
| Exit context | Rich: `atr_pct, phase, r_multiple, peak_r, highest_pnl, highest_price, stop_pct, pnl_pct, days_held, current_price, entry_price, regime, scales_completed, breakeven_active` (`2205-2220`) | **Never transmitted.** Node sends only `{pnl_pct, exit_reason, days_held}` |
| Upgrade candidates (phase 1, −stop < pnl < 1%) | Emitted (`2304-2309`) | **Dead** (`runUpgradeCandidates` never called) |

Comment disagreement: `server/bot.ts:6584-6586` says "nothing ever deletes a closed ticker's entry" from the stop-state file. Python does prune tickers that are no longer held (`bot_engine.py:2323-2332`).

### 2.3 Per-position kill (`server/bot.ts:6411-6550`)

**Thresholds.** `posKillThresholdPct(isOption, side, stockPct, optionShortPct, optionLongPct)` (`server/bot.ts:437-441`) returns:
- −25.0 for stocks (`STOCK_KILL_LOSS_PCT`).
- −150.0 for a short option (`OPTION_SHORT_KILL_LOSS_PCT`, i.e. mark ≥ 2.5× credit).
- −75.0 for a long option (`OPTION_LONG_KILL_LOSS_PCT`).

`isPosKillExemptOption(symbol, assetClass)` = `assetClass == "us_option" || isOptionSymbol(symbol)` (`server/bot.ts:430-432`, `414-417`).

**P&L basis.** Alpaca `unrealized_plpc × 100`. For options this is measured against the premium.

**Sequence.**
1. An option between −25% and its backstop gets one `POS-KILL-OPTION-HOLD` line per UTC day.
2. At or below the threshold: duplicate guards (in-memory, then Alpaca open orders) → `POS-KILL-SKIP`.
3. Otherwise `POS-KILL` at **full live qty**, with `getOrderParams(current, 'stop_loss')`: market during regular hours, limit at 0.995 with `extended_hours` outside them.
4. Registered with `pendingExitFill {exitReason: "position_kill"}` (no `entryDate`, so `days_held = 0`).

**Warning.** −15% → `POS-WARN`, at most once per hour per ticker.

**Python twins are unwired.** `risk_kill_switch.check_position_risk` (`POSITION_KILL_LOSS_PCT = -0.25`, `POSITION_WARN_LOSS_PCT = -0.15`, `risk_kill_switch.py:505-550`) is registered in the daemon (`voltrade_daemon.py:284`) but **never called**. Node re-implements it.

### 2.4 TIME-EXIT, 14 days (`server/bot.ts:6378-6405`)

- Rule: age > 14 days and P&L in [−5%, +2%] → market order for the full qty, audit `TIME-EXIT`.
- No duplicate guard, no `openOrders` registration, **no fill recording**.
- Age comes from `pos.created_at || pos.submitted_at` on the Alpaca **position** object. The Alpaca Position model has no such fields, so this **probably never fires** (code-path inference; verify with `/api/diag/positions-detail`).

### 2.5 Options: `options_manager.manage_options_positions` (`options_manager.py:855-1343`)

**Scope.** Options positions are those with symbol length > 10 or `asset_class == "option"` (the filter tests `"option"`, not `"us_option"`, `891-895`). Symbols marked `strategy: convexity_hedge` or `managed_by: convexity_overlay` are excluded (`899-916`). Legs whose state `strategy ∈ MULTI_LEG_STRATEGIES` are grouped (`76-80`, `955-973`).

**Constants** (`options_manager.py:50-59`): `DTE_EXIT_THRESHOLD = 21`, `DTE_CRITICAL = 5`, `PROFIT_TARGET_PCT = 0.50`, `LOSS_LIMIT_MULTIPLIER = 2.0`, `IC_MAX_LOSS_EXIT_PCT = 0.50`, `GAMMA_THRESHOLD = 0.08`, `DELTA_DRIFT_THRESHOLD = 0.25`, `ASSIGNMENT_DELTA_THRESHOLD = 0.80`, `ROLL_MIN_CREDIT = 0.05`, `MIN_HOLD_MINUTES = 60`.

**Standalone legs, in order.** Every `CLOSE` is a DAY limit for the full qty via `_submit_close_order` (`303-337`).
1. `dte ≤ 5` → `CLOSE type: dte_critical`, limit at mid. Ignores the minimum hold (`1069-1092`).
2. Held < 60 minutes → skip (`1094-1095`). Hold is measured from `entry_timestamp`. For tier-dispatched CSPs that is the **first time options_manager saw the position**, because `register_options_entry` is not called on that path (code-path inference: the `server/bot.ts:4796-4800` one-liner never calls it).
3. Short leg with |delta| > 0.80 and DTE ≤ 10 → `_attempt_roll` (`ROLL type: assignment_roll`). If the roll fails → `CLOSE type: assignment_close`, limit at ask (`1097-1141`).
4. DTE ≤ 21:
   - Short leg → roll (`ROLL type: dte_roll`), else `CLOSE type: dte_close` at ask.
   - Long leg → `CLOSE type: dte_close_bought` at bid (`1143-1200`).
5. Short leg: `(initial_credit − price) / initial_credit ≥ 0.50` → `CLOSE type: profit_target`, limit at mid (`1202-1230`).
6. Loss limit (`1232-1279`):
   - Short leg: `price / initial_credit ≥ 2.0` → `CLOSE type: loss_limit`, limit at ask.
   - Long leg: loss ≥ 50% → `CLOSE type: bought_loss_limit`, limit at bid. The comment at `1232` says 100% for bought; the code uses 50%.
7. |gamma| > 0.08 and DTE ≤ 30 → `CLOSE type: gamma_risk`, limit at mid (`1281-1302`).
8. Delta drift > 0.25 → `WARNING type: delta_drift`, no order (`1304-1318`).

**Recording.** `_record_options_exit_feedback` is called **at submission, not at fill** (`106-150`). It records `exit_price = current mark` and `pnl_pct = unrealized_pl / (entry × qty × 100)`. Keys sent: `exit_reason` and `exit_context`. State for the symbol is popped immediately. If the limit order does not fill, the next cycle re-initializes state (new `entry_timestamp`, so another 60-minute hold). There is no open-close-order guard.

**Roll (`_attempt_roll`, `353-499`).**
- Searches the same option type with strike ±$1 and expiry 30–50 days out.
- `net_credit = new_bid − close_ask` for a sold leg. The roll proceeds only if `net_credit ≥ 0.05`.
- Submits two independent DAY limits: close at ask, open at the new bid. **This is not atomic.** If the open fails after the close, the result is `"position is flat"` with `closed_old: True` and is reported as not rolled.
- New state comes from `rolled_position_state()` (`227-245`): `initial_credit` = the new limit, `entry_timestamp` reset.

**Multi-leg groups (`_manage_strategy_group`, `571-852`).**
- Close is via **MARKET** mleg (`_close_strategy_mleg`, `502-568`), falling back to per-leg limits.
- Rules in order: DTE ≤ 5 → `strategy_dte_critical` (ignores the minimum hold); minimum hold; credit strategies P&L / net credit ≥ 0.50 → `strategy_profit_target`; IC loss ≥ 50% of `max_loss` → `strategy_ic_max_loss_exit`; credit P&L / **gross** entry ≤ −2.0 or debit ≤ −0.50 → `strategy_loss_limit`; DTE ≤ 21 → `strategy_dte_exit`; credit with VRP_20d < −3% → `sabr_edge_exit`.
- **Not recorded to `trade_feedback`** (KNOWN BROKEN #12(c), `options_manager.py:63-75`).

**Tier metadata exit fields are never read.** `take_profit_pct`, `stop_loss_mult` and `force_close_dte` (`tiered_strategy.py:518-525`) have no consumer, nor do T3's `stop_loss_ticker` and `exit_on_ma_break` (`682-683`). options_manager's own constants happen to match the T1 values.

### 2.6 Exits for floor, leg and tier tickers

| Holding | Exit owner | Rule |
|---|---|---|
| QQQ/SMH/KWEB/VXUS basket | `_manage_spy_floor` | Rebalance when a member drifts ≥ 3% of equity. Target 0 sells **QQQ only** (§0 #10) |
| GLD | `_manage_defensive_floor` | Rebalance on regime change or drift ≥ 10%. Target never 0 today (min 8%) |
| SVXY (third leg) | `_run_third_leg` | Market sell when raw regime ∈ {NEUTRAL, BULL} (`bot_engine.py:4334-4352`). No stop |
| T3 SPY | **None.** In `FLOOR_AND_LEG_TICKERS`, so no WS exit and no `POS-KILL`; not a basket member; legacy floor exit only touches `FLOOR_TICKER` | — |
| T3 QQQ | The floor basket rebalancer, which treats it as floor and sells back to the basket target | — |
| QQQ convexity puts | `_run_convexity_overlay` | `sell_to_close` when DTE < 21 (`bot_engine.py:4518-4545`) |
| SQQQ | `_close_sqqq_position` | Market sell all on every scan (`4372-4411`) |

### 2.7 End-of-day and overnight

- There is no end-of-day flatten for stocks or options; positions are held overnight.
- `intraday_shorts.close_open_shorts()` runs at 15:50 ET or later on each Tier-2 scan (`server/bot.ts:4645-4660`). Shorts are disabled, so this is normally a no-op.
- Outside regular hours, stop exits become limits at 0.995 × bar with `extended_hours`. Python option, floor and hedge orders are plain DAY orders.
- The defensive floor has no regular-hours guard (`bot_engine.py:5370-5600`).
- The SPY floor defers outside regular hours (`4754-4782`).

### 2.8 Entry-time stop and target: computed, never used

`calculate_position` returns `stop_loss`, `take_profit`, `stop_distance_pct` and `tp_distance_pct`. `instrument_selector._build_stop_config` returns `stop_config{stop_price, take_profit, min_hold_days: 4, exit_rule}` (`instrument_selector.py:1165-1234`, `1489`, `1525`).

None of these is:
- sent to Alpaca (no bracket or OCO orders),
- consumed by Node (`grep`: `stop_loss` appears in `server/bot.ts` only as an exit-type string), or
- persisted to `trade_feedback`.

They live only in the in-memory `lastScanResult` (`/api/bot/last-scan`) until the next scan. `SIZING_HISTORY_PATH` is declared but never written (`position_sizing.py:146`).

### 2.9 The UI stop is not the enforced stop

`/api/bot/positions` computes `stopPrice` from `current_stop_pct` (raw ATR%) as follows (`server/bot.ts:1343-1350`):
- phase ≥ 2: `current × (1 − atr%)`
- phase 1: `entry × (1 − atr%)`, which is **1 × ATR**, while the live phase-1 stop is 2 × ATR clamped to 1.5–8%
- `takeProfitPrice = entry × (1 + 1.5 × atr%)`, while live is `max(4, min(3 × atr, 15))`

The bot.ts constants `STOP_LOSS_PCT = 0.15` and `TAKE_PROFIT_PCT = 0.25` only feed a `RULES` audit line (`server/bot.ts:362-363`, `7906-7907`).

---

## 3. SCALE-OUTS: definitive answer

**Yes. The live bot partially exits stock and ETF positions in thirds.**

- **Evidence.** `checkPositionOnTick` submits a sell for `floor(originalQty / 3)` at R ≥ 1.0 (leg 1) and at R ≥ 1.5/2.0/3.0 depending on regime (leg 2). The final third exits through the trailing, breakeven, take-profit or time-stop rules (`server/bot.ts:6765-6950`, `7028-7222`). The audit type is `WS-SCALE-OUT`. The Python twin exists but is advisory (`bot_engine.py:2080-2109`).
- **No partial exits exist** in options (whole-position closes), floors (rebalance deltas, which are partial in effect but not legs of one trade), `POS-KILL` (full live qty) or `TIME-EXIT` (full).

**How each leg is recorded today.**

| Leg | Order | Fill-confirmed? | `trade_feedback` | Other trace |
|---|---|---|---|---|
| Leg 1 (1/3) | DAY limit at bar close | **No**: no `pendingExitFill`; the sweep cancels it after 12 min if unfilled | **None** (`server/exitFill.ts:14-17`) | `WS-SCALE-OUT` audit text; stop-state file `scales_completed=1, remaining_qty, original_qty, breakeven_active=true`; Alpaca order history; in-memory `tradeResults` (in-memory closed-order rows, max 200, pairing by unweighted average, `server/bot.ts:851-930`) |
| Leg 2 (2/3) | Same | No | None | Same with `scales_completed=2` |
| Final leg (remaining) | Market (stop) or limit (TP) | Yes, via `resolvePendingExitFill` | Closes the entry record. `pnl_pct` = **trigger P&L of the final leg only**; `exit_price` = final-leg fill; `qty` on the record = entry qty | `WS-EXIT` audit |

**Known leg-integrity hazards (code-path inference).**
1. **Phantom Python legs.** `manage_positions` runs about every 45 s before `syncMonitoredPositions` (`server/bot.ts:4210-4234`). When it sees R ≥ 1 it writes `scales_completed=1`, `remaining_qty −= third` and `breakeven_active=true` **without any order**. Node then adopts `max/min` of those values. The real leg-1 sell is skipped, and later the final exit sells only `remainingQty`, leaving shares behind. The `POS-MONITOR-SYNC "(scale_out)"` audit line marks each Python write.
2. **Unfilled legs counted as done.** The counters advance at submission. If the 12-minute sweep cancels the limit, the counters are not rolled back.
3. **`pendingExit` latch.** Node sets it on the final exit. It is cleared only by the stale-sweep branches, not when Alpaca reports the order filled (`server/bot.ts:3866-3874`). If shares remain after the final exit (because of 1 or 2 above), the ticker stays `pendingExit=true` in memory and no WS exits fire until a process restart. `POS-KILL` still applies.

**What per-leg attribution in the cockpit requires.** None of this exists today:
- A `trade_id` / `position_id` that links the entry order, each leg order and the final exit (matching is currently by ticker only).
- A per-leg record: `{leg_no, rule ("scale_out_1R" | "scale_out_2R" | trailing_stop | breakeven | take_profit | time_stop | position_kill), trigger_time, trigger_bar_close, threshold_price, r_at_trigger, regime, qty_intended, order_id, order_type, limit_price, filled_qty, filled_avg_price, filled_at, status (filled | canceled | partial)}`.
- Scale-out fills resolved through the same `resolvePendingExitFill` contract.
- `original_qty` taken from the **filled** entry qty, not the submitted qty or the first observation.
- Python's state mutation removed, or Python made the single writer.

---

## 4. PORTFOLIO RISK

### 4.1 Halt mechanisms (all independent)

| # | Mechanism | Trigger | Peak or state | Confirmation | Effect | Latch and resume |
|---|---|---|---|---|---|---|
| A | **Node max-drawdown kill** `checkDrawdownKillSwitch` | `evaluateDrawdown(equity, equityPeak, −10)`: DD ≤ −10% (`server/drawdownGuard.ts:31-44`; `server/bot.ts:498`, `3922-3986`); also the `/api/bot/account` route (`1235-1252`) | `voltrade_equity_peak.json {equityPeak, savedAt}`, raised on every valid read (`604-638`) | Non-positive or non-finite equity is ignored (`EQUITY-READ-INVALID`) | `killSwitch=true`, `active=false`, **DELETE all open orders**, `DRAWDOWN-KILL` audit and email; optional `LIQUIDATE-ALL` (env + DD ≤ −25%, market orders, `3951-3984`) | **Latched** in `voltrade_kill_switch.json {killSwitch, reason, savedAt, lastAutoResumeAt}` (`645-684`). Clears via owner `/api/bot/kill` (`2371-2385`) or **auto-resume** (§4.3). While latched, Tier-1 runs only `killSwitchAutoResumeTick` (`7605`) |
| B | **Python portfolio DD halt** `update_equity_peak` | DD ≥ `DRAWDOWN_HALT_PCT` 18% (`bot_engine.py:576-717`; `system_config.py:308-323`) | `voltrade_portfolio_dd.json`; peak only ratchets up; sampled at Tier-2 cadence | Trips at once (unconfirmed). The first read ≥ 60 s later either confirms (DD ≥ 9%) or releases as `anomaly_release` (`669-691`); invalid equity ignored (`599-610`) | `scan_market` returns `{halted: True, halt_reason, peak_equity, current_equity, dd_pct, regime}` before scanning, which **also skips manage_positions-in-scan, options_manager, the third leg, the convexity overlay and both floors** (`2703-2716`). Node audits `DD-HALT` (`server/bot.ts:4450-4460`) | One-way ratchet: resumes only when regime ∈ {BULL, NEUTRAL} **and** equity is within 5% of peak (`693-702`). Re-baselined by `paper_resume_sync` on auto-resume |
| C | **`risk_kill_switch.check_kill_switches`** (frozen file) | Manual file `voltrade_MANUAL_KILL`; DD ≤ −20% (`PORTFOLIO_DD_KILL`); ≥ 5 consecutive losers within 24 h (`CONSECUTIVE_LOSS_KILL`) (`risk_kill_switch.py:154-290`) | Peak = `max(tiered peak passed in, equity, state.peak_equity)`; state in `voltrade_killswitch.json`; history in `voltrade_kill_history.json` (last 500) | None | When `killed`: `bot_engine` skips the tier engine and `suppress_entries_on_kill` empties scanner trades (`bot_engine.py:3908-3990`, `2433-2441`). Management is **not** stopped. Node audits `KILL-SWITCH FIRED` and `KILL-WARN` (`server/bot.ts:4980-4988`) | The DD kill latches. Auto-resume after ≥ 4 h **and** equity ≥ 95% of peak **and** VXX < 1.20 (`260-290`). The consecutive-loss kill self-clears 24 h after the last loss. Manual only via `reset_kill_state()` |
| C' | `risk_kill_switch` warnings (no consumer) | Daily min-of-day P&L ≤ −3% (`DAILY_LOSS_LIMIT`); VXX ≥ 1.40; correlation group > 60% of total BP capacity; free BP < 10% (`292-365`) | — | — | Sets `blocks_entries` / `tier_blocks`, **which nothing reads**. Only `KILL-WARN` audit lines | — |
| D | **`tiered_strategy.master_kill_switch`** | DD ≤ −20% vs `voltrade_peak_equity.json`; daily P&L ≤ −3%; **invested ≥ `MAX_TOTAL_EXPOSURE`** (PANIC 30%, BEAR 50%, CAUTION 60%, BULL/NEUTRAL 95%) (`tiered_strategy.py:279-304`) | `voltrade_peak_equity.json {peak_equity, last_updated}` (`954-986`); never lowered, not re-baselined by auto-resume | None | Blocks all tier actions, including CSPs. Node audits `TIER-KILL` (`server/bot.ts:4995-4997`) | Not latched; re-evaluated each scan |
| E | **Node Tier-2 daily loss** | `evaluateDailyPnl(equity, last_equity) ≤ −3%` (`server/bot.ts:4609-4620`) | — | Invalid reads skipped | `TIER2-LIMIT` audit and **`return`**: skips `executeTrades` **and** tier dispatch that cycle. The Python scan, including its option exits and floors, has already run | Not latched |
| F | **Node circuit breaker** | 3 WS stop or trailing exits (same ticker deduplicated within 5 min) (`server/bot.ts:7165-7212`) | in-memory `consecutiveStopLosses` | — | (a) If `state.dailyPnL / equityPeak ≤ −5%`: pause until about 20:00 container time, minimum 1 h. `state.dailyPnL` is refreshed **only by the `/api/bot/account` UI route** (`1230`). (b) Otherwise, if ≥ 2 unique tickers: pause 1 h. (c) Otherwise `TICKER-BLOCKED` for the day. The pause gates Tier-1 and Tier-2, **not WS exits** | A take-profit resets the counter; daily reset at 04:00 ET (`6199-6206`); owner reset route. Config `CIRCUIT_BREAKER_*` is **not read** |
| G | Weekly loss limit `check_weekly_loss` (−8%) | **DEAD**: called with a one-point history (`server/bot.ts:4624-4640`; `diagnostics.py:34, 44-45`) | — | — | — | — |
| H | Diagnostics | `should_pause` → skip cycle; `position_size_multiplier` → lowers the `SIZE-CAP` ceiling; `min_score_threshold` (`server/bot.ts:4357-4374`) | — | — | — | — |

Inert config keys: `DAILY_LOSS_LIMIT_PCT = 5.0`, `STOP_COOLDOWN_SECONDS`, `CIRCUIT_BREAKER_STOPS`, `CIRCUIT_BREAKER_HOURS`, `ATR_STOP_MULTIPLIER`, `MIN/MAX_STOP_PCT`, `MIN/MAX_TP_PCT`, `ATR_TP_MULTIPLIER`, `TIME_STOP_DAYS` (and all of their regime overrides) (`system_config.py:263-331`). I grepped for readers across `.py` and `.ts` and found none.

`record_trade_outcome` (used by C's consecutive-loss counter) fires only on exit fills that **match an entry record**: full WS exits, `POS-KILL` and single-leg option closes. Scale-out legs and orphan exits do not count (`ml_model_v2.py:2456-2461`).

### 4.2 Three peak files

| File | Writer | Key | Used by |
|---|---|---|---|
| `voltrade_equity_peak.json` | Node `saveEquityPeak` | `equityPeak` | Node −10% kill (and sizing DD scalar, broken by the key mismatch) |
| `voltrade_portfolio_dd.json` | `bot_engine._save_dd_state` | `peak_equity` (+ halt fields) | Python 18% halt; convexity DD escalation (`should_escalate_hedge`, ≥ 10%, `bot_engine.py:738-747`) |
| `voltrade_peak_equity.json` | `tiered_strategy.update_peak_equity`, `risk_kill_switch.set_peak_equity` | `peak_equity` | Tier DD gates (−8% T2/T3, −10%/−15% T1 scalar, −20% master kill); `risk_kill_switch` |

`voltrade_killswitch.json.peak_equity` is a fourth internal ratchet.

### 4.3 Kill-switch auto-resume (`server/killSwitchAutoResume.ts`, wired at `server/bot.ts:7477-7600`)

**Eligibility, checked before any broker read** (`autoResumeEligibility`, `:237-255`):
- the latch is set;
- `VOLTRADE_KILL_AUTO_RESUME` is not off;
- every Alpaca base URL is `paper-api.alpaca.markets`;
- the latch reason starts with `drawdown-kill`. Owner and unknown latches never auto-clear.

**Evaluation, every 45 s, counted only while the market is open** (`evaluateAutoResume`, `:259-353`). Each evaluation requires valid account, positions and fills reads and an account that is not blocked.

| Basis | Condition | Peak handling |
|---|---|---|
| `RECOVERED` | DD > −10% + 2 pp, i.e. above −8% | Peak kept |
| `DATA_ANOMALY` | Positions + fills explain < 25% of the reported drop; not within 24 h of the last auto-resume | Peak re-baselined to equity |
| `PAPER_REBASE` | Latched ≥ 2 × 6.5 market hours | Peak re-baselined to equity |

- The latch must be ≥ 30 minutes old (`COOLDOWN_MINUTES`).
- The condition must hold on **3 consecutive** evaluations.
- Explained loss = −(Σ unrealized P&L + realized P&L of fills over a window from 10 days before the latch, at most 5 pages of 100).
- Classifications: `NO_DRAWDOWN | DATA_ANOMALY | REAL_LOSS | UNRECONCILED` (`:281-321`).

**On resume** (`performKillSwitchAutoResume`, `server/bot.ts:7525-7600`):
1. `paper_resume_sync.rebaseline_python_halts` re-baselines the Python DD halt and resets the `risk_kill_switch` DD kill (not a manual kill). The tiered peak is reported but **not lowered** (`paper_resume_sync.py:1-40`).
2. Node peak re-baseline when the basis requires it.
3. `killSwitch=false`, then `saveKillSwitch("auto-resume: <basis>")` and `startBotActivity`.

**Audit trail.** `KILL-AUTO-RESUME-EVAL` (written on change only) and `KILL SWITCH AUTO-RESUME`. Status goes into `autoResumeStatus` (`/api/bot/status`, `/api/health.checks.bot.autoResume`). Fields: `eligible, latchReason, outcome, classification, basis, reason, consecutiveOk, required, lastEvaluation, nextCheck, numbers{equity, peak, drawdownPct, resumeAbovePct, latchedMinutes, marketHoursSinceLatch, rebaseAfterMarketHours, reportedDrop, unrealizedPl, realizedPl, explainedLoss, explainedRatio}, lastResume`.

### 4.4 Correlation and sector caps

- `check_sector_correlation` (scan): sector count against the tier cap (2 under $500k), plus the high-beta and average-beta blocks (`bot_engine.py:1881-1961`).
- `check_correlation_pre_trade` (Node, `max_sector_pct = 0.40`): maps only tech, fin, broad and vol; unknown tickers are allowed; fails open (`risk_kill_switch.py:558-620`).
- `CORRELATION_CAP = 0.60` by group: **warning only** (C').
- `is_ticker_blocked_by_correlation`: **unwired** (`risk_kill_switch.py:399-420`).
- T1 CSP de-dupes by underlying (`tiered_strategy.py:448-449`).

### 4.5 Hedges and floors (all inside `scan_market`, every Tier-2 scan, **skipped while DD-halted**)

**Floor regime debounce** (`_resolve_floor_regime`, `bot_engine.py:4825-4945`).
- A raw `get_market_regime` label must be seen on `FLOOR_REGIME_CONFIRM_CYCLES = 3` distinct macro snapshot timestamps before it takes effect.
- Escalation into CAUTION, BEAR or PANIC applies immediately.
- A VXX fallback reading holds the current regime.
- State keys in `voltrade_floor_state.json`: `floor_regime, floor_regime_raw, floor_regime_pending, floor_regime_pending_count, floor_regime_last_obs`.

**SPY/QQQ floor basket** (`_manage_spy_floor`, `4948-5367`).
- `target_pct = SPY_FLOOR_<regime>`: BULL 0.70, NEUTRAL 0.90, CAUTION 0.35, BEAR 0, PANIC 0 (`system_config.py:135-139`).
- Trend cap, using QQQ 50/200-day MAs from macro or computed from 210 bars: death cross (50 < 200 and price < 200) → `min(target, 0.20)`; price < 200 only → `min(target, 0.50)` (`5023-5048`).
- Basket mode (`FLOOR_BASKET_ENABLED = True`): `member_target = equity × target_pct × weight`, weights QQQ 0.50, SMH 0.15, KWEB 0.10, VXUS 0.15, CASH 0.10 (virtual). Trades a member when `|current − target| / equity ≥ 0.03`, using **MARKET** DAY orders, regular hours only, otherwise `buy_deferred` / `sell_deferred` (`5067-5175`).
- **When target is 0, only QQQ is sold** (legacy branch `5178-5214`, CC unwind first). See §0 #10.
- `last_regime` and `last_rebalance` are kept in the same state file. The single-ticker legacy mode (10% band, limit orders, `5216-5365`) is unused while basket mode is on.

**Defensive floor, GLD** (`_manage_defensive_floor`, `5370-~5600`).
- `target = DEFENSIVE_FLOOR_<regime>`: BULL 0.08, NEUTRAL 0.08, CAUTION 0.15, BEAR 0.30, PANIC 0.40 (`system_config.py:195-200`).
- Death-cross override raises the target to at least 0.25 (`5427-5446`).
- Rebalances on regime change or drift ≥ 10%, with a DAY limit at the last price and **no regular-hours guard**.
- State in `voltrade_defensive_floor_state.json {last_regime, last_rebalance}`.
- The docstring says BULL/NEUTRAL 0% and CAUTION 10%; config says 8% and 15%.

**Third leg, VRP harvest via SVXY** (`_run_third_leg`, `4218-4369`).
- `vrp_pct = min(LEG3_VRP_PCT = 0.40, (1 − deployed) × 0.6)`. The docstring says 15%; config says 0.40.
- Entry when 1.05 ≤ VXX ratio ≤ 1.25, VXX is falling (last close < close 5 bars back) and SVXY is neither held nor pending: DAY limit at last × 1.001.
- Exit: market sell of `_third_leg_exit_tickers` = {SVXY, ITA, XOM, LMT} (GLD removed as floor-owned) when the raw regime ∈ {NEUTRAL, BULL}.
- No stop and no regular-hours guard.

**Convexity overlay, QQQ puts** (`_run_convexity_overlay`, `4414-4715`). It **ignores the `enabled: False` flag**.
- Budget = equity × (4% if PANIC/BEAR or `should_escalate_hedge()` [DD ≥ 10% in the Python DD file]; 3% in CAUTION; 2% otherwise).
- Strike ≈ 0.80 × QQQ, DTE 45–75 (target 60).
- `qty = max(1, int(budget / (mid × 100)))`, limit at mid. Without a quote, the mid falls back to 0.7% of QQQ.
- Roll when DTE < 21 with a market `sell_to_close`.
- Writes `{strategy: 'convexity_hedge', managed_by: 'convexity_overlay'}` into `voltrade_options_state.json` **whether or not the order succeeded**.
- Probable order rejection noted in §0 #9.

**Other hedges.**
- SQQQ legacy cleanup on every scan (`4372-4411`).
- T4 tail hedge disabled (`tiered_strategy.py:82-87`).
- Covered-call unwind: Python `_unwind_covered_calls` is used only by legacy floor sells (`4718-4751`); Node CC-UNWIND runs before WS scale-outs and full exits.

### 4.6 State files (`DATA_DIR` = `/data/voltrade`, falling back to `/tmp`)

| File | Keys |
|---|---|
| `voltrade_kill_switch.json` (Node) | `killSwitch, reason, savedAt, lastAutoResumeAt` |
| `voltrade_equity_peak.json` (Node) | `equityPeak, savedAt` |
| `voltrade_liveness.json` (Node) | `lastActiveAt` |
| `voltrade_portfolio_dd.json` | `peak_equity, last_equity, last_updated, halted, halt_reason, halt_started_at, halt_confirmed, halt_trip_ts, halt_trip_equity, last_anomaly_release{released_at, trip_reason, trip_equity, confirm_equity, confirm_dd_pct, peak_equity}` |
| `voltrade_killswitch.json` | `killed, kill_reason, killed_at, peak_equity, consecutive_losses, last_loss_ts, daily_loss_date, daily_loss_pct` |
| `voltrade_kill_history.json` | `[{timestamp, reason, details}]` |
| `voltrade_MANUAL_KILL` | touch file |
| `voltrade_peak_equity.json` | `peak_equity, last_updated` |
| `voltrade_stop_state.json` | per ticker: `initial_risk_pct, highest_pnl, highest_price, phase, current_stop_pct (= raw ATR%), r_multiple, entry_date, entry_timestamp, days_held, original_qty, remaining_qty, scales_completed, breakeven_active, regime, short_hold_suppression?` |
| `voltrade_stop_cooldown.json` | `{ticker: epoch_seconds}` (24 h retention) |
| `voltrade_options_state.json` | per OCC symbol: `entry_price, entry_delta, entry_date, entry_timestamp, initial_credit, max_profit_target, highest_value, strategy, setup, ticker, side, qty, max_loss, pending_entry_feedback?, current_delta, current_gamma, current_theta, current_price, dte, last_checked` |
| `voltrade_floor_state.json`, `voltrade_defensive_floor_state.json` | See §4.5 |
| `voltrade_trade_feedback.json` | Entry and exit records (§2.1) |

---

## 5. Healthy values and failure signatures

| Signal | Healthy | Failure signature and meaning |
|---|---|---|
| `state.killSwitch` (`/api/bot/status`, `/api/health.checks.bot.status`) | `false` / `"active"` | `"killed"` for more than 2 market hours means **loop dark**: no exits of any kind, no option management, no floor rotation; open orders were cancelled at the trip; `liveness.dark = true` and health degraded. `autoResume.outcome` of `INELIGIBLE` means an owner or unknown latch that only the owner can clear |
| `autoResume.basis` / `consecutiveOk` | null / 0 while not latched | Stuck at `UNRECONCILED` with no basis means waiting for `PAPER_REBASE` (13 market hours) |
| `DD-HALT` audit lines each Tier-2 | absent | Present means the Python halt is active: scans return 0 candidates, **and option exits, floors and hedges are frozen**. A large `dd_pct` far from Node's DD points to a bad peak (`anomaly_release` handles this) |
| `KILL-SWITCH FIRED` | absent | The `risk_kill_switch` latch can persist ≥ 4 h until equity is ≥ 95% of peak and VXX < 1.20; `kill_suppressed_trades` is non-empty |
| `KILL-WARN` | occasional | Constant "Correlation cap breach: INDEX_ETF" or "Free BP below 10%" is warning only, with no effect |
| `TIER-KILL "Portfolio invested X% >= Y%"` | rare | Frequent means the floor plus active book is at the regime cap, and **all CSP entries are blocked** (KNOWN BROKEN #20 class) |
| `TIER2-LIMIT` | absent | Present means daily ≤ −3%: entries and tier dispatch skipped for the cycle |
| `CIRCUIT-BREAKER` / `TICKER-BLOCKED` | rare | Pause ≥ 1 h; Tier-1 and Tier-2 stop, WS exits continue |
| `WS-SCALE-OUT` followed by `SWEEP … sell` of the same qty | never | A leg counted as done but not filled (§3) |
| `POS-MONITOR-SYNC (scale_out)` | rare | Python wrote a phantom scale-out; check whether a matching `WS-SCALE-OUT` exists |
| `POS-QTY-CORRECTED` | rare | `remaining_qty` was stale against the live qty |
| `WS-EXIT "skipping — N open sell orders"` repeating | rare | An exit order is stuck |
| `WS-EXIT-ERROR` with 403 | never | Exit qty is larger than held |
| No `STREAM "Real-time feed live"`, or `STREAM-ERROR` | — | WS exits are starved, so stops never fire on stocks |
| `POS-KILL` | rare | A single name is down 25% or more |
| `POS-KILL-OPTION-HOLD` daily | normal for fresh short premium | — |
| `EQUITY-READ-INVALID` | rare | Broker read garbage; kill and halt evaluation skipped |
| `T1-FAIL "insufficient options buying power"` | rare | CSP budget is above Alpaca capacity |
| `stop_state` entries == held non-floor stocks | equal | Extra entries are stale (Python prunes them only if it runs) |
| `trade_feedback` share of `orphan_exit` | low | Currently high (83% noted at `ml_model_v2.py:2410-2420`). Tier CSPs never register an entry (§2.5), so their exits orphan |

---

## 6. Gaps for a cockpit doing ideal-vs-real exit attribution

1. **No trade or position identity.** Entry and exit are matched by ticker plus "most recent open record" (`ml_model_v2.py:2333-2352`). Needed: a `position_id` minted at entry submission and carried on every leg and order (Alpaca `client_order_id`).
2. **No per-leg records.** Scale-out legs are absent from `trade_feedback` and are not fill-confirmed. Needed: a leg table as specified in §3.
3. **The intended stop or threshold price is never computed as a price.** Stops are percentages of P&L (phases 2–3) or of the high (phase 4). Needed: persist `threshold_price` per rule at trigger time:
   - phase 1: `entry × (1 − stop%)`
   - phase 2–3: `entry × (1 + (highestPnl − stop%)/100)`
   - phase 4: `highestPrice × (1 − stop%)`
   - breakeven: `entry`
   - leg 1: `entry × (1 + initial_risk%)`
   - leg 2: `entry × (1 + k × initial_risk%)`
4. **The trigger observation is lost.** The trigger is a 1-minute IEX bar close. Its time and price exist only in the audit text (`@ $bar`). Needed: `trigger_ts`, `trigger_price`, `bar_source: "iex_1m"`. Without these, gap-through cost (trigger versus threshold) and latency cost (threshold versus fill) cannot be separated.
5. **Exit slippage is unmeasurable from `trade_feedback`.** The exit payload sets `expected_price = fill_price` (`server/exitFill.ts:51-52`). `pnl_pct` is the **trigger** P&L, not the realized one (`ml_model_v2.py:2447-2456`). `time_filled` is the sweep time, not Alpaca `filled_at`, and `session` is always "regular".
6. **The real entry fill is not stored.** The regular entry record stores `fill_price = realisticFillPrice`, a synthetic random haircut on Alpaca's fill. `alpaca_fill_price` is sent but dropped by `track_fill` (`server/bot.ts:5670-5686`; `ml_model_v2.py:2502-2523`). ETF entries store the **underlying's** price (`server/bot.ts:5411`).
7. **The sizing decision is not persisted.** `scalars`, `kelly_base`, the bucket, `stop_loss` / `take_profit` from the sizer, `stop_config` from the selector, and the tier `size_pct` live only in the in-memory `lastScanResult`. Needed: write the sizing record keyed by `position_id` (§1.1 keys) so the cockpit can show *ideal size versus filled size* (Node halves ETF shares; CSPs ignore the tier size; quantity caps apply).
8. **Python-side exits are not in the Node audit trail or in feedback.** Floor rebalances, third-leg, convexity and SQQQ orders, multi-leg option closes and rolls appear only in Python logs and `lastScanResult` (no `server/bot.ts` reader for `spy_floor`, `defensive_floor`, `third_leg`, `convexity_overlay` or `options_management`). Option single-leg exits are recorded at **submission** with the mark price.
9. **Rule context at exit is discarded.** Python's rich `exit_context` is never sent; Node sends `{pnl_pct, exit_reason, days_held}` only. Needed: `phase, peak_r, r_at_exit, highest_price, stop_pct, atr_pct, regime, scales_completed`.
10. **Halt and latch periods are not attributed to positions.** While the kill switch is latched or the DD halt is active, exits that "should" have fired do not fire. The cockpit needs a halt timeline: Node latch on/off, `DD-HALT` intervals, `risk_kill_switch` `killed_at`, `TIER-KILL` cycles, circuit-breaker windows.
11. **Phantom-state ambiguity.** Python and Node both write `voltrade_stop_state.json`. One writer is needed, or the leg counters must advance only on confirmed fills.
12. **The UI stop is not the enforced stop** (§2.9). The cockpit must compute the stop from the live rules, not from `/api/bot/positions.stopPrice`.

---

## 7. Disagreements between documentation, comments and code

- The `manage_positions` docstring says its scale-out, stop and breakeven logic runs. The code only advises and Node executes (`bot_engine.py:1965-1975` vs `server/bot.ts:4218-4225`).
- The `position_sizing` stocks-prior comment says "small starter size". The code blocks (§1.1).
- `_regime_scalar` docstring bands differ from the code bands (`position_sizing.py:579-583` vs `595-606`).
- `_portfolio_heat_scalar` says "QQQ floor is excluded". The other four floor ETFs are counted.
- The DD-sizing comment implies it is active. It is dead (key mismatch).
- `CONVEXITY_OVERLAY.enabled = False` with comment "Disabled 2026-04-17". The overlay still runs.
- The `system_config` "STOP LOSS & TAKE PROFIT" block (6% SL / 12% TP / 10-day hold) and the regime overrides are presented as live. No reader exists.
- `/api/bot/status` hardcodes 8% / 95% / 5%. The bot.ts header comment's regime table ("BULL: … 12% TP, 6% SL") does not describe the live exits (`server/bot.ts:364-371`).
- Defensive floor docstring (0% in BULL/NEUTRAL, 10% CAUTION) vs config (8% / 8% / 15%).
- Third-leg docstring (15%) vs `LEG3_VRP_PCT = 0.40`.
- `shadow_portfolio` claims its +2% / −4% / 5-day labels "match what manage_positions() actually does in production". They do not: the live rules are ATR phases, thirds and a 7-day time stop (`shadow_portfolio.py:24-26`, `83-84`, `446-447`).
- options_manager Check 5 comment says 100% for bought options. The code uses 50%.
- `system_config` comment says DD confirmation happens "3 min later". The setting is 60 s (next scan) (`system_config.py:314-322`).
- `server/bot.ts:6584` says stop state is never pruned. Python prunes it.
- `instrument_selector` `min_hold_days: 4` says "Position manager must respect this field". The live (Node) manager does not.
- The CAUTION `MAX_POSITIONS = 4` is overridden by the size tier (`system_config.py:509` vs `567-572`).
