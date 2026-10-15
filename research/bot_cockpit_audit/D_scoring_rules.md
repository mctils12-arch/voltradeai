# Audit D: Candidate Generation, Scoring, and Entry Rules/Vetoes

Repo: /home/user/voltradeai @ 89983285 (2026-10-15), package.json v1.0.1084.
Scope: everything from universe to `executeTrades()`. This covers stocks, the options scanner, the covered-call sweep, and the tier engine (CSP/T2/T3/T4).
Method: I read whole functions. Live evidence comes from read-only `/api/diag/*` probes (token-gated, human-authorized), queried 2026-10-15 ~03:15Z.
Notation: `be` = bot_engine.py, `ts` = tiered_strategy.py, `sc` = system_config.py, `os_` = options_scanner.py, `cu` = csp_universe.py, `is_` = instrument_selector.py, `oe` = options_execution.py, `ps` = position_sizing.py, `rks` = risk_kill_switch.py, `bot` = server/bot.ts, `sp` = shadow_portfolio.py.

---

## 0. TOP FINDINGS (new or under-documented bugs that matter for the cockpit)

1. **The tie-break re-ranking never runs. It raises on every scan.** At `be:3081` it reads `current_positions`, but that variable is first assigned later at `be:3186`. Python therefore treats it as a function-local, so the read raises `UnboundLocalError`. The `except` at `be:3157-3160` catches it and logs `tie-breaking failed: ...`, then the sort falls back to score only.
   - Consequence: `_rank`, `_tiebreak`, `_factor_strength` and `_sector_clash` are never set.
   - `top_10[].tiebreak_score` and `factor_strength` are always `null`, and `sector_clash` is always `false`. `rank` is just list position (`be:4034-4037`).
   - The UI and `bot:4499-4504` display these as if they were real. I verified the scoping with `symtable`.
2. **Tier 3 (trend capture) can never fire from the live path.** `TierContext.regime` is set at `be:3920` as `_regime_ctx.get(...) if '_regime_ctx' in locals() else "NEUTRAL"`. `_regime_ctx` is a local of `deep_score`, not of `_scan_market_inner`, so the regime is always `"NEUTRAL"`. `tier3_trend_capture` requires `ctx.regime in ("BULL","NEUTRAL_BULL","bull")` (`ts:641`), so it always returns `[]`. Live timings agree: `tier3_sec: 0`.
3. **The sector-correlation veto blocks every candidate whose sector resolves to "Unknown" once the book holds 2+ options.**
   - `current_tickers` includes OCC option symbols (`be:3187-3190`). `check_sector_correlation` re-adds them as "pending stock tickers" (`be:1913`), and `_get_sector(OCC)` returns `"Unknown"`.
   - Floor ETFs KWEB, VXUS and SMH also resolve to "Unknown" unless the sector cache has them.
   - I reproduced this locally with the live book: an `ACME` candidate with sector Unknown or None was blocked, while Technology and Healthcare passed.
   - This lands in shadow as `rejected_heat` (493 live records, 24% +5d win rate).
4. **The shadow decision `"taken"` does not mean "taken".** `deep_score` labels it `taken` whenever `score >= 63` (`be:1822-1828`). The 63 is effectively hard-coded: `_scan_params` is never in `deep_score`'s locals.
   - Only the sector veto (`rejected_heat`) and the spread veto (`rejected_other`) relabel a record afterward (`be:3349-3355`, `be:3427-3434`).
   - In the NEUTRAL regime the scan loop does not evaluate any stock candidate at all (`slots_available=0` → `break` at `be:3239`), yet each still logs as `taken`.
   - Live: 8,372 of 20,000 shadow records say `taken`, while live stock entries are essentially zero.
5. **Covered-call candidates are dead on arrival, yet they generate most of the live "trade candidates".**
   - The sweep (`be:3621-3705`) emits `SELL COVERED CALL` for any stock position of 100 shares or more. That includes the floor ETFs KWEB (388) and VXUS (133).
   - In `executeTrades` they reach `evaluate_and_execute` → `oe.should_use_options` (`oe:212-331`). That function doesn't recognise "SELL COVERED CALL" and the payload carries vrp=0, so it returns stock. The result is an `OPTIONS-SKIP` audit and `continue` (`bot:5563-5566`).
   - Live, they never reach that point anyway: 158 `OPTIONS-SLOT-FULL` lines in about 6 hours (VXUS 49, KWEB 49, TSLA 38…), because options slots are 6/6.
   - Off-hours they are queued (`QUEUE VXUS queued for market open (score 75)`). The morning queue counts **all** positions (11 live ≥ 8), so it breaks at once with `MORNING Max positions (8) reached` (`bot:4039-4042`). If it got past that, the trade would hit `MORNING-SKIP` because qty is 0.
6. **Node and Python disagree on several thresholds:**

   | Threshold | Python | Node |
   |---|---|---|
   | Options position cap | `be:192` `MAX_OPTIONS_POSITIONS=8` | `bot:361` `=6`; `sc:110` also says 6 |
   | Score floor | `MIN_SCORE` 63/67 | `state.minScoreThreshold` 65 by default, **raised to 75** when diagnostics sees "Low win rate" — active live: `DIAGNOSTIC [MEDIUM] Low win rate: 27.8% over 36 trades` |
   | Options scanner score floor | 60 | the same Node 65/75 floor overrides it |

7. **`STREAM-SIGNAL` triggers are non-causal.** `tier2Intelligence` reads the queue, audits `STREAM-PRIORITY ... processing first` and clears it (`bot:4381-4388`). The scan is still called with `{}` (`bot:4406-4412`), so the streamed tickers never influence candidate generation.
8. **`runUpgradeCandidates` (`bot:5953`) has no call site.** `upgrade_candidates` from `manage_positions` are returned but never consumed, and the `_consumed` check at `bot:5274` is dead. `ml_attribution_summary` (`be:4080`) is also never called, and its input fields `rules_score`/`ml_score_raw` are never written to feedback, so it would always return INSUFFICIENT_DATA.
9. **The ML fallback is blended in as if it were ML.** `deep_score` excludes `model_type == "fallback"` (`be:1745`), but `_rule_based_fallback` returns `"fallback_rules"` (`ml_model_v2.py:1712`). When no model is loaded, a rules heuristic gets the ML weight and is labelled "ML model: …".
10. **Tier 1/2 sizing metadata is thrown away at dispatch.** The SELL_CSP dispatch passes only ticker/strategy/price/equity/cash to `select_contract` (`bot:4801`). `size_pct` (including the T2 leverage multiplier) and `metadata` (`target_delta`, `target_dte_*`, `min_iv_rank=40`) never reach contract selection. `select_contract` recomputes size via `_dynamic_options_size` (`oe:444`) and uses its own delta of −0.30 (`oe:1040`). **T2 leverage and T1_MIN_IV_RANK are effectively unenforced.**
11. **The shadow regime label is effectively constant.** `deep_score` calls the Markov regime with `_spy_rets = []` (`be:1454`). Live `by_regime`: NEUTRAL_BULL 19,984 / NEUTRAL 16. Shadow records therefore cannot be regime-conditioned, and the same label feeds `regime_at_entry` on stock trades.
12. **The stock trade dict carries no `volume` key** (`be:3554-3597`). Node's fill slippage haircut falls back to `volume || 1000000` (`bot:5660`). 1,000,000 is not `> 1000000`, so every stock fill lands in the lowest-liquidity `else` bucket: a 0.12–0.25% haircut (`bot:5665`). This is a measurement concern for Audit E/F.

---

## 1. PIPELINE ORDER (universe → executeTrades)

### 1.0 Trigger (what wakes a scan)

| Trigger | Where | Effect |
|---|---|---|
| Tier-2 timer | `scheduleTier2` `bot:7659-7731`, cadence `getTier2Interval` `bot:7619-7636` | Every 60s from 9:30–10:30, 3m to 12:00, 7m to 14:00, 2m to 16:00, 15m from 16:00–20:00, 5m from 04:00–09:30, 30m overnight. Runs only if `clock.is_open` or ET hour is in [4,20) (`bot:7681`). Skipped while `!state.active`, the kill switch is on, a scan is already running, or the circuit breaker is active (`bot:7667-7674`). Failure backoff runs 60→600s (`bot:7717-7725`). |
| Overnight research | `runOvernightResearch` `bot:6010+`, called from `bot:5209-5214` | 20:00–22:00 ET: `bot_engine.py full` in a subprocess. It sets `lastScanResult` but **does not execute or queue** trades. Python-side floors and third leg do submit their own orders inside that scan. 02:00–04:00 ET: `bot_engine.py scan`, which only produces a report. |
| STREAM-SIGNAL | WS handler `bot:7310-7333` | Triggers when the 1-min bar shows ≥2.5× volume, ≥0.8% bar change and ≥0.3% 5-bar move. **Non-causal** (Top Finding 7). |
| Manual | `/api/bot/refresh-signals` → `generateSignals` (`bot:3596-3678`, UI signals only, never trades); `/api/bot/trade` → `checkTradeAllowed` (`bot:1044-1062`; **this is the only caller**, `bot:3200`) | Neither is part of the bot loop. |

Python entry: `pythonCall("run_full_scan", {}, ...)` (`bot:4406`) → daemon method table `voltrade_daemon.py:303` → `be.scan_market` (`be:2461`). Outside the daemon, the subprocess fallback runs `python3 bot_engine.py full` (`be:5618+`; `scan`, `full` and `__main__` all call `scan_market`). Timeouts: SIGALRM at 55s only in the main thread (`be:2494-2502`); daemon/Node at 300s. On `TimeoutError` the scan returns `_partial_scan_result` without an `error` key (`be:2512-2518`).

### 1.1 Pre-gates (Python, `_scan_market_inner`)

1. **ULTRA_SURVIVAL_MODE** (`be:2644-2676`): if env `VOLTRADE_ULTRA_SURVIVAL_MODE` is 1/true/yes, the scan returns early with `scanned:0, ultra_survival_mode:true` and an empty everything.
2. **DD halt** (`be:2686-2721`): `update_equity_peak(equity, regime)` (`be:576-717`, `DRAWDOWN_HALT_PCT=18` from `sc:308`, confirm/resume `sc:311-322`). If halted, the scan returns `{trades:[], new_trades:[], top_10:[], halted:true, halt_reason, peak_equity, current_equity, dd_pct, regime}`. This blocks **everything in the scan**: options scanner, floors and tier engine. Node audits it as `DD-HALT` (`bot:4464-4473`).

### 1.2 Universe (`be:2723-2756`)

`_get_full_universe()` (`be:2390-2429`) reads `/v2/assets?status=active&asset_class=us_equity`. It keeps symbols that are `tradable`, contain no ".", don't end in "W", and are ≤5 characters. The list is cached for 24h in `DATA_DIR/universe_cache.json`. The top-50 gainers and losers from `/v1beta1/screener/stocks/movers?top=50` are appended. If both are empty, the fallback is `most-actives top=100`; if that also fails: `{"error":"Could not fetch market universe"}`.

### 1.3 Snapshots → quick scan (`be:2758-2895`)

- Snapshots are fetched in serial batches of 400 (`be:2771`). `_parse_snapshot_batch` (`be:2525-2568`) keeps `(c,o,v,pc)` where `c>0`.
- **Quick-scan filters** (per symbol): `c < MIN_PRICE(5)` or `v < _min_vol` skips the symbol (`be:2836`). `_min_vol` is 100,000 from 9:30–9:59 ET and `MIN_VOLUME` = 500,000 otherwise (`be:2816-2817`, constants `be:200-201`). A symbol containing "." or longer than 5 characters also skips (`be:2838`).
- **quick_score** (`be:2840-2859`) = `min(|chg|,15)*3 + min(v/1e6,5)*2 + (5 if |chg|>10 else 2 if |chg|>5 else 0) + (-30 if |chg|>50 else -15 if |chg|>30 else 0)`. **Its maximum is 60**, so a candidate that never gets deep-scored can never pass `MIN_SCORE` (63).
- **quick_result keys:** `ticker, price, close, open, prev_close, volume, dollar_volume, change_pct, quick_score, above_vwap (=c>o), range_pct (always 0), vwap_dist (always 0), reasons ([])`.
- If nothing passes: `{"error":"Could not fetch market data from Alpaca", "debug_detail", "trades":[]}` (`be:2868-2873`).
- The list is sorted by quick_score (`be:2880`). The partial checkpoint (`be:2884-2890`) is written as `top_10` from the quick results.

### 1.4 Memory gates → deep_score (`be:2897-3064`)

- **SURVIVAL_MODE** (`be:2915-2959`): triggers when env is "1", or when unset and RSS ≥ `VOLTRADE_SURVIVAL_THRESHOLD_MB` (700). It returns quick-only `top_10` (`action_label:"WATCH"`) with `deep_analyzed:0, new_trades:[], survival_mode:true`. Options and tiers are skipped too.
- **Deep cap**: env `VOLTRADE_DEEP_SCORE_CAP`, default **5**, clamped to [1,10] (`be:2969-2973`). The config key `DEEP_SCORE_TOP_N` (`sc:364`) is **not read**.
- **mem guard** `_deep_score_guard_decision` (`be:281-320`) with skip=550MB / trim=400MB (env `VOLTRADE_MEM_SKIP_DEEP_MB` / `VOLTRADE_MEM_TRIM_DEEP_MB`, `be:2987-2994`). On `skip`, `deep_scored = scored[:10]` and nothing is deep-scored, so every stock score ≤60 and **zero stock trades are possible**. On `trim`, the cap drops to 3.
- Hard ceiling of 15 (`be:3036-3038`).
- **Serial loop with a 35s wall cap** (`be:3044-3061`). Once 35s have elapsed the loop breaks, and candidates not yet reached stay `None` and are **dropped** (`be:3064`). Live timing on 2026-10-14 23:52Z: `deep_score` took **35.67s**, so the cap is binding.
- `_deep_one` swallows exceptions and returns the quick dict (`be:3025-3030`).
- The list is sorted by `deep_score`, else `quick_score` (`be:3067`). The tie-break at `be:3079-3160` is broken (Top Finding 1).

### 1.5 `deep_score(ticker, quick_result, _diag)` (`be:775-1877`) — see §4 for the score math

1. `get_stock_details` (`be:467-501`) runs `python3 analyze.py TICKER --mode=scan` as a subprocess with a 30s timeout.
   - **`--mode=scan` is ignored.** `analyze.py:2642-2663` routes any non-"SCAN" argv[1] to the full `analyze_ticker()` (`analyze.py:2647, 2659`).
   - If the response is missing or contains `"error"`, deep_score **returns the quick_result unchanged** (`be:794-795`): no deep_score, no shadow log, and the ticker cannot be traded.
   - `analyze_ticker` returns an error for any ticker **without a listed options market** (`analyze.py:1998-1999`), and also when there's no data (`analyze.py:1959-1960`). This is an **implicit "must have listed options" gate** that leaves no record.
2. Five parallel enrichment fetchers (macro+news, intel, alt, social, finnhub), each with a 15s `.result` timeout (`be:859-904`). Errors go to `_diag` and surface as `data_source_errors` → `/api/diag/scanner`.
3. yfinance fundamentals in a subprocess with a 5s hard kill and 5-min cache (`be:906-964`).
4. 400 days / 300 daily bars from Alpaca for ADX/OBV/momentum/MA200 (`be:977-1045`).
5. Strategy sub-scores → rules composite → ML blend → side/label → sector cache → **shadow `log_candidate`** (`be:1819-1844`).

**deep_score output keys** (`be:1846-1877`): all quick keys, plus `ml_features` and `regime_label` (mutated onto quick_result at `be:1782-1786`, only when the ML block got that far), plus `deep_score, reasons, vrp, side, trade_type, action_label, recommendation, rec_reasoning, sentiment_score, rsi, momentum_score, mean_reversion_score, vrp_score, squeeze_score, volume_score, ewma_rv, garch_rv, horizon, leveraged_bull, leveraged_bear, entry_features, rules_only_score, ml_only_score, sector, above_ma200, ma200_pct_above`.

### 1.6 Stock trade loop (`be:3175-3597`)

- Account (`be:3176-3182`, fallback 100000). Positions (`be:3185-3199`): `num_positions` counts **stock positions only (floor ETFs included)**.
- `_macro = get_macro_snapshot()` (`be:3208-3212`). `_slots_params = get_adaptive_params(vxx, spy_vs_ma50, account_equity)` (note: **without `spy_below_200_days`**). `slots_available = max(0, MAX_POSITIONS_adaptive − num_positions)` (`be:3217-3229`).
- For each stock in `deep_scored`, the rules in the §2 table run in order. A survivor is appended as a **trade dict** (`be:3554-3597`): `action, side, trade_type, ticker, shares, price, score, reasons, recommendation, rec_reasoning, stop_loss, take_profit, position_value, sizing_reasoning, sizing_scalars, momentum_score, mean_reversion_score, vrp_score, squeeze_score, volume_score, ewma_rv, garch_rv, rsi, vrp, instrument, instrument_strategy, instrument_reasoning, instrument_ticker, instrument_scores, use_options, options_strategy, options_reasoning, options_edge_pct, rules_score, ml_score_raw, entry_features, entry_date, regime_at_entry`. There is **no `volume` key**.
- Partial checkpoint (`be:3601-3613`).

### 1.7 Covered-call sweep (`be:3621-3705`) → options scanner (`be:3731-3800`)

- **Covered calls**: one per stock position with ≥100 shares and no existing short call, not already in trades, passing the 7-day earnings guard and not in BEAR/PANIC. Fixed `score:75`, `options_strategy:"covered_call"`, `options_edge_pct:3.0`, `shares_held`. The dict has no `entry_features`, `entry_date` or `regime_at_entry`.
- `_timing_log("step6a_trade_loop_and_covered_calls")` (`be:3725`).
- **Options scanner**: `options_slots = 8 − existing_options` (`be:3734-3740`, module constant 8). It calls `get_options_trades(equity, current_tickers + trade tickers, max_new=max(1,options_slots), min_score=60.0)` and converts each result into the trade-dict shape with `regime_at_entry:"OPTIONS_SCANNER"`, `rules_score = ot.score`, plus every setup-specific key passed through (`be:3753-3797`). An exception there is swallowed silently (`be:3799-3800`).

### 1.8 Non-entry steps inside the same scan (outside this audit's scope, listed for completeness)

`manage_positions` (`be:3810`) → `position_actions` and `upgrade_candidates`. Intraday shorts are hard-disabled (`be:3818`). `manage_options_positions` (`be:3824-3828`), third leg (`be:3837`), convexity overlay (`be:3848`; disabled by config, `sc:242`), floor regime and SPY/basket floor (`be:3861-3873`), and defensive GLD floor (`be:3879-3885`). The floors and third leg **submit Alpaca orders directly from Python** (`be:4309, 4340, 4398, 5110…`).

### 1.9 Tier engine (`be:3892-3992`)

- `check_kill_switches(...)` (`rks:154-365`). **If killed**, `suppress_entries_on_kill(trades, kill_status)` (`be:2433-2441`) empties **all** `new_trades` (stock, covered-call and options-scanner). It logs only via the Python logger; the tickers go into `kill_suppressed_trades`, which **Node never reads**. The tier engine does not run.
- **If not killed**: `TieredStrategy().run_tiers(ctx)` (`ts:792-936`): `master_kill_switch` → T1 `tier1_csp_core` → T2 `tier2_leverage_multiplier` → T3 → T4 (disabled) → `_enforce_exposure_cap`.
- Output `tier_actions[]` = `{tier, action, ticker, strategy, size_pct, reason, metadata}` (`be:4055-4061`), plus `tier_stats`, `tier_kill_status`, and the `tier_engine_breakdown` timing entry.

### 1.10 Scan return (`be:4006-4076`)

`timestamp, scanned, filtered (= #quick passes), deep_analyzed, portfolio_value, cash, current_positions, slots_available, options_slots_available, new_trades, options_trades_added, position_actions, upgrade_candidates, top_10[], third_leg, convexity_overlay, intraday_shorts, options_management, spy_floor, defensive_floor, tier_actions, tier_stats, kill_status, kill_suppressed_trades, tier_kill_status, scan_phase_times, size_tier, data_source_errors` (+ `dd_event` added at `be:2508-2510`).

`top_10[]` = `{ticker, price, score, change_pct, side, action_label, reasons[:3], rank, tiebreak_score, factor_strength, sector_clash, sector, factors{momentum, mean_reversion, vrp, squeeze, volume}}`.

### 1.11 Node admission (`tier2Intelligence` `bot:4335-5216`)

1. Alpaca ping; a failure skips the cycle (`bot:4341-4353`).
2. Every 5th cycle, `diagnostics.get_auto_fix_params` sets `state.minScoreThreshold` (default 65; **75** on a "Low win rate" problem, `diagnostics.py:369-376`) and `state.positionSizeMultiplier` (`bot:4356-4377`). `should_pause` is never set True by `get_auto_fix_params` (`diagnostics.py:526-565`), so `DIAGNOSTIC-PAUSE` is dead.
3. Scan → `TIER2 Scanned N stocks, M trade candidates (via …)` (`bot:4448`). `lastScanResult = result` is **in memory only**, served by `/api/bot/last-scan` (`bot:7772-7774`).
4. UI signals from `top_10` with score ≥60, plus new_trades, position_actions and tier_actions; in memory, capped at 30 entries and 1h (`bot:4481-4559`).
5. Daily loss: `evaluateDailyPnl(equity, last_equity)`. At ≤ −3% it audits `TIER2-LIMIT` and **returns**, so no stock or tier dispatch happens (`bot:4610-4621`). Weekly loss check (`bot:4624-4641`).
6. EOD short-cover at 15:50 (`bot:4646-4659`).
7. **Market open** → `executeTrades(new_trades, equity)` (`bot:5220-5705`). **Market closed** → the first 8 (`MAX_POSITIONS`) of `new_trades` go into `morningQueue`, deduplicated by ticker, with an audit `QUEUE` (`bot:4661-4671`). `executeMorningQueue` runs on the first market-open Tier-1 cycle (`bot:4236-4240`, `bot:3990-4177`).
8. **Tier dispatcher** (`bot:4676-4978`), followed by `KILL-SWITCH`/`KILL-WARN`/`TIER-KILL` audits (`bot:4981-4997`).

---

## 2. RULE TABLE (evaluation order; "Records" = what's left behind when it blocks)

Records legend:
- **SILENT**: nothing anywhere.
- **PYLOG**: Python logger/stderr only; container logs, not persisted to any API.
- **AUDIT X**: SQLite audit row of type X (rolling 2,000 rows, messages truncated to 500 characters, `bot:1160-1180`).
- **SHADOW d**: `sp.log_candidate` / `update_last_decision` with decision `d`.

### 2A. Scan-level gates (Python, before candidates)

| # | Rule | file:line | Condition | Threshold / source | Regime | Records |
|---|---|---|---|---|---|---|
| A1 | ULTRA_SURVIVAL_MODE | be:2644-2676 | env set | env | – | stderr+PYLOG; result `ultra_survival_mode:true` → TIER2 "Scanned 0 stocks" |
| A2 | DD halt | be:2686-2721; be:576-717 | dd ≥ halt% (with confirm/release) | `DRAWDOWN_HALT_PCT` 18, `RESUME_REGIMES` [BULL,NEUTRAL], `RESUME_EQUITY_PCT` 5, `CONFIRM_MIN_SECONDS` 60, `CONFIRM_FRACTION` 0.5 (sc:308-322) | resume is regime-gated | AUDIT `DD-HALT` (bot:4464-4473) |
| A3 | Universe symbol filter | be:2414-2421 | tradable, no ".", not *W, len≤5 | hardcoded | – | SILENT |
| A4 | Snapshot usable | be:2563 | dailyBar c>0 | hardcoded | – | SILENT (all-fail → TIER2 "Scan returned error …debug_detail") |
| A5 | Price floor | be:2836 | c < 5 | `be:201 MIN_PRICE=5` (**`sc:66 MIN_PRICE` not read**) | – | SILENT |
| A6 | Volume floor | be:2817, 2836 | v < 500k (100k 9:30–9:59) | `be:200 MIN_VOLUME`; 100k hardcoded. **Contradicts** `sc:68 OPEN_HOUR_MIN_VOLUME=1M` ("first 30 min need MORE volume"), which is not read | time-of-day | SILENT |
| A7 | Symbol format | be:2838 | "." or len>5 | hardcoded | – | SILENT |
| A8 | Extreme-move soft penalty | be:2842 | |chg|>30 → −15, >50 → −30 to quick_score | hardcoded (`sc:79 MAX_CHANGE_PCT=35` is diagnostic-only) | – | SILENT (rank effect) |
| A9 | SURVIVAL_MODE | be:2915-2959 | RSS ≥700MB | env `VOLTRADE_SURVIVAL_THRESHOLD_MB` | – | stderr+PYLOG; result `survival_mode:true` |
| A10 | Deep cap (rank cutoff) | be:2969-2973, 3036 | rank > cap | env `VOLTRADE_DEEP_SCORE_CAP`=5 (≤10, hard 15) | – | SILENT (only `deep_analyzed` count) |
| A11 | Mem guard skip/trim | be:2996-3017, 281-320 | RSS ≥550 skip / ≥400 trim→3 | env, defaults 550/400 | – | stderr+PYLOG `[mem_guard …]` |
| A12 | Deep 35s wall cap | be:3049-3054 | elapsed > 35s | hardcoded | – | PYLOG "Deep scoring hit 35s cap at candidate i/n"; timing phase `deep_score`≈35s |
| A13 | analyze.py failure / no options market | be:793-795; analyze.py:1959,1999 | detail missing or `error` | – | – | `_diag["stock_details"]` → `data_source_errors` → /api/diag/scanner. **No shadow record** |

### 2B. Stock candidate loop (Python, `be:3238-3597`), in order

| # | Rule | file:line | Condition | Threshold / source | Regime-dependent? | Records |
|---|---|---|---|---|---|---|
| B1 | **Max stock slots** | be:3229, 3239 | `len(trades) >= slots_available` → **break** (stops the whole loop) | `get_adaptive_params().MAX_POSITIONS` (0 in PANIC/BEAR/NEUTRAL, else size-tier max: 4/6/8… see §2F) minus held stock positions (**floor ETFs count**) | **Yes** | SILENT. Shadow still says `taken` for score≥63 |
| B2 | Already held | be:3246 | ticker in `current_tickers` | – | – | SILENT |
| B3 | Stop-loss re-entry cooldown | be:3267-3298 | now − last_stop < cooldown | hardcoded map PANIC 4h / BEAR 3h / CAUTION 2.5h / NEUTRAL 2h / BULL 1.5h; state in `voltrade_stop_cooldown.json` (`sc:329 STOP_COOLDOWN_SECONDS` not read) | **Yes** | SILENT |
| B4 | No stock longs in BEAR/PANIC | be:3301-3302 | `_scan_regime in (BEAR, PANIC)` (`get_market_regime` incl. 200d inputs) | regime_util thresholds (§2F) | **Yes** | SILENT |
| B5 | MIN_SCORE | be:3307-3318 | final_score < MIN_SCORE | `get_adaptive_params(..., spy_below_200_days)` MIN_SCORE: BULL 63, NEUTRAL 63 (base `sc:62`), CAUTION 67, BEAR/PANIC 75 (moot after B4) | **Yes** | SILENT. Shadow label uses fixed 63, so CAUTION 63–66.9 shows as `taken` |
| B6 | Sector / correlation | be:3324-3356 → `check_sector_correlation` be:1881-1960 | (1) same-sector count ≥ cap; (2) if ≥3 existing tickers: new beta ≥1.5 and ≥3 high-beta held; or avg beta >1.6 (hardcoded `_beta_map`, default 1.0) | cap = `_slots_params.MAX_SECTOR_POSITIONS` (size tier: 2 under $500k); fail-open on exception (be:3341-3346) | tier-dependent | **SHADOW `rejected_heat`** "sector/correlation block (max N/sector)". **Bug: OCC/“Unknown” sector inflation (Top Finding 3)** |
| B7 | Dollar-volume floor | be:3368-3373 | price×volume < $50M | hardcoded `_MIN_DOLLAR_VOL` | – | SILENT |
| B8 | Blocked tickers | be:3379-3384 | ticker in set | `sc:335-345 BLOCKED_TICKERS` (DKNG, RBLX, SQQQ, TQQQ, SPXU, UPRO, UVXY, ABNB, DASH, LYFT) ∪ {"UBER"} | – | SILENT |
| B9 | Extreme mover | be:3389-3405 | change_pct > 50 (upside only) | hardcoded | – | appended to `DATA_DIR/extreme_movers_today.json` {ticker, price, change_pct, volume, score, timestamp, high, setup}; reviewed by overnight research (bot:6028+) |
| B10 | Short → buy rewrite | be:3408-3410 | side sell/short and trade_type≠options → BUY | hardcoded (shorting disabled) | – | not a veto; `deep_score`'s side is overwritten silently (an RSI>70 "SELL" becomes a BUY) |
| B11 | Spread | be:3416-3437 | (ask−bid)/mid > 0.5% | hardcoded 0.005 (`sc:80 MAX_SPREAD_PCT=0.5` not read here); fail-open on a quote error | – | **SHADOW `rejected_other`** "spread x > 0.005 max" |
| B12 | Instrument selection | be:3440-3475 → `is_.select_instrument` is_:1277-1528 | stock / 2× ETF / options pick (not a veto). Sub-gates: ETF 200-MA gate (is_:585-610), ETF PANIC block (is_:621-638, **dead**: reads `trade["macro"]`, which is never set), ETF hold ≤5d and volume ≥100k (is_:1409-1424); options require regular hours, score ≥ `MIN_OPTIONS_SCORE` 65, total options exposure < 16%, no naked calls (is_:1430-1453) | is_:154-158 constants | partially | reasoning string carried in the trade (`instrument_reasoning`, incl. "Options ruled out: …") → AUDIT `INSTRUMENT` at execution |
| B13 | Sizing block | be:3478-3500 → `ps.calculate_position` ps:878-1193 | halted asset (ps:919-921), price ≤0, equity ≤0, **negative Kelly expectancy** (ps:950-954), shares ≤0 after clamp (ps:1058-1059) | ps constants (ABS_MAX 8%, MIN 1%) | indirectly | **SILENT** (`block_reason` discarded); exception → PYLOG |
| B14 | shares ≤0 | be:3517-3518 | – | – | – | SILENT |
| B15 | Volume-share cap | be:3531-3552 | position_value / dollar_volume > cap | `_slots_params.MAX_VOLUME_SHARE_PCT` (size tier 5%…0.8%) | tier | PYLOG `VOL_SHARE_BLOCK …` |

### 2C. Covered-call sweep (`be:3621-3705`)

| Rule | line | Condition | Records |
|---|---|---|---|
| Not option / ≥100 shares / OCC length | 3628 | skip otherwise | SILENT |
| Already short call | 3632-3642 | – | SILENT |
| Already in options trades | 3645 | – | SILENT |
| Earnings guard 7d | 3649-3654 → oe:115-127 | `_earnings_scalar < 0.85` (fails closed to 0.85 on lookup failure, so it **passes**) | SILENT |
| BEAR/PANIC | 3657-3659 | `_scan_regime` (falls back to "NEUTRAL" when the stock loop never set it) | SILENT |
| price ≤0 | 3662 | – | SILENT |

### 2D. Options scanner (`os_`), called at be:3741

| Rule | file:line | Condition / threshold | Records |
|---|---|---|---|
| Calendar | os_:1764-1768 | `market_calendar.should_skip_new_options()` (half-days, pre-long-weekend) → `[]` | SILENT |
| Regular hours | os_:1487-1496 | otherwise returns "CLOSED" | SILENT |
| Candidate stage 1 | os_:1383-1403 | c ≥ $10, v ≥ 1M; high_iv when |chg|>2%, top 400 by |chg| (os_:1421); 20 anchors (os_:1247-1252) | SILENT |
| Setup 1 earnings IV crush | os_:707-846 | 0≤days≤7; VXX<1.30; both legs ~50Δ; leg spread ≤10%; implied move ≥2%; 30% of straddle ≥ 2× spread cost; avg IV ≥0.40. Score = min(95, .6·(60+ivbonus+liq+urgency) + .4·ML) | SILENT |
| Setup 2 VXX panic put sale | os_:849-961 | VXX≥1.30, SPY/MA50 ≥0.94, SPY puts 92–95% strike, 14–30 DTE, bid ≥0.50, spread <6%. Score = min(90, .6·(65+panic) + .4·ML); strategy bull_put_credit_spread | SILENT |
| Setup 3/7 high-IV premium sale | os_:964-1108, 1586-1663 | VXX<1.25; IVR≥50 or VXX in [1.05,1.25]; 20Δ legs with OI≥300, spread ≤10%; avg IV ≥0.30; 2× spread rule. Score = min(88, .6·(62+ivbonus) + .4·ML). Setup 7 (VRP+) adds +4 capped at 88 with a 60s scan-wide surface budget (os_:1279) | SILENT |
| HIGH_EDGE gate | os_:1793-1803 | setup ∈ {earnings_iv_crush, vxx_panic_put_sale, high_iv_premium_sale}; high-IV only iron_condor | SILENT |
| Surface adjust | os_:1806-1829 | sell & VRP>2% → +5; sell & VRP<−2% → −8 (floor 40) | text appended to `reasoning` |
| Min score | os_:1782-1785, 1831 | max(60, min_score=60); +5 on short week | SILENT |
| Held underlying | os_:1833 | `opp.ticker in current_tickers`. **Bug**: current_tickers holds OCC symbols, so an underlying already held via options is not deduplicated | SILENT |
| Max new | os_:1789; be:3750 | Python options slots (8 − held options) | SILENT |
| Sizing | os_:1844-1851 | 6/7/8% of equity by setup | – |

### 2E. Tier engine (`ts`)

| Rule | file:line | Condition / threshold | Records |
|---|---|---|---|
| check_kill_switches → killed | rks:154-290; be:3908-3989 | MANUAL kill file; DD ≤ −20% (rks:51, **sticky** until resume gates pass, rks:261-290); 5 consecutive losses within 24h (rks:53). On kill: **suppresses all scanner new_trades**, tier engine skipped | AUDIT `KILL-SWITCH FIRED: …` (bot:4982-4984); suppressed tickers PYLOG only |
| check_kill_switches → daily loss / regime / correlation / free-BP | rks:292-365 | daily ≤ −3%, VXX ≥1.40, group >60%, free BP <10%: these only populate `warnings` / `blocks_entries` / `tier_blocks`, **which nobody consumes** | AUDIT `KILL-WARN` |
| master_kill_switch | ts:279-304 | DD ≤ −20%; daily ≤ −3%; invested ≥ MAX_TOTAL_EXPOSURE (regime: PANIC .30, BEAR .50, CAUTION .60, NEUTRAL/BULL .95). Kills **all tiers** | AUDIT `TIER-KILL` (bot:4995-4997); **SHADOW `rejected_masterkill`** for what T1 would have proposed (ts:315-377, score 0, no features) |
| T1 enabled | ts:409, 82-87 | TIERS_ENABLED[1] | – |
| T1 slot cap | ts:422, 453-456, 493-494 | 6 − held options (sc:110) → 0 means no actions | SILENT |
| T1 size scalar | ts:427-441 | VXX ≥1.40 → .30, ≥1.20 → .60, ≥1.05 → .85; DD ≤ −15% ×.5, ≤ −10% ×.75 | in `reason` string |
| T1 universe | ts:148-176 → cu:832-857 | csp_universe top 200 (§2G), else 17 hardcoded fallback names | – |
| T1 held underlying | ts:449-451, 491 | OCC → underlying (fixed 2026-10-13) | SILENT |
| T1 affordability | ts:470-504 | underlying price > equity × MAX_OPTIONS_PCT(.08) × scalar / 95 | logger.debug only |
| T1 max 3/scan | ts:538 | first 3 in universe rank order (**no score threshold**) | SILENT |
| T2 | ts:545-616 | requires PM (multiplier ≥4, ts:1009-1011), DD > −8%, VXX <1.40, headroom >0. **Effect lost at dispatch** (Top Finding 10) | logger.info |
| T3 | ts:623-688 | VXX ≤0.95, SPY/MA50 ≥1.01, DD > −8%, **regime ∈ BULL/NEUTRAL_BULL/bull: never true** (Top Finding 2) | SILENT |
| T4 | ts:695-734 | TIERS_ENABLED[4]=False | – |
| Allocator | ts:741-783 | scales sizes to headroom | `reason` suffix "[scaled x% by cap enforcer]" |

### 2F. Regime classification and `get_adaptive_params` table

**Canonical 5-level regime** (`regime_util.classify_regime_5level`, regime_util.py:74-126; used by `sc.get_market_regime` sc:386-411). Checks run top to bottom; the first match wins.

| Regime | Condition |
|---|---|
| PANIC | VXX ratio ≥1.40 |
| BEAR | VXX ≥1.15, or SPY/MA50 <0.94, or `spy_below_200_days` ≥10, or (not above 200d and ≥5 days below) |
| CAUTION | VXX ≥1.05 |
| BULL | VXX ≤0.95 and SPY/MA50 ≥0.98 and SPY above its 200d MA |
| NEUTRAL | everything else |

Other modules classify regime their own way and disagree with this:
- options_scanner's own regime (os_:1513-1522): PANIC needs VXX≥1.30 **and** SPY<0.94.
- instrument_selector uses ratio bands.
- Markov `regime_label` (markov_regime.py:206-210) uses BULL / NEUTRAL_BULL / NEUTRAL / NEUTRAL_BEAR / BEAR.

**get_adaptive_params** (sc:441-598), regime overrides first, then the size-tier overlay:

| Key | PANIC | BEAR | CAUTION | NEUTRAL | BULL | base (sc) |
|---|---|---|---|---|---|---|
| MAX_POSITIONS (stock) | 0 | 0 | 4 → tier max | **0** | 8 → tier max | 6 |
| MAX_POSITION_PCT | .06 | .08 | .10 | .08 | .15 | .08 (then min with tier max_single) |
| MIN_SCORE | 75 | 75 | 67 | 63 | 63 | 63 |
| MAX_TOTAL_EXPOSURE | .30 | .50 | .60 | .95 | .95 | .95 |
| MAX_STOP_PCT | 5.0 | 6.0 | 6.0 | 6.0 | 6.0 | 6.0 |
| TIME_STOP_DAYS | 3 | 5 | 10 | 10 | 10 | 10 |
| STREAM_VOL_SPIKE_RATIO | 3.0 | 2.8 | 2.5 | 2.5 | 2.2 | 2.5 |
| ATR_STOP_MULTIPLIER | 1.5 | 1.5 | 1.5 | 1.5 | 2.0 | 1.5 |
| MAX_OPTIONS_POSITIONS | 6 | 6 | 6 | 6 | 6 | 6 (regime-invariant by design) |
| MAX_OPTIONS_PCT | .08 | .08 | .08 | .08 | .08 | .08 (protected key) |

**Size-tier overlay** (`ps.SIZE_TIER_TABLE` ps:74-87):
- Columns: equity threshold → `max_positions` / `max_single` / `max_sector` / `max_vol_share` / `min_$`.
- starter $0: 4 / .20 / 2 / .05 / 1k
- small $50k: 6 / .15 / 2 / .04 / 2.5k
- growing $100k: 8 / .12 / 2 / .03 / 5k
- mid $250k: 10 / .10 / 2 / .025 / 10k
- large $500k: 12 / .08 / 3 / .02 / 15k
- … up to mega $25M: 35 / .03 / 5 / .008 / 300k.
- A regime's 0 positions wins (sc:567-572). MAX_SECTOR_POSITIONS, MAX_VOLUME_SHARE_PCT and MIN_POSITION_DOLLAR come from the tier (sc:577-583).
- `time_of_day` is never passed by the entry path, so it is always "regular".

**Practical consequence (live book: 5 floor ETFs, about $85–100k equity):**
- In BULL at the "small" tier, stock slots = 6 − 5 = **1**.
- In CAUTION, min(4 → tier 6) − 5 = **1**.
- In NEUTRAL/BEAR/PANIC there are **0**.
- The floor basket consumes the active-trade slots.

**Config keys defined but not read by the entry path** (orphans): `MIN_PRICE`, `MIN_VOLUME`, `OPEN_HOUR_MIN_VOLUME`, `MAX_SPREAD_PCT` (bot_engine hardcodes), `STOP_COOLDOWN_SECONDS`, `ML_CONFIDENCE_THRESH`, `DEEP_SCORE_TOP_N`, `DEEP_SCORE_WORKERS`, `CIRCUIT_BREAKER_*`, `DAILY_LOSS_LIMIT_PCT=5.0` (sc:307; the live limits are −3% in ts:92, rks:52 and bot:350).

### 2G. CSP universe (`cu`), feeding T1

- **Layer 1** (cu:198-320, 15-min cache):
  - Excludes `CSP_BLOCKED_TICKERS` (cu:62-74) and leveraged-ETF heuristics.
  - Requires price ≥$10, price ≤ equity×0.25/95 (default $250), volume ≥1M, dollar volume ≥$20M.
  - Adds 17 anchors (cu:51-56) regardless.
- **Layer 2** (cu:612-809, 15-min cache):
  - Composite = .25 IVR + .20 VRP + .15 liquidity + .10 put skew + .10 earnings + .10 historical WR + .10 stability.
  - Earnings score ≤10 (≤7 days to earnings) → **dropped** (cu:770-771). A calendar-fetch failure scores 80 (passes).
  - Full factors only for the top 150 by dollar volume within a 45s prefetch budget (cu:671, 690). The rest get baseline 40/40/50.
  - Top 200 by composite (cu:791-792).
- Records: cache files `voltrade_csp_universe_cache.json` and `voltrade_csp_scores_cache.json` hold factors for the top 200 (exposed via `/api/system/snapshot` csp_universe). Dropped names are SILENT.

### 2H. Node admission rules (`bot`)

| # | Rule | file:line | Condition / threshold | Records |
|---|---|---|---|---|
| N0 | Daily loss | bot:4610-4621 | (equity−last_equity)/last_equity ≤ −3% → return (no entries, no tier dispatch) | AUDIT `TIER2-LIMIT` (or `EQUITY-READ-INVALID`) |
| N0b | Weekly loss | bot:4624-4641 | diagnostics.check_weekly_loss → pause/reduce | AUDIT `WEEKLY-LIMIT`/`WEEKLY-WARNING` |
| N1 | Market closed → morning queue | bot:4661-4671 | first 8 new_trades, dedupe by ticker | AUDIT `QUEUE` |
| **executeTrades** (bot:5220-5705), per trade: | | | | |
| E1 | Kill switch | 5258 | `state.killSwitch` → break | SILENT |
| E2 | Options slots | 5261-5265 | options in positions + open option orders ≥ 6 (`bot:361`) | AUDIT `OPTIONS-SLOT-FULL` |
| E3 | Stock slots | 5266-5268 | **all us_equity positions (floor included)** ≥ 8 (`bot:353`) → **break** | **SILENT** |
| E4 | Score threshold | 5269-5272 | score < `state.minScoreThreshold` (65; **75 live**) | AUDIT `SKIP … score x < threshold y` |
| E5 | Held | 5273 | `held.includes(ticker)` (OCC symbols for options, so no underlying dedupe) | SILENT |
| E6 | `_consumed` | 5274 | dead (never set) | – |
| E7 | Daily blocked tickers | 5276-5279 | 3 same-ticker stops today | AUDIT `SKIP … blocked for today` |
| E8 | Size ceiling | 5282-5286 | position_value > equity × 0.20 × positionSizeMultiplier | AUDIT `SIZE-CAP` |
| E9 | Heat cap | 5289-5292 | active deployed (excl. FLOOR_AND_LEG_TICKERS, bot:393) + value > 30% equity (`bot:352`) → **break** | AUDIT `HEAT-CAP` |
| E10 | Sector concentration | 5300-5313 → rks:577-622 | rks sector share >40% after adding an estimated 5%; fail-open | AUDIT `CORR-CAP` |
| E11 | Exchange halt | 5317-5330 | `check_halt_status` | AUDIT `HALT-SKIP` / `HALT-CHECK-WARN` |
| E12 | Max retries | 5334-5338 | ≥10 attempts per ticker per day (`bot:3764`) | AUDIT `MAX-RETRY` |
| E13 | Duplicate open order | 5341-5347 | any open order on the symbol | AUDIT `DUP-SKIP` |
| E14 | Buying power → replaceIfBetter | 5350-5356; 3878-3902 | value > buying_power; cancel the weakest open **entry** order if new score > weakest+10. Note: tier SELL_CSP/T3/T4 orders are tracked with **score 0**, so they're the first to be cancelled | AUDIT `REPLACE`; failure is **SILENT continue** |
| E15 | Instrument execution | 5363-5572 | ETF path; options-scanner direct path; stock→options path **re-decides** with `oe.should_use_options` (oe:212-331: score ≥65, regular hours, options exposure ≤ MAX_TOTAL_OPTIONS_PCT, VRP>2 or "SELL OPTIONS" → CSP; earnings guards). This **can disagree** with the scan's `select_instrument` choice | AUDIT `INSTRUMENT`, `ETF-TRADE`, `OPTIONS-SCANNER-EXEC`, `OPTIONS-TRADE`, `OPTIONS-FALLBACK`, `OPTIONS-SKIP`, `OPTIONS-ERROR` |
| E16 | Stock short block | 5582-5586 | side sell/short → BUY | AUDIT `SHORT-BLOCKED` |
| E17 | Submit | 5589-5605 | – | AUDIT `TRADE … Score: x \| Rules: r ML: m Blend: s \| $v (p%) \| Sizing: …` |
| **Morning queue** (bot:3990-4177), a **weaker** gate set: | | | | |
| M1 | killSwitch | 3994, 4038 | | SILENT |
| M2 | Stale >16h | 3998-4006 | | AUDIT `MORNING-STALE` |
| M3 | Slots | 4039-4042 | **all positions incl. options** ≥ 8 → break | AUDIT `MORNING` |
| M4 | Size | 4043-4046 | > 20% × multiplier | AUDIT `MORNING-BLOCKED` |
| M5 | Gap | 4049-4071 | >5% overnight gap skip; >2% re-qty | AUDIT `MORNING-GAP` / `MORNING-ADJUST` |
| M6 | qty<1 | 4087-4095 | (all options trades land here) | AUDIT `MORNING-SKIP` |
| **Tier dispatcher** (bot:4676-4978): | | | | |
| D1 | killSwitch | 4750 | break | SILENT |
| D2 | Off-hours defer | 4751-4754 → tierDispatchGate.ts:40-49 | SELL_CSP/BUY/BUY_PUT when closed | AUDIT `TIER-DEFER` (one per scan) |
| D3 | Options slots (fresh) | 4695-4707, 4755-4758 | positions + open opening orders ≥ 6 | AUDIT `OPTIONS-SLOT-FULL … tier dispatcher` |
| D4 | CSP swept cooldown / resting STO | 4763-4774 → cspSweepCooldown.ts:118-145 | 60 min (`bot:3777`) | AUDIT `CSP-SKIP` |
| D5 | Contract selection | 4795-4865; oe:338-495, 855-1092 | chain, liquidity, DTE > exit zone (CSP_ENTRY_MIN_DTE, oe:107), affordability vs `min(equity×size_pct, options_buying_power)`, stretch to 20%, qty ≤2 | AUDIT `T1/T2` or `T1-FAIL/T2-FAIL`, `-ERR`; **SHADOW `rejected_capital`** when the error contains "budget"/"capital" (bot:4811-4823) |
| D6 | T3 buy | 4869-4916 | target $ ≤ buying power and price >0 | AUDIT `T3` / `T3-SKIP` / `T3-FAIL` |

---

## 3. COUNTERFACTUAL LOGGING

**Exists:** `shadow_portfolio.py`.

**File:** `DATA_DIR/voltrade_shadow_candidates.json` (sp:76). It is a single JSON array capped at 20,000 records with the newest kept (sp:81, 281-282), rewritten atomically under a lock on every write (sp:172-208). The stats summary goes to `voltrade_shadow_stats.json`.

**Record shape** (sp:260-276):
```
{ticker, timestamp (UTC ISO), score, decision, decision_reason[:200], entry_price, vxx_ratio,
 regime_label, side, action_label, trade_type,
 features{34 ML features as floats}, outcomes{"+5d":null,"+10d":null,"+20d":null},
 code_version:"1.0.34-shadow", [_bf_attempts{}]}
```
- An outcome is filled as `{return_pct, label (1/0/-1), exit_price, exit_date, exit_reason: target|stop|timeout|ambiguous|no_price_data, label_method:"path_dependent_v2"}` (sp:683-690, 512-519).

**Which rules feed it:**

| Decision | Writer | Coverage |
|---|---|---|
| `taken` / `rejected_score` | `deep_score` be:1819-1844 | only deep-scored candidates, and only when analyze.py **and** the ML feature build succeeded (`ml_features` must exist). Threshold fixed at 63 |
| `rejected_heat` | be:3349-3355 (`update_last_decision`, sp:288-338, only moves a ≤120s-old `taken`) | sector/correlation veto (B6) |
| `rejected_other` | be:3427-3434 | spread veto (B11) |
| `rejected_masterkill` | ts:315-377 | T1 CSP proposals blocked by master_kill_switch (score 0, features {}) |
| `rejected_capital` | bot:4811-4823 (inline Python) | T1/T2 CSP contract selection "budget"/"capital" errors (score 0, features {}) |
| `rejected_halt`, `rejected_earnings` | **declared in the docstring (sp:239-241), never written** | – |

**Rules that do NOT feed it** (every other veto in §2):
- Quick-scan floors, deep cap and memory/35s truncation.
- analyze.py failure.
- Slots/NEUTRAL break, cooldown, BEAR/PANIC, regime MIN_SCORE above 63, $50M dollar volume, blocked tickers, extreme mover (it has its own JSON), sizing/Kelly blocks, volume-share cap.
- Kill-switch suppression.
- All options-scanner rejections; covered-call exclusions.
- T1 slot/affordability, T2/T3 gates.
- **Every Node gate** (SKIP, SIZE-CAP, HEAT-CAP, CORR-CAP, HALT, DUP, MAX-RETRY, OPTIONS-SLOT-FULL, stock-slot break, BP).

These all leave the record as `taken`, or leave no record at all.

**Outcome check job:**
- `checkShadowBackfill()` (bot:4281-4313) runs `backfill_outcomes(3000)` once per day at 22:00 UTC, audited as `SHADOW-BACKFILL`.
- Labeling is **path-dependent** (sp:426-493): +2% target / −4% stop over N trading bars, where N = 5/10/20. A same-day touch of both counts as a loss. Unresolved after N bars → label by the close.
- Horizons are **+5d/+10d/+20d, not the +1d/+5d/+20d that CLAUDE.md specifies**.
- It labels a **long-stock** path for every record, including option-sale candidates, `SELL` labels and CSP rejections (directional proxy only, per ts:335-339).
- Missing prices are retried up to 3 nights, then given a terminal `label -1` (sp:496-521).

**Live state** (`/api/diag/shadow`, 2026-10-15):
- 20,000 records spanning 2026-08-03 → 2026-10-14, about 277 per day.
- By decision: `rejected_score` 10,438, `taken` 8,372, `rejected_other` 681, `rejected_heat` 493, `rejected_masterkill` 16, `rejected_capital` 0.
- +5d win rates: taken 51.5% (n=6,938), rejected_score 51.8%, rejected_other 52.2%, rejected_heat 24.1%, masterkill 37.5% (n=16).
- by_regime: NEUTRAL_BULL 19,984, NEUTRAL 16, so the regime label is effectively constant.

**Other partial counterfactual stores:**
- `extreme_movers_today.json` (B9).
- `scripts/illiquid_universe_probe_threshold_ablation.py` (offline research).
- The CSP capital tests (`server/cspCapitalCounterfactualLogging.test.ts`).

There is no generic `{date, ticker, rule, entry_price, score}` logger with a `rule` field. The `decision` bucket is the only "rule" identifier, and it is coarse.

---

## 4. CONVICTION

### 4.1 Stock composite (`deep_score`)

**Sub-scores (0–100):**
- **momentum**: `strategies/momentum.score(mom_12_1, mom_1m, avg_volume)`. 12-1 = 252-bar minus 21-bar return; with 21–251 bars, 12-1 is taken as half the 1-month return (be:1057-1088). Returns 0 for negative momentum (not 50), and 0 with signal "SKIP" when volume is under 100k.
- **mean_reversion**: `strategies/mean_reversion.score(rsi14, real 5d return, volume_ratio_5d)` (be:1091-1110).
- **vrp**: 85 if >8, 70 if >5, 55 if >0, 65 if < −3, else 45 (be:1113-1126). VRP = ATM IV − RV20 from analyze.py.
- **squeeze**: `strategies/squeeze.score(short%, DTC, sentiment, vol ratio, reddit)` (be:1136-1145).
- **volume**: 90 at >3×, 75 at >2×, 60 at >1.5×, 50 at >1×, else 35 (be:1148-1158).

**Rules score** (be:1227-1444):
```
rules = clamp0..100( .25·mom + .20·MR + .25·VRP + .15·squeeze + .15·volume
        + vol_premium(EWMA−RV20 >5 → +8, < −5 → +5)
        + macro(VIX extreme ±5/−10, VIX low ±5, sector ETF momentum ±8, risk_off −5/risk_on +3)
        + news(≥60 +10, ≥20 +5, ≤−60 −12, ≤−20 −6)
        + ADX(>40 +6, >25 +3, <15 −4) + OBV(±5)
        + intel(clamp ±15 of intel_score·0.3; trap → ≤−10; finnhub MSPR ±5 and analyst +3 — ONLY if intel non-empty; sell-the-news −5)
        + alt(clamp ±20) + social(±15, halved if low confidence)
        + fundamentals(rec ±5/−6, target upside +6/+3/−5, fwdPE −4/+3, EV/EBITDA −3/+2)
        + analyze recommendation (+8 BUY / +6 SELL OPTIONS)
        + sentiment (+10 Squeeze Watch / +8 Buy the Dip / +4 if >75)
        + RS vs SPY (>5 → +6) )
```
`rules_only_score = round(rules, 1)` (be:1444).

**ML blend** (be:1448-1775):
- `ml_model_v2.ml_score(34 features)` returns `{ml_score (10..90), ml_confidence, ml_signal BUY>60/SELL<40/HOLD, model_type, win_probability, regime_used}` (ml_model_v2.py:1573-1685).
- Unless `model_type == "fallback"` (never true; the fallback says "fallback_rules"):
  - `ml_only_score = ml`
  - `ml_eff = 50 if |ml−50| < 8 else ml`
  - `final = rules·rule_w + ml_eff·ml_w`
- rule_w/ml_w come from `voltrade_blend_tracker.json` (default .6/.4). That file is written by `diagnostics.run_diagnostics` (diagnostics.py:476-500): ml_w = .4, moving toward .55 when recent WR >60 and toward .20 when recent WR <40. **At the live recent WR of 20%, ml_w ≈ .28 / rule_w ≈ .72.**
- `final` is clamped to 0–100 and stored as `deep_score`.

**Labels** (be:1789-1811), computed on the final score:
- if final > 65 and vrp > 5 → `side:sell, trade_type:options, action_label:"SELL OPTIONS"`
- elif final > 65 and rsi > 70 → `side:sell, action_label:"SELL"` (the scan loop later rewrites this to BUY)
- else `buy / stock / BUY`.

**Feature-quality notes** (relevant for any feature panel):
- `price_vs_52w_high` is always 0, because quick_result has no `high_52w`.
- `range_pct` is always 0.
- `vwap_position` means close > open, not VWAP.
- `intel_score`, `news_sentiment` and `insider_signal` are forced to 0.0, and `days_to_earnings` to 0.5, to match training (be:1728-1735).
- `markov_state` and `regime_score` come from an empty SPY return list.

### 4.2 Thresholds downstream

- Python: `MIN_SCORE` by regime (§2F).
- Node: `minScoreThreshold` 65/75. The Python and Node thresholds differ (Top Finding 6).
- Options: `is_.MIN_OPTIONS_SCORE` 65 and `oe.should_use_options` ≥65 (stock→options); options scanner ≥60 (+5 in a short week).

### 4.3 Options-scanner conviction

Per setup: `score = min(cap, .6·rules + .4·options_ml_score×100)`. The setup's rules score is described in §2D. `options_ml_score` comes from ml_model_v2, or a hardcoded win-rate fallback (os_:81-99). The ±5/−8 surface adjust follows. Covered calls use a fixed 75. T1 CSP has **no conviction score** (only its universe rank; `score:0` in openOrders).

### 4.4 Reasons/signals fields and example content

**`reasons: List[str]`** is built in deep_score. Literal formats, from code:
- `"Momentum: BUY (12mo: +34.2%, 1mo: +6.1%)"`
- `"MeanRev: WATCH (RSI: 38, 5d drop: -4.2%, vol: 1.2x)"`
- `"Squeeze: POSSIBLE (Short: 18.0%, Cover: 6d, Sentiment: 61)"`
- `"High VRP (+9.1%) — sell options edge"`, `"Cheap IV (-4.0%) — buy options edge"`
- `"EWMA vol (31.2%) >> RV20 (24.0%) — elevated vol"`
- `"VIX low (13.2) — favoring momentum"`
- `"Sector tailwind: Technology +2.1%"`
- `"Market regime: risk-off"`
- `"News bullish (7 articles)"`
- `"ADX 31.2 — moderate trend"`
- `"OBV rising — smart money accumulating"`
- `"TRAP WARNING: …"`, `"Insiders heavily buying (n buys)"`, `"Finnhub insider sentiment: bullish (MSPR 42)"`
- `"Wikipedia attention spike: 3.1x normal views"`, `"Google Trends spike: …"`, `"Reddit buzz: high (…)"`, `"WSB hype warning …"`
- `"Analyst consensus: BUY (23 analysts)"`, `"Analyst target: $210 (+18% upside)"`, `"Expensive: Forward P/E 61.2"`
- `"AI recommends: BUY STOCK"`, `"SQUEEZE WATCH — retail hype + high short interest"`
- `"Outperforming SPY by 7.3%"`
- `"ML: uncertain (52), using rules-only"` or `"ML model: BUY (34% confidence)"`, then `"Blend: 72% rules + 28% ML"`

Note: momentum reasons include "AVOID" (only NO DATA/SKIP/NEUTRAL are excluded).

**Other carriers:**
- Options scanner: `reasons:[reasoning]` (one string, e.g. "Earnings in 3d. Options implying 6.2% move. IV rank 71/100 … Strategy: iron_condor. [Surface: VRP +3.1%, premium selling favored]").
- Tier actions: `reason`, e.g. "T1 CSP harvest (regime=NEUTRAL, vxx=0.98, DD=-3.1%, size=100%)".
- Instrument: `instrument_reasoning`, e.g. "CHOSEN: STOCK (buy_stock) | Scores: [stock=58 | options=0] | Stock: score 71, … | Options ruled out: score 64 < 65".
- Sizing: `sizing_reasoning` ("N shares @ $p = $v (x% of portfolio) | Kelly base: … × Vol … × Heat …") and `sizing_scalars{kelly_base, volatility, confidence, regime, earnings, time_of_day, portfolio_heat, liquidity, drawdown_scalar, drawdown_pct, stress, stress_level, stress_value, combined}` (ps:1130-1145).

### 4.5 Is per-candidate reasoning persisted after the scan?

| Store | What survives | Lifetime |
|---|---|---|
| `lastScanResult` → `GET /api/bot/last-scan` (bot:4478, 7772-7774) | full scan dict: `new_trades` with full `reasons`, sub-scores, rules/ML, instrument and sizing; `top_10` (reasons[:3], factors); tier_actions; kill_status; phase times | **in memory, last scan only; lost on restart.** Overnight runs overwrite it too (bot:6024, 6187) |
| `signals` array (bot:4492-4559) | joined reasons string per ticker | in memory, ≤30 entries, ≤1h |
| Shadow JSON | score, decision bucket, 34 features, side/label, entry price, outcomes | 20k rolling (about 72 days) — **no reasons list, no rules/ML split, no sub-scores** |
| Audit SQLite | TIER2 counts; TRADE line (Rules/ML/Blend + sizing ≤150 chars); INSTRUMENT (reasoning ≤120 chars); Node veto lines; T1/T1-FAIL; TIER-KILL; KILL-*; QUEUE/MORNING-* | **rolling 2,000 rows**, 500 chars each. TIER2 alone wrote 200 rows in about 9h on 2026-10-14, and the static RULES/EXECUTION startup prose (bot:7899-7920+, partly stale: "Scan interval 45s", "Minimum score 65", "→ SHORT") adds dozens per restart |
| Fills/feedback (`ml_model_v2.track_fill` ml_model_v2.py:2501-2526) | executed entries only: score, entry_features, code_version, config fingerprint | durable |
| `voltrade_scan_timings.json` (be:2585-2616, 3997-4005; `/api/diag/timings`) | per-phase durations of the last scan | overwritten each scan |

---

## 5. STRATEGY / TRADE TYPES EMITTED

| Type (as emitted) | Producer | Path to broker |
|---|---|---|
| Stock long `BUY` (`instrument:"stock"`, `instrument_strategy:"buy_stock"`) | deep_score + scan loop (be:3238-3597) | Node `executeTrades` stock path (bot:5574-5605) or morning queue |
| 2× leveraged ETF long (`instrument:"etf"`, `buy_etf_bull`) | `is_._score_etf` via select_instrument | `executeTrades` ETF path (bot:5369-5433), qty = floor(shares/2) |
| Stock→options: `sell_cash_secured_put` / `bull_put_credit_spread` / `qqq_iron_condor` / `covered_call` (selector), re-decided as `sell_cash_secured_put` or `bear_put_spread` by `oe.should_use_options` | is_:834-859 then oe:283-301 | `evaluate_and_execute` (bot:5529-5550) |
| `"SELL OPTIONS"` candidate (trade_type options, side "sell" kept at be:3408) whose selector did not choose options | deep_score label | Selector chose stock: falls through to the **stock BUY** path (SHORT-BLOCKED converts it), with stop/TP computed short-side (ps:1110-1112). Selector chose ETF: side≠buy picks the **bear** 2× ETF (is_:566-569), which Node buys (bot:5378). It still counts against **options** slots (bot:5259) |
| `SELL COVERED CALL` (`covered_call`) | sweep be:3621-3705 | effectively dead (Top Finding 5) |
| Options scanner: `earnings_iv_crush` → `iron_condor`; `vxx_panic_put_sale` → `bull_put_credit_spread` (SPY/QQQ); `high_iv_premium_sale` → `iron_condor` (incl. VRP+ Setup 7) | os_:707-1108, 1737-1873 | `executeTrades` scanner path (bot:5440-5526) |
| T1 `SELL_CSP` (`csp_core`) / T2 `leveraged_csp` | ts:407-616 | tier dispatcher (bot:4780-4865) |
| T3 `BUY` SPY/QQQ (`trend_long`) | ts:623-688 | dispatcher (bot:4869-4916); **never produced** (Top Finding 2) |
| T4 `BUY_PUT` (`tail_hedge`) | ts:695-734 | disabled (TIERS_ENABLED[4]=False) |
| Passive floor basket QQQ/SMH/KWEB/VXUS(+cash), GLD defensive floor | be:_manage_spy_floor (4948+), _manage_defensive_floor (5370+) | **Python submits orders directly** |
| Third leg VRP harvest (SVXY) / sector rotation (disabled) | be:_run_third_leg 4218+ | Python direct |
| Convexity overlay (QQQ puts) | be:_run_convexity_overlay 4414+ | disabled (sc:242) |
| Intraday shorts | intraday_shorts.py | disabled (be:3818); only `close_open_shorts` at 15:50 (bot:4646-4659) |
| Disabled or removed | options-scanner Setup 4/5 removed (os_:1596-1612), Setup 6 CSP-normal commented (os_:1664-1671); buy_call/bull_spread routed to stock (is_:861-888, oe:308-322) | – |

**Modules that do not gate entries:**
- `probability_engine.py`: only `/api/system/snapshot` (bot:2171).
- `manipulation_detect.py`: Tier-3 hourly `MANIPULATION` audit (bot:5802-5812). `is_ticker_flagged` has no caller.
- `insights.py`: routes.ts UI only.
- `analyze.py`: used only via `get_stock_details` subprocess (full `analyze_ticker`), instrument_selector imports, and `quick_scan` for overnight extreme movers.

---

## 6. HEALTHY VALUES / FAILURE SIGNATURES

**Live baseline (2026-10-14/15):**
- **Scan count:** `TIER2 Scanned ~12,138 stocks, 2–4 trade candidates (via daemon)` about every 2–15 min. The candidates are **almost all covered calls** (VXUS, KWEB) **plus options-scanner picks** (TSLA, APH, CLF…). All were blocked by `OPTIONS-SLOT-FULL (6/6)` (158 lines between 14:18 and 20:00Z).
- **Book:** 5 floor ETFs and 6 short puts. Stock entries: 0.
- **Timings (after-hours scan at 23:52Z, `/api/diag/timings`):**
  - universe 0.6s, snapshots 4.45s, quick 0.16s.
  - **deep_score 35.67s (35s cap binding)**.
  - Step 6 and options 0.05s (options scanner returns immediately outside regular hours).
  - **tier1 57s** (CSP Layer 2 prefetch `budget_exceeded:true`, 106/150), even though T1 has 0 slots, so the work is wasted.
  - Total 99s.
- **Shadow:** about 277 records/day. The decision mix is the normal distribution described in §3.
- **Diagnostics:** "Low win rate 27.8% over 36 trades", so Node minScoreThreshold is 75 and the blend is about 72/28.
- **`/api/diag/scanner`:** `degraded:false, consecutiveFailures:0, dataSourceErrors:{}`.

**Not visible from diag** (only `/api/bot/last-scan` has them, owner auth, in memory): `filtered`, `deep_analyzed`, `slots_available`, `top_10` scores.

**Failure signatures and whether they surface:**

| Symptom | Meaning | Surfaced? |
|---|---|---|
| "Scanned 0 stocks, 0 trade candidates" | ULTRA mode / DD halt / empty universe | DD-HALT has its own audit; ULTRA only stderr |
| TIER2 "Scan returned error: Could not fetch market data from Alpaca — <detail>" | snapshot entitlement/HTTP failure | yes (audit + /api/diag/scanner) |
| deep_score phase ≈35s every scan | 35s cap truncating; few candidates scored | /api/diag/timings only |
| deep_score phase ≈0s, `deep_analyzed:0`, top_10 scores ≤60 with empty reasons | mem-guard skip or SURVIVAL_MODE, so zero stock trades possible | stderr/PYLOG only; last-scan in memory |
| shadow `newest` timestamp stale | deep_score not completing (analyze.py failing, ML import failing, 35s cap reached before candidate 1) | /api/diag/shadow |
| `data_source_errors` non-empty / `stock_details` key | enrichment or analyze.py failing, so scores collapse toward base | /api/diag/scanner |
| PYLOG `tie-breaking failed: cannot access local variable 'current_positions'` | **every scan** (Top Finding 1) | container logs only |
| TIER2-ERROR "Scan failed via daemon … scan_timings={last_phase=…}" | 300s daemon timeout; the phase tells where | yes |
| `tier_engine_tier1` ≫ 30s, `csp_layer2_prefetch.budget_exceeded:true` | CSP universe recompute slow | timings + daemon health on error |
| `KILL-WARN` repeating, no `KILL-SWITCH` | rks warnings that are not enforced | yes |
| `OPTIONS-SLOT-FULL` floods | slots saturated; options candidates churn every scan | yes (and it crowds the 2,000-row audit) |
| Morning `MORNING-SKIP qty=0` | options/covered-call trades queued off-hours | yes |
| Composite scores clustered in the 25–45 band | normal: the sub-scores often sit at 0–50; reaching 63 needs several bonuses | – |

---

## 7. GAPS FOR A COCKPIT (5-layer chain recoverability today)

| Layer | Recoverable today | Lost / not recorded |
|---|---|---|
| **1. TRIGGER** | Scan cadence and start (`TIER2 Starting scan (interval …)`); STREAM-SIGNAL rows (but non-causal); per-candidate `change_pct_today` in shadow features; quick_score only in in-memory last-scan top_10 | No per-candidate "why woken" record (quick-score rank, mover list membership, options setup trigger such as earnings/VXX/|chg|>2%). Triggers for T1 (universe rank) and covered calls (≥100 shares) aren't recorded. Overnight scans overwrite last-scan |
| **2. FEATURES** | 34 ML features for deep-scored candidates (shadow, about 72 days) and for executed entries (feedback `entry_features`); CSP universe factor breakdown for the top 200 (cache file, current only) | Raw enrichment inputs (VRP/IV, sentiment, intel, alt, social, fundamentals, ADX/OBV), the five strategy sub-scores and quick-scan metrics are not persisted (in-memory last-scan only). Several features are constants (§4.1 notes). Options-scanner inputs (IVR, implied move, OI) exist only in the trade dict |
| **3. REASONING (fired + vetoes, in order)** | Node vetoes as typed audit rows (SKIP, SIZE-CAP, HEAT-CAP, CORR-CAP, HALT-SKIP, MAX-RETRY, DUP-SKIP, OPTIONS-SLOT-FULL, OPTIONS-SKIP, CSP-SKIP, TIER-DEFER, TIER-KILL, KILL-*, DD-HALT, TIER2-LIMIT, MORNING-*); shadow buckets rejected_heat / rejected_other / rejected_masterkill / rejected_capital; extreme_movers_today.json | **Most Python vetoes are silent** (§2B/2D/2E): slots/NEUTRAL break, cooldown, BEAR/PANIC, regime MIN_SCORE, $50M dollar volume, blocked tickers, Kelly/sizing blocks, volume-share (PYLOG), all options-scanner setup rejections, T1/T2/T3 gates, kill suppression (PYLOG). Node stock-slot `break` and BP-replace failure are also silent. The `reasons` list (signals that fired) is never persisted; there is no rule-order trace and no stable rule IDs. Audit retention is about 1 day |
| **4. CONVICTION** | Final score (shadow and feedback); TRADE audit line has Rules/ML/Blend for executed stock trades; blend weights (current only); options scanner score in the trade | rules_only/ml_only split for non-executed candidates; ML confidence/signal/model_type (incl. fallback masquerade); the per-bonus decomposition behind the rules score; historical blend weights; Node threshold in force at decision time (only the DIAGNOSTIC line implies it) |
| **5. ACTION** | Audit TRADE/ETF-TRADE/OPTIONS-TRADE/OPTIONS-SCANNER-EXEC/T1/T2/T3/MORNING-TRADE/QUEUE; Alpaca orders and fills; feedback records | Link from an action back to its scan or candidate record (no candidate ID or scan ID join key; join only by ticker + time). Python-side floor and third-leg orders bypass the Node audit path |

**Minimum instrumentation a cockpit needs** (for whoever builds it; these are not proposals to edit frozen paths):
- A `scan_id` and per-candidate `decision_trace[]` of `{rule_id, passed, threshold, value}` emitted from the existing loop.
- Persisting that trace along with `reasons` and the rules/ML split (a JSONL append next to the shadow log).
- Making the shadow `taken` label final only after the Node admission result.
