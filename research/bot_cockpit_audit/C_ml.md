# C — FEATURE PIPELINE + ML LAYER: ground-truth audit

Repo: /home/user/voltradeai at the current checkout (2026-10-15). Every claim below cites file:line.
Where the code disagrees with a comment, docstring, CLAUDE.md, or an endpoint label, that is marked **DISAGREES**.
"Confirmed" means read in code end to end. "Likely" means it depends on an external API schema I could not run here (lightgbm and sklearn are not installed in this sandbox).

---

## 0. TOP FINDINGS (read first)

1. **Train/serve skew is severe. About 10 of the 34 features are constants at serve time**, and several more are computed with a different formula than in training (§1.3). The most consequential case: `vxx_ratio` is read from `intel` (`bot_engine.py:1702`), but `intelligence.get_full_intelligence()` returns no `vxx_ratio` key (`intelligence.py:557-565`). So **vxx_ratio = 1.0 for every live prediction**. The real value exists in `macro["vxx_ratio"]` (`macro_data.py:266`) but is not used here. Consequences:
   - `ml_score` can never route to the `bull` regime model, because bull needs vxx ≤ 0.95 (`regime_util.py:69`).
   - Markov `regime_score` is pinned to [43.5, 68.5].
   - `regime_label` is only ever NEUTRAL or NEUTRAL_BULL.
2. **The rule-based fallback is blended as if it were ML.** The guard is `if ml_result.get("model_type") != "fallback"` (`bot_engine.py:1745`), but the fallback returns `"model_type": "fallback_rules"` (`ml_model_v2.py:1712`). So whenever the model is missing or older than 24 h, a hand-written heuristic gets the ML weight (default 40%) and is recorded as `ml_only_score`/`ml_score_raw`.
3. **The ML toggle is cosmetic.** `ml_toggle.json` is read only by `ml_status.py:21-26` and the daemon fallback (`voltrade_daemon.py:508-514`). Nothing in scoring reads it (`deep_score`, `ml_score`, `options_scanner`). ML always blends, but the UI shows ON/OFF from this flag (`client/src/pages/bot.tsx:760,775`). The default when the file is missing is `enabled:false`, which reads "OFF".
4. **The "4am daily retrain" never runs.** `runOvernightResearch` is only called when `etHour >= 20 || etHour < 4` (`server/bot.ts:5208-5209`). The retrain block inside it requires `Math.floor(etHour) === 4` (`server/bot.ts:6199`). The only automatic retrain path is Tier 3 hourly when model age > 24 h (`server/bot.ts:5713-5798`). `/api/bot/ml-status` still advertises `'retrain_schedule': 'Daily at 4am ET'` (`server/bot.ts:8000`).
   - Cross-stage note: the same unreachable block holds the daily state reset and the nightly backup (`server/bot.ts:6200-6268`).
5. **The retrain is hard-killed at 90 s, not 900 s.** `execPythonSerialized` caps `hardTimeout = Math.min(opts.timeout || 90000, 90000)` (`server/bot.ts:241`). The `{timeout: 900000}` passed at `server/bot.ts:5739` only lengthens the mutex wait. The comment there claiming 15 min of headroom is wrong. If killed, the audit reads `TIER3-ML-ERROR ... signal=SIGKILL [likely OOM kill ...]` (`server/bot.ts:5785-5786`), which is misclassified.
6. **"Purged walk-forward CV" is not temporal.** `_build_training_data` appends rows ticker by ticker (`ml_model_v2.py:835-926`). So `_walk_forward_cv` folds (`:711-740`) and the "last 20% = test" holdout (`:689-708`) split by ticker blocks, not by date. Train and test share the same calendar dates. The market-wide features (VXX, SPY, credit, breadth, put_call_ratio, vol_of_vol) act as date fingerprints, so accuracy is likely inflated. The embargo of 5 × n_tickers rows assumes date-interleaved rows and has no temporal meaning here.
7. **Self-learning rows mostly never reach the production model.** Feedback and shadow rows are appended at the end of `X_all` (`:1354, :1379`). The production model trains only on `X_tr = X_all[:train_end]`, with test = last 20% (`:1467-1475`). When feedback + shadow are under about 20% of rows, they land entirely in the holdout, and the 3× decay weighting never applies.
8. **Open or unmatched trades are trained as losses.** `_load_trade_feedback` keeps any record with `entry_features` regardless of `outcome` (`:1111-1121`). `_build_feedback_training_data` turns `pnl_pct=None` into 0, which becomes label 0 (`:1152-1161`). This is the exact Bug #13 failure mode, still present for open entries.
9. **Dead ML machinery that docs present as live:**
   - meta-model: `train_meta_model` and `meta_score` have zero call sites.
   - options model training: only reachable via the manual `python3 bot_engine.py train` path, so `options_ml_score` effectively always uses rule fallback.
   - `ml_attribution_summary`: zero call sites, and it reads `rules_score`/`ml_score_raw`, which track_fill never writes.
   - config-fingerprint similarity weighting: `_config_similarity_weight` is never called.
   - `ML_RETRAIN_TRADES`, `ML_MIN_SAMPLES`, `ML_TARGET_HORIZON`, `ML_TARGET_RETURN`, `ML_CONFIDENCE_THRESH`: never read.
10. **No training-feature distribution baseline file exists.** The only persisted training statistics are each `StandardScaler`'s `mean_`/`var_` inside `voltrade_ml_v2.pkl` (global plus one per regime model), plus gain feature importances. No predictions are logged anywhere, and `model_type` (real model vs fallback) is never persisted or surfaced.

---

## 1. FEATURE PIPELINE

### 1.1 Canonical feature list — `FEATURE_COLS` (`ml_model_v2.py:140-184`, asserted 34 at `:186`)

The order below is the order of the model input vector.

| # | key | # | key |
|---|---|---|---|
| 1 | momentum_1m | 18 | cross_sec_rank |
| 2 | momentum_3m | 19 | earnings_surprise |
| 3 | rsi_14 | 20 | put_call_proxy |
| 4 | price_vs_52w_high | 21 | vol_of_vol |
| 5 | volume_ratio | 22 | frac_diff_price |
| 6 | ewma_vol | 23 | idiosyncratic_ret |
| 7 | atr_pct | 24 | change_pct_today |
| 8 | vrp | 25 | above_ma10 |
| 9 | iv_rank_proxy | 26 | trend_strength |
| 10 | vwap_position | 27 | intel_score |
| 11 | adx | 28 | news_sentiment |
| 12 | range_pct | 29 | insider_signal |
| 13 | vxx_ratio | 30 | iv_rank_stock |
| 14 | spy_vs_ma50 | 31 | days_to_earnings |
| 15 | markov_state | 32 | credit_spread |
| 16 | regime_score | 33 | market_breadth |
| 17 | sector_momentum | 34 | put_call_ratio |

- `system_config.BASE_CONFIG["ML_FEATURE_COUNT"] = 34` (`system_config.py:278`). It is pinned by `test_audit_critical.py:27`.
- `ml_model_v2`'s import-failure fallback says 26 (`ml_model_v2.py:67`).
- `OPTIONS_FEATURE_COLS` = `FEATURE_COLS + ["iv_rank","days_to_earn","vrp_magnitude"]`, 37 features (`:1905-1909`). **DISAGREES:** `options_scanner.py:111` says "28-feature dict".

### 1.2 Training feature computation — `_compute_features(bars, idx, ...)` (`ml_model_v2.py:364-615`)

Inputs per row:
- the ticker's daily bars (Alpaca `/v2/stocks/bars`, `adjustment=all`, feed = `bars_feed()`, `:281-285`)
- VXX/SPY/TLT/HYG bars, calendar-date-aligned to the ticker (`_aligned`, `:847-874`)
- `earnings_surprise` per ticker
- `breadth_pct` per date
- `cross_sec_rank` (overridden after the call, `:890-896`)
- `news_sentiment=0.0` and `insider_signal=0.0`, always constant (`:878-885`)

Row rejection rules:
- `idx < 25`, or `idx >= len(bars)-5` (`:376`). The outer loop is `range(25, len(bars)-6)` (`:876`).
- `close <= 0`, or `volume[idx] < 300_000` (`:386`).

"Today" is bar `idx` at its close. All outputs are rounded (`:578-615`).

| feature | training formula (file:line) | clip / range in training |
|---|---|---|
| momentum_1m | (c − close[idx−21]) / close[idx−21] × 100; 0 if idx < 21 (`:390`) | unclipped % |
| momentum_3m | same with idx−63; 0 if idx < 63 (`:391`). About 1 year of bars, so roughly 15% of rows are forced to 0 | unclipped % |
| rsi_14 | simple-average RSI over `closes[idx−15:idx]` (excludes today's close); 50 if `al == 0` (`:408-415`) | 0–100 |
| price_vs_52w_high | (c − max(highs[idx−252:idx])) / that × 100 (`:418-419`). Window is capped by the ~250-bar history in fast mode | can be > 0 (today excluded) |
| volume_ratio | vol[idx] / mean(prior 20 nonzero vols) (`:394-396`) | clipped at 10 (`:583`) |
| ewma_vol | **plain std** of ~20 prior daily returns × 100, not EWMA; default 2.0 (`:399-400`). DISAGREES with comment "EWMA realized vol" (`:149`) | daily % |
| atr_pct | mean true range over bars idx−14..idx−1 / c × 100; default c × 2% (`:402-405`) | % |
| vrp | **VXX proxy:** clamp((vxx_ratio − 1) × 50, −20, 20) (`:461`). Not IV − RV | [−20, 20] |
| iv_rank_proxy | VXX percentile in its prior-252-bar min/max (`:456-460`) | 0–100 |
| vwap_position | 1 if close > open else 0 (`:486`) | {0, 1} |
| adx | DX proxy: \|ΣΔhigh⁺ − ΣΔlow⁺\| / (sum) × 100 over 14 bars; 20 if idx < 14 (`:493-500`) | 0–100 |
| range_pct | (h − l) / c × 100 of bar idx (`:487`) | % |
| vxx_ratio | VXX close / mean(prior 30 VXX closes) (`:442-447`) | ~0.7–1.6 |
| spy_vs_ma50 | SPY close / mean(prior 50) (`:449-454`) | ~0.85–1.10 |
| markov_state | 2 if mean of last 5 SPY returns > 0.002, 0 if < −0.002, else 1 (`:467-475`) | {0, 1, 2} |
| regime_score | clamp((spy − 0.94)/0.12 × 50 + (1.3 − vxx)/0.6 × 30 + 20, 0, 100) (`:463`) | 0–100 |
| sector_momentum | stock 5-day % return − SPY 5-day % return (`:479-483`) | % |
| cross_sec_rank | percentile of today's % change among the training universe on the same date (if > 10 names), else 0.5 (`:890-896`) | 0–1 |
| earnings_surprise | Finnhub `surprisePercent` of the **most recent** quarter: > 2 → +1, < −2 → −1, else 0 (`:297-309`). Fetched only for `ticker_list[:30]` (`:1326-1327`) and applied to every historical row of that ticker, which is a **look-ahead** | {−1, 0, 1} |
| put_call_proxy | −1 if vrp > 8, +1 if vrp < −5, else 0 (`:543`) | {−1, 0, 1} |
| vol_of_vol | std of 10 VXX daily returns × 100; 2.0 if fewer than 5 (`:505-510`) | % |
| frac_diff_price | `_frac_diff(closes[idx−20:idx])`, d = 0.4, 20 weights, clip ±3 (`:120-135`, `:516`). Weight sum ≈ 0.2055, so the value ≈ 0.2055 × ln(price) + small terms. It is effectively a **log-price-level** feature (≈ 0.33 at $5, 1.65 at $3000) | [−3, 3] |
| idiosyncratic_ret | 21-day stock return − β × 21-day SPY return, with β = 1.2 if volume_ratio > 2 else 0.8 (`:523-540`) | % |
| change_pct_today | (c − prev) / prev × 100 (`:488`) | % |
| above_ma10 | c > mean(prior 10 closes) (`:489`) | {0, 1} |
| trend_strength | min(\|change\| / max(ewma_vol, 0.1), 5) (`:490`) | 0–5 |
| intel_score | 0.4 × 0 + 0.3 × 0 + 0.3 × earnings_surprise (`:546`). So **0.3 × earnings_surprise** in training | {−0.3, 0, 0.3} |
| news_sentiment | always 0.0 (`:881`) | 0 |
| insider_signal | always 0.0 (`:882`) | 0 |
| iv_rank_stock | min(100, iv_rank(VXX) × min(volume_ratio, 3)/2) (`:550-551`) | 0–100 |
| days_to_earnings | always 0.5 (`:555`) | 0.5 |
| credit_spread | (TLT 21d return − HYG 21d return), aligned bars (`:558-570`) | ~±0.1 |
| market_breadth | share of training-universe tickers above their own 50d MA on that date; 0.5 if missing (`:816-833`, `:877`) | 0–1 |
| put_call_ratio | VXX close / 20 (`:576`) | ~0.5–3 |

There is no other normalization. A `StandardScaler` is fitted on the training split inside `_train_single_lgbm` (`:1203-1205`).

### 1.3 Live (serve) feature build — `bot_engine.deep_score` (`bot_engine.py:1448-1740`), and train/serve skew

The live dict `ml_features` is assembled at `bot_engine.py:1687-1739` and scored at `:1740`. Sources:
- `detail` = `analyze.py TICKER --mode=scan`. The `--mode` argument is ignored: `analyze.py:2647-2660` routes any non-"SCAN" first argument to `analyze_ticker`.
- `vol_metrics` = `detail["vol_metrics"]` (`bot_engine.py:971`)
- ~300 daily bars `_deep_closes` (`:986-996`)
- `quick_result` from the snapshot quick scan (`:2846-2862`)
- `macro` = `get_macro_snapshot()`
- `intel` = `get_full_intelligence()`

The keys are identical to training (all 34), but values differ.

| feature | serve source (bot_engine.py) | skew vs training |
|---|---|---|
| momentum_1m | `mom_1m` = (c[−1]/c[−21] − 1) × 100 (`:1068-1073`, `:1679`) | ~match (off by one bar; intraday partial bar) |
| momentum_3m | (c[−1]/c[−63] − 1) × 100 (`:1681-1685`) | ~match |
| rsi_14 | `vol_metrics["rsi_14"] or 50` (`:973`, `:1691`; analyze `compute_rsi`, `analyze.py:2144`) | method differs (includes today) |
| volume_ratio | `vol_metrics["volume_ratio_5d"]` = mean(5d vol) / mean(90d vol) (`analyze.py:250-253`; `bot_engine.py:974`, `:1692`) | **DIFFERENT definition** (5d/90d vs today/20d) |
| vwap_position | `quick_result["above_vwap"]` = close > open (`:2860`, `:1693`) | match |
| adx | Wilder-smoothed DX over full history (`:1002-1024`); 20 if None | **DIFFERENT formula** |
| ewma_vol | `vol_metrics.get("ewma_vol", 2)` (`:1695`). analyze puts `ewma_rv`, never `ewma_vol` (`analyze.py:2148`, `:2499`) | **CONSTANT 2** |
| range_pct | `quick_result["range_pct"]`, hard-coded 0 (`:2861`, `:1696`) | **CONSTANT 0** |
| price_vs_52w_high | `quick_result.get("high_52w", price)` (`:1464-1465`). quick_result never has `high_52w` (only `:1464` references it) | **CONSTANT 0** |
| vrp | `detail["vrp"]` = atm_iv − rv20 in vol points; 0 if None (`:967`; `analyze.py:568`) | **DIFFERENT meaning** (real VRP vs VXX proxy) |
| iv_rank_proxy | `_compute_stock_iv_rank(ticker, closes)`: stock's 30d-HV rank (`:20-33`, `:1700`) | **DIFFERENT** (stock HV vs VXX) |
| atr_pct | Wilder ATR14 / price × 100; default 3.0 (`:1489-1491`) | ~match (method differs) |
| vxx_ratio | `intel.get("vxx_ratio", 1.0)` (`:1702`). intel has no such key (`intelligence.py:557-565`) | **CONSTANT 1.0** |
| spy_vs_ma50 | `macro["spy_vs_ma50"]` (`:1704`; `macro_data.py:311`, includes today) | ~match |
| markov_state | `_get_markov_regime(_spy_rets=[], ...)` (`:1453-1458`). `_spy_macro` is computed but never passed. `get_current_state` returns 1 when < 3 returns (`markov_regime.py:128-129`) | **CONSTANT 1** |
| regime_score | Markov `get_full_regime_score` with vxx = 1.0 → 26.0 + 17.5 + clamp((spy − 0.94)/0.12 × 25, 0, 25) (`markov_regime.py:193-203`) | **DIFFERENT formula, range [43.5, 68.5]** |
| sector_momentum | `macro["sector_momentum"][SECTOR_MAP[ticker]]` = sector ETF 1-day % change (`macro_data.py:158-171`; `bot_engine.py:1192-1193`, `:1707`); "Other" → 0 | **DIFFERENT definition** |
| cross_sec_rank | percentile vs `s.get("change")` in Alpaca most-actives top 100 (`:1539-1560`) | **Likely CONSTANT 0.5.** Alpaca's documented most-actives rows are {symbol, volume, trade_count}; `change` belongs to the movers endpoint. Unverified here |
| earnings_surprise | `intel.get("earnings_surprise", 0)` (`:1709`). intel has no such key | **CONSTANT 0** |
| put_call_proxy | same thresholds applied to the real vrp (`:1493`) | thresholds were tuned for the proxy vrp |
| vol_of_vol | std of last 10 VXX returns × 100 from `macro["vxx_closes"]` (≤ 40 bars); 2.0 fallback (`:1500-1528`) | ~match |
| frac_diff_price | `_frac_diff(_deep_closes)`: last 20 closes incl. today (`:1531-1533`) | ~match |
| idiosyncratic_ret | change_pct − `macro.get("spy_change_pct", 0)` (`:1494`). macro never sets `spy_change_pct`, so it **equals change_pct_today** | **DIFFERENT** (1-day, no beta, vs 21-day β-adjusted) |
| change_pct_today | `quick_result["change_pct"]` (snapshot, intraday) | timing (intraday vs close) |
| above_ma10 | last close > mean(last 10, incl. today) (`:1468-1471`) | ~match |
| trend_strength | min(\|change\| / max(ewma_vol = 2, 0.1), 5) (`:1718`) | = \|change\|/2 because ewma is constant |
| intel_score, news_sentiment, insider_signal | forced 0.0 on purpose (`:1720-1732`) | training intel_score = 0.3 × surprise, not 0 |
| iv_rank_stock | `_compute_stock_iv_rank` (`:1587`, `:1734`) | **identical to iv_rank_proxy at serve** |
| days_to_earnings | forced 0.5 (`:1737`); the computed `_days_to_earnings` (`:1597-1614`) is unused | constant both sides (dead feature) |
| credit_spread | TLT/HYG 21d spread, 60 s cache (`:1619-1644`) | ~match |
| market_breadth | `macro.get("market_breadth_ma50")`, which is never set by macro_data, so the proxy 0.5 + (spy_vs_ma50 − 1) × 6 clamped [0, 1] applies (`:1656-1668`) | **DIFFERENT: a deterministic function of spy_vs_ma50** |
| put_call_ratio | `macro["vxx_latest"]` / 20, default 25 → 1.25 (`:1675-1676`) | ~match. Macro VXX bars are raw (no `adjustment`) while training uses `adjustment=all` |

**Serve-time feature coverage today:**
- Effectively constant or identical: vxx_ratio, markov_state, ewma_vol, range_pct, price_vs_52w_high, earnings_surprise, intel_score, news_sentiment, insider_signal, days_to_earnings, cross_sec_rank (likely). That is about 11/34.
- Deterministic functions of other features: regime_score, market_breadth, idiosyncratic_ret, iv_rank_stock, trend_strength.
- Different definition than in training: volume_ratio, adx, vrp, iv_rank_proxy, sector_momentum, put_call_proxy.

**Silent failure inside the ML block.** The whole block is wrapped in `try … except Exception: pass` (`bot_engine.py:1448`, `:1774-1775`). Any exception (import failure, etc.) silently leaves the score rules-only. `ml_features` then stays undefined, so `entry_features` = None (`:1869`).

### 1.4 Ranges / persisted baselines

- **No baseline file** exists for training feature distributions (no quantiles, min/max, or null rates).
- The only persisted statistics are `bundle["scaler"].mean_` and `.var_` (34 features, global model training split) and `bundle["regime_models"][r]["scaler"]` (per regime), inside `voltrade_ml_v2.pkl` (`ml_model_v2.py:1203-1205`, `:1496`, `:1512-1530`). Also `bundle["feature_importance"]` (LightGBM gain, global model only, `:1503-1509`).
- **Serve-side feature history IS persisted, incidentally:**
  - `shadow_portfolio.log_candidate(features=ml_features, ...)` writes every deep-scored candidate's 34-feature dict with a timestamp to `voltrade_shadow_candidates.json` (cap 20,000 records) (`bot_engine.py:1817-1843`; `shadow_portfolio.py:81`, `:262-283`).
  - `entry_features` on trade_feedback entry records.
- Nothing persists ML outputs (`win_probability`, `model_type`, `regime_used`).

---

## 2. MODEL

**Architecture.** A "regime-conditional ensemble": one global model plus up to 3 regime models (bull / bear / neutral). Each regime needs ≥ 50 train and ≥ 10 test rows of that regime (`ml_model_v2.py:1488-1497`). Each model is `StandardScaler` → LightGBM Booster via `_train_single_lgbm` (`:1197-1248`).

LightGBM params (`:1212-1220`):

| param | value |
|---|---|
| objective | binary |
| metric | binary_logloss |
| learning_rate | 0.04 |
| num_leaves | 24 |
| min_child_samples | 10 |
| feature_fraction | 0.75 |
| bagging_fraction | 0.75 |
| bagging_freq | 5 |
| verbose | −1 |
| scale_pos_weight | max(1, n_neg/n_pos) |
| min_gain_to_split | 0.01 |

- `num_boost_round=300` with `early_stopping(30)` on the **test split** (`:1226-1229`). The reported accuracy is therefore on the same set used for early stopping.
- **Spread gate:** if p90 − p10 of test probabilities is < 0.08, the LGB model is discarded and sklearn `GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.08, subsample=0.8, min_samples_leaf=8)` is used instead (no random_state) (`:1233-1248`).
- **DISAGREES:** `ml_score` still labels such a model `"lightgbm_global"` / `"lightgbm_{regime}"` (`:1618-1642`).
- Regime models are trained **without** sample weights (`:1494`).

**Target.** Binary: 1 = triple-barrier win; 0 = loss, neutral timeout, or stop (§3).

**Calibration.** An `IsotonicRegression(out_of_bounds='clip')` is fit on the concatenated walk-forward out-of-sample probabilities (≥ 50 rows) and saved to `voltrade_ml_calibrator.pkl` (`:1455-1464`). It is applied to whichever model scores (global or regime), although it was fit on global walk-forward fold models.

**`ml_score(features_dict)` output keys** (`:1676-1683`):
- `ml_score` = round(p × 100, 1), where p is clamped to [0.10, 0.90] both before and after calibration (`:1656`, `:1669`)
- `ml_confidence` = round(\|p − 0.5\| × 2, 3), at most 0.8
- `ml_signal`: "BUY" if p > 0.60, "SELL" if p < 0.40, else "HOLD"
- `model_type`: "lightgbm_global" \| "lightgbm_bull" \| "lightgbm_bear" \| "lightgbm_neutral"
- `win_probability` = round(p, 3)
- `regime_used`: "global" \| "bull" \| "bear" \| "neutral"

DISAGREES: the docstring at `:1583` omits `regime_used`.

Regime routing uses `_classify_regime(features["vxx_ratio"], features["spy_vs_ma50"])` → `regime_util.classify_regime` (`regime_util.py:51-71`):
- bear if vxx ≥ 1.15 or spy < 0.94
- bull if vxx ≤ 0.95 and spy ≥ 0.98
- else neutral

Live vxx is always 1.0, so routing is **bear iff spy_vs_ma50 < 0.94, otherwise neutral; never bull**.

Booster prediction goes through `_get_prob` (`:1621-1635`). `hasattr(mdl, "feature_name_")` is the sklearn-wrapper attribute; per the LightGBM API, Booster has `feature_name()` instead. So Boosters fall to the last-resort `mdl.predict(X_sc)`, which still returns probabilities for a binary objective. The result is functionally correct, but the commented branch is dead. (Not verifiable here because lightgbm is not installed.)

**`_rule_based_fallback(features)`** (`:1689-1715`):

```
score = 50 + min(mom_1m × 1.5, 15) + min((volume_ratio − 1) × 8, 10)
           + (regime_score − 50) × 0.2 + min(idiosyncratic_ret × 0.5, 8)
           + 8 (if rsi < 35) − 5 (if rsi > 70)
```

- No lower bound on the momentum, volume, or idio terms. The final score is clamped to [0, 100].
- Output: `ml_score` = score; `win_probability` = score/100; `ml_signal` BUY ≥ 62, SELL ≤ 38; `model_type` = **"fallback_rules"**; `regime_used` = "fallback".

**Fallback triggers** (`:1586-1605`, `:1649-1653`, `:1685-1686`):
- `MODEL_PATH` missing
- model file mtime age > `BASE_CONFIG["ML_MAX_AGE_HOURS"]` = **24** (`system_config.py:285`; the code default is 26 if the key is absent)
- `HAS_SKLEARN` false
- bundle is None
- global-model predict raises
- any exception (e.g., joblib.load of a partially written pkl, since `joblib.dump` is not atomic, `:1530`)

**Meta model (DEAD).**
- `train_meta_model` / `_train_meta_model_impl` (`:1722-1872`) and `meta_score` (`:1875-1898`) have **zero call sites** in runtime code (repo-wide grep).
- Design: 5 features `[primary_score, regime_score, iv_rank_proxy, vxx_ratio, regime_stability]`; label pnl > 0; needs ≥ 100 feedback trades with score ≥ 55; LGB num_leaves 8, lr 0.05, 100 rounds; fallback `LogisticRegression(C=1.0)`; 3-fold walk-forward; written to `voltrade_ml_meta.pkl`.

**Options model (effectively DEAD in production).**
- `train_options_model` (`:2110-2253`) is reachable only from `bot_engine.py:5674-5677` (inline-train path, gated off by default, §6) and dev scripts.
- Design:
  - synthetic VXX/SPY regime labels (`:1912-2036`)
  - plus options feedback ×3. The filter needs `instrument=="options"` or `use_options`, which no track_fill entry record carries (§8), so this is always empty.
  - LGB lr 0.04, num_leaves 20, min_child_samples 10, feature/bagging fraction 0.8 / freq 5, min_gain 0.005, 250 rounds, early stop 25
  - GBM(200, depth 4, lr 0.06, subsample 0.8, min_samples_leaf 8, random_state 42) fallback
  - metric AUC × 100, reported as "accuracy"
  - written to `voltrade_ml_options.pkl`
- `options_ml_score` (`:2056-2107`):
  - Uses the model only if the file exists and is < 7 days old.
  - Applies the **stock** model's `CALIBRATOR_PATH` to the options output (`:2081-2087`), which is a mismatch.
  - Otherwise uses rule fallback: 0.70 (vxx ≥ 1.30), 0.67 (earnings within 1–7 days and iv_rank > 60), 0.65 (iv_rank > 70), 0.55 (iv_rank < 20), else 0.50, minus a vol-of-vol penalty, floor 0.40.
- Consumers: `options_scanner._options_ml_score` (`options_scanner.py:81-99`) in 4 setups with a **fixed** blend:

| setup | blend (rules / ML) | location |
|---|---|---|
| earnings_iv_crush | 60/40 | `:813-823` |
| vxx_panic_put_sale | 60/40 | `:925-931` |
| high_iv_premium_sale | 60/40 | `:1080-1086` |
| csp_normal_market | 70/30 | `:1206-1212` |

  The blend tracker is not used here.

---

## 3. LABELING (exact)

**Stock, generic data:** `_triple_barrier_label(bars, idx, atr_pct, pt_mult=1.5, sl_mult=1.0, max_days=5)` (`ml_model_v2.py:622-658`).

- entry = `close[idx]`
- upper barrier = entry × (1 + max(0.005, atr_pct × 1.5/100))
- lower barrier = entry × (1 − max(0.003, atr_pct × 1.0/100))
- `atr_pct` is the feature (mean TR of the prior 14 bars, %)
- Vertical barrier: bars idx+1 … idx+5 (5 trading bars). Returns 0 if `idx + 5 >= len(bars)`.
- Touch test per bar: **high ≥ PT checked first → +1**, then **low ≤ SL → −1**. A same-bar touch of both labels **+1** (optimistic). Touches use bar **high/low**, not close.
- Timeout relabel (`:902-914`): if the label is 0, compute pnl = close[idx+5] vs entry.
  - pnl > +0.3 × atr_pct → +1
  - pnl < −0.3 × atr_pct → −1
  - otherwise 0
- Binary target: y = 1 iff label == +1, else 0 (`:917`).
- `bundle["label_method"] = "triple_barrier_volatility_adjusted_timeout_relabel"` (`:1524`).
- **DISAGREES:** the docstring "Maps directly to actual bot behavior" (`:635`) is false. Live exits use min TP 12%, max stop 6%, 10-day time stop (`system_config.py:269-275`), not 1.5/1.0 ATR over 5 days.

**Feedback rows** (`_build_feedback_training_data`, `:1134-1190`):
- y = 1 iff pnl_pct > 1.5 × entry_features.atr_pct (default 2.0); else 0.
- pnl None → 0 → label 0. This includes **open trades**.
- \|pnl\| > 20 dropped.

**Shadow rows** (`shadow_portfolio.load_shadow_data(horizon_days=10, shadow_weight=0.3)`, `:714-790`; labels from `_label_from_path`, `:425-492`):
- Fixed **+2% target / −4% stop**, 10 trading bars.
- A same-bar double touch → 0 (conservative).
- Timeout → 1 only if close ≥ +2%.

So **three different label definitions** are mixed in one training matrix.

---

## 4. CV (exact)

**`_walk_forward_cv(X, y, n_splits=5, embargo_days=5, n_tickers_approx=_n_tickers)`** (`ml_model_v2.py:711-740`, called at `:1417-1418`):
- `fold_size = n // 6`
- For k = 1..5: train rows `[0, k·fold)`; embargo = 5 × n_tickers rows; test rows `[k·fold + embargo, (k+1)·fold)`.
- Skip a fold if `test_start >= n` or `train_end < max(500, fold_size)`.
- Skip folds with < 50 train or < 10 test rows (`:1426`).
- Metric: **accuracy at 0.5**, averaged across folds → `wf_accuracy`. Default 0.5 if no folds (`:1451`).
- Per-regime fold accuracy (≥ 5 rows per regime per fold, `:1443-1450`).
- No AUC, logloss, or Brier score is computed.

`_n_tickers` = count of `all_bars` keys excluding SPY, QQQ, IWM, VXX, GLD, TLT (`:1326`, `:1416`), about 70–80.

**Production holdout `_purged_train_test_split(X, y, embargo_periods=5, n_tickers_per_date=_n_tickers)`** (`:665-708`):
- test = last 20%; embargo rows = min(5 × n_tickers, n//5); train = rows before `test_start − embargo`.
- If train_end ≤ 50, a plain 70/30 split.
- There is **no purge** of overlapping label windows. Only the embargo gap exists.

**How tickers interleave: they don't.** Rows are appended per ticker, then per bar (`:835-926`). Then feedback rows are appended, then shadow rows (`:1354`, `:1379`). In fast mode the last 10,000 rows are kept (`:1404-1410`). DISAGREES: the comment calls these "most recent contiguous 10K rows"; they are actually the last tickers' blocks plus feedback plus shadow.

Therefore:
- Folds and the holdout are **cross-ticker splits over the same dates**, not temporal.
- The embargo has no temporal meaning.
- Date-level features leak the outcome of a given date across tickers.
- `bundle["cv_method"]="walk_forward_5fold_purged"` (`:1525`) overstates what the code does.

**Reported metrics:**
- `train_model` returns `accuracy` = wf_accuracy × 100 (`:1554`) and `wf_fold_accs` (`:1563`).
- `bundle["accuracy"]` = wf_accuracy as a fraction (`:1517`).
- `regime_accs` = walk-forward per-regime (%) or holdout per-regime (`:1499-1500`, `:1518`).

---

## 5. TRAINING DATA

**Universe** (`_fetch_training_bars`, `:206-294`):
- Alpaca screener most-actives `by=volume&top=50` (today's list: selection/survivorship bias), plus a 30-name core list: AAPL, MSFT, NVDA, AMD, META, GOOGL, AMZN, TSLA, JPM, BAC, V, MA, COIN, MSTR, PLTR, CRWD, HOOD, SOFI, AFRM, UPST, SPY, QQQ, IWM, VXX, GLD, XLE, XLF, XLK, TLT, HYG (`:249-251`).
- Deduplicated and capped at `max_tickers`.
- Reference-only (no rows): SPY, QQQ, IWM, VXX, GLD, TLT, HYG (`:798`, `:836`). XLE, XLF, XLK are trained as stocks.
- Bars are fetched 10 symbols per request, `limit: 10000`, `adjustment: all`, feed `bars_feed()` (`:278-293`). There is no `next_page_token` handling. That is harmless in fast mode (~250 bars × 10 symbols), but truncates in full mode (2020 start).
- `bars_feed()` returns "iex" when the account is in delayed_sip state (`alpaca_feed.py:113-136`). IEX volume undercounts consolidated volume roughly 30–50×, and `_compute_features` drops bars with volume < 300,000 (`:386`). In that state, most bars of most tickers are filtered out of training: a selection bias that depends on account state.

**Lookback (production path is `fast_mode=True` via `ml_retrain_safe.py:69`):**
- `max_tickers = min(BASE_CONFIG.get("ML_MAX_TICKERS", 200), 100)` = 100 (the key is absent from config).
- `days = min(ML_LOOKBACK_DAYS = 60, 730)` = 60.
- `start = now − max(days, 365)` = **365 calendar days** (`:238-240`, `:267-269`).
- **DISAGREES:** config says "Training window: 60 days of history" (`system_config.py:281`).
- Non-fast mode (manual `bot_engine.py train`) uses a fixed 2020-01-01 start unless `ML_USE_DAYS_PARAMETER` is set (`:260-274`). That flag is not in config.

**Feedback blending** (`:1337-1369`):
- `_load_trade_feedback` (`:1085-1131`):
  - drops records with no `entry_features`, i.e. seeds, orphan exits, morning-queue/ETF/options entries (§8)
  - marks `code_version < MIN_FEEDBACK_VERSION "1.0.33"` (`:947`, tuple compare `:1058-1083`) with `_legacy_weight=0.4`
  - **does not filter on outcome**
- `_build_feedback_training_data` sample weight = 3.0 × 2^(−age_days/30) × `_legacy_weight` (`:1163-1175`).
- Used only if ≥ 15 feedback records and ≥ 15 rows (`:1339`, `:1184`).
- Capped at 15% of generic rows, keeping the last ones (`:1348-1353`).
- Generic rows get weight 1.0.
- Feedback regime is always "neutral": the code looks for a top-level `regime_at_entry`, which track_fill never writes (`:1181-1182`).
- **Effective today:** boot cleanup deletes every non-seed record < 1.0.34 (`feedback_boot_cleanup.py:33`, `:51-60`; run at `server/bot.ts:7867-7895`). So the 0.4× legacy weight never applies in practice.
- **Config-fingerprint similarity weighting is DEAD:** `_get_current_config_fingerprint` and `_config_similarity_weight` (`:950-988`) have no callers. track_fill does write `config_fingerprint` and `config_snapshot` (`:2529-2554`). CLAUDE.md / docs implying fingerprint-weighted training are wrong.

**Shadow blending** (`:1372-1385`):
- `load_shadow_data(horizon_days=10, shadow_weight=0.3)`, requires ≥ 50 labeled +10d shadow records.
- Weight = 0.3 × 2^(−age/30).
- Regime forced to "neutral".
- Shadow features are **serve-time** vectors, carrying the §1.3 skew. So the training matrix mixes two feature distributions.

**Poisoned-record cleanup.** There are three distinct mechanisms:

1. **`ml_model_v2._clean_poisoned_feedback_on_startup()`** runs on every import (`:998-1054`) but is **one-shot**, gated by the marker `DATA_DIR/voltrade_feedback_cleaned.marker`.
   - Poisoned = (pnl_pct == 0 or None) and no code_version.
   - If ≥ 20% of records are poisoned: archive them to `voltrade_trade_feedback_poisoned_{ts}.json` and rewrite the file. Otherwise write a "skipped" marker.
2. **Boot cleanup every deploy** (`server/bot.ts:7862-7898` → `feedback_boot_cleanup.clean_feedback`):
   - keep `_seed` records; drop non-dicts, empty tickers, and `code_version < (1,0,34)` (non-strings fail)
   - non-atomic write, outside the fcntl lock
   - audit `SYSTEM "FEEDBACK CLEANUP: removed N ..."` or `"FEEDBACK: N trades, all clean"`
   - 5 s mutex wait, otherwise silently skipped
3. **`cleanup_feedback.py`** is a manual one-off CLI (`cleanup_feedback.py:22-36`): drops empty tickers and (pnl None/0 and no version and outcome None). No runtime caller.

**Seeding.** `seed_feedback_from_backtest.py` writes about 2,400 `_seed: True` records from `backtest_10yr_results.json` (csp_options / etf / stocks with hold ≥ 4 days), with no `entry_features` (`:127-152`). The daemon auto-runs it at startup if the feedback file is missing or has < 100 records, unless `VOLTRADE_AUTOSEED=false` (`voltrade_daemon.py:686-713`). Seeds never enter ML training (schema drop). They feed Kelly / performance views and can be rotated out by track_fill's 5,000-record cap.

---

## 6. RETRAINING — triggers, call paths, logging, replacement gate

### Call paths for `train_model`

| # | path | trigger | effective budget | live? |
|---|---|---|---|---|
| A | **Tier 3 hourly**, `tier3Strategic()` step 1 (`server/bot.ts:5709-5798`) | inline Python check: `ML_MODEL_PATH` missing **or** `time.time() − mtime > 86400` (`:5713-5721`) | `execPythonSerialized("python3 ml_retrain_safe.py", {timeout: 900000})` (`:5739`) → **90 s SIGKILL** (`:241`, `:271`); mutex wait up to 900 s | **YES**, the only automatic path |
| A′ | Tier 3 startup run at boot + 30 s (`:7768`) | same check; **no `state.active` check** | same | yes, every deploy |
| B | "4am daily retrain" (`server/bot.ts:6199-6220`) | `Math.floor(etHour) === 4` inside `runOvernightResearch`, whose only caller requires `etHour >= 20 \|\| etHour < 4` (`:5207-5213`) | 90 s | **UNREACHABLE** |
| C | `POST /api/ml/retrain` (`server/routes.ts:7161-7194`) | manual, session auth | raw `execAsync`, 300 s, **no python mutex** | manual only |
| D | `bot_engine.py __main__` inline train (`bot_engine.py:5636-5680`) | `_inline_train_allowed`: mode == "train" or env `VOLTRADE_INLINE_ML_TRAIN == "1"` (`:323-340`), **and** (model missing or > 24 h, or ≥ 20 trades whose `exit_date` is newer than the model; track_fill never writes `exit_date`, so this branch never fires) | subprocess caller's timeout | manual only. Calls `train_model()` with **fast_mode=False** plus `train_options_model()` |
| E | dev scripts (`sim_trading_day_v2.py:39-51`, `test_full_system.py:43-58`) | — | — | no |

Hourly Tier 3 additionally requires:
- the `setInterval(..., 3600000)` tick (`server/bot.ts:7736-7759`)
- `state.active`
- `!tier3Running`
- `VOLTRADE_DISABLE_TIER3 != "1"` (`server/bisectionFlags.ts:33-35`)

Net cadence: the model is retrained at the first hourly tick after it passes 24 h of age, so the steady state cycles from 0 to roughly 25 h. **Between 24 h and the next successful retrain, `ml_score` serves `fallback_rules`**, because `ML_MAX_AGE_HOURS` = 24 (`system_config.py:285`), and that output gets blended as ML (Finding 2).

**Other config retrain knobs are not wired.** `diagnostics.get_auto_fix_params()["force_retrain"]` (`diagnostics.py:542-546`) is consumed nowhere (bot.ts reads only `position_size_multiplier`, `min_score_threshold`, `should_pause`, `server/bot.ts:4365-4373`). `ML_RETRAIN_TRADES=20` is never read.

### What a run writes

1. **The model.** `train_model()` (`ml_model_v2.py:1251-1284`) wraps `_train_model_impl`.
   - Success writes `voltrade_ml_v2.pkl` (bundle) and `voltrade_ml_calibrator.pkl`. The calibrator is written at step 5b, **before** the global model is trained (`:1460`), so a run that dies or returns `insufficient_data` after 5b leaves a **new calibrator paired with the old model**.
   - **No accuracy or AUC floor.** Any run reaching `:1529-1530` overwrites the model unconditionally. The only things that prevent replacement are early returns: no bars, no data, < 50 rows, < 30/10 train/test rows, global model None.
   - Bundle keys (`:1512-1528`): `model, scaler, regime_models{bull|bear|neutral:{model,scaler}}, feature_names, accuracy (fraction), regime_accs, samples, feedback_trades (records loaded, not rows used), data_source, timestamp (ISO, local), feature_importance, label_method, cv_method, architecture, calibrator_path`.
2. **Return dict on success** (`:1552-1566`): `status:"trained", accuracy (%), regime_accs, features (=34), samples, feedback_trades, data_source ("generic(N)+feedback(M)" | "generic_only(N)" | "feedback_only(M)"), label_method, cv_method, regime_models [list], wf_fold_accs [%], calibrated (bool), training_seconds`.
3. **Failure shapes:**
   - Exception: `{status:"error: <msg>", error_location, accuracy:None, features:None, samples:None, error_detail, traceback_tail}` (`:1278-1284`)
   - `{status:"failed", reason:"Could not fetch training bars (HTTP …)"}` (`:1319-1323`)
   - `{status:"failed", reason:"No training data"}` (`:1369`)
   - `{status:"insufficient_data", samples}` (`:1413`, `:1471`)
   - `{status:"failed", reason:"Model training failed"}` (`:1477`)
   - `{status:"skipped", reason:"sklearn not installed"}` (`:1305`)
4. **`ml_retrain_safe.py`** adds `steps` (`["python_started","numpy_ok","lightgbm_ok","import_ok","training_start","training_done"]`), or on its own exception returns `{status:"error", steps_completed, error, traceback, error_class?}` (`ml_retrain_safe.py:46-93`). It prints JSON to stdout and **overwrites** `voltrade_last_retrain.json` (in /data/voltrade or /tmp) with `completed_at` / `completed_at_iso` (`:95-115`). **Nothing reads that file:** `ml_diagnostics.py:253-257` looks for `ml_retrain_status.json` / `ml_last_retrain.json`. Nothing is written if SIGKILLed.
5. **Audit strings (Tier 3, `server/bot.ts`):**
   - `TIER3` "Starting strategic scan..." (`:5710`)
   - `TIER3` "ML model stale or missing — triggering retrain..." (`:5725`)
   - `TIER3` "ML retrain complete — status: ${status}, accuracy: ${accuracy||'N/A'}, features: ${feature_count||'N/A'}, samples: ${samples||sample_count||'N/A'}" (`:5763`). `feature_count` never exists, so it always reads "features: N/A". Statuses "insufficient_data", "skipped", ml_retrain_safe's own "error" (no colon), and "fatal" **also land here, logged as "complete"**.
   - `TIER3-ML-ERROR` "ML retrain failed: ${status}${ — reason}${ @ location}${ | tb: …}" only when status starts with "error:" or "failed" (`:5749-5761`; pinned by `server/mlRetrainReasonVisibility.test.ts`)
   - `TIER3-ML-ERROR` "ML retrain failed (code=… signal=…)[likely OOM kill during import lightgbm/ml_model_v2 | timed out or killed externally | exec timeout exceeded]: <stderr/stdout tail>" (`:5765-5793`). The 90 s SIGKILL produces the "likely OOM" tag.
   - `TIER3` "ML model fresh (${age_hours}h old) — skipping retrain" (`:5796`)
   - If the age check itself fails (e.g., "Python mutex timeout" after its 5 s wait), the error goes **only** to `console.error("[tier3-ml]")` (`:5798`). **No audit**, and that hour's retrain is silently skipped.
   - The 4am `RETRAIN` / `RETRAIN-ERROR` strings (`:6212-6219`) never appear.
6. **Manual retrain** writes `ml_status.json` = the raw ml_retrain_safe output (`server/routes.ts:7171-7178`) and sets `ml_toggle.json {enabled:true}` if `steps` includes "training_done", even when training failed (`:7180-7182`). Tier 3 never writes `ml_status.json`, so `ml_status.py`'s `last_accuracy`/`last_status` reflect **manual retrains only**.
7. Python `logger.info` lines (walk-forward fold accuracies, shadow blend) are suppressed: `basicConfig(level=WARNING)` at `ml_model_v2.py:61`. Their stderr is discarded on success anyway.

### Where model age is exposed

| surface | location | thresholds |
|---|---|---|
| `GET /api/bot/ml-status` | `server/bot.ts:7988-8018` | `model_exists, model_age_hours, needs_retrain (> 24), retrain_threshold_hours: 24, retrain_schedule: 'Daily at 4am ET'` (false), `total_fills, learned_weights` (from `voltrade_weights.json`, which has no live writer; legacy `ml_model.py:961`) |
| `GET /api/ml/status` | `routes.ts:7127-7143` → `ml_status.py:36-52` | no auth; keys `model_exists, model_age_hours, enabled, last_accuracy, last_samples, last_features` (reads `feature_count`, always None), `last_status, last_error, last_traceback, last_steps, last_train_time` (reads `timestamp`, always None), `contributes_to_cagr, note` |
| `GET /api/monitoring/overview` → `overview.ml` | `server/bot.ts:3391-3398` | daemon `ml_status` route maps to nonexistent module `ml_status_impl`, so **always** `_ml_status_fallback` (`voltrade_daemon.py:490-528`): only `model_exists, model_age_hours, enabled, contributes_to_cagr, last_status` |
| `GET /api/diag/ml` (token) | `server/bot.ts:2456-2538` | `model_exists, model_age_hours, retrain_needed` (from `check_model_health`: **> 7 days**), `retrain_overdue` (**> 10 days**) (`diagnostics.py:237-238`), plus feedback stats |
| `GET /api/system/snapshot` → `ml` | `server/bot.ts:2107-2205` | `model_age_hours`, feedback counts |
| `/api/debug/ml-diagnostics` (DEBUG_ENABLED) | `routes.ts:6513` → `ml_diagnostics.py` | file ages. Feedback section reads the **wrong file** `trade_feedback.json` (`ml_diagnostics.py:128`, `:163`), so it always reports `exists:false` |
| diagnostics cache freshness | `diagnostics.py:91` | `ml_model` stale only at > 168 h × 1.5 = 252 h |
| UI ML panel | `client/src/pages/bot.tsx:744-813` | amber only if > 48 h |

Six different "stale" thresholds coexist: 24 h (actual fallback), 25 h (practical), 48 h (UI), 168 h, 240 h, 252 h. **Only 24 h matters for scoring.**

---

## 7. ML ENABLE / TOGGLE / BLEND

- **Toggle file.** `ml_toggle.json {enabled: bool}`, written by `ml_toggle.py` (`POST /api/ml/toggle`, `routes.ts:7145-7159`), the daemon fallback, and manual retrain. **No scoring code reads it.** ML is always invoked in `deep_score` (`bot_engine.py:1448-1775`) and `options_scanner`. `contributes_to_cagr = model_exists && enabled` (`ml_status.py:48`) is therefore meaningless.
- **Stock blend location:** `bot_engine.py:1741-1773`.
  - `rules_only_score` is captured before the blend (`:1444`).
  - If `model_type != "fallback"` (always true, Finding 2): `ml_only_score = round(ml_s, 1)`.
  - Weights `rule_weight`/`ml_weight` come from `BLEND_TRACKER_PATH` = `DATA_DIR/voltrade_blend_tracker.json` (`storage_config.py:29`); default 0.6 / 0.4.
  - **Conviction gate:** if \|ml_s − 50\| < 8, ml_s is replaced with 50.
  - `combined = combined × rule_weight + ml_s_blended × ml_weight`, then clamped to 0–100 (`:1777`).
  - DISAGREES: the comment "no drag" (`:1764-1767`) is wrong. Substituting 50 still pulls a rules score of 80 down to 68 at 60/40.
  - Reasons strings appended: "ML: uncertain (N), using rules-only", or "ML model: {signal} ({conf}% confidence)" (also shown for fallback_rules), and "Blend: X% rules + Y% ML".
- **Blend tracker writer:** `diagnostics.run_diagnostics()` (`diagnostics.py:477-505`), called from Tier 3 step 3 (hourly), every 5th Tier 2 cycle via `get_auto_fix_params` (`server/bot.ts:4355-4364`), and `/api/bot/diagnostics`.
  - Only updates when `check_model_health().performance.total_trades ≥ 20`: non-seed records with non-null pnl and a ticker, **including options exits**.
  - `recent_wr` = win rate of the last 20.
    - If > 60: `ml_w = min(0.55, 0.4 + (wr − 50) × 0.003)`
    - If < 40: `ml_w = max(0.2, 0.4 − (50 − wr) × 0.004)`
    - Otherwise 0.4
  - `rule_w = 1 − ml_w`. Writes `{rule_weight, ml_weight, updated}`.
  - It is driven by **overall bot win rate, not ML accuracy**.
- **Options blend:** fixed 60/40 or 70/30 (§2). Not driven by the tracker.

---

## 8. track_fill / trade feedback record (`voltrade_trade_feedback.json`)

`track_fill(order_data)` (`ml_model_v2.py:2355-2566`). Daemon method `track_fill` (`voltrade_daemon.py:273`) with a subprocess one-liner fallback at every bot.ts site.

- Under a thread lock: load the whole file, mutate, keep the **last 5,000** records (`:2558-2559`), atomic write with tempfile, rename, and fcntl lock on `FEEDBACK_PATH + ".lock"` (`:2275-2310`).
- Any exception is logged at debug level only (`:2563-2566`).
- qty = `qty` or `qty_filled` or `qty_requested`. If ≤ 0: warning "track_fill DROP" and return (`:2382-2392`).
- `slippage_pct = |fill − expected| / expected × 100` (`:2394-2396`).

**Exit detection.** `exit_context` or `exit_reason` or `is_close` truthy (`:2313-2330`). Matching is by **ticker only**: the most recent record with `outcome is None` (`:2333-2352`). There is no instrument or position id, so stock and options records on the same underlying can cross-match.

**ENTRY record keys** (`:2502-2554`):

```
ticker, side, qty, expected_price, fill_price, slippage_pct, volume, session,
time_placed, time_filled, score, entry_features, outcome:null, pnl_pct:null,
code_version, config_fingerprint, config_snapshot
```

- `config_snapshot` keys: max_position_pct, max_positions, max_total_exposure, spy_floor_neutral, spy_floor_bull, defensive_floor_bull, defensive_floor_neutral, absolute_max_position_pct, absolute_max_positions, min_options_score.
- **Not persisted** even when passed: `instrument`, `alpaca_fill_price`, `slippage_applied_pct`, `order_type`, `partial_fill`, `order_status`, `qty_requested`, `rules_score`, `ml_score_raw`, `win_probability`, `model_type`, `regime_at_entry`.

**On a matched EXIT, the entry record gains** (`:2454-2469`):
- `outcome` ("win" if pnl > 0, "flat" if 0, "loss")
- `pnl_pct` (3 dp; `exit_context.pnl_pct` wins over the price recompute)
- `exit_price, exit_time, exit_reason, days_held, exit_context`

It also calls `risk_kill_switch.record_trade_outcome(pnl_pct)` (`:2457-2461`).

**ORPHAN record** (no match, or matched entry has fill_price ≤ 0, in which case the entry stays open) (`:2472-2499`):

```
ticker, side, qty, fill_price, time_filled, outcome:"orphan_exit", pnl_pct:null,
exit_reason, note:"Exit fill with no matching open entry", code_version
```

The no-match case also logs a WARNING "ORPHAN EXIT".

**Writers, and whether they carry `entry_features`:**

| writer | location | entry_features? |
|---|---|---|
| regular-hours entry | `server/bot.ts:5659-5697` | **yes**, `pending.trade.entry_features`. `fill_price` = Alpaca avg ± a **Math.random() synthetic slippage haircut**, `expected` = scan price |
| morning-queue entry | `:4127-4160` | **no**. `fill_price` fully synthetic: `trade.price × (1 ± slip)` |
| ETF entry, `buildEntryFillPayload` | `:5398-5425`; `server/entryFill.ts` | **no**. expected == fill → slippage 0 |
| options pending entry, confirmed at first `/v2/positions` sighting | `options_manager.py:1040-1054`, payload `:1405-1411` | `{}` |
| stock full exits (WS stop / trailing / TP / time stop / position_kill) | `recordExitFill` via `resolvePendingExitFill`, only on Alpaca status "filled" (`server/bot.ts:199-213`, `:3795-3799`, `:6528-6531`, `:7132-7135`; payload `server/exitFill.ts:38-73`) | n/a (exit) |
| options exits, 8 reasons | `options_manager._record_options_exit_feedback` (`options_manager.py:106-147`) | n/a (exit) |
| seeds | §5 | none |

So **only regular-hours stock entries can become training rows**, and their open state counts as label 0 until closed (Finding 8).

**`fills_slippage_stats(feedback)`** (`ml_model_v2.py:2627-2650`):
- Returns `{count, avg_slippage_pct, total_slippage_cost}` over records with truthy `expected_price` and non-null `slippage_pct`, i.e. entry records.
- total cost = Σ expected × qty × slip/100.
- These numbers are **largely synthetic**: random haircut on regular-hours entries, fully synthetic morning-queue fills, zeros for ETF and options.
- Consumers: `/api/bot/performance` (`server/bot.ts:1717-1720`), `/api/diag/ml` `fills_count` (`:2472`), `/api/bot/ml-status` `total_fills` (`:8007`).

**`options_outcome_breakdown(feedback)`** (`ml_model_v2.py:2608-2624`):
- Outcome-bucket counts over non-seed records whose `exit_reason` ∈ `OPTIONS_EXIT_REASONS` = {dte_critical, assignment_close, dte_close, dte_close_bought, profit_target, loss_limit, bought_loss_limit, gamma_risk} (`:2602-2605`).
- Missing outcome counts as "open", though open entries have no exit_reason, so in practice "open" never appears.
- Exposed as `/api/diag/ml` `live_options_outcome_breakdown` (`server/bot.ts:2508`).

**`ml_attribution_summary()`** (`bot_engine.py:4080-4170`): DEAD (no callers). It needs `rules_score` and `ml_score_raw` on feedback records, which track_fill never writes, so it would always return `{"verdict":"INSUFFICIENT_DATA","n_trades":0,…}`. The only per-trade ML attribution that exists is the `TRADE` audit text "| Rules: X ML: Y Blend: Z" (`server/bot.ts:5600-5601`). `TradeResult.rulesScore`/`mlScore` in bot.ts are always null (`server/bot.ts:545-546`, `:877-878`).

---

## 9. HEALTHY VALUES & FAILURE SIGNATURES

### What normal looks like (derived from code; live samples from `research/experiments.md`)

- `model_age_hours`: cycles 0 → about 25. Live samples seen: 2, 11.1, 14.7, 21.7, 23.6 (`research/experiments.md:44273, 25191, 51860, 31229, 73469`). **> 24 h means scoring is on `fallback_rules`.** > ~26 h means the retrain is failing.
- Audit: one `TIER3 "ML retrain complete — status: trained, accuracy: NN.N, features: N/A, samples: ≤10000"` per ~25 h; `TIER3 "ML model fresh (Nh old) — skipping retrain"` on the other hourly ticks; zero `TIER3-ML-ERROR`.
- Train result:
  - `status:"trained"`, `calibrated:true`
  - `regime_models ⊆ ["bull","bear","neutral"]`
  - `samples` = 10,000 when generic data is plentiful (fast-mode cap)
  - `data_source` "generic(N)+feedback(M)" or "generic_only(N)"
  - **`training_seconds` must be < 90** (hard kill)
- Accuracy: no historical value is recorded anywhere in the repo, so there is no baseline. Mean walk-forward accuracy at threshold 0.5 with balanced class weighting. Interpret against the positive-class base rate, which is **not reported** (§10). Because of the same-date cross-ticker split (§4), a "high" accuracy is more likely leakage than skill.
- Live `ml_score` from a real model is always in [10, 90], with `ml_confidence` ≤ 0.8. Any value outside [10, 90] proves it came from `fallback_rules`.

### Failure modes

| failure | what happens | surfaced today? |
|---|---|---|
| Model file missing / > 24 h / corrupt / sklearn missing | `ml_score` → `fallback_rules`, blended at ML weight and recorded as `ml_score_raw` (Finding 2) | **No.** `model_type` is never persisted or exposed. Only model_age_hours > 24 can be inferred via `/api/bot/ml-status needs_retrain`. `/api/diag/ml retrain_needed` waits for 7 days |
| `ml_model_v2` import or feature-build exception in `deep_score` | silently rules-only; `entry_features` None (`bot_engine.py:1774-1775`) | **No** |
| Retrain killed at 90 s | `TIER3-ML-ERROR … signal=SIGKILL [likely OOM …]` | Yes, but misclassified |
| Retrain returns insufficient_data / skipped / ml_retrain_safe "error" / "fatal" | logged as `TIER3 "ML retrain complete — status: …"` | Misleading (reads as success) |
| Age-check Python hits the 5 s mutex timeout | that hour's retrain silently skipped (`console.error` only) | **No** |
| Bars fetch fails (feed 400, 429) | `TIER3-ML-ERROR "… failed — Could not fetch training bars (HTTP …)"` | Yes (KNOWN BROKEN #17/#23 fixes) |
| Tier 3 disabled or bot inactive | no hourly retrain. Model goes stale; fallback after 24 h | `TIER3-DISABLED` audit only for the env flag; `state.active=false` is silent |
| Partial retrain (calibrator saved, model not) | old model scored through the new calibrator | **No.** Detectable by comparing calibrator mtime > model mtime |
| Constant or degenerate predictions | the only guard is the training spread check (p90 − p10 ≥ 0.08, else GBM fallback) | **No.** No prediction log. About 11 constant serve features compress output variance, unmeasured |
| Feature NaN | `float(x or 0)` maps None → 0 but NaN passes. LGB treats NaN as missing; the sklearn GBM path raises → fallback | **No** |
| Live feature skew (§1.3) | wrong or constant inputs | **No** |
| Regime routing stuck (never bull) | neutral or bear models only | **No.** `regime_used` is not persisted |
| Feedback loop not producing labels (orphans) | `/api/diag/ml live_outcome_breakdown` | **Yes** (`server/bot.ts:2493-2499`; `scripts/session_health_check.py:262-282`) |
| Open trades labeled as losses | biases feedback rows | **No** |
| Options model absent | `options_ml_score` = rule constants | **No** |
| Toggle says OFF while ML blends | UI misinforms | **No** |

`/api/health` has **no ML check** (checks: server, database, alpaca, python, bot, scanner, feeds, licensing, process, memory; `server/bot.ts:1456-1625`). The session health check only flags `retrain_overdue` (> 10 days) (`scripts/session_health_check.py:269-270`).

---

## 10. GAPS FOR A COCKPIT (what's missing and what each would need)

1. **Live "ML mode" indicator** (real model / fallback_rules / no-ML). It needs `model_type` and `regime_used` counted per scan. Nothing records these today. Minimum change: the daemon scan returns a `{model_type: count}` histogram, or `deep_score` writes `ml_model_type` into the result and shadow record. Also fix the `"fallback"` vs `"fallback_rules"` check, otherwise the indicator would show the bug.
2. **Drift detection.** No training feature bands are persisted. It needs a `voltrade_ml_feature_baseline.json` sidecar at train time: per feature n, mean, std, p1/p5/p50/p95/p99, min/max, fraction-zero or fraction-constant, from `X_tr`, per regime. Serve side, rolling windows can already be computed from `voltrade_shadow_candidates.json` features (timestamped, ≤ 20k).
   - Interim, zero code change: `scaler.mean_`/`var_` are in the pkl.
   - Caveat: a z-score of the mean will **not** catch serve-time constants such as vxx_ratio = 1.0 (≈ the training mean). The cockpit needs **variance ratio and %-constant** per feature, not just mean shift.
3. **Feature-coverage gauge:** count of the 34 features that are constant over the last N serve vectors. Today about 11 (§1.3).
4. **Retrain ledger.** Only the latest result exists (`voltrade_last_retrain.json`, overwritten, unread) plus audit text. It needs an append-only record per run:

   ```
   ts, trigger (tier3 | startup | manual), status, reason, training_seconds (vs 90 s),
   steps, samples, data_source, wf_accuracy, wf_fold_accs, regime_accs,
   positive-class base rate, AUC / logloss / Brier, calibrated,
   top-k feature_importance, model + calibrator sha / mtime
   ```

5. **Metrics beyond accuracy.** The stock model reports only accuracy at 0.5. There is no AUC, logloss, Brier, base rate, or prediction spread. Add them to the bundle and the return dict.
6. **Promotion gate.** There is none. A cockpit "candidate vs live" view needs the new run's metrics compared against the previous bundle before replacement. That requires keeping the previous bundle and adding a compare step.
7. **Live calibration / hit rate of ML.** Feedback records don't store `ml_score_raw`, `win_probability`, `model_type`, `rules_score`, or `regime_at_entry`. track_fill must persist them (they are already in the trade dict at `bot_engine.py:3588-3596` but dropped by the payload and track_fill). This is also what `ml_attribution_summary` needs.
8. **Truthful age thresholds.** Show 24 h (`ML_MAX_AGE_HOURS`) as the red line, not 48 h, 7 d, or 10 d.
9. **Truthful toggle state.** Show "ML influence: ALWAYS ON (toggle not wired)" until the toggle is wired into `deep_score` and `options_scanner`.
10. **Retrain schedule truth.** "Hourly check; retrain when age > 24 h; 90 s budget", not "Daily at 4am ET".
11. **Blend weights panel.** Show `voltrade_blend_tracker.json` {rule_weight, ml_weight, updated}, note that it is driven by overall win rate (including options), and apply the conviction-gate window [42, 58].
12. **Options-model status.** File existence and age of `voltrade_ml_options.pkl`. Today the options path uses rule constants whenever the file is missing or older than 7 days.
13. **Data-source health for training:** `bars_feed()` value (iex vs sip) and rows surviving the 300k volume floor. In iex mode the training universe silently shrinks.
14. **Feedback label quality:** counts of open, matched, orphan, and with/without entry_features records, plus "open records currently being trained as label 0".

---

## Appendix A — exact artifact paths

DATA_DIR = `/data/voltrade` if `/data` exists, else `/tmp` (`system_config.py:24`, `storage_config.py:10-13`).

| artifact | written by | read by |
|---|---|---|
| `voltrade_ml_v2.pkl` (MODEL_PATH / ML_MODEL_PATH) | `_train_model_impl` | `ml_score`; age checks everywhere |
| `voltrade_ml_calibrator.pkl` | `_train_model_impl` step 5b | `ml_score`, `options_ml_score` |
| `voltrade_ml_meta.pkl` | never in prod | `meta_score` (never called) |
| `voltrade_ml_options.pkl` | manual train only | `options_ml_score` |
| `voltrade_trade_feedback.json` (+ `.lock`) | track_fill, boot cleanup, seeder | training, diag, performance, Kelly |
| `voltrade_feedback_cleaned.marker`, `voltrade_trade_feedback_poisoned_<ts>.json` | one-shot import cleanup | — |
| `voltrade_last_retrain.json` (/data/voltrade or /tmp) | `ml_retrain_safe` | **nobody** |
| `ml_status.json` | manual `/api/ml/retrain` only | `ml_status.py` |
| `ml_toggle.json` | toggle route / ml_toggle.py / manual retrain | `ml_status.py` only |
| `voltrade_blend_tracker.json` | `diagnostics.run_diagnostics` | `deep_score` |
| `voltrade_shadow_candidates.json` | `shadow_portfolio.log_candidate` | `load_shadow_data` (training) |
| `voltrade_weights.json` | legacy `ml_model.py` only | `/api/bot/ml-status` |
| `voltrade_ml_model.pkl` | legacy v1 path | deleted by the inline-train path if 0 or 52 features (`bot_engine.py:5645-5654`) |

`ml_model.py` (v1) has **no runtime importer** (only `test_auto_discovery.py:131` and `test_silent_except_ratchet.py:68`). Its `MODEL_PATH` now resolves to `voltrade_ml_v2.pkl` (`ml_model.py:38`, `:46`, `:697`), so running its `train_model` would overwrite the v2 model.

## Appendix B — other doc/code disagreements

- `regime_util.py:19-22` says markov_regime imports `classify_regime`. `markov_regime.py` does not import regime_util.
- `ml_status.py:49` note says "Auto-retrain runs at 4am ET daily and hourly via Tier 3". The 4am path is unreachable.
- `ml_retrain_safe.py:5-29` docstring says "8GB container" and that the bot.ts timeout was raised; the timeout is effectively 90 s.
- `bot_engine.py:5639-5640` says "Daily retrain at 4am runs via Tier 3 / ml_retrain_safe.py". The 4am path is unreachable; Tier 3 hourly is real.
- CLAUDE.md "purged walk-forward CV" (codebase map): not temporal (§4).
- CLAUDE.md "trade_feedback training with code_version gating (legacy weighted 0.4x)": implemented, but effectively moot after boot cleanup.
- `bot_engine.py:1720-1730` comment "Once retrained, restore real values here": the training side was never enriched, so it still applies.
- The daemon module docstring lists `ml_status`/`ml_toggle` as supported. Their modules don't exist; the inline fallbacks always run (`voltrade_daemon.py:343-351`).
