# H: Edge measurement, baselines, attribution, and the bot's known-broken state

Auditor H of 8. This is ground truth as of repo HEAD `8998328` (package.json `1.0.1084`, 2026-10-15).
Every claim cites file:line in `/home/user/voltradeai`. I read code only and changed no repo file.
I reproduced one Python expression in a scratch shell (noted where). I did not hit live endpoints:
the sandbox has no owner cookie or DIAG token. Live numbers below are quoted from append-only logs,
with dates.

---

## 0. Verdict for the cockpit's top question ("is the edge real right now?")

**No code anywhere computes it.** These pieces are missing:

- **Expectancy per trade after costs.** Never computed. No `expectancy` key exists in any live
  endpoint. The only expectancy-shaped math is the Kelly negative-EV gate in
  `position_sizing.py:153-208`, which feeds sizing and is not reported.
- **Win rate vs breakeven win rate.** No payoff-ratio or breakeven-win-rate code exists (grep for
  payoff, breakeven, win_loss_ratio found only stop-loss "breakeven" mechanics). The dashboard
  colours win rate against fixed 55/45 thresholds (`client/src/pages/bot.tsx:317`), not against
  breakeven.
- **Return vs buy-and-hold SPY over the same window.** No live code does this. Only the offline
  backtests do it (`backtest_v2.py:519-525`, `bot_backtest.py`). Sessions have computed it by hand
  in logs, e.g. `research/experiments.md:107072-107073`, `:107482-107483`. `HANDOFF.md` tells the human to
  "glance at equity curve vs SPY on the site", but that view does not exist.
- **Leak attribution (ideal vs real).** Not computed. Slippage data is partly synthetic
  (random haircuts, §1.1). The real broker fill price is passed to `track_fill` but dropped. There
  is no decision-price → order-price → fill-price chain, and no per-trade fee or financing record.

**What the live numbers in the logs imply (my arithmetic, not the bot's):**

- Last full read, `/api/diag/ml` on 2026-09-02 (`research/experiments.md:23641-23642`):
  `total_trades 29, win_rate 24.1, avg_win 6.14, avg_loss -4.78, recent_win_rate_20 15`.
- That gives payoff ratio 1.28 and breakeven win rate 4.78/(6.14+4.78) = 43.8%.
- Expectancy is 0.241×6.14 − 0.759×4.78 ≈ **−2.15% per closed trade**, before any extra costs.
- The sample is tiny: 29–31 closed trades over the bot's whole life (`:21245`, `:107076`).
- The 10-year backtest artifact loses to SPY in every config (§1.12).
- The shadow counterfactual base rate is negative: mean −1.02% at +5d (`:107172`).

---

## 1. Measurement code inventory (Part 1)

### 1.1 The substrate: `trade_feedback.json` (`storage_config.TRADE_FEEDBACK_PATH`)

Almost every live metric reads this one JSON list. Its record shapes matter more than any formula.

| Writer | Record shape / key facts | file:line |
|---|---|---|
| `ml_model_v2.track_fill` ENTRY | `ticker, side, qty, expected_price, fill_price, slippage_pct, volume, session, time_placed, time_filled, score, entry_features, outcome=None, pnl_pct=None, code_version, config_fingerprint, config_snapshot`. **No `strategy`, no `instrument`, no `alpaca_fill_price`, no $ P&L, no fees.** Any payload key not in this literal is dropped. | `ml_model_v2.py:2502-2555` |
| `track_fill` EXIT (matched) | Updates the matching open entry in place: `outcome` = win if pnl>0, flat if 0, else loss; `pnl_pct` (bot-reported `exit_context.pnl_pct` wins over the price recomputation); `exit_price, exit_time, exit_reason, days_held, exit_context`. Calls `risk_kill_switch.record_trade_outcome`. | `ml_model_v2.py:2433-2469` |
| `track_fill` EXIT (unmatched) | Appends `outcome:"orphan_exit", pnl_pct:None`. Matching is "most recent open record with the same ticker" (`_find_entry_record`, `:2333-2352`). | `ml_model_v2.py:2470-2499` |
| Rotation | Keeps the last 5000 records. | `ml_model_v2.py:2557-2559` |
| Seeder | `_seed: True` records from `backtest_10yr_results.json`, a **hand-picked "successful subset"** (CSPs, leveraged ETFs, stocks held ≥4d; drops 1-3d stocks and spreads). | `seed_feedback_from_backtest.py:1-31,151` |
| Regular-hours fill payload | `fill_price = alpaca_fill × (1 ± random 0.02–0.25%)`, a **synthetic haircut on the real fill**. `alpaca_fill_price` is sent but never stored. | `server/bot.ts:5660-5683` |
| Morning-queue fill payload | `fill_price = trade.price × (1 ± random 0.05–0.25%)`. **Never reads Alpaca's fill at all.** | `server/bot.ts:4129-4150` |
| ETF entry payload | `expected_price = fill_price = fillPrice`, so slippage is 0 by construction. | `server/entryFill.ts:42-56` |
| Exit payload (WS monitor / POS-KILL) | `fill_price = expected_price` = "best-known execution price at submit time (WS current price)". `exit_context.pnl_pct` = the bot's own position accounting, not the broker fill. | `server/exitFill.ts:22-28,45-72` |
| Options single-leg entries/exits | Through `track_fill`, with `pnl_pct` from Alpaca `unrealized_pl`. Multi-leg strategies and rolls are excluded. | `research/open_questions.md:878-900` (#12c) |

**Composition on record (2026-09-02):** 270 non-seed records =
`{orphan_exit: 225, win: 7, loss: 22, open: 16}` (`research/experiments.md:23654`).
So 83% of live records carry no P&L at all.

### 1.2 `GET /api/bot/performance` (owner-gated) — `server/bot.ts:1660-1921`

There are two code paths.

**Path A — Python inline (`server/bot.ts:1666-1750`).**

- **Input:** `trade_feedback`, minus `_seed` records (`:1685-1686`), minus records with an empty
  ticker or with `pnl_pct==0 and outcome is None` (`:1689`).
- `winRate = count(pnl_pct > 0) / N × 100`. Zero-P&L trades count as losses (`:1691-1693`).
- `avgGain` = mean of winning `pnl_pct`. `avgLoss` = mean of `pnl_pct <= 0`, a negative number
  (`:1694-1695`).
- `totalPnlPct` = **sum** of per-trade % returns. Not compounded, not dollar-weighted, not a
  portfolio return (`:1696`).
- `profitFactor` = sum of winning % / |sum of losing %|. It returns "inf" when there are no losses
  (`:1699-1701`, `:1740`).
- `byStrategy[strategy] = {wins, losses, total_pnl}`, keyed on `t.get('strategy','unknown')`. Live
  records have no `strategy` key (§1.1), so every live trade lands in `"unknown"` (`:1704-1710`).
- `realisticPnlPct = totalPnlPct − avg_slippage_pct × 2 × N` (`:1723-1726`). The trailing
  `/100*100` is a no-op. This assumes exit slippage equals entry slippage. It also **double-counts**:
  `pnl_pct` is already computed from fill prices, and `slippage_pct` is unsigned
  (`abs(...)`, `ml_model_v2.py:2396`), so price improvement is counted as cost.
- `slippageGapPct = totalPnlPct − realisticPnlPct`. `avgSlippagePct`, `totalSlippageCost` and
  `totalFills` come from `fills_slippage_stats` (`server/bot.ts:1717-1720`, `:1744-1746`).
- Also returns: `bestTrade`, `worstTrade`, `recentTrades` (the last 20 records, reversed),
  `seededTrades` (`:1729-1749`).

**DEFECT FOUND (not in any doc): Path A crashes whenever any open or orphan record exists.**

- Open entries and orphan exits store `pnl_pct: None` (`ml_model_v2.py:2477,2495,2516`).
- `t.get('pnl_pct', 0)` returns `None` because the key exists.
- `None == 0` is False, so the filter at `:1689` keeps those records.
- `None > 0` at `:1691` then raises `TypeError`. I reproduced this exact expression in a scratch
  shell: `TypeError: '>' not supported between instances of 'NoneType' and 'int'`.
- The same crash class was fixed in `position_sizing.py:300-312` on 2026-07-28 but never here.
- The live file holds 16 open and 225 orphan records (§1.1), so Path A should always throw. The
  catch at `server/bot.ts:1753-1755` then falls through to Path B.
- The test `test_fixes_pr11.py:18-48` re-implements the filter instead of running bot.ts's inline
  script, and its fixtures contain no `pnl_pct=None` records, so CI cannot catch this.

**Path B — Alpaca fallback, used when `totalTrades === 0` (`server/bot.ts:1758-1901`). This is what
the owner almost certainly sees.**

- **Window = TODAY only**: closed orders after 4 AM ET of the current trading day (`:1761-1775`).
- Pairs each sell with the earliest same-day buy (FIFO). If there is none, it uses the current
  position's `avg_entry_price` (`:1805-1833`).
- **Open positions are added as "trades" using unrealized P&L** (`:1836-1852`).
- `totalPnlPct` / `realisticPnlPct` = the account's **daily** change, (equity−last_equity)/last_equity
  (`:1854-1859`, `:1879`, `:1892`). So the key means something different from Path A.
- `byStrategy: {}`. All slippage keys are 0 (`:1882`, `:1893-1895`).
- Options P&L is not ×100-aware here; % is per-share price change.

**Always appended (`:1912-1917`):** `equityCurve` (§1.7), `equityPeak` (TS state), and
`currentDrawdown = (equityCurve[0].value − state.equityPeak)/state.equityPeak × 100`.

**Client:** `client/src/pages/bot.tsx:296-319` reads `winRate, profitFactor, avgGain, avgLoss,
totalPnlPct, currentDrawdown, equityCurve, bestTrade, worstTrade, recentTrades`. Colours are fixed
at win rate >55/<45 (`:317`) and PF >1.5/<1.0 (`:319`). The equity chart plots the array in the
order given (`:357`, `:434`; no reverse or sort in `:296-450`). The server array is **newest-first**,
so **the dashboard's time axis runs right-to-left**.

### 1.3 `GET /api/monitoring/overview` (owner) — `server/bot.ts:3347-3401`

- **Keys:** `timestamp`, `daemon` (RPC health), `kill_switches`, `memory.rss_mb`, `drawdown`, `ml`.
- `kill_switches` comes from `risk_kill_switch.get_kill_switch_status()` (`risk_kill_switch.py:708-734`):
  `peak_equity` (from `voltrade_peak_equity.json`), `consecutive_losses` (read from
  `consec_losses.json`, **which no code writes** — grep found no writer, so it is always 0),
  `daily_pnl_pct` (from `daily_pnl.json`, also never written, always 0), and the limits.
- `drawdown` = `drawdownStatus(evaluateDrawdown(acct.equity, state.equityPeak, -10))`
  (`server/drawdownGuard.ts:100-108`): `current_pct, kill_threshold_pct, proximity_pct, status`.
  CRITICAL at ≤ 75% of the threshold, WARNING at ≤ 50%. This was a dead metric that always read
  "OK" until 2026-08-09 (`:87-98`).
- `ml` comes from `ml_status.py` (model age and accuracy; no P&L).
- **No P&L, win rate, or SPY figures.**

### 1.4 `GET /api/monitoring/tier-pnl?days=30` (owner) — `server/bot.ts:3416-3518`

- **Buckets:** `csp_options, etf, stocks, vrp, other`.
- Records are filtered by `time_filled` (falling back to `timestamp`, then `time_placed`) ≥ the
  cutoff. Records with `pnl_pct is None` are skipped (`:3461-3467`).
- **Bucket key = `r.get('strategy','other')`** (`:3469-3471`). Live track_fill records never carry
  `strategy` (§1.1), so **every live trade lands in `other`**. Only seeded records populate the real
  buckets.
- **Per bucket:** `trade_count` (live only), `total_pnl_pct` (sum), `avg_pnl_pct` (mean),
  `win_rate`, and `seeded{count, total_pnl_pct, win_rate}` (`:3491-3505`).
- `avg_pnl_pct` is the only per-bucket expectancy-like number in the codebase. It is gross, % per
  trade, and lifetime-or-window.

### 1.5 `GET /api/monitoring/drawdown` (owner) — `server/bot.ts:3521-3546`

- `peak` = `risk_kill_switch.get_peak_equity()`, i.e. `voltrade_peak_equity.json`, written by
  `tiered_strategy.update_peak_equity` (`tiered_strategy.py:954-979`, called from `bot_engine.py:3904`).
  It falls back to current equity when the value is 0 (`server/bot.ts:3531`).
- `drawdown_pct = (equity−peak)/peak × 100` (the code comment wrongly says "basis points").
- `kill_threshold_pct` is **hardcoded −20**.
- **Status ladder:** KILLED ≤ −20, CRITICAL ≤ −15, WARNING ≤ −10, NOTICE ≤ −5, else OK
  (`:3537-3541`).
- **Disagrees with /overview**: different peak file, and −20 vs −10 threshold. See §1.16.

### 1.6 `adjustStrategyWeights`, `strategyWeights`, `trackClosedTrades` — `server/bot.ts:843-1019`

- `strategyWeights = {momentum .25, mean_reversion .20, vrp .25, squeeze .15, volume .15}` is
  in-memory with no file (`:843-849`). The only "weights file", `WEIGHTS_PATH` /
  `voltrade_weights.json`, is read by `/api/bot/ml-status` (`:8010-8012`) but written only by the
  retired `ml_model.py:961`.
- `trackClosedTrades()` runs from `tier1Reflex` (`:4244`): every 45s, market hours only, bot active
  (`:7601-7616`). It pulls the last 50 closed orders and pushes **one entry per fill** into in-memory
  `tradeResults` with `pnl: 0`, `strategy: "auto"` (`:853-886`). It then averages all buys vs all
  sells per symbol across the last 100 fills. That mixes different round trips; it is not FIFO
  (`:893-930`). It backfills **only `pnlPct`**.
- `adjustStrategyWeights()` (`:994-1019`):
  - Takes the first 20 fills with `timestamp >= DEPLOY_TIMESTAMP`. That timestamp was minted once
    and persisted to `voltrade_deploy_timestamp.json` (`:963-992`).
  - `winRate = count(t.pnl > 0)/n`.
  - **DEFECT FOUND:** `tradeResults[].pnl` is assigned only at `:873` (always 0; grep shows no other
    write). So **winRate is structurally 0%**, and whenever ≥5 fills exist the function always takes
    the `<0.4` branch: VRP +0.02 up to a 0.40 cap, momentum and volume −0.01 down to their floors
    (`:1003-1011`).
  - It writes the audit line `LEARN "Win rate 0% (N post-deploy trades)"`. That line was quoted live
    on 2026-10-06 as "win rate 0% (13 post-deploy trades)" (`research/experiments.md:107075`).
  - KNOWN BROKEN #38 blamed this on deploy-timestamp noise (`research/open_questions.md:5782-5800`).
    The real cause is that `pnl` is never set.
- **`strategyWeights` is never read anywhere** (grep: only `:843-1015`). It is not exposed or
  persisted and has no trading effect. **The LEARN audit line is a false signal and should not
  appear in a cockpit.**

### 1.7 `recordDailyEquity()` and the equity curve — `server/bot.ts:556-596`, `:1021-1041`

- **Point** = `{date: UTC YYYY-MM-DD, value: acct.portfolio_value, pnl: equity − last_equity}`.
  One row per UTC day, overwritten each call (`:1024-1033`).
- **Called from** `tier3Strategic` step 4 (`:5869-5870`), which runs hourly. The interval is gated
  on `state.active` (`:7746`), plus a boot run 30s after start (`:7768`).
- **When the curve stops:** a kill sets `state.active = false` (`:3939`), so the curve stops
  recording until a redeploy boots `active: true` (`:486`). Continuity during halts therefore
  depends on deploy frequency.
- **Persisted** to `/data/voltrade/voltrade_equity_curve.json` (`/tmp` fallback) after every update
  (`:561-596`). Loaded on boot with a shape filter and the first 365 kept (`:563-581`).
- **Ordering:** newest-first (`unshift`, `:1034`).
- **LATENT DEFECT:** `if (equityCurve.length > 365) equityCurve.shift()` (`:1035`) removes index 0,
  which is the **newest** row just added. Once 365 rows exist, the curve freezes and stops accepting
  new days. The comment at `:2593` claims "oldest-day pruned", which is wrong. Not live yet: the
  curve started around 2026-05-05.
- **Exposed** at `/api/bot/performance.equityCurve` (owner) and `/api/diag/equity_curve?days=N`
  (token, capped at 365, `:2589-2601`).

### 1.8 `fills_slippage_stats`, `options_outcome_breakdown`, `/api/diag/ml`

- **`fills_slippage_stats(feedback)`** (`ml_model_v2.py:2627-2650`)
  - Subset: records with `expected_price` and `slippage_pct`, i.e. entry fills only.
  - Returns `{count, avg_slippage_pct (mean of unsigned %), total_slippage_cost (Σ expected×qty×slip/100, $)}`.
  - Used by `/api/bot/performance` (`server/bot.ts:1717`), `/api/diag/ml` (`:2472`),
    `/api/bot/ml-status` (`:8007`). Wiring is pinned by `server/fillsSlippageWiring.test.ts:31`.
  - **The numbers mix real and synthetic slippage** (§1.1).
- **`options_outcome_breakdown(feedback)`** (`ml_model_v2.py:2602-2624`)
  - Outcome counts restricted to records whose `exit_reason` ∈ `OPTIONS_EXIT_REASONS` (8 strings:
    `dte_critical, assignment_close, dte_close, dte_close_bought, profit_target, loss_limit,
    bought_loss_limit, gamma_risk`).
  - Last live read: `{"orphan_exit": 10}`, zero wins or losses
    (`research/open_questions.md:1154`, 2026-09-17).
- **`/api/diag/ml`** (token; `server/bot.ts:2455-2548`) returns `feedback_count`, `fills_count`,
  `feedback_seeded_count`, `feedback_live_count`, `retrain_*`, `live_performance` (=
  `check_model_health().performance`), `live_outcome_breakdown`, `live_options_outcome_breakdown`,
  and shape signatures, code_version and session histograms. **This is the most complete live
  "edge" read that exists today.**

### 1.9 `diagnostics.py` (live-wired; it changes trading)

- **`check_model_health()`** (`:221-300`)
  - Excludes `_seed` and records with `pnl_pct is None`.
  - Computes `{total_trades, win_rate, avg_win, avg_loss, recent_win_rate_20}` and
    `degradation_detected` when recent < lifetime − 15 pts.
  - This has the right fields for a payoff ratio but never combines them.
- **`run_diagnostics()`** auto-fixes, all keyed on **win rate only, never payoff**:
  - Degradation → `force_retrain_and_reduce` at 0.5× size (`:361-367`).
  - `total_trades > 20 and win_rate < 40` → `increase_score_threshold`, min_score 75 (`:369-375`).
  - Writes ML/rule **blend weights** from `recent_win_rate_20` to `voltrade_blend_tracker.json`
    (`:476-503`). These are read by `bot_engine.py:1750-1752`.
- **`get_auto_fix_params`** runs every 5th Tier-2 cycle and sets `state.positionSizeMultiplier` and
  `state.minScoreThreshold` (`server/bot.ts:4355-4372`). Those gate trades at `:5269`, `:5282`,
  `:4043`.
  - Live 2026-09-05: "Low win rate: 24.1% over 29 trades", with the threshold raised to 75
    (`research/experiments.md:23643-23647`).
- **`check_weekly_loss(history)`** (`:38-80`) — **DEFECT FOUND (not in any doc): DEAD.**
  - The only caller passes a **one-element** history: today's equity (`server/bot.ts:4623-4629`).
  - The function returns `continue` when `len < 2` (`:44-45`).
  - So the −8% weekly pause and −4% half-size branches can never fire.
  - The TS caller's `WEEKLY-LIMIT` / `WEEKLY-WARNING` handling at `:4631-4638` is dead too.

### 1.10 `bot_engine.ml_attribution_summary()` — `bot_engine.py:4080-4160+`

- **Orphaned:** grep finds no caller in `.py` or `.ts`.
- It needs top-level `rules_score` and `ml_score_raw` on feedback records. `track_fill` never writes
  them (§1.1), so it would always return `INSUFFICIENT_DATA`.
- Its design: "ML lift" = mean pnl of ML-boosted minus ML-suppressed trades (±2 score), with a
  verdict `ML_HELPS` above +0.5 or `ML_HURTS` below −0.5.

### 1.11 `shadow_portfolio.py`: the counterfactual and rule-cost engine (live-wired)

**What it logs.** Every deep-scored candidate goes to `voltrade_shadow_candidates.json` (20k cap,
`:76-86`).

- **Source** (`bot_engine.py:1817-1843`): decision `taken` if `combined_score >= MIN_SCORE`, else
  `rejected_score`.
- **Corrections** by `update_last_decision` (`shadow_portfolio.py:288-339`; called at
  `bot_engine.py:3350`, `:3429`): `rejected_heat` (sector/correlation) and `rejected_other` (spread).
- **Plus:** `rejected_masterkill` from `tiered_strategy.log_masterkill_csp_shadow`
  (`tiered_strategy.py:315-376`) and `rejected_capital` from the CSP path (`server/bot.ts:4805-4826`).
- **Fields** (`:262-276`): `ticker, timestamp, score, decision, decision_reason, entry_price,
  vxx_ratio, regime_label, side, action_label, trade_type (since 2026-10-06), features, outcomes{+5d,+10d,+20d}`.

**Backfill.** Nightly at 22:00 UTC on its own 5-minute poll, ungated by trading state
(`server/bot.ts:4281-4300`, `:7388`). It calls `backfill_outcomes(3000)`.

**Labeler** `_label_from_path` (`:426-493`):

- Walks daily bars with a **+2% target / −4% stop / horizon timeout**.
- Same-day ambiguity counts as a loss.
- Writes `{return_pct, label, exit_price, exit_date, exit_reason, label_method:"path_dependent_v2"}`
  (`:681-688`).
- **Scale:** % return of a long-stock position. No size, no costs.
- **Breakeven win rate under these exits is ≈ 66.7%**, stated in `research/experiments.md:107134`.

**Two views:**

- **`get_shadow_stats()`** (`:947-1068`), exposed at `/api/diag/shadow` (`server/bot.ts:2853-2869`):
  `by_decision, by_regime, labeled_by_horizon, win_rate_by_decision` (n ≥ 5),
  `backfill_progress_by_decision, win_rate_by_change_pct_band, win_rate_by_masterkill_reason`.
  Also written to `voltrade_shadow_stats.json` after each backfill (`:696-702`).
- **`shadow_band_report()` — "the band report"** (`:1094-1173`), exposed at `/api/diag/shadow_bands`
  (`server/bot.ts:2871-2881`).
  - Cuts: `decision`, `decision_x_band` (signed same-day change bands `<-10 … >=30`), `band_all`,
    `score_quintile`, `decision_x_regime`, `decision_x_action`, `action_x_band`, `all`.
  - Per horizon: `{n, win_rate, mean_return_pct, median_return_pct}`, reported only when n ≥ 20.
  - **This is the only place `mean_return_pct` (an expectancy proxy) and a base rate (`all`) are
    computed.**

**Live read 2026-10-06** (`research/experiments.md:107166-107215`, base rate `:107172`, quintiles `:107175`, worst cell `:107177`; 19,876 labeled, 2026-07-25 to
09-08, about 90% NEUTRAL_BULL):

- Base rate +5d: win 49.2%, **mean −1.02%**.
- Score quintile is **monotone inverse**: Q1 −0.60% … Q5 −1.46%.
- `taken|10..20`: n=3,704, 35.1% win, −1.79%.

**Three known contaminations** (same entry):

1. `taken` ≠ bought. High scorers route to SELL OPTIONS or SELL.
2. The labeler's exits do not match live exits (ATR 1.5× stop clamped 2-6%, TP 4× stop, 10-day
   time stop). The "matches bot's take-profit" comment at `:83` is stale.
3. Repeated logging of the same ticker on every scan overstates independence.

Status: step (a) re-read of action-tagged records is gated to ≥ 2026-10-16 (`:107802-107809`; chain order `:107207-107211`).

### 1.12 Backtest engine: `backtest_v2.py` + `backtest.py` + `backtest_10yr_results.json`

**`backtest_v2.simulate`** (`:368-484`):

- **Scope:** single ticker, one strategy, `momentum` or `mean_reversion` (`strategies/`). Squeeze
  returns zero trades with a note (`:507-517`). **Options, CSP, floors, third leg, tiers and the
  DD halt are not modeled** (`research/experiments.md:107099`, `:107195`, `:107517`, `:107725`).
- **Timing:** signal on close[i], entry at open[i+1] plus cost (`:432-433`).
- **Exits** at close: ATR stop (1.5×ATR, clamped 2-6%), 2R target, or regime time stop
  (`:131-141`, `:396-406`).
- **Regime gating:** NEUTRAL, BEAR and PANIC have `max_pos 0`.
- **Sizing:** `equity × max_position_pct`.
- **Cost model:** per-side `liquidity_cost_pct` = max(volume tier on 20d average share volume, Reg
  NMS half-tick floor). Tiers: >20M 0.035%, >5M 0.075%, >1M 0.115%, else 0.185%. Fallback flat
  0.05% with no volume history (`:59-124`). Costs are deterministic by design, so runs are
  reproducible.

**`compute_metrics(equity, n)`** (`:281-298`): `total_return_pct`, `cagr_pct` (years =
n/252), `sharpe` (daily mean/sd × √252, population sd, rf = 0), `max_drawdown_pct` (absolute
value).

**Per-strategy output** (`:463-483`):

- `strategy, sharpe, totalReturn, maxDrawdown` (bot.ts contract).
- `config_name, final_capital, total_return_pct, cagr_pct, max_drawdown_pct, n_trades, win_rate_pct, data_quality, trades[]`.
- Trade fields: `date_entry, date_exit, ticker, instrument, strategy, entry_price, exit_price,
  holding_days, pnl, net_pct, exit_reason, score, vxx_ratio, vrp`.
- **No expectancy, payoff or profit-factor keys.**

**SPY benchmark:** `compute_metrics(spy_closes)` over the same fetched window, including the
400-day warmup in which momentum cannot trade. Adds `spy_cagr` and `alpha = cagr − spy_cagr` per
result, plus top-level `spy_total_return` and `spy_cagr` (`:519-536`).

**`backtest.py` (CLI)** (`:1-57`): `python3 backtest.py <ticker> <strategy> <years>`. JSON to
stdout, exit 0 always, trades capped at 200 per strategy.

**Wiring:**

- `POST /api/bot/backtest` (owner, on demand; `server/bot.ts:3681-3696`).
- Nightly 22:00-24:00 ET research pass runs `backtest.py SPY all 2` (`server/bot.ts:6143-6158`).
  It audits `RESEARCH "Backtest <s>: Sharpe x, Return y%, Drawdown z% — PASSING|MARGINAL|FAILING"`
  (thresholds: Sharpe ≥ 1.0 / ≥ 0.5).
- **Persisted only as audit-log lines (SQLite). Never compared to live.**

**`backtest_10yr_results.json`** (repo root, 1.98 MB). Shape:

- Top level: `{generated_at:"2026-04-07T14:57:47Z", period_start "2016-01-04", period_end "2026-04-07", trading_days 2579, starting_capital 100000, spy_total_return 281.17, spy_cagr 13.97, results{OLD,NEW,NEW_CSP_ONLY}, all_trades{… 1551 each}}`.
- Per-config keys: `final_capital, total_return_pct, cagr_pct, sharpe, max_drawdown_pct, n_trades, win_rate_pct, options_/stock_/etf_trades, *_win_rate, *_avg_return, options_pct_of_trades, config{…}, config_name, spy_*, alpha`.

| config | CAGR | Sharpe | MaxDD | WR | alpha vs SPY |
|---|---|---|---|---|---|
| OLD | 0.27% | 0.076 | 21.67% | 51.6% | −13.70 |
| NEW | −0.27% | 0.012 | 15.66% | 61.6% | −14.24 |
| NEW_CSP_ONLY | 5.68% | 0.682 | 3.38% | 67.7% | −8.29 |

**Can it serve as a frozen "known-good" baseline? No.**

1. It was produced by the original engine, which "is not in the repo" (`bot_backtest.py:5-8`), so it
   cannot be regenerated.
2. Every config loses to SPY by 8-14 CAGR points.
3. Its options trades are synthetic re-scalings of stock moves (a CSP trade carries COP's stock
   entry/exit 30.03→28.48).
4. It already contaminates live measurement:
   - A cherry-picked subset is seeded into `trade_feedback`.
   - Its per-bucket stats are the hardcoded Kelly priors (`position_sizing.py:239-287`).
   - `_get_historical_stats()` **does not exclude `_seed` records** (`:297-356`), so the Kelly
     negative-EV gate judges seeded and live trades pooled.

At most it is a "known-bad historical bar", not a known-good one.

### 1.13 Other offline harnesses (none live-wired)

- **`bot_backtest.py`**: monthly sector-ETF momentum rotation vs SPY, with its own `metrics()`
  (`:80-98`; same 4 metrics, CAGR in calendar years). CLI only; output goes to stdout.
- **`bot_backtest_subperiods.py`** (`:1-40`): calendar sub-period out-of-sample judgment of
  dual-momentum vs SPY. CLI only (tests in `test_bot_backtest_subperiods.py`).
- **`sim_trading_day_v2.py`**: a v1.0.13 (April 2026) phase-by-phase smoke test with PASS/WARN/FAIL
  counts. Not a performance measure; not wired.
- **`system_status.py`**: subsystem wiring self-test behind `/api/bot/system-status`, gated by
  `DEBUG_ENABLED` (`server/routes.ts:6576`). No performance content.

### 1.14 Existing primitives a cockpit could reuse

- **`server/killSwitchAutoResume.ts:215-260+`**: `parseFills` reads Alpaca `/v2/account/activities/FILL`,
  and `realizedPnlFromFills` computes average-cost realized $ P&L from the **broker's own fill
  ledger**, with an options ×100 multiplier and explicit `unpricedCloses`. Fetching is bounded and
  paged in `server/bot.ts:7442-7461`. Today it is used only for auto-resume reconciliation. **This
  is the closest existing thing to a truthful realized-P&L ledger.**
- **`server/reconstructPnl.ts:79`** `reconstructPortfolioPnl`, exposed at `/api/diag/reconstruct_pnl`
  (`server/bot.ts:2602-2665`): per-leg close-to-close attribution of current holdings for one day.
  Equities only; options excluded and stated. Sessions used it on 2026-10-13 to show the −3.0% week
  was the passive floor basket (`research/experiments.md:107482-107487`).
- **`position_sizing._kelly_fraction`** (`:153-208`): f* = (p·W − q·L)/W, and **0 when f* ≤ 0**.
  That is mathematically "p below breakeven p\* = L/(W+L)". It is the only payoff-aware gate in the
  system, applied pre-trade per bucket, divided by 3.
- **`scripts/statsUtils.ts`**: `clusterMeanTTest`, `binomialUpperTailP`. Usable for "is this win
  rate or expectancy significantly different from breakeven" given n.

### 1.15 The "baseline" concept

`grep -i baseline` over `*.py`, `server/*.ts` and `scripts/` turns up only:

- Memory, RSS and CI ratchet baselines.
- The equity-peak **re-baseline** (`paper_resume_sync.py:48-130`; `server/bot.ts:7552-7575`;
  `PAPER_REBASE` / `DATA_ANOMALY` in `server/killSwitchAutoResume.ts:31-37`).

**There is no frozen performance baseline.** The re-baseline actively erases the drawdown-from-
inception reference. After a `PAPER_REBASE`, `equityPeak` resets to current equity and the prior
peak survives only in an audit line (`server/bot.ts:7566-7568`).

### 1.16 The drawdown and peak zoo: four independent peaks, six thresholds

| Mechanism | Peak store | Threshold | Effect | Where shown |
|---|---|---|---|---|
| TS kill (`checkDrawdownKillSwitch`) | `state.equityPeak` → `voltrade_equity_peak.json` (`server/bot.ts:604-638`) | −10% (`:498`) | latch + cancel orders; liquidate only if −25% and `VOLTRADE_LIQUIDATE_ON_KILL` | `/api/health` `bot.drawdownPct`, `/api/diag/account`, `/overview`, `/performance.currentDrawdown` |
| bot_engine DD-HALT (`update_equity_peak`) | `voltrade_portfolio_dd.json` (`bot_engine.py:534-708`) | 18% (`system_config.py:308`); resume only in BULL/NEUTRAL and within 5% of peak; trip confirm at 9% after 60s | **scans return 0 stocks** (no entries) | audit `DD-HALT` only |
| tiered `master_kill_switch` | `voltrade_peak_equity.json` (`tiered_strategy.py:954`) | −20% DD, −3% day, invested ≥ `MAX_TOTAL_EXPOSURE` (`tiered_strategy.py:90-92`, `:286-300`) | tier actions blocked; logged `rejected_masterkill` | `/api/monitoring/drawdown` (hardcoded −20) |
| risk_kill_switch | `KILLSWITCH_PATH` state.peak_equity (`risk_kill_switch.py:93-198`) | −20% DD, −3% daily, consecutive losses | kill | `/overview.kill_switches` |
| TS daily loss | `equity/last_equity` | −3% (`server/bot.ts:4616`) | Tier-2 return | audit `TIER2-LIMIT` |
| TS weekly loss | n/a | −8% / −4% | **dead** (§1.9) | none |

- `/api/bot/status` advertises `dailyLossLimitPct: 5` (`server/bot.ts:2330`), but the live halt is
  −3. Stale display constant.
- The two live peaks already disagree: Python $111,736.69 vs JS $110,727
  (`research/experiments.md:107069-107070`).

### 1.17 Wiring summary

| Measure | Live-wired? | Cadence | Persisted | Exposed |
|---|---|---|---|---|
| `/api/bot/performance` | yes, but Path A crashes → Path B (today only) | on request | none | owner route, `bot.tsx` |
| tier-pnl | yes (live trades all in `other`) | on request | none | owner route |
| monitoring/overview | yes | on request | none | owner route |
| monitoring/drawdown | yes (different peak) | on request | none | owner route |
| `adjustStrategyWeights` | runs; effect nil; false LEARN line | 45s (T1) | none | audit `LEARN` |
| equity curve | yes | hourly (T3, needs `state.active`) | `voltrade_equity_curve.json` | `/performance`, `/api/diag/equity_curve` |
| `check_model_health` and auto-fix | yes; **changes min score and size** | every 5th T2; hourly T3 | `voltrade_diagnostics.json`, blend tracker | audit `DIAGNOSTIC`/`TIER3-DIAG`, `/api/diag/ml`, `/api/bot/diagnostics` |
| `check_weekly_loss` | called; **dead** | every T2 | none | none |
| `fills_slippage_stats` | yes (synthetic inputs) | on request | none | `/performance`, `/api/diag/ml`, `/ml-status` |
| `ml_attribution_summary` | **orphaned** | none | none | none |
| shadow log, backfill, bands | yes | per scan; nightly 22Z | `voltrade_shadow_candidates.json`, `voltrade_shadow_stats.json` | `/api/diag/shadow`, `/api/diag/shadow_bands` |
| backtest_v2 | on demand + nightly SPY 2y | nightly | audit lines only | `POST /api/bot/backtest`, audit `RESEARCH` |
| bot_backtest*, sim_day, 10yr json | no | none | static file | none |

---

## 2. Breakeven math and SPY benchmark (Part 2)

**Payoff ratio (avg win / avg loss): not computed anywhere as a reported number.** The ingredients
exist in three places, each with different filters:

- `/api/bot/performance` `avgGain` / `avgLoss` (`server/bot.ts:1694-1695`). Path A is broken (§1.2).
- `check_model_health` `avg_win` / `avg_loss` (`diagnostics.py:262-263`). Live-reachable via
  `/api/diag/ml.live_performance`.
- `position_sizing._get_historical_stats` (`:318-320`). Includes seeded records.

**Breakeven win rate: not computed.** It is implicit only in Kelly's negative-EV gate
(`position_sizing.py:197-201`). The shadow-label breakeven (≈67%) is stated in prose only
(`research/experiments.md:107134`).

**Expectancy: not computed** as a per-trade number on live trades. Closest equivalents:

- `tier-pnl.avg_pnl_pct` (per bucket, gross, broken bucketing).
- `shadow_band_report.mean_return_pct` (counterfactual, labeler exits, long-stock only).

**Win-rate semantics differ by surface.**

- Zero counts as a loss in `/performance`, `check_model_health` and tier-pnl.
- `track_fill` labels exactly 0 as `flat` (`ml_model_v2.py:2455`).
- `adjustStrategyWeights` is always 0 (§1.6).
- The cockpit should define one win rate and state its treatment of zero.

**SPY benchmark:**

- **Live: none.** No live code fetches SPY to compare against the equity curve.
- **Offline:** `backtest_v2` (same fetched window, including the warmup), `bot_backtest*`, and the
  static 10-year file (2016-01-04 → 2026-04-07).
- **By hand, in logs:**
  - 2026-09-08→10-06: account +3.1% vs SPY +2.3%, earned from holdings while not trading
    (`research/experiments.md:107072-107074`).
  - 2026-10-06→10-12: account −3.0% vs SPY −0.1% (`:107482-107483`).
- **The account is mostly a passive allocation.** The floor basket (QQQ/SMH/KWEB/VXUS) is about
  76% of equity (`:107484-107485`). An honest SPY comparison must separate the passive floor from
  the active sleeve. Otherwise "vs SPY" mostly measures the floor's tech tilt.

---

## 3. Current state of the bot (Part 3)

### 3.1 Is it trading?

**Yes, but thinly.** This is from logs; none of it is live-verified this session.

- On 2026-10-06 the DD-HALT was released and Tier-2 resumed acting: SELL_CSP on IEF and TLT, an
  FCEL scale-out and take-profit (`research/experiments.md:107159`).
- On 2026-10-12 it was trading heavily but pathologically: about $300k notional of QQQ+GLD
  round-trips ending flat, realized P&L ≈ $0, 51 cancels (`:107486-107487`).
- Seven fixes shipped 2026-10-13 (v1.0.1054–1060, `:107481-107781`).
- Every scheduled session since reports `/api/health` ok, alpaca ACTIVE, liveness not dark, scanner
  0 failures (`:27-29`, `:107811-107815`).

### 3.2 Equity, drawdown and kill-switch state (last known)

- **Equity:** about $102.1k, inferred from DD −7.7% on the $110,727 JS peak.
- **Drawdown:** `drawdownPct` −7.7% (2026-10-13 20:17Z, `:107805`), −7.8% (2026-10-14 11:01Z,
  `:107814`), −7.7% (`:29`).
- **JS peak:** $110,727.04 (`research/open_questions.md:6948`).
- **Python DD peak:** $111,736.69 (`research/experiments.md:107072`).
- **JS kill switch:** false since the 2026-09-29 auto-resume (`:106709-106711`, v1.0.1004). The resume basis is not recorded in the logs; the peak stayed $110.7k afterwards (`:106731`), which fits RECOVERED (peak kept) rather than a re-baseline.
- **bot_engine DD-HALT:** released 2026-10-06 (`:107159`, "DD-HALT RELEASED as bad-data trip").
- **History:** −8% on paper at peak 110.7k; equity at the 09-09 anomaly trough was a bogus $91,185;
  real low about $100.8k (`:107067`).

### 3.3 Live performance on record

| Date | Source | Numbers | Cite |
|---|---|---|---|
| 2026-09-02 | `/api/diag/ml` | 29 trades, WR 24.1%, avg win +6.14%, avg loss −4.78%, recent-20 WR 15%; breakdown orphan 225 / win 7 / loss 22 / open 16 | `research/experiments.md:23641-23642`, `:23654` |
| 2026-09-05 → 10-06 | `TIER3-DIAG` | "Low win rate: 25.8% over 31 trades" | `:21245`, `:107076` |
| 2026-09-04..14 | last 100 orders | 15 reconstructable round trips netted −$23.47 (single contracts, a few $ each) | `:107076-107078` |
| 2026-07-10..30 | `/api/diag/ml` | 70/70 live records `orphan_exit` | `:81757-81760` |

### 3.4 Known live-vs-backtest divergences

**No mechanism measures this** (HONESTY METRIC unimplemented). Documented divergences:

1. **Win rate.**
   - Live 24-26% vs the seeded and backtest priors: 55.8% ETF, 71.6% CSP, 49% stocks
     (`position_sizing.py:268-287`).
   - Live 24-26% vs the shadow `taken` 42-47% (`research/experiments.md:107125-107126`).
2. **Shadow labeler exits** (+2/−4/5d) do not match the live ATR / 4× / 10d exits (`:107188-107190`).
3. **Backtest scope.** `backtest_v2` cannot model the scan/quick-score/deep-score funnel, the
   floors, the tiers or options. A funnel ablation is therefore impossible (`:107195-107197`).
   Every DD-HALT, third-leg and tier change records "backtest unaffected by construction"
   (`:107099`, `:107517`, `:107725`).
4. **Options fills.** Options P&L uses Alpaca `unrealized_pl`, not quote-based fills. Options "fill
   realism" is open (#12 NEXT(2), `research/open_questions.md:892-897`).
5. **Seeded 10-year history** loses to SPY by 8-14 CAGR points (§1.12), yet it is the Kelly prior.

### 3.5 KNOWN BROKEN (`research/open_questions.md:3-7330`): bot-side status

| # | Status | Bot-relevant summary | Line |
|---|---|---|---|
| 3 | fixed 07-11 | CSP execution cascade (May-2026 CHANGELOG) | 18 |
| 12 | 3 of 4 fixed; **(c) partially open** | feedback loop recorded nothing for 2.5 months; multi-leg options and rolls still unrecorded; options outcomes still only orphans | 757, 878-900, 1150-1160 |
| 20 | **open** (judgment call) | `master_kill_switch` may over-kill CSP; shadow evidence `rejected_masterkill` at 85.8% / 91.2% WR; split by reason | 107, 290-363 |
| 29 | resolved, pending monitoring | false "Multiple API sources down" drove a 0.6× size haircut for 40h+ | 4317 |
| 30 | closed | options slots exceeded the cap 3 times via 3 mechanisms | 4713 |
| 32/33/34/35 | closed | stale-order sweeper dead code; unswept order sites; POS-KILL resubmits | 4922-5322 |
| 36 | resolved, monitoring | experiments.md ordering drift made the thrash ratio stale | 5322 |
| 38 | closed (misdiagnosed — see §1.6) | `DEPLOY_TIMESTAMP` reset per deploy | 5782 |
| 41 | mitigated; **(a) open** | OOM crash loop 09-08; health-gate 503 outage 09-10→09-16; Node market-hours memory growth not root-caused | 6097, 6483-6530 |
| 42 | **doc still open**; resolved in practice | bogus −11.8% daily / −18% DD reading 09-09 (Alpaca data anomaly) | 6620 |
| 43 | **doc still open**; resolved in practice | −10% kill was unreachable; then latched 09-10; auto-resumed 09-29 (v1.0.1004) | 6939-7232 |
| 44 | **open, FROZEN Dockerfile** | any `scripts/*.py`-backed diag probe 500s in prod (`scripts/` not copied into the image) | 7236 |

**Open bot-side defects not in KNOWN BROKEN (found this audit):**

- (a) `/api/bot/performance` Path A `None > 0` crash, falling back to today-only (§1.2).
- (b) `adjustStrategyWeights` win rate structurally 0%; `strategyWeights` write-only (§1.6).
- (c) `check_weekly_loss` dead (§1.9).
- (d) `/api/monitoring/tier-pnl` lumps all live trades into `other` (§1.4).
- (e) `realisticPnlPct` double-counts slippage, which is unsigned and partly synthetic (§1.1, §1.2).
- (f) Equity-curve 365-row freeze (latent, §1.7).
- (g) Dashboard equity chart time axis reversed (§1.2).
- (h) `get_kill_switch_status` `consecutive_losses` and `daily_pnl_pct` read unwritten files, so
  they are always 0 (§1.3).
- (i) `/api/bot/status` `dailyLossLimitPct: 5` vs the real −3 (§1.16).
- (j) `ml_attribution_summary` orphaned (§1.10).
- (k) Kelly stats pool seeded and live records (§1.12).

**Queued bot work:** the shadow-band chain. (a) Re-read action-tagged records on or after
2026-10-16. (b) Align the labeler's exits through a parallel column. (c) Scoring change: stop
admitting 10-20% same-day gainers as BUYs. (`research/experiments.md:107207-107211`, `:107221`).

### 3.6 Failure catalogue: incidents, how long undetected, how found

| # | Incident | Undetected / dark for | How found | Cite |
|---|---|---|---|---|
| F1 | track_fill qty guard dropped every regular-hours entry fill | **~2.5 months** (2026-04-23 → 07-06) | session reading `/api/diag/ml` (feedback count anomalies) | `research/open_questions.md:757-764`; `ml_model_v2.py:2375-2381` |
| F2 | Exit-fill machinery had no caller, so zero live outcome labels | since v1.0.34 → 07-06 | same trace (R12 D2) | `server/exitFill.ts:2-7` |
| F3 | ETF entry path never called track_fill; 70/70 exits orphaned | ~3 weeks (07-10 → 07-31) | session `/api/diag/ml` `live_outcome_breakdown` | `server/entryFill.ts:2-16`; `research/experiments.md:81757` |
| F4 | `code_version` hardcoded "1.0.34", so live attribution never worked | since v1.0.34 → 07-11 | session cross-checking `live_code_versions` | `research/open_questions.md:3334-3341` |
| F5 | Shadow backfill **never ran** (unreachable 22Z branch inside a market-hours loop) | **~3.5 months** (04-20 → 08-04; 12,048 unlabeled) | `/api/diag/shadow` zero outcomes | `server/bot.ts:4266-4296` |
| F6 | Fills file had no writer, so realistic P&L and slippage silently 0 | unknown → 07-05 | measurement-integrity read | `ml_model_v2.py:2569-2581` |
| F7 | WS position-exit monitor silently dead | 22h+ | `/api/diag/audit` (no STREAM lines) | `research/open_questions.md:1291-1298` |
| F8 | Kill-switch latch memory-only: deploy booted ACTIVE, ~50 min, $21.6k of buys | ~50 min (07-07) | live incident | `server/bot.ts:640-646` |
| F9 | `equityPeak` memory-only: every deploy re-based DD protection | since inception → 07-03 | audit KNOWN BROKEN #7 | `server/bot.ts:598-603` |
| F10 | CSP options channel: zero fills (sizing collapsed to the 1% floor) | **17+ days** | `/api/diag/audit?type=T2-FAIL` | `research/open_questions.md:4193-4200` |
| F11 | Tier-2 scan dead (`scan_market()` args) | hours (07-20) | audit TIER2-ERROR | `research/open_questions.md:3987-3994` |
| F12 | `/overview` drawdown always "OK" / 100% proximity (dead field) | since 04-20 → 08-09 | code read | `server/drawdownGuard.ts:87-98` |
| F13 | `/api/health` `drawdownPct` hardcoded 0 (`state.lastEquity` undefined) — "every session log going back weeks quoted 0.0 as no drawdown" | weeks → 09-09 | code read during incident | `server/bot.ts:1500-1511` |
| F14 | **−10% DD kill structurally unreachable** (inside `executeMorningQueue`); never fired at −17.9% | since inception → 09-09 | session saw −17.9% on /api/health with zero DRAWDOWN-KILL audits | `research/open_questions.md:6939-6975` |
| F15 | Bad Alpaca equity read (09-09) → JS kill latched; loop dark | **19 days** (09-10 → 09-29) | known; resolved by human directive (auto-resume) | `research/experiments.md:106547-106576` |
| F16 | **bot_engine DD-HALT kept "Scanned 0 stocks" while liveness read healthy** | **~27 days** (09-09 → 10-06); sessions on 09-29 logged it "by design" | **the human** asked why the bot seemed flat | `research/experiments.md:107054-107066`, `:106712-106722` |
| F17 | Health-gate 503: every new container refused on the latched liveness alarm; site down | **~5.5 days** (09-10 ~15:30Z → 09-16 02:40Z); misdiagnosed for days as restart-policy exhaustion | human-directed interactive session | `research/open_questions.md:6483-6530`; CLAUDE.md KNOWN STATE |
| F18 | OOM crash loop from the port-dwell fold | ~36h after merge (09-08) | `/api/health` uptime resets | `research/open_questions.md:6097-6110` |
| F19 | Bogus −11.8% daily-loss halt fired 36+ times | hours (09-09 pre-market) | routine `/api/diag/audit` | `research/open_questions.md:6620-6632` |
| F20 | `DEPLOY_TIMESTAMP` reset per deploy (LEARN noise) | unknown → 08-28 | `/api/diag/audit` | `research/open_questions.md:5782-5800` |
| F21 | SWEEP 204-parse made the CSP submit/cancel loop invisible; QQQ/GLD churn ~$300k/day | at least 1 day (10-12) | **the human** asked "whats going on" | `research/experiments.md:107481-107488`, `:107739-107752` |
| F22 | Shadow buckets masterkill/heat/other: zero labels | weeks (04-20 → 08-19+) | `/api/diag/shadow` | `research/open_questions.md:290-312` |
| F23 | Dead config gates MAX_CHANGE_PCT / SCORE_BAND_* "read nowhere" | since defined → 08-17 | grep audit | `research/open_questions.md:561-568` |
| F24 | **(new)** `/api/bot/performance` Path A crash → today-only fallback | since open/orphan `None` records existed (≥ 07-06) → still live | this audit (code read + repro) | §1.2 |
| F25 | **(new)** weekly-loss limit dead | since written → still live | this audit | §1.9 |
| F26 | **(new)** LEARN "win rate 0%" structurally false | since written → still live | this audit | §1.6 |

**Pattern for the cockpit.** The long-dark incidents (F1, F5, F14, F16, F17) share one trait. Each
is a **silent absence** — no fills recorded, no labels, no kill reachable, no entries — not an error.

- Liveness, which tracks only `state.active && !killSwitch` (`server/bot.ts:1496-1499`;
  `server/liveness.ts:1-60`), cannot see any of them.
- Neither can the scanner health check, which counts only exceptions
  (`server/scannerHealth.ts:21-23`).
- Two (F16, F21) were caught by the human looking at results, not by any alarm.

### 3.7 Documentation staleness relevant to a designer

- KNOWN BROKEN #42 and #43 are not marked closed, despite the 09-29 auto-resume and the 10-06
  release.
- `research/wishlist.md:480` still headers "🔴 ACTIVE LIVE CONCERN" for the 09-09 drawdown.
- `shadow_portfolio.py:83` claims its exits match the bot's take-profit.
- `server/bot.ts:2593` claims oldest-day pruning.
- `research/position_audit_2026-07-18.md` is a power-plant coordinate audit, **not** a trading
  position audit. Irrelevant to the bot.

---

## 4. Gaps: what an edge scorecard and leak waterfall need beyond what exists (Part 4)

**G1. A canonical round-trip ledger built from the broker, not from `trade_feedback`.**

- One row per closed trade: `entry_ts, exit_ts, symbol, instrument (stock/etf/option/multi-leg), sleeve (floor / third-leg / tier1-CSP / tier2 / scan-stock / convexity hedge), strategy, regime_at_entry, code_version, qty, multiplier, entry/exit fill $, realized $, realized % of position, % of equity, fees/financing, holding days, exit_reason`.
- Seed it from `/v2/account/activities/FILL`; `realizedPnlFromFills` already does average-cost
  matching (§1.14). Join to the bot's own intent records by order id.
- `trade_feedback` cannot serve: 83% orphan rate; no strategy, instrument or $ fields; synthetic
  fills; multi-leg and rolls missing.

**G2. Sleeve separation.** About 76% of equity is a passive floor basket. Without a sleeve tag, any
"edge vs SPY" is dominated by the floor. The scorecard needs active-sleeve P&L vs a matched passive
benchmark, and floor vs its own target weights.

**G3. Benchmark series.**

- Persist daily SPY close (and the floor basket's weighted benchmark) next to the equity curve, with
  the same UTC-date key.
- Compute window return, alpha, and tracking for 1w / 1m / since-reset / since-inception.
- Handle cash flows and the re-baselines explicitly: `PAPER_REBASE` must not reset "since
  inception".

**G4. Edge math, computed and exposed, with explicit n.**

- Per sleeve, strategy, regime and code_version: expectancy (mean net % and $ per trade), win rate,
  avg win, avg loss, payoff ratio, **breakeven WR = L/(W+L)**, WR − breakeven, profit factor.
- Add a significance test vs breakeven, e.g. `scripts/statsUtils.ts` `binomialUpperTailP` or a
  bootstrap CI on mean.
- Today n ≈ 31 lifetime. The scorecard must show "insufficient n" honestly.
- One documented win-rate convention (zero → flat, excluded).

**G5. The leak waterfall needs fields that are not recorded today.**

| Stage | Missing field |
|---|---|
| Ideal | signal/decision price at scan time (`entry_price` exists in shadow; `expected_price` exists but equals the fill on the ETF path) |
| Order | submitted limit price (in memory only, `server/bot.ts:4118-4125` area) |
| Fill | the real broker fill (`alpaca_fill_price` dropped by `track_fill`) |
| Costs | fees, options per-contract fees, borrow/financing |
| Exit | real exit fill (today it is the WS price at submit) |

- Remove the random synthetic haircut from stored data, or store it separately as "modeled cost"
  (`server/bot.ts:5660-5668`, `:4130-4136`).
- Add a signed slippage sign convention.

**G6. Counterfactual layers, using what shadow already logs.**

- **Ideal vs real:**
  - shadow `taken` (BUY-tagged, after 10-06) under the live-matched labeler
  - vs actually filled trades
  - vs `rejected_*` buckets: the rule-cost P&L CLAUDE.md demands
- This needs labeler step (b), live-exit alignment, and independence correction (dedupe per
  ticker per window).

**G7. Live-vs-backtest divergence: the HONESTY METRIC.**

- A backtest that models the live funnel does not exist. backtest_v2 is single-ticker equity only.
- The minimum viable version: a rolling comparison of live per-bucket expectancy vs
  that bucket's prior (Kelly defaults) and vs shadow `taken`. Flag divergence beyond a CI.

**G8. A "silent absence" alarm family**, the missing alarms behind F1/F5/F14/F16/F21:

- trading days since last entry;
- scans with N > 0 candidates but 0 orders;
- fills recorded vs broker fills (orphan rate);
- labels per night;
- churn (notional traded ÷ |net position change|);
- cancels vs SWEEP audits.

**G9. One source of truth for drawdown.**

- Show all four peaks and their thresholds side by side, or consolidate them (§1.16).
- The cockpit should label which mechanism a given drawdown number belongs to.

**G10. Fix or quarantine the broken surfaces before reuse.**

- `/api/bot/performance` Path A crash and its today-only fallback.
- tier-pnl bucketing.
- `realisticPnlPct` double count.
- LEARN line.
- weekly-loss check.
- equity-curve 365 freeze.
- reversed chart.

A cockpit built on today's `/api/bot/performance` would show **today's** stats under lifetime
labels.

---

## 5. Exact vocabulary (shared names)

- **Endpoints:** `/api/bot/performance`, `/api/monitoring/overview`, `/api/monitoring/tier-pnl`,
  `/api/monitoring/drawdown`, `/api/diag/ml`, `/api/diag/shadow`, `/api/diag/shadow_bands`,
  `/api/diag/equity_curve`, `/api/diag/account`, `/api/diag/reconstruct_pnl`, `/api/bot/backtest`,
  `/api/bot/diagnostics`, `/api/bot/ml-status`, `/api/bot/status`, `/api/health` (`checks.bot.{status,
  equityPeak, drawdownPct, liveness, autoResume}`).
- **Files** (under `/data/voltrade`):
  - `voltrade_trade_feedback.json`, `voltrade_equity_curve.json`, `voltrade_equity_peak.json`
  - `voltrade_kill_switch.json`, `voltrade_portfolio_dd.json`, `voltrade_peak_equity.json`
  - `voltrade_shadow_candidates.json`, `voltrade_shadow_stats.json`
  - `voltrade_deploy_timestamp.json`, `voltrade_blend_tracker.json`, `voltrade_diagnostics.json`
- **Functions:** `track_fill`, `fills_slippage_stats`, `options_outcome_breakdown`,
  `check_model_health`, `run_diagnostics`, `get_auto_fix_params`, `check_weekly_loss`,
  `ml_attribution_summary`, `log_candidate`, `update_last_decision`, `backfill_outcomes`,
  `_label_from_path`, `get_shadow_stats`, `shadow_band_report`, `simulate`, `compute_metrics`,
  `run_backtest`, `liquidity_cost_pct`, `_kelly_fraction`, `_get_historical_stats`,
  `evaluateDrawdown`, `drawdownStatus`, `evaluateDailyPnl`, `checkDrawdownKillSwitch`,
  `update_equity_peak`, `master_kill_switch`, `check_kill_switches`, `realizedPnlFromFills`,
  `reconstructPortfolioPnl`, `recordDailyEquity`, `trackClosedTrades`, `adjustStrategyWeights`.
- **Shadow decision buckets:** `taken`, `rejected_score`, `rejected_heat`, `rejected_other`,
  `rejected_masterkill`, `rejected_capital` (plus `rejected_halt` and `rejected_earnings` in the
  docstring, `shadow_portfolio.py:240-242`).
- **Feedback outcomes:** `win`, `loss`, `flat`, `orphan_exit`, `None` (open).
- **Audit types:** `LEARN`, `DIAGNOSTIC`, `TIER3-DIAG`, `DRAWDOWN-KILL`, `DD-HALT`, `TIER2-LIMIT`,
  `EQUITY-READ-INVALID`, `RESEARCH` (nightly backtest), `SHADOW-BACKFILL`, `SWEEP`, `CSP-SKIP`,
  `TIER-DEFER`.
