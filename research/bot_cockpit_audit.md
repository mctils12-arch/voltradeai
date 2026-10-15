# VolTradeAI Bot — Stage 1 System Audit (ground truth, 2026-10-15)

Deliverable ordered by `research/directives/bot_cockpit_prompt_2026-10-15.md` §2
("SYSTEM AUDIT"). Repo HEAD `8998328` (package.json `1.0.1084`). This is the
map of the bot **as it actually exists today**: real file, real function, real
fields. It reports what the code *does*, not what comments, docstrings,
CLAUDE.md or the current dashboard say it does. Where they disagree, the code
wins and the disagreement is listed (§9).

**How it was produced.** Eight parallel readers each owned one stage and read
whole functions, not grep hits (their full reports are the appendices in
`research/bot_cockpit_audit/A…H_*.md`, ~500 KB, every claim `file:line`).
Live production was read the same night through the public health endpoints
and the token-gated read-only `/api/diag/*` probes
(`research/bot_cockpit_audit/live_snapshot_2026-10-15.md`). An
adversarial verification pass (two independent refuters per claim unit, a
code-semantics lens and a live/reachability lens, tiebreak on disagreement)
was launched over 37 claim units; it hit the account's session limit after
one verifier completed (V01), so the author verified the headline claims
directly in-session by reading the cited code, and the rest stand as
**reader-reported** with `file:line`. The **Defect register** (§8) states
each unit's status honestly: `CONFIRMED` (in-session read or completed
verifier), `PARTIAL` (verifier, with the corrected statement), or
`reader-reported` (one stage reader, not independently re-read). Fifteen
existence/caller claims were verified mechanically by grep and are marked
`[grep]`. The evidence per unit and the resume command for the blocked
pass are in `research/bot_cockpit_audit/verification_2026-10-15.md`.

**How to read the stage map (§3).** Each stage has four parts, as the
directive asks: *what it does* · *the real fields it emits* (exact JSON keys,
audit `type` strings, file names) · *healthy-state values* (anchored to the
live snapshot) · *how you'd know it broke* (failure signature, and whether it
is currently surfaced anywhere or silent). The vocabulary in `monospace` is
the vocabulary Design and Build must reuse verbatim.

---

## 1. The one-page verdict

1. **The bot is alive but it is not, in any meaningful sense, trading.** The
   paper account is $99,615 (peak $110,727, −10.04%). 83.8% of equity is a
   passive ETF floor (QQQ/SMH/KWEB/VXUS + GLD) that Python rebalances; the
   active sleeve is six one-contract short puts. In the audit window
   (Oct-14 13:58Z → Oct-15 03:15Z, a full market afternoon) there were **zero
   `TRADE`, zero `T1` CSP dispatches, zero stock exits, zero `SWEEP`** — and
   158 `OPTIONS-SLOT-FULL` lines, which is the only thing the scan tried to
   do. The stock scanner has not opened a position in months.
2. **The reason stock entries are ~zero is structural, not market-driven.**
   Four independent gates each suffice: (a) `get_adaptive_params` gives
   `MAX_POSITIONS 0` in NEUTRAL and the five floor ETFs consume the stock
   slots in every other regime (§S5 B1); (b) the Kelly sizer blocks the
   `stocks` bucket outright with the shipped priors (§S6, V18); (c) the
   sector veto counts held option symbols as stocks and blocks any
   "Unknown"-sector name once ≥2 options are held (§S5 B6, V09); (d) Node's
   `minScoreThreshold` has been raised to 75 by the win-rate diagnostic. The
   counterfactual log calls 8,372 of 20,000 candidates `taken` — a label that
   only means "score ≥ 63", not "an order was placed" (V10).
3. **Edge is negative on the record and nothing computes it.** The 36 closed
   live trades: win rate 27.8%, avg win +9.08%, avg loss −5.16% ⇒ breakeven
   win rate 36.2%, expectancy ≈ **−1.20% per trade** (our arithmetic; n = 36).
   The dashboard's `/api/bot/performance` cannot show this: its Python path
   throws `TypeError` on every open/orphan record and silently falls back to a
   today-only Alpaca view (V25). No code computes expectancy, payoff ratio,
   breakeven win rate, or a live return vs SPY (§S11).
4. **Leak attribution is not computable from persisted state.** The stored
   `fill_price` is a random synthetic haircut on the broker fill (or, for the
   morning queue, no fill at all); the real `alpaca_fill_price` is dropped;
   the decision price is a 15-minute-delayed daily-bar close, not a quote;
   bid/ask are fetched and discarded; exit P&L is the trigger-time bar price;
   scale-out legs are never recorded; no `client_order_id` joins a decision
   to an order (§5, V22–V24).
5. **The five-layer decision chain is mostly lost after each scan.** Trigger:
   lost. Features: 34 ML features survive (≈11 of them are constants at
   serve time — V14/§S3). Reasoning: the `reasons` list and most Python
   vetoes live only in the in-memory `lastScanResult`, overwritten every
   scan; the audit log keeps ~13 h. Conviction: final score only. Action:
   audit text + Alpaca. No `scan_id`, no `candidate_id` (§4).
6. **Silent failure is the house style.** Nearly every feed catches its own
   exception and returns a neutral-looking value; health judges feeds by
   cache-file age, and failed fetches write the cache; two sub-feeds are
   structurally dead (EDGAR insider filter on a key that doesn't exist,
   USPTO endpoint retired); a Reddit failure fabricates a *bullish* reading
   (§S1, V33/V34). `/api/health` checks neither the daemon nor the ML model.
   The 26 incidents in §7 share one trait: a silent absence, not an error.
7. **The current page is the opposite of the cockpit.** A 920-px centred
   column of 13 stacked cards, all polling, no SSE client, calls a clock
   route that does not exist (so it always says AFTER HOURS), draws the
   equity curve backwards in time, omits the ×100 option multiplier, prints
   a −5% daily-loss limit while −3% is enforced, and renders every error as
   `$0.00` or "No positions" (§S12, V29).
8. **A live risk for the next session:** equity sits 0.035 pp below the Node
   −10% kill line. The first Tier-1 cycle after 09:30 ET on Oct 15 (or the
   first owner load of the dashboard, which trips the same check from a GET)
   will latch the kill switch, cancel all open orders and email, unless
   equity is ≥ $99,654.34 by then. Auto-resume then takes over (V37).
   **Happened as predicted (added 2026-10-15 17:57Z):** `DRAWDOWN-KILL` at
   13:30Z on the first Tier-1 cycle, equity $99,008 (−10.6%); loop dark
   4.4 market hours, `/api/health` `degraded`, `liveness.dark true`;
   auto-resume evaluated `UNRECONCILED` (−9.5% not above −8%), waiting on
   `PAPER_REBASE` after 13 market hours.

---

## 2. Live snapshot (2026-10-15 03:12Z) — the numbers HEALTHY VALUES are anchored to

Full digest: `research/bot_cockpit_audit/live_snapshot_2026-10-15.md`.

| probe | value |
|---|---|
| `/api/health` | `status ok`, `checks.bot {status active, equityPeak 110727.04, drawdownPct "-10.0", liveness.dark false, autoResume.outcome NOT_EVALUATED}`, server `rss_mb 1741`, `scanner.consecutiveFailures 0` |
| `/api/diag/account` | `equity 99615.03, last_equity 102226.48, cash 17601.88, buying_power 174663.94, multiplier 4, drawdownPct -10.0355, killSwitch false` |
| `/api/diag/positions-detail` | 11 positions: QQQ 61, SMH 20, KWEB 388, VXUS 133, GLD 21 (= $83,517, **83.8% of equity**) + 6 short puts qty −1 (CRWV, PFE, TEM, VZ, WFC, XLF, all 261120 P). No non-floor stock. |
| `/api/diag/ml` | `model_age_hours 19.4`; feedback 1674 (1326 `_seed`, 348 live); `live_performance {total_trades 36, win_rate 27.8, avg_win 9.08, avg_loss -5.16, recent_win_rate_20 20}`; `live_outcome_breakdown {orphan_exit 293, win 10, loss 26, open 19}`; `live_pnl_null_count 312`; live sessions `{None 293, regular 55}` |
| `/api/ml/status` (public) | `last_status trained, last_accuracy 58, last_samples 10000, last_steps […training_done]` (written only by the manual `POST /api/ml/retrain`) |
| `/api/diag/timings` | scan total 99.1 s: `deep_score 35.67` (35 s cap binding), `tier_engine_tier1 57.04` (CSP Layer-2 prefetch `budget_exceeded true` 106/150), `step9b_convexity_overlay 0.34` (it runs) |
| `/api/diag/daemon` | `alive, uptime 1846 s, rss_mb 166, modules [storage_config, position_sizing, options_execution, ml_model_v2, bot_engine]` |
| `/api/diag/scanner` | `degraded false, consecutiveFailures 0, dataSourceErrors {}` |
| `/api/diag/shadow` | 20,000 records (2026-08-03 → 10-14); `by_decision {rejected_score 10438, taken 8372, rejected_other 681, rejected_heat 493, rejected_masterkill 16}`; `by_regime {NEUTRAL_BULL 19984, NEUTRAL 16}`; +5d win: taken 51.5%, rejected_heat 24.1% |
| `/api/diag/shadow_bands` | base rate `all` +5d mean −1.0%; score quintiles are **inversely** monotone (Q1 56.2% win / Q5 42.5% at +5d) |
| `/api/diag/orders` | last 100: 58 options / 42 equity; equities `market`, options `limit`; 34 limit orders canceled; no QQQ option orders |
| audit log (2000 rows ≈ 13 h) | 25 boots (24 redeploys); `TIER2` "Scanned 12138 stocks, 2 trade candidates" ×18; `OPTIONS-SLOT-FULL` ×158; `QUEUE` ×26 (VXUS/KWEB covered calls); `LEARN "Win rate 0%"` ×17; `EVENTLOOP-LAG` ×106; **zero** TRADE / T1 / SWEEP / WS-EXIT / WS-SCALE-OUT / POS-KILL / DD-HALT / DRAWDOWN-KILL / RESEARCH / BACKUP / RETRAIN |

---

## 3. STAGE MAP

### S0 — Process, scheduler, plumbing (orchestration)

**What it does.** Railway runs `run_with_daemon.sh` → Node `dist/index.cjs`
(`server/bot.ts`, the orchestrator) plus a supervised Python RPC daemon
(`voltrade_daemon.py`, Unix socket `/tmp/voltrade_daemon.sock`, restarts 2 s
after any exit, self-kills at peak RSS > 1024 MB). Node runs three tiers:
**Tier 1** every 45 s during `/v2/clock.is_open` (drawdown kill check → stale
order sweep → `manage_positions` subprocess → position sync → morning queue →
closed-trade tracking); **Tier 2** on an adaptive timer (60 s at the open,
3/7/2 min through the day, 15 min 16–20 ET, 5 min 04:00–09:30, 30 min
overnight; runs when `clock.is_open || 4 ≤ ET < 20`, weekday- and
holiday-blind) → daemon `run_full_scan` → admission → `executeTrades` → tier
dispatcher; **Tier 3** hourly + 30 s after boot (ML age check/retrain,
manipulation scan, diagnostics, equity-curve row, COT, Form 4, port-dwell).
A market-data WebSocket (`wss://stream.data.alpaca.markets/v2/iex`, 1-min
bars, 43 tickers + held names) drives the Node exit engine. Every `audit()`
call writes SQLite `audit_log` (last 2000 rows), pushes to SSE
`/api/bot/stream`, and prints to stdout. (Appendix A.)

**Real fields.**
- In-memory `state` (`bot.ts:485-499`): `active` (boots `true`), `killSwitch`
  (from disk), `dailyPnL`, `dailyLossLimit -3`, `positionSizeMultiplier`,
  `minScoreThreshold 65`, `diagCycleCount`, `consecutiveStopLosses`,
  `circuitBreakerUntil`, `alpacaFailCount`, `morningQueueExecuted`,
  `equityPeak` (from disk), `maxDrawdownPct -10`; ad-hoc `dailyBlockedTickers`,
  `recentStopTickers`, `lastStopTimes`, `streamSignalQueue`.
- Audit `type` strings (253 literal `audit("…")` calls; full list Appendix G §4):
  lifecycle `STARTUP, SHUTDOWN, SYSTEM, RULES, SCHEDULE, EXECUTION, START,
  STOP, EVENTLOOP-LAG, DB-SLOW-WRITE`; scheduler `TIER2, TIER2-ERROR,
  TIER2-FAIL, TIER2-BACKOFF, TIER2-LIMIT, TIER2-DISABLED, TIER3, TIER3-ERROR,
  TIER3-DIAG, TIER3-ML-ERROR, TIER-DEFER, TIER-DISPATCH-ERR, T1/T2/T3/T4 (+
  -FAIL/-ERR/-SKIP)`; stream `STREAM, STREAM-ERROR, STREAM-DISCONNECT,
  STREAM-SIGNAL, STREAM-PRIORITY`; risk `DRAWDOWN-KILL, DD-HALT, KILL-SWITCH,
  KILL-WARN, TIER-KILL, CIRCUIT-BREAKER, CIRCUIT-BREAKER-RESET,
  EQUITY-READ-INVALID, KILL-AUTO-RESUME-EVAL, KILL SWITCH AUTO-RESUME,
  LIQUIDATE-ALL`.
- SSE event payload (`bot.ts:511`): `{time, type, action, detail, message}`
  (`type === action`, `message === detail`); no `id`, no `retry`, no
  heartbeat; the only producer is `audit()`.
- Daemon `health` RPC: `{alive, uptime_seconds, rss_mb (peak maxrss),
  max_rss_mb, modules_loaded[], active_dispatches, active_dispatch_detail
  [{method, elapsed_sec}], layer2_prefetch{…}, layer1_stats{…},
  surface_score_stats{…}, pid}`.
- Liveness file `voltrade_liveness.json {lastActiveAt, lastReminderAt?,
  savedAt}`; `/api/health.checks.bot.liveness {dark} | {dark, marketHours,
  wallHours, detail}`; thresholds `LOOP_DARK_MARKET_HOURS 2`,
  `LOOP_DARK_WALLCLOCK_HOURS 24` (`server/liveness.ts`).
- Kill latch `voltrade_kill_switch.json {killSwitch, reason, savedAt,
  lastAutoResumeAt?}`; `reason ∈ {"drawdown-kill (tier1)", "drawdown-kill
  (account route)", "owner toggle ON|OFF", "auto-resume: RECOVERED |
  DATA_ANOMALY | PAPER_REBASE"}`.

**Healthy values.** During RTH: a `TIER2` "Starting scan (interval: Nmin…)" →
"Starting intelligence scan..." → "Scanned 12138 stocks, M trade candidates
(via daemon)" triple every 1/3/7/2 min; `TIER3` "Starting…"/"Strategic scan
complete" hourly; one `STREAM "Real-time feed live — 43 tickers"` after boot
and no `STREAM-DISCONNECT` churn; daemon `uptime_seconds` monotonic, `rss_mb
< 1024`, `active_dispatches ≤ 1`; `/api/health` `status ok`, `liveness.dark
false`, `scanner.consecutiveFailures 0`, `memory.pressure ok`;
`EVENTLOOP-LAG` lines are normal background (106 in 13 h live).

**Failure signature — and whether it is surfaced.**
- Tier 1 has **no heartbeat and no audit line on a healthy cycle**; its
  errors go to `console.error("[tier1]")` only. An Alpaca 401/outage during
  Tier 1 is silent in the audit log. There is **no re-entrancy guard**
  (`tier2Running`/`tier3Running` exist, `tier1Running` does not [grep]).
- Daemon down: `TIER2-ERROR "Scan failed via daemon: … Socket: connect
  ENOENT …"` (surfaced). Daemon **restarts are never audited**; visible only
  as an `uptime_seconds` reset if someone polls. `/api/health.checks.python`
  is `python3 -c "print('ok')"` — it does not check the daemon.
- WebSocket: `STREAM-ERROR`, `STREAM-DISCONNECT "… reconnecting in 10s"`;
  flat 10 s retry, no ping/pong, no bar-staleness watchdog; `streamConnected`
  and last-bar time are **exposed by no endpoint**. A half-open socket is
  undetectable.
- Liveness: `dark` only measures `state.active && !state.killSwitch`;
  broker-unreadable, Python DD-HALT, `TIER-KILL`, and "active but every scan
  returns 0 candidates" are all invisible to it (`checks.scanner` catches
  scan exceptions only).
- `/api/system/health-check` is permanently `critical` and
  `/api/system/snapshot.scan_stats.success` permanently 0: both look for a
  TIER2 message containing `"complete"`, which is never written (the success
  line is "Scanned N stocks…") [grep] (V03/A3).
- Redeploys: every SIGTERM runs `DELETE /v2/orders` (24× "All open orders
  cancelled" in 13 h) and wipes `openOrders`, `morningQueue`,
  `monitoredPositions`, notifications, circuit-breaker state, `lastScanResult`,
  `ownerStopped`. The bot boots `active: true`, so an owner Pause is undone by
  the next merge (V03).
- Overnight research / 4 am daily reset: the scheduler never enters the
  20:00–04:00 window and the 4 am branch requires an hour the caller forbids,
  so daily resets, the "4am retrain" and the nightly local backup happen only
  on process restart; `BACKUP` and `RESEARCH` rows are 0 in the live window
  (V02).
- The subprocess runner classifies any Python command ≤150 chars without a
  literal heavy import as "trivial" (no mutex, **10 s SIGKILL**), which is the
  path `python3 ml_retrain_safe.py`, the Tier-1 `manage_positions` one-liner,
  the shadow backfill, backup and backtest take (V01 — see register for how
  much of this is live-bound).

### S1 — Data feeds / inputs (everything that reaches a trading decision)

**What it does.** The scan consumes Alpaca (assets, movers, snapshots, bars,
quotes, news, options chains, account/positions/clock/assets), yfinance
(VIX/TNX/DXY, fundamentals, option chains, earnings calendar, insider
transactions), Finnhub (insider MSPR, recommendations, earnings calendar),
EDGAR full-text search, Wikimedia pageviews, FRED, GDELT, USPTO (dead),
Polygon (reference, needs `POLYGON_KEY`), StockTwits, Reddit JSON/RSS,
pytrends (disabled), three news RSS feeds. The daemon keeps module caches
resident; Tier-1 subprocesses start cold every 45 s. (Appendix B — the full
30-row catalog with cadence, cache path/TTL, failure value and consumer.)

**Real fields (the ones a cockpit would key on).**
- Scan-level: `scanned` (≈12,138), `filtered`, `deep_analyzed` (≤5),
  `data_source_errors{macro, intel, alt, social, finnhub, stock_details}`
  (→ `/api/diag/scanner.dataSourceErrors`), `scan_phase_times`,
  `voltrade_scan_timings.json {status, phases[{phase, duration_sec,
  total_sec}], last_phase_completed, last_phase_at, scan_started}`.
- Macro (`macro_data.get_macro_snapshot`, cache `/tmp/voltrade_macro_cache.json`
  300 s, **ephemeral, written even when every field is a default**): `vix,
  vix_change, vix_regime, treasury_10y, dollar_index, sector_momentum{11
  SPDR sectors}, strongest_sector, weakest_sector, market_regime
  (risk_on|risk_off|neutral), spy_below_200_days, spy_vs_ma200,
  spy_above_200d, vxx_ratio, vxx_latest, vxx_avg30, vxx_closes, spy_vs_ma50,
  spy_ma50, spy_close, spy_closes_60d, regime, data_quality
  (good|degraded; "poor" unreachable), data_quality_failures ⊆ ["vxx_ratio"]`.
- Alpaca feed tier (`alpaca_feed.data_feed/bars_feed/options_feed`): `sip →
  delayed_sip` on 403, `bars_feed` maps `delayed_sip → iex`, options `opra →
  indicative`; `feed_status()` returns `{feed, degraded, downgraded_since}`
  but has **zero callers** — the production tier is recorded only in code
  comments (SIP 403, OPRA 403) (V35).
- Enrichment outputs consumed by `deep_score`: `intel{intel_score, news,
  insider{net_direction, buy_count, sell_count, insider_signal}, earnings_pattern,
  trap_warning, trap_reason}`, `alt{alt_score, wiki, short_interest,
  congressional, patent, ftd, fred_macro, geopolitical, fred_adjustment}`,
  `social{combined_signal, confidence, reddit, google_trends(0), news_multi,
  sources_active}`, `finnhub{insider_mspr, insider_signal, analyst_consensus,
  analyst_buy_count}`, analyze.py `detail{vrp, recommendation, sentiment{score,
  signal, bull_pct, reddit_buzz, trend_spike, contrarian_flag}, edge_factors,
  vol_metrics{rsi_14, volume_ratio_5d, ewma_rv, garch_rv, rv20}, valuation.sector,
  data_source}`.

**Healthy values.** `scanned ≈ 12,138`; `snapshot_fetch ≈ 4–5 s`;
`dataSourceErrors {}` *(but see below — this is also what "dead" looks like)*;
macro `vix 12–35`, `vxx_ratio 0.8–1.4` (BULL ≤ 0.95, CAUTION ≥ 1.05, BEAR ≥
1.15, PANIC ≥ 1.40 per `regime_util.py:43-50`), `spy_vs_ma50 0.94–1.06`,
`data_quality good`; FRED `T10Y2Y −1…+1.5`, `BAMLC0A0CM 0.8–2.0`, `ICSA
200–260k`; `event_memory.json` mtime advancing (the one real liveness signal
for Alpaca news).

**Failure signature.** Almost always **silent by construction** (V33):
- Every enrichment module catches its own exceptions and returns a neutral
  dict, so `data_source_errors` records only import errors, the 15 s
  `.result` timeout, and analyze.py failures. Live it reads `{}` while two
  sub-feeds are structurally dead: EDGAR insider (`intelligence.py:260`
  filters on `_source.form_type`; EFTS returns `form`/`root_forms`) → buys
  and sells always 0; USPTO (`alt_data.py:442`, endpoint 301 → HTML) → patent
  signal cached 0 for 24 h.
- `alt_data`, `social_data`, `finnhub_data` (caches `{"error":…}` 6–12 h),
  the insider cache and the macro cache **write the cache file on failure**;
  `diagnostics.py` judges a source "up" by file existence/mtime
  (`diagnostics.py:86-94, 389-395, 446-453`) → a dead upstream reads healthy
  and never triggers the 0.6× size haircut.
- Defaults with a direction (V34): VIX fetch fail → `vix 20.0` → sizing
  ×0.80; sector-ETF fail → all sectors 0.0 → `market_regime risk_off` → −5
  on every deep score; VXX/SPY fail → regime NEUTRAL → `MAX_POSITIONS 0`
  (fail-closed); Reddit non-200 → `reddit_buzz 0.5` → component 100 →
  sentiment 60 "Bullish" (+10 squeeze); stress index with every upstream
  dead → 38.75 "elevated" → ×0.75; Python account read failure → equity/cash
  fabricated as 100,000 (`bot_engine.py:3176-3182`).
- Contract mismatches (consumer reads a key the producer never emits) make
  these **constants, not data**: `intel["vxx_ratio"]` → 1.0 (also pins the
  ML regime router — the bull model can never be selected),
  `intel["earnings_surprise"]` → 0, `vol_metrics["ewma_vol"]` → 2,
  `quick_result["high_52w"]` → `price_vs_52w_high 0`, `range_pct 0`,
  `edge["short_float_pct"/"days_to_cover"]` → 0/0, Markov fed `_spy_rets=[]`
  → `markov_state 1`, `macro["market_breadth_ma50"/"spy_change_pct"]` never
  set (V14).
- The universe filter drops every symbol ending in `W` (meant for warrants;
  also LOW, NOW, SNOW, DOW, CDW, EW, TROW) (V35).
- Orphaned from the decision path (proof in Appendix B §4):
  `institutional_data`, `etf_data_sources`, `alphadesk/`, `probability_engine`
  (snapshot only), Google Trends, `intelligence.record_earnings_outcome`,
  the ML news/insider fetchers, `macro_data.get_sector_for_ticker`, daemon
  RPC `macro_snapshot`. `manipulation_detect`, `cftc_cot`, `sec_form4_bulk`
  only write audit/archives. **No datacore signal reaches the bot** (`/api/v1`
  is consumed by nothing on the trading path).

### S2 — Candidate generation: universe → quick scan → deep score

**What it does.** `bot_engine.scan_market()` → `_scan_market_inner()`
(`bot_engine.py:2571-4080`), run in the daemon as `run_full_scan`. Pre-gates:
`VOLTRADE_ULTRA_SURVIVAL_MODE` env; Python DD halt (18%). Universe:
`/v2/assets` (tradable, no ".", not `*W`, ≤5 chars; 24 h cache
`universe_cache.json`) + top-50 movers. Snapshots in 400-symbol batches →
quick score per symbol (max 60) → sort → top `VOLTRADE_DEEP_SCORE_CAP` (default
5, ≤10, hard 15; the config key `DEEP_SCORE_TOP_N` is not read) → memory
guard (skip ≥550 MB RSS, trim to 3 ≥400 MB) → serial `deep_score` with a
**35 s wall cap** (live: 35.67 s — binding; candidates not reached are
dropped) → tie-break re-ranking (broken, V07) → stock loop (§S5) →
covered-call sweep → options scanner → manage_positions (advisory) → options
manager → third leg → convexity overlay → SPY floor → defensive floor →
tier engine. (Appendix D §1.)

**Real fields.**
- `quick_result`: `ticker, price (= round(dailyBar.c,2)), close, open,
  prev_close, volume, dollar_volume, change_pct, quick_score, above_vwap
  (= c>o, not VWAP), range_pct (always 0), vwap_dist (always 0), reasons []`.
  `quick_score = min(|chg|,15)×3 + min(v/1e6,5)×2 + (5 if |chg|>10 else 2 if
  |chg|>5 else 0) − (30 if |chg|>50 else 15 if |chg|>30 else 0)` — max 60,
  below `MIN_SCORE` 63, so a candidate that is never deep-scored can never trade.
- `deep_score` output adds: `deep_score, reasons[], vrp, side, trade_type,
  action_label (BUY | SELL | SELL OPTIONS), recommendation, rec_reasoning,
  sentiment_score, rsi, momentum_score, mean_reversion_score, vrp_score,
  squeeze_score, volume_score, ewma_rv, garch_rv, horizon, leveraged_bull,
  leveraged_bear, entry_features (34), rules_only_score, ml_only_score,
  sector, above_ma200, ma200_pct_above, ml_features, regime_label`.
- Scan return (`bot_engine.py:4006-4076`, served whole by `/api/bot/last-scan`):
  `timestamp, scanned, filtered, deep_analyzed, portfolio_value, cash,
  current_positions, slots_available, options_slots_available, new_trades[],
  options_trades_added, position_actions[], upgrade_candidates[], top_10[{ticker,
  price, score, change_pct, side, action_label, reasons[:3], rank,
  tiebreak_score, factor_strength, sector_clash, sector, factors{momentum,
  mean_reversion, vrp, squeeze, volume}}], third_leg, convexity_overlay,
  intraday_shorts, options_management, spy_floor, defensive_floor,
  tier_actions[{tier, action, ticker, strategy, size_pct, reason, metadata}],
  tier_stats, kill_status, kill_suppressed_trades, tier_kill_status,
  scan_phase_times, size_tier{label, max_positions, max_sector_positions,
  max_single_pct, max_volume_share_pct, min_position_dollar, next_tier_at},
  data_source_errors`; halted scans add `{halted, halt_reason, peak_equity,
  current_equity, dd_pct, dd_event?}`.
- Trigger of a scan: `TIER2 "Starting scan (interval: Nmin based on market
  time)"`; `STREAM-SIGNAL` (vol ≥2.5×, |bar| ≥0.8%, |mom5| ≥0.3%) is audited
  and **cleared without being passed to the scan** — non-causal [grep].

**Healthy values.** `scanned ≈ 12,138`, `filtered` in the low thousands,
`deep_analyzed 5` (cap), `deep_score` phase 20–35 s, composite scores cluster
25–45 (reaching 63 needs several bonuses), "2–4 trade candidates" per scan
(live these are almost all covered calls + options-scanner picks).

**Failure signature.** "Scanned 0 stocks" (DD-HALT has its own audit; ULTRA
mode is stderr only); `TIER2 "Scan returned error: Could not fetch market
data from Alpaca — <detail>"` (surfaced, note the type is `TIER2` not
`TIER2-ERROR`); `deep_score` ≈ 0 s with `deep_analyzed 0` and all scores ≤60 =
memory-guard skip / SURVIVAL_MODE (stderr only → zero stock trades possible);
shadow `newest` timestamp stale = deep_score not completing; `tie-breaking
failed: cannot access local variable 'current_positions'` on **every scan**
(container logs only — V07): `top_10[].tiebreak_score` and `factor_strength`
are always `null`, `sector_clash` always `false`, `rank` is list position, and
the UI shows them as real. Tickers without a listed options market get no
deep score and **no shadow record** (analyze.py returns an error; an implicit
"must have options" gate with no trace — V14).

### S3 — Feature pipeline (the 34 ML features, live vs training)

**What it does.** `FEATURE_COLS` (`ml_model_v2.py:140-184`, asserted 34) is
built two ways: at training by `_compute_features(bars, idx, …)` from daily
bars, and at serve time by `deep_score` (`bot_engine.py:1687-1740`) from
analyze.py, macro, intel and the quick-scan snapshot. The keys match; the
values do not. (Appendix C §1 has the full per-feature formula table with
clips.)

**Real fields (serve source → training formula → status).**

| # | feature | serve source | status vs training |
|---|---|---|---|
| 1–2 | `momentum_1m`, `momentum_3m` | 400-d bars (c[−1]/c[−21]−1, c[−63]) | ≈ match |
| 3 | `rsi_14` | analyze `vol_metrics.rsi_14` (default 50) | method differs |
| 4 | `price_vs_52w_high` | `quick_result.high_52w` (never present) | **constant 0** |
| 5 | `volume_ratio` | analyze `volume_ratio_5d` (5d/90d) | **different definition** (train: today/20d) |
| 6 | `ewma_vol` | `vol_metrics.ewma_vol` (analyze emits `ewma_rv`) | **constant 2** |
| 7 | `atr_pct` | Wilder ATR14/price (default 3.0) | ≈ match |
| 8 | `vrp` | analyze atm_iv − rv20 | **different meaning** (train: VXX proxy clamp ±20) |
| 9 | `iv_rank_proxy` | stock 30-d HV rank | **different** (train: VXX percentile) |
| 10 | `vwap_position` | `above_vwap` = close > open | match (proxy both sides) |
| 11 | `adx` | Wilder DX full history (default 20) | different formula |
| 12 | `range_pct` | hardcoded 0 | **constant 0** |
| 13 | `vxx_ratio` | `intel.get("vxx_ratio", 1.0)` — key absent | **constant 1.0** |
| 14 | `spy_vs_ma50` | macro | ≈ match |
| 15 | `markov_state` | Markov on `_spy_rets=[]` | **constant 1** |
| 16 | `regime_score` | Markov with vxx=1.0 | range pinned [43.5, 68.5] |
| 17 | `sector_momentum` | sector-ETF 1-day % via `SECTOR_MAP` (~50 tickers, else 0) | different definition |
| 18 | `cross_sec_rank` | most-actives `change` field (likely absent) | likely **constant 0.5** |
| 19 | `earnings_surprise` | `intel.earnings_surprise` — key absent | **constant 0** |
| 20 | `put_call_proxy` | from real vrp | thresholds tuned for proxy vrp |
| 21 | `vol_of_vol` | macro `vxx_closes` | ≈ match |
| 22 | `frac_diff_price` | last 20 closes (≈ 0.21·ln price) | ≈ match |
| 23 | `idiosyncratic_ret` | change_pct − `macro.spy_change_pct` (never set) | = `change_pct_today` |
| 24 | `change_pct_today` | snapshot (15-min delayed) | timing |
| 25 | `above_ma10` | deep closes | ≈ match |
| 26 | `trend_strength` | \|change\| / ewma_vol(=2) | degenerate |
| 27–29 | `intel_score`, `news_sentiment`, `insider_signal` | forced 0.0 | constant by design (train: 0.3×surprise) |
| 30 | `iv_rank_stock` | same as #9 | identical to #9 |
| 31 | `days_to_earnings` | forced 0.5 | constant both sides (dead) |
| 32 | `credit_spread` | TLT/HYG 21-d (60 s cache) | ≈ match |
| 33 | `market_breadth` | `0.5 + (spy_vs_ma50−1)×6` proxy | deterministic function of #14 |
| 34 | `put_call_ratio` | `vxx_latest/20` (default 1.25) | ≈ match |

**Healthy values.** No training-feature baseline exists (no quantiles, min/max,
null rates). The only persisted statistics are each `StandardScaler`'s
`mean_`/`var_` inside `voltrade_ml_v2.pkl`. Serve-time vectors ARE persisted
incidentally: every deep-scored candidate's 34-feature dict goes to
`voltrade_shadow_candidates.json` (20 k cap, ≈72 days), and executed entries
carry `entry_features`. "Healthy" today = the ~11 constants above are constant;
drift detection must therefore use **variance ratio and %-constant per
feature**, not mean shift (a z-score of vxx_ratio=1.0 ≈ the training mean).

**Failure signature.** The whole ML block is `try … except Exception: pass`
(`bot_engine.py:1448, 1774-1775`): an import or feature-build error silently
leaves the score rules-only and `entry_features = None`. Nothing records
`model_type`, `regime_used`, or `win_probability` per candidate. Feature NaN
passes `float(x or 0)`; LightGBM treats it as missing, the sklearn fallback
raises → rules fallback.

### S4 — ML layer

**What it does.** Regime-conditional ensemble: one global LightGBM + up to
three regime models (bull/bear/neutral, each ≥50 train / ≥10 test rows of
that regime), each `StandardScaler → Booster`. Params `objective binary,
learning_rate 0.04, num_leaves 24, min_child_samples 10, feature_fraction
0.75, bagging_fraction 0.75, bagging_freq 5, scale_pos_weight max(1,
n_neg/n_pos), min_gain_to_split 0.01, num_boost_round 300, early_stopping 30`
on the test split. If p90−p10 of test probabilities < 0.08 the LGB model is
discarded for sklearn `GradientBoostingClassifier(150, depth 4, lr 0.08,
subsample 0.8, min_samples_leaf 8)` but still labelled `lightgbm_*`. Isotonic
calibration fitted on walk-forward OOS probabilities →
`voltrade_ml_calibrator.pkl`. Target: binary triple-barrier win. (Appendix C.)

**Labels (exact).**
- Generic rows: `_triple_barrier_label(bars, idx, atr_pct, pt_mult=1.5,
  sl_mult=1.0, max_days=5)` — entry `close[idx]`; upper `entry×(1+max(0.005,
  atr_pct×1.5/100))`; lower `entry×(1−max(0.003, atr_pct×1.0/100))`; vertical
  barrier bars idx+1…idx+5; touch test uses bar **high/low**, **high ≥ PT is
  checked first** so a same-bar double touch is a win; timeout relabel:
  `pnl > +0.3×atr_pct → +1`, `< −0.3×atr_pct → −1`, else 0; `y = 1 iff label
  == +1`. The docstring's "maps directly to actual bot behaviour" is false
  (live exits are ATR phases, thirds and a 7-day time stop).
- Feedback rows: `y = 1 iff pnl_pct > 1.5×entry_features.atr_pct`; **`pnl_pct
  None → 0 → label 0`** (open trades trained as losses); |pnl| > 20 dropped.
- Shadow rows: +2% target / −4% stop / 10 bars; double touch → 0.
- Three label definitions are mixed in one matrix.

**CV (exact).** `_walk_forward_cv(X, y, n_splits=5, embargo_days=5,
n_tickers_approx)`: `fold_size = n//6`; fold k trains on `[0, k·fold)`,
embargo `5×n_tickers` rows, tests `[k·fold+embargo, (k+1)·fold)`; skip if
train < max(500, fold) or <50 train / <10 test rows; metric = **accuracy at
0.5** averaged (`wf_accuracy`); per-regime fold accuracy; no AUC/logloss/
Brier. Production holdout `_purged_train_test_split`: test = last 20%,
embargo `min(5×n_tickers, n//5)`, **no purge of overlapping label windows**.
**Rows are appended ticker-by-ticker, then feedback, then shadow**, so folds
and the holdout split by *ticker over the same dates*, not by time (V15):
market-wide features act as date fingerprints and the reported accuracy is
inflated; feedback/shadow rows (appended last) land mostly in the holdout.

**Training data.** Universe = most-actives top 50 + 30 core names, capped
100 (`fast_mode`); window 365 calendar days (config says `ML_LOOKBACK_DAYS
60`; code `max(days, 365)`); bars via `bars_feed()` = **iex** under the
production 403 state, and rows with volume < 300,000 are dropped → the
training universe silently shrinks; fast mode keeps the last 10,000 rows.
Feedback blend: records with `entry_features` only (so **only regular-hours
stock entries can become training rows**), weight `3.0×2^(−age/30)×legacy`,
≥15 records, capped at 15% of generic rows; `code_version < 1.0.33` weighted
0.4× (moot — boot cleanup deletes every non-seed record < 1.0.34).
Config-fingerprint weighting (`_config_similarity_weight`) has **no caller**.
Shadow blend: `load_shadow_data(horizon 10, weight 0.3)`, ≥50 labeled records.

**Retrain trigger and where it is logged (the directive's explicit ask).**

| path | trigger | budget | live? | logs |
|---|---|---|---|---|
| Tier 3 hourly `tier3Strategic` step 1 (`bot.ts:5713-5798`) + once at boot+30 s | inline Python: model missing **or** `time.time() − mtime > 86400` | `execPythonSerialized("python3 ml_retrain_safe.py", {timeout: 900000})` → effective cap min(90 s, and the "trivial" 10 s path, V01) | **the only automatic path** | `TIER3 "ML model stale or missing — triggering retrain..."` → `TIER3 "ML retrain complete — status: X, accuracy: Y, features: N/A, samples: Z"` **or** `TIER3-ML-ERROR "ML retrain failed: …"` (only when status starts `error:`/`failed`; `insufficient_data`, `skipped`, wrapper `error`, `fatal` are audited as *complete*) **or** `TIER3-ML-ERROR "(code=… signal=SIGKILL)[likely OOM kill…]"` (the exec-cap kill, misclassified); `TIER3 "ML model fresh (Nh old) — skipping retrain"` otherwise; an age-check mutex timeout is `console.error` only (that hour silently skipped) |
| "4 am daily retrain" (`bot.ts:6199-6220`) | `Math.floor(etHour) === 4` inside `runOvernightResearch` | — | **unreachable** (V02) | `RETRAIN`/`RETRAIN-ERROR` never appear; `/api/bot/ml-status.retrain_schedule` still says "Daily at 4am ET" |
| `POST /api/ml/retrain` (`routes.ts:7162`) | manual, **any logged-in user** | raw exec 300 s, no mutex | manual | writes `ml_status.json` (the only writer) and sets `ml_toggle.json {enabled:true}` if `steps` includes `training_done` |
| `bot_engine.py train` / `VOLTRADE_INLINE_ML_TRAIN=1` | manual | caller's | manual | also trains the options model (the only path that does) |

What a run writes: `voltrade_ml_v2.pkl` bundle `{model, scaler, regime_models,
feature_names, accuracy, regime_accs, samples, feedback_trades, data_source,
timestamp, feature_importance, label_method, cv_method, architecture,
calibrator_path}` — **no accuracy/AUC gate before overwrite**; the calibrator
is written at step 5b *before* the global model, so a run dying after 5b
leaves a new calibrator paired with the old model. `ml_retrain_safe.py`
returns `{status, accuracy, regime_accs, features (34), samples,
feedback_trades, data_source, label_method, cv_method, regime_models,
wf_fold_accs, calibrated, training_seconds, steps[]}` and overwrites
`voltrade_last_retrain.json` (+`completed_at`), which **nothing reads**.

**Scoring.** `ml_score(features)` → `{ml_score (10..90), ml_confidence (≤0.8),
ml_signal BUY>60/SELL<40/HOLD, model_type lightgbm_global|lightgbm_<regime>,
win_probability, regime_used}`; regime routing via `classify_regime(vxx_ratio,
spy_vs_ma50)` — with live vxx pinned at 1.0: bear iff spy < 0.94, else
neutral, **never bull**. Fallback `_rule_based_fallback` (model missing, >24 h
old = `ML_MAX_AGE_HOURS`, load error) returns `model_type "fallback_rules"`;
`deep_score` only excludes the literal `"fallback"` [grep] → the heuristic is
blended at the ML weight and labelled "ML model: …". Blend: `final = rules ×
rule_w + ml_eff × ml_w`, `ml_eff = 50 if |ml−50| < 8`, weights from
`voltrade_blend_tracker.json {rule_weight, ml_weight, updated}` (default
0.6/0.4; written by `diagnostics.run_diagnostics` from the last-20 **bot win
rate**, not ML accuracy: wr > 60 → ml_w up to 0.55; wr < 40 → down to 0.20;
at the live recent WR of 20% ≈ 0.72/0.28). The ML on/off toggle
(`ml_toggle.json`) is read by status reporters only [grep] — **ML always
blends; the UI shows OFF by default**.

**Healthy values.** `model_age_hours` cycling 0 → ~25; live `ml_score` ∈
[10, 90] with `ml_confidence ≤ 0.8` (any value outside [10, 90] proves
`fallback_rules`); one `TIER3 "ML retrain complete — status: trained,
accuracy: ~55-60, samples: 10000"` per ~25 h and zero `TIER3-ML-ERROR`;
`training_seconds` well under the exec cap; `/api/ml/status.last_accuracy 58`
(manual-retrain record). No historical accuracy is recorded anywhere, so
there is no baseline; the positive-class base rate is not reported.

**Failure signature.** Model > 24 h → silent `fallback_rules` blended as ML
(nothing persists `model_type`); retrain killed by the exec cap → `TIER3-ML-ERROR
… SIGKILL [likely OOM]` (misclassified); retrain `insufficient_data` → audited
as "complete"; bars fetch fail → `TIER3-ML-ERROR … Could not fetch training
bars (HTTP …)` (surfaced); Tier 3 not running (bot stopped/killed-in-process)
→ model goes stale silently; partial retrain (calibrator newer than model) →
undetected; constant predictions → undetected (no prediction log). `/api/health`
has **no ML check**. Six different "stale" thresholds coexist (24 h actual,
48 h UI, 168/240/252 h elsewhere).

### S5 — Rules, vetoes and filters (the evaluation order, with thresholds)

**What it does.** A candidate passes ~60 gates between the universe and an
order. Appendix D §2 is the complete table (file:line, threshold source,
regime dependence, what each records). Essentials, in order:

**Scan-level (Python, before candidates).** A1 `VOLTRADE_ULTRA_SURVIVAL_MODE`
→ 0 stocks (stderr). A2 **DD halt** `DRAWDOWN_HALT_PCT 18` (confirm ≥9% on a
re-read ≥60 s later; resume only in BULL/NEUTRAL within 5% of peak) → whole
scan returns `halted` → audit `DD-HALT`. A3 universe symbol filter (silent).
A5 price < `MIN_PRICE 5` (hardcoded `be:201`; `sc:66` not read). A6 volume <
500k (100k 09:30–09:59; `sc:68 OPEN_HOUR_MIN_VOLUME 1M` not read — contradicts
its own comment). A8 extreme-move penalty −15/−30 to quick score. A9
`SURVIVAL_MODE` RSS ≥ 700 MB. A10 deep cap 5. A11 mem guard 550/400 MB. A12
35 s wall cap. A13 analyze.py error / no options market → candidate dropped
with no shadow record. All SILENT except A2.

**Stock loop (Python `be:3238-3597`).**

| # | rule | threshold (source) | regime? | records |
|---|---|---|---|---|
| B1 | **max stock slots** → `break` (ends loop) | `get_adaptive_params().MAX_POSITIONS` − held equities (**floor ETFs count**): PANIC/BEAR/**NEUTRAL 0**, CAUTION 4→tier, BULL 8→tier; tier = 4/6/8/10/12… by equity ($0/50k/100k/250k/500k) | yes | SILENT; shadow still `taken` |
| B2 | already held | — | — | SILENT |
| B3 | stop-loss re-entry cooldown | PANIC 4h / BEAR 3h / CAUTION 2.5h / NEUTRAL 2h / BULL 1.5h (hardcoded; `STOP_COOLDOWN_SECONDS` not read); `voltrade_stop_cooldown.json` | yes | SILENT |
| B4 | no stock longs in BEAR/PANIC | `get_market_regime` | yes | SILENT |
| B5 | `MIN_SCORE` | BULL 63 / NEUTRAL 63 / CAUTION 67 / BEAR+PANIC 75 (`sc:62` + regime) | yes | SILENT (shadow uses fixed 63) |
| B6 | sector / correlation | same-sector count ≥ `MAX_SECTOR_POSITIONS` (tier: 2 under $500k); new beta ≥1.5 with ≥3 high-beta held; avg beta >1.6 (hardcoded `_beta_map`); fail-open | tier | **shadow `rejected_heat`**; option OCC symbols counted as stocks → "Unknown" sector blocked (V09) |
| B7 | dollar-volume floor | `_MIN_DOLLAR_VOL $50M` hardcoded | — | SILENT |
| B8 | blocked tickers | `BLOCKED_TICKERS` (DKNG, RBLX, SQQQ, TQQQ, SPXU, UPRO, UVXY, ABNB, DASH, LYFT) ∪ {UBER} | — | SILENT |
| B9 | extreme mover > +50% | hardcoded | — | `extreme_movers_today.json` |
| B10 | short → BUY rewrite | shorting disabled | — | silent overwrite of `side` |
| B11 | spread | (ask−bid)/mid > 0.5% hardcoded (`sc:80` not read); fail-open; quote discarded | — | **shadow `rejected_other`** |
| B12 | instrument selection | stock / 2× ETF / options; options need regular hours, score ≥ `MIN_OPTIONS_SCORE 65`, options exposure < 16%, no naked calls | partly | `instrument_reasoning` → audit `INSTRUMENT` at execution |
| B13 | **sizing block** | halted asset; price/equity ≤ 0; **negative Kelly for the bucket** (V18); shares ≤ 0 | indirectly | **SILENT** (`block_reason` discarded) |
| B15 | volume-share cap | position/dollar_volume > tier `MAX_VOLUME_SHARE_PCT` (5%…0.8%) | tier | PYLOG `VOL_SHARE_BLOCK` |

**Covered-call sweep** (`be:3621-3705`): any stock position ≥100 shares
(floor ETFs included) without a short call, not BEAR/PANIC, 7-day earnings
guard → `SELL COVERED CALL` score 75. These are the live "trade candidates"
(VXUS, KWEB) and are dead on arrival: slots full → `OPTIONS-SLOT-FULL`; if
through, `should_use_options` doesn't recognise the action → `OPTIONS-SKIP`;
off-hours → `QUEUE` → morning queue counts all 11 positions ≥ 8 → break (V11).

**Options scanner** (`options_scanner.py`): regular hours only; calendar
`should_skip_new_options()`; candidates c ≥ $10, v ≥ 1M, top 400 by |chg|;
setups `earnings_iv_crush` (0–7 d to earnings, VXX < 1.30, avg IV ≥ 0.40 →
iron_condor), `vxx_panic_put_sale` (VXX ≥ 1.30, SPY puts 92–95%, 14–30 DTE →
bull_put_credit_spread), `high_iv_premium_sale` (IVR ≥ 50, VXX < 1.25 →
iron_condor; Setup 7 VRP+ adds +4); score = `min(cap, 0.6·rules + 0.4·ML)`;
surface adjust ±5/−8; floor 60 (+5 short week). All rejections SILENT.

**Tier engine** (`tiered_strategy.py`): `check_kill_switches` (manual file;
DD ≤ −20%; 5 consecutive losses/24 h; daily ≤ −3%, VXX ≥ 1.40, correlation
> 60%, free BP < 10% are **warnings only** — `blocks_entries`/`tier_blocks`
have no consumer) → `master_kill_switch` (DD ≤ −20%, daily ≤ −3%, invested ≥
`MAX_TOTAL_EXPOSURE` PANIC .30 / BEAR .50 / CAUTION .60 / NEUTRAL+BULL .95)
→ T1 CSP core (slots 6 − held options; VXX size scalar .30/.60/.85; DD ×.5
≤ −15% / ×.75 ≤ −10%; csp_universe top 200; ≤3 per scan, **no score
threshold**) → T2 leverage (needs `multiplier ≥ 4`, DD > −8%; effect lost at
dispatch, V13) → T3 trend (never fires — regime always "NEUTRAL", V08) → T4
disabled → allocator.

**Node admission** (`executeTrades`, `bot.ts:5220-5705`), per trade: E1
killSwitch → break (silent). E2 options slots ≥ `MAX_OPTIONS_POSITIONS 6`
(`bot:361`, `sc:110`; Python uses **8** at `be:192`) → `OPTIONS-SLOT-FULL`.
E3 stock slots ≥ `MAX_POSITIONS 8` (all equities incl. floor) → **silent
break**. E4 `score < state.minScoreThreshold` (65; **75 live** via
diagnostics) → `SKIP`. E5 held. E7 `dailyBlockedTickers` → `SKIP`. E8
`position_value > equity × 0.20 × positionSizeMultiplier` → `SIZE-CAP`
(block, not resize). E9 `totalDeployed + value > equity × 0.30` (floor
excluded) → `HEAT-CAP` **break**. E10 sector share > 40% after +5% →
`CORR-CAP` (fail-open). E11 exchange halt → `HALT-SKIP`. E12 ≥10 attempts
→ `MAX-RETRY`. E13 open order on symbol → `DUP-SKIP`. E14 buying power →
`replaceIfBetter` (cancels the weakest open entry if new score > weakest+10;
tier CSPs are tracked with score 0 so they are cancelled first; the 204
response is misread as failure, V27). E15 instrument re-decision
(`should_use_options`, can disagree with the scan's selection). E16 short →
BUY (`SHORT-BLOCKED`). E17 submit → `TRADE "LIMIT BUY q TICKER @ ~$p |
Score: s | Rules: r ML: m Blend: b | $v (p%) | Sizing: …"`.

**Morning queue** (`executeMorningQueue`): stale > 16 h (`MORNING-STALE`);
slots count **all positions incl. options** ≥ 8 → break (`MORNING`); size >
20% (`MORNING-BLOCKED`); gap > 5% skip / > 2% re-qty (`MORNING-GAP`/
`MORNING-ADJUST`); qty < 1 (`MORNING-SKIP`); submit `MORNING-TRADE`.

**Node Tier-2 pre-gates.** Daily loss `evaluateDailyPnl ≤ −3%` → `TIER2-LIMIT`
+ return (no entries, no tier dispatch this cycle). Weekly loss: **dead**
(history of length 1 [grep]). `DIAGNOSTIC-PAUSE`: dead (`should_pause` never
set).

**Counterfactual logging (the RULE REVIEW requirement).** Exists as
`shadow_portfolio.py` → `voltrade_shadow_candidates.json` (20 k cap, atomic
under flock), record `{ticker, timestamp, score, decision, decision_reason,
entry_price, vxx_ratio, regime_label, side, action_label, trade_type (since
10-06), features{34}, outcomes{"+5d","+10d","+20d"}, code_version
"1.0.34-shadow"}`; outcomes labeled nightly at 22:00Z by `backfill_outcomes`
(+2%/−4% path-dependent, long-stock proxy for every record incl. option
sales). Decision buckets that are actually written: `taken` (= score ≥ 63),
`rejected_score`, `rejected_heat` (B6), `rejected_other` (B11),
`rejected_masterkill`, `rejected_capital` (CSP budget); `rejected_halt` and
`rejected_earnings` are in the docstring but never written. **Every other
veto — all of A*, B1–B5, B7–B10, B13–B15, the options scanner, T1/T2/T3
gates, kill suppression, and every Node gate — leaves no counterfactual
record.** Horizons are +5/+10/+20 d, not the +1/+5/+20 d CLAUDE.md asks for.
There is no `{date, ticker, rule, entry_price, score}` logger with a `rule`
field.

**Healthy values.** Scan audit "2–4 trade candidates"; shadow ≈ 277
records/day; `rejected_heat` a small minority; `TIER-KILL` and `KILL-WARN`
rare; `OPTIONS-SLOT-FULL` absent (live: 158×).

**Failure signature.** `KILL-WARN` repeating with no `KILL-SWITCH` = unenforced
warnings; `TIER-KILL "Portfolio invested X% >= Y%"` frequent = floor + book at
the regime cap, **all CSP entries blocked** (KNOWN BROKEN #20 class);
`OPTIONS-SLOT-FULL` floods = slot saturation + candidate churn (and it crowds
the 2000-row audit log); `MORNING-SKIP qty=0` = options/covered-call trades
queued off-hours; composite scores stuck 25–45 with empty `reasons` =
enrichment collapsed to defaults.

### S6 — Position sizing

**What it does.** `position_sizing.calculate_position(trade, equity,
current_positions, macro)` (`position_sizing.py:878-1193`), called from the
stock loop with the deep-scored candidate (the instrument decision is **not**
merged in). `size_portfolio()` is unused by the scan. (Appendix E §1.)

**Formula.**
```
kelly_base = _kelly_fraction(bucket.win_rate, bucket.avg_win, bucket.avg_loss)
   f* = (WR·W − (1−WR)·L)/W ; f* ≤ 0 → 0 → BLOCK "Negative expectancy for strategy '<bucket>'"
   else f*/3 clamped [0.01, 0.08]            (KELLY_DIVISOR config key not read)
final_pct = kelly_base × s_volatility × s_confidence × s_regime × s_earnings × s_time × s_heat × s_liquidity
            × dd_scalar (DEAD, V19) × stress_mult (only if < 1)
final_pct = clamp(final_pct, 0.01, 0.08)     (ABSOLUTE_MIN/MAX_POSITION_PCT)
shares = floor(equity × final_pct / price)
```
Bucket (`_infer_strategy`): `options_strategy` keyword → `csp_options`;
`instrument == "etf"` or known leveraged ETF → `etf`; dominant factor score
`vrp` → `vrp`; otherwise → `stocks`. Bucket stats (`_get_historical_stats`):
hardcoded priors unless ≥10 closed feedback records; per-bucket override only
from records whose `strategy` key equals the bucket (≥5) — **live
`track_fill` records never carry `strategy`**, only `_seed` records do (which
`_get_historical_stats` does not exclude).

| bucket | prior (WR, W, L) | Kelly base |
|---|---|---|
| `stocks` | 0.49, 4.19, 4.25 | **f* = −0.027 → 0 → BLOCKED** (comment says "small starter size" — wrong) |
| `etf` | 0.558, 3.82, 3.55 | 4.91% |
| `vrp` | 0.60, 2.0, 2.0 | 6.67% |
| `csp_options` | 0.716, 0.69, 0.46 | 8.00% (cap) |

Scalars: volatility `2.0/max(daily_atr%, 0.5)` ×0.80 if GARCH > 1.5×ATR,
clamp [0.25, 1.20]; confidence `0.6 + (score−65)/35×0.55` ×1.10/0.75 by
`ml_confidence`, clamp [0.4, 1.25]; regime by VIX (<15 1.10 / 15–20 1.0 /
20–25 **0.80** / 25–30 0.65 / 30–40 0.50 / ≥40 0.30 — the docstring bands
differ); earnings (>7 d 1.0 / 5–7 d 0.85 / 2–4 d 0.60 / 0–1 d 0.35 /
lookup fail **0.85 fail-closed**); time-of-day (09:30–10:00 0.80 / 10:00–15:30
1.0 / 15:30–16:00 0.70 / else 0.50); heat = `max(0.4, 1 − 0.10×n_active)`
(**only QQQ excluded; SMH/KWEB/VXUS/GLD count → ≈×0.6 with the live floor**)
× deploy (>90% 0.5 / >80% 0.7 / >70% 0.85) × sector (never fires — Alpaca
positions have no `sector`); liquidity by % of dollar volume (>2% 0.25 / >1%
0.5 / >0.5% 0.75) else by share volume (<500k 0.5 / <1M 0.7 / <5M 0.85);
drawdown scalar **dead** (reads `peak_equity` from `voltrade_equity_peak.json`
whose Node key is `equityPeak`); stress (<30 1.0 / ≥30 0.75 / ≥55 0.5 / ≥75
0.0 → **clamped back to 1%**).

**Real fields (output, `:1173-1193`).** `ticker, shares, price, side,
position_value, position_pct, stop_loss, take_profit, stop_distance_pct,
tp_distance_pct, risk_per_share, risk_total, max_loss_pct, costs{commission_entry,
commission_exit, estimated_slippage_pct, total_cost_pct, total_cost_dollars},
scalars{kelly_base, volatility, confidence, regime, earnings, time_of_day,
portfolio_heat, liquidity, drawdown_scalar, drawdown_pct, stress, stress_level,
stress_value, combined}, reasoning, blocked, block_reason, hist_stats{win_rate,
avg_win, avg_loss, total_trades}`. Of these the trade dict keeps `shares,
position_value, stop_loss, take_profit, sizing_reasoning, sizing_scalars`;
Node keeps `shares`/`position_value` and the first 150 chars of
`sizing_reasoning` in the `TRADE` line. **`stop_loss`, `take_profit`,
`costs`, `scalars` are never sent, enforced or persisted.**
`SIZING_HISTORY_PATH` is declared and never written.

**Other sizing paths.** Options (`_dynamic_options_size`): `csp_options`
Kelly × same scalars × `OPTIONS_LEVERAGE_DISCOUNT 0.50`, clamp [1%, 8%];
scanner/tier paths pass no trade context → liquidity 0.5 → ≈1.0–1.4% of
equity. CSP contract: budget `max(equity×size_pct, equity×0.02)` capped by
options BP, **stretch to 20% of equity** for the smallest OTM strike, qty
`min(budget/(strike×100), 2)`, delta ≈ −0.30, limit `mid − 0.30×(mid−bid)`
— the tier's `size_pct` and `metadata` never reach `select_contract` (V13).
ETF path: `etfShares = max(1, floor(trade.shares/2))` applied to a
differently priced ETF — dollar exposure unrelated to `position_value`.
T3: `min(levered, 2×MAX_POSITION_PCT)` each for SPY/QQQ (never fires).

**Healthy values.** `final_pct` 1–8%; `sizing_reasoning` "N shares @ $p = $v
(x% of portfolio) | Kelly base: … × Vol … × Heat …"; `TRADE` lines appearing.

**Failure signature.** Zero `TRADE` lines with scans reporting candidates
(live); `block_reason` is discarded so a Kelly block looks identical to any
other silent drop (V18); `hist_stats.total_trades 0` for the `stocks` bucket
while 348 live records exist = the `strategy`-key gap.

### S7 — Execution: signal → order → broker → fill, and what is recorded

**What it does.** (Appendix F §1 — all 13 Node and 11 Python order sites.)
Node builds every order with `getOrderParams(price, context)`
(`server/orderParams.ts`): `new_entry` = **DAY limit at round2(price ×
1.001)** in regular hours (×1.005 + `extended_hours` outside) — *not* "at
ask"; `stop_loss`/`trailing_stop` = market in RTH (limit ×0.995 extended);
`take_profit` = limit at price. Raw POST `alpaca("/v2/orders")` (frozen):
15 s abort, throws on non-2xx, **calls `r.json()` on every 2xx so a 204 from
a single-order DELETE throws**. No `client_order_id` is ever set [grep]; the
Alpaca order id lives only in the in-memory `openOrders[]`
(`TrackedOrder{orderId, ticker, score, placedAt, side, qty, limitPrice,
isExit?, pendingExitFill?, cspUnderlying?}`). Fill detection is **polling
only** (no `trade_updates` stream [grep]): one poll 2 s after a Tier-2 batch;
`sweepStaleOrders()` every 45 s **RTH only, active only** (cancels tracked
orders > `STALE_ORDER_MINUTES 12`, audit `SWEEP`, resolves
`pendingExitFill`); options entries resolved when the OCC symbol appears in
`/v2/positions`.

**What is recorded at each step (stock entry).**

| step | recorded | where | lost |
|---|---|---|---|
| decision | `trade.price` = snapshot `dailyBar.c` (15-min delayed under `delayed_sip`), `score`, `entry_features` | memory → feedback `expected_price` | quote bid/ask/mid (fetched for the spread filter, discarded), observation time, `stop_loss`/`take_profit` from sizing |
| submit | audit `TRADE` text; `addPositionToMonitor` **before fill**; `openOrders` push | SQLite (2000 rows) / memory | order id (not in audit text), limit price durably |
| fill check | one GET after 2 s | — | any fill after 2 s |
| record | `track_fill` payload `{ticker, order_type:"market" (**wrong**), side, qty_requested, qty_filled: filledQty ‖ requestedQty, partial_fill, expected_price, alpaca_fill_price, fill_price: alpaca × (1 ± random haircut), slippage_applied_pct, order_status, time_placed (= submit+2 s), session "regular", volume: 1000000, score, instrument, entry_features, code_version}` | `voltrade_trade_feedback.json` | `track_fill` keeps only `ticker, side, qty, expected_price, fill_price, slippage_pct (unsigned), volume, session, time_placed, time_filled (= Python now), score, entry_features, outcome null, pnl_pct null, code_version, config_fingerprint, config_snapshot` — **drops `alpaca_fill_price`, `slippage_applied_pct`, `order_type`, `qty_requested`, `partial_fill`, `order_status`, `instrument`** |

The synthetic haircut (`bot.ts:5660-5668`): tiers by `trade.volume` (>20M
0.02–0.05% / >5M 0.05–0.10% / >1M 0.08–0.15% / else **0.12–0.25%**) — but the
trade dict has no `volume` key, `|| 1000000` is not `> 1000000`, so **every
regular entry draws the illiquid tier**; morning queue 0.15–0.25% applied to
the *signal price* with no fill read; ETF entries record the **underlying's**
price as both expected and fill (V22/V28). `backtest_v2` charges by real
volume → a built-in live-vs-backtest cost gap. Unfilled stock entries become
phantom "filled" records with requested qty and the signal price, never
corrected.

**Other entry paths.** Options scanner path → `select_contract` →
`submit_options_order` → `register_options_entry` →
`voltrade_options_state.json`. **Tier-dispatcher SELL_CSP / BUY_PUT (the main
options path) never call `register_options_entry` → no feedback entry, ever**;
`manage_options_positions` later initialises the unknown OCC with `strategy
"unknown"` and `entry_timestamp now` (V24). Tier-3 BUY fetches its price from
the trading host (no market data there) → price 0 → `T3-SKIP` always (V27).
Manual `/api/bot/trade` and `/api/bot/close`: no feedback, no monitor
registration.

**Exit recording.** WS full exits and `POS-KILL` carry `pendingExitFill{ticker,
exitSide, pnlPct, exitReason, entryDate, codeVersion}` and are recorded only
on Alpaca `status == "filled"` at `filled_avg_price`/`filled_qty`
(`buildExitFillPayload`: `{ticker, side, qty_filled, fill_price, expected_price
(= fill_price!), time_filled (= sweep time), session "regular", exit_reason,
exit_context{pnl_pct (TRIGGER-time), exit_reason, days_held}, code_version}`);
`track_fill` matches the newest open record **by ticker only** and lets
`exit_context.pnl_pct` override fill-based P&L. Options single-leg exits are
recorded at **submission** with the mark (`_record_options_exit_feedback`);
an unfilled close is re-recorded next cycle → duplicate `orphan_exit`. Never
recorded: scale-out legs (by design, `exitFill.ts:14-17`), TIME-EXIT,
CC-UNWIND, LIQUIDATE-ALL, manual close, options rolls, multi-leg closes
(KNOWN BROKEN #12c), every Python floor/hedge order (also unaudited by Node).

**Paper vs live.** Node: `ALPACA_BASE = "https://paper-api.alpaca.markets"`
hardcoded (`bot.ts:319`); Python: env `ALPACA_BASE_URL` (default paper);
`csp_universe.py:180` reads a **different** var `ALPACA_TRADING_URL`;
`mode: "paper"` in `/api/bot/account` and `/api/bot/status` is a string
literal [grep]. The only computed check is `allPaperAlpacaUrls()` used for
auto-resume eligibility. A live `ALPACA_BASE_URL` would split-brain the
system while the UI still says paper.

**Healthy values.** Orders filled within seconds (last 100: equities market
filled ~1 s; option limits fill or expire at the close); `SWEEP` rows ≤ a few
per day; `orphan_exit` share low; `fills_count` growing with `TRADE` lines.

**Failure signature.** Alpaca open orders > 13 min in RTH or any age outside
RTH/killed/after restart (tracker empty); feedback entry with `outcome null`
and no nearby Alpaca fill (phantom); `orphan_exit` share high (live 293/348 =
84%); `FILL-CHECK-WARN`; `ORDER-REJECTED`; `PARTIAL-FILL`; `track_fill`
failures are swallowed at debug level and Node `.catch(()=>{})` — **no audit
line**; `REPLACE` never followed by a `TRADE` for the new ticker (V27b).

### S8 — Position management and exits (Node WS engine, scale-outs, options, floors)

**What it does.** Three executors fire exits; the Python "exit engine" is not
one of them. `bot_engine.manage_positions()` runs every Tier-1 cycle and
inside every scan but its actions are only audited as `POS-MONITOR-SYNC`; it
**writes** `voltrade_stop_state.json` (and the stop cooldown) which Node
adopts. Stock/ETF exits that fire are Node `checkPositionOnTick(ticker,
close)` (`bot.ts:6720-7222`) on each 1-min IEX bar; `syncMonitoredPositions()`
(every 60 s, **not RTH-gated**, plus Tier 1) fires `POS-KILL` and `TIME-EXIT`;
option exits/rolls fire inside Python `options_manager.manage_options_positions()`
each scan; floors, third leg and the convexity overlay submit their own orders
inside Python each scan. (Appendix E §2–4.)

**Node rule order per bar** (monitor state: `ticker, side, entryPrice (Alpaca
avg_entry_price after sync), qty, phase, initialRiskPct, highestPnl,
currentStopPct, tpPct, entryDate, originalQty, remainingQty, scalesCompleted,
breakevenActive, regime, atrPct, highestPrice, pendingExit, positionType…`;
**1R = 2×ATR% at first observation** (`be:2069`), not the sizer's stop):

1. **Scale-out** (returns early when fired): `scalesCompleted < 2 &&
   remainingQty > 1`, `thirdQty = floor(originalQty/3)`; leg 1 at R ≥ 1.0;
   leg 2 at R ≥ 3.0 (BULL) / 1.5 (BEAR, CAUTION, PANIC) / 2.0 (NEUTRAL);
   DAY limit at the bar close (`take_profit` params); counters advance **at
   submit**; no `pendingExitFill` → **never fill-confirmed, never recorded**;
   audit `WS-SCALE-OUT "SCALE-OUT 1/3 at +x% (yR) | LIMIT sell q of N TICKER @
   $p (r remaining)"`.
2. Stop level by phase: 4 → `atr × (3 bull / 1.5 bear / 2 neutral)` from
   `highestPrice`; 3 → `max(1.0, atr)`; 2 → `max(1.5, 1.5×atr)` points down
   from peak P&L; 1 → `max(1.5, min(2×atr, 8))` (4.0 if just added). No
   `STOP_SCALAR_*`, no minimum hold, no −20% hard stop (Python-only, advisory).
3. Trigger: `trailing_stop` / `stop_loss` / breakeven (phase ≥2,
   `breakevenActive`, pnl < 0 — fills slightly below entry).
4. Take-profit: `pnl ≥ tpPct (= max(4, min(3×atr, 15))) && phase < 3 &&
   scalesCompleted ≥ 2` — unreachable in BULL and NEUTRAL (leg 2 needs phase
   ≥3), fires right after leg 2 in CAUTION/BEAR/PANIC.
5. Time stop: ≥7 calendar days and |pnl| < 2% (`TIME_STOP_DAYS` config not
   read).
6. Fire: CC unwind first (`CC-UNWIND`); skip if any open sell order; qty =
   `remainingQty`; market (stops) / limit (TP); `pendingExitFill`; audit
   `WS-EXIT "<reason> (final r/N shares, k prior scale-outs) | MARKET|LIMIT
   sell q TICKER @ $bar"`; stop exits write `voltrade_stop_cooldown.json` and
   drive the circuit breaker.

**Scale-out hazards (V16).** Python `manage_positions` also advances
`scales_completed` / `remaining_qty` in the shared stop-state file **without
placing an order** (`be:2083-2109`); Node adopts `max(memory, file)` →
a real leg can be skipped and shares stranded; unfilled legs count as done;
`pendingExit` is cleared only by the sweep, so a position with leftover
shares can stay frozen until restart.

**POS-KILL / POS-WARN** (`syncMonitoredPositions`): stock ≤ −25%, short option
≤ −150% (mark ≥ 2.5× credit), long option ≤ −75% (`posKillThresholdPct`,
`isPosKillExemptOption`), full live qty, `exitReason "position_kill"`;
`POS-WARN` at −15% hourly (live: CRWV put ×34); `POS-KILL-OPTION-HOLD` daily
between −25% and the backstop. **TIME-EXIT** (14 d, pnl −5…+2%) reads
`pos.created_at ‖ submitted_at` which Alpaca Position objects don't carry →
probably never fires (V17c).

**Options manager** (constants `DTE_EXIT_THRESHOLD 21, DTE_CRITICAL 5,
PROFIT_TARGET_PCT 0.50, LOSS_LIMIT_MULTIPLIER 2.0, IC_MAX_LOSS_EXIT_PCT 0.50,
GAMMA_THRESHOLD 0.08, DELTA_DRIFT_THRESHOLD 0.25, ASSIGNMENT_DELTA_THRESHOLD
0.80, ROLL_MIN_CREDIT 0.05, MIN_HOLD_MINUTES 60`), in order per standalone
leg: DTE ≤5 → `dte_critical` (mid); hold < 60 min → skip (for tier CSPs the
clock starts when the manager first *sees* the position); |delta| > 0.80 &
DTE ≤10 → roll else `assignment_close` (ask); DTE ≤21 → roll else
`dte_close`/`dte_close_bought`; `(credit−price)/credit ≥ 0.50` →
`profit_target`; `price/credit ≥ 2.0` → `loss_limit` (long: 50% →
`bought_loss_limit`; comment says 100%); |gamma| > 0.08 & DTE ≤30 →
`gamma_risk`; delta drift > 0.25 → warning. Rolls are two non-atomic DAY
limits. Multi-leg groups close via market mleg, unrecorded.

**Floors and hedges** (every scan, **skipped while DD-halted**): floor regime
debounce `FLOOR_REGIME_CONFIRM_CYCLES 3` (escalations immediate; VXX
fallback holds); SPY/QQQ basket target `SPY_FLOOR_<regime>` BULL .70 /
NEUTRAL .90 / CAUTION .35 / BEAR 0 / PANIC 0, death-cross caps (.20 / .50),
weights QQQ .50 SMH .15 KWEB .10 VXUS .15 CASH .10, trade a member on ≥3%
drift with **market** DAY orders in RTH; **at target 0 only QQQ is sold —
SMH/KWEB/VXUS have no exit owner** (V21); defensive GLD
`DEFENSIVE_FLOOR_<regime>` .08/.08/.15/.30/.40, death-cross ≥ .25, no RTH
guard; third leg SVXY `LEG3_VRP_PCT 0.40` (docstring says 15%) entered when
1.05 ≤ VXX ≤ 1.25 and VXX falling, exited at market in NEUTRAL/BULL, no stop;
**convexity overlay runs every scan although `CONVEXITY_OVERLAY.enabled =
False` is never read** [grep], budget 2–4% of equity in QQQ puts ~20% OTM,
DTE 45–75, orders with `side: "buy_to_open"/"sell_to_close"` (V20); SQQQ
cleanup every scan; T4 disabled.

**When the Node kill switch is latched: nothing manages positions** — Tier 1
runs only the auto-resume tick, the position-sync interval is gated on
`!killSwitch`, `checkPositionOnTick` returns when killed, Tier 2 (and so
options manager, floors, hedges) is skipped, and all open orders including
exit orders were cancelled at the trip (V17a). When the Python DD halt is
active, option exits, floors and hedges freeze while Node WS stock exits keep
running.

**Real fields.** `voltrade_stop_state.json` per ticker `{initial_risk_pct,
highest_pnl, highest_price, phase, current_stop_pct (= raw ATR%),
r_multiple, entry_date, entry_timestamp, days_held, original_qty,
remaining_qty, scales_completed, breakeven_active, regime,
short_hold_suppression?}`; `voltrade_options_state.json` per OCC `{entry_price,
entry_delta, entry_date, entry_timestamp, initial_credit, max_profit_target,
highest_value, strategy, setup, ticker, side, qty, max_loss,
pending_entry_feedback?, current_delta, current_gamma, current_theta,
current_price, dte, last_checked}`; `voltrade_stop_cooldown.json {ticker:
epoch_s}`; `voltrade_floor_state.json {floor_regime, floor_regime_raw,
floor_regime_pending, floor_regime_pending_count, floor_regime_last_obs,
last_regime, last_rebalance}`; `voltrade_defensive_floor_state.json
{last_regime, last_rebalance}`; audit `WS-SCALE-OUT, WS-EXIT (+ -WARN/-ERROR),
POS-KILL (+ -SKIP/-WARN/-ERROR/-OPTION-HOLD), POS-WARN, POS-MONITOR,
POS-MONITOR-SYNC, POS-QTY-CORRECTED, TIME-EXIT, CC-UNWIND, STREAM`;
`/api/bot/positions` per row `{ticker, qty, side, entryPrice, currentPrice,
marketValue, pnl, pnlPct, stopPrice, takeProfitPrice, phase, rMultiple,
highestPnl, daysHeld, asset_class, isOption, optionMeta{underlyingTicker,
expiry, strike, optionType, strategy, setup, daysToExpiry}}` — **`stopPrice`
is synthesised from `current_stop_pct` (1×ATR) and is not the stop the engine
enforces (2×ATR clamped 1.5–8%)**.

**Healthy values.** `STREAM "Real-time feed live — 43 tickers"` once per
boot; `stop_state` entries == held non-floor stocks; `WS-SCALE-OUT` followed
by a fill (visible only in Alpaca); `POS-QTY-CORRECTED`/`POS-MONITOR-SYNC
(scale_out)` rare; `POS-KILL` rare.

**Failure signature.** No `STREAM` line / `STREAM-ERROR` → stops are starved
(IEX-only prints also mean thin names get few bar ticks); `WS-SCALE-OUT`
followed by `SWEEP … sell` of the same qty = leg counted but not filled;
`POS-MONITOR-SYNC (scale_out)` without a matching `WS-SCALE-OUT` = phantom
Python leg; `WS-EXIT "skipping — N open sell orders"` repeating = stuck exit;
`WS-EXIT-ERROR` 403 = qty > held; `killSwitch true` > 2 market hours = loop
dark **and no exits of any kind**; `DD-HALT` each scan = options/floors
frozen.

### S9 — Portfolio risk: six halts, three peak files, auto-resume

| mechanism | trigger | peak / state | effect | latch & resume | audit |
|---|---|---|---|---|---|
| **Node drawdown kill** `checkDrawdownKillSwitch` (Tier 1) **and** the owner `GET /api/bot/account` route | `evaluateDrawdown(equity, equityPeak, −10) ≤ −10%` (invalid reads skipped) | `voltrade_equity_peak.json {equityPeak, savedAt}` | `killSwitch true`, `active false`, **DELETE all open orders**, email; `LIQUIDATE-ALL` only with `VOLTRADE_LIQUIDATE_ON_KILL=true` and ≤ −25% | latched in `voltrade_kill_switch.json`; cleared by owner toggle (does not re-activate) or **auto-resume** (paper only, `drawdown-kill` reasons only; cooldown 30 min; 3 consecutive OK evaluations in RTH; bases `RECOVERED` (DD > −8%, peak kept), `DATA_ANOMALY` (positions+fills explain < 25% of drop; peak re-based), `PAPER_REBASE` (latched ≥ 13 market hours; peak re-based)) | `DRAWDOWN-KILL`, `EQUITY-READ-INVALID`, `KILL-AUTO-RESUME-EVAL` (on change), `KILL SWITCH AUTO-RESUME` |
| **Python DD halt** `update_equity_peak` (scan pre-gate) | DD ≥ `DRAWDOWN_HALT_PCT 18`; confirm ≥ 9% on a read ≥ 60 s later else `anomaly_release` | `voltrade_portfolio_dd.json {peak_equity, last_equity, last_updated, halted, halt_reason, halt_started_at, halt_confirmed, halt_trip_ts, halt_trip_equity, last_anomaly_release{…}}` | scan returns `halted` → **no candidates, no option exits, no floors, no hedges** | one-way; resumes only in BULL/NEUTRAL within 5% of peak; re-based by `paper_resume_sync` on auto-resume | `DD-HALT`, `DD-HALT RELEASED…` |
| **`risk_kill_switch.check_kill_switches`** (frozen file) | manual `voltrade_MANUAL_KILL`; DD ≤ −20%; 5 consecutive losses / 24 h | `voltrade_killswitch.json {killed, kill_reason, killed_at, peak_equity, consecutive_losses, last_loss_ts, daily_loss_date, daily_loss_pct}`, history `voltrade_kill_history.json` | tier engine skipped; `suppress_entries_on_kill` empties all scanner trades (tickers → `kill_suppressed_trades`, which Node never reads) | DD kill latches; auto-clears after ≥ 4 h with equity ≥ 95% of peak and VXX < 1.20; loss kill clears 24 h after last loss | `KILL-SWITCH FIRED`, `KILL-WARN` (daily ≤ −3%, VXX ≥ 1.40, correlation > 60%, free BP < 10% are **warnings only** — nothing reads `blocks_entries`/`tier_blocks`) |
| **`tiered_strategy.master_kill_switch`** | DD ≤ −20%; daily ≤ −3%; invested ≥ `MAX_TOTAL_EXPOSURE` (regime) | `voltrade_peak_equity.json {peak_equity, last_updated}` (never lowered; not re-based by auto-resume) | all tier actions blocked | re-evaluated each scan | `TIER-KILL`; shadow `rejected_masterkill` |
| **Node Tier-2 daily loss** | `evaluateDailyPnl ≤ −3%` | — | skips `executeTrades` and tier dispatch this cycle (the Python scan incl. option exits already ran) | not latched | `TIER2-LIMIT` |
| **Node circuit breaker** | 3 WS stop/trailing exits | in-memory `consecutiveStopLosses` | if `dailyPnL/equityPeak ≤ −5%` pause to ~20:00 (min 1 h) — `state.dailyPnL` is refreshed only by the dashboard account route (V05b); else ≥2 unique tickers → 1 h; else `TICKER-BLOCKED` | reset at the (unreachable) 4 am branch, owner route, or restart | `CIRCUIT-BREAKER`, `CIRCUIT-BREAKER-RESET`, `TICKER-BLOCKED` |
| Weekly loss `check_weekly_loss` (−8% / −4%) | **dead** — called with a one-point history [grep] | — | — | — | `WEEKLY-LIMIT`/`WEEKLY-WARNING` never |

Four drawdown thresholds (−10, 18, −20, −20) over three peak files that
already disagree (Python $111,736.69 vs Node $110,727.04 on 2026-10-13);
`/api/monitoring/drawdown` reports **−20** with the Python peak while
`/api/health` and `/api/monitoring/overview` report −10 with the Node peak.
Python never reads Node's latch. The equity-peak **re-baseline** on
auto-resume erases drawdown-from-inception (the prior peak survives only in
an audit line).

**Healthy values.** `/api/bot/status.killSwitch false`,
`autoResume.outcome NOT_LATCHED|NOT_EVALUATED`, no `DD-HALT`/`TIER-KILL`/
`TIER2-LIMIT` rows, `KILL-WARN` occasional.

**Failure signature.** `status "killed"` > 2 market hours = loop dark, no
exits; `autoResume.outcome INELIGIBLE` = owner/unknown latch only the owner
can clear; stuck `UNRECONCILED` = waiting for `PAPER_REBASE`; `DD-HALT` each
scan with `dd_pct` far from Node's number = bad peak; `TIER-KILL "Portfolio
invested X% >= Y%"` frequent = floor + book at the regime cap, all CSPs
blocked. **Live now:** −10.04% with the kill unlatched — the next RTH Tier-1
cycle or dashboard load will latch it (V37).

### S10 — State inventory (where positions, equity, history live; the database)

`DATA_DIR` = `/data/voltrade` if `/data` exists else `/tmp`
(`storage_config.py:10-15`; Node hardcodes the same pair per file and falls to
`/tmp` only when the write throws — a `/tmp` fallback is lost on the next
deploy and the stale `/data` copy wins at boot; failures are `console.error`
only). **SQLite:** `better-sqlite3` at `DB_PATH || (/data | cwd) + "/voltrade.db"`
(`server/auth.ts:12-16`) — i.e. `/data/voltrade.db`, *outside* DATA_DIR; the
bot's only table is `audit_log(id INTEGER PK, time TEXT, type TEXT, message
TEXT)` (WAL; **last 2000 rows**, pruned on every insert ≈ 13 h live); the same
DB holds `users, sessions, password_resets, watchlists, api_keys`.
`shared/schema.ts`/`server/storage.ts` define no schema (`NoopStorage`). The
nightly local backup copies `DATA_DIR/*.json|pkl|db` and so **misses
`/data/voltrade.db`**; only `backup_to_github.py` (branch `data-backup`,
needs `GITHUB_BACKUP_TOKEN`) copies it. (Appendix F §3 — full table with
writer, reader, shape, retention, atomicity.)

| state | source of truth / path | shape | cadence / retention |
|---|---|---|---|
| live positions | **Alpaca `/v2/positions`** (no local mirror of record); Node `monitoredPositions` (memory, rebuilt every 60 s) | Alpaca Position; monitor fields §S8 | — |
| equity curve | `voltrade_equity_curve.json` (newest-first `[{date (UTC), value: portfolio_value, pnl: equity − last_equity}]`) | written **only** by Tier 3 `recordDailyEquity` (hourly, needs `state.active`) → no rows while killed/stopped in-process; `shift()` after `unshift` at 365 rows drops the **newest** → freezes ~May 2027 (V26) | 365 |
| equity peak | `voltrade_equity_peak.json {equityPeak, savedAt}` (Node); `voltrade_portfolio_dd.json` (Python); `voltrade_peak_equity.json` (tiers/rks) | three disagreeing peaks | overwrite |
| trade history | Alpaca orders (`/api/bot/history` = last 50 closed incl. canceled); `voltrade_trade_feedback.json` (entries mutated in place on exit; `orphan_exit`; `_seed` rows) — the only durable "closed trade" record; in-memory `tradeResults[]` (≤200, `pnl` always 0) | §S7 | 5000 records |
| fills | **no fills file** (`voltrade_fills.json` has no writer); Alpaca `/v2/account/activities/FILL` (used only by auto-resume via `realizedPnlFromFills`) | — | — |
| audit log | SQLite `audit_log`; SSE live; `/api/bot/audit` (100) | `{time, type, message}` ≤500 chars | 2000 rows |
| kill / liveness / dd / kill-switch / peaks / floors / stops / options / cooldown / shadow / blend / diagnostics / timings / universe / csp caches | see §S8–S9 and Appendix F §3.2 | | |
| in-memory only (lost on every deploy) | `openOrders`, `morningQueue`, `monitoredPositions`, `notifications` (≤100), `signals` (≤30, 1 h), `strategyWeights` (write-only), `tradeResults`, `lastScanResult`, circuit-breaker counters, `ownerStopped`, `tier2ConsecutiveFailures`, auto-resume streak | | |

**Failure signature.** `/tmp` fallback (console only); feedback write failure
(swallowed); `DB-SLOW-WRITE`; `/api/diag/equity_curve` gaps; audit history
vanishing within a day on chatty days (boot prose alone ≈35 lines × 25 boots
= 875 of the 2000 rows live).

### S11 — Measurement surfaces (what claims to measure the edge today)

| surface | formula / inputs | defect |
|---|---|---|
| `GET /api/bot/performance` | Path A (Python over non-seed feedback): `winRate = count(pnl>0)/N`, `avgGain`, `avgLoss`, `totalPnlPct = Σ pnl_pct` (not compounded), `profitFactor`, `byStrategy` (always "unknown"), `realisticPnlPct = total − avg_slip×2×N` (**double-counts** unsigned, partly synthetic slippage), `bestTrade/worstTrade/recentTrades`; Path B (fallback when A throws or N = 0): **today-only** Alpaca orders, open positions counted as trades, `totalPnlPct` = daily account change, `avgSlippagePct 0`; + `equityCurve, equityPeak, currentDrawdown` | Path A throws `TypeError` on every `pnl_pct: None` record (312 live) → **the owner sees Path B under lifetime labels, with no flag** (V25) |
| `/api/monitoring/tier-pnl` | buckets `csp_options|etf|stocks|vrp|other` by `r.get('strategy','other')` | all live trades in `other` |
| `/api/monitoring/overview` | daemon health, `risk_kill_switch.get_kill_switch_status` (`consecutive_losses`/`daily_pnl_pct` from files nothing writes → 0), Node drawdown −10 | |
| `/api/monitoring/drawdown` | Python peak, hardcoded −20, bands KILLED/CRITICAL/WARNING/NOTICE | disagrees with `/overview` |
| `/api/diag/ml.live_performance` | `check_model_health`: `{total_trades, win_rate, avg_win, avg_loss, recent_win_rate_20}` over non-seed, non-null-pnl records | **the most complete live edge read that exists** — but never combined into payoff/breakeven/expectancy |
| `adjustStrategyWeights` / `LEARN` | `tradeResults[].pnl` is only ever assigned 0 → win rate 0% → `LEARN "Win rate 0%…"` every 30 min; `strategyWeights` read by nothing [grep] | false signal (KNOWN BROKEN #38 blamed `DEPLOY_TIMESTAMP`; the cause is `pnl` never set) |
| `diagnostics.run_diagnostics` | auto-fix on **win rate only**: >20 trades & WR < 40 → `min_score 75`; degradation → 0.5× size; blend weights | live: threshold 75, blend ≈0.72/0.28 |
| shadow `get_shadow_stats` / `shadow_band_report` | `by_decision, by_regime, labeled_by_horizon, win_rate_by_decision, backfill_progress_by_decision, score_quintile, decision_x_band…` per horizon `{n, win_rate, mean_return_pct, median_return_pct}` | the only `mean_return_pct` (expectancy proxy) and base rate in the codebase; labeler exits ≠ live exits; `taken` ≠ bought; repeated tickers overstate independence |
| `backtest_v2` (`POST /api/bot/backtest`, nightly `backtest.py SPY all 2` → audit `RESEARCH … PASSING|MARGINAL|FAILING`) | single ticker, momentum/mean_reversion only; entry at open[i+1]; ATR 1.5× stop clamped 2–6%, 2R target, regime time stop; cost = real-volume tiers; metrics `total_return_pct, cagr_pct, sharpe, max_drawdown_pct, n_trades, win_rate_pct` + `spy_cagr, alpha` | models none of the scan funnel, floors, tiers, options; persisted only as audit lines; never compared to live |
| `backtest_10yr_results.json` | 2016–2026, configs OLD/NEW/NEW_CSP_ONLY: CAGR 0.27 / −0.27 / 5.68 vs SPY 13.97 | from an engine not in the repo; loses to SPY in every config; seeds the Kelly priors and 1326 `_seed` feedback rows — a known-**bad** bar, not a known-good baseline |
| `realizedPnlFromFills` (`server/killSwitchAutoResume.ts`) | average-cost realized $ P&L from Alpaca FILL activities, options ×100 aware | **the closest existing truthful ledger**; used only by auto-resume |
| `reconstructPortfolioPnl` (`/api/diag/reconstruct_pnl`) | per-leg close-to-close attribution of current holdings for one day, equities only | |

**Not computed anywhere:** expectancy per trade, payoff ratio, breakeven win
rate, live return vs SPY (no SPY series is persisted), live-vs-backtest
divergence (the HONESTY METRIC), per-rule counterfactual P&L with a `rule`
field, a frozen known-good baseline of any kind.

### S12 — The current dashboard and the API it could read

**Route and auth.** `/app#/bot`, tab `"bot"` labelled "AI Engine", shown only
when `GET /api/auth/me` returns `isOwner` (`user.email === OWNER_EMAIL`,
`server/auth.ts:96-120`); `server/bot.ts` routes are `requireOwner`.
**Layout:** rendered inside `<main className="page-container tab-page">`
whose `.page-container` is `max-width: 920px; margin: 0 auto` [grep] — a
centred column (~824 px effective at 1440) of ~13 stacked cards, one long
vertical scroll, all inline styles, three off-token palettes. The full-bleed
precedent to copy is `.vt-map-page` (`position: fixed; top: 56px; left/right/
bottom: 0`; mobile `top 48px; bottom 64px`).

**Panels, in order, and their feeds** (Appendix G §1.4 for loading/empty/
error behaviour of each): Header (`/api/bot/status` 5 s; badge reads PAUSED
while loading) · NotificationBell (`/api/bot/notifications` 15 s) ·
CalendarBanner (`/api/bot/calendar` once; UTC "tomorrow") · Account Overview
(`/api/bot/account` 10 s; **this GET can trip the kill switch**; shows `$0.00`
on error) · PERFORMANCE (`/api/bot/performance` 60 s; equity canvas plots
`equityCurve[0]` on the left while the array is newest-first → **time axis
reversed**) · ML MODEL (`/api/ml/status` 30 s; toggle wired to nothing) ·
Open Positions (`/api/bot/positions` 15 s + nonexistent `/api/bot/clock`
60 s → "AFTER HOURS" and "AH" badges **always** shown; `change_today`/
`lastday_price` not returned → "Today:" always +0.00%; options Cost/Value
without ×100) · TODAY'S TRADES + OPEN ORDERS (`/api/trades/today`,
`/api/orders/open`, raw `setInterval` 30 s, any failure → "No trades today";
option P&L without ×100) · TRADE HISTORY (`/api/trades/history` 60 s; CSP
round trip shows as a "standalone sell" with pnl 0) · Position Rules
(`/api/bot/last-scan.size_tier` only) · AI Trade Signals (`/api/bot/signals`
30 s; "Confidence" = capped composite score, not a probability; stale during
scanner outages) · LIVE POSITIONS charts (`/api/bot/bars/:ticker`; re-fetches
on every re-render) · Security Controls (literal 8% / 95% / **−5%** while −3%
is enforced; Kill toggle is a `div`) · Activity Log (`/api/bot/audit` 15 s).

**Controls (one click, no confirmation, failures silent):** Start, Pause
(in-memory; undone by deploy), Kill toggle (cancels ALL open orders + email;
Unkill does not re-activate), Close (market liquidation), Close All Legs
(N concurrent closes, no combo), Reset Circuit Breaker, Enable/Disable ML
(cosmetic; **any logged-in user**), Snapshot download. Acting endpoints the
page never calls: `POST /api/bot/trade`, `/run-now`, `/refresh-signals`,
`/backtest` (**body interpolated into a shell string**, owner-only), `/backup`,
`/api/ml/retrain` (any logged-in user).

**SSE `/api/bot/stream`:** exists, owner-gated, carries only audit entries,
no heartbeat/id/retry, possibly buffered behind global `compression()`
without `res.flush()` (V31); **no client subscribes** [grep].

**Endpoint catalog:** 28 bot-side routes with exact response shapes are in
Appendix G §3. Twenty-two exist but are unused by the page (`/api/bot/stream,
/history, /export-trades, /backup, /api/system/health-check, /verify-change,
/api/bot/trade, /api/daemon/health, /api/monitoring/overview|tier-pnl|
drawdown|caches, /api/bot/market-status, /refresh-signals, /backtest, /run-now,
/diagnostics, /ml-status, /api/bot/system-status, /api/ml/retrain,
/api/positions, /api/shorts/dashboard`). Not exposed at all: benchmark vs
SPY, live-vs-backtest divergence, counterfactual rule P&L, current regime
(only inside the ~20 s `/api/system/snapshot`), scan failure detail for the
owner (token-gated `/api/diag/scanner` only), WebSocket state, bot-tracked
resting orders, kill reason/timestamp as a field, `asOf` timestamps on any
payload, one consistent drawdown, the actually-enforced limits, options-aware
per-trade realized P&L.

**Mechanical gates a replacement trips** (Appendix G §5): `scripts/visual_check.mjs`
has no bot page and a global unauthenticated fixture map; exact-pinned CI
counters `hardcoded_palette_hex == 402` (bot.tsx contributes 41), `ts_any ==
1247`, `empty_ts_catch == 493` must be lowered in `ci/counter_baseline.txt`
and the test literals in the same PR; canvases must size via
`surfacePixelRatio()`; every `setInterval` needs a `clearInterval`; theme
tokens only (DESIGN.md); 44 px touch targets at 390; designed empty/error
states; last-known data with a visible timestamp. Reusable: `client/src/render/
frameCore.ts` (the one rAF loop + springs), `perfMetrics.ts`/`perfHud.ts`
(`?perf=1`), `freshness.ts` (`freshnessLevel`, `formatDataAge`,
`FreshnessTracker`), shadcn `alert-dialog`/`skeleton`/`resizable`,
`lightweight-charts` 5.1.

---

## 4. The five-layer decision chain — what is recoverable today

| layer | recoverable today | lost / not recorded |
|---|---|---|
| **1 Trigger** (what woke it) | scan cadence (`TIER2 "Starting scan (interval…)"`), `STREAM-SIGNAL` rows (non-causal), `change_pct_today` inside shadow `features`, quick score only in the in-memory `top_10` | no per-candidate "why woken" (quick-score rank, mover-list membership, options setup trigger); T1 trigger (universe rank) and covered-call trigger (≥100 shares) unrecorded; overnight runs overwrite `lastScanResult` |
| **2 Features** (what the model saw) | 34 ML features for every deep-scored candidate (shadow, ~72 d) and for executed regular-hours entries (`entry_features`); CSP universe factor breakdown for the top 200 (current cache only) | raw enrichment inputs (VRP/IV, sentiment, intel, alt, social, fundamentals, ADX/OBV), the five strategy sub-scores, quick-scan metrics (memory only); ~11 features are constants; no training bands to compare against |
| **3 Reasoning** (rules fired / vetoed, in order) | Node vetoes as typed audit rows (`SKIP, SIZE-CAP, HEAT-CAP, CORR-CAP, HALT-SKIP, MAX-RETRY, DUP-SKIP, OPTIONS-SLOT-FULL, OPTIONS-SKIP, CSP-SKIP, TIER-DEFER, TIER-KILL, KILL-*, DD-HALT, TIER2-LIMIT, MORNING-*`) for ~13 h; shadow buckets `rejected_heat / rejected_other / rejected_masterkill / rejected_capital`; `extreme_movers_today.json` | **most Python vetoes are silent** (slots/NEUTRAL break, cooldown, BEAR/PANIC, regime MIN_SCORE, $50M floor, blocked tickers, Kelly/sizing block, volume share, all options-scanner rejections, T1/T2/T3 gates, kill suppression); the `reasons` list is never persisted; no rule-order trace, no stable rule ids |
| **4 Conviction** (score, probability, label) | final score (shadow, feedback); `TRADE` line "Rules: r ML: m Blend: b" for executed stock trades; current blend weights | rules/ML split for non-executed candidates; `ml_confidence`, `ml_signal`, `model_type` (real vs `fallback_rules`), `win_probability`, `regime_used`; the per-bonus decomposition of the rules score; blend-weight history; the Node threshold in force at decision time; the triple-barrier label being bet on is not attached to anything |
| **5 Action** (size, entry, exits set, what happened) | audit `TRADE/ETF-TRADE/OPTIONS-TRADE/OPTIONS-SCANNER-EXEC/T1/T2/MORNING-TRADE/QUEUE`; Alpaca orders and fills; feedback records (with the caveats of §S7) | link from an action back to its scan/candidate (**no `scan_id`, no `candidate_id`, no `client_order_id`**; join is ticker + time); intended stop/target; sizing scalars; Python-side floor/hedge/third-leg orders bypass the Node audit entirely |

Minimum instrumentation the Build stage will need (none of this exists; all
are ordinary PRs in MUTABLE territory, none touch FROZEN paths): a `scan_id`
and a per-candidate `decision_trace[] = {rule_id, passed, threshold, value}`
emitted from the existing loop; persistence of that trace together with
`reasons`, the rules/ML split and `model_type` (append-only JSONL next to the
shadow log); the shadow `taken` label made final only after Node admission;
a `position_id` minted at submission and carried as Alpaca `client_order_id`
on every order and leg.

---

## 5. Edge and leak attribution — computable today vs missing

**Edge scorecard inputs.** Today: `check_model_health().performance` gives
`total_trades, win_rate, avg_win, avg_loss, recent_win_rate_20` over the
non-seed, non-null-pnl feedback records (36 live). From these: payoff ratio
= avg_win/|avg_loss| = 1.76; breakeven WR = |L|/(W+|L|) = 36.2%; expectancy
= 0.278×9.08 − 0.722×5.16 = −1.20%/trade. Caveats that make this an
**upper bound on trust**: `pnl_pct` is the trigger-time bar P&L not the
realized one; `fill_price` carries the synthetic haircut; 84% of live records
are orphans with no P&L at all; zero counts as a loss here but `flat` in
`track_fill`; `_seed` rows are excluded here but **not** in the Kelly sizer.
No SPY series is persisted; the last hand computations in the logs were
+3.1% vs SPY +2.3% (09-08→10-06, earned while *not* trading) and −3.0% vs
−0.1% (10-06→10-12, the floor basket). With 83.8% of equity in a passive
floor, any "vs SPY" number without sleeve separation measures the floor's
tech tilt.

**Leak waterfall buckets.**

| bucket | computable today? | from | blocking gap |
|---|---|---|---|
| entry slippage (decision → fill), stocks | partially | feedback `expected_price` (daily-bar close) vs Alpaca `filled_avg_price`, joined by ticker + time | no order id on the record; `fill_price` synthetic; `alpaca_fill_price` dropped; reference is a delayed daily-bar close not a quote; morning/ETF records have no real fill |
| entry slippage, options | no (dispatcher CSPs) / partial (scanner path) | Alpaca only | no decision bid/ask/mid; no feedback entry for dispatcher CSPs |
| spread / half-spread at decision | no | — | quote fetched for the 0.5% filter and discarded; options bid/ask not in the contract dict |
| limit-price buffer (paid vs signal) | yes (stocks) | `limit = round2(price×1.001)` reconstructable; Alpaca `limit_price` | — |
| exit slippage (WS / POS-KILL) | back-derivable only | confirmed `exit_price` vs `entryPrice×(1+pnl_pct/100)` | exit decision price/timestamp not stored (only `@ $bar` in audit text); `expected_price == fill_price` in the exit payload |
| **each scale-out leg separately** | **no** | Alpaca order history only | legs never recorded; counters advance at submit; Python phantom legs; no `position_id` |
| exit slippage (options) | no | — | recorded at submit with the mark; no fill confirmation |
| fees / commissions | trivially 0 on paper, not recorded | — | no fee field; $0.65/contract only inside the sizing model |
| synthetic haircut (modelled cost) | no | — | `slippage_applied_pct` dropped; `Math.random()` not reproducible |
| timing: decision → submit | no (durably) | scan `timestamp` (memory) + audit `TRADE` time | price observation time not stored; scans take 55–300 s |
| timing: submit → fill | via Alpaca only | `submitted_at` / `filled_at` | feedback `time_placed` = submit+2 s (or queue time), `time_filled` = Python `now()` |
| regime at entry / exit | partial | `regime_at_entry` in the trade dict (dropped by `track_fill`); shadow `regime_label` (constant) | |
| opportunity cost of vetoes | Python-side `rejected_*` only | shadow outcomes | every Node veto is audit-text only; `taken` ≠ placed |
| orders lost to deploys | partial | `SHUTDOWN` rows + Alpaca canceled orders | |
| Python-side orders (floors, hedges, rolls) | Alpaca only | `lastScanResult.{spy_floor, defensive_floor, third_leg, convexity_overlay, options_management}.actions` (memory, latest scan) | not audited, not persisted |

Fields the ledger needs (to be added through normal PRs): decision
`{signal_price, signal_price_source, signal_price_ts, quote_bid, quote_ask,
quote_ts, spread_pct, intended_stop, intended_target, scan_ts, scan_id,
regime, ml_model_type, win_probability}`; submit `{position_id,
client_order_id, alpaca_order_id, submit_ts, order_type, limit_price,
time_in_force, extended_hours, sleeve}`; fill `{broker_filled_avg_price,
broker_filled_qty, broker_filled_at, synthetic_slip_pct (separate), fees}`;
exit per leg `{leg_no, rule_id, trigger_ts, trigger_price, threshold_price,
r_at_trigger, qty_intended, order_id, filled_qty, filled_avg_price,
filled_at, status}`; options `{bid, ask, mid, premium}` at both ends. The
broker's own FILL activity feed plus `realizedPnlFromFills` is the right seed
for a round-trip ledger; `trade_feedback.json` cannot serve as its basis.

---

## 6. Failure-variable catalogue (current behaviour → surfaced? → what the cockpit must do)

| variable | current behaviour | surfaced today? | cockpit behaviour required |
|---|---|---|---|
| a feed goes stale or dies | fetcher returns a neutral/default value; cache written on failure; diagnostics read mtime as healthy; `dataSourceErrors {}` | **no** (except import errors / 15 s timeouts / analyze.py) | per-feed `last_ok_at` vs `last_attempt_at`, HTTP status, `default_substituted` flag, Alpaca tier, WS throughput; "neutral value" rendered as *absent*, never as data |
| the ML stops predicting | >24 h model → `fallback_rules` blended as ML; import error → silent rules-only, `entry_features None` | **no** | `model_type` histogram per scan; red line at 24 h (not 48 h); "ML influence ALWAYS ON (toggle not wired)" |
| retraining silently stops firing | Tier 3 gated on `state.active`; age-check mutex timeout → console only; status `insufficient_data` audited as *complete*; exec-cap kill tagged "likely OOM" | partly (`TIER3-ML-ERROR` only for some statuses) | retrain ledger (append-only): trigger, status, seconds vs budget, metrics; alarm when age > 25 h |
| the websocket drops | `STREAM-DISCONNECT` + flat 10 s reconnect; half-open socket undetectable; `streamConnected` unexposed | partly | stream state + last-bar age + bars/min as a first-class gauge; "stops starved" alarm when held names get no bars in RTH |
| a value leaves its known band | nothing compares live features to training | **no** | per-feature variance ratio + %-constant vs a train-time baseline sidecar (to be created); today ≈11/34 constant |
| broker connection fails | Node `alpaca()` throws → Tier 1 console-only, Tier 2 `TIER2-FAIL "Alpaca down (n)"` (not counted as a scanner failure); Python account read → **fabricated equity 100,000**; `/api/health.checks.alpaca` = account read only | partly | explicit broker gauge with latency and error rate (none measured today); reject fabricated-equity sizing visibly |
| latency spikes | no Alpaca/RPC/Tier-2 latency metric; only `EVENTLOOP-LAG` (106/13 h live) and `DB-SLOW-WRITE`; scan phase timings exist (`/api/diag/timings`) | partly | timings as a live panel; 35 s deep cap and CSP prefetch `budget_exceeded` as named alarms |
| a panel gets no data | `$0.00`, "PAUSED", "No positions", "No trades today" | **no** (errors look like calm) | every tile carries `asOf`, source and state ∈ {fresh, stale, dead, unknown} (`render/freshness.ts`); last-known value with age, never a zero |
| the market is closed / holiday / half-day | Tier 1 RTH-gated correctly; Tier 2 and liveness are weekday- and holiday-blind (scans on weekends/holidays; dark-hours over-count); `market_calendar.py` 2026 only, used by the options scanner only; EOD short-cover never on half-days; calendar banner computes "tomorrow" in UTC | partly | session clock from `/api/bot/market-status` + `/api/bot/calendar.earlyCloses`; expected cadence per session type; "idle because closed" vs "idle because broken" distinguished |
| conflicting or missing state | three peaks, four thresholds; Node vs Python stop-state double writer; `/tmp` fallback; corrupt kill file boots active; corrupt liveness file resets the dark clock; deploy wipes memory | **no** | show all peaks side by side with their owner; flag `/tmp` fallback; flag post-deploy "memory-only state reset" explicitly |
| loop alive but not trading | DD-HALT ("Scanned 0 stocks") for 27 days read healthy; slots full; Kelly block; sector veto | **no** (liveness sees only `active && !killSwitch`) | "silent absence" alarms: trading days since last entry; scans with candidates but 0 orders; orphan rate; labels per night; churn (notional / net change); cancels vs `SWEEP` |
| kill latched | no exits of any kind; auto-resume status in `/api/bot/status.autoResume` | yes (health degraded, email if `RESEND_KEY`) | top-of-screen state with latch reason, time latched, `autoResume` numbers and the next evaluation |
| the deploy cancelled my orders | 24× per day live | yes (`SHUTDOWN` rows) | deploy markers on the timeline; "orders cancelled by deploy" count |
| CI/deploy outage | the 2026-09-10 → 09-16 503 outage; now gated by `healthGate.ts` (server + database only) | yes (`/api/health.serving`) | version + boot time + `serving.failing` on screen |

---

## 7. Incident history — how long each went undetected and how it was found

From `research/open_questions.md` KNOWN BROKEN and `research/experiments.md`
(Appendix H §3.6 has citations per row). The long-dark ones are all **silent
absences**, not errors; two were found by the human asking why the bot seemed
flat.

| # | incident | dark for | found by |
|---|---|---|---|
| F1 | `track_fill` qty guard dropped every regular-hours entry fill | ~2.5 months (04-23 → 07-06) | session reading `/api/diag/ml` |
| F2 | exit-fill machinery had no caller → zero live outcome labels | since v1.0.34 → 07-06 | same trace |
| F3 | ETF path never called `track_fill`; 70/70 exits orphaned | ~3 weeks | `/api/diag/ml` |
| F4 | `code_version` hardcoded "1.0.34" → attribution never worked | → 07-11 | cross-check |
| F5 | shadow backfill **never ran** (unreachable 22Z branch) | ~3.5 months (12,048 unlabeled) | `/api/diag/shadow` |
| F7 | WS exit monitor silently dead | 22 h+ | no `STREAM` lines |
| F8 | kill latch memory-only: deploy booted ACTIVE, $21.6 k of buys | ~50 min | live incident |
| F9 | `equityPeak` memory-only: every deploy re-based DD protection | since inception → 07-03 | audit |
| F10 | CSP channel zero fills (sizing collapsed to 1% floor) | 17+ days | `T2-FAIL` audits |
| F12 | `/overview` drawdown always "OK" (dead field) | 04-20 → 08-09 | code read |
| F13 | `/api/health.drawdownPct` hardcoded 0 — "every session log for weeks quoted 0.0" | weeks → 09-09 | code read during incident |
| F14 | −10% DD kill structurally unreachable; never fired at −17.9% | since inception → 09-09 | health showed −17.9% with zero `DRAWDOWN-KILL` |
| F15 | bad Alpaca equity read → kill latched; loop dark | **19 days** (09-10 → 09-29) | known; human-directed auto-resume |
| F16 | Python DD-HALT "Scanned 0 stocks" while liveness read healthy; sessions logged it "by design" | **~27 days** (09-09 → 10-06) | **the human** asked why the bot was flat |
| F17 | health-gate 503: every new container refused; site down; misdiagnosed for days | ~5.5 days (09-10 → 09-16) | human-directed session |
| F18 | OOM crash loop (port-dwell fold) | ~36 h | `/api/health` uptime resets |
| F19 | bogus −11.8% daily-loss halt fired 36+ times pre-market | hours | routine audit read |
| F21 | SWEEP 204-parse made the CSP submit/cancel loop invisible; QQQ/GLD churn ~$300 k/day | ≥1 day (10-12) | **the human** ("whats going on") |
| F22 | shadow buckets masterkill/heat/other: zero labels | weeks | `/api/diag/shadow` |
| F24 | (new) `/api/bot/performance` Path A crash → today-only fallback | since ≥07-06 → still live | this audit |
| F25 | (new) weekly-loss limit dead | since written → still live | this audit |
| F26 | (new) `LEARN "Win rate 0%"` structurally false | since written → still live | this audit |

The cockpit's health strip must therefore be built on **presence of expected
output per stage against a known cadence** (fills recorded per trade placed,
labels per night, entries per week, exits per stop signal, scans per
interval, bars per minute), not on exception counts. That is the design
lesson of this table.

---

## 8. Defect register (verified)

Each row is one claim unit from the eight reports. `verdict` is `CONFIRMED`
(author's in-session code read, or the completed independent verifier),
`PARTIAL` (verifier, with the correction), or `reader-reported` (stated by one
stage reader with `file:line`; independent re-verification was blocked by the
session limit — see the verification file for the resume command). Letters
(a)(b)(c) refer to sub-claims in the verification file. `filed` says whether
research/ already knows about it (KNOWN BROKEN #, experiments.md date) or
NEW.

| id | stage | claim (short) | verdict | filed | cockpit consequence |
|---|---|---|---|---|---|
| V01 | S0 | "trivial" Python path → 10 s SIGKILL for retrain / manage_positions / backfill / backup / backtest | PARTIAL — independent verifier (code lens, confidence 0.86): the 10 s cap and the mutex bypass are real for all eleven cited commands, but most jobs complete under it live (manipulation, shadow backfill, COT proven by audit/archive rows); the 4 am block is dead code anyway; the Tier-3 retrain and the Tier-1 `manage_positions` one-liner are undecidable from live evidence | NEW | retrain budget is not 900 s; show real `training_seconds` vs the cap |
| V02 | S0/S4 | overnight research + 4 am daily reset unreachable from the scheduler | CONFIRMED — in-session code read (`bot.ts:7681` gate vs `:5209` dispatch; `floor(etHour)===4` at `:6199` unreachable; sole caller `:5211`); live BACKUP/RESEARCH/RETRAIN rows = 0 | NEW | "Daily at 4am" labels are false; daily counters reset only on redeploy |
| V03 | S0/S9 | kill OFF doesn't re-activate; Pause lost on deploy; corrupt latch boots active; Tier 3 runs while killed-at-boot; SIGTERM cancels all orders; circuit breaker pauses the DD check; `/api/health` doesn't check the daemon | reader-reported (A); (e) live-confirmed (24× "All open orders cancelled"); (g) `print('ok')` per A | partly (F8/F9) | state after any deploy must be re-derived, never assumed |
| V04 | S0 | no Tier-1 re-entrancy guard; WS reconnect churn; weekend/holiday-blind Tier 2 + liveness; liveness ignores broker-unreadable | reader-reported (A); absence of `tier1Running` [grep] | NEW | expected-cadence model must know the session type |
| V05 | S5/S11 | `DIAGNOSTIC-PAUSE` dead; `state.dailyPnL` written only by the dashboard route; email "sent" on reject; diagnostics raise `min_score` to 75 on win rate alone | reader-reported (A, H); (b) weekly history length 1 [grep]; (d) live `DIAGNOSTIC` ×7 "Low win rate 27.8%" implies threshold 75 | NEW | show the threshold in force and *why* |
| V06 | S0 | daemon table lacks `snapshot_python_state`/`strategy_pnl_stats`; `manage_positions` bypasses the daemon; `ru_maxrss` is peak | reader-reported (A) | NEW | `via` field on every Python call |
| V07 | S2 | tie-break `UnboundLocalError` every scan → `top_10` rank/tiebreak/factor_strength/sector_clash fabricated | CONFIRMED — in-session: `current_positions` read at `be:3081`, first assigned `be:3186`, both inside `_scan_market_inner` (2571–4080), no `global` declaration → UnboundLocalError; `except` at 3157–3160 | NEW | do not render those fields |
| V08 | S5 | tier regime always "NEUTRAL" → T3 never fires | CONFIRMED — in-session: `be:3921` `… if '_regime_ctx' in locals() else "NEUTRAL"`; `_regime_ctx` is assigned only inside `deep_score` (1451, 1458) | NEW | T3 panel must say "structurally disabled" |
| V09 | S5 | sector veto counts OCC symbols → "Unknown"-sector candidates blocked with ≥2 options held | CONFIRMED — in-session: `be:1905–1915` excludes OCC symbols from `stock_tickers` then re-adds them via `pending_stock_tickers`; D reproduced the block with the live book | NEW | `rejected_heat` is partly an artefact |
| V10 | S5 | shadow `taken` = score ≥ 63 only; NEUTRAL → 0 slots before evaluation | reader-reported (D); live counts consistent (8,372 `taken`, 0 `TRADE`) | partly (10-06 entry) | never label `taken` as "traded" |
| V11 | S5 | covered-call candidates dead on arrival; dominate "trade candidates" | slot gate CONFIRMED live (`OPTIONS-SLOT-FULL` ×158, `QUEUE` ×26 VXUS/KWEB); downstream hops (`should_use_options`, morning M3) reader-reported (D) | NEW | "candidates" ≠ actionable |
| V12 | S5 | Node/Python threshold disagreements (options cap 8 vs 6; score 63/67 vs 65→75; unread config keys) | CONFIRMED — in-session: `be:192`=8 vs `bot:361`=6 vs `sc:110`=6; `sc:35` 30_000 vs 45 s; `sc:307` 5.0 vs `bot:350` −3 | partly | display the enforced value with its owner |
| V13 | S6 | tier dispatch drops `size_pct`/metadata; CSP size = 2% budget stretch to 20%, qty ≤ 2 | reader-reported (D, E) | NEW | "T2 leverage" and `min_iv_rank` are display-only |
| V14 | S3 | regime label constant (`vxx_ratio` from intel; Markov on []); `--mode=scan` ignored; no-options tickers dropped silently | (a) CONFIRMED [grep `intel.get("vxx_ratio")`, `_spy_rets=[]`]; (b),(c) reader-reported (B, C, D) | NEW | regime panel must show the *computed* inputs |
| V15 | S4 | CV by ticker not time; `pnl None → loss`; barrier definition; no promotion gate; GBM fallback labelled lightgbm | reader-reported (C) | NEW | accuracy numbers are not comparable to live |
| V16 | S8 | Python advisory only; thirds at 1R / 3R-1.5R-2R; legs unrecorded; phantom Python legs; 1R = 2×ATR | reader-reported (E) | partly | per-leg ledger is new work |
| V17 | S8/S9 | kill latch stops all management; POS-KILL thresholds; TIME-EXIT dead; displayed stop ≠ enforced | (c) CONFIRMED — in-session `bot.ts:6390` reads `pos.created_at || pos.submitted_at` on an Alpaca Position (schema has neither; live probe shows neither); (a),(b),(d) reader-reported (E) | partly (#43) | compute the stop from the live rules, not `/positions.stopPrice` |
| V18 | S6 | `stocks` bucket Kelly-blocked by priors; live records lack `strategy` → root cause candidate for 0 stock entries | CONFIRMED — in-session: `_get_historical_stats` loads the whole feedback file (no `_seed` filter); stocks f* < 0 under the priors (.49/4.19/4.25 → −0.027) and under the seeded stats (44.6% WR); `_infer_strategy` defaults to `stocks` → `calculate_position` blocks; whether this is THE root cause is moot — B1 (NEUTRAL → 0 slots) also suffices | NEW | sizing panel must show `block_reason` |
| V19 | S6 | DD scalar dead (key mismatch); stress crisis clamped; sector heat dead; floor ETFs count toward heat | reader-reported (E) | NEW | scalars panel must mark dead inputs |
| V20 | S8 | convexity overlay ignores `enabled=False`; `buy_to_open` side values | CONFIRMED (code) — `enabled` never read [grep]; `side: "sell_to_close"` at `be:4531/4541`, `"buy_to_open"` at `be:4657/4668`; Alpaca rejection probable (side enum is buy|sell) but not observed live. D's "disabled by config" is wrong | NEW | hedge panel: "runs; orders likely rejected; unaudited" |
| V21 | S8 | floor basket: at target 0 only QQQ sold; SMH/KWEB/VXUS/T3-SPY no exit owner | reader-reported (E) | partly (10-13 ownership fix) | ownership column per holding |
| V22 | S7 | synthetic fills; tier collapse to 0.12–0.25%; phantom filled entries | (a),(b) CONFIRMED — in-session: trade dict `be:3554–3597` has no `volume` key; `bot.ts:5660` `|| 1000000` → else branch 0.12–0.25%; (c) 2 s single poll reader-reported (F) | partly (#35 exits only) | `fill_price` is not the broker fill |
| V23 | S7 | decision price = delayed daily-bar close; exit pnl = trigger bar; options exits at submit | reader-reported (F, E) | partly (#12) | slippage shown today is not slippage |
| V24 | S7 | dispatcher CSPs / floors / manual / legs / rolls record nothing; Python orders unaudited | reader-reported (F); live: 6 short puts held, `live_options_outcome_breakdown {orphan_exit 30}` | partly (#12c) | ledger must be broker-seeded |
| V25 | S11 | `/api/bot/performance` Path A `TypeError` → today-only fallback; export 500 | CONFIRMED — in-session: `bot.ts:1689` filter keeps `None`, `:1691` `None > 0` raises, catch `:1753` → `if (perf.totalTrades === 0)` fallback `:1758`; live `live_pnl_null_count 312` | NEW | never reuse this endpoint as-is |
| V26 | S10 | equity curve gaps while killed; UTC key; 365-row freeze; chart reversed | CONFIRMED — in-session: `bot.ts:1034–1035` `unshift` then `if (length > 365) shift()` removes index 0 (newest); writer gating per A | NEW | persist from a broker-backed series |
| V27 | S7/S11 | T3 price from trading host; `replaceIfBetter` misreads 204; `LEARN` 0% structural; local backup misses `voltrade.db` | (c) CONFIRMED — in-session: `pnl: 0` at `bot.ts:873` is the only assignment; `strategyWeights` read nowhere [grep]; (a),(b),(d) reader-reported (F, H) | partly (#38 misdiagnosed) | drop the LEARN line |
| V28 | S7 | ETF entries priced/recorded with the underlying's price | reader-reported (F) | NEW | |
| V29 | S12 | options ×100 omitted; status literals 8/95/5; `GET /api/bot/account` trips the kill | CONFIRMED — in-session: `bot.tsx:1039–1040` `qty*entry`/`qty*current`; `routes.ts:7070` `(price − matchBuy.price) × qty`; `bot.ts:2328–2330` literals 8/95/5; (c) account-route kill per `bot.ts:1235–1250` read in-session | NEW | reads must be pure |
| V30 | S12 | session-only auth on bot-data routes; unauthenticated ML/system/market-status; backtest shell interpolation | backtest interpolation CONFIRMED — in-session: `bot.ts:3681–3697` builds `python3 "…" "${ticker}" "${strategy}" "${years}"` and passes it to a shell `exec` (owner-only); per-route auth table reader-reported (G) | NEW | security — see §10 |
| V31 | S12 | SSE behind compression w/o flush; TradeChart re-fetch storm; stale signals during outages | reader-reported (G) | NEW | |
| V32 | S11 | tier-pnl all `other`; `realisticPnlPct` double-count; unwritten kill-switch state files; seeded priors | (c) CONFIRMED — `risk_kill_switch.py:738–760` only reads the two files; no writer repo-wide [grep]; (d) CONFIRMED (no `_seed` filter, see V18); (a),(b) reader-reported (H) | NEW | |
| V33 | S1 | feed failures swallowed; health by cache mtime; EDGAR filter key; USPTO dead | reader-reported (B) with sandbox probes for (c) EFTS schema and (d) USPTO 301 | NEW | feed strip needs new instrumentation |
| V34 | S1 | directional defaults (Reddit → bullish; VIX 20 → ×0.80; sectors 0 → risk_off −5; VXX fail → 0 slots) | reader-reported (B) | NEW | "default substituted" flag |
| V35 | S1/S2 | Alpaca tier invisible (`feed_status` uncalled); universe drops `*W` | reader-reported (B, D); `feed_status()` zero callers per B | NEW | show the tape and its delay |
| V36 | S4 | toggle inert; meta/options models dead; six staleness thresholds; mis-audited retrain statuses | (a) CONFIRMED [grep]; rest reader-reported (C) | partly | truthful ML status panel |
| V37 | live | −10.04% vs the −10% kill: latches at the next RTH Tier-1 cycle or dashboard load | CONFIRMED — arithmetic −10.0355%, kill line $99,654.34; Tier-1 RTH gate and account-route trip site read in-session; `/api/health` computes but never kills (`bot.ts:1513`). **Confirmed live:** `DRAWDOWN-KILL` 2026-10-15T13:30Z, equity $99,008 (−10.6%); auto-resume `UNRECONCILED` ×2 by 16:14Z | — | predicted state change on 2026-10-15 |

---

## 9. Documentation / comment vs code — consolidated disagreements (code wins)

The appendices list ~60 individually. The ones a designer or builder would
otherwise trust:

- CLAUDE.md CODEBASE MAP: "Tier 1 stop monitoring ~30s" → 45 s hardcoded;
  "purged walk-forward CV" → not temporal; "trade_feedback training with
  code_version gating (0.4×)" → moot after boot cleanup; "SPINOUT-READY — the
  bot consumes datacore the way a customer would" → no consumption exists;
  "convexity overlay (QQQ puts)" listed as a mechanism and `system_config`
  says disabled → it runs every scan.
- `/api/bot/status` `maxPositionPct 8 / maxExposurePct 95 / dailyLossLimitPct 5`
  → enforced 0.20 / 0.30 / −3. `/api/bot/ml-status.retrain_schedule "Daily at
  4am ET"` → hourly age check, 4 am unreachable. `mode: "paper"` → literal.
- `server/orderParams.ts:47` "Limit at ask + 0.1%" → input is the daily-bar
  close. Morning-queue "Market during regular hours" → DAY limit. Fill payload
  `order_type: "market"` → limit (field dropped anyway).
- `bot.ts:4778` "SELL_CSP via placeOptionsOrder" → `placeOptionsOrder` has no
  callers. `bot.ts:2593` "oldest-day pruned past 365" → newest. `bot.ts:6584`
  "nothing prunes stop state" → Python prunes it.
- `ml_model_v2` docstring "maps directly to actual bot behaviour" (labels) →
  false; `cv_method "walk_forward_5fold_purged"` → overstated; comment "EWMA
  realized vol" → plain std; "most recent contiguous 10K rows" → last tickers'
  blocks + feedback + shadow. `ml_retrain_safe.py` docstring "8 GB container,
  timeout raised" → effective cap 90 s/10 s. `ml_status.py` note "4am daily
  and hourly" → hourly only.
- `position_sizing` stocks-prior comment "small starter size" → blocked;
  `_regime_scalar` docstring bands → differ; `_portfolio_heat_scalar` "QQQ
  excluded" → the other four floor ETFs count; DD-sizing comment → dead.
- `shadow_portfolio.py:83` "matches bot's take-profit" → +2/−4/5d vs ATR
  phases/thirds/7d. Defensive-floor docstring 0%/10% → config 8%/15%.
  Third-leg docstring 15% → `LEG3_VRP_PCT 0.40`. `options_manager` Check 5
  "100% for bought" → 50%.
- `alpaca_feed.py` "scan results carry data_feed" → `feed_status()` uncalled;
  `options_feed()` docstring "still hardcode opra" → stale; `macro_data` "Polygon
  news" → Alpaca; `diagnostics.py:390` labels the macro cache "polygon".
- `system_config` "STOP LOSS & TAKE PROFIT" block (6%/12%/10 d) and every
  regime override, `TIME_STOP_DAYS`, `MIN/MAX_STOP_PCT`, `CIRCUIT_BREAKER_*`,
  `STOP_COOLDOWN_SECONDS`, `MIN_PRICE`, `MIN_VOLUME`, `OPEN_HOUR_MIN_VOLUME`,
  `MAX_SPREAD_PCT`, `DEEP_SCORE_TOP_N`, `ML_LOOKBACK_DAYS 60`, `KELLY_DIVISOR`,
  `TIER1_INTERVAL_MS 30000` → not read by the live path.
- research/: KNOWN BROKEN #42/#43 still read as open (resolved in practice);
  wishlist "🔴 ACTIVE LIVE CONCERN" (09-09) stale; KNOWN BROKEN #38's diagnosis
  of the `LEARN 0%` line (deploy timestamp) is wrong — `pnl` is never set;
  `research/position_audit_2026-07-18.md` audits power-plant coordinates, not
  positions.

---

## 10. Open decisions for Mike (raise, don't guess)

From the directive §6, plus what the audit surfaced:

1. **Read-only, or can it act?** Today every control is one un-confirmed
   click; Kill cancels every open order (exit orders included); Close All
   Legs fires N independent market closes; Pause is undone by the next deploy;
   `POST /api/ml/toggle` and `/api/ml/retrain` accept **any logged-in user**;
   `GET /api/bot/account` is a read that can trip the kill switch. If the
   cockpit acts, it needs: idempotent set-not-toggle endpoints, confirmations
   with the consequence spelled out, persisted owner intent, and the account
   read made pure. Recommendation: ship read-only first; add actions behind
   confirm-with-consequence; never wire voice to anything that cancels orders.
2. **Off-screen alerts.** The only outbound channel is email via Resend
   (`sendEmailAlert`), logged "sent" even on rejection, used for drawdown-kill,
   liveness-dark (≤1/24 h) and auto-resume. Nothing fires for "alive but not
   trading", stale feeds, ML on fallback, or exit-engine starvation. Decide
   the channel (push/SMS/email) and the alarm family (§6 "silent absence").
3. **Paper vs live vs backtest on screen.** `mode` must be derived
   (`allPaperAlpacaUrls` over Node's hardcoded base, `ALPACA_BASE_URL`, and
   `ALPACA_TRADING_URL`), never the literal; the three can diverge today.
4. **Data freshness tier.** The bot's own cadence is 1-min IEX bars for exits,
   45 s Tier 1, 1–7 min Tier 2 scans, 15-min-delayed snapshots for decisions.
   Tick-by-tick is not what the bot sees; a 1 s cockpit would be showing more
   than the bot knows. Recommendation: SSE for audit events (fix flush +
   heartbeat + ids), 1-min bars for the chart, per-panel `asOf`.
5. **Baseline source for drift.** No training-feature baseline exists; the
   only train-time statistics are the scaler mean/var in the pkl. Options:
   (a) train-run sidecar `voltrade_ml_feature_baseline.json` (quantiles,
   %-constant, per regime) — honest, needs a small ML PR; (b) hand-set bands
   — fast, but a hand-set band on a constant feature hides the constant.
   Recommendation: (a), plus the "%-constant" gauge that catches today's 11.
6. **Persistence of pinned / rearranged views.** No precedent on the bot
   page; the /data map persists layer state in URL hash + localStorage.
   Recommendation: same pattern, per-device.
7. **(new) Which drawdown is "the" drawdown.** Four thresholds, three peaks;
   auto-resume re-bases one of them. The cockpit must either show all four
   with owners or the human picks one canonical definition (an amendment to
   risk rules — human-approved only).
8. **(new) Sleeve separation for any "vs SPY".** 83.8% of equity is a passive
   floor. Without a sleeve tag the headline number measures the floor.
9. **(new) Security items found in passing** (not the audit's scope, raised
   because they affect what a cockpit may expose): session-only auth on
   `/api/trades/*`, `/api/orders/open`, `/api/positions`, `/api/ml/toggle`,
   `/api/ml/retrain`; no auth on `/api/ml/status`, `/api/bot/system-status`;
   `POST /api/bot/backtest` interpolates the request body into a shell string
   (owner-only). Fixes are small and in MUTABLE territory; decide whether they
   ride the cockpit PRs or ship first.
10. **(new) The −10% latch is about to fire.** Decide before the open whether
    that is the intended behaviour at −10.04% on a floor-dominated paper book,
    or whether the auto-resume/re-baseline should run first.

---

## 11. Canonical vocabulary (use verbatim)

- **Stages:** `S0 orchestration`, `S1 feeds`, `S2 candidates`, `S3 features`,
  `S4 ml`, `S5 rules`, `S6 sizing`, `S7 execution`, `S8 exits`, `S9 risk`,
  `S10 state`, `S11 measurement`, `S12 dashboard`.
- **Tiers / loops:** `tier1Reflex` (45 s, RTH), `tier2Intelligence` /
  `scheduleTier2` / `getTier2Interval`, `tier3Strategic` (hourly),
  `runOvernightResearch` (unreachable), `executeMorningQueue`,
  `syncMonitoredPositions` (60 s), `startStreaming` / `checkPositionOnTick`,
  `sweepStaleOrders`, `trackClosedTrades`, `adjustStrategyWeights`,
  `checkDrawdownKillSwitch`, `killSwitchAutoResumeTick`,
  `performKillSwitchAutoResume`, `executeTrades`, `replaceIfBetter`,
  `resolvePendingExitFill`, `recordExitFill`, `recordDailyEquity`.
- **Python entry points:** `scan_market` / `_scan_market_inner`, `deep_score`,
  `check_sector_correlation`, `manage_positions` (advisory), `_get_atr`,
  `update_equity_peak`, `_run_third_leg`, `_run_convexity_overlay`,
  `_resolve_floor_regime`, `_manage_spy_floor`, `_manage_defensive_floor`,
  `calculate_position`, `_kelly_fraction`, `_get_historical_stats`,
  `_infer_strategy`, `ml_score`, `train_model`, `_triple_barrier_label`,
  `_walk_forward_cv`, `_purged_train_test_split`, `_rule_based_fallback`,
  `track_fill`, `fills_slippage_stats`, `options_outcome_breakdown`,
  `check_model_health`, `run_diagnostics`, `get_auto_fix_params`,
  `check_weekly_loss` (dead), `log_candidate`, `update_last_decision`,
  `backfill_outcomes`, `_label_from_path`, `get_shadow_stats`,
  `shadow_band_report`, `select_contract`, `submit_options_order` (frozen),
  `register_options_entry`, `manage_options_positions`,
  `_record_options_exit_feedback`, `get_macro_snapshot`,
  `get_full_intelligence`, `get_alt_data_score`, `get_social_intelligence`,
  `get_options_trades`, `check_kill_switches`, `master_kill_switch`,
  `run_tiers`, `realizedPnlFromFills`, `reconstructPortfolioPnl`,
  `evaluateDrawdown`, `evaluateDailyPnl`, `drawdownStatus`.
- **Shadow decisions:** `taken`, `rejected_score`, `rejected_heat`,
  `rejected_other`, `rejected_masterkill`, `rejected_capital`.
- **Feedback outcomes:** `win`, `loss`, `flat`, `orphan_exit`, `None` (open).
- **Exit reasons:** stock `stop_loss`, `trailing_stop`, `take_profit`,
  `time_stop`, `position_kill`; options `dte_critical`, `assignment_close`,
  `dte_close`, `dte_close_bought`, `profit_target`, `loss_limit`,
  `bought_loss_limit`, `gamma_risk`; groups `strategy_*`, `sabr_edge_exit`.
- **Regimes:** 5-level `PANIC | BEAR | CAUTION | NEUTRAL | BULL`
  (`regime_util.classify_regime_5level`); Markov `regime_label` BULL |
  NEUTRAL_BULL | NEUTRAL | NEUTRAL_BEAR | BEAR (live: constant NEUTRAL_BULL);
  floor regime (debounced); options scanner has its own thresholds; the tier
  engine always sees "NEUTRAL".
- **Endpoints:** see §S12 and Appendix G §3; diag probes `/api/diag/{audit,
  ml, daemon, positions, positions-detail, account, equity_curve,
  reconstruct_pnl, orders, scanner, timings, shadow, shadow_bands}`.
- **State files:** see §S10 and Appendix F §3.2.

---

### Appendices (this directory)

- `research/bot_cockpit_audit/A_orchestration_health.md` — S0, S9 plumbing
- `research/bot_cockpit_audit/B_data_feeds.md` — S1 (30-feed catalog, contract mismatches, fabricated-value table)
- `research/bot_cockpit_audit/C_ml.md` — S3, S4 (per-feature formulas, labels, CV, retrain paths, record shapes)
- `research/bot_cockpit_audit/D_scoring_rules.md` — S2, S5 (the ~60-rule table, score formula, counterfactual coverage)
- `research/bot_cockpit_audit/E_sizing_risk_exits.md` — S6, S8, S9 (sizing formula, exit rule order, halts, floors)
- `research/bot_cockpit_audit/F_execution_state.md` — S7, S10 (order sites, fill recording, state inventory)
- `research/bot_cockpit_audit/G_dashboard_api.md` — S12 (panel table, endpoint catalog, SSE, CI gates)
- `research/bot_cockpit_audit/H_edge_measurement_state.md` — S11, §7 (measurement code, incident catalogue)
- `research/bot_cockpit_audit/live_snapshot_2026-10-15.md` — §2
- `research/bot_cockpit_audit/verification_2026-10-15.md` — §8 evidence
