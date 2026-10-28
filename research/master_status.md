# VolTradeAI master status

Last updated: 2026-10-28 (master-plan session). The daily after-close routine (`voltrade-improve-close`) updates this file from now on.

**Where the bot stands.** The account is down 7.71% since May 6 while SPY is up 5.46% (Sharpe −0.40 vs 0.99; max drawdown −18.6% vs −4.5%). The options sleeve has lost $9,177 realized, and the floor basket is the only sleeve making money. Today's work made the bot safer and more honest; it did not add a proven edge. With the leaks removed, nothing the bot currently scores with shows out-of-sample skill.

## Done (merged to main)

### Phase 0: inventory
- **Audit (#1476):** `docs/audits/trading-engine-audit.md` (the Prompt 1 audit).
- **Brain inventory (#1476):** `research/brain_inventory.md` lists every input, rule and model, each marked live, constant, dead or broken.

### Phase 2: the bot runs correctly
| PR | Fix |
|---|---|
| #1480 | **N3:** an RSI-overbought "SELL" is no longer executed as a BUY. It is dropped and logged as `rejected_bearish_stock_signal`. |
| #1489 | **N2:** the morning queue can no longer open a naked stock short. Options-intended entries are skipped and audited. |
| #1481 | **N15:** a halt (kill switch, circuit breaker or weekly pause) now blocks entries only. Stops, scale-outs, trails, time stops and POS-KILL keep running, and the −10% drawdown check runs under the breaker. |
| #1482 | **N15:** while halted, `options_manager` exits keep running, with rolls turned into closes. |
| #1483 | **N9:** the Python 18% drawdown halt keeps options exits running. |
| #1495 | **N14:** SVXY third-leg size capped at 15% of equity (was 40%). |
| #1492 | **Scheduler:** the daily 4am branch (resets, retrain, backup) now actually runs, once per ET day and restart-safe. The off-site backup push to this **public** repo is disabled. |
| #1499 | **N14/N20:** SVXY gets a vol-spike exit (VXX ratio > 1.25), a −10% stop, a 24h re-entry cooldown, and buy/exit dwell. |
| #1500 | **N18:** a 0% floor target (BEAR/PANIC) now exits every basket member (QQQ, SMH, KWEB, VXUS), unwinding covered calls first. |
| #1501 | **Options archive:** captures after the close (16:30–18:30 ET), labels files with the ET trading date, and runs once per day. |
| #1502 | Regime engine: an unbound `logger` in two error handlers fixed (prerequisite for turning the flags on). |
| #1503, #1504 | Merged by the new improvement routines: N13 (the drawdown sizing scalar now reads `equityPeak`) and N11 (the Python options-slot cap reads `system_config`). |
| #1494 | **Timeouts:** callers' timeouts are honoured (no more 10 s kill), the ML retrain runs in its own lane with a 20-minute budget, and a timeout kills the whole process group so no orphans survive. |

### Phase 3: no silent failures
- **#1490:** every scheduled job writes a heartbeat (start, finish, status). The `/api/diag/heartbeats` probe evaluates each job as ok, late, missing, errored or not due. `/api/health` shows a jobs block that never changes the deploy gate.
- **Independent watchdog:** the `voltrade-watchdog` routine runs outside `bot.ts` and the daemon. It checks hourly from 10:23 to 16:23 ET and again at 21:23 ET on weekdays, and alerts only on failure.
- **Daily pre-open check:** the `voltrade-preopen-health` routine runs at 08:41 ET on weekdays. It checks that scans ran, the retrain finished and exits fired, plus errors and performance vs SPY, and alerts only on failure.

### Phase 6: research and data
- **#1478 `options_entitlements` probe.** Live result:
  - **Alpaca:** gives free real daily open interest (in the contracts list), historical options trades and daily bars with volume. OPRA real-time is refused (403), and Alpaca has no historical quotes.
  - **Polygon:** reference data only; snapshots, trades and quotes are refused (403).
- **#1479:** three published edges pre-registered for shadow tests: dealer-gamma volatility timing, opportunistic insider buying, and an implied-volatility-spread screen for puts to sell. Floor-basket memo: keep it, and test the exposure schedule instead.

### Phase 8: continuous loop
- **New routines:** `voltrade-improve-midday` (12:37 ET) and `voltrade-improve-close` (16:47 ET). Each run watches the bot, fixes bugs through PRs (bug fixes auto-merge on green), ships strategy, sizing and model changes in shadow only, and sends one daily summary.
- **No more empty log PRs:** the new routines never open a PR just to log "[NO-ACTION]".

### Findings
- **Shadow dedup (#1475):** taken candidates are no better than rejected ones; the whole candidate pool loses money on average (−1.45% at +5 days); score quintiles run roughly backwards. So I did not build the BUY-only filter.
- **ML honesty:** with the leaks removed, AUC is about 0.50 for both the current model and the v3 candidate. The current "58% accuracy" comes from the leak.

## In flight (master-plan session; built and tested, being merged one PR at a time — routines: do NOT re-implement these)

Each item has a finished commit and passing tests; they are merged in this order:
2. Regime engine flags ON by config default, env var wins (human directive 2026-10-28); a missing regime state keeps legacy sizing
3. Regime legacy-vs-engine counterfactual log + `regime_counterfactual` probe
4. Regime ML-ready `regime_snapshot` on shadow records (logging only; ML stays gated, FINDINGS F1)
5. Regime docs: FINDINGS "Next steps after close" (F1 ML gate, F2–F7 follow-ups) and RUNBOOK §7
6. Options archive part 2: full in-scope SPY/QQQ/IWM chains (server-side bounds, pagination to the end)
7. Options archive part 3: store gamma, session volume and capture-time spot (schema v2)
8. N6a: ETF substitution sized and priced by the ETF's own quote; 3x leverage blocked
9. N6b: leveraged substitute ETFs exit after 5 days in `checkPositionOnTick`
10. N8: tier size and metadata reach `select_contract`; CSP collateral capped at 8% (was 20% by stretch)
11. Measurement: real broker entry fills, with the synthetic haircut kept as a comparison field
12. Measurement: orphan option exits attributed
13. Measurement: audit log kept 30 days (300k-row cap), plus `?since=`
14. Measurement: dashboard numbers (orders-panel shape, drawdown peak, daily-loss limit shown)
15. ML Phase 4: date-ordered purged CV, earnings look-ahead removed, v3 point-in-time features, exit-aligned label, v3 candidate (shadow only; holdout AUC ≈ 0.50, so not promoted)
16. Phase 7 weekly forecast screen (shadow)
17. Phase 5 source-validation harness + forward selection + ablation (probe `source_validation`)

## Waiting on me (Mike)

1. **Old routines: disable them, plus email access.** I can't edit routines you created yourself; the system only lets me change routines I made.
   - **Overlap:** five old routines duplicate the new ones and each opens a "[NO-ACTION]" PR. Those PRs are the GitHub emails you get. Please turn them off at claude.ai/code/routines:
     - voltrade-daily-am
     - voltrade-daily-midday
     - voltrade-daily-close
     - voltrade-edge
     - voltrade-edge-late
     - optionally voltrade-product-am / -pm / -eve
   - **New routines I made:**
     - voltrade-watchdog
     - voltrade-preopen-health
     - voltrade-improve-midday
     - voltrade-improve-close
   - **Gmail:** I can't attach Gmail to routines I create. Until you add it in the routines page, alerts and the daily summary arrive as push notifications in the Claude app. Once Gmail is attached, they email you.
2. **The repo is public, so off-site backups are off.** The nightly backup used to push every /data state file (fills, account and model files) to the `data-backup` branch of this public repo. It last ran 2026-04-05, and that branch is still public.
   - I disabled the push; local backups continue.
   - Your call: delete the `data-backup` branch, make the repo private, or give me a private backup destination.
3. **About 900 old remote branches.** Most are squash-merged routine branches or leftovers from the history rewrite. I can bulk-delete the merged ones if you say yes.
4. **Data spend (optional).** The live check shows Alpaca gives free real open interest, historical options trades and daily bars. OPRA real-time and Polygon options (snapshot, trades, quotes) are refused. Two options:
   - A price panel that includes delisted stocks (wishlist 2026-10-18, $20–199) unblocks Lazy Prices and the insider-purchase test.
   - Historical chain data (~$2.2k) only if the implied-volatility put-screen test is inconclusive at 52 weeks.
5. **Regime engine (your 2026-10-28 request):**
   - All flags are on by config, because this sandbox has no Railway access.
   - **Optional:** to keep any flag off, set its env var to `off` on Railway, e.g. `VOLTRADE_REGIME_SIZING=off`.
   - The ML follow-up stays closed until the live run passes its promotion rules **and** the Section 10 strategy-level backtest exists. That backtest needs the delisted-inclusive price panel (item 4).
   - Optional decision: the old options-archive history (07-06 to 10-27) is shifted by one day, and the cached results derived from it keep that shift. Should it be relabeled?
6. **Rule calls the agents flagged:**
   - **Kill toggle scope:** the owner kill toggle now keeps protective exits running; only `/api/bot/stop` freezes everything. Confirm that's the meaning you want.
   - **N3 overbought signal:** RSI>70 overbought candidates are now dropped instead of bought. Turning them into an explicit momentum-buy rule would need validation first.
   - **SVXY limits:** the stop is 10% with a 24-hour re-entry cooldown. You ordered "a stop and/or spike exit" without levels, so these are my choices.
   - **N8 put sizing:** each put's cash collateral is now capped at 8% of equity (was up to 20%). But in calm markets a typical put sale may now be larger than the old ~2–4% sizing.
   - **Regime v2:** flipping it live waits on Tier-3 exits being implemented and the SPY/QQQ ownership clash being fixed.
   - **Overnight research window (8pm–4am):** still unreachable. Reviving it would re-enable unvalidated extreme-mover entries, so I left it off.

## Parked in shadow mode (built; they log only and never trade until they pass their test)
| Item | What it would do | Test it must pass | Read date |
|---|---|---|---|
| ML v3 candidate (`ml_model_v3.py`) | Replace the leaky v2 model | Beat random entry under the live exits (+0.36%/trade) and a constant base rate on an honest holdout. **Today: AUC ≈ 0.50, no skill.** | weekly nightly job; probe `ml_v3` |
| Weekly forecast screen | Direction, probability and range calls for SPY, positions and candidates (display only) | Positive Brier skill vs base rate after 8 weekly batches | ~2026-12-25 |
| Source-validation harness and ablation | Tells which data sources and score adjustments help or hurt | Benjamini-Hochberg pass per source; ablation un-inverts the score | from ~2026-11-02 (needs labeled components) |
| Regime v2 (master-plan version) | Superseded by the regime engine; **dropped** | — | — |
| Dealer-gamma volatility timing (P1) | SPY realized vs implied vol, beyond what VIX already says | Pre-registered in `research/edge_preregistration.md` | runnable now |
| Opportunistic insider purchases (P2) | Cohen-Malloy-Pomorski insider-buying signal | t ≥ 2.8 on a universe that includes delisted names | blocked on the price panel |
| IV-spread put-selling screen (P3) | Rank puts to sell by implied-volatility spread | 52-week forward read | 2027-10 |
| BUY-only band filter (3b) | — | Evidence did not hold (n = 25–31, not significant); **not built** | — |
