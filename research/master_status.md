# VolTradeAI master status

Last updated: 2026-10-29 ~06:00 UTC (master-plan session, overnight push). The daily after-close routine (`voltrade-improve-close`) updates this file from now on.

**Where the bot stands.** Since May 6 the account is −7.73% while SPY is +5.78% (gap −13.5 pp, `/api/diag/benchmark`, 2026-10-29). Since 2026-10-28 evening, **new stock entries and every options open are in SHADOW MODE** (human directive): the bot logs what it would buy instead of buying. The floor basket (QQQ/SMH/KWEB/VXUS + GLD) and every exit stay live. The regime engine is live (CALM_BULL, exposure 0.75 on day 1). Nothing the bot scores with has shown out-of-sample skill yet; the new Brain page and the 2016+ walk-forward exist to find out honestly.

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

### Overnight 2026-10-28 → 29 (human: "merge freely tonight")
| PR | What |
|---|---|
| #1502, #1508–#1510, #1514 | **Regime engine live** (flags on by config, env wins), legacy-vs-engine counterfactual log, ML-ready regime snapshot (ML not built), FINDINGS next steps + RUNBOOK §7 |
| #1511, #1512 | **N6:** ETF substitution sized by the ETF's own price; leveraged ETFs force-exit after 5 days |
| #1513 | **N8:** tier sizing reaches CSP execution; collateral capped at 8% |
| #1524 | **Shadow mode:** stock entries + whole options sleeve logged, not traded (`VOLTRADE_SHADOW_ENTRIES=off` on Railway rolls back without a deploy). Found: the QQQ-put convexity overlay was live despite `enabled=False` (no reader) — now shadowed |
| #1538 | Bot page shows the shadow state + today's would-be entries |
| #1517, #1518 | Options archive: full chains, gamma/volume/spot (schema v2) |
| #1519–#1521, #1525 | Measurement: real broker fills, orphan option exits attributed, 30-day audit retention, cockpit numbers |
| #1526 | **Daily bot-vs-SPY scoreboard** on the bot page (`/api/bot/scoreboard`) |
| #1527, #1528 | Source-validation harness + forward selection + ablation (shadow) |
| #1529–#1533 | ML honesty: purged CV, earnings look-ahead removed, v3 PIT features + exit-aligned label, v3 candidate (shadow; AUC ≈ 0.50) |
| #1535 | Weekly forecast screen (shadow) |
| #1515 | **Edge P1 dealer gamma: gate-2 PASS** (t −9.6 full, −7.3 holdout); nothing trades — gate 1 (own GEX from Alpaca OI) open |
| #1516, #1523, #1534, #1537 | Alpaca free bars: history starts **2016-01-04, not 2010**; delisted names covered from 2016 → nightly **2016+ survivorship-reduced price panel** (first build running) |
| #1536 | Branch-cleanup plan + apply script (agent proxy can't delete branches — human runs it) |
| #1539 | Expired login now sends you to sign-in instead of a fake "service unavailable" |
| #1540 | **Brain page** (owner tab): every input/rule/model live/shadow/off, short- vs long-term evidence, best combinations discounted for N tried |
| #1541 | **Ticker dossier v1** (replaces the AI-Engine look-up): market → sector → stock → options → events, calibrated 68%/90% range bands, trade plan (stop/target/reward:risk/P(target first), position-size helper), direction only with a measured record; every served range/plan graded nightly |
| #1542 | **Event-reaction study** (pre-registered, 4,511 issuers 2016–2026): insider clusters, 8-K 1.01/4.02/5.02 — reactions happen on day 0–1; **no "not yet priced in" signal** passes |
| #1543, #1544 | `price_panel.py` (Python reader, TS parity) + **2016+ point-in-time walk-forward** of the score's price inputs vs SPY and random entry (weekly job; feeds the Brain long-term column) |
| #1545 | Dossier events: the unvalidated "not priced in" flag replaced by measured event history (5.02 deliberately not mapped) |
| (other session) #1522 | Edge P1 GEX/DIX shadow archive stream |

### Findings
- **Shadow dedup (#1475):** taken candidates are no better than rejected ones; the whole candidate pool loses money on average (−1.45% at +5 days); score quintiles run roughly backwards. So I did not build the BUY-only filter.
- **ML honesty:** with the leaks removed, AUC is about 0.50 for both the current model and the v3 candidate. The current "58% accuracy" comes from the leak.

## In flight
Nothing queued from the master-plan session (as of 2026-10-29 ~06:10 UTC). Routines: new work starts from the "Waiting on me" and "Parked" lists below.

## Waiting on me (Mike)

1. **Old routines + email.** I can't edit routines you created. Please disable voltrade-daily-am / -midday / -close, voltrade-edge / -edge-late (they open "[NO-ACTION]" PRs = the GitHub emails), optionally voltrade-product-am/-pm/-eve. Attach Gmail to the new routines (watchdog, preopen-health, improve-midday, improve-close, regime reads) if you want email instead of push notifications.
2. **Branch cleanup (approved, needs your hands).** The agent proxy refuses branch deletes/tag pushes. From any clone with push access: `bash scripts/branch_cleanup_apply.sh` (dry run) then `--apply` — tags 86 unique branches `archive/<name>`, deletes 805 merged + `data-backup`; skips main, celestial-catalog-data, open PRs, <24h tips, branches named in code. Also: turn on GitHub "Automatically delete head branches".
3. **Price history before 2016** is not free (Alpaca starts 2016-01-04). EODHD ≈ $20 would extend the panel; no-paid-data rule means it waits for your call.
4. **Shadow mode is your switch.** To re-enable entries: Railway env `VOLTRADE_SHADOW_ENTRIES=off` (or `stocks` / `options` to keep one sleeve shadowed). The routines are told never to flip it.
5. **Rule calls (unchanged from 10-28):** kill-toggle scope, N3 overbought drop, SVXY 10% stop/24h cooldown, N8 8% collateral cap, overnight research window off. Options-archive day-shift relabel (07-06→10-27) still open.
6. **Dossier choice made for you:** 8-K Item 5.02 (officer changes) shows no history line, because the study's −1.16% applies only to text-classified sudden CEO/CFO exits. Say if you'd rather see it with a caveat.

## Parked in shadow mode (built; log only; never trade until they pass)
| Item | What it would do | Test it must pass | Read |
|---|---|---|---|
| Stock entries + options sleeve (shadow mode) | Everything the bot would have bought | `shadow_mode_entry` outcomes vs the pre-switch taken bucket and vs SPY | +20 trading days (~2026-11-27); human decides |
| Ticker dossier ranges + trade plans | Calibrated ranges and stop/target plans | 68% band contains 60–80%, 90% band ≥ 83%, plan outcomes ≈ simulated odds (n ≥ 100) | nightly ledger, on the dossier |
| 2016+ walk-forward of score inputs | Long-term evidence for the stock score | IC/top-10 vs random entry after Bonferroni/DSR | first weekend after a complete panel |
| ML v3 candidate | Replace the leaky v2 | Beat random entry + base rate on an honest holdout (today AUC ≈ 0.50) | weekly job; probe `ml_v3` |
| Weekly forecast screen | Direction/range calls | Positive Brier skill after 8 weekly batches | ~2026-12-25 |
| Source-validation + ablation | Which sources help/hurt | BH pass per source; ablation un-inverts the score | needs ~100 labelled rows |
| Edge P1 dealer gamma | Hold short premium when gamma is low | Gate 1: own GEX (Alpaca OI) vs SqueezeMetrics; then gate 3 rule sim | after gate 1 |
| Event history (dossier) | Context only — no signal | E4 replication on the untouched 61% of 5.02; E2 1.01+(3.02/2.03) pre-reg; rerun on the delisting-inclusive panel | open questions |
| IV-spread put screen (P3) | Rank puts to sell | 52-week forward read | 2027-10 |
