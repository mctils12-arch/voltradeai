# AUDIT — regime, volatility, breadth and risk-off logic as it exists in the code

Stage A, Part 2 of the Market Regime & Volatility Engine program. Read-only; nothing in the repo was changed to produce this document. Repo HEAD `786c88ce` (`main` after PR #1471), version 1.0.1168, 2026-10-27. Line numbers are from this checkout; `bot_engine.py` and `server/bot.ts` move often, so each citation also names the function.

Method: five parallel read-only audits (Python regime/vol/risk logic; the options stack; the Node orchestrator and data inventory; the universe and the repo's own edge evidence; the git/research history) plus a direct read of every module the engine will touch (`regime_util.py`, `regime_inputs.py`, `markov_regime.py`, `stress_index.py`, `system_config.py`, `macro_data.py`, `backtest_v2.py`, `position_sizing.py`, `tiered_strategy.py`, the floor managers and the scoring blend in `bot_engine.py`, `server/cboeVix.ts`, `server/fredMacro.ts`, `alt_data.py`). The research-versus-code gap table is §11.

---

## 1. Executive summary

1. **There is no market-environment layer.** The system has a *classifier* (`regime_util.classify_regime_5level`, two inputs: VXX ÷ its 30-day mean and SPY ÷ its 50-day MA, plus a 200-day streak) that maps a point reading to one of five labels, with no hysteresis, no dwell, no confirmation, no confidence, and no persisted history. It does not answer "should we be in the market, how much, which strategies, how fast should we change our mind" — it answers "which bucket is VXX in right now".
2. **Two regime vocabularies are live and are conflated.** Vocabulary A (PANIC/BEAR/CAUTION/NEUTRAL/BULL, `regime_util.py:41-128`) gates stock slots, exposure caps, the floors and the tier engine. Vocabulary B (BULL/NEUTRAL_BULL/NEUTRAL/NEUTRAL_BEAR/BEAR from `markov_regime.get_full_regime_score`, `markov_regime.py:206-210`) drives stock exit geometry in Node (`server/bot.ts:7300-7312`), the ML regime features and every shadow-portfolio label — 17,630 of 20,000 shadow records carry `NEUTRAL_BULL` because that path's VXX input was pinned at 1.0 until 2026-10-19 (v1.0.1106). A third set (`risk_on/risk_off/neutral` from VIX < 18 and sector breadth, `macro_data.py:182-190`) adjusts `deep_score` by ±3/−5 (`bot_engine.py:1236-1241`); a fourth (`vix_regime` low/medium/high/extreme, `macro_data.py:109-118`) feeds a sizing scalar whose keys do not match it (`position_sizing.py:586-591`).
3. **Volatility enters as a VXX ETN ratio, not as VIX, and never as term structure.** Spot `^VIX` is fetched by yfinance and silently defaults to 20.0 on failure (`macro_data.py:94-107`); VIX9D/VIX/VIX3M/VIX6M/VVIX have been archived daily since 2022-05-16 by `server/cboeVix.ts` and are read by nothing on the trading path (its own header says so). Realized-versus-implied exists only per single name (vol_surface VRP) and as an ML feature. No VIX percentile, no spike-speed input, no universe realized vol.
4. **Breadth has never been built.** Three proxies stand in for it: a linear function of `spy_vs_ma50` labelled "breadth" (`bot_engine.py:1688-1712`, whose preferred input key `market_breadth_ma50` has no producer), `stress_index._breadth_stress` which is SPY versus its own MAs (`stress_index.py:109-126`), and an advance/decline count among the 100 most-active names. Nothing measures the percentage of the universe above its 50/200-day, new highs minus lows, equal-weight versus cap-weight, or the universe's median drawdown. The index therefore hides the damage underneath it by construction: with SPY near its highs the classifier reads BULL or NEUTRAL whatever the median stock is doing.
5. **The falling-knife path is the mean-reversion sub-score with no guard.** `strategies/mean_reversion.py` awards +40 for RSI < 20, +30 for a 5-day drop beyond −10 %, +20 for 2× volume — "STRONG BUY" at 65 — and enters `deep_score` at 0.20 weight (`bot_engine.py:1261`). No check of price versus its own 50-day, no 20-day-low or base-building rule, no relative strength versus a benchmark (only a +6 bonus for beating SPY by 5 %, `bot_engine.py:1470-1473`), no sector-relative test for names outside a ~50-ticker map, and no regime or breadth condition on any of it. The quick-score funnel that feeds `deep_score` ranks by |today's move| regardless of sign, so the candidates are by construction the day's largest gaps. Honesty note: the repo's own aggregate evidence (shadow bands, 2026-10-06) says the worst cell is *chasing* +10..20 % gainers (35 % win) while buying ≥10 % droppers was the best cell (65 % win) under a +2/−4 labeler; the knife that has verifiably cut this account is the CSP sold into an earnings gap (APH −48 %, 2026-07-16).
6. **The volatility whipsaw on options is structural and mostly unrelated to the regime label.** The manager's 2× stop and 50 % target are evaluated on broker marks from the free indicative feed every 1–7 minutes and close at the ask; there is no re-entry cooldown after a manager close; the non-marketable-limit → 12-minute sweep → re-dispatch loop re-sells into wider quotes; three Node paths (option POS-KILL backstop, TIME-EXIT, LIQUIDATE-ALL) act on the same lots outside the manager's 60-minute hold; the tier engine's vol-scaled size is computed and then dropped before `select_contract` (`server/bot.ts:5251` vs `:5263`; `options_execution.py:486-489`), so no contract count, delta or spread-versus-naked choice changes with volatility. The one live vol rule on the CSP path is the whole-portfolio exposure cap (PANIC 30 % invested blocks every entry) — all-or-nothing.
7. **Exposure is cut by at least seven independent rules reading the same input.** Regime caps (`MAX_POSITIONS`/`MAX_TOTAL_EXPOSURE`), the VIX sizing scalar, the stress-index multiplier (which itself contains VXX, VIX and SPY-vs-MA), the drawdown scalar, the tier `size_scalar`, the floor schedule, and four separately tracked drawdown peaks (Node −10 %, Python 18 %, `risk_kill_switch` −20 %, tier −8 %/−20 %). None has a single owner; several are dead or unreachable (§9). This is the mechanism behind "one changed variable moves five others".
8. **The biggest exposure lever has never been tested, only defended.** The floor schedule (.70/.90/.35/0/0 of equity) moves 70–90 % of the account on the same unhysteresed label; research test 4 (2026-10-20) found that holding SPY at the floor's own regime exposure made 7.41 % CAGR versus 13.58 % buy-and-hold at the same Sharpe over 2014–2026, because the BEAR label (639 days at zero floor) coincided with the strongest rebound days. The 2026-10-13 dwell stopped the label flicker (7 QQQ round trips a day) but left every percentage untouched.
9. **Churn is real and has a signature.** The history audit (A5) found the hedge leg rebuilt six times in fifteen days, the floor churned four times by the same class of defect, NEUTRAL stock trading proposed and reverted three times, thresholds redefined four times, and the convexity overlay "disabled" since 2026-04-17 while its code still runs every scan (`bot_engine.py:4063-4071` never reads `enabled`). Every same-day add/retune/revert came from an in-sample sweep on a survivor universe; every durable finding came from a prior-first, regime-split, random-baseline test.
10. **The inputs fail more often than the thresholds.** `vxx_ratio` sat at 1.0 for long stretches twice (feed bug 2026-07-18; ML path until 2026-10-19), `spy_above_200d` defaulted True, `spy_below_200_days` is still hard-coded to 0 inside the tier engine (`tiered_strategy.py:114-123`, KNOWN BROKEN #20a), the daily-loss kill input was a literal 0 for months, and both drawdown halts latched on one anomalous equity snapshot. **New this audit:** the Markov detector is fed *fractional* daily returns (`regime_inputs.py:60`) against *percent* thresholds (`markov_regime.py:77-84, 100-109`), so every return falls under the 0.3 floor, the state is pinned at Neutral and the Vocabulary-B label collapses to a VXX/SPY score with a constant Markov term — and `test_regime_inputs.py` pins that feed. The stress index's "crisis" multiplier of 0.0 is clamped back to the 1 % minimum position (`position_sizing.py:1050`), so its `halt_new` never halts; the regime `MAX_POSITION_PCT` is never read by sizing (the absolute 8 % clamp is the only per-position cap); sizing's volatility scalar reads `ewma_vol` while the analyzer emits `ewma_rv`, so it likely sits at its default. Any regime engine must prove its inputs live (data age, provenance, fallback flags) before its thresholds mean anything.

**Verdict on the question asked:** the system is **not** properly set up for market regime, volatility or breadth. It has a volatility bucket classifier wired to several uncoordinated throttles, no breadth, no term-structure or credit input on the trading path, no transition logic outside the floor managers, and no persisted regime history. The biggest weaknesses, in order of expected impact: (1) the floor exposure schedule driven by an unhysteresed VXX label; (2) seven stacked exposure cuts with no single owner; (3) no breadth or relative-strength guard on the mean-reversion score and the biggest-mover funnel; (4) vol state never reaching option contract selection; (5) the dead and duplicated regime code that makes every change unpredictable. FINDINGS.md ranks the fixes.

---

## 2. Inventory of existing regime / trend / volatility / breadth / risk-off logic

### 2.1 Label producers

| # | Component | Location | Inputs | Output / thresholds | Hysteresis / dwell | Consumers | Status |
|---|---|---|---|---|---|---|---|
| P1 | `classify_regime_5level` (Vocabulary A) | `regime_util.py:74-127`; delegated to by `system_config.get_market_regime` (`:402-411`) | `vxx_ratio` (VXX close ÷ 30-day mean, `macro_data.py:247-267`), `spy_vs_ma50` (`:281-318`), `spy_below_200_days`, `spy_above_200d` (`:192-239`) | PANIC ≥1.40; BEAR ≥1.15 or SPY/MA50 <0.94 or ≥10 closes below 200-d; CAUTION ≥1.05; BULL ≤0.95 & ≥0.98 & above 200-d; else NEUTRAL | none | `get_adaptive_params`, floors, third leg, convexity overlay, `tiered_strategy.get_regime_caps`, `backtest_v2.regime_series`, `DRAWDOWN_HALT_RESUME_REGIMES`, `lookup_ticker` | LIVE, canonical since 2026-04-20 |
| P2 | `get_market_regime` legacy body | `system_config.py:413-438` | same | 0.90/1.02 BULL tiers, Markov tie-break | none | unreachable (ImportError fallback only) | DEAD decoy |
| P3 | `markov_regime.get_full_regime_score` (Vocabulary B) | `markov_regime.py:258-305`; fed by `regime_inputs.regime_inputs` (`regime_inputs.py:36-85`) | last 5–25 SPY daily returns (order-2 transition matrix hard-coded from 2016–2026), `vxx_ratio`, `spy_vs_ma50` | score 0–100 → BULL ≥75 / NEUTRAL_BULL ≥55 / NEUTRAL ≥40 / NEUTRAL_BEAR ≥25 / BEAR; `size_multiplier` 0.5–1.3 (unused) | none | `deep_score` ML features (`bot_engine.py:1489-1502`), `regime_label` → `regime_at_entry` (`:3797`) → Node exit geometry (`server/bot.ts:7300-7312`), shadow labels, `TierContext.regime` (`:4140`) | LIVE but demoted: 2026-07-09 research found no forward-vol power (ρ −0.07, sign-flipped), "NOT ACTED ON" |
| P4 | `macro_data` `market_regime` | `macro_data.py:182-190` | spot VIX (< 18 / > 25), share of 11 SPDRs green today | risk_on / risk_off / neutral | none | `deep_score` macro adjustment ±3/−5 (`bot_engine.py:1236-1241`) | LIVE, unvalidated |
| P5 | `macro_data` `vix_regime` | `macro_data.py:109-118` | spot VIX | low <15 / medium <20 / high <30 / extreme | none | `position_sizing._regime_scalar` (keys low/elevated/high — mismatch) | LIVE, label mismatch |
| P6 | `stress_index.compute_stress_index` | `stress_index.py:159-221` | FRED T10Y2Y and IG OAS (`alt_data.get_fred_macro`), VXX ratio bands, spot VIX bands, SPY vs MA50/MA200 ("breadth"), SPDR defensive-minus-offensive one-day rotation | weighted 0–100 → calm <30 / elevated <55 / stressed <75 / crisis | none (10-min cache) | `position_sizing` multiplier 1.0/0.75/0.5/0.0 (`position_sizing.py:1028-1047`); system snapshot | LIVE-but-unvalidated (never passed a ladder gate); replaced the T4 tail hedge 2026-04-22 |
| P7 | Floor regime resolver | `bot_engine._resolve_floor_regime` `:5047-5167` | P1 label per macro reading | confirmed floor regime | **yes**: calming/lateral changes need `FLOOR_REGIME_CONFIRM_CYCLES = 3` distinct macro readings; escalation into CAUTION/BEAR/PANIC immediate; fallback VXX holds the confirmed label | `_manage_spy_floor`, `_manage_defensive_floor` | LIVE since 2026-10-13 (v1.0.1055) — the only hysteresis in the system |
| P8 | `backtest_v2.regime_series` | `backtest_v2.py:374-416` | SPY + VXX daily bars | P1 labels per day; quality flag (VXX real only from 2018-01-25) | none | research scripts, `backtest_v2` gating | RESEARCH |
| P9 | QQQ trend override on the floor | `bot_engine._manage_spy_floor` `:5230-5275` | QQQ price, 50-d, 200-d | death cross → cap 0.20; below 200-d → cap 0.50 | none | floor target | LIVE, never ablated |
| P10 | `options_scanner` own VXX ratio / SPY-vs-MA50 | `options_scanner.py:203-262` | its own Alpaca fetches (60-s cache) | duplicates P1's inputs | none | scanner setups (VXX bands 1.05/1.15/1.25/1.30) | LIVE duplicate |
| P11 | `alphadesk/market_context.py` | `alphadesk/alphadesk/market_context.py:21-45` | 11 SPDRs, SPY 200-d | fraction of sector ETFs above the 50-d | none | alphadesk only | NOT WIRED to the bot |
| P12 | `probability_engine` | `probability_engine.py` | SPY vs ACWX/IWM/DBC, XLK vs defensives, IVW vs IVE, 2Y vs 10Y | advisory dimensions | none | system snapshot | advisory only |
| P13 | ML regime routing | `ml_model_v2.py` (regime-conditional models keyed on `classify_regime` 3-level) | P1 inputs | bear/neutral/bull model choice | none | `ml_score` | LIVE; the bull model was unreachable until 2026-10-19 |

### 2.2 Consumers that throttle exposure (the "who cuts on the same input" list)

| # | Throttle | Location | Keyed on | Values | Double-cut / defect |
|---|---|---|---|---|---|
| T1 | Stock slots `MAX_POSITIONS` | `system_config.py:457-512`, size-tier overlay `:545-583`; applied `bot_engine.py:3361-3381`, Node `server/bot.ts:5777` | P1 label | BULL 8→6 (size tier), CAUTION 4→6, NEUTRAL/BEAR/PANIC 0 | Non-monotonic (NEUTRAL 0 < CAUTION 6). Floor ETFs count as stock slots on both sides (`bot_engine.py:3344-3348`, `server/bot.ts:5710-5712`). Node's own absolute 8. |
| T2 | `MAX_TOTAL_EXPOSURE` | `system_config.py:463-512`; enforced `tiered_strategy.master_kill_switch` `:292-302`, position_sizing heat; Node `MAX_TOTAL_EXPOSURE 0.30` of non-floor (`server/bot.ts:363, 5805`) | P1 label | 0.30/0.50/0.60/0.95/0.95 of whole portfolio (Python) vs 0.30 of non-floor (Node) | Different denominators; the master kill judges CSP collateral against whole-portfolio exposure the floor creates (KB#20b). |
| T3 | `MIN_SCORE` | `system_config.py` per regime; `bot_engine.py:3490-3507`; Node `minScoreThreshold` 65→75 (`server/bot.ts:507, 4786`) | P1 label (Python), win-rate diagnostic (Node) | 63/63/67/75/75; 65 or 75 | Two thresholds on the same candidate; the score itself is anti-predictive (2026-10-26). |
| T4 | `_regime_scalar` | `position_sizing.py:575-606`, applied `:929-990` | spot VIX level (P5) | 1.10 → 0.30 | `macro=None` on the CSP path → 1.0; label keys mismatch. |
| T5 | stress-index multiplier | `position_sizing.py:1028-1047` | P6 | 1.0/0.75/0.5/0.0 | Stacks on T4 and T6 ("applied AFTER DD scaling so both stack"); P6 already contains VXX, VIX and SPY-vs-MA. |
| T6 | drawdown scalar | `position_sizing.py:996-1026` | equity vs `voltrade_equity_peak.json` | 1.0/0.8/0.6/0.4 | Dead on a key mismatch (`peak_equity` vs `equityPeak`, cockpit audit 2026-10-15). |
| T7 | Tier-1 `size_scalar` | `tiered_strategy.py:486-500` | VXX ratio, drawdown | 1.0/0.85/0.60/0.30, ×0.75/×0.50 | Computed, written to the dispatch payload, **never passed to `select_contract`** (`server/bot.ts:5251` vs `:5263`); survives only as the affordability price cap. |
| T8 | Floor schedule | `system_config.py:135-139, 195-200`; `bot_engine.py:5170-5700` | P7 (debounced P1) | basket .70/.90/.35/0/0; GLD .08/.08/.15/.30/.40; QQQ caps .20/.50 | 70–90 % of equity; the largest lever; untested. |
| T9 | Regime cooldown after a stop | `bot_engine.py:3463-3477` | P1 | 1.5–4 h | Never tested. |
| T10 | `regime_kill_for_tier` | `tiered_strategy.py:307-312` | VXX ratio ≥ 1.40 | T2/T3/T4 off, T1 always on | — |
| T11 | Drawdown halts | Node −10 % (`server/bot.ts:513-514, 4252-4315`); Python 18 % (`bot_engine.py:584-719`, `system_config.py:308-320`); `risk_kill_switch` −20 % (`:51`, FROZEN); tier −20 %/−8 % (`tiered_strategy.py:90-91`) | four separately tracked peaks | — | Mechanisms repaired seven times; thresholds never tuned; one bad snapshot latched two of them for 26–46 days. |
| T12 | Daily loss | Node −3 % (`server/bot.ts:5047-5059`, display says 5 %), `risk_kill_switch.DAILY_LOSS_LIMIT −0.03`, `tiered_strategy −0.03` | same input | triple check | Attribution split across three audit vocabularies. |
| T13 | Regime-aware stock exits | `server/bot.ts:7300-7312` | **Vocabulary B** per position | 2nd scale-out at 3R/1.5R/2R, trail 3×/1.5×/2× ATR | Found the worst of four exit models (2026-10-20). |
| T14 | Dead regime keys | `STOP_SCALAR_*` (`system_config.py:374-381`), `TIME_STOP_DAYS` overrides, the configured stop/TP block (`:264-274`) | — | — | No reader (KNOWN BROKEN #48). |
| T15 | Convexity overlay budget | `bot_engine.py:4695-4719` | P1 (PANIC/BEAR 4 %, CAUTION 3 %, else 2 %) + DD ≥ 10 % | buys QQQ puts | Runs although `CONVEXITY_OVERLAY.enabled` is False (`system_config.py:242`; flag never read at `bot_engine.py:4063-4071, 4695`). |

### 2.3 Regime use on the options path (summary of A2)

- Tier-1 CSP: regime enters only as T2 (exposure cap) and T7's affordability cap. `T1_MIN_IV_RANK = 40` is metadata with no consumer (`tiered_strategy.py:181, 584`); the only `ivr > 40` test in the stack is the bull-put-spread *upgrade* on the stock→options path (`instrument_selector.py:853-859`). "IV rank" on the universe ranker is 0.7 × (ATM IV ÷ 30-day HV rank) + 0.3 × HV percentile (`options_scanner.py:575-631`) — mostly a VRP proxy, not an IV history percentile.
- Scanner setups switch on VXX bands (earnings-crush off ≥1.30; high-IV condor off ≥1.25 and on only if IVR ≥ 50 or 1.05–1.25; panic put-spread on ≥1.30 with SPY/MA50 ≥ 0.94); the QQQ condor only in 0.80–1.05; covered call off in BEAR/PANIC; instrument score *rises* with VXX so more stock candidates are routed to options in high vol (`instrument_selector.py:904-918`).
- Manager exits (`options_manager.py`): DTE-critical 5, min hold 60 min, assignment delta 0.80, 21-DTE close, 50 % target (mark), 2× stop (mark, closes at ask), gamma 0.08, VRP-evaporation (grouped only). Nothing in the manager is vol-conditioned.
- Contract selection: strike nearest −0.30 delta, expiries 30–50 DTE (page-fills at ≈30), `qty = min(budget/(strike×100), 2)` — never changes with vol.

---

## 3. Decision-flow map (data ingest → universe → signals → selection → options → sizing → orders → paper bot)

```
DATA INGEST (per Tier-2 scan, 60 s open / 180 s / 420 s / 120 s power hour; see server/bot.ts:8170-8187)
  Alpaca /v2/assets (daily cache) ─┐
  Alpaca /v2/stocks/snapshots ─────┼─→ bot_engine._scan_market_inner (2700+)
  macro_data.get_macro_snapshot ───┘      VXX 40 bars, SPY 220 bars, yfinance ^VIX, 11 SPDR snapshots, FRED via alt_data (6 h)
UNIVERSE  ~12k symbols → price ≥ $5, vol ≥ 500k → quick score (|Δ%|, volume) → top 5 (VOLTRADE_DEEP_SCORE_CAP)
SIGNALS   deep_score (bot_engine.py:807-1900): momentum .25 + mean_reversion .20 + VRP .25 + squeeze .15 + volume .15
          + macro (P4 ±3/−5) + news + ADX/OBV + intel/alt/social/fundamental + sector momentum (±8, ~50 mapped names)
          + relative strength (+6 if > SPY by 5 %) → rules score → ML blend (ml_model_v2, regime-conditional on P1 inputs)
          → shadow record (taken / rejected_* with +5/+10/+20 d labels)
REGIME    P1 label from macro → get_adaptive_params → MAX_POSITIONS / MIN_SCORE / MAX_TOTAL_EXPOSURE / cooldown
          P3 label → regime_at_entry per candidate (exits) ; P6 stress multiplier (sizing) ; P7 floor regime (floors)
STOCK SELECTION  post-score gates (bot_engine.py:3361-3749): slots, held, cooldown, BEAR/PANIC block, adaptive MIN_SCORE,
          sector veto, $50M dollar-volume floor, blocked list, extreme mover, spread ≤ 0.5 %, instrument selector, Kelly (stocks bucket −EV → blocked), volume share
OPTION SELECTION  tiered_strategy.tier1_csp_core (universe csp_universe top 200 ∪ anchors, affordability ≈ price ≤ $81, first 3 unheld)
          → Node dispatcher (slots 6, CSP-SKIP cooldowns, market-closed defer) → options_execution.select_contract (−0.30 delta, 30–50 DTE, ≤ 2 contracts)
          scanner setups (vxx bands) and stock→options upgrades via instrument_selector
SIZING    position_sizing.calculate_position: Kelly bucket × vol × confidence × VIX × earnings × time × heat × liquidity × DD × stress; clamp 1–8 %
          Node ceilings: 20 % per position × positionSizeMultiplier, 30 % non-floor heat cap
ORDERS    Node executeTrades (15 gates, server/bot.ts:5702-6247) → DAY limit POST (FROZEN); options via select_contract + submit_options_order (FROZEN)
FLOORS    every scan: _resolve_floor_regime → _manage_spy_floor (basket .70/.90/.35/0/0 of equity, QQQ trend caps) → _manage_defensive_floor (GLD)
          + third leg (SVXY VRP harvest 40 % when VXX ratio 1.05–1.25 and falling) + convexity overlay (runs despite enabled: False)
MANAGEMENT  Tier-1 reflex 45 s: manage_positions sync, Node stop/trail/scale-out ladder (Vocabulary B), option POS-KILL backstop, TIME-EXIT;
          options_manager every scan (targets/stops on marks, 21-DTE, DTE-critical, rolls)
NIGHTLY   shadow backfill 22:00 UTC; lookup-ledger backfill; Tier-3 hourly: ML retrain, diagnostics, equity curve, COT/Form-4 updates
```

What Node computes about regime: nothing. It reads `regime_at_entry` per position for exit geometry and ships `macro`/`current_adaptive_params`/`stress_index`/`floor_state` in `/api/system/snapshot`. There is no `REGIME-*` or `FLOOR-*` audit type; floor rebalances are visible only in Python stderr (`[FLOOR-REGIME]`).

---

## 4. The universe today (A4, verified against the code)

- **Stock scan**: `_get_full_universe` (`bot_engine.py:2471-2516`) = every active tradable US equity on Alpaca (~11.6–12.1k), cached 24 h in `universe_cache.json`; fresh movers (top 50 gainers + losers) appended; snapshots in batches of 400; floors `MIN_PRICE 5`, `MIN_VOLUME 500k` (100k in the first half hour) hard-coded in the module — the `system_config` keys of the same name are orphans; quick score ranks by |Δ%| and volume (`:2968-2996`); only the top 5 are deep-scored (`VOLTRADE_DEEP_SCORE_CAP`, `:3097-3146`); the tie-break re-rank is dead (`UnboundLocalError`, `:3228-3310`).
- **CSP universe** (`csp_universe.py:203-326, 618-819`): price $10..equity × 0.25 / 95 (≈ $255), volume ≥ 1M, dollar volume ≥ $20M, not leveraged/inverse (by `/v2/assets` name, `leveraged_etf_filter.py`) or blocked → composite 0.25 IVR + 0.20 VRP + 0.15 liquidity + 0.10 put skew + 0.10 earnings + 0.10 historical + 0.10 stability (only the top 150 by dollar volume get the IVR/VRP prefetch) → top 200 ∪ 17 anchors (anchors bypass every gate) → Tier-1 affordability ≈ price ≤ equity × 0.08 × size_scalar / 95 (≈ $81) → the first 3 unheld names in rank order with no conviction score.
- **Options scanner**: price ≥ $10, volume ≥ 1M, |Δ%| > 2 % → top 400 by |Δ%| + 20 anchors (UVXY among them despite being blocked elsewhere).
- **Cap mix**: no market-cap field exists on any selection path (yfinance `marketCap` is fetched in `analyze.py:2176` and ignored). The only proxies are price × single-day volume tiers. The names traded span mega-cap (NVDA, ORCL, META) to micro-cap (DRAM, BMNR, CIFR, FCEL). The stock funnel is biased to the day's biggest movers (small/micro-cap heavy); the CSP funnel to sub-$81 names with rich IV (1,717 distinct underlyings in 76 archived days).
- **Refresh and reproducibility**: asset list daily, quick screen every scan, CSP layers every 15 min, scanner tier every 60 s. **No point-in-time copy of yesterday's universe exists** (caches overwritten in place; the shadow log keeps only what reached `deep_score`). The universe cannot be audited for survivorship or lookahead after the fact.
- **Hardcoded ticker lists on the selection/regime/sizing path**: ~20 (A4 §2), the consequential ones being `SECTOR_MAP` (~50 names, the only source of the ±8 sector adjustment and the sector-veto fallback), `_beta_map` (~45), `risk_kill_switch._SECTOR_MAP` (~27; everything else is "other"), `CSP_ANCHOR_TICKERS` (17, bypass all gates), `T1_TICKERS_FALLBACK` (17), `BLOCKED_TICKERS`, the floor basket, `_OPTIONS_ANCHOR` (20), the ML `core` training list (30).
- **Where the edge is, as measured by the repo itself** (A4 §5): the composite score is anti-predictive and monotonic (Q5 > 77.4: 42.5 % win / −2.01 % at +20 d; Q1 < 48.2: 56.2 % / +1.04 %); the sector/correlation veto blocks losers (24.1 % win on what it blocks, contaminated by the Unknown-sector bug); momentum entries do not beat random regime-gated entries in BULL and are positive only in CAUTION; the live exit ladder is the worst of four models; mean reversion shows positive alpha versus random on large caps and in illiquid small caps (in-sample, single draw); the floor's exposure schedule halves compounding at equal Sharpe; options lost on sub-day churn (stopped 2026-10-13), the 21-DTE close is the expensive manager rule (+$71/contract if removed, in-sample); live stock trades: 36 closed, 27.8 % win rate, profit factor 0.68.

---

## 5. Data inventory and gaps (A3, verified by live probes this session)

| Need | Available today | Gap / cheapest free fill |
|---|---|---|
| Daily bars, any symbol | Alpaca `/v2/stocks/bars` (active names only, no indices, no delisted); `backtest_v2.fetch_bars(sym, days)` Alpaca-first then Yahoo, 24-h disk cache — **verified this session**: `^VIX`, `^VIX3M`, `^VIX9D`, `^VVIX`, `^SKEW`, RSP, IWM, QQQ, HYG, IEF, LQD, SPY, VXX all fetch; SPY/^VIX/^VIX3M/IWM/QQQ/RSP/HYG/IEF back to 2007-08-28, ^VIX9D to 2011-01-04 | Delisted names are absent everywhere (survivorship for any stock-level backtest). |
| VIX family | spot `^VIX` via yfinance (default 20.0 on failure); VIX1D/9D/VIX/3M/6M/VVIX archived daily since 2022-05-16 at `<DATA_DIR>/datacore_archive/cboevix/<YYYY>.jsonl(.gz)` by `server/cboeVix.ts` (gate-1 validated against FRED VIXCLS); FRED `VIXCLS` archived by `fredMacro.ts` | Wiring gap, not a data gap: Python can read the year file directly (same volume). History before 2022 via `fetch_bars("^VIX3M")` / Cboe `_History.csv`. `^SKEW` is fetched nowhere (Yahoo works). |
| Credit / stress | live 21-day TLT−HYG return spread as an ML feature (`bot_engine.py:1657-1686`, bars without `adjustment=all`); FRED IG OAS `BAMLC0A0CM` and `T10Y2Y` latest values via `alt_data.get_fred_macro` (6-h cache); FRED `BAMLH0A0HYM2` HY OAS, `STLFSI4`, `NFCI` archived by `fredMacro.ts` (key-gated, no Python reader) | HYG/IEF daily ratio from `fetch_bars` (total-return via Yahoo adj close); HY OAS from the fredmacro archive or keyless `fredgraph.csv`. Licence posture: internal signal use only (ICE BofA series are redistribution-restricted; same posture as Cboe). |
| Breadth | none (proxies only; `market_breadth_ma50` has no producer; training computes real breadth on a ≤200-name hand list — train/serve skew) | Build it: once per day post-close, daily bars for the price/volume-passing universe (~1–2k names, ~40 batched calls through `alpaca_throttle`) → % above 50/200-d, new 20-d/52-w highs−lows, median drawdown from 52-w high, RSP/SPY and IWM/SPY 1- and 3-month relative strength. |
| Cap-matched benchmarks | QQQ fetched every scan (floor trend caps); IWM streamed and anchored; **RSP never fetched** | one `fetch_bars` call per day each for SPY/RSP/IWM/QQQ. |
| Sector ETF per stock | 11 SPDRs → sector names (`macro_data.SECTOR_ETFS`), one-day momentum only; stock→sector only for ~50 mapped names + a sparse yfinance cache of deep-scored names | SPDR 50-day MAs are one batched bars call; the stock→sector join for 12k names needs EDGAR SIC (free) — a small pipeline, filed, not built here. |
| Market cap per name | none on the bot path (Polygon reference if `POLYGON_KEY`; Finnhub profile; yfinance `info`) | SEC XBRL frames `EntityCommonStockSharesOutstanding` × close, quarterly, keyless — filed. |
| Options chains / IV | Alpaca snapshots (indicative feed when OPRA not entitled); chain archive since 2026-07-04 (no OI); `chainIv.ts` ATM-30 IV per underlying-day | IV *rank* (52-week percentile of ATM IV) needs the archive to age; today's "IVR" is an HV-rank/VRP proxy. |
| Regime history for research | `backtest_v2.regime_series` from VXX (real from 2018-01-25); `regime_detector_compare.py` | VIX/VIX3M history to 2007 covers 2008, 2015–16, 2018, 2020, 2022 at the index level. |

Every input above is free or already paid for. No new paid data is proposed.

---

## 6. Where the falling-knife failure comes from in the code

1. **The funnel selects gaps.** `quick_score = 3·min(|Δ%|,15) + 2·min(vol/1M,5) + bonuses` (`bot_engine.py:2968-2996`) is sign-blind: a −30 % name and a +30 % name rank the same. Only the top 5 are deep-scored, so the candidate set *is* the day's largest moves.
2. **The mean-reversion sub-score rewards the knife, and so do four other components.** `strategies/mean_reversion.py:24-45`: RSI < 20 → +40, 5-day change < −10 % → +30, volume ratio > 2 → +20; "STRONG BUY" ≥ 65; weight 0.20 in `deep_score` (`bot_engine.py:1261`). A stock down 35 % in a month on heavy volume scores near the maximum on this component. The momentum component uses 12-1 momentum (`strategies/momentum.py:5-24`), which *subtracts the crash month*, so a prior-year winner that just fell 35 % still scores +60 on momentum while the same crash is credited again by mean reversion; ADX is direction-agnostic (+6 for a "strong trend" that is a collapse, `:1271-1274`); elevated IV on a crashing name earns the VRP bonus (+5…+9, `:1144-1158`) and the vol-premium bonus (+8); analyst "upside" after a crash earns +6; a "Buy the Dip" contrarian flag earns +8 twice (`:1449-1453, 1463-1465`). A plausible combined score for such a name is 65–80, clearing `MIN_SCORE` 63/67, and with VRP > 5 the side flips to a short ~30-delta put (`:1862-1866`) — long delta either way.
3. **No trend, base or relative-strength guard exists on a single name.** `deep_score` computes `_above_ma10` and `_above_ma200` only as ML features (`bot_engine.py:1512-1530`); the +6 relative-strength bonus fires only when the name beats SPY by 5 % (`:1470-1473`); the ±8 sector-momentum adjustment reaches only the ~50 names in `SECTOR_MAP` (`:1223-1232`); `price_vs_52w_high` is always 0 live because no scan path writes `quick_result["high_52w"]` (`:1508-1509`, train/serve skew on what training calls its strongest predictor). There is no "new 20-day low in the last 3 sessions" check, no reclaim-the-10-day test, no sector-ETF-below-its-50-day test.
4. **No regime or breadth condition on dip-buying.** The only regime effects on a stock entry are the slot count (0 in NEUTRAL/BEAR/PANIC), MIN_SCORE, and the ±3/−5 macro adjustment from `risk_on/risk_off`. With SPY within 3 % of its high the label is BULL or NEUTRAL regardless of breadth, so in exactly the narrow tape the owner describes the system is either fully open (BULL, 6 slots) or closed for an unrelated reason (NEUTRAL, 0 slots) — never "open to leaders, closed to laggards".
5. **The ML blend cannot condition on regime.** 99.9 % of its serve-time records carried one regime label until 2026-10-19, and its `price_vs_52w_high` input is a constant.
6. **What the evidence says.** The shadow bands (2026-10-06, +2/−4 labeler) rank buying a ≥10 % same-day *drop* as the best taken cell (65 % win, −0.10 %) and chasing a +10..20 % *gain* as the worst (35.1 %, −1.79 %); the lookup ledger built to test "looks good when down big on bad news" has no readout yet (n < 20). The stock sleeve's loss is small (−$499 on 178 fills). The knife that measurably cut the account is the CSP sold into earnings gaps (APH 2026-07-16 −48 %; 8 of the worst 10 simulated entries had an earnings date inside the window) and the sub-day options churn. The guard the owner asks for is therefore justified by *design* (there is nothing that would stop a falling-knife buy) and by the CSP evidence, not yet by stock-level evidence — which is why it ships flagged and counterfactual-logged, not as a conviction.

---

## 7. Where the volatility-whipsaw failure comes from in the code (A2 §4, summarised)

| # | Path | Mechanism | Evidence |
|---|---|---|---|
| W1 | Mark-based stop and target (`options_manager.py:1203-1255`) | `current_price / initial_credit ≥ 2.0` evaluated every scan on Alpaca marks from the indicative feed; closes at the **ask**; a wider quote alone moves the mark toward the stop | sim: the −2× stop decides nothing on daily marks (paired +$0.5), so live cost is intraday quote width |
| W2 | No cooldown after a manager close | `tier1_csp_core` de-dupes against *held* underlyings only (`tiered_strategy.py:508-510`); the 15-minute-cached universe re-proposes the name next scan | NVDA 133 fills −$1,126; PATH 42 fills |
| W3 | Entry/exit DTE conflict | fixed (v1.0.1056/1117); the remaining structure is the ≈30-DTE page-fill entry meeting the 21-DTE close 8–9 days later (`CSP_ENTRY_MIN_DTE 30`, `DTE_EXIT_THRESHOLD 21`) | sim: `no_dte_exit` +$71 ± 18 per contract (in-sample) |
| W4 | Paths outside the manager's hold | Node option POS-KILL backstop (−150 %, market, ~45 s loop, `server/bot.ts:6982-7043`), TIME-EXIT (>14 d, market), LIQUIDATE-ALL, DTE-critical on weeklies, timestamp-less state (`_held_minutes = 999` default) | 168 sub-hour closes exceed what the DTE conflict alone produced |
| W5 | Non-marketable limit → 12-min sweep → 60-min cooldown → re-dispatch | `_optimized_limit_price` (`options_execution.py:766-803`), `sweepStaleOrders`, `cspSweepCooldown.ts` | 51 cancels on 2026-10-12; broker brake v1.0.1158 |
| W6 | Liquidity proxy | no OI in the snapshot; `_is_liquid` passes on quote size × 10 or 10 contracts of volume with a 10 % spread | thin chains in exactly the tape where size is pulled |
| W7 | Two rulers on one lot | manager at Tier-2 cadence vs Node reflex at 45 s with different thresholds | — |
| W8 | Vol scaling dropped | tier `size_scalar` not passed to `select_contract`; `_regime_scalar(None)` = 1.0; `qty = min(budget/strike, 2)`; delta fixed at −0.30; no spread-versus-naked choice; no pause in PANIC other than the exposure cap | §2.3 |

The regime label is not the proximate cause of any of W1–W7; it is the cause of W8 (nothing vol-aware reaches the contract) and of the all-or-nothing exposure cap.

---

## 8. Transition behaviour today

- Only P7 (the floors) has hysteresis: 3 distinct macro readings (≈10–15 min at scan cadence) to confirm a calmer label, immediate escalation, fallback holds. Everything else re-derives the label from a 5-minute cached snapshot on every scan.
- Downgrades and upgrades are symmetric everywhere except P7. There is no minimum dwell, no confirmation count in days, no confidence, no `pending_transition` flag, no transition log, no persisted history. The dwell counts *readings*, not trading days.
- Exposure steps are coarse and inconsistent: floor 90 → 70 → 35 → 0 → 0; stock slots 0 → 6 → 6 → 0 → 0; options cap constant; GLD 8 → 8 → 15 → 30 → 40.

---

## 9. Dead, duplicate and silently interacting code found (removal or consolidation candidates)

| Item | Location | Kind |
|---|---|---|
| Legacy `get_market_regime` body incl. Markov tie-break | `system_config.py:413-438` | dead decoy |
| `REGIME_*` mirrors in `BASE_CONFIG` (`REGIME_BEAR_THRESHOLD 0.96`, `REGIME_VXX_*`) | `system_config.py:289-302` | legacy fallback only |
| `STOP_SCALAR_*`, `TIME_STOP_DAYS` regime overrides, configured stop/TP block | `system_config.py:264-274, 374-381` | dead (KB#48) |
| `MarkovRegime.size_multiplier`, STRONG_BUY/SELL signals | `markov_regime.py:166-184` | unused |
| Convexity overlay running against `enabled: False` | `bot_engine.py:4063-4071, 4695` | config-disabled, code-live |
| `options_scanner` own VXX/SPY-MA fetchers | `options_scanner.py:203-262` | duplicate of macro_data |
| `vix_regime` label mismatch | `macro_data.py:109-118` vs `position_sizing.py:586-591` | silent interaction |
| DD sizing scalar key mismatch | `position_sizing.py:996-1026` | dead |
| `get_regime_caps` hard-codes `markov_state=1, spy_below_200_days=0` | `tiered_strategy.py:114-123` | KB#20a |
| Tier `size_scalar` dropped at dispatch | `server/bot.ts:5251` vs `:5263` | dead by wiring |
| Three floor-ticker ownership lists | `bot_engine.py:123-128`, `server/bot.ts:409`, `_floor_owned_tickers` | duplicate |
| `MIN_PRICE`, `MIN_VOLUME`, `OPEN_HOUR_MIN_VOLUME`, `SCREENER_*`, `DEEP_SCORE_TOP_N`, `STOP_COOLDOWN_SECONDS` | `system_config.py` | orphan keys |
| Node `minScoreThreshold` beside the adaptive `MIN_SCORE` | `server/bot.ts:507, 4786` | duplicate gate |
| `stress_index` "breadth" and "vix_term_structure" components | `stress_index.py:76-132` | misnamed (SPY-vs-MA; VXX ratio) |
| Markov detector fed fractional returns against percent thresholds | `regime_inputs.py:60` vs `markov_regime.py:77-109` (pinned by `test_regime_inputs.py`) | units bug — state pinned Neutral |
| Stress "crisis" multiplier 0.0 clamped back to the 1 % floor | `position_sizing.py:1028-1050` | halt that never halts |
| Regime `MAX_POSITION_PCT` never read by sizing; `ewma_vol` vs `ewma_rv` key mismatch | `position_sizing.py:1050`, `:967` vs `analyze.py:1193` | inert cap; vol scalar at default |
| `config_overrides.json` populates `cfg`, which no trading path reads | `system_config.py:612-639` | inert override mechanism |
| Three stale PANIC-1.30 fallback copies | `system_config.py:414-438`, `macro_data.py:338-347`, `bot_engine.py:3467-3472` | exception-path decoys |
| `risk_kill_switch` `tier_blocks` / `blocks_entries` / `should_liquidate_all` | `risk_kill_switch.py:306-308, 630-639` | computed, no consumer (mechanisms FROZEN — note only) |

---

## 10. What the history forbids or constrains (A5, abbreviated)

- Do not re-propose: dual-momentum SPY/QQQ (killed OOS 2026-07-04), the HMM 5-state variant (rejected 2026-04-07 on an in-sample sweep — a new attempt needs its own OOS test), the GEX proxy, VRP timing, SSO/QLD in T3, adaptive-lookback shorts, and the eight foreign-field transition probes (CSD, R_t, Omori, hazard rate, permutation entropy, Hurst, CUSUM-on-insiders, Kuramoto — all failed gate 2).
- Human decisions on record: no paid tail hedge (2026-04-17, 2026-04-22 twice); GLD always-on inflation thesis and the 65 %-tech basket (2026-04-22); floor stays 70 % in BULL (2026-05-18); composition is the human's call (2026-10-14/20); kill-switch mechanisms frozen; "research what is best before implementing" (2026-10-20); the dip-buy intuition to be *measured*, not traded (2026-10-22); "increase risk within reason" explicitly deferred until the score inversion is fixed (2026-10-26). GOAL = log-wealth vs buy-and-hold SPY.
- Lessons: every threshold move hit a second reader (MAX_POSITIONS → CSP blackout; PANIC 1.30 → 1.40 moved the exposure cap 20 pp; floor step × drift band × boundary flicker = 7 round trips a day); inputs failed more often than thresholds; "disabled" is a claim, not a state; same-day add/retune/revert came from in-sample sweeps on survivor universes.
- Order of operations already on record: score decomposition → MIN_SCORE ablation → regime-slot counterfactuals (first read 2026-11-16) → floor exposure schedule → VIX3M/HY-OAS regime upgrade (gate 1 first). This program's plan respects that order (FINDINGS.md).

---

## 11. Research-versus-code gap analysis

Filled from RESEARCH.md (Stage A, Part 1). "Does it" uses four states: **does**, **partly**, **wrong**, **doesn't**.

| Research finding (RESEARCH.md §) | Evidence | Code state | Verdict |
|---|---|---|---|
| Long-horizon trend filter on the index (10-month SMA / 12-month TSMOM / 200-day) cuts drawdowns at a CAGR cost; whipsaw cost in sideways tapes; the slow leg governs re-entry, fast legs (1–3-month, vol) only downgrade (§2.1, §4.4) | strong, long OOS | 200-day enters only as a 10-day *streak* that forces BEAR (floor → 0, slots → 0); the 50-day ratio (0.94/0.98) is the main trend input and is fast; no TSMOM; QQQ death-cross caps on the floor only | **partly, and tuned fast** |
| Volatility targeting / vol-managed exposure improves Sharpe and tails for equities, with leverage capped and a 20–60-day realized window (§3.1) | strong (Moreira–Muir; Man Group); mixed on CAGR after costs | per-name ATR scalar only; no portfolio-level realized-vol target; VIX *level* scalar (dead on the CSP path); stress multiplier stacks | **partly / wrong (stacked)** |
| VIX term structure (VIX/VIX3M > 1, VIX9D/VIX > 1) is a *coincident-to-slightly-leading* stress flag with few false positives; VIX level percentile beats raw level (§3.3) | mixed-to-strong (practitioner, Cboe, academic basis) | VXX ÷ 30-day mean only; term structure archived, unread; no percentile | **doesn't** |
| Breadth (% above 200-d, new highs−lows, equal-weight vs cap-weight) identifies narrow markets where dip-buying fails; leads index drawdowns modestly, coincides at best (§2.3) | mixed (practitioner-heavy; coincident more than leading) | none | **doesn't** |
| Short-term reversal / dip-buying works in calm, broad tapes and fails in downtrends and high vol; the 52-week-high effect favours leaders over laggards (§2.2) | strong (academic) | mean-reversion sub-score unconditioned; no RS filter; funnel selects gaps | **wrong** |
| Credit spreads (HY OAS, HYG/IEF) are a slow, confirming risk-off input, not a trigger (§3.5) | mixed | IG OAS latest value inside the stress index; HY OAS archived unread | **partly** |
| Dwell (6-day majority filter) + small band + asymmetric *signal speed* (fast legs downgrade, slow leg re-enters — not an extra N-day re-entry delay) reduce whipsaw at no CAGR cost; graduated exposure beats binary (§4.3) | mixed-to-strong (practitioner; ensemble-lookback literature) | floors only (3 readings, escalation immediate); everything else symmetric and instantaneous; exposure steps coarse | **partly (floor), doesn't (rest)** |
| HMM / jump models as one vote, walk-forward refit only; label switching and lag are the failure modes (§1) | mixed; no robust OOS edge over simple rules for a small system | Markov order-2 with a hard-coded 2016–2026 matrix, demoted but still labelling exits and shadows | **wrong (hard-coded, not refit, no OOS)** |
| Options: sell premium by IV rank/percentile, reduce size and go further OTM when IV is elevated, prefer defined risk in stress, no new short puts in crisis, rule-based exits (§3.6) | mixed (practitioner) to strong (PUT index behaviour in crises) | IVR is an HV/VRP proxy with no gate; delta fixed; qty ≤ 2 regardless; vol scaling dropped; no crisis pause; exits mark-based | **doesn't / wrong** |
| Small caps fall harder, recover later, are more credit-sensitive; thresholds may differ by cap (§2.5) | strong (descriptive) | no cap field; SPY-only regime | **doesn't** |
| ML regime classifiers rarely beat simple rules OOS; use for monitoring, not gating (§4.2) | mixed-to-weak | ML blend is regime-conditional but its regime input was constant; score anti-predictive | **wrong** |
| Parameter robustness: round numbers, ±20 % sensitivity, few parameters (§4.6) | strong (Zakamulin; Newfound) | thresholds set by in-sample sweeps on survivor universes (0.94/0.95/0.98/1.05/1.15/1.40) | **wrong** |
