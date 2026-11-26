# EXECUTION LAYER + DATA LIFECYCLE — program charter

Installed 2026-11-19. PART 1 and PART 2 below are the human's text,
verbatim, unedited. Everything above PART 1 is session-maintained.

## READ THIS FIRST

- MANUAL, FOREGROUND PROJECT. Scheduled routines do NOT build any of this
  and do not take work from this file. Only a session the human is
  directing in the foreground works on it.
- Every PR of the execution layer carries the `manual-review` label and
  never auto-merges; the human merges. (Addendum 1 names the exceptions
  that use the normal CI-gated auto-merge: the A8 rules PR, the A9 UI
  catch-up PRs, the A10 triage, the A11 rule consolidation.)
- UNTIL `ci.yml` excludes `manual-review` from the auto-merge job, NO
  execution-layer PR may be opened from a `claude/` branch: it would
  merge and deploy on green. That exclusion is the first Phase 0 change.
- Stop at every phase gate and wait for the human.
- Fail closed: missing, stale or indicative-only data means skip.

## RESUME STATE (session-maintained; newest first)

2026-11-25 (foreground session, the human's answers of that day)
- OPTION QUOTES: the human will NOT pay for an options quote feed ("i dont
  want to pay for options quotes"). The account stays on Alpaca's
  indicative feed. Consequence under this charter's own fail-closed rule
  ("missing, stale or indicative-only data means skip"): Phase 0 (logging)
  and Phase 1 (shadow, sends nothing) can run; Phase 2 onward cannot price
  an option leg and stops at its gate until this is revisited or a free
  real-time source is found (BUILD-FIRST: none identified).
- OWNER BUTTONS: approved ("i like the onwer buttons"). They use the
  EXISTING routes and order path (this charter's Phase 4 rule is not yet in
  force); when the layer reaches Phase 4 the manual order form must route
  through it like every other order.
- STILL OPEN, asked again that day: decision 1 (the ci.yml edit), 2 (own
  service vs daemon), 3 (paper vs live), 4 (model input). The human asked
  what Phase 0 is and was told; no go-ahead yet. Phase 0: NOT STARTED.
- Routine fixes: the human asked "if the routines are out dated then
  what"; the proposal (pause product-pm, edge-late, daily-midday; add the
  attach-the-repo step to the older prompts) was restated and awaits a yes.

2026-11-19 (install session)
- Section 14 report delivered to the human: six routine runs failed with
  `USAGE_LIMIT_REACHED` (edge-late 11-18; product-am, preopen-health,
  daily-midday, improve-midday, product-pm 11-19); `voltrade-edge`
  11-19 "succeeded" with no repository attached; the weekly FAA chart
  bake has been blocked on two missing environment values. Proposed
  fixes are with the human; none applied to a routine yet.
- Addendum 1 work started (it carries no approval stop): A8 standing
  rules (this PR); A10 open-PR triage (one open PR, #1706, superseded
  by 666765a4 on main); A9 UI coverage audit next.
- Phase 0: NOT STARTED. Nothing in `execution/` exists.
- OPEN DECISIONS put to the human 2026-11-19, unanswered at install:
  1. `.github/workflows/` is a FROZEN PATH; the spec orders one edit to
     `ci.yml` (the `manual-review` exclusion). Addendum 1 §A11 also
     names `ci.yml`. Read as the human's override; confirm.
  2. "Own Railway service" and "communicate over the existing
     Unix-socket RPC daemon" cannot both hold: a Unix socket does not
     cross Railway services. Recommended: separate service over
     Railway's private network. The human must create the service (no
     Railway access from a session; Dockerfile and railway files are
     FROZEN PATHS).
  3. Paper vs live. The paper account's fills are a simulation; fitting
     A and k on paper in Phase 2 measures the simulator. Phases 3-4 say
     "live" and only a paper account exists today.
  4. The model input. NetEdge ranks by calibrated p_win / p_loss /
     p_time; composite score v2 holds the ML input neutral because it
     measured as noise (#1660). Recommended: proceed through Phase 1
     (sends nothing) and let its Brier / reliability report decide.
- TO VERIFY IN PHASE 0, not assumed: the Addendum's statement that the
  pattern-day-trader rule ended 2026-06-04 (check Alpaca's current
  account rules; no PDT logic is built either way without the human);
  the options data feed (indicative vs full OPRA); mleg support; lot
  selection by API; websocket symbol cap; rate limits.
- FACTS THE SPEC ASSUMES THAT ARE NOT TRUE TODAY: there is no Postgres
  and no object-storage tier for tables (state is SQLite + JSON files on
  one Railway volume; an R2 bucket exists for map tiles only). Where the
  Section 8 tables live is a Phase 0 report item.

---

## PART 1 — the human's spec (2026-11-19, verbatim)

# VolTradeAI: Execution Layer + Data Lifecycle Build

You are building a new execution layer for VolTradeAI and adding a platform-wide data lifecycle rule. Read this whole prompt before writing code. Work in phases, stop at every gate, and wait for Mike's approval before starting the next phase.

## 0. Ground rules for this project

- This is a **manual, foreground project**. Do NOT add any of this work to the daily Claude Code routines' task pool. Do NOT let it auto-merge. Label every PR `manual-review` and exclude that label from the auto-merge job in `ci.yml`.
- Work on a dedicated branch per phase. New code lives in a new `execution/` module in the Python trading engine.
- Feature flag `EXEC_LAYER_MODE` = `off | shadow | paper | ab | live`. Default `off`. The existing order path is untouched until Phase 4 and remains the fallback.
- Run the fast execution worker as its **own Railway service** so its failures can't take down the engine or the map.
- Fail closed: missing, stale or indicative-only data means skip the trade, never guess.
- Every order sent must be a genuine attempt to fill. No probing, layering, or orders placed to move prices. No multiple API keys/accounts to exceed rate limits.
- Before building, verify against current Alpaca docs and report back: (a) multi-leg (mleg) options order support, order classes and price increments; (b) whether specific-lot selection is supported via API; (c) which options data feed this account has (indicative vs full OPRA); (d) websocket option-symbol subscription cap; (e) current Trading API and Market Data API rate limits. Put findings in `execution/BROKER_NOTES.md`. If the account only has the indicative options feed, flag it: the layer cannot go past Phase 1 without the full feed.
- Tax logic is engineering only, driven by a config rules table to be reviewed by a CPA. Do not hard-code tax rates.

## 1. Objective

One execution layer every strategy routes through. Rank every candidate trade by:

```
NetEdge = E[R] - Slip_exp - Fees - Tax_exp        ($ per contract/share)
Score   = fill_prob * NetEdge / CapitalAtRisk
```

Tax is a cost term, never a hard veto. Trade only if NetEdge > 0 after sizing.

Scope: equities, single-leg equity and index options, vertical spreads via complex orders. Crypto later, behind a rules switch.

## 2. Architecture: two speeds

**Slow layer** (every 60 s, configurable 15–300 s; runs in the trading engine):
1. Pull current LightGBM outputs: direction, calibrated p_win, p_loss, p_time, triple-barrier levels, max hold.
2. Pull one option chain snapshot per signaled underlying per cycle.
3. Generate candidates: the stock; single options delta 0.20–0.70 in allowed expiries; verticals 1–5 strikes wide, same expiry; index substitutes (SPX/XSP/NDX) for ETF-equivalent signals.
4. Apply hard filters (Section 4), compute all variables (Section 3), costs, score, Kelly size.
5. Shortlist top K=3 per signal with all variables frozen; TTL = 2 slow cycles.

**Fast layer** (separate async worker, event-driven):
1. Websocket quotes for shortlisted legs only.
2. Pre-flight re-check on fresh quotes; if NetEdge dropped >25% or < 0, go to next shortlist entry.
3. Slice, walk, fill, log, update lots and tax ledger.
4. On cap without fill: cancel, log no-fill, cooldown; re-queue only if the next slow cycle still ranks it.

**Shared:** rate-limit governor (token bucket, one per account, covers ALL REST calls), quote freshness monitor (max age 2 s fast / 90 s slow), append-only execution log and lot ledger, kill switch `EXEC_KILL=1` that cancels all working orders and halts new ones within one governor cycle. Slow and fast layers communicate over the existing Unix-socket RPC daemon.

## 3. Variables (compute and log every one, per candidate, per cycle)

Spread uses legs L (long) and S (short); single instruments use one leg.

**Price/spread**
- bid/ask per leg; mid_leg = (bid+ask)/2
- mid_pkg = mid_L − mid_S
- natural_pkg = ask_L − bid_S
- half_width = natural_pkg − mid_pkg
- rel_width = half_width / mid_pkg
- tick = exchange min increment (0.01 / 0.05); all prices round to tick in our favor
- strike_width = strike_S − strike_L; max_payoff = strike_width − debit; payoff_ratio = max_payoff / debit

**3x cap** (R_min default 3.0, per strategy config):
```
Cap_3x = strike_width / (1 + R_min)
Limit  = min(Cap_3x, natural_pkg, NetEdge-breakeven price)
```
Discard the candidate if mid_pkg > Cap_3x.

**Liquidity/depth**
- size_bid, size_ask per leg; depth_pkg = min(size_ask_L, size_bid_S); depth_ratio = depth_pkg / qty
- volume_today, open_interest per leg
- liq_score = weighted z-score of −log(rel_width), log(depth_pkg), log(volume), log(OI) (weights fitted)
- liq_bucket = quintile of liq_score over last 20 sessions (1 thin … 5 deep)

**Market state**
- tod_bucket: open 9:30–10:00, mid-morning, midday, afternoon, close 15:30–16:00 ET
- iv per leg, iv_rank (252d), rv_5m of underlying, dte, moneyness = log(K/S)
- delta, gamma, theta, vega per leg and net
- quote_age, event_flag (earnings/FOMC/CPI within hold window)

**Model/payoff**
- p_win, p_loss, p_time (sum to 1; isotonic-calibrated)
- R_win, R_loss, R_time = instrument P/L at each barrier, repriced through the instrument (Black-Scholes at barrier spot, IV assumption, remaining DTE), not the stock move
- E_R = Σ p_o · R_o; hold_exp from barrier model

**Costs**
- Slip_exp and fill_prob from the fill model (Section 5), entry AND exit
- Fees from a fee table (broker + regulatory + exchange, per contract per leg, both sides)
- Tax_exp (Section 9)
- CapitalAtRisk = debit × 100 × qty for spreads; margin requirement otherwise

## 4. Hard filters (before scoring)

quote_age within limit · rel_width < 0.15 · depth_pkg ≥ 1 per leg · OI ≥ 100 per leg · dte 7–60 (config) · payoff_ratio at mid ≥ R_min · event_flag excluded unless strategy allows · net delta within portfolio exposure cap. Ties within 5% on Score → higher liq_bucket, then shorter hold_exp.

## 5. Fill model (Avellaneda–Stoikov exponential)

d = distance of limit from natural in ticks.
```
λ(d)       = A · exp(−k·d)                 fills per second
P_fill(d,t) = 1 − exp(−λ(d)·t)
```
- Fit A, k per key = (instrument_class, liq_bucket, tod_bucket, dte_bucket). Classes: equity; single option near-the-money; single option far OTM/short DTE; vertical via complex order; index option.
- Fitting: each walk step is an exposure interval (distance d, time t, filled y/n). Max-likelihood exponential survival, hierarchical partial pooling to the class mean. Need ≥30 fill events per key to use its own fit; else class-level. Refit nightly, keep last 3 versions, alert if k moves >50% day over day.
- Cold start: assume 50% fill at mid within 30 s for bucket 5, down to 10% for bucket 1; solve A, k from that. Stay in paper until each class has 30 events.
- Expected slippage over a schedule of steps j:
```
P_j      = P_fill(d_j, t_j) · Π_{m<j} (1 − P_fill(d_m, t_m))
Slip_exp = Σ_j P_j · (fillprice_j − mid_pkg) · 100 · qty
fill_prob = Σ_j P_j
```
- Model the exit side too, using the key shifted one dte_bucket shorter.

## 6. Kelly sizing (on net edge)

```
f* = argmax_f Σ_{o∈{win,loss,time}} p_o · ln(1 + f · R_o,net / |R_loss,net|)
f_used = f* · κ · c_calib · c_liq
```
- κ = 0.25 default, never > 0.5
- c_calib ∈ [0,1] from recent Brier score of p_win
- c_liq = min(1, depth_ratio)
- Hard caps: 2% equity at risk per trade, 6% per underlying, 20% total open risk; per-class max contracts from config. f* ≤ 0 → no trade.
- qty = floor(f_used · equity / risk_per_contract); qty = 0 → drop, size next shortlist entry.

## 7. Execution

**Order types:** limit DAY only, never market. Verticals = single multi-leg (mleg) limit at net debit to the complex order book; both legs fill together or not at all. Exits mirror at net credit.

**Start price:** `start = mid_pkg − θ_start · half_width`, θ_start fitted per key (0–0.1 on deep keys).

**Exponential walk:**
```
price_j = min(Cap, start + (natural_pkg − start)·(1 − exp(−α·j)))
dwell_j = clamp(ln2 / λ(d_j), 2 s, 60 s)
```
α fitted per key (small on deep books, larger on thin). Each step = one cancel-replace call. Max 10 steps per child; cap reached and dwelled out → cancel, log no-fill.

**Re-anchoring:** on each quote update, recompute mid/natural. Mid moves against by > one half-width → re-check NetEdge, cancel if < 0. Mid moves in favor → re-anchor start to new mid. Never exceed Cap.

**Slicing:** child = min(qty, max(1, floor(depth_pkg × 0.5))); sequential per instrument; cooldown between children = last fill time, min 5 s. Each child = its own tax lot. At 1–10 contracts most orders are one child.

**Rate governor:** ceiling 120 req/min (60% of 200); 20/min reserved for positions/account/cancels; remainder token-bucketed for order steps (~1 replace/s sustained). On 429: backoff 1,2,4,8 s, pause new walks, reconcile order state before any resend. Market data via websocket, outside this budget.

**Conduct:** one working order per instrument per side; alert if cancels:fills > 20:1, auto-pause > 50:1; client order ids = (signal id, child index) for idempotent restarts.

## 8. Logging tables (append-only)

| Table | Row per | Key fields |
| --- | --- | --- |
| exec_candidates | candidate per cycle (top 20 per signal + all shortlisted only) | all Section 3 vars, score, rank |
| exec_orders | child order | signal, key, start, cap, qty, schedule params, model versions |
| exec_steps | price step | price, d ticks, dwell, quote snapshot, outcome |
| exec_fills | fill | price, qty, time-to-fill, distance from mid at send/fill, fees |
| lots | lot | id, instrument, qty, basis incl fees, open ts, wash/holding adjustments |
| tax_events | realized G/L | lot ids, character, wash adjustments |
| param_versions | nightly fit | A, k, α, θ_start per key, n, diagnostics |

**Daily metrics:** effective spread captured = (natural − fill)/half_width; implementation shortfall vs decision-time mid; fill rate, median time-to-fill, no-fill rate; markouts at 5 s / 60 s / 5 min; predicted vs realized slippage; p_win Brier + reliability; request rate vs ceiling; cancel:fill; after-tax vs pre-tax P/L.

## 9. Tax engine (config-driven, CPA review before live)

Rules table, versioned, per asset class:
- Equity/ETF: ST ≤ 1 yr, LT > 1 yr; wash sale ON (±30 days; options on the stock count as substantially identical).
- Equity options: mostly ST; wash sale ON.
- Index options (SPX, XSP, NDX, RUT): Section 1256 60/40 regardless of hold; year-end mark-to-market; wash-sale treatment per CPA.
- Crypto spot: wash sale OFF by default with a switch ready (pending legislation, not law as of Nov 2026).

```
Tax_exp = Σ_o p_o · max(0, G_o) · τ(o) − Shield
```
G_o uses the lots that would actually be sold; τ from config (federal + state, by character); Shield = YTD realized net losses available to offset.

**Lot selection on exit (specific ID if broker supports it; else simulate and report):** loss lots first when offsetting gains with no wash conflict → highest basis → LT before ST at equal gain → don't sell a lot within 21 days of its 1-year mark if `G·(τ_ST − τ_LT) > E[drawdown over remaining days]`.

**Wash-sale guard:** before any buy, add any disallowed-loss value to NetEdge cost; before any loss sale, block the slicer from re-buying that instrument for 31 days unless NetEdge still clears. Slicer and harvester share one calendar. Holding-period hedges (protective puts) stay disabled until CPA review.

## 10. Data lifecycle (platform-wide; add to CLAUDE.md in Phase 0)

Add this section to CLAUDE.md verbatim as part of Phase 0, as its own PR:

> **Data Lifecycle: Use, Learn, Compact, Delete.** Raw data is kept only while something still learns from it, then replaced by the compact form the system actually uses, and deleted. Unbounded data growth is a bug.
> - Tiers: Hot (raw, in use, Postgres/RAM, days–weeks) → Learned (fitted params/model weights, versioned, last 3) → Compact (aggregates, bars, per-key stats, quantile sketches; Parquet in object storage) → Sample (~1% raw, Parquet) → Deleted.
> - Every new table, cache, or archive must declare in its PR: hot window, compact form, sample rate, protected y/n. No declaration, no merge.
> - No unbounded in-memory structures: every cache has max size (LRU) + TTL; every buffer has a cap.
> - Nightly compaction per table: select rows past hot window → confirm all consumers are done (fits ran, no open trade/audit/job references) → write compact + sample → verify counts/checksums → only then delete raw and vacuum. Any failure aborts the delete.
> - Prefer storing what was learned over what was seen.
> - Re-fetchable public data (NOAA, NASA, EIA, SEC, OSM, etc.) may be compacted and deleted freely.
> - Market-hours-only services sleep outside market hours.
> - Weekly storage/memory report per service; steadily climbing memory between deploys is investigated as a leak.
> - Protected (never compacted or deleted): tax/trade/order/fill/lot records (≥ 7 years); proprietary non-refetchable archives (AIS transit history, archived space-weather sequences, ADS-B history for replay) move to cold object storage instead; anything a human flags protected.

**Execution layer retention:** exec_candidates hot 7 days → daily per-key counts, score quantiles, shortlist rate. exec_steps hot 30 days → per-key fill-at-distance curves. exec_orders, exec_fills, lots, tax_events kept permanently.

Do NOT retrofit existing tables in this project. Existing-data cleanup is a separate manual pass Mike will trigger later.

## 11. Capacity and cost targets

- Trading API usage ≤ 60% of limit over any 1-minute window.
- Market data REST: chain pagination at ~20 underlyings ≈ 60–100 calls/min; requires a plan above Basic. Back off or lengthen the slow cycle for low-priority underlyings if near limit.
- Fast worker < 256 MB RAM; slow layer < 512 MB. Chains dropped from memory after scoring.
- Fast worker sleeps outside 09:25–16:05 ET and on market holidays.
- Nightly fit runs on existing compute; no GPU.

## 12. Phases and gates (stop and report at each)

0. **Broker verification + baseline logging.** Write BROKER_NOTES.md. Add CLAUDE.md data lifecycle section. Instrument the CURRENT order path with Section 8 tables. Gate: 2 weeks of clean logs.
1. **Shadow.** New layer computes candidates, scores, sizes, schedules; sends nothing. Gate: predicted vs actual legacy fills compared; 5 sessions without crashes.
2. **Paper.** Full walk on paper account. Gate: ≥30 fill events per instrument class; A, k stable.
3. **A/B live, 1 contract max.** Randomize 50/50 at signal level until 200 fills per arm. Gate: effective spread captured improves with p < 0.05, fill rate drops ≤ 10 points.
4. **Live with Kelly sizing.** Legacy path becomes fallback.
5. **Tax engine live** after CPA review of the rules table. Crypto switch stays off unless law changes.

## 13. Acceptance

- Every Section 3 variable computed, logged, and visible in a debug view for any trade.
- No spread ever fills one leg alone.
- Request rate never exceeds 60% of broker limit in any 1-minute window.
- Kill switch tested in paper with orders working.
- Effective spread captured beats baseline significantly.
- Memory and storage within Section 11 targets.

## 14. Before starting Phase 0

The daily routines currently have failing runs. Before this project takes any capacity, report the failing routines and their error messages so they can be fixed first.

References: Avellaneda & Stoikov (2008); Almgren & Chriss (2000); Guéant, Lehalle & Fernandez-Tapia (2012); Cartea, Jaimungal & Penalva (2015); Kelly (1956).

---

## PART 2 — Addendum 1 (2026-11-19, verbatim)

# VolTradeAI Execution Layer Addendum 1

Paste this into the same Claude Code session that received the main execution-layer prompt. It adds rules on top of that prompt; everything in the main prompt still applies. Same ground rules: manual project, `manual-review` label, no auto-merge, not in the daily routines' task pool, stop at each phase gate.

## A1. Early assignment risk (American-style short legs)

Applies only to short option legs on American-style, physically settled options (equity and ETF options). Does not apply to long options, stock, or European-style cash-settled index options (SPX, XSP, NDX, RUT). Look up exercise style per instrument; never assume.

**Data needed:** ex-dividend date and dividend amount per underlying, refreshed daily from a reliable source. Missing dividend data for a candidate with a short American leg = skip that candidate (fail closed).

**Assignment-risk test for a short call** (run per short leg):

```
extrinsic_S = price_S - max(0, spot - strike_S)
assign_risk = ex_div_date falls before planned exit AND dividend > extrinsic_S
```

For short puts, flag when deep in the money with near-zero extrinsic value close to expiry (early exercise of puts is driven by interest, not dividends).

**Before entry (selection layer):**

- Extend `event_flag` to include ex-dividend dates inside the hold window.
- For any candidate whose hold window crosses an ex-div date and whose short leg could plausibly reach `assign_risk` at the barrier prices, add an assignment-risk barrier: shorten expected hold to the day before ex-div and add the expected early-exit cost (Section 5 exit slippage) to NetEdge.
- Index substitutes (SPX/XSP/NDX) receive no assignment cost, so they naturally rank higher when the equivalent ETF spread carries this risk.

**After entry (daily check, fast layer worker or scheduled job):**

- Every trading day by 14:00 ET, check all open short American legs.
- If tomorrow is ex-dividend and `assign_risk` is true, close or roll the spread that afternoon via the normal complex-order walk, with the cap relaxed to natural if needed to finish before 15:45 ET.
- Log every trigger and its outcome.

**Barrier integration:** early assignment risk is a fourth barrier that overrides profit target, stop, and max hold.

## A2. Expiry and pin risk

- Never hold a spread with a short leg into the close on expiration day.
- On expiry day, close any spread whose short leg is within 2% of spot (configurable) by 15:00 ET; close all remaining short-leg positions by 15:30 ET unless both legs are clearly out of the money by more than 5%.
- Treat this as another override barrier.
- If assignment occurs anyway (detected via account positions), alert immediately and flatten the resulting stock position using a limit walk, not a market order.

## A3. Live barrier monitoring

- Profit, stop, assignment and expiry barriers are checked on the fast layer from streamed quotes, not once per slow cycle.
- Barrier hit = start the exit walk immediately using the exit side of the fill model.
- Use package mid for barrier checks, not last trade, to avoid false triggers on stale prints.
- Add a debounce: a barrier must hold for 2 consecutive quote updates (or 3 s) before triggering, except the stop barrier on a gap, which triggers immediately.

## A4. Buying power (PDT rule removed)

- The pattern day trader rule and its $25,000 minimum were eliminated effective June 4, 2026. Do not build PDT counting or day-trade limits.
- Intraday buying power is now based on real-time margin excess. Before every order (each child slice), check live buying power through the reserved account-call budget; if insufficient, skip and log.

## A5. Tables and logging additions

- Add `dividends` table: underlying, ex_div_date, amount, source, fetched_at (hot window 1 year; re-fetchable, so compact freely).
- Add `barrier_events` table: position id, barrier type (profit, stop, time, assignment, expiry), trigger time, mid at trigger, exit result. Keep permanently with trade records.
- Add assignment triggers, pin closes and buying-power skips to the daily metrics report.

## A6. Acceptance additions

- No short American leg is held through an ex-dividend date when `assign_risk` is true.
- No short leg held into the expiry-day close.
- Barrier exits start within 5 s of a confirmed trigger during market hours.
- Paper test: simulate an ex-div assignment scenario and an expiry pin scenario, confirm both close correctly.

## A7. Missing-variable discovery (inside the nightly fit)

- Do NOT create a new scheduled job, routine, or alerting service for this. Add it as a final step of the existing nightly fill-model fit, so nothing new can fail on its own schedule.
- Each night, compute residuals: realized vs predicted slippage, fill probability, and time-to-fill, per key.
- Test residuals against candidate dimensions not already in the model (day of week, minutes to close, days to ex-div, underlying price level, recent news flag, spread-width regime, broker latency).
- Write findings to a single overwritten report file/table `exec_discovery_report` (latest run only), e.g. "Friday 15:00-16:00 fills miss prediction by 2.1 cents, n=84, p=0.01; consider adding day-of-week."
- Report-only: no alerts, no auto-added variables, no code changes. Nothing enters the model until Mike approves it.
- If this step errors, log once and let the nightly fit complete normally; the discovery step must never block or fail the fit.
- Minimum sample before reporting a finding: 50 observations and p < 0.01, to avoid noise.

## A8. UI integration and strategy coverage (applies to the main prompt and this addendum)

- Everything built in the main execution-layer prompt and in this addendum must be integrated into the existing integrated UI wherever it is relevant to the user. Do not build a separate standalone page; extend the existing views, cards and panels in the current design system.
- Examples of where it belongs: execution status, fill quality and slippage on existing trade/position views; barrier state (profit, stop, time, assignment, expiry) on open positions; after-tax P/L alongside pre-tax; the discovery report surfaced where model diagnostics already live. Pure plumbing (governor, compaction jobs) needs no UI.
- Follow the standing UI law already in CLAUDE.md: nothing that pops up may cover other UI, panels stay draggable but never go off-page, layout is screen-size aware.
- Every existing strategy and every new strategy added later must route orders through the execution layer. No strategy may send orders directly to the broker once the layer reaches Phase 4.
- Add this as a standing rule to CLAUDE.md, as its own PR: **"Every new backend feature, strategy, or data output must be integrated into the existing integrated UI where it is relevant to the user, extending existing views rather than creating standalone pages, and every strategy must route orders through the execution layer."** This applies to all future work by any session or routine, not just this project.
- This also covers updates: any change to an existing feature, strategy, data layer or backend output must update its UI counterpart in the same PR, so the UI never falls out of sync with the system.
- Use the codebase's own understanding of the integrated UI (its existing components, layout, design system and conventions) to decide where and how each item fits. Inspect the current UI code before placing anything; do not invent a new pattern when an existing one fits.
- Include this update rule in the CLAUDE.md standing rule as well: **"Any update to existing functionality must update its UI counterpart in the same PR, using the existing integrated UI's components and conventions."**
- Before adding either standing rule, read CLAUDE.md. If an equivalent rule already exists, do not add a duplicate: merge any missing parts into the existing rule and note the change in the PR description.

## A9. Backend-to-UI audit and catch-up

- Merge the A8 standing rules into CLAUDE.md as the FIRST PR of this addendum, before any other work, so every session and routine follows them from that point on.
- Then run a one-time audit: list every backend feature, data layer, strategy, endpoint and model output that exists today, and mark each as surfaced in the integrated UI, partially surfaced, or missing. Include anything the daily routines built that never got a UI counterpart.
- Write the audit to `UI_COVERAGE.md` in the repo: feature, backend location, UI status, proposed placement in the existing UI, and whether it is user-relevant or pure plumbing.
- No approval stop for the audit: proceed straight to integrating the user-relevant gaps.
- Integrate the gaps in a few batched PRs (grouped by UI area, never one PR per item) using the existing UI components and conventions. These UI catch-up PRs use the normal CI-gated auto-merge, not manual review. Keep at most 3 of these PRs open at once; finish and merge each before opening the next, and finish any PR stale for more than 48 hours instead of abandoning it (see A10).
- Keep `UI_COVERAGE.md` current: every future PR that adds or changes a backend feature updates its row.

## A10. Finish what was started (platform-wide)

PRs have been opened, left unfinished and forgotten, even when the work was useful. From now on, finishing existing work comes before starting new work.

- Add as a standing rule in CLAUDE.md (merge if an equivalent exists): **"Before starting any new work, check open PRs. Finish, fix and merge open PRs first. Do not abandon work in progress."**
- One-time cleanup, right after the A8 rules PR: list every open PR with what it does, how far along it is, and whether it still applies to the current code. Save as `OPEN_PR_TRIAGE.md`.
- For each open PR: if the work is still useful, rebase it, fix CI, complete it and merge it. If it duplicates later work, fold its useful parts into the current code, then close it with a note saying where the work went. Close only PRs that are fully superseded or no longer apply, always with a reason.
- Nothing useful is thrown away: closing a PR without carrying its useful work forward is not allowed.
- Ongoing: any PR open more than 48 hours gets picked up and finished first by the next session or routine. Keep total open PRs under 10 across the repo; when at the cap, no new PRs are opened until existing ones merge.

## A11. Rule consolidation (keep the rulebook lean)

Too many overlapping rules create limitations that slow the system down. After adding the rules from both prompts, consolidate.

- Review CLAUDE.md, CI config (`ci.yml`), routine definitions, lint/config files, and any rule-like constants or guards in code.
- Find rules that are duplicated, overlap, conflict, or are outdated (for example, rules about PDT limits, superseded workflows, or limits no longer needed).
- Merge duplicates into one clear rule, resolve conflicts by checking each rule against the code as it exists today and keeping whichever matches how the system actually works now (newer is not automatically right; if neither fits the current code, remove both or rewrite one rule that does), and remove rules that no longer apply. Keep the rule text short.
- Do not remove safety rules: the kill switch, fail-closed data checks, rate-limit governor, conduct guardrails, risk caps, protected-data retention, and the no-auto-merge rule for execution-layer PRs.
- Write a short `RULES_CHANGELOG.md` listing each rule merged, changed or removed and why.
- Do this as one PR, through normal CI-gated auto-merge.
- Ongoing: any new rule must check for an existing equivalent first and extend it rather than adding a new one.
