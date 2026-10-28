# CHANGELOG — regime & volatility engine program (Stage B)

One entry per shipped change: what, why (with the evidence), measured effect, rollback. Steps refer to `FINDINGS.md`; component effects are mapped in `IMPACT.md`; the frozen reference is `BASELINE.md`. Versions are `package.json` `code_version`s, so live results attribute to the change.

## Stage A — 2026-10-28, PR #1484 (docs only, no version)
- **What:** `BASELINE.md`, `AUDIT.md`, `RESEARCH.md`, `FINDINGS.md`, `IMPACT.md`.
- **Why:** the program requires the audit, the research and the plan before any code change.
- **Measured effect:** none (docs).
- **Rollback:** n/a.

## Step 1 — 2026-10-28, PR #1485, v1.0.1176 — `scripts/regime_backtest.py`
- **What:** the isolated, index-level regime-layer tool (+20 tests) and its first committed run `research/regime_backtest_2026-10-28.md`.
- **Why:** RESEARCH.md §5 has to be reproducible from the repo; the engine's exposure path is evaluated by the same code before any consumer exists.
- **Evidence / measured effect:** SPY 2007-08-28 → 2026-10-26, lag 2, 5 bp: buy-and-hold 11.05% / −55.2% DD; current exposure-schedule proxy 4.30% / −20.2% with 22.9 flips/yr and 39% of sessions fully out; Faber monthly 7.93% / −27.7% / 1.6 flips; never-fully-out 8.64% vs fully-out 7.05%. One Stage A prior was wrong and is recorded: the loose vol dial (20% target) is CAGR-neutral while cutting max DD to −40%. All five pre-registered claims hold under 34 ±20%/lag/cost variants.
- **Rollback:** none needed (no runtime path).

## Step 2 — 2026-10-28 — `regime_engine.py` behind `VOLTRADE_REGIME_ENGINE` (shadow)
- **What:** the single owner of regime state (`RegimeState`: primary trend from the slow 120–240-day ensemble with ±1% bands and a 6-day majority filter; fast 22/65-day legs that only downgrade exposure; EWMA-vol state with hysteresis and a `VIX/VIX3M` crisis flag; basket breadth; Daniel–Moskowitz panic state; composite + legacy label; quarter-stepped `exposure_multiplier` with fast-out / glide-in; `core_multiplier ≥ 0.5`; allowed strategies; confidence; dwell; flip budget with alarm; stress flags; Fed EBP), the daily input builder (Alpaca batch → Yahoo fallback, cached per day), persistence (`voltrade_regime_state.json` + append-only `voltrade_regime_history.jsonl`), the daemon method `regime_state`, the config mirror `REGIME_ENGINE_ENABLED` (off), and the engine as `engine_*` arms in the backtest tool (one implementation). 33 tests.
- **Why:** AUDIT §2 found five regime vocabularies and no owner; RESEARCH §6 ranks the inputs this state is built from; FINDINGS step 2.
- **Evidence / measured effect (tool, SPY 2007–2026, same window as Step 1):** `engine_quantized` (what consumers will read) 8.34% CAGR / −20.5% max DD / Sharpe 0.76, 9.4 exposure-level changes/yr, 89% time in market; episodes: 2008 DD −17.1%, 2020 −15.2%, 2022 −11.0%, 2015–16 −7.5% CAGR, late 2018 −17.0% annualized; `engine_core` (the never-fully-out multiplier for the index sleeve) 9.26% / −36.7% / 0.72. Primary (slow-trend) flips 2.7/yr after the filter (4.7 raw; research budget 0.5–2, alarm 6); composite changes 3.2/yr. Pre-registered flip ablation: widening the band to 2–3% cut raw flips to 3.1–2.2/yr at −0.3 pp CAGR — rejected (three variants tried, documented value kept). With the flag off nothing runs; with it on, one computation per scan (7.6 s cold on the Yahoo fallback, 0 s on the daily cache) and no consumer.
- **Rollback:** unset `VOLTRADE_REGIME_ENGINE` (and keep `REGIME_ENGINE_ENABLED` false); the state file is inert.

## Step 3 — 2026-10-28 — shadow run and observability
- **What:** the scan hook (`bot_engine._scan_market_inner`: one guarded `regime_engine.update_state` per scan after the macro snapshot, only when the engine flag is on; the state rides the scan result as `regime`), `server/regimeReport.ts` (pure shaping: readers, health block, report body, transition audit text) + 6 tests, the owner route `GET /api/bot/regime`, the diag probe `regime` (same body), the NON-GATING `checks.regime` block on `/api/health`, and the `REGIME-TRANSITION` / `REGIME-STALE` audit lines; `RUNBOOK.md`; `test_regime_scan_hook.py` (AST ratchet: one guarded hook, after the macro snapshot, result key present).
- **Why:** FINDINGS step 3 — fourteen days of persisted history must exist before any consumer flag is turned on; a stale state must be loud, never silent (RENDERING & MOTION Law V's "silent degradation" lesson applied to the bot).
- **Measured effect:** none on trading. Flag off: the hook imports the module (pure Python, no I/O) and evaluates `engine_allowed()` only — no bars are fetched and no state is written; the health block reads `off`. Flag on: one state per scan, ~1 KB/day of history, a report at `/api/diag/regime`.
- **Rollback:** unset `VOLTRADE_REGIME_ENGINE`; the route and health block then report `off` with the last persisted state.
