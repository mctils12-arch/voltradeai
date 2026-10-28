# RUNBOOK — regime & volatility engine (operate, read, roll back)

For the operator and for later sessions. Program docs: `FINDINGS.md` (plan), `IMPACT.md` (what each flag touches), `CHANGELOG.md` (what shipped), `BASELINE.md` (the frozen reference every read compares against).

## 1. Flags (all default OFF; each is independent)

| Flag | Where it is read | What ON does | Rollback |
|---|---|---|---|
| `VOLTRADE_REGIME_ENGINE` (env `1/true/on`) or `BASE_CONFIG["REGIME_ENGINE_ENABLED"]` | `regime_engine.engine_allowed`; the scan hook in `bot_engine._scan_market_inner` | SHADOW: one `RegimeState` per scan, persisted and served; no consumer acts | unset the env var (no deploy needed; the next scan skips the engine); the state file is inert |
| `VOLTRADE_REGIME_SELECTION` | stock-selection guards (step 4) | promotes the named guards from shadow labels to gates (`knife,dip,panic,rs,breadth`) | unset → scores identical to baseline |
| `VOLTRADE_REGIME_OPTIONS` | Tier-1 CSP guards (step 5) | CRISIS / VRP-negative blocks, inversion and ELEVATED size cuts, earnings skip, low-return floor, IV−HV ordering | unset |
| `VOLTRADE_REGIME_SIZING` | `position_sizing`, options sizers, Tier-1 budget (step 6) | one regime multiplier; VIX-level scalar and stress multiplier → 1.0 | unset |
| `VOLTRADE_REGIME_LABELS` | `deep_score` label consumers (step 7) | engine `legacy_label` replaces the Markov label on shadow records / `regime_at_entry` / `TierContext.regime` | unset |

Nothing here edits a kill switch, a drawdown halt, the floor schedule, or an order path.

## 2. Files and routes

- State: `<DATA_DIR>/voltrade_regime_state.json` (latest), `<DATA_DIR>/voltrade_regime_history.jsonl` (one row per trading day or transition, append-only). `DATA_DIR` = `/data/voltrade` on Railway, `/tmp` locally.
- Input cache: `<CACHE_DIR>/regime_bars_SPY.json` (closes, refreshed once per UTC day), `<CACHE_DIR>/regime_ebp.json` (Fed excess bond premium, 7-day cache). `CACHE_DIR` = `/tmp/voltrade_alt_cache`.
- Read it: owner `GET /api/bot/regime`; sessions `GET /api/diag/regime` with `-H "x-diag-token: $DIAG_TOKEN"` (never print the token). Same body: `health`, `state`, `history` (last 60 rows), `transitions`, `time_at_levels`.
- Health: `GET /api/health` → `checks.regime` = `off` / `ok` / `warning` (stale > 2 market days, last update failed, flip alarm). It never gates serving (`healthGate.ts` unchanged).
- Audit: `GET /api/diag/audit?type=REGIME-TRANSITION` (one line per composite change), `type=REGIME-STALE` (an update failed; the engine keeps serving the last state with `stale: true`).
- Daemon: method `regime_state` (`{"gating":"SPY","force":false,"compute":true}`); CLI `VOLTRADE_REGIME_ENGINE=1 python3 regime_engine.py SPY` prints the state.
- Research tool: `python3 scripts/regime_backtest.py --sweep --out research/regime_backtest_<date>.md --json research/regime_backtest_<date>.json` (index-level; the engine's own exposure path appears as the `engine_*` arms). Not a strategy backtest.

## 3. Reading the state (what each field means)

- `composite_regime` — CALM_BULL / VOLATILE_BULL / CHOPPY / STRESSED / BEAR / CRISIS (6-day majority filter); `legacy_label` is the BULL/NEUTRAL/CAUTION/BEAR/PANIC equivalent for old consumers.
- `trend_state` — the primary regime from the slow 120–240-day ensemble (filtered; `trend_state_raw` is the unfiltered vote). `fast_state` (22/65-day) only downgrades exposure.
- `vol_state` — LOW / NORMAL / ELEVATED / CRISIS from the EWMA vol percentile of the gating index's own history, with hysteresis; `stress_flags.vix_vix3m_inverted` is the crisis trigger.
- `exposure_multiplier` — {0, .25, .5, .75, 1} for NEW discretionary risk (stock entries, new short-put notional). `core_multiplier` ≥ 0.5 for the index floor sleeve (never fully out). `vol_dial` is the continuous realized-vol factor behind both.
- `allowed_strategies` — what may OPEN in this state (`dip_buy` only in CALM_BULL/CHOPPY with HEALTHY breadth; nothing in CRISIS; no `momentum_long` in the panic state).
- `confidence` — share of the filter window agreeing with the composite × data quality (0.7 when the VIX family is missing, 0.5 when the bars are stale).
- `days_in_regime`, `pending_transition` (candidate, votes, needed), `flips_12m` (primary flips; `alarm` above 6), `composite_changes_12m`.
- `stress_flags` — VIX, VIX/VIX3M, VRP (vol points), 1-year VIX percentile, VVIX, realized and EWMA vol, own-history percentile, σ_target. `ebp` — latest Fed excess bond premium row.
- `inputs.sources` — provider per series (`alpaca` / `yahoo` / `missing`).

## 4. Daily / weekly checks

1. `checks.regime.status` on `/api/health` is `ok` (engine on) and `age_market_days ≤ 1` on a trading day.
2. `/api/diag/regime` → `history_days` grows by one per trading day; `transitions` are few (budget: 0.5–2 primary flips a year; alarm above 6).
3. On a `REGIME-TRANSITION` audit line: confirm the inputs make sense (the report carries them) — a transition driven by `sources: missing` or `stale: true` is a data problem, not a market one.
4. Reads against `BASELINE.md`: 2026-11-16, 2026-12-15, 2027-01-27 (FINDINGS.md step 9) — per-guard `rejected_regime_*` shadow buckets on `/api/diag/shadow`, sleeve hold-time ledger on `/api/diag/sleeves`, rolling return vs SPY on `/api/diag/benchmark`, and the engine's own history vs the tool's replay of the same dates (HONESTY METRIC).

## 5. Rollback

- Any consumer: unset its flag — the baseline path is bit-identical (each step's tests pin this).
- The engine: unset `VOLTRADE_REGIME_ENGINE`; delete nothing. If the state file is corrupt, remove `voltrade_regime_state.json` (the history is append-only and stays); the next scan rebuilds it.
- A stale state never cuts size silently: sizing falls back to 1.0 with a `REGIME-STALE` audit line (step 6).
