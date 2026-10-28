# RUNBOOK — regime & volatility engine (operate, read, roll back)

For the operator and for later sessions. Program docs: `FINDINGS.md` (plan), `IMPACT.md` (what each flag touches), `CHANGELOG.md` (what shipped), `BASELINE.md` (the frozen reference every read compares against).

## 1. Flags (all default ON since 2026-10-28; each is independent)

**2026-10-28 human directive (paper account):** every flag below defaults ON in code — `system_config.BASE_CONFIG` `REGIME_ENGINE_ENABLED=True`, `REGIME_SELECTION_DEFAULT="all"`, `REGIME_OPTIONS_DEFAULT="all"`, `REGIME_SIZING_DEFAULT="all"`, `REGIME_LABELS_DEFAULT=True` (prior: all off) — skipping the §6 shadow-only wait. The default is read ONLY when the env var is ABSENT; an env var that is present always wins, including `off` / `0` / an empty string (`regime_engine.flag_raw`; Node mirror `server/regimeReport.ts` `REGIME_FLAG_DEFAULTS`, pinned against BASE_CONFIG by `test_regime_flag_defaults.py`). So "unset the flag" in the rollback column below now means **set the env var to `off`** on Railway (no deploy); reverting the BASE_CONFIG block rolls every default back with a deploy. `/api/bot/regime` → `flags` shows each flag's effective value and whether it came from `env` or `config`. No risk limit moved: drawdown ladders, kill switches, halts and caps are untouched.

| Flag | Where it is read | What ON does | Rollback |
|---|---|---|---|
| `VOLTRADE_REGIME_ENGINE` (env `1/true/on`) or `BASE_CONFIG["REGIME_ENGINE_ENABLED"]` | `regime_engine.engine_allowed`; the scan hook in `bot_engine._scan_market_inner` | SHADOW: one `RegimeState` per scan, persisted and served; no consumer acts | unset the env var (no deploy needed; the next scan skips the engine); the state file is inert |
| `VOLTRADE_REGIME_SELECTION` | stock-selection guards (step 4) | promotes the named guards from shadow labels to gates (`knife,dip,panic,rs,breadth`) | unset → scores identical to baseline |
| `VOLTRADE_REGIME_OPTIONS` | Tier-1 CSP guards + the scanner's sell-side setups (step 5) | names the guards promoted to gates (`crisis,vrp,inversion,elevated,earnings,ebp` or `all`); ANY value turns on `crisis` and `vrp` (blocks); `inversion`/`elevated`/`ebp` halve `size_scalar` (contract count stays `select_contract`'s until N8); `earnings` skips the name. Not built: low-return floor, IV−HV ordering (no chain data at the decision point). Shadow buckets `would_reject_regime_<guard>` always on while the engine runs | unset (labels stay) |
| `VOLTRADE_REGIME_SIZING` | `position_sizing.calculate_position`, both options sizers, Tier-1 budget (step 6) | names the sleeves sized by the engine (`stock,options,tier1` or `all`): `exposure_multiplier` replaces the VIX-level scalar + stress multiplier (stock; 0 → the entry is blocked) and the Tier-1 VXX ladder; options sleeves floored at 0.5 (`csp`) / 0.25 (`csp_reduced`, CRISIS); drawdown ladders stay; stale state → 1.0 + counted + `REGIME-STALE` | unset (next call sizes the legacy way) |
| `VOLTRADE_REGIME_LABELS` (env `1/true/on/yes`) | `deep_score` (one swap after the Markov call; step 7) | the engine's `legacy_label` (BULL/NEUTRAL/CAUTION/BEAR/PANIC) replaces the Markov label on shadow records (`regime_label`, old label kept as `regime_label_markov`), `regime_at_entry`, `TierContext.regime`; only while the state is fresh (stale → Markov label stands). The Node exit geometry is NOT wired (never was — filed) | unset |

Nothing here edits a kill switch, a drawdown halt, the floor schedule, or an order path.

## 2. Files and routes

- State: `<DATA_DIR>/voltrade_regime_state.json` (latest), `<DATA_DIR>/voltrade_regime_history.jsonl` (one row per trading day or transition, append-only), `<DATA_DIR>/voltrade_regime_sizing.json` (today's sizing-multiplier counters, rewritten once per scan while a sleeve is flagged). `DATA_DIR` = `/data/voltrade` on Railway, `/tmp` locally.
- Counterfactual (2026-10-28 live flip): `<DATA_DIR>/voltrade_regime_counterfactual.jsonl` — one row per decision while the engine is on: `kind` (selection | options | sizing | label), what LEGACY would have done at that decision point (no regime gate = take; the legacy regime scalars a sizer would have applied, pre-clamp; the Markov label) next to what the ENGINE did (`applied: false` = that consumer's flag is off, the row carries the engine's would-be value), `differs`, `shadow_decision` (the `rejected_regime_*` / `would_reject_regime_*` bucket — join on ticker + date for the shadow outcomes), the state snapshot (`as_of`, composite, exposure) and the effective flags. Deduped per (kind, key, source, content) per UTC day, ≤ 5000 rows/day per process, rotated to `.1` above 32 MB. Read it aggregated: `GET /api/diag/regime_counterfactual?days=14` (per day per kind: counts, differs, mean legacy vs engine multiplier per sleeve, blocks by guard, label pairs; no tickers).
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
- A stale state never cuts size silently: sizing falls back to 1.0, counts the fallback (`sizing.stale_fallbacks` on `/api/diag/regime`, a `checks.regime` warning the same day) and audits `REGIME-STALE` once per new occurrence (step 6).

## 6. Flag-flip order (the program's own recommendation; every flag was OFF at close)

2026-10-28: the human directive applied steps 1, 3, 4 and 5 at once with every guard and sleeve named (`all`), skipping the fourteen-day shadow wait and the one-at-a-time promotion; the reads below (2026-11-16 onward) now judge live gates instead of shadow buckets, and the opposite-sign rule turns a single guard off by setting its env var to the remaining guards (e.g. `VOLTRADE_REGIME_SELECTION=knife,panic,rs,breadth`) or to `off`.

1. `VOLTRADE_REGIME_ENGINE=1` — shadow only: one state per scan, the report, the history. Wait fourteen trading days with `checks.regime.status` ok and no gap in `history_days`.
2. Read 2026-11-16 (FINDINGS step 9): per-guard buckets `would_reject_regime_*` on `/api/diag/shadow` (+5/+20 d vs the taken bucket, n ≥ 30, ≥ 3 pp at +20 d to promote), the sleeve hold-time ledger, the engine's history vs the tool's replay (HONESTY METRIC).
3. Promote one guard at a time: `VOLTRADE_REGIME_SELECTION=<guard>` / `VOLTRADE_REGIME_OPTIONS=<guard>` (any value of the options flag also turns on `crisis` and `vrp`).
4. `VOLTRADE_REGIME_SIZING=<sleeve>` one sleeve at a time; 30 trading days with no sizing anomaly (`sizing.stale_fallbacks` stays 0 on trading days, no position above the absolute caps) before the next sleeve.
5. `VOLTRADE_REGIME_LABELS=1` whenever the shadow records' `regime_label` vs `regime_label_markov` read says the engine's label is the better split — it changes labels, never a trade.
Rollback at any step: unset the flag (RUNBOOK §5); the opposite sign at a read with n ≥ 30 turns a guard off and is logged.
