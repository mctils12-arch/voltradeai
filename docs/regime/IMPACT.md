# IMPACT — change → affected components → expected effect → risk (Stage B map, 2026-10-28)

Written before the first Stage B code change, per the program's rule, and updated as each step ships (CHANGELOG.md records what actually happened). Component names and line anchors are those of `AUDIT.md` §2 (producers P1–P13, throttles T1–T15) at the baseline `v1.0.1168`; they drift, so search by function name. "Observed by" names the probe or file a reader uses to see the effect without owner cookies.

## 0. The five regime vocabularies today, and what happens to each under the flags

| Vocabulary (AUDIT ref) | Today | Under `REGIME_ENGINE` (shadow) | Under `REGIME_SIZING` | Under `REGIME_SELECTION` / `REGIME_OPTIONS` | Under `REGIME_LABELS` |
|---|---|---|---|---|---|
| A — `classify_regime_5level` (P1) → `get_adaptive_params` caps (T1–T3), floors (T8), tier caps, third leg, convexity (T15), DD-halt resume regimes | the live classifier | unchanged; the engine publishes `legacy_label` beside it for comparison | unchanged (caps and floors keep reading P1) | unchanged | unchanged |
| B — Markov `regime_label` (P3) → shadow labels, `regime_at_entry`, Node exit geometry (T13), `TierContext.regime` | pinned to NEUTRAL_BULL by the units bug | unchanged | unchanged | unchanged | replaced by the engine's `legacy_label` on shadow records, `regime_at_entry`, `TierContext.regime`; `markov_state`/`regime_score` ML features unchanged |
| C — spot-VIX `vix_regime` (P5) → `_regime_scalar` (T4) | 1.10→0.30 ladder; dead on the CSP path | unchanged | `_regime_scalar` → 1.0 on every path; `exposure_multiplier` applied once | unchanged | unchanged |
| D — stress index (P6) → stress multiplier (T5) | 1.0/0.75/0.5/0.0, clamps to the 1% floor | unchanged | multiplier → 1.0 | unchanged | unchanged |
| E — Tier-1 VXX `size_scalar` (T7) | computed, dropped before `select_contract` | unchanged | replaced by `exposure_multiplier` in the Tier-1 budget | ELEVATED/CRISIS/inversion/VRP guards act before it | unchanged |
| (none) — the new engine | — | `voltrade_regime_state.json` + history written once per scan; routes read it | the only regime scalar in sizing | the only regime input to the new guards | source of the legacy label |

Invariant across all flags: `risk_kill_switch.py` mechanisms, the Node −10% kill, the Python 18% halt, the tier −20%/−8% gates, the daily −3% checks, the drawdown scalar (T6, owned by the master plan's N13), `submit_options_order` and every order POST path are untouched.

## 1. Step 1 — `scripts/regime_backtest.py`

| Field | Content |
|---|---|
| Change | New research tool (index-level regime-layer harness) + its test + first committed run. |
| Affected components | `scripts/`, `research/regime_backtest_<date>.md`; reads `backtest_v2.fetch_bars` and its `.bt_cache`. No runtime path. |
| Expected effect | RESEARCH §5 becomes reproducible; the engine's exposure path is evaluated by the same code before any consumer exists. |
| Risk | None to trading. Network use: one bar fetch per symbol per day when the cache is cold (≈ 20 symbols). |
| Observed by | the committed markdown/JSON; `python3 scripts/regime_backtest.py --help`. |

## 2. Step 2 — `regime_engine.py` behind `VOLTRADE_REGIME_ENGINE` (shadow compute)

| Field | Content |
|---|---|
| Change | New module: `RegimeState` dataclass, pure `compute_state`, transition filter, input builder, persistence helpers, RPC wrapper `regime_state_rpc`; one line in `voltrade_daemon.py`'s method table; two paths in `storage_config.py`; one `BASE_CONFIG` comment/param block. |
| Affected components | `voltrade_daemon.py` (method table only — the subprocess fallback uses the same function), `storage_config.py`, `system_config.py` (additive keys). Nothing reads the state yet. |
| Expected effect | With the flag off: zero (the scan hook imports the pure-Python module and evaluates `engine_allowed()` only; no fetch, no write). With the flag on: one state computation per scan (cached per bar date), ~20 cached bar fetches per day, a few KB written per day. |
| Risk | Bar fetch latency on a cold cache inside the scan (bounded: the builder has a 20 s budget and falls back to the last persisted state with `confidence` reduced and a `stale` flag). Daemon RSS: the module holds ~20 × 2,500 floats. |
| Observed by | Step 3's route/probe; until then the state file in `DATA_DIR`. |

## 3. Step 3 — shadow run and observability

| Field | Content |
|---|---|
| Change | `_scan_market_inner` calls the engine once after the macro snapshot (guarded, never raises); history append; `server/regimeReport.ts` (pure); `/api/bot/regime` (owner) and diag probe `regime`; `checks.regime` on `/api/health` (non-gating); `REGIME-TRANSITION` audit line. |
| Affected components | `bot_engine.py` (one guarded block in `_scan_market_inner`, after the macro snapshot — shared with the master-plan session: edit kept to ≤ 15 lines), `server/bot.ts` (route + health block + diag case — last commit before the PR), `server/diag.ts` (whitelist entry). |
| Expected effect | A daily regime report exists; transitions are visible in the audit log; staleness is visible on health. No trading effect. |
| Risk | `/api/health` payload grows by one block (serving verdict unchanged — `healthGate.SERVING_CHECKS` not edited). Route sanitization must not leak keys (reuses `sanitizeDiag`). |
| Observed by | `GET /api/diag/regime` (token), `GET /api/health` → `checks.regime`, `GET /api/diag/audit?type=REGIME-TRANSITION`. |

## 4. Step 4 — stock-selection guards behind `VOLTRADE_REGIME_SELECTION` (shadow labels on with the engine; gates off)

| Guard | Affected code | Effect when logging only | Effect when gating | Risk | Observed by |
|---|---|---|---|---|---|
| knife (price < 75% of 252-day high, or new 20-day low) | `deep_score` → `_scan_market_inner` post-score gates | shadow record relabelled `rejected_regime_knife` | candidate blocked | blocks the deep-discount names the funnel favours today; fewer admissions in narrow tapes (intended) | `/api/diag/shadow` bucket win rates |
| dip (MR sub-score and "Buy the Dip" neutralized unless gating index > 200-day and composite CALM_BULL / CHOPPY+HEALTHY) | `deep_score` `:1122-1140`, `:1261`, `:1463` | score recomputed in shadow (`score_breakdown` carries both) | the live score loses up to ~10 points on dip names in bear states | changes which names clear `MIN_SCORE`; interacts with T3 (two thresholds) — logged in the breakdown | shadow `score_breakdown`, `rejected_regime_dip` |
| panic (no momentum longs in `panic_state`) | post-score gate | `rejected_regime_panic` | blocked | rare state (24-month return < 0 and top-quintile variance); zero effect in 2026 so far | shadow bucket |
| relative strength (3-month return vs gating index < −10 pp) | post-score gate | `rejected_regime_rs` | blocked only after a positive read | placeholder threshold | shadow bucket |
| breadth no-add (index within 3% of 52-week high and market breadth WEAK) | admission loop | `rejected_regime_breadth` | no net new stock exposure that scan | can idle the stock sleeve in narrow tapes (intended; the floor keeps exposure) | shadow bucket, `REGIME` report `breadth_state` |
| sector check | — | NOT BUILT (Step 4 shipped 2026-10-28): the ±8 sector-momentum adjustment stays; a sector-ETF-above-50-day flag has no evidence and no data path in `deep_score`, so it is neither gated nor logged | — | none | — |

Shipped 2026-10-28 (CHANGELOG step 4): the five guards above as `score_breakdown.regime_<guard>` labels whenever the engine is on; `regime_mr_delta` records the score points the dip guard would remove; gates only for guards named in `VOLTRADE_REGIME_SELECTION`.

Downstream chain (CLAUDE.md reasoning standard 1): fewer admissions → fewer stock slots used → lower sector concentration and fewer correlation blocks → realized stock exposure falls in bear/narrow tapes and is unchanged in CALM_BULL; options and floor sleeves unaffected.

## 5. Step 5 — options-sleeve guards behind `VOLTRADE_REGIME_OPTIONS` (shadow via `log_csp_gate_shadow`)

| Guard | Affected code | Effect | Risk | Observed by |
|---|---|---|---|---|
| CRISIS → no new short puts | `tier1_csp_core`; scanner call site in `_scan_market_inner` (sell-side setups only) | no new CSP/scanner short-vol actions while `vol_state == CRISIS` (ON with ANY value of the flag) | the sleeve stops writing exactly when premium is richest and the loss state is most likely (intended per PUT 2008 / I&N worst decile) | `would_reject_regime_crisis` (shadow) / `rejected_regime_crisis` (gated) |
| VRP < 0 → skip new short vol (ON with any value of the flag) | same | skip | rare; coincides with crashes | `would_reject_regime_vrp` / `rejected_regime_vrp` |
| `VIX/VIX3M > 1` → notional × 0.5 | `tier1_csp_core` `size_scalar` (affordability cap + `size_pct`) | halves the budget; the contract count is `select_contract`'s until N8 (AUDIT T7 — same limitation as the VXX ladder) | fewer/cheaper names in stress | metadata `regime_size_mult`, report `stress_flags` |
| ELEVATED → contracts × 0.5, `target_delta` 0.15 | `size_scalar` + metadata | halves the budget; `target_delta` 0.15 in the metadata (honoured by `select_contract` only once N8 threads trade context through) | until N8: metadata + affordability only (honest) | metadata `regime_guards`, report |
| earnings inside DTE+2 → skip | `tier1_csp_core` using `options_scanner._fetch_earnings_calendar_with_status` (15-min cache) | hard skip (today: soft score) | fewer candidates near earnings; calendar outage fails OPEN with `earnings_lookup_ok: false` on the record (never blocks on missing data) | `would_reject_regime_earnings` / `rejected_regime_earnings` |
| LOW vol → annualized premium/collateral ≥ 6% | NOT BUILT (Step 5 shipped 2026-10-28): no premium at hand where Tier-1 decides (the chain is fetched inside `select_contract`); waits for N8 | — | — | — |
| IV−HV ranking after a ≤ 2% quoted-spread filter | NOT BUILT (same reason: no IV or quote at the decision point); waits for N8 | — | — | — |
| EBP cap (× 0.5 when `ebp > 0`) — applies to EVERY CSP (no market-cap field exists on the selection path, AUDIT §4) | `size_scalar` | halves when the Fed series is stressed | monthly series; two documented false positives | metadata `regime_guards.flags.ebp` |

Shipped 2026-10-28 (CHANGELOG step 5): shadow records are one per name, guard and day (`regime_engine.shadow_once_per_day`), bounded to 3 per guard per scan — the same 3 names a scan returns.

Downstream chain: fewer/lower CSPs in stress → lower options collateral → lower whole-portfolio exposure → fewer `master_kill_switch` exposure kills (KB #20b) → fewer `rejected_masterkill` shadow records. In calm states: unchanged.

## 6. Step 6 — one sizing path behind `VOLTRADE_REGIME_SIZING`

| Field | Content |
|---|---|
| Change | `calculate_position`, the `instrument_selector` and `options_execution` options sizers, and the Tier-1 budget read `exposure_multiplier` once; `_regime_scalar` and the stress multiplier return 1.0 under the flag; Tier-1's VXX ladder is replaced. |
| Combined regime-related scalar per composite state (before → after) | CALM_BULL: 1.10×1.0×1.0 → 1.0. VOLATILE_BULL: 0.80–0.65 × 0.75 (stress elevated) → 0.75. CHOPPY: 1.0 × 1.0 → 0.75–1.0 (by votes). STRESSED: 0.65–0.50 × 0.5 → 0.5 × vol_dial. BEAR: 0.50 × 0.5 → 0.25–0.5. CRISIS: 0.30 × 0.0 (clamped to the 1% floor) → 0 for new risk (no new entries), `core_multiplier` ≥ 0.5 for the index sleeve. Exact numbers are recorded per state in CHANGELOG.md from the first live week. |
| Risk | A stale state (> 2 market days) must not cut silently: the multiplier falls back to 1.0 with a `REGIME-STALE` audit line. Double application is prevented by a test that the product of regime scalars equals the multiplier. |
| Observed by | `/api/diag/regime` (applied-multiplier counts), fills tracker position sizes, `/api/diag/sleeves`. |

## 7. Step 7 — legacy-label subordination behind `VOLTRADE_REGIME_LABELS`

| Field | Content |
|---|---|
| Change | `_regime_ctx["regime_label"]` ← engine `legacy_label` on the three consumers (shadow record, `regime_at_entry`, `TierContext.regime`). |
| Effect | Node exit geometry (scale-out 2 at 3R/2R/1.5R, trail 3×/2×/1.5× ATR) follows the tape instead of a constant; shadow label distribution becomes informative. |
| Risk | Exit geometry changes are KB #48 territory; skipped if the exit-model consolidation lands first. |
| Observed by | shadow `regime_label` distribution; audit `SCALE-OUT`/`TRAIL` reasons. |

## 8. Not changed (handed off) — see FINDINGS.md Step 8

Floor schedule (`SPY_FLOOR_*`, `DEFENSIVE_FLOOR_*`), drawdown halt levels, convexity flag, kill switches, order paths, exit model, ML features. Each is filed with the harness evidence rather than changed.
