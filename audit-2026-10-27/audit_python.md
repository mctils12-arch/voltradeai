# Python-side audit — voltradeai (read-only, 2026-10-27)

Method: stdlib `ast` import graph over every `.py` outside node_modules (447 files, 499 repo-local
edges, 0 parse errors, 0 unresolved repo-local imports), BFS reachability from the runtime roots
below, plus regex sweeps of `server/*.ts`, `scripts/*.sh`, CI, Dockerfile and `research/*.md`.
Scripts: `scratchpad/import_graph.py`, `scratchpad/classify.py`; raw data `graph.json`, `classified.json`.

Inventory: root 292 `.py` (68 non-test modules + 220 `test_*.py` + 3 `*_test.py` + `conftest.py`),
`strategies/` 4, `alphadesk/alphadesk/` 17, `scripts/` 134 (133 tooling + `scripts/jodi_gate2_test.py`).
**`datacore/` contains no Python at all** (it is TS + data); the "Python under datacore/" premise is false.

---

## 1. Entry points (with evidence)

| Root | Evidence |
|---|---|
| `voltrade_daemon.py` | `run_with_daemon.sh` → `python3 voltrade_daemon.py` (Railway `startCommand: ./run_with_daemon.sh`, railway.json/toml). Static imports: stdlib + numpy/pandas/requests + `storage_config`. Lazy `__import__` via `_lazy_import` (daemon:243-253) for the method table (daemon:264-332) and warm-up (daemon:753-757). |
| Daemon RPC method table → module | `track_fill`→`ml_model_v2`; `check_halt`→`position_sizing`; `select_contract`/`submit_options_order`/`evaluate_and_execute`→`options_execution`; `risk_status`/`get_peak_equity`/`check_position_risk`/`check_correlation_pre_trade`→`risk_kill_switch`; `paper_resume_sync`→`paper_resume_sync`; `macro_snapshot`→`macro_data`; `shadow_stats`/`shadow_backfill_exit_aligned`/`shadow_update_decisions`→`shadow_portfolio`; `csp_gate_shadow`→`tiered_strategy`; `run_full_scan`→`bot_engine.scan_market`; `lookup_ticker`→`lookup_ticker`; `lookup_ledger_report`→`lookup_ledger`; `health`/`ping`/`cache_inventory`→local. **`ml_status`→`"ml_status_impl"` and `ml_toggle`→`"ml_toggle_impl"` name modules that do not exist**, so `_lazy_import` always returns None and the daemon always runs its inline `_ml_status_fallback`/`_ml_toggle_fallback` (daemon:371-379, 518-575). |
| Daemon startup subprocess | `voltrade_daemon.py:733-736` runs `seed_feedback_from_backtest.py` when `trade_feedback.json` is missing/sparse (`VOLTRADE_AUTOSEED`). |
| Node `pythonCall(method, args, subprocessCmd)` (bot.ts:162) | daemon-first, `execPythonSerialized` subprocess fallback (heavy `run_full_scan` is daemon-only). 19 call sites; fallbacks import: `ml_model_v2.track_fill` (×5), `lookup_ticker`, `lookup_ledger`, `risk_kill_switch`, `position_sizing`, `options_execution`, `tiered_strategy`, `shadow_portfolio`, `paper_resume_sync`, `python3 ml_status.py`, `python3 bot_engine.py full`. |
| Node direct `execPythonSerialized("python3 -c …")` (bot.ts, ~44 sites) | imports `storage_config`, `system_config`, `diagnostics` (check_model_health/get_auto_fix_params/check_weekly_loss/run_diagnostics), `stress_index`, `probability_engine`, `csp_universe.get_universe_snapshot`, `options_manager` (register_options_entry/merge_options_meta), `intraday_shorts.close_open_shorts`, `shadow_portfolio.backfill_outcomes/backfill_exit_aligned`, `lookup_ledger.backfill`, `manipulation_detect.scan_for_manipulation`, `cftc_cot.run_daily_update`, `sec_form4_bulk.run_update`, `ml_insight.ml_insight`, `feedback_boot_cleanup.clean_feedback`, `macro_data`, `analyze`. |
| Node script spawns | bot.ts: `python3 analyze.py <t> --mode=scan` (3924), `python3 backtest.py` (4010, 6695), `python3 bot_engine.py full|scan` (4812, 6568, 6732), `python3 ml_retrain_safe.py` (6280, 6763), `python3 backup_to_github.py` (6801), `python3 ml_status.py` (3718). routes.ts: `analyze.py` (280, 675), `insights.py` (713), `etf_analyzer.py` (6634), `finnhub_data.py --raw insider` (6680), `ml_diagnostics.py` (6727), `system_status.py` (6783), `python3 ml_toggle.py <action>` (7362), `python3 -c "from intraday_shorts import get_dashboard_data"` (7202), `python3 -m alphadesk …` with `cwd=alphadesk/` (774, 6584, 6610). |
| Docker image scope | `Dockerfile` copies only `*.py` (root), `strategies/`, `alphadesk/`, `backtest_10yr_results.json`. **`scripts/` is never shipped**; every `scripts/*.py` mention in non-test `server/*.ts` is a comment/provenance string (apiProduct.ts, jodiOil.ts, railTraffic.ts, unComtrade.ts, cdcCancer.ts, qualityDashboard.ts) — nothing spawns them at runtime. Consistent. |
| CI | `.github/workflows/ci.yml` job `python-tests`: `pip install -r requirements.txt && pip install pytest`, `python -m compileall -q .`, then pytest on only 4 files (`test_risk_controls.py test_audit_critical.py test_diagnostic_false_positives.py test_patches_verification.py`). Job `test`: installs requirements + requirements-dev, runs `scripts/gated_tests.sh` → full `python3 -m pytest -q` (conftest `collect_ignore` drops `test_auto_discovery.py`, `test_full_system.py`) + deploy-gate smoke. Both block automerge. |
| Dev-only shell → Python | `scripts/program_status.sh` runs `scripts/design_token_drift.py`, `hardcoded_palette_hex.py`, `param_shadows_outer_binding.py`, `duplicate_route_registration.py`; `counter_ratchet.sh`/`gated_tests.sh` only run inline heredocs. |
| In-Python schedulers | None. All cadence is bot.ts tier timers; the daemon has only a memory-watchdog thread. (`while True` hits in `sec_filings_text.py` are pagination loops.) |

Aside: bot.ts:2541-2546 (`/api/system/verify-change`) `fs.readFileSync`s a fixed list of .py files as a text haystack — not an execution path.

## 2. Reachability summary

| Set | Count |
|---|---|
| Non-test, non-scripts modules (root 68 + strategies 4 + alphadesk 17) | 89 |
| ACTIVE (reachable from a runtime root) | **75** (root 54, strategies 4, alphadesk 17) |
| DEAD — no importer, no runtime/CI/tooling reference | 1 |
| DEAD at runtime — imported only by tests (research/one-off CLIs) | 8 |
| UNCLEAR | 3 |
| TOOLING-ONLY (root module used only by scripts/) | 2 |
| `scripts/` (133 non-test) | TOOLING 123 (tested / invoked / imported), DOC-REF-ONLY 9, ORPHAN 1 — none runtime |
| Test files | 225 (220 `test_*.py` + 3 root `*_test.py` + `scripts/jodi_gate2_test.py` + `conftest.py`) |

## 3. ACTIVE table (module | why / chain)

Shortest chain from a root; "fail-soft" = imported inside `try/except` at call time, so absence degrades silently.

| Module | Lines | Chain |
|---|---|---|
| voltrade_daemon.py | 784 | run_with_daemon.sh |
| bot_engine.py | 5929 | daemon `run_full_scan` → `bot_engine.scan_market`; subprocess `python3 bot_engine.py full` |
| ml_model_v2.py | 2651 | daemon `track_fill`; warm-up; bot_engine → ml_model_v2 |
| options_execution.py | 2704 | daemon `select_contract`/`submit_options_order`/`evaluate_and_execute` |
| options_scanner.py | 1926 | ml_model_v2 → options_scanner (also bot_engine, csp_universe) |
| options_manager.py | 1437 | bot.ts -c `from options_manager import register_options_entry`; options_execution → options_manager |
| vol_surface.py | 2016 | options_execution → vol_surface |
| csp_universe.py | 923 | bot.ts -c `get_universe_snapshot`; tiered_strategy → csp_universe |
| tiered_strategy.py | 1108 | daemon `csp_gate_shadow`; bot_engine → tiered_strategy |
| position_sizing.py | 1452 | daemon `check_halt`; warm-up |
| risk_kill_switch.py | 774 | daemon `risk_status` etc.; bot.ts -c |
| system_config.py | 640 | bot.ts -c `from system_config import BASE_CONFIG`; bot_engine → system_config |
| storage_config.py | 39 | daemon static import; bot.ts -c |
| shadow_portfolio.py | 2205 | daemon `shadow_*`; bot.ts -c backfill_* |
| paper_resume_sync.py | 170 | daemon `paper_resume_sync` |
| macro_data.py | 521 | daemon `macro_snapshot`; position_sizing → macro_data |
| lookup_ticker.py / lookup_ledger.py | 326 / 309 | daemon `lookup_ticker`/`lookup_ledger_report`; bot.ts -c |
| alpaca_feed.py / alpaca_rate_limiter.py | 246 / 172 | ml_model_v2 → alpaca_feed (18 importers) |
| market_calendar.py | 145 | options_scanner → market_calendar |
| regime_util.py / regime_inputs.py / markov_regime.py | 183 / 81 / 234 | ml_model_v2 → regime_util; bot_engine → regime_inputs; bot_engine → markov_regime (fail-soft, bot_engine:131) |
| instrument_selector.py | 1588 | bot_engine → instrument_selector |
| leveraged_etf_filter.py | 186 | bot_engine → leveraged_etf_filter |
| stress_index.py / probability_engine.py | 235 / 273 | bot.ts:2415/2420 -c; position_sizing → stress_index |
| intraday_shorts.py | 530 | bot.ts:5087 `close_open_shorts`; routes.ts:7202 `get_dashboard_data` |
| diagnostics.py | 592 | bot.ts -c (4 functions) |
| manipulation_detect.py / cftc_cot.py / sec_form4_bulk.py | 198 / 280 / 307 | bot.ts Tier-3 -c one-liners (6351, 6428, 6449) |
| ml_insight.py / ml_retrain_safe.py / ml_status.py / ml_toggle.py / ml_diagnostics.py | 168/116/54/35/406 | bot.ts:8555 -c; `python3 ml_retrain_safe.py`; `python3 ml_status.py` (fallback only); routes.ts:7362 `python3 ml_toggle.py`; routes.ts:6727 |
| feedback_boot_cleanup.py | 69 | bot.ts:8437 -c `clean_feedback` |
| seed_feedback_from_backtest.py | 217 | daemon startup subprocess |
| analyze.py | 2664 | routes.ts/bot.ts `python3 analyze.py`; instrument_selector → analyze; backtest_v2 → analyze |
| backtest.py → backtest_v2.py → live_exit_policy.py | 63 / 728 / 289 | bot.ts `python3 backtest.py` |
| insights.py → institutional_data.py | 744 / 543 | routes.ts:713 `python3 insights.py`; institutional_data fail-soft (insights:151) |
| etf_analyzer.py → etf_data_sources.py | 585 / 948 | routes.ts:6634 |
| finnhub_data.py | 534 | routes.ts:6680 script; bot_engine:867 fail-soft |
| alt_data.py / social_data.py / intelligence.py | 571 / 473 / 657 | bot_engine:846-860 fail-soft `_fetch_alt/_fetch_social/_fetch_intel`; ml_model_v2:351 → intelligence. **Wired, not orphaned** — but every call is inside `except Exception: return {}`, so a broken module is invisible except via the `_diag` fetch trace. |
| backup_to_github.py / system_status.py | 167 / 279 | bot.ts:6801; routes.ts:6783 |
| strategies/{__init__,momentum,mean_reversion,squeeze}.py | 1/25/44/33 | bot_engine:140-155 `sys.path.insert(strategies/)` + bare `import momentum/mean_reversion/squeeze` (fail-soft); backtest_v2:315 `from strategies import momentum`. Note: two import spellings create two module objects (`momentum` vs `strategies.momentum`) in any process that loads both. |
| alphadesk/alphadesk/*.py (17 files, 2,968 lines) | — | routes.ts `python3 -m alphadesk` (cwd alphadesk/) → `__main__` → analysis, catalyst, config, edgar, engine, llm_reader, market, market_context, models, options, planner, providers, report, tax, tax_engine. stdlib + optional `anthropic`. |

## 4. DEAD table (module | lines | evidence | superseded-by)

| Module | Lines | Evidence | Superseded by / note |
|---|---|---|---|
| **ml_model.py** | **1149** | 0 importers anywhere (py/ts/sh); not in daemon table; not in CI; only mentions are comments ("retired") and a per-file count key in `test_silent_except_ratchet.py:68` (that test walks the tree, so deleting the file only lowers the baseline). Classified TOOLING-ONLY by the script purely because of that string. | `ml_model_v2.py` (shares `ml_score`/`track_fill`/`train_model`). **Largest dead file.** |
| sim_trading_day_v2.py | 334 | no importers (runtime or test); not in server/*.ts; not in daemon table; not in CI; `sys.path.insert(0,'/tmp/vt_test')` + no `__main__` guard (executes on import); 3 research-doc mentions only. | v1.0.13-era simulation; `test_full_system.py` / `scripts/session_health_check.py` cover the ground now. |
| cleanup_feedback.py | 90 | one-off ("Run on Railway: python3 cleanup_feedback.py"); importers: 1 test only. | `feedback_boot_cleanup.py` (active) + ml_model_v2 startup poisoned-record cleanup. |
| bot_backtest_subperiods.py | 121 | importers: 1 test only; research CLI (sub-period judgment of dual-momentum). | finding already filed in open_questions.md |
| cftc_tff_tlt_disjoint_replication.py | 99 | importers: 1 test only; ladder step-1 replication script. | — |
| cot_gate2_oos.py | 67 | importers: 1 test only. | — |
| regime_detector_compare.py | 277 | importers: 1 test only; [RESEARCH] one-shot comparing markov_regime vs regime_util. | — |
| sec_filings_classify.py / sec_filings_similarity.py / sec_filings_gate2.py | 113 / 336 / 309 | importers: tests only ("Lazy Prices PR 2b/2c/3"); not in server/*.ts; not in daemon table. | research pipeline, see UNCLEAR |

## 5. UNCLEAR table

| Module | Lines | Why unclear |
|---|---|---|
| sec_filings_text.py | **1331** | Root-level "Lazy Prices PR 1 of N" EDGAR archive CLI. Only importers are the dead-at-runtime `sec_filings_{classify,similarity,universe}` + 4 tests. Not shipped-useful (no runtime caller) but an in-progress, test-covered research pipeline; belongs in `scripts/` or datacore tooling rather than root. |
| sec_filings_universe.py | 157 | Same chain (PR 2a); importer is `sec_filings_classify` only. |
| bot_backtest.py | 239 | Standalone momentum backtest harness; only importer is dead `bot_backtest_subperiods.py`; named in `test_durability.py:24` allowlist ("offline .bt_cache"); 3 research mentions. Kept alive as a CLI, not wired anywhere. |
| gate2_stats.py (TOOLING-ONLY) | 107 | Shared Newey-West gate-2 core imported by `scripts/{crop_conditions,eia930,midas,rail_traffic}_gate2.py` and the root `*_gate2_test.py` CLIs. Alive as research library; not runtime. |

`scripts/` with no test, no shell/python caller, only research-doc mentions (DOC-REF-ONLY): `boundaries_admin1_build.py`, `build_entity_map.py`, `build_nasr_airways.py`, `build_tank_registry.py`, `gem_suite_ingest.py`, `gtrends_probe.py`, `labeler_before_after.py`, `military_installations_build.py`, `site_candidate_verify.py`. ORPHAN (no reference anywhere): `scripts/faa_charts/make_fixture.py` (44 lines).

## 6. Stale / broken tests

- Total test files: 225. `conftest.py` ignores `test_auto_discovery.py` and `test_full_system.py` (standalone scripts).
- **Broken: none.** Every static repo-local import resolves; all 104 `spec_from_file_location(... 'scripts', 'x.py')` paths used by tests exist on disk.
- **Stale (every target is off-runtime):** `test_bot_backtest_subperiods.py`, `test_cftc_tff_tlt_disjoint.py`, `test_cot_gate2_oos.py`, `test_regime_detector_compare.py`, `test_sec_filings_gate2.py`; plus the Lazy Prices set `test_sec_filings_{classify,similarity,text,universe}.py` (targets research-only modules). They pass, they just guard code nothing runs.
- **Mis-collected CLIs:** `cftc_tff_gate2_test.py`, `cot_gate2_test.py`, `form4_gate2_test.py`, `scripts/jodi_gate2_test.py` match pytest's `*_test.py` pattern, contain **0 `test_` functions**, and import `backtest_v2`/`gate2_stats`/`cftc_cot`/`sec_form4_bulk` at collection time. Harmless today; rename to `*_gate2.py` to stop pytest importing them.
- 98 of 225 tests have no static repo-local import (they load `scripts/` by path, grep source, or shell out) — the import graph alone under-counts coverage of `scripts/`.

## 7. Requirements vs imports

- `requirements.txt` (10): numpy, scipy, scikit-learn, joblib, lightgbm, yfinance, pandas, requests, pytrends, anthropic — **all imported by at least one ACTIVE module; 0 unused.** (scipy only by `analyze.py`; pytrends only by `analyze.py`/`social_data.py`; anthropic only by `alphadesk/llm_reader.py`.)
- ACTIVE modules import **nothing undeclared** (the three "missing" names `momentum`/`mean_reversion`/`squeeze` are the strategies/ shim, not packages).
- Imported anywhere but declared in neither requirements file: `laspy` (`scripts/gridvision_lidar_probe.py`), `ultralytics` (`scripts/gridvision_train/train.py`). Both session/GPU tooling — belong in `requirements-dev.txt`.
- `requirements-dev.txt` (12) covers openpyxl/xlrd/tifffile/rasterio/Pillow/pyshp/pyyaml/shapely/pmtiles/boto3/botocore/pytest; CI's `test` job installs it, `python-tests` job does not (only runs 4 files, so fine).

## 8. Env vars (Python)

| Var | Modules |
|---|---|
| ALPACA_KEY / ALPACA_SECRET | 20+ ACTIVE (alpaca_feed, bot_engine, analyze, backtest_v2, csp_universe, options_*, position_sizing, tiered_strategy, …) + alphadesk/config |
| ALPACA_BASE_URL | bot_engine, intraday_shorts, options_execution/manager/scanner, position_sizing, tiered_strategy |
| ALPACA_DATA_FEED / ALPACA_OPTIONS_FEED | alpaca_feed |
| ALPACA_DATA_URL, APCA_API_KEY_ID, APCA_API_SECRET_KEY, EDGAR_USER_AGENT, ALPHADESK_LLM_MODEL, ANTHROPIC_API_KEY | alphadesk/config.py only (second naming family for the same Alpaca creds) |
| ALPACA_TRADING_URL | csp_universe |
| FINNHUB_KEY | diagnostics, etf_data_sources, finnhub_data, ml_model_v2, options_scanner, alphadesk/config |
| POLYGON_KEY | alt_data, bot_engine, etf_data_sources, intelligence, macro_data, social_data, alphadesk/config, **ml_model.py (dead)** |
| POLYGON_API_KEY | bot_engine, etf_data_sources, alphadesk/config (duplicate name of POLYGON_KEY) |
| GITHUB_BACKUP_TOKEN | backup_to_github |
| DATA_DIR | intraday_shorts, ml_status, ml_toggle, voltrade_daemon, sec_filings_text, scripts/earnings_language_gate2 |
| VOLTRADE_DATA_DIR | lookup_ticker, ml_model_v2, options_scanner, **sim_trading_day_v2 (dead)**, tests |
| VOLTRADE_AUTOSEED, VOLTRADE_DAEMON_MAX_INFLIGHT, VOLTRADE_DAEMON_MAX_RSS_MB, VOLTRADE_DAEMON_SOCKET | voltrade_daemon |
| VOLTRADE_DEEP_SCORE_CAP, VOLTRADE_MEM_SKIP_DEEP_MB, VOLTRADE_MEM_TRIM_DEEP_MB, VOLTRADE_SURVIVAL_THRESHOLD_MB, VOLTRADE_TIER2_SURVIVAL_MODE, VOLTRADE_ULTRA_SURVIVAL_MODE | bot_engine |
| VOLTRADE_LIQUIDATE_ON_KILL | risk_kill_switch |
| Tooling-only (scripts/): CDSE_CLIENT_ID/SECRET, DIAG_TOKEN, DOD_INSTALLATIONS_URL, EIA_API_KEY, R2_ACCESS_KEY_ID/ACCOUNT_ID/PUBLIC_URL/SECRET_ACCESS_KEY/TILES_BUCKET, RUNPOD_API_KEY, TMPDIR, VOLTRADE_BASE_URL, VOLTRADE_GEO_CACHE | — |

No env var is read **only** by a dead module; the two dead readers above are shadows of live ones.

## 9. Duplicate groups

1. **ML layer:** `ml_model.py` (v1 sklearn RF+GB, dead) vs `ml_model_v2.py` (LightGBM, active) — shared public names `ml_score`, `track_fill`, `train_model`.
2. **Backtest engines:** `backtest.py`→`backtest_v2.py` (active, Node-spawned), `bot_backtest.py` (+`_subperiods`), `sim_trading_day_v2.py`; `seed_feedback_from_backtest.py` consumes `backtest_10yr_results.json` whose generator is not in the repo (bot_backtest.py docstring says so).
3. **Regime detection, three coexisting:** `markov_regime.get_regime`, `regime_util.classify_regime/_5level/regime_probability` (VXX-ratio), `regime_inputs.regime_inputs`, plus ml_model_v2's own regime classification; `regime_detector_compare.py` exists precisely because they coexist.
4. **Feedback cleanup ×3:** `cleanup_feedback.py` (one-off), `feedback_boot_cleanup.clean_feedback` (active), ml_model_v2 startup cleanup.
5. **Alpaca HTTP:** `alpaca_feed.py` is the declared central feed, yet 25 non-test modules still hit `alpaca.markets` directly (analyze, bot_engine, csp_universe, instrument_selector, intelligence, intraday_shorts, macro_data, options_*, paper_resume_sync, position_sizing, probability_engine, shadow_portfolio, social_data, tiered_strategy, vol_surface…); 18 of them also import alpaca_feed — consolidation half-done.
6. **Finnhub:** direct `finnhub.io` calls in `finnhub_data.py`, `etf_data_sources.py`, `ml_model_v2.py`, `options_scanner.py`, `alphadesk/market.py`.
7. **Insider / EDGAR:** `intelligence.get_insider_activity` (sec.gov), `finnhub_data.get_insider_sentiment`, `sec_form4_bulk.run_update`, `institutional_data` (13F), `alt_data` (sec.gov), `alphadesk/edgar.py`+`providers.py`, the Lazy Prices `sec_filings_*` chain — and `server/edgarForm4.ts` on the Node side.
8. **Polygon:** `alt_data`, `etf_data_sources`, `macro_data`, `alphadesk/market`, `ml_model` (dead); two env names (`POLYGON_KEY`, `POLYGON_API_KEY`).
9. **yfinance spread:** analyze, bot_engine, etf_analyzer, insights, macro_data, position_sizing (conftest has to mock it suite-wide).
10. **Gate-2 harness copies:** `cot_gate2_test.py`, `cftc_tff_gate2_test.py`, `form4_gate2_test.py`, `scripts/jodi_gate2_test.py`, `scripts/*_gate2.py` share `gate2_stats`; `scripts/crop_conditions_gate2.py` ↔ `scripts/eia930_gate2.py` share 6 identical function names (`bucket_for`, `compute_forward_returns`, `evaluate_pass_bar`, `hac_significance`, `summarize`, `main`); `rail_traffic_gate2` ↔ `usaspending_gate2` share 4.
11. **scripts/ copy-paste families:** `cdse_chips.py` ↔ `cdse_s1_chips.py` (13 shared names) ↔ `cdse_site_chips.py` (9); `rail_traffic_gate1` ↔ `un_comtrade_gate1` (FRED CSV fetch); `sentinel2_refresh` ↔ `snapshot_refresh` (`build_plan`/`pick_python`/`interpreter_ok`); `csp_manager_rule_sim` ↔ `vrp_single_name_vs_index` ↔ `exit_model_reconciliation` ↔ `floor_basket_vs_index` (`aggregate`/`render`/`verdict` scaffold); `illiquid_universe_probe*.py` (7 variants); `nhtsa_gate1_probe` ↔ `nhtsa_gate2_probe`.

## 10. Architecture notes (for the report)

1. Railway runs `run_with_daemon.sh`: Node (`dist/index.cjs`) plus a supervised `voltrade_daemon.py` that imports numpy/pandas once and serves JSON-RPC over `/tmp/voltrade_daemon.sock`.
2. `server/bot.ts` owns all scheduling (Tier 1/2/3 timers). It reaches Python two ways: `pythonCall(method, args, fallbackCmd)` = daemon RPC first, then an `execPythonSerialized` subprocess (one at a time, SIGKILL timeouts); and ~44 direct `python3 -c` one-liners that import modules by name.
3. Heart of the bot: `bot_engine.py` (scan_market/deep_score/manage_positions) → `ml_model_v2.py` (LightGBM, track_fill, feedback) → `options_scanner.py`/`options_manager.py`/`options_execution.py`/`vol_surface.py`/`csp_universe.py` (options stack) → `tiered_strategy.py` → `position_sizing.py` + `risk_kill_switch.py` (sizing/halts), all parameterised by `system_config.py`, all paths from `storage_config.py`, market data through `alpaca_feed.py` + `alpaca_rate_limiter.py`.
4. Daemon method table is a whitelist of 20 names; `run_full_scan` is daemon-only (subprocess fallback would OOM); light methods fall back to inline `python3 -c`.
5. Two table entries (`ml_status_impl`, `ml_toggle_impl`) are phantom modules — the daemon silently always serves its inline fallbacks; `ml_status.py`/`ml_toggle.py` survive only as subprocess scripts (routes.ts still spawns `ml_toggle.py` directly).
6. Alternative-data modules (`alt_data`, `social_data`, `intelligence`, `finnhub_data`, `institutional_data`, `markov_regime`, `strategies/*`) are ACTIVE but fail-soft: imported inside `try/except … return {}` at call time, so breakage never raises.
7. `strategies/` is loaded two ways (bare import via sys.path shim in bot_engine; package import in backtest_v2) — same files, two module identities.
8. Research/validation lives at root as standalone CLIs (`*_gate2_test.py`, `sec_filings_*`, `bot_backtest*`, `regime_detector_compare`) and in `scripts/` (133 files, never shipped to Railway, exercised by 100+ tests via `spec_from_file_location`).
9. `alphadesk/` is a self-contained stdlib package invoked only as `python3 -m alphadesk` from routes.ts; it duplicates Alpaca/Finnhub/EDGAR fetch code and has its own env-var names.
10. CI = `compileall` + 4 offline tests (`python-tests`) and the full pytest gate (`test` job via `scripts/gated_tests.sh`), both required for automerge; `python3 -m pytest -q` collects 225 files (2 ignored via conftest).
