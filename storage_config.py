"""
VolTradeAI Storage Configuration
Persistent data goes to DATA_DIR (Railway volume mount).
Temporary cache goes to CACHE_DIR (/tmp).
"""
import os

# Persistent data survives deploys (Railway volume at /data)
# Falls back to /tmp if /data doesn't exist (local dev)
if os.path.isdir("/data"):
    DATA_DIR = "/data/voltrade"
else:
    DATA_DIR = "/tmp"

os.makedirs(DATA_DIR, exist_ok=True)

# Temporary cache (fine to lose on redeploy)
CACHE_DIR = "/tmp/voltrade_alt_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Persistent file paths (survive deploys)
ML_MODEL_PATH = os.path.join(DATA_DIR, "voltrade_ml_v2.pkl")   # v1.0.26: points to ml_model_v2 (was voltrade_ml_model.pkl)
TRADE_FEEDBACK_PATH = os.path.join(DATA_DIR, "voltrade_trade_feedback.json")
# REGIME ENGINE 2026-10-28 (docs/regime/FINDINGS.md step 2): latest state + append-only daily history
REGIME_STATE_PATH = os.path.join(DATA_DIR, "voltrade_regime_state.json")
REGIME_HISTORY_PATH = os.path.join(DATA_DIR, "voltrade_regime_history.jsonl")
REGIME_SIZING_PATH = os.path.join(DATA_DIR, "voltrade_regime_sizing.json")   # step 6: daily sizing-multiplier counters (regime_engine.flush_sizing_counts)
# 2026-10-28 live flip: append-only legacy-vs-engine counterfactual rows, one per decision (regime_engine.log_counterfactual; rotated to .1 above the size cap)
REGIME_COUNTERFACTUAL_PATH = os.path.join(DATA_DIR, "voltrade_regime_counterfactual.jsonl")
EVENT_MEMORY_PATH = os.path.join(DATA_DIR, "voltrade_event_memory.json")
EARNINGS_MEMORY_PATH = os.path.join(DATA_DIR, "voltrade_earnings_memory.json")
FILLS_PATH = os.path.join(DATA_DIR, "voltrade_fills.json")
WEIGHTS_PATH = os.path.join(DATA_DIR, "voltrade_weights.json")
HEALTH_LOG_PATH = os.path.join(DATA_DIR, "voltrade_health_log.json")
BLEND_TRACKER_PATH = os.path.join(DATA_DIR, "voltrade_blend_tracker.json")
INSIDER_CACHE_PATH = os.path.join(DATA_DIR, "voltrade_insider_cache.json")
COT_ARCHIVE_PATH = os.path.join(DATA_DIR, "voltrade_cot_archive.json")
COT_CHECKPOINT_PATH = os.path.join(DATA_DIR, "voltrade_cot_checkpoint.json")
FORM4_BULK_DIR = os.path.join(DATA_DIR, "form4_bulk")
os.makedirs(FORM4_BULK_DIR, exist_ok=True)
FORM4_BULK_CHECKPOINT_PATH = os.path.join(DATA_DIR, "voltrade_form4_bulk_checkpoint.json")
# 2026-10-28 price panel (server/pricePanel.ts writes it nightly; Python readers
# load <PRICE_PANEL_DIR>/current.json -> <build_id>/bars/*.jsonl.gz). Not created here.
PRICE_PANEL_DIR = os.path.join(DATA_DIR, "price_panel")
# 2026-10-29 long-term walk-forward of the stock score's price inputs over that
# panel (walkforward.py, weekly via server/walkforward.ts): latest.json (last
# success, read by the Brain page) + last_run.json (every run). Not created here.
WALKFORWARD_DIR = os.path.join(DATA_DIR, "walkforward")

# Weekly forecast screen (forecast.py, 2026-10-27, SHADOW): append-only JSONL
# — the calls (+ one batch line per week) and, separately, their scores.
FORECASTS_PATH = os.path.join(DATA_DIR, "voltrade_forecasts.jsonl")
FORECAST_SCORES_PATH = os.path.join(DATA_DIR, "voltrade_forecast_scores.jsonl")

# Ticker dossier calibration ledger (dossier_ranges.py, 2026-10-29): append-only
# JSONL — every served range band (ticker x base date x horizon) and trade plan
# (ticker x serve date), first served wins; scores go to their own file.
DOSSIER_LEDGER_PATH = os.path.join(DATA_DIR, "voltrade_dossier_ledger.jsonl")
DOSSIER_SCORES_PATH = os.path.join(DATA_DIR, "voltrade_dossier_scores.jsonl")

# v2 ML model (clean 25-feature, self-learning)
ML_MODEL_V2_PATH = os.path.join(DATA_DIR, "voltrade_ml_v2.pkl")
