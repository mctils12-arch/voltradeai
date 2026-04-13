#!/usr/bin/env python3
"""ML model status check. Usage: python3 ml_status.py"""
import json, os, time

try:
    from storage_config import DATA_DIR, ML_MODEL_PATH as model_path
except ImportError:
    DATA_DIR = os.environ.get("DATA_DIR", "/tmp")
    model_path = os.path.join(DATA_DIR, "voltrade_ml_v2.pkl")

status_path = os.path.join(DATA_DIR, "ml_status.json")
toggle_path = os.path.join(DATA_DIR, "ml_toggle.json")

# Check model file
model_exists = os.path.exists(model_path)
model_age_hours = None
if model_exists:
    model_age_hours = round((time.time() - os.path.getmtime(model_path)) / 3600, 1)

# Check toggle state
enabled = False
try:
    with open(toggle_path) as f:
        enabled = json.load(f).get("enabled", False)
except Exception:
    pass

# Check last training result
last_train = {}
try:
    with open(status_path) as f:
        last_train = json.load(f)
except Exception:
    pass

result = {
    "model_exists": model_exists,
    "model_age_hours": model_age_hours,
    "enabled": enabled,
    "last_accuracy": last_train.get("accuracy"),
    "last_samples": last_train.get("samples"),
    "last_features": last_train.get("feature_count"),
    "last_status": last_train.get("status", "unknown"),
    "last_error": last_train.get("error"),
    "last_traceback": last_train.get("traceback"),
    "last_steps": last_train.get("steps") or last_train.get("steps_completed"),
    "last_train_time": last_train.get("timestamp"),
    "contributes_to_cagr": model_exists and enabled,
    "note": "ML enabled. Auto-retrain runs at 4am ET daily and hourly via Tier 3."
        if enabled
        else "ML disabled. POST /api/ml/toggle with {enabled: true} to enable.",
}
print(json.dumps(result))
