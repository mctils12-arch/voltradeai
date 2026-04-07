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
EVENT_MEMORY_PATH = os.path.join(DATA_DIR, "voltrade_event_memory.json")
EARNINGS_MEMORY_PATH = os.path.join(DATA_DIR, "voltrade_earnings_memory.json")
FILLS_PATH = os.path.join(DATA_DIR, "voltrade_fills.json")
WEIGHTS_PATH = os.path.join(DATA_DIR, "voltrade_weights.json")
HEALTH_LOG_PATH = os.path.join(DATA_DIR, "voltrade_health_log.json")
BLEND_TRACKER_PATH = os.path.join(DATA_DIR, "voltrade_blend_tracker.json")
INSIDER_CACHE_PATH = os.path.join(DATA_DIR, "voltrade_insider_cache.json")

# v2 ML model (clean 25-feature, self-learning)
ML_MODEL_V2_PATH = os.path.join(DATA_DIR, "voltrade_ml_v2.pkl")
