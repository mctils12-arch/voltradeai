#!/usr/bin/env python3
"""
Safe ML retrain wrapper. Never crashes — always prints valid JSON.
Used by bot.ts to avoid 'Command failed' errors.
"""
import json, sys, traceback

def safe_retrain():
    try:
        from ml_model_v2 import train_model
        result = train_model(fast_mode=True)
        return result
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)[:300],
            "traceback": traceback.format_exc()[-500:],
            "python_version": sys.version,
        }

if __name__ == "__main__":
    print(json.dumps(safe_retrain()))
