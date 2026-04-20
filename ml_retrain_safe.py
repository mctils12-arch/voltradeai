#!/usr/bin/env python3
"""
Safe ML retrain wrapper — ultra lightweight.
Step 1: Just prove Python can run this file on Railway.
Step 2: Try importing.
Step 3: Try training (with minimal data).
"""
import json, sys, os

def safe_retrain():
    steps = []
    try:
        steps.append("python_started")
        
        # Step 1: Can we even import numpy?
        import numpy as np
        steps.append("numpy_ok")
        
        # Step 2: Can we import lightgbm?
        import lightgbm as lgb
        steps.append("lightgbm_ok")
        
        # Step 3: Can we import our module?
        from ml_model_v2 import train_model
        steps.append("import_ok")
        
        # Step 4: Set memory limits
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        
        # Step 5: Actually train
        steps.append("training_start")
        result = train_model(fast_mode=True)
        steps.append("training_done")
        result["steps"] = steps
        return result
        
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "steps_completed": steps,
            "error": str(e)[:400],
            "traceback": traceback.format_exc()[-600:],
        }

if __name__ == "__main__":
    try:
        r = safe_retrain()
        print(json.dumps(r))
    except Exception:
        print('{"status":"fatal","error":"json.dumps failed"}')
