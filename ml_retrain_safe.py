#!/usr/bin/env python3
"""
Safe ML retrain wrapper. Never crashes — always prints valid JSON.
Memory-light mode for Railway's constrained environment.
"""
import json, sys, os, traceback

def safe_retrain():
    result = {"status": "starting"}
    try:
        # Step 1: Check available memory
        try:
            import resource
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            result["mem_limit"] = f"soft={soft}, hard={hard}"
        except:
            pass

        # Step 2: Try importing — if this fails, it's a missing dependency
        result["status"] = "importing"
        from ml_model_v2 import train_model
        result["status"] = "imported"

        # Step 3: Train with fast_mode and memory limits
        # Set environment to reduce memory usage
        os.environ["OMP_NUM_THREADS"] = "1"  # LightGBM single thread
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        
        result["status"] = "training"
        train_result = train_model(fast_mode=True)
        result = train_result
        return result

    except MemoryError:
        result["status"] = "error"
        result["error"] = "OUT OF MEMORY — Railway killed the process"
        result["fix"] = "Need to reduce training data size"
        return result
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)[:500]
        result["traceback"] = traceback.format_exc()[-800:]
        result["python"] = sys.version
        result["cwd"] = os.getcwd()
        result["files"] = [f for f in os.listdir('.') if f.endswith('.py')][:10]
        return result

if __name__ == "__main__":
    try:
        print(json.dumps(safe_retrain()))
    except Exception as e:
        # Last resort — even json.dumps failed
        print(json.dumps({"status": "fatal", "error": str(e)[:200]}))
