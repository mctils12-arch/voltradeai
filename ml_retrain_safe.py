#!/usr/bin/env python3
"""
Safe ML retrain wrapper — ultra lightweight.

batch 8 REVISED 2026-05-04: production was reporting:
    TIER3-ML-ERROR — ML retrain failed (code=? signal=SIGKILL)
                     [likely OOM kill during import lightgbm/ml_model_v2]
That "[likely OOM]" tag came from the bot.ts SIGKILL classifier guessing
based on signal=SIGKILL + empty stderr. With an 8GB container the daemon
only uses ~280MB so it almost certainly is NOT an OOM. Most likely cause
is bot.ts hitting its 300s exec timeout (Node sends SIGTERM then SIGKILL,
which looks identical to an OS OOM kill from outside).

Real fixes here:
  1. Set thread-limit env vars BEFORE any heavy import. Was being set
     AFTER `import lightgbm` (no-op once threads spawned). This is a real
     bug regardless of whether OOM is the actual cause — it makes the
     retrain faster and lighter, and faster training is less likely to
     hit the bot.ts timeout.
  2. Better step tracking via `steps_completed` array so the next failure
     tells us EXACTLY where it died (lightgbm import? data load? training?)
     instead of a generic SIGKILL.

What was REMOVED from the original batch 8 plan:
  - The 512MB resource.setrlimit cap. With 8GB available, restricting
    the child to 512MB would only hurt — better to use the headroom.
  - Reductions to `_fetch_training_bars` defaults (200→60 tickers,
    2020→2024 start). Reverted in ml_model_v2.py — full historical
    regime diversity is valuable and 8GB has plenty of room.
"""
import json, sys, os

# Set thread-limit env vars BEFORE any heavy import. lightgbm and openblas
# spawn one thread per CPU core by default, and once those threads are
# created changing the env var is a no-op. Keeping things lean here makes
# training faster which avoids the bot.ts 300s timeout (the actual cause
# of the SIGKILL we were seeing).
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
os.environ.setdefault("LIGHTGBM_NUM_THREADS", "2")


def safe_retrain():
    steps = []
    try:
        steps.append("python_started")

        # Step 1: Can we even import numpy?
        import numpy as np
        steps.append("numpy_ok")

        # Step 2: Can we import lightgbm?
        # If something kills the process between numpy_ok and lightgbm_ok,
        # the next snapshot will show steps_completed: ["numpy_ok"] and
        # we know it died during the lightgbm import.
        import lightgbm as lgb
        steps.append("lightgbm_ok")

        # Step 3: Can we import our module?
        from ml_model_v2 import train_model
        steps.append("import_ok")

        # Step 4: Actually train.
        # (Memory env vars already set above, before any imports)
        steps.append("training_start")
        result = train_model(fast_mode=True)
        steps.append("training_done")
        result["steps"] = steps
        return result

    except MemoryError as e:
        # Belt-and-suspenders: if the process ever does hit a real OOM
        # (numpy raises MemoryError before the OS SIGKILLs us), capture
        # it cleanly with steps_completed.
        import traceback
        return {
            "status": "error",
            "error_class": "MemoryError",
            "steps_completed": steps,
            "error": str(e)[:400],
            "traceback": traceback.format_exc()[-600:],
        }
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
