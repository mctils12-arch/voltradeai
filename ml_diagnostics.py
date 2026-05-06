#!/usr/bin/env python3
"""
ml_diagnostics.py — One-shot ML process inspector.

ALPHA AUDIT 2026-05-04 batch 8 add-on: gathers EVERY piece of state
about the ML pipeline in a single JSON payload. The user hits
/api/debug/ml-diagnostics, screenshots the JSON, sends it to us,
and we see exactly what's broken without further round-trips.

What this captures:
  - Environment (Python version, RSS, env vars, container info)
  - File system state (model exists? age? size? readable?)
  - Trade feedback file state (count, fingerprint distribution, version skew)
  - Library availability (numpy, lightgbm, sklearn, joblib — versions)
  - Last retrain attempt (if a state file exists)
  - Static smoke test of train_model dependencies (can we import?
    can we read the feedback file? can we instantiate a tiny model?)
  - Recent ML audit entries (if accessible)

Output: pure JSON on stdout. Always exits 0 — wraps everything in
try/except so a partial failure still gives us partial data.

Usage from CLI:
    python3 ml_diagnostics.py

Behind the /api/debug/ml-diagnostics route this is invoked via execAsync
and gated by DEBUG_ENABLED=true (same pattern as Finnhub debug).
"""
import json
import os
import sys
import time
import traceback
from datetime import datetime


def _safe(fn, *args, **kwargs):
    """Call fn, never raise — return {error: <msg>} on failure."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        return {"error": str(e)[:200], "type": type(e).__name__}


# ────────────────────────────────────────────────────────────────────────────
# Section 1: Environment
# ────────────────────────────────────────────────────────────────────────────

def env_section():
    out = {
        "python_version": sys.version.split()[0],
        "platform": sys.platform,
        "executable": sys.executable,
        "cwd": os.getcwd(),
    }

    # Memory (Linux only)
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    out["rss_kb"] = int(line.split()[1])
                    out["rss_mb"] = round(out["rss_kb"] / 1024, 1)
                if line.startswith("VmPeak:"):
                    out["peak_kb"] = int(line.split()[1])
    except Exception:
        out["rss_mb"] = None

    # Container memory limit (Linux cgroup)
    for path in ["/sys/fs/cgroup/memory.max",
                 "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            with open(path) as f:
                v = f.read().strip()
                if v.isdigit():
                    out["container_memory_limit_mb"] = round(int(v) / (1024 * 1024), 0)
                    break
                elif v == "max":
                    out["container_memory_limit_mb"] = "max (no limit)"
                    break
        except Exception:
            continue

    # Thread-related env vars (these matter for OOM diagnosis)
    out["thread_env"] = {
        k: os.environ.get(k, "(unset)")
        for k in [
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
            "LIGHTGBM_NUM_THREADS",
        ]
    }

    # CPU count
    try:
        out["cpu_count_logical"] = os.cpu_count()
    except Exception:
        pass

    # Resource limits (rlimit) if available
    try:
        import resource
        rss_soft, rss_hard = resource.getrlimit(resource.RLIMIT_AS)
        out["rlimit_as_soft_mb"] = (rss_soft / (1024 * 1024)) if rss_soft != resource.RLIM_INFINITY else "unlimited"
        out["rlimit_as_hard_mb"] = (rss_hard / (1024 * 1024)) if rss_hard != resource.RLIM_INFINITY else "unlimited"
    except Exception:
        pass

    return out


# ────────────────────────────────────────────────────────────────────────────
# Section 2: File system state
# ────────────────────────────────────────────────────────────────────────────

def filesystem_section():
    out = {}
    try:
        import ml_model_v2 as mlm
        paths_to_check = {
            "model": getattr(mlm, "MODEL_PATH", None),
            "meta_model": getattr(mlm, "META_MODEL_PATH", None),
            "options_model": getattr(mlm, "OPTIONS_MODEL_PATH", None),
        }
        try:
            from storage_config import DATA_DIR
            paths_to_check["data_dir"] = DATA_DIR
            paths_to_check["feedback_file"] = os.path.join(DATA_DIR, "trade_feedback.json")
        except ImportError:
            pass

        for name, path in paths_to_check.items():
            if not path:
                out[name] = {"path": None, "configured": False}
                continue
            info = {"path": path, "exists": os.path.exists(path)}
            if info["exists"]:
                try:
                    info["size_bytes"] = os.path.getsize(path)
                    info["size_mb"] = round(info["size_bytes"] / (1024 * 1024), 3)
                    info["mtime"] = datetime.fromtimestamp(os.path.getmtime(path)).isoformat()
                    info["age_hours"] = round((time.time() - os.path.getmtime(path)) / 3600, 1)
                    info["readable"] = os.access(path, os.R_OK)
                    info["writable"] = os.access(path, os.W_OK)
                except Exception as e:
                    info["error"] = str(e)[:120]
            out[name] = info
    except Exception as e:
        out["error"] = f"could not import ml_model_v2: {str(e)[:120]}"
    return out


# ────────────────────────────────────────────────────────────────────────────
# Section 3: Trade feedback file analysis
# ────────────────────────────────────────────────────────────────────────────

def feedback_section():
    try:
        from storage_config import DATA_DIR
    except Exception:
        return {"error": "storage_config not importable"}

    fb_path = os.path.join(DATA_DIR, "trade_feedback.json")
    if not os.path.exists(fb_path):
        return {"exists": False, "path": fb_path}

    try:
        with open(fb_path) as f:
            records = json.load(f)
    except Exception as e:
        return {"exists": True, "path": fb_path, "parse_error": str(e)[:200]}

    if not isinstance(records, list):
        return {"exists": True, "path": fb_path, "shape_error": f"expected list, got {type(records).__name__}"}

    out = {
        "exists": True,
        "path": fb_path,
        "count": len(records),
    }

    # Categorize by fingerprint and source
    from collections import Counter
    fingerprints = Counter()
    sources = Counter()
    code_versions = Counter()
    bucket_counts = Counter()
    has_outcome = 0
    has_pnl = 0

    for rec in records:
        if not isinstance(rec, dict):
            continue
        fp = rec.get("fingerprint", "<missing>")
        fingerprints[str(fp)[:40]] += 1
        sources[rec.get("source", "<unset>")] += 1
        code_versions[str(rec.get("code_version", "<unset>"))[:20]] += 1
        bucket_counts[rec.get("bucket", "<unset>")] += 1
        if "outcome" in rec or "pnl_pct" in rec:
            has_outcome += 1
        if "pnl" in rec or "pnl_pct" in rec:
            has_pnl += 1

    out["fingerprints"] = dict(fingerprints.most_common(10))
    out["sources"] = dict(sources)
    out["code_versions"] = dict(code_versions)
    out["bucket_counts"] = dict(bucket_counts)
    out["records_with_outcome"] = has_outcome
    out["records_with_pnl"] = has_pnl

    # Sample first/last records for shape inspection
    if records:
        out["first_record_keys"] = sorted(records[0].keys()) if isinstance(records[0], dict) else None
        out["last_record_keys"] = sorted(records[-1].keys()) if isinstance(records[-1], dict) else None

    return out


# ────────────────────────────────────────────────────────────────────────────
# Section 4: Library availability and versions
# ────────────────────────────────────────────────────────────────────────────

def libraries_section():
    out = {}
    libs = ["numpy", "pandas", "scipy", "sklearn", "lightgbm", "joblib", "requests"]
    for lib in libs:
        info = {"installed": False, "version": None, "import_time_sec": None}
        t0 = time.time()
        try:
            mod = __import__(lib)
            info["installed"] = True
            info["version"] = getattr(mod, "__version__", "unknown")
            info["import_time_sec"] = round(time.time() - t0, 3)
        except ImportError as e:
            info["error"] = str(e)[:120]
        except Exception as e:
            info["error"] = f"unexpected: {str(e)[:120]}"
        out[lib] = info
    return out


# ────────────────────────────────────────────────────────────────────────────
# Section 5: Last retrain attempt state
# ────────────────────────────────────────────────────────────────────────────

def last_retrain_section():
    """Look for any state file the bot writes after a retrain attempt."""
    try:
        from storage_config import DATA_DIR
    except Exception:
        return {"error": "storage_config not available"}

    candidates = [
        os.path.join(DATA_DIR, "ml_retrain_status.json"),
        os.path.join(DATA_DIR, "ml_last_retrain.json"),
        "/tmp/ml_retrain_status.json",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                data["_read_from"] = path
                data["_age_hours"] = round((time.time() - os.path.getmtime(path)) / 3600, 1)
                return data
            except Exception as e:
                return {"error": f"could not read {path}: {str(e)[:120]}"}
    return {"info": "no retrain status file found at expected locations"}


# ────────────────────────────────────────────────────────────────────────────
# Section 6: Static smoke test — can we even START a retrain?
# ────────────────────────────────────────────────────────────────────────────

def smoke_test_section():
    """Step through what a retrain would do, WITHOUT actually retraining.
    Each step has its own try/except so a failure in one doesn't block
    visibility into the others."""
    steps = []

    # Step A: import numpy
    try:
        t0 = time.time()
        import numpy as np
        steps.append({"step": "import_numpy", "ok": True,
                      "time_sec": round(time.time() - t0, 3),
                      "version": np.__version__})
    except Exception as e:
        steps.append({"step": "import_numpy", "ok": False, "error": str(e)[:200],
                      "type": type(e).__name__})
        return {"steps": steps, "halt_reason": "numpy unavailable"}

    # Step B: import lightgbm (the big one — historically the SIGKILL spot)
    try:
        t0 = time.time()
        import lightgbm as lgb
        steps.append({"step": "import_lightgbm", "ok": True,
                      "time_sec": round(time.time() - t0, 3),
                      "version": lgb.__version__})
    except Exception as e:
        steps.append({"step": "import_lightgbm", "ok": False, "error": str(e)[:200],
                      "type": type(e).__name__})
        # Continue anyway — sklearn fallback exists

    # Step C: import sklearn
    try:
        t0 = time.time()
        import sklearn
        steps.append({"step": "import_sklearn", "ok": True,
                      "time_sec": round(time.time() - t0, 3),
                      "version": sklearn.__version__})
    except Exception as e:
        steps.append({"step": "import_sklearn", "ok": False, "error": str(e)[:200]})
        return {"steps": steps, "halt_reason": "sklearn unavailable — required"}

    # Step D: import ml_model_v2 (our module)
    try:
        t0 = time.time()
        import ml_model_v2
        steps.append({"step": "import_ml_model_v2", "ok": True,
                      "time_sec": round(time.time() - t0, 3),
                      "has_train_model": hasattr(ml_model_v2, "train_model")})
    except Exception as e:
        steps.append({"step": "import_ml_model_v2", "ok": False, "error": str(e)[:200],
                      "traceback": traceback.format_exc()[-400:]})
        return {"steps": steps, "halt_reason": "ml_model_v2 import failed"}

    # Step E: load the trade feedback file (no training, just load)
    try:
        t0 = time.time()
        feedback = ml_model_v2._load_trade_feedback()
        steps.append({"step": "load_trade_feedback", "ok": True,
                      "time_sec": round(time.time() - t0, 3),
                      "records_loaded": len(feedback) if isinstance(feedback, list) else None})
    except Exception as e:
        steps.append({"step": "load_trade_feedback", "ok": False,
                      "error": str(e)[:200],
                      "traceback": traceback.format_exc()[-400:]})

    # Step F: instantiate a tiny LightGBM model (proves the library actually
    # works for training, not just import)
    try:
        import lightgbm as lgb
        import numpy as np
        t0 = time.time()
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)
        model = lgb.LGBMClassifier(n_estimators=10, num_leaves=4, verbose=-1)
        model.fit(X, y)
        steps.append({"step": "lightgbm_smoke_fit", "ok": True,
                      "time_sec": round(time.time() - t0, 3),
                      "info": "trained tiny model on 100 random rows successfully"})
    except Exception as e:
        steps.append({"step": "lightgbm_smoke_fit", "ok": False,
                      "error": str(e)[:200],
                      "type": type(e).__name__})

    # Step G: fetch a tiny sample of training bars
    try:
        import ml_model_v2
        t0 = time.time()
        bars = ml_model_v2._fetch_training_bars(days=30, max_tickers=3)
        steps.append({"step": "fetch_training_bars_tiny", "ok": True,
                      "time_sec": round(time.time() - t0, 3),
                      "tickers_returned": len(bars) if isinstance(bars, dict) else None,
                      "info": "fetched 30 days × 3 tickers as a smoke test"})
    except Exception as e:
        steps.append({"step": "fetch_training_bars_tiny", "ok": False,
                      "error": str(e)[:200],
                      "traceback": traceback.format_exc()[-400:]})

    return {"steps": steps, "halt_reason": None}


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def gather_diagnostics():
    """Run every section. Each is wrapped in try/except so partial
    failures still yield partial data."""
    return {
        "generated_at": datetime.now().isoformat(),
        "schema_version": "ml-diagnostics-v1",
        "environment":      _safe(env_section),
        "filesystem":       _safe(filesystem_section),
        "feedback_file":    _safe(feedback_section),
        "libraries":        _safe(libraries_section),
        "last_retrain":     _safe(last_retrain_section),
        "smoke_test":       _safe(smoke_test_section),
    }


if __name__ == "__main__":
    try:
        result = gather_diagnostics()
        print(json.dumps(result, default=str, indent=2))
    except Exception as e:
        # Last-resort fallback so the route always gets valid JSON
        print(json.dumps({
            "fatal_error": str(e)[:200],
            "traceback": traceback.format_exc()[-500:],
        }))
    sys.exit(0)
