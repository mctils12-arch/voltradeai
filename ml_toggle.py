#!/usr/bin/env python3
"""Toggle ML on/off. Usage: python3 ml_toggle.py enable|disable|status"""
import json, os, sys

try:
    from storage_config import DATA_DIR
except ImportError:
    DATA_DIR = os.environ.get("DATA_DIR", "/tmp")

TOGGLE_PATH = os.path.join(DATA_DIR, "ml_toggle.json")
os.makedirs(DATA_DIR, exist_ok=True)

def get_status():
    try:
        with open(TOGGLE_PATH) as f:
            return json.load(f).get("enabled", False)
    except Exception:
        return False

def set_toggle(enabled: bool):
    with open(TOGGLE_PATH, "w") as f:
        json.dump({"enabled": enabled}, f)
    return enabled

if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "status"
    if action == "enable":
        set_toggle(True)
        print(json.dumps({"enabled": True, "status": "ok"}))
    elif action == "disable":
        set_toggle(False)
        print(json.dumps({"enabled": False, "status": "ok"}))
    else:
        print(json.dumps({"enabled": get_status(), "status": "ok"}))
