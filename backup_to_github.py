#!/usr/bin/env python3
"""
VolTradeAI — Off-Site Backup to GitHub
=======================================
Pushes learning data files to the 'data-backup' branch of the repo.
Called nightly at 4am ET by bot.ts alongside ML retrain.

Requires GITHUB_BACKUP_TOKEN env variable (GitHub PAT with repo scope).
If not set, falls back to local-only backup silently.

Usage:
  python3 backup_to_github.py          # Run backup
  python3 backup_to_github.py status   # Check last backup status
"""

import os
import sys
import json
import shutil
import subprocess
import time
from datetime import datetime

try:
    from storage_config import DATA_DIR
except ImportError:
    DATA_DIR = "/data/voltrade" if os.path.isdir("/data") else "/tmp"

GITHUB_TOKEN = os.environ.get("GITHUB_BACKUP_TOKEN", "")
REPO_URL = "https://github.com/mctils12-arch/voltradeai.git"
BRANCH = "data-backup"
CLONE_DIR = "/tmp/voltrade_backup_clone"
STATUS_FILE = os.path.join(DATA_DIR, "backup_status.json")

# Files to back up (only JSON + model — skip large/temp files)
BACKUP_EXTENSIONS = ('.json', '.pkl')
MAX_FILE_SIZE_MB = 50  # Skip files larger than 50MB


def run_backup():
    """Clone data-backup branch, copy data files, commit and push."""
    if not GITHUB_TOKEN:
        return {"status": "skipped", "reason": "GITHUB_BACKUP_TOKEN not set. Add a GitHub PAT to Railway env variables."}

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    auth_url = REPO_URL.replace("https://", f"https://x-access-token:{GITHUB_TOKEN}@")

    try:
        # Clean previous clone
        if os.path.exists(CLONE_DIR):
            shutil.rmtree(CLONE_DIR)

        # Clone just the data-backup branch (shallow = fast)
        result = subprocess.run(
            ["git", "clone", "--branch", BRANCH, "--depth", "1", auth_url, CLONE_DIR],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            return {"status": "error", "reason": f"Clone failed: {result.stderr[:200]}"}

        # Configure git identity
        subprocess.run(["git", "config", "user.email", "bot@voltradeai.com"], cwd=CLONE_DIR, capture_output=True)
        subprocess.run(["git", "config", "user.name", "VolTradeAI Bot"], cwd=CLONE_DIR, capture_output=True)

        # Copy data files
        files_copied = []
        if os.path.isdir(DATA_DIR):
            for fname in os.listdir(DATA_DIR):
                fpath = os.path.join(DATA_DIR, fname)
                if not os.path.isfile(fpath):
                    continue
                if not any(fname.endswith(ext) for ext in BACKUP_EXTENSIONS):
                    continue
                # Skip overly large files
                size_mb = os.path.getsize(fpath) / (1024 * 1024)
                if size_mb > MAX_FILE_SIZE_MB:
                    continue
                shutil.copy2(fpath, os.path.join(CLONE_DIR, fname))
                files_copied.append(fname)

        if not files_copied:
            return {"status": "skipped", "reason": "No data files found to back up"}

        # Also copy the SQLite database if it exists
        for db_name in ["voltrade_auth.db", "voltrade.db"]:
            db_path = os.path.join(DATA_DIR, db_name)
            if not os.path.exists(db_path):
                # Check parent dir
                db_path = os.path.join(os.path.dirname(DATA_DIR), db_name)
            if os.path.exists(db_path):
                size_mb = os.path.getsize(db_path) / (1024 * 1024)
                if size_mb < MAX_FILE_SIZE_MB:
                    shutil.copy2(db_path, os.path.join(CLONE_DIR, db_name))
                    files_copied.append(db_name)

        # Stage, commit, push
        subprocess.run(["git", "add", "-A"], cwd=CLONE_DIR, capture_output=True)

        # Check if there are actual changes
        diff_result = subprocess.run(
            ["git", "diff", "--cached", "--stat"], cwd=CLONE_DIR, capture_output=True, text=True
        )
        if not diff_result.stdout.strip():
            status = {"status": "no_changes", "reason": "Data unchanged since last backup", "files": files_copied, "timestamp": timestamp}
            _save_status(status)
            return status

        commit_result = subprocess.run(
            ["git", "commit", "-m", f"Nightly backup {timestamp} — {len(files_copied)} files"],
            cwd=CLONE_DIR, capture_output=True, text=True, timeout=10
        )

        push_result = subprocess.run(
            ["git", "push", "origin", BRANCH],
            cwd=CLONE_DIR, capture_output=True, text=True, timeout=30
        )

        if push_result.returncode != 0:
            return {"status": "error", "reason": f"Push failed: {push_result.stderr[:200]}"}

        status = {
            "status": "success",
            "files": files_copied,
            "count": len(files_copied),
            "timestamp": timestamp,
            "branch": BRANCH,
        }
        _save_status(status)
        return status

    except subprocess.TimeoutExpired:
        return {"status": "error", "reason": "Git operation timed out"}
    except Exception as e:
        return {"status": "error", "reason": str(e)[:200]}
    finally:
        # Clean up clone
        if os.path.exists(CLONE_DIR):
            shutil.rmtree(CLONE_DIR, ignore_errors=True)


def get_status():
    """Get the last backup status."""
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"status": "never_run"}


def _save_status(status):
    """Persist backup status."""
    try:
        with open(STATUS_FILE, "w") as f:
            json.dump(status, f)
    except Exception:
        pass


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        print(json.dumps(get_status(), indent=2))
    else:
        result = run_backup()
        print(json.dumps(result, indent=2))
