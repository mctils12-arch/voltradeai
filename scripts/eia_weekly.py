#!/usr/bin/env python3
"""scripts/eia_weekly.py: compatibility shim with no logic (2026-11-25, open_questions (ah4)).

The module lives at the repo root (eia_weekly.py). The production Docker image
copies the root *.py files and never scripts/, so code the server runs
(snapshot_refresh_job.py, the
nightly refresh of the served snapshots) cannot live here.
This shim keeps `python3 scripts/eia_weekly.py ...` and the sibling research
scripts that load this path working: run as a script it executes the root file
as __main__; loaded by path it loads the root file by path and re-exports its
namespace.
"""
import importlib.util as _ilu
import os as _os
import runpy as _runpy

_ROOT_FILE = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "eia_weekly.py")

if __name__ == "__main__":
    _runpy.run_path(_ROOT_FILE, run_name="__main__")
else:
    _spec = _ilu.spec_from_file_location("eia_weekly", _ROOT_FILE)
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    globals().update({_k: _v for _k, _v in vars(_mod).items()
                      if not (_k.startswith("__") and _k.endswith("__"))})
