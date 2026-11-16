#!/usr/bin/env python3
"""scripts/earnings_language_gate2.py: compatibility shim with no logic (KNOWN BROKEN #44 class, 2026-11-16).

The module lives at the repo root (earnings_language_gate2.py). The production Docker image
copies the root *.py files and never scripts/, so code the server runs (the
earnings_qoq_weekly job runs earnings_language_qoq_gate2.py, which imports the
level script and its robustness gate) cannot live here.
This shim keeps `python3 scripts/earnings_language_gate2.py ...` and the sibling research
scripts that load this path working: run as a script it executes the root file
as __main__; loaded by path it loads the root file by path and re-exports its
namespace.
"""
import importlib.util as _ilu
import os as _os
import runpy as _runpy

_ROOT_FILE = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "earnings_language_gate2.py")

if __name__ == "__main__":
    _runpy.run_path(_ROOT_FILE, run_name="__main__")
else:
    _spec = _ilu.spec_from_file_location("earnings_language_gate2", _ROOT_FILE)
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    globals().update({_k: _v for _k, _v in vars(_mod).items()
                      if not (_k.startswith("__") and _k.endswith("__"))})
