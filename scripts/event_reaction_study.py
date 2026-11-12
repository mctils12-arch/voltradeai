#!/usr/bin/env python3
"""
scripts/event_reaction_study.py — CLI shim for the PRE-REGISTERED
event-reaction study (research/event_reaction_study.md §1).

The study's code moved to the repo-root module event_reaction.py on
2026-11-06 (the production image ships only root *.py, and the
server-side delisting-inclusive panel re-run, event_reaction_panel.py, imports
it). This file keeps the documented commands working unchanged:
  python3 scripts/event_reaction_study.py fetch [--cache DIR]
  python3 scripts/event_reaction_study.py run [--cache DIR] [--workers 6]
  python3 scripts/event_reaction_study.py report [--cache DIR]
and re-exports every public name, so loading this file (as
test_event_reaction_study.py does) yields the same functions and constants.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from event_reaction import *  # noqa: E402,F401,F403 — the shim's whole job is re-export
from event_reaction import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
