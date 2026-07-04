# pytest collection config — makes the constitutional local gate
# (`python3 -m pytest -q`, CLAUDE.md promotion rule 1) runnable repo-wide.
#
# Two root-level files carry the test_ prefix but are STANDALONE SCRIPTS
# with their own runners, not pytest suites. Collecting them breaks the
# entire gate (KNOWN BROKEN #6, resolved 2026-07-04):
#
# - test_auto_discovery.py: executes its whole discovery protocol at
#   import and ends with sys.exit() -> pytest INTERNALERROR that kills
#   collection for every other file. Run it directly:
#   `python3 test_auto_discovery.py`.
# - test_full_system.py: defines a module-level helper `def test(phase,
#   name, fn)` that pytest collects as a test and then fails to resolve
#   `phase` as a fixture. Run it directly: `python3 test_full_system.py`.
#
# Excluding them here removes NO assertions from the gate — neither file
# can execute under pytest at all; both remain runnable as scripts.
# test_collection_health.py is the regression guard that keeps future
# collection breakers out of the gate.
collect_ignore = [
    "test_auto_discovery.py",
    "test_full_system.py",
]
