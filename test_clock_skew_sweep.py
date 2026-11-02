"""Tests for scripts/clock_skew_sweep.py parsing + bomb logic, and the preload."""
import os, subprocess, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
import clock_skew_sweep as c

TAP = """TAP version 13
ok 1 - parse: inner subtest names are ignored
not ok 2 - server/a.test.ts
  ---
  error: 'test failed'
not ok 3 - inner subtest, not a file
not ok 4 - server/b.test.ts
# fail 2
"""

def test_parse_only_top_level_test_files():
    assert c.parse_tap_failures(TAP) == {"server/a.test.ts", "server/b.test.ts"}

def test_bombs_are_new_failures_with_smallest_skew():
    base = {"server/old.test.ts"}
    by = {45: {"server/old.test.ts", "server/x.test.ts"},
          120: {"server/old.test.ts", "server/x.test.ts", "server/y.test.ts"}}
    assert c.find_bombs(base, by) == {"server/x.test.ts": 45, "server/y.test.ts": 120}

def test_no_bombs_when_skew_adds_nothing():
    assert c.find_bombs({"a"}, {30: {"a"}}) == {}

def test_preload_shifts_real_clock_not_explicit_dates():
    code = ("const n=Date.now(),r=new Date().getTime();"
            "console.log(Math.round((n-%d)/86400000), new Date('2020-01-01T00:00:00Z').toISOString())")
    import time
    out = subprocess.run(
        ["node", "-e", code % int(time.time() * 1000)],
        env={**os.environ, "SKEW_DAYS": "30", "NODE_OPTIONS": f"--require {c.PRELOAD}"},
        capture_output=True, text=True).stdout.split()
    assert out[0] == "30" and out[1] == "2020-01-01T00:00:00.000Z"

def test_skew_incompatible_files_carry_a_reason():
    assert "server/gatedTests.test.ts" in c.SKEW_INCOMPATIBLE
    assert all(r.strip() for r in c.SKEW_INCOMPATIBLE.values())
