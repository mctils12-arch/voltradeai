"""
Durability ratchet, Python side (audit 2026-07-05).

The Railway volume mounts at /data; storage_config.DATA_DIR is the one
sanctioned resolver (/data/voltrade with /tmp fallback for local dev).
The audit found intraday_shorts.py resolving its "persistent" log via a
bare DATA_DIR env read (unset on Railway -> /tmp) — its short-trade log
silently reset on every redeploy. These tests pin the fix and prevent the
class from recurring.

Run: python3 -m pytest test_durability.py -q
"""
import os
import re
import unittest

REPO = os.path.dirname(os.path.abspath(__file__))

# Runtime modules whose ONLY writes are regenerable /tmp caches by design.
TMP_BY_DESIGN = {
    "macro_data.py", "alt_data.py", "social_data.py", "finnhub_data.py",
    "institutional_data.py", "etf_analyzer.py", "etf_data_sources.py",
    "voltrade_daemon.py",  # /tmp trace file; real state via storage_config
    "bot_backtest.py", "backtest_v2.py",  # offline .bt_cache, not live loop
}

# The dangerous pattern: resolving a data dir from the env with a /tmp
# default INSTEAD of importing storage_config (env unset on Railway).
BARE_ENV_DATA_DIR = re.compile(
    r"""os\.environ\.get\(\s*['"]DATA_DIR['"]\s*,\s*['"]/tmp['"]?\s*\)""")


class TestDurability(unittest.TestCase):
    def test_intraday_shorts_log_is_durable(self):
        """The audit's stray: SHORTS_LOG_PATH must resolve under
        storage_config.DATA_DIR (the volume), not a bare /tmp default."""
        import storage_config
        import intraday_shorts
        self.assertTrue(
            intraday_shorts.SHORTS_LOG_PATH.startswith(storage_config.DATA_DIR),
            f"intraday shorts log at {intraday_shorts.SHORTS_LOG_PATH} — "
            f"expected under {storage_config.DATA_DIR}")

    def test_no_runtime_module_uses_bare_env_data_dir_with_tmp_default(self):
        """No runtime module may resolve DATA_DIR from the env with a /tmp
        default unless it ALSO imports storage_config as the primary source
        (the env read is then a test-override, which is fine)."""
        offenders = []
        for f in os.listdir(REPO):
            if not f.endswith(".py") or f.startswith("test_"):
                continue
            if f in TMP_BY_DESIGN:
                continue
            src = open(os.path.join(REPO, f), encoding="utf-8",
                       errors="replace").read()
            if BARE_ENV_DATA_DIR.search(src) and "storage_config" not in src:
                offenders.append(f)
        self.assertEqual(offenders, [],
            "modules resolve DATA_DIR from the env with a /tmp default and "
            "no storage_config import — their 'persistent' files silently "
            f"land in /tmp on Railway: {offenders}")

    def test_storage_config_prefers_the_volume(self):
        """The mount contract: when /data exists, DATA_DIR is under it."""
        import storage_config
        if os.path.isdir("/data"):
            self.assertTrue(storage_config.DATA_DIR.startswith("/data/"))
        else:
            self.assertTrue(storage_config.DATA_DIR.startswith("/tmp"))


if __name__ == "__main__":
    unittest.main()
