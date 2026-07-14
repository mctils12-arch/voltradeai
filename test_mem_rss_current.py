"""
KNOWN BROKEN #21 root-cause fix (2026-07-14) — regression test for
_mem_rss_mb() reading CURRENT process RSS instead of the historical peak.

Prior to this fix, _mem_rss_mb() read
resource.getrusage(RUSAGE_SELF).ru_maxrss, a per-process high-water mark
that never decreases for the life of the process (POSIX semantics). Every
memory-pressure guard in bot_engine.py (SURVIVAL_MODE @700MB, the
pre_deep_score skip/trim guard @130MB/100MB) compares against that value
expecting "current" pressure, so once RSS peaked past a threshold even
once, the guard stayed tripped for the rest of the process's uptime —
permanently disabling deep_score()'s 5-source enrichment fetchers
(macro/intel/alt/social/finnhub) and get_stock_details() itself. This
matches the live symptom KNOWN BROKEN #21 confirmed: wikipedia/gdelt/fred
caches absent for a multi-day span with dataSourceErrors staying
permanently empty (get_stock_details never even reached).

This test pins the fix: the function must report actual current RSS (from
/proc/self/status on Linux, what Railway runs), which can be LOWER than a
prior peak, and must fall back to the old ru_maxrss path when /proc is
unavailable (e.g. local macOS dev).
"""
import unittest
from unittest.mock import patch, mock_open

import bot_engine


class TestMemRssCurrent(unittest.TestCase):
    def test_reads_current_rss_from_proc_status(self):
        proc_status = "VmPeak:\t  900000 kB\nVmRSS:\t  204800 kB\nVmSize:\t 900000 kB\n"
        with patch("builtins.open", mock_open(read_data=proc_status)):
            mb = bot_engine._mem_rss_mb()
        self.assertEqual(mb, 200)  # 204800 KB / 1024 = 200 MB

    def test_current_rss_can_be_lower_than_a_prior_peak(self):
        """The regression pin: a high historical peak (ru_maxrss) must NOT
        leak into the result once /proc/self/status reports a lower
        current value — exactly the scenario that permanently wedged the
        mem-pressure guards before this fix."""
        proc_status = "VmRSS:\t  102400 kB\n"  # current = 100MB, under both guards
        with patch("builtins.open", mock_open(read_data=proc_status)), \
             patch("resource.getrusage") as mock_getrusage:
            mock_getrusage.return_value.ru_maxrss = 800 * 1024  # stale peak: 800MB
            mb = bot_engine._mem_rss_mb()
        self.assertEqual(mb, 100)
        mock_getrusage.assert_not_called()  # /proc path wins; ru_maxrss never consulted

    def test_falls_back_to_ru_maxrss_when_proc_unavailable(self):
        with patch("builtins.open", side_effect=FileNotFoundError), \
             patch("resource.getrusage") as mock_getrusage:
            mock_getrusage.return_value.ru_maxrss = 150 * 1024  # KB, Linux-style
            mb = bot_engine._mem_rss_mb()
        self.assertEqual(mb, 150)

    def test_returns_zero_when_both_paths_fail(self):
        with patch("builtins.open", side_effect=FileNotFoundError), \
             patch("resource.getrusage", side_effect=Exception("no resource module")):
            mb = bot_engine._mem_rss_mb()
        self.assertEqual(mb, 0)

    def test_malformed_proc_status_falls_back_cleanly(self):
        """No VmRSS line at all (unexpected /proc format) must not crash —
        falls through to the ru_maxrss path rather than raising."""
        with patch("builtins.open", mock_open(read_data="VmPeak:\t 900000 kB\n")), \
             patch("resource.getrusage") as mock_getrusage:
            mock_getrusage.return_value.ru_maxrss = 130 * 1024
            mb = bot_engine._mem_rss_mb()
        self.assertEqual(mb, 130)


if __name__ == "__main__":
    unittest.main()
