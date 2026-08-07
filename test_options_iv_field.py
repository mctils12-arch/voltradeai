"""IV field-read ratchet (repair 2026-08-06, full-code-review finding,
adversarially verified before fixing).

The defect: options_scanner._fetch_options_chain read IV from
greeks["iv"] — a field Alpaca option snapshots do not carry (IV is
TOP-LEVEL `impliedVolatility`; greeks = delta/gamma/rho/theta/vega).
Every contract therefore scored iv=0, avg_iv was always 0, and the
earnings-IV-crush (avg_iv >= 0.40) and high-IV-premium (avg_iv >= 0.30)
setup gates rejected every candidate on every scan since they shipped —
2 of the 3 HIGH_EDGE_SETUPS silently dead. Same wrong key in
options_manager._get_option_snapshot (display/roll economics).

Fixtures mirror the real Alpaca snapshot shape (the repo's own
options_execution.py:626 and optionsChainArchive.ts fixtures agree).
"""
import unittest
from unittest import mock

import options_scanner
import options_manager


def _snapshot_fixture(iv: float) -> dict:
    # Real shape: impliedVolatility TOP-LEVEL, greeks WITHOUT iv.
    return {
        "latestQuote": {"bp": 4.10, "ap": 4.30, "bs": 12, "as": 9},
        "greeks": {"delta": -0.32, "gamma": 0.02, "theta": -0.05, "vega": 0.11, "rho": -0.01},
        "impliedVolatility": iv,
        "dailyBar": {"v": 850, "c": 4.20},
    }


class _Resp:
    status_code = 200
    def __init__(self, payload): self._p = payload
    def json(self): return self._p


class TestScannerChainIv(unittest.TestCase):
    def test_chain_contract_carries_top_level_implied_volatility(self):
        occ = "AAPL260918P00145000"  # plain root, 15-char body
        payload = {"snapshots": {occ: _snapshot_fixture(0.62)}, "next_page_token": None}
        options_scanner._chain_cache.clear()
        options_scanner._chain_cache_ts.clear()
        with mock.patch.object(options_scanner.requests, "get", return_value=_Resp(payload)), \
             mock.patch.object(options_scanner.alpaca_throttle, "acquire", lambda *a, **k: None):
            contracts = options_scanner._fetch_options_chain("AAPL", 150.0)
        self.assertEqual(len(contracts), 1)
        self.assertAlmostEqual(contracts[0]["iv"], 0.62, places=6,
                               msg="IV must come from top-level impliedVolatility — greeks carries no iv")

    def test_setup_gates_can_now_see_real_iv(self):
        # the dead-channel consequence: with the old read this was always 0
        occ = "AAPL260918P00145000"
        payload = {"snapshots": {occ: _snapshot_fixture(0.55)}, "next_page_token": None}
        options_scanner._chain_cache.clear()
        options_scanner._chain_cache_ts.clear()
        with mock.patch.object(options_scanner.requests, "get", return_value=_Resp(payload)), \
             mock.patch.object(options_scanner.alpaca_throttle, "acquire", lambda *a, **k: None):
            contracts = options_scanner._fetch_options_chain("AAPL", 150.0)
        avg_iv = sum(c["iv"] for c in contracts) / len(contracts)
        self.assertGreater(avg_iv, 0.40, "avg_iv over a 55%-IV chain must clear the 0.40 setup gate")


class TestManagerSnapshotIv(unittest.TestCase):
    def test_get_option_snapshot_reads_top_level_iv(self):
        occ = "QQQ260918P00400000"
        payload = {"snapshots": {occ: _snapshot_fixture(0.48)}}
        with mock.patch.object(options_manager.requests, "get", return_value=_Resp(payload)):
            snap = options_manager._get_option_snapshot(occ)
        self.assertAlmostEqual(snap["iv"], 0.48, places=6)
        self.assertAlmostEqual(snap["delta"], -0.32, places=6, msg="greeks fields unaffected")


class TestSourceRatchet(unittest.TestCase):
    """A refactor must never regress to the greeks-only read: every iv read
    in the two repaired files must try top-level impliedVolatility first."""

    def _bad_reads(self, path: str) -> list:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        bad = []
        for i, line in enumerate(lines, 1):
            if 'greeks.get("iv"' in line and "impliedVolatility" not in line:
                bad.append(f"{path}:{i}: {line.strip()}")
        return bad

    def test_no_greeks_only_iv_reads_remain(self):
        bad = self._bad_reads("options_scanner.py") + self._bad_reads("options_manager.py")
        self.assertEqual(bad, [], "greeks-only iv reads (always 0 on Alpaca snapshots): " + "; ".join(bad))


if __name__ == "__main__":
    unittest.main()
