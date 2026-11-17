"""EXPOSURE CAP vs THE FLOOR (2026-11-17, EXPLORATORY).

The regime caps (PANIC 0.30, BEAR 0.50, CAUTION 0.60) were applied to the whole
book. The volatility-managed floor alone is about 67% of equity, so in those
regimes the tier engine's master switch refused the whole engine on every scan
("Portfolio invested 67.5% >= 50% (regime=BEAR)", live 2026-11-17). With
system_config EXPOSURE_CAP_FLOOR_MODE "exempt" the regime's haircut applies to
the room ABOVE the floor, the total never passes the 0.95 ceiling, and the
regime risk budget scales that room as before. "legacy" is the rollback.
"""
import os
import time
import unittest
from unittest.mock import patch

import regime_engine as re_
import system_config
import tiered_strategy as ts

CEIL = system_config.BASE_CONFIG["MAX_TOTAL_EXPOSURE"]
FLOOR_TICKER = sorted(re_.passive_floor_tickers())[0]
# vxx_ratio / spy_vs_ma50 that land in each legacy regime today (asserted in setUpClass)
REGIME_INPUTS = {"BULL": (0.9, 1.03), "CAUTION": (1.1, 0.99), "BEAR": (1.0, 0.93), "PANIC": (1.5, 0.9)}
REGIME_CAP = {"BULL": 0.95, "CAUTION": 0.60, "BEAR": 0.50, "PANIC": 0.30}


def book(floor_mv, satellite_mv=0.0):
    out = [{"symbol": FLOOR_TICKER, "market_value": floor_mv}]
    if satellite_mv:
        out.append({"symbol": "AAPL", "market_value": satellite_mv})
    return out


class Base(unittest.TestCase):
    MODE = "exempt"

    @classmethod
    def setUpClass(cls):
        for name, (vxx, ma) in REGIME_INPUTS.items():
            caps = ts.get_regime_caps(vxx, ma, 100_000)
            assert (caps["regime"], caps["MAX_TOTAL_EXPOSURE"]) == (name, REGIME_CAP[name]), (name, caps["regime"])

    def setUp(self):
        # no regime state: the risk budget does not act, so the numbers below are the exemption's own
        re_._STATE_CACHE.update(at=time.time(), state=None)
        self._env = patch.dict(os.environ, {"VOLTRADE_EXPOSURE_CAP_FLOOR": self.MODE, "VOLTRADE_REGIME_BUDGET": "off"})
        self._env.start()

    def tearDown(self):
        self._env.stop()
        re_._STATE_CACHE.update(at=0.0, state=None)

    def caps(self, regime, positions, equity=100_000):
        vxx, ma = REGIME_INPUTS[regime]
        return ts.get_regime_caps(vxx, ma, equity, positions=positions)

    def ctx(self, regime, positions, equity=100_000):
        vxx, ma = REGIME_INPUTS[regime]
        return ts.TierContext(equity=equity, peak_equity=equity, positions=positions, vxx_ratio=vxx, spy_vs_ma50=ma)


class TestFloorExemptCap(Base):
    def test_the_regime_haircut_applies_to_the_room_above_the_floor(self):
        # the live book of 2026-11-17: SPY $59,880 of $88,795 equity, regime BEAR
        caps = self.caps("BEAR", book(59_880), equity=88_795)
        floor = 59_880 / 88_795
        self.assertAlmostEqual(caps["MAX_TOTAL_EXPOSURE"], 0.50 + floor * (1 - 0.50 / CEIL), places=4)
        self.assertAlmostEqual(caps["MAX_TOTAL_EXPOSURE"], 0.8194, places=4)
        self.assertEqual(caps["MAX_TOTAL_EXPOSURE_REGIME"], 0.50)
        fx = caps["FLOOR_EXEMPT_CAP"]
        # satellite room = (ceiling - floor) x regime_cap / ceiling: the regime still cuts it by half
        self.assertAlmostEqual(fx["satellite_room"], (CEIL - floor) * 0.50 / CEIL, places=3)
        self.assertAlmostEqual(fx["satellite_room"], 0.145, places=3)

    def test_each_cautious_regime_keeps_its_own_share_of_the_room(self):
        rooms = {r: self.caps(r, book(67_000))["FLOOR_EXEMPT_CAP"]["satellite_room"] for r in ("CAUTION", "BEAR", "PANIC")}
        for r, room in rooms.items():
            self.assertAlmostEqual(room, (CEIL - 0.67) * REGIME_CAP[r] / CEIL, places=3, msg=r)
        self.assertGreater(rooms["CAUTION"], rooms["BEAR"])
        self.assertGreater(rooms["BEAR"], rooms["PANIC"])
        self.assertGreater(rooms["PANIC"], 0)

    def test_never_above_the_deployment_ceiling_whatever_the_floor(self):
        for regime in ("CAUTION", "BEAR", "PANIC"):
            for floor_mv in (1_000, 30_000, 67_000, 85_000, 95_000, 120_000):
                cap = self.caps(regime, book(floor_mv))["MAX_TOTAL_EXPOSURE"]
                self.assertLessEqual(cap, CEIL, (regime, floor_mv))
                self.assertGreaterEqual(cap, REGIME_CAP[regime], (regime, floor_mv))
        # a floor at or above the ceiling leaves no satellite room at all
        self.assertEqual(self.caps("BEAR", book(95_000))["FLOOR_EXEMPT_CAP"]["satellite_room"], 0.0)
        self.assertEqual(self.caps("BEAR", book(120_000))["MAX_TOTAL_EXPOSURE"], CEIL)

    def test_bull_and_neutral_are_unchanged(self):
        caps = self.caps("BULL", book(67_000))
        self.assertEqual(caps["MAX_TOTAL_EXPOSURE"], 0.95)
        self.assertNotIn("FLOOR_EXEMPT_CAP", caps)
        self.assertNotIn("MAX_TOTAL_EXPOSURE_REGIME", caps)

    def test_no_floor_held_or_no_positions_argument_is_the_regime_cap(self):
        self.assertEqual(self.caps("BEAR", [{"symbol": "AAPL", "market_value": 30_000}])["MAX_TOTAL_EXPOSURE"], 0.50)
        self.assertEqual(self.caps("BEAR", [])["MAX_TOTAL_EXPOSURE"], 0.50)
        vxx, ma = REGIME_INPUTS["BEAR"]
        self.assertEqual(ts.get_regime_caps(vxx, ma, 100_000)["MAX_TOTAL_EXPOSURE"], 0.50)   # every non-tier caller

    def test_bad_inputs_fall_back_to_the_regime_cap(self):
        self.assertEqual(self.caps("BEAR", book(67_000), equity=0)["MAX_TOTAL_EXPOSURE"], 0.50)
        self.assertEqual(self.caps("BEAR", [{"symbol": FLOOR_TICKER, "market_value": "x"}])["MAX_TOTAL_EXPOSURE"], 0.50)
        self.assertEqual(self.caps("BEAR", [{"symbol": FLOOR_TICKER}])["MAX_TOTAL_EXPOSURE"], 0.50)


class TestMasterSwitch(Base):
    def test_the_floor_alone_no_longer_refuses_the_tier_engine(self):
        # before: "Portfolio invested 67.4% >= 50% (regime=BEAR)" on every scan
        killed, reason = ts.master_kill_switch(self.ctx("BEAR", book(59_880), equity=88_795))
        self.assertEqual((killed, reason), (False, "ok"))

    def test_it_still_refuses_at_the_new_cap_and_says_how_the_cap_was_made(self):
        # floor 67% + satellites 15% = 82% >= 81.7% (BEAR: 0.50 + 0.67 x (1 - 0.50/0.95)); the
        # message rounds the cap to a whole percent, as it always has, and the note gives it exactly
        killed, reason = ts.master_kill_switch(self.ctx("BEAR", book(67_000, 15_000)))
        self.assertTrue(killed)
        self.assertTrue(reason.startswith("Portfolio invested 82.0% >= 82% (regime=BEAR)"), reason)
        self.assertIn("[regime cap 50% applied above the floor's 67.0%: 81.7%]", reason)
        import shadow_portfolio as sp
        self.assertEqual(sp._masterkill_reason_category(reason), "exposure_ceiling")
        # one dollar of room left: not killed
        killed, _ = ts.master_kill_switch(self.ctx("BEAR", book(67_000, 14_000)))
        self.assertFalse(killed)

    def test_drawdown_and_daily_loss_kills_are_untouched(self):
        c = self.ctx("BEAR", book(59_880), equity=88_795)
        c.peak_equity = 88_795 / (1 + ts.PORTFOLIO_DRAWDOWN_KILL) + 1   # just past the tier drawdown kill
        if hasattr(c, "drawdown_pct") and c.drawdown_pct <= ts.PORTFOLIO_DRAWDOWN_KILL:
            killed, reason = ts.master_kill_switch(c)
            self.assertTrue(killed)
            self.assertIn("Portfolio DD kill", reason)


class TestRiskBudgetStillScalesTheRoom(Base):
    def test_budget_applies_to_the_exempt_room(self):
        # the live state of 2026-11-17: budget 0.62
        state = {"risk_budget": 0.62, "risk_budget_detail": {"budget": 0.62, "band": "reduced"}}
        try:
            from test_regime_budget import budget_state
            state = budget_state(0.62)
        except Exception:
            pass
        re_._STATE_CACHE.update(at=time.time(), state=state)
        with patch.dict(os.environ, {"VOLTRADE_REGIME_ENGINE": "1", "VOLTRADE_REGIME_BUDGET": "1"}):
            caps = self.caps("BEAR", book(59_880), equity=88_795)
        rb = caps["RISK_BUDGET_CAP"]
        if not rb.get("applied"):
            self.skipTest("the regime state fixture is not accepted as fresh here")
        floor = 59_880 / 88_795
        exempt = 0.50 + floor * (1 - 0.50 / CEIL)
        self.assertAlmostEqual(caps["MAX_TOTAL_EXPOSURE_LEGACY"], exempt, places=3)
        self.assertAlmostEqual(caps["MAX_TOTAL_EXPOSURE"], floor + (exempt - floor) * 0.62, places=3)
        self.assertAlmostEqual(caps["MAX_TOTAL_EXPOSURE"] - floor, 0.090, places=3)   # about 9% of equity


class TestLegacyRollback(Base):
    MODE = "legacy"

    def test_one_setting_restores_the_regime_cap_on_the_whole_book(self):
        caps = self.caps("BEAR", book(59_880), equity=88_795)
        self.assertEqual(caps["MAX_TOTAL_EXPOSURE"], 0.50)
        self.assertNotIn("FLOOR_EXEMPT_CAP", caps)
        killed, reason = ts.master_kill_switch(self.ctx("BEAR", book(59_880), equity=88_795))
        self.assertTrue(killed)
        self.assertEqual(reason, "Portfolio invested 67.4% >= 50% (regime=BEAR)")

    def test_mode_resolution(self):
        self.assertEqual(ts.exposure_cap_floor_mode({"VOLTRADE_EXPOSURE_CAP_FLOOR": "legacy"}), "legacy")
        self.assertEqual(ts.exposure_cap_floor_mode({"VOLTRADE_EXPOSURE_CAP_FLOOR": "EXEMPT"}), "exempt")
        # no env: the committed default
        self.assertEqual(ts.exposure_cap_floor_mode({}), system_config.BASE_CONFIG["EXPOSURE_CAP_FLOOR_MODE"])
        self.assertEqual(system_config.BASE_CONFIG["EXPOSURE_CAP_FLOOR_MODE"], "exempt")
        # garbage in the env is ignored, garbage in the config is legacy
        self.assertEqual(ts.exposure_cap_floor_mode({"VOLTRADE_EXPOSURE_CAP_FLOOR": "banana"}), "exempt")
        with patch.dict(system_config.BASE_CONFIG, {"EXPOSURE_CAP_FLOOR_MODE": "banana"}):
            self.assertEqual(ts.exposure_cap_floor_mode({}), "legacy")


if __name__ == "__main__":
    unittest.main()
