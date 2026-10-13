#!/usr/bin/env python3
"""
VolTradeAI — floor regime dwell regression tests (2026-10-13).

Defect: bot_engine._manage_spy_floor (FLOOR_BASKET branch) re-derived the
regime from get_market_regime() on EVERY scan and re-targeted at once.
SPY_FLOOR_BULL 0.70 vs SPY_FLOOR_NEUTRAL 0.90 x QQQ basket weight 0.50 is a
10%-of-equity QQQ step — bigger than FLOOR_BASKET_DRIFT_THRESHOLD (3%) — and
vxx_ratio sat on regime_util's BULL boundary (0.95) with no hysteresis, so
~12 QQQ (~$9k) were bought/sold on every BULL<->NEUTRAL flip (live
2026-10-12: 7 round trips). _manage_defensive_floor's regime_changed keyed
off the same flickering label, and a fallback macro (VXX fetch failed ->
vxx_ratio=1.0 -> NEUTRAL) moved the floors as if it were a real reading.

Fix pinned here: both floors act on one debounced "floor regime" persisted
in voltrade_floor_state.json — a calmer label must be seen on
FLOOR_REGIME_CONFIRM_CYCLES consecutive calls; escalations into
CAUTION/BEAR/PANIC apply on the first call; a fallback macro holds the last
confirmed regime. All Alpaca I/O is mocked; get_market_regime is REAL.
"""
import os
import sys
import json
import tempfile
import unittest
from unittest.mock import patch, MagicMock

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import bot_engine  # noqa: E402
import system_config  # noqa: E402

EQUITY = 100000.0
PRICES = {"QQQ": 500.0, "SMH": 250.0, "KWEB": 30.0, "VXUS": 60.0, "GLD": 200.0}

# QQQ comfortably above both MAs so the trend-override branch neither fires
# nor fetches bars. vxx_closes/vxx_latest mark a REAL VXX reading.
_BASE_MACRO = {
    "qqq_price": 500.0, "qqq_ma50": 480.0, "qqq_ma200": 460.0,
    "spy_vs_ma50": 1.02, "spy_below_200_days": 0, "spy_above_200d": True,
    "vxx_closes": [20.0] * 30, "vxx_latest": 20.0,
}
BULL_MACRO = {**_BASE_MACRO, "vxx_ratio": 0.94}      # just under BULL 0.95
NEUTRAL_MACRO = {**_BASE_MACRO, "vxx_ratio": 0.97}   # just over it
BEAR_MACRO = {**_BASE_MACRO, "vxx_ratio": 1.20}
# VXX fetch failed: macro_data substitutes vxx_ratio=1.0 with no closes.
FALLBACK_MACRO = {**_BASE_MACRO, "vxx_ratio": 1.0, "vxx_closes": []}
FALLBACK_MACRO.pop("vxx_latest")
CAUTION_MACRO = {**_BASE_MACRO, "vxx_ratio": 1.08}


def _snap(macro, ts):
    """One macro_data snapshot: _save_cache stamps every result with
    "timestamp" and get_macro_snapshot() serves that same dict for
    CACHE_TTL (300s), so every scan inside the TTL sees the same stamp."""
    return {**macro, "timestamp": float(ts)}


def _cfg():
    cfg = dict(system_config.BASE_CONFIG)
    cfg["FLOOR_BASKET_ENABLED"] = True
    cfg["DEFENSIVE_FLOOR_ENABLED"] = True
    cfg["FLOOR_REGIME_CONFIRM_CYCLES"] = 3
    return cfg


def _basket_positions(regime, gld_pct=None):
    """Positions sitting exactly on the basket target for `regime`."""
    cfg = _cfg()
    target = cfg[f"SPY_FLOOR_{regime}"]
    pos = []
    for t, w in cfg["FLOOR_BASKET"].items():
        if t == "CASH" or target <= 0:
            continue
        mv = EQUITY * target * w
        pos.append({"symbol": t, "qty": str(int(mv / PRICES[t])),
                    "market_value": str(mv), "asset_class": "us_equity"})
    gld = cfg[f"DEFENSIVE_FLOOR_{regime}"] if gld_pct is None else gld_pct
    if gld > 0:
        mv = EQUITY * gld
        pos.append({"symbol": "GLD", "qty": str(int(mv / PRICES["GLD"])),
                    "market_value": str(mv), "asset_class": "us_equity"})
    return pos


def _mock_get(url, headers=None, params=None, timeout=None):
    m = MagicMock()
    if url.endswith("/v2/account"):
        m.json.return_value = {"equity": str(EQUITY)}
    elif url.endswith("/v2/stocks/snapshots"):
        sym = (params or {}).get("symbols")
        m.json.return_value = {sym: {"latestTrade": {"p": PRICES[sym]}}}
    elif url.endswith("/snapshot"):
        sym = url.rsplit("/", 2)[-2]
        m.json.return_value = {"latestTrade": {"p": PRICES[sym]}}
    else:
        raise AssertionError(f"unexpected GET {url}")
    return m


class _Base(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="vt_floor_dwell_")
        self.positions = []
        for p in [
            patch.object(bot_engine, "DATA_DIR", self._tmpdir),
            patch.object(bot_engine, "data_feed", return_value="iex"),
            patch.object(bot_engine, "_is_regular_hours", return_value=True),
            patch("system_config.BASE_CONFIG", _cfg()),
            patch.object(bot_engine, "get_alpaca_account",
                         return_value={"equity": str(EQUITY)}),
            patch.object(bot_engine, "get_alpaca_positions",
                         side_effect=lambda: list(self.positions)),
            patch.object(bot_engine.requests, "get", side_effect=_mock_get),
        ]:
            p.start()
            self.addCleanup(p.stop)
        _post_patch = patch.object(bot_engine.requests, "post")
        self.post = _post_patch.start()
        self.addCleanup(_post_patch.stop)

    def _seed(self, path_name, state):
        with open(os.path.join(self._tmpdir, path_name), "w") as f:
            json.dump(state, f)

    def _state(self):
        with open(os.path.join(self._tmpdir, "voltrade_floor_state.json")) as f:
            return json.load(f)

    def _posted(self, symbol=None):
        out = [c.kwargs["json"] for c in self.post.call_args_list]
        return [j for j in out if symbol is None or j.get("symbol") == symbol]


class TestRegimeFixturesClassifyAsIntended(unittest.TestCase):
    def test_fixture_labels(self):
        gmr = system_config.get_market_regime
        for macro, want in ((BULL_MACRO, "BULL"), (NEUTRAL_MACRO, "NEUTRAL"),
                            (BEAR_MACRO, "BEAR"), (FALLBACK_MACRO, "NEUTRAL"),
                            (CAUTION_MACRO, "CAUTION")):
            self.assertEqual(gmr(macro["vxx_ratio"], macro["spy_vs_ma50"],
                                 spy_below_200_days=0, spy_above_200d=True), want)


class TestFlickerDoesNotChurnFloor(_Base):
    def test_alternating_bull_neutral_places_no_qqq_orders(self):
        self.positions = _basket_positions("BULL")
        # Initial state: first observation is BULL, holdings already on target.
        bot_engine._manage_spy_floor(BULL_MACRO)
        self.post.reset_mock()

        regimes = []
        for i in range(6):
            macro = NEUTRAL_MACRO if i % 2 == 0 else BULL_MACRO
            regimes.append(bot_engine._manage_spy_floor(macro).get("regime"))

        self.assertEqual(self._posted("QQQ"), [],
                         "regime flicker must not buy/sell the QQQ floor leg")
        self.assertEqual(self._posted(), [], "no basket orders on flicker")
        self.assertEqual(regimes, ["BULL"] * 6)

    def test_defensive_floor_regime_does_not_flicker(self):
        self.positions = _basket_positions("BULL")
        bot_engine._manage_spy_floor(BULL_MACRO)
        for i in range(6):
            macro = NEUTRAL_MACRO if i % 2 == 0 else BULL_MACRO
            fr = bot_engine._resolve_floor_regime(macro)
            bot_engine._manage_spy_floor(macro, fr)
            res = bot_engine._manage_defensive_floor(macro, fr)
            self.assertEqual(res.get("regime"), "BULL", f"flip {i}")
        with open(os.path.join(self._tmpdir,
                               "voltrade_defensive_floor_state.json")) as f:
            self.assertEqual(json.load(f).get("last_regime"), "BULL")


class TestSustainedCalmerRegimeConfirms(_Base):
    def test_neutral_takes_effect_on_third_distinct_reading(self):
        self.positions = _basket_positions("BULL")
        bot_engine._manage_spy_floor(_snap(BULL_MACRO, 1000))
        self.post.reset_mock()

        for n in (1, 2):
            res = bot_engine._manage_spy_floor(_snap(NEUTRAL_MACRO, 1000 + 300 * n))
            self.assertEqual(res["regime"], "BULL")
            self.assertEqual(self._state()["floor_regime_pending_count"], n)
        self.assertEqual(self._posted(), [])

        res = bot_engine._manage_spy_floor(_snap(NEUTRAL_MACRO, 1900))
        self.assertEqual(res["regime"], "NEUTRAL")
        buys = self._posted("QQQ")
        self.assertEqual(len(buys), 1)
        self.assertEqual(buys[0]["side"], "buy")
        # 10% of equity step at $500 = 20 shares.
        self.assertEqual(buys[0]["qty"], "20")

    def test_shared_resolution_advances_once_per_scan(self):
        self.positions = _basket_positions("BULL")
        bot_engine._manage_spy_floor(BULL_MACRO)
        fr = bot_engine._resolve_floor_regime(NEUTRAL_MACRO)
        bot_engine._manage_spy_floor(NEUTRAL_MACRO, fr)
        bot_engine._manage_defensive_floor(NEUTRAL_MACRO, fr)
        self.assertEqual(self._state()["floor_regime_pending_count"], 1)
        self.assertEqual(self._state()["floor_regime"], "BULL")


class TestEscalationIsImmediate(_Base):
    def test_neutral_to_bear_acts_on_first_call(self):
        self._seed("voltrade_floor_state.json",
                   {"floor_regime": "NEUTRAL", "last_regime": "NEUTRAL"})
        self._seed("voltrade_defensive_floor_state.json",
                   {"last_regime": "NEUTRAL"})
        self.positions = _basket_positions("NEUTRAL")

        fr = bot_engine._resolve_floor_regime(BEAR_MACRO)
        self.assertEqual(fr["regime"], "BEAR")
        self.assertEqual(fr["confirmed_by"], "escalation")

        spy = bot_engine._manage_spy_floor(BEAR_MACRO, fr)
        self.assertEqual(spy["regime"], "BEAR")
        qqq = self._posted("QQQ")
        self.assertEqual(len(qqq), 1)
        self.assertEqual(qqq[0]["side"], "sell")

        dfl = bot_engine._manage_defensive_floor(BEAR_MACRO, fr)
        self.assertEqual(dfl["regime"], "BEAR")
        gld = self._posted("GLD")
        self.assertEqual(len(gld), 1)
        self.assertEqual(gld[0]["side"], "buy")

    def test_escalation_immediate_without_shared_resolution(self):
        self._seed("voltrade_floor_state.json",
                   {"floor_regime": "NEUTRAL", "last_regime": "NEUTRAL"})
        self.positions = _basket_positions("NEUTRAL")
        res = bot_engine._manage_spy_floor(BEAR_MACRO)
        self.assertEqual(res["regime"], "BEAR")
        self.assertEqual([j["side"] for j in self._posted("QQQ")], ["sell"])

    def test_de_escalation_from_bear_is_debounced(self):
        self._seed("voltrade_floor_state.json",
                   {"floor_regime": "BEAR", "last_regime": "BEAR"})
        fr = bot_engine._resolve_floor_regime(NEUTRAL_MACRO)
        self.assertEqual(fr["regime"], "BEAR")
        self.assertEqual(fr["confirmed_by"], "pending")


class TestCachedSnapshotCountsOnce(_Base):
    """Review block (2026-10-13): Tier-2 scans every 60s (open) / 120s
    (power hour) but macro_data caches one snapshot for 300s, so one VXX
    reading is seen on 2-5 consecutive scans. The debounce must count
    distinct readings, not scans."""

    def test_same_snapshot_on_many_scans_does_not_confirm(self):
        self.positions = _basket_positions("BULL")
        bot_engine._manage_spy_floor(_snap(BULL_MACRO, 1000))
        self.post.reset_mock()

        one_neutral_reading = _snap(NEUTRAL_MACRO, 1300)
        for _ in range(5):          # 1-minute cadence inside one 300s TTL
            res = bot_engine._manage_spy_floor(dict(one_neutral_reading))
            self.assertEqual(res["regime"], "BULL")
        self.assertEqual(self._posted(), [],
                         "one cached NEUTRAL reading must not trade the floor")
        st = self._state()
        self.assertEqual(st["floor_regime"], "BULL")
        self.assertEqual(st["floor_regime_pending"], "NEUTRAL")
        self.assertEqual(st["floor_regime_pending_count"], 1)

    def test_live_sequence_cached_flicker_places_no_orders(self):
        """Scans every 60s for 60 min; the snapshot refreshes every 300s and
        its label alternates BULL/NEUTRAL (vxx_ratio on the 0.95 boundary)."""
        self.positions = _basket_positions("BULL")
        bot_engine._manage_spy_floor(_snap(BULL_MACRO, 0))
        self.post.reset_mock()
        regimes = set()
        for minute in range(1, 61):
            snap_idx = minute // 5
            base = NEUTRAL_MACRO if snap_idx % 2 else BULL_MACRO
            fr = bot_engine._resolve_floor_regime(_snap(base, snap_idx * 300))
            regimes.add(bot_engine._manage_spy_floor(_snap(base, snap_idx * 300), fr)["regime"])
            bot_engine._manage_defensive_floor(_snap(base, snap_idx * 300), fr)
        self.assertEqual(regimes, {"BULL"})
        self.assertEqual(self._posted("QQQ"), [])
        self.assertEqual(self._posted(), [])

    def test_repeats_between_distinct_readings_neither_advance_nor_reset(self):
        self.positions = _basket_positions("BULL")
        bot_engine._manage_spy_floor(_snap(BULL_MACRO, 0))
        self.post.reset_mock()
        seq = [300, 300, 300, 600, 600]          # two distinct NEUTRAL readings
        for ts in seq:
            self.assertEqual(
                bot_engine._manage_spy_floor(_snap(NEUTRAL_MACRO, ts))["regime"], "BULL")
        self.assertEqual(self._state()["floor_regime_pending_count"], 2)
        self.assertEqual(self._posted(), [])
        res = bot_engine._manage_spy_floor(_snap(NEUTRAL_MACRO, 900))   # third distinct
        self.assertEqual(res["regime"], "NEUTRAL")
        self.assertEqual([(j["side"], j["qty"]) for j in self._posted("QQQ")],
                         [("buy", "20")])

    def test_untimestamped_identical_reading_counts_once(self):
        self._seed("voltrade_floor_state.json",
                   {"floor_regime": "BULL", "last_regime": "BULL"})
        for _ in range(4):
            fr = bot_engine._resolve_floor_regime(dict(NEUTRAL_MACRO))
            self.assertEqual(fr["regime"], "BULL")
        self.assertEqual(self._state()["floor_regime_pending_count"], 1)

    def test_escalation_still_immediate_on_cached_snapshot(self):
        self._seed("voltrade_floor_state.json",
                   {"floor_regime": "NEUTRAL", "last_regime": "NEUTRAL",
                    "floor_regime_last_obs": "ts:1300.0"})
        fr = bot_engine._resolve_floor_regime(_snap(BEAR_MACRO, 1300))
        self.assertEqual(fr["regime"], "BEAR")
        self.assertEqual(fr["confirmed_by"], "escalation")


class TestDeEscalationDoesNotStall(_Base):
    def test_alternating_calmer_labels_confirm_toward_least_calm(self):
        self._seed("voltrade_floor_state.json",
                   {"floor_regime": "BEAR", "last_regime": "BEAR"})
        labels = []
        for i, base in enumerate((CAUTION_MACRO, NEUTRAL_MACRO, CAUTION_MACRO)):
            labels.append(bot_engine._resolve_floor_regime(_snap(base, 300 * (i + 1)))["regime"])
        self.assertEqual(labels, ["BEAR", "BEAR", "CAUTION"])

    def test_opposite_side_reading_resets_pending(self):
        self._seed("voltrade_floor_state.json",
                   {"floor_regime": "NEUTRAL", "last_regime": "NEUTRAL"})
        bot_engine._resolve_floor_regime(_snap(BULL_MACRO, 300))
        bot_engine._resolve_floor_regime(_snap(BULL_MACRO, 600))
        self.assertEqual(self._state()["floor_regime_pending_count"], 2)
        bot_engine._resolve_floor_regime(_snap(NEUTRAL_MACRO, 900))  # == confirmed
        st = self._state()
        self.assertEqual(st["floor_regime"], "NEUTRAL")
        self.assertIsNone(st["floor_regime_pending"])
        self.assertEqual(st["floor_regime_pending_count"], 0)


class TestFallbackMacroHoldsConfirmedRegime(_Base):
    def test_vxx_fallback_does_not_move_floor(self):
        self._seed("voltrade_floor_state.json",
                   {"floor_regime": "BULL", "last_regime": "BULL",
                    "floor_regime_pending": "NEUTRAL",
                    "floor_regime_pending_count": 2})
        self.positions = _basket_positions("BULL")
        res = bot_engine._manage_spy_floor(FALLBACK_MACRO)
        self.assertEqual(res["regime"], "BULL")
        self.assertEqual(self._posted(), [])
        st = self._state()
        self.assertEqual(st["floor_regime"], "BULL")
        # A fallback reading must not count toward confirming a change.
        self.assertEqual(st["floor_regime_pending_count"], 0)

    def test_data_quality_failure_marks_fallback(self):
        self.assertTrue(bot_engine._floor_macro_vxx_is_fallback(
            {**BULL_MACRO, "data_quality_failures": ["vxx_ratio"]}))
        self.assertTrue(bot_engine._floor_macro_vxx_is_fallback(FALLBACK_MACRO))
        self.assertFalse(bot_engine._floor_macro_vxx_is_fallback(BULL_MACRO))


if __name__ == "__main__":
    unittest.main()
