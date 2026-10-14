"""Unit tests for scripts/kuramoto_sync_probe.py — synthetic data only, no
network (repo convention for foreign-field probes)."""

import importlib.util
import math
import os
import random
import unittest

_ROOT = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "kuramoto_sync_probe", os.path.join(_ROOT, "scripts", "kuramoto_sync_probe.py"))
ksp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ksp)


def _osc(n, period, phase0=0.0, amp=0.02):
    return [100 * math.exp(amp * math.sin(2 * math.pi * t / period + phase0) + 0.0003 * t)
            for t in range(n)]


class TestOrderParameter(unittest.TestCase):
    def test_identical_phases_give_one(self):
        self.assertAlmostEqual(ksp.order_parameter([0.7] * 9), 1.0, places=9)

    def test_uniform_phases_give_near_zero(self):
        ph = [2 * math.pi * k / 8 for k in range(8)]
        self.assertLess(ksp.order_parameter(ph), 1e-9)

    def test_ignores_none_and_needs_two(self):
        self.assertIsNone(ksp.order_parameter([None, 0.1]))
        self.assertAlmostEqual(ksp.order_parameter([None, 0.1, 0.1]), 1.0, places=9)


class TestPhases(unittest.TestCase):
    def test_warmup_is_none_then_defined(self):
        ph = ksp.phases(_osc(120, 40))
        self.assertTrue(all(p is None for p in ph[: ksp.EMA_SLOW + ksp.DERIV_LAG]))
        self.assertIsNotNone(ph[-1])

    def test_causal_no_lookahead(self):
        full = _osc(150, 40)
        a = ksp.phases(full)
        b = ksp.phases(full[:100])
        self.assertEqual(a[:100], b)

    def test_locked_oscillators_sync_more_than_scattered(self):
        n = 200
        locked = [ksp.phases(_osc(n, 40, 0.0)) for _ in range(6)]
        scattered = [ksp.phases(_osc(n, 40, 2 * math.pi * k / 6)) for k in range(6)]
        r_l = ksp.sync_series(locked)[-1]
        r_s = ksp.sync_series(scattered)[-1]
        self.assertGreater(r_l, 0.99)
        self.assertLess(r_s, r_l - 0.5)


class TestRows(unittest.TestCase):
    def test_targets_strictly_after_decision_day(self):
        random.seed(1)
        n = 300
        etfs = [_osc(n, 40, k * 0.3) for k in range(4)]
        spy = [100 * math.exp(0.0004 * t + 0.001 * random.gauss(0, 1) * t ** 0.5) for t in range(n)]
        rows = ksp.build_rows(etfs, spy)
        r = rows[10]
        t = r["t"]
        self.assertAlmostEqual(r["fwd_ret"], math.log(spy[t + ksp.HORIZON] / spy[t]), places=9)
        self.assertEqual(rows[-1]["t"], n - ksp.HORIZON - 1)


class TestPartialSpearman(unittest.TestCase):
    def test_control_explains_away_spurious_link(self):
        random.seed(7)
        n = 400
        z = [random.gauss(0, 1) for _ in range(n)]
        x = [v + 0.3 * random.gauss(0, 1) for v in z]
        y = [v + 0.3 * random.gauss(0, 1) for v in z]
        raw = ksp.raw_spearman(x, y)
        part = ksp.partial_spearman(x, y, [z])
        self.assertGreater(raw["rho"], 0.7)
        self.assertLess(abs(part["rho"]), 0.15)

    def test_real_link_survives_control(self):
        random.seed(8)
        n = 400
        z = [random.gauss(0, 1) for _ in range(n)]
        x = [random.gauss(0, 1) for _ in range(n)]
        y = [xi + 0.5 * zi + 0.3 * random.gauss(0, 1) for xi, zi in zip(x, z)]
        self.assertGreater(ksp.partial_spearman(x, y, [z])["rho"], 0.7)

    def test_too_few_returns_none(self):
        self.assertIsNone(ksp.partial_spearman([1, 2, 3], [1, 2, 3], []))

    def test_gate2_bar(self):
        self.assertTrue(ksp.passes_gate2({"rho": -0.31, "p_value": 0.01, "n": 90}))
        self.assertFalse(ksp.passes_gate2({"rho": 0.29, "p_value": 0.01, "n": 90}))
        self.assertFalse(ksp.passes_gate2({"rho": 0.5, "p_value": 0.2, "n": 90}))
        self.assertFalse(ksp.passes_gate2(None))


if __name__ == "__main__":
    unittest.main()
