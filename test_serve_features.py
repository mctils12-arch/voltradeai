"""
[REPAIR] 2026-11-03 (approved item B3, T-BOT) — serve-time ML features must
match training. The bot page's Issues panel read "12 / 34 model features never
move at serve time"; read against ml_model_v2._compute_features (training) and
bot_engine.deep_score (serve) those 12 are 6 per-ticker features wrongly
constant at serve + 1 market-wide feature wrongly constant across days + 5
features constant in TRAINING by construction (serve_features module docstring
has the per-feature root causes).

Three guarantees are pinned here:
  1. PARITY — every corrected serve formula equals ml_model_v2._compute_features
     on the same bars (the training builder is the oracle, not a hand copy).
  2. THE TRAINING-CONSTANT SET IS WHAT THE REAL BUILDER PRODUCES — the five
     features serve_features classifies as never-varied-in-training are exactly
     the zero-variance columns of ml_model_v2._build_training_data on a
     synthetic multi-ticker universe, and every other column varies there.
  3. THE HEADLINE — for every feature that varied in that training fixture and
     is per-ticker, the corrected serve vectors of a synthetic scan are NOT all
     identical; the pre-fix vectors (deep_score's constants, as live) ARE.
Deterministic (seeded), no network, no files.
"""
import unittest
from datetime import date, timedelta

import numpy as np

import serve_features as sf
from ml_model_v2 import FEATURE_COLS, TRAINING_CORE_TICKERS, _build_training_data, _compute_features


def _bars(n, seed, base=100.0, drift=0.0005, vol=0.02, start=date(2025, 1, 1), big_change=None, big_at=None):
    """Deterministic synthetic daily bars (dicts with t/o/h/l/c/v, shared dates).
    `big_change` replaces the return INTO bar `big_at` (the decision bar of a
    scan that selects big movers)."""
    rng = np.random.RandomState(seed)
    c = base
    out = []
    for i in range(n):
        o = c
        r = rng.normal(drift, vol)
        if big_change is not None and i == big_at:
            r = big_change
        c = c * (1 + r)
        u = abs(rng.normal(0, 0.01))
        h = max(o, c) * (1 + u)
        l = min(o, c) * (1 - u)
        out.append({"t": (start + timedelta(days=i)).isoformat() + "T05:00:00Z",
                    "o": o, "h": h, "l": l, "c": c, "v": int(1_000_000 + rng.randint(0, 5_000_000))})
    return out


def _vxx_bars(n, seed, start=date(2025, 1, 1)):
    """VXX with a +-30 % swing so the training vrp / put_call_proxy take every value."""
    rng = np.random.RandomState(seed)
    out = []
    for i in range(n):
        c = 20.0 * (1 + 0.3 * np.sin(i / 9.0)) * (1 + rng.normal(0, 0.01))
        out.append({"t": (start + timedelta(days=i)).isoformat() + "T05:00:00Z",
                    "o": c, "h": c * 1.01, "l": c * 0.99, "c": c, "v": 5_000_000})
    return out


def _spy_bars(n, seed, start=date(2025, 1, 1)):
    """SPY with alternating +-0.4 %/day drift segments so markov_state takes 0/1/2."""
    rng = np.random.RandomState(seed)
    c = 500.0
    out = []
    for i in range(n):
        seg = (i // 20) % 3
        drift = (0.004, -0.004, 0.0)[seg]
        o = c
        c = c * (1 + drift + rng.normal(0, 0.003))
        out.append({"t": (start + timedelta(days=i)).isoformat() + "T05:00:00Z",
                    "o": o, "h": max(o, c) * 1.002, "l": min(o, c) * 0.998, "c": c, "v": 50_000_000})
    return out


N = 320
IDX = N - 6   # the last index _compute_features will score (it needs 5 forward bars)
STOCK = _bars(N, 1)
SPY = _spy_bars(N, 2)
VXX = _vxx_bars(N, 3)
TLT = _bars(N, 4, base=90.0, vol=0.01)
HYG = _bars(N, 5, base=80.0, vol=0.005)


def _training_row(stock=STOCK, idx=IDX):
    return _compute_features(stock, idx, {}, "T", VXX, SPY, 0.0, cross_sec_rank=0.5,
                             news_sentiment=0.0, insider_signal=0.0, tlt_bars=TLT, hyg_bars=HYG,
                             breadth_pct=0.5)


class TestModeFlag(unittest.TestCase):
    def test_default_is_shadow(self):
        self.assertEqual(sf.serve_fix_mode({}), "shadow")
        self.assertEqual(sf.serve_fix_mode({sf.ENV_VAR: ""}), "shadow")
        self.assertEqual(sf.serve_fix_mode({sf.ENV_VAR: "shadow"}), "shadow")

    def test_off_and_on_spellings(self):
        for v in ("off", "OFF", "0", "false", "none"):
            self.assertEqual(sf.serve_fix_mode({sf.ENV_VAR: v}), "off", v)
        for v in ("on", "ON", "1", "true", "live"):
            self.assertEqual(sf.serve_fix_mode({sf.ENV_VAR: v}), "on", v)

    def test_unrecognised_value_falls_back_to_the_default_not_to_on(self):
        self.assertEqual(sf.serve_fix_mode({sf.ENV_VAR: "yes please"}), "shadow")


class TestFeatureKinds(unittest.TestCase):
    def test_every_training_column_has_exactly_one_kind(self):
        kinds = {c: sf.feature_kind(c) for c in FEATURE_COLS}
        self.assertEqual(set(kinds.values()), {"per_ticker", "market_wide", "training_constant"})
        self.assertEqual({c for c, k in kinds.items() if k == "training_constant"}, set(sf.TRAINING_CONSTANT_FEATURES))
        self.assertEqual({c for c, k in kinds.items() if k == "market_wide"}, set(sf.MARKET_WIDE_FEATURES))
        self.assertTrue(set(sf.TRAINING_CONSTANT_FEATURES).isdisjoint(sf.MARKET_WIDE_FEATURES))

    def test_fixed_features_are_real_columns_and_never_training_constant(self):
        for name in sf.FIXED_FEATURES:
            self.assertIn(name, FEATURE_COLS)
            self.assertNotIn(name, sf.TRAINING_CONSTANT_FEATURES)
        self.assertEqual(len(set(sf.FIXED_FEATURES)), 7)
        self.assertIn("credit_spread", sf.MARKET_WIDE_FEATURES)
        self.assertEqual([f for f in sf.FIXED_FEATURES if sf.feature_kind(f) == "market_wide"], ["credit_spread"])


class TestTrainingFormulaParity(unittest.TestCase):
    """The training builder is the oracle: the serve formula on bars[:idx+1]
    must reproduce _compute_features(bars, idx)."""

    def setUp(self):
        self.row = _training_row()
        self.assertIsNotNone(self.row)
        self.sub = STOCK[:IDX + 1]
        self.closes = [b["c"] for b in self.sub]
        self.chg = (self.closes[-1] - self.closes[-2]) / self.closes[-2] * 100

    def test_price_vs_52w_high(self):
        got = sf.price_vs_52w_high([b["h"] for b in self.sub], self.closes[-1])
        self.assertAlmostEqual(round(got, 2), self.row["price_vs_52w_high"], places=9)

    def test_ewma_vol(self):
        self.assertAlmostEqual(round(sf.ewma_vol_20(self.closes), 3), self.row["ewma_vol"], places=9)

    def test_range_pct(self):
        got = sf.range_pct(self.sub[-1]["h"], self.sub[-1]["l"], self.closes[-1])
        self.assertAlmostEqual(round(got, 3), self.row["range_pct"], places=9)

    def test_sector_momentum(self):
        got = sf.sector_momentum_5d(self.closes, [b["c"] for b in SPY[:IDX + 1]])
        self.assertAlmostEqual(round(got, 3), self.row["sector_momentum"], places=9)

    def test_trend_strength(self):
        got = sf.trend_strength(self.chg, sf.ewma_vol_20(self.closes))
        self.assertAlmostEqual(round(got, 3), self.row["trend_strength"], places=9)

    def test_credit_spread(self):
        got = sf.credit_spread_21d([b["c"] for b in TLT[:IDX + 1]], [b["c"] for b in HYG[:IDX + 1]])
        self.assertAlmostEqual(round(got, 4), self.row["credit_spread"], places=9)

    def test_cross_sec_rank_is_the_builders_rank_rule(self):
        # _build_training_data: rank_pct = sum(x <= stock_chg) / len(day_changes) when > 10 names
        day = [float(x) for x in range(-5, 7)]   # 12 moves incl. the stock's own 2.0
        self.assertAlmostEqual(sf.cross_sec_rank(2.0, day), sum(1 for x in day if x <= 2.0) / len(day))
        self.assertIsNone(sf.cross_sec_rank(2.0, day[:10]), "10 names or fewer -> training keeps 0.5; serve keeps today's value")

    def test_fixed_vector_reproduces_the_training_row_for_every_fixed_feature(self):
        current = {k: 0.0 for k in FEATURE_COLS}
        current.update({"ewma_vol": 2.0, "cross_sec_rank": 0.5, "change_pct_today": self.chg})
        pop = [float(x) for x in range(-5, 6)]  # 11 other moves
        fixed, diff = sf.fixed_feature_vector(
            current, symbol="T", bars=self.sub, closes=self.closes, change_pct=self.chg,
            price=self.closes[-1], spy_closes=[b["c"] for b in SPY[:IDX + 1]],
            scan_changes={f"S{i}": x for i, x in enumerate(pop)},
            credit_spread_fixed=sf.credit_spread_21d([b["c"] for b in TLT[:IDX + 1]], [b["c"] for b in HYG[:IDX + 1]]))
        for name in ("price_vs_52w_high", "ewma_vol", "range_pct", "sector_momentum", "trend_strength", "credit_spread"):
            self.assertAlmostEqual(fixed[name], self.row[name], places=9, msg=name)
        self.assertAlmostEqual(fixed["cross_sec_rank"], round(sum(1 for x in pop + [self.chg] if x <= self.chg) / 12, 3))
        self.assertEqual(set(diff), set(sf.FIXED_FEATURES))
        for name, d in diff.items():
            self.assertEqual(d["current"], current[name], name)
            self.assertEqual(d["fixed"], fixed[name], name)
        # untouched columns are untouched
        for name in FEATURE_COLS:
            if name not in sf.FIXED_FEATURES:
                self.assertEqual(fixed[name], current[name], name)


class TestFixedVectorHonesty(unittest.TestCase):
    def test_missing_inputs_keep_the_current_value_and_report_no_diff(self):
        current = {"range_pct": 0.0, "ewma_vol": 2.0, "cross_sec_rank": 0.5, "credit_spread": 0.0,
                   "price_vs_52w_high": 0.0, "sector_momentum": 0.0, "trend_strength": 5.0, "change_pct_today": 12.0}
        fixed, diff = sf.fixed_feature_vector(current)
        self.assertEqual(diff, {})
        self.assertEqual(fixed, current)

    def test_unchanged_value_is_not_a_diff(self):
        closes = [100.0 + i for i in range(30)]
        ev = round(sf.ewma_vol_20(closes), 3)
        fixed, diff = sf.fixed_feature_vector({"ewma_vol": ev, "change_pct_today": 1.0, "trend_strength": round(min(1.0 / max(ev, 0.1), 5.0), 3)}, closes=closes)
        self.assertNotIn("ewma_vol", diff)
        self.assertNotIn("trend_strength", diff)

    def test_trend_strength_follows_the_corrected_ewma_vol(self):
        closes = [100.0 * (1 + 0.03 * ((-1) ** i)) for i in range(30)]   # ~3 % daily swings
        fixed, diff = sf.fixed_feature_vector({"ewma_vol": 2.0, "trend_strength": 5.0, "change_pct_today": 10.0}, closes=closes)
        self.assertIn("ewma_vol", diff)
        self.assertAlmostEqual(fixed["trend_strength"], round(min(10.0 / max(fixed["ewma_vol"], 0.1), 5.0), 3))
        self.assertNotEqual(fixed["trend_strength"], 5.0)

    def test_cross_sec_population_prefers_the_training_universe_and_counts_the_candidate_once(self):
        scan = {f"X{i}": float(i) for i in range(40)}
        scan["ME"] = 99.0   # the scan's own (stale) entry for the candidate
        universe = [f"X{i}" for i in range(12)] + ["ME", "NOT_IN_SCAN"]
        pop = sf.cross_sec_population(3.5, scan, universe, symbol="ME")
        self.assertEqual(sorted(pop), sorted([float(i) for i in range(12)] + [3.5]))
        # too few universe names in the scan -> the whole scan (minus the stale own entry) + own move
        pop2 = sf.cross_sec_population(3.5, scan, ["X0", "X1"], symbol="ME")
        self.assertEqual(len(pop2), 41)
        self.assertNotIn(99.0, pop2)
        self.assertIn(3.5, pop2)
        # a population under the training floor is no population
        self.assertIsNone(sf.cross_sec_population(3.5, {f"X{i}": 1.0 for i in range(9)}, None, symbol="ME"))
        self.assertIsNone(sf.cross_sec_population(3.5, None, None))

    def test_bad_bars_do_not_fabricate(self):
        self.assertIsNone(sf.range_pct(1.0, 2.0, 100.0))        # high below low
        self.assertIsNone(sf.range_pct(None, 1.0, 100.0))
        self.assertIsNone(sf.price_vs_52w_high([1, 2, 3], 2.0))  # fewer than training's 6 bars
        self.assertIsNone(sf.ewma_vol_20([100.0]))
        self.assertIsNone(sf.credit_spread_21d([1.0] * 21, [1.0] * 22))
        self.assertIsNone(sf.sector_momentum_5d([1.0] * 5, [1.0] * 6))


def _training_fixture():
    """A synthetic universe through the REAL training builder: 12 names
    (> 10, so cross_sec_rank and market_breadth are computed per date) plus
    the reference series. Returns (X, var_by_column)."""
    all_bars = {"SPY": SPY, "VXX": VXX, "TLT": TLT, "HYG": HYG}
    for i in range(12):
        all_bars[f"T{i:02d}"] = _bars(N, 100 + i, base=50.0 + 10 * i, drift=0.001 * ((-1) ** i), vol=0.015 + 0.002 * i)
    X, y, regimes = _build_training_data(all_bars)
    assert X is not None and len(X) > 500
    var = {col: float(np.var(X[:, j])) for j, col in enumerate(FEATURE_COLS)}
    return X, var


class TestTrainingConstantSetMatchesTheRealBuilder(unittest.TestCase):
    def test_zero_variance_columns_are_exactly_the_training_constant_set(self):
        _X, var = _training_fixture()
        zero = {c for c, v in var.items() if v == 0.0}
        self.assertEqual(zero, set(sf.TRAINING_CONSTANT_FEATURES),
                         "serve_features.TRAINING_CONSTANT_FEATURES must be exactly the columns "
                         "ml_model_v2._build_training_data never varies")
        for c in FEATURE_COLS:
            if c not in sf.TRAINING_CONSTANT_FEATURES:
                self.assertGreater(var[c], 0.0, f"{c} should vary in the training fixture")


class TestConstantAcrossScanWhenTrainingVaried(unittest.TestCase):
    """THE HEADLINE TEST. A synthetic scan of 8 candidates, each with its own
    bars, scored against one macro snapshot. Pre-fix vectors carry deep_score's
    serve-time constants exactly as live (0 / 2 / 0 / 0.5 / 0 / cap / 0.0);
    post-fix vectors come from serve_features. For every feature that varied
    in the training fixture AND is per-ticker, the post-fix vectors across the
    scan must not all be identical."""

    N_CAND = 8

    @classmethod
    def setUpClass(cls):
        cls.X, cls.var = _training_fixture()
        cls.spy_closes = [b["c"] for b in SPY[:IDX + 1]]
        cls.credit = sf.credit_spread_21d([b["c"] for b in TLT[:IDX + 1]], [b["c"] for b in HYG[:IDX + 1]])
        rng = np.random.RandomState(7)
        # the scan's own change distribution (30 liquid names) — training's floor
        cls.scan_changes = {f"U{i:02d}": float(rng.normal(0, 2.5)) for i in range(30)}
        cls.cands = []
        for i in range(cls.N_CAND):
            # the scan selects big movers: every candidate moved >= 10 % today,
            # which is what saturated trend_strength at its cap live
            bars = _bars(N, 500 + i, base=30.0 + 7 * i, vol=0.02 + 0.003 * i, big_change=0.10 + 0.01 * i, big_at=IDX)
            sub = bars[:IDX + 1]
            closes = [b["c"] for b in sub]
            chg = (closes[-1] - closes[-2]) / closes[-2] * 100
            row = _compute_features(bars, IDX, {}, f"C{i}", VXX, SPY, 0.0, cross_sec_rank=0.5,
                                    news_sentiment=0.0, insider_signal=0.0, tlt_bars=TLT, hyg_bars=HYG,
                                    breadth_pct=0.5)
            assert row is not None
            current = dict(row)
            # serve-side per-ticker values for the columns training derives from
            # VXX but deep_score derives per name (analyze VRP / per-stock HV rank)
            vrp = float(rng.uniform(-10, 15))
            current.update({"vrp": round(vrp, 3), "put_call_proxy": -1.0 if vrp > 8 else (1.0 if vrp < -5 else 0.0),
                            "iv_rank_proxy": float(rng.uniform(0, 100)), "iv_rank_stock": float(rng.uniform(0, 100)),
                            "vwap_position": float(i % 2), "above_ma10": float((i // 2) % 2)})
            # deep_score's constants, exactly as served live (serve_features docstring)
            current.update({"price_vs_52w_high": 0.0, "ewma_vol": 2.0, "range_pct": 0.0, "cross_sec_rank": 0.5,
                            "sector_momentum": 0.0, "trend_strength": round(min(abs(chg) / 2.0, 5.0), 3),
                            "credit_spread": 0.0})
            cls.scan_changes[f"C{i}"] = chg
            cls.cands.append({"symbol": f"C{i}", "bars": sub, "closes": closes, "chg": chg, "current": current})

    def _vectors(self, fixed):
        out = []
        for c in self.cands:
            if not fixed:
                out.append(c["current"])
                continue
            vec, _diff = sf.fixed_feature_vector(
                c["current"], symbol=c["symbol"], bars=c["bars"], closes=c["closes"], change_pct=c["chg"],
                price=c["closes"][-1], spy_closes=self.spy_closes, scan_changes=self.scan_changes,
                universe_symbols=None, credit_spread_fixed=self.credit)
            out.append(vec)
        return out

    @staticmethod
    def _constant_across(vectors, name):
        return len({round(float(v[name]), 6) for v in vectors}) == 1

    def test_pre_fix_vectors_show_the_live_defect(self):
        vectors = self._vectors(fixed=False)
        for name in sf.FIXED_FEATURES:
            self.assertTrue(self._constant_across(vectors, name), f"{name} should be constant pre-fix (as live)")
            if name != "credit_spread":
                self.assertGreater(self.var[name], 0.0, f"{name} varies in training — so pre-fix it is a serve defect")

    def test_no_per_ticker_feature_that_varied_in_training_is_constant_across_the_fixed_scan(self):
        vectors = self._vectors(fixed=True)
        offenders = [c for c in FEATURE_COLS
                     if self.var[c] > 0.0 and sf.feature_kind(c) == "per_ticker" and self._constant_across(vectors, c)]
        self.assertEqual(offenders, [], f"constant across the scan although training varied them: {offenders}")

    def test_market_wide_and_training_constant_features_are_identical_across_a_scan_by_nature(self):
        """Why ml_insight judges these two kinds differently: one macro snapshot
        per scan means every market-wide feature is identical for all
        candidates even when nothing is wrong; the training-constant five
        never varied anywhere."""
        vectors = self._vectors(fixed=True)
        for c in FEATURE_COLS:
            if sf.feature_kind(c) in ("market_wide", "training_constant"):
                self.assertTrue(self._constant_across(vectors, c), c)

    def test_fix_changes_every_targeted_feature_and_nothing_else(self):
        for c in self.cands:
            vec, diff = sf.fixed_feature_vector(
                c["current"], symbol=c["symbol"], bars=c["bars"], closes=c["closes"], change_pct=c["chg"],
                price=c["closes"][-1], spy_closes=self.spy_closes, scan_changes=self.scan_changes,
                universe_symbols=None, credit_spread_fixed=self.credit)
            # the six formula features always change from their constants; trend_strength
            # changes only when the corrected |chg| / ewma_vol leaves the 5.0 cap (C0: 10 % / ~2 %
            # stays capped — an honest "unchanged", not a missed fix)
            self.assertTrue(set(diff) <= set(sf.FIXED_FEATURES), c["symbol"])
            self.assertTrue(set(sf.FIXED_FEATURES) - {"trend_strength"} <= set(diff), c["symbol"])
            # training computes trend from the UNROUNDED ewma_vol and rounds the result
            self.assertEqual(vec["trend_strength"], round(sf.trend_strength(c["chg"], sf.ewma_vol_20(c["closes"])), 3))
            for name in FEATURE_COLS:
                if name not in sf.FIXED_FEATURES:
                    self.assertEqual(vec[name], c["current"][name], name)
            # the corrected values are the training formulas on this candidate's own bars
            row = _compute_features(c["bars"] + STOCK[IDX + 1:], IDX, {}, c["symbol"], VXX, SPY, 0.0,
                                    cross_sec_rank=0.5, news_sentiment=0.0, insider_signal=0.0,
                                    tlt_bars=TLT, hyg_bars=HYG, breadth_pct=0.5)
            for name in ("price_vs_52w_high", "ewma_vol", "range_pct", "sector_momentum", "trend_strength", "credit_spread"):
                self.assertAlmostEqual(vec[name], row[name], places=9, msg=f"{c['symbol']} {name}")

    def test_core_universe_constant_is_the_training_list(self):
        self.assertEqual(len(TRAINING_CORE_TICKERS), 30)
        self.assertIn("SPY", TRAINING_CORE_TICKERS)
        self.assertIn("AAPL", TRAINING_CORE_TICKERS)


if __name__ == "__main__":
    unittest.main()
