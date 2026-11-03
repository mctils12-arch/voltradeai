"""
[REPAIR] 2026-11-03 (approved item B3, T-BOT) — the SHADOW-FIRST wiring of the
serve-time feature fix inside bot_engine.deep_score, run end to end against
stubbed sources (the same stubbing shape as test_deep_score_credit_spread_cache):

  off     byte-identical to the pre-fix path: one scoring, no `serve_fix` on the
          shadow record, the seven constants served exactly as live
  shadow  (default) the SERVED score, entry_features and the record's `features`
          are byte-identical to `off`; the record additionally carries
          `serve_fix` with the corrected probability and the per-feature diff
  on      the corrected vector is served and recorded; the current one rides
          along in `features_current_diff` / `ml_prob_current`
  error   a failure inside the fix path serves exactly what `off` serves and
          records the error

Plus the record sanitizer, the scan-change publisher, the corrected
credit-spread fetch, and text pins on the wiring.
"""
import os
import re
import unittest
from unittest.mock import patch

import numpy as np

import bot_engine
import serve_features as sf
import shadow_portfolio as sp

CANDS = ("QQAA", "QQAB", "QQAC")           # not in SECTOR_MAP, like most scanned names
SCAN_SYMS = [f"U{i:02d}" for i in range(30)]
_rng = np.random.RandomState(11)
SCAN_CHANGES = {s: float(_rng.normal(0, 2.0)) for s in SCAN_SYMS}
CAND_CHANGE = {"QQAA": 11.0, "QQAB": 12.5, "QQAC": -10.5}   # big movers, as the scan selects


def _bars(n, seed, base, last_change):
    rng = np.random.RandomState(seed)
    c = base
    out = []
    for i in range(n):
        o = c
        r = rng.normal(0.0005, 0.02) if i < n - 1 else last_change
        c = c * (1 + r)
        u = abs(rng.normal(0, 0.01))
        out.append({"t": f"2025-01-{(i % 28) + 1:02d}T05:00:00Z", "o": o, "h": max(o, c) * (1 + u),
                    "l": min(o, c) * (1 - u), "c": c, "v": 2_000_000})
    return out


BARS = {sym: _bars(300, 40 + i, 50.0 + 20 * i, CAND_CHANGE[sym] / 100.0) for i, sym in enumerate(CANDS)}
TLT = [{"c": 90.0 + 0.05 * i} for i in range(45)]
HYG = [{"c": 80.0 - 0.02 * i} for i in range(45)]
SPY_CLOSES = [500.0 + 1.5 * i for i in range(60)]
MACRO = {
    "vix": 18.0, "vix_regime": "medium", "sector_momentum": {}, "market_regime": "neutral",
    "vxx_ratio": 1.05, "vxx_latest": 21.0, "vxx_avg30": 20.0, "vxx_closes": [20.0 + 0.1 * (i % 7) for i in range(30)],
    "spy_vs_ma50": 1.02, "spy_closes_60d": SPY_CLOSES, "spy_change_pct": 0.4, "data_quality": "ok",
}


class _Resp:
    def __init__(self, data):
        self._d = data

    def json(self):
        return self._d

    def raise_for_status(self):
        pass


def _fake_get(url, params=None, headers=None, timeout=None, **kw):
    params = params or {}
    if params.get("symbols") == "TLT,HYG":
        return _Resp({})                      # the live failure: the multi-symbol request feeds nothing
    if "most-actives" in url:
        return _Resp({"most_actives": [{"symbol": s, "volume": 1, "trade_count": 1} for s in SCAN_SYMS]})
    m = re.search(r"/v2/stocks/([A-Z]+)/bars", url)
    if m:
        sym = m.group(1)
        if sym in BARS:
            return _Resp({"bars": BARS[sym], "symbol": sym})
        if sym == "TLT":
            return _Resp({"bars": TLT, "symbol": sym})
        if sym == "HYG":
            return _Resp({"bars": HYG, "symbol": sym})
    return _Resp({})


def _fake_ml_score(features):
    """Deterministic and sensitive to the whole vector."""
    s = sum(round(float(v), 4) * (i + 1) for i, (_k, v) in enumerate(sorted(features.items())))
    prob = 0.5 + ((abs(s) * 1000.0) % 100.0) / 1000.0
    return {"ml_score": round(prob * 100, 1), "ml_confidence": round(abs(prob - 0.5) * 2, 3), "ml_signal": "BUY",
            "model_type": "lightgbm_global", "win_probability": round(prob, 3), "regime_used": "global"}


class _Proc:
    stdout = "{}"
    stderr = ""
    returncode = 0


def _quick(sym):
    return {"ticker": sym, "price": round(BARS[sym][-1]["c"], 2), "close": BARS[sym][-1]["c"],
            "prev_close": BARS[sym][-2]["c"], "volume": 5_000_000, "change_pct": CAND_CHANGE[sym],
            "above_vwap": True, "range_pct": 0, "vwap_dist": 0, "reasons": [], "quick_score": 40}


def _detail():
    return {"price": 100.0, "vrp": 6.5, "recommendation": {}, "sentiment": {}, "edge_factors": {},
            "vol_metrics": {"rsi_14": 58.0, "volume_ratio_5d": 1.4, "ewma_rv": 41.2, "rv20": 38.0}, "valuation": {}}


def _reset_caches():
    bot_engine._CREDIT_SPREAD_CACHE = 0.0
    bot_engine._CREDIT_SPREAD_CACHE_TIME = 0.0
    bot_engine._CREDIT_SPREAD_FIX_CACHE["value"] = None
    bot_engine._CREDIT_SPREAD_FIX_CACHE["time"] = 0.0
    bot_engine._MOST_ACTIVES_CACHE = []
    bot_engine._MOST_ACTIVES_CACHE_TIME = 0.0
    bot_engine._SCAN_CHANGES = {}
    bot_engine._SCAN_CHANGES_TIME = 0.0


class _DeepScoreHarness(unittest.TestCase):
    def setUp(self):
        _reset_caches()
        self.records = []

        def _capture(**kw):
            self.records.append(sp._build_record(
                kw["ticker"], kw["features"], kw["score"], kw["decision"], kw.get("decision_reason", ""),
                kw.get("entry_price", 0.0), kw.get("vxx_ratio", 1.0), kw.get("regime_label", "neutral"),
                kw.get("side", ""), kw.get("action_label", ""), kw.get("trade_type", ""), kw.get("ml"),
                kw.get("score_breakdown"), kw.get("regime_snapshot"), kw.get("shadow_entry"), kw.get("serve_fix")))

        patches = [
            patch("bot_engine.get_stock_details", side_effect=lambda t, _diag=None: _detail()),
            patch("bot_engine.requests.get", side_effect=_fake_get),
            patch("bot_engine.subprocess.run", return_value=_Proc()),
            patch("macro_data.get_macro_snapshot", return_value=dict(MACRO)),
            patch("macro_data.get_news_sentiment", return_value={}),
            patch("intelligence.get_full_intelligence", return_value={}),
            patch("alt_data.get_alt_data_score", return_value={}),
            patch("social_data.get_social_intelligence", return_value={}),
            patch("finnhub_data.get_insider_sentiment", return_value={}),
            patch("finnhub_data.get_recommendation_trends", return_value={}),
            patch("ml_model_v3.shadow_score", return_value=None),
            patch("ml_model_v2.ml_score", side_effect=_fake_ml_score),
            patch("shadow_portfolio.log_candidate", side_effect=_capture),
            patch.dict(os.environ, {"VOLTRADE_REGIME_ENGINE": "off"}),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def _run(self, mode, sym="QQAA"):
        _reset_caches()
        self.records.clear()
        bot_engine.publish_scan_changes([{"ticker": s, "change_pct": c, "volume": 1_000_000} for s, c in SCAN_CHANGES.items()]
                                        + [{"ticker": s, "change_pct": CAND_CHANGE[s], "volume": 5_000_000} for s in CANDS])
        with patch.dict(os.environ, {sf.ENV_VAR: mode}):
            res = bot_engine.deep_score(sym, _quick(sym), min_score=65.0)
        self.assertEqual(len(self.records), 1, "deep_score logs exactly one shadow record per candidate")
        return res, self.records[0]


class TestModesEndToEnd(_DeepScoreHarness):
    def test_off_is_the_pre_fix_path_and_shows_the_live_constants(self):
        res, rec = self._run("off")
        f = res["entry_features"]
        self.assertEqual(f["range_pct"], 0)
        self.assertEqual(f["ewma_vol"], 2)
        self.assertEqual(f["cross_sec_rank"], 0.5)
        self.assertEqual(f["price_vs_52w_high"], 0)
        self.assertEqual(f["credit_spread"], 0.0)
        self.assertEqual(f["sector_momentum"], 0)
        self.assertEqual(f["trend_strength"], 5.0)
        self.assertNotIn("serve_fix", rec)
        self.assertEqual(rec["features"]["range_pct"], 0.0)

    def test_shadow_serves_byte_identical_scores_and_records_the_counterfactual(self):
        off, rec_off = self._run("off")
        shadow, rec = self._run("shadow")
        for k in ("deep_score", "ml_only_score", "rules_only_score", "entry_features", "side", "action_label"):
            self.assertEqual(shadow[k], off[k], k)
        self.assertEqual(rec["features"], rec_off["features"])
        self.assertEqual(rec["ml"]["win_probability"], rec_off["ml"]["win_probability"])
        self.assertEqual(rec["score"], rec_off["score"])
        sfx = rec["serve_fix"]
        self.assertEqual(sfx["mode"], "shadow")
        self.assertEqual(sfx["served"], "current")
        self.assertIsNone(sfx["error"])
        self.assertEqual(set(sfx["features_fixed_diff"]), set(sf.FIXED_FEATURES))
        self.assertEqual(set(sfx["features_current_diff"]), set(sf.FIXED_FEATURES))
        self.assertEqual(sfx["n_changed"], 7)
        self.assertEqual(sfx["ml_prob_current"], rec["ml"]["win_probability"])
        self.assertNotEqual(sfx["ml_prob_fixed"], sfx["ml_prob_current"])
        self.assertEqual(sfx["model_type_fixed"], "lightgbm_global")
        # the recorded diff is the corrected vector the fake model was scored on
        fixed_vec = {**rec["features"], **sfx["features_fixed_diff"]}
        self.assertEqual(sfx["ml_prob_fixed"], _fake_ml_score(fixed_vec)["win_probability"])
        # and the corrected values are not the constants
        d = sfx["features_fixed_diff"]
        self.assertNotEqual(d["range_pct"], 0.0)
        self.assertNotEqual(d["ewma_vol"], 2.0)
        self.assertNotEqual(d["cross_sec_rank"], 0.5)
        self.assertNotEqual(d["price_vs_52w_high"], 0.0)
        self.assertNotEqual(d["credit_spread"], 0.0)
        self.assertNotEqual(d["sector_momentum"], 0.0)
        self.assertLess(d["trend_strength"], 5.0)

    def test_shadow_is_the_default_mode(self):
        res, rec = self._run("")
        self.assertEqual(rec["serve_fix"]["mode"], "shadow")
        self.assertEqual(res["entry_features"]["range_pct"], 0)

    def test_on_serves_the_corrected_vector_and_keeps_the_current_one_recorded(self):
        off, _ = self._run("off")
        _shadow, rec_shadow = self._run("shadow")
        on, rec = self._run("on")
        f = on["entry_features"]
        for name, val in rec_shadow["serve_fix"]["features_fixed_diff"].items():
            self.assertAlmostEqual(f[name], val, places=6, msg=name)
        self.assertEqual(rec["features"], {k: float(v) for k, v in f.items()})
        sfx = rec["serve_fix"]
        self.assertEqual(sfx["served"], "fixed")
        self.assertEqual(sfx["mode"], "on")
        for name, val in sfx["features_current_diff"].items():
            self.assertAlmostEqual(off["entry_features"][name], val, places=6, msg=name)
        self.assertEqual(on["ml_only_score"], _fake_ml_score(f)["ml_score"])
        self.assertEqual(sfx["ml_prob_current"], rec_shadow["ml"]["win_probability"])
        self.assertNotEqual(on["ml_only_score"], off["ml_only_score"])
        self.assertEqual(rec["ml"]["win_probability"], sfx["ml_prob_fixed"])

    def test_a_failing_fix_path_serves_exactly_what_off_serves(self):
        off, _ = self._run("off")
        with patch("serve_features.fixed_feature_vector", side_effect=RuntimeError("boom")):
            res, rec = self._run("on")
        self.assertEqual(res["deep_score"], off["deep_score"])
        self.assertEqual(res["entry_features"], off["entry_features"])
        self.assertEqual(rec["serve_fix"]["served"], "current")
        self.assertTrue(rec["serve_fix"]["error"].startswith("RuntimeError"))
        self.assertEqual(rec["serve_fix"]["features_fixed_diff"], {})

    def test_cross_sec_rank_is_ranked_against_the_scan_universe(self):
        _res, rec = self._run("shadow", sym="QQAC")   # -10.5 %: near the bottom of the population
        rank = rec["serve_fix"]["features_fixed_diff"]["cross_sec_rank"]
        pop = list(SCAN_CHANGES.values()) + [CAND_CHANGE["QQAC"]]   # most-actives subset (30) + own move
        self.assertAlmostEqual(rank, round(sum(1 for x in pop if x <= CAND_CHANGE["QQAC"]) / len(pop), 3))
        _res, rec = self._run("shadow", sym="QQAB")   # +12.5 %: the top
        self.assertEqual(rec["serve_fix"]["features_fixed_diff"]["cross_sec_rank"], 1.0)

    def test_without_a_scan_population_the_rank_keeps_todays_value(self):
        _reset_caches()
        self.records.clear()
        with patch.dict(os.environ, {sf.ENV_VAR: "shadow"}):
            bot_engine.deep_score("QQAA", _quick("QQAA"), min_score=65.0)
        d = self.records[0]["serve_fix"]["features_fixed_diff"]
        self.assertNotIn("cross_sec_rank", d)
        self.assertEqual(self.records[0]["features"]["cross_sec_rank"], 0.5)
        self.assertEqual(len(d), 6)


class TestScanChangesAndCreditSpreadHelpers(unittest.TestCase):
    def setUp(self):
        _reset_caches()

    def test_publish_applies_the_training_volume_floor_and_skips_bad_rows(self):
        n = bot_engine.publish_scan_changes([
            {"ticker": "aaa", "change_pct": 1.5, "volume": 300_000},
            {"ticker": "BBB", "change_pct": -2.0, "volume": 299_999},
            {"ticker": "CCC", "change_pct": None, "volume": 1_000_000},
            {"ticker": "DDD", "change_pct": True, "volume": 1_000_000},
            {"ticker": "", "change_pct": 1.0, "volume": 1_000_000},
            "not a dict",
        ], now=1000.0)
        self.assertEqual(n, 1)
        self.assertEqual(bot_engine.scan_changes_snapshot(now=1001.0), {"AAA": 1.5})
        self.assertIsNone(bot_engine.scan_changes_snapshot(now=1000.0 + sf.SCAN_CHANGES_MAX_AGE_S + 1))

    def test_publish_of_nothing_clears_the_snapshot(self):
        bot_engine.publish_scan_changes([{"ticker": "AAA", "change_pct": 1.0, "volume": 1_000_000}], now=10.0)
        self.assertEqual(bot_engine.publish_scan_changes([], now=11.0), 0)
        self.assertIsNone(bot_engine.scan_changes_snapshot(now=12.0))

    def test_cross_sec_universe_is_most_actives_plus_the_training_core(self):
        bot_engine._MOST_ACTIVES_CACHE = [{"symbol": "zzz"}, {"nope": 1}, "x"]
        u = bot_engine._cross_sec_universe()
        self.assertIn("ZZZ", u)
        self.assertIn("AAPL", u)
        self.assertIn("SPY", u)

    def test_credit_spread_fixed_uses_single_symbol_requests_and_caches_the_value(self):
        urls = []

        def _get(url, params=None, headers=None, timeout=None, **kw):
            urls.append(url)
            return _fake_get(url, params=params, headers=headers, timeout=timeout)

        with patch("bot_engine.requests.get", side_effect=_get):
            v1 = bot_engine._credit_spread_fixed(now=100.0)
            v2 = bot_engine._credit_spread_fixed(now=130.0)
        self.assertEqual(len(urls), 2)
        self.assertTrue(any("/v2/stocks/TLT/bars" in u for u in urls))
        self.assertTrue(any("/v2/stocks/HYG/bars" in u for u in urls))
        self.assertTrue(all("limit=60" in u for u in urls), "a per-symbol limit: the multi-symbol request shared its limit")
        self.assertAlmostEqual(v1, sf.credit_spread_21d([b["c"] for b in TLT], [b["c"] for b in HYG]))
        self.assertEqual(v1, v2, "second call inside the TTL is served from the cache")

    def test_credit_spread_fixed_failure_is_none_and_cached_for_the_cycle(self):
        calls = {"n": 0}

        def _boom(*a, **kw):
            calls["n"] += 1
            raise ConnectionError("down")

        with patch("bot_engine.requests.get", side_effect=_boom):
            self.assertIsNone(bot_engine._credit_spread_fixed(now=100.0))
            self.assertIsNone(bot_engine._credit_spread_fixed(now=120.0))
        self.assertEqual(calls["n"], 1, "a failing endpoint is asked once per cycle, not once per candidate")
        with patch("bot_engine.requests.get", side_effect=_fake_get):
            self.assertIsNotNone(bot_engine._credit_spread_fixed(now=200.0))


class TestShadowRecordServeFix(unittest.TestCase):
    def test_absent_in_off_mode_and_for_older_callers(self):
        rec = sp._build_record("AMD", {"rsi_14": 55.0}, 70.0, "taken", "ok", 100.0)
        self.assertNotIn("serve_fix", rec)
        rec = sp._build_record("AMD", {"rsi_14": 55.0}, 70.0, "taken", "ok", 100.0, serve_fix=None)
        self.assertNotIn("serve_fix", rec)
        self.assertIsNone(sp._sanitize_serve_fix({}))
        self.assertIsNone(sp._sanitize_serve_fix("shadow"))

    def test_sanitized_shape(self):
        out = sp._sanitize_serve_fix({
            "mode": "shadow", "served": "current", "n_changed": 2,
            "features_fixed_diff": {"range_pct": 1.23456789, "ewma_vol": 3.1, "bad": "x", "flag": True},
            "features_current_diff": {"range_pct": 0.0, "ewma_vol": 2.0},
            "ml_prob_current": 0.512345, "ml_prob_fixed": float("nan"), "ml_score_current": 51.2,
            "model_type_current": "lightgbm_global", "unknown_key": 1,
        })
        self.assertEqual(out["mode"], "shadow")
        self.assertEqual(out["served"], "current")
        self.assertEqual(out["n_changed"], 2)
        self.assertEqual(out["features_fixed_diff"], {"range_pct": 1.234568, "ewma_vol": 3.1})
        self.assertEqual(out["features_current_diff"], {"range_pct": 0.0, "ewma_vol": 2.0})
        self.assertEqual(out["ml_prob_current"], 0.5123)
        self.assertIsNone(out["ml_prob_fixed"])
        self.assertIsNone(out["ml_score_fixed"])
        self.assertIsNone(out["model_type_fixed"])
        self.assertIsNone(out["error"])
        self.assertNotIn("unknown_key", out)
        self.assertNotIn("truncated", out)

    def test_oversized_block_keeps_scalars_and_drops_diffs(self):
        huge = {f"f{i}": float(i) for i in range(2000)}
        out = sp._sanitize_serve_fix({"mode": "shadow", "served": "current", "ml_prob_fixed": 0.6,
                                      "features_fixed_diff": huge, "features_current_diff": huge})
        self.assertTrue(out["truncated"])
        self.assertEqual(out["features_fixed_diff"], {})
        self.assertEqual(out["ml_prob_fixed"], 0.6)

    def test_record_carries_the_block_and_the_rest_of_the_shape_is_unchanged(self):
        base = sp._build_record("AMD", {"rsi_14": 55.0}, 70.0, "taken", "ok", 100.0)
        rec = sp._build_record("AMD", {"rsi_14": 55.0}, 70.0, "taken", "ok", 100.0,
                               serve_fix={"mode": "on", "served": "fixed", "ml_prob_fixed": 0.61, "ml_prob_current": 0.55})
        self.assertEqual(rec["serve_fix"]["served"], "fixed")
        self.assertEqual(set(rec) - {"serve_fix", "timestamp"}, set(base) - {"timestamp"})


class TestWiringPins(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bot_engine.py"), encoding="utf-8") as fh:
            cls.src = fh.read()

    def test_deep_score_reads_the_mode_and_passes_the_block_to_the_shadow_record(self):
        self.assertIn("_sf_mode = _sf_mode_fn()", self.src)
        self.assertIn("serve_fix=_serve_fix if '_serve_fix' in locals() else None,", self.src)
        # off mode is the one plain scoring call, exactly the pre-fix statement
        self.assertIn('        if _sf_mode == "off":\n            ml_result = ml_score(ml_features)\n', self.src)

    def test_scan_publishes_its_change_distribution_right_after_the_quick_pass(self):
        i_sort = self.src.index('    quick_results.sort(key=lambda x: x["quick_score"], reverse=True)\n    scored = quick_results\n')
        i_pub = self.src.index("    publish_scan_changes(quick_results)\n")
        i_deep = self.src.index("    def _deep_one(args):")
        self.assertLess(i_sort, i_pub)
        self.assertLess(i_pub, i_deep)

    def test_served_vector_is_what_entry_features_and_the_record_carry(self):
        i_on = self.src.index('                if _sf_mode == "on":\n                    ml_features = _sf_fixed\n                    ml_result = _sf_fix_res\n')
        i_store = self.src.index('        quick_result["ml_features"]  = ml_features')
        self.assertLess(i_on, i_store)


if __name__ == "__main__":
    unittest.main()
