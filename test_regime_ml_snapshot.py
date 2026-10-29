"""
ML-ready logging (2026-10-28 regime live flip, human directive; ML is NOT built
or enabled): every shadow record the regime engine touches carries a compact
`regime_snapshot` — the state's inputs at decision time (stress flags, trend /
fast / vol / breadth states, composite, exposure, confidence, as_of), the engine
label and the Markov label, and the engine's gate decision next to legacy's —
and the existing backfills attach realized outcomes to the same record. With
the engine off the key is absent (pre-flip record shape). Logging only.
"""
import json
import os
import random
import tempfile
import unittest
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch

import regime_engine as re_
import shadow_portfolio as sp
import tiered_strategy as ts

ROOT = os.path.dirname(os.path.abspath(__file__))
BOT_SRC = open(os.path.join(ROOT, "bot_engine.py")).read()


def _bdays(n, start="2015-01-02"):
    d = date.fromisoformat(start)
    out = []
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def full_state():
    n = 700
    rnd = random.Random(4)
    g = [100.0]
    for _ in range(n - 1):
        g.append(g[-1] * (1 + 0.0004 + rnd.gauss(0, 0.005)))
    basket = {f"S{j}": [c * (1 + 0.01 * j) for c in g] for j in range(8)}
    st = re_.compute_state(_bdays(n), g, basket, [18.0] * n, [20.0] * n, gating="SPY",
                           sources={"SPY": "alpaca", "vvix_latest": 85.0}, ebp={"date": "2026-09-01", "ebp": -0.3})
    return asdict(st)


SEL_VERDICT = {"flags": {"knife": 1, "dip": 1, "panic": 0, "rs": 0, "breadth": 0}, "would_block": ["knife"],
               "neutralize_mr": True, "mr_delta": 2.5, "reasons": {}}

SNAPSHOT_KEYS = ("v", "as_of", "pv", "stale", "composite", "trend", "fast", "vol", "breadth", "panic",
                 "exposure", "exposure_c", "vol_dial", "confidence", "days_in_regime", "stress", "inputs", "ebp",
                 "label_engine", "label_markov", "label_used", "engine_decision", "legacy_decision")


class TestSnapshot(unittest.TestCase):
    def test_every_field_a_model_needs_is_present_and_compact(self):
        st = full_state()
        s = re_.ml_snapshot(st, kind="selection", markov_label="NEUTRAL_BULL", engine_label={"label": "BULL"},
                            verdict=SEL_VERDICT, gates={"knife", "dip"})
        for k in SNAPSHOT_KEYS:
            self.assertIn(k, s)
        self.assertEqual((s["as_of"], s["composite"], s["label_engine"]), (st["as_of"], st["composite_regime"], st["legacy_label"]))
        self.assertEqual((s["label_markov"], s["label_used"]), ("NEUTRAL_BULL", "engine"))
        self.assertEqual((s["pv"], s["stale"]), (re_.PARAMS_VERSION, False))
        self.assertEqual(s["exposure"], st["exposure_multiplier"])
        self.assertEqual(s["exposure_c"], st["exposure_continuous"])
        self.assertEqual(s["confidence"], st["confidence"])
        self.assertEqual((s["trend"], s["fast"], s["vol"], s["breadth"]), (st["trend_state"], st["fast_state"], st["vol_state"], st["breadth_state"]))
        for k in ("vix", "vix_vix3m", "vrp", "vix_pct_1y", "vvix", "rv21", "ewma_vol", "vol_pct"):
            self.assertIn(k, s["stress"])
        self.assertEqual(s["stress"]["vix"], 18.0)
        self.assertEqual(s["stress"]["vix_vix3m"], 0.9)
        self.assertEqual(s["stress"]["vvix"], 85.0)
        self.assertEqual(s["ebp"], -0.3)
        for k in ("f_slow", "f_fast", "breadth", "above_200d", "dd52", "mom24m", "var126_pct"):
            self.assertIn(k, s["inputs"])
        self.assertEqual(s["engine_decision"], {"gate": "block", "blocked_by": ["knife"], "would_block": ["knife"],
                                                "dip_neutralized": True, "mr_delta": 2.5})
        self.assertEqual(s["legacy_decision"], {"gate": "pass"})
        self.assertLess(len(json.dumps(s)), 1000)                         # compact: the shadow log is rewritten per record
        self.assertLess(len(json.dumps(s)), sp.REGIME_SNAPSHOT_MAX_BYTES)

    def test_markov_label_stood_and_shadow_only_gates(self):
        s = re_.ml_snapshot(full_state(), markov_label="NEUTRAL_BULL", engine_label=None, verdict=SEL_VERDICT, gates=set())
        self.assertEqual(s["label_used"], "markov")
        self.assertEqual((s["engine_decision"]["gate"], s["engine_decision"]["dip_neutralized"]), ("pass", False))

    def test_options_kind(self):
        st = dict(full_state(), vol_state="CRISIS")
        v = re_.csp_verdict(st, {"days_to_earnings": None, "dte_max": 45})
        s = re_.ml_snapshot(st, kind="options", verdict=v, gates=set(re_.OPTIONS_GUARDS))
        self.assertEqual((s["engine_decision"]["gate"], s["engine_decision"]["blocked_by"]), ("block", ["crisis"]))
        self.assertIn("size_mult", s["engine_decision"])
        self.assertEqual(s["legacy_decision"], {"gate": "pass", "size_mult": 1.0})

    def test_no_state_means_none(self):
        for st in (None, {}, {"composite_regime": "BEAR"}):
            self.assertIsNone(re_.ml_snapshot(st, verdict=SEL_VERDICT, gates={"knife"}))


class ShadowBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._paths = (sp.SHADOW_LOG_PATH, sp.SHADOW_LOCK_PATH)
        sp.SHADOW_LOG_PATH = os.path.join(self.tmp.name, "shadow.json")
        sp.SHADOW_LOCK_PATH = sp.SHADOW_LOG_PATH + ".lock"

    def tearDown(self):
        sp.SHADOW_LOG_PATH, sp.SHADOW_LOCK_PATH = self._paths
        self.tmp.cleanup()


class TestShadowRecord(ShadowBase):
    def test_snapshot_rides_the_record_and_is_absent_when_off(self):
        snap = re_.ml_snapshot(full_state(), markov_label="NEUTRAL_BULL", verdict=SEL_VERDICT, gates={"knife"})
        sp.log_candidate("AMD", {"rsi": 50.0}, 70.0, "taken", entry_price=100.0, regime_label="BULL", regime_snapshot=snap)
        sp.log_candidate("NVDA", {"rsi": 50.0}, 70.0, "taken", entry_price=100.0)
        sp.log_candidate("TSLA", {}, 70.0, "taken", entry_price=100.0, regime_snapshot={"x": "y" * 5000})   # oversized
        a, b, c = sp._load_shadow_log()
        self.assertEqual(a["regime_snapshot"], json.loads(json.dumps(snap)))
        self.assertNotIn("regime_snapshot", b)
        self.assertNotIn("regime_snapshot", c)

    def test_backfill_attaches_outcomes_to_a_record_with_a_snapshot(self):
        snap = re_.ml_snapshot(full_state(), verdict=SEL_VERDICT, gates={"knife"})
        ts_ = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        sp._save_shadow_log([{"ticker": "WIN", "timestamp": ts_, "score": 70.0, "decision": "rejected_regime_knife",
                              "decision_reason": "x", "entry_price": 100.0, "vxx_ratio": 1.0, "regime_label": "BULL",
                              "features": {}, "outcomes": {"+5d": None, "+10d": None, "+20d": None},
                              "code_version": "t", "regime_snapshot": snap}])
        start = datetime.now(timezone.utc) - timedelta(days=59)
        bars = [{"t": (start + timedelta(days=i)).strftime("%Y-%m-%dT00:00:00Z"), "h": 103.0, "l": 99.5, "c": 103.0}
                for i in range(30)]
        with patch.object(sp, "_fetch_historical_bars_batch", side_effect=lambda t, s, e: {x: bars for x in t}):
            stats = sp.backfill_outcomes(max_records=50)
        (rec,) = sp._load_shadow_log()
        self.assertEqual(stats["updated"], 3)
        for h in ("+5d", "+10d", "+20d"):
            self.assertIsNotNone(rec["outcomes"][h])
        self.assertEqual(rec["regime_snapshot"], json.loads(json.dumps(snap)))   # the backfill keeps the snapshot


class TestTier1ShadowRecords(ShadowBase):
    def test_blocked_csp_shadow_record_carries_the_snapshot(self):
        st = dict(full_state(), vol_state="CRISIS")
        captured = []
        ctx = ts.TierContext(equity=100_000.0, peak_equity=100_000.0, buying_power=100_000.0, positions=[], vxx_ratio=1.0, spy_vs_ma50=1.0)
        with patch("tiered_strategy._get_t1_universe", return_value=["AMD"]), \
             patch("csp_universe._load_layer2_cache", return_value={}), \
             patch("tiered_strategy._t1_earnings_calendar", return_value=({}, True)), \
             patch("tiered_strategy.log_csp_gate_shadow", side_effect=lambda r: captured.extend(r) or {}), \
             patch("regime_engine.cached_state", return_value=st), \
             patch("regime_engine.shadow_once_per_day", return_value=True), \
             patch("regime_engine.log_counterfactual", return_value=False), \
             patch.dict(os.environ, {"VOLTRADE_REGIME_ENGINE": "1", "VOLTRADE_REGIME_OPTIONS": "all"}):
            ts.tier1_csp_core(ctx)
        self.assertEqual([r["decision"] for r in captured], ["rejected_regime_crisis"])
        snap = captured[0]["metadata"]["regime_snapshot"]
        self.assertEqual((snap["vol"], snap["engine_decision"]["gate"]), ("CRISIS", "block"))
        with patch("tiered_strategy._layer2_price_lookup", return_value={"AMD": 150.0}):
            out = ts.log_csp_gate_shadow(captured)
        self.assertEqual(out["logged"], 1)
        (rec,) = sp._load_shadow_log()
        self.assertEqual(rec["regime_snapshot"]["engine_decision"]["blocked_by"], ["crisis"])

    def test_engine_off_no_snapshot_anywhere(self):
        with patch("tiered_strategy._layer2_price_lookup", return_value={"AMD": 150.0}):
            ts.log_csp_gate_shadow([{"ticker": "AMD", "decision": "rejected_options_slots", "reason": "x", "metadata": {"vxx_ratio": 1.0}}])
        (rec,) = sp._load_shadow_log()
        self.assertNotIn("regime_snapshot", rec)
        ctx = ts.TierContext(equity=100_000.0, peak_equity=100_000.0, buying_power=100_000.0, positions=[], vxx_ratio=1.0, spy_vs_ma50=1.0)
        with patch("tiered_strategy._get_t1_universe", return_value=["AMD"]), \
             patch("csp_universe._load_layer2_cache", return_value={}), \
             patch.dict(os.environ, {"VOLTRADE_REGIME_ENGINE": "off"}):
            actions = ts.tier1_csp_core(ctx)
        self.assertTrue(actions)
        self.assertIsNone(actions[0].metadata["regime_guards"])
        self.assertNotIn("regime_snapshot", actions[0].metadata)


class TestWiring(unittest.TestCase):
    def test_deep_score_passes_the_snapshot_only_while_the_engine_is_on(self):
        i_snap = BOT_SRC.index("_regime_snapshot = _re_ml.ml_snapshot(")
        i_gate = BOT_SRC.index("            if _re_ml.engine_allowed():\n                _regime_snapshot = _re_ml.ml_snapshot(")
        i_log = BOT_SRC.index("            regime_snapshot=_regime_snapshot,\n        )")
        self.assertLess(i_gate, i_snap)
        self.assertLess(i_snap, i_log)
        self.assertIn("markov_label=_markov_lbl if '_markov_lbl' in locals() else None,", BOT_SRC)
        self.assertIn("verdict=_regime_sel, gates=_re_ml.selection_gates())", BOT_SRC)

    def test_scanner_shadow_record_passes_the_snapshot(self):
        self.assertIn('"gates": _re_opt.options_gates(), "engine": _re_opt, "state": _rg_opt_state}', BOT_SRC)
        self.assertIn('_rg_opt.get("state"), kind="options", verdict=_rg_opt["verdict"], gates=_rg_opt["gates"]),', BOT_SRC)


if __name__ == "__main__":
    unittest.main()
