"""
[RULE-REVIEW] 2026-11-06 — signal_attribution.py: one row per stock-score
signal joining the long-term walk-forward and the forward shadow-log tests,
with pre-registered verdicts, the action the served score takes, the ML
training-set duplication, the B3 serve_fix comparison and the composite
legacy-vs-v2 comparison. Every test here fails before the module exists.
"""
import json
import os
import re
from datetime import datetime, timedelta, timezone
from unittest import mock

import numpy as np
import pytest

import signal_attribution as sa
import source_validation as sv
import walkforward as wf


# ── fixtures ────────────────────────────────────────────────────────────────

def shadow_records(n_days=45, per_day=8, copies=3, seed=3, with_v2=False, with_serve_fix=True):
    """per_day candidates a day, each logged `copies` times (one per scan, as
    the live scan does). Outcome y = g + noise. Planted: `news` carries +g
    (forward PASS), `volume` carries -g (forward FAIL_INVERTED), `social` is
    always 0 (CONSTANT). The served ML probability is noise; serve_fix's
    corrected probability tracks g."""
    rng = np.random.default_rng(seed)
    recs, d, day = [], datetime(2026, 8, 3, 14, 0, tzinfo=timezone.utc), 0
    while day < n_days:
        if d.weekday() < 5:
            for k in range(per_day):
                g = float(rng.normal())
                y = g + 0.6 * float(rng.normal())
                bd = {key: round(float(rng.normal()), 2) for key, _ in sv.ADJUSTMENT_SOURCES}
                bd.update({"news": round(4 * g + 0.5 * float(rng.normal()), 2),
                           "volume": round(50 - 10 * g + float(rng.normal()), 2), "social": 0.0})
                legacy = 60 + 5 * float(rng.normal())
                if with_v2:
                    bd["composite_legacy"] = round(legacy, 2)
                    bd["composite_v2"] = round(60 + 8 * g + 2 * float(rng.normal()), 2)
                feats = {f: float(rng.normal()) for f, _ in sv.FEATURE_SOURCES}
                p_cur = 0.5 + 0.05 * float(rng.normal())
                for c in range(copies):
                    ts = d + timedelta(minutes=15 * c)
                    rec = {"ticker": f"Z{day:02d}{k}", "timestamp": ts.isoformat(), "score": legacy,
                           "decision": "taken", "vxx_ratio": 1.0, "features": feats,
                           "outcomes": {"+5d": {"label": int(y > 0), "return_pct": y},
                                        "+10d": {"label": int(y > 0), "return_pct": y}},
                           "score_breakdown": bd,
                           "ml": {"win_probability": p_cur, "blended": True, "gated": True}}
                    if with_serve_fix:
                        rec["serve_fix"] = {"mode": "shadow", "served": "current", "n_changed": 2,
                                            "features_fixed_diff": {"ewma_vol": 1.7, "range_pct": 3.2},
                                            "features_current_diff": {"ewma_vol": 2, "range_pct": 0},
                                            "ml_prob_current": p_cur, "ml_prob_fixed": 0.5 + 0.1 * g}
                    recs.append(rec)
            day += 1
        d += timedelta(days=1)
    return recs


def _hz(ic, t, pb, q=None, top=None, ttop=None, by_regime=None, by_year=None):
    return {"ic": {"mean": ic, "t_nw": t, "p": pb / 30 if pb is not None else None, "p_bonferroni": pb, "n_dates": 513,
                   "hit_rate": 0.45},
            "quintiles": {"mean_net_pct": q or [0.1, 0.05, 0.0, -0.05, -0.1], "spread_net_long_short_pct": -0.2,
                          "t_nw_long_short": -2.1},
            "top_n": {"excess_vs_random_pct": top, "t_nw_vs_random": ttop, "excess_vs_spy_pct": -0.5},
            "by_regime": by_regime or {}, "by_year": by_year or {}}


def walkforward_latest(with_v2_input=False):
    years_neg = {str(y): {"mean_ic": -0.01 if y != 2020 else 0.004} for y in range(2017, 2027)}
    inputs = {
        "quick_score": {"verdict": "null", "horizons": {"5": _hz(-0.011, -1.9, 0.34, top=-1.63, ttop=-6.1)}},
        "momentum": {"verdict": "null", "horizons": {"1": _hz(-0.004, -0.8, 1.0), "5": _hz(0.011, 1.7, 1.0, top=-0.49, ttop=-1.2),
                                                       "20": _hz(0.018, 1.9, 1.0)}},
        "mean_reversion": {"verdict": "null", "horizons": {"5": _hz(-0.004, -0.6, 1.0)}},
        "volume": {"verdict": "negative", "verdict_note": "5d IC -0.0153 survives Bonferroni",
                   "horizons": {"5": _hz(-0.0153, -5.2, 0.0, top=-0.81, ttop=-3.6,
                                         by_regime={"NEUTRAL": {"n_dates": 300, "mean_ic": -0.014, "t_nw": -4.0,
                                                                "top_n_excess_vs_random_pct": -0.7}},
                                         by_year=years_neg),
                                "20": _hz(-0.0147, -3.3, 0.02)},
                   "live_exit": {"n_picks": 4000, "excess_vs_random_pct": -0.136, "t_nw_vs_random": -1.1}},
        "vol_premium": {"verdict": "negative", "horizons": {"5": _hz(-0.0167, -4.1, 0.0012), "20": _hz(-0.0269, -3.9, 0.003)}},
        "adx": {"verdict": "null", "horizons": {"5": _hz(-0.0046, -0.7, 1.0)}},
        "obv": {"verdict": "null", "horizons": {"5": _hz(-0.0101, -1.8, 0.38)}},
        "rel_strength": {"verdict": "negative", "horizons": {"5": _hz(-0.0198, -4.0, 0.0012), "20": _hz(-0.0246, -3.5, 0.01)}},
        "price_composite": {"verdict": "null", "horizons": {"5": _hz(-0.0089, -1.5, 1.0, top=-1.17, ttop=-4.0)}},
    }
    if with_v2_input:
        inputs["price_composite_v2"] = {"verdict": "null", "horizons": {"5": _hz(0.001, 0.2, 1.0, top=-0.4, ttop=-1.0)}}
    return {"schema": "walkforward/v1", "generated_at": "2026-11-01T18:54:41+00:00", "as_of": "2026-10-27",
            "code_version": "1.0.1243",
            "window": {"first_rebalance": "2017-01-06", "last_rebalance": "2026-10-27", "n_rebalances_scored": 513,
                       "horizons_days": [5, 20]},
            "universe": {"names_per_date": {"mean": 2061.6}, "delisted_symbol_share": 0.30},
            "multiple_testing": {"ic_tests": 18}, "baselines": {"5": {"random_entry_mean_net_pct": 0.1}},
            "inputs": inputs, "not_measurable": dict(wf.NOT_MEASURABLE)}


ML_V3 = {"candidate_exists": True, "promoted": False, "trained_at": "2026-11-05", "rows": 36581,
         "date_range": ["2024-01-08", "2026-10-06"], "auc_ci95_date_block": [0.4939, 0.5295],
         "pooled": {"n": 32209, "auc": 0.5129, "log_loss": 0.69396, "base_rate_log_loss": 0.69265}, "variants_tried": 1}
CFG_V2 = {"enabled": True, "neutralize": ("volume", "vol_premium", "rel_strength"), "ml_neutral": True,
          "since": "2026-11-06"}


@pytest.fixture(scope="module")
def table():
    recs = shadow_records()
    return sa.build_attribution(recs, walkforward=walkforward_latest(), ml_v3=ML_V3, blend={"rule_weight": 0.72, "ml_weight": 0.28},
                                cfg=CFG_V2, now=datetime(2026, 11, 6, tzinfo=timezone.utc))


def _row(t, sid):
    return next(r for r in t["signals"] if r["id"] == sid)


# ── registry ────────────────────────────────────────────────────────────────

def test_registry_points_at_real_harness_names():
    fw_names = {s["name"] for s in sv.source_registry()}
    for s in sa.SIGNAL_REGISTRY:
        if s.get("fw"):
            assert s["fw"] in fw_names, s
        if s.get("lt"):
            assert s["lt"] in wf.INPUTS, s
        if s.get("lt_nm"):
            assert s["lt_nm"] in wf.NOT_MEASURABLE, s
    for c in sa.COMPOSITE_ROWS:
        assert c["lt"] in wf.INPUTS
    # every deep_score component the shadow record persists has a row
    assert {k for k, _ in sv.ADJUSTMENT_SOURCES} <= {s["id"] for s in sa.SIGNAL_REGISTRY}
    assert len({s["id"] for s in sa.SIGNAL_REGISTRY}) == len(sa.SIGNAL_REGISTRY)


def test_combine_verdict_truth_table():
    cv = sa.combine_verdict
    assert cv("negative", None) == "anti_predictive" == cv(None, "FAIL_INVERTED") == cv("null", "FAIL_INVERTED")
    assert cv("positive", "FAIL") == "predictive" == cv("null", "PASS") == cv(None, "PASS")
    assert cv("negative", "PASS") == "conflicting" == cv("positive", "FAIL_INVERTED")
    assert cv(None, "CONSTANT") == "dead" == cv("insufficient", "CONSTANT")
    assert cv("null", None) == "noise" == cv(None, "FAIL") == cv("null", "FAIL")
    assert cv("null", "CONSTANT") == "dead"           # a live input that never moves outranks "noise"
    assert cv("insufficient", "INSUFFICIENT") == "untested" == cv(None, None)


def test_bonferroni_bar():
    assert sa.bonferroni_z(1) == pytest.approx(1.95996, abs=1e-4)
    assert sa.bonferroni_z(30) == pytest.approx(3.1440, abs=1e-3)
    assert sa.bonferroni_z(30) > sa.bonferroni_z(18) > sa.bonferroni_z(1)


# ── long-term block ─────────────────────────────────────────────────────────

def test_long_term_block_reads_every_horizon_regime_and_year():
    lt = sa.long_term_block(walkforward_latest(), "volume")
    assert lt["verdict"] == "negative" and lt["measurable"]
    assert lt["h5"]["ic"] == -0.0153 and lt["h5"]["t"] == -5.2 and lt["h5"]["p_bonferroni"] == 0.0
    assert lt["h5"]["top10_vs_random_pct"] == -0.81 and lt["h20"]["ic"] == -0.0147 and "h1" not in lt
    assert lt["h5"]["quintile_mean_net_pct"] == [0.1, 0.05, 0.0, -0.05, -0.1]
    assert lt["by_regime_5d"]["NEUTRAL"]["ic"] == -0.014
    assert lt["years"] == {"n": 10, "positive": 1, "negative": 9}
    assert lt["live_exit_vs_random_pct"] == -0.136
    m = sa.long_term_block(walkforward_latest(), "momentum")
    assert m["h1"]["ic"] == -0.004


def test_long_term_block_missing_input_file_and_unmeasurable():
    lt = sa.long_term_block(walkforward_latest(), "price_composite_v2")
    assert lt["verdict"] == "insufficient" and "predates this input" in lt["reason"]
    assert sa.long_term_block(None, "volume")["verdict"] == "insufficient"
    nm = sa.long_term_block(walkforward_latest(), None, "vrp")
    assert nm["measurable"] is False and nm["reason"] == wf.NOT_MEASURABLE["vrp"]
    assert sa.long_term_block(None, None, None)["measurable"] is False


# ── action ──────────────────────────────────────────────────────────────────

def test_action_reads_the_composite_config():
    assert sa.action_for("volume", None)[0] == "kept"
    assert sa.action_for("volume", CFG_V2)[0] == "cut"
    assert sa.action_for("momentum", CFG_V2)[0] == "kept"
    assert sa.action_for("volume", dict(CFG_V2, enabled=False))[0] == "kept"
    assert sa.action_for("ml_v2", CFG_V2)[0] == "cut"
    assert sa.action_for("ml_v2", dict(CFG_V2, ml_neutral=False))[0] == "kept"
    assert sa.action_for("earnings_beats", CFG_V2)[0] == "not_used"
    assert sa.composite_config(False) is None and sa.composite_config(CFG_V2) is CFG_V2
    with mock.patch.dict("system_config.BASE_CONFIG", {"SCORE_COMPOSITE_V2": CFG_V2}):
        assert sa.composite_config() is CFG_V2
    with mock.patch.dict("system_config.BASE_CONFIG", {"SCORE_COMPOSITE_V2": "junk"}):
        assert sa.composite_config() is None


# ── the table ───────────────────────────────────────────────────────────────

def test_table_verdicts_follow_the_evidence(table):
    assert _row(table, "volume")["verdict"] == "anti_predictive"           # long-term negative (and forward inverted)
    assert _row(table, "volume")["evidence"] == "both"
    assert _row(table, "rel_strength")["verdict"] == "anti_predictive"     # long-term negative
    assert _row(table, "news")["verdict"] == "predictive"                  # forward PASS, no long-term
    assert _row(table, "news")["forward"]["verdict"] == "PASS"
    assert _row(table, "social")["verdict"] == "dead"                      # never moves
    assert _row(table, "momentum")["verdict"] == "noise"                   # long-term null
    assert _row(table, "quick_score")["verdict"] == "noise"
    assert _row(table, "vrp")["long_term"]["measurable"] is False
    eb = _row(table, "earnings_beats")
    assert eb["verdict"] == "untested" and eb["action"] == "not_used" and "Not used" in eb["action_text"]
    assert "no report dates, so no point-in-time history exists" in eb["reason"]
    assert "deep_score does not fetch it" in eb["reason"] and eb["reason"].count("point-in-time history") == 1


def test_actions_and_review_flags(table):
    assert _row(table, "volume")["action"] == "cut" and not _row(table, "volume")["review"]
    assert _row(table, "momentum")["action"] == "kept" and _row(table, "momentum")["review"]
    assert _row(table, "news")["review"] is False
    assert table["served_composite"] == "composite_v2"
    assert table["composite_config"]["neutralize"] == ["volume", "vol_premium", "rel_strength"]


def test_reasons_are_plain_english_with_the_numbers(table):
    v = _row(table, "volume")["reason"]
    assert v.startswith("Anti-predictive.") and "LOWER 5-day return" in v and "−0.015" in v
    assert "9 of 10 years" in v and "trailed random entry by 0.81 points per 5 days" in v
    assert "Bonferroni" in v
    q = _row(table, "quick_score")["reason"]
    assert "beyond the Bonferroni bar" in q                                 # t -6.1 vs the 18-test bar
    for r in table["signals"]:
        assert not re.search(r"\bNone\b|\bnan\b|\(\(|\.\)", r["reason"]), (r["id"], r["reason"])
        assert r["verdict"] in sa.VERDICTS
    assert "No signal" not in table["headline"]                             # news is predictive in this fixture
    assert "trailed random entry by 1.63 points" in table["headline"]


def test_ml_row_reports_honest_oos_and_the_training_duplication(table):
    ml = _row(table, "ml_v2")
    assert ml["verdict"] == "noise" and ml["action"] == "cut"
    assert "AUC 0.513" in ml["reason"] and "repeated 3.0 times each" in ml["reason"]
    dup = table["ml"]["v2_live"]["training_set"]
    assert dup["duplication"] == 3.0 and dup["rows"] == 3 * dup["unique_ticker_days"]
    assert table["ml"]["v3_candidate"]["auc_ci95_date_block"] == [0.4939, 0.5295]


def test_training_duplication_mirrors_load_shadow_data_filter():
    recs = shadow_records(n_days=3, per_day=2, copies=5)
    recs[0]["outcomes"]["+10d"]["label"] = -1            # unlabelable: excluded like load_shadow_data
    recs[1]["features"] = {}                              # no features: excluded
    d = sa.training_duplication(recs)
    assert d["rows"] == len(recs) - 2 and d["unique_ticker_days"] == 6
    assert d["duplication"] == round((len(recs) - 2) / 6, 2)
    assert sa.training_duplication([])["duplication"] is None


def test_serve_fix_summary_measures_delta_and_paired_oos(table):
    sf = table["ml"]["serve_fix"]
    assert sf["records"] > 0 and sf["modes"] == {"shadow": sf["records"]}
    assert sf["prob_delta"]["n"] == sf["records"] and sf["prob_delta"]["mean"] > 0
    assert sf["features_changed"] == {"ewma_vol": sf["records"], "range_pct": sf["records"]}
    oos = sf["oos"]
    assert oos["status"] == "OK" and oos["b_better"] is True and oos["ic_b"] > oos["ic_a"]
    none = sa.serve_fix_summary(shadow_records(n_days=12, with_serve_fix=False))
    assert none["records"] == 0 and none["oos"]["status"] == "INSUFFICIENT"


def test_composite_comparison_insufficient_until_v2_is_logged_then_paired(table):
    assert table["composite_comparison"]["status"] == "INSUFFICIENT"
    t2 = sa.build_attribution(shadow_records(with_v2=True), walkforward=walkforward_latest(with_v2_input=True),
                              ml_v3=ML_V3, cfg=CFG_V2)
    c = t2["composite_comparison"]
    assert c["status"] == "OK" and c["b_better"] is True and c["n"] >= sv.MIN_N
    lt_v2 = next(r for r in t2["composites"] if r["id"] == "composite_v2")["long_term"]
    assert lt_v2["h5"]["ic"] == 0.001


def test_no_config_means_legacy_served_and_nothing_cut():
    t = sa.build_attribution(shadow_records(n_days=12), walkforward=walkforward_latest(), cfg=False)
    assert t["served_composite"] == "composite_legacy" and t["composite_config"] is None
    assert all(r["action"] in ("kept", "not_used") for r in t["signals"])
    assert _row(t, "volume")["review"] is True


def test_payload_is_aggregate_only_and_within_the_diag_sanitizer(table):
    txt = json.dumps(table)
    assert "Z0" not in txt, "no ticker leaves the table"

    def walk(v, depth=0):
        assert depth <= 8
        if isinstance(v, dict):
            assert len(v) <= 100
            for x in v.values():
                walk(x, depth + 1)
        elif isinstance(v, list):
            assert len(v) <= 200
            for x in v:
                walk(x, depth + 1)
        elif isinstance(v, str):
            assert len(v) <= 2000
    walk(table)


def test_production_loader_reads_the_volume_files(tmp_path):
    wfp = tmp_path / "latest.json"
    wfp.write_text(json.dumps(walkforward_latest()))
    recs = shadow_records(n_days=12)
    with mock.patch("shadow_portfolio._load_shadow_log", return_value=recs), \
            mock.patch.object(sa, "_read_walkforward", return_value=json.loads(wfp.read_text())), \
            mock.patch("brain_evidence._ml_v3_summary", return_value=ML_V3), \
            mock.patch("brain_evidence.v3_shadow_ic", return_value={"n": 0}):
        out = sa.get_signal_attribution()
    assert out["shadow_log"] == {"records": len(recs)} and "errors" not in out
    assert out["sources"]["long_term"]["as_of"] == "2026-10-27"
    assert {r["id"] for r in out["signals"]} >= {"volume", "ml_v2", "quick_score"}
    assert sa._read_walkforward(str(tmp_path / "absent.json")) is None
    assert sa._read_walkforward(str(wfp))["schema"] == "walkforward/v1"
