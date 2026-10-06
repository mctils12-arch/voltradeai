"""shadow_band_report (2026-10-06): aggregate view of labeled shadow records by
decision x signed same-day move x score quintile x regime, so the scan's
expectancy (not just hit rate) is visible. Aggregate-only by construction."""
import json
import shadow_portfolio as sp


def _rec(dec, cp, score, ret, label, regime="NEUTRAL_BULL", ticker="ZZZ"):
    return {"ticker": ticker, "decision": dec, "score": score, "regime_label": regime,
            "entry_price": 12.34, "timestamp": "2026-09-01T14:00:00+00:00",
            "features": {"change_pct_today": cp},
            "outcomes": {"+5d": {"return_pct": ret, "label": label},
                         "+10d": None, "+20d": {"label": -1}}}


def test_groups_by_decision_band_and_reports_mean_return():
    recs = [_rec("taken", 18.0, 80, -4.0, 0) for _ in range(30)] + \
           [_rec("taken", 18.0, 80, 2.0, 1) for _ in range(10)] + \
           [_rec("rejected_score", 1.0, 40, 2.0, 1) for _ in range(25)]
    r = sp.shadow_band_report(recs, min_n=20)
    t = r["decision"]["taken"]["+5d"]
    assert t["n"] == 40 and t["win_rate"] == 25.0
    assert t["mean_return_pct"] == -2.5          # 30x-4 + 10x2 over 40
    assert r["decision_x_band"]["taken|10..20"]["+5d"]["n"] == 40
    assert r["band_all"]["-3..3"]["+5d"]["win_rate"] == 100.0
    assert r["all"]["all"]["+5d"]["n"] == 65
    assert "+10d" not in r["decision"]["taken"], "unlabeled horizons are skipped, not counted"


def test_small_cells_report_n_only():
    r = sp.shadow_band_report([_rec("taken", 5.0, 50, 1.0, 1) for _ in range(3)], min_n=20)
    assert r["decision"]["taken"]["+5d"] == {"n": 3}


def test_signed_bands_keep_down_moves_separate():
    assert sp._band_of(-18.0) == "<-10" and sp._band_of(18.0) == "10..20"
    assert sp._band_of(-1.0) == "-3..3" and sp._band_of(35.0) == ">=30"


def test_no_ticker_price_or_time_leaves_the_report():
    recs = [_rec("taken", 18.0, 80, -4.0, 0, ticker="SECRETCO") for _ in range(25)]
    blob = json.dumps(sp.shadow_band_report(recs))
    assert "SECRETCO" not in blob and "12.34" not in blob and "2026-09-01" not in blob
