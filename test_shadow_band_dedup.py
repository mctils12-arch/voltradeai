"""shadow_band_report_deduped (2026-10-27 [RULE-REVIEW]): the shadow logger
writes the same candidate on every scan, so shadow_bands' n counts per-scan
copies, not events (10/20: 5,192 labeled 'taken' records were 148
ticker-days). The deduped probe keeps ONE first-seen record per (ticker, ET
entry session), for both rulers, with the full population's quintile edges.
shadow_bands itself is unchanged."""
import json
import shadow_portfolio as sp


def _rec(ticker, ts, ret, label, dec="taken", action="BUY", cp=15.0, score=80, xa=None):
    r = {"ticker": ticker, "decision": dec, "score": score, "regime_label": "BULL",
         "action_label": action, "entry_price": 12.34, "timestamp": ts,
         "features": {"change_pct_today": cp},
         "outcomes": {"+5d": {"return_pct": ret, "label": label}, "+10d": None, "+20d": None}}
    if xa is not None:
        r[sp.XA_KEY] = {"label": xa[1], "return_pct": xa[0], "exit_reason": "stop",
                        "censored": False, "regime_class": "bullish"}
    return r


def test_one_record_per_ticker_day_and_first_seen_wins():
    # 30 scans of the same ticker on the same ET day: the first scan lost -4,
    # later copies (different entry prices) "won" — only the first may count.
    dup = [_rec("AAA", "2026-10-07T13:40:00+00:00", -4.0, 0)] + \
          [_rec("AAA", f"2026-10-07T{14 + i // 6:02d}:{(i % 6) * 10:02d}:00+00:00", 2.0, 1) for i in range(29)]
    others = [_rec(f"T{i}", "2026-10-08T14:00:00+00:00", 2.0, 1) for i in range(20)]
    rep = sp.shadow_band_report_deduped(dup + others, min_n=20)
    old = rep["old"]
    assert old["n_records"] == 50 and old["n_unique"] == 21
    cell = old["decision_x_action"]["taken|BUY"]["+5d"]
    assert cell["n"] == 21, "dozens of same-day copies collapse to one event"
    assert cell["win_rate"] == round(100 * 20 / 21, 1), "the FIRST-SEEN copy (the -4 loss) is the one kept"


def test_different_et_days_stay_separate():
    recs = [_rec("AAA", "2026-10-07T14:00:00+00:00", -4.0, 0),
            _rec("AAA", "2026-10-08T14:00:00+00:00", 2.0, 1)]
    assert sp.shadow_band_report_deduped(recs, min_n=1)["old"]["n_unique"] == 2


def test_min_n_now_counts_events_not_copies():
    # 40 copies of 2 ticker-days: shadow_bands reports full stats, the deduped
    # probe correctly reports n only (2 < 20).
    recs = [_rec("AAA", f"2026-10-07T14:{i:02d}:00+00:00", -4.0, 0) for i in range(20)] + \
           [_rec("BBB", f"2026-10-07T14:{i:02d}:00+00:00", -4.0, 0) for i in range(20)]
    assert "mean_return_pct" in sp.shadow_band_report(recs)["decision_x_action"]["taken|BUY"]["+5d"]
    assert sp.shadow_band_report_deduped(recs)["old"]["decision_x_action"]["taken|BUY"]["+5d"] == {"n": 2}


def test_exit_aligned_ruler_is_deduped_on_its_own_labels():
    recs = [_rec("AAA", f"2026-10-07T14:{i:02d}:00+00:00", -4.0, 0, xa=(-8.0, 0)) for i in range(25)] + \
           [_rec(f"T{i}", "2026-10-08T14:00:00+00:00", 2.0, 1, xa=(5.0, 1)) for i in range(20)] + \
           [_rec("NOXA", "2026-10-09T14:00:00+00:00", 2.0, 1)]  # old label only
    xa = sp.shadow_band_report_deduped(recs, min_n=20)["exit_aligned"]
    assert xa["n_records"] == 45 and xa["n_unique"] == 21
    assert xa["decision_x_action"]["taken|BUY"]["xa"]["n"] == 21


def test_quintile_edges_come_from_the_full_population():
    # Full population: scores 0..99 (one copy each, distinct days). Dedup
    # collapses 50 copies of one high-score ticker-day; the quintile LABEL of
    # a score must not shift just because duplicates were removed.
    base = [_rec(f"T{i}", f"2026-09-{1 + i % 28:02d}T14:00:00+00:00", 1.0, 1, score=i) for i in range(100)]
    dups = [_rec("HI", f"2026-10-07T14:{i:02d}:00+00:00", 1.0, 1, score=99) for i in range(50)]
    full_edges = sp._score_quintile_edges(base + dups)
    rep = sp.shadow_band_report_deduped(base + dups, min_n=1)
    assert rep["old"]["score_quintile_edges"] == [round(e, 3) for e in full_edges]


def test_shadow_bands_output_is_unchanged_by_the_edges_parameter():
    recs = [_rec(f"T{i}", "2026-10-08T14:00:00+00:00", (-4.0 if i % 3 else 2.0), (0 if i % 3 else 1), score=i)
            for i in range(60)]
    assert sp.shadow_band_report(recs) == sp.shadow_band_report(recs, edges=None)


def test_aggregate_only():
    recs = [_rec("SECRETCO", f"2026-10-07T14:{i:02d}:00+00:00", -4.0, 0, xa=(-8.0, 0)) for i in range(25)]
    blob = json.dumps(sp.shadow_band_report_deduped(recs))
    assert "SECRETCO" not in blob and "12.34" not in blob and "2026-10-07" not in blob
