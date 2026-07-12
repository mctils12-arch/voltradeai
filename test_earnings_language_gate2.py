"""test_earnings_language_gate2.py — pure-function + wiring checks for
scripts/earnings_language_gate2.py (SEC 8-K earnings-language gate 2,
research/open_questions.md NEW DATA ROOTS #1). No live network calls —
fetch_bars is monkeypatched with synthetic bars, mirroring the injected-
fetch pattern server/sec8kEarnings.test.ts and edgarForm4.test.ts use on
the TypeScript side of this same stream."""
import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "earnings_language_gate2",
    os.path.join(os.path.dirname(__file__), "scripts", "earnings_language_gate2.py"))
g2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g2)


def test_tone_score_counts_known_words_and_ignores_neutral_text():
    t = g2.tone_score("Revenue grew and margins improved, a strong record quarter with no setbacks.")
    assert t["pos_count"] >= 4  # grew, improved, strong, record
    assert t["neg_count"] == 1  # setbacks
    assert t["tone_per_1000w"] > 0


def test_tone_score_negative_dominant_text():
    t = g2.tone_score("Revenue declined sharply amid a shortfall, impairment, and restructuring.")
    assert t["neg_count"] >= 3
    assert t["tone_per_1000w"] < 0


def test_tone_score_empty_text_is_none_not_zero():
    assert g2.tone_score("")["tone_per_1000w"] is None
    assert g2.tone_score(None)["tone_per_1000w"] is None


def test_tone_score_hyphenated_negative_term_matches():
    t = g2.tone_score("The company recorded a write-down this quarter.")
    assert t["neg_count"] == 1


DATES = ["2026-07-01", "2026-07-02", "2026-07-03", "2026-07-06", "2026-07-07", "2026-07-08"]


def test_reaction_indices_after_hours_filing_reacts_next_day():
    # filed 2026-07-02 at 16:15 ET (after close) -> pre=07-02 close, anchor=07-03
    ri = g2.reaction_indices(DATES, "2026-07-02", "2026-07-02T16:15:00-04:00")
    assert ri == (1, 2)


def test_reaction_indices_pre_market_filing_reacts_same_day():
    # filed 2026-07-03 at 07:00 ET (pre-market) -> pre=07-02 close, anchor=07-03
    ri = g2.reaction_indices(DATES, "2026-07-03", "2026-07-03T07:00:00-04:00")
    assert ri == (1, 2)


def test_reaction_indices_unknown_timestamp_treated_conservatively_as_same_day_reaction():
    ri = g2.reaction_indices(DATES, "2026-07-03", None)
    assert ri == (1, 2)


def test_reaction_indices_filing_on_non_trading_day_rolls_to_next_session():
    # 2026-07-04/05 not in DATES (weekend/holiday) -> rolls to 07-06
    ri = g2.reaction_indices(DATES, "2026-07-04", None)
    assert ri == (2, 3)


def test_reaction_indices_none_when_no_prior_bar_exists():
    ri = g2.reaction_indices(DATES, "2026-07-01", None)  # first bar itself -> no pre_idx
    assert ri is None


def test_reaction_indices_none_when_filing_is_after_all_available_bars():
    ri = g2.reaction_indices(DATES, "2026-07-20", None)
    assert ri is None


def test_forward_returns_only_reports_computable_horizons():
    closes = [10.0, 10.0, 11.0, 11.5, 9.0, 9.5]
    out = g2.forward_returns(DATES, closes, pre_idx=1, anchor_idx=2, horizons=(1, 3, 10))
    assert set(out.keys()) == {1, 3}  # horizon 10 has no bar this far out
    assert out[1][0] == "2026-07-03" and round(out[1][1], 4) == round(11.0 / 10.0 - 1, 4)
    assert out[3][0] == "2026-07-07" and round(out[3][1], 4) == round(9.0 / 10.0 - 1, 4)


def test_pearson_perfect_and_zero_and_too_few_points():
    assert round(g2.pearson([1, 2, 3], [1, 2, 3]), 4) == 1.0
    assert round(g2.pearson([1, 2, 3], [3, 2, 1]), 4) == -1.0
    assert g2.pearson([1, 2], [1, 2]) is None  # n<3 -> never a spurious r


def _fake_bars(dates, closes):
    return {"date": dates, "open": closes, "high": closes, "low": closes, "close": closes, "volume": [0] * len(dates)}


def test_compute_end_to_end_wiring_insufficient_sample_below_min(monkeypatch):
    # Two filings, opposite tone and opposite forward return, monkeypatched
    # bars (no live network) — verifies compute() wires tone -> return ->
    # verdict correctly. n=2 << MIN_SAMPLE, so both horizons must report
    # INSUFFICIENT SAMPLE, never a spurious PASS/FAIL off a 2-point sample.
    bars_by_ticker = {
        "SPY": _fake_bars(DATES, [500, 500, 500, 500, 500, 500]),
        "AAA": _fake_bars(DATES, [10, 10, 12, 12, 12, 12]),   # +20% day-1 pop
        "BBB": _fake_bars(DATES, [20, 20, 16, 16, 16, 16]),   # -20% day-1 drop
    }
    monkeypatch.setattr(g2, "fetch_bars", lambda ticker, days: bars_by_ticker[ticker])
    filings = [
        {"ticker": "AAA", "accession": "acc-1", "filedAt": "2026-07-02",
         "acceptanceDatetime": "2026-07-02T16:15:00-04:00",
         "text": "Revenue grew and margins improved, a strong record quarter."},
        {"ticker": "BBB", "accession": "acc-2", "filedAt": "2026-07-02",
         "acceptanceDatetime": "2026-07-02T16:15:00-04:00",
         "text": "Revenue declined amid a shortfall, impairment, and restructuring."},
    ]
    result = g2.compute(filings)
    assert result["data"]["scored_with_tone"] == 2
    assert result["horizons"]["1"]["n"] == 2
    assert result["horizons"]["1"]["verdict"] == "INSUFFICIENT SAMPLE"
    # directionally sane even though unverdicted: positive tone -> positive alpha
    aaa_row = [s for s in result["samples"][1] if s["ticker"] == "AAA"][0]
    bbb_row = [s for s in result["samples"][1] if s["ticker"] == "BBB"][0]
    assert aaa_row["tone_per_1000w"] > 0 and aaa_row["alpha"] > 0
    assert bbb_row["tone_per_1000w"] < 0 and bbb_row["alpha"] < 0


def test_compute_skips_filings_with_no_ticker_or_no_text():
    filings = [
        {"ticker": None, "accession": "acc-1", "filedAt": "2026-07-02", "text": "Revenue grew."},
        {"ticker": "AAA", "accession": "acc-2", "filedAt": "2026-07-02", "text": ""},
    ]
    result = g2.compute(filings)
    assert result["data"]["skipped_no_ticker"] == 1
    assert result["data"]["scored_with_tone"] == 0
