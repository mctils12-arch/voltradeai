"""Offline tests for lazy_prices_panel.py (Lazy Prices PR 3c): the join must
count every drop, eligibility must never see the entry bar, and the adapter must
drive sec_filings_gate2 end to end on synthetic data (a plumbing proof only)."""
import numpy as np

import lazy_prices_panel as lp
import price_panel as pp
import sec_cik_symbol_map as cmap

D0 = pp.day_num("2016-01-04")


def _series(sym, n=1200, start=0, px=10.0, vol=1_000_000.0, open_mult=1.0):
    days = np.arange(D0 + start, D0 + start + n, dtype=np.int64)
    c = np.full(n, px)
    return pp.Series(symbol=sym, mode="default", seg=0, part=0, asset_ids=[], status="active", event=None,
                     flags=[], days=days, o=c * open_mult, h=c, l=c, c=c, v=np.full(n, vol),
                     raw_close=c.copy())


def test_prepare_rows_counts_every_drop_and_keeps_orig_cik():
    m = cmap.build_map([("AAA", "1", "2017-03-01"), ("GOOG", "9", "2019-01-10"), ("GOOGL", "9", "2019-01-12")])
    rows = [{"cik": "0000001", "accepted_at": "2017-03-02T12:00:00Z"},
            {"cik": "9", "accepted_at": "2019-01-11T12:00:00Z"},      # ambiguous
            {"cik": "77", "accepted_at": "2018-01-01T12:00:00Z"},     # none
            {"cik": "1", "accepted_at": None}]                        # undated
    out, st = lp.prepare_rows(rows, m)
    assert [r["cik"] for r in out] == ["AAA"] and out[0]["cik_orig"] == "0000001"
    assert (st["padded"], st["ambiguous"], st["none"], st["undated"], st["rows_out"]) == (1, 1, 1, 1, 1)


def test_eligibility_uses_the_bar_before_entry_never_the_entry_bar():
    s = _series("X", n=60, vol=1_000_000.0)
    v = s.v.copy(); v[30] = 1e12; v[31] = 0.0      # a huge spike ON bar 30, dead volume on 31
    s.v = v
    el = lp._pre_entry_eligible(s)
    assert not el[0]                                # no prior bar
    assert el[31]                                   # bar 30's trailing ADV (spike) is visible to bar 31's entry
    assert el[30] == lp.pp.eligible_mask(s, lp.MIN_PRICE, lp.MIN_ADV)[29]   # shifted by exactly one bar
    low = _series("L", n=60, px=1.0)                # raw close below the $2 floor
    assert not lp._pre_entry_eligible(low).any()


def test_merge_dedupes_days_and_counts_multi_series(monkeypatch):
    spy = _series("SPY", n=1200)
    a, b = _series("ABC", n=100, start=100), _series("ABC", n=100, start=150, px=99.0)   # overlap 150..199
    monkeypatch.setattr(pp, "iter_series", lambda d: iter([spy, a, b]))
    pa, st = lp.build_panel("ignored", {"ABC"})
    assert st["multi_series_symbols"] == 1 and st["symbols_in_panel"] == 1
    d150 = pp.day_str(D0 + 160)
    assert pa.close("ABC", d150) == 10.0            # first series wins on a duplicate day
    assert pa.close("ABC", pp.day_str(D0 + 230)) == 99.0
    assert pa.close("ABC", pp.day_str(D0 + 20)) is None            # no forward-filled price
    assert pa.last_close_on_or_before("ABC", pp.day_str(D0 + 900)) == (pp.day_str(D0 + 249), 99.0)


def test_end_to_end_plumbing_and_resolved_not_in_panel(monkeypatch):
    spy = _series("SPY", n=1200)
    names = [_series(f"N{i}", n=1200) for i in range(3)]
    monkeypatch.setattr(pp, "iter_series", lambda d: iter([spy] + names))
    m = cmap.build_map([(f"N{i}", str(i + 1), "2017-01-05") for i in range(3)] + [("GHOST", "50", "2017-01-05")])
    rows = [{"form": "10-K", "cik": str(i + 1), "accession": f"a{i}", "accepted_at": "2017-01-20T12:00:00Z",
             "sections": {"1A": {"cosine_tfidf": 0.5 + i / 10}}} for i in range(3)]
    rows.append({"form": "10-K", "cik": "50", "accession": "g", "accepted_at": "2017-01-20T12:00:00Z",
                 "sections": {"1A": {"cosine_tfidf": 0.9}}})
    res = lp.run_gate2(rows, m, "ignored")
    assert res["join"]["resolved_not_in_panel"] == 1           # GHOST resolved but has no prices: counted, not dropped silently
    assert res["n_obs"] == 3 and res["verdict"]["result"] in ("INSUFFICIENT", "FAIL", "PASS", "PASS_PROVISIONAL", "INCONCLUSIVE")
