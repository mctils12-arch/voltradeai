"""Offline tests for sec_filings_gate2.py (Lazy Prices PR 3) — synthetic panels only.
The harness must (a) recover a planted effect, (b) not find one in noise,
(c) be unable to see the future."""
import random
from datetime import date, timedelta

import sec_filings_gate2 as g


def trading_days(start="2013-01-02", n=1500):
    d, out = date.fromisoformat(start), []
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


DAYS = trading_days()


def make_world(seed, n_per_month=40, effect=0.0, first="2013-03", months=60, noise=0.03):
    """One 10-K row per name per month; flat price path then a total 63d return
    of effect*(sim-0.5)*2 + noise, spread linearly. Returns (rows, panel)."""
    rng = random.Random(seed)
    rows, px = [], {}
    y, m = int(first[:4]), int(first[5:])
    for _ in range(months):
        month_days = [d for d in DAYS if d.startswith(f"{y:04d}-{m:02d}")]
        if month_days:
            acc_day = month_days[2]
            for i in range(n_per_month):
                cik = f"{y}{m:02d}{i:03d}"
                sim = rng.random()
                total = effect * (sim - 0.5) * 2 + rng.gauss(0, noise)
                rows.append({"cik": cik, "accession": cik, "form": "10-K",
                             "accepted_at": f"{acc_day}T12:00:00Z",   # 07:00/08:00 ET
                             "accepted_at_source": "index_headers",
                             "sections": {"1A": {"excluded": None, "cosine_tfidf": sim}}})
                idx0 = DAYS.index(acc_day)
                path = {}
                for k in range(idx0, min(idx0 + 70, len(DAYS))):
                    frac = (k - idx0) / 62 if k - idx0 <= 62 else 1.0
                    p = 100 * (1 + total * frac)
                    path[DAYS[k]] = (p, p)
                path[DAYS[idx0]] = (100.0, 100 * (1 + total * 0))
                px[cik] = path
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return rows, g.SimplePanel(DAYS, px)


def test_entry_index_never_before_public():
    days = ["2024-03-04", "2024-03-05", "2024-03-06"]   # Mon..Wed
    # 06:30 ET Tue (EST = UTC-5 => 11:30Z): tradable at Tue's own open
    assert days[g.entry_index(days, "2024-03-05T11:30:00Z")] == "2024-03-05"
    # 16:08 ET Tue: next session
    assert days[g.entry_index(days, "2024-03-05T21:08:00Z")] == "2024-03-06"
    # 09:15 ET Tue (after the 09:00 cutoff): next session, not the open it missed
    assert days[g.entry_index(days, "2024-03-05T14:15:00Z")] == "2024-03-06"
    # weekend/holiday-style gap: Saturday acceptance -> next listed session
    sat = ["2024-03-08", "2024-03-11"]
    assert sat[g.entry_index(sat, "2024-03-09T15:00:00Z")] == "2024-03-11"
    # DST: 12:00Z in July is 08:00 EDT (before cutoff), in January 07:00 EST
    jul = ["2024-07-09", "2024-07-10"]
    assert jul[g.entry_index(jul, "2024-07-09T12:00:00Z")] == "2024-07-09"
    # undated / past the end of the panel -> None, never a guess
    assert g.entry_index(days, None) is None
    assert g.entry_index(days, "2024-03-06T21:00:00Z") is None


def test_pre_acceptance_price_move_is_invisible():
    """Mutation guard: a huge jump the day BEFORE the filing must not enter the return."""
    days = trading_days(n=200)
    acc = days[100]
    px = {"a": {d: (100.0, 100.0) for d in days[90:180]}}
    px["a"][days[99]] = (100.0, 500.0)   # pre-filing spike
    panel = g.SimplePanel(days, px)
    idx = g.entry_index(days, f"{acc}T12:00:00Z")
    r, tag = g.forward_return(panel, "a", idx, 63)
    assert tag == "ok" and abs(r) < 1e-12


def test_delisted_exit_uses_last_close_and_is_flagged():
    days = trading_days(n=200)
    idx = 50
    px = {"a": {d: (100.0, 100.0) for d in days[idx:idx + 20]}}
    px["a"][days[idx + 19]] = (100.0, 40.0)    # collapses, then the series ends
    r, tag = g.forward_return(g.SimplePanel(days, px), "a", idx, 63)
    assert tag == "delisted_exit" and abs(r - (-0.6)) < 1e-12


def test_unmeasurable_rows_are_dropped_with_reasons():
    days = trading_days(n=100)
    panel = g.SimplePanel(days, {"a": {days[90]: (1.0, 1.0)}, "b": {d: (1.0, 1.0) for d in days}},
                          eligible=lambda c, d: c != "c")
    assert g.forward_return(panel, "a", 90, 63) == (None, "no_future")
    assert g.forward_return(panel, "zz", 10, 63) == (None, "no_entry_price")
    assert g.forward_return(panel, "c", 10, 63) == (None, "ineligible")


def test_signal_value_respects_form_exclusion_and_missing():
    row = {"form": "10-K", "sections": {"1A": {"excluded": None, "cosine_tfidf": 0.8}}}
    assert g.signal_value(row) == 0.8
    assert g.signal_value({**row, "form": "10-Q"}) is None
    assert g.signal_value({"form": "10-K", "sections": {"1A": {"excluded": "short", "cosine_tfidf": 0.8}}}) is None
    assert g.signal_value({"form": "10-K", "sections": {"1A": {"excluded": None, "cosine_tfidf": None}}}) is None


def test_quantile_sort_is_deterministic_and_balanced():
    obs = [{"cik": str(i), "signal": (i * 7) % 10 / 10, "ret": 0.0} for i in range(50)]
    b1, b2 = g.quantile_sort(obs), g.quantile_sort(list(reversed(obs)))
    assert [len(q) for q in b1] == [10] * 5
    assert [[o["cik"] for o in q] for q in b1] == [[o["cik"] for o in q] for q in b2]
    assert max(o["signal"] for o in b1[0]) <= min(o["signal"] for o in b1[-1])


def test_planted_effect_is_recovered_oos_and_costs_are_subtracted():
    rows, panel = make_world(1, effect=0.06)
    res = g.run(rows, panel, variants=3)
    oos = res["summary"]["out_of_sample"]
    assert oos["cohorts"] >= g.MIN_OOS_COHORTS
    assert oos["mean_gross"] > 0.03 and oos["t_nw_net"] > 5
    assert abs((oos["mean_gross"] - oos["mean_net"]) - 0.002) < 1e-12   # 2 legs x 10 bps
    assert res["verdict"]["result"] == "PASS"
    ex = oos["q_excess_mean"]
    assert ex[0] < ex[2] < ex[4]            # monotone around the base rate


def test_reversed_effect_fails():
    rows, panel = make_world(2, effect=-0.06)
    assert g.run(rows, panel)["verdict"]["result"] == "FAIL"


def test_null_world_does_not_pass():
    passes = sum(g.run(*make_world(s, effect=0.0), variants=3)["verdict"]["result"] == "PASS" for s in range(10, 16))
    assert passes == 0


def test_effect_smaller_than_costs_fails_net():
    rows, panel = make_world(3, effect=0.0005, noise=0.002)   # gross spread ~0.05% << 20 bps
    res = g.run(rows, panel)
    assert res["summary"]["out_of_sample"]["mean_gross"] > 0
    assert res["summary"]["out_of_sample"]["mean_net"] < 0
    assert res["verdict"]["result"] == "FAIL"


def test_short_oos_window_is_insufficient_not_fail():
    rows, panel = make_world(4, effect=0.06, first="2014-06", months=12)   # only ~6 OOS cohorts
    assert g.run(rows, panel)["verdict"]["result"] == "INSUFFICIENT"


def test_in_sample_never_decides_the_verdict():
    rows, panel = make_world(5, effect=0.06, first="2013-03", months=22)    # all pre-2015
    res = g.run(rows, panel)
    assert res["summary"]["in_sample"]["cohorts"] > 12 and res["summary"]["out_of_sample"]["cohorts"] == 0
    assert res["verdict"]["result"] == "INSUFFICIENT"


def test_nw_t_matches_plain_t_at_lag_zero_and_shrinks_with_autocorrelation():
    x = [0.01, 0.03, -0.01, 0.02, 0.025, 0.0, 0.015, 0.03]
    n, m = len(x), sum(x) / len(x)
    sd = (sum((v - m) ** 2 for v in x) / n) ** 0.5
    assert abs(g.nw_tstat(x, 0) - m / (sd / n ** 0.5)) < 1e-12
    pos = [0.01 * (1 + (i // 4)) for i in range(24)]    # strongly autocorrelated trend
    assert g.nw_tstat(pos, 2) < g.nw_tstat(pos, 0)
    assert g.nw_tstat([1.0], 2) is None and g.nw_tstat([0.5] * 5, 2) is None


def test_critical_t_rises_with_variants_tried():
    assert abs(g.critical_t(1) - 1.96) < 0.01
    assert g.critical_t(3) > g.critical_t(1) and g.critical_t(10) > g.critical_t(3)


def test_small_cohorts_are_skipped():
    rows, panel = make_world(6, n_per_month=10, effect=0.06)
    res = g.run(rows, panel)
    assert res["cohorts"] == [] and res["verdict"]["result"] == "INSUFFICIENT"


def test_unverified_timestamp_is_surfaced_not_hidden():
    rows, panel = make_world(7, effect=0.0, months=3)
    for r in rows[:5]:
        r["accepted_at_source"] = "submissions_json_unverified"
    assert g.run(rows, panel)["drops"].get("note_unverified_timestamp") == 5
