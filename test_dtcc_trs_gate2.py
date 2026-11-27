"""dtcc_trs_gate2.py — the registered gate-2 test of the DTCC equity-swap root.

Pins the registration's rules as code (research/open_questions.md 2026-11-25
and the 2026-11-26 addenda): the verdict table, the one-look rule, the join
order, the filter lines, entry at the known day's close with nothing read
before it or after the exit, the comparison set, the arming guard, and that
the output names no security. End to end on a synthetic price panel: a
planted 20-day drift is found, noise is killed by the cost, and a move that
happened before the entry is not credited.
"""
import datetime as dt
import gzip
import json
import math
import os
import random

import numpy as np
import pytest

import cusip_ftd
import dtcc_trs_gate2 as g2
import dtcc_trs_panel as panel
import price_panel as pp
import walkforward_synth as ws

CLOSED = {d.isoformat() for d in panel.nyse_closures()}
DAYS = ws.bdays("2024-01-02", "2025-12-31", CLOSED)
AS_OF = DAYS[-1]


# ── statistics and the verdict table ────────────────────────────────────────
def test_clustered_mean_is_the_plain_standard_error_when_every_event_is_its_own_day():
    rng = np.random.default_rng(3)
    x = rng.normal(0.01, 0.05, 500)
    b = g2.clustered_mean(x, np.arange(500))
    plain = x.std(ddof=1) / math.sqrt(500)
    assert b["n"] == 500 and b["clusters"] == 500
    assert abs(b["se_pct"] / 100.0 - plain) < 1e-5
    assert abs(b["t"] - x.mean() / plain) < 0.01


def test_clustered_mean_widens_when_a_day_moves_together():
    rng = np.random.default_rng(4)
    day_shock = np.repeat(rng.normal(0, 0.05, 50), 10)       # 50 entry days, 10 events each, one shock a day
    x = 0.01 + day_shock + rng.normal(0, 0.01, 500)
    by_day = g2.clustered_mean(x, np.repeat(np.arange(50), 10))
    naive = g2.clustered_mean(x, np.arange(500))
    assert by_day["se_pct"] > 2.5 * naive["se_pct"]
    assert g2.clustered_mean(np.array([]), np.array([]))["mean_pct"] is None
    assert g2.clustered_mean(np.array([0.1, 0.2]), np.array([7, 7]))["t"] is None      # one entry day: no standard error


def _pooled(n, mean, t):
    return {"n": n, "mean_pct": mean, "t": t}


def _split(a, b):
    return {"first": {"mean_pct": a}, "second": {"mean_pct": b}}


def test_verdict_is_item_8_of_the_registration():
    assert g2.verdict(_pooled(299, 5.0, 9.0), _split(1, 1))["verdict"] == "INSUFFICIENT"
    assert g2.verdict(_pooled(300, 0.0, 0.0), _split(1, -1))["verdict"] == "KILLED"          # at or below 0
    assert g2.verdict(_pooled(300, -0.4, -2.0), _split(1, 1))["verdict"] == "KILLED"
    assert g2.verdict(_pooled(300, 0.4, 0.99), _split(1, 1))["verdict"] == "KILLED"          # t under 1.0
    assert g2.verdict(_pooled(300, 0.4, 1.0), _split(1, 1))["verdict"] == "INCONCLUSIVE"
    assert g2.verdict(_pooled(300, 0.4, 2.99), _split(1, 1))["verdict"] == "INCONCLUSIVE"
    assert g2.verdict(_pooled(300, 0.4, 3.0), _split(0.1, 0.7))["verdict"] == "PASS"
    assert g2.verdict(_pooled(300, 0.4, 5.0), _split(-0.1, 0.9))["verdict"] == "INCONCLUSIVE"  # one half not positive
    assert g2.verdict(_pooled(300, 0.4, 5.0), _split(0.0, 0.9))["verdict"] == "INCONCLUSIVE"
    assert g2.verdict(_pooled(300, 0.4, None), _split(1, 1))["verdict"] == "INSUFFICIENT"
    assert (g2.MIN_TESTED, g2.T_PASS, g2.T_KILL, g2.COST, g2.HORIZON) == (300, 3.0, 1.0, 0.01, 20)
    assert (g2.MIN_BARS_BEFORE, g2.MIN_RAW_CLOSE, g2.DV_WINDOW, g2.DV_MIN, g2.DV_MAX) == (126, 2.0, 20, 500_000.0, 25_000_000.0)


def test_one_look_then_one_more_after_126_trading_days_then_killed():
    cal = DAYS
    assert g2.look_due(None, cal[300], cal) == (True, 1, "no look recorded")
    killed = {"looks": [{"verdict": "KILLED", "at": "t", "events_last_testable": cal[300]}], "final": "KILLED"}
    assert g2.look_due(killed, cal[-1], cal)[0] is False
    inc = {"looks": [{"verdict": "INCONCLUSIVE", "at": "t", "events_last_testable": cal[300]}], "final": None}
    due, n, why = g2.look_due(inc, cal[300 + 125], cal)
    assert due is False and "125 so far" in why
    assert g2.look_due(inc, cal[300 + 126], cal)[:2] == (True, 2)
    spent = {"looks": [inc["looks"][0], inc["looks"][0]], "final": None}
    assert g2.look_due(spent, cal[-1], cal)[0] is False
    assert g2.final_after(1, "PASS") == "PASS" and g2.final_after(1, "KILLED") == "KILLED"
    assert g2.final_after(1, "INCONCLUSIVE") is None
    assert g2.final_after(2, "INCONCLUSIVE") == "KILLED" and g2.final_after(2, "PASS") == "PASS"
    assert g2.ladder_word("PASS") == "PASS" and g2.ladder_word("KILLED") == "FAIL" and g2.ladder_word(None) is None


# ── the join ────────────────────────────────────────────────────────────────
def test_join_order_is_dated_pair_then_todays_ticker_then_a_later_pair():
    us = {"found": True, "us": True, "ticker": "NEW"}
    gone = {"found": True, "us": False, "ticker": "XLON"}
    before = [("2025-03-01", "2025-03-20", "OLD", 9)]
    after = [("2025-04-10", "2025-04-20", "LATE", 5)]
    assert g2.join_symbol(before, us, "2025-03-25") == ("OLD", "a")        # the dated pair wins over today's ticker
    assert g2.join_symbol(after, us, "2025-03-25") == ("NEW", "b")         # a pair printed only afterwards does not
    assert g2.join_symbol(after, gone, "2025-03-25") == ("LATE", "c")
    assert g2.join_symbol([], gone, "2025-03-25") == (None, "none")
    assert g2.join_symbol([], None, "2025-03-25") == (None, "none")
    assert g2.join_symbol(before, gone, "2025-09-01") == (None, "none")    # outside the 45 days
    # the shortcut is taken only when every branch would give the same symbol
    assert g2.one_symbol([("2025-03-01", "2025-03-20", "NEW", 9)], us) == "NEW"
    assert g2.one_symbol([], us) == "NEW"
    assert g2.one_symbol(before, us) is None and g2.one_symbol([], gone) is None


# ── the price pass on hand-made series ──────────────────────────────────────
def _flat_rows(days, closes, volume=100_000):
    return [{"d": d, "o": c, "h": c, "l": c, "c": c, "v": volume, "rc": c} for d, c in zip(days, closes)]


def _series(tmp_path, lines):
    root = str(tmp_path / "pp")
    ws.write_panel(root, lines, AS_OF)
    return {s.symbol: s for s in pp.iter_series(pp.current_build(root)["dir"])}


def _price_one(s, t, known, exit20, exit5):
    rows = g2.Rows(1)
    ids = np.array([0])
    f = lambda d: np.array([pp.day_num(d)])  # noqa: E731
    g2.price_series(s, ids, f(t), f(known), {20: f(exit20), 5: f(exit5)}, rows)
    return rows


def test_entry_is_the_known_days_close_and_nothing_after_the_exit_is_read(tmp_path):
    days = DAYS[:200]
    closes = [10.0 + 0.01 * i for i in range(200)]
    t = 150
    a = _series(tmp_path / "a", [ws.line("AAA", _flat_rows(days, closes))])["AAA"]
    r = _price_one(a, days[t], days[t + 1], days[t + 21], days[t + 6])
    assert r.hits[0] == 1 and r.bars_before[0] == t and r.entry_ok[0]
    assert r.ret[20][0] == pytest.approx(closes[t + 21] / closes[t + 1] - 1.0, abs=1e-9)
    assert r.ret[5][0] == pytest.approx(closes[t + 6] / closes[t + 1] - 1.0, abs=1e-9)
    assert r.raw_close[0] == pytest.approx(closes[t]) and not r.ended[0]
    # the 20 bars strictly BEFORE the event day: a volume spike ON the day changes nothing
    spiked = _flat_rows(days, closes)
    spiked[t]["v"] = 10**9
    b = _series(tmp_path / "b", [ws.line("AAA", spiked)])["AAA"]
    assert _price_one(b, days[t], days[t + 1], days[t + 21], days[t + 6]).dv_median[0] == pytest.approx(r.dv_median[0])
    assert r.dv_median[0] == pytest.approx(np.median([closes[i] * 100_000 for i in range(t - 20, t)]), rel=1e-6)
    # prices after the exit day, and before the entry, do not move the return
    later = list(closes)
    for i in range(t + 22, 200):
        later[i] *= 3.0
    later[t] *= 0.5
    c = _series(tmp_path / "c", [ws.line("AAA", _flat_rows(days, later))])["AAA"]
    assert _price_one(c, days[t], days[t + 1], days[t + 21], days[t + 6]).ret[20][0] == pytest.approx(r.ret[20][0], abs=1e-9)


def test_a_series_that_ends_before_the_horizon_exits_at_its_last_close_and_no_bar_means_no_series(tmp_path):
    days = DAYS[:200]
    closes = [10.0] * 151 + [10.0, 11.0, 12.0, 6.0]         # bars to index 154; the name is gone after
    s = _series(tmp_path, [ws.line("GONE", _flat_rows(days[:155], closes), status="inactive")])["GONE"]
    r = _price_one(s, days[150], days[151], days[171], days[156])
    assert r.entry_ok[0] and r.ended[0]
    assert r.ret[20][0] == pytest.approx(6.0 / 10.0 - 1.0, abs=1e-9)       # the loss is counted, not dropped
    none = _price_one(s, days[160], days[161], days[181], days[166])       # no bar on the event day
    assert none.hits[0] == 0
    last = _price_one(s, days[154], days[155], days[175], days[160])       # a bar on the event day, none on the entry day
    assert last.hits[0] == 1 and not last.entry_ok[0] and math.isnan(last.ret[20][0])


def test_classify_applies_the_filter_lines_in_order():
    rows = g2.Rows(9)
    rows.hits[:] = [0, 0, 2, 1, 1, 1, 1, 1, 1]
    rows.bars_before[:] = [0, 0, 500, 125, 500, 500, 500, 500, 500]
    rows.raw_close[:] = [np.nan, np.nan, 9, 9, 1.99, 9, 9, 9, 9]
    rows.dv_median[:] = [np.nan, np.nan, 1e6, 1e6, 1e6, 499_999, 1e6, 1e6, 25e6]
    rows.entry_ok[:] = [False, False, True, True, True, True, False, True, True]
    rows.ret[20][:] = [np.nan, np.nan, 0.1, 0.1, 0.1, 0.1, np.nan, 0.1, 0.1]
    has = np.array([False, True, True, True, True, True, True, True, True])
    assert g2.classify(rows, has).tolist() == [g2.NO_SYMBOL, g2.NO_SERIES, g2.AMBIGUOUS, g2.SHORT_HISTORY, g2.LOW_PRICE,
                                                 g2.TOO_THIN, g2.NO_ENTRY_BAR, g2.SMALL, g2.LARGE]


# ── end to end on a synthetic world ─────────────────────────────────────────
def _walk(rng, n, p0=20.0, vol=0.02):
    out, p = [], p0
    for _ in range(n):
        p *= math.exp(rng.gauss(0, vol))
        out.append(p)
    return out


def _world(tmp_path, n_days=40, per_day=10, effect=0.0, jump_before_entry=0.0, n_cmp=80, drop_event_prices=False,
           reproduced=True, extra=None):
    """A DTCC root and a price panel: `per_day` events on each of `n_days`
    event days, each on its own name, and `n_cmp` names with a swap row on
    every event day. `effect` is a drift planted over the 20 days after the
    entry close; `jump_before_entry` a move between the event day's close
    and the entry close."""
    rng = random.Random(11)
    dtcc, prices = str(tmp_path / "dtcc"), str(tmp_path / "price_panel")
    os.makedirs(os.path.join(dtcc, "panel"))
    os.makedirs(os.path.join(dtcc, "figi"))
    ev_idx = [300 + 3 * k for k in range(n_days)]
    lines, events, panel_rows, figi = [], [], [], []

    def name(sym, cusip, closes, volume=100_000, stype="Common Stock", days=DAYS, **kw):
        if closes is not None:
            lines.append(ws.line(sym, _flat_rows(days, closes, volume), **kw))
        figi.append({"cusip": cusip, "asked_at": "t", "found": True, "us": True, "listings": 1, "ticker": sym, "name": sym,
                     "security_type": stype, "market_sector": "Equity"})

    for j in range(n_cmp):
        name(f"C{j:03d}", f"00000C{j:03d}", _walk(rng, len(DAYS)))
    for d, i in enumerate(ev_idx):
        for k in range(per_day):
            sym, cusip = f"E{d:02d}{k:02d}", f"0000E{d:02d}{k:02d}"
            closes = _walk(rng, len(DAYS))
            if jump_before_entry:
                for q in range(i + 1, len(DAYS)):
                    closes[q] *= 1.0 + jump_before_entry
            if effect:
                for q in range(i + 2, len(DAYS)):
                    closes[q] *= (1.0 + effect) ** (min(q - (i + 1), 20) / 20.0)
            name(sym, cusip, None if drop_event_prices else closes)
            events.append({"cusip": cusip, "date": DAYS[i], "known": DAYS[i + 1], "newt_usd": 5_000_000, "newt_n": 3,
                           "prior_max_usd": 0, "prior_days_with_new": 0})
            panel_rows.append((DAYS[i], cusip, 3, 5_000_000, 0))
        for j in range(n_cmp):
            panel_rows.append((DAYS[i], f"00000C{j:03d}", 1, 1000, 0))
    for fn in (extra or []):
        fn(dict(name=name, events=events, panel_rows=panel_rows, ev_idx=ev_idx, rng=rng, dtcc=dtcc))
    ws.write_panel(prices, lines, AS_OF)
    with gzip.open(os.path.join(dtcc, "panel", "panel_2025Q1.csv.gz"), "wt", newline="") as f:
        f.write(",".join(panel.PANEL_FIELDS) + "\n")
        for d, c, n, usd, cap in sorted(panel_rows):
            f.write(f"{d},{c},{n},{usd},{cap},0,0,0,0,0\n")
    with open(os.path.join(dtcc, "panel", "manifest.json"), "w") as f:
        json.dump({"files": {"panel_2025Q1.csv.gz": {}}}, f)
    events.sort(key=lambda e: (e["date"], e["cusip"]))
    with gzip.open(os.path.join(dtcc, "panel", panel.EVENTS_FILE), "wt") as f:
        json.dump({"events": events, "summary": {"last_testable": events[-1]["date"]}}, f)
    with open(os.path.join(dtcc, "panel", "events_check.json"), "w") as f:
        json.dump({"events": len(events), "first_testable": events[0]["date"], "last_testable": events[-1]["date"],
                   "registered": {"match": reproduced}}, f)
    with open(os.path.join(dtcc, "figi", "cusip_figi.jsonl"), "w") as f:
        for e in figi:
            f.write(json.dumps(e) + "\n")
    return dtcc, prices, str(tmp_path / "out")


def _read(out, name):
    with open(os.path.join(out, name)) as f:
        return json.load(f)


def test_noise_is_killed_by_the_cost_and_the_look_is_recorded_once(tmp_path):
    dtcc, prices, out = _world(tmp_path)
    res = g2.run(dtcc, prices, out, code_version="t")
    assert res["status"] == "ok" and res["look"] == 1 and res["tested"] == 400
    assert res["verdict"] == "KILLED" and res["final"] == "KILLED"
    latest = _read(out, "latest.json")
    assert -2.0 < latest["statistic"]["pooled"]["mean_pct"] < 0.0           # about minus the 1.0% cost
    assert latest["statistic"]["pooled"]["clusters"] == 40
    assert latest["arming"]["armed"] and latest["arming"]["priced_share"] == 1.0
    assert latest["arming"]["comparison_names_per_entry_day"]["median"] == 80
    assert latest["counts"]["events_tested"] == 400 and latest["funnel_events"]["passes, under 25,000,000 USD"] == 400
    assert latest["beside_never_the_verdict"]["horizon_20d_before_cost"]["mean_pct"] == pytest.approx(
        latest["statistic"]["pooled"]["mean_pct"] + 1.0, abs=1e-3)
    reg = _read(out, "registered.json")
    assert len(reg["looks"]) == 1 and reg["final"] == "KILLED" and reg["ladder_verdict"] == "FAIL"
    # derived statistics only: no CUSIP and no ticker of the fixture anywhere in what is written
    text = "".join(open(os.path.join(out, n)).read() for n in ("latest.json", "registered.json", "last_run.json"))
    assert "0000E" not in text and "00000C" not in text and "E0001" not in text and "C000" not in text
    # a second run stops before a price file is opened and changes nothing
    before = open(os.path.join(out, "registered.json")).read()
    again = g2.run(dtcc, str(tmp_path / "no_such_price_panel"), out, code_version="t2")
    assert again["status"] == "skipped"
    again = g2.run(dtcc, prices, out, code_version="t2")
    assert again["status"] == "skipped" and "verdict of record: KILLED" in again["reason"]
    assert open(os.path.join(out, "registered.json")).read() == before


def test_a_planted_drift_after_the_entry_passes(tmp_path):
    dtcc, prices, out = _world(tmp_path, effect=0.08)
    res = g2.run(dtcc, prices, out, code_version="t")
    assert res["verdict"] == "PASS" and res["final"] == "PASS"
    latest = _read(out, "latest.json")
    assert 5.5 < latest["statistic"]["pooled"]["mean_pct"] < 8.5 and latest["statistic"]["pooled"]["t"] > 3.0
    assert latest["statistic"]["halves"]["first"]["mean_pct"] > 0 and latest["statistic"]["halves"]["second"]["mean_pct"] > 0
    assert latest["statistic"]["halves"]["first"]["n"] + latest["statistic"]["halves"]["second"]["n"] == 400
    assert _read(out, "registered.json")["ladder_verdict"] == "PASS"
    assert latest["beside_never_the_verdict"]["by_regime_at_entry"]["no label recorded"]["n"] == 400


def test_a_move_between_the_event_close_and_the_entry_close_is_not_credited(tmp_path):
    dtcc, prices, out = _world(tmp_path, jump_before_entry=0.10)
    res = g2.run(dtcc, prices, out, code_version="t")
    assert res["verdict"] == "KILLED"
    assert _read(out, "latest.json")["statistic"]["pooled"]["mean_pct"] < 0.0


def test_under_300_tested_events_writes_counts_and_no_statistic_and_takes_no_look(tmp_path):
    dtcc, prices, out = _world(tmp_path, n_days=20, per_day=10)
    res = g2.run(dtcc, prices, out, code_version="t")
    assert res["status"] == "insufficient" and res["tested"] == 200
    latest = _read(out, "latest.json")
    assert latest["statistic"] is None and latest["verdict"] == "INSUFFICIENT" and latest["counts"]["events_tested"] == 200
    assert "beside_never_the_verdict" not in latest
    assert not os.path.exists(os.path.join(out, "registered.json"))


def test_a_run_that_cannot_find_prices_is_not_armed_and_takes_no_look(tmp_path):
    dtcc, prices, out = _world(tmp_path, drop_event_prices=True)
    res = g2.run(dtcc, prices, out, code_version="t")
    assert res["status"] == "not_armed"
    latest = _read(out, "latest.json")
    assert latest["arming"]["armed"] is False and latest["arming"]["priced_share"] == 0.0
    assert latest["statistic"] is None and latest["look"] is None
    assert latest["funnel_events"]["no price series with a bar on the day"] == 400
    assert not os.path.exists(os.path.join(out, "registered.json"))
    # too few comparison names is the other half of the guard
    dtcc, prices, out = _world(tmp_path / "thin", n_cmp=10)
    assert g2.run(dtcc, prices, out, code_version="t")["status"] == "not_armed"


def test_the_test_does_not_start_on_a_list_that_is_not_the_registered_one(tmp_path):
    dtcc, prices, out = _world(tmp_path, n_days=2, per_day=2, reproduced=False)
    res = g2.run(dtcc, str(tmp_path / "no_such_price_panel"), out, code_version="t")
    assert res["status"] == "insufficient" and "NOT reproduced" in res["reason"]
    assert not os.path.exists(os.path.join(out, "latest.json"))


def test_filter_lines_types_the_comparison_window_and_a_renamed_name(tmp_path):
    def more(w):
        name, events, panel_rows, ev_idx, rng, dtcc = (w[k] for k in ("name", "events", "panel_rows", "ev_idx", "rng", "dtcc"))
        i = ev_idx[5]
        d, known = DAYS[i], DAYS[i + 1]

        def event(cusip):
            events.append({"cusip": cusip, "date": d, "known": known, "newt_usd": 9_000_000, "newt_n": 1,
                           "prior_max_usd": 0, "prior_days_with_new": 0})
            panel_rows.append((d, cusip, 1, 9_000_000, 0))

        name("PENNY", "0000X0001", [1.5] * len(DAYS)); event("0000X0001")                       # raw close under 2 USD
        name("THIN", "0000X0002", _walk(rng, len(DAYS)), volume=1_000); event("0000X0002")      # about 20,000 USD a day
        name("BIG", "0000X0003", _walk(rng, len(DAYS)), volume=5_000_000); event("0000X0003")   # about 100 million USD
        name("NEWCO", "0000X0004", _walk(rng, 100), days=DAYS[i - 99:i + 1] + DAYS[i + 1:][:0]); event("0000X0004")  # 99 bars before
        name("FUND", "0000X0005", _walk(rng, len(DAYS)), stype="ETP"); event("0000X0005")       # not a kept type
        # a capped row on a passing event
        cap = next(r for r in panel_rows if r[1] == "0000E0500")
        panel_rows[panel_rows.index(cap)] = cap[:4] + (2,)
        # a comparison name inside its own event's window: C000 had an event 5 trading days earlier
        events.append({"cusip": "00000C000", "date": DAYS[i - 5], "known": DAYS[i - 4], "newt_usd": 2_000_000, "newt_n": 1,
                       "prior_max_usd": 0, "prior_days_with_new": 0})
        # a renamed company: OpenFIGI knows it as RENAMED today; the SEC printed it as OLDNAME around the event
        name("RENAMED", "0000X0006", None)
        w["name"]("OLDNAME", "0000X0099", _walk(rng, i + 40), days=DAYS[:i + 40], status="inactive")
        event("0000X0006")
        cusip_ftd.write_period(os.path.join(dtcc, "ftd"), cusip_ftd.period_of(dt.date.fromisoformat(d)),
                               {("0000X0006", "OLDNAME"): [DAYS[i - 3], DAYS[i - 1], 3]})

    dtcc, prices, out = _world(tmp_path, extra=[more])
    res = g2.run(dtcc, prices, out, code_version="t")
    assert res["status"] == "ok"
    latest = _read(out, "latest.json")
    f = latest["funnel_events"]
    assert f["raw close under 2 USD"] == 1 and f["median dollar volume under 500,000 USD"] == 1
    assert f["passes, at or above 25,000,000 USD"] == 1 and f["under 126 bars before the day"] == 1
    assert latest["counts"]["events_of_other_types"] == {"ETP": 1}
    # 400 planted events, the renamed one joined through the SEC pair, and C000's own event, which passes
    # the filter but falls on a day with no comparison name: not tested, counted (R5)
    assert f["passes, under 25,000,000 USD"] == 402 and res["tested"] == 401
    assert latest["counts"]["events_on_a_day_with_no_comparison_name"] == 1
    carriers = latest["beside_never_the_verdict"]["tested_by_join_carrier"]
    assert carriers == {"a": 1, "b": 400, "c": 0}
    assert latest["beside_never_the_verdict"]["events_with_a_capped_row"]["n"] == 1
    assert latest["beside_never_the_verdict"]["events_at_or_above_25m_dollar_volume"]["n"] == 0     # no comparison name that large
    # C000 is out of the comparison set for the 20 trading days after its event: 7 of the 40 event days
    assert latest["counts"]["comparison_rows_dropped_inside_own_event_window"] == 7
    assert latest["counts"]["comparison_rows"] == 40 * 80 - 7
    assert latest["join"]["events_by_half_year"]["2025H1"]["a"] == 1


def test_an_event_whose_window_ends_after_the_price_panel_is_counted_and_not_tested(tmp_path):
    def late(w):
        d = DAYS[-10]
        w["name"]("LATE", "0000X0007", _walk(w["rng"], len(DAYS)))
        w["events"].append({"cusip": "0000X0007", "date": d, "known": DAYS[-9], "newt_usd": 5_000_000, "newt_n": 1,
                            "prior_max_usd": 0, "prior_days_with_new": 0})
        w["panel_rows"].append((d, "0000X0007", 1, 5_000_000, 0))

    dtcc, prices, out = _world(tmp_path, extra=[late])
    res = g2.run(dtcc, prices, out, code_version="t")
    c = _read(out, "latest.json")["counts"]
    assert res["tested"] == 400 and c["events_window_after_price_panel"] == 1 and c["events_in_list"] == 401


def test_cli_prints_one_json_line_and_exits_zero_without_inputs(tmp_path, capsys):
    rc = g2.main(["--run", "--dtcc-root", str(tmp_path / "none"), "--out-dir", str(tmp_path / "out")])
    line = json.loads(capsys.readouterr().out.strip().split("\n")[-1])
    assert rc == 0 and line["status"] == "insufficient"


def test_regime_label_is_the_one_recorded_on_or_just_before_the_entry_day(tmp_path):
    p = tmp_path / "hist.jsonl"
    p.write_text('{"as_of": "2026-11-05", "composite_regime": "RISK_ON"}\nnot json\n'
                 '{"as_of": "2026-11-10", "composite_regime": "CRISIS"}\n{"as_of": "2026-11-12"}\n')
    labels = g2.read_regime_labels(str(p))
    assert labels == [("2026-11-05", "RISK_ON"), ("2026-11-10", "CRISIS")]
    assert g2.regime_on(labels, "2026-11-04") is None                 # nothing recorded yet: never a later label
    assert g2.regime_on(labels, "2026-11-05") == "RISK_ON" and g2.regime_on(labels, "2026-11-09") == "RISK_ON"
    assert g2.regime_on(labels, "2026-11-10") == "CRISIS" and g2.regime_on(labels, "2026-11-15") == "CRISIS"
    assert g2.regime_on(labels, "2026-11-16") is None                 # older than 5 days: not a label for that day
    assert g2.read_regime_labels(str(tmp_path / "none.jsonl")) == [] and g2.read_regime_labels(None) == []
