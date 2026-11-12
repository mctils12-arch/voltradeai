"""
event_reaction_panel.py — the pre-registered event-reaction study re-run on the
delisting-inclusive price panel. Synthetic panels and SEC fixtures only, no
network. Pins: the panel source reproduces the survivor (Yahoo-shaped) source
exactly on an overlap; a delisted name's event is measured with the panel's
delisting assumption (checked against an independent re-implementation of the
windows and controls); the raw-close floor; identity rules for survivors,
delisted CIKs and reused tickers; equivalence of the SEC stages to the core's;
the rule-4 reading on the panel; an offline end-to-end run that writes
aggregates only; the memory budget; the study-code hash shared with
server/eventStudy.ts.
"""
import io
import json
import os
import random
import zipfile
from collections import Counter

import pytest

import event_reaction as er
import event_reaction_panel as erp
import price_panel as pp
import sec_cik_symbol_map as csm
import walkforward_synth as ws

DAYS = ws.bdays("2016-01-04", "2018-12-31")
E2, E3, E4 = er.EVENT_TYPES[1], er.EVENT_TYPES[2], er.EVENT_TYPES[3]


def _bars(rows):
    return {"date": [r["d"] for r in rows], "close": [r["c"] for r in rows], "high": [r["h"] for r in rows],
            "low": [r["l"] for r in rows], "volume": [r["v"] for r in rows]}


def _panel(tmp_path, lines, as_of=None):
    root = str(tmp_path / "pp")
    ws.write_panel(root, lines, as_of=as_of or DAYS[-1])
    build = pp.current_build(root)
    spy, idx = erp.index_panel(build["dir"])
    days = spy.dates
    spy_ret, spy_bull = er.spy_series(days, erp.series_bars(spy))
    return build, days, spy_ret, spy_bull, idx


def _spy_line(rng):
    return ws.line("SPY", ws.path_rows(DAYS, 200.0, 0.0003, 0.01, 8e7, rng))


def _ev(sym_events, origin="survivor", confirms=True, span=None):
    return [(et, i0, "0000000001", origin, confirms, span) for et, i0 in sym_events]


# ── the panel source reproduces the survivor source on an overlap ──────────
def test_panel_source_matches_yahoo_source_on_overlap(tmp_path):
    rng = random.Random(3)
    spy_rows = ws.path_rows(DAYS, 200.0, 0.0003, 0.01, 8e7, rng)
    rows = ws.path_rows(DAYS, 40.0, 0.0, 0.02, 3e6, rng)          # raw close == adjusted close
    build, days, spy_ret, spy_bull, _ = _panel(tmp_path, [ws.line("SPY", spy_rows), ws.line("QZA", rows)])
    assert days == DAYS
    # the panel's decoded SPY gives the same market series as the original rows
    yr, yb = er.spy_series(DAYS, _bars(spy_rows))
    assert all((a is None and b is None) or abs(a - b) < 1e-12 for a, b in zip(yr, spy_ret))
    assert yb == spy_bull
    events = {E2: [150, 300, 301, 450, 600], E4: [500]}
    yahoo, ym_y = er.measure_ticker("QZA", days, spy_ret, spy_bull, _bars(rows), events)
    recs, yearly, diag, unmatched = erp.measure_panel(
        build["dir"], days, spy_ret, spy_bull, DAYS[-1], {"QZA": _ev([(et, i) for et, l in events.items() for i in l])},
        {"QZA"}, 4000)
    assert len(yahoo) == len(recs) == 5          # 300/301 dedup to one
    key = lambda r: (r["type"], r["day0"])
    for a, b in zip(sorted(yahoo, key=key), sorted(recs, key=key)):
        for f in ("type", "day0", "regime", "n_ctrl"):
            assert a[f] == b[f]
        for f in ("medvol", "spread"):
            assert a[f] == pytest.approx(b[f], rel=1e-12, abs=1e-15)
        for f in ("car", "excess"):
            assert set(a[f]) == set(b[f])
            for w in a[f]:
                assert a[f][w] == pytest.approx(b[f][w], abs=1e-12)
        assert b["origin"] == "survivor" and b["delist_day"] is None and "ticker" not in b
    assert yearly == {y: [v] for y, v in ym_y.items()}
    assert not unmatched


# ── a delisted name's event is in, with the delisting assumption applied ───
def _ref_ar(close, spy_ret, last, r_del, n):
    """Independent re-implementation of PD3: abnormal returns with the delisting tail."""
    ar = [None] * n
    for t in range(1, n):
        if t <= last:
            if close[t] is not None and close[t - 1] is not None:
                ar[t] = close[t] / close[t - 1] - 1 - spy_ret[t]
        elif t == last + 1:
            ar[t] = r_del - spy_ret[t]
        else:
            ar[t] = 0.0
    return ar


def _ref_car(ar, close, j, lo, hi):
    e = j + lo - 1
    if e < 0 or close[e] is None or j + hi >= len(ar) or j + lo < 1:
        return None
    vals = ar[j + lo:j + hi + 1]
    return None if any(v is None for v in vals) else sum(vals)


def _check_delisted(tmp_path, status, event, flags, expect_r):
    rng = random.Random(5)
    last = 400
    rows = ws.path_rows(DAYS[:last + 1], 30.0, 0.0, 0.02, 2e6, rng)
    ev = event(rows) if callable(event) else event
    build, days, spy_ret, spy_bull, _ = _panel(tmp_path, [_spy_line(rng), ws.line("QZD", rows, status=status, event=ev, flags=flags)])
    i0 = last - 5
    recs, _y, _d, _u = erp.measure_panel(build["dir"], days, spy_ret, spy_bull, DAYS[-1], {"QZD": _ev([(E3, i0)])}, {"QZD"}, 4000)
    assert len(recs) == 1
    r = recs[0]
    assert r["delist_day"] == 6 and r["delist_kind"] == ("merger" if expect_r == 0.0 else "other")
    n = len(days)
    close = [x["c"] for x in rows] + [None] * (n - len(rows))
    ar = _ref_ar(close, spy_ret, last, expect_r, n)
    for w, (lo, hi) in er.WINDOWS.items():
        assert r["car"][w] == pytest.approx(_ref_car(ar, close, i0, lo, hi), abs=1e-12), w
    # controls: drawn only from anchors with real bars on day -1 and day 0 (never after the last bar)
    anchored = lambda j: 1 <= j <= last and _ref_car(ar, close, j, 0, 20) is not None
    pool = [j for j in er.control_pool(i0, n, [i0]) if anchored(j)]
    ctrl = er.event_rng("QZD", i0, E3).sample(pool, min(er.N_CONTROLS, len(pool)))
    assert r["n_ctrl"] == len(ctrl) == er.N_CONTROLS and max(ctrl) <= last
    for w, (lo, hi) in er.WINDOWS.items():
        cv = [x for x in (_ref_car(ar, close, j, lo, hi) for j in ctrl) if x is not None]
        assert r["excess"][w] == pytest.approx(r["car"][w] - sum(cv) / len(cv), abs=1e-12), w
    # the survivor (no-delisting) path cannot see this event: its window runs past the last bar
    yahoo, _ = er.measure_ticker("QZD", days, spy_ret, spy_bull, _bars(rows), {E3: [i0]})
    assert yahoo == []


def test_delisted_event_included_with_minus_30pct(tmp_path):
    _check_delisted(tmp_path, "inactive", None, ["inactive_no_event_date"], erp.DELIST_RETURN_OTHER)


def test_delisted_event_at_dated_merger_earns_zero(tmp_path):
    _check_delisted(tmp_path, "inactive", lambda rows: {"kind": "merger", "date": rows[-1]["d"]}, [],
                    erp.DELIST_RETURN_MERGER)


def test_event_filed_after_the_last_trade_is_counted_not_measured(tmp_path):
    rng = random.Random(9)
    last = 400
    rows = ws.path_rows(DAYS[:last + 1], 30.0, 0.0, 0.02, 2e6, rng)
    build, days, spy_ret, spy_bull, _ = _panel(tmp_path, [_spy_line(rng), ws.line("QZE", rows, status="inactive", flags=["inactive_no_event_date"])])
    recs, _y, diag, _u = erp.measure_panel(build["dir"], days, spy_ret, spy_bull, DAYS[-1],
                                           {"QZE": _ev([(E4, last + 1), (E2, 300)])}, {"QZE"}, 4000)
    assert [r["type"] for r in recs] == [E2]
    assert diag[("no_bar_on_day0", E4)] == 1


def test_series_ending_inside_the_as_of_grace_is_not_delisted(tmp_path):
    # last bar 3 days before as_of: still trading (not terminated) -> no delisting tail, window unmeasurable
    rng = random.Random(4)
    rows = ws.path_rows(DAYS[:-3], 30.0, 0.0, 0.02, 2e6, rng)
    build, days, spy_ret, spy_bull, _ = _panel(tmp_path, [_spy_line(rng), ws.line("QZG", rows)])
    recs, *_ = erp.measure_panel(build["dir"], days, spy_ret, spy_bull, DAYS[-1],
                                 {"QZG": _ev([(E2, len(days) - 15)])}, {"QZG"}, 4000)
    assert recs == []


def test_floor_reads_the_raw_close(tmp_path):
    rng = random.Random(8)
    hi_adj = ws.path_rows(DAYS, 2.0, 0.0, 0.004, 2e6, rng, raw_ratio=0.3)     # adjusted ~$2, raw ~$0.6
    lo_adj = ws.path_rows(DAYS, 0.6, 0.0, 0.004, 2e6, rng, raw_ratio=3.0)     # adjusted ~$0.6, raw ~$1.8
    build, days, spy_ret, spy_bull, _ = _panel(tmp_path, [_spy_line(rng), ws.line("QZH", hi_adj), ws.line("QZL", lo_adj)])
    ev = {"QZH": _ev([(E2, 400)]), "QZL": _ev([(E2, 400)])}
    recs, _y, diag, _u = erp.measure_panel(build["dir"], days, spy_ret, spy_bull, DAYS[-1], ev, {"QZH", "QZL"}, 4000)
    assert len(recs) == 1 and diag[("floor_raw_excludes", E2)] == 1 and diag[("floor_raw_admits", E2)] == 1
    # the adjusted-close (Yahoo-shaped) rule decides the opposite way on both names
    assert len(er.measure_ticker("QZH", days, spy_ret, spy_bull, _bars(hi_adj), {E2: [400]})[0]) == 1
    assert er.measure_ticker("QZL", days, spy_ret, spy_bull, _bars(lo_adj), {E2: [400]})[0] == []


# ── identity: survivors, delisted CIKs, reused tickers ──────────────────────
def test_map_events_identity_rules():
    days = ws.bdays("2017-01-02", "2017-12-29")
    listed = {"0000000001": "BRK-B"}
    pm = csm.build_map([("BRK.B", "1", "2017-02-01"), ("BRK.B", "1", "2017-11-01"),
                        ("OLDCO", "2", "2016-01-05"), ("OLDCO", "2", "2017-03-01"),
                        ("AAA", "3", "2017-01-03"), ("AAA", "3", "2017-12-01"), ("BBB", "3", "2017-01-03"), ("BBB", "3", "2017-12-01")])
    raw = [("0000000001", E2, "2017-06-01T14:00:00.000Z", "2017-06-01"),
           ("0000000002", E2, None, "2017-06-01"),                      # padded (after its last filing)
           ("0000000003", E2, None, "2017-06-01"),                      # two tickers in force -> ambiguous
           ("0000000004", E2, None, "2017-06-01"),                      # never filed -> unmapped
           ("0000000001", E2, None, "2017-12-29")]                      # no trading day after it
    dropped = Counter()
    by = erp.map_events(raw, days, listed, pm, dropped)
    assert set(by) == {"BRK.B", "OLDCO"}
    (et, i0, cik, origin, confirms, span), = by["BRK.B"]
    assert days[i0] == "2017-06-01" and origin == "survivor" and confirms
    assert span == (pp.day_num("2017-02-01"), pp.day_num("2017-11-01"))
    (_et, _i0, _c, origin2, confirms2, span2), = by["OLDCO"]
    assert origin2 == "delisted" and confirms2 and span2 == (pp.day_num("2016-01-05"), pp.day_num("2017-03-01"))
    assert dropped[("identity_ambiguous", E2)] == 1 and dropped[("identity_none", E2)] == 1
    assert dropped[("beyond_panel_calendar", E2)] == 1


def test_delisted_candidates_need_no_current_listing_and_a_panel_overlap():
    pm = csm.build_map([("OLDCO", "2", "2016-03-01"), ("OLDCO", "2", "2017-01-01"),
                        ("LIVE", "5", "2017-01-01"), ("NOPE", "6", "2017-01-01"), ("LATE", "7", "2014-01-02")])
    idx = {"OLDCO": [(pp.day_num("2016-01-04"), pp.day_num("2017-06-30"), "default")],
           "LIVE": [(pp.day_num("2016-01-04"), pp.day_num("2018-12-31"), "default")],
           "LATE": [(pp.day_num("2018-01-02"), pp.day_num("2018-12-31"), "default")]}
    out = erp.delisted_candidates(pm, {"0000000005"}, idx)
    assert out == {"0000000002": ["OLDCO"]}


def test_reused_ticker_guards(tmp_path):
    rng = random.Random(12)
    cut = 380
    rows = ws.path_rows(DAYS, 20.0, 0.0, 0.02, 2e6, rng)
    raw_line = ws.line("QZR", rows, mode="raw", status="active", event={"kind": "name_change", "date": DAYS[cut]},
                       flags=["reuse_suspect"])
    build, days, spy_ret, spy_bull, _ = _panel(tmp_path, [_spy_line(rng), raw_line])
    span = lambda a, b: (pp.day_num(DAYS[a]), pp.day_num(DAYS[b]))
    ev = {"QZR": [
        (E2, 200, "0000000001", "survivor", False, None),             # old company's part: dated ticker differs
        (E2, 600, "0000000001", "survivor", True, span(390, 700)),    # new company's part, confirmed
        (E4, 250, "0000000001", "survivor", True, span(400, 740)),    # confirmed only by padding BEFORE its first
                                                                      # filing, while the old company held the ticker
        (E3, 210, "0000000009", "delisted", True, span(100, 300)),    # old company = this delisted CIK
        (E3, 650, "0000000008", "delisted", True, span(100, 300)),    # ticker re-assigned after its last filing
        (E2, 300, "0000000007", "delisted", True, span(500, 700)),    # padded before its first filing: the old
                                                                      # company's series must not be used
    ]}
    recs, _y, diag, _u = erp.measure_panel(build["dir"], days, spy_ret, spy_bull, DAYS[-1], ev, {"QZR"}, 4000)
    got = sorted((r["type"], r["day0"], r["origin"]) for r in recs)
    assert got == sorted([(E2, DAYS[600], "survivor"), (E3, DAYS[210], "delisted")])
    assert diag[("reuse_unconfirmed", E2)] == 1 and diag[("reuse_unconfirmed", E4)] == 1
    assert diag[("symbol_reassigned", E3)] == 1 and diag[("symbol_reassigned", E2)] == 1


# ── SEC stages are the core's rules ─────────────────────────────────────────
def test_split_listing_is_the_core_universe_rule(tmp_path):
    d = {"fields": ["cik", "name", "ticker", "exchange"], "data": [
        [320193, "A", "AAPL", "Nasdaq"], [320193, "A", "AAPL2", "NYSE"], [1, "O", "OTCX", "OTC"], [2, "T", "TXS", "TXSE"],
        [3, "N", "NOEX", None], [4, "C", "CBX", "CBOE"], [5, "E", "", "NYSE"], [6, "B", "BF-B", "NYSE"]]}

    class C:
        def get_json(self, url):
            assert url == er.TICKERS_EXCHANGE_URL
            return d
    out = erp.split_listing(d)
    assert out["listed"] == er.stage_universe(str(tmp_path), C())
    assert set(out["on_exchange"]) == {f"{c:010d}" for c in (320193, 2, 4, 5, 6)}
    assert erp.panel_symbol(out["listed"][f"{6:010d}"]) == "BF.B"


def _tsv(rows):
    return ("\t".join(rows[0]) + "\n" + "".join("\t".join(r) + "\n" for r in rows[1:])).encode()


def _zip_bytes():
    sub = [["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERCIK", "ISSUERTRADINGSYMBOL"],
           ["a1", "29-JAN-2016", "4", "100", "QZA"], ["a2", "29-JAN-2016", "4/A", "100", "QZA"],
           ["a3", "12-FEB-2016", "4", "200", "QZB"], ["a4", "31-FEB-2016", "4", "300", "QZC"],
           ["a5", "01-MAR-2016", "3", "100", "QZA"], ["a6", "02-MAR-2016", "4", "900", "QZZ"]]
    own = [["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP"],
           ["a1", "o1", "Director"], ["a1", "o2", "TenPercentOwner"], ["a1", "o4", "Director,Officer"],
           ["a2", "o1", "Director"], ["a3", "o3", "Officer"], ["a4", "o5", "Officer"], ["a6", "o6", "Director"]]
    tr = [["ACCESSION_NUMBER", "TRANS_CODE", "TRANS_ACQUIRED_DISP_CD", "TRANS_DATE"],
          ["a1", "P", "A", "27-JAN-2016"], ["a1", "P", "A", "27-JAN-2016"], ["a1", "S", "D", "27-JAN-2016"],
          ["a1", "P", "D", "27-JAN-2016"], ["a1", "P", "A", "bad"], ["a2", "P", "A", "27-JAN-2016"],
          ["a3", "P", "A", "10-FEB-2016"], ["a4", "P", "A", "10-FEB-2016"], ["a6", "P", "A", "01-MAR-2016"]]
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as zf:
        zf.writestr("SUBMISSION.tsv", _tsv(sub))
        zf.writestr("REPORTINGOWNER.tsv", _tsv(own))
        zf.writestr("NONDERIV_TRANS.tsv", _tsv(tr))
    return b.getvalue()


class _OneQuarterClient:
    """Serves one quarter's zip; every other URL is 'not published' (404)."""

    def __init__(self, key, blob):
        self.key, self.blob, self.urls = key, blob, []

    def get(self, url):
        import sec_filings_text as sft
        self.urls.append(url)
        if f"/{self.key}_form345.zip" in url:
            return self.blob
        raise sft.EdgarHTTPError(url, 404)


def test_form345_stage_matches_core_purchases_and_feeds_the_pairs_map(tmp_path):
    blob = _zip_bytes()
    core_cache = tmp_path / "core"
    (core_cache / "form345").mkdir(parents=True)
    (core_cache / "form345" / "2016q1.zip").write_bytes(blob)
    uni = {f"{100:010d}", f"{200:010d}"}
    core = er.stage_purchases(str(core_cache), _OneQuarterClient("2016q1", blob), {c: "X" for c in uni})
    cl = _OneQuarterClient("2016q1", blob)
    f = erp.stage_form345(str(tmp_path / "panel"), cl)
    assert "2016q1" in f["pairs"] and "2016q1" in f["purchases"] and "2016q2" in f["missing"]
    mine = erp.load_purchases(str(tmp_path / "panel"), f["purchases"], uni)
    k = lambda p: (p["issuer"], p["owner"], p["accession"], p["trans_date"])
    assert sorted(mine, key=k) == sorted(core, key=k) and len(mine) == 3
    pm = erp.load_pairs_map(str(tmp_path / "panel"), f["pairs"])
    assert csm.resolve(pm, "100", "2016-02-15") == ("QZA", "exact")
    # cached: a second pass downloads nothing for the cached quarter
    cl2 = _OneQuarterClient("2016q1", b"")
    erp.stage_form345(str(tmp_path / "panel"), cl2)
    assert not any("2016q1" in u for u in cl2.urls)


def test_purchase_quarters_start_where_the_core_starts():
    assert erp.PURCHASES_FROM == er._quarters()[0]
    assert set(er.EXCHANGES) <= set(erp.CURRENT_EXCHANGES)


# ── verdicts on the panel (PD6) ─────────────────────────────────────────────
def _blk(cells, t01):
    tbl = {"pooled": {"0_1": {"t": t01}}}
    for b, v in zip(("small", "mid", "large"), cells):
        tbl[b] = {"verdict": v}
    return {"n": 10, "t_bar": er.T_BAR, "table": tbl, "verdict": er.type_verdict(cells, t01)}


def test_rule4_on_the_panel():
    s = {"P": _blk(["positive_unconfirmed", "fail", "fail"], 1.0), "N": _blk(["candidate", "positive_unconfirmed", "fail"], 1.0),
         "F": _blk(["fail"] * 3, 5.0), "Z": _blk(["fail"] * 3, 1.0)}
    out = erp.panel_verdicts(s)
    assert out["P"]["verdict"] == erp.PANEL_PROVISIONAL and out["P"]["verdict_core"] == "POSITIVE ON SURVIVORS — UNCONFIRMED"
    assert out["P"]["table"]["small"] == {"verdict": "candidate_provisional", "verdict_core": "positive_unconfirmed"}
    assert out["N"]["verdict"] == "CANDIDATE SIGNAL (gate 2)"
    assert out["F"]["verdict"] == "PRICED IN FAST / NO DRIFT" and out["Z"]["verdict"] == "NO MEASURABLE REACTION"


# ── end to end, offline ─────────────────────────────────────────────────────
class _NoNetwork:
    def get(self, url):
        raise AssertionError(f"network call {url}")

    def get_json(self, url):
        raise AssertionError(f"network call {url}")


def _offline_world(tmp_path, n_fill=34):
    rng = random.Random(11)
    lines = [_spy_line(rng), ws.line("QZXA", ws.path_rows(DAYS, 40.0, 0.0, 0.02, 3e6, rng))]
    for k in range(n_fill):
        lines.append(ws.line(f"QZF{k:02d}", ws.path_rows(DAYS, rng.uniform(5, 80), 0.0, 0.02, rng.uniform(2e5, 9e6), rng)))
    b_rows = ws.path_rows(DAYS[:501], 25.0, 0.0, 0.02, 2e6, rng)
    c_rows = ws.path_rows(DAYS[:451], 60.0, 0.0, 0.02, 4e6, rng)
    lines.append(ws.line("QZXB", b_rows, status="inactive", flags=["inactive_no_event_date"]))
    lines.append(ws.line("QZXC", c_rows, status="inactive", event={"kind": "merger", "date": c_rows[-1]["d"]}))
    root = str(tmp_path / "pp")
    ws.write_panel(root, lines, as_of=DAYS[-1])
    cache = erp.prepare_cache(str(tmp_path / "cache"), "b20261029")
    c1, c2, c3 = (f"{i:010d}" for i in (1, 2, 3))
    fill = {f"{100 + k:010d}": f"QZF{k:02d}" for k in range(n_fill)}
    listed = {c1: "QZXA", **fill}
    er._jdump(os.path.join(cache, "listing.json.gz"), {"listed": listed, "on_exchange": sorted(listed)})
    os.makedirs(os.path.join(cache, "form345"), exist_ok=True)
    purchases = {"2017q2": [[c1, "o1", "2017-05-01", "2017-05-02", "f1"], [c1, "o2", "2017-05-04", "2017-05-05", "f2"]],
                 "2016q4": [[c2, "o7", "2016-11-01", "2016-11-02", "f3"], [c2, "o8", "2016-11-03", "2016-11-04", "f4"]]}
    for (y, q) in er._quarters(start=erp.PAIRS_FROM):
        key = f"{y}q{q}"
        pairs = []
        if "2016q1" <= key <= "2018q4":
            mid = f"{y}-{3 * q - 1:02d}-15"
            pairs = [["QZXA", "1", mid]]
            if key <= "2017q4":
                pairs.append(["QZXB", "2", mid])
            if key <= "2017q3":
                pairs.append(["QZXC", "3", mid])
        pa, pu = erp._f345_paths(cache, key)
        er._jdump(pa, pairs)
        if (y, q) >= erp.PURCHASES_FROM:
            er._jdump(pu, purchases.get(key, []))
    at = lambda i, hh="15": f"{DAYS[i]}T{hh}:00:00.000Z"          # 10:00 ET in winter, 11:00 ET in summer
    k8 = lambda acc, i, items, hh="15": {"acc": acc, "fd": DAYS[i], "acc_utc": at(i, hh), "items": items, "doc": "d.htm"}
    subs = {
        c1: {"s": {"has_10kq": True, "f4": {"f2": at(DAYS.index("2017-05-05"))},
                   "eightk": [k8("e1", 300, ["1.01"]), k8("e2", 420, ["1.01"]), k8("e3", 560, ["5.02"])]}, "f4": ["f2"]},
        c2: {"s": {"has_10kq": True, "f4": {},
                   "eightk": [k8("e4", 494, ["4.02"]), k8("e5", 500, ["1.01"], hh="22")]}, "f4": ["f4"]},
        c3: {"s": {"has_10kq": True, "f4": {}, "eightk": [k8("e6", 435, ["5.02"]), k8("e7", 380, ["1.01"])]}, "f4": []},
    }
    for c in fill:
        subs[c] = {"s": {"has_10kq": True, "f4": {}, "eightk": []}, "f4": []}
    er._jdump(os.path.join(cache, "submissions.json.gz"), subs)
    er._jdump(os.path.join(cache, "e4_class.json.gz"), {"e3": True, "e6": True})
    return root, str(tmp_path / "cache")


def test_end_to_end_offline_writes_aggregates_only(tmp_path):
    root, cache = _offline_world(tmp_path)
    out = str(tmp_path / "out")
    res = erp.run(root, out, cache, 4000, "1.0.test", client=_NoNetwork())
    assert res["status"] == "ok", res
    latest = json.load(open(os.path.join(out, "latest.json")))
    text = open(os.path.join(out, "latest.json")).read() + open(os.path.join(out, "last_run.json")).read()
    assert "QZ" not in text, "a ticker leaked into the result files"
    assert len(text) < 250_000
    assert latest["schema"] == erp.SCHEMA and latest["prior"] == erp.PRIOR and latest["deviations"] == erp.DEVIATIONS
    assert latest["study_code"] == erp.study_code_hash() and latest["code_version"] == "1.0.test"
    assert set(latest["verdicts"]) == set(er.EVENT_TYPES) and set(latest["survivor_subset"]["verdicts"]) == set(er.EVENT_TYPES)
    m = latest["meta"]
    assert m["delisted_candidate_ciks"] == 2 and m["universe_ciks"] == 37
    assert m["measured_by_origin"]["survivor"].get(E2) == 2 and m["measured_by_origin"]["delisted"].get(E3) == 1
    assert m["measured_by_origin"]["delisted"].get(E4) == 1 and m["measured_by_origin"]["delisted"].get(E2) == 1
    assert m["stopped_trading_at_event"] == {"no_bar_on_day0": {E2: 1}}
    assert m["delisting_return_in_window"][E3] == {"other_0_20": 1}
    assert m["delisting_return_in_window"][E4] == {"merger_0_20": 1}
    assert m["e1_clusters_all"] == 2 and m["raw_events_by_type"][er.EVENT_TYPES[0]] == 2
    assert m["sec_cache_bytes"] > 0
    assert latest["summary"][E2]["table"]["pooled"]["2_20"]["n"] == 3
    # the keys server/eventStudy.ts cells() reads (its fixture mirrors these)
    for t in er.EVENT_TYPES:
        blk = latest["summary"][t]
        assert {"n", "t_bar", "verdict", "verdict_core", "table"} <= set(blk)
        assert {"n", "t_bar", "verdict", "table"} <= set(latest["survivor_subset"]["summary"][t])
        for g in ("pooled", "small", "mid", "large"):
            row = blk["table"][g]
            assert {"0_1", "2_20", "2_60", "halves", "spread_median", "late_share"} <= set(row)
            assert {"2016_2020", "2021_on"} <= set(row["halves"])
            if g != "pooled":
                assert {"verdict", "verdict_core"} <= set(row)
    last = json.load(open(os.path.join(out, "last_run.json")))
    assert last["status"] == "ok" and last["verdicts"] == latest["verdicts"]


def test_e4_uses_only_this_runs_seeded_selection(tmp_path, monkeypatch):
    # both 5.02 filings carry a cached True label, but a 1-filing cap selects one of them
    monkeypatch.setattr(er, "MAX_502_DOCS", 1)
    root, cache = _offline_world(tmp_path)
    res = erp.run(root, str(tmp_path / "out"), cache, 4000, "x", client=_NoNetwork())
    m = json.load(open(tmp_path / "out" / "latest.json"))["meta"]
    assert res["status"] == "ok" and m["e4_docs_selected"] == 1 and m["e4_positive"] == 1
    assert m["raw_events_by_type"][E4] == 1


def test_sec_fetch_errors_make_the_run_insufficient_not_thin(tmp_path):
    root, cache = _offline_world(tmp_path)
    d = os.path.join(cache, "b20261029-c1", "submissions.json.gz")
    subs = er._jload(d)
    for c in (f"{100:010d}", f"{101:010d}"):        # 2 of 37 CIKs uncached; the client fails them
        subs.pop(c)
    er._jdump(d, subs)
    out = str(tmp_path / "out")
    res = erp.run(root, out, cache, 4000, "x", client=_NoNetwork())
    assert res["status"] == "insufficient" and "submissions 2/37" in res["reason"]
    assert not os.path.exists(os.path.join(out, "latest.json"))


def test_errored_e4_documents_are_retried(tmp_path):
    cache = str(tmp_path)
    er._jdump(os.path.join(cache, "e4_class.json.gz"), {"a1": True, "a2": "error: TimeoutError", "a3": False})
    assert erp.clear_e4_errors(cache) == 1
    assert er._jload(os.path.join(cache, "e4_class.json.gz")) == {"a1": True, "a3": False}
    assert erp.clear_e4_errors(cache) == 0 and erp.clear_e4_errors(str(tmp_path / "none")) == 0


def test_cache_is_per_build_and_bounded(tmp_path):
    base = str(tmp_path / "c")
    old = erp.prepare_cache(base, "b20260901")
    open(os.path.join(old, "x"), "w").write("1")
    os.makedirs(os.path.join(base, "unrelated"))
    new = erp.prepare_cache(base, "b20261029")
    assert not os.path.exists(old) and os.path.isdir(new) and os.path.isdir(os.path.join(base, "unrelated"))


def test_no_panel_is_skipped_and_memory_budget_aborts_cleanly(tmp_path, capsys):
    out = str(tmp_path / "out")
    assert erp.main(["--run", "--panel-root", str(tmp_path / "none"), "--out-dir", out, "--cache-dir", str(tmp_path / "c")]) == 0
    assert json.loads(capsys.readouterr().out.strip())["status"] == "skipped"
    root, cache = _offline_world(tmp_path)
    rc = erp.main(["--run", "--panel-root", root, "--out-dir", out, "--cache-dir", cache, "--max-rss-mb", "1"])
    res = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rc == 1 and res["status"] == "error" and "memory budget" in res["reason"]
    assert not os.path.exists(os.path.join(out, "latest.json"))
    assert json.load(open(os.path.join(out, "last_run.json")))["status"] == "error"


def test_study_code_hash_is_pinned_for_the_ts_side(tmp_path):
    # server/eventStudy.test.ts pins the same digest for the same two files
    (tmp_path / "event_reaction.py").write_bytes(b"alpha\n")
    (tmp_path / "event_reaction_panel.py").write_bytes(b"beta\n")
    assert erp.study_code_hash(str(tmp_path)) == "686200ca135543a4"
    os.remove(tmp_path / "event_reaction_panel.py")
    assert erp.study_code_hash(str(tmp_path)) == "missing:event_reaction_panel.py"
    assert erp.STUDY_FILES == ("event_reaction.py", "event_reaction_panel.py")
    assert all(os.path.exists(os.path.join(os.path.dirname(os.path.abspath(erp.__file__)), f)) for f in erp.STUDY_FILES)
