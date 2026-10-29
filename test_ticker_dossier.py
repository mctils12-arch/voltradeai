"""ticker_dossier: the layers' pure pieces and the orchestrator over injected I/O."""
import math
import random
from datetime import date, datetime, timedelta, timezone

import ticker_dossier as TD


def _bars(n=800, vol=0.03, p0=20.0, seed=1, end=date(2026, 10, 28)):
    rnd = random.Random(seed)
    days, d = [], end
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d -= timedelta(days=1)
    out, p = [], p0
    for day in reversed(days):
        p *= math.exp(rnd.gauss(0, vol))
        out.append({"t": f"{day.isoformat()}T04:00:00Z", "o": p, "h": p * 1.01, "l": p * 0.99, "c": p})
    return out


def test_credibility_tiers_are_static_and_documented():
    assert TD.credibility_tier("SEC") == 1
    assert TD.credibility_tier("Reuters") == 2 and TD.credibility_tier("GlobeNewswire") == 2
    assert TD.credibility_tier("benzinga") == 3
    assert TD.credibility_tier("some blog") == 4 and TD.credibility_tier(None) == 4
    assert set(TD.CREDIBILITY_TIERS) == {1, 2, 3, 4}


def test_sector_for_static_industry_then_spdr_then_unknown():
    s = TD.sector_for("RKLB", None)
    assert s["key"] == "space" and s["etfs"][0] == "UFO" and "static" in s["source"]
    assert TD.sector_for("JPM", "Financials")["etfs"] == ["XLF"]
    assert TD.sector_for("ZZZZ", "Unknown")["key"] == "unknown"
    assert TD.sector_for("SPY", None)["key"] == "index"


def test_trend_and_relative_strength():
    up = [float(x) for x in range(1, 260)]
    t = TD.trend_of(up)
    assert t["label"] == "up" and t["ret_20d_pct"] > 0
    assert TD.rel_strength(up, [100.0] * 259, 20) == TD.ret_pct(up, 20)
    assert TD.ret_pct([1.0], 5) is None


def test_parse_submissions_and_8k_items():
    sub = {"cik": "0001819994", "filings": {"recent": {
        "form": ["8-K", "4", "10-Q"], "filingDate": ["2026-10-20", "2026-10-21", "2026-08-01"],
        "acceptanceDateTime": ["2026-10-20T17:05:00.000Z", "2026-10-21T09:00:00.000Z", "2026-08-01T08:00:00.000Z"],
        "items": ["1.01,9.01", "", ""], "accessionNumber": ["0001-26-1", "0001-26-2", "0001-26-3"],
        "primaryDocument": ["a.htm", "b.xml", "c.htm"], "primaryDocDescription": ["8-K", "", ""]}}}
    fs = TD.parse_submissions(sub, "2026-09-01")
    assert [f["form"] for f in fs] == ["4", "8-K"]     # newest first, window applied
    ev = TD.filing_event(fs[1])
    assert ev["material"] and ev["tier"] == 1 and "material definitive agreement" in ev["title"]
    assert ev["url"].startswith("https://www.sec.gov/Archives/edgar/data/1819994/")
    assert TD.filing_event(fs[0])["material"] is False


def test_event_history_only_for_measured_types_and_never_a_flag():
    def ev(form, items):
        return {"kind": "filing", "form": form, "items": [{"item": i, "label": ""} for i in items]}
    h = TD.event_history_for(ev("8-K", ["4.02", "9.01"]))
    assert h["key"] == "8k_4.02" and h["n"] == 715 and h["car01_pct"] == -1.51 and "drift" in h
    assert h["study_path"] == "research/event_reaction_study.md" and "not a prediction" in h["study_caveat"]
    assert TD.event_history_for(ev("8-K", ["1.01"]))["n"] == 42956
    # 5.02 covers routine appointments too; the study's number applies only to
    # text-classified sudden exits, so no history line until that classifier exists
    assert TD.event_history_for(ev("8-K", ["5.02"])) is None
    # unmeasured: other items, amendments, lawsuits (8.01), news
    assert TD.event_history_for(ev("8-K", ["8.01", "9.01"])) is None
    assert TD.event_history_for(ev("8-K/A", ["4.02"])) is None
    assert TD.event_history_for({"kind": "news", "title": "x"}) is None
    # the table carries every number with its n and t; nothing claims drift or "priced in"
    for row in TD.EVENT_HISTORY.values():
        assert row["n"] > 0 and isinstance(row["t01"], float)
        assert "priced in" not in row["drift"]
    assert not hasattr(TD, "reaction_check")


def test_score_deciles_against_base_rate():
    rnd = random.Random(2)
    recs = []
    for k in range(400):
        s = rnd.uniform(40, 90)
        ret = (s - 65) / 10 + rnd.gauss(0, 1)
        recs.append({"score": s, "decision": "rejected" if s < 63 else "taken",
                     "outcomes": {"+5d": {"return_pct": ret, "label": int(ret > 0)}, "+20d": None}})
    recs.append({"score": 70, "outcomes": {"+5d": None}})   # unsettled: excluded
    d = TD.score_deciles(recs)
    assert d["n"] == 400 and len(d["buckets"]) == 10 and not d["thin"]
    assert sum(b["count"] for b in d["buckets"]) == 400
    assert d["buckets"][9]["d5"]["mean_pct"] > d["base"]["d5"]["mean_pct"] > d["buckets"][0]["d5"]["mean_pct"]
    assert d["buckets"][0]["d20"]["n"] == 0
    assert TD.score_deciles([])["buckets"] == []


def test_options_summary_implied_move_and_skew():
    cs = []
    for exp, dte, iv in (("2026-11-06", 8, 0.80), ("2026-11-28", 30, 0.70)):
        for k in (80, 90, 95, 100, 105, 110, 120):
            for t in ("call", "put"):
                skew = 0.10 if (t == "put" and k < 100) else 0.0
                cs.append({"exp_date": exp, "days_out": dte, "strike": k, "opt_type": t, "iv": iv + skew, "moneyness": k / 100})
    o = TD.options_summary(cs, 100.0, 0.08, 0.035)
    assert o["front"]["exp_date"] == "2026-11-06" and o["atm_iv"] == 80.0
    assert math.isclose(o["implied_move_5d_pct"], (math.exp(0.8 * math.sqrt(5 / 252)) - 1) * 100, abs_tol=0.01)
    assert o["skew_put_minus_call_pts"] == 10.0
    assert o["implied_vs_model_5d"] > 1
    assert TD.options_summary([], 100.0, None, None)["atm_iv"] is None


def test_direction_says_coin_flip_without_a_call():
    d = TD.direction_from_store("RKLB", [], [], {}, date(2026, 10, 29))
    assert d["status"] == "none" and "coin flip" in d["statement"]


def test_direction_shows_call_with_track_record():
    batches = [{"kind": "batch", "id": "b1", "week_key": "2026-W44", "made_at": "2026-10-24T22:00:00Z"}]
    fc = [{"kind": "forecast", "id": "f1", "week_key": "2026-W44", "symbol": "RKLB", "p_up": 0.58, "p_call": 0.58,
           "direction": "up", "base_date": "2026-10-23", "horizon_end": "2026-10-30"},
          {"kind": "forecast", "id": "f0", "week_key": "2026-W43", "symbol": "SPY", "p_up": 0.56, "direction": "up",
           "horizon_end": "2026-10-23"}]
    scores = {"f0": {"status": "resolved", "outcome_up": 1, "hit": 1, "in_range": 1}}
    d = TD.direction_from_store("RKLB", batches, fc, scores, date(2026, 10, 29))
    assert d["status"] == "call" and d["call"]["p_up"] == 0.58
    assert d["model_accuracy"]["n"] == 1 and d["model_accuracy"]["thin"] is True
    assert d["calibration_bin"]["lo"] == 0.55
    # after the horizon the call is not current
    assert TD.direction_from_store("RKLB", batches, fc, scores, date(2026, 11, 2))["status"] == "none"


def _io(recorded):
    bars = {"RKLB": _bars(seed=1), "SPY": _bars(vol=0.01, p0=500, seed=2), "UFO": _bars(vol=0.015, seed=3)}
    return {
        "bars": lambda syms, a, b: ({k: bars[k] for k in syms if k in bars}, set()),
        "regime": lambda: {"as_of": "2026-10-28", "composite_regime": "RISK_ON", "exposure_multiplier": 1.0, "inputs": {}},
        "sector": lambda s: None,
        "edgar": lambda s: None,
        "news": lambda s: [{"headline": "Contract win", "source": "Reuters", "created_at": "2026-10-27T13:00:00Z"}],
        "chain": lambda s, p: [],
        "earnings": lambda s: {"status": "unavailable", "reason": "no key"},
        "forecast_store": lambda: ([], [], {}),
        "shadow": lambda: None,
        "record": lambda rows: recorded.extend(rows) or len(rows),
        "ledger_report": lambda t: {"all": {}, "ticker": {}},
    }


def test_build_rejects_bad_symbols():
    assert "symbol" in TD.build("RK-LB; rm")["errors"]
    assert "symbol" in TD.build("")["errors"]


def test_build_end_to_end_over_injected_io():
    rec = []
    now = datetime(2026, 10, 29, 22, 0, tzinfo=timezone.utc)
    o = TD.build("rklb", now=now, io=_io(rec))
    assert o["symbol"] == "RKLB"
    assert o["market"]["status"] == "live" and o["market"]["composite"] == "RISK_ON"
    assert o["sector"]["status"] == "live" and o["sector"]["etf"] == "UFO"
    st = o["stock"]
    assert st["status"] == "live" and st["base_date"] == "2026-10-28"
    b1 = st["ranges"]["1"]["bands"]
    assert b1["lo90"] < b1["lo68"] < st["base_close"] < b1["hi68"] < b1["hi90"]
    assert st["ranges"]["5"]["backtest"]["n"] > 0
    assert o["options"]["status"] == "unavailable" and o["options"]["earnings"]["status"] == "unavailable"
    assert o["events"]["status"] == "partial" and o["events"]["items"][0]["tier"] == 2
    assert all("reaction" not in e for e in o["events"]["items"])
    assert "not yet priced in" in o["events"]["history_note"] and o["events"]["event_history"]["insider_cluster"]["n"] == 10090
    assert o["direction"]["status"] == "none"
    assert o["plan"]["stop"] < st["price"] < o["plan"]["target"]
    assert o["bot_guards"]["composite"] == "RISK_ON"
    kinds = sorted(r["kind"] for r in rec)
    assert kinds == ["plan", "range", "range"]
    assert {r["id"] for r in rec} == {"range|RKLB|2026-10-28|1", "range|RKLB|2026-10-28|5", "plan|RKLB|2026-10-29"}


def test_build_with_no_history_says_so():
    io = _io([])
    io["bars"] = lambda syms, a, b: ({}, set(syms))
    o = TD.build("ZZZZ", now=datetime(2026, 10, 29, 22, 0, tzinfo=timezone.utc), io=io)
    assert o["stock"]["status"] == "unavailable" and "no price history" in o["stock"]["reason"]
