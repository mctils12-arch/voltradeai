"""
test_cusip_figi.py: the CUSIP -> security type record (cusip_figi.py). No
network: OpenFIGI is a stand-in that answers in the shape read from the real
service on 2026-11-26.
"""
import json

import cusip_figi as cf

AAPL = {"figi": "BBG000B9XRY4", "name": "APPLE INC", "ticker": "AAPL", "exchCode": "US",
        "compositeFIGI": "BBG000B9XRY4", "securityType": "Common Stock", "marketSector": "Equity",
        "shareClassFIGI": "BBG001S5N8V8", "securityType2": "Common Stock"}
SPY = dict(AAPL, figi="BBG000BDTBL9", name="SPDR S&P 500 ETF TRUST", ticker="SPY", securityType="ETP",
           securityType2="Mutual Fund")
KNOWN = {"037833100": [dict(AAPL, exchCode="UA"), AAPL], "78462F103": [SPY]}


def openfigi(calls=None, fail=(), status_for=None, headers=None):
    """A stand-in for cusip_figi.http_post. `fail`: request numbers (from 1)
    that get `status_for` (default 500)."""
    n = {"i": 0}

    def post(url, body, hdrs):
        n["i"] += 1
        jobs = json.loads(body)
        if calls is not None:
            calls.append([j["idValue"] for j in jobs])
        assert url == cf.OPENFIGI_URL and all(j["idType"] == "ID_CUSIP" for j in jobs)
        assert set(hdrs) <= {"Content-Type", "User-Agent", "X-OPENFIGI-APIKEY"}
        if n["i"] in fail:
            return (status_for or 500), b"upstream said no", (headers or {})
        out = [{"data": KNOWN[j["idValue"]]} if j["idValue"] in KNOWN else {"warning": "No identifier found."}
               for j in jobs]
        return 200, json.dumps(out).encode(), {}
    return post


def resolve(tmp_path, cusips, **kw):
    kw.setdefault("post", openfigi())
    kw.setdefault("sleep", lambda s: None)
    kw.setdefault("now_iso", lambda: "2026-11-26T03:00:00+00:00")
    return cf.resolve(cusips, str(tmp_path / "figi" / cf.CACHE_FILE), **kw)


def cusips(n, start=0):
    return [f"{i:09d}" for i in range(start, start + n)]


def test_the_us_composite_listing_is_kept_with_its_type_and_a_cusip_is_asked_once(tmp_path):
    calls = []
    out = resolve(tmp_path, ["037833100", "78462F103", "000000000", "037833100"], post=openfigi(calls))
    assert calls == [["037833100", "78462F103", "000000000"]]            # the repeat is not asked twice
    assert (out["wanted"], out["asked"], out["found"], out["not_found"], out["owed"]) == (3, 3, 2, 1, 0)
    assert out["status"] == "ok" and out["stopped"] is None
    cache = cf.read_cache(str(tmp_path / "figi" / cf.CACHE_FILE))
    a = cache["037833100"]
    assert (a["ticker"], a["exch_code"], a["us"], a["listings"]) == ("AAPL", "US", True, 2)   # not the first venue
    assert (a["security_type"], a["security_type2"], a["market_sector"]) == ("Common Stock", "Common Stock", "Equity")
    assert a["composite_figi"] == "BBG000B9XRY4" and a["share_class_figi"] == "BBG001S5N8V8"
    assert cache["000000000"] == {"cusip": "000000000", "asked_at": "2026-11-26T03:00:00+00:00",
                                  "found": False, "reason": "No identifier found."}
    # a second run asks nothing: found and not found are both answers
    calls2 = []
    again = resolve(tmp_path, ["037833100", "78462F103", "000000000"], post=openfigi(calls2))
    assert calls2 == [] and again["requests"] == 0 and again["held"] == 3 and again["status"] == "ok"


def test_ten_cusips_a_request_without_a_key_a_hundred_with_one_and_the_key_goes_in_its_header(tmp_path):
    calls, waits = [], []
    resolve(tmp_path, cusips(25), post=openfigi(calls), sleep=waits.append)
    assert [len(c) for c in calls] == [10, 10, 5]
    assert waits == [cf.KEYLESS_WAIT_S, cf.KEYLESS_WAIT_S]                 # none after the last request
    assert 60.0 / cf.KEYLESS_WAIT_S < 25                                   # under 25 requests a minute
    seen = {}

    def keyed(url, body, hdrs):
        seen.update(hdrs)
        return 200, json.dumps([{"warning": "No identifier found."}] * len(json.loads(body))).encode(), {}
    out = resolve(tmp_path, cusips(150, start=1000), post=keyed, api_key="k")
    assert seen["X-OPENFIGI-APIKEY"] == "k" and out["requests"] == 2 and out["keyed"] is True


def test_the_time_budget_stops_between_requests_and_the_next_run_resumes(tmp_path):
    t = {"now": 0.0}

    def clock():
        t["now"] += 4.0
        return t["now"]
    out = resolve(tmp_path, cusips(50), budget_s=10.0, clock=clock)
    assert out["stopped"] == "time budget" and 0 < out["asked"] < 50 and out["owed"] == 50 - out["asked"]
    rest = resolve(tmp_path, cusips(50))
    assert rest["asked"] == out["owed"] and rest["owed"] == 0


def test_a_failed_request_is_asked_again_and_three_in_a_row_stop_the_run(tmp_path):
    calls = []
    out = resolve(tmp_path, cusips(20), post=openfigi(calls, fail={1}))
    assert calls[0] == calls[1] and out["failed_requests"] == 1 and out["asked"] == 20 and out["status"] == "ok"
    assert out["last_error"].startswith("HTTP 500")
    dead = resolve(tmp_path, cusips(20, start=500), post=openfigi(fail={1, 2, 3, 4}))
    assert dead["stopped"] == "3 requests in a row got no usable answer"
    assert dead["asked"] == 0 and dead["owed"] == 20 and dead["status"] == "error"   # wanted answers, got none


def test_a_rate_limit_waits_what_the_header_says_or_leaves_it_to_the_next_run(tmp_path):
    waits = []
    out = resolve(tmp_path, cusips(10), post=openfigi(fail={1}, status_for=429, headers={"ratelimit-reset": "7"}),
                  sleep=waits.append)
    assert waits == [7.0] and out["asked"] == 10
    out = resolve(tmp_path, cusips(10, start=100), budget_s=30.0,
                  post=openfigi(fail={1}, status_for=429, headers={"ratelimit-reset": "60"}), sleep=waits.append)
    assert out["stopped"] == "rate limited by OpenFIGI" and out["asked"] == 0 and waits == [7.0]


def test_a_job_level_error_is_not_an_answer_about_the_cusip_unless_it_says_the_value_is_no_cusip(tmp_path):
    def post(url, body, hdrs):
        jobs = json.loads(body)
        return 200, json.dumps([{"error": "Internal error."}, {"error": "Invalid idValue format."}]
                               + [{"data": [AAPL]}] * (len(jobs) - 2)).encode(), {}
    out = resolve(tmp_path, ["46090E103", "SPXSRT5E0", "037833100"], post=post)
    assert out["job_errors"] == 1 and out["asked"] == 2 and out["owed"] == 1 and out["not_found"] == 1
    cache = cf.read_cache(str(tmp_path / "figi" / cf.CACHE_FILE))
    assert "46090E103" not in cache                       # asked again next run
    # a nine-character index code (real, 2026-11-26): kept as not found, never asked again
    assert cache["SPXSRT5E0"]["found"] is False and cache["SPXSRT5E0"]["reason"] == "Invalid idValue format."


def test_something_that_is_not_a_cusip_is_counted_and_never_sent(tmp_path):
    calls = []
    out = resolve(tmp_path, ["037833100", "US0378331005", "", "0378 3100"], post=openfigi(calls))
    assert calls == [["037833100"]] and out["malformed"] == 3 and out["wanted"] == 1


def test_a_line_cut_by_a_kill_is_skipped_and_a_later_line_wins(tmp_path):
    p = tmp_path / cf.CACHE_FILE
    p.write_text('{"cusip": "037833100", "found": false, "reason": "old"}\n'
                 '{"cusip": "037833100", "found": true, "us": true, "security_type": "Common Stock"}\n'
                 '{"cusip": "78462F103", "fou')
    cache = cf.read_cache(str(p))
    assert list(cache) == ["037833100"] and cache["037833100"]["found"] is True
    assert cf.read_cache(str(tmp_path / "absent.jsonl")) == {}


def test_the_tradable_class_is_exactly_three_types_on_a_us_equity_listing():
    e = lambda **kw: dict({"found": True, "us": True, "market_sector": "Equity", "security_type": "Common Stock"}, **kw)  # noqa: E731
    assert cf.tradable_class(e()) == ("common", "common")
    assert cf.tradable_class(e(security_type="ADR")) == ("adr", "adr")
    assert cf.tradable_class(e(security_type="REIT")) == ("reit", "reit")
    for st in ("ETP", "Closed-End Fund", "Preference", "Unit", "Warrant", "Right", "MLP", "Ltd Part",
               "NY Reg Shrs", "Tracking Stk", "Open-End Fund", "common stock"):
        assert cf.tradable_class(e(security_type=st)) == (None, st)
    # no US composite today (bought or renamed since) is still its class:
    # dropping it would filter on what happened after the event
    assert cf.tradable_class(e(us=False)) == ("common", "common")
    assert cf.tradable_class(e(us=False, security_type="ETP")) == (None, "ETP")
    assert cf.tradable_class(e(market_sector="Pfd", security_type="PUBLIC")) == (None, "PUBLIC (Pfd)")
    assert cf.tradable_class({"found": False, "reason": "No identifier found."}) == (None, "unresolved")
    assert cf.tradable_class(None) == (None, "not asked")
    assert cf.TRADABLE_TYPES == {"Common Stock": "common", "ADR": "adr", "REIT": "reit"}
    cache = {"A": e(), "B": e(security_type="ETP"), "C": e(security_type="ETP", us=False), "D": {"found": False},
             "F": e(us=False), "G": e(us=False, security_type="REIT")}
    assert cf.class_counts(cache, ["A", "B", "C", "D", "E", "F", "G"]) == {
        "ETP": 2, "common": 2, "not asked": 1, "reit": 1, "unresolved": 1, cf.NO_US_KEY: 2}
    assert cf.NO_US_KEY not in cf.class_counts(cache, ["A", "B"])
