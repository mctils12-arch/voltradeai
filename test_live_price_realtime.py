"""
Real-time display prices under the delayed feed (2026-11-12, post-open check of
the stale-ticker fix): this account's plan rejects recent SIP data, so
alpaca_feed.data_feed() is "delayed_sip" and every live price on the Analyze
panel, the Ticker Dossier and the ETF Builder was 15 minutes old (SPY's latest
trade read 09:53 ET at 10:08 ET). The plan carries IEX in real time;
live_price.freshest() takes its trade when it is a NEWER REGULAR-HOURS print,
the consolidated snapshot staying the base (its bar's volume is the whole
tape's). Pinned here: the time and hours helpers, every branch of freshest(),
and fetch_snapshots() reading both tapes only under delayed_sip.
"""
from datetime import datetime, timezone

import live_price as LP

DELAYED = {
    "latestTrade": {"p": 758.21, "t": "2026-11-12T14:54:10.929790855Z"},
    "dailyBar": {"t": "2026-11-12T05:00:00Z", "o": 753.98, "h": 758.89, "l": 753.67, "c": 758.38, "v": 5728453},
    "prevDailyBar": {"t": "2026-11-11T05:00:00Z", "c": 759.81},
}
IEX = {
    "latestTrade": {"p": 759.4, "t": "2026-11-12T15:09:02.123456789Z"},
    "dailyBar": {"t": "2026-11-12T05:00:00Z", "o": 754.0, "h": 759.5, "l": 753.7, "c": 759.4, "v": 81234},
    "prevDailyBar": {"t": "2026-11-11T05:00:00Z", "c": 759.8},
}


def _p(snap):
    return LP.parse_snapshot("SPY", snap)


def _with_trade(snap, **trade):
    return {**snap, "latestTrade": {**snap["latestTrade"], **trade}}


def test_trade_time_and_regular_hours():
    assert LP.trade_time("2026-11-12T15:09:02.123456789Z") == datetime(2026, 11, 12, 15, 9, 2, 123456, tzinfo=timezone.utc)
    assert LP.trade_time("2026-11-12T10:09:02-05:00") == datetime(2026, 11, 12, 15, 9, 2, tzinfo=timezone.utc)
    assert LP.trade_time(None) is None and LP.trade_time("") is None and LP.trade_time("2026-11-12") is None
    ri = LP.in_regular_hours
    assert ri(LP.trade_time("2026-11-12T14:30:00Z")) is True            # 09:30 ET
    assert ri(LP.trade_time("2026-11-12T14:29:59Z")) is False           # pre-market
    assert ri(LP.trade_time("2026-11-12T20:59:59Z")) is True            # 15:59:59 ET
    assert ri(LP.trade_time("2026-11-12T21:00:00Z")) is False           # 16:00 ET
    assert ri(LP.trade_time("2026-11-14T15:00:00Z")) is False           # Saturday
    assert ri(None) is False


def test_same_session_newer_iex_trade_is_the_price_and_the_volume_stays():
    out = LP.freshest(_p(DELAYED), _p(IEX))
    assert out["feed"] == "iex" and out["source"] == "latest_trade, iex real-time"
    assert out["price"] == 759.4 and out["asof"] == "2026-11-12T15:09:02.123456789Z"
    assert out["session_date"] == "2026-11-12"
    assert out["prev_close"] == 759.81, "the consolidated tape's previous close"
    assert out["bar"]["c"] == 759.4 and out["bar"]["h"] == 759.4 and out["bar"]["l"] == 753.67
    assert out["bar"]["v"] == 5728453.0, "never IEX's exchange-only volume"


def test_the_delayed_price_stands_unless_iex_has_a_newer_regular_hours_trade():
    for name, rt in [
        ("older", _with_trade(IEX, t="2026-11-12T14:50:00Z")),
        ("pre-market", _with_trade(IEX, t="2026-11-12T14:20:00Z")),
        ("after hours", _with_trade(IEX, t="2026-11-12T21:30:00Z")),
        ("no trade, bar only", {**IEX, "latestTrade": {}}),
    ]:
        out = LP.freshest(_p(DELAYED), _p(rt))
        assert out["feed"] == "delayed_sip", name
        assert out["price"] == 758.21, name
        assert out["source"] == "latest_trade, 15-min delayed", name
    assert LP.freshest(_p(DELAYED), None)["feed"] == "delayed_sip"
    assert LP.freshest(None, None) is None


def test_iex_on_a_newer_session_in_the_opens_first_minutes():
    delayed_yesterday = {
        "latestTrade": {"p": 759.81, "t": "2026-11-11T20:59:59Z"},
        "dailyBar": {"t": "2026-11-11T05:00:00Z", "o": 756.8, "h": 762.51, "l": 754.9, "c": 759.81, "v": 53905000},
        "prevDailyBar": {"t": "2026-11-10T05:00:00Z", "c": 757.0},
    }
    rt = _with_trade(IEX, p=754.1, t="2026-11-12T14:35:00Z")
    out = LP.freshest(_p(delayed_yesterday), _p(rt))
    assert out["feed"] == "iex" and out["session_date"] == "2026-11-12" and out["price"] == 754.1
    assert LP.is_newer_session(out, "2026-11-11") is True, "the callers overlay it on yesterday's bars"


def test_without_a_delayed_snapshot_only_a_regular_hours_iex_trade_serves():
    assert LP.freshest(None, _p(IEX))["feed"] == "iex"
    assert LP.freshest(None, _p(_with_trade(IEX, t="2026-11-12T21:30:00Z"))) is None


class _Resp:
    def __init__(self, body, status=200):
        self._b, self.status_code = body, status

    def json(self):
        return self._b


def test_fetch_reads_both_tapes_only_under_delayed_sip(monkeypatch):
    monkeypatch.setenv("ALPACA_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET", "s")
    feeds = []

    def get(url, params=None, headers=None, timeout=None):
        feeds.append(params["feed"])
        return _Resp({"SPY": DELAYED if params["feed"] == "delayed_sip" else IEX})

    out = LP.fetch_snapshots(["spy"], get=get, feed="delayed_sip")
    assert feeds == ["delayed_sip", "iex"]
    assert out["SPY"]["price"] == 759.4 and out["SPY"]["feed"] == "iex"
    feeds.clear()
    out = LP.fetch_snapshots(["SPY"], get=get, feed="sip")
    assert feeds == ["sip"], "a real-time consolidated feed needs no second tape"
    assert "feed" not in out["SPY"]


def test_either_tape_failing_never_fails_the_fetch(monkeypatch):
    monkeypatch.setenv("ALPACA_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET", "s")

    def iex_down(url, params=None, headers=None, timeout=None):
        if params["feed"] == "iex":
            raise OSError("iex unreachable")
        return _Resp({"SPY": DELAYED})

    out = LP.fetch_snapshots(["SPY"], get=iex_down, feed="delayed_sip")
    assert out["SPY"]["price"] == 758.21 and out["SPY"]["feed"] == "delayed_sip"

    def delayed_403(url, params=None, headers=None, timeout=None):
        return _Resp({}, 403) if params["feed"] == "delayed_sip" else _Resp({"SPY": IEX})

    out = LP.fetch_snapshots(["SPY"], get=delayed_403, feed="delayed_sip")
    assert out["SPY"]["price"] == 759.4 and out["SPY"]["feed"] == "iex"


def test_latest_hands_the_callers_the_fresher_price(monkeypatch):
    """analyze, the dossier and the ETF Builder all read live_price.latest()."""
    monkeypatch.setenv("ALPACA_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET", "s")

    def get(url, params=None, headers=None, timeout=None):
        return _Resp({"SPY": DELAYED if params["feed"] == "delayed_sip" else IEX})

    got = LP.latest("SPY", "2026-11-11", get=get, feed="delayed_sip")
    assert got["price"] == 759.4 and got["source"] == "latest_trade, iex real-time"
    assert LP.latest("SPY", "2026-11-12", get=get, feed="delayed_sip") is None, "same session as the bars: no overlay"
