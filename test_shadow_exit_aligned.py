"""Exit-aligned shadow ruler (2026-10-14, [RULE-REVIEW] measurement change).

shadow_portfolio's old ruler labels every record +2% target / -4% stop /
N-bar timeout and claims that "matches bot's take-profit" — it does not. The
only code that places live stock exits is server/bot.ts checkPositionOnTick:
a 2xATR R-ladder with scale-outs in thirds, breakeven, phase trails, a
conditional take-profit and a 7-calendar-day flat-position time stop. These
tests pin the NEW ruler (_label_exit_aligned / backfill_exit_aligned /
shadow_band_report()["exit_aligned"]) and, just as hard, that the OLD ruler,
its record keys, get_shadow_stats(), load_shadow_data() (ML training input)
and every pre-existing probe key are byte-identical with the new data
present. Every test here fails on the pre-change code (the new names do not
exist there).
"""
import copy
import json
import os
from datetime import date, datetime, timedelta, timezone

import pytest

import shadow_portfolio as sp

HERE = os.path.dirname(os.path.abspath(__file__))


# ── fixtures ─────────────────────────────────────────────────────────────────
def B(day, o, h, lo, c):
    return {"t": f"{day}T04:00:00Z", "o": o, "h": h, "l": lo, "c": c, "v": 1000}


def weekdays(start, n):
    """n weekday dates (YYYY-MM-DD) strictly after `start`."""
    d = datetime.strptime(start, "%Y-%m-%d").date()
    out = []
    while len(out) < n:
        d += timedelta(days=1)
        if d.weekday() < 5:
            out.append(d.isoformat())
    return out


def weekdays_before(end, n):
    d = datetime.strptime(end, "%Y-%m-%d").date()
    out = []
    while len(out) < n:
        d -= timedelta(days=1)
        if d.weekday() < 5:
            out.append(d.isoformat())
    return sorted(out)


def flat_bars(days, base=100.0, half_range=1.0):
    """Bars with o=c=base and a fixed high-low range: every true range is
    2*half_range, so Wilder ATR == 2*half_range exactly."""
    return [B(d, base, base + half_range, base - half_range, base) for d in days]


def drift_bars(days, level):
    """Quiet bars pinned near `level` (range +/-0.3)."""
    return [B(d, level, level + 0.3, level - 0.3, level) for d in days]


@pytest.fixture
def tmp_log(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "SHADOW_LOG_PATH", str(tmp_path / "shadow.json"))
    monkeypatch.setattr(sp, "SHADOW_LOCK_PATH", str(tmp_path / "shadow.json.lock"))
    monkeypatch.setattr(sp, "SHADOW_STATS_PATH", str(tmp_path / "shadow_stats.json"))
    return tmp_path


def _rec(ticker="AAA", ts="2026-06-01T15:00:00+00:00", entry=100.0, decision="taken",
         features=None, vxx=1.0, outcomes=None, action="BUY", score=70.0):
    return {
        "ticker": ticker, "timestamp": ts, "score": score, "decision": decision,
        "decision_reason": "seed", "entry_price": entry, "vxx_ratio": vxx,
        "regime_label": "NEUTRAL", "side": "buy", "action_label": action, "trade_type": "stock",
        "features": {} if features is None else features,
        "outcomes": outcomes if outcomes is not None else {"+5d": None, "+10d": None, "+20d": None},
        "code_version": "test-shadow",
    }


ENTRY = "2026-06-01"  # a Monday


# ── 1. the pure labeler mirrors checkPositionOnTick ─────────────────────────
def test_phase1_stop_fills_at_the_2x_atr_level():
    out = sp._label_exit_aligned(100.0, [B("2026-06-02", 100, 100.5, 95, 96)], 2.0, "neutral", ENTRY)
    assert (out["label"], out["exit_reason"], out["return_pct"]) == (0, "stop", -4.0)
    assert out["stop_pct"] == 4.0 and out["risk_pct"] == 4.0 and out["r_multiple"] == -1.0


def test_gap_through_the_stop_fills_at_the_open_not_the_stop():
    """The old ruler fills a stop at stop_price even when the bar opened far
    below it (optimistic); live fills on the first tick, i.e. near the open."""
    out = sp._label_exit_aligned(100.0, [B("2026-06-02", 90, 91, 89, 90.5)], 2.0, "neutral", ENTRY)
    assert (out["exit_reason"], out["return_pct"]) == ("stop", -10.0)
    old = sp._label_from_path(100.0, [B("2026-06-02", 90, 91, 89, 90.5)], max_bars=5)
    assert old[0:3] == (0, 96.0, "stop"), "old ruler unchanged: still fills at its stop level"


@pytest.mark.parametrize("atr_pct,stop_pct,low,exit_ret", [
    (0.5, 1.5, 98.0, -1.5),   # 2 x 0.5 = 1.0 -> clamped UP to 1.5
    (6.0, 8.0, 91.0, -8.0),   # 2 x 6.0 = 12  -> clamped DOWN to 8.0
    (3.0, 6.0, 93.0, -6.0),   # inside the clamp: 2 x ATR
])
def test_phase1_stop_clamp_at_both_bounds(atr_pct, stop_pct, low, exit_ret):
    out = sp._label_exit_aligned(100.0, [B("2026-06-02", 100, 100.2, low, low + 0.5)],
                                 atr_pct, "neutral", ENTRY)
    assert out["stop_pct"] == stop_pct
    assert out["risk_pct"] == 2 * atr_pct, "R stays UNclamped 2 x ATR% (bot_engine.py initial_risk_pct)"
    assert (out["exit_reason"], out["return_pct"]) == ("stop", exit_ret)


def test_scale_out_at_1r_then_breakeven_on_the_rest():
    # ATR 0.6 -> R 1.2%; phase-2 trail max(1.5, 0.9)=1.5 sits BELOW entry, so breakeven binds.
    bars = [B("2026-06-02", 100.2, 101.3, 100.1, 101.0),
            B("2026-06-03", 100.8, 100.9, 99.0, 99.5)]
    out = sp._label_exit_aligned(100.0, bars, 0.6, "neutral", ENTRY)
    assert out["exit_reason"] == "breakeven" and out["scale_outs"] == 1
    assert out["return_pct"] == pytest.approx(1.2 / 3, abs=1e-3)   # 1/3 at +1.2%, 2/3 at 0
    assert out["label"] == 1


def test_bullish_runner_trails_3x_atr_from_the_highest_price():
    bars = [B("2026-06-02", 100.5, 104.5, 100, 104), B("2026-06-03", 104, 113, 103.5, 112.5),
            B("2026-06-04", 113, 130, 112, 129), B("2026-06-05", 128, 128.5, 120, 121)]
    out = sp._label_exit_aligned(100.0, bars, 2.0, "bullish", ENTRY)
    # 1/3 at 1R (+4), 1/3 at 3R (+12), last third trails 3 x 2% = 6% below the 130 high -> 122.2
    assert out["exit_reason"] == "trail" and out["scale_outs"] == 2
    assert out["return_pct"] == pytest.approx((4 + 12 + 22.2) / 3, abs=1e-3)
    assert out["mfe_pct"] == 30.0 and out["exit_date"] == "2026-06-05"


def test_bearish_takes_profit_right_after_the_second_scale_out():
    bars = [B("2026-06-02", 100.5, 104.5, 100, 104), B("2026-06-03", 104, 107, 103.5, 106.5)]
    out = sp._label_exit_aligned(100.0, bars, 2.0, "bearish", ENTRY)
    # scale 2 at 1.5R = +6%; TP = max(4, min(3 x 2, 15)) = 6% while phase < 3 -> last third at +6%
    assert (out["exit_reason"], out["scale_outs"]) == ("take_profit", 2)
    assert out["return_pct"] == pytest.approx((4 + 6 + 6) / 3, abs=1e-3)


def test_neutral_never_takes_profit_live_quirk_mirrored():
    """scaleOut2R = 2.0R in NEUTRAL promotes phase 3 on the same tick, and TP
    requires phase < 3 — so the last third never takes profit there."""
    bars = [B("2026-06-02", 100.5, 104.5, 100, 104), B("2026-06-03", 104, 109, 103.5, 108.5),
            B("2026-06-04", 108.5, 109.0, 105.0, 105.5)]
    out = sp._label_exit_aligned(100.0, bars, 2.0, "neutral", ENTRY)
    assert out["exit_reason"] == "trail", out
    # phase 3 trail = max(1.0, 2.0) = 2 P&L points below the +9% peak -> exit at +7%
    assert out["return_pct"] == pytest.approx((4 + 8 + 7) / 3, abs=1e-3)


def test_time_stop_fires_after_7_calendar_days_inside_the_band():
    days = weekdays(ENTRY, 6)                  # Tue 06-02 .. Fri 06-05, Mon 06-08, Tue 06-09
    bars = drift_bars(days[:4], 101.0) + [B(days[4], 101.4, 101.6, 101.0, 101.5),
                                          B(days[5], 101.4, 101.6, 101.0, 101.5)]
    out = sp._label_exit_aligned(100.0, bars, 2.0, "neutral", ENTRY)
    # four sessions at +1% (inside the band) do NOT exit: 06-05 is only day 4.
    # 06-08 is calendar day 7 -> fires on that session's first tick (the open)
    assert days[4] == "2026-06-08"
    assert (out["exit_reason"], out["exit_date"], out["return_pct"]) == ("time_stop", "2026-06-08", 1.4)
    assert out["bars_held"] == 5


def test_time_stop_counts_7_calendar_days_from_a_mid_week_entry():
    """A Wednesday entry puts calendar days 5 and 6 on trading sessions (Mon,
    Tue), so a 5- or 6-day rule would fire there; live fires on day 7 only."""
    wed = "2026-06-03"
    assert datetime.strptime(wed, "%Y-%m-%d").weekday() == 2
    days = weekdays(wed, 6)                    # Thu 06-04, Fri 06-05, Mon 06-08 (day 5), Tue 06-09, Wed 06-10, Thu 06-11
    out = sp._label_exit_aligned(100.0, drift_bars(days, 101.0), 2.0, "neutral", wed)
    assert days[2:5] == ["2026-06-08", "2026-06-09", "2026-06-10"]
    assert (out["exit_reason"], out["exit_date"], out["bars_held"]) == ("time_stop", "2026-06-10", 5)
    assert out["return_pct"] == 1.0


def test_time_stop_band_entered_from_above_fills_at_the_band_edge():
    days = weekdays(ENTRY, 6)
    bars = drift_bars(days[:5], 103.0) + [B(days[5], 103.0, 103.2, 101.0, 101.5)]
    out = sp._label_exit_aligned(100.0, bars, 2.0, "neutral", ENTRY)
    assert out["exit_reason"] == "time_stop" and out["return_pct"] == pytest.approx(2.0, abs=1e-3)


def test_strict_level_exactly_at_the_open_fires_on_the_next_tick_not_the_bar_low():
    """|pnl| < 2.0 is strict: an open exactly at +2.0% does not exit, but the
    first tick below it does — so the fill is ~+2.0%, not the bar's low."""
    days = weekdays(ENTRY, 5)
    bars = drift_bars(days[:4], 103.0) + [B(days[4], 102.0, 102.4, 101.0, 101.5)]
    out = sp._label_exit_aligned(100.0, bars, 2.0, "neutral", ENTRY)
    assert (out["exit_reason"], out["exit_date"]) == ("time_stop", "2026-06-08")
    assert out["return_pct"] == pytest.approx(2.0, abs=1e-3)


def test_flat_position_outside_the_band_runs_to_the_cap_and_is_censored():
    days = weekdays(ENTRY, 25)
    bars = drift_bars(days, 103.0)             # +3%: under 1R (4%), outside the +/-2% band
    out = sp._label_exit_aligned(100.0, bars, 2.0, "neutral", ENTRY)
    assert out["censored"] is True and out["exit_reason"] == "open_at_cap"
    assert out["bars_held"] == sp.XA_CAP_BARS == 20 and out["exit_date"] == days[19]
    assert out["return_pct"] == 3.0 and out["label"] == 1


def test_same_bar_stop_and_profit_resolves_stop_first():
    """OPEN -> LOW -> HIGH -> CLOSE: the adverse leg comes first, so a bar
    touching both the stop and 1R is a stop — the old 'ambiguous' rule."""
    bar = [B("2026-06-02", 100, 105, 95, 104)]
    out = sp._label_exit_aligned(100.0, bar, 2.0, "neutral", ENTRY)
    assert (out["label"], out["exit_reason"], out["return_pct"]) == (0, "stop", -4.0)
    assert sp._label_from_path(100.0, bar, max_bars=5)[0:3] == (0, 96.0, "ambiguous")


def test_short_path_without_an_exit_is_retryable_never_labeled():
    out = sp._label_exit_aligned(100.0, drift_bars(weekdays(ENTRY, 5), 103.0), 2.0, "neutral", ENTRY)
    assert (out["label"], out["exit_reason"]) == (-1, "incomplete_path")


# ── 2. ATR, regime and entry session: decision-time only ─────────────────────
def _path_record_bars(entry_day=ENTRY, prior_half_range=1.0):
    prior = flat_bars(weekdays_before(entry_day, 41), half_range=prior_half_range)
    entry_bar = [B(entry_day, 100, 101, 99, 100)]
    fwd = drift_bars(weekdays(entry_day, 25), 103.0)
    return prior + entry_bar + fwd


def test_prior_bar_atr_ignores_every_bar_on_or_after_the_entry_session():
    rec = _rec(features={})
    bars = _path_record_bars()
    base = sp._exit_aligned_outcome(rec, bars)
    assert base["atr_src"] == "prior_bars" and base["atr_pct"] == pytest.approx(2.0)
    assert base["atr_pct_prior"] == pytest.approx(2.0) and base["stop_pct"] == pytest.approx(4.0)

    spiked = copy.deepcopy(bars)
    for b in spiked:
        if b["t"][:10] >= ENTRY:               # entry session + the whole forward path
            b["h"] = b["h"] * 1.25
    after = sp._exit_aligned_outcome(rec, spiked)
    for k in ("atr_pct", "atr_pct_prior", "risk_pct", "stop_pct", "regime_class", "atr_src"):
        assert after[k] == base[k], f"{k} moved on a post-entry spike (lookahead)"

    control = copy.deepcopy(bars)
    control[-30]["h"] += 20                     # a PRE-entry bar: the ATR must notice
    assert sp._exit_aligned_outcome(rec, control)["atr_pct_prior"] != base["atr_pct_prior"]


def test_prior_bar_atr_is_bot_engine_get_atr_method(monkeypatch):
    """Same Wilder ATR(14) seeded with the mean of the first 14 TRs."""
    import requests
    import bot_engine
    days = weekdays_before(ENTRY, 41)
    bars = [B(d, 100 + i * 0.3, 101 + i * 0.3 + (i % 5) * 0.4, 99 + i * 0.3 - (i % 3) * 0.2, 100.2 + i * 0.3)
            for i, d in enumerate(days)]

    class _R:
        status_code = 200

        def json(self):
            return {"bars": bars}

    monkeypatch.setattr(requests, "get", lambda *a, **k: _R())
    live_atr = bot_engine._get_atr("AAA")
    assert sp._xa_prior_atr_pct(bars, 100.0) == pytest.approx(live_atr / 100.0 * 100.0)


def test_feature_atr_is_preferred_and_the_3pct_default_falls_back():
    bars = _path_record_bars()
    assert sp._exit_aligned_outcome(_rec(features={"atr_pct": 2.5}), bars)["atr_src"] == "feature"
    assert sp._exit_aligned_outcome(_rec(features={"atr_pct": 2.5}), bars)["risk_pct"] == 5.0
    fb = sp._exit_aligned_outcome(_rec(features={"atr_pct": 3.0}), bars)
    assert fb["atr_src"] == "prior_bars" and fb["atr_pct"] == pytest.approx(2.0)


def test_prior_bars_only_mode_ignores_the_feature(monkeypatch):
    monkeypatch.setattr(sp, "XA_ATR_PREFER_FEATURE", False)
    out = sp._exit_aligned_outcome(_rec(features={"atr_pct": 5.0}), _path_record_bars())
    assert out["atr_src"] == "prior_bars" and out["atr_pct"] == pytest.approx(2.0)


def test_too_few_prior_bars_and_no_feature_is_no_atr():
    bars = _path_record_bars()[30:]             # 11 prior bars left
    out = sp._exit_aligned_outcome(_rec(features={}), bars)
    assert (out["label"], out["exit_reason"]) == (-1, "no_atr")


def test_scale_guard_rejects_a_split_rescaled_path():
    out = sp._exit_aligned_outcome(_rec(entry=1000.0, features={"atr_pct": 2.0}), _path_record_bars())
    assert (out["label"], out["exit_reason"]) == (-1, "scale_mismatch")


def test_forward_path_starts_strictly_after_the_entry_session():
    """The entry session's own bar (its range includes prices from BEFORE the
    scan) must never enter the path: a crash to 90 and a spike to 125 inside
    the entry session change nothing. With the entry bar in the path, the
    leg 100 -> 90 would stop the trade out on the entry date."""
    rec = _rec(features={"atr_pct": 2.0})
    bars = _path_record_bars()
    base = sp._exit_aligned_outcome(rec, bars)
    assert base["exit_reason"] == "open_at_cap" and base["exit_date"] > ENTRY

    wild = copy.deepcopy(bars)
    entry_bar = [b for b in wild if b["t"][:10] == ENTRY]
    assert len(entry_bar) == 1
    entry_bar[0].update({"o": 100, "h": 125, "l": 90, "c": 100})
    assert sp._exit_aligned_outcome(rec, wild) == base


def test_scale_guard_uses_the_entry_session_bar_so_a_real_gap_is_labeled():
    """A genuine overnight gap-down (entry 100 inside the entry session's
    99-101 range, next session 78-82) is a STOP filled at the open, not a
    split. Judged on the next session alone, 100 > 1.15 x 82 would read as a
    rescaled path and the trade would be dropped as scale_mismatch."""
    bars = _path_record_bars()
    d1 = weekdays(ENTRY, 1)[0]
    bars = [b for b in bars if b["t"][:10] != d1] + [B(d1, 80, 82, 78, 80)]
    out = sp._exit_aligned_outcome(_rec(features={"atr_pct": 2.0}), bars)
    assert (out["label"], out["exit_reason"], out["exit_date"], out["return_pct"]) == (0, "stop", d1, -20.0)


def test_entry_session_is_the_new_york_date_not_the_utc_date():
    assert sp._xa_entry_date_et("2026-06-01T15:00:00+00:00") == "2026-06-01"
    assert sp._xa_entry_date_et("2026-06-02T01:30:00+00:00") == "2026-06-01"   # 21:30 ET
    # a 21:30-ET scan's path starts at the NEXT session (06-02), which the
    # old UTC-date filter (t[:10] > "2026-06-02") would have skipped
    bars = _path_record_bars()
    bars = [b for b in bars if b["t"][:10] != "2026-06-02"] + [B("2026-06-02", 95, 95.5, 94, 94.5)]
    out = sp._exit_aligned_outcome(_rec(ts="2026-06-02T01:30:00+00:00", features={"atr_pct": 2.0}), bars)
    assert (out["exit_reason"], out["exit_date"], out["return_pct"]) == ("stop", "2026-06-02", -5.0)


@pytest.mark.parametrize("vxx,spy,cls", [
    (0.90, 1.02, "bullish"),    # BULL
    (1.00, 1.00, "neutral"),    # NEUTRAL
    (1.07, 1.00, "bearish"),    # CAUTION
    (1.20, 1.00, "bearish"),    # BEAR
    (1.50, 1.00, "bearish"),    # PANIC
])
def test_regime_class_uses_get_market_regime_not_the_markov_label(vxx, spy, cls):
    rec = _rec(vxx=vxx, features={"spy_vs_ma50": spy})
    rec["regime_label"] = "NEUTRAL_BULL"      # Markov taxonomy — must be ignored
    assert sp._xa_regime_class(rec) == cls


# ── 3. paginated fetch ───────────────────────────────────────────────────────
def test_fetch_follows_next_page_token(monkeypatch):
    import requests
    calls = []
    pages = [
        {"bars": {"AAA": [B("2026-06-02", 1, 1, 1, 1), B("2026-06-01", 1, 1, 1, 1)],
                  "BBB": [B("2026-06-01", 2, 2, 2, 2)]}, "next_page_token": "tok1"},
        {"bars": {"BBB": [B("2026-06-02", 2, 2, 2, 2)]}, "next_page_token": None},
    ]

    class _R:
        status_code = 200

        def __init__(self, body):
            self._b = body

        def json(self):
            return self._b

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append(dict(params))
        return _R(pages[len(calls) - 1])

    monkeypatch.setattr(requests, "get", fake_get)
    out = sp._fetch_bars_paginated(["AAA", "BBB"], "2026-04-01", "2026-07-30")
    assert [b["t"][:10] for b in out["AAA"]] == ["2026-06-01", "2026-06-02"]
    assert [b["t"][:10] for b in out["BBB"]] == ["2026-06-01", "2026-06-02"]
    assert len(calls) == 2 and "page_token" not in calls[0] and calls[1]["page_token"] == "tok1"
    assert calls[0]["limit"] == 10000 and calls[0]["adjustment"] == "split"


class _Resp:
    def __init__(self, status=200, body=None, bad_json=False):
        self.status_code = status
        self._b = body if body is not None else {"bars": {}}
        self._bad = bad_json

    def json(self):
        if self._bad:
            raise ValueError("not json")
        return self._b


def test_fetch_marks_every_symbol_of_an_incomplete_chunk_unfetched(monkeypatch):
    """Page 1 succeeds, page 2 fails: the chunk's data is partial, so ALL of
    its symbols are unfetched (not 'missing'); a completed chunk is not."""
    import requests
    monkeypatch.setattr(sp, "BATCH_SIZE", 2)
    pages = {("AAA,BBB", None): _Resp(body={"bars": {"AAA": [B("2026-06-02", 1, 1, 1, 1)]}, "next_page_token": "p2"}),
             ("AAA,BBB", "p2"): _Resp(status=500),
             ("CCC", None): _Resp(body={"bars": {}, "next_page_token": None})}
    monkeypatch.setattr(requests, "get",
                        lambda url, params=None, headers=None, timeout=None: pages[(params["symbols"], params.get("page_token"))])
    unfetched = set()
    out = sp._fetch_bars_paginated(["AAA", "BBB", "CCC"], "2026-04-01", "2026-07-30", unfetched=unfetched)
    assert unfetched == {"AAA", "BBB"}
    assert "CCC" not in out, "a completed chunk with no bars is genuinely missing data"


def test_no_request_starts_unless_its_timeout_fits_before_the_deadline(monkeypatch):
    import requests
    calls = []
    monkeypatch.setattr(requests, "get", lambda *a, **k: calls.append(1) or _Resp())
    unfetched = set()
    deadline = sp.time.monotonic() + sp.XA_REQUEST_TIMEOUT_S - 1
    assert sp._fetch_bars_paginated(["AAA", "BBB"], "2026-04-01", "2026-07-30",
                                    deadline=deadline, unfetched=unfetched) == {}
    assert calls == [] and unfetched == {"AAA", "BBB"}


def test_429_back_off_runs_only_when_it_fits_in_the_budget(monkeypatch):
    import requests
    slept = []
    monkeypatch.setattr(sp.time, "sleep", lambda s: slept.append(s))
    seq = []

    def fake_get(url, params=None, headers=None, timeout=None):
        seq.append(timeout)
        return _Resp(status=429) if len(seq) == 1 else _Resp(body={"bars": {"AAA": [B("2026-06-02", 1, 1, 1, 1)]}})

    monkeypatch.setattr(requests, "get", fake_get)
    unfetched = set()
    tight = sp.time.monotonic() + sp.XA_429_BACKOFF_S + sp.XA_REQUEST_TIMEOUT_S - 1
    assert sp._fetch_bars_paginated(["AAA"], "2026-04-01", "2026-07-30", deadline=tight, unfetched=unfetched) == {}
    assert slept == [] and unfetched == {"AAA"} and len(seq) == 1, "no 30s sleep when it cannot fit"

    seq.clear()
    unfetched = set()
    out = sp._fetch_bars_paginated(["AAA"], "2026-04-01", "2026-07-30", deadline=None, unfetched=unfetched)
    assert slept == [sp.XA_429_BACKOFF_S] and len(seq) == 2 and "AAA" in out and not unfetched
    assert seq == [sp.XA_REQUEST_TIMEOUT_S] * 2


def test_old_fetch_helper_is_untouched():
    """Fixing the old helper's limit=100 truncation changes the OLD ruler's
    inputs — that is a separate change, not this one."""
    import inspect
    src = inspect.getsource(sp._fetch_historical_bars_batch)
    assert src.count('"limit": 100') == 2 and "page_token" not in src


# ── 4. backfill: new keys only, newest first, bounded, merge-save ───────────
def _fake_fetch_factory(by_ticker):
    def fake(tickers, start, end, **kw):   # **kw: deadline / unfetched (a fake fetch always completes)
        return {t: copy.deepcopy(by_ticker[t]) for t in tickers if t in by_ticker}
    return fake


def _ts_days_ago(days, hour=15):
    d = datetime.now(timezone.utc) - timedelta(days=days)
    return d.replace(hour=hour, minute=0, second=0, microsecond=0).isoformat()


def _bars_for_entry(ts):
    return _path_record_bars(entry_day=sp._xa_entry_date_et(ts))


def test_backfill_labels_only_eligible_records_and_writes_only_new_keys(tmp_log, monkeypatch):
    old_ts, new_ts = _ts_days_ago(60), _ts_days_ago(10)
    old_outcomes = {"+5d": {"return_pct": -4.0, "label": 0}, "+10d": None, "+20d": None}
    recs = [_rec("AAA", old_ts, outcomes=copy.deepcopy(old_outcomes)), _rec("BBB", new_ts)]
    recs[0]["_bf_attempts"] = {"+10d": 1}
    sp._save_shadow_log(recs)
    before = copy.deepcopy(sp._load_shadow_log())
    monkeypatch.setattr(sp, "_fetch_bars_paginated", _fake_fetch_factory({"AAA": _bars_for_entry(old_ts)}))

    stats = sp.backfill_exit_aligned(max_records=100, max_windows=5, time_budget_s=30)
    after = sp._load_shadow_log()
    assert stats["updated"] == 1 and stats["not_eligible"] == 1 and stats["saved"] is True
    assert after[0][sp.XA_KEY]["label"] in (0, 1)
    assert after[0][sp.XA_KEY]["label_method"] == "exit_aligned_v1"
    assert sp.XA_KEY not in after[1]
    for b, a in zip(before, after):
        assert {k: v for k, v in a.items() if k not in (sp.XA_KEY, sp.XA_ATTEMPTS_KEY)} == b, \
            "the old ruler's record keys (outcomes, _bf_attempts, ...) must be byte-identical"


def test_eligibility_mirrors_the_plus20d_calendar_convention(tmp_log, monkeypatch):
    sp._save_shadow_log([_rec("AAA", _ts_days_ago(50)), _rec("BBB", _ts_days_ago(52))])
    fetched = []
    monkeypatch.setattr(sp, "_fetch_bars_paginated",
                        lambda t, s, e, **kw: fetched.extend(t) or {})
    stats = sp.backfill_exit_aligned()
    assert sp.XA_ELIGIBLE_CALENDAR_DAYS == 51 and fetched == ["BBB"]
    assert stats["not_eligible"] == 1


def test_backfill_goes_newest_day_first_and_respects_the_window_cap(tmp_log, monkeypatch):
    older, newer = _ts_days_ago(90), _ts_days_ago(60)
    sp._save_shadow_log([_rec("OLD", older), _rec("NEW", newer)])
    monkeypatch.setattr(sp, "_fetch_bars_paginated", _fake_fetch_factory(
        {"OLD": _bars_for_entry(older), "NEW": _bars_for_entry(newer)}))
    stats = sp.backfill_exit_aligned(max_windows=1)
    recs = {r["ticker"]: r for r in sp._load_shadow_log()}
    assert sp.XA_KEY in recs["NEW"] and sp.XA_KEY not in recs["OLD"]
    assert stats["windows_fetched"] == 1 and stats["windows_deferred"] == 1
    sp.backfill_exit_aligned(max_windows=1)                     # next night: the backlog drains
    assert sp.XA_KEY in {r["ticker"]: r for r in sp._load_shadow_log()}["OLD"]


def test_missing_bars_retry_on_their_own_counter_then_go_terminal(tmp_log, monkeypatch):
    sp._save_shadow_log([_rec("GONE", _ts_days_ago(70))])
    monkeypatch.setattr(sp, "_fetch_bars_paginated", lambda t, s, e, **kw: {})
    sp.backfill_exit_aligned()
    rec = sp._load_shadow_log()[0]
    assert rec[sp.XA_ATTEMPTS_KEY] == 1 and sp.XA_KEY not in rec and "_bf_attempts" not in rec
    sp.backfill_exit_aligned()
    stats = sp.backfill_exit_aligned()
    rec = sp._load_shadow_log()[0]
    assert rec[sp.XA_KEY] == {"label": -1, "exit_reason": "no_price_data", "label_method": "exit_aligned_v1"}
    assert sp.XA_ATTEMPTS_KEY not in rec and stats["permanently_unlabelable"] == 1
    assert rec["outcomes"] == {"+5d": None, "+10d": None, "+20d": None}


@pytest.mark.parametrize("failure", ["http_503", "http_401", "exception", "bad_json"])
def test_a_failed_fetch_never_strikes_a_record(tmp_log, monkeypatch, failure):
    """An Alpaca outage / bad key must not retire records as no_price_data:
    four nights of failed fetches leave the record exactly as it was."""
    import requests

    def fake_get(url, params=None, headers=None, timeout=None):
        if failure == "exception":
            raise requests.ConnectionError("down")
        if failure == "bad_json":
            return _Resp(bad_json=True)
        return _Resp(status=int(failure.split("_")[1]))

    monkeypatch.setattr(requests, "get", fake_get)
    sp._save_shadow_log([_rec("AAA", _ts_days_ago(70))])
    before = sp._load_shadow_log()
    for _night in range(sp.MAX_BACKFILL_ATTEMPTS + 1):
        stats = sp.backfill_exit_aligned()
        assert (stats["fetch_failed"], stats["missing_price"], stats["permanently_unlabelable"]) == (1, 0, 0)
    assert sp._load_shadow_log() == before, "no XA key, no retry counter"


def test_a_completed_fetch_with_no_bars_still_strikes(tmp_log, monkeypatch):
    """The no-strike rule is for FAILED fetches only: a 200 with no bars for
    the symbol is missing data and counts toward MAX_BACKFILL_ATTEMPTS."""
    import requests
    monkeypatch.setattr(sp, "BATCH_SIZE", 1)
    monkeypatch.setattr(requests, "get", lambda url, params=None, headers=None, timeout=None:
                        _Resp(status=503) if params["symbols"] == "BBB" else _Resp(body={"bars": {}}))
    sp._save_shadow_log([_rec("AAA", _ts_days_ago(70)), _rec("BBB", _ts_days_ago(70))])
    stats = sp.backfill_exit_aligned()
    recs = {r["ticker"]: r for r in sp._load_shadow_log()}
    assert recs["AAA"][sp.XA_ATTEMPTS_KEY] == 1
    assert sp.XA_ATTEMPTS_KEY not in recs["BBB"] and sp.XA_KEY not in recs["BBB"]
    assert (stats["missing_price"], stats["fetch_failed"]) == (1, 1)


class _Killed(BaseException):
    """Stands in for the SIGKILL: nothing in the job may catch it."""


def _days_ago_records(n, first=60):
    return [_rec(f"T{i}", _ts_days_ago(first + i)) for i in range(n)]


def _labeled_tickers():
    return sorted(r["ticker"] for r in sp._load_shadow_log() if isinstance(r.get(sp.XA_KEY), dict))


def test_a_kill_keeps_every_window_saved_before_it(tmp_log, monkeypatch):
    """Checkpoint every XA_CHECKPOINT_WINDOWS windows: killed while fetching
    the 7th window, the first 5 windows' labels are already on disk (the
    round-1 job saved only at the end, so a kill wrote nothing, forever)."""
    recs = _days_ago_records(8)
    sp._save_shadow_log(recs)
    bars = {r["ticker"]: _bars_for_entry(r["timestamp"]) for r in recs}
    real = _fake_fetch_factory(bars)
    calls = []

    def fetch(tickers, start, end, **kw):
        calls.append(tickers[0])
        if len(calls) == 7:
            raise _Killed()
        return real(tickers, start, end)

    monkeypatch.setattr(sp, "XA_CHECKPOINT_SECONDS", 1e9)
    monkeypatch.setattr(sp, "_fetch_bars_paginated", fetch)
    with pytest.raises(_Killed):
        sp.backfill_exit_aligned()
    assert sp.XA_CHECKPOINT_WINDOWS == 5
    assert _labeled_tickers() == sorted(calls[:5]), "windows 1-5 survive the kill; 6 is lost, 7 never finished"
    stats = sp.backfill_exit_aligned()                          # the next run resumes
    assert stats["already_filled"] == 5 and _labeled_tickers() == sorted(r["ticker"] for r in recs)


class _Clock:
    """Fake monotonic clock for sp.time (other attributes pass through)."""
    def __init__(self):
        import time as _t
        self._t = _t
        self.now = 1000.0

    def monotonic(self):
        return self.now

    def __getattr__(self, name):
        return getattr(self._t, name)


def test_a_time_based_checkpoint_bounds_the_loss_on_slow_windows(tmp_log, monkeypatch):
    recs = _days_ago_records(6)
    sp._save_shadow_log(recs)
    real = _fake_fetch_factory({r["ticker"]: _bars_for_entry(r["timestamp"]) for r in recs})
    clock = _Clock()
    monkeypatch.setattr(sp, "time", clock)
    calls = []

    def slow_fetch(tickers, start, end, **kw):
        calls.append(tickers[0])
        if len(calls) == 4:
            raise _Killed()
        clock.now += 12.0                       # 12s per window: 2 windows cross XA_CHECKPOINT_SECONDS=20
        return real(tickers, start, end)

    monkeypatch.setattr(sp, "_fetch_bars_paginated", slow_fetch)
    with pytest.raises(_Killed):
        sp.backfill_exit_aligned()
    assert sp.XA_CHECKPOINT_SECONDS == 20.0
    assert _labeled_tickers() == sorted(calls[:2])


def test_the_run_stops_at_its_wall_clock_budget_from_entry(tmp_log, monkeypatch):
    """The budget is wall time from entry (log load included), and the real
    fetcher gets the same deadline (pinned above: no request starts unless
    its timeout fits). 3s load + 7s windows in a 60s budget: a window starts
    only while XA_REQUEST_TIMEOUT_S still fits, so 7 windows, not 40 — the
    rest are deferred, not counted as fetched-and-failed."""
    recs = _days_ago_records(30)
    sp._save_shadow_log(recs)
    real = _fake_fetch_factory({r["ticker"]: _bars_for_entry(r["timestamp"]) for r in recs})
    clock = _Clock()
    monkeypatch.setattr(sp, "time", clock)
    real_load = sp._load_shadow_log

    def slow_load():
        clock.now += 3.0                        # the log load counts against the budget
        return real_load()

    deadlines = []

    def fetch(tickers, start, end, deadline=None, unfetched=None):
        deadlines.append(deadline)
        clock.now += 7.0
        return real(tickers, start, end)

    monkeypatch.setattr(sp, "_load_shadow_log", slow_load)
    monkeypatch.setattr(sp, "_fetch_bars_paginated", fetch)
    stats = sp.backfill_exit_aligned(time_budget_s=60)
    # windows start at t=3,10,...,45 (15s left: still fits); at t=52 only 8s are left
    assert (stats["windows_fetched"], stats["windows_deferred"], stats["fetch_failed"]) == (7, 23, 0)
    assert set(deadlines) == {1000.0 + 60}, "every fetch is handed the job's own deadline"
    assert stats["elapsed_s"] <= 60 and stats["saved"] is True
    assert stats["time_budget_s"] == 60.0 and stats["checkpoints"] >= 2


def test_a_failed_checkpoint_is_retried_and_never_double_counted(tmp_log, monkeypatch):
    recs = _days_ago_records(3)
    sp._save_shadow_log(recs)
    monkeypatch.setattr(sp, "_fetch_bars_paginated",
                        _fake_fetch_factory({r["ticker"]: _bars_for_entry(r["timestamp"]) for r in recs}))
    monkeypatch.setattr(sp, "XA_CHECKPOINT_WINDOWS", 1)
    real_replace = os.replace
    replaced = []

    def flaky_replace(src, dst):                # the 1st checkpoint applies its results, then the write fails
        replaced.append(dst)
        if len(replaced) == 1:
            raise OSError("disk full")
        return real_replace(src, dst)

    monkeypatch.setattr(sp.os, "replace", flaky_replace)
    stats = sp.backfill_exit_aligned()
    assert (stats["checkpoints"], stats["save_failures"], stats["saved"]) == (3, 1, True)
    assert stats["updated"] == 3, "the failed write's counts were not committed, so its retry counts once"
    assert _labeled_tickers() == sorted(r["ticker"] for r in recs)
    assert not [f for f in os.listdir(tmp_log) if f.startswith(".shadow.")], "no orphaned temp file"


def test_merge_save_keeps_a_record_appended_during_the_fetch(tmp_log, monkeypatch):
    ts = _ts_days_ago(60)
    sp._save_shadow_log([_rec("AAA", ts)])
    real = _fake_fetch_factory({"AAA": _bars_for_entry(ts)})

    def fetch_while_scanning(tickers, start, end, **kw):
        sp.log_candidate("LATE", {"change_pct_today": 1.0}, 50.0, "rejected_score", entry_price=10.0)
        return real(tickers, start, end)

    monkeypatch.setattr(sp, "_fetch_bars_paginated", fetch_while_scanning)
    sp.backfill_exit_aligned()
    recs = sp._load_shadow_log()
    assert [r["ticker"] for r in recs] == ["AAA", "LATE"], "the mid-run append must survive the save"
    assert sp.XA_KEY in recs[0]


def test_old_backfill_output_identical_with_and_without_the_new_fields(tmp_log, monkeypatch):
    """Before/after on identical inputs for the OLD ruler: backfill_outcomes()
    must produce byte-identical outcomes whether or not records carry the
    exit-aligned key and counter, and the new job must not touch them."""
    ts = _ts_days_ago(60)
    bars = _bars_for_entry(ts)
    monkeypatch.setattr(sp, "_fetch_historical_bars_batch",
                        lambda tickers, s, e: {t: copy.deepcopy(bars) for t in tickers})

    sp._save_shadow_log([_rec("AAA", ts), _rec("BBB", ts, entry=101.0)])
    sp.backfill_outcomes(max_records=50)
    plain = [r["outcomes"] for r in sp._load_shadow_log()]
    assert all(o["+10d"] is not None for o in plain)

    seeded = [_rec("AAA", ts), _rec("BBB", ts, entry=101.0)]
    seeded[0][sp.XA_KEY] = {"label": 1, "return_pct": 9.0, "label_method": "exit_aligned_v1"}
    seeded[1][sp.XA_ATTEMPTS_KEY] = 2
    sp._save_shadow_log(seeded)
    sp.backfill_outcomes(max_records=50)
    assert [r["outcomes"] for r in sp._load_shadow_log()] == plain

    monkeypatch.setattr(sp, "_fetch_bars_paginated", _fake_fetch_factory({"AAA": bars, "BBB": bars}))
    sp.backfill_exit_aligned()
    assert [r["outcomes"] for r in sp._load_shadow_log()] == plain


# ── 5. consumers: stats, ML inputs and old probe keys are invariant ──────────
def _labeled_records():
    recs = []
    for i in range(30):
        win = i % 3 == 0
        r = _rec(f"T{i % 7}", f"2026-06-{1 + i % 20:02d}T15:00:00+00:00", entry=12.34,
                 decision="taken" if i % 2 else "rejected_score",
                 features={"change_pct_today": float(i % 25 - 5), "atr_pct": 2.0, "rsi_14": 40.0 + i},
                 outcomes={"+5d": {"return_pct": 2.0 if win else -4.0, "label": 1 if win else 0},
                           "+10d": {"return_pct": 2.0 if win else -1.5, "label": 1 if win else 0},
                           "+20d": {"label": -1}},
                 score=40.0 + i)
        recs.append(r)
    return recs


def _with_xa(recs):
    out = copy.deepcopy(recs)
    for i, r in enumerate(out):
        if i % 5 == 4:
            r[sp.XA_ATTEMPTS_KEY] = 1
            continue
        lbl = 1 if i % 4 == 0 else 0
        r[sp.XA_KEY] = {"label": lbl, "return_pct": 6.5 if lbl else -3.25, "r_multiple": 1.0,
                        "exit_reason": "trail" if lbl else "stop", "censored": i % 6 == 0,
                        "regime_class": "neutral", "label_method": "exit_aligned_v1"}
    out[3][sp.XA_KEY] = {"label": -1, "exit_reason": "scale_mismatch", "label_method": "exit_aligned_v1"}
    return out


def test_get_shadow_stats_identical_with_and_without_the_new_fields(tmp_log):
    sp._save_shadow_log(_labeled_records())
    plain = sp.get_shadow_stats()
    sp._save_shadow_log(_with_xa(_labeled_records()))
    assert sp.get_shadow_stats() == plain


def test_load_shadow_data_identical_with_and_without_the_new_fields(tmp_log, monkeypatch):
    import numpy as np
    monkeypatch.setattr(sp.time, "time", lambda: 1_790_000_000.0)
    sp._save_shadow_log(_labeled_records())
    plain = sp.load_shadow_data(horizon_days=10, min_records=5)
    sp._save_shadow_log(_with_xa(_labeled_records()))
    with_xa = sp.load_shadow_data(horizon_days=10, min_records=5)
    assert plain is not None and with_xa is not None
    for a, b in zip(plain, with_xa):
        assert np.array_equal(a, b) and a.dtype == b.dtype


OLD_BAND_REPORT_KEYS = {"labeling", "score_quintile_edges", "change_bands", "min_n", "decision",
                        "decision_x_band", "band_all", "score_quintile", "decision_x_regime",
                        "decision_x_action", "action_x_band", "all"}


def test_band_report_old_keys_identical_and_exit_aligned_added():
    plain = sp.shadow_band_report(_labeled_records(), min_n=5)
    both = sp.shadow_band_report(_with_xa(_labeled_records()), min_n=5)
    assert set(both) == set(plain) == OLD_BAND_REPORT_KEYS | {"exit_aligned"}, \
        "exactly ONE new top-level key; every old key still present"
    for k in OLD_BAND_REPORT_KEYS:
        assert both[k] == plain[k], f"old probe key {k!r} changed when exit-aligned data is present"
    assert plain["labeling"] == {
        "method": "path_dependent_v2", "win_pct": 2.0, "stop_pct": -4.0,
        "note": "return_pct is entry->exit under the labeler's target/stop/timeout rules, "
                "not a raw forward return; 'all' is the base rate across every scored candidate. "
                "Labels are LONG-stock outcomes: they mean what they say only for action BUY. "
                "Records before 2026-10-06 carry no action ('unrecorded')."}, \
        "the old 'labeling' key keeps describing the OLD ruler, verbatim"

    xa = both["exit_aligned"]
    assert xa["labeling"]["method"] == "exit_aligned_v1"
    assert "STOP FIRST" in xa["labeling"]["same_bar_rule"]
    assert xa["labeling"]["constants"]["scale_out_2_r"] == {"bullish": 3.0, "bearish": 1.5, "neutral": 2.0}
    cell = xa["all"]["all"]["xa"]
    assert cell["n"] == 23 and cell["win_rate"] == pytest.approx(100 * 6 / 23, abs=0.1)
    assert cell["mean_net_return_pct"] == pytest.approx(cell["mean_return_pct"] - 0.10, abs=1e-3)
    assert "censored_pct" in cell
    cov = xa["coverage"]
    assert sum(c["labeled"] for c in cov.values()) == 23
    assert sum(c["pending_retry"] for c in cov.values()) == 6
    assert sum(c["unlabelable"].get("scale_mismatch", 0) for c in cov.values()) == 1
    assert "unique_ticker_entry_days" in cov["taken"]
    paired = xa["paired_vs_old"]["all"]
    assert paired["n"] == 23 and sum(paired["matrix"].values()) == 23
    assert {"mean_return_pct_old_plus10d", "mean_return_pct_xa"} <= set(paired)
    for dim in ("decision", "decision_x_action", "action_x_band", "score_quintile",
                "decision_x_regime", "regime_class", "exit_reason"):
        assert dim in xa, dim


def test_exit_aligned_tables_leak_no_ticker_price_or_timestamp():
    recs = _with_xa(_labeled_records())
    for r in recs:
        r["ticker"] = "SECRETCO"
    blob = json.dumps(sp.shadow_band_report(recs, min_n=5)["exit_aligned"])
    assert "SECRETCO" not in blob and "12.34" not in blob and "2026-06" not in blob


def test_probe_entry_point_serves_both_rulers(tmp_log):
    sp._save_shadow_log(_with_xa(_labeled_records()))
    rep = sp.get_shadow_band_report()
    assert rep["labeling"]["method"] == "path_dependent_v2"
    assert rep["exit_aligned"]["labeling"]["method"] == "exit_aligned_v1"
    json.dumps(rep)


# ── 6. drift pin: the mirrored literals must still exist in server/bot.ts ────
BOT_TS_LITERALS = [
    "ps.initial_risk_pct || atrPct * 2.0",
    "else stopPct = Math.max(1.5, Math.min(atrPct * 2.0, 8.0));",
    "const tpPct = phase < 3 ? Math.max(4.0, Math.min(atrPct * 3.0, 15.0)) : 999;",
    "if (peakR >= 3 && pos.phase < 4) pos.phase = 4;",
    "else if (peakR >= 2 && pos.phase < 3) pos.phase = 3;",
    "else if (peakR >= 1 && pos.phase < 2) pos.phase = 2;",
    'const isBullish = regime === "BULL" || regime === "NEUTRAL_BULL";',
    'const isBearish = regime === "BEAR" || regime === "PANIC" || regime === "CAUTION";',
    "const scaleOut1R = 1.0;",
    "const scaleOut2R = isBullish ? 3.0 : (isBearish ? 1.5 : 2.0);",
    "const trailAtrMult = isBullish ? 3.0 : (isBearish ? 1.5 : 2.0);",
    "const thirdQty = Math.floor(pos.originalQty / 3);",
    "return; // Don't evaluate full exit on the same tick as a scale-out",
    "stopPct = Math.max(1.0, atr * 1.0);",
    "stopPct = Math.max(1.5, atr * 1.5);",
    "const drawdownFromHighPrice = ((pos.highestPrice - currentPrice) / pos.highestPrice) * 100;",
    "const drawdownFromPeak = pos.highestPnl - pnlPct;",
    "if (!shouldStop && breakevenFloor && pnlPct < 0) {",
    "if (pnlPct <= -stopPct) {",
    "if (!shouldStop && pnlPct >= pos.tpPct && pos.phase < 3 && pos.scalesCompleted >= 2) {",
    "if (daysHeld >= 7 && Math.abs(pnlPct) < 2.0) {",
]


@pytest.mark.parametrize("literal", BOT_TS_LITERALS)
def test_bot_ts_exit_literals_still_mirrored(literal):
    src = open(os.path.join(HERE, "server", "bot.ts")).read()
    assert literal in src, (
        f"server/bot.ts no longer contains {literal!r}. The live exit policy changed: update "
        "shadow_portfolio's XA_* constants / _XaPosition.tick to match (and bump the label_method), "
        "or the exit-aligned ruler silently stops describing live exits.")


def test_python_constants_match_the_pinned_literals():
    assert (sp.XA_RISK_ATR_MULT, sp.XA_P1_STOP_ATR_MULT, sp.XA_P1_STOP_MIN_PCT, sp.XA_P1_STOP_MAX_PCT) == (2.0, 2.0, 1.5, 8.0)
    assert (sp.XA_TP_MIN_PCT, sp.XA_TP_ATR_MULT, sp.XA_TP_MAX_PCT) == (4.0, 3.0, 15.0)
    assert (sp.XA_TIME_STOP_DAYS, sp.XA_TIME_STOP_BAND_PCT, sp.XA_SCALE1_R) == (7, 2.0, 1.0)
    assert sp.XA_SCALE2_R == sp.XA_TRAIL4_ATR_MULT == {"bullish": 3.0, "bearish": 1.5, "neutral": 2.0}
    assert (sp.XA_P2_TRAIL_MIN_PCT, sp.XA_P2_TRAIL_ATR_MULT, sp.XA_P3_TRAIL_MIN_PCT, sp.XA_P3_TRAIL_ATR_MULT) == (1.5, 1.5, 1.0, 1.0)
    engine = open(os.path.join(HERE, "bot_engine.py")).read()
    assert 'initial_risk_pct = ps.get("initial_risk_pct", atr_pct * 2.0)' in engine
    assert "lookback_days = max(60, period * 4)" in engine and sp.XA_ATR_LOOKBACK_DAYS == 60


def test_daemon_registers_the_new_entry_point():
    from voltrade_daemon import RPCDispatcher
    assert RPCDispatcher()._routes["shadow_backfill_exit_aligned"] == ("shadow_portfolio", "backfill_exit_aligned")
    assert callable(sp.backfill_exit_aligned)


def test_old_labeler_outputs_pinned_on_the_fixtures():
    """The old ruler on the same fixtures this file uses — values pinned so a
    change to _label_from_path cannot ride along with this ruler."""
    assert sp._label_from_path(100.0, [B("2026-06-02", 100, 100.5, 95, 96)], max_bars=5) == (0, 96.0, "stop", "2026-06-02")
    assert sp._label_from_path(100.0, [B("2026-06-02", 100, 102.5, 99.5, 102)], max_bars=5) == (1, 102.0, "target", "2026-06-02")
    bars = drift_bars(weekdays(ENTRY, 25), 101.5)
    assert sp._label_from_path(100.0, bars, max_bars=10) == (0, 101.5, "timeout", bars[9]["t"][:10])
    assert (sp.WIN_THRESHOLD_PCT, sp.LOSS_THRESHOLD_PCT, sp.FORWARD_HORIZONS_DAYS) == (2.0, -4.0, [5, 10, 20])
