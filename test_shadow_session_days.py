"""[RULE-REVIEW] 2026-11-15 — a shadow record written on a day with no session
is not a ticker-day.

server/bot.ts scheduleTier2 scans 04:00-20:00 ET on every calendar day, so a
Friday candidate is logged again on Saturday and Sunday with Friday's close
and Friday's features. The first-seen dedupe keys on (ticker, ET date), so
those copies used to survive as two more "independent" ticker-days. Pinned:
  1. _is_session_day: weekends and full-closure holidays are not sessions; a
     half day is; an unreadable date is kept;
  2. _first_seen_per_ticker_day drops non-session records (the Friday row
     stays, the Saturday and Sunday copies go) and still keys on the ET date
     (Friday 21:00 ET = Saturday 02:00Z is Friday);
  3. the off switch restores the prior rows exactly;
  4. load_shadow_data (the ML training input) follows the same rule;
  5. session_day_counts / get_shadow_stats report what the rule removes
     without changing any other stats key.
Tests 2, 4 and 5 fail on the pre-change code."""
import json

import pytest

import shadow_portfolio as sp

FRI = "2026-11-13T15:00:00+00:00"       # 10:00 ET Friday, a session
FRI_LATE = "2026-11-14T02:00:00+00:00"  # 21:00 ET Friday (UTC Saturday)
SAT = "2026-11-14T15:00:00+00:00"       # Saturday
SUN = "2026-11-15T15:00:00+00:00"       # Sunday
MON = "2026-11-16T15:00:00+00:00"       # Monday, a session
THANKSGIVING = "2026-11-26T15:00:00+00:00"   # full closure
BLACK_FRIDAY = "2026-11-27T15:00:00+00:00"   # half day: a session


@pytest.fixture
def tmp_log(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "SHADOW_LOG_PATH", str(tmp_path / "shadow.json"))
    monkeypatch.setattr(sp, "SHADOW_LOCK_PATH", str(tmp_path / "shadow.json.lock"))
    monkeypatch.setattr(sp, "SHADOW_STATS_PATH", str(tmp_path / "shadow_stats.json"))
    monkeypatch.delenv(sp.SESSION_DAYS_ONLY_ENV, raising=False)
    sp._shadow_log_cache_reset()
    yield tmp_path
    sp._shadow_log_cache_reset()


def _rec(ticker, ts, label=1):
    out = {"label": label, "return_pct": 2.0 if label == 1 else -4.0,
           "exit_reason": "target" if label == 1 else "stop",
           "label_method": "path_dependent_v2", "exit_date": "2026-11-30"}
    return {"ticker": ticker, "timestamp": ts, "score": 60.0, "decision": "rejected_score",
            "entry_price": 10.0, "features": {"rsi_14": 50.0, "atr_pct": 2.0},
            "outcomes": {"+5d": None, "+10d": out, "+20d": None}}


def _write(path, recs):
    with open(path, "w") as fh:
        json.dump(recs, fh)
    sp._shadow_log_cache_reset()


WEEKEND_LOG = [
    _rec("AAA", FRI), _rec("AAA", SAT), _rec("AAA", SUN), _rec("AAA", MON),
    _rec("BBB", FRI_LATE), _rec("BBB", SAT),
    _rec("CCC", SUN),                       # first seen on a closed day only
    _rec("DDD", THANKSGIVING), _rec("DDD", BLACK_FRIDAY),
]


def test_session_day_is_a_weekday_that_is_not_a_full_closure():
    assert sp._is_session_day("2026-11-13") is True
    assert sp._is_session_day("2026-11-14") is False     # Saturday
    assert sp._is_session_day("2026-11-15") is False     # Sunday
    assert sp._is_session_day("2026-11-26") is False     # Thanksgiving
    assert sp._is_session_day("2026-11-27") is True      # half day is a session
    assert sp._is_session_day("2027-01-01") is False     # 2027 table is read too
    # the rule only removes what it can place on a closed day
    assert sp._is_session_day(None) is True
    assert sp._is_session_day("") is True
    assert sp._is_session_day("not-a-date") is True
    # a year outside market_calendar's tables: weekend test alone
    assert sp._is_session_day("2031-07-04") is True      # a Friday; no 2031 table
    assert sp._is_session_day("2031-07-05") is False     # Saturday


def test_dedupe_drops_copies_written_on_closed_days(monkeypatch):
    monkeypatch.delenv(sp.SESSION_DAYS_ONLY_ENV, raising=False)
    dd = sp._first_seen_per_ticker_day(WEEKEND_LOG, lambda r: True)
    got = sorted((r["ticker"], r["timestamp"]) for r in dd)
    assert got == [("AAA", FRI), ("AAA", MON), ("BBB", FRI_LATE), ("DDD", BLACK_FRIDAY)]
    # CCC was only ever logged on a Sunday: it is no ticker-day at all
    assert not any(r["ticker"] == "CCC" for r in dd)
    # the input is not modified
    assert len(WEEKEND_LOG) == 9


@pytest.mark.parametrize("off", ["0", "false", "off", "no"])
def test_off_switch_restores_every_calendar_day(monkeypatch, off):
    monkeypatch.setenv(sp.SESSION_DAYS_ONLY_ENV, off)
    dd = sp._first_seen_per_ticker_day(WEEKEND_LOG, lambda r: True)
    got = sorted((r["ticker"], r["timestamp"]) for r in dd)
    assert got == sorted([
        ("AAA", FRI), ("AAA", SAT), ("AAA", SUN), ("AAA", MON),
        ("BBB", FRI_LATE), ("BBB", SAT), ("CCC", SUN),
        ("DDD", THANKSGIVING), ("DDD", BLACK_FRIDAY),
    ])


def test_training_rows_follow_the_same_rule(tmp_log, monkeypatch):
    _write(sp.SHADOW_LOG_PATH, WEEKEND_LOG)
    out = sp.load_shadow_data(horizon_days=10, min_records=1, return_meta=True)
    assert out is not None
    X, y, _w, meta = out
    assert len(X) == len(y) == 4
    assert sorted(zip(meta["tickers"], meta["dates"])) == [
        ("AAA", "2026-11-13"), ("AAA", "2026-11-16"), ("BBB", "2026-11-13"), ("DDD", "2026-11-27")]
    assert meta["rows_labeled"] == 9          # the count before the dedupe is unchanged
    # dedupe=False is every labeled scan row, as before
    assert len(sp.load_shadow_data(horizon_days=10, min_records=1, dedupe=False)[0]) == 9
    # and the off switch gives back the prior deduped rows
    monkeypatch.setenv(sp.SESSION_DAYS_ONLY_ENV, "0")
    assert len(sp.load_shadow_data(horizon_days=10, min_records=1)[0]) == 9


def test_counts_report_what_the_rule_removes(tmp_log, monkeypatch):
    c = sp.session_day_counts(WEEKEND_LOG + ["not a record"])
    assert c == {"rule_on": True, "records": 9, "non_session_records": 5,
                 "ticker_days": 9, "non_session_ticker_days": 5}
    monkeypatch.setenv(sp.SESSION_DAYS_ONLY_ENV, "0")
    assert sp.session_day_counts(WEEKEND_LOG)["rule_on"] is False   # counted either way
    assert sp.session_day_counts(WEEKEND_LOG)["non_session_records"] == 5


def test_stats_gain_one_key_and_keep_the_rest(tmp_log, monkeypatch):
    _write(sp.SHADOW_LOG_PATH, WEEKEND_LOG)
    stats = sp.get_shadow_stats()
    assert stats["session_days"]["non_session_records"] == 5
    assert stats["session_days"]["non_session_ticker_days"] == 5
    # raw counts are every record, as before: the rule is in the dedupe only
    assert stats["total_records"] == 9
    assert stats["by_decision"] == {"rejected_score": 9}
    assert stats["labeled_by_horizon"]["+10d"] == {"wins": 9, "losses": 0, "unknown": 0}
    monkeypatch.setenv(sp.SESSION_DAYS_ONLY_ENV, "0")
    sp._shadow_log_cache_reset()
    off = sp.get_shadow_stats()
    assert {k: v for k, v in off.items() if k != "session_days"} == \
           {k: v for k, v in stats.items() if k != "session_days"}
