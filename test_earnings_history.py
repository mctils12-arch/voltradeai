#!/usr/bin/env python3
"""
EARNINGS HISTORY (2026-11-06, earnings_history.py; human refocus spec §3 step 5
"earnings history (beats last 3 quarters)", §7 "log every signal"). LOGGING ONLY.

Pins:
- the Tier-3 fill against a mocked Finnhub, through finnhub_data's real
  rate-limited GET (token bucket) and this job's own spacing: only names
  deep_score asked about in the last 24 h, never-fetched first, the per-run cap
  and time budget, 401/403/429 stop the run, a failed refresh keeps the last good
  entry, no key = one status change and the reason no_api_key;
- deep_score reads ONLY the cache (every network path patched to raise), logs
  the block on the shadow record and the candidate, and its score / side /
  sub-scores / features are byte-identical with the feature off;
- null handling: a missing value is None with a reason, never a zero;
- the shadow record block stays inside the 1000-byte snapshot budget;
- the decision journal's selection stage text, and its parity with
  server/decisionJournal.ts;
- the attribution table lists the earnings row: logged, not used by the score,
  untested until labels accrue (its forward test, source_validation's
  logged-only source, is pinned in test_source_validation_logged.py);
- config parity with system_config, the env off switch, the daemon route.

Run: bash pytest_locked.sh -q test_earnings_history.py
"""
import inspect
import json
import os
import re
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import requests

import earnings_history as eh
import finnhub_data
import shadow_portfolio as sp

_ROOT = os.path.dirname(os.path.abspath(__file__))
NOW = datetime(2026, 11, 6, 15, 0, tzinfo=timezone.utc)
CFG = eh.config(eh.DEFAULTS)

# Finnhub /stock/earnings rows (shuffled; plus a quarter that has not ended yet)
AAA_ROWS = [
    {"actual": 1.10, "estimate": 1.15, "period": "2026-06-30", "quarter": 2, "year": 2026, "symbol": "AAA"},
    {"actual": 1.30, "estimate": 1.20, "period": "2026-09-30", "quarter": 3, "year": 2026, "symbol": "AAA"},
    {"actual": None, "estimate": 1.40, "period": "2026-12-31", "quarter": 4, "year": 2026, "symbol": "AAA"},
    {"actual": 0.90, "estimate": 0.95, "period": "2025-12-31", "quarter": 4, "year": 2025, "symbol": "AAA"},
    {"actual": 1.05, "estimate": 1.00, "period": "2026-03-31", "quarter": 1, "year": 2026, "symbol": "AAA"},
]


def iso(dt):
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def entry(status="ok", hours_old=1.0, quarters=None, **extra):
    q = quarters if quarters is not None else eh.parse_quarters(AAA_ROWS, NOW.date())
    return {"fetched_at": iso(NOW - timedelta(hours=hours_old)), "status": status, "quarters": q, **extra}


class _Resp:
    def __init__(self, status, data):
        self.status_code, self._data = status, data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(response=self)

    def json(self):
        return self._data


class _TmpData(unittest.TestCase):
    """DATA_DIR -> a temp dir; the env switch unset; a Finnhub key present."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        for p in (patch.object(eh, "DATA_DIR", tmp.name), patch.object(finnhub_data, "FINNHUB_KEY", "test-key"),
                  patch.dict(os.environ, {}, clear=False)):
            p.start()
            self.addCleanup(p.stop)
        os.environ.pop(eh.ENV_VAR, None)

    def write_cache(self, entries, last_run=None):
        with open(eh.cache_path(), "w") as fh:
            json.dump({"v": 1, "entries": entries, "last_run": last_run}, fh)

    def write_wanted(self, names):
        with open(eh.wanted_path(), "w") as fh:
            json.dump({"v": 1, "names": names}, fh)

    def cache(self):
        with open(eh.cache_path()) as fh:
            return json.load(fh)


# ── the fill (Tier 3) ─────────────────────────────────────────────────────────
class TestFill(_TmpData):
    def fake_get(self, url, params=None, headers=None, timeout=None, **kw):
        self.assertIn("finnhub.io/api/v1/stock/earnings", url)
        self.assertEqual((params["limit"], params["token"]), (4, "test-key"))
        sym = params["symbol"]
        self.calls.append(sym)
        return {"AAA": _Resp(200, AAA_ROWS), "ETF1": _Resp(200, []), "ERR1": _Resp(502, {}),
                "LIM1": _Resp(429, {})}.get(sym, _Resp(200, []))

    def setUp(self):
        super().setUp()
        self.calls, self.sleeps = [], []
        p = patch.object(finnhub_data.requests, "get", side_effect=self.fake_get)
        p.start()
        self.addCleanup(p.stop)

    def run_fill(self, **kw):
        return eh.refresh_wanted(now=NOW, sleep=self.sleeps.append, **kw)

    def test_fetches_the_days_wanted_names_through_the_rate_limited_get(self):
        self.write_wanted({"AAA": iso(NOW - timedelta(hours=2)), "ETF1": iso(NOW - timedelta(hours=1)),
                           "ERR1": iso(NOW - timedelta(minutes=5)),
                           "OLD": iso(NOW - timedelta(hours=30)),          # asked about > 24 h ago: not today's
                           "FRESH": iso(NOW - timedelta(minutes=1))})      # cached 1 h ago: not due
        self.write_cache({"FRESH": entry(hours_old=1.0)})
        with patch.object(finnhub_data._bucket, "acquire", wraps=finnhub_data._bucket.acquire) as bucket:
            out = self.run_fill()
        # never-fetched names, most recently asked-about first; OLD and FRESH are left alone
        self.assertEqual(self.calls, ["ERR1", "ETF1", "AAA"])
        self.assertEqual(bucket.call_count, 3, "every call goes through finnhub_data's token bucket")
        self.assertEqual(self.sleeps, [2.0, 2.0], "this job's own spacing between calls (<= 30/min)")
        self.assertEqual({k: out[k] for k in ("status", "wanted", "due", "attempted", "ok", "empty", "failed", "left", "stopped")},
                         {"status": "done", "wanted": 4, "due": 3, "attempted": 3, "ok": 1, "empty": 1, "failed": 1,
                          "left": 0, "stopped": None})
        self.assertIn("ERR1: HTTP 502", out["first_error"])
        ents = self.cache()["entries"]
        self.assertEqual(ents["AAA"]["status"], "ok")
        self.assertEqual([q["period"] for q in ents["AAA"]["quarters"]],
                         ["2026-09-30", "2026-06-30", "2026-03-31", "2025-12-31"], "newest first; the unended quarter dropped")
        self.assertEqual(ents["ETF1"], {"fetched_at": iso(NOW), "status": "empty", "quarters": []})
        self.assertEqual((ents["ERR1"]["status"], ents["ERR1"]["quarters"]), ("error", []))
        self.assertNotIn("OLD", ents)
        self.assertEqual(self.cache()["last_run"], {"at": iso(NOW), "status": "done"})

    def test_a_429_stops_the_run_and_writes_nothing_for_that_name(self):
        self.write_wanted({"AAA": iso(NOW - timedelta(minutes=3)), "LIM1": iso(NOW - timedelta(minutes=2)),
                           "ETF1": iso(NOW - timedelta(minutes=1))})
        out = self.run_fill()
        self.assertEqual(self.calls, ["ETF1", "LIM1"])
        self.assertEqual((out["stopped"], out["ok"] + out["empty"], out["left"]), ("Finnhub answered HTTP 429", 1, 2))
        self.assertNotIn("LIM1", self.cache()["entries"])
        self.assertNotIn("AAA", self.cache()["entries"])

    def test_a_failed_refresh_keeps_the_last_good_entry(self):
        self.write_wanted({"ERR1": iso(NOW - timedelta(minutes=1))})
        self.write_cache({"ERR1": entry(hours_old=13.0)})       # older than ttl_h: due
        out = self.run_fill()
        self.assertEqual((out["failed"], self.calls), (1, ["ERR1"]))
        kept = self.cache()["entries"]["ERR1"]
        self.assertEqual((kept["status"], len(kept["quarters"]), kept["fetched_at"]), ("ok", 4, iso(NOW - timedelta(hours=13))))
        self.assertIn("HTTP 502", kept["last_error"])
        blk = eh.read_for_candidate("ERR1", now=NOW, mark=False)
        self.assertEqual((blk["beats_last_3"], blk["cache_age_h"], blk["reason"]), (2, 13.0, None))

    def test_the_per_run_cap_and_the_time_budget(self):
        self.write_wanted({f"N{i:02d}": iso(NOW - timedelta(minutes=i)) for i in range(20)})
        out = self.run_fill()
        self.assertEqual((out["attempted"], out["left"], out["due"]), (15, 5, 20))
        self.assertEqual(self.calls[:2], ["N00", "N01"])
        # a slow Finnhub: each budget check reads the clock 15 s later (t0 = 0; 15, 30 <= 40; 45 > 40)
        # -> two calls, then the run stops
        self.calls.clear()
        os.remove(eh.cache_path())
        ticks = iter(range(0, 1000, 15))
        out = eh.refresh_wanted(now=NOW, sleep=self.sleeps.append, clock=lambda: float(next(ticks)))
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(out["stopped"], "run budget of 40s reached")

    def test_no_key_is_one_status_change_and_the_reason_on_the_record(self):
        self.write_wanted({"AAA": iso(NOW - timedelta(minutes=1))})
        with patch.object(finnhub_data, "FINNHUB_KEY", ""):
            first, second = self.run_fill(), self.run_fill()
        self.assertEqual((first["status"], first["status_changed"]), ("no_key", True))
        self.assertEqual((second["status"], second["status_changed"]), ("no_key", False))
        self.assertEqual(self.calls, [])
        blk = eh.read_for_candidate("AAA", now=NOW, mark=False)
        self.assertEqual((blk["reason"], blk["beats_last_3"], blk["quarters_seen"]), ("no_api_key", None, None))

    def test_off_and_idle_do_nothing(self):
        self.write_wanted({"AAA": iso(NOW - timedelta(minutes=1))})
        with patch.dict(os.environ, {eh.ENV_VAR: "off"}):
            self.assertEqual(self.run_fill()["status"], "off")
        self.assertFalse(os.path.exists(eh.cache_path()), "off writes nothing")
        self.write_cache({"AAA": entry(hours_old=1.0)}, last_run={"at": iso(NOW), "status": "idle"})
        out = self.run_fill()
        self.assertEqual((out["status"], out["status_changed"], self.calls), ("idle", False, []))

    def test_a_raising_fetch_is_recorded_not_raised(self):
        self.write_wanted({"AAA": iso(NOW - timedelta(minutes=1))})

        def boom(sym):
            raise ValueError("Expecting value: line 1 column 1 (char 0)")
        out = self.run_fill(fetch=boom)
        self.assertEqual((out["status"], out["failed"]), ("done", 1))
        self.assertEqual(self.cache()["entries"]["AAA"]["status"], "error")

    def test_finnhub_raw_fetch_is_uncached_and_not_zero_filled(self):
        with patch.object(finnhub_data, "_finnhub_get", return_value=[{"actual": None, "estimate": 1.0}]) as g:
            self.assertEqual(finnhub_data.get_earnings_history_raw("aaa"), [{"actual": None, "estimate": 1.0}])
        g.assert_called_once_with("/stock/earnings", {"symbol": "AAA", "limit": 4})
        self.assertNotIn("_cache_set", inspect.getsource(finnhub_data.get_earnings_history_raw))


# ── the read (deep_score) ────────────────────────────────────────────────────
def _no_network(*a, **k):
    raise AssertionError("the earnings read path must not touch the network")


class TestReadOnlyCache(_TmpData):
    def test_reads_the_cache_and_never_the_network(self):
        self.write_cache({"AAA": entry(hours_old=3.0)})
        with patch.object(requests, "get", side_effect=_no_network), \
                patch.object(requests.Session, "request", side_effect=_no_network), \
                patch.object(finnhub_data, "_finnhub_get", side_effect=_no_network), \
                patch.object(finnhub_data, "get_earnings_history_raw", side_effect=_no_network):
            blk = eh.read_for_candidate("aaa", now=NOW)
        self.assertEqual(blk, {"source": "finnhub", "beats_last_3": 2, "surprise_pct_last_3": [8.33, -4.35, 5.0],
                               "quarters_seen": 3, "latest_period": "2026-09-30", "cache_age_h": 3.0, "reason": None})
        self.assertFalse(os.path.exists(eh.wanted_path()), "a fresh entry needs no fetch: nothing written")
        src = inspect.getsource(eh.read_for_candidate) + inspect.getsource(eh.build_block)
        self.assertNotIn("finnhub_data", src)
        self.assertNotIn("requests", src)

    def test_a_miss_or_a_due_entry_goes_on_the_want_list(self):
        self.write_cache({"OLDE": entry(hours_old=13.0)})
        self.assertEqual(eh.read_for_candidate("NEWB", now=NOW)["reason"], "not_cached")
        self.assertEqual(eh.read_for_candidate("OLDE", now=NOW)["beats_last_3"], 2)   # still logged, refresh due
        with open(eh.wanted_path()) as fh:
            names = json.load(fh)["names"]
        self.assertEqual(set(names), {"NEWB", "OLDE"})
        self.assertFalse(eh.mark_wanted("NEWB", now=NOW + timedelta(minutes=5)), "re-stamped at most every 15 min")
        self.assertTrue(eh.mark_wanted("NEWB", now=NOW + timedelta(minutes=16)))
        eh.mark_wanted("LATE", now=NOW + timedelta(hours=25))
        with open(eh.wanted_path()) as fh:
            self.assertEqual(set(json.load(fh)["names"]), {"LATE"}, "names asked about > 24 h ago are pruned")

    def test_off_reads_nothing(self):
        self.write_cache({"AAA": entry()})
        with patch.dict(os.environ, {eh.ENV_VAR: "0"}):
            self.assertIsNone(eh.read_for_candidate("AAA", now=NOW))
        self.assertFalse(os.path.exists(eh.wanted_path()))

    def test_an_unreadable_cache_is_a_reason_not_an_exception(self):
        with open(eh.cache_path(), "w") as fh:
            fh.write("{not json")
        self.assertEqual(eh.read_for_candidate("AAA", now=NOW)["reason"], "cache_unreadable")


class TestNullHandling(unittest.TestCase):
    def block(self, e, **kw):
        return eh.build_block(e, NOW, CFG, **kw)

    def assertNoValues(self, blk):
        for k in ("beats_last_3", "surprise_pct_last_3", "quarters_seen", "latest_period"):
            self.assertIsNone(blk[k], k)

    def test_missing_is_none_with_a_reason_never_zero(self):
        blk, due = self.block(None)
        self.assertNoValues(blk)
        self.assertEqual((blk["reason"], blk["cache_age_h"], due), ("not_cached", None, True))
        self.assertEqual(self.block(None, run_status="no_key")[0]["reason"], "no_api_key")
        self.assertEqual(self.block(entry(), read_error="ValueError: x")[0]["reason"], "cache_unreadable")
        self.assertEqual(self.block({"status": "ok"})[0]["reason"], "not_cached")      # no fetch time

    def test_a_failed_fetch(self):
        blk, due = self.block(entry(status="error", quarters=[], error="HTTP 500 from Finnhub endpoint /stock/earnings"))
        self.assertNoValues(blk)
        self.assertEqual((blk["reason"], blk["error"], due), ("fetch_error", "HTTP 500 from Finnhub endpoint /stock/earnings", True))

    def test_no_reports_is_a_real_zero_count_of_quarters(self):
        blk, due = self.block(entry(status="empty", quarters=[]))
        self.assertEqual((blk["quarters_seen"], blk["beats_last_3"], blk["reason"], due), (0, None, "no_reports", False))

    def test_fewer_than_three_quarters(self):
        blk, _ = self.block(entry(quarters=eh.parse_quarters(AAA_ROWS[:2], NOW.date())))
        self.assertEqual((blk["beats_last_3"], blk["quarters_seen"], blk["surprise_pct_last_3"], blk["reason"]),
                         (None, 2, [8.33, -4.35], "fewer_than_3"))

    def test_a_missing_figure_is_incomplete_and_a_zero_estimate_has_no_percentage(self):
        q = [{"period": "2026-09-30", "actual": 1.3, "estimate": None},
             {"period": "2026-06-30", "actual": 1.0, "estimate": 1.0},
             {"period": "2026-03-31", "actual": 0.9, "estimate": 1.0}]
        blk, _ = self.block(entry(quarters=q))
        self.assertEqual((blk["beats_last_3"], blk["surprise_pct_last_3"], blk["reason"]), (None, [None, 0.0, -10.0], "incomplete"))
        q[0]["estimate"] = 0.0           # a $0.00 estimate: the beat is known, the percentage is not
        blk, _ = self.block(entry(quarters=q))
        self.assertEqual((blk["beats_last_3"], blk["surprise_pct_last_3"], blk["reason"]), (1, [None, 0.0, -10.0], None))

    def test_in_line_is_not_a_beat_and_losses_compare_correctly(self):
        self.assertIs(eh.beat({"actual": 1.0, "estimate": 1.0}), False)
        self.assertIs(eh.beat({"actual": -0.40, "estimate": -0.50}), True)      # a smaller loss than expected
        self.assertEqual(eh.surprise_pct({"actual": -0.40, "estimate": -0.50}), 20.0)
        self.assertIsNone(eh.beat({"actual": True, "estimate": 1.0}), "a bool is not a number")

    def test_stale_entries_are_not_logged_as_current(self):
        blk, due = self.block(entry(hours_old=49.0))
        self.assertNoValues(blk)
        self.assertEqual((blk["reason"], blk["cache_age_h"], due), ("stale", 49.0, True))
        self.assertEqual(self.block(entry(hours_old=47.0))[0]["beats_last_3"], 2)

    def test_parse_quarters_keeps_nulls_and_drops_what_cannot_be_ordered(self):
        rows = AAA_ROWS + [{"actual": 2.0, "estimate": 1.0, "period": "2026-09-30"},   # duplicate period
                           {"actual": 1.0, "estimate": 1.0, "period": "n/a"}, "junk"]
        q = eh.parse_quarters(rows, NOW.date())
        self.assertEqual([x["period"] for x in q], ["2026-09-30", "2026-06-30", "2026-03-31", "2025-12-31"])
        self.assertEqual(q[0], {"period": "2026-09-30", "actual": 1.3, "estimate": 1.2})
        self.assertEqual(eh.parse_quarters([{"actual": None, "estimate": None, "period": "2026-09-30"}], NOW.date()),
                         [{"period": "2026-09-30", "actual": None, "estimate": None}])
        self.assertEqual(eh.parse_quarters({"error": "x"}, NOW.date()), [])

    def test_due_order_never_fetched_then_failed_then_old(self):
        wanted = {"OK1": iso(NOW - timedelta(minutes=1)), "STALE": iso(NOW - timedelta(minutes=2)),
                  "ERR": iso(NOW - timedelta(minutes=3)), "NEW2": iso(NOW - timedelta(minutes=9)),
                  "NEW1": iso(NOW - timedelta(minutes=4)), "GONE": iso(NOW - timedelta(hours=25))}
        entries = {"OK1": entry(hours_old=2), "STALE": entry(hours_old=12.5), "ERR": entry(status="error", quarters=[])}
        self.assertEqual(eh.due_names(wanted, entries, NOW, CFG), ["NEW1", "NEW2", "ERR", "STALE"])


# ── the shadow record ────────────────────────────────────────────────────────
class TestShadowRecord(unittest.TestCase):
    def test_compact_inside_the_snapshot_budget(self):
        blk, _ = eh.build_block(entry(), NOW, CFG)
        rec = sp._build_record("AAA", {"rsi_14": 55.0}, 70.0, "taken", "ok", 100.0, earnings_history=blk)
        self.assertEqual(rec["earnings_history"], blk)
        self.assertLess(len(json.dumps(rec["earnings_history"])), 300)
        worst = {"source": "s" * 999, "beats_last_3": 3, "surprise_pct_last_3": [-123456.789] * 9, "quarters_seen": 3,
                 "latest_period": "2026-09-30T00:00:00", "cache_age_h": 99999.99, "reason": "r" * 999, "error": "e" * 999}
        self.assertLess(len(json.dumps(sp._sanitize_earnings_history(worst))), 1000)
        self.assertEqual(sp.EARNINGS_HISTORY_MAX_BYTES, 1000)
        self.assertLess(len(json.dumps(sp._sanitize_earnings_history(worst))), sp.EARNINGS_HISTORY_MAX_BYTES)

    def test_junk_is_none_and_off_is_absent(self):
        junk = {"beats_last_3": True, "surprise_pct_last_3": [float("nan"), "5", 1.234567], "quarters_seen": float("inf"),
                "cache_age_h": "3", "reason": "not_cached"}
        s = sp._sanitize_earnings_history(junk)
        self.assertEqual((s["beats_last_3"], s["surprise_pct_last_3"], s["quarters_seen"], s["cache_age_h"]),
                         (None, [None, None, 1.23], None, None))
        rec = sp._build_record("AAA", {"rsi_14": 55.0}, 70.0, "taken", "ok", 100.0)
        self.assertNotIn("earnings_history", rec)
        self.assertNotIn("earnings_history", sp._build_record("AAA", {}, 70.0, "taken", earnings_history={}))
        self.assertNotIn("beats_last_3", rec["features"], "never inside the ML feature vector")


# ── deep_score end to end ────────────────────────────────────────────────────
from test_serve_feature_fix_deep_score import _DeepScoreHarness, _fake_get  # noqa: E402


class TestDeepScore(_DeepScoreHarness):
    def setUp(self):
        super().setUp()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        with open(os.path.join(tmp.name, eh.CACHE_FILENAME), "w") as fh:
            json.dump({"v": 1, "entries": {"QQAA": {**entry(), "fetched_at": iso(datetime.now(timezone.utc) - timedelta(hours=2))}}}, fh)
        self.tmp = tmp.name

        def _capture(**kw):
            self.records.append(sp._build_record(
                kw["ticker"], kw["features"], kw["score"], kw["decision"], kw.get("decision_reason", ""),
                kw.get("entry_price", 0.0), kw.get("vxx_ratio", 1.0), kw.get("regime_label", "neutral"),
                kw.get("side", ""), kw.get("action_label", ""), kw.get("trade_type", ""), kw.get("ml"),
                kw.get("score_breakdown"), kw.get("regime_snapshot"), kw.get("shadow_entry"), kw.get("serve_fix"),
                kw.get("earnings_history")))

        def _guard(url, *a, **k):
            if "finnhub" in str(url):
                raise AssertionError(f"deep_score called Finnhub: {url}")
            return _fake_get(url, *a, **k)

        for p in (patch.object(eh, "DATA_DIR", tmp.name),
                  patch("shadow_portfolio.log_candidate", side_effect=_capture),
                  patch("bot_engine.requests.get", side_effect=_guard),
                  patch.object(finnhub_data, "_finnhub_get", side_effect=_no_network),
                  patch.object(finnhub_data, "get_earnings_history_raw", side_effect=_no_network)):
            p.start()
            self.addCleanup(p.stop)
        os.environ.pop(eh.ENV_VAR, None)

    def test_the_block_rides_the_record_and_the_candidate_and_changes_nothing_scored(self):
        res, rec = self._run("off")
        want = {"source": "finnhub", "beats_last_3": 2, "surprise_pct_last_3": [8.33, -4.35, 5.0], "quarters_seen": 3,
                "latest_period": "2026-09-30", "cache_age_h": 2.0, "reason": None}
        self.assertEqual(res["earnings_history"], want)
        self.assertEqual(rec["earnings_history"], want)
        self.assertNotIn("beats_last_3", res["entry_features"])
        with patch.dict(os.environ, {eh.ENV_VAR: "off"}):
            res_off, rec_off = self._run("off")
        self.assertIsNone(res_off["earnings_history"])
        self.assertNotIn("earnings_history", rec_off)
        for k in ("deep_score", "rules_only_score", "ml_only_score", "score_legacy", "side", "trade_type", "action_label",
                  "reasons", "score_breakdown", "entry_features"):
            self.assertEqual(res[k], res_off[k], k)
        for k in ("score", "features", "ml", "score_breakdown", "decision", "side"):
            self.assertEqual(rec[k], rec_off[k], k)

    def test_an_uncached_candidate_is_logged_as_missing_and_wanted(self):
        res, rec = self._run("off", sym="QQAB")
        self.assertEqual((rec["earnings_history"]["reason"], rec["earnings_history"]["beats_last_3"]), ("not_cached", None))
        with open(os.path.join(self.tmp, eh.WANTED_FILENAME)) as fh:
            self.assertIn("QQAB", json.load(fh)["names"])

    def test_the_scan_trade_dict_carries_it(self):
        src = open(os.path.join(_ROOT, "bot_engine.py"), encoding="utf-8").read()
        scan = src[src.index("def _scan_market_inner("):]
        self.assertIn('"earnings_history": stock.get("earnings_history"),', scan)
        ds = src[src.index("def deep_score("):src.index("def check_sector_correlation(")]
        # read after the side is final, never before the score
        self.assertLess(ds.index('action_label = "SELL"'), ds.index("_earnings_hist = _eh_mod.read_for_candidate(ticker)"))
        self.assertIn("earnings_history=_earnings_hist,", ds)
        self.assertIn('"earnings_history": _earnings_hist,', ds)


# ── the decision journal ─────────────────────────────────────────────────────
class TestJournal(unittest.TestCase):
    def setUp(self):
        import decision_journal as dj
        self.dj = dj

    def cand(self, blk):
        return {"ticker": "AAA", "deep_score": 70.0, "score_breakdown": {"momentum": 80.0}, "reasons": ["Momentum strong"],
                "earnings_history": blk}

    def test_selection_stage_says_it_in_plain_english(self):
        blk = {"source": "finnhub", "beats_last_3": 3, "surprise_pct_last_3": [12.35, 4.1, 0.52], "quarters_seen": 3,
               "latest_period": "2026-09-30", "cache_age_h": 2.5, "reason": None}
        ev = self.dj.selection_stage(self.cand(blk), threshold=63)["earnings"]
        self.assertEqual(ev["text"], "beat estimates 3 of the last 3 quarters (EPS surprise +12.35%, +4.10%, +0.52%, "
                                     "newest first); logged, not used in the score")
        self.assertEqual((ev["beats_last_3"], ev["quarters_seen"], ev["cache_age_h"]), (3, 3, 2.5))

    def test_why_there_is_no_count(self):
        v = self.dj.earnings_view
        self.assertEqual(v(eh.empty_block("not_cached"))["text"],
                         "earnings history unavailable: it has not been fetched yet (the hourly Finnhub refresh adds it); "
                         "logged, not used in the score")
        self.assertEqual(v({**eh.empty_block("stale"), "cache_age_h": 52.7})["text"],
                         "earnings history unavailable: the cached copy is 52 hours old, too old to use; logged, not used in the score")
        self.assertEqual(v({**eh.empty_block("fewer_than_3"), "quarters_seen": 2, "surprise_pct_last_3": [5.0, -1.25]})["text"],
                         "earnings history incomplete: fewer than 3 quarters are on record (EPS surprise +5.00%, -1.25%, "
                         "newest first); logged, not used in the score")
        self.assertEqual(v({**eh.empty_block("no_reports"), "quarters_seen": 0})["text"],
                         "earnings history unavailable: Finnhub has no earnings reports for it; logged, not used in the score")
        self.assertIsNone(v(None))
        self.assertNotIn("earnings", self.dj.selection_stage(self.cand(None)), "no block, no key: older rows keep their shape")

    def test_every_reason_has_words_and_the_node_twin_says_the_same(self):
        self.assertEqual(set(self.dj.EARNINGS_REASON_TEXT), set(eh.REASONS))
        ts = open(os.path.join(_ROOT, "server", "decisionJournal.ts"), encoding="utf-8").read()
        block = ts[ts.index("export const EARNINGS_REASON_TEXT"):ts.index("function signedPct")]
        self.assertEqual(dict(re.findall(r'\n  (\w+): "([^"]+)",', block)), self.dj.EARNINGS_REASON_TEXT)

    def test_a_scan_skip_row_carries_it(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        with patch.object(self.dj, "DATA_DIR", tmp.name), patch.dict(os.environ, {}, clear=False):
            os.environ.pop(self.dj.ENV_VAR, None)
            self.dj._reset_dedupe()
            blk, _ = eh.build_block(entry(), NOW, CFG)
            n = self.dj.record_scan_skips([{"ticker": "AAA", "decision": "rejected_score", "reason": "score 61 < 63"}],
                                          [self.cand(blk)], threshold=63)
            rows = self.dj.read_rows()
            self.dj._reset_dedupe()
        self.assertEqual(n, 1)
        self.assertTrue(rows[0]["stages"]["selection"]["earnings"]["text"].startswith("beat estimates 2 of the last 3 quarters"))


# ── the attribution table ─────────────────────────────────────────────────────
class TestAttributionRow(unittest.TestCase):
    def test_listed_untested_and_not_used_by_the_score(self):
        import signal_attribution as sa
        from test_source_validation import make_records
        row = next(s for s in sa.SIGNAL_REGISTRY if s["id"] == "earnings_beats")
        self.assertEqual(row["role"], "logged on every deep-scored candidate since 2026-11-06 "
                                      "(Finnhub EPS actual vs estimate); not used by the score")
        self.assertEqual(sa.action_for("earnings_beats", None), ("not_used", "Not used by the score (logged only, since 2026-11-06)."))
        table = sa.build_attribution(make_records(), walkforward=None, cfg=False, now=NOW)
        out = next(r for r in table["signals"] if r["id"] == "earnings_beats")
        self.assertEqual((out["verdict"], out["action"], out["review"]), ("untested", "not_used", False))
        self.assertIn("earnings_beats", table["by_verdict"]["untested"])


# ── config, switch, daemon ───────────────────────────────────────────────────
class TestConfig(unittest.TestCase):
    def test_system_config_matches_the_module_defaults(self):
        from system_config import BASE_CONFIG
        live = BASE_CONFIG["EARNINGS_HISTORY"]
        self.assertEqual(set(live), set(eh.DEFAULTS))
        for k, v in eh.DEFAULTS.items():
            self.assertEqual(live[k], v, k)
        self.assertEqual(eh.config(), eh.config(eh.DEFAULTS))
        self.assertTrue(live["enabled"], "new features ship ON")

    def test_bounds_keep_the_job_inside_the_tier3_kill(self):
        c = eh.config({"run_budget_s": 500, "call_interval_s": 0.0, "ttl_h": 24, "max_age_h": 1, "max_fetch_per_run": -3})
        self.assertEqual((c["run_budget_s"], c["call_interval_s"], c["max_age_h"], c["max_fetch_per_run"]), (45.0, 1.0, 24.0, 0))
        self.assertEqual(eh.config({"ttl_h": "junk"})["ttl_h"], 12.0)

    def test_env_wins_whenever_present(self):
        self.assertFalse(eh.enabled(env={eh.ENV_VAR: "off"}, cfg={"enabled": True}))
        self.assertTrue(eh.enabled(env={eh.ENV_VAR: "1"}, cfg={"enabled": False}))
        self.assertFalse(eh.enabled(env={}, cfg={"enabled": False}))
        self.assertTrue(eh.enabled(env={}, cfg={}))

    def test_daemon_route_and_files_on_the_volume(self):
        import voltrade_daemon
        self.assertEqual(voltrade_daemon.RPCDispatcher()._routes["earnings_history_refresh"],
                         ("earnings_history", "refresh_wanted"))
        import storage_config
        self.assertTrue(os.path.join(storage_config.DATA_DIR, eh.CACHE_FILENAME).endswith("voltrade_earnings_history.json"))
        self.assertIn("from storage_config import DATA_DIR", open(os.path.join(_ROOT, "earnings_history.py")).read())


if __name__ == "__main__":
    unittest.main()
