#!/usr/bin/env python3
"""
Regression tests for shadow_portfolio.update_last_decision() and its two
call sites in bot_engine.py's scan_market() loop.

WHY THIS EXISTS (BUILD ORDER 4 #6, T-BOT half, 2026-07-05): log_candidate()
runs inside deep_score(), before scan_market()'s downstream per-candidate
filters (sector/correlation, spread) get a chance to reject a candidate
that already cleared MIN_SCORE. Before this fix, every such candidate was
permanently mislabeled "taken" in the shadow log even though it was never
traded — silently corrupting the exact RULE COST AUDIT questions
(open_questions.md) about correlation-block and spread-filter cost.
update_last_decision() corrects the label immediately after the real
rejection, without touching either filter's actual trading behavior.
"""
import importlib
import inspect
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

import shadow_portfolio


class TestUpdateLastDecision(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._orig_log_path = shadow_portfolio.SHADOW_LOG_PATH
        self._orig_lock_path = shadow_portfolio.SHADOW_LOCK_PATH
        shadow_portfolio.SHADOW_LOG_PATH = os.path.join(self._tmpdir.name, "shadow.json")
        shadow_portfolio.SHADOW_LOCK_PATH = shadow_portfolio.SHADOW_LOG_PATH + ".lock"

    def tearDown(self):
        shadow_portfolio.SHADOW_LOG_PATH = self._orig_log_path
        shadow_portfolio.SHADOW_LOCK_PATH = self._orig_lock_path
        self._tmpdir.cleanup()

    def _seed(self, ticker, decision="taken", age_seconds=1.0):
        ts = (datetime.now(timezone.utc) - timedelta(seconds=age_seconds)).isoformat()
        records = shadow_portfolio._load_shadow_log()
        records.append({
            "ticker": ticker, "timestamp": ts, "score": 70.0,
            "decision": decision, "decision_reason": "seed",
            "entry_price": 100.0, "vxx_ratio": 1.0, "regime_label": "NEUTRAL",
            "features": {}, "outcomes": {"+5d": None, "+10d": None, "+20d": None},
            "code_version": "test-shadow",
        })
        shadow_portfolio._save_shadow_log(records)

    def test_updates_fresh_taken_record(self):
        self._seed("AMD", decision="taken", age_seconds=1.0)
        ok = shadow_portfolio.update_last_decision("AMD", "rejected_heat", "sector block")
        self.assertTrue(ok)
        records = shadow_portfolio._load_shadow_log()
        self.assertEqual(records[-1]["decision"], "rejected_heat")
        self.assertEqual(records[-1]["decision_reason"], "sector block")

    def test_noop_when_already_resolved(self):
        self._seed("MSFT", decision="rejected_score", age_seconds=1.0)
        ok = shadow_portfolio.update_last_decision("MSFT", "rejected_heat", "sector block")
        self.assertFalse(ok)
        records = shadow_portfolio._load_shadow_log()
        self.assertEqual(records[-1]["decision"], "rejected_score")

    def test_noop_when_record_is_stale(self):
        self._seed("NVDA", decision="taken", age_seconds=300.0)
        ok = shadow_portfolio.update_last_decision(
            "NVDA", "rejected_other", "spread", max_age_seconds=120.0,
        )
        self.assertFalse(ok)
        records = shadow_portfolio._load_shadow_log()
        self.assertEqual(records[-1]["decision"], "taken")

    def test_noop_when_no_matching_ticker(self):
        self._seed("TSLA", decision="taken", age_seconds=1.0)
        ok = shadow_portfolio.update_last_decision("AAPL", "rejected_heat", "sector block")
        self.assertFalse(ok)

    def test_only_touches_most_recent_record_for_ticker(self):
        # An older AMD record already resolved must be untouched; only the
        # freshest "taken" record for the ticker may be corrected.
        self._seed("AMD", decision="rejected_score", age_seconds=90.0)
        self._seed("AMD", decision="taken", age_seconds=1.0)
        ok = shadow_portfolio.update_last_decision("AMD", "rejected_heat", "sector block")
        self.assertTrue(ok)
        records = shadow_portfolio._load_shadow_log()
        amd_records = [r for r in records if r["ticker"] == "AMD"]
        self.assertEqual(amd_records[0]["decision"], "rejected_score")
        self.assertEqual(amd_records[1]["decision"], "rejected_heat")

    def test_empty_ticker_is_noop(self):
        self.assertFalse(shadow_portfolio.update_last_decision("", "rejected_heat"))


class TestBotEngineCallSites(unittest.TestCase):
    """
    Source-inspection: confirms scan_market()'s correlation and spread
    rejection sites call update_last_decision() with the correct decision
    bucket, wrapped defensively so a shadow-logging failure can never
    break the trading loop. Mirrors test_voltrade_daemon.py's pattern of
    pinning wiring via source inspection rather than a live scan.
    """

    @classmethod
    def setUpClass(cls):
        import bot_engine
        # The rejection sites live in _scan_market_inner(), the closure
        # scan_market() dispatches to under its timeout guard.
        cls.source = inspect.getsource(bot_engine._scan_market_inner)

    def test_correlation_block_logs_rejected_heat(self):
        idx = self.source.find("check_sector_correlation(")
        self.assertNotEqual(idx, -1, "check_sector_correlation call site not found")
        window = self.source[idx:idx + 800]
        self.assertIn("update_last_decision", window)
        self.assertIn('"rejected_heat"', window)
        self.assertIn("except Exception", window)

    def test_spread_filter_logs_rejected_other(self):
        idx = self.source.find("_spread_pct > 0.005")
        self.assertNotEqual(idx, -1, "spread filter call site not found")
        window = self.source[idx:idx + 600]
        self.assertIn("update_last_decision", window)
        self.assertIn('"rejected_other"', window)
        self.assertIn("except Exception", window)


class TestBackfillOutcomesBatching(unittest.TestCase):
    """
    REPAIR 2026-08-07 (KNOWN BROKEN #10/#20 follow-up): backfill_outcomes()
    used to fetch Alpaca bars with ONE HTTP CALL PER UNIQUE TICKER even
    though _fetch_historical_bars_batch supports up to BATCH_SIZE tickers
    per call — ticker count, not the token bucket, was the real throughput
    ceiling. With ~13k archived records processed oldest-first, that pinned
    decision buckets that started existing later in the archive's timeline
    (e.g. rejected_masterkill, first logged 2026-07-11) to being unreached
    for months at the old max_records=500/night rate. These tests pin the
    fix: tickers sharing the same day+horizon window collapse into ONE
    _fetch_historical_bars_batch call, distinct windows stay separate calls,
    and the resulting labels are unchanged from what the per-ticker code
    would have computed from the same bars.
    """

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._orig_log_path = shadow_portfolio.SHADOW_LOG_PATH
        self._orig_lock_path = shadow_portfolio.SHADOW_LOCK_PATH
        shadow_portfolio.SHADOW_LOG_PATH = os.path.join(self._tmpdir.name, "shadow.json")
        shadow_portfolio.SHADOW_LOCK_PATH = shadow_portfolio.SHADOW_LOG_PATH + ".lock"
        self._orig_fetch = shadow_portfolio._fetch_historical_bars_batch
        self._calls = []

    def tearDown(self):
        shadow_portfolio.SHADOW_LOG_PATH = self._orig_log_path
        shadow_portfolio.SHADOW_LOCK_PATH = self._orig_lock_path
        shadow_portfolio._fetch_historical_bars_batch = self._orig_fetch
        self._tmpdir.cleanup()

    def _seed(self, ticker, entry_price, age_days, decision="rejected_masterkill"):
        ts = (datetime.now(timezone.utc) - timedelta(days=age_days)).isoformat()
        records = shadow_portfolio._load_shadow_log()
        records.append({
            "ticker": ticker, "timestamp": ts, "score": 70.0,
            "decision": decision, "decision_reason": "seed",
            "entry_price": entry_price, "vxx_ratio": 1.0, "regime_label": "NEUTRAL",
            "features": {}, "outcomes": {"+5d": None, "+10d": None, "+20d": None},
            "code_version": "test-shadow",
        })
        shadow_portfolio._save_shadow_log(records)

    def _flat_bars(self, price, n=25, start_days_ago=59):
        bars = []
        start = datetime.now(timezone.utc) - timedelta(days=start_days_ago)
        for i in range(n):
            d = (start + timedelta(days=i)).strftime("%Y-%m-%dT00:00:00Z")
            bars.append({"t": d, "h": price * 1.001, "l": price * 0.999, "c": price})
        return bars

    def test_same_day_horizon_batches_far_fewer_calls_than_tickers(self):
        # 20 tickers logged the same day share the same 3 (one per horizon)
        # windows. The old per-ticker code made one HTTP call PER TICKER
        # (20 calls); the fix should make at most one call per distinct
        # (horizon) window — i.e. <= 3, not 20 — while still covering every
        # ticker across those calls.
        tickers = [f"T{i}" for i in range(20)]
        for t in tickers:
            self._seed(t, entry_price=100.0, age_days=60)

        def fake_fetch(tick_list, start, end):
            self._calls.append(sorted(tick_list))
            return {t: self._flat_bars(100.0) for t in tick_list}

        shadow_portfolio._fetch_historical_bars_batch = fake_fetch
        stats = shadow_portfolio.backfill_outcomes(max_records=200)

        self.assertLessEqual(len(self._calls), 3,
                              f"expected <=3 batched calls (one per horizon), got {len(self._calls)}: {self._calls}")
        self.assertLess(len(self._calls), len(tickers),
                         "batching must use fewer calls than the old one-call-per-ticker code")
        seen = set()
        for call in self._calls:
            seen.update(call)
        self.assertEqual(seen, set(tickers), "every ticker must still be fetched")
        self.assertGreater(stats["updated"], 0)

    def test_labels_unchanged_from_per_ticker_baseline(self):
        # WIN2 hits the +2% target on day 2 — every horizon should label it
        # a WIN with exit_reason "target", exactly as _label_from_path would
        # compute from the same bars regardless of how they were fetched.
        self._seed("WIN2", entry_price=100.0, age_days=60)

        def fake_fetch(tick_list, start, end):
            self._calls.append(sorted(tick_list))
            start_dt = datetime.now(timezone.utc) - timedelta(days=59)
            bars = [
                {"t": start_dt.strftime("%Y-%m-%dT00:00:00Z"), "h": 100.5, "l": 99.5, "c": 100.2},
                {"t": (start_dt + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z"), "h": 103.0, "l": 100.0, "c": 102.5},
            ]
            for i in range(2, 25):
                d = (start_dt + timedelta(days=i)).strftime("%Y-%m-%dT00:00:00Z")
                bars.append({"t": d, "h": 102.6, "l": 102.4, "c": 102.5})
            return {t: bars for t in tick_list}

        shadow_portfolio._fetch_historical_bars_batch = fake_fetch
        stats = shadow_portfolio.backfill_outcomes(max_records=100)
        self.assertGreater(stats["updated"], 0)

        records = shadow_portfolio._load_shadow_log()
        rec = next(r for r in records if r["ticker"] == "WIN2")
        for h in ("+5d", "+10d", "+20d"):
            self.assertEqual(rec["outcomes"][h]["label"], 1, f"{h} should be a WIN")
            self.assertEqual(rec["outcomes"][h]["exit_reason"], "target")

    def test_distinct_windows_stay_separate_calls(self):
        # OLD3 (90d old) is observable at all 3 horizons; YOUNG3 (40d old)
        # is only observable at +5d/+10d (needs 51d for +20d). Different
        # rec_time -> different (window_start, window_end) even for shared
        # horizons, so they must not collapse into a single fetch call that
        # would silently widen either record's intended window.
        self._seed("OLD3", entry_price=50.0, age_days=90)
        self._seed("YOUNG3", entry_price=50.0, age_days=40)

        def fake_fetch(tick_list, start, end):
            self._calls.append((tuple(sorted(tick_list)), start, end))
            return {t: self._flat_bars(50.0, start_days_ago=90) for t in tick_list}

        shadow_portfolio._fetch_historical_bars_batch = fake_fetch
        shadow_portfolio.backfill_outcomes(max_records=100)

        distinct_windows = set((c[1], c[2]) for c in self._calls)
        self.assertGreaterEqual(len(distinct_windows), 2,
                                 f"expected records at different ages to use different windows, got {self._calls}")
        seen_tickers = set()
        for tickers, _, _ in self._calls:
            seen_tickers.update(tickers)
        self.assertEqual(seen_tickers, {"OLD3", "YOUNG3"})


class TestBackfillPermanentFailure(unittest.TestCase):
    """
    REPAIR 2026-08-17 (KNOWN BROKEN #20 follow-up): a (record, horizon) pair
    with no available price data got re-queued and re-fetched on every
    nightly backfill run forever, since nothing was ever written to
    outcomes[horizon_key] on failure. Live evidence (/api/diag/shadow +
    /api/diag/audit?type=SHADOW-BACKFILL, 2026-08-16): missing_price=1469 of
    3002 processed jobs (49%) in one night, while rejected_masterkill /
    rejected_heat / rejected_other — the exact evidence KNOWN BROKEN #20
    needs — still showed zero labeled outcomes weeks after becoming old
    enough. These tests pin the fix: after MAX_BACKFILL_ATTEMPTS consecutive
    failed nights, the pair gets a terminal label=-1 sentinel and stops
    consuming nightly Alpaca-call budget, without ever fabricating a
    win/loss; a pair that recovers before hitting the cap is labeled
    normally, not prematurely retired.
    """

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._orig_log_path = shadow_portfolio.SHADOW_LOG_PATH
        self._orig_lock_path = shadow_portfolio.SHADOW_LOCK_PATH
        shadow_portfolio.SHADOW_LOG_PATH = os.path.join(self._tmpdir.name, "shadow.json")
        shadow_portfolio.SHADOW_LOCK_PATH = shadow_portfolio.SHADOW_LOG_PATH + ".lock"
        self._orig_fetch = shadow_portfolio._fetch_historical_bars_batch

    def tearDown(self):
        shadow_portfolio.SHADOW_LOG_PATH = self._orig_log_path
        shadow_portfolio.SHADOW_LOCK_PATH = self._orig_lock_path
        shadow_portfolio._fetch_historical_bars_batch = self._orig_fetch
        self._tmpdir.cleanup()

    def _seed(self, ticker, entry_price=50.0, age_days=90, decision="rejected_masterkill"):
        ts = (datetime.now(timezone.utc) - timedelta(days=age_days)).isoformat()
        records = shadow_portfolio._load_shadow_log()
        records.append({
            "ticker": ticker, "timestamp": ts, "score": 70.0,
            "decision": decision, "decision_reason": "seed",
            "entry_price": entry_price, "vxx_ratio": 1.0, "regime_label": "NEUTRAL",
            "features": {}, "outcomes": {"+5d": None, "+10d": None, "+20d": None},
            "code_version": "test-shadow",
        })
        shadow_portfolio._save_shadow_log(records)

    def _flat_bars(self, price, n=25, start_days_ago=89):
        bars = []
        start = datetime.now(timezone.utc) - timedelta(days=start_days_ago)
        for i in range(n):
            d = (start + timedelta(days=i)).strftime("%Y-%m-%dT00:00:00Z")
            bars.append({"t": d, "h": price * 1.001, "l": price * 0.999, "c": price})
        return bars

    def test_retired_after_max_attempts_and_stops_costing_calls(self):
        self._seed("DELIST1", entry_price=50.0, age_days=90)
        call_count = [0]

        def always_empty(tick_list, start, end):
            call_count[0] += 1
            return {}  # no bars for any ticker — simulates a dead symbol

        shadow_portfolio._fetch_historical_bars_batch = always_empty

        for i in range(shadow_portfolio.MAX_BACKFILL_ATTEMPTS):
            stats = shadow_portfolio.backfill_outcomes(max_records=100)
            self.assertGreater(stats["missing_price"], 0, f"run {i}: expected missing_price jobs")

        self.assertEqual(stats["permanently_unlabelable"], 3)  # +5d, +10d, +20d all retired this run
        records = shadow_portfolio._load_shadow_log()
        rec = next(r for r in records if r["ticker"] == "DELIST1")
        for h in ("+5d", "+10d", "+20d"):
            self.assertEqual(rec["outcomes"][h]["label"], -1)
            self.assertEqual(rec["outcomes"][h]["exit_reason"], "no_price_data")
        self.assertEqual(rec.get("_bf_attempts", {}), {},
                          "attempt counters must be cleared once a pair is retired")

        calls_before = call_count[0]
        stats2 = shadow_portfolio.backfill_outcomes(max_records=100)
        self.assertEqual(call_count[0], calls_before,
                          "a retired record must not trigger any further Alpaca fetch calls")
        self.assertEqual(stats2["missing_price"], 0)
        self.assertGreaterEqual(stats2["already_filled"], 3)

    def test_permanently_unlabelable_excluded_from_win_rate(self):
        self._seed("DELIST2", entry_price=50.0, age_days=90, decision="rejected_masterkill")
        # Also seed 5 "rejected_masterkill" records that DO label as wins, so
        # the decision bucket clears get_shadow_stats' own n>=5 reporting
        # floor and proves the retired record isn't silently counted in it
        # either direction.
        for i in range(5):
            self._seed(f"WIN{i}", entry_price=100.0, age_days=90, decision="rejected_masterkill")

        def fetch(tick_list, start, end):
            out = {}
            for t in tick_list:
                if t == "DELIST2":
                    continue  # no data for this one
                out[t] = self._flat_bars(103.0)  # +3% -> hits the +2% win target
            return out

        shadow_portfolio._fetch_historical_bars_batch = fetch

        for _ in range(shadow_portfolio.MAX_BACKFILL_ATTEMPTS):
            shadow_portfolio.backfill_outcomes(max_records=100)

        stats = shadow_portfolio.get_shadow_stats()
        by_decision = stats["win_rate_by_decision"].get("rejected_masterkill", {})
        self.assertIn("+5d", by_decision, "the 5 winning records should clear the n>=5 reporting floor")
        self.assertEqual(by_decision["+5d"]["n"], 5,
                          "the permanently-unlabelable record must not be counted as a win or a loss")
        self.assertEqual(by_decision["+5d"]["win_rate"], 100.0)

    def test_recovers_before_hitting_cap_is_not_retired(self):
        self._seed("FLAKY1", entry_price=50.0, age_days=90)
        attempt = [0]

        def flaky_then_recovers(tick_list, start, end):
            attempt[0] += 1
            if attempt[0] <= 1:  # fails only the very first call, well below MAX_BACKFILL_ATTEMPTS
                return {}
            return {t: self._flat_bars(50.0) for t in tick_list}

        shadow_portfolio._fetch_historical_bars_batch = flaky_then_recovers

        for _ in range(3):
            shadow_portfolio.backfill_outcomes(max_records=100)

        records = shadow_portfolio._load_shadow_log()
        rec = next(r for r in records if r["ticker"] == "FLAKY1")
        for h in ("+5d", "+10d", "+20d"):
            self.assertNotEqual(rec["outcomes"][h]["label"], -1,
                                 "a record that recovers before the attempt cap must get a real label")
            self.assertEqual(rec["outcomes"][h]["exit_reason"], "timeout")


class TestBackfillProgressByDecision(unittest.TestCase):
    """
    REPAIR 2026-08-19 (KNOWN BROKEN #20 follow-up): win_rate_by_decision's
    n>=5 reporting floor makes "not enough labeled data yet" and "this
    bucket's records are stuck" look identical from outside — both just show
    the bucket absent. Live evidence this session: /api/diag/shadow showed
    rejected_masterkill/rejected_heat/rejected_other (156 + 493 + 350
    records, oldest from 2026-04-20) with ZERO real win/loss labels at any
    horizon, unchanged across two separate nightly-backfill throughput fixes
    (2026-08-07 windowed batching, 2026-08-17 permanent-failure retirement).
    These tests pin get_shadow_stats()'s new backfill_progress_by_decision
    field, which reports labeled / permanently_unlabelable / pending_retry /
    not_yet_attempted counts per decision bucket per horizon, so the next
    check of this evidence gate is diagnostic (never reached vs. actively
    failing vs. already given up on) instead of just another "still empty".
    """

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._orig_log_path = shadow_portfolio.SHADOW_LOG_PATH
        self._orig_lock_path = shadow_portfolio.SHADOW_LOCK_PATH
        shadow_portfolio.SHADOW_LOG_PATH = os.path.join(self._tmpdir.name, "shadow.json")
        shadow_portfolio.SHADOW_LOCK_PATH = shadow_portfolio.SHADOW_LOG_PATH + ".lock"

    def tearDown(self):
        shadow_portfolio.SHADOW_LOG_PATH = self._orig_log_path
        shadow_portfolio.SHADOW_LOCK_PATH = self._orig_lock_path
        self._tmpdir.cleanup()

    def _seed(self, ticker, decision, outcomes=None, bf_attempts=None, age_days=90):
        ts = (datetime.now(timezone.utc) - timedelta(days=age_days)).isoformat()
        records = shadow_portfolio._load_shadow_log()
        rec = {
            "ticker": ticker, "timestamp": ts, "score": 70.0,
            "decision": decision, "decision_reason": "seed",
            "entry_price": 50.0, "vxx_ratio": 1.0, "regime_label": "NEUTRAL",
            "features": {}, "outcomes": outcomes or {"+5d": None, "+10d": None, "+20d": None},
            "code_version": "test-shadow",
        }
        if bf_attempts:
            rec["_bf_attempts"] = bf_attempts
        records.append(rec)
        shadow_portfolio._save_shadow_log(records)

    def test_distinguishes_all_four_states_per_bucket(self):
        # labeled (real win)
        self._seed("A", "rejected_masterkill",
                    outcomes={"+5d": {"label": 1, "return_pct": 3.0, "exit_price": 51.5,
                                       "exit_date": "2026-05-01", "exit_reason": "target",
                                       "label_method": "path_dependent_v2"},
                              "+10d": None, "+20d": None})
        # permanently unlabelable (retired after MAX_BACKFILL_ATTEMPTS)
        self._seed("B", "rejected_masterkill",
                    outcomes={"+5d": {"label": -1, "return_pct": None, "exit_price": None,
                                       "exit_date": None, "exit_reason": "no_price_data",
                                       "label_method": "path_dependent_v2"},
                              "+10d": None, "+20d": None})
        # pending retry (attempted at least once, not yet at the cap)
        self._seed("C", "rejected_masterkill",
                    outcomes={"+5d": None, "+10d": None, "+20d": None},
                    bf_attempts={"+5d": 1})
        # never attempted (frontier hasn't reached it yet, or too recent)
        self._seed("D", "rejected_masterkill",
                    outcomes={"+5d": None, "+10d": None, "+20d": None})

        stats = shadow_portfolio.get_shadow_stats()
        by_h = stats["backfill_progress_by_decision"]["rejected_masterkill"]["+5d"]
        self.assertEqual(by_h["labeled"], 1)
        self.assertEqual(by_h["permanently_unlabelable"], 1)
        self.assertEqual(by_h["pending_retry"], 1)
        self.assertEqual(by_h["not_yet_attempted"], 1)

    def test_buckets_stay_independent(self):
        self._seed("E", "taken",
                    outcomes={"+5d": {"label": 1, "return_pct": 3.0, "exit_price": 51.5,
                                       "exit_date": "2026-05-01", "exit_reason": "target",
                                       "label_method": "path_dependent_v2"},
                              "+10d": None, "+20d": None})
        self._seed("F", "rejected_masterkill",
                    outcomes={"+5d": None, "+10d": None, "+20d": None})

        stats = shadow_portfolio.get_shadow_stats()
        progress = stats["backfill_progress_by_decision"]
        self.assertEqual(progress["taken"]["+5d"]["labeled"], 1)
        self.assertEqual(progress["taken"]["+5d"]["not_yet_attempted"], 0)
        self.assertEqual(progress["rejected_masterkill"]["+5d"]["labeled"], 0)
        self.assertEqual(progress["rejected_masterkill"]["+5d"]["not_yet_attempted"], 1)


if __name__ == "__main__":
    unittest.main()
