"""
Regression tests for scripts/event_reaction_study.py (pre-registered
event-reaction study, research/event_reaction_study.md). Pure functions only,
synthetic data, no network: event-day alignment incl. the after-close rule,
abnormal-return math, no-lookahead guarantees, cluster logic, controls,
statistics and the pre-registered verdict bar.
"""
import importlib.util
import os
import random
import unittest

_spec = importlib.util.spec_from_file_location(
    "event_reaction_study",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "event_reaction_study.py"))
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

# Tue 2024-01-02 .. Mon 2024-01-08 (Sat/Sun 6-7 absent), plus a July DST week.
DAYS = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08",
        "2024-07-01", "2024-07-02", "2024-07-03", "2024-07-05"]


class TestEventDayAlignment(unittest.TestCase):
    def test_acceptance_json_is_true_utc(self):
        # 2026-07-30T20:30:28Z is the AAPL 8-K whose index-headers say 16:30:28 ET
        et = m.acceptance_to_et("2026-07-30T20:30:28.000Z")
        self.assertEqual((et.hour, et.minute), (16, 30))

    def test_intraday_filing_is_same_day(self):
        # 15:59 EST on Wed 01-03
        self.assertEqual(DAYS[m.event_day0(DAYS, "2024-01-03T20:59:00.000Z")], "2024-01-03")

    def test_pre_open_filing_is_same_day(self):
        self.assertEqual(DAYS[m.event_day0(DAYS, "2024-01-03T11:00:00.000Z")], "2024-01-03")

    def test_after_close_filing_is_next_day(self):
        # 16:00 exactly and 16:01 EST -> next trading day
        self.assertEqual(DAYS[m.event_day0(DAYS, "2024-01-03T21:00:00.000Z")], "2024-01-04")
        self.assertEqual(DAYS[m.event_day0(DAYS, "2024-01-03T21:01:00.000Z")], "2024-01-04")

    def test_utc_date_rollover_uses_et_date(self):
        # 2024-01-05T02:00Z is 21:00 ET on Thu 01-04 -> Fri 01-05 (not Mon)
        self.assertEqual(DAYS[m.event_day0(DAYS, "2024-01-05T02:00:00.000Z")], "2024-01-05")

    def test_weekend_filing_is_next_trading_day(self):
        self.assertEqual(DAYS[m.event_day0(DAYS, "2024-01-06T15:00:00.000Z")], "2024-01-08")

    def test_dst_offset(self):
        # 19:59Z in July = 15:59 EDT -> same day; 20:01Z = 16:01 EDT -> next
        self.assertEqual(DAYS[m.event_day0(DAYS, "2024-07-02T19:59:00.000Z")], "2024-07-02")
        self.assertEqual(DAYS[m.event_day0(DAYS, "2024-07-02T20:01:00.000Z")], "2024-07-03")

    def test_holiday_filing_rolls_forward(self):
        # July 4 is absent from the calendar -> Jul 5
        self.assertEqual(DAYS[m.event_day0(DAYS, "2024-07-04T14:00:00.000Z")], "2024-07-05")

    def test_fallback_filing_date_is_strictly_after(self):
        # no acceptance time: never same day (conservative)
        self.assertEqual(DAYS[m.event_day0(DAYS, None, "2024-01-03")], "2024-01-04")
        self.assertEqual(DAYS[m.event_day0(DAYS, None, "2024-01-05")], "2024-01-08")

    def test_day0_never_precedes_public_time(self):
        rng = random.Random(1)
        for _ in range(500):
            day = rng.choice(DAYS[:-1])
            hh, mm = rng.randrange(24), rng.randrange(60)
            iso = f"{day}T{hh:02d}:{mm:02d}:00.000Z"
            i = m.event_day0(DAYS, iso)
            if i is None:
                continue
            et = m.acceptance_to_et(iso)
            # day 0 close (16:00 ET of DAYS[i]) is at or after public time
            self.assertGreaterEqual((DAYS[i], 16 * 60), (et.date().isoformat(), et.hour * 60 + et.minute))

    def test_beyond_calendar_is_none(self):
        self.assertIsNone(m.event_day0(DAYS, "2024-07-05T21:00:00.000Z"))
        self.assertIsNone(m.event_day0(DAYS))


class TestAbnormalReturns(unittest.TestCase):
    def test_simple_returns(self):
        r = m.simple_returns([100.0, 110.0, None, 121.0, 0.0])
        self.assertIsNone(r[0])
        self.assertAlmostEqual(r[1], 0.10)
        self.assertIsNone(r[2])
        self.assertIsNone(r[3])
        self.assertIsNone(r[4])

    def test_car_is_sum_of_excess(self):
        s = [None, 0.02, 0.01, -0.03, 0.05]
        mk = [None, 0.01, 0.01, 0.00, 0.01]
        self.assertAlmostEqual(m.car(s, mk, 1, 0, 1), 0.01 + 0.0)
        self.assertAlmostEqual(m.car(s, mk, 1, 2, 3), -0.03 + 0.04)

    def test_car_full_coverage_rule(self):
        s = [None, 0.01, None, 0.01]
        mk = [None, 0.0, 0.0, 0.0]
        self.assertIsNone(m.car(s, mk, 1, 0, 2))
        self.assertIsNone(m.car(s, mk, 1, 0, 5))   # beyond data
        self.assertIsNone(m.car(s, mk, 0, 0, 1))   # day 0 has no prior close

    def test_car_uses_no_data_before_window(self):
        # a huge move on day 0 must not leak into CAR[2,x]
        s = [None, 0.50, 0.0, 0.01, 0.02]
        mk = [None, 0.0, 0.0, 0.0, 0.0]
        self.assertAlmostEqual(m.car(s, mk, 1, 2, 3), 0.03)
        self.assertAlmostEqual(m.car(s, mk, 1, 0, 1), 0.50)

    def test_car_index_matches_car(self):
        rng = random.Random(7)
        s = [None] + [None if rng.random() < 0.05 else rng.gauss(0, 0.02) for _ in range(300)]
        mk = [None] + [rng.gauss(0, 0.01) for _ in range(300)]
        ci = m.CarIndex(s, mk)
        for _ in range(400):
            i0, a = rng.randrange(0, 301), rng.randrange(0, 3)
            b = a + rng.randrange(0, 60)
            x, y = m.car(s, mk, i0, a, b), ci.car(i0, a, b)
            if x is None:
                self.assertIsNone(y)
            else:
                self.assertAlmostEqual(x, y, places=12)

    def test_market_move_cancels(self):
        s = [None, 0.03, -0.02]
        self.assertAlmostEqual(m.car(s, list(s), 1, 0, 1), 0.0)


class TestEventConstruction(unittest.TestCase):
    def _p(self, owner, td, fd, acc, issuer="A"):
        return {"issuer": issuer, "owner": owner, "trans_date": td, "filing_date": fd, "accession": acc}

    def test_cluster_requires_two_distinct_owners(self):
        ps = [self._p("o1", "2020-01-01", "2020-01-03", "a1"),
              self._p("o1", "2020-01-02", "2020-01-04", "a2")]
        self.assertEqual(m.insider_clusters(ps), [])

    def test_cluster_fires_on_completing_filing(self):
        ps = [self._p("o1", "2020-01-01", "2020-01-03", "a1"),
              self._p("o2", "2020-01-08", "2020-01-10", "a2")]
        ev = m.insider_clusters(ps)
        self.assertEqual(ev, [{"issuer": "A", "filing_date": "2020-01-10", "accession": "a2"}])

    def test_cluster_window_is_10_days(self):
        ps = [self._p("o1", "2020-01-01", "2020-01-02", "a1"),
              self._p("o2", "2020-01-12", "2020-01-13", "a2")]
        self.assertEqual(m.insider_clusters(ps), [])

    def test_cluster_is_backward_only(self):
        # o2 traded BEFORE o1 but filed later: the cluster can only fire when
        # o2's filing is public, never at o1's earlier filing date
        ps = [self._p("o1", "2020-01-15", "2020-01-16", "a1"),
              self._p("o2", "2020-01-14", "2020-01-20", "a2")]
        ev = m.insider_clusters(ps)
        self.assertEqual([e["filing_date"] for e in ev], ["2020-01-20"])

    def test_cluster_window_trails_the_filing_date(self):
        # a stale late-filed trade (outside the 10 days before the filing)
        # does not complete a cluster
        ps = [self._p("o1", "2020-01-05", "2020-01-06", "a1"),
              self._p("o2", "2020-01-04", "2020-01-20", "a2")]
        self.assertEqual(m.insider_clusters(ps), [])

    def test_cluster_cooldown(self):
        ps = [self._p("o1", "2020-01-01", "2020-01-02", "a1"),
              self._p("o2", "2020-01-02", "2020-01-03", "a2"),
              self._p("o3", "2020-01-10", "2020-01-11", "a3"),
              self._p("o1", "2020-02-10", "2020-02-11", "a4"),
              self._p("o4", "2020-02-11", "2020-02-12", "a5")]
        self.assertEqual([e["accession"] for e in m.insider_clusters(ps)], ["a2", "a5"])

    def test_dedup_keeps_first(self):
        self.assertEqual(m.dedup_day0([100, 105, 120, 121, 150]), [100, 121, 150])

    def test_parse_items(self):
        self.assertEqual(m.parse_items("1.01,2.03, 9.01"), {"1.01", "2.03", "9.01"})
        self.assertEqual(m.parse_items(""), set())

    def test_departure_classifier(self):
        self.assertTrue(m.classify_departure(
            "On May 1, John Doe resigned as Chief Financial Officer of the Company, effective immediately."))
        self.assertTrue(m.classify_departure("The Board terminated the employment of Jane Roe, our CEO."))
        self.assertFalse(m.classify_departure(
            "John Doe, Chief Executive Officer, announced his retirement and will step down in June."))
        self.assertFalse(m.classify_departure("The Board appointed Jane Roe as Chief Financial Officer."))
        self.assertFalse(m.classify_departure("A director resigned from the Board."))
        # 'ceo' lowercase inside another word must not match the short form
        self.assertFalse(m.classify_departure("The Theceology director departed."))

    def test_parse_sec_date(self):
        self.assertEqual(m.parse_sec_date("31-MAR-2016"), "2016-03-31")
        self.assertEqual(m.parse_sec_date("2016-03-31"), "2016-03-31")
        self.assertIsNone(m.parse_sec_date("n/a"))
        self.assertIsNone(m.parse_sec_date("31-FEB-2016"))
        self.assertIsNone(m.parse_sec_date("31-XYZ-2016"))
        self.assertEqual(m.parse_sec_date("05-dec-2020"), "2020-12-05")


class TestControls(unittest.TestCase):
    def test_pool_excludes_event_neighbourhoods(self):
        pool = m.control_pool(300, 1000, [300, 400], span=250, exclude=20)
        self.assertTrue(all(abs(j - 300) > 20 and abs(j - 400) > 20 for j in pool))
        self.assertEqual(min(pool), 50)
        self.assertEqual(max(pool), 550)

    def test_pool_clipped_to_data(self):
        pool = m.control_pool(10, 100, [10])
        self.assertGreaterEqual(min(pool), 1)
        self.assertLess(max(pool), 100)

    def test_rng_is_deterministic(self):
        a = m.event_rng("XYZ", 500, "E3").sample(range(1000), 5)
        b = m.event_rng("XYZ", 500, "E3").sample(range(1000), 5)
        self.assertEqual(a, b)


class TestStats(unittest.TestCase):
    def test_winsorize(self):
        v = list(range(101))
        w = m.winsorize(v, 0.01, 0.99)
        self.assertEqual(min(w), 1)
        self.assertEqual(max(w), 99)

    def test_clustered_mean_t_matches_iid_when_singletons(self):
        vals = [0.01, 0.03, -0.02, 0.04, 0.00]
        st = m.clustered_mean_t(vals, list(range(5)))
        n = len(vals)
        mean = sum(vals) / n
        # cluster-robust with singleton clusters = n/(n-1) * sum e^2 / n^2 = s^2/n
        s2 = sum((x - mean) ** 2 for x in vals) / (n - 1)
        self.assertAlmostEqual(st["se"], (s2 / n) ** 0.5)

    def test_clustering_widens_se_for_correlated_clusters(self):
        rng = random.Random(3)
        vals, cl = [], []
        for g in range(40):
            shock = rng.gauss(0, 0.05)
            for _ in range(10):
                vals.append(shock + rng.gauss(0, 0.01))
                cl.append(g)
        iid = m.clustered_mean_t(vals, list(range(len(vals))))
        clu = m.clustered_mean_t(vals, cl)
        self.assertGreater(clu["se"], 2 * iid["se"])

    def test_week_cluster(self):
        self.assertEqual(m.week_cluster("2024-01-03"), m.week_cluster("2024-01-05"))
        self.assertNotEqual(m.week_cluster("2024-01-05"), m.week_cluster("2024-01-08"))

    def test_abdi_ranaldo_zero_when_close_at_mid(self):
        h = [11.0] * 30
        lo = [9.0] * 30
        c = [99 ** 0.5] * 30   # log close == mid of log H/L
        self.assertAlmostEqual(m.abdi_ranaldo(h, lo, c), 0.0, places=9)

    def test_abdi_ranaldo_bid_ask_bounce(self):
        rng = random.Random(5)
        h, lo, c = [], [], []
        for _ in range(250):
            mid = 10.0
            c.append(mid * (1.01 if rng.random() < 0.5 else 0.99))  # 2% spread bounce
            h.append(mid * 1.01)
            lo.append(mid * 0.99)
        s = m.abdi_ranaldo(h, lo, c)
        self.assertGreater(s, 0.01)
        self.assertLess(s, 0.04)

    def test_size_bucket(self):
        self.assertEqual(m.size_bucket(1, (2, 5)), "small")
        self.assertEqual(m.size_bucket(3, (2, 5)), "mid")
        self.assertEqual(m.size_bucket(9, (2, 5)), "large")
        self.assertIsNone(m.size_bucket(None, (2, 5)))
        self.assertIsNone(m.size_bucket(3, None))

    def test_late_share(self):
        self.assertAlmostEqual(m.late_share(0.01, 0.04, 3.0), 0.25)
        self.assertIsNone(m.late_share(0.01, 0.04, 1.5))
        self.assertIsNone(m.late_share(-0.01, 0.04, 3.0))


class TestVerdict(unittest.TestCase):
    H = ({"mean": -0.02, "t": -1.5}, {"mean": -0.015, "t": -1.2})

    def test_negative_cell_passes(self):
        st = {"mean": -0.02, "t": -3.5}
        self.assertEqual(m.cell_verdict(st, self.H, 0.01, m.T_BAR), "candidate")

    def test_positive_cell_cannot_pass_on_survivors(self):
        st = {"mean": 0.02, "t": 3.5}
        h = ({"mean": 0.02, "t": 1.5}, {"mean": 0.01, "t": 1.1})
        self.assertEqual(m.cell_verdict(st, h, 0.01, m.T_BAR), "positive_unconfirmed")

    def test_bar_requirements(self):
        self.assertEqual(m.cell_verdict({"mean": -0.005, "t": -5}, self.H, 0.001, m.T_BAR), "fail")
        self.assertEqual(m.cell_verdict({"mean": -0.02, "t": -2.8}, self.H, 0.001, m.T_BAR), "fail")
        self.assertEqual(m.cell_verdict({"mean": -0.02, "t": -2.9}, self.H, 0.001, m.T_BAR_E1), "fail")
        flip = ({"mean": -0.02, "t": -1.5}, {"mean": 0.01, "t": 1.2})
        self.assertEqual(m.cell_verdict({"mean": -0.02, "t": -4}, flip, 0.001, m.T_BAR), "fail")
        weak = ({"mean": -0.02, "t": -1.5}, {"mean": -0.01, "t": -0.5})
        self.assertEqual(m.cell_verdict({"mean": -0.02, "t": -4}, weak, 0.001, m.T_BAR), "fail")
        # cost: drift smaller than the round-trip spread proxy
        self.assertEqual(m.cell_verdict({"mean": -0.02, "t": -4}, self.H, 0.03, m.T_BAR), "fail")
        self.assertEqual(m.cell_verdict({"mean": -0.02, "t": -4}, self.H, None, m.T_BAR), "fail")

    def test_bonferroni_bars(self):
        from statistics import NormalDist
        z = NormalDist().inv_cdf
        self.assertAlmostEqual(m.T_BAR, z(1 - 0.05 / 12 / 2), delta=0.01)
        self.assertAlmostEqual(m.T_BAR_E1, z(1 - 0.05 / 20 / 2), delta=0.01)

    def test_type_verdict(self):
        self.assertEqual(m.type_verdict(["fail", "candidate", "fail"], 1.0), "CANDIDATE SIGNAL (gate 2)")
        self.assertEqual(m.type_verdict(["fail", "positive_unconfirmed", "fail"], 1.0),
                         "POSITIVE ON SURVIVORS — UNCONFIRMED")
        self.assertEqual(m.type_verdict(["fail"] * 3, -5.0), "PRICED IN FAST / NO DRIFT")
        self.assertEqual(m.type_verdict(["fail"] * 3, 1.0), "NO MEASURABLE REACTION")


class TestMeasureTickerNoLookahead(unittest.TestCase):
    def _days(self, n):
        from datetime import date, timedelta
        d, out = date(2015, 1, 5), []
        while len(out) < n:
            if d.weekday() < 5:
                out.append(d.isoformat())
            d += timedelta(days=1)
        return out

    def test_post_window_shock_does_not_move_reaction(self):
        days = self._days(700)
        rng = random.Random(11)
        closes = [50.0]
        for _ in range(699):
            closes.append(closes[-1] * (1 + rng.gauss(0, 0.01)))
        spy_ret = [None] + [0.0] * 699
        bull = [True] * 700
        bars = {"date": days, "close": closes, "high": [c * 1.01 for c in closes],
                "low": [c * 0.99 for c in closes], "volume": [1e6] * 700}
        ev = {"E3_item_4_02": [400]}
        base, _ = m.measure_ticker("T", days, spy_ret, bull, bars, ev)
        shocked = dict(bars)
        shocked["close"] = closes[:425] + [c * 0.5 for c in closes[425:]]  # crash on day +25
        after, _ = m.measure_ticker("T", days, spy_ret, bull, shocked, ev)
        self.assertEqual(len(base), 1)
        for w in ("0_1", "2_5", "2_20", "0_20"):
            self.assertAlmostEqual(base[0]["car"][w], after[0]["car"][w])
        self.assertNotAlmostEqual(base[0]["car"]["2_60"], after[0]["car"]["2_60"])
        # bucket inputs use only pre-event data
        self.assertEqual(base[0]["medvol"] is None, after[0]["medvol"] is None)

    def test_event_reaction_recovered(self):
        days = self._days(700)
        closes = [20.0] * 700
        for t in range(400, 700):
            closes[t] = 18.0   # -10% on day 0, flat afterwards
        bars = {"date": days, "close": closes, "high": [c * 1.01 for c in closes],
                "low": [c * 0.99 for c in closes], "volume": [1e6] * 700}
        recs, _ = m.measure_ticker("T", days, [None] + [0.0] * 699, [True] * 700, bars,
                                   {"E3_item_4_02": [400]})
        self.assertAlmostEqual(recs[0]["car"]["0_1"], -0.10)
        self.assertAlmostEqual(recs[0]["car"]["2_20"], 0.0)
        # controls are flat -> excess equals raw
        self.assertAlmostEqual(recs[0]["excess"]["0_1"], -0.10)

    def test_penny_and_short_history_excluded(self):
        days = self._days(700)
        bars = {"date": days, "close": [0.5] * 700, "high": [0.51] * 700, "low": [0.49] * 700,
                "volume": [1e6] * 700}
        recs, _ = m.measure_ticker("T", days, [None] + [0.0] * 699, [True] * 700, bars,
                                   {"E2_item_1_01": [400]})
        self.assertEqual(recs, [])
        bars2 = dict(bars, close=[5.0] * 700, high=[5.1] * 700, low=[4.9] * 700)
        recs2, _ = m.measure_ticker("T", days, [None] + [0.0] * 699, [True] * 700, bars2,
                                    {"E2_item_1_01": [30]})
        self.assertEqual(recs2, [])


if __name__ == "__main__":
    unittest.main()
