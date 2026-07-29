"""
Feed-resolution battery ([REPAIR 2026-07-06]): the SIP-403 entitlement
fallback that unblinded the Tier2 scan, plus the ratchet banning
hardcoded feed choices from ever creeping back into runtime code.

Run: python3 -m pytest test_alpaca_feed.py -q
"""
import contextlib
import io
import os
import re
import unittest

import alpaca_feed

REPO = os.path.dirname(os.path.abspath(__file__))


class TestFeedResolution(unittest.TestCase):
    def setUp(self):
        alpaca_feed._reset_for_tests()
        os.environ.pop("ALPACA_DATA_FEED", None)

    def test_sip_when_entitled(self):
        self.assertEqual(alpaca_feed.data_feed(now=1000, probe=lambda: 200), "sip")
        self.assertFalse(alpaca_feed.feed_status()["degraded"])

    def test_403_downgrades_to_delayed_sip_never_iex(self):
        """The 2026-07-06 incident: SIP entitlement rejected. The fallback
        MUST be delayed_sip (full consolidated volume, 15-min delay) —
        feed=iex would undercount volume ~30-50x and silently poison every
        dollar-volume floor (measurement integrity beats freshness)."""
        feed = alpaca_feed.data_feed(now=1000, probe=lambda: 403)
        self.assertEqual(feed, "delayed_sip")
        st = alpaca_feed.feed_status()
        self.assertTrue(st["degraded"])
        self.assertIsNotNone(st["downgraded_since"])

    def test_probe_cached_within_ttl_and_recovery_after(self):
        calls = []
        def probe403():
            calls.append(1)
            return 403
        alpaca_feed.data_feed(now=1000, probe=probe403)
        alpaca_feed.data_feed(now=1000 + alpaca_feed.PROBE_TTL_S - 1, probe=probe403)
        self.assertEqual(len(calls), 1, "within TTL the probe never re-fires")
        # subscription restored -> next probe upgrades back automatically
        feed = alpaca_feed.data_feed(now=1000 + alpaca_feed.PROBE_TTL_S + 1, probe=lambda: 200)
        self.assertEqual(feed, "sip")
        self.assertFalse(alpaca_feed.feed_status()["degraded"])

    def test_inconclusive_probe_keeps_current_feed(self):
        """Timeouts/5xx are NOT entitlement answers — never downgrade on them."""
        self.assertEqual(alpaca_feed.data_feed(now=1000, probe=lambda: 0), "sip")
        alpaca_feed.data_feed(now=1000 + alpaca_feed.PROBE_TTL_S + 1, probe=lambda: 403)
        self.assertEqual(alpaca_feed.data_feed(now=1000 + 2 * (alpaca_feed.PROBE_TTL_S + 1),
                                               probe=lambda: 500), "delayed_sip",
                         "5xx while degraded keeps delayed_sip, no flapping")

    def test_state_transition_logs_never_touch_stdout(self):
        """[REPAIR 2026-07-07] alpaca_feed is imported by one-shot subprocesses
        whose ENTIRE stdout is JSON.parse'd by the caller (ml_retrain_safe.py
        -> bot.ts tier3Strategic; manipulation_detect scan). A fresh
        subprocess always re-probes (probed_at resets to 0.0), so while SIP
        stays 403 a stdout print here breaks that JSON.parse on EVERY
        invocation — confirmed live in production: 3/3 hourly retrains failed
        with "Unexpected token 'F', \\"[FEED] SIP \\"... is not valid JSON".
        Any transition log MUST go to stderr, never stdout."""
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            feed = alpaca_feed.data_feed(now=1000, probe=lambda: 403)
        self.assertEqual(feed, "delayed_sip")
        self.assertEqual(out.getvalue(), "",
            "a 403 downgrade transition wrote to stdout — this corrupts any "
            "JSON.parse(subprocess_stdout) caller, e.g. ml_retrain_safe.py")

        out2 = io.StringIO()
        with contextlib.redirect_stdout(out2):
            feed = alpaca_feed.data_feed(now=1000 + alpaca_feed.PROBE_TTL_S + 1,
                                          probe=lambda: 200)
        self.assertEqual(feed, "sip")
        self.assertEqual(out2.getvalue(), "",
            "a restore transition wrote to stdout — same corruption risk")

    def test_env_override_forces_and_skips_probe(self):
        os.environ["ALPACA_DATA_FEED"] = "iex"
        try:
            def boom():
                raise AssertionError("probe must not fire under an env override")
            self.assertEqual(alpaca_feed.data_feed(now=1, probe=boom), "iex")
        finally:
            os.environ.pop("ALPACA_DATA_FEED", None)


class TestBarsFeedResolution(unittest.TestCase):
    """[REPAIR 2026-07-18]: live production evidence — TIER3-ML-ERROR
    recurring every ML retrain cycle since ~2026-07-17T19:00Z with
    HTTP 400 {"message":"invalid feed: delayed_sip"} from Alpaca's
    /v2/stocks/bars endpoint, while the SAME account/state's snapshot
    calls (data_feed() resolving to the identical "delayed_sip" value)
    keep succeeding — Tier2 scans completed normally throughout. The bars
    endpoint's feed enum doesn't include "delayed_sip" (a real-time-tape
    delay concept meaningless for already-historical daily bars);
    bars_feed() swaps in "iex" — the free value alphadesk/alphadesk/
    market.py already uses successfully for this same endpoint — only for
    that one rejected value, leaving every other resolution untouched."""

    def setUp(self):
        alpaca_feed._reset_for_tests()
        os.environ.pop("ALPACA_DATA_FEED", None)

    def test_sip_when_entitled_passes_through(self):
        alpaca_feed.data_feed(now=1000, probe=lambda: 200)
        self.assertEqual(alpaca_feed.bars_feed(), "sip")

    def test_delayed_sip_downgrades_to_iex_for_bars_only(self):
        alpaca_feed.data_feed(now=1000, probe=lambda: 403)
        self.assertEqual(alpaca_feed.data_feed(now=1000), "delayed_sip",
            "sanity: the account IS in the downgraded state this test targets")
        self.assertEqual(alpaca_feed.bars_feed(), "iex",
            "bars endpoint rejects delayed_sip with HTTP 400 — iex is the "
            "only free value it accepts; data_feed() must stay delayed_sip "
            "for every non-bars (snapshot/quote/trade) call site")

    def test_env_override_delayed_sip_still_downgrades_for_bars(self):
        """An explicit ALPACA_DATA_FEED=delayed_sip override (tests/
        emergencies) must not bypass the bars-endpoint substitution —
        the override forces WHAT feed(), not which endpoint validates it."""
        os.environ["ALPACA_DATA_FEED"] = "delayed_sip"
        try:
            self.assertEqual(alpaca_feed.bars_feed(), "iex")
        finally:
            os.environ.pop("ALPACA_DATA_FEED", None)

    def test_env_override_other_value_passes_through_unchanged(self):
        os.environ["ALPACA_DATA_FEED"] = "otc"
        try:
            self.assertEqual(alpaca_feed.bars_feed(), "otc")
        finally:
            os.environ.pop("ALPACA_DATA_FEED", None)


class TestOptionsFeedResolution(unittest.TestCase):
    """[REPAIR 2026-07-25] KNOWN BROKEN #25's diagnosability fix (v1.0.481)
    surfaced the real cause of the Tier1/2 CSP "no options contracts"
    failures: options_execution.py hardcoded feed=opra (Algo Trader Plus),
    and live audit trail confirmed HTTP 403 "subscription does not permit
    querying OPRA data" for SPYM/UBER/HYG on 2026-07-25. options_feed()
    mirrors data_feed()'s probe/cache/downgrade design for the options
    snapshot endpoint, falling back to the free "indicative" feed."""

    def setUp(self):
        alpaca_feed._reset_for_tests()
        os.environ.pop("ALPACA_OPTIONS_FEED", None)

    def test_opra_when_entitled(self):
        self.assertEqual(alpaca_feed.options_feed(now=1000, probe=lambda: 200), "opra")
        self.assertFalse(alpaca_feed.options_feed_status()["degraded"])

    def test_403_downgrades_to_indicative(self):
        feed = alpaca_feed.options_feed(now=1000, probe=lambda: 403)
        self.assertEqual(feed, "indicative")
        st = alpaca_feed.options_feed_status()
        self.assertTrue(st["degraded"])
        self.assertIsNotNone(st["downgraded_since"])

    def test_probe_cached_within_ttl_and_recovery_after(self):
        calls = []
        def probe403():
            calls.append(1)
            return 403
        alpaca_feed.options_feed(now=1000, probe=probe403)
        alpaca_feed.options_feed(now=1000 + alpaca_feed.PROBE_TTL_S - 1, probe=probe403)
        self.assertEqual(len(calls), 1, "within TTL the probe never re-fires")
        feed = alpaca_feed.options_feed(now=1000 + alpaca_feed.PROBE_TTL_S + 1, probe=lambda: 200)
        self.assertEqual(feed, "opra")
        self.assertFalse(alpaca_feed.options_feed_status()["degraded"])

    def test_inconclusive_probe_keeps_current_feed(self):
        self.assertEqual(alpaca_feed.options_feed(now=1000, probe=lambda: 0), "opra")
        alpaca_feed.options_feed(now=1000 + alpaca_feed.PROBE_TTL_S + 1, probe=lambda: 403)
        self.assertEqual(alpaca_feed.options_feed(now=1000 + 2 * (alpaca_feed.PROBE_TTL_S + 1),
                                                   probe=lambda: 500), "indicative",
                         "5xx while degraded keeps indicative, no flapping")

    def test_state_transition_logs_never_touch_stdout(self):
        """Same JSON.parse(subprocess_stdout) hazard as data_feed() — see
        TestFeedResolution.test_state_transition_logs_never_touch_stdout."""
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            feed = alpaca_feed.options_feed(now=1000, probe=lambda: 403)
        self.assertEqual(feed, "indicative")
        self.assertEqual(out.getvalue(), "",
            "a 403 downgrade transition wrote to stdout")

    def test_env_override_forces_and_skips_probe(self):
        os.environ["ALPACA_OPTIONS_FEED"] = "opra"
        try:
            def boom():
                raise AssertionError("probe must not fire under an env override")
            self.assertEqual(alpaca_feed.options_feed(now=1, probe=boom), "opra")
        finally:
            os.environ.pop("ALPACA_OPTIONS_FEED", None)

    def test_data_feed_and_options_feed_state_are_independent(self):
        """The two entitlements (SIP market data vs. OPRA options data) are
        billed and can be rejected independently — a downgrade in one must
        never bleed into the other's cached state."""
        alpaca_feed.data_feed(now=1000, probe=lambda: 403)
        alpaca_feed.options_feed(now=1000, probe=lambda: 200)
        self.assertEqual(alpaca_feed.data_feed(now=1000), "delayed_sip")
        self.assertEqual(alpaca_feed.options_feed(now=1000), "opra")


class TestOptionsExecutionNoHardcodedOpra(unittest.TestCase):
    """RATCHET: every options-data call site must resolve its feed via
    alpaca_feed.options_feed(), never a hardcoded "opra" literal — that
    hardcode is exactly what made the OPRA-403 entitlement rejection
    silent in the first place. [2026-07-28] Widened from
    options_execution.py-only to ALL runtime modules: the remaining three
    deliberate hardcodes (options_scanner.py, options_manager.py,
    vol_surface.py — KNOWN BROKEN #25's filed follow-ups) are now wired,
    so the whole options data plane rides one resolver and the next
    entitlement change is a one-module event."""

    HARDCODED = re.compile(r'''["']feed["']\s*:\s*["']opra["']''')
    WIRED = ["options_execution.py", "options_scanner.py",
             "options_manager.py", "vol_surface.py"]

    def test_no_runtime_module_hardcodes_opra(self):
        offenders = []
        for f in os.listdir(REPO):
            if not f.endswith(".py") or f.startswith("test_") or f == "alpaca_feed.py":
                continue
            src = open(os.path.join(REPO, f), encoding="utf-8", errors="replace").read()
            for m in self.HARDCODED.finditer(src):
                offenders.append(f"{f}: {m.group(0)}")
        self.assertEqual(offenders, [],
            f"hardcoded opra feed found — use alpaca_feed.options_feed(): {offenders}")

    def test_options_modules_call_the_resolver(self):
        for f in self.WIRED:
            src = open(os.path.join(REPO, f), encoding="utf-8").read()
            self.assertIn("alpaca_feed.options_feed()", src,
                f"{f} must call alpaca_feed.options_feed() for its options fetches")


class TestNoHardcodedFeeds(unittest.TestCase):
    HARDCODED = re.compile(r'''feed=sip|["']feed["']\s*:\s*["'](?:sip|iex|delayed_sip)["']''')

    def test_runtime_modules_use_the_resolver(self):
        """RATCHET: no runtime module may hardcode an Alpaca feed choice.
        44 hardcoded feed=sip sites made the 2026-07-06 entitlement loss a
        whole-stack outage; every data request now flows through
        alpaca_feed.data_feed() so the next entitlement change is a
        one-module event."""
        offenders = []
        for f in os.listdir(REPO):
            if not f.endswith(".py") or f.startswith("test_") or f == "alpaca_feed.py":
                continue
            src = open(os.path.join(REPO, f), encoding="utf-8", errors="replace").read()
            for m in self.HARDCODED.finditer(src):
                offenders.append(f"{f}: {m.group(0)}")
        self.assertEqual(offenders, [],
            f"hardcoded Alpaca feed found — use alpaca_feed.data_feed(): {offenders}")


class TestBarsEndpointsUseBarsFeed(unittest.TestCase):
    """RATCHET [REPAIR 2026-07-18]: every /v2/stocks/bars (single- or
    multi-symbol) call site must resolve its feed via bars_feed(), never
    data_feed() — the 2026-07-18 incident (see TestBarsFeedResolution)
    was exactly a bars call site using data_feed()'s "delayed_sip" value,
    which that endpoint's API rejects outright with HTTP 400. Would have
    caught every one of the 29 sites this repair fixed across 13 files."""

    BARS_URL = re.compile(r'/v2/stocks/(?:\{[^}]+\}/)?bars')

    def test_bars_call_sites_use_bars_feed(self):
        offenders = []
        for f in os.listdir(REPO):
            if not f.endswith(".py") or f.startswith("test_") or f == "alpaca_feed.py":
                continue
            src = open(os.path.join(REPO, f), encoding="utf-8", errors="replace").read()
            for m in self.BARS_URL.finditer(src):
                window = src[m.end():m.end() + 400]
                # Stop the search scope at the next bars URL occurrence so one
                # call's feed param can't be mistaken for a neighboring call's.
                next_bars = self.BARS_URL.search(window[1:])
                scope = window if next_bars is None else window[:next_bars.start() + 1]
                if "feed" not in scope:
                    continue  # this bars call passes no feed param — different bug class
                if "bars_feed()" in scope:
                    continue
                if "data_feed()" in scope:
                    offenders.append(f"{f}: /bars call near offset {m.start()} "
                                      f"still resolves its feed via data_feed()")
        self.assertEqual(offenders, [],
            f"bars endpoint(s) using data_feed() instead of bars_feed() — "
            f"Alpaca's bars API rejects the delayed_sip value data_feed() can "
            f"return with HTTP 400: {offenders}")


if __name__ == "__main__":
    unittest.main()
