"""
Feed-resolution battery ([REPAIR 2026-07-06]): the SIP-403 entitlement
fallback that unblinded the Tier2 scan, plus the ratchet banning
hardcoded feed choices from ever creeping back into runtime code.

Run: python3 -m pytest test_alpaca_feed.py -q
"""
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

    def test_env_override_forces_and_skips_probe(self):
        os.environ["ALPACA_DATA_FEED"] = "iex"
        try:
            def boom():
                raise AssertionError("probe must not fire under an env override")
            self.assertEqual(alpaca_feed.data_feed(now=1, probe=boom), "iex")
        finally:
            os.environ.pop("ALPACA_DATA_FEED", None)


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


if __name__ == "__main__":
    unittest.main()
