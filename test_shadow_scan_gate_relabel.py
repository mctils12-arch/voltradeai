"""
[RULE-REVIEW] 2026-10-26 (step 2 of "Expand counterfactual logging to ALL
rule gates"): _scan_market_inner's per-candidate loop runs AFTER deep_score
has logged every candidate "taken", and ten of its filters dropped a
candidate with a bare `continue` — the record stayed "taken" and the rule
had no prevention-P&L. Each site now notes its rejection and ONE
update_decisions_batch call per scan relabels them (batched because the
regime gate drops every candidate in BEAR/PANIC). Source-inspection, the
pattern test_shadow_portfolio.TestBotEngineCallSites established for this
function: the loop needs Alpaca, so the wiring is pinned on the source.
"""
import inspect
import re
import unittest

import bot_engine

SRC = inspect.getsource(bot_engine._scan_market_inner)

# (bucket, the statement that drops the candidate after the note)
SITES = [
    ("rejected_held", "continue"),
    ("rejected_cooldown", "continue"),
    ("rejected_regime_gate", "continue"),
    ("rejected_score", "continue"),
    ("rejected_dollar_volume", "continue"),
    ("rejected_blocked_ticker", "continue"),
    ("rejected_extreme_mover", "continue"),
    ("rejected_sizing_error", "continue"),
    ("rejected_zero_shares", "continue"),
    ("rejected_volume_share", "continue"),
    ("rejected_stock_slots", "break"),
]


class TestScanGateSites(unittest.TestCase):
    def _site(self, bucket, exit_stmt="continue"):
        # The per-candidate sites note `ticker`; the slot-break remainder
        # notes `_rt` (and also uses rejected_held, so the continue-site must
        # be found by its own call shape, not by the first bucket mention).
        call = f'_note_gate(_rt, "{bucket}"' if exit_stmt == "break" else f'_note_gate(ticker, "{bucket}"'
        start = SRC.find(call)
        self.assertNotEqual(start, -1, f"no _note_gate site for {bucket}: {call}")
        return start

    def test_every_site_notes_then_drops(self):
        for bucket, exit_stmt in SITES:
            with self.subTest(bucket=bucket):
                start = self._site(bucket, exit_stmt)
                window = SRC[start:start + 700]
                m = re.search(rf"\n\s+{exit_stmt}\b", window)
                self.assertIsNotNone(m, f"{bucket}: no `{exit_stmt}` after the note")
                nxt = window.find("_note_gate(", 1)
                self.assertTrue(nxt == -1 or nxt > m.start(),
                                f"{bucket}: the `{exit_stmt}` must come before the next site")

    def test_slot_break_labels_the_remainder(self):
        self.assertIn("for _di, stock in enumerate(deep_scored):", SRC)
        head = SRC[SRC.find("for _di, stock in enumerate(deep_scored):"):]
        head = head[:head.find("final_score = stock.get(")]
        self.assertIn("for _rest in deep_scored[_di:]:", head)
        self.assertIn('"rejected_held"', head)
        self.assertIn('"rejected_stock_slots"', head)
        self.assertIn("adaptive MAX_POSITIONS", head, "the reason must carry the adaptive cap so a regime-zeroed slot count is readable")
        self.assertLess(head.find('"rejected_stock_slots"'), head.find("break"))

    def test_adaptive_threshold_site_names_the_regime(self):
        start = self._site("rejected_score")
        window = SRC[start:start + 300]
        self.assertIn("adaptive MIN_SCORE", window)
        self.assertIn("_min_score_threshold", window)
        self.assertIn("_scan_regime", window)

    def test_buckets_are_exactly_the_expected_set(self):
        used = set(re.findall(r'_note_gate\([^,]+,\s*"(rejected_[a-z_]+)"', SRC))
        self.assertEqual(used, {b for b, _ in SITES})

    def test_existing_inline_sites_untouched(self):
        # The three sites that already relabel inline keep doing so — this
        # change adds the missing ones, it does not re-route the working ones.
        for bucket in ('"rejected_heat"', '"rejected_other"'):
            idx = SRC.find(bucket)
            self.assertNotEqual(idx, -1)
            self.assertIn("update_last_decision", SRC[idx - 200:idx])


class TestScanGateFlush(unittest.TestCase):
    def test_one_batched_flush_after_the_loop_before_the_partial_result(self):
        last_site = SRC.find('"rejected_volume_share"')
        flush = SRC.find("update_decisions_batch(_gate_relabels)")
        # _partial_scan_result is assigned more than once in the scan; the one
        # that must follow the flush is the post-loop one that carries the counts.
        partial = SRC.find('"shadow_gate_relabels": _shadow_gate_relabels,')
        self.assertTrue(-1 < last_site < flush < partial,
                        "the flush must sit after the last gate site and before the partial result that reports it")
        self.assertEqual(SRC.count("update_decisions_batch("), 1, "exactly one relabel call per scan")

    def test_flush_failure_is_logged_not_swallowed(self):
        flush = SRC.find("update_decisions_batch(_gate_relabels)")
        window = SRC[flush - 200:flush + 600]
        self.assertIn("try:", window)
        self.assertIn("except Exception as _gr_err:", window)
        self.assertIn(".warning(", window)
        self.assertNotIn("except Exception:\n            pass", window)

    def test_counts_travel_in_both_scan_results(self):
        self.assertEqual(SRC.count('"shadow_gate_relabels":'), 2, "partial and final scan results both carry the counts")
        self.assertIn('"by_decision"', SRC)

    def test_note_gate_truncates_reason_like_the_record_field(self):
        helper = SRC[SRC.find("def _note_gate("):SRC.find("def _note_gate(") + 300]
        self.assertIn("str(_reason)[:200]", helper)
        self.assertIn('"ticker": _t', helper)
        self.assertIn('"decision": _decision', helper)


if __name__ == "__main__":
    unittest.main()
