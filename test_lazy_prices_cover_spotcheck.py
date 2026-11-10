"""Offline tests for scripts/lazy_prices_cover_spotcheck.py (gate-1 join clause)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import lazy_prices_cover_spotcheck as sc  # noqa: E402
import sec_cik_symbol_map as cm  # noqa: E402

HEAD = ('<ix:nonNumeric name="dei:TradingSymbol" contextRef="c1"><span>ALL</span></ix:nonNumeric>'
        '<ix:nonNumeric contextRef="c2" name="dei:TradingSymbol">brk.b &amp; X</ix:nonNumeric>')


def test_cover_tickers_parses_ix_tags_and_dedupes():
    assert sc.cover_tickers(HEAD) == ["ALL", "BRK.B", "X"]
    assert sc.cover_tickers("<html>no xbrl cover</html>") == []
    assert sc.cover_tickers(None) == []


def test_judge_ignores_class_punctuation_and_empty_cover():
    assert sc.judge("BRKB", ["BRK.B"]) == "match"
    assert sc.judge("AAA", ["BBB"]) == "mismatch"
    assert sc.judge("AAA", []) == "no_cover_ticker"


def test_summarize_excludes_unreadable_and_applies_bars():
    def mk(v, n):
        return [{"verdict": v}] * n
    s = sc.summarize(mk("match", 23) + mk("no_cover_ticker", 2))
    assert s["readable"] == 23 and s["verdict"] == "PASSES_GATE_1_JOIN_CLAUSE"
    assert sc.summarize(mk("match", 17) + mk("mismatch", 3))["verdict"] == "BETWEEN_BARS_NOT_PASSED"
    assert sc.summarize(mk("match", 7) + mk("mismatch", 3))["verdict"] == "KILLED_AT_GATE_1"
    assert sc.summarize(mk("no_cover_ticker", 2))["verdict"] == "INCONCLUSIVE_NO_READABLE_ROWS"


def test_sample_is_seeded_and_only_resolved_rows():
    m = cm.build_map([("AAA", "1", "2021-01-01"), ("AAA", "1", "2021-06-01")])
    rows = [{"cik": "0000000001", "filingDate": "2021-03-01", "name": "A"},
            {"cik": "0000000002", "filingDate": "2021-03-01", "name": "B"}]
    a = sc.sample_resolved(rows, m, cm.resolve, 5, 1)
    assert [r["ticker"] for r in a] == ["AAA"] and len(a) == 1
    assert a == sc.sample_resolved(rows, m, cm.resolve, 5, 1)
