"""
AlphaDesk CLI.

  python -m alphadesk AAPL
  python -m alphadesk AAPL --json
  python -m alphadesk AAPL --bracket 0.37 --state 0.093 --ltcg 0.20
  python -m alphadesk selftest
"""
from __future__ import annotations

import argparse
import json
import sys

from . import analyze, TaxProfile, SampleProvider
from .providers import make_provider
from .report import render_text


def _build_tax(args) -> TaxProfile:
    p = TaxProfile()
    if args.bracket is not None:
        p.ordinary_marginal_rate = args.bracket
    if args.ltcg is not None:
        p.ltcg_rate = args.ltcg
    if args.state is not None:
        p.state_rate = args.state
    if args.no_niit:
        p.niit_applies = False
    return p


def _selftest() -> int:
    checks = []

    def ok(name, cond):
        checks.append((name, bool(cond)))

    r = analyze("AAPL", provider=SampleProvider())
    ok("verdict assigned", r.verdict in
       {"Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"})
    ok("composite 0-100", 0 <= r.composite_score <= 100)
    ok("conviction 0-100", 0 <= r.conviction <= 100)
    ok("two horizons", len(r.horizons) == 2)
    ok("after-tax <= pre-tax on gains",
       all(h.after_tax_return <= h.expected_return + 1e-9 or h.expected_return <= 0
           for h in r.horizons))
    ok("deterministic", analyze("AAPL").composite_score == analyze("AAPL").composite_score)
    ok("best horizon is one of the two",
       r.best_horizon in {h.label for h in r.horizons})
    ok("never asserts certainty", "Not investment advice" in r.disclaimer)

    # Live field-mapping is verifiable offline: feed a Finnhub-shaped metric
    # object through the same pure mapper LiveProvider uses and confirm fields
    # land in the right place with the right scaling (percent -> fraction for
    # margins/growth/returns; raw for ratios). No network or keys required.
    from .providers import _map_finnhub_metrics
    fm = _map_finnhub_metrics(
        {"peTTM": 31.2, "psTTM": 8.1, "epsTTM": 6.13,
         "currentRatioQuarterly": 0.92, "revenueGrowthTTMYoy": 7.8,
         "grossMarginTTM": 43.3, "netProfitMarginTTM": 25.0, "roeTTM": 147.4,
         # mislabeled keys that must NOT be consumed as forward P/E or EV/EBITDA
         "peExclExtraTTM": 30.9, "currentEv/freeCashFlowTTM": 29.7},
        SampleProvider().fundamentals("AAPL"))
    ok("finnhub P/E mapped raw", abs(fm.pe - 31.2) < 1e-9)
    ok("finnhub P/S mapped raw", abs(fm.price_sales - 8.1) < 1e-9)
    ok("finnhub margin -> fraction", abs(fm.gross_margin - 0.433) < 1e-9)
    ok("finnhub growth -> fraction", abs(fm.revenue_growth_yoy - 0.078) < 1e-9)
    ok("finnhub ROE -> fraction", abs(fm.roe - 1.474) < 1e-9)
    ok("EV/EBITDA not fed EV/FCF", fm.ev_ebitda != 29.7)
    ok("forward P/E not fed trailing", fm.forward_pe != 30.9)

    # SEC EDGAR parsing is verifiable offline: feed EDGAR-shaped payloads
    # through the same pure functions LiveProvider.filings uses and confirm CIK
    # resolution, form filtering + URL construction, and HTML->text extraction.
    from . import edgar
    tickers_json = {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"},
    }
    ok("EDGAR resolves CIK (zero-padded)", edgar.resolve_cik("aapl", tickers_json) == "0000320193")
    ok("EDGAR unknown ticker -> None", edgar.resolve_cik("ZZZZ", tickers_json) is None)

    submissions = {"filings": {"recent": {
        "form": ["4", "10-Q", "8-K", "10-K"],
        "filingDate": ["2026-06-17", "2026-05-01", "2026-04-10", "2026-02-01"],
        "accessionNumber": ["0000320193-26-000001", "0000320193-26-000050",
                            "0000320193-26-000040", "0000320193-26-000010"],
        "primaryDocument": ["x/f4.xml", "q.htm", "ev.htm", "k.htm"],
        "primaryDocDescription": ["FORM 4", "10-Q", "8-K", "10-K"],
    }}}
    fil = edgar.parse_recent_filings("AAPL", "0000320193", submissions, limit=5)
    ok("EDGAR keeps only material forms", [f.form for f in fil] == ["10-Q", "8-K", "10-K"])
    ok("EDGAR builds archive URL",
       fil[0].url == "https://www.sec.gov/Archives/edgar/data/320193/000032019326000050/q.htm")
    ok("EDGAR respects limit",
       len(edgar.parse_recent_filings("AAPL", "0000320193", submissions, limit=1)) == 1)
    txt = edgar.html_to_text("<html><style>x{}</style><body><p>Going&nbsp;concern  &amp;   risk</p></body></html>")
    ok("EDGAR strips HTML to text", txt == "Going concern & risk")
    ok("EDGAR caps text length", len(edgar.html_to_text("<p>" + "a" * 100000 + "</p>", cap=100)) == 100)

    # The Claude filing reader is verifiable offline by injecting a fake client
    # shaped like the SDK response (a thinking block then a JSON text block) and
    # confirming the reader parses structured output, clamps sentiment, and tags
    # its source — no API key or network required.
    from .llm_reader import ClaudeFilingReader, make_filing_reader
    from .models import Filing
    from .config import Keys

    class _FakeBlock:
        def __init__(self, type, text=""):
            self.type = type
            self.text = text

    class _FakeResp:
        def __init__(self, content):
            self.content = content

    class _FakeMessages:
        def __init__(self, payload):
            self._payload = payload

        def create(self, **_):
            return _FakeResp([_FakeBlock("thinking"), _FakeBlock("text", json.dumps(self._payload))])

    class _FakeClient:
        def __init__(self, payload):
            self.messages = _FakeMessages(payload)

    reader = ClaudeFilingReader(client=_FakeClient(
        {"summary": "Steady quarter.", "sentiment": 2.5,
         "red_flags": ["Litigation disclosed"], "catalysts": []}))
    fr = reader.read([Filing(ticker="AAPL", form="10-Q", filed_at="2026-05-01", text="body")])
    ok("LLM reader parses structured output",
       fr.summary == "Steady quarter." and fr.red_flags == ["Litigation disclosed"] and fr.catalysts == [])
    ok("LLM reader clamps sentiment to [-1,1]", fr.sentiment == 1.0)
    ok("LLM reader tags its source", fr.source.startswith("llm:"))
    ok("LLM reader disabled without ANTHROPIC key", make_filing_reader(Keys()) is None)

    # Long-term should be taxed at a lower rate than short-term for same profile.
    st = next(h for h in r.horizons if "<1y" in h.label)
    lt = next(h for h in r.horizons if ">1y" in h.label)
    ok("LTCG rate < STCG rate", lt.tax_rate < st.tax_rate)

    passed = sum(1 for _, c in checks if c)
    print(json.dumps({
        "passed": passed, "total": len(checks),
        "all_pass": passed == len(checks),
        "checks": [{"name": n, "pass": c} for n, c in checks],
    }, indent=2))
    return 0 if passed == len(checks) else 1


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0].lower() == "selftest":
        return _selftest()
    if argv and argv[0].lower() == "keys":
        from .config import status
        print(json.dumps(status(), indent=2))
        return 0

    ap = argparse.ArgumentParser(prog="alphadesk")
    ap.add_argument("ticker")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of text")
    ap.add_argument("--sample", action="store_true", help="force offline sample data")
    ap.add_argument("--bracket", type=float, help="ordinary marginal rate, e.g. 0.32")
    ap.add_argument("--ltcg", type=float, help="long-term cap-gains rate, e.g. 0.15")
    ap.add_argument("--state", type=float, help="flat state rate, e.g. 0.05")
    ap.add_argument("--no-niit", action="store_true", help="disable 3.8%% NIIT surtax")
    ap.add_argument("--no-llm", action="store_true",
                    help="skip the Claude filing reader even if ANTHROPIC_API_KEY is set")
    args = ap.parse_args(argv)

    provider = make_provider(force_sample=args.sample)
    # Use the Claude filing reader when an Anthropic key is configured (and the
    # SDK is installed); otherwise the engine falls back to the heuristic scan.
    filing_llm = None
    if not args.no_llm and not args.sample:
        from .llm_reader import make_filing_reader
        filing_llm = make_filing_reader()
    report = analyze(args.ticker, provider=provider, tax=_build_tax(args), filing_llm=filing_llm)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(render_text(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
