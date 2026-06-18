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
    args = ap.parse_args(argv)

    provider = make_provider(force_sample=args.sample)
    report = analyze(args.ticker, provider=provider, tax=_build_tax(args))
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(render_text(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
