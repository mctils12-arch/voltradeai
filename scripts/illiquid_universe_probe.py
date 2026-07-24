#!/usr/bin/env python3
"""
illiquid_universe_probe.py — EDGE DOCTRINE #2 ("fish where whales can't")
axis-(b) research probe: does the bot's EXISTING momentum/mean_reversion
strategy logic show a differential edge in a genuinely illiquid small-cap
corner vs. the liquid mega-cap names the live bot actually trades?

Gated on the 2026-07-23 fix (v1.0.480, PR #588): backtest_v2.py's fill
cost was a flat 5bps for every ticker regardless of volume — fiction for
a thin name. liquidity_cost_pct() now tiers cost by trailing 20d share
volume, so this is the first session where an illiquid-universe backtest
result isn't simulator fiction (see research/experiments.md 2026-07-23
[RULE-REVIEW] entry).

PRIOR, stated 2026-07-24 BEFORE running (Reasoning Standard #10): EDGE
DOCTRINE #2 predicts illiquid names should show MORE exploitable
structure because whale capital can't fit. Second-order-thinking
counter-prior: momentum.py/mean_reversion.py were developed and tuned
against the bot's actual live universe, which skews liquid/mega-cap
(tiered_strategy.T1_TICKERS_FALLBACK) — so the unmodified strategy LOGIC
was expected to transfer poorly to microcap dynamics (dilution, gap
risk, thin analyst coverage) even if the underlying capacity-constrained
opportunity is structurally real. Expected: illiquid group underperforms
liquid group on BOTH strategies, unmodified.

UNIVERSE CONSTRUCTION (documented, not guessed from memory/training
data — CLAUDE.md READ BEFORE WRITE / no-fabrication discipline):
  - Candidate pool: nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt
    (live NASDAQ symbol directory, fetched 2026-07-24), filtered to
    Market Category "S" (Nasdaq Capital Market — the smallest listing
    tier, a real structural proxy for "small cap" that needs no
    market-cap data field), ETF=N, Test Issue=N, common-stock names
    only (excludes Warrant/Right/Unit/Acquisition/Preferred/Depositary
    security-name substrings and multi-class ticker suffixes).
  - Systematic sample: every 40th row of that filtered list (1,334
    names -> 34 candidates) — not cherry-picked.
  - Screened via backtest_v2._yahoo_bars for >=500 trading days of
    history; the resulting trailing-252d average share volume was then
    bucketed using backtest_v2.py's OWN existing liquidity_cost_pct()
    tier boundaries (>5M / 1-5M / <=1M shares/day) so the tier labels
    here are literally the tiers the cost model already prices.
  - LIQUID comparison group: 7 names from tiered_strategy.
    T1_TICKERS_FALLBACK (the live bot's own liquid anchor universe),
    excluding its 3 ETF members (SPY/QQQ/IWM, since this probe is
    single-name vs. single-name) — a mega-cap tech/industrial mix
    (AAPL, MSFT, NVDA, AMD, AMZN, CAT, GE), not the fallback's full
    17-name list.

The 34-candidate screen (frozen here for reproducibility — re-running
the live NASDAQ directory fetch on a later date will return a
DIFFERENT candidate pool, since Nasdaq listings change daily):
  ILLIQUID (<=1M shares/day trailing 252d): AXG, CISO, DYAI, EPOW,
    GALT, NRXP, PROF, SNOA, TRAW, WNW
  MODERATE (1-5M shares/day):               CRDF, IMNM, KTTA, ONCY,
    SXTP, VIVO, ZCMD
  Screened out (>5M shares/day, insufficient history, or fetch error):
    ABAT, APPS, BMHL, BYFC, DCX, FIEE, GROW, HOLO, IVF, KTTA(kept
    above), LSTA, MGN, NAGE, PCYO, RAYA, ROC, SGLY, SXTP(kept above)

HONEST LIMITATIONS (state before anyone reuses this result):
  1. SURVIVORSHIP: the candidate pool is TODAY's live NASDAQ listing —
     any illiquid name that fully delisted inside the study window is
     invisible here. Group mean buy-and-hold was still -74% to -75%
     over 4y (see MEASURED RESULTS below), i.e. brutal even among
     survivors; true population reality (including delistings-to-zero)
     is almost certainly worse, not better.
  2. SINGLE SAMPLE DRAW: one systematic sample of 10 illiquid / 7
     moderate names, no independent re-draw, no formal significance
     test (t-test/bootstrap) run this session — Sharpe means below are
     descriptive, not confirmed-significant. Per Reasoning Standard #4,
     discount accordingly.
  3. NO OUT-OF-SAMPLE SPLIT: the entire 4y window was used for both
     "discovering" the pattern and reporting it — classic in-sample
     read. A follow-up MUST split train/test before this could move
     toward the LOGIC validation-ladder gate.
  4. NOT a new "data root": momentum.py/mean_reversion.py already exist
     and already cleared their own LOGIC gate on the liquid universe
     historically — this probe asks whether that already-built logic
     generalizes to a new market-cap segment, not whether a new data
     pipeline is trustworthy. It does not itself grant any ladder gate.

MEASURED RESULTS (this run, 2026-07-24, years=4, engine=backtest_v2 post
v1.0.480 fix — see research/experiments.md for the full write-up):
  illiquid (n=10): buy&hold mean -74.7%; momentum mean_sharpe -0.236
    (3/10 positive); mean_reversion mean_sharpe +0.246 (8/10 positive,
    147 total trades, mean win rate 46.0%).
  moderate (n=7):  buy&hold mean -74.6%; momentum mean_sharpe +0.025
    (3/7 positive); mean_reversion mean_sharpe -0.222 (3/7 positive).
  liquid (n=7):    buy&hold mean +375.7%; momentum mean_sharpe +0.559
    (6/7 positive); mean_reversion mean_sharpe +0.319 but only 11 total
    trades across 7 tickers — mega-caps rarely trigger the oversold+
    volume-spike setup in a trending bull tape, so this liquid
    mean_reversion reading has effectively no statistical power and is
    NOT a fair mean_reversion baseline.

INTERPRETATION: prior partially confirmed, partially updated. Momentum
matches the prior (illiquid/moderate clearly worse than liquid) — the
strategy logic likely doesn't transfer, consistent with momentum relying
on sustained institutional/analyst-driven trend continuation that thin
names lack. Mean_reversion did NOT match the prior: it shows a modest
but consistent positive edge specifically in the illiquid tier (best of
the three groups by pos_frac and comparable-or-better mean Sharpe to the
statistically-starved liquid reading), with a plausible structural
story (Reasoning Standard #5: thin order books mean oversold conditions
are retail-panic-driven and mechanically exhaust, without the sustained
institutional selling flow that can keep grinding a liquid name down —
a capacity/mandate-constraint explanation, not "nobody noticed").
Magnitude is modest (mean total_return_pct +4.32% over 4y, ~1%/yr) — a
real but not dramatic signal-layer finding, exactly what open_questions
should hold for further testing, not a threshold change to ship.
"""
import csv
import io
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ILLIQUID = ["AXG", "CISO", "DYAI", "EPOW", "GALT", "NRXP", "PROF", "SNOA", "TRAW", "WNW"]
MODERATE = ["CRDF", "IMNM", "KTTA", "ONCY", "SXTP", "VIVO", "ZCMD"]
LIQUID = ["AAPL", "MSFT", "NVDA", "AMD", "AMZN", "CAT", "GE"]

YEARS = 4
NASDAQ_SYMDIR_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
_EXCLUDE_NAME_SUBSTRINGS = ("Warrant", "Right", "Unit", "Acquisition", "Preferred",
                            "Depositary", "Rights", "Units")


def fetch_nasdaq_capital_market_candidates(sample_stride: int = 40):
    """Live re-derivation of the candidate pool this probe was built from.
    NOT called by the frozen ILLIQUID/MODERATE lists above — those are
    pinned for reproducibility. This exists so a future session can
    re-draw an independent sample instead of reusing this one (see
    LIMITATION #2 in the module docstring)."""
    req = urllib.request.Request(NASDAQ_SYMDIR_URL, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=20).read().decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(raw), delimiter="|")
    clean = []
    for row in reader:
        if row.get("Market Category") != "S":
            continue
        if row.get("ETF") != "N" or row.get("Test Issue") != "N":
            continue
        name = row.get("Security Name", "")
        sym = row.get("Symbol", "")
        if any(b in name for b in _EXCLUDE_NAME_SUBSTRINGS):
            continue
        if "." in sym or len(sym) > 5:
            continue
        clean.append(sym)
    return clean[::sample_stride]


def buy_and_hold_pct(bars: dict) -> float:
    c = bars["close"]
    return round((c[-1] / c[0] - 1) * 100, 2)


def classify_liquidity_tier(avg_vol_252d: float) -> str:
    """Mirrors backtest_v2.liquidity_cost_pct()'s own tier boundaries."""
    if avg_vol_252d > 5_000_000:
        return "liquid_5m_plus"
    if avg_vol_252d > 1_000_000:
        return "moderate"
    return "illiquid"


def summarize(group_results: list) -> dict:
    """Pure aggregation over already-computed per-ticker rows (see
    run_group in __main__) — no network, safe to unit test."""
    out = {}
    bh = [r["buy_and_hold_pct"] for r in group_results if "buy_and_hold_pct" in r]
    out["n"] = len(group_results)
    out["buy_and_hold_mean_pct"] = round(sum(bh) / len(bh), 2) if bh else None
    for strat in ("momentum", "mean_reversion"):
        rows = [r[strat] for r in group_results if strat in r]
        if not rows:
            out[strat] = None
            continue
        sharpes = [r["sharpe"] for r in rows]
        out[strat] = {
            "mean_sharpe": round(sum(sharpes) / len(sharpes), 3),
            "pos_frac": f"{sum(1 for s in sharpes if s > 0)}/{len(sharpes)}",
            "total_trades": sum(r["n_trades"] for r in rows),
            "mean_return_pct": round(sum(r["total_return_pct"] for r in rows) / len(rows), 2),
        }
    return out


def run_group(tickers, years=YEARS):
    from backtest_v2 import run_backtest, fetch_bars
    out = []
    for t in tickers:
        try:
            days = int(years * 365) + 400
            bars = fetch_bars(t, days)
            resp = run_backtest(t, "all", years)
        except Exception as e:
            out.append({"ticker": t, "error": str(e)[:200]})
            continue
        row = {"ticker": t, "buy_and_hold_pct": buy_and_hold_pct(bars), "n_bars": len(bars["close"])}
        for r in resp["results"]:
            if r["strategy"] in ("momentum", "mean_reversion"):
                row[r["strategy"]] = {
                    "sharpe": r["sharpe"],
                    "total_return_pct": r["total_return_pct"],
                    "max_drawdown_pct": r["max_drawdown_pct"],
                    "n_trades": r["n_trades"],
                    "win_rate_pct": r["win_rate_pct"],
                }
        out.append(row)
    return out


if __name__ == "__main__":
    results = {}
    for name, tickers in [("illiquid", ILLIQUID), ("moderate", MODERATE), ("liquid", LIQUID)]:
        print(f"=== {name} ===", file=sys.stderr)
        results[name] = run_group(tickers)
        print(json.dumps(summarize(results[name]), indent=2), file=sys.stderr)
    print(json.dumps({g: summarize(r) for g, r in results.items()}, indent=2))
