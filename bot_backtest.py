#!/usr/bin/env python3
"""
bot_backtest.py — a real, reproducible backtest of the bot's EQUITY strategy logic.

Why this exists: the original backtest engine that produced
backtest_10yr_results.json is not in the repo, and that artifact shows the live
(options-heavy) bot badly lagging SPY (~0.3% CAGR vs ~14%). The options leg can't
be backtested faithfully without historical options prices, and those static
results already pin it as the main drag. So this harness tests the part we CAN
verify honestly: the bot's momentum scoring (strategies/momentum.py) run as a
monthly sector-rotation, with and without a market-regime filter, vs SPY buy&hold.

Data: Yahoo Finance daily adjusted closes (no key, no extra deps). Pure stdlib.
Cached under .bt_cache/ so re-runs are instant.

    python3 bot_backtest.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strategies import momentum  # the bot's actual momentum scoring

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".bt_cache")
os.makedirs(CACHE, exist_ok=True)

# Sector SPDRs + growth/broad — a clean momentum-rotation universe with full
# history since 2015 (avoids single-stock survivorship bias).
UNIVERSE = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "XLRE", "QQQ", "SPY"]
BENCH = "SPY"
START = "2016-01-01"
END = "2026-04-07"      # match the static artifact's window
COST = 0.0005           # 5 bps per position change (round-trip ~ realistic)


# ── data ──────────────────────────────────────────────────────────────────
def _ts(d: str) -> int:
    return int(datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def fetch(ticker: str) -> list[tuple[str, float]]:
    """Return [(YYYY-MM-DD, adjclose)] for the full window, cached."""
    cp = os.path.join(CACHE, f"{ticker}.json")
    if os.path.exists(cp):
        with open(cp) as f:
            return [(d, c) for d, c in json.load(f)]
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
           f"?period1={_ts(START)}&period2={_ts(END)}&interval=1d&events=div%2Csplit")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(4):
        try:
            raw = json.loads(urllib.request.urlopen(req, timeout=20).read())
            break
        except Exception as e:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))
    res = raw["chart"]["result"][0]
    stamps = res["timestamp"]
    quote = res["indicators"]["quote"][0]
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose") or quote["close"]
    out = []
    for i, t in enumerate(stamps):
        c = adj[i] if adj[i] is not None else quote["close"][i]
        if c is None:
            continue
        out.append((datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d"), round(float(c), 4)))
    with open(cp, "w") as f:
        json.dump(out, f)
    return out


# ── metrics ───────────────────────────────────────────────────────────────
def metrics(equity: list[float], dates: list[str]) -> dict:
    if len(equity) < 2:
        return {}
    rets = [equity[i] / equity[i - 1] - 1 for i in range(1, len(equity))]
    n = len(equity)
    years = (datetime.strptime(dates[-1], "%Y-%m-%d") - datetime.strptime(dates[0], "%Y-%m-%d")).days / 365.25
    total = equity[-1] / equity[0] - 1
    cagr = (equity[-1] / equity[0]) ** (1 / years) - 1 if years > 0 else 0.0
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / len(rets)
    sd = var ** 0.5
    sharpe = (mean / sd) * (252 ** 0.5) if sd > 0 else 0.0
    peak = equity[0]
    maxdd = 0.0
    for v in equity:
        peak = max(peak, v)
        maxdd = min(maxdd, v / peak - 1)
    return {"total_return_pct": round(total * 100, 2), "cagr_pct": round(cagr * 100, 2),
            "sharpe": round(sharpe, 3), "max_drawdown_pct": round(maxdd * 100, 2)}


# ── backtest ──────────────────────────────────────────────────────────────
def backtest(data: dict, dates: list[str], *, top_n: int, min_score: int,
             regime: bool, universe: list[str] | None = None) -> dict:
    """Monthly momentum rotation. Each month-start: score every name with the
    bot's momentum.score, hold the top_n with score>=min_score (equal weight).
    Optional regime filter: only hold when SPY is above its 200d MA, else cash."""
    pool = universe or list(data.keys())
    idx = {t: {d: i for i, (d, _) in enumerate(zip(dates, [data[t].get(d) for d in dates]))} for t in data}
    px = data  # px[ticker][date] -> adjclose or None
    spy = px[BENCH]

    equity = [100000.0]
    eq_dates = [dates[0]]
    held: list[str] = []
    wins = trades = 0
    entry_px: dict[str, float] = {}

    def price(t, d):
        return px[t].get(d)

    def sma(series_t, d_i, win):
        vals = [spy[dates[j]] for j in range(max(0, d_i - win + 1), d_i + 1) if spy.get(dates[j]) is not None]
        return sum(vals) / len(vals) if len(vals) >= win else None

    month = None
    for di in range(252, len(dates)):  # need 252d history for 12-1 momentum
        d = dates[di]
        # daily mark-to-market of current holdings
        if held:
            rs = []
            for t in held:
                p0, p1 = price(t, dates[di - 1]), price(t, d)
                if p0 and p1:
                    rs.append(p1 / p0 - 1)
            equity.append(equity[-1] * (1 + (sum(rs) / len(rs) if rs else 0.0)))
        else:
            equity.append(equity[-1])
        eq_dates.append(d)

        m = d[:7]
        if m == month:
            continue
        month = m  # rebalance at each new month

        # regime: SPY above 200d MA?
        risk_on = True
        if regime:
            s = sma(spy, di, 200)
            risk_on = bool(s and spy.get(d) and spy[d] > s)

        # score everyone with the bot's momentum logic
        ranked = []
        if risk_on:
            for t in pool:
                p_now, p_1m, p_12m = price(t, d), price(t, dates[di - 21]), price(t, dates[di - 252])
                if not (p_now and p_1m and p_12m):
                    continue
                mom_12_1 = (p_1m / p_12m - 1) * 100
                mom_1m = (p_now / p_1m - 1) * 100
                r = momentum.score(mom_12_1, mom_1m, 5_000_000)
                if r["score"] >= min_score:
                    ranked.append((r["score"], mom_12_1, t))
            ranked.sort(reverse=True)
        new_held = [t for _, _, t in ranked[:top_n]]

        # tally win/loss on names being exited + apply turnover cost
        exiting = set(held) - set(new_held)
        for t in exiting:
            if t in entry_px and price(t, d):
                trades += 1
                if price(t, d) > entry_px[t]:
                    wins += 1
        turnover = len(set(new_held) ^ set(held))
        if turnover:
            equity[-1] *= (1 - COST * turnover / max(1, len(new_held) or 1))
        for t in set(new_held) - set(held):
            if price(t, d):
                entry_px[t] = price(t, d)
        held = new_held

    out = metrics(equity, eq_dates)
    out["n_rebalanced_trades"] = trades
    out["win_rate_pct"] = round(100 * wins / trades, 1) if trades else 0.0
    out["avg_positions"] = top_n
    return out


def main():
    print(f"Fetching {len(UNIVERSE)} tickers …", file=sys.stderr)
    data = {}
    for t in UNIVERSE:
        try:
            data[t] = dict(fetch(t))
            print(f"  {t}: {len(data[t])} bars", file=sys.stderr)
        except Exception as e:
            print(f"  {t}: FAILED {e}", file=sys.stderr)
    # common date axis = SPY's
    dates = [d for d, _ in sorted([(d, c) for d, c in data[BENCH].items()])]

    # SPY buy & hold benchmark
    spy_eq = [data[BENCH][d] for d in dates if data[BENCH].get(d)]
    bench = metrics(spy_eq, [d for d in dates if data[BENCH].get(d)])

    # QQQ buy & hold reference (pure tech beta, not a strategy)
    qqq_eq = [data["QQQ"][d] for d in dates if data["QQQ"].get(d)]
    qqq_bh = metrics(qqq_eq, [d for d in dates if data["QQQ"].get(d)])

    configs = [
        ("Momentum top3 (score>=50)",        dict(top_n=3, min_score=50, regime=False)),
        ("Momentum top4 (score>=50)",        dict(top_n=4, min_score=50, regime=False)),
        ("Momentum top3 + 200dma regime",    dict(top_n=3, min_score=50, regime=True)),
        ("Momentum top1 (winner-take-all)",  dict(top_n=1, min_score=50, regime=False)),
        ("Momentum top1 + 200dma regime",    dict(top_n=1, min_score=50, regime=True)),
        ("Dual-momentum SPY/QQQ",            dict(top_n=1, min_score=0, regime=False, universe=["SPY", "QQQ"])),
        ("Dual-momentum SPY/QQQ + regime",   dict(top_n=1, min_score=0, regime=True, universe=["SPY", "QQQ"])),
    ]
    results = {"SPY buy & hold": bench, "QQQ buy & hold": qqq_bh}
    for name, cfg in configs:
        results[name] = backtest(data, dates, **cfg)

    # report
    print(f"\nBacktest {dates[0]} → {dates[-1]}  ({len(dates)} days)\n")
    hdr = f"{'Strategy':<34}{'CAGR%':>8}{'Total%':>9}{'Sharpe':>8}{'MaxDD%':>8}{'Win%':>7}{'AlphaVsSPY':>11}"
    print(hdr); print("-" * len(hdr))
    spy_cagr = bench["cagr_pct"]
    for name, r in results.items():
        if not r:
            continue
        alpha = "" if name.startswith("SPY") else f"{r['cagr_pct'] - spy_cagr:+.2f}"
        print(f"{name:<34}{r['cagr_pct']:>8}{r['total_return_pct']:>9}{r.get('sharpe',0):>8}"
              f"{r.get('max_drawdown_pct',0):>8}{r.get('win_rate_pct',''):>7}{alpha:>11}")
    print("\nReference — static artifact (live options-heavy bot): "
          "OLD CAGR 0.27%, NEW CAGR -0.27%, SPY CAGR 13.97%.")
    if "--json" in sys.argv:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
