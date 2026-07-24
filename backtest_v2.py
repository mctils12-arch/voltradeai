#!/usr/bin/env python3
"""
backtest_v2.py — the real backtest engine (rebuild of the un-ported original).

Replaces the backtest.py stub's missing core. Simulates the bot's EQUITY
strategy logic (strategies/momentum.py, strategies/mean_reversion.py) on daily
bars with the SAME regime gating the live system uses, and emits both the
bot.ts contract fields (strategy/sharpe/totalReturn/maxDrawdown) and the
backtest_10yr_results.json trade/metrics schema.

Design rules (CLAUDE.md REASONING STANDARD):
  - No lookahead: signals computed on close[i], entries at open[i+1].
  - Costs first: liquidity_cost_pct() applied per side on every fill, tiered
    by trailing volume — see REASONING STANDARD #6 (costs/frictions must
    reflect the liquidity actually available, not a flat number that is
    fiction for a thin name).
  - Regime-conditioned: classify_regime_5level from regime_util (same code
    path as live); when VXX history is unavailable the engine degrades to
    vxx_ratio=1.0 exactly like macro_data.py and flags data_quality.
  - Honest scope: options/squeeze legs are NOT simulated (no historical
    options / short-interest data — see research/wishlist.md). They return
    zero-trade entries with a note instead of fabricated results.

Data: Alpaca-first via analyze._fetch_alpaca_bars (same helper as live) when
ALPACA_KEY/ALPACA_SECRET are set; otherwise Yahoo daily bars (stdlib urllib).
Bars are injectable for offline tests.

Invoked by backtest.py (see bot.ts contract): JSON to stdout, exit 0 always.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from datetime import datetime, timedelta, timezone

# Same classifier the live system uses (system_config.get_market_regime
# delegates here). Import guarded so the engine still runs if moved.
try:
    from regime_util import classify_regime_5level
except Exception:  # pragma: no cover - regime_util is in-repo
    classify_regime_5level = None

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".bt_cache")
# MEASUREMENT INTEGRITY (2026-07-23): COST_PCT used to be applied flat to
# EVERY ticker regardless of trading volume — realistic for a liquid
# mega-cap/ETF like SPY, fiction for a thin small/micro-cap (open_questions.md
# "Options fill realism" entry named the same failure mode for options; this
# is its equity-side sibling, and the live system's own EDGE DOCTRINE #2
# explicitly points research at small/illiquid names, which this flat cost
# would silently under-charge). liquidity_cost_pct() below tiers the per-side
# cost by trailing volume, mirroring the four share-volume breakpoints
# server/bot.ts's live fill-tracking haircut already uses (>20M/>5M/>1M/else)
# — but with the MIDPOINT of each live tier's range instead of live's random
# jitter, because a backtest must return the identical result on every run
# for PROMOTION RULE 3's Sharpe/max-drawdown comparison across code changes;
# a random cost would make that comparison meaningless.
COST_PCT = 0.0005          # fallback only (no volume history) — see below
STARTING_CAPITAL = 100_000.0
TRADING_DAYS = 252

# Share-volume breakpoint -> per-side cost (mirrors server/bot.ts's
# fill-tracking tiers: >20M shares/day 0.02-0.05%, >5M 0.05-0.10%,
# >1M 0.08-0.15%, else 0.12-0.25% — values here are each range's midpoint).
_LIQUIDITY_TIERS = (
    (20_000_000, 0.00035),
    (5_000_000, 0.00075),
    (1_000_000, 0.00115),
)
_ILLIQUID_COST_PCT = 0.00185  # <=1M shares/day trailing average


def liquidity_cost_pct(bars: dict, i: int) -> float:
    """Per-side slippage+fee cost at bar i, tiered by the trailing 20-day
    average share volume ending at bar i (inclusive) — no lookahead, since
    the window never reaches past the bar the decision/fill is keyed on.
    Falls back to COST_PCT when there is no volume history at all."""
    vols = bars["volume"]
    window = vols[max(0, i - 19):i + 1]
    if not window:
        return COST_PCT
    avg_vol = sum(window) / len(window)
    for threshold, cost in _LIQUIDITY_TIERS:
        if avg_vol > threshold:
            return cost
    return _ILLIQUID_COST_PCT

# ── Regime → backtest params ─────────────────────────────────────────────────
# Mirrors system_config.get_adaptive_params regime overrides (MAX_POSITIONS /
# MIN_SCORE) in the backtest engine's compact dialect. test_audit_critical.py
# asserts the NEUTRAL line blocks trades, and a consistency test asserts these
# match the live values in system_config.py.
REGIME_PARAMS = {
    "PANIC":   {"max_pos": 0, "min_score": 75, "max_position_pct": 0.06, "time_stop_days": 3},
    "BEAR":    {"max_pos": 0, "min_score": 75, "max_position_pct": 0.08, "time_stop_days": 5},
    "CAUTION": {"max_pos": 4, "min_score": 67, "max_position_pct": 0.10, "time_stop_days": 10},
    "NEUTRAL": {"max_pos": 0, "min_score": 63, "max_position_pct": 0.08, "time_stop_days": 10},
    "BULL":    {"max_pos": 8, "min_score": 63, "max_position_pct": 0.15, "time_stop_days": 10},
}
ATR_STOP_MULT = 1.5        # system_config ATR_STOP_MULTIPLIER
MIN_STOP_PCT = 2.0         # system_config MIN_STOP_PCT
MAX_STOP_PCT = 6.0         # system_config MAX_STOP_PCT
TAKE_PROFIT_RR = 2.0       # target = 2R (asymmetry over accuracy)


def get_adaptive_params(regime: str) -> dict:
    """Backtest-side regime params. NEUTRAL/BEAR/PANIC block new entries with
    max_pos 0 — identical intent to system_config.get_adaptive_params."""
    return dict(REGIME_PARAMS.get(regime, REGIME_PARAMS["NEUTRAL"]))


# ── Data ─────────────────────────────────────────────────────────────────────
def _yahoo_bars(symbol: str, days: int) -> dict:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
           f"?period1={int(start.timestamp())}&period2={int(end.timestamp())}"
           f"&interval=1d&events=div%2Csplit")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    last_err = None
    for attempt in range(3):
        try:
            raw = json.loads(urllib.request.urlopen(req, timeout=20).read())
            break
        except Exception as e:  # transient network / rate limit
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    else:
        raise RuntimeError(f"yahoo fetch failed for {symbol}: {last_err}")
    res = raw["chart"]["result"][0]
    stamps = res["timestamp"]
    q = res["indicators"]["quote"][0]
    adj = (res["indicators"].get("adjclose", [{}])[0].get("adjclose")) or q["close"]
    out = {"date": [], "open": [], "high": [], "low": [], "close": [], "volume": []}
    for i, t in enumerate(stamps):
        if q["close"][i] is None or q["open"][i] is None:
            continue
        # scale OHLC by the adjusted/raw close factor so splits/dividends
        # don't fabricate gaps (lookahead/survivorship guard #7)
        f = (adj[i] / q["close"][i]) if (adj[i] and q["close"][i]) else 1.0
        out["date"].append(datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d"))
        out["open"].append(round(q["open"][i] * f, 4))
        out["high"].append(round(q["high"][i] * f, 4))
        out["low"].append(round(q["low"][i] * f, 4))
        out["close"].append(round(adj[i] if adj[i] else q["close"][i], 4))
        out["volume"].append(q["volume"][i] or 0)
    return out


def _alpaca_bars(symbol: str, days: int) -> dict | None:
    """Same data path as live (analyze._fetch_alpaca_bars). Returns None when
    keys are absent, the import is unavailable, or the fetch comes back empty."""
    if not (os.environ.get("ALPACA_KEY") and os.environ.get("ALPACA_SECRET")):
        return None
    try:
        from analyze import _fetch_alpaca_bars
        df = _fetch_alpaca_bars(symbol, days=days)
        if df is None or len(df) == 0:
            return None
        return {
            "date": [d.strftime("%Y-%m-%d") for d in df.index],
            "open": [float(x) for x in df["Open"]],
            "high": [float(x) for x in df["High"]],
            "low": [float(x) for x in df["Low"]],
            "close": [float(x) for x in df["Close"]],
            "volume": [float(x) for x in df["Volume"]],
        }
    except Exception:
        return None


def fetch_bars(symbol: str, days: int, use_cache: bool = True) -> dict:
    """Alpaca-first (identical data to live), Yahoo fallback, disk-cached."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cp = os.path.join(CACHE_DIR, f"bt2_{symbol}_{days}.json")
    if use_cache and os.path.exists(cp) and time.time() - os.path.getmtime(cp) < 86400:
        with open(cp) as f:
            return json.load(f)
    bars = _alpaca_bars(symbol, days) or _yahoo_bars(symbol, days)
    if use_cache:
        with open(cp, "w") as f:
            json.dump(bars, f)
    return bars


# ── Indicators (plain python, no pandas needed at runtime) ──────────────────
def sma(vals, i, win):
    if i + 1 < win:
        return None
    return sum(vals[i + 1 - win:i + 1]) / win


def rsi14(closes, i, period=14):
    if i < period:
        return None
    gains = losses = 0.0
    for j in range(i - period + 1, i + 1):
        d = closes[j] - closes[j - 1]
        if d >= 0:
            gains += d
        else:
            losses -= d
    if losses == 0:
        return 100.0
    rs = (gains / period) / (losses / period)
    return 100 - 100 / (1 + rs)


def atr14(bars, i, period=14):
    if i < period:
        return None
    trs = []
    for j in range(i - period + 1, i + 1):
        h, l, pc = bars["high"][j], bars["low"][j], bars["close"][j - 1]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return sum(trs) / period


# ── Scoring: the bot's ACTUAL strategy modules ──────────────────────────────
def score_candidate(strategy: str, bars: dict, i: int) -> int:
    closes, vols = bars["close"], bars["volume"]
    if strategy == "momentum":
        if i < 252:
            return 0
        from strategies import momentum
        mom_12_1 = (closes[i - 21] / closes[i - 252] - 1) * 100
        mom_1m = (closes[i] / closes[i - 21] - 1) * 100
        avg_vol = sum(vols[i - 20:i]) / 20 if i >= 20 else 0
        return momentum.score(mom_12_1, mom_1m, avg_vol)["score"]
    if strategy == "mean_reversion":
        if i < 20:
            return 0
        from strategies import mean_reversion
        rsi = rsi14(closes, i)
        chg5 = (closes[i] / closes[i - 5] - 1) * 100 if i >= 5 else 0
        avg_vol = sum(vols[i - 20:i]) / 20
        vr = (vols[i] / avg_vol) if avg_vol else 1.0
        return mean_reversion.score(rsi, chg5, vr)["score"]
    return 0


# ── Metrics ──────────────────────────────────────────────────────────────────
def compute_metrics(equity: list, n_days: int) -> dict:
    if len(equity) < 2 or equity[0] <= 0:
        return {"total_return_pct": 0.0, "cagr_pct": 0.0, "sharpe": 0.0,
                "max_drawdown_pct": 0.0}
    total = equity[-1] / equity[0] - 1
    years = max(n_days / TRADING_DAYS, 1e-9)
    cagr = (equity[-1] / equity[0]) ** (1 / years) - 1
    rets = [equity[i] / equity[i - 1] - 1 for i in range(1, len(equity))]
    mean = sum(rets) / len(rets)
    sd = (sum((r - mean) ** 2 for r in rets) / len(rets)) ** 0.5
    sharpe = (mean / sd) * (TRADING_DAYS ** 0.5) if sd > 0 else 0.0
    peak, maxdd = equity[0], 0.0
    for v in equity:
        peak = max(peak, v)
        maxdd = min(maxdd, v / peak - 1)
    return {"total_return_pct": round(total * 100, 2),
            "cagr_pct": round(cagr * 100, 2),
            "sharpe": round(sharpe, 3),
            "max_drawdown_pct": round(abs(maxdd) * 100, 2)}


# ── Regime series from SPY + VXX (same inputs as live macro_data) ───────────
def regime_series(spy: dict, vxx: dict | None) -> tuple[list, str]:
    """Per-SPY-day regime labels. vxx_ratio = VXX close / VXX 30d avg
    (macro_data.py definition); 1.0 (degraded) when VXX data is missing."""
    vxx_by_date = {}
    if vxx:
        vc = vxx["close"]
        for i, d in enumerate(vxx["date"]):
            avg30 = sma(vc, i, 30)
            vxx_by_date[d] = (vc[i] / avg30) if avg30 else 1.0
    quality = "ok" if vxx_by_date else "degraded"

    labels = []
    below_200_run = 0
    sc = spy["close"]
    for i, d in enumerate(spy["date"]):
        ma50 = sma(sc, i, 50)
        ma200 = sma(sc, i, 200)
        spy_vs_ma50 = (sc[i] / ma50) if ma50 else 1.0
        above200 = bool(ma200 and sc[i] > ma200)
        below_200_run = 0 if above200 else below_200_run + 1
        vr = vxx_by_date.get(d, 1.0)
        if classify_regime_5level is not None:
            labels.append(classify_regime_5level(vr, spy_vs_ma50,
                                                 spy_below_200_days=below_200_run,
                                                 spy_above_200d=above200))
        else:  # minimal inline fallback mirroring regime_util thresholds
            labels.append("BULL" if (vr <= 0.90 and spy_vs_ma50 >= 1.0 and above200)
                          else "NEUTRAL")
    return labels, quality


# ── Core simulation ──────────────────────────────────────────────────────────
def simulate(ticker: str, strategy: str, bars: dict, spy: dict,
             vxx: dict | None) -> dict:
    """Daily-bar simulation. Signal on close[i] -> entry at open[i+1] (no
    lookahead). Exits: ATR stop / 2R take-profit / regime time stop."""
    regimes, data_quality = regime_series(spy, vxx)
    regime_by_date = dict(zip(spy["date"], regimes))

    equity = [STARTING_CAPITAL]
    cash = STARTING_CAPITAL
    pos = None            # {shares, entry, stop, target, entry_i, score, regime}
    trades = []
    n = len(bars["date"])

    for i in range(n - 1):   # signals through n-2; entries fill at i+1's open
        d = bars["date"][i]
        closes = bars["close"]
        regime = regime_by_date.get(d, "NEUTRAL")
        params = get_adaptive_params(regime)

        # mark-to-market
        mtm = cash + (pos["shares"] * closes[i] if pos else 0.0)
        equity.append(mtm)

        # exits (checked on bar i using its close; filled at close with cost)
        if pos:
            px = closes[i]
            held = i - pos["entry_i"]
            reason = None
            if px <= pos["stop"]:
                reason = "stop_loss"
            elif px >= pos["target"]:
                reason = "take_profit"
            elif held >= pos["time_stop"]:
                reason = "time_stop"
            if reason:
                fill = px * (1 - liquidity_cost_pct(bars, i))
                cash += pos["shares"] * fill
                pnl = pos["shares"] * (fill - pos["entry"])
                trades.append({
                    "date_entry": bars["date"][pos["entry_i"]],
                    "date_exit": d, "ticker": ticker,
                    "instrument": "etf" if ticker in ("SPY", "QQQ") else "stock",
                    "strategy": strategy,
                    "entry_price": round(pos["entry"], 2),
                    "exit_price": round(fill, 2),
                    "holding_days": held,
                    "pnl": round(pnl, 2),
                    "net_pct": round((fill / pos["entry"] - 1) * 100, 2),
                    "exit_reason": reason, "score": pos["score"],
                    "vxx_ratio": pos["vxx_ratio"], "vrp": 0.0,
                })
                pos = None

        # entries: regime gate + score gate, fill next open
        if pos is None and params["max_pos"] > 0:
            score = score_candidate(strategy, bars, i)
            if score >= params["min_score"]:
                entry = bars["open"][i + 1] * (1 + liquidity_cost_pct(bars, i))
                atr = atr14(bars, i) or (entry * 0.02)
                stop_pct = min(max(ATR_STOP_MULT * atr / entry * 100,
                                   MIN_STOP_PCT), MAX_STOP_PCT)
                size = equity[-1] * params["max_position_pct"]
                shares = int(size / entry)
                if shares > 0:
                    cash -= shares * entry
                    # vxx_ratio at signal time, for the trade record
                    vr = 1.0
                    if vxx:
                        vd = {dd: j for j, dd in enumerate(vxx["date"])}
                        j = vd.get(d)
                        if j is not None:
                            a30 = sma(vxx["close"], j, 30)
                            vr = round(vxx["close"][j] / a30, 4) if a30 else 1.0
                    pos = {"shares": shares, "entry": entry,
                           "stop": entry * (1 - stop_pct / 100),
                           "target": entry * (1 + TAKE_PROFIT_RR * stop_pct / 100),
                           "entry_i": i + 1, "score": score,
                           "time_stop": params["time_stop_days"],
                           "vxx_ratio": vr}

    # close any open position at the last bar (end_of_backtest)
    if pos:
        fill = bars["close"][n - 1] * (1 - liquidity_cost_pct(bars, n - 1))
        cash += pos["shares"] * fill
        trades.append({
            "date_entry": bars["date"][pos["entry_i"]],
            "date_exit": bars["date"][n - 1], "ticker": ticker,
            "instrument": "etf" if ticker in ("SPY", "QQQ") else "stock",
            "strategy": strategy,
            "entry_price": round(pos["entry"], 2), "exit_price": round(fill, 2),
            "holding_days": n - 1 - pos["entry_i"],
            "pnl": round(pos["shares"] * (fill - pos["entry"]), 2),
            "net_pct": round((fill / pos["entry"] - 1) * 100, 2),
            "exit_reason": "end_of_backtest", "score": pos["score"],
            "vxx_ratio": pos["vxx_ratio"], "vrp": 0.0,
        })
    equity.append(cash)

    m = compute_metrics(equity, n)
    wins = sum(1 for t in trades if t["pnl"] > 0)
    return {
        # bot.ts scheduler contract (camelCase)
        "strategy": strategy,
        "sharpe": m["sharpe"],
        "totalReturn": m["total_return_pct"],
        "maxDrawdown": m["max_drawdown_pct"],
        # backtest_10yr_results.json per-config schema
        "config_name": strategy,
        "final_capital": round(equity[-1], 2),
        "total_return_pct": m["total_return_pct"],
        "cagr_pct": m["cagr_pct"],
        "max_drawdown_pct": m["max_drawdown_pct"],
        "n_trades": len(trades),
        "win_rate_pct": round(100 * wins / len(trades), 1) if trades else 0.0,
        "data_quality": data_quality,
        "trades": trades,
    }


def run_backtest(ticker: str = "SPY", strategy: str = "all", years: float = 3,
                 bars: dict | None = None, spy: dict | None = None,
                 vxx: dict | None = None) -> dict:
    """Entry point. bars/spy/vxx are injectable for offline tests."""
    ticker = ticker.upper()
    years = max(0.5, min(float(years), 10))
    days = int(years * 365) + 400   # warmup for 252d momentum + 200d MA

    # Injected bars => fully hermetic run (tests): never auto-fetch companions.
    offline = bars is not None
    if bars is None:
        bars = fetch_bars(ticker, days)
    if spy is None:
        spy = bars if (offline or ticker == "SPY") else fetch_bars("SPY", days)
    if vxx is None and not offline:
        try:
            vxx = fetch_bars("VXX", days)
        except Exception:
            vxx = None   # degrade exactly like macro_data (vxx_ratio=1.0)

    strategies = (["momentum", "mean_reversion"] if strategy == "all"
                  else [strategy])
    results, all_trades = [], {}
    for s in strategies:
        if s in ("momentum", "mean_reversion"):
            r = simulate(ticker, s, bars, spy, vxx)
            all_trades[s] = r.pop("trades")
            results.append(r)
        elif s == "squeeze":
            results.append({
                "strategy": "squeeze", "sharpe": 0.0, "totalReturn": 0.0,
                "maxDrawdown": 0.0, "config_name": "squeeze", "n_trades": 0,
                "win_rate_pct": 0.0, "final_capital": STARTING_CAPITAL,
                "total_return_pct": 0.0, "cagr_pct": 0.0,
                "max_drawdown_pct": 0.0, "data_quality": "unavailable",
                "note": ("not simulated: requires historical short-interest "
                         "data (research/wishlist.md) — refusing to fabricate"),
            })
            all_trades["squeeze"] = []
        else:
            results.append({"strategy": s, "sharpe": 0.0, "totalReturn": 0.0,
                            "maxDrawdown": 0.0, "n_trades": 0,
                            "note": f"unknown strategy '{s}'"})

    # SPY buy & hold benchmark over the same window (base rate, check #3)
    spy_closes = spy["close"]
    spy_bench = compute_metrics(spy_closes, len(spy_closes))
    for r in results:
        if "cagr_pct" in r:
            r["spy_cagr"] = spy_bench["cagr_pct"]
            r["alpha"] = round(r["cagr_pct"] - spy_bench["cagr_pct"], 2)

    return {
        "ok": True,
        "engine": "backtest_v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "request": {"ticker": ticker, "strategy": strategy, "years": years},
        "period_start": bars["date"][0],
        "period_end": bars["date"][-1],
        "trading_days": len(bars["date"]),
        "starting_capital": STARTING_CAPITAL,
        "spy_total_return": spy_bench["total_return_pct"],
        "spy_cagr": spy_bench["cagr_pct"],
        "results": results,
        "all_trades": all_trades,
    }
