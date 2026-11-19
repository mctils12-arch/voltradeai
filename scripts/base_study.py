#!/usr/bin/env python3
"""
base_study.py — the pre-registered base study (2026-11-19).

Owner: "why not hold the top names or the best names in the market and build a
better base of the s&p ... and why hold that much cash on the side".

The menu, priors, assumptions and decision rules A/B/C are fixed in
research/base_study_2026-11-19.md (§ Pre-registration, committed before any
price was fetched). This script implements exactly those and nothing else.

Reused, not re-invented:
- scripts/floor_study.py: the Yahoo loader (price-only or adjusted), daily_rf,
  returns, metrics, worst_year, rolling_beat, the cost / financing / fee
  constants.
- floor_vol_managed.plan() / cash_plan(): the live floor's own planning code,
  with system_config FLOOR_VM_* values.
- signal_attribution.bonferroni_z, source_validation.benjamini_hochberg.

Usage:
    python3 scripts/base_study.py --fetch <data dir>     # network: Yahoo chart JSON, once
    python3 scripts/base_study.py <data dir> <out.json>  # deterministic, no network
    python3 scripts/base_study.py --md <out.json>        # the write-up's tables
Timing: every decision at close t uses closes <= t and is held t -> t+1.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import urllib.request
from collections import deque
from datetime import date
from statistics import NormalDist
from typing import Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

_spec = importlib.util.spec_from_file_location("floor_study", os.path.join(HERE, "floor_study.py"))
fs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fs)

import floor_vol_managed as fvm                      # noqa: E402
from signal_attribution import bonferroni_z          # noqa: E402
from source_validation import benjamini_hochberg     # noqa: E402
from system_config import BASE_CONFIG                # noqa: E402

# ── pre-registered constants (research/base_study_2026-11-19.md) ─────────────
END_DAY = "2026-11-18"                       # last session used
FETCH_PERIOD2 = 1795046400                   # 2026-11-19 00:00 UTC: the fetch never reads today's bar
BENCH = "SPY"
LEVERED = "SSO"
CANDIDATES: Dict[str, Dict[str, float]] = {  # name -> target weights
    "OEF": {"OEF": 1.0}, "XLG": {"XLG": 1.0}, "MGK": {"MGK": 1.0},
    "MTUM": {"MTUM": 1.0}, "SPMO": {"SPMO": 1.0},
    "QUAL": {"QUAL": 1.0}, "SPHQ": {"SPHQ": 1.0},
    "MTUM+QUAL": {"MTUM": 0.5, "QUAL": 0.5},
    "RSP": {"RSP": 1.0},
    "USMV": {"USMV": 1.0}, "SPLV": {"SPLV": 1.0},
}
ETFS = ["SPY", "SSO", "OEF", "XLG", "MGK", "MTUM", "SPMO", "QUAL", "SPHQ", "RSP", "USMV", "SPLV"]
GRID = {  # cell -> (vol rule on, cash treatment)
    "B0-i": (False, "cash0"), "B0-ii": (False, "invested"), "B0-iii": (False, "tbill"),
    "B1-i": (True, "cash0"), "B1-ii": (True, "invested"), "B1-iii": (True, "tbill"),
}
FAMILY = list(CANDIDATES) + ["B1-i", "B1-ii", "B1-iii", "B0-i", "B0-iii"]      # the 16 series tested vs SPY
START_EQUITY = 100_000.0
COST_BP = fs.COST_BP                         # 2 bp of traded notional
FUND_FEE = 0.0019                            # the T-bill fund's fee, per year (BOXX)
TAX_SHORT, TAX_LONG = 0.24, 0.15             # server/taxLedger.ts DEFAULT_TAX_RATES
LONG_TERM_DAYS = 365                         # server/taxLedger.ts LONG_TERM_MS: "more than one year"
FAMILY_ALPHA = 0.05
MAX_DD_SLACK_A, MAX_DD_SLACK_B, MAX_DD_SLACK_C_TBILL, MAX_DD_SLACK_C_INVESTED = 5.0, 10.0, 0.5, 5.0
VM_KEYS = ("FLOOR_VM_WINDOW", "FLOOR_VM_TARGET_VOL", "FLOOR_VM_CAP", "FLOOR_VM_BAND", "FLOOR_VM_CAPITAL_PCT",
           "FLOOR_VM_CASH_BAND", "FLOOR_VM_CASH_MIN_TRADE_DOLLARS")
TD = fs.TRADING_DAYS


def live_cfg() -> Dict:
    """The live floor's parameters, read from system_config (never restated here)."""
    return {k: BASE_CONFIG[k] for k in VM_KEYS}


# ── data ─────────────────────────────────────────────────────────────────────
def fetch(out_dir: str) -> None:
    """Yahoo chart API, the endpoint backtest_v2._yahoo_bars uses; raw JSON kept so
    floor_study.load can read PRICE-ONLY closes (fetch_bars' Yahoo path returns
    dividend-adjusted closes). A failed ticker is reported, never substituted."""
    os.makedirs(out_dir, exist_ok=True)
    for sym, name in [(s, s) for s in ETFS] + [("%5EIRX", "IRX")]:
        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
               f"?period1=0&period2={FETCH_PERIOD2}&interval=1d&events=div%2Csplit")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        err: Optional[str] = None
        for attempt in range(3):
            try:
                raw = urllib.request.urlopen(req, timeout=30).read()
                json.loads(raw)["chart"]["result"][0]["timestamp"]
                with open(os.path.join(out_dir, f"h_{name}.json"), "wb") as f:
                    f.write(raw)
                err = None
                break
            except (OSError, ValueError, KeyError, TypeError, IndexError) as e:
                err = f"{type(e).__name__}: {e}"
                time.sleep(2.0 * (attempt + 1))
        print(f"{name}: {'ok' if err is None else 'FAILED ' + err}")
        time.sleep(1.0)


def load_all(src: str) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], Dict[str, float], Dict[str, Dict]]:
    """(price-only closes, adjusted closes, T-bill yield, per-file facts). A missing
    file drops the ticker and is recorded in facts[ticker]['dropped']."""
    px, tr, facts = {}, {}, {}
    for s in ETFS:
        p = os.path.join(src, f"h_{s}.json")
        if not os.path.exists(p):
            facts[s] = {"dropped": "no data file (fetch failed)"}
            continue
        a = {d: v for d, v in fs.load(p, adjusted=False).items() if d <= END_DAY and v > 0}
        b = {d: v for d, v in fs.load(p, adjusted=True).items() if d <= END_DAY and v > 0}
        days = sorted(set(a) & set(b))
        px[s] = {d: a[d] for d in days}
        tr[s] = {d: b[d] for d in days}
        r = [px[s][days[i]] / px[s][days[i - 1]] - 1.0 for i in range(1, len(days))]
        facts[s] = {"first": days[0], "last": days[-1], "sessions": len(days),
                    "max_abs_daily_move_pct": round(max(abs(x) for x in r) * 100, 2),
                    "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]}
    irx = {d: v for d, v in fs.load(os.path.join(src, "h_IRX.json"), adjusted=False).items() if d <= END_DAY}
    return px, tr, irx, facts


def common_days(series: Dict[str, Dict[str, float]], tickers: Sequence[str], start: Optional[str] = None) -> List[str]:
    days = set(series[tickers[0]])
    for t in tickers[1:]:
        days &= set(series[t])
    return sorted(d for d in days if start is None or d >= start)


def synthetic_2x(tr_spy: Dict[str, float], irx: Dict[str, float]) -> Dict[str, float]:
    """A 2x daily fund before SSO existed (floor_study.run_v2's arithmetic):
    2 x SPY total return - (T-bill + FIN_SPREAD) - LEV_ETF_FEE, compounded."""
    days = sorted(tr_spy)
    rf = fs.daily_rf(irx, days)
    out, level = {days[0]: 100.0}, 100.0
    for i in range(1, len(days)):
        r = tr_spy[days[i]] / tr_spy[days[i - 1]] - 1.0
        level *= 1.0 + 2.0 * r - (rf[i] + fs.FIN_SPREAD / TD) - fs.LEV_ETF_FEE / TD
        out[days[i]] = level
    return out


# ── tax (an ESTIMATE; assumptions in the pre-registration) ───────────────────
def _d(day: str) -> date:
    return date(int(day[:4]), int(day[5:7]), int(day[8:10]))


class Lots:
    """FIFO tax lots for one ticker."""

    def __init__(self) -> None:
        self.lots: deque = deque()            # [day, units, cost per unit]
        self.sold_dollars = self.sold_dollar_days = 0.0      # for the report: how old the sold lots were

    def units(self) -> float:
        return sum(l[1] for l in self.lots)

    def buy(self, day: str, units: float, price: float) -> None:
        if units > 0:
            self.lots.append([day, units, price])

    def sell(self, day: str, units: float, price: float) -> Tuple[float, float]:
        """Sell oldest first. Returns (short-term gain, long-term gain)."""
        st = lt = 0.0
        left = min(units, self.units())
        while left > 1e-12 and self.lots:
            lot = self.lots[0]
            take = min(left, lot[1])
            gain = take * (price - lot[2])
            age = (_d(day) - _d(lot[0])).days
            if age > LONG_TERM_DAYS:
                lt += gain
            else:
                st += gain
            self.sold_dollars += take * price
            self.sold_dollar_days += take * price * age
            lot[1] -= take
            left -= take
            if lot[1] <= 1e-12:
                self.lots.popleft()
        return st, lt

    def unrealized(self, day: str, price: float) -> Tuple[float, float]:
        st = lt = 0.0
        for lot_day, units, cost in self.lots:
            gain = units * (price - cost)
            if (_d(day) - _d(lot_day)).days > LONG_TERM_DAYS:
                lt += gain
            else:
                st += gain
        return st, lt


def tax_due(st: float, lt: float, carry: float) -> Tuple[float, float]:
    """One year's tax on net short- and long-term gains. Losses net across the
    two kinds, then a carried loss offsets short-term first; an unused loss is
    carried forward (no refund, no ordinary-income offset). Returns (tax, carry)."""
    if st < 0 < lt:
        off = min(-st, lt)
        st, lt = st + off, lt - off
    elif lt < 0 < st:
        off = min(-lt, st)
        st, lt = st - off, lt + off
    for kind in ("st", "lt"):
        gain = st if kind == "st" else lt
        use = min(carry, max(gain, 0.0))
        carry -= use
        if kind == "st":
            st -= use
        else:
            lt -= use
    tax = TAX_SHORT * max(st, 0.0) + TAX_LONG * max(lt, 0.0)
    return tax, carry + max(-st, 0.0) + max(-lt, 0.0)


class Book:
    """Holdings in units with FIFO lots, a T-bill fund balance, cash at 0%."""

    def __init__(self, tickers: Sequence[str], taxed: bool) -> None:
        self.taxed = taxed
        self.units = {t: 0.0 for t in tickers}
        self.lots = {t: Lots() for t in tickers}
        self.fund = 0.0
        self.cash = START_EQUITY
        self.st = self.lt = self.carry = 0.0
        self.tax_paid = self.cost_paid = self.traded = 0.0
        self.real_st = self.real_lt = self.fund_income = 0.0      # running totals, for the report only

    def value(self, t: str, px: Dict[str, float]) -> float:
        return self.units[t] * px[t]

    def equity(self, px: Dict[str, float]) -> float:
        return sum(self.units[t] * px[t] for t in self.units) + self.fund + self.cash

    def trade(self, t: str, dollars: float, px: Dict[str, float], day: str, cost_bp: float) -> None:
        """dollars > 0 buys, < 0 sells, at the close; the cost comes out of cash."""
        if abs(dollars) < 1e-9:
            return
        price = px[t]
        if dollars > 0:
            self.units[t] += dollars / price
            self.lots[t].buy(day, dollars / price, price)
        else:
            u = min(-dollars / price, self.units[t])
            st, lt = self.lots[t].sell(day, u, price)
            self.st, self.lt = self.st + st, self.lt + lt
            self.real_st, self.real_lt = self.real_st + st, self.real_lt + lt
            self.units[t] -= u
            dollars = -u * price
        cost = abs(dollars) * cost_bp / 1e4
        self.cash -= dollars + cost
        self.cost_paid += cost
        self.traded += abs(dollars)

    def fund_move(self, dollars: float, cost_bp: float) -> None:
        dollars = max(dollars, -self.fund)
        cost = abs(dollars) * cost_bp / 1e4
        self.fund += dollars
        self.cash -= dollars + cost
        self.cost_paid += cost
        self.traded += abs(dollars)

    def accrue_fund(self, rate: float) -> None:
        income = self.fund * rate
        self.fund += income
        self.st += income                     # the fund's return is taxed at the short-term rate
        self.fund_income += income

    def raise_cash(self, need: float, px: Dict[str, float], day: str, cost_bp: float) -> None:
        """Cash first, then the fund, then every holding pro rata."""
        short = need - self.cash
        if short <= 0:
            return
        take = min(short, self.fund)
        if take > 0:
            self.fund_move(-take, cost_bp)
        short = need - self.cash
        held = sum(self.value(t, px) for t in self.units)
        if short > 0 and held > 0:
            frac = min(1.0, short / held * (1.0 + cost_bp / 1e4))
            for t in list(self.units):
                self.trade(t, -frac * self.value(t, px), px, day, cost_bp)

    def settle_year(self, px: Dict[str, float], day: str, cost_bp: float) -> None:
        """Pay last year's tax on the first session of the new year."""
        if not self.taxed:
            self.st = self.lt = 0.0
            return
        tax, self.carry = tax_due(self.st, self.lt, self.carry)
        self.st = self.lt = 0.0
        if tax > 0:
            self.raise_cash(tax, px, day, cost_bp)
            self.cash -= tax
            self.tax_paid += tax

    def terminal(self, px: Dict[str, float], day: str) -> Dict[str, float]:
        """After-tax wealth at the end: 'holding' owes only this year's realized
        gains; 'liquidated' also sells every lot (taxed by its age)."""
        eq = self.equity(px)
        if not self.taxed:
            return {"equity": eq, "holding": eq, "liquidated": eq}
        hold_tax, _ = tax_due(self.st, self.lt, self.carry)
        ust = ult = 0.0
        for t in self.units:
            a, b = self.lots[t].unrealized(day, px[t])
            ust, ult = ust + a, ult + b
        liq_tax, _ = tax_due(self.st + ust, self.lt + ult, self.carry)
        return {"equity": eq, "holding": eq - hold_tax, "liquidated": eq - liq_tax}


# ── simulators ───────────────────────────────────────────────────────────────
def _result(days: Sequence[str], path: Sequence[float], book: Book, px_last: Dict[str, float], extra: Dict) -> Dict:
    rets = [path[i] / path[i - 1] - 1.0 for i in range(1, len(path))]
    years = len(rets) / TD
    term = book.terminal(px_last, days[-1])
    sold = sum(l.sold_dollars for l in book.lots.values())
    out = {"days": list(days[1:]), "rets": rets, "years": years,
           "log_growth_holding_pct": math.log(term["holding"] / START_EQUITY) / years * 100,
           "log_growth_liquidated_pct": math.log(term["liquidated"] / START_EQUITY) / years * 100,
           "tax_paid_pct_of_start": book.tax_paid / START_EQUITY * 100,
           "realized_short_term_pct_of_start": book.real_st / START_EQUITY * 100,
           "realized_long_term_pct_of_start": book.real_lt / START_EQUITY * 100,
           "fund_income_pct_of_start": book.fund_income / START_EQUITY * 100,
           "mean_age_of_sold_lots_days": (sum(l.sold_dollar_days for l in book.lots.values()) / sold if sold > 0 else None),
           "turnover_per_year": book.traded / years / (sum(path) / len(path)),
           "cost_paid_pct_of_start": book.cost_paid / START_EQUITY * 100}
    out.update(extra)
    return out


def simulate_weights(days: Sequence[str], prices: Dict[str, Sequence[float]], weights: Dict[str, float],
                     taxed: bool, cost_bp: float = COST_BP) -> Dict:
    """Buy at the first close, hold; with more than one ticker, rebalance to the
    target weights at the last session of each calendar year."""
    tickers = list(weights)
    book = Book(tickers, taxed)
    px = {t: prices[t][0] for t in tickers}
    for t in tickers:
        book.trade(t, weights[t] * START_EQUITY, px, days[0], 0.0)      # no entry cost for anyone
    book.traded = 0.0                                                   # turnover counts rebalancing, not the entry
    path, rebalances = [book.equity(px)], 0
    for i in range(1, len(days)):
        px = {t: prices[t][i] for t in tickers}
        if days[i][:4] != days[i - 1][:4]:
            book.settle_year(px, days[i], cost_bp)
        last_of_year = i + 1 < len(days) and days[i + 1][:4] != days[i][:4]
        if last_of_year and len(tickers) > 1:
            eq = book.equity(px)
            deltas = {t: weights[t] * eq - book.value(t, px) for t in tickers}
            for t in sorted(tickers, key=lambda k: deltas[k]):          # sells first
                book.trade(t, deltas[t], px, days[i], cost_bp)
            rebalances += 1
        path.append(book.equity(px))
    return _result(days, path, book, px, {"rebalances": rebalances})


def simulate_floor(days: Sequence[str], base_px: Sequence[float], lev_px: Sequence[float],
                   signal_closes: Sequence[float], sig_pos: Sequence[int], fund_ret: Sequence[float],
                   vol_rule: bool, cash_mode: str, taxed: bool, cfg: Optional[Dict] = None,
                   cost_bp: float = COST_BP) -> Dict:
    """The live floor's planning code on daily closes.

    `signal_closes[sig_pos[i]]` is the base ticker's PRICE-ONLY close on days[i]
    (what the live floor reads); the window before days[0] comes from the same
    list. `fund_ret[i]` is the T-bill fund's return INTO days[i].
    cash_mode: 'cash0' (the reserve sits in cash at 0%), 'invested' (no
    reserve: the floor's capital is all of equity), 'tbill' (everything outside
    the core goes to the T-bill fund)."""
    cfg = dict(cfg or live_cfg())
    if not vol_rule:
        cfg["FLOOR_VM_TARGET_VOL"], cfg["FLOOR_VM_CAP"] = 1e9, 1.0     # the same code, target pinned at 1.0
    if cash_mode == "invested":
        cfg["FLOOR_VM_CAPITAL_PCT"] = 1.0
    pct = float(cfg["FLOOR_VM_CAPITAL_PCT"])
    window = int(cfg["FLOOR_VM_WINDOW"])
    book = Book(["base", "lev"], taxed)
    path: List[float] = []
    exposures, cash_share, fund_share, rebalances = [], [], [], 0
    for i in range(len(days)):
        px = {"base": base_px[i], "lev": lev_px[i]}
        c = 0.0 if i == 0 else cost_bp
        if i > 0:
            book.accrue_fund(fund_ret[i])
            if days[i][:4] != days[i - 1][:4]:
                book.settle_year(px, days[i], cost_bp)
        eq = book.equity(px)
        j = sig_pos[i]
        closes = signal_closes[max(0, j - window - 1):j + 1]            # closes <= t only
        p = fvm.plan(closes, eq, book.value("base", px), book.value("lev", px), 0.0, cfg)
        traded_core = False
        if p["ok"] and (p["rebalance"] or i == 0):
            d_base = p["target_base_dollars"] - book.value("base", px)
            d_lev = p["target_levered_dollars"] - book.value("lev", px)
            for t, d in (("base", d_base), ("lev", d_lev)):
                if d < 0:
                    book.trade(t, d, px, days[i], c)
            buys = sum(d for d in (d_base, d_lev) if d > 0)
            need = buys * (1.0 + c / 1e4)
            if need > book.cash and book.fund > 0:
                book.fund_move(-min(book.fund, (need - book.cash) * (1.0 + c / 1e4)), c)   # cash comes first
            scale = min(1.0, book.cash / need) if need > 0 else 0.0
            for t, d in (("base", d_base), ("lev", d_lev)):
                if d > 0:
                    book.trade(t, d * scale, px, days[i], c)
            traded_core = True
            rebalances += 0 if i == 0 else 1
        eq = book.equity(px)
        core = book.value("base", px) + book.value("lev", px)
        if cash_mode == "tbill":
            cp = fvm.cash_plan(eq, core, book.fund, book.cash, 0.0, cfg, True, not traded_core or i == 0)
        else:
            cp = fvm.cash_plan(pct * eq, core, book.fund, book.cash, (1.0 - pct) * eq, cfg, True,
                               not traded_core or i == 0)
        if cp["action"] == "buy":
            book.fund_move(min(cp["dollars"], book.cash / (1.0 + c / 1e4)), c)
        elif cp["action"] == "sell":
            book.fund_move(-cp["dollars"], c)
        if i == 0:
            book.traded = 0.0                                           # turnover counts rebalancing, not the entry
        eq = book.equity(px)
        path.append(eq)
        exposures.append((book.value("base", px) + fvm.LEVERAGE_OF_LEVERED * book.value("lev", px)) / eq)
        cash_share.append(book.cash / eq)
        fund_share.append(book.fund / eq)
    n = len(days)
    return _result(days, path, book, px, {
        "rebalances": rebalances, "rebalances_per_year": rebalances / ((n - 1) / TD),
        "mean_exposure_of_equity": sum(exposures) / n, "mean_cash_share": sum(cash_share) / n,
        "mean_fund_share": sum(fund_share) / n, "exposures": exposures})


# ── comparison metrics ───────────────────────────────────────────────────────
def log_growth_pct(rets: Sequence[float]) -> Optional[float]:
    if not rets:
        return None
    return sum(math.log1p(x) for x in rets) / len(rets) * TD * 100


def tracking(rets: Sequence[float], bench: Sequence[float]) -> Tuple[float, Optional[float]]:
    """(tracking error %, information ratio), daily excess, annualised."""
    ex = [a - b for a, b in zip(rets, bench)]
    n = len(ex)
    mu = sum(ex) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in ex) / (n - 1))
    te = sd * math.sqrt(TD)
    return te * 100, (mu * TD / te if te > 1e-12 else None)


def monthly(days: Sequence[str], rets: Sequence[float]) -> Tuple[List[str], List[float]]:
    months: List[str] = []
    vals: List[float] = []
    for d, x in zip(days, rets):
        if not months or months[-1] != d[:7]:
            months.append(d[:7])
            vals.append(1.0)
        vals[-1] *= 1.0 + x
    return months, [v - 1.0 for v in vals]


def excess_t(days: Sequence[str], rets: Sequence[float], bench: Sequence[float]) -> Dict:
    """t-statistic of the monthly excess return; two-sided normal p-value."""
    _, a = monthly(days, rets)
    _, b = monthly(days, bench)
    ex = [x - y for x, y in zip(a, b)]
    n = len(ex)
    mu = sum(ex) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in ex) / (n - 1)) if n > 1 else 0.0
    if sd <= 1e-15:
        return {"months": n, "mean_monthly_excess_pct": mu * 100, "t": 0.0, "p": 1.0}
    t = mu / (sd / math.sqrt(n))
    return {"months": n, "mean_monthly_excess_pct": mu * 100, "t": t, "p": 2.0 * (1.0 - NormalDist().cdf(abs(t)))}


def longest_underperformance_months(days: Sequence[str], rets: Sequence[float], bench: Sequence[float]) -> int:
    """The longest time, in months, that wealth relative to the benchmark stays
    below an earlier month-end high (an unfinished stretch counts)."""
    _, a = monthly(days, rets)
    _, b = monthly(days, bench)
    rel, high, since, longest = 0.0, 0.0, 0, 0
    for x, y in zip(a, b):
        rel += math.log1p(x) - math.log1p(y)
        if rel >= high:
            high, since = rel, 0
        else:
            since += 1
            longest = max(longest, since)
    return longest


def month_regimes(days: Sequence[str], signal_closes: Sequence[float], sig_pos: Sequence[int], window: int = 20) -> Dict[str, str]:
    """month -> 'high' | 'low': SPY's 20-day realized vol at the last close of
    the PREVIOUS month vs the median of that reading over this sample's months.
    `days` are return days; signal_closes[sig_pos[i]] is the close on days[i]."""
    reading: Dict[str, float] = {}
    for i, d in enumerate(days):
        if i == 0 or days[i - 1][:7] != d[:7]:                         # first return day of a month
            j = sig_pos[i] - 1                                          # the close before it
            if j >= 0:
                sig = fvm.realized_vol(signal_closes[max(0, j - window):j + 1], window)
                if sig is not None:
                    reading[d[:7]] = sig
    if not reading:
        return {}
    vals = sorted(reading.values())
    m = len(vals)
    median = vals[m // 2] if m % 2 else (vals[m // 2 - 1] + vals[m // 2]) / 2.0
    return {k: ("high" if v > median else "low") for k, v in reading.items()}


def thirds(n: int) -> List[Tuple[int, int]]:
    return [(0, n // 3), (n // 3, 2 * n // 3), (2 * n // 3, n)]


def compare(days: Sequence[str], pre: Sequence[float], after: Sequence[float], bench: Sequence[float],
            rf: Sequence[float], regimes: Dict[str, str]) -> Dict:
    """Everything the tables show for one series against the benchmark on one
    sample. Risk metrics and the t-statistic use the pre-tax, after-cost path;
    sub-period and regime growth use the after-tax path (rule A)."""
    m = fs.metrics(pre, rf)
    b = fs.metrics(bench, rf)
    te, ir = tracking(pre, bench)
    wy = fs.worst_year(days, pre)
    out = {"log_growth_pct": log_growth_pct(pre), "spy_log_growth_pct": log_growth_pct(bench),
           "vol_pct": m["vol_pct"], "sharpe": m["sharpe"], "spy_sharpe": b["sharpe"],
           "max_dd_pct": m["max_dd_pct"], "spy_max_dd_pct": b["max_dd_pct"],
           "worst_year": [wy[0], wy[1]], "spy_worst_year": list(fs.worst_year(days, bench)),
           "tracking_error_pct": te, "information_ratio": ir,
           "rolling3y_beat_spy_pct": fs.rolling_beat(pre, bench),
           "longest_underperformance_months": longest_underperformance_months(days, pre, bench),
           "excess": excess_t(days, pre, bench), "from": days[0], "to": days[-1], "years": len(days) / TD}
    subs = []
    for lo, hi in thirds(len(days)):
        a, s = log_growth_pct(after[lo:hi]), log_growth_pct(bench[lo:hi])
        ms, mb = fs.metrics(pre[lo:hi], rf[lo:hi]), fs.metrics(bench[lo:hi], rf[lo:hi])
        subs.append({"from": days[lo], "to": days[hi - 1], "log_growth_pct": a, "spy_log_growth_pct": s,
                     "pre_log_growth_pct": log_growth_pct(pre[lo:hi]),
                     "beats_spy": a > s, "sharpe": ms["sharpe"], "spy_sharpe": mb["sharpe"],
                     "max_dd_pct": ms["max_dd_pct"], "spy_max_dd_pct": mb["max_dd_pct"]})
    out["subperiods"] = subs
    reg = {}
    for label in ("high", "low"):
        idx = [i for i, d in enumerate(days) if regimes.get(d[:7]) == label]
        a = log_growth_pct([after[i] for i in idx])
        s = log_growth_pct([bench[i] for i in idx])
        reg[label] = {"sessions": len(idx), "log_growth_pct": a, "spy_log_growth_pct": s,
                      "beats_spy": bool(idx) and a > s}
    out["regimes"] = reg
    return out


# ── decision rules (pre-registered) ──────────────────────────────────────────
def rule_a(own: Dict, common: Dict, survives_bonferroni: bool, own_tr: Optional[Dict] = None,
           common_tr: Optional[Dict] = None) -> Dict:
    checks = {
        "1_after_tax_liquidated_growth_above_spy_own_and_common":
            own["liq_log_growth_pct"] > own["spy_liq_log_growth_pct"]
            and common["liq_log_growth_pct"] > common["spy_liq_log_growth_pct"],
        "2_beats_spy_in_2_of_3_subperiods": sum(1 for s in own["subperiods"] if s["beats_spy"]) >= 2,
        "3_beats_spy_in_high_and_low_vol_months": own["regimes"]["high"]["beats_spy"] and own["regimes"]["low"]["beats_spy"],
        "4_max_dd_within_5_points_own_and_common":
            own["max_dd_pct"] >= own["spy_max_dd_pct"] - MAX_DD_SLACK_A
            and common["max_dd_pct"] >= common["spy_max_dd_pct"] - MAX_DD_SLACK_A,
        "5_sharpe_not_below_spy": own["sharpe"] >= own["spy_sharpe"],
        "6_survives_bonferroni": bool(survives_bonferroni),
    }
    first5 = all(v for k, v in checks.items() if not k.startswith("6"))
    outcome = "REPLACE" if first5 and checks["6_survives_bonferroni"] else ("SPY STAYS (not disproved)" if first5 else "SPY STAYS")
    out = {"checks": checks, "meets_1_to_5": first5, "outcome": outcome}
    if own_tr is not None and common_tr is not None:
        tr1 = (own_tr["liq_log_growth_pct"] > own_tr["spy_liq_log_growth_pct"]
               and common_tr["liq_log_growth_pct"] > common_tr["spy_liq_log_growth_pct"])
        out["condition_1_on_total_return"] = tr1
        out["paper_artifact"] = bool(first5 and not tr1)
    return out


def rule_b_sample(on: Dict, off: Dict) -> Dict:
    """The floor study's ship rule, vol rule ON vs OFF, one sample."""
    wins = sum(1 for a, b in zip(on["sub_log_growth_pct"], off["sub_log_growth_pct"]) if a > b)
    checks = {"log_growth_higher": on["log_growth_pct"] > off["log_growth_pct"],
              "sharpe_not_lower": on["sharpe"] >= off["sharpe"],
              "max_dd_within_10_points": on["max_dd_pct"] >= off["max_dd_pct"] - MAX_DD_SLACK_B,
              "higher_growth_in_2_of_3_subperiods": wins >= 2}
    return {"checks": checks, "passes": all(checks.values()),
            "better_on_risk": on["sharpe"] > off["sharpe"] and on["max_dd_pct"] > off["max_dd_pct"]}


def rule_b(modern: Dict, long_: Dict) -> str:
    if modern["passes"] and long_["passes"]:
        return "earns its keep"
    if modern["better_on_risk"] and long_["better_on_risk"]:
        return "earns it on risk (Sharpe and drawdown), not on growth"
    return "does not earn it on this evidence"


def rule_c(cells: Dict[str, Dict], bench_max_dd: float) -> Dict:
    tbill = {}
    for row in ("B0", "B1"):
        i, iii = cells[f"{row}-i"], cells[f"{row}-iii"]
        tbill[row] = iii["log_growth_pct"] > i["log_growth_pct"] and iii["max_dd_pct"] >= i["max_dd_pct"] - MAX_DD_SLACK_C_TBILL
    ii, iii = cells["B1-ii"], cells["B1-iii"]
    wins = sum(1 for a, b in zip(ii["sub_after_tax_log_growth_pct"], iii["sub_after_tax_log_growth_pct"]) if a > b)
    inv = {"after_tax_liquidated_growth_higher": ii["liq_log_growth_pct"] > iii["liq_log_growth_pct"],
           "higher_in_2_of_3_subperiods": wins >= 2,
           "max_dd_within_5_points_of_spy": ii["max_dd_pct"] >= bench_max_dd - MAX_DD_SLACK_C_INVESTED}
    return {"tbill_fund_preferred_to_cash_at_0": tbill["B0"] and tbill["B1"], "tbill_by_row": tbill,
            "fully_invested_checks": inv, "fully_invested_preferred_to_tbill_fund": all(inv.values())}


# ── the study ────────────────────────────────────────────────────────────────
def _round(o, nd: int = 4):
    if isinstance(o, float):
        return round(o, nd) if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _round(v, nd) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_round(v, nd) for v in o]
    return o


def _strip(run: Dict) -> Dict:
    return {k: v for k, v in run.items() if k not in ("days", "rets", "exposures")}


def run_basis(px: Dict[str, Dict[str, float]], signal: Dict[str, float], irx: Dict[str, float],
              lev_long: Dict[str, float], available: Sequence[str]) -> Dict:
    """Every series on one price basis (`px`: price-only or total return). The
    floor's volatility signal is always SPY's price-only close (`signal`), the
    series the live floor reads."""
    sig_days = sorted(signal)
    sig_closes = [signal[d] for d in sig_days]
    sig_idx = {d: i for i, d in enumerate(sig_days)}
    window = int(live_cfg()["FLOOR_VM_WINDOW"])
    first_ok = sig_days[window]                                          # sigma needs window + 1 closes

    def spy_runs(days):
        prices = {BENCH: [px[BENCH][d] for d in days]}
        return (simulate_weights(days, prices, {BENCH: 1.0}, taxed=False),
                simulate_weights(days, prices, {BENCH: 1.0}, taxed=True))

    def pack(pre, post, spy_pre, spy_post):
        rdays = pre["days"]
        rf = fs.daily_rf(irx, rdays)
        regimes = month_regimes(rdays, sig_closes, [sig_idx[d] for d in rdays], window)
        c = compare(rdays, pre["rets"], post["rets"], spy_pre["rets"], rf, regimes)
        c["liq_log_growth_pct"] = post["log_growth_liquidated_pct"]
        c["holding_log_growth_pct"] = post["log_growth_holding_pct"]
        c["spy_liq_log_growth_pct"] = spy_post["log_growth_liquidated_pct"]
        c["tax_drag_holding_pp"] = c["log_growth_pct"] - post["log_growth_holding_pct"]
        c["tax_drag_liquidated_pp"] = c["log_growth_pct"] - post["log_growth_liquidated_pct"]
        c["sim"] = _strip(pre)
        c["tax"] = {k: post[k] for k in ("tax_paid_pct_of_start", "realized_short_term_pct_of_start",
                                         "realized_long_term_pct_of_start", "fund_income_pct_of_start",
                                         "mean_age_of_sold_lots_days", "rebalances")}
        return c

    def floor_runs(days, lev_series, cell):
        vol_rule, cash_mode = GRID[cell]
        fund = [r - FUND_FEE / TD for r in fs.daily_rf(irx, days)]
        args = (days, [px[BENCH][d] for d in days], [lev_series[d] for d in days], sig_closes,
                [sig_idx[d] for d in days], fund, vol_rule, cash_mode)
        return simulate_floor(*args, taxed=False), simulate_floor(*args, taxed=True)

    out: Dict = {"own": {}, "common": {}, "modern_grid": {}, "long_grid": {}}
    common_start = max(min(px[t]) for t in ETFS if t in available)
    out["common_start"] = common_start

    # base candidates: own longest sample and the all-common sample
    for name, w in CANDIDATES.items():
        if any(t not in available for t in w):
            out["own"][name] = out["common"][name] = {"dropped": "a ticker has no data"}
            continue
        for label, start in (("own", None), ("common", common_start)):
            days = [d for d in common_days(px, [BENCH] + list(w), start) if d >= first_ok]
            prices = {t: [px[t][d] for d in days] for t in w}
            pre, post = simulate_weights(days, prices, w, taxed=False), simulate_weights(days, prices, w, taxed=True)
            out[label][name] = pack(pre, post, *spy_runs(days))

    # the grid on the modern sample (real SSO); B1-i also on the all-common sample
    if LEVERED in available:
        modern = [d for d in common_days(px, [BENCH, LEVERED]) if d >= first_ok]
        spy_m = spy_runs(modern)
        for cell in GRID:
            out["modern_grid"][cell] = pack(*floor_runs(modern, px[LEVERED], cell), *spy_m)
        cdays = [d for d in modern if d >= common_start]
        out["common"]["B1-i"] = pack(*floor_runs(cdays, px[LEVERED], "B1-i"), *spy_runs(cdays))
        out["own"]["B1-i"] = out["modern_grid"]["B1-i"]

    # the grid on the long sample (the synthetic 2x fund)
    long_days = [d for d in common_days({BENCH: px[BENCH], "2x": lev_long}, [BENCH, "2x"]) if d >= first_ok]
    spy_l = spy_runs(long_days)
    for cell in GRID:
        out["long_grid"][cell] = pack(*floor_runs(long_days, lev_long, cell), *spy_l)
    return out


def grid_view(c: Dict) -> Dict:
    """What rules B and C read from one grid cell."""
    return {"log_growth_pct": c["log_growth_pct"], "sharpe": c["sharpe"], "max_dd_pct": c["max_dd_pct"],
            "liq_log_growth_pct": c["liq_log_growth_pct"],
            "sub_log_growth_pct": [s["pre_log_growth_pct"] for s in c["subperiods"]],
            "sub_after_tax_log_growth_pct": [s["log_growth_pct"] for s in c["subperiods"]]}


def significance(own_of: Dict[str, Dict]) -> Dict:
    """One test per series (monthly excess return vs SPY), one family: Bonferroni
    over the pre-registered 16 and Benjamini-Hochberg q-values."""
    names = [n for n in FAMILY if n in own_of and "dropped" not in own_of[n]]
    bar = bonferroni_z(len(FAMILY), FAMILY_ALPHA)
    q = benjamini_hochberg([own_of[n]["excess"]["p"] for n in names])
    return {"family_size": len(FAMILY), "tested": len(names), "bonferroni_abs_t": bar, "series": {
        n: {"t": own_of[n]["excess"]["t"], "p": own_of[n]["excess"]["p"], "bh_q": q[i],
            "months": own_of[n]["excess"]["months"],
            "survives_bonferroni": abs(own_of[n]["excess"]["t"]) > bar,
            "survives_bh_10pct": q[i] < 0.10} for i, n in enumerate(names)}}


def main(src: str, out_path: str) -> None:
    px, tr, irx, facts = load_all(src)
    available = [t for t in ETFS if t in px]
    if BENCH not in available:
        raise SystemExit("SPY has no data: the study cannot run")
    lev_long = synthetic_2x(tr[BENCH], irx)
    res: Dict = {"end_day": END_DAY, "data": facts, "family": FAMILY, "live_cfg": live_cfg(),
                 "assumptions": {"cost_bp": COST_BP, "fund_fee": FUND_FEE, "tax_short": TAX_SHORT, "tax_long": TAX_LONG,
                                 "long_term_days": LONG_TERM_DAYS, "start_equity": START_EQUITY}}
    for t in available:                                                 # the dividend part, per ticker
        days = sorted(px[t])
        yrs = (len(days) - 1) / TD
        po_g = math.log(px[t][days[-1]] / px[t][days[0]]) / yrs * 100
        tr_g = math.log(tr[t][days[-1]] / tr[t][days[0]]) / yrs * 100
        facts[t].update({"price_only_log_growth_pct": po_g, "total_return_log_growth_pct": tr_g,
                         "dividend_part_pp": tr_g - po_g})

    # data checks: the floor study's own code on today's download; the synthetic 2x fund against real SSO
    sd = sorted(d for d in tr[BENCH] if d <= "2026-11-04")
    r = fs.returns([tr[BENCH][d] for d in sd])
    rfc = fs.daily_rf(irx, sd[1:])
    v1, _ = fs.run_v1(r, rfc)
    keys = ("log_growth_pct", "sharpe", "max_dd_pct")
    res["data_checks"] = {
        "floor_study_v1_rerun": {k: fs.metrics(v1, rfc)[k] for k in keys},
        "floor_study_v1_published": {"log_growth_pct": 10.41, "sharpe": 0.573, "max_dd_pct": -48.41},
        "floor_study_spy_rerun": {k: fs.metrics(r, rfc)[k] for k in keys},
        "floor_study_spy_published": {"log_growth_pct": 10.26, "sharpe": 0.513, "max_dd_pct": -55.2}}
    if LEVERED in available:
        od = [d for d in sorted(px[LEVERED]) if d in lev_long]
        real = [px[LEVERED][od[i]] / px[LEVERED][od[i - 1]] - 1.0 for i in range(1, len(od))]
        real_tr = [tr[LEVERED][od[i]] / tr[LEVERED][od[i - 1]] - 1.0 for i in range(1, len(od))]
        syn = [lev_long[od[i]] / lev_long[od[i - 1]] - 1.0 for i in range(1, len(od))]
        res["data_checks"]["synthetic_2x_vs_sso"] = {
            "from": od[0], "sso_price_only_log_growth_pct": log_growth_pct(real),
            "sso_total_return_log_growth_pct": log_growth_pct(real_tr),
            "synthetic_log_growth_pct": log_growth_pct(syn), "tracking_error_pct": tracking(syn, real)[0]}

    po = run_basis(px, px[BENCH], irx, lev_long, available)
    tot = run_basis(tr, px[BENCH], irx, lev_long, available)
    res["price_only"], res["total_return"] = po, tot

    # significance: one test per series on its own longest sample (grid cells: the modern sample)
    own_of = dict(po["own"])
    own_of.update({c: v for c, v in po["modern_grid"].items() if c != "B0-ii"})
    res["significance"] = sig = significance(own_of)

    res["rule_a"] = {}
    for n in list(CANDIDATES) + ["B1-i"]:
        if n not in sig["series"]:
            res["rule_a"][n] = {"outcome": "DROPPED (no data)"}
            continue
        ok6 = sig["series"][n]["survives_bonferroni"] and sig["series"][n]["t"] > 0
        res["rule_a"][n] = rule_a(po["own"][n], po["common"][n], ok6, tot["own"][n], tot["common"][n])
    winners = [n for n, v in res["rule_a"].items() if v["outcome"] == "REPLACE"]
    res["rule_a_decision"] = ("REPLACE with " + ", ".join(winners)) if winners else "SPY stays"

    if po["modern_grid"]:
        bm = rule_b_sample(grid_view(po["modern_grid"]["B1-iii"]), grid_view(po["modern_grid"]["B0-iii"]))
        bl = rule_b_sample(grid_view(po["long_grid"]["B1-iii"]), grid_view(po["long_grid"]["B0-iii"]))
        res["rule_b"] = {"modern": bm, "long": bl, "verdict": rule_b(bm, bl)}
        res["rule_c"] = rule_c({k: grid_view(v) for k, v in po["modern_grid"].items()},
                               po["modern_grid"]["B0-ii"]["max_dd_pct"])
    with open(out_path, "w") as f:
        json.dump(_round(res), f, indent=1)
    print(json.dumps(_round({k: res[k] for k in ("data_checks", "rule_a_decision", "rule_b", "rule_c") if k in res}, 3), indent=1))


# ── the write-up's tables, generated from the JSON (no hand-copied number) ───
def _f(x, nd: int = 2, sign: bool = False) -> str:
    if x is None:
        return "n/a"
    return (f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}").replace("-", "−")


def _yn(b) -> str:
    return "yes" if b else "NO"


def markdown(res: Dict) -> str:
    po, tot, sig = res["price_only"], res["total_return"], res["significance"]["series"]
    L: List[str] = []
    L += ["### Data (every ticker's first date; Yahoo daily, through " + res["end_day"] + ")", "",
          "| ticker | first date | sessions | largest 1-day move | price-only growth/yr | total-return growth/yr | dividend part |",
          "|---|---|---|---|---|---|---|"]
    for t in ETFS:
        d = res["data"].get(t, {})
        if "dropped" in d:
            L.append(f"| {t} | DROPPED: {d['dropped']} | | | | | |")
            continue
        L.append(f"| {t} | {d['first']} | {d['sessions']} | {_f(d['max_abs_daily_move_pct'])}% | {_f(d['price_only_log_growth_pct'])}% | "
                 f"{_f(d['total_return_log_growth_pct'])}% | {_f(d['dividend_part_pp'])} pp |")

    def table(title: str, block: Dict, names: Sequence[str], with_sig: bool) -> None:
        L.extend(["", title, "",
                  "| base | from | yrs | growth/yr (SPY) | after-tax liquidated (SPY) | Sharpe (SPY) | max DD (SPY) | worst year | "
                  "track. err | info ratio | 3y windows beating SPY | longest lag (months) | t" + (" | p | BH q |" if with_sig else " |"),
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|" + ("---|---|" if with_sig else "")])
        for n in names:
            c = block.get(n)
            if not c or "dropped" in c:
                L.append(f"| {n} | DROPPED | | | | | | | | | | | |" + (" | |" if with_sig else ""))
                continue
            row = (f"| {n} | {c['from']} | {_f(c['years'], 1)} | {_f(c['log_growth_pct'])}% ({_f(c['spy_log_growth_pct'])}%) | "
                   f"{_f(c['liq_log_growth_pct'])}% ({_f(c['spy_liq_log_growth_pct'])}%) | {_f(c['sharpe'], 3)} ({_f(c['spy_sharpe'], 3)}) | "
                   f"{_f(c['max_dd_pct'], 1)}% ({_f(c['spy_max_dd_pct'], 1)}%) | {c['worst_year'][0]} {_f(c['worst_year'][1], 1)}% | "
                   f"{_f(c['tracking_error_pct'], 1)}% | {_f(c['information_ratio'], 2, True)} | {_f(c['rolling3y_beat_spy_pct'], 1)}% | "
                   f"{c['longest_underperformance_months']} | {_f(c['excess']['t'], 2, True)} |")
            if with_sig:
                row += f" {_f(sig[n]['p'], 3)} | {_f(sig[n]['bh_q'], 3)} |" if n in sig else " | |"
            L.append(row)

    names = list(CANDIDATES) + ["B1-i"]
    table("### Each base against SPY on its OWN LONGEST sample (price-only; the decision basis)", po["own"], names, True)
    table(f"### Every base on the ALL-COMMON sample (from {po['common_start']}; price-only)", po["common"], names, False)

    L += ["", "### Sub-periods and volatility regimes (own longest sample, price-only; growth/yr minus SPY's, in points; after tax paid along the way)", "",
          "| base | first third | middle third | last third | thirds won | high-vol months | low-vol months |", "|---|---|---|---|---|---|---|"]
    for n in names:
        c = po["own"].get(n)
        if not c or "dropped" in c:
            continue
        subs = c["subperiods"]
        cells = [f"{_f(s['log_growth_pct'] - s['spy_log_growth_pct'], 2, True)} ({s['from'][:7]}→{s['to'][:7]})" for s in subs]
        reg = c["regimes"]
        L.append(f"| {n} | {cells[0]} | {cells[1]} | {cells[2]} | {sum(1 for s in subs if s['beats_spy'])}/3 | "
                 f"{_f(reg['high']['log_growth_pct'] - reg['high']['spy_log_growth_pct'], 2, True)} | "
                 f"{_f(reg['low']['log_growth_pct'] - reg['low']['spy_log_growth_pct'], 2, True)} |")

    L += ["", "### Total-return check (dividends included on both sides; not the decision basis)", "",
          "| base | own sample: growth/yr minus SPY (price-only) | same, total return | common sample (price-only) | same, total return |",
          "|---|---|---|---|---|"]
    for n in names:
        a, b, c, d = po["own"].get(n), tot["own"].get(n), po["common"].get(n), tot["common"].get(n)
        if not a or "dropped" in a:
            continue
        L.append(f"| {n} | {_f(a['log_growth_pct'] - a['spy_log_growth_pct'], 2, True)} | {_f(b['log_growth_pct'] - b['spy_log_growth_pct'], 2, True)} | "
                 f"{_f(c['log_growth_pct'] - c['spy_log_growth_pct'], 2, True)} | {_f(d['log_growth_pct'] - d['spy_log_growth_pct'], 2, True)} |")

    L += ["", f"### Rule A, check by check (Bonferroni bar: |t| > {_f(res['significance']['bonferroni_abs_t'], 2)} over {res['significance']['family_size']} series)", "",
          "| base | 1 after-tax growth, own + common | 2 two of three thirds | 3 both vol regimes | 4 drawdown within 5 pts | 5 Sharpe | 6 Bonferroni | "
          "condition 1 on total return | outcome |", "|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        v = res["rule_a"].get(n, {})
        if "checks" not in v:
            L.append(f"| {n} | | | | | | | | {v.get('outcome', 'n/a')} |")
            continue
        ck = list(v["checks"].values())
        L.append(f"| {n} | " + " | ".join(_yn(x) for x in ck) + f" | {_yn(v.get('condition_1_on_total_return'))} | "
                 f"{v['outcome']}{' — PAPER ARTIFACT' if v.get('paper_artifact') else ''} |")
    L += ["", f"**Rule A decision: {res['rule_a_decision']}.**"]

    L += ["", "### Differences from SPY, own longest sample (price-only; what the priors were about)", "",
          "| base | growth/yr, points | after-tax liquidated growth/yr, points | Sharpe | max drawdown, points (+ = smaller) |", "|---|---|---|---|---|"]
    for n in names:
        c = po["own"].get(n)
        if not c or "dropped" in c:
            continue
        L.append(f"| {n} | {_f(c['log_growth_pct'] - c['spy_log_growth_pct'], 2, True)} | "
                 f"{_f(c['liq_log_growth_pct'] - c['spy_liq_log_growth_pct'], 2, True)} | {_f(c['sharpe'] - c['spy_sharpe'], 3, True)} | "
                 f"{_f(c['max_dd_pct'] - c['spy_max_dd_pct'], 1, True)} |")

    L += ["", "### Tax ESTIMATE for the bases that trade (price-only; FIFO lots, 24% short-term, 15% long-term)", "",
          "| base | sample | growth/yr before tax | after tax, still holding | after tax, liquidated | SPY liquidated | "
          "realized short-term gains | realized long-term gains | tax paid | mean age of lots sold | rebalances |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    rows = [("MTUM+QUAL", "own", po["own"].get("MTUM+QUAL")), ("MTUM+QUAL", "common", po["common"].get("MTUM+QUAL"))]
    rows += [(cell, "modern", po["modern_grid"].get(cell)) for cell in ("B1-i", "B1-ii", "B1-iii") if po["modern_grid"]]
    rows += [("B1-i", "common", po["common"].get("B1-i")), ("B1-i", "long", po["long_grid"]["B1-i"])]
    for n, label, c in rows:
        if not c or "dropped" in c:
            continue
        t = c["tax"]
        L.append(f"| {n} | {label}, {c['from'][:4]}+ | {_f(c['log_growth_pct'])}% | {_f(c['holding_log_growth_pct'])}% | {_f(c['liq_log_growth_pct'])}% | "
                 f"{_f(c['spy_liq_log_growth_pct'])}% | {_f(t['realized_short_term_pct_of_start'], 0)}% | {_f(t['realized_long_term_pct_of_start'], 0)}% | "
                 f"{_f(t['tax_paid_pct_of_start'], 0)}% | {_f(t['mean_age_of_sold_lots_days'], 0)} days | {t['rebalances']} |")
    L += ["", "(Gains and tax are in % of the starting equity, summed over the sample.)"]

    def grid(title: str, block: Dict) -> None:
        L.extend(["", title, "",
                  "| cell | growth/yr | after tax, still holding | after tax, liquidated | Sharpe | max DD | worst year | mean exposure | "
                  "mean cash | mean T-bill fund | rebalances/yr | turnover/yr | t vs SPY |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"])
        for cell in GRID:
            c = block[cell]
            s = c["sim"]
            L.append(f"| {cell} | {_f(c['log_growth_pct'])}% | {_f(c['holding_log_growth_pct'])}% | {_f(c['liq_log_growth_pct'])}% | "
                     f"{_f(c['sharpe'], 3)} | {_f(c['max_dd_pct'], 1)}% | {c['worst_year'][0]} {_f(c['worst_year'][1], 1)}% | "
                     f"{_f(s['mean_exposure_of_equity'], 2)}x | {_f(s['mean_cash_share'] * 100, 1)}% | {_f(s['mean_fund_share'] * 100, 1)}% | "
                     f"{_f(s['rebalances_per_year'], 1)} | {_f(s['turnover_per_year'] * 100, 0)}% | {_f(c['excess']['t'], 2, True)} |")

    if po["modern_grid"]:
        m = po["modern_grid"]["B0-ii"]
        grid(f"### The cash and volatility-rule grid, modern sample (real SSO; {m['from']} → {m['to']}, {_f(m['years'], 1)} years; price-only)", po["modern_grid"])
        grid("### The same grid on total return (check)", tot["modern_grid"])
    lg = po["long_grid"]["B0-ii"]
    grid(f"### The grid on the long sample (synthetic 2x fund; {lg['from']} → {lg['to']}, {_f(lg['years'], 1)} years; price-only)", po["long_grid"])
    grid("### The long-sample grid on total return (check; the floor study's basis)", tot["long_grid"])
    for label, block in (("modern", po.get("modern_grid") or {}), ("long", po["long_grid"])):
        if not block:
            continue
        L += ["", f"Volatility rule ON minus OFF by third, {label} sample, column (iii), growth/yr in points: "
              + ", ".join(f"{_f(a['pre_log_growth_pct'] - b['pre_log_growth_pct'], 2, True)} ({a['from'][:4]}–{a['to'][:4]})"
                          for a, b in zip(block["B1-iii"]["subperiods"], block["B0-iii"]["subperiods"])) + "."]
    if "rule_b" in res:
        rb, rc = res["rule_b"], res["rule_c"]
        L += ["", "**Rule B (the volatility rule, ON vs OFF, column iii):** modern sample " + ", ".join(f"{k} {_yn(v)}" for k, v in rb["modern"]["checks"].items())
              + "; long sample " + ", ".join(f"{k} {_yn(v)}" for k, v in rb["long"]["checks"].items()) + f". Verdict: **{rb['verdict']}**.",
              "", f"**Rule C (the idle 15%):** T-bill fund preferred to cash at 0%: **{_yn(rc['tbill_fund_preferred_to_cash_at_0'])}** "
              f"(vol rule off {_yn(rc['tbill_by_row']['B0'])}, on {_yn(rc['tbill_by_row']['B1'])}). Fully invested preferred to the T-bill fund: "
              f"**{_yn(rc['fully_invested_preferred_to_tbill_fund'])}** (" + ", ".join(f"{k} {_yn(v)}" for k, v in rc["fully_invested_checks"].items()) + ")."]
    dc = res["data_checks"]
    L += ["", "### Data checks", "",
          f"- The floor study's own V1 code on today's download, through 2026-11-04: log growth {dc['floor_study_v1_rerun']['log_growth_pct']}%, "
          f"Sharpe {dc['floor_study_v1_rerun']['sharpe']}, max DD {dc['floor_study_v1_rerun']['max_dd_pct']}% (published: 10.41%, 0.573, −48.41%). "
          f"SPY: {dc['floor_study_spy_rerun']['log_growth_pct']}%, {dc['floor_study_spy_rerun']['sharpe']}, {dc['floor_study_spy_rerun']['max_dd_pct']}% (published 10.26%, 0.513, −55.2%)."]
    if "synthetic_2x_vs_sso" in dc:
        s = dc["synthetic_2x_vs_sso"]
        L.append(f"- The synthetic 2x fund against real SSO from {s['from']}: synthetic {_f(s['synthetic_log_growth_pct'])}%/yr, SSO price-only "
                 f"{_f(s['sso_price_only_log_growth_pct'])}%/yr, SSO total return {_f(s['sso_total_return_log_growth_pct'])}%/yr; daily tracking error {_f(s['tracking_error_pct'])}%/yr.")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--fetch":
        fetch(sys.argv[2])
    elif len(sys.argv) == 3 and sys.argv[1] == "--md":
        print(markdown(json.load(open(sys.argv[2]))))
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        raise SystemExit(__doc__)
