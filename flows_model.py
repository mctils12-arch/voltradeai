#!/usr/bin/env python3
"""
flows_model.py — the institutional flow model (refocus spec 2026-11-05, decision
pipeline steps 2-4: calendar effects, institutional flows, sector rotation).

OUTPUT, an ESTIMATE (never a forecast of record): the expected net buying or
selling pressure on broad US equities and on each sector over the next
FLOWS_HORIZON_SESSIONS sessions, in [-1, 1] (negative = expected net selling),
with every driver's value and a plain-English reason. Written once per
completed session to DATA_DIR/voltrade_flows.json (Node serves it on
GET /api/bot/flows) and appended to voltrade_flows_history.jsonl, so every
estimate can later be scored against what SPY and the sectors actually did.

USED BY (each ON by default, one system_config switch each, every effect logged):
- the volatility-managed floor: floor_multiplier() is a multiplier in
  [1 - FLOWS_FLOOR_MAX_CUT, 1.0] on the floor's target exposure E*. Never above
  1.0, so it can only cut exposure and never adds leverage.
- satellite stock sizing: sector_size_adjustment() scales a candidate's shares
  by 1 + FLOWS_SECTOR_SIZE_K x its sector's pressure, capped at the sizer's
  per-position maximum.

THE DRIVERS (research read 2026-11-06; validation in research/, see
scripts/flows_validation.py):
1. Calendar rebalancing (Harvey, Mazzoleni & Melone, "The Unintended
   Consequences of Rebalancing", NBER w33554, 2025, rev. 2026; their EDHEC
   slides of 2026-02-10). A 60/40 stock/bond portfolio that resets at each
   month end drifts with relative returns; its pre-reset distance from 60% is
   the calendar signal. In the last week of the month, an overweight in stocks
   predicts lower stock returns the next day (S&P 500 coefficient -0.27,
   t ~ -3.5, daily 1997-2023). Quarterly rebalancers add the drift since the
   last quarter end at quarter ends. Here: stocks = SPY, bonds = IEF (7-10y
   Treasuries, the closest ETF to their 10-year note futures).
2. Threshold rebalancing (same paper): the distance of a portfolio that resets
   whenever it is more than delta from 60%, averaged over delta in 0..2%,
   predicts the next day's return at any time of the month.
3. Month-end cash needs (Etula, Rinne, Suominen & Vaittinen 2020, "Dash for
   Cash", RFS 33(1)): selling before the month end to fund payments, then a
   reversal and new-month inflows (calendar_effects.month_end_phase).
4. Tax-loss selling (Gibson, Safieddine & Titman 2000, JFQA 35(3)): funds
   with October 31 year ends sell losers in October; rebound in November. The
   December version (individuals, calendar-year funds) is included on the same
   logic with no claimed size. Sector level only: the YTD-worst sector ETFs.
5. Sector rotation: industry momentum (Moskowitz & Grinblatt 1999, JF 54(4):
   past-winning industries keep outperforming) as "where flow is heading",
   minus a crowding penalty for a 1-month run more than 2 sigma above SPY
   ("reduce crowded sectors after big runs", the human's spec; not a result
   from the papers read).
6. Index events (S&P 500 quarterly rebalance, Russell reconstitution,
   Nasdaq-100 reconstitution): heavy flows whose direction is unknown from
   the calendar alone. They are listed and named in the drivers, with zero
   directional weight.
13F / fund holdings: the repo has a 13F pipeline (server/edgar13f.ts, ladder
gate 1 only: focused managers with at most 250 holdings, no sector
aggregation, a 45-day filing lag). It is NOT used here.

Pure core (no I/O): rebalance paths, sector scores, compute_flows(),
floor_multiplier(), sector_size_adjustment(). I/O: fetch_closes(), refresh_daily(),
load_cached(). Never raises out of the I/O entry points.
"""
from __future__ import annotations

import json
import logging
import math
import os
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import calendar_effects as ce

log = logging.getLogger("flows_model")

LABEL = "ESTIMATE"
SCHEMA_VERSION = 1
FLOWS_FILE = "voltrade_flows.json"
FLOWS_HISTORY_FILE = "voltrade_flows_history.jsonl"

TARGET_EQUITY = 0.60
THRESHOLD_DELTAS = tuple(round(0.0025 * i, 4) for i in range(9))   # 0 .. 2%

BENCH = "SPY"
DEFAULT_BOND = "IEF"
SECTORS: Tuple[Tuple[str, str], ...] = (
    ("Technology", "XLK"), ("Semiconductors", "SMH"), ("Financials", "XLF"), ("Energy", "XLE"),
    ("Healthcare", "XLV"), ("Consumer Discretionary", "XLY"), ("Consumer Staples", "XLP"),
    ("Industrials", "XLI"), ("Materials", "XLB"), ("Utilities", "XLU"), ("Real Estate", "XLRE"),
    ("Communications", "XLC"),
)
SECTOR_ETF = dict(SECTORS)
# The sector names bot_engine.SECTOR_MAP and the yfinance sector cache use.
SECTOR_ALIASES = {
    "technology": "Technology", "information technology": "Technology",
    "financials": "Financials", "financial services": "Financials", "financial": "Financials",
    "energy": "Energy", "healthcare": "Healthcare", "health care": "Healthcare",
    "consumer discretionary": "Consumer Discretionary", "consumer cyclical": "Consumer Discretionary",
    "consumer staples": "Consumer Staples", "consumer defensive": "Consumer Staples",
    "industrials": "Industrials", "materials": "Materials", "basic materials": "Materials",
    "utilities": "Utilities", "real estate": "Real Estate",
    "communications": "Communications", "communication services": "Communications",
    "semiconductors": "Semiconductors",
}
# Semiconductor companies (fact, not inference) get the SMH reading, not XLK's.
SEMICONDUCTOR_TICKERS = frozenset({
    "NVDA", "AMD", "INTC", "QCOM", "TXN", "AVGO", "AMAT", "MU", "LRCX", "KLAC", "ADI", "MRVL",
    "NXPI", "ON", "MCHP", "TSM", "ASML", "ARM", "SMCI", "MPWR", "TER", "SWKS", "QRVO", "GFS",
    "ENTG", "WOLF", "LSCC", "SLAB", "CRUS", "SYNA", "AMKR", "COHR", "MTSI", "RMBS",
})
NASDAQ100_HEAVY = ("Technology", "Semiconductors", "Communications", "Consumer Discretionary")

# Default parameters: a MIRROR of system_config.BASE_CONFIG's FLOWS_* block
# (the tunables live there; test_flows_model.py pins this table to it). Used
# only when a caller passes no config (the pure functions, the validation).
# Weights and scales set 2026-11-06 from scripts/flows_validation.py (daily
# SPY/IEF/VFITX/sector-ETF history 1993-2026; out of sample = 2014 on):
# - threshold rebalancing: next-day slope t -2.6 out of sample (IEF,
#   Newey-West), -3.0 in sample, the strongest driver, but next-day only
#   (2-10 day horizons mixed in a post-hoc check);
# - calendar rebalancing: right sign, t -1.7 in sample, -1.0 out of sample,
#   ~0 since the paper's data end (2023-04);
# - month-end cash windows: -13 bp/day t -2.5 in the paper's 1995-2013, -4
#   bp/day t -0.8 after, wrong sign (n 29) in the T+1 era;
# - sector momentum and crowding: no predictive power (|t| < 1 both halves);
# - tax-loss: December losers -1.3% vs the sector average (t -2.25), January
#   rebound +0.7% (t 1.4); OCTOBER THE OPPOSITE of the prediction (+1.8%,
#   t 1.7, n 18), so October/November carry a quarter weight.
# The scales are ~1.5 standard deviations of each signal (2.1 and 0.42 points).
DEFAULTS = {
    "FLOWS_ENABLED": True,
    "FLOWS_FLOOR_ENABLED": True,
    "FLOWS_SECTOR_SIZING_ENABLED": True,
    "FLOWS_HORIZON_SESSIONS": 5,
    "FLOWS_BOND_PROXY": DEFAULT_BOND,
    "FLOWS_FLOOR_MAX_CUT": 0.10,
    "FLOWS_FLOOR_DEADBAND": 0.10,
    "FLOWS_SECTOR_SIZE_K": 0.10,
    "FLOWS_MAX_AGE_DAYS": 4,
    "FLOWS_REBAL_SCALE": 0.03,
    "FLOWS_THRESHOLD_SCALE": 0.006,
    "FLOWS_W_REBALANCE": 0.45,
    "FLOWS_W_THRESHOLD": 0.35,
    "FLOWS_W_MONTH_END": 0.20,
    "FLOWS_W_ROTATION": 0.3,
    "FLOWS_W_TAX": 0.3,
    "FLOWS_W_BROAD": 0.5,
    "FLOWS_W_MOMENTUM": 1.0,
    "FLOWS_W_CROWDING": 0.25,
    "FLOWS_TAX_WINDOW_WEIGHTS": {"fund_tax_loss": 0.25, "fund_tax_loss_reversal": 0.25,
                                 "december_tax_loss": 1.0, "turn_of_year": 1.0},
}

# How much each driver's evidence supports it (0..1), from the same check. The
# confidence of an estimate is the evidence-weighted share of its drivers.
EVIDENCE = {
    "rebalance": 0.4, "threshold": 0.6, "month_end": 0.2,
    "momentum": 0.1, "crowding": 0.05, "tax_loss": 0.2, "broad": 0.5,
}

DIRECTION_BAND = 0.15   # |pressure| below this reads "neutral"


def cfg_value(cfg: Optional[Dict], key: str):
    if cfg and key in cfg:
        return cfg[key]
    return DEFAULTS[key]


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def direction_of(p: float) -> str:
    return "sell" if p <= -DIRECTION_BAND else "buy" if p >= DIRECTION_BAND else "neutral"


def _d(s: str) -> date:
    return date.fromisoformat(str(s)[:10])


def _utcnow() -> datetime:
    """Naive UTC now (the file's computedAt carries a 'Z')."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ── rebalancing signals (Harvey, Mazzoleni & Melone) ──────────────────────────

def drift_path(eq: Sequence[float], bond: Sequence[float], reset: Sequence[bool],
               target: float = TARGET_EQUITY) -> List[float]:
    """Pre-reset distance of the equity weight from target, every day, for a
    stock/bond portfolio that drifts with returns and resets to target at the
    close of each day flagged in `reset` (their "Distance_t"). out[0] = 0."""
    n = len(eq)
    out = [0.0] * n
    w = target
    for i in range(1, n):
        ge, gb = eq[i] / eq[i - 1], bond[i] / bond[i - 1]
        pre = w * ge / (w * ge + (1.0 - w) * gb)
        out[i] = pre - target
        w = target if reset[i] else pre
    return out


def threshold_path(eq: Sequence[float], bond: Sequence[float], deltas: Sequence[float] = THRESHOLD_DELTAS,
                   target: float = TARGET_EQUITY) -> List[float]:
    """Their threshold signal: for each delta, a portfolio that resets when its
    distance reaches delta (either side); the signal is the pre-reset distance,
    averaged over the deltas (0..2%: heterogeneous investors' policies)."""
    n = len(eq)
    acc = [0.0] * n
    for delta in deltas:
        w = target
        for i in range(1, n):
            ge, gb = eq[i] / eq[i - 1], bond[i] / bond[i - 1]
            pre = w * ge / (w * ge + (1.0 - w) * gb)
            dist = pre - target
            acc[i] += dist
            w = target if abs(dist) >= delta else pre
    k = float(len(deltas))
    return [a / k for a in acc]


def month_end_flags(dates: Sequence[date], last_is_month_end: Optional[bool] = None) -> Tuple[List[bool], List[bool]]:
    """(month-end, quarter-end) flags for a sorted session list: a session is a
    month end when the next session is in another month. The last session's
    flag comes from the calendar (`last_is_month_end`), since its successor is
    not in the data yet."""
    n = len(dates)
    me = [False] * n
    for i in range(n - 1):
        me[i] = dates[i + 1].month != dates[i].month
    if n:
        me[-1] = bool(last_is_month_end) if last_is_month_end is not None else False
    qe = [me[i] and dates[i].month in ce.QUARTER_END_MONTHS for i in range(n)]
    return me, qe


def rebalance_signal(month_drift: float, quarter_drift: float, quarter_end: bool) -> float:
    """The calendar rebalancing signal for one month end: monthly rebalancers'
    drift, plus quarterly rebalancers' drift when the month end is a quarter end."""
    return month_drift + (quarter_drift if quarter_end else 0.0)


# ── sector scores ─────────────────────────────────────────────────────────────

def _series_at(series: Dict[str, float], days: Sequence[str], i: int) -> Optional[float]:
    if i < 0 or i >= len(days):
        return None
    v = series.get(days[i])
    return v if v and v > 0 and math.isfinite(v) else None


def relative_stats(sector: Dict[str, float], bench: Dict[str, float], days: Sequence[str], i: int,
                   vol_window: int = 252) -> Optional[Dict[str, float]]:
    """z-scores of the sector's 126- and 21-session return relative to SPY
    (log returns), in units of the relative return's own daily volatility over
    the last vol_window sessions. None when the history is too short."""
    if i < max(vol_window, 126):
        return None
    rr = []
    for j in range(i - vol_window + 1, i + 1):
        s0, s1 = _series_at(sector, days, j - 1), _series_at(sector, days, j)
        b0, b1 = _series_at(bench, days, j - 1), _series_at(bench, days, j)
        if None in (s0, s1, b0, b1):
            continue
        rr.append(math.log(s1 / s0) - math.log(b1 / b0))
    if len(rr) < vol_window * 0.8:
        return None
    mu = sum(rr) / len(rr)
    sd = math.sqrt(sum((x - mu) ** 2 for x in rr) / (len(rr) - 1))
    if sd <= 0:
        return None
    out = {"rel_vol": sd}
    for n in (126, 21):
        s0, s1 = _series_at(sector, days, i - n), _series_at(sector, days, i)
        b0, b1 = _series_at(bench, days, i - n), _series_at(bench, days, i)
        if None in (s0, s1, b0, b1):
            return None
        rel = math.log(s1 / s0) - math.log(b1 / b0)
        out[f"rel{n}"] = rel
        out[f"z{n}"] = rel / (sd * math.sqrt(n))
    return out


def rotation_score(stats: Optional[Dict[str, float]], cfg: Optional[Dict] = None) -> Optional[Dict[str, float]]:
    """momentum in [-1, 1] = clip(z126 / 2); crowding in [0, 1] = clip((z21 - 2) / 2);
    rotation = clip(W_MOM x momentum - W_CROWD x crowding)."""
    if not stats:
        return None
    mom = _clip(stats["z126"] / 2.0)
    crowd = _clip((stats["z21"] - 2.0) / 2.0, 0.0, 1.0)
    rot = _clip(float(cfg_value(cfg, "FLOWS_W_MOMENTUM")) * mom - float(cfg_value(cfg, "FLOWS_W_CROWDING")) * crowd)
    return {"momentum": mom, "crowding": crowd, "rotation": rot}


def tax_loss_scores(ytd: Dict[str, Optional[float]], selling: bool, n_losers: int = 3) -> Dict[str, float]:
    """The YTD-worst sectors with a YTD LOSS: -1, -2/3, -1/3 while funds sell
    (selling=True), the same with a + sign in the rebound window. A sector up
    on the year has no loss to harvest and scores 0."""
    ranked = sorted(((r, s) for s, r in ytd.items() if r is not None), key=lambda t: t[0])
    out = {s: 0.0 for s in ytd}
    for rank, (r, s) in enumerate(ranked[:n_losers]):
        if r < 0:
            mag = (n_losers - rank) / float(n_losers)
            out[s] = -mag if selling else mag
    return out


# ── the estimate ──────────────────────────────────────────────────────────────

def _fmt_pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


def _evidence_conf(parts: Iterable[Tuple[str, float]]) -> float:
    tot = sum(abs(v) for _, v in parts)
    if tot <= 1e-9:
        return 0.0
    return round(sum(abs(v) * EVIDENCE.get(k, 0.0) for k, v in parts) / tot * min(1.0, tot * 2), 2)


def compute_flows(closes: Dict[str, List[Tuple[str, float]]], as_of: Optional[date] = None,
                  cfg: Optional[Dict] = None, now: Optional[datetime] = None) -> Dict:
    """The full estimate from completed daily closes {symbol: [(YYYY-MM-DD, close)]}.
    Pure. `as_of` defaults to the last SPY date. Missing inputs degrade the
    estimate (named in inputs.missing); they never fabricate a reading."""
    bond_sym = str(cfg_value(cfg, "FLOWS_BOND_PROXY"))
    horizon = int(cfg_value(cfg, "FLOWS_HORIZON_SESSIONS"))
    series = {s: {str(d)[:10]: float(c) for d, c in rows if c and float(c) > 0}
              for s, rows in (closes or {}).items()}
    bench = series.get(BENCH) or {}
    bond = series.get(bond_sym) or {}
    missing = [s for s in [BENCH, bond_sym] + [e for _, e in SECTORS] if not series.get(s)]
    spy_days = sorted(d for d in bench if as_of is None or _d(d) <= as_of)
    if len(spy_days) < 30:
        return {"ok": False, "label": LABEL, "error": f"need {BENCH} daily closes, have {len(spy_days)}",
                "inputs": {"missing": missing, "bond_proxy": bond_sym}}
    as_of = _d(spy_days[-1])          # the last completed session at or before as_of
    nxt = ce.sessions_after(as_of, horizon)

    # ── equity drivers ──
    eq_days = [d for d in spy_days if d in bond]
    drivers: List[Dict] = []
    reb_parts: List[Tuple[str, float]] = []
    per_session = [0.0] * horizon
    w_r = float(cfg_value(cfg, "FLOWS_W_REBALANCE"))
    w_t = float(cfg_value(cfg, "FLOWS_W_THRESHOLD"))
    w_m = float(cfg_value(cfg, "FLOWS_W_MONTH_END"))
    rebal_scale = float(cfg_value(cfg, "FLOWS_REBAL_SCALE"))
    thr_scale = float(cfg_value(cfg, "FLOWS_THRESHOLD_SCALE"))
    signals: Dict[str, Optional[float]] = {"month_drift": None, "quarter_drift": None, "threshold": None}
    if len(eq_days) >= 30 and eq_days[-1] == spy_days[-1]:
        dts = [_d(d) for d in eq_days]
        eq = [bench[d] for d in eq_days]
        bd = [bond[d] for d in eq_days]
        me, qe = month_end_flags(dts, last_is_month_end=(as_of == ce.month_last_session(as_of.year, as_of.month)))
        m_path = drift_path(eq, bd, me)
        q_path = drift_path(eq, bd, qe)
        th_path = threshold_path(eq, bd)
        signals = {"month_drift": m_path[-1], "quarter_drift": q_path[-1], "threshold": th_path[-1]}
        # rebalancing: session x_j is affected when its signal day (the session
        # before it) is in the last REBALANCE_WINDOW_SESSIONS of a month
        sig_days = [as_of] + nxt[:-1]
        reb_vals, reb_end = [], None
        for j, t in enumerate(sig_days):
            win = ce.rebalance_window(t)
            if not win:
                reb_vals.append(0.0)
                continue
            s = rebalance_signal(signals["month_drift"], signals["quarter_drift"], win["quarter_end"])
            v = -_clip(s / rebal_scale)
            reb_vals.append(v)
            reb_end = reb_end or win
        reb = sum(reb_vals) / horizon
        for j, v in enumerate(reb_vals):
            per_session[j] += w_r * v
        if reb_end:
            s_now = rebalance_signal(signals["month_drift"], signals["quarter_drift"], reb_end["quarter_end"])
            kind = "quarter-end" if reb_end["quarter_end"] else "month-end"
            who = "monthly and quarterly" if reb_end["quarter_end"] else "monthly"
            reason = (f"Stocks have {'beaten' if s_now > 0 else 'trailed'} bonds since the last rebalance: a 60/40 fund "
                      f"is {abs(s_now) * 100:.1f} points {'over' if s_now > 0 else 'under'}weight stocks, so {who} "
                      f"rebalancers {'sell' if s_now > 0 else 'buy'} stocks into the {kind} on "
                      f"{reb_end['month_end'].isoformat()} (Harvey, Mazzoleni & Melone 2025).")
        else:
            nxt_end = ce.month_last_session(as_of.year, as_of.month)
            reason = (f"No month-end rebalancing in the next {horizon} sessions (next month end "
                      f"{nxt_end.isoformat()}; current drift {signals['month_drift'] * 100:+.1f} points).")
        drivers.append({"name": "Pension/balanced-fund rebalancing", "key": "rebalance", "value": round(reb, 3),
                        "weight": w_r, "reason": reason,
                        "inputs": {"month_drift_pts": round(signals["month_drift"] * 100, 2),
                                   "quarter_drift_pts": round(signals["quarter_drift"] * 100, 2),
                                   "bond_proxy": bond_sym}})
        reb_parts.append(("rebalance", w_r * reb))
        thr = -_clip(signals["threshold"] / thr_scale)
        per_session[0] += w_t * thr
        thr_h = thr / horizon
        drivers.append({"name": "Threshold rebalancing", "key": "threshold", "value": round(thr_h, 3),
                        "weight": w_t,
                        "reason": (f"Funds that rebalance on a drift band (0-2%) are on average "
                                   f"{signals['threshold'] * 100:+.2f} points from target: "
                                   f"{'selling' if signals['threshold'] > 0 else 'buying'} pressure on the next session."),
                        "inputs": {"threshold_pts": round(signals["threshold"] * 100, 3)}})
        reb_parts.append(("threshold", w_t * thr_h))
    else:
        drivers.append({"name": "Pension/balanced-fund rebalancing", "key": "rebalance", "value": 0.0, "weight": w_r,
                        "reason": f"Not computed: no aligned {BENCH}/{bond_sym} closes through {as_of.isoformat()}."})
    # month-end cash needs
    phase_val = {"cash_sell": -1.0, "reversal": 1.0, "new_month_buy": 1.0}
    phases = [ce.month_end_phase(x) for x in nxt]
    me_vals = [phase_val.get(p, 0.0) for p in phases]
    for j, v in enumerate(me_vals):
        per_session[j] += w_m * v
    me_avg = sum(me_vals) / horizon if horizon else 0.0
    counts = {p: phases.count(p) for p in ("cash_sell", "reversal", "new_month_buy") if phases.count(p)}
    if counts:
        words = {"cash_sell": "month-end cash raise (selling)", "reversal": "month-end reversal (buying)",
                 "new_month_buy": "new-month inflows (buying)"}
        reason = ("Next sessions: " + ", ".join(f"{n} in the {words[p]}" for p, n in counts.items())
                  + f" (Etula et al. 2020; windows set for T+{ce.settlement_days(as_of)} settlement).")
    else:
        reason = f"None of the next {horizon} sessions is in a month-end cash window."
    drivers.append({"name": "Month-end cash needs", "key": "month_end", "value": round(me_avg, 3), "weight": w_m,
                    "reason": reason})
    reb_parts.append(("month_end", w_m * me_avg))
    eq_pressure = _clip(sum(v for _, v in reb_parts))
    # index events: named, no direction
    cal = ce.calendar_window(as_of)
    horizon_end = nxt[-1] if nxt else as_of
    idx_events = [r for r in cal if r["kind"] in ("sp500_rebalance", "russell_recon", "ndx_recon")
                  and _d(r["date"]) <= horizon_end]
    for r in idx_events:
        drivers.append({"name": r["event"], "key": "index_event", "value": 0.0, "weight": 0.0,
                        "reason": f"{r['date']}: {r['effect']}"})
    equity = {"pressure": round(eq_pressure, 3), "direction": direction_of(eq_pressure),
              "confidence": _evidence_conf(reb_parts), "drivers": drivers,
              "sessions": [{"date": x.isoformat(), "pressure": round(_clip(p), 3), "phase": ph}
                           for x, p, ph in zip(nxt, per_session, phases)]}

    # ── sector drivers ──
    days_all = spy_days
    i_last = len(days_all) - 1
    w_rot = float(cfg_value(cfg, "FLOWS_W_ROTATION"))
    w_tax = float(cfg_value(cfg, "FLOWS_W_TAX"))
    w_broad = float(cfg_value(cfg, "FLOWS_W_BROAD"))
    tax_win = None
    for x in nxt:
        tw = ce.tax_window(x)
        if tw:
            tax_win = tw
            break
    selling = tax_win in ("fund_tax_loss", "december_tax_loss")
    rebound = tax_win in ("fund_tax_loss_reversal", "turn_of_year")
    # YTD basis: last close of the prior year; in a rebound window, ranked as of
    # the end of the selling window (Oct 31 / Dec 23) — the positions funds sold
    year = as_of.year if not (as_of.month == 1 and tax_win == "turn_of_year") else as_of.year - 1
    base_day = max((d for d in days_all if _d(d) < date(year, 1, 1)), default=None)
    if tax_win == "fund_tax_loss_reversal":
        rank_day = max((d for d in days_all if _d(d) <= date(year, 10, 31)), default=None)
    elif tax_win == "turn_of_year":
        rank_day = max((d for d in days_all if _d(d) <= date(year, 12, 23)), default=None)
    else:
        rank_day = days_all[-1]
    ytd: Dict[str, Optional[float]] = {}
    for name, etf in SECTORS:
        s = series.get(etf) or {}
        if base_day and rank_day and s.get(base_day) and s.get(rank_day):
            ytd[name] = s[rank_day] / s[base_day] - 1.0
        else:
            ytd[name] = None
    tax = tax_loss_scores(ytd, selling=selling) if (selling or rebound) else {n: 0.0 for n, _ in SECTORS}
    tax_factor = float((cfg_value(cfg, "FLOWS_TAX_WINDOW_WEIGHTS") or {}).get(tax_win, 0.0)) if tax_win else 0.0
    ndx_in = any(r["kind"] == "ndx_recon" for r in idx_events)
    sectors = []
    for name, etf in SECTORS:
        s = series.get(etf) or {}
        st = relative_stats(s, bench, days_all, i_last) if s else None
        rs = rotation_score(st, cfg)
        sd: List[Dict] = []
        parts: List[Tuple[str, float]] = []
        if rs:
            sd.append({"name": "Sector momentum (6 months vs SPY)", "key": "momentum",
                       "value": round(rs["momentum"], 3), "weight": w_rot,
                       "reason": (f"{etf} {_fmt_pct(math.exp(st['rel126']) - 1)} vs SPY over 6 months "
                                  f"(z {st['z126']:+.1f}): money has been moving "
                                  f"{'into' if st['rel126'] > 0 else 'out of'} the sector "
                                  f"(industry momentum, Moskowitz & Grinblatt 1999).")})
            parts.append(("momentum", w_rot * float(cfg_value(cfg, "FLOWS_W_MOMENTUM")) * rs["momentum"]))
            if rs["crowding"] > 0:
                sd.append({"name": "Crowding after a big run", "key": "crowding", "value": round(-rs["crowding"], 3),
                           "weight": w_rot,
                           "reason": (f"{etf} ran {_fmt_pct(math.exp(st['rel21']) - 1)} ahead of SPY in a month "
                                      f"(z {st['z21']:+.1f} > 2): crowded, so the tilt is cut.")})
                parts.append(("crowding", -w_rot * float(cfg_value(cfg, "FLOWS_W_CROWDING")) * rs["crowding"]))
        else:
            sd.append({"name": "Sector momentum (6 months vs SPY)", "key": "momentum", "value": 0.0, "weight": w_rot,
                       "reason": f"Not computed: {etf} has under a year of aligned closes."})
        tv = tax.get(name, 0.0)
        if tax_win and ytd.get(name) is not None:
            if tv:
                verb = "sell" if selling else "buy back"
                when = {"fund_tax_loss": "before their October 31 year end", "december_tax_loss": "before December 31",
                        "fund_tax_loss_reversal": "after the October selling", "turn_of_year": "in early January"}[tax_win]
                caveat = ("" if tax_factor >= 1.0 else
                          f" Weighted x{tax_factor:.2f}: our 1999-2025 check found loser sectors beat the average in "
                          "October (the opposite sign) and only a weak November rebound.")
                src = ("Gibson, Safieddine & Titman 2000" if tax_win.startswith("fund_")
                       else "calendar-year tax selling, a long-studied hypothesis")
                sd.append({"name": "Tax-loss selling" if selling else "Tax-loss rebound", "key": "tax_loss",
                           "value": round(tv, 3), "weight": round(w_tax * tax_factor, 4),
                           "reason": (f"{etf} is {_fmt_pct(ytd[name])} on the year, among the worst sectors: funds "
                                      f"{verb} losers {when} ({src}).{caveat}")})
                parts.append(("tax_loss", w_tax * tax_factor * tv))
        broad = w_broad * eq_pressure
        if abs(eq_pressure) > 1e-9:
            sd.append({"name": "Broad equity flows", "key": "broad", "value": round(eq_pressure, 3), "weight": w_broad,
                       "reason": f"Every sector shares the broad-market estimate ({direction_of(eq_pressure)})."})
            parts.append(("broad", broad))
        if ndx_in and name in NASDAQ100_HEAVY:
            sd.append({"name": "Nasdaq-100 reconstitution", "key": "index_event", "value": 0.0, "weight": 0.0,
                       "reason": "Index funds trade the Nasdaq-100 changes in this window; direction depends on the changes."})
        p = _clip(sum(v for _, v in parts))
        sectors.append({"sector": name, "etf": etf, "pressure": round(p, 3), "direction": direction_of(p),
                        "confidence": _evidence_conf(parts), "drivers": sd,
                        "ytd": None if ytd.get(name) is None else round(ytd[name], 4)})
    sectors.sort(key=lambda r: r["pressure"])

    notes = [
        "ESTIMATE, not a forecast of record: the size of each effect is small and its evidence is mixed "
        "(confidence is the evidence-weighted share of the drivers).",
        f"Rebalancing drift is a 60/40 {BENCH}/{bond_sym} portfolio reset at each month (quarter) end; "
        "real funds' targets and timing vary.",
        "Index rebalances and reconstitutions are named but carry no direction: the calendar alone does not say "
        "which way the flow goes.",
        "13F / fund-holdings data is not used (the repo's 13F pipeline is at ladder gate 1, focused managers only).",
    ]
    if missing:
        notes.append("Missing inputs: " + ", ".join(missing) + ".")
    return {
        "ok": True, "label": LABEL, "version": SCHEMA_VERSION, "asOf": as_of.isoformat(),
        "computedAt": (now or _utcnow()).replace(microsecond=0).isoformat() + "Z",
        "horizonDays": horizon, "horizonEnd": horizon_end.isoformat(),
        "equity": equity, "sectors": sectors, "calendar": cal, "notes": notes,
        "signals": {k: (None if v is None else round(v, 6)) for k, v in signals.items()},
        "inputs": {"bond_proxy": bond_sym, "missing": missing, "spy_sessions": len(spy_days),
                   "holiday_source": ce.holiday_source(as_of)},
    }


# ── hooks (pure) ──────────────────────────────────────────────────────────────

def _stale(doc: Optional[Dict], today: date, max_age_days: int) -> Optional[str]:
    if not doc or not doc.get("ok"):
        return "no flows estimate"
    try:
        age = (today - _d(doc["asOf"])).days
    except (KeyError, ValueError, TypeError):
        return "flows estimate has no as-of date"
    if age > max_age_days:
        return f"flows estimate is {age} days old (max {max_age_days})"
    return None


def floor_multiplier(doc: Optional[Dict], cfg: Optional[Dict] = None, today: Optional[date] = None) -> Dict:
    """Multiplier on the floor's target exposure: 1 - MAX_CUT x (selling
    pressure beyond the deadband, rescaled to 0..1). Always in
    [1 - MAX_CUT, 1.0]: it can only cut the floor's exposure, never raise it,
    so it never adds leverage."""
    max_cut = _clip(float(cfg_value(cfg, "FLOWS_FLOOR_MAX_CUT")), 0.0, 0.5)
    band = _clip(float(cfg_value(cfg, "FLOWS_FLOOR_DEADBAND")), 0.0, 0.9)
    out = {"multiplier": 1.0, "applied": False, "pressure": None, "asOf": None, "label": LABEL}
    if not (cfg_value(cfg, "FLOWS_ENABLED") and cfg_value(cfg, "FLOWS_FLOOR_ENABLED")):
        out["reason"] = "flows floor adjustment off (FLOWS_FLOOR_ENABLED)"
        return out
    why = _stale(doc, today or date.today(), int(cfg_value(cfg, "FLOWS_MAX_AGE_DAYS")))
    if why:
        out["reason"] = why + ": no adjustment"
        return out
    p = float(doc["equity"]["pressure"])
    out.update(pressure=p, asOf=doc.get("asOf"))
    sell = max(0.0, -p - band) / (1.0 - band) if band < 1 else 0.0
    m = round(1.0 - max_cut * min(1.0, sell), 4)
    m = min(1.0, max(1.0 - max_cut, m))
    out["multiplier"] = m
    out["applied"] = m < 1.0
    top = sorted((d for d in doc["equity"].get("drivers", []) if d.get("value")),
                 key=lambda d: d["value"] * d.get("weight", 1.0))[:2]
    because = "; ".join(d["reason"] for d in top) if top else "no driver"
    if m < 1.0:
        out["reason"] = (f"Expected net equity selling (pressure {p:+.2f}, {LABEL.lower()}): floor target x{m:.2f}. "
                         f"{because}")
    else:
        out["reason"] = f"No forced selling expected beyond the deadband (pressure {p:+.2f}): floor target x1.00."
    return out


def sized_shares(shares: int, price: float, equity: float, multiplier: float, max_position_pct: float) -> int:
    """Whole shares after the sector multiplier. A cut rounds down; a raise
    never takes the position above max_position_pct of equity (the sizer's own
    cap) and never below the sizer's number."""
    if shares <= 0 or price <= 0:
        return max(0, int(shares))
    new = int(math.floor(shares * max(0.0, multiplier) + 1e-9))
    if multiplier > 1.0:
        cap = int(math.floor(max(0.0, equity) * max(0.0, max_position_pct) / price + 1e-9))
        new = max(int(shares), min(new, cap))
    return new


def canonical_sector(sector: Optional[str], ticker: Optional[str] = None) -> Optional[str]:
    if ticker and str(ticker).upper() in SEMICONDUCTOR_TICKERS:
        return "Semiconductors"
    if not sector:
        return None
    return SECTOR_ALIASES.get(str(sector).strip().lower())


def sector_size_adjustment(doc: Optional[Dict], sector: Optional[str], ticker: Optional[str] = None,
                           cfg: Optional[Dict] = None, today: Optional[date] = None) -> Dict:
    """Share multiplier for a satellite candidate: 1 + K x its sector's
    pressure, in [1 - K, 1 + K]. The caller still caps the position at the
    sizer's maximum; this never changes a stop or a target."""
    k = _clip(float(cfg_value(cfg, "FLOWS_SECTOR_SIZE_K")), 0.0, 0.5)
    name = canonical_sector(sector, ticker)
    out = {"multiplier": 1.0, "applied": False, "sector": name, "etf": SECTOR_ETF.get(name) if name else None,
           "pressure": None, "label": LABEL}
    if not (cfg_value(cfg, "FLOWS_ENABLED") and cfg_value(cfg, "FLOWS_SECTOR_SIZING_ENABLED")):
        out["reason"] = "flows sector sizing off (FLOWS_SECTOR_SIZING_ENABLED)"
        return out
    why = _stale(doc, today or date.today(), int(cfg_value(cfg, "FLOWS_MAX_AGE_DAYS")))
    if why:
        out["reason"] = why + ": no adjustment"
        return out
    if not name:
        out["reason"] = f"sector {sector!r} has no sector ETF reading: no adjustment"
        return out
    row = next((r for r in doc.get("sectors", []) if r.get("sector") == name), None)
    if not row:
        out["reason"] = f"no flows reading for {name}: no adjustment"
        return out
    p = float(row["pressure"])
    m = round(1.0 + k * _clip(p), 4)
    out.update(multiplier=m, applied=abs(m - 1.0) > 1e-9, pressure=p, etf=row.get("etf"))
    top = max(row.get("drivers", []), key=lambda d: abs(d.get("value", 0) * d.get("weight", 1)), default=None)
    out["reason"] = (f"{name} ({row.get('etf')}) flow pressure {p:+.2f} ({row.get('direction')}, "
                     f"{LABEL.lower()}): size x{m:.2f}." + (f" {top['reason']}" if top else ""))
    return out


# ── I/O ───────────────────────────────────────────────────────────────────────

def _data_dir(data_dir: Optional[str] = None) -> str:
    if data_dir:
        return data_dir
    from storage_config import DATA_DIR
    return DATA_DIR


def _cfg() -> Dict:
    try:
        from system_config import BASE_CONFIG
        return {k: BASE_CONFIG[k] for k in DEFAULTS if k in BASE_CONFIG}
    except ImportError as e:
        log.warning(f"[FLOWS] system_config unavailable, defaults used: {e}")
        return {}


def universe(cfg: Optional[Dict] = None) -> List[str]:
    return [BENCH, str(cfg_value(cfg, "FLOWS_BOND_PROXY"))] + [e for _, e in SECTORS]


def last_completed_session(now_et: datetime) -> date:
    """Today once the session has closed (16:30 ET, a margin past 16:00), else
    the session before. Half days close at 13:00, so 16:30 is safe too."""
    d = now_et.date()
    if ce.is_trading_day(d) and (now_et.hour, now_et.minute) >= (16, 30):
        return d
    return ce.prev_session(d)


def completed_only(closes: Dict[str, List[Tuple[str, float]]], last_session: date) -> Dict[str, List[Tuple[str, float]]]:
    """Drop any bar dated after the last completed session (an in-progress daily bar)."""
    cut = last_session.isoformat()
    return {s: [(d, c) for d, c in rows if str(d)[:10] <= cut] for s, rows in closes.items()}


def fetch_closes(symbols: Sequence[str], days: int = 760,
                 fetch: Optional[Callable[[Sequence[str], int], Dict]] = None) -> Dict[str, List[Tuple[str, float]]]:
    """{symbol: [(date, close)]} adjusted (splits + dividends) daily closes from
    Alpaca's batched bars endpoint (etf_data_sources.fetch_bars_alpaca, the
    regime engine's fetcher). Injectable for tests."""
    if fetch is None:
        from etf_data_sources import fetch_bars_alpaca as fetch
    got = fetch(list(symbols), days) or {}
    return {s.upper(): [(str(d)[:10], float(c)) for d, c in rows] for s, rows in got.items() if rows}


def _write_json(path: str, payload: Dict) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(payload, f, default=str)
    os.replace(tmp, path)


def load_cached(data_dir: Optional[str] = None) -> Optional[Dict]:
    path = os.path.join(_data_dir(data_dir), FLOWS_FILE)
    try:
        with open(path) as f:
            doc = json.load(f)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as e:
        log.warning(f"[FLOWS] {path} unreadable: {e}")
        return None
    return doc if isinstance(doc, dict) else None


def history_row(doc: Dict) -> Dict:
    """One line of voltrade_flows_history.jsonl: what was estimated, so it can be
    scored later against the realized returns over [asOf, horizonEnd]."""
    return {"asOf": doc.get("asOf"), "computedAt": doc.get("computedAt"), "horizonEnd": doc.get("horizonEnd"),
            "equity": {"pressure": doc["equity"]["pressure"], "direction": doc["equity"]["direction"],
                       "confidence": doc["equity"]["confidence"],
                       "drivers": {d["key"]: d["value"] for d in doc["equity"]["drivers"] if d.get("key") != "index_event"}},
            "floor_multiplier": (doc.get("floor") or {}).get("multiplier"),
            "sectors": {r["etf"]: r["pressure"] for r in doc.get("sectors", [])},
            "signals": doc.get("signals")}


RECOMPUTE_MIN_MINUTES = 50


def _age_minutes(stamp: Optional[str]) -> float:
    try:
        t = datetime.fromisoformat(str(stamp).replace("Z", ""))
    except (TypeError, ValueError):
        return float("inf")
    return (_utcnow() - t).total_seconds() / 60.0


def refresh_daily(force: bool = False, data_dir: Optional[str] = None, fetch=None, now_et: Optional[datetime] = None) -> Dict:
    """Recompute the estimate when a new session has completed since the cached
    one (or `force`), write voltrade_flows.json, append the history line. Cheap
    when nothing new completed (one file read). Never raises: a failure keeps
    the last good file and is returned as status "error"."""
    cfg = _cfg()
    if not cfg_value(cfg, "FLOWS_ENABLED"):
        return {"status": "off", "reason": "FLOWS_ENABLED is False"}
    try:
        if now_et is None:
            from zoneinfo import ZoneInfo
            now_et = datetime.now(ZoneInfo("America/New_York"))
        last = last_completed_session(now_et)
        ddir = _data_dir(data_dir)
        cached = load_cached(ddir)
        if not force and cached and cached.get("ok") and cached.get("asOf") == last.isoformat():
            return {"status": "cached", "asOf": cached["asOf"], "pressure": cached["equity"]["pressure"]}
        if not force and cached and cached.get("ok") and _age_minutes(cached.get("computedAt")) < RECOMPUTE_MIN_MINUTES:
            # the newest session's bars are not in yet: retry on a later hourly call, not every call
            return {"status": "cached_recent", "asOf": cached["asOf"], "pressure": cached["equity"]["pressure"]}
        raw = fetch_closes(universe(cfg), fetch=fetch)
        closes = completed_only(raw, last)
        doc = compute_flows(closes, cfg=cfg, now=_utcnow())
        if not doc.get("ok"):
            return {"status": "error", "reason": doc.get("error"), "asOf": cached.get("asOf") if cached else None}
        if not force and cached and cached.get("ok") and cached.get("asOf") == doc["asOf"]:
            # Alpaca's batched daily bars end at UTC midnight, so the session that
            # closed at 16:00 ET is not served until ~20:00 ET: keep the file (and
            # the history) as is and try again on the next hourly call
            return {"status": "no_new_session", "asOf": doc["asOf"], "pressure": cached["equity"]["pressure"],
                    "reason": f"bars end {doc['asOf']}; {last.isoformat()} is not in yet"}
        doc["floor"] = floor_multiplier(doc, cfg, today=last)
        doc["config"] = {k: cfg_value(cfg, k) for k in DEFAULTS}
        _write_json(os.path.join(ddir, FLOWS_FILE), doc)
        with open(os.path.join(ddir, FLOWS_HISTORY_FILE), "a") as f:
            f.write(json.dumps(history_row(doc), default=str) + "\n")
        return {"status": "computed", "asOf": doc["asOf"], "pressure": doc["equity"]["pressure"],
                "direction": doc["equity"]["direction"], "floor_multiplier": doc["floor"]["multiplier"],
                "missing": doc["inputs"]["missing"]}
    except Exception as e:  # the scan and Tier 3 must never die on the estimate
        log.warning(f"[FLOWS] refresh failed: {type(e).__name__}: {e}")
        return {"status": "error", "reason": f"{type(e).__name__}: {str(e)[:200]}"}


def report(data_dir: Optional[str] = None) -> Dict:
    """The cached estimate as served (read-only, no network)."""
    doc = load_cached(data_dir)
    return doc if doc else {"ok": False, "label": LABEL, "error": "no flows estimate yet"}


if __name__ == "__main__":
    import sys
    print(json.dumps(refresh_daily(force="--force" in sys.argv), default=str))
