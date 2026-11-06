"""
VolTradeAI — fair value before paying for an option (2026-11-06)
================================================================
Human refocus spec §5 (2026-11-05): "estimate fair value before paying, do
not overpay on IV".

Every option entry is built by options_execution.select_contract, which
calls apply() last, before the contract goes back to its caller (and so
before submit_options_order sends anything). apply() compares the implied
volatility each judged leg trades at with a FAIR volatility estimated from
the underlying's own realized volatility:

    fair vol = rv_short_weight x RV(20 sessions) + (1 - rv_short_weight) x RV(60 sessions)

(weight 0.5; RV = annualized close-to-close volatility, from the vol surface
the selector already built, else from 90 daily bars). This is an ESTIMATE of
the volatility the option will see, not a model of the smile: the skew a put
carries over the at-the-money volatility counts as paying over fair, on
purpose (it is the part of the premium the variance-risk-premium literature
finds bought options overpay for).

  bought premium (buy_call, buy_put; the bought legs of a debit structure:
  a vertical's long leg, a straddle's two legs, a calendar's or diagonal's
  long leg): refused when the leg's IV is more than max_buy_premium (25%)
  over fair;
  sold premium (sell_cash_secured_put, covered_call; the sold legs of a
  credit structure: a condor's two short legs, a credit spread's short leg,
  a short straddle's two legs): refused when the leg's IV is below
  fair x min_sell_ratio (1.00) — selling volatility for less than the stock
  has been realizing.

The other legs of a structure (a credit spread's bought wing, a debit
spread's sold leg) are its insurance or its financing and are not judged.

Modes (system_config.OPTIONS_FAIR_VALUE["mode"]): "enforce" refuses (the
contract comes back with an `error`, which every caller already treats as no
order: the tier dispatcher audits T<n>-FAIL, the scanner path records a
rejected_options_nocontract gate, evaluate_and_execute falls back to the
stock); "flag" only records; "off" skips the check. Every judged entry gets
a plain-English `fair_value` block on the contract and one line in
DATA_DIR/voltrade_options_fair_value_log.jsonl. When fair value cannot be
estimated (no realized-vol history, no IV and no two-sided quote to solve
one from) the entry is allowed and the line says why: a missing input never
blocks trading, it is only made visible.
"""
import json
import logging
import math
import os
from datetime import datetime

logger = logging.getLogger("option_fair_value")

try:
    from storage_config import DATA_DIR
except ImportError:
    DATA_DIR = "/data/voltrade" if os.path.isdir("/data") else "/tmp"

LOG_FILE = "voltrade_options_fair_value_log.jsonl"
DEFAULTS = {"mode": "enforce", "max_buy_premium": 0.25, "min_sell_ratio": 1.00, "rv_short_weight": 0.5}
_BOUNDS = {"max_buy_premium": (0.0, 5.0), "min_sell_ratio": (0.0, 3.0), "rv_short_weight": (0.0, 1.0)}
MODES = ("enforce", "flag", "off")

# which legs carry the premium each strategy buys or sells (contract keys)
BOUGHT = "bought"
SOLD = "sold"
JUDGED_LEGS = {
    "buy_call": [("occ_symbol", BOUGHT)],
    "buy_put": [("occ_symbol", BOUGHT)],
    "sell_cash_secured_put": [("occ_symbol", SOLD)],
    "covered_call": [("occ_symbol", SOLD)],
    "bull_call_spread": [("long_leg", BOUGHT)],
    "bear_put_spread": [("long_leg", BOUGHT)],
    "calendar_spread": [("long_leg", BOUGHT)],
    "diagonal_spread": [("long_leg", BOUGHT)],
    "buy_straddle": [("call_leg", BOUGHT), ("put_leg", BOUGHT)],
    "short_straddle": [("call_leg", SOLD), ("put_leg", SOLD)],
    "bull_put_credit_spread": [("short_leg", SOLD)],
    "iron_condor": [("short_call", SOLD), ("short_put", SOLD)],
    "qqq_iron_condor": [("short_call", SOLD), ("short_put", SOLD)],
}


def settings() -> dict:
    """system_config.BASE_CONFIG["OPTIONS_FAIR_VALUE"] over the defaults; an
    unknown mode or an out-of-bounds number keeps the default (logged)."""
    cfg = dict(DEFAULTS)
    try:
        from system_config import BASE_CONFIG
        raw = BASE_CONFIG.get("OPTIONS_FAIR_VALUE", {})
    except Exception as e:
        logger.warning(f"system_config unreadable, the fair-value check runs on its defaults: {e}")
        raw = {}
    if not isinstance(raw, dict):
        raw = {}
    mode = str(raw.get("mode", cfg["mode"])).strip().lower()
    if mode in MODES:
        cfg["mode"] = mode
    else:
        logger.warning(f"OPTIONS_FAIR_VALUE mode {raw.get('mode')!r} is not one of {MODES}: {cfg['mode']} used")
    for key, (lo, hi) in _BOUNDS.items():
        if key not in raw:
            continue
        try:
            value = float(raw[key])
        except (TypeError, ValueError):
            value = float("nan")
        if math.isfinite(value) and lo <= value <= hi:
            cfg[key] = value
        else:
            logger.warning(f"OPTIONS_FAIR_VALUE[{key!r}]={raw[key]!r} is outside [{lo}, {hi}]: default {cfg[key]} used")
    return cfg


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def realized_vols(ticker: str, surface: dict = None):
    """(RV20, RV60, source): from the selector's vol surface when it carries
    them, else from 90 daily bars (one request); (None, None, why) otherwise."""
    vrp = (surface or {}).get("vrp") if isinstance(surface, dict) else None
    if isinstance(vrp, dict):
        rv20, rv60 = _num(vrp.get("realized_vol_20d")), _num(vrp.get("realized_vol_60d"))
        if (rv20 or 0) > 0 or (rv60 or 0) > 0:
            return (rv20 if (rv20 or 0) > 0 else None), (rv60 if (rv60 or 0) > 0 else None), "vol surface"
    try:
        import vol_surface
        bars = vol_surface._fetch_historical_bars(ticker, 90)
        rv20 = vol_surface._compute_realized_vol(bars, 20)
        rv60 = vol_surface._compute_realized_vol(bars, 60)
    except Exception as e:
        logger.warning(f"realized-vol bars unavailable for {ticker}: {type(e).__name__}: {e}")
        return None, None, f"daily bars unavailable ({type(e).__name__})"
    if (rv20 or 0) <= 0 and (rv60 or 0) <= 0:
        return None, None, f"not enough daily bars ({len(bars or [])})"
    return (rv20 if (rv20 or 0) > 0 else None), (rv60 if (rv60 or 0) > 0 else None), "daily bars"


def fair_vol(rv20, rv60, short_weight: float):
    """The blend of the two realized vols (the one that exists, alone)."""
    if rv20 and rv60:
        return short_weight * rv20 + (1.0 - short_weight) * rv60
    return rv20 or rv60 or None


def _leg_quote(occ: str, contract: dict, key: str, chain: list) -> dict:
    """The leg's chain row (iv, bid, ask, strike, expiry, type, DTE); a single
    leg falls back to the contract's own fields."""
    for c in chain or []:
        if isinstance(c, dict) and c.get("occ_symbol") == occ:
            return c
    if key == "occ_symbol":
        return {"occ_symbol": occ, "iv": contract.get("iv"), "strike": contract.get("strike"),
                "option_type": contract.get("option_type"), "days_to_expiry": contract.get("days_to_expiry"),
                "bid": contract.get("bid"), "ask": contract.get("ask"), "mid": contract.get("limit_price")}
    return {}


def _solve_iv(price: float, q: dict, spot: float):
    """Black-Scholes IV of `price` for this leg (vol_surface.implied_vol), or None."""
    strike, dte = _num(q.get("strike")), _num(q.get("days_to_expiry"))
    if not price or price <= 0 or not strike or dte is None or not spot:
        return None
    try:
        import vol_surface
        solved = vol_surface.implied_vol(price, spot, strike, max(dte, 0.5) / 365.0, vol_surface.RISK_FREE_RATE,
                                         "call" if str(q.get("option_type", "")).lower().startswith("c") else "put")
    except Exception as e:
        logger.debug(f"IV not solved for {q.get('occ_symbol')}: {e}")
        return None
    return solved if solved and solved > 0 else None


def _leg_iv(q: dict, spot: float, limit: float = None):
    """(iv, how): a single leg is judged at the price the order will carry
    (its limit: the mid walked toward the natural side), a structure's leg at
    the quote's IV; failing those, an IV solved from the quote's mid."""
    if limit:
        at_limit = _solve_iv(limit, q, spot)
        if at_limit:
            return at_limit, f"at the ${limit:.2f} limit"
    iv = _num(q.get("iv"))
    if iv and iv > 0:
        return iv, "quoted"
    bid, ask = _num(q.get("bid")) or 0.0, _num(q.get("ask")) or 0.0
    mid = (bid + ask) / 2.0 if bid > 0 and ask >= bid else (_num(q.get("mid")) or 0.0)
    if mid <= 0:
        return None, "no IV and no two-sided quote to solve one from"
    solved = _solve_iv(mid, q, spot)
    return (solved, "solved from the mid") if solved else (None, "IV could not be solved from the mid")


def _fair_price(q: dict, spot: float, vol: float):
    strike, dte = _num(q.get("strike")), _num(q.get("days_to_expiry"))
    if not strike or dte is None or not spot or not vol:
        return None
    try:
        import vol_surface
        return vol_surface.bs_price(spot, strike, max(dte, 0.5) / 365.0, vol_surface.RISK_FREE_RATE, vol,
                                    "call" if str(q.get("option_type", "")).lower().startswith("c") else "put")
    except Exception as e:
        logger.debug(f"fair price not computed: {e}")
        return None


def check(contract: dict, strategy: str, ticker: str, spot: float, surface: dict = None,
          chain: list = None, cfg: dict = None) -> dict:
    """The fair-value verdict for one constructed entry: {"verdict": "fair" |
    "overpriced" | "underpriced" | "unpriced" | "not_judged", "allowed",
    "reason", "fair_vol", "rv20", "rv60", "legs": [...]} — pure apart from the
    realized-vol lookup and the vol_surface math. A single leg is judged at
    the IV its own limit price implies (what the order will pay or receive);
    a structure's leg at its quoted IV (the net limit is not split by leg)."""
    cfg = cfg or settings()
    judged = JUDGED_LEGS.get(strategy)
    out = {"strategy": strategy, "ticker": ticker, "mode": cfg["mode"], "spot": spot,
           "max_buy_premium": cfg["max_buy_premium"], "min_sell_ratio": cfg["min_sell_ratio"]}
    if not judged:
        return {**out, "verdict": "not_judged", "allowed": True,
                "reason": f"FAIR VALUE: {strategy} is not a strategy this check knows how to judge — entry allowed unjudged"}
    rv20, rv60, source = realized_vols(ticker, surface)
    fv = fair_vol(rv20, rv60, cfg["rv_short_weight"])
    out.update(rv20=rv20, rv60=rv60, rv_source=source, fair_vol=fv)
    if not fv:
        return {**out, "verdict": "unpriced", "allowed": True,
                "reason": f"FAIR VALUE: not estimated for {ticker} — {source}; entry allowed unpriced"}
    rv_words = ", ".join(w for w in ((f"20-day {rv20:.0%}" if rv20 else ""), (f"60-day {rv60:.0%}" if rv60 else "")) if w)
    legs, refusals = [], []
    for key, kind in judged:
        occ = contract.get(key)
        if isinstance(occ, dict):
            occ = occ.get("occ_symbol") or occ.get("symbol")
        if not occ:
            return {**out, "verdict": "unpriced", "allowed": True, "legs": legs,
                    "reason": f"FAIR VALUE: {strategy} on {ticker} has no {key} to judge — entry allowed unpriced"}
        q = _leg_quote(str(occ), contract, key, chain)
        iv, how = _leg_iv(q, spot, _num(contract.get("limit_price")) if key == "occ_symbol" else None)
        if iv is None:
            return {**out, "verdict": "unpriced", "allowed": True, "legs": legs,
                    "reason": f"FAIR VALUE: {occ} — {how}; entry allowed unpriced"}
        ratio = iv / fv
        bid, ask = _num(q.get("bid")) or 0.0, _num(q.get("ask")) or 0.0
        mid = (bid + ask) / 2.0 if bid > 0 and ask >= bid else _num(q.get("mid"))
        price = _num(contract.get("limit_price")) if key == "occ_symbol" else mid
        fair_px = _fair_price(q, spot, fv)
        leg = {"occ_symbol": occ, "kind": kind, "iv": round(iv, 4), "iv_source": how, "ratio": round(ratio, 3),
               "price": round(price, 2) if price else None, "fair_price": round(fair_px, 2) if fair_px else None}
        legs.append(leg)
        money = (f" — {'paying' if kind == BOUGHT else 'receiving'} ${price:.2f} vs fair ${fair_px:.2f}"
                 if price and fair_px else "")
        if kind == BOUGHT and ratio > 1.0 + cfg["max_buy_premium"]:
            refusals.append((BOUGHT, f"buying {occ} at IV {iv:.0%} is {ratio - 1:.0%} over the realized-vol fair "
                                     f"{fv:.0%} ({rv_words}; limit +{cfg['max_buy_premium']:.0%}){money}"))
        elif kind == SOLD and ratio < cfg["min_sell_ratio"]:
            refusals.append((SOLD, f"selling {occ} at IV {iv:.0%} is below the realized-vol fair {fv:.0%} "
                                   f"({rv_words}; floor {cfg['min_sell_ratio']:.0%} of fair){money}"))
        else:
            leg["ok"] = (f"{'buying' if kind == BOUGHT else 'selling'} {occ} at IV {iv:.0%} vs fair {fv:.0%} "
                         f"({ratio - 1:+.0%}){money}")
    out["legs"] = legs
    if refusals:
        verdict = "overpriced" if refusals[0][0] == BOUGHT else "underpriced"
        head = "refused" if cfg["mode"] == "enforce" else "flagged (not enforced)"
        return {**out, "verdict": verdict, "allowed": cfg["mode"] != "enforce",
                "reason": f"FAIR VALUE: {head} — " + "; ".join(r for _k, r in refusals)}
    return {**out, "verdict": "fair", "allowed": True,
            "reason": "FAIR VALUE OK: " + "; ".join(l["ok"] for l in legs)}


def _log(rec: dict) -> None:
    line = {"at": datetime.now().isoformat(), **rec}
    try:
        with open(os.path.join(DATA_DIR, LOG_FILE), "a") as f:
            f.write(json.dumps(line, default=str) + "\n")
    except OSError as e:
        logger.warning(f"fair-value log line not written: {e}")


def apply(contract: dict, strategy: str, ticker: str, spot: float, surface: dict = None,
          chain: list = None) -> dict:
    """select_contract's last step: attach the verdict to a constructed entry,
    refuse it in enforce mode, and log it. A contract that already carries an
    error, and every contract while the mode is "off", is returned untouched."""
    if not isinstance(contract, dict) or contract.get("error"):
        return contract
    cfg = settings()
    if cfg["mode"] == "off":
        return contract
    verdict = check(contract, strategy, ticker, spot, surface, chain, cfg)
    contract["fair_value"] = verdict
    if not verdict["allowed"]:
        contract["error"] = verdict["reason"]
    _log({k: verdict.get(k) for k in ("ticker", "strategy", "mode", "verdict", "allowed", "reason", "spot",
                                      "fair_vol", "rv20", "rv60", "rv_source", "legs")})
    (logger.warning if not verdict["allowed"] else logger.info)(verdict["reason"])
    return contract
