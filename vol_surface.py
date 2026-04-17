"""
VolTradeAI — Volatility Surface Model
======================================

Production-grade 2D implied volatility surface (strike × time-to-expiry)
built from live OPRA data via Alpaca. Computes true probabilities, skew
analytics, fair strategy values, and the Volatility Risk Premium.
"""

import os
import logging
import time
import re
import requests
from math import log, sqrt, exp, pi
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

# ─── Configuration ───────────────────────────────────────────────────────────

ALPACA_KEY = os.environ.get("ALPACA_KEY", "")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET", "")
DATA_URL = "https://data.alpaca.markets"
RISK_FREE_RATE = 0.045
CACHE_TTL = 3600  # 60 min — vol surface doesn't change fast intraday

_HEADERS = {
    "APCA-API-KEY-ID": ALPACA_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET,
}

# ─── Module-level cache ─────────────────────────────────────────────────────

_surface_cache: dict = {}  # {ticker: {"timestamp": float, "surface": dict}}
_sabr_cache: dict = {}  # {(ticker, expiry): {"timestamp": float, "params": dict}}
SABR_CACHE_TTL = 3600  # 60 min


# ─── Standard Normal CDF (Abramowitz & Stegun) ──────────────────────────────

def _norm_cdf(x: float) -> float:
    """Standard normal CDF — rational approximation (A&S 26.2.17)."""
    if x < -10.0:
        return 0.0
    if x > 10.0:
        return 1.0
    if x < 0:
        return 1.0 - _norm_cdf(-x)
    t = 1.0 / (1.0 + 0.2316419 * x)
    d = 1.0 / sqrt(2.0 * pi)
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    return 1.0 - d * exp(-x * x / 2.0) * poly


def _norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return exp(-x * x / 2.0) / sqrt(2.0 * pi)


# ─── Black-Scholes Pricing ──────────────────────────────────────────────────

def bs_price(S: float, K: float, T: float, r: float, sigma: float, opt_type: str = "call") -> float:
    """Black-Scholes European option price."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(0.0, (S - K) if opt_type == "call" else (K - S))
    d1 = (log(S / K) + (r + sigma ** 2 / 2.0) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if opt_type == "call":
        return S * _norm_cdf(d1) - K * exp(-r * T) * _norm_cdf(d2)
    else:
        return K * exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def bs_delta(S: float, K: float, T: float, r: float, sigma: float, opt_type: str = "call") -> float:
    """Black-Scholes delta."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        if opt_type == "call":
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
    d1 = (log(S / K) + (r + sigma ** 2 / 2.0) * T) / (sigma * sqrt(T))
    if opt_type == "call":
        return _norm_cdf(d1)
    else:
        return _norm_cdf(d1) - 1.0


def bs_prob_itm(S: float, K: float, T: float, r: float, sigma: float, opt_type: str = "call") -> float:
    """Risk-neutral probability of finishing in-the-money."""
    if T <= 0 or sigma <= 0:
        if opt_type == "call":
            return 1.0 if S > K else 0.0
        else:
            return 1.0 if S < K else 0.0
    d2 = (log(S / K) + (r - sigma ** 2 / 2.0) * T) / (sigma * sqrt(T))
    return _norm_cdf(d2) if opt_type == "call" else _norm_cdf(-d2)


def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes vega (sensitivity to IV)."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (log(S / K) + (r + sigma ** 2 / 2.0) * T) / (sigma * sqrt(T))
    return S * _norm_pdf(d1) * sqrt(T)


def implied_vol(price: float, S: float, K: float, T: float, r: float,
                opt_type: str = "call", tol: float = 1e-6, max_iter: int = 100) -> float:
    """Newton-Raphson implied volatility solver."""
    if T <= 0 or price <= 0:
        return 0.0

    # Intrinsic value check
    intrinsic = max(0.0, (S - K) if opt_type == "call" else (K - S))
    if price < intrinsic:
        return 0.0

    # Initial guess using Brenner-Subrahmanyam approximation
    sigma = sqrt(2.0 * pi / T) * price / S
    sigma = max(0.01, min(sigma, 5.0))

    for _ in range(max_iter):
        bs_val = bs_price(S, K, T, r, sigma, opt_type)
        vega = bs_vega(S, K, T, r, sigma)
        if vega < 1e-12:
            break
        diff = bs_val - price
        sigma -= diff / vega
        sigma = max(0.001, min(sigma, 10.0))
        if abs(diff) < tol:
            break

    return sigma if 0.001 < sigma < 10.0 else 0.0


# ─── OCC Symbol Parser ──────────────────────────────────────────────────────

def parse_occ_symbol(symbol: str) -> dict:
    """
    Parse an OCC option symbol.
    Format: TICKER + YYMMDD + C/P + 8-digit strike (strike * 1000)
    Example: SPY260416C00510000 → SPY, 2026-04-16, call, 510.0
    """
    match = re.match(r'^([A-Z]+)(\d{6})([CP])(\d{8})$', symbol)
    if not match:
        return {}
    ticker = match.group(1)
    date_str = match.group(2)
    opt_type = "call" if match.group(3) == "C" else "put"
    strike = int(match.group(4)) / 1000.0
    expiry = datetime.strptime(date_str, "%y%m%d").strftime("%Y-%m-%d")
    return {
        "ticker": ticker,
        "expiry": expiry,
        "opt_type": opt_type,
        "strike": strike,
        "symbol": symbol,
    }


# ─── Alpaca Data Fetchers ───────────────────────────────────────────────────

# High-volume index ETFs that have huge chains — narrow DTE to avoid timeouts
_HEAVY_CHAIN_TICKERS = {"SPY", "QQQ", "IWM", "DIA", "SPX", "NDX", "VIX"}


def _fetch_options_chain(ticker: str, dte_min: int = 7, dte_max: int = 90) -> dict:
    """
    Fetch the full options chain from Alpaca OPRA.
    Returns raw snapshots dict keyed by OCC symbol.
    Handles pagination automatically.

    For heavy-chain tickers (SPY, QQQ, IWM etc.) caps DTE to 30 days to
    avoid 4000+ contract scans that can blow the per-scan time budget.
    """
    # Narrow DTE for index ETFs with massive chains — we trade weekly/monthly
    # premium not far-dated, so capping at 30 DTE costs us nothing useful but
    # cuts contract count by ~70% on SPY/QQQ.
    if ticker.upper() in _HEAVY_CHAIN_TICKERS and dte_max > 30:
        dte_max = 30

    today = datetime.now(timezone.utc).date()
    gte = (today + timedelta(days=dte_min)).isoformat()
    lte = (today + timedelta(days=dte_max)).isoformat()

    all_snapshots = {}
    page_token = None
    # Pagination cap: 10 pages × 200/page = 2000 contracts max. Previously
    # 30 pages allowed 6000 contracts which on a slow connection could take
    # 30 × 8s = 240s wall-clock — well past the 50s scan cap. Hard-cap to 10
    # pages × 4s timeout = 40s worst case (still within scan budget).
    max_pages = 10

    for page_num in range(max_pages):
        params = {
            "feed": "opra",
            "limit": 200,
            "expiration_date_gte": gte,
            "expiration_date_lte": lte,
        }
        if page_token:
            params["page_token"] = page_token

        try:
            resp = requests.get(
                f"{DATA_URL}/v1beta1/options/snapshots/{ticker}",
                params=params,
                headers=_HEADERS,
                timeout=4,  # Reduced from 8s — 10 pages × 4s = 40s worst case
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.warning("Options chain fetch slow/failed (page %d): %s", page_num, e)
            break  # Return partial chain — still useful for vol surface

        data = resp.json()
        snapshots = data.get("snapshots", {})
        all_snapshots.update(snapshots)
        page_token = data.get("next_page_token")

        logger.debug("Page %d: fetched %d snapshots (total: %d)", page_num + 1, len(snapshots), len(all_snapshots))

        if not page_token:
            break

    logger.info("Fetched %d option contracts for %s (%d-%d DTE)", len(all_snapshots), ticker, dte_min, dte_max)
    return all_snapshots


def _fetch_spot_price(ticker: str) -> float:
    """Get the latest spot price for a ticker."""
    try:
        resp = requests.get(
            f"{DATA_URL}/v2/stocks/trades/latest",
            params={"symbols": ticker, "feed": "sip"},
            headers=_HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        trades = data.get("trades", {})
        if ticker in trades:
            return float(trades[ticker]["p"])
    except requests.RequestException as e:
        logger.warning("Latest trade fetch failed for %s: %s", ticker, e)

    # Fallback: last daily bar
    try:
        today = datetime.now(timezone.utc).date()
        start = (today - timedelta(days=7)).isoformat()
        resp = requests.get(
            f"{DATA_URL}/v2/stocks/bars",
            params={
                "symbols": ticker,
                "timeframe": "1Day",
                "start": start,
                "limit": 5,
                "adjustment": "all",
                "feed": "sip",
            },
            headers=_HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        bars = data.get("bars", {}).get(ticker, [])
        if bars:
            return float(bars[-1]["c"])
    except requests.RequestException as e:
        logger.error("Spot price fallback failed for %s: %s", ticker, e)

    return 0.0


def _fetch_historical_bars(ticker: str, days: int = 90) -> list:
    """Fetch daily bars for realized volatility calculation."""
    today = datetime.now(timezone.utc).date()
    start = (today - timedelta(days=int(days * 1.6))).isoformat()  # Buffer for weekends/holidays

    all_bars = []
    page_token = None

    for _ in range(5):  # Max pages
        params = {
            "symbols": ticker,
            "timeframe": "1Day",
            "start": start,
            "limit": 200,
            "adjustment": "all",
            "feed": "sip",
        }
        if page_token:
            params["page_token"] = page_token

        try:
            resp = requests.get(
                f"{DATA_URL}/v2/stocks/bars",
                params=params,
                headers=_HEADERS,
                timeout=8,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.warning("Historical bars fetch slow/failed: %s", e)
            break

        data = resp.json()
        bars = data.get("bars", {}).get(ticker, [])
        all_bars.extend(bars)
        page_token = data.get("next_page_token")
        if not page_token:
            break

    logger.info("Fetched %d daily bars for %s", len(all_bars), ticker)
    return all_bars


def _compute_realized_vol(bars: list, window: int) -> float:
    """Compute annualized realized volatility from daily bars."""
    if len(bars) < window + 1:
        logger.warning("Not enough bars (%d) for %d-day realized vol", len(bars), window)
        return 0.0

    # Use the most recent 'window' returns
    closes = [float(b["c"]) for b in bars]
    recent = closes[-(window + 1):]
    log_returns = [log(recent[i] / recent[i - 1]) for i in range(1, len(recent)) if recent[i - 1] > 0]

    if len(log_returns) < 2:
        return 0.0

    mean_ret = sum(log_returns) / len(log_returns)
    variance = sum((r - mean_ret) ** 2 for r in log_returns) / (len(log_returns) - 1)
    return sqrt(variance * 252)  # Annualize


# ─── SABR Implied Volatility (Hagan et al. 2002) ────────────────────────────

def sabr_implied_vol(F: float, K: float, T: float, alpha: float, beta: float,
                     rho: float, nu: float) -> float:
    """
    Hagan et al. (2002) SABR implied volatility approximation.

    Args:
        F: Forward price
        K: Strike price
        T: Time to expiry (years)
        alpha: Initial volatility level (ATM vol parameter)
        beta: CEV exponent (use 1.0 for lognormal, 0.0 for normal)
        rho: Correlation between asset and vol processes (-1 to 1)
             rho < 0 means puts are overpriced (negative skew — normal for equities)
        nu: Vol-of-vol (how much the volatility itself moves)

    Returns:
        Implied volatility (annualized), or 0.0 on failure.
    """
    if F <= 0 or K <= 0 or T <= 0 or alpha <= 0:
        return 0.0
    if nu < 0:
        return 0.0
    rho = max(-0.999, min(0.999, rho))
    if T < 1e-10:
        if abs(beta - 1.0) < 1e-10:
            return alpha / F if F > 0 else 0.0
        return alpha / (F ** (1.0 - beta)) if F > 0 else 0.0
    if nu < 1e-10:
        return _sabr_vol_zero_nu(F, K, T, alpha, beta)
    try:
        return _sabr_hagan_formula(F, K, T, alpha, beta, rho, nu)
    except (ValueError, ZeroDivisionError, OverflowError):
        return 0.0


def _sabr_vol_zero_nu(F: float, K: float, T: float, alpha: float,
                      beta: float) -> float:
    """SABR with nu=0 (no vol-of-vol). Reduces to a CEV-like model."""
    omb = 1.0 - beta
    if abs(F - K) < 1e-10 * F:
        if abs(omb) < 1e-10:
            return alpha
        return alpha / (F ** omb)
    FK = F * K
    if FK <= 0:
        return 0.0
    FK_beta = FK ** (omb / 2.0)
    logFK = log(F / K)
    denom = FK_beta
    if abs(omb) > 1e-10:
        denom *= (1.0 + omb ** 2 / 24.0 * logFK ** 2
                  + omb ** 4 / 1920.0 * logFK ** 4)
    if abs(denom) < 1e-30:
        return 0.0
    if abs(omb) < 1e-10:
        correction = 1.0
    else:
        FK_1mb = FK ** omb
        correction = 1.0 + (omb ** 2 / 24.0 * alpha ** 2 / FK_1mb) * T
    if abs(logFK) > 1e-30:
        return alpha * logFK / denom * correction
    return alpha / (F ** omb)


def _sabr_hagan_formula(F: float, K: float, T: float, alpha: float,
                        beta: float, rho: float, nu: float) -> float:
    """Full Hagan et al. 2002 SABR formula."""
    omb = 1.0 - beta
    FK = F * K
    if FK <= 0:
        return 0.0
    logFK = log(F / K)
    FK_half_omb = FK ** (omb / 2.0)

    # ATM case
    if abs(logFK) < 1e-7:
        FM = F
        FM_omb = FM ** omb
        FM_half_omb = FM ** (omb / 2.0)
        if abs(FM_omb) < 1e-30:
            return 0.0
        t1 = omb ** 2 / 24.0 * alpha ** 2 / FM_omb
        t2 = 0.25 * rho * beta * nu * alpha / FM_half_omb
        t3 = (2.0 - 3.0 * rho ** 2) / 24.0 * nu ** 2
        sigma_atm = (alpha / FM_half_omb) * (1.0 + (t1 + t2 + t3) * T)
        return max(0.0, sigma_atm)

    # General case
    logFK2 = logFK ** 2
    logFK4 = logFK2 ** 2
    denom_log = (1.0 + omb ** 2 / 24.0 * logFK2 + omb ** 4 / 1920.0 * logFK4)
    denominator = FK_half_omb * denom_log
    if abs(denominator) < 1e-30:
        return 0.0
    z = (nu / alpha) * FK_half_omb * logFK
    disc = 1.0 - 2.0 * rho * z + z * z
    if disc < 0:
        disc = 0.0
    sqrt_disc = sqrt(disc)
    num_xz = sqrt_disc + z - rho
    den_xz = 1.0 - rho
    if abs(den_xz) < 1e-30 or num_xz <= 0:
        return _sabr_hagan_formula(F, F, T, alpha, beta, rho, nu)
    x_z = log(num_xz / den_xz)
    factor1 = z / x_z if abs(x_z) > 1e-30 else 1.0
    FK_1mb = FK ** omb
    if abs(FK_1mb) < 1e-30:
        return 0.0
    t1 = omb ** 2 / 24.0 * alpha ** 2 / FK_1mb
    t2 = 0.25 * rho * beta * nu * alpha / FK_half_omb
    t3 = (2.0 - 3.0 * rho ** 2) / 24.0 * nu ** 2
    factor2 = 1.0 + (t1 + t2 + t3) * T
    sigma = (alpha / denominator) * factor1 * factor2
    return max(0.0, sigma)


# ─── Nelder-Mead Simplex Optimization ───────────────────────────────────────

def _nelder_mead(func, x0: list, max_iter: int = 500, tol: float = 1e-8,
                 bounds: list = None) -> tuple:
    """
    Nelder-Mead simplex optimization (no scipy dependency).

    Args:
        func: Objective function f(x) -> float to minimize
        x0: Initial guess (list of floats)
        max_iter: Maximum iterations
        tol: Convergence tolerance on function value spread
        bounds: Optional [(lo, hi), ...] for each parameter

    Returns:
        (best_x, best_value)
    """
    n = len(x0)
    alpha_nm = 1.0
    gamma_nm = 2.0
    rho_nm = 0.5
    sigma_nm = 0.5

    def clamp(x):
        if bounds is None:
            return x
        return [max(lo, min(hi, xi)) for xi, (lo, hi) in zip(x, bounds)]

    simplex = [list(x0)]
    for i in range(n):
        v = list(x0)
        v[i] += max(0.05 * abs(x0[i]), 0.001)
        simplex.append(clamp(v))
    f_vals = [func(v) for v in simplex]

    for _ in range(max_iter):
        order = sorted(range(n + 1), key=lambda i: f_vals[i])
        simplex = [simplex[i] for i in order]
        f_vals = [f_vals[i] for i in order]
        if abs(f_vals[-1] - f_vals[0]) < tol:
            break
        centroid = [sum(simplex[i][j] for i in range(n)) / n for j in range(n)]
        worst = simplex[-1]
        ref = clamp([centroid[j] + alpha_nm * (centroid[j] - worst[j]) for j in range(n)])
        f_ref = func(ref)
        if f_vals[0] <= f_ref < f_vals[-2]:
            simplex[-1] = ref
            f_vals[-1] = f_ref
            continue
        if f_ref < f_vals[0]:
            exp_pt = clamp([centroid[j] + gamma_nm * (ref[j] - centroid[j]) for j in range(n)])
            f_exp = func(exp_pt)
            if f_exp < f_ref:
                simplex[-1] = exp_pt
                f_vals[-1] = f_exp
            else:
                simplex[-1] = ref
                f_vals[-1] = f_ref
            continue
        if f_ref < f_vals[-1]:
            con = clamp([centroid[j] + rho_nm * (ref[j] - centroid[j]) for j in range(n)])
            f_con = func(con)
            if f_con <= f_ref:
                simplex[-1] = con
                f_vals[-1] = f_con
                continue
        else:
            con = clamp([centroid[j] + rho_nm * (worst[j] - centroid[j]) for j in range(n)])
            f_con = func(con)
            if f_con < f_vals[-1]:
                simplex[-1] = con
                f_vals[-1] = f_con
                continue
        best = simplex[0]
        for i in range(1, n + 1):
            simplex[i] = clamp([best[j] + sigma_nm * (simplex[i][j] - best[j]) for j in range(n)])
            f_vals[i] = func(simplex[i])

    best_idx = min(range(n + 1), key=lambda i: f_vals[i])
    return simplex[best_idx], f_vals[best_idx]


# ─── SABR Calibration ───────────────────────────────────────────────────────

def calibrate_sabr(F: float, strikes: list, market_vols: list, T: float,
                   beta: float = 1.0) -> dict:
    """
    Calibrate SABR parameters (alpha, rho, nu) to fit market implied vols.

    Args:
        F: Forward price
        strikes: list of strike prices
        market_vols: list of market implied vols at those strikes
        T: Time to expiry (years)
        beta: Fixed (typically 1.0 for equities)

    Returns:
        {"alpha": float, "rho": float, "nu": float, "beta": float, "fit_error": float}
    """
    if len(strikes) < 3 or len(strikes) != len(market_vols):
        return {"alpha": 0.0, "rho": 0.0, "nu": 0.0, "beta": beta, "fit_error": 1.0}
    if T <= 0 or F <= 0:
        return {"alpha": 0.0, "rho": 0.0, "nu": 0.0, "beta": beta, "fit_error": 1.0}

    valid = [(k, v) for k, v in zip(strikes, market_vols) if v > 0.001 and k > 0]
    if len(valid) < 3:
        return {"alpha": 0.0, "rho": 0.0, "nu": 0.0, "beta": beta, "fit_error": 1.0}
    vs = [v[0] for v in valid]
    vv = [v[1] for v in valid]

    atm_idx = min(range(len(vs)), key=lambda i: abs(vs[i] - F))
    atm_vol = vv[atm_idx]
    alpha0 = atm_vol if abs(beta - 1.0) < 1e-10 else atm_vol * F ** (1.0 - beta)
    bnd = [(0.001, 5.0), (-0.999, 0.999), (0.001, 5.0)]

    def objective(params):
        a, r, n = params
        if a <= 0 or n <= 0 or abs(r) >= 1.0:
            return 1e10
        sse = 0.0
        for k, mv in zip(vs, vv):
            mv_model = sabr_implied_vol(F, k, T, a, beta, r, n)
            if mv_model <= 0:
                sse += 1.0
            else:
                sse += (mv_model - mv) ** 2
        return sse

    best_x, best_val = _nelder_mead(objective, [alpha0, -0.3, 0.3],
                                     max_iter=500, tol=1e-10, bounds=bnd)
    for alt_rho in [-0.5, -0.1, 0.1]:
        for alt_nu in [0.15, 0.5, 1.0]:
            ax, av = _nelder_mead(objective, [alpha0, alt_rho, alt_nu],
                                  max_iter=300, tol=1e-10, bounds=bnd)
            if av < best_val:
                best_x, best_val = ax, av

    n_pts = len(vs)
    rmse = sqrt(best_val / n_pts) if n_pts > 0 else 1.0
    return {
        "alpha": round(best_x[0], 8),
        "rho": round(best_x[1], 6),
        "nu": round(best_x[2], 6),
        "beta": beta,
        "fit_error": round(rmse, 8),
    }


# ─── SABR Probability ───────────────────────────────────────────────────────

def sabr_probability_otm(F: float, K: float, T: float, alpha: float,
                         beta: float, rho: float, nu: float,
                         opt_type: str = "call") -> float:
    """
    True probability of finishing OTM using SABR-calibrated vol.

    Uses P(ITM) = N(d2) where d2 uses SABR vol at that specific strike.
    SABR gives a DIFFERENT vol for each strike, so probability changes
    non-linearly across strikes — unlike Black-Scholes with one vol.

    Returns probability of finishing OTM (0 to 1).
    """
    if T <= 0 or alpha <= 0 or F <= 0 or K <= 0:
        if opt_type == "call":
            return 0.0 if F > K else 1.0
        else:
            return 0.0 if F < K else 1.0
    sigma = sabr_implied_vol(F, K, T, alpha, beta, rho, nu)
    if sigma <= 0:
        if opt_type == "call":
            return 0.0 if F > K else 1.0
        else:
            return 0.0 if F < K else 1.0
    d2 = (log(F / K) + (-0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    if opt_type == "call":
        return 1.0 - _norm_cdf(d2)
    else:
        return 1.0 - _norm_cdf(-d2)


# ─── SABR Strategy Pricer ───────────────────────────────────────────────────

def sabr_price_strategy(F: float, legs: list, T: float, alpha: float,
                        beta: float, rho: float, nu: float,
                        r: float = 0.045, spot: float = 0.0,
                        surface: dict = None) -> dict:
    """
    Price a multi-leg strategy using SABR-calibrated vols.

    Each leg gets its OWN implied vol from the SABR smile, meaning:
    - OTM puts get higher vol (overpriced relative to ATM)
    - OTM calls get lower vol (underpriced relative to ATM)

    Args:
        F: Forward price
        legs: [{"strike": K, "opt_type": "call"/"put", "side": "buy"/"sell"}]
        T: Time to expiry
        alpha, beta, rho, nu: Calibrated SABR params
        r: Risk-free rate
        spot: Spot price (defaults to F if not provided)
        surface: Optional vol surface dict for market prices

    Returns:
        dict with fair_value, market_price, edge, probability_of_profit,
        expected_value, skew_impact
    """
    if spot <= 0:
        spot = F
    atm_vol = sabr_implied_vol(F, F, T, alpha, beta, rho, nu)
    if atm_vol <= 0:
        atm_vol = alpha

    total_fair = 0.0
    total_flat = 0.0
    total_market = 0.0

    for leg in legs:
        strike = leg["strike"]
        opt_type = leg["opt_type"]
        side = leg["side"]
        sign = 1.0 if side == "buy" else -1.0
        sv = sabr_implied_vol(F, strike, T, alpha, beta, rho, nu)
        if sv <= 0:
            sv = atm_vol
        fair = bs_price(spot, strike, T, r, sv, opt_type)
        flat = bs_price(spot, strike, T, r, atm_vol, opt_type)
        mkt = 0.0
        if surface:
            expiry = leg.get("expiry", "")
            ed = surface.get("expirations", {}).get(expiry, {})
            sd = ed.get("strikes", {}).get(strike, {})
            mkt = sd.get(f"{opt_type}_mid", 0)
        if mkt <= 0:
            mkt = fair
        total_fair += sign * fair
        total_flat += sign * flat
        total_market += sign * mkt

    edge = total_fair - total_market
    skew_impact = total_fair - total_flat
    pop, ev = _sabr_strategy_pop(legs, spot, F, T, alpha, beta, rho, nu, r, total_market)
    return {
        "fair_value": round(total_fair, 4),
        "market_price": round(total_market, 4),
        "edge": round(edge, 4),
        "probability_of_profit": round(pop, 4),
        "expected_value": round(ev, 4),
        "skew_impact": round(skew_impact, 4),
    }


def _sabr_strategy_pop(legs: list, spot: float, F: float, T: float,
                       alpha: float, beta: float, rho: float, nu: float,
                       r: float, net_premium: float) -> tuple:
    """POP and EV for multi-leg strategy using SABR-calibrated distribution."""
    if T <= 0 or alpha <= 0:
        return 0.5, 0.0
    atm_vol = sabr_implied_vol(F, F, T, alpha, beta, rho, nu)
    if atm_vol <= 0:
        atm_vol = alpha
    low = spot * 0.70
    high = spot * 1.30
    n_pts = 500
    step = (high - low) / n_pts
    drift = (r - 0.5 * atm_vol ** 2) * T
    vst = atm_vol * sqrt(T)
    if vst < 1e-10:
        return 0.5, 0.0
    w_pnl = 0.0
    prof_prob = 0.0
    tot_prob = 0.0
    for i in range(n_pts + 1):
        term = low + i * step
        if term <= 0:
            continue
        pnl = -net_premium
        for leg in legs:
            strike = leg["strike"]
            opt_type = leg["opt_type"]
            sign = 1.0 if leg["side"] == "buy" else -1.0
            if opt_type == "call":
                intr = max(0.0, term - strike)
            else:
                intr = max(0.0, strike - term)
            pnl += sign * intr
        lr = log(term / spot)
        z_val = (lr - drift) / vst
        pdf = exp(-0.5 * z_val * z_val) / (term * vst)
        w_pnl += pnl * pdf
        tot_prob += pdf
        if pnl > 0:
            prof_prob += pdf
    if tot_prob > 0:
        return prof_prob / tot_prob, w_pnl / tot_prob
    return 0.5, 0.0


# ─── SABR Calibration Helper for Surface Builder ────────────────────────────

def _calibrate_expiry_sabr(ticker: str, expiry: str, strikes_data: dict,
                            spot: float, dte: int,
                            beta: float = 1.0) -> dict:
    """
    Calibrate SABR params for a single expiry from the surface strikes data.
    Uses caching to avoid recalibration within TTL.
    """
    cache_key = (ticker, expiry)
    cached = _sabr_cache.get(cache_key)
    if cached and (time.time() - cached["timestamp"]) < SABR_CACHE_TTL:
        return cached["params"]

    T = dte / 365.0
    if T <= 0:
        return {}

    F = spot * exp(RISK_FREE_RATE * T)

    # Collect strikes and their average IVs
    cal_strikes = []
    cal_vols = []
    for strike, sd in sorted(strikes_data.items()):
        call_iv = sd.get("call_iv", 0)
        put_iv = sd.get("put_iv", 0)
        if call_iv > 0 and put_iv > 0:
            iv = (call_iv + put_iv) / 2.0
        elif call_iv > 0:
            iv = call_iv
        elif put_iv > 0:
            iv = put_iv
        else:
            continue
        # Only use strikes within reasonable range (50%-150% of spot)
        if 0.5 * spot <= strike <= 1.5 * spot:
            cal_strikes.append(strike)
            cal_vols.append(iv)

    if len(cal_strikes) < 5:
        return {}

    params = calibrate_sabr(F, cal_strikes, cal_vols, T, beta=beta)
    if params.get("alpha", 0) > 0 and params.get("fit_error", 1.0) < 0.10:
        _sabr_cache[cache_key] = {"timestamp": time.time(), "params": params}
        return params
    return {}


# ─── Surface Builder ─────────────────────────────────────────────────────────

# Per-ticker time budget. If build_surface takes longer than this, we return
# what we have so the caller (scan_market) can move on. Thread-safe (no
# signal-based timeout) so it works from ThreadPoolExecutor workers.
BUILD_SURFACE_TIME_BUDGET_S = 12.0


def build_surface(ticker: str, time_budget_s: float = BUILD_SURFACE_TIME_BUDGET_S) -> dict:
    """
    Build a 2D implied volatility surface from live OPRA data.

    Returns a structured dict with expirations, strikes, IVs, greeks,
    ATM vol, realized vol, and the volatility risk premium.

    Has a soft time budget (default 12s). If exceeded mid-build, returns
    a partial result with whatever expiries were processed before the
    budget ran out. This prevents one slow ticker from blowing the
    enclosing scan_market 50s hard cap.
    """
    _t_start = time.time()

    def _budget_left() -> float:
        return time_budget_s - (time.time() - _t_start)

    # Check cache
    cached = _surface_cache.get(ticker)
    if cached and (time.time() - cached["timestamp"]) < CACHE_TTL:
        logger.info("Returning cached surface for %s (%.0fs old)", ticker, time.time() - cached["timestamp"])
        return cached["surface"]

    logger.info("Building volatility surface for %s (budget=%.1fs)", ticker, time_budget_s)

    spot = _fetch_spot_price(ticker)
    if spot <= 0:
        logger.error("Could not determine spot price for %s", ticker)
        return {"ticker": ticker, "error": "spot_price_unavailable"}

    if _budget_left() <= 0:
        logger.warning("build_surface budget exceeded for %s after spot fetch", ticker)
        return {"ticker": ticker, "spot_price": spot, "error": "time_budget_exceeded"}

    raw_chain = _fetch_options_chain(ticker)
    if not raw_chain:
        logger.error("No options data for %s", ticker)
        return {"ticker": ticker, "spot_price": spot, "error": "no_options_data"}

    if _budget_left() <= 0:
        logger.warning("build_surface budget exceeded for %s after chain fetch", ticker)
        return {"ticker": ticker, "spot_price": spot, "error": "time_budget_exceeded"}

    today = datetime.now(timezone.utc).date()
    expirations: dict = {}

    for occ_symbol, snapshot in raw_chain.items():
        parsed = parse_occ_symbol(occ_symbol)
        if not parsed:
            continue

        expiry = parsed["expiry"]
        strike = parsed["strike"]
        opt_type = parsed["opt_type"]
        expiry_date = datetime.strptime(expiry, "%Y-%m-%d").date()
        dte = (expiry_date - today).days
        if dte < 1:
            continue

        # Extract quote data
        quote = snapshot.get("latestQuote", {})
        bid = float(quote.get("bp", 0) or 0)
        ask = float(quote.get("ap", 0) or 0)
        if bid <= 0 and ask <= 0:
            continue  # No valid quotes

        mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else max(bid, ask)

        # Extract greeks and IV from Alpaca if available
        api_iv = float(snapshot.get("impliedVolatility", 0) or 0)
        greeks = snapshot.get("greeks", {})
        api_delta = float(greeks.get("delta", 0) or 0)

        # If Alpaca didn't provide IV, compute it from mid price
        T = dte / 365.0
        if api_iv <= 0 and mid > 0:
            api_iv = implied_vol(mid, spot, strike, T, RISK_FREE_RATE, opt_type)

        # Skip if IV is still 0 (no usable data)
        if api_iv <= 0:
            continue

        # Compute delta if not provided
        if api_delta == 0:
            api_delta = bs_delta(spot, strike, T, RISK_FREE_RATE, api_iv, opt_type)

        # Build expiration bucket
        if expiry not in expirations:
            expirations[expiry] = {"dte": dte, "strikes": {}}

        if strike not in expirations[expiry]["strikes"]:
            expirations[expiry]["strikes"][strike] = {}

        strike_data = expirations[expiry]["strikes"][strike]
        prefix = "call" if opt_type == "call" else "put"
        strike_data[f"{prefix}_iv"] = round(api_iv, 6)
        strike_data[f"{prefix}_delta"] = round(api_delta, 6)
        strike_data[f"{prefix}_bid"] = bid
        strike_data[f"{prefix}_ask"] = ask
        strike_data[f"{prefix}_mid"] = round(mid, 4)

    # Sort expirations by date
    expirations = dict(sorted(expirations.items()))

    # Compute ATM IV (weighted average of nearest-to-50-delta call and put IVs)
    atm_iv = _compute_atm_iv(expirations, spot)

    # ── SABR calibration per expiry ──
    # Skip the whole SABR loop if we're already over budget. SABR is a
    # nice-to-have for skew analysis but not required for VRP / scoring.
    sabr_params = {}
    if _budget_left() <= 1.0:
        logger.info("build_surface skipping SABR for %s (budget %.1fs left)", ticker, _budget_left())
    else:
        for expiry, exp_data in expirations.items():
            if _budget_left() <= 0.5:
                logger.info("build_surface SABR loop hit budget for %s after %d expiries", ticker, len(sabr_params))
                break
            dte_val = exp_data.get("dte", 0)
            strikes_data = exp_data.get("strikes", {})
            if dte_val > 0 and len(strikes_data) >= 5:
                try:
                    sp = _calibrate_expiry_sabr(ticker, expiry, strikes_data, spot, dte_val)
                    if sp and sp.get("alpha", 0) > 0:
                        sabr_params[expiry] = sp
                        # Store SABR params in the expiry dict too for easy access
                        exp_data["sabr"] = sp
                except Exception as e:
                    # Demoted from warning — calibration failures are common on
                    # noisy short-DTE chains and don't affect downstream VRP/score.
                    logger.debug("SABR calibration failed for %s %s: %s", ticker, expiry, e)

    # Compute realized vol
    bars = _fetch_historical_bars(ticker, days=90)
    rv_20 = _compute_realized_vol(bars, 20)
    rv_60 = _compute_realized_vol(bars, 60)
    vrp = atm_iv - rv_20 if (atm_iv > 0 and rv_20 > 0) else 0.0

    surface = {
        "ticker": ticker,
        "spot_price": spot,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "expirations": expirations,
        "sabr_params": sabr_params,
        "atm_iv": round(atm_iv, 6),
        "realized_vol_20d": round(rv_20, 6),
        "realized_vol_60d": round(rv_60, 6),
        "vrp": round(vrp, 6),
    }

    # Cache it
    _surface_cache[ticker] = {"timestamp": time.time(), "surface": surface}
    logger.info("Surface built for %s: %d expirations, ATM IV=%.2f%%, RV20=%.2f%%, VRP=%.2f%%",
                ticker, len(expirations), atm_iv * 100, rv_20 * 100, vrp * 100)
    return surface


def _compute_atm_iv(expirations: dict, spot: float) -> float:
    """
    Compute ATM implied vol by finding the nearest-to-ATM strike
    across the shortest few expirations and averaging.
    """
    atm_ivs = []

    for expiry, exp_data in expirations.items():
        strikes = exp_data["strikes"]
        if not strikes:
            continue

        # Find strike closest to spot
        closest_strike = min(strikes.keys(), key=lambda k: abs(k - spot))
        sd = strikes[closest_strike]

        call_iv = sd.get("call_iv", 0)
        put_iv = sd.get("put_iv", 0)

        if call_iv > 0 and put_iv > 0:
            atm_ivs.append((call_iv + put_iv) / 2.0)
        elif call_iv > 0:
            atm_ivs.append(call_iv)
        elif put_iv > 0:
            atm_ivs.append(put_iv)

        # Use first 3 expirations for a stable ATM IV estimate
        if len(atm_ivs) >= 3:
            break

    return sum(atm_ivs) / len(atm_ivs) if atm_ivs else 0.0


# ─── Skew Analysis ───────────────────────────────────────────────────────────

def analyze_skew(surface: dict) -> dict:
    """
    Compute volatility skew metrics from the surface.

    Returns put-call skew, term structure slope, smile curvature,
    and the estimated dollar edge from skew mispricing.
    """
    if "error" in surface:
        return {"error": surface["error"]}

    spot = surface["spot_price"]
    expirations = surface.get("expirations", {})
    if not expirations:
        return {"error": "no_expirations"}

    # ── Per-expiration skew data ──
    skew_by_expiry = []

    for expiry, exp_data in expirations.items():
        dte = exp_data["dte"]
        strikes = exp_data["strikes"]
        if not strikes:
            continue

        # Find 25-delta put and 25-delta call strikes
        put_25_strike, put_25_iv = _find_delta_strike(strikes, spot, target_delta=-0.25, opt_type="put")
        call_25_strike, call_25_iv = _find_delta_strike(strikes, spot, target_delta=0.25, opt_type="call")

        # ATM IV for this expiry
        closest_atm = min(strikes.keys(), key=lambda k: abs(k - spot))
        sd = strikes[closest_atm]
        atm_iv = ((sd.get("call_iv", 0) or 0) + (sd.get("put_iv", 0) or 0))
        atm_iv = atm_iv / 2.0 if (sd.get("call_iv", 0) > 0 and sd.get("put_iv", 0) > 0) else max(sd.get("call_iv", 0), sd.get("put_iv", 0))

        if put_25_iv > 0 and call_25_iv > 0 and atm_iv > 0:
            skew_by_expiry.append({
                "expiry": expiry,
                "dte": dte,
                "put_25_iv": put_25_iv,
                "call_25_iv": call_25_iv,
                "atm_iv": atm_iv,
                "put_call_skew": put_25_iv - call_25_iv,
                "smile_curv": ((put_25_iv + call_25_iv) / 2.0) - atm_iv,
            })

    if not skew_by_expiry:
        return {
            "put_call_skew": 0.0,
            "skew_percentile": 50.0,
            "term_structure_slope": 0.0,
            "smile_curvature": 0.0,
            "overpriced_side": "neutral",
            "skew_edge": 0.0,
        }

    # ── Aggregate metrics ──

    # Primary skew: use the nearest liquid expiry
    primary = skew_by_expiry[0]
    put_call_skew = primary["put_call_skew"]

    # Term structure slope: far IV - near IV (positive = contango)
    if len(skew_by_expiry) >= 2:
        near_iv = skew_by_expiry[0]["atm_iv"]
        far_iv = skew_by_expiry[-1]["atm_iv"]
        term_slope = far_iv - near_iv
    else:
        term_slope = 0.0

    # Smile curvature: average wing excess over ATM
    smile_curv = sum(s["smile_curv"] for s in skew_by_expiry) / len(skew_by_expiry)

    # Skew percentile heuristic:
    # Typical equity skew ranges from 0.01 to 0.10 (1% to 10% IV points).
    # Map current skew into a 0-100 range using empirical bounds.
    skew_abs = abs(put_call_skew)
    skew_pct = min(100.0, max(0.0, (skew_abs - 0.01) / (0.10 - 0.01) * 100.0))

    # Overpriced side
    if put_call_skew > 0.005:
        overpriced = "puts"
    elif put_call_skew < -0.005:
        overpriced = "calls"
    else:
        overpriced = "neutral"

    # Estimated skew edge in dollar terms:
    # The mispricing from skew is roughly half the skew * vega * 100 (per contract)
    atm_iv = surface.get("atm_iv", 0.25)
    dte_near = primary["dte"]
    T = dte_near / 365.0
    vega_atm = bs_vega(spot, spot, T, RISK_FREE_RATE, atm_iv) if T > 0 else 0.0
    skew_edge = abs(put_call_skew) / 2.0 * vega_atm * 100  # Per contract (100 shares)

    # SABR rho from calibrated params (if available)
    sabr_params = surface.get("sabr_params", {})
    sabr_rho = None
    sabr_nu = None
    if sabr_params:
        for _exp, sp in sorted(sabr_params.items()):
            if sp.get("alpha", 0) > 0:
                sabr_rho = sp.get("rho")
                sabr_nu = sp.get("nu")
                break

    return {
        "put_call_skew": round(put_call_skew, 6),
        "skew_percentile": round(skew_pct, 2),
        "term_structure_slope": round(term_slope, 6),
        "smile_curvature": round(smile_curv, 6),
        "overpriced_side": overpriced,
        "skew_edge": round(skew_edge, 2),
        "sabr_rho": round(sabr_rho, 4) if sabr_rho is not None else None,
        "sabr_nu": round(sabr_nu, 4) if sabr_nu is not None else None,
        "detail_by_expiry": skew_by_expiry,
    }


def _find_delta_strike(strikes: dict, spot: float, target_delta: float, opt_type: str) -> tuple:
    """
    Find the strike closest to a target delta.
    Falls back to moneyness proxy if delta data is unavailable.

    Returns (strike, iv).
    """
    delta_key = f"{opt_type}_delta"
    iv_key = f"{opt_type}_iv"

    # Try delta-based matching first
    best_strike = None
    best_iv = 0.0
    best_diff = float("inf")

    for strike, sd in strikes.items():
        delta = sd.get(delta_key, 0)
        iv = sd.get(iv_key, 0)
        if delta == 0 or iv <= 0:
            continue
        diff = abs(delta - target_delta)
        if diff < best_diff:
            best_diff = diff
            best_strike = strike
            best_iv = iv

    if best_strike is not None:
        return best_strike, best_iv

    # Fallback: moneyness proxy
    # 25-delta call ≈ spot * 1.05, 25-delta put ≈ spot * 0.95
    if opt_type == "call":
        target_strike = spot * 1.05
    else:
        target_strike = spot * 0.95

    closest = min(strikes.keys(), key=lambda k: abs(k - target_strike), default=None)
    if closest is not None:
        iv = strikes[closest].get(iv_key, 0)
        return closest, iv

    return None, 0.0


# ─── Strategy Pricer ─────────────────────────────────────────────────────────

def price_strategy(surface: dict, legs: list) -> dict:
    """
    Price a multi-leg options strategy using the vol surface.

    Each leg: {"symbol": "OCC", "side": "buy"/"sell", "strike": float,
               "expiry": "YYYY-MM-DD", "opt_type": "call"/"put"}

    Returns fair value, market price, edge, probability of profit,
    expected value, max profit/loss, and breakevens.
    """
    if "error" in surface:
        return {"error": surface["error"]}

    spot = surface["spot_price"]
    expirations = surface.get("expirations", {})
    r = RISK_FREE_RATE

    total_fair = 0.0
    total_market = 0.0
    leg_details = []

    for leg in legs:
        strike = leg["strike"]
        expiry = leg["expiry"]
        opt_type = leg["opt_type"]
        side = leg["side"]  # "buy" or "sell"
        sign = 1.0 if side == "buy" else -1.0

        # Look up IV and market price from surface
        exp_data = expirations.get(expiry, {})
        strikes_data = exp_data.get("strikes", {})
        strike_data = strikes_data.get(strike, {})

        iv_key = f"{opt_type}_iv"
        mid_key = f"{opt_type}_mid"
        bid_key = f"{opt_type}_bid"
        ask_key = f"{opt_type}_ask"

        iv = strike_data.get(iv_key, 0)
        mid = strike_data.get(mid_key, 0)
        bid = strike_data.get(bid_key, 0)
        ask = strike_data.get(ask_key, 0)

        # If exact strike not found, interpolate IV from neighbors
        if iv <= 0 and strikes_data:
            iv = _interpolate_iv(strikes_data, strike, opt_type)

        dte = exp_data.get("dte", 0)
        T = dte / 365.0 if dte > 0 else 0.001

        # Fair value: prefer SABR vol, fall back to surface IV, then BS
        sabr_p = surface.get("sabr_params", {}).get(expiry, {})
        sabr_vol = 0.0
        if sabr_p and sabr_p.get("alpha", 0) > 0:
            F_leg = spot * exp(r * T)
            sabr_vol = sabr_implied_vol(
                F_leg, strike, T,
                sabr_p["alpha"], sabr_p.get("beta", 1.0),
                sabr_p["rho"], sabr_p["nu"]
            )

        effective_iv = sabr_vol if sabr_vol > 0 else iv
        fair_price = bs_price(spot, strike, T, r, effective_iv, opt_type) if effective_iv > 0 else mid

        # Market mid
        market_mid = mid if mid > 0 else fair_price

        total_fair += sign * fair_price
        total_market += sign * market_mid

        leg_details.append({
            "symbol": leg.get("symbol", ""),
            "strike": strike,
            "expiry": expiry,
            "opt_type": opt_type,
            "side": side,
            "iv": round(iv, 6),
            "sabr_iv": round(sabr_vol, 6) if sabr_vol > 0 else None,
            "effective_iv": round(effective_iv, 6),
            "fair_price": round(fair_price, 4),
            "market_mid": round(market_mid, 4),
            "delta": round(bs_delta(spot, strike, T, r, effective_iv, opt_type), 4) if effective_iv > 0 else 0.0,
        })

    edge = total_fair - total_market
    edge_pct = (edge / abs(total_fair)) * 100 if abs(total_fair) > 0.001 else 0.0

    # Compute max profit, max loss, breakevens, and probability of profit
    payoff_analysis = _analyze_strategy_payoff(legs, spot, surface)

    return {
        "fair_value": round(total_fair, 4),
        "market_price": round(total_market, 4),
        "edge": round(edge, 4),
        "edge_pct": round(edge_pct, 2),
        "probability_of_profit": round(payoff_analysis["pop"], 4),
        "expected_value": round(payoff_analysis["ev"], 4),
        "max_profit": round(payoff_analysis["max_profit"], 4),
        "max_loss": round(payoff_analysis["max_loss"], 4),
        "breakevens": payoff_analysis["breakevens"],
        "legs": leg_details,
    }


def _interpolate_iv(strikes_data: dict, target_strike: float, opt_type: str) -> float:
    """Linear interpolation of IV between two nearest strikes."""
    iv_key = f"{opt_type}_iv"
    valid = [(k, v.get(iv_key, 0)) for k, v in strikes_data.items() if v.get(iv_key, 0) > 0]
    if not valid:
        return 0.0
    valid.sort(key=lambda x: x[0])

    strikes = [v[0] for v in valid]
    ivs = [v[1] for v in valid]

    if target_strike <= strikes[0]:
        return ivs[0]
    if target_strike >= strikes[-1]:
        return ivs[-1]

    for i in range(len(strikes) - 1):
        if strikes[i] <= target_strike <= strikes[i + 1]:
            w = (target_strike - strikes[i]) / (strikes[i + 1] - strikes[i])
            return ivs[i] + w * (ivs[i + 1] - ivs[i])

    return ivs[0]


def _analyze_strategy_payoff(legs: list, spot: float, surface: dict) -> dict:
    """
    Analyze a multi-leg strategy's payoff profile.
    Computes max profit, max loss, breakevens, POP, and expected value.
    Uses a grid of terminal prices to evaluate the payoff.
    """
    expirations = surface.get("expirations", {})

    # Determine the cost/credit to enter
    net_premium = 0.0
    for leg in legs:
        expiry = leg["expiry"]
        strike = leg["strike"]
        opt_type = leg["opt_type"]
        side = leg["side"]
        sign = 1.0 if side == "buy" else -1.0

        exp_data = expirations.get(expiry, {})
        strikes_data = exp_data.get("strikes", {})
        sd = strikes_data.get(strike, {})
        mid = sd.get(f"{opt_type}_mid", 0)
        net_premium += sign * mid  # Positive = debit, negative = credit

    # Build payoff at expiration across a grid of terminal prices
    low = spot * 0.70
    high = spot * 1.30
    n_points = 1000
    step = (high - low) / n_points

    payoffs = []
    for i in range(n_points + 1):
        terminal = low + i * step
        pnl = -net_premium  # Start with premium paid/received (inverted for PnL)
        for leg in legs:
            strike = leg["strike"]
            opt_type = leg["opt_type"]
            side = leg["side"]
            sign = 1.0 if side == "buy" else -1.0

            if opt_type == "call":
                intrinsic = max(0.0, terminal - strike)
            else:
                intrinsic = max(0.0, strike - terminal)

            pnl += sign * intrinsic

        payoffs.append((terminal, pnl))

    # Max profit / max loss
    pnl_values = [p[1] for p in payoffs]
    max_profit = max(pnl_values)
    max_loss = min(pnl_values)

    # Breakevens: where payoff crosses zero
    breakevens = []
    for i in range(1, len(payoffs)):
        prev_pnl = payoffs[i - 1][1]
        curr_pnl = payoffs[i][1]
        if (prev_pnl < 0 and curr_pnl >= 0) or (prev_pnl >= 0 and curr_pnl < 0):
            # Linear interpolation to find zero crossing
            if abs(curr_pnl - prev_pnl) > 1e-10:
                frac = abs(prev_pnl) / abs(curr_pnl - prev_pnl)
                be = payoffs[i - 1][0] + frac * step
                breakevens.append(round(be, 2))

    # Probability of profit — use ATM vol and log-normal distribution
    # for the terminal price distribution
    atm_iv = surface.get("atm_iv", 0.20)
    # Use the first leg's DTE for time horizon
    first_expiry = legs[0]["expiry"] if legs else None
    dte = expirations.get(first_expiry, {}).get("dte", 30) if first_expiry else 30
    T = dte / 365.0

    pop, ev = _monte_carlo_pop(payoffs, spot, atm_iv, T, net_premium)

    return {
        "max_profit": max_profit,
        "max_loss": max_loss,
        "breakevens": breakevens,
        "pop": pop,
        "ev": ev,
    }


def _monte_carlo_pop(payoffs: list, spot: float, sigma: float, T: float,
                     net_premium: float) -> tuple:
    """
    Estimate probability of profit and expected value using a
    log-normal distribution (analytical approximation over the payoff grid).
    """
    if sigma <= 0 or T <= 0:
        profitable = sum(1 for _, pnl in payoffs if pnl > 0)
        avg_pnl = sum(pnl for _, pnl in payoffs) / len(payoffs) if payoffs else 0
        return profitable / len(payoffs) if payoffs else 0.5, avg_pnl

    r = RISK_FREE_RATE
    drift = (r - 0.5 * sigma ** 2) * T
    vol_sqrt_t = sigma * sqrt(T)

    weighted_pnl = 0.0
    profitable_prob = 0.0
    total_prob = 0.0

    for terminal, pnl in payoffs:
        if terminal <= 0:
            continue
        # Log-normal probability density at this terminal price
        log_ratio = log(terminal / spot)
        z = (log_ratio - drift) / vol_sqrt_t
        # Probability density (unnormalized for grid)
        pdf_val = exp(-0.5 * z * z) / (terminal * vol_sqrt_t)
        weighted_pnl += pnl * pdf_val
        total_prob += pdf_val
        if pnl > 0:
            profitable_prob += pdf_val

    if total_prob > 0:
        pop = profitable_prob / total_prob
        ev = weighted_pnl / total_prob
    else:
        pop = 0.5
        ev = 0.0

    return pop, ev


# ─── VRP Calculator ──────────────────────────────────────────────────────────

def get_vrp(ticker: str) -> dict:
    """
    Calculate the Volatility Risk Premium for a ticker.

    VRP = Implied Vol - Realized Vol. Positive VRP indicates a premium
    for selling options (insurance premium).
    """
    surface = build_surface(ticker)
    if "error" in surface:
        return {"error": surface["error"], "ticker": ticker}

    atm_iv = surface["atm_iv"]
    rv_20 = surface["realized_vol_20d"]
    rv_60 = surface["realized_vol_60d"]

    vrp_20 = atm_iv - rv_20 if (atm_iv > 0 and rv_20 > 0) else 0.0
    vrp_60 = atm_iv - rv_60 if (atm_iv > 0 and rv_60 > 0) else 0.0

    # VRP percentile heuristic:
    # Typical VRP for equities is 2-8% (0.02-0.08) annualized.
    # Map to 0-100 scale.
    vrp_ref = vrp_20 if vrp_20 != 0 else vrp_60
    vrp_pct = min(100.0, max(0.0, (vrp_ref - (-0.05)) / (0.15 - (-0.05)) * 100.0))

    # Recommendation logic
    if vrp_20 > 0.04:
        recommendation = "sell_premium"
    elif vrp_20 < -0.02:
        recommendation = "buy_premium"
    else:
        recommendation = "neutral"

    return {
        "ticker": ticker,
        "implied_vol": round(atm_iv, 6),
        "realized_vol_20d": round(rv_20, 6),
        "realized_vol_60d": round(rv_60, 6),
        "vrp_20d": round(vrp_20, 6),
        "vrp_60d": round(vrp_60, 6),
        "vrp_percentile": round(vrp_pct, 2),
        "recommendation": recommendation,
    }


# ─── Surface Score ───────────────────────────────────────────────────────────

def get_surface_score(ticker: str) -> dict:
    """
    One-call function: build surface, analyze skew, compute VRP,
    return a single composite score with a strategy recommendation.
    """
    surface = build_surface(ticker)
    if "error" in surface:
        return {"ticker": ticker, "error": surface["error"]}

    skew = analyze_skew(surface)
    vrp = get_vrp(ticker)

    # ── Composite score (0-100) ──
    # Components:
    #   1. VRP score (0-40): higher VRP → higher score for selling premium
    #   2. Skew score (0-30): higher put-call skew → more opportunity
    #   3. Term structure score (0-15): contango (normal) vs backwardation (fear)
    #   4. Smile curvature score (0-15): convex smile → tail risk priced in

    # VRP component
    vrp_val = vrp.get("vrp_20d", 0)
    if vrp_val > 0:
        vrp_score = min(40.0, (vrp_val / 0.10) * 40.0)  # 10% VRP = max score
    else:
        vrp_score = max(0.0, 20.0 + vrp_val / 0.05 * 20.0)  # Negative VRP still gets some score

    # Skew component
    skew_val = abs(skew.get("put_call_skew", 0))
    skew_score = min(30.0, (skew_val / 0.08) * 30.0)

    # Term structure component
    term_slope = skew.get("term_structure_slope", 0)
    if term_slope > 0:
        # Contango (normal) — good for selling
        term_score = min(15.0, (term_slope / 0.05) * 15.0)
    else:
        # Backwardation (fear) — elevated near-term risk
        term_score = max(0.0, 7.5 + (term_slope / 0.05) * 7.5)

    # Smile curvature component
    curv = abs(skew.get("smile_curvature", 0))
    curv_score = min(15.0, (curv / 0.05) * 15.0)

    surface_score = round(vrp_score + skew_score + term_score + curv_score, 2)
    surface_score = min(100.0, max(0.0, surface_score))

    # ── Strategy recommendation ──
    best_strategy, edge_estimate, reasoning = _recommend_strategy(
        surface, skew, vrp, surface_score
    )

    return {
        "ticker": ticker,
        "surface_score": surface_score,
        "best_strategy": best_strategy,
        "edge_estimate": round(edge_estimate, 2),
        "skew": skew,
        "vrp": vrp,
        "reasoning": reasoning,
    }


def _recommend_strategy(surface: dict, skew: dict, vrp: dict, score: float) -> tuple:
    """
    Determine the best strategy based on surface characteristics.
    Returns (strategy_name, edge_estimate, reasoning).
    """
    spot = surface["spot_price"]
    vrp_20 = vrp.get("vrp_20d", 0)
    put_call_skew = skew.get("put_call_skew", 0)
    term_slope = skew.get("term_structure_slope", 0)
    overpriced = skew.get("overpriced_side", "neutral")
    skew_edge = skew.get("skew_edge", 0)

    reasons = []

    # Decision tree
    if vrp_20 > 0.04 and overpriced == "puts":
        strategy = "put_credit_spread"
        edge = skew_edge * 0.5
        reasons.append(f"VRP of {vrp_20:.1%} favors premium selling")
        reasons.append(f"Put skew of {put_call_skew:.1%} indicates overpriced downside protection")
        reasons.append("Put credit spreads capture both VRP and skew edge")

    elif vrp_20 > 0.04 and abs(put_call_skew) < 0.02:
        strategy = "iron_condor"
        edge = skew_edge * 0.3
        reasons.append(f"VRP of {vrp_20:.1%} favors premium selling")
        reasons.append("Symmetric skew allows balanced iron condor")
        reasons.append("Collect premium from both sides of the range")

    elif vrp_20 > 0.02:
        strategy = "sell_premium"
        edge = skew_edge * 0.4
        reasons.append(f"Moderate VRP of {vrp_20:.1%}")
        if overpriced == "puts":
            reasons.append("Lean toward selling puts")
        elif overpriced == "calls":
            reasons.append("Lean toward selling calls")

    elif vrp_20 < -0.02 and term_slope < -0.01:
        strategy = "straddle"
        edge = abs(vrp_20) * spot * 0.01
        reasons.append(f"Negative VRP ({vrp_20:.1%}) — implied vol is cheap")
        reasons.append("Backwardation signals near-term event risk")
        reasons.append("Buy straddle to capture potential move")

    elif vrp_20 < -0.02:
        strategy = "buy_premium"
        edge = abs(vrp_20) * spot * 0.005
        reasons.append(f"Negative VRP ({vrp_20:.1%}) — options are cheap")
        reasons.append("Consider buying premium ahead of expected moves")

    elif term_slope < -0.02:
        strategy = "calendar_spread"
        edge = abs(term_slope) * spot * 0.1
        reasons.append(f"Inverted term structure (slope: {term_slope:.1%})")
        reasons.append("Sell expensive near-term, buy cheap far-term")

    else:
        strategy = "neutral"
        edge = 0.0
        reasons.append("No strong directional signal from vol surface")
        reasons.append(f"VRP: {vrp_20:.1%}, Skew: {put_call_skew:.1%}")

    reasoning = "; ".join(reasons)
    return strategy, edge, reasoning


# ─── SABR Skew Edge ──────────────────────────────────────────────────────────

def get_skew_edge(ticker: str) -> dict:
    """
    Identify exploitable mispricing from volatility skew using SABR.

    The key insight quants use:
    - rho parameter tells us the correlation between price and vol
    - Negative rho (typical for equities) means puts are systematically overpriced
    - The DEGREE of overpricing varies with market conditions
    - When rho is MORE negative than historical average -> puts are EXTRA overpriced -> sell them
    - When rho is LESS negative than usual -> puts are fairly priced -> no edge

    Returns:
        dict with rho, rho_percentile, put_overpricing_pct, best_action,
        edge_per_contract, and reasoning.
    """
    surface = build_surface(ticker)
    if "error" in surface:
        return {"error": surface["error"], "ticker": ticker}

    spot = surface["spot_price"]
    expirations = surface.get("expirations", {})
    sabr_params = surface.get("sabr_params", {})

    if not sabr_params:
        return {
            "ticker": ticker,
            "error": "no_sabr_params",
            "rho": 0.0,
            "rho_percentile": 50.0,
            "put_overpricing_pct": 0.0,
            "best_action": "neutral",
            "edge_per_contract": 0.0,
            "reasoning": "SABR calibration not available — insufficient data.",
        }

    rho_values = []
    nu_values = []
    primary_params = None
    primary_expiry = None
    primary_dte = None

    for expiry_key, params in sorted(sabr_params.items()):
        if params.get("alpha", 0) <= 0 or params.get("fit_error", 1.0) > 0.05:
            continue
        rho_values.append(params["rho"])
        nu_values.append(params["nu"])
        if primary_params is None:
            primary_params = params
            primary_expiry = expiry_key
            ed = expirations.get(expiry_key, {})
            primary_dte = ed.get("dte", 30)

    if primary_params is None:
        # Relax fit_error threshold and try again
        for expiry_key, params in sorted(sabr_params.items()):
            if params.get("alpha", 0) <= 0:
                continue
            rho_values.append(params["rho"])
            nu_values.append(params["nu"])
            if primary_params is None:
                primary_params = params
                primary_expiry = expiry_key
                ed = expirations.get(expiry_key, {})
                primary_dte = ed.get("dte", 30)

    if primary_params is None:
        return {
            "ticker": ticker,
            "rho": 0.0,
            "rho_percentile": 50.0,
            "put_overpricing_pct": 0.0,
            "best_action": "neutral",
            "edge_per_contract": 0.0,
            "reasoning": "SABR calibration failed for all expirations.",
        }

    current_rho = primary_params["rho"]
    current_nu = primary_params["nu"]
    current_alpha = primary_params["alpha"]
    current_beta = primary_params.get("beta", 1.0)

    # Rho percentile: -0.8 (extreme) -> 100, 0.0 (flat) -> 0
    rho_pct = min(100.0, max(0.0, (-current_rho) / 0.8 * 100.0))

    T = primary_dte / 365.0 if primary_dte and primary_dte > 0 else 30.0 / 365.0
    F = spot * exp(RISK_FREE_RATE * T)
    atm_vol = sabr_implied_vol(F, F, T, current_alpha, current_beta, current_rho, current_nu)
    put_25d_strike = spot * 0.95
    put_25d_vol = sabr_implied_vol(F, put_25d_strike, T, current_alpha, current_beta,
                                    current_rho, current_nu)

    if atm_vol > 0 and put_25d_vol > 0:
        put_overpricing = (put_25d_vol - atm_vol) / atm_vol * 100.0
    else:
        put_overpricing = 0.0

    sabr_put_price = bs_price(spot, put_25d_strike, T, RISK_FREE_RATE, put_25d_vol, "put") if put_25d_vol > 0 else 0
    flat_put_price = bs_price(spot, put_25d_strike, T, RISK_FREE_RATE, atm_vol, "put") if atm_vol > 0 else 0
    edge_per_contract = (sabr_put_price - flat_put_price) * 100

    reasons = []
    if current_rho < -0.5:
        best_action = "sell_puts"
        reasons.append(f"Rho = {current_rho:.2f}: steep negative skew — puts significantly overpriced")
        reasons.append(f"Put overpricing at 25-delta: {put_overpricing:.1f}% above ATM vol")
        reasons.append("Sell OTM put spreads to capture skew premium")
    elif current_rho < -0.3:
        best_action = "sell_condor"
        reasons.append(f"Rho = {current_rho:.2f}: moderate negative skew — puts mildly overpriced")
        reasons.append("Iron condor captures premium on both sides with put skew edge")
    elif current_rho > -0.1 and current_nu > 0.5:
        best_action = "buy_straddle"
        reasons.append(f"Rho = {current_rho:.2f}: flat skew with high vol-of-vol (nu={current_nu:.2f})")
        reasons.append("Vol-of-vol suggests large moves expected but not priced in skew")
    else:
        best_action = "neutral"
        reasons.append(f"Rho = {current_rho:.2f}: no strong skew signal")
        reasons.append("Wait for skew to become more extreme before trading")

    if len(rho_values) > 1:
        avg_rho = sum(rho_values) / len(rho_values)
        reasons.append(f"Average rho across expirations: {avg_rho:.2f}")

    return {
        "ticker": ticker,
        "rho": round(current_rho, 4),
        "rho_percentile": round(rho_pct, 2),
        "put_overpricing_pct": round(put_overpricing, 2),
        "best_action": best_action,
        "edge_per_contract": round(edge_per_contract, 2),
        "reasoning": "; ".join(reasons),
        "sabr_params": primary_params,
        "expiry_used": primary_expiry,
    }


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    ticker = "SPY"
    print(f"\n{'='*60}")
    print(f"  VolTradeAI — Volatility Surface: {ticker}")
    print(f"{'='*60}\n")

    # 1. Build the surface
    print("[1] Building volatility surface...")
    surface = build_surface(ticker)
    if "error" in surface:
        print(f"  ERROR: {surface['error']}")
    else:
        print(f"  Spot: ${surface['spot_price']:.2f}")
        print(f"  Expirations: {len(surface['expirations'])}")
        total_strikes = sum(len(e['strikes']) for e in surface['expirations'].values())
        print(f"  Total strike entries: {total_strikes}")
        print(f"  ATM IV: {surface['atm_iv']:.2%}")
        print(f"  Realized Vol (20d): {surface['realized_vol_20d']:.2%}")
        print(f"  Realized Vol (60d): {surface['realized_vol_60d']:.2%}")
        print(f"  VRP: {surface['vrp']:.2%}")

    # 2. Analyze skew
    print(f"\n[2] Analyzing skew...")
    skew = analyze_skew(surface)
    if "error" not in skew:
        print(f"  Put-Call Skew: {skew['put_call_skew']:.4f}")
        print(f"  Skew Percentile: {skew['skew_percentile']:.1f}")
        print(f"  Term Structure Slope: {skew['term_structure_slope']:.4f}")
        print(f"  Smile Curvature: {skew['smile_curvature']:.4f}")
        print(f"  Overpriced Side: {skew['overpriced_side']}")
        print(f"  Skew Edge: ${skew['skew_edge']:.2f}")

    # 3. VRP
    print(f"\n[3] Computing VRP...")
    vrp = get_vrp(ticker)
    if "error" not in vrp:
        print(f"  Implied Vol: {vrp['implied_vol']:.2%}")
        print(f"  Realized Vol 20d: {vrp['realized_vol_20d']:.2%}")
        print(f"  VRP (20d): {vrp['vrp_20d']:.2%}")
        print(f"  VRP (60d): {vrp['vrp_60d']:.2%}")
        print(f"  VRP Percentile: {vrp['vrp_percentile']:.1f}")
        print(f"  Recommendation: {vrp['recommendation']}")

    # 4. Surface Score
    print(f"\n[4] Computing Surface Score...")
    score = get_surface_score(ticker)
    if "error" not in score:
        print(f"  Surface Score: {score['surface_score']:.1f}/100")
        print(f"  Best Strategy: {score['best_strategy']}")
        print(f"  Edge Estimate: ${score['edge_estimate']:.2f}")
        print(f"  Reasoning: {score['reasoning']}")

    # 5. Test strategy pricing (ATM straddle)
    if "error" not in surface and surface['expirations']:
        print(f"\n[5] Pricing ATM Straddle...")
        first_expiry = list(surface['expirations'].keys())[0]
        exp_data = surface['expirations'][first_expiry]
        strikes = sorted(exp_data['strikes'].keys())
        atm_strike = min(strikes, key=lambda k: abs(k - surface['spot_price']))

        straddle_legs = [
            {"symbol": "", "side": "buy", "strike": atm_strike, "expiry": first_expiry, "opt_type": "call"},
            {"symbol": "", "side": "buy", "strike": atm_strike, "expiry": first_expiry, "opt_type": "put"},
        ]
        straddle = price_strategy(surface, straddle_legs)
        print(f"  Expiry: {first_expiry} ({exp_data['dte']} DTE)")
        print(f"  Strike: ${atm_strike:.2f}")
        print(f"  Fair Value: ${straddle['fair_value']:.2f}")
        print(f"  Market Price: ${straddle['market_price']:.2f}")
        print(f"  Edge: ${straddle['edge']:.2f} ({straddle['edge_pct']:.1f}%)")
        print(f"  P(Profit): {straddle['probability_of_profit']:.1%}")
        print(f"  Expected Value: ${straddle['expected_value']:.2f}")
        print(f"  Max Profit: ${straddle['max_profit']:.2f}")
        print(f"  Max Loss: ${straddle['max_loss']:.2f}")
        print(f"  Breakevens: {straddle['breakevens']}")

    # 6. SABR model testing
    print(f"\n[6] SABR Model Testing...")

    # 6a. Unit test: known SABR params should produce correct ATM vol
    test_F = 100.0
    test_alpha = 0.20
    test_beta = 1.0
    test_rho = -0.30
    test_nu = 0.40
    test_T = 0.25
    atm_sabr = sabr_implied_vol(test_F, test_F, test_T, test_alpha, test_beta, test_rho, test_nu)
    print(f"  SABR ATM vol (F=100, alpha=0.20, rho=-0.30, nu=0.40, T=0.25): {atm_sabr:.4f}")
    assert atm_sabr > 0.15 and atm_sabr < 0.30, f"ATM vol out of range: {atm_sabr}"

    # 6b. Skew: OTM put should have higher vol than ATM (for rho < 0)
    put_strike = 90.0
    call_strike = 110.0
    put_sabr = sabr_implied_vol(test_F, put_strike, test_T, test_alpha, test_beta, test_rho, test_nu)
    call_sabr = sabr_implied_vol(test_F, call_strike, test_T, test_alpha, test_beta, test_rho, test_nu)
    print(f"  SABR put vol (K=90):  {put_sabr:.4f}")
    print(f"  SABR ATM vol (K=100): {atm_sabr:.4f}")
    print(f"  SABR call vol (K=110): {call_sabr:.4f}")
    assert put_sabr > atm_sabr, "Negative rho should make OTM put vol > ATM vol"
    print(f"  Skew confirmed: put vol > ATM vol > call vol ✓")

    # 6c. Calibration test with synthetic smile
    syn_strikes = [85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0]
    syn_vols = [sabr_implied_vol(test_F, k, test_T, test_alpha, test_beta, test_rho, test_nu)
                for k in syn_strikes]
    cal_result = calibrate_sabr(test_F, syn_strikes, syn_vols, test_T, beta=test_beta)
    print(f"  Calibration test:")
    print(f"    True:       alpha={test_alpha:.4f}, rho={test_rho:.4f}, nu={test_nu:.4f}")
    print(f"    Calibrated: alpha={cal_result['alpha']:.4f}, rho={cal_result['rho']:.4f}, nu={cal_result['nu']:.4f}")
    print(f"    Fit error (RMSE): {cal_result['fit_error']:.6f}")
    assert cal_result['fit_error'] < 0.01, f"Calibration RMSE too high: {cal_result['fit_error']}"
    print(f"  Calibration accuracy ✓")

    # 6d. SABR probability test
    p_otm_call = sabr_probability_otm(test_F, 110.0, test_T, test_alpha, test_beta, test_rho, test_nu, "call")
    p_otm_put = sabr_probability_otm(test_F, 90.0, test_T, test_alpha, test_beta, test_rho, test_nu, "put")
    print(f"  P(OTM) for 110 call: {p_otm_call:.4f}")
    print(f"  P(OTM) for 90 put:  {p_otm_put:.4f}")
    assert 0.5 < p_otm_call < 1.0, "OTM call should have > 50% chance of expiring OTM"
    assert 0.5 < p_otm_put < 1.0, "OTM put should have > 50% chance of expiring OTM"
    print(f"  OTM probabilities ✓")

    # 6e. SABR params from live surface
    if "error" not in surface:
        sabr_p = surface.get("sabr_params", {})
        if sabr_p:
            print(f"  Live SABR params ({len(sabr_p)} expirations calibrated):")
            for exp_key in list(sorted(sabr_p.keys()))[:3]:
                sp = sabr_p[exp_key]
                print(f"    {exp_key}: alpha={sp['alpha']:.4f}, rho={sp['rho']:.4f}, "
                      f"nu={sp['nu']:.4f}, err={sp['fit_error']:.6f}")
        else:
            print(f"  No SABR params calibrated (data may be insufficient)")

    # 6f. Skew edge analysis
    print(f"\n[7] Skew Edge Analysis...")
    edge_result = get_skew_edge(ticker)
    if "error" not in edge_result or edge_result.get("rho", 0) != 0:
        print(f"  Rho: {edge_result.get('rho', 'N/A')}")
        print(f"  Rho Percentile: {edge_result.get('rho_percentile', 'N/A')}")
        print(f"  Put Overpricing: {edge_result.get('put_overpricing_pct', 'N/A')}%")
        print(f"  Best Action: {edge_result.get('best_action', 'N/A')}")
        print(f"  Edge/Contract: ${edge_result.get('edge_per_contract', 0):.2f}")
        print(f"  Reasoning: {edge_result.get('reasoning', 'N/A')}")
    else:
        print(f"  Skew edge: {edge_result}")

    print(f"\n{'='*60}")
    print("  Done.")
    print(f"{'='*60}")
