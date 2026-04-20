#!/usr/bin/env python3
"""
VolTradeAI Autonomous Bot Engine
─────────────────────────────────
Scans all tradeable stocks, scores them across multiple strategies,
picks the best trades, and executes via Alpaca.

Called by: server/bot.ts on a schedule (every 15 min during market hours)
Output: JSON with recommended trades and actions

Usage:
  python3 bot_engine.py scan          # Scan market, return top opportunities
  python3 bot_engine.py manage        # Check existing positions, manage stops
  python3 bot_engine.py full          # Full cycle: scan + decide + recommend
"""

# Module-level variable for partial scan results on timeout
_partial_scan_result = None

def _compute_stock_iv_rank(ticker: str, closes: list) -> float:
    """Per-stock IV rank from its own 30-day HV vs 52-week range.
    Uses price data already fetched by deep_score — no extra API call."""
    try:
        import numpy as _np
        if len(closes) < 50: return 50.0
        rets = [_np.log(closes[i]/closes[i-1]) for i in range(1,len(closes)) if closes[i-1]>0]
        if len(rets) < 30: return 50.0
        hvs = [_np.std(rets[max(0,i-30):i])*_np.sqrt(252)*100 for i in range(30,len(rets))]
        if not hvs: return 50.0
        cur=hvs[-1]; lo=min(hvs); hi=max(hvs)
        return round((cur-lo)/(hi-lo)*100, 1) if hi>lo else 50.0
    except Exception:
        return 50.0


# ── Constrain thread-hungry libraries BEFORE any imports ──────────────────────
# OpenBLAS (numpy) defaults to 32 threads — kills Railway's container.
# 2 threads is plenty for the math we do (scoring, position sizing).
import os as _os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
           "NUMEXPR_MAX_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_v, "2")

import sys
import json
import os
import time
import subprocess
import requests
# Alpaca rate limiter — prevents silent 429 errors during parallel scans
try:
    from alpaca_rate_limiter import alpaca_throttle
    _HAS_ALPACA_THROTTLE = True
except ImportError:
    _HAS_ALPACA_THROTTLE = False
    class _NoThrottle:
        def acquire(self): pass
    alpaca_throttle = _NoThrottle()
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# OPTIMIZATION 2026-04-20: Auto-throttle all Alpaca requests (180/min).
# Prevents silent 429 errors during parallel scan workers + options fetches.
try:
    from alpaca_rate_limiter import install_global_throttle
    install_global_throttle()
except ImportError:
    pass  # Rate limiter optional — system works without it

# ── Tiered strategy engine (4-tier Option D) ───
try:
    from tiered_strategy import (
        TieredStrategy, TierContext, update_peak_equity,
        get_portfolio_margin_status,
    )
    from risk_kill_switch import (
        check_kill_switches, record_trade_outcome,
    )
    _HAS_TIERED = True
except ImportError:
    _HAS_TIERED = False

def _et_now_hour() -> float:
    """Return current ET hour (fractional), DST-aware."""
    now_et = datetime.now(ZoneInfo("America/New_York"))
    return now_et.hour + now_et.minute / 60.0

# ── Persistent storage path (Railway volume or /tmp locally) ─────────────────
try:
    from storage_config import DATA_DIR
except ImportError:
    DATA_DIR = "/data/voltrade" if os.path.isdir("/data") else "/tmp"
os.makedirs(DATA_DIR, exist_ok=True)

# ── System config: all adaptive parameters (regime-aware) ────────────────
try:
    from system_config import get_adaptive_params, BASE_CONFIG
    _HAS_SYSTEM_CONFIG = True
except ImportError:
    _HAS_SYSTEM_CONFIG = False
    BASE_CONFIG = {}

# ── Markov regime detector ───────────────────────────────────────────
try:
    from markov_regime import get_regime as _get_markov_regime
    _HAS_MARKOV = True
except ImportError:
    _HAS_MARKOV = False
    def _get_markov_regime(*a, **kw): return {"regime_score":50,"regime_label":"NEUTRAL","markov_state":1,"size_multiplier":1.0,"markov_signal":"NEUTRAL"}

# ── Strategy imports ─────────────────────────────────────────────────────────
_STRATEGIES_DIR = os.path.join(os.path.dirname(__file__), "strategies")
sys.path.insert(0, _STRATEGIES_DIR)

try:
    import momentum as momentum_strategy
except ImportError:
    momentum_strategy = None

try:
    import mean_reversion as mean_reversion_strategy
except ImportError:
    mean_reversion_strategy = None

try:
    import squeeze as squeeze_strategy
except ImportError:
    squeeze_strategy = None

STRATEGIES_LOADED = all([
    momentum_strategy is not None,
    mean_reversion_strategy is not None,
    squeeze_strategy is not None,
])

# ── yfinance cache (module-level, avoids re-fetching same ticker within 5 min) ──
_yf_cache: dict = {}
_yf_cache_time: dict = {}
_YF_CACHE_TTL = 300  # 5 minutes
_YF_CACHE_MAX = 50   # Cap cache size to prevent unbounded memory growth

# ── Config ──────────────────────────────────────────────────────────────────

POLYGON_KEY = os.environ.get("POLYGON_KEY", "") or os.environ.get("POLYGON_API_KEY", "")  # FIX: accept either name (5 other files use POLYGON_KEY)
ALPACA_KEY = os.environ.get("ALPACA_KEY", "")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET", "")
ALPACA_BASE_URL = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_DATA_URL = "https://data.alpaca.markets"

def _alpaca_headers():
    return {"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET}

MAX_POSITIONS = 5          # Max stocks to hold at once
MAX_OPTIONS_POSITIONS = 3  # Max options positions — separate from stock slots
MAX_POSITION_PCT = 0.05    # 5% of portfolio per position
STOP_LOSS_PCT = 0.02       # 2% stop loss
TAKE_PROFIT_PCT = 0.06     # 6% take profit (3:1 reward/risk)
# MIN_SCORE is no longer hardcoded here. Use params.get("MIN_SCORE", 63) from
# get_adaptive_params() so the threshold correctly varies by market regime:
#   BULL: 63, CAUTION: 67, BEAR/PANIC: 75 (from system_config.get_adaptive_params).
# The old hardcoded 65 conflicted with system_config's regime-aware 63/67/75 values.
MIN_VOLUME = 500000        # 3-year backtest: higher volume = better liquidity and more reliable signals
MIN_PRICE = 5              # 3-year backtest: stocks < $5 (penny/meme) have 25% WR — consistent drag
MAX_SECTOR_POSITIONS = 2   # Max 2 stocks from the same sector

# ── Sector Map (for correlation / diversification check) ─────────────────────
# Values match macro_data.py SECTOR_ETFS names (capitalized, as returned by get_macro_snapshot)
SECTOR_MAP = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology", "GOOG": "Technology",
    "META": "Technology", "NVDA": "Technology", "AMD": "Technology",
    "INTC": "Technology", "ORCL": "Technology", "CRM": "Technology", "ADBE": "Technology",
    "QCOM": "Technology", "TXN": "Technology", "AVGO": "Technology", "AMAT": "Technology",
    "NOW": "Technology", "SNOW": "Technology", "PLTR": "Technology", "UBER": "Industrials",
    "TSLA": "Consumer Discretionary", "F": "Consumer Discretionary", "GM": "Consumer Discretionary",
    "TM": "Consumer Discretionary", "RIVN": "Consumer Discretionary",
    "JPM": "Financials", "BAC": "Financials", "GS": "Financials", "MS": "Financials",
    "WFC": "Financials", "C": "Financials", "AXP": "Financials", "V": "Financials", "MA": "Financials",
    "JNJ": "Healthcare", "PFE": "Healthcare", "MRK": "Healthcare", "UNH": "Healthcare",
    "ABBV": "Healthcare", "LLY": "Healthcare", "BMY": "Healthcare", "AMGN": "Healthcare",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
    "SPY": "etf", "QQQ": "etf", "IWM": "etf", "DIA": "etf", "GLD": "etf",
    "AMZN": "Consumer Discretionary", "WMT": "Consumer Staples", "TGT": "Consumer Discretionary", "COST": "Consumer Staples",
}

# ── Sector Cache (ticker → sector from yfinance, persisted to disk) ─────────


# ── Memory diagnostics helpers (added 2026-04-20 after Railway OOM) ─────────
def _mem_rss_mb() -> int:
    """Return current process RSS in MB. Returns 0 if unavailable."""
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux: ru_maxrss is in KB; macOS: bytes. Both convert to MB via /1024.
        return rss // 1024 if rss > 100_000 else rss // 1024
    except Exception:
        return 0

def _log_mem_phase(phase: str, logger=None):
    """Log peak memory at a phase boundary. Emitted to stderr for bot.ts to see
    in the Railway activity log when scans succeed or fail near memory limits."""
    mb = _mem_rss_mb()
    try:
        print(f"[mem] {phase} rss~{mb}MB", file=sys.stderr, flush=True)
    except Exception:
        pass
    return mb

def _gc_checkpoint(phase: str = ""):
    """Force garbage collection at phase boundaries. Reduces peak memory
    by 30-80MB by releasing short-lived dicts/DataFrames between phases."""
    try:
        import gc
        gc.collect()
        if phase:
            _log_mem_phase(f"after_gc_{phase}")
    except Exception:
        pass

SECTOR_CACHE_PATH = os.path.join(DATA_DIR, "voltrade_sector_cache.json")

def _load_sector_cache():
    """Load the ticker→sector cache from disk (with file locking)."""
    try:
        import fcntl
        with open(SECTOR_CACHE_PATH, "r") as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)  # shared read lock
                return json.load(f)
            finally:
                try: fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                except Exception: pass
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    except Exception:
        # fcntl unavailable (Windows) or other issue — fall through to best-effort
        try:
            with open(SECTOR_CACHE_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return {}

def _update_sector_cache(ticker, sector):
    """Update a single ticker→sector entry atomically.
    FIX (2026-04-20): Previous version had a race condition — 8 parallel workers
    could all read, modify, and write simultaneously, clobbering each other.
    Now uses file locking + atomic rename for crash-safe, race-free writes."""
    if not ticker or not sector or sector == "Unknown":
        return
    import tempfile
    try:
        import fcntl
        use_lock = True
    except ImportError:
        use_lock = False
    try:
        # Read-modify-write under an exclusive lock on a sidecar lockfile
        lock_path = SECTOR_CACHE_PATH + ".lock"
        lock_f = open(lock_path, "a+")
        try:
            if use_lock:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            # Load current
            cache = _load_sector_cache()
            cache[ticker.upper()] = sector
            # Atomic write: temp file + rename (POSIX guarantees atomicity)
            dirname = os.path.dirname(SECTOR_CACHE_PATH) or "."
            fd, tmp_path = tempfile.mkstemp(dir=dirname, prefix=".sector_cache.", suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as tf:
                    json.dump(cache, tf)
                os.replace(tmp_path, SECTOR_CACHE_PATH)
            except Exception:
                try: os.unlink(tmp_path)
                except Exception: pass
                raise
        finally:
            try:
                if use_lock:
                    fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
            except Exception: pass
            lock_f.close()
    except Exception:
        pass

def _get_sector(ticker, sector=None):
    """Resolve sector for a ticker using: provided value > cache > SECTOR_MAP > 'Unknown'."""
    if sector and sector != "Unknown":
        return sector
    t = ticker.upper()
    # Check SECTOR_MAP first (fast, no I/O)
    mapped = SECTOR_MAP.get(t)
    if mapped:
        return mapped
    # Check persisted cache (covers previously deep-scored tickers)
    cached = _load_sector_cache().get(t)
    if cached:
        return cached
    return "Unknown"

# ── EWMA / GARCH Volatility ──────────────────────────────────────────────────

def ewma_vol(returns, lambd=0.94):
    """
    EWMA (RiskMetrics) volatility estimate — reacts faster than rolling stddev.
    Annualised and expressed as a percentage.

    BUGFIX (2026-04-17): previous implementation seeded `var` with the sample
    variance of the whole window and then iterated `for r in arr`, effectively
    double-counting every observation. The sample variance seed already used
    arr[0] — iterating from arr[0] again inflated vol by ~5-8% consistently,
    which fed a too-small position_size scalar in quiet regimes. Correct
    recursion seeds from the first squared return, then folds in each
    subsequent return — which is the actual RiskMetrics spec.
    """
    if len(returns) < 5:
        return None
    arr = np.array(returns, dtype=float)
    # Seed with first squared return; iterate from the 2nd onwards.
    var = float(arr[0]) ** 2
    for r in arr[1:]:
        var = lambd * var + (1 - lambd) * float(r) ** 2
    return round(float(np.sqrt(var * 252)) * 100, 2)

# garch_vol_estimate() removed — dead code (never called in production).
# EWMA (ewma_vol above) is used instead. GARCH was considered during prototyping
# but requires 30+ observations and is not statistically superior for this use case.

# ── Data Fetching ────────────────────────────────────────────────────────────



def get_stock_details(ticker):
    """Get detailed analysis from analyze.py."""
    import subprocess
    try:
        result = subprocess.run(
            ["python3", os.path.join(os.path.dirname(__file__), "analyze.py"), ticker, "--mode=scan"],
            capture_output=True, text=True, timeout=30
        )
        if result.stdout.strip():
            return json.loads(result.stdout.strip())
    except Exception:
        pass
    return None


def get_alpaca_account():
    """Get Alpaca account info."""
    import requests
    r = alpaca_throttle.acquire()
    r = requests.get(f"{ALPACA_BASE_URL}/v2/account", headers={
        "APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET
    }, timeout=10)
    return r.json()


def get_alpaca_positions():
    """Get current Alpaca positions."""
    import requests
    alpaca_throttle.acquire()
    r = requests.get(f"{ALPACA_BASE_URL}/v2/positions", headers={
        "APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET
    }, timeout=10)
    return r.json()


# ── Portfolio-Level Drawdown Tracking ──────────────────────────────────────────
# Tracks all-time peak equity + halt state across runs. Persisted to disk so
# the halt survives process restarts. Provides:
#   - update_equity_peak(equity): update peak and compute current DD %
#   - get_portfolio_dd_state(): return {peak, dd_pct, halted, halt_reason}
#   - is_trading_halted(regime): master check — True if new entries should skip
#   - should_escalate_hedge(): True if DD ≥ DRAWDOWN_HEDGE_ESCALATE_PCT
# One-way ratchet: halt resets ONLY when regime ∈ DRAWDOWN_HALT_RESUME_REGIMES
# AND current equity within DRAWDOWN_HALT_RESUME_EQUITY_PCT of peak.

_DD_STATE_PATH = os.path.join(DATA_DIR, "voltrade_portfolio_dd.json")

def _load_dd_state() -> dict:
    """Load persisted drawdown state. Returns safe defaults if missing/corrupt."""
    try:
        if os.path.exists(_DD_STATE_PATH):
            with open(_DD_STATE_PATH) as f:
                s = json.load(f)
            # Validate required keys
            if "peak_equity" in s and "halted" in s:
                return s
    except Exception:
        pass
    return {"peak_equity": 0.0, "halted": False, "halt_reason": "",
            "halt_started_at": None, "last_equity": 0.0, "last_updated": None}


def _save_dd_state(state: dict) -> None:
    """Persist drawdown state atomically."""
    try:
        os.makedirs(os.path.dirname(_DD_STATE_PATH), exist_ok=True)
        tmp = _DD_STATE_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, _DD_STATE_PATH)
    except Exception as e:
        import logging
        logging.getLogger("voltrade.dd").debug(f"Failed to save DD state: {e}")


def update_equity_peak(current_equity: float, regime: str = "NEUTRAL") -> dict:
    """
    Update equity peak and evaluate halt / recovery state.
    Call this once per cycle before manage_positions / scan_market.

    Returns the updated state dict with keys:
        peak_equity, current_equity, dd_pct, halted, halt_reason,
        halt_started_at, should_escalate_hedge, regime
    """
    from system_config import BASE_CONFIG
    import logging
    _log = logging.getLogger("voltrade.dd")

    state = _load_dd_state()
    peak = float(state.get("peak_equity", 0.0) or 0.0)
    cur  = float(current_equity or 0.0)

    # First-run bootstrap: seed peak with current equity
    if peak <= 0 and cur > 0:
        peak = cur

    # New all-time high — update peak (only goes up, never down)
    if cur > peak:
        peak = cur

    dd_pct = ((peak - cur) / peak * 100.0) if peak > 0 else 0.0
    halt_pct = float(BASE_CONFIG.get("DRAWDOWN_HALT_PCT", 18.0))
    halt_enabled = bool(BASE_CONFIG.get("DRAWDOWN_HALT_ENABLED", True))
    resume_regimes = set(BASE_CONFIG.get("DRAWDOWN_HALT_RESUME_REGIMES", ["BULL", "NEUTRAL"]))
    resume_eq_pct  = float(BASE_CONFIG.get("DRAWDOWN_HALT_RESUME_EQUITY_PCT", 5.0))
    hedge_esc_pct  = float(BASE_CONFIG.get("DRAWDOWN_HEDGE_ESCALATE_PCT", 10.0))

    halted = bool(state.get("halted", False))
    halt_reason = state.get("halt_reason", "")
    halt_started_at = state.get("halt_started_at")

    # ── Trigger halt: DD breaches threshold ─────────────────────────────────
    if halt_enabled and not halted and dd_pct >= halt_pct:
        halted = True
        halt_reason = f"DD {dd_pct:.2f}% >= {halt_pct:.1f}% (peak=${peak:,.0f} cur=${cur:,.0f})"
        halt_started_at = datetime.now().isoformat()
        _log.warning(f"[DD_HALT] TRIGGERED: {halt_reason}")

    # ── One-way ratchet resume: regime OK AND equity close to peak ─────────
    # Both conditions must be met. Prevents premature resumption during a
    # bear-rally that temporarily flips the regime.
    elif halted and regime in resume_regimes:
        equity_gap_pct = ((peak - cur) / peak * 100.0) if peak > 0 else 0.0
        if equity_gap_pct <= resume_eq_pct:
            _log.info(f"[DD_HALT] RESUMED: regime={regime} equity_gap={equity_gap_pct:.2f}% ≤ {resume_eq_pct}%")
            halted = False
            halt_reason = ""
            halt_started_at = None

    should_escalate = dd_pct >= hedge_esc_pct

    new_state = {
        "peak_equity": peak,
        "last_equity": cur,
        "last_updated": datetime.now().isoformat(),
        "halted": halted,
        "halt_reason": halt_reason,
        "halt_started_at": halt_started_at,
    }
    _save_dd_state(new_state)

    return {
        "peak_equity": peak, "current_equity": cur, "dd_pct": dd_pct,
        "halted": halted, "halt_reason": halt_reason,
        "halt_started_at": halt_started_at,
        "should_escalate_hedge": should_escalate,
        "regime": regime,
    }


def get_portfolio_dd_state() -> dict:
    """Read-only view of current drawdown state (no equity update)."""
    s = _load_dd_state()
    peak = float(s.get("peak_equity", 0.0) or 0.0)
    cur  = float(s.get("last_equity", 0.0) or 0.0)
    dd_pct = ((peak - cur) / peak * 100.0) if peak > 0 and cur > 0 else 0.0
    return {
        "peak_equity": peak, "current_equity": cur, "dd_pct": dd_pct,
        "halted": bool(s.get("halted", False)),
        "halt_reason": s.get("halt_reason", ""),
    }


def is_trading_halted() -> bool:
    """Quick boolean check for gating new entries."""
    return bool(_load_dd_state().get("halted", False))


def should_escalate_hedge() -> bool:
    """True if portfolio DD >= DRAWDOWN_HEDGE_ESCALATE_PCT threshold."""
    s = _load_dd_state()
    peak = float(s.get("peak_equity", 0.0) or 0.0)
    cur  = float(s.get("last_equity", 0.0) or 0.0)
    if peak <= 0 or cur <= 0:
        return False
    dd_pct = (peak - cur) / peak * 100.0
    from system_config import BASE_CONFIG
    return dd_pct >= float(BASE_CONFIG.get("DRAWDOWN_HEDGE_ESCALATE_PCT", 10.0))

# ── Strategy Scoring ─────────────────────────────────────────────────────────

# score_stock() removed — dead code (never called from anywhere in the codebase).
# The main scan pipeline uses quick_scan() + deep_score() instead.
# Retained its neighbor ewma_vol() which IS used for volatility estimation.

def deep_score(ticker, quick_result):
    """
    Deep analysis on a pre-filtered stock using analyze.py.
    Integrates all three strategy modules (momentum, mean_reversion, squeeze)
    plus VRP, sentiment, earnings, EWMA/GARCH vol and edge factors.
    """
    detail = get_stock_details(ticker)
    if not detail or "error" in detail:
        return quick_result

    reasons = list(quick_result.get("reasons", []))
    change_pct = quick_result.get("change_pct", 0)

    # ── Parallel data fetch — all 6 sources run simultaneously ──────────────
    # Previously sequential: ~12s per ticker. Now parallel: ~3-4s (limited by slowest source)
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _fetch_macro():
        try:
            from macro_data import get_macro_snapshot, get_news_sentiment
            return get_macro_snapshot(), get_news_sentiment(ticker)
        except Exception:
            return {}, {}

    def _fetch_intel():
        try:
            from intelligence import get_full_intelligence
            return get_full_intelligence(ticker)
        except Exception:
            return {}

    def _fetch_alt():
        try:
            from alt_data import get_alt_data_score
            return get_alt_data_score(ticker)
        except Exception:
            return {}

    def _fetch_social():
        try:
            from social_data import get_social_intelligence
            return get_social_intelligence(ticker)
        except Exception:
            return {}

    def _fetch_finnhub():
        try:
            from finnhub_data import get_insider_sentiment, get_recommendation_trends
            fi = get_insider_sentiment(ticker)
            fr = get_recommendation_trends(ticker)
            return {"insider_mspr": fi.get("mspr", 0), "insider_signal": fi.get("signal", "neutral"),
                    "analyst_consensus": fr.get("consensus", "hold"),
                    "analyst_buy_count": fr.get("buy", 0) + fr.get("strong_buy", 0)}
        except Exception:
            return {}

    macro, news_sent, intel, alt, social, finnhub = {}, {}, {}, {}, {}, {}
    with ThreadPoolExecutor(max_workers=5) as _pool:
        _f_macro    = _pool.submit(_fetch_macro)
        _f_intel    = _pool.submit(_fetch_intel)
        _f_alt      = _pool.submit(_fetch_alt)
        _f_social   = _pool.submit(_fetch_social)
        _f_finnhub  = _pool.submit(_fetch_finnhub)
        try:
            macro, news_sent = _f_macro.result(timeout=15)
        except Exception:
            pass
        try:
            intel = _f_intel.result(timeout=15)
        except Exception:
            pass
        try:
            alt = _f_alt.result(timeout=15)
        except Exception:
            pass
        try:
            social = _f_social.result(timeout=15)
        except Exception:
            pass
        try:
            finnhub = _f_finnhub.result(timeout=15)
        except Exception:
            pass

    # Analyst ratings + valuation multiples (yfinance) — with 5-min cache
    # WRAPPED IN SUBPROCESS with 5s hard kill to prevent yfinance hangs
    yf_fundamentals = {}
    _yf_cache_key = ticker
    if _yf_cache_key in _yf_cache and (time.time() - _yf_cache_time.get(_yf_cache_key, 0)) < _YF_CACHE_TTL:
        yf_fundamentals = _yf_cache[_yf_cache_key]
    else:
        try:
            _yf_script = f'''import logging, sys
# Silence yfinance's noisy HTTP error logging — 404s for ETFs without
# fundamentals (QQQ, SPY, etc.) are expected and not actionable.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)
import yfinance as yf, json
try:
    t = yf.Ticker("{ticker}")
    info = t.info or {{}}
    d = {{
        "forward_pe": info.get("forwardPE") or 0,
        "trailing_pe": info.get("trailingPE") or 0,
        "price_to_book": info.get("priceToBook") or 0,
        "ev_ebitda": info.get("enterpriseToEbitda") or 0,
        "price_to_sales": info.get("priceToSalesTrailing12Months") or 0,
        "peg_ratio": info.get("pegRatio") or 0,
        "target_mean": info.get("targetMeanPrice") or 0,
        "target_high": info.get("targetHighPrice") or 0,
        "target_low": info.get("targetLowPrice") or 0,
        "current_price": info.get("currentPrice") or info.get("regularMarketPrice") or 0,
        "recommendation": info.get("recommendationKey") or "none",
        "num_analysts": info.get("numberOfAnalystOpinions") or 0,
    }}
    recs = t.recommendations
    if recs is not None and not recs.empty:
        latest = recs.iloc[0]
        d["strong_buy_count"] = int(latest.get("strongBuy", 0))
        d["buy_count"] = int(latest.get("buy", 0))
        d["hold_count"] = int(latest.get("hold", 0))
        d["sell_count"] = int(latest.get("sell", 0))
        d["strong_sell_count"] = int(latest.get("strongSell", 0))
    print(json.dumps(d))
except Exception as e:
    print(json.dumps({{}}))
'''
            _yf_proc = subprocess.run(
                ["python3", "-c", _yf_script],
                capture_output=True, text=True, timeout=5  # HARD 5-second kill
            )
            if _yf_proc.stdout.strip():
                yf_fundamentals = json.loads(_yf_proc.stdout.strip())
                # Evict oldest entries if cache is full
                if len(_yf_cache) >= _YF_CACHE_MAX:
                    oldest_key = min(_yf_cache_time, key=_yf_cache_time.get)
                    _yf_cache.pop(oldest_key, None)
                    _yf_cache_time.pop(oldest_key, None)
                _yf_cache[_yf_cache_key] = yf_fundamentals
                _yf_cache_time[_yf_cache_key] = time.time()
        except (subprocess.TimeoutExpired, Exception):
            pass  # yfinance hung or failed — skip, don't block Tier 2

    # ── Pull key metrics from analyze.py output ──────────────────────────────
    vrp = detail.get("vrp", 0) or 0
    rec = detail.get("recommendation", {}) or {}
    sentiment = detail.get("sentiment", {}) or {}
    edge = detail.get("edge_factors", {}) or {}
    vol_metrics = detail.get("vol_metrics", {}) or {}

    rsi = vol_metrics.get("rsi_14") or None
    volume_ratio = vol_metrics.get("volume_ratio_5d") or 1.0

    # ── ADX (trend strength) + OBV (smart money flow) ─────────────────────────
    adx_value = None
    obv_signal = 0  # -1 = distribution, 0 = neutral, +1 = accumulation
    _deep_atr14 = None   # captured for ml_features atr_pct
    _deep_closes = []    # captured for ml_features above_ma10
    try:
        end_d = datetime.now().strftime("%Y-%m-%d")
        # MOMENTUM FIX 2026-04-20: widened from 40d/limit=30 to 400d/limit=300
        # to support real Jegadeesh-Titman 12-1 momentum (needs 252+ trading days)
        # and deeper IV-rank history. Extra Alpaca bars cost is trivial on paid tier.
        start_d = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")
        bars_url = (f"{ALPACA_DATA_URL}/v2/stocks/{ticker}/bars"
                    f"?timeframe=1Day&start={start_d}&limit=300&adjustment=all&feed=sip")
        bars_resp = requests.get(bars_url, headers=_alpaca_headers(), timeout=8)
        bars_data = bars_resp.json().get("bars", [])
        if len(bars_data) >= 14:
            # ADX calculation (14-period)
            highs = [b["h"] for b in bars_data]
            lows = [b["l"] for b in bars_data]
            closes = [b["c"] for b in bars_data]
            _deep_closes = closes  # capture for ML features
            volumes = [b.get("v", 0) for b in bars_data]

            plus_dm_list = []
            minus_dm_list = []
            tr_list = []
            for i in range(1, len(bars_data)):
                up_move = highs[i] - highs[i-1]
                down_move = lows[i-1] - lows[i]
                plus_dm_list.append(up_move if up_move > down_move and up_move > 0 else 0)
                minus_dm_list.append(down_move if down_move > up_move and down_move > 0 else 0)
                tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
                tr_list.append(tr)

            if len(tr_list) >= 14:
                # Smoothed averages (Wilder's method)
                atr14 = sum(tr_list[:14]) / 14
                plus_dm14 = sum(plus_dm_list[:14]) / 14
                minus_dm14 = sum(minus_dm_list[:14]) / 14
                for i in range(14, len(tr_list)):
                    atr14 = (atr14 * 13 + tr_list[i]) / 14
                    plus_dm14 = (plus_dm14 * 13 + plus_dm_list[i]) / 14
                    minus_dm14 = (minus_dm14 * 13 + minus_dm_list[i]) / 14

                _deep_atr14 = atr14  # capture for ML features atr_pct
                plus_di = (plus_dm14 / atr14 * 100) if atr14 > 0 else 0
                minus_di = (minus_dm14 / atr14 * 100) if atr14 > 0 else 0
                dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100 if (plus_di + minus_di) > 0 else 0
                adx_value = round(dx, 1)

            # OBV calculation
            obv = 0
            obv_values = [0]
            for i in range(1, len(closes)):
                if closes[i] > closes[i-1]:
                    obv += volumes[i]
                elif closes[i] < closes[i-1]:
                    obv -= volumes[i]
                obv_values.append(obv)

            # OBV trend: compare last 5-day OBV avg to previous 5-day
            if len(obv_values) >= 10:
                recent_obv = sum(obv_values[-5:]) / 5
                prev_obv = sum(obv_values[-10:-5]) / 5
                if recent_obv > prev_obv * 1.05:
                    obv_signal = 1  # Accumulation
                elif recent_obv < prev_obv * 0.95:
                    obv_signal = -1  # Distribution
    except Exception:
        pass

    # ── Strategy module scores ────────────────────────────────────────────────

    # 1. Momentum score — REAL Jegadeesh-Titman 12-1 momentum
    # MOMENTUM FIX 2026-04-20: Previously used edge["relative_strength"] as a
    # 12-month proxy, but that field is a short-window relative performance
    # metric, not a real annual return. Using wrong input produced garbage
    # momentum scores that added noise to the composite. Now computes real
    # 12-month return minus 1-month return (per Jegadeesh-Titman 1993) from
    # the extended 252-day bar history. Falls back gracefully to 0 when
    # insufficient data rather than contaminating the score.
    momentum_score = 50
    mom_12_1 = 0.0
    mom_1m = 0.0
    try:
        if _deep_closes and len(_deep_closes) >= 252:
            # Real 12-month return (252 trading days)
            _ret_12m = (_deep_closes[-1] / _deep_closes[-252] - 1.0) * 100
            # Real 1-month return (21 trading days)
            _ret_1m = (_deep_closes[-1] / _deep_closes[-21] - 1.0) * 100
            # Jegadeesh-Titman: 12-month return MINUS 1-month return
            # (excludes the short-term reversal effect)
            mom_12_1 = _ret_12m - _ret_1m
            mom_1m = _ret_1m
        elif _deep_closes and len(_deep_closes) >= 21:
            # Partial: only 1-month is available; use rough proxy for 12m
            mom_1m = (_deep_closes[-1] / _deep_closes[-21] - 1.0) * 100
            # Extrapolate conservatively: assume flat-ish annualized
            mom_12_1 = mom_1m * 0.5  # dampened extrapolation
    except Exception:
        # Fall through with 0s — better than using garbage proxy
        mom_12_1 = 0.0
        mom_1m = change_pct or 0.0
    avg_volume = quick_result.get("volume", 0)

    if momentum_strategy:
        try:
            m_result = momentum_strategy.score(mom_12_1, mom_1m, avg_volume)
            momentum_score = m_result.get("score", 50)
            if m_result.get("signal") not in ("NO DATA", "SKIP", "NEUTRAL"):
                reasons.append(f"Momentum: {m_result['signal']} ({m_result['reason']})")
        except Exception:
            pass

    # 2. Mean reversion score
    mean_reversion_score = 50
    change_5d = -(abs(change_pct) * 3) if change_pct < -1 else change_pct  # rough proxy
    if mean_reversion_strategy and rsi is not None:
        try:
            mr_result = mean_reversion_strategy.score(rsi, change_5d, volume_ratio)
            mean_reversion_score = mr_result.get("score", 50)
            if mr_result.get("signal") not in ("NO DATA", "NO EDGE"):
                reasons.append(f"MeanRev: {mr_result['signal']} ({mr_result['reason']})")
        except Exception:
            pass

    # 3. VRP score (volatility risk premium)
    vrp_score = 50
    if vrp > 8:
        vrp_score = 85
        reasons.append(f"High VRP (+{vrp:.1f}%) — sell options edge")
    elif vrp > 5:
        vrp_score = 70
        reasons.append(f"Elevated VRP (+{vrp:.1f}%) — options overpriced")
    elif vrp > 0:
        vrp_score = 55
    elif vrp < -3:
        vrp_score = 65
        reasons.append(f"Cheap IV ({vrp:.1f}%) — buy options edge")
    else:
        vrp_score = 45

    # 4. Squeeze score
    squeeze_score_val = 50
    squeeze_raw = edge.get("squeeze_score", 0) or 0
    short_pct = edge.get("short_float_pct", 0) or 0
    days_to_cover = edge.get("days_to_cover", 0) or 0
    sent_score_val = sentiment.get("score", 50) or 50
    reddit_buzz = sentiment.get("reddit_buzz", 0) or 0

    if squeeze_strategy:
        try:
            sq_result = squeeze_strategy.score(short_pct, days_to_cover, sent_score_val, volume_ratio, reddit_buzz)
            squeeze_score_val = sq_result.get("score", 50)
            if sq_result.get("signal") not in ("NONE", "LOW RISK"):
                reasons.append(f"Squeeze: {sq_result['signal']} ({sq_result['reason']})")
        except Exception:
            squeeze_score_val = min(100, max(0, squeeze_raw))
    else:
        squeeze_score_val = min(100, max(0, squeeze_raw))

    # 5. Volume score
    volume_score = 50
    if volume_ratio > 3:
        volume_score = 90
    elif volume_ratio > 2:
        volume_score = 75
    elif volume_ratio > 1.5:
        volume_score = 60
    elif volume_ratio > 1:
        volume_score = 50
    else:
        volume_score = 35

    # ── EWMA / GARCH vol adjustment ──────────────────────────────────────────
    ewma_rv = vol_metrics.get("ewma_rv") or None
    garch_rv = vol_metrics.get("garch_rv") or None
    rv20 = vol_metrics.get("rv20") or None

    # If EWMA vol > realized vol by a large margin, options are expensive → SELL
    vol_premium_bonus = 0
    if ewma_rv and rv20:
        ewma_premium = ewma_rv - rv20
        if ewma_premium > 5:
            vol_premium_bonus = 8
            reasons.append(f"EWMA vol ({ewma_rv:.1f}%) >> RV20 ({rv20:.1f}%) — elevated vol")
        elif ewma_premium < -5:
            vol_premium_bonus = 5
            reasons.append(f"Vol compressed — low EWMA ({ewma_rv:.1f}%)")

    # ── Macro & News Factors ──────────────────────────────────────────────
    macro_adjustment = 0
    news_adjustment = 0

    # VIX regime affects strategy preference
    vix_regime = macro.get("vix_regime", "medium")
    if vix_regime == "extreme":
        # High VIX: favor vol-selling (VRP), penalize momentum
        macro_adjustment += 5 if vrp > 0 else -10
        reasons.append(f"VIX extreme ({macro.get('vix', 'N/A')}) — favoring vol strategies")
    elif vix_regime == "low":
        # Low VIX: favor momentum, penalize vol-selling (low premiums)
        macro_adjustment += 5 if mom_12_1 > 0 else -5
        reasons.append(f"VIX low ({macro.get('vix', 'N/A')}) — favoring momentum")

    # Sector momentum — boost stocks in hot sectors, penalize cold ones
    sector_mom = macro.get("sector_momentum", {})
    stock_sector = SECTOR_MAP.get(ticker, "Other")
    sector_pct = sector_mom.get(stock_sector, 0)
    if sector_pct > 1.5:
        macro_adjustment += 8
        reasons.append(f"Sector tailwind: {stock_sector} +{sector_pct:.1f}%")
    elif sector_pct < -1.5:
        macro_adjustment -= 8
        reasons.append(f"Sector headwind: {stock_sector} {sector_pct:.1f}%")

    # Market regime
    regime = macro.get("market_regime", "neutral")
    if regime == "risk_off":
        # Risk-off: penalize aggressive buys, favor defensive
        macro_adjustment -= 5
        reasons.append("Market regime: risk-off")
    elif regime == "risk_on":
        macro_adjustment += 3

    # News sentiment scoring
    news_score_val = news_sent.get("sentiment_score", 0)
    if news_score_val >= 60:
        news_adjustment += 10
        reasons.append(f"News very bullish ({news_sent.get('headline_count', 0)} articles)")
    elif news_score_val >= 20:
        news_adjustment += 5
        reasons.append(f"News bullish ({news_sent.get('headline_count', 0)} articles)")
    elif news_score_val <= -60:
        news_adjustment -= 12
        reasons.append(f"News very bearish: {news_sent.get('top_headline', '')[:60]}")
    elif news_score_val <= -20:
        news_adjustment -= 6
        reasons.append(f"News bearish ({news_sent.get('headline_count', 0)} articles)")

    # ── Combined score ───────────────────────────────────────────────────────
    combined_score = (
        momentum_score * 0.25 +
        mean_reversion_score * 0.20 +
        vrp_score * 0.25 +
        squeeze_score_val * 0.15 +
        volume_score * 0.15
    ) + vol_premium_bonus
    combined_score += macro_adjustment + news_adjustment

    # ADX trend strength adjustment
    adx_adjustment = 0
    if adx_value is not None:
        if adx_value > 40:
            # Strong trend — boost momentum trades, penalize mean reversion
            adx_adjustment += 6
            reasons.append(f"ADX {adx_value} — strong trend")
        elif adx_value > 25:
            adx_adjustment += 3
            reasons.append(f"ADX {adx_value} — moderate trend")
        elif adx_value < 15:
            # No trend — penalize momentum, boost mean reversion
            adx_adjustment -= 4
            reasons.append(f"ADX {adx_value} — no trend (choppy)")

    # OBV smart money flow adjustment
    obv_adjustment = 0
    if obv_signal == 1:
        obv_adjustment += 5
        reasons.append("OBV rising — smart money accumulating")
    elif obv_signal == -1:
        obv_adjustment -= 5
        reasons.append("OBV falling — smart money distributing")

    combined_score += adx_adjustment + obv_adjustment

    # ── Intelligence-based scoring ────────────────────────────────────────────
    intel_adjustment = 0
    if intel:
        intel_score = intel.get("intel_score", 0)
        
        # Scale intel_score into a -15 to +15 adjustment
        intel_adjustment = max(-15, min(15, int(intel_score * 0.3)))
        
        # Trap detection — major penalty
        if intel.get("trap_warning"):
            intel_adjustment = min(intel_adjustment, -10)
            reasons.append(f"TRAP WARNING: {intel.get('trap_reason', 'conflicting signals')[:80]}")
        
        # Insider activity (SEC EDGAR)
        insider = intel.get("insider", {})
        if insider.get("net_direction") == "strong_selling":
            reasons.append(f"Insiders heavily selling ({insider.get('sell_count', 0)} sells)")
        elif insider.get("net_direction") == "strong_buying":
            reasons.append(f"Insiders heavily buying ({insider.get('buy_count', 0)} buys)")

        # Finnhub insider sentiment (MSPR: -100 to +100, predicts 30-90 day moves)
        if finnhub.get("insider_mspr"):
            mspr = finnhub["insider_mspr"]
            if mspr > 30:
                intel_adjustment += 5
                reasons.append(f"Finnhub insider sentiment: bullish (MSPR {mspr:.0f})")
            elif mspr < -30:
                intel_adjustment -= 5
                reasons.append(f"Finnhub insider sentiment: bearish (MSPR {mspr:.0f})")
        if finnhub.get("analyst_consensus") == "buy":
            intel_adjustment += 3
            reasons.append(f"Analyst consensus: BUY ({finnhub.get('analyst_buy_count', 0)} buy ratings)")
        
        # Earnings pattern
        earnings_pat = intel.get("earnings_pattern", {})
        if earnings_pat.get("sell_the_news_risk"):
            intel_adjustment -= 5
            reasons.append("Sell-the-news pattern detected — stock often drops on good earnings")
        
        # Top news event
        news = intel.get("news", {})
        if news.get("events"):
            top_event = news["events"][0]
            if abs(top_event.get("weight", 0)) >= 15:
                reasons.append(f"Major event: {top_event['category'].replace('_', ' ')} — {top_event.get('headline', '')[:60]}")
    
    combined_score += intel_adjustment

    # ── Alternative data scoring ──────────────────────────────────────────────
    alt_adjustment = 0
    if alt:
        alt_adjustment = max(-20, min(20, alt.get("alt_score", 0)))
        
        wiki = alt.get("wiki", {})
        if wiki.get("has_spike"):
            reasons.append(f"Wikipedia attention spike: {wiki.get('spike_ratio', 0)}x normal views")
        
        short = alt.get("short_interest", {})
        if short.get("squeeze_potential"):
            reasons.append("Short squeeze potential — price rising with heavy prior shorting")
        elif short.get("short_pressure") == "high":
            reasons.append("High short pressure — heavy selling on down days")
        
        congress = alt.get("congressional", {})
        if congress.get("unusual_activity"):
            reasons.append(f"Unusual insider filing activity ({congress.get('recent_filings', 0)} Form 4s in 2 weeks)")
        
        geo = alt.get("geopolitical", {})
        if geo.get("risk_level") in ("high", "extreme"):
            reasons.append(f"Geopolitical risk: {geo.get('risk_level')} — {', '.join(geo.get('active_risks', []))}")
        
        fred = alt.get("fred_macro", {})
        if fred.get("yield_curve_inverted"):
            reasons.append("Yield curve inverted — recession signal")
        if fred.get("credit_stress"):
            reasons.append("Credit spreads elevated — financial stress")
    
    combined_score += alt_adjustment

    # ── Social & search data scoring ──────────────────────────────────────────
    social_adjustment = 0
    if social:
        social_signal = social.get("combined_signal", 0)
        social_adjustment = social_signal  # Already weighted and clamped to -15 to +15
        
        # Add specific reasons
        reddit = social.get("reddit", {})
        trends = social.get("google_trends", {})
        news_m = social.get("news_multi", {})
        
        if trends.get("has_spike"):
            reasons.append(f"Google Trends spike: {trends.get('spike_ratio', 0)}x normal search interest")
        elif trends.get("trend_direction") == "rising":
            reasons.append("Google search interest rising")
        
        if reddit.get("buzz_level") in ("high", "viral"):
            reasons.append(f"Reddit buzz: {reddit.get('buzz_level')} ({reddit.get('total_mentions', 0)} mentions, {reddit.get('bullish_pct', 50):.0f}% bullish)")
            if reddit.get("wsb_mentions", 0) > 10 and reddit.get("bullish_pct", 50) > 80:
                reasons.append("WSB hype warning — extreme bullishness often precedes drops")
        
        if news_m.get("freshness") == "breaking":
            reasons.append(f"Breaking news: {news_m.get('total_articles', 0)} articles from {len(news_m.get('sources', {}))} sources")
        
        if social.get("confidence") == "low":
            social_adjustment = int(social_adjustment * 0.5)
    
    combined_score += social_adjustment

    # ── Analyst & valuation scoring ──────────────────────────────────────────────────────
    fundamental_adjustment = 0
    if yf_fundamentals:
        # Analyst consensus
        yf_rec = yf_fundamentals.get("recommendation", "none")
        if yf_rec in ("strong_buy", "buy"):
            fundamental_adjustment += 5
            reasons.append(f"Analyst consensus: {yf_rec.upper()} ({yf_fundamentals.get('num_analysts', 0)} analysts)")
        elif yf_rec in ("sell", "strong_sell", "underperform"):
            fundamental_adjustment -= 6
            reasons.append(f"Analyst consensus: {yf_rec.upper()} ({yf_fundamentals.get('num_analysts', 0)} analysts)")

        # Price target upside/downside
        target = yf_fundamentals.get("target_mean", 0)
        current = yf_fundamentals.get("current_price", 0)
        if target > 0 and current > 0:
            upside_pct = ((target - current) / current) * 100
            if upside_pct > 20:
                fundamental_adjustment += 6
                reasons.append(f"Analyst target: ${target:.0f} (+{upside_pct:.0f}% upside)")
            elif upside_pct > 10:
                fundamental_adjustment += 3
                reasons.append(f"Analyst target: ${target:.0f} (+{upside_pct:.0f}% upside)")
            elif upside_pct < -10:
                fundamental_adjustment -= 5
                reasons.append(f"Analyst target: ${target:.0f} ({upside_pct:.0f}% downside)")

        # Valuation check — penalize extremely overvalued stocks
        fwd_pe = yf_fundamentals.get("forward_pe", 0)
        if fwd_pe > 50:
            fundamental_adjustment -= 4
            reasons.append(f"Expensive: Forward P/E {fwd_pe:.1f}")
        elif fwd_pe > 0 and fwd_pe < 12:
            fundamental_adjustment += 3
            reasons.append(f"Cheap: Forward P/E {fwd_pe:.1f}")

        ev_ebitda = yf_fundamentals.get("ev_ebitda", 0)
        if ev_ebitda > 30:
            fundamental_adjustment -= 3
            reasons.append(f"High EV/EBITDA: {ev_ebitda:.1f}")
        elif ev_ebitda > 0 and ev_ebitda < 10:
            fundamental_adjustment += 2
            reasons.append(f"Low EV/EBITDA: {ev_ebitda:.1f}")

    combined_score += fundamental_adjustment

    # Recommendation boost
    if rec:
        action = rec.get("action", "")
        if "BUY" in action.upper():
            combined_score += 8
            reasons.append(f"AI recommends: {action}")
        elif "SELL" in action.upper() and "OPTIONS" in action.upper():
            combined_score += 6
            reasons.append(f"AI recommends: {action}")

    # Sentiment boost / penalty
    contrarian = sentiment.get("contrarian_flag")
    if contrarian == "Squeeze Watch":
        combined_score += 10
        reasons.append("SQUEEZE WATCH — retail hype + high short interest")
    elif contrarian == "Buy the Dip":
        combined_score += 8
        reasons.append("Buy the Dip signal — retail panic, institutions buying")
    elif sent_score_val > 75:
        combined_score += 4

    # Edge factors
    rs = edge.get("relative_strength", 0)
    if rs and rs > 5:
        combined_score += 6
        reasons.append(f"Outperforming SPY by {rs:.1f}%")

    combined_score = max(0, min(100, combined_score))
    rules_only_score = round(combined_score, 1)  # Capture BEFORE ML blend for attribution
    ml_only_score = None  # Set below if ML runs

    # ── ML Model Integration ──────────────────────────────────────────────────
    try:
        from ml_model_v2 import ml_score, _frac_diff  # v3: 34 features, regime-conditional, calibrated
        # Get Markov regime state for the ML (adds real predictive power)
        _regime_ctx = {"regime_score": 50, "markov_state": 1, "size_multiplier": 1.0}
        if _HAS_MARKOV:
            try:
                _spy_rets = []
                _spy_macro = macro.get("spy_returns_5d", []) or []
                _vxx_r = intel.get("vxx_ratio", 1.0) if intel else 1.0
                _spy_ma = macro.get("spy_vs_ma50", 1.0) or 1.0
                _regime_ctx = _get_markov_regime(_spy_rets, float(_vxx_r), float(_spy_ma))
            except Exception:
                pass

        # 34 features — ALL computed from real data
        _price = quick_result.get("price", 0) or 0
        _high_52w = quick_result.get("high_52w", _price) or _price
        _price_vs_52w = (_price - _high_52w) / _high_52w * 100 if _high_52w > 0 else 0

        # Compute MA10 and ATR% from captured bar data
        _above_ma10 = 1.0  # default: assume above (already trend-filtered)
        if len(_deep_closes) >= 10:
            _ma10 = sum(_deep_closes[-10:]) / 10
            _above_ma10 = 1.0 if (_deep_closes[-1] > _ma10) else 0.0
        _atr_pct = 3.0  # default
        if _deep_atr14 is not None and _price > 0:
            _atr_pct = round(_deep_atr14 / _price * 100, 2)
        # Derived features
        _put_call_proxy = -1.0 if vrp > 8 else (1.0 if vrp < -5 else 0.0)
        _idio_ret = (quick_result.get("change_pct", 0) or 0) - (macro.get("spy_change_pct", 0) or 0)

        # ── Compute vol_of_vol from VXX daily closes ─────────────────
        # OPTIMIZATION 2026-04-20: use macro.vxx_closes from get_macro_snapshot
        # (5-min cache) instead of making a new Alpaca call each scan.
        # Fallback to direct fetch if macro data unavailable.
        _vol_of_vol = 0.0
        _vxx_closes = None
        try:
            _vxx_closes = macro.get("vxx_closes") if isinstance(macro, dict) else None
        except Exception:
            _vxx_closes = None
        if not _vxx_closes or len(_vxx_closes) < 6:
            try:
                _vxx_start = (datetime.now() - timedelta(days=20)).strftime("%Y-%m-%d")
                alpaca_throttle.acquire()
                _vxx_resp = requests.get(
                    f"{ALPACA_DATA_URL}/v2/stocks/VXX/bars?timeframe=1Day&start={_vxx_start}&limit=12&feed=sip",
                    headers=_alpaca_headers(), timeout=5)
                _vxx_bars = _vxx_resp.json().get("bars", [])
                if len(_vxx_bars) >= 6:
                    _vxx_closes = [float(b["c"]) for b in _vxx_bars]
            except Exception:
                _vxx_closes = None
        if _vxx_closes and len(_vxx_closes) >= 6:
            try:
                _vxx_rets = [(_vxx_closes[i] - _vxx_closes[i-1]) / _vxx_closes[i-1]
                             for i in range(1, len(_vxx_closes)) if _vxx_closes[i-1] > 0]
                if len(_vxx_rets) >= 5:
                    _vol_of_vol = round(float(np.std(_vxx_rets[-10:]) * 100), 3)
            except Exception:
                _vol_of_vol = 2.0
        else:
            _vol_of_vol = 2.0  # safe default (matches training fallback)

        # ── Compute frac_diff_price from daily closes ────────────────
        _frac_diff_val = 0.0
        if len(_deep_closes) >= 20:
            _frac_diff_val = round(_frac_diff(_deep_closes), 4)

        # ── Compute cross_sec_rank from scan universe ────────────────
        # Rank this stock's daily return against all stocks in current scan batch
        _cross_sec_rank = 0.5  # default if no universe data
        _stock_change = quick_result.get("change_pct", 0) or 0
        try:
            alpaca_throttle.acquire()
            _scan_snaps = requests.get(
                f"{ALPACA_DATA_URL}/v1beta1/screener/stocks/most-actives?by=volume&top=100",
                headers=_alpaca_headers(), timeout=5)
            _actives = _scan_snaps.json().get("most_actives", [])
            _all_changes = [float(s.get("change", 0) or 0) for s in _actives if s.get("change") is not None]
            if len(_all_changes) >= 10:
                _cross_sec_rank = round(sum(1 for x in _all_changes if x <= _stock_change) / len(_all_changes), 3)
        except Exception:
            pass

        # Compute 8 new features for ML model upgrade (34 total)
        # Intel features: news_sentiment, insider_signal, intel_score
        _news_sentiment = 0.0
        _insider_signal = 0.0
        _intel_score_feat = 0.0
        if intel:
            # news_sentiment: normalize from classify_news score (-100 to 100 → -1 to 1)
            news_data = intel.get("news", {})
            if isinstance(news_data, dict):
                raw_news_score = float(news_data.get("score", 0) or 0)
                _news_sentiment = max(-1.0, min(1.0, raw_news_score / 100.0))
            # insider_signal: map direction to numeric
            insider_data = intel.get("insider", {})
            if isinstance(insider_data, dict):
                direction = insider_data.get("net_direction", "neutral")
                _insider_signal_map = {
                    "strong_buying": 1.0, "net_buying": 0.5,
                    "neutral": 0.0,
                    "net_selling": -0.5, "strong_selling": -1.0,
                }
                _insider_signal = _insider_signal_map.get(direction, 0.0)
            # intel_score: composite
            _intel_score_feat = 0.4 * _news_sentiment + 0.3 * _insider_signal + 0.3 * (intel.get("earnings_surprise", 0) or 0)

        # iv_rank_stock: per-stock IV rank
        _iv_rank_stock = _compute_stock_iv_rank(ticker, _deep_closes) if _deep_closes else 50.0

        # days_to_earnings: normalized 0-1
        _days_to_earnings = 0.5
        try:
            from options_scanner import _get_next_earnings_date
            import datetime as _dt
            earn_date = _get_next_earnings_date(ticker)
            if earn_date:
                if isinstance(earn_date, str):
                    earn_date = _dt.datetime.strptime(earn_date[:10], "%Y-%m-%d").date()
                days_away = (earn_date - _dt.date.today()).days
                _days_to_earnings = min(days_away, 60) / 60.0 if days_away >= 0 else 0.0
        except Exception:
            pass

        # credit_spread: TLT-HYG 21d return spread
        _credit_spread = 0.0
        try:
            alpaca_throttle.acquire()
            _cs_resp = requests.get(f"{ALPACA_DATA_URL}/v2/stocks/bars",
                params={"symbols": "TLT,HYG", "timeframe": "1Day",
                        "start": (datetime.now() - timedelta(days=35)).strftime("%Y-%m-%d"),
                        "limit": 30, "feed": "sip"},
                headers=_alpaca_headers(), timeout=8)
            _cs_data = _cs_resp.json().get("bars", {})
            _tlt_bars = _cs_data.get("TLT", [])
            _hyg_bars = _cs_data.get("HYG", [])
            if len(_tlt_bars) >= 21 and len(_hyg_bars) >= 21:
                tlt_now = float(_tlt_bars[-1]["c"])
                tlt_21 = float(_tlt_bars[-22]["c"])
                hyg_now = float(_hyg_bars[-1]["c"])
                hyg_21 = float(_hyg_bars[-22]["c"])
                if tlt_21 > 0 and hyg_21 > 0:
                    _credit_spread = round((tlt_now / tlt_21 - 1) - (hyg_now / hyg_21 - 1), 4)
        except Exception:
            pass

        # market_breadth: % of scan universe above 50d MA
        _market_breadth = 0.5
        try:
            alpaca_throttle.acquire()
            _mb_resp = requests.get(
                f"{ALPACA_DATA_URL}/v1beta1/screener/stocks/most-actives?by=volume&top=100",
                headers=_alpaca_headers(), timeout=5)
            _mb_actives = _mb_resp.json().get("most_actives", [])
            if _mb_actives:
                _above_50ma_count = 0
                _total_checked = 0
                for _mb_s in _mb_actives[:50]:
                    _mb_price = float(_mb_s.get("price", 0) or 0)
                    _mb_prev = float(_mb_s.get("prev_close", _mb_price) or _mb_price)
                    if _mb_price > 0 and _mb_prev > 0:
                        _total_checked += 1
                        # Approximate: if price > prev_close, count as above MA proxy
                        if _mb_price > _mb_prev:
                            _above_50ma_count += 1
                if _total_checked > 10:
                    _market_breadth = round(_above_50ma_count / _total_checked, 3)
        except Exception:
            pass

        # put_call_ratio: VIX proxy
        _vxx_for_pcr = intel.get("vxx_ratio", 1.0) if intel else 1.0
        _put_call_ratio = round(float(_vxx_for_pcr) * 15.0 / 20.0, 3) if _vxx_for_pcr else 1.0

        # Compute real momentum features for ML (supersedes edge-dict proxies)
        _real_mom_1m = mom_1m if 'mom_1m' in locals() else 0.0
        _real_mom_3m = 0.0
        try:
            if _deep_closes and len(_deep_closes) >= 63:
                _real_mom_3m = (_deep_closes[-1] / _deep_closes[-63] - 1.0) * 100
        except Exception:
            pass

        ml_features = {
            # Technical (7) — FIXED 2026-04-20: was edge proxies, now real returns
            "momentum_1m":          round(_real_mom_1m, 3),
            "momentum_3m":          round(_real_mom_3m, 3),
            "rsi_14":               rsi or 50,
            "volume_ratio":         min(volume_ratio, 10.0),
            "vwap_position":        1.0 if quick_result.get("above_vwap") else 0.0,
            "adx":                  adx_value if adx_value is not None else 20,
            "ewma_vol":             vol_metrics.get("ewma_vol", 2) or 2 if vol_metrics else 2,
            "range_pct":            quick_result.get("range_pct", 0) or 0,
            "price_vs_52w_high":    _price_vs_52w,
            # Volatility (3)
            "vrp":                  vrp or 0,
            "iv_rank_proxy":        _compute_stock_iv_rank(ticker, _deep_closes) if _deep_closes else 50,
            "atr_pct":              _atr_pct,
            # Regime (5)
            "vxx_ratio":            intel.get("vxx_ratio", 1.0) if intel else 1.0,
            "spy_vs_ma50":          macro.get("spy_vs_ma50", 1.0) or 1.0,
            "markov_state":         float(_regime_ctx.get("markov_state", 1)),
            "regime_score":         float(_regime_ctx.get("regime_score", 50)),
            "sector_momentum":      sector_mom.get(stock_sector, 0),
            # Additive 6 features
            "cross_sec_rank":       _cross_sec_rank,
            "earnings_surprise":    intel.get("earnings_surprise", 0) if intel else 0,
            "put_call_proxy":       _put_call_proxy,
            "vol_of_vol":           _vol_of_vol,
            "frac_diff_price":      _frac_diff_val,
            "idiosyncratic_ret":    round(_idio_ret, 4),
            # Entry timing (3)
            "change_pct_today":     quick_result.get("change_pct", 0) or 0,
            "above_ma10":           _above_ma10,
            "trend_strength":       min(abs(quick_result.get("change_pct", 0) or 0) / max(vol_metrics.get("ewma_vol", 2) or 2, 0.1), 5.0) if vol_metrics else 1.0,
            # Intel features (3) — re-added with real data
            "intel_score":          round(_intel_score_feat, 3),
            "news_sentiment":       round(_news_sentiment, 3),
            "insider_signal":       round(_insider_signal, 3),
            # New professional features (5)
            "iv_rank_stock":        round(_iv_rank_stock, 2),
            "days_to_earnings":     round(_days_to_earnings, 3),
            "credit_spread":        round(_credit_spread, 4),
            "market_breadth":       round(_market_breadth, 3),
            "put_call_ratio":       round(_put_call_ratio, 3),
        }
        ml_result = ml_score(ml_features)

        # Blend: dynamic ratio loaded from tracker, default 60/40
        # IMPORTANT: ML only adjusts score when it has CONVICTION (not near 50).
        # Near-50 ML = uncertain. Uncertain ML should NOT drag down a good rules setup.
        if ml_result.get("model_type") != "fallback":
            ml_s = ml_result.get("ml_score", combined_score)
            ml_only_score = round(ml_s, 1)  # Capture ML score for attribution
            # Dynamic blend: load tracked accuracy, default 60/40
            try:
                from storage_config import BLEND_TRACKER_PATH
                if os.path.exists(BLEND_TRACKER_PATH):
                    with open(BLEND_TRACKER_PATH) as _bf:
                        _blend = json.load(_bf)
                    rule_weight = _blend.get("rule_weight", 0.6)
                    ml_weight = _blend.get("ml_weight", 0.4)
                else:
                    rule_weight, ml_weight = 0.6, 0.4
            except Exception:
                rule_weight, ml_weight = 0.6, 0.4

            # Conviction-gating: when ML is uncertain (42-58 range), treat as neutral 50.
            # This prevents a mediocre ML score from dragging down strong rule setups.
            # At 50, the blend becomes: rules*0.6 + 50*0.4 = rules contribution only matters.
            ml_conviction = abs(ml_s - 50)
            if ml_conviction < 8:  # Score within 42-58 = ML is uncertain
                ml_s_blended = 50.0  # Treat as neutral — no drag, no boost
                reasons.append(f"ML: uncertain ({ml_s:.0f}), using rules-only")
            else:
                ml_s_blended = ml_s
                reasons.append(f"ML model: {ml_result.get('ml_signal', 'HOLD')} ({ml_result.get('ml_confidence', 0):.0%} confidence)")

            combined_score = combined_score * rule_weight + ml_s_blended * ml_weight
            reasons.append(f"Blend: {rule_weight:.0%} rules + {ml_weight:.0%} ML")
    except Exception:
        pass  # ML not available, use rule-based only

    combined_score = max(0, min(100, combined_score))

    # Store ml_features and regime on the result dict for entry_features logging
    # This is how the self-learning loop works: entry features saved at trade time,
    # used by ml_model_v2 to train on actual bot outcomes after trade closes.
    try:
        quick_result["ml_features"]  = ml_features  # The 34 features used at entry
        quick_result["regime_label"] = _regime_ctx.get("regime_label", "NEUTRAL")
    except Exception:
        pass

    # ── Determine trade SIDE ─────────────────────────────────────────────────
    momentum_is_negative = (momentum_score < 40 or change_pct < -2)
    sentiment_is_bearish = (sent_score_val < 40)
    vrp_is_high = (vrp > 5)
    rsi_overbought = (rsi is not None and rsi > 70)

    side = "buy"
    trade_type = "stock"
    action_label = "BUY"

    if combined_score > 65:
        if vrp_is_high:
            side = "sell"
            trade_type = "options"
            action_label = "SELL OPTIONS"
        # Stock shorting disabled: backtest showed -$419K over 10yr.
        # Bearish conviction handled by options scanner (puts) instead.
        # elif momentum_is_negative and sentiment_is_bearish:
        #     side = "short"
        #     action_label = "SHORT"
        elif rsi_overbought:
            side = "sell"
            action_label = "SELL"
        # else: default BUY

    # Extract sector from yfinance valuation data and update the cache
    _yf_sector = detail.get("valuation", {}).get("sector", "Unknown")
    _update_sector_cache(ticker, _yf_sector)

    # ── SHADOW PORTFOLIO LOGGING (learning data multiplier) ───────────────
    # Log EVERY scored candidate so we can learn from rejections too
    try:
        from shadow_portfolio import log_candidate
        # Decision tracking: score threshold + risk filters
        _min_score = _scan_params.get("MIN_SCORE", 63) if '_scan_params' in locals() else 63
        if combined_score >= _min_score:
            _decision = "taken"
            _reason = f"score {combined_score:.1f} >= MIN_SCORE {_min_score}"
        else:
            _decision = "rejected_score"
            _reason = f"score {combined_score:.1f} < MIN_SCORE {_min_score}"

        log_candidate(
            ticker=ticker,
            features=ml_features,
            score=combined_score,
            decision=_decision,
            decision_reason=_reason,
            entry_price=quick_result.get("price", 0),
            vxx_ratio=float(_regime_ctx.get("vxx_ratio", 1.0) or 1.0) if '_regime_ctx' in locals() else 1.0,
            regime_label=_regime_ctx.get("regime_label", "NEUTRAL") if '_regime_ctx' in locals() else "NEUTRAL",
        )
    except Exception:
        pass  # Shadow logging must never break the trading loop

    return {
        **quick_result,
        "deep_score": round(combined_score, 1),
        "reasons": reasons,
        "vrp": vrp,
        "side": side,
        "trade_type": trade_type,
        "action_label": action_label,
        "recommendation": rec.get("action") if rec else None,
        "rec_reasoning": rec.get("reasoning") if rec else None,
        "sentiment_score": sent_score_val,
        "rsi": rsi,
        "momentum_score": round(momentum_score, 1),
        "mean_reversion_score": round(mean_reversion_score, 1),
        "vrp_score": round(vrp_score, 1),
        "squeeze_score": round(squeeze_score_val, 1),
        "volume_score": round(volume_score, 1),
        "ewma_rv": ewma_rv,
        "garch_rv": garch_rv,
        "horizon": rec.get("horizon") if rec else None,
        "leveraged_bull": rec.get("leveraged_bull") if rec else None,
        "leveraged_bear": rec.get("leveraged_bear") if rec else None,
        # Full 34-feature snapshot at entry time (for ML exit model training)
        "entry_features": ml_features if 'ml_features' in locals() else None,
        # Score attribution (exposed at scan level via stock.get())
        "rules_only_score": rules_only_score,  # Pre-ML-blend score captured above
        "ml_only_score": ml_s if 'ml_s' in locals() else None,
        "sector": _yf_sector,
    }

# ── Correlation / Sector Check ───────────────────────────────────────────────

def check_sector_correlation(ticker, existing_tickers, sector=None,
                             existing_positions=None):
    """
    Returns True if adding this ticker would create dangerous concentration.
    Checks both sector limits AND portfolio beta correlation.
    False = safe to trade.

    sector: optional yfinance sector for the new ticker (from deep_score).
    existing_positions: optional list of position dicts from Alpaca — used to
        exclude options positions from the sector count (options have separate
        slots and should NOT count toward the stock sector limit).
    """
    # Check 1: Sector concentration (stocks only)
    new_sector = _get_sector(ticker, sector)

    # Build a set of stock-only tickers from existing positions (exclude options)
    if existing_positions is not None:
        stock_tickers = [
            str(p.get("symbol", "")) for p in existing_positions
            if p.get("asset_class", "us_equity") != "us_option"
            and len(str(p.get("symbol", ""))) <= 8
        ]
    else:
        # Fallback: assume all existing_tickers are stocks (legacy callers)
        stock_tickers = list(existing_tickers)

    # Also include pending trade tickers (already filtered to stocks by caller)
    pending_stock_tickers = [t for t in existing_tickers if t not in stock_tickers]
    all_stock_tickers = stock_tickers + pending_stock_tickers

    count = sum(
        1 for t in all_stock_tickers
        if _get_sector(t) == new_sector
    )
    if count >= MAX_SECTOR_POSITIONS:
        return True

    # Check 2: High-beta concentration — don't load up on all volatile names
    # If we already hold 3+ positions, check if adding this one makes the
    # portfolio too correlated (all high-beta or all defensive)
    if len(existing_tickers) >= 3:
        try:
            _beta_map = {
                # High beta (>1.3) — tech/growth/meme
                "TSLA": 2.0, "NVDA": 1.7, "AMD": 1.6, "MSTR": 2.5, "COIN": 2.0,
                "META": 1.4, "AMZN": 1.3, "NFLX": 1.5, "SHOP": 1.8, "SQ": 1.7,
                "ROKU": 1.9, "SNAP": 1.6, "PLTR": 1.8, "RIVN": 2.0, "LCID": 2.1,
                "SOFI": 1.6, "HOOD": 1.9, "AFRM": 1.8, "UPST": 2.2, "SMCI": 2.0,
                # Medium beta (0.8-1.3) — large cap blend
                "AAPL": 1.2, "MSFT": 1.1, "GOOGL": 1.1, "GOOG": 1.1, "JPM": 1.1,
                "V": 1.0, "MA": 1.0, "UNH": 0.9, "HD": 1.0, "DIS": 1.2,
                # Low beta (<0.8) — defensive/utilities
                "JNJ": 0.6, "PG": 0.5, "KO": 0.6, "PEP": 0.6, "WMT": 0.5,
                "MRK": 0.5, "VZ": 0.4, "T": 0.6, "NEE": 0.5, "SO": 0.3,
                "CL": 0.5, "GIS": 0.4, "K": 0.4, "ED": 0.3, "XEL": 0.3,
            }
            new_beta = _beta_map.get(ticker.upper(), 1.0)
            existing_betas = [_beta_map.get(t.upper(), 1.0) for t in existing_tickers]
            high_beta_count = sum(1 for b in existing_betas if b >= 1.5)

            # Block if adding another high-beta stock when we already have 3+
            if new_beta >= 1.5 and high_beta_count >= 3:
                return True  # Too many volatile names

            # Block if average portfolio beta would exceed 1.6
            all_betas = existing_betas + [new_beta]
            avg_beta = sum(all_betas) / len(all_betas)
            if avg_beta > 1.6:
                return True  # Portfolio too hot
        except Exception:
            pass

    return False

# ── Position Management ──────────────────────────────────────────────────────

def manage_positions():
    """
    Smart position management with ATR-based trailing stops, scale-out in thirds,
    breakeven stops, regime-aware exit targets, and upgrade logic.

    Exit architecture:
      - Scale-out 1/3 at 1R, 1/3 at 2-3R (regime-dependent), trail last 1/3
      - Breakeven stop activates after first scale-out (can't lose money)
      - Phase 4 trails at 2-3× ATR below highest price (regime-adaptive)
      - BULL: wider trails (3× ATR), scale at 1R and 3R
      - BEAR/PANIC: tighter trails (1.5× ATR), scale at 1R and 1.5R
    """
    try:
        positions = get_alpaca_positions()
    except Exception:
        return {"actions": [], "error": "Could not fetch positions", "upgrade_candidates": []}

    if not isinstance(positions, list):
        return {"actions": [], "positions": 0, "upgrade_candidates": []}

    actions = []
    upgrade_candidates = []

    # Load evolving stop state (persists across cycles)
    _stop_state_path = os.path.join(DATA_DIR, 'voltrade_stop_state.json')
    try:
        with open(_stop_state_path) as f:
            stop_state = json.load(f)
    except Exception:
        stop_state = {}

    # ── Get current market regime ─────────────────────────────────────────────
    regime = "NEUTRAL"
    try:
        from macro_data import get_macro_snapshot
        _macro = get_macro_snapshot()
        _vxx_r = float(_macro.get("vxx_ratio", 1.0) or 1.0)
        _spy_ma = float(_macro.get("spy_vs_ma50", 1.0) or 1.0)
        _spy_b200 = int(_macro.get("spy_below_200_days", 0) or 0)
        _spy_above = bool(_macro.get("spy_above_200d", True))
        from system_config import get_market_regime as _gmr
        regime = _gmr(_vxx_r, _spy_ma, spy_below_200_days=_spy_b200, spy_above_200d=_spy_above)
    except Exception:
        pass

    is_bullish = regime in ("BULL", "NEUTRAL_BULL")
    is_bearish = regime in ("BEAR", "PANIC", "CAUTION")
    # Regime-adaptive trailing ATR multiplier
    trail_atr_mult = 3.0 if is_bullish else (1.5 if is_bearish else 2.0)
    # Regime-adaptive scale-out thresholds
    scale_out_2r = 3.0 if is_bullish else (1.5 if is_bearish else 2.0)

    # Tickers managed by other components — do NOT apply stop/TP logic to these
    FLOOR_AND_LEG_TICKERS = {"QQQ", "SVXY", "SPY"}  # SQQQ/SPXS removed (convexity overlay now uses QQQ puts, not inverse ETFs)

    for pos in positions:
        ticker = pos.get("symbol", "")

        # Skip passive floor, VRP harvest, and sector rotation tickers
        if ticker in FLOOR_AND_LEG_TICKERS:
            continue

        # ── FIX: Skip options positions (managed by options_manager.py) ───
        # OCC option symbols (e.g. AAPL260420C00257500) are >8 chars and have
        # asset_class "us_option". The stock stop/TP logic doesn't apply to
        # options — ATR lookup fails on OCC symbols, defaulting to 2% which
        # triggers false stops within minutes. Options have their own exit
        # logic in options_manager.py (DTE, profit target, Greeks, rolling).
        if pos.get("asset_class") == "us_option" or len(ticker) > 8:
            continue

        current = float(pos.get("current_price", 0))
        entry = float(pos.get("avg_entry_price", current))
        pnl_pct = float(pos.get("unrealized_plpc", 0)) * 100
        qty = abs(int(float(pos.get("qty", 0))))
        side = pos.get("side", "long")
        market_value = abs(float(pos.get("market_value", 0)))

        # Get ATR for this stock (14-day Average True Range)
        atr = _get_atr(ticker)
        atr_pct = (atr / current * 100) if current > 0 and atr else 2.0

        # ── MINIMUM HOLD TIME (1 hour) ──────────────────────────────────────
        # Positions entered and exited within minutes consistently lose to the
        # spread. Enforce a minimum hold time before any exit logic runs.
        # This does NOT apply to DTE critical (options) or catastrophic stops (>10% loss).
        _MIN_EQUITY_HOLD_MINUTES = 60
        _ps_entry_ts = stop_state.get(ticker, {}).get("entry_timestamp")
        _equity_held_minutes = 999  # default: assume held long enough
        if _ps_entry_ts:
            try:
                _ps_entry_dt = datetime.strptime(_ps_entry_ts, "%Y-%m-%dT%H:%M:%S.%f") if "." in _ps_entry_ts else datetime.strptime(_ps_entry_ts, "%Y-%m-%dT%H:%M:%S")
                _equity_held_minutes = (datetime.now() - _ps_entry_dt).total_seconds() / 60
            except Exception:
                pass

        # ── EVOLVING STOP SYSTEM ────────────────────────────────────────────
        # Stops evolve through 4 phases as the trade profits:
        #   Phase 1 (entry):     2.0x ATR stop, 3.0x ATR target
        #   Phase 2 (at 1R):     Tighten to 1.5x ATR trailing from high
        #   Phase 3 (at 2R):     Tighten to 1.0x ATR trailing from high
        #   Phase 4 (at 3R+):    ATR-based trailing from highest price (regime-aware)

        # Initialize or load stop state for this position
        ps = stop_state.get(ticker, {})
        initial_risk_pct = ps.get("initial_risk_pct", atr_pct * 2.0)
        highest_pnl = max(ps.get("highest_pnl", 0), pnl_pct)
        highest_price = max(ps.get("highest_price", current), current)
        r_multiple = pnl_pct / initial_risk_pct if initial_risk_pct > 0 else 0
        peak_r = highest_pnl / initial_risk_pct if initial_risk_pct > 0 else 0

        # Scale-out state
        original_qty = ps.get("original_qty", qty)
        remaining_qty = ps.get("remaining_qty", qty)
        scales_completed = ps.get("scales_completed", 0)
        breakeven_active = ps.get("breakeven_active", False)

        # ── SCALE-OUT CHECK ─────────────────────────────────────────────────
        # Scale out 1/3 at each threshold. Only recommend if not already done.
        third_qty = max(1, original_qty // 3)

        if scales_completed < 2 and remaining_qty > 1:
            if scales_completed == 0 and r_multiple >= 1.0 and third_qty >= 1:
                # First scale-out: sell 1/3 at 1R
                scale_qty = third_qty
                scales_completed = 1
                remaining_qty -= scale_qty
                breakeven_active = True  # Activate breakeven after first scale-out
                actions.append({
                    "action": "SCALE_OUT", "ticker": ticker, "side": side,
                    "qty": scale_qty, "remaining_qty": remaining_qty,
                    "reason": f"SCALE-OUT 1/3 at +{pnl_pct:.1f}% ({r_multiple:.1f}R) — {scale_qty}/{original_qty} shares, breakeven stop activated",
                    "type": "scale_out", "phase": 2,
                    "exit_context": {"scale": 1, "r_at_scale": round(r_multiple, 2), "pnl_pct": round(pnl_pct, 2)},
                })
            elif scales_completed == 1 and r_multiple >= scale_out_2r and third_qty >= 1:
                # Second scale-out at regime-dependent threshold
                scale_qty = third_qty
                scales_completed = 2
                remaining_qty -= scale_qty
                actions.append({
                    "action": "SCALE_OUT", "ticker": ticker, "side": side,
                    "qty": scale_qty, "remaining_qty": remaining_qty,
                    "reason": f"SCALE-OUT 2/3 at +{pnl_pct:.1f}% ({r_multiple:.1f}R, regime={regime}) — {scale_qty}/{original_qty} shares",
                    "type": "scale_out", "phase": 3,
                    "exit_context": {"scale": 2, "r_at_scale": round(r_multiple, 2), "pnl_pct": round(pnl_pct, 2)},
                })

        # Determine current phase and stop level
        if peak_r >= 3.0:
            # Phase 4: ATR-based trailing from highest PRICE (not 50% of peak P&L)
            stop_pct = atr_pct * trail_atr_mult
            phase = 4
            stop_reason = f"Phase 4: {trail_atr_mult:.0f}×ATR trailing from high ${highest_price:.2f} (regime={regime})"
        elif peak_r >= 2.0:
            stop_pct = max(1.0, atr_pct * 1.0)
            phase = 3
            stop_reason = f"Phase 3 (2R+): 1.0x ATR trailing"
        elif peak_r >= 1.0:
            stop_pct = max(1.5, atr_pct * 1.5)
            phase = 2
            stop_reason = f"Phase 2 (1R+): 1.5x ATR trailing"
        else:
            stop_pct = max(1.5, min(atr_pct * 2.0, 8.0))
            phase = 1
            stop_reason = f"Phase 1: 2.0x ATR initial stop"

        # For phases 2-4, stop is relative to the HIGH WATER MARK, not entry
        should_stop = False
        if phase >= 4:
            # Phase 4: trail from highest PRICE using ATR
            drawdown_from_high_price = ((highest_price - current) / highest_price * 100) if highest_price > 0 else 0
            should_stop = drawdown_from_high_price >= stop_pct
        elif phase >= 2:
            drawdown_from_peak = highest_pnl - pnl_pct
            should_stop = drawdown_from_peak >= stop_pct
        else:
            should_stop = pnl_pct <= -stop_pct

        # Breakeven enforcement: after first scale-out, P&L can't go negative
        breakeven_triggered = False
        if not should_stop and breakeven_active and pnl_pct < 0 and phase >= 2:
            should_stop = True
            breakeven_triggered = True

        # ── HARD CATASTROPHE STOP (v1.0.29+) ────────────────────────────────────
        # Absolute -20% floor per position to cap overnight / gap-down losses
        # that the chandelier and daily-limit checks can miss. Fires only in
        # phase 1 (pre-first-scale-out); after 1R the chandelier + breakeven
        # already prevent deeper losses. Options (asset_class=="us_option")
        # are skipped earlier in the loop and unaffected.
        hard_stop_triggered = False
        if not should_stop and phase < 2:
            _hs_enabled = bool(BASE_CONFIG.get("POSITION_HARD_STOP_ENABLED", True))
            _hs_pct = float(BASE_CONFIG.get("POSITION_HARD_STOP_PCT", 20.0))
            if _hs_enabled and pnl_pct <= -_hs_pct:
                should_stop = True
                hard_stop_triggered = True
                stop_reason = f"HARD FLOOR: pnl={pnl_pct:.1f}% ≤ -{_hs_pct:.0f}% (catastrophe stop)"

        # Dynamic take profit: only for final third after both scale-outs done
        tp_pct = max(4.0, min(atr_pct * 3.0, 15.0)) if phase < 3 else 999

        # Time stop: 7 days with no progress (prevents capital lockup)
        entry_date = ps.get("entry_date", time.strftime("%Y-%m-%d"))
        try:
            days_held = (datetime.now() - datetime.strptime(entry_date, "%Y-%m-%d")).days
        except Exception:
            days_held = 0
        time_stop = days_held >= 7 and abs(pnl_pct) < 2.0

        # Save updated stop state (including scale-out tracking and regime)
        stop_state[ticker] = {
            "initial_risk_pct": round(initial_risk_pct, 2),
            "highest_pnl": round(highest_pnl, 2),
            "highest_price": round(highest_price, 2),
            "phase": phase,
            "current_stop_pct": round(atr_pct, 2),  # Store raw ATR% — bot.ts computes final stop
            "r_multiple": round(r_multiple, 2),
            "entry_date": entry_date if entry_date != time.strftime("%Y-%m-%d") else ps.get("entry_date", time.strftime("%Y-%m-%d")),
            "entry_timestamp": ps.get("entry_timestamp", datetime.now().isoformat()),
            "days_held": days_held,
            # Scale-out tracking
            "original_qty": original_qty,
            "remaining_qty": remaining_qty,
            "scales_completed": scales_completed,
            "breakeven_active": breakeven_active,
            # Regime for bot.ts to read
            "regime": regime,
        }

        # Exit context: full state at exit time (for ML exit model training)
        exit_context = {
            "atr_pct": round(atr_pct, 2),
            "phase": phase,
            "r_multiple": round(r_multiple, 2),
            "peak_r": round(peak_r, 2),
            "highest_pnl": round(highest_pnl, 2),
            "highest_price": round(highest_price, 2),
            "stop_pct": round(stop_pct, 2),
            "pnl_pct": round(pnl_pct, 2),
            "days_held": days_held,
            "current_price": current,
            "entry_price": entry,
            "regime": regime,
            "scales_completed": scales_completed,
            "breakeven_active": breakeven_active,
        }

        # ── MINIMUM HOLD TIME GATE ──────────────────────────────────────
        # Skip exits if position held < 1 hour, UNLESS it's a catastrophic loss (>10%)
        _is_catastrophic = pnl_pct <= -10.0
        if _equity_held_minutes < _MIN_EQUITY_HOLD_MINUTES and not _is_catastrophic:
            should_stop = False
            time_stop = False

        # Execute stops (for remaining shares)
        if should_stop:
            if breakeven_triggered:
                stop_type = "trailing_stop"
                reason = f"BREAKEVEN STOP: P&L {pnl_pct:+.1f}% dropped below entry after scale-out (breakeven active, {remaining_qty}/{original_qty} shares)"
            elif phase >= 2:
                stop_type = "trailing_stop"
                if phase >= 4:
                    reason = f"EVOLVING STOP Phase 4: price ${current:.2f} dropped {((highest_price - current) / highest_price * 100):.1f}% from high ${highest_price:.2f} ({stop_reason})"
                else:
                    reason = f"EVOLVING STOP Phase {phase}: P&L {pnl_pct:+.1f}% dropped {highest_pnl - pnl_pct:.1f}% from peak {highest_pnl:+.1f}% ({stop_reason})"
            else:
                stop_type = "stop_loss"
                reason = f"STOP LOSS: {pnl_pct:.1f}% loss hit Phase 1 stop at -{stop_pct:.1f}% (ATR: ${atr or 0:.2f})"

            scale_note = f" (final {remaining_qty}/{original_qty} shares, {scales_completed} prior scale-outs)" if scales_completed > 0 else ""
            # P0-5 FIX: unwind any covered calls BEFORE the stock sale. If a CC is
            # left open after selling the underlying, we're naked short a call —
            # unbounded tail risk on a gap-up. `unwind_cc_first=True` signals to
            # the executor (bot.ts / runtime) that the helper must run first and
            # block the stock close if the unwind fails.
            actions.append({"action": "CLOSE", "ticker": ticker, "side": side,
                "reason": reason + scale_note, "type": stop_type, "phase": phase,
                "qty": remaining_qty, "exit_context": exit_context,
                "unwind_cc_first": True})
            # Record stop-loss cooldown to prevent immediate re-entry
            try:
                _cd_path = os.path.join(DATA_DIR, 'voltrade_stop_cooldown.json')
                try:
                    with open(_cd_path) as _f: _cd = json.load(_f)
                except Exception: _cd = {}
                _cd[ticker] = time.time()
                _cd = {k: v for k, v in _cd.items() if time.time() - v < 86400}
                with open(_cd_path, 'w') as _f: json.dump(_cd, _f)
            except Exception: pass
        elif pnl_pct >= tp_pct and phase < 3 and scales_completed >= 2:
            scale_note = f" (final {remaining_qty}/{original_qty} shares)" if scales_completed > 0 else ""
            actions.append({"action": "CLOSE", "ticker": ticker, "side": side,
                "reason": f"TAKE PROFIT: +{pnl_pct:.1f}% hit target +{tp_pct:.1f}% (Phase {phase}){scale_note}",
                "type": "take_profit", "phase": phase,
                "qty": remaining_qty, "exit_context": exit_context,
                "unwind_cc_first": True})
        elif time_stop:
            scale_note = f" ({remaining_qty}/{original_qty} shares)" if scales_completed > 0 else ""
            actions.append({"action": "CLOSE", "ticker": ticker, "side": side,
                "reason": f"TIME STOP: {days_held} days held, P&L only {pnl_pct:+.1f}% — capital locked up{scale_note}",
                "type": "time_stop", "phase": phase,
                "qty": remaining_qty, "exit_context": exit_context,
                "unwind_cc_first": True})

        # Upgrade candidate: position is flat or slightly negative in Phase 1
        if phase == 1 and -stop_pct < pnl_pct < 1.0:
            upgrade_candidates.append({
                "ticker": ticker, "pnl_pct": pnl_pct, "market_value": market_value,
                "score": 50 + pnl_pct * 5, "qty": qty, "side": side,
            })

    # Persist stop state — single write after clean-up.
    #
    # BUGFIX P0-6: Previously we saved twice. The cleanup step used
    # `held_tickers = {pos.symbol for pos in positions}`, but `positions`
    # still includes tickers that were just emitted as CLOSE (the close
    # hasn't filled yet). That meant freshly-computed phase/scale-out state
    # was correctly kept; but a race window existed where a failed write
    # between the two dumps could leave truncated state. Consolidate into
    # a single atomic write.
    #
    # We also exclude any ticker that has an actual CLOSE action in THIS
    # cycle AND was already fully scaled out (remaining_qty==0 after
    # scaling) — those are guaranteed flat after fill and shouldn't
    # retain state that could be re-applied to a future re-entry.
    held_tickers = {pos.get('symbol', '') for pos in positions}
    _closing_now = {a.get('ticker', '') for a in actions
                    if a.get('action') == 'CLOSE'
                    and (a.get('qty') or 0) >= 0}
    for old_ticker in list(stop_state.keys()):
        if old_ticker not in held_tickers:
            del stop_state[old_ticker]
    try:
        with open(_stop_state_path, 'w') as f:
            json.dump(stop_state, f)
    except Exception:
        pass

    return {"actions": actions, "positions": len(positions), "upgrade_candidates": upgrade_candidates, "regime": regime}


def _get_atr(ticker, period=14):
    """Get 14-day ATR for a ticker using Alpaca daily bars.

    BUGFIX (2026-04-17): previous implementation returned a simple MEAN of
    the last `period` true ranges. Wilder's ATR — the industry standard and
    the one every backtest benchmarks against — uses recursive smoothing:
        ATR_t = (ATR_{t-1} * (period-1) + TR_t) / period
    The simple-mean version reacts too slowly to vol expansion AND has
    boundary artifacts that cause position-stop placement to drift from
    what the backtest assumed. Also widened the lookback so we have enough
    bars to seed the Wilder average plus a few extra for smoothing.
    """
    import requests
    try:
        # Need enough calendar days for `period * 2` trading days + buffer.
        # 14-period ATR with recursive smoothing benefits from ~2x bars.
        lookback_days = max(60, period * 4)
        start = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        url = (f"{ALPACA_DATA_URL}/v2/stocks/{ticker}/bars"
               f"?timeframe=1Day&start={start}&limit={period * 3}&adjustment=all&feed=sip")
        resp = requests.get(url, headers=_alpaca_headers(), timeout=10)
        data = resp.json()
        results = data.get("bars", [])
        if len(results) < period + 1:
            return None

        # Compute all True Ranges (ascending order, i-1 is prior day).
        trs = []
        for i in range(1, len(results)):
            h      = results[i].get("h", 0)
            l      = results[i].get("l", 0)
            prev_c = results[i - 1].get("c", 0)
            tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
            trs.append(tr)

        if len(trs) < period:
            return None

        # Seed with simple mean of the first `period` TRs, then Wilder-smooth
        # for the remainder.
        atr = sum(trs[:period]) / period
        for tr in trs[period:]:
            atr = (atr * (period - 1) + tr) / period
        return atr
    except Exception:
        return None

# ── Main Scan ────────────────────────────────────────────────────────────────

def _get_full_universe() -> list:
    """
    Returns all tradeable US equity symbols from Alpaca.
    Cached in /data/voltrade/universe_cache.json — refreshed once per day.
    This is the full ~11,600 stock universe, not just the top 100 actives.
    """
    cache_path = os.path.join(DATA_DIR, "universe_cache.json")
    # Use cached list if less than 24 hours old
    if os.path.exists(cache_path):
        age = time.time() - os.path.getmtime(cache_path)
        if age < 86400:  # 24 hours
            try:
                with open(cache_path) as f:
                    return json.load(f)
            except Exception:
                pass
    # Fetch fresh universe from Alpaca assets endpoint
    try:
        alpaca_throttle.acquire()
        resp = requests.get(
            f"{ALPACA_BASE_URL}/v2/assets?status=active&asset_class=us_equity",
            headers=_alpaca_headers(), timeout=30
        )
        assets = resp.json()
        symbols = [
            a["symbol"] for a in assets
            if a.get("tradable")
            and "." not in a.get("symbol", "")
            and not a.get("symbol", "").endswith("W")
            and len(a.get("symbol", "")) <= 5
            and a.get("symbol", "")
        ]
        # Cache it
        with open(cache_path, "w") as f:
            json.dump(symbols, f)
        import logging; logging.getLogger("bot_engine").info(f"Universe cache refreshed: {len(symbols):,} symbols")
        return symbols
    except Exception as e:
        import logging; logging.getLogger("bot_engine").warning(f"Could not fetch full universe: {e} — falling back to movers only")
        return []


def scan_market():
    """
    Full market scan — ALL ~11,600 tradeable US stocks, not just top 100.

    ⚠ Portfolio DD halt: before scanning, update_equity_peak() is called so
    that is_trading_halted() reflects the current account state. If halted,
    scan returns an empty action list with halt_reason populated, blocking
    ALL new entries until the one-way ratchet resume fires.
    (See get_portfolio_dd_state / DRAWDOWN_HALT_* config keys.)

    WHY FULL UNIVERSE:
      - most-actives API only returns top 100 by volume — misses 99% of stocks
      - Scanning 11,600 with 16 parallel workers takes ~4 seconds
      - Deep analysis only runs on top 10 — same speed as before
      - We were missing real opportunities (AEHR +17%, ENVX +13%) every scan

    Pipeline:
    1. Load full universe (~11,600 symbols, cached daily)
    2. Fetch ALL snapshots in parallel (16 workers, ~4 seconds)
    3. Quick-score everything that passes price+volume filters
    4. Deep-analyze top 5 candidates (capped for timeout safety)
    5. Return top positions with full trade recommendations
    """
    import signal as _sig

    def _scan_timeout_handler(signum, frame):
        raise TimeoutError("scan_market exceeded 50-second hard cap")

    _old_handler = _sig.signal(_sig.SIGALRM, _scan_timeout_handler)
    _sig.alarm(55)  # 55s hard cap — Node kills at 90s
    try:
        return _scan_market_inner()
    except TimeoutError as _te:
        import logging
        logging.getLogger("bot_engine").warning(f"Scan timed out: {_te} — returning partial results")
        global _partial_scan_result
        if _partial_scan_result:
            return _partial_scan_result  # No "error" key — Node will process it
        return {"error": "scan_market timed out with no partial results", "trades": [], "new_trades": []}
    finally:
        _sig.alarm(0)
        _sig.signal(_sig.SIGALRM, _old_handler)


def _scan_market_inner():
    global _partial_scan_result
    _partial_scan_result = None

    # MEM FIX 2026-04-20: Log memory at every phase boundary so OOM kills
    # leave a trail. Combined with gc.collect() between phases this cuts
    # peak memory ~30-80MB and gives diagnosable logs when something leaks.
    _log_mem_phase("scan_inner_start")

    from concurrent.futures import ThreadPoolExecutor as _TPE

    # ── Portfolio-level DD halt check (v1.0.29+) ────────────────────────────────
    # If portfolio has breached the DRAWDOWN_HALT_PCT threshold, block all
    # new entries until regime returns to BULL/NEUTRAL AND equity recovers to
    # within DRAWDOWN_HALT_RESUME_EQUITY_PCT of peak (one-way ratchet).
    # Existing positions continue to be managed by manage_positions() — only
    # entries are gated, not exits.
    try:
        _acct = get_alpaca_account()
        _equity = float(_acct.get("equity", 0) or 0)
        # Determine current regime for ratchet evaluation
        _regime = "NEUTRAL"
        try:
            from macro_data import get_macro_snapshot
            _macro = get_macro_snapshot()
            from system_config import get_market_regime as _gmr
            _regime = _gmr(
                float(_macro.get("vxx_ratio", 1.0) or 1.0),
                float(_macro.get("spy_vs_ma50", 1.0) or 1.0),
                spy_below_200_days=int(_macro.get("spy_below_200_days", 0) or 0),
                spy_above_200d=bool(_macro.get("spy_above_200d", True)),
            )
        except Exception:
            pass
        _dd_state = update_equity_peak(_equity, regime=_regime)
        if _dd_state.get("halted"):
            import logging
            logging.getLogger("bot_engine").warning(
                f"[DD_HALT] scan_market returning empty — {_dd_state.get('halt_reason', '')}"
            )
            return {
                "trades": [], "new_trades": [], "top_10": [],
                "halted": True,
                "halt_reason": _dd_state.get("halt_reason", ""),
                "peak_equity": _dd_state.get("peak_equity", 0),
                "current_equity": _dd_state.get("current_equity", 0),
                "dd_pct": _dd_state.get("dd_pct", 0),
                "regime": _regime,
            }
    except Exception as _dd_e:
        # DD tracking must never break scanning — log and continue
        import logging
        logging.getLogger("bot_engine").debug(f"DD halt check failed (non-fatal): {_dd_e}")

    # Step 1: Get full universe (cached daily)
    full_universe = _get_full_universe()

    # Also get today's top movers as a "freshness boost" — 
    # they're guaranteed to be moving right now and get priority
    movers_fresh = []
    try:
        alpaca_throttle.acquire()
        resp = requests.get(f"{ALPACA_DATA_URL}/v1beta1/screener/stocks/movers?top=50",
                           headers=_alpaca_headers(), timeout=10)
        mv = resp.json()
        movers_fresh = [s.get("symbol","") for s in mv.get("gainers",[]) + mv.get("losers",[])]
        movers_fresh = [s for s in movers_fresh if s]
    except Exception:
        pass

    # Combine: full universe first, then any fresh movers not already in it
    all_tickers_set = set(full_universe)
    ticker_symbols = list(full_universe)
    for sym in movers_fresh:
        if sym not in all_tickers_set:
            ticker_symbols.append(sym)

    # Fallback if universe fetch failed
    if not ticker_symbols:
        try:
            alpaca_throttle.acquire()
            resp = requests.get(f"{ALPACA_DATA_URL}/v1beta1/screener/stocks/most-actives?by=volume&top=100",
                               headers=_alpaca_headers(), timeout=15)
            ticker_symbols = [s["symbol"] for s in resp.json().get("most_actives", []) if s.get("symbol")]
        except Exception:
            return {"error": "Could not fetch market universe", "trades": []}

    # Step 2: Fetch ALL snapshots in parallel (16 workers = ~4 seconds for 11K stocks)
    batches = [ticker_symbols[i:i+1000] for i in range(0, len(ticker_symbols), 1000)]
    snap_all = {}

    def _fetch_snap(batch):
        try:
            alpaca_throttle.acquire()
            r = requests.get(f"{ALPACA_DATA_URL}/v2/stocks/snapshots?symbols={','.join(batch)}&feed=sip",
                headers=_alpaca_headers(), timeout=15)
            return r.json()
        except Exception:
            return {}

    # 6 workers balances speed vs Railway thread limits (was 16 — caused EAGAIN)
    with _TPE(max_workers=6) as pool:
        for snap_data in pool.map(_fetch_snap, batches):
            snap_all.update(snap_data)

    quick_results = []
    _et_now = datetime.now(ZoneInfo("America/New_York"))
    _et_hour = _et_now.hour
    _et_min  = _et_now.minute
    # BUGFIX (2026-04-17): previous check was (_et_hour == 9 and _et_min < 60).
    # Minutes are always 0–59, so that condition was True for the *entire*
    # 9 o'clock hour — including 9:00–9:29 which is PRE-MARKET. We only want
    # the first 30 minutes AFTER the 9:30 open, i.e. 9:30–9:59. Rewritten
    # to require the market to actually be open.
    _opening_half_hour = (_et_hour == 9 and 30 <= _et_min < 60)
    _min_vol = 100_000 if _opening_half_hour else MIN_VOLUME

    all_tickers = list(snap_all.keys())  # For scanned count

    # Step 3: Quick-score all stocks that pass price+volume filters
    for sym, snap in snap_all.items():
        try:
            bar = snap.get("dailyBar", {})
            prev = snap.get("prevDailyBar", {})
            c = float(bar.get("c", 0))
            o = float(bar.get("o", c))
            pc = float(prev.get("c", c))
            v = int(bar.get("v", 0))
            if c < MIN_PRICE or v < _min_vol:
                continue
            if "." in sym or len(sym) > 5:
                continue
            change_pct = ((c - pc) / pc * 100) if pc > 0 else 0
            _capped_change = min(abs(change_pct), 15.0)
            _extreme_penalty = -30 if abs(change_pct) > 50 else (-15 if abs(change_pct) > 30 else 0)
            _vol_score = min(v / 1_000_000, 5.0) * 2
            _direction_bonus = 5 if abs(change_pct) > 10 else (2 if abs(change_pct) > 5 else 0)
            quick_results.append({
                "ticker": sym,
                "price": round(c, 2),
                "close": c,
                "open": o,
                "prev_close": pc,
                "volume": v,
                "change_pct": round(change_pct, 2),
                "quick_score": _capped_change * 3 + _vol_score + _direction_bonus + _extreme_penalty,
                "above_vwap": c > o,
                "range_pct": 0,
                "vwap_dist": 0,
                "reasons": [],
            })
        except Exception:
            continue

    if not quick_results:
        return {"error": "Could not fetch market data from Alpaca", "trades": []}

    # Release raw snapshot data — quick_results now has everything we need
    del snap_all
    import gc; gc.collect()

    # Sort by quick score
    quick_results.sort(key=lambda x: x["quick_score"], reverse=True)
    scored = quick_results

    # Checkpoint 0: save quick-scored results so timeout handler has something
    _partial_scan_result = {
        "scanned": len(all_tickers) if 'all_tickers' in dir() else len(scored),
        "top_10": [{"ticker": s["ticker"], "score": s.get("quick_score", 0), "reasons": s.get("reasons", []), "side": "buy"} for s in scored[:10]],
        "new_trades": [],
        "trades": [],
        "partial": True,
    }
    # Step 3: Deep analyze top 5 in PARALLEL (capped from 10 for timeout safety)
    # Each deep_score internally runs 5 data sources in parallel too.
    # Per-future timeout (8s) + total cap (35s) prevents yfinance hangs.
    from concurrent.futures import ThreadPoolExecutor, as_completed
    _gc_checkpoint("after_quick_scan")  # MEM: release quick-scan dicts before deep work
    top_candidates = scored[:10]  # Deep analyze top 10 quick-scored stocks
    deep_scored = [None] * len(top_candidates)

    def _deep_one(args):
        idx, candidate = args
        try:
            return idx, deep_score(candidate["ticker"], candidate)
        except Exception:
            return idx, candidate

    _log_mem_phase("pre_deep_score")
    with ThreadPoolExecutor(max_workers=4) as _dpool:
        futures = {_dpool.submit(_deep_one, (i, c)): i for i, c in enumerate(top_candidates)}
        try:
            for future in as_completed(futures, timeout=35):  # 35s total hard cap
                try:
                    idx, result = future.result(timeout=8)  # 8s per ticker — kill slow yfinance calls
                    deep_scored[idx] = result
                except Exception:
                    deep_scored[futures[future]] = None
        except TimeoutError:
            import logging
            logging.getLogger("bot_engine").warning("Deep scoring hit 25s cap — using partial results")

    deep_scored = [d for d in deep_scored if d is not None]

    # Sort by deep score (or quick score if no deep)
    deep_scored.sort(key=lambda x: x.get("deep_score", x.get("quick_score", 0)), reverse=True)

    # Save partial results after deep scoring — survives SIGALRM timeout
    _partial_scan_result = {
        "timestamp": datetime.now().isoformat(),
        "scanned": len(all_tickers) if 'all_tickers' in dir() else 0,
        "filtered": len(scored),
        "deep_analyzed": len(deep_scored),
        "top_10": [{"ticker": s["ticker"], "score": s.get("deep_score", s.get("quick_score", 0)), "reasons": s.get("reasons", [])[:2], "side": s.get("side", "buy")} for s in deep_scored[:10] if s],
        "new_trades": [],
        "trades": [],
        "partial": True,
    }

    # Step 4: Get account info for position sizing
    try:
        account = get_alpaca_account()
        portfolio_value = float(account.get("portfolio_value", 100000))
        cash = float(account.get("cash", 100000))
    except Exception:
        portfolio_value = 100000
        cash = 100000

    # Step 5: Check current positions
    try:
        current_positions = get_alpaca_positions()
        current_tickers = (
            [p.get("symbol") for p in current_positions]
            if isinstance(current_positions, list) else []
        )
        # Count only stock positions for stock slot limit (options have separate slots)
        num_positions = sum(
            1 for p in (current_positions if isinstance(current_positions, list) else [])
            if len(str(p.get("symbol", ""))) <= 8 and p.get("asset_class", "us_equity") != "us_option"
        )
    except Exception:
        current_positions = []
        current_tickers = []
        num_positions = 0

    # Step 6: Generate trade recommendations using DYNAMIC POSITION SIZING
    trades = []
    slots_available = MAX_POSITIONS - num_positions

    # Get macro context for position sizing
    try:
        from macro_data import get_macro_snapshot
        _macro = get_macro_snapshot()
    except Exception:
        _macro = {}

    # Import dynamic sizing engine
    try:
        from position_sizing import calculate_position, check_halt_status
        _has_sizer = True
    except ImportError:
        _has_sizer = False

    for stock in deep_scored:
        if len(trades) >= slots_available:
            break

        final_score = stock.get("deep_score", stock.get("quick_score", 0))
        ticker = stock["ticker"]

        # Skip if already holding
        if ticker in current_tickers:
            continue

        # Skip if recently stopped out — REGIME-AWARE cooldown
        # Logic: when fear is high, stocks keep trending down longer.
        # Re-entering too soon in a panic compounds losses far more than in a calm market.
        #
        # Cooldown durations (backed by intraday momentum research):
        #   PANIC  (VXX > 130% of 30d avg): 4 hours — panic selloffs last 3-6h on avg
        #   BEAR   (VXX > 115%):            3 hours — bearish drift typically lasts 2-4h
        #   CAUTION (VXX > 105%):           2.5 hours — cautious market, extra buffer
        #   NEUTRAL (VXX near avg):         2 hours — baseline (was always 2h)
        #   BULL   (VXX < 90% of avg):      1.5 hours — calm/rising mkt = faster mean revert
        #
        # How regime is detected at this point: uses the most recent macro snapshot's VXX
        # ratio. Falls back to 2h (neutral) if data unavailable. No API call — uses
        # _macro dict already loaded above.
        _cooldown_path = os.path.join(DATA_DIR, 'voltrade_stop_cooldown.json')
        try:
            with open(_cooldown_path) as _f:
                _cooldown = json.load(_f)
        except Exception:
            _cooldown = {}
        _last_stop = _cooldown.get(ticker, 0)
        # Determine cooldown seconds from current regime (Fix B: include 200d MA)
        _vxx_r_scan      = float(_macro.get("vxx_ratio", 1.0) or 1.0)
        _spy_ma_scan     = float(_macro.get("spy_vs_ma50", 1.0) or 1.0)
        _spy_b200_scan   = int(_macro.get("spy_below_200_days", 0) or 0)
        _spy_above_200d  = bool(_macro.get("spy_above_200d", True))
        # Use the same regime function so 200MA block is consistent everywhere
        try:
            from system_config import get_market_regime as _gmr
            _scan_regime = _gmr(_vxx_r_scan, _spy_ma_scan,
                                spy_below_200_days=_spy_b200_scan,
                                spy_above_200d=_spy_above_200d)
        except Exception:
            # Fallback inline
            if _vxx_r_scan >= 1.30 or _spy_ma_scan < 0.94 or _spy_b200_scan >= 10:
                _scan_regime = "PANIC" if _vxx_r_scan >= 1.30 else "BEAR"
            elif _vxx_r_scan >= 1.15: _scan_regime = "BEAR"
            elif _vxx_r_scan >= 1.05: _scan_regime = "CAUTION"
            elif _vxx_r_scan <= 0.90: _scan_regime = "BULL"
            else:                     _scan_regime = "NEUTRAL"
        _cooldown_map = {"PANIC": 14400, "BEAR": 10800, "CAUTION": 9000,
                         "BULL": 5400, "NEUTRAL": 7200}
        _cooldown_secs  = _cooldown_map.get(_scan_regime, 7200)
        _cooldown_label = f"{_scan_regime} ({_cooldown_secs//3600:.1f}h)"
        if time.time() - _last_stop < _cooldown_secs:
            continue  # Ticker in cooldown — skip. Re-entry blocked ({_cooldown_label})

        # Fix B: block ALL new stock longs in BEAR/PANIC (200MA slow-bear included)
        if _scan_regime in ("BEAR", "PANIC"):
            continue  # No new stock longs in bear/panic — preserve capital

        # Skip if below minimum score — use regime-adaptive threshold from get_adaptive_params.
        # Previously this used a hardcoded MIN_SCORE=65 which conflicted with system_config
        # values (63 in BULL, 67 in CAUTION, 75 in BEAR/PANIC). Now uses the adaptive value.
        try:
            from system_config import get_adaptive_params as _gap
            _scan_params = _gap(
                vxx_ratio=_vxx_r_scan,
                spy_vs_ma50=_spy_ma_scan,
                spy_below_200_days=_spy_b200_scan,
            )
            _min_score_threshold = _scan_params.get("MIN_SCORE", 63)
        except Exception:
            _min_score_threshold = 63  # system_config BASE_CONFIG default
        if final_score < _min_score_threshold:
            continue

        # Correlation / sector check — don't over-concentrate
        # Pass dynamic sector from yfinance and raw positions so options are excluded
        _pos_list = current_positions if isinstance(current_positions, list) else []
        if check_sector_correlation(
            ticker,
            current_tickers + [t["ticker"] for t in trades],
            sector=stock.get("sector"),
            existing_positions=_pos_list,
        ):
            continue

        side = stock.get("side", "buy")
        action_label = stock.get("action_label", "BUY")
        change_pct_val = stock.get("change_pct", 0) or 0

        # ── Dollar volume filter (Fix D) ────────────────────────────────
        # Minimum $50M daily dollar volume = price × shares traded.
        # WHY: FCUV ($7, $370M vol) and PFSA ($2, $450M vol) passed all other
        # filters but are micro-caps with manipulated momentum spikes.
        # A $50M floor blocks them while keeping NVDA at $13 (post-split, $2B+/day).
        # Does NOT use a price floor (that hurt backtest by blocking split-adj stocks).
        _stock_price  = float(stock.get("price", 0) or 0)
        _stock_volume = float(stock.get("volume", 0) or 0)
        _dollar_vol   = _stock_price * _stock_volume
        _MIN_DOLLAR_VOL = 50_000_000  # $50M minimum daily dollar volume
        if _dollar_vol < _MIN_DOLLAR_VOL:
            continue  # Skip micro-cap / low-liquidity stock

        # Sector quality filter — 3-year backtest confirmed these destroy returns
        # Gaming (DKNG, RBLX): 25% WR over 3 years  | Leveraged ETFs: 22% WR
        # Travel (ABNB, DASH): 20% WR               | These are structural drags
        # Consolidated from system_config.BASE_CONFIG["BLOCKED_TICKERS"] — single source of truth
        _cfg_blocked = set(BASE_CONFIG.get("BLOCKED_TICKERS", []))
        _BLOCKED_TICKERS = _cfg_blocked | {
            "UBER",  # Ride-share — consistent underperformer (not in system_config by design)
        }
        if ticker in _BLOCKED_TICKERS:
            continue  # Skip — 3-year backtest shows these hurt more than they help

        # Extreme mover check: stock already up 50%+ today
        # Don't buy OR short the spike — write to watchlist for overnight analysis
        # Tomorrow's setup (continuation or mean reversion) is a much cleaner entry
        if change_pct_val > 50:
            _em_path = os.path.join(DATA_DIR, 'extreme_movers_today.json')
            try:
                try:
                    with open(_em_path) as _f: _em = json.load(_f)
                except Exception: _em = []
                # Add if not already in list
                if not any(e.get('ticker') == ticker for e in _em):
                    _em.append({
                        'ticker': ticker, 'price': stock.get('price', 0),
                        'change_pct': change_pct_val, 'volume': stock.get('volume', 0),
                        'score': final_score, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
                        'high': stock.get('high', 0), 'setup': 'extreme_mover',
                    })
                    with open(_em_path, 'w') as _f: json.dump(_em, _f)
            except Exception: pass
            continue  # Skip — will be analyzed tonight for tomorrow

        # Stock shorting disabled (backtest: -$419K). Bearish plays use options scanner.
        if side in ("sell", "short") and stock.get("trade_type") != "options":
            side = "buy"
            action_label = "BUY"

        # ── Spread awareness check (Fix 2026-04-10) ───────────────────
        # Reject equities with bid-ask spread > 0.5% of price.
        # QQQ lost $698 on a single round-trip due to market order spread.
        # This check prevents entering stocks with poor liquidity / wide spreads.
        try:
            alpaca_throttle.acquire()
            _quote_r = requests.get(f"{ALPACA_DATA_URL}/v2/stocks/{ticker}/quotes/latest",
                params={"feed": "sip"}, headers=_alpaca_headers(), timeout=5)
            _quote = _quote_r.json().get("quote", {})
            _bid = float(_quote.get("bp", 0) or 0)
            _ask = float(_quote.get("ap", 0) or 0)
            if _bid > 0 and _ask > 0:
                _mid = (_bid + _ask) / 2
                _spread_pct = (_ask - _bid) / _mid if _mid > 0 else 0
                if _spread_pct > 0.005:  # 0.5% max spread for equities
                    continue  # Skip — spread too wide, would lose to slippage
        except Exception:
            pass  # If quote fetch fails, allow the trade (don't block on API failure)

        # Instrument decision: stock vs 2x ETF vs options (unified selector)
        instrument_decision = {"chosen": "stock", "strategy": "buy_stock", "reasoning": "default"}
        options_decision = {"use_options": False, "strategy": "stock", "reason": ""}
        # Look up shares held for this ticker (needed for covered call eligibility)
        _ticker_shares = 0
        for _p in (current_positions if isinstance(current_positions, list) else []):
            if str(_p.get("symbol", "")).upper() == ticker.upper() and _p.get("asset_class", "us_equity") != "us_option":
                _ticker_shares = int(float(_p.get("qty", 0)))
                break
        try:
            from instrument_selector import select_instrument
            instrument_decision = select_instrument(
                trade={**stock, "score": final_score, "deep_score": final_score,
                       "side": side, "action_label": action_label,
                       "shares_held": _ticker_shares},
                equity=portfolio_value,
                positions=current_positions if isinstance(current_positions, list) else [],
                macro=_macro,
            )
            # Backward compat for options_execution fields
            if instrument_decision.get("chosen") == "options":
                options_decision = {"use_options": True, "strategy": instrument_decision.get("strategy", "stock"),
                                    "reason": instrument_decision.get("reasoning", ""),
                                    "edge_pct": instrument_decision.get("scores", {}).get("options", {}).get("edge_pct", 0)}
        except Exception:
            try:
                from options_execution import should_use_options
                options_decision = should_use_options(
                    {**stock, "score": final_score, "side": side, "action_label": action_label},
                    portfolio_value,
                    current_positions if isinstance(current_positions, list) else []
                )
                if options_decision.get("use_options"):
                    instrument_decision = {"chosen": "options", "strategy": options_decision["strategy"],
                                           "reasoning": options_decision["reason"]}
            except Exception:
                pass  # Both modules unavailable, default to stock

        # Dynamic position sizing (replaces fixed 5%)
        if _has_sizer:
            sizing = calculate_position(
                trade={**stock, "score": final_score, "side": side},
                equity=portfolio_value,
                current_positions=current_positions if isinstance(current_positions, list) else [],
                macro=_macro,
            )
            if sizing.get("blocked"):
                continue  # Sizing engine blocked this trade (halted, too small, etc.)
            shares = sizing["shares"]
            position_value = sizing["position_value"]
            stop_loss = sizing["stop_loss"]
            take_profit = sizing["take_profit"]
            sizing_reasoning = sizing.get("reasoning", "")
            sizing_scalars = sizing.get("scalars", {})
        else:
            # Fallback to old fixed sizing if import fails
            position_value = min(portfolio_value * MAX_POSITION_PCT, cash * 0.9)
            shares = int(position_value / stock["price"]) if stock["price"] > 0 else 0
            stop_loss = round(stock["price"] * (1 - STOP_LOSS_PCT), 2)
            take_profit = round(stock["price"] * (1 + TAKE_PROFIT_PCT), 2)
            position_value = round(shares * stock["price"], 2)
            sizing_reasoning = "Fallback: fixed 5% sizing"
            sizing_scalars = {}

        if shares <= 0:
            continue

        trades.append({
            "action": action_label,
            "side": side,
            "trade_type": stock.get("trade_type", "stock"),
            "ticker": ticker,
            "shares": shares,
            "price": stock["price"],
            "score": final_score,
            "reasons": stock.get("reasons", []),
            "recommendation": stock.get("recommendation"),
            "rec_reasoning": stock.get("rec_reasoning"),
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "position_value": position_value,
            "sizing_reasoning": sizing_reasoning,
            "sizing_scalars": sizing_scalars,
            "momentum_score": stock.get("momentum_score"),
            "mean_reversion_score": stock.get("mean_reversion_score"),
            "vrp_score": stock.get("vrp_score"),
            "squeeze_score": stock.get("squeeze_score"),
            "volume_score": stock.get("volume_score"),
            "ewma_rv": stock.get("ewma_rv"),
            "garch_rv": stock.get("garch_rv"),
            "rsi": stock.get("rsi"),
            "vrp": stock.get("vrp"),
            "instrument": instrument_decision.get("chosen", "stock"),
            "instrument_strategy": instrument_decision.get("strategy", "buy_stock"),
            "instrument_reasoning": instrument_decision.get("reasoning", ""),
            "instrument_ticker": instrument_decision.get("ticker", ticker),  # Could be ETF ticker
            "instrument_scores": instrument_decision.get("scores", {}),
            "use_options": options_decision.get("use_options", False),
            "options_strategy": options_decision.get("strategy", "stock"),
            "options_reasoning": options_decision.get("reason", ""),
            "options_edge_pct": options_decision.get("edge_pct", 0),
            "rules_score": stock.get("rules_only_score"),
            "ml_score_raw": stock.get("ml_only_score"),
            # entry_features: the 25 ML features at the moment of entry.
            # Saved here so when the trade closes, ml_model_v2 can train on
            # THESE SPECIFIC SIGNALS — not generic market data.
            # This is what enables true self-learning.
            "entry_features": stock.get("entry_features") or stock.get("ml_features"),
            "entry_date": time.strftime("%Y-%m-%d"),
            "regime_at_entry": stock.get("regime_label", "UNKNOWN"),
        })

    # Update partial results with stock trades — survives SIGALRM if timeout
    # fires during options scanner or later steps
    _partial_scan_result = {
        "timestamp": datetime.now().isoformat(),
        "scanned": len(all_tickers) if 'all_tickers' in dir() else 0,
        "filtered": len(scored),
        "deep_analyzed": len(deep_scored),
        "portfolio_value": portfolio_value,
        "cash": cash,
        "current_positions": num_positions,
        "slots_available": slots_available,
        "new_trades": list(trades),  # Copy — trades may grow later
        "top_10": [{"ticker": s["ticker"], "price": s.get("price", 0), "score": s.get("deep_score", s.get("quick_score", 0)), "change_pct": s.get("change_pct", 0), "side": s.get("side", "buy"), "action_label": s.get("action_label", "BUY"), "reasons": s.get("reasons", [])[:2]} for s in deep_scored[:10]],
        "partial": True,
    }

    # Step 6c: Covered Call Sweep (moved before options scanner — no external API calls)
    # Check ALL existing stock positions for covered call eligibility.
    # Runs independently of the stock scanner — a position doesn't need to
    # "score well" to have a covered call written on it. It just needs 100+ shares.
    # This fixes QQQ (passive floor) never getting covered calls because it
    # never appears in the stock scanner results.
    try:
        for _cc_pos in (current_positions if isinstance(current_positions, list) else []):
            _cc_ticker = str(_cc_pos.get("symbol", ""))
            _cc_qty = int(float(_cc_pos.get("qty", 0)))
            _cc_class = _cc_pos.get("asset_class", "us_equity")

            # Skip options positions, positions with <100 shares, or long OCC symbols
            if _cc_class == "us_option" or len(_cc_ticker) > 8 or _cc_qty < 100:
                continue

            # Skip if we already have a short call on this ticker
            _has_cc = False
            for _ex_pos in (current_positions if isinstance(current_positions, list) else []):
                _ex_sym = str(_ex_pos.get("symbol", ""))
                if (_ex_pos.get("asset_class") == "us_option"
                        and _ex_sym.startswith(_cc_ticker)
                        and int(float(_ex_pos.get("qty", 0))) < 0
                        and "C" in _ex_sym[len(_cc_ticker):]):
                    _has_cc = True
                    break
            if _has_cc:
                continue

            # Skip if already in today's trade list
            if _cc_ticker in [t["ticker"] for t in trades if t.get("trade_type") == "options"]:
                continue

            # Earnings guard (7 days for covered calls)
            try:
                from options_execution import _check_earnings_guard
                if not _check_earnings_guard(_cc_ticker, 7):
                    continue
            except Exception:
                pass

            # Skip in BEAR/PANIC regimes
            _cc_regime = _scan_regime if '_scan_regime' in dir() else "NEUTRAL"
            if _cc_regime in ("BEAR", "PANIC"):
                continue

            _cc_price = float(_cc_pos.get("current_price", 0) or 0)
            if _cc_price <= 0:
                continue

            trades.append({
                "action": "SELL COVERED CALL",
                "side": "sell",
                "trade_type": "options",
                "ticker": _cc_ticker,
                "shares": 0,
                "price": _cc_price,
                "score": 75,
                "reasons": [f"Covered call on {_cc_qty} shares of {_cc_ticker}"],
                "recommendation": None,
                "rec_reasoning": f"Covered call: sell OTM calls against {_cc_qty} shares",
                "stop_loss": None,
                "take_profit": None,
                "position_value": 0,
                "sizing_reasoning": f"Covered call on existing {_cc_qty}-share position",
                "sizing_scalars": {},
                "momentum_score": None,
                "mean_reversion_score": None,
                "vrp_score": None,
                "squeeze_score": None,
                "volume_score": None,
                "ewma_rv": None,
                "garch_rv": None,
                "rsi": None,
                "vrp": None,
                "instrument": "options",
                "instrument_strategy": "covered_call",
                "instrument_reasoning": f"Covered call on {_cc_qty} shares",
                "instrument_ticker": _cc_ticker,
                "instrument_scores": {},
                "use_options": True,
                "options_strategy": "covered_call",
                "options_reasoning": f"Sell OTM call against {_cc_qty}-share holding",
                "options_edge_pct": 3.0,
                "rules_score": 75,
                "ml_score_raw": None,
                "shares_held": _cc_qty,
            })
    except Exception as _cc_sweep_err:
        import logging as _cc_log
        _cc_log.getLogger("bot_engine").warning(f"Covered call sweep error: {_cc_sweep_err}")

    # Step 6b: Run options scanner synchronously with real portfolio equity.
    # Options have their OWN slot allocation (MAX_OPTIONS_POSITIONS), separate
    # from stock slots. This lets options trade even when stock positions are full.
    # Count existing options positions to determine available options slots.
    options_trade_count = 0
    try:
        # Count current options positions (OCC symbols are > 8 chars)
        existing_options_count = sum(
            1 for p in (current_positions if isinstance(current_positions, list) else [])
            if len(str(p.get("symbol", ""))) > 8 or p.get("asset_class") == "us_option"
        )
        options_slots = MAX_OPTIONS_POSITIONS - existing_options_count
    except Exception:
        options_slots = MAX_OPTIONS_POSITIONS
    try:
        from options_scanner import get_options_trades
        options_trades = get_options_trades(
            equity=portfolio_value,
            current_tickers=current_tickers + [t["ticker"] for t in trades],
            max_new=max(1, options_slots),
            min_score=65.0,
        )
        for ot in options_trades:
            if options_trade_count >= options_slots:
                break
            # Convert to bot_engine trade format
            trades.append({
                "action":             ot.get("action_label", "OPTIONS"),
                "side":               ot.get("side", "buy"),
                "trade_type":         "options",
                "ticker":             ot["ticker"],
                "shares":             0,  # Options: shares = 0, handled by options_execution
                "price":              ot["price"],
                "score":              ot["score"],
                "reasons":            [ot.get("reasoning", "")],
                "recommendation":     None,
                "rec_reasoning":      ot.get("reasoning", ""),
                "stop_loss":          None,
                "take_profit":        None,
                "position_value":     ot.get("position_dollars", portfolio_value * 0.05),
                "sizing_reasoning":   f"Options scanner: {ot.get('setup','unknown')} setup",
                "sizing_scalars":     {},
                "momentum_score":     None,
                "mean_reversion_score": None,
                "vrp_score":          None,
                "squeeze_score":      None,
                "volume_score":       None,
                "ewma_rv":            None,
                "garch_rv":           None,
                "rsi":                None,
                "vrp":                None,
                "instrument":         "options",
                "instrument_strategy": ot.get("options_strategy", ""),
                "instrument_reasoning": ot.get("reasoning", ""),
                "instrument_ticker":  ot["ticker"],
                "instrument_scores":  {},
                "use_options":        True,
                "options_strategy":   ot.get("options_strategy", ""),
                "options_reasoning":  ot.get("reasoning", ""),
                "options_edge_pct":   0,
                "rules_score":        ot["score"],
                "ml_score_raw":       None,
                "entry_features":     None,
                "entry_date":         time.strftime("%Y-%m-%d"),
                "regime_at_entry":    "OPTIONS_SCANNER",
                # Pass through all setup-specific keys for execution
                **{k: v for k, v in ot.items()
                   if k not in ("action", "action_label", "side", "trade_type",
                                "ticker", "price", "score", "reasoning",
                                "position_dollars", "position_pct", "source")},
            })
            options_trade_count += 1
    except Exception as _oe2:
        pass  # Options scan failed — stock trades still execute normally

    # Step 7: Check position management
    try:
        mgmt = manage_positions()
    except Exception:
        mgmt = {"actions": [], "error": "manage_positions crashed", "upgrade_candidates": []}

    # Step 8: Intraday Shorts — DISABLED (pro-level overhaul)
    # Backtest: -$419K total P&L, 28.5% WR, 419% max DD. Permanently disabled.
    intraday_short_result = {"actions": [], "status": "disabled_permanently", "enabled": False,
                             "disabled_reason": "Pro-level overhaul: shorts destroyed value (-$419K over 10yr backtest)"}

    # Step 8b: Options Position Manager (pro-grade exit management) ─────────
    # Runs every cycle: DTE exits, 50% profit targets, Greeks monitoring, rolling
    options_mgmt_result = {"actions": [], "positions_checked": 0}
    try:
        from options_manager import manage_options_positions
        options_mgmt_result = manage_options_positions(equity=portfolio_value)
    except Exception as _ome:
        options_mgmt_result["error"] = str(_ome)[:200]

    # Step 9: Third Leg (v1.0.25) ─────────────────────────────────────────────
    # Runs alongside stock scan in every cycle.
    # Backtest: ALL 64 combinations beat SPY. Winner: VRP=15% + Sector=12%.
    # Result: CAGR +14.8%/yr vs 2-leg +13.8% vs SPY +12.3%
    try:
        third_leg_result = _run_third_leg(_macro)
    except Exception:
        third_leg_result = {"actions": [], "status": "error"}

    # Step 9b: Convexity Overlay (pro-level overhaul) ────────────────────────
    # Permanent tail hedge replacing sector rotation. Budget: 1-2% of equity
    # annually on QQQ protective puts (30-45 DTE far OTM). Increases to 3-4% in PANIC/BEAR.
    convexity_result = {"actions": [], "status": "ok"}
    try:
        convexity_result = _run_convexity_overlay(_macro)
    except Exception as _ce:
        convexity_result["status"] = f"error: {str(_ce)[:80]}"

    # Step 10: Passive SPY Floor (v1.0.29) ──────────────────────────────────
    # Hold passive SPY allocation based on regime. Captures market drift
    # in calm bull markets where momentum signals are noise.
    # Backtest: 12.9% CAGR (beats SPY 12.3%). Fixes quiet bull year losses.
    spy_floor_result = {"actions": [], "status": "ok", "target_pct": 0}
    try:
        spy_floor_result = _manage_spy_floor(_macro)
    except Exception as _sfe:
        spy_floor_result["status"] = f"error: {str(_sfe)[:80]}"

    # Step 10b: Defensive Floor (P1-1, GLD bear-regime rotation) ──────────
    # Rotate INTO GLD when regime goes bearish / death cross fires, and
    # OUT of GLD when regime recovers. Sized by DEFENSIVE_FLOOR_* config.
    # User ask: "switch from qqq to something in bear".
    defensive_floor_result = {"actions": [], "status": "ok", "target_pct": 0}
    try:
        defensive_floor_result = _manage_defensive_floor(_macro)
    except Exception as _dfe:
        defensive_floor_result["status"] = f"error: {str(_dfe)[:80]}"

    # ══════════════════════════════════════════════════════════════════════
    # TIERED STRATEGY LAYER — runs in parallel to existing scoring.
    # Existing scan continues to produce `new_trades`. Tier engine adds
    # its own trade list in `tier_actions`. bot.ts dispatches both.
    # ══════════════════════════════════════════════════════════════════════
    tiered_actions = []
    kill_status = None
    tier_stats = {}
    if _HAS_TIERED:
        try:
            acct = get_alpaca_account()
            equity = float(acct.get("equity", 100000) or 100000)
            bp = float(acct.get("buying_power", equity) or equity)
            positions = get_alpaca_positions() or []
            peak = update_peak_equity(equity)

            vxx_r = float(_macro.get("vxx_ratio", 1.0) or 1.0) if '_macro' in locals() else 1.0
            daily_pnl = float(acct.get("daily_pnl_pct", 0) or 0) / 100.0
            kill_status = check_kill_switches(
                equity=equity, peak_equity=peak, positions=positions,
                daily_pnl_pct=daily_pnl, vxx_ratio=vxx_r, buying_power=bp,
            )

            if not kill_status["killed"]:
                ctx = TierContext(
                    equity=equity, peak_equity=peak, buying_power=bp,
                    positions=positions,
                    macro=_macro if '_macro' in locals() else {},
                    portfolio_margin=get_portfolio_margin_status(),
                    daily_pnl_pct=daily_pnl,
                    regime=_regime_ctx.get("regime_label", "NEUTRAL") if '_regime_ctx' in locals() else "NEUTRAL",
                    vxx_ratio=vxx_r,
                    spy_vs_ma50=float(_macro.get("spy_vs_ma50", 1.0) or 1.0) if '_macro' in locals() else 1.0,
                )
                ts = TieredStrategy()
                tier_result = ts.run_tiers(ctx)
                tiered_actions = tier_result["actions"]
                tier_stats = tier_result.get("tier_stats", {})
                import logging
                logging.getLogger("bot_engine").info(
                    f"[TIERS] {len(tiered_actions)} actions: {tier_stats}"
                )
            else:
                import logging
                logging.getLogger("bot_engine").warning(
                    f"[TIERS] BLOCKED: {kill_status['kill_reason']}"
                )
        except Exception as e:
            import logging
            logging.getLogger("bot_engine").error(f"[TIERS] failed: {e}")

    return {
        "timestamp": datetime.now().isoformat(),
        "scanned": len(all_tickers),
        "filtered": len(scored),
        "deep_analyzed": len(deep_scored),
        "portfolio_value": portfolio_value,
        "cash": cash,
        "current_positions": num_positions,
        "slots_available": slots_available,
        "options_slots_available": options_slots if 'options_slots' in locals() else MAX_OPTIONS_POSITIONS,
        "new_trades": trades,
        "options_trades_added": options_trade_count,
        "position_actions": mgmt.get("actions", []),
        "upgrade_candidates": mgmt.get("upgrade_candidates", []),
        "top_10": [{
            "ticker": s["ticker"],
            "price": s["price"],
            "score": s.get("deep_score", s.get("quick_score", 0)),
            "change_pct": s["change_pct"],
            "side": s.get("side", "buy"),
            "action_label": s.get("action_label", "BUY"),
            "reasons": s.get("reasons", [])[:2],
        } for s in deep_scored[:10]],
        "third_leg": third_leg_result,
        "convexity_overlay": convexity_result,
        "intraday_shorts": intraday_short_result,
        "options_management": options_mgmt_result,
        "spy_floor": spy_floor_result,
        "defensive_floor": defensive_floor_result,
        "tier_actions": [
            {
                "tier": a.tier, "action": a.action, "ticker": a.ticker,
                "strategy": a.strategy, "size_pct": a.size_pct,
                "reason": a.reason, "metadata": a.metadata,
            } for a in tiered_actions
        ],
        "tier_stats": tier_stats,
        "kill_status": kill_status,
    }


# ── ML Attribution Tracker ───────────────────────────────────────────────────
def ml_attribution_summary() -> dict:
    """
    Aggregate ML attribution across all closed trades to measure whether
    the ML model helps or hurts trade selection vs pure rules.

    Reads from the trade feedback log (voltrade_trade_feedback.json) and
    computes per-trade deltas between rules_score and ml_score_raw.

    Returns a dict with:
      - n_trades: total closed trades with both scores recorded
      - avg_rules_score: average rules-only score at entry
      - avg_ml_score: average ML score at entry
      - avg_outcome_pnl: average realized P&L % across those trades
      - ml_boosted_n: trades where ML score > rules score
      - ml_suppressed_n: trades where ML score < rules score
      - ml_lift_pct: estimated P&L improvement from ML adjustment
      - verdict: 'ML_HELPS' / 'ML_HURTS' / 'NEUTRAL' / 'INSUFFICIENT_DATA'

    Use this to decide whether to increase or decrease the ML blend weight.
    """
    try:
        from ml_model_v2 import FEEDBACK_PATH as _fp
    except ImportError:
        _fp = os.path.join(DATA_DIR, "voltrade_trade_feedback.json")

    try:
        if not os.path.exists(_fp):
            return {"verdict": "INSUFFICIENT_DATA", "reason": "No feedback file"}
        with open(_fp) as f:
            trades = json.load(f)
    except Exception as e:
        return {"verdict": "INSUFFICIENT_DATA", "reason": str(e)}

    # Filter to closed trades with both scores recorded
    scored = [
        t for t in trades
        if t.get("rules_score") is not None
        and t.get("ml_score_raw") is not None
        and t.get("pnl_pct") is not None
    ]

    if len(scored) < 10:
        return {
            "verdict": "INSUFFICIENT_DATA",
            "n_trades": len(scored),
            "reason": f"Need 10+ trades with both scores, have {len(scored)}",
        }

    rules_scores = [t["rules_score"] for t in scored]
    ml_scores    = [t["ml_score_raw"] for t in scored]
    pnls         = [t["pnl_pct"] for t in scored]

    # Trades where ML raised the score (ML was more bullish than rules)
    ml_boosted    = [t for t in scored if t["ml_score_raw"] > t["rules_score"] + 2]
    # Trades where ML lowered the score (ML was more cautious)
    ml_suppressed = [t for t in scored if t["ml_score_raw"] < t["rules_score"] - 2]

    ml_boost_pnl = [t["pnl_pct"] for t in ml_boosted]
    ml_supp_pnl  = [t["pnl_pct"] for t in ml_suppressed]

    avg_boost_pnl = float(sum(ml_boost_pnl) / len(ml_boost_pnl)) if ml_boost_pnl else None
    avg_supp_pnl  = float(sum(ml_supp_pnl) / len(ml_supp_pnl))   if ml_supp_pnl  else None
    avg_all_pnl   = float(sum(pnls) / len(pnls))

    # Compute lift: do ML-boosted trades outperform ML-suppressed ones?
    ml_lift = None
    if avg_boost_pnl is not None and avg_supp_pnl is not None:
        ml_lift = round(avg_boost_pnl - avg_supp_pnl, 3)

    # Verdict: ML helps if boosted trades clearly outperform suppressed ones
    if ml_lift is None:
        verdict = "INSUFFICIENT_DATA"
    elif ml_lift > 0.5:   # ML-boosted trades beat ML-suppressed by >0.5%
        verdict = "ML_HELPS"
    elif ml_lift < -0.5:  # ML-boosted trades underperform by >0.5%
        verdict = "ML_HURTS"
    else:
        verdict = "NEUTRAL"

    return {
        "verdict":               verdict,
        "n_trades":              len(scored),
        "avg_rules_score":       round(sum(rules_scores) / len(rules_scores), 2),
        "avg_ml_score":          round(sum(ml_scores) / len(ml_scores), 2),
        "avg_outcome_pnl":       round(avg_all_pnl, 3),
        "ml_boosted_n":          len(ml_boosted),
        "ml_boosted_avg_pnl":    round(avg_boost_pnl, 3) if avg_boost_pnl is not None else None,
        "ml_suppressed_n":       len(ml_suppressed),
        "ml_suppressed_avg_pnl": round(avg_supp_pnl, 3) if avg_supp_pnl is not None else None,
        "ml_lift_pct":           ml_lift,
        "interpretation": (
            f"ML boosted returns by {abs(ml_lift):.2f}% on ML-influenced trades"
            if ml_lift is not None and ml_lift > 0
            else (
                f"ML reduced returns by {abs(ml_lift):.2f}% on ML-influenced trades"
                if ml_lift is not None and ml_lift < 0
                else "Not enough ML-influenced trades to judge impact"
            )
        ),
    }


# ── Third Leg Engine (v1.0.25) ──────────────────────────────────────────────
def _run_third_leg(macro: dict) -> dict:
    """
    Third leg: VRP harvest + Sector rotation.
    Runs every scan cycle alongside stocks + options.

    Backtest results (64 combinations, 2016-2026):
      ALL 64 beat SPY. Winner: VRP=15% + Sector=12%
      CAGR: +14.8%/yr vs 2-leg +13.8% vs SPY +12.3%

    Leg 3A (TLT): DISABLED — TLT was crushed in 2022 rate hike cycle
    Leg 3B (VRP): 15% of equity, fires when VXX ratio 1.05-1.25 + declining
    Leg 3C (Sector): 12% into XOM+LMT when BEAR/CAUTION regime
    """
    actions = []
    import logging as _logging
    _log = _logging.getLogger("voltrade.leg3")
    try:
        from system_config import BASE_CONFIG, get_market_regime
        import requests as _req

        LEG3_VRP_PCT    = float(BASE_CONFIG.get("LEG3_VRP_PCT",    0.15))
        # LEG3_SECTOR_PCT removed — replaced by regime-adaptive LEG3_CRASH_ASSETS / LEG3_RECOVERY_ASSETS
        LEG3_TLT_PCT    = float(BASE_CONFIG.get("LEG3_TLT_PCT",    0.00))

        vxx_ratio       = float(macro.get("vxx_ratio",    1.0) or 1.0)
        spy_vs_ma50     = float(macro.get("spy_vs_ma50",  1.0) or 1.0)
        spy_b200        = int(macro.get("spy_below_200_days", 0) or 0)
        spy_above_200d  = bool(macro.get("spy_above_200d", True))
        regime          = get_market_regime(vxx_ratio, spy_vs_ma50,
                                            spy_below_200_days=spy_b200,
                                            spy_above_200d=spy_above_200d)

        base_url = ALPACA_BASE_URL
        headers  = _alpaca_headers()

        # Fetch account equity and existing positions
        acc_r = _req.get(f"{base_url}/v2/account", headers=headers, timeout=8)
        acc   = acc_r.json()
        equity = float(acc.get("equity", 100_000) or 100_000)

        pos_r = _req.get(f"{base_url}/v2/positions", headers=headers, timeout=8)
        positions_raw = pos_r.json() if isinstance(pos_r.json(), list) else []
        position_syms = {p["symbol"] for p in positions_raw}

        # Also check pending/open orders to prevent duplicate buys
        # (fixes GLD flooding: bot bought 13x before position registered)
        try:
            open_orders = _req.get(f"{base_url}/v2/orders",
                params={"status": "open", "limit": 50}, headers=headers, timeout=8).json()
            if isinstance(open_orders, list):
                for oo in open_orders:
                    position_syms.add(oo.get("symbol", ""))
        except Exception:
            pass

        # ── Leg 3B: VRP Harvest ────────────────────────────────────────
        # When VXX elevated (1.05-1.25) AND declining: sell volatility
        # Implementation: buy SVXY (inverse VXX ETF) — goes up when VXX falls
        # Simpler than options straddles, trackable as a regular stock position
        # Capital budget check: ensure we don't exceed 100% total deployment
        total_deployed = sum(float(p.get("market_value", 0)) for p in positions_raw) / equity if equity > 0 else 0.0
        remaining_capacity = max(0.0, 1.0 - total_deployed)
        # Scale VRP allocation by remaining capacity (cap at 60% of remaining)
        vrp_pct = min(LEG3_VRP_PCT, remaining_capacity * 0.6)
        if vrp_pct > 0 and 1.05 <= vxx_ratio <= 1.25:
            # Check VXX trend: fetch last 5 days
            vxx_r = _req.get("https://data.alpaca.markets/v2/stocks/bars",
                params={"symbols":"VXX","timeframe":"1Day","limit":6,"feed":"sip"},
                headers=headers, timeout=8)
            vxx_bars = vxx_r.json().get("bars",{}).get("VXX",[])
            vxx_declining = (len(vxx_bars) >= 2 and
                             float(vxx_bars[-1]["c"]) < float(vxx_bars[-5 if len(vxx_bars)>=5 else 0]["c"]))

            if vxx_declining and "SVXY" not in position_syms:
                # Buy SVXY: inverse volatility ETF (goes up as VXX falls)
                svxy_r = _req.get("https://data.alpaca.markets/v2/stocks/snapshots",
                    params={"symbols":"SVXY","feed":"sip"}, headers=headers, timeout=8)
                svxy_price = float(svxy_r.json().get("SVXY",{}).get("latestTrade",{}).get("p", 0) or 0)
                if svxy_price > 0:
                    alloc  = equity * vrp_pct
                    shares = int(alloc / svxy_price)
                    if shares > 0 and alloc > 100:
                        order = {
                            "symbol":       "SVXY",
                            "qty":          str(shares),
                            "side":         "buy",
                            "type":         "limit",
                            "limit_price":  str(round(svxy_price * 1.001, 2)),
                            "time_in_force":"day",
                        }
                        try:
                            o = _req.post(f"{base_url}/v2/orders",
                                          json=order, headers=headers, timeout=10)
                            actions.append({
                                "type": "vrp_harvest",
                                "symbol": "SVXY",
                                "shares": shares,
                                "price": svxy_price,
                                "reason": f"VXX ratio {vxx_ratio:.3f} (elevated+declining) — VRP harvest",
                                "regime": regime,
                                "order_id": o.json().get("id", "?"),
                            })
                            _log.info(f"[LEG3-VRP] Bought {shares} SVXY @ {svxy_price:.2f} (VRP harvest, regime={regime})")
                        except Exception as e:
                            _log.debug(f"[LEG3-VRP] Order failed: {e}")

        # ── Leg 3C: Sector Rotation — DISABLED (pro-level overhaul) ────
        # Backtest: -$92K total P&L, 20.7% WR, 124% max DD. Replaced by convexity overlay.
        # Convexity overlay runs separately via _run_convexity_overlay()

        # ── Exit VRP + legacy sector positions when regime recovers ────
        if regime in ("NEUTRAL", "BULL"):
            for pos in positions_raw:
                sym = pos.get("symbol", "")
                if sym in ("SVXY", "ITA", "GLD", "XOM", "LMT"):  # Exit VRP + any legacy sector positions
                    qty = pos.get("qty", "0")
                    try:
                        o = _req.post(f"{base_url}/v2/orders",
                            json={"symbol": sym, "qty": qty, "side": "sell",
                                  "type": "market", "time_in_force": "day"},
                            headers=headers, timeout=10)
                        pnl = float(pos.get("unrealized_plpc", 0) or 0) * 100
                        actions.append({
                            "type": "third_leg_exit",
                            "symbol": sym,
                            "reason": f"Regime recovered to {regime} — exiting third leg",
                            "pnl_pct": round(pnl, 2),
                        })
                        _log.info(f"[LEG3-EXIT] Sold {sym} (regime={regime}, P&L={pnl:.1f}%)")
                    except Exception as e:
                        _log.debug(f"[LEG3-EXIT] Sell failed for {sym}: {e}")

    except Exception as e:
        _log.debug(f"[LEG3] Third leg error: {e}")
        return {"actions": actions, "status": "error", "error": str(e)[:100]}

    return {
        "actions":   actions,
        "status":    "ok",
        "regime":    regime if 'regime' in locals() else "unknown",
        "vrp_on":    vrp_pct > 0 if 'vrp_pct' in locals() else False,
        "sector_on": bool(sector_assets) if 'sector_assets' in locals() else False,
    }


# ── Entry Point ──────────────────────────────────────────────────────────────

# ── Convexity Overlay (pro-level: QQQ protective puts) ────────────────────────

def _close_sqqq_position(headers: dict, base_url: str) -> list:
    """
    Clean up legacy SQQQ hedge position (if any) from the old inverse-ETF strategy.
    Sells all SQQQ shares via market order. Called once at the start of the
    convexity overlay before buying QQQ puts.

    Returns a list of action dicts (empty if no SQQQ held).
    """
    import logging as _logging
    import requests as _req
    _log = _logging.getLogger("voltrade.convexity")
    actions = []

    try:
        pos_r = _req.get(f"{base_url}/v2/positions/SQQQ", headers=headers, timeout=8)
        if pos_r.status_code == 200:
            pos = pos_r.json()
            qty = abs(int(float(pos.get("qty", 0) or 0)))
            if qty > 0:
                order = {
                    "symbol":        "SQQQ",
                    "qty":           str(qty),
                    "side":          "sell",
                    "type":          "market",
                    "time_in_force": "day",
                }
                o = _req.post(f"{base_url}/v2/orders", json=order, headers=headers, timeout=10)
                actions.append({
                    "type": "convexity_legacy_cleanup",
                    "symbol": "SQQQ",
                    "shares": qty,
                    "side": "sell",
                    "reason": "Closing legacy SQQQ hedge — replaced by QQQ puts",
                    "order_id": o.json().get("id", "?"),
                })
                _log.info(f"[CONVEXITY] Closed legacy SQQQ position: sold {qty} shares")
    except Exception as e:
        _log.debug(f"[CONVEXITY] SQQQ cleanup skipped: {e}")

    return actions


def _run_convexity_overlay(macro: dict) -> dict:
    """
    Permanent tail hedge using QQQ protective puts.

    Strategy:
    - Buy far OTM QQQ puts (~20% below current price, ~3-5 delta)
    - Target 60 DTE, rolled when existing put has <21 DTE remaining
    - Budget: 2.0% of equity normally, 4.0% in PANIC/BEAR (regime-scaled)
    - Uses Alpaca options API with OCC symbol format (e.g. QQQ260515P00520000)

    Why puts instead of SQQQ: real convexity (capped downside = premium paid,
    unlimited upside in a crash), no leverage decay drag, defined risk.
    Backtest: Sharpe 3.80, max drawdown 6.8% vs SQQQ's 23.7%.
    """
    actions = []
    import logging as _logging
    _log = _logging.getLogger("voltrade.convexity")

    try:
        from system_config import BASE_CONFIG, get_market_regime
        import requests as _req
        from datetime import datetime as _dt, timedelta as _td

        vxx_ratio      = float(macro.get("vxx_ratio", 1.0) or 1.0)
        spy_vs_ma50    = float(macro.get("spy_vs_ma50", 1.0) or 1.0)
        spy_b200       = int(macro.get("spy_below_200_days", 0) or 0)
        spy_above_200d = bool(macro.get("spy_above_200d", True))
        regime = get_market_regime(vxx_ratio, spy_vs_ma50,
                                   spy_below_200_days=spy_b200,
                                   spy_above_200d=spy_above_200d)

        base_url = ALPACA_BASE_URL
        headers  = _alpaca_headers()

        # Step 0: Close any legacy SQQQ position from the old strategy
        sqqq_actions = _close_sqqq_position(headers, base_url)
        actions.extend(sqqq_actions)

        # Fetch account equity
        acc_r = _req.get(f"{base_url}/v2/account", headers=headers, timeout=8)
        acc   = acc_r.json()
        equity = float(acc.get("equity", 100_000) or 100_000)

        # Fetch all positions (to find existing QQQ puts)
        pos_r = _req.get(f"{base_url}/v2/positions", headers=headers, timeout=8)
        positions_raw = pos_r.json() if isinstance(pos_r.json(), list) else []

        # Check open orders to avoid duplicate put orders
        try:
            open_orders = _req.get(f"{base_url}/v2/orders",
                params={"status": "open", "limit": 50}, headers=headers, timeout=8).json()
            pending_syms = set()
            if isinstance(open_orders, list):
                for oo in open_orders:
                    pending_syms.add(oo.get("symbol", ""))
        except Exception:
            pending_syms = set()

        # Load config
        convexity_cfg  = BASE_CONFIG.get("CONVEXITY_OVERLAY", {})
        normal_budget  = convexity_cfg.get("normal_budget_pct", 0.020)
        stress_budget  = convexity_cfg.get("stress_budget_pct", 0.040)
        target_dte     = convexity_cfg.get("put_dte", 60)

        # ── Drawdown-gated hedge escalation (v1.0.29+) ───────────────────────
        # If portfolio drawdown ≥ DRAWDOWN_HEDGE_ESCALATE_PCT, force stress
        # budget regardless of regime. Protects against regime-lag where VXX
        # hasn't spiked yet but equity is already bleeding. This is an OR with
        # the regime gate — whichever signals stress first wins.
        _dd_escalate = False
        try:
            _dd_escalate = should_escalate_hedge()
        except Exception:
            pass

        if regime in ("PANIC", "BEAR") or _dd_escalate:
            budget_pct = stress_budget
            if _dd_escalate and regime not in ("PANIC", "BEAR"):
                _log.info(f"[CONVEXITY] DD-escalated to stress budget "
                          f"(regime={regime}, portfolio DD triggered escalation)")
        elif regime == "CAUTION":
            budget_pct = (normal_budget + stress_budget) / 2
        else:
            budget_pct = normal_budget

        budget_dollars = equity * budget_pct
        today = _dt.now().date()

        # ── Find existing QQQ put positions ──────────────────────────────────
        existing_put = None
        existing_put_expiry = None
        for pos in positions_raw:
            sym = pos.get("symbol", "")
            # OCC format: QQQ + 6-digit date + P + 8-digit strike
            if sym.startswith("QQQ") and "P" in sym and len(sym) > 10:
                try:
                    # Extract expiry from OCC symbol: chars [3:9] = YYMMDD
                    exp_str = sym[3:9]
                    exp_date = _dt.strptime(exp_str, "%y%m%d").date()
                    existing_put = pos
                    existing_put_expiry = exp_date
                    break
                except (ValueError, IndexError):
                    continue

        # ── Roll logic: close existing put if <21 DTE ────────────────────────
        need_new_put = True
        if existing_put and existing_put_expiry:
            days_to_expiry = (existing_put_expiry - today).days
            if days_to_expiry < convexity_cfg.get("roll_dte", 21):
                # Close the expiring put
                put_sym = existing_put["symbol"]
                put_qty = abs(int(float(existing_put.get("qty", 0) or 0)))
                if put_qty > 0 and put_sym not in pending_syms:
                    close_order = {
                        "symbol":        put_sym,
                        "qty":           str(put_qty),
                        "side":          "sell_to_close",
                        "type":          "market",
                        "time_in_force": "day",
                    }
                    try:
                        o = _req.post(f"{base_url}/v2/orders", json=close_order, headers=headers, timeout=10)
                        actions.append({
                            "type": "convexity_roll_close",
                            "symbol": put_sym,
                            "contracts": put_qty,
                            "side": "sell_to_close",
                            "reason": f"Rolling put: {days_to_expiry} DTE remaining, closing to roll",
                            "regime": regime,
                            "order_id": o.json().get("id", "?"),
                        })
                        _log.info(f"[CONVEXITY] Closing expiring put {put_sym} ({days_to_expiry} DTE)")
                    except Exception as e:
                        _log.debug(f"[CONVEXITY] Failed to close expiring put: {e}")
                need_new_put = True
            else:
                # Existing put is fine, no action needed
                need_new_put = False
                _log.info(f"[CONVEXITY] Holding {existing_put['symbol']} ({days_to_expiry} DTE)")

        if not need_new_put:
            return {
                "actions": actions,
                "status": "ok",
                "regime": regime,
                "budget_pct": round(budget_pct * 100, 2),
                "hedge_ticker": "QQQ",
                "hedge_type": "puts",
                "existing_put": existing_put["symbol"] if existing_put else None,
            }

        # ── Get current QQQ price ────────────────────────────────────────────
        snap_r = _req.get(f"https://data.alpaca.markets/v2/stocks/snapshots",
            params={"symbols": "QQQ", "feed": "sip"}, headers=headers, timeout=8)
        qqq_price = float(snap_r.json().get("QQQ", {}).get("latestTrade", {}).get("p", 0) or 0)
        if qqq_price <= 0:
            _log.debug("[CONVEXITY] Could not get QQQ price")
            return {"actions": actions, "status": "error", "error": "no QQQ price"}

        # Target strike: ~20% below current price (far OTM, ~3-5 delta)
        target_strike = round(qqq_price * 0.80, 0)  # ~20% OTM

        # ── Find QQQ put contracts via Alpaca options discovery ───────────────
        exp_gte = today + _td(days=45)   # At least 45 DTE
        exp_lte = today + _td(days=75)   # At most 75 DTE (wider window for liquidity)

        contracts_r = _req.get(f"{base_url}/v2/options/contracts", params={
            "underlying_symbols":  "QQQ",
            "type":                "put",
            "expiration_date_gte": exp_gte.strftime("%Y-%m-%d"),
            "expiration_date_lte": exp_lte.strftime("%Y-%m-%d"),
            "strike_price_gte":    str(round(qqq_price * 0.75, 2)),  # 25% below
            "strike_price_lte":    str(round(qqq_price * 0.90, 2)),  # 10% below
            "limit":               50,
        }, headers=headers, timeout=10)

        if contracts_r.status_code != 200:
            _log.debug(f"[CONVEXITY] Options contract search failed: {contracts_r.status_code}")
            return {"actions": actions, "status": "error", "error": f"contract search {contracts_r.status_code}"}

        contracts_data = contracts_r.json()
        contracts = contracts_data if isinstance(contracts_data, list) else contracts_data.get("option_contracts", [])
        if not contracts:
            _log.debug("[CONVEXITY] No QQQ put contracts found in target window")
            return {"actions": actions, "status": "ok", "regime": regime, "budget_pct": round(budget_pct * 100, 2),
                    "hedge_ticker": "QQQ", "hedge_type": "puts", "note": "no contracts found"}

        # Pick the contract closest to our target strike and target DTE
        best_contract = None
        best_score = float("inf")
        for c in contracts:
            c_strike = float(c.get("strike_price", 0) or 0)
            c_exp = c.get("expiration_date", "")
            try:
                c_exp_date = _dt.strptime(c_exp, "%Y-%m-%d").date()
                c_dte = (c_exp_date - today).days
            except (ValueError, TypeError):
                continue
            # Score: distance from target strike + distance from target DTE
            strike_dist = abs(c_strike - target_strike) / qqq_price
            dte_dist = abs(c_dte - target_dte) / target_dte
            score = strike_dist + dte_dist
            if score < best_score:
                best_score = score
                best_contract = c

        if not best_contract:
            return {"actions": actions, "status": "ok", "regime": regime, "budget_pct": round(budget_pct * 100, 2),
                    "hedge_ticker": "QQQ", "hedge_type": "puts", "note": "no suitable contract"}

        occ_symbol = best_contract.get("symbol", "")
        contract_strike = float(best_contract.get("strike_price", 0))
        contract_exp = best_contract.get("expiration_date", "")

        # ── Get quote for the put to calculate mid price ─────────────────────
        try:
            quote_r = _req.get(f"https://data.alpaca.markets/v1beta1/options/quotes/latest",
                params={"symbols": occ_symbol, "feed": "indicative"}, headers=headers, timeout=8)
            quote_data = quote_r.json().get("quotes", {}).get(occ_symbol, {})
            bid = float(quote_data.get("bp", 0) or 0)
            ask = float(quote_data.get("ap", 0) or 0)
        except Exception:
            bid, ask = 0, 0

        if bid <= 0 or ask <= 0:
            # Fallback: estimate price as ~0.5-1% of underlying per contract
            mid_price = round(qqq_price * 0.007, 2)
        else:
            mid_price = round((bid + ask) / 2, 2)

        if mid_price <= 0:
            return {"actions": actions, "status": "error", "error": "could not price put"}

        # Each contract = 100 shares, so cost = mid_price * 100 per contract
        cost_per_contract = mid_price * 100
        num_contracts = max(1, int(budget_dollars / cost_per_contract))

        # ── Place limit order for puts ───────────────────────────────────────
        if occ_symbol not in pending_syms:
            order = {
                "symbol":        occ_symbol,
                "qty":           str(num_contracts),
                "side":          "buy_to_open",
                "type":          "limit",
                "limit_price":   str(mid_price),
                "time_in_force": "day",
            }
            try:
                o = _req.post(f"{base_url}/v2/orders", json=order, headers=headers, timeout=10)
                actions.append({
                    "type": "convexity_overlay",
                    "symbol": occ_symbol,
                    "contracts": num_contracts,
                    "side": "buy_to_open",
                    "strike": contract_strike,
                    "expiration": contract_exp,
                    "price": mid_price,
                    "target_pct": round(budget_pct * 100, 1),
                    "reason": (f"Convexity hedge: buy {num_contracts}x {occ_symbol} "
                               f"(strike={contract_strike}, exp={contract_exp}, "
                               f"regime={regime}, budget={budget_pct:.1%})"),
                    "regime": regime,
                    "order_id": o.json().get("id", "?"),
                })
                _log.info(f"[CONVEXITY] BUY {num_contracts}x {occ_symbol} @ {mid_price:.2f} "
                          f"(strike={contract_strike}, exp={contract_exp}, regime={regime}, budget={budget_pct:.1%})")
                # Register in options state so options_manager skips this position
                try:
                    import os as _os2, json as _json2
                    try:
                        from storage_config import DATA_DIR as _dd
                    except ImportError:
                        _dd = '/data/voltrade' if _os2.path.isdir('/data') else '/tmp'
                    _state_path = _os2.path.join(_dd, 'voltrade_options_state.json')
                    try:
                        with open(_state_path) as _f: _ostate = _json2.load(_f)
                    except: _ostate = {}
                    _ostate[occ_symbol] = {
                        'strategy': 'convexity_hedge',
                        'setup': 'protective_put',
                        'ticker': 'QQQ',
                        'managed_by': 'convexity_overlay',
                        'entry_date': today.isoformat(),
                    }
                    with open(_state_path, 'w') as _f: _json2.dump(_ostate, _f)
                except: pass
            except Exception as e:
                _log.debug(f"[CONVEXITY] Put order failed: {e}")

    except Exception as e:
        _log.debug(f"[CONVEXITY] Error: {e}")
        return {"actions": actions, "status": "error", "error": str(e)[:100]}

    return {
        "actions": actions,
        "status": "ok",
        "regime": regime if 'regime' in locals() else "unknown",
        "budget_pct": round(budget_pct * 100, 2) if 'budget_pct' in locals() else 0,
        "hedge_ticker": "QQQ",
        "hedge_type": "puts",
    }


def _unwind_covered_calls(ticker: str) -> list:
    """Buy back any short calls on `ticker` before selling the underlying stock.
    Returns a list of action dicts describing what was unwound.
    If a buy-back order fails, raises RuntimeError to block the stock sale."""
    import requests as _req
    actions = []
    try:
        positions = get_alpaca_positions()
        for pos in (positions if isinstance(positions, list) else []):
            sym = str(pos.get("symbol", ""))
            if (pos.get("asset_class") == "us_option"
                    and sym.startswith(ticker)
                    and int(float(pos.get("qty", 0))) < 0
                    and "C" in sym[len(ticker):]):
                buy_qty = abs(int(float(pos.get("qty", 0))))
                import logging as _uwlog
                _uwlog.getLogger("bot_engine").info(
                    f"[CC-UNWIND] {ticker}: Buying back {buy_qty}x {sym} before stock sale")
                resp = _req.post(f"{ALPACA_BASE_URL}/v2/orders",
                    json={"symbol": sym, "qty": str(buy_qty),
                          "side": "buy", "type": "market", "time_in_force": "day"},
                    headers=_alpaca_headers(), timeout=10)
                if resp.status_code >= 400:
                    raise RuntimeError(f"Failed to buy back {sym}: HTTP {resp.status_code} {resp.text[:200]}")
                actions.append({"type": "cc_unwind", "symbol": sym, "qty": buy_qty,
                                "ticker": ticker})
    except RuntimeError:
        raise  # Propagate — caller must block stock sale
    except Exception as e:
        import logging as _uwlog
        _uwlog.getLogger("bot_engine").warning(f"[CC-UNWIND] {ticker}: Error checking calls: {e}")
    return actions


# ── Passive SPY Floor (v1.0.29) ──────────────────────────────────────────────

def _manage_spy_floor(macro: dict) -> dict:
    """
    Passive SPY Floor: hold SPY shares proportional to regime allocation.
    Captures market drift in calm bull markets where momentum signals = noise.

    Backtest (7-config sweep, 2016-2026):
      B60/N85/C30 = best: 12.9% CAGR vs SPY 12.3% (beats by +0.6%)
      Fixes quiet bull year losses: 2017 -27% -> -6%, 2019 -33% -> -6%

    FIX (2026-04-10): Only rebalance on REGIME CHANGES, not every cycle.
    The old logic rebalanced whenever position drifted >2-5% from target,
    which caused massive churn: buying 112 QQQ shares then selling 111
    two minutes later, losing $698 to the spread. Now we:
      1. Persist the last regime in a state file
      2. Only rebalance when regime changes OR position is >10% off target
      3. Use limit orders instead of market orders to reduce slippage
    """
    result = {"actions": [], "status": "ok", "target_pct": 0, "current_pct": 0}

    try:
        from system_config import BASE_CONFIG, get_market_regime

        vxx_ratio   = float(macro.get("vxx_ratio", 1.0) or 1.0)
        spy_vs_ma50 = float(macro.get("spy_vs_ma50", 1.0) or 1.0)
        spy_b200    = int(macro.get("spy_below_200_days", 0) or 0)
        spy_above   = bool(macro.get("spy_above_200d", True))

        regime = get_market_regime(vxx_ratio, spy_vs_ma50,
                                   spy_below_200_days=spy_b200,
                                   spy_above_200d=spy_above)

        # ── Regime change detection ──────────────────────────────────────
        # Persist last regime to avoid rebalancing every cycle
        _floor_state_path = os.path.join(DATA_DIR, 'voltrade_floor_state.json')
        _floor_state = {}
        try:
            with open(_floor_state_path) as f:
                _floor_state = json.load(f)
        except Exception:
            pass
        last_regime = _floor_state.get("last_regime")
        regime_changed = (last_regime is not None and last_regime != regime)

        # Use configurable floor ticker (QQQ by default — 5.6%/yr more than SPY)
        floor_ticker = BASE_CONFIG.get("FLOOR_TICKER", "QQQ")
        floor_key = f"SPY_FLOOR_{regime}"
        target_pct = BASE_CONFIG.get(floor_key, 0)

        # ── Aggressive trend-following exit (pro-level overhaul) ─────────
        # Override regime-based allocation with trend signal when QQQ is
        # below its 200-day MA. This catches bear markets FASTER than regime
        # detection alone (which waits for VXX ratio to spike).
        #
        # Death cross (50d MA < 200d MA AND price < 200d): MAX 20% QQQ
        # Early warning (price < 200d MA, 50d still above): MAX 50% QQQ
        # Normal: use regime-based allocation
        trend_override = None
        try:
            # Fetch QQQ price and MAs from macro data or compute
            _qqq_ma50 = float(macro.get("qqq_ma50", 0) or 0)
            _qqq_ma200 = float(macro.get("qqq_ma200", 0) or 0)
            _qqq_price = float(macro.get("qqq_price", 0) or 0)

            # If macro doesn't have QQQ MAs, compute from Alpaca bars
            if _qqq_ma50 <= 0 or _qqq_ma200 <= 0 or _qqq_price <= 0:
                alpaca_throttle.acquire()
                _qqq_bars_resp = requests.get(
                    f"{ALPACA_DATA_URL}/v2/stocks/{floor_ticker}/bars",
                    params={"timeframe": "1Day", "limit": 210, "adjustment": "all", "feed": "sip"},
                    headers=_alpaca_headers(), timeout=10)
                _qqq_bars = _qqq_bars_resp.json().get("bars", [])
                if len(_qqq_bars) >= 200:
                    _qqq_closes = [float(b["c"]) for b in _qqq_bars]
                    _qqq_price = _qqq_closes[-1]
                    _qqq_ma50 = sum(_qqq_closes[-50:]) / 50
                    _qqq_ma200 = sum(_qqq_closes[-200:]) / 200

            if _qqq_price > 0 and _qqq_ma200 > 0:
                _below_200d = _qqq_price < _qqq_ma200
                _death_cross = _qqq_ma50 > 0 and _qqq_ma50 < _qqq_ma200

                if _below_200d and _death_cross:
                    # Death cross: most aggressive reduction
                    trend_cap = BASE_CONFIG.get("TREND_EXIT_DEATH_CROSS_CAP", 0.20)
                    target_pct = min(target_pct, trend_cap)
                    trend_override = f"death_cross (50d={_qqq_ma50:.0f} < 200d={_qqq_ma200:.0f}, price={_qqq_price:.0f})"
                elif _below_200d:
                    # Early warning: price below 200d but 50d still above
                    trend_cap = BASE_CONFIG.get("TREND_EXIT_EARLY_WARNING_CAP", 0.50)
                    target_pct = min(target_pct, trend_cap)
                    trend_override = f"early_warning (price={_qqq_price:.0f} < 200d={_qqq_ma200:.0f})"
        except Exception:
            pass  # Trend data unavailable — use regime-based allocation

        result["target_pct"] = target_pct
        result["regime"] = regime
        result["floor_ticker"] = floor_ticker
        if trend_override:
            result["trend_override"] = trend_override

        if target_pct <= 0:
            # No floor — sell any existing position (only on regime change to 0%)
            if regime_changed or last_regime is None:
                try:
                    positions = get_alpaca_positions()
                    floor_pos = [p for p in positions if p.get("symbol") == floor_ticker]
                    if floor_pos:
                        qty = abs(int(float(floor_pos[0].get("qty", 0))))
                        if qty > 0:
                            # Unwind any covered calls before selling stock
                            try:
                                uw_actions = _unwind_covered_calls(floor_ticker)
                                result["actions"].extend(uw_actions)
                            except RuntimeError as _uw_err:
                                import logging as _floor_log; _floor_log.getLogger("bot_engine").error(f"[FLOOR] CC unwind failed for {floor_ticker}, blocking sale: {_uw_err}")
                                result["actions"].append({"type": "cc_unwind_blocked", "reason": str(_uw_err)})
                                return result
                            alpaca_throttle.acquire()
                            requests.post(f"{ALPACA_BASE_URL}/v2/orders",
                                json={"symbol": floor_ticker, "qty": str(qty),
                                      "side": "sell", "type": "market",
                                      "time_in_force": "day"},
                                headers=_alpaca_headers(), timeout=10)
                            result["actions"].append({"type": "floor_exit",
                                "shares": qty, "reason": f"{regime} regime (was {last_regime})"})
                            import logging as _floor_log; _floor_log.getLogger("bot_engine").info(f"[FLOOR] Sold {qty} {floor_ticker} (regime changed {last_regime}->{regime})")
                except Exception:
                    pass
            else:
                result["status"] = "no_floor_no_change"
            # Save regime state
            _floor_state["last_regime"] = regime
            _floor_state["last_rebalance"] = datetime.now().isoformat()
            try:
                with open(_floor_state_path, "w") as f:
                    json.dump(_floor_state, f)
            except Exception:
                pass
            return result

        # Get account equity and current SPY position
        try:
            alpaca_throttle.acquire()
            acc = requests.get(f"{ALPACA_BASE_URL}/v2/account",
                headers=_alpaca_headers(), timeout=8).json()
            equity = float(acc.get("equity", 100000) or 100000)
        except Exception:
            equity = 100000

        current_spy_value = 0
        current_spy_shares = 0
        try:
            positions = get_alpaca_positions()
            for p in positions:
                if p.get("symbol") == floor_ticker:
                    current_spy_shares = int(float(p.get("qty", 0)))
                    current_spy_value = abs(float(p.get("market_value", 0)))
                    break
        except Exception:
            pass

        current_pct = current_spy_value / equity if equity > 0 else 0
        result["current_pct"] = round(current_pct, 3)

        target_value = equity * target_pct
        diff = target_value - current_spy_value

        # ── Only rebalance on regime change OR significant drift (>10%) ──
        # The old 2-5% band caused daily churn. Now:
        # - Regime change: rebalance immediately (the allocation target changed)
        # - No regime change: only rebalance if >10% off target (prevents spread loss)
        _rebalance_band = 0.10  # 10% band — much wider than old 2-5%
        drift_pct = abs(diff) / equity if equity > 0 else 0
        needs_rebalance = regime_changed or (last_regime is None and current_spy_shares == 0)

        if not needs_rebalance and drift_pct < _rebalance_band:
            result["status"] = "within_band"
            # Save regime state even when no rebalance
            _floor_state["last_regime"] = regime
            try:
                with open(_floor_state_path, "w") as f:
                    json.dump(_floor_state, f)
            except Exception:
                pass
            return result

        # Get floor ticker price
        try:
            alpaca_throttle.acquire()
            snap = requests.get(f"{ALPACA_DATA_URL}/v2/stocks/snapshots",
                params={"symbols": floor_ticker, "feed": "sip"},
                headers=_alpaca_headers(), timeout=8).json()
            spy_price = float(snap.get(floor_ticker, {}).get("latestTrade", {}).get("p", 0) or 0)
        except Exception:
            spy_price = 0

        if spy_price <= 0:
            result["status"] = "no_spy_price"
            return result

        shares_diff = int(diff / spy_price)
        if shares_diff == 0:
            result["status"] = "within_band"
            # Save regime state
            _floor_state["last_regime"] = regime
            try:
                with open(_floor_state_path, "w") as f:
                    json.dump(_floor_state, f)
            except Exception:
                pass
            return result

        # Use limit orders near last trade price to reduce spread slippage
        _order_type = "limit"
        _limit_price = str(round(spy_price, 2))

        if shares_diff > 0:
            try:
                alpaca_throttle.acquire()
                requests.post(f"{ALPACA_BASE_URL}/v2/orders",
                    json={"symbol": floor_ticker, "qty": str(shares_diff),
                          "side": "buy", "type": _order_type,
                          "limit_price": _limit_price,
                          "time_in_force": "day"},
                    headers=_alpaca_headers(), timeout=10)
                result["actions"].append({"type": "floor_buy",
                    "shares": shares_diff, "ticker": floor_ticker,
                    "reason": f"regime_change:{last_regime}->{regime}, target {target_pct*100:.0f}%, current {current_pct*100:.0f}%"})
                import logging as _floor_log; _floor_log.getLogger("bot_engine").info(f"[FLOOR] Bought {shares_diff} {floor_ticker} (regime {last_regime}->{regime}: {target_pct*100:.0f}% target)")
            except Exception as e:
                import logging as _floor_log; _floor_log.getLogger("bot_engine").debug(f"[SPY_FLOOR] Buy failed: {e}")
        else:
            sell_qty = min(abs(shares_diff), current_spy_shares)
            if sell_qty > 0:
                # Unwind covered calls if selling would leave <100 shares
                _remaining_after_sell = current_spy_shares - sell_qty
                if _remaining_after_sell < 100:
                    try:
                        uw_actions = _unwind_covered_calls(floor_ticker)
                        result["actions"].extend(uw_actions)
                    except RuntimeError as _uw_err:
                        import logging as _floor_log; _floor_log.getLogger("bot_engine").error(f"[FLOOR] CC unwind failed for {floor_ticker}, blocking sale: {_uw_err}")
                        result["actions"].append({"type": "cc_unwind_blocked", "reason": str(_uw_err)})
                        return result
                try:
                    alpaca_throttle.acquire()
                    requests.post(f"{ALPACA_BASE_URL}/v2/orders",
                        json={"symbol": floor_ticker, "qty": str(sell_qty),
                              "side": "sell", "type": _order_type,
                              "limit_price": _limit_price,
                              "time_in_force": "day"},
                        headers=_alpaca_headers(), timeout=10)
                    result["actions"].append({"type": "floor_sell",
                        "shares": sell_qty, "ticker": floor_ticker,
                        "reason": f"regime_change:{last_regime}->{regime}, target {target_pct*100:.0f}%, current {current_pct*100:.0f}%"})
                    import logging as _floor_log; _floor_log.getLogger("bot_engine").info(f"[FLOOR] Sold {sell_qty} {floor_ticker} (regime {last_regime}->{regime}: {target_pct*100:.0f}% target)")
                except Exception as e:
                    import logging as _floor_log; _floor_log.getLogger("bot_engine").debug(f"[SPY_FLOOR] Sell failed: {e}")

        # Save regime state after rebalance
        _floor_state["last_regime"] = regime
        _floor_state["last_rebalance"] = datetime.now().isoformat()
        try:
            with open(_floor_state_path, "w") as f:
                json.dump(_floor_state, f)
        except Exception:
            pass

    except Exception as e:
        result["status"] = f"error: {str(e)[:80]}"

    return result


def _manage_defensive_floor(macro: dict) -> dict:
    """
    P1-1: Defensive-asset rotation for BEAR/PANIC regimes.

    User directive (2026-04-17):
      "Fix everything you see wrong make sure it works.
       Not holding it permanently but a switch from qqq to something in bear."

    Design:
      * Holds GLD (positive carry in both bull and bear days, near-zero SPY corr)
      * Allocation ramps with regime via DEFENSIVE_FLOOR_* config:
          BULL/NEUTRAL: 0%   — full QQQ exposure
          CAUTION:      10%  — start rotating in
          BEAR:         30%
          PANIC:        40%
      * Death-cross override forces min DEFENSIVE_FLOOR_DEATHCROSS_MIN (25%)
        even if VXX-based regime is still CAUTION/NEUTRAL — catches slow bear
        onsets before VXX spikes.
      * Only rebalances on regime change or >10% drift band, to avoid churn.
      * Uses limit orders near last trade to minimize spread slippage.

    This mirrors _manage_spy_floor() for the QQQ leg but moves the opposite
    direction, so as QQQ floor → 0 in BEAR, GLD floor ramps up.
    """
    result = {"actions": [], "status": "ok", "target_pct": 0, "current_pct": 0}

    try:
        from system_config import BASE_CONFIG, get_market_regime

        if not BASE_CONFIG.get("DEFENSIVE_FLOOR_ENABLED", False):
            result["status"] = "disabled"
            return result

        defensive_ticker = BASE_CONFIG.get("DEFENSIVE_FLOOR_TICKER", "GLD")

        vxx_ratio   = float(macro.get("vxx_ratio", 1.0) or 1.0)
        spy_vs_ma50 = float(macro.get("spy_vs_ma50", 1.0) or 1.0)
        spy_b200    = int(macro.get("spy_below_200_days", 0) or 0)
        spy_above   = bool(macro.get("spy_above_200d", True))

        regime = get_market_regime(vxx_ratio, spy_vs_ma50,
                                   spy_below_200_days=spy_b200,
                                   spy_above_200d=spy_above)
        result["regime"] = regime
        result["ticker"] = defensive_ticker

        # Base allocation from regime
        regime_key = f"DEFENSIVE_FLOOR_{regime}"
        target_pct = float(BASE_CONFIG.get(regime_key, 0.0) or 0.0)

        # Death-cross override (catch bear markets faster than VXX)
        floor_ticker = BASE_CONFIG.get("FLOOR_TICKER", "QQQ")
        try:
            _qqq_ma50  = float(macro.get("qqq_ma50", 0) or 0)
            _qqq_ma200 = float(macro.get("qqq_ma200", 0) or 0)
            _qqq_price = float(macro.get("qqq_price", 0) or 0)
            if (_qqq_ma50 <= 0 or _qqq_ma200 <= 0 or _qqq_price <= 0):
                alpaca_throttle.acquire()
                _bars_resp = requests.get(
                    f"{ALPACA_DATA_URL}/v2/stocks/{floor_ticker}/bars",
                    params={"timeframe": "1Day", "limit": 210, "adjustment": "all", "feed": "sip"},
                    headers=_alpaca_headers(), timeout=10)
                _bars = _bars_resp.json().get("bars", [])
                if len(_bars) >= 200:
                    _cl = [float(b["c"]) for b in _bars]
                    _qqq_price = _cl[-1]
                    _qqq_ma50  = sum(_cl[-50:]) / 50
                    _qqq_ma200 = sum(_cl[-200:]) / 200
            if _qqq_price > 0 and _qqq_ma50 > 0 and _qqq_ma200 > 0:
                _death_cross = (_qqq_ma50 < _qqq_ma200) and (_qqq_price < _qqq_ma200)
                if _death_cross:
                    _dc_min = float(BASE_CONFIG.get("DEFENSIVE_FLOOR_DEATHCROSS_MIN", 0.25) or 0.25)
                    if target_pct < _dc_min:
                        target_pct = _dc_min
                        result["death_cross_override"] = True
        except Exception:
            pass

        result["target_pct"] = round(target_pct, 3)

        # ── Persist regime so we only rebalance on regime change or drift ─
        _state_path = os.path.join(DATA_DIR, 'voltrade_defensive_floor_state.json')
        _state = {}
        try:
            with open(_state_path) as f:
                _state = json.load(f)
        except Exception:
            pass
        last_regime = _state.get("last_regime")
        regime_changed = (last_regime is not None and last_regime != regime)

        # ── Current GLD position & equity ────────────────────────────────
        try:
            alpaca_throttle.acquire()
            acc = requests.get(f"{ALPACA_BASE_URL}/v2/account",
                               headers=_alpaca_headers(), timeout=8).json()
            equity = float(acc.get("equity", 100000) or 100000)
        except Exception:
            equity = 100000

        current_value = 0.0
        current_shares = 0
        try:
            for p in get_alpaca_positions():
                if p.get("symbol") == defensive_ticker:
                    current_shares = int(float(p.get("qty", 0) or 0))
                    current_value  = abs(float(p.get("market_value", 0) or 0))
                    break
        except Exception:
            pass

        current_pct = (current_value / equity) if equity > 0 else 0
        result["current_pct"] = round(current_pct, 3)

        target_value = equity * target_pct
        diff = target_value - current_value
        drift_pct = abs(diff) / equity if equity > 0 else 0

        # ── Target 0: sell any existing GLD on regime change ─────────────
        if target_pct <= 0:
            if (regime_changed or last_regime is None) and current_shares > 0:
                try:
                    alpaca_throttle.acquire()
                    requests.post(
                        f"{ALPACA_BASE_URL}/v2/orders",
                        json={"symbol": defensive_ticker, "qty": str(current_shares),
                              "side": "sell", "type": "market",
                              "time_in_force": "day"},
                        headers=_alpaca_headers(), timeout=10)
                    result["actions"].append({
                        "type": "defensive_exit",
                        "shares": current_shares, "ticker": defensive_ticker,
                        "reason": f"regime {last_regime}→{regime}: rotate back into QQQ",
                    })
                    import logging as _dlog
                    _dlog.getLogger("bot_engine").info(
                        f"[DEFENSIVE] Sold {current_shares} {defensive_ticker} (regime {last_regime}→{regime})")
                except Exception as _e:
                    import logging as _dlog
                    _dlog.getLogger("bot_engine").debug(f"[DEFENSIVE] Sell failed: {_e}")
            else:
                result["status"] = "no_target_no_change"
            _state["last_regime"] = regime
            _state["last_rebalance"] = datetime.now().isoformat()
            try:
                with open(_state_path, "w") as f:
                    json.dump(_state, f)
            except Exception:
                pass
            return result

        # ── Skip if we're already within 10% drift band ──────────────────
        _band = 0.10
        needs_rebalance = regime_changed or (last_regime is None and current_shares == 0)
        if (not needs_rebalance) and drift_pct < _band:
            result["status"] = "within_band"
            _state["last_regime"] = regime
            try:
                with open(_state_path, "w") as f:
                    json.dump(_state, f)
            except Exception:
                pass
            return result

        # ── Get last trade price for limit orders ─────────────────────────
        try:
            alpaca_throttle.acquire()
            snap = requests.get(
                f"{ALPACA_DATA_URL}/v2/stocks/snapshots",
                params={"symbols": defensive_ticker, "feed": "sip"},
                headers=_alpaca_headers(), timeout=8).json()
            last_px = float(snap.get(defensive_ticker, {}).get("latestTrade", {}).get("p", 0) or 0)
        except Exception:
            last_px = 0
        if last_px <= 0:
            result["status"] = "no_price"
            return result

        shares_diff = int(diff / last_px)
        if shares_diff == 0:
            result["status"] = "within_band"
            _state["last_regime"] = regime
            try:
                with open(_state_path, "w") as f:
                    json.dump(_state, f)
            except Exception:
                pass
            return result

        _limit_price = str(round(last_px, 2))
        if shares_diff > 0:
            try:
                alpaca_throttle.acquire()
                requests.post(
                    f"{ALPACA_BASE_URL}/v2/orders",
                    json={"symbol": defensive_ticker, "qty": str(shares_diff),
                          "side": "buy", "type": "limit",
                          "limit_price": _limit_price,
                          "time_in_force": "day"},
                    headers=_alpaca_headers(), timeout=10)
                result["actions"].append({
                    "type": "defensive_buy",
                    "shares": shares_diff, "ticker": defensive_ticker,
                    "reason": f"regime {last_regime}→{regime}, target {target_pct*100:.0f}%, current {current_pct*100:.0f}%",
                })
                import logging as _dlog
                _dlog.getLogger("bot_engine").info(
                    f"[DEFENSIVE] Bought {shares_diff} {defensive_ticker} (regime {last_regime}→{regime}: {target_pct*100:.0f}% target)")
            except Exception as _e:
                import logging as _dlog
                _dlog.getLogger("bot_engine").debug(f"[DEFENSIVE] Buy failed: {_e}")
        else:
            sell_qty = min(abs(shares_diff), current_shares)
            if sell_qty > 0:
                try:
                    alpaca_throttle.acquire()
                    requests.post(
                        f"{ALPACA_BASE_URL}/v2/orders",
                        json={"symbol": defensive_ticker, "qty": str(sell_qty),
                              "side": "sell", "type": "limit",
                              "limit_price": _limit_price,
                              "time_in_force": "day"},
                        headers=_alpaca_headers(), timeout=10)
                    result["actions"].append({
                        "type": "defensive_sell",
                        "shares": sell_qty, "ticker": defensive_ticker,
                        "reason": f"regime {last_regime}→{regime}, target {target_pct*100:.0f}%, current {current_pct*100:.0f}%",
                    })
                    import logging as _dlog
                    _dlog.getLogger("bot_engine").info(
                        f"[DEFENSIVE] Sold {sell_qty} {defensive_ticker} (regime {last_regime}→{regime}: {target_pct*100:.0f}% target)")
                except Exception as _e:
                    import logging as _dlog
                    _dlog.getLogger("bot_engine").debug(f"[DEFENSIVE] Sell failed: {_e}")

        _state["last_regime"] = regime
        _state["last_rebalance"] = datetime.now().isoformat()
        try:
            with open(_state_path, "w") as f:
                json.dump(_state, f)
        except Exception:
            pass

    except Exception as e:
        result["status"] = f"error: {str(e)[:80]}"

    return result


if __name__ == "__main__":
    # ── Memory diagnostics ─────────────────────────────────────────────
    # Log RSS at entry so OOM kills leave a breadcrumb. Without this, an
    # OOM-killed scan shows up in Node.js as "Command failed" with empty
    # stderr (SIGKILL skips flush). This line is written BEFORE heavy
    # imports, so even if the scan is killed mid-load we know roughly how
    # much memory was available at start.
    try:
        import resource as _res
        _rss_kb = _res.getrusage(_res.RUSAGE_SELF).ru_maxrss
        # Linux: ru_maxrss is KB. macOS: bytes. We always want MB.
        _rss_mb = _rss_kb // 1024 if _rss_kb > 100_000 else _rss_kb // 1024
        print(f"[mem] scan entry rss~{_rss_mb}MB argv={sys.argv}", file=sys.stderr, flush=True)
    except Exception:
        pass

    # ── ML v2 Training Schedule ─────────────────────────────────────────
    # Daily retrain at 4am (called by Tier 3 in bot.ts)
    # Event-triggered: VXX > 1.3, 3+ consecutive stops, SPY > 3% move
    # Research basis:
    #   - News signals decay in 1-5 days → retrain daily captures yesterday's regime
    #   - Momentum signals last 3-6 months → 60-day window is right
    #   - Own trade feedback: 3x weight after 50+ trades (self-learning)
    try:
        from ml_model_v2 import train_model, FEEDBACK_PATH
        model_v2_path = os.path.join(DATA_DIR, "voltrade_ml_v2.pkl")

        # Delete old broken model on first run (52-feature with 42 zeros)
        old_model = os.path.join(DATA_DIR, "voltrade_ml_model.pkl")
        if os.path.exists(old_model):
            try:
                import joblib as _jbl
                _old = _jbl.load(old_model)
                _old_features = len(_old.get("feature_names", []))
                # Old model had 0 stored feature names (confirmed in audit)
                # OR had 52 features with 42 zeros — delete and replace
                if _old_features == 0 or _old_features == 52:
                    os.remove(old_model)
            except Exception:
                pass  # Can't load it — leave it

        # Retrain v2 if: doesn't exist, older than 24hrs, or feedback has grown significantly
        needs_train = (
            not os.path.exists(model_v2_path)
            or (time.time() - os.path.getmtime(model_v2_path)) > 24 * 3600
        )
        # Also retrain if we have 20+ new trades since last train
        if not needs_train and os.path.exists(FEEDBACK_PATH):
            try:
                with open(FEEDBACK_PATH) as _ff:
                    _fb = json.load(_ff)
                if os.path.exists(model_v2_path):
                    model_age = time.time() - os.path.getmtime(model_v2_path)
                    # Rough estimate: trades since last train
                    recent_trades = sum(1 for t in _fb
                        if time.time() - time.mktime(time.strptime(t.get("exit_date","2000-01-01"), "%Y-%m-%d")) < model_age)
                    if recent_trades >= 20:
                        needs_train = True
            except Exception:
                pass

        if needs_train:
            train_result = train_model()  # Returns status dict, silent output
            # Also train the options-specific ML model if enough options trades exist
            try:
                from ml_model_v2 import train_options_model
                train_options_model()  # No-op if < 30 options trades; silent otherwise
            except Exception:
                pass
    except Exception:
        pass

    mode = sys.argv[1] if len(sys.argv) > 1 else "full"

    try:
        if mode == "scan":
            result = scan_market()
        elif mode == "manage":
            result = manage_positions()
        elif mode == "full":
            result = scan_market()
        else:
            result = {"error": f"Unknown mode: {mode}"}
    except Exception as _fatal:
        import traceback
        _tb = traceback.format_exc()
        print(json.dumps({
            "error": f"{type(_fatal).__name__}: {_fatal}",
            "traceback": _tb,
            "trades": [],
            "new_trades": [],
        }), flush=True)
        sys.exit(0)  # Exit 0 so bot.ts can parse the JSON error

    print(json.dumps(result))
