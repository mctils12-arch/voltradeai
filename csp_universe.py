#!/usr/bin/env python3
"""
Dynamic CSP Universe Ranker (2026-04-22)
==========================================
Replaces Tier 1's hardcoded 16-ticker list with a dynamic 200-ticker
universe, ranked every 15 minutes by 7-factor CSP attractiveness score.

ARCHITECTURE:
  Layer 1: Hard gates (run on ~11,600 tickers) → ~400 candidates
  Layer 2: 7-factor scoring (run on ~400) → ranked top 200
  Layer 3: Tier 1 consumer uses ranked list, skipping held names

ALPHA PRINCIPLE:
  CSP edge = (IV - RV) × probability_OTM × premium_capture_pct
  We score each component separately and compose them. Top 200 are
  where this product is highest empirically.

WHY 200 NOT 1000:
  Beyond rank 200, options volume drops below 1,000 contracts/day,
  spreads widen to 5%+, and fills become unreliable. Additional
  tickers contribute diminishing alpha per name. 200 is the
  empirical sweet spot for retail-scale CSP.

REUSES EXISTING INFRASTRUCTURE:
  - bot_engine._get_full_universe() for base symbol list
  - options_scanner._fetch_iv_rank() for per-ticker IV rank
  - vol_surface.get_surface_score() for VRP + skew + surface
  - options_scanner._fetch_earnings_calendar() for earnings gates
  - options_scanner._fetch_options_chain() for liquidity check (Layer 1)
"""
import json
import os
import logging
import time
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger("voltrade.csp_universe")

try:
    from storage_config import DATA_DIR
except ImportError:
    DATA_DIR = "/tmp"

UNIVERSE_CACHE_PATH = os.path.join(DATA_DIR, "voltrade_csp_universe_cache.json")
SCORES_CACHE_PATH = os.path.join(DATA_DIR, "voltrade_csp_scores_cache.json")
LAYER1_CACHE_TTL = 900   # 15 min — universe filter changes slowly
LAYER2_CACHE_TTL = 900   # 15 min — ranked scores

# Hard-coded anchors always included (bulletproof fallback if scoring fails)
CSP_ANCHOR_TICKERS = [
    "SPY", "QQQ", "IWM",
    "AAPL", "MSFT", "NVDA", "META", "GOOGL",
    "AMZN", "TSLA", "AMD", "AVGO", "CRM", "ORCL",
    "COIN", "MSTR", "UBER",
]

# Blocked tickers (leveraged/inverse ETFs, known scam patterns)
CSP_BLOCKED_TICKERS = {
    "TQQQ", "SQQQ", "UPRO", "SPXU", "UVXY", "SVXY", "TMF", "TMV",
    "SOXL", "SOXS", "TNA", "TZA", "LABU", "LABD", "SPXL", "SPXS",
    "FAS", "FAZ", "YINN", "YANG", "JNUG", "JDST", "NUGT", "DUST",
    "BOIL", "KOLD", "UGL", "GLL", "UCO", "SCO", "UVIX", "SVIX",
}

# Max universe size after Layer 2 ranking
MAX_UNIVERSE_SIZE = 200


def _is_likely_etf_leveraged(ticker: str) -> bool:
    """Heuristic: avoid tickers matching known leveraged ETF patterns."""
    if ticker in CSP_BLOCKED_TICKERS:
        return True
    # Common leveraged ETF suffixes
    if len(ticker) == 4:
        suffixes = ("BULL", "BEAR", "SOXL", "SOXS", "SQQQ", "TQQQ")
        if ticker.endswith(("L", "S")) and ticker[:-1] in ("TQQQ", "SQQQ", "SOXL", "SOXS"):
            return True
    return False


def _layer1_hard_gates(snap_data: Optional[Dict] = None) -> List[Tuple[str, float, int, float]]:
    """
    Layer 1: run hard gates on full universe. Returns list of
    (ticker, price, volume, dollar_volume) that pass all filters.

    Cached 15 min in SCORES_CACHE_PATH for layer_2 to consume.
    """
    # Check cache
    if os.path.exists(UNIVERSE_CACHE_PATH):
        try:
            with open(UNIVERSE_CACHE_PATH) as f:
                cached = json.load(f)
            if time.time() - cached.get("_cached_at", 0) < LAYER1_CACHE_TTL:
                logger.debug(f"Layer 1 cache hit: {len(cached.get('candidates', []))} tickers")
                return [tuple(c) for c in cached.get("candidates", [])]
        except Exception as e:
            logger.debug(f"Layer 1 cache read failed: {e}")

    # Get snapshot data — either passed in or fetch fresh
    if snap_data is None:
        try:
            from options_scanner import _get_options_candidates
            # Side effect: this populates snap_data internally AND applies Stage 1 filters
            # Problem: it returns tuples (ticker, price, setup_type), not raw snaps.
            # So we fetch snap_data ourselves via bot_engine helper.
            snap_data = _fetch_snap_data_for_universe()
        except Exception as e:
            logger.warning(f"Could not fetch snap_data: {e}")
            return []

    if not snap_data:
        logger.warning("Layer 1: empty snap_data, returning anchors only")
        return [(t, 0.0, 0, 0.0) for t in CSP_ANCHOR_TICKERS]

    candidates = []
    for sym, snap in snap_data.items():
        if not sym or "." in sym or len(sym) > 5:
            continue
        if sym in CSP_BLOCKED_TICKERS or _is_likely_etf_leveraged(sym):
            continue
        try:
            bar = snap.get("dailyBar", {}) or {}
            price = float(bar.get("c", 0) or 0)
            volume = int(bar.get("v", 0) or 0)
        except (ValueError, TypeError):
            continue
        if price < 10.0:
            continue
        if volume < 1_000_000:
            continue
        dollar_volume = price * volume
        if dollar_volume < 20_000_000:
            continue
        candidates.append((sym, round(price, 2), volume, round(dollar_volume, 2)))

    # Ensure anchors always included even if they didn't pass (liquid floor)
    existing = {c[0] for c in candidates}
    for anchor in CSP_ANCHOR_TICKERS:
        if anchor not in existing:
            snap = snap_data.get(anchor, {})
            if snap:
                try:
                    bar = snap.get("dailyBar", {}) or {}
                    price = float(bar.get("c", 0) or 0)
                    volume = int(bar.get("v", 0) or 0)
                    if price > 0:
                        candidates.append((anchor, round(price, 2), volume, round(price * volume, 2)))
                except Exception:
                    pass

    # Persist
    try:
        tmp_path = UNIVERSE_CACHE_PATH + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump({
                "candidates": candidates,
                "count": len(candidates),
                "_cached_at": time.time(),
            }, f)
        os.replace(tmp_path, UNIVERSE_CACHE_PATH)
    except Exception as e:
        logger.debug(f"Layer 1 cache write failed: {e}")

    logger.info(f"Layer 1: {len(candidates)} candidates passed hard gates")
    return candidates


def _fetch_snap_data_for_universe() -> Dict[str, Dict]:
    """Fetch snapshot data for full universe. Helper for Layer 1."""
    try:
        import requests
        from bot_engine import _get_full_universe, ALPACA_DATA_URL, _alpaca_headers
        from concurrent.futures import ThreadPoolExecutor
        try:
            from alpaca_rate_limiter import alpaca_throttle
        except ImportError:
            class _NoT:
                def acquire(self): pass
            alpaca_throttle = _NoT()

        universe = _get_full_universe()
        if not universe:
            return {}

        snap_all = {}
        batches = [universe[i:i+500] for i in range(0, len(universe), 500)]

        def _fetch_batch(batch):
            try:
                alpaca_throttle.acquire()
                r = requests.get(
                    f"{ALPACA_DATA_URL}/v2/stocks/snapshots",
                    params={"symbols": ",".join(batch), "feed": "sip"},
                    headers=_alpaca_headers(),
                    timeout=15,
                )
                return r.json()
            except Exception:
                return {}

        with ThreadPoolExecutor(max_workers=2) as pool:
            for result in pool.map(_fetch_batch, batches):
                if isinstance(result, dict):
                    snap_all.update(result)

        return snap_all
    except Exception as e:
        logger.warning(f"_fetch_snap_data_for_universe failed: {e}")
        return {}


# ── Layer 2 scoring factors ──────────────────────────────────────


def _score_iv_rank(ticker: str, iv_rank_cache: Dict[str, Optional[float]]) -> float:
    """IV rank score: higher IVR = more premium to collect.
    0-100 scale, sweet spot at IVR 60-80."""
    if ticker in iv_rank_cache:
        ivr = iv_rank_cache[ticker]
    else:
        try:
            from options_scanner import _fetch_iv_rank
            ivr = _fetch_iv_rank(ticker)
        except Exception:
            ivr = None
        iv_rank_cache[ticker] = ivr

    if ivr is None:
        return 30.0  # Unknown — give modest baseline so we don't fully skip
    # Scoring curve: target IVR 60-80 as sweet spot, penalize extremes
    if ivr >= 80:
        return 85.0  # Very high IVR — often signals imminent event
    elif ivr >= 60:
        return 95.0  # Sweet spot for CSP
    elif ivr >= 40:
        return 70.0
    elif ivr >= 25:
        return 50.0
    else:
        return 20.0  # Low IVR — not enough premium


def _score_vrp(ticker: str, surface_cache: Dict[str, Optional[Dict]]) -> float:
    """VRP score: higher VRP = more structural edge for premium selling.
    0-100 scale based on vrp_20d from vol_surface."""
    if ticker in surface_cache:
        surface = surface_cache[ticker]
    else:
        try:
            from vol_surface import get_surface_score
            surface = get_surface_score(ticker)
        except Exception:
            surface = None
        surface_cache[ticker] = surface

    if not surface or "error" in (surface or {}):
        return 40.0  # Unknown — modest baseline
    vrp = surface.get("vrp", {}).get("vrp_20d", 0)
    if vrp >= 0.08:
        return 95.0
    elif vrp >= 0.05:
        return 85.0
    elif vrp >= 0.03:
        return 70.0
    elif vrp >= 0.01:
        return 55.0
    elif vrp >= 0:
        return 40.0
    elif vrp >= -0.02:
        return 25.0
    else:
        return 10.0  # Negative VRP — options cheap, don't sell


def _score_liquidity(ticker: str, price: float, volume: int, dollar_volume: float) -> float:
    """Liquidity score from base metrics. Higher = more fillable.
    Already filtered to dollar_volume > $20M, so range starts at passing."""
    # Dollar volume tiers (already passed $20M minimum)
    if dollar_volume >= 1_000_000_000:
        dv_score = 100.0
    elif dollar_volume >= 500_000_000:
        dv_score = 90.0
    elif dollar_volume >= 200_000_000:
        dv_score = 80.0
    elif dollar_volume >= 100_000_000:
        dv_score = 70.0
    elif dollar_volume >= 50_000_000:
        dv_score = 60.0
    else:
        dv_score = 45.0
    # Price tier (prefer $30-$500 sweet spot — not penny, not too expensive to CSP)
    if 30 <= price <= 500:
        price_score = 100.0
    elif 15 <= price < 30 or 500 < price <= 800:
        price_score = 80.0
    elif price > 800:
        price_score = 60.0  # expensive CSP ties up too much cash
    else:
        price_score = 50.0  # $10-15 range, lower scoring
    return (dv_score * 0.7 + price_score * 0.3)


def _score_put_skew(ticker: str, surface_cache: Dict[str, Optional[Dict]]) -> float:
    """Put-call skew: how much put premium exceeds call premium.
    Higher = more fear priced in = more edge for put sellers."""
    surface = surface_cache.get(ticker)
    if not surface or "error" in (surface or {}):
        return 50.0
    skew_data = surface.get("skew", {})
    put_call_skew = abs(skew_data.get("put_call_skew", 0))
    # Typical range 0 to 0.15
    if put_call_skew >= 0.10:
        return 90.0
    elif put_call_skew >= 0.07:
        return 80.0
    elif put_call_skew >= 0.04:
        return 65.0
    elif put_call_skew >= 0.02:
        return 50.0
    else:
        return 35.0


def _score_earnings(ticker: str, earnings_cal: Dict[str, int]) -> float:
    """Earnings proximity penalty. Closer = more gap risk."""
    days = earnings_cal.get(ticker, 99)
    if days <= 2:
        return 0.0    # reject — way too close
    elif days <= 7:
        return 10.0   # very penalized
    elif days <= 14:
        return 50.0   # moderate
    elif days <= 30:
        return 80.0   # mild
    else:
        return 100.0  # no earnings concern


def _score_historical(ticker: str) -> float:
    """Look up past CSP win rate for this ticker from ML feedback.
    Returns 50 (neutral) if insufficient data."""
    try:
        import json as _json
        from ml_model_v2 import FEEDBACK_PATH
        if not os.path.exists(FEEDBACK_PATH):
            return 50.0
        with open(FEEDBACK_PATH) as f:
            feedback = _json.load(f)
        if not isinstance(feedback, list):
            return 50.0
        # Filter to CSP trades on this ticker
        ticker_upper = ticker.strip().upper()
        csp_trades = [
            r for r in feedback
            if r.get("ticker", "").strip().upper() == ticker_upper
            and r.get("side") == "sell"
            and r.get("outcome") in ("win", "loss")
        ]
        if len(csp_trades) < 10:
            return 50.0  # insufficient data, neutral
        wins = sum(1 for r in csp_trades if r.get("outcome") == "win")
        win_rate = wins / len(csp_trades)
        # Convert win rate to score (baseline 50% → 50 score)
        return min(95.0, max(20.0, win_rate * 100))
    except Exception:
        return 50.0


def _score_stability(ticker: str, price: float, snap_data: Dict) -> float:
    """Price stability proxy from snapshot data.
    Avoid stocks crashing hard or rallying parabolically."""
    snap = snap_data.get(ticker, {})
    if not snap:
        return 50.0
    try:
        bar = snap.get("dailyBar", {}) or {}
        prev = snap.get("prevDailyBar", {}) or {}
        c = float(bar.get("c", 0) or 0)
        pc = float(prev.get("c", c) or c)
        high = float(bar.get("h", c) or c)
        low = float(bar.get("l", c) or c)
        if pc <= 0:
            return 50.0
        daily_change_pct = abs((c - pc) / pc * 100)
        intraday_range = (high - low) / pc * 100 if pc > 0 else 0
        # Penalize extreme daily moves (> 8% is usually news/crash)
        if daily_change_pct >= 10:
            return 10.0
        elif daily_change_pct >= 6:
            return 30.0
        elif daily_change_pct >= 4:
            return 55.0
        elif daily_change_pct >= 2:
            return 75.0
        else:
            return 90.0  # calm stock — best for CSP
    except Exception:
        return 50.0


def _layer2_score(candidates: List[Tuple], snap_data: Dict = None) -> List[Dict]:
    """
    Score every Layer 1 candidate with 7-factor formula.
    Returns list of {ticker, score, factors, price, ...} dicts, ranked.

    Caches 15 min. Uses per-scan memoization for expensive factors.
    """
    if os.path.exists(SCORES_CACHE_PATH):
        try:
            with open(SCORES_CACHE_PATH) as f:
                cached = json.load(f)
            if time.time() - cached.get("_cached_at", 0) < LAYER2_CACHE_TTL:
                logger.debug(f"Layer 2 cache hit: {len(cached.get('scores', []))} tickers")
                return cached.get("scores", [])
        except Exception:
            pass

    if not candidates:
        return []

    # Per-scan caches for expensive factors (avoid duplicate API calls)
    iv_rank_cache: Dict[str, Optional[float]] = {}
    surface_cache: Dict[str, Optional[Dict]] = {}

    # Pull earnings calendar once
    earnings_cal: Dict[str, int] = {}
    try:
        from options_scanner import _fetch_earnings_calendar
        earnings_cal = _fetch_earnings_calendar(days_ahead=30) or {}
    except Exception as e:
        logger.debug(f"earnings calendar fetch failed: {e}")

    # If snap_data not provided, skip stability scoring (50 fallback)
    snap_data = snap_data or {}

    scored = []
    # LIMIT expensive factor fetches to top N (by dollar_volume). We
    # sort candidates by dollar_volume descending, score top 500 with
    # full factor stack (IVR + surface), rest get lighter scoring.
    # This prevents 11,600 IVR fetches per cycle.
    sorted_candidates = sorted(candidates, key=lambda c: c[3], reverse=True)
    deep_score_limit = 500  # full-factor scoring applies to top 500 only

    # PARALLEL-SCORE 2026-04-23: parallelize expensive factor fetches
    # (IV rank + surface score). Previously serial 500x = 250-500s hang.
    # Now 8 workers = ~30-50s typical.
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Split into two passes: top deep_score_limit get full scoring,
    # rest get cheap scoring.
    deep_candidates = sorted_candidates[:deep_score_limit]
    shallow_candidates = sorted_candidates[deep_score_limit:]

    # Prefetch IV rank and surface score IN PARALLEL for deep candidates
    def _prefetch(ticker):
        try:
            _score_iv_rank(ticker, iv_rank_cache)
            _score_vrp(ticker, surface_cache)
        except Exception:
            pass

    if deep_candidates:
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(_prefetch, [c[0] for c in deep_candidates], timeout=60))

    scored = []
    for idx, (ticker, price, volume, dollar_volume) in enumerate(sorted_candidates):
        full_scoring = idx < deep_score_limit
        liquidity = _score_liquidity(ticker, price, volume, dollar_volume)
        earnings = _score_earnings(ticker, earnings_cal)
        stability = _score_stability(ticker, price, snap_data)
        historical = _score_historical(ticker)
        if full_scoring:
            # Caches populated by parallel prefetch above
            iv_rank_s = _score_iv_rank(ticker, iv_rank_cache)
            vrp_s = _score_vrp(ticker, surface_cache)
            skew_s = _score_put_skew(ticker, surface_cache)
        else:
            iv_rank_s = 40.0
            vrp_s = 40.0
            skew_s = 50.0
        composite = round(
            0.25 * iv_rank_s +
            0.20 * vrp_s +
            0.15 * liquidity +
            0.10 * skew_s +
            0.10 * earnings +
            0.10 * historical +
            0.10 * stability,
            2
        )
        if earnings <= 10:
            continue
        scored.append({
            "ticker": ticker,
            "score": composite,
            "price": price,
            "volume": volume,
            "dollar_volume": dollar_volume,
            "factors": {
                "iv_rank": round(iv_rank_s, 2),
                "vrp": round(vrp_s, 2),
                "liquidity": round(liquidity, 2),
                "put_skew": round(skew_s, 2),
                "earnings": round(earnings, 2),
                "historical": round(historical, 2),
                "stability": round(stability, 2),
            },
            "full_scoring": full_scoring,
        })

    # Sort by score descending, take top MAX_UNIVERSE_SIZE
    scored.sort(key=lambda x: x["score"], reverse=True)
    top = scored[:MAX_UNIVERSE_SIZE]

    # Persist
    try:
        tmp_path = SCORES_CACHE_PATH + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump({
                "scores": top,
                "count": len(top),
                "total_scored": len(scored),
                "_cached_at": time.time(),
            }, f)
        os.replace(tmp_path, SCORES_CACHE_PATH)
    except Exception as e:
        logger.debug(f"Layer 2 cache write failed: {e}")

    logger.info(f"Layer 2: scored {len(scored)}, returned top {len(top)}")
    return top


def get_top_csp_candidates(n: int = MAX_UNIVERSE_SIZE,
                           snap_data: Optional[Dict] = None) -> List[str]:
    """
    PUBLIC API: returns ranked list of top N CSP tickers.
    Used by tier1_csp_core in place of hardcoded T1_TICKERS.

    Guaranteed to return at least CSP_ANCHOR_TICKERS (no empty list).
    """
    try:
        candidates = _layer1_hard_gates(snap_data=snap_data)
        if not candidates:
            logger.warning("Layer 1 empty, returning anchors only")
            return list(CSP_ANCHOR_TICKERS)
        scored = _layer2_score(candidates, snap_data=snap_data)
        if not scored:
            logger.warning("Layer 2 empty, returning anchors only")
            return list(CSP_ANCHOR_TICKERS)
        tickers = [s["ticker"] for s in scored[:n]]
        # Ensure anchors always present (union)
        for anchor in CSP_ANCHOR_TICKERS:
            if anchor not in tickers:
                tickers.append(anchor)
        return tickers[:n]
    except Exception as e:
        logger.error(f"get_top_csp_candidates failed: {e} — returning anchors")
        return list(CSP_ANCHOR_TICKERS)


def get_universe_snapshot() -> Dict[str, Any]:
    """
    For observability via /api/system/snapshot.
    Returns current universe state + top-20 picks with factor breakdowns.
    """
    result = {
        "mode": "dynamic_200",
        "anchors_count": len(CSP_ANCHOR_TICKERS),
        "blocked_count": len(CSP_BLOCKED_TICKERS),
        "max_universe_size": MAX_UNIVERSE_SIZE,
    }
    try:
        if os.path.exists(UNIVERSE_CACHE_PATH):
            with open(UNIVERSE_CACHE_PATH) as f:
                l1 = json.load(f)
            result["layer1"] = {
                "candidate_count": l1.get("count", 0),
                "cache_age_seconds": int(time.time() - l1.get("_cached_at", 0)),
            }
    except Exception as e:
        result["layer1_error"] = str(e)[:100]
    try:
        if os.path.exists(SCORES_CACHE_PATH):
            with open(SCORES_CACHE_PATH) as f:
                l2 = json.load(f)
            result["layer2"] = {
                "ranked_count": l2.get("count", 0),
                "total_scored": l2.get("total_scored", 0),
                "cache_age_seconds": int(time.time() - l2.get("_cached_at", 0)),
                "top_20": l2.get("scores", [])[:20],
            }
    except Exception as e:
        result["layer2_error"] = str(e)[:100]
    return result


if __name__ == "__main__":
    # Manual test
    print("Fetching Layer 1 universe...")
    candidates = _layer1_hard_gates()
    print(f"Layer 1: {len(candidates)} candidates")
    if candidates:
        print("Sample:", candidates[:5])

    print("\nRunning Layer 2 scoring...")
    scored = _layer2_score(candidates)
    print(f"Layer 2: {len(scored)} scored")
    if scored:
        print("\nTop 10 CSP candidates:")
        for r in scored[:10]:
            print(f"  {r['ticker']:6s} score={r['score']:5.1f}  price=${r['price']:7.2f}  "
                  f"DV=${r['dollar_volume']/1e6:7.1f}M  IVR={r['factors']['iv_rank']:.0f} "
                  f"VRP={r['factors']['vrp']:.0f}")
