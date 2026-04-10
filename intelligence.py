"""
VolTradeAI Data Intelligence Module
- Smart news classification with trap detection
- SEC insider transaction tracking
- Earnings surprise pattern detection
- Event memory database for ML learning
All free data sources.
"""
import json
import os
import time
import re
import requests
from datetime import datetime, timedelta

try:
    from storage_config import EVENT_MEMORY_PATH, EARNINGS_MEMORY_PATH, INSIDER_CACHE_PATH
except ImportError:
    EVENT_MEMORY_PATH = "/tmp/voltrade_event_memory.json"
    EARNINGS_MEMORY_PATH = "/tmp/voltrade_earnings_memory.json"
    INSIDER_CACHE_PATH = "/tmp/voltrade_insider_cache.json"

POLYGON_KEY = os.environ.get("POLYGON_KEY", "")
EVENT_DB_PATH = EVENT_MEMORY_PATH
EARNINGS_DB_PATH = EARNINGS_MEMORY_PATH

# ── News Event Classification ─────────────────────────────────────────────────

# Event categories with impact weights
EVENT_CATEGORIES = {
    "fda_approval": {
        "keywords": ["fda approves", "fda approval", "fda clears", "fda grants", "fda authorized"],
        "weight": 25, "direction": "bullish"
    },
    "fda_rejection": {
        "keywords": ["fda rejects", "fda denies", "fda refuses", "complete response letter", "crl"],
        "weight": -30, "direction": "bearish"
    },
    "earnings_beat": {
        "keywords": ["beats estimates", "beats expectations", "tops estimates", "exceeds expectations",
                     "earnings beat", "revenue beat", "eps beat", "above consensus", "better than expected"],
        "weight": 12, "direction": "bullish"
    },
    "earnings_miss": {
        "keywords": ["misses estimates", "misses expectations", "below estimates", "falls short",
                     "earnings miss", "revenue miss", "eps miss", "below consensus", "worse than expected"],
        "weight": -15, "direction": "bearish"
    },
    "merger_acquisition": {
        "keywords": ["acquires", "acquisition", "merger", "buyout", "takeover", "bid for",
                     "agrees to buy", "deal to acquire", "purchase agreement"],
        "weight": 18, "direction": "bullish"
    },
    "lawsuit_investigation": {
        "keywords": ["lawsuit", "sued", "class action", "investigation", "sec investigation",
                     "doj investigation", "fraud", "indicted", "subpoena", "regulatory action",
                     "securities class action", "shareholder lawsuit"],
        "weight": -12, "direction": "bearish"
    },
    "insider_selling": {
        "keywords": ["insider sells", "insider selling", "ceo sells", "cfo sells", "director sells",
                     "officer sells", "insider dumped", "stock sale by"],
        "weight": -8, "direction": "bearish"
    },
    "insider_buying": {
        "keywords": ["insider buys", "insider buying", "insider purchase", "ceo buys", "director buys",
                     "open market purchase"],
        "weight": 10, "direction": "bullish"
    },
    "analyst_upgrade": {
        "keywords": ["upgrade", "upgrades", "raised price target", "raises target", "overweight",
                     "outperform", "strong buy", "initiated with buy"],
        "weight": 8, "direction": "bullish"
    },
    "analyst_downgrade": {
        "keywords": ["downgrade", "downgrades", "lowered price target", "lowers target", "underweight",
                     "underperform", "sell rating", "initiated with sell"],
        "weight": -10, "direction": "bearish"
    },
    "guidance_raised": {
        "keywords": ["raises guidance", "raised guidance", "raises outlook", "raised forecast",
                     "boosts guidance", "upward revision", "increases forecast"],
        "weight": 14, "direction": "bullish"
    },
    "guidance_lowered": {
        "keywords": ["lowers guidance", "lowered guidance", "cuts guidance", "reduces outlook",
                     "downward revision", "guidance cut", "warns on earnings"],
        "weight": -16, "direction": "bearish"
    },
    "bankruptcy_default": {
        "keywords": ["bankruptcy", "chapter 11", "chapter 7", "defaults on", "debt default",
                     "files for bankruptcy", "insolvency"],
        "weight": -35, "direction": "very_bearish"
    },
    "stock_buyback": {
        "keywords": ["buyback", "share repurchase", "repurchase program", "stock repurchase"],
        "weight": 8, "direction": "bullish"
    },
    "layoffs_restructuring": {
        "keywords": ["layoffs", "layoff", "job cuts", "workforce reduction", "restructuring",
                     "downsizing", "cost cutting", "headcount reduction"],
        "weight": -5, "direction": "mixed"  # Can be positive for profitability
    },
    "dividend_change": {
        "keywords": ["raises dividend", "increases dividend", "special dividend", "dividend hike"],
        "weight": 7, "direction": "bullish"
    },
    "dividend_cut": {
        "keywords": ["cuts dividend", "suspends dividend", "eliminates dividend", "dividend reduction"],
        "weight": -12, "direction": "bearish"
    },
}


def classify_news(ticker: str) -> dict:
    """
    Fetch and classify news for a ticker.
    Returns detailed event classification, trap detection, and weighted score.
    """
    try:
        alpaca_headers = {
            "APCA-API-KEY-ID": os.environ.get("ALPACA_KEY", ""),
            "APCA-API-SECRET-KEY": os.environ.get("ALPACA_SECRET", ""),
        }
        url = f"https://data.alpaca.markets/v1beta1/news?limit=15&sort=desc&symbols={ticker}"
        resp = requests.get(url, headers=alpaca_headers, timeout=10)
        data = resp.json()
        articles = [{"title": a.get("headline", ""), "description": a.get("summary", ""), "published_utc": a.get("created_at", "")} for a in data.get("news", [])]
    except Exception:
        return {"events": [], "score": 0, "trap_warning": False, "headline_count": 0}

    if not articles:
        return {"events": [], "score": 0, "trap_warning": False, "headline_count": 0}

    events = []
    total_score = 0
    bullish_count = 0
    bearish_count = 0

    for article in articles:
        title = article.get("title", "").lower()
        desc = (article.get("description", "") or "").lower()
        text = title + " " + desc
        published = article.get("published_utc", "")

        # Classify against all categories
        matched_category = None
        matched_weight = 0

        for cat_name, cat_info in EVENT_CATEGORIES.items():
            for keyword in cat_info["keywords"]:
                if keyword in text:
                    matched_category = cat_name
                    matched_weight = cat_info["weight"]
                    break
            if matched_category:
                break

        if matched_category:
            events.append({
                "category": matched_category,
                "weight": matched_weight,
                "headline": article.get("title", "")[:120],
                "published": published,
                "direction": EVENT_CATEGORIES[matched_category]["direction"],
            })
            total_score += matched_weight

            if matched_weight > 0:
                bullish_count += 1
            elif matched_weight < 0:
                bearish_count += 1

    # ── Trap Detection ──
    # "Sell the news" pattern: lots of bullish news but check if it's all upgrades/earnings
    # with no actual catalyst (could be a pump)
    trap_warning = False
    trap_reason = ""

    # Trap 1: Mixed signals — bullish and bearish news at the same time
    if bullish_count >= 2 and bearish_count >= 2:
        trap_warning = True
        trap_reason = f"Mixed signals: {bullish_count} bullish + {bearish_count} bearish headlines — conflicting information"

    # Trap 2: Only analyst upgrades (no real catalyst) — could be wall street pushing stock
    upgrade_count = sum(1 for e in events if e["category"] == "analyst_upgrade")
    if upgrade_count >= 3 and len(events) == upgrade_count:
        trap_warning = True
        trap_reason = f"Wall Street pump? {upgrade_count} analyst upgrades with no fundamental catalyst"

    # Trap 3: Lawsuit filed right after good news — classic trap
    has_good_news = any(e["weight"] >= 10 for e in events)
    has_lawsuit = any(e["category"] == "lawsuit_investigation" for e in events)
    if has_good_news and has_lawsuit:
        trap_warning = True
        trap_reason = "Good news + lawsuit = possible trap. Insiders may know something."
        total_score = min(total_score, 0)  # Cap score at 0 when trap detected

    # Trap 4: Insider selling + bullish news
    has_insider_sell = any(e["category"] == "insider_selling" for e in events)
    if has_good_news and has_insider_sell:
        trap_warning = True
        trap_reason = "Bullish news but insiders are selling — they may not believe their own story"
        total_score = int(total_score * 0.3)  # Slash score by 70%

    # Store to event memory
    _store_events(ticker, events, total_score, trap_warning)

    return {
        "events": events[:10],  # Top 10 events
        "score": total_score,
        "trap_warning": trap_warning,
        "trap_reason": trap_reason,
        "headline_count": len(articles),
        "bullish_events": bullish_count,
        "bearish_events": bearish_count,
        "top_event": events[0]["category"] if events else None,
        "top_headline": events[0]["headline"] if events else "",
    }


# ── SEC EDGAR Insider Transaction Tracking ────────────────────────────────────

def get_insider_activity(ticker: str) -> dict:
    """
    Check recent insider buying/selling from SEC EDGAR.
    Uses the free EDGAR full-text search API.
    Returns: {net_direction, buy_count, sell_count, recent_transactions, insider_signal}
    """
    # Check cache first (cache for 1 hour)
    cache = _load_json(INSIDER_CACHE_PATH)
    cache_key = f"{ticker}_{datetime.now().strftime('%Y%m%d%H')}"
    if cache and cache.get(cache_key):
        return cache[cache_key]

    result = {
        "net_direction": "neutral",
        "buy_count": 0,
        "sell_count": 0,
        "recent_transactions": [],
        "insider_signal": 0,  # -100 to +100
        "total_buy_value": 0,
        "total_sell_value": 0,
    }

    try:
        # Use SEC EDGAR Full-Text Search for Form 4 filings
        thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        url = (f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22"
               f"&forms=4&dateRange=custom&startdt={thirty_days_ago}")
        headers = {"User-Agent": "VolTradeAI research@voltradeai.com"}
        resp = requests.get(url, timeout=10, headers=headers)

        if resp.status_code == 200:
            data = resp.json()
            hits = data.get("hits", {}).get("hits", [])

            for hit in hits[:20]:
                source = hit.get("_source", {})
                form_type = source.get("form_type", "")
                if "4" not in str(form_type):
                    continue

                # Parse the filing text for buy/sell signals
                filing_text = (source.get("display_names", [""])[0] + " " +
                              source.get("file_description", "")).lower()

                is_purchase = any(w in filing_text for w in ["purchase", "bought", "acquisition", "exercise"])
                is_sale = any(w in filing_text for w in ["sale", "sold", "disposition", "dispose"])

                if is_purchase:
                    result["buy_count"] += 1
                elif is_sale:
                    result["sell_count"] += 1

                result["recent_transactions"].append({
                    "date": source.get("file_date", ""),
                    "type": "BUY" if is_purchase else "SELL" if is_sale else "OTHER",
                    "filer": source.get("display_names", ["Unknown"])[0][:60],
                })
    except Exception:
        pass

    # Also check Alpaca for insider news signals
    try:
        alpaca_headers = {
            "APCA-API-KEY-ID": os.environ.get("ALPACA_KEY", ""),
            "APCA-API-SECRET-KEY": os.environ.get("ALPACA_SECRET", ""),
        }
        url = f"https://data.alpaca.markets/v1beta1/news?limit=5&sort=desc&symbols={ticker}"
        resp = requests.get(url, headers=alpaca_headers, timeout=5)
        articles = resp.json().get("news", [])
        for article in articles:
            title = (article.get("headline", "") or "").lower()
            if "insider" in title and "buy" in title:
                result["buy_count"] += 1
            elif "insider" in title and ("sell" in title or "sale" in title):
                result["sell_count"] += 1
    except Exception:
        pass

    # Calculate insider signal
    total = result["buy_count"] + result["sell_count"]
    if total > 0:
        result["insider_signal"] = int(((result["buy_count"] - result["sell_count"]) / total) * 100)

    if result["buy_count"] > result["sell_count"] * 2:
        result["net_direction"] = "strong_buying"
    elif result["buy_count"] > result["sell_count"]:
        result["net_direction"] = "net_buying"
    elif result["sell_count"] > result["buy_count"] * 2:
        result["net_direction"] = "strong_selling"
    elif result["sell_count"] > result["buy_count"]:
        result["net_direction"] = "net_selling"

    # Cache result
    if not cache:
        cache = {}
    cache[cache_key] = result
    # Keep cache small
    if len(cache) > 200:
        keys = sorted(cache.keys())
        for k in keys[:100]:
            del cache[k]
    _save_json(INSIDER_CACHE_PATH, cache)

    return result


# ── Earnings Surprise Pattern Detection ───────────────────────────────────────

def check_earnings_pattern(ticker: str) -> dict:
    """
    Check for sell-the-news patterns around earnings.
    Looks at: did the stock drop on good earnings? (classic trap)
    Returns: {pattern, sell_the_news_risk, earnings_reaction_history}
    """
    result = {
        "pattern": "unknown",
        "sell_the_news_risk": False,
        "sell_the_news_confidence": 0,
        "historical_reactions": [],
    }

    # Load historical earnings reactions from memory
    earnings_db = _load_json(EARNINGS_DB_PATH) or {}
    ticker_history = earnings_db.get(ticker, [])

    if ticker_history:
        # Count sell-the-news events (good earnings but stock dropped)
        sell_news_count = 0
        for entry in ticker_history[-8:]:  # Last 8 quarters
            if entry.get("beat") and entry.get("price_reaction_pct", 0) < -2:
                sell_news_count += 1

        total_earnings = min(len(ticker_history), 8)
        if total_earnings >= 3 and sell_news_count >= 2:
            result["pattern"] = "sell_the_news"
            result["sell_the_news_risk"] = True
            result["sell_the_news_confidence"] = round(sell_news_count / total_earnings, 2)
        elif total_earnings >= 3:
            beat_and_up = sum(1 for e in ticker_history[-8:]
                           if e.get("beat") and e.get("price_reaction_pct", 0) > 0)
            if beat_and_up / total_earnings > 0.6:
                result["pattern"] = "earnings_mover"

        result["historical_reactions"] = ticker_history[-4:]

    # Check if earnings are coming up (from Alpaca news)
    try:
        alpaca_headers = {
            "APCA-API-KEY-ID": os.environ.get("ALPACA_KEY", ""),
            "APCA-API-SECRET-KEY": os.environ.get("ALPACA_SECRET", ""),
        }
        url = f"https://data.alpaca.markets/v1beta1/news?limit=5&sort=desc&symbols={ticker}"
        resp = requests.get(url, headers=alpaca_headers, timeout=5)
        articles = resp.json().get("news", [])
        for article in articles:
            title = (article.get("headline", "") or "").lower()
            if any(w in title for w in ["earnings", "quarterly results", "q1", "q2", "q3", "q4", "fiscal"]):
                # Check if it mentions beat/miss
                is_beat = any(w in title for w in ["beat", "tops", "exceeds", "above"])
                is_miss = any(w in title for w in ["miss", "falls short", "below"])
                if is_beat or is_miss:
                    result["latest_earnings"] = {
                        "beat": is_beat,
                        "headline": article.get("headline", "")[:120],
                        "date": article.get("created_at", ""),
                    }
    except Exception:
        pass

    return result


def record_earnings_outcome(ticker: str, beat: bool, price_reaction_pct: float,
                           eps_surprise: float = 0, revenue_surprise: float = 0):
    """Store an earnings outcome for pattern learning."""
    earnings_db = _load_json(EARNINGS_DB_PATH) or {}
    if ticker not in earnings_db:
        earnings_db[ticker] = []

    earnings_db[ticker].append({
        "date": datetime.now().strftime("%Y-%m-%d"),
        "beat": beat,
        "price_reaction_pct": round(price_reaction_pct, 2),
        "eps_surprise": round(eps_surprise, 2),
        "revenue_surprise": round(revenue_surprise, 2),
        "sell_the_news": beat and price_reaction_pct < -2,
    })

    # Keep last 20 quarters per ticker
    earnings_db[ticker] = earnings_db[ticker][-20:]
    _save_json(EARNINGS_DB_PATH, earnings_db)


# ── Event Memory Database ─────────────────────────────────────────────────────

def _store_events(ticker: str, events: list, score: int, trap: bool):
    """Store classified events for ML to learn from over time."""
    db = _load_json(EVENT_DB_PATH) or {}
    if ticker not in db:
        db[ticker] = []

    db[ticker].append({
        "timestamp": datetime.now().isoformat(),
        "events": [{"cat": e["category"], "w": e["weight"]} for e in events[:5]],
        "score": score,
        "trap": trap,
        # Price outcome will be filled in later by the learning loop
        "price_5d_pct": None,
    })

    # Keep last 50 entries per ticker, max 500 tickers
    db[ticker] = db[ticker][-50:]
    if len(db) > 500:
        # Remove oldest tickers
        sorted_tickers = sorted(db.keys(), key=lambda t: db[t][-1]["timestamp"] if db[t] else "")
        for t in sorted_tickers[:100]:
            del db[t]

    _save_json(EVENT_DB_PATH, db)


def update_event_outcomes(ticker: str, price_change_pct: float):
    """
    Called by the learning loop: fill in what happened to the price
    after news events were detected. This is how the ML learns whether
    good news actually led to price increases.
    """
    db = _load_json(EVENT_DB_PATH) or {}
    if ticker not in db:
        return

    # Update entries from ~5 days ago that don't have outcomes yet
    five_days_ago = (datetime.now() - timedelta(days=5)).isoformat()
    for entry in db[ticker]:
        if entry.get("price_5d_pct") is None and entry["timestamp"] < five_days_ago:
            entry["price_5d_pct"] = round(price_change_pct, 2)

    _save_json(EVENT_DB_PATH, db)


def get_event_learning_stats(ticker: str = None) -> dict:
    """
    Analyze event memory to find patterns:
    - Which event types actually lead to price moves?
    - How often does good news lead to drops? (sell-the-news frequency)
    """
    db = _load_json(EVENT_DB_PATH) or {}

    if ticker:
        entries = db.get(ticker, [])
    else:
        entries = []
        for t_entries in db.values():
            entries.extend(t_entries)

    # Filter entries with known outcomes
    with_outcomes = [e for e in entries if e.get("price_5d_pct") is not None]
    if len(with_outcomes) < 10:
        return {"status": "not_enough_data", "entries": len(with_outcomes)}

    # Analyze by event category
    category_stats = {}
    trap_accuracy = {"correct": 0, "wrong": 0}

    for entry in with_outcomes:
        price_change = entry["price_5d_pct"]
        was_trap = entry.get("trap", False)

        if was_trap:
            if price_change < 0:
                trap_accuracy["correct"] += 1
            else:
                trap_accuracy["wrong"] += 1

        for event in entry.get("events", []):
            cat = event["cat"]
            if cat not in category_stats:
                category_stats[cat] = {"count": 0, "avg_move": 0, "wins": 0}
            category_stats[cat]["count"] += 1
            category_stats[cat]["avg_move"] += price_change
            if (event["w"] > 0 and price_change > 0) or (event["w"] < 0 and price_change < 0):
                category_stats[cat]["wins"] += 1

    # Compute averages
    for cat in category_stats:
        stats = category_stats[cat]
        if stats["count"] > 0:
            stats["avg_move"] = round(stats["avg_move"] / stats["count"], 2)
            stats["win_rate"] = round(stats["wins"] / stats["count"] * 100, 1)

    total_traps = trap_accuracy["correct"] + trap_accuracy["wrong"]

    return {
        "status": "ok",
        "total_events_tracked": len(with_outcomes),
        "category_stats": category_stats,
        "trap_detection_accuracy": round(trap_accuracy["correct"] / total_traps * 100, 1) if total_traps > 0 else 0,
        "trap_sample_size": total_traps,
    }


# ── Full Intelligence Report ──────────────────────────────────────────────────

def get_full_intelligence(ticker: str) -> dict:
    """
    Run all intelligence systems for a ticker.
    Returns combined assessment.
    """
    news = classify_news(ticker)
    insider = get_insider_activity(ticker)
    earnings = check_earnings_pattern(ticker)

    # Combined intelligence score
    intel_score = news["score"]

    # Insider signal adjustment
    if insider["net_direction"] == "strong_selling":
        intel_score -= 15
    elif insider["net_direction"] == "net_selling":
        intel_score -= 8
    elif insider["net_direction"] == "strong_buying":
        intel_score += 12
    elif insider["net_direction"] == "net_buying":
        intel_score += 6

    # Earnings pattern adjustment
    if earnings["sell_the_news_risk"]:
        intel_score -= 10

    # Trap override
    if news["trap_warning"]:
        intel_score = min(intel_score, -5)

    return {
        "intel_score": intel_score,
        "news": news,
        "insider": insider,
        "earnings_pattern": earnings,
        "trap_warning": news["trap_warning"],
        "trap_reason": news.get("trap_reason", ""),
        "summary": _generate_summary(news, insider, earnings, intel_score),
    }


def _generate_summary(news, insider, earnings, score):
    """Plain English summary of intelligence findings."""
    parts = []

    if news["events"]:
        top = news["events"][0]
        parts.append(f"Top news: {top['category'].replace('_', ' ')} ({top['headline'][:60]})")

    if insider["net_direction"] != "neutral":
        parts.append(f"Insiders: {insider['net_direction'].replace('_', ' ')} ({insider['buy_count']} buys, {insider['sell_count']} sells)")

    if earnings["sell_the_news_risk"]:
        parts.append(f"WARNING: This stock has a sell-the-news pattern ({earnings['sell_the_news_confidence']:.0%} of beats led to drops)")

    if news["trap_warning"]:
        parts.append(f"TRAP ALERT: {news.get('trap_reason', 'conflicting signals')}")

    if not parts:
        parts.append("No significant intelligence signals")

    return " | ".join(parts)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_json(path):
    try:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    except Exception:
        pass
    return None

def _save_json(path, data):
    try:
        with open(path, "w") as f:
            json.dump(data, f)
    except Exception:
        pass


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print(f"=== Full Intelligence Report for {ticker} ===")
    report = get_full_intelligence(ticker)
    print(json.dumps(report, indent=2, default=str))
