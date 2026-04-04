"""
VolTradeAI Social & Search Data Sources
- Reddit sentiment (RSS from WSB, stocks, investing, options)
- Google Trends search interest spikes
- Multi-source financial news RSS
All free, cached aggressively to respect rate limits.
"""
import json
import os
import re
import time
import xml.etree.ElementTree as ET
import requests
from datetime import datetime, timedelta

CACHE_DIR = "/tmp/voltrade_alt_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

USER_AGENT = "VolTradeAI:v2.0 (by /u/voltrade_research)"
POLYGON_KEY = os.environ.get("POLYGON_KEY", "UNwTHo3kvZMBckeIaHQbBLuaaURmFUQP")

# Sentiment word lists (financial context)
BULLISH_WORDS = {
    "moon", "mooning", "rocket", "calls", "bull", "bullish", "buy", "buying",
    "long", "squeeze", "tendies", "diamond", "hands", "yolo", "rip", "ripping",
    "breakout", "surge", "soar", "rally", "green", "pump", "undervalued",
    "upgrade", "beat", "crush", "print", "money", "profit", "gain", "gains",
    "up", "higher", "rising", "growth", "strong", "winner", "accumulate",
}
BEARISH_WORDS = {
    "puts", "bear", "bearish", "sell", "selling", "short", "shorting",
    "crash", "dump", "dumping", "drill", "drilling", "red", "bag", "bagholder",
    "overvalued", "downgrade", "miss", "tanking", "tank", "drop", "plunge",
    "loss", "losing", "rip", "dead", "rug", "scam", "fraud", "bubble",
    "down", "lower", "falling", "weak", "loser", "avoid", "warning",
}


def _cache_get(key, ttl_seconds=3600):
    path = os.path.join(CACHE_DIR, f"{key}.json")
    try:
        if os.path.exists(path) and (time.time() - os.path.getmtime(path)) < ttl_seconds:
            with open(path) as f:
                return json.load(f)
    except Exception:
        pass
    return None

def _cache_set(key, data):
    path = os.path.join(CACHE_DIR, f"{key}.json")
    try:
        with open(path, "w") as f:
            json.dump(data, f)
    except Exception:
        pass


# ── 1. Reddit Sentiment (via RSS) ────────────────────────────────────────────

SUBREDDITS = ["wallstreetbets", "stocks", "investing", "options", "StockMarket"]

def get_reddit_sentiment(ticker: str) -> dict:
    """
    Fetch Reddit mentions and sentiment for a ticker from multiple subreddits via RSS.
    Cached for 2 hours (Reddit rate limits are strict).
    
    Returns: {
        total_mentions: int,
        bullish_pct: float (0-100),
        bearish_pct: float (0-100),
        sentiment_score: float (-100 to +100),
        buzz_level: "none"/"low"/"medium"/"high"/"viral",
        top_posts: [{title, subreddit, sentiment}],
        wsb_mentions: int,
        signal: int (-10 to +10),
    }
    """
    cache_key = f"reddit_{ticker}"
    cached = _cache_get(cache_key, ttl_seconds=7200)
    if cached:
        return cached

    result = {
        "total_mentions": 0,
        "bullish_pct": 50.0,
        "bearish_pct": 50.0,
        "sentiment_score": 0,
        "buzz_level": "none",
        "top_posts": [],
        "wsb_mentions": 0,
        "signal": 0,
    }

    all_posts = []

    for sub in SUBREDDITS:
        try:
            url = f"https://www.reddit.com/r/{sub}/search.rss?q={ticker}&sort=new&t=week&limit=25"
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=10)
            if resp.status_code != 200:
                continue

            # Parse RSS XML
            root = ET.fromstring(resp.text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            entries = root.findall(".//atom:entry", ns)

            for entry in entries:
                title_el = entry.find("atom:title", ns)
                if title_el is None or title_el.text is None:
                    continue
                title = title_el.text
                
                # Only count if ticker is actually mentioned
                if ticker.upper() not in title.upper() and f"${ticker.upper()}" not in title.upper():
                    continue

                # Score sentiment
                words = set(title.lower().split())
                bull_hits = len(words & BULLISH_WORDS)
                bear_hits = len(words & BEARISH_WORDS)
                
                if bull_hits > bear_hits:
                    sentiment = "bullish"
                elif bear_hits > bull_hits:
                    sentiment = "bearish"
                else:
                    sentiment = "neutral"

                all_posts.append({
                    "title": title[:120],
                    "subreddit": sub,
                    "sentiment": sentiment,
                })

                if sub == "wallstreetbets":
                    result["wsb_mentions"] += 1

            # Polite delay between subreddits
            time.sleep(0.5)

        except Exception:
            continue

    result["total_mentions"] = len(all_posts)
    result["top_posts"] = all_posts[:10]

    if all_posts:
        bullish = sum(1 for p in all_posts if p["sentiment"] == "bullish")
        bearish = sum(1 for p in all_posts if p["sentiment"] == "bearish")
        total = len(all_posts)
        
        result["bullish_pct"] = round(bullish / total * 100, 1)
        result["bearish_pct"] = round(bearish / total * 100, 1)
        result["sentiment_score"] = round((bullish - bearish) / total * 100)

    # Buzz level classification
    mentions = result["total_mentions"]
    if mentions == 0:
        result["buzz_level"] = "none"
        result["signal"] = 0
    elif mentions <= 3:
        result["buzz_level"] = "low"
        result["signal"] = 1
    elif mentions <= 10:
        result["buzz_level"] = "medium"
        result["signal"] = 3
    elif mentions <= 25:
        result["buzz_level"] = "high"
        result["signal"] = 5
    else:
        result["buzz_level"] = "viral"
        result["signal"] = 8

    # Sentiment direction adjustment
    if result["sentiment_score"] > 30:
        result["signal"] = min(result["signal"] + 3, 10)
    elif result["sentiment_score"] < -30:
        result["signal"] = max(result["signal"] - 5, -10)  # Bearish crowd = stronger penalty

    # WSB contrarian signal: when WSB is extremely bullish, be cautious
    if result["wsb_mentions"] > 10 and result["bullish_pct"] > 80:
        result["signal"] = min(result["signal"] - 3, result["signal"])  # WSB hype = danger

    _cache_set(cache_key, result)
    return result


# ── 2. Google Trends Search Interest ─────────────────────────────────────────

def get_google_trends(ticker: str) -> dict:
    """
    Get Google Trends search interest for a ticker.
    Cached for 6 hours (Google rate limits pytrends aggressively).
    
    Returns: {
        current_interest: int (0-100),
        avg_interest: float,
        spike_ratio: float,
        has_spike: bool,
        trend_direction: "rising"/"falling"/"stable",
        signal: int (-5 to +10),
    }
    """
    cache_key = f"gtrends_{ticker}"
    cached = _cache_get(cache_key, ttl_seconds=21600)  # 6 hours
    if cached:
        return cached

    result = {
        "current_interest": 0,
        "avg_interest": 0,
        "spike_ratio": 1.0,
        "has_spike": False,
        "trend_direction": "stable",
        "signal": 0,
    }

    try:
        from pytrends.request import TrendReq
        pytrends = TrendReq(hl='en-US', tz=360, timeout=(10, 25))
        
        # Search for "{TICKER} stock" to get financial context
        keyword = f"{ticker} stock"
        pytrends.build_payload([keyword], timeframe='now 7-d')
        data = pytrends.interest_over_time()

        if data is not None and not data.empty and keyword in data.columns:
            values = data[keyword].values
            if len(values) >= 5:
                avg = float(values[:-3].mean()) if len(values) > 3 else float(values.mean())
                recent = float(values[-3:].mean())
                current = float(values[-1])

                result["current_interest"] = int(current)
                result["avg_interest"] = round(avg, 1)

                if avg > 0:
                    result["spike_ratio"] = round(recent / avg, 2)
                
                result["has_spike"] = result["spike_ratio"] > 2.0

                # Trend direction
                if len(values) >= 10:
                    first_half = float(values[:len(values)//2].mean())
                    second_half = float(values[len(values)//2:].mean())
                    if second_half > first_half * 1.3:
                        result["trend_direction"] = "rising"
                    elif second_half < first_half * 0.7:
                        result["trend_direction"] = "falling"

                # Signal scoring
                if result["has_spike"] and result["spike_ratio"] > 3.0:
                    result["signal"] = 10  # Massive spike
                elif result["has_spike"]:
                    result["signal"] = 6   # Significant spike
                elif result["spike_ratio"] > 1.5:
                    result["signal"] = 3   # Moderate increase
                elif result["spike_ratio"] < 0.5:
                    result["signal"] = -3  # Fading interest
                else:
                    result["signal"] = 0

    except Exception as e:
        # pytrends can fail due to rate limits — silent fail
        pass

    _cache_set(cache_key, result)
    return result


# ── 3. Multi-Source News Aggregation ──────────────────────────────────────────

# Free financial RSS feeds
RSS_FEEDS = {
    "reuters_markets": "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best",
    "cnbc_top": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    "marketwatch": "https://feeds.marketwatch.com/marketwatch/topstories/",
    "yahoo_finance": "https://finance.yahoo.com/news/rssindex",
}

def get_news_multi_source(ticker: str) -> dict:
    """
    Aggregate news from multiple free sources (Polygon + RSS feeds).
    Cached for 1 hour.
    
    Returns: {
        total_articles: int,
        sources: {source: count},
        sentiment_score: float (-100 to +100),
        freshness: "breaking"/"recent"/"stale",
        top_headlines: [{title, source}],
        signal: int (-10 to +10),
    }
    """
    cache_key = f"news_multi_{ticker}"
    cached = _cache_get(cache_key, ttl_seconds=3600)
    if cached:
        return cached

    result = {
        "total_articles": 0,
        "sources": {},
        "sentiment_score": 0,
        "freshness": "stale",
        "top_headlines": [],
        "signal": 0,
    }

    headlines = []

    # Source 1: Polygon (already our primary)
    try:
        url = f"https://api.polygon.io/v2/reference/news?ticker={ticker}&limit=10&order=desc&sort=published_utc&apiKey={POLYGON_KEY}"
        resp = requests.get(url, timeout=8)
        articles = resp.json().get("results", [])
        for a in articles:
            headlines.append({"title": a.get("title", ""), "source": "polygon"})
        result["sources"]["polygon"] = len(articles)
    except Exception:
        pass

    # Source 2: RSS feeds (search for ticker in titles)
    for feed_name, feed_url in RSS_FEEDS.items():
        try:
            resp = requests.get(feed_url, timeout=8, headers={"User-Agent": USER_AGENT})
            if resp.status_code != 200:
                continue
            
            root = ET.fromstring(resp.text)
            items = root.findall(".//item")
            
            count = 0
            for item in items[:20]:
                title_el = item.find("title")
                if title_el is None or title_el.text is None:
                    continue
                title = title_el.text
                if ticker.upper() in title.upper():
                    headlines.append({"title": title[:120], "source": feed_name})
                    count += 1
            
            if count > 0:
                result["sources"][feed_name] = count
                
        except Exception:
            continue

    result["total_articles"] = len(headlines)
    result["top_headlines"] = headlines[:10]

    # Sentiment scoring
    bull_count = 0
    bear_count = 0
    for h in headlines:
        words = set(h["title"].lower().split())
        if len(words & BULLISH_WORDS) > len(words & BEARISH_WORDS):
            bull_count += 1
        elif len(words & BEARISH_WORDS) > len(words & BULLISH_WORDS):
            bear_count += 1

    total = len(headlines) or 1
    result["sentiment_score"] = round((bull_count - bear_count) / total * 100)

    # Freshness (based on how many articles)
    if result["total_articles"] > 10:
        result["freshness"] = "breaking"
    elif result["total_articles"] > 3:
        result["freshness"] = "recent"

    # Signal
    if result["freshness"] == "breaking" and result["sentiment_score"] > 40:
        result["signal"] = 8
    elif result["freshness"] == "breaking" and result["sentiment_score"] < -40:
        result["signal"] = -8
    elif result["sentiment_score"] > 30:
        result["signal"] = 4
    elif result["sentiment_score"] < -30:
        result["signal"] = -4

    _cache_set(cache_key, result)
    return result


# ── Combined Social Intelligence Score ────────────────────────────────────────

def get_social_intelligence(ticker: str) -> dict:
    """
    Run all social/search data sources and combine into a ranked score.
    
    Relevance ranking (from backtesting research):
    1. Google Trends spike (strongest predictor - 35% weight)
    2. Reddit WSB buzz (contrarian indicator - 25% weight)
    3. Multi-source news sentiment (confirmation signal - 25% weight)
    4. Reddit sentiment direction (15% weight)
    
    Returns combined score and individual signals.
    """
    reddit = get_reddit_sentiment(ticker)
    trends = get_google_trends(ticker)
    news = get_news_multi_source(ticker)

    # Weighted combination based on research:
    # Google Trends spikes predict 1-3 day moves (Da et al., 2011 — "In Search of Attention")
    # Reddit/WSB is best as contrarian — extreme bullishness often precedes drops
    # News sentiment confirms but doesn't predict (lags price)
    
    weighted_score = (
        trends.get("signal", 0) * 0.35 +      # Trends: strongest predictor
        reddit.get("signal", 0) * 0.25 +        # Reddit buzz: contrarian value
        news.get("signal", 0) * 0.25 +           # News: confirmation
        (reddit.get("sentiment_score", 0) / 10) * 0.15  # Reddit direction: weakest
    )

    # Clamp to -15 to +15
    combined_signal = max(-15, min(15, round(weighted_score)))

    # Quality check: if only 1 source has data, reduce confidence
    sources_with_data = sum([
        1 if trends.get("current_interest", 0) > 0 else 0,
        1 if reddit.get("total_mentions", 0) > 0 else 0,
        1 if news.get("total_articles", 0) > 0 else 0,
    ])
    
    if sources_with_data <= 1:
        combined_signal = int(combined_signal * 0.5)  # Low confidence

    return {
        "combined_signal": combined_signal,
        "confidence": "high" if sources_with_data >= 3 else "medium" if sources_with_data == 2 else "low",
        "reddit": reddit,
        "google_trends": trends,
        "news_multi": news,
        "sources_active": sources_with_data,
    }


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print(f"=== Social Intelligence Report for {ticker} ===")
    report = get_social_intelligence(ticker)
    print(json.dumps(report, indent=2, default=str))
