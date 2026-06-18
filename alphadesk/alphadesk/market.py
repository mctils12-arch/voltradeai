"""
AlphaDesk — live market adapter (Alpaca + Polygon + Finnhub).

Uses your existing keys (see config.py). Standard-library HTTP only, no extra
dependencies. Each method is defensive: on any failure it raises, and the
LiveProvider that wraps this falls back to sample data for that field, so a
single flaky endpoint never takes down a report.

  Alpaca   — daily bars (history) + latest trade (price)
  Polygon  — /v3/reference/tickers/{T}: market cap, shares, SIC, list date
  Finnhub  — /stock/metric: P/E, beta, margins, 52w range, dividend yield

VERIFY-FIRST: these are implemented against the vendors' documented response
shapes, but field names can drift and free tiers vary. The first handoff task
is to run `python -m alphadesk <TICKER>` with your keys set and confirm the
mappings; adjust the `_map_*` helpers if a field comes back under a different
name. Until verified, treat live numbers as provisional.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import List

from .config import Keys
from .models import Quote, PriceBar


def _get_json(url: str, headers: dict | None = None, timeout: int = 20):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


class MarketAdapter:
    def __init__(self, keys: Keys):
        self.k = keys

    # ---- Alpaca ----------------------------------------------------------- #
    def _alpaca_headers(self) -> dict:
        return {
            "APCA-API-KEY-ID": self.k.alpaca_key or "",
            "APCA-API-SECRET-KEY": self.k.alpaca_secret or "",
        }

    def quote(self, ticker: str) -> Quote:
        t = ticker.upper()
        base = self.k.alpaca_data_url.rstrip("/")
        latest = _get_json(f"{base}/v2/stocks/{t}/trades/latest",
                           headers=self._alpaca_headers())
        price = float(latest["trade"]["p"])
        # Previous close from the latest daily bar.
        prev = None
        try:
            bars = _get_json(f"{base}/v2/stocks/{t}/bars?timeframe=1Day&limit=2&feed=iex",
                            headers=self._alpaca_headers())
            blist = bars.get("bars", [])
            if blist:
                prev = float(blist[-1]["c"])
        except Exception:
            pass
        return Quote(ticker=t, price=round(price, 2), prev_close=prev)

    def price_history(self, ticker: str, days: int = 365) -> List[PriceBar]:
        t = ticker.upper()
        base = self.k.alpaca_data_url.rstrip("/")
        start = (datetime.now(timezone.utc) - timedelta(days=int(days * 1.5))).strftime("%Y-%m-%d")
        out: List[PriceBar] = []
        page_token = None
        for _ in range(6):  # paginate, capped
            q = {"timeframe": "1Day", "start": start, "limit": "1000", "feed": "iex"}
            if page_token:
                q["page_token"] = page_token
            url = f"{base}/v2/stocks/{t}/bars?{urllib.parse.urlencode(q)}"
            data = _get_json(url, headers=self._alpaca_headers())
            for b in data.get("bars", []):
                out.append(PriceBar(date=b["t"][:10], close=float(b["c"])))
            page_token = data.get("next_page_token")
            if not page_token:
                break
        return out[-days:] if out else out

    # ---- Polygon ---------------------------------------------------------- #
    def polygon_reference(self, ticker: str) -> dict:
        t = ticker.upper()
        url = f"https://api.polygon.io/v3/reference/tickers/{t}?apiKey={self.k.polygon_key}"
        data = _get_json(url)
        return data.get("results", {}) or {}

    # ---- Finnhub ---------------------------------------------------------- #
    def finnhub_metrics(self, ticker: str) -> dict:
        t = ticker.upper()
        url = (f"https://finnhub.io/api/v1/stock/metric?symbol={t}"
               f"&metric=all&token={self.k.finnhub_key}")
        data = _get_json(url)
        return data.get("metric", {}) or {}
