"""
AlphaDesk — configuration.

Reads the SAME environment-variable names you already have set for your data
vendors, so existing keys work with zero changes. These are your own vendor
credentials (Alpaca / Polygon / Finnhub) — not anything carried over from a
prior codebase. Each source is independently optional; the engine degrades to
sample data for whatever isn't configured.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Keys:
    alpaca_key: str | None = None
    alpaca_secret: str | None = None
    alpaca_data_url: str = "https://data.alpaca.markets"
    polygon_key: str | None = None
    finnhub_key: str | None = None
    edgar_user_agent: str = "AlphaDesk research@example.com"

    @property
    def have_alpaca(self) -> bool:
        return bool(self.alpaca_key and self.alpaca_secret)

    @property
    def have_polygon(self) -> bool:
        return bool(self.polygon_key)

    @property
    def have_finnhub(self) -> bool:
        return bool(self.finnhub_key)

    @property
    def have_edgar(self) -> bool:
        """True when a descriptive (non-default) EDGAR User-Agent is configured.
        EDGAR needs no API key — only SEC's required identifying User-Agent — so
        a custom UA is enough to enable real filings."""
        return self.edgar_user_agent != "AlphaDesk research@example.com"


def load_keys() -> Keys:
    return Keys(
        alpaca_key=os.environ.get("ALPACA_KEY") or os.environ.get("APCA_API_KEY_ID"),
        alpaca_secret=os.environ.get("ALPACA_SECRET") or os.environ.get("APCA_API_SECRET_KEY"),
        alpaca_data_url=os.environ.get("ALPACA_DATA_URL", "https://data.alpaca.markets"),
        # VolTrade used either POLYGON_KEY or POLYGON_API_KEY — accept both.
        polygon_key=os.environ.get("POLYGON_KEY") or os.environ.get("POLYGON_API_KEY"),
        finnhub_key=os.environ.get("FINNHUB_KEY"),
        edgar_user_agent=os.environ.get("EDGAR_USER_AGENT", "AlphaDesk research@example.com"),
    )


def status() -> dict:
    k = load_keys()
    return {
        "alpaca": k.have_alpaca,
        "polygon": k.have_polygon,
        "finnhub": k.have_finnhub,
        "edgar_user_agent_set": k.edgar_user_agent != "AlphaDesk research@example.com",
    }
