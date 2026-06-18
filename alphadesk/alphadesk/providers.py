"""
AlphaDesk — data providers.

The engine talks to a `DataProvider`. Two are shipped:

  SampleProvider  deterministic, offline, seeded by ticker. Lets the whole
                  engine + tests run with zero network or API keys.

  LiveProvider    real adapters: SEC EDGAR for filings/fundamentals (free,
                  public) and a pluggable market-data source for quotes/
                  ownership. Network-gated; runs in your environment, not in
                  an offline sandbox. Falls back to SampleProvider fields it
                  can't fill so the engine always gets a complete object.

Add a new source by implementing the same methods. The engine never imports a
concrete provider directly.
"""
from __future__ import annotations

import hashlib
from typing import List, Optional, Protocol

from .models import (
    Quote, Fundamentals, Ownership, Filing, MarketContext, PriceBar, OptionsSnapshot,
)


# --------------------------------------------------------------------------- #
class DataProvider(Protocol):
    def quote(self, ticker: str) -> Quote: ...
    def fundamentals(self, ticker: str) -> Fundamentals: ...
    def ownership(self, ticker: str) -> Ownership: ...
    def filings(self, ticker: str, limit: int = 5) -> List[Filing]: ...
    def price_history(self, ticker: str, days: int = 365) -> List[PriceBar]: ...
    def market_context(self) -> MarketContext: ...
    def options(self, ticker: str) -> OptionsSnapshot: ...
    @property
    def name(self) -> str: ...


# --------------------------------------------------------------------------- #
def _u(t: str, salt: str) -> float:
    """Deterministic uniform[0,1) from ticker+salt."""
    h = hashlib.sha256(f"{t.upper()}|{salt}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _between(t: str, salt: str, lo: float, hi: float) -> float:
    return lo + (hi - lo) * _u(t, salt)


class SampleProvider:
    """Plausible, internally-consistent synthetic data. Deterministic."""

    name = "sample"

    def quote(self, ticker: str) -> Quote:
        t = ticker.upper()
        price = round(_between(t, "px", 8, 600), 2)
        shares = round(_between(t, "shares", 50e6, 4e9))
        return Quote(
            ticker=t, price=price, prev_close=round(price * _between(t, "pc", 0.97, 1.03), 2),
            day_volume=int(_between(t, "vol", 5e5, 8e7)),
            avg_volume_30d=int(_between(t, "avol", 5e5, 6e7)),
            market_cap=round(price * shares),
        )

    def fundamentals(self, ticker: str) -> Fundamentals:
        t = ticker.upper()
        rev = round(_between(t, "rev", 2e8, 4e11))
        gm = round(_between(t, "gm", 0.18, 0.78), 3)
        om = round(gm * _between(t, "om", 0.25, 0.7), 3)
        nm = round(om * _between(t, "nm", 0.5, 0.85), 3)
        pe = round(_between(t, "pe", 9, 55), 1)
        return Fundamentals(
            ticker=t, revenue_ttm=rev,
            revenue_growth_yoy=round(_between(t, "rg", -0.1, 0.45), 3),
            gross_margin=gm, operating_margin=om, net_margin=nm,
            fcf_ttm=round(rev * nm * _between(t, "fcfq", 0.6, 1.2)),
            fcf_margin=round(nm * _between(t, "fcfm", 0.6, 1.2), 3),
            roe=round(_between(t, "roe", 0.02, 0.4), 3),
            roic=round(_between(t, "roic", 0.02, 0.35), 3),
            net_debt_to_ebitda=round(_between(t, "nd", -1.0, 4.5), 2),
            current_ratio=round(_between(t, "cr", 0.8, 3.5), 2),
            eps_ttm=round(_between(t, "eps", 0.5, 18), 2),
            eps_growth_yoy=round(_between(t, "epsg", -0.2, 0.5), 3),
            pe=pe, forward_pe=round(pe * _between(t, "fpe", 0.75, 1.05), 1),
            ev_ebitda=round(_between(t, "ev", 6, 28), 1),
            price_sales=round(_between(t, "ps", 0.8, 14), 1),
            pe_5y_median=round(pe * _between(t, "pem", 0.7, 1.3), 1),
            sector_pe_median=round(_between(t, "spe", 12, 30), 1),
        )

    def ownership(self, ticker: str) -> Ownership:
        t = ticker.upper()
        return Ownership(
            ticker=t,
            float_shares=round(_between(t, "float", 30e6, 3e9)),
            short_pct_float=round(_between(t, "short", 0.005, 0.32), 3),
            short_days_to_cover=round(_between(t, "dtc", 0.3, 9), 1),
            insider_net_6m=round(_between(t, "ins", -2e6, 3e6)),
            institutional_pct=round(_between(t, "inst", 0.3, 0.95), 3),
            institutional_net_qoq=round(_between(t, "instq", -0.08, 0.12), 3),
            borrow_fee_pct=round(_between(t, "borrow", 0.003, 0.4), 3),
        )

    def filings(self, ticker: str, limit: int = 5) -> List[Filing]:
        t = ticker.upper()
        forms = ["10-K", "10-Q", "8-K", "4", "10-Q"]
        out = []
        for i, f in enumerate(forms[:limit]):
            out.append(Filing(
                ticker=t, form=f, filed_at=f"2026-{(i % 12) + 1:02d}-15",
                title=f"{t} {f} sample filing",
                url=f"https://www.sec.gov/sample/{t}/{f}",
                text=_sample_filing_text(t, f),
            ))
        return out

    def price_history(self, ticker: str, days: int = 365) -> List[PriceBar]:
        t = ticker.upper()
        p = _between(t, "histstart", 8, 600)
        drift = _between(t, "drift", -0.0008, 0.0014)
        bars = []
        for d in range(days):
            p *= (1 + drift + (_u(f"{t}{d}", "wiggle") - 0.5) * 0.03)
            bars.append(PriceBar(date=f"day-{days - d}", close=round(p, 2)))
        return bars

    def market_context(self) -> MarketContext:
        # Stable within a session; seeded by a fixed key.
        bias = _u("MKT", "bias")
        risk_on = bias > 0.5
        return MarketContext(
            spx_above_200dma=risk_on,
            breadth_pct_above_50dma=round(_between("MKT", "breadth", 0.3, 0.75), 2),
            vix=round(_between("MKT", "vix", 12, 30), 1),
            yield_10y=round(_between("MKT", "y10", 3.2, 4.8), 2),
            yield_curve_2s10s=round(_between("MKT", "curve", -0.6, 0.8), 2),
            credit_spread_hy=round(_between("MKT", "hy", 300, 600)),
            trend_label="risk_on" if risk_on else "risk_off",
        )

    def options(self, ticker: str) -> OptionsSnapshot:
        from .options import expected_move
        t = ticker.upper()
        iv = round(_between(t, "iv", 0.18, 0.65), 3)
        hv = round(iv * _between(t, "ivhv", 0.7, 1.2), 3)
        return OptionsSnapshot(
            ticker=t,
            iv_30d=iv,
            hv_30d=hv,
            iv_rank=round(_between(t, "ivrank", 8, 92), 1),
            put_call_skew=round(_between(t, "skew", 0.0, 0.06), 3),
            expected_move_30d=expected_move(iv, 30),
        )


def _sample_filing_text(t: str, form: str) -> str:
    tone = _u(t, "filingtone")
    pos = ("Revenue grew on strong demand; management raised full-year guidance "
           "and expanded buyback authorization.")
    neg = ("Management cited softening demand and margin pressure; the company "
           "disclosed a new investigation and going-concern language was added.")
    body = pos if tone > 0.5 else neg
    return (f"[{form}] {t}. {body} Risk factors include competition, supply "
            f"concentration, and regulatory exposure. "
            "(Sample text; the live provider returns full extracted filing body.)")


# --------------------------------------------------------------------------- #
class LiveProvider:
    """
    Real data using your existing vendor keys (Alpaca/Polygon/Finnhub) plus the
    free public SEC EDGAR. Built from `config.load_keys()`. Every field that a
    live source can't fill is backfilled from SampleProvider, so the engine
    always receives a complete object and one flaky endpoint never breaks a
    report. Each call is wrapped — a vendor error logs and falls back rather
    than raising.

    EDGAR (free, public; requires a descriptive User-Agent via EDGAR_USER_AGENT):
      - Company lookup:  https://www.sec.gov/files/company_tickers.json
      - Submissions:     https://data.sec.gov/submissions/CIK##########.json
      - Company facts:   https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json
    """

    name = "live"

    def __init__(self, keys=None):
        from .config import load_keys
        from .market import MarketAdapter
        self.keys = keys or load_keys()
        self.ua = self.keys.edgar_user_agent
        self.market = MarketAdapter(self.keys) if self.keys.have_alpaca else None
        self._fallback = SampleProvider()

    @staticmethod
    def _safe(fn, fallback_fn, *a):
        try:
            return fn(*a)
        except Exception:
            return fallback_fn(*a)

    def quote(self, ticker: str) -> Quote:
        if self.market:
            return self._safe(self.market.quote, self._fallback.quote, ticker)
        return self._fallback.quote(ticker)

    def fundamentals(self, ticker: str) -> Fundamentals:
        base = self._fallback.fundamentals(ticker)  # start from a complete object
        if not (self.market and self.keys.have_finnhub):
            return base
        try:
            m = self.market.finnhub_metrics(ticker)
            base = _map_finnhub_metrics(m, base)
        except Exception:
            pass
        return base

    def ownership(self, ticker: str) -> Ownership:
        base = self._fallback.ownership(ticker)
        if self.market and self.keys.have_polygon:
            try:
                ref = self.market.polygon_reference(ticker)
                shares = (ref.get("share_class_shares_outstanding")
                          or ref.get("weighted_shares_outstanding"))
                if shares:
                    base.shares_outstanding = float(shares)
            except Exception:
                pass
        return base

    def filings(self, ticker: str, limit: int = 5) -> List[Filing]:
        # Real SEC EDGAR: ticker -> CIK -> recent submissions -> latest N docs
        # with extracted text. Any failure (network, unknown ticker, empty
        # result) degrades to the sample filings so a report is always produced.
        try:
            from . import edgar
            out = edgar.fetch_filings(ticker, self.ua, limit=limit)
            return out or self._fallback.filings(ticker, limit)
        except Exception:
            return self._fallback.filings(ticker, limit)

    def price_history(self, ticker: str, days: int = 365) -> List[PriceBar]:
        if self.market:
            return self._safe(self.market.price_history, self._fallback.price_history, ticker, days)
        return self._fallback.price_history(ticker, days)

    def market_context(self) -> MarketContext:
        # Real backdrop from live prices: SPY vs its 200dma (index trend) and the
        # 11 SPDR sector ETFs' breadth above their 50dma, plus best-effort VIX via
        # Polygon. Yields/credit have no keyless source wired, so they keep the
        # sample values. Any failure leaves the corresponding field on sample.
        base = self._fallback.market_context()
        if not self.market:
            return base
        try:
            from . import market_context as mc
            spy = [b.close for b in self.market.price_history("SPY", days=400)]
            above = mc.above_sma(spy, 200)
            if above is not None:
                base.spx_above_200dma = above

            series = []
            for etf in mc.SECTOR_ETFS:
                try:
                    series.append([b.close for b in self.market.price_history(etf, days=80)])
                except Exception:
                    pass
            bf = mc.breadth_fraction(series, 50)
            if bf is not None:
                base.breadth_pct_above_50dma = round(bf, 2)

            if self.keys.have_polygon:
                try:
                    v = self.market.polygon_index_value("I:VIX")
                    if v:
                        base.vix = round(v, 1)
                except Exception:
                    pass

            base.trend_label = mc.derive_trend(
                base.spx_above_200dma, base.breadth_pct_above_50dma, base.vix)
        except Exception:
            pass
        return base

    def options(self, ticker: str) -> OptionsSnapshot:
        # Realized vol + a realized-vol-based expected move come from real prices
        # (Alpaca). Implied-vol fields (iv_30d, iv_rank, skew) have no cheap
        # keyless source, so they keep the sample values — the strategy builder
        # and scorer degrade gracefully on the realized-vol signal.
        base = self._fallback.options(ticker)
        if not self.market:
            return base
        try:
            from . import options as opt
            closes = [b.close for b in self.market.price_history(ticker, days=60)]
            hv = opt.realized_vol(closes, 30)
            if hv is not None:
                base.hv_30d = hv
                base.expected_move_30d = opt.expected_move(hv, 30)
        except Exception:
            pass
        return base


def _pct(v, default):
    """Finnhub returns margins/growth as percent (e.g. 42.1); engine wants 0.421."""
    if v is None:
        return default
    try:
        return round(float(v) / 100.0, 4)
    except Exception:
        return default


def _map_finnhub_metrics(m: dict, base: Fundamentals) -> Fundamentals:
    """
    Map a Finnhub /stock/metric?metric=all `metric` object onto Fundamentals.

    Pure function (no network) so it can be unit-checked offline — see
    `alphadesk selftest`. `base` is a complete object (sample fallback); only
    fields Finnhub actually provides are overwritten, everything else is left
    untouched. Percent-valued fields (margins, growth, returns) are divided by
    100 via `_pct`; ratios/multiples are taken as-is.

    Verified against documented Finnhub field names. Two fields that the prior
    handoff mapped are deliberately NOT mapped here because they are mislabeled:
      - forward_pe <- peExclExtraTTM : peExclExtraTTM is *trailing* P/E excluding
        extraordinary items, not a forward estimate. /stock/metric exposes no
        forward P/E, so forward_pe is left to the sample fallback.
      - ev_ebitda  <- currentEv/freeCashFlowTTM : that key is EV/Free-Cash-Flow,
        not EV/EBITDA. Feeding it into the EV/EBITDA factor mis-scores valuation,
        so ev_ebitda is left to the sample fallback.
    """
    # Valuation / per-share (raw ratios — no /100)
    base.pe = m.get("peTTM", base.pe)
    base.price_sales = m.get("psTTM", base.price_sales)
    base.eps_ttm = m.get("epsTTM", base.eps_ttm)
    base.current_ratio = m.get("currentRatioQuarterly", base.current_ratio)
    # Growth + margins + returns (Finnhub reports these as percent)
    base.revenue_growth_yoy = _pct(m.get("revenueGrowthTTMYoy"), base.revenue_growth_yoy)
    base.eps_growth_yoy = _pct(m.get("epsGrowthTTMYoy"), base.eps_growth_yoy)
    base.gross_margin = _pct(m.get("grossMarginTTM"), base.gross_margin)
    base.operating_margin = _pct(m.get("operatingMarginTTM"), base.operating_margin)
    base.net_margin = _pct(m.get("netProfitMarginTTM"), base.net_margin)
    base.roe = _pct(m.get("roeTTM"), base.roe)
    base.roic = _pct(m.get("roiTTM"), base.roic)
    return base


def make_provider(force_sample: bool = False) -> DataProvider:
    """SampleProvider unless any live keys are present (or forced off)."""
    from .config import load_keys
    if force_sample:
        return SampleProvider()
    k = load_keys()
    if k.have_alpaca or k.have_polygon or k.have_finnhub or k.have_edgar:
        return LiveProvider(k)
    return SampleProvider()
