# ETF Builder — Production Data Source Edition

This drop replaces the prior `yfinance + Stooq` data layer with the **same
sources the trading bot uses for live decisions**: Alpaca SIP, Polygon, and
Finnhub. yfinance is kept only for the ETF-specific fund_data accessors it's
genuinely good at (expense ratio, sector weightings). Everything else now
runs on the institutional-grade feeds you're already paying for.

## What changed since the last drop

Previously I added a multi-source layer using NASDAQ + Stooq + yfinance. After
auditing the rest of the codebase I realised the bot already uses:

- **Alpaca Market Data API** (`data.alpaca.markets/v2/stocks/bars` with
  `feed=sip&adjustment=all`) — used in `bot_engine.py`, `intraday_shorts.py`,
  `csp_universe.py`, etc. The existing `_fetch_alpaca_bars` helper in
  `analyze.py` handled this for individual stocks but the ETF Builder
  was still on yfinance for some reason.
- **Polygon `/v3/reference/tickers/{T}`** — already used in `alt_data.py` and
  `macro_data.py` for shares outstanding. Returns full company reference data
  including a clean `type` field (ETF / CS / REIT / FUND / ETN / etc).
- **Finnhub `/stock/metric`** via the existing `finnhub_data.py` wrapper —
  P/E, beta, dividend yield, 52w range.

Stooq was an OK free fallback but it's no longer needed when Alpaca's SIP
feed is available. NASDAQ is still the holdings-list source (Polygon's ETF
holdings is on their premium tier).

## Files

```
etf_data_sources.py     REWRITE — Alpaca + Polygon + Finnhub + NASDAQ
etf_analyzer.py         REWRITE — uses batch endpoints, much faster
ETFBuilderView.tsx      UPDATED — classification panel, full-holdings UI
analyze.tsx             UPDATED — unified search
home.tsx                UPDATED — hover dropdown
InsightsView.tsx        UPDATED — filter prop
routes.ts               UPDATED — /api/etf/:ticker endpoint
index.css               UPDATED — dropdown + section switcher
```

## How data flows now

Single ETF request (e.g. `GET /api/etf/QQQ`):

1. **Classify the ticker** via Polygon `/v3/reference/tickers/QQQ`. The `type`
   field is authoritative (`ETF`, `CS`, `REIT`, `FUND`, `ETN`, `ADRC`, `PFD`,
   `WARRANT`, etc.). If type ≠ ETF/ETN/ETV, return a friendly classified
   error immediately. No yfinance call needed for the rejection.

2. **Get full holdings list** from NASDAQ's public API
   (`api.nasdaq.com/api/quote/QQQ/etfdetail/holdings?limit=9999`). Returns
   all 100 holdings of QQQ, all 500 of SPY. Falls back to yfinance top-10
   only if NASDAQ fails.

3. **ONE batch Alpaca call** for daily bars on `[ETF_ticker] + [all 100
   holdings]`. Up to 50 symbols per HTTP request, paginated, parallelized.
   For QQQ: ~3 HTTP requests total covers the ETF + all 100 holdings with
   5 years of daily SIP-adjusted closes. Then slice in code for all six
   periods (1M/3M/6M/YTD/1Y/5Y) and current price.

4. **Polygon ticker details** for the top 30 holdings (parallel,
   ThreadPoolExecutor with 8 workers). Sector via SIC code mapping
   (full table in `_SIC_RANGE_OVERRIDES`), industry from SIC description,
   `list_date` for first trade date, market cap, shares outstanding,
   description, exchange, country, employees, website.

5. **Finnhub metrics** for the same top 30 (also parallel). Trailing P/E,
   forward P/E, beta, dividend yield, 52w high/low. Fills the gaps Polygon
   doesn't have in its reference endpoint.

6. **yfinance** for the ETF-level fund data only: expense ratio
   (`fund_operations`), sector weightings (`sector_weightings`), total AUM
   (`info.totalAssets`), 30D SEC yield (`info.yield`), category, family.
   These are the fund-specific things yfinance's `funds_data` accessor
   actually does well.

7. **Long-tail holdings** (positions 31 through N) get a slim record:
   symbol, name, weight, current price, multi-period returns from the
   Alpaca batch. No Polygon or Finnhub call — keeps SPY-sized funds fast.

End result for QQQ: ~6 HTTP requests total (1 NASDAQ + 3 Alpaca bars batches
+ 30 Polygon details + 30 Finnhub metrics, with Polygon/Finnhub running
concurrently). With caching, repeat calls are near-instant.

## Data source priority per concern

| Concern                    | Primary       | Fallback        | Notes |
|---------------------------|---------------|------------------|-------|
| Ticker classification      | Polygon `type` field | yfinance heuristic | Polygon is authoritative |
| Full holdings list         | NASDAQ public API | yfinance top-10 | Polygon ETF holdings is paid tier |
| Daily price history        | Alpaca SIP    | (none — Alpaca is reliable) | Adjusted for splits + divs |
| Current price              | Alpaca latest bar | yfinance.info | |
| Per-holding company data   | Polygon reference | (slim record if unavailable) | sector via SIC mapping |
| P/E, beta, dividend yield  | Finnhub metrics | (omitted if unavailable) | |
| ETF expense ratio          | yfinance `fund_operations` | yfinance `info` | |
| ETF sector weightings      | yfinance `sector_weightings` | (omitted) | |
| ETF AUM                    | yfinance `totalAssets` | (omitted) | |
| ETF inception date         | Polygon `list_date` | Static table → yfinance `firstTradeDate` | |

## Graceful degradation

The data layer checks for API keys at startup (`have_alpaca()`,
`have_polygon()`, `have_finnhub()`). Each source is independently optional:

- **No `ALPACA_KEY`?** → No price history, no current prices.
  Performance returns will all be None. (Set the env vars on Railway and
  this fills in.)
- **No `POLYGON_KEY`?** → No rich per-holding metadata. Classification
  falls back to yfinance heuristic. Slim records still work via Alpaca.
- **No `FINNHUB_KEY`?** → No P/E / beta / dividend yield. Everything else
  still works.
- **All three missing?** → Backend still serves a response but most fields
  are None. The UI will show the holdings list (from NASDAQ + yfinance)
  with a lot of dashes. Same as the old yfinance-only behavior.

The response includes `data_sources_used: {alpaca, polygon, finnhub}`
booleans so you can verify the env vars are loaded on Railway.

## SIC → GICS-style sector mapping

Polygon returns the SEC's SIC code on every company. The new
`_sic_to_sector()` function maps SIC ranges to coarse GICS-style sectors:

- 1311-1389, 2900-2999 → Energy
- 2800-2829, 2837-2899 → Basic Materials
- 2830-2836 → Healthcare (pharma)
- 4812-4899 → Communication Services
- 4911-4961 → Utilities
- 5411-5499 → Consumer Defensive
- 5810-5829 → Consumer Cyclical
- 6020-6411 → Financial Services
- 6500-6799 → Real Estate (incl. REITs at 6798-6799)
- 7370-7389 → Technology
- 8000-8099 → Healthcare
- Fallback: first-digit bucket (3xxx → Industrials, etc.)

Tested against 15 sample SIC codes spanning all sectors — all map correctly.
Order matters: narrow ranges (pharma 2830-2836) are checked before wider
ones (chemicals 2800-2899) so they don't get shadowed.

## Classification IDs returned on rejection

When a ticker isn't an ETF, the backend returns `error_classification`
which the frontend renders with a coloured badge + suggested alternative
ETF where applicable:

| ID            | Source                       | Suggested ETF |
|---------------|------------------------------|---------------|
| `etf`         | Polygon type=ETF/ETN/ETV     | — (proceeds)  |
| `stock`       | Polygon type=CS              | —             |
| `reit`        | Polygon type=REIT or SIC 6798| VNQ           |
| `mreit`       | mortgage in SIC desc or name | REM           |
| `mutual_fund` | Polygon type=FUND            | —             |
| `adr`         | Polygon type=ADRC/ADRP       | —             |
| `preferred`   | Polygon type=PFD             | —             |
| `warrant`     | Polygon type=WARRANT         | —             |
| `index`       | symbol starts with `^`       | SPY           |
| `crypto`      | symbol ends with `-USD`      | —             |
| `unknown`     | nothing matched              | —             |

## Required env vars on Railway

Your bot is already using all three but verify these are set in your
Railway service:

```
ALPACA_KEY=...            # Alpaca account key
ALPACA_SECRET=...         # Alpaca secret
POLYGON_KEY=...           # or POLYGON_API_KEY — both checked
FINNHUB_KEY=...           # already in use for insights
```

If any are missing the data layer degrades silently rather than crashing.

## Cache strategy

- ETF payload: 15min
- NASDAQ holdings: 6h
- Polygon ticker details: 24h (reference data changes slowly)
- Finnhub metrics: 6h
- Per-holding combined record (Polygon + Finnhub merged): 1h

Cache key prefix is `etf_v3_*` — old `etf_v2_*` and `etf_*` entries are
ignored, so deploying invalidates the prior cache automatically.

## Verifying after deploy

`GET /api/etf/QQQ` should return:
- `holdings_source: "nasdaq"` not `"yfinance_top_n"`
- `holdings_returned: 100` not 10
- `holdings_coverage: ~0.99`
- `holdings_truncated: false`
- `etf_performance` filled in (all six periods)
- `info.inception_date: "1999-03-10"` (from Polygon's list_date)
- `data_sources_used: {alpaca: true, polygon: true, finnhub: true}`
- Each holding has `data_level: "full"` (top 30) or `"slim"` (rest)
- Top 30 holdings have non-null `sector`, `industry`, `market_cap`,
  `trailing_pe`, `beta`, etc.

`GET /api/etf/ORC`:
- `error_classification: "mreit"` (mortgage REIT)
- `suggested_ticker: "REM"`
- Friendly message explaining what ORC actually is

`GET /api/etf/AAPL`:
- `error_classification: "stock"`
- Message points user to the regular Analyze view

## A note on the rest of the site

You mentioned the bot uses better sources than the public-facing analyze
views. Quick audit: `analyze.py` already calls `_fetch_alpaca_bars()` for
price history (good), but `insights.py` and some others still fall back
to yfinance for per-ticker metadata. The same `fetch_ticker_details_polygon()`
+ `fetch_metrics_finnhub()` helpers in `etf_data_sources.py` could be
imported and used to upgrade those too — happy to do that as a follow-up
pass if you want the whole site moved off yfinance for per-ticker data.

Most likely candidates for migration:
- `insights.py` — uses yfinance.info for company sector/industry/employees;
  Polygon reference would be faster + more reliable
- `analyze.py` lines 1902-1920 — yfinance.info for company fundamentals;
  could mostly move to Polygon + Finnhub
- Any place using `yf.Ticker(x).info` for sector / industry / marketcap /
  shares_outstanding / business summary
