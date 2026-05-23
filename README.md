# ETF Builder + Analyze Dropdown — install guide

## What you got

### New files (drop in)
| File | Goes to |
|---|---|
| `etf_analyzer.py` | repo root (next to `analyze.py`, `insights.py`) |
| `ETFBuilderView.tsx` | `client/src/pages/ETFBuilderView.tsx` |

### Modified files (replace existing)
| File | Goes to |
|---|---|
| `home.tsx` | `client/src/pages/home.tsx` |
| `analyze.tsx` | `client/src/pages/analyze.tsx` |
| `InsightsView.tsx` | `client/src/pages/InsightsView.tsx` |
| `routes.ts` | `server/routes.ts` |
| `index.css` | `client/src/index.css` |

No new dependencies — `yfinance` is already in `requirements.txt`.

## What it does

### 1. Top-nav "Analyze" hover dropdown
Hovering the **Analyze** tab now opens a menu with four sections, each routing
to the right view:
- **Options & Volatility** — original options view (IV/HV, VRP, spreads, vol cone)
- **Smart Money** — Insider + Institutional + Flow + Options Smart Money cards
- **Structure** — Float/Short + Support & Resistance cards
- **ETF Builder** — the new view

Deep-linkable: `/app#/analyze/etf-builder` etc.

Mobile: dropdown doesn't fire on touch, so mobile still uses the in-page
sub-tab switcher (which is now 3-way: Options · Smart Money · Structure). The
ETF Builder is currently desktop-dropdown-only on mobile — see "Known
limitations" below if you want it added to the mobile bottom bar too.

### 2. ETF Builder / Analyzer
Enter an ETF ticker (SPY, QQQ, VOO, ARKK, etc.). You get:

- **Header card** — long name, family, expense ratio (bps), AUM, yield,
  inception, **active vs passive** badge, **tracks index**, **rebalance
  schedule** (typical-by-family — Yahoo doesn't publish actual dates).
- **Performance & Contribution** — switch between 1M/3M/6M/YTD/1Y/5Y.
  Top contributors and biggest drags are called out. Sortable table with
  per-holding return, vs-ETF, and contribution-to-return (in bps).
- **Replication Calculator** — slider ($100–$1M) + numeric input. Side-by-side
  card: "Buy the ETF (one fund)" vs "Replicate with N holdings". Per-holding
  table shows weight, $ allocation, price, share count, actual cost.
  Fractional shares toggle.
- **Sector Breakdown** — bar chart.
- **Holdings Explorer** — expandable row per holding with the full
  Yahoo Finance dump: sector, industry, country, exchange, currency,
  **first trade date**, **years listed**, **SPAC heuristic flag**,
  headquarters, employees, market cap, shares outstanding, trailing/forward
  P/E, beta, dividend yield, 52-week high/low with position bar, full business
  summary, and website link.

## Data sources & honest limits

- **All data from Yahoo Finance via `yfinance`**. Backend caches 1h per
  holding, 15 min per full ETF payload.
- **Holdings**: Yahoo returns top-N (typically 10) per ETF. Full holdings
  require scraping the issuer (SSGA/iShares/Vanguard) — not implemented. UI
  surfaces "Partial holdings: covers X% of fund weight" when truncated.
- **Rebalance dates**: Yahoo doesn't publish actual rebalance dates. Backend
  returns a *typical schedule by family* (S&P quarterly, Russell annual,
  CRSP quarterly, etc.) and labels it as such. Active funds get
  "discretionary, no fixed schedule".
- **Active vs passive**: heuristic combining a known-actives list (ARK
  family, JEPI/JEPQ, AVUV, etc.) with name pattern matching.
- **SPAC flag**: name-pattern heuristic only (matches "Acquisition Corp"
  etc.). For real SPAC detection you'd need SEC filing analysis. UI labels
  the flag as "flagged by name pattern" so users know it's not definitive.
- **Listing date**: `firstTradeDateEpochUtc`. For de-SPAC'd companies this is
  the original SPAC IPO date, NOT the operating company's public debut. UI
  labels it "First trade date" accurately rather than "IPO date".

## First-run perf

Per-holding `yfinance.info` calls are ~1–2s each. Backend parallelizes 6 at
a time, so a 10-holding ETF takes ~3–5s on the first uncached request,
~instant after. The frontend shows "10–30 seconds on first run" to set
expectations.

## Known limitations / possible follow-ups

1. **Mobile ETF Builder access** — there's no mobile bottom-bar entry for
   it; mobile users can only reach it via deep link
   (`/app#/analyze/etf-builder`) or by editing the URL hash. If you want a
   mobile affordance, the cleanest options are: (a) add it as a 4th button
   in the in-page sub-tab switcher (but the model is different — ETF
   ticker not stock ticker — so behavior is jarring); (b) add a 6th icon
   to the mobile bottom bar; (c) use a long-press menu on the mobile
   Analyze button.
2. **Full holdings** — if you want all 500 SPY holdings, the path is
   per-issuer scraping (SSGA CSV download for SPY, iShares JSON for IVV
   etc.). Brittle but doable as a follow-up.
3. **Real rebalance dates** — also per-issuer scraping (SSGA discloses,
   iShares doesn't always).
4. **Authoritative SPAC detection** — requires SEC EDGAR filings analysis
   (S-1A filings, 8-K with item 2.01 for business combination). Out of scope
   here.

## Code quality notes

- Python matches the project's existing pattern (`analyze.py`, `insights.py`,
  `finnhub_data.py`): JSON-to-stdout, atomic cache writes via
  `tempfile + os.replace`, `_clean_nan` traversal before serialization,
  `_safe_float`/`_safe_int` coercion for numpy types.
- React matches the project's inline-style + design-token approach.
- yfinance schema drift is handled with multiple fallback paths
  (`top_holdings` as DataFrame OR dict, weights as 0-1 OR 0-100,
  `fund_operations` row names varying by version, etc.) — should survive a
  yfinance version bump or two.

## How to test

```bash
# Backend smoke test (from repo root, after `pip install -r requirements.txt`)
python3 etf_analyzer.py SPY | python3 -m json.tool | less

# Should return an ETF result with `info`, `holdings`, `sector_weights`,
# `etf_performance`. AAPL (not an ETF) should return
# {"error": "AAPL is a Equity, not an ETF..."}
```

Frontend just builds normally — no new packages.
