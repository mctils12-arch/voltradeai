# AlphaDesk — build backlog (priority order)

Pick the top unchecked item. Keep `python -m alphadesk selftest` green.

- [ ] 1. VERIFY LIVE ADAPTER. Run `python -m alphadesk AAPL` with keys set.
      Confirm price/history (Alpaca) and metrics (Finnhub) populate. Fix field
      mappings in `market.py` and `LiveProvider.fundamentals` if a value is
      None or wrong (Finnhub metric names drift by tier). Add a `--json` diff
      against `--sample` to see what live actually filled.
- [ ] 2. SEC EDGAR FILINGS. Implement `LiveProvider.filings`: company_tickers
      -> CIK -> submissions -> fetch latest N (10-K/10-Q/8-K) and extract text.
      Use the EDGAR_USER_AGENT header. Cache by CIK (filings change slowly).
- [ ] 3. REAL MARKET CONTEXT. Replace sample `market_context` with real SPX vs
      200dma, breadth, VIX, 2s10s, HY spread (Alpaca/Polygon where possible).
- [ ] 4. LLM FILING READER. Object with `read(filings)->FilingRead` calling
      Claude over MD&A/risk-factors/8-K text. Pass via `analyze(filing_llm=...)`.
      Keep the heuristic as fallback.
- [ ] 5. OPTIONS PILLAR. New module: IV rank, skew, expected move; map the
      equity verdict to a defined-risk options structure. Add a Pillar + weight.
- [ ] 6. CALIBRATION/BACKTEST. Validate pillar weights + score->return mapping
      on history. Until then, absolute return numbers are provisional.
- [ ] 7. API + UI. FastAPI wrapper around `analyze()`, then React + Vite +
      Tailwind terminal (search, verdict, pillar breakdown, horizon/tax view).
