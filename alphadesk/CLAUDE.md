# CLAUDE.md — AlphaDesk

Read this first. It tells you (Claude Code) what AlphaDesk is, how it's built,
what's real vs stubbed, and what to build next. Keep this file updated as you go.

## What AlphaDesk is
An explainable equity (and soon options) research engine. Given a ticker it
returns a buy/sell verdict assembled the way a quant + fund manager would think:
fundamentals, valuation, supply/demand, SEC filings, market context, and an
**after-tax** comparison across holding horizons (total alpha net of tax).

Clean-room: do NOT import or copy from any other project. Only the user's own
vendor API keys are reused.

## Guardrails (do not violate)
- Research/education tool. Not investment advice. Keep the disclaimer in output.
- This repo does **not** place orders. If an execution feature is ever added it
  must be paper/sandbox-first behind an explicit, off-by-default switch.
- Verdict math must stay explainable — every score traces to named factors.

## Architecture
```
alphadesk/
  models.py     dataclasses (inputs from providers; outputs from engine)
  config.py     reads vendor keys from env (Alpaca/Polygon/Finnhub + EDGAR UA)
  providers.py  DataProvider protocol; SampleProvider (offline) + LiveProvider
                + make_provider() (auto-picks live when keys present)
  market.py     MarketAdapter: Alpaca bars/quote, Polygon ref, Finnhub metrics
  analysis.py   the five scored pillars + FilingReasoner (heuristic; LLM-ready)
  tax.py        after-tax mechanics (configurable rates, not hardcoded law)
  engine.py     orchestrates -> composite -> verdict -> horizons; ModelScorer hook
  report.py     text rendering
  __main__.py   CLI: <TICKER>, --json, --sample, selftest, keys
```
Nothing imports a concrete provider/model directly — swap data sources, drop in
a trained model, or change the LLM without touching the core.

## Run it
```
pip install -e .            # no deps for the core
python -m alphadesk keys    # shows which vendor keys are detected
python -m alphadesk AAPL    # auto: live if keys set, else sample
python -m alphadesk AAPL --sample --json
python -m alphadesk selftest
```

## Keys (the user already has these)
Set in the environment — same names the user already uses:
`ALPACA_KEY`, `ALPACA_SECRET`, `POLYGON_KEY` (or `POLYGON_API_KEY`),
`FINNHUB_KEY`, and `EDGAR_USER_AGENT="AlphaDesk you@example.com"`.
See `.env.example`. Each source is independently optional; missing ones fall
back to sample data.

## Status: done vs stubbed
DONE: engine, five pillars, tax horizons, CLI, selftest (9 checks), offline
sample provider, live provider selection + Alpaca/Polygon/Finnhub adapter,
per-field graceful fallback.

STUBBED / TODO (in priority order — see TASKS.md):
1. **Verify the live adapter** against real responses; fix any field mappings
   in `market.py` / `LiveProvider.fundamentals` (Finnhub field names especially).
2. **SEC EDGAR filings** — real CIK→submissions→document fetch in
   `LiveProvider.filings` (currently sample).
3. **Market context** — real index/breadth/VIX/credit in
   `LiveProvider.market_context` (currently sample).
4. **LLM filing reader** — implement an object with `read(filings)->FilingRead`
   that calls Claude (model e.g. `claude-sonnet-4-6`) over extracted filing
   text; pass it as `filing_llm=` to `analyze()`.
5. **Options pillar** — IV rank, skew, expected move; translate an equity
   verdict into a defined-risk options expression. New module + new Pillar.
6. **Calibration/backtest** — validate weights and the score→return mapping on
   history before trusting absolute numbers.
7. **FastAPI service + React/Vite/Tailwind UI** — the Yahoo-Finance-style front
   end over the engine.

## Conventions
- Margins/ratios as fractions internally (0.42, not 42). `_pct()` converts.
- Every new pillar returns a `Pillar` of `Factor`s; add its weight to
  `DEFAULT_WEIGHTS` in engine.py.
- Keep `python -m alphadesk selftest` green; add checks for new behavior.
