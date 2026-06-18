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
DONE: engine, six pillars, tax horizons, CLI, selftest (51 checks), offline
sample provider, live provider selection + Alpaca/Polygon/Finnhub adapter,
per-field graceful fallback.
- Pre-trade planner (`planner.py`, `python -m alphadesk plan --payload <b64>`):
  fetches live price + realized vol, suggests a volatility-based stop, sizes the
  position to account value x risk-per-trade, and lays out 1R/2R/3R (+ manual)
  targets with reward:risk. Exposed via POST /api/plan and the "Planner" tab.
  Planner-only — never executes (keeps us out of broker-dealer territory).
- Tax estimator (`tax_engine.py`, `python -m alphadesk taxes --payload <b64>`):
  standalone, account-aware capital-gains + income estimate (filing status,
  state, W-2/1099 income; ST vs LT stacking; NIIT; wash-sale flagging; gains in
  Roth/IRA/HSA sheltered). Exposed via POST /api/tax/estimate and the "Taxes"
  tab. Replaced the old inline "after-tax horizon" toy in the research view.
  TY2025 brackets; estimate/education only, not tax advice. selftest covers it.
- Catalyst / news layer (`catalyst.py` + `provider.news()`): pulls Finnhub
  company-news, detects a pending cash acquisition + deal price + status, and
  derives the merger-arb framing (spread, estimated break price, market-implied
  close probability, "buy if you believe odds > X%" assessment). Surfaced as a
  top-of-report "Special situation" banner that leads the thesis. Fixes the
  ZIM-class blind spot where fundamentals miss a buyout. EDGAR now also reads
  foreign-issuer 6-K/20-F (was 10-K/10-Q/8-K only) so deal filings are seen.
- Task #1 — live Finnhub field mappings verified/corrected (see `_map_finnhub_metrics`).
- Task #2 — real SEC EDGAR filings (`edgar.py`): ticker→CIK→submissions→latest
  N material 10-K/10-Q/8-K docs with extracted text, wired into
  `LiveProvider.filings` with sample fallback. EDGAR needs no API key, only a
  descriptive `EDGAR_USER_AGENT`; a custom UA alone now activates the live
  provider (`Keys.have_edgar`).
- Task #4 — Claude filing reader (`llm_reader.py`): `ClaudeFilingReader.read()`
  sends the real EDGAR filing text to Claude (`claude-opus-4-8`, structured
  outputs) for a genuine summary / sentiment / red-flags / catalysts read,
  injected via `analyze(filing_llm=...)`. Auto-enabled when `ANTHROPIC_API_KEY`
  is set and the `anthropic` package is installed (`Keys.have_anthropic`);
  `FilingReasoner` falls back to the heuristic on any error. CLI: `--no-llm`.
- Task #3 — live market context (`market_context.py`): SPY vs 200dma (index
  trend), 11 SPDR sector ETFs' breadth above 50dma, best-effort VIX via Polygon,
  and a derived risk_on/neutral/risk_off label, wired into
  `LiveProvider.market_context`. Yields/credit have no keyless source wired and
  keep sample values; any failure leaves a field on sample.
- Task #5 — options pillar (`options.py` + `analysis.score_options`): a 6th
  pillar scoring the implied-risk environment (IV rank, put/call skew, IV vs
  realized) plus a defined-risk **OptionsStrategy** derived from the verdict +
  IV regime + expected move (credit spread when premium is rich, debit when
  cheap, iron condor / stand-aside when neutral), strikes anchored to the
  expected move. `provider.options()`: SampleProvider synthesizes a snapshot;
  LiveProvider computes realized vol + expected move from real Alpaca prices and
  leaves IV fields on sample (no keyless IV source). Surfaced on the Research
  tab. Education only — risk-defined by construction, no order placement.
- Site integration — exposed as the "Research" tab in the VolTrade site
  (`GET /api/research/:ticker` -> `python -m alphadesk --json`; React page in
  `client/src/pages/research.tsx`).

All six pillars now run on live data when keys are present.

STUBBED / TODO (in priority order — see TASKS.md):
1. **Calibration/backtest** — validate weights and the score→return mapping on
   history before trusting absolute numbers.
2. **UI polish** — the engine is already surfaced as the site's "Research" tab;
   remaining work is richer visualization / a dedicated standalone view.
3. **Yields/credit in market context** — wire a source for 10y yield, 2s10s,
   and HY OAS (e.g. Treasury par-yield API + a FRED key) to replace the sample
   values those two fields still use.

## Conventions
- Margins/ratios as fractions internally (0.42, not 42). `_pct()` converts.
- Every new pillar returns a `Pillar` of `Factor`s; add its weight to
  `DEFAULT_WEIGHTS` in engine.py.
- Keep `python -m alphadesk selftest` green; add checks for new behavior.
