# AlphaDesk

A clean-room equity research engine: give it a ticker, it returns an
**explainable buy/sell verdict** built the way a quant *and* a fund manager
would think — fundamentals, valuation, SEC filings, share supply/demand, the
broad-market backdrop, and **after-tax** outcomes across holding horizons, so
you're optimizing total alpha net of taxes rather than a headline return.

Nothing here is derived from any prior project. Fresh architecture, fresh code.

> Research and education only. **Not investment advice.** Verdicts are model
> output, not recommendations, and don't account for your full situation.

---

## What it does today (runs offline, right now)

```
python -m alphadesk NVDA --bracket 0.37 --state 0.093 --ltcg 0.20
python -m alphadesk AAPL --json
python -m alphadesk selftest        # 9 checks, all green
```

A report contains:

- **Verdict + conviction** (Strong Buy … Strong Sell) from a transparent,
  weighted composite of five pillars — no hidden weights.
- **Pillars**, each broken into auditable factors with the raw number and how
  it mapped to 0–100:
  - *Fundamentals* — growth, margins, FCF, ROIC, leverage
  - *Valuation* — P/E vs own 5y history and sector, EV/EBITDA
  - *Supply / Demand* — short %, days-to-cover, borrow fee, insider &
    institutional flow, volume (squeeze + accumulation read)
  - *Market Context* — index trend, breadth, VIX, credit spreads
  - *Filings* — narrative read of recent 10-K/10-Q/8-K (catalysts + red flags)
- **Tax-aware horizon comparison** — expected pre-tax return for a swing
  (<1y, taxed as ordinary income) vs a core hold (>1y, long-term rate), each
  converted to **after-tax** return, and the better one flagged.
- **Thesis + risks** in plain language.

## Architecture

```
alphadesk/
  models.py      dataclasses for inputs (Quote/Fundamentals/Ownership/Filing/...)
                 and outputs (Pillar/Factor/HorizonView/ResearchReport)
  providers.py   DataProvider protocol
                 - SampleProvider: deterministic offline data (no keys needed)
                 - LiveProvider:   real SEC EDGAR + pluggable market adapter
  analysis.py    the pillars + FilingReasoner (heuristic now, LLM-ready)
  tax.py         after-tax mechanics (configurable rates, not hardcoded law)
  engine.py      orchestration -> composite -> verdict -> horizons; ML hook
  report.py      text rendering
  __main__.py    CLI + selftest
```

The engine never imports a concrete provider or model — everything plugs in
through small interfaces, so you can swap data sources, drop in a trained
model, or change the LLM without touching the core.

## Going live (in your environment, with network + keys)

1. **SEC filings & fundamentals** — `LiveProvider` targets the free public
   EDGAR endpoints (`company_tickers.json`, `submissions`, XBRL `companyfacts`).
   Set `EDGAR_USER_AGENT="AlphaDesk you@example.com"` (SEC requires it). The
   adapter outline is in `providers.py`; fields it can't fill fall back to the
   sample provider so the engine always gets a complete object.

2. **Market data** (quotes, ownership, history) — pass any market adapter to
   `LiveProvider(market_adapter=...)` that implements `quote` / `price_history`.
   Use whatever vendor you have (it's vendor-neutral by design).

3. **Filing reader (the "reads like a human" part)** — implement an object with
   `read(filings) -> FilingRead` that calls an LLM, and pass it as
   `filing_llm=...` to `analyze(...)`. A Claude call (e.g. model
   `claude-sonnet-4-6`) over the extracted MD&A / risk factors / 8-K text gives
   genuine narrative analysis; the engine already routes to it with the
   heuristic as fallback.

4. **ML overlay** — implement `ModelScorer.score(features) -> 0..100` and pass
   `model=...`. The engine blends it 50/50 with the rules score so a model
   informs but never produces an unexplainable verdict.

## On the roadmap (honest status)

- **Options** — you asked for equities *and* options. The equity engine is
  built; the options layer (IV rank, skew, expected move, and translating an
  equity verdict into a defined-risk options expression) is the next pillar.
- **Live adapters** wired end-to-end (EDGAR fundamentals parsing, a concrete
  market vendor).
- **Web UI** — the Yahoo-Finance-style terminal (React + Vite + Tailwind)
  consuming a FastAPI wrapper around this engine.
- **Calibration & backtest** — turn the heuristic weights into a model trained
  and validated on history before trusting absolute numbers.

## Caveat on the numbers

The pillar weights, score mappings, and the composite→expected-return mapping
are sensible defaults, **not** calibrated against historical outcomes yet.
Treat the relative ranking as more meaningful than the absolute return figures
until the calibration pass is done. Tax rates are configurable defaults to
verify against current IRS/state guidance — not a statement of current law.
