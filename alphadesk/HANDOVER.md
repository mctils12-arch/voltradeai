# AlphaDesk — Session Handover (Task #1: verify the live adapter)

Date: 2026-06-18

## TL;DR
Task #1's substantive work is **done and verified offline**: the live Finnhub
field mappings were wrong in two places and incomplete in four more — all fixed,
selftest raised from 9 to **16 checks, all green**. The work could **not** be
committed to the right repo or verified against real vendor data **from this
session**, for two environment reasons (wrong repo scope; keys live on Railway,
not here). Deliverables are handed back as a zip + patch to apply where AlphaDesk
actually lives.

---

## What AlphaDesk is (context for whoever picks this up)
Explainable equity research engine. Ticker in → buy/sell verdict from five
weighted pillars (fundamentals, valuation, supply/demand, market context,
filings) + an after-tax horizon comparison. Clean interface seams; the engine
never imports a concrete provider. `SampleProvider` (offline, deterministic) and
`LiveProvider` (Alpaca/Polygon/Finnhub + SEC EDGAR) behind a `DataProvider`
protocol. Guardrails: research/education only (disclaimer in every output), no
order placement, explainable scores, keep `selftest` green, clean-room.

## Environment findings (important — these shaped everything)
- **No vendor keys in this sandbox.** `python -m alphadesk keys` → all `false`.
  The keys live on **Railway**. Consequence: in this session "live" falls back
  100% to sample — `AAPL --json` and `AAPL --sample --json` are byte-identical,
  both report `provider: sample`. Real "live numbers fill in" verification can
  only happen on Railway, where the keys are.
- **Network is open** (Finnhub/Alpaca/Polygon/SEC all reachable). Keys, not
  connectivity, are the only gate.
- **Wrong repo connected.** This session is scoped to `mctils12-arch/across-the-table`
  (the Charlotte events app). AlphaDesk belongs with `mctils12-arch/voltradeai`.
  A read of voltradeai was **access-denied**, and the add-repo tool isn't
  available in this session — so nothing could be pushed to the right place from
  here. Work was handed back as files instead of committed.

## What was changed (Task #1)
File: `alphadesk/providers.py` and `alphadesk/__main__.py`.

Two **wrong** Finnhub mappings (fed bad numbers into valuation factors):
| Field | Was | Problem | Now |
|---|---|---|---|
| `forward_pe` | `peExclExtraTTM` | trailing ex-extraordinary P/E, NOT forward; spuriously triggered the "earnings expected to grow" valuation bonus | left to sample fallback (Finnhub /stock/metric has no forward P/E) |
| `ev_ebitda` | `currentEv/freeCashFlowTTM` | that's EV/**Free-Cash-Flow**, not EV/EBITDA; mis-scored the EV/EBITDA factor | left to sample fallback |

Four fields Finnhub **provides but the code never read** (so they stayed
synthetic under "live"):
- `revenue_growth_yoy` ← `revenueGrowthTTMYoy` (÷100)
- `eps_growth_yoy` ← `epsGrowthTTMYoy` (÷100)
- `eps_ttm` ← `epsTTM`
- `current_ratio` ← `currentRatioQuarterly`

Other:
- Polygon shares: falls back to `weighted_shares_outstanding` if
  `share_class_shares_outstanding` is absent.
- Refactor: extracted the mapping into a **pure** `_map_finnhub_metrics(m, base)`
  (no network) so it's unit-checkable offline.
- `selftest`: added 7 offline checks that push a Finnhub-shaped payload through
  the mapper and assert correct scaling (percent→fraction for margins/growth/
  returns; raw for ratios) and that the two mislabeled keys are NOT consumed.
  **9 → 16 checks, all green.**

Unchanged (already correct): `pe←peTTM`, `price_sales←psTTM`, margins
(`grossMarginTTM`/`operatingMarginTTM`/`netProfitMarginTTM`), `roe←roeTTM`,
`roic←roiTTM`, and the Alpaca quote/price-history adapter.

## Before/after proof (realistic AAPL Finnhub payload, run offline)
```
field                 SAMPLE     OLD map     NEW map   note
forward_pe              18.0        30.9        18.0   OLD=trailing artifact (WRONG); NEW=fallback
ev_ebitda              21.9        29.7        21.9   OLD=EV/FCF (WRONG factor); NEW=fallback
eps_ttm               17.81       17.81        6.13   NEW now from live (was sample-only)
current_ratio          0.87        0.87        0.92   NEW now from live (was sample-only)
revenue_growth_yoy    0.194       0.194       0.078   NEW now from live (was sample-only)
eps_growth_yoy        0.485       0.485       0.114   NEW now from live (was sample-only)
```

## Deliverables (handed back as files)
- `alphadesk-task1-fixed.zip` — full corrected project (no .git/.pyc).
- `alphadesk_task1_live_adapter.patch` — standalone diff of the two changed files.
- `HANDOVER.md` — this file.

Apply from the AlphaDesk project root (dir containing the inner `alphadesk/`
package):
```bash
git apply alphadesk_task1_live_adapter.patch
python -m alphadesk selftest        # expect all_pass, 16/16
```

## What's NOT done / next steps
1. **Real live verification** — must run on Railway (keys are there). After
   applying: `python -m alphadesk AAPL --json` vs `--sample --json` with keys set;
   confirm live numbers populate. If your Finnhub tier returns a metric under a
   different key, it's a one-line change in `_map_finnhub_metrics`.
2. **Get the code into the right repo.** Re-scope a Claude Code web session to
   `voltradeai` (or wherever AlphaDesk should live). Open question: is AlphaDesk
   already a folder in voltradeai (then the patch applies), or should it be its
   own repo (then it's "add the project," not "patch")? Clean-room note suggests
   possibly its own repo.
3. **Railway deploy** won't auto-update from a feature-branch push — Railway
   redeploys on a push to the branch it watches (usually `main`), and only if
   it's connected with auto-deploy on. Merge to the deploy branch to redeploy.
4. **Then Task #2** (per kickoff, stop and confirm first): SEC EDGAR filings —
   real CIK→submissions→document fetch in `LiveProvider.filings` (currently
   sample). After that: real market context (#3), LLM filing reader (#4), options
   pillar (#5), calibration/backtest (#6), FastAPI+React UI (#7).

## Guardrails kept
No execution/trading code added. Disclaimer untouched. Scores still trace to
named factors. selftest green (added a check for the changed behavior).
Clean-room: nothing imported/copied from another project.
