# Options exit study — the account's own closed lots, replayed under candidate profit-taking rules (2026-11-02)

[RESEARCH] · T-BOT territory (read-only study; no rule changed here) · human question
2026-11-02: "and for the other options, figure out what's best for taking profit".
Data: `/api/diag/lots` (the sleeve ledger's closed-lot walk since 2026-05-07, 2,105
events, 405 option events → 400 lots) and `/api/diag/option_bars` (Alpaca daily
option bars for every contract held more than one day; 96 of 106 contracts returned
bars, one batch answered HTTP 502 and is excluded). Script:
`scripts/options_exit_study.py` (re-runnable with DIAG_TOKEN against the same
probes; writes results.json / results.md next to itself). PRIOR, stated before the replay (open_questions.md
2026-11-02 RULE-REVIEW candidate): a scale-out + trailing rule on bought options
raises expectancy per lot versus hold-to-21-DTE, positive but small.

## Method and its limits

Every rule is judged on DAILY CLOSES between a lot's open and its actual close:
a rule fires on the first close that satisfies it and exits at that close; a
lot closed the same day it opened has no path and keeps its actual result. This
is conservative (an intraday touch of a target is invisible) and it cannot
re-trade what the engine never held: a rule that would have kept a lot OPEN
longer than the engine did is not simulated (the engine's actual exit is the
path's end). Multiple-comparison discount: 7 bought-option rules and 3 sold-
option rules were tried; any single "winner" would carry that much selection
noise — none did.

## Result 1 — bought options: nothing to take

### Bought lots, all (same-day lots keep their actual result)

| rule | n | fired | total $ | mean $ | median $ | winners | worst | best | mean ret % |
|---|---|---|---|---|---|---|---|---|---|
| actual | 91 | 0 | -4165.0 | -45.8 | -18.0 | 8 | -1018.0 | 98.0 | -9.2 |
| tp+50% | 91 | 0 | -4165.0 | -45.8 | -18.0 | 8 | -1018.0 | 98.0 | -9.2 |
| tp+100% | 91 | 0 | -4165.0 | -45.8 | -18.0 | 8 | -1018.0 | 98.0 | -9.2 |
| tp+200% | 91 | 0 | -4165.0 | -45.8 | -18.0 | 8 | -1018.0 | 98.0 | -9.2 |
| trail35%after+50% | 91 | 0 | -4165.0 | -45.8 | -18.0 | 8 | -1018.0 | 98.0 | -9.2 |
| trail40%after+50% | 91 | 0 | -4165.0 | -45.8 | -18.0 | 8 | -1018.0 | 98.0 | -9.2 |
| trail25%after+100% | 91 | 0 | -4165.0 | -45.8 | -18.0 | 8 | -1018.0 | 98.0 | -9.2 |

### Bought lots, multi-day with a price path

| rule | n | fired | total $ | mean $ | median $ | winners | worst | best | mean ret % |
|---|---|---|---|---|---|---|---|---|---|
| actual | 7 | 0 | -1689.0 | -241.3 | -64.0 | 1 | -1018.0 | 25.0 | -33.7 |
| tp+50% | 7 | 0 | -1689.0 | -241.3 | -64.0 | 1 | -1018.0 | 25.0 | -33.7 |
| tp+100% | 7 | 0 | -1689.0 | -241.3 | -64.0 | 1 | -1018.0 | 25.0 | -33.7 |
| tp+200% | 7 | 0 | -1689.0 | -241.3 | -64.0 | 1 | -1018.0 | 25.0 | -33.7 |
| trail35%after+50% | 7 | 0 | -1689.0 | -241.3 | -64.0 | 1 | -1018.0 | 25.0 | -33.7 |
| trail40%after+50% | 7 | 0 | -1689.0 | -241.3 | -64.0 | 1 | -1018.0 | 25.0 | -33.7 |
| trail25%after+100% | 7 | 0 | -1689.0 | -241.3 | -64.0 | 1 | -1018.0 | 25.0 | -33.7 |

All 7 multi-day bought lots are LEGS OF STRUCTURES (two straddles, two
condors, one spread — `legs_opened_together` 2–4 in results.json); the account
holds no standalone bought option in its history (the selector disabled
directional buys after the 10-year backtest: buy_call WR 22 %, −2.44 %/trade).
None of the 7 ever closed a day above +50 % of its premium; the best path
topped at +$25. A profit target, a trailing stop or a scale-out rule for
bought options would have changed **no** result. The one bought leg that did
run (GOOGL 300P, +340 %) is not a closed lot: it is the wing the condor's
dismantling left behind — a STRUCTURE-management failure, not a missing
target (experiments.md 2026-11-02 ~21:45Z; fixed in v1.0.1251 by closing the
group as a unit, wing included).

VERDICT: the RULE-REVIEW candidate "profit-taking for bought options" is
WITHDRAWN for lack of a population — nothing to tune. Should the engine ever
buy standalone options again, re-run this script first.

## Result 2 — sold options: the 50 % target holds; tighter is worse

### Sold lots, all

| rule | n | fired | total $ | mean $ | median $ | winners | worst | best | mean ret % |
|---|---|---|---|---|---|---|---|---|---|
| actual | 309 | 0 | -9428.0 | -30.5 | -12.0 | 63 | -4130.0 | 1300.0 | -18.9 |
| keep25% | 309 | 29 | -10624.0 | -34.4 | -12.0 | 67 | -4130.0 | 520.0 | -19.1 |
| keep50% | 309 | 5 | -9455.0 | -30.6 | -12.0 | 63 | -4130.0 | 1300.0 | -19.0 |
| keep75% | 309 | 1 | -9451.0 | -30.6 | -12.0 | 63 | -4130.0 | 1300.0 | -18.9 |

### Sold lots, multi-day with a price path

| rule | n | fired | total $ | mean $ | median $ | winners | worst | best | mean ret % |
|---|---|---|---|---|---|---|---|---|---|
| actual | 95 | 0 | -5154.0 | -54.3 | -20.0 | 33 | -4130.0 | 1300.0 | -31.8 |
| keep25% | 95 | 29 | -6350.0 | -66.8 | -17.0 | 37 | -4130.0 | 520.0 | -32.5 |
| keep50% | 95 | 5 | -5181.0 | -54.5 | -20.0 | 33 | -4130.0 | 1300.0 | -32.3 |
| keep75% | 95 | 1 | -5177.0 | -54.5 | -20.0 | 33 | -4130.0 | 1300.0 | -32.1 |

Daily closes under-count the live rule (it fires intraday on the manager's
ticks), so "fired" is a floor. Direction is unambiguous: a 25 % target cuts the
winners (best lot $1,300 → $520) and lowers the total by ≈ $1,200; 75 % is
indistinguishable from 50 %. PROFIT_TARGET_PCT stays at 0.50 — no change
proposed, prior value logged for the record.

## Result 3 — where the money actually goes: same-day exits

Hold buckets of the 400 option lots: {'<1d': 288, '1-7d': 104, '8-30d': 7, '>30d': 1}. The sleeve ledger's
own P&L by hold (sleeves probe, 376 priced events): <1 h −$4,812 · <1 d −$5,011 ·
1–7 d −$3,784 · 8–30 d +$845 · >30 d +$104. 288 of 400 lots never saw a second
session. What closes them: the single-leg loss limit on a fresh sale (bid/ask
noise reads as −50 % within seconds; the 2026-10-13 POS-KILL scope note
recorded −54.7 % in 27 s), the Node per-leg backstop, and — on structures —
the single-leg rules running on legs that lost their grouping. This, not a
missing profit target, is the options sleeve's exit problem.

## Case study — the "option up 300 %" (GOOGL 300P, 2026-10-22 → open)

Read live through `/api/diag/positions-detail` and the audit `POS-KILL`
slice (2026-10-27 →) plus the last 200 closed orders:

| when (UTC) | what | evidence |
|---|---|---|
| ≤ 2026-10-22 | GOOGL Nov-27 iron condor opened: short 310P / long 300P, short 345C / long 355C, 2 contracts per leg | orders + the wing's $4.70 entry |
| 2026-10-22 20:15 | manager's close attempt: two per-leg day-limit buys (345C @7.25, 310P @9.30) — **canceled unfilled**; the mleg path had failed | closed orders |
| 2026-10-28 11:30 | Tier-1 POS-KILL reads both LONG WINGS as "down −99.8 %" and starts liquidating them (pre-market, order retried every minute: 122 audit lines for the 355C) | audit |
| 2026-10-28 18:56 | engine tries to OPEN another GOOGL condor ×3 — Alpaca rejects the mleg (intent mismatch: legs would close existing positions) | OPTIONS-FALLBACK |
| 2026-10-29 11:30 | short 310P read "down −309 %" (threshold −150 %) → kill | audit |
| 2026-10-29 13:30 | fills at the open: 355C wing sold @0.06, 345C short bought back @0.95, 310P short bought back @27.60; the 300P wing submitted for a market SELL on a ZERO mark ("down −100 %") — Alpaca rejected it (documented near-miss; mark guard shipped after) | orders + audit |
| 2026-11-02 | 300P wing still open: 2 ct, entry $4.70, mark $20.70, +$3,200; GOOGL root realized −$3,932 in October | positions-detail, sleeves |

Reading: the +340 % is the condor's **protective wing doing its job after
the structure it protected was dismantled leg by leg**. Net, the GOOGL
condor is ≈ −$730 (−$3,932 realized + $3,200 open). Holding the wing now
is a naked long put ≈ short 180 GOOGL shares; the 21-DTE rule sells it on
2026-11-05 (first day `dte <= 21`).

Two exit defects, neither a missing profit target:
1. **A per-leg percent kill cannot judge a defined-risk structure.** A long
   wing at −99 % is the condor working; a short leg at −150 % of its credit
   is the GROUP's loss to manage (IC half-max-loss / 2× rules in
   `_manage_strategy_group`). The Tier-1 backstop (`OPTION_LONG_KILL_LOSS_PCT`
   −75 / `OPTION_SHORT_KILL_LOSS_PCT` −150, server/bot.ts) sees legs, not
   structures.
2. **The group close is not reliable.** mleg close failed, the fallback
   placed per-leg day-limit orders at the mark that never filled, and nothing
   retried — the condor rode into earnings unmanaged.


## What ships / what is next

- SHIPPED v1.0.1251 (own PR, [REPAIR]): a multi-leg strategy is closed as a
  unit and the close is pursued until the legs are gone; shorts bought back
  before wings are sold; regular hours only; a leg that lost its label is
  adopted by its siblings.
- NEXT (own PR, [RULE-REVIEW]): the Node POS-KILL holds its hand on legs of
  a defined-risk structure (open_questions.md KNOWN BROKEN #45 (b)).
- NEXT (candidate, needs its own evidence run): the same-day exit churn on
  single sold legs — a counterfactual of "no loss-limit judgement inside the
  first session unless the mark is ≥ 3× credit" on the 288 same-day lots
  requires INTRADAY marks the archive does not keep; the options chain
  archive (post-close snapshots since 2026-10-28) is the first data that
  can answer it, after it has accumulated ~30 sessions.
- GOOGL 300P (live, +340 %): a manual sale by the human; the 21-DTE rule sells
  it on 2026-11-05 otherwise.
