# Research test 2 (gate 3): the manager's rules on the CSPs the bot would have sold — 2026-10-26

Generated 2026-10-26T21:30:21Z by scripts/csp_manager_rule_sim.py — entry rows 8635 (913 with a selectable contract) over 80 archived days and 1821 underlyings; deduped to 448 entries (5-day window); bars 262 names; source /tmp/claude-0/-home-user-voltradeai/1c64125e-f4dc-5e5f-9984-1cb6a5ea06e4/scratchpad/csp_sim/entries_dump.json.

## PRIOR (stated before the run)

Stated before the run: 60% that the live rules show NEGATIVE expectancy per contract net of the archived spread on the deduped single-name set (the sleeve lost when it held); 70% that `no_stop` beats `live` (the -2x stop crystallizes the left tail); 55% that `no_dte_exit` beats `live` by less than the spread it saves; 65% that `earnings_gate` removes most of the worst-10 entries.

## VERDICT

live rules: expectancy -0.23% of collateral per entry (mean $-21 ± 28 SE per contract, win rate 64%, n 381); no_stop: +0.15 pts vs live (n 375; paired $+0 ± 2 per contract, 23 of 375 entries changed); no_dte_exit: +0.79 pts vs live (n 379; paired $+71 ± 18 per contract, 155 of 379 entries changed); earnings_gate: +0.23 pts vs live (n 233; dropped entries averaged $-65 under live rules (n 148), $-22 without APH). Four variants were tried: discount the best by four. Model marks, not fills.

## Variants (per contract; sold at the archived bid, bought back at the MODEL mark + half the entry spread)

| set | n | closed | censored | mean $ | SE | median $ | p5 | p95 | win | exp % coll | spread $ | hold d |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| live | 442 | 381 | 61 | -21 | 28 | +47 | -692 | +505 | 64% | -0.23 | 28 | 13.1 |
| no_dte_exit | 442 | 379 | 63 | +51 | 31 | +103 | -734 | +684 | 78% | +0.56 | 27 | 18.1 |
| no_stop | 442 | 375 | 67 | -3 | 27 | +51 | -648 | +509 | 65% | -0.07 | 28 | 13.6 |
| earnings_gate | 249 | 233 | 16 | +6 | 25 | +43 | -558 | +461 | 64% | +0.00 | 27 | 13.4 |

## Exit reasons

| variant | profit_target | loss_limit | dte_exit | dte_critical | expiry | censored | skipped (earnings) |
|---|---|---|---|---|---|---|---|
| live | 193 | 31 | 157 | 0 | 0 | 61 | 0 |
| no_dte_exit | 290 | 55 | 0 | 0 | 34 | 63 | 0 |
| no_stop | 193 | 0 | 182 | 0 | 0 | 67 | 0 |
| earnings_gate | 118 | 18 | 97 | 0 | 0 | 16 | 195 |

## Paired vs live (same entry, both closed; $ per contract = variant minus live; unpaired means above mix the rule with the censoring it changes)

| variant | set | n | mean diff $ | SE | t | entries changed |
|---|---|---|---|---|---|---|
| no_stop | all | 375 | +0.5 | 2.2 | 0.23 | 23 |
| no_stop | BEAR | 27 | +9.6 | 6.65 | 1.44 | 2 |
| no_stop | BULL | 248 | +0.9 | 2.07 | 0.45 | 15 |
| no_stop | NEUTRAL | 100 | -3.1 | 6.18 | -0.49 | 6 |
| no_dte_exit | all | 379 | +71.4 | 17.65 | 4.05 | 155 |
| no_dte_exit | BEAR | 27 | +239.5 | 79.31 | 3.02 | 13 |
| no_dte_exit | BULL | 251 | +50.0 | 20.03 | 2.5 | 112 |
| no_dte_exit | NEUTRAL | 101 | +79.6 | 37.4 | 2.13 | 30 |

earnings_gate is a subset, not a rule change — under LIVE rules the entries it keeps averaged $+6 (n 233) and the entries it drops $-65 (n 148); without the single worst dropped entry (APH 2026-07-16, $-6,353) the dropped set averaged $-22.

## Live rules by entry-day regime

| set | n | closed | censored | mean $ | SE | median $ | p5 | p95 | win | exp % coll | spread $ | hold d |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BEAR | 27 | 27 | 0 | -196 | 105 | -33 | -815 | +199 | 48% | -2.05 | 29 | 18.8 |
| BULL | 265 | 253 | 12 | -57 | 38 | +42 | -711 | +399 | 61% | -0.47 | 27 | 13.3 |
| CAUTION | 4 | 0 | 4 | — | — | — | — | — | — | — | — | — |
| NEUTRAL | 146 | 101 | 45 | +113 | 37 | +92 | -347 | +551 | 75% | +0.87 | 31 | 11.0 |

## Live rules by entry DTE

| set | n | closed | censored | mean $ | SE | median $ | p5 | p95 | win | exp % coll | spread $ | hold d |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30-35 | 138 | 104 | 34 | -24 | 36 | +23 | -540 | +325 | 61% | -0.32 | 21 | 8.8 |
| 36-42 | 154 | 133 | 21 | -61 | 65 | +55 | -937 | +448 | 65% | -0.56 | 31 | 12.2 |
| 43-50 | 150 | 144 | 6 | +17 | 38 | +68 | -732 | +519 | 65% | +0.15 | 31 | 17.0 |

## Worst 10 entries, live rules

| u | entry | exp | strike | credit | exit | reason | pnl | S at exit |
|---|---|---|---|---|---|---|---|---|
| APH | 2026-07-16 | 2026-08-21 | 145.0 | 571 | 2026-07-17 | loss_limit | -6,353 | 75.48 |
| RDDT | 2026-07-13 | 2026-08-21 | 175.0 | 1095 | 2026-07-31 | loss_limit | -2,587 | 140.67 |
| NBIS | 2026-08-17 | 2026-09-18 | 240.0 | 1505 | 2026-08-28 | dte_exit | -2,572 | 209.18 |
| RDDT | 2026-07-08 | 2026-08-21 | 175.0 | 1144 | 2026-07-31 | loss_limit | -2,507 | 140.67 |
| ARM | 2026-10-12 | 2026-11-20 | 240.0 | 1122 | 2026-10-22 | loss_limit | -2,454 | 210.10 |
| ANET | 2026-10-12 | 2026-11-20 | 210.0 | 1124 | 2026-10-16 | loss_limit | -2,412 | 176.98 |
| TPR | 2026-08-06 | 2026-09-18 | 150.0 | 638 | 2026-08-13 | loss_limit | -1,728 | 127.91 |
| SKHY | 2026-07-15 | 2026-08-21 | 165.0 | 1692 | 2026-07-31 | dte_exit | -1,356 | 143.73 |
| MXL | 2026-08-17 | 2026-09-18 | 80.0 | 661 | 2026-08-28 | dte_exit | -1,291 | 61.73 |
| NBIS | 2026-07-23 | 2026-08-28 | 225.0 | 4321 | 2026-08-07 | dte_exit | -1,219 | 187.97 |

## Live sleeve vs simulation, by underlying (names in both; live = realized from fills, sim = live-rules sum)

| u | live realized $ | live fills | sim n | sim pnl sum $ |
|---|---|---|---|---|
| BAC | -226 | 8 | 1 | +29 |
| BB | -484 | 14 | 1 | -90 |
| BMNR | -111 | 5 | 4 | +342 |
| BMY | -49 | 4 | 1 | +58 |
| CIFR | -263 | 11 | 1 | -69 |
| DAL | -56 | 2 | 4 | +261 |
| DRAM | -110 | 29 | 3 | +390 |
| EWZ | -366 | 10 | 1 | -1 |
| HOOD | -66 | 12 | 1 | +216 |
| HPQ | -48 | 3 | 1 | +42 |
| NEE | -161 | 7 | 1 | -22 |
| NKE | -234 | 14 | 2 | -86 |
| NVDA | -1,126 | 133 | 2 | +707 |
| ORCL | -315 | 8 | 1 | +396 |
| PATH | -972 | 42 | 3 | +125 |
| SLB | -88 | 5 | 2 | +145 |
| SMCI | -46 | 4 | 1 | +91 |
| STM | -95 | 6 | 1 | +109 |
| T | -199 | 25 | 3 | +47 |
| TEM | -422 | 10 | 1 | +119 |
| VALE | -60 | 8 | 1 | +24 |
| VZ | -63 | 10 | 1 | -141 |
| WMB | -425 | 5 | 1 | -51 |

## Honesty

- Every exit is a Black–Scholes MODEL MARK at the entry's implied vol (constant), not a fill: it smooths the intraday path the −2× stop fires on, so the stop's cost reads as a LOWER bound, and it ignores the vol change a crash brings (an under-mark on the way down).
- Costs: sold at the archived bid; bought back at the model mark plus half the entry spread (modelled). r = 4.00%, a stated constant. Calendar DTE as the manager uses it.
- The probe's liquidity test lacks the volume branch the live `_is_liquid` has: this universe is a SUBSET of the live one. Indicative quotes throughout.
- The 60-minute minimum hold cannot bind at daily resolution; the VRP-evaporation exit is not replicated (no historical vol surface).
- Overlap: one entry per name per 5 days was kept (448 of 913); SEs are plain, not HAC — treat them as optimistic.
- Selection cuts: picks with |delta| > 0.4 dropped (253) — the archive carries no open interest, so the probe's size-only liquidity test skews picks toward the money against the live −0.30 target; leveraged / inverse funds dropped as the live bot does (153 rows, 38 names, by asset name where yfinance had one, else the seed list).
- Four variants were tried; discount the best by four (REASONING STANDARD #4). Nothing here trades or changes a rule.
