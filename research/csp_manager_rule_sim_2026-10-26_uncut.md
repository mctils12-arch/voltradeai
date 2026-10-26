# Research test 2 (gate 3): the manager's rules on the CSPs the bot would have sold — 2026-10-26

Generated 2026-10-26T21:30:27Z by scripts/csp_manager_rule_sim.py — entry rows 8635 (913 with a selectable contract) over 80 archived days and 1821 underlyings; deduped to 769 entries (5-day window); bars 388 names; source /tmp/claude-0/-home-user-voltradeai/1c64125e-f4dc-5e5f-9984-1cb6a5ea06e4/scratchpad/csp_sim/entries_dump.json.

## PRIOR (stated before the run)

Stated before the run: 60% that the live rules show NEGATIVE expectancy per contract net of the archived spread on the deduped single-name set (the sleeve lost when it held); 70% that `no_stop` beats `live` (the -2x stop crystallizes the left tail); 55% that `no_dte_exit` beats `live` by less than the spread it saves; 65% that `earnings_gate` removes most of the worst-10 entries.

## VERDICT

live rules: expectancy -0.85% of collateral per entry (mean $-45 ± 20 SE per contract, win rate 59%, n 635); no_stop: +0.29 pts vs live (n 624; paired $+1 ± 2 per contract, 35 of 624 entries changed); no_dte_exit: +0.80 pts vs live (n 632; paired $+67 ± 17 per contract, 310 of 632 entries changed); earnings_gate: -0.20 pts vs live (n 399; dropped entries averaged $-43 under live rules (n 236), $-16 without APH). Four variants were tried: discount the best by four. Model marks, not fills.

## Variants (per contract; sold at the archived bid, bought back at the MODEL mark + half the entry spread)

| set | n | closed | censored | mean $ | SE | median $ | p5 | p95 | win | exp % coll | spread $ | hold d |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| live | 760 | 635 | 125 | -45 | 20 | +42 | -818 | +464 | 59% | -0.85 | 35 | 14.1 |
| no_dte_exit | 760 | 632 | 128 | +23 | 24 | +105 | -1,039 | +703 | 72% | -0.04 | 32 | 20.7 |
| no_stop | 760 | 624 | 136 | -27 | 19 | +44 | -784 | +467 | 60% | -0.55 | 35 | 14.6 |
| earnings_gate | 450 | 399 | 51 | -47 | 21 | +28 | -792 | +407 | 56% | -1.05 | 34 | 14.4 |

## Exit reasons

| variant | profit_target | loss_limit | dte_exit | dte_critical | expiry | censored | skipped (earnings) |
|---|---|---|---|---|---|---|---|
| live | 273 | 49 | 313 | 0 | 0 | 125 | 0 |
| no_dte_exit | 442 | 87 | 0 | 0 | 103 | 128 | 0 |
| no_stop | 273 | 0 | 351 | 0 | 0 | 136 | 0 |
| earnings_gate | 165 | 32 | 202 | 0 | 0 | 51 | 314 |

## Paired vs live (same entry, both closed; $ per contract = variant minus live; unpaired means above mix the rule with the censoring it changes)

| variant | set | n | mean diff $ | SE | t | entries changed |
|---|---|---|---|---|---|---|
| no_stop | all | 624 | +1.1 | 2.23 | 0.48 | 35 |
| no_stop | BEAR | 47 | +5.5 | 3.85 | 1.43 | 2 |
| no_stop | BULL | 418 | +1.2 | 2.86 | 0.43 | 25 |
| no_stop | CAUTION | 1 | +0.0 | — | — | 0 |
| no_stop | NEUTRAL | 158 | -0.7 | 4.38 | -0.15 | 8 |
| no_dte_exit | all | 632 | +67.4 | 17.15 | 3.93 | 310 |
| no_dte_exit | BEAR | 47 | +307.3 | 82.74 | 3.71 | 27 |
| no_dte_exit | BULL | 424 | +34.6 | 20.42 | 1.7 | 225 |
| no_dte_exit | CAUTION | 1 | +148.8 | — | — | 1 |
| no_dte_exit | NEUTRAL | 160 | +83.4 | 31.01 | 2.69 | 57 |

earnings_gate is a subset, not a rule change — under LIVE rules the entries it keeps averaged $-47 (n 399) and the entries it drops $-43 (n 236); without the single worst dropped entry (APH 2026-07-16, $-6,353) the dropped set averaged $-16.

## Live rules by entry-day regime

| set | n | closed | censored | mean $ | SE | median $ | p5 | p95 | win | exp % coll | spread $ | hold d |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BEAR | 47 | 47 | 0 | -230 | 84 | -39 | -1,319 | +399 | 45% | -4.27 | 45 | 20.4 |
| BULL | 462 | 426 | 36 | -75 | 26 | +24 | -872 | +401 | 55% | -1.14 | 33 | 14.2 |
| CAUTION | 6 | 1 | 5 | +259 | — | +259 | +259 | +259 | 100% | +3.81 | 69 | 9.0 |
| NEUTRAL | 245 | 161 | 84 | +86 | 30 | +92 | -550 | +533 | 71% | +0.90 | 38 | 11.8 |

## Live rules by entry DTE

| set | n | closed | censored | mean $ | SE | median $ | p5 | p95 | win | exp % coll | spread $ | hold d |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30-35 | 218 | 170 | 48 | -35 | 30 | +27 | -823 | +411 | 59% | -0.60 | 30 | 9.1 |
| 36-42 | 269 | 219 | 50 | -76 | 42 | +34 | -817 | +404 | 56% | -1.45 | 36 | 13.2 |
| 43-50 | 273 | 246 | 27 | -25 | 30 | +67 | -814 | +519 | 60% | -0.48 | 39 | 18.2 |

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
| BEX | 2026-10-07 | 2026-11-20 | 40.0 | 752 | 2026-10-23 | loss_limit | -1,634 | 16.48 |
| AMDL | 2026-10-07 | 2026-11-06 | 79.0 | 603 | 2026-10-16 | loss_limit | -1,633 | 57.90 |
| RKLX | 2026-07-06 | 2026-08-21 | 44.0 | 1227 | 2026-07-31 | dte_exit | -1,598 | 16.37 |

## Live sleeve vs simulation, by underlying (names in both; live = realized from fills, sim = live-rules sum)

| u | live realized $ | live fills | sim n | sim pnl sum $ |
|---|---|---|---|---|
| BAC | -226 | 8 | 1 | +29 |
| BB | -484 | 14 | 1 | -90 |
| BMNR | -111 | 5 | 4 | +342 |
| BMY | -49 | 4 | 1 | +58 |
| CIFR | -263 | 11 | 2 | +62 |
| DAL | -56 | 2 | 4 | +261 |
| DRAM | -110 | 29 | 3 | +390 |
| EWZ | -366 | 10 | 1 | -1 |
| FIG | -117 | 2 | 2 | +235 |
| HOOD | -66 | 12 | 1 | +216 |
| HPQ | -48 | 3 | 1 | +42 |
| MUU | -1,700 | 8 | 2 | -223 |
| NEE | -161 | 7 | 2 | -140 |
| NKE | -234 | 14 | 3 | -125 |
| NVDA | -1,126 | 133 | 2 | +707 |
| ONON | -52 | 3 | 1 | -529 |
| ORCL | -315 | 8 | 2 | +1,239 |
| PATH | -972 | 42 | 4 | -239 |
| SLB | -88 | 5 | 3 | +253 |
| SMCI | -46 | 4 | 1 | +91 |
| STM | -95 | 6 | 2 | -139 |
| T | -199 | 25 | 5 | +3 |
| TEM | -422 | 10 | 1 | +119 |
| VALE | -60 | 8 | 1 | +24 |
| VZ | -63 | 10 | 3 | +42 |
| WMB | -425 | 5 | 2 | -253 |

## Honesty

- Every exit is a Black–Scholes MODEL MARK at the entry's implied vol (constant), not a fill: it smooths the intraday path the −2× stop fires on, so the stop's cost reads as a LOWER bound, and it ignores the vol change a crash brings (an under-mark on the way down).
- Costs: sold at the archived bid; bought back at the model mark plus half the entry spread (modelled). r = 4.00%, a stated constant. Calendar DTE as the manager uses it.
- The probe's liquidity test lacks the volume branch the live `_is_liquid` has: this universe is a SUBSET of the live one. Indicative quotes throughout.
- The 60-minute minimum hold cannot bind at daily resolution; the VRP-evaporation exit is not replicated (no historical vol surface).
- Overlap: one entry per name per 5 days was kept (769 of 913); SEs are plain, not HAC — treat them as optimistic.
- Selection cuts: NO delta cut — picks skew toward the money (archive has no open interest) and this set is riskier than the live −0.30 target; leveraged / inverse funds NOT excluded — the pre-2026-10-20 live universe included them, and they dominate the selectable rows.
- Four variants were tried; discount the best by four (REASONING STANDARD #4). Nothing here trades or changes a rule.
