# Options bleed — diagnosis (2026-11-04)

Human (2026-11-04): "figure out the option bleed". Read-only study. Data: `/api/diag/lots` (376 option closing events, 373 lots, 279 inferred structures, −$12,658), audit slices, the newest 200 orders, the live Book, and hourly bars and earnings dates for all 85 underlyings. It extends `research/options_exit_study_2026-11-02.md` rather than redoing it.

The sample reproduces the AI Engine panel exactly: n=376, win rate 18.9%, average win +31.7%, average loss −28.5%, expectancy −17.10% per closed lot.

## Method

Each lot's P&L is split with Black–Scholes at the IV implied by its own entry fill into two parts:

- **Model $**: the underlying's path plus theta.
- **Residual $**: everything else, meaning spread, bad fills and IV change.

There are no intraday option quotes in the archive, so the residual cannot separate spread from IV change.

## Headline

- **74% of the loss is execution and rule friction, not market moves.** Model −$3,264, residual −$9,394.
  - For lots closed before the 10-13 fixes (349 events, −$8,471), the model is −$54: that whole loss was friction.
- **Most of the loss generators were fixed between 08-24 and 11-02.** Two of those fixes, #1582 and #1583, shipped after entries went shadow on 10-28, so they had never run on a live entry until v1.0.1272.
- **Live bugs on 2026-11-04:**
  1. Single-leg option exits decided on pre-market or opening marks.
  2. POS-KILL confirming option breaches off-hours.
  3. The condor loss limit's denominator.
  4. Setup-3 condors' missing earnings check.
  5. The CSP earnings guard failing open. **Live example:** IREN, put sold 2026-11-04 18:41Z, with earnings on 11-05 after the close.
- **There is no edge underneath.**
  - Replaying the current exit rules on the bot's own 117 CSP entries at 30+ DTE gives +$0.9/contract (SE $13.9).
  - Fixing the bleed should bring the sleeve to roughly zero, not past SPY.

## Where the money went

### By strategy

Structure is inferred from legs opened in the same minute.

| strategy | lots | $ | %/lot | model $ | resid $ |
|---|---|---|---|---|---|
| CSP, single short put | 244 | −4,742 | −21.4 | +215 | −4,957 |
| CSP, 2 puts on the same name | 4 | −67 | −15.3 | +32 | −99 |
| iron condor, 30 complete | 122 | −2,954 | −5.7 | +18 | −2,972 |
| GOOGL Nov-27 condor, 3 legs closed (300P wing still open) | 3 | −3,848 | −102.9 | −3,525 | −323 |
| duplicate-close overfills (MUU ×2, NVDL) | 3 | −1,047 | −53.2 | −4 | −1,043 |

### By hold time

| hold | lots | $ | model | resid |
|---|---|---|---|---|
| <1h | 168 | −4,812 | −279 | −4,533 |
| <1d | 143 | −5,011 | −2,209 | −2,802 |
| 1–7d | 57 | −3,784 | −1,829 | −1,955 |
| 8–30d | 7 | +845 | +949 | −104 |
| >30d | 1 | +104 | +105 | −1 |

### By exit mechanism

Attribution is by signature. Points are shares of the panel's −17.1%/lot.

| mechanism | lots | $ | model | resid | pts | status |
|---|---|---|---|---|---|---|
| A. Closed in the first 5 minutes of the session | 53 | −5,083 | −2,141 | −2,942 | −10.4 | single-leg exits were live until this batch |
| I. Structure taken apart leg by leg (GOOGL, PATH, NVDA) | 12 | −4,553 | −3,509 | −1,044 | −1.5 | fixed #1582/#1583 (11-02) |
| B. POS-KILL at −25% on options | 78 | −2,960 | −1,275 | −1,685 | −5.8 | fixed v1.0.1058 (10-13) |
| J. Condor closed 2.7–10 minutes after opening | 109 | −2,159 | +3 | −2,162 | −1.2 | fixed v1.0.1116/1117 (~10-22) |
| H. Duplicate-close overfill | 3 | −1,047 | −4 | −1,043 | −0.4 | guard #35 (08-24) |
| E. Loss limit at 2× credit | 2 | −324 | −162 | −162 | −0.6 | — |
| G. Other | 8 | −312 | −193 | −119 | −0.3 | — |
| C. 21-DTE close about an hour after an entry already inside 21 DTE | 73 | −91 | −52 | −39 | −1.1 | fixed v1.0.1056/1117 |
| K. Condor closed after 1–3 hours | 4 | −90 | −1 | −89 | 0.0 | — |
| F. Gamma rule (likely) | 2 | −34 | −2 | −32 | −0.4 | — |
| D. 50% profit target | 32 | +3,995 | +4,073 | −78 | +4.5 | working |

**Churn lots distort the panel.** Classes C, J and K are 186 of the 376 "lots" but only −$2,340. Without them: n=190, win rate 21.6%, −29.4%/lot, −$10,318. Grouped per structure: n=279, win rate 19.4%, −20.5% per structure, dollar-weighted −12.0% of basis.

### Earnings

9 structures were held through a report: −$4,751 realized.

- GOOGL −3,848
- RKLB −675 and +552
- HPE −290
- CIFR −132
- FIG −117
- IREN −107
- BAC −78
- DAL −56

In the 10-29 → 11-04 shadow log, the guards that should have blocked these were also leaking:

- **Condors:** 14 of 19 proposed condor names were within 0–8 days of earnings.
- **CSPs:** 6 of 31 proposed CSP names (IREN, IONQ, WULF, SMCI, PFE, KDP) were 0–3 days before earnings.

### Opening-print evidence

Of the 53 lots closed in the first 5 minutes, 29 moved ≤3%. They lost −$1,378 while the model says +$266, and the median fill was 1.41× fair value.

**Audit, 2026-10-29 11:31:35Z (07:31 ET):** "XLF261127P00055500: LOSS LIMIT: Cost to close $5.00 is 6.9x the credit … order submitted". That day limit filled at the open at $1.61, against a fair value near $0.49, with XLF *up* 1.2%.

## Base rates and edge

- **What spread alone costs.** For 218 lots with an underlying move under 1%, held under a day and closed outside the open, the residual was a median −5.7% of premium (mean −10.8%). By premium tier:

  | premium | mean residual |
  |---|---|
  | <$0.30 | −18.8% |
  | $0.30–0.75 | −19.4% |
  | $0.75–2 | −8.8% |
  | ≥$2 | −3.9% |

  Applied to every lot, that is −8.5% to −11.2% per lot. The rest of the realized −17.1% is the threshold kills and the opening prints.
- **CSPs under the current rules (replay).** 117 lots, hourly marks at the entry IV, decisions in regular hours only:
  - **Actual result:** −$3,966.
  - **Replay:** −$689, +$0.9 ± 13.9 per contract, 64% wins.
  - **Exits fired:** profit target 62, gamma rule 25, loss limit 18, 21-DTE 11.
  - **By regime:** BULL n=86, −$8.2/contract (SE 13.9). NEUTRAL n=28, +$24/contract (SE 39). Same direction as the 10-26 CSP sim; one cell of about 70, so it stays a hypothesis.
  - **Rule variants:**
    - Dropping the gamma rule: +$3.7/contract (SE 2.3), not significant.
    - Dropping the 21-DTE rule: −$13.6 (SE 9.5).
- **The current machine's own record.** 11 CSPs closed after the 10-13 fixes, net −$144:
  - **Profit target:** 5 lots, +$923.
  - **Loss limit:** 1 lot, −$294.
  - **Opening-print closes:** 5 lots, −$773, including three fills far above fair value (XLF, BAC, CRWV). Without those three fills the post-fix CSPs were about +$215.
- **Verdict:** nothing is shown to be profitable after costs.
- **Discount.** About 70 cells were sliced across 13 dimensions, plus 85 underlyings, 4 replay variants and 6 replay sub-splits. No recommendation comes from picking the best slice.

## Fixes shipped from this study (2026-11-04)

One PR each, merged after the close so they are live before the 11-05 open:

1. **Single-leg option exits wait for regular hours**, like group closes (`options_manager` Phase 2).
2. **POS-KILL judges an option's breach only on regular-hours ticks**, and a breach pending at the close resets (`server/bot.ts`, outside the order POST).
3. **The group loss limit is measured against the net credit (debit)**, as its comment says, and the IC max-loss exit derives max loss from the legs when state lacks it.
4. **Setup-3 condors enforce their documented earnings rule.**
5. **The CSP earnings guard fails closed**, with per-candidate confirmation when the window calendar is missing or truncated.
   - **Live confirmation (20:15Z):** all three calendar windows (21, 30 and 47 days) returned exactly 1,500 rows, starting 11-09 to 11-11 instead of 11-04. Every earnings gate was blind to the next 5–7 days. The structural fix, fetching the calendar in date chunks, follows as its own PR.

## Not done here (filed)

- **POS-KILL order type for options (frozen path):** wishlist.
- **Opening-window exit rule (09:45 ET):** structure-level and dollar-weighted expectancy on the panel, TIME-EXIT applying the stock stale rule to options, gamma-threshold units, and the CSP regime split: open_questions.

## Data gaps

- **Audit log:** starts 2026-10-28, so exit reasons before that date are inferred from signatures.
- **Orders:** only the newest 200 are available, and `orderRow` has no `client_order_id`.
- **Option bars:** `option_bars` returned 429, a rate limit shared with production.
- **Earnings timing:** from yfinance; BMO/AMC is approximate.
- **Replay:** uses model marks, not fills, n=117.
