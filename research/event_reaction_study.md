# Event-reaction study — insider buys, material agreements, restatements, executive departures

Purpose: the ticker dossier wants to flag events "the stock hasn't factored in
yet". Before any such flag can be trusted, each event type needs a MEASURED
reaction base: how stocks move after it, and how much of the move arrives LATE
(after day 0-1), i.e. whether a slower market leaves drift on the table —
especially in small caps (EDGE DOCTRINE #2). This file is that base.

Tooling: `scripts/event_reaction_study.py` (reproducible; caches raw SEC and
price downloads in a temp dir, never in the repo), tests in
`test_event_reaction_study.py`.

---

## 1. PRE-REGISTRATION (written before any event list or return was computed)

TIME ORDER, stated honestly. This section was written at 2026-10-29 ~03:20Z.
Before writing it, the session had only (a) read the existing repo code
(`sec_filings_universe.py`, `sec_filings_text.py`, `backtest_v2._yahoo_bars`)
and the repo's prior Form 4 history (open_questions.md Form 4 entry,
edge_preregistration.md P2), and (b) probed endpoint AVAILABILITY: one
submissions JSON (AAPL, to learn its field names and that
`acceptanceDateTime` is true UTC — checked against the filing's
index-headers `ACCEPTANCE-DATETIME` 16:30:28 ET = 20:30:28Z), one insider
data-set zip (2016q1, schema only), the ticker/exchange list (counts by
exchange), and Yahoo bar lengths for 8 tickers (timing only; no return was
computed). No event was built and no abnormal return was computed before this
section existed. Its sha256 at the moment of freezing is recorded in §6. It is
committed in the same commit as the results; nothing in §1 was edited after
results were seen (any later deviation is listed in §5, never patched in here).

### 1.1 Event types (four, theory-motivated; no others will be tested)

| # | Event | Source | Definition |
|---|---|---|---|
| E1 | Insider purchase CLUSTER | SEC Insider Transactions Data Sets (quarterly TSV, 2016q1→latest) | Form 4 (not 4/A), non-derivative TRANS_CODE `P`, acquired `A`, by a reporting owner whose relationship includes Director or Officer (10%-owner-only filers excluded, as in the repo's prior). A cluster fires on the first purchase FILING at which ≥2 DISTINCT such insiders have purchase transaction dates within the trailing 10 calendar days (counting only filings already filed — backward-only). One cluster per issuer per 30 calendar days. |
| E2 | 8-K Item 1.01 | EDGAR submissions JSON `items` | Original 8-K (not 8-K/A) listing item 1.01 (entry into a material definitive agreement). |
| E3 | 8-K Item 4.02 | same | Original 8-K listing item 4.02 (non-reliance on previously issued financials — a restatement; a negative credibility event). |
| E4 | Sudden CEO/CFO departure | same + filing primary document text | Original 8-K listing item 5.02 whose primary document contains a sentence naming the CEO or CFO role (`chief executive officer`, `chief financial officer`, `principal executive officer`, `principal financial officer`, or `CEO`/`CFO` as words) together with a departure word (`resign*`, `terminat*`, `depart*`, `step* down`, `stepped down`, `separat*`, `ceas*`, `removed`) and NOT containing `retire*` in that sentence. Planned retirements and appointment-only / compensation-only 5.02s are thereby excluded. |

LAWSUITS are deliberately NOT measured: EDGAR has no clean lawsuit form
(8-K Item 8.01 "other events" and 10-Q Part II Item 1 "Legal Proceedings" are
dominated by boilerplate and routine disclosures; a regex over them would be a
fishing expedition with an unknown label error rate). Documented as a gap with
a proposed free approach in §4.

### 1.2 Universe, data, sample

- Universe: CIKs in SEC `company_tickers_exchange.json` listed on NYSE /
  Nasdaq / CBOE (OTC excluded: spreads make any drift untradeable and Yahoo
  coverage is poor), one ticker per CIK (first listed), restricted to
  operating companies (≥1 10-K/10-Q in their submissions history since 2015).
- SURVIVORSHIP: this is a SURVIVOR panel — only companies with a CURRENT
  ticker; every delisted, acquired, or bankrupt company is missing. Prices are
  Yahoo daily (adjusted close), the repo's keyless fallback. The 2016+
  delisting-inclusive price panel (wishlist, human decision) is the fix.
- Sample window: events with day 0 between 2016-01-01 and the last day that
  still has a full [0,+20] window in the price data.
- Full sweep intended (all universe CIKs). If price/SEC fetching for the full
  universe is infeasible tonight, a seeded uniform random subsample of CIKs
  (seed 20261029) will be used and its fraction stated in §5 BEFORE results
  are computed. E4 needs one document fetch per 5.02 filing: if more than
  30,000 5.02 filings qualify, a seeded uniform random 30,000 are classified
  (rule fixed now; the fraction is determined by the count, not by results).
- SEC fair access: declared User-Agent `voltradeai-datacore/1.0
  (research@voltradeai.com)` (the repo's existing EDGAR UA), the repo's
  process-wide limiter (8 req/s, hard cap 10).

### 1.3 Event-day alignment (no lookahead)

- Public time = EDGAR `acceptanceDateTime` (UTC) converted to US/Eastern.
- Day 0 = the acceptance date if it is a trading day and acceptance is before
  16:00 ET (pre-open and intraday filings); otherwise the NEXT trading day
  (after-close, weekend, holiday filings).
- E1 uses the acceptance time of the cluster-completing Form 4 when the
  issuer's submissions JSON lists that accession; otherwise (fallback) day 0 =
  the first trading day STRICTLY AFTER the Form 4 FILING_DATE (conservative —
  it can only drop day-0 reaction into the pre-window, never leak).
- Trading calendar = SPY's trading days.
- Same ticker, same event type within 20 trading days of a kept event → only
  the first is kept.
- Inclusion: close ≥ $1 on day −1; ≥ 60 trading days of history before day 0;
  full price coverage over the window measured.

### 1.4 Measurement

- Daily abnormal return AR_t = r_i,t − r_SPY,t (simple returns on adjusted
  closes). CAR[a,b] = Σ_{t=a..b} AR_t, windows (trading days, day 0 = first
  public day): **[0,1]** (reaction), **[2,5]**, **[2,20]** (PRIMARY drift),
  **[2,60]**, plus [0,20] for the late share.
- BASE RATE (REASONING STANDARD #3): every event gets 5 same-ticker CONTROL
  dates drawn uniformly (seeded per event) from that ticker's trading days
  within ±250 trading days of day 0, excluding days within ±20 trading days of
  any same-type event of that ticker and requiring the same full window. The
  PRIMARY statistic is the EXCESS CAR = event CAR − mean(control CARs) — this
  removes the ticker's own drift (incl. the upward drift a survivor panel
  carries) and its beta-to-SPY drift. Raw SPY-adjusted CAR is reported beside it.
- Excess CARs are winsorized at the 1st/99th percentile within event type for
  means and t-stats; medians and hit rates (share > 0) use unwinsorized values.
- Standard errors clustered by calendar week of day 0 (events bunch in time;
  cross-sectional correlation inside a week). t = mean / clustered SE.
- Size buckets: liquidity terciles (market cap is not freely point-in-time):
  event's median daily dollar volume (adj close × volume) over trading days
  [−250, −6], against breakpoints = 33rd/67th percentiles of all universe
  tickers' median daily dollar volume over the PRIOR calendar year (≥ 200
  days). Bucket = small / mid / large.
- Regime: SPY close on day −1 above (bull) or below (bear) its 200-day SMA.
  Reported descriptively only (not a verdict input).
- Late share = mean excess CAR[2,20] / mean excess CAR[0,20]; reported only
  when the [0,20] mean has |t| ≥ 2 and the same sign as [2,20]; else "n/a".
- Cost context: Abdi–Ranaldo (2017) close/high/low effective-spread estimate
  over [−60,−1] per event; median per cell = round-trip cost proxy.

### 1.5 Verdict bar (fixed now)

Primary family = 4 types × 3 size buckets = 12 cells on excess CAR[2,20].
Two-sided Bonferroni at α = 0.05 → p < 0.00417 → **|t| ≥ 2.87**. E1 belongs to
the repo's Form 4 root, which already spent 8 tested cells (open_questions.md
2026-07-19/07-22; edge_preregistration.md P2), so E1 cells use 0.05/(12+8) →
**|t| ≥ 3.02**.

A cell is a **CANDIDATE SIGNAL (gate 2)** only if ALL hold:
1. |mean excess CAR[2,20]| ≥ 1.0% and |t| ≥ the bar above;
2. same sign in both halves (2016–2020 and 2021–latest), each half |t| ≥ 1.0;
3. |mean| exceeds the cell's median round-trip spread proxy;
4. SURVIVORSHIP DIRECTION RULE: a survivor panel flatters post-event returns
   (the firms that died after the event are missing). So a NEGATIVE drift
   finding is conservative and may pass; a POSITIVE drift finding may NOT pass
   on this panel — it is reported as "positive on survivors — unconfirmed,
   needs the delisting-inclusive panel" (the same rule P2 already declared for
   insider purchases).

Per type, verdict text:
- "CANDIDATE SIGNAL (gate 2)" — ≥1 cell passes;
- "POSITIVE ON SURVIVORS — UNCONFIRMED" — a cell meets 1–3 with a positive sign;
- "PRICED IN FAST / NO DRIFT" — pooled [0,1] excess |t| ≥ 2.87 but no cell
  passes on [2,20];
- "NO MEASURABLE REACTION" — neither.
[2,5] and [2,60] and the regime split are secondary/descriptive and can never
pass a cell on their own.

### 1.6 Priors (stated before running)

| Event | [0,1] excess, pooled | [2,20] excess, small bucket | P(≥1 cell CANDIDATE / meets 1–3) |
|---|---|---|---|
| E1 insider cluster | +1.0% (small +1.5%, large +0.2%) | +1.0% | 25% meets 1–3 positive (cannot pass on survivors); this repo's 2025–26 test found NO 20d separation and NEGATIVE 60d vs own-ticker baseline, so I hold this low |
| E2 Item 1.01 | +0.3% (small +0.8%) | ~0 (±0.5%) | 10% — most 1.01s are routine credit agreements/leases; the item is noisy |
| E3 Item 4.02 | −3% (small −5%) | −1.5% | 25% — literature (Palmrose et al. 2004) shows large negative day-0 and continued drift, but survivors bias toward zero and n is small (power-limited) |
| E4 sudden CEO/CFO exit | −1.0% | −0.5% | 15% |

Late share prior: E1 ~50%, E3 ~30%, E4 ~30%, E2 undefined (no reaction).
Expected counts (survivor universe, 2016–2026): E1 ~5–8k, E2 ~15–30k,
E3 ~400–900, E4 ~3–6k.

### 1.7 What the dossier may display, decided in advance

- Event FACTS (filed X at time T, source EDGAR, credibility = primary
  regulatory filing) are RAW and may always be shown with source attribution.
- A "not yet priced in" / drift flag may be shown ONLY for a cell that is a
  CANDIDATE SIGNAL (gate 2), labeled with its measured mean, t, n, sample
  window, and "gate 2 only — not traded". All other cells may display at most
  the historical average reaction as context ("historically, similar filings
  moved X% on day 0–1"), never a forecast.

---

## 2. DATA AND SAMPLE (run 2026-10-29, 03:22Z–04:43Z)

| Item | Value |
|---|---|
| Universe | 6,055 NYSE/Nasdaq/CBOE CIKs (SEC `company_tickers_exchange.json`); 4,511 operating (≥1 10-K/10-Q since 2015). All 4,511 priced on Yahoo (0 errors). SPY 2014-06-02 → 2026-10-28 |
| SEC fetches | 6,055 submissions JSONs + extra pages (0 errors), 44 insider data-set zips (2015q4–2026q3; 2026q4 not yet published), 30,000 Item 5.02 primary documents. Declared UA, ≤8 req/s |
| Day-0 range | 2016-01-04 → 2026-09-30 (the last day with a full [0,20] window) |
| E1 insider clusters | 118,894 officer/director P rows → 11,879 clusters → 11,479 at operating issuers → **10,090 measured**. Acceptance time found for 11,479 of the 11,879 completing Form 4s |
| E2 Item 1.01 | 59,846 → **42,956 measured** (after 20-day dedup, $1 floor, history and window rules) |
| E3 Item 4.02 | 823 → **715 measured** |
| E4 sudden CEO/CFO exit | 76,970 Item 5.02 rows → random 30,000 classified (39%) → 3,460 positive rows (3,475 events incl. co-registrants) → **3,155 measured**. Label precision ≈ 68% (D5) |
| Size breakpoints | median daily $ volume, prior calendar year, all universe tickers: e.g. 2016 small ≤ $0.74M, large > $14.6M; 2025 small ≤ $1.46M, large > $28.5M |
| Clusters (weeks) | 561 calendar-week clusters for E1/E2; 320 for E3 |

Survivor panel: every delisted, acquired or bankrupt issuer is missing. The
direction of the bias is in §3.4.

## 3. RESULTS

### 3.1 Headline — excess CAR (event minus same-ticker random-date control), SPY-adjusted, winsorized 1/99, SEs clustered by week

| Event | Bucket | n | CAR[0,1] | CAR[2,5] | **CAR[2,20]** | CAR[2,60] | median[2,20] | hit[2,20] | late share | spread proxy | MDE[2,20] | cell |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 insider cluster | pooled | 10,090 | **+2.16% (t +21.8)** | +0.20% (t +1.4) | +0.79% (t +2.65) | +2.72% (t +5.2) | +0.30% | 51% | 26% | 0.72% | 0.90% | — |
| | small | 4,738 | +2.29% (t +18.6) | +0.35% (t +2.4) | +0.74% (t +2.22) | +2.00% (t +3.5) | −0.15% | 50% | 24% | 1.15% | 1.01% | fail |
| | mid | 3,429 | +2.40% (t +17.5) | +0.13% (t +0.7) | +0.82% (t +1.91) | +3.34% (t +4.5) | +0.36% | 52% | 25% | 0.44% | 1.29% | fail |
| | large | 1,916 | +1.44% (t +9.2) | −0.04% (t −0.2) | +0.83% (t +1.88) | +3.44% (t +3.9) | +0.99% | 54% | 37% | 0.00% | 1.33% | fail |
| E2 Item 1.01 | pooled | 42,956 | +0.14% (t +2.3) | −0.25% (t −3.6) | −0.72% (t −3.75) | −2.06% (t −5.6) | −0.59% | 48% | n/a | 0.52% | 0.56% | — |
| | small | 16,272 | +0.18% (t +1.7) | −0.40% (t −3.2) | **−1.23% (t −3.85)** | −3.12% (t −5.0) | −1.30% | 47% | n/a | 1.45% | 0.92% | fail (halves, cost) |
| | mid | 13,236 | +0.25% (t +2.9) | −0.18% (t −2.0) | −0.48% (t −2.13) | −1.65% (t −3.9) | −0.45% | 48% | n/a | 0.38% | 0.65% | fail |
| | large | 13,407 | −0.01% (t −0.2) | −0.14% (t −2.0) | −0.36% (t −2.12) | −1.20% (t −4.0) | −0.22% | 49% | n/a | 0.19% | 0.49% | fail |
| E3 Item 4.02 | pooled | 715 | **−1.51% (t −3.4)** | −0.12% (t −0.3) | −0.19% (t −0.17) | +0.43% (t +0.2) | +0.11% | 50% | n/a | 0.64% | 3.19% | — |
| | small | 411 | −1.26% (t −2.2) | −0.12% (t −0.2) | −0.63% (t −0.42) | −0.50% (t −0.2) | +0.00% | 50% | n/a | 1.16% | 4.24% | fail |
| | mid | 217 | −2.44% (t −2.8) | +0.31% (t +0.4) | +1.69% (t +1.32) | +1.97% (t +0.8) | +1.11% | 53% | n/a | 0.28% | 3.68% | fail |
| | large | 86 | −0.37% (t −0.4) | −1.25% (t −1.2) | −2.80% (t −0.85) | +0.69% (t +0.2) | −0.80% | 47% | n/a | 0.35% | 9.44% | fail |
| E4 CEO/CFO exit | pooled | 3,155 | **−1.16% (t −7.2)** | −0.12% (t −0.7) | −1.00% (t −2.45) | −0.19% (t −0.3) | −0.98% | 47% | 53% | 0.55% | 1.17% | — |
| | small | 1,087 | −0.82% (t −2.6) | +0.05% (t +0.2) | −1.96% (t −2.28) | −1.76% (t −1.0) | −2.11% | 44% | 96% | 1.59% | 2.47% | fail |
| | mid | 1,026 | −1.38% (t −5.3) | −0.08% (t −0.3) | −0.86% (t −1.42) | −0.11% (t −0.1) | −1.07% | 47% | 39% | 0.43% | 1.74% | fail |
| | large | 1,041 | −1.29% (t −5.9) | −0.35% (t −1.8) | −0.13% (t −0.29) | +1.38% (t +1.8) | −0.19% | 49% | 10% | 0.14% | 1.32% | fail |

How to read the columns:
- MDE = the bar's t × the cell's SE: the smallest drift this cell could have
  passed.
- "late share" = mean[2,20] / mean[0,20], shown only when [0,20] has |t| ≥ 2
  with the same sign as [2,20].
- spread proxy = median Abdi–Ranaldo round-trip estimate (a floor-at-zero
  estimator, so liquid names read ~0).
- Bars: |t| ≥ 3.02 for E1, ≥ 2.87 for the rest.

### 3.2 Verdicts against the pre-registered bar (§1.5)

| Event | Verdict | Why |
|---|---|---|
| **E1 insider cluster** | **PRICED IN FAST / NO DRIFT** | Large, fast reaction: +2.2% on [0,1] (t 21.8), mostly on day 0. Only ~26% of the [0,20] move arrives late. [2,20] is +0.7% to +0.8% in every bucket, below both the 1% move bar and the t ≥ 3.02 bar. The [2,60] drift (+2.7%, t 5.2) is secondary and positive, and §1.5 rule 4 says a positive result on a survivor panel can never pass. |
| **E2 Item 1.01** | **NO MEASURABLE REACTION** (mechanical: pooled [0,1] t 2.3 < 2.87) | Day 0 is flat. Afterwards there is NEGATIVE drift in every bucket. The small-cap cell (−1.23%, t −3.85) clears the move and t bars but fails two others: the halves rule (2016–20 is −0.00%, so all of it is 2021+) and the cost rule (1.23% < the 1.45% spread proxy). Not a candidate. See §3.5 for why "no measurable reaction" undersells this cell. |
| **E3 Item 4.02** | **PRICED IN FAST / NO DRIFT** | [0,1] −1.5% (t −3.4); the drift is ~0, with a slight pre-drift on days −3..−1. Underpowered: the cell MDE is 3–4% (n = 715). "No drift" here means no drift larger than ~3–4%. It is not evidence of zero drift. |
| **E4 sudden CEO/CFO exit** | **PRICED IN FAST / NO DRIFT** | [0,1] −1.2% (t −7.2). Pooled [2,20] −1.0% (t −2.45; late share 53%) and small −1.96% (t −2.28) are suggestive but below the 2.87 bar. Both halves are negative (pooled 2016–20 −1.48%, t −2.4; 2021+ −0.75%, t −1.4), and the effect is concentrated in bull regimes. The labels are ~68% precise (D5), so the true effect is probably larger in magnitude than measured. This is the one type worth a held-out replication (§8). |

**No cell is a CANDIDATE SIGNAL (gate 2).** Priors vs outcome:
- E1 met the "fast reaction" prior. The drift share (26%) came in below
  the 50% prior.
- E2 met its "~0 reaction" prior, but drift came out NEGATIVE, not ~0. Not
  anticipated.
- E3's [0,1] (−1.5%) is half the −3% prior. Survivors are the likely reason:
  the worst restaters delisted.
- E4's [0,1] (−1.2%) matched the −1.0% prior. The drift (−1.0%) is larger
  than the −0.5% prior.

### 3.3 Halves and regime (excess CAR[2,20]; descriptive only)

| Event | Group | 2016–2020 | 2021+ | bull [0,1] | bull [2,20] | bear [0,1] | bear [2,20] |
|---|---|---|---|---|---|---|---|
| E1 | pooled | +1.08% (t +2.2) | +0.57% (t +1.5) | +2.18% (t +23.3) | +0.85% (t +2.9) | +2.13% (t +7.8) | +0.61% (t +0.9) |
| E1 | small | +1.25% (t +2.7) | +0.38% (t +0.8) | +2.26% | +0.72% (t +2.0) | +2.38% | +0.83% (t +1.1) |
| E2 | pooled | +0.13% (t +0.4) | −1.27% (t −5.0) | +0.12% | −1.12% (t −5.5) | +0.25% | **+1.32% (t +2.8)** |
| E2 | small | −0.00% (t −0.0) | −1.90% (t −4.6) | +0.20% | −1.79% (t −5.2) | +0.13% | +1.49% (t +2.1) |
| E3 | pooled | −1.29% (t −0.6) | +0.11% (t +0.1) | −1.28% (t −2.7) | −0.33% (t −0.3) | −2.67% (t −2.3) | +0.54% (t +0.3) |
| E4 | pooled | −1.48% (t −2.4) | −0.75% (t −1.4) | −1.22% (t −6.8) | −1.34% (t −3.0) | −0.90% (t −2.7) | +0.50% (t +0.6) |
| E4 | small | −2.20% (t −1.7) | −1.84% (t −1.7) | −1.02% (t −2.9) | −2.29% (t −2.4) | +0.05% (t +0.1) | −0.54% (t −0.3) |

Full per-bucket halves/regime and raw (control-free) tables are printed by
`python3 scripts/event_reaction_study.py report --cache <dir>`.

### 3.4 Checks that the numbers measure what they claim

- **Timing (no lookahead), daily abnormal return profile.** Winsorized
  mean, days −3..+3:
  - E1: −0.51, −0.04, +0.16, **+1.86**, +0.24, +0.03, +0.01 %
  - E3: −0.48, −0.26, −0.25, **−1.13**, −0.33, −0.15, −0.05 %
  - E4: −0.11, +0.02, −0.15, **−0.88**, −0.30, −0.11, −0.17 %

  The reaction lands on day 0, as the after-close rule intends. Day −1 is
  small. E1's day −3 dip is consistent with insiders buying after declines
  (trades precede the filing by ~2 days). E3 shows mild pre-drift, which
  fits restatement news leaking or being foreshadowed.
- **Base rate.** The same-ticker control's [0,1] mean is ≈ 0 for E1, E2
  and E4. For E3 small it is +1.3%, driven by outliers: the median is
  −0.26%, and winsorizing the excess handles it.
- **Raw vs winsorized.** E2 small's raw [0,1] mean of +1.42% collapses to
  +0.37% when winsorized (median −0.41%): a few extreme small-cap
  reactions. The raw (unwinsorized) [2,60] means are dominated by outliers
  (E2 pooled +6.4%, t 1.0) and should not be read.
- **Survivorship direction.** Missing delisted firms bias every post-event
  mean UPWARD. That makes E1's positive drift an upper bound and the
  negative E2/E4 drifts conservative. The same-ticker control removes the
  generic survivor drift but not the event-conditional part (a restater
  that survived is a selected restater).
- **Repo history (multiple testing).**
  - This is the Form 4 root's 9th–11th tested cells (E1 bar 3.02 accounts
    for the 8 earlier ones).
  - The earlier 2025–26 screen (open_questions.md) found buys UNDER their
    own-ticker baseline at 60d. This study finds +2.7% at [2,60] over
    2016–2026. The designs differ: a 10-year sample vs 2 years, a ±250-day
    local control vs the ticker's whole forward history, and SPY-adjusted
    vs raw returns.
  - Neither is a pass. P2 (opportunistic/routine split on a
    delisting-inclusive panel) remains the right test for insider buys.

### 3.5 Post-hoc observations (NOT findings — filed as hypotheses, each a new variant)

- **E2 negative drift is a 2021+ and bull-regime phenomenon in small caps.**
  2016–20 is 0.00%, 2021+ is −1.90% (t −4.6), and the bear regime flips
  sign. One candidate mechanism: 1.01 filings by small caps in 2021+ are
  heavily financing agreements (securities purchase agreements, ATM
  programs, convertible notes, often co-filed with Items 3.02 or 2.03). They
  are followed by dilutive selling. That would be a structural reason the
  edge exists, and also a reason it may be unharvestable (short borrow on
  those names is scarce or costly). It was found after looking at 12+
  cells, so it carries that discount. It needs its own pre-registration:
  restrict to 1.01 ∧ (3.02 ∨ 2.03), with the 2016–2020 half as a holdout
  where the effect should be absent if it is an era effect.
- **E4's drift is concentrated in bull regimes** (bull [2,20] −1.34%,
  t −3.0; bear +0.50%). It is one of ~24 regime cells looked at, so it is
  not believed.

## 4. LAWSUITS — documented gap, not measured

Why not measured here: EDGAR has no lawsuit form. The candidate sources are
all noisy. 8-K Item 8.01 ("Other Events") covers anything a company chooses
to disclose voluntarily. 10-Q/10-K "Legal Proceedings" is mostly the same
boilerplate repeated every quarter, and its date is the report date, not the
day the suit was filed. 8-K Item 1.03 covers bankruptcy only. A regex over
these would be a fifth, un-pre-registered event type with an unknown
label-error rate. That is the fishing this study set out to avoid.

Proposed free approach (BUILD-FIRST: we can get the raw material free; what
is missing is the processing):

1. **Source: CourtListener / RECAP (Free Law Project).** Its v4 REST API has
   dockets, docket entries and search over federal PACER records that RECAP
   users have contributed. Terms as checked live this session
   (courtlistener.com/help/api/rest/): access needs a free account token
   (unauthenticated search returns 401 "Authentication credentials were not
   provided"). The default limits are **5 requests/minute, 50/hour, 125/day**,
   with "Free Law Project membership" for more and "commercial agreements"
   available. So this is fine for research, but a commercial /data surface
   would need either their commercial agreement or our own derived product.
   That must go through the MONETIZATION TRIPWIRE / licensing review before
   anything is sold. Free Law Project also publishes bulk data files; their
   licence and their coverage of dockets were not verified this session.
2. **Event definition to pre-register next time:** federal civil dockets
   with nature-of-suit **850 (Securities/Commodities)** or **Section 10(b)
   class actions**, where an exchange-listed company is a NAMED DEFENDANT.
   Event time = docket `date_filed`. The public time is the filing date, so
   use day 0 = the next trading day (court-time stamps are unreliable).
3. **Entity join:** match defendant names to CIKs through the repo's entity
   spine (`scripts/build_entity_spine.py`). The join's precision must be
   checked by hand on 50 matches BEFORE any return is computed (gate 1 DATA).
4. **Coverage caveat that must be stated in any result:** RECAP holds only
   the dockets that some user bought from PACER. High-profile securities
   suits are well covered; obscure ones are not, so coverage is SELECTED
   toward salient cases. The only complete source is PACER itself, which is
   paid ($0.10/page). That belongs on the wishlist only if the RECAP version
   shows a signal worth completing.
5. **A known confound to design around:** securities class actions are
   usually filed AFTER a price drop, because the drop is the damages
   predicate. A naive [0,1] reaction will look small. The pre-event
   [−20,−1] window must be reported, and the claim tested is "further drift
   after the suit is filed", not "the suit causes a drop".

Until that study runs, the dossier may show a lawsuit only as a RAW fact with
its source (court, docket number, filing date, CourtListener link). It may
never carry a "not priced in" flag.

---
## 5. DEVIATIONS AND IMPLEMENTATION DECISIONS (logged as they happened)

Entries D1–D5 were written BEFORE any abnormal return was viewed.

- **D1 (03:19Z, before any data was built) — E1 "trailing 10 calendar
  days" pinned down.** §1.1 does not say trailing from WHAT. The first draft
  of the code used the completing filing's latest TRANSACTION date; a unit
  test (a late filer whose stale trade predates the window) exposed the
  ambiguity. Implemented as the literal reading: both insiders' transaction
  dates fall in the 10 calendar days up to and including the completing
  filing's FILING date. A late-filed stale trade therefore does not complete
  a cluster. No result existed at the time.
- **D2 (03:21Z) — run split for the shared lock.** Network stages
  (`fetch`) ran outside `heavy.sh` at `nice 10` (low CPU / memory, ~1h of
  waiting on SEC/Yahoo would have blocked other agents' test gates); the
  CPU-bound measurement (`run`, everything cached) ran under `heavy.sh`.
  No methodological effect.
- **D3 (03:39Z) — E4 subsample, per the §1.2 rule.** 76,970 original 8-Ks
  with Item 5.02 (Dec 2015 → Oct 2026) from the 4,511 operating universe
  CIKs exceeded the 30,000 cap, so a seeded (20261029) uniform random
  30,000 (39.0%) were text-classified. E1–E3 use the full universe (no
  subsample).
- **D4 (03:39Z) — counts at fetch time (no returns yet).** Universe 6,055
  exchange-listed CIKs; submissions fetched for all 6,055 (0 errors); 4,511
  operating (≥1 10-K/10-Q since 2015). Insider data sets 2015q4–2026q3
  (2026q4 not yet published): 118,894 officer/director purchase rows → 11,879
  clusters (before operating filter / dedup); the cluster-completing Form 4's
  acceptance timestamp was found in the issuer's submissions JSON for 11,479
  (the rest use the conservative next-trading-day fallback). Raw 8-K item
  counts in operating CIKs: 1.01 = 59,846; 4.02 = 823; 5.02 = 76,970.
  Day-0 sample start 2016-01-01 is applied at measurement.
- **D5 (04:43Z — written while the measurement process was running, before any result was viewed) — E4 label spot-check (gate-1 style;
  rule NOT changed).** Of 29,963 classified 5.02 documents, 3,451 were
  labeled sudden CEO/CFO departures. (The 30,000 sampled rows hold 29,963
  unique accessions: 37 rows are co-registrant 8-Ks listed under two CIKs.
  This was first misread here as "37 unfetchable"; corrected after the run,
  and there were 0 fetch errors.) I read 25 random positives
  and 25 random negatives (seed 7). Negatives: 25/25 correct (board changes,
  comp plans, a CFO *retirement* correctly excluded). Positives: **17/25 true
  (precision ≈ 68%, 95% CI ≈ 48–83%)**. The 8 false positives:
  `depart\w*` matching "departments" (1); a departure that was "previously
  disclosed" in an earlier 8-K (2); the CEO of a division or subsidiary, not
  the issuer (2); an interim CFO handing over to a permanent hire (1); a
  director resigning to become CFO of a subsidiary (1); a non-CEO officer
  whose sentence only mentions the CEO (1). The pre-registered rule is kept
  as frozen; changing it after seeing labels would be a second variant. The
  consequence: if false positives carry no reaction, E4 effects are
  attenuated by about ×0.68 toward zero. That is a power loss, not a bias in
  sign. A tightened E4 v2 (word-boundary "depart(ed|ure|s)", exclude
  "previously disclosed" and division/subsidiary roles) would be its own
  pre-registered variant.

Post-result entries (written after the results were seen):

- **D6 — verdict text vs. the E2 cell.** The mechanical type verdict for E2
  is "NO MEASURABLE REACTION", because the §1.5 type labels key on [0,1]
  only. That label does not describe a cell with significant negative
  drift that failed only on halves and cost. §3.2 reports both rather than
  re-labelling: the pre-registered label stands, and the gap in the label
  scheme is noted here for any future pre-registration (add a "drift fails
  robustness" label).
- **D7 — no sector-ETF adjustment.** The task brief allowed a sector-ETF
  benchmark "if mapping exists". No SIC→sector-ETF map exists in the repo,
  and §1.4 did not pre-register one, so only SPY plus the same-ticker
  control were used. The control absorbs the ticker's own beta and drift
  over ±250 days, which covers most of what a sector ETF would.
- **D8 — sample is the full universe.** No CIK subsample was needed. E4 is
  the only subsampled type (D3).

## 6. PROVENANCE

- §1 was frozen at 2026-10-29T03:16:26Z (sha256
  `fd7cda3958ffd9b53d41d804fe7831c8cde05770d9987dca357e7ef9d1fd467c`).
  The first 10865 bytes of this file still hash to exactly that value:
  `head -c 10865 research/event_reaction_study.md | sha256sum`.
- Timeline:
  - §1 frozen 03:16Z (its header says "~03:20Z").
  - SEC/Yahoo fetch 03:22–04:42Z.
  - Measurement 04:42:53–04:43:30Z.
  - Results first viewed ~04:44Z.
  - §§2–4, D6–D8 and §§6–8 written afterwards.
- Reproduce:
  - `python3 scripts/event_reaction_study.py fetch --cache DIR` (network,
    ~80 min, mostly the 30k 5.02 documents)
  - then `run --cache DIR` (~40 s)
  - then `report --cache DIR`.

  Raw SEC and price data stay in DIR (default $TMPDIR); none is committed.
- Determinism: the controls use a per-event seeded RNG and the E4
  subsample a fixed seed (20261029). Yahoo adjusted closes can be revised
  retroactively (dividend adjustments), so a re-run months later may move
  the second decimal.

## 7. WHAT THE TICKER DOSSIER MAY AND MAY NOT DISPLAY (applying §1.7)

- **May display (RAW, always):** the event fact. That is the filing type
  and item, the acceptance time in ET, a link to the EDGAR filing, and
  source credibility = "primary regulatory filing (SEC EDGAR)". For Form 4:
  the insiders, roles, shares and prices as filed. For lawsuits (§4): only a
  court docket fact with its source, once that pipeline exists.
- **May display (CONTEXT, labeled historical, never a forecast):** the
  measured day-0–1 reaction of the type, with n, window and the survivor
  caveat. For example: "Insider purchase clusters: survivors' shares
  historically rose about 2.2% vs SPY over the filing day and next day
  (n = 10,090, 2016–2026; most of the move came on day 0)". Equivalents:
  CEO/CFO exits −1.2% (n = 3,155), restatements −1.5% (n = 715), Item 1.01
  +0.1% (no reliable day-0 move).
- **May NOT display:** any "not yet priced in", "drift expected" or
  "the market hasn't reacted" flag for any of the four types. No cell
  passed gate 2. For E1 and E3/E4 the honest reading is that the market
  reacts on day 0. A dossier that claimed slower absorption would be
  claiming something this study measured and did not find.
- Credibility weighting (source-level, not predictive): Form 4 and 8-K are
  sworn primary filings. Form 4 transactions are facts; an 8-K's item code
  is the filer's own classification. E4 is an INFERRED label from text, with
  measured precision ~68%, and must be shown as "detected from filing text"
  with the filing link, never as a fact.

## 8. NEXT STEPS (each its own pre-registration; none run here)

1. **E4 held-out replication.** The 46,970 Item 5.02 rows NOT in the D3
   sample (61%) have never been looked at. That is a clean holdout. The
   bar should be fixed BEFORE fetching: same frozen rule, pooled excess
   CAR[2,20] < 0 with t ≤ −2.0. Optionally run a tightened E4 v2 classifier
   (D5) as a separately counted variant on the same holdout. About 100
   minutes of SEC fetching.
2. **E2 financing-agreement drift (§3.5).** Pre-register 1.01 ∧ (3.02 ∨
   2.03) in small caps, with 2016–2020 as the era-holdout. Add a
   borrow-feasibility note before any gate 3: shorting these names is the
   trade, and borrow is the cost that decides it.
3. **Delisting-inclusive price panel** (wishlist, human decision, shared
   with Lazy Prices and P2). This is the only way a POSITIVE drift (E1's
   [2,60]) could ever pass. It also tightens E3, whose worst cases are
   exactly the delisted ones.
4. **Lawsuits** per §4: CourtListener token + NOS-850 dockets + an entity
   join gate 1 first.
