# Beating the S&P 500 after tax: what the published evidence supports

Date: 2026-11-19. Scope: literature review only. No backtests were run.
Account assumed: about $88,000, taxable, US stocks/ETFs/options, roughly 24% tax on short-term gains and 15% on long-term gains.

How to read this file:

- "Edge" always means extra return per year versus simply holding an S&P 500 ETF, after realistic costs.
- "Before tax" and "after tax" are stated each time. Most academic numbers are before tax and before costs.
- Where a number is my own arithmetic rather than a published figure, it is labelled **(my arithmetic)**.
- Where I could not open the primary source, the claim is listed again in section C, "What I could not verify".
- Live ETF returns are total returns through 2026-11-18 from [totalrealreturns.com](https://totalrealreturns.com/n/SPY,MTUM,SPMO,QUAL,SPHQ,USMV,RSP), a third-party site. I did not compute them.

The one-sentence answer: **the evidence supports roughly "the index, plus about 0.3% to 1.2% a year (centre about 0.7%) from tax handling, a small tilt and modest leverage, with a real chance of lagging for five years or more". It does not support a large, reliable edge from any single lever. Several popular levers (trend exits, option selling for income, low-volatility funds, concentrated picks) have lagged the S&P 500 in live records.**

---

## A. One-page table: what the evidence supports

Verdict scale: STRONG / MODEST / WEAK / NEGATIVE. Edge is per year, relative to holding SPY, for this account.

| # | Lever | Verdict | Expected edge per year (range) | Confidence | How to implement here | Main risk |
|---|---|---|---|---|---|---|
| 1 | Stay fully invested; avoid idle cash | STRONG | Each 10% held in cash costs about 0.4% to 0.6% (my arithmetic: 10% x a 4-6% stock premium over T-bills) | High | Keep the core at 100% stock exposure. Park only true working cash in T-bill funds | None versus the benchmark. It is the benchmark |
| 2 | Low turnover, hold more than one year | STRONG | Avoids a drag of about 0.4% to 1.7% that a high-turnover strategy pays (my arithmetic, section 3) | High | Do not sell winners inside 12 months. Rebalance with new cash and with losers | Temptation to trade. Rules that force sales |
| 3 | Tax-loss harvesting with ETFs | MODEST | +0.2% to +0.6% after tax in the first years, fading towards +0.05% to +0.3% | Medium | Daily scan. Sell lots below cost. Swap into a similar but not identical ETF for 31 days. Reinvest the saving | Wash-sale mistakes. Benefit fades as the portfolio fills with gains |
| 4 | Momentum + quality tilt (ETFs) | MODEST to WEAK | 0% to +0.5% at account level with a 25% sleeve. Honest range -1% to +1% | Low to medium | 10-15% SPMO or MTUM, 10-15% QUAL or SPHQ, held long term | Can lag for 5+ years. MTUM trails SPY by 2.8% a year over the last 5 years |
| 5 | Low-volatility, equal-weight, small or value tilts | WEAK to NEGATIVE (as return sources) | -3% to 0% in live records | Medium | Do not use to beat the S&P | Long lags: USMV -2.9% a year for 15 years, RSP -3% a year for 10 years |
| 6 | Concentrate in "top" or "best" names | NEGATIVE without proven stock-picking skill | Unknown mean, much wider spread of outcomes. Most single stocks lose to T-bills | Medium | Do not concentrate. SPY already holds about 40% in its top 10 | Missing the few big winners. Top-10 stocks lagged over the next 5-10 years historically |
| 7 | Volatility-managed exposure (scale by recent volatility) | MODEST for risk, WEAK for return | 0% to +0.8% in risk-equivalent terms. Not statistically reliable | Low to medium | Keep the current rule as risk control. Slow it down to limit taxable trades | Whipsaw. Realised short-term gains. Works only in some periods |
| 8 | Trend or drawdown exits (move to cash on a signal) | NEGATIVE for return after tax | Live record -4.9% a year (PTLC vs SPY, 2015-2026). Plus about 1% tax drag | Medium to high | Do not use to beat the S&P. Keep only as a disaster brake | Selling the low, missing the rebound, taxes on every exit |
| 9 | Modest leverage (1.1x to 1.3x) | MODEST (it is pay for risk, not skill) | 0% to +1.0%. Centre about +0.5% | Medium | 10-20% of the account in SSO alongside SPY, or a constant modest target | Drawdowns deepen in proportion or more. SSO fell 85% in 2007-09 |
| 10 | Selling index puts or covered calls for "income" | NEGATIVE versus the S&P on return | -1.5% to -7% a year in live indexes and funds since 2006 | Medium to high | Do not use to beat the S&P | Capped upside, full downside, premiums taxed at the short-term rate |
| 11 | Selling puts on cheap, volatile small caps | NEGATIVE / unsupported | No published after-cost edge found. Spreads average about 12.6% of price for retail option trades | Medium | Stop, or cap at a tiny experimental size | Wide spreads, gap risk, assignment in the worst stocks |
| 12 | Insider open-market purchases, small caps | WEAK to MODEST | 0% to +3% on the sleeve before tax. At a 10% sleeve: 0% to +0.3% | Low | 5-10% satellite. Cluster buys by officers. Hold about 6-12 months | Most of the move happens in the first 1-3 days after filing. Costs and taxes |
| 13 | Earnings drift, 10-K text changes, short interest, analyst revisions | WEAK | About 0% after costs for a long-only retail account | Low to medium | Research only. Do not fund | Published anomalies lose about half or more of their return after publication |
| 14 | Frequent active trading of any kind | NEGATIVE (base rate) | Active households lagged by 6.5% a year in the classic study | High | Keep turnover low by rule | Costs, taxes, behaviour |

---

## B. Findings by question

### 1. Factor tilts as a base instead of the plain S&P 500

**Verdict: MODEST for momentum and quality/profitability. WEAK for value in large caps. WEAK to NEGATIVE for low-volatility, equal-weight and size as ways to beat the S&P.**
Expected edge after fees, for a 25% sleeve: 0% to +0.5% a year at account level. It can lag for five to ten years.

What the long-run studies say (before costs and taxes, long-short "paper" portfolios):

- 1927-2013, US: winners minus losers (momentum) averaged 8.3% a year, cheap minus expensive (value) 4.7%, small minus large 2.9%. Sharpe ratios 0.50, 0.39, 0.26. In large stocks only, 1991-2013, value had a Sharpe ratio of 0.06 and momentum 0.24. Source: [Asness, Frazzini, Israel, Moskowitz 2014, "Fact, Fiction and Momentum Investing"](https://www.aqr.com/-/media/AQR/Documents/Journal-Articles/JPM-Fact-Fiction-and-Momentum-Investing.pdf?sc_lang=en). These are long-short spreads. A long-only ETF can capture only part of them.
- Profitability: [Novy-Marx 2013, "The Other Side of Value"](https://www.nber.org/papers/w15940) found profitable firms beat unprofitable ones with about the same power as value. A 2025 follow-up says profitability kept working after publication and explains most of "quality" ([Medhat and Novy-Marx 2025](https://www.nber.org/papers/w33601)). Both authors are tied to Dimensional, which sells profitability funds. I could not read the post-2013 numbers.
- Quality: [Asness, Frazzini, Pedersen 2019, "Quality Minus Junk"](https://link.springer.com/10.1007/s11142-018-9470-2) report high risk-adjusted returns to quality stocks. Others argue it is mostly profitability and investment in disguise.

What happened after publication and after ETFs launched:

- [McLean and Pontiff 2016](https://www.gwern.net/doc/economics/2016-mclean.pdf) studied 97 published predictors. Returns were 26% lower out of sample and 58% lower after publication. The fall was bigger for the predictors with the best backtests. I read the abstract directly.
- [Harvey, Liu, Zhu 2016](https://www.nber.org/papers/w20592) counted 316 published factors and concluded most claimed findings are likely false. They argue a new factor should clear a t-statistic of 3.0, not 2.0.
- [Huang, Song, Xiang 2024, "The Smart Beta Mirage"](https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/smart-beta-mirage/35CA1CEA091A485DDA960BB2D9930BC9): smart-beta indexes beat the market by about 3% a year in the backtest before their ETF listed, then by about -0.5% to -1% a year after listing. The authors blame data mining in index design, not fees.
- [Chen and Velikov 2023](https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/zeroing-in-on-the-expected-returns-of-anomalies/945133D5A3ECEEAF466AEE91551FD225): after trading costs and post-publication decay, the average anomaly earns about 4 basis points a month. The strongest earn about 10. Combinations earn about 20 (roughly 2.4% a year, long-short, before shorting fees).
- [Novy-Marx and Velikov 2016](https://www.nber.org/papers/w20721): strategies with turnover under 50% a month mostly survive costs. Few with higher turnover do. Size, value and profitability can absorb the most money.
- Research Affiliates argued much of the past factor return came from factors getting more expensive ([Arnott, Beck, Kalesnik, West 2016](https://www.researchaffiliates.com/content/dam/ra/publications/pdf/442-how-can-smart-beta-go-horribly-wrong.pdf); [Arnott, Harvey, Kalesnik, Linnainmaa 2019](https://researchaffiliates.com/publications/articles/710-alices-adventures-in-factorland-three-blunders-that-plague-factor-investing)). AQR disputes the size of that effect ([Asness, "My Factor Philippic"](https://www.aqr.com/Insights/Perspectives/My-Factor-Philippic)).
- A more hopeful view: [Jensen, Kelly, Pedersen 2023](https://nber.org/system/files/working_papers/w28432/w28432.pdf) find most factors do replicate when grouped into themes. So the factors are probably real but smaller than first reported.

Live ETF records (total return, annualised, through 2026-11-18; fees included, taxes not):

| Fund | What it is | Period | Fund | SPY | Gap |
|---|---|---|---|---|---|
| [MTUM](https://totalrealreturns.com/n/SPY,MTUM,QUAL) | iShares momentum | 2013-07 to 2026-11 | 13.82% | 13.49% | +0.3% |
| [QUAL](https://totalrealreturns.com/n/SPY,MTUM,QUAL) | iShares quality | 2013-07 to 2026-11 | 13.36% | 13.49% | -0.1% |
| [SPHQ](https://totalrealreturns.com/n/SPY,RSP,SPHQ,USMV) | Invesco S&P 500 quality | 2011-10 to 2026-11 | 14.60% | 14.59% | 0.0% |
| [USMV](https://totalrealreturns.com/n/SPY,RSP,SPHQ,USMV) | iShares minimum volatility | 2011-10 to 2026-11 | 11.67% | 14.59% | -2.9% |
| [RSP](https://totalrealreturns.com/n/SPY,RSP) | Equal-weight S&P 500 | 2003-04 to 2026-11 | 11.16% | 11.20% | 0.0% |
| [RSP](https://totalrealreturns.com/n/SPY,RSP) | same, last 10 years | 2016-11 to 2026-11 | 11.65% | 14.62% | -3.0% |
| [SPMO](https://totalrealreturns.com/n/SPY,MTUM,SPMO,QUAL,SPHQ,USMV,RSP) | Invesco S&P 500 momentum | 2015-10 to 2026-11 | 17.05% | 14.12% | +2.9% |
| [MTUM](https://totalrealreturns.com/n/SPY,MTUM,SPMO,QUAL,SPHQ,USMV,RSP) | same factor, same window | 2015-10 to 2026-11 | 13.73% | 14.12% | -0.4% |
| [SPLV](https://totalrealreturns.com/n/SPY,SPHQ,SPLV,VTV,IWM,SSO) | S&P 500 low volatility | 2011-05 to 2026-11 | 10.10% | 13.51% | -3.4% |
| [VTV](https://totalrealreturns.com/n/SPY,SPHQ,SPLV,VTV,IWM,SSO) | Vanguard large value | 2011-05 to 2026-11 | 11.88% | 13.51% | -1.6% |
| [IWM](https://totalrealreturns.com/n/SPY,SPHQ,SPLV,VTV,IWM,SSO) | Russell 2000 small caps | 2011-05 to 2026-11 | 9.37% | 13.51% | -4.1% |

How to read that table:

- Quality funds matched the S&P 500 almost exactly for 13-15 years. They did not beat it.
- Momentum is split. SPMO beat by 2.9% a year. MTUM, built on the same idea by a different index firm, did not. A 3.3% a year gap between two funds on the "same" factor says fund design and luck dominate the factor itself. SPMO's lead came mostly from 2023-2025 (3-year return 28.0% vs 18.9%). Over the last 12 months it trails (10.3% vs 11.8%). Over the last 5 years MTUM trails SPY by 2.8% a year (7.9% vs 10.7%).
- Low volatility, equal weight, value and small caps all lagged. Low-volatility funds did not even fall less in March 2020 (USMV -33.1%, SPLV -36.3%, SPY -33.7%).
- Multi-factor funds from Dimensional and Avantis are close to the market: since 2021-06, AVUS 11.4%, DFAC 10.7%, SPY 12.0%, total market VTI 10.9% ([source](https://totalrealreturns.com/n/SPY,VTI,AVUS,DFAC,AVUV,VTV,VBR)). Five years is too short to judge.

Taxes on factor tilts:

- [Israel and Moskowitz, "How Tax Efficient Are Equity Styles?"](https://elmwealth.com/wp-content/uploads/2017/06/how_tax_efficient_are_equity_styles.pdf) (1974-2010 indexes): momentum has five to seven times the turnover of value but about the same tax rate, because it sells losers early (short-term losses) and holds winners (long-term gains). Value is hurt by dividends: it loses more than 1.2% a year to tax in up and down markets. After tax, momentum beat growth by about 2.2% a year in large caps at 2011 tax rates. These numbers are before trading costs.
- Held inside an ETF, most of the turnover does not reach the holder. ETFs avoid distributing gains through in-kind redemptions ([Moussawi, Shen, Velthuis 2025](https://academic.oup.com/rfs/article/38/10/2988/8191041)). So a momentum ETF held for years is taxed much like SPY. Switching between funds is what triggers tax.

Other points:

- Momentum crashes are real in long-short form. In March-May 2009 past losers rose 163% and past winners 8% ([Daniel and Moskowitz 2016](https://www.nber.org/papers/w20439)). Long-only momentum funds lag in sharp rebounds but do not crash like that.
- Equal weight: the long-run edge came from small and value exposure plus monthly rebalancing ([Plyakha, Uppal, Vilkov](https://climateinstitute.edhec.edu/publications/why-does-equal-weighted-portfolio-outperform-value-and-price-weighted-portfolios)), with higher volatility (17.9% vs 15.8%). It has not paid for a decade.
- Low volatility: the paper premium shrinks sharply after costs and is largely explained by profitability ([summary of Novy-Marx and Velikov 2022](https://alphaarchitect.com/betting-against-beta-new-insights/)).

**What a careful practitioner would hold:** a cap-weighted core, plus a minority tilt to momentum and to quality/profitability, in cheap ETFs, held for years. Momentum and quality tend to do well at different times. Expect the tilt to add little, and treat any gain as a bonus.

**How long it can lag:** five to ten years is normal for a single factor. Large-cap value earned almost no premium for 23 years (Sharpe ratio 0.06, 1991-2013).

### 2. "Hold the top / best names": concentration

**Verdict: NEGATIVE for a retail account without proven stock-picking skill. The index already gives you the top names.**

- [Bessembinder 2018, "Do Stocks Outperform Treasury Bills?"](https://econpapers.repec.org/RePEc:eee:jfinec:v:129:y:2018:i:3:p:440-457) (US, 1926-2016): 58% of stocks returned less than one-month T-bills over their lives. About 4% of stocks (roughly 1,092 of 26,000) account for all net wealth created. A [2024 update](https://news.wpcarey.asu.edu/20240805-asu-study-stock-market-winners-uncovers-some-surprises) covering 29,078 stocks through 2023 confirms the skew. A newer working paper through 2025 reports that long-term holders of nearly 60% of stocks lost wealth ([ASU data page](https://wpcarey.asu.edu/department-finance/faculty-research/do-stocks-outperform-treasury-bills)).
  - What it means: a randomly chosen handful of stocks will most likely lag the index, because the index always owns the rare giant winners. Concentration lowers your expected chance of matching the market unless you can pick winners in advance.
- Stocks after they reach the top 10 by size: [Dimensional, "Large and In Charge?"](https://www.dimensional.com/us-en/insights/large-and-in-charge-giant-firms-atop-market-is-nothing-new) (US, 1927-2019). They beat the market by 0.7% a year over the next three years, then lagged it on average over five and ten years. The exact five- and ten-year numbers are in a chart I could not read.
- "Best ideas" of professional managers: [Antón, Cohen, Polk 2021, "Best Ideas"](https://personal.lse.ac.uk/polk/research/BestIdeas.pdf). The single highest-conviction stock of each fund manager beat the market, and the rest of that manager's portfolio, by about 2.8% to 4.5% a year, depending on the benchmark. Before costs. The other holdings showed no significant outperformance. I read the abstract directly.
  - This is evidence that professionals have a little skill in their top pick. It is not evidence that a small account can find such picks. A separate study of hedge funds found their best ideas added little ([Hedgeweek summary](https://hedgeweek.com/news/hedge-funds-best-ideas-are-not-holy-grail-performance)).
- Mega-cap dominance: the top 10 stocks were about 40% of the S&P 500 in 2025, the highest since the 1960s. Studies disagree on what follows. Dimensional and State Street find concentration barely predicts returns ([Dimensional 2026](https://www.dimensional.com/chmedia/477974/source/market-concentration-at-high-valuations-what-to-do-pro.pdf); [review of "The Fallacy of Concentration Risk"](https://quantpedia.com/the-fallacy-of-concentration-risk/)). Goldman Sachs and others find weaker returns after past peaks such as 1973 and 2000 ([Goldman Sachs](https://www.goldmansachs.com/insights/articles/is-the-sp-too-concentrated)). Nobody has shown that today's leaders will keep leading.

**Practical reading:** holding SPY already puts about 40 cents of every dollar in the ten largest companies. Adding more is a bet with no published support. Holding fewer than about 30 names mostly adds risk.

### 3. Taxes as an edge

**Verdict: STRONG for deferral and long-term rates (it is arithmetic, not a forecast). MODEST for tax-loss harvesting at this bracket: +0.2% to +0.6% a year early, fading.**

The value of simply not selling (my arithmetic, 10% a year pre-tax return, no dividends):

| Approach | After-tax return, 10 years | After-tax return, 20 years |
|---|---|---|
| Buy and hold, pay 15% once at the end | 8.9% | 9.3% |
| Realise every gain each year at 15% (long-term) | 8.5% | 8.5% |
| Realise every gain each year at 24% (short-term) | 7.6% | 7.6% |

So a strategy that turns its whole book over inside a year must earn about 1.3% to 1.7% a year more before tax just to tie buy-and-hold. This is the largest single number in this file, and it is certain.

Tax-loss harvesting (selling positions that are below cost to bank the loss, then buying something similar):

- [Chaudhuri, Burnham, Lo 2020](https://rpc.cfainstitute.org/research/financial-analysts-journal/2020/empirical-evaluation-tax-loss-harvesting-alpha) ([MIT copy](https://dspace.mit.edu/handle/1721.1/135992)): the 500 largest US stocks, 1926-2018, monthly harvesting, tax rates 35% short-term and 15% long-term. Benefit 1.08% a year before costs. With the wash-sale rule 0.82%. Trading costs take about 0.13%. By period: 2.13% (1926-49), 0.51% (1949-72), 1.08% (1972-95), 0.81% (1995-2018). It is largest in volatile, falling markets.
  - Those numbers assume a 35% short-term rate and that every harvested loss offsets short-term gains. This account's rate is 24%, so scale down.
- [Vanguard 2024, "Tax-loss harvesting: Why a personalized approach is important"](https://corporate.vanguard.com/content/dam/corp/research/pdf/tax_loss_harvesting_why_a_personalized_approach_is_important.pdf) (simulations on 1982-2023 returns, 15-year horizon). I read the paper. Their profile with **24% ordinary and 15% long-term rates** is the closest match to this account:
  - Best practice (daily scan, all tax savings reinvested, regular new deposits, 400-stock direct index): **0.65% a year**.
  - Poor practice (quarterly scan, half of savings reinvested, no deposits, one fund): **0.05% a year**.
  - What drives the result: tax rates 27%, reinvesting the savings 25%, volatility 17%, market return 12%, scan frequency 7%, having gains to offset 5%, owning individual stocks instead of funds only 3%, new deposits 2%, time 2%.
  - The benefit fades as the portfolio fills with gains ("ossification").
- [Kitces](https://www.kitces.com/blog/evaluating-the-tax-deferral-and-tax-bracket-arbitrage-benefits-of-tax-loss-harvesting/): harvesting is mostly an interest-free loan, not a tax saving, because the new shares have a lower cost and a bigger gain later. At a 15% rate he values it at up to about 0.30% a year, falling to just under 0.20% in the long run. The real gain comes from the rate gap: deducting at a high rate now, paying at a lower rate later.
- The decay: AQR reports that the tax benefit of plain direct indexing falls close to zero after a few years, and cumulative net losses top out around 30% of the starting capital ([AQR 2021](https://www.aqr.com/Insights/Research/Tax-Aware-Investing/Improving-Direct-Indexing-13030-and-15050-Strategies)). The exact year-by-year path is in a chart I could not read.

What it is worth for this account (my arithmetic):

- Up to $3,000 of net losses a year can be deducted against ordinary income. At 24% that saves $720 now. The replacement shares will later owe about 15% on the same $3,000, or $450. Net permanent gain about $270 a year, plus the use of $450 for years. On $88,000 that is roughly 0.3% to 0.8% a year, in years when $3,000 of losses can be found.
- If the account also makes short-term gains (active trades, option premiums), harvested losses offset those at 24%. Then the harvest is worth more.
- Honest range: +0.2% to +0.6% a year in the first few years, less after a long bull run.

Rules that matter ([SEC investor.gov on wash sales](https://www.investor.gov/introduction-investing/investing-basics/glossary/wash-sales); [Schwab summary](https://www.schwab.com/learn/story/year-end-tax-trading-wash-sales-and-more)):

- A loss is disallowed if you buy the same or a "substantially identical" security within 30 days before or after the sale. Options on the same stock count. Swap SPY into a fund on a different index for 31 days, not into another S&P 500 fund.
- Net losses offset $3,000 of ordinary income a year. The rest carries forward without limit.

Direct indexing (owning 50-100+ individual stocks to harvest more losses) at $88,000:

- Vanguard's own numbers say owning individual stocks explains only about 3% of the variation in the benefit (9% for wealthy investors). Tax rate and reinvesting matter far more.
- It costs more trades, more wash-sale bookkeeping, and tracking error versus the index.
- Cheap commercial versions exist (for example an S&P 500 direct-index account with a $5,000 minimum and a 0.09% fee, [Wealthfront](https://support.wealthfront.com/hc/en-us/articles/49518180999188-Wealthfront-Direct-Indexing-Accounts-Fees-Minimums)). Building one in-house for $88,000 is not worth the complexity.
- **Verdict: harvest at the ETF level with three to six funds. Skip direct indexing.**

### 4. Cash drag, market timing and drawdown control

**Verdict: holding cash is a certain cost. Volatility scaling has MODEST support as risk control and WEAK support as a return source. Trend or drawdown exits are NEGATIVE for beating the S&P after tax.**

Cash drag:

- US stocks beat T-bills by 7.8% a year on average over 1926-2016 (Table 1 of [Cederburg et al. 2020](https://www.lehigh.edu/~xuy219/research/COWY.pdf)). Looking forward I assume a more cautious 4% to 6%. Each 10% of the account left in cash therefore costs about 0.4% to 0.6% a year on average, and cost about 0.8% in the past. (my arithmetic)
- Investing at once beat spreading purchases out about two-thirds of the time, 1976-2022 ([Vanguard 2023](https://corporate.vanguard.com/content/dam/corp/research/pdf/cost_averaging_invest_now_or_temporarily_hold_your_cash.pdf)).

Volatility-managed portfolios (hold more when recent volatility is low, less when it is high):

- For: [Moreira and Muir 2017](https://www.nber.org/papers/w22208) report an "alpha" of 4.9% a year for the volatility-managed US market, and a Sharpe ratio about 25% higher. Monthly data from 1926. Before costs.
- For: [Harvey, Hoyle, Korgaonkar, Rattray, Sargaison, Van Hemert 2018](https://people.duke.edu/~charvey/Research/Published_Papers/P135_The_impact_of.pdf) (US stocks 1926-2017, plus 60 other assets). The Sharpe ratio of US stocks rises from 0.40 to between 0.48 and 0.51. After their cost estimate it is 0.477 versus 0.483 before. Volatility scaling also cuts extreme losses. It helps stocks and credit, and does nothing for bonds, currencies or commodities. I read this in the paper.
- Against: [Cederburg, O'Doherty, Wang, Yan 2020](https://www.lehigh.edu/~xuy219/research/COWY.pdf) tested 103 equity strategies through 2016. Volatility-managed versions beat the plain versions in 53 cases and lost in 50. Only 8 differences were statistically significant. For the market itself the Sharpe ratio went from 0.42 to 0.51, with a p-value of 0.30. That is not distinguishable from luck. The real-time versions of the Moreira-Muir strategy usually did worse than simply holding. I read this in the paper.
- Against: [Barroso and Detzel 2021](https://econpapers.repec.org/RePEc:eee:jfinec:v:140:y:2021:i:3:p:744-767): after trading costs the benefit disappears for every factor except the market, and for the market it appears only in high-sentiment periods.
- Net reading: for the stock market as a whole there is a small, uncertain risk-adjusted gain (Sharpe +0.05 to +0.10) and a clearer reduction in the worst outcomes. In return terms that is worth 0% to about +0.8% a year at equal risk (my arithmetic). None of these studies include taxes. Every exposure cut that sells shares at a gain creates tax.

Trend following (hold when price is above a moving average, otherwise cash):

- [Moskowitz, Ooi, Pedersen 2012](https://alphaarchitect.com/time-series-momentum-aka-trend-following-the-historical-evidence/) and [Hurst, Ooi, Pedersen 2017](https://fairmodel.econ.yale.edu/ec439/hurst.pdf) show trend following worked across dozens of futures markets for a century. That evidence is for diversified long-short futures portfolios, not for timing one stock index. [Huang, Li, Wang, Zhou 2020](https://buttondown.com/mdickens/archive/contra-time-series-momentum-is-it-there/) argue the asset-by-asset statistical evidence is weak.
- [Faber 2007](https://papers.ssrn.com/abstract=962461) popularised the 10-month moving-average rule. In his tests it gave similar returns to buy-and-hold with smaller drawdowns, largely thanks to avoiding 2008. I could not obtain his own out-of-sample numbers.
- Live record: the Pacer Trendpilot US Large Cap ETF (PTLC) applies a 200-day rule to the S&P 500. From 2015-06 to 2026-11 it returned **8.52% a year versus 13.37% for SPY**. Its worst drawdown was -26.6% versus -33.7% ([source](https://totalrealreturns.com/n/SPY,PTLC,PBP)). It gave up 4.9% a year to save 7 points of drawdown once.
- Taxes: trend strategies realise gains often. Newfound Research estimates about 1% a year of extra tax drag at 15%/25% rates ([Newfound](https://blog.thinknewfound.com/tag/taxes/)). Practitioner reviews advise holding trend strategies in tax-sheltered accounts.

**Verdict on exposure-cutting rules:** a taxable account that wants to beat the S&P should not use a rule that moves to cash on price signals. If one is kept for safety, make it rare and deep (a true disaster brake), and accept that it costs return. A slow volatility-scaling rule is the most defensible form, used for risk control, with a wide no-trade band so it rarely sells.

**How long it can lag:** PTLC has lagged for 11 years. Volatility targeting lags in calm-then-crash-then-rebound sequences such as 2020.

### 5. Leverage on the index (1.2x to 1.5x)

**Verdict: MODEST. Leverage raises expected return because it raises risk. It adds nothing risk-adjusted. Expected gain at 1.25x is about 0% to +1% a year, centre +0.5%.**

- Theory for it: [Ayres and Nalebuff 2008](https://papers.ssrn.com/abstract=1149340). Young investors with decades of future savings should hold up to 2:1 stock exposure early in life. In simulations on 1871-2007 data this beat 100% stocks. The result depends on a low borrowing cost and on the investor not quitting after a crash. Critics focus on the second point.
- Daily-reset 2x funds: return is roughly twice the index's compounded growth, minus a "volatility drag" of about (index volatility squared) per year, minus financing and fees ([Avellaneda and Zhang 2009, also crediting Cheng and Madhavan 2009](https://math.nyu.edu/~avellane/LeveragedETF20090515.pdf); [Trainor 2026](https://www.mdpi.com/1911-8074/19/1/20)). At 16% volatility the drag is about 2.6% a year. At 19% it is 3.6%. (my arithmetic from their formula)
- Live record: [SSO](https://totalrealreturns.com/n/SPY,SSO) (2x S&P 500), 2006-06 to 2026-11: **14.99% a year versus 11.03% for SPY**. Worst drawdown **-84.7% versus -55.2%**. So 2x exposure delivered about 1.36x the return and 1.5x the drawdown, in a period that was very good for US stocks.
- Kelly sizing (the leverage that maximises long-run growth) equals the expected excess return divided by the variance. With the 1926-2016 numbers (7.8% excess, 18.6% volatility) that is about 2.25x. With a more cautious forward estimate (4% excess, 17% volatility) it is about 1.4x. Because the inputs are uncertain, practitioners use half of Kelly: about 0.7x to 1.1x. (my arithmetic; the inputs are from the Cederburg table.) I did not find a published S&P 500 Kelly study to cite.
- Expected gain at 1.25x (my arithmetic): extra return 0.25 x (4.5% to 6.5%) = 1.1% to 1.6%. Minus extra volatility drag about 0.8%. Minus SSO fees and financing spread about 0.3%. Net about 0% to +0.5%. Scaling SSO's live record gives about +1%. Hence 0% to +1%.

**What a sober practitioner concludes:** 1.1x to 1.3x is defensible for a long-horizon investor who will not sell in a crash. 1.5x is at or above full Kelly if future returns are lower than the past. Leverage is the only lever here with a large effect, and it works by taking more risk. It makes "risk-adjusted" comparison harder, not easier.

**How long it can lag:** leveraged holdings fall more in every bear market. After 2007-09, SSO took years longer than SPY to recover.

### 6. Option selling

**Verdict: NEGATIVE for beating the S&P on return. MODEST as a pre-tax risk reducer on the index. NEGATIVE / unsupported for small-cap single stocks. Worse after tax.**

Index put-writing and covered calls:

- [Bondarenko 2019, for Cboe](https://cdn.cboe.com/resources/education/research_publications/PutWriteCBOE19_v14_by_Prof_Oleg_Bondarenko_as_of_June_14.pdf). I read the paper. Cboe PutWrite index (PUT: sell one-month at-the-money S&P 500 puts, hold T-bills), 1986-06 to 2018-12: **9.54% a year versus 9.80% for the S&P 500**, with volatility 9.95% versus 14.93%. Sharpe 0.65 versus 0.49. The index was launched in 2007. From 2006 to 2018: **PUT 5.97%, S&P 500 7.59%**, Sharpe 0.50 versus 0.51. So the risk-adjusted advantage disappeared after the index went live. Before trading costs. Implied volatility averaged 19.3% versus realised 15.1% (1990-2018). That gap is the premium sellers collect.
- More recent: one data site shows about 8.5% a year for PUT versus about 15.4% for the S&P 500 over the last ten years ([PortfoliosLab](https://portfolioslab.com/tools/stock-comparison/%5ESP500TR/%5EPUT); as-of date unclear).
- Live funds: the Invesco S&P 500 BuyWrite ETF (PBP) returned 6.13% a year versus 13.37% for SPY since 2015-06, with the same -33% drawdown in 2020 ([source](https://totalrealreturns.com/n/SPY,PTLC,PBP)). Global X's covered-call fund XYLD returned 8.08% versus 14.03% since 2013-06, again with the same drawdown ([source](https://totalrealreturns.com/n/SPY,XYLD)).
- [Israelov and Nielsen 2014, "Covered Call Strategies: One Fact and Eight Myths"](https://www.aqr.com/-/media/AQR/Documents/Insights/Journal-Article/FAJ-Covered-Call-Strategies-One-Fact-and-Eight-Myths.pdf). 1986-2013: S&P 500 excess return 5.4% with 18.5% volatility. Covered-call index 4.4% with 13.4%. Worst drawdown -61.7% versus -43.0%. Their points: option premium is not "income". Covered calls do not protect the downside. The strategy is simply about two-thirds stock exposure plus a short-volatility bet.
- [Israelov, "PutWrite versus BuyWrite"](https://www.aqr.com/Research-Archive/Research/White-Papers/PutWrite-versus-BuyWrite-Yes-Put-Call-Parity-Holds-Here-Too): the put-write index's 1.1% a year lead over the covered-call index is a quirk of index construction on expiry mornings, not an edge.

Single stocks and small caps:

- The reliable premium is in **index** options. For single stocks the average gap between implied and realised volatility is close to zero. The index premium is mostly a premium for correlation risk ([Ilmanen, "Expected Returns", chapter on volatility selling](https://www.oreilly.com/library/view/expected-returns-an/9781119990772/ilma_9781119990772_oeb_c15_r1.html), summarising Driessen, Maenhout and Vilkov).
- [Cao and Han 2013](https://econpapers.repec.org/RePEc:eee:jfinec:v:108:y:2013:i:1:p:231-249): options on stocks with high stock-specific volatility are relatively overpriced, so selling them and hedging earned a premium on paper. The authors read it as pay for market makers' costs in hard-to-hedge names. About 40% of it is explained by trading frictions. I found no study showing it survives real bid-ask spreads for an unhedged retail seller.
- Retail fills: [Bryzgalova, Pavlova, Sikorskaya 2023](https://lbsresearch.london.edu/id/eprint/2827): retail option trades face average quoted spreads of about 12.6% and lose money on average. Three wholesalers capture most of the flow.
- An unhedged short put on a cheap, volatile small cap is mainly a bet on the stock. That corner of the market (small, volatile, lottery-like) is where long-run stock returns are worst.

Taxes: premiums from stock and ETF options are short-term gains (24% here). Broad index options such as SPX are generally taxed 60% long-term / 40% short-term whatever the holding period (standard "Section 1256" treatment; I did not open an IRS source, see section C). Either way the strategy converts deferred long-term stock gains into yearly taxable income.

**How long it can lag:** covered-call and put-write funds have lagged the S&P 500 for the whole period since they became investable (2006 onward).

### 7. Signals a small account can use that big funds cannot

**Verdict: WEAK overall. Insider purchases in small caps are the best of a thin field (WEAK to MODEST). Expect roughly zero after costs and short-term tax, and size any test at 10% of the account or less.**

General rule from the anomaly literature: published signals lose about 58% of their return after publication (McLean and Pontiff, above), and after trading costs the average one earns about 0.5% a year long-short (Chen and Velikov, above). Decay is slower in illiquid, volatile stocks. That is where a small account has an advantage, and also where spreads are widest.

Insider open-market purchases:

- [Lakonishok and Lee 2001](https://econpapers.repec.org/RePEc:oup:rfinst:v:14:y:2001:i:1:p:79-111) (1975-1995): insider buying predicts returns. The effect comes from purchases, not sales, and from smaller firms.
- [Jeng, Metrick, Zeckhauser 2003](https://econpapers.repec.org/RePEc:tpr:restat:v:85:y:2003:i:2:p:453-471): insiders' own purchases earned more than 6% a year abnormal return. Sales earned nothing. That is the insider's return from the trade date, not an outsider's return from the filing date.
- [Cohen, Malloy, Pomorski 2012, "Decoding Inside Information"](https://www.nber.org/papers/w16454) (1986-2007): about 55% of insider trades are "routine" (same month every year) and predict nothing. The rest, "opportunistic" trades, earned 0.82% a month value-weighted and 1.80% a month equal-weighted in a long-short portfolio. Before costs.
- After publication: I found no peer-reviewed study that isolates 2015-2025. A master's thesis reports the opportunistic effect is still positive but smaller ([Aalto](https://aaltodoc.aalto.fi/bitstreams/fbbd1ec6-d5b4-44dd-881d-144ffd0ea21c/download)). A 2026 paper finds insider trades became less informative after the 2022 rule change on pre-planned trades ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S105905602600763X)).
- Microcaps, 2018-2024: a preprint on 13,534 purchases in $30M-$500M companies finds the price reaction happens in the first two to three sessions after the filing (first-day abnormal return 0.9% to 4.1%, larger for beaten-down stocks). A portfolio that skips the first day earns no significant alpha ([Zhao 2026, arXiv](https://arxiv.org/abs/2602.06198)). Not peer reviewed. The author withdrew several claims in a revision.
- Reading: the signal is real but small, fast, and concentrated in illiquid names. A long-only account that buys the morning after a filing and holds for months may earn 0% to 3% a year more than the small-cap market, before costs and tax. Small caps themselves have trailed the S&P 500 by about 4% a year since 2011 (IWM above), so the sleeve can beat its own benchmark and still lag SPY.

Post-earnings-announcement drift (prices keep moving in the direction of an earnings surprise):

- [Martineau 2022, "Rest in Peace Post-Earnings Announcement Drift"](https://nowpublishers.com/article/Details/CFR-0122): gone in large and mid-size stocks since about 2006. Gone in microcaps more recently. Prices now react on the announcement day.
- Replies: two 2025 papers say drift survives. [Subrahmanyam's reconciliation](https://anderson-review.ucla.edu/is-post-earnings-announcement-drift-a-thing-again) (2001-2024): with all stocks the drift has a t-statistic of 2.18. Without microcaps it is 1.43, not significant. Microcaps are about 3% of market value.
- Reading: whatever is left lives in the smallest, costliest stocks. WEAK.

"Lazy Prices":

- [Cohen, Malloy, Nguyen 2020](https://www.nber.org/papers/w25084): firms that change the wording of their annual and quarterly reports underperform. A long-short portfolio earned up to 1.88% a month (1995-2014). Before costs. Most of the profit is on the short side, which a long-only account cannot capture. I found no published out-of-sample test. A student replication through 2024 gave mixed results ([project page](https://akankshagavade.github.io/project)). WEAK until tested.

Short interest:

- [Boehmer, Jones, Zhang 2008](https://econpapers.repec.org/RePEc:bla:jfinan:v:63:y:2008:i:2:p:491-527): heavily shorted stocks lagged lightly shorted ones by 1.16% over 20 trading days. Later work finds it persisted in 2010-2015. For a long-only account this is an **avoid** list, not a buy signal. Useful as a filter. WEAK to MODEST.

Analyst revisions: I did not find a clean post-2015 study in the time available. Not verified.

Microcap liquidity premium:

- [Ben-Rephael, Kadan, Wohl 2015](https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/abs/diminishing-liquidity-premium/41DA4AD9FB6258DFBA52BF8DA1242252): the premium for holding illiquid US stocks has shrunk over four decades and now exists only in the smallest stocks. Strategies built on it are "virtually unprofitable". A later 45-country study agrees ([summary](https://alphaarchitect.com/the-vanishing-illiquidity-premium/)). WEAK.

### 8. Base rates: who actually beats the S&P 500

**Verdict: the base rate is NEGATIVE. After costs and taxes, well under one in ten professional funds beats the index over 20 years.**

- Mutual funds, [SPIVA US year-end 2025](https://www.spglobal.com/spdji/en/spiva/article/spiva-us-year-end-2025/): 79% of active large-cap funds lagged the S&P 500 in 2025 (65% in 2024). Over three and five years: 67% and 89% ([S&P, SPIVA by the numbers](https://www.spglobal.com/spdji/en/index-tv/article/spiva-by-the-numbers-a-global-perspective/); [InvestmentNews](https://investmentnews.com/equities/active-managers-stumble-again-in-2025-as-large-caps-dominate/265541)). Over 20 years about 93% lagged and only about a third of funds survived. The 20-year figures come from secondary summaries. I could not open the S&P PDF.
- After tax it is worse. [SPIVA After-Tax Scorecard, year-end 2023](https://spglobal.com/spdji/en/documents/spiva/spiva-after-tax-scorecard-year-end-2023.pdf): over 20 years 94.0% of all US stock funds lagged before tax and **98.3% after tax**.
- Persistence: of US stock funds in the top quarter in 2021, almost none (reported as 0.46%) were still there every year through 2025 ([S&P Persistence Scorecard 2025](https://www.spglobal.com/spdji/en/documents/spiva/persistence-scorecard-year-end-2025.pdf), via secondary summary). Past winners do not stay winners.
- [Fama and French 2010, "Luck versus Skill"](https://econpapers.repec.org/RePEc:bla:jfinan:v:65:y:2010:i:5:p:1915-1947): funds as a group hold the market and lose by their costs. A few managers have real skill before costs. After costs, true alpha is negative for most, even for funds with strong records ([Chicago Booth summary](https://www.chicagobooth.edu/review/measuring-chance)).
- [Bessembinder, Cooper, Zhang 2023](https://econpapers.repec.org/RePEc:eee:jfinec:v:147:y:2023:i:1:p:132-158): over 1991-2020 more than two-thirds of US stock funds lagged SPY on compound return ([SMU summary](https://www.smu.edu/cox/coxtoday-magazine/2022-07-21-mind-the-gap-mutual-funds-often-underperform-in-long-run)). Funds that look fine on monthly averages often lose over decades.
- Retail traders, [Barber and Odean 2000](https://faculty.haas.berkeley.edu/odean/Papers%20current%20versions/Individual_Investor_Performance_Final.pdf): of 66,465 households, the most active traders earned 11.4% a year while the market earned 17.9% (1991-1996). Among day traders, fewer than 1% were reliably profitable after costs ([Barber, Lee, Liu, Odean](https://faculty.haas.berkeley.edu/odean/papers/day%20traders/Day%20Trading%20Skill%20110523.pdf)).
- Behaviour: fund investors earned about 1.2% a year less than their own funds over the ten years to 2025 because of badly timed buying and selling ([Morningstar, Mind the Gap 2025](https://www.morningstar.com/business/insights/research/mind-the-gap-2025)).
- Hedge funds and retail "quant" strategies: I did not find a current, clean number. Not verified. Older comparisons show hedge fund indexes trailing the S&P 500 after fees since the 2000s, with lower risk.

What separates the few that win, according to these sources: low cost, low turnover, long holding periods, a small asset base in a niche too small for big funds, and a real information or structural edge. None of them win by trading more.

---

## C. What I could not verify

Things to treat with care. I say what I looked for and what I got.

1. **Live ETF returns.** All come from one third-party site, read through an automated page reader. Issuer fact sheets (iShares, Invesco) would not load their performance tables. I did confirm from iShares pages that 2026 year-to-date returns to 2026-11-18 were MTUM +4.65%, QUAL +10.28%, USMV +8.69%, IVV +8.03%. Spot-check the longer figures against issuer data before quoting them publicly.
2. **SPIVA 20-year figures for 2025** (about 93% lagging, about 35% survival) and the persistence figure (0.46%). S&P's site blocked direct reading. The numbers are from secondary summaries. The 2023 after-tax figures are from a search extract of the S&P document.
3. **PUT index over the last ten years** (about 8.5% vs 15.4%). One data site, date unclear. The 1986-2018 and 2006-2018 figures are from the Cboe paper itself.
4. **Faber's own out-of-sample results** and the exact numbers in Hurst, Ooi, Pedersen and Moskowitz, Ooi, Pedersen. Not opened. The PTLC live record is my main evidence on trend rules.
5. **Dimensional's 5- and 10-year numbers** for stocks after joining the top 10, and **AQR's year-by-year decay** of direct-indexing benefits. Both are in charts, not text.
6. **Medhat and Novy-Marx 2025** post-2013 profitability numbers. Abstract only.
7. **Driessen, Maenhout, Vilkov** on single-stock versus index option premiums. Cited through Ilmanen's book summary, not the paper.
8. **Insider-signal performance after 2015.** No peer-reviewed study found. The evidence is a thesis, one 2026 paper on a rule change, and one unreviewed preprint.
9. **"Lazy Prices" after 2014, analyst revisions after 2015, hedge fund results, retail quant results.** Nothing solid found.
10. **Kelly leverage for the S&P 500.** No published study found. The figures are my arithmetic on published return and volatility numbers.
11. **Tax effects of volatility targeting and of leveraged ETFs in a taxable account.** I found no study. The claim that they raise taxes is reasoning, not a citation.
12. **Moussawi et al.'s 0.92% a year** ETF tax advantage over mutual funds is quoted in other papers. I did not see it in the article itself, so I did not use the number.
13. **Chaudhuri, Burnham, Lo** details come from the CFA Institute summary. MIT's copy was rate-limited.
14. All "(my arithmetic)" figures are simple calculations. They are not backtests and not forecasts.
15. Two statements rest on general knowledge rather than a source opened today: that small, volatile, lottery-like stocks have had poor long-run returns, and the 60/40 tax treatment of broad index options. Check the second with a tax adviser or IRS Publication 550 before relying on it.

---

## D. Recommended design for this account

Priority order. Sizes are shares of the roughly $88,000 account.

A worked example that adds up: 55% SPY, 13% SPMO, 12% QUAL, 15% SSO, 5% satellite. Stock exposure is then 115% of the account (SSO counts twice).

**1. Core: 50-60% in a plain S&P 500 ETF, held and not traded.**
SPY, IVV or VOO. This is the benchmark, so it cannot lag. Never sell a lot at a gain inside 12 months. Expected edge: 0%.

**2. Stay fully invested. Target 100% to 125% stock exposure at all times. Hold no strategic cash.**
Any standing cash reserve costs about 0.4% to 0.6% a year per 10% of the account. Keep only working cash, in a T-bill fund.

**3. Tilt sleeve: about 25% in momentum and quality ETFs, held for years.**
For example 12-13% SPMO (or MTUM) and 12-13% QUAL (or SPHQ). Do not rotate between them on performance. Expected edge at account level: 0% to +0.5% a year. Realistic range -1% to +1%. It can lag for five years.

**4. Tax management on everything above.**
- Hold more than one year before selling any gain.
- Scan daily for lots below cost. Sell them, and move the money for 31 days into a similar fund on a different index (for example SPY to VTI or SCHX, SPMO to MTUM, QUAL to SPHQ). Reinvest the tax saved.
- Aim to bank at least $3,000 of net losses in years when the market offers them.
- No direct indexing.
Expected edge: +0.2% to +0.6% a year after tax at first, fading. Plus the avoided drag of 0.4% to 1.7% that an active book would pay.

**5. Modest leverage: 10-20% in SSO, giving 1.10x to 1.20x total exposure.**
Use the existing volatility rule only to move within that band, slowly, with a wide no-trade zone. Cap total exposure at about 1.3x, not 1.5x. Expected edge: 0% to +1% a year. This is pay for extra risk. Expect drawdowns 1.2 to 1.4 times the index's.

**6. Satellite: 0-10% (5% in the example) for one capacity-limited test, insider cluster buys in small caps.**
Rules: open-market purchases only, two or more officers or directors, not "routine" same-month buyers, buy the first morning after the filing, hold 6-12 months, avoid heavily shorted names. Run it as a labelled experiment with a kill date. Expected edge: 0% to +0.3% a year at account level. It may well be negative after costs.

**7. Stop or shrink to near zero.**
- Selling puts or calls for income, especially on small caps.
- Any rule that sells the core to cash on a price trend or a moderate drawdown. Keep one deep disaster brake at most.
- Low-volatility, equal-weight and small-cap funds as "beat the market" tools.
- Concentrated single-stock bets.
- Any strategy whose holding period is under a year, unless its measured edge clears about 2% a year after costs.

**Expected total, said plainly:**

- Versus SPY, after tax: about **+0.3% to +1.2% a year** on average, centre about **+0.7%**. Roughly: tax handling +0.2% to +0.6%, leverage 0% to +1%, tilt 0% to +0.5%, satellite about 0%. The pieces will not all land at the top of their ranges together.
- On a risk-adjusted basis the honest figure is smaller, about **+0.2% to +0.7% a year**, because leverage is pay for risk and not an edge.
- Uncertainty: the tilt and leverage can each cost 1% to 3% in a bad year. In my judgement (not a published figure) there is roughly a one-in-three chance this design trails SPY over any given five-year period.
- Nothing in the published evidence supports expecting 3% or more a year over the S&P 500 from rules a small account can run.

**What this means for the current build** (based on the project notes I was given, not on a code review):

- The floor at 85% of equity with a cash reserve is a structural drag of about 0.6% to 0.9% a year unless leverage makes it up. The evidence favours a 100% base.
- A 1.5x cap on the SPY+SSO floor is at or above full Kelly under cautious return assumptions. 1.25x to 1.3x is easier to defend.
- The volatility rule has modest support as risk control. It has no reliable support as a source of return. Its taxable trading is unmeasured in the literature and should be measured here.
- Kill-switch and drawdown rules that cut exposure behave like trend exits. The live record of trend exits is poor. Keep them rare.
- The option-selling stack (cash-secured puts on a screened universe) is the lever with the weakest published support and the worst tax treatment.
- The tax filter, long-term-hold rule and harvesting modules are aimed at the lever with the strongest support. They deserve priority over new signals.
- Insider (EDGAR Form 4) work fits the "small account" niche, but the published edge is small and fast. Size it as an experiment.
