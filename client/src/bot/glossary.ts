// Bot page — the glossary behind every "?" (client/src/bot/help.tsx). One
// plain line per term, written from the code that produces the number, not
// from textbook definitions: when a mechanism changes, its line changes with
// it (glossary.test.ts pins that every key the panels use exists here).
// Pure data; no DOM.

import { KILL_LINE_GLOSS } from "../../../shared/haltWords";

export interface GlossaryEntry { term: string; text: string }

export const GLOSSARY = {
  // ── KPI strip ──
  expectancy: { term: "Expectancy per closed lot", text: "Every lot the trading sleeves (options + stocks, not the ETF floor) closed at the broker since the window start, averaged as P&L ÷ cost per lot — a sold option's cost is its credit. Dim beneath: the bot log's trigger-price number over the few records that carry one, frozen since its last counted exit and blind to unattributed exits." },
  winRate: { term: "Win rate · breakeven", text: "Share of the broker's closed lots (options + stocks) that made money, beside the win rate the strategy NEEDS to break even given its average win and loss per lot; below it the edge is negative. Beneath: payoff = average win ÷ average loss (why a 40 % win rate can still pay), pf = gross profit ÷ gross loss (above 1.0 the sleeves grew)." },
  vsSpy: { term: "vs holding SPY", text: "The account's return minus what simply buying and holding SPY would have returned over the same days, on price alone: dividends are left out because the paper account is credited none. The benchmark the whole project is measured against." },
  scoreboard: { term: "Daily vs SPY", text: "Each trading day: the account's equity change against SPY's price change (no dividends, like the paper account) over the same close-to-close span, the difference in points, and both compounded since the first day. Equity is the daily after-close snapshot; a day the snapshot missed folds into the next row, marked." },
  // between sessions the tile is the last session's, named (2026-11-06, server/todayTile.ts)
  today: { term: "Today · vs SPY", text: "One session's move vs SPY. In a session, today: equity now vs Alpaca's prior close (last_equity), $ and %, beside SPY's bar vs its prior close (price only); the headline is the gap in points. Between sessions it names the last one (Thu Nov 5 session): its prior close to its own close, or to equity now until Alpaca rolls last_equity." },
  equityPeak: { term: "Equity · from peak", text: `Account value now, and how far it sits below its highest value. The kill line (−20 % from the peak) is ${KILL_LINE_GLOSS}.` },
  session: { term: "Session", text: "Whether regular trading hours (09:30–16:00 ET) are open, and the time until the next open or close. Early closes come from the NYSE calendar." },
  now: { term: "Now", text: "What the bot is doing this second, read from its own audit log: idle between scans, scanning, deciding on candidates, an order working, exiting, halted, or closed for the day." },
  hold: { term: "Hold the layout", text: "Freezes the page's self-arranging layout so nothing moves while you read. The bot keeps running; every number keeps updating." },
  // ── stages ──
  stages: { term: "Stages", text: "Each line is one part of the loop with its last ten expected ticks: a green box is a tick that happened on time, an outlined box a missing one, red a dead one. A quiet stage is a problem, not calm." },
  // ── issues ──
  issues: { term: "Issues", text: "Standing problems, each as condition → consequence → what clears it, with an action where one exists. Red is halting, orange is degrading, grey is informational." },
  // ── decision chain ──
  decision: { term: "Decision", text: "The chain for the top candidate of the last scan: what triggered it, the features it was scored on, the rules that fired or vetoed it, the conviction numbers, and what Node did with it." },
  conviction: { term: "Conviction", text: "The rules-only score, the ML model's score, and the blended final score the admission threshold is applied to. A dash means that part was not recorded for this candidate." },
  // ── model ──
  model: { term: "Model", text: "The LightGBM model that scores candidates: how old it is against the 24 h fallback line, which regime it is routed to, how its score blends with the rules, and what it actually saw at serve time." },
  modelAge: { term: "Age · fallback at", text: "Hours since the model was last trained. Past the fallback line the bot scores on rules alone until a retrain succeeds." },
  regimeRoute: { term: "Regime route", text: "Which regime's model served the recent candidates. Routing uses the VXX ratio and SPY's moving averages; a pinned ratio means one regime is unreachable." },
  vxxRatio: { term: "VXX ratio", text: "Short-term volatility ETN price relative to its average, the regime classifier's fear gauge. 'Pinned' means the feed returned the same value every time — the classifier is blind on that axis." },
  blend: { term: "Blend r / ml", text: "The weights the final score gives the rules score and the ML score. They are set from the recent win rate, so a losing streak pushes weight back to the rules." },
  gated: { term: "Gated · blended", text: "Of the recently traced candidates, how many the ML gate rejected outright and how many were scored by the blend." },
  calibrator: { term: "Calibrator", text: "The probability calibration fitted after training so the model's win probabilities match observed rates. 'Paired' means it is from the same training run as the model." },
  whatItSaw: { term: "What it saw", text: "The features the model was scored on in the last candidates, with how many times each moved. 'Constant' means the input never changed at serve time — the model is reading a dead sensor." },
  retrainChecks: { term: "Retrain checks", text: "The hourly tier-3 ledger: each line is one check and what it did (retrained, skipped because the model is fresh, or failed)." },
  winProbability: { term: "Win probability", text: "The 10th, 50th and 90th percentile of the model's predicted win probability over the recent candidates. A narrow band means the model barely discriminates." },
  scoreQuintile: { term: "Score → forward win rate", text: "Recent candidates split into five score buckets, against how often the price was higher 5 and 20 days later. A useful score rises left to right; a flat or falling chart means the score does not predict." },
  byDecision: { term: "By decision", text: "Forward win rate of the candidates the bot took versus the ones it rejected, by rejection reason. If the rejected ones did better, the filter is costing money." },
  // ── chart ──
  chart: { term: "Chart", text: "Our own candle chart of the selected position (an option leg shows its underlying) with every exit level the engine actually uses drawn on it. Drag to pan, wheel or pinch to zoom, double-click to fit." },
  entry: { term: "Entry", text: "The average price the position was filled at. The small triangle marks the bar it was entered on." },
  stop: { term: "Stop", text: "Where the engine closes the position. In phase 1 it sits 2×ATR below entry (a −20 % hard floor catches gaps); after the first ⅓ is sold it is raised to breakeven and then trails the trade's best profit, later its highest price. 'Waits' means a grace period (first 60 min, then to day 4) holds it off unless the loss reaches −10 % / −15 %." },
  trailingStop: { term: "Trailing stop", text: "After the trade reaches +1R the stop no longer sits below entry: it follows the trade's best profit (phases 2–3) or its highest price (phase 4) at a distance set by the ATR and the regime." },
  hardFloor: { term: "Hard floor", text: "An absolute −20 % stop in phase 1 for gap-downs the normal stop would miss. Drawn faint and allowed to sit off-screen." },
  scaleOut: { term: "Sell ⅓ (scale-out)", text: "The engine takes profit in thirds: one third at +1R, one third at the regime's second threshold (3R in a bull tape, 1.5R in a bear tape, 2R otherwise), and lets the trailing stop handle the rest." },
  target: { term: "Target", text: "A fixed take-profit, max(4 %, min(3×ATR, 15 %)) above entry, that only fires after both thirds have been sold. Until then it is drawn faint." },
  rMultiple: { term: "R", text: "Profit in units of the initial risk: 1R is the distance from entry to the first stop (2×ATR). The phases, the scale-outs and the breakeven stop are all defined in R." },
  atr: { term: "ATR", text: "Average true range as a percent of price — the stock's typical daily swing. Stop widths and targets are multiples of it, so volatile names get wider stops." },
  phase: { term: "Phase", text: "Where the trade is in its exit ladder: 1 until the first +1R, 2 after it, 3 after +2R, 4 after +3R. Each phase tightens how the stop trails." },
  timeStop: { term: "Time stop", text: "A position that is still within ±2 % after 7 days is closed to free the capital. The engine hard-codes 7 days; the config's 10-day value is not read on that path." },
  strike: { term: "Strike", text: "The option's strike price, drawn on the underlying's chart. For a sold put the trade keeps its whole premium if the stock finishes above it at expiry." },
  breakevenAtExpiry: { term: "Breakeven at expiry", text: "The underlying price at which the option leg itself is flat on expiry day: strike minus the premium for a put, strike plus the premium for a call." },
  managerTarget: { term: "Manager target", text: "The options manager closes a sold option once it has captured 50 % of the premium, rather than holding to expiry." },
  priceAtScan: { term: "Price at scan", text: "The price the scanner saw when it scored this ticker." },
  // ── watch ──
  watch: { term: "Watch", text: "What the last tier-2 scan looked at: its top candidates with their scores and factor breakdown, the trades it handed to Node, and what Node refused and why. Click a ticker to chart it." },
  admission: { term: "Admission threshold", text: "The minimum blended score Node accepts a scanner trade at; a candidate below it is skipped with an audit row. The scanner itself uses a regime-dependent minimum (63–75) one step earlier." },
  score: { term: "Score", text: "The candidate's blended score (0–100): rules and ML combined per the blend weights. The bar shows it against the admission threshold tick." },
  factors: { term: "Factors", text: "The five rule families behind the score — momentum, mean reversion, volatility-risk premium, squeeze, volume — each 0–100 as the scanner scored it." },
  outcome: { term: "Outcome", text: "What happened after the scan: queued for the open, handed to Node, refused by a Node gate (score, size, heat, correlation, halt, duplicate, options slots) or by sizing; 'no decision row' means Node wrote nothing for it yet. 'Sector clash' on the action: another candidate or position shares its sector, so the correlation cap may refuse it." },
  refused: { term: "Refused · why", text: "The audit rows where Node or the sizer declined a scanner trade, newest first, with the reason the code gave." },
  decided: { term: "Decided", text: "The trades the scan itself produced and handed to Node for admission, sizing and placement." },
  // ── book ──
  book: { term: "Book", text: "Every open position as a tape row — quantity with its unit, last price in dollars, today's move, and P&L since entry in percent and dollars; above it, where the account's profit and loss has come from, by sleeve." },
  sleeve: { term: "Sleeve", text: "One of three parts of the strategy: the options sleeve (sold premium and spreads), the ETF floor (the passive core and its legs), and single stocks. P&L is attributed to each from the broker's fill ledger." },
  realized: { term: "Realized", text: "Profit or loss on closed lots since the window start, from the broker's fills on an average-cost basis. Open positions count separately as unrealized." },
  worstOptions: { term: "Worst options", text: "Realized P&L by underlying for the options sleeve, worst first, so the names that cost the most are visible." },
  optionsByHold: { term: "Options by hold", text: "Realized options P&L grouped by how long the closed lot was held. Most of the sleeve's loss sits in lots closed within a day." },
  // ── expectancy by sleeve (B2, 2026-11-03; the first sentence is the human's own wording, verbatim) ──
  expectancyBySleeve: { term: "Expectancy by sleeve", text: "average % return per fully closed position, from Alpaca paper fills. One row per sleeve since the window start — options, stocks, their total (the Expectancy tile's headline), the ETF floor, the VRP leg (SVXY) and every lot together. A sold option's cost is its credit; a red dot marks a sleeve below zero over 30 or more lots." },
  payoffRatio: { term: "Payoff", text: "Average winning lot ÷ average losing lot, both as a percent of the lot's cost. Above 1 the wins are bigger than the losses, which is how a low win rate can still pay; below 1 the sleeve has to win more often than it loses." },
  breakevenWinRate: { term: "Breakeven win rate", text: "avg loss ÷ (avg win + avg loss), both as positive percents of cost: the win rate at which the sleeve's expectancy is exactly zero. A win rate below it loses on average — the win column turns red under this number and green above it." },
  taxView: { term: "If taxed", text: "What the same closed trades would owe in a taxable account, estimated at the rates shown (federal 24 % short-term / 15 % long-term by default, no state) — the paper account pays nothing. Terms come from the sleeve ledger's average-cost lot walk; the $3,000 ordinary-income offset is not modelled." },
  holdingTerm: { term: "Short · long term", text: "Short-term: the lot was held a year or less, so the gain is taxed as ordinary income. Long-term: more than a year, at the lower capital-gains rate. Lots older than the fetched history read as unknown and are taxed as short-term, the conservative choice." },
  washSale: { term: "Wash sales", text: "A loss is disallowed for now when the same symbol is re-entered on the same side within 30 days after the close, in proportion to the replacement quantity; it defers into the new lot's basis. Counted here for exact-symbol re-entries after the loss only — a lower bound." },
  afterTax: { term: "After tax", text: "Realized P&L in the window minus the estimated tax after netting short against long (a remaining net loss carries forward). Buy-and-hold SPY defers its tax until sold, so its drag is smaller — the comparison the account must win." },
  otherCash: { term: "Other cash · unexplained", text: "Cash flows the fill ledger cannot see (dividends, interest, fees, journals, option events), and what no ledger accounts for when the equity change is reconciled against realized + open + other cash." },
  // ── orders / timeline ──
  orders: { term: "Orders", text: "Orders working at the broker right now, with their status and age. 'Cancelled by deploy' counts open orders a restart cancelled today." },
  timeline: { term: "Today", text: "Every audit event of the ET session on a time axis: scans, trades, exits, halts, emails. Hatched = the kill switch was latched. Drag to replay the page at an earlier moment." },
  replay: { term: "Replay", text: "The page as it was at the chosen time, rebuilt from the audit rows up to then. Live data keeps flowing underneath; move the slider to the end to return to now." },
  afterTaxVsSpy: { term: "After tax vs SPY", text: "The same comparison if both were sold at the window's end: the account's return minus its estimated tax (closed trades at the ledger's terms, open positions as short-term) against SPY's price return minus one sale's tax at its holding term — SPY defers until sold, so its drag is smaller. An estimate; the paper account pays nothing." },
  dayChange: { term: "Day", text: "Today's move in the position's own price: last against the previous close, in percent — the tape's number, not the position's result. The P&L column is the result since entry." },
  bookPnl: { term: "P&L since entry", text: "What the position has made or lost since it was opened — as a percent of its cost and in dollars, unrealized until it closes. An option's percent is measured against the premium, so a cheap contract swings wide." },
  bookOptionExit: { term: "Option rows", text: "Line 3: bought or sold, call or put, days to expiry, the setup it was opened as (or 'not recorded'), single leg or legs the manager closes as one. Line 4: the exits, always the whole lot: 21 DTE, 5 DTE force, a sold leg's 50 % target and 2× credit stop, a bought leg's −50 % stop (no profit target), and the −75 % / −150 % floor." },
  bookStockExit: { term: "Stock rows", text: "Line 3: the engine's exit ladder for this lot — the stop and what it trails (entry, peak P&L, the high, breakeven), the phase-1 hard floor, the two thirds it sells at +1R and at the regime's 2R, the target it takes once both thirds are out, and the day-7 time stop. 'Not recorded yet' until the first manage tick writes the stop state." },
  lookup: { term: "Look-up", text: "Type a stock symbol (or an option's OCC code — looked up by its underlying) to open its dossier: market, sector, the stock's calibrated price ranges, options, news and a trade plan — with the bot's own look-up of the name through the scan's path." },
  lookupVerdict: { term: "Verdict", text: "What the scan would do with this name now: would place it, or the first gate it stops at — universe, regime, cooldown, admission, sector room, sizing. Undecided means a gate could not run (the card says which) and nothing is inferred in its place." },
  lookupGates: { term: "Gates", text: "The scan's checks in the scan's order, each with its reason: in the universe (price, volume, symbol pattern) · regime (no new longs in a bear tape) · stop cooldown · admission (score vs MIN_SCORE) · sector room · sizing (the Kelly refusal). After the first failure the rest are skipped, as the scan skips them." },
  lookupFactors: { term: "Factors", text: "The signed contributions behind the score — momentum, mean reversion, VRP, squeeze, volume and the rest — largest first. A name down big on bad news usually scores on mean reversion and sentiment; this is where you see that." },
  lookupChain: { term: "Chain", text: "The options scanner's own fetch for this underlying: contracts in its 7–60 day window, the at-the-money implied vol, the IV rank, and the put and call its delta finder picks at 0.25 — plus the setup the scanner itself would choose. Bid prices and premium as a percent of the stock price." },
  // ── overview (2026-11-06, human refocus spec §8) ──
  viewMode: { term: "Overview · Cockpit", text: "Two views of the same bot. Overview: how the account is doing against SPY after tax, why money is being lost, and a plain-English card for every pick. Cockpit: every engine panel — stages, the scan, the model, the candle chart, the day's timeline." },
  perfChart: { term: "Portfolio vs SPY after tax", text: "Both lines start at 0 on the first day and show the return if everything had been sold at that day's close, after an estimated tax: SPY's whole gain taxed once at its holding term, the portfolio's closed trades taxed by the ledger. Dashed: the same lines before tax." },
  soldNow: { term: "Sold now", text: "The vs-SPY tile's after-tax figure: both sold today, with the tax selling every open position would add. The line leaves that out for past days because their open P&L was not recorded, so the ring can sit off the line's end." },
  drawdown: { term: "Drawdown from peak", text: "How far each one sits below its own highest value so far, before tax. Losing less in down markets is the edge the bot is meant to have, so the portfolio's dips should be shallower than SPY's." },
  picks: { term: "Picks", text: "One card per position (the legs of one options strategy are one card): why the bot bought it, what it passed over and why, the stages of the decision, its size, today's P&L and when the engine will sell. From the decision journal; a position the journal never saw says so." },
  pickWhat: { term: "What it is", text: "The instrument in plain words from the position's own fields: shares bought and when, or the option sold or bought, its strike, expiry, premium and the price where it starts to lose money at expiry." },
  pickPassedOver: { term: "Passed over", text: "The alternatives the decision rejected and the reason the journal gave for each — another instrument (shares instead of an option, or the other way round), a candidate that lost on tax after short-term rates, or one the sizing refused." },
  pickStages: { term: "How it decided", text: "The decision pipeline's stages as the journal recorded them for this trade: market regime, calendar, institutional flows, sector, stock selection, options positioning and the tax check, each in one plain line." },
  why: { term: "Why we're losing money", text: "The P&L over the window split four ways — by the signal that picked the trade, the sector, the instrument and the timing — worst first, each line with the journal's reason. Without the journal: the fill ledger's realized P&L by sleeve, options underlying and hold time." },
  regimeBudget: { term: "Market regime · risk budget", text: "The first stage of every decision: VIX level and term structure, the broad trend and breadth set how much risk the bot may take, from full to minimal. Each input says how much it cut and why; the floor and the sizer read the budget when it is switched on." },
  flows: { term: "Money flows ahead", text: "The institutional flow model: expected net buying or selling by big funds over the next days — pension and balanced-fund rebalancing, fund tax-loss selling, index reconstitutions — for stocks as a whole and by sector. A model estimate, not a measured flow." },
  taxFilter: { term: "Tax filter", text: "Before a trade, its expected return after short-term tax is set against simply holding SPY; a trade whose edge does not survive is skipped. Also: losses that could be harvested (with the wash-sale date) and winners close to the lower long-term rate." },
  signalsWork: { term: "Which signals work", text: "The audit of each signal and model input: whether it adds return out of sample, looks like noise, or hurts — with the measure and the sample size it was judged on. Noise is a candidate to cut; nothing here is believed without out-of-sample proof." },
} as const satisfies Record<string, GlossaryEntry>;

export type GlossaryKey = keyof typeof GLOSSARY;
export const GLOSSARY_KEYS = Object.keys(GLOSSARY) as GlossaryKey[];
export function glossary(k: GlossaryKey): GlossaryEntry { return GLOSSARY[k]; }
