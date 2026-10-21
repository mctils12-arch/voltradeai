// Bot page — the glossary behind every "?" (client/src/bot/help.tsx). One
// plain line per term, written from the code that produces the number, not
// from textbook definitions: when a mechanism changes, its line changes with
// it (glossary.test.ts pins that every key the panels use exists here).
// Pure data; no DOM.

export interface GlossaryEntry { term: string; text: string }

export const GLOSSARY = {
  // ── KPI strip ──
  expectancy: { term: "Expectancy per trade", text: "What an average trade has made, net: win rate × average win − loss rate × average loss, in percent of the position. Negative means the strategy loses money on average, even if some trades win big." },
  winRate: { term: "Win rate · breakeven", text: "Share of closed trades that made money, beside the win rate the strategy NEEDS to break even given how big its wins and losses are; below that number the edge is negative. Beneath: payoff = average win ÷ average loss (why a 40 % win rate can still pay), pf = gross profit ÷ gross loss (above 1.0 the account grew)." },
  vsSpy: { term: "vs holding SPY", text: "The account's return minus what simply buying and holding SPY (dividends included) would have returned over the same days. The benchmark the whole project is measured against." },
  equityPeak: { term: "Equity · from peak", text: "Account value now, and how far it sits below its highest value. The kill line is the drawdown at which the bot stops trading automatically (−10 % from the peak)." },
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
  book: { term: "Book", text: "Every open position with quantity, last price and open P&L in percent; above it, where the account's profit and loss has come from, by sleeve." },
  sleeve: { term: "Sleeve", text: "One of three parts of the strategy: the options sleeve (sold premium and spreads), the ETF floor (the passive core and its legs), and single stocks. P&L is attributed to each from the broker's fill ledger." },
  realized: { term: "Realized", text: "Profit or loss on closed lots since the window start, from the broker's fills on an average-cost basis. Open positions count separately as unrealized." },
  worstOptions: { term: "Worst options", text: "Realized P&L by underlying for the options sleeve, worst first, so the names that cost the most are visible." },
  optionsByHold: { term: "Options by hold", text: "Realized options P&L grouped by how long the closed lot was held. Most of the sleeve's loss sits in lots closed within a day." },
  otherCash: { term: "Other cash · unexplained", text: "Cash flows the fill ledger cannot see (dividends, interest, fees, journals, option events), and what no ledger accounts for when the equity change is reconciled against realized + open + other cash." },
  // ── orders / timeline ──
  orders: { term: "Orders", text: "Orders working at the broker right now, with their status and age. 'Cancelled by deploy' counts open orders a restart cancelled today." },
  timeline: { term: "Today", text: "Every audit event of the ET session on a time axis: scans, trades, exits, halts, emails. Hatched = the kill switch was latched. Drag to replay the page at an earlier moment." },
  replay: { term: "Replay", text: "The page as it was at the chosen time, rebuilt from the audit rows up to then. Live data keeps flowing underneath; move the slider to the end to return to now." },
} as const satisfies Record<string, GlossaryEntry>;

export type GlossaryKey = keyof typeof GLOSSARY;
export const GLOSSARY_KEYS = Object.keys(GLOSSARY) as GlossaryKey[];
export function glossary(k: GlossaryKey): GlossaryEntry { return GLOSSARY[k]; }
