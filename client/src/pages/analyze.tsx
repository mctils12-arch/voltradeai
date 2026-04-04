import { useState, useRef, useEffect, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import {
  Search, TrendingUp, TrendingDown, Minus, ChevronUp, ChevronDown,
  Activity, BarChart2, Zap, Moon, Sun, RefreshCw, Target, Volume2,
  DollarSign, AlertTriangle, CheckCircle, XCircle, Info, ArrowUp, ArrowDown,
  Building2, Users, Calendar, Layers, PieChart, Percent, Flame, Radio,
  TrendingUp as TrendUp, Cpu, Eye, ShieldAlert, Newspaper
} from "lucide-react";

// Props interface for AnalyzePage
export interface AnalyzePageProps {
  initialTicker?: string;
}

// ────────────────────────────────────────────────────────────────────────────
// Types
// ────────────────────────────────────────────────────────────────────────────

interface VolSurface { expiry: string; days: number; atm_iv: number; vrp: number; }

interface Spread {
  type: string; expiry: string; days_to_expiry: number;
  long_strike: number; short_strike: number;
  net_debit: number; max_profit: number | null; max_loss: number | null;
  prob_profit: number; iv: number; delta: number; score: number; description: string;
}

interface ValuationSignal {
  factor: string; value: number; benchmark: number; verdict: string; detail: string;
}

interface Valuation {
  verdict: "Undervalued" | "Fairly Valued" | "Overvalued";
  color: "green" | "neutral" | "red";
  score: number;
  sector: string;
  signals: ValuationSignal[];
  factors_count: number;
}

interface VolMetrics {
  avg_volume_90d?: number; volume_ratio_5d?: number;
  hv10?: number; hv20?: number; hv30?: number; hv60?: number;
  hv_percentile?: number; term_structure_slope?: number;
  vol_of_vol?: number; atr_pct?: number;
  // New edge engine fields
  gex?: number; gex_regime?: string; iv_rank?: number;
  flow_ratio?: number; flow_signal?: string;
  rsi_14?: number; mfi?: number; insider_net?: number;
}

interface Sentiment {
  score: number;
  signal: string;
  bull_pct: number;
  reddit_buzz: number;
  trend_spike: number;
  contrarian_flag: string | null;
}

interface EarningsIntel {
  timing: string;
  days_to_earnings: number;
  beat_count: number;
  miss_count: number;
  beat_pct: number;
  avg_surprise_pct: number;
  avg_move_pct: number;
  max_drop_pct: number;
  iv_implied_move: number;
  historical_avg_move: number;
  options_edge: number;
  gap_and_hold: boolean;
  earnings_recommendation: string;
  caution_flag: string | null;
}

interface Recommendation {
  action: string;
  signal: string;
  reasoning: string;
  horizon: string;
  alt?: string;
  leveraged_bull?: string;
  leveraged_bear?: string;
}

interface EdgeFactors {
  put_call_ratio?: number;
  put_call_signal?: string;
  put_call_interp?: string;
  iv_crush_score?: number;
  iv_crush_pct?: number;
  iv_crush_rec?: string;
  squeeze_score?: number;
  squeeze_signal?: string;
  squeeze_desc?: string;
  gamma_pin?: number;
  gamma_pin_oi?: number;
  gamma_pin_dist_pct?: number;
  relative_strength?: number;
  relative_strength_signal?: string;
}

interface Fundamentals {
  // Live price
  open?: number; day_high?: number; day_low?: number;
  day_volume?: number; avg_vol_10d?: number; avg_vol_3m?: number;
  market_cap?: number; beta?: number;
  // Multiples
  trailing_pe?: number; forward_pe?: number; peg_ratio?: number;
  price_book?: number; price_sales?: number; ev_ebitda?: number;
  trailing_eps?: number; forward_eps?: number;
  // Income
  total_revenue?: number; ebitda?: number; fcf?: number; op_cashflow?: number;
  // Margins
  gross_margin?: number; op_margin?: number; net_margin?: number;
  // Returns / leverage
  roe?: number; roa?: number; debt_equity?: number;
  current_ratio?: number; quick_ratio?: number;
  // Growth
  rev_growth?: number; earn_growth?: number; earn_qtr_growth?: number;
  // Dividend
  div_yield?: number; div_rate?: number;
  // Analyst
  target_mean?: number; target_high?: number; target_low?: number;
  rec_key?: string; num_analysts?: number; upside_pct?: number;
  // Float / short
  shares_out?: number; float_shares?: number;
  short_ratio?: number; short_pct_float?: number;
  // Calendar
  earnings_date?: string;
}

interface AnalysisResult {
  ticker: string; company_name: string;
  spot: number; price_change: number; price_change_pct: number;
  atm_iv: number; rv10: number; rv20: number; rv30: number;
  vrp: number; vrp_regime: string; vrp_signal: string;
  high_52: number | null; low_52: number | null;
  vol_surface: VolSurface[]; top_spreads: Spread[]; spread_count: number;
  vol_metrics?: VolMetrics;
  skew?: number | null;
  valuation?: Valuation | null;
  fundamentals?: Fundamentals;
  sentiment?: Sentiment;
  earnings_intel?: EarningsIntel;
  recommendation?: Recommendation;
  edge_factors?: EdgeFactors;
  error?: string;
}

interface ScanResult {
  ticker: string; spot: number; price_change_pct: number;
  atm_iv: number; rv20: number; vrp: number;
  hv_percentile?: number; volume_ratio?: number;
  best_strategy?: string; top_spread_score: number; scan_score: number;
  sentiment_score?: number; sentiment_signal?: string;
  rec_action?: string; rec_signal?: string; freshness?: string;
}

interface ScanProgressResponse {
  progress: number;
  cached_count: number;
}

interface ScanResponse {
  results: ScanResult[];
  full_results?: ScanResult[];
  cached: boolean;
  age_seconds: number;
  progress?: number;
}

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

function fmt(n: number) { return n.toLocaleString(); }
function fmtVol(n: number) {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)}K`;
  return String(n);
}
function fmtBig(n: number) {
  if (n >= 1e12) return `$${(n / 1e12).toFixed(2)}T`;
  if (n >= 1e9)  return `$${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6)  return `$${(n / 1e6).toFixed(1)}M`;
  return `$${n.toLocaleString()}`;
}
function recLabel(key: string) {
  const map: Record<string, string> = {
    "strongBuy": "Strong Buy", "buy": "Buy", "hold": "Hold",
    "sell": "Sell", "strongSell": "Strong Sell",
  };
  return map[key] ?? key;
}
function recColor(key: string) {
  if (key === "strongBuy" || key === "buy") return "text-emerald-400 bg-emerald-400/10 border-emerald-400/30";
  if (key === "sell" || key === "strongSell") return "text-rose-400 bg-rose-400/10 border-rose-400/30";
  return "text-amber-400 bg-amber-400/10 border-amber-400/30";
}

// ────────────────────────────────────────────────────────────────────────────
// Metric glossary — every term explained for beginners
// ────────────────────────────────────────────────────────────────────────────

interface GlossaryEntry {
  term: string;
  what: string;         // one-sentence plain-English definition
  how: string;          // how to use / interpret it
  good: string;         // what "good" looks like
  bad: string;          // what "bad" looks like
  example?: string;     // optional concrete example
}

const GLOSSARY: Record<string, GlossaryEntry> = {
  "P/E (TTM)": {
    term: "Price-to-Earnings (Trailing)",
    what: "How much investors pay per $1 of profit the company earned over the last 12 months.",
    how: "Divide the stock price by annual earnings per share. Lower = cheaper relative to profits.",
    good: "Below the sector average — stock may be undervalued.",
    bad: "Far above sector average — investors are paying a premium; growth must justify it.",
    example: "P/E of 20 means you pay $20 for every $1 of annual profit.",
  },
  "P/E (Fwd)": {
    term: "Price-to-Earnings (Forward)",
    what: "Same as P/E but uses next year's estimated earnings instead of last year's actual earnings.",
    how: "Lower forward P/E than trailing P/E = analysts expect earnings to grow.",
    good: "Forward P/E below trailing P/E — earnings growth is expected.",
    bad: "Forward P/E higher than trailing — earnings may be declining.",
  },
  "PEG Ratio": {
    term: "Price/Earnings-to-Growth Ratio",
    what: "P/E ratio divided by the annual earnings growth rate. Adjusts valuation for growth speed.",
    how: "A PEG of 1.0 means the stock is fairly valued for its growth rate.",
    good: "Below 1.0 — stock may be undervalued relative to its growth.",
    bad: "Above 2.0 — paying a big premium even after accounting for growth.",
    example: "Stock with P/E 30 growing 30% has PEG = 1.0 (fair). Same P/E with 10% growth = PEG 3.0 (expensive).",
  },
  "P/Book": {
    term: "Price-to-Book Ratio",
    what: "Stock price divided by the company's net asset value (assets minus liabilities) per share.",
    how: "Compares market value to accounting value. Banks and industrials are often valued on this metric.",
    good: "Below 1.0 — trading below book value (rare, may signal deep value).",
    bad: "Very high P/B — market assigns huge value beyond physical assets (common for software companies).",
    example: "P/B of 3 means the market values the company at 3× what its balance sheet says it's worth.",
  },
  "P/Sales": {
    term: "Price-to-Sales Ratio",
    what: "Stock price divided by revenue per share. Useful for companies with no earnings yet.",
    how: "Lower = cheaper relative to revenue. Good for comparing early-stage companies.",
    good: "Below 2–3x for most industries.",
    bad: "Above 10x — very expensive relative to sales; needs massive margin expansion to justify.",
  },
  "EV/EBITDA": {
    term: "Enterprise Value / EBITDA",
    what: "Total company value (including debt) divided by operating profit before interest, taxes, and non-cash items.",
    how: "Compares companies across different capital structures. Lower = potentially cheaper.",
    good: "Below 10x for most industries.",
    bad: "Above 20x — expensive; common in high-growth tech.",
    example: "EBITDA removes accounting differences so you can compare a debt-heavy company fairly with a cash-rich one.",
  },
  "EPS (TTM)": {
    term: "Earnings Per Share (Trailing 12 Months)",
    what: "Total profit divided by number of shares — the actual profit each share earned last year.",
    how: "Higher and growing EPS = company is becoming more profitable per share.",
    good: "Positive and growing year-over-year.",
    bad: "Negative (company losing money) or declining.",
  },
  "EPS (Fwd)": {
    term: "Earnings Per Share (Forward Estimate)",
    what: "Analysts' consensus estimate of profit per share for the next 12 months.",
    how: "Compare to trailing EPS — if higher, earnings growth is expected.",
    good: "Higher than trailing EPS — analysts expect profit growth.",
    bad: "Lower than trailing EPS — analysts expect earnings to shrink.",
  },
  "Revenue (TTM)": {
    term: "Total Revenue (Trailing 12 Months)",
    what: "All money the company brought in from selling products or services over the last year.",
    how: "Growing revenue is the foundation of a healthy business.",
    good: "Consistently growing year-over-year.",
    bad: "Declining or stagnant revenue signals slowing demand.",
  },
  "EBITDA": {
    term: "Earnings Before Interest, Taxes, Depreciation & Amortization",
    what: "A measure of core operating profitability — profit before accounting adjustments and financing costs.",
    how: "Think of it as 'raw operating profit.' Useful for comparing companies across industries.",
    good: "Positive and growing, with healthy EBITDA margins.",
    bad: "Negative or shrinking — operations may not be profitable.",
  },
  "Free Cash Flow": {
    term: "Free Cash Flow (FCF)",
    what: "Cash left over after paying for operations and capital expenditures — the money the company can actually use.",
    how: "FCF funds dividends, buybacks, acquisitions, and debt paydown. It's often considered the truest measure of financial health.",
    good: "Positive and growing — company generates more cash than it spends.",
    bad: "Negative — company is burning cash and may need to raise money.",
    example: "A company with $10B revenue but negative FCF is less healthy than one with $1B revenue and strong positive FCF.",
  },
  "Op. Cash Flow": {
    term: "Operating Cash Flow",
    what: "Cash generated from the company's core business operations before capital expenditures.",
    how: "Should be positive and ideally growing. More reliable than net income since it's harder to manipulate.",
    good: "Positive and growing; tracks or exceeds net income.",
    bad: "Negative, or far below net income (suggests income is not converting to real cash).",
  },
  "Div. Yield": {
    term: "Dividend Yield",
    what: "Annual dividend payment as a percentage of the stock price.",
    how: "Shows the income return from holding the stock, independent of price appreciation.",
    good: "Stable and sustainable yield above the risk-free rate (currently ~5%).",
    bad: "Extremely high yield (>8%) can signal the dividend may be cut; no yield means no income.",
    example: "A $100 stock paying $3/year in dividends has a 3% yield.",
  },
  "Div. Rate": {
    term: "Annual Dividend Rate",
    what: "The total dollar amount paid per share in dividends over the next 12 months.",
    how: "Multiply by shares owned to estimate your annual dividend income.",
    good: "Stable or growing over time.",
    bad: "Declining dividend rate often precedes a cut.",
  },
  "Gross": {
    term: "Gross Margin",
    what: "Revenue minus the direct cost of making the product, as a percentage of revenue.",
    how: "Shows how much profit is left after production costs. Higher = more room for overhead and profit.",
    good: "Above 40% for most businesses; 70%+ for software.",
    bad: "Below 20% — thin margins, leaves little room for error.",
    example: "Software (70–90%), Retail (20–30%), Grocery (2–5%).",
  },
  "Operating": {
    term: "Operating Margin",
    what: "Profit from core operations as a percentage of revenue, after all operating expenses.",
    how: "Shows how efficiently the company runs its core business.",
    good: "Above 20% is strong for most sectors; 30%+ is exceptional.",
    bad: "Below 5% or negative — company struggles to cover its costs.",
  },
  "Net": {
    term: "Net Profit Margin",
    what: "Final profit (after all expenses including taxes and interest) as a percentage of revenue.",
    how: "The bottom line — what the company actually keeps for every dollar earned.",
    good: "Above 10–15% for most sectors.",
    bad: "Negative — company is losing money overall.",
    example: "Net margin of 25% means for every $100 in sales, $25 is profit.",
  },
  "Revenue": {
    term: "Revenue Growth (YoY)",
    what: "How much the company's sales grew compared to the same period last year.",
    how: "Consistent revenue growth is a sign of expanding market share or demand.",
    good: "Above 10–15% for established companies; 30%+ for high-growth.",
    bad: "Negative — company is shrinking.",
  },
  "Earnings": {
    term: "Earnings Growth (YoY)",
    what: "How much the company's profit grew compared to the same period last year.",
    how: "Faster earnings growth than revenue growth = improving efficiency.",
    good: "Consistently above 15%.",
    bad: "Negative or slower than revenue growth — margins compressing.",
  },
  "EPS (Qtr)": {
    term: "Quarterly EPS Growth (YoY)",
    what: "How much earnings per share changed vs. the same quarter last year.",
    how: "Quarterly beats/misses vs. estimates drive short-term stock moves.",
    good: "Positive and accelerating — business momentum is building.",
    bad: "Consecutive misses often precede selloffs.",
  },
  "Return on Equity": {
    term: "Return on Equity (ROE)",
    what: "How much profit the company generates for every dollar of shareholders' equity.",
    how: "Higher ROE = management is efficiently using shareholder capital. Compare within same sector.",
    good: "Above 15% is strong; above 20% is excellent.",
    bad: "Below 10% or negative — capital being used inefficiently.",
    example: "ROE of 30% means for every $1 shareholders own, the company generates $0.30 in annual profit.",
  },
  "Return on Assets": {
    term: "Return on Assets (ROA)",
    what: "Profit generated per dollar of total assets — measures how efficiently assets are deployed.",
    how: "Useful for comparing asset-heavy businesses like banks and manufacturers.",
    good: "Above 5% is solid for most industries; banks aim for ~1%.",
    bad: "Below 1% or negative.",
  },
  "Debt / Equity": {
    term: "Debt-to-Equity Ratio",
    what: "Total debt divided by shareholders' equity — shows how much the company relies on borrowed money.",
    how: "High D/E can amplify gains in good times but is risky during downturns.",
    good: "Below 50% for most sectors; lower is safer.",
    bad: "Above 200% — heavily leveraged, vulnerable to rising interest rates.",
    example: "D/E of 100% means debt equals equity; the company is half-financed by borrowing.",
  },
  "Current Ratio": {
    term: "Current Ratio",
    what: "Current assets divided by current liabilities — can the company pay its bills due within one year?",
    how: "A ratio above 1 means assets exceed near-term liabilities.",
    good: "Above 1.5 — comfortable liquidity cushion.",
    bad: "Below 1.0 — potential short-term cash crunch.",
  },
  "Quick Ratio": {
    term: "Quick Ratio (Acid Test)",
    what: "Like the current ratio but excludes inventory — a stricter measure of short-term liquidity.",
    how: "Inventory may be hard to sell quickly; this ratio shows if the company can survive without it.",
    good: "Above 1.0 — can cover short-term obligations without selling inventory.",
    bad: "Below 0.5 — may struggle with short-term cash needs.",
  },
  "Short % of Float": {
    term: "Short Interest as % of Float",
    what: "The percentage of tradeable shares currently sold short (borrowed and sold, betting price falls).",
    how: "High short interest can be bearish — OR can fuel a 'short squeeze' rally if the stock rises.",
    good: "Below 5% — minimal bearish bets.",
    bad: "Above 20% — significant bearish conviction, or potential squeeze target.",
    example: "GameStop had 140% short float before its famous 2021 squeeze.",
  },
  "Days to Cover": {
    term: "Short Ratio (Days to Cover)",
    what: "How many days of average trading volume it would take for all short sellers to buy back their shares.",
    how: "Higher days-to-cover = shorts are more trapped if the stock rises.",
    good: "Below 5 days — short interest is manageable.",
    bad: "Above 10 days — crowded short that could squeeze violently on good news.",
  },
  "Shares Out": {
    term: "Shares Outstanding",
    what: "Total number of shares that exist (including those held by insiders and institutions).",
    how: "Market cap = share price × shares outstanding. Shrinking share count (buybacks) = bullish.",
    good: "Declining over time — company is buying back shares.",
    bad: "Rapidly increasing — dilution, may hurt earnings per share.",
  },
  "Float": {
    term: "Public Float",
    what: "Shares available for public trading (excludes insider/restricted shares).",
    how: "Lower float = more volatile; a small number of buyers/sellers can move the price significantly.",
    good: "Large float = stable, liquid trading.",
    bad: "Very small float = easily manipulated, wide bid-ask spreads.",
  },
  // Vol / options terms
  "ATM Implied Vol": {
    term: "At-the-Money Implied Volatility",
    what: "The market's expectation of how much the stock will move over the next 30 days, expressed as an annualized %.",
    how: "Derived from options prices. High IV = expensive options. Sell vol when IV is elevated vs. history.",
    good: "Low relative to historical volatility — options are cheap, good to buy.",
    bad: "High relative to history — options are expensive, better to sell.",
    example: "IV of 40% means the market expects ~40% annualized moves ≈ ~2.3% daily.",
  },
  "Realized Vol (20d)": {
    term: "20-Day Realized (Historical) Volatility",
    what: "How much the stock actually moved over the past 20 trading days, annualized.",
    how: "The benchmark for implied vol. If IV >> RV, options are overpriced — consider selling.",
    good: "Lower than implied vol — suggests option sellers have an edge.",
    bad: "Higher than implied vol — stock moved more than options priced in.",
  },
  "Vol Risk Premium": {
    term: "Volatility Risk Premium (VRP)",
    what: "The difference between implied volatility and realized volatility (IV − RV).",
    how: "Core concept from Colin Bennett's book. Positive VRP = options are expensive vs. actual moves.",
    good: "Positive and high (>5%) — strong edge for selling options.",
    bad: "Negative — implied vol is underpricing actual moves; selling is risky.",
    example: "IV = 35%, RV = 20% → VRP = +15%. Options sellers have historically profited when VRP > 0.",
  },
  "Beta": {
    term: "Beta",
    what: "How much the stock moves relative to the overall market (S&P 500 = 1.0).",
    how: "Beta > 1 = amplified market moves. Beta < 1 = more stable than market. Negative = inverse.",
    good: "Low beta (~0.5) if you want stability; no 'good' beta universally — depends on goals.",
    bad: "Very high beta (>2.0) = very volatile, large swings on market moves.",
    example: "Beta of 1.5: if market drops 10%, stock typically drops 15%.",
  },
  "Market Cap": {
    term: "Market Capitalization",
    what: "Total value of all shares — what the entire company would cost to buy at today's price.",
    how: "Large-cap (>$10B) = more stable. Small-cap (<$2B) = higher growth potential, higher risk.",
    good: "Depends on goal: large-cap for stability, small-cap for growth.",
    bad: "Micro-cap (<$300M) = highly speculative, illiquid.",
    example: "Apple at ~$3.7T is the world's largest company by market cap.",
  },
  "HV Percentile (1yr)": {
    term: "Historical Volatility Percentile",
    what: "Where current 20-day realized vol sits relative to its own 1-year range.",
    how: "100th percentile = vol is at its highest in a year. Used to gauge if vol is cheap or expensive.",
    good: "Low percentile (below 30%) = vol is cheap, good time to buy options.",
    bad: "High percentile (above 70%) = vol is elevated, good time to sell options.",
  },
  "Volume Spike (5d/90d)": {
    term: "Volume Spike Ratio",
    what: "Average daily volume over the past 5 days divided by the 90-day average.",
    how: "A ratio above 2 means volume has doubled — often signals institutional activity or news.",
    good: "Moderate elevation on up days — buyers stepping in.",
    bad: "Extreme spike on down days — distribution or panic selling.",
    example: "Ratio of 3.0 means the stock traded 3× its normal daily volume this week.",
  },
  "Put-Call Skew": {
    term: "Put-Call Volatility Skew",
    what: "The difference between implied vol of out-of-money puts (90% strike) and out-of-money calls (110% strike).",
    how: "Positive skew = puts are more expensive than calls — market fears downside more.",
    good: "Moderate skew (2–5%) is normal. Low skew = market is complacent.",
    bad: "Extreme positive skew (>10%) = panic; crash insurance is very expensive.",
    example: "Skew of 8% means a 90% put costs 8% more IV than a 110% call.",
  },
  "Mean Target": {
    term: "Analyst Mean Price Target",
    what: "The average price target across all Wall Street analysts covering the stock.",
    how: "Compare to current price for implied upside/downside from professional estimates.",
    good: "Target significantly above current price — analysts expect it to rise.",
    bad: "Target below current price — analysts think the stock is overvalued.",
  },
  "Upside / Downside": {
    term: "Analyst Upside / Downside",
    what: "How far the mean analyst price target is above or below the current stock price.",
    how: "Purely based on consensus estimates. Useful as one input — not a guarantee.",
    good: "Positive upside >10% with multiple analysts covering.",
    bad: "Negative or near zero — analysts don't see much appreciation.",
  },
};

// ── Tooltip popover component ────────────────────────────────────────────────
// Uses position:fixed + getBoundingClientRect so the popover escapes every
// overflow:hidden / overflow:scroll parent and is never clipped.

function MetricTooltip({ metricKey }: { metricKey: string }) {
  const [open, setOpen] = useState(false);
  const [pos, setPos]   = useState<{ top: number; left: number } | null>(null);
  const wrapRef    = useRef<HTMLDivElement>(null);
  const btnRef     = useRef<HTMLButtonElement>(null);
  const popoverRef = useRef<HTMLDivElement>(null);
  const entry = GLOSSARY[metricKey];
  if (!entry) return null;

  // Compute popover position relative to viewport (fixed positioning)
  const openPopover = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (!open && btnRef.current) {
      const r       = btnRef.current.getBoundingClientRect();
      const popW    = 288; // matches CSS width
      const margin  = 8;
      let left      = r.left + r.width / 2 - popW / 2;
      // Clamp horizontally within viewport
      left = Math.max(margin, Math.min(left, window.innerWidth - popW - margin));
      // Default: show above the trigger; fall back to below if not enough room
      const popH    = 260; // estimated height
      const top     = r.top - popH - 8 > margin
        ? r.top - popH - 8
        : r.bottom + 8;
      setPos({ top, left });
    }
    setOpen(o => !o);
  };

  // Recompute position if popover height differs from estimate once rendered
  useEffect(() => {
    if (open && btnRef.current && popoverRef.current) {
      const r      = btnRef.current.getBoundingClientRect();
      const popW   = popoverRef.current.offsetWidth  || 288;
      const popH   = popoverRef.current.offsetHeight || 260;
      const margin = 8;
      let left     = r.left + r.width / 2 - popW / 2;
      left = Math.max(margin, Math.min(left, window.innerWidth - popW - margin));
      const top    = r.top - popH - 8 > margin
        ? r.top - popH - 8
        : r.bottom + 8;
      setPos({ top, left });
    }
  }, [open]);

  // Close on outside click or Escape
  useEffect(() => {
    if (!open) return;
    const handleClick = (e: MouseEvent) => {
      if (
        popoverRef.current && !popoverRef.current.contains(e.target as Node) &&
        wrapRef.current    && !wrapRef.current.contains(e.target as Node)
      ) setOpen(false);
    };
    const handleKey = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    document.addEventListener("mousedown", handleClick);
    document.addEventListener("keydown",   handleKey);
    return () => {
      document.removeEventListener("mousedown", handleClick);
      document.removeEventListener("keydown",   handleKey);
    };
  }, [open]);

  return (
    <div className="metric-tooltip-wrap" ref={wrapRef}>
      <button
        ref={btnRef}
        className="tooltip-trigger"
        onClick={openPopover}
        aria-label={`Explain ${entry.term}`}
        data-testid={`tooltip-btn-${metricKey.replace(/\s/g, '-')}`}
      >
        ?
      </button>
      {open && pos && (
        <div
          ref={popoverRef}
          className="tooltip-popover"
          role="dialog"
          aria-label={entry.term}
          style={{ position: "fixed", top: pos.top, left: pos.left }}
        >
          <button className="tooltip-close" onClick={() => setOpen(false)} aria-label="Close">✕</button>
          <div className="tooltip-term">{entry.term}</div>
          <p className="tooltip-what">{entry.what}</p>
          <div className="tooltip-rows">
            <div className="tooltip-row">
              <span className="tooltip-row-label">How to use</span>
              <span className="tooltip-row-body">{entry.how}</span>
            </div>
            <div className="tooltip-row tooltip-good">
              <span className="tooltip-row-label">✓ Good</span>
              <span className="tooltip-row-body">{entry.good}</span>
            </div>
            <div className="tooltip-row tooltip-bad">
              <span className="tooltip-row-label">✗ Watch out</span>
              <span className="tooltip-row-body">{entry.bad}</span>
            </div>
            {entry.example && (
              <div className="tooltip-row tooltip-example">
                <span className="tooltip-row-label">Example</span>
                <span className="tooltip-row-body">{entry.example}</span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Labeled metric row with optional tooltip ─────────────────────────────────

function FundRow({ label, tooltipKey, children }: {
  label: string;
  tooltipKey?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="fund-row">
      <span className="fund-row-label">
        {label}
        {tooltipKey && <MetricTooltip metricKey={tooltipKey} />}
      </span>
      <span>{children}</span>
    </div>
  );
}

function MargRow({ label, tooltipKey, val, good = 20 }: {
  label: string; tooltipKey?: string; val: number; good?: number;
}) {
  const color = val >= good ? "bg-emerald-500" : val >= good * 0.5 ? "bg-amber-500" : "bg-rose-500";
  return (
    <div className="marg-row">
      <span className="marg-label">
        {label}
        {tooltipKey && <MetricTooltip metricKey={tooltipKey} />}
      </span>
      <div className="marg-track">
        <div className={`marg-fill ${color}`} style={{ width: `${Math.min(Math.max(val, 0), 100)}%` }} />
      </div>
      <span className="marg-value">{val.toFixed(1)}%</span>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// Score bar
// ────────────────────────────────────────────────────────────────────────────

function ScoreBar({ score }: { score: number }) {
  const color = score >= 70 ? "bg-emerald-500" : score >= 45 ? "bg-amber-500" : "bg-rose-500";
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 rounded-full bg-white/10">
        <div className={`h-full rounded-full ${color} transition-all duration-700`} style={{ width: `${score}%` }} />
      </div>
      <span className="text-xs font-mono font-semibold w-8 text-right"
        style={{ color: score >= 70 ? '#10b981' : score >= 45 ? '#f59e0b' : '#f43f5e' }}>
        {score.toFixed(0)}
      </span>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// Metric card
// ────────────────────────────────────────────────────────────────────────────

function MetricCard({ label, value, sub, highlight }: {
  label: string; value: string; sub?: string; highlight?: 'up' | 'down' | 'neutral'
}) {
  const subColor = highlight === 'up' ? 'text-emerald-400' : highlight === 'down' ? 'text-rose-400' : 'text-slate-400';
  return (
    <div className="metric-card">
      <div className="text-xs text-slate-400 mb-1 tracking-wide uppercase">{label}</div>
      <div className="text-xl font-mono font-bold text-white">{value}</div>
      {sub && <div className={`text-xs font-mono mt-0.5 ${subColor}`}>{sub}</div>}
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// Spread card
// ────────────────────────────────────────────────────────────────────────────

function SpreadCard({ spread, rank }: { spread: Spread; rank: number }) {
  const isCredit = spread.net_debit < 0;
  const typeColor = spread.type.includes("Bull") ? "text-emerald-400 bg-emerald-400/10 border-emerald-400/20"
    : spread.type.includes("Bear") ? "text-rose-400 bg-rose-400/10 border-rose-400/20"
    : "text-violet-400 bg-violet-400/10 border-violet-400/20";

  return (
    <div className="spread-card" data-testid={`spread-card-${rank}`}>
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="rank-badge">#{rank}</span>
          <span className={`type-badge border ${typeColor}`}>{spread.type}</span>
        </div>
        <span className="text-xs text-slate-400 font-mono">{spread.days_to_expiry}d exp.</span>
      </div>
      <p className="text-xs text-slate-300 mb-3 leading-relaxed">{spread.description}</p>
      <div className="grid grid-cols-3 gap-2 mb-3">
        <div className="mini-metric">
          <div className="mini-label">Win %</div>
          <div className={`mini-value ${spread.prob_profit >= 60 ? 'text-emerald-400' : spread.prob_profit >= 40 ? 'text-amber-400' : 'text-rose-400'}`}>
            {spread.prob_profit.toFixed(1)}%
          </div>
        </div>
        <div className="mini-metric">
          <div className="mini-label">{isCredit ? "Credit" : "Debit"}</div>
          <div className="mini-value">${Math.abs(spread.net_debit).toFixed(2)}</div>
        </div>
        <div className="mini-metric">
          <div className="mini-label">Max Profit</div>
          <div className="mini-value text-emerald-400">
            {spread.max_profit !== null ? `$${spread.max_profit.toFixed(2)}` : "Unlim."}
          </div>
        </div>
      </div>
      <div className="mb-1">
        <div className="flex justify-between text-xs mb-1">
          <span className="text-slate-400">Algorithm Score</span>
        </div>
        <ScoreBar score={spread.score} />
      </div>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// Vol surface chart
// ────────────────────────────────────────────────────────────────────────────

function VolSurfaceChart({ data }: { data: VolSurface[] }) {
  if (!data || data.length === 0) return null;
  const maxIv = Math.max(...data.map(d => d.atm_iv)) * 1.15;
  return (
    <div className="mt-4">
      <div style={{ fontSize: '11px', color: 'var(--text-secondary)', marginBottom: '8px', fontWeight: 500 }}>
        Term Structure — ATM IV by Expiry
      </div>
      {/* Fixed-height chart area — bars grow from bottom up, no overflow */}
      <div style={{ display: 'flex', alignItems: 'flex-end', gap: '6px', height: '56px' }}>
        {data.map((d, i) => {
          const h = Math.max(6, (d.atm_iv / maxIv) * 56);
          const vrpColor = d.vrp > 3 ? '#10b981' : d.vrp < -2 ? '#f43f5e' : '#f59e0b';
          return (
            <div key={i} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'flex-end', height: '56px' }}
              title={`${d.expiry}: IV=${d.atm_iv}%, VRP=${d.vrp > 0 ? '+' : ''}${d.vrp}%`}>
              <div style={{ width: '100%', height: `${h}px`, backgroundColor: vrpColor, opacity: 0.85, borderRadius: '3px 3px 0 0' }} />
            </div>
          );
        })}
      </div>
      {/* Labels row BELOW the bars */}
      <div style={{ display: 'flex', gap: '6px', marginTop: '4px' }}>
        {data.map((d, i) => (
          <div key={i} style={{ flex: 1, textAlign: 'center' }}>
            <div style={{ fontSize: '9px', color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>{d.atm_iv.toFixed(1)}</div>
            <div style={{ fontSize: '8px', color: 'var(--text-tertiary)' }}>{d.days}d</div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// Valuation panel
// ────────────────────────────────────────────────────────────────────────────

function ValuationPanel({ valuation }: { valuation: Valuation }) {
  const verdictColor = valuation.color === "green"
    ? "text-emerald-400 bg-emerald-400/10 border-emerald-400/30"
    : valuation.color === "red"
    ? "text-rose-400 bg-rose-400/10 border-rose-400/30"
    : "text-amber-400 bg-amber-400/10 border-amber-400/30";

  const verdictIcon = valuation.color === "green"
    ? <CheckCircle size={16} />
    : valuation.color === "red"
    ? <XCircle size={16} />
    : <Info size={16} />;

  return (
    <div className="panel" data-testid="section-valuation">
      <div className="panel-title">
        <DollarSign size={14} />
        Fundamental Valuation
        <span className={`ml-auto flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold border ${verdictColor}`}>
          {verdictIcon}
          {valuation.verdict}
        </span>
      </div>

      {/* Overall score bar */}
      <div className="mb-4">
        <div className="flex justify-between text-xs text-slate-400 mb-1.5">
          <span>Valuation Score ({valuation.factors_count} factors)</span>
          <span className="font-mono font-semibold text-white">{valuation.score.toFixed(0)}/100</span>
        </div>
        <ScoreBar score={valuation.score} />
        <div className="flex justify-between text-xs mt-1">
          <span className="text-rose-400">Overvalued</span>
          <span className="text-amber-400">Fair</span>
          <span className="text-emerald-400">Undervalued</span>
        </div>
      </div>

      {/* Factor breakdown */}
      <div className="space-y-2">
        {valuation.signals.map((sig, i) => {
          const isGood = sig.verdict.includes("cheap") || sig.verdict.includes("underval");
          const isBad  = sig.verdict.includes("expensive") || sig.verdict.includes("overval");
          const dotColor = isGood ? "bg-emerald-400" : isBad ? "bg-rose-400" : "bg-amber-400";
          return (
            <div key={i} className="flex items-start gap-2.5 p-2 rounded bg-white/[0.03]">
              <div className={`w-1.5 h-1.5 rounded-full mt-1 flex-shrink-0 ${dotColor}`} />
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-semibold text-slate-200">{sig.factor}</span>
                  <span className={`text-xs px-1.5 py-0.5 rounded font-mono ${isGood ? 'text-emerald-400 bg-emerald-400/10' : isBad ? 'text-rose-400 bg-rose-400/10' : 'text-amber-400 bg-amber-400/10'}`}>
                    {sig.verdict}
                  </span>
                </div>
                <p className="text-xs text-slate-400 mt-0.5">{sig.detail}</p>
              </div>
            </div>
          );
        })}
      </div>
      <p className="text-xs text-slate-500 mt-3">
        Sector: {valuation.sector} · Based on P/E, PEG, P/B, P/S, analyst targets, earnings yield.
      </p>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// Volume & Bennett metrics panel
// ────────────────────────────────────────────────────────────────────────────

function VolMetricsPanel({ metrics, skew, atm_iv, rv20 }: {
  metrics: VolMetrics; skew?: number | null; atm_iv: number; rv20: number;
}) {
  const volRatio = metrics.volume_ratio_5d;
  const volRatioColor = volRatio && volRatio > 2 ? "text-amber-400"
    : volRatio && volRatio > 1.5 ? "text-blue-400" : "text-slate-300";

  const hvPct = metrics.hv_percentile;
  const hvPctColor = hvPct && hvPct > 75 ? "text-amber-400"
    : hvPct && hvPct > 50 ? "text-blue-400" : "text-emerald-400";

  const skewColor = skew !== null && skew !== undefined
    ? skew > 5 ? "text-rose-400" : skew > 2 ? "text-amber-400" : "text-emerald-400"
    : "text-slate-400";

  const tsSlope = metrics.term_structure_slope;
  const tsSlopeLabel = tsSlope === null || tsSlope === undefined ? "—"
    : tsSlope > 0.1 ? "Steep (normal)" : tsSlope < -0.1 ? "Inverted (backwardation)" : "Flat";
  const tsSlopeColor = tsSlope !== null && tsSlope !== undefined
    ? tsSlope > 0.1 ? "text-emerald-400" : tsSlope < -0.1 ? "text-rose-400" : "text-amber-400"
    : "text-slate-400";

  return (
    <div className="panel" data-testid="section-vol-metrics">
      <div className="panel-title">
        <Volume2 size={14} />
        Volume &amp; Volatility Metrics
        <span className="ml-auto text-xs text-slate-500">Bennett Framework</span>
      </div>

      <div className="grid grid-cols-2 gap-2 mb-3">
        {metrics.avg_volume_90d !== undefined && (
          <div className="mini-metric">
            <div className="mini-label">Avg Daily Vol (90d)</div>
            <div className="mini-value">{fmtVol(metrics.avg_volume_90d)}</div>
          </div>
        )}
        {volRatio !== undefined && (
          <div className="mini-metric">
            <div className="mini-label">Volume Spike (5d/90d)</div>
            <div className={`mini-value font-mono ${volRatioColor}`}>
              {volRatio.toFixed(2)}x
              {volRatio > 2 && <span className="text-amber-400 ml-1">↑</span>}
            </div>
          </div>
        )}
        {hvPct !== undefined && (
          <div className="mini-metric">
            <div className="mini-label">HV Percentile (1yr)</div>
            <div className={`mini-value ${hvPctColor}`}>{hvPct.toFixed(0)}th</div>
          </div>
        )}
        {metrics.vol_of_vol !== undefined && (
          <div className="mini-metric">
            <div className="mini-label">Vol-of-Vol</div>
            <div className="mini-value">{metrics.vol_of_vol.toFixed(1)}%</div>
          </div>
        )}
        {metrics.atr_pct !== undefined && (
          <div className="mini-metric">
            <div className="mini-label">ATR % (14d)</div>
            <div className="mini-value">{metrics.atr_pct.toFixed(2)}%</div>
          </div>
        )}
        {skew !== null && skew !== undefined && (
          <div className="mini-metric">
            <div className="mini-label">Put-Call Skew</div>
            <div className={`mini-value ${skewColor}`}>
              {skew > 0 ? '+' : ''}{skew.toFixed(2)}%
            </div>
          </div>
        )}
      </div>

      {/* HV Cone comparison */}
      {(metrics.hv10 || metrics.hv20 || metrics.hv30 || metrics.hv60) && (
        <div className="mt-3">
          <div className="text-xs text-slate-400 uppercase tracking-wide mb-2">Historical Volatility Cone</div>
          {[
            { label: "HV 10d", val: metrics.hv10 },
            { label: "HV 20d", val: metrics.hv20 },
            { label: "HV 30d", val: metrics.hv30 },
            { label: "HV 60d", val: metrics.hv60 },
            { label: "ATM IV", val: atm_iv, isIV: true },
          ].filter(x => x.val !== undefined).map(({ label, val, isIV }) => {
            const maxVal = Math.max(metrics.hv10 ?? 0, metrics.hv20 ?? 0, metrics.hv30 ?? 0, metrics.hv60 ?? 0, atm_iv, 1);
            const pct = ((val as number) / maxVal) * 100;
            return (
              <div key={label} className="cone-row">
                <span className="cone-label">{label}</span>
                <div className="cone-bar-track">
                  <div className={`cone-bar ${isIV ? "cone-bar-iv" : "cone-bar-rv"}`} style={{ width: `${pct}%` }} />
                </div>
                <span className="cone-value">{(val as number).toFixed(1)}%</span>
              </div>
            );
          })}
        </div>
      )}

      {/* Term structure slope */}
      {tsSlope !== null && tsSlope !== undefined && (
        <div className="mt-3 flex items-center gap-2">
          <span className="text-xs text-slate-400">Term Structure:</span>
          <span className={`text-xs font-semibold ${tsSlopeColor}`}>{tsSlopeLabel}</span>
          <span className="text-xs text-slate-500 ml-auto font-mono">{tsSlope.toFixed(3)}</span>
        </div>
      )}

      {/* Edge Signals */}
      {(metrics.gex !== undefined || metrics.iv_rank !== undefined || metrics.rsi_14 !== undefined ||
        metrics.mfi !== undefined || metrics.flow_signal || metrics.insider_net !== undefined) && (
        <div className="mt-3 pt-3 border-t border-white/[0.06]">
          <div className="text-xs text-slate-400 uppercase tracking-wide mb-3">Edge Signals</div>
          <div className="space-y-3">

            {metrics.gex !== undefined && (
              <div style={{ padding: '8px 10px', background: 'rgba(0, 20, 40, 0.5)', borderRadius: '8px' }}>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs font-semibold text-slate-200">GEX Regime</span>
                  <span className={`text-xs font-bold ${metrics.gex_regime === 'pinned' ? 'text-amber-400' : 'text-blue-400'}`}>
                    {metrics.gex_regime === 'pinned' ? '📌 Pinned' : '💥 Explosive'}
                    <span className="text-slate-500 ml-1 font-mono text-xs">{(metrics.gex > 0 ? '+' : '') + (metrics.gex / 1e6).toFixed(0)}M</span>
                  </span>
                </div>
                <p className="text-xs text-slate-500 leading-relaxed">
                  <b>Gamma Exposure (GEX)</b> — measures what market makers must do to stay hedged.
                  {metrics.gex_regime === 'pinned'
                    ? ' Positive GEX = makers sell options and buy/sell stock to keep price stable. Expect LOW volatility, stock stays in a range. Bad for buying options, good for selling them.'
                    : ' Negative GEX = makers amplify moves in the same direction. Expect HIGH volatility and strong trends. Good for buying options or directional trades.'}
                </p>
              </div>
            )}

            {metrics.iv_rank !== undefined && (
              <div style={{ padding: '8px 10px', background: 'rgba(0, 20, 40, 0.5)', borderRadius: '8px' }}>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs font-semibold text-slate-200">IV Rank (1-Year)</span>
                  <span className={`text-xs font-bold ${metrics.iv_rank > 75 ? 'text-rose-400' : metrics.iv_rank < 30 ? 'text-emerald-400' : 'text-amber-400'}`}>
                    {metrics.iv_rank.toFixed(0)}th percentile — {metrics.iv_rank > 75 ? '✦ Sell Options' : metrics.iv_rank < 30 ? '✦ Buy Options' : 'Neutral'}
                  </span>
                </div>
                <p className="text-xs text-slate-500 leading-relaxed">
                  <b>IV Rank</b> = Where today's implied volatility sits in its own 1-year history (0–100).
                  {metrics.iv_rank > 75
                    ? ' 75th+ percentile = IV is near its yearly HIGH. Options are expensive. Sell a call, sell a put, or sell an iron condor to collect inflated premium.'
                    : metrics.iv_rank < 30
                    ? ' Below 30th percentile = IV is near its yearly LOW. Options are cheap. Buy a call (bullish) or buy a put (bearish) for less cost.'
                    : ' Mid-range — no strong edge from IV level alone.'}
                </p>
              </div>
            )}

            {metrics.flow_signal && (
              <div style={{ padding: '8px 10px', background: 'rgba(0, 20, 40, 0.5)', borderRadius: '8px' }}>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs font-semibold text-slate-200">Options Flow</span>
                  <span className={`text-xs font-bold ${
                    metrics.flow_signal === 'Bullish Flow' ? 'text-emerald-400' :
                    metrics.flow_signal === 'Bearish Flow' ? 'text-rose-400' : 'text-slate-300'
                  }`}>
                    {metrics.flow_signal}
                    {metrics.flow_ratio !== undefined && <span className="text-slate-500 ml-1 font-mono">({metrics.flow_ratio.toFixed(2)}x calls/puts)</span>}
                  </span>
                </div>
                <p className="text-xs text-slate-500 leading-relaxed">
                  <b>Options Flow</b> = ratio of call volume vs put volume today.
                  {metrics.flow_signal === 'Bullish Flow'
                    ? ' More calls traded than puts — traders are betting on upside. Often a sign of institutional or smart money positioning bullish.'
                    : metrics.flow_signal === 'Bearish Flow'
                    ? ' More puts traded than calls — traders are hedging or betting on downside. Watch for a potential drop.'
                    : ' Balanced call/put activity — no strong directional signal from flow today.'}
                </p>
              </div>
            )}

            {metrics.rsi_14 !== undefined && (
              <div style={{ padding: '8px 10px', background: 'rgba(0, 20, 40, 0.5)', borderRadius: '8px' }}>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs font-semibold text-slate-200">RSI — 14 Day</span>
                  <span className={`text-xs font-bold font-mono ${
                    metrics.rsi_14 > 70 ? 'text-rose-400' : metrics.rsi_14 < 30 ? 'text-emerald-400' : 'text-slate-300'
                  }`}>
                    {metrics.rsi_14.toFixed(1)} — {metrics.rsi_14 > 70 ? 'Overbought ⚠️' : metrics.rsi_14 < 30 ? 'Oversold 🔥' : 'Neutral'}
                  </span>
                </div>
                <p className="text-xs text-slate-500 leading-relaxed">
                  <b>RSI (Relative Strength Index)</b> = measures momentum on a 0–100 scale.
                  {metrics.rsi_14 > 70
                    ? ' Above 70 = overbought. Stock has risen too fast and may pull back. Consider selling calls, buying puts, or waiting for a better entry to buy the stock.'
                    : metrics.rsi_14 < 30
                    ? ' Below 30 = oversold. Stock has dropped too fast and a bounce is likely. Good time to buy calls, sell puts, or buy the stock at a discount.'
                    : ' Between 30–70 = normal range, no extreme signal.'}
                </p>
              </div>
            )}

            {metrics.mfi !== undefined && (
              <div style={{ padding: '8px 10px', background: 'rgba(0, 20, 40, 0.5)', borderRadius: '8px' }}>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs font-semibold text-slate-200">Money Flow Index (MFI)</span>
                  <span className={`text-xs font-bold font-mono ${
                    metrics.mfi > 80 ? 'text-rose-400' : metrics.mfi < 20 ? 'text-emerald-400' : 'text-slate-300'
                  }`}>
                    {metrics.mfi.toFixed(1)} — {metrics.mfi > 80 ? 'Overbought ⚠️' : metrics.mfi < 20 ? 'Oversold 🔥' : 'Neutral'}
                  </span>
                </div>
                <p className="text-xs text-slate-500 leading-relaxed">
                  <b>MFI</b> = Like RSI but also factors in VOLUME — more reliable.
                  {metrics.mfi > 80
                    ? ' Above 80 = money is flowing OUT at high prices (distribution). Strong warning sign — insiders may be selling into strength.'
                    : metrics.mfi < 20
                    ? ' Below 20 = money is flowing IN at low prices (accumulation). Strong buy signal — institutions may be loading up.'
                    : ' 20–80 = normal range, no extreme signal.'}
                </p>
              </div>
            )}

            {metrics.insider_net !== undefined && (
              <div style={{ padding: '8px 10px', background: 'rgba(0, 20, 40, 0.5)', borderRadius: '8px' }}>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs font-semibold text-slate-200">Insider Net Activity (90 days)</span>
                  <span className={`text-xs font-bold font-mono ${metrics.insider_net > 0 ? 'text-emerald-400' : metrics.insider_net < 0 ? 'text-rose-400' : 'text-slate-400'}`}>
                    {metrics.insider_net > 0 ? '+' : ''}{fmtBig(metrics.insider_net)} — {metrics.insider_net > 0 ? 'Net Buying' : metrics.insider_net < 0 ? 'Net Selling' : 'Neutral'}
                  </span>
                </div>
                <p className="text-xs text-slate-500 leading-relaxed">
                  <b>Insider Activity</b> = Net dollars of stock bought minus sold by executives and board members in the last 90 days (SEC filings).
                  {metrics.insider_net > 0
                    ? ' Insiders are buying their own company stock — they believe it is undervalued. One of the most reliable bullish signals.'
                    : metrics.insider_net < 0
                    ? ' Insiders are net sellers — could be routine profit-taking or a warning sign. Check if multiple insiders are selling.'
                    : ' No significant insider buying or selling reported.'}
                </p>
              </div>
            )}

          </div>
        </div>
      )}

      <div className="mt-3 text-xs text-slate-500 leading-relaxed">
        Vol-of-vol measures daily return roughness. Put-call skew compares 90% put IV vs 110% call IV.
        High skew indicates demand for downside protection.
      </div>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// Live price bar (open / high / low / volume / mkt cap / beta)
// ────────────────────────────────────────────────────────────────────────────

function LivePriceBar({ f, spot, high52, low52 }: {
  f: Fundamentals; spot: number; high52: number | null; low52: number | null;
}) {
  // Day range progress bar
  const rangePct = (f.day_high && f.day_low && f.day_high > f.day_low)
    ? ((spot - f.day_low) / (f.day_high - f.day_low)) * 100
    : null;
  // 52-week range progress bar
  const range52Pct = (high52 && low52 && high52 > low52)
    ? ((spot - low52) / (high52 - low52)) * 100
    : null;

  return (
    <div className="live-price-bar" data-testid="section-live-price">
      {/* Row 1: quick stats */}
      <div className="live-row">
        {f.open !== undefined && (
          <div className="live-stat">
            <span className="live-label">Open</span>
            <span className="live-value">${f.open.toFixed(2)}</span>
          </div>
        )}
        {f.day_high !== undefined && (
          <div className="live-stat">
            <span className="live-label">Day High</span>
            <span className="live-value text-emerald-400">${f.day_high.toFixed(2)}</span>
          </div>
        )}
        {f.day_low !== undefined && (
          <div className="live-stat">
            <span className="live-label">Day Low</span>
            <span className="live-value text-rose-400">${f.day_low.toFixed(2)}</span>
          </div>
        )}
        {f.day_volume !== undefined && (
          <div className="live-stat">
            <span className="live-label">Volume</span>
            <span className="live-value">{fmtVol(f.day_volume)}</span>
          </div>
        )}
        {f.avg_vol_3m !== undefined && (
          <div className="live-stat">
            <span className="live-label">Avg Vol (3m)</span>
            <span className="live-value">{fmtVol(f.avg_vol_3m)}</span>
          </div>
        )}
        {f.market_cap !== undefined && (
          <div className="live-stat">
            <span className="live-label">Market Cap</span>
            <span className="live-value">{fmtBig(f.market_cap)}</span>
          </div>
        )}
        {f.beta !== undefined && (
          <div className="live-stat">
            <span className="live-label">Beta</span>
            <span className={`live-value ${f.beta > 1.5 ? 'text-amber-400' : ''}`}>{f.beta.toFixed(2)}</span>
          </div>
        )}
        {f.earnings_date && (
          <div className="live-stat">
            <span className="live-label">Next Earnings</span>
            <span className="live-value text-blue-400">{f.earnings_date}</span>
          </div>
        )}
      </div>

      {/* Day range bar */}
      {rangePct !== null && f.day_low !== undefined && f.day_high !== undefined && (
        <div className="range-bar-row">
          <div className="range-bar-inner">
            <span className="range-label">${f.day_low.toFixed(2)}</span>
            <div className="range-track">
              <div className="range-fill" style={{ width: `${Math.min(rangePct, 100)}%` }} />
              <div className="range-dot" style={{ left: `${Math.min(rangePct, 100)}%` }} />
            </div>
            <span className="range-label">${f.day_high.toFixed(2)}</span>
          </div>
          <span className="range-caption">Day Range</span>
        </div>
      )}

      {/* 52-week range bar */}
      {range52Pct !== null && low52 !== null && high52 !== null && (
        <div className="range-bar-row">
          <div className="range-bar-inner">
            <span className="range-label">${low52.toFixed(2)}</span>
            <div className="range-track">
              <div className="range-fill range-fill-52" style={{ width: `${Math.min(range52Pct, 100)}%` }} />
              <div className="range-dot" style={{ left: `${Math.min(range52Pct, 100)}%` }} />
            </div>
            <span className="range-label">${high52.toFixed(2)}</span>
          </div>
          <span className="range-caption">52-Week</span>
        </div>
      )}
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// Fundamentals panel (valuation multiples, margins, growth, analyst)
// ────────────────────────────────────────────────────────────────────────────

function FundamentalsPanel({ f }: { f: Fundamentals }) {
  const hasMultiples = f.trailing_pe || f.forward_pe || f.peg_ratio || f.price_book || f.price_sales || f.ev_ebitda;
  const hasMargins   = f.gross_margin !== undefined || f.op_margin !== undefined || f.net_margin !== undefined;
  const hasGrowth    = f.rev_growth !== undefined || f.earn_growth !== undefined;
  const hasHealth    = f.roe !== undefined || f.debt_equity !== undefined || f.current_ratio !== undefined;
  const hasAnalyst   = f.target_mean !== undefined || f.rec_key;
  const hasIncome    = f.total_revenue !== undefined || f.ebitda !== undefined || f.fcf !== undefined;

  function GrowthChip({ val, label, tk }: { val: number; label: string; tk: string }) {
    const color = val > 15 ? "text-emerald-400" : val > 0 ? "text-blue-400" : "text-rose-400";
    const arrow = val > 0 ? "↑" : "↓";
    return (
      <div className="growth-chip">
        <span className="mini-label growth-chip-label">
          {label}
          <MetricTooltip metricKey={tk} />
        </span>
        <span className={`growth-val ${color}`}>{arrow} {Math.abs(val).toFixed(1)}%</span>
      </div>
    );
  }

  return (
    <div className="fund-panel" data-testid="section-fundamentals">
      <div className="panel-title">
        <Building2 size={14} />
        Real-Time Fundamentals
        <span className="ml-auto text-xs text-slate-500">via Yahoo Finance · click ? to learn any term</span>
      </div>

      <div className="fund-grid">

        {/* Valuation multiples */}
        {hasMultiples && (
          <div className="fund-section">
            <div className="fund-section-title"><Layers size={11} /> Valuation Multiples</div>
            <div className="fund-rows">
              {f.trailing_pe  && <FundRow label="P/E (TTM)"  tooltipKey="P/E (TTM)"><span className="fund-val">{f.trailing_pe.toFixed(1)}x</span></FundRow>}
              {f.forward_pe   && <FundRow label="P/E (Fwd)"  tooltipKey="P/E (Fwd)"><span className="fund-val">{f.forward_pe.toFixed(1)}x</span></FundRow>}
              {f.peg_ratio    && <FundRow label="PEG Ratio"  tooltipKey="PEG Ratio"><span className={`fund-val ${f.peg_ratio < 1 ? 'text-emerald-400' : f.peg_ratio < 2 ? '' : 'text-rose-400'}`}>{f.peg_ratio.toFixed(2)}</span></FundRow>}
              {f.price_book   && <FundRow label="P/Book"     tooltipKey="P/Book"><span className="fund-val">{f.price_book.toFixed(2)}x</span></FundRow>}
              {f.price_sales  && <FundRow label="P/Sales"    tooltipKey="P/Sales"><span className="fund-val">{f.price_sales.toFixed(2)}x</span></FundRow>}
              {f.ev_ebitda    && <FundRow label="EV/EBITDA"  tooltipKey="EV/EBITDA"><span className="fund-val">{f.ev_ebitda.toFixed(1)}x</span></FundRow>}
              {f.trailing_eps && <FundRow label="EPS (TTM)"  tooltipKey="EPS (TTM)"><span className="fund-val">${f.trailing_eps.toFixed(2)}</span></FundRow>}
              {f.forward_eps  && <FundRow label="EPS (Fwd)"  tooltipKey="EPS (Fwd)"><span className="fund-val">${f.forward_eps.toFixed(2)}</span></FundRow>}
            </div>
          </div>
        )}

        {/* Income & cash flow */}
        {hasIncome && (
          <div className="fund-section">
            <div className="fund-section-title"><DollarSign size={11} /> Income &amp; Cash Flow</div>
            <div className="fund-rows">
              {f.total_revenue && <FundRow label="Revenue (TTM)" tooltipKey="Revenue (TTM)"><span className="fund-val">{fmtBig(f.total_revenue)}</span></FundRow>}
              {f.ebitda        && <FundRow label="EBITDA"         tooltipKey="EBITDA"><span className="fund-val">{fmtBig(f.ebitda)}</span></FundRow>}
              {f.fcf           && <FundRow label="Free Cash Flow" tooltipKey="Free Cash Flow"><span className={`fund-val ${f.fcf > 0 ? 'text-emerald-400' : 'text-rose-400'}`}>{fmtBig(f.fcf)}</span></FundRow>}
              {f.op_cashflow   && <FundRow label="Op. Cash Flow"  tooltipKey="Op. Cash Flow"><span className="fund-val">{fmtBig(f.op_cashflow)}</span></FundRow>}
              {f.div_yield !== undefined && <FundRow label="Div. Yield" tooltipKey="Div. Yield"><span className="fund-val text-blue-400">{f.div_yield.toFixed(2)}%</span></FundRow>}
              {f.div_rate      && <FundRow label="Div. Rate"      tooltipKey="Div. Rate"><span className="fund-val">${f.div_rate.toFixed(2)}</span></FundRow>}
            </div>
          </div>
        )}

        {/* Margins */}
        {hasMargins && (
          <div className="fund-section">
            <div className="fund-section-title"><PieChart size={11} /> Margins</div>
            {f.gross_margin !== undefined && <MargRow label="Gross"     tooltipKey="Gross"     val={f.gross_margin} good={40} />}
            {f.op_margin    !== undefined && <MargRow label="Operating" tooltipKey="Operating" val={f.op_margin}    good={20} />}
            {f.net_margin   !== undefined && <MargRow label="Net"       tooltipKey="Net"       val={f.net_margin}   good={15} />}
          </div>
        )}

        {/* Growth */}
        {hasGrowth && (
          <div className="fund-section">
            <div className="fund-section-title"><TrendingUp size={11} /> Growth (YoY)</div>
            <div className="growth-chips">
              {f.rev_growth      !== undefined && <GrowthChip val={f.rev_growth}      label="Revenue" tk="Revenue" />}
              {f.earn_growth     !== undefined && <GrowthChip val={f.earn_growth}     label="Earnings" tk="Earnings" />}
              {f.earn_qtr_growth !== undefined && <GrowthChip val={f.earn_qtr_growth} label="EPS (Qtr)" tk="EPS (Qtr)" />}
            </div>
          </div>
        )}

        {/* Financial health */}
        {hasHealth && (
          <div className="fund-section">
            <div className="fund-section-title"><Activity size={11} /> Financial Health</div>
            <div className="fund-rows">
              {f.roe           !== undefined && <FundRow label="Return on Equity" tooltipKey="Return on Equity"><span className={`fund-val ${f.roe > 15 ? 'text-emerald-400' : f.roe > 0 ? '' : 'text-rose-400'}`}>{f.roe.toFixed(1)}%</span></FundRow>}
              {f.roa           !== undefined && <FundRow label="Return on Assets" tooltipKey="Return on Assets"><span className={`fund-val ${f.roa > 5 ? 'text-emerald-400' : ''}`}>{f.roa.toFixed(1)}%</span></FundRow>}
              {f.debt_equity   !== undefined && <FundRow label="Debt / Equity"    tooltipKey="Debt / Equity"><span className={`fund-val ${f.debt_equity > 100 ? 'text-rose-400' : f.debt_equity > 50 ? 'text-amber-400' : 'text-emerald-400'}`}>{f.debt_equity.toFixed(1)}%</span></FundRow>}
              {f.current_ratio !== undefined && <FundRow label="Current Ratio"    tooltipKey="Current Ratio"><span className={`fund-val ${f.current_ratio >= 1.5 ? 'text-emerald-400' : f.current_ratio >= 1 ? 'text-amber-400' : 'text-rose-400'}`}>{f.current_ratio.toFixed(2)}</span></FundRow>}
              {f.quick_ratio   !== undefined && <FundRow label="Quick Ratio"      tooltipKey="Quick Ratio"><span className="fund-val">{f.quick_ratio.toFixed(2)}</span></FundRow>}
            </div>
          </div>
        )}

        {/* Analyst consensus */}
        {hasAnalyst && (
          <div className="fund-section">
            <div className="fund-section-title"><Users size={11} /> Analyst Consensus</div>
            {f.rec_key && (
              <div className="rec-badge-row">
                <span className={`rec-badge border ${recColor(f.rec_key)}`}>{recLabel(f.rec_key)}</span>
                {f.num_analysts && <span className="text-xs text-slate-500">{f.num_analysts} analysts</span>}
              </div>
            )}
            {f.target_mean !== undefined && (
              <div className="target-row">
                <FundRow label="Mean Target"      tooltipKey="Mean Target"><span className="fund-val">${f.target_mean.toFixed(2)}</span></FundRow>
                {f.upside_pct !== undefined && (
                  <FundRow label="Upside / Downside" tooltipKey="Upside / Downside">
                    <span className={`fund-val font-semibold ${f.upside_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                      {f.upside_pct >= 0 ? '+' : ''}{f.upside_pct.toFixed(1)}%
                    </span>
                  </FundRow>
                )}
                {f.target_high !== undefined && (
                  <FundRow label="High / Low Target"><span className="fund-val">${f.target_high.toFixed(0)} / ${f.target_low?.toFixed(0) ?? '—'}</span></FundRow>
                )}
              </div>
            )}
            {f.target_low !== undefined && f.target_high !== undefined && f.target_mean !== undefined && (
              <div className="range-bar-row">
                <div className="range-bar-inner">
                  <span className="range-label">${f.target_low.toFixed(0)}</span>
                  <div className="range-track">
                    {(() => {
                      const range = f.target_high - (f.target_low ?? 0);
                      const meanPct = range > 0 ? ((f.target_mean - (f.target_low ?? 0)) / range) * 100 : 50;
                      return (
                        <>
                          <div className="range-fill" style={{ width: `${Math.min(meanPct, 100)}%`, background: 'var(--accent)' }} />
                          <div className="range-dot" style={{ left: `${Math.min(meanPct, 100)}%` }} />
                        </>
                      );
                    })()}
                  </div>
                  <span className="range-label">${f.target_high.toFixed(0)}</span>
                </div>
                <span className="range-caption">Price Targets</span>
              </div>
            )}
          </div>
        )}

        {/* Short interest */}
        {(f.short_pct_float !== undefined || f.short_ratio !== undefined) && (
          <div className="fund-section">
            <div className="fund-section-title"><AlertTriangle size={11} /> Short Interest</div>
            <div className="fund-rows">
              {f.short_pct_float !== undefined && (
                <FundRow label="Short % of Float" tooltipKey="Short % of Float">
                  <span className={`fund-val ${f.short_pct_float > 20 ? 'text-rose-400' : f.short_pct_float > 10 ? 'text-amber-400' : 'text-slate-300'}`}>
                    {f.short_pct_float.toFixed(1)}%
                  </span>
                </FundRow>
              )}
              {f.short_ratio   !== undefined && <FundRow label="Days to Cover" tooltipKey="Days to Cover"><span className={`fund-val ${f.short_ratio > 10 ? 'text-rose-400' : ''}`}>{f.short_ratio.toFixed(1)}d</span></FundRow>}
              {f.shares_out    !== undefined && <FundRow label="Shares Out"    tooltipKey="Shares Out"><span className="fund-val">{fmtVol(f.shares_out)}</span></FundRow>}
              {f.float_shares  !== undefined && <FundRow label="Float"         tooltipKey="Float"><span className="fund-val">{fmtVol(f.float_shares)}</span></FundRow>}
            </div>
          </div>
        )}

      </div>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// Recommendation card — the main action card
// ────────────────────────────────────────────────────────────────────────────

function RecommendationCard({ rec, ticker }: { rec: Recommendation; ticker: string }) {
  // Determine primary color theme based on signal
  const isPositive = rec.signal.includes('\ud83d\ude80') || rec.signal.includes('\ud83d\udcc8') || rec.signal.includes('\u2705') || rec.action.toLowerCase().includes('buy');
  const isNegative = rec.signal.includes('\u26a0\ufe0f') || rec.signal.includes('\u2620\ufe0f') || rec.signal.includes('\ud83d\udcc9') || rec.action.toLowerCase().includes('sell');
  const isNeutral  = rec.signal.includes('\u26aa') || rec.action.toLowerCase().includes('hold') || rec.action.toLowerCase().includes('neutral');

  const borderColor = isPositive ? 'border-emerald-400/30' : isNegative ? 'border-rose-400/30' : 'border-amber-400/30';
  const bgColor     = isPositive ? 'bg-emerald-400/5'      : isNegative ? 'bg-rose-400/5'      : 'bg-amber-400/5';
  const textColor   = isPositive ? 'text-emerald-400'      : isNegative ? 'text-rose-400'      : 'text-amber-400';

  return (
    <div className={`panel border ${borderColor} ${bgColor} mb-4`} data-testid="section-recommendation">
      <div className="flex items-start justify-between gap-3 flex-wrap">
        <div className="flex items-center gap-3">
          <span className="text-3xl" style={{ lineHeight: 1 }}>{rec.signal.split(' ')[0]}</span>
          <div>
            <div className={`text-xl font-bold ${textColor}`}>{rec.action}</div>
            <div className="text-xs text-slate-400 mt-0.5">Horizon: {rec.horizon}</div>
          </div>
        </div>
        {/* Leveraged ETF alternatives */}
        {(rec.leveraged_bull || rec.leveraged_bear) && (
          <div className="flex gap-2 flex-wrap">
            {rec.leveraged_bull && (
              <div className="flex flex-col items-center px-3 py-1.5 rounded-lg bg-emerald-400/10 border border-emerald-400/20">
                <span className="text-xs text-slate-400">Leveraged Bull</span>
                <span className="text-sm font-bold text-emerald-400 font-mono">{rec.leveraged_bull}</span>
              </div>
            )}
            {rec.leveraged_bear && (
              <div className="flex flex-col items-center px-3 py-1.5 rounded-lg bg-rose-400/10 border border-rose-400/20">
                <span className="text-xs text-slate-400">Leveraged Bear</span>
                <span className="text-sm font-bold text-rose-400 font-mono">{rec.leveraged_bear}</span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Reasoning */}
      <p className="text-sm text-slate-200 mt-3 leading-relaxed">{rec.reasoning}</p>

      {/* What exactly to do */}
      <div style={{ marginTop: '12px', padding: '10px 12px', background: 'rgba(0, 20, 40, 0.6)', borderRadius: '10px', borderLeft: '3px solid var(--accent)' }}>
        <div className="text-xs font-semibold text-slate-300 mb-1">What to do:</div>
        <div className="text-xs text-slate-400 leading-relaxed">
          {rec.action.includes('BUY STOCK') || rec.action.includes('Buy Stock') ? (
            <span>📈 <b>Buy the stock</b> — purchase shares outright. Profit when the stock goes up. More capital required but simpler than options. Set a stop-loss below recent support.</span>
          ) : rec.action.includes('SELL OPTIONS') || rec.action.includes('Sell Options') || rec.action.includes('SELL') ? (
            <span>💰 <b>Sell options to collect premium</b> — sell a call (bearish/neutral) or sell a put (bullish/neutral). You get paid upfront and keep the money if the stock doesn't move past your strike. Risk: stock can move against you.</span>
          ) : rec.action.includes('BUY OPTIONS') || rec.action.includes('Buy Options') || rec.action.includes('BUY CALL') || rec.action.includes('BUY PUT') ? (
            <span>🎯 <b>Buy options for leverage</b> — buy a call if bullish, buy a put if bearish. You pay a premium upfront for the right (not obligation) to buy/sell at a set price. Max loss = premium paid. Max gain = unlimited (calls) or large (puts).</span>
          ) : rec.action.includes('WAIT') || rec.action.includes('HOLD') ? (
            <span>⏳ <b>Wait for a better setup</b> — signals are mixed right now. No strong edge detected. Watch the stock and re-analyze when IV changes or a catalyst appears.</span>
          ) : (
            <span>📊 Review the signals above and the best spread recommendations below to determine your entry.</span>
          )}
        </div>
        {(rec.leveraged_bull || rec.leveraged_bear) && (
          <div className="text-xs text-slate-500 mt-2">
            <b>Leveraged alternative:</b> Instead of options, use <span className="text-emerald-400 font-mono">{rec.leveraged_bull}</span> (2x bull) or <span className="text-rose-400 font-mono">{rec.leveraged_bear}</span> (2x bear) ETF for a simpler leveraged position with no expiry.
          </div>
        )}
      </div>

      {/* Alt trade idea */}
      {rec.alt && (
        <div className="mt-2 flex items-start gap-2 text-xs text-slate-400">
          <span className="text-blue-400 flex-shrink-0">Alt:</span>
          <span>{rec.alt}</span>
        </div>
      )}
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// Sentiment panel
// ────────────────────────────────────────────────────────────────────────────

function SentimentPanel({ sentiment }: { sentiment: Sentiment }) {
  const { score, signal, bull_pct, reddit_buzz, trend_spike, contrarian_flag } = sentiment;

  const scoreColor = score > 65 ? 'text-emerald-400' : score < 35 ? 'text-rose-400' : 'text-amber-400';
  const bullBarColor = bull_pct > 60 ? 'bg-emerald-500' : bull_pct < 40 ? 'bg-rose-500' : 'bg-amber-500';

  // Contrarian flag color
  const cfColor = contrarian_flag === 'Squeeze Watch' ? 'text-blue-400 bg-blue-400/10 border-blue-400/30'
    : contrarian_flag === 'Sell the Hype' ? 'text-orange-400 bg-orange-400/10 border-orange-400/30'
    : contrarian_flag === 'Buy the Dip' ? 'text-emerald-400 bg-emerald-400/10 border-emerald-400/30'
    : contrarian_flag === 'Fade the Fear' ? 'text-violet-400 bg-violet-400/10 border-violet-400/30'
    : '';

  return (
    <div className="panel" data-testid="section-sentiment">
      <div className="panel-title">
        <Radio size={14} />
        Retail Sentiment
        <span className="ml-auto text-xs text-slate-500">Reddit · StockTwits · Google Trends</span>
      </div>

      <div className="grid grid-cols-2 gap-3 mb-3">
        {/* Score dial */}
        <div className="mini-metric">
          <div className="mini-label">Sentiment Score</div>
          <div className={`mini-value text-2xl font-bold ${scoreColor}`}>{Number(score ?? 0).toFixed(0)}<span className="text-sm">/100</span></div>
          <div className="text-xs text-slate-500 mt-0.5">{signal}</div>
        </div>
        {/* Bull % */}
        <div className="mini-metric">
          <div className="mini-label">Bullish Posts %</div>
          <div className="flex items-center gap-2 mt-1">
            <div className="flex-1 h-2 rounded-full bg-white/10">
              <div className={`h-full rounded-full ${bullBarColor}`} style={{ width: `${bull_pct}%` }} />
            </div>
            <span className={`text-sm font-mono font-bold ${bullBarColor.replace('bg-', 'text-').replace('-500', '-400')}`}>{Number(bull_pct ?? 0).toFixed(0)}%</span>
          </div>
          <div className="flex justify-between text-xs text-slate-500 mt-0.5">
            <span>Bearish</span><span>Bullish</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2 mb-3">
        <div className="mini-metric">
          <div className="mini-label">Reddit Buzz</div>
          <div className={`mini-value ${reddit_buzz > 7 ? 'text-orange-400' : reddit_buzz > 4 ? 'text-amber-400' : 'text-slate-300'}`}>
            {Number(reddit_buzz ?? 0).toFixed(1)}/10
            {reddit_buzz > 7 && <span className="ml-1">🔥</span>}
          </div>
        </div>
        <div className="mini-metric">
          <div className="mini-label">Google Trend Spike</div>
          <div className={`mini-value ${trend_spike > 75 ? 'text-rose-400' : trend_spike > 50 ? 'text-amber-400' : 'text-slate-300'}`}>
            {Number(trend_spike ?? 0).toFixed(0)}<span className="text-slate-500">/100</span>
          </div>
        </div>
      </div>

      {/* Contrarian flag */}
      {contrarian_flag && (
        <div className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm font-semibold ${cfColor}`}>
          {contrarian_flag === 'Squeeze Watch' && <span>💥</span>}
          {contrarian_flag === 'Sell the Hype' && <span>💰</span>}
          {contrarian_flag === 'Buy the Dip' && <span>📈</span>}
          {contrarian_flag === 'Fade the Fear' && <span>🛡️</span>}
          {contrarian_flag}
          <span className="ml-auto text-xs font-normal opacity-70">
            {contrarian_flag === 'Squeeze Watch' && 'Heavy short + retail hype = squeeze risk'}
            {contrarian_flag === 'Sell the Hype' && 'Quant firms often sell into retail euphoria'}
            {contrarian_flag === 'Buy the Dip' && 'Retail panic selling — institutions often buy'}
            {contrarian_flag === 'Fade the Fear' && 'Oversold with fear signal — mean reversion watch'}
          </span>
        </div>
      )}

      <p className="text-xs text-slate-500 mt-2 leading-relaxed">
        Contrarian signal: when retail sentiment is extreme, quant firms often trade the opposite direction.
        High buzz + high short interest = potential squeeze.
      </p>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// Earnings intelligence panel
// ────────────────────────────────────────────────────────────────────────────

function EarningsIntelPanel({ intel }: { intel: EarningsIntel }) {
  const totalGames = intel.beat_count + intel.miss_count;
  const beatBarPct = totalGames > 0 ? (intel.beat_count / totalGames) * 100 : 50;

  const ivVsHistColor = intel.options_edge > 0 ? 'text-emerald-400' : 'text-rose-400';
  const maxHistIV = Math.max(intel.iv_implied_move ?? 0, intel.historical_avg_move ?? 0, 1);

  const timingColor = intel.timing === 'AMC' ? 'bg-violet-400/20 text-violet-300 border border-violet-400/30'
    : intel.timing === 'BMO' ? 'bg-blue-400/20 text-blue-300 border border-blue-400/30'
    : 'bg-slate-700/50 text-slate-300 border border-slate-600/30';

  const earningsRec = intel.earnings_recommendation ?? '';
  const recColor2 = earningsRec.toLowerCase().includes('buy') ? 'text-emerald-400'
    : earningsRec.toLowerCase().includes('avoid') || earningsRec.toLowerCase().includes('sell') ? 'text-rose-400'
    : 'text-amber-400';

  return (
    <div className="panel" data-testid="section-earnings-intel">
      <div className="panel-title">
        <Newspaper size={14} />
        Earnings Intelligence
        {/* Timing badge */}
        {intel.timing && intel.timing !== 'Unknown' && (
          <span className={`ml-2 px-2 py-0.5 rounded text-xs font-semibold ${timingColor}`}>
            {intel.timing}
          </span>
        )}
        {intel.days_to_earnings != null && intel.days_to_earnings >= 0 && intel.days_to_earnings <= 60 && (
          <span className="ml-auto text-xs text-amber-400">⚠️ {intel.days_to_earnings}d to earnings</span>
        )}
        {intel.days_to_earnings != null && intel.days_to_earnings < 0 && (
          <span className="ml-auto text-xs text-slate-500">Last earnings {Math.abs(intel.days_to_earnings)}d ago</span>
        )}
      </div>

      {/* Beat/miss history bar */}
      {totalGames > 0 && (
        <div className="mb-3">
          <div className="flex justify-between text-xs text-slate-400 mb-1">
            <span>Beat/Miss History ({totalGames} quarters)</span>
            <span className={intel.beat_pct >= 60 ? 'text-emerald-400' : 'text-rose-400'}>
              {Number(intel.beat_pct ?? 0).toFixed(0)}% beat rate
            </span>
          </div>
          <div className="h-3 rounded-full bg-white/10 overflow-hidden">
            <div
              className={`h-full rounded-full transition-all ${intel.beat_pct >= 60 ? 'bg-emerald-500' : intel.beat_pct >= 40 ? 'bg-amber-500' : 'bg-rose-500'}`}
              style={{ width: `${beatBarPct}%` }}
            />
          </div>
          <div className="flex justify-between text-xs text-slate-500 mt-1">
            <span>Miss ({intel.miss_count})</span>
            {intel.avg_surprise_pct !== 0 && (
              <span className={intel.avg_surprise_pct > 0 ? 'text-emerald-400' : 'text-rose-400'}>
                Avg surprise: {intel.avg_surprise_pct > 0 ? '+' : ''}{Number(intel.avg_surprise_pct ?? 0).toFixed(1)}%
              </span>
            )}
            <span>Beat ({intel.beat_count})</span>
          </div>
        </div>
      )}

      {/* IV implied vs historical move comparison */}
      {(intel.iv_implied_move != null && intel.historical_avg_move != null) && (
        <div className="mb-3">
          <div className="text-xs text-slate-400 mb-2">Expected Move Comparison</div>
          {[
            { label: 'IV Implied Move', val: intel.iv_implied_move, color: 'bg-blue-500' },
            { label: 'Historical Avg Move', val: intel.historical_avg_move, color: 'bg-amber-500' },
          ].map(({ label, val, color }) => (
            <div key={label} className="cone-row">
              <span className="cone-label">{label}</span>
              <div className="cone-bar-track">
                <div className={`cone-bar ${color}`} style={{ width: `${(val / maxHistIV) * 100}%` }} />
              </div>
              <span className="cone-value">{val.toFixed(1)}%</span>
            </div>
          ))}
          {intel.options_edge != null && (
            <div className={`text-xs font-semibold mt-1.5 ${ivVsHistColor}`}>
              Options edge: {intel.options_edge > 0 ? '+' : ''}{intel.options_edge.toFixed(1)}%
              <span className="text-slate-500 font-normal ml-1">
                ({intel.options_edge > 0 ? 'IV overpriced vs history — consider selling' : 'IV underpriced vs history — consider buying'})
              </span>
            </div>
          )}
        </div>
      )}

      {/* Gap-and-hold + recommendation */}
      <div className="flex items-start justify-between gap-3 flex-wrap">
        <div className="mini-metric">
          <div className="mini-label">Gap &amp; Hold Pattern</div>
          <div className={`mini-value ${intel.gap_and_hold ? 'text-emerald-400' : 'text-slate-400'}`}>
            {intel.gap_and_hold ? '✓ Yes — often holds post-earnings gap' : '✗ No — gaps tend to fade'}
          </div>
        </div>
        {intel.max_drop_pct > 0 && (
          <div className="mini-metric">
            <div className="mini-label">Max Historical Drop</div>
            <div className="mini-value text-rose-400">-{Number(intel.max_drop_pct ?? 0).toFixed(1)}%</div>
          </div>
        )}
      </div>

      {/* Earnings recommendation */}
      {earningsRec && (
        <div className={`mt-3 px-3 py-2 rounded-lg bg-white/[0.04] border border-white/[0.08]`}>
          <div className="text-xs text-slate-400 mb-0.5">Earnings Play Recommendation</div>
          <div className={`text-sm font-semibold ${recColor2}`}>{earningsRec}</div>
        </div>
      )}

      {/* Caution flag */}
      {intel.caution_flag && (
        <div className="flex items-center gap-2 mt-2 text-xs text-rose-400">
          <ShieldAlert size={12} />
          {intel.caution_flag}
        </div>
      )}
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// Scan progress bar — polls /api/scan/progress every 3s
// ────────────────────────────────────────────────────────────────────────────

function ScanProgressBar() {
  const { data } = useQuery<ScanProgressResponse>({
    queryKey: ["/api/scan/progress"],
    queryFn: async () => {
      const res = await apiRequest("GET", "/api/scan/progress");
      return res.json();
    },
    refetchInterval: 3000,
    staleTime: 0,
  });

  const progress = Number(data?.progress ?? 0);
  if (!data || progress >= 100) return null;

  return (
    <div className="flex items-center gap-3 px-3 py-2 rounded-lg bg-blue-400/5 border border-blue-400/20 mb-2">
      <RefreshCw size={12} className="text-blue-400 animate-spin flex-shrink-0" />
      <div className="flex-1">
        <div className="flex justify-between text-xs text-slate-400 mb-1">
          <span>Scanning full market</span>
          <span className="font-mono text-blue-400">{data.cached_count.toLocaleString()} / ~4,500 tickers</span>
        </div>
        <div className="h-1 rounded-full bg-white/10">
          <div
            className="h-full rounded-full bg-blue-500 transition-all duration-500"
            style={{ width: `${Math.min(progress, 100)}%` }}
          />
        </div>
      </div>
      <span className="text-xs text-blue-400 font-mono">{progress.toFixed(0)}%</span>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// Edge Factors Panel
// ────────────────────────────────────────────────────────────────────────────

function EdgeFactorsPanel({ ef, spot }: { ef: EdgeFactors; spot: number }) {
  return (
    <div className="panel" data-testid="section-edge-factors">
      <div className="panel-title">
        <Zap size={14} />
        Edge Factors
        <span className="ml-auto text-xs text-slate-500">Advanced signals for higher accuracy</span>
      </div>

      <div className="grid grid-cols-2 gap-3">

        {/* Put/Call Ratio */}
        {ef.put_call_ratio != null && (
          <div className="mini-metric col-span-2">
            <div className="mini-label">Put/Call Ratio (Open Interest)</div>
            <div className={`mini-value ${
              ef.put_call_signal === 'Extreme Fear' ? 'text-emerald-400' :
              ef.put_call_signal === 'Extreme Greed' ? 'text-rose-400' :
              ef.put_call_signal === 'Bearish' ? 'text-amber-400' :
              ef.put_call_signal === 'Bullish' ? 'text-blue-400' : 'text-slate-300'
            }`}>
              {Number(ef.put_call_ratio).toFixed(2)} — {ef.put_call_signal}
            </div>
            <div className="text-xs text-slate-500 mt-0.5">{ef.put_call_interp}</div>
          </div>
        )}

        {/* Short Squeeze */}
        {ef.squeeze_score != null && (
          <div className="mini-metric col-span-2">
            <div className="mini-label">Short Squeeze Score</div>
            <div className="flex items-center gap-2 mt-1">
              <div className="flex-1 h-2 rounded-full bg-white/10">
                <div
                  className={`h-full rounded-full ${ef.squeeze_score >= 70 ? 'bg-rose-500' : ef.squeeze_score >= 50 ? 'bg-amber-500' : 'bg-slate-500'}`}
                  style={{ width: `${ef.squeeze_score}%` }}
                />
              </div>
              <span className={`text-sm font-mono font-bold ${ef.squeeze_score >= 70 ? 'text-rose-400' : ef.squeeze_score >= 50 ? 'text-amber-400' : 'text-slate-400'}`}>
                {ef.squeeze_score}/100
              </span>
            </div>
            <div className="text-xs font-semibold mt-1">{ef.squeeze_signal}</div>
            <div className="text-xs text-slate-500 mt-0.5">{ef.squeeze_desc}</div>
          </div>
        )}

        {/* IV Crush */}
        {ef.iv_crush_score != null && (
          <div className="mini-metric col-span-2">
            <div className="mini-label">IV Crush Score (Earnings)</div>
            <div className={`mini-value ${ef.iv_crush_score >= 60 ? 'text-emerald-400' : ef.iv_crush_score >= 30 ? 'text-amber-400' : 'text-slate-300'}`}>
              {ef.iv_crush_score}/100
              {ef.iv_crush_pct != null && <span className="text-slate-500 ml-2 text-xs font-normal">Expected crush: {Number(ef.iv_crush_pct).toFixed(1)}%</span>}
            </div>
            <div className="text-xs text-slate-500 mt-0.5">{ef.iv_crush_rec}</div>
          </div>
        )}

        {/* Gamma Pin */}
        {ef.gamma_pin != null && (
          <div className="mini-metric">
            <div className="mini-label">Gamma Pin Level</div>
            <div className="mini-value text-blue-400">${Number(ef.gamma_pin).toFixed(2)}</div>
            <div className="text-xs text-slate-500 mt-0.5">
              {ef.gamma_pin_dist_pct != null && `${ef.gamma_pin_dist_pct > 0 ? '+' : ''}${Number(ef.gamma_pin_dist_pct).toFixed(1)}% from spot`}
              {ef.gamma_pin_oi != null && ` · ${(ef.gamma_pin_oi / 1000).toFixed(0)}K OI`}
            </div>
          </div>
        )}

        {/* Relative Strength */}
        {ef.relative_strength != null && (
          <div className="mini-metric">
            <div className="mini-label">Relative Strength (20d vs SPY)</div>
            <div className={`mini-value ${ef.relative_strength > 3 ? 'text-emerald-400' : ef.relative_strength < -3 ? 'text-rose-400' : 'text-slate-300'}`}>
              {ef.relative_strength > 0 ? '+' : ''}{Number(ef.relative_strength).toFixed(1)}%
            </div>
            <div className="text-xs text-slate-500 mt-0.5">{ef.relative_strength_signal}</div>
          </div>
        )}

      </div>

      <div className="mt-3 text-xs text-slate-500 leading-relaxed">
        <strong>P/C Ratio</strong> = Put/Call open interest ratio. High = fear (buy signal). Low = greed (sell signal).<br/>
        <strong>Squeeze Score</strong> = probability of a short squeeze based on short float, days to cover, and sentiment.<br/>
        <strong>IV Crush</strong> = how much IV is overpriced before earnings — sell options if high.<br/>
        <strong>Gamma Pin</strong> = the strike price where market makers will pin the stock near expiry (highest open interest).<br/>
        <strong>Relative Strength</strong> = how much this stock is outperforming/underperforming SPY over 20 days.
      </div>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// Market scanner
// ────────────────────────────────────────────────────────────────────────────

function MarketScanner({ onSelectTicker }: { onSelectTicker: (t: string) => void }) {
  const [expanded, setExpanded] = useState(false);

  const { data, isLoading, isError, refetch, isFetching } = useQuery<ScanResponse>({
    queryKey: ["/api/scan"],
    queryFn: async () => {
      const res = await apiRequest("GET", "/api/scan?top_n=10");
      return res.json();
    },
    staleTime: 5 * 60 * 1000,
    refetchInterval: (data: any) => {
      // Keep polling every 15s until we have results
      if (!data || !data.results || data.results.length === 0) return 15000;
      return false;
    },
    retry: false,
  });

  const results = data?.results ?? [];
  const displayResults = expanded ? results : results.slice(0, 5);

  if (isLoading || isFetching) {
    return (
      <div className="scanner-panel">
        <div className="scanner-header">
          <Target size={14} className="text-blue-400" />
          <span>Market Scanner</span>
          <div className="ml-auto flex items-center gap-1.5 text-xs text-slate-400">
            <RefreshCw size={12} className="animate-spin" />
            Loading scanner data… (may take 30s on first load)
          </div>
        </div>
        <div className="scanner-loading">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="animate-pulse flex items-center gap-3 p-2 rounded bg-white/[0.03]">
              <div className="w-12 h-4 bg-white/10 rounded" />
              <div className="flex-1 h-3 bg-white/5 rounded" />
              <div className="w-8 h-4 bg-white/10 rounded" />
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (isError || !data) {
    return (
      <div className="scanner-panel">
        <div className="scanner-header">
          <Target size={14} className="text-blue-400" />
          <span>Market Scanner</span>
          <button className="ml-auto text-xs text-blue-400 hover:text-blue-300" onClick={() => refetch()}>Retry</button>
        </div>
        <p className="text-xs text-slate-500 p-3">Scanner unavailable. Use the tickers below.</p>
        <div className="chip-row">
          {["SPY","QQQ","AAPL","TSLA","NVDA","AMZN"].map(t => (
            <button key={t} className="chip" onClick={() => onSelectTicker(t)} data-testid={`chip-${t}`}>{t}</button>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="scanner-panel">
      <div className="scanner-header">
        <Target size={14} className="text-blue-400" />
        <span className="font-semibold">Top Opportunities</span>
        <span className="text-slate-500 text-xs ml-1">— ranked by algorithm score</span>
        <div className="ml-auto flex items-center gap-2">
          {data.cached && (
            <span className="text-xs text-slate-500">{Math.floor(data.age_seconds / 60)}m cached</span>
          )}
          <button
            onClick={() => refetch()}
            disabled={isFetching}
            className="text-xs text-blue-400 hover:text-blue-300 flex items-center gap-1"
            title="Refresh scan"
            data-testid="button-refresh-scan"
          >
            <RefreshCw size={11} className={isFetching ? "animate-spin" : ""} />
            Refresh
          </button>
        </div>
      </div>

      {/* Live market scan progress bar */}
      <ScanProgressBar />

      {/* Header row */}
      <div className="scanner-row-header">
        <span className="w-4">#</span>
        <span className="w-14">Ticker</span>
        <span className="flex-1">Action / Signal</span>
        <span className="w-12 text-right">ATM IV</span>
        <span className="w-10 text-right">VRP</span>
        <span className="w-14 text-right">Sentiment</span>
        <span className="w-14 text-right">Score</span>
      </div>

      {displayResults.map((r, i) => {
        const vrpColor = (r.vrp ?? 0) > 5 ? "text-emerald-400" : (r.vrp ?? 0) > 0 ? "text-blue-400" : "text-rose-400";
        const scoreColor = (r.scan_score ?? 0) >= 65 ? "text-emerald-400" : (r.scan_score ?? 0) >= 45 ? "text-amber-400" : "text-slate-400";
        const hvPct = r.hv_percentile;
        // Freshness dot
        const freshDot = r.freshness === 'fresh' ? '🟢' : r.freshness === 'recent' ? '🟡' : r.freshness === 'stale' ? '🔴' : '';
        // Sentiment score color
        const sentColor = r.sentiment_score !== undefined
          ? r.sentiment_score > 65 ? 'text-emerald-400' : r.sentiment_score < 35 ? 'text-rose-400' : 'text-amber-400'
          : 'text-slate-500';
        // Action display — prefer rec_signal, fall back to best_strategy
        const actionDisplay = r.rec_signal ?? r.best_strategy ?? '—';

        return (
          <button
            key={r.ticker}
            className="scanner-row"
            onClick={() => onSelectTicker(r.ticker)}
            data-testid={`scanner-row-${r.ticker}`}
          >
            <span className="w-4 text-slate-500 font-mono text-xs">{i + 1}</span>
            <span className="w-14 font-mono font-bold text-white text-sm">
              {freshDot && <span className="mr-0.5" style={{ fontSize: '8px' }}>{freshDot}</span>}
              {r.ticker}
            </span>
            <span className="flex-1 text-xs text-slate-300 truncate">
              {actionDisplay}
            </span>
            <span className="w-12 text-right font-mono text-xs text-slate-300">{r.atm_iv != null ? r.atm_iv.toFixed(1) + '%' : '—'}</span>
            <span className={`w-10 text-right font-mono text-xs ${vrpColor}`}>
              {r.vrp != null ? (r.vrp > 0 ? '+' : '') + r.vrp.toFixed(1) : '—'}
            </span>
            <span className={`w-14 text-right font-mono text-xs ${sentColor}`}>
              {r.sentiment_score !== undefined ? r.sentiment_score.toFixed(0) : '—'}
            </span>
            <span className={`w-14 text-right font-mono text-xs font-semibold ${scoreColor}`}>
              {(r.scan_score ?? 0).toFixed(0)}
            </span>
          </button>
        );
      })}

      {results.length > 5 && (
        <button
          className="w-full text-xs text-slate-400 hover:text-slate-200 py-2 mt-1 transition-colors"
          onClick={() => setExpanded(!expanded)}
        >
          {expanded ? "Show less ▲" : `Show all ${results.length} results ▼`}
        </button>
      )}

      <p className="text-xs text-slate-500 mt-2 px-0.5">
        💡 How the scanner works: It scores every ticker on options edge (how much implied vol exceeds real vol), 
        the quality of the best available spread, how elevated volatility is vs its own history, and recent volume spikes.
        A score of 70+ = strong edge. 🟢 fresh &lt;5min · 🟡 recent &lt;20min · 🔴 stale. Click any row to deep-analyze.
      </p>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// Loading skeleton
// ────────────────────────────────────────────────────────────────────────────

function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`animate-pulse bg-white/5 rounded ${className}`} />;
}

function LoadingSkeleton() {
  return (
    <div className="space-y-4 mt-8">
      <div className="flex gap-2 items-center">
        <Skeleton className="h-6 w-40" />
        <Skeleton className="h-4 w-24" />
      </div>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {[...Array(4)].map((_, i) => <Skeleton key={i} className="h-20" />)}
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {[...Array(4)].map((_, i) => <Skeleton key={i} className="h-40" />)}
      </div>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// Main page
// ────────────────────────────────────────────────────────────────────────────

export default function AnalyzePage({ initialTicker }: AnalyzePageProps = {}) {
  const [input, setInput] = useState(initialTicker ?? "");
  const [ticker, setTicker] = useState<string | null>(initialTicker ?? null);
  const inputRef = useRef<HTMLInputElement>(null);
  const resultsRef = useRef<HTMLDivElement>(null);

  const { data, isLoading, isError, error } = useQuery<AnalysisResult>({
    queryKey: ["/api/analyze", ticker],
    queryFn: async () => {
      const res = await apiRequest("GET", `/api/analyze/${ticker}`);
      return res.json();
    },
    enabled: !!ticker,
    retry: false,
    staleTime: 30000,
  });

  // Update when initialTicker changes
  useEffect(() => {
    if (initialTicker && initialTicker !== ticker) {
      setInput(initialTicker);
      setTicker(initialTicker);
    }
  }, [initialTicker]);

  // Scroll to results when loaded
  useEffect(() => {
    if (data && !data.error && resultsRef.current) {
      setTimeout(() => resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 100);
    }
  }, [data]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const t = input.trim().toUpperCase();
    if (t) setTicker(t);
  };

  const selectTicker = (t: string) => {
    setInput(t);
    setTicker(t);
  };

  const vrpColor = data?.vrp_regime === "high" ? "text-emerald-400"
    : data?.vrp_regime === "low" ? "text-rose-400" : "text-amber-400";

  const priceUp = data && data.price_change >= 0;

  return (
    <div>

        {/* Page heading badge */}
        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "1rem" }}>
          <span style={{
            fontSize: 9, padding: "3px 8px", borderRadius: 3,
            background: "rgba(0, 229, 255, 0.1)", border: "1px solid rgba(0, 229, 255, 0.2)",
            color: "#00e5ff", fontFamily: "'JetBrains Mono', monospace",
            letterSpacing: "0.12em", textTransform: "uppercase", fontWeight: 600,
          }}>
            MARKET INTEL
          </span>
        </div>

        {/* Search */}
        <form onSubmit={handleSubmit} className="search-form" data-testid="form-ticker">
          <div className="search-wrapper">
            <Search className="search-icon" size={18} />
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={e => setInput(e.target.value.toUpperCase())}
              placeholder="Enter a ticker — AAPL, SPY, TSLA…"
              className="search-input"
              maxLength={10}
              autoComplete="off"
              autoCapitalize="characters"
              spellCheck={false}
              data-testid="input-ticker"
            />
          </div>
          <button type="submit" className="analyze-btn" disabled={!input.trim()} data-testid="button-analyze">
            <Zap size={16} />
            Analyze
          </button>
        </form>

        {/* Quick tickers */}
        <div className="chip-row">
          {["SPY","QQQ","AAPL","TSLA","NVDA","AMZN","MSFT","AMD"].map(t => (
            <button key={t} className="chip" onClick={() => selectTicker(t)}>{t}</button>
          ))}
        </div>

        {/* Loading */}
        {isLoading && <LoadingSkeleton />}

        {/* Error */}
        {(isError || data?.error) && !isLoading && (
          <div className="error-card" data-testid="text-error">
            <span className="text-rose-400 font-semibold">
              {data?.error || "Something went wrong. Please try again."}
            </span>
            {(data as any)?.no_options && (
              <div className="mt-3">
                <p className="text-xs text-slate-400 mb-2">These always work — click to analyze:</p>
                <div className="chip-row">
                  {["SPY","QQQ","AAPL","TSLA","NVDA","MSFT","AMZN","META","AMD","COIN","GME"].map(t => (
                    <button key={t} className="chip" onClick={() => selectTicker(t)} data-testid={`chip-${t}`}>{t}</button>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Results */}
        {data && !data.error && !isLoading && (
          <div className="results" ref={resultsRef} data-testid="section-results">

            {/* Company header */}
            <div className="company-header">
              <div className="flex items-center gap-3 flex-wrap">
                <div>
                  <h1 className="company-name" data-testid="text-company">{data.company_name}</h1>
                  <span className="ticker-tag" data-testid="text-ticker">{data.ticker}</span>
                </div>
                {/* Valuation verdict badge */}
                {data.valuation && (
                  <span className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold border ${
                    data.valuation.color === "green"
                      ? "text-emerald-400 bg-emerald-400/10 border-emerald-400/30"
                      : data.valuation.color === "red"
                      ? "text-rose-400 bg-rose-400/10 border-rose-400/30"
                      : "text-amber-400 bg-amber-400/10 border-amber-400/30"
                  }`}>
                    {data.valuation.color === "green" ? <CheckCircle size={12} />
                      : data.valuation.color === "red" ? <XCircle size={12} />
                      : <Info size={12} />}
                    {data.valuation.verdict}
                  </span>
                )}
              </div>
              <div className="price-group">
                <span className="spot-price" data-testid="text-spot">${data.spot.toFixed(2)}</span>
                <span className={`price-change ${priceUp ? "text-emerald-400" : "text-rose-400"}`} data-testid="text-price-change">
                  {priceUp ? <ChevronUp size={14} className="inline" /> : <ChevronDown size={14} className="inline" />}
                  {Math.abs(data.price_change).toFixed(2)} ({Math.abs(data.price_change_pct).toFixed(2)}%)
                </span>
              </div>
            </div>

            {/* Live price bar */}
            {data.fundamentals && (
              <LivePriceBar
                f={data.fundamentals}
                spot={data.spot}
                high52={data.high_52}
                low52={data.low_52}
              />
            )}

            {/* Key metrics */}
            <div className="metrics-grid">
              <MetricCard label="ATM Implied Vol" value={`${data.atm_iv.toFixed(1)}%`} sub="30-day options IV" />
              <MetricCard label="Realized Vol (20d)" value={`${data.rv20.toFixed(1)}%`} sub="Historical volatility" />
              <MetricCard
                label="Vol Risk Premium"
                value={`${data.vrp > 0 ? "+" : ""}${data.vrp.toFixed(1)}%`}
                sub={data.vrp_regime === "high" ? "Sell options ↑ (IV expensive)" : data.vrp_regime === "low" ? "Buy options ↓ (IV cheap)" : "Neutral →"}
                highlight={data.vrp_regime === "high" ? "up" : data.vrp_regime === "low" ? "down" : "neutral"}
              />
              <MetricCard
                label="52-Week Range"
                value={data.low_52 && data.high_52 ? `$${data.low_52}` : "—"}
                sub={data.high_52 ? `High: $${data.high_52}` : undefined}
              />
            </div>

            {/* VRP signal banner */}
            <div className={`vrp-banner ${data.vrp_regime}`} data-testid="text-vrp-signal">
              <Activity size={14} className="flex-shrink-0 mt-0.5" />
              <div style={{ flex: 1 }}>
                <span className="font-semibold">Volatility Signal: </span>
                {data.vrp_signal}
                <span className="text-xs ml-2 opacity-60">
                  — IV vs RV spread: {data.vrp > 0 ? "+" : ""}{data.vrp.toFixed(1)}%
                  {data.skew !== null && data.skew !== undefined && (
                    <> · Skew: {data.skew > 0 ? '+' : ''}{data.skew.toFixed(1)}%</>
                  )}
                </span>
                <div className="text-xs mt-1 opacity-70">
                  {data.vrp_regime === "high" && (
                    <span>
                      💡 <strong>Sell options</strong> — IV is inflated {data.vrp.toFixed(1)}% above realized moves.
                      Historical win rate for selling when VRP is this high: <strong style={{color:'#30d158'}}>~65-75%</strong>.
                      <br/>
                      • <strong>Sell a Call</strong> (bearish/neutral) — ~68% profit probability at 30 delta
                      <br/>
                      • <strong>Sell a Put</strong> (bullish/neutral) — ~70% profit probability at 30 delta
                      <br/>
                      • <strong>Iron Condor</strong> (neutral) — ~60% profit probability, capped risk both sides
                      <br/>
                      • <strong>Covered Call</strong> (own shares) — ~75% probability of keeping premium
                    </span>
                  )}
                  {data.vrp_regime === "low" && (
                    <span>
                      💡 <strong>Buy options</strong> — IV is {Math.abs(data.vrp).toFixed(1)}% below realized moves. Options are underpriced.
                      Historical win rate for buying when VRP is negative: <strong style={{color:'#30d158'}}>~55-65%</strong> (with proper timing).
                      <br/>
                      • <strong>Buy a Call</strong> (bullish) — ~40-50% win rate but 2-3x reward when right
                      <br/>
                      • <strong>Buy a Put</strong> (bearish) — ~40-50% win rate, hedge against drops
                      <br/>
                      • <strong>Debit Spread</strong> — ~50-55% win rate, lower cost, defined risk
                      <br/>
                      Key: cheap IV means you pay less for big upside potential.
                    </span>
                  )}
                  {data.vrp_regime === "neutral" && (
                    <span>
                      💡 Options are fairly priced — no strong vol edge (VRP ~0%).
                      Win rate is close to 50/50 for both buying and selling.
                      <br/>
                      • <strong>Buy a Call</strong> if you think stock goes up — ~45% win rate
                      <br/>
                      • <strong>Buy a Put</strong> if you think stock goes down — ~45% win rate
                      <br/>
                      Focus on direction and use the Recommendation card below for the best specific trade.
                    </span>
                  )}
                </div>
              </div>
            </div>

            {/* 🌟 RECOMMENDATION CARD — main action at top of results */}
            {data.recommendation && (
              <RecommendationCard rec={data.recommendation} ticker={data.ticker} />
            )}

            {/* Three-column section: Vol Cone | Best Opportunity | Valuation */}
            <div className="three-col">
              {/* Vol Cone */}
              <div className="panel">
                <div className="panel-title">
                  <BarChart2 size={14} />
                  Volatility Cone
                </div>
                <div className="vol-cone-grid">
                  {[
                    { label: "10-day RV", val: data.rv10 },
                    { label: "20-day RV", val: data.rv20 },
                    { label: "30-day RV", val: data.rv30 },
                    { label: "ATM IV",    val: data.atm_iv },
                  ].map(({ label, val }) => {
                    const maxVal = Math.max(data.rv10, data.rv20, data.rv30, data.atm_iv, 1);
                    const pct = (val / maxVal) * 100;
                    const isIV = label === "ATM IV";
                    return (
                      <div key={label} className="cone-row">
                        <span className="cone-label">{label}</span>
                        <div className="cone-bar-track">
                          <div className={`cone-bar ${isIV ? "cone-bar-iv" : "cone-bar-rv"}`} style={{ width: `${pct}%` }} />
                        </div>
                        <span className="cone-value">{val.toFixed(1)}%</span>
                      </div>
                    );
                  })}
                </div>
                {data.vol_surface && data.vol_surface.length > 0 && (
                  <VolSurfaceChart data={data.vol_surface} />
                )}
                {/* Acronym legend */}
                <div style={{ marginTop: '12px', padding: '8px 10px', background: 'rgba(0, 20, 40, 0.5)', borderRadius: '8px', fontSize: '10px', color: 'var(--text-tertiary)', lineHeight: 1.7 }}>
                  <strong style={{ color: 'var(--text-secondary)', display: 'block', marginBottom: '2px' }}>What these mean:</strong>
                  <b>RV</b> = Realized Volatility — how much the stock <em>actually moved</em> over the last N days (annualized %)<br/>
                  <b>IV</b> = Implied Volatility — how much the <em>options market expects</em> the stock to move (annualized %)<br/>
                  <b>ATM IV</b> = At-the-Money Implied Vol — IV of the option closest to the current stock price<br/>
                  <b>VRP</b> = Volatility Risk Premium — IV minus RV. Positive = options are overpriced vs real moves<br/>
                  <b>Term Structure</b> = IV across different expiry dates. Rising = normal (calm market). Flat/inverted = stress<br/>
                  <b>Bars color:</b> <span style={{color:'#10b981'}}>Green</span> = high VRP (sell options edge) · <span style={{color:'#f43f5e'}}>Red</span> = negative VRP (buy options edge) · <span style={{color:'#f59e0b'}}>Amber</span> = neutral
                </div>
              </div>

              {/* Best opportunity */}
              {data.top_spreads && data.top_spreads.length > 0 && (
                <div className="panel">
                  <div className="panel-title">
                    <TrendingUp size={14} />
                    Best Opportunity
                  </div>
                  <div className="best-spread">
                    <div className="flex items-center gap-2 mb-2">
                      <span className="rank-badge">#1</span>
                      <span className="text-white font-semibold text-sm">{data.top_spreads[0].type}</span>
                    </div>
                    <p className="text-xs text-slate-300 mb-3 leading-relaxed">{data.top_spreads[0].description}</p>
                    <div className="best-stats">
                      <div className="best-stat">
                        <div className="mini-label">Probability of Profit</div>
                        <div className={`best-stat-val ${data.top_spreads[0].prob_profit >= 55 ? 'text-emerald-400' : 'text-amber-400'}`}>
                          {data.top_spreads[0].prob_profit.toFixed(1)}%
                        </div>
                      </div>
                      <div className="best-stat">
                        <div className="mini-label">Max Profit</div>
                        <div className="best-stat-val text-emerald-400">
                          {data.top_spreads[0].max_profit !== null ? `$${data.top_spreads[0].max_profit.toFixed(2)}` : "Unlimited"}
                        </div>
                      </div>
                      <div className="best-stat">
                        <div className="mini-label">Cost / Risk</div>
                        <div className="best-stat-val">${Math.abs(data.top_spreads[0].net_debit).toFixed(2)}</div>
                      </div>
                      <div className="best-stat">
                        <div className="mini-label">Algorithm Score</div>
                        <div className="best-stat-val">{data.top_spreads[0].score.toFixed(0)}/100</div>
                      </div>
                    </div>
                    <ScoreBar score={data.top_spreads[0].score} />
                    <p className="text-xs text-slate-500 mt-3 leading-relaxed">
                      Score = 40% IV edge + 25% bid-ask quality + 25% prob profit + 10% R/R. Bennett VRP framework.
                    </p>
                  </div>
                </div>
              )}

              {/* Valuation panel (if data available) */}
              {data.valuation && (
                <ValuationPanel valuation={data.valuation} />
              )}
            </div>

            {/* Sentiment + Earnings Intel side by side */}
            {(data.sentiment || data.earnings_intel) && (
              <div className="two-col">
                {data.sentiment && <SentimentPanel sentiment={data.sentiment} />}
                {data.earnings_intel && <EarningsIntelPanel intel={data.earnings_intel} />}
              </div>
            )}

            {/* Edge Factors Panel */}
            {data.edge_factors && Object.keys(data.edge_factors).length > 0 && (
              <EdgeFactorsPanel ef={data.edge_factors} spot={data.spot} />
            )}

            {/* Volume & Bennett metrics */}
            {data.vol_metrics && Object.keys(data.vol_metrics).length > 0 && (
              <VolMetricsPanel
                metrics={data.vol_metrics}
                skew={data.skew}
                atm_iv={data.atm_iv}
                rv20={data.rv20}
              />
            )}

            {/* Fundamentals panel */}
            {data.fundamentals && <FundamentalsPanel f={data.fundamentals} />}

            {/* All ranked spreads */}
            {data.top_spreads && data.top_spreads.length > 0 && (
              <div>
                <div className="section-heading">
                  <TrendingUp size={15} />
                  Top Ranked Spreads
                  <span className="section-count">{data.top_spreads.length} of {data.spread_count} scanned</span>
                </div>
                <div className="spreads-grid">
                  {data.top_spreads.map((spread, i) => (
                    <SpreadCard key={i} spread={spread} rank={i + 1} />
                  ))}
                </div>
              </div>
            )}

            {/* Footer */}
            <p className="disclaimer">
              For educational purposes only. Not financial advice. Options trading involves substantial risk of loss. Always consult a licensed financial professional before trading.
            </p>
          </div>
        )}

        {/* Empty state */}
        {!ticker && !isLoading && (
          <div className="empty-state">
            <div className="empty-icon">
              <BarChart2 size={32} className="opacity-40" />
            </div>
            <h2 className="empty-title">Analyze any stock's options</h2>
            <p className="empty-sub">
              Enter a ticker above to see volatility metrics, fundamental valuation,
              volume analytics, and the top-ranked options spreads — all scored by the VRP algorithm.
            </p>
            <div className="feature-list">
              <div className="feature">🌟 Full recommendation: Buy stock, sell options, leveraged ETF</div>
              <div className="feature">📊 Volatility Risk Premium signal &amp; vol cone</div>
              <div className="feature">📱 Retail sentiment (Reddit · StockTwits · Google Trends)</div>
              <div className="feature">📅 Earnings intelligence: beat/miss history, AMC/BMO timing</div>
              <div className="feature">🧠 GEX regime, RSI, MFI, Options Flow, Insider Net</div>
              <div className="feature">🎯 Top-ranked spreads with win probability</div>
            </div>
          </div>
        )}
    </div>
  );
}
