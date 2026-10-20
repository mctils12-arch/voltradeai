// Bot page v2 — every query the page reads, with its cadence. All routes are
// the existing owner-gated ones plus /api/bot/insight (v1.0.1096). Each
// hook result carries dataUpdatedAt so every panel can print its own age.
import { useQuery, type UseQueryResult } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";

/** Every Bot page read is polled, so none may hang forever (a hung request
 *  blocks its query's polling until a manual reload — lib/queryClient.ts
 *  apiRequest). These are quick reads; /api/bot/insight runs Python with a
 *  40 s server-side cap, so it gets more room. */
export const BOT_READ_TIMEOUT_MS = 20_000;
export const BOT_INSIGHT_TIMEOUT_MS = 60_000;

async function getJson<T>(url: string, timeoutMs: number = BOT_READ_TIMEOUT_MS): Promise<T> {
  const r = await apiRequest("GET", url, undefined, { timeoutMs });
  return (await r.json()) as T;
}

export interface BotStatus {
  active: boolean; killSwitch: boolean; equityPeak: number; maxDrawdownPct: number;
  autoResume?: unknown; mode?: string; circuitBreakerActive?: boolean; consecutiveStopLosses?: number;
}
export interface BotAccount {
  equity: number; lastEquity?: number; cash?: number; buyingPower?: number; dailyPnL?: number; dailyPnLPct?: number;
  drawdownPct?: number; equityPeak?: number;
}
/** Shape of /api/bot/positions (server/bot.ts maps Alpaca's raw position
 *  into this; pnlPct is already a percent). */
export interface BotPosition {
  ticker: string; qty: number; side: string; entryPrice: number; currentPrice: number; marketValue: number;
  pnl: number; pnlPct: number; stopPrice: number; takeProfitPrice: number | null; phase: number;
  rMultiple: number; highestPnl: number; daysHeld: number;
  asset_class?: string; isOption?: boolean; optionMeta?: unknown;
}
export interface AuditRowApi { time: string; type: string; message: string; action?: string; detail?: string }
export interface Performance {
  totalTrades?: number; winRate?: number; avgGain?: number; avgLoss?: number; profitFactor?: number;
  equityCurve?: Array<{ date: string; value: number; pnl?: number }>; realisticPnlPct?: number; totalPnlPct?: number;
}
export interface MarketStatus { isOpen: boolean; nextOpen: string; nextClose: string; timestamp: string }
export interface Calendar { earlyCloses?: Array<{ date: string; close: string }>; holidays?: Array<{ date: string; name: string }>; tomorrowStatus?: string }
export interface MlStatus { model_exists?: boolean; model_age_hours?: number; last_accuracy?: number; last_samples?: number; last_status?: string }
export interface InsightFeature { name: string; n: number; mode: number; mode_fraction: number; variance: number; constant: boolean; last: number }
export interface Insight {
  generated_at?: string;
  model?: { exists: boolean; age_hours: number | null; fallback_at_hours: number | null; calibrator_newer_than_model: boolean | null };
  blend?: { rule_weight: number | null; ml_weight: number | null; updated?: string | null };
  performance?: { total_trades?: number; win_rate?: number; avg_win?: number; avg_loss?: number; recent_win_rate_20?: number };
  recent?: {
    window: number; traced: number; model_types: Record<string, number>; regime_used: Record<string, number>;
    gated: number; blended: number; errors: number;
    win_probability: { n: number; p10: number | null; p50: number | null; p90: number | null };
    features: InsightFeature[]; constant_features: string[]; feature_count: number;
    regime_labels: Record<string, number>; vxx_ratio: { last: number | null; min: number | null; max: number | null };
    last_timestamp?: string | null; last_score?: number | null; error?: string;
  };
  bands?: {
    labeling?: { win_pct?: number; stop_pct?: number; note?: string };
    score_quintile_edges?: number[];
    score_quintile?: Record<string, Record<string, { n: number; win_rate?: number; mean_return_pct?: number }>>;
    decision?: Record<string, Record<string, { n: number; win_rate?: number }>>;
    decision_x_band?: Record<string, Record<string, { n: number; win_rate?: number }>>;
    error?: string;
  };
  error?: string;
}
export interface LastScan { timestamp?: string; scanned?: number; new_trades?: unknown[]; top_10?: Array<{ ticker: string; score: number; side?: string; action_label?: string; reasons?: string[] }>; sizing_blocks?: Array<{ ticker: string; reason: string; score?: number | null; strategy?: string }>; scan_phase_times?: unknown }
/** GET /api/bot/benchmark (server/spyBenchmark.ts): the account vs
 *  buy-and-hold SPY on dividend-adjusted closes. Null windows = not enough
 *  data on one side — never a zero. */
export interface BenchWindow { from: string; to: string; days: number; accountPct: number; spyPct: number; gapPp: number }
export interface Benchmark {
  since: BenchWindow | null; d30: BenchWindow | null; spyLast: { date: string; pct: number } | null;
  barsThrough: string | null; fetchedAt?: string | null; adjustment?: string; error?: string;
}
/** GET /api/bot/sleeves (server/sleeveLedger.ts): realized P&L by sleeve from
 *  Alpaca's fill ledger + open P&L from current positions. */
export interface SleeveRow { sleeve: "options" | "floor" | "stocks"; fills: number; symbols: number; realized: number; unpricedCloses: number; unrealized: number | null; openPositions: number; byMonth: Record<string, number> }
export interface UnderlyingRow { root: string; fills: number; realized: number; byMonth: Record<string, number>; open: number | null; openPositions: number }
export type ActivityGroup = "dividends" | "interest" | "fees" | "journals" | "optionEvents" | "corporate" | "transfers" | "other";
/** server/accountActivities.ts: the non-trade cash flows the fill ledger cannot see, and the reconciliation to the equity curve */
export interface NonTradeLedger { total: number; count: number; nonCashRows: number; groups: Record<ActivityGroup, number>; byMonth: Record<string, number>; types: Array<{ type: string; group: ActivityGroup; count: number; net: number; byMonth: Record<string, number> }>; first: string | null; last: string | null }
export interface Reconciliation { equityFrom: string; equityTo: string; equityStart: number; equityEnd: number; equityChange: number; realized: number; unrealized: number | null; nonTrade: number; explained: number | null; residual: number | null }
export interface Sleeves {
  rows?: SleeveRow[]; realizedTotal?: number; unrealizedTotal?: number | null; firstFill?: string | null; lastFill?: string | null;
  optionsByUnderlying?: UnderlyingRow[]; optionsUnderlyings?: number;
  nonTrade?: NonTradeLedger; reconciliation?: Reconciliation | null; malformedActivities?: number;
  /** every lot's basis comes from the full fill history; these fills precede the window and only served as basis */
  windowStart?: string | null; preWindowFills?: number;
  since?: string; truncated?: boolean; malformedFills?: number; fetchedAt?: string; stale?: boolean; pending?: boolean; error?: string; lastError?: string | null;
}
export interface OpenOrder { id?: string; symbol: string; side: string; qty?: string | number; type?: string; limit_price?: string | number | null; status?: string; submitted_at?: string; filled_avg_price?: string | number | null }

export interface BotData {
  status: UseQueryResult<BotStatus>;
  account: UseQueryResult<BotAccount>;
  positions: UseQueryResult<BotPosition[]>;
  audit: UseQueryResult<AuditRowApi[]>;
  performance: UseQueryResult<Performance>;
  market: UseQueryResult<MarketStatus>;
  calendar: UseQueryResult<Calendar>;
  ml: UseQueryResult<MlStatus>;
  insight: UseQueryResult<Insight>;
  lastScan: UseQueryResult<LastScan>;
  orders: UseQueryResult<OpenOrder[]>;
  benchmark: UseQueryResult<Benchmark>;
  sleeves: UseQueryResult<Sleeves>;
}

export function useBotData(): BotData {
  const status = useQuery<BotStatus>({ queryKey: ["/api/bot/status"], queryFn: () => getJson("/api/bot/status"), refetchInterval: 5_000 });
  const account = useQuery<BotAccount>({ queryKey: ["/api/bot/account"], queryFn: () => getJson("/api/bot/account"), refetchInterval: 10_000 });
  const positions = useQuery<BotPosition[]>({ queryKey: ["/api/bot/positions"], queryFn: () => getJson("/api/bot/positions"), refetchInterval: 15_000 });
  const audit = useQuery<AuditRowApi[]>({ queryKey: ["/api/bot/audit"], queryFn: () => getJson("/api/bot/audit"), refetchInterval: 10_000 });
  const performance = useQuery<Performance>({ queryKey: ["/api/bot/performance"], queryFn: () => getJson("/api/bot/performance"), refetchInterval: 60_000 });
  const market = useQuery<MarketStatus>({ queryKey: ["/api/bot/market-status"], queryFn: () => getJson("/api/bot/market-status"), refetchInterval: 60_000 });
  const calendar = useQuery<Calendar>({ queryKey: ["/api/bot/calendar"], queryFn: () => getJson("/api/bot/calendar"), refetchInterval: 10 * 60_000 });
  const ml = useQuery<MlStatus>({ queryKey: ["/api/ml/status"], queryFn: () => getJson("/api/ml/status"), refetchInterval: 60_000 });
  const insight = useQuery<Insight>({ queryKey: ["/api/bot/insight"], queryFn: () => getJson("/api/bot/insight", BOT_INSIGHT_TIMEOUT_MS), refetchInterval: 5 * 60_000, staleTime: 2 * 60_000 });
  const lastScan = useQuery<LastScan>({ queryKey: ["/api/bot/last-scan"], queryFn: () => getJson("/api/bot/last-scan"), refetchInterval: 60_000 });
  const orders = useQuery<OpenOrder[]>({ queryKey: ["/api/orders/open"], queryFn: () => getJson("/api/orders/open"), refetchInterval: 30_000 });
  const benchmark = useQuery<Benchmark>({ queryKey: ["/api/bot/benchmark"], queryFn: () => getJson("/api/bot/benchmark"), refetchInterval: 10 * 60_000 });
  // pending while the server's first ledger computation runs: poll sooner
  const sleeves = useQuery<Sleeves>({ queryKey: ["/api/bot/sleeves"], queryFn: () => getJson("/api/bot/sleeves"), refetchInterval: (q) => (q.state.data?.pending ? 20_000 : 10 * 60_000) });
  return { status, account, positions, audit, performance, market, calendar, ml, insight, lastScan, orders, benchmark, sleeves };
}

export function num(v: unknown, fallback: number | null = null): number | null {
  if (v === null || v === undefined || v === "") return fallback;
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? n : fallback;
}
