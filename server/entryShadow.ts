/**
 * entryShadow.ts — Node half of the ENTRY-SHADOW switch (human directive
 * 2026-10-28, paper account: "move the options sleeve and stock entries to
 * shadow mode. Keep the floor basket and all exits live."). Python half and
 * the full contract: entry_shadow.py.
 *
 * Shadow mode = the bot computes the would-be entry and LOGS it
 * (shadow_portfolio "shadow_mode_entry") instead of ordering it. Gated in
 * bot.ts: executeTrades' stock + ETF buys (options opens are gated in the
 * Python they call), the morning queue, the tier dispatcher's Tier-3 BUY and
 * SELL_CSP / BUY_PUT. NEVER gated: exits (WS stops, scale-outs, TIME-EXIT,
 * POS-KILL, kill-switch liquidation), the floor basket (Python
 * _manage_spy_floor / _manage_defensive_floor), the owner's manual
 * /api/bot/trade route.
 *
 * Flag read exactly like entry_shadow.flag_raw / regimeReport.regimeFlagRaw:
 * VOLTRADE_SHADOW_ENTRIES when PRESENT (any value — "off"/"0"/"" included —
 * wins), else the default below, a MIRROR of system_config.py
 * BASE_CONFIG["SHADOW_ENTRY_SLEEVES"] (test_entry_shadow.py pins the parity).
 * Pure (node:test safe).
 */

export const ENTRY_SHADOW_ENV = "VOLTRADE_SHADOW_ENTRIES";
export const ENTRY_SHADOW_DEFAULTS: Readonly<Record<string, string>> = Object.freeze({
  VOLTRADE_SHADOW_ENTRIES: "stocks,options",
});

/** shadow_portfolio.SHADOW_ENTRY_DECISION / SHADOW_REPEAT_DECISION. */
export const SHADOW_ENTRY_DECISION = "shadow_mode_entry";
export const SHADOW_REPEAT_DECISION = "shadow_mode_repeat";
/** entry_shadow.SHADOW_ORDER_STATUS: what a Python options gate returns instead of an order. */
export const SHADOW_ORDER_STATUS = "shadowed";

export type EntrySleeve = "stocks" | "options";
const OFF_VALUES = new Set(["", "0", "off", "none", "false"]);
const ALIASES: Record<string, EntrySleeve> = { stock: "stocks", stocks: "stocks", option: "options", options: "options" };

export interface EntryShadowFlag { value: string; source: "env" | "config" }

export function entryShadowRaw(env: NodeJS.ProcessEnv = process.env): EntryShadowFlag {
  const v = env[ENTRY_SHADOW_ENV];
  if (v !== undefined && v !== null) return { value: String(v), source: "env" };
  return { value: ENTRY_SHADOW_DEFAULTS[ENTRY_SHADOW_ENV] ?? "", source: "config" };
}

/** entry_shadow.parse_sleeves: off value -> none; "all" -> both; unknown tokens ignored. */
export function parseShadowSleeves(raw: string): Set<EntrySleeve> {
  const v = String(raw ?? "").trim().toLowerCase();
  const out = new Set<EntrySleeve>();
  if (OFF_VALUES.has(v)) return out;
  for (const tokRaw of v.replace(/;/g, ",").replace(/ /g, ",").split(",")) {
    const tok = tokRaw.trim();
    if (tok === "all") { out.add("stocks"); out.add("options"); }
    else if (ALIASES[tok]) out.add(ALIASES[tok]);
  }
  return out;
}

export function isEntryShadowed(sleeve: EntrySleeve, env: NodeJS.ProcessEnv = process.env): boolean {
  return parseShadowSleeves(entryShadowRaw(env).value).has(sleeve);
}

/** The /api/health bot block's `entry_shadow` (and the sleeves probe's): never gates serving. */
export function entryShadowStatus(env: NodeJS.ProcessEnv = process.env): { stocks: boolean; options: boolean; source: "env" | "config" } {
  const raw = entryShadowRaw(env);
  const on = parseShadowSleeves(raw.value);
  return { stocks: on.has("stocks"), options: on.has("options"), source: raw.source };
}

/** ET calendar date (YYYY-MM-DD) — entry_shadow.et_day. */
export function etDay(nowMs: number): string {
  return new Intl.DateTimeFormat("en-CA", { timeZone: "America/New_York", year: "numeric", month: "2-digit", day: "2-digit" })
    .format(new Date(nowMs));
}

/**
 * Sleeve+ticker pairs shadow-entered today (ET), in memory. Emulates "already
 * holding": a candidate already logged today is not re-evaluated every scan
 * (that would re-run the correlation/halt/options-chain calls and re-log the
 * same decision), and the per-ticker-per-day audit rate limit is the same set.
 * Python's shadow log is the persistent dedupe across restarts.
 */
/** What the Bot page shows for one would-be entry (GET /api/bot/status
 *  `shadow_entries_today`). side/notional are null when the Python half
 *  reported only the ticker (scan-side third leg, convexity puts, rolls). */
export interface ShadowEntryRow {
  time: string;            // ISO, when first shadowed today
  sleeve: EntrySleeve;
  ticker: string;
  side: string | null;
  notional: number | null; // USD the entry would have used; null = not reported
  source: string;
  instrument: string | null; // set only when it differs from ticker (ETF / OCC symbol)
}
/** Max rows the status route carries (newest first); `count` stays exact. */
export const SHADOW_ROWS_CAP = 50;

/** Shape one book entry for the page — pure, never throws on partial input. */
export function shadowEntryRow(
  sleeve: EntrySleeve, ticker: string, nowMs: number,
  detail?: { side?: unknown; notional?: unknown; source?: unknown; instrument?: unknown },
): ShadowEntryRow {
  const t = String(ticker).toUpperCase();
  const n = Number(detail?.notional);
  const inst = detail?.instrument ? String(detail.instrument).toUpperCase() : "";
  return {
    time: new Date(nowMs).toISOString(),
    sleeve,
    ticker: t,
    side: detail?.side ? String(detail.side).toLowerCase() : null,
    notional: Number.isFinite(n) && n > 0 ? Math.round(n * 100) / 100 : null,
    source: detail?.source ? String(detail.source) : "unknown",
    instrument: inst && inst !== t ? inst : null,
  };
}

export class ShadowEntryBook {
  private day = "";
  private keys = new Set<string>();
  private rows: ShadowEntryRow[] = [];
  private roll(nowMs: number): void {
    const d = etDay(nowMs);
    if (d !== this.day) { this.day = d; this.keys = new Set(); this.rows = []; }
  }
  has(sleeve: EntrySleeve, ticker: string, nowMs: number = Date.now()): boolean {
    this.roll(nowMs);
    return this.keys.has(`${sleeve}|${String(ticker).toUpperCase()}`);
  }
  /** true the first time this sleeve+ticker is added today (=> audit it).
   *  `detail` (side/notional/source/instrument) feeds the Bot page row. */
  add(sleeve: EntrySleeve, ticker: string, nowMs: number = Date.now(),
      detail?: { side?: unknown; notional?: unknown; source?: unknown; instrument?: unknown }): boolean {
    this.roll(nowMs);
    const k = `${sleeve}|${String(ticker).toUpperCase()}`;
    if (this.keys.has(k)) return false;
    this.keys.add(k);
    this.rows.push(shadowEntryRow(sleeve, ticker, nowMs, detail));
    // memory bound: the book's key set stays exact, rows keep the newest cap
    if (this.rows.length > SHADOW_ROWS_CAP) this.rows.splice(0, this.rows.length - SHADOW_ROWS_CAP);
    return true;
  }
  /** Today's (ET) would-be entries, newest first, capped; count is exact. */
  today(nowMs: number = Date.now()): { day: string; count: number; entries: ShadowEntryRow[] } {
    this.roll(nowMs);
    return { day: this.day, count: this.keys.size, entries: this.rows.slice().reverse() };
  }
}

/** One would-be entry Node logs (entry_shadow.build_entry's input). */
export interface NodeShadowEntry {
  sleeve: EntrySleeve;
  ticker: string;          // the underlying the bot scored (what the backfills measure)
  source: string;          // call site, e.g. "executeTrades.stock"
  side: string;
  qty: number;
  price: number;           // underlying price
  notional: number;
  score: number;
  instrument?: string;     // what would have been ordered (ETF symbol), default = ticker
  instrument_price?: number;
  relabel?: boolean;       // a fresh deep_score "taken" record exists for this ticker
  reason?: string;
}

export function shadowEntryAuditLine(e: Pick<NodeShadowEntry, "sleeve" | "ticker" | "source" | "side" | "qty" | "notional"> & { instrument?: string }): string {
  const forUnderlying = e.instrument && e.instrument !== e.ticker ? ` for ${e.ticker}` : "";
  return `${e.sleeve}: would ${String(e.side || "buy").toUpperCase()} ${e.qty} ${e.instrument || e.ticker}${forUnderlying} ~$${Math.round(e.notional || 0)} via ${e.source} — logged, no order (VOLTRADE_SHADOW_ENTRIES)`;
}
