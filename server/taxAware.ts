// taxAware.ts — GET /api/bot/tax (refocus spec 2026-11-05 §6; [RULE-REVIEW]
// 2026-11-06). The tax filter itself is Python (tax_filter.py): before every
// new entry it compares the trade's after-tax expected return with holding
// SPY for the same days after long-term tax, skips the ones that do not beat
// it by a small margin, and appends every decision to
// DATA_DIR/voltrade_tax_decisions.jsonl. This module only READS that log and
// shapes the route body (the UI contract in the build preamble):
//   {rates:{shortTerm, longTerm}, decisions:[{ts, ticker, side, allowed,
//    expectedPct, afterTaxPct, spyAfterTaxPct, holdingDays, reason, ...}],
//    harvest:[...], ltApproaching:[...]}
// plus a summary over the decisions, the live broker expectancy per sleeve
// next to the sizer's estimate (the filter prices trades with the Kelly
// sizer's stats; when live disagrees, the page must say so) and notes.
// Every number is an ESTIMATE at assumed rates; the paper account pays no tax.
// Pure except readTaxDecisions (one bounded file read).
import fs from "node:fs";
import path from "node:path";
import { DEFAULT_TAX_RATES } from "./taxLedger";
import type { EdgeLedger, EdgeRow } from "./edgeLedger";

export const TAX_DECISIONS_FILE = "voltrade_tax_decisions.jsonl";
export const TAX_DECISIONS_DEFAULT_LIMIT = 100;
export const TAX_DECISIONS_MAX_LIMIT = 500;
/** Bytes read from the end of the log — the newest rows (the writer rotates at 4 MB). */
export const TAX_DECISIONS_TAIL_BYTES = 1024 * 1024;

/** One decision as the route serves it. The first nine fields are the UI contract. */
export interface TaxDecision {
  ts: string;
  ticker: string;
  side: string;
  allowed: boolean;
  expectedPct: number | null;
  afterTaxPct: number | null;
  spyAfterTaxPct: number | null;
  holdingDays: number | null;
  reason: string;
  kind: string;
  instrument: string | null;
  source: string | null;
  /** the verdict before the off switch: false = the filter would skip it */
  wouldAllow: boolean;
  /** false while TAX_FILTER.ENABLED is off — every row is then allowed */
  enforced: boolean;
  /** false = the filter could not price it (no measured stats, a hedge, an error) */
  priced: boolean;
  costPct: number | null;
  edgePp: number | null;
  marginPp: number | null;
  bucket: string | null;
  strategy: string | null;
}

export interface HarvestRow { ticker: string; unrealized: number; replacement: string | null; washSaleUntil: string | null; reason: string }
export interface LtApproachingRow { ticker: string; daysToLongTerm: number; gain: number; reason: string }

export interface LiveExpectancy { n: number; winRatePct: number | null; expectancyPct: number | null; lastCloseAt: string | null }

export interface TaxBody {
  rates: { shortTerm: number; longTerm: number };
  decisions: TaxDecision[];
  harvest: HarvestRow[];
  ltApproaching: LtApproachingRow[];
  summary: { rows: number; allowed: number; skipped: number; wouldSkip: number; notPriced: number; since: string | null; enforced: boolean | null };
  /** the Python assumption the newest row was computed with (SPY's annual return, %) */
  spyAssumptionAnnualPct: number | null;
  /** the broker's own expectancy per closed lot (server/edgeLedger.ts), beside the sizer's estimate */
  liveExpectancy: { options: LiveExpectancy; stocks: LiveExpectancy; basis: string } | null;
  notes: string[];
}

const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const str = (v: unknown): string | null => (typeof v === "string" && v ? v : null);

/** One log line -> a route row; null for a torn or foreign line. */
export function taxDecisionRow(raw: unknown): TaxDecision | null {
  if (!raw || typeof raw !== "object") return null;
  const r = raw as Record<string, unknown>;
  const ts = str(r.ts), ticker = str(r.ticker);
  if (!ts || !ticker || typeof r.allowed !== "boolean") return null;
  return {
    ts, ticker, side: str(r.side) ?? "buy", allowed: r.allowed,
    expectedPct: num(r.expectedPct), afterTaxPct: num(r.afterTaxPct), spyAfterTaxPct: num(r.spyAfterTaxPct),
    holdingDays: num(r.holdingDays), reason: str(r.reason) ?? "",
    kind: str(r.kind) ?? "entry", instrument: str(r.instrument), source: str(r.source),
    wouldAllow: typeof r.wouldAllow === "boolean" ? r.wouldAllow : r.allowed,
    enforced: r.enforced !== false, priced: r.priced !== false,
    costPct: num(r.costPct), edgePp: num(r.edgePp), marginPp: num(r.marginPp),
    bucket: str(r.bucket), strategy: str(r.strategy),
  };
}

/** Parse JSONL text (oldest first) into rows, newest first, at most `limit`. */
export function parseTaxDecisions(text: string, limit: number): { rows: TaxDecision[]; spyAssumption: number | null } {
  const lines = text.split("\n");
  const rows: TaxDecision[] = [];
  let spyAssumption: number | null = null;
  for (let i = lines.length - 1; i >= 0 && rows.length < limit; i--) {
    const line = lines[i].trim();
    if (!line) continue;
    let parsed: unknown;
    try { parsed = JSON.parse(line); } catch { continue; } // a torn tail line or the cut first line of the window
    const row = taxDecisionRow(parsed);
    if (!row) continue;
    if (spyAssumption === null) spyAssumption = num((parsed as Record<string, unknown>).spyAssumptionAnnualPct);
    rows.push(row);
  }
  return { rows, spyAssumption };
}

/** ?limit -> 1..TAX_DECISIONS_MAX_LIMIT, default TAX_DECISIONS_DEFAULT_LIMIT. */
export function taxLimit(q: unknown): number {
  const n = Math.floor(Number(q));
  return Number.isFinite(n) && n >= 1 ? Math.min(n, TAX_DECISIONS_MAX_LIMIT) : TAX_DECISIONS_DEFAULT_LIMIT;
}

/** The newest rows of the decision log in `dir` (the last TAX_DECISIONS_TAIL_BYTES only). */
export function readTaxDecisions(dir: string, limit: number): { rows: TaxDecision[]; spyAssumption: number | null; error: string | null } {
  const p = path.join(dir, TAX_DECISIONS_FILE);
  let fd: number | null = null;
  try {
    fd = fs.openSync(p, "r");
    const size = fs.fstatSync(fd).size;
    const len = Math.min(size, TAX_DECISIONS_TAIL_BYTES);
    const buf = Buffer.alloc(len);
    fs.readSync(fd, buf, 0, len, size - len);
    return { ...parseTaxDecisions(buf.toString("utf8"), limit), error: null };
  } catch (err) {
    const code = (err as NodeJS.ErrnoException)?.code;
    // no file yet = no decision made yet; anything else is reported, not hidden
    return { rows: [], spyAssumption: null, error: code === "ENOENT" ? null : String(err instanceof Error ? err.message : err).slice(0, 200) };
  } finally {
    if (fd !== null) fs.closeSync(fd);
  }
}

function liveRow(r: EdgeRow | undefined): LiveExpectancy {
  return { n: r?.n ?? 0, winRatePct: r?.winRate ?? null, expectancyPct: r?.expectancyPct ?? null, lastCloseAt: r?.lastCloseAt ?? null };
}

export interface TaxBodyInput {
  rows: TaxDecision[];
  spyAssumption: number | null;
  readError: string | null;
  edge: EdgeLedger | null;
  harvest?: HarvestRow[];
  ltApproaching?: LtApproachingRow[];
  extraNotes?: string[];
}

/** Pure: the route body. */
export function taxBody(input: TaxBodyInput): TaxBody {
  const { rows } = input;
  const skipped = rows.filter((r) => !r.allowed).length;
  const notes = [
    "estimates at assumed rates (24 % short-term, 15 % long-term federal, no state, no NIIT); the paper account pays no tax",
    "each entry's expected return is the Kelly sizer's own per-trade expectancy for its bucket (win rate x average win - loss rate x average loss) over the mean holding period of the sample it was measured on; the composite score is not used (the shadow study found it anti-predictive)",
    "SPY leg: an assumed long-run return (TAX_FILTER.SPY_EXPECTED_RETURN_ANNUAL_PCT, not a forecast) for the same days, taxed at the long-term rate because buy-and-hold defers its tax",
    "costs: the repo's round-trip model is subtracted on top of stats that are already net of their own trades' costs, so the filter errs strict",
    "rows are one per ticker, instrument and verdict per day; skipped stock entries also appear as SIZE-BLOCK audit rows, skipped option opens as T-FAIL / OPTIONS-SKIP rows",
  ];
  if (input.edge) notes.push("liveExpectancy: the broker's realized expectancy per closed lot (% of the premium for options, % of cost for stocks) — when it disagrees with the sizer's estimate the filter still uses the sizer's; that disagreement is the sizer's open question");
  if (input.readError) notes.push(`decision log unreadable: ${input.readError}`);
  if (!rows.length && !input.readError) notes.push("no tax decision logged yet");
  for (const n of input.extraNotes ?? []) notes.push(n);
  return {
    rates: { shortTerm: DEFAULT_TAX_RATES.shortTerm, longTerm: DEFAULT_TAX_RATES.longTerm },
    decisions: rows,
    harvest: input.harvest ?? [],
    ltApproaching: input.ltApproaching ?? [],
    summary: {
      rows: rows.length, allowed: rows.length - skipped, skipped,
      wouldSkip: rows.filter((r) => !r.wouldAllow).length,
      notPriced: rows.filter((r) => !r.priced).length,
      since: rows.length ? rows[rows.length - 1].ts : null,
      enforced: rows.length ? rows[0].enforced : null,
    },
    spyAssumptionAnnualPct: input.spyAssumption,
    liveExpectancy: input.edge ? { options: liveRow(input.edge.options), stocks: liveRow(input.edge.stocks), basis: "per closed lot at the broker (server/edgeLedger.ts)" } : null,
    notes,
  };
}
