/**
 * xbrlFramesCore.ts — the pure half of the SEC XBRL fundamentals record
 * (server/xbrlFrames.ts does the reading and writing).
 *
 * WHAT THE SOURCE IS. data.sec.gov/api/xbrl/frames/<taxonomy>/<tag>/<unit>/
 * <period>.json returns one fact per company for one reporting concept and
 * one calendar period: the number the company tagged in its own 10-Q or
 * 10-K. Keyless, public domain; SEC asks for a descriptive User-Agent and at
 * most 10 requests a second.
 *
 * WHY WE RECORD IT INSTEAD OF ASKING WHEN NEEDED (measured 2026-11-21, 72
 * frames, 241,402 facts): a frame keeps ONE fact per company and replaces it
 * whenever a later filing reports the same period again. For the quarter
 * ended September 2025, 86% of the net-income facts already carry a 2026
 * accession number: the comparative column of this year's 10-Q has replaced
 * the filing that first reported the number. So a frame read today cannot
 * say what was known a year ago, or when. A nightly read that keeps the
 * first value with the night it appeared, and every later change, can.
 *
 * WHAT A ROW CLAIMS. `seen` is the time of OUR nightly read that first held
 * the value: a one-day-resolution observation, not SEC's acceptance time
 * (that is in the filing the accession number names). Rows written on the
 * first read of a frame carry `backfill: true`: for those, `seen` is the day
 * recording began and says nothing about when the number was published.
 *
 * RAW record. No score, no ranking, no predictive claim.
 */

export type FrameKind = "duration" | "instant";

export interface XbrlConcept {
  taxonomy: "us-gaap" | "dei";
  tag: string;
  unit: "USD" | "shares";
  kind: FrameKind;
  /** the word the page shows */
  label: string;
}

/** The fixed roster. Changing it changes what is recorded from that night
 *  on; a concept added later starts with a backfill-marked first read. */
export const XBRL_CONCEPTS: readonly XbrlConcept[] = [
  { taxonomy: "us-gaap", tag: "Revenues", unit: "USD", kind: "duration", label: "Revenue" },
  { taxonomy: "us-gaap", tag: "RevenueFromContractWithCustomerExcludingAssessedTax", unit: "USD", kind: "duration", label: "Revenue from customers" },
  { taxonomy: "us-gaap", tag: "CostOfRevenue", unit: "USD", kind: "duration", label: "Cost of revenue" },
  { taxonomy: "us-gaap", tag: "GrossProfit", unit: "USD", kind: "duration", label: "Gross profit" },
  { taxonomy: "us-gaap", tag: "OperatingIncomeLoss", unit: "USD", kind: "duration", label: "Operating income" },
  { taxonomy: "us-gaap", tag: "NetIncomeLoss", unit: "USD", kind: "duration", label: "Net income" },
  { taxonomy: "us-gaap", tag: "Assets", unit: "USD", kind: "instant", label: "Total assets" },
  { taxonomy: "us-gaap", tag: "Liabilities", unit: "USD", kind: "instant", label: "Total liabilities" },
  { taxonomy: "us-gaap", tag: "StockholdersEquity", unit: "USD", kind: "instant", label: "Shareholders' equity" },
  { taxonomy: "us-gaap", tag: "CashAndCashEquivalentsAtCarryingValue", unit: "USD", kind: "instant", label: "Cash" },
  { taxonomy: "us-gaap", tag: "LongTermDebtNoncurrent", unit: "USD", kind: "instant", label: "Long-term debt" },
  { taxonomy: "dei", tag: "EntityCommonStockSharesOutstanding", unit: "shares", kind: "instant", label: "Shares outstanding" },
] as const;

/** Calendar quarters read each night, newest first: the quarter in progress
 *  (companies whose fiscal quarter ends early report into it before it
 *  ends) and the six before it, so a late filer or a restatement inside
 *  about eighteen months is seen. */
export const QUARTERS_BACK = 6;
/** Full calendar years read for the duration concepts: a 10-K reports the
 *  year, and most companies never tag a three-month fourth quarter
 *  (measured: 380 revenue facts in CY2025Q4 against 1,932 in CY2025Q3). */
export const YEARS_BACK = 2;

/** A read that returns fewer than this share of the companies the last read
 *  held is not believed (a cut-short or partial answer would otherwise be
 *  recorded as thousands of dropped facts). Small frames are exempt. */
export const SHRINK_GUARD_RATIO = 0.5;
export const SHRINK_GUARD_MIN_PREV = 100;

export interface FrameRef { concept: XbrlConcept; period: string; key: string; url: string }

/** `CY2026Q3` (three months) or `CY2026Q3I` (the balance at its end). */
export function quarterPeriods(now: Date, kind: FrameKind, back: number = QUARTERS_BACK): string[] {
  let y = now.getUTCFullYear();
  let q = Math.floor(now.getUTCMonth() / 3) + 1;
  const out: string[] = [];
  for (let i = 0; i <= back; i++) {
    out.push(`CY${y}Q${q}${kind === "instant" ? "I" : ""}`);
    q -= 1;
    if (q === 0) { q = 4; y -= 1; }
  }
  return out;
}

/** `CY2025`: the last `back` full calendar years. */
export function yearPeriods(now: Date, back: number = YEARS_BACK): string[] {
  const y = now.getUTCFullYear();
  const out: string[] = [];
  for (let i = 1; i <= back; i++) out.push(`CY${y - i}`);
  return out;
}

export function frameKey(c: XbrlConcept, period: string): string {
  return `${c.taxonomy}/${c.tag}/${c.unit}/${period}`;
}

export function frameUrl(c: XbrlConcept, period: string): string {
  return `https://data.sec.gov/api/xbrl/frames/${frameKey(c, period)}.json`;
}

/** Every frame one night reads, in a fixed order. */
export function frameRoster(now: Date, concepts: readonly XbrlConcept[] = XBRL_CONCEPTS): FrameRef[] {
  const out: FrameRef[] = [];
  for (const concept of concepts) {
    const periods = quarterPeriods(now, concept.kind);
    if (concept.kind === "duration") periods.push(...yearPeriods(now));
    for (const period of periods) out.push({ concept, period, key: frameKey(concept, period), url: frameUrl(concept, period) });
  }
  return out;
}

/** The snapshot file's name for a frame: letters, digits and underscores. */
export function frameFileName(key: string): string {
  return key.replace(/[^A-Za-z0-9]+/g, "_");
}

export interface FrameFact { cik: number; accn: string; end: string; start?: string; val: number }
export interface FrameEntity { name: string; loc: string | null }
export interface ParsedFrame { facts: FrameFact[]; entities: Map<number, FrameEntity>; rejected: number; duplicates: number }

const DAY_RE = /^\d{4}-\d{2}-\d{2}$/;
const ACCN_RE = /^\d{10}-\d{2}-\d{6}$/;

/** Reads SEC's answer. A row without a whole-number CIK, a well-formed
 *  accession number, an end date and a finite value is counted in
 *  `rejected` and left out. A second row for a company already read is
 *  counted in `duplicates` and left out (SEC documents one fact per company
 *  per frame; the first is kept so a repeat cannot read as a revision). */
export function parseFrame(body: unknown): ParsedFrame {
  const facts: FrameFact[] = [];
  const entities = new Map<number, FrameEntity>();
  let rejected = 0;
  let duplicates = 0;
  const data = (body as { data?: unknown } | null)?.data;
  if (!Array.isArray(data)) throw new Error("xbrl frame: answer has no data list");
  const seen = new Set<number>();
  for (const raw of data as Array<Record<string, unknown>>) {
    const cik = raw?.cik;
    const accn = raw?.accn;
    const end = raw?.end;
    const val = raw?.val;
    if (typeof cik !== "number" || !Number.isInteger(cik) || cik <= 0
      || typeof accn !== "string" || !ACCN_RE.test(accn)
      || typeof end !== "string" || !DAY_RE.test(end)
      || typeof val !== "number" || !Number.isFinite(val)) { rejected++; continue; }
    if (seen.has(cik)) { duplicates++; continue; }
    seen.add(cik);
    const fact: FrameFact = { cik, accn, end, val };
    if (typeof raw.start === "string" && DAY_RE.test(raw.start)) fact.start = raw.start;
    facts.push(fact);
    if (typeof raw.entityName === "string" && raw.entityName) {
      entities.set(cik, { name: raw.entityName, loc: typeof raw.loc === "string" && raw.loc ? raw.loc : null });
    }
  }
  return { facts, entities, rejected, duplicates };
}

/** What the last read held for one frame: CIK -> [value, accession, end]. */
export type FrameSnapshot = Record<string, [number, string, string]>;

export type RowKind = "first" | "revised" | "dropped";

/** One line of the record. */
export interface XbrlRow {
  /** time of the nightly read that first held this value (ISO, UTC) */
  seen: string;
  /** taxonomy/tag/unit/period */
  frame: string;
  cik: number;
  kind: RowKind;
  /** the filing SEC's frame named on that night */
  accn: string;
  end: string;
  start?: string;
  /** the value as tagged; null on a dropped row */
  val: number | null;
  /** first read of this frame: `seen` is when recording began, not when the value was published */
  backfill?: true;
  /** revised and dropped rows: what the read before held */
  prev_val?: number;
  prev_accn?: string;
  prev_end?: string;
}

export interface DiffStats {
  held: number;
  first: number;
  revised: number;
  dropped: number;
  /** same value and period, named by a later filing: counted, not written */
  refiled: number;
  backfill: boolean;
}

export interface FrameDiff { rows: XbrlRow[]; next: FrameSnapshot; stats: DiffStats }

export class FrameShrunkError extends Error {
  constructor(public readonly frame: string, public readonly prev: number, public readonly now: number) {
    super(`xbrl frame ${frame}: ${now} companies against ${prev} at the last read; not recorded`);
    this.name = "FrameShrunkError";
  }
}

/**
 * Compares tonight's frame with the last read.
 *   - no last read (prev null): every fact is a `first` row marked backfill;
 *   - a company not held before: `first`;
 *   - a different value or period end: `revised`, with what was held;
 *   - the same value named by a later filing: nothing written, counted;
 *   - a company held before and absent now: `dropped`.
 * Throws FrameShrunkError when the frame lost more than half its companies.
 */
export function diffFrame(frame: string, prev: FrameSnapshot | null, facts: readonly FrameFact[], seen: string): FrameDiff {
  const backfill = prev === null;
  const before = prev ?? {};
  const prevCount = Object.keys(before).length;
  if (prevCount >= SHRINK_GUARD_MIN_PREV && facts.length < prevCount * SHRINK_GUARD_RATIO) {
    throw new FrameShrunkError(frame, prevCount, facts.length);
  }
  const rows: XbrlRow[] = [];
  const next: FrameSnapshot = {};
  const stats: DiffStats = { held: facts.length, first: 0, revised: 0, dropped: 0, refiled: 0, backfill };
  for (const f of facts) {
    const id = String(f.cik);
    next[id] = [f.val, f.accn, f.end];
    const was = before[id];
    if (!was) {
      const row: XbrlRow = { seen, frame, cik: f.cik, kind: "first", accn: f.accn, end: f.end, val: f.val };
      if (f.start) row.start = f.start;
      if (backfill) row.backfill = true;
      rows.push(row);
      stats.first++;
    } else if (was[0] !== f.val || was[2] !== f.end) {
      const row: XbrlRow = { seen, frame, cik: f.cik, kind: "revised", accn: f.accn, end: f.end, val: f.val, prev_val: was[0], prev_accn: was[1], prev_end: was[2] };
      if (f.start) row.start = f.start;
      rows.push(row);
      stats.revised++;
    } else if (was[1] !== f.accn) {
      stats.refiled++;
    }
  }
  for (const id of Object.keys(before)) {
    if (next[id]) continue;
    const was = before[id];
    rows.push({ seen, frame, cik: Number(id), kind: "dropped", accn: was[1], end: was[2], val: null, prev_val: was[0], prev_accn: was[1], prev_end: was[2] });
    stats.dropped++;
  }
  return { rows, next, stats };
}

/** The ET calendar day of an instant, as YYYY-MM-DD. */
export function etDay(at: Date): string {
  return new Intl.DateTimeFormat("en-CA", { timeZone: "America/New_York", year: "numeric", month: "2-digit", day: "2-digit" }).format(at);
}

/** The ET hour (0-23) of an instant. */
export function etHour(at: Date): number {
  return Number(new Intl.DateTimeFormat("en-US", { timeZone: "America/New_York", hour: "numeric", hourCycle: "h23" }).format(at));
}

/** Whether the nightly read should start now: inside the window, and not
 *  already finished or tried for this ET day. */
export function dueNow(at: Date, lastDay: string | null, startHourEt: number, endHourEt: number): boolean {
  const h = etHour(at);
  if (h < startHourEt || h >= endHourEt) return false;
  return lastDay !== etDay(at);
}

/** SEC's filing index page for an accession number. */
export function filingUrl(cik: number, accn: string): string {
  return `https://www.sec.gov/Archives/edgar/data/${cik}/${accn.replace(/-/g, "")}/${accn}-index.htm`;
}
