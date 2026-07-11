/**
 * dataQuality.ts — the DATA QUALITY GATE for every /data layer
 * (human-directed 2026-07-11, research/location_context_engine.md).
 *
 * WHY: the Location Context Engine cross-joins many layers into one reading.
 * A single inaccurate input silently skewing that join is the worst failure
 * mode — worse than a missing layer. This is Priority 2 (protect the integrity
 * of learning) + ROOT VALIDATION LADDER gate 1 (DATA: "verified before anything
 * downstream"), made a reusable per-record gate.
 *
 * Contract: validators return the record split into CLEAN (passed) and SUSPECT
 * (failed, with reasons). A failing record is QUARANTINED — never fed to the
 * dossier/graph as if clean. Callers archive/serve clean rows and surface the
 * suspect count honestly; they never silently drop-and-forget.
 *
 * Pure + dependency-free so it unit-tests without network or fixtures and can
 * wrap any layer's ingest.
 */

/** A single failed check on a record. */
export interface QualityIssue { field: string; rule: string; detail?: string }

/** One field's expected shape. Only the checks you set are enforced. */
export interface FieldRule {
  required?: boolean;
  /** numeric lower/upper physical bounds (inclusive) */
  min?: number;
  max?: number;
  /** allowed exact values (enum) */
  oneOf?: Array<string | number>;
  /** regex the string value must match */
  pattern?: RegExp;
  /** treat as a number even if arriving as a numeric string */
  numeric?: boolean;
}

export type Schema = Record<string, FieldRule>;

const asNumber = (v: any): number | null => {
  if (v == null || v === "") return null;
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? n : null;
};

/** Validate one record against a schema. Returns the issues found (empty = ok). */
export function validateRecord(rec: Record<string, any>, schema: Schema): QualityIssue[] {
  const issues: QualityIssue[] = [];
  for (const [field, rule] of Object.entries(schema)) {
    const raw = rec[field];
    const present = raw != null && raw !== "";
    if (!present) {
      if (rule.required) issues.push({ field, rule: "required" });
      continue;
    }
    if (rule.numeric || rule.min != null || rule.max != null) {
      const n = asNumber(raw);
      if (n == null) { issues.push({ field, rule: "numeric", detail: String(raw) }); continue; }
      if (rule.min != null && n < rule.min) issues.push({ field, rule: "min", detail: `${n} < ${rule.min}` });
      if (rule.max != null && n > rule.max) issues.push({ field, rule: "max", detail: `${n} > ${rule.max}` });
    }
    if (rule.oneOf && !rule.oneOf.includes(raw)) issues.push({ field, rule: "oneOf", detail: String(raw) });
    if (rule.pattern && !rule.pattern.test(String(raw))) issues.push({ field, rule: "pattern", detail: String(raw) });
  }
  return issues;
}

/** Valid earth coordinate: in range AND not the (0,0) "null island" that a
 *  dropped/failed geocode collapses to (a classic silent-corruption tell). */
export function coordValid(lat: any, lon: any): boolean {
  const la = asNumber(lat), lo = asNumber(lon);
  if (la == null || lo == null) return false;
  if (la < -90 || la > 90 || lo < -180 || lo > 180) return false;
  if (Math.abs(la) < 1e-6 && Math.abs(lo) < 1e-6) return false; // null island
  return true;
}

export interface Partition<T> {
  clean: T[];
  suspect: Array<{ record: T; issues: QualityIssue[] }>;
}

/** Split records into clean vs quarantined-suspect by a per-record validator.
 *  The suspect list is never discarded silently — callers report its size. */
export function partitionValid<T>(records: T[], validate: (r: T) => QualityIssue[]): Partition<T> {
  const clean: T[] = [];
  const suspect: Array<{ record: T; issues: QualityIssue[] }> = [];
  for (const r of records) {
    const issues = validate(r as any);
    if (issues.length === 0) clean.push(r);
    else suspect.push({ record: r, issues });
  }
  return { clean, suspect };
}

export type Freshness = "fresh" | "stale" | "unknown";

/** Staleness is a DATA ERROR, not a cosmetic detail: a value older than its
 *  cadence budget is `stale` and must be surfaced, never served as current. */
export function freshnessStatus(sourceDateMs: number | null | undefined, budgetMs: number, nowMs: number): Freshness {
  if (sourceDateMs == null || !Number.isFinite(sourceDateMs)) return "unknown";
  return nowMs - sourceDateMs <= budgetMs ? "fresh" : "stale";
}

/** Flag a value that is an implausible step-change vs the layer's own recent
 *  history — the classic corrupt-feed signature. Returns true = OUTLIER (hold
 *  for review before it can move a combined reading). Needs a few peers to mean
 *  anything; too-thin history returns false (withhold judgement, never guess). */
export function isOutlier(value: number, recent: number[], z = 5, minPeers = 5): boolean {
  if (recent.length < minPeers) return false;
  const mean = recent.reduce((a, b) => a + b, 0) / recent.length;
  const variance = recent.reduce((a, b) => a + (b - mean) ** 2, 0) / recent.length;
  const sd = Math.sqrt(variance);
  if (sd === 0) return value !== mean; // a flatline that suddenly moves is suspect
  return Math.abs(value - mean) / sd > z;
}

/** Summary a layer attaches to its response so the UI shows how trusted +
 *  fresh each ingredient is (PREMIUM EXPERIENCE STANDARD (c)). */
export interface LayerHealth {
  source: string;
  clean: number;
  suspect: number;
  freshness: Freshness;
  source_date?: string;
}

export function layerHealth(source: string, part: Partition<any>, freshness: Freshness = "unknown", sourceDate?: string): LayerHealth {
  return { source, clean: part.clean.length, suspect: part.suspect.length, freshness, source_date: sourceDate };
}
