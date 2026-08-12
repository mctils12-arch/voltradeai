/**
 * gnssIntegrityQuery.ts — GNSS integrity passthrough, PHASE 3: the read/query
 * path. Phase 1 (2026-08-11) validated the ground-truth reality of the signal
 * against live snapshots (research/experiments.md); Phase 2 (2026-08-11,
 * v1.0.662) built the writer so the aircraft archive stops discarding
 * nic/nac_p/.../pos_type. Neither phase gave the archive a way to be QUERIED
 * in aggregate — every check so far re-derived counts by hand from raw
 * /api/diag/archive rows. This is the compiled tool (EDGE DOCTRINE #3:
 * "never analyze the same thing twice with reasoning — the second occurrence
 * becomes a script the bot runs free forever").
 *
 * SHAPE, dictated by the 2026-08-11 adversarial-verification verdict (the
 * strongest find of that scan, research/open_questions.md): a bare
 * "X% of rows have nic==0" is not trustworthy on its own — that same scan
 * found 91.7-96.4% of nic==0 rows in the anomalous region carry `mlat`
 * fields, meaning readsb's ground network COMPUTED nic/rc for that row
 * rather than the aircraft broadcasting it. Any aggregate that doesn't
 * split by origin silently conflates "the aircraft told us its GPS is
 * degraded" with "the ground station's own multilateration solver reported
 * a wide radius" — two different phenomena. This module therefore:
 *   1. NEVER returns a rate — only numerator (n_zero) and denominator
 *      (n_total) together; the caller computes and displays the ratio,
 *      never the reverse.
 *   2. ALWAYS splits by origin (broadcast | ground | mode_s | unknown, via
 *      the single-source-of-truth originOfPosType in datacoreArchive.ts).
 *   3. ALWAYS splits by altitude band — the verified signature is a
 *      line-of-sight gradient (0% on the ground rising to the majority at
 *      cruise), so a pooled altitude figure hides the one pattern that
 *      discriminates real GPS interference from a ground-clutter artifact.
 *   4. Excludes rows with no `ni` reading from every denominator rather
 *      than treating "no field" as "field is zero" — silence is not a
 *      measurement.
 *
 * Pure aggregator (aggregateGnssIntegrity) + a thin async multi-day reader
 * (readGnssIntegrityWindow) built on the existing readArchiveDayEvenSample/
 * streamJsonlLines primitives — no new I/O format. Backs the
 * "gnss_integrity" diag probe (server/bot.ts / diag.ts).
 *
 * READER CHOICE (2026-08-12): uses readArchiveDayEvenSample, not the plain
 * readArchiveDay every other archive probe uses — a bbox-scoped query over
 * this feed's real daily volume needs a sample spread across the WHOLE day,
 * not the first N chronological hours (see readArchiveDayEvenSample's own
 * doc comment for the live symptom this fixes: a Baltic bbox silently
 * returning zero cells because a straight per-day row cap never got past
 * the day's early UTC hours, when Baltic local traffic is overnight-quiet).
 *
 * BBOX PUSHDOWN (2026-08-12, same session): when a bbox is given, it is
 * now passed to the reader as a `rowFilter`, not applied only afterward in
 * aggregateGnssIntegrity. The FIRST live run after the even-sampling fix
 * above still came back thin for a real region (Baltic: 20 broadcast/
 * cruise rows vs. 3157 for a much larger US+Europe control box, same row
 * budget) because a small bbox is a small slice of GLOBAL traffic — most
 * of the per-file budget was being spent on rows the aggregator would
 * throw away. Filtering during the read means the SAME row budget is
 * spent entirely on rows that end up counted, without reading any more of
 * the underlying files than before.
 */
import { readArchiveDayEvenSample, originOfPosType, type ArchiveOrigin } from "./datacoreArchive";

/** Minimal shape this module reads off an archived aircraft JSONL row —
 *  the short-key schema `aircraftIntegrityFields`/`archiveAircraft` write
 *  (datacoreArchive.ts). Extra fields pass through untouched. */
export interface ArchiveAircraftRow {
  i?: string;      // icao24
  la?: number;
  lo?: number;
  al?: number;     // altitude, meters (archive stores rounded meters)
  g?: boolean;      // on_ground
  ni?: number;      // nic
  pt?: string;      // pos_type (position source)
  [k: string]: unknown;
}

export interface Bbox { lamin: number; lamax: number; lomin: number; lomax: number }

export type AltitudeBand = "ground" | "low" | "mid" | "cruise" | "unknown";

/** Band thresholds in meters (archive unit). Comments carry the feet
 *  convention every prior session's finding was stated in, so a reader
 *  cross-checking against research/open_questions.md doesn't have to
 *  convert by hand: low < 10,000 ft (3048 m); mid 10,000-25,000 ft
 *  (3048-7620 m); cruise >= 25,000 ft (7620 m) — the threshold the
 *  2026-08-11 verification used for its Baltic/control comparison. */
export const LOW_ALT_MAX_M = 3048;
export const MID_ALT_MAX_M = 7620;

export function bandForRow(r: ArchiveAircraftRow): AltitudeBand {
  if (r.g) return "ground";
  if (typeof r.al !== "number" || !Number.isFinite(r.al)) return "unknown";
  if (r.al < LOW_ALT_MAX_M) return "low";
  if (r.al < MID_ALT_MAX_M) return "mid";
  return "cruise";
}

export function inBbox(r: ArchiveAircraftRow, bbox?: Bbox): boolean {
  if (!bbox) return true;
  if (typeof r.la !== "number" || typeof r.lo !== "number") return false;
  return r.la >= bbox.lamin && r.la <= bbox.lamax && r.lo >= bbox.lomin && r.lo <= bbox.lomax;
}

export interface IntegrityCell {
  band: AltitudeBand;
  origin: ArchiveOrigin;
  /** Denominator: rows in this band/origin with a present `ni` reading. */
  n_total: number;
  /** Numerator: of those, rows reporting nic==0 (zero containment). */
  n_zero: number;
  /** Informational only (never gates a count): distinct icao24s contributing. */
  distinct_airframes: number;
}

/**
 * Aggregate raw archived aircraft rows into band x origin cells. Rows with
 * no `ni` field are skipped entirely (excluded from every denominator —
 * absence is not a zero). `bbox`, when given, restricts to a lat/lon box
 * (e.g. a Baltic candidate region vs. a control region) so the caller can
 * run the SAME query twice and compare cells directly.
 */
export function aggregateGnssIntegrity(rows: ArchiveAircraftRow[], bbox?: Bbox): IntegrityCell[] {
  const cells = new Map<string, { n_total: number; n_zero: number; airframes: Set<string> }>();
  for (const r of rows) {
    if (typeof r.ni !== "number") continue;
    if (!inBbox(r, bbox)) continue;
    const band = bandForRow(r);
    const origin = originOfPosType(r.pt);
    const key = `${band}|${origin}`;
    let c = cells.get(key);
    if (!c) { c = { n_total: 0, n_zero: 0, airframes: new Set() }; cells.set(key, c); }
    c.n_total++;
    if (r.ni === 0) c.n_zero++;
    if (r.i) c.airframes.add(r.i);
  }
  return Array.from(cells.entries())
    .map(([key, c]) => {
      const [band, origin] = key.split("|") as [AltitudeBand, ArchiveOrigin];
      return { band, origin, n_total: c.n_total, n_zero: c.n_zero, distinct_airframes: c.airframes.size };
    })
    .sort((a, b) => (a.band === b.band ? a.origin.localeCompare(b.origin) : a.band.localeCompare(b.band)));
}

export interface GnssIntegrityWindowResult {
  cells: IntegrityCell[];
  days_read: string[];
  days_missing: string[];
  rows_scanned: number;
  /** True if ANY day's read hit its row cap — cells may undercount that day. */
  truncated: boolean;
}

/**
 * Multi-day read + aggregate over the aircraft archive, built on the
 * existing readArchiveDay/streamJsonlLines primitives (no new archive I/O).
 * `days` are read independently and unioned; a day with no archive
 * directory/files for that date is reported in `days_missing` rather than
 * silently dropped, so a caller can tell "no GPS anomaly" apart from
 * "no data was read".
 */
export async function readGnssIntegrityWindow(
  days: string[], bbox?: Bbox, baseDir?: string, perDayLimit = 20_000,
): Promise<GnssIntegrityWindowResult> {
  const allRows: ArchiveAircraftRow[] = [];
  const daysRead: string[] = [];
  const daysMissing: string[] = [];
  let truncated = false;
  const rowFilter = bbox ? (r: ArchiveAircraftRow) => inBbox(r, bbox) : undefined;
  for (const day of days) {
    const result = await readArchiveDayEvenSample("aircraft", day, baseDir, perDayLimit, rowFilter);
    if (!result || result.files.length === 0) { daysMissing.push(day); continue; }
    daysRead.push(day);
    if (result.truncated) truncated = true;
    for (const row of result.rows) allRows.push(row as ArchiveAircraftRow);
  }
  return {
    cells: aggregateGnssIntegrity(allRows, bbox),
    days_read: daysRead,
    days_missing: daysMissing,
    rows_scanned: allRows.length,
    truncated,
  };
}
