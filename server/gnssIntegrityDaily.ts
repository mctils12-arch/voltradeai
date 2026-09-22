/**
 * gnssIntegrityDaily.ts — permanent daily archive for the gnss_integrity_adsb
 * root, folded from raw aircraft hour files BEFORE datacoreArchive's generic
 * rollupOldDaysAsync deletes them past RAW_RETENTION_DAYS (30).
 *
 * BUG THIS FIXES (found this session while trying to build a GNSS-jamming ×
 * market-return correlation probe, research/open_questions.md's FUSION
 * HYPOTHESES / ACTIVE ANGLE-HUNTING — live-verified against production
 * 2026-09-22, not assumed): `rollupOldDaysAsync`'s per-entity daily-track
 * summary (`accumulateTrackLine` in datacoreArchive.ts) keeps only
 * `{i, n, t0, t1, bbox, pl}` — position/time/polyline. It does NOT carry the
 * `ni` (nic) / `pt` (pos_type) / `al` (altitude) fields this root's entire
 * signal is built from. So once a raw aircraft hour file ages past 30 days
 * and gets rolled up, its GNSS-integrity information is gone forever — the
 * archive can structurally never exceed ~30 days of usable depth for this
 * root, no matter how long the system runs. Confirmed live: `/api/diag/
 * gnss_integrity?days=2026-08-11..2026-08-23` (all recorded as the writer's
 * "live since" date in gnssIntegritySignal.ts) returned `days_missing` for
 * every one of those days on 2026-09-22, while 2026-08-24 onward (within the
 * 30-day window) still read fine. This is the SAME bug shape the 2026-08-02
 * fleet-utilization session found and fixed for aircraft hours in general
 * (`preserveWeeklyBeforeRollup`, server/fleetUtilization.ts) — that fix does
 * NOT cover this root, because it only preserves per-owner session/hour
 * deltas, never the integrity/altitude fields this root reads.
 *
 * PRACTICAL IMPACT TODAY: the already-shipped, already-gate2_pass LIVE
 * computation (`computeGnssIntegritySignal`, gnssIntegritySignal.ts) is NOT
 * broken by this — it only ever reads a rolling <=21-day window, which is
 * still fully within the 30-day raw-retention floor. The gap only bites any
 * future use that needs MORE than ~30 days of history — e.g. a market-return
 * correlation study (research/open_questions.md's new 2026-09-22 entry),
 * which needs a real multi-month daily series to have any statistical power.
 * This module is that fix, scoped exactly like fleet-utilization's: fold a
 * small permanent per-day summary (band x origin cells, for the SAME two
 * canonical bboxes gate 2 already validated — CANDIDATE_BBOX/CONTROL_BBOX,
 * gnssIntegritySignal.ts) before the raw file is deleted, so depth
 * accumulates going forward without ever growing the raw retention window
 * itself (no change to RAW_RETENTION_DAYS, no new disk-growth risk — see
 * MEMORY LAW's "the archive grows forever; the render must not" spirit,
 * applied here to a tiny (~60 numbers/day) derived summary, not raw rows).
 *
 * HONEST COST, stated up front (same as the fleet-utilization precedent):
 * every day already rolled up before this fix ships (2026-08-11 through
 * roughly 2026-08-23, per the live check above) is UNRECOVERABLE — this
 * closes the gap going FORWARD only. A future correlation probe should
 * verify the accumulated day count live before trusting it, not assume the
 * calendar date alone implies depth.
 *
 * WHAT IS NOT DONE HERE (deliberately, one logical change per PR): this
 * module only WRITES the permanent archive. It does not expose a reader
 * for it outside this process (no new diag probe, no change to
 * computeGnssIntegritySignal's live rolling-window computation, which stays
 * byte-for-byte behaviorally unchanged — MEASUREMENT INTEGRITY: a metric
 * definition and its data-availability fix should not ship in the same
 * diff). A follow-up session adds the read path (mirroring gnss_integrity's
 * own precedent of shipping Phase 1/2/3 as separate sessions) once enough
 * days have actually accumulated to be worth reading.
 */
import fs from "fs";
import path from "path";
import zlib from "zlib";
import readline from "readline";
import { archiveBaseDir, RAW_RETENTION_DAYS } from "./datacoreArchive";
import { aggregateGnssIntegrity, type ArchiveAircraftRow, type IntegrityCell } from "./gnssIntegrityQuery";
import { CANDIDATE_BBOX, CONTROL_BBOX } from "./gnssIntegritySignal";

const GNSS_DAILY_ARCHIVE_NAME = "gnss_integrity_daily.json.gz";
// Files this old are certainly already deleted by rollupOldDaysAsync's own
// cutoff (it runs every 6h) — safe to drop from the processed-file tracking
// set so it doesn't grow forever (same idiom as fleetUtilization.ts).
const PROCESSED_PRUNE_DAYS = RAW_RETENTION_DAYS + 3;
const HOUR_FILE_RE = /^(\d{4}-\d{2}-\d{2})-\d{2}\.jsonl(\.gz)?$/;

export interface GnssDailyEntry {
  candidate: IntegrityCell[];
  control: IntegrityCell[];
}

interface GnssDailyArchive {
  /** raw hour filenames already folded in — makes
   *  preserveGnssIntegrityDailyBeforeRollup safe to run twice on the same
   *  files (e.g. a crash/restart between this step and datacoreArchive's
   *  rollup delete) without double-counting. */
  processedFiles: string[];
  days: Record<string, GnssDailyEntry>;
}

function archivePath(base: string): string {
  return path.join(base, GNSS_DAILY_ARCHIVE_NAME);
}

function loadArchive(base: string): GnssDailyArchive {
  try {
    const parsed = JSON.parse(zlib.gunzipSync(fs.readFileSync(archivePath(base))).toString());
    return {
      processedFiles: Array.isArray(parsed.processedFiles) ? parsed.processedFiles : [],
      days: parsed.days && typeof parsed.days === "object" ? parsed.days : {},
    };
  } catch {
    return { processedFiles: [], days: {} };
  }
}

function saveArchive(base: string, archive: GnssDailyArchive): void {
  fs.writeFileSync(archivePath(base), zlib.gzipSync(JSON.stringify(archive)));
}

/** Merge-add `from` cells into `into` (never overwrite — a day can
 *  legitimately accumulate contributions from more than one fold call if
 *  its hour files don't all age past retention in the same tick, mirroring
 *  fleetUtilization.ts's addWeekly). n_total/n_zero are always safely
 *  additive (processedFiles guarantees no single file is ever folded
 *  twice). distinct_airframes is summed too, which can OVER-count an
 *  airframe seen across two separate fold events for the same day — an
 *  accepted approximation because the field is explicitly informational
 *  only and never gates a count (aggregateGnssIntegrity's own contract, see
 *  gnssIntegrityQuery.ts) and, in practice, a full calendar day's hour
 *  files age past the retention cutoff together (rollupOldDaysAsync's own
 *  byDay grouping uses the same day-midnight boundary), so this multi-fold
 *  path is a defensive rare case, not the routine one. */
function mergeCells(into: IntegrityCell[], from: IntegrityCell[]): IntegrityCell[] {
  const byKey = new Map<string, IntegrityCell>(into.map((c) => [`${c.band}|${c.origin}`, { ...c }]));
  for (const c of from) {
    const key = `${c.band}|${c.origin}`;
    const cur = byKey.get(key);
    if (cur) {
      cur.n_total += c.n_total;
      cur.n_zero += c.n_zero;
      cur.distinct_airframes += c.distinct_airframes;
    } else {
      byKey.set(key, { ...c });
    }
  }
  return Array.from(byKey.values())
    .sort((a, b) => (a.band === b.band ? a.origin.localeCompare(b.origin) : a.band.localeCompare(b.band)));
}

/** Stream one hour file (gz or plain) into its raw JSON lines. Never throws
 *  — an unreadable file resolves to whatever lines were read before the
 *  error (matching fleetUtilization.ts's scanFile / datacoreArchive.ts's
 *  streamJsonlLines discipline: a corrupt file degrades, never crashes the
 *  shared event loop the trading tier scheduler also runs on). */
function readLinesFromFile(fp: string): Promise<string[]> {
  return new Promise((resolve) => {
    const lines: string[] = [];
    try {
      let stream: NodeJS.ReadableStream = fs.createReadStream(fp);
      if (fp.endsWith(".gz")) stream = stream.pipe(zlib.createGunzip());
      const rl = readline.createInterface({ input: stream, crlfDelay: Infinity });
      rl.on("line", (l) => { if (l.trim()) lines.push(l); });
      rl.on("close", () => resolve(lines));
      stream.on("error", () => resolve(lines));
      // readline re-emits a piped stream's error on itself too — unlistened,
      // that crashes the process on a truncated/corrupt .gz (see
      // datacoreArchive.ts's streamJsonlLines for the full writeup).
      rl.on("error", () => resolve(lines));
    } catch {
      resolve(lines);
    }
  });
}

/** Fold every raw aircraft hour file older than `retentionDays` that hasn't
 *  already been folded into the permanent daily archive. Call this BEFORE
 *  datacoreArchive's rollupOldDaysAsync deletes the same files each rollup
 *  tick — the two select file eligibility the same way (day-midnight vs.
 *  cutoff), so calling this first is what makes the fold-before-delete
 *  guarantee hold. Best-effort background job — never throws, matching the
 *  rest of the archive pipeline's discipline. */
export async function preserveGnssIntegrityDailyBeforeRollup(
  base = archiveBaseDir(), nowMs = Date.now(), retentionDays = RAW_RETENTION_DAYS,
): Promise<{ filesFolded: number; daysTouched: number }> {
  const dir = path.join(base, "aircraft");
  let names: string[] = [];
  try { names = fs.readdirSync(dir); } catch { return { filesFolded: 0, daysTouched: 0 }; }
  const cutoff = nowMs - retentionDays * 86400_000;
  const archive = loadArchive(base);
  const processed = new Set(archive.processedFiles);

  const byDay: Record<string, string[]> = {};
  for (const f of names) {
    const m = HOUR_FILE_RE.exec(f);
    if (!m || processed.has(f)) continue;
    if (Date.parse(m[1] + "T00:00:00Z") >= cutoff) continue;
    (byDay[m[1]] ||= []).push(f);
  }
  const days = Object.keys(byDay);
  if (!days.length) return { filesFolded: 0, daysTouched: 0 };

  let filesFolded = 0;
  for (const day of days) {
    const files = byDay[day];
    const rows: ArchiveAircraftRow[] = [];
    for (const f of files) {
      for (const line of await readLinesFromFile(path.join(dir, f))) {
        try {
          const r = JSON.parse(line);
          if (r && typeof r.i === "string") rows.push(r as ArchiveAircraftRow);
        } catch { continue; } // skip corrupt line, matches every other archive reader
      }
    }
    const candidate = aggregateGnssIntegrity(rows, CANDIDATE_BBOX);
    const control = aggregateGnssIntegrity(rows, CONTROL_BBOX);
    const existing = archive.days[day];
    archive.days[day] = existing
      ? { candidate: mergeCells(existing.candidate, candidate), control: mergeCells(existing.control, control) }
      : { candidate, control };
    for (const f of files) processed.add(f);
    filesFolded += files.length;
  }

  const pruneCutoff = nowMs - PROCESSED_PRUNE_DAYS * 86400_000;
  archive.processedFiles = Array.from(processed).filter((f) => {
    const m = HOUR_FILE_RE.exec(f);
    return !!m && Date.parse(m[1] + "T00:00:00Z") >= pruneCutoff;
  });
  saveArchive(base, archive);
  return { filesFolded, daysTouched: days.length };
}

/** Read-back for tests and any future in-process consumer — NOT exposed as
 *  a diag probe yet (see this module's header: the reader is deliberately
 *  deferred to a follow-up PR). */
export function loadGnssIntegrityDailyArchive(base = archiveBaseDir()): Record<string, GnssDailyEntry> {
  return loadArchive(base).days;
}

export function _gnssDailyArchivePathForTests(base: string): string {
  return archivePath(base);
}
