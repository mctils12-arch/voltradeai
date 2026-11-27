/**
 * dtccCapMarks.ts — which DTCC swap events the source published with its
 * cap mark ("250,000,000+"), kept per vendor day beside the day files
 * (open_questions.md (bc1), 2026-11-27).
 *
 * WHY THIS EXISTS. From the stream's first poll (2026-08-22) until v1.0.1572
 * (2026-11-27) the parser read a notional written "250,000,000+" as a
 * malformed number and archived null. A null in a day file of that period is
 * therefore either a field the source left blank or a value at or above the
 * source's dissemination cap, and the row cannot say which. The day files are
 * append-only and are not rewritten. The source still serves each dated file
 * for two years, so the mark can be read again: this module is the record of
 * that reading.
 *
 * THE RECORD: `<archive>/dtccswaps/_cap_marks.jsonl`, append-only, one line
 * per event:
 *   {day, at, via, usRows, ids}   the vendor file dated `day` was read and
 *                                 these Dissemination Identifiers (US-underlier
 *                                 rows only, the archive's scope) carried the
 *                                 mark. An empty list is a reading: none did.
 *   {day, at, unread: status}     the file did not answer; it is asked again.
 *   {day, at, lost: status}       the file did not answer and is past the
 *                                 vendor's two-year window: the mark for that
 *                                 day's rows was never read and cannot be.
 * A line with `ids` is final for its day; `lost` is final unless `ids` exist.
 *
 * WHO WRITES IT (server/dtccSwaps.ts): every read of a vendor file — the
 * poll's, the catch-up's, the edge read's — records that file's marks, and
 * after them each poll re-reads up to DTCC_CAP_MARKS_PER_POLL vendor days the
 * source-day ledger holds as read and this record does not, OLDEST FIRST (the
 * oldest leaves the vendor first). A seed read by a session on 2026-11-27
 * (datacore/dtcc_cap_marks_seed.json, written by
 * scripts/dtcc_cap_marks_seed.ts through the same parser) covers the days
 * that were hours from leaving the window when this shipped.
 *
 * HOW A READER USES IT: join on the Dissemination Identifier. A day-file row
 * with a null notional whose ID is in a day's `ids` is a capped value (floor
 * DTCC_NOTIONAL_CAP_USD); a null whose vendor day has a reading and no such
 * ID is a blank; a null from a `lost` day is unknown and must be said so.
 * Rows archived from v1.0.1572 on also carry `notionalCappedLeg1` themselves.
 *
 * This module is storage and pure rules only. It imports nothing from
 * dtccSwaps.ts at run time (types only), so the two cannot form an import
 * cycle; the reads over the network live in dtccSwaps.ts.
 */
import fs from "fs";
import path from "path";
import type { DtccSourceDays } from "./dtccSwaps";

export const DTCC_CAP_MARKS_FILE = "_cap_marks.jsonl";
/** Vendor files downloaded per poll for a re-read (the catch-up's pace). */
export const DTCC_CAP_MARKS_PER_POLL = 2;
/** Requests per poll including the ones the vendor refuses (a refusal is one
 *  small response; without this two refused days would use the poll's turn). */
export const DTCC_CAP_MARKS_ATTEMPTS_PER_POLL = 6;
/** A day inside the vendor's window that did not answer is asked again after this long. */
export const DTCC_CAP_MARKS_RETRY_MS = 7 * 86_400_000;
export const DTCC_CAP_MARKS_ENV = "VOLTRADE_DTCC_CAP_MARKS";

export type DtccCapMarkVia = "poll" | "catchup" | "edge" | "reread" | "seed";

/** The state of one vendor day, folded from the record's lines. */
export interface DtccCapMarkDay {
  at: string;
  /** Set when the file was read: who read it, its US-underlier rows, the marked IDs. */
  via?: DtccCapMarkVia;
  usRows?: number;
  ids?: string[];
  /** HTTP status of the request that found the day past the vendor's window (0 = no response). */
  lost?: number;
  /** HTTP status of the latest request that did not answer, the day still inside the window. */
  unread?: number;
}
export type DtccCapMarks = Record<string, DtccCapMarkDay>;

const YMD_RE = /^\d{4}-\d{2}-\d{2}$/;

function marksPath(dir: string): string {
  return path.join(dir, DTCC_CAP_MARKS_FILE);
}

/** Folds the record's text. A malformed line is skipped and counted. Pure. */
export function foldDtccCapMarks(text: string): { days: DtccCapMarks; malformed: number } {
  const days: DtccCapMarks = {};
  let malformed = 0;
  for (const line of text.split("\n")) {
    if (!line.trim()) continue;
    let o: any;
    try { o = JSON.parse(line); } catch { malformed++; continue; }
    if (!o || typeof o.day !== "string" || !YMD_RE.test(o.day) || typeof o.at !== "string") { malformed++; continue; }
    const prev = days[o.day];
    if (Array.isArray(o.ids) && o.ids.every((x: unknown) => typeof x === "string")) {
      days[o.day] = { at: o.at, via: o.via, usRows: typeof o.usRows === "number" ? o.usRows : undefined, ids: o.ids };
    } else if (typeof o.lost === "number") {
      if (!prev?.ids) days[o.day] = { at: o.at, lost: o.lost };
    } else if (typeof o.unread === "number") {
      if (!prev?.ids && prev?.lost === undefined) days[o.day] = { at: o.at, unread: o.unread };
    } else malformed++;
  }
  return { days, malformed };
}

/** `dir` is the stream's own directory (<archive>/dtccswaps). */
export function readDtccCapMarks(dir: string): DtccCapMarks {
  let text: string;
  try {
    text = fs.readFileSync(marksPath(dir), "utf8");
  } catch (e) {
    if ((e as NodeJS.ErrnoException)?.code !== "ENOENT") console.error(`[datacore] dtccswaps cap-mark record unreadable (${(e as Error)?.message || e}) — the days are read again; no row is lost`);
    return {};
  }
  const { days, malformed } = foldDtccCapMarks(text);
  if (malformed > 0) console.error(`[datacore] dtccswaps cap-mark record: ${malformed} line(s) could not be read and were skipped`);
  return days;
}

function appendLine(dir: string, line: Record<string, unknown>): boolean {
  try {
    fs.mkdirSync(dir, { recursive: true });
    fs.appendFileSync(marksPath(dir), JSON.stringify(line) + "\n");
    return true;
  } catch (e) {
    // the record is a reading of the vendor file, not the archive: a lost
    // write means the day is read again
    console.error(`[datacore] dtccswaps cap-mark record not written: ${(e as Error)?.message || e}`);
    return false;
  }
}

/** Records one vendor file's marked IDs. A day that already holds the same
 *  list is left alone (the poll reads the newest file several times a day).
 *  Returns true when a line was written. */
export function recordDtccCapMarks(dir: string, day: string, via: DtccCapMarkVia, usRows: number, ids: string[], nowMs: number): boolean {
  if (!YMD_RE.test(day)) return false;
  const unique = Array.from(new Set(ids)).sort();
  const prev = readDtccCapMarks(dir)[day];
  if (prev?.ids && prev.ids.length === unique.length && prev.ids.every((x, i) => x === unique[i])) return false;
  return appendLine(dir, { day, at: new Date(nowMs).toISOString(), via, usRows, ids: unique });
}

export function recordDtccCapMarksUnread(dir: string, day: string, status: number, lost: boolean, nowMs: number): boolean {
  if (!YMD_RE.test(day)) return false;
  return appendLine(dir, { day, at: new Date(nowMs).toISOString(), [lost ? "lost" : "unread"]: status });
}

function addDaysYmd(ymd: string, n: number): string {
  return new Date(Date.parse(`${ymd}T00:00:00Z`) + n * 86_400_000).toISOString().slice(0, 10);
}

/** True when a vendor file dated `day` is past the window on UTC day of `nowMs`
 *  (the oldest file a GET still answers is dated today - oldestDays). Pure. */
export function dtccDayPastVendorWindow(day: string, nowMs: number, oldestDays: number): boolean {
  return day < addDaysYmd(new Date(nowMs).toISOString().slice(0, 10), -oldestDays);
}

/** The vendor days whose marks are owed: held as read by the source-day
 *  ledger, with no reading and no `lost` line here. Oldest first. A day that
 *  did not answer is asked again by every poll when it is older than `floor`
 *  (an edge day has no later), otherwise after DTCC_CAP_MARKS_RETRY_MS. Pure. */
export function dtccCapMarkCandidates(sourceDays: DtccSourceDays, marks: DtccCapMarks, nowMs: number, floor: string): string[] {
  const out: string[] = [];
  for (const day of Object.keys(sourceDays).sort()) {
    if (!YMD_RE.test(day) || sourceDays[day].unavailable !== undefined) continue;
    const m = marks[day];
    if (!m) { out.push(day); continue; }
    if (m.ids || m.lost !== undefined) continue;
    const last = Date.parse(m.at);
    if (day < floor || !Number.isFinite(last) || nowMs - last >= DTCC_CAP_MARKS_RETRY_MS) out.push(day);
  }
  return out;
}

export function dtccCapMarksEnabled(env: NodeJS.ProcessEnv = process.env): boolean {
  const v = (env[DTCC_CAP_MARKS_ENV] || "").trim().toLowerCase();
  return !(v === "0" || v === "off" || v === "false");
}

export interface DtccCapMarksSummary {
  enabled: boolean;
  /** Vendor days the source-day ledger holds as read. */
  days_read: number;
  /** Of those: the mark was read, is still owed, or can no longer be read. */
  recorded: number;
  owed: number;
  lost: number;
  lost_days: string[];
  /** Marked events over every recorded day. */
  capped_events: number;
}

/** Counts for the routes, over the vendor days the ledger holds as read. Pure. */
export function summarizeDtccCapMarks(sourceDays: DtccSourceDays, marks: DtccCapMarks, enabled: boolean): DtccCapMarksSummary {
  const s: DtccCapMarksSummary = { enabled, days_read: 0, recorded: 0, owed: 0, lost: 0, lost_days: [], capped_events: 0 };
  for (const day of Object.keys(sourceDays).sort()) {
    if (sourceDays[day].unavailable !== undefined) continue;
    s.days_read++;
    const m = marks[day];
    if (m?.ids) { s.recorded++; s.capped_events += m.ids.length; }
    else if (m?.lost !== undefined) { s.lost++; s.lost_days.push(day); }
    else s.owed++;
  }
  return s;
}

/** Every marked Dissemination Identifier on record, for a reader's join. */
export function dtccCappedIdSet(marks: DtccCapMarks): Set<string> {
  const out = new Set<string>();
  for (const m of Object.values(marks)) for (const id of m.ids ?? []) out.add(id);
  return out;
}

/** What a null notional in the archive means for one row, given the vendor
 *  day that supplied it (when known). Pure; the reader's rule in one place. */
export function dtccNullNotionalMeaning(id: string, vendorDay: string | null, marks: DtccCapMarks, capped: Set<string> = dtccCappedIdSet(marks)): "capped" | "blank" | "unknown" {
  if (capped.has(id)) return "capped";
  const m = vendorDay ? marks[vendorDay] : undefined;
  return m?.ids ? "blank" : "unknown";
}

export interface DtccCapMarksSeed {
  days: Record<string, { usRows: number; ids: string[]; readAt: string }>;
}

/** Writes the seed's days the record does not hold. Returns the days written. */
export function applyDtccCapMarksSeed(dir: string, seed: DtccCapMarksSeed | null | undefined, nowMs: number): string[] {
  const written: string[] = [];
  const days = seed?.days;
  if (!days || typeof days !== "object") return written;
  const held = readDtccCapMarks(dir);
  for (const day of Object.keys(days).sort()) {
    const d = days[day];
    if (held[day]?.ids || !d || !Array.isArray(d.ids) || typeof d.usRows !== "number") continue;
    if (recordDtccCapMarks(dir, day, "seed", d.usRows, d.ids, nowMs)) written.push(day);
  }
  return written;
}
