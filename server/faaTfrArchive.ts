// faaTfrArchive.ts — the record of US flight restrictions: when each one
// appeared on the FAA's public list and when it left.
//
// WHY ([PRODUCT] 2026-11-22; research/open_questions.md 2026-11-21 (u3)(2)
// and 2026-11-22 (x5)). tfr.faa.gov publishes the restrictions in force NOW.
// It publishes no history: once the FAA takes a notice off its list, the
// outline, the limits and the fact that it was ever listed are gone from
// that site. faaTfrList.ts already reads the list every 30 minutes and each
// notice's detail file once, so it sees every US restriction appear and
// leave. This module writes down what that read saw, append-only. Nobody can
// buy back the days before a record began (BUILD-FIRST: accumulation).
//
// WHAT IS WRITTEN, one JSON object per line, day files by the UTC day of OUR
// read (<archive>/flightrestrictions/YYYY-MM-DD.jsonl, gz after 2 days):
//   listed  a notice is on the list and was not at the read before. `seen`
//           is our read; `prev_read` the read before it (the notice appeared
//           between the two). `backfill: true` = it was already listed when
//           recording began, so `seen` says nothing about when it appeared.
//           `row` is the notice as served (text, dates, the FAA's outline,
//           each area's limits) when its detail file was read in the same
//           pass, else null.
//   detail  the detail file of a notice whose `listed` line had `row: null`.
//   left    a notice is no longer on the list. `last_listed` is the last read
//           that held it, `seen` the first read that did not: it left
//           between the two. `published_end` is the end the notice itself
//           gave; whether it left early is arithmetic the reader can do, and
//           is not written here as a verdict.
//   run     once per UTC day, at the first read of the day: the counts. A day
//           with a `run` line and nothing else is a day on which nothing
//           appeared or left; a day with no file is a day we did not record.
//
// Beside the day files, NOT part of the record: _open.json, the notices
// listed at the last recorded read (written beside, renamed over).
//
// WHAT THIS CANNOT KNOW. Thirty-minute resolution: a restriction listed and
// delisted between two reads is never seen. Hours when the server or
// tfr.faa.gov was down are a gap; the `prev_read` / `last_listed` of the
// next lines say how wide. A changed restriction is a new notice number at
// the FAA, so it appears as one `left` and one `listed`; nothing here links
// the two. US restrictions on the FAA's public list only: the notice feed's
// rows (other countries) are not recorded here. A restart between appending
// a read's lines and writing _open.json repeats those lines at the next
// read (same notice, later `seen`): keep the earliest.
//
// ONE RULE FIXED IN ADVANCE: a read in which more than half of at least
// FAA_TFR_MASS_LEAVE_MIN recorded notices are missing is not recorded the
// first time (a cut or partial answer would write a false mass ending); if
// the next read says the same, it is recorded as read.
//
// RAW record. No score, no ranking, no trading path reads this module.
import fs from "fs";
import path from "path";
import zlib from "zlib";
import { promisify } from "util";
import { archiveBaseDir } from "./datacoreArchive";
import { MIN_FREE_BYTES, readFreeBytes, volumeAllowsWrite } from "./globalScopes";
import type { FaaListEntry, FaaTfrRow } from "./faaTfrList";

const gzip = promisify(zlib.gzip);

export const FAA_TFR_ARCHIVE_STREAM = "flightrestrictions";
export const FAA_TFR_ARCHIVE_STATE_VERSION = 1;
export const FAA_TFR_MASS_LEAVE_MIN = 10;
/** day files older than this are compressed */
export const FAA_TFR_ARCHIVE_GZ_AFTER_MS = 2 * 86_400_000;
/** _open.json above this is not read (the FAA lists 50 to 70 notices; 500 is the reader's own cap) */
export const FAA_TFR_ARCHIVE_STATE_MAX_BYTES = 2 * 1024 * 1024;

/** On unless VOLTRADE_FAA_TFR_ARCHIVE=off|0|false|no. It records only what faaTfrList.ts reads, so it is idle where that is off. */
export function faaTfrArchiveEnabled(env: Record<string, string | undefined> = process.env): boolean {
  const v = String(env.VOLTRADE_FAA_TFR_ARCHIVE ?? "").trim().toLowerCase();
  return !(v === "off" || v === "0" || v === "false" || v === "no");
}

export function faaTfrArchiveDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "flightrestrictions");
}

export interface OpenNotice {
  /** our read that first held the notice (ms) */
  first_seen: number;
  backfill: boolean;
  /** true once a line carrying the notice's detail row has been written */
  detail: boolean;
  entry: FaaListEntry;
  /** the notice's own published period, once its detail file was read (ms; null = none published or not read) */
  start: number | null;
  end: number | null;
}
export interface FaaTfrArchiveState {
  v: number;
  /** the first read this recording run recorded (ms) */
  started_at: number;
  /** the last read recorded (ms) */
  last_read_at: number;
  /** UTC day of the last `run` line */
  last_run_day: string | null;
  /** set when a mass-leave read was refused; cleared by the next recorded read */
  refused_at: number | null;
  open: Record<string, OpenNotice>;
}

type ListFields = Pick<FaaListEntry, "type" | "facility" | "state" | "description">;
export type FaaTfrArchiveRow =
  | { kind: "listed"; seen: string; prev_read: string | null; backfill: boolean; notice: string; citation: string; list: ListFields; row: FaaTfrRow | null }
  | { kind: "detail"; seen: string; notice: string; citation: string; first_seen: string; row: FaaTfrRow }
  | { kind: "left"; seen: string; last_listed: string; notice: string; citation: string; first_seen: string; backfill: boolean; detail: boolean; list: ListFields; published_start: string | null; published_end: string | null }
  | { kind: "run"; seen: string; listed: number; with_outline: number; without_outline: number; detail_pending: number };

const iso = (ms: number) => new Date(ms).toISOString();
const isoOrNull = (ms: number | null) => (ms === null ? null : iso(ms));
const listFields = (e: FaaListEntry): ListFields => ({ type: e.type, facility: e.facility, state: e.state, description: e.description });

export interface FaaTfrDiff { rows: FaaTfrArchiveRow[]; next: FaaTfrArchiveState; refused: boolean }

/** One successful list read against what the last recorded read held. Pure.
 *  `state` null = recording begins with this read (every line `backfill`). */
export function diffFaaListRead(state: FaaTfrArchiveState | null, entries: readonly FaaListEntry[], rowOf: (notice: string) => FaaTfrRow | undefined, now: number): FaaTfrDiff {
  const listed = new Set(entries.map((e) => e.notice));
  if (state) {
    const open = Object.keys(state.open);
    const leaving = open.filter((n) => !listed.has(n)).length;
    if (open.length >= FAA_TFR_MASS_LEAVE_MIN && leaving * 2 > open.length && state.refused_at === null) {
      return { rows: [], next: { ...state, refused_at: now }, refused: true };
    }
  }
  const first = state === null;
  const seen = iso(now);
  const prevRead = state ? iso(state.last_read_at) : null;
  const rows: FaaTfrArchiveRow[] = [];
  const open: Record<string, OpenNotice> = {};
  let withO = 0, withoutO = 0, pending = 0;
  for (const e of entries) {
    const row = rowOf(e.notice);
    if (!row) pending++; else if (row.geometry) withO++; else withoutO++;
    const was = state?.open[e.notice];
    if (!was) {
      rows.push({ kind: "listed", seen, prev_read: prevRead, backfill: first, notice: e.notice, citation: `FDC ${e.notice}`, list: listFields(e), row: row ?? null });
      open[e.notice] = { first_seen: now, backfill: first, detail: !!row, entry: e, start: row?.start ?? null, end: row?.end ?? null };
    } else if (!was.detail && row) {
      rows.push({ kind: "detail", seen, notice: e.notice, citation: `FDC ${e.notice}`, first_seen: iso(was.first_seen), row });
      open[e.notice] = { ...was, detail: true, start: row.start, end: row.end };
    } else {
      open[e.notice] = was;
    }
  }
  if (state) {
    for (const [notice, was] of Object.entries(state.open)) {
      if (listed.has(notice)) continue;
      rows.push({
        kind: "left", seen, last_listed: iso(state.last_read_at), notice, citation: `FDC ${notice}`, first_seen: iso(was.first_seen),
        backfill: was.backfill, detail: was.detail, list: listFields(was.entry), published_start: isoOrNull(was.start), published_end: isoOrNull(was.end),
      });
    }
  }
  const day = seen.slice(0, 10);
  const runDue = !state || state.last_run_day !== day;
  if (runDue) rows.push({ kind: "run", seen, listed: entries.length, with_outline: withO, without_outline: withoutO, detail_pending: pending });
  return {
    rows, refused: false,
    next: { v: FAA_TFR_ARCHIVE_STATE_VERSION, started_at: state?.started_at ?? now, last_read_at: now, last_run_day: runDue ? day : state!.last_run_day, refused_at: null, open },
  };
}

const fin = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const NOTICE_RE = /^\d\/\d{4}$/;

/** _open.json as parsed -> a state, or null when it is not one. Pure. */
export function readFaaTfrArchiveState(raw: unknown): FaaTfrArchiveState | null {
  const d = raw as Partial<FaaTfrArchiveState> | null;
  if (!d || typeof d !== "object" || d.v !== FAA_TFR_ARCHIVE_STATE_VERSION || !fin(d.started_at) || !fin(d.last_read_at) || !d.open || typeof d.open !== "object") return null;
  const open: Record<string, OpenNotice> = {};
  for (const [notice, v] of Object.entries(d.open)) {
    const o = v as Partial<OpenNotice> | null;
    if (!NOTICE_RE.test(notice) || !o || !fin(o.first_seen) || !o.entry || o.entry.notice !== notice) return null;
    open[notice] = {
      first_seen: o.first_seen, backfill: o.backfill === true, detail: o.detail === true,
      entry: { notice, type: o.entry.type ?? null, facility: o.entry.facility ?? null, state: o.entry.state ?? null, description: o.entry.description ?? null },
      start: fin(o.start) ? o.start : null, end: fin(o.end) ? o.end : null,
    };
  }
  return {
    v: FAA_TFR_ARCHIVE_STATE_VERSION, started_at: d.started_at, last_read_at: d.last_read_at,
    last_run_day: typeof d.last_run_day === "string" && /^\d{4}-\d{2}-\d{2}$/.test(d.last_run_day) ? d.last_run_day : null,
    refused_at: fin(d.refused_at) ? d.refused_at : null, open,
  };
}

export type FaaTfrRecordStatus = "recorded" | "refused_mass_leave" | "skipped_low_disk" | "skipped_cut_list" | "state_unreadable" | "write_failed";
export interface FaaTfrRecordResult { status: FaaTfrRecordStatus; rows: number; listed: number; left: number; open: number }

interface Loaded { state: FaaTfrArchiveState | null; unreadable: boolean }
async function loadState(dir: string): Promise<Loaded> {
  const fp = path.join(dir, "_open.json");
  let body: string;
  try {
    const st = await fs.promises.stat(fp);
    if (st.size > FAA_TFR_ARCHIVE_STATE_MAX_BYTES) return { state: null, unreadable: true };
    body = await fs.promises.readFile(fp, "utf8");
  } catch (e) {
    if ((e as NodeJS.ErrnoException)?.code === "ENOENT") return { state: null, unreadable: false };
    console.error(`[faaTfrArchive] state not read: ${(e as NodeJS.ErrnoException)?.code ?? "error"}`);
    return { state: null, unreadable: true };
  }
  let parsed: unknown = null;
  try { parsed = JSON.parse(body); } catch (e) {
    console.error(`[faaTfrArchive] state is not JSON: ${e instanceof Error ? e.message.slice(0, 80) : "error"}`);
    return { state: null, unreadable: true };
  }
  const state = readFaaTfrArchiveState(parsed);
  return { state, unreadable: state === null };
}

/** Compress day files older than two days. Returns how many. */
export async function gzipOldFaaTfrDays(dir: string, nowMs: number): Promise<number> {
  let n = 0;
  for (const f of await fs.promises.readdir(dir)) {
    if (f.startsWith("_") || !/^\d{4}-\d{2}-\d{2}\.jsonl$/.test(f)) continue;
    if (nowMs - Date.parse(f.slice(0, 10)) < FAA_TFR_ARCHIVE_GZ_AFTER_MS) continue;
    const fp = path.join(dir, f);
    await fs.promises.writeFile(`${fp}.gz.tmp`, await gzip(await fs.promises.readFile(fp)));
    await fs.promises.rename(`${fp}.gz.tmp`, `${fp}.gz`);
    await fs.promises.unlink(fp);
    n++;
  }
  return n;
}

// what this process knows of the record, for the route and the card
let current: FaaTfrArchiveState | null = null;
let lastResult: (FaaTfrRecordResult & { at: number }) | null = null;
let rowsThisProcess = 0;

/** Record one successful list read. Never throws. A state file that exists
 *  and cannot be read stops the recording (and is said): starting again from
 *  nothing would write every listed notice as newly begun. */
export async function recordFaaTfrRead(input: {
  entries: readonly FaaListEntry[]; rowOf: (notice: string) => FaaTfrRow | undefined; now: number;
  /** the list answer was cut at the reader's cap: the notices beyond it would read as left */
  cut?: number; baseDir?: string; freeBytes?: (dir: string) => number | null;
}): Promise<FaaTfrRecordResult> {
  const dir = faaTfrArchiveDir(input.baseDir);
  const done = (status: FaaTfrRecordStatus, d?: FaaTfrDiff): FaaTfrRecordResult => {
    const r = {
      status, rows: d?.rows.length ?? 0, listed: d?.rows.filter((x) => x.kind === "listed").length ?? 0,
      left: d?.rows.filter((x) => x.kind === "left").length ?? 0, open: current ? Object.keys(current.open).length : 0,
    };
    lastResult = { ...r, at: input.now };
    return r;
  };
  try {
    if (input.cut) return done("skipped_cut_list");
    await fs.promises.mkdir(dir, { recursive: true });
    if (!volumeAllowsWrite((input.freeBytes ?? readFreeBytes)(dir), MIN_FREE_BYTES)) return done("skipped_low_disk");
    const loaded = await loadState(dir);
    if (loaded.unreadable) return done("state_unreadable");
    const diff = diffFaaListRead(loaded.state, input.entries, input.rowOf, input.now);
    if (diff.rows.length) {
      await fs.promises.appendFile(path.join(dir, `${iso(input.now).slice(0, 10)}.jsonl`), diff.rows.map((r) => JSON.stringify(r)).join("\n") + "\n");
    }
    const tmp = path.join(dir, `_open.json.${process.pid}.tmp`);
    await fs.promises.writeFile(tmp, JSON.stringify(diff.next));
    await fs.promises.rename(tmp, path.join(dir, "_open.json"));
    current = diff.next;
    rowsThisProcess += diff.rows.length;
    if (diff.rows.some((r) => r.kind === "run")) await gzipOldFaaTfrDays(dir, input.now);
    return done(diff.refused ? "refused_mass_leave" : "recorded", diff);
  } catch (e) {
    console.error(`[faaTfrArchive] not recorded: ${(e as NodeJS.ErrnoException)?.code ?? (e instanceof Error ? e.message.slice(0, 120) : "error")}`);
    return done("write_failed");
  }
}

/** Boot: what the record holds as listed, so a card can say since when before the first read of this process. */
export async function loadFaaTfrArchive(baseDir?: string): Promise<number> {
  const l = await loadState(faaTfrArchiveDir(baseDir));
  if (l.state) current = l.state;
  return l.state ? Object.keys(l.state.open).length : 0;
}

/** since when the record has held a listed notice; null = not in the record */
export function faaTfrListedSince(notice: string): { first_seen: number; backfill: boolean } | null {
  const o = current?.open[notice];
  return o ? { first_seen: o.first_seen, backfill: o.backfill } : null;
}

export interface FaaTfrArchiveView {
  /** off = switched off; not_yet = no read recorded by this server yet; recording = the last read was recorded;
   *  paused = the last read was not (why is in `last`) */
  status: "off" | "not_yet" | "recording" | "paused";
  /** the first read of this recording run (ms) */
  started_at: number | null;
  last_recorded_read: number | null;
  /** notices the record holds as listed */
  open: number;
  last: { at: number; status: FaaTfrRecordStatus; rows: number } | null;
  rows_this_process: number;
}
export function faaTfrArchiveView(env: Record<string, string | undefined> = process.env): FaaTfrArchiveView {
  const on = faaTfrArchiveEnabled(env);
  const paused = !!lastResult && lastResult.status !== "recorded";
  return {
    status: !on ? "off" : paused ? "paused" : current ? "recording" : "not_yet",
    started_at: current?.started_at ?? null, last_recorded_read: current?.last_read_at ?? null,
    open: current ? Object.keys(current.open).length : 0,
    last: lastResult ? { at: lastResult.at, status: lastResult.status, rows: lastResult.rows } : null,
    rows_this_process: rowsThisProcess,
  };
}


// ── reading the record back ((y2), 2026-11-22/23) ─────────────────────────
// A bounded read of the newest day files: what appeared on and left the
// FAA's list, WITHOUT the outlines and the notice text (the day files hold
// both; a history list needs neither). Bounded by days, by file size and by
// events (RENDERING & MOTION LAW IV: the record grows forever, a read of it
// must not). The answer is kept for a minute, so a burst of readers is one
// read of the files.

export const FAA_TFR_HISTORY_DEFAULT_DAYS = 7;
export const FAA_TFR_HISTORY_MAX_DAYS = 31;
export const FAA_TFR_HISTORY_MAX_EVENTS = 500;
/** a day file above this on disk is not read and is said (the first day of a run is about 0.2 MB, a later day far less) */
export const FAA_TFR_HISTORY_MAX_FILE_BYTES = 8 * 1024 * 1024;
export const FAA_TFR_HISTORY_CACHE_MS = 60_000;
const DAY_MS = 86_400_000;

export interface FaaTfrHistoryEvent {
  kind: "listed" | "left";
  /** our read that showed the change (ISO, UTC) */
  seen: string;
  /** the read before it: the notice appeared (or left) between `prev_read` and `seen`; null = none recorded */
  prev_read: string | null;
  notice: string; citation: string;
  list: ListFields;
  /** left only: our read that first held the notice, and whether it was already listed when the record began */
  first_seen: string | null; first_seen_backfill: boolean;
  /** the period the notice itself published (null = none published, or its detail file was not read) */
  published_start: string | null; published_end: string | null;
  /** listed only: whether the FAA publishes an outline for it (null = its detail file was not read in this window) */
  outline: boolean | null;
}
export interface FaaTfrHistoryDay {
  day: string;
  /** false = no file for the day: the day was not recorded (or is before the record began) */
  recorded: boolean;
  /** the day's `run` line: the counts at its first read */
  run: { listed: number; with_outline: number; without_outline: number; detail_pending: number } | null;
  /** notices that appeared / left on the day, after repeats are dropped */
  listed: number; left: number;
  /** notices written as already listed when a recording run began (not appearances; not in `events`) */
  already_listed: number;
  /** lines that could not be read as a line of this record */
  unreadable_lines: number;
  /** set when the file was not read: "too_large" | "read_failed" */
  skipped: "too_large" | "read_failed" | null;
}
export interface FaaTfrHistory {
  days_requested: number;
  from_day: string; to_day: string;
  days: FaaTfrHistoryDay[];
  /** newest first; at most FAA_TFR_HISTORY_MAX_EVENTS */
  events: FaaTfrHistoryEvent[];
  /** older events left out by the cap */
  events_cut: number;
}

export function faaTfrHistoryDays(raw: unknown): number {
  const n = typeof raw === "string" && /^\d{1,4}$/.test(raw.trim()) ? parseInt(raw, 10) : NaN;
  return Number.isFinite(n) && n >= 1 ? Math.min(FAA_TFR_HISTORY_MAX_DAYS, n) : FAA_TFR_HISTORY_DEFAULT_DAYS;
}

const strOrNull = (v: unknown): string | null => (typeof v === "string" && v ? v : null);
const isoOk = (v: unknown): v is string => typeof v === "string" && Number.isFinite(Date.parse(v));
function listOf(v: unknown): ListFields {
  const o = (v && typeof v === "object" ? v : {}) as Record<string, unknown>;
  return { type: strOrNull(o.type), facility: strOrNull(o.facility), state: strOrNull(o.state), description: strOrNull(o.description) };
}

/** The lines of consecutive day files, oldest day first -> the history. Pure.
 *  REPEATS: a restart between appending a read's lines and writing the state
 *  repeats them at the next read with a later `seen` (the manifest's limit
 *  5). A `listed` line for a notice already listed with no `left` between is
 *  such a repeat, and so is a second `left` with no `listed` between: the
 *  earliest is kept. */
export function buildFaaTfrHistory(files: ReadonlyArray<{ day: string; lines: readonly string[] | null; skipped?: "too_large" | "read_failed" | null }>, maxEvents = FAA_TFR_HISTORY_MAX_EVENTS): Pick<FaaTfrHistory, "days" | "events" | "events_cut"> {
  const days: FaaTfrHistoryDay[] = [];
  const events: FaaTfrHistoryEvent[] = [];
  // per notice: the last line kind seen in this window, and its `listed` event (a `detail` line completes it)
  const last = new Map<string, "listed" | "left">();
  const listedEvent = new Map<string, FaaTfrHistoryEvent>();
  for (const f of files) {
    const d: FaaTfrHistoryDay = { day: f.day, recorded: f.lines !== null || !!f.skipped, run: null, listed: 0, left: 0, already_listed: 0, unreadable_lines: 0, skipped: f.skipped ?? null };
    days.push(d);
    for (const line of f.lines ?? []) {
      if (!line) continue;
      let r: Record<string, unknown> | null = null;
      try { const p = JSON.parse(line); r = p && typeof p === "object" && !Array.isArray(p) ? p : null; } catch { r = null; }
      if (!r || !isoOk(r.seen)) { d.unreadable_lines++; continue; }
      if (r.kind === "run") {
        const n = (v: unknown) => (fin(v) && v >= 0 ? Math.floor(v) : 0);
        if (!d.run) d.run = { listed: n(r.listed), with_outline: n(r.with_outline), without_outline: n(r.without_outline), detail_pending: n(r.detail_pending) };
        continue;
      }
      const notice = typeof r.notice === "string" && NOTICE_RE.test(r.notice) ? r.notice : null;
      if (!notice) { d.unreadable_lines++; continue; }
      const row = (r.row && typeof r.row === "object" ? r.row : null) as { start?: unknown; end?: unknown; geometry?: unknown } | null;
      if (r.kind === "listed") {
        if (last.get(notice) === "listed") continue; // a repeat
        last.set(notice, "listed");
        if (r.backfill === true) { d.already_listed++; continue; }
        const e: FaaTfrHistoryEvent = {
          kind: "listed", seen: r.seen, prev_read: isoOk(r.prev_read) ? r.prev_read : null, notice, citation: `FDC ${notice}`, list: listOf(r.list),
          first_seen: null, first_seen_backfill: false,
          published_start: row && fin(row.start) ? iso(row.start) : null, published_end: row && fin(row.end) ? iso(row.end) : null,
          outline: row ? !!row.geometry : null,
        };
        events.push(e); listedEvent.set(notice, e); d.listed++;
      } else if (r.kind === "detail") {
        const e = listedEvent.get(notice);
        if (e && e.outline === null && row) {
          e.published_start = fin(row.start) ? iso(row.start) : null;
          e.published_end = fin(row.end) ? iso(row.end) : null;
          e.outline = !!row.geometry;
        }
      } else if (r.kind === "left") {
        if (last.get(notice) === "left") continue; // a repeat
        last.set(notice, "left"); listedEvent.delete(notice);
        events.push({
          kind: "left", seen: r.seen, prev_read: isoOk(r.last_listed) ? r.last_listed : null, notice, citation: `FDC ${notice}`, list: listOf(r.list),
          first_seen: isoOk(r.first_seen) ? r.first_seen : null, first_seen_backfill: r.backfill === true,
          published_start: isoOk(r.published_start) ? r.published_start : null, published_end: isoOk(r.published_end) ? r.published_end : null,
          outline: null,
        });
        d.left++;
      } else {
        d.unreadable_lines++;
      }
    }
  }
  events.reverse(); // the files are in time order: newest first for the reader
  const cut = Math.max(0, events.length - maxEvents);
  if (cut) events.length = maxEvents;
  days.reverse();
  return { days, events, events_cut: cut };
}

const gunzip = promisify(zlib.gunzip);

async function readDayLines(dir: string, day: string): Promise<{ lines: string[] | null; skipped: "too_large" | "read_failed" | null }> {
  for (const name of [`${day}.jsonl`, `${day}.jsonl.gz`]) {
    const fp = path.join(dir, name);
    try {
      const st = await fs.promises.stat(fp);
      if (st.size > FAA_TFR_HISTORY_MAX_FILE_BYTES) return { lines: null, skipped: "too_large" };
      const buf = await fs.promises.readFile(fp);
      const text = (name.endsWith(".gz") ? await gunzip(buf, { maxOutputLength: 8 * FAA_TFR_HISTORY_MAX_FILE_BYTES }) : buf).toString("utf8");
      return { lines: text.split("\n"), skipped: null };
    } catch (e) {
      if ((e as NodeJS.ErrnoException)?.code === "ENOENT") continue;
      console.error(`[faaTfrArchive] ${name} not read: ${(e as NodeJS.ErrnoException)?.code ?? (e instanceof Error ? e.message.slice(0, 80) : "error")}`);
      return { lines: null, skipped: "read_failed" };
    }
  }
  return { lines: null, skipped: null };
}

const historyCache = new Map<string, { at: number; value: Promise<FaaTfrHistory> }>();

/** The last `days` UTC days of the record, ending with the day of `now`. Never throws: a day that cannot be read is said on its row. */
export function readFaaTfrHistory(input: { days?: number; now: number; baseDir?: string }): Promise<FaaTfrHistory> {
  const days = Math.min(FAA_TFR_HISTORY_MAX_DAYS, Math.max(1, Math.floor(input.days ?? FAA_TFR_HISTORY_DEFAULT_DAYS)));
  const dir = faaTfrArchiveDir(input.baseDir);
  const toDay = iso(input.now).slice(0, 10);
  const key = `${dir}|${days}|${toDay}`;
  const hit = historyCache.get(key);
  if (hit && input.now - hit.at < FAA_TFR_HISTORY_CACHE_MS && input.now >= hit.at) return hit.value;
  const value = (async (): Promise<FaaTfrHistory> => {
    const end = Date.parse(toDay);
    const files: Array<{ day: string; lines: string[] | null; skipped: "too_large" | "read_failed" | null }> = [];
    for (let i = days - 1; i >= 0; i--) {
      const day = iso(end - i * DAY_MS).slice(0, 10);
      files.push({ day, ...(await readDayLines(dir, day)) });
    }
    return { days_requested: days, from_day: files[0].day, to_day: toDay, ...buildFaaTfrHistory(files) };
  })();
  if (historyCache.size > 2 * FAA_TFR_HISTORY_MAX_DAYS) historyCache.clear();
  historyCache.set(key, { at: input.now, value });
  return value;
}

export function _resetFaaTfrArchiveForTests(): void { current = null; lastResult = null; rowsThisProcess = 0; historyCache.clear(); }
