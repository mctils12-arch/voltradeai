// restrictionHistory.ts — what appeared on and left the FAA's public list of
// flight restrictions, as this server recorded it. Data: GET
// /api/data/swim/notam/tfrs/history (server/faaTfrArchive.ts, the record;
// server/swimCapture.ts, the route). Shown under the "Flight restrictions"
// row of the /data layers panel while the layer is on.
//
// RAW (RAW OVERLAYS vs SIGNALS): every line is two reads of ours and the
// FAA's own words for the notice. Nothing here says a restriction "ended
// early": the line gives when it left the list and the end the notice
// published, and the comparison is the reader's.
//
// Not a map layer: nothing is drawn and no map event is listened to. The
// answer is bounded by the server (days, events) and the panel shows at most
// RESTRICTION_HISTORY_MAX_ROWS of it, saying how many more there are.
import { zuluAt } from "./airportTraffic.js";

export const RESTRICTION_HISTORY_DAYS = 7;
/** the list is read every 30 minutes; the panel asks every 5 while the tab is visible */
export const RESTRICTION_HISTORY_REFRESH_MS = 5 * 60_000;
export const RESTRICTION_HISTORY_MAX_ROWS = 8;
/** two reads further apart than this had a gap between them (the reads are 30 minutes apart) */
export const READ_GAP_MS = 45 * 60_000;
export const RESTRICTION_HISTORY_URL = `/api/data/swim/notam/tfrs/history?days=${RESTRICTION_HISTORY_DAYS}`;

export interface RestrictionHistoryEvent {
  kind: "listed" | "left";
  /** our read that showed the change, and the read before it */
  seenMs: number; prevReadMs: number | null;
  /** "FDC 6/9306" */
  citation: string;
  /** the FAA list's own words */
  type: string | null; state: string | null; place: string | null;
  /** left only: since when the record held it; backfill = it was already listed when the record began */
  firstSeenMs: number | null; firstSeenBackfill: boolean;
  publishedEndMs: number | null;
  /** listed only: false = the FAA publishes no outline for it; null = not known yet */
  outline: boolean | null;
}
export type RecordStatus = "off" | "not_yet" | "recording" | "paused";
export interface RestrictionHistoryView {
  atMs: number | null;
  daysRequested: number;
  /** newest first */
  events: RestrictionHistoryEvent[];
  /** events the server left out at its cap */
  eventsCut: number;
  /** counted over the whole window, whatever the cap cut */
  appeared: number; left: number;
  /** days of the window with a file; days after the record began with none; days whose file could not be read */
  daysRecorded: number; daysMissing: number; daysUnread: number;
  /** the first read of this recording run; null = the server does not say */
  startedMs: number | null;
  status: RecordStatus | null;
}

const obj = (v: unknown): Record<string, unknown> | null => (v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : null);
const fin = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const str = (v: unknown): string | null => (typeof v === "string" && v.trim() ? v.trim() : null);
const ms = (v: unknown): number | null => { const t = typeof v === "string" ? Date.parse(v) : NaN; return Number.isFinite(t) ? t : null; };
const count = (v: unknown): number => (fin(v) && v >= 0 ? Math.floor(v) : 0);
const STATUS: readonly RecordStatus[] = ["off", "not_yet", "recording", "paused"];

/** The FAA list's description is "<place>, <weekday>, <date> through ...": the place alone. */
export function placeOf(description: string | null): string | null {
  if (!description) return null;
  const cut = description.search(/,\s*(?:Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)day,/i);
  const place = (cut > 0 ? description.slice(0, cut) : description).trim();
  return place ? (place.length > 60 ? `${place.slice(0, 59)}…` : place) : null;
}

/** GET /api/data/swim/notam/tfrs/history -> the panel's view; null on another shape. */
export function parseRestrictionHistory(raw: unknown): RestrictionHistoryView | null {
  const o = obj(raw);
  if (!o || !Array.isArray(o.events) || !Array.isArray(o.days)) return null;
  const events: RestrictionHistoryEvent[] = [];
  for (const ev of o.events) {
    const e = obj(ev);
    const seenMs = e ? ms(e.seen) : null;
    const citation = e ? str(e.citation) : null;
    if (!e || seenMs === null || !citation || (e.kind !== "listed" && e.kind !== "left")) continue;
    const list = obj(e.list);
    events.push({
      kind: e.kind, seenMs, prevReadMs: ms(e.prev_read), citation,
      type: list ? str(list.type) : null, state: list ? str(list.state) : null, place: placeOf(list ? str(list.description) : null),
      firstSeenMs: ms(e.first_seen), firstSeenBackfill: e.first_seen_backfill === true,
      publishedEndMs: ms(e.published_end), outline: typeof e.outline === "boolean" ? e.outline : null,
    });
  }
  events.sort((a, b) => b.seenMs - a.seenMs);
  const rec = obj(o.record);
  const startedMs = rec && fin(rec.started_at) && rec.started_at > 0 ? rec.started_at : null;
  const startDay = startedMs !== null ? new Date(startedMs).toISOString().slice(0, 10) : null;
  let appeared = 0, left = 0, daysRecorded = 0, daysMissing = 0, daysUnread = 0;
  for (const dv of o.days) {
    const d = obj(dv);
    const day = d ? str(d.day) : null;
    if (!d || !day) continue;
    appeared += count(d.listed); left += count(d.left);
    if (d.recorded === true) { daysRecorded++; if (str(d.skipped)) daysUnread++; }
    else if (startDay !== null && day >= startDay) daysMissing++;
  }
  return {
    atMs: fin(o.at) ? o.at : null, daysRequested: count(o.days_requested) || o.days.length, events, eventsCut: count(o.events_cut),
    appeared, left, daysRecorded, daysMissing, daysUnread, startedMs,
    status: rec ? STATUS.find((s) => s === rec.status) ?? null : null,
  };
}

const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;

/** The sentence above the rows: the window, the counts, and what the record cannot say. */
export function historyHeadWords(v: RestrictionHistoryView, nowMs: number): string {
  const window = `last ${plural(v.daysRequested, "day")}`;
  const counts = v.appeared || v.left ? `${v.appeared} appeared, ${v.left} left` : "none appeared or left";
  const parts = [`This server's record of the FAA's list, ${window}: ${counts}.`];
  const windowStart = nowMs - v.daysRequested * 86_400_000;
  if (v.startedMs !== null && v.startedMs > windowStart) parts.push(`The record began ${zuluAt(v.startedMs, nowMs)}; nothing before that is in it.`);
  else if (v.startedMs === null && v.daysRecorded === 0) parts.push("No day of this window has been recorded.");
  if (v.daysMissing) parts.push(`${plural(v.daysMissing, "day")} of this window ${v.daysMissing === 1 ? "was" : "were"} not recorded.`);
  if (v.daysUnread) parts.push(`${plural(v.daysUnread, "day")} could not be read.`);
  if (v.status === "paused") parts.push("Recording is paused: changes since the last recorded read are not in it.");
  if (v.status === "off") parts.push("Recording is switched off: nothing is being added.");
  return parts.join(" ");
}

/** "between Nov 22 14:30Z and 15:00Z" (the same UTC day is said once), with the width of a gap between the two reads */
export function betweenWords(prevReadMs: number | null, seenMs: number, nowMs: number): string {
  if (prevReadMs === null) return `by ${zuluAt(seenMs, nowMs)}`;
  const sameDay = new Date(prevReadMs).toISOString().slice(0, 10) === new Date(seenMs).toISOString().slice(0, 10);
  const span = `between ${zuluAt(prevReadMs, nowMs)} and ${sameDay ? zuluAt(seenMs, seenMs) : zuluAt(seenMs, nowMs)}`;
  const gap = seenMs - prevReadMs;
  if (gap <= READ_GAP_MS) return span;
  const h = gap / 3600_000;
  return `${span} (the two reads were ${h < 10 ? h.toFixed(1) : Math.round(h)} h apart: the list was not read in between)`;
}

export interface HistoryRow { key: string; mark: "+" | "−"; head: string; body: string }

/** One event as the panel writes it. */
export function historyRow(e: RestrictionHistoryEvent, nowMs: number): HistoryRow {
  const head = [e.citation, e.type, e.place ?? e.state].filter(Boolean).join(" · ");
  const when = betweenWords(e.prevReadMs, e.seenMs, nowMs);
  if (e.kind === "listed") {
    const tail = [
      e.publishedEndMs !== null ? `published end ${zuluAt(e.publishedEndMs, nowMs)}` : null,
      e.outline === false ? "no outline published, not drawn" : null,
    ].filter(Boolean).join("; ");
    return { key: `listed:${e.citation}:${e.seenMs}`, mark: "+", head, body: `appeared ${when}${tail ? `; ${tail}` : ""}` };
  }
  const end = e.publishedEndMs !== null ? `its published end was ${zuluAt(e.publishedEndMs, nowMs)}` : "no published end on record";
  const since = e.firstSeenMs === null ? null
    : e.firstSeenBackfill ? "on the list since before the record began" : `first seen ${zuluAt(e.firstSeenMs, nowMs)}`;
  return { key: `left:${e.citation}:${e.seenMs}`, mark: "−", head, body: [`left the list ${when}`, end, since].filter(Boolean).join("; ") };
}

/** The rows shown and how many of the window's changes are not: the panel's cap plus the server's. */
export function historyRows(v: RestrictionHistoryView, nowMs: number, max = RESTRICTION_HISTORY_MAX_ROWS): { rows: HistoryRow[]; more: number } {
  const rows = v.events.slice(0, max).map((e) => historyRow(e, nowMs));
  return { rows, more: Math.max(0, v.events.length - rows.length) + v.eventsCut };
}

export const HISTORY_FOOT = "+ appeared on the FAA's list, − left it. Times are this server's reads of tfr.faa.gov, half an hour apart, in UTC; the FAA's site itself keeps no history.";
