// FIND A PLANE ANYWHERE (owner report 2026-11-12: "not tracking or picking
// up on adsb ... the planes right away i am looking for n843s"). The Find
// box searched only the rows inside the current viewport, so a tail that was
// airborne elsewhere — or already heard by the provider without a position —
// answered `no "N843S" in the current snapshot`.
//
// The page still searches its own viewport rows first (instant). On a miss
// it asks GET /api/data/aircraft/find?q= and this module turns the answer
// into what the page does and says — pure, so every wording is testable
// headless:
//   live       -> fly to it and open its normal card;
//   heard      -> the transponder is heard, no position yet: say so, with
//                 the altitude and ground speed it sent. No position is shown
//                 because none exists;
//   last_seen  -> say when and where, and open its archive card (trips);
//   unknown    -> say where we looked.
// Every reading carries its age (Law V). Altitude goes through the units
// formatter; knots stay knots (aircraft convention).
//
// Law I: the page calls this from a user action (Enter in the Find box),
// never from a map-event handler.

import { fmtMeters, getUnits, type UnitSystem } from '../units.js';
import { WATCH_LIVE_MAX_AGE_MS, decodeLiveRow, fmtAgeShort, type WatchedRow } from './watchedLookup.js';

const FT_TO_M = 0.3048;
/** a `heard` reading this young reads "is being heard now" */
export const HEARD_NOW_MS = 15_000;

/** The server's reading of a transponder being heard (trackedPlanes.ts). */
export interface HeardReading {
  at: number;
  alt_ft: number | null;
  gs_kt: number | null;
  source?: string | null;
  has_position: boolean;
}

export interface FindLastSeen {
  at: number;
  lat: number;
  lon: number;
  alt_ft: number | null;
  via?: string | null;
}

/** GET /api/data/aircraft/find — the fields this page reads. */
export interface PlaneFindAnswer {
  state: 'live' | 'heard' | 'last_seen' | 'unknown';
  hex?: string | null;
  reg?: string | null;
  at?: number;
  fields?: unknown;
  rows?: unknown;
  heard?: HeardReading | null;
  last_seen?: FindLastSeen | null;
  tracked?: boolean;
  upstream?: string | null;
}

/** 'invalid' = the server refused the query's shape; null = unreachable. */
export type PlaneFindOutcome = PlaneFindAnswer | 'invalid' | null;

export type PlaneFindDecision =
  | { kind: 'live'; row: WatchedRow; note: string }
  | { kind: 'heard'; note: string }
  | { kind: 'last_seen'; hex: string | null; reg: string | null; note: string }
  | { kind: 'unknown'; note: string };

const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);

/** "850 ft, 144 kt" — only what the transponder actually sent. */
export function heardValues(h: Pick<HeardReading, 'alt_ft' | 'gs_kt'>, system: UnitSystem = getUnits()): string {
  const alt = finite(h.alt_ft) ? fmtMeters(h.alt_ft * FT_TO_M, 0, system) : 'altitude not sent';
  const gs = finite(h.gs_kt) ? `${Math.round(h.gs_kt)} kt` : 'speed not sent';
  return `${alt}, ${gs}`;
}

/** "41.79°N 87.80°W" */
export function fmtLatLon(lat: number, lon: number): string {
  return `${Math.abs(lat).toFixed(2)}°${lat >= 0 ? 'N' : 'S'} ${Math.abs(lon).toFixed(2)}°${lon >= 0 ? 'E' : 'W'}`;
}

const VIA_TEXT: Record<string, string> = {
  snapshot: 'our worldwide feed',
  tracked: 'the 24/7 tracked poll',
  archive: 'our archive',
  upstream: 'adsb.lol',
};

/** the heard sentence, shared by the Find note and the watched-plane card */
export function heardSentence(
  label: string, h: HeardReading, nowMs: number, lastSeen: FindLastSeen | null | undefined,
  system: UnitSystem = getUnits(),
): string {
  const age = Math.max(0, nowMs - h.at);
  const when = age <= HEARD_NOW_MS ? 'is being heard now' : `was heard ${fmtAgeShort(age)} ago`;
  const head = `${label}'s transponder ${when} (${heardValues(h, system)})`;
  if (lastSeen && finite(lastSeen.at)) {
    return `${head} but its last position is ${fmtAgeShort(Math.max(0, nowMs - lastSeen.at))} old — it will reappear as soon as it broadcasts a new one`;
  }
  return `${head} but it has not broadcast a position yet — it will appear as soon as it does`;
}

/** shown at once on a viewport miss, while the server is asked */
export function findPendingNote(q: string): string {
  return `no "${q}" in the current view — checking the worldwide feed…`;
}

const upstreamCaveat = (u: string | null | undefined): string =>
  u === 'limited' ? ' (the provider lookup is rate-limited just now — try again in a few seconds)'
    : u === 'error' ? ' (the provider lookup failed just now — try again in a few seconds)'
      : '';

/**
 * Pure: the server's answer -> what the page does + the note under the box.
 * `q` is the query as typed (uppercased by the caller).
 */
export function decidePlaneFind(
  q: string, ans: PlaneFindOutcome, nowMs: number, system: UnitSystem = getUnits(),
): PlaneFindDecision {
  if (ans === null) {
    return { kind: 'unknown', note: `no "${q}" in the current view, and the worldwide lookup could not be reached just now — try again; watched planes stay reachable via their archive` };
  }
  if (ans === 'invalid') {
    return { kind: 'unknown', note: `no "${q}" in the current view — the worldwide lookup takes a tail number, a callsign or a 6-digit hex; type searches (e.g. B738) cover the current view only` };
  }
  const label = String(ans.reg || q);
  if (ans.state === 'live') {
    const row = decodeLiveRow(ans);
    if (row) {
      const age = row.seen_pos == null ? null : row.seen_pos * 1000;
      return {
        kind: 'live', row,
        note: `${label} is outside this view — flying to it${age == null ? '' : ` (position ${fmtAgeShort(age)} old)`}`,
      };
    }
  }
  if (ans.state === 'heard' && ans.heard && finite(ans.heard.at)) {
    return { kind: 'heard', note: heardSentence(label, ans.heard, nowMs, ans.last_seen, system) };
  }
  const ls = ans.last_seen;
  if (ans.state === 'last_seen' && ls && finite(ls.at) && finite(ls.lat) && finite(ls.lon)) {
    const head = ans.upstream === 'empty' ? `no signal from ${label} right now`
      : `${label} is not in our worldwide feed right now${upstreamCaveat(ans.upstream)}`;
    const alt = finite(ls.alt_ft) ? ` at ${fmtMeters(ls.alt_ft * FT_TO_M, 0, system)}` : '';
    const via = VIA_TEXT[String(ls.via || '')];
    return {
      kind: 'last_seen', hex: ans.hex ? String(ans.hex).toLowerCase() : null, reg: ans.reg ?? null,
      note: `${head} — last seen ${fmtAgeShort(Math.max(0, nowMs - ls.at))} ago near ${fmtLatLon(ls.lat, ls.lon)}${alt}${via ? ` (${via})` : ''}`
        + (ans.hex ? '; its archived trips are on the card' : ''),
    };
  }
  return {
    kind: 'unknown',
    note: `no "${q}" in the current view, our worldwide feed, the 24/7 tracked list or the provider's lookup right now${upstreamCaveat(ans.upstream)}`
      + ' — it may be on the ground with its transponder off, or outside receiver coverage; type searches (e.g. B738) cover the current view only; watched planes stay reachable via their archive',
  };
}

/** Ask the server. 400 -> 'invalid'; any other failure -> null. Abortable. */
export async function fetchPlaneFind(
  q: string, opts: { fetchImpl?: typeof fetch; signal?: AbortSignal } = {},
): Promise<PlaneFindOutcome> {
  const f = opts.fetchImpl ?? ((...a: Parameters<typeof fetch>) => fetch(...a));
  try {
    const r = await f(`/api/data/aircraft/find?q=${encodeURIComponent(q)}`, { signal: opts.signal, cache: 'no-store' });
    if (r.status === 400) return 'invalid';
    if (!r.ok) return null;
    const d = (await r.json()) as PlaneFindAnswer | null;
    return d && typeof d === 'object' && typeof d.state === 'string' ? d : null;
  } catch {
    return null; // aborted or unreachable — the caller words it, a lookup never throws
  }
}

// ── the three honest signal states of a tracked / watched plane ─────────────
export type PlaneSignal = 'live' | 'heard' | 'dark';
/** ● a position under 2 min old · ◐ transponder heard, no position · ○ neither */
export const SIGNAL_MARK: Record<PlaneSignal, string> = { live: '●', heard: '◐', dark: '○' };

export interface TrackedSignalRow {
  reg?: string | null;
  hex?: string | null;
  last_seen?: number | null;
  heard?: HeardReading | null;
}

export function planeSignal(p: TrackedSignalRow, nowMs: number): PlaneSignal {
  if (finite(p.last_seen) && nowMs - p.last_seen <= WATCH_LIVE_MAX_AGE_MS) return 'live';
  const h = p.heard;
  if (h && finite(h.at) && !h.has_position && nowMs - h.at <= WATCH_LIVE_MAX_AGE_MS) return 'heard';
  return 'dark';
}

/** The tracked-planes chip: "● N843S · 12s ago" / "◐ N843S · heard 5s ago,
 *  no position yet" / "○ N843S · 3h ago" / "○ N843S · awaiting first signal". */
export function trackedChipText(p: TrackedSignalRow, nowMs: number): string {
  const s = planeSignal(p, nowMs);
  const name = String(p.reg || p.hex || '');
  if (s === 'heard' && p.heard) return `${SIGNAL_MARK.heard} ${name} · heard ${fmtAgeShort(Math.max(0, nowMs - p.heard.at))} ago, no position yet`;
  if (!finite(p.last_seen)) return `${SIGNAL_MARK[s]} ${name} · awaiting first signal`;
  return `${SIGNAL_MARK[s]} ${name} · ${fmtAgeShort(Math.max(0, nowMs - p.last_seen))} ago`;
}

// The tracked registry's rows (GET /api/data/aircraft/tracked), shared by the
// tracked-planes panel and the legend's watchlist chips — one store, so the
// two never disagree and never both poll: the panel publishes what its own
// 30 s poll returns; the legend calls refreshTrackedSignals(), which only
// fetches when nothing was published recently (the panel is closed — e.g. on
// a phone, where the layers sheet is usually shut). A hex the registry does
// not hold has NO mark, and marks expire with their data: a state is never
// guessed and never outlives its reading.
export const TRACKED_SIGNAL_REFRESH_MS = 30_000;
/** refresh only when the newest publish is older than this (a poll + slack) */
export const TRACKED_SIGNAL_STALE_MS = TRACKED_SIGNAL_REFRESH_MS + 2_000;
/** marks older than three missed polls are dropped */
export const TRACKED_SIGNAL_MAX_AGE_MS = 3 * TRACKED_SIGNAL_REFRESH_MS;

let signals = new Map<string, TrackedSignalRow>();
let signalsAt = 0;
let refreshing: Promise<void> | null = null;
const signalSubs = new Set<() => void>();

export function publishTrackedSignals(rows: TrackedSignalRow[], nowMs: number = Date.now()): void {
  const next = new Map<string, TrackedSignalRow>();
  for (const r of rows) {
    const h = String(r?.hex || '').toLowerCase();
    if (/^[0-9a-f]{6}$/.test(h)) next.set(h, r);
  }
  signals = next;
  signalsAt = nowMs;
  signalSubs.forEach((fn) => fn());
}

/** null = this hex is not on the tracked list, or the list we hold is too
 *  old to speak for it (state unknown) */
export function trackedSignalFor(hex: string | null | undefined, nowMs: number): PlaneSignal | null {
  if (nowMs - signalsAt > TRACKED_SIGNAL_MAX_AGE_MS) return null;
  const row = signals.get(String(hex || '').toLowerCase());
  return row ? planeSignal(row, nowMs) : null;
}

export function subscribeTrackedSignals(fn: () => void): () => void {
  signalSubs.add(fn);
  return () => { signalSubs.delete(fn); };
}

/** Fetch the registry unless someone published it recently or a fetch is in
 *  flight. A failure leaves the old rows to expire. Abortable. */
export function refreshTrackedSignals(
  opts: { fetchImpl?: typeof fetch; signal?: AbortSignal; nowMs?: () => number } = {},
): Promise<void> {
  const now = opts.nowMs ?? Date.now;
  if (refreshing) return refreshing;
  if (now() - signalsAt < TRACKED_SIGNAL_STALE_MS) return Promise.resolve();
  const f = opts.fetchImpl ?? ((...a: Parameters<typeof fetch>) => fetch(...a));
  refreshing = f('/api/data/aircraft/tracked', { signal: opts.signal })
    .then((r) => (r.ok ? r.json() : null))
    .then((d: { planes?: TrackedSignalRow[] } | null) => {
      if (d && Array.isArray(d.planes)) publishTrackedSignals(d.planes, now());
    })
    .catch(() => undefined) // aborted or unreachable: the old marks expire on their own
    .finally(() => { refreshing = null; });
  return refreshing;
}

/** test hook */
export function _resetTrackedSignals(): void {
  signals = new Map();
  signalsAt = 0;
  refreshing = null;
}
