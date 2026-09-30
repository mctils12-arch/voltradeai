// WATCHED-PLANE LOOKUP (field bug 2026-09-30, N843S / ab8c8e): clicking a
// watched or tracked plane while it was OUTSIDE the viewport feed opened an
// ARCHIVE card saying "not currently broadcasting" — false: the server's
// worldwide snapshot and the 24/7 tracked poller both held a fix seconds old.
// The page only looked in its own viewport rows.
//
// This module is the decision, kept pure so it is testable headless:
//   1. the viewport/global row the page already holds (unless it is stale);
//   2. the server's snapshot row  (GET /api/data/aircraft/live/:hex);
//   3. the tracked registry's last_seen/last_pos (GET /api/data/aircraft/tracked);
// the first one with a fix younger than WATCH_LIVE_MAX_AGE_MS opens the
// NORMAL live card at that position. Only when none is fresh does the page
// show the archive card — and then it says how long ago the plane was last
// seen instead of claiming it is not broadcasting.
//
// Law I: the page calls this from a user click (flyTo is a user action),
// never from a map-event handler.

import { adaptGlobalPayload, type AdaptedAircraft } from './globalFeed.js';

/** A fix younger than this is "broadcasting now" (the global feed's own
 *  STALE_ROW_MS threshold — one honesty line for both). */
export const WATCH_LIVE_MAX_AGE_MS = 120_000;

/** The subset of a viewport-feed row the live card needs. */
export interface WatchedRow {
  icao24: string | null;
  callsign: string;
  registration: string;
  lon: number;
  lat: number;
  altitude_m: number | null;
  on_ground: boolean;
  velocity_ms: number | null;
  heading: number | null;
  type: string | null;
  category: string | null;
  /** fix age in seconds (null = the feed did not say — treated as fresh
   *  for the page's own viewport rows, which are a fresh poll) */
  seen_pos?: number | null;
  stale?: boolean;
}

export interface TrackedEntry {
  reg: string;
  hex?: string | null;
  /** epoch ms of the last broadcast the tracked poller saw; null = never */
  last_seen?: number | null;
  last_pos?: { la: number; lo: number; al: number | null } | null;
}

export type WatchedDecision =
  | { kind: 'live'; row: WatchedRow; source: 'viewport' | 'snapshot' | 'tracked'; ageMs: number | null }
  | { kind: 'archive'; lastSeenMs: number | null };

const rowAgeMs = (r: WatchedRow): number | null =>
  r.seen_pos == null || !Number.isFinite(r.seen_pos) ? null : Math.max(0, r.seen_pos * 1000);

const rowFresh = (r: WatchedRow): boolean => {
  if (!Number.isFinite(r.lat) || !Number.isFinite(r.lon)) return false;
  if (r.stale === true) return false;
  const a = rowAgeMs(r);
  return a == null || a <= WATCH_LIVE_MAX_AGE_MS;
};

/**
 * Pure: pick the live fix to open, or say it is an archive card.
 * `local` = the row the page already holds (viewport or global feed);
 * `snapshot` = the server's /live/:hex row (seen_pos = age at the server's
 * clock); `tracked` = the registry entry for this hex (ages against nowMs).
 */
export function decideWatchedOpen(input: {
  hex: string;
  local: WatchedRow | null;
  snapshot: WatchedRow | null;
  tracked: TrackedEntry | null;
  nowMs: number;
}): WatchedDecision {
  const { local, snapshot, tracked, nowMs } = input;
  if (local && rowFresh(local)) return { kind: 'live', row: local, source: 'viewport', ageMs: rowAgeMs(local) };
  if (snapshot && rowFresh(snapshot)) return { kind: 'live', row: snapshot, source: 'snapshot', ageMs: rowAgeMs(snapshot) };
  const seen = tracked?.last_seen;
  const pos = tracked?.last_pos;
  if (seen != null && Number.isFinite(seen) && nowMs - seen <= WATCH_LIVE_MAX_AGE_MS
      && pos && Number.isFinite(pos.la) && Number.isFinite(pos.lo)) {
    // the registry carries position + altitude only — speed/heading stay
    // unknown ("—") until the first viewport poll, never invented
    return {
      kind: 'live', source: 'tracked', ageMs: Math.max(0, nowMs - seen),
      row: {
        icao24: input.hex, callsign: '', registration: tracked?.reg ?? '',
        lon: pos.lo, lat: pos.la, altitude_m: pos.al, on_ground: false,
        velocity_ms: null, heading: null, type: null, category: null,
        seen_pos: Math.max(0, nowMs - seen) / 1000,
      },
    };
  }
  // archive: the newest time ANY source saw it (honest "last seen")
  const cands: number[] = [];
  if (seen != null && Number.isFinite(seen)) cands.push(seen);
  for (const r of [local, snapshot]) {
    const a = r ? rowAgeMs(r) : null;
    if (a != null) cands.push(nowMs - a);
  }
  return { kind: 'archive', lastSeenMs: cands.length ? Math.max(...cands) : null };
}

/** Compact age text: "42s", "7m", "3h", "2d". */
export function fmtAgeShort(ms: number): string {
  const s = Math.max(0, Math.round(ms / 1000));
  if (s < 90) return `${s}s`;
  if (s < 5400) return `${Math.round(s / 60)}m`;
  if (s < 172_800) return `${Math.round(s / 3600)}h`;
  return `${Math.round(s / 86_400)}d`;
}

/**
 * The aircraft card subtitle's freshness clause, derived from the newest fix
 * time the card knows (epoch seconds) — never a click-time constant
 * ("the subtitle stayed 'not currently broadcasting' after live values
 * appeared", 2026-09-30).
 */
export function aircraftFreshnessClause(lastFixSec: number | null | undefined, nowMs: number): string {
  if (lastFixSec == null || !Number.isFinite(lastFixSec) || lastFixSec <= 0) return 'no recent signal';
  const age = nowMs - lastFixSec * 1000;
  if (age <= WATCH_LIVE_MAX_AGE_MS) return 'broadcasting now';
  return `last seen ${fmtAgeShort(age)} ago`;
}

/** Decode GET /api/data/aircraft/live/:hex (the /global wire shape) with
 *  the same adapter the global feed uses; null when absent/malformed. */
export function decodeLiveRow(raw: unknown): WatchedRow | null {
  const rows: AdaptedAircraft[] = adaptGlobalPayload(raw).aircraft;
  return rows.length ? rows[0] : null;
}

/** Both server reads in parallel, abortable. Failures degrade to null
 *  (the archive card still opens — and says it could not check). */
export async function fetchWatchedSources(
  hex: string,
  opts: { fetchImpl?: typeof fetch; signal?: AbortSignal } = {},
): Promise<{ snapshot: WatchedRow | null; tracked: TrackedEntry | null; failed: boolean }> {
  const f = opts.fetchImpl ?? ((...a: Parameters<typeof fetch>) => fetch(...a));
  const h = hex.toLowerCase();
  let failed = false;
  const [snap, tr] = await Promise.all([
    f(`/api/data/aircraft/live/${h}`, { signal: opts.signal, cache: 'no-store' })
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => (d ? decodeLiveRow(d) : null))
      .catch(() => { failed = true; return null; }),
    f('/api/data/aircraft/tracked', { signal: opts.signal })
      .then((r) => (r.ok ? r.json() : null))
      .then((d: { planes?: TrackedEntry[] } | null) =>
        (d?.planes || []).find((p) => String(p.hex || '').toLowerCase() === h) ?? null)
      .catch(() => { failed = true; return null; }),
  ]);
  return { snapshot: snap, tracked: tr, failed };
}
