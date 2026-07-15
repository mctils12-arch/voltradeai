// GLOBAL TIME AXIS — EARTH TWIN E1 slice 1 (research/earth_twin_program.md
// A3: "the fourth dimension, unified").
//
// One shared time state for the whole map: LIVE, or a historical instant.
// The Time Machine panel (components/TimeScrubber.tsx) is the axis's UI —
// scrubbing it already replayed our position archives; publishing through
// this store makes every DATED layer follow the same instant, so one slider
// moves the whole world. Subscribe-store pattern mirrors lib/units.ts.
//
// HONESTY RAILS (charter A3): a layer follows the axis only to dates it
// really has — a dated GIBS product snaps to its own honest ceiling
// (yesterday, or further back for high-latency products like SMAP), never
// pretending same-day data exists; the snap is reported so the caller can
// say so on-panel. Live-only layers simply don't subscribe (their honest
// state is "no history at this layer"). Returning to LIVE restores every
// follower to its own default.

import { gibsDefaultDate } from "./gibs.js";

export type TimeAxisState =
  | { mode: "live" }
  | { mode: "historical"; atMs: number };

let state: TimeAxisState = { mode: "live" };
const listeners = new Set<() => void>();

export function getTimeAxis(): TimeAxisState {
  return state;
}

/** Set the axis; no-op (no notify) when nothing changes. */
export function setTimeAxis(next: TimeAxisState): void {
  const same =
    next.mode === state.mode &&
    (next.mode === "live" || (state.mode === "historical" && next.atMs === state.atMs));
  if (same) return;
  state = next;
  // Array.from (not spread): the repo's tsc target predates Set iteration.
  for (const fn of Array.from(listeners)) {
    try { fn(); } catch { /* one bad subscriber never breaks the axis */ }
  }
}

export function subscribeTimeAxis(fn: () => void): () => void {
  listeners.add(fn);
  return () => { listeners.delete(fn); };
}

/** Human-readable UTC instant for the historical-mode badge (charter E1
 *  remainder: "a visible LIVE/HISTORICAL mode chip outside the panel" —
 *  the Time Machine panel is small and can scroll out of view, especially
 *  on mobile, so the world needs a persistent reminder that dated layers
 *  are showing the past, not now). */
export function formatAxisInstant(atMs: number): string {
  return new Date(atMs).toISOString().slice(0, 16).replace("T", " ") + " UTC";
}

export interface GibsAxisDate {
  /** the date the layer should display (YYYY-MM-DD, UTC). */
  dateISO: string;
  /** true when the axis is historical and the layer is following it. */
  followingAxis: boolean;
  /** true when the axis asked for a date newer than the layer honestly has
   *  (snapped back to the layer's own latency ceiling — say so on-panel). */
  snapped: boolean;
}

/**
 * The date a dated GIBS layer should show for the current axis. LIVE →
 * the layer's own default (its honest latest, latency-aware). HISTORICAL →
 * the axis date, clamped to the layer's honest ceiling.
 */
export function gibsDateForAxis(
  axis: TimeAxisState,
  nowMs: number,
  latencyDays = 1,
): GibsAxisDate {
  const ceiling = gibsDefaultDate(nowMs, latencyDays);
  if (axis.mode === "live") return { dateISO: ceiling, followingAxis: false, snapped: false };
  const want = new Date(axis.atMs).toISOString().slice(0, 10);
  if (want <= ceiling) return { dateISO: want, followingAxis: true, snapped: false };
  return { dateISO: ceiling, followingAxis: true, snapped: true };
}
