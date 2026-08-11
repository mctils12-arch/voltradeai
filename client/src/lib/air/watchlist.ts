/** Plane watchlist (plane-tracking T3, human directive 2026-08-08: "have a
 *  watch list and have multiple planes"). localStorage-persisted, capped,
 *  subscribe pattern per lib/units.ts. Pure add/remove/toggle helpers so
 *  node:test pins the contract without a DOM. */

export interface WatchedPlane {
  hex: string;           // icao24, lowercase — the identity key
  reg?: string | null;   // registration when known
  callsign?: string | null;
  type?: string | null;  // ICAO type designator
  addedAt: number;       // unix ms
}

export const WATCHLIST_KEY = "vt-plane-watchlist";
export const WATCHLIST_CAP = 50;

export function normalizeHex(hex: string | null | undefined): string | null {
  const h = String(hex || "").trim().toLowerCase();
  return /^[0-9a-f]{6}$/.test(h) ? h : null;
}

/** Pure: add (idempotent by hex, refreshes metadata), newest first, capped. */
export function addWatch(list: WatchedPlane[], p: WatchedPlane): WatchedPlane[] {
  const hex = normalizeHex(p.hex);
  if (!hex) return list;
  const rest = list.filter((w) => w.hex !== hex);
  return [{ ...p, hex }, ...rest].slice(0, WATCHLIST_CAP);
}

/** Pure: remove by hex. */
export function removeWatch(list: WatchedPlane[], hex: string): WatchedPlane[] {
  const h = normalizeHex(hex);
  return h ? list.filter((w) => w.hex !== h) : list;
}

export function isWatched(list: WatchedPlane[], hex: string | null | undefined): boolean {
  const h = normalizeHex(hex);
  return !!h && list.some((w) => w.hex === h);
}

// ── persisted store (module singleton; storage failures degrade to memory) ──
let cache: WatchedPlane[] | null = null;
const subs = new Set<() => void>();

function read(): WatchedPlane[] {
  if (cache) return cache;
  try {
    const raw = JSON.parse(window.localStorage.getItem(WATCHLIST_KEY) ?? "[]");
    cache = Array.isArray(raw) ? raw.filter((w) => normalizeHex(w?.hex)) : [];
  } catch { cache = []; }
  return cache!;
}

function write(next: WatchedPlane[]): void {
  cache = next;
  try { window.localStorage.setItem(WATCHLIST_KEY, JSON.stringify(next)); } catch { /* memory-only */ }
  subs.forEach((fn) => { try { fn(); } catch { /* subscriber's problem */ } });
}

export function getWatchlist(): WatchedPlane[] { return read(); }
export function watchPlane(p: Omit<WatchedPlane, "addedAt"> & { addedAt?: number }): void {
  write(addWatch(read(), { addedAt: Date.now(), ...p }));
}
export function unwatchPlane(hex: string): void { write(removeWatch(read(), hex)); }
export function subscribeWatchlist(fn: () => void): () => void {
  subs.add(fn);
  return () => subs.delete(fn);
}
