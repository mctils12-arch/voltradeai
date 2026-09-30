// SELECTED-AIRCRAFT FAST LANE, client half (human 2026-09-30: "optimize for
// the plane you have clicked on and are watching — I don't want it to lag if
// there is data"). While ONE aircraft is selected this polls
// GET /api/data/aircraft/live/:hex every FAST_POLL_MS (the server answers
// from its snapshot when fresh, else asks adsb.lol for that one hex) and
// hands each NEW fix to the page's live path — the same one the viewport
// poll feeds (dead-reckoning anchor, breadcrumbs + the colored curtain's
// live tail, card readouts, plan seam). Zoomed out, the page's own feed is
// the worldwide snapshot on a 20 s poll with ~minute-old rows; this lane is
// what keeps the selected plane's "last position" at a few seconds.
//
// Lifecycle (tested headless): one request at a time (setTimeout chain, not
// setInterval — a slow answer never stacks requests), AbortController per
// request, paused while the tab is hidden, stop() aborts + clears. A fix is
// delivered only when its OWN time is newer than the last delivered one — an
// unchanged upstream never re-stamps an old position as fresh.

import { adaptGlobalPayload } from './globalFeed.js';
import type { WatchedRow } from './watchedLookup.js';

export const FAST_POLL_MS = 2_500;

export interface FastFix {
  row: WatchedRow;
  /** the fix's own time, epoch ms (server `at` − row age) */
  fixMs: number;
  /** broadcast barometric vertical rate for this fix, ft/min (null = not sent) */
  baroRateFpm: number | null;
  /** where the server got it: snapshot | cache | upstream | limited | error */
  source: string;
}

/** Pure: decode the /live/:hex payload (the /global wire shape + baroRate). */
export function decodeFastFix(raw: unknown): FastFix | null {
  const ad = adaptGlobalPayload(raw);
  const row = ad.aircraft[0];
  if (!row) return null;
  const d = (raw && typeof raw === 'object' ? raw : {}) as { fields?: unknown; rows?: unknown; source?: unknown };
  const fields: string[] = Array.isArray(d.fields) ? d.fields.map(String) : [];
  const first = Array.isArray(d.rows) && Array.isArray(d.rows[0]) ? (d.rows[0] as unknown[]) : [];
  const bi = fields.indexOf('baroRate');
  const b = bi >= 0 ? first[bi] : null;
  const age = row.seen_pos == null ? 0 : row.seen_pos * 1000;
  return {
    row,
    fixMs: ad.at - age,
    baroRateFpm: typeof b === 'number' && Number.isFinite(b) ? b : null,
    source: typeof d.source === 'string' ? d.source : 'snapshot',
  };
}

export interface FastLaneOptions {
  hex: string;
  onFix: (f: FastFix) => void;
  /** every answer, fix or not (freshness chip / "upstream has nothing newer") */
  onPoll?: (f: FastFix | null) => void;
  fetchImpl?: typeof fetch;
  isHidden?: () => boolean;
  pollMs?: number;
  setTimeout?: (fn: () => void, ms: number) => unknown;
  clearTimeout?: (h: unknown) => void;
}

export interface FastLaneHandle {
  stop(): void;
  isStopped(): boolean;
  /** exposed for tests */
  pollNow(): Promise<void>;
}

export function startSelectedFastLane(o: FastLaneOptions): FastLaneHandle {
  const hex = String(o.hex || '').toLowerCase();
  const fetchImpl = o.fetchImpl ?? ((...a: Parameters<typeof fetch>) => fetch(...a));
  const setTo = o.setTimeout ?? ((fn: () => void, ms: number) => setTimeout(fn, ms));
  const clearTo = o.clearTimeout ?? ((h: unknown) => clearTimeout(h as ReturnType<typeof setTimeout>));
  const isHidden = o.isHidden ?? (() => typeof document !== 'undefined' && document.hidden);
  const every = o.pollMs ?? FAST_POLL_MS;
  let stopped = false;
  let timer: unknown = null;
  let ac: AbortController | null = null;
  let lastFixMs = -Infinity;

  const poll = async (): Promise<void> => {
    if (stopped || !/^[0-9a-f]{6}$/.test(hex)) return;
    if (isHidden()) return; // backgrounded: no polling (the next visible tick resumes)
    ac?.abort();
    const mine = new AbortController();
    ac = mine;
    try {
      const r = await fetchImpl(`/api/data/aircraft/live/${hex}`, { signal: mine.signal, cache: 'no-store' });
      if (!r.ok) throw new Error(`live ${r.status}`);
      const d: unknown = await r.json();
      if (stopped || mine.signal.aborted) return;
      const f = decodeFastFix(d);
      o.onPoll?.(f);
      if (f && f.fixMs > lastFixMs) {
        lastFixMs = f.fixMs;
        o.onFix(f);
      }
    } catch (e) {
      if (stopped || mine.signal.aborted) return;
      o.onPoll?.(null); // a failed poll is just a missed sample; the next one retries
    } finally {
      if (ac === mine) ac = null;
    }
  };
  const loop = () => {
    timer = null;
    if (stopped) return;
    void poll().finally(() => { if (!stopped) timer = setTo(loop, every); });
  };
  timer = setTo(loop, 0); // first poll right after selection

  return {
    stop() {
      if (stopped) return;
      stopped = true;
      ac?.abort();
      ac = null;
      if (timer != null) { clearTo(timer); timer = null; }
    },
    isStopped: () => stopped,
    pollNow: poll,
  };
}
