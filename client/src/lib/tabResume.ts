// tabResume — ONE page-wide "this tab is live again" signal for /data pollers.
//
// [REPAIR 2026-10-14] root cause of the recurring stale-tab reports (2026-07-22
// "I leave the page… come back and the data is stale"; 2026-10-14 "leave the
// tab up overnight and the adsb gets stale and the other stuff"). Every
// poller did its own return handling, or none at all:
//   - 30 hidden-gated feed pollers skipped their ticks while hidden and then
//     waited out a full interval (5–60 min) after the tab came back;
//   - the handlers that did exist listened ONLY for visibilitychange, which
//     never fires when the computer sleeps and wakes with the tab in front,
//     or when the network drops and comes back.
// This module turns all of those cases into one event:
//   visible  — the tab came back after being hidden
//   wake     — the timer heartbeat jumped (computer slept / tab was frozen)
//              while the tab is in front
//   online   — the browser regained its network connection
//   pageshow — the page was restored from the back/forward cache
// pollWhenVisible() wraps the standard hidden-gated interval: on any resume it
// reloads right away IF its interval has elapsed since the last load, so a
// quick tab flip costs nothing and an overnight absence refreshes at once.
//
// DOM-free (injectable deps) so tsx --test drives it with fake time.

export type ResumeReason = "visible" | "wake" | "online" | "pageshow";
export interface ResumeEvent { reason: ResumeReason; awayMs: number }
export type ResumeListener = (ev: ResumeEvent) => void;

/** heartbeat period for wake detection */
export const RESUME_HEARTBEAT_MS = 15_000;
/** a heartbeat gap above this, with the tab in front, means the machine slept */
export const RESUME_SLEEP_GAP_MS = 60_000;

interface EventTargetLike {
  addEventListener(type: string, fn: (e: any) => void): void;
  removeEventListener(type: string, fn: (e: any) => void): void;
}
export interface TabResumeDeps {
  doc: EventTargetLike & { hidden: boolean };
  win: EventTargetLike;
  now?: () => number;
  setInterval?: (fn: () => void, ms: number) => unknown;
  clearInterval?: (h: unknown) => void;
}

export interface TabResume {
  subscribe(fn: ResumeListener): () => void;
  dispose(): void;
}

export function createTabResume(deps: TabResumeDeps): TabResume {
  const now = deps.now ?? (() => Date.now());
  const setI = deps.setInterval ?? ((fn: () => void, ms: number) => setInterval(fn, ms));
  const clearI = deps.clearInterval ?? ((h: unknown) => clearInterval(h as ReturnType<typeof setInterval>));
  const { doc, win } = deps;
  const listeners = new Set<ResumeListener>();
  let hiddenAt: number | null = doc.hidden ? now() : null;
  let offlineAt: number | null = null;
  let lastBeat = now();

  const emit = (ev: ResumeEvent) => {
    lastBeat = now(); // a resume resets the wake clock: never double-fire
    for (const fn of Array.from(listeners)) {
      try { fn(ev); } catch { /* one bad listener never starves the rest */ }
    }
  };
  const onVisibility = () => {
    if (doc.hidden) { hiddenAt = now(); return; }
    const away = hiddenAt == null ? 0 : now() - hiddenAt;
    hiddenAt = null;
    emit({ reason: "visible", awayMs: away });
  };
  const onOnline = () => {
    const away = offlineAt == null ? 0 : now() - offlineAt;
    offlineAt = null;
    emit({ reason: "online", awayMs: away });
  };
  const onOffline = () => { offlineAt = now(); };
  const onPageShow = (e: { persisted?: boolean }) => {
    if (e && e.persisted) emit({ reason: "pageshow", awayMs: now() - lastBeat });
  };
  const beat = setI(() => {
    const t = now();
    const gap = t - lastBeat;
    lastBeat = t;
    // hidden tabs are timer-throttled (Chrome: ~1/min after 5 min), so a gap
    // there is expected; the visibilitychange path covers their return
    if (!doc.hidden && gap > RESUME_SLEEP_GAP_MS) emit({ reason: "wake", awayMs: gap });
  }, RESUME_HEARTBEAT_MS);

  doc.addEventListener("visibilitychange", onVisibility);
  win.addEventListener("online", onOnline);
  win.addEventListener("offline", onOffline);
  win.addEventListener("pageshow", onPageShow);

  return {
    subscribe(fn) {
      listeners.add(fn);
      return () => { listeners.delete(fn); };
    },
    dispose() {
      listeners.clear();
      clearI(beat);
      doc.removeEventListener("visibilitychange", onVisibility);
      win.removeEventListener("online", onOnline);
      win.removeEventListener("offline", onOffline);
      win.removeEventListener("pageshow", onPageShow);
    },
  };
}

// page-wide singleton, created on first use
let shared: TabResume | null = null;

/** Subscribe to the page-wide resume signal. Returns the unsubscribe. */
export function onTabResume(fn: ResumeListener): () => void {
  if (typeof document === "undefined" || typeof window === "undefined") return () => {};
  if (!shared) shared = createTabResume({ doc: document, win: window });
  return shared.subscribe(fn);
}

export interface PollOpts {
  /** keep ticking while the tab is hidden (the pre-existing ungated pollers) */
  whileHidden?: boolean;
  now?: () => number;
  isHidden?: () => boolean;
  setInterval?: (fn: () => void, ms: number) => unknown;
  clearInterval?: (h: unknown) => void;
  subscribe?: (fn: ResumeListener) => () => void;
}

/**
 * The standard /data poller: `load` every `intervalMs` (skipped while the tab
 * is hidden unless `whileHidden`), plus an immediate `load` on resume when a
 * full interval has passed since the last one. The caller does the first
 * load itself, as before. Returns the stop function for the effect cleanup.
 */
export function pollWhenVisible(load: () => unknown, intervalMs: number, opts: PollOpts = {}): () => void {
  const now = opts.now ?? (() => Date.now());
  const isHidden = opts.isHidden ?? (() => typeof document !== "undefined" && document.hidden);
  const setI = opts.setInterval ?? ((fn: () => void, ms: number) => window.setInterval(fn, ms));
  const clearI = opts.clearInterval ?? ((h: unknown) => window.clearInterval(h as number));
  const subscribe = opts.subscribe ?? onTabResume;
  let last = now();
  const run = () => { last = now(); void load(); };
  const iv = setI(() => { if (opts.whileHidden || !isHidden()) run(); }, intervalMs);
  const off = subscribe(() => { if (!isHidden() && now() - last >= intervalMs) run(); });
  return () => { clearI(iv); off(); };
}

/** hard cap on one live-feed request (aircraft/vessels/trains). Polls re-arm
 *  only after the previous load settles, so without a cap one request stuck
 *  on a dead connection stops the feed. */
export const LIVE_FETCH_TIMEOUT_MS = 30_000;

/** An AbortSignal that fires after `ms`, so a request stuck on a dead
 *  connection (typical right after a sleep) fails instead of hanging. */
export function timeoutSignal(ms: number): AbortSignal | undefined {
  try {
    const AS = AbortSignal as unknown as { timeout?: (ms: number) => AbortSignal };
    if (typeof AS.timeout === "function") return AS.timeout(ms);
    const ac = new AbortController();
    setTimeout(() => ac.abort(), ms);
    return ac.signal;
  } catch { return undefined; }
}
