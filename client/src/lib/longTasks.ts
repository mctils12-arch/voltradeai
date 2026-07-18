// LONG-TASK WATCHDOG — celestial v2 directive §6 (2026-07-18): "Add a
// long-task watchdog (PerformanceObserver) that logs offenders in dev."
// The v1.0.396 freeze (layoutLabelStacks float fixed point, root-caused
// with CDP Debugger.pause) blocked the main thread inside one rAF draw —
// this watchdog makes any recurrence LOUD in dev instead of silent until
// Chrome's "Page Unresponsive" dialog.
//
// Prod-inert by default: active only in dev builds, or when explicitly
// armed (?lt URL param / localStorage vt-lt=1) so the built-dist harness
// drives can assert on it. Entries ring-buffer on window.__vtLongTasks
// (same prod-inert seam idiom as __vtMap/__vtSpace).

const MAX_ENTRIES = 200;

export interface LongTaskEntry {
  /** task start, ms since navigation start (rounded). */
  startMs: number;
  /** task duration, ms (rounded; the longtask threshold is 50ms). */
  durMs: number;
}

/** true when the watchdog should run: dev build, ?lt, or vt-lt=1. */
export function longTaskWatchdogArmed(): boolean {
  try {
    if (import.meta.env?.DEV) return true;
  } catch {
    /* non-vite context (tests) */
  }
  try {
    return (
      new URLSearchParams(window.location.search).has("lt") ||
      window.localStorage?.getItem("vt-lt") === "1"
    );
  } catch {
    return false;
  }
}

/**
 * Start observing long tasks (>50ms main-thread blocks, the browser's
 * longtask threshold). Logs each offender to the console in dev and keeps
 * the last MAX_ENTRIES on window.__vtLongTasks for harness assertions.
 * Returns a stop function; safe to call when unsupported (no-op).
 */
export function startLongTaskWatchdog(): () => void {
  if (typeof PerformanceObserver === "undefined") return () => {};
  const buf: LongTaskEntry[] = [];
  (window as any).__vtLongTasks = buf;
  let obs: PerformanceObserver | null = null;
  try {
    obs = new PerformanceObserver((list) => {
      for (const e of list.getEntries()) {
        const entry: LongTaskEntry = {
          startMs: Math.round(e.startTime),
          durMs: Math.round(e.duration),
        };
        buf.push(entry);
        if (buf.length > MAX_ENTRIES) buf.splice(0, buf.length - MAX_ENTRIES);
        // eslint-disable-next-line no-console
        console.warn(`[longtask] ${entry.durMs}ms main-thread block @ ${entry.startMs}ms`);
      }
    });
    obs.observe({ type: "longtask", buffered: true });
  } catch {
    return () => {};
  }
  return () => {
    try {
      obs?.disconnect();
    } catch {
      /* already gone */
    }
    try {
      delete (window as any).__vtLongTasks;
    } catch {
      /* non-configurable */
    }
  };
}
