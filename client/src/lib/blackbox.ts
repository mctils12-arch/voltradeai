// blackbox — a flight recorder for crashes that run NO JavaScript.
//
// WHY THIS EXISTS. The /data map has now died three ways in one week: GL context
// loss, Chrome blocking WebGL after a loss cascade, and finally a renderer OOM
// kill ("Not enough memory to open this page") while holding zoom-out from Earth
// toward the Moon — after which the tab crashed on every start.
//
// Every diagnostic before this one shared a fatal assumption: that our code gets
// to run at the moment of failure. An OOM kill does not unwind, does not fire an
// event, and does not run a handler — the process is simply gone. So a recorder
// that reacts to a crash can never see the worst crashes, and crash-loop
// protection that writes a flag "on the way down" never gets to write it.
//
// The inversion that makes this work: don't try to observe the crash. Observe
// whether the PREVIOUS boot ever reached a healthy state. That single bit is
// enough, it needs no cooperation from the dying process, and it is correct for
// every failure mode including a hard kill or a yanked power cord:
//
//   boot N   : bootBegin() writes {bootId, startedAt} to IN_PROGRESS
//              mark(...) appends breadcrumbs as the page comes up
//              bootComplete() DELETES the marker once the page is healthy
//   boot N+1 : bootBegin() finds IN_PROGRESS still there  =>  boot N died young
//              its breadcrumb trail is right there, naming the last thing it did
//
// Hermetic: pure functions over an injectable storage, no DOM, no timers —
// `npx tsx --test`-able.

export interface Crumb {
  /** ms since that boot's start — relative, so trails from different sessions compare */
  t: number;
  step: string;
  [k: string]: unknown;
}

export interface BootReport {
  /** true when a PREVIOUS HEALTHY session died without a clean close — a late
   *  crash (e.g. the 190s OOM). Reported, but NOT a reason for safe mode: a
   *  crash three minutes in is not a start-up problem. */
  prevEndedAbruptly: boolean;
  /** how long that session had been alive at its last heartbeat, ms */
  prevAliveMs: number | null;
  /** true when the previous boot never reached bootComplete() */
  prevCrashed: boolean;
  /** how long the previous boot survived, ms (null if unknown) */
  prevSurvivedMs: number | null;
  /** the previous boot's breadcrumbs — what it was doing when it died */
  prevTrail: Crumb[];
  /** consecutive dead boots. 2+ means a loop, not a one-off. */
  consecutive: number;
  /** this boot's id */
  bootId: string;
}

export interface Storage {
  getItem(k: string): string | null;
  setItem(k: string, v: string): void;
  removeItem(k: string): void;
}

const IN_PROGRESS = "vt-boot-inprogress";
const HEARTBEAT = "vt-boot-heartbeat";
const TRAIL = "vt-boot-trail";
const PREV = "vt-boot-prev";
const STREAK = "vt-boot-deadstreak";

/** Breadcrumbs kept per boot. Small on purpose: this is written to localStorage
 *  repeatedly, and a trail nobody can read in one screen is not a trail. */
export const TRAIL_MAX = 40;
/** Consecutive dead boots before we stop trusting the normal configuration. */
export const SAFE_MODE_AFTER = 1;

function readJson<T>(s: Storage, k: string, fallback: T): T {
  try { const raw = s.getItem(k); return raw ? (JSON.parse(raw) as T) : fallback; } catch { return fallback; }
}
function writeJson(s: Storage, k: string, v: unknown): void {
  try { s.setItem(k, JSON.stringify(v)); } catch { /* storage full/blocked — recorder must never throw */ }
}

/**
 * Call once, as early as possible. Rotates the previous boot's state into a
 * report and arms the marker for this boot.
 */
export function bootBegin(s: Storage, now: number, bootId: string): BootReport {
  const prevMarker = readJson<{ bootId?: string; startedAt?: number } | null>(s, IN_PROGRESS, null);
  const beat = readJson<{ aliveMs?: number; lastStep?: string } | null>(s, HEARTBEAT, null);
  const prevTrail = readJson<Crumb[]>(s, TRAIL, []);
  const prevCrashed = !!prevMarker;

  // a dead session's trail is the evidence — keep it where the UI can show it.
  // TWO death shapes, both persisted (2026-07-31 live gap: a plane-click crash
  // AFTER a healthy boot left report:null in the copied payload — the
  // heartbeat had detected the abrupt end, but its details were never written
  // anywhere the Copy button could read):
  //  · prevCrashed      — died DURING startup (drives safe mode)
  //  · endedAbruptly    — died AFTER going healthy (no safe mode; a crash
  //    minutes in is not a startup problem, but its record matters MORE,
  //    because no context-loss snapshot exists for a hard renderer kill)
  let consecutive = readJson<number>(s, STREAK, 0);
  if (prevCrashed) {
    consecutive = consecutive + 1;
    writeJson(s, PREV, {
      kind: "boot-crash",
      bootId: prevMarker?.bootId ?? null,
      startedAt: prevMarker?.startedAt ?? null,
      survivedMs: prevMarker?.startedAt != null ? Math.max(0, now - prevMarker.startedAt) : null,
      trail: prevTrail,
    });
  } else if (beat) {
    // healthy session that never closed cleanly: keep how long it lived, the
    // LAST THING IT DID, and its full breadcrumb trail (read above, before the
    // trail is reset for this boot)
    consecutive = 0;
    writeJson(s, PREV, {
      kind: "abrupt-end",
      aliveMs: beat.aliveMs ?? null,
      lastStep: beat.lastStep ?? null,
      trail: prevTrail,
    });
  } else {
    consecutive = 0;
    try { s.removeItem(PREV); } catch { /* fine */ }
  }
  writeJson(s, STREAK, consecutive);

  // arm this boot and start a fresh trail
  writeJson(s, IN_PROGRESS, { bootId, startedAt: now });
  writeJson(s, TRAIL, []);

  // a heartbeat left behind means the last session was never closed cleanly
  const prevEndedAbruptly = !prevCrashed && !!beat;
  try { s.removeItem(HEARTBEAT); } catch { /* fine */ }

  return {
    prevCrashed,
    prevEndedAbruptly,
    prevAliveMs: beat?.aliveMs ?? null,
    prevSurvivedMs: prevMarker?.startedAt != null ? Math.max(0, now - prevMarker.startedAt) : null,
    prevTrail,
    consecutive,
    bootId,
  };
}

/** Append a breadcrumb. Ring-buffered, and never throws. */
export function mark(s: Storage, startedAt: number, now: number, step: string, extra: Record<string, unknown> = {}): void {
  const trail = readJson<Crumb[]>(s, TRAIL, []);
  trail.push({ t: Math.max(0, Math.round(now - startedAt)), step, ...extra });
  writeJson(s, TRAIL, trail.slice(-TRAIL_MAX));
}

/**
 * The page is healthy. Disarms the marker so the NEXT boot knows this one
 * survived. Everything hinges on this being called only when genuinely stable —
 * call it too early and a crash-loop looks like a clean run.
 */
export function bootComplete(s: Storage): void {
  try { s.removeItem(IN_PROGRESS); } catch { /* fine */ }
  writeJson(s, STREAK, 0);
}

/** Call periodically while alive, and clear on a clean close. This is what turns
 *  "the tab vanished" into "the tab vanished 190s in, mid space-view". */
export function heartbeat(s: Storage, aliveMs: number, lastStep: string): void {
  writeJson(s, HEARTBEAT, { aliveMs: Math.round(aliveMs), lastStep });
}
export function closeCleanly(s: Storage): void {
  try { s.removeItem(HEARTBEAT); } catch { /* fine */ }
}

/** Reduced configuration is warranted when the previous boot(s) died young. */
export function shouldRunSafe(report: BootReport): boolean {
  return report.consecutive >= SAFE_MODE_AFTER;
}

/** The previous dead boot's full record, for display/copy. */
export function lastCrashReport(s: Storage): unknown | null {
  return readJson<unknown | null>(s, PREV, null);
}

/** Clear every recorder key AND the caller-supplied preference keys — the
 *  "get me out of this" button. */
export function resetAll(s: Storage, extraKeys: string[] = []): void {
  for (const k of [IN_PROGRESS, TRAIL, PREV, STREAK, HEARTBEAT, ...extraKeys]) {
    try { s.removeItem(k); } catch { /* fine */ }
  }
}

export const KEYS = { IN_PROGRESS, TRAIL, PREV, STREAK, HEARTBEAT } as const;
