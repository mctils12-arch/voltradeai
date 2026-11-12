/**
 * liveness.ts — LIVENESS ALARM runtime half (Amendment 2, human-approved
 * 2026-07-04): the trading loop paused, halted, or broker-unreadable for
 * more than 2 market hours (or 24 hours wall-clock) must be a degraded
 * state on /api/health — the loop going dark is surfaced loudly, never
 * discovered by the human on a dashboard.
 *
 * PURE module (node:test safe — no sqlite, no fs). Market hours are
 * approximated as NYSE weekday sessions 9:30–16:00 ET, DST-correct via
 * Intl. HOLIDAYS ARE NOT EXCLUDED: a loop paused across a market holiday
 * can alarm up to one session early — the alarm errs toward loud by
 * design, and market_calendar.py (Python, frozen facts) stays the single
 * holiday source of truth; duplicating its table here would rot.
 */

export const LOOP_DARK_MARKET_HOURS = 2;
export const LOOP_DARK_WALLCLOCK_HOURS = 24;

// Re-escalation cadence (filed 2026-09-23, KNOWN BROKEN #42/#43): the
// initial alarm above is a one-time trip — nothing re-notifies while the
// loop stays dark. That let a real 13-day-and-counting dark stretch go a
// full week without a fresh nudge once sessions started treating "reading
// unchanged" as a reason not to re-notify. The reading being unchanged
// doesn't make the cost of continued silence unchanged — it compounds
// daily. This constant governs how often a reminder fires while dark.
export const LIVENESS_REMINDER_INTERVAL_HOURS = 24;

// SUSTAINED ACTIVE (2026-11-12 [REPAIR], filed 2026-11-05): one active tick used
// to stamp lastActiveAt. Live 2026-11-05: two owner toggles (17:26Z, 17:32Z)
// each re-latched the drawdown kill within one tick, and each pushed the
// 2-market-hour alarm back by the whole dark stretch. The loop now counts as
// active only once it has been active for LIVENESS_SUSTAINED_ACTIVE_MINUTES
// without a break; until then the dark clock keeps its start. A run survives a
// restart whose gap is at most LIVENESS_RUN_GAP_MINUTES (a deploy takes 1-3
// minutes, and deploys must not defang the alarm in either direction).
export const LIVENESS_SUSTAINED_ACTIVE_MINUTES = 10;
export const LIVENESS_RUN_GAP_MINUTES = 5;

export interface LivenessFile {
  lastActiveAt: number; // epoch ms the loop was last counted active (a sustained run)
  savedAt?: string;
  lastReminderAt?: number; // epoch ms the last re-escalation reminder fired
  activeSince?: number;    // epoch ms the current unbroken active run began
  activeSeenAt?: number;   // epoch ms of that run's newest active tick
}

/** A liveness file's body -> LivenessFile; null without a finite lastActiveAt.
 *  The optional fields are kept only when finite (older files carry none). Pure. */
export function parseLivenessFile(raw: unknown): LivenessFile | null {
  if (!raw || typeof raw !== "object") return null;
  const o = raw as Record<string, unknown>;
  if (typeof o.lastActiveAt !== "number" || !Number.isFinite(o.lastActiveAt)) return null;
  const out: LivenessFile = { lastActiveAt: o.lastActiveAt };
  for (const k of ["lastReminderAt", "activeSince", "activeSeenAt"] as const) {
    const v = o[k];
    if (typeof v === "number" && Number.isFinite(v)) out[k] = v;
  }
  return out;
}

/** Heartbeat transition — pure.
 *  No prior record (fresh install / first boot after upgrade): SEED now so a
 *  new deployment doesn't false-alarm — darkness accrues from this moment.
 *  Inactive: keep the last stamp so restarts never reset the clock (the
 *  equityPeak lesson: deploys must not defang the alarm); an active run in
 *  progress ends.
 *  Active: the run continues when its newest tick is at most
 *  LIVENESS_RUN_GAP_MINUTES old, else a new run starts now. lastActiveAt is
 *  stamped only once the run is LIVENESS_SUSTAINED_ACTIVE_MINUTES long; a
 *  shorter run changes neither the stamp nor the reminder record, so a toggle
 *  that re-latches at once leaves the dark clock where it was. */
export function nextLiveness(
  prev: LivenessFile | null, activeNow: boolean, nowMs: number,
): LivenessFile {
  const seeded = !!prev && Number.isFinite(prev.lastActiveAt);
  if (!activeNow) {
    if (!seeded) return { lastActiveAt: nowMs };
    if (prev!.activeSince === undefined && prev!.activeSeenAt === undefined) return prev!;
    const { activeSince: _since, activeSeenAt: _seen, ...rest } = prev!;
    return rest;
  }
  if (!seeded) return { lastActiveAt: nowMs, activeSince: nowMs, activeSeenAt: nowMs };
  const p = prev!;
  const gapMs = typeof p.activeSeenAt === "number" ? nowMs - p.activeSeenAt : NaN;
  const continuing = typeof p.activeSince === "number" && Number.isFinite(p.activeSince)
    && gapMs >= 0 && gapMs <= LIVENESS_RUN_GAP_MINUTES * 60_000;
  const activeSince = continuing ? (p.activeSince as number) : nowMs;
  if (nowMs - activeSince >= LIVENESS_SUSTAINED_ACTIVE_MINUTES * 60_000) {
    return { lastActiveAt: nowMs, activeSince, activeSeenAt: nowMs };
  }
  return { ...p, activeSince, activeSeenAt: nowMs };
}

const ET_FMT = new Intl.DateTimeFormat("en-US", {
  timeZone: "America/New_York",
  weekday: "short", hour: "2-digit", minute: "2-digit", hour12: false,
});

function marketOpenAt(ms: number): boolean {
  const parts = ET_FMT.formatToParts(ms);
  const get = (t: string) => parts.find((p) => p.type === t)?.value || "";
  const dow = get("weekday");
  if (dow === "Sat" || dow === "Sun") return false;
  const mins = parseInt(get("hour"), 10) * 60 + parseInt(get("minute"), 10);
  return mins >= 570 && mins < 960; // 9:30–16:00 ET
}

/** Overlap of [startMs, endMs] with NYSE weekday sessions, in hours.
 *  5-minute quantization (±5min on a 2h threshold — fine); scan capped at
 *  14 days because the 24h wall-clock trigger fires long before that. */
export function marketHoursBetween(startMs: number, endMs: number): number {
  if (!(endMs > startMs)) return 0;
  const STEP = 5 * 60_000;
  const cap = Math.min(endMs, startMs + 14 * 86_400_000);
  let open = 0;
  for (let t = startMs; t < cap; t += STEP) if (marketOpenAt(t)) open += STEP;
  return open / 3_600_000;
}

/** The alarm decision. Only meaningful when the loop is NOT active. */
export function loopDark(
  prev: LivenessFile | null, activeNow: boolean, nowMs: number,
): { dark: boolean; marketHours: number; wallHours: number; detail: string } {
  if (activeNow || !prev) return { dark: false, marketHours: 0, wallHours: 0, detail: "" };
  const wallHours = (nowMs - prev.lastActiveAt) / 3_600_000;
  const marketHours = marketHoursBetween(prev.lastActiveAt, nowMs);
  const dark = marketHours > LOOP_DARK_MARKET_HOURS || wallHours > LOOP_DARK_WALLCLOCK_HOURS;
  return {
    dark, marketHours, wallHours,
    detail: dark
      ? `LIVENESS ALARM: trading loop dark for ${marketHours.toFixed(1)} market hours ` +
        `(${wallHours.toFixed(1)}h wall-clock) since ${new Date(prev.lastActiveAt).toISOString()} — ` +
        `paused/halted/stopped state is a top-of-report alarm, never a dashboard discovery`
      : "",
  };
}

/** Ongoing re-escalation gate, pure. True only once per
 *  LIVENESS_REMINDER_INTERVAL_HOURS while the loop is dark — never while
 *  active, never twice within the interval, and immediately the first
 *  time a dark loop has no reminder on record yet. Callers own persisting
 *  the returned "yes, fire" decision (this function has no side effects
 *  and takes no dependency on wall-clock `Date.now()` internally, so it
 *  stays trivially testable). */
export function shouldSendLivenessReminder(
  dark: boolean, lastReminderAt: number | null | undefined, nowMs: number,
): boolean {
  if (!dark) return false;
  if (lastReminderAt == null || !Number.isFinite(lastReminderAt)) return true;
  return (nowMs - lastReminderAt) / 3_600_000 >= LIVENESS_REMINDER_INTERVAL_HOURS;
}
