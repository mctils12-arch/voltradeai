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

export interface LivenessFile {
  lastActiveAt: number; // epoch ms the loop was last seen active
  savedAt?: string;
}

/** Heartbeat transition — pure. Active now: stamp now. Inactive with no
 *  prior record (fresh install / first boot after upgrade): SEED now so a
 *  new deployment doesn't false-alarm — darkness accrues from this moment.
 *  Otherwise keep the last stamp so restarts never reset the clock (the
 *  equityPeak lesson: deploys must not defang the alarm). */
export function nextLiveness(
  prev: LivenessFile | null, activeNow: boolean, nowMs: number,
): LivenessFile {
  if (activeNow) return { lastActiveAt: nowMs };
  if (!prev || !Number.isFinite(prev.lastActiveAt)) return { lastActiveAt: nowMs };
  return prev;
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
