// simClock — ONE SIMULATION CLOCK for the celestial system (celestial v2 B3,
// directive §3, 2026-07-18: "Time rate drives EVERYTHING consistently from
// one simulation clock: orbital positions, rotations, Earth's terminator,
// moon phase, satellite propagation epoch. At 1× the view always tracks
// real time").
//
// THE MODEL: simulated time is an affine map of real time,
//     simNow = anchorSimMs + rate · (realNow − anchorRealMs)
// re-anchored on every rate change so the simulated instant is CONTINUOUS
// across rate switches (no jumps — pausing holds the instant you were
// looking at; resuming continues from it). rate 0 = paused.
//
// THE REGRESSION CONTRACT (charter: "when the clock is at 1× real time,
// behavior must be EXACTLY today's"): a freshly-reset clock has
// anchorSimMs === anchorRealMs and rate === 1, so simNowMs(st, t) === t
// BIT-EXACTLY for every t (pure addition of a zero offset in doubles is
// the identity). isRealtime() detects exactly that state, and every
// consumer takes its pre-B3 code path when it holds — the clock module
// itself guarantees Date.now() passthrough. Pinned by tests.
//
// ⟲now ("snap back to real time") resets to that identity state. A rate-1
// clock that was warped and brought back WITHOUT ⟲now keeps its offset
// (running parallel to real time) — that is a NON-realtime state and the
// always-visible offset chip stays up: simulated ≠ real is never silent
// (PREMIUM EXPERIENCE STANDARD honesty machinery).
//
// DELIBERATELY NOT PERSISTED: a reload returns to live. Persisting a
// warped clock would silently present non-live positions as the current
// sky on the next visit — an honesty hazard with no offsetting value.
//
// Store pattern: lib/units.ts / scaleModel.ts (get/set/subscribe, no DOM
// beyond the store). Pure math exported for `npx tsx --test`.

/** The directive's rates: 1× (real time), 60× (1 min/s), 3600× (1 h/s),
 *  86400× (1 day/s). 0 = pause. */
export const SIM_RATES: readonly number[] = Object.freeze([1, 60, 3600, 86400]);

export interface SimClockState {
  /** simulated milliseconds per real millisecond; 0 = paused. */
  rate: number;
  /** real-clock ms (Date.now domain) at the last re-anchor. */
  anchorRealMs: number;
  /** simulated ms at the last re-anchor. */
  anchorSimMs: number;
}

/** The simulated instant for a real-clock reading. Affine, pure. */
export function simNowMs(st: SimClockState, realNowMs: number): number {
  return st.anchorSimMs + st.rate * (realNowMs - st.anchorRealMs);
}

/** Signed simulated−real offset, ms (what the honesty chip prints). */
export function simOffsetMs(st: SimClockState, realNowMs: number): number {
  return simNowMs(st, realNowMs) - realNowMs;
}

/**
 * TRUE exactly when the clock is the identity map of real time — rate 1
 * anchored with zero offset (the reset state). This is the gate every
 * consumer uses to take its pre-B3 code path, so at 1× real time the
 * system's behavior is EXACTLY the un-warped one.
 */
export function isRealtime(st: SimClockState): boolean {
  return st.rate === 1 && st.anchorSimMs === st.anchorRealMs;
}

/** The reset (⟲now) state: simulated ≡ real, advancing at 1×. */
export function resetState(realNowMs: number): SimClockState {
  return { rate: 1, anchorRealMs: realNowMs, anchorSimMs: realNowMs };
}

/**
 * Change rate WITHOUT moving the simulated instant: re-anchor at the
 * current (real, sim) pair. simNowMs is continuous across the switch —
 * pinned by test. Non-finite/negative rates clamp to 0 (paused beats a
 * runaway clock).
 */
export function withRate(st: SimClockState, rate: number, realNowMs: number): SimClockState {
  const r = Number.isFinite(rate) && rate > 0 ? rate : 0;
  return { rate: r, anchorRealMs: realNowMs, anchorSimMs: simNowMs(st, realNowMs) };
}

/** Rate → the control's label. */
export function simRateLabel(rate: number): string {
  if (rate === 0) return "paused";
  if (rate === 1) return "1×";
  if (rate === 86400) return "1 day/s";
  return `${rate}×`;
}

/**
 * Signed compact offset for the chip: the two most significant of
 * d/h/m/s (e.g. "+3h 12m", "−2d 4h", "+45s"). Sub-second reads "0s"
 * (the chip is not shown at realtime anyway).
 */
export function fmtSimOffset(ms: number): string {
  const sign = ms < 0 ? "−" : "+";
  let s = Math.round(Math.abs(ms) / 1000);
  if (s === 0) return "0s";
  const d = Math.floor(s / 86400); s -= d * 86400;
  const h = Math.floor(s / 3600); s -= h * 3600;
  const m = Math.floor(s / 60); s -= m * 60;
  const parts: string[] = [];
  if (d) parts.push(`${d}d`);
  if (h) parts.push(`${h}h`);
  if (m) parts.push(`${m}m`);
  if (s) parts.push(`${s}s`);
  return sign + parts.slice(0, 2).join(" ");
}

// ── the store (one clock per page; subscribe like lib/units.ts) ─────────────

let current: SimClockState = resetState(Date.now());
const listeners = new Set<() => void>();

function notify(): void {
  // Array.from (not spread): the repo's tsc target predates Set iteration.
  for (const fn of Array.from(listeners)) {
    try { fn(); } catch { /* one bad subscriber never breaks the clock */ }
  }
}

export function getSimClock(): SimClockState {
  return current;
}

/** Set the rate (0/1/60/3600/86400), keeping the simulated instant. */
export function setSimRate(rate: number): void {
  const next = withRate(current, rate, Date.now());
  if (next.rate === current.rate && isRealtime(current) && next.rate === 1) return; // realtime 1× → 1× is a no-op
  current = next;
  notify();
}

/** ⟲now — snap simulated time back to real time at 1×. */
export function resetSimClock(): void {
  if (isRealtime(current)) return;
  current = resetState(Date.now());
  notify();
}

export function subscribeSimClock(fn: () => void): () => void {
  listeners.add(fn);
  return () => { listeners.delete(fn); };
}

/** The simulated instant right now (Date.now through the clock).
 *  At realtime this IS Date.now(), bit-exactly. */
export function simNow(): number {
  return simNowMs(current, Date.now());
}
