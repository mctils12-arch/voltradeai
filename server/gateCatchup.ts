/**
 * gateCatchup.ts — one rule for "a weekly gate job's registered Sunday run died
 * on plumbing; may it try again before next Sunday?" (pure; no I/O).
 *
 * WHY: the weekly ladder jobs act once per ISO week inside a Sunday window. On
 * 2026-11-15/16 two first registered runs died in under 100 ms each (a script
 * the image does not ship; a bars feed the endpoint refuses) — neither read a
 * return, and each then waited a full week. A week of delay for a failure that
 * spent no look at the data buys nothing (open_questions.md 2026-11-16).
 *
 * THE RULE (all must hold):
 *   1. it is Monday or Tuesday ET, inside the job's own clock window;
 *   2. the job has at least one run since the preceding Sunday's window opened,
 *      and EVERY run since then ended "error" — an "ok" or "skipped" run (a
 *      verdict, a fresh result, a Python-side skip) never retries;
 *   3. no result file was written since that Sunday window opened (a run that
 *      wrote a result and then errored has spent its look);
 *   4. no attempt has started yet on this ET day — so at most two catch-up
 *      attempts a week (Mon, Tue), and a deploy cannot multiply them: the
 *      evidence is the persisted heartbeat, not process memory.
 * One look stays one look: nothing here changes a registered statistic, and a
 * result (PASS, FAIL or WAITING) always ends the week.
 *
 * NOT COVERED, on purpose: a Sunday on which the job never started at all (a
 * deploy across the whole window). Catching that up would let a newly deployed
 * job run on a weekday before its registered not_before Sunday.
 *
 * WHO OPTS IN is decided per job, because the Sunday windows were chosen for an
 * idle lane: github_gate2_weekly (in-process, one paged request, seconds) does;
 * earnings_qoq_weekly (~1,000 per-symbol bar requests against the 200/min
 * account limit the live bot shares on a weekday evening), lazy_prices_weekly
 * and edge_p2_weekly (1.5 GB, hours) do not until a free weekday lane is shown.
 *
 * OFF SWITCH: VOLTRADE_GATE_CATCHUP=0 (default on).
 */
import { zonedInstant, zonedParts, type JobRun } from "./jobHeartbeat";

const ET = "America/New_York";
/** Days after the registered Sunday on which a catch-up may start. */
export const CATCHUP_WEEKDAYS: Readonly<Record<string, number>> = { Mon: 1, Tue: 2 };

export function gateCatchupFlag(env: Record<string, string | undefined>): { value: string; source: "env" | "default"; on: boolean } {
  const raw = env.VOLTRADE_GATE_CATCHUP;
  const value = raw === undefined ? "1" : String(raw);
  const v = value.trim().toLowerCase();
  return { value, source: raw === undefined ? "default" : "env", on: v === "1" || v === "true" || v === "on" || v === "yes" };
}

export interface CatchupInput {
  enabled: boolean;
  /** the job's persisted heartbeat runs (oldest first), or undefined when none */
  runs: readonly JobRun[] | undefined;
  /** generated_at of the job's result file, or null when there is none */
  latestGeneratedAt: string | null;
  /** the job's Sunday clock window, minutes after ET midnight, [start, end) */
  windowStartMin: number;
  windowEndMin: number;
}

export type CatchupDecision =
  | { eligible: false; reason: string }
  | { eligible: true; reason: string; registeredSunday: string; etDate: string };

function shiftDate(date: string, days: number): string {
  const [y, mo, d] = date.split("-").map(Number);
  return new Date(Date.UTC(y, mo - 1, d + days)).toISOString().slice(0, 10);
}

/** Pure: may this tick start a catch-up attempt for last Sunday's registered run? */
export function catchupDecision(nowMs: number, s: CatchupInput): CatchupDecision {
  if (!s.enabled) return { eligible: false, reason: "VOLTRADE_GATE_CATCHUP off" };
  const p = zonedParts(nowMs, ET);
  const daysAfter = CATCHUP_WEEKDAYS[p.weekday];
  if (daysAfter === undefined || p.minutes < s.windowStartMin || p.minutes >= s.windowEndMin) {
    return { eligible: false, reason: "outside the Mon/Tue catch-up window" };
  }
  const registeredSunday = shiftDate(p.date, -daysAfter);
  const sundayOpen = zonedInstant(registeredSunday, s.windowStartMin, ET);
  const since = (s.runs ?? []).filter((r) => r.startedAt >= sundayOpen);
  if (since.length === 0) return { eligible: false, reason: `no attempt since the ${registeredSunday} window opened — nothing to catch up` };
  const settled = since.find((r) => r.status !== "error");
  if (settled) {
    return { eligible: false, reason: `the ${registeredSunday} run is settled or still open (${settled.status ?? "running"}) — one look stays one look` };
  }
  const resultAt = s.latestGeneratedAt ? Date.parse(s.latestGeneratedAt) : NaN;
  if (Number.isFinite(resultAt) && resultAt >= sundayOpen) {
    return { eligible: false, reason: `a result was written ${s.latestGeneratedAt} — the look is spent` };
  }
  if (since.some((r) => zonedParts(r.startedAt, ET).date === p.date)) {
    return { eligible: false, reason: `already attempted today (${p.date})` };
  }
  return {
    eligible: true, registeredSunday, etDate: p.date,
    reason: `catch-up for the ${registeredSunday} run: ${since.length} attempt(s) since, all errored, no result written`,
  };
}
