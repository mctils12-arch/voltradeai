// Bot page v2 — the director decides WHEN the page may re-arrange. Pure
// (tested with a manual clock). Rules from the NOC / attention research:
// an incident takes the layout immediately; every other change waits a
// minimum dwell; leaving an incident waits a recede grace; nothing moves
// while the user holds the layout.
import type { PageState } from "./pageState";

export interface DirectorOptions { minDwellMs?: number; recedeGraceMs?: number }

export class Director {
  readonly minDwellMs: number;
  readonly recedeGraceMs: number;
  current: PageState | null = null;
  private shownSince = 0;
  private candidate: PageState | null = null;
  private candidateSince = 0;
  log: Array<{ t: number; from: PageState | null; to: PageState; why: string }> = [];

  constructor(opts: DirectorOptions = {}) {
    this.minDwellMs = opts.minDwellMs ?? 8_000;
    this.recedeGraceMs = opts.recedeGraceMs ?? 20_000;
  }

  private apply(to: PageState, t: number, why: string): PageState {
    this.log.push({ t, from: this.current, to, why });
    if (this.log.length > 50) this.log.shift();
    this.current = to;
    this.shownSince = t;
    this.candidate = null;
    return to;
  }

  /** Feed the derived state every tick; returns the state to display. */
  propose(state: PageState, nowMs: number, hold = false): PageState {
    if (this.current === null) return this.apply(state, nowMs, "first");
    if (state === this.current) { this.candidate = null; return this.current; }
    if (hold) { this.candidate = null; return this.current; }
    if (state === "incident") return this.apply(state, nowMs, "incident takes the layout");
    if (this.candidate !== state) { this.candidate = state; this.candidateSince = nowMs; }
    const dwelt = nowMs - this.shownSince >= this.minDwellMs;
    const leavingIncident = this.current === "incident";
    const persisted = nowMs - this.candidateSince >= (leavingIncident ? this.recedeGraceMs : 0);
    if (dwelt && persisted) return this.apply(state, nowMs, leavingIncident ? "incident receded" : "dwell satisfied");
    return this.current;
  }
}
