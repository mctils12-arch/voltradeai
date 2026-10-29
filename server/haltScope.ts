// HALT SCOPE (2026-10-27, audit §6.8 N15) — the ONE decision for what a halt
// blocks in bot.ts.
//
// THE BUG: every Node halt used to gate EXITS as well as entries. While the
// drawdown kill was latched (it also writes state.active = false), while the
// circuit breaker was open, and while the weekly-loss pause held:
//   - checkPositionOnTick (WS stops, scale-outs, trails, breakeven, time stop)
//     returned on `!state.active || state.killSwitch`;
//   - the 60s position sync (TIME-EXIT, POS-KILL) returned on the same test;
//   - the Tier 1 interval skipped tier1Reflex (stop-state refresh, sync) and,
//     under the circuit breaker, the -10% drawdown check itself;
//   - the WS feed did not reconnect after a drop.
// So the moment a halt fired because positions were losing money, nothing
// protected those positions any more. The Python halts (bot_engine DD halt,
// risk_kill_switch) document the opposite contract: "only entries are gated,
// not exits".
//
// THE CONTRACT (pure, unit-tested in haltScope.test.ts):
//   - entry   (new positions: morning queue, stream signals, Tier 2 scan
//              dispatch) is allowed only when nothing is halted;
//   - exit    (protective exits on positions we already hold) is allowed in
//              every halt; only an explicit owner STOP (/api/bot/stop) turns
//              it off, because the owner stopped the bot to take manual
//              control and automatic exits would fight them;
//   - ddCheck (the -10%-from-peak drawdown kill check) runs in every halt
//              except while the kill is already latched — then the paper
//              auto-resume evaluation owns that tick (killSwitchAutoResume.ts),
//              exactly as before.
// Order transmission, the kill trip, the order cancel on a trip and the -25%
// liquidate-on-kill rule are untouched: this only decides WHICH paths run.

export type HaltAction = "entry" | "exit" | "ddCheck";

export interface HaltState {
  /** the owner pressed /api/bot/stop (in-memory, cleared by /api/bot/start) */
  ownerStopped: boolean;
  /** state.killSwitch — the drawdown kill or the owner kill toggle is latched */
  killed: boolean;
  /** state.circuitBreakerUntil is in the future (3 consecutive stops) */
  circuitBreaker: boolean;
  /**
   * state.active is false but the owner did not stop the bot: a system halt
   * wrote it (the weekly-loss pause; the drawdown kill writes it as well).
   */
  paused: boolean;
}

/** Build a HaltState from bot.ts's raw flags (pure, so the mapping is tested). */
export function haltStateFrom(raw: {
  ownerStopped: boolean;
  active: boolean;
  killSwitch: boolean;
  circuitBreakerUntil: number;
  nowMs: number;
}): HaltState {
  return {
    ownerStopped: raw.ownerStopped,
    killed: raw.killSwitch,
    circuitBreaker: raw.circuitBreakerUntil > raw.nowMs,
    paused: !raw.active && !raw.ownerStopped,
  };
}

export function haltAllows(s: HaltState, action: HaltAction): boolean {
  if (s.ownerStopped) return false;
  switch (action) {
    case "entry":
      return !s.killed && !s.circuitBreaker && !s.paused;
    case "exit":
      return true;
    case "ddCheck":
      return !s.killed;
  }
}

/**
 * "full"       — normal operation (entries + exits);
 * "exits_only" — a halt is active: run protective exits, place no entries;
 * "off"        — the owner stopped the bot.
 */
export type HaltMode = "full" | "exits_only" | "off";

export function haltMode(s: HaltState): HaltMode {
  if (haltAllows(s, "entry")) return "full";
  if (haltAllows(s, "exit")) return "exits_only";
  return "off";
}
