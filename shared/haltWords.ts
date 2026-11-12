// HALT WORDS (2026-11-06, rule call "kill-toggle scope", decided under the
// human's "fix all of this") — the ONE source of the words the owner reads
// about what a kill latch and an owner STOP do. The bot page and the
// server's audit / notification / email lines all import these, and
// server/haltWords.test.ts ties every claim to server/haltScope.ts's truth
// table, so the words cannot drift from the code again.
//
// WHY: since 2026-10-27 (audit N15, server/haltScope.ts) a kill latch — the
// owner kill toggle or the drawdown kill, one flag — blocks NEW ENTRIES only:
// WebSocket stops, scale-outs, trails, TIME-EXIT, POS-KILL and the options
// manager's exits keep running, and since the human's 2026-11-05 decision the
// floor may only REDUCE exposure while it is tripped (no buys, no leverage).
// Only an owner STOP (/api/bot/stop) freezes everything, exits included. The
// page said the reverse: the kill "stops all exits", "no exits of any kind",
// the bot "is not ... managing exits", and STOP "stops keep firing".

/** The kill latch's consequence, one line (bot page issue + incident panel). */
export const KILL_CONSEQUENCE =
  "no new entries · protective exits keep running (stops, option exits; the floor may only reduce) · open orders cancelled";

/** The kill banner's body text. */
export const KILL_BANNER_TEXT =
  "New entries are blocked. Protective exits keep running: stops, option exits, and the floor may only reduce exposure. Click Resume to clear the latch and re-activate.";

/** Confirm text before the owner presses Kill. */
export const KILL_CONFIRM =
  "cancels every open order and blocks new entries until resumed; stops, option exits and the floor's reduce-only pass keep running";

/** Confirm text before the owner presses Stop. */
export const STOP_CONFIRM =
  "freezes everything — no scans, no entries and no exits, stops included — until Start or the next deploy";

/** Glossary: what the drawdown kill line does (completes "The kill line … is"). */
export const KILL_LINE_GLOSS =
  "the drawdown at which the bot stops opening new positions automatically; protective exits keep running";

// ── server lines (audit / notification / email) ─────────────────────────────

/** Audit line when the owner turns the kill switch ON. */
export const KILL_ON_AUDIT =
  "New entries halted; protective exits keep running (stops, option exits, floor reduce-only). Cancelling open orders.";

/** In-app notification when the owner turns the kill switch ON. */
export const KILL_ON_NOTIFY =
  "KILL SWITCH ACTIVATED — new entries halted, open orders cancelled; protective exits keep running";

/** Email body when the owner turns the kill switch ON. */
export const KILL_ON_EMAIL =
  "Kill switch manually activated. New entries are blocked and open orders were cancelled. Protective exits keep running (stops, option exits; the floor may only reduce exposure). Only /api/bot/stop freezes everything.";

/** Sentence appended to a DRAWDOWN-KILL audit line. */
export const DRAWDOWN_KILL_SUFFIX = "New entries stopped; protective exits keep running.";

/** Subject of the drawdown-kill email. */
export const DRAWDOWN_KILL_EMAIL_SUBJECT = "MAX DRAWDOWN — New Entries Stopped";

/** Why the pre-trade check refuses a trade while the latch is set. */
export const KILL_PRETRADE_REASON =
  "Kill switch is active: new entries are blocked (protective exits keep running).";
