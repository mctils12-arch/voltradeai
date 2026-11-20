// limitsView.test — the safety limits, parsed from the shape
// GET /api/bot/safety-limits answers with (the LIVE fixture is the production
// reading of 2026-11-20 22:17Z) and from the shapes the page must survive.
import test from "node:test";
import assert from "node:assert/strict";
import { parseLimits, limitsSentence, usedWords, NEAR_FRACTION } from "./limitsView.ts";
import { KILL_CONSEQUENCE } from "../../../shared/haltWords.ts";

const LIMITS = {
  kill_pct: -20, daily_loss_pct: -3, streak_losses: 5, streak_pause_hours: 24, vxx_ratio_stop: 1.4, min_free_bp_pct: 10, correlated_group_cap_pct: 60,
  position_stop_pct: -25, position_warn_pct: -15, resume_recovery_pct: 95, resume_cooldown_hours: 4,
  engine_halt_pct: -18, engine_halt_enabled: true, engine_halt_resume_within_pct: 5, engine_halt_resume_regimes: ["BULL", "NEUTRAL"],
};
const LIVE = {
  via: "daemon",
  node: { killSwitch: false, reason: null, equityPeak: 99962.18, maxDrawdownPct: -20, equity: 86035.68, drawdownPct: -13.93 },
  engine: {
    checked_at: "2026-11-20T22:17:04.640367+00:00", equity: 86035.68, state_read: true,
    kill_switch: { killed: false, reason: "", killed_at: null, manual_kill_file: false, peak_equity: 99962.18, drawdown_pct: -13.93, kill_equity: 79969.74, resume_equity: 94964.07 },
    engine_halt: { halted: false, reason: "", halted_at: null, peak_equity: 99956.88, drawdown_pct: -13.93 },
    streak: { losses: 0, last_loss_at: null },
    day: { worst_pct: -2.41, date: "2026-11-20", is_today: true },
    limits: LIMITS,
    recent_events: [
      { at: "2026-07-07T18:11:08.708672", reason: "RESUME", equity: 107882.58, drawdown_pct: -3.45 },
      { at: "2026-11-06T15:32:44.341096", reason: "DRAWDOWN", equity: 89370.65, drawdown_pct: -20.02 },
    ],
  },
};
const clone = <T,>(v: T): T => JSON.parse(JSON.stringify(v));

test("the live reading: five guards, each with its own reading, limit and share used", () => {
  const v = parseLimits(LIVE)!;
  assert.deepEqual(v.rows.map((r) => [r.id, r.reading, r.limit, r.state]), [
    ["kill", "−13.93% from its $99,962 peak", "−20% ($79,970)", "ok"],
    ["engineKill", "−13.93% from its $99,962 peak", "−20% ($79,970)", "ok"],
    ["engineHalt", "−13.93% from its $99,957 peak", "−18% ($81,965)", "near"],
    ["day", "worst today −2.41%", "−3%", "near"],
    ["streak", "0 losses in a row", "5 in a row", "ok"],
  ]);
  const used = Object.fromEntries(v.rows.map((r) => [r.id, r.used === null ? null : Math.round(r.used * 1000) / 1000]));
  assert.deepEqual(used, { kill: 0.697, engineKill: 0.697, engineHalt: 0.774, day: 0.803, streak: 0 });
  assert.deepEqual(v.rows.map(usedWords), ["70% of the way", "70% of the way", "77% of the way", "80% of the way", "0% of the way"]);
  assert.deepEqual(limitsSentence(v), {
    text: "Near a limit: engine drawdown halt 77% of the way, daily loss 80% of the way · drawdown −13.93% of −18% · worst today −2.41% of −3% · losing streak 0 of 5",
    tone: "warn",
  });
  assert.equal(v.peaksNote, null);
  assert.equal(v.engineError, null);
  assert.equal(v.equity, 86035.68);
});

test("what each guard does at its limit is worded from the code, with the answer's own figures", () => {
  const v = parseLimits(LIVE)!;
  const effect = Object.fromEntries(v.rows.map((r) => [r.id, r.effect]));
  assert.equal(effect.kill, KILL_CONSEQUENCE);
  assert.equal(effect.engineKill, "blocks every new entry · lifts by itself only after 4 h, with the account back to $94,964 and volatility calm");
  assert.equal(effect.engineHalt, "halts new entries · lifts in a BULL or NEUTRAL market with the account within 5% of its peak");
  assert.equal(effect.day, "blocks new entries for the rest of the day · closes nothing");
  assert.equal(effect.streak, "pauses new entries for 24 h after the last loss · a winning trade resets the count");
  assert.deepEqual(v.others, [
    "Volatility stop: tiers 2 to 4 take no new entries while the VXX ratio is 1.40 or higher.",
    "Position stop: a position down 25% is marked to be closed (a warning from 15%).",
    "Correlated names: one group above 60% of buying power is warned about, and checked before each trade.",
    "Buying power: a warning when less than 10% of it is free.",
  ]);
  assert.deepEqual(v.events, ["2026-11-06 drawdown kill at $89,371 (−20.02%)", "2026-07-07 resumed at $107,883 (−3.45%)"]);
});

test("a tripped guard leads the line in the alarm tone, with the guard's own reason", () => {
  const raw = clone(LIVE);
  raw.engine.kill_switch = { ...raw.engine.kill_switch, killed: true, reason: "Portfolio DD -20.0% <= -20%", peak_equity: 110727.04, drawdown_pct: -22.3, kill_equity: 88581.63, resume_equity: 105190.69 };
  raw.engine.streak = { losses: 5, last_loss_at: "2026-11-20T15:01:00" } as never;
  const v = parseLimits(raw)!;
  const ks = v.rows.find((r) => r.id === "engineKill")!;
  assert.equal(ks.tripped, "Portfolio DD -20.0% <= -20%");
  assert.equal(usedWords(ks), "tripped");
  const streak = v.rows.find((r) => r.id === "streak")!;
  assert.equal(streak.reading, "5 losses in a row (last 2026-11-20)");
  assert.equal(usedWords(streak), "at its limit");
  const say = limitsSentence(v);
  assert.equal(say.tone, "bad");
  assert.ok(say.text.startsWith("Tripped: engine kill switch, losing streak · drawdown −22.30% of −20%"), say.text);
  // the guards now measure from different peaks: said, because that disagreement went unseen for three days once
  assert.equal(v.peaksNote, "The drawdown guards disagree by 8.4 points: each measures from its own peak.");
  const manual = clone(LIVE);
  manual.engine.kill_switch.manual_kill_file = true;
  assert.equal(parseLimits(manual)!.rows.find((r) => r.id === "engineKill")!.tripped, "manual kill file present");
  const node = clone(LIVE);
  node.node.killSwitch = true; (node.node as { reason: string | null }).reason = "Drawdown -20.1% breached -20%";
  assert.equal(parseLimits(node)!.rows[0].tripped, "Drawdown -20.1% breached -20%");
});

test("a reading that is not on record is said to be missing, never shown as zero", () => {
  const raw = clone(LIVE);
  raw.engine.state_read = false;
  raw.engine.kill_switch = { killed: null, reason: "", killed_at: null, manual_kill_file: false, peak_equity: null, drawdown_pct: null, kill_equity: null, resume_equity: null } as never;
  (raw.engine as { engine_halt: unknown }).engine_halt = null;
  raw.engine.streak = { losses: null, last_loss_at: null } as never;
  raw.engine.day = { worst_pct: null, date: null, is_today: false } as never;
  const v = parseLimits(raw)!;
  assert.deepEqual(v.rows.slice(1).map((r) => [r.id, r.reading, r.used, r.state]), [
    ["engineKill", "it has recorded no state yet", null, "unknown"],
    ["engineHalt", "it has recorded no state yet", null, "unknown"],
    ["day", "no reading on record", null, "unknown"],
    ["streak", "no count on record", null, "unknown"],
  ]);
  assert.deepEqual(v.rows.map(usedWords).slice(1), ["no reading", "no reading", "no reading", "no reading"]);
  assert.equal(limitsSentence(v).text, "No limit near · drawdown −13.93% of −20%");
  // yesterday's day figure is not today's: it is quoted with its date and uses none of today's limit
  const stale = clone(LIVE);
  stale.engine.day = { worst_pct: -3.1, date: "2026-11-19", is_today: false };
  const day = parseLimits(stale)!.rows.find((r) => r.id === "day")!;
  assert.deepEqual([day.reading, day.used, day.state], ["no reading yet today (last, 2026-11-19: −3.10%)", null, "unknown"]);
  // a day that is up uses none of the loss limit
  const up = clone(LIVE);
  up.engine.day.worst_pct = 0.4;
  assert.deepEqual([parseLimits(up)!.rows.find((r) => r.id === "day")!.used, parseLimits(up)!.rows.find((r) => r.id === "day")!.reading], [0, "worst today +0.40%"]);
  // the halt switched off in the settings is said, not measured
  const off = clone(LIVE);
  off.engine.limits.engine_halt_enabled = false;
  const halt = parseLimits(off)!.rows.find((r) => r.id === "engineHalt")!;
  assert.deepEqual([halt.reading, halt.state], ["switched off in the settings", "unknown"]);
});

test("the engine half missing: this loop's kill is still shown and the gap is said; other shapes are null", () => {
  const v = parseLimits({ via: "subprocess", node: LIVE.node, engine_error: "safety_limits_view failed" })!;
  assert.deepEqual(v.rows.map((r) => r.id), ["kill"]);
  assert.equal(v.engineError, "safety_limits_view failed");
  assert.deepEqual(v.others, []);
  assert.deepEqual(limitsSentence(v), { text: "No limit near · drawdown −13.93% of −20% · the engine's own guards did not answer", tone: "ok" });
  const blind = parseLimits({ node: { killSwitch: false, equityPeak: 99962.18, maxDrawdownPct: -20, equity: null, drawdownPct: null }, engine_error: "x" })!;
  assert.equal(blind.rows[0].reading, "account value not read");
  assert.deepEqual(limitsSentence(blind), { text: "No reading for any limit · the engine's own guards did not answer", tone: "none" });
  assert.equal(parseLimits(null), null);
  assert.equal(parseLimits([]), null);
  assert.equal(parseLimits({ error: "boom" }), null);
  assert.equal(parseLimits({ hello: 1 }), null);
  assert.equal(NEAR_FRACTION, 0.75);
});
