// Bot page — the safety limits ([PRODUCT] 2026-11-20, UI coverage batch A,
// third part). GET /api/bot/safety-limits: this loop's drawdown kill (`node`)
// and the engine's own guards (`engine`, safety_limits_view.py): its kill
// switch and drawdown halt measured from their own peaks, the day's worst
// loss against the daily limit, the losing streak against its pause, and the
// limits that have no reading in the answer (volatility stop, free buying
// power, correlated names, the per-position stop).
//
// Pure; pinned by limitsView.test.ts. Every figure is a route field: a limit
// is never restated here, a reading that is not on record is said to be
// missing (never shown as 0), and what a guard does at its limit is worded
// from the code that enforces it (shared/haltWords for this loop's kill).
import { fmtMoney, fmtPct } from "./format";
import { KILL_CONSEQUENCE } from "../../../shared/haltWords";

const fin = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const str = (v: unknown): string | null => (typeof v === "string" && v.trim() ? v.trim() : null);
const obj = (v: unknown): Record<string, unknown> | null => (v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : null);
/** a percentage without trailing zeros: −20%, −2.41% */
const pct = (v: number): string => fmtPct(v, Number.isInteger(v) ? 0 : 2);
const day10 = (iso: string): string => iso.slice(0, 10);

export type LimitId = "kill" | "engineKill" | "engineHalt" | "day" | "streak";
export type LimitState = "ok" | "near" | "hit" | "unknown";
/** a guard is "near" from three quarters of the way to its limit */
export const NEAR_FRACTION = 0.75;
export interface LimitRow {
  id: LimitId; name: string;
  /** the guard's own reading, in words */
  reading: string;
  /** the limit, in words; null when the answer does not carry it */
  limit: string | null;
  /** share of the limit used (0 = none, 1 = at the limit, may exceed 1); null = no reading */
  used: number | null;
  state: LimitState;
  /** what the guard does at its limit */
  effect: string;
  /** set while the guard itself says it is tripped: its own reason */
  tripped: string | null;
}
export interface LimitsView {
  rows: LimitRow[];
  /** limits the answer carries no reading for, one sentence each */
  others: string[];
  equity: number | null;
  /** the engine half did not answer: only this loop's kill is shown */
  engineError: string | null;
  /** set when the drawdown guards measure from peaks that differ */
  peaksNote: string | null;
  events: string[];
}

function stateOf(used: number | null, tripped: boolean): LimitState {
  if (tripped) return "hit";
  if (used === null) return "unknown";
  return used >= 1 ? "hit" : used >= NEAR_FRACTION ? "near" : "ok";
}
/** share of a NEGATIVE limit a reading has used: a gain uses none of it */
const usedOf = (reading: number | null, limit: number | null): number | null => (reading === null || limit === null || !(limit < 0) ? null : Math.max(0, reading / limit));

function drawdownRow(id: LimitId, name: string, dd: number | null, limit: number | null, peak: number | null, tripped: string | null, effect: string, absent: string): LimitRow {
  const used = usedOf(dd, limit);
  return {
    id, name, used, tripped, effect, state: stateOf(used, tripped !== null),
    reading: dd === null ? absent : `${pct(dd)}${peak !== null ? ` from its ${fmtMoney(peak)} peak` : ""}`,
    limit: limit === null ? null : `${pct(limit)}${peak !== null ? ` (${fmtMoney(peak * (1 + limit / 100))})` : ""}`,
  };
}

const EVENT_WORDS: Readonly<Record<string, string>> = { DRAWDOWN: "drawdown kill", RESUME: "resumed", MANUAL: "manual kill" };

/** GET /api/bot/safety-limits → the view; null when the body is not the route's shape. */
export function parseLimits(raw: unknown): LimitsView | null {
  const o = obj(raw);
  const node = obj(o?.node), eng = obj(o?.engine);
  if (!o || (!node && !eng)) return null;
  const lim = obj(eng?.limits) ?? {};
  const num = (v: unknown): number | null => (fin(v) ? v : null);
  const rows: LimitRow[] = [];

  if (node) {
    const reason = str(node.reason);
    rows.push(drawdownRow("kill", "Drawdown kill", num(node.drawdownPct), num(node.maxDrawdownPct), fin(node.equityPeak) && node.equityPeak > 0 ? node.equityPeak : null,
      node.killSwitch === true ? reason ?? "tripped" : null, KILL_CONSEQUENCE, "account value not read"));
  }
  if (eng) {
    const ks = obj(eng.kill_switch) ?? {};
    const recorded = eng.state_read === true;
    const cooldown = num(lim.resume_cooldown_hours);
    const resumeAt = num(ks.resume_equity);
    const ksTripped = ks.manual_kill_file === true ? "manual kill file present" : ks.killed === true ? str(ks.reason) ?? "tripped" : null;
    rows.push(drawdownRow("engineKill", "Engine kill switch", recorded ? num(ks.drawdown_pct) : null, num(lim.kill_pct), recorded ? num(ks.peak_equity) : null, ksTripped,
      `blocks every new entry${cooldown !== null && resumeAt !== null ? ` · lifts by itself only after ${cooldown} h, with the account back to ${fmtMoney(resumeAt)} and volatility calm` : ""}`,
      recorded ? "no reading (account value or peak missing)" : "it has recorded no state yet"));

    const eh = obj(eng.engine_halt);
    const within = num(lim.engine_halt_resume_within_pct);
    const regimes = Array.isArray(lim.engine_halt_resume_regimes) ? lim.engine_halt_resume_regimes.map(str).filter((x): x is string => !!x) : [];
    const halt = drawdownRow("engineHalt", "Engine drawdown halt", eh ? num(eh.drawdown_pct) : null, num(lim.engine_halt_pct), eh ? num(eh.peak_equity) : null,
      eh?.halted === true ? str(eh.reason) ?? "halted" : null,
      `halts new entries${within !== null && regimes.length ? ` · lifts in a ${regimes.join(" or ")} market with the account within ${within}% of its peak` : ""}`,
      eh ? "no reading (account value or peak missing)" : "it has recorded no state yet");
    if (lim.engine_halt_enabled === false) { halt.reading = "switched off in the settings"; halt.used = null; halt.state = "unknown"; }
    rows.push(halt);

    const day = obj(eng.day) ?? {};
    const worst = num(day.worst_pct), dayLimit = num(lim.daily_loss_pct), date = str(day.date);
    const today = day.is_today === true && worst !== null;
    const dayUsed = today ? usedOf(worst, dayLimit) : null;
    rows.push({
      id: "day", name: "Daily loss", used: dayUsed, tripped: null, state: stateOf(dayUsed, false),
      reading: today ? `worst today ${pct(worst!)}` : worst !== null && date ? `no reading yet today (last, ${date}: ${pct(worst)})` : "no reading on record",
      limit: dayLimit === null ? null : pct(dayLimit),
      effect: "blocks new entries for the rest of the day · closes nothing",
    });

    const streak = obj(eng.streak) ?? {};
    const losses = num(streak.losses), maxLosses = num(lim.streak_losses), pause = num(lim.streak_pause_hours), lastLoss = str(streak.last_loss_at);
    const streakUsed = losses !== null && maxLosses !== null && maxLosses > 0 ? losses / maxLosses : null;
    rows.push({
      id: "streak", name: "Losing streak", used: streakUsed, tripped: null, state: stateOf(streakUsed, false),
      reading: losses === null ? "no count on record" : `${losses} ${losses === 1 ? "loss" : "losses"} in a row${losses > 0 && lastLoss ? ` (last ${day10(lastLoss)})` : ""}`,
      limit: maxLosses === null ? null : `${maxLosses} in a row`,
      effect: `pauses new entries${pause !== null ? ` for ${pause} h after the last loss` : ""} · a winning trade resets the count`,
    });
  }

  const others: string[] = [];
  const vxx = num(lim.vxx_ratio_stop), bp = num(lim.min_free_bp_pct), corr = num(lim.correlated_group_cap_pct), stop = num(lim.position_stop_pct), warn = num(lim.position_warn_pct);
  if (vxx !== null) others.push(`Volatility stop: tiers 2 to 4 take no new entries while the VXX ratio is ${vxx.toFixed(2)} or higher.`);
  if (stop !== null) others.push(`Position stop: a position down ${Math.abs(stop)}% is marked to be closed${warn !== null ? ` (a warning from ${Math.abs(warn)}%)` : ""}.`);
  if (corr !== null) others.push(`Correlated names: one group above ${corr}% of buying power is warned about, and checked before each trade.`);
  if (bp !== null) others.push(`Buying power: a warning when less than ${bp}% of it is free.`);

  // the three drawdown guards keep their own peaks: say so when they disagree (KNOWN BROKEN #49 was exactly this, unseen)
  const dds = [num(node?.drawdownPct), eng ? num(obj(eng.kill_switch)?.drawdown_pct) : null, eng ? num(obj(eng.engine_halt)?.drawdown_pct) : null].filter((x): x is number => x !== null);
  const spread = dds.length > 1 ? Math.max(...dds) - Math.min(...dds) : 0;
  const events: string[] = [];
  for (const e of Array.isArray(eng?.recent_events) ? eng!.recent_events : []) {
    const r = obj(e); const at = r ? str(r.at) : null, reason = r ? str(r.reason) : null;
    if (!r || !at || !reason) continue;
    events.push(`${day10(at)} ${EVENT_WORDS[reason] ?? reason.toLowerCase()}${fin(r.equity) ? ` at ${fmtMoney(r.equity)}` : ""}${fin(r.drawdown_pct) ? ` (${pct(r.drawdown_pct)})` : ""}`);
  }
  return {
    rows, others, events: events.reverse(), equity: num(node?.equity) ?? num(eng?.equity),
    engineError: eng ? null : str(o.engine_error) ?? "the engine's guards did not answer",
    peaksNote: spread > 0.5 ? `The drawdown guards disagree by ${spread.toFixed(1)} points: each measures from its own peak.` : null,
  };
}

const short = (r: LimitRow): string => `${r.name.toLowerCase()} ${Math.round((r.used ?? 0) * 100)}% of the way`;
/** The block's one line: what is tripped or near, then the three headline readings. */
export function limitsSentence(v: LimitsView): { text: string; tone: "ok" | "warn" | "bad" | "none" } {
  const hit = v.rows.filter((r) => r.state === "hit"), near = v.rows.filter((r) => r.state === "near");
  const known = v.rows.filter((r) => r.state !== "unknown");
  const lead = hit.length ? `Tripped: ${hit.map((r) => r.name.toLowerCase()).join(", ")}`
    : near.length ? `Near a limit: ${near.map(short).join(", ")}`
    : known.length ? "No limit near" : "No reading for any limit";
  const parts: string[] = [];
  const by = (id: LimitId) => v.rows.find((r) => r.id === id);
  const dd = [by("kill"), by("engineKill"), by("engineHalt")].filter((r): r is LimitRow => !!r && r.used !== null).sort((a, b) => b.used! - a.used!)[0];
  if (dd && dd.limit) parts.push(`drawdown ${dd.reading.split(" from ")[0]} of ${dd.limit.split(" (")[0]}`);
  const day = by("day");
  if (day && day.used !== null && day.limit) parts.push(`${day.reading} of ${day.limit}`);
  const streak = by("streak");
  if (streak && streak.used !== null && streak.limit) parts.push(`losing streak ${streak.reading.split(" ")[0]} of ${streak.limit.split(" ")[0]}`);
  if (v.engineError) parts.push("the engine's own guards did not answer");
  return { text: [lead, ...parts].join(" · "), tone: hit.length ? "bad" : near.length ? "warn" : known.length ? "ok" : "none" };
}
/** "80% of the way" / "at its limit" / "no reading" for a row's meter label */
export function usedWords(r: LimitRow): string {
  if (r.tripped !== null) return "tripped";
  if (r.used === null) return "no reading";
  return r.used >= 1 ? "at its limit" : `${Math.round(r.used * 100)}% of the way`;
}
