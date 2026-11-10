// ENGINE KILL VISIBILITY (2026-11-10, KNOWN BROKEN #49).
//
// The Python engine carries its own kill latches (risk_kill_switch.py's
// check_kill_switches, run by bot_engine.scan_market on every Tier-2 scan):
// while `killed`, every scanner entry is suppressed and the whole tier engine
// is skipped — a full entry halt. Node only ever audited it ("KILL-SWITCH
// FIRED: ...", one line per scan) and /api/health kept reading
// `bot.status: "active"` from Node's OWN latch (state.killSwitch).
//
// Found live: the Python latch tripped "Portfolio DD -20.0%" against the
// stale ~$110.7K peak it still measures from on 2026-11-06, while Node's
// re-baselined peak ($99,962) put the account at -9.7%. Zero new entries for
// three trading days; every routine read "loop alive". CLAUDE.md's LIVENESS
// ALARM says a halt must be surfaced loudly, never discovered on a dashboard.
//
// This module is the PURE half: it shapes the scan result's kill_status into a
// snapshot, classifies the reason into an enum fit for the public /api/health
// payload (no dollar figures), decides when the two sides DISAGREE (engine
// latched, Node not), and rate-limits the reminder audit line. It changes no
// kill mechanism and no threshold: it only makes an existing state visible.
// bot.ts owns the state and the wiring (see engineKill.test.ts's wiring half).

export type EngineKillKind = "PORTFOLIO_DD" | "DAILY_LOSS" | "CONSECUTIVE_LOSS" | "MANUAL" | "OTHER";

export interface EngineKillSnapshot {
  /** risk_kill_switch.check_kill_switches().killed as the scan reported it */
  killed: boolean;
  /** its kill_reason, verbatim (percentages only — never dollars) */
  reason: string;
  kind: EngineKillKind | null;
  /** tiered_strategy.master_kill_switch — a second, narrower block */
  tierKilled: boolean;
  tierReason: string;
  /** when the scan that reported this finished (ms epoch) */
  at: number;
}

export interface EngineKillHealth {
  /** ok = engine not latched; latched = engine refuses every entry;
   *  unknown = no scan has reported since this process started */
  status: "ok" | "latched" | "unknown";
  kind: EngineKillKind | null;
  /** minutes since the scan that reported the snapshot */
  ageMinutes: number | null;
  /** Node's own latch at the same instant */
  nodeKillSwitch: boolean;
  /** engine latched while Node reads active: the state nothing surfaced */
  disagree: boolean;
  /** flips /api/health `status` to "degraded" (never the HTTP code —
   *  healthGate.ts SERVING_CHECKS stay server + database) */
  degrade: boolean;
}

/** Reason strings come from risk_kill_switch.py (check_kill_switches) —
 *  matched by their stable leading words, never parsed for numbers. */
export function classifyEngineKill(reason: unknown): EngineKillKind {
  const r = String(reason ?? "").trim();
  if (/^Portfolio DD\b/i.test(r)) return "PORTFOLIO_DD";
  if (/^Daily loss/i.test(r)) return "DAILY_LOSS";
  if (/consecutive loss/i.test(r)) return "CONSECUTIVE_LOSS";
  if (/^MANUAL\b/i.test(r)) return "MANUAL";
  return "OTHER";
}

/** The shape bot_engine.scan_market returns for the two kill fields. */
interface KillStatusLike { killed?: unknown; kill_reason?: unknown }
interface ScanResultLike { kill_status?: KillStatusLike | null; tier_kill_status?: KillStatusLike | null }

function killStatusOf(v: unknown): KillStatusLike | null {
  return v && typeof v === "object" ? (v as KillStatusLike) : null;
}

/** The scan result's kill_status / tier_kill_status as one snapshot, or null
 *  when the scan carried no kill_status at all (older engine, failed scan). */
export function engineKillSnapshot(result: unknown, nowMs: number): EngineKillSnapshot | null {
  const r: ScanResultLike = result && typeof result === "object" ? (result as ScanResultLike) : {};
  const ks = killStatusOf(r.kill_status);
  if (!ks) return null;
  const killed = Boolean(ks.killed);
  const reason = String(ks.kill_reason ?? "");
  const tk = killStatusOf(r.tier_kill_status);
  return {
    killed,
    reason,
    kind: killed ? classifyEngineKill(reason) : null,
    tierKilled: Boolean(tk?.killed),
    tierReason: tk ? String(tk.kill_reason ?? "") : "",
    at: nowMs,
  };
}

export function engineKillHealth(
  snap: EngineKillSnapshot | null,
  nodeKillSwitch: boolean,
  nowMs: number,
): EngineKillHealth {
  if (!snap) {
    return { status: "unknown", kind: null, ageMinutes: null, nodeKillSwitch, disagree: false, degrade: false };
  }
  const ageMinutes = Math.max(0, Math.round((nowMs - snap.at) / 60_000));
  const disagree = snap.killed && !nodeKillSwitch;
  return {
    status: snap.killed ? "latched" : "ok",
    kind: snap.kind,
    ageMinutes,
    nodeKillSwitch,
    // When Node is ALSO latched the existing liveness/killed path already
    // degrades health and the auto-resume evaluator owns the state; the new
    // alarm is for the case nothing covered — the engine alone.
    disagree,
    degrade: disagree,
  };
}

/** The scan already audits "KILL-SWITCH FIRED" every cycle (~2-7 min); the
 *  DISAGREEMENT line is the one a session greps for and must not be buried
 *  under its own repetitions — once per hour while the state persists. */
export const ENGINE_KILL_AUDIT_INTERVAL_MS = 60 * 60 * 1000;

export function shouldAuditEngineKill(
  health: EngineKillHealth,
  lastAuditAtMs: number,
  nowMs: number,
  intervalMs: number = ENGINE_KILL_AUDIT_INTERVAL_MS,
): boolean {
  if (!health.disagree) return false;
  return nowMs - lastAuditAtMs >= intervalMs;
}

export function engineKillAuditLine(snap: EngineKillSnapshot, health: EngineKillHealth): string {
  const tier = snap.tierKilled ? `; tier engine also blocked: ${snap.tierReason}` : "";
  return (
    `engine kill latched (${health.kind ?? "OTHER"}: ${snap.reason}) while Node's kill switch is OFF — ` +
    `every scanner entry suppressed and the tier engine skipped since this latch; ` +
    `read /api/diag/kill_state for the peak it measures from${tier} (KNOWN BROKEN #49)`
  );
}
