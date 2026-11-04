// A6 DAEMON ROUTING (2026-11-03, memory audit fix A6; territory T-BOT).
//
// WHAT WAS WRONG. Two bot.ts call sites spawned a fresh Python interpreter for
// work the resident daemon (voltrade_daemon.py) already had loaded:
//   * Tier 1's 45 s stop-state sync ran `python3 -c "from bot_engine import
//     manage_positions; ..."` through execPythonSerialized — a full bot_engine
//     import per call, ~80 cold starts per market hour — although scan_market's
//     step 7 runs the very same manage_positions() INSIDE the daemon on every
//     Tier-2 cycle (live /api/diag/timings: step7_manage_positions 0.07 s).
//   * Every per-candidate research call (`python3 analyze.py <T> --mode=scan`:
//     generateSignals and the 00:00-02:00 ET earnings loop) paid the
//     numpy/pandas/scipy/yfinance import stack for one analyze_ticker().
// Each fresh interpreter sits beside the daemon and Node in the same cgroup;
// the 2026-11-03 audit put the import baseline at ~250 MB per interpreter on
// Railway.
//
// THE FIX, as pure functions bot.ts wires in (tested in daemonRouting.test.ts):
//   * two env flags, default ON, send each site through pythonRpc first;
//   * classifyRpc() decides from the RPC envelope alone whether the daemon
//     NEVER RAN the method (socket down, disabled, busy, route missing) — the
//     only cases in which the old fresh-interpreter command may run as the
//     fallback. A daemon TIMEOUT or an in-method error never falls back: the
//     daemon's dispatch thread cannot be killed (voltrade_daemon.py's join()
//     only abandons it), so a subprocess started then would be a SECOND
//     concurrent manage_positions() racing the first on voltrade_stop_state.json
//     — a hazard the SIGKILLed subprocess of the old path never had;
//   * the Tier-1 consumer logic (which actions earn a POS-MONITOR-SYNC line)
//     lives here, so the test can feed the same JSON through both paths and
//     prove the result bot.ts consumes is identical.
// Rollback: VOLTRADE_TIER1_VIA_DAEMON=0 / VOLTRADE_RESEARCH_VIA_DAEMON=0 restore
// the fresh-interpreter paths exactly (same commands, same budgets).

export const TIER1_VIA_DAEMON_ENV = "VOLTRADE_TIER1_VIA_DAEMON";
export const RESEARCH_VIA_DAEMON_ENV = "VOLTRADE_RESEARCH_VIA_DAEMON";

/** Tier 1's stop-state sync budget: the 15 s its execPythonSerialized call always had. Both paths keep it. */
export const TIER1_SYNC_TIMEOUT_MS = 15_000;
/** One per-ticker analyze call: the 30 s the `analyze.py <T> --mode=scan` spawn always had. Both paths keep it. */
export const RESEARCH_ANALYZE_TIMEOUT_MS = 30_000;

type Env = Record<string, string | undefined>;

/** "0"/"false"/"off"/"no" -> false; "1"/"true"/"on"/"yes" -> true; unset, blank or anything else -> the default. */
export function envFlag(env: Env, name: string, defaultOn: boolean): boolean {
  const raw = env[name];
  if (raw === undefined) return defaultOn;
  const v = raw.trim().toLowerCase();
  if (v === "0" || v === "false" || v === "off" || v === "no") return false;
  if (v === "1" || v === "true" || v === "on" || v === "yes") return true;
  return defaultOn;
}

export function tier1ViaDaemon(env: Env = process.env): boolean {
  return envFlag(env, TIER1_VIA_DAEMON_ENV, true);
}

export function researchViaDaemon(env: Env = process.env): boolean {
  return envFlag(env, RESEARCH_VIA_DAEMON_ENV, true);
}

/** What pythonRpc resolves with: voltrade_daemon.py's dispatch()/handler envelope, or pythonRpc's own error. */
export interface RpcEnvelope {
  status?: unknown;
  result?: unknown;
  error_message?: unknown;
  traceback?: unknown;
}

export type RpcOutcome =
  /** The method ran and returned. */
  | { kind: "ok"; result: unknown }
  /** Never reached the daemon: socket down (ENOENT/ECONNREFUSED/...) or VOLTRADE_DAEMON_ENABLED=false. */
  | { kind: "unreachable"; message: string }
  /** Reached the daemon, which declined before running anything: busy, thread limit, unknown method, module/attr missing. */
  | { kind: "rejected"; message: string }
  /** The client (or the daemon's own join) gave up; the dispatch thread may still be running inside the daemon. */
  | { kind: "timeout"; message: string }
  /** The method ran and raised, or its output could not be parsed. */
  | { kind: "error"; message: string };

// pythonRpc's own messages: `Socket: <err>` (connect failed), `Daemon disabled via env`,
// `Daemon timeout`, `Parse: <err>`. The daemon handler's: `Daemon busy: ...`,
// `Daemon thread creation failed: ...`, `Request timed out after Ns`. dispatch()'s
// routing errors: `Unknown method: X`, `Module X not loaded`, `Method X not found
// in Y` — and, for an exception raised by the method itself, error_message plus a
// `traceback` key.
const UNREACHABLE_RE = /^(Socket:|Daemon disabled)/;
const REJECTED_RE = /^(Daemon busy|Daemon thread creation failed|Unknown method|Module \S+ not loaded|Method \S+ not found in)/;
const TIMEOUT_RE = /^(Daemon timeout|Request timed out)/;

export function classifyRpc(envelope: unknown): RpcOutcome {
  const env = (envelope && typeof envelope === "object" ? envelope : {}) as RpcEnvelope;
  if (env.status === "ok") return { kind: "ok", result: env.result };
  const message = typeof env.error_message === "string" && env.error_message
    ? env.error_message
    : `malformed RPC envelope: ${JSON.stringify(envelope ?? null).slice(0, 200)}`;
  if (UNREACHABLE_RE.test(message)) return { kind: "unreachable", message };
  if (TIMEOUT_RE.test(message)) return { kind: "timeout", message };
  if (env.traceback === undefined && REJECTED_RE.test(message)) return { kind: "rejected", message };
  return { kind: "error", message };
}

/**
 * The fresh-interpreter fallback may run only when the daemon never started the
 * method. After a timeout the dispatch thread may still be running; after an
 * in-method error a rerun would fail the same way — neither may start a second
 * concurrent run.
 */
export function rpcMayFallBack(outcome: RpcOutcome): boolean {
  return outcome.kind === "unreachable" || outcome.kind === "rejected";
}

/**
 * The POS-MONITOR-SYNC lines Tier 1 audits from a manage_positions() result —
 * the same loop bot.ts ran inline before A6 (`for (const action of (mgmt.actions
 * || [])) if (monitoredPositions[action.ticker]) audit(...)`), over the parsed
 * subprocess stdout or the daemon envelope's `result` alike. Bot_engine's actions
 * are never executed here: the WS handler fires exits; this only refreshes the
 * on-disk stop state syncMonitoredPositions() reads.
 */
export function posMonitorSyncLines(mgmt: unknown, isMonitored: (ticker: string) => boolean): string[] {
  const actions = (mgmt as { actions?: unknown } | null | undefined)?.actions;
  if (!Array.isArray(actions)) return [];
  const lines: string[] = [];
  for (const action of actions as Array<{ ticker?: unknown; type?: unknown } | null>) {
    const ticker = action?.ticker;
    if (typeof ticker !== "string" || !ticker || !isMonitored(ticker)) continue;
    lines.push(`bot_engine flagged ${ticker} (${String(action?.type)}) — WS monitor should handle`);
  }
  return lines;
}

export type SyncVia = "daemon" | "subprocess";

/**
 * One TIER1-SYNC audit line per path CHANGE (never per cycle: Tier 1 runs every
 * 45 s). The daemon path is the expected state, so a boot that lands there says
 * nothing; a boot that lands on the fresh interpreter says why.
 */
export function viaTransitionLine(prev: SyncVia | null, next: SyncVia, reason?: string): string | null {
  if (prev === next) return null;
  const why = reason ?? "daemon unavailable";
  if (next === "subprocess") {
    return prev === null
      ? `stop-state sync running in a fresh interpreter (${why})`
      : `stop-state sync fell back to a fresh interpreter (${why})`;
  }
  return prev === null ? null : "stop-state sync back on the daemon RPC";
}

/** analyze.py's analyze_ticker() result, typed only as far as bot.ts reads it (the CLI and the daemon route print the same dict). */
export interface AnalyzeTickerJson {
  ticker?: string;
  error?: string;
  vrp?: number | null;
  recommendation?: { action?: string; reasoning?: string; signal?: string } | null;
  edge_factors?: { squeeze_score?: number | null; squeeze_desc?: string } | null;
  earnings_intel?: { days_to_earnings: number; timing?: string; beat_pct?: number | null; options_edge: number } | null;
}
