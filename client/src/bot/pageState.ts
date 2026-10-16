// Bot page v2 — the page state machine, derived from REAL audit rows and
// status fields only (research/bot_cockpit_design_brief_v2.md §3). Pure.
export type PageState = "closed" | "idle" | "scanning" | "candidate" | "placed" | "exit" | "incident";

export interface AuditRow { time: string; type: string; message: string }

export interface StateInput {
  audit: AuditRow[];                 // any order; newest resolved by time
  killSwitch: boolean;
  active: boolean;
  marketOpen: boolean | null;        // null = unknown
  nowMs: number;
  modelAgeHours?: number | null;
  fallbackAtHours?: number | null;
  healthDegraded?: boolean;
}

export interface IncidentInfo { kind: "kill" | "halt" | "model" | "scanner" | "stream" | "liveness"; row: AuditRow | null; detail: string }

export interface DerivedState {
  state: PageState;
  since: string | null;
  detail: string;
  scan: { startedAt: string | null; intervalMin: number | null; nextAtMs: number | null; elapsedMs: number | null; scanned: number | null; candidates: number | null };
  incident: IncidentInfo | null;
}

const TRADE_TYPES = new Set(["TRADE", "ETF-TRADE", "OPTIONS-TRADE", "MORNING-TRADE", "OPTIONS-SCANNER-EXEC", "T1", "T2"]);
const EXIT_TYPES = new Set(["WS-EXIT", "EXIT-FILL", "POS-KILL", "POS-KILL-OPTION", "OPTIONS-EXIT", "STOP-LOSS", "TAKE-PROFIT", "TRAILING-STOP"]);

export const SCAN_STALE_MS = 10 * 60_000;
export const TRADE_WINDOW_MS = 15 * 60_000;
export const STREAM_DEAD_MS = 60_000;

function t(row: AuditRow | null | undefined): number {
  if (!row) return NaN;
  const ms = Date.parse(row.time);
  return Number.isFinite(ms) ? ms : NaN;
}

export function sortedNewestFirst(rows: AuditRow[]): AuditRow[] {
  return [...rows].filter((r) => r && typeof r.time === "string").sort((a, b) => t(b) - t(a));
}

export function parseScanSummary(message: string): { scanned: number | null; candidates: number | null } {
  const m = /Scanned\s+([\d,]+)\s+stocks,\s+(\d+)\s+trade candidates/i.exec(message || "");
  if (!m) return { scanned: null, candidates: null };
  return { scanned: Number(m[1].replace(/,/g, "")), candidates: Number(m[2]) };
}

export function parseScanInterval(message: string): number | null {
  const m = /interval:\s*(\d+(?:\.\d+)?)\s*min/i.exec(message || "");
  return m ? Number(m[1]) : null;
}

export function derivePageState(i: StateInput): DerivedState {
  const rows = sortedNewestFirst(i.audit || []);
  const now = i.nowMs;
  const first = (pred: (r: AuditRow) => boolean) => rows.find(pred) || null;

  const lastStart = first((r) => r.type === "TIER2" && /^Starting scan/i.test(r.message));
  const lastDone = first((r) => r.type === "TIER2" && /^Scanned\s/i.test(r.message));
  const summary = lastDone ? parseScanSummary(lastDone.message) : { scanned: null, candidates: null };
  const intervalMin = lastStart ? parseScanInterval(lastStart.message) : null;
  const startedMs = t(lastStart);
  const scan = {
    startedAt: lastStart ? lastStart.time : null,
    intervalMin,
    nextAtMs: Number.isFinite(startedMs) && intervalMin ? startedMs + intervalMin * 60_000 : null,
    elapsedMs: Number.isFinite(startedMs) ? now - startedMs : null,
    scanned: summary.scanned,
    candidates: summary.candidates,
  };

  // ── incidents (top alert level only reconfigures the layout) ──
  let incident: IncidentInfo | null = null;
  const killRow = first((r) => r.type === "DRAWDOWN-KILL");
  if (i.killSwitch) incident = { kind: "kill", row: killRow, detail: killRow ? killRow.message : "kill switch latched" };
  const haltRow = first((r) => r.type === "DD-HALT");
  if (!incident && haltRow && now - t(haltRow) < 24 * 3600_000) incident = { kind: "halt", row: haltRow, detail: haltRow.message };
  if (!incident && typeof i.modelAgeHours === "number" && typeof i.fallbackAtHours === "number" && i.modelAgeHours > i.fallbackAtHours) {
    incident = { kind: "model", row: null, detail: `model ${i.modelAgeHours.toFixed(1)} h old · scoring on the rules fallback past ${i.fallbackAtHours} h` };
  }
  if (!incident) {
    const fails = rows.filter((r) => (r.type === "TIER2-FAIL" || r.type === "TIER2-ERROR") && now - t(r) < 3600_000);
    if (fails.length >= 3 || rows.some((r) => r.type === "TIER2-FAIL" && now - t(r) < SCAN_STALE_MS)) {
      incident = { kind: "scanner", row: fails[0] || null, detail: `${fails.length} scan failures in the last hour` };
    }
  }
  if (!incident) {
    const disc = first((r) => r.type === "STREAM-DISCONNECT");
    const live = first((r) => r.type === "STREAM" && /feed live/i.test(r.message));
    if (disc && now - t(disc) > STREAM_DEAD_MS && (!live || t(live) < t(disc)) && i.marketOpen) {
      incident = { kind: "stream", row: disc, detail: "real-time feed down · stops starve" };
    }
  }

  if (incident) return { state: "incident", since: incident.row?.time ?? null, detail: incident.detail, scan, incident };

  if (i.marketOpen === false) return { state: "closed", since: null, detail: "market closed", scan, incident: null };

  if (lastStart && (!lastDone || t(lastStart) > t(lastDone)) && now - t(lastStart) < SCAN_STALE_MS) {
    return { state: "scanning", since: lastStart.time, detail: "tier 2 scan running", scan, incident: null };
  }

  const trade = first((r) => TRADE_TYPES.has(r.type));
  const exitRow = first((r) => EXIT_TYPES.has(r.type));
  if (exitRow && now - t(exitRow) < TRADE_WINDOW_MS && (!trade || t(exitRow) >= t(trade))) {
    return { state: "exit", since: exitRow.time, detail: exitRow.message, scan, incident: null };
  }
  if (trade && now - t(trade) < TRADE_WINDOW_MS) {
    return { state: "placed", since: trade.time, detail: trade.message, scan, incident: null };
  }

  const queued = first((r) => r.type === "QUEUE");
  if ((lastDone && (summary.candidates ?? 0) > 0 && now - t(lastDone) < SCAN_STALE_MS) || (queued && now - t(queued) < SCAN_STALE_MS)) {
    return { state: "candidate", since: (lastDone || queued)!.time, detail: `${summary.candidates ?? ""} candidates`, scan, incident: null };
  }

  return { state: "idle", since: lastDone ? lastDone.time : null, detail: i.active ? "between scans" : "bot paused", scan, incident: null };
}
