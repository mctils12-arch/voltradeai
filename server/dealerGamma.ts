/**
 * dealerGamma.ts — Node side of the Edge P1 dealer-gamma short-premium filter
 * (dealer_gamma.py, 2026-11-06). Python decides and writes; this module only
 * READS what it wrote and shapes:
 *   (a) the GAMMA-SKIP / GAMMA-WOULD-SKIP audit rows from a scan result's
 *       `dealer_gamma` field (one row per NEW record — Python already bounds
 *       repeats to once per ET day per ticker, setup and action), and
 *   (b) the read-only `gamma` diag probe body: the filter's latest state, its
 *       plain-English label, recent decisions and per-day counts, beside the
 *       squeeze_gex archive's own freshness (server/squeezeGex.ts).
 * No raw GEX value is read or served: SqueezeMetrics' DIX.csv is their
 * published ESTIMATE, licence unverified — rank, date and verdict only.
 * Pure except for the two file readers (node:test safe).
 */
import fs from "fs";
import path from "path";
import type { SqueezeReport } from "./squeezeGex";

export const DEALER_GAMMA_STATE_FILE = "voltrade_dealer_gamma_state.json";      // storage_config.DEALER_GAMMA_STATE_PATH
export const DEALER_GAMMA_DECISIONS_FILE = "voltrade_dealer_gamma_decisions.jsonl"; // storage_config.DEALER_GAMMA_DECISIONS_PATH
export const DEALER_GAMMA_PROBE_DECISIONS = 50;
export const DEALER_GAMMA_AUDIT_ROWS_MAX = 20;
export const DEALER_GAMMA_TAIL_BYTES = 256 * 1024;

export interface DealerGammaState {
  asOf?: string | null;
  rank252?: number | null;
  lowGamma?: boolean;
  threshold?: number;
  source?: string;
  stale?: boolean;
  ageSessions?: number | null;
  maxStaleSessions?: number;
  active?: boolean;
  enabled?: boolean;
  enforceShortPuts?: boolean;
  rows?: number;
  fetchedAt?: string | null;
  computedAt?: string;
  gexNegative?: boolean | null;
  reason?: string;
  label?: string;
}

export interface DealerGammaDecision {
  ts?: string;
  day?: string;
  ticker?: string;
  source?: string;
  setup?: string;
  strategy?: string;
  structure?: string;
  action?: string;
  enforced?: boolean;
  rank252?: number | null;
  threshold?: number | null;
  asOf?: string | null;
  score?: number | null;
  price?: number | null;
  reason?: string;
  new?: boolean;
}

export interface AuditRow { type: string; message: string }

/** Python's storage_config.DATA_DIR as the Node process sees it (same rule as regimeReport.regimeDataDir). */
export function dealerGammaDataDir(env: NodeJS.ProcessEnv = process.env): string {
  return env.DATA_DIR || (fs.existsSync("/data") ? "/data/voltrade" : "/tmp");
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return !!v && typeof v === "object" && !Array.isArray(v);
}

/** One audit row per NEW skip / would-skip in a scan's `dealer_gamma` field, plus
 *  a GAMMA-ERROR row when the field reports its own failure (never silent). */
export function dealerGammaAuditRows(field: unknown): AuditRow[] {
  if (!isRecord(field)) return [];
  const rows: AuditRow[] = [];
  if (typeof field.error === "string" && field.error) {
    rows.push({ type: "GAMMA-ERROR", message: `dealer gamma filter field failed this scan: ${field.error.slice(0, 160)} (filter fails open)` });
  }
  const decisions = Array.isArray(field.decisions) ? field.decisions : [];
  for (const d of decisions) {
    if (rows.length >= DEALER_GAMMA_AUDIT_ROWS_MAX) break;
    if (!isRecord(d) || d.new === false || typeof d.ticker !== "string" || !d.ticker) continue;
    if (d.action !== "skip" && d.action !== "would_skip") continue;
    const what = String(d.setup || d.strategy || "short premium");
    const where = d.source ? ` via ${String(d.source)}` : "";
    rows.push({
      type: d.action === "skip" ? "GAMMA-SKIP" : "GAMMA-WOULD-SKIP",
      message: `${d.ticker} ${what}${where}: ${String(d.reason || "dealer gamma low").slice(0, 260)}`,
    });
  }
  return rows;
}

export function readDealerGammaState(dir: string): DealerGammaState | null {
  try {
    const parsed: unknown = JSON.parse(fs.readFileSync(path.join(dir, DEALER_GAMMA_STATE_FILE), "utf8"));
    return isRecord(parsed) ? (parsed as DealerGammaState) : null;
  } catch {
    return null; // absent until the first scan reads the filter
  }
}

/** The last `limit` decisions (oldest first), read from the file's tail only. */
export function readDealerGammaDecisions(dir: string, limit = DEALER_GAMMA_PROBE_DECISIONS): DealerGammaDecision[] {
  let text = "";
  try {
    const fp = path.join(dir, DEALER_GAMMA_DECISIONS_FILE);
    const size = fs.statSync(fp).size;
    const start = Math.max(0, size - DEALER_GAMMA_TAIL_BYTES);
    const fd = fs.openSync(fp, "r");
    try {
      const buf = Buffer.alloc(size - start);
      fs.readSync(fd, buf, 0, buf.length, start);
      text = buf.toString("utf8");
    } finally {
      fs.closeSync(fd);
    }
    if (start > 0) text = text.slice(text.indexOf("\n") + 1); // drop the partial first line
  } catch {
    return []; // absent until the filter logs its first decision
  }
  const out: DealerGammaDecision[] = [];
  for (const line of text.split("\n")) {
    if (!line.trim()) continue;
    try {
      const parsed: unknown = JSON.parse(line);
      if (isRecord(parsed)) out.push(parsed as DealerGammaDecision);
    } catch {
      continue; // a torn line from a concurrent append is skipped, never guessed
    }
  }
  return out.slice(-limit);
}

export interface DealerGammaProbe {
  probe: "gamma";
  label: string;
  state: DealerGammaState | null;
  state_age_minutes: number | null;
  decisions_recent: DealerGammaDecision[];
  counts_by_day: Record<string, { skip: number; would_skip: number }>;
  archive: Pick<SqueezeReport, "rows" | "latest_date" | "latest_rank" | "latest_age_days" | "last_refresh_ok"> | null;
  note: string;
}

export function dealerGammaProbeBody(
  state: DealerGammaState | null, decisions: DealerGammaDecision[], archive: SqueezeReport | null, nowMs = Date.now(),
): DealerGammaProbe {
  const counts: Record<string, { skip: number; would_skip: number }> = {};
  for (const d of decisions) {
    if (!d.day || (d.action !== "skip" && d.action !== "would_skip")) continue;
    const c = counts[d.day] || (counts[d.day] = { skip: 0, would_skip: 0 });
    c[d.action] += 1;
  }
  const at = state?.computedAt ? Date.parse(state.computedAt) : NaN;
  return {
    probe: "gamma",
    label: state?.label || "dealer gamma: no state yet (written by the first scan that reads the filter)",
    state,
    state_age_minutes: Number.isFinite(at) ? Math.round((nowMs - at) / 60_000) : null,
    decisions_recent: decisions,
    counts_by_day: counts,
    archive: archive ? {
      rows: archive.rows, latest_date: archive.latest_date, latest_rank: archive.latest_rank,
      latest_age_days: archive.latest_age_days, last_refresh_ok: archive.last_refresh_ok,
    } : null,
    note: "Edge P1 short-premium filter (dealer_gamma.py). Source: SqueezeMetrics' published GEX estimate "
      + "(not ours), rank = trailing-252 percentile. state = what the trading path last read; archive = the "
      + "server's squeeze_gex cache. Read-only; no raw GEX value is served.",
  };
}
