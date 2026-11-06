/**
 * flowsReport.ts — read-only shaping of the institutional flow model's daily
 * ESTIMATE (2026-11-06; the model is flows_model.py + calendar_effects.py).
 *
 * flows_model.refresh_daily writes DATA_DIR/voltrade_flows.json once per
 * completed session (bot.ts Tier 3 runs it: FLOWS_REFRESH_CMD below). This
 * module reads that file and shapes the body of the owner's GET /api/bot/flows
 * and the token-gated diag probe "flows":
 *   {asOf, horizonDays,
 *    equity: {pressure, direction, confidence, drivers: [{name, value, reason}]},
 *    sectors: [{sector, etf, pressure, direction, drivers: [...]}],
 *    calendar: [{date, event, effect}], notes: string[]}
 * plus label "ESTIMATE", computedAt, ageDays, stale, the floor multiplier the
 * floor applies, and the per-session path. Nothing here computes a flow,
 * trades, or gates serving. Pure except the file reader (node:test safe).
 */
import fs from "fs";
import path from "path";

export const FLOWS_FILE = "voltrade_flows.json";
export const FLOWS_HORIZON_DEFAULT = 5;          // == flows_model DEFAULTS FLOWS_HORIZON_SESSIONS
export const FLOWS_STALE_DAYS = 4;              // == flows_model DEFAULTS FLOWS_MAX_AGE_DAYS
export const FLOWS_REFRESH_TIMEOUT_MS = 60_000; // fetches ~14 symbols of daily bars once a day
export const FLOWS_REFRESH_CMD =
  `python3 -c "import json; from flows_model import refresh_daily; print(json.dumps(refresh_daily(), default=str))"`;

export type FlowDirection = "buy" | "sell" | "neutral";
export interface FlowDriver { name: string; value: number; reason: string; key?: string; weight?: number }
export interface FlowSector { sector: string; etf: string; pressure: number; direction: FlowDirection; drivers: FlowDriver[]; confidence?: number; ytd?: number | null }
export interface FlowCalendarRow { date: string; event: string; effect: string; end?: string; kind?: string; source?: string }
export interface FlowsBody {
  asOf: string | null;
  horizonDays: number;
  equity: { pressure: number; direction: FlowDirection; confidence: number; drivers: FlowDriver[] };
  sectors: FlowSector[];
  calendar: FlowCalendarRow[];
  notes: string[];
  label: "ESTIMATE";
  computedAt: string | null;
  ageDays: number | null;
  stale: boolean;
  floor: { multiplier: number; applied: boolean; reason: string } | null;
  sessions: Array<{ date: string; pressure: number; phase: string | null }>;
}

type Obj = Record<string, unknown>;
const isObj = (v: unknown): v is Obj => !!v && typeof v === "object" && !Array.isArray(v);
const arr = (v: unknown): unknown[] => (Array.isArray(v) ? v : []);
const str = (v: unknown): string => (typeof v === "string" ? v : v === null || v === undefined ? "" : String(v));
const fin = (v: unknown, dflt = 0): number => (typeof v === "number" && Number.isFinite(v) ? v : dflt);
const unit = (v: unknown): number => Math.max(-1, Math.min(1, fin(v)));
const dir = (v: unknown): FlowDirection => (v === "buy" || v === "sell" ? v : "neutral");

export function readFlows(dir: string): Obj | null {
  try {
    const parsed: unknown = JSON.parse(fs.readFileSync(path.join(dir, FLOWS_FILE), "utf8"));
    return isObj(parsed) ? parsed : null;
  } catch {
    return null;   // absent or mid-write: the body says "no estimate yet"
  }
}

function driver(v: unknown): FlowDriver {
  const d = isObj(v) ? v : {};
  const out: FlowDriver = { name: str(d.name), value: Math.round(unit(d.value) * 1000) / 1000, reason: str(d.reason) };
  if (typeof d.key === "string") out.key = d.key;
  if (typeof d.weight === "number" && Number.isFinite(d.weight)) out.weight = d.weight;
  return out;
}

/** Whole days from the estimate's as-of session to `now` (UTC dates). */
export function flowsAgeDays(asOf: string | null, now: Date): number | null {
  if (!asOf || !/^\d{4}-\d{2}-\d{2}$/.test(asOf)) return null;
  const a = Date.parse(`${asOf}T00:00:00Z`);
  const b = Date.parse(`${now.toISOString().slice(0, 10)}T00:00:00Z`);
  return Number.isFinite(a) ? Math.round((b - a) / 86_400_000) : null;
}

export function flowsBody(doc: Obj | null, now: Date = new Date()): FlowsBody {
  if (!doc || doc.ok !== true) {
    const why = doc && typeof doc.error === "string" ? doc.error : "the daily estimate has not been computed yet";
    return {
      asOf: null, horizonDays: FLOWS_HORIZON_DEFAULT,
      equity: { pressure: 0, direction: "neutral", confidence: 0, drivers: [] },
      sectors: [], calendar: [], label: "ESTIMATE", computedAt: null, ageDays: null, stale: true, floor: null, sessions: [],
      notes: [`No flows estimate: ${why}. Tier 3 computes it once per completed session.`],
    };
  }
  const eq = isObj(doc.equity) ? doc.equity : {};
  const asOf = typeof doc.asOf === "string" ? doc.asOf : null;
  const ageDays = flowsAgeDays(asOf, now);
  const fl = isObj(doc.floor) ? doc.floor : null;
  const stale = ageDays === null || ageDays > FLOWS_STALE_DAYS;
  const notes = arr(doc.notes).map(str).filter((s) => s.length > 0);
  if (stale) notes.unshift(`Stale: this estimate is ${ageDays ?? "?"} days old; the floor and the sizer ignore it after ${FLOWS_STALE_DAYS}.`);
  return {
    asOf,
    horizonDays: Math.round(fin(doc.horizonDays, FLOWS_HORIZON_DEFAULT)),
    equity: {
      pressure: unit(eq.pressure), direction: dir(eq.direction),
      confidence: Math.max(0, Math.min(1, fin(eq.confidence))), drivers: arr(eq.drivers).map(driver),
    },
    sectors: arr(doc.sectors).filter(isObj).map((s) => ({
      sector: str(s.sector), etf: str(s.etf), pressure: unit(s.pressure), direction: dir(s.direction),
      drivers: arr(s.drivers).map(driver), confidence: Math.max(0, Math.min(1, fin(s.confidence))),
      ytd: typeof s.ytd === "number" && Number.isFinite(s.ytd) ? s.ytd : null,
    })),
    calendar: arr(doc.calendar).filter(isObj).map((c) => {
      const row: FlowCalendarRow = { date: str(c.date), event: str(c.event), effect: str(c.effect) };
      if (typeof c.end === "string") row.end = c.end;
      if (typeof c.kind === "string") row.kind = c.kind;
      if (typeof c.source === "string") row.source = c.source;
      return row;
    }),
    notes,
    label: "ESTIMATE",
    computedAt: typeof doc.computedAt === "string" ? doc.computedAt : null,
    ageDays,
    stale,
    floor: fl ? { multiplier: Math.min(1, Math.max(0, fin(fl.multiplier, 1))), applied: fl.applied === true, reason: str(fl.reason) } : null,
    sessions: arr(eq.sessions).filter(isObj).map((s) => ({
      date: str(s.date), pressure: unit(s.pressure), phase: typeof s.phase === "string" ? s.phase : null,
    })),
  };
}

/**
 * The audit row for one Tier 3 refresh. A new estimate ("computed") is one
 * FLOWS row in plain English (the body read back from the file it just wrote);
 * a failure is one FLOWS-ERROR row; a cached pass writes nothing.
 */
export function flowsAuditMessage(result: unknown, body: FlowsBody | null): { type: "FLOWS" | "FLOWS-ERROR"; message: string } | null {
  const r = isObj(result) ? result : {};
  const status = str(r.status);
  if (status === "error") return { type: "FLOWS-ERROR", message: `flows estimate failed: ${str(r.reason).slice(0, 200) || "unknown error"}` };
  if (status !== "computed" || !body || !body.asOf) return null;
  const e = body.equity;
  const top = [...e.drivers].filter((d) => d.value !== 0).sort((a, b) => Math.abs(b.value * (b.weight ?? 1)) - Math.abs(a.value * (a.weight ?? 1)))[0];
  const sold = body.sectors.filter((s) => s.direction === "sell").slice(0, 3).map((s) => `${s.etf} ${s.pressure.toFixed(2)}`);
  const bought = body.sectors.filter((s) => s.direction === "buy").slice(-3).reverse().map((s) => `${s.etf} +${s.pressure.toFixed(2)}`);
  const floor = body.floor ? `; floor x${body.floor.multiplier.toFixed(2)}` : "";
  const parts = [
    `${body.asOf} ESTIMATE: equities ${e.pressure >= 0 ? "+" : ""}${e.pressure.toFixed(2)} (${e.direction}, confidence ${e.confidence.toFixed(2)})${floor}`,
    top ? `main driver: ${top.reason}` : "no driver active",
    sold.length ? `selling: ${sold.join(", ")}` : "",
    bought.length ? `buying: ${bought.join(", ")}` : "",
  ].filter((p) => p.length > 0);
  return { type: "FLOWS", message: parts.join(" | ").slice(0, 900) };
}
