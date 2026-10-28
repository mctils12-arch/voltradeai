/**
 * regimeCounterfactual.ts — read-only aggregation of the regime engine's
 * legacy-vs-engine counterfactual log (2026-10-28 live flip, human directive).
 *
 * regime_engine.log_counterfactual (Python) appends one JSONL row per decision
 * to `<DATA_DIR>/voltrade_regime_counterfactual.jsonl` (rotated to `.1` above
 * its size cap): kind (selection | options | sizing | label), what LEGACY would
 * have done at that decision point, what the ENGINE did, `differs`, the
 * rejected_regime_* / would_reject_regime_* shadow bucket it links to, and the
 * regime state snapshot. This module reads those rows and returns AGGREGATES
 * only — per day per kind: counts, how many differ, mean legacy vs engine
 * multiplier per sizing sleeve, blocks by guard, label pairs. No ticker leaves
 * it (the diag probe "regime_counterfactual" serves this body).
 *
 * Pure except for the file reader (node:test safe).
 */
import fs from "fs";
import path from "path";

export const REGIME_COUNTERFACTUAL_FILE = "voltrade_regime_counterfactual.jsonl";
export const REGIME_COUNTERFACTUAL_DEFAULT_DAYS = 14;
export const REGIME_COUNTERFACTUAL_MAX_DAYS = 120;

export type CounterfactualRow = {
  date?: string;
  kind?: string;
  sleeve?: string | null;
  source?: string;
  legacy?: Record<string, unknown>;
  engine?: Record<string, unknown>;
  differs?: boolean | null;
  shadow_decision?: string | null;
  regime?: { as_of?: string; composite?: string; stale?: boolean } | null;
};

/** Rows dated on/after `sinceDay` (YYYY-MM-DD) from the current file and its one rotated generation. */
export function readRegimeCounterfactual(dir: string, sinceDay: string): CounterfactualRow[] {
  const out: CounterfactualRow[] = [];
  for (const name of [`${REGIME_COUNTERFACTUAL_FILE}.1`, REGIME_COUNTERFACTUAL_FILE]) {
    let raw: string;
    try {
      raw = fs.readFileSync(path.join(dir, name), "utf8");
    } catch {
      continue;   // absent file = no rows (the engine has not logged yet, or never rotated)
    }
    for (const line of raw.split("\n")) {
      if (!line.trim()) continue;
      const row = parseCounterfactualLine(line);
      if (row && typeof row.date === "string" && row.date >= sinceDay) out.push(row);
    }
  }
  return out;
}

/** One row, or null for a torn line (the writer appends one row at a time; a partial last line is possible mid-append). */
export function parseCounterfactualLine(line: string): CounterfactualRow | null {
  try {
    const row = JSON.parse(line);
    return row && typeof row === "object" ? (row as CounterfactualRow) : null;
  } catch {
    return null;
  }
}

function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function bump(m: Record<string, number>, k: string, by = 1): void {
  m[k] = (m[k] || 0) + by;
}

type SleeveAgg = { n: number; applied: number; stale: number; legacy_sum: number; engine_sum: number; legacy_n: number; engine_n: number };
type GateAgg = { n: number; differs: number; decisions: Record<string, number>; blocks_by_guard: Record<string, number>; would_block_by_guard: Record<string, number>; size_cut: number; dip_neutralized: number; linked_shadow: number };

function gateAgg(): GateAgg {
  return { n: 0, differs: 0, decisions: {}, blocks_by_guard: {}, would_block_by_guard: {}, size_cut: 0, dip_neutralized: 0, linked_shadow: 0 };
}

/** Per-day, per-kind aggregates (no tickers). */
export function summarizeRegimeCounterfactual(rows: CounterfactualRow[]) {
  const days: Record<string, {
    sizing: Record<string, SleeveAgg>;
    selection: GateAgg;
    options: GateAgg;
    label: { n: number; differs: number; applied: number; pairs: Record<string, number> };
    composites: Record<string, number>;
  }> = {};
  for (const r of rows) {
    const d = String(r.date);
    const day = (days[d] ??= {
      sizing: {}, selection: gateAgg(), options: gateAgg(),
      label: { n: 0, differs: 0, applied: 0, pairs: {} }, composites: {},
    });
    if (r.regime?.composite) bump(day.composites, r.regime.composite);
    const eng = (r.engine ?? {}) as Record<string, unknown>;
    const leg = (r.legacy ?? {}) as Record<string, unknown>;
    if (r.kind === "sizing") {
      const s = (day.sizing[String(r.sleeve ?? "unknown")] ??= { n: 0, applied: 0, stale: 0, legacy_sum: 0, engine_sum: 0, legacy_n: 0, engine_n: 0 });
      s.n += 1;
      if (eng.applied === true) s.applied += 1;
      if (eng.stale === true) s.stale += 1;
      const lm = num(leg.multiplier);
      const em = num(eng.multiplier);
      if (lm !== null) { s.legacy_sum += lm; s.legacy_n += 1; }
      if (em !== null) { s.engine_sum += em; s.engine_n += 1; }
    } else if (r.kind === "selection" || r.kind === "options") {
      const g = r.kind === "selection" ? day.selection : day.options;
      g.n += 1;
      if (r.differs) g.differs += 1;
      bump(g.decisions, String(eng.decision ?? "unknown"));
      for (const x of Array.isArray(eng.blocked_by) ? eng.blocked_by : []) bump(g.blocks_by_guard, String(x));
      for (const x of Array.isArray(eng.would_block) ? eng.would_block : []) bump(g.would_block_by_guard, String(x));
      const sm = num(eng.size_mult);
      if (sm !== null && sm < 1) g.size_cut += 1;
      if (eng.dip_neutralized === true) g.dip_neutralized += 1;
      if (r.shadow_decision) g.linked_shadow += 1;
    } else if (r.kind === "label") {
      day.label.n += 1;
      if (r.differs) day.label.differs += 1;
      if (eng.applied === true) day.label.applied += 1;
      bump(day.label.pairs, `${String(leg.label ?? "?")}>${String(eng.label ?? "?")}`);
    }
  }
  const round = (x: number) => Math.round(x * 10000) / 10000;
  return Object.keys(days).sort().map((date) => {
    const d = days[date];
    const sizing: Record<string, unknown> = {};
    for (const [sleeve, s] of Object.entries(d.sizing)) {
      sizing[sleeve] = {
        n: s.n, applied: s.applied, stale: s.stale,
        mean_legacy_multiplier: s.legacy_n ? round(s.legacy_sum / s.legacy_n) : null,
        mean_engine_multiplier: s.engine_n ? round(s.engine_sum / s.engine_n) : null,
      };
    }
    return { date, sizing, selection: d.selection, options: d.options, label: d.label, composites: d.composites };
  });
}

/** The probe body: the last `days` UTC days (bounded), aggregates only. */
export function regimeCounterfactualBody(dir: string, now: Date, daysParam?: unknown) {
  const n = Number(daysParam);
  const days = Number.isFinite(n) && n >= 1 ? Math.min(Math.floor(n), REGIME_COUNTERFACTUAL_MAX_DAYS) : REGIME_COUNTERFACTUAL_DEFAULT_DAYS;
  const since = new Date(now.getTime() - (days - 1) * 86400000).toISOString().slice(0, 10);
  const rows = readRegimeCounterfactual(dir, since);
  return {
    days,
    since,
    rows: rows.length,
    by_day: summarizeRegimeCounterfactual(rows),
    note: "legacy = what the pre-engine system would have done at the same decision point; engine = what the regime engine did (applied=false: its flag was off, the engine's would-be value). Shadow outcomes for blocked names: /api/diag/shadow buckets rejected_regime_* / would_reject_regime_*.",
  };
}
