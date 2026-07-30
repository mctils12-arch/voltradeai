/**
 * signalLadder.ts — MAP V2 ROADMAP R6(a) SIGNAL-STRENGTH dashboard
 * (research/open_questions.md: "ladder position of every root, gate
 * passed/date/next gate, from research/ bookkeeping made machine-readable").
 *
 * Pure read over the hand-compiled datacore/signal_ladder.json registry
 * (same provenance discipline as entity_map.json: every entry carries a
 * source_ref a human can re-check). This module does NOT compute gate
 * status itself — that requires reading research/ prose and judgment,
 * which stays a human/session compilation task (see the registry's own
 * _doc) — it only aggregates the already-compiled snapshot for display.
 * Zero network calls, zero pollers.
 */
import signalLadderJson from "../datacore/signal_ladder.json";

export type LadderStatus =
  | "raw_only"
  | "gate1_pending"
  | "gate1_pass"
  | "gate1_fail"
  | "gate2_pending"
  | "gate2_pass"
  | "gate2_fail"
  | "gate3_pass"
  | "gate4_pass"
  | "gate5_pass"
  | "killed";

export interface LadderRoot {
  id: string;
  name: string;
  category: string;
  status: LadderStatus;
  current_gate: number;
  last_update_date: string;
  note: string;
  source_ref: string;
}

export interface LadderSummary {
  total: number;
  by_status: Record<string, number>;
  by_category: Record<string, number>;
  gate_counts: { gate: number; count: number }[]; // current_gate 0-5, index = gate
  killed_count: number;
  raw_only_count: number;
  furthest_gate_reached: number;
}

export function summarizeLadder(roots: readonly LadderRoot[]): LadderSummary {
  const by_status: Record<string, number> = {};
  const by_category: Record<string, number> = {};
  const gateCounts = [0, 0, 0, 0, 0, 0]; // gates 0-5
  let killed_count = 0;
  let raw_only_count = 0;
  let furthest_gate_reached = 0;

  for (const r of roots) {
    by_status[r.status] = (by_status[r.status] || 0) + 1;
    by_category[r.category] = (by_category[r.category] || 0) + 1;
    const g = Math.min(5, Math.max(0, r.current_gate));
    gateCounts[g]++;
    if (r.status === "killed") killed_count++;
    if (r.status === "raw_only") raw_only_count++;
    // "furthest gate reached" only counts genuine PASSES, never a killed
    // or failed root's current_gate (that number marks where it died,
    // not how far it got trusted) and never raw_only (gate doesn't apply).
    if (r.status.endsWith("_pass") && g > furthest_gate_reached) {
      furthest_gate_reached = g;
    }
  }

  return {
    total: roots.length,
    by_status,
    by_category,
    gate_counts: gateCounts.map((count, gate) => ({ gate, count })),
    killed_count,
    raw_only_count,
    furthest_gate_reached,
  };
}

export function loadSignalLadder(): {
  compiled: string;
  sources: string[];
  roots: LadderRoot[];
  summary: LadderSummary;
} {
  const data = signalLadderJson as {
    compiled: string;
    sources: string[];
    roots: LadderRoot[];
  };
  return {
    compiled: data.compiled,
    sources: data.sources,
    roots: data.roots,
    summary: summarizeLadder(data.roots),
  };
}
