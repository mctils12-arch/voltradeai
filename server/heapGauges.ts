/**
 * heapGauges.ts — the gauges that live outside their module (memory audit
 * 2026-11-03, A0). Modules whose structure is reachable through an existing
 * export are gauged here so the diagnostic never has to edit them; modules
 * with private state register their own gauge next to the declaration
 * (routes.ts caches, aircraftTrips bloom, airportsIndex, the dedup sets).
 */
import { registerHeapGauge } from "./heapDiag";
import { latestDtcc } from "./dtccSwaps";
import { cachedGraphSync } from "./entityGraph";
import { analyticsSnapshot } from "./requestLog";

/** Rough per-ID retained cost of a 10-20 char string in a Set (header +
 *  one-byte chars + hash-table slot). Audit anchor: 6.56M IDs ≈ 0.4-0.5 GB. */
export const DTCC_SEEN_ID_BYTES_EST = 72;

export function registerBuiltinHeapGauges(): void {
  // DTCC swaps: the in-memory dedup Set is the single largest retained
  // structure (A1 moves it to disk; this gauge is how A1's win is measured).
  registerHeapGauge("dtcc.seenIds", () => {
    const d = latestDtcc();
    const entries = d ? d.totalArchived : 0;
    return { entries, bytes: entries * DTCC_SEEN_ID_BYTES_EST, estimate: "entries*72B" };
  });
  registerHeapGauge("entityGraph.graphCache", () => {
    const g = cachedGraphSync();
    if (!g) return { entries: 0 };
    return { entries: g.counts.nodes + g.counts.edges, nodes: g.counts.nodes, edges: g.counts.edges, age_s: Math.round((Date.now() - g.built_at) / 1000) };
  });
  registerHeapGauge("requestLog.ring", () => {
    const a = analyticsSnapshot();
    return { entries: a.ringSize, cap: a.ringCap };
  });
}
