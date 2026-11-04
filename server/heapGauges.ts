/**
 * heapGauges.ts — the gauges that live outside their module (memory audit
 * 2026-11-03, A0). Modules whose structure is reachable through an existing
 * export are gauged here so the diagnostic never has to edit them; modules
 * with private state register their own gauge next to the declaration
 * (routes.ts caches, aircraftTrips bloom, airportsIndex, the dedup sets).
 */
import { registerHeapGauge } from "./heapDiag";
import { latestDtcc, dtccIdsStats } from "./dtccSwaps";
import { cachedGraphSync, graphPollStats } from "./entityGraph";
import { analyticsSnapshot } from "./requestLog";
import { shadowFleetFoldStats } from "./shadowFleet";
import { portDwellFoldStats } from "./portDwell";
import { gnssFoldStats } from "./gnssIntegrityFold";

/** Rough per-ID retained cost of a 10-20 char string in a Set (header +
 *  one-byte chars + hash-table slot). Audit anchor: 6.56M IDs ≈ 0.4-0.5 GB. */
export const DTCC_SEEN_ID_BYTES_EST = 72;

export function registerBuiltinHeapGauges(): void {
  // DTCC swaps: the in-memory dedup Set is the single largest retained
  // structure (A1 moves it to disk; this gauge is how A1's win is measured).
  registerHeapGauge("dtcc.seenIds", () => {
    // A1 (2026-11-03): the Set may no longer exist (sqlite mode) — report the
    // store's own view: mode, Set size (null = not built = the heap win),
    // DB count, dual-run mismatches, backfill state, last lookup latency.
    const st = dtccIdsStats();
    const d = latestDtcc();
    const entries = st.memoryCount ?? 0;
    return {
      entries,
      bytes: entries * DTCC_SEEN_ID_BYTES_EST,
      estimate: "memoryCount*72B",
      mode: st.mode,
      memoryCount: st.memoryCount,
      dbCount: st.dbCount,
      totalArchived: d ? d.totalArchived : null,
      mismatches: st.mismatches,
      lastLookupMs: st.lastLookupMs == null ? null : Math.round(st.lastLookupMs * 10) / 10,
      backfill: st.backfill ? st.backfill.state : null,
      dbBytes: st.dbBytes,
      diskGuard: st.diskGuard.active,
    };
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
  // A2 (2026-11-04): the old-vs-new fold comparisons. Each fold ships in
  // `shadow` mode (both folds run, the FULL result is served); the switch to
  // `incremental` is allowed only after 24 h of mismatches === 0, and this is
  // where that is read (FOLD-MISMATCH rows also land in the audit log).
  // entries = cached partials; the counters are copied as the module reports them.
  registerHeapGauge("fold.shadowfleet", () => {
    const f = shadowFleetFoldStats();
    return { entries: f.partialsCached, bytes: f.partialBytesEstimate, mode: f.mode, mismatches: f.mismatches, matches: f.matches, compareSkipped: f.compareSkipped, fallbacks: f.fallbacks, lastFullMs: f.lastFullMs, lastIncrementalMs: f.lastIncrementalMs, lastMismatchAt: f.lastMismatchAt, lastRunAt: f.lastRunAt };
  });
  registerHeapGauge("fold.portdwell", () => {
    const f = portDwellFoldStats();
    return { entries: f.partialsCached, bytes: f.partialBytesEstimate, mode: f.mode, mismatches: f.mismatches, matches: f.matches, compareSkipped: f.compareSkipped, fallbacks: f.fallbacks, lastFullMs: f.lastFullMs, lastIncrementalMs: f.lastIncrementalMs, lastMismatchAt: f.lastMismatchAt, lastRunAt: f.lastRunAt };
  });
  registerHeapGauge("fold.gnss", () => {
    const f = gnssFoldStats();
    return { entries: f.partialsCached, mode: f.mode, mismatches: f.mismatches, cycles: f.cycles, lastFullMs: f.lastFullMs, lastIncrementalMs: f.lastIncrementalMs, lastMismatchAt: f.lastMismatchAt, lastCycleAt: f.lastCycleAt, lastCyclePeakRows: f.lastCyclePeakRows };
  });
  registerHeapGauge("fold.graph", () => {
    const f = graphPollStats();
    return { entries: f.partialsCached, mode: f.mode, mismatches: f.mismatches, polls: f.polls, lastFullMs: f.lastFullMs, lastIncrementalMs: f.lastIncrementalMs, lastMismatchAt: f.lastMismatchAt, lastPollAt: f.lastPollAt };
  });
}
