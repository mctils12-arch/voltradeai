/**
 * gridDemandRevision.ts — ROOT VALIDATION LADDER gate 1 for EIA-930 grid
 * demand (`server/gridDemand.ts`, live since v1.0.163). Pre-stated gate-1
 * criterion (module header, `open_questions.md`/`experiments.md`
 * 2026-07-06): "US48 daily sum vs EIA's own Grid Monitor dashboard." EIA's
 * Grid Monitor dashboard is itself rendered from this exact region-data API
 * (no separate third-party ground truth exists for hourly grid demand), so
 * the only honestly-available external check is an INDEPENDENT re-draw of
 * the same API at a later point in time — this either confirms our archive
 * captures the same values EIA continues to publish, or surfaces a
 * revision/instability large enough to matter.
 *
 * A single anecdotal data point already existed (experiments.md, this
 * file's own commit history 2026-07-31 session start): one US48 hour
 * pulled twice ~2 minutes apart showed a 1.1% revision, then held stable on
 * a third pull 4 minutes later. `computeRevisionStats` below turns that
 * one-off manual check into a reusable, systematic instrument covering
 * every (respondent, period, type) cell in a fetch window at once — see
 * `scripts/gate1_eia930_revision_probe.ts` for the runner.
 *
 * IMPORTANT SCOPE NOTE: this measures EIA's OWN revision behavior between
 * two of ITS API responses, not our archiver's parsing/aggregation
 * correctness (that risk is separately near-zero — `parseDemand` just
 * extracts the `value` field, no summing/derivation of our own). It also
 * does not change `gridDemand.ts`'s archive/dedup behavior in any way —
 * this is a read-only, side-effect-free measurement.
 */
import type { DemandObs } from "./gridDemand";

/** Below this magnitude, a value change is normal report-settle noise, not
 *  a data-quality concern — informed by the single 1.1% anecdote above;
 *  set with margin (2x) since one sample cannot pin an exact bound. Only
 *  used to label rows in the report, never to reject/quarantine anything —
 *  this stream carries no predictive claim yet, nothing to gate. */
export const MATERIAL_REVISION_PCT = 2;

export interface RevisedCell {
  respondent: string;
  period: string;
  type: "D" | "DF";
  v1: number;
  v2: number;
  diff_pct: number;
  hours_old_at_draw1: number | null;
}

export interface RevisionReport {
  draw1_at: number;
  draw2_at: number;
  gap_minutes: number;
  compared: number;          // cells present (non-null value) in both draws
  only_in_draw1: number;
  only_in_draw2: number;
  revised: number;           // compared cells whose value changed at all
  revised_pct: number;       // of `compared`
  material_revised: number;  // revised AND |diff_pct| >= MATERIAL_REVISION_PCT
  max_abs_diff_pct: number;
  worst: RevisedCell[];      // top N by |diff_pct|, for hand verification
}

const cellKey = (o: DemandObs) => `${o.respondent}|${o.period}|${o.type || "D"}`;

/** Hours between a period's own timestamp and the draw-1 fetch time — the
 *  T13 anecdote and EIA-930's stated 1-2h publication lag both point to
 *  revision risk concentrating in the newest/leading-edge hours, so the
 *  report buckets by this instead of assuming it's uniform across the
 *  window. */
function hoursOld(period: string, drawAtMs: number): number | null {
  const periodMs = Date.parse(`${period}:00:00Z`);
  if (!Number.isFinite(periodMs)) return null;
  return Math.round((drawAtMs - periodMs) / 3_600_000);
}

export function computeRevisionStats(
  draw1: DemandObs[], draw1At: number,
  draw2: DemandObs[], draw2At: number,
  worstN = 20,
): RevisionReport {
  const m1 = new Map<string, DemandObs>();
  for (const o of draw1) if (o.mwh != null) m1.set(cellKey(o), o);
  const m2 = new Map<string, DemandObs>();
  for (const o of draw2) if (o.mwh != null) m2.set(cellKey(o), o);

  let onlyIn1 = 0, onlyIn2 = 0, compared = 0, revised = 0, material = 0, maxAbsDiff = 0;
  const revisedCells: RevisedCell[] = [];
  m1.forEach((o1, key) => {
    const o2 = m2.get(key);
    if (!o2) { onlyIn1++; return; }
    compared++;
    const v1 = o1.mwh as number, v2 = o2.mwh as number;
    if (v1 === v2) return;
    revised++;
    const diffPct = v1 === 0 ? (v2 === 0 ? 0 : Infinity) : ((v2 - v1) / v1) * 100;
    maxAbsDiff = Math.max(maxAbsDiff, Math.abs(diffPct));
    if (Math.abs(diffPct) >= MATERIAL_REVISION_PCT) material++;
    revisedCells.push({
      respondent: o1.respondent, period: o1.period, type: (o1.type || "D") as "D" | "DF",
      v1, v2, diff_pct: Math.round(diffPct * 100) / 100,
      hours_old_at_draw1: hoursOld(o1.period, draw1At),
    });
  });
  m2.forEach((_o2, key) => { if (!m1.has(key)) onlyIn2++; });

  revisedCells.sort((a, b) => Math.abs(b.diff_pct) - Math.abs(a.diff_pct));

  return {
    draw1_at: draw1At, draw2_at: draw2At,
    gap_minutes: Math.round((draw2At - draw1At) / 60_000),
    compared, only_in_draw1: onlyIn1, only_in_draw2: onlyIn2,
    revised, revised_pct: compared ? Math.round((revised / compared) * 10000) / 100 : 0,
    material_revised: material,
    max_abs_diff_pct: Math.round(maxAbsDiff * 100) / 100,
    worst: revisedCells.slice(0, worstN),
  };
}
