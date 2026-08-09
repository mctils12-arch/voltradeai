// nrcReactorStatus.ts — pure helpers for the NRC daily reactor-status
// dedicated /data page (client/src/pages/nrcReactorStatus.tsx). Split out
// of the page component (rather than inlined, like mapIcons.ts/deviceTier.ts)
// so the sort/rank logic gets a real behavioral unit test instead of the
// source-inspection style datamap.tsx's colocated tests use for JSX-heavy
// wiring — this file has zero React/JSX, so a direct import is the more
// honest test.
//
// Data source: server/nrcReactorStatus.ts's `joinToPlants` output, served
// live at /api/data/nrc-reactor-status (`plants` field). GATE 1 (DATA)
// passed 2026-08-04; the outage-adjacent SIGNAL hypothesis stays
// gate-2-locked (research/open_questions.md POWER-PLANT SIGNAL
// HYPOTHESES) — this view is RAW display only, no predictive claim.

export type PlantPowerStatus = "full" | "reduced" | "outage" | "unknown";

export interface ReactorUnitReading {
  unit: string;
  power: number | null;
}

export interface PlantReactorStatus {
  idx: number;
  name: string;
  lat: number;
  lon: number;
  mw: number;
  owner: string;
  units: ReactorUnitReading[];
  avgPower: number | null;
  status: PlantPowerStatus | string;
}

export const STATUS_LABEL: Record<PlantPowerStatus, string> = {
  outage: "Outage",
  reduced: "Reduced",
  unknown: "No reading",
  full: "Full power",
};

// Worst-first: an operator/analyst opening this page wants to see what's
// DOWN before what's fine. Matches the map layer's own color severity
// (nrcReactorStatusColor: red outage / amber reduced / gray unknown /
// green full) so the two views never disagree on what "worse" means.
const STATUS_RANK: Record<string, number> = { outage: 0, reduced: 1, unknown: 2, full: 3 };

export function statusRank(status: string | null | undefined): number {
  const r = STATUS_RANK[status ?? ""];
  return r === undefined ? 4 : r;
}

/** Worst-status-first, then lowest reported power within the same status
 *  tier (a 40%-power "reduced" plant is more notable than an 80%-power
 *  one), then alphabetical for a stable, readable order. Never mutates
 *  the input array. */
export function sortPlantsByStatus<T extends { status: string; avgPower: number | null; name: string }>(
  plants: T[],
): T[] {
  return plants.slice().sort((a, b) => {
    const r = statusRank(a.status) - statusRank(b.status);
    if (r !== 0) return r;
    const ap = a.avgPower ?? Infinity;
    const bp = b.avgPower ?? Infinity;
    if (ap !== bp) return ap - bp;
    return a.name.localeCompare(b.name);
  });
}

export function statusCounts<T extends { status: string }>(plants: T[]): Record<PlantPowerStatus, number> {
  const out: Record<PlantPowerStatus, number> = { outage: 0, reduced: 0, unknown: 0, full: 0 };
  for (const p of plants) {
    const s = (p.status as PlantPowerStatus) in out ? (p.status as PlantPowerStatus) : "unknown";
    out[s]++;
  }
  return out;
}
