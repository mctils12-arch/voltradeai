// portDwellView.ts — pure helpers for the port-dwell dedicated /data page
// (client/src/pages/portDwell.tsx). Same split as nrcReactorStatus.ts: the
// sort/filter logic gets a real behavioral unit test, the page component
// stays presentation-only.
//
// Data source: server/portDwell.ts's computePortDwellAsync output, served
// live at /api/data/portdwell (`ports` field). GATE 1 (DATA) passed
// 2026-09-02 for the port-presence reader (datacore/signal_ladder.json,
// port_dwell_maritime_transit) — the dwell-anomaly-vs-forward-returns
// SIGNAL hypothesis stays gate-2-locked (still accumulating weekly
// snapshots, datacore/port_dwell_weekly.json). This view is RAW display
// only: no predictive claim, per-vessel anomaly flags are the 3x-median
// rule already computed server-side (server/portDwell.ts's ANOMALY_FACTOR),
// not a new claim invented here.

export interface PortAnomalyExample {
  mmsi: string;
  name?: string;
  dwell_h: number;
  median_h: number;
}

export interface PortDwellPortRow {
  id: string;
  name: string;
  lat: number;
  lon: number;
  visits_completed: number;
  unique_vessels: number;
  in_port_now: number;
  dwell_median_h: number | null;
  dwell_p90_h: number | null;
  dwell_max_h: number | null;
  anomaly_count: number;
  anomaly_examples: PortAnomalyExample[];
}

export type PortDwellFilter = "all" | "active" | "anomaly";

// Busiest-first: in_port_now (what's happening right now) outranks
// visits_completed (historical activity within the window), then name for
// a stable, readable order. Never mutates the input array.
export function sortPortsByActivity<T extends { in_port_now: number; visits_completed: number; name: string }>(
  ports: T[],
): T[] {
  return ports.slice().sort((a, b) => {
    if (a.in_port_now !== b.in_port_now) return b.in_port_now - a.in_port_now;
    if (a.visits_completed !== b.visits_completed) return b.visits_completed - a.visits_completed;
    return a.name.localeCompare(b.name);
  });
}

export function filterPorts<T extends { in_port_now: number; anomaly_count: number }>(
  ports: T[],
  filter: PortDwellFilter,
): T[] {
  if (filter === "active") return ports.filter((p) => p.in_port_now > 0);
  if (filter === "anomaly") return ports.filter((p) => p.anomaly_count > 0);
  return ports;
}

export function totalAnomalies<T extends { anomaly_count: number }>(ports: T[]): number {
  return ports.reduce((sum, p) => sum + p.anomaly_count, 0);
}
