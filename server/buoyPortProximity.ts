/**
 * buoyPortProximity.ts — geographic join: NDBC buoy stations -> nearest of
 * the 9 imagery-verified port terminals (datacore/sites/strategic_sites.json,
 * the same REFERENCE DATA ACCURACY set portDwell.ts uses for vessel-arrival
 * fences), reusing those coordinates rather than introducing a second,
 * independently-unverified list of shipping-lane/chokepoint coordinates.
 *
 * ANGLE-HUNTING (research/open_questions.md, MARINE/BUOY HAZARD-ADJACENT
 * HYPOTHESES: "sea-state as a shipping-cost/insurance-exposure proxy",
 * FOREIGN-FIELD IMPORT from marine forecasting): the hypothesis's own text
 * names its gate-2 join target as "major shipping lanes/chokepoints (Gulf
 * of Mexico, mid-Atlantic, West Coast approaches) — cross-reference against
 * the existing AIS port-dwell archive's traffic density." That geographic
 * spread IS the 9 ports already in portDwell.ts (Houston = Gulf; NY/NJ,
 * Norfolk, Charleston, Savannah = mid/south Atlantic; LA, Long Beach,
 * Oakland, Seattle = West Coast) — so this join reuses that verified set
 * instead of hardcoding a fresh, unverified chokepoint list.
 *
 * Honesty: a port with no NDBC station inside the search radius returns
 * null, never a forced nearest-anyway match — some of the 9 ports may
 * simply have no buoy close enough, and that gap must stay visible rather
 * than be papered over with a distant "nearest" pick that isn't really
 * representative of that port's approach seas.
 *
 * SCOPE — gate 1 (mechanical) only: does a real, physically-plausible
 * buoy exist near each port's approach at all. This module makes no
 * predictive claim and does not read wave-height magnitude. Gate 2 (does
 * WVHT there predict forward port-dwell/sector returns) is unattempted —
 * both archives (buoys since 2026-07-08, port-dwell since 2026-07-03) are
 * still too thin for the N/5/20-day-horizon test the hypothesis's own
 * ladder entry specifies; see research/open_questions.md for the
 * earliest-attempt date this session logged.
 *
 * Pure module: no fs/network, no imports from trading logic. Callers pass
 * in ports + buoys (e.g. portsFromSites(...) and a parsed/archived NDBC
 * pull).
 */
import type { PortDef } from "./portDwell";
import type { BuoyObs } from "./ndbcBuoys";

const kmBetween = (aLat: number, aLon: number, bLat: number, bLon: number) => {
  const R = 6371, dLat = ((bLat - aLat) * Math.PI) / 180, dLon = ((bLon - aLon) * Math.PI) / 180;
  const s = Math.sin(dLat / 2) ** 2 +
    Math.cos((aLat * Math.PI) / 180) * Math.cos((bLat * Math.PI) / 180) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(s));
};

/** Buoys typically sit tens to a couple hundred km off the coast they
 *  serve — far outside the ~5km port-arrival fence portDwell.ts uses for
 *  vessel-in-port detection. This is a coarser "serves the same approach"
 *  radius, not a port-visit fence. 300km brackets the outer edge of NDBC's
 *  US coastal station spacing without reaching across to an unrelated
 *  port's nearest buoy. */
export const MAX_JOIN_KM = 300;

export interface PortBuoyMatch {
  portId: string;
  portName: string;
  station: string;
  distance_km: number;
}

/** LIVE-PROBED HONESTY FINDING (2026-08-09): geographic proximity alone is
 *  NOT enough for this hypothesis. A live run found that the single
 *  nearest station to every one of the 9 ports is a harbor/C-MAN station
 *  reporting `waveHeight: null` for all 9 — coastal tide/wind gauges, not
 *  open-ocean wave buoys. The genuinely wave-reporting station nearest each
 *  port sits 6-232km out instead. Callers that care about sea state (the
 *  actual hypothesis) MUST filter on capability, not just distance — the
 *  `requireWaveHeight` helper below does that; the raw nearest-any-station
 *  functions above stay generic for other future join uses (e.g. pressure
 *  tendency, which several of the null-wave harbor stations DO report). */
export const hasWaveHeight = (b: BuoyObs): boolean => b.waveHeight != null;

/** Every buoy within maxKm of a port matching `filter` (default: any
 *  reporting station), nearest first. A port can have several candidate
 *  stations — gate-2 work will want the option to use more than the single
 *  nearest (e.g. average WVHT across a cluster), so the full sorted list is
 *  exposed, not just the top pick. Buoys with a null lat/lon (malformed
 *  feed rows) are always skipped, never treated as distance 0. */
export function buoysNearPort(port: PortDef, buoys: BuoyObs[], maxKm = MAX_JOIN_KM,
                              filter: (b: BuoyObs) => boolean = () => true): PortBuoyMatch[] {
  const out: PortBuoyMatch[] = [];
  for (const b of buoys) {
    if (b.lat == null || b.lon == null || !filter(b)) continue;
    const d = kmBetween(port.lat, port.lon, b.lat, b.lon);
    if (d <= maxKm) out.push({ portId: port.id, portName: port.name, station: b.station, distance_km: Math.round(d * 10) / 10 });
  }
  return out.sort((a, b) => a.distance_km - b.distance_km);
}

/** Nearest buoy matching `filter` within maxKm of each port, or null if
 *  none qualifies — the gate-1 pass/fail unit. Pass `hasWaveHeight` as
 *  `filter` for the sea-state hypothesis; the default (any station)
 *  answers the weaker "does NDBC cover this approach at all" question. */
export function nearestBuoyPerPort(ports: PortDef[], buoys: BuoyObs[], maxKm = MAX_JOIN_KM,
                                   filter: (b: BuoyObs) => boolean = () => true): Map<string, PortBuoyMatch | null> {
  const out = new Map<string, PortBuoyMatch | null>();
  for (const p of ports) {
    const near = buoysNearPort(p, buoys, maxKm, filter);
    out.set(p.id, near.length ? near[0] : null);
  }
  return out;
}

export interface ProximityReport {
  generated_at: string;
  max_join_km: number;
  ports_matched: number;
  ports_total: number;
  matches: PortBuoyMatch[];
  unmatched_ports: Array<{ portId: string; portName: string }>;
}

/** Top-level gate-1 report: for every port, its single nearest buoy or an
 *  honest gap. A genuine exact-distance tie keeps the first-seen station
 *  (deterministic given a fixed buoy array order, but not meaningful) —
 *  no gate-2 code should depend on tie-breaking behavior. Pass
 *  `hasWaveHeight` as `filter` to answer the sea-state-specific question
 *  instead of the weaker "any station nearby" one (see the 2026-08-09
 *  honesty finding above the filter helpers). */
export function buoyPortProximityReport(ports: PortDef[], buoys: BuoyObs[], maxKm = MAX_JOIN_KM, nowMs?: number,
                                        filter: (b: BuoyObs) => boolean = () => true): ProximityReport {
  const nearest = nearestBuoyPerPort(ports, buoys, maxKm, filter);
  const matches: PortBuoyMatch[] = [];
  const unmatched: Array<{ portId: string; portName: string }> = [];
  for (const p of ports) {
    const m = nearest.get(p.id);
    if (m) matches.push(m); else unmatched.push({ portId: p.id, portName: p.name });
  }
  return {
    generated_at: new Date(nowMs ?? Date.now()).toISOString(),
    max_join_km: maxKm,
    ports_matched: matches.length,
    ports_total: ports.length,
    matches,
    unmatched_ports: unmatched,
  };
}
