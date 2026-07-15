// Aircraft operator filter — EARTH TWIN O6-4 (human directive: "apply this
// logic to other real time moving objects so you could see all American
// flights at a moment").
//
// HONESTY: the filter key is the BROADCAST CALLSIGN PREFIX — airlines file
// flights under their ICAO telephony code (AAL123 = American Airlines 123),
// a checkable broadcast fact. This is labeled as a callsign filter, never
// as a verified operator registry: general aviation, ferry flights, and
// blocked callsigns simply don't match, and that's stated in the note.

export interface AirlinePreset {
  key: string;
  label: string;
  /** ICAO airline designator = callsign prefix. */
  prefix: string;
}

export const AIRLINE_PRESETS: AirlinePreset[] = [
  { key: 'aal', label: 'American', prefix: 'AAL' },
  { key: 'ual', label: 'United', prefix: 'UAL' },
  { key: 'dal', label: 'Delta', prefix: 'DAL' },
  { key: 'swa', label: 'Southwest', prefix: 'SWA' },
  { key: 'baw', label: 'British Airways', prefix: 'BAW' },
  { key: 'dlh', label: 'Lufthansa', prefix: 'DLH' },
  { key: 'uae', label: 'Emirates', prefix: 'UAE' },
  { key: 'fdx', label: 'FedEx', prefix: 'FDX' },
  { key: 'ups', label: 'UPS', prefix: 'UPS' },
];

export function filterAircraftByPrefix<T extends { callsign?: string | null }>(
  rows: T[],
  prefix: string | null,
): T[] {
  if (!prefix) return rows;
  const p = prefix.trim().toUpperCase();
  if (!p) return rows;
  return rows.filter((a) => String(a.callsign || '').toUpperCase().startsWith(p));
}

/** The wireLivePoints transformData for the aircraft payload: filters every
 *  renderer (2D symbols, vectors, 3D silhouettes) through one pass and
 *  reports what the filter did — never a silent drop. */
export function applyAirlineFilter(d: any, prefix: string | null): any {
  if (!prefix || !d || !Array.isArray(d.aircraft)) return d;
  const rows = filterAircraftByPrefix(d.aircraft, prefix);
  return {
    ...d,
    aircraft: rows,
    count: rows.length,
    filter_note: `callsign filter ${prefix.toUpperCase()}* — ${rows.length} of ${d.aircraft.length} aircraft shown (broadcast flight IDs; non-matching callsigns are simply not ${prefix.toUpperCase()} flights, not hidden data)`,
  };
}
