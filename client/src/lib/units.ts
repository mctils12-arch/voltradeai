// units — ONE user preference (imperial | metric) that every user-facing
// measurement on the site renders through (human directive 2026-07-13:
// "have the ability to change these variables all over to imperial system
// or metric ... and always transfer this info or feature when adding new
// data sources").
//
// CONTRACT for new data sources/layers: store and compute in the source's
// native units (usually SI — never convert archived data), and render ANY
// distance/speed/temperature/length through these formatters. Direct
// `${x} km`-style display strings in new code are a review defect.
//
// Deliberately NOT unit-switched (domain conventions, same in both systems):
// knots for vessels/aircraft/wind arrows, hPa for barometric pressure (NDBC/
// NWS marine convention), MW for plant capacity, Kelvin for fire radiative
// brightness, µSv/h for dose rate, earthquake magnitude.
//
// Hermetic: no React, no DOM requirement (localStorage guarded), pure
// conversion math — unit-testable with `npx tsx --test`.

export type UnitSystem = "imperial" | "metric";

export const UNITS_PREF_KEY = "vt-units";

// Imperial default: US-centric product (US grid, NYSE, FEMA, NOAA surfaces)
// and consistent with the weather layer's existing °F-default toggle.
const DEFAULT_SYSTEM: UnitSystem = "imperial";

export function readUnitsPref(): UnitSystem {
  try {
    const v = globalThis.localStorage?.getItem(UNITS_PREF_KEY);
    if (v === "imperial" || v === "metric") return v;
  } catch { /* no storage (SSR/tests) */ }
  return DEFAULT_SYSTEM;
}

let current: UnitSystem = readUnitsPref();
const listeners = new Set<() => void>();

export function getUnits(): UnitSystem {
  return current;
}

export function setUnits(u: UnitSystem): void {
  if (u === current) return;
  current = u;
  try { globalThis.localStorage?.setItem(UNITS_PREF_KEY, u); } catch { /* best-effort */ }
  listeners.forEach((fn) => { try { fn(); } catch { /* listener owns its errors */ } });
}

/** Subscribe to preference changes (useSyncExternalStore-compatible). */
export function subscribeUnits(fn: () => void): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

// ── conversions (exact factors) ──────────────────────────────────────────────
const KM_PER_MI = 1.609344;
const M_PER_FT = 0.3048;

// ── formatters ───────────────────────────────────────────────────────────────
// All take the value in the source's native METRIC unit and return a display
// string in the CURRENT system (or an explicit `system` for testability).
// null/undefined/NaN → "no data" (matching the buoy card's honesty rule:
// a missing sensor is not a zero).

const bad = (v: unknown): v is null | undefined =>
  v == null || Number.isNaN(Number(v));

/** Distance in km → "85 km" | "53 mi". */
export function fmtKm(km: number | null | undefined, digits = 0, system: UnitSystem = current): string {
  if (bad(km)) return "no data";
  return system === "imperial"
    ? `${(Number(km) / KM_PER_MI).toFixed(digits)} mi`
    : `${Number(km).toFixed(digits)} km`;
}

/** Length/altitude in meters → "12 m" | "39 ft". */
export function fmtMeters(m: number | null | undefined, digits = 0, system: UnitSystem = current): string {
  if (bad(m)) return "no data";
  return system === "imperial"
    ? `${(Number(m) / M_PER_FT).toFixed(digits)} ft`
    : `${Number(m).toFixed(digits)} m`;
}

/** Wave height & similar small lengths in meters → "0.6 m" | "2.0 ft". */
export function fmtMetersSmall(m: number | null | undefined, system: UnitSystem = current): string {
  return fmtMeters(m, 1, system);
}

/** Speed in m/s → "3.0 m/s" | "6.7 mph". */
export function fmtMetersPerSec(ms: number | null | undefined, system: UnitSystem = current): string {
  if (bad(ms)) return "no data";
  return system === "imperial"
    ? `${(Number(ms) * 3600 / (KM_PER_MI * 1000)).toFixed(1)} mph`
    : `${Number(ms).toFixed(1)} m/s`;
}

/** Speed in km/h → "121 km/h" | "75 mph". */
export function fmtKmh(kmh: number | null | undefined, system: UnitSystem = current): string {
  if (bad(kmh)) return "no data";
  return system === "imperial"
    ? `${(Number(kmh) / KM_PER_MI).toFixed(0)} mph`
    : `${Number(kmh).toFixed(0)} km/h`;
}

/** Split a formatter's OUTPUT into { num, unit } for stat-chip layouts where
 *  the unit belongs in the LABEL ("ALT MI / 254" — satellite-UX design
 *  2026-07-18 §1). The value is still produced by the unit-system formatter —
 *  this only re-typesets it (plus thousands grouping on big integers, e.g.
 *  "17139" → "17,139"). "no data" and non-numeric strings pass through whole. */
export function splitUnit(formatted: string): { num: string; unit: string | null } {
  const i = formatted.lastIndexOf(" ");
  if (i <= 0) return { num: formatted, unit: null };
  const rawNum = formatted.slice(0, i);
  const unit = formatted.slice(i + 1);
  if (!/^-?[\d.,]+$/.test(rawNum)) return { num: formatted, unit: null };
  const num = /^-?\d{5,}$/.test(rawNum) ? Number(rawNum).toLocaleString("en-US") : rawNum;
  return { num, unit };
}

/** Temperature in °C → "28.3 °C" | "82.9 °F". */
export function fmtCelsius(c: number | null | undefined, digits = 1, system: UnitSystem = current): string {
  if (bad(c)) return "no data";
  return system === "imperial"
    ? `${(Number(c) * 9 / 5 + 32).toFixed(digits)} °F`
    : `${Number(c).toFixed(digits)} °C`;
}
