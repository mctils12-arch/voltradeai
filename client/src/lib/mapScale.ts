// Our own map scale/zoom readout (2026-07-22 human directive: "at a scale
// of the map indicator, the amount of zoom that you're looking at on the
// map can we build our own, and put that and the capture data in one thing
// at the bottom"). Replaces MapLibre's ScaleControl so the scale, the zoom,
// and the imagery capture date live in ONE compact element.
//
// Pure + unit-tested: given the camera (zoom, latitude) and the user's unit
// system, produce a "nice" round scale distance and the pixel width its bar
// should occupy. Web-mercator ground resolution at the equator is
// 156543.03392 m/px at z0; latitude compresses it by cos(lat).

export type UnitSystem = "imperial" | "metric";

const EQUATOR_M_PER_PX_Z0 = 156543.03392804097;
const FEET_PER_M = 3.28084;
const FEET_PER_MILE = 5280;
const M_PER_MILE = 1609.344;

/** meters covered by one screen pixel at this zoom + latitude */
export function metersPerPixel(zoom: number, latDeg: number): number {
  const latClamped = Math.max(-85, Math.min(85, latDeg));
  return (EQUATOR_M_PER_PX_Z0 * Math.cos((latClamped * Math.PI) / 180)) / Math.pow(2, zoom);
}

/** the largest "nice" number (1/2/3/5 × 10ⁿ) not exceeding v */
function niceRound(v: number): number {
  if (!(v > 0) || !Number.isFinite(v)) return 0;
  const pow = Math.pow(10, Math.floor(Math.log10(v)));
  const f = v / pow;
  const nice = f >= 5 ? 5 : f >= 3 ? 3 : f >= 2 ? 2 : 1;
  return nice * pow;
}

export interface ScaleReading {
  /** human label, e.g. "200 mi" or "50 km" */
  label: string;
  /** how many screen pixels that distance spans (the bar width) */
  widthPx: number;
}

/**
 * Compute a scale bar targeting ~`targetPx` pixels: pick a nice round
 * distance in the user's units, then report the exact pixel width it maps
 * to (so the bar is honest — the number is round, the bar length is exact).
 */
export function scaleReading(
  zoom: number,
  latDeg: number,
  system: UnitSystem,
  targetPx = 90,
): ScaleReading {
  const mPerPx = metersPerPixel(zoom, latDeg);
  if (!(mPerPx > 0)) return { label: "—", widthPx: 0 };
  const targetMeters = mPerPx * targetPx;

  if (system === "imperial") {
    // sub-mile → feet; else miles
    if (targetMeters < M_PER_MILE) {
      const targetFeet = targetMeters * FEET_PER_M;
      const feet = niceRound(targetFeet);
      return { label: `${fmtInt(feet)} ft`, widthPx: (feet / FEET_PER_M) / mPerPx };
    }
    const targetMiles = targetMeters / M_PER_MILE;
    const miles = niceRound(targetMiles);
    return { label: `${fmtInt(miles)} mi`, widthPx: (miles * M_PER_MILE) / mPerPx };
  }
  // metric: sub-km → m; else km
  if (targetMeters < 1000) {
    const m = niceRound(targetMeters);
    return { label: `${fmtInt(m)} m`, widthPx: m / mPerPx };
  }
  const targetKm = targetMeters / 1000;
  const km = niceRound(targetKm);
  return { label: `${fmtInt(km)} km`, widthPx: (km * 1000) / mPerPx };
}

function fmtInt(n: number): string {
  return Math.round(n).toLocaleString("en-US");
}

/** compact zoom label for the readout (e.g. "z9.4") */
export function zoomLabel(zoom: number): string {
  if (!Number.isFinite(zoom)) return "";
  return `z${zoom.toFixed(1)}`;
}
