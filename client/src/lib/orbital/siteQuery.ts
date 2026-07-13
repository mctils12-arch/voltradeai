// Ground-site coverage query for a named satellite constellation (ORBITAL
// program O7: research/orbital_program.md's "COVERAGE / FOOTPRINT ANALYSIS"
// tools, Starlink slice).
//
// The charter's other O7 example — GPS DOP at a point — is NOT buildable
// against this catalog yet: GPS is a MEO constellation (~718 min orbital
// period), and propagate.ts's near-earth-only SGP4 kernel returns null
// (deep-space, no position) for every orbit >= 225 min by design (see its
// own docstring). Querying GPS today would honestly report zero visible
// satellites forever — not a real "no coverage" finding, just a modeling
// gap. Starlink is LEO (~95 min period), propagates correctly with the same
// kernel, and is the constellation the charter names as public/well-known
// (~25deg user-terminal elevation mask) — so this module ships that slice.
// GPS DOP stays filed in orbital_program.md's RESUME STATE, gated on an
// SDP4 upgrade, not built here.
//
// Pure + hermetic: reads an already-propagated position buffer (the same
// one satLayer/pick.ts consume) and the index-aligned GP catalog; no DOM,
// no network, no clock. Directly unit-testable (`npx tsx --test`).

import { elevationAzimuthFromSite, STARLINK_MIN_ELEV_DEG } from './geometry.js';
import { readSatAt, mercatorToLonLat } from './satBuffer.js';
import type { GpRecord } from './tle.js';

/** One constellation member visible above the site's elevation mask. */
export interface SiteSatellite {
  noradId: number;
  name: string | null;
  elevationDeg: number;
  azimuthDeg: number;
  rangeKm: number;
}

export interface SiteCoverageReport {
  /**
   * Catalog entries matching the name filter that ALSO have a valid
   * propagated position this tick (deep-space/invalid slots are excluded,
   * not silently counted as "no coverage") — the honest denominator for
   * "N of M modeled Starlinks currently cover this point."
   */
  totalModeled: number;
  /** Visible members (elevationDeg >= minElevDeg), highest elevation first. */
  visible: SiteSatellite[];
}

/** True for CelesTrak GP names in the Starlink constellation (e.g. "STARLINK-1007"). */
export function isStarlinkName(name: string | null | undefined): boolean {
  return typeof name === 'string' && /^STARLINK\b/i.test(name.trim());
}

/**
 * Click-routing guard for the O7 coverage fallback (datamap's satellite
 * click handler). The /data basemap is raster-only (background + imagery
 * tiles), so ANY rendered vector feature under the click belongs to a data
 * layer (port markers, power plants, alert polygons, …) — that layer's own
 * handler (or deliberate lack of one) owns the click. The Starlink coverage
 * card may only claim clicks on genuinely empty ground; it must never paint
 * over — or race — a feature layer's popup (live bug: clicking the LA port
 * marker surfaced "Starlink coverage" instead of the port).
 */
export function coverageQueryAllowed(featuresAtPoint: readonly unknown[]): boolean {
  return featuresAtPoint.length === 0;
}

/**
 * Which members of a named constellation currently cover (siteLatDeg,
 * siteLonDeg) above `minElevDeg`, from an already-propagated position
 * buffer. `positions`/`stride`/`gp` are the SAME index-aligned triple
 * pick.ts's picking uses (satWorker.ts's INDEX ALIGNMENT contract) — this
 * function does not propagate anything itself, only re-projects buffer
 * slots that are already valid this tick.
 *
 * Zero visible members with totalModeled > 0 is a genuine "blackout" — no
 * currently-modeled member of the constellation covers this point right
 * now. Zero visible WITH totalModeled === 0 means nothing in the current
 * catalog fetch matched the name filter at all (a different, honest,
 * distinguishable case the caller should word differently).
 */
export function siteCoverageReport(
  positions: ArrayLike<number>,
  stride: number,
  gp: GpRecord[],
  siteLatDeg: number,
  siteLonDeg: number,
  opts: { namePredicate?: (name: string | null) => boolean; minElevDeg?: number } = {},
): SiteCoverageReport {
  const namePredicate = opts.namePredicate ?? isStarlinkName;
  const minElevDeg = opts.minElevDeg ?? STARLINK_MIN_ELEV_DEG;
  const total = Math.min(gp.length, Math.floor(positions.length / stride));

  let totalModeled = 0;
  const visible: SiteSatellite[] = [];
  for (let i = 0; i < total; i++) {
    const g = gp[i];
    if (!g || !namePredicate(g.name)) continue;
    const slot = readSatAt(positions, i);
    if (!slot.valid) continue; // deep-space/invalid this tick — not modeled, not counted
    totalModeled++;
    const { lonDeg, latDeg } = mercatorToLonLat(slot.mercX, slot.mercY);
    const look = elevationAzimuthFromSite(
      latDeg, lonDeg, slot.altMeters / 1000, siteLatDeg, siteLonDeg,
    );
    if (look.elevationDeg >= minElevDeg) {
      visible.push({
        noradId: g.noradId,
        name: g.name,
        elevationDeg: look.elevationDeg,
        azimuthDeg: look.azimuthDeg,
        rangeKm: look.rangeKm,
      });
    }
  }
  visible.sort((a, b) => b.elevationDeg - a.elevationDeg);
  return { totalModeled, visible };
}
