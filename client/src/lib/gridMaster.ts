// ALL POWER GRIDS master control (human request 2026-07-31: "can we get an
// all power grid option"). One switch that shows every grid we have mapped
// so far by flipping the CONTINENTAL master layers together — the
// OSM-pipeline masters, which share one source, one styling family, and one
// build script (scripts/build_power_tiles.sh). HIFLD (US authoritative
// transmission) stays an individual toggle: it OVERLAYS the OSM US master
// rather than replacing it, and doubling both by default would double-draw
// the US.
//
// STATE RULE (single source of truth): this is a CONVENIENCE control over
// the layer panel's existing `enabled` map — never parallel state. Its
// displayed position derives from the masters' state; switching it on sets
// only the masters listed in GRID_MASTER_IDS; switching it off clears EVERY
// powergrid_* id (masters + any per-state/province/country picks) so "off"
// honestly means a grid-free map.
//
// Pure + hermetic: unit-tested with `npx tsx --test`.

/** The continental OSM masters — everything mapped so far. Waves 3+ append
 *  here as new continents ship (Africa/Oceania pending). Asia shipped
 *  2026-08-13 (`powergrid_asia`, 37 countries + Russia, R2-served) but was
 *  never added here — found stale 2026-08-28/re-confirmed 2026-08-31 via
 *  `datacore/layers.json` still carrying it `status: "live"` while this
 *  list stopped at four; the master switch and the "All power grids"
 *  toggle silently omitted an entire live continent. See
 *  `gridMaster.test.ts`'s registry-parity test, which now fails the build
 *  the next time this drifts. */
export const GRID_MASTER_IDS = [
  "powergrid", // US, all 50 states + DC
  "powergrid_canada", // 13 provinces & territories
  "powergrid_southamerica", // 13 countries
  "powergrid_europe", // 48 countries/territories
  "powergrid_asia", // 37 countries/regions + Russia
] as const;

/** Every layer the OFF position clears shares this id prefix (masters,
 *  HIFLD variants, and all per-state/province/country toggles). */
export const GRID_ID_PREFIX = "powergrid";

/** The switch reads ON only when every continental master is on. */
export function allGridsOn(enabled: Record<string, boolean>): boolean {
  return GRID_MASTER_IDS.every((id) => !!enabled[id]);
}

/**
 * Next `enabled` state for the master switch. ON: GRID_MASTER_IDS's
 * continental masters only — finer per-region picks are left alone (they
 * draw the same OSM data the masters already show; harmless overlap, user's
 * choice).
 * OFF: every powergrid_* key present goes false, masters included — one
 * motion clears the whole family. Input is never mutated.
 */
export function setAllGrids(
  enabled: Record<string, boolean>,
  on: boolean,
): Record<string, boolean> {
  const next: Record<string, boolean> = { ...enabled };
  if (on) {
    for (const id of GRID_MASTER_IDS) next[id] = true;
  } else {
    for (const k of Object.keys(next)) {
      if (k.startsWith(GRID_ID_PREFIX)) next[k] = false;
    }
    for (const id of GRID_MASTER_IDS) next[id] = false;
  }
  return next;
}
