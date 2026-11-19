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

// ── ALL STATES (human request 2026-11-19: "I want the option to have all
// power grids for states, but the way you have all the drop-downs"). One
// control at the top of the "Power grid — US (by state)" drop-down that
// turns every state grid layer on or off.
//
// SAME STATE RULE as the master above: a convenience control over the
// panel's `enabled` map, never parallel state. Its three displayed
// positions (all / some / none) are DERIVED from the per-state values, so a
// user who picked three states sees "some", never a switch that lies.
//
// NOT THE US MASTER. The `powergrid` master draws the whole country from
// ONE overview tile set; the states are 51 separate, more detailed tile
// sets. Measured 2026-11-19 from the served files' own headers: the master
// is tiled to z7 with its densest lines thinned at every zoom, the states
// to z9 (Texas z12). The two are different pictures, so this control turns
// the real state layers on and never swaps in the master behind the label.

/** A US per-state grid layer id: `powergrid_` + the two-letter state code
 *  (`powergrid_tx`; DC is `powergrid_dc`). Every other grid id has a longer
 *  or prefixed suffix (`powergrid_ca_ab`, `powergrid_eu_de`,
 *  `powergrid_hifld`, `powergrid_canada`, the bare `powergrid` master), so
 *  the shape alone separates the states from the masters and the other
 *  regions. `gridMaster.test.ts` pins this against the registry. (It is a
 *  rule about LAYER ids: the US master's map source happens to be named
 *  `powergrid_us`, which is a source, not a layer, and not a state.) */
export function isUsStateGridId(id: string): boolean {
  return /^powergrid_[a-z]{2}$/.test(id);
}

/** The three honest positions of a group control. */
export type GridGroupState = "all" | "some" | "none";

/** How many of `ids` are on. */
export function countGridGroupOn(
  enabled: Record<string, boolean>,
  ids: readonly string[],
): number {
  let n = 0;
  for (const id of ids) if (enabled[id]) n++;
  return n;
}

/** Derived position: "all" only when every id is on, "none" when no id is
 *  on (an empty group is "none": there is nothing to show as on). */
export function gridGroupState(
  enabled: Record<string, boolean>,
  ids: readonly string[],
): GridGroupState {
  const n = countGridGroupOn(enabled, ids);
  if (n === 0) return "none";
  return n === ids.length ? "all" : "some";
}

/** Next `enabled` with exactly `ids` set to `on`. Nothing else is touched:
 *  the masters, HIFLD and every other region keep their values. Input is
 *  never mutated. */
export function setGridGroup(
  enabled: Record<string, boolean>,
  ids: readonly string[],
  on: boolean,
): Record<string, boolean> {
  const next: Record<string, boolean> = { ...enabled };
  for (const id of ids) next[id] = on;
  return next;
}

/** One press of the group control: "all" turns the group off; "some" and
 *  "none" turn it all on (a partly-on group completes, as a tri-state
 *  checkbox does, so a second press is always a clean off). */
export function toggleGridGroup(
  enabled: Record<string, boolean>,
  ids: readonly string[],
): Record<string, boolean> {
  return setGridGroup(enabled, ids, gridGroupState(enabled, ids) !== "all");
}
