// Single source of truth for the /data map's 3D-terrain vertical
// exaggeration (3D TERRAIN UPGRADES handoff §A). The multiplier the DEM
// mesh is deformed by — and the SAME datum the aircraft 3D layer, the 3D
// trail curtains, and the drain-the-ocean mesh swap lock their altitudes
// to (setAltScale coupling). Kept in its own module (the GLOBE_PREF_KEY /
// units.ts precedent) so it is hermetically testable and so datamap.tsx's
// non-React callbacks can read the persisted value without importing React.
//
// Default 1.3× is the shipped production constant — UNCHANGED. The slider
// range is 1.0×–3.0×; anything outside clamps.
export const TERRAIN_EXAG_KEY = "vt-terrain-exag";
export const TERRAIN_EXAG_DEFAULT = 1.3;
export const TERRAIN_EXAG_MIN = 1;
export const TERRAIN_EXAG_MAX = 3;

/** Clamp any candidate into the valid slider band. Non-finite → default. */
export const clampTerrainExag = (v: number): number => {
  if (!Number.isFinite(v)) return TERRAIN_EXAG_DEFAULT;
  return Math.min(TERRAIN_EXAG_MAX, Math.max(TERRAIN_EXAG_MIN, v));
};

/** Read the persisted exaggeration, clamped; the shipped default when unset
 *  or unreadable (SSR / private-mode / corrupt value). */
export const readTerrainExag = (): number => {
  try {
    const raw = window.localStorage.getItem(TERRAIN_EXAG_KEY);
    if (raw == null) return TERRAIN_EXAG_DEFAULT;
    const v = parseFloat(raw);
    if (Number.isFinite(v)) return clampTerrainExag(v);
  } catch {}
  return TERRAIN_EXAG_DEFAULT;
};

/** Persist the exaggeration (clamped first so storage never holds an
 *  out-of-band value). Swallows storage failures like every other pref. */
export const writeTerrainExag = (v: number): void => {
  try { window.localStorage.setItem(TERRAIN_EXAG_KEY, String(clampTerrainExag(v))); } catch {}
};
