// ── TERRAIN VERTICAL EXAGGERATION pref store (user-adjustable, replaces the
// historic hard-coded 1.3). Same localStorage pattern as GLOBE_PREF_KEY in
// datamap.tsx, extracted here so the clamp/read logic is unit-testable.
// Single source of truth: the terrain mesh setTerrain, the aircraft 3D
// altitude datum (airLayer.setAltScale), and the 3D trail curtains all read
// this ONE value so vertical datums stay in lock-step (a plane above a peak
// must clear the *exaggerated* peak). Persisted so relief survives reload.
export const TERRAIN_EXAG_KEY = "vt-terrain-exag";
export const TERRAIN_EXAG_MIN = 1;
export const TERRAIN_EXAG_MAX = 3;
export const TERRAIN_EXAG_DEFAULT = 1.3; // historic production constant — default unchanged

/** Clamp an arbitrary parsed value into the legal exaggeration range;
 *  anything non-finite falls back to the default. Pure — the testable core. */
export function clampTerrainExag(v: number): number {
  if (!Number.isFinite(v)) return TERRAIN_EXAG_DEFAULT;
  return Math.min(TERRAIN_EXAG_MAX, Math.max(TERRAIN_EXAG_MIN, v));
}

export function readTerrainExag(): number {
  try {
    const v = parseFloat(window.localStorage.getItem(TERRAIN_EXAG_KEY) ?? "");
    if (Number.isFinite(v)) return clampTerrainExag(v);
  } catch {}
  return TERRAIN_EXAG_DEFAULT;
}
