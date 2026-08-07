/** Curtain/ground-trace display-datum resolver (repair 2026-08-07, field
 *  report with phone screenshots: "the line for the terrain doesn't follow
 *  sometimes when it's over terrain").
 *
 *  The ground trace is drawn by the custom GL layer at a precomputed
 *  groundZ — it is NOT draped by MapLibre's terrain RTT — so it hugs the
 *  on-screen surface only as well as groundZ matches the RENDERED mesh.
 *  The old build preferred our own fixed-zoom DEM decode (lib/elevation,
 *  ~150 m/px) and used the mesh query only as a fallback: at far camera
 *  distances the rendered mesh is coarser LOD than that sample, so the
 *  trace visibly floated above valleys and cut through ridges. Meanwhile
 *  the moving tail used the OPPOSITE preference (mesh only), so the base
 *  datum jumped at the tail seam.
 *
 *  Resolution order, one rule for track AND tail:
 *   1. mesh query ≠ 0  → trust it: it IS the displayed surface.
 *   2. mesh query = 0 (tile not loaded — or genuinely sea level) → our own
 *      DEM sample × exaggeration. At true sea level the DEM is ≈0 too, so
 *      the ambiguity collapses; off-viewport (no mesh tiles ever) this is
 *      the only real reading — the round-17 sea-level-plunge fix survives.
 *   3. DEM tile still in flight → 0 now, pending=true so the caller's
 *      retry repaint refines it (established behavior).
 *  Below-sea-level terrain (negative mesh reading) is ≠ 0 and correctly
 *  taken from the mesh in step 1. */
export function resolveGroundDisplayZ(
  gMesh: number,
  gDem: number | null,
  altScale: number,
): { g: number; source: "mesh" | "dem" | "pending" } {
  if (gMesh !== 0 && Number.isFinite(gMesh)) return { g: gMesh, source: "mesh" };
  if (gDem != null && Number.isFinite(gDem)) return { g: gDem * altScale, source: "dem" };
  return { g: 0, source: "pending" };
}
