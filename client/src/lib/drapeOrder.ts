// DRAPE-ORDER GUARD — keeps custom GL layers ABOVE every terrain-draped
// style layer, permanently.
//
// WHY (probe-proven 2026-07-20, scratchpad/perf_audit/mechanism.mjs): with
// 3D terrain on, MapLibre renders draped layer types (background/fill/
// line/raster/hillshade/color-relief — LAYERS_TO_TEXTURES, maplibre
// render_to_texture.ts) into per-tile textures. CONSECUTIVE draped layers
// share one texture "stack"; a custom layer sandwiched between draped
// layers SPLITS the stack, and a split defeats the cross-frame texture
// cache entirely — every visible tile re-rasterizes every frame. Measured:
// aircraft-3d buried under aircraft-veclines = 588ms/frame with 12 texture
// allocations per frame; the same layer moved to the top = 180ms/frame
// with 0 (the full "terrain very laggy" report). A no-op render at the
// buried position still cost 585ms — position is everything.
//
// The burial is an add-ORDER accident (custom layers mount before their
// sibling style layers arrive, later layers append on top), so the guard
// re-floats buried custom layers on every styledata event: idempotent —
// once nothing is buried it does nothing, so the event it itself triggers
// terminates the loop. Visual note: custom layers here are the 3D craft /
// track / orbit renderers, which are meant to draw above ground overlays.

export const DRAPED_TYPES: ReadonlySet<string> = new Set([
  "background", "fill", "line", "raster", "hillshade", "color-relief",
]);

export interface OrderedLayer { id: string; type: string }

/** Pure core: ids of custom layers that have at least one DRAPED layer
 *  above them (those are the stack-splitters; a custom layer under only
 *  symbols/circles splits nothing). Order = bottom → top. */
export function buriedCustomIds(layers: OrderedLayer[]): string[] {
  const out: string[] = [];
  for (let i = 0; i < layers.length; i++) {
    if (layers[i].type !== "custom") continue;
    for (let j = i + 1; j < layers.length; j++) {
      if (DRAPED_TYPES.has(layers[j].type)) { out.push(layers[i].id); break; }
    }
  }
  return out;
}

/** Minimal structural view of a MapLibre map for the guard (style._order /
 *  _layers are stable public-ish internals; getStyle() omits custom layers
 *  so it cannot be used here). */
interface MapLike {
  style?: { _order?: string[]; _layers?: Record<string, { id: string; type: string } | undefined> };
  moveLayer(id: string): void;
  on(ev: string, fn: () => void): unknown;
  off(ev: string, fn: () => void): unknown;
}

/** Read the FULL layer order (custom layers included) off the style
 *  internals. Empty when the style isn't ready — the guard just no-ops. */
export function fullLayerOrder(map: MapLike): OrderedLayer[] {
  try {
    const order = map.style?._order;
    const layers = map.style?._layers;
    if (!order || !layers) return [];
    const out: OrderedLayer[] = [];
    for (const id of order) {
      const l = layers[id];
      if (l) out.push({ id: l.id, type: l.type });
    }
    return out;
  } catch {
    return [];
  }
}

/** Float every buried custom layer to the top (relative order preserved —
 *  they are moved bottom-most first). Returns how many moved. No-throw. */
export function floatBuriedCustomLayers(map: MapLike): number {
  let moved = 0;
  try {
    for (const id of buriedCustomIds(fullLayerOrder(map))) {
      try { map.moveLayer(id); moved++; } catch { /* layer vanished mid-pass */ }
    }
  } catch { /* style torn down */ }
  return moved;
}

/** Install the guard: re-float on every styledata (layer adds/removes fire
 *  it). Reentrancy-safe — the pass only mutates when something is buried,
 *  so the styledata its own moveLayer fires finds a clean order and stops.
 *  Returns the uninstaller. */
export function installDrapeOrderGuard(map: MapLike): () => void {
  let inPass = false;
  const pass = () => {
    if (inPass) return;
    inPass = true;
    try { floatBuriedCustomLayers(map); } finally { inPass = false; }
  };
  map.on("styledata", pass);
  pass(); // current state may already be buried
  return () => { try { map.off("styledata", pass); } catch {} };
}
