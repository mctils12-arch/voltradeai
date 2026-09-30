// CUSTOM-LAYER KEEPER (2026-09-30, live report: "I clicked the map's
// fullscreen button, the selected plane's curtain disappeared and only came
// back after clicking the plane again").
//
// MapLibre keeps custom layers only as long as nothing rebuilds the style's
// layer list: a GL context restore re-applies the SERIALIZED style (which
// skips type 'custom'), and a style/basemap/chart-view rebuild does the
// same. Every owner used to re-add its own layer on its own schedule — the
// flight track only on its next repaint (next NEW fix or the 30 s archive
// refresh), and it was never in the restore registry at all — so a canvas
// resize that cost the context (fullscreen at 1920 px) dropped the curtain
// until the next click.
//
// The keeper is the ONE re-add path: registered with frameCore (Law I — a
// frame-loop poll, never a map-event handler), it checks the registry every
// KEEPER_CHECK_MS and re-adds any registered layer the style no longer has,
// in registration order, with an optional beforeId (the gray plan sits just
// under the live curtain). It never re-adds a layer its owner deliberately
// removed: owners delete from the registry first (intentional removal is not
// a restore case). Zero cost when nothing is missing: one getLayer per
// registered id per check.

import { frameCore, PRIORITY, type FrameLoop } from '../render/frameCore.js';

/** How often the keeper looks (ms of frame time). */
export const KEEPER_CHECK_MS = 250;

export interface KeeperMapLike {
  getLayer(id: string): unknown;
  addLayer(layer: unknown, beforeId?: string): unknown;
  /** false while a style is (re)loading — addLayer would throw */
  isStyleLoaded?(): boolean | void;
  triggerRepaint?(): void;
}

export interface RestoreResult {
  restored: string[];
  failed: string[];
}

/**
 * Pure (given the map): re-add every registered layer the map no longer
 * has. `beforeOf(id)` names the layer it must sit under (only honoured when
 * that layer is present). A throwing addLayer (style mid-load) is reported in
 * `failed` — the next check retries.
 */
export function restoreMissingLayers(
  map: KeeperMapLike,
  registry: ReadonlyMap<string, unknown>,
  beforeOf?: (id: string) => string | undefined,
): RestoreResult {
  const out: RestoreResult = { restored: [], failed: [] };
  for (const [id, impl] of registry) {
    let present = false;
    try { present = !!map.getLayer(id); } catch (e) { out.failed.push(id); continue; }
    if (present) continue;
    try {
      const b = beforeOf?.(id);
      const before = b && map.getLayer(b) ? b : undefined;
      map.addLayer(impl, before);
      if (map.getLayer(id)) out.restored.push(id);
      else out.failed.push(id);
    } catch (e) {
      out.failed.push(id);
    }
  }
  return out;
}

export interface LayerKeeperOptions {
  map: KeeperMapLike;
  registry: ReadonlyMap<string, unknown>;
  beforeOf?: (id: string) => string | undefined;
  /** called (from the frame loop) after a check re-added at least one layer */
  onRestored?: (ids: string[]) => void;
  loop?: FrameLoop | null;
  checkMs?: number;
}

/** Start the keeper; returns its stop function (Law IV teardown). */
export function startLayerKeeper(o: LayerKeeperOptions): () => void {
  const loop = o.loop ?? frameCore();
  const every = o.checkMs ?? KEEPER_CHECK_MS;
  let acc = every; // first frame checks immediately
  const unregister = loop.register((dt) => {
    acc += dt;
    if (acc < every) return;
    acc = 0;
    if (!o.registry.size) return;
    try { if (o.map.isStyleLoaded && o.map.isStyleLoaded() === false) return; } catch (e) { return; }
    const r = restoreMissingLayers(o.map, o.registry, o.beforeOf);
    if (r.restored.length) {
      try { o.onRestored?.(r.restored); } finally { o.map.triggerRepaint?.(); }
    }
  }, PRIORITY.STREAM, { label: 'layerKeeper' });
  return unregister;
}
