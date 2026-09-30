// PROCEDURE LAYER — the selected flight's instrument procedure on the map
// (the ForeFlight "plate on the map" look): the FAA CIFP path as a line in
// the procedure accent (--accent-purple — the theme's closest token to the
// aviation magenta line), fix SYMBOLS by kind with their altitude
// constraints, and — only when the server georeferenced the chart — the
// plate's plan view as a raster underneath, with a user opacity.
//
// Native MapLibre sources/layers (GPU symbol/line/raster paths, viewport-
// culled by MapLibre). Law I: no map-event handlers anywhere in this module;
// opacity/appearance changes are TARGETS that MapLibre's own per-frame paint
// transitions interpolate toward (the 250 ms fade doubles as Law II's
// crossfade — the plate is added only after its bitmap is decoded, at 0
// opacity, then eased in). A style reload (basemap switch) drops native
// sources, so a cheap frame-loop check re-installs whatever is current.
// Law IV: maxFeatures caps the drawn path (legs + fixes); the plate raster is
// capped at PLATE_MAX_PX on its longest side; dispose() removes every
// source/layer, revokes the plate's blob URL and unregisters the frame check.

import type { FrameLoop } from "../../render/frameCore.js";
import { PRIORITY } from "../../render/frameCore.js";
import { PROC_MAX_FEATURES, fixSymbol, pathToLayers, type ProcedurePath } from "./procedures.js";

export const maxFeatures = PROC_MAX_FEATURES;
/** longest side of the plate raster, px (1536² RGBA ≈ 9.4 MB) */
export const PLATE_MAX_PX = 1536;
export const vramBudget = 10; // MB — one plate raster + a few hundred line/symbol vertices
export const FADE_MS = 250;
/** how often (frames) the style-reload check runs */
export const ENSURE_EVERY_FRAMES = 30;

export const SRC_LEGS = "vt-proc-legs";
export const SRC_FIXES = "vt-proc-fixes";
export const SRC_PLATE = "vt-proc-plate";
export const LYR_PLATE = "vt-proc-plate";
export const LYR_CASING = "vt-proc-legs-casing";
export const LYR_LEGS = "vt-proc-legs";
export const LYR_DASH = "vt-proc-legs-dash";
export const LYR_FIXES = "vt-proc-fixes";
const ALL_LAYERS = [LYR_FIXES, LYR_DASH, LYR_LEGS, LYR_CASING, LYR_PLATE];

/** The subset of maplibre-gl's Map this layer uses (tests pass a fake). */
export interface ProcMapLike {
  getSource(id: string): unknown;
  addSource(id: string, src: unknown): unknown;
  removeSource(id: string): unknown;
  getLayer(id: string): unknown;
  addLayer(layer: unknown, beforeId?: string): unknown;
  removeLayer(id: string): unknown;
  setPaintProperty(layer: string, prop: string, value: unknown): unknown;
  hasImage?(id: string): boolean;
}

export interface PlateOverlay {
  /** object/blob URL of the decoded plan-view crop */
  url: string;
  /** [lon, lat] TL, TR, BR, BL */
  corners: Array<[number, number]>;
}

export interface ProcedureLayerOpts {
  loop?: FrameLoop | null;
  /** resolves a theme token (CSS custom property) to a colour */
  color?: (token: string) => string;
  /** revoke a plate URL we own (tests inject a spy) */
  revoke?: (url: string) => void;
  /** registers the SDF fix symbols if the style lost them */
  ensureIcons?: () => void;
  onError?: (where: string, e: unknown) => void;
}

const EMPTY = { type: "FeatureCollection", features: [] as unknown[] };

export class ProcedureLayer {
  private path: ProcedurePath | null = null;
  private plate: PlateOverlay | null = null;
  private opacity = 0.7;
  private disposed = false;
  private unregister: (() => void) | null = null;
  private frames = 0;

  constructor(private readonly map: ProcMapLike, private readonly opts: ProcedureLayerOpts = {}) {
    if (opts.loop) {
      this.unregister = opts.loop.register(() => {
        if (++this.frames % ENSURE_EVERY_FRAMES === 0) this.ensure();
      }, PRIORITY.STREAM, { label: "procedure-layer" });
    }
  }

  private color(token: string): string { return this.opts.color ? this.opts.color(token) : "white"; }
  private err(where: string, e: unknown) { this.opts.onError?.(where, e); }

  /** Draw (or clear with null) the procedure path. */
  setPath(path: ProcedurePath | null): void {
    if (this.disposed) return;
    this.path = path;
    const L = pathToLayers(path, maxFeatures);
    for (const f of L.fixes.features) f.properties = { ...f.properties, icon: fixSymbol(f.properties) };
    this.ensure();
    this.setData(SRC_LEGS, L.legs);
    this.setData(SRC_FIXES, L.fixes);
    this.applyTargets();
  }

  /** Place (or remove with null) the georeferenced plate raster. The URL
   *  must point at an already-decoded bitmap (ready-gate is the caller's
   *  decode; this layer then fades it in from 0). */
  setPlate(plate: PlateOverlay | null): void {
    if (this.disposed) { if (plate) this.revoke(plate.url); return; }
    const prev = this.plate;
    this.removePlate();
    if (prev && prev.url !== plate?.url) this.revoke(prev.url);
    this.plate = plate;
    if (plate) this.installPlate();
  }

  setOpacity(v: number): void {
    this.opacity = Math.max(0, Math.min(1, v));
    if (this.plate) this.paint(LYR_PLATE, "raster-opacity", this.opacity);
  }

  getOpacity(): number { return this.opacity; }
  hasPlate(): boolean { return !!this.plate; }
  isDisposed(): boolean { return this.disposed; }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.unregister?.();
    this.unregister = null;
    for (const id of ALL_LAYERS) this.safe("remove-layer", () => { if (this.map.getLayer(id)) this.map.removeLayer(id); });
    for (const id of [SRC_FIXES, SRC_LEGS, SRC_PLATE]) this.safe("remove-source", () => { if (this.map.getSource(id)) this.map.removeSource(id); });
    if (this.plate) this.revoke(this.plate.url);
    this.plate = null;
    this.path = null;
  }

  // ── internals ─────────────────────────────────────────────────────────────

  /** Opacity TARGETS (MapLibre's paint transitions ease toward them per
   *  frame): unselected transitions of the procedure read fainter. */
  private applyTargets() {
    const on = !!this.path;
    const sel = (a: number, b: number) => (on ? ["case", ["==", ["get", "selected"], false], a, b] : 0);
    this.paint(LYR_CASING, "line-opacity", sel(0.35, 0.75));
    this.paint(LYR_LEGS, "line-opacity", sel(0.45, 1));
    this.paint(LYR_DASH, "line-opacity", sel(0.45, 1));
    this.paint(LYR_FIXES, "icon-opacity", on ? 1 : 0);
    this.paint(LYR_FIXES, "text-opacity", on ? 1 : 0);
  }

  private revoke(url: string) { try { (this.opts.revoke ?? ((u: string) => URL.revokeObjectURL(u)))(url); } catch (e: unknown) { this.err("revoke", e); } }

  private safe(where: string, fn: () => void) { try { fn(); } catch (e: unknown) { this.err(where, e); } }

  private paint(layer: string, prop: string, v: unknown) {
    this.safe("paint", () => { if (this.map.getLayer(layer)) this.map.setPaintProperty(layer, prop, v); });
  }

  private setData(src: string, data: unknown) {
    this.safe("set-data", () => {
      const s = this.map.getSource(src) as { setData?: (d: unknown) => void } | undefined;
      s?.setData?.(data);
    });
  }

  /** (Re)install sources + layers that are missing (first use, or after a
   *  style reload dropped them). Idempotent and cheap when all present. */
  ensure(): void {
    if (this.disposed) return;
    const m = this.map;
    const missing = !m.getSource(SRC_LEGS) || !m.getLayer(LYR_LEGS) || !m.getSource(SRC_FIXES) || !m.getLayer(LYR_FIXES);
    if (!missing && (!this.plate || m.getLayer(LYR_PLATE))) return;
    this.safe("ensure", () => {
      this.opts.ensureIcons?.();
      const accent = this.color("--accent-purple");
      const muted = this.color("--text-secondary");
      const halo = this.color("--bg-primary");
      const L = pathToLayers(this.path, maxFeatures);
      for (const f of L.fixes.features) f.properties = { ...f.properties, icon: fixSymbol(f.properties) };
      if (!m.getSource(SRC_LEGS)) m.addSource(SRC_LEGS, { type: "geojson", data: this.path ? L.legs : EMPTY });
      if (!m.getSource(SRC_FIXES)) m.addSource(SRC_FIXES, { type: "geojson", data: this.path ? L.fixes : EMPTY });
      const fade = { duration: FADE_MS, delay: 0 };
      const on = 0; // every layer is added transparent, then eased to its target
      if (!m.getLayer(LYR_CASING)) {
        m.addLayer({
          id: LYR_CASING, type: "line", source: SRC_LEGS,
          layout: { "line-join": "round", "line-cap": "round" },
          paint: { "line-color": halo, "line-width": 5.5, "line-opacity": on, "line-opacity-transition": fade },
        });
      }
      if (!m.getLayer(LYR_LEGS)) {
        m.addLayer({
          id: LYR_LEGS, type: "line", source: SRC_LEGS,
          filter: ["all", ["!=", ["get", "approx"], true], ["!=", ["get", "missed"], true]],
          layout: { "line-join": "round", "line-cap": "round" },
          paint: { "line-color": accent, "line-width": 3, "line-opacity": on, "line-opacity-transition": fade },
        });
      }
      if (!m.getLayer(LYR_DASH)) {
        // approximated legs (to-altitude / intercept / vectors / holds) and
        // the missed approach are DASHED: drawn honestly as not-fixed paths
        m.addLayer({
          id: LYR_DASH, type: "line", source: SRC_LEGS,
          filter: ["any", ["==", ["get", "approx"], true], ["==", ["get", "missed"], true]],
          layout: { "line-join": "round" },
          paint: {
            "line-color": ["case", ["==", ["get", "missed"], true], muted, accent],
            "line-width": 2.4, "line-dasharray": [2, 1.6], "line-opacity": on, "line-opacity-transition": fade,
          },
        });
      }
      if (!m.getLayer(LYR_FIXES)) {
        m.addLayer({
          id: LYR_FIXES, type: "symbol", source: SRC_FIXES,
          layout: {
            "icon-image": ["get", "icon"], "icon-size": 0.42, "icon-allow-overlap": true,
            "text-field": ["get", "label"], "text-font": ["Open Sans Semibold"], "text-size": 11,
            "text-offset": [0.9, 0], "text-anchor": "left", "text-optional": true,
          },
          paint: {
            "icon-color": ["case", ["==", ["get", "missed"], true], muted, accent],
            "text-color": this.color("--text-primary"), "text-halo-color": halo, "text-halo-width": 1.4,
            "icon-opacity": on, "text-opacity": on, "icon-opacity-transition": fade, "text-opacity-transition": fade,
          },
        });
      }
      if (this.plate && !m.getLayer(LYR_PLATE)) this.installPlate();
      this.applyTargets();
    });
  }

  private installPlate() {
    const p = this.plate;
    if (!p) return;
    this.safe("plate", () => {
      if (!this.map.getSource(SRC_PLATE)) this.map.addSource(SRC_PLATE, { type: "image", url: p.url, coordinates: p.corners });
      if (!this.map.getLayer(LYR_PLATE)) {
        this.map.addLayer({
          id: LYR_PLATE, type: "raster", source: SRC_PLATE,
          paint: { "raster-opacity": 0, "raster-opacity-transition": { duration: FADE_MS, delay: 0 }, "raster-fade-duration": 0 },
        }, this.map.getLayer(LYR_CASING) ? LYR_CASING : undefined);
      }
      // ease in to the user's opacity (Law II crossfade: never a pop)
      this.map.setPaintProperty(LYR_PLATE, "raster-opacity", this.opacity);
    });
  }

  private removePlate() {
    this.safe("plate-remove", () => {
      if (this.map.getLayer(LYR_PLATE)) this.map.removeLayer(LYR_PLATE);
      if (this.map.getSource(SRC_PLATE)) this.map.removeSource(SRC_PLATE);
    });
  }
}
