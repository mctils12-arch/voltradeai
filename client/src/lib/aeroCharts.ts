// aeroCharts — FAA aeronautical chart base views on the /data map
// (client half of server/aeroCharts.ts). Pure + hermetic: the view-state
// reducer, persistence, the MapLibre source/paint builders and the badge
// text all live here so they are testable without a map or a DOM.
//
// RENDERING & MOTION LAW, Law II — how the chart raster complies. The chart
// is a MapLibre raster source exactly like the Earth base imagery, so it
// inherits the same pipeline render/mapBaseConfig.ts documents: MapLibre's
// own ready-gate (a child tile is drawn only once decoded + uploaded; the
// parent/overzoomed ancestor keeps covering its footprint meanwhile),
// abortable requests cancelled on viewport exit, its LRU tile cache, and the
// shared RASTER_FADE_MS crossfade via baseRasterPaint(). `maxzoom` is the
// chart's real top LOD, so deeper zooms overzoom the last real tile (never
// request a 404). The pinned-base clause is met by the satellite imagery
// underneath: outside chart coverage (and while a chart tile streams) the
// satellite shows through — never a black hole. Law II.8: every tile URL is
// OUR origin (/tiles/aero/...), never the FAA/ArcGIS service.
//
// Law I: nothing here listens to a map event. Switching view / opacity is a
// user action that sets style state once; MapLibre's own frame loop draws.

import { baseRasterPaint } from "../render/mapBaseConfig.ts";

export const AERO_VIEW_IDS = ["satellite", "sectional", "tac", "ifrlow", "ifrhigh"] as const;
export type AeroViewId = (typeof AERO_VIEW_IDS)[number];
export type AeroChartViewId = Exclude<AeroViewId, "satellite">;

/** Fallback metadata (verified against the FAA services 2026-09-30) — used
 *  until the registry / status endpoint answers. Edition deliberately null:
 *  no freshness is claimed before the server says what it has. */
export interface AeroChartMeta {
  id: AeroChartViewId;
  label: string;
  short: string;
  minzoom: number;
  maxzoom: number;
  edition: string | null;
  effective: string | null;
  expires: string | null;
  expired: boolean;
  behindCurrentCycle: boolean;
  tiles: string;
}

export const AERO_CHART_DEFAULTS: Record<AeroChartViewId, AeroChartMeta> = {
  sectional: { id: "sectional", label: "VFR Sectional", short: "Sectional", minzoom: 8, maxzoom: 12,
    edition: null, effective: null, expires: null, expired: false, behindCurrentCycle: false,
    tiles: "/tiles/aero/sectional/{z}/{x}/{y}" },
  tac: { id: "tac", label: "VFR Terminal Area (TAC)", short: "Terminal", minzoom: 10, maxzoom: 12,
    edition: null, effective: null, expires: null, expired: false, behindCurrentCycle: false,
    tiles: "/tiles/aero/tac/{z}/{x}/{y}" },
  ifrlow: { id: "ifrlow", label: "IFR Low (Enroute + Area)", short: "IFR Low", minzoom: 7, maxzoom: 12,
    edition: null, effective: null, expires: null, expired: false, behindCurrentCycle: false,
    tiles: "/tiles/aero/ifrlow/{z}/{x}/{y}" },
  ifrhigh: { id: "ifrhigh", label: "IFR Enroute High", short: "IFR High", minzoom: 5, maxzoom: 9,
    edition: null, effective: null, expires: null, expired: false, behindCurrentCycle: false,
    tiles: "/tiles/aero/ifrhigh/{z}/{x}/{y}" },
};

export function isAeroViewId(s: unknown): s is AeroViewId {
  return typeof s === "string" && (AERO_VIEW_IDS as readonly string[]).includes(s);
}

/** Only accept a server row whose tile template is OUR origin (Law II.8) —
 *  a registry entry pointing anywhere else is ignored, never rendered. */
export function mergeAeroMeta(rows: unknown): Record<AeroChartViewId, AeroChartMeta> {
  const out: Record<AeroChartViewId, AeroChartMeta> = { ...AERO_CHART_DEFAULTS };
  if (!Array.isArray(rows)) return out;
  for (const r of rows as Array<Record<string, unknown> | null>) {
    const id: unknown = r?.id;
    if (!r || !isAeroViewId(id) || id === "satellite") continue;
    const base = AERO_CHART_DEFAULTS[id];
    const tiles = typeof r.tiles === "string" && r.tiles.startsWith(`/tiles/aero/${id}/`) ? r.tiles : base.tiles;
    const num = (v: unknown, d: number) => (typeof v === "number" && Number.isInteger(v) ? v : d);
    const str = (v: unknown) => (typeof v === "string" && v ? v : null);
    out[id] = {
      ...base,
      minzoom: num(r.minzoom, base.minzoom), maxzoom: num(r.maxzoom, base.maxzoom),
      edition: str(r.edition), effective: str(r.effective), expires: str(r.expires),
      expired: r.expired === true, behindCurrentCycle: r.behindCurrentCycle === true,
      tiles,
    };
  }
  return out;
}

// ── view state (reducer) ────────────────────────────────────────────────────

export interface AeroViewState {
  view: AeroViewId;
  /** chart opacity over the satellite, 0..100 */
  opacity: number;
}

export const AERO_VIEW_DEFAULT: AeroViewState = { view: "satellite", opacity: 100 };
export const AERO_OPACITY_MIN = 10;

export type AeroViewAction =
  | { type: "setView"; view: AeroViewId }
  | { type: "setOpacity"; opacity: number }
  | { type: "reset" };

export function clampOpacity(v: number): number {
  if (!Number.isFinite(v)) return AERO_VIEW_DEFAULT.opacity;
  return Math.round(Math.min(100, Math.max(AERO_OPACITY_MIN, v)));
}

export function aeroViewReducer(s: AeroViewState, a: AeroViewAction): AeroViewState {
  switch (a.type) {
    case "setView":
      return isAeroViewId(a.view) && a.view !== s.view ? { ...s, view: a.view } : s;
    case "setOpacity": {
      const o = clampOpacity(a.opacity);
      return o === s.opacity ? s : { ...s, opacity: o };
    }
    case "reset":
      return AERO_VIEW_DEFAULT;
    default:
      return s;
  }
}

export const AERO_VIEW_PREF_KEY = "vt-aero-view";

interface StorageLike { getItem(k: string): string | null; setItem(k: string, v: string): void }

function storage(): StorageLike | null {
  try {
    return globalThis.localStorage ?? null;
  } catch {
    // private mode / blocked site data: preference simply isn't remembered
    return null;
  }
}

/** Per-viewer convenience only — any failure yields the default. */
export function readAeroViewPref(store: StorageLike | null = storage()): AeroViewState {
  if (!store) return AERO_VIEW_DEFAULT;
  try {
    const raw = store.getItem(AERO_VIEW_PREF_KEY);
    if (!raw) return AERO_VIEW_DEFAULT;
    const j = JSON.parse(raw) as { view?: unknown; opacity?: unknown };
    return {
      view: isAeroViewId(j.view) ? j.view : AERO_VIEW_DEFAULT.view,
      opacity: clampOpacity(typeof j.opacity === "number" ? j.opacity : AERO_VIEW_DEFAULT.opacity),
    };
  } catch {
    return AERO_VIEW_DEFAULT; // unreadable / corrupt JSON -> default view
  }
}

export function writeAeroViewPref(s: AeroViewState, store: StorageLike | null = storage()): boolean {
  if (!store) return false;
  try {
    store.setItem(AERO_VIEW_PREF_KEY, JSON.stringify({ view: s.view, opacity: s.opacity }));
    return true;
  } catch {
    return false; // quota / blocked storage: the choice lasts this visit only
  }
}

// ── MapLibre builders ───────────────────────────────────────────────────────

export const AERO_SOURCE_ID = "aero-chart";
export const AERO_LAYER_ID = "aero-chart";
export const AERO_ATTRIBUTION = "FAA Aeronautical Information Services charts (public domain) · NOT FOR NAVIGATION";

export function aeroSourceSpec(meta: AeroChartMeta): {
  type: "raster"; tiles: string[]; tileSize: number; minzoom: number; maxzoom: number; attribution: string;
} {
  return {
    type: "raster", tiles: [meta.tiles], tileSize: 256,
    minzoom: meta.minzoom, maxzoom: meta.maxzoom, attribution: AERO_ATTRIBUTION,
  };
}

export function aeroPaint(opacity: number): Record<string, unknown> {
  return baseRasterPaint({ "raster-opacity": clampOpacity(opacity) / 100 });
}

/** The id of the style layer directly ABOVE `anchorId`, i.e. the beforeId
 *  that puts a new layer immediately on top of the base imagery and under
 *  every data layer (aircraft, curtains, overlays). Undefined = anchor is
 *  the top layer (or absent): append. */
export function beforeIdAbove(layerIds: readonly string[], anchorId: string): string | undefined {
  const i = layerIds.indexOf(anchorId);
  if (i < 0) return undefined;
  return layerIds[i + 1];
}

// ── badge / legend text ─────────────────────────────────────────────────────

function fmtDay(iso: string | null): string {
  if (!iso) return "";
  const d = new Date(`${iso}T00:00:00Z`);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" });
}

export interface AeroBadge {
  title: string;
  edition: string;
  tone: "ok" | "warn" | "unknown";
  coverage: string;
}

export function aeroBadge(meta: AeroChartMeta, nowMs: number = Date.now()): AeroBadge {
  const coverage = `shown at zoom ${meta.minzoom}–${meta.maxzoom} where FAA publishes this chart; satellite elsewhere`;
  if (!meta.edition || !meta.effective || !meta.expires) {
    return { title: meta.label, edition: "edition unverified — FAA service metadata unreachable", tone: "unknown", coverage };
  }
  const range = `${fmtDay(meta.effective)} – ${fmtDay(meta.expires)}`;
  const today = new Date(nowMs).toISOString().slice(0, 10);
  const expired = meta.expired || today >= meta.expires;
  if (expired || meta.behindCurrentCycle) {
    return {
      title: meta.label,
      edition: `edition ${range} · ${expired ? "expired" : "superseded"} — FAA's newer cycle not yet on its tile service`,
      tone: "warn", coverage,
    };
  }
  return { title: meta.label, edition: `edition ${range}`, tone: "ok", coverage };
}

export const AERO_NOT_FOR_NAV = "Not for navigation";
