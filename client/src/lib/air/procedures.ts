// PROCEDURES CLIENT — the consumer half of /api/data/procedures/* and
// /api/data/plates/* (server/procedures.ts). Pure module: wire types, the
// abortable fetch helpers, wire validation, and the geometry/text helpers the
// Procedures card row and ProcedureLayer use. No DOM, no map — unit-testable
// with `npx tsx --test`.
//
// HONESTY (repo constitution): a DP/STAR shown as FILED came from the FAA
// flight plan's route text; approaches are only ever SUGGESTED (ATC assigns
// them). The procedure path is FAA CIFP; the plate is a backdrop and is only
// placed on the map when the server's georeference passed its gate. Every
// surface says NOT FOR NAVIGATION.

export interface ChartRef { name: string; code: string; pdf: string; url: string; amdt: string | null; amdtDate: string | null }

export interface ProcSummary {
  id: string;
  kind: "SID" | "STAR" | "IAP";
  name: string;
  runway: string | null;
  typeName: string | null;
  transitions: Array<{ id: string; role: string }>;
  runways: string[];
  charts: ChartRef[];
}

export interface FiledProc extends ProcSummary {
  transition: string | null;
  filedAs: string | null;
  versionMismatch: boolean;
  airport: string;
}

export interface SuggestedApproach { id: string; name: string; runway: string | null; reason: string; charts: ChartRef[] }

export interface CycleInfo { ident: string; effective?: string; expires?: string; from?: string | null; to?: string | null }

export interface FlightProcedures {
  callsign: string;
  planSource: "FILED_FAA" | "CLIENT_PLAN" | "NONE";
  departure: { icao: string; name: string | null } | null;
  arrival: { icao: string; name: string | null } | null;
  filed: { dp: FiledProc | null; star: FiledProc | null; note: string };
  suggestedApproaches: SuggestedApproach[];
  wind: { dirDeg: number | null; speedKt: number; gustKt: number | null; obsTime: string | null; raw: string | null } | null;
  cycle: CycleInfo | null;
  dtppCycle: CycleInfo | null;
  dtppError: string | null;
  approachHonesty: string;
  notForNavigation: string;
}

export type PathProps = Record<string, string | number | boolean | null>;
export interface PathFeature {
  type: "Feature";
  geometry: { type: "LineString"; coordinates: Array<[number, number]> } | { type: "Point"; coordinates: [number, number] };
  properties: PathProps;
}
export interface ProcedurePath {
  type: "FeatureCollection";
  features: PathFeature[];
  meta: {
    airport: string; procedure: string; kind: string; name: string;
    selectedTransition: string | null;
    legs: number; approxLegs: number; unresolvedLegs: number; truncated: boolean;
    bbox: [number, number, number, number] | null;
    nasrCrossCheck?: { checked: number; maxDiffNm: number; disagreeing: string[]; placedFromNasr: string[] } | null;
  };
  cycle?: CycleInfo;
}

export interface PlateGeoref {
  georeferenced: boolean;
  reason: string;
  rmsNm: number | null;
  maxResidualNm: number | null;
  controlPoints: Array<{ fix: string; residualNm: number }>;
  scaleNmPerInch: number | null;
  page: { width: number; height: number; rotate: number };
  planView: { x0: number; y0: number; x1: number; y1: number } | null;
  /** [lon, lat] top-left, top-right, bottom-right, bottom-left */
  corners: Array<[number, number]> | null;
  embeddedGeoPdf: boolean;
  embeddedAgreementNm: number | null;
  method: "geopdf-seeded" | "ransac" | null;
  cycle: string;
  pdf: string;
  chart: string;
  proc: string;
  pdfUrl: string;
}

// ── fetch helpers (every request abortable) ─────────────────────────────────

export const CALLSIGN_RE = /^[A-Z0-9]{2,8}$/;
export const AIRPORT_RE = /^[A-Z0-9]{3,4}$/;

export function flightProceduresUrl(callsign: string, dep: string | null, arr: string | null): string | null {
  const cs = String(callsign || "").trim().toUpperCase();
  if (!CALLSIGN_RE.test(cs)) return null;
  const q = new URLSearchParams();
  if (dep && AIRPORT_RE.test(dep)) q.set("dep", dep);
  if (arr && AIRPORT_RE.test(arr)) q.set("arr", arr);
  const qs = q.toString();
  return `/api/data/procedures/flight/${cs}${qs ? `?${qs}` : ""}`;
}
export const procedurePathUrl = (airport: string, procId: string, transition: string | null) =>
  `/api/data/procedures/${encodeURIComponent(airport)}/${encodeURIComponent(procId)}/path${transition ? `?transition=${encodeURIComponent(transition)}` : ""}`;
export const plateGeorefUrl = (chart: ChartRef, procId: string) => `${chart.url}/georef?proc=${encodeURIComponent(procId)}`;

async function getJson<T>(url: string, signal: AbortSignal, fetchImpl: typeof fetch): Promise<T> {
  const r = await fetchImpl(url, { signal, headers: { Accept: "application/json" } });
  if (!r.ok) {
    let msg = `HTTP ${r.status}`;
    try {
      const j = (await r.json()) as { error?: unknown };
      if (typeof j?.error === "string") msg = j.error;
    } catch (e: unknown) {
      void e; // body was not JSON — the status line is the message
    }
    throw new Error(msg);
  }
  return (await r.json()) as T;
}

export function fetchFlightProcedures(url: string, signal: AbortSignal, fetchImpl: typeof fetch = fetch): Promise<FlightProcedures> {
  return getJson<FlightProcedures>(url, signal, fetchImpl).then((j) => {
    if (!j || typeof j !== "object" || !j.filed || !Array.isArray(j.suggestedApproaches)) throw new Error("malformed procedures response");
    return j;
  });
}
export function fetchProcedurePath(url: string, signal: AbortSignal, fetchImpl: typeof fetch = fetch): Promise<ProcedurePath> {
  return getJson<ProcedurePath>(url, signal, fetchImpl).then((j) => {
    if (!j || j.type !== "FeatureCollection" || !Array.isArray(j.features)) throw new Error("malformed path response");
    return j;
  });
}
export function fetchPlateGeoref(url: string, signal: AbortSignal, fetchImpl: typeof fetch = fetch): Promise<PlateGeoref> {
  return getJson<PlateGeoref>(url, signal, fetchImpl).then((j) => {
    if (!j || typeof j.georeferenced !== "boolean") throw new Error("malformed georef response");
    return j;
  });
}

// ── geometry for the map ────────────────────────────────────────────────────

/** Law IV cap for the drawn procedure (legs + fix points). */
export const PROC_MAX_FEATURES = 1500;

const finitePair = (p: unknown): p is [number, number] =>
  Array.isArray(p) && p.length >= 2 && Number.isFinite(p[0]) && Number.isFinite(p[1]) && Math.abs(p[0] as number) <= 180 && Math.abs(p[1] as number) <= 90;

export interface PathLayers {
  legs: { type: "FeatureCollection"; features: PathFeature[] };
  fixes: { type: "FeatureCollection"; features: PathFeature[] };
  dropped: number;
}

/** Split a path response into the line source and the fix-symbol source;
 *  invalid coordinates are dropped (never guessed); capped at `max`. */
export function pathToLayers(path: ProcedurePath | null, max = PROC_MAX_FEATURES): PathLayers {
  const legs: PathFeature[] = [];
  const fixes: PathFeature[] = [];
  let dropped = 0;
  for (const f of path?.features ?? []) {
    if (legs.length + fixes.length >= max) { dropped++; continue; }
    if (f.geometry.type === "LineString") {
      const cs = f.geometry.coordinates.filter(finitePair);
      if (cs.length < 2) { dropped++; continue; }
      legs.push({ ...f, geometry: { type: "LineString", coordinates: cs } });
    } else if (f.geometry.type === "Point" && finitePair(f.geometry.coordinates)) {
      fixes.push(f);
    } else {
      dropped++;
    }
  }
  return { legs: { type: "FeatureCollection", features: legs }, fixes: { type: "FeatureCollection", features: fixes }, dropped };
}

/** Map symbol for a fix (registry shape name, SYMBOLS NOT DOTS): the kind
 *  comes from the CIFP section the server reported. */
export function fixSymbol(p: PathProps): string {
  if (p.role === "FAF") return "vt-fix-faf";
  if (p.fixKind === "navaid") return "vt-fix-nav";
  return "vt-fix-wpt";
}

/** The plate overlay is placed only for a passed georeference with 4 sane
 *  corners spanning a plausible plan view (< 3° either way). */
export function plateCornersUsable(g: PlateGeoref | null): g is PlateGeoref & { corners: Array<[number, number]>; planView: NonNullable<PlateGeoref["planView"]> } {
  if (!g || !g.georeferenced || !g.planView || !Array.isArray(g.corners) || g.corners.length !== 4) return false;
  if (!g.corners.every(finitePair)) return false;
  const lons = g.corners.map((c) => c[0]), lats = g.corners.map((c) => c[1]);
  return Math.max(...lons) - Math.min(...lons) < 3 && Math.max(...lats) - Math.min(...lats) < 3;
}

/** Canvas pixel rect of the plan-view crop (PDF points, origin bottom-left)
 *  on a page rendered at `scale` px/pt (canvas origin top-left). */
export function cropPixelRect(pv: { x0: number; y0: number; x1: number; y1: number }, pageHeightPt: number, scale: number) {
  return {
    x: Math.max(0, Math.floor(pv.x0 * scale)),
    y: Math.max(0, Math.floor((pageHeightPt - pv.y1) * scale)),
    w: Math.max(1, Math.round((pv.x1 - pv.x0) * scale)),
    h: Math.max(1, Math.round((pv.y1 - pv.y0) * scale)),
  };
}

/** Render scale for the overlay crop: longest side of the crop at most
 *  `maxPx` (Law IV raster budget), never below 1 px/pt. */
export function overlayScale(pv: { x0: number; y0: number; x1: number; y1: number }, maxPx: number): number {
  const longest = Math.max(pv.x1 - pv.x0, pv.y1 - pv.y0);
  return longest > 0 ? Math.max(1, maxPx / longest) : 1;
}

// ── display strings ─────────────────────────────────────────────────────────

/** "CIFP 2609 · valid to Oct 1" — the cycle is the procedure data's age. */
export function cycleBadge(c: CycleInfo | null | undefined): string {
  if (!c?.ident) return "cycle unknown";
  const exp = c.expires ? new Date(c.expires) : null;
  const to = exp && Number.isFinite(exp.getTime())
    ? ` · valid to ${exp.toLocaleString("en-US", { month: "short", day: "numeric", timeZone: "UTC" })}`
    : "";
  return `CIFP ${c.ident}${to}`;
}

export function filedLabel(kind: "DP" | "STAR", p: FiledProc): string {
  const tr = p.transition ? ` · ${p.transition} transition` : "";
  const v = p.versionMismatch && p.filedAs ? ` (filed ${p.filedAs})` : "";
  return `${kind} ${p.name}${tr}${v}`;
}

export function windText(w: FlightProcedures["wind"]): string | null {
  if (!w) return null;
  const dir = w.dirDeg == null ? "VRB" : String(Math.round(w.dirDeg)).padStart(3, "0");
  return `wind ${dir}@${Math.round(w.speedKt)}${w.gustKt ? `G${Math.round(w.gustKt)}` : ""} kt`;
}
