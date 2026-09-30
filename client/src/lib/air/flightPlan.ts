// FLIGHT PLAN CLIENT — the consumer half of the shared flight-plan contract
// (FLIGHT PROGRAM 2026-09-28; the server half is
// GET /api/data/aircraft/plan/:hex). Pure module: contract types, the one
// abortable fetch helper, wire-shape validation and the display strings the
// planned-route card row and the destination label render. No DOM, no map —
// unit-testable with `npx tsx --test`.
//
// HONESTY (repo constitution): FILED (an FAA flight plan) and PREDICTED
// (route database / great-circle / our own history) are never blurred —
// every string built here carries which one it is, and a NONE plan draws
// nothing and says so. The wire is validated defensively: a point without
// a finite position is dropped (never guessed), an unknown `source` is
// treated as NONE (never promoted to FILED).

import { isSynthesizedFixName } from "../../../../shared/flightPlanGeometry.js";

export type PlanSource ="FILED_FAA" | "ROUTE_DB_PREDICTED" | "HISTORY_PREDICTED" | "NONE";

export interface PlanAirport {
  icao: string | null;
  iata: string | null;
  name: string | null;
  lat: number;
  lon: number;
  /** field elevation, meters MSL (null = not published). */
  elevM: number | null;
}

export interface PlanPoint {
  lon: number;
  lat: number;
  /** meters MSL; null = no altitude for this point (an honest gap). */
  altM: number | null;
  /** true = the altitude is the server's estimated climb/cruise/descent
   *  profile, not a filed value. */
  altEstimated: boolean;
  name: string | null;
  /** additive (2026-09-30): the segment FROM this point to the next is ATC
   *  vectoring (a connector the aircraft is being flown along near the
   *  destination), NOT part of the filed/predicted route. */
  vectors?: boolean;
}

export type DeviationState = "ON_PLAN" | "OFF_PLAN" | "UNKNOWN";

export interface PlanDeviation {
  state: DeviationState;
  crossTrackNm: number | null;
  since: number | null;
}

export type PlanEventType = "DEVIATION_START" | "DEVIATION_END" | "REPLANNED" | "PLAN_AMENDED";

export interface PlanEvent {
  t: number;
  type: PlanEventType;
  detail: string;
}

export interface FlightPlan {
  hex: string;
  callsign: string | null;
  source: PlanSource;
  label: string;
  origin: PlanAirport | null;
  destination: PlanAirport | null;
  cruiseAltFt: number | null;
  cruiseAltEstimated: boolean;
  /** CURRENT plan, origin → destination order (the re-planned ahead part
   *  when OFF_PLAN). */
  points: PlanPoint[];
  /** the plan before re-planning, when it differs (OFF_PLAN) — drawn as a
   *  faint line so filed-vs-actual stays visible. */
  originalPoints: PlanPoint[] | null;
  deviation: PlanDeviation;
  events: PlanEvent[];
  /** server epoch ms the plan data was fetched upstream. */
  fetchedAt: number | null;
  /** server-reported age of the plan data at response time, seconds. */
  ageSec: number | null;
  honesty: string;
  /** true when the lateral path itself is an estimate (great circle / last
   *  flight); absent on the wire = treated as estimated (never promoted). */
  pathEstimated: boolean;
  /** additive (2026-09-30): the aircraft is inside the destination's terminal
   *  area and off the route — the connector to the route is ATC vectors. */
  terminalVectoring: boolean;
}

/** Periodic re-fetch while a plane is selected. */
export const PLAN_REFRESH_MS = 60_000;
/** Re-fetch IMMEDIATELY once the plane is this far off the drawn plan
 *  (10 nautical miles — the brief's re-plan trigger). */
export const PLAN_DEVIATION_REFETCH_M = 10 * 1852;
/** …but never more often than this (the server may legitimately still be
 *  off-plan too — a deviation must not become a request storm). */
export const PLAN_DEVIATION_MIN_GAP_MS = 15_000;

const SOURCES: readonly PlanSource[] = ["FILED_FAA", "ROUTE_DB_PREDICTED", "HISTORY_PREDICTED", "NONE"];
const DEV_STATES: readonly DeviationState[] = ["ON_PLAN", "OFF_PLAN", "UNKNOWN"];
const EVENT_TYPES: readonly PlanEventType[] = ["DEVIATION_START", "DEVIATION_END", "REPLANNED", "PLAN_AMENDED"];

const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const str = (v: unknown): string | null => (typeof v === "string" && v.trim() ? v.trim() : null);

function normAirport(v: unknown): PlanAirport | null {
  if (!v || typeof v !== "object") return null;
  const o = v as Record<string, unknown>;
  const lat = num(o.lat), lon = num(o.lon);
  if (lat == null || lon == null || Math.abs(lat) > 90 || Math.abs(lon) > 180) return null;
  return { icao: str(o.icao), iata: str(o.iata), name: str(o.name), lat, lon, elevM: num(o.elevM) };
}

function normPoints(v: unknown): PlanPoint[] {
  if (!Array.isArray(v)) return [];
  const out: PlanPoint[] = [];
  for (const p of v) {
    if (!p || typeof p !== "object") continue;
    const o = p as Record<string, unknown>;
    const lat = num(o.lat), lon = num(o.lon);
    if (lat == null || lon == null || Math.abs(lat) > 90 || Math.abs(lon) > 180) continue; // never guessed
    const altM = num(o.altM);
    out.push({ lon, lat, altM, altEstimated: o.altEstimated === true, name: str(o.name), ...(o.vectors === true ? { vectors: true } : {}) });
  }
  return out;
}

/** Validate a wire response into a FlightPlan (null when it is not a plan
 *  object at all). An unknown `source` becomes NONE — never promoted. */
export function normalizePlan(raw: unknown): FlightPlan | null {
  if (!raw || typeof raw !== "object") return null;
  const o = raw as Record<string, unknown>;
  const source: PlanSource = SOURCES.includes(o.source as PlanSource) ? (o.source as PlanSource) : "NONE";
  const dv = (o.deviation && typeof o.deviation === "object" ? o.deviation : {}) as Record<string, unknown>;
  const events: PlanEvent[] = [];
  if (Array.isArray(o.events)) {
    for (const e of o.events) {
      if (!e || typeof e !== "object") continue;
      const eo = e as Record<string, unknown>;
      const t = num(eo.t);
      if (t == null || !EVENT_TYPES.includes(eo.type as PlanEventType)) continue;
      events.push({ t, type: eo.type as PlanEventType, detail: str(eo.detail) ?? "" });
    }
  }
  const points = source === "NONE" ? [] : normPoints(o.points);
  const orig = source === "NONE" || !Array.isArray(o.originalPoints) ? null : normPoints(o.originalPoints);
  return {
    hex: (str(o.hex) ?? "").toLowerCase(),
    callsign: str(o.callsign),
    source,
    label: str(o.label) ?? (source === "NONE" ? "No flight plan available" : ""),
    origin: normAirport(o.origin),
    destination: normAirport(o.destination),
    cruiseAltFt: num(o.cruiseAltFt),
    cruiseAltEstimated: o.cruiseAltEstimated === true,
    points,
    originalPoints: orig && orig.length >= 2 ? orig : null,
    deviation: {
      state: DEV_STATES.includes(dv.state as DeviationState) ? (dv.state as DeviationState) : "UNKNOWN",
      crossTrackNm: num(dv.crossTrackNm),
      since: num(dv.since),
    },
    events,
    fetchedAt: num(o.fetchedAt),
    ageSec: num(o.ageSec),
    honesty: str(o.honesty) ?? "",
    pathEstimated: o.pathEstimated !== false,
    terminalVectoring: source !== "NONE" && o.terminalVectoring === true,
  };
}

/** What the client knows about the plane when it asks for its plan. */
export interface PlanQuery {
  callsign?: string | null;
  lat?: number | null;
  lon?: number | null;
  /** meters MSL (the live feed's native unit). */
  altM?: number | null;
  /** broadcast track, degrees true. */
  trkDeg?: number | null;
}

const M_TO_FT = 3.28084;

/**
 * Query string for the plan request. `alt` is sent in FEET (the aviation
 * convention the response's cruiseAltFt uses) AND `altM` in meters, so the
 * server never has to guess the unit of an unlabeled number. Absent values
 * are omitted rather than sent as empty/NaN.
 */
export function planQueryString(q: PlanQuery): string {
  const p = new URLSearchParams();
  const cs = typeof q.callsign === "string" ? q.callsign.trim().toUpperCase() : "";
  if (cs) p.set("callsign", cs);
  const lat = num(q.lat), lon = num(q.lon);
  if (lat != null && lon != null) {
    p.set("lat", lat.toFixed(5));
    p.set("lon", lon.toFixed(5));
  }
  const alt = num(q.altM);
  if (alt != null) {
    p.set("alt", String(Math.round(alt * M_TO_FT)));
    p.set("altM", String(Math.round(alt)));
  }
  const trk = num(q.trkDeg);
  if (trk != null) p.set("trk", String(Math.round(((trk % 360) + 360) % 360)));
  const s = p.toString();
  return s ? `?${s}` : "";
}

export const HEX_RE = /^[0-9a-f]{6}$/i;

/** The one abortable fetch. Throws on transport/HTTP failure (the caller
 *  shows the designed "unavailable — retrying" state); returns the
 *  validated plan (a NONE plan is a valid answer, not an error). */
export async function fetchFlightPlan(
  hex: string,
  q: PlanQuery,
  signal?: AbortSignal,
  fetchImpl: typeof fetch = fetch,
): Promise<FlightPlan> {
  if (!HEX_RE.test(hex)) throw new Error(`invalid hex ${hex}`);
  const r = await fetchImpl(`/api/data/aircraft/plan/${hex.toLowerCase()}${planQueryString(q)}`, { signal });
  if (!r.ok) throw new Error(`plan HTTP ${r.status}`);
  const plan = normalizePlan(await r.json());
  if (!plan) throw new Error("plan response malformed");
  return plan;
}

/** FILED | PREDICTED | null (NONE). */
export function planKind(source: PlanSource): "FILED" | "PREDICTED" | null {
  if (source === "FILED_FAA") return "FILED";
  if (source === "ROUTE_DB_PREDICTED" || source === "HISTORY_PREDICTED") return "PREDICTED";
  return null;
}

/** True when the plan has something to draw. */
export function planDrawable(plan: FlightPlan | null | undefined): plan is FlightPlan {
  return !!plan && plan.source !== "NONE" && plan.points.length >= 2;
}

export function airportCode(a: PlanAirport | null): string | null {
  if (!a) return null;
  return a.iata || a.icao || null;
}

/** "SFO → SIN" (either end may be unknown: "→ SIN", "SFO →"); null when
 *  neither end is known. */
export function planRouteText(plan: FlightPlan): string | null {
  const o = airportCode(plan.origin);
  const d = airportCode(plan.destination);
  if (!o && !d) return null;
  return `${o ?? "?"} → ${d ?? "?"}`;
}

/** The destination label text: "SFO → SIN · FILED" / "… · PREDICTED".
 *  null for NONE (nothing is drawn, so nothing is labeled). */
export function planDestLabel(plan: FlightPlan | null | undefined): string | null {
  if (!planDrawable(plan)) return null;
  const kind = planKind(plan.source);
  const route = planRouteText(plan);
  return route ? `${route} · ${kind}` : `${kind} ROUTE`;
}

/**
 * Deviation readout for the card: "ON PLAN" | "OFF PLAN by 3.4 mi" |
 * "OFF PLAN" (distance not reported) | "PLAN CONFORMANCE UNKNOWN".
 * The distance goes through the caller's unit formatter (units.ts fmtKm —
 * UNITS PREFERENCE); the wire value is nautical miles.
 */
export function deviationText(
  dev: PlanDeviation,
  fmtDistKm: (km: number, digits?: number) => string,
): string {
  if (dev.state === "ON_PLAN") return "ON PLAN";
  if (dev.state === "OFF_PLAN") {
    return dev.crossTrackNm != null ? `OFF PLAN by ${fmtDistKm(dev.crossTrackNm * 1.852, 1)}` : "OFF PLAN";
  }
  return "PLAN CONFORMANCE UNKNOWN";
}

/** Plan data age now, seconds: the server's age at response time plus the
 *  time since the client received it. null when the server did not say. */
export function planAgeSec(plan: FlightPlan, receivedAtMs: number, nowMs: number): number | null {
  if (plan.ageSec == null) return null;
  return Math.max(0, plan.ageSec + Math.max(0, nowMs - receivedAtMs) / 1000);
}

/** Compact age: "12 s" / "4 min" / "2 h 5 min". */
export function fmtAgeShort(sec: number | null): string {
  if (sec == null || !Number.isFinite(sec)) return "age unknown";
  const s = Math.round(sec);
  if (s < 90) return `${s} s`;
  const m = Math.round(s / 60);
  if (m < 90) return `${m} min`;
  const h = Math.floor(m / 60);
  return `${h} h ${m % 60} min`;
}

/** "ATC vectors — not part of the filed route" (FILED) / "…predicted route"
 *  when the server flagged terminal-area vectoring; null otherwise. Same
 *  wording as the server's honesty sentence (server/flightPlans.ts
 *  vectoringNote). */
export function planVectoringText(plan: FlightPlan | null | undefined): string | null {
  if (!planDrawable(plan) || !plan.terminalVectoring) return null;
  return `ATC vectors — not part of the ${plan.source === "FILED_FAA" ? "filed" : "predicted"} route`;
}

/** May the client draw a live-computed "vectoring" connector for this plan?
 *  Not against a FILED plan whose path is a great-circle stand-in (route
 *  text only): being off a line we never had is not evidence of vectors. */
export function planVectoringAllowed(plan: FlightPlan | null | undefined): boolean {
  return planDrawable(plan) && !(plan.source === "FILED_FAA" && plan.pathEstimated);
}

/** Indices of the plan's labelable NAMED FIXES: a name that is not one the
 *  server synthesized ("present position…", "TOC/TOD/peak (est.)"), and not
 *  the last point (the destination carries its own label). */
export function labelableFixIndices(points: readonly PlanPoint[]): number[] {
  const out: number[] = [];
  for (let i = 0; i < points.length - 1; i++) {
    const nm = points[i].name;
    if (nm && !isSynthesizedFixName(nm)) out.push(i);
  }
  return out;
}

/** True when the plan was just re-planned (the card says so). */
export function wasReplanned(plan: FlightPlan): boolean {
  return plan.events.some((e) => e.type === "REPLANNED");
}

/**
 * Should the plane's cross-track distance from the DRAWN plan trigger an
 * immediate re-fetch? >10 nm off, nothing in flight, and at least
 * PLAN_DEVIATION_MIN_GAP_MS since the last request started.
 */
export function shouldDeviationRefetch(
  crossTrackM: number | null,
  lastFetchStartMs: number,
  nowMs: number,
  inFlight: boolean,
): boolean {
  if (inFlight || crossTrackM == null || !Number.isFinite(crossTrackM)) return false;
  if (crossTrackM <= PLAN_DEVIATION_REFETCH_M) return false;
  return nowMs - lastFetchStartMs >= PLAN_DEVIATION_MIN_GAP_MS;
}

/**
 * Geometry identity of a plan: two responses with the same key draw the
 * same curtain, so the 60s refresh does NOT rebuild (or crossfade) when
 * nothing changed. Positions are keyed at ~1 m, altitudes at 1 m.
 */
export function planGeometryKey(plan: FlightPlan | null): string {
  if (!planDrawable(plan)) return "none";
  const pk = (pts: PlanPoint[] | null) =>
    (pts ?? []).map((p) => `${p.lon.toFixed(5)},${p.lat.toFixed(5)},${p.altM == null ? "-" : Math.round(p.altM)}${p.altEstimated ? "e" : ""}${p.vectors ? "v" : ""}`).join(";");
  return `${plan.source}|${pk(plan.points)}|${pk(plan.originalPoints)}`;
}
