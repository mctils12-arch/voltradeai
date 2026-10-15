// AIRCRAFT INTENT C1 (2026-10-14) — the plain-English status line on the
// aircraft card, built from GET /api/data/aircraft/live/:hex's `intent`
// block (server/aircraftIntent.ts).
//
// HONESTY RULES:
//   - only FRESH values are used (the server serves nothing older than 30 s
//     as current) — a field the aircraft does not send never appears;
//   - "Autopilot off" is said only when mode bits ARE sent without the
//     autopilot bit; an aircraft that never sends mode bits gets no
//     autopilot claim at all;
//   - inferred states ("vectored", "climbing to …") carry an `inferred` flag
//     the card renders as a label — never shown as broadcast fact;
//   - units: altitudes and temperatures go through lib/units.ts; flight
//     levels, knots and hPa are fixed aviation conventions (UNITS
//     PREFERENCE domain list).

import { fmtCelsius, fmtMeters, getUnits, type UnitSystem } from "../units.js";

export type NavMode = "autopilot" | "vnav" | "althold" | "approach" | "lnav" | "tcas";

export interface IntentValuesWire {
  nav_modes?: NavMode[];
  nav_heading?: number;
  nav_altitude_mcp?: number;
  nav_altitude_fms?: number;
  nav_qnh?: number;
  wd?: number;
  ws?: number;
  oat?: number;
  ias?: number;
  mach?: number;
  alt_baro?: number;
  alt_geom?: number;
  baro_rate?: number;
  geom_rate?: number;
  acas_ra?: { advisory: string | null } | null;
  squawk?: string;
  emergency?: string;
}

export interface IntentViewWire {
  values: IntentValuesWire;
  stale?: string[];
  capability?: { mode_bits?: boolean };
  inferred?: {
    vectored?: { active: boolean; reason: string; heading_source: string | null; deviation_deg: number | null } | null;
  };
}

export interface StatusPart {
  key: string;
  text: string;
  /** an inference, not a broadcast value — the card labels it */
  inferred?: boolean;
  /** draws attention (TCAS RA, emergency) */
  alert?: boolean;
}

/** FL for selected altitudes at/above 18,000 ft (the US transition level);
 *  below, the altitude in the user's units. */
export const FL_FROM_FT = 18_000;
const FT_M = 0.3048;
/** within this of the selected altitude = level at it */
export const LEVEL_BAND_FT = 200;
/** vertical rate magnitude that counts as climbing/descending */
export const RATE_FPM = 300;

export function fmtSelectedAlt(ft: number, system: UnitSystem = getUnits()): string {
  return ft >= FL_FROM_FT ? `FL${String(Math.round(ft / 100)).padStart(3, "0")}` : fmtMeters(Math.round(ft / 100) * 100 * FT_M, 0, system);
}

const MODE_LABEL: Record<NavMode, string> = {
  autopilot: "AP", vnav: "VNAV", althold: "ALT HOLD", approach: "APPROACH", lnav: "LNAV", tcas: "TCAS",
};

const isNum = (x: unknown): x is number => typeof x === "number" && Number.isFinite(x);
const deg3 = (d: number) => String(Math.round(((d % 360) + 360) % 360) || 360).padStart(3, "0");

/** Pure: the status parts for one intent view (empty when nothing fresh). */
export function intentStatusParts(view: IntentViewWire | null | undefined, system: UnitSystem = getUnits()): StatusPart[] {
  const v = view?.values;
  if (!v) return [];
  const out: StatusPart[] = [];

  if (v.acas_ra && v.acas_ra.advisory) out.push({ key: "ra", text: `TCAS RA: ${v.acas_ra.advisory}`, alert: true });
  if (v.emergency && v.emergency !== "none") out.push({ key: "em", text: `Emergency: ${v.emergency}`, alert: true });

  if (Array.isArray(v.nav_modes)) {
    const on = v.nav_modes.includes("autopilot");
    const others = v.nav_modes.filter((m) => m !== "autopilot").map((m) => MODE_LABEL[m]).filter(Boolean);
    out.push({ key: "ap", text: on ? `Autopilot on${others.length ? ` · ${others.join(" ")}` : ""}` : `Autopilot off${others.length ? ` · ${others.join(" ")}` : ""}` });
  }

  if (isNum(v.nav_altitude_mcp)) {
    const sel = fmtSelectedAlt(v.nav_altitude_mcp, system);
    const alt = v.alt_baro;
    const rate = isNum(v.baro_rate) ? v.baro_rate : isNum(v.geom_rate) ? v.geom_rate : null;
    if (isNum(alt) && Math.abs(alt - v.nav_altitude_mcp) <= LEVEL_BAND_FT) {
      out.push({ key: "alt", text: `Level at ${sel} (selected)` });
    } else if (isNum(alt) && rate !== null && alt < v.nav_altitude_mcp && rate >= RATE_FPM) {
      out.push({ key: "alt", text: `Climbing to ${sel} (selected)` });
    } else if (isNum(alt) && rate !== null && alt > v.nav_altitude_mcp && rate <= -RATE_FPM) {
      out.push({ key: "alt", text: `Descending to ${sel} (selected)` });
    } else {
      out.push({ key: "alt", text: `Selected altitude ${sel}` });
    }
  }
  if (isNum(v.nav_altitude_fms) && v.nav_altitude_fms !== v.nav_altitude_mcp) {
    out.push({ key: "fms", text: `FMS ${fmtSelectedAlt(v.nav_altitude_fms, system)}` });
  }

  if (isNum(v.nav_heading)) {
    const vec = view?.inferred?.vectored;
    out.push({ key: "hdg", text: `Selected heading ${deg3(v.nav_heading)}°` });
    if (vec?.active) out.push({ key: "vec", text: "Vectored", inferred: true });
  }

  if (isNum(v.nav_qnh)) out.push({ key: "qnh", text: `QNH ${Math.round(v.nav_qnh)} hPa` });
  if (isNum(v.wd) && isNum(v.ws)) out.push({ key: "wind", text: `Wind ${deg3(v.wd)}° ${Math.round(v.ws)} kt` });
  if (isNum(v.oat)) out.push({ key: "oat", text: `OAT ${fmtCelsius(v.oat, 0, system)}` });
  if (isNum(v.mach)) out.push({ key: "mach", text: `M${v.mach.toFixed(2)}` });
  else if (isNum(v.ias)) out.push({ key: "ias", text: `IAS ${Math.round(v.ias)} kt` });
  return out;
}
