// CLOSE-APPROACH VIEW helpers — display formatting and the live separation
// readout for the Time Machine's close-approach list and connectors
// (FLIGHT PROGRAM 2026-09-28; server/closeApproach.ts is the detector).
//
// UNITS PREFERENCE: horizontal separation is shown in nautical miles — the
// separation standard's own unit, the same domain convention as knots —
// WITH the user's unit system alongside via fmtKm; vertical through
// fmtMeters (ft | m). Pure; no DOM.

import { fmtKm, fmtMeters, type UnitSystem, getUnits } from "../units.ts";
import { distMeters } from "./trackModel.ts";

export type Confidence = "high" | "medium" | "low";

/** the server's close-approach entry (server/closeApproach.ts CloseApproach) */
export interface CloseApproachEntry {
  a: string;
  b: string;
  ca?: string;
  cb?: string;
  t: number; // epoch ms
  horizNm: number;
  vertFt: number;
  confidence: Confidence;
  basis: string;
  lat: number;
  lon: number;
  altAFt?: number;
  altBFt?: number;
}

export const CA_HORIZ_NM = 5;
export const CA_VERT_FT = 1000;
const NM_KM = 1.852;
const FT_M = 0.3048;

/** "2.1 nm (2.4 mi)" | "2.1 nm (3.9 km)" */
export function fmtHorizNm(nm: number, system: UnitSystem = getUnits()): string {
  if (!Number.isFinite(nm)) return "no data";
  return `${nm.toFixed(1)} nm (${fmtKm(nm * NM_KM, 1, system)})`;
}

/** "400 ft" | "122 m" */
export function fmtVertFt(ft: number | null, system: UnitSystem = getUnits()): string {
  if (ft == null || !Number.isFinite(ft)) return "altitude unknown";
  return fmtMeters(ft * FT_M, 0, system);
}

/** "2.1 nm (2.4 mi) · 400 ft" */
export function fmtSeparation(nm: number, ft: number | null, system: UnitSystem = getUnits()): string {
  return `${fmtHorizNm(nm, system)} · ${fmtVertFt(ft, system)}`;
}

export function confidenceText(c: Confidence): string {
  return c === "high" ? "high confidence — both aircraft had a fix within 20 s"
    : c === "medium" ? "medium confidence — nearest fixes within 60 s"
    : "low confidence — nearest fixes within 90 s";
}

/** display name for one side of a pair */
export function sideLabel(hex: string, callsign?: string): string {
  const cs = (callsign || "").trim();
  return cs ? cs : hex.toUpperCase();
}

/** stable id for a pair entry (list keys, focus identity) */
export function approachKey(e: Pick<CloseApproachEntry, "a" | "b" | "t">): string {
  return `${e.a}|${e.b}|${e.t}`;
}

/** live separation between two interpolated heads (display readout only —
 *  the server's horizNm/vertFt, computed on full-fidelity fixes, stays the
 *  authoritative minimum) */
export function liveSeparation(
  a: { lat: number; lon: number; altM: number },
  b: { lat: number; lon: number; altM: number },
): { horizNm: number; vertFt: number | null } {
  const horizNm = distMeters(a.lat, a.lon, b.lat, b.lon) / (NM_KM * 1000);
  const vertFt = Number.isFinite(a.altM) && Number.isFinite(b.altM) ? Math.abs(a.altM - b.altM) / FT_M : null;
  return { horizNm, vertFt };
}

/** inside the en-route minima right now (vertical must be KNOWN) */
export function withinMinima(sep: { horizNm: number; vertFt: number | null }): boolean {
  return sep.horizNm < CA_HORIZ_NM && sep.vertFt != null && sep.vertFt < CA_VERT_FT;
}

/** "14:02:31 UTC" */
export function fmtUtcTime(ms: number): string {
  return new Date(ms).toISOString().slice(11, 19) + " UTC";
}
