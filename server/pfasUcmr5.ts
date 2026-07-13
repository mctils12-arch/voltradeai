/**
 * pfasUcmr5.ts — EPA UCMR 5 PFAS occurrence data, served from a static
 * artifact (Location Context Engine hazard layer #4 — the queued
 * "PFAS (EPA UCMR5/SDWIS)" item in research/location_context_engine.md).
 *
 * Unlike superfund.ts/waterViolators.ts (live-polled ArcGIS/ECHO
 * services refreshed on a schedule), this layer's source data is a
 * ~300MB EPA bulk file with no lat/lon of its own — geocoding each
 * public water system requires a two-hop join through EPA ECHO/FRS
 * that is expensive to repeat on every server boot. scripts/pfas_ucmr5.py
 * does that work OFFLINE (session-run, not on a schedule — UCMR5's
 * 2023-2025 monitoring window is complete, so this changes only when
 * EPA posts a data update) and writes datacore/pfas/pfas_ucmr5.json,
 * which this module imports directly (same convention dossier.ts uses
 * for datacore/sites/strategic_sites.json — bundled at build time by
 * esbuild, sidestepping the R14 runtime-fs-read packaging pitfall).
 *
 * RAW/FACTUAL layer: each record is EPA's own lab result (contaminant,
 * concentration in µg/L, MRL, collection date) for a system that
 * detected PFAS above the method reporting limit. NOT a risk, safety,
 * or violation claim — EPA's PFAS NPDWR MCLs carry a 2029 compliance
 * deadline that has not arrived; see the artifact's own `caveat` field,
 * surfaced verbatim in the API response.
 */
import pfasArtifact from "../datacore/pfas/pfas_ucmr5.json";
import { coordValid } from "./dataQuality";

export interface PfasDetection {
  contaminant: string;
  result: number;
  mrl: number;
  units: string;
  date: string;
}

export interface PfasSystem {
  pws_id: string;
  name: string;
  state: string;
  region: string;
  lat: number;
  lon: number;
  n_detections: number;
  max_result_ug_l: number;
  detections: PfasDetection[];
}

interface PfasArtifact {
  source: string;
  geocode_source: string;
  attribution: string;
  built_at: string;
  selection: string;
  caveat: string;
  counts: {
    pfas_rows_all_result_signs: number;
    pws_monitored_for_pfas: number;
    pws_with_detection: number;
    pws_geocoded: number;
    pws_missing_geocode: number;
    bad_rows: number;
  };
  systems: PfasSystem[];
}

const artifact = pfasArtifact as unknown as PfasArtifact;

/** Coordinate-guarded view of the committed artifact's systems — a
 *  build-time data error (impossible lat/lon) is never served silently. */
export function pfasSystems(): PfasSystem[] {
  return artifact.systems.filter((s) => coordValid(s.lat, s.lon));
}

export function pfasArtifactMeta() {
  const { source, geocode_source, attribution, built_at, selection, caveat, counts } = artifact;
  return { source, geocode_source, attribution, built_at, selection, caveat, counts };
}
