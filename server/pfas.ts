/**
 * pfas.ts — EPA UCMR 5 PFAS drinking-water detections
 * (Location Context Engine hazard layer — research/location_context_engine.md;
 * the queued "PFAS (EPA UCMR5/SDWIS)" item, logged as still-missing across
 * at least six prior session entries before this one).
 *
 * Source: datacore/pfas/ucmr5_detections.json, built by
 * scripts/pfas_ucmr5.py from EPA's UCMR 5 national occurrence data file
 * (public domain) joined to EPA's Community Water System Service Areas
 * ArcGIS FeatureServer for a service-area-centroid location per system —
 * see the script's own header for the full source/join/caveat writeup.
 * Static artifact, WHOLE-FILE REBUILD per run (like JODI/eiaweekly) — NOT
 * a live server-side poll like superfund.ts/waterViolators.ts, because EPA
 * republishes this file only every few months and the raw file is 300MB+
 * uncompressed (unsuitable for a live in-request Node fetch+parse; see the
 * script header for why this stayed a session-run build instead).
 *
 * RAW/FACTUAL: a detected concentration is EPA's own published federal
 * monitoring record. NOT a health-risk or MCL-exceedance claim — see
 * `latestPfas().meta.caveat`, echoed verbatim in the API response.
 * predictive:false throughout.
 *
 * DATA QUALITY GATE: re-validated at serve time (coordinates in range, not
 * null-island, required fields present) even though the build script
 * already filters at ingest — defense in depth, matching every other
 * hazard layer's contract (server/dataQuality.ts).
 */
import {
  validateRecord, coordValid, partitionValid, freshnessStatus, layerHealth,
  type LayerHealth, type QualityIssue,
} from "./dataQuality";
import pfasJson from "../datacore/pfas/ucmr5_detections.json";

export interface PfasDetection {
  contaminant: string;
  units: string;
  mrl: number | null;
  max_value: number;
  first_detected: string;
  last_detected: string;
  n_detections: number;
}

export interface PfasSystem {
  pwsid: string;
  name: string;
  state: string | null;
  lat: number;
  lon: number;
  population_served: number | null;
  n_analytes_detected: number;
  detections: PfasDetection[];
}

export interface PfasArtifact {
  source: string;
  source_last_modified: string | null;
  geocode_source: string;
  built_at: string;
  attribution: string;
  predictive: boolean;
  caveat: string;
  totals: Record<string, number>;
  system_count: number;
  systems: any[];
}

const SCHEMA = { pwsid: { required: true }, name: { required: true } };

/** One artifact record -> validated PfasSystem, or the quality issues that
 *  quarantine it. Coordinates checked with the shared null-island guard. */
export function systemFromRecord(r: any): { system?: PfasSystem; issues: QualityIssue[] } {
  const issues = validateRecord(r, SCHEMA);
  if (!coordValid(r?.lat, r?.lon)) issues.push({ field: "coordinates", rule: "coordValid", detail: `${r?.lat},${r?.lon}` });
  if (!Array.isArray(r?.detections) || !r.detections.length) issues.push({ field: "detections", rule: "required" });
  if (issues.length) return { issues };
  return {
    issues: [],
    system: {
      pwsid: String(r.pwsid),
      name: String(r.name),
      state: r.state || null,
      lat: Number(r.lat),
      lon: Number(r.lon),
      population_served: r.population_served == null ? null : Number(r.population_served),
      n_analytes_detected: Number(r.n_analytes_detected) || r.detections.length,
      detections: r.detections,
    },
  };
}

/** Full artifact -> {clean systems, suspect count}. Pure — testable on any
 *  artifact shape, not just the committed one. */
export function parsePfasArtifact(artifact: Pick<PfasArtifact, "systems">): { systems: PfasSystem[]; suspect: number } {
  const part = partitionValid(artifact.systems || [], (r) => systemFromRecord(r).issues);
  const systems = part.clean.map((r) => systemFromRecord(r).system!).filter(Boolean);
  return { systems, suspect: part.suspect.length };
}

const raw = pfasJson as unknown as PfasArtifact;
const FRESH_BUDGET_MS = 120 * 86400_000; // EPA republishes roughly quarterly; generous budget before "stale"

const cached = (() => {
  const part = partitionValid(raw.systems || [], (r) => systemFromRecord(r).issues);
  const systems = part.clean.map((r) => systemFromRecord(r).system!).filter(Boolean);
  const builtAtMs = Date.parse(raw.built_at);
  const health = layerHealth(
    raw.attribution,
    part,
    freshnessStatus(Number.isFinite(builtAtMs) ? builtAtMs : null, FRESH_BUDGET_MS, Date.now()),
    raw.built_at,
  );
  return { systems, health };
})();

export function latestPfas(): { systems: PfasSystem[]; health: LayerHealth; meta: PfasArtifact } {
  return { systems: cached.systems, health: cached.health, meta: raw };
}
