/**
 * qualityDashboard.ts — MAP V2 ROADMAP R6(b) DATA-QUALITY dashboard
 * (research/open_questions.md: "feed freshness + per-provider status,
 * archive growth, verification coverage"). Charter directive 2026-07-04:
 * "dashboards from monitoring we already emit, no new collection."
 *
 * NOT the same module as server/dataQuality.ts (the per-record ingest
 * validation gate for the Location Context Engine) — this one is pure
 * read-side aggregation for a /data dashboard panel.
 *
 * Every number here is a pure join over data three other modules already
 * compute — zero new pollers, zero new network calls:
 *   - streams health -> streamsInventory.ts (per-stream freshness, already
 *     served at /api/data/streams)
 *   - archive growth -> datacoreArchive.ts's archiveStats() (already served
 *     at /api/data/archive/stats)
 *   - verification   -> datacore/sites/strategic_sites.json's own
 *     "ALL 16 imagery-verified" doc claim (ports are the category:"port"
 *     subset of that same coordinate set) and datacore/powerplants/
 *     us_power_plants.json's verified_count/count (top-100-by-MW
 *     imagery-verified, per imagery_verified.json's audit artifact)
 * Counts are read from the source files at call time, never hardcoded —
 * they grow as the underlying registries grow (same discipline as the
 * landing-page hero counters).
 */
import sitesJson from "../datacore/sites/strategic_sites.json";
import plantsJson from "../datacore/powerplants/us_power_plants.json";
import type { StreamInventoryEntry } from "./streamsInventory";

export interface StreamHealthSummary {
  total: number;
  live: number;
  recent: number;
  stale: number;
  no_data: number;
}

export interface ArchiveKindSummary {
  kind: string;
  files: number;
  bytes: number;
}

export interface ArchiveSummary {
  total_bytes: number;
  total_files: number;
  kinds: ArchiveKindSummary[]; // sorted largest-first
}

export interface VerificationCategory {
  verified: number;
  total: number;
  method: string;
}

export interface VerificationCoverage {
  sites: VerificationCategory;
  ports: VerificationCategory;
  plants: VerificationCategory;
}

export function buildStreamHealthSummary(
  streams: readonly Pick<StreamInventoryEntry, "health">[],
): StreamHealthSummary {
  const out: StreamHealthSummary = { total: streams.length, live: 0, recent: 0, stale: 0, no_data: 0 };
  for (const s of streams) {
    if (s.health === "live") out.live++;
    else if (s.health === "recent") out.recent++;
    else if (s.health === "stale") out.stale++;
    else out.no_data++;
  }
  return out;
}

export function buildArchiveSummary(
  stats: { kinds: Record<string, { files: number; bytes: number }>; totalBytes: number },
): ArchiveSummary {
  const kinds = Object.entries(stats.kinds)
    .map(([kind, v]) => ({ kind, files: v.files, bytes: v.bytes }))
    .sort((a, b) => b.bytes - a.bytes);
  const total_files = kinds.reduce((s, k) => s + k.files, 0);
  return { total_bytes: stats.totalBytes, total_files, kinds };
}

export function verificationCoverage(): VerificationCoverage {
  const sites: any[] = (sitesJson as any).sites || [];
  const ports = sites.filter((s) => s.category === "port");
  const plants = plantsJson as any;
  return {
    // strategic_sites.json's own _doc: "ALL 16 coordinates imagery-verified
    // 2026-07-03 ... scripts/site_verify.py renders each coordinate on Esri
    // World Imagery with a crosshair" — a blanket claim over every entry,
    // not a per-row flag, so verified === total here by construction.
    sites: {
      verified: sites.length,
      total: sites.length,
      method: "Esri World Imagery crosshair check, scripts/site_verify.py (2026-07-03)",
    },
    // Ports are the category:"port" subset of that same coordinate set —
    // inherits the same blanket verification, not a separate claim.
    ports: {
      verified: ports.length,
      total: ports.length,
      method: "subset of the imagery-verified strategic sites (category = port)",
    },
    plants: {
      verified: typeof plants.verified_count === "number" ? plants.verified_count : 0,
      total: typeof plants.count === "number" ? plants.count : 0,
      method: "top-100-by-MW plants imagery-verified against Esri World Imagery (datacore/powerplants/imagery_verified.json); remainder carry EIA-860/GPPD coordinates, unverified",
    },
  };
}
