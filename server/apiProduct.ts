/**
 * apiProduct.ts — the /api/v1 data-product foundation (throughput/API
 * directive 2026-07-04). Everything buildable PRE-REVENUE: versioned read
 * endpoints over the datacore archives, API-key auth scaffolding, per-key
 * rate limits, and usage metering from day one. The last mile (key sales,
 * billing, pricing enablement) waits for the human's go on the
 * MONETIZATION READINESS CHECKLIST (wishlist.md) — nothing here charges,
 * bills, or gates payment.
 *
 * Pre-revenue key issuance: keys are ENV-SEEDED only (API_PRODUCT_KEYS =
 * "key:label:tier,key2:label2:tier2"). No signup flow exists on purpose —
 * issuance binds to billing later, per the checklist.
 *
 * LICENSE MARKS: every response envelope names the license of what it
 * carries (the resell-vs-display audit in wishlist.md) — ODbL share-alike
 * for aircraft-derived data, conditional for AIS-derived, public-domain
 * for US-gov streams. Endpoints over gated SIGNALS (tank-fill, entity
 * timelines) do not exist until their ladder gates pass.
 *
 * Pure module: no express, no db imports (the auth.ts import hangs the
 * test runner — standing rule); storage/metering writers injected or
 * fs-appended under the archive base like every other stream.
 */
import fs from "fs";
import path from "path";
import crypto from "crypto";
import { archiveBaseDir } from "./datacoreArchive";

export type ApiTier = "dev" | "pro" | "enterprise";
export interface ApiKeyInfo { label: string; tier: ApiTier }

/** Per-tier limits — {perMinute, perDay}. Enterprise is contract-shaped
 *  later; scaffolding keeps it finite so a leaked key can't melt the box. */
export const TIER_LIMITS: Record<ApiTier, { perMinute: number; perDay: number }> = {
  dev: { perMinute: 60, perDay: 10_000 },
  pro: { perMinute: 600, perDay: 100_000 },
  enterprise: { perMinute: 3_000, perDay: 1_000_000 },
};

export function parseApiKeys(env: NodeJS.ProcessEnv = process.env): Map<string, ApiKeyInfo> {
  const out = new Map<string, ApiKeyInfo>();
  for (const part of (env.API_PRODUCT_KEYS || "").split(",")) {
    const [key, label, tier] = part.trim().split(":");
    if (!key || key.length < 12) continue; // refuse trivially guessable keys
    out.set(key, { label: label || "unlabeled", tier: (["dev", "pro", "enterprise"].includes(tier) ? tier : "dev") as ApiTier });
  }
  return out;
}

/** Sliding-window limiter (minute + day windows per key). Pure/testable:
 *  caller supplies now. */
export function makeRateLimiter() {
  const hits = new Map<string, number[]>();
  return {
    allow(key: string, tier: ApiTier, now = Date.now()): { ok: boolean; retryAfterSec?: number } {
      const lim = TIER_LIMITS[tier];
      const arr = (hits.get(key) || []).filter((t) => now - t < 86_400_000);
      const lastMin = arr.filter((t) => now - t < 60_000);
      if (lastMin.length >= lim.perMinute) return { ok: false, retryAfterSec: 60 };
      if (arr.length >= lim.perDay) return { ok: false, retryAfterSec: 3600 };
      arr.push(now);
      hits.set(key, arr);
      return { ok: true };
    },
    size() { return hits.size; },
  };
}

/** Keys never land raw in logs/archives — hash for metering identity. */
export function keyId(key: string): string {
  return crypto.createHash("sha256").update(key).digest("hex").slice(0, 12);
}

/** Usage metering: day-JSONL under the archive base (a stream like any
 *  other — manifested in datacore/manifests/apiusage.json). */
export function meterUsage(rec: { key: string; endpoint: string; status: number; tier: ApiTier },
                           baseDir?: string, nowMs?: number): void {
  const base = baseDir || archiveBaseDir();
  const dir = path.join(base, "apiusage");
  const now = nowMs ?? Date.now();
  try {
    fs.mkdirSync(dir, { recursive: true });
    fs.appendFileSync(path.join(dir, `${new Date(now).toISOString().slice(0, 10)}.jsonl`),
      JSON.stringify({ t: Math.floor(now / 1000), k: keyId(rec.key), e: rec.endpoint, s: rec.status, ti: rec.tier }) + "\n");
  } catch {}
}

/** The resell-vs-display audit, applied per endpoint (wishlist checklist
 *  item 2). Every v1 response carries its mark — honesty travels with the
 *  data. */
export const LICENSE_MARKS: Record<string, { license: string; attribution: string; resell: "ok" | "share-alike" | "conditional" }> = {
  "tracks/aircraft": {
    license: "ODbL 1.0 share-alike (adsb.lol-derived database) + non-commercial fallback sources until the monetization switch",
    attribution: "adsb.lol + airplanes.live + adsb.fi",
    resell: "share-alike",
  },
  "tracks/vessels": {
    license: "aisstream.io terms (redistribution CONDITIONAL — re-verify at monetization switch)",
    attribution: "aisstream.io",
    resell: "conditional",
  },
  "tracks/trains": {
    license: "CC BY 4.0 (Digitraffic) + NLOD (Entur)",
    attribution: "Digitraffic Finland + Entur Norway",
    resell: "ok",
  },
  "stats/portdwell": {
    license: "derived from AIS positions — inherits aisstream conditionality",
    attribution: "VolTradeAI datacore over aisstream.io",
    resell: "conditional",
  },
  "stats/shadow": {
    license: "derived from AIS positions — inherits aisstream conditionality",
    attribution: "VolTradeAI datacore over aisstream.io",
    resell: "conditional",
  },
  "stats/archive": {
    license: "VolTradeAI operational metadata",
    attribution: "VolTradeAI datacore",
    resell: "ok",
  },
};

/** Self-documenting endpoint reference — /developers renders this; gated
 *  items are listed as coming so the docs stay honest about what exists. */
export function apiMeta() {
  return {
    version: "v1",
    auth: "x-api-key header (or ?api_key=). Keys are invite-only during the preview — join the waitlist on /developers.",
    endpoints: [
      { path: "/api/v1/tracks/:kind/:id", params: "kind=aircraft|vessels|trains; id=icao24|MMSI|train id; ?hours<=168", desc: "Recent position track from our own archive (recording since 2026-07-03)." },
      { path: "/api/v1/stats/portdwell", params: "-", desc: "Per-port dwell statistics (completed calls, in-port-now, medians, 3x-median anomaly flags) over the 9 imagery-verified port geofences." },
      { path: "/api/v1/stats/shadow", params: "-", desc: "Dark-ship RAW statistics: AIS gap events, identity candidates, STS-zone loitering — counts with honest coverage caveats." },
      { path: "/api/v1/stats/archive", params: "-", desc: "Archive growth metadata (streams, samples, days recorded)." },
      { path: "/api/v1/meta", params: "-", desc: "This document." },
    ],
    coming_gated: [
      "entity timelines (Everything Graph v1 — aircraft continuity spine in build)",
      "tank-fill readings (Sentinel-2 — ladder gate 2 not yet passed; experimental readings stay internal)",
    ],
    limits: TIER_LIMITS,
    license_marks: LICENSE_MARKS,
    disclaimer: "Data as-is; not for safety-of-life use; attribution and share-alike marks travel with each response.",
  };
}
