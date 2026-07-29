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
  graph: {
    license: "derived from EDGAR Form 4 filings, entity_map, and our own AIS position archive — inherits aisstream conditionality via calls_at edges",
    attribution: "VolTradeAI datacore (SEC EDGAR, GEM ownership, aisstream.io)",
    resell: "conditional",
  },
  "stats/plant-operations": {
    license: "U.S. EPA Clean Air Markets Division (CAMD) unit-level CEMS reporting — public domain (US federal government work)",
    attribution: "U.S. EPA Clean Air Markets Division (CAMD)",
    resell: "ok",
  },
  "stats/secftd": {
    license: "U.S. SEC CNS fails-to-deliver, half-month files — public domain (US federal government work)",
    attribution: "U.S. Securities and Exchange Commission (CNS fails-to-deliver)",
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
      { path: "/api/v1/stats/portdwell", params: "-", desc: "Per-port dwell statistics (completed calls, in-port-now, medians, 3x-median anomaly flags) over the 9 imagery-verified port geofences.", preview: "/api/data/portdwell" },
      { path: "/api/v1/stats/shadow", params: "-", desc: "Dark-ship RAW statistics: AIS gap events, identity candidates, STS-zone loitering — counts with honest coverage caveats.", preview: "/api/data/shadowstats" },
      { path: "/api/v1/stats/archive", params: "-", desc: "Archive growth metadata (streams, samples, days recorded).", preview: "/api/data/archive/stats" },
      { path: "/api/v1/graph", params: "?entity=<ticker|MMSI|CIK|facility id>&hops<=3 (omit entity for counts-only)", desc: "Everything Graph v1 — Form 4 insiders, entity_map operator->ticker, and AIS port-call edges, joined into one node/edge graph. RAW (asserts filed relationships with provenance; no predictive claim).", preview: "/api/data/graph" },
      { path: "/api/v1/stats/plant-operations", params: "-", desc: "Per-facility power-plant utilization ground truth (sum grossLoad MW-days, sum operating hours) from EPA's own unit-level CEMS reporting, TX pilot scope, quarterly cadence. RAW, no predictive claim — public-domain US federal data, resell ok.", preview: "/api/data/plant-operations" },
      { path: "/api/v1/stats/secftd", params: "-", desc: "SEC CNS fails-to-deliver leaderboard: newest settlement date's top fail balances (>=100k share floor, stated). A level, not a daily flow, published on a 2.5-4.5 week SEC lag. RAW, no predictive claim — public-domain US federal data, resell ok.", preview: "/api/data/ftd" },
      { path: "/api/v1/meta", params: "-", desc: "This document.", preview: "/api/v1/meta" },
    ],
    coming_gated: [
      "tank-fill readings (Sentinel-2 — ladder gate 2 not yet passed; experimental readings stay internal)",
    ],
    agent_tools: "/api/v1/agent-tools",
    limits: TIER_LIMITS,
    license_marks: LICENSE_MARKS,
    disclaimer: "Data as-is; not for safety-of-life use; attribution and share-alike marks travel with each response.",
  };
}

/** Agent tool spec — the LIVE API rendered as function-calling tool
 *  definitions so a developer can hand VolTradeAI's verified physical-world
 *  data straight to an AI agent (Anthropic tool use, OpenAI functions, or an
 *  MCP server). Derived from the SAME live endpoint set as apiMeta(), so
 *  gated signals can never leak in; each tool names the license_marks key(s)
 *  of what it returns, so provenance and freshness travel into the agent's
 *  context, not just the raw number. Public — it is documentation, not data;
 *  the calls themselves still require an x-api-key. This is the "ground-truth
 *  layer for AI agents" surface: an agent grounded here answers from observed,
 *  archived measurement instead of model-generated plausibility. */
export function agentToolSpec(baseUrl = "https://voltradeai.com") {
  const tools = [
    {
      name: "voltrade_get_track",
      description: "Recent position track for one aircraft, vessel, or train from VolTradeAI's own continuously-recorded archive. Returns observed, timestamped positions — ground truth, not a prediction.",
      input_schema: {
        type: "object",
        properties: {
          kind: { type: "string", enum: ["aircraft", "vessels", "trains"], description: "Asset class." },
          id: { type: "string", description: "icao24 (aircraft), MMSI (vessel), or train id." },
          hours: { type: "integer", minimum: 1, maximum: 168, default: 24, description: "Lookback window in hours (max 168)." },
        },
        required: ["kind", "id"],
      },
      endpoint: "GET /api/v1/tracks/{kind}/{id}?hours={hours}",
      returns_provenance: ["tracks/aircraft", "tracks/vessels", "tracks/trains"],
    },
    {
      name: "voltrade_port_dwell_stats",
      description: "Per-port dwell statistics over 9 imagery-verified port geofences: completed calls, ships in-port now, median dwell, and 3x-median anomaly flags. RAW overlay — descriptive, not a trading signal.",
      input_schema: { type: "object", properties: {}, required: [] },
      endpoint: "GET /api/v1/stats/portdwell",
      returns_provenance: ["stats/portdwell"],
    },
    {
      name: "voltrade_shadow_fleet_stats",
      description: "Dark-ship RAW statistics: AIS gap events, identity candidates, and STS-zone loitering counts, with honest coverage caveats. RAW overlay — not a signal.",
      input_schema: { type: "object", properties: {}, required: [] },
      endpoint: "GET /api/v1/stats/shadow",
      returns_provenance: ["stats/shadow"],
    },
    {
      name: "voltrade_archive_stats",
      description: "Archive growth metadata: streams recorded, sample counts, and days of history — how much verified physical-economy data the platform holds.",
      input_schema: { type: "object", properties: {}, required: [] },
      endpoint: "GET /api/v1/stats/archive",
      returns_provenance: ["stats/archive"],
    },
    {
      name: "voltrade_get_graph",
      description: "Everything Graph v1 — Form 4 insider filings, entity_map operator->ticker joins, and AIS port-call edges, joined into one node/edge graph. Omit entity for counts-only; pass an entity to get its neighborhood. RAW overlay — asserts filed relationships with provenance, no predictive claim.",
      input_schema: {
        type: "object",
        properties: {
          entity: { type: "string", description: "Optional: ticker, MMSI, CIK, or facility id. Omit for graph-wide counts only." },
          hops: { type: "integer", minimum: 0, maximum: 3, default: 1, description: "Neighborhood radius when entity is given." },
        },
        required: [],
      },
      endpoint: "GET /api/v1/graph?entity={entity}&hops={hops}",
      returns_provenance: ["graph"],
    },
    {
      name: "voltrade_plant_operations_stats",
      description: "Per-facility power-plant utilization ground truth (summed gross load MW-days, summed operating hours) from the U.S. EPA's own unit-level Continuous Emissions Monitoring (CEMS) reporting, TX pilot scope, quarterly cadence. Public-domain US federal data, freely resellable. RAW overlay — direct plant-utilization ground truth, not a trading signal.",
      input_schema: { type: "object", properties: {}, required: [] },
      endpoint: "GET /api/v1/stats/plant-operations",
      returns_provenance: ["stats/plant-operations"],
    },
    {
      name: "voltrade_secftd_stats",
      description: "SEC CNS fails-to-deliver leaderboard for the newest published settlement date: top fail balances by share quantity (>=100k share floor, stated), from the SEC's own half-month CNS files. A fail BALANCE (level), not a daily flow, published on a 2.5-4.5 week SEC lag. Public-domain US federal data, freely resellable. RAW overlay — a crowded/settlement-stress indicator, not a standalone trading signal.",
      input_schema: { type: "object", properties: {}, required: [] },
      endpoint: "GET /api/v1/stats/secftd",
      returns_provenance: ["stats/secftd"],
    },
  ];
  return {
    version: "v1",
    format: "JSON-Schema tool definitions — drop-in for Anthropic tool use, OpenAI function calling, or an MCP server.",
    base_url: baseUrl,
    auth: "Send x-api-key on every call (invite-only during the preview — join the waitlist on /developers). This spec is public; the data behind each tool requires a key.",
    ground_truth_note: "Every tool returns observed, archived measurements carrying provenance and a generated_at timestamp — built to ground AI agents in what is physically true rather than model-generated plausibility.",
    tools,
    license_marks: LICENSE_MARKS,
    excluded_gated: apiMeta().coming_gated,
    disclaimer: apiMeta().disclaimer,
  };
}
