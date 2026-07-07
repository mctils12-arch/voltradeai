/**
 * analyst.ts — ANALYST CONSOLE W6 (SERVER half): the LLM analyst tool-loop
 * (research/console_charter.md W6). A key-gated, server-side Anthropic
 * Messages-API tool-use loop that answers natural-language questions
 * STRICTLY from the platform's own data tools (archive/cache reads only —
 * never a network fetch at request time), and drives the /data map via a
 * typed map_command protocol. The client chat pane is a separate later PR;
 * this module returns the full structured envelope that pane will render:
 * answer + tool_trace + map_commands + source chips + usage + budget.
 *
 * Key gating (euLoad awaiting_key pattern): without ANTHROPIC_API_KEY the
 * route answers { awaiting_key: true, ... } honestly, HTTP 200 — the
 * analyst ACTIVATES ON KEY DETECT the moment the key lands in Railway.
 * The key is never logged, echoed, or included in any response or archived
 * record; every outgoing envelope is scrubbed against it as a last line of
 * defense (transport errors can quote request headers).
 *
 * Charter honesty rules enforced here, not just prompted:
 *  - the model sees ONLY tool results (system prompt forbids memory facts;
 *    tools are our own caches/archives with their own honesty notes intact);
 *  - "I don't have that" is a valid answer; awaiting_key / warming_up /
 *    budget_exhausted are first-class honest states, never silent degrades;
 *  - grid_stress carries predictive:false through to the model verbatim —
 *    gate-2-failed data may never be re-labeled by an LLM summary;
 *  - budgets (8 tool calls, 4 LLM round-trips, ANALYST_DAILY_TOKENS/day)
 *    are stated in the envelope and force-close loudly, never quietly.
 *
 * Cost: model = ANALYST_MODEL (default claude-haiku-4-5, cheapest tier per
 * the charter), max ANALYST_MAX_TOKENS output tokens per turn (default
 * 1024). Daily spend is tracked in a day-keyed JSON state file on the
 * Railway volume (/data/voltrade, /tmp local fallback — bot.ts precedent)
 * so a redeploy never resets the budget.
 *
 * Conversation state v1 is SINGLE-QUESTION: request = { question } and the
 * loop starts fresh each call. Multi-turn history is a later PR.
 */

import fs from "fs";
import path from "path";
import { queryWindowCached, SUPPORTED_LAYERS } from "./queryEngine";
import { satellitesResponse, GROUPS } from "./satellites";
import { latestAlerts } from "./nwsAlerts";
import { latestGridStress, gridStressEnabled } from "./gridStress";
import { latestLoad, euLoadEnabled } from "./euLoad";
import { siteTimelineCached, type SiteRef } from "./siteTimeline";
import datacoreLayers from "../datacore/layers.json";
import datacoreSites from "../datacore/sites/strategic_sites.json";

// ── configuration ────────────────────────────────────────────────────────────

export const ANALYST_MODEL_DEFAULT = "claude-haiku-4-5"; // cheapest tier — charter demands minimal cost
export const ANALYST_MAX_TOKENS_DEFAULT = 1024;          // per-turn output cap
export const MAX_TOOL_CALLS = 8;                         // per question, then force-close (stated)
export const MAX_ROUNDTRIPS = 4;                         // LLM calls per question, then force-close (stated)
export const MAX_CONCURRENT = 2;                         // LLM calls are slow; public-ish surface
export const DAILY_TOKENS_DEFAULT = 500_000;             // input+output tokens per UTC day
export const QUESTION_MAX_CHARS = 2000;

export function analystEnabled(env: NodeJS.ProcessEnv = process.env): boolean {
  return Boolean(env.ANTHROPIC_API_KEY);
}

function modelFor(env: NodeJS.ProcessEnv): string {
  return env.ANALYST_MODEL || ANALYST_MODEL_DEFAULT;
}

function maxTokensFor(env: NodeJS.ProcessEnv): number {
  const n = parseInt(String(env.ANALYST_MAX_TOKENS || ""), 10);
  return Number.isFinite(n) && n > 0 ? n : ANALYST_MAX_TOKENS_DEFAULT;
}

export function dailyTokenLimit(env: NodeJS.ProcessEnv = process.env): number {
  const n = parseInt(String(env.ANALYST_DAILY_TOKENS || ""), 10);
  return Number.isFinite(n) && n > 0 ? n : DAILY_TOKENS_DEFAULT;
}

// ── system prompt (ships verbatim) ───────────────────────────────────────────

export const SYSTEM_PROMPT =
  "You are the VolTradeAI analyst: you answer questions about the physical economy " +
  "(facilities, fleets, flows, grids, filings) STRICTLY from the platform's own data tools. " +
  "Binding rules:\n" +
  "1. TOOLS ARE THE ONLY GROUND TRUTH. Every fact, number, entity, and date in your answer " +
  "must come from a tool result in this conversation. Never answer from memory or general " +
  "knowledge, and never invent an entity, number, or date.\n" +
  "2. If the tools cannot answer the question, say exactly that — e.g. \"the platform's data " +
  "cannot answer this\" — and name what was missing. Refusing to guess is a correct answer.\n" +
  "3. CITE EVERY FIGURE: state which tool and parameters produced it, in parentheses, e.g. " +
  "(query_window lat=35.94 lon=-96.74 days=7) or (nws_alerts). Include the data's timestamp " +
  "or freshness when the tool result carries one.\n" +
  "4. NO PREDICTIONS AND NO TRADING ADVICE. Descriptive analysis only: what the data shows, " +
  "not what will happen or what anyone should buy/sell/hold. Data labeled descriptive or " +
  "predictive:false must never be presented as predictive.\n" +
  "5. REPORT DATA STATE HONESTLY: if a tool result says warming_up, awaiting a key, stale, " +
  "truncated, or enabled:false, say so instead of papering over it.\n" +
  "6. You may drive the user's map with the map_command tool (fly_to a location you are " +
  "discussing, toggle_layer for a layer you are discussing). Map commands render for the " +
  "user; they return no data.\n" +
  "7. BE CONCISE: a few sentences, or a short list for multi-part answers.";

// ── tool schemas (written for a small model: short, explicit units, examples) ─

export const VALID_MAP_LAYERS: string[] =
  ((datacoreLayers as any).layers || []).map((l: any) => String(l.id));

const SITES: SiteRef[] = ((datacoreSites as any).sites || [])
  .filter((s: any) => s.lat != null && s.lon != null)
  .map((s: any) => ({ id: s.id, name: s.name, lat: s.lat, lon: s.lon }));

export const DATA_TOOL_NAMES = [
  "query_window", "satellites", "nws_alerts", "grid_stress", "eu_load", "site_timeline",
] as const;
export type DataToolName = (typeof DATA_TOOL_NAMES)[number];

export const ANALYST_TOOLS: Array<{ name: string; description: string; input_schema: any }> = [
  {
    name: "query_window",
    description:
      "Cross-layer archive query: everything the platform's own archives recorded near a point. " +
      "Returns per-layer archived-point counts by day, top entities, recent events, freshness and " +
      "provenance. Layers: aircraft, vessels, trains (positions), fires, alerts, gauges (events). " +
      "Example: lat=35.94, lon=-96.74, radiusKm=50, days=7 for Cushing OK. Counts are lower bounds " +
      "under adaptive archive thinning.",
    input_schema: {
      type: "object",
      properties: {
        lat: { type: "number", description: "Latitude in decimal degrees, -90 to 90. Example: 35.94" },
        lon: { type: "number", description: "Longitude in decimal degrees, -180 to 180. Example: -96.74" },
        radiusKm: { type: "number", description: "Search radius in kilometers. Default 50, max 250." },
        days: { type: "integer", description: "Days back from now. Default 7, max 7 (raw retention)." },
        layers: {
          type: "array", items: { type: "string", enum: [...SUPPORTED_LAYERS] },
          description: "Layers to query. Omit for all six.",
        },
      },
      required: ["lat", "lon"],
    },
  },
  {
    name: "satellites",
    description:
      "Latest satellite orbital element sets per group, from the CelesTrak GP cache (6h poll). " +
      "group='all' returns per-group counts and newest epoch; a specific group returns element " +
      "sets (NORAD id, name, epoch). No positions — mean elements only.",
    input_schema: {
      type: "object",
      properties: {
        group: {
          type: "string", enum: ["all", ...GROUPS],
          description: "Satellite group. 'all' for the summary. Example: 'starlink'.",
        },
      },
      required: [],
    },
  },
  {
    name: "nws_alerts",
    description:
      "Currently active US severe-weather alerts (National Weather Service cache, ~10 min poll): " +
      "event type, severity, area, end time. Polygon-carrying alerts only; zone-only alerts are " +
      "counted in zone_only. No input.",
    input_schema: { type: "object", properties: {}, required: [] },
  },
  {
    name: "grid_stress",
    description:
      "TX/ERCOT grid-stress reading (descriptive composite of demand percentile, day-ahead " +
      "forecast strain, weather degree-days). IMPORTANT: predictive:false — this reading FAILED " +
      "ladder gate 2 and is descriptive only; never present it as predictive. No input.",
    input_schema: { type: "object", properties: {}, required: [] },
  },
  {
    name: "eu_load",
    description:
      "Latest realised electricity load per EU bidding zone (ENTSO-E cache, 2h poll): latest MW, " +
      "48h window min/max/mean MW, per-zone data issues. Zones: DE_LU, FR, ES, IT, NL, PL, BE, SE. " +
      "No input.",
    input_schema: { type: "object", properties: {}, required: [] },
  },
  {
    name: "site_timeline",
    description:
      "7-day event timeline for one tracked strategic site (tank farms, steel mills, ports): " +
      "nearby NWS alerts, fire detections, river-gauge readings, plus daily aircraft/vessel " +
      "archived-point density within 50 km. Call with an unknown or missing siteId to get the " +
      "list of valid site ids. Example siteId: 'cushing_hub'.",
    input_schema: {
      type: "object",
      properties: {
        siteId: { type: "string", description: "Site id, e.g. 'cushing_hub'. Omit to list known sites." },
      },
      required: [],
    },
  },
  {
    name: "map_command",
    description:
      "Drive the user's map. Returns no data — it renders for the user. command='fly_to' needs " +
      "lat (-90..90), lon (-180..180), optional zoom (1..20). command='toggle_layer' needs layer " +
      "(a /data layer id, e.g. 'aircraft', 'alerts', 'sites') and on (true/false).",
    input_schema: {
      type: "object",
      properties: {
        command: { type: "string", enum: ["fly_to", "toggle_layer"], description: "Which map action." },
        lat: { type: "number", description: "fly_to: latitude, -90..90." },
        lon: { type: "number", description: "fly_to: longitude, -180..180." },
        zoom: { type: "number", description: "fly_to: optional zoom level, 1..20." },
        layer: { type: "string", description: "toggle_layer: layer id from the /data registry." },
        on: { type: "boolean", description: "toggle_layer: true = show, false = hide." },
      },
      required: ["command"],
    },
  },
];

// ── map_command validation (errors surface to the MODEL as tool_result, never a throw) ──

export interface MapCommand {
  command: "fly_to" | "toggle_layer";
  lat?: number; lon?: number; zoom?: number;
  layer?: string; on?: boolean;
}

export function validateMapCommand(input: any): { ok: true; cmd: MapCommand } | { ok: false; error: string } {
  const command = input?.command;
  if (command === "fly_to") {
    const lat = Number(input?.lat), lon = Number(input?.lon);
    if (!Number.isFinite(lat) || lat < -90 || lat > 90) {
      return { ok: false, error: `fly_to: lat must be a number in [-90, 90], got ${JSON.stringify(input?.lat)}` };
    }
    if (!Number.isFinite(lon) || lon < -180 || lon > 180) {
      return { ok: false, error: `fly_to: lon must be a number in [-180, 180], got ${JSON.stringify(input?.lon)}` };
    }
    const cmd: MapCommand = { command: "fly_to", lat, lon };
    if (input?.zoom != null) {
      const zoom = Number(input.zoom);
      if (!Number.isFinite(zoom) || zoom < 1 || zoom > 20) {
        return { ok: false, error: `fly_to: zoom must be in [1, 20], got ${JSON.stringify(input?.zoom)}` };
      }
      cmd.zoom = zoom;
    }
    return { ok: true, cmd };
  }
  if (command === "toggle_layer") {
    const layer = String(input?.layer ?? "");
    if (!VALID_MAP_LAYERS.includes(layer)) {
      return { ok: false, error: `toggle_layer: unknown layer '${layer}'. Valid layers: ${VALID_MAP_LAYERS.join(", ")}` };
    }
    if (typeof input?.on !== "boolean") {
      return { ok: false, error: "toggle_layer: 'on' must be boolean true/false" };
    }
    return { ok: true, cmd: { command: "toggle_layer", layer, on: input.on } };
  }
  return { ok: false, error: `unknown map command ${JSON.stringify(command)} — use 'fly_to' or 'toggle_layer'` };
}

// ── per-tool slim(): what the MODEL sees. A small model answers worse, not
//    better, from a megabyte dump — arrays are capped, verbose fields dropped,
//    and every truncation is STATED in the tool_result content. ───────────────

export const SLIM_MAX_CHARS = 12_000; // hard serialized ceiling per tool result

export function slim(tool: string, result: any): { payload: any; truncation: string | null } {
  let payload = result;
  const notes: string[] = [];
  try {
    if (tool === "satellites" && Array.isArray(result?.satellites)) {
      const total = result.satellites.length;
      const keep = result.satellites.slice(0, 25).map((r: any) => ({
        NORAD_CAT_ID: r.NORAD_CAT_ID, OBJECT_NAME: r.OBJECT_NAME ?? null, EPOCH: r.EPOCH,
      }));
      if (total > 25) notes.push(`satellites: showing 25 of ${total}`);
      if (total > 0) notes.push("satellites: orbital element fields dropped, kept NORAD_CAT_ID/OBJECT_NAME/EPOCH");
      payload = { ...result, satellites: keep };
    } else if (tool === "nws_alerts" && Array.isArray(result?.alerts)) {
      const total = result.alerts.length;
      const keep = result.alerts.slice(0, 20).map((a: any) => {
        const { rings, ...rest } = a || {};
        return rest;
      });
      if (total > 20) notes.push(`alerts: showing 20 of ${total}`);
      notes.push("alerts: polygon geometry (rings) dropped");
      payload = { ...result, alerts: keep };
    } else if (tool === "query_window" && result?.layers && typeof result.layers === "object") {
      const layers: Record<string, any> = {};
      for (const [name, lr] of Object.entries<any>(result.layers)) {
        const out: any = { ...lr };
        if (Array.isArray(lr?.topEntities) && lr.topEntities.length > 5) {
          out.topEntities = lr.topEntities.slice(0, 5);
          notes.push(`${name}: showing 5 of ${lr.topEntities.length} top entities`);
        }
        if (Array.isArray(lr?.events) && lr.events.length > 10) {
          out.events = lr.events.slice(0, 10);
          notes.push(`${name}: showing 10 of ${lr.events.length} events`);
        }
        layers[name] = out;
      }
      payload = { ...result, layers };
    }
  } catch { /* slimming must never break a tool result — fall through to the hard cap */ }

  // hard serialized ceiling, stated when it bites
  let s: string;
  try { s = JSON.stringify(payload); } catch { s = String(payload); payload = s; }
  if (s.length > SLIM_MAX_CHARS) {
    notes.push(`hard-truncated: serialized result was ${s.length} chars, cap ${SLIM_MAX_CHARS}`);
    payload = s.slice(0, SLIM_MAX_CHARS);
  }
  return { payload, truncation: notes.length ? notes.join("; ") : null };
}

/** tool_result content string: truncation stated FIRST so the model can't miss it. */
export function toolResultContent(tool: string, result: any): string {
  const { payload, truncation } = slim(tool, result);
  const body = typeof payload === "string" ? payload : JSON.stringify(payload);
  return truncation ? `NOTE — result truncated for context (${truncation}).\n${body}` : body;
}

// ── default tool executors (OUR code directly; cache/archive reads only) ─────

type Executor = (input: any) => any | Promise<any>;

const iso = (ms: number) => new Date(ms).toISOString();

export const defaultExecutors: Record<DataToolName, Executor> = {
  query_window: (input) => queryWindowCached({
    lat: Number(input?.lat), lon: Number(input?.lon),
    radiusKm: input?.radiusKm != null ? Number(input.radiusKm) : undefined,
    days: input?.days != null ? Number(input.days) : undefined,
    layers: Array.isArray(input?.layers) ? input.layers.map(String) : undefined,
  }),
  satellites: (input) => {
    const { status, body } = satellitesResponse(input?.group ?? "all");
    if (status !== 200) throw new Error(String(body?.error || `satellites returned ${status}`));
    return body;
  },
  nws_alerts: () => {
    const hit = latestAlerts();
    if (!hit) return { warming_up: true, note: "NWS alert cache not populated yet — first poll pending" };
    return {
      source: "National Weather Service (api.weather.gov), ~10 min poll",
      fetched_at: iso(hit.at), count: hit.display.length, zone_only: hit.zoneOnly,
      note: "polygon-carrying alerts only; zone-coded alerts without geometry counted in zone_only",
      alerts: hit.display,
    };
  },
  grid_stress: () => {
    if (!gridStressEnabled()) {
      return { enabled: false, predictive: false, reason: "EIA_API_KEY not set — grid-stress reading unavailable" };
    }
    const hit = latestGridStress();
    if (!hit) return { predictive: false, warming_up: true, note: "grid-stress cache not populated yet" };
    return {
      source: "VoltradeAI derived composite (EIA-930 ERCOT + NOAA CPC)",
      predictive: false,
      validation_status: "GATE 2 NOT PASSED — descriptive only; never present as predictive",
      computed_at: iso(hit.at), history_depth_days: hit.history_depth_days, reading: hit.reading,
    };
  },
  eu_load: () => {
    if (!euLoadEnabled()) {
      return { enabled: false, reason: "ENTSOE_API_KEY not set — EU load unavailable" };
    }
    const hit = latestLoad();
    if (!hit) return { warming_up: true, note: "EU load cache not populated yet" };
    return {
      source: "ENTSO-E Transparency Platform (actual total load), 2h poll",
      fetched_at: iso(hit.at),
      note: "realised MW per bidding zone, ~1-2h publication lag; zones absent from a sweep listed in issues",
      zones: hit.stats, issues: hit.issues,
    };
  },
  site_timeline: async (input) => {
    const siteId = input?.siteId != null ? String(input.siteId) : null;
    const site = SITES.find((s) => s.id === siteId);
    if (!site) {
      return {
        unknown_site: siteId,
        known_sites: SITES.map((s) => ({ id: s.id, name: s.name })),
        note: "call again with one of these site ids",
      };
    }
    const { timelines, as_of } = await siteTimelineCached(SITES);
    const t = timelines.get(site.id);
    return {
      source: "own archives: NWS alerts + FIRMS fires + USGS gauges + position density",
      as_of, site: site.id, name: site.name, window_days: 7,
      note: "events capped at 12 newest-first; density = archived points within 50 km per day under adaptive sampling; absent days absent, not zero",
      events: t?.events ?? [], density: t?.density ?? {},
    };
  },
};

/** Best-effort freshness for the source chip (null when the tool has none). */
export function freshnessOf(tool: string, result: any): string | null {
  try {
    if (tool === "query_window" && result?.layers) {
      let max: string | null = null;
      for (const lr of Object.values<any>(result.layers)) {
        if (typeof lr?.freshness === "string" && (!max || lr.freshness > max)) max = lr.freshness;
      }
      return max;
    }
    if (tool === "satellites") return result?.freshness?.newest_epoch ?? null;
    if (tool === "nws_alerts" || tool === "eu_load") return result?.fetched_at ?? null;
    if (tool === "grid_stress") return result?.computed_at ?? null;
    if (tool === "site_timeline") return result?.as_of || null;
  } catch {}
  return null;
}

/** Compact one-line summary for the tool_trace (never the payload). */
function summarize(tool: string, result: any): string {
  try {
    if (tool === "query_window" && result?.layers) {
      const parts = Object.entries<any>(result.layers).map(([n, lr]) => `${n}:${lr?.points ?? 0}`);
      return `points ${parts.join(" ")}`;
    }
    if (tool === "satellites") {
      if (Array.isArray(result?.summary)) return `groups: ${result.summary.map((g: any) => `${g.group}=${g.count}`).join(" ")}`;
      return `${result?.group ?? "?"}: ${result?.count ?? 0} satellites`;
    }
    if (tool === "nws_alerts") return result?.warming_up ? "warming up" : `${result?.count ?? 0} active alerts (+${result?.zone_only ?? 0} zone-only)`;
    if (tool === "grid_stress") {
      if (result?.enabled === false) return "unavailable (no key)";
      return result?.warming_up ? "warming up" : `composite pct ${result?.reading?.composite_percentile ?? "null"} on ${result?.reading?.date ?? "?"} (descriptive, predictive:false)`;
    }
    if (tool === "eu_load") {
      if (result?.enabled === false) return "unavailable (no key)";
      return result?.warming_up ? "warming up" : `${(result?.zones || []).length} zones`;
    }
    if (tool === "site_timeline") {
      if (result?.unknown_site !== undefined) return `unknown site — listed ${(result?.known_sites || []).length} valid ids`;
      return `${result?.site}: ${(result?.events || []).length} events, ${Object.keys(result?.density || {}).length} density days`;
    }
  } catch {}
  return "ok";
}

// ── daily token budget (day-keyed JSON state file; /data volume, /tmp fallback) ──

const BUDGET_FILE = "voltrade_analyst_budget.json";

export function analystStateDir(env: NodeJS.ProcessEnv = process.env): string {
  return env.DATA_DIR || (fs.existsSync("/data") ? "/data/voltrade" : "/tmp");
}

interface BudgetState { day: string; used: number }

let memBudget: BudgetState | null = null; // in-memory mirror of the state file

function budgetPath(dir: string): string { return path.join(dir, BUDGET_FILE); }

function readBudget(dir: string): BudgetState | null {
  try {
    const raw = JSON.parse(fs.readFileSync(budgetPath(dir), "utf8"));
    if (typeof raw?.day === "string" && Number.isFinite(raw?.used)) return { day: raw.day, used: raw.used };
  } catch {}
  return null;
}

function budgetFor(dir: string, day: string): BudgetState {
  if (!memBudget) memBudget = readBudget(dir); // survives module state loss via the file
  if (!memBudget || memBudget.day !== day) memBudget = readBudget(dir);
  if (!memBudget || memBudget.day !== day) memBudget = { day, used: 0 }; // day rollover resets
  return memBudget;
}

export function tokensSpentToday(dir: string, nowMs: number): number {
  return budgetFor(dir, iso(nowMs).slice(0, 10)).used;
}

export function recordTokenSpend(dir: string, nowMs: number, tokens: number): void {
  const b = budgetFor(dir, iso(nowMs).slice(0, 10));
  b.used += tokens;
  try {
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(budgetPath(dir), JSON.stringify(b));
  } catch (e: any) {
    console.error("[analyst] budget persist:", String(e?.message || e));
  }
}

/** First instant of the next UTC day — when the budget resets. */
export function nextUtcDay(nowMs: number): string {
  return iso(Date.parse(iso(nowMs).slice(0, 10)) + 86400_000).slice(0, 10) + "T00:00:00Z";
}

// ── key scrubbing (constitution security rule: the key never leaves) ─────────

/** Deep-scrubs any occurrence of the key from a JSON-serializable body. */
export function scrubKey<T>(body: T, key: string | undefined): T {
  if (!key) return body;
  try {
    const s = JSON.stringify(body);
    if (!s.includes(key)) return body;
    return JSON.parse(s.split(key).join("[redacted]"));
  } catch { return body; }
}

// ── transport (injectable; ALL tests use a fake — never a live call in CI) ───

export type AnalystTransport = (body: Record<string, unknown>) => Promise<any>;

export function httpTransport(key: string): AnalystTransport {
  return async (body) => {
    const r = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
      },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(120_000) as any,
    });
    const json: any = await r.json().catch(() => null);
    if (!r.ok) {
      // status + api error type only — never the request (it carries the key header)
      throw new Error(`anthropic http ${r.status}: ${String(json?.error?.message || json?.error?.type || "").slice(0, 200)}`);
    }
    return json;
  };
}

// ── the tool loop ────────────────────────────────────────────────────────────

export interface AnalystOptions {
  transport?: AnalystTransport;
  env?: NodeJS.ProcessEnv;
  nowMs?: number;
  stateDir?: string;
  /** test override for individual data-tool executors */
  executors?: Partial<Record<DataToolName, Executor>>;
}

interface ToolTraceEntry { tool: string; input: any; ok: boolean; summary: string }
interface SourceChip { tool: string; params: any; freshness?: string | null }

let inFlight = 0; // module-level concurrency gate (MAX_CONCURRENT)

/** TEST-ONLY: clears module state (budget mirror + concurrency counter). */
export function _resetAnalystForTests(): void {
  memBudget = null;
  inFlight = 0;
}

async function runTool(
  name: string, input: any, executors: Record<DataToolName, Executor>,
  mapCommands: MapCommand[], sources: SourceChip[],
): Promise<{ ok: boolean; content: string; summary: string }> {
  if (name === "map_command") {
    const v = validateMapCommand(input);
    if (!v.ok) return { ok: false, content: `map_command rejected: ${v.error}`, summary: `rejected: ${v.error.slice(0, 120)}` };
    mapCommands.push(v.cmd);
    const summary = v.cmd.command === "fly_to"
      ? `fly_to ${v.cmd.lat},${v.cmd.lon}${v.cmd.zoom != null ? ` z${v.cmd.zoom}` : ""}`
      : `toggle_layer ${v.cmd.layer} ${v.cmd.on ? "on" : "off"}`;
    return { ok: true, content: "map command accepted and sent to the user's map (no data returned)", summary };
  }
  const exec = executors[name as DataToolName];
  if (!exec) {
    return { ok: false, content: `unknown tool '${name}' — available: ${[...DATA_TOOL_NAMES, "map_command"].join(", ")}`, summary: "unknown tool" };
  }
  try {
    const raw = await exec(input);
    sources.push({ tool: name, params: input ?? {}, freshness: freshnessOf(name, raw) });
    return { ok: true, content: toolResultContent(name, raw), summary: summarize(name, raw) };
  } catch (e: any) {
    const msg = String(e?.message || e).slice(0, 300);
    return { ok: false, content: `tool error: ${msg}`, summary: `error: ${msg.slice(0, 120)}` };
  }
}

/**
 * Answers one natural-language question via the Anthropic tool loop.
 * Returns { status, body } so routes.ts stays a thin shell (satellitesResponse
 * precedent). Never throws; never leaks the key.
 */
export async function analystResponse(question: unknown, opts: AnalystOptions = {}): Promise<{ status: number; body: any }> {
  const env = opts.env ?? process.env;
  const key = env.ANTHROPIC_API_KEY;
  const scrub = <T,>(b: T) => scrubKey(b, key);

  if (!analystEnabled(env)) {
    return {
      status: 200,
      body: {
        awaiting_key: true,
        enabled: false,
        reason: "ANTHROPIC_API_KEY not set (Railway env — see wishlist). The analyst activates automatically the moment the key lands; no data is faked meanwhile.",
      },
    };
  }

  const q = typeof question === "string" ? question.trim() : "";
  if (!q || q.length > QUESTION_MAX_CHARS) {
    return { status: 400, body: { error: `question must be a non-empty string of at most ${QUESTION_MAX_CHARS} characters` } };
  }

  if (inFlight >= MAX_CONCURRENT) {
    return { status: 429, body: { error: `analyst busy — max ${MAX_CONCURRENT} concurrent questions`, retry_after_sec: 10 } };
  }

  const nowMs = opts.nowMs ?? Date.now();
  const stateDir = opts.stateDir ?? analystStateDir(env);
  const limit = dailyTokenLimit(env);
  const spent = tokensSpentToday(stateDir, nowMs);
  if (spent >= limit) {
    return {
      status: 200,
      body: {
        budget_exhausted: true,
        resets: nextUtcDay(nowMs),
        budget: { spent_today: spent, daily_limit: limit },
        note: "daily analyst token budget spent — honest stop, not a silent degrade; resets at the next UTC day",
      },
    };
  }

  const transport = opts.transport ?? httpTransport(key!);
  const executors: Record<DataToolName, Executor> = { ...defaultExecutors, ...(opts.executors || {}) };
  const model = modelFor(env);
  const maxTokens = maxTokensFor(env);

  const toolTrace: ToolTraceEntry[] = [];
  const mapCommands: MapCommand[] = [];
  const sources: SourceChip[] = [];
  let inputTokens = 0, outputTokens = 0, requests = 0, toolCalls = 0;
  let answer = "";
  let closeNote: string | null = null;
  let transportError: string | null = null;

  inFlight++;
  try {
    const messages: any[] = [{ role: "user", content: q }];
    for (let round = 0; round < MAX_ROUNDTRIPS; round++) {
      let resp: any;
      try {
        resp = await transport({ model, max_tokens: maxTokens, system: SYSTEM_PROMPT, messages, tools: ANALYST_TOOLS });
      } catch (e: any) {
        transportError = String(e?.message || e).slice(0, 300);
        break;
      }
      requests++;
      inputTokens += Number(resp?.usage?.input_tokens) || 0;
      outputTokens += Number(resp?.usage?.output_tokens) || 0;
      const content = Array.isArray(resp?.content) ? resp.content : [];
      const text = content.filter((b: any) => b?.type === "text" && typeof b.text === "string")
        .map((b: any) => b.text).join("\n").trim();
      if (text) answer = answer ? `${answer}\n${text}` : text;

      const uses = content.filter((b: any) => b?.type === "tool_use");
      if (resp?.stop_reason !== "tool_use" || !uses.length) break; // end_turn (or anything non-tool) closes the loop

      messages.push({ role: "assistant", content });
      const results: any[] = [];
      for (const u of uses) {
        if (toolCalls >= MAX_TOOL_CALLS) {
          closeNote = `tool-call cap (${MAX_TOOL_CALLS} per question) reached — loop force-closed; the answer reflects only the tools that ran`;
          results.push({
            type: "tool_result", tool_use_id: u.id, is_error: true,
            content: `tool-call budget exhausted (max ${MAX_TOOL_CALLS} per question) — answer NOW from the results you already have and state what remains unqueried.`,
          });
          continue;
        }
        toolCalls++;
        const r = await runTool(String(u.name), u.input, executors, mapCommands, sources);
        toolTrace.push({ tool: String(u.name), input: u.input, ok: r.ok, summary: r.summary });
        results.push({ type: "tool_result", tool_use_id: u.id, content: r.content, ...(r.ok ? {} : { is_error: true }) });
      }
      messages.push({ role: "user", content: results });
      if (round === MAX_ROUNDTRIPS - 1 && !closeNote) {
        closeNote = `LLM round-trip cap (${MAX_ROUNDTRIPS}) reached — loop force-closed; the answer reflects only the rounds that ran`;
      }
    }
  } finally {
    inFlight--;
    if (inputTokens + outputTokens > 0) recordTokenSpend(stateDir, nowMs, inputTokens + outputTokens);
  }

  const spentNow = tokensSpentToday(stateDir, nowMs);

  if (transportError && !answer) {
    return {
      status: 502,
      body: scrub({
        error: `analyst transport error: ${transportError}`,
        tool_trace: toolTrace, map_commands: mapCommands, sources,
        usage: { input_tokens: inputTokens, output_tokens: outputTokens, requests },
        budget: { spent_today: spentNow, daily_limit: limit },
        model,
      }),
    };
  }

  if (!answer) {
    answer = "The analyst loop closed without producing a text answer — no result is being invented in its place.";
    if (!closeNote) closeNote = "model returned no text";
  }

  return {
    status: 200,
    body: scrub({
      answer,
      tool_trace: toolTrace,
      map_commands: mapCommands,
      sources,
      usage: { input_tokens: inputTokens, output_tokens: outputTokens, requests },
      budget: { spent_today: spentNow, daily_limit: limit },
      model,
      ...(closeNote ? { note: closeNote } : {}),
      ...(transportError ? { transport_error: `partial answer — a later round failed: ${transportError}` } : {}),
    }),
  };
}
