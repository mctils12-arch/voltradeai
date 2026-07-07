/**
 * gridDemand.ts — EIA-930 hourly electric grid demand
 * (DATACORE MAXIMUS Phase 0 / BUILD ORDER 6 #6, key landed 2026-07-06).
 *
 * Source: EIA v2 API, electricity/rto/region-data — free key
 * (EIA_API_KEY, now in Railway), US government public domain. Shape
 * verified live 2026-07-06 with DEMO_KEY: response.data rows
 * {period "YYYY-MM-DDTHH", respondent, type "D", value (string, MWh)}.
 * Bracketed params MUST be URL-encoded. Hourly cadence, ~1-2h lag,
 * API history to 2019.
 *
 * Key-gated on the fredMacro/censusImports pattern: activates wherever
 * the key lands, honest enabled:false on the route without it.
 *
 * Scope v1: demand (type D) for US48 + 8 major balancing authorities —
 * a bounded 9-call sweep. Respondent list is a curated constant;
 * expansion is a one-line change.
 *
 * v2 (2026-07-07, GRID VISION gate-2 prereq 1): day-ahead demand
 * forecast (type DF) rides the SAME call via a second type facet
 * (length doubled to keep the 48h window across two series). Archived
 * rows carry an explicit type; LEGACY lines without one are demand
 * (the facet forced D before this change) and dedup as such —
 * forecast strain = realized D minus DF for the same hour.
 *
 * HYPOTHESIS (gate-locked in the build order — archive + RAW only):
 * weather-adjusted regional demand residuals (join: our CPC degree-days
 * archive) nowcast industrial activity; joins the power-plants layer
 * and the Phase-2 grid layer. Gate 1 = cross-check US48 daily sum vs
 * EIA's own Grid Monitor; gate 2 = demand residual vs industrial-sector
 * returns.
 */

import fs from "fs";
import path from "path";
import zlib from "zlib";
import { archiveBaseDir } from "./datacoreArchive";

export function gridDemandEnabled(env: NodeJS.ProcessEnv = process.env): boolean {
  return Boolean(env.EIA_API_KEY);
}

/** US48 total + the big regional BAs (EIA-930 respondent codes). */
export const RESPONDENTS = ["US48", "CISO", "ERCO", "MISO", "PJM", "NYIS", "ISNE", "SWPP", "FPL"];
/** Trailing window per fetch — covers lag + restarts without bulk. */
export const HOURS_PER_FETCH = 48;
const CALL_SPACING_MS = 300;

export interface DemandObs {
  period: string;            // "YYYY-MM-DDTHH" (UTC hour as published)
  respondent: string;        // BA code
  type?: "D" | "DF";         // demand | day-ahead forecast; absent = D (legacy)
  mwh: number | null;        // megawatthours (demand or forecast per type)
  rt: string;                // as-seen UTC date
}

const num = (v: any): number | null => {
  if (v == null || v === "") return null;
  const n = typeof v === "number" ? v : parseFloat(String(v));
  return Number.isFinite(n) ? n : null;
};

/** EIA v2 envelope -> DemandObs (types D and DF, defensive). */
export function parseDemand(json: any, rt: string): DemandObs[] {
  const data = json?.response?.data;
  if (!Array.isArray(data)) return [];
  const out: DemandObs[] = [];
  for (const r of data) {
    const period = r?.period;
    const respondent = r?.respondent;
    if (typeof period !== "string" || !/^\d{4}-\d{2}-\d{2}T\d{2}$/.test(period)) continue;
    if (!respondent) continue;
    const type = r.type == null ? "D" : r.type; // facet forced D pre-v2
    if (type !== "D" && type !== "DF") continue;
    out.push({ period, respondent: String(respondent), type, mwh: num(r.value), rt });
  }
  return out;
}

// ── Fetch (key never logged; brackets encoded) ──────────────────────────────

type FetchFn = (url: string, init?: any) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export function demandUrl(respondent: string, key: string): string {
  return "https://api.eia.gov/v2/electricity/rto/region-data/data/" +
    `?api_key=${encodeURIComponent(key)}` +
    "&frequency=hourly&data%5B0%5D=value" +
    `&facets%5Brespondent%5D%5B%5D=${encodeURIComponent(respondent)}` +
    "&facets%5Btype%5D%5B%5D=D" +
    "&facets%5Btype%5D%5B%5D=DF" + // day-ahead forecast rides the same call
    "&sort%5B0%5D%5Bcolumn%5D=period&sort%5B0%5D%5Bdirection%5D=desc" +
    `&length=${HOURS_PER_FETCH * 2}`; // two series x the same 48h window
}

export async function fetchDemand(fetchImpl: FetchFn = fetch as any,
                                  env: NodeJS.ProcessEnv = process.env,
                                  nowMs?: number, spacingMs = CALL_SPACING_MS): Promise<DemandObs[]> {
  const key = env.EIA_API_KEY || "";
  if (!key) return [];
  const rt = new Date(nowMs ?? Date.now()).toISOString().slice(0, 10);
  const out: DemandObs[] = [];
  for (const resp of RESPONDENTS) {
    try {
      const r = await fetchImpl(demandUrl(resp, key), {
        headers: { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)" },
        signal: AbortSignal.timeout(30000) as any,
      });
      if (!r.ok) {
        console.error(`[datacore] griddemand ${resp} -> ${r.status}`);
      } else {
        out.push(...parseDemand(JSON.parse(await r.text()), rt));
      }
    } catch (e: any) {
      console.error(`[datacore] griddemand ${resp}:`, e?.message || e);
    }
    if (spacingMs > 0) await sleep(spacingMs);
  }
  return out;
}

// ── Archive (event-identity dedup respondent|period|type, day-file per UTC day) ──

const seenObs = new Set<string>();
let seeded = false;

// legacy archived lines carry no type field — they are demand rows
// (the pre-v2 facet forced D), so they hash identically to new D rows
const obsKey = (o: DemandObs) => `${o.respondent}|${o.period}|${o.type || "D"}`;

function demandDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "griddemand");
}

function seedSeen(dir: string): void {
  try {
    for (const f of fs.readdirSync(dir)) {
      if (!/^\d{4}-\d{2}-\d{2}\.jsonl(\.gz)?$/.test(f)) continue;
      const fp = path.join(dir, f);
      let text: string;
      try {
        text = f.endsWith(".gz")
          ? zlib.gunzipSync(fs.readFileSync(fp)).toString("utf8")
          : fs.readFileSync(fp, "utf8");
      } catch { continue; }
      for (const line of text.split("\n")) {
        if (!line) continue;
        try { seenObs.add(obsKey(JSON.parse(line))); } catch {}
      }
    }
  } catch {}
}

/** Appends UNSEEN hourly observations to day-files keyed by the OBSERVATION
 *  day (period date), not the fetch day — hours land where they belong. */
export function archiveDemand(obs: DemandObs[], baseDir?: string): number {
  if (!obs.length) return 0;
  const dir = demandDir(baseDir);
  if (!seeded) { seedSeen(dir); seeded = true; }
  const fresh = obs.filter((o) => !seenObs.has(obsKey(o)));
  if (!fresh.length) return 0;
  try {
    fs.mkdirSync(dir, { recursive: true });
    const byDay = new Map<string, DemandObs[]>();
    for (const o of fresh) {
      const day = o.period.slice(0, 10);
      if (!byDay.has(day)) byDay.set(day, []);
      byDay.get(day)!.push(o);
    }
    byDay.forEach((rows, day) => {
      fs.appendFileSync(path.join(dir, `${day}.jsonl`),
                        rows.map((r) => JSON.stringify(r)).join("\n") + "\n");
    });
    fresh.forEach((o) => seenObs.add(obsKey(o)));
    return fresh.length;
  } catch (e: any) {
    console.error("[datacore] griddemand archive:", e?.message || e);
    return 0;
  }
}

export function gzipOldDemandDays(baseDir?: string, nowMs?: number): number {
  const dir = demandDir(baseDir);
  const now = nowMs ?? Date.now();
  let n = 0;
  try {
    for (const f of fs.readdirSync(dir)) {
      if (!f.endsWith(".jsonl")) continue;
      // hours for a day keep arriving up to the lag window — gz after 3d
      if (now - Date.parse(f.slice(0, 10)) < 3 * 86400_000) continue;
      const fp = path.join(dir, f);
      fs.writeFileSync(`${fp}.gz`, zlib.gzipSync(fs.readFileSync(fp)));
      fs.unlinkSync(fp);
      n++;
    }
  } catch {}
  return n;
}

// ── Cache + poll ────────────────────────────────────────────────────────────

export interface RespondentStat {
  respondent: string;
  latest_period: string;
  latest_mwh: number | null;
  latest_forecast_mwh: number | null;  // newest DF value; null until DF flows
  hours_in_window: number;             // demand (D) rows only — meaning unchanged
}

let cache: { at: number; stats: RespondentStat[] } | null = null;
let polling = false;

export function latestDemand() {
  return cache;
}

export async function refreshDemand(fetchImpl: FetchFn = fetch as any,
                                    env: NodeJS.ProcessEnv = process.env,
                                    nowMs?: number, baseDir?: string,
                                    spacingMs = CALL_SPACING_MS): Promise<void> {
  try {
    if (!gridDemandEnabled(env)) return;
    const obs = await fetchDemand(fetchImpl, env, nowMs, spacingMs);
    if (obs.length) {
      archiveDemand(obs, baseDir);
      const byResp = new Map<string, DemandObs[]>();
      for (const o of obs) {
        if (!byResp.has(o.respondent)) byResp.set(o.respondent, []);
        byResp.get(o.respondent)!.push(o);
      }
      const stats: RespondentStat[] = [];
      byResp.forEach((rows, respondent) => {
        const d = rows.filter((r) => (r.type || "D") === "D");
        const df = rows.filter((r) => r.type === "DF");
        const newest = d.length
          ? d.reduce((mx, r) => (r.period > mx.period ? r : mx), d[0])
          : rows.reduce((mx, r) => (r.period > mx.period ? r : mx), rows[0]);
        const newestDf = df.length
          ? df.reduce((mx, r) => (r.period > mx.period ? r : mx), df[0])
          : null;
        stats.push({ respondent, latest_period: newest.period,
                     latest_mwh: d.length ? newest.mwh : null,
                     latest_forecast_mwh: newestDf ? newestDf.mwh : null,
                     hours_in_window: d.length });
      });
      stats.sort((a, b) => a.respondent.localeCompare(b.respondent));
      cache = { at: Date.now(), stats };
    }
    gzipOldDemandDays(baseDir, nowMs);
  } catch (e: any) {
    console.error("[datacore] griddemand refresh:", e?.message || e);
  }
}

/** Hourly source with ~1-2h lag — 2h poll (9 spaced calls/cycle) keeps
 *  the archive within ~2h of real time without hammering. Eager boot. */
export function bootGridDemandPoll(intervalMs = 2 * 60 * 60_000): void {
  if (polling) return;
  polling = true;
  refreshDemand();
  setInterval(() => { refreshDemand(); }, intervalMs).unref?.();
}
