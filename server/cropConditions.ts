/**
 * cropConditions.ts — USDA NASS weekly crop condition ratings
 * (DATACORE MAXIMUS Phase 0b / BUILD ORDER 6 #7, key landed 2026-07-06).
 *
 * Source: NASS QuickStats API (quickstats.nass.usda.gov/api/api_GET) —
 * free key (NASS_API_KEY, now in Railway). Keyless probe 2026-07-06
 * returned {"error":["unauthorized"]}, so the ROW SHAPE here follows
 * the documented QuickStats contract and the censusImports precedent:
 * parse defensively (Value/value keys both tolerated, comma-grouped
 * numbers stripped), log the NASS error body VERBATIM on failure so
 * the next session fixes the query from prod logs instead of guessing.
 * LIVE VERIFICATION PENDING — first check post-deploy.
 *
 * Scope v1: NATIONAL weekly CONDITION ratings for CORN + SOYBEANS
 * (5 classes each: EXCELLENT/GOOD/FAIR/POOR/VERY POOR as separate
 * rows via short_desc). State-level expansion is a param change.
 *
 * NASS terms: no endorsement implication, attribute to NASS, content
 * not modified-and-misattributed. Attribution "USDA National
 * Agricultural Statistics Service (QuickStats)".
 *
 * HYPOTHESIS (gate-locked in the build order — archive + RAW only):
 * week-over-week condition deltas lead grain futures and ag small
 * caps; pairs with the live Drought Monitor stream (belt-weighted
 * ag-stress index is the join). Gate 1 = values vs the published Crop
 * Progress report; gate 2 = condition-delta vs forward futures returns.
 */

import fs from "fs";
import path from "path";
import zlib from "zlib";
import { archiveBaseDir } from "./datacoreArchive";

export function cropConditionsEnabled(env: NodeJS.ProcessEnv = process.env): boolean {
  return Boolean(env.NASS_API_KEY);
}

export const COMMODITIES = ["CORN", "SOYBEANS"];
const BASE_URL = "https://quickstats.nass.usda.gov/api/api_GET/";
const CALL_SPACING_MS = 400;

export interface ConditionObs {
  commodity: string;         // CORN | SOYBEANS
  week_ending: string;       // YYYY-MM-DD
  item: string;              // short_desc verbatim (carries the condition class)
  pct: number | null;        // Value, percent
  rt: string;                // as-seen UTC date
}

const num = (v: any): number | null => {
  if (v == null || v === "") return null;
  const n = parseFloat(String(v).replace(/,/g, ""));
  return Number.isFinite(n) ? n : null;
};

/** QuickStats {data:[...]} -> ConditionObs; rows without a week_ending or
 *  short_desc are dropped; Value may arrive as "Value" or "value". */
export function parseConditions(json: any, commodity: string, rt: string): ConditionObs[] {
  const data = json?.data;
  if (!Array.isArray(data)) return [];
  const out: ConditionObs[] = [];
  for (const r of data) {
    const week = String(r?.week_ending || "").slice(0, 10);
    const item = r?.short_desc;
    if (!/^\d{4}-\d{2}-\d{2}$/.test(week) || !item) continue;
    out.push({
      commodity,
      week_ending: week,
      item: String(item),
      pct: num(r.Value ?? r.value),
      rt,
    });
  }
  return out;
}

// ── Fetch (key never logged; NASS error bodies logged verbatim) ─────────────

type FetchFn = (url: string, init?: any) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export function conditionsUrl(commodity: string, year: number, key: string): string {
  const p = new URLSearchParams({
    key,
    source_desc: "SURVEY",
    sector_desc: "CROPS",
    commodity_desc: commodity,
    statisticcat_desc: "CONDITION",
    agg_level_desc: "NATIONAL",
    year: String(year),
    format: "JSON",
  });
  return `${BASE_URL}?${p.toString()}`;
}

export async function fetchConditions(fetchImpl: FetchFn = fetch as any,
                                      env: NodeJS.ProcessEnv = process.env,
                                      nowMs?: number, spacingMs = CALL_SPACING_MS): Promise<ConditionObs[]> {
  const key = env.NASS_API_KEY || "";
  if (!key) return [];
  const now = nowMs ?? Date.now();
  const rt = new Date(now).toISOString().slice(0, 10);
  const year = new Date(now).getUTCFullYear();
  const out: ConditionObs[] = [];
  for (const commodity of COMMODITIES) {
    try {
      const r = await fetchImpl(conditionsUrl(commodity, year, key), {
        headers: { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)" },
        signal: AbortSignal.timeout(30000) as any,
      });
      const body = await r.text();
      if (!r.ok) {
        // NASS errors are human-readable JSON — log verbatim (minus any
        // echo of the key, which NASS does not include in bodies).
        console.error(`[datacore] cropconditions ${commodity} -> ${r.status}: ${body.slice(0, 300)}`);
      } else {
        out.push(...parseConditions(JSON.parse(body), commodity, rt));
      }
    } catch (e: any) {
      console.error(`[datacore] cropconditions ${commodity}:`, e?.message || e);
    }
    if (spacingMs > 0) await sleep(spacingMs);
  }
  return out;
}

// ── Archive (event-identity dedup commodity|week|item, day-file per fetch) ──

const seenObs = new Set<string>();
let seeded = false;

const obsKey = (o: ConditionObs) => `${o.commodity}|${o.week_ending}|${o.item}`;

function conditionsDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "cropconditions");
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

export function archiveConditions(obs: ConditionObs[], baseDir?: string, nowMs?: number): number {
  if (!obs.length) return 0;
  const dir = conditionsDir(baseDir);
  if (!seeded) { seedSeen(dir); seeded = true; }
  const fresh = obs.filter((o) => !seenObs.has(obsKey(o)));
  if (!fresh.length) return 0;
  try {
    fs.mkdirSync(dir, { recursive: true });
    const day = new Date(nowMs ?? Date.now()).toISOString().slice(0, 10);
    fs.appendFileSync(path.join(dir, `${day}.jsonl`),
                      fresh.map((r) => JSON.stringify(r)).join("\n") + "\n");
    fresh.forEach((o) => seenObs.add(obsKey(o)));
    return fresh.length;
  } catch (e: any) {
    console.error("[datacore] cropconditions archive:", e?.message || e);
    return 0;
  }
}

export function gzipOldConditionDays(baseDir?: string, nowMs?: number): number {
  const dir = conditionsDir(baseDir);
  const now = nowMs ?? Date.now();
  let n = 0;
  try {
    for (const f of fs.readdirSync(dir)) {
      if (!f.endsWith(".jsonl")) continue;
      if (now - Date.parse(f.slice(0, 10)) < 2 * 86400_000) continue;
      const fp = path.join(dir, f);
      fs.writeFileSync(`${fp}.gz`, zlib.gzipSync(fs.readFileSync(fp)));
      fs.unlinkSync(fp);
      n++;
    }
  } catch {}
  return n;
}

// ── Cache + poll ────────────────────────────────────────────────────────────

let cache: { at: number; latest_week: string; rows: ConditionObs[] } | null = null;
let polling = false;

export function latestConditions() {
  return cache;
}

export async function refreshConditions(fetchImpl: FetchFn = fetch as any,
                                        env: NodeJS.ProcessEnv = process.env,
                                        nowMs?: number, baseDir?: string,
                                        spacingMs = CALL_SPACING_MS): Promise<void> {
  try {
    if (!cropConditionsEnabled(env)) return;
    const obs = await fetchConditions(fetchImpl, env, nowMs, spacingMs);
    if (obs.length) {
      archiveConditions(obs, baseDir, nowMs);
      const newest = obs.reduce((mx, o) => (o.week_ending > mx ? o.week_ending : mx), "");
      cache = { at: Date.now(), latest_week: newest,
                rows: obs.filter((o) => o.week_ending === newest) };
    }
    gzipOldConditionDays(baseDir, nowMs);
  } catch (e: any) {
    console.error("[datacore] cropconditions refresh:", e?.message || e);
  }
}

/** Weekly source (Monday afternoon ET in season) — 12h poll makes
 *  off-days a dedup no-op. Eager boot per KNOWN BROKEN #9. */
export function bootCropConditionsPoll(intervalMs = 12 * 60 * 60_000): void {
  if (polling) return;
  polling = true;
  refreshConditions();
  setInterval(() => { refreshConditions(); }, intervalMs).unref?.();
}
