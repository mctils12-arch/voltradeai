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
import { validateRecord } from "./dataQuality";

// DATA QUALITY GATE (research/location_context_engine.md): hourly demand /
// forecast MWh is physically non-negative and far below 5,000,000 MWh/h (US
// total peak ~ 720k MW). A negative or absurd value is a corrupt reading —
// quarantine it at the archive boundary so it can never skew a stress metric.
const DEMAND_BOUNDS = { mwh: { min: 0, max: 5_000_000 } };

export function gridDemandEnabled(env: NodeJS.ProcessEnv = process.env): boolean {
  return Boolean(env.EIA_API_KEY);
}

/** US48 total + the big regional BAs + the Southeast/Northwest/Southwest
 *  region aggregates (all EIA-930 respondent codes on the region-data route).
 *  SE/NW/SW are region rollups, not single BAs — the endpoint accepts them the
 *  same way it accepts the US48 aggregate already in this list. */
export const RESPONDENTS = ["US48", "CISO", "ERCO", "MISO", "PJM", "NYIS", "ISNE", "SWPP", "FPL", "SE", "NW", "SW"];
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

/** Seed-window bound: after the historical backfill the archive holds
 *  ~2,700 day-files (~1.2M rows) — seeding them ALL into the in-memory
 *  set would cost a material slice of the 512MB Node heap on every
 *  restart. The live poll only fetches a trailing 48h window, so dedup
 *  only ever needs recent days; 120d is a huge margin. Re-running the
 *  backfill therefore requires deleting the archive dir with the
 *  done-marker (stated in the marker file), or old rows would dupe. */
export const SEED_WINDOW_DAYS = 120;

export function seedFileInWindow(fileName: string, nowMs: number): boolean {
  const day = Date.parse(fileName.slice(0, 10));
  return Number.isFinite(day) && nowMs - day <= SEED_WINDOW_DAYS * 86400_000;
}

function seedSeen(dir: string): void {
  try {
    const nowMs = Date.now();
    for (const f of fs.readdirSync(dir)) {
      if (!/^\d{4}-\d{2}-\d{2}\.jsonl(\.gz)?$/.test(f)) continue;
      if (!seedFileInWindow(f, nowMs)) continue;
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
  // data-quality gate: quarantine implausible readings before they enter the
  // archive that gridStress reads (quarantine-don't-propagate).
  const valid = obs.filter((o) => validateRecord(o as any, DEMAND_BOUNDS).length === 0);
  const quarantined = obs.length - valid.length;
  if (quarantined > 0) console.warn(`[datacore] griddemand: quarantined ${quarantined} implausible row(s) (data-quality gate)`);
  const fresh = valid.filter((o) => !seenObs.has(obsKey(o)));
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
        // SAME-period DF only — DF is day-ahead, so its newest rows sit in
        // FUTURE hours where aggregates are partial (live probe 2026-07-07:
        // US48 DF T+2h = 74k vs 550k once fully reported). The comparable
        // number is the forecast for the hour the demand reading covers.
        const dfAtPeriod = df.find((r) => r.period === newest.period) || null;
        stats.push({ respondent, latest_period: newest.period,
                     latest_mwh: d.length ? newest.mwh : null,
                     latest_forecast_mwh: dfAtPeriod ? dfAtPeriod.mwh : null,
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

// ── Historical backfill (env-gated OFF — R8 lesson; GRID VISION gate-2 prereq 2) ──
// The region-data endpoint serves history to ~2019 (module header,
// verified live 2026-07-06). Gate-2 wants the deepest honest history
// for the fit/validate split — one-time walk, oldest-first, done-marker.
// Volume: 9 respondents x 2 series x hourly x ~7.5yr ≈ 1.2M rows ≈
// ~110MB plain -> ~20MB once the day-files gz (compressed explicitly at
// the end of the pass, not left for the 3-day cycle).

export const GRID_DEMAND_BACKFILL_START_YEAR = 2019;
const BACKFILL_PAGE = 5000;          // EIA v2 max length per request
const BACKFILL_MAX_PAGES_PER_YEAR = 8; // 2 series x 8784h = 17,568 rows -> 4 pages; guard x2

export function gridDemandBackfillEnabled(env: NodeJS.ProcessEnv = process.env): boolean {
  return env.GRID_DEMAND_BACKFILL === "1"; // opt-in OFF (R8 lesson; Mike flips it)
}

const backfillMarker = (baseDir?: string) => path.join(demandDir(baseDir), "backfill_done.json");

export function backfillUrl(respondent: string, key: string, year: number, offset: number): string {
  return "https://api.eia.gov/v2/electricity/rto/region-data/data/" +
    `?api_key=${encodeURIComponent(key)}` +
    "&frequency=hourly&data%5B0%5D=value" +
    `&facets%5Brespondent%5D%5B%5D=${encodeURIComponent(respondent)}` +
    "&facets%5Btype%5D%5B%5D=D" +
    "&facets%5Btype%5D%5B%5D=DF" +
    `&start=${year}-01-01T00&end=${year}-12-31T23` +
    "&sort%5B0%5D%5Bcolumn%5D=period&sort%5B0%5D%5Bdirection%5D=asc" +
    `&length=${BACKFILL_PAGE}&offset=${offset}`;
}

/** Drop a completed backfill year's keys from the in-memory set —
 *  years are walked once, oldest-first, so their keys are dead weight
 *  (1.2M keys would otherwise sit in the 512MB heap for the process
 *  lifetime). The current live window is never pruned. */
function pruneSeenYear(year: number): void {
  const tag = `|${year}-`;
  for (const k of Array.from(seenObs)) {
    if (k.includes(tag)) seenObs.delete(k);
  }
}

export async function gridDemandBackfillIfEnabled(
  fetchImpl: FetchFn = fetch as any,
  env: NodeJS.ProcessEnv = process.env,
  nowMs?: number,
  baseDir?: string,
  spacingMs = 1500,
  startYear = GRID_DEMAND_BACKFILL_START_YEAR,
): Promise<void> {
  if (!gridDemandBackfillEnabled(env)) return;
  if (fs.existsSync(backfillMarker(baseDir))) return;
  const key = env.EIA_API_KEY || "";
  if (!key) return;
  const now = nowMs ?? Date.now();
  const nowYear = new Date(now).getUTCFullYear();
  const rt = new Date(now).toISOString().slice(0, 10);
  console.log(`[datacore] griddemand backfill: ${startYear}..${nowYear}, ${RESPONDENTS.length} respondents (one-time)`);
  let rows = 0, calls = 0;
  for (let y = startYear; y <= nowYear; y++) {       // oldest-first
    for (const resp of RESPONDENTS) {
      for (let page = 0; page < BACKFILL_MAX_PAGES_PER_YEAR; page++) {
        let got = 0;
        try {
          calls++;
          const r = await fetchImpl(backfillUrl(resp, key, y, page * BACKFILL_PAGE), {
            headers: { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)" },
            signal: AbortSignal.timeout(60000) as any,
          });
          if (!r.ok) {
            console.error(`[datacore] griddemand backfill ${resp} ${y} p${page} -> ${r.status}`);
            break; // next respondent; dedup makes any re-run safe
          }
          const obs = parseDemand(JSON.parse(await r.text()), rt);
          got = obs.length;
          rows += archiveDemand(obs, baseDir);
        } catch (e: any) {
          console.error(`[datacore] griddemand backfill ${resp} ${y}:`, e?.message || e);
          break;
        }
        if (spacingMs > 0) await sleep(spacingMs);
        if (got < BACKFILL_PAGE) break;             // year exhausted for this respondent
      }
    }
    if (y < nowYear) pruneSeenYear(y);              // heap hygiene; live window untouched
  }
  // compress immediately — don't leave ~110MB plain for the 3-day cycle
  const gzd = gzipOldDemandDays(baseDir, now);
  try {
    fs.writeFileSync(backfillMarker(baseDir), JSON.stringify({
      done_rt: new Date(now).toISOString(), rows_archived: rows, calls,
      start_year: startYear, gz_files: gzd,
      note: "one-time pass; to re-run, delete this marker AND the day-files " +
            "(dedup only seeds the last " + SEED_WINDOW_DAYS + " days — old rows would dupe)",
    }));
  } catch {}
  console.log(`[datacore] griddemand backfill done: ${rows} rows, ${calls} calls, ${gzd} day-files gz`);
}

/** Hourly source with ~1-2h lag — 2h poll (9 spaced calls/cycle) keeps
 *  the archive within ~2h of real time without hammering. Eager boot;
 *  historical backfill (env opt-in, one-time) after current data is up
 *  — the OCC deep-backfill pattern. */
export function bootGridDemandPoll(intervalMs = 2 * 60 * 60_000): void {
  if (polling) return;
  polling = true;
  refreshDemand().then(() => gridDemandBackfillIfEnabled())
    .catch((e) => console.error("[datacore] griddemand boot:", e?.message || e));
  setInterval(() => { refreshDemand(); }, intervalMs).unref?.();
}
