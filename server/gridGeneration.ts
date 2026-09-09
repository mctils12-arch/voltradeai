/**
 * gridGeneration.ts — EIA-930 hourly net generation by balancing authority
 * and fuel type (electricity/rto/fuel-type-data — the sibling series to
 * gridDemand.ts's region-data; same EIA v2 API, same EIA_API_KEY gate,
 * same US government public-domain licensing).
 *
 * WHY THIS EXISTS (2026-09-09, scheduled-routine [PRODUCT] session):
 * CLAUDE.md's FUSION HYPOTHESES entry (b) ("generation shifts x utility
 * tickers") names its GATE 1 GROUND TRUTH as "EIA-930 totals reconciling
 * to registry capacity within ~5% per region" — that reconciliation needs
 * generation BY FUEL TYPE per region, which gridDemand.ts does not carry
 * (that module's `type` facet is D/DF — demand and demand-forecast only,
 * never generation-by-source). This module is the missing raw ingredient;
 * it does not itself attempt the fusion hypothesis's gate 1 (that needs
 * days of accumulated archive depth across a full diurnal generation
 * cycle before a same-region reconciliation is meaningful — see the
 * signal_ladder.json entry and open_questions.md for the follow-up path).
 *
 * Shape verified live 2026-09-09 (real EIA_API_KEY, not DEMO_KEY):
 * response.data rows {period "YYYY-MM-DDTHH", respondent, respondent-name,
 * fueltype, type-name, value (string, MWh)}. US48 carries ~16 fuel-type
 * rows per hour (BAT/COL/GEO/NG/NUC/OES/OIL/OTH/PS/SNB/SUN/UES/UNK/WAT/
 * WNB/WND); smaller BAs report fewer. Storage fuel types (BAT/OES/PS/UES)
 * legitimately go NEGATIVE (charging draws net power) — confirmed live
 * (UES read -18 and -87 MWh in the same 30-row sample) — so the
 * data-quality bound below is a single wide physical-plausibility range,
 * not a floor-at-zero like gridDemand's (demand itself can never be
 * negative; generation-by-source, with storage in the mix, can).
 *
 * Scope v1: same RESPONDENTS list as gridDemand.ts (US48 + 8 major BAs +
 * SE/NW/SW rollups) for direct comparability with the existing demand
 * series — expansion is a one-line change, same convention.
 *
 * v1 ships WITHOUT historical backfill (gridDemand.ts's backfill was
 * itself a deliberately separate v2 addition, per that module's own
 * header) — deepening the archive is queued as NEXT, not blocking this
 * shipment; the live poll alone is enough to start accumulating the
 * diurnal-cycle depth the fusion gate 1 needs.
 */

import fs from "fs";
import path from "path";
import zlib from "zlib";
import { archiveBaseDir } from "./datacoreArchive";
import { validateRecord } from "./dataQuality";

// DATA QUALITY GATE (research/location_context_engine.md): a single
// respondent x fuel-type hourly reading is physically bounded well under
// US48's own peak (~720k MW) on the high side; storage types legitimately
// read negative (charging) but not by more than a small fraction of that
// same ceiling — quarantine anything outside this wide plausibility band
// rather than propagate a corrupt reading downstream.
const GENERATION_BOUNDS = { mwh: { min: -50_000, max: 800_000 } };

export function gridGenerationEnabled(env: NodeJS.ProcessEnv = process.env): boolean {
  return Boolean(env.EIA_API_KEY);
}

/** Same respondent list as gridDemand.ts (RESPONDENTS) — kept as an
 *  independent constant rather than a shared import so either module's
 *  scope can change without silently moving the other. */
export const RESPONDENTS = ["US48", "CISO", "ERCO", "MISO", "PJM", "NYIS", "ISNE", "SWPP", "FPL", "SE", "NW", "SW"];
/** Trailing window per fetch — covers lag + restarts without bulk. */
export const HOURS_PER_FETCH = 48;
/** Generous headroom over US48's own observed ~16 fuel-type rows/hour —
 *  a respondent reporting more codes than this in one fetch is unlikely
 *  but the length param just caps rows returned, never errors. */
const FUEL_TYPES_PER_HOUR_MAX = 20;
const CALL_SPACING_MS = 300;

export interface GenerationObs {
  period: string;       // "YYYY-MM-DDTHH" (UTC hour as published)
  respondent: string;   // BA code
  fueltype: string;     // EIA-930 fuel-type code (COL, NG, NUC, SUN, WND, ...)
  mwh: number | null;   // net generation for this respondent+fueltype+hour
  rt: string;            // as-seen UTC date
}

const num = (v: any): number | null => {
  if (v == null || v === "") return null;
  const n = typeof v === "number" ? v : parseFloat(String(v));
  return Number.isFinite(n) ? n : null;
};

/** EIA v2 envelope -> GenerationObs (defensive against malformed rows). */
export function parseGeneration(json: any, rt: string): GenerationObs[] {
  const data = json?.response?.data;
  if (!Array.isArray(data)) return [];
  const out: GenerationObs[] = [];
  for (const r of data) {
    const period = r?.period;
    const respondent = r?.respondent;
    const fueltype = r?.fueltype;
    if (typeof period !== "string" || !/^\d{4}-\d{2}-\d{2}T\d{2}$/.test(period)) continue;
    if (!respondent || !fueltype) continue;
    out.push({ period, respondent: String(respondent), fueltype: String(fueltype), mwh: num(r.value), rt });
  }
  return out;
}

// ── Fetch (key never logged; brackets encoded) ──────────────────────────────

type FetchFn = (url: string, init?: any) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export function generationUrl(respondent: string, key: string): string {
  return "https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/" +
    `?api_key=${encodeURIComponent(key)}` +
    "&frequency=hourly&data%5B0%5D=value" +
    `&facets%5Brespondent%5D%5B%5D=${encodeURIComponent(respondent)}` +
    "&sort%5B0%5D%5Bcolumn%5D=period&sort%5B0%5D%5Bdirection%5D=desc" +
    `&length=${HOURS_PER_FETCH * FUEL_TYPES_PER_HOUR_MAX}`;
}

export async function fetchGeneration(fetchImpl: FetchFn = fetch as any,
                                      env: NodeJS.ProcessEnv = process.env,
                                      nowMs?: number, spacingMs = CALL_SPACING_MS): Promise<GenerationObs[]> {
  const key = env.EIA_API_KEY || "";
  if (!key) return [];
  const rt = new Date(nowMs ?? Date.now()).toISOString().slice(0, 10);
  const out: GenerationObs[] = [];
  for (const resp of RESPONDENTS) {
    try {
      const r = await fetchImpl(generationUrl(resp, key), {
        headers: { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)" },
        signal: AbortSignal.timeout(30000) as any,
      });
      if (!r.ok) {
        console.error(`[datacore] gridgeneration ${resp} -> ${r.status}`);
      } else {
        out.push(...parseGeneration(JSON.parse(await r.text()), rt));
      }
    } catch (e: any) {
      console.error(`[datacore] gridgeneration ${resp}:`, e?.message || e);
    }
    if (spacingMs > 0) await sleep(spacingMs);
  }
  return out;
}

// ── Archive (event-identity dedup respondent|period|fueltype, day-file per UTC day) ──

const seenObs = new Set<string>();
let seeded = false;

const obsKey = (o: GenerationObs) => `${o.respondent}|${o.period}|${o.fueltype}`;

function generationDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "gridgeneration");
}

/** Same seed-window bound rationale as gridDemand.ts's SEED_WINDOW_DAYS:
 *  the live poll only ever needs a trailing 48h window for dedup, so
 *  seeding a deep archive into the in-memory set on every restart would
 *  cost heap for no benefit. Re-running a future backfill would need the
 *  same "delete the archive dir with the done-marker" re-run contract. */
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
export function archiveGeneration(obs: GenerationObs[], baseDir?: string): number {
  if (!obs.length) return 0;
  const dir = generationDir(baseDir);
  if (!seeded) { seedSeen(dir); seeded = true; }
  // data-quality gate: quarantine implausible readings before they enter
  // the archive any future reconciliation/stress reader would consume.
  const valid = obs.filter((o) => validateRecord(o as any, GENERATION_BOUNDS).length === 0);
  const quarantined = obs.length - valid.length;
  if (quarantined > 0) console.warn(`[datacore] gridgeneration: quarantined ${quarantined} implausible row(s) (data-quality gate)`);
  const fresh = valid.filter((o) => !seenObs.has(obsKey(o)));
  if (!fresh.length) return 0;
  try {
    fs.mkdirSync(dir, { recursive: true });
    const byDay = new Map<string, GenerationObs[]>();
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
    console.error("[datacore] gridgeneration archive:", e?.message || e);
    return 0;
  }
}

export function gzipOldGenerationDays(baseDir?: string, nowMs?: number): number {
  const dir = generationDir(baseDir);
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

export interface FuelMixEntry { fueltype: string; latest_mwh: number | null }
export interface RespondentGenerationStat {
  respondent: string;
  latest_period: string;
  total_mwh: number | null;   // sum of latest-period fuel readings (null values excluded)
  fuel_mix: FuelMixEntry[];   // sorted by latest_mwh desc
  hours_in_window: number;    // distinct periods observed for this respondent
}

let cache: { at: number; stats: RespondentGenerationStat[] } | null = null;
let polling = false;

export function latestGeneration() {
  return cache;
}

export async function refreshGeneration(fetchImpl: FetchFn = fetch as any,
                                        env: NodeJS.ProcessEnv = process.env,
                                        nowMs?: number, baseDir?: string,
                                        spacingMs = CALL_SPACING_MS): Promise<void> {
  try {
    if (!gridGenerationEnabled(env)) return;
    const obs = await fetchGeneration(fetchImpl, env, nowMs, spacingMs);
    if (obs.length) {
      archiveGeneration(obs, baseDir);
      const byResp = new Map<string, GenerationObs[]>();
      for (const o of obs) {
        if (!byResp.has(o.respondent)) byResp.set(o.respondent, []);
        byResp.get(o.respondent)!.push(o);
      }
      const stats: RespondentGenerationStat[] = [];
      byResp.forEach((rows, respondent) => {
        const latestPeriod = rows.reduce((mx, r) => (r.period > mx ? r.period : mx), rows[0].period);
        const atLatest = rows.filter((r) => r.period === latestPeriod);
        const fuel_mix = atLatest
          .map((r) => ({ fueltype: r.fueltype, latest_mwh: r.mwh }))
          .sort((a, b) => (b.latest_mwh ?? -Infinity) - (a.latest_mwh ?? -Infinity));
        const total_mwh = atLatest.some((r) => r.mwh != null)
          ? atLatest.reduce((sum, r) => sum + (r.mwh ?? 0), 0)
          : null;
        const distinctPeriods = new Set(rows.map((r) => r.period)).size;
        stats.push({ respondent, latest_period: latestPeriod, total_mwh, fuel_mix, hours_in_window: distinctPeriods });
      });
      stats.sort((a, b) => a.respondent.localeCompare(b.respondent));
      cache = { at: Date.now(), stats };
    }
    gzipOldGenerationDays(baseDir, nowMs);
  } catch (e: any) {
    console.error("[datacore] gridgeneration refresh:", e?.message || e);
  }
}

/** Hourly source with ~1-2h lag — 2h poll (RESPONDENTS.length spaced calls
 *  per cycle) keeps the archive within ~2h of real time without hammering,
 *  same cadence as gridDemand.ts's sibling series. Eager boot. */
export function bootGridGenerationPoll(intervalMs = 2 * 60 * 60_000): void {
  if (polling) return;
  polling = true;
  refreshGeneration().catch((e) => console.error("[datacore] gridgeneration boot:", e?.message || e));
  setInterval(() => { refreshGeneration(); }, intervalMs).unref?.();
}
