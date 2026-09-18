/**
 * nhtsaComplaints.ts — NHTSA vehicle complaints, curated watchlist
 * (BUILD ORDER 6 #4, filed + probed 2026-07-06).
 *
 * Source: api.nhtsa.gov/complaints/complaintsByVehicle — keyless JSON,
 * public domain (NHTSA ODI). The daily bulk FLAT_CMPL.zip is 84+MB —
 * unbootable under the memory/event-loop rules — so this stream polls
 * a CURATED ticker-mapped make/model/year watchlist instead
 * (datacore/nhtsa_vehicles.json, static import per the Docker rule;
 * the wikiAttention curated-seed pattern). Shape verified live
 * 2026-07-06: {count, results:[{odiNumber, crash, fire,
 * numberOfInjuries, dateComplaintFiled MM/DD/YYYY, components,
 * summary, ...}]}.
 *
 * HYPOTHESIS (gate-locked in the build order — archive + RAW only):
 * complaint-rate acceleration per make/model (esp. crash/fire flags)
 * precedes recalls and NHTSA investigations, which move automakers and
 * their suppliers. Gate 1 = complaint counts vs NHTSA's published
 * recall timeline for 3 known cases; gate 2 = velocity anomalies vs
 * forward returns. Supplier mapping via the components field is the
 * Everything-Graph follow-up.
 */

import fs from "fs";
import path from "path";
import zlib from "zlib";
import { archiveBaseDir } from "./datacoreArchive";
import watchlist from "../datacore/nhtsa_vehicles.json";

const API = "https://api.nhtsa.gov/complaints/complaintsByVehicle";
/** Politeness spacing between per-vehicle calls (one poll cycle = ~20 calls). */
export const CALL_SPACING_MS = 400;

export interface WatchVehicle {
  ticker: string;
  make: string;
  model: string;
  modelYear: number;
}

export const VEHICLES: WatchVehicle[] = (watchlist as any).vehicles;

export interface ComplaintEvent {
  odi: number;               // ODI number — the event-identity key
  ticker: string;            // from the watchlist mapping
  make: string;
  model: string;
  model_year: number;
  filed: string | null;      // YYYY-MM-DD (normalized from MM/DD/YYYY)
  incident: string | null;   // YYYY-MM-DD
  crash: boolean;
  fire: boolean;
  injuries: number | null;
  components: string | null; // verbatim (supplier-mapping raw material)
  rt: string;                // as-seen UTC date
}

/** "06/26/2026" -> "2026-06-26"; null on garbage. */
export function normalizeUsDate(v: any): string | null {
  const m = String(v ?? "").match(/^(\d{1,2})\/(\d{1,2})\/(\d{4})$/);
  if (!m) return null;
  return `${m[3]}-${m[1].padStart(2, "0")}-${m[2].padStart(2, "0")}`;
}

/** One vehicle's API response -> ComplaintEvents (summaries deliberately
 *  NOT archived — free-text bulk belongs to the flat-file follow-up; the
 *  velocity signal needs counts, flags, dates, components). */
export function parseComplaints(json: any, v: WatchVehicle, rt: string): ComplaintEvent[] {
  const results = json?.results;
  if (!Array.isArray(results)) return [];
  const out: ComplaintEvent[] = [];
  for (const r of results) {
    const odi = r?.odiNumber;
    if (typeof odi !== "number" || !Number.isFinite(odi)) continue;
    out.push({
      odi,
      ticker: v.ticker,
      make: v.make,
      model: v.model,
      model_year: v.modelYear,
      filed: normalizeUsDate(r.dateComplaintFiled),
      incident: normalizeUsDate(r.dateOfIncident),
      crash: r.crash === true,
      fire: r.fire === true,
      injuries: typeof r.numberOfInjuries === "number" ? r.numberOfInjuries : null,
      components: r.components != null ? String(r.components) : null,
      rt,
    });
  }
  return out;
}

// ── Fetch ────────────────────────────────────────────────────────────────────

type FetchFn = (url: string, init?: any) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export async function fetchVehicleComplaints(v: WatchVehicle, fetchImpl: FetchFn = fetch as any,
                                             nowMs?: number): Promise<ComplaintEvent[]> {
  const rt = new Date(nowMs ?? Date.now()).toISOString().slice(0, 10);
  const url = `${API}?make=${encodeURIComponent(v.make)}&model=${encodeURIComponent(v.model)}&modelYear=${v.modelYear}`;
  try {
    const r = await fetchImpl(url, {
      headers: { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)" },
      signal: AbortSignal.timeout(30000) as any,
    });
    if (!r.ok) {
      console.error(`[datacore] nhtsa ${v.make}/${v.model} -> ${r.status}`);
      return [];
    }
    return parseComplaints(JSON.parse(await r.text()), v, rt);
  } catch (e: any) {
    console.error(`[datacore] nhtsa ${v.make}/${v.model}:`, e?.message || e);
    return [];
  }
}

// ── Archive (event-identity dedup by ODI number, day-file per fetch date) ───

const seenOdi = new Set<number>();
let seeded = false;

function complaintsDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "nhtsacomplaints");
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
        try { seenOdi.add(JSON.parse(line).odi); } catch {}
      }
    }
  } catch {}
}

/** Appends UNSEEN complaints to today's day-file. Returns count written. */
export function archiveNewComplaints(events: ComplaintEvent[], baseDir?: string, nowMs?: number): number {
  if (!events.length) return 0;
  const dir = complaintsDir(baseDir);
  if (!seeded) { seedSeen(dir); seeded = true; }
  const fresh = events.filter((e) => !seenOdi.has(e.odi));
  if (!fresh.length) return 0;
  try {
    fs.mkdirSync(dir, { recursive: true });
    const day = new Date(nowMs ?? Date.now()).toISOString().slice(0, 10);
    fs.appendFileSync(path.join(dir, `${day}.jsonl`),
                      fresh.map((r) => JSON.stringify(r)).join("\n") + "\n");
    fresh.forEach((e) => seenOdi.add(e.odi));
    return fresh.length;
  } catch (e: any) {
    console.error("[datacore] nhtsa archive:", e?.message || e);
    return 0;
  }
}

export function gzipOldComplaintDays(baseDir?: string, nowMs?: number): number {
  const dir = complaintsDir(baseDir);
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

export interface VehicleStat {
  ticker: string;
  make: string;
  model: string;
  model_year: number;
  total_complaints: number;   // API-reported total for the vehicle
  crash_count: number;        // within the fetched window
  fire_count: number;
  newest_filed: string | null;
}

let cache: { at: number; stats: VehicleStat[] } | null = null;
let polling = false;

export function latestComplaintStats() {
  return cache;
}

/** Test-only: this module's dedup/cache state is module-level singleton
 *  (same class of problem as githubOrgActivity.ts's/euLoad.ts's own
 *  `_reset*ForTests`), so a test exercising the cold-cache backfill path
 *  must be able to reset it rather than rely on file execution order to
 *  find `cache` still null. */
export function _resetComplaintsForTests(): void {
  seenOdi.clear();
  seeded = false;
  cache = null;
  polling = false;
}

/** Per-vehicle stats from a set of complaint events — pure, so the live
 *  sweep path and the disk-backfill path (below) compute the exact same
 *  shape from whichever source produced events this cycle. Groups by the
 *  vehicle identity (ticker+make+model+year) rather than assuming a
 *  single vehicle, so it also works on a multi-vehicle archive scan. */
export function computeVehicleStats(events: ComplaintEvent[]): VehicleStat[] {
  const byVehicle = new Map<string, ComplaintEvent[]>();
  for (const e of events) {
    const key = `${e.ticker}|${e.make}|${e.model}|${e.model_year}`;
    if (!byVehicle.has(key)) byVehicle.set(key, []);
    byVehicle.get(key)!.push(e);
  }
  const stats: VehicleStat[] = [];
  byVehicle.forEach((rows) => {
    const v0 = rows[0];
    stats.push({
      ticker: v0.ticker, make: v0.make, model: v0.model, model_year: v0.model_year,
      total_complaints: rows.length,
      crash_count: rows.filter((e) => e.crash).length,
      fire_count: rows.filter((e) => e.fire).length,
      newest_filed: rows.reduce<string | null>(
        (mx, e) => (e.filed && (!mx || e.filed > mx) ? e.filed : mx), null),
    });
  });
  return stats;
}

/** Archived complaint events over the last `lookbackDays` calendar days
 *  (plain or gz) — used to backfill the live cache when a cold boot's or
 *  a live-outage cycle's sweep produces zero events across the ENTIRE
 *  watchlist, so a transport failure doesn't report `warming_up` over
 *  real archived history already on disk (Freshness Law; same pattern as
 *  euLoad.ts's `readRecentArchivedLoad`/satellites.ts's
 *  `readArchivedGroup`, the shipped precedents this generalizes). Kept
 *  separate from `seedSeen`: that only tracks ODI dedup keys, this
 *  returns full rows. 30-day default (vs. euLoad's 5): this is a
 *  low-volume curated watchlist polled every 12h, not a high-frequency
 *  series, so a short window would starve `total_complaints` of most of
 *  what is actually archived. */
export function readArchivedComplaints(baseDir?: string, nowMs?: number, lookbackDays = 30): ComplaintEvent[] {
  const now = nowMs ?? Date.now();
  const dir = complaintsDir(baseDir);
  const out: ComplaintEvent[] = [];
  for (let i = 0; i < lookbackDays; i++) {
    const iso = new Date(now - i * 86400_000).toISOString().slice(0, 10);
    for (const fp of [path.join(dir, `${iso}.jsonl`), path.join(dir, `${iso}.jsonl.gz`)]) {
      let text: string | null = null;
      try {
        text = fp.endsWith(".gz")
          ? zlib.gunzipSync(fs.readFileSync(fp)).toString("utf8")
          : fs.readFileSync(fp, "utf8");
      } catch { continue; }
      for (const line of text.split("\n")) {
        if (!line) continue;
        try { out.push(JSON.parse(line)); } catch { continue; }
      }
    }
  }
  return out;
}

/** Full watchlist sweep with politeness spacing; archives new events and
 *  rebuilds the per-vehicle stat cache. */
export async function refreshComplaints(fetchImpl: FetchFn = fetch as any, nowMs?: number,
                                        baseDir?: string, spacingMs = CALL_SPACING_MS): Promise<void> {
  const events: ComplaintEvent[] = [];
  for (const v of VEHICLES) {
    const vEvents = await fetchVehicleComplaints(v, fetchImpl, nowMs);
    if (vEvents.length) {
      archiveNewComplaints(vEvents, baseDir, nowMs);
      events.push(...vEvents);
    }
    if (spacingMs > 0) await sleep(spacingMs);
  }
  if (events.length > 0) {
    cache = { at: Date.now(), stats: computeVehicleStats(events) };
  } else if (!cache) {
    const archived = readArchivedComplaints(baseDir, nowMs);
    if (archived.length > 0) cache = { at: Date.now(), stats: computeVehicleStats(archived) };
  }
  gzipOldComplaintDays(baseDir, nowMs);
}

/** Bulk file updates daily; per-vehicle API reflects it — 12h poll keeps
 *  the sweep cheap (~20 calls/cycle). Eager boot per KNOWN BROKEN #9. */
export function bootComplaintsPoll(intervalMs = 12 * 60 * 60_000): void {
  if (polling) return;
  polling = true;
  refreshComplaints();
  setInterval(() => { refreshComplaints(); }, intervalMs).unref?.();
}
