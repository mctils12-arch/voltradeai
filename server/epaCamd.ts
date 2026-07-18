/**
 * epaCamd.ts — EPA CAMD CEMS (Continuous Emissions Monitoring) pipeline.
 * Data census SECTION 3 item 1 ("★ THE STANDOUT", research/data_census.md,
 * probed 2026-07-06) — the single highest-signal-value unbuilt census item.
 *
 * SIGNAL (not traded here): unit-level grossLoad x opTime IS direct
 * plant-utilization GROUND TRUTH for every US Part-75 generating unit —
 * the ladder gate-1 truth source for the whole power vertical (it is what
 * validates GPPD/satellite-thermal inference roots — e.g. the tank-shadow
 * estimator's successors — and a merchant-generator output proxy). This
 * module ships the RAW archive only; any predictive claim built from it is
 * its own gate-2 work, not attempted here.
 *
 * SCOPE v1 (pilot, matches GRID VISION's existing TX/ERCOT focus and keeps
 * DEMO_KEY usage trivially small — see KEY below): stateCode=TX only.
 * Widening to more states is a follow-up once a dedicated EPA_CAMD_API_KEY
 * (wishlist 9a, free instant api.data.gov signup) replaces the fallback.
 *
 * CADENCE HONESTY (live-probed 2026-07-18, not assumed from the census
 * note): this is QUARTERLY government data, not daily/real-time, despite
 * daily/hourly granularity WITHIN each record — a quarter only becomes
 * queryable once it has fully closed (probed 2026-07-18: Q3-2026
 * beginDate/endDate both rejected, "quarter ending on 06/30/2026" as the
 * live upper bound; Q2-2026 was fully queryable). This module uses the
 * daily-apportioned endpoint (aggregated per unit per day — 9,009 TX rows
 * for a whole quarter, vs. 2,376 rows for ONE hour-window on ONE day from
 * the hourly endpoint) and fetches an ENTIRE QUARTER in one API call.
 *
 * KEY: EPA_CAMD_API_KEY (wishlist 9a) or the shared "DEMO_KEY" fallback —
 * UNLIKE nasaFirms.ts/fredMacro.ts this module is NOT fully key-gated-
 * dormant, because DEMO_KEY was live-verified working this session
 * (facility attributes + daily emissions both 200'd against the real API).
 * DEMO_KEY is a globally shared, tightly-rate-limited key (api.data.gov:
 * ~30 req/hr, ~50 req/day, shared across every DEMO_KEY caller worldwide),
 * so this module makes AT MOST one quarter-data call plus an occasional
 * (7-day-cached) facility-attributes call per poll, and treats 429/403 as
 * a normal silent-next-poll outcome, never a crash. A dedicated key only
 * raises the ceiling and removes shared-key collision risk — it does not
 * change what this module does.
 *
 * Public domain (US EPA, 40 CFR Part 75). Attribution: "U.S. EPA Clean Air
 * Markets Division (CAMD)".
 */

import fs from "fs";
import path from "path";
import zlib from "zlib";
import { archiveBaseDir } from "./datacoreArchive";

const STATE = "TX";
const BACKFILL_QUARTERS = 8; // ~2 years
const FACILITY_ATTRS_TTL_MS = 7 * 86_400_000;

export interface EpaCamdDailyRow {
  stateCode: string;
  facilityId: number;
  facilityName: string | null;
  unitId: string;
  date: string;
  countOpTime: number | null;
  sumOpTime: number | null;
  grossLoad: number | null;
  steamLoad: number | null;
  so2Mass: number | null;
  co2Mass: number | null;
  noxMass: number | null;
  heatInput: number | null;
  primaryFuelInfo: string | null;
  unitType: string | null;
  lat: number | null;
  lon: number | null;
  ownerOperator: string | null;
  rt: string; // as-seen UTC date
}

export interface FacilityAttr { lat: number | null; lon: number | null; ownerOperator: string | null }

export interface FacilitySummary {
  facilityId: number;
  facilityName: string | null;
  unitCount: number;
  sumOpTime: number;
  sumGrossLoad: number;
  primaryFuelInfo: string | null;
  lat: number | null;
  lon: number | null;
  ownerOperator: string | null;
}

function numOrNull(v: any): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

// ── Parse (documented api.epa.gov/easey streaming-services shapes,
// live-verified 2026-07-18) ─────────────────────────────────────────────────

export function parseFacilityAttrs(json: any): Map<string, FacilityAttr> {
  const m = new Map<string, FacilityAttr>();
  for (const f of Array.isArray(json) ? json : []) {
    if (f?.facilityId == null || !f?.unitId) continue;
    m.set(`${f.facilityId}|${f.unitId}`, {
      lat: numOrNull(f.latitude),
      lon: numOrNull(f.longitude),
      ownerOperator: f.ownerOperator ?? null,
    });
  }
  return m;
}

export function parseDailyEmissions(
  json: any, rt: string, attrs: Map<string, FacilityAttr>,
): EpaCamdDailyRow[] {
  const out: EpaCamdDailyRow[] = [];
  for (const r of Array.isArray(json) ? json : []) {
    if (r?.facilityId == null || !r?.unitId || !r?.date) continue;
    const a = attrs.get(`${r.facilityId}|${r.unitId}`) || { lat: null, lon: null, ownerOperator: null };
    out.push({
      stateCode: r.stateCode ?? STATE,
      facilityId: r.facilityId,
      facilityName: r.facilityName ?? null,
      unitId: r.unitId,
      date: r.date,
      countOpTime: numOrNull(r.countOpTime),
      sumOpTime: numOrNull(r.sumOpTime),
      grossLoad: numOrNull(r.grossLoad),
      steamLoad: numOrNull(r.steamLoad),
      so2Mass: numOrNull(r.so2Mass),
      co2Mass: numOrNull(r.co2Mass),
      noxMass: numOrNull(r.noxMass),
      heatInput: numOrNull(r.heatInput),
      primaryFuelInfo: r.primaryFuelInfo ?? null,
      unitType: r.unitType ?? null,
      lat: a.lat,
      lon: a.lon,
      ownerOperator: a.ownerOperator,
      rt,
    });
  }
  return out;
}

/** Per-facility rollup of the newest archived quarter — what a gate-1
 *  comparison against a satellite/GPPD inference actually wants, and a
 *  far lighter API payload than 9k+ per-unit-per-day rows. */
export function aggregateByFacility(rows: EpaCamdDailyRow[]): FacilitySummary[] {
  const m = new Map<number, FacilitySummary>();
  for (const r of rows) {
    let s = m.get(r.facilityId);
    if (!s) {
      s = {
        facilityId: r.facilityId, facilityName: r.facilityName, unitCount: 0,
        sumOpTime: 0, sumGrossLoad: 0, primaryFuelInfo: r.primaryFuelInfo,
        lat: r.lat, lon: r.lon, ownerOperator: r.ownerOperator,
      };
      m.set(r.facilityId, s);
    }
    if (r.sumOpTime != null) s.sumOpTime += r.sumOpTime;
    if (r.grossLoad != null) s.sumGrossLoad += r.grossLoad;
    if (s.lat == null && r.lat != null) s.lat = r.lat;
    if (s.lon == null && r.lon != null) s.lon = r.lon;
    if (s.ownerOperator == null && r.ownerOperator != null) s.ownerOperator = r.ownerOperator;
  }
  // unitCount = distinct unitId per facility (second pass, avoids a nested Set-in-loop bug above)
  const units = new Map<number, Set<string>>();
  for (const r of rows) {
    if (!units.has(r.facilityId)) units.set(r.facilityId, new Set());
    units.get(r.facilityId)!.add(r.unitId);
  }
  for (const [id, s] of m) s.unitCount = units.get(id)?.size ?? 0;
  return [...m.values()].sort((a, b) => b.sumGrossLoad - a.sumGrossLoad);
}

// ── Quarter math ─────────────────────────────────────────────────────────

export function quarterOf(d: Date): { year: number; quarter: number } {
  return { year: d.getUTCFullYear(), quarter: Math.floor(d.getUTCMonth() / 3) + 1 };
}

export function quarterBounds(year: number, quarter: number): { begin: string; end: string } {
  const startMonth = (quarter - 1) * 3;
  const begin = new Date(Date.UTC(year, startMonth, 1));
  const end = new Date(Date.UTC(year, startMonth + 3, 0)); // day 0 of next month = last day of this quarter
  return { begin: begin.toISOString().slice(0, 10), end: end.toISOString().slice(0, 10) };
}

/** Most recent quarter that has FULLY closed as of `now` — EPA CAMD only
 *  publishes completed quarters (live-probed 2026-07-18: the in-progress
 *  quarter is rejected, with the prior quarter's end date as the API's own
 *  stated upper bound). */
export function latestClosedQuarter(now: Date): { year: number; quarter: number } {
  const { year, quarter } = quarterOf(now);
  return quarter === 1 ? { year: year - 1, quarter: 4 } : { year, quarter: quarter - 1 };
}

export function priorQuarters(
  latest: { year: number; quarter: number }, n: number,
): Array<{ year: number; quarter: number }> {
  const out: Array<{ year: number; quarter: number }> = [];
  let { year, quarter } = latest;
  for (let i = 0; i < n; i++) {
    out.push({ year, quarter });
    quarter -= 1;
    if (quarter === 0) { quarter = 4; year -= 1; }
  }
  return out;
}

// ── Fetch (injectable for tests) ────────────────────────────────────────────

type FetchFn = (url: string, init?: any) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;

export function epaCamdKey(env: NodeJS.ProcessEnv = process.env): string {
  return env.EPA_CAMD_API_KEY || "DEMO_KEY";
}

export function epaCamdUsingDemoKey(env: NodeJS.ProcessEnv = process.env): boolean {
  return !env.EPA_CAMD_API_KEY;
}

async function fetchJson(url: string, fetchImpl: FetchFn): Promise<any> {
  const r = await fetchImpl(url, { signal: AbortSignal.timeout(30000) as any });
  if (!r.ok) throw new Error(`epacamd -> ${r.status}`);
  return JSON.parse(await r.text());
}

export async function fetchFacilityAttrs(
  year: number, apiKey: string, fetchImpl: FetchFn = fetch as any,
): Promise<Map<string, FacilityAttr>> {
  const url = "https://api.epa.gov/easey/streaming-services/facilities/attributes" +
    `?stateCode=${STATE}&year=${year}&api_key=${encodeURIComponent(apiKey)}`;
  return parseFacilityAttrs(await fetchJson(url, fetchImpl));
}

export async function fetchQuarterDaily(
  year: number, quarter: number, apiKey: string, attrs: Map<string, FacilityAttr>,
  fetchImpl: FetchFn = fetch as any, nowMs?: number,
): Promise<EpaCamdDailyRow[]> {
  const { begin, end } = quarterBounds(year, quarter);
  const rt = new Date(nowMs ?? Date.now()).toISOString().slice(0, 10);
  const url = "https://api.epa.gov/easey/streaming-services/emissions/apportioned/daily" +
    `?stateCode=${STATE}&year=${year}&quarter=${quarter}&beginDate=${begin}&endDate=${end}` +
    `&api_key=${encodeURIComponent(apiKey)}`;
  return parseDailyEmissions(await fetchJson(url, fetchImpl), rt, attrs);
}

// ── Archive (one JSONL file per quarter; vintage dedup seeded from the
// FULL existing file, not a bounded day-window — a quarter file already IS
// the complete relevant dedup domain, avoiding the fredMacro bounded-seed
// defect class by construction) ─────────────────────────────────────────────

function epaCamdDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "epacamd");
}

function quarterFileBase(dir: string, year: number, quarter: number): string {
  return path.join(dir, `${STATE}_${year}Q${quarter}.jsonl`);
}

/** True if a non-empty archive file exists for this quarter, gzipped or not. */
function quarterArchived(dir: string, year: number, quarter: number): boolean {
  const base = quarterFileBase(dir, year, quarter);
  for (const fp of [base, `${base}.gz`]) {
    try { if (fs.statSync(fp).size > 0) return true; } catch {}
  }
  return false;
}

/** A quarter is SEALED (never re-fetched again) once archived AND more than
 *  45 days past its own quarter-end — comfortably past EPA's documented
 *  ~30-day post-quarter final-data deadline, so quota is never spent
 *  re-checking settled history. */
export function quarterSealed(year: number, quarter: number, nowMs: number, baseDir?: string): boolean {
  const { end } = quarterBounds(year, quarter);
  const daysSinceEnd = (nowMs - Date.parse(`${end}T00:00:00Z`)) / 86_400_000;
  return daysSinceEnd >= 45 && quarterArchived(epaCamdDir(baseDir), year, quarter);
}

function rowKey(r: EpaCamdDailyRow): string {
  return `${r.facilityId}|${r.unitId}|${r.date}`;
}

function rowSignature(r: EpaCamdDailyRow): string {
  return JSON.stringify([r.countOpTime, r.sumOpTime, r.grossLoad, r.steamLoad,
    r.so2Mass, r.co2Mass, r.noxMass, r.heatInput]);
}

/** Appends only rows whose value signature differs from the LATEST
 *  previously-archived row for that (facility,unit,date) — true vintage
 *  dedup: a within-window correction appends a new row with a new rt,
 *  an identical re-fetch is a no-op. */
export function archiveQuarterRows(
  rows: EpaCamdDailyRow[], year: number, quarter: number, baseDir?: string,
): number {
  const dir = epaCamdDir(baseDir);
  const fp = quarterFileBase(dir, year, quarter);
  const latest = new Map<string, string>();
  try {
    for (const line of fs.readFileSync(fp, "utf8").split("\n")) {
      if (!line) continue;
      try { const r = JSON.parse(line); latest.set(rowKey(r), rowSignature(r)); } catch {}
    }
  } catch {}
  const fresh = rows.filter((r) => latest.get(rowKey(r)) !== rowSignature(r));
  if (!fresh.length) return 0;
  try {
    fs.mkdirSync(dir, { recursive: true });
    fs.appendFileSync(fp, fresh.map((r) => JSON.stringify(r)).join("\n") + "\n");
    return fresh.length;
  } catch (e: any) {
    console.error("[datacore] epacamd archive:", e?.message || e);
    return 0;
  }
}

export function gzipSealedQuarters(nowMs: number, baseDir?: string): number {
  const dir = epaCamdDir(baseDir);
  let n = 0;
  try {
    for (const f of fs.readdirSync(dir)) {
      const m = f.match(/^([A-Z]{2})_(\d{4})Q(\d)\.jsonl$/);
      if (!m) continue;
      if (!quarterSealed(Number(m[2]), Number(m[3]), nowMs, baseDir)) continue;
      const fp = path.join(dir, f);
      fs.writeFileSync(`${fp}.gz`, zlib.gzipSync(fs.readFileSync(fp)));
      fs.unlinkSync(fp);
      n++;
    }
  } catch {}
  return n;
}

/** Picks the single quarter to fetch THIS poll — at most one API call for
 *  quarter data per invocation, keeping shared-DEMO_KEY usage trivial.
 *  Priority: (1) the oldest quarter in the backfill window that has NEVER
 *  been archived (chronological backfill, a few quarters per day); once
 *  every quarter in range has at least one archived pass, (2) recheck the
 *  latest closed quarter while it's still within its 45-day correction
 *  window; otherwise (3) nothing to do — steady state. */
export function pickQuarterToFetch(
  nowMs: number, baseDir?: string, backfillQuarters = BACKFILL_QUARTERS,
): { year: number; quarter: number } | null {
  const latest = latestClosedQuarter(new Date(nowMs));
  const quarters = priorQuarters(latest, backfillQuarters); // newest -> oldest
  const dir = epaCamdDir(baseDir);
  const neverArchived = [...quarters].reverse().find((q) => !quarterArchived(dir, q.year, q.quarter));
  if (neverArchived) return neverArchived;
  if (!quarterSealed(latest.year, latest.quarter, nowMs, baseDir)) return latest;
  return null;
}

// ── Cache + poll ────────────────────────────────────────────────────────────

let facilityAttrsCache: { at: number; year: number; attrs: Map<string, FacilityAttr> } | null = null;

async function getFacilityAttrs(
  apiKey: string, year: number, fetchImpl: FetchFn, nowMs: number,
): Promise<Map<string, FacilityAttr>> {
  if (facilityAttrsCache && nowMs - facilityAttrsCache.at < FACILITY_ATTRS_TTL_MS) {
    return facilityAttrsCache.attrs;
  }
  const attrs = await fetchFacilityAttrs(year, apiKey, fetchImpl);
  facilityAttrsCache = { at: nowMs, year, attrs };
  return attrs;
}

let cache: { at: number; year: number; quarter: number; rows: EpaCamdDailyRow[] } | null = null;
let polling = false;

export function latestEpaCamd(): { at: number; year: number; quarter: number; rows: EpaCamdDailyRow[] } | null {
  return cache;
}

export async function refreshEpaCamd(
  fetchImpl: FetchFn = fetch as any, env: NodeJS.ProcessEnv = process.env, nowMs?: number,
): Promise<void> {
  const now = nowMs ?? Date.now();
  const target = pickQuarterToFetch(now);
  if (target) {
    const apiKey = epaCamdKey(env);
    try {
      const attrs = await getFacilityAttrs(apiKey, new Date(now).getUTCFullYear(), fetchImpl, now);
      const rows = await fetchQuarterDaily(target.year, target.quarter, apiKey, attrs, fetchImpl, now);
      archiveQuarterRows(rows, target.year, target.quarter);
      cache = { at: now, year: target.year, quarter: target.quarter, rows };
    } catch (e: any) {
      console.error("[datacore] epacamd refresh:", e?.message || e);
    }
  }
  try { gzipSealedQuarters(now); } catch {}
}

/** 12h poll — quarterly-cadence data, at most one quarter-data call per
 *  invocation (see pickQuarterToFetch); boots eagerly like the other RAW
 *  government streams, DEMO_KEY fallback included, since it was
 *  live-verified working without any Railway key. */
export function bootEpaCamdPoll(intervalMs = 12 * 60 * 60_000): void {
  if (polling) return;
  polling = true;
  refreshEpaCamd();
  setInterval(() => { refreshEpaCamd(); }, intervalMs).unref?.();
}
