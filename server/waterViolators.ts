/**
 * waterViolators.ts — EPA Clean Water Act chronic violators (ECHO)
 * (Location Context Engine hazards wave 2 — research/location_context_engine.md;
 * the human's "factories' water quality" case).
 *
 * Source: EPA ECHO (Enforcement and Compliance History Online), Clean Water
 * Act facility search — active NPDES facilities with MORE THAN 8 OF THE LAST
 * 12 QUARTERS in noncompliance (p_qiv=GT8, ~25k facilities). Public domain.
 * Two-step fetch: get_facilities (returns a QueryID) -> get_download (bulk
 * CSV of every row, selected columns) — the paged JSON API returns one row
 * per call, so CSV is the only sane bulk path.
 *
 * RAW/FACTUAL: a facility's violation quarters + SNC status are EPA's own
 * published compliance records, shown as-is. NOT a water-safety claim about
 * any location (that would be a ladder-gated signal). predictive:false.
 *
 * DATA QUALITY GATE: coords (range + null-island), name required, quarters
 * in [0,12]. Failures quarantined + counted, never served as clean. Weekly
 * server-side refresh (Railway); a failed refresh keeps the last good
 * snapshot marked stale.
 */
import { validateRecord, coordValid, freshnessStatus, type LayerHealth, type QualityIssue } from "./dataQuality";

const BASE = "https://echodata.epa.gov/echo/cwa_rest_services";
// qcolumns: 1 CWPName, 2 SourceID, 4 CWPCity, 5 CWPState, 24 FacLat,
// 25 FacLong, 54 CWPPermitTypeDesc, 98 CWPSNCStatus, 101 CWPQtrsWithNC,
// 114 CWPFormalEaCnt (verified against cwa_rest_services.metadata 2026-07-12)
const QCOLUMNS = "1,2,4,5,24,25,54,98,101,114";

export interface WaterViolator {
  name: string;
  id: string;            // NPDES permit id
  city: string | null;
  state: string | null;
  lat: number;
  lon: number;
  permit: string | null; // permit type description
  snc: string | null;    // current Significant Noncompliance category (may be blank)
  qtrs: number;          // quarters (of last 12) with noncompliance
  actions: number | null; // formal enforcement action count
}

/** Minimal CSV parser (quoted fields, embedded commas). Line-oriented — ECHO
 *  emits no embedded newlines in these columns. */
export function parseCsvLine(line: string): string[] {
  const out: string[] = [];
  let cur = "", inQ = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (inQ) {
      if (ch === '"') { if (line[i + 1] === '"') { cur += '"'; i++; } else inQ = false; }
      else cur += ch;
    } else if (ch === '"') inQ = true;
    else if (ch === ",") { out.push(cur); cur = ""; }
    else cur += ch;
  }
  out.push(cur);
  return out;
}

const SCHEMA = {
  CWPName: { required: true },
  CWPQtrsWithNC: { numeric: true, min: 0, max: 12 }, // of the last 12 quarters
};

/** One CSV record (as a header->value map) -> validated WaterViolator or issues. */
export function violatorFromRow(r: Record<string, string>): { v?: WaterViolator; issues: QualityIssue[] } {
  const issues = validateRecord(r, SCHEMA);
  if (!coordValid(r.FacLat, r.FacLong)) issues.push({ field: "coordinates", rule: "coordValid", detail: `${r.FacLat},${r.FacLong}` });
  if (issues.length) return { issues };
  return {
    issues: [],
    v: {
      name: r.CWPName.trim(),
      id: r.SourceID || "",
      city: r.CWPCity || null,
      state: r.CWPState || null,
      lat: Number(r.FacLat),
      lon: Number(r.FacLong),
      permit: r.CWPPermitTypeDesc || null,
      snc: r.CWPSNCStatus || null,
      qtrs: Number(r.CWPQtrsWithNC),
      actions: r.CWPFormalEaCnt ? Number(r.CWPFormalEaCnt) : null,
    },
  };
}

export function parseDownloadCsv(csv: string): { violators: WaterViolator[]; suspect: number } {
  const lines = csv.split("\n").filter((l) => l.trim().length);
  if (lines.length < 2) return { violators: [], suspect: 0 };
  const header = parseCsvLine(lines[0]);
  const violators: WaterViolator[] = [];
  let suspect = 0;
  for (let i = 1; i < lines.length; i++) {
    const cells = parseCsvLine(lines[i]);
    const rec: Record<string, string> = {};
    header.forEach((h, j) => { rec[h] = cells[j] ?? ""; });
    const { v, issues } = violatorFromRow(rec);
    if (v) violators.push(v); else if (issues.length) suspect++;
  }
  return { violators, suspect };
}

type FetchFn = (url: string, init?: any) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;

export function facilitiesUrl(): string {
  return `${BASE}.get_facilities?output=JSON&p_act=Y&p_qiv=GT8&responseset=1&p_maxrows=1`;
}
export function downloadUrl(qid: string): string {
  return `${BASE}.get_download?output=CSV&qid=${encodeURIComponent(qid)}&qcolumns=${QCOLUMNS}`;
}

export async function fetchWaterViolators(fetchImpl: FetchFn = fetch as any):
    Promise<{ violators: WaterViolator[]; suspect: number }> {
  const ua = { headers: { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)" } };
  try {
    const r1 = await fetchImpl(facilitiesUrl(), { ...ua, signal: AbortSignal.timeout(60000) as any });
    if (!r1.ok) { console.error(`[datacore] waterviolators qid -> ${r1.status}`); return { violators: [], suspect: 0 }; }
    const qid = JSON.parse(await r1.text())?.Results?.QueryID;
    if (!qid) { console.error("[datacore] waterviolators: no QueryID"); return { violators: [], suspect: 0 }; }
    const r2 = await fetchImpl(downloadUrl(String(qid)), { ...ua, signal: AbortSignal.timeout(180000) as any });
    if (!r2.ok) { console.error(`[datacore] waterviolators csv -> ${r2.status}`); return { violators: [], suspect: 0 }; }
    return parseDownloadCsv(await r2.text());
  } catch (e: any) {
    console.error("[datacore] waterviolators:", e?.message || e);
    return { violators: [], suspect: 0 };
  }
}

// ── cache + weekly refresh (compliance quarters update quarterly upstream) ──

interface WvCache { at: number; violators: WaterViolator[]; health: LayerHealth }
let cache: WvCache | null = null;
let polling = false;
const FRESH_BUDGET_MS = 14 * 86400_000;

export function latestWaterViolators(): WvCache | null { return cache; }

export async function refreshWaterViolators(fetchImpl?: FetchFn, nowMs?: number): Promise<void> {
  try {
    const { violators, suspect } = await fetchWaterViolators(fetchImpl);
    if (!violators.length) {
      if (cache) cache.health = { ...cache.health, freshness: "stale" };
      return;
    }
    const now = nowMs ?? Date.now();
    cache = {
      at: now, violators,
      health: {
        source: "EPA ECHO Clean Water Act compliance (public domain)",
        clean: violators.length, suspect,
        freshness: freshnessStatus(now, FRESH_BUDGET_MS, now),
        source_date: new Date(now).toISOString().slice(0, 10),
      },
    };
  } catch (e: any) { console.error("[datacore] waterviolators refresh:", e?.message || e); }
}

export function bootWaterViolatorsPoll(intervalMs = 7 * 86400_000): void {
  if (polling) return;
  polling = true;
  refreshWaterViolators().catch((e) => console.error("[datacore] waterviolators boot:", e?.message || e));
  setInterval(() => { refreshWaterViolators(); }, intervalMs).unref?.();
}
