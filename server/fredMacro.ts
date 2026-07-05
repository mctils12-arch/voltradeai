/**
 * fredMacro.ts — FRED macro-series pipeline (St. Louis Fed API).
 * Stream #3 of the DATA STREAM EXPANSION build order (directive 2026-07-05;
 * hypothesis + ladder filed in research/open_questions.md BEFORE this build).
 *
 * HYPOTHESIS (filed): NOT a direct trading signal — this is the REGIME
 * INPUT feed for regime classification and for gate-2 conditioning of
 * every other stream. Never traded alone.
 *
 * LADDER gate 1 (DATA): values must match the FRED web UI on 10 spot
 * checks — verified post-deploy against the keyless fredgraph.csv export
 * (the web UI's own download), since the API key lives only on Railway.
 *
 * POINT-IN-TIME HONESTY (Reasoning Standard #7 — lookahead is a silent
 * killer): FRED silently revises history. Every archived observation
 * carries rt = the date WE saw it, and a revision (same series+date, new
 * value) is archived as a NEW row, never overwritten. Recording from
 * today forward, time turns this free feed into the paid ALFRED-style
 * vintage dataset (BUILD-FIRST rule #2: accumulation substitutes for
 * purchase). Consumers who want "what was known on day X" filter rt <= X.
 *
 * LICENSING (checked first, per the stream doctrine): FRED data is free
 * with attribution "Source: FRED, Federal Reserve Bank of St. Louis" —
 * but a handful of popular series are THIRD-PARTY COPYRIGHTED (CBOE VIX,
 * ICE BofA credit spreads, UMich sentiment). Those are marked
 * license:"restricted" below: archived for INTERNAL regime use only and
 * NEVER exposed through the public /api/data/macro route or any /data or
 * /api/v1 product surface. Terms re-check at the monetization switch per
 * the standing wishlist rule.
 *
 * Key-gated exactly like nasaFirms.ts: without FRED_API_KEY the module
 * stays dormant. Boundary: no imports from trading logic.
 */

import fs from "fs";
import path from "path";
import zlib from "zlib";
import { archiveBaseDir } from "./datacoreArchive";

export interface FredSeriesDef {
  id: string;
  label: string;
  cadence: "daily" | "weekly" | "monthly";
  unit: string;
  /** public = US-gov/Fed-produced, exposable on product surfaces.
   *  restricted = third-party copyrighted, internal regime use ONLY. */
  license: "public" | "restricted";
}

/** ~28 regime-relevant series (rates/curve, stress, labor, inflation,
 *  activity, liquidity, commodities). Selection favors US-government and
 *  Fed-produced series so the product surface stays clean. */
export const FRED_SERIES: FredSeriesDef[] = [
  // rates & curve
  { id: "DGS3MO", label: "3-Month Treasury", cadence: "daily", unit: "%", license: "public" },
  { id: "DGS2", label: "2-Year Treasury", cadence: "daily", unit: "%", license: "public" },
  { id: "DGS10", label: "10-Year Treasury", cadence: "daily", unit: "%", license: "public" },
  { id: "DGS30", label: "30-Year Treasury", cadence: "daily", unit: "%", license: "public" },
  { id: "T10Y2Y", label: "10Y–2Y Spread", cadence: "daily", unit: "%", license: "public" },
  { id: "T10Y3M", label: "10Y–3M Spread", cadence: "daily", unit: "%", license: "public" },
  { id: "FEDFUNDS", label: "Fed Funds Rate", cadence: "monthly", unit: "%", license: "public" },
  { id: "SOFR", label: "SOFR", cadence: "daily", unit: "%", license: "public" },
  { id: "T5YIE", label: "5Y Breakeven Inflation", cadence: "daily", unit: "%", license: "public" },
  { id: "T10YIE", label: "10Y Breakeven Inflation", cadence: "daily", unit: "%", license: "public" },
  // financial stress (Fed-produced indexes)
  { id: "STLFSI4", label: "St. Louis Fed Stress Index", cadence: "weekly", unit: "index", license: "public" },
  { id: "NFCI", label: "Chicago Fed Financial Conditions", cadence: "weekly", unit: "index", license: "public" },
  // labor
  { id: "ICSA", label: "Initial Jobless Claims", cadence: "weekly", unit: "claims", license: "public" },
  { id: "CCSA", label: "Continuing Claims", cadence: "weekly", unit: "claims", license: "public" },
  { id: "UNRATE", label: "Unemployment Rate", cadence: "monthly", unit: "%", license: "public" },
  { id: "PAYEMS", label: "Nonfarm Payrolls", cadence: "monthly", unit: "thousands", license: "public" },
  // inflation
  { id: "CPIAUCSL", label: "CPI (All Items)", cadence: "monthly", unit: "index", license: "public" },
  { id: "CPILFESL", label: "Core CPI", cadence: "monthly", unit: "index", license: "public" },
  { id: "PCEPILFE", label: "Core PCE Price Index", cadence: "monthly", unit: "index", license: "public" },
  // activity
  { id: "INDPRO", label: "Industrial Production", cadence: "monthly", unit: "index", license: "public" },
  { id: "HOUST", label: "Housing Starts", cadence: "monthly", unit: "thousands", license: "public" },
  { id: "PERMIT", label: "Building Permits", cadence: "monthly", unit: "thousands", license: "public" },
  { id: "RSAFS", label: "Retail Sales", cadence: "monthly", unit: "$M", license: "public" },
  // money & liquidity
  { id: "M2SL", label: "M2 Money Stock", cadence: "monthly", unit: "$B", license: "public" },
  { id: "WALCL", label: "Fed Balance Sheet", cadence: "weekly", unit: "$M", license: "public" },
  { id: "RRPONTSYD", label: "Overnight Reverse Repo", cadence: "daily", unit: "$B", license: "public" },
  // commodities & dollar (EIA / Fed)
  { id: "DCOILWTICO", label: "WTI Crude Spot", cadence: "daily", unit: "$/bbl", license: "public" },
  { id: "DTWEXBGS", label: "Trade-Weighted Dollar", cadence: "daily", unit: "index", license: "public" },
  // third-party copyrighted — INTERNAL regime use only, never product-surfaced
  { id: "VIXCLS", label: "VIX (CBOE)", cadence: "daily", unit: "index", license: "restricted" },
  { id: "BAMLH0A0HYM2", label: "HY OAS (ICE BofA)", cadence: "daily", unit: "%", license: "restricted" },
  { id: "UMCSENT", label: "Consumer Sentiment (UMich)", cadence: "monthly", unit: "index", license: "restricted" },
];

export const FRED_ATTRIBUTION = "Source: FRED, Federal Reserve Bank of St. Louis";

/** One archived observation. rt = the UTC date WE fetched it (point-in-time
 *  vintage capture; revisions append a new row with a new rt). */
export interface FredObs { s: string; d: string; v: number; rt: string }

export function fredEnabled(env: NodeJS.ProcessEnv = process.env): boolean {
  return Boolean(env.FRED_API_KEY);
}

// ── Parse (documented FRED JSON shape: values are STRINGS, "." = missing) ──

export function parseObservations(seriesId: string, json: any, rtDate: string): FredObs[] {
  const out: FredObs[] = [];
  for (const o of json?.observations || []) {
    if (!o?.date || o.value == null || o.value === ".") continue;
    const v = parseFloat(o.value);
    if (!Number.isFinite(v)) continue;
    out.push({ s: seriesId, d: o.date, v, rt: rtDate });
  }
  return out;
}

// ── Fetch (injectable for tests; FRED limit is 120 req/min — 31 series with
// spacing is far under) ─────────────────────────────────────────────────────

type FetchFn = (url: string, init?: any) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;

async function fetchSeries(
  id: string, apiKey: string, obsStart: string, fetchImpl: FetchFn,
): Promise<any> {
  const url = `https://api.stlouisfed.org/fred/series/observations?series_id=${encodeURIComponent(id)}` +
    `&api_key=${encodeURIComponent(apiKey)}&file_type=json&observation_start=${obsStart}&sort_order=asc`;
  const r = await fetchImpl(url, { signal: AbortSignal.timeout(15000) as any });
  if (!r.ok) throw new Error(`FRED ${id} -> ${r.status}`);
  return JSON.parse(await r.text());
}

// ── Archive (append-only JSONL day-files; dedup by (series,date,value) so
// REVISIONS append as new rows — that is the vintage record) ───────────────

// [REPAIR 2026-07-05, audit defect #6] Vintage dedup is LATEST-VALUE per
// (series, date), not set-of-(s,d,v):
//  - the old Set seeded only 3 days of files while every poll fetches a
//    120-day window — any restart >3 days after backfill re-appended
//    ~120d x 31 series as duplicate rows with a fresh rt (vintage noise);
//  - a revision that REVERTS to a previously-seen value was silently
//    dropped by the (s,d,v) set — but a revert IS a new vintage event.
// Now: seed the CURRENT value per (s,d) from up to 130 days of files
// (covering the fetch window), append only when the value CHANGES —
// duplicates die, every transition (including reverts) is recorded.
const latestVal = new Map<string, number>();
let seeded = false;

/** test hook: simulate a process restart (state re-seeds from disk) */
export function _resetFredArchiveState(): void {
  latestVal.clear();
  seeded = false;
}

function fredDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "fredmacro");
}

function seedSeen(dir: string, nowMs: number): void {
  // oldest -> newest so the LAST write per (s,d) wins — the current vintage
  for (let i = 130; i >= 0; i--) {
    const day = new Date(nowMs - i * 86400_000).toISOString().slice(0, 10);
    for (const fp of [path.join(dir, `${day}.jsonl`), path.join(dir, `${day}.jsonl.gz`)]) {
      let text: string | null = null;
      try {
        text = fp.endsWith(".gz")
          ? zlib.gunzipSync(fs.readFileSync(fp)).toString("utf8")
          : fs.readFileSync(fp, "utf8");
      } catch { continue; }
      for (const line of text.split("\n")) {
        if (!line) continue;
        try { const o = JSON.parse(line); latestVal.set(`${o.s}|${o.d}`, o.v); } catch {}
      }
    }
  }
}

export function archiveFredObs(obs: FredObs[], baseDir?: string, nowMs?: number): number {
  const dir = fredDir(baseDir);
  const now = nowMs ?? Date.now();
  if (!seeded) { seedSeen(dir, now); seeded = true; }
  const fresh = obs.filter((o) => latestVal.get(`${o.s}|${o.d}`) !== o.v);
  if (!fresh.length) return 0;
  try {
    fs.mkdirSync(dir, { recursive: true });
    const fp = path.join(dir, `${new Date(now).toISOString().slice(0, 10)}.jsonl`);
    fs.appendFileSync(fp, fresh.map((o) => JSON.stringify(o)).join("\n") + "\n");
    fresh.forEach((o) => latestVal.set(`${o.s}|${o.d}`, o.v));
    if (latestVal.size > 500_000) latestVal.clear(); // bound memory (~31 series x years)
    return fresh.length;
  } catch (e: any) {
    console.error("[datacore] fredmacro archive:", e?.message || e);
    return 0;
  }
}

export function gzipOldFredDays(baseDir?: string, nowMs?: number): number {
  const dir = fredDir(baseDir);
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

// ── Cache + public payload ──────────────────────────────────────────────────

export interface FredSeriesSnapshot extends FredSeriesDef {
  latest: { d: string; v: number } | null;
  prev: { d: string; v: number } | null;
  /** last ≤30 observations, oldest first — sparkline material */
  history: Array<{ d: string; v: number }>;
}

let cache: { at: number; series: FredSeriesSnapshot[] } | null = null;
let polling = false;

export function latestFredSeries(): { at: number; series: FredSeriesSnapshot[] } | null {
  return cache;
}

/** The public route payload: license:"restricted" series are EXCLUDED —
 *  third-party copyrighted data never reaches a product surface. */
export function buildMacroPayload(hit: { at: number; series: FredSeriesSnapshot[] } | null) {
  if (!hit) return null;
  return {
    kind: "raw" as const,
    source: `FRED API — ${FRED_ATTRIBUTION}`,
    attribution: FRED_ATTRIBUTION,
    time: hit.at,
    note: "Values as currently published by FRED (revisions overwrite here; our archive keeps every vintage as-seen).",
    series: hit.series.filter((s) => s.license === "public")
      .map(({ license: _license, ...pub }) => pub),
  };
}

export async function refreshFredCache(
  env: NodeJS.ProcessEnv = process.env,
  fetchImpl: FetchFn = fetch as any,
  delayMs = 250,
  nowMs?: number,
): Promise<void> {
  const key = env.FRED_API_KEY || "";
  if (!key) return;
  const now = nowMs ?? Date.now();
  const rt = new Date(now).toISOString().slice(0, 10);
  const obsStart = new Date(now - 120 * 86400_000).toISOString().slice(0, 10);
  const snapshots: FredSeriesSnapshot[] = [];
  const allObs: FredObs[] = [];
  for (const def of FRED_SERIES) {
    try {
      const json = await fetchSeries(def.id, key, obsStart, fetchImpl);
      const obs = parseObservations(def.id, json, rt);
      allObs.push(...obs);
      const hist = obs.slice(-30).map(({ d, v }) => ({ d, v }));
      snapshots.push({
        ...def,
        latest: hist[hist.length - 1] || null,
        prev: hist[hist.length - 2] || null,
        history: hist,
      });
    } catch (e: any) {
      console.error("[datacore] fredmacro fetch:", def.id, e?.message || e);
      snapshots.push({ ...def, latest: null, prev: null, history: [] });
    }
    if (delayMs > 0) await new Promise((res) => setTimeout(res, delayMs));
  }
  if (snapshots.some((s) => s.latest) || !cache) cache = { at: now, series: snapshots };
  try { archiveFredObs(allObs, undefined, now); } catch {}
  try { gzipOldFredDays(undefined, now); } catch {}
}

/** 6h poll: daily series update once per business day; the extra cycles
 *  catch weekly/monthly releases and revisions (each revision lands in the
 *  archive as a new vintage row). No-ops without FRED_API_KEY. */
export function bootFredPoll(env: NodeJS.ProcessEnv = process.env, intervalMs = 6 * 60 * 60_000): void {
  if (polling || !fredEnabled(env)) return;
  polling = true;
  refreshFredCache(env);
  setInterval(() => { refreshFredCache(env); }, intervalMs).unref?.();
}
