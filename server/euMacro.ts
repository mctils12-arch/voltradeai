/**
 * euMacro.ts — European macro regime feed: ECB + Eurostat + Bundesbank
 * (DATACORE MAXIMUS census build #7; all three contracts LIVE-VERIFIED
 * 2026-07-07 by the source workup — every quirk below was probed).
 *
 * HYPOTHESIS (same as fredMacro, gate-locked): REGIME INPUT only —
 * weekly Eurosystem balance-sheet level is the least-crowded piece;
 * never traded alone.
 *
 * Pattern: fredMacro.ts clone, KEYLESS (all three sources free with
 * attribution, commercial reuse permitted — exact strings verified from
 * each license doc and carried per-series). Vintage honesty identical:
 * every observation carries rt; revisions append as new rows (ILM and
 * Eurostat revise in place — documented).
 *
 * VERIFIED SOURCE QUIRKS (coding consequences from the workup):
 *  - ECB: format=csvdata; column POSITIONS differ per dataflow — parse
 *    by HEADER NAME (TIME_PERIOD/OBS_VALUE only). 4xx bodies can be
 *    HTML — branch on status, never parse error bodies. ILM periods
 *    are ISO weeks ("2026-W26") → normalized to the statement's Friday
 *    reference date (the ECB publishes Tuesday for the preceding
 *    Friday), raw week kept alongside.
 *  - Eurostat: JSON-stat 2.0; with all non-time dims pinned, the flat
 *    index IS the time index; `value` is SPARSE (listed period, no data
 *    yet — verified live) and a WRONG dimension value returns a SILENT
 *    200 with empty value{} — an empty result is treated as a fetch
 *    failure and logged loudly, never cached as truth. status flags
 *    (e.g. "i" = imputed) carried into the archive.
 *  - Bundesbank: format=json; observation values are STRINGS ("2.98");
 *    missing days present as [null, flag] rows — skipped, never zeroed;
 *    a 200 can contain zero data rows — count parsed observations.
 */

import fs from "fs";
import path from "path";
import zlib from "zlib";
import { archiveBaseDir } from "./datacoreArchive";

export interface EuSeriesDef {
  key: string;                       // our stable series id in the archive
  source: "ecb" | "eurostat" | "bbk";
  label: string;
  cadence: "daily" | "weekly" | "monthly";
  unit: string;
  attribution: string;
}

export const ECB_ATTRIBUTION = "Source: ECB statistics.";
export const EUROSTAT_ATTRIBUTION = "Source: Eurostat, sts_inpr_m";
export const BBK_ATTRIBUTION = "Source: Deutsche Bundesbank";

export const EU_SERIES: EuSeriesDef[] = [
  { key: "ECB_EURUSD", source: "ecb", label: "EUR/USD reference rate", cadence: "daily", unit: "USD", attribution: ECB_ATTRIBUTION },
  { key: "ECB_ESTR", source: "ecb", label: "€STR (euro short-term rate)", cadence: "daily", unit: "%", attribution: ECB_ATTRIBUTION },
  { key: "ECB_BS_TOTAL", source: "ecb", label: "Eurosystem balance sheet (total assets)", cadence: "weekly", unit: "EUR millions", attribution: ECB_ATTRIBUTION },
  { key: "EU_INDPROD", source: "eurostat", label: "EA20 industrial production (SCA, 2021=100)", cadence: "monthly", unit: "index", attribution: EUROSTAT_ATTRIBUTION },
  { key: "DE_BUND10Y", source: "bbk", label: "10Y Bund yield (Svensson, listed Federal securities)", cadence: "daily", unit: "%", attribution: BBK_ATTRIBUTION },
];

// Verified request URLs (workup 2026-07-07)
const ECB_URLS: Record<string, string> = {
  ECB_EURUSD: "https://data-api.ecb.europa.eu/service/data/EXR/D.USD.EUR.SP00.A?format=csvdata&lastNObservations=30",
  ECB_ESTR: "https://data-api.ecb.europa.eu/service/data/EST/B.EU000A2X2A25.WT?format=csvdata&lastNObservations=30",
  ECB_BS_TOTAL: "https://data-api.ecb.europa.eu/service/data/ILM/W.U2.C.T000000.Z5.Z01?format=csvdata&lastNObservations=30",
};
const BBK_URL = "https://api.statistiken.bundesbank.de/rest/data/BBSIS/D.I.ZAR.ZI.EUR.S1311.B.A604.R10XX.R.A.A._Z._Z.A?format=json&lastNObservations=30";
const eurostatUrl = (since: string) =>
  `https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/sts_inpr_m?format=JSON&lang=EN&geo=EA20&nace_r2=B-D&s_adj=SCA&unit=I21&sinceTimePeriod=${since}`;

/** One archived observation; wk = raw ISO week for weekly ECB series;
 *  flag = source status flag (Eurostat "i" = imputed) when present. */
export interface EuObs { s: string; d: string; v: number; rt: string; wk?: string; flag?: string }

// ── Parsers (each shape verified live) ──────────────────────────────────────

/** ISO week (e.g. 2026, 26) → that week's FRIDAY as ISO date. The ECB
 *  weekly statement "relates to the preceding Friday" — Friday is the
 *  calendar anchor downstream regime features join on. */
export function isoWeekFriday(year: number, week: number): string {
  // ISO: week 1 contains Jan 4; weeks start Monday.
  const jan4 = Date.UTC(year, 0, 4);
  const jan4Dow = new Date(jan4).getUTCDay() || 7; // 1=Mon..7=Sun
  const week1Monday = jan4 - (jan4Dow - 1) * 86400_000;
  const friday = week1Monday + ((week - 1) * 7 + 4) * 86400_000;
  return new Date(friday).toISOString().slice(0, 10);
}

/** ECB SDMX-CSV: header-name parse of TIME_PERIOD/OBS_VALUE (column
 *  positions vary per dataflow — verified 31/30/29 cols). Weekly
 *  periods ("2026-W26") normalize to the Friday reference date. */
export function parseEcbCsv(seriesKey: string, text: string, rt: string): EuObs[] {
  const lines = text.split(/\r?\n/).filter((l) => l.length);
  if (lines.length < 2) return [];
  const header = lines[0].split(",");
  const ti = header.indexOf("TIME_PERIOD");
  const vi = header.indexOf("OBS_VALUE");
  if (ti < 0 || vi < 0) {
    console.error(`[datacore] eumacro ${seriesKey}: TIME_PERIOD/OBS_VALUE not in header — shape changed?`);
    return [];
  }
  const out: EuObs[] = [];
  for (const line of lines.slice(1)) {
    const p = line.split(",");
    const period = p[ti], raw = p[vi];
    if (!period || raw == null || raw === "") continue;
    const v = parseFloat(raw);
    if (!Number.isFinite(v)) continue;
    const wm = period.match(/^(\d{4})-W(\d{2})$/);
    if (wm) {
      out.push({ s: seriesKey, d: isoWeekFriday(parseInt(wm[1], 10), parseInt(wm[2], 10)), v, rt, wk: period });
    } else {
      out.push({ s: seriesKey, d: period, v, rt });
    }
  }
  out.sort((a, b) => (a.d < b.d ? -1 : 1));
  return out;
}

/** Eurostat JSON-stat 2.0: with all non-time dims pinned, the flat index
 *  is the time index; value{} is SPARSE; status flags carried. Returns
 *  null when the response is the verified silent-empty case (bad
 *  dimension or nothing published) — the caller alarms. */
export function parseEurostatJsonStat(seriesKey: string, json: any, rt: string): EuObs[] | null {
  const timeIndex = json?.dimension?.time?.category?.index;
  const values = json?.value;
  if (!timeIndex || !values || Object.keys(values).length === 0) return null;
  const status = json?.status || {};
  const out: EuObs[] = [];
  for (const [period, idx] of Object.entries(timeIndex)) {
    const v = values[String(idx)];
    if (v == null) continue; // sparse: listed period, no data yet — honest skip
    const n = Number(v);
    if (!Number.isFinite(n)) continue;
    const flag = status[String(idx)];
    out.push({ s: seriesKey, d: `${period}-01`, v: n, rt, ...(flag ? { flag } : {}) });
  }
  out.sort((a, b) => (a.d < b.d ? -1 : 1));
  return out;
}

/** Bundesbank SDMX-JSON: values are STRINGS, nulls are placeholder rows
 *  for non-business days — skipped, never zeroed. */
export function parseBbkJson(seriesKey: string, json: any, rt: string): EuObs[] {
  const ds = json?.data?.dataSets?.[0]?.series || {};
  const firstKey = Object.keys(ds)[0];
  const observations = firstKey ? ds[firstKey]?.observations || {} : {};
  const times = json?.data?.structure?.dimensions?.observation?.[0]?.values || [];
  const out: EuObs[] = [];
  for (const [idx, arr] of Object.entries(observations)) {
    const raw = Array.isArray(arr) ? arr[0] : null;
    if (raw == null) continue;
    const v = parseFloat(String(raw));
    if (!Number.isFinite(v)) continue;
    const d = times[parseInt(idx, 10)]?.id;
    if (!d) continue;
    out.push({ s: seriesKey, d, v, rt });
  }
  out.sort((a, b) => (a.d < b.d ? -1 : 1));
  return out;
}

// ── Fetch (injectable; never JSON.parse an error body — ECB 4xx is HTML) ────

type FetchFn = (url: string, init?: any) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;
const UA = { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)" };

async function fetchText(url: string, fetchImpl: FetchFn): Promise<string | null> {
  const r = await fetchImpl(url, { headers: UA, signal: AbortSignal.timeout(30000) as any });
  const text = await r.text();
  if (!r.ok) {
    console.error(`[datacore] eumacro ${url.slice(0, 80)} -> ${r.status}: ${text.slice(0, 160)}`);
    return null;
  }
  return text;
}

// ── Archive (fredMacro vintage mechanism: latest-value per (s,d); every
// transition including reverts appends as a new vintage row) ────────────────

const latestVal = new Map<string, number>();
let seeded = false;

export function _resetEuMacroForTests(): void {
  latestVal.clear();
  seeded = false;
  cache = null;
  polling = false;
}

function euDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "eumacro");
}

function seedSeen(dir: string, nowMs: number): void {
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

export function archiveEuObs(obs: EuObs[], baseDir?: string, nowMs?: number): number {
  const dir = euDir(baseDir);
  const now = nowMs ?? Date.now();
  if (!seeded) { seedSeen(dir, now); seeded = true; }
  const fresh = obs.filter((o) => latestVal.get(`${o.s}|${o.d}`) !== o.v);
  if (!fresh.length) return 0;
  try {
    fs.mkdirSync(dir, { recursive: true });
    const fp = path.join(dir, `${new Date(now).toISOString().slice(0, 10)}.jsonl`);
    fs.appendFileSync(fp, fresh.map((o) => JSON.stringify(o)).join("\n") + "\n");
    fresh.forEach((o) => latestVal.set(`${o.s}|${o.d}`, o.v));
    if (latestVal.size > 100_000) latestVal.clear(); // 5 series x years — generous bound
    return fresh.length;
  } catch (e: any) {
    console.error("[datacore] eumacro archive:", e?.message || e);
    return 0;
  }
}

export function gzipOldEuDays(baseDir?: string, nowMs?: number): number {
  const dir = euDir(baseDir);
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

// ── Cache + payload ─────────────────────────────────────────────────────────

export interface EuSeriesSnapshot extends EuSeriesDef {
  latest: { d: string; v: number } | null;
  prev: { d: string; v: number } | null;
  history: Array<{ d: string; v: number }>;
}

let cache: { at: number; series: EuSeriesSnapshot[] } | null = null;
let polling = false;

export function latestEuMacro(): { at: number; series: EuSeriesSnapshot[] } | null {
  return cache;
}

export async function refreshEuMacro(
  fetchImpl: FetchFn = fetch as any,
  nowMs?: number,
  baseDir?: string,
  delayMs = 300,
): Promise<void> {
  const now = nowMs ?? Date.now();
  const rt = new Date(now).toISOString().slice(0, 10);
  const since = new Date(now - 200 * 86400_000).toISOString().slice(0, 7); // ~6 months of monthlies
  const snapshots: EuSeriesSnapshot[] = [];
  const allObs: EuObs[] = [];
  for (const def of EU_SERIES) {
    let obs: EuObs[] = [];
    try {
      if (def.source === "ecb") {
        const text = await fetchText(ECB_URLS[def.key], fetchImpl);
        if (text !== null) obs = parseEcbCsv(def.key, text, rt);
      } else if (def.source === "eurostat") {
        const text = await fetchText(eurostatUrl(since), fetchImpl);
        if (text !== null) {
          const parsed = parseEurostatJsonStat(def.key, JSON.parse(text), rt);
          if (parsed === null) {
            // the verified SILENT-EMPTY 200 (bad dim or nothing published)
            console.error(`[datacore] eumacro ${def.key}: silent-empty Eurostat response — filter drift or publication gap; NOT cached as truth`);
          } else obs = parsed;
        }
      } else {
        const text = await fetchText(BBK_URL, fetchImpl);
        if (text !== null) {
          obs = parseBbkJson(def.key, JSON.parse(text), rt);
          if (!obs.length) console.error(`[datacore] eumacro ${def.key}: 200 with zero parsed observations`);
        }
      }
    } catch (e: any) {
      console.error(`[datacore] eumacro ${def.key}:`, e?.message || e);
    }
    allObs.push(...obs);
    const hist = obs.slice(-30).map(({ d, v }) => ({ d, v }));
    snapshots.push({
      ...def,
      latest: hist[hist.length - 1] || null,
      prev: hist[hist.length - 2] || null,
      history: hist,
    });
    if (delayMs > 0) await new Promise((res) => setTimeout(res, delayMs));
  }
  if (snapshots.some((s) => s.latest) || !cache) cache = { at: now, series: snapshots };
  try { archiveEuObs(allObs, baseDir, now); } catch {}
  try { gzipOldEuDays(baseDir, now); } catch {}
}

/** 6h poll catches every source's publish window within a business day:
 *  €STR 08:00 CET, BBK ~midday CET, EXR 16:00 CET, ILM Tuesday PM,
 *  Eurostat monthly ~t+35d. Keyless — always on. Eager boot. */
export function bootEuMacroPoll(intervalMs = 6 * 60 * 60_000): void {
  if (polling) return;
  polling = true;
  refreshEuMacro().catch((e) => console.error("[datacore] eumacro boot:", e?.message || e));
  setInterval(() => { refreshEuMacro(); }, intervalMs).unref?.();
}
