/**
 * spaceWeather.ts — NOAA SWPC space weather (keyless; open_questions.md
 * SPACE WEATHER hypothesis, gate-1 archive + RAW display layer). RAW
 * overlay per the standing raw-vs-signals rule: official NOAA readings
 * and NOAA's own model forecast displayed as-is with attribution — the
 * trading hypothesis (K-index/G-scale spikes → grid-disturbance reports →
 * exposed-utility moves) stays ladder-gated; this module only RECORDS the
 * evidence gate 1 needs (Kp, scales, alert messages) and shows conditions.
 *
 * services.swpc.noaa.gov: US government work, public domain, no key.
 * Feeds (all live-probed 2026-07-29):
 *  · products/noaa-planetary-k-index.json — OBSERVED 3-hourly planetary Kp
 *    (preliminary estimates; definitive Kp posts later — labeled as such).
 *  · products/noaa-scales.json — R/S/G storm scales: key "0" = current
 *    observed conditions, "1".."3" = NOAA's own 3-day FORECAST rows
 *    (probability fields), "-1" = yesterday. Observed vs predicted is
 *    carried per row, never blended.
 *  · products/alerts.json — official SWPC alert/watch/warning messages
 *    (immutable once issued; dedup by product_id+issue time).
 *  · products/summary/solar-wind-{speed,mag-field}.json — latest solar
 *    wind speed / IMF Bt+Bz one-row summaries (the 1.5–2.5 MB rtsw
 *    streams exist but the summary rows carry what the card needs).
 *  · json/ovation_aurora_latest.json — OVATION Prime aurora MODEL
 *    FORECAST grid (1°, 65k cells). Display-only: served thresholded +
 *    2°-aggregated; the archive keeps a per-poll summary, never the 900 KB
 *    raster (the validation archive needs Kp/scales/alerts, not the oval).
 */

import fs from "fs";
import path from "path";
import zlib from "zlib";
import { archiveBaseDir } from "./datacoreArchive";

const BASE = "https://services.swpc.noaa.gov";
const URLS = {
  kp: `${BASE}/products/noaa-planetary-k-index.json`,
  scales: `${BASE}/products/noaa-scales.json`,
  alerts: `${BASE}/products/alerts.json`,
  windSpeed: `${BASE}/products/summary/solar-wind-speed.json`,
  windMag: `${BASE}/products/summary/solar-wind-mag-field.json`,
  ovation: `${BASE}/json/ovation_aurora_latest.json`,
} as const;

// ── parse: planetary Kp ─────────────────────────────────────────────────────

export interface KpRec {
  t: string; // time_tag (UTC, 3-hour bin start)
  kp: number;
  a: number | null; // a_running
  n: number | null; // station_count
  rt: string; // as-seen UTC date (archive vintage)
}

export function parseKp(json: any, rt: string): KpRec[] {
  const out: KpRec[] = [];
  for (const r of Array.isArray(json) ? json : []) {
    const t = r?.time_tag;
    const kp = Number(r?.Kp);
    if (typeof t !== "string" || !Number.isFinite(kp)) continue;
    out.push({
      t,
      kp,
      a: Number.isFinite(Number(r?.a_running)) ? Number(r.a_running) : null,
      n: Number.isFinite(Number(r?.station_count)) ? Number(r.station_count) : null,
      rt,
    });
  }
  return out;
}

// ── parse: R/S/G scales ─────────────────────────────────────────────────────

export interface ScaleRow {
  /** "current" (key 0, observed now) | "yesterday" (-1) | "forecast" (1..3) */
  kind: "current" | "yesterday" | "forecast";
  date: string | null;
  time: string | null;
  r: string | null; // R scale 0..5 as string, null on forecast rows
  s: string | null;
  g: string | null; // G scale is populated on forecast rows too
  rMinorProb: number | null; // forecast probabilities (%)
  rMajorProb: number | null;
  sProb: number | null;
}

const num = (v: any): number | null => (v == null || v === "" || !Number.isFinite(Number(v)) ? null : Number(v));

export function parseScales(json: any): { current: ScaleRow | null; forecast: ScaleRow[] } {
  const row = (k: string, kind: ScaleRow["kind"]): ScaleRow | null => {
    const r = json?.[k];
    if (!r) return null;
    return {
      kind,
      date: typeof r.DateStamp === "string" ? r.DateStamp : null,
      time: typeof r.TimeStamp === "string" ? r.TimeStamp : null,
      r: r.R?.Scale ?? null,
      s: r.S?.Scale ?? null,
      g: r.G?.Scale ?? null,
      rMinorProb: num(r.R?.MinorProb),
      rMajorProb: num(r.R?.MajorProb),
      sProb: num(r.S?.Prob),
    };
  };
  const forecast: ScaleRow[] = [];
  for (const k of ["1", "2", "3"]) {
    const fr = row(k, "forecast");
    if (fr) forecast.push(fr);
  }
  return { current: row("0", "current"), forecast };
}

// ── parse: SWPC alert messages ──────────────────────────────────────────────

export interface SwpcAlertRec {
  id: string; // product_id + issue_datetime — messages are immutable
  productId: string;
  issued: string;
  code: string | null; // "Space Weather Message Code: X"
  serial: number | null;
  title: string | null; // first ALERT/WARNING/WATCH/SUMMARY line
  message: string;
  rt: string;
}

const MSG_MAX = 2000;

export function parseSwpcAlerts(json: any, rt: string): SwpcAlertRec[] {
  const out: SwpcAlertRec[] = [];
  for (const r of Array.isArray(json) ? json : []) {
    const pid = r?.product_id;
    const issued = r?.issue_datetime;
    const msg = r?.message;
    if (typeof pid !== "string" || typeof issued !== "string" || typeof msg !== "string") continue;
    const code = msg.match(/Space Weather Message Code:\s*([A-Z0-9]+)/)?.[1] ?? null;
    const serialStr = msg.match(/Serial Number:\s*(\d+)/)?.[1];
    const title =
      msg
        .split(/\r?\n/)
        .map((l) => l.trim())
        .find((l) => /^(EXTENDED |CONTINUED |CANCEL )?(WATCH|WARNING|ALERT|SUMMARY)\b/i.test(l)) ?? null;
    out.push({
      id: `${pid}:${issued}`,
      productId: pid,
      issued,
      code,
      serial: serialStr ? Number(serialStr) : null,
      title,
      message: msg.slice(0, MSG_MAX),
      rt,
    });
  }
  return out;
}

// ── parse: solar-wind summary ───────────────────────────────────────────────

export interface WindSummary {
  speedKms: number | null;
  speedAt: string | null;
  btNt: number | null;
  bzNt: number | null;
  magAt: string | null;
}

export function parseWindSummary(speedJson: any, magJson: any): WindSummary {
  const sp = Array.isArray(speedJson) ? speedJson[0] : null;
  const mg = Array.isArray(magJson) ? magJson[0] : null;
  return {
    speedKms: num(sp?.proton_speed),
    speedAt: typeof sp?.time_tag === "string" ? sp.time_tag : null,
    btNt: num(mg?.bt),
    bzNt: num(mg?.bz_gsm),
    magAt: typeof mg?.time_tag === "string" ? mg.time_tag : null,
  };
}

// ── parse: OVATION aurora forecast grid ─────────────────────────────────────

export interface AuroraGrid {
  obs: string | null; // Observation Time (model input)
  forecast: string | null; // Forecast Time — this is a FORECAST product
  max: number; // max probability (%) anywhere on the full grid
  aggDeg: number;
  minVal: number;
  /** [lonWest(-180..180-agg), latSouth, maxProb%] per aggregated cell */
  cells: Array<[number, number, number]>;
}

/**
 * Threshold + aggregate the 1° OVATION grid: keep cells ≥ minVal, fold onto
 * an aggDeg° grid taking the MAX probability (max, not mean — thinning a
 * forecast must never dim it). Input longitudes are 0..359; output cells are
 * west-edge/south-edge anchored in -180..180 for direct polygon building.
 */
export function parseOvation(json: any, aggDeg = 2, minVal = 2): AuroraGrid {
  const coords: any[] = Array.isArray(json?.coordinates) ? json.coordinates : [];
  const agg = new Map<string, [number, number, number]>();
  let max = 0;
  for (const row of coords) {
    const v = Number(row?.[2]);
    if (!Number.isFinite(v)) continue;
    if (v > max) max = v;
    if (v < minVal) continue;
    let lon = Number(row[0]);
    const lat = Number(row[1]);
    if (!Number.isFinite(lon) || !Number.isFinite(lat)) continue;
    if (lon >= 180) lon -= 360; // 0..359 → -180..179
    const lonW = Math.floor(lon / aggDeg) * aggDeg;
    const latS = Math.floor(lat / aggDeg) * aggDeg;
    const key = `${lonW},${latS}`;
    const hit = agg.get(key);
    if (!hit) agg.set(key, [lonW, latS, v]);
    else if (v > hit[2]) hit[2] = v;
  }
  return {
    obs: typeof json?.["Observation Time"] === "string" ? json["Observation Time"] : null,
    forecast: typeof json?.["Forecast Time"] === "string" ? json["Forecast Time"] : null,
    max,
    aggDeg,
    minVal,
    cells: Array.from(agg.values()),
  };
}

// ── fetch (partial-failure tolerant — one dead feed never blanks the rest) ──

type FetchFn = (url: string, init?: any) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;

export interface SpaceWeatherPull {
  kp: KpRec[];
  scales: { current: ScaleRow | null; forecast: ScaleRow[] };
  alerts: SwpcAlertRec[];
  wind: WindSummary;
  aurora: AuroraGrid | null;
  errors: string[];
}

async function getJson(fetchImpl: FetchFn, url: string): Promise<any> {
  const r = await fetchImpl(url, {
    headers: { "User-Agent": "voltradeai-datacore/1.0 (mctils12@gmail.com)" },
    signal: AbortSignal.timeout(30000) as any,
  });
  if (!r.ok) throw new Error(`${url.slice(BASE.length + 1)} -> ${r.status}`);
  return JSON.parse(await r.text());
}

export async function fetchSpaceWeather(fetchImpl: FetchFn = fetch as any, nowMs?: number): Promise<SpaceWeatherPull> {
  const rt = new Date(nowMs ?? Date.now()).toISOString().slice(0, 10);
  const [kpR, scR, alR, wsR, wmR, ovR] = await Promise.allSettled([
    getJson(fetchImpl, URLS.kp),
    getJson(fetchImpl, URLS.scales),
    getJson(fetchImpl, URLS.alerts),
    getJson(fetchImpl, URLS.windSpeed),
    getJson(fetchImpl, URLS.windMag),
    getJson(fetchImpl, URLS.ovation),
  ]);
  const errors: string[] = [];
  const val = <T,>(r: PromiseSettledResult<any>, f: (j: any) => T, empty: T): T => {
    if (r.status === "fulfilled") {
      try { return f(r.value); } catch (e: any) { errors.push(String(e?.message || e)); return empty; }
    }
    errors.push(String(r.reason?.message || r.reason));
    return empty;
  };
  return {
    kp: val(kpR, (j) => parseKp(j, rt), []),
    scales: val(scR, parseScales, { current: null, forecast: [] }),
    alerts: val(alR, (j) => parseSwpcAlerts(j, rt), []),
    wind: parseWindSummary(
      wsR.status === "fulfilled" ? wsR.value : (errors.push(String(wsR.reason?.message || wsR.reason)), null),
      wmR.status === "fulfilled" ? wmR.value : (errors.push(String(wmR.reason?.message || wmR.reason)), null),
    ),
    aurora: val(ovR, (j) => parseOvation(j), null),
    errors,
  };
}

// ── archive (JSONL per day under spaceweather/; dedup Sets, never clear) ────

function swDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "spaceweather");
}

const seenKp = new Set<string>();
const seenAlerts = new Set<string>();
const seenCond = new Set<string>();
let seeded = false;

function seedPrefix(dir: string, prefix: string, set: Set<string>, idOf: (r: any) => string, nowMs: number): void {
  for (let i = 0; i < 3; i++) {
    const day = new Date(nowMs - i * 86400_000).toISOString().slice(0, 10);
    for (const fp of [path.join(dir, `${prefix}-${day}.jsonl`), path.join(dir, `${prefix}-${day}.jsonl.gz`)]) {
      let text: string | null = null;
      try {
        text = fp.endsWith(".gz")
          ? zlib.gunzipSync(fs.readFileSync(fp)).toString("utf8")
          : fs.readFileSync(fp, "utf8");
      } catch { continue; }
      for (const line of text.split("\n")) {
        if (!line) continue;
        try { set.add(idOf(JSON.parse(line))); } catch {}
      }
    }
  }
}

function trimSet(set: Set<string>): void {
  if (set.size <= 100_000) return;
  // oldest-half trim, never clear() (nasaFirms lesson, audit #10b)
  const it = set.values();
  const drop = set.size >> 1;
  for (let i = 0; i < drop; i++) {
    const v = it.next();
    if (v.done) break;
    set.delete(v.value);
  }
}

function appendFresh<T>(
  dir: string, prefix: string, set: Set<string>, idOf: (r: T) => string, recs: T[], nowMs: number,
): number {
  const fresh = recs.filter((r) => !set.has(idOf(r)));
  if (!fresh.length) return 0;
  fs.mkdirSync(dir, { recursive: true });
  const fp = path.join(dir, `${prefix}-${new Date(nowMs).toISOString().slice(0, 10)}.jsonl`);
  fs.appendFileSync(fp, fresh.map((r) => JSON.stringify(r)).join("\n") + "\n");
  fresh.forEach((r) => set.add(idOf(r)));
  trimSet(set);
  return fresh.length;
}

/** One compact conditions row per distinct upstream stamp — the gate-1
 *  evidence trail (Kp latest + scales + wind + aurora summary), never the
 *  900 KB ovation raster. */
export interface ConditionsRec {
  id: string; // composite upstream stamp — dedup key
  at: string;
  kpT: string | null;
  kp: number | null;
  r: string | null;
  s: string | null;
  g: string | null;
  windKms: number | null;
  btNt: number | null;
  bzNt: number | null;
  auroraMax: number | null;
  auroraObs: string | null;
}

export function conditionsRow(pull: SpaceWeatherPull, nowMs: number): ConditionsRec {
  const kpLast = pull.kp.length ? pull.kp[pull.kp.length - 1] : null;
  const cur = pull.scales.current;
  return {
    id: [kpLast?.t ?? "", cur?.date ?? "", cur?.time ?? "", pull.wind.speedAt ?? "", pull.wind.magAt ?? "", pull.aurora?.obs ?? ""].join("|"),
    at: new Date(nowMs).toISOString(),
    kpT: kpLast?.t ?? null,
    kp: kpLast?.kp ?? null,
    r: cur?.r ?? null,
    s: cur?.s ?? null,
    g: cur?.g ?? null,
    windKms: pull.wind.speedKms,
    btNt: pull.wind.btNt,
    bzNt: pull.wind.bzNt,
    auroraMax: pull.aurora?.max ?? null,
    auroraObs: pull.aurora?.obs ?? null,
  };
}

export function archiveSpaceWeather(pull: SpaceWeatherPull, baseDir?: string, nowMs?: number): { kp: number; alerts: number; conditions: number } {
  const dir = swDir(baseDir);
  const now = nowMs ?? Date.now();
  if (!seeded) {
    seedPrefix(dir, "kp", seenKp, (r) => r.t, now);
    seedPrefix(dir, "alerts", seenAlerts, (r) => r.id, now);
    seedPrefix(dir, "conditions", seenCond, (r) => r.id, now);
    seeded = true;
  }
  try {
    return {
      kp: appendFresh(dir, "kp", seenKp, (r: KpRec) => r.t, pull.kp, now),
      alerts: appendFresh(dir, "alerts", seenAlerts, (r: SwpcAlertRec) => r.id, pull.alerts, now),
      conditions: appendFresh(dir, "conditions", seenCond, (r: ConditionsRec) => r.id, [conditionsRow(pull, now)], now),
    };
  } catch (e: any) {
    console.error("[datacore] spaceweather archive:", e?.message || e);
    return { kp: 0, alerts: 0, conditions: 0 };
  }
}

export function gzipOldSpaceWeatherDays(baseDir?: string, nowMs?: number): number {
  const dir = swDir(baseDir);
  const now = nowMs ?? Date.now();
  let n = 0;
  try {
    for (const f of fs.readdirSync(dir)) {
      const m = f.match(/(\d{4}-\d{2}-\d{2})\.jsonl$/);
      if (!m) continue;
      if (now - Date.parse(m[1]) < 2 * 86400_000) continue;
      const fp = path.join(dir, f);
      fs.writeFileSync(`${fp}.gz`, zlib.gzipSync(fs.readFileSync(fp)));
      fs.unlinkSync(fp);
      n++;
    }
  } catch {}
  return n;
}

// ── cache + poll ────────────────────────────────────────────────────────────

export interface SpaceWeatherCache {
  at: number;
  kpRecent: KpRec[]; // last 8 bins (24 h)
  scales: { current: ScaleRow | null; forecast: ScaleRow[] };
  wind: WindSummary;
  aurora: AuroraGrid | null;
  alertsRecent: SwpcAlertRec[]; // newest first, capped
  errors: string[];
}

let cache: SpaceWeatherCache | null = null;
let polling = false;

export function latestSpaceWeather(): SpaceWeatherCache | null {
  return cache;
}

const ALERTS_RECENT = 20;

export async function refreshSpaceWeatherCache(fetchImpl: FetchFn = fetch as any, nowMs?: number): Promise<void> {
  try {
    const pull = await fetchSpaceWeather(fetchImpl, nowMs);
    const gotAnything = pull.kp.length || pull.scales.current || pull.aurora || pull.alerts.length;
    if (gotAnything || !cache) {
      cache = {
        at: Date.now(),
        kpRecent: pull.kp.slice(-8),
        scales: pull.scales,
        wind: pull.wind,
        aurora: pull.aurora ?? cache?.aurora ?? null, // keep last good oval across a single feed hiccup
        alertsRecent: [...pull.alerts].sort((a, b) => (a.issued < b.issued ? 1 : -1)).slice(0, ALERTS_RECENT),
        errors: pull.errors,
      };
    }
    try { archiveSpaceWeather(pull, undefined, nowMs); } catch {}
    try { gzipOldSpaceWeatherDays(undefined, nowMs); } catch {}
  } catch (e: any) {
    console.error("[datacore] spaceweather refresh:", e?.message || e);
  }
}

/** 10-min poll (ovation updates ~5-min upstream, Kp 3-hourly; 10 min keeps
 *  us current without hammering). Eager boot — KNOWN BROKEN #9 rule:
 *  pollers fire once at startup, never wait an interval. */
export function bootSpaceWeatherPoll(intervalMs = 10 * 60_000): void {
  if (polling) return;
  polling = true;
  refreshSpaceWeatherCache();
  setInterval(() => { refreshSpaceWeatherCache(); }, intervalMs).unref?.();
}
