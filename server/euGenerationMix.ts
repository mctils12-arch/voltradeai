/**
 * euGenerationMix.ts — ENTSO-E actual generation per production type
 * (fuel mix), hourly by EU bidding zone (wishlist 9c follow-up, filed
 * 2026-07-07 alongside euLoad.ts, built this session — same token,
 * separate PR per one-logical-change-per-PR).
 *
 * Source: same ENTSO-E Transparency Platform RESTful API host/auth as
 * euLoad.ts (web-api.tp.entsoe.eu/api) — documentType=A75 ("Actual
 * generation per type"), processType=A16 (realised), in_Domain=<EIC>
 * (NOT outBiddingZone_Domain — a different query parameter than load).
 * Both A65 (load) and A75 (generation) share the same GL_MarketDocument
 * response schema family (urn:iec62325.351:tc57wg16:451-6:
 * generationloaddocument:3:0) per ENTSO-E's XSD — confirmed against the
 * ecosystem-canonical entsoe-py library's query params and PSRTYPE
 * mapping this session; NOT yet confirmed against a live authenticated
 * response (this sandbox has no ENTSOE_API_KEY — only Railway does).
 * HONESTY NOTE / NEXT CHECK: once deployed, read `/api/data/eu-
 * generation-mix`'s `issues` field for the first sweep — an unexpected
 * ack or an empty-but-200 response would mean the documentType/param
 * contract needs adjustment despite the entsoe-py cross-check. The
 * per-TimeSeries difference from load is the added <MktPSRType>
 * <psrType> child identifying the fuel/technology; each TimeSeries here
 * is ONE zone x ONE psrType x the requested window (a single call can
 * return many TimeSeries, one per fuel type active in that zone/window).
 *
 * PSRTYPE_MAPPINGS below is the entsoe-py canonical code table (same
 * provenance convention as euLoad.ts's ZONES/EIC table).
 *
 * Scope v1: same 8 country-level zones as euLoad.ts. Rate limit 400
 * req/min; the 2h poll's 8 spaced calls (now generation, separate from
 * load's own 8) stays nowhere near it.
 *
 * HYPOTHESIS (gate-locked; joins eumacro census #7 + GEM/OSM power
 * graph): zonal fuel-mix share (renewables vs. fossil gas/coal, hourly)
 * is a direct gas-burn/carbon-price demand proxy and a ground-truth
 * cross-check against GEM's announced/under-construction capacity
 * mix — the EU analog of EPA CAMD's US unit-level utilization truth.
 * RAW archive only until ladder gates; nothing traded or sold on this
 * alone.
 */

import fs from "fs";
import path from "path";
import zlib from "zlib";
import { archiveBaseDir } from "./datacoreArchive";
import { ZONES } from "./euLoad";

export function euGenerationMixEnabled(env: NodeJS.ProcessEnv = process.env): boolean {
  return Boolean(env.ENTSOE_API_KEY || env.ENTSOE_TOKEN);
}

export { ZONES };

/** entsoe-py canonical PSR type -> human fuel/technology name. */
export const PSRTYPE_MAPPINGS: Record<string, string> = {
  A03: "Mixed", A04: "Generation", A05: "Load",
  B01: "Biomass", B02: "Fossil Brown coal/Lignite", B03: "Fossil Coal-derived gas",
  B04: "Fossil Gas", B05: "Fossil Hard coal", B06: "Fossil Oil",
  B07: "Fossil Oil shale", B08: "Fossil Peat", B09: "Geothermal",
  B10: "Hydro Pumped Storage", B11: "Hydro Run-of-river and poundage",
  B12: "Hydro Water Reservoir", B13: "Marine", B14: "Nuclear",
  B15: "Other renewable", B16: "Solar", B17: "Waste",
  B18: "Wind Offshore", B19: "Wind Onshore", B20: "Other",
  B21: "AC Link", B22: "DC Link", B23: "Substation", B24: "Transformer",
  B25: "Energy storage",
};

export const HOURS_PER_FETCH = 48;
const CALL_SPACING_MS = 1000;

export interface GenMixObs {
  zone: string;
  psr: string;          // PSRTYPE_MAPPINGS code, e.g. "B16"
  ts: string;
  mw: number | null;
  res: string;
  rt: string;
}

const RES_MINUTES: Record<string, number> = { PT15M: 15, PT30M: 30, PT60M: 60, P1D: 1440 };

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

function pad(n: number): string { return String(n).padStart(2, "0"); }

export function periodStamp(ms: number): string {
  const d = new Date(ms);
  return `${d.getUTCFullYear()}${pad(d.getUTCMonth() + 1)}${pad(d.getUTCDate())}${pad(d.getUTCHours())}${pad(d.getUTCMinutes())}`;
}

/** in_Domain (not outBiddingZone_Domain) — the A75 query contract per
 *  entsoe-py's query_generation, cross-checked this session. */
export function genMixUrl(eic: string, key: string, startMs: number, endMs: number): string {
  return "https://web-api.tp.entsoe.eu/api" +
    `?securityToken=${encodeURIComponent(key)}` +
    "&documentType=A75&processType=A16" +
    `&in_Domain=${encodeURIComponent(eic)}` +
    `&periodStart=${periodStamp(startMs)}&periodEnd=${periodStamp(endMs)}`;
}

const tagText = (block: string, tag: string): string | null => {
  const m = block.match(new RegExp(`<${tag}[^>]*>([\\s\\S]*?)<\\/${tag}>`, "i"));
  return m ? m[1].trim() : null;
};
const tagBlocks = (xml: string, tag: string): string[] => {
  const out: string[] = [];
  const re = new RegExp(`<${tag}[^>]*>([\\s\\S]*?)<\\/${tag}>`, "gi");
  let m;
  while ((m = re.exec(xml))) out.push(m[1]);
  return out;
};

export function parseAck(xml: string): string | null {
  if (!/Acknowledgement_MarketDocument/i.test(xml)) return null;
  const reason = tagBlocks(xml, "Reason")[0] || "";
  return (tagText(reason, "text") || tagText(reason, "code") || "unknown acknowledgement").slice(0, 200);
}

/** GL_MarketDocument -> GenMixObs[]. Each TimeSeries carries its own
 *  MktPSRType/psrType — series with an unrecognized or missing psrType
 *  are skipped (never guessed) rather than mis-attributed to a fuel. */
export function parseGenMix(xml: string, zone: string, rt: string): GenMixObs[] {
  if (!/GL_MarketDocument/i.test(xml)) return [];
  const out: GenMixObs[] = [];
  for (const series of tagBlocks(xml, "TimeSeries")) {
    const psrBlock = tagBlocks(series, "MktPSRType")[0];
    const psr = psrBlock ? tagText(psrBlock, "psrType") : null;
    if (!psr || !PSRTYPE_MAPPINGS[psr]) continue;
    for (const period of tagBlocks(series, "Period")) {
      const start = tagText(period, "start");
      const res = tagText(period, "resolution") || "";
      const stepMin = RES_MINUTES[res];
      if (!start || !stepMin) continue;
      const startMs = Date.parse(start);
      if (!Number.isFinite(startMs)) continue;
      for (const point of tagBlocks(period, "Point")) {
        const pos = parseInt(tagText(point, "position") || "", 10);
        const qty = tagText(point, "quantity");
        if (!Number.isFinite(pos) || pos < 1) continue;
        const n = qty == null || qty === "" ? null : parseFloat(qty);
        const ts = new Date(startMs + (pos - 1) * stepMin * 60_000)
          .toISOString().slice(0, 16);
        out.push({ zone, psr, ts, mw: n != null && Number.isFinite(n) ? n : null, res, rt });
      }
    }
  }
  return out;
}

// ── Fetch (token never logged) ──────────────────────────────────────────────

type FetchFn = (url: string, init?: any) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;

const lastIssues: Record<string, string> = {};

export function sweepIssues(): Record<string, string> {
  return { ...lastIssues };
}

export async function fetchGenMix(fetchImpl: FetchFn = fetch as any,
                                   env: NodeJS.ProcessEnv = process.env,
                                   nowMs?: number, spacingMs = CALL_SPACING_MS): Promise<GenMixObs[]> {
  const key = env.ENTSOE_API_KEY || env.ENTSOE_TOKEN || "";
  if (!key) return [];
  const now = nowMs ?? Date.now();
  const rt = new Date(now).toISOString().slice(0, 10);
  const out: GenMixObs[] = [];
  for (const k of Object.keys(lastIssues)) delete lastIssues[k];
  for (const [zone, eic] of Object.entries(ZONES)) {
    try {
      const r = await fetchImpl(genMixUrl(eic, key, now - HOURS_PER_FETCH * 3600_000, now), {
        headers: { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)" },
        signal: AbortSignal.timeout(30000) as any,
      });
      const body = await r.text();
      if (!r.ok) {
        lastIssues[zone] = `http ${r.status}: ${parseAck(body) || ""}`.trim().slice(0, 200);
        console.error(`[datacore] eugenmix ${zone} -> ${lastIssues[zone]}`);
      } else {
        const ack = parseAck(body);
        if (ack) {
          lastIssues[zone] = `ack: ${ack}`;
          console.error(`[datacore] eugenmix ${zone} ${lastIssues[zone]}`);
        } else {
          const rows = parseGenMix(body, zone, rt);
          if (!rows.length) lastIssues[zone] = "empty document (no parseable points)";
          out.push(...rows);
        }
      }
    } catch (e: any) {
      lastIssues[zone] = String(e?.message || e).slice(0, 200);
      console.error(`[datacore] eugenmix ${zone}:`, lastIssues[zone]);
    }
    if (spacingMs > 0) await sleep(spacingMs);
  }
  return out;
}

// ── Archive (dedup zone|psr|ts|res|VALUE — vintage capture; day-file per UTC day) ──

const seenObs = new Set<string>();
let seeded = false;

const obsKey = (o: GenMixObs) => `${o.zone}|${o.psr}|${o.ts}|${o.res}|${o.mw}`;

function genMixDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "eugenmix");
}

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

export function archiveGenMix(obs: GenMixObs[], baseDir?: string): number {
  if (!obs.length) return 0;
  const dir = genMixDir(baseDir);
  if (!seeded) { seedSeen(dir); seeded = true; }
  const fresh = obs.filter((o) => !seenObs.has(obsKey(o)));
  if (!fresh.length) return 0;
  try {
    fs.mkdirSync(dir, { recursive: true });
    const byDay = new Map<string, GenMixObs[]>();
    for (const o of fresh) {
      const day = o.ts.slice(0, 10);
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
    console.error("[datacore] eugenmix archive:", e?.message || e);
    return 0;
  }
}

export function gzipOldGenMixDays(baseDir?: string, nowMs?: number): number {
  const dir = genMixDir(baseDir);
  const now = nowMs ?? Date.now();
  let n = 0;
  try {
    for (const f of fs.readdirSync(dir)) {
      if (!f.endsWith(".jsonl")) continue;
      if (now - Date.parse(f.slice(0, 10)) < 3 * 86400_000) continue;
      const fp = path.join(dir, f);
      const gzPath = `${fp}.gz`;
      const prior = fs.existsSync(gzPath) ? zlib.gunzipSync(fs.readFileSync(gzPath)) : Buffer.alloc(0);
      fs.writeFileSync(gzPath, zlib.gzipSync(Buffer.concat([prior, fs.readFileSync(fp)])));
      fs.unlinkSync(fp);
      n++;
    }
  } catch {}
  return n;
}

// ── Cache + poll ────────────────────────────────────────────────────────────

export interface GenMixStat {
  zone: string;
  psr: string;
  psr_name: string;
  latest_ts: string;
  latest_mw: number | null;
  resolution: string;
  points_in_window: number;
  window_min_mw: number | null;
  window_max_mw: number | null;
  window_mean_mw: number | null;
}

let cache: { at: number; stats: GenMixStat[]; issues: Record<string, string> } | null = null;
let polling = false;

export function latestGenMix() {
  return cache;
}

export async function refreshGenMix(fetchImpl: FetchFn = fetch as any,
                                     env: NodeJS.ProcessEnv = process.env,
                                     nowMs?: number, baseDir?: string,
                                     spacingMs = CALL_SPACING_MS): Promise<void> {
  try {
    if (!euGenerationMixEnabled(env)) return;
    const obs = await fetchGenMix(fetchImpl, env, nowMs, spacingMs);
    if (obs.length) {
      archiveGenMix(obs, baseDir);
      const byKey = new Map<string, GenMixObs[]>();
      for (const o of obs) {
        const k = `${o.zone}|${o.psr}`;
        if (!byKey.has(k)) byKey.set(k, []);
        byKey.get(k)!.push(o);
      }
      const stats: GenMixStat[] = [];
      byKey.forEach((rows, key) => {
        const [zone, psr] = key.split("|");
        const newest = rows.reduce((mx, r) => (r.ts > mx.ts ? r : mx), rows[0]);
        const vals = rows.map((r) => r.mw).filter((v): v is number => v != null);
        stats.push({ zone, psr, psr_name: PSRTYPE_MAPPINGS[psr] || psr,
                     latest_ts: newest.ts, latest_mw: newest.mw,
                     resolution: newest.res, points_in_window: rows.length,
                     window_min_mw: vals.length ? Math.min(...vals) : null,
                     window_max_mw: vals.length ? Math.max(...vals) : null,
                     window_mean_mw: vals.length
                       ? Math.round(vals.reduce((a, b) => a + b, 0) / vals.length)
                       : null });
      });
      stats.sort((a, b) => a.zone.localeCompare(b.zone) || a.psr.localeCompare(b.psr));
      cache = { at: Date.now(), stats, issues: sweepIssues() };
    }
    gzipOldGenMixDays(baseDir, nowMs);
  } catch (e: any) {
    console.error("[datacore] eugenmix refresh:", e?.message || e);
  }
}

/** Realised generation publishes with ~1-2h lag, same cadence as load —
 *  2h poll, 8 spaced calls per cycle. Eager boot. */
export function bootEuGenerationMixPoll(intervalMs = 2 * 60 * 60_000): void {
  if (polling) return;
  polling = true;
  refreshGenMix();
  setInterval(() => { refreshGenMix(); }, intervalMs).unref?.();
}
