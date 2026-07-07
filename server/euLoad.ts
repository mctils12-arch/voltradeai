/**
 * euLoad.ts — ENTSO-E actual total load, hourly/quarter-hourly by
 * bidding zone (wishlist 9c, activated the day ENTSOE_API_KEY landed
 * in Railway — 2026-07-07; the EIA-930 analog for Europe).
 *
 * Source: ENTSO-E Transparency Platform RESTful API
 * (web-api.tp.entsoe.eu/api) — free token, re-use with source
 * attribution per the TP terms. Contract verified live 2026-07-07:
 * keyless probe returns an Acknowledgement_MarketDocument with
 * <Reason><code>999</code><text>Authentication failed.</text>; load
 * = documentType A65 + processType A16 (realised) +
 * outBiddingZone_Domain=<EIC> + periodStart/End as YYYYMMDDHHMM UTC.
 * Response: GL_MarketDocument > TimeSeries > Period with
 * timeInterval(start/end) + resolution (PT15M/PT30M/PT60M) + Point
 * (position, quantity in MW). EIC codes verified against the
 * ecosystem-canonical entsoe-py mapping table same day.
 *
 * Key-gated (fredMacro/gridDemand pattern): ENTSOE_API_KEY (as Mike
 * set it) or ENTSOE_TOKEN (the name the wishlist proposed) — either
 * activates; honest enabled:false on the route without one. Token is
 * never logged (URLs are never printed; errors log zone+status only).
 *
 * Scope v1: 8 country-level bidding zones, realised load only —
 * DE_LU, FR, ES, IT, NL, PL, BE, SE. Rate limit is 400 req/min; the
 * 2h poll's 8 spaced calls are nowhere near it.
 *
 * HYPOTHESIS (gate-locked; joins eumacro census #7): EU zonal load
 * vs its seasonal norm = the demand side of European energy regime
 * features (gas/power/carbon-adjacent tickers); pairs with ECB/
 * Eurostat industrial production for nowcast residuals. RAW archive
 * only until ladder gates.
 */

import fs from "fs";
import path from "path";
import zlib from "zlib";
import { archiveBaseDir } from "./datacoreArchive";

export function euLoadEnabled(env: NodeJS.ProcessEnv = process.env): boolean {
  return Boolean(env.ENTSOE_API_KEY || env.ENTSOE_TOKEN);
}

/** zone -> EIC (verified from the entsoe-py canonical mapping, 2026-07-07). */
export const ZONES: Record<string, string> = {
  DE_LU: "10Y1001A1001A82H",
  FR: "10YFR-RTE------C",
  ES: "10YES-REE------0",
  IT: "10YIT-GRTN-----B",
  NL: "10YNL----------L",
  PL: "10YPL-AREA-----S",
  BE: "10YBE----------2",
  SE: "10YSE-1--------K",
};

export const HOURS_PER_FETCH = 48;
const CALL_SPACING_MS = 1000;

export interface LoadObs {
  zone: string;         // our short code
  ts: string;           // point timestamp, UTC "YYYY-MM-DDTHH:MM"
  mw: number | null;    // actual total load, MW
  res: string;          // publication resolution ("PT15M"|"PT60M"|...)
  rt: string;           // as-seen UTC date
}

const RES_MINUTES: Record<string, number> = { PT15M: 15, PT30M: 30, PT60M: 60, P1D: 1440 };

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

function pad(n: number): string { return String(n).padStart(2, "0"); }

/** UTC ms -> ENTSO-E period stamp YYYYMMDDHHMM. */
export function periodStamp(ms: number): string {
  const d = new Date(ms);
  return `${d.getUTCFullYear()}${pad(d.getUTCMonth() + 1)}${pad(d.getUTCDate())}${pad(d.getUTCHours())}${pad(d.getUTCMinutes())}`;
}

export function loadUrl(eic: string, key: string, startMs: number, endMs: number): string {
  return "https://web-api.tp.entsoe.eu/api" +
    `?securityToken=${encodeURIComponent(key)}` +
    "&documentType=A65&processType=A16" +
    `&outBiddingZone_Domain=${encodeURIComponent(eic)}` +
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

/** Acknowledgement (error) document -> reason text, else null. */
export function parseAck(xml: string): string | null {
  if (!/Acknowledgement_MarketDocument/i.test(xml)) return null;
  const reason = tagBlocks(xml, "Reason")[0] || "";
  return (tagText(reason, "text") || tagText(reason, "code") || "unknown acknowledgement").slice(0, 200);
}

/** GL_MarketDocument -> LoadObs[]. Points are stored as PUBLISHED
 *  (position -> start + (position-1)*resolution); absent positions are
 *  simply absent — never interpolated or back-filled. */
export function parseLoad(xml: string, zone: string, rt: string): LoadObs[] {
  if (!/GL_MarketDocument/i.test(xml)) return [];
  const out: LoadObs[] = [];
  for (const series of tagBlocks(xml, "TimeSeries")) {
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
        out.push({ zone, ts, mw: n != null && Number.isFinite(n) ? n : null, res, rt });
      }
    }
  }
  return out;
}

// ── Fetch (token never logged) ──────────────────────────────────────────────

type FetchFn = (url: string, init?: any) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;

export async function fetchLoad(fetchImpl: FetchFn = fetch as any,
                                env: NodeJS.ProcessEnv = process.env,
                                nowMs?: number, spacingMs = CALL_SPACING_MS): Promise<LoadObs[]> {
  const key = env.ENTSOE_API_KEY || env.ENTSOE_TOKEN || "";
  if (!key) return [];
  const now = nowMs ?? Date.now();
  const rt = new Date(now).toISOString().slice(0, 10);
  const out: LoadObs[] = [];
  for (const [zone, eic] of Object.entries(ZONES)) {
    try {
      const r = await fetchImpl(loadUrl(eic, key, now - HOURS_PER_FETCH * 3600_000, now), {
        headers: { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)" },
        signal: AbortSignal.timeout(30000) as any,
      });
      const body = await r.text();
      if (!r.ok) {
        console.error(`[datacore] euload ${zone} -> ${r.status} ${parseAck(body) || ""}`);
      } else {
        const ack = parseAck(body);
        if (ack) console.error(`[datacore] euload ${zone} ack: ${ack}`);
        else out.push(...parseLoad(body, zone, rt));
      }
    } catch (e: any) {
      console.error(`[datacore] euload ${zone}:`, e?.message || e);
    }
    if (spacingMs > 0) await sleep(spacingMs);
  }
  return out;
}

// ── Archive (dedup zone|ts|res, day-file per UTC day; gridDemand pattern) ──

const seenObs = new Set<string>();
let seeded = false;

const obsKey = (o: LoadObs) => `${o.zone}|${o.ts}|${o.res}`;

function loadDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "euload");
}

export const SEED_WINDOW_DAYS = 120; // heap bound, gridDemand precedent

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

/** Appends UNSEEN points to day-files keyed by the OBSERVATION day. */
export function archiveLoad(obs: LoadObs[], baseDir?: string): number {
  if (!obs.length) return 0;
  const dir = loadDir(baseDir);
  if (!seeded) { seedSeen(dir); seeded = true; }
  const fresh = obs.filter((o) => !seenObs.has(obsKey(o)));
  if (!fresh.length) return 0;
  try {
    fs.mkdirSync(dir, { recursive: true });
    const byDay = new Map<string, LoadObs[]>();
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
    console.error("[datacore] euload archive:", e?.message || e);
    return 0;
  }
}

export function gzipOldLoadDays(baseDir?: string, nowMs?: number): number {
  const dir = loadDir(baseDir);
  const now = nowMs ?? Date.now();
  let n = 0;
  try {
    for (const f of fs.readdirSync(dir)) {
      if (!f.endsWith(".jsonl")) continue;
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

export interface ZoneStat {
  zone: string;
  latest_ts: string;
  latest_mw: number | null;
  resolution: string;
  points_in_window: number;
}

let cache: { at: number; stats: ZoneStat[] } | null = null;
let polling = false;

export function latestLoad() {
  return cache;
}

export async function refreshLoad(fetchImpl: FetchFn = fetch as any,
                                  env: NodeJS.ProcessEnv = process.env,
                                  nowMs?: number, baseDir?: string,
                                  spacingMs = CALL_SPACING_MS): Promise<void> {
  try {
    if (!euLoadEnabled(env)) return;
    const obs = await fetchLoad(fetchImpl, env, nowMs, spacingMs);
    if (obs.length) {
      archiveLoad(obs, baseDir);
      const byZone = new Map<string, LoadObs[]>();
      for (const o of obs) {
        if (!byZone.has(o.zone)) byZone.set(o.zone, []);
        byZone.get(o.zone)!.push(o);
      }
      const stats: ZoneStat[] = [];
      byZone.forEach((rows, zone) => {
        const newest = rows.reduce((mx, r) => (r.ts > mx.ts ? r : mx), rows[0]);
        stats.push({ zone, latest_ts: newest.ts, latest_mw: newest.mw,
                     resolution: newest.res, points_in_window: rows.length });
      });
      stats.sort((a, b) => a.zone.localeCompare(b.zone));
      cache = { at: Date.now(), stats };
    }
    gzipOldLoadDays(baseDir, nowMs);
  } catch (e: any) {
    console.error("[datacore] euload refresh:", e?.message || e);
  }
}

/** Realised load publishes with ~1-2h lag — 2h poll (8 spaced calls per
 *  cycle, far under the 400/min limit). Eager boot. */
export function bootEuLoadPoll(intervalMs = 2 * 60 * 60_000): void {
  if (polling) return;
  polling = true;
  refreshLoad();
  setInterval(() => { refreshLoad(); }, intervalMs).unref?.();
}
