/**
 * euDayAheadPrices.ts — ENTSO-E day-ahead auction clearing price, hourly
 * (or zone-native finer resolution) by EU bidding zone (wishlist 9c's
 * LAST open follow-up — load and generation-mix shipped 2026-07-07 and
 * 2026-07-21; this closes the trio, same token, own PR per one-logical-
 * change-per-PR).
 *
 * Source: same ENTSO-E Transparency Platform RESTful API host/auth as
 * euLoad.ts/euGenerationMix.ts (web-api.tp.entsoe.eu/api) —
 * documentType=A44 ("Price Document"), in_Domain=out_Domain=<EIC> (both
 * required and identical for a single-zone day-ahead query per the
 * ENTSO-E API guide and cross-checked against the ecosystem-canonical
 * entsoe-py library's query_day_ahead_prices this session). UNLIKE A65
 * (load) and A75 (generation), the price document takes NO processType
 * parameter and its root element is `Publication_MarketDocument`, not
 * `GL_MarketDocument` — genuinely a different schema family
 * (urn:iec62325.351:tc57wg16:451-3:publicationdocument:...), not a copy
 * of the load/generation parser with a renamed value tag. The value tag
 * itself is `<price.amount>` (can be NEGATIVE — renewables oversupply
 * routinely clears negative day-ahead prices, never clamp to zero), with
 * `<currency_Unit.name>`/`<price_Measure_Unit.name>` carried per
 * TimeSeries (stored per-observation rather than assumed EUR/MWh for
 * every zone — SE's day-ahead auction has historically settled in
 * non-EUR currency on some platforms; never guessed here, read as
 * published).
 *
 * HONESTY NOTE / NEXT CHECK (same discipline euGenerationMix's build
 * session used, no ENTSOE_API_KEY in this sandbox either — only Railway
 * has it): once deployed, read `/api/data/eu-day-ahead-prices`'s
 * `issues` field for the first sweep. An unexpected ack or an empty-but-
 * 200 response would mean the documentType/domain contract needs
 * adjustment despite the entsoe-py cross-check.
 *
 * FETCH WINDOW is deliberately NOT a trailing-only window like load/
 * generation (those are REALISED — backward-looking only). Day-ahead
 * prices are FORWARD-published: the auction for "tomorrow" clears and
 * publishes once daily, typically ~11:00-13:00 CET. A trailing-only
 * window would miss today's just-published tomorrow prices until the
 * NEXT UTC-day boundary. Window here is (now-24h, now+48h) — covers
 * yesterday's settled auction, today's, and tomorrow's once published;
 * unpublished future points are simply absent, never fabricated.
 *
 * Scope v1: same 8 country-level zones as euLoad.ts/euGenerationMix.ts.
 * Rate limit 400 req/min; the 2h poll's 8 spaced calls (separate from
 * load's and generation's own 8 each) stays nowhere near it.
 *
 * HYPOTHESIS (gate-locked; joins eumacro census #7 + euLoad/
 * euGenerationMix): zonal day-ahead price level and volatility (incl.
 * negative-price frequency, a direct renewables-oversupply signal) is
 * a demand/supply-balance proxy for European power/gas/carbon-adjacent
 * tickers, and the price side of the euLoad (demand) + euGenerationMix
 * (fuel mix) pair — RAW archive only until ladder gates; nothing traded
 * or sold on this alone.
 */

import fs from "fs";
import path from "path";
import zlib from "zlib";
import { archiveBaseDir } from "./datacoreArchive";
import { ZONES } from "./euLoad";

export function euDayAheadPricesEnabled(env: NodeJS.ProcessEnv = process.env): boolean {
  return Boolean(env.ENTSOE_API_KEY || env.ENTSOE_TOKEN);
}

export { ZONES };

export const HOURS_BEFORE = 24;
export const HOURS_AFTER = 48;
const CALL_SPACING_MS = 1000;

export interface PriceObs {
  zone: string;
  ts: string;              // point timestamp, UTC "YYYY-MM-DDTHH:MM"
  price: number | null;    // day-ahead clearing price, AS PUBLISHED (can be negative)
  currency: string;        // e.g. "EUR" — read from currency_Unit.name, never assumed
  unit: string;            // e.g. "MWH" — read from price_Measure_Unit.name, never assumed
  res: string;              // publication resolution ("PT15M"|"PT30M"|"PT60M"|...)
  rt: string;               // as-seen UTC date
}

const RES_MINUTES: Record<string, number> = { PT15M: 15, PT30M: 30, PT60M: 60, P1D: 1440 };

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

function pad(n: number): string { return String(n).padStart(2, "0"); }

/** UTC ms -> ENTSO-E period stamp YYYYMMDDHHMM. */
export function periodStamp(ms: number): string {
  const d = new Date(ms);
  return `${d.getUTCFullYear()}${pad(d.getUTCMonth() + 1)}${pad(d.getUTCDate())}${pad(d.getUTCHours())}${pad(d.getUTCMinutes())}`;
}

/** in_Domain AND out_Domain both = the same zone EIC — the A44 query
 *  contract per entsoe-py's query_day_ahead_prices, cross-checked this
 *  session. No processType (unlike A65/A75). */
export function priceUrl(eic: string, key: string, startMs: number, endMs: number): string {
  return "https://web-api.tp.entsoe.eu/api" +
    `?securityToken=${encodeURIComponent(key)}` +
    "&documentType=A44" +
    `&in_Domain=${encodeURIComponent(eic)}&out_Domain=${encodeURIComponent(eic)}` +
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

/** Acknowledgement (error) document -> reason text, else null. Same
 *  shape regardless of which document type was requested. */
export function parseAck(xml: string): string | null {
  if (!/Acknowledgement_MarketDocument/i.test(xml)) return null;
  const reason = tagBlocks(xml, "Reason")[0] || "";
  return (tagText(reason, "text") || tagText(reason, "code") || "unknown acknowledgement").slice(0, 200);
}

/** Publication_MarketDocument -> PriceObs[]. currency_Unit.name /
 *  price_Measure_Unit.name are read per-TimeSeries (a series missing
 *  either is stored with an empty string, never a guessed "EUR"/"MWH"). */
export function parsePrices(xml: string, zone: string, rt: string): PriceObs[] {
  if (!/Publication_MarketDocument/i.test(xml)) return [];
  const out: PriceObs[] = [];
  for (const series of tagBlocks(xml, "TimeSeries")) {
    const currency = tagText(series, "currency_Unit\\.name") || "";
    const unit = tagText(series, "price_Measure_Unit\\.name") || "";
    for (const period of tagBlocks(series, "Period")) {
      const start = tagText(period, "start");
      const res = tagText(period, "resolution") || "";
      const stepMin = RES_MINUTES[res];
      if (!start || !stepMin) continue;
      const startMs = Date.parse(start);
      if (!Number.isFinite(startMs)) continue;
      for (const point of tagBlocks(period, "Point")) {
        const pos = parseInt(tagText(point, "position") || "", 10);
        const amt = tagText(point, "price\\.amount");
        if (!Number.isFinite(pos) || pos < 1) continue;
        const n = amt == null || amt === "" ? null : parseFloat(amt);
        const ts = new Date(startMs + (pos - 1) * stepMin * 60_000)
          .toISOString().slice(0, 16);
        out.push({ zone, ts, price: n != null && Number.isFinite(n) ? n : null, currency, unit, res, rt });
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

export async function fetchPrices(fetchImpl: FetchFn = fetch as any,
                                  env: NodeJS.ProcessEnv = process.env,
                                  nowMs?: number, spacingMs = CALL_SPACING_MS): Promise<PriceObs[]> {
  const key = env.ENTSOE_API_KEY || env.ENTSOE_TOKEN || "";
  if (!key) return [];
  const now = nowMs ?? Date.now();
  const rt = new Date(now).toISOString().slice(0, 10);
  const out: PriceObs[] = [];
  for (const k of Object.keys(lastIssues)) delete lastIssues[k];
  for (const [zone, eic] of Object.entries(ZONES)) {
    try {
      const r = await fetchImpl(priceUrl(eic, key, now - HOURS_BEFORE * 3600_000, now + HOURS_AFTER * 3600_000), {
        headers: { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)" },
        signal: AbortSignal.timeout(30000) as any,
      });
      const body = await r.text();
      if (!r.ok) {
        lastIssues[zone] = `http ${r.status}: ${parseAck(body) || ""}`.trim().slice(0, 200);
        console.error(`[datacore] euprices ${zone} -> ${lastIssues[zone]}`);
      } else {
        const ack = parseAck(body);
        if (ack) {
          lastIssues[zone] = `ack: ${ack}`;
          console.error(`[datacore] euprices ${zone} ${lastIssues[zone]}`);
        } else {
          const rows = parsePrices(body, zone, rt);
          if (!rows.length) lastIssues[zone] = "empty document (no parseable points)";
          out.push(...rows);
        }
      }
    } catch (e: any) {
      lastIssues[zone] = String(e?.message || e).slice(0, 200);
      console.error(`[datacore] euprices ${zone}:`, lastIssues[zone]);
    }
    if (spacingMs > 0) await sleep(spacingMs);
  }
  return out;
}

// ── Archive (dedup zone|ts|res|VALUE — vintage capture; day-file per UTC day) ──
// A day-ahead price is normally published once and never revised (unlike
// load/generation's leading-edge partial points) — but the same
// value-aware dedup key as euLoad/euGenerationMix is used anyway, at zero
// extra cost, so an ENTSO-E republication-with-correction (rare, but on
// record for other TSO data) still lands as a new vintage row instead of
// silently overwriting history.

const seenObs = new Set<string>();
let seeded = false;

const obsKey = (o: PriceObs) => `${o.zone}|${o.ts}|${o.res}|${o.price}`;

function priceDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "eudayaheadprices");
}

export const SEED_WINDOW_DAYS = 120; // heap bound, euLoad precedent

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

/** Appends UNSEEN points to day-files keyed by the OBSERVATION day (can
 *  be a future UTC day — tomorrow's auction result lands in tomorrow's
 *  file the moment it publishes, not deferred to when that day arrives). */
export function archivePrices(obs: PriceObs[], baseDir?: string): number {
  if (!obs.length) return 0;
  const dir = priceDir(baseDir);
  if (!seeded) { seedSeen(dir); seeded = true; }
  const fresh = obs.filter((o) => !seenObs.has(obsKey(o)));
  if (!fresh.length) return 0;
  try {
    fs.mkdirSync(dir, { recursive: true });
    const byDay = new Map<string, PriceObs[]>();
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
    console.error("[datacore] euprices archive:", e?.message || e);
    return 0;
  }
}

export function gzipOldPriceDays(baseDir?: string, nowMs?: number): number {
  const dir = priceDir(baseDir);
  const now = nowMs ?? Date.now();
  let n = 0;
  try {
    for (const f of fs.readdirSync(dir)) {
      if (!f.endsWith(".jsonl")) continue;
      // future/today's day-files (tomorrow's auction, already published)
      // must never be gz'd early — only files whose OWN day is >=3 days
      // in the past are eligible, same 3-day-lag rule as euLoad/euGenerationMix
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

export interface PriceStat {
  zone: string;
  latest_ts: string;
  latest_price: number | null;
  currency: string;
  unit: string;
  resolution: string;
  points_in_window: number;
  window_min_price: number | null;
  window_max_price: number | null;
  window_mean_price: number | null;
  negative_price_points: number;   // count of published points < 0 in window
}

let cache: { at: number; stats: PriceStat[]; issues: Record<string, string> } | null = null;
let polling = false;

export function latestPrices() {
  return cache;
}

export async function refreshPrices(fetchImpl: FetchFn = fetch as any,
                                    env: NodeJS.ProcessEnv = process.env,
                                    nowMs?: number, baseDir?: string,
                                    spacingMs = CALL_SPACING_MS): Promise<void> {
  try {
    if (!euDayAheadPricesEnabled(env)) return;
    const obs = await fetchPrices(fetchImpl, env, nowMs, spacingMs);
    if (obs.length) {
      archivePrices(obs, baseDir);
      const byZone = new Map<string, PriceObs[]>();
      for (const o of obs) {
        if (!byZone.has(o.zone)) byZone.set(o.zone, []);
        byZone.get(o.zone)!.push(o);
      }
      const stats: PriceStat[] = [];
      byZone.forEach((rows, zone) => {
        const newest = rows.reduce((mx, r) => (r.ts > mx.ts ? r : mx), rows[0]);
        const vals = rows.map((r) => r.price).filter((v): v is number => v != null);
        stats.push({ zone, latest_ts: newest.ts, latest_price: newest.price,
                     currency: newest.currency, unit: newest.unit,
                     resolution: newest.res, points_in_window: rows.length,
                     window_min_price: vals.length ? Math.min(...vals) : null,
                     window_max_price: vals.length ? Math.max(...vals) : null,
                     window_mean_price: vals.length
                       ? Math.round((vals.reduce((a, b) => a + b, 0) / vals.length) * 100) / 100
                       : null,
                     negative_price_points: vals.filter((v) => v < 0).length });
      });
      stats.sort((a, b) => a.zone.localeCompare(b.zone));
      cache = { at: Date.now(), stats, issues: sweepIssues() };
    }
    gzipOldPriceDays(baseDir, nowMs);
  } catch (e: any) {
    console.error("[datacore] euprices refresh:", e?.message || e);
  }
}

/** Day-ahead auction publishes once daily (~11:00-13:00 CET); 2h poll
 *  catches the new publication same day without over-polling. Eager boot. */
export function bootEuDayAheadPricesPoll(intervalMs = 2 * 60 * 60_000): void {
  if (polling) return;
  polling = true;
  refreshPrices();
  setInterval(() => { refreshPrices(); }, intervalMs).unref?.();
}
