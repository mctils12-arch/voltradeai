/**
 * usgsVolcanoes.ts — USGS Volcano Hazards Program elevated-alert-level feed,
 * joined against the Smithsonian Global Volcanism Program's (GVP) volcano
 * coordinate database. RAW-DATA overlay only (CLAUDE.md RAW-vs-SIGNAL
 * surface rule): displays USGS-reported alert levels as-is with source
 * attribution, no predictive claim — ships ungated, no key required.
 *
 * WHY A JOIN IS NEEDED (open_questions.md "NOT PREVIOUSLY FILED" C.2,
 * corrected 2026-08-01): USGS's own getElevatedVolcanoes listing endpoint
 * carries vnum/volcano_name/obs_abbr/alert_level/color_code but NO lat/lon —
 * coordinates only exist as free-text DMS strings buried in each notice's
 * HTML body, and one notice can bundle multiple volcanoes. The 2026-08-01
 * filing assumed this would need a STATIC vnum->coordinate reference table.
 * CORRECTION (this session, 2026-08-02): a better source exists — GVP's own
 * public GeoServer WFS endpoint (webservices.volcano.si.edu) returns
 * Volcano_Number/Latitude/Longitude/Country/Elevation as live JSON, filtered
 * by CQL_FILTER to exactly the vnums currently elevated (typically single
 * digits at a time — verified live 2026-08-02: 5 elevated volcanoes). This
 * is a LIVE per-request join, not a static table — no stale-reference-table
 * maintenance burden, and it stays correct if GVP revises a coordinate.
 *
 * LICENSING (verified this session via volcano.si.edu/gvp_termsofuse.cfm,
 * live fetch 2026-08-02 — read in full, not assumed): the GVP site's
 * general "Content" terms restrict commercial use, BUT the Terms of Use
 * page explicitly carves out the "Volcanoes of the World database" (exactly
 * what this module reads — Volcano_Number/Name/Country/Latitude/Longitude/
 * Elevation) as falling under category (2) of their own IP-rights section:
 * "created by an employee of the United States as part of his or her
 * official duties" — i.e. US federal government work product, not eligible
 * for copyright under 17 U.S.C. 105 regardless of the site's general policy
 * for narrative/photo Content. Attribution is still required per GVP's own
 * citation policy ("Global Volcanism Program, Smithsonian Institution" +
 * link) and is carried honestly in the /api/data response and the /data
 * layer registry. The USGS elevated-volcanoes feed itself is separately
 * US-government public domain (same policy already relied on for
 * usgsQuakes.ts). DELIBERATELY NOT shipped as an /api/v1 resale mirror this
 * session — this ships as a /data RAW OVERLAY only (RAW-vs-SIGNAL rule: no
 * predictive claim, source attribution, no ladder gate). A future session
 * adding a v1 mirror should re-examine the fact-vs-compilation distinction
 * explicitly before resale, the same "mark it honestly" discipline the
 * app-store/earnings-language v1 mirrors already use for their own
 * licensing nuances (see open_questions.md).
 *
 * EDGE DOCTRINE #1 (build data, don't buy it) + cross-ties (filed
 * open_questions.md, all still gate-2 hypotheses, none claimed here):
 * volcanic degassing vs. the SO2 GIBS overlay (v1.0.570), aviation ash risk
 * vs. the aircraft layer, and USGS earthquake swarms near active volcanoes.
 *
 * Boundary: no imports from trading logic — pure fetch/parse/archive, same
 * day-file-JSONL-with-dedup shape as usgsQuakes.ts/nasaFirms.ts.
 */

import fs from "fs";
import path from "path";
import zlib from "zlib";
import { archiveBaseDir } from "./datacoreArchive";

export const ELEVATED_FEED_URL = "https://volcanoes.usgs.gov/hans-public/api/volcano/getElevatedVolcanoes";
export const GVP_WFS_URL = "https://webservices.volcano.si.edu/geoserver/GVP-VOTW/ows";

export interface VolcanoAlert {
  vnum: string;              // USGS/GVP shared volcano number, e.g. "311120"
  name: string;               // as reported by USGS (usually matches GVP's Volcano_Name)
  obs: string | null;         // observatory abbreviation, e.g. "avo"
  obsFullname: string | null;
  alertLevel: string | null;  // NORMAL | ADVISORY | WATCH | WARNING (USGS's own vocabulary)
  colorCode: string | null;   // GREEN | YELLOW | ORANGE | RED (USGS's own vocabulary)
  noticeId: string | null;
  noticeUrl: string | null;
  sentUtc: string | null;
  sentUnixtime: number | null;
  lat: number | null;         // joined from GVP; null when the vnum has no GVP match
  lon: number | null;
  elevationM: number | null;
  country: string | null;
  rt: string;                 // as-seen UTC date (day WE fetched it)
}

interface GvpCoord {
  lat: number;
  lon: number;
  elevationM: number | null;
  country: string | null;
}

function toNumOrNull(v: any): number | null {
  const n = typeof v === "number" ? v : parseFloat(v);
  return Number.isFinite(n) ? n : null;
}

/** Parses USGS's getElevatedVolcanoes response shape into partial records
 *  (no coordinates yet — join happens separately via joinVolcanoAlerts).
 *  Rows with no vnum are dropped (nothing to dedup/archive/join against). */
export function parseElevatedVolcanoes(json: any, rt: string): Omit<VolcanoAlert, "lat" | "lon" | "elevationM" | "country">[] {
  const out: Omit<VolcanoAlert, "lat" | "lon" | "elevationM" | "country">[] = [];
  for (const r of Array.isArray(json) ? json : []) {
    const vnum = r?.vnum != null ? String(r.vnum) : null;
    if (!vnum) continue;
    out.push({
      vnum,
      name: r?.volcano_name || "Unknown",
      obs: r?.obs_abbr || null,
      obsFullname: r?.obs_fullname || null,
      alertLevel: r?.alert_level || null,
      colorCode: r?.color_code || null,
      noticeId: r?.notice_identifier || null,
      noticeUrl: r?.notice_url || null,
      sentUtc: r?.sent_utc || null,
      sentUnixtime: toNumOrNull(r?.sent_unixtime),
      rt,
    });
  }
  return out;
}

/** Builds the GVP GeoServer WFS request URL for exactly the given vnums —
 *  a CQL_FILTER IN() list, not a full-database fetch (courtesy: don't pull
 *  ~1400 Holocene volcanoes when we need 5). Pure/testable; returns null
 *  when there's nothing to ask for. */
export function buildGvpWfsUrl(vnums: string[]): string | null {
  const clean = Array.from(new Set(vnums.filter((v) => /^\d+$/.test(v))));
  if (!clean.length) return null;
  const params = new URLSearchParams({
    service: "WFS",
    version: "2.0.0",
    request: "GetFeature",
    typeName: "GVP-VOTW:Smithsonian_VOTW_Holocene_Volcanoes",
    outputFormat: "application/json",
    propertyName: "Volcano_Number,Volcano_Name,Country,Latitude,Longitude,Elevation",
    CQL_FILTER: `Volcano_Number IN (${clean.join(",")})`,
  });
  return `${GVP_WFS_URL}?${params.toString()}`;
}

/** Parses a GVP WFS GeoJSON FeatureCollection into a vnum-keyed coordinate
 *  map. Malformed/missing lat-lon rows are skipped (never a fabricated 0,0). */
export function parseGvpFeatureCollection(json: any): Map<string, GvpCoord> {
  const out = new Map<string, GvpCoord>();
  for (const f of json?.features || []) {
    const p = f?.properties || {};
    const vnum = p.Volcano_Number != null ? String(p.Volcano_Number) : null;
    const lat = toNumOrNull(p.Latitude);
    const lon = toNumOrNull(p.Longitude);
    if (!vnum || lat == null || lon == null) continue;
    out.set(vnum, { lat, lon, elevationM: toNumOrNull(p.Elevation), country: p.Country || null });
  }
  return out;
}

/** Joins USGS alert rows against a GVP vnum->coordinate map. A vnum with no
 *  GVP match is KEPT with null lat/lon/elevationM/country (archive keeps
 *  everything the source reported; the client's own render filter is where
 *  un-placeable rows drop off the map, matching the fires.json/usgsQuakes.ts
 *  "archive keeps all, display filters" precedent). */
export function joinVolcanoAlerts(
  elevated: Omit<VolcanoAlert, "lat" | "lon" | "elevationM" | "country">[],
  coords: Map<string, GvpCoord>,
): VolcanoAlert[] {
  return elevated.map((e) => {
    const c = coords.get(e.vnum);
    return { ...e, lat: c?.lat ?? null, lon: c?.lon ?? null, elevationM: c?.elevationM ?? null, country: c?.country ?? null };
  });
}

// ── fetch (injectable, mirrors usgsQuakes.ts/nasaFirms.ts) ─────────────────
type FetchFn = (url: string, init?: any) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;
const UA = { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)" };

export async function fetchElevatedVolcanoes(fetchImpl: FetchFn = fetch as any, nowMs?: number) {
  const rt = new Date(nowMs ?? Date.now()).toISOString().slice(0, 10);
  const r = await fetchImpl(ELEVATED_FEED_URL, { headers: UA, signal: AbortSignal.timeout(15000) as any });
  if (!r.ok) throw new Error(`USGS elevated volcanoes ${r.status}`);
  return parseElevatedVolcanoes(JSON.parse(await r.text()), rt);
}

export async function fetchGvpCoords(vnums: string[], fetchImpl: FetchFn = fetch as any): Promise<Map<string, GvpCoord>> {
  const url = buildGvpWfsUrl(vnums);
  if (!url) return new Map();
  const r = await fetchImpl(url, { headers: UA, signal: AbortSignal.timeout(15000) as any });
  if (!r.ok) throw new Error(`GVP WFS ${r.status}`);
  return parseGvpFeatureCollection(JSON.parse(await r.text()));
}

/** Fetches USGS's elevated list, then joins coordinates ONLY for vnums not
 *  already in the in-process coordinate cache — courtesy to GVP's server
 *  (a volcano number's coordinates never change between polls; USGS's own
 *  alert level/notice does). */
const coordCache = new Map<string, GvpCoord>();

export async function fetchVolcanoAlerts(fetchImpl: FetchFn = fetch as any, nowMs?: number): Promise<VolcanoAlert[]> {
  const elevated = await fetchElevatedVolcanoes(fetchImpl, nowMs);
  const uncached = elevated.map((e) => e.vnum).filter((v) => !coordCache.has(v));
  if (uncached.length) {
    const fresh = await fetchGvpCoords(uncached, fetchImpl);
    fresh.forEach((c, vnum) => coordCache.set(vnum, c));
  }
  return joinVolcanoAlerts(elevated, coordCache);
}

// ── archive (day-file JSONL, dedup by vnum+notice_identifier — a new
// notice_identifier means USGS published a new update for that volcano;
// gzipped after 2 days — identical shape to usgsQuakes.ts) ─────────────────
const archivedKeys = new Set<string>();
let seeded = false;

function volcanoesDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "volcanoes");
}

function seedSeen(dir: string, nowMs: number): void {
  for (const dayMs of [nowMs, nowMs - 86400_000]) {
    const fp = path.join(dir, `${new Date(dayMs).toISOString().slice(0, 10)}.jsonl`);
    try {
      for (const line of fs.readFileSync(fp, "utf8").split("\n")) {
        if (!line) continue;
        try {
          const r = JSON.parse(line);
          archivedKeys.add(`${r.vnum}:${r.noticeId}`);
        } catch {}
      }
    } catch {}
  }
}

export function archiveVolcanoAlerts(alerts: VolcanoAlert[], baseDir?: string, nowMs?: number): number {
  const dir = volcanoesDir(baseDir);
  const now = nowMs ?? Date.now();
  if (!seeded) {
    seedSeen(dir, now);
    seeded = true;
  }
  const fresh = alerts.filter((a) => a.vnum && !archivedKeys.has(`${a.vnum}:${a.noticeId}`));
  if (!fresh.length) return 0;
  try {
    fs.mkdirSync(dir, { recursive: true });
    const fp = path.join(dir, `${new Date(now).toISOString().slice(0, 10)}.jsonl`);
    fs.appendFileSync(fp, fresh.map((a) => JSON.stringify(a)).join("\n") + "\n");
    fresh.forEach((a) => archivedKeys.add(`${a.vnum}:${a.noticeId}`));
    if (archivedKeys.size > 50_000) {
      let drop = archivedKeys.size >> 1;
      const it = archivedKeys.values();
      let cur = it.next();
      while (!cur.done && drop-- > 0) { archivedKeys.delete(cur.value); cur = it.next(); }
    }
    return fresh.length;
  } catch (e: any) {
    console.error("[datacore] volcanoes archive:", e?.message || e);
    return 0;
  }
}

export function gzipOldVolcanoDays(baseDir?: string, nowMs?: number): number {
  const dir = volcanoesDir(baseDir);
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

// ── in-memory cache + poll loop (mirrors usgsQuakes.ts's boot pattern) ─────
let cache: { at: number; alerts: VolcanoAlert[] } | null = null;
let polling = false;

export function latestVolcanoAlerts(): { at: number; alerts: VolcanoAlert[] } | null {
  return cache;
}

export async function refreshVolcanoesCache(fetchImpl: FetchFn = fetch as any, nowMs?: number): Promise<void> {
  try {
    const alerts = await fetchVolcanoAlerts(fetchImpl, nowMs);
    // DELIBERATE DIFFERENCE from usgsQuakes.ts's refreshQuakesCache: quakes
    // suppresses an empty successful fetch (implausible for a global M2.5+
    // 24h window to genuinely hit zero, so empty more likely means a parse
    // hiccup — keep stale data over a false "nothing's happening"). An empty
    // elevated-volcanoes list is a real, common, and meaningful state
    // (no volcano currently elevated) — always overwrite, honestly.
    cache = { at: Date.now(), alerts };
    try { archiveVolcanoAlerts(alerts, undefined, nowMs); } catch {}
    try { gzipOldVolcanoDays(undefined, nowMs); } catch {}
  } catch (e: any) {
    console.error("[datacore] volcanoes refresh:", e?.message || e);
  }
}

/** Keyless, no gating — boots eagerly at server start (KNOWN BROKEN #9's
 *  lesson: a lazy first-request connect leaves a cold gap in the archive).
 *  30-min cadence: elevated-status changes are rare (weekly-to-monthly per
 *  volcano in steady state) compared to earthquakes' near-continuous stream
 *  — far under any reasonable courtesy-use rate for two keyless public
 *  government feeds. */
export function bootVolcanoesPoll(intervalMs = 30 * 60_000): void {
  if (polling) return;
  polling = true;
  refreshVolcanoesCache();
  setInterval(() => { refreshVolcanoesCache(); }, intervalMs).unref?.();
}
