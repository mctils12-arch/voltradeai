// aeroBake.ts — our OWN bake of the FAA charts as a tile source for
// server/aeroCharts.ts.
//
// WHY: the FAA's chart tile service lags the FAA's 56-day chart cycle (on
// 2026-10-06 it still carried 07-09-2026 — one full cycle behind — and IFR
// High 05-14-2026, two behind), while the current cycle's GeoTIFFs are on
// aeronav.faa.gov ~20 days before they take effect. scripts/faa_charts/run.py
// bakes those GeoTIFFs every cycle (chart borders cut along edges measured
// against the FAA's own mosaic, one seamless pyramid z2..native, WebP) into
// one PMTiles per chart family, gates it against the FAA mosaic, and uploads
// it with a manifest to the public tiles bucket (R2_PUBLIC_URL).
//
// RULE: a bake replaces the FAA service for a chart only when it is
//   (a) EFFECTIVE — its edition date has arrived (charts posted early are
//       not yet in force), and
//   (b) NEWER than the edition the FAA service carries.
// Otherwise the FAA service path stays exactly as it was. The manifest keeps
// the previous bake too, so a just-posted (not yet effective) edition never
// knocks the current one off the map.

import { PMTiles, type RangeResponse, type Source } from "pmtiles";

import type { AeroChartId } from "./aeroCharts";

export const AERO_BAKE_MANIFEST_KEY = "tiles/faa/manifest.json";
export const AERO_BAKE_MANIFEST_TTL_MS = 30 * 60_000;
export const AERO_BAKE_MANIFEST_RETRY_MS = 5 * 60_000;
export const AERO_BAKE_RANGE_TIMEOUT_MS = 8_000;
const DEFAULT_PUBLIC_BASE = "https://pub-4d65a892936747ada1c67a1f00e286c8.r2.dev"; // same default as /tiles-r2 (server/routes.ts)

export interface BakeEntry {
  edition: string;
  key: string;
  minZoom: number;
  maxZoom: number;
  /** the FAA service's lowest level for this chart (where it draws chart) */
  faaMinZoom: number;
  bakedAt: string | null;
  charts: number | null;
  verify: { coverage: number; spill: number; median_structural_diff: number; sampled_tiles: number } | null;
}

export type BakeManifest = Partial<Record<AeroChartId, BakeEntry[]>>;

const ISO_DAY = /^\d{4}-\d{2}-\d{2}$/;

function zoom(v: unknown): number | null {
  return Number.isInteger(v) && (v as number) >= 0 && (v as number) <= 22 ? (v as number) : null;
}

function parseEntry(chart: string, e: unknown): BakeEntry | null {
  if (!e || typeof e !== "object") return null;
  const o = e as Record<string, unknown>;
  const edition = typeof o.edition === "string" && ISO_DAY.test(o.edition) ? o.edition : null;
  if (!edition) return null;
  // the key must be exactly the run's own naming — never an arbitrary path
  const key = typeof o.key === "string" && o.key === `tiles/faa/${chart}/${edition}.pmtiles` ? o.key : null;
  const minZoom = zoom(o.min_zoom), maxZoom = zoom(o.max_zoom), faaMinZoom = zoom(o.faa_min_zoom);
  if (!key || minZoom === null || maxZoom === null || faaMinZoom === null || minZoom > maxZoom) return null;
  const v = o.verify as Record<string, unknown> | undefined;
  const verify = v && typeof v.coverage === "number" && typeof v.spill === "number"
    && typeof v.median_structural_diff === "number" && typeof v.sampled_tiles === "number"
    ? { coverage: v.coverage, spill: v.spill, median_structural_diff: v.median_structural_diff, sampled_tiles: v.sampled_tiles }
    : null;
  return {
    edition, key, minZoom, maxZoom, faaMinZoom,
    bakedAt: typeof o.baked_at === "string" ? o.baked_at : null,
    charts: Number.isInteger(o.charts) ? (o.charts as number) : null,
    verify,
  };
}

/** Validate the manifest run.py writes. Per chart: the latest entry and the
 *  one it replaced (`previous`), newest first. Anything malformed is
 *  dropped — a bad manifest degrades to "use the FAA service". */
export function parseBakeManifest(j: unknown, charts: readonly AeroChartId[]): BakeManifest {
  const out: BakeManifest = {};
  const fam = j && typeof j === "object" ? (j as Record<string, unknown>).families : null;
  if (!fam || typeof fam !== "object") return out;
  for (const id of charts) {
    const raw = (fam as Record<string, unknown>)[id];
    const cur = parseEntry(id, raw);
    if (!cur) continue;
    const list = [cur];
    const prev = parseEntry(id, (raw as Record<string, unknown>).previous);
    if (prev && prev.edition < cur.edition) list.push(prev);
    out[id] = list;
  }
  return out;
}

/** The bake to serve, or null for "use the FAA service": the newest entry
 *  that is effective (edition <= today) and newer than the service's
 *  edition (or the service edition is unknown). */
export function chooseBake(entries: readonly BakeEntry[] | undefined, serviceEdition: string | null, todayIso: string): BakeEntry | null {
  for (const e of entries ?? []) {
    if (e.edition > todayIso) continue;
    if (serviceEdition && e.edition <= serviceEdition) return null;
    return e;
  }
  return null;
}

export function bakePublicBase(env: NodeJS.ProcessEnv): string {
  return (env.R2_PUBLIC_URL || DEFAULT_PUBLIC_BASE).replace(/\/+$/, "");
}

/** pmtiles Source over HTTP range requests to the public tiles bucket. */
export class RangeFetchSource implements Source {
  constructor(private url: string, private fetchImpl: typeof fetch, private timeoutMs = AERO_BAKE_RANGE_TIMEOUT_MS) {}

  getKey(): string {
    return this.url;
  }

  async getBytes(offset: number, length: number, signal?: AbortSignal): Promise<RangeResponse> {
    const ac = new AbortController();
    const t = setTimeout(() => ac.abort(), this.timeoutMs);
    const onAbort = () => ac.abort();
    signal?.addEventListener("abort", onAbort);
    try {
      const res = await this.fetchImpl(this.url, {
        signal: ac.signal,
        headers: { range: `bytes=${offset}-${offset + length - 1}`, "user-agent": "VolTradeAI-datamap/1.0 (+own FAA chart bake)" },
      });
      if (res.status !== 206 && res.status !== 200) throw new Error(`bake range HTTP ${res.status}`);
      let buf = await res.arrayBuffer();
      // a server that ignored the Range header sent the whole object
      if (res.status === 200 && buf.byteLength > length) buf = buf.slice(offset, offset + length);
      return { data: buf, etag: res.headers.get("etag") ?? undefined };
    } finally {
      clearTimeout(t);
      signal?.removeEventListener("abort", onAbort);
    }
  }
}

/** One PMTiles reader per published archive (header + directories cached by
 *  the library); a handful of keys at most (4 charts x current/previous). */
export function createBakeReader(base: string, fetchImpl: typeof fetch) {
  const archives = new Map<string, PMTiles>();
  return {
    async tile(key: string, z: number, x: number, y: number): Promise<Buffer | null> {
      let p = archives.get(key);
      if (!p) {
        p = new PMTiles(new RangeFetchSource(`${base}/${key}`, fetchImpl));
        archives.set(key, p);
        while (archives.size > 8) archives.delete(archives.keys().next().value as string);
      }
      const r = await p.getZxy(z, x, y);
      return r && r.data.byteLength ? Buffer.from(r.data) : null;
    },
    forget(key: string) {
      archives.delete(key);
    },
  };
}
