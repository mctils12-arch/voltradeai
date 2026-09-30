// globalSnapshot.ts — the in-memory GLOBAL live aircraft snapshot
// (FLIGHT PROGRAM B1, 2026-09-28): latest fix per hex from EVERY live path
// (viewport discs, the global sweep's type + disc lanes, mil/ladd/pia
// scopes, optional OpenSky). Pure + clock-injectable; no I/O.
//
// MERGE RULE: per hex the FRESHEST fix wins (by the fix's own time, seenAt =
// upstream clock minus the feed's seen_pos age). On a near-tie (< TIE_MS)
// the adsb.lol fix wins — the ODbL, monetization-lawful provider — so the
// lawful subset stays as large as the data allows. `src` is the provenance
// of the POSITION fix and is never rewritten; identity fields (type, reg,
// category) may be carried from an earlier fix of the same hex ONLY when
// that earlier fix was adsb.lol's or the same provider's — so a row tagged
// adsblol never contains a non-commercial provider's value, and
// `?lawful=1` (src === "adsblol") is a provable ODbL-only subset.
//
// EVICTION: a hex with no fix newer than EVICT_MS (10 min) is dropped. 10
// rather than 5 min because the sweep's disc lane revisits a low-traffic
// disc only every ~15-50 min: 5 min would blank legitimately-tracked
// untyped aircraft between visits for no honesty gain — every row carries
// seenAt, and the client dims anything older than 2 min. A position older
// than 10 min is ~150 km stale at cruise and is not shown at all.
//
// HARD CAP: CAP rows (25,000 — above the busiest global instant any free
// feed reports). Past it the OLDEST fixes are dropped first and the drop is
// counted (no silent caps); memory stays bounded (~25k small objects).

import type { AircraftPoint } from "./datacoreArchive";
import type { FixBatch } from "./aircraftFixBus";

export const SNAPSHOT_EVICT_MS = 10 * 60_000;
export const SNAPSHOT_CAP = 25_000;
/** fixes this close in time are a tie → prefer the lawful provider */
export const TIE_MS = 1_000;
export const LAWFUL_PROVIDER = "adsblol";

export interface SnapRow {
  hex: string;
  lon: number;
  lat: number;
  /** barometric (else geometric) altitude, ft; null on ground/unknown */
  altFt: number | null;
  /** ground speed, knots */
  gsKt: number | null;
  /** true track, degrees */
  trk: number | null;
  callsign: string | null;
  type: string | null;
  cat: string | null;
  gnd: boolean;
  reg: string | null;
  /** provenance of this POSITION fix (adsblol | airplaneslive | adsbfi | opensky) */
  src: string;
  /** which path delivered it (viewport | sweep-type | sweep-disc | scopes | opensky) */
  via: string;
  /** fix time, unix ms */
  seenAt: number;
  /** SERVER clock when this row last changed in the snapshot (set by
   *  ingest; never on the wire). The exact delta cursor for `changed=`:
   *  seenAt lags ingest by the provider's latency (live p50 ~16 s), so a
   *  seenAt cursor silently skips rows that land late. */
  ingestAt?: number;
}

/** Wire order of GET /api/data/aircraft/global rows. */
export const ROW_FIELDS = [
  "hex", "lon", "lat", "altFt", "gsKt", "trk", "callsign", "type", "seenAt",
  "cat", "gnd", "reg", "src",
] as const;

const M_TO_FT = 1 / 0.3048;
const MS_TO_KT = 1 / 0.5144; // the pipeline's own kt→m/s factor, inverted exactly

/** Pipeline row (metric; mapPointAircraft / OpenSky normalizer) → SnapRow.
 *  altFt/gsKt are converted back from the pipeline's rounded metric fields
 *  (±2 ft / ±1 kt — display precision, stated in the endpoint honesty). */
export function toSnapRow(p: AircraftPoint & { seen_at_ms?: number | null }, batch: FixBatch): SnapRow | null {
  if (!p || !p.icao24 || !Number.isFinite(p.lat) || !Number.isFinite(p.lon)) return null;
  const base = Number.isFinite(batch.upstreamNowMs as number) ? (batch.upstreamNowMs as number) : batch.fetchedAt;
  const seenAt = Number.isFinite(p.seen_at_ms as number)
    ? (p.seen_at_ms as number)
    : base - (Number.isFinite(p.seen_pos as number) ? Math.max(0, p.seen_pos as number) * 1000 : 0);
  return {
    hex: String(p.icao24).toLowerCase(),
    lon: +p.lon.toFixed(4),
    lat: +p.lat.toFixed(4),
    altFt: p.on_ground || p.altitude_m == null ? null : Math.round(p.altitude_m * M_TO_FT),
    gsKt: p.velocity_ms == null ? null : Math.round(p.velocity_ms * MS_TO_KT),
    trk: p.heading == null || !Number.isFinite(p.heading) ? null : Math.round(p.heading * 10) / 10,
    callsign: p.callsign ? String(p.callsign).trim() || null : null,
    type: p.type || null,
    cat: p.category || null,
    gnd: !!p.on_ground,
    reg: p.registration || null,
    src: batch.provider,
    via: batch.origin,
    seenAt: Math.round(seenAt),
  };
}

export interface BBox { lamin: number; lamax: number; lomin: number; lomax: number }

export function rowInBBox(r: { lat: number; lon: number }, b: BBox): boolean {
  if (r.lat < b.lamin || r.lat > b.lamax) return false;
  if (b.lomin <= b.lomax) return r.lon >= b.lomin && r.lon <= b.lomax;
  return r.lon >= b.lomin || r.lon <= b.lomax; // antimeridian-crossing bbox
}

export class GlobalSnapshot {
  readonly evictMs: number;
  readonly cap: number;
  private rowsByHex = new Map<string, SnapRow>();
  /** bumps on every content change (ETag / response-cache key) */
  version = 0;
  droppedAtCap = 0;
  evictedTotal = 0;
  lastIngestAt = 0;
  private perSourceLastAt: Record<string, number> = {};

  constructor(o: { evictMs?: number; cap?: number } = {}) {
    this.evictMs = o.evictMs ?? SNAPSHOT_EVICT_MS;
    this.cap = o.cap ?? SNAPSHOT_CAP;
  }

  get size(): number { return this.rowsByHex.size; }

  /** Merge one batch; returns how many rows changed. */
  ingest(batch: FixBatch, now: number = Date.now()): number {
    let changed = 0;
    for (const p of batch.aircraft || []) {
      const r = toSnapRow(p, batch);
      if (!r) continue;
      if (r.seenAt < now - this.evictMs) continue; // already too old to show
      const prev = this.rowsByHex.get(r.hex);
      if (prev) {
        const newer = r.seenAt > prev.seenAt + TIE_MS;
        const tieLawful = Math.abs(r.seenAt - prev.seenAt) <= TIE_MS
          && r.src === LAWFUL_PROVIDER && prev.src !== LAWFUL_PROVIDER;
        const tieSameSrcNewer = Math.abs(r.seenAt - prev.seenAt) <= TIE_MS
          && r.src === prev.src && r.seenAt > prev.seenAt;
        if (!newer && !tieLawful && !tieSameSrcNewer) continue;
        // identity carry-over (static airframe facts) — lawful-safe only
        if (prev.src === LAWFUL_PROVIDER || prev.src === r.src) {
          if (r.type == null) r.type = prev.type;
          if (r.reg == null) r.reg = prev.reg;
          if (r.cat == null) r.cat = prev.cat;
          if (r.callsign == null) r.callsign = prev.callsign;
        }
      }
      r.ingestAt = now;
      this.rowsByHex.set(r.hex, r);
      changed++;
    }
    if (changed) {
      this.version++;
      this.lastIngestAt = now;
      this.perSourceLastAt[batch.provider] = now;
      if (this.rowsByHex.size > this.cap) this.enforceCap();
    }
    return changed;
  }

  private enforceCap() {
    const over = this.rowsByHex.size - this.cap;
    if (over <= 0) return;
    const oldest = Array.from(this.rowsByHex.values()).sort((a, b) => a.seenAt - b.seenAt).slice(0, over);
    for (const r of oldest) this.rowsByHex.delete(r.hex);
    this.droppedAtCap += over;
  }

  /** Drop rows older than evictMs; returns the number evicted. */
  evict(now: number = Date.now()): number {
    const cut = now - this.evictMs;
    let n = 0;
    for (const [hex, r] of Array.from(this.rowsByHex.entries())) {
      if (r.seenAt < cut) { this.rowsByHex.delete(hex); n++; }
    }
    if (n) { this.version++; this.evictedTotal += n; }
    return n;
  }

  rows(opt: { bbox?: BBox | null; lawfulOnly?: boolean; sinceMs?: number | null; changedSinceMs?: number | null } = {}): SnapRow[] {
    const out: SnapRow[] = [];
    for (const r of this.rowsByHex.values()) {
      if (opt.lawfulOnly && r.src !== LAWFUL_PROVIDER) continue;
      if (opt.sinceMs != null && !(r.seenAt > opt.sinceMs)) continue;
      // >= (not >): a row ingested in the same ms the previous response was
      // built is re-sent rather than skipped — duplicates merge by hex
      if (opt.changedSinceMs != null && !((r.ingestAt ?? 0) >= opt.changedSinceMs)) continue;
      if (opt.bbox && !rowInBBox(r, opt.bbox)) continue;
      out.push(r);
    }
    return out;
  }

  get(hex: string): SnapRow | undefined {
    return this.rowsByHex.get(hex.toLowerCase());
  }

  /** rows per provenance + age distribution — the coverage block's input */
  summary(now: number = Date.now()) {
    const bySrc: Record<string, number> = {};
    const byVia: Record<string, number> = {};
    let fresh2m = 0;
    let oldest = Infinity;
    for (const r of Array.from(this.rowsByHex.values())) {
      bySrc[r.src] = (bySrc[r.src] || 0) + 1;
      byVia[r.via] = (byVia[r.via] || 0) + 1;
      if (now - r.seenAt <= 120_000) fresh2m++;
      if (r.seenAt < oldest) oldest = r.seenAt;
    }
    return {
      rows: this.rowsByHex.size,
      rows_by_source: bySrc,
      rows_by_path: byVia,
      rows_fresh_2m: fresh2m,
      oldest_row_age_s: Number.isFinite(oldest) ? Math.round((now - oldest) / 1000) : null,
      source_last_ingest_at: { ...this.perSourceLastAt },
      dropped_at_cap_total: this.droppedAtCap,
      evicted_total: this.evictedTotal,
    };
  }

  /** type histogram (non-null types) — feeds the sweep's type promotion */
  typeCounts(): Map<string, number> {
    const m = new Map<string, number>();
    for (const r of Array.from(this.rowsByHex.values())) {
      if (r.type) m.set(r.type, (m.get(r.type) || 0) + 1);
    }
    return m;
  }
}

/** Compact positional encoding (ROW_FIELDS order). gnd → 0/1. */
export function encodeRow(r: SnapRow): (string | number | null)[] {
  return [r.hex, r.lon, r.lat, r.altFt, r.gsKt, r.trk, r.callsign, r.type, r.seenAt,
    r.cat, r.gnd ? 1 : 0, r.reg, r.src];
}
