/**
 * datacoreArchive.ts — the permanent position archive (SPINOUT-READY layer).
 *
 * ARCHIVE EVERYTHING (human directive 2026-07-03): every aircraft/vessel
 * position we ingest is recorded from day one — unrecorded days are
 * unrecoverable proprietary data. This module is pure data-layer: no trading
 * imports, no knowledge of trading logic (datacore boundary rules).
 *
 * Engineering:
 *  - Adaptive thinning: full resolution near strategic sites and for
 *    low-altitude flight / slow-or-maneuvering vessels; sparser sampling for
 *    oceanic/cruise traffic. Per-entity last-written timestamps enforce the
 *    cadence.
 *  - Storage: append-only JSONL, one file per UTC hour, on the Railway
 *    volume (/data/voltrade/datacore_archive; /tmp fallback locally).
 *  - Compression: files older than ~2h are gzipped in place.
 *  - Rollup: raw hours older than RAW_RETENTION_DAYS are summarized into
 *    per-entity daily track records (first/last/bbox/n + a coarse polyline),
 *    then the raw files are deleted.
 *  - Volume watch: stats() reports bytes/files so growth is observable
 *    (wishlist flag threshold; see research/wishlist.md).
 *
 * All functions take an optional baseDir for hermetic tests.
 */
import fs from "fs";
import path from "path";
import zlib from "zlib";

export const RAW_RETENTION_DAYS = 7;
export type ArchiveKind = "aircraft" | "vessels" | "trains";

// Strategic sites (lat, lon) — near these we keep full resolution. Loaded
// from the bundled datacore sites JSON at import time by the caller and
// passed in, to keep this module dependency-free and testable.
export interface SitePoint { lat: number; lon: number }

export interface AircraftPoint {
  icao24: string;
  callsign?: string;
  lat: number;
  lon: number;
  altitude_m: number | null;
  on_ground: boolean;
  velocity_ms: number | null;
  heading: number | null;
  type?: string | null;      // ICAO type designator when the feed provides it
  category?: string | null;  // ADS-B emitter category when provided
}

export interface VesselPoint {
  mmsi: string;
  name?: string;
  lat: number;
  lon: number;
  sog: number | null;        // knots
  cog: number | null;
  shiptype?: number | null;  // AIS ship-type code when broadcast
  destination?: string | null;
}

function defaultBase(): string {
  const dataDir = process.env.DATA_DIR || (fs.existsSync("/data") ? "/data/voltrade" : "/tmp");
  return path.join(dataDir, "datacore_archive");
}

function hourFile(kind: ArchiveKind, when: Date, base: string): string {
  const d = when.toISOString().slice(0, 13); // YYYY-MM-DDTHH
  return path.join(base, kind, `${d.replace("T", "-")}.jsonl`);
}

// ── adaptive thinning ────────────────────────────────────────────────────────
const kmBetween = (aLat: number, aLon: number, bLat: number, bLon: number) => {
  const R = 6371, dLat = ((bLat - aLat) * Math.PI) / 180, dLon = ((bLon - aLon) * Math.PI) / 180;
  const s = Math.sin(dLat / 2) ** 2 +
    Math.cos((aLat * Math.PI) / 180) * Math.cos((bLat * Math.PI) / 180) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(s));
};

export function nearAnySite(lat: number, lon: number, sites: SitePoint[], km = 150): boolean {
  return sites.some((s) => kmBetween(lat, lon, s.lat, s.lon) <= km);
}

/** Sampling interval (ms) for an aircraft point under adaptive thinning. */
export function aircraftIntervalMs(p: AircraftPoint, sites: SitePoint[]): number {
  if (p.on_ground) return 5 * 60_000;                       // ground: 5 min
  if (nearAnySite(p.lat, p.lon, sites)) return 30_000;      // near strategic sites: 30s
  if (p.altitude_m != null && p.altitude_m < 3000) return 60_000;   // low altitude: 1 min
  return 5 * 60_000;                                        // oceanic/cruise: 5 min
}

/** Sampling interval (ms) for a vessel point under adaptive thinning. */
export function vesselIntervalMs(p: VesselPoint, sites: SitePoint[]): number {
  if (nearAnySite(p.lat, p.lon, sites, 80)) return 2 * 60_000;   // near ports/sites: 2 min
  if (p.sog != null && p.sog < 1) return 30 * 60_000;            // anchored: 30 min
  return 10 * 60_000;                                            // open water: 10 min
}

/** Sampling interval (ms) for a train point: fixed cadence — trains are
 *  few (hundreds, not 10k), slow relative to aircraft, and not near-site
 *  weighted (no rail strategic sites yet). 2 min balances track fidelity
 *  against volume. */
export function trainIntervalMs(): number {
  return 2 * 60_000;
}

// Per-entity last-write clock (in-memory; a restart just writes one extra
// sample per entity, which is harmless).
const lastWrite: Map<string, number> = new Map();

function shouldWrite(key: string, intervalMs: number, now: number): boolean {
  const last = lastWrite.get(key) || 0;
  if (now - last < intervalMs) return false;
  lastWrite.set(key, now);
  if (lastWrite.size > 100_000) lastWrite.clear(); // bound memory, worst case = extra samples
  return true;
}

// ── append ───────────────────────────────────────────────────────────────────
function appendLines(kind: ArchiveKind, lines: string[], base: string, now: Date) {
  if (!lines.length) return;
  const fp = hourFile(kind, now, base);
  fs.mkdirSync(path.dirname(fp), { recursive: true });
  fs.appendFileSync(fp, lines.join("\n") + "\n");
}

export function archiveAircraft(points: AircraftPoint[], sites: SitePoint[],
                                baseDir?: string, nowMs?: number): number {
  const base = baseDir || defaultBase();
  const now = nowMs ?? Date.now();
  const t = Math.floor(now / 1000);
  const lines: string[] = [];
  for (const p of points) {
    if (p.lat == null || p.lon == null || !p.icao24) continue;
    if (!shouldWrite(`a:${p.icao24}`, aircraftIntervalMs(p, sites), now)) continue;
    lines.push(JSON.stringify({
      t, i: p.icao24, c: p.callsign || undefined,
      la: +p.lat.toFixed(4), lo: +p.lon.toFixed(4),
      al: p.altitude_m == null ? undefined : Math.round(p.altitude_m),
      g: p.on_ground || undefined,
      v: p.velocity_ms == null ? undefined : Math.round(p.velocity_ms),
      h: p.heading == null ? undefined : Math.round(p.heading),
      ty: p.type || undefined, ca: p.category || undefined,
    }));
  }
  try { appendLines("aircraft", lines, base, new Date(now)); } catch (e: any) {
    console.error("[archive] aircraft append:", e?.message || e);
    return 0;
  }
  return lines.length;
}

export function archiveVessels(points: VesselPoint[], sites: SitePoint[],
                               baseDir?: string, nowMs?: number): number {
  const base = baseDir || defaultBase();
  const now = nowMs ?? Date.now();
  const t = Math.floor(now / 1000);
  const lines: string[] = [];
  for (const p of points) {
    if (p.lat == null || p.lon == null || !p.mmsi) continue;
    if (!shouldWrite(`v:${p.mmsi}`, vesselIntervalMs(p, sites), now)) continue;
    lines.push(JSON.stringify({
      t, i: p.mmsi, c: p.name || undefined,
      la: +p.lat.toFixed(4), lo: +p.lon.toFixed(4),
      v: p.sog == null ? undefined : Math.round(p.sog * 10) / 10,
      h: p.cog == null ? undefined : Math.round(p.cog),
      st: p.shiptype ?? undefined, de: p.destination || undefined,
    }));
  }
  try { appendLines("vessels", lines, base, new Date(now)); } catch (e: any) {
    console.error("[archive] vessels append:", e?.message || e);
    return 0;
  }
  return lines.length;
}

export interface TrainPoint {
  id: string;            // country-prefixed, e.g. "FI-62" / "NO-71-12"
  country: string;       // coverage tag shown on the map
  lat: number; lon: number;
  speed_kmh?: number | null;
  bearing?: number | null;
  label?: string | null; // train number / line ref
}

export function archiveTrains(points: TrainPoint[], baseDir?: string, nowMs?: number): number {
  const base = baseDir || defaultBase();
  const now = nowMs ?? Date.now();
  const t = Math.floor(now / 1000);
  const lines: string[] = [];
  for (const p of points) {
    if (p.lat == null || p.lon == null || !p.id) continue;
    if (!shouldWrite(`t:${p.id}`, trainIntervalMs(), now)) continue;
    lines.push(JSON.stringify({
      t, i: p.id, c: p.label || undefined, co: p.country,
      la: +p.lat.toFixed(4), lo: +p.lon.toFixed(4),
      v: p.speed_kmh == null ? undefined : Math.round(p.speed_kmh),
      h: p.bearing == null ? undefined : Math.round(p.bearing),
    }));
  }
  try { appendLines("trains", lines, base, new Date(now)); } catch (e: any) {
    console.error("[archive] trains append:", e?.message || e);
    return 0;
  }
  return lines.length;
}

// ── compression + rollup (maintenance; call periodically) ───────────────────
export function compressOldHours(baseDir?: string, nowMs?: number): number {
  const base = baseDir || defaultBase();
  const now = nowMs ?? Date.now();
  let done = 0;
  for (const kind of ["aircraft", "vessels", "trains"] as const) {
    const dir = path.join(base, kind);
    if (!fs.existsSync(dir)) continue;
    for (const f of fs.readdirSync(dir)) {
      if (!f.endsWith(".jsonl")) continue;
      // file hour from name: YYYY-MM-DD-HH.jsonl
      const stamp = f.replace(".jsonl", "");
      const fileMs = Date.parse(`${stamp.slice(0, 10)}T${stamp.slice(11, 13)}:00:00Z`);
      if (!Number.isFinite(fileMs) || now - fileMs < 2 * 3600_000) continue;
      const fp = path.join(dir, f);
      try {
        const gz = zlib.gzipSync(fs.readFileSync(fp));
        fs.writeFileSync(fp + ".gz", gz);
        fs.unlinkSync(fp);
        done++;
      } catch (e: any) {
        console.error("[archive] gzip:", e?.message || e);
      }
    }
  }
  return done;
}

/** Roll raw hours older than RAW_RETENTION_DAYS into per-entity daily track
 *  summaries, then delete the raw files. Summary: one JSON line per entity per
 *  day: {i, d, n, t0, t1, bbox, pl: coarse polyline (max ~50 pts)}. */
export function rollupOldDays(baseDir?: string, nowMs?: number): number {
  const base = baseDir || defaultBase();
  const now = nowMs ?? Date.now();
  const cutoff = now - RAW_RETENTION_DAYS * 86400_000;
  let rolled = 0;
  for (const kind of ["aircraft", "vessels", "trains"] as const) {
    const dir = path.join(base, kind);
    if (!fs.existsSync(dir)) continue;
    // group files by day
    const byDay: Record<string, string[]> = {};
    for (const f of fs.readdirSync(dir)) {
      const m = f.match(/^(\d{4}-\d{2}-\d{2})-\d{2}\.jsonl(\.gz)?$/);
      if (!m) continue;
      const dayMs = Date.parse(m[1] + "T00:00:00Z");
      if (dayMs >= cutoff) continue;
      (byDay[m[1]] ||= []).push(f);
    }
    for (const [day, files] of Object.entries(byDay)) {
      const tracks: Record<string, { n: number; t0: number; t1: number;
        minLa: number; maxLa: number; minLo: number; maxLo: number;
        pl: Array<[number, number, number]> }> = {};
      for (const f of files) {
        try {
          const fp = path.join(dir, f);
          const raw = f.endsWith(".gz") ? zlib.gunzipSync(fs.readFileSync(fp)).toString() : fs.readFileSync(fp, "utf8");
          for (const line of raw.split("\n")) {
            if (!line) continue;
            let r: any; try { r = JSON.parse(line); } catch { continue; }
            const tr = (tracks[r.i] ||= { n: 0, t0: r.t, t1: r.t, minLa: r.la, maxLa: r.la, minLo: r.lo, maxLo: r.lo, pl: [] });
            tr.n++; tr.t0 = Math.min(tr.t0, r.t); tr.t1 = Math.max(tr.t1, r.t);
            tr.minLa = Math.min(tr.minLa, r.la); tr.maxLa = Math.max(tr.maxLa, r.la);
            tr.minLo = Math.min(tr.minLo, r.lo); tr.maxLo = Math.max(tr.maxLo, r.lo);
            tr.pl.push([r.t, r.la, r.lo]);
          }
        } catch (e: any) { console.error("[archive] rollup read:", e?.message || e); }
      }
      const out: string[] = [];
      for (const [id, tr] of Object.entries(tracks)) {
        tr.pl.sort((a, b) => a[0] - b[0]);
        const step = Math.max(1, Math.floor(tr.pl.length / 50));
        const pl = tr.pl.filter((_, idx) => idx % step === 0).map(([, la, lo]) => [la, lo]);
        out.push(JSON.stringify({ i: id, d: day, n: tr.n, t0: tr.t0, t1: tr.t1,
          bbox: [tr.minLa, tr.minLo, tr.maxLa, tr.maxLo], pl }));
      }
      try {
        const tdir = path.join(base, kind + "_tracks");
        fs.mkdirSync(tdir, { recursive: true });
        fs.writeFileSync(path.join(tdir, `${day}.jsonl.gz`), zlib.gzipSync(out.join("\n") + "\n"));
        for (const f of files) fs.unlinkSync(path.join(dir, f));
        rolled++;
      } catch (e: any) { console.error("[archive] rollup write:", e?.message || e); }
    }
  }
  return rolled;
}

// ── reads ────────────────────────────────────────────────────────────────────
/** Recent trail for one entity from today's + yesterday's raw hours. */
export function recentTrack(kind: ArchiveKind, id: string,
                            baseDir?: string, nowMs?: number, maxPoints = 500): Array<{ t: number; la: number; lo: number; al?: number }> {
  const base = baseDir || defaultBase();
  const now = nowMs ?? Date.now();
  const dir = path.join(base, kind);
  if (!fs.existsSync(dir)) return [];
  const days = [new Date(now), new Date(now - 86400_000)].map((d) => d.toISOString().slice(0, 10));
  const pts: Array<{ t: number; la: number; lo: number; al?: number }> = [];
  for (const f of fs.readdirSync(dir).sort()) {
    if (!days.some((d) => f.startsWith(d))) continue;
    try {
      const fp = path.join(dir, f);
      const raw = f.endsWith(".gz") ? zlib.gunzipSync(fs.readFileSync(fp)).toString() : fs.readFileSync(fp, "utf8");
      for (const line of raw.split("\n")) {
        if (!line) continue;
        try {
          const r = JSON.parse(line);
          if (r.i === id) pts.push({ t: r.t, la: r.la, lo: r.lo, al: r.al });
        } catch {}
      }
    } catch {}
  }
  pts.sort((a, b) => a.t - b.t);
  return pts.slice(-maxPoints);
}

export function archiveStats(baseDir?: string): any {
  const base = baseDir || defaultBase();
  const out: any = { base, kinds: {} };
  for (const kind of ["aircraft", "vessels", "trains", "aircraft_tracks", "vessels_tracks", "trains_tracks"]) {
    const dir = path.join(base, kind);
    if (!fs.existsSync(dir)) { out.kinds[kind] = { files: 0, bytes: 0 }; continue; }
    let bytes = 0, files = 0, oldest: string | null = null, newest: string | null = null;
    for (const f of fs.readdirSync(dir).sort()) {
      files++; bytes += fs.statSync(path.join(dir, f)).size;
      oldest = oldest || f; newest = f;
    }
    out.kinds[kind] = { files, bytes, oldest, newest };
  }
  out.totalBytes = Object.values(out.kinds).reduce((s: number, k: any) => s + k.bytes, 0);
  return out;
}
