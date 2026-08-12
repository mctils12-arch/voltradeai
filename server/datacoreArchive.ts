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
import readline from "readline";
import { pipeline } from "stream/promises";

// 7 -> 30 (human directive 2026-08-11: "store all the data some how for up
// to a month for all planes... this should be for all thing track like
// boats plane"). Applies to every archive kind. Expected effect: raw
// (gzipped after ~2h) grows to roughly 4x the 7-day steady state; the
// global-scopes volume guard (1 GiB floor) pauses the fastest writer
// first, and the human's stated future direction is MORE storage, never
// offloading — so retention rises now and the guard is the safety.
// Rollup past this window still runs (coarse per-day polylines).
export const RAW_RETENTION_DAYS = 30;
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
  registration?: string | null; // tail number when broadcast (plane-tracking T1, 2026-08-08)
  // ── Navigation integrity / accuracy (readsb/tar1090 v2 broadcast/derived
  //    fields; GNSS-integrity passthrough 2026-08-11). NAMES THE OBSERVATION,
  //    not a cause: these record measured navigation-integrity/accuracy — low
  //    values have several candidate causes (RF interference, receiver
  //    geometry, avionics faults, coverage). Every field is NULLABLE and a
  //    reported 0 is a real category distinct from null (silence); the writer
  //    keeps that distinction. The archive never stored these before, so the
  //    series cannot be backfilled — capture starts now. ──
  nic?: number | null;       // Navigation Integrity Category (0 = unknown/no containment)
  nac_p?: number | null;     // Navigation Accuracy Category — position
  nac_v?: number | null;     // Navigation Accuracy Category — velocity
  sil?: number | null;       // Source Integrity Level
  sil_type?: string | null;  // SIL basis ("perhour" | "persample" | "unknown")
  rc?: number | null;        // Radius of Containment (m)
  gva?: number | null;       // Geometric Vertical Accuracy
  sda?: number | null;       // System Design Assurance
  nic_baro?: number | null;  // barometric-altitude integrity
  // ORIGIN discriminator (constraint: aircraft-broadcast vs ground-derived
  // must travel with the row). pos_type is the readsb position source:
  //   adsb_icao / adsb_icao_nt / adsr_icao / adsb_other = aircraft-broadcast
  //   mlat = ground-computed (multilateration);  tisb_* = ground-derived (TIS-B)
  //   mode_s = Mode-S (no ADS-B position). See ORIGIN_OF_POS_TYPE decode.
  pos_type?: string | null;
  mlat_fields?: string[] | null; // field names multilateration-derived (ground-computed), not broadcast
  tisb_fields?: string[] | null; // field names sourced from TIS-B (ground-derived)
  // Last-known-good position the feed still reports after a position-lock loss
  lkg_lat?: number | null;   // gpsOkLat
  lkg_lon?: number | null;   // gpsOkLon
  lkg_before?: number | null;// gpsOkBefore (seconds the LKG fix was last valid)
  seen_pos?: number | null;  // age (s) of the last position (feed seen_pos)
  // Provenance — which upstream served this row. Persisted as a real field so
  // an adsb.lol-only (ODbL) subset is a single filter and is provable; never
  // inferred later. adsblol | airplaneslive | adsbfi.
  provider?: string | null;
}

export type ArchiveOrigin = "broadcast" | "ground" | "mode_s" | "unknown";

/** ORIGIN decode for pos_type — the ONE place broadcast vs ground-derived is
 *  defined, so every API/export/UI reads the same table. "unknown" for an
 *  absent or unrecognised value (honest: never guessed). */
export function originOfPosType(posType: string | null | undefined): ArchiveOrigin {
  if (!posType) return "unknown";
  if (posType.startsWith("adsb") || posType.startsWith("adsr")) return "broadcast";
  if (posType === "mlat" || posType.startsWith("tisb")) return "ground";
  if (posType === "mode_s") return "mode_s";
  return "unknown";
}

/** Navigation-integrity / origin / provenance block for the aircraft archive
 *  JSONL. Short-key decode (single source of truth):
 *    ni nic · np nac_p · nv nac_v · si sil · st sil_type · rc rc · gv gva ·
 *    sd sda · nb nic_baro · pt pos_type · ml mlat-derived fields · tb tis-b
 *    fields · kla/klo last-known-good lat/lon · kb gpsOkBefore(s) · sp seen_pos
 *    age(s) · pv provider.
 *  NULL-IS-NOT-ZERO: every numeric uses a STRICT null check, so a reported 0
 *  is written as 0 and only a genuinely-absent field is omitted. NEVER
 *  `x || undefined` here — that would drop a real 0 and manufacture the exact
 *  integrity signal this archive exists to record. Empty mlat/tisb arrays are
 *  omitted (row-level origin already lives in pt); they carry meaning only
 *  when non-empty. */
export function aircraftIntegrityFields(p: AircraftPoint): Record<string, unknown> {
  const num = (x: number | null | undefined) => (x == null ? undefined : x);
  const arr = (x: string[] | null | undefined) => (x && x.length ? x : undefined);
  return {
    ni: num(p.nic), np: num(p.nac_p), nv: num(p.nac_v), si: num(p.sil),
    st: p.sil_type ?? undefined, rc: num(p.rc), gv: num(p.gva), sd: num(p.sda),
    nb: num(p.nic_baro), pt: p.pos_type ?? undefined,
    ml: arr(p.mlat_fields), tb: arr(p.tisb_fields),
    kla: num(p.lkg_lat), klo: num(p.lkg_lon), kb: num(p.lkg_before),
    sp: num(p.seen_pos), pv: p.provider ?? undefined,
  };
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

export function archiveBaseDir(): string {
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
  // oceanic/cruise 5min → 75s (2026-07-21, the twice-filed 3D-trail fix):
  // 5-min fixes at cruise ≈ 68-140km straight slabs in the curtain —
  // geometrically honest but visually broken at grazing angles (live
  // report). 75s ≈ 17-23km segments. Volume: ≤4× cruise rows, ≈ a few
  // MB/day gz per the filed build-first analysis; measurement-neutral
  // (raw fixes archived either way, only cadence changes).
  return 75_000;
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

/** TRACE BACKFILL writer (2026-08-08, nonstop tracker phase 2): archive
 *  fixes that carry their OWN timestamps (a fetched day-trace spans many
 *  hours), bucketed into the correct hour files. Append-only like every
 *  other writer; caller is responsible for not re-writing ranges it already
 *  backfilled (read-side t-dedupe in aircraftTrips.fullTrackAsync makes
 *  accidental overlap harmless — same-second same-hex fixes collapse). */
export function archiveAircraftAt(
  fixes: Array<AircraftPoint & { tSec: number }>, baseDir?: string,
): number {
  const base = baseDir || archiveBaseDir();
  const byHour = new Map<string, string[]>();
  for (const p of fixes) {
    if (p.lat == null || p.lon == null || !p.icao24 || !Number.isFinite(p.tSec)) continue;
    const d = new Date(p.tSec * 1000);
    const key = d.toISOString().slice(0, 13);
    const line = JSON.stringify({
      t: Math.floor(p.tSec), i: p.icao24, c: p.callsign || undefined,
      la: +p.lat.toFixed(4), lo: +p.lon.toFixed(4),
      al: p.altitude_m == null ? undefined : Math.round(p.altitude_m),
      g: p.on_ground || undefined,
      v: p.velocity_ms == null ? undefined : Math.round(p.velocity_ms),
      h: p.heading == null ? undefined : Math.round(p.heading),
      ty: p.type || undefined, ca: p.category || undefined,
      rg: p.registration || undefined,
      ...aircraftIntegrityFields(p),
    });
    const arr = byHour.get(key) || [];
    arr.push(line);
    byHour.set(key, arr);
  }
  let n = 0;
  for (const [, lines] of Array.from(byHour.entries())) {
    try {
      const at = new Date(lines.length ? JSON.parse(lines[0]).t * 1000 : Date.now());
      appendLines("aircraft", lines, base, at);
      n += lines.length;
    } catch (e: any) {
      console.error("[archive] aircraft backfill append:", e?.message || e);
    }
  }
  return n;
}

export function archiveAircraft(points: AircraftPoint[], sites: SitePoint[],
                                baseDir?: string, nowMs?: number,
                                // tracked-plane poller (2026-08-08): override
                                // the adaptive thinning so a followed tail
                                // number keeps EVERY polled fix, ground taxi
                                // included — the human's "track all the data
                                // nonstop". Additive; undefined = unchanged.
                                intervalMsOverride?: number): number {
  const base = baseDir || archiveBaseDir();
  const now = nowMs ?? Date.now();
  const t = Math.floor(now / 1000);
  const lines: string[] = [];
  for (const p of points) {
    if (p.lat == null || p.lon == null || !p.icao24) continue;
    if (!shouldWrite(`a:${p.icao24}`, intervalMsOverride ?? aircraftIntervalMs(p, sites), now)) continue;
    lines.push(JSON.stringify({
      t, i: p.icao24, c: p.callsign || undefined,
      la: +p.lat.toFixed(4), lo: +p.lon.toFixed(4),
      al: p.altitude_m == null ? undefined : Math.round(p.altitude_m),
      g: p.on_ground || undefined,
      v: p.velocity_ms == null ? undefined : Math.round(p.velocity_ms),
      h: p.heading == null ? undefined : Math.round(p.heading),
      ty: p.type || undefined, ca: p.category || undefined,
      // rg added 2026-08-08 (schema_version 2, additive): tail-number search
      // over history needs the registration IN the archive — the FAA spine
      // only covers matched US hexes
      rg: p.registration || undefined,
      // integrity/origin/provenance (schema_version 3, additive, 2026-08-11)
      ...aircraftIntegrityFields(p),
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
  const base = baseDir || archiveBaseDir();
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
  const base = baseDir || archiveBaseDir();
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
  const base = baseDir || archiveBaseDir();
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

/**
 * PERF (session #2, user-reported freezes): the sync compressOldHours above
 * gzipSyncs a whole hour file (tens of MB) in one event-loop turn on a
 * 30-minute timer — a periodic multi-second stall for every response AND
 * the trading loop, with zero user interaction. This variant selects the
 * same files but compresses each via streamed pipeline (read→gzip→write in
 * chunks; the loop breathes), verifying the .gz landed before unlinking the
 * raw (a failed pipeline removes its partial .gz and KEEPS the raw — the
 * archive is never the casualty of a compression error). In-flight latch:
 * a run outliving the timer interval never overlaps itself. The tiny
 * both-files-exist read window (readers scan .jsonl and .jsonl.gz) also
 * existed on the sync path; duplicate trail points sort/round away.
 */
let compressInFlight = false;
export async function compressOldHoursAsync(baseDir?: string, nowMs?: number): Promise<number> {
  if (compressInFlight) return 0;
  compressInFlight = true;
  try {
    const base = baseDir || archiveBaseDir();
    const now = nowMs ?? Date.now();
    let done = 0;
    for (const kind of ["aircraft", "vessels", "trains"] as const) {
      const dir = path.join(base, kind);
      if (!fs.existsSync(dir)) continue;
      let files: string[] = [];
      try { files = fs.readdirSync(dir); } catch { continue; }
      for (const f of files) {
        if (!f.endsWith(".jsonl")) continue;
        const stamp = f.replace(".jsonl", "");
        const fileMs = Date.parse(`${stamp.slice(0, 10)}T${stamp.slice(11, 13)}:00:00Z`);
        if (!Number.isFinite(fileMs) || now - fileMs < 2 * 3600_000) continue;
        const fp = path.join(dir, f);
        try {
          await pipeline(fs.createReadStream(fp), zlib.createGzip(), fs.createWriteStream(fp + ".gz"));
          await fs.promises.unlink(fp);
          done++;
        } catch (e: any) {
          console.error("[archive] gzip (async):", e?.message || e);
          await fs.promises.unlink(fp + ".gz").catch(() => {}); // drop the partial, keep the raw
        }
      }
    }
    return done;
  } finally {
    compressInFlight = false;
  }
}

/** Roll raw hours older than RAW_RETENTION_DAYS into per-entity daily track
 *  summaries, then delete the raw files. Summary: one JSON line per entity per
 *  day: {i, d, n, t0, t1, bbox, pl: coarse polyline (max ~50 pts)}. */
export function rollupOldDays(baseDir?: string, nowMs?: number): number {
  const base = baseDir || archiveBaseDir();
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
      const tracks: RollupTracks = {};
      let allReadOk = true;
      for (const f of files) {
        try {
          const fp = path.join(dir, f);
          const raw = f.endsWith(".gz") ? zlib.gunzipSync(fs.readFileSync(fp)).toString() : fs.readFileSync(fp, "utf8");
          for (const line of raw.split("\n")) {
            if (!line) continue;
            accumulateTrackLine(tracks, line);
          }
        } catch (e: any) {
          console.error("[archive] rollup read:", e?.message || e);
          allReadOk = false;
        }
      }
      // An unreadable/corrupt hour file must never be silently discarded —
      // hold the WHOLE day back (no summary write, no deletion) and retry
      // next run rather than deleting raw data that was never actually
      // rolled into the summary (audit finding: rollup was previously
      // deleting every file in the day group unconditionally, including
      // ones that failed to read).
      if (!allReadOk) {
        console.error(`[archive] rollup: skipping ${kind}/${day} (unreadable file), retry next run`);
        continue;
      }
      const out = emitDaySummary(tracks, day);
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

// Shared rollup internals — extracted so the sync path (tests, back-compat)
// and the streamed async path below are provably the same computation.
type RollupTracks = Record<string, { n: number; t0: number; t1: number;
  minLa: number; maxLa: number; minLo: number; maxLo: number;
  pl: Array<[number, number, number]> }>;

function accumulateTrackLine(tracks: RollupTracks, line: string): void {
  let r: any; try { r = JSON.parse(line); } catch { return; }
  const tr = (tracks[r.i] ||= { n: 0, t0: r.t, t1: r.t, minLa: r.la, maxLa: r.la, minLo: r.lo, maxLo: r.lo, pl: [] });
  tr.n++; tr.t0 = Math.min(tr.t0, r.t); tr.t1 = Math.max(tr.t1, r.t);
  tr.minLa = Math.min(tr.minLa, r.la); tr.maxLa = Math.max(tr.maxLa, r.la);
  tr.minLo = Math.min(tr.minLo, r.lo); tr.maxLo = Math.max(tr.maxLo, r.lo);
  tr.pl.push([r.t, r.la, r.lo]);
}

function emitDaySummary(tracks: RollupTracks, day: string): string[] {
  const out: string[] = [];
  for (const [id, tr] of Object.entries(tracks)) {
    tr.pl.sort((a, b) => a[0] - b[0]);
    const step = Math.max(1, Math.floor(tr.pl.length / 50));
    const pl = tr.pl.filter((_, idx) => idx % step === 0).map(([, la, lo]) => [la, lo]);
    out.push(JSON.stringify({ i: id, d: day, n: tr.n, t0: tr.t0, t1: tr.t1,
      bbox: [tr.minLa, tr.minLo, tr.maxLa, tr.maxLo], pl }));
  }
  return out;
}

/**
 * PERF (session #2): streamed rollup — the sync path above reads + parses
 * ENTIRE archived days (potentially hundreds of MB decompressed) in one
 * event-loop turn on a 6-hour timer. Same file selection, same accumulation
 * (shared helpers above), but hour files stream line-by-line so the loop
 * breathes; the summary write stays gzipSync (summaries are ~50 points per
 * entity — small). In-flight latch prevents self-overlap.
 */
let rollupInFlight = false;
export async function rollupOldDaysAsync(baseDir?: string, nowMs?: number): Promise<number> {
  if (rollupInFlight) return 0;
  rollupInFlight = true;
  try {
    const base = baseDir || archiveBaseDir();
    const now = nowMs ?? Date.now();
    const cutoff = now - RAW_RETENTION_DAYS * 86400_000;
    let rolled = 0;
    for (const kind of ["aircraft", "vessels", "trains"] as const) {
      const dir = path.join(base, kind);
      if (!fs.existsSync(dir)) continue;
      const byDay: Record<string, string[]> = {};
      let names: string[] = [];
      try { names = fs.readdirSync(dir); } catch { continue; }
      for (const f of names) {
        const m = f.match(/^(\d{4}-\d{2}-\d{2})-\d{2}\.jsonl(\.gz)?$/);
        if (!m) continue;
        const dayMs = Date.parse(m[1] + "T00:00:00Z");
        if (dayMs >= cutoff) continue;
        (byDay[m[1]] ||= []).push(f);
      }
      for (const [day, files] of Object.entries(byDay)) {
        const tracks: RollupTracks = {};
        let allReadOk = true;
        for (const f of files) {
          const ok = await streamJsonlLines(path.join(dir, f), f.endsWith(".gz"), (line) => accumulateTrackLine(tracks, line));
          if (!ok) allReadOk = false;
        }
        // Same fix as the sync path above: a file streamJsonlLines couldn't
        // read (corrupt gzip, vanished mid-scan) must hold the whole day
        // back instead of being deleted unrolled.
        if (!allReadOk) {
          console.error(`[archive] rollup (async): skipping ${kind}/${day} (unreadable file), retry next run`);
          continue;
        }
        const out = emitDaySummary(tracks, day);
        try {
          const tdir = path.join(base, kind + "_tracks");
          fs.mkdirSync(tdir, { recursive: true });
          fs.writeFileSync(path.join(tdir, `${day}.jsonl.gz`), zlib.gzipSync(out.join("\n") + "\n"));
          for (const f of files) fs.unlinkSync(path.join(dir, f));
          rolled++;
        } catch (e: any) { console.error("[archive] rollup write (async):", e?.message || e); }
      }
    }
    return rolled;
  } finally {
    rollupInFlight = false;
  }
}

// ── reads ────────────────────────────────────────────────────────────────────
/** Recent trail for one entity from today's + yesterday's raw hours. */
export function recentTrack(kind: ArchiveKind, id: string,
                            baseDir?: string, nowMs?: number, maxPoints = 500): Array<{ t: number; la: number; lo: number; al?: number }> {
  const base = baseDir || archiveBaseDir();
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

/**
 * PERF (EARTH TWIN session #2, user-reported freezes): the sync recentTrack
 * above readFileSync+gunzipSync+JSON.parses up to 48 hour-files IN ONE EVENT-
 * LOOP TURN — a multi-second stall that freezes every concurrent response AND
 * the trading loop (one Node process; eventLoopLag.ts audits exactly this at
 * ≥500ms), and the /data client re-fires it every 30s while a detail card is
 * open. This variant returns IDENTICAL results but streams each file through
 * createGunzip+readline in chunks (the loop breathes between chunks) and
 * skips JSON.parse for the ~99% of lines that cannot contain the id (cheap
 * substring prefilter — a SUPERSET filter; matches still verify r.i === id,
 * so output is exactly the sync path's). Sync recentTrack stays for tests /
 * any caller that genuinely wants it.
 */
export async function recentTrackAsync(kind: ArchiveKind, id: string,
                                       baseDir?: string, nowMs?: number, maxPoints = 500): Promise<Array<{ t: number; la: number; lo: number; al?: number }>> {
  const base = baseDir || archiveBaseDir();
  const now = nowMs ?? Date.now();
  const dir = path.join(base, kind);
  if (!fs.existsSync(dir)) return [];
  const days = [new Date(now), new Date(now - 86400_000)].map((d) => d.toISOString().slice(0, 10));
  const pts: Array<{ t: number; la: number; lo: number; al?: number }> = [];
  let files: string[] = [];
  try { files = fs.readdirSync(dir).sort(); } catch { return []; }
  for (const f of files) {
    if (!days.some((d) => f.startsWith(d))) continue;
    await streamJsonlLines(path.join(dir, f), f.endsWith(".gz"), (line) => {
      if (!line.includes(id)) return; // prefilter: parse only candidate lines
      try {
        const r = JSON.parse(line);
        if (r.i === id) pts.push({ t: r.t, la: r.la, lo: r.lo, al: r.al });
      } catch {}
    });
  }
  pts.sort((a, b) => a.t - b.t);
  return pts.slice(-maxPoints);
}

/**
 * Stream a (possibly gzipped) JSONL file line-by-line, yielding to the event
 * loop between chunks. Event-based per file: stream errors (corrupt gz,
 * vanished file) bail that file only — mirrors the sync paths' per-file
 * try/catch. for-await is avoided deliberately: fs-stream errors don't
 * propagate through pipe() to the readline iterator and would crash the
 * process. Shared by recentTrackAsync + rollupOldDaysAsync.
 */
export function streamJsonlLines(fp: string, isGz: boolean, onLine: (line: string) => void): Promise<boolean> {
  return new Promise<boolean>((resolve) => {
    let src: fs.ReadStream;
    try { src = fs.createReadStream(fp); } catch { resolve(false); return; }
    const input = isGz ? src.pipe(zlib.createGunzip()) : src;
    const rl = readline.createInterface({ input, crlfDelay: Infinity });
    // Return value: true = the file streamed to completion with no error;
    // false = it bailed partway (corrupt/truncated gz, vanished file) — a
    // caller that deletes source files after reading MUST treat false as
    // "do not delete, this file was not fully accounted for" (see the
    // rollupOldDaysAsync fix this return value exists for).
    // resolve(false) BEFORE rl.close(): readline's close() emits its
    // 'close' event synchronously, so calling it first would let the
    // rl.on("close", ...) handler below win the promise with resolve(true)
    // — found via this fix's own regression test.
    const bail = () => { resolve(false); try { rl.close(); } catch {} };
    src.on("error", bail);
    if (input !== src) (input as NodeJS.ReadableStream).on("error", bail);
    // readline.Interface re-emits its input stream's 'error' on ITSELF too
    // (a separate EventEmitter emission from the input.on("error", ...)
    // above) — with no listener here, a corrupt/truncated .gz crashes the
    // WHOLE PROCESS (verified: this exact file's src/input guards do NOT
    // prevent it). Found 2026-07-22 while building an unrelated feature;
    // the same missing-rl-error-listener pattern was copy-pasted into 7
    // other files (aircraftEntities/fleetUtilization/gridStress/
    // platformStats/queryEngine/shadowFleet(x2)/siteTimeline) — fixed there
    // too, same PR.
    rl.on("error", bail);
    rl.on("line", (line) => { if (line) onLine(line); });
    rl.on("close", () => resolve(true));
  });
}

// Short-TTL track cache: the /data client refreshes an open card's trail
// every 30s and users re-click the same entity — each was a full archive
// scan. FIFO-capped; injectable clock for tests.
const trackCache = new Map<string, { at: number; points: Array<{ t: number; la: number; lo: number; al?: number }> }>();
const TRACK_CACHE_TTL_MS = 30_000;
const TRACK_CACHE_MAX = 64;

export async function recentTrackCached(kind: ArchiveKind, id: string,
                                        baseDir?: string, nowMs?: number): Promise<Array<{ t: number; la: number; lo: number; al?: number }>> {
  const now = nowMs ?? Date.now();
  const key = `${kind}:${id}:${baseDir ?? ""}`;
  const hit = trackCache.get(key);
  if (hit && now - hit.at < TRACK_CACHE_TTL_MS) return hit.points;
  const points = await recentTrackAsync(kind, id, baseDir, nowMs);
  trackCache.set(key, { at: now, points });
  if (trackCache.size > TRACK_CACHE_MAX) {
    const oldest = trackCache.keys().next().value;
    if (oldest !== undefined) trackCache.delete(oldest);
  }
  return points;
}

/** Test seam: reset the track cache between hermetic cases. */
export function clearTrackCache(): void {
  trackCache.clear();
}

/**
 * List the archive files for one stream/day, covering BOTH file-naming
 * conventions already live in datacore/ (per each stream's manifest
 * "storage" field): day-granularity streams like usaspending write
 * `<dir>/YYYY-MM-DD.jsonl(.gz)` directly; position streams (aircraft/
 * vessels/trains) write hour files `<dir>/YYYY-MM-DD-HH.jsonl(.gz)`.
 * Detecting by what's actually on disk (rather than hardcoding a
 * per-stream naming table here) means a new stream needs zero changes
 * to this reader to become readable.
 */
function archiveDayFiles(dir: string, day: string): string[] {
  const exact = [`${day}.jsonl`, `${day}.jsonl.gz`]
    .map((f) => path.join(dir, f)).filter((fp) => fs.existsSync(fp));
  if (exact.length) return exact;
  let names: string[] = [];
  try { names = fs.readdirSync(dir); } catch { return []; }
  return names.filter((f) => f.startsWith(day)).sort().map((f) => path.join(dir, f));
}

/**
 * Generic one-day archive read for ANY datacore stream (wishlist item
 * filed 2026-07-26: the USAspending gate-2 statistical test — award/mcap
 * ratio vs. 5-20d forward returns — has no way to read the multi-week
 * historical archive from outside the Railway volume; /api/diag/archive
 * in server/bot.ts is the read-only, token-gated surface this backs).
 * `stream` must already exist as a real archive directory (see the
 * caller's whitelist check) — this function itself does no path
 * validation beyond a plain path.join, so callers MUST reject anything
 * that isn't `^[a-z0-9_]+$` before reaching here.
 */
export async function readArchiveDay(
  stream: string, day: string, baseDir?: string, limit = 1000,
): Promise<{ dir: string; files: string[]; rows: any[]; truncated: boolean } | null> {
  const base = baseDir || archiveBaseDir();
  const dir = path.join(base, stream);
  if (!fs.existsSync(dir) || !fs.statSync(dir).isDirectory()) return null;
  const files = archiveDayFiles(dir, day);
  const rows: any[] = [];
  let truncated = false;
  for (const fp of files) {
    if (rows.length >= limit) { truncated = true; break; }
    await streamJsonlLines(fp, fp.endsWith(".gz"), (line) => {
      if (rows.length >= limit) { truncated = true; return; }
      try { rows.push(JSON.parse(line)); } catch {}
    });
  }
  return { dir, files: files.map((f) => path.basename(f)), rows, truncated };
}

export function archiveStats(baseDir?: string): any {
  const base = baseDir || archiveBaseDir();
  const out: any = { base, kinds: {} };
  // [REPAIR 2026-07-05, audit defect #3] Enumerate the archive from DISK
  // instead of a hardcoded kind list — the old six-kind list left fires,
  // filings, earnings8k, filings13f, fredmacro, optionchains, usaspending,
  // fda, usgswater, gdelt invisible to /api/data/archive/stats, making
  // the archive-gap rule ("gaps are findings") unenforceable for most of
  // the archive. The position kinds stay listed explicitly so they report
  // {files:0} even before their first write — a missing position archive
  // must be loud, not absent.
  const kinds = new Set<string>(["aircraft", "vessels", "trains", "aircraft_tracks", "vessels_tracks", "trains_tracks"]);
  try {
    for (const e of fs.readdirSync(base, { withFileTypes: true })) {
      if (e.isDirectory()) kinds.add(e.name);
    }
  } catch {}
  for (const kind of Array.from(kinds).sort()) {
    const dir = path.join(base, kind);
    if (!fs.existsSync(dir)) { out.kinds[kind] = { files: 0, bytes: 0 }; continue; }
    let bytes = 0, files = 0, oldest: string | null = null, newest: string | null = null;
    for (const f of fs.readdirSync(dir).sort()) {
      const st = fs.statSync(path.join(dir, f));
      if (!st.isFile()) continue;
      files++; bytes += st.size;
      oldest = oldest || f; newest = f;
    }
    out.kinds[kind] = { files, bytes, oldest, newest };
  }
  out.totalBytes = Object.values(out.kinds).reduce((s: number, k: any) => s + k.bytes, 0);
  return out;
}
