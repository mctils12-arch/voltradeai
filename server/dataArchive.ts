// ─── Position Archive (MAP V2 ROADMAP R1 — CLAUDE.md open_questions.md) ──────
// Raw ADS-B/AIS positions are ephemeral (in-memory only) upstream of this
// module. R2's maritime-transit signal and the archive-enabled hypotheses
// (corporate jet activity, tanker routing anomalies, destination prediction)
// all depend on OUR OWN accumulating history — a day not recorded here is
// unrecoverable. This module is the recorder: time-gated sampling (not every
// poll — clients may poll every 10-30s) into compact, append-only, per-day
// files on the Railway volume, with size-bounded retention.
//
// Pure helpers are exported and unit-tested directly (dataArchive.test.ts);
// the stateful recordX functions carry module-level "last write" timestamps
// so they're cheap to call from every request handler and self-throttle.
import fs from "fs";
import path from "path";

export type ArchiveKind = "aircraft" | "vessels";

// Sampling interval: bounds disk growth independent of client poll rate.
// At 30 min: aircraft (~800 recs/sample, ~35B/rec compact array) ~= 1.3MB/day
// = ~40MB/mo; vessels (~1500 recs/sample, ~30B/rec) ~= 2.1MB/day = ~65MB/mo.
// Combined ~105MB/mo, roughly matching the wishlist.md volume estimate.
export const SAMPLE_INTERVAL_MS = 30 * 60_000;

// Raw daily files older than this are pruned (rolled into the tiny summary
// file first) — geofence-level transit counters (R2) are a future gate-1
// build on top of the summary, not raw-position replay this far back.
export const RETENTION_DAYS = 90;

export function dayKeyUTC(ts: number): string {
  return new Date(ts).toISOString().slice(0, 10); // YYYY-MM-DD
}

export function archiveDir(dataDir: string, kind: ArchiveKind): string {
  return path.join(dataDir, "archive", kind);
}

export function archiveFilePath(dataDir: string, kind: ArchiveKind, day: string): string {
  return path.join(archiveDir(dataDir, kind), `${day}.jsonl`);
}

export function rollupFilePath(dataDir: string, kind: ArchiveKind): string {
  return path.join(dataDir, "archive", `${kind}_rollup.json`);
}

export function shouldSample(lastWriteAt: number | undefined, now: number, intervalMs: number): boolean {
  return lastWriteAt === undefined || now - lastWriteAt >= intervalMs;
}

// [icao24, lat, lon, altitude_m|null, heading|null] — positional array, not an
// object, so no repeated field names bloat every one of ~800 lines/sample.
export type CompactAircraft = [string, number, number, number | null, number | null];
export type CompactVessel = [string, number, number, number | null, number | null];

const round4 = (n: number) => Math.round(n * 10_000) / 10_000; // ~11m precision

export function compactAircraftRecord(a: { icao24: string; lat: number; lon: number; altitude_m?: number | null; heading?: number | null }): CompactAircraft {
  return [a.icao24, round4(a.lat), round4(a.lon), a.altitude_m ?? null, a.heading != null ? Math.round(a.heading) : null];
}

export function compactVesselRecord(v: { mmsi: string; lat: number; lon: number; sog?: number | null; cog?: number | null }): CompactVessel {
  return [v.mmsi, round4(v.lat), round4(v.lon), v.sog ?? null, v.cog != null ? Math.round(v.cog) : null];
}

export function buildSampleLine(t: number, records: Array<CompactAircraft | CompactVessel>): string {
  return JSON.stringify({ t, n: records.length, recs: records });
}

interface Rollup {
  totalSamples: number;
  totalRecords: number;
  firstDay: string | null;
  lastDay: string | null;
  days: Record<string, { samples: number; records: number }>;
}

function loadRollup(dataDir: string, kind: ArchiveKind): Rollup {
  try {
    const raw = fs.readFileSync(rollupFilePath(dataDir, kind), "utf8");
    return JSON.parse(raw);
  } catch {
    return { totalSamples: 0, totalRecords: 0, firstDay: null, lastDay: null, days: {} };
  }
}

function saveRollup(dataDir: string, kind: ArchiveKind, rollup: Rollup): void {
  fs.mkdirSync(path.dirname(rollupFilePath(dataDir, kind)), { recursive: true });
  fs.writeFileSync(rollupFilePath(dataDir, kind), JSON.stringify(rollup));
}

function updateRollupForSample(dataDir: string, kind: ArchiveKind, day: string, recordCount: number): void {
  const rollup = loadRollup(dataDir, kind);
  rollup.totalSamples += 1;
  rollup.totalRecords += recordCount;
  rollup.firstDay = rollup.firstDay === null || day < rollup.firstDay ? day : rollup.firstDay;
  rollup.lastDay = rollup.lastDay === null || day > rollup.lastDay ? day : rollup.lastDay;
  const d = rollup.days[day] || { samples: 0, records: 0 };
  d.samples += 1;
  d.records += recordCount;
  rollup.days[day] = d;
  saveRollup(dataDir, kind, rollup);
}

// Deletes raw daily files past RETENTION_DAYS. Their counts already live in
// the rollup (updated per-sample above), so nothing is lost except the
// ability to replay exact positions from that far back.
function pruneOldArchiveFiles(dataDir: string, kind: ArchiveKind, now: number): void {
  const dir = archiveDir(dataDir, kind);
  let files: string[];
  try {
    files = fs.readdirSync(dir);
  } catch {
    return;
  }
  const cutoff = dayKeyUTC(now - RETENTION_DAYS * 86_400_000);
  for (const f of files) {
    const day = f.replace(/\.jsonl$/, "");
    if (day < cutoff) {
      try { fs.unlinkSync(path.join(dir, f)); } catch {}
    }
  }
}

const lastWrite: Record<ArchiveKind, number | undefined> = { aircraft: undefined, vessels: undefined };

function recordSample(dataDir: string, kind: ArchiveKind, records: Array<CompactAircraft | CompactVessel>, now: number): void {
  if (!shouldSample(lastWrite[kind], now, SAMPLE_INTERVAL_MS)) return;
  lastWrite[kind] = now;
  if (records.length === 0) return; // nothing to record (feed empty/down) — don't burn a rollup entry on zero data
  const day = dayKeyUTC(now);
  const file = archiveFilePath(dataDir, kind, day);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.appendFileSync(file, buildSampleLine(now, records) + "\n");
  updateRollupForSample(dataDir, kind, day, records.length);
  pruneOldArchiveFiles(dataDir, kind, now);
}

export function recordAircraftSample(dataDir: string, aircraft: Array<{ icao24: string; lat: number; lon: number; altitude_m?: number | null; heading?: number | null }>, now: number): void {
  recordSample(dataDir, "aircraft", aircraft.map(compactAircraftRecord), now);
}

export function recordVesselSample(dataDir: string, vessels: Array<{ mmsi: string; lat: number; lon: number; sog?: number | null; cog?: number | null }>, now: number): void {
  recordSample(dataDir, "vessels", vessels.map(compactVesselRecord), now);
}

export interface ArchiveStats {
  intervalMinutes: number;
  retentionDays: number;
  kinds: Record<ArchiveKind, {
    daysOnDisk: number;
    totalSamples: number;
    totalRecords: number;
    firstDay: string | null;
    lastDay: string | null;
    approxBytesOnDisk: number;
  }>;
}

export function getArchiveStats(dataDir: string): ArchiveStats {
  const kinds: any = {};
  for (const kind of ["aircraft", "vessels"] as ArchiveKind[]) {
    const rollup = loadRollup(dataDir, kind);
    let approxBytesOnDisk = 0;
    let daysOnDisk = 0;
    try {
      const dir = archiveDir(dataDir, kind);
      const files = fs.readdirSync(dir);
      daysOnDisk = files.length;
      for (const f of files) {
        try { approxBytesOnDisk += fs.statSync(path.join(dir, f)).size; } catch {}
      }
    } catch {}
    kinds[kind] = {
      daysOnDisk,
      totalSamples: rollup.totalSamples,
      totalRecords: rollup.totalRecords,
      firstDay: rollup.firstDay,
      lastDay: rollup.lastDay,
      approxBytesOnDisk,
    };
  }
  return { intervalMinutes: SAMPLE_INTERVAL_MS / 60_000, retentionDays: RETENTION_DAYS, kinds };
}
