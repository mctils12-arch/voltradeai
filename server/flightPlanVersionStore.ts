// FLIGHT PLAN VERSIONS (2026-11-16): every version of a filed plan, with its time.
//
// server/swimSfdps.ts keeps ONE current plan per flight and a COUNT of
// amendments. This module keeps the versions themselves: version 1 is the plan
// as we first received its route, and each later version is recorded when
// planDiff (the store's own change test: route text, route geometry,
// destination, cruise altitude) says the plan changed. A message that changes
// none of those records nothing.
//
// NOT A VERSION EITHER (2026-11-17, from 307 recorded changes on 111 live
// flights): a message whose route only RESTATES the route held — the FAA
// re-issues an airborne route from the aircraft's position ("./.", often with
// a fix-radial-distance position that keeps moving) and still sends the full
// text — and a message whose route text the flight already had in an EARLIER
// version, because different centers hold different texts for one flight and
// their messages alternate. Both are counted (routeRestatements,
// earlierTextAgain). What the second rule misses: a clearance that genuinely
// puts a flight back on a route text it had before is not recorded as a new
// version. Fixes dropped from the START of a present-position route are
// still a version; whether they were flown or skipped is judged when the
// history is served, from the aircraft's position (shared/flightPlanDiff.ts).
//
// TWO TIERS
//   memory  every version of every plan the SWIM store holds, bounded by the
//           caps below and dropped with the plan (plus one hour after a flight
//           is cancelled or completed, so a just-landed flight can be read).
//   disk    <archive>/flight_plan_versions/YYYY-MM-DD-HH.jsonl, CHANGES ONLY:
//           a flight's version 1 is written when its first change arrives,
//           together with that change. A flight that is never amended never
//           touches the disk (counted: originalsNotArchived). Reason and
//           numbers are in the DISK COST note below.
//
// HONESTY
//   * Version 1 is "as filed" only when it came from a flight-plan (FH)
//     message that was the first message of any kind we saw for the flight,
//     with no amendment seen before it. Otherwise it is the plan as first
//     SEEN (we joined the flight late, or restarted) and is flagged
//     originalAsFiled=false: earlier versions may exist that we never held.
//   * `receivedAtMs` is our clock; `swimMessageMs` is the message's own
//     timestamp when it carried one.
//   * Nothing is dropped silently: every skipped write and every version
//     dropped from memory has a counter on /api/data/aircraft/plan-status.
//
// LADD: a callsign known to be in the FAA LADD privacy program is neither held
// nor written (server/laddRegistry.ts). An aircraft learned to be LADD later
// is withheld at read time by the serving module, hex and callsign both.
//
// DISK COST (feed rates measured 2026-11-16 on the live site, the rest estimated)
//   Feed, 3.2 h after a restart: 25,219 FH and 82,459 AH messages, i.e. about
//   7,900 FH and 25,800 AH an hour in US daytime.
//   Share of AH messages that change something we version: 9 of 59 amendment
//   events in our own flight_events archive (5 tracked flights, 30 days), 15%.
//   A small sample; 7-27% is plausible.
//   Changes a day: 25,800 x 18 daytime-equivalent hours x 15% = about 70,000,
//   plus one original per changed flight (1.8 changes a flight: about 39,000).
//   Line size, measured on 11,024 built lines (real FAA fix names, routes of
//   4-14 points, a 110-character route text, a real-length GUFI): 552 bytes
//   with a route, 205 without, gzip 5.2:1.
//   -> about 109,000 lines, 55 MB raw, 10.5 MB gzipped a day
//      (26-99 MB raw, 5-19 MB gzipped across the 7-27% range).
//   Writing every flight's version 1 as well (60,000-140,000 more lines) would
//   add 33-77 MB raw a day and pass 15 MB gzipped, which is why the disk tier
//   is changes only.
//   HARD BOUNDS, whatever the estimate: FPV_DAY_CAP_BYTES (72 MiB) of raw
//   lines per UTC day, about 14 MB gzipped; FPV_RETENTION_DAYS (7, short
//   because the volume is tight: raise it when there is room); hour files
//   gzipped after two hours; the FPV_MIN_FREE_BYTES low-disk floor (fails
//   closed); and the switch
//   FLIGHT_PLAN_VERSIONS_ARCHIVE=0. Worst case on disk: 7 x 14 MB = about
//   100 MB, plus up to three hours of raw lines (about 15 MB).
//   If the true share of changing amendments is above about 20%, the daily cap
//   skips lines late in the UTC day; the skipped.day_cap counter shows it.

import { isRouteRestatement } from "../shared/flightPlanDiff";
import fs from "fs";
import path from "path";
import zlib from "zlib";
import { promisify } from "util";
import { appendLines, archiveBaseDir, streamJsonlLines } from "./datacoreArchive";
import { MIN_FREE_BYTES, readFreeBytes } from "./globalScopes";
import { registerHeapGauge } from "./heapDiag";
import {
  packRoute, planDiff, SWIM_STORE_MAX,
  type StoredSwimPlan, type SwimFlightMessage, type SwimRouteKind, type SwimRoutePoint,
} from "./swimSfdps";

// ── caps (memory) ───────────────────────────────────────────────────────────
// MEMORY MATH. One held version = a small object (about 160 B) + its own route
// Float32Array (12 B a point + about 100 B header) + the route text and point
// names (1 B a character; shared with the SWIM store while it is the current
// version). A typical 10-point, 110-character plan is about 0.55 KB; the
// largest the SWIM store allows (250 points, 1,000 characters) is about 6 KB.
//   live today: 5,464 plans x about 1.3 versions = 7,100 versions = about 4 MB
//   at FPV_MAX_VERSIONS_TOTAL, typical: 30,000 x 0.55 KB = 16.5 MB
//   at FPV_MAX_VERSIONS_TOTAL, all at the largest size: 180 MB, which
//   FPV_MAX_BYTES (48 MiB, counted with the same arithmetic) cuts off first.
/** versions kept per flight: version 1 and the newest ones; the middle is dropped and counted */
export const FPV_MAX_VERSIONS_PER_FLIGHT = 12;
/** flights with held versions (the SWIM plan store's own cap) */
export const FPV_MAX_FLIGHTS = SWIM_STORE_MAX;
/** versions across all flights */
export const FPV_MAX_VERSIONS_TOTAL = 30_000;
/** arithmetic estimate of retained bytes across all versions */
export const FPV_MAX_BYTES = 48 * 1024 * 1024;
/** a cancelled / completed flight's versions stay readable this long */
export const FPV_ENDED_KEEP_MS = 60 * 60_000;
const VERSION_BASE_BYTES = 260;

// ── caps (disk) ─────────────────────────────────────────────────────────────
export const FPV_STREAM = "flight_plan_versions";
/** "0" turns the disk tier off; memory keeps working */
export const FPV_ARCHIVE_ENV = "FLIGHT_PLAN_VERSIONS_ARCHIVE";
/** writes stop below this much free space. It sits 256 MiB ABOVE the 1 GiB
 *  floor that protects bot state (globalScopes MIN_FREE_BYTES), so this stream
 *  stops before it can push the volume to that floor. Unreadable free space
 *  fails CLOSED, like the sweep archive's gate. */
export const FPV_MIN_FREE_BYTES = MIN_FREE_BYTES + 256 * 1024 * 1024;
/** raw line bytes per UTC day; lines past it are skipped and counted */
export const FPV_DAY_CAP_BYTES = 72 * 1024 * 1024;
export const FPV_RETENTION_DAYS = 7;
export const FPV_FLUSH_MS = 15_000;
/** lines waiting for the next flush; past it lines are skipped and counted */
export const FPV_BUFFER_MAX_LINES = 4_000;
export const FPV_COMPRESS_AFTER_MS = 2 * 3600_000;
export const FPV_MAINTAIN_EVERY_MS = 3600_000;
const FREE_RECHECK_MS = 60_000;
const DAY_MS = 86_400_000;

// ── types ───────────────────────────────────────────────────────────────────
export interface PlanVersion {
  /** 1-based sequence number within this process run (version 1 = first route we held) */
  n: number;
  /** our clock when the message was applied */
  receivedAtMs: number;
  /** the SWIM message's own timestamp, when it carried one */
  swimMessageMs: number | null;
  /** SFDPS source code of the message that produced this version (FH, AH, ...) */
  messageType: string | null;
  departure: string | null;
  arrival: string | null;
  cruiseAltFt: number | null;
  routeText: string | null;
  /** packed [lat, lon, altFt|NaN] triplets, as the SWIM store packs them */
  route: Float32Array;
  routeNames: string;
  routeKind?: SwimRouteKind;
  /** planDiff's own words against the previous version (null on version 1) */
  detail: string | null;
  heldIn: "memory" | "archive";
}

export interface FlightVersions {
  key: string;
  callsign: string | null;
  versions: PlanVersion[];
  /** versions recorded this run (the newest version's n) */
  recorded: number;
  /** versions no longer in `versions` (per-flight cap) */
  droppedFromMemory: number;
  /** when the SWIM store first saw the flight (may predate version 1) */
  planFirstSeenMs: number;
  /** see HONESTY in the header */
  originalAsFiled: boolean;
  /** set when the flight was cancelled / completed */
  endedAtMs: number | null;
  lastChangeMs: number;
  bytes: number;
  /** what the disk tier holds for this flight */
  disk: { originalWritten: boolean; text: string | null; route: Float32Array | null };
}

const sameRoute = (a: Float32Array | null, b: Float32Array | null): boolean => {
  if (a === b) return true;
  if (!a || !b || a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i] && !(Number.isNaN(a[i]) && Number.isNaN(b[i]))) return false;
  }
  return true;
};
const versionBytes = (v: PlanVersion) =>
  VERSION_BASE_BYTES + v.route.byteLength + (v.routeText?.length ?? 0) + v.routeNames.length;

/** unpack a version's route (5 dp, like the SWIM store's routePointsOf) */
export function versionRoutePoints(v: Pick<PlanVersion, "route" | "routeNames">): SwimRoutePoint[] {
  const names = v.routeNames ? v.routeNames.split("|") : [];
  const out: SwimRoutePoint[] = [];
  for (let i = 0; i < v.route.length / 3; i++) {
    const alt = v.route[i * 3 + 2];
    const name = names[i];
    out.push({
      lat: Math.round(v.route[i * 3] * 1e5) / 1e5,
      lon: Math.round(v.route[i * 3 + 1] * 1e5) / 1e5,
      ...(name ? { name } : {}),
      ...(Number.isFinite(alt) ? { altFt: Math.round(alt) } : {}),
    });
  }
  return out;
}

// ── disk line format ────────────────────────────────────────────────────────
// One JSON object per line, short keys. `rs`/`ps` = 1 means "route text" /
// "route points" are the same as the previous line of this flight; the writer
// only uses them when that previous line was itself queued for writing.
interface VersionLine {
  k: string; n: number; t: number; mt?: number; ty?: string; cs?: string;
  dep?: string; arr?: string; alt?: number;
  rt?: string; rs?: 1;
  rk?: string; pn?: string; pc?: number[]; pz?: Array<number | null>; ps?: 1;
  d?: string; o?: 0 | 1; fs?: number;
}

export function versionToLine(rec: FlightVersions, v: PlanVersion, same: { text: boolean; route: boolean }): string {
  const o: VersionLine = { k: rec.key, n: v.n, t: v.receivedAtMs };
  if (v.swimMessageMs != null) o.mt = v.swimMessageMs;
  if (v.messageType) o.ty = v.messageType;
  if (rec.callsign) o.cs = rec.callsign;
  if (v.departure) o.dep = v.departure;
  if (v.arrival) o.arr = v.arrival;
  if (v.cruiseAltFt != null) o.alt = v.cruiseAltFt;
  if (same.text) o.rs = 1; else if (v.routeText) o.rt = v.routeText;
  if (same.route) o.ps = 1;
  else if (v.route.length) {
    const pts = versionRoutePoints(v);
    if (v.routeKind) o.rk = v.routeKind;
    o.pn = v.routeNames;
    o.pc = pts.flatMap((p) => [p.lat, p.lon]);
    if (pts.some((p) => p.altFt != null)) o.pz = pts.map((p) => p.altFt ?? null);
  }
  if (v.detail) o.d = v.detail;
  if (v.n === 1) { o.o = rec.originalAsFiled ? 1 : 0; o.fs = rec.planFirstSeenMs; }
  return JSON.stringify(o);
}

const ROUTE_KINDS: SwimRouteKind[] = ["expanded", "airway", "direct", "radial", "corridor"];
const str = (x: unknown): string | null => (typeof x === "string" && x ? x : null);
const fin = (x: unknown): number | null => (typeof x === "number" && Number.isFinite(x) ? x : null);

/** One archived line, still carrying its "same as previous" markers. */
export interface ArchivedVersionRow {
  key: string; callsign: string | null; version: PlanVersion;
  sameText: boolean; sameRoute: boolean;
  originalAsFiled: boolean | null; planFirstSeenMs: number | null;
}

export function parseVersionLine(line: string): ArchivedVersionRow | null {
  let o: Record<string, unknown>;
  try { o = JSON.parse(line) as Record<string, unknown>; } catch { return null; }
  if (!o || typeof o !== "object") return null;
  const key = str(o.k), n = fin(o.n), t = fin(o.t);
  if (!key || n == null || t == null) return null;
  const pc = Array.isArray(o.pc) ? o.pc : [];
  const pz = Array.isArray(o.pz) ? o.pz : [];
  const names = (str(o.pn) ?? "").split("|");
  const pts: SwimRoutePoint[] = [];
  for (let i = 0; i + 1 < pc.length; i += 2) {
    const lat = fin(pc[i]), lon = fin(pc[i + 1]);
    if (lat == null || lon == null) continue;
    const nm = names[i / 2], alt = fin(pz[i / 2]);
    pts.push({ lat, lon, ...(nm ? { name: nm } : {}), ...(alt != null ? { altFt: alt } : {}) });
  }
  const packed = packRoute(pts);
  const rkRaw = str(o.rk);
  const rk = ROUTE_KINDS.find((x) => x === rkRaw);
  return {
    key, callsign: str(o.cs),
    version: {
      n, receivedAtMs: t, swimMessageMs: fin(o.mt), messageType: str(o.ty),
      departure: str(o.dep), arrival: str(o.arr), cruiseAltFt: fin(o.alt),
      routeText: str(o.rt), route: packed.route, routeNames: packed.routeNames,
      ...(rk ? { routeKind: rk } : {}),
      detail: str(o.d), heldIn: "archive",
    },
    sameText: o.rs === 1, sameRoute: o.ps === 1,
    originalAsFiled: o.o === 1 ? true : o.o === 0 ? false : null,
    planFirstSeenMs: fin(o.fs),
  };
}

// ── disk tier ───────────────────────────────────────────────────────────────
export type ArchiveSkipReason = "disabled" | "low_disk" | "day_cap" | "buffer_full" | "write_error";

export interface PlanVersionArchiveOptions {
  baseDir?: string;
  env?: NodeJS.ProcessEnv;
  now?: () => number;
  /** free-space probe (tests inject; default statfs on the archive dir) */
  readFree?: (dir: string) => number | null;
  /** the writer (tests inject; default datacoreArchive.appendLines) */
  append?: (lines: string[], base: string, at: Date) => void;
  dayCapBytes?: number;
  minFreeBytes?: number;
}

const gzip = promisify(zlib.gzip);
const HOUR_FILE_RE = /^(\d{4}-\d{2}-\d{2})-(\d{2})\.jsonl(\.gz)?$/;
const hourOfFile = (name: string): number | null => {
  const m = HOUR_FILE_RE.exec(name);
  if (!m) return null;
  const t = Date.parse(`${m[1]}T${m[2]}:00:00Z`);
  return Number.isFinite(t) ? t : null;
};

export class PlanVersionArchive {
  private buffer: string[] = [];
  private bufferBytes = 0;
  private freeAt = 0;
  private freeBytes: number | null = null;
  private lowDiskWarned = false;
  private day = "";
  private dayBytes = 0;
  private lastMaintainAt = 0;
  private errorCount = 0;
  readonly counters = {
    linesWritten: 0, bytesWritten: 0, flushes: 0,
    skipped: { disabled: 0, low_disk: 0, day_cap: 0, buffer_full: 0, write_error: 0 } as Record<ArchiveSkipReason, number>,
    filesCompressed: 0, filesExpired: 0, maintenanceErrors: 0,
    lastFlushAt: null as number | null, lastError: null as string | null,
  };

  constructor(private opts: PlanVersionArchiveOptions = {}) {}

  private now(): number { return (this.opts.now ?? Date.now)(); }
  get baseDir(): string { return this.opts.baseDir || archiveBaseDir(); }
  get dir(): string { const base = this.baseDir; return path.join(base, "flight_plan_versions"); }
  get enabled(): boolean { return String((this.opts.env ?? process.env)[FPV_ARCHIVE_ENV] ?? "1").trim() !== "0"; }

  private guardAllows(t: number): boolean {
    if (this.freeAt === 0 || t - this.freeAt > FREE_RECHECK_MS) {
      this.freeAt = t;
      this.freeBytes = (this.opts.readFree ?? readFreeBytes)(this.baseDir);
    }
    const floor = this.opts.minFreeBytes ?? FPV_MIN_FREE_BYTES;
    const ok = this.freeBytes != null && this.freeBytes >= floor; // unreadable = closed
    if (!ok && !this.lowDiskWarned) {
      this.lowDiskWarned = true;
      console.error(`[plan-versions] disk writes PAUSED: ${this.freeBytes == null ? "free space unreadable (fails closed)" : `${this.freeBytes} bytes free`} < ${floor} floor; versions stay in memory and skipped lines are counted`);
    } else if (ok && this.lowDiskWarned) {
      this.lowDiskWarned = false;
      console.error("[plan-versions] disk headroom back above the floor: writes resumed");
    }
    return ok;
  }

  /** Queue the line(s) for a flight's newest version: its version 1 first when
   *  that is not on disk yet. Returns the reason when nothing was queued. */
  write(rec: FlightVersions, v: PlanVersion): ArchiveSkipReason | null {
    const t = this.now();
    const pending: PlanVersion[] = [];
    if (!rec.disk.originalWritten && rec.versions[0] && rec.versions[0] !== v) pending.push(rec.versions[0]);
    pending.push(v);
    const skip = (why: ArchiveSkipReason): ArchiveSkipReason => {
      this.counters.skipped[why] += pending.length;
      // the next line that does get written must stand on its own
      rec.disk.text = null; rec.disk.route = null;
      return why;
    };
    if (!this.enabled) return skip("disabled");
    if (!this.guardAllows(t)) return skip("low_disk");
    if (this.buffer.length + pending.length > FPV_BUFFER_MAX_LINES) return skip("buffer_full");
    const day = new Date(t).toISOString().slice(0, 10);
    if (day !== this.day) { this.day = day; this.dayBytes = 0; }
    let text = rec.disk.text, route = rec.disk.route;
    const lines: string[] = [];
    let bytes = 0;
    for (const x of pending) {
      const line = versionToLine(rec, x, {
        text: text != null && x.routeText === text,
        route: route != null && sameRoute(x.route, route),
      });
      lines.push(line); bytes += Buffer.byteLength(line) + 1;
      text = x.routeText; route = x.route;
    }
    if (this.dayBytes + bytes > (this.opts.dayCapBytes ?? FPV_DAY_CAP_BYTES)) return skip("day_cap");
    this.dayBytes += bytes;
    this.buffer.push(...lines); this.bufferBytes += bytes;
    rec.disk.originalWritten = true; rec.disk.text = text; rec.disk.route = route;
    return null;
  }

  get pendingLines(): number { return this.buffer.length; }

  /** Append everything queued. A failed append is counted, never retried
   *  (a retry could double-write), and reported once and then every 100th time. */
  flush(): number {
    if (!this.buffer.length) return 0;
    const lines = this.buffer, bytes = this.bufferBytes;
    this.buffer = []; this.bufferBytes = 0;
    const t = this.now();
    try {
      (this.opts.append ?? ((ls, base, at) => appendLines("flight_plan_versions", ls, base, at)))(lines, this.baseDir, new Date(t));
      this.counters.linesWritten += lines.length;
      this.counters.bytesWritten += bytes;
      this.counters.flushes++;
      this.counters.lastFlushAt = t;
      return lines.length;
    } catch (e) {
      this.counters.skipped.write_error += lines.length;
      this.counters.lastError = e instanceof Error ? e.message : String(e);
      if (this.errorCount++ % 100 === 0) console.error(`[plan-versions] append failed (x${this.errorCount}), ${lines.length} lines lost and counted:`, this.counters.lastError);
      return 0;
    }
  }

  /** gzip hour files older than two hours; delete files past the retention */
  async maintain(nowMs = this.now()): Promise<{ compressed: number; expired: number }> {
    let names: string[] = [];
    try { names = await fs.promises.readdir(this.dir); } catch (e) {
      if ((e as { code?: string })?.code !== "ENOENT") this.counters.maintenanceErrors++;
      return { compressed: 0, expired: 0 };
    }
    let compressed = 0, expired = 0;
    for (const f of names) {
      const hour = hourOfFile(f);
      if (hour == null) continue;
      const fp = path.join(this.dir, f);
      try {
        if (nowMs - hour > FPV_RETENTION_DAYS * DAY_MS) { await fs.promises.unlink(fp); expired++; continue; }
        if (f.endsWith(".jsonl") && nowMs - hour > FPV_COMPRESS_AFTER_MS) {
          const gz = await gzip(await fs.promises.readFile(fp));
          await fs.promises.writeFile(`${fp}.gz.tmp`, gz);
          await fs.promises.rename(`${fp}.gz.tmp`, `${fp}.gz`);
          await fs.promises.unlink(fp);
          compressed++;
        }
      } catch (e) {
        this.counters.maintenanceErrors++;
        this.counters.lastError = e instanceof Error ? e.message : String(e);
      }
    }
    this.counters.filesCompressed += compressed;
    this.counters.filesExpired += expired;
    return { compressed, expired };
  }

  /** flush every FPV_FLUSH_MS, maintain hourly; returns the stop function */
  start(flushMs = FPV_FLUSH_MS): () => void {
    const timer = setInterval(() => {
      this.flush();
      const t = this.now();
      if (t - this.lastMaintainAt >= FPV_MAINTAIN_EVERY_MS) {
        this.lastMaintainAt = t;
        void this.maintain(t).catch((e: unknown) => {
          this.counters.maintenanceErrors++;
          this.counters.lastError = e instanceof Error ? e.message : String(e);
        });
      }
    }, flushMs);
    timer.unref?.();
    return () => { clearInterval(timer); this.flush(); };
  }

  status() {
    return {
      enabled: this.enabled,
      stream: FPV_STREAM,
      policy: "changes only: a flight's version 1 is written with its first change; never-amended flights are not written",
      ...this.counters, skipped: { ...this.counters.skipped },
      pendingLines: this.buffer.length,
      dayBytes: this.dayBytes, dayCapBytes: this.opts.dayCapBytes ?? FPV_DAY_CAP_BYTES,
      freeBytes: this.freeBytes, minFreeBytes: this.opts.minFreeBytes ?? FPV_MIN_FREE_BYTES,
      retentionDays: FPV_RETENTION_DAYS,
    };
  }
}

// ── read-back ───────────────────────────────────────────────────────────────
export const FPV_READBACK_MAX_FILES = 30;
export const FPV_READBACK_BUDGET_MS = 2_500;

export interface ArchivedFlight {
  key: string; callsign: string | null; versions: PlanVersion[];
  originalAsFiled: boolean | null; planFirstSeenMs: number | null;
  /** versions whose route referred to an earlier line that was not in the scan */
  routeUnknown: number;
}

/** Versions written for one callsign in hour files between fromMs and toMs
 *  (file hour = when the line was WRITTEN, so a version 1 sits in the file of
 *  the change that caused it to be written). Bounded by file count and time. */
export async function readArchivedVersions(opts: {
  callsign: string; fromMs: number; toMs: number; baseDir?: string;
  maxFiles?: number; budgetMs?: number;
}): Promise<{ flights: ArchivedFlight[]; filesScanned: number; complete: boolean; badLines: number }> {
  const base = opts.baseDir || archiveBaseDir();
  const dir = path.join(base, "flight_plan_versions");
  let names: string[] = [];
  try { names = await fs.promises.readdir(dir); } catch (e) {
    return { flights: [], filesScanned: 0, complete: (e as { code?: string })?.code === "ENOENT", badLines: 0 };
  }
  const wanted = names
    .map((f) => ({ f, hour: hourOfFile(f) }))
    .filter((x): x is { f: string; hour: number } => x.hour != null && x.hour + 3600_000 > opts.fromMs && x.hour <= opts.toMs)
    .sort((a, b) => a.hour - b.hour || a.f.localeCompare(b.f));
  const maxFiles = opts.maxFiles ?? FPV_READBACK_MAX_FILES;
  const picked = wanted.slice(-maxFiles); // the newest hours win
  let complete = picked.length === wanted.length;
  const started = Date.now();
  const rows: ArchivedVersionRow[] = [];
  const needle = `"cs":"${opts.callsign}"`;
  let filesScanned = 0, badLines = 0;
  for (const { f } of picked) {
    if (Date.now() - started > (opts.budgetMs ?? FPV_READBACK_BUDGET_MS)) { complete = false; break; }
    const ok = await streamJsonlLines(path.join(dir, f), f.endsWith(".gz"), (line) => {
      if (!line.includes(needle)) return;
      const row = parseVersionLine(line);
      if (!row) { badLines++; return; }
      if (row.callsign === opts.callsign) rows.push(row);
    });
    if (!ok) complete = false;
    filesScanned++;
  }
  const byKey = new Map<string, ArchivedFlight>();
  const lastFull = new Map<string, PlanVersion>();
  for (const r of rows) {
    let fl = byKey.get(r.key);
    if (!fl) { fl = { key: r.key, callsign: r.callsign, versions: [], originalAsFiled: null, planFirstSeenMs: null, routeUnknown: 0 }; byKey.set(r.key, fl); }
    if (fl.versions.some((x) => x.n === r.version.n && x.receivedAtMs === r.version.receivedAtMs)) continue; // the same line twice
    const prev = lastFull.get(r.key);
    const v = r.version;
    if (r.sameText) { if (prev) v.routeText = prev.routeText; else fl.routeUnknown++; }
    if (r.sameRoute) {
      if (prev) { v.route = prev.route; v.routeNames = prev.routeNames; if (prev.routeKind) v.routeKind = prev.routeKind; }
      else if (!r.sameText) fl.routeUnknown++;
    }
    lastFull.set(r.key, v);
    // the EARLIEST version 1 speaks for the flight (a later run restarts at n=1 as "first seen")
    if (v.n === 1 && !fl.versions.length) { fl.originalAsFiled = r.originalAsFiled; fl.planFirstSeenMs = r.planFirstSeenMs; }
    fl.versions.push(v);
  }
  const flights = Array.from(byKey.values());
  for (const fl of flights) fl.versions.sort((a, b) => a.receivedAtMs - b.receivedAtMs || a.n - b.n);
  return { flights, filesScanned, complete, badLines };
}

// ── memory tier ─────────────────────────────────────────────────────────────
export type MemoryDropReason = "per_flight_cap" | "total_cap" | "bytes_cap" | "flights_cap" | "plan_expired" | "ended_grace_over";

export class PlanVersionStore {
  private byKey = new Map<string, FlightVersions>(); // insertion order = last change
  private keysByCallsign = new Map<string, Set<string>>();
  private ended = new Map<string, number>();          // key -> endedAt, insertion order = end order
  private versionCount = 0;
  private bytes = 0;
  readonly counters = {
    versionsRecorded: 0, firstVersions: 0, changeVersions: 0,
    /** plan-content messages that changed nothing we version (no version recorded) */
    messagesNoChange: 0,
    /** messages whose route text only restated the same remaining route from
     *  the aircraft's position (or sent the full text again), with nothing
     *  else changed: not a version (measured 2026-11-17: at least 22% of the
     *  changes recorded before this rule) */
    routeRestatements: 0,
    /** messages whose route text equals an EARLIER version still held for the
     *  flight (not the newest), with nothing else changed: another center's
     *  view of the same flight, not a version */
    earlierTextAgain: 0,
    /** a held version gained a field it lacked (not a new version) */
    fieldsFilled: 0,
    skippedLadd: 0,
    versionsDropped: { per_flight_cap: 0, total_cap: 0, bytes_cap: 0, flights_cap: 0, plan_expired: 0, ended_grace_over: 0 } as Record<MemoryDropReason, number>,
    /** flights that left memory with version 1 never written to disk (never amended, or every write skipped) */
    originalsNotArchived: 0,
  };

  constructor(private opts: {
    archive?: PlanVersionArchive | null;
    /** FAA LADD membership by callsign; such flights are neither held nor written */
    isLadd?: (callsign: string | null) => boolean;
    maxPerFlight?: number; maxFlights?: number; maxVersions?: number; maxBytes?: number; endedKeepMs?: number;
  } = {}) {}

  get flights(): number { return this.byKey.size; }
  get versions(): number { return this.versionCount; }
  get estimatedBytes(): number { return this.bytes; }
  get archive(): PlanVersionArchive | null { return this.opts.archive ?? null; }

  /** SwimPlanStore.onPlanContent: record version 1, a new version, or nothing. */
  record(plan: StoredSwimPlan, msg: Pick<SwimFlightMessage, "messageType" | "timestamp">, now: number): PlanVersion | null {
    if (!plan.routeText && !plan.route.length) return null; // no route yet: nothing to version
    let rec = this.byKey.get(plan.key);
    if (!rec) {
      if (this.opts.isLadd?.(plan.callsign)) { this.counters.skippedLadd++; return null; }
      const v = this.versionOf(plan, msg, 1, now, null);
      rec = {
        key: plan.key, callsign: plan.callsign, versions: [v], recorded: 1, droppedFromMemory: 0,
        planFirstSeenMs: plan.firstSeen,
        originalAsFiled: v.messageType === "FH" && plan.firstSeen === now && plan.amendments === 0,
        endedAtMs: null, lastChangeMs: now, bytes: versionBytes(v),
        disk: { originalWritten: false, text: null, route: null },
      };
      this.byKey.set(rec.key, rec);
      this.index(rec.callsign, rec.key);
      this.versionCount++; this.bytes += rec.bytes;
      this.counters.versionsRecorded++; this.counters.firstVersions++;
      this.evict(now);
      return v;
    }
    const last = rec.versions[rec.versions.length - 1];
    const detail = planDiff(last, plan);
    if (!detail) {
      this.fill(rec, last, plan);
      this.counters.messagesNoChange++;
      return null;
    }
    const otherChange = (plan.arrival && last.arrival && plan.arrival !== last.arrival)
      || (plan.cruiseAltFt != null && last.cruiseAltFt != null && plan.cruiseAltFt !== last.cruiseAltFt);
    if (!otherChange && isRouteRestatement(last.routeText, plan.routeText)) {
      this.fill(rec, last, plan);
      this.counters.routeRestatements++;
      return null;
    }
    // different FAA centers hold different texts for one flight and their
    // messages alternate, so a text this flight already had in an earlier
    // version is that other view again, not a new amendment. (A genuine
    // return to an earlier route is missed by this rule; see the header.)
    if (!otherChange && plan.routeText && rec.versions.some((x) => x !== last && x.routeText === plan.routeText)) {
      this.counters.earlierTextAgain++;
      return null;
    }
    const v = this.versionOf(plan, msg, rec.recorded + 1, now, detail);
    rec.versions.push(v); rec.recorded = v.n; rec.lastChangeMs = now;
    const vb = versionBytes(v);
    rec.bytes += vb; this.bytes += vb; this.versionCount++;
    if (rec.endedAtMs != null) { rec.endedAtMs = null; this.ended.delete(rec.key); }
    if (plan.callsign !== rec.callsign) { this.unindex(rec.callsign, rec.key); rec.callsign = plan.callsign; this.index(rec.callsign, rec.key); }
    this.counters.versionsRecorded++; this.counters.changeVersions++;
    // disk first: the per-flight cap below may drop a middle version, never version 1 or this one
    this.opts.archive?.write(rec, v);
    const cap = this.opts.maxPerFlight ?? FPV_MAX_VERSIONS_PER_FLIGHT;
    while (rec.versions.length > cap && rec.versions.length > 2) {
      const [gone] = rec.versions.splice(1, 1); // version 1 is the original: keep it
      const gb = versionBytes(gone);
      rec.bytes -= gb; this.bytes -= gb; this.versionCount--;
      rec.droppedFromMemory++; this.counters.versionsDropped.per_flight_cap++;
    }
    this.byKey.delete(rec.key); this.byKey.set(rec.key, rec); // LRU by last change
    this.evict(now);
    return v;
  }

  private versionOf(plan: StoredSwimPlan, msg: Pick<SwimFlightMessage, "messageType" | "timestamp">, n: number, now: number, detail: string | null): PlanVersion {
    return {
      n, receivedAtMs: now, swimMessageMs: msg.timestamp ?? null,
      // the STORED (detached) copy of the type, never the payload's own string
      messageType: msg.messageType ? plan.messageType : null,
      departure: plan.departure, arrival: plan.arrival, cruiseAltFt: plan.cruiseAltFt,
      routeText: plan.routeText, route: plan.route, routeNames: plan.routeNames,
      ...(plan.routeKind ? { routeKind: plan.routeKind } : {}),
      detail, heldIn: "memory",
    };
  }

  /** a later message carried a field the newest version lacked (planDiff does
   *  not call null -> value a change): complete the version in place */
  private fill(rec: FlightVersions, v: PlanVersion, plan: StoredSwimPlan): void {
    const before = versionBytes(v);
    let filled = false;
    if (v.departure == null && plan.departure) { v.departure = plan.departure; filled = true; }
    if (v.arrival == null && plan.arrival) { v.arrival = plan.arrival; filled = true; }
    if (v.cruiseAltFt == null && plan.cruiseAltFt != null) { v.cruiseAltFt = plan.cruiseAltFt; filled = true; }
    if (v.routeText == null && plan.routeText) { v.routeText = plan.routeText; filled = true; }
    if (!v.route.length && plan.route.length) {
      v.route = plan.route; v.routeNames = plan.routeNames;
      if (plan.routeKind) v.routeKind = plan.routeKind;
      filled = true;
    }
    if (!filled) return;
    const d = versionBytes(v) - before;
    rec.bytes += d; this.bytes += d;
    rec.disk.text = null; rec.disk.route = null; // a written copy of this version is now incomplete
    this.counters.fieldsFilled++;
  }

  /** SwimPlanStore.onRemoved */
  planRemoved(key: string, reason: "ended" | "expired", now: number): void {
    const rec = this.byKey.get(key);
    if (!rec) return;
    if (reason === "ended") {
      rec.endedAtMs = now;
      this.ended.delete(key); this.ended.set(key, now);
      this.evict(now);
    } else this.drop(rec, "plan_expired");
  }

  forKey(key: string): FlightVersions | null { return this.byKey.get(key) ?? null; }

  /** the most recently changed held flight for a callsign */
  forCallsign(callsign: string): FlightVersions | null {
    const set = this.keysByCallsign.get(callsign);
    if (!set) return null;
    let best: FlightVersions | null = null;
    for (const k of Array.from(set)) {
      const r = this.byKey.get(k);
      if (!r) { set.delete(k); continue; }
      if (!best || r.lastChangeMs > best.lastChangeMs) best = r;
    }
    return best;
  }

  private index(cs: string | null, key: string): void {
    if (!cs) return;
    const set = this.keysByCallsign.get(cs) || new Set<string>();
    set.add(key);
    this.keysByCallsign.set(cs, set);
  }
  private unindex(cs: string | null, key: string): void {
    if (!cs) return;
    const set = this.keysByCallsign.get(cs);
    set?.delete(key);
    if (set && !set.size) this.keysByCallsign.delete(cs);
  }

  private drop(rec: FlightVersions, why: MemoryDropReason): void {
    this.byKey.delete(rec.key);
    this.ended.delete(rec.key);
    this.unindex(rec.callsign, rec.key);
    this.versionCount -= rec.versions.length;
    this.bytes -= rec.bytes;
    this.counters.versionsDropped[why] += rec.versions.length;
    if (!rec.disk.originalWritten) this.counters.originalsNotArchived++;
  }

  evict(now: number): void {
    const keep = this.opts.endedKeepMs ?? FPV_ENDED_KEEP_MS;
    for (const [k, at] of this.ended) {
      if (now - at <= keep) break; // end order: the rest ended later
      const rec = this.byKey.get(k);
      if (rec) this.drop(rec, "ended_grace_over"); else this.ended.delete(k);
    }
    const maxFlights = this.opts.maxFlights ?? FPV_MAX_FLIGHTS;
    const maxVersions = this.opts.maxVersions ?? FPV_MAX_VERSIONS_TOTAL;
    const maxBytes = this.opts.maxBytes ?? FPV_MAX_BYTES;
    while (this.byKey.size > 1) {
      const why: MemoryDropReason | null = this.byKey.size > maxFlights ? "flights_cap"
        : this.versionCount > maxVersions ? "total_cap"
        : this.bytes > maxBytes ? "bytes_cap" : null;
      if (!why) break;
      const oldest = this.byKey.values().next().value;
      if (!oldest) break;
      this.drop(oldest, why);
    }
  }

  status() {
    return {
      flightsHeld: this.byKey.size, versionsHeld: this.versionCount, estimatedBytes: this.bytes,
      caps: {
        perFlight: this.opts.maxPerFlight ?? FPV_MAX_VERSIONS_PER_FLIGHT, flights: this.opts.maxFlights ?? FPV_MAX_FLIGHTS,
        versions: this.opts.maxVersions ?? FPV_MAX_VERSIONS_TOTAL, bytes: this.opts.maxBytes ?? FPV_MAX_BYTES,
      },
      ...this.counters, versionsDropped: { ...this.counters.versionsDropped },
      archive: this.opts.archive ? this.opts.archive.status() : null,
    };
  }
}

/** Heap gauge for the process-wide store (memory audit convention). */
export function registerPlanVersionGauge(store: PlanVersionStore): void {
  registerHeapGauge("flightPlanVersions.store", () => ({
    entries: store.versions, cap: FPV_MAX_VERSIONS_TOTAL, bytes: store.estimatedBytes,
    flights: store.flights, bytesCap: FPV_MAX_BYTES,
  }));
}
