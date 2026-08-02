/**
 * fleetUtilization.ts — corporate-fleet utilization series (BUILD ORDER 3
 * #1, 2026-07-05; the first FUSION product: aircraft position archive x
 * entity spine). GATE 1 (join accuracy) PASSED 2026-07-05 under
 * pre-stated criteria: 20/20 stratified hexes matched independent
 * adsbdb registrations exactly (experiments.md). GATE 2 (utilization x
 * earnings) NOT attempted — nothing here is a SIGNAL.
 *
 * The series: per spine owner (corporations + LLCs), weekly flight
 * counts and airborne hours derived by SESSIONIZING archived airborne
 * points per hex (gap > SESSION_GAP_MIN between consecutive points =
 * new flight). HONESTY built into the numbers:
 *  - the archive samples adaptively (thinned away from sites), so
 *    session hours are LOWER BOUNDS and flight counts can split a long
 *    cruise with a coverage gap into two — the route note says so;
 *  - trustee/leasing registrations (TVPX, banks) hide beneficial
 *    owners — the gate-1 sample surfaced exactly this, so the payload
 *    labels owners as REGISTRANTS, never "the company flying it";
 *  - weeks with no archive coverage are absent, not zero.
 *
 * Same expensive-scan discipline as aircraftEntities: at most one fold
 * per TTL window, stale served during refresh.
 */
import fs from "fs";
import path from "path";
import zlib from "zlib";
import readline from "readline";
import { archiveBaseDir, RAW_RETENTION_DAYS } from "./datacoreArchive";
import { loadEntitySpine, type SpineRecord } from "./aircraftEntities";
import { resolveOperator, isTrusteeRegistrant, operatorGroup } from "./operatorResolution";

export const SESSION_GAP_MIN = 45;
const CORPORATE_TYPES = new Set(["corporation", "llc"]);

export interface HexWeekly {
  /** ISO date of the week's Monday -> {f: flights, h: airborne hours} */
  [weekStart: string]: { f: number; h: number };
}

/** Monday of the week containing epoch-seconds t, as ISO date. */
export function weekStart(tSec: number): string {
  const d = new Date(tSec * 1000);
  const dow = (d.getUTCDay() + 6) % 7; // Mon=0
  d.setUTCDate(d.getUTCDate() - dow);
  return d.toISOString().slice(0, 10);
}

/** Sessionize one hex's sorted airborne timestamps into flights and fold
 *  into weekly buckets. A session's hours = last-first (lower bound); a
 *  session is attributed to the week it STARTED in. */
export function foldSessions(timesSec: number[], out: HexWeekly = {}, gapMin = SESSION_GAP_MIN): HexWeekly {
  if (!timesSec.length) return out;
  const gap = gapMin * 60;
  let start = timesSec[0];
  let prev = timesSec[0];
  const commit = (s: number, e: number) => {
    const wk = weekStart(s);
    const cur = out[wk] || (out[wk] = { f: 0, h: 0 });
    cur.f += 1;
    cur.h = +(cur.h + (e - s) / 3600).toFixed(2);
  };
  for (let i = 1; i < timesSec.length; i++) {
    const t = timesSec[i];
    if (t - prev > gap) {
      commit(start, prev);
      start = t;
    }
    prev = t;
  }
  commit(start, prev);
  return out;
}

// ── archive scan (airborne timestamps per hex) ──────────────────────────────

interface HexAcc { times: number[]; calls: string[] }

function scanFile(fp: string, acc: Map<string, HexAcc>): Promise<void> {
  return new Promise((resolve) => {
    try {
      let stream: NodeJS.ReadableStream = fs.createReadStream(fp);
      if (fp.endsWith(".gz")) stream = stream.pipe(zlib.createGunzip());
      const rl = readline.createInterface({ input: stream, crlfDelay: Infinity });
      rl.on("line", (l) => {
        if (!l.trim()) return;
        let r: any;
        try { r = JSON.parse(l); } catch { return; }
        if (!r || typeof r.i !== "string" || typeof r.t !== "number" || r.g) return; // ground points excluded
        const hex = r.i.toLowerCase();
        let a = acc.get(hex);
        if (!a) acc.set(hex, (a = { times: [], calls: [] }));
        a.times.push(r.t);
        // occurrences, not distinct strings: one callsign seen twice is two
        // observations of the same operator (min-samples rule counts these)
        if (r.c && a.calls.length < 32) a.calls.push(r.c);
      });
      rl.on("close", () => resolve());
      stream.on("error", () => resolve());
      // readline.Interface re-emits a piped-in stream's error on ITSELF too
      // (separate from stream.on("error", ...) above) — unlistened, that
      // crashes the whole process on a truncated/corrupt .gz. See
      // datacoreArchive.ts's streamJsonlLines for the full writeup.
      rl.on("error", () => resolve());
    } catch { resolve(); }
  });
}

export interface OwnerSeries {
  owner: string;
  /** listed parent group for ticker-level studies (Envoy -> American, ...);
   *  equals owner when no mapping applies */
  group: string;
  /** how airframes joined this series: callsign-resolved operator vs FAA
   *  registrant (BUILD ORDER 4 #1 — basis always labeled, never mixed
   *  silently) */
  resolution: { callsign: number; registrant: number };
  /** airframes whose FAA registrant is a trustee/leasing shell */
  trustee_airframes: number;
  registrant_type: string;
  n_airframes: number;
  weekly: HexWeekly;
}

/** Which owner key a hex's accumulated points join to, and by what basis —
 *  callsign-resolved operator when its archived callsigns carry a known
 *  ICAO prefix (works even for non-US hexes the spine can't match),
 *  otherwise the spine corporate/LLC registrant, otherwise null (doesn't
 *  join any series). Shared by buildFleetSeries (live series) and
 *  preserveWeeklyBeforeRollup (permanent weekly archive below) so the two
 *  can never drift on WHICH hex belongs to WHICH owner. */
function resolveOwnerKey(calls: string[], e: SpineRecord | null | undefined):
    { key: string; rtype: string; basis: "callsign" | "registrant" } | null {
  const op = resolveOperator(calls);
  if (op) return { key: op.operator, rtype: "operator", basis: "callsign" };
  if (e && CORPORATE_TYPES.has(e.registrant_type || "")) return { key: e.owner, rtype: e.registrant_type!, basis: "registrant" };
  return null;
}

/** Full scan -> per-OPERATOR weekly series. */
export async function buildFleetSeries(base = archiveBaseDir(), spineFp?: string): Promise<OwnerSeries[]> {
  const spine = loadEntitySpine(spineFp);
  const dir = path.join(base, "aircraft");
  let files: string[] = [];
  try {
    files = fs.readdirSync(dir).filter((f) => f.endsWith(".jsonl") || f.endsWith(".jsonl.gz")).sort();
  } catch { return []; }
  const acc = new Map<string, HexAcc>();
  for (const f of files) await scanFile(path.join(dir, f), acc);

  const byOwner = new Map<string, OwnerSeries>();
  acc.forEach((a, hex) => {
    const e = spine?.[hex];
    const resolved = resolveOwnerKey(a.calls, e);
    if (!resolved) return;
    const { key, rtype, basis } = resolved;
    a.times.sort((x: number, y: number) => x - y);
    let s = byOwner.get(key);
    if (!s) byOwner.set(key, (s = { owner: key, group: operatorGroup(key), resolution: { callsign: 0, registrant: 0 }, trustee_airframes: 0, registrant_type: rtype, n_airframes: 0, weekly: {} }));
    s.n_airframes++;
    s.resolution[basis]++;
    if (isTrusteeRegistrant(e?.owner)) s.trustee_airframes++;
    foldSessions(a.times, s.weekly);
  });

  // Merge in weeks PERMANENTLY PRESERVED before their source raw files aged
  // past datacoreArchive's RAW_RETENTION_DAYS and were deleted (see
  // preserveWeeklyBeforeRollup below) — without this, the scan above can
  // only ever see RAW_RETENTION_DAYS (7 days) of history no matter how long
  // the archive has been running. An owner with ONLY historical weeks (no
  // airframes in the current live window) still appears, honestly labeled.
  const historical = loadWeeklyArchive(base);
  for (const [key, weekly] of Object.entries(historical.owners)) {
    let s = byOwner.get(key);
    if (!s) byOwner.set(key, (s = { owner: key, group: operatorGroup(key), resolution: { callsign: 0, registrant: 0 }, trustee_airframes: 0, registrant_type: "historical-archive-only", n_airframes: 0, weekly: {} }));
    addWeekly(s.weekly, weekly);
  }

  return Array.from(byOwner.values()).sort((a, b) => b.n_airframes - a.n_airframes);
}

// ── permanent weekly archive (survives the 7-day raw-file rollup) ──────────
// buildFleetSeries above can only ever see RAW_RETENTION_DAYS of raw-file
// history — datacoreArchive.rollupOldDaysAsync deletes hour files past that
// window, and its own daily-track summary drops callsigns entirely (can't
// be replayed for operator resolution). Found this session while trying to
// actually run the BUILD ORDER 3d mining design's pre-registered
// "fleet-utilization week-over-week outliers" pass (research/
// open_questions.md, RE-RUN TRIGGER ~2026-08-03 — archive depth alone
// looked ready, but the live series only ever showed ONE week no matter
// how old the archive was). FIX: fold each raw hour file into permanent
// per-owner weekly {f,h} totals BEFORE the generic rollup deletes it,
// merge-add into a small persisted file (weeks are ~52/year x a few
// hundred owners — trivial size, same small-derived-artifact discipline as
// entity_spine.json). Data already rolled up by the PRE-EXISTING rollup
// (everything older than ~7 days as of this fix shipping) is unrecoverably
// lost — this closes the gap going forward only; see the research log for
// the honest re-baselined trigger date.
const WEEKLY_ARCHIVE_NAME = "aircraft_fleet_weekly.json.gz";
// Files this old are certainly already deleted by rollupOldDaysAsync's own
// cutoff (it runs every 6h) — safe to drop from the processed-file
// tracking set below so it doesn't grow forever.
const PROCESSED_PRUNE_DAYS = RAW_RETENTION_DAYS + 3;
const HOUR_FILE_RE = /^(\d{4}-\d{2}-\d{2})-\d{2}\.jsonl(\.gz)?$/;

interface WeeklyArchive {
  /** raw hour filenames already folded in — makes preserveWeeklyBeforeRollup
   *  safe to run twice on the same files (e.g. a crash/restart between this
   *  step and datacoreArchive's rollup delete) without double-counting. */
  processedFiles: string[];
  owners: Record<string, HexWeekly>;
}

function weeklyArchivePath(base: string): string {
  return path.join(base, WEEKLY_ARCHIVE_NAME);
}

function loadWeeklyArchive(base: string): WeeklyArchive {
  try {
    const parsed = JSON.parse(zlib.gunzipSync(fs.readFileSync(weeklyArchivePath(base))).toString());
    return { processedFiles: Array.isArray(parsed.processedFiles) ? parsed.processedFiles : [], owners: parsed.owners || {} };
  } catch {
    return { processedFiles: [], owners: {} };
  }
}

function saveWeeklyArchive(base: string, archive: WeeklyArchive): void {
  fs.writeFileSync(weeklyArchivePath(base), zlib.gzipSync(JSON.stringify(archive)));
}

/** Merge-add `from`'s week buckets into `into` (never overwrite — a week
 *  can legitimately accumulate contributions from multiple rollup ticks
 *  when only PART of its files have aged past retention so far). */
function addWeekly(into: HexWeekly, from: HexWeekly): void {
  for (const [wk, v] of Object.entries(from)) {
    const cur = into[wk] || (into[wk] = { f: 0, h: 0 });
    cur.f += v.f;
    cur.h = +(cur.h + v.h).toFixed(2);
  }
}

/** Fold every raw aircraft hour file older than `retentionDays` that hasn't
 *  already been folded into the permanent weekly archive. Call this BEFORE
 *  datacoreArchive's rollupOldDaysAsync deletes the same files each rollup
 *  tick — the two independently select the identical file set from the
 *  same cutoff, so calling this first is what makes the fold-before-delete
 *  guarantee hold; the processedFiles set makes a second call (before the
 *  delete actually happens) a safe no-op rather than a double-count.
 *  Best-effort background job — never throws, matching the rest of the
 *  archive pipeline's discipline. */
export async function preserveWeeklyBeforeRollup(base = archiveBaseDir(), nowMs = Date.now(), retentionDays = RAW_RETENTION_DAYS, spineFp?: string): Promise<{ filesFolded: number; ownersTouched: number }> {
  const dir = path.join(base, "aircraft");
  let names: string[] = [];
  try { names = fs.readdirSync(dir); } catch { return { filesFolded: 0, ownersTouched: 0 }; }
  const cutoff = nowMs - retentionDays * 86400_000;
  const archive = loadWeeklyArchive(base);
  const processed = new Set(archive.processedFiles);
  const toFold = names.filter((f) => {
    const m = HOUR_FILE_RE.exec(f);
    if (!m || processed.has(f)) return false;
    return Date.parse(m[1] + "T00:00:00Z") < cutoff;
  });
  if (!toFold.length) return { filesFolded: 0, ownersTouched: 0 };

  const spine = loadEntitySpine(spineFp);
  const acc = new Map<string, HexAcc>();
  for (const f of toFold) await scanFile(path.join(dir, f), acc);

  const touched = new Set<string>();
  acc.forEach((a, hex) => {
    const resolved = resolveOwnerKey(a.calls, spine?.[hex]);
    if (!resolved) return;
    a.times.sort((x: number, y: number) => x - y);
    const delta: HexWeekly = {};
    foldSessions(a.times, delta);
    addWeekly(archive.owners[resolved.key] || (archive.owners[resolved.key] = {}), delta);
    touched.add(resolved.key);
  });

  const pruneCutoff = nowMs - PROCESSED_PRUNE_DAYS * 86400_000;
  archive.processedFiles = archive.processedFiles.filter((f) => {
    const m = HOUR_FILE_RE.exec(f);
    return !!m && Date.parse(m[1] + "T00:00:00Z") >= pruneCutoff;
  }).concat(toFold);
  saveWeeklyArchive(base, archive);
  return { filesFolded: toFold.length, ownersTouched: touched.size };
}

// ── cached assembly ─────────────────────────────────────────────────────────
const TTL_MS = 6 * 60 * 60_000;
let cache: { at: number; series: OwnerSeries[] } | null = null;
let inflight: Promise<OwnerSeries[]> | null = null;

export async function fleetSeriesCached(base = archiveBaseDir(), nowMs = Date.now()): Promise<{ series: OwnerSeries[]; as_of: string }> {
  if (!cache || nowMs - cache.at > TTL_MS) {
    if (!inflight) {
      inflight = buildFleetSeries(base).then((series) => {
        cache = { at: Date.now(), series };
        inflight = null;
        return series;
      });
    }
    if (!cache) await inflight; // first call waits; later calls serve stale
  }
  return { series: cache?.series ?? [], as_of: cache ? new Date(cache.at).toISOString() : "" };
}

export function _resetFleetCache() { cache = null; inflight = null; }
