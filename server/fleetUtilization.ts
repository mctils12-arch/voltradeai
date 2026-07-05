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
import { archiveBaseDir } from "./datacoreArchive";
import { loadEntitySpine } from "./aircraftEntities";

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

function scanFile(fp: string, acc: Map<string, number[]>): Promise<void> {
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
        let arr = acc.get(hex);
        if (!arr) acc.set(hex, (arr = []));
        arr.push(r.t);
      });
      rl.on("close", () => resolve());
      stream.on("error", () => resolve());
    } catch { resolve(); }
  });
}

export interface OwnerSeries {
  owner: string;
  registrant_type: string;
  n_airframes: number;
  weekly: HexWeekly;
}

/** Full scan -> per-owner weekly series (corporations + LLCs from the
 *  bundled spine; unmatched hexes simply don't contribute). */
export async function buildFleetSeries(base = archiveBaseDir(), spineFp?: string): Promise<OwnerSeries[]> {
  const spine = loadEntitySpine(spineFp);
  if (!spine) return [];
  const dir = path.join(base, "aircraft");
  let files: string[] = [];
  try {
    files = fs.readdirSync(dir).filter((f) => f.endsWith(".jsonl") || f.endsWith(".jsonl.gz")).sort();
  } catch { return []; }
  const acc = new Map<string, number[]>();
  for (const f of files) await scanFile(path.join(dir, f), acc);

  const byOwner = new Map<string, OwnerSeries>();
  acc.forEach((times, hex) => {
    const e = spine[hex];
    if (!e || !CORPORATE_TYPES.has(e.registrant_type || "")) return;
    times.sort((a: number, b: number) => a - b);
    let s = byOwner.get(e.owner);
    if (!s) byOwner.set(e.owner, (s = { owner: e.owner, registrant_type: e.registrant_type!, n_airframes: 0, weekly: {} }));
    s.n_airframes++;
    foldSessions(times, s.weekly);
  });
  return Array.from(byOwner.values()).sort((a, b) => b.n_airframes - a.n_airframes);
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
