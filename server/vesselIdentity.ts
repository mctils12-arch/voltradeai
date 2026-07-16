/**
 * vesselIdentity.ts — IMO-based vessel identity resolution over our own AIS
 * archive (GIP BUILD QUEUE item 3, research/open_questions.md ~line 4105;
 * EDGE DOCTRINE #1/#3: the raw material — aisstream.io's ShipStaticData
 * messages — is already streaming into server/routes.ts's vessel-stream
 * handler every second. Only IMO number, call sign, and dimensions were
 * ever parsed OUT of those messages before this module: shiptype,
 * destination, and name. This is pure processing over data already paid
 * for and already in hand, zero new external source, zero new key).
 *
 * An IMO number is assigned to a HULL for its operating life (unlike MMSI,
 * which a vessel can legally re-register with a new flag state, and which
 * a bad actor can spoof or reassign). Global Fishing Watch / Park et al.
 * 2023's published method: a checksum-VALID IMO number observed under two
 * or more distinct MMSIs over time is strong, deterministic evidence of a
 * hull-identity change (reflag, MMSI reassignment, or spoofing) — replacing
 * shadowFleet.ts's existing `identity_candidates` heuristic (name
 * reappearing under a new MMSI, or a new MMSI's first point landing near
 * where another went dark — both guesses) with a published, checksum-
 * verifiable signal.
 *
 * RAW STATISTIC per the Map v2.2 RAW/SIGNAL boundary: this reports an
 * OBSERVED FACT in our own archive (an IMO co-occurring with >1 MMSI), not
 * an interpreted "this is shadow fleet" claim — same posture as
 * gap_events/loiter_events. Ladder gate 1 (validating this against
 * documented reflag cases) is still open; see open_questions.md.
 *
 * Storage: one append-only JSONL registry — NOT hour-bucketed like the
 * position archive, because this is a slowly-changing identity table, not
 * a position firehose. A NEW line is written only the first time a given
 * (mmsi, imo) PAIR is observed (in-process dedupe via a lazily-loaded
 * Set), so millions of repeated static messages collapse to at most a
 * handful of new lines per newly-registered or newly-reflagged hull —
 * the registry stays trivially small indefinitely.
 *
 * Pure module: fs reads/writes only, baseDir-injectable, no trading
 * imports (datacore boundary rule).
 */
import fs from "fs";
import path from "path";
import { archiveBaseDir } from "./datacoreArchive";

export interface VesselStaticMessage {
  mmsi: string;
  imo?: number | string | null;
  callsign?: string | null;
  name?: string | null;
}

export interface IdentityObservation {
  t: number;             // unix seconds, first-observed time of this (mmsi, imo) pair
  mmsi: string;
  imo: number;            // checksum-valid IMO only — invalid/absent IMOs are never recorded
  callsign?: string;
  name?: string;
}

export interface ImoReuseCandidate {
  imo: number;
  mmsis: string[];        // >=2 distinct MMSIs observed under this IMO, oldest-first
  firstSeen: number;
  lastSeen: number;
  names: string[];        // distinct non-empty names seen under this IMO
}

/** IMO number check-digit validation (IMO Res. A.600(15) / ISO 6346-style
 *  scheme): a 7-digit number whose 7th digit equals
 *  sum(digit_i * (7 - i) for i in 1..6) mod 10. "0000000" and any
 *  non-7-digit value are never valid — AIS broadcasts 0 for vessels with
 *  no assigned IMO (most small/fishing craft), which must not be treated
 *  as a real identity. */
export function isValidImo(imo: number | string | null | undefined): imo is number {
  if (imo == null) return false;
  const s = String(imo).trim();
  if (!/^\d{7}$/.test(s) || s === "0000000") return false;
  const digits = s.split("").map(Number);
  const sum = digits.slice(0, 6).reduce((acc, d, i) => acc + d * (7 - i), 0);
  return sum % 10 === digits[6];
}

function registryPath(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "vessel_identity", "registry.jsonl");
}

// In-memory dedupe of "mmsi:imo" pairs already known — loaded lazily from
// disk on first use per baseDir so a process restart doesn't re-append
// history it already wrote, without keeping the file re-read on every
// message (the common case: a Set.has lookup).
let seenPairs: Set<string> | null = null;
let seenPairsBase: string | null = null;

function loadSeenPairs(baseDir?: string): Set<string> {
  const base = baseDir || archiveBaseDir();
  if (seenPairs && seenPairsBase === base) return seenPairs;
  const set = new Set<string>();
  try {
    const text = fs.readFileSync(registryPath(baseDir), "utf8");
    for (const line of text.split("\n")) {
      if (!line) continue;
      try {
        const r = JSON.parse(line);
        if (r.mmsi && r.imo) set.add(`${r.mmsi}:${r.imo}`);
      } catch { /* one malformed line never invalidates the rest */ }
    }
  } catch { /* no registry yet — first observation on this baseDir */ }
  seenPairs = set;
  seenPairsBase = base;
  return set;
}

/** Test-only: drop the in-memory dedupe cache so a fresh baseDir/registry
 *  state is picked up (hermetic tests use a new tmpdir per test, but the
 *  module-level cache would otherwise carry stale state across a `baseDir`
 *  reused between test files running in the same process). */
export function clearIdentityCache(): void { seenPairs = null; seenPairsBase = null; }

/** Records a NEW (mmsi, imo) pairing if, and only if, the IMO passes
 *  checksum validation and this exact pairing has never been recorded
 *  before. Returns true iff a new line was appended. */
export function recordIdentityObservation(msg: VesselStaticMessage, baseDir?: string, nowMs?: number): boolean {
  if (!msg.mmsi || !isValidImo(msg.imo)) return false;
  const imo = Number(msg.imo);
  const key = `${msg.mmsi}:${imo}`;
  const set = loadSeenPairs(baseDir);
  if (set.has(key)) return false;
  const fp = registryPath(baseDir);
  const rec: IdentityObservation = {
    t: Math.floor((nowMs ?? Date.now()) / 1000),
    mmsi: msg.mmsi, imo,
    callsign: msg.callsign?.trim() || undefined,
    name: msg.name?.trim() || undefined,
  };
  try {
    fs.mkdirSync(path.dirname(fp), { recursive: true });
    fs.appendFileSync(fp, JSON.stringify(rec) + "\n");
  } catch (e: any) {
    console.error("[vesselIdentity] registry append:", e?.message || e);
    return false;
  }
  set.add(key);
  return true;
}

/** Reads the full registry (small by construction — dedupe keeps it
 *  bounded by distinct-hulls-ever-observed, not message volume). */
export function readIdentityRegistry(baseDir?: string): IdentityObservation[] {
  const out: IdentityObservation[] = [];
  let text: string;
  try { text = fs.readFileSync(registryPath(baseDir), "utf8"); } catch { return out; }
  for (const line of text.split("\n")) {
    if (!line) continue;
    try { out.push(JSON.parse(line)); } catch { /* one bad line doesn't drop the rest */ }
  }
  return out;
}

/** GIP BUILD QUEUE item 3's core detector: checksum-valid IMO numbers
 *  observed under >=2 DISTINCT MMSIs in our own archive — the published-
 *  method (Park et al. 2023) hull-identity-change signal. Sorted by most
 *  recently observed first. */
export function detectImoReuse(records: IdentityObservation[]): ImoReuseCandidate[] {
  const byImo = new Map<number, IdentityObservation[]>();
  for (const r of records) {
    let arr = byImo.get(r.imo);
    if (!arr) { arr = []; byImo.set(r.imo, arr); }
    arr.push(r);
  }
  const out: ImoReuseCandidate[] = [];
  for (const [imo, recsUnsorted] of byImo) {
    const recs = [...recsUnsorted].sort((a, b) => a.t - b.t);
    const mmsis: string[] = [];
    const mmsiSeen = new Set<string>();
    for (const r of recs) { if (!mmsiSeen.has(r.mmsi)) { mmsiSeen.add(r.mmsi); mmsis.push(r.mmsi); } }
    if (mmsis.length < 2) continue;
    out.push({
      imo,
      mmsis,
      firstSeen: recs[0].t,
      lastSeen: recs[recs.length - 1].t,
      names: Array.from(new Set(recs.map((r) => r.name).filter((n): n is string => !!n))),
    });
  }
  return out.sort((a, b) => b.lastSeen - a.lastSeen);
}

/** Convenience: reads the registry fresh from disk and returns the
 *  checksum-verified reuse count + top examples — the shape shadowFleet.ts
 *  merges into ShadowStats. Kept as a single function so both the sync and
 *  async ShadowStats builders call the identical code path (byte-identical
 *  ratchet requirement). */
export function imoIdentityStats(baseDir?: string, maxExamples = 5): {
  imo_confirmed_identity_changes: number;
  imo_confirmed_examples: ImoReuseCandidate[];
} {
  const candidates = detectImoReuse(readIdentityRegistry(baseDir));
  return {
    imo_confirmed_identity_changes: candidates.length,
    imo_confirmed_examples: candidates.slice(0, maxExamples),
  };
}
