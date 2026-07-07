/**
 * satellites.ts — CelesTrak GP (general perturbations) element sets,
 * archived per group (ANALYST CONSOLE W2, research/console_charter.md —
 * orbit history is another accumulating asset; the client globe layer
 * propagates these with satellite.js in a later PR).
 *
 * Source: CelesTrak GP query API, OMM JSON format
 * (https://celestrak.org/NORAD/elements/gp.php?GROUP=<g>&FORMAT=json).
 * Contract live-verified 2026-07-07: array of OMM objects with
 * OBJECT_NAME, OBJECT_ID, EPOCH, MEAN_MOTION, ECCENTRICITY, INCLINATION,
 * RA_OF_ASC_NODE, ARG_OF_PERICENTER, MEAN_ANOMALY, EPHEMERIS_TYPE,
 * CLASSIFICATION_TYPE, NORAD_CAT_ID, ELEMENT_SET_NO, REV_AT_EPOCH,
 * BSTAR, MEAN_MOTION_DOT, MEAN_MOTION_DDOT. JSON (not TLE) is the format
 * CelesTrak explicitly directs users to — TLE breaks at catalog number
 * 69999, predicted mid-July 2026.
 *
 * Licensing (usage-policy.php, verified 2026-07-07): data is freely
 * available to all users (non-profit; no redistribution restriction
 * stated; donations encouraged). Courtesy limits are the binding terms:
 * GP data updates once every 2 hours — never fetch a group more often;
 * on ANY non-200 response, STOP querying (403/404 will not change on
 * retry and repeat offenders are firewalled). Our cadence: each group
 * every 6 HOURS (3x the update interval), spaced within a sweep, one
 * eager sweep on boot. Attribution carried on every surface.
 *
 * Keyless. Dedup key NORAD_CAT_ID|EPOCH: an element set only changes
 * when its EPOCH advances, so re-fetching the same epoch never
 * duplicates, while every epoch advance appends — the archive is the
 * orbit HISTORY, not just the current state.
 */

import fs from "fs";
import path from "path";
import zlib from "zlib";
import { archiveBaseDir } from "./datacoreArchive";

/** Verified against https://celestrak.org/NORAD/elements/ 2026-07-07.
 *  NOTE: the console charter's "active-geosynchronous" is not a CelesTrak
 *  group — the documented group for "Active Geosynchronous" is `geo`. */
export const GROUPS = ["stations", "starlink", "gps-ops", "geo"] as const;
export type GpGroup = (typeof GROUPS)[number];

export const GROUP_LABELS: Record<GpGroup, string> = {
  stations: "Space stations",
  starlink: "Starlink",
  "gps-ops": "GPS operational",
  geo: "Active geosynchronous",
};

export const POLL_INTERVAL_MS = 6 * 60 * 60_000; // 6h — 3x CelesTrak's 2h update cycle
const GROUP_SPACING_MS = 10_000;                 // stagger groups within a sweep

export const ATTRIBUTION =
  "Orbital data courtesy of CelesTrak (celestrak.org), Dr. T.S. Kelso";

/** One OMM record as CelesTrak publishes it, plus our fetch timestamp. */
export interface GpRecord {
  NORAD_CAT_ID: number;
  EPOCH: string;        // UTC ISO — lexicographic order = time order
  OBJECT_NAME?: string;
  rt?: string;          // as-fetched UTC ISO timestamp (ours)
  [k: string]: unknown;
}

export function gpUrl(group: string): string {
  return `https://celestrak.org/NORAD/elements/gp.php?GROUP=${encodeURIComponent(group)}&FORMAT=json`;
}

/** Parsed-JSON payload -> valid GpRecords. Malformed entries are dropped,
 *  never guessed; a non-array payload yields []. */
export function parseGp(payload: unknown): GpRecord[] {
  if (!Array.isArray(payload)) return [];
  const out: GpRecord[] = [];
  for (const r of payload) {
    if (!r || typeof r !== "object") continue;
    const cat = (r as any).NORAD_CAT_ID;
    const epoch = (r as any).EPOCH;
    if (!Number.isFinite(cat) || typeof epoch !== "string" || !epoch) continue;
    out.push(r as GpRecord);
  }
  return out;
}

// ── Fetch ───────────────────────────────────────────────────────────────────

type FetchFn = (url: string, init?: any) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** Per-group outcome of the LAST sweep (euLoad precedent: a silently
 *  missing group must be self-diagnosing from the route). */
const lastIssues: Record<string, string> = {};

export function satIssues(): Record<string, string> {
  return { ...lastIssues };
}

export async function fetchGroup(group: GpGroup, fetchImpl: FetchFn = fetch as any): Promise<GpRecord[] | null> {
  try {
    const r = await fetchImpl(gpUrl(group), {
      headers: { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)" },
      signal: AbortSignal.timeout(60000) as any,
    });
    if (!r.ok) {
      // CelesTrak policy: a non-200 will not change on retry — record and
      // stop; the next attempt is the next 6h sweep, never a tight loop.
      lastIssues[group] = `http ${r.status} — per CelesTrak usage policy, not retried until next sweep`;
      console.error(`[datacore] satellites ${group} -> ${lastIssues[group]}`);
      return null;
    }
    let parsed: unknown;
    try { parsed = JSON.parse(await r.text()); } catch {
      lastIssues[group] = "unparseable JSON body";
      console.error(`[datacore] satellites ${group}: unparseable JSON body`);
      return null;
    }
    const recs = parseGp(parsed);
    if (!recs.length) {
      lastIssues[group] = "empty document (no valid GP records)";
      return null;
    }
    delete lastIssues[group];
    return recs;
  } catch (e: any) {
    lastIssues[group] = String(e?.message || e).slice(0, 200);
    console.error(`[datacore] satellites ${group}:`, lastIssues[group]);
    return null;
  }
}

// ── Archive (append-only JSONL day-file per UTC fetch day per group) ────────

const seenByGroup = new Map<string, Set<string>>();
const seededGroups = new Set<string>();

const gpKey = (r: GpRecord) => `${r.NORAD_CAT_ID}|${r.EPOCH}`;

function groupDir(group: string, baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "satellites", group);
}

/** GP epochs are days old at most — 30d bounds the seed heap (euLoad
 *  120d precedent scaled to this stream: starlink alone is ~10k keys/day). */
export const SEED_WINDOW_DAYS = 30;

export function seedFileInWindow(fileName: string, nowMs: number): boolean {
  const day = Date.parse(fileName.slice(0, 10));
  return Number.isFinite(day) && nowMs - day <= SEED_WINDOW_DAYS * 86400_000;
}

function seedSeen(group: string, dir: string, nowMs: number): Set<string> {
  const seen = new Set<string>();
  try {
    for (const f of fs.readdirSync(dir)) {
      if (!/^\d{4}-\d{2}-\d{2}\.jsonl(\.gz)?$/.test(f)) continue;
      if (!seedFileInWindow(f, nowMs)) continue;
      const fp = path.join(dir, f);
      let text: string;
      try {
        text = f.endsWith(".gz")
          ? zlib.gunzipSync(fs.readFileSync(fp)).toString("utf8")
          : fs.readFileSync(fp, "utf8");
      } catch { continue; }
      for (const line of text.split("\n")) {
        if (!line) continue;
        try { seen.add(gpKey(JSON.parse(line))); } catch {}
      }
    }
  } catch {}
  return seen;
}

/** Appends UNSEEN element sets (NORAD_CAT_ID|EPOCH) to the group's
 *  day-file for the UTC FETCH day, stamping each row with rt. Returns
 *  rows appended. Same-epoch re-fetches append nothing. */
export function archiveGp(group: GpGroup, records: GpRecord[], baseDir?: string, nowMs?: number): number {
  if (!records.length) return 0;
  const now = nowMs ?? Date.now();
  const dir = groupDir(group, baseDir);
  if (!seededGroups.has(group)) {
    seenByGroup.set(group, seedSeen(group, dir, now));
    seededGroups.add(group);
  }
  const seen = seenByGroup.get(group)!;
  const rt = new Date(now).toISOString();
  const fresh = records.filter((r) => !seen.has(gpKey(r)));
  if (!fresh.length) return 0;
  try {
    fs.mkdirSync(dir, { recursive: true });
    const day = rt.slice(0, 10);
    fs.appendFileSync(path.join(dir, `${day}.jsonl`),
                      fresh.map((r) => JSON.stringify({ ...r, rt })).join("\n") + "\n");
    fresh.forEach((r) => seen.add(gpKey(r)));
    return fresh.length;
  } catch (e: any) {
    console.error("[datacore] satellites archive:", e?.message || e);
    return 0;
  }
}

/** gz day-files older than 3 days, merge-not-overwrite (a late append can
 *  land a plain file after the day was gz'd — euLoad precedent). */
export function gzipOldSatDays(baseDir?: string, nowMs?: number): number {
  const now = nowMs ?? Date.now();
  let n = 0;
  for (const group of GROUPS) {
    const dir = groupDir(group, baseDir);
    try {
      for (const f of fs.readdirSync(dir)) {
        if (!f.endsWith(".jsonl")) continue;
        if (now - Date.parse(f.slice(0, 10)) < 3 * 86400_000) continue;
        const fp = path.join(dir, f);
        const gzPath = `${fp}.gz`;
        const prior = fs.existsSync(gzPath) ? zlib.gunzipSync(fs.readFileSync(gzPath)) : Buffer.alloc(0);
        fs.writeFileSync(gzPath, zlib.gzipSync(Buffer.concat([prior, fs.readFileSync(fp)])));
        fs.unlinkSync(fp);
        n++;
      }
    } catch {}
  }
  return n;
}

// ── Cache + poll (route is cache-only; never fetches on request) ────────────

export interface GroupCache {
  at: number;            // fetch wall time (ms)
  newest_epoch: string;  // newest EPOCH across the group
  records: GpRecord[];   // latest element set per NORAD_CAT_ID
}

const cache = new Map<GpGroup, GroupCache>();

export function latestGroup(group: GpGroup): GroupCache | null {
  return cache.get(group) ?? null;
}

/** Latest element set per NORAD_CAT_ID (highest EPOCH wins; feed is
 *  normally one-per-satellite, but never trust that). */
export function latestPerSatellite(records: GpRecord[]): GpRecord[] {
  const best = new Map<number, GpRecord>();
  for (const r of records) {
    const prior = best.get(r.NORAD_CAT_ID);
    if (!prior || r.EPOCH > prior.EPOCH) best.set(r.NORAD_CAT_ID, r);
  }
  return Array.from(best.values());
}

export async function refreshSatellites(fetchImpl: FetchFn = fetch as any,
                                        nowMs?: number, baseDir?: string,
                                        spacingMs = GROUP_SPACING_MS): Promise<void> {
  for (const group of GROUPS) {
    const recs = await fetchGroup(group, fetchImpl);
    if (recs) {
      archiveGp(group, recs, baseDir, nowMs);
      const latest = latestPerSatellite(recs);
      let newest = "";
      for (const r of latest) if (r.EPOCH > newest) newest = r.EPOCH;
      cache.set(group, { at: nowMs ?? Date.now(), newest_epoch: newest, records: latest });
    }
    if (spacingMs > 0) await sleep(spacingMs);
  }
  try { gzipOldSatDays(baseDir, nowMs); } catch {}
}

let polling = false;

/** 6h cadence per group (CelesTrak updates GP every 2h; usage policy asks
 *  for at most once per update — we poll at 3x that interval). Eager boot. */
export function bootSatellitesPoll(intervalMs = POLL_INTERVAL_MS): void {
  if (polling) return;
  polling = true;
  refreshSatellites();
  setInterval(() => { refreshSatellites(); }, intervalMs).unref?.();
}

// ── Route response (logic here so rejection/warming/envelope are testable;
//    routes.ts stays a thin cache-only shell) ────────────────────────────────

export function satellitesResponse(groupParam: unknown): { status: number; body: any } {
  const g = typeof groupParam === "string" && groupParam ? groupParam : "all";
  const base = {
    kind: "raw" as const,
    source: "CelesTrak GP element sets (OMM JSON)",
    attribution: ATTRIBUTION,
    groups: [...GROUPS],
  };
  if (g === "all") {
    const summary = GROUPS.map((group) => {
      const hit = cache.get(group);
      return {
        group,
        label: GROUP_LABELS[group],
        count: hit ? hit.records.length : 0,
        newest_epoch: hit ? hit.newest_epoch : null,
        fetched_at: hit ? hit.at : null,
        warming_up: !hit,
      };
    });
    return { status: 200, body: { ...base, group: "all", summary, issues: satIssues() } };
  }
  if (!(GROUPS as readonly string[]).includes(g)) {
    return { status: 400, body: { error: `unknown group '${g}'`, groups: [...GROUPS] } };
  }
  const hit = cache.get(g as GpGroup);
  if (!hit) {
    return { status: 200, body: { ...base, group: g, warming_up: true, count: 0, satellites: [],
                                  issue: lastIssues[g] || null } };
  }
  return {
    status: 200,
    body: {
      ...base,
      group: g,
      count: hit.records.length,
      freshness: { newest_epoch: hit.newest_epoch, fetched_at: hit.at },
      note: "latest GP element set per NORAD_CAT_ID from the last 6h poll; " +
            "EPOCH is the element-set epoch (propagate with SGP4, e.g. satellite.js) — " +
            "positions are NOT included, only mean elements; full epoch history accumulates in the archive",
      satellites: hit.records,
    },
  };
}

/** TEST-ONLY: clears module state so restart seeding is testable. */
export function resetSatellitesForTests(): void {
  seenByGroup.clear();
  seededGroups.clear();
  cache.clear();
  for (const k of Object.keys(lastIssues)) delete lastIssues[k];
}
