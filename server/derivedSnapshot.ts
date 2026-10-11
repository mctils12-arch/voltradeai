/**
 * derivedSnapshot.ts — durable last-known-good copy of an expensive derived
 * cache, so a deploy/restart serves the previous result instead of
 * `warming_up` while the archive fold re-runs.
 *
 * WHY (2026-10-11 live finding, PRODUCT session): 2h+ after boot,
 * /api/data/portdwell and /api/data/shadowstats still answered
 * `warming_up:true` (and the /api/v1 twins 503). Their refresh is wrapped by
 * crashSafeRefresh.guardedRefresh: a container killed mid-fold (every deploy
 * can do this) leaves an unresolved marker and the next boot defers the fold
 * for up to 6h, with the in-memory cache null the whole time. The crash
 * guard is right to defer the fold; serving nothing meanwhile violates the
 * Freshness Law (CLAUDE.md Law V: render last-known state immediately).
 *
 * Honesty: a loaded snapshot keeps its ORIGINAL `at`, so generated_at /
 * v1Envelope freshness show its true age; nothing is re-stamped as fresh.
 */
import fs from "node:fs";
import path from "node:path";
import { refreshStateDir } from "./crashSafeRefresh";

export interface DerivedSnapshot { at: number; data: unknown }

/** Older than this a snapshot is not worth serving at all (7d = the heaviest fold window). */
export const SNAPSHOT_MAX_AGE_MS = 7 * 24 * 3600_000;

/** DATA_DIR (the /data/voltrade volume convention) first; refreshStateDir() supplies the /data-or-/tmp fallback. */
export function snapshotDir(env: NodeJS.ProcessEnv = process.env): string {
  return env.DATA_DIR || refreshStateDir(env);
}

function snapFile(dir: string, name: string): string {
  return path.join(dir, `voltrade_derived_snapshot_${name}.json`);
}

/** Atomic write (tmp + rename) so a kill mid-write never leaves a torn file. Never throws. */
export function saveSnapshot(name: string, snap: DerivedSnapshot, dir = snapshotDir()): boolean {
  try {
    fs.mkdirSync(dir, { recursive: true });
    const f = snapFile(dir, name);
    const tmp = `${f}.${process.pid}.tmp`;
    fs.writeFileSync(tmp, JSON.stringify(snap));
    fs.renameSync(tmp, f);
    return true;
  } catch { return false; }
}

/** Returns the snapshot, or null if absent/corrupt/malformed/too old. Never throws. */
export function loadSnapshot(name: string, dir = snapshotDir(), nowMs = Date.now()): DerivedSnapshot | null {
  try {
    const raw = JSON.parse(fs.readFileSync(snapFile(dir, name), "utf8"));
    if (typeof raw?.at !== "number" || raw.data == null || typeof raw.data !== "object") return null;
    if (nowMs - raw.at > SNAPSHOT_MAX_AGE_MS || raw.at > nowMs + 60_000) return null;
    return { at: raw.at, data: raw.data };
  } catch { return null; }
}
