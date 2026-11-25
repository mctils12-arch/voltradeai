/**
 * snapshotStreams.ts — build stamps of the git-versioned, bundled datasets
 * (streams whose archive is a whole-file JSON snapshot in datacore/, NOT a
 * volume JSONL dir). The /data Streams inventory scans only the volume, so
 * these streams read "no-data — key-gated, warming up" while their data is
 * in fact being served (2026-10-12 audit). The stamp is the dataset's own
 * embedded build time — honest age, not a deploy mtime.
 *
 * [A7 2026-11-03] Five of the six snapshots are LARGE tables that no longer
 * ship inlined in the bundle (server/lazyAssets.ts). A stamp only needs the
 * dataset's top-level `built`/`built_at`, so it is taken through the
 * table's peek(): the memoized object when a consumer has already loaded
 * the table, otherwise a one-off parse that is garbage once the stamp is
 * kept — the inventory's boot refresh must not pin ~7.5MB of tables in
 * memory for the sake of six strings. Stamps are memoized for the process
 * lifetime (the files are immutable within one deploy; the four the nightly
 * snapshot refresh rewrites on the volume are re-stamped when it does:
 * resetSnapshotStreams()). The 19KB UN
 * Comtrade snapshot stays a static import (small; inlined like before).
 */
import unComtrade from "../datacore/un_comtrade/bilateral_trade.json";
import { LAZY_TABLES, type LazyJsonTable } from "./lazyAssets";

export interface SnapshotStamp {
  artifact: string; // repo-relative path shown as the "newest file"
  built: string | null; // ISO build time embedded in the dataset
  /** "refreshed" when the served copy is the nightly job's volume copy
   *  (server/snapshotRefresh.ts), not the file bundled in the image. */
  origin?: "bundled" | "refreshed";
}

/** Pure: the dataset's embedded build time, or null when absent/unparseable. */
export function stamp(artifact: string, doc: { built?: unknown; built_at?: unknown }): SnapshotStamp {
  const b = doc.built ?? doc.built_at;
  return { artifact, built: typeof b === "string" && !Number.isNaN(Date.parse(b)) ? b : null };
}

/** stream id -> [artifact path, how to see its document], in the inventory's
 *  registration order. Lazy tables go through peek() (not retained); the
 *  one inlined snapshot is the import itself. */
type Snap = readonly [string, () => unknown, (() => "bundled" | "refreshed" | null)?];
const lazy = (t: LazyJsonTable): Snap => [t.rel, () => t.peek(), () => t.origin()];
const SNAPSHOTS: ReadonlyArray<readonly [string, Snap]> = [
  ["eiaweekly", lazy(LAZY_TABLES.eiaWeekly)],
  ["cpcdegreedays", lazy(LAZY_TABLES.cpcDegreeDays)],
  ["jodi", lazy(LAZY_TABLES.jodiPrimaryStocks)],
  ["railep724", lazy(LAZY_TABLES.railEp724)],
  ["uncomtrade", ["datacore/un_comtrade/bilateral_trade.json", () => unComtrade]],
  ["aircraftspine", lazy(LAZY_TABLES.entitySpine)],
];

let memo: Record<string, SnapshotStamp> | null = null;

/** Every snapshot stream's build stamp. A table whose staged file cannot be
 *  read stamps `built: null` (the inventory then shows that stream as
 *  no-data with "no embedded build time") and the failure is logged with
 *  the path — one unreadable file must not take the whole inventory down,
 *  and must never be silent either (packaging defect class, repoFiles.ts). */
export function snapshotStreams(): Record<string, SnapshotStamp> {
  if (memo) return memo;
  const out: Record<string, SnapshotStamp> = {};
  for (const [id, [artifact, doc, origin]] of SNAPSHOTS) {
    try {
      out[id] = stamp(artifact, doc() as { built?: unknown; built_at?: unknown });
      if (origin?.() === "refreshed") out[id].origin = "refreshed";
    } catch (e: unknown) {
      console.error(`[snapshotStreams] ${artifact}: ${e instanceof Error ? e.message : String(e)}`);
      out[id] = { artifact, built: null };
    }
  }
  memo = out;
  return memo;
}

/** Forget the memoized stamps: the nightly snapshot refresh calls this after
 *  it has written a newer copy (the files are immutable within one deploy
 *  EXCEPT for that job's four). */
export function resetSnapshotStreams(): void {
  memo = null;
}

/** Test-only: forget the memoized stamps. */
export function _resetSnapshotStreamsForTests(): void {
  memo = null;
}
