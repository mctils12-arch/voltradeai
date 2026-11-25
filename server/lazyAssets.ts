/**
 * lazyAssets.ts — on-demand loading of the LARGE committed datacore JSON
 * tables and of the image library (sharp/libvips), behind one flag.
 *
 * WHY (A7, memory audit 2026-11-03): dist/index.cjs was 18.4MB, ~18MB of
 * which was esbuild-inlined JSON — 15 tables of 0.2–4MB each (aircraft
 * entity spine, military installations, UCMR5 PFAS, quake history, CDC
 * county rates + TopoJSON, JODI stocks, US power plants, CPC degree days,
 * EP724 rail, Natural Earth admin0/admin1, nuclear tests, EIA weekly,
 * signal ladder). Every one was parsed at boot and held at module scope
 * for the life of the process, and V8 keeps the bundle's source text
 * resident too — so each table byte cost roughly twice over whether or not
 * any request ever asked for it (most only back a dashboard route).
 * Separately, `import sharp from "sharp"` in aeroOverview.ts loaded libvips
 * (~30–60MB RSS) at boot although only the FAA-chart tile path uses it.
 * Live: Node RSS 2.55–2.84GB vs heap ~1.9–2.1GB.
 *
 * WHAT: each table is a memoized loader that reads its staged file from
 * disk and JSON.parses it on FIRST USE; sharp is a memoized dynamic
 * import. The parsed value is deep-equal to what the static import gave
 * (lazyAssets.test.ts pins that for every table against the real files),
 * so a consumer only swaps `tableJson` for `LAZY_TABLES.<name>.get()`.
 *
 * WHERE THE FILES COME FROM AT RUNTIME: the frozen Dockerfile ships dist/
 * only, never the repo datacore/ tree ([REPAIR 2026-07-07], repoFiles.ts),
 * so script/build.ts stages every table in LAZY_TABLES into dist/datacore/
 * and lazyAssets.test.ts ratchets that the staging list covers the
 * registry. Resolution order: repoDataPath() (working tree, then dist/
 * under cwd), then relative to THIS module's own directory (the bundle's
 * dist/, or server/../ in the source tree) so a process whose cwd is not
 * the app root still finds its own staged copy — aircraftEntities.test.ts's
 * chdir ratchet (spine must serve with no datacore/ under cwd) pins that.
 * A table that is genuinely missing THROWS naming the path: never a
 * silently empty dataset (the 2026-07-05 / 2026-07-20 defect class).
 *
 * FLAG: VOLTRADE_LAZY_ASSETS — unset or "1" (default) = lazy; "0" = eager.
 * server/index.ts calls preloadLazyAssets() before routes register; in
 * eager mode that runs EVERY loader here (tables + sharp) at boot — the
 * pre-A7 memory profile, as a rollback that needs no code change. One
 * code path either way: eager mode only calls the same loaders early.
 *
 * NOT HERE: idle-release of a table nobody has asked for in a while (the
 * A3 idle-TTL work's lazyDataset()) — deliberately not depended on. A
 * table once loaded stays memoized for the process lifetime, exactly as
 * the static import did; the two compose as a follow-up.
 */
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import type SharpLib from "sharp";
import { repoDataPath } from "./repoFiles";

export const LAZY_ASSETS_ENV = "VOLTRADE_LAZY_ASSETS";

/** "0" = eager (preload every asset at boot). Anything else = lazy. */
export function lazyAssetsEnabled(env: NodeJS.ProcessEnv = process.env): boolean {
  return env[LAZY_ASSETS_ENV] !== "0";
}

/** The directory this module runs from: dist/ in the CJS bundle
 *  (__dirname), server/ under tsx/ESM (import.meta.url); null if neither
 *  is available. The bundle never reaches the import.meta branch — esbuild
 *  leaves it empty in cjs output and script/build.ts silences that notice. */
function moduleDir(): string | null {
  if (typeof __dirname === "string") return __dirname;
  try {
    return path.dirname(fileURLToPath(import.meta.url));
  } catch {
    return null;
  }
}

/** Where a staged table lives right now (see the module doc for the order).
 *  When nothing exists anywhere, the repoDataPath() answer is returned so
 *  get() can name the path it failed on. Exported for tests. */
export function resolveLazyAssetPath(rel: string, cwd: string = process.cwd()): string {
  const viaRepo = repoDataPath(rel, cwd);
  if (fs.existsSync(viaRepo)) return viaRepo;
  const here = moduleDir();
  if (here) {
    for (const cand of [path.resolve(here, rel), path.resolve(here, "..", rel)]) {
      if (fs.existsSync(cand)) return cand;
    }
  }
  return viaRepo;
}

// ── refreshed copies on the volume ───────────────────────────────────────────
// [2026-11-25, open_questions (ah4)] Four whole-file snapshots (EIA weekly,
// CPC degree days, STB EP724 rail, JODI stocks) are refreshed nightly by
// snapshot_refresh_job.py (server/snapshotRefresh.ts) into
// <DATA_DIR>/snapshot_refresh/<rel>. A table created with `refreshable`
// serves the NEWER of that copy and the bundled one, by the build time the
// document itself carries. The bundled copy is always read first and a
// missing bundled file still throws: the volume copy can replace the data,
// never hide a packaging defect. A volume copy that is unreadable, not JSON
// or carries no build time is logged and passed over.
// ROLLBACK: VOLTRADE_SNAPSHOT_REFRESH=0 — the volume copies are ignored and
// every table serves its bundled file, as before.
export const SNAPSHOT_REFRESH_ENV = "VOLTRADE_SNAPSHOT_REFRESH";
export const SNAPSHOT_REFRESH_SUBDIR = "snapshot_refresh";

/** Unset or "1"/"true"/"on"/"yes" = on (default); anything else = off. */
export function snapshotRefreshEnabled(env: Record<string, string | undefined> = process.env): boolean {
  const raw = env[SNAPSHOT_REFRESH_ENV];
  const v = (raw === undefined ? "1" : raw).trim().toLowerCase();
  return v === "1" || v === "true" || v === "on" || v === "yes";
}

export function snapshotRefreshRoot(env: Record<string, string | undefined> = process.env): string {
  return path.join(env.DATA_DIR || (fs.existsSync("/data") ? "/data/voltrade" : "/tmp"), SNAPSHOT_REFRESH_SUBDIR);
}

/** The document's embedded build time in ms (`built`, or JODI's `built_at`); NaN when absent. */
export function snapshotBuiltMs(doc: unknown): number {
  if (!doc || typeof doc !== "object") return NaN;
  const d = doc as { built?: unknown; built_at?: unknown };
  const b = d.built ?? d.built_at;
  return typeof b === "string" ? Date.parse(b) : NaN;
}

export type LazyTableOrigin = "bundled" | "refreshed";

/** Pure: which copy is served. The refreshed one only when both carry a build
 *  time and its own is strictly later. */
export function pickSnapshotOrigin(bundledBuiltMs: number, refreshedBuiltMs: number): LazyTableOrigin {
  return Number.isFinite(refreshedBuiltMs) && Number.isFinite(bundledBuiltMs) && refreshedBuiltMs > bundledBuiltMs ? "refreshed" : "bundled";
}

export interface LazyJsonTable {
  /** repo-relative path, e.g. "datacore/quake_history.json" — also what build.ts stages */
  readonly rel: string;
  /** The parsed table: read + parsed on the first call, memoized for the
   *  process lifetime. Throws (naming the path) when the file is missing or
   *  is not JSON — never a silently empty dataset. */
  get(): unknown;
  /** The parsed table WITHOUT retaining it: the memoized object when one
   *  exists, otherwise a one-off parse that is garbage once the caller drops
   *  it. For header-only readers (snapshotStreams' build stamps). */
  peek(): unknown;
  /** true once get() has memoized the table */
  loaded(): boolean;
  /** size in bytes of the file as last read (0 before any read) */
  bytes(): number;
  /** drop the memoized table (tests, and the nightly snapshot refresh after
   *  it has written a newer copy) */
  reset(): void;
  /** which copy the last read served; null before any read */
  origin(): LazyTableOrigin | null;
}

/** The volume copy of a refreshable table, or null (absent, switched off, or
 *  not usable: said on stderr, never thrown). */
function readRefreshed(rel: string, env: Record<string, string | undefined>): { value: unknown; bytes: number } | null {
  if (!snapshotRefreshEnabled(env)) return null;
  const fp = path.join(snapshotRefreshRoot(env), rel);
  let buf: Buffer;
  try {
    buf = fs.readFileSync(fp);
  } catch (e: unknown) {
    if ((e as NodeJS.ErrnoException)?.code !== "ENOENT") console.error(`[lazyAssets] refreshed copy ${fp} unreadable, serving the bundled file: ${e instanceof Error ? e.message : String(e)}`);
    return null;
  }
  try {
    return { value: JSON.parse(buf.toString("utf8")), bytes: buf.length };
  } catch (e: unknown) {
    console.error(`[lazyAssets] refreshed copy ${fp} is not valid JSON, serving the bundled file: ${e instanceof Error ? e.message : String(e)}`);
    return null;
  }
}

function readTable(rel: string, refreshable: boolean, env: Record<string, string | undefined>): { value: unknown; bytes: number; origin: LazyTableOrigin } {
  const bundled = readBundled(rel);
  if (!refreshable) return { ...bundled, origin: "bundled" };
  const refreshed = readRefreshed(rel, env);
  if (refreshed && pickSnapshotOrigin(snapshotBuiltMs(bundled.value), snapshotBuiltMs(refreshed.value)) === "refreshed") {
    return { ...refreshed, origin: "refreshed" };
  }
  return { ...bundled, origin: "bundled" };
}

export interface LazyJsonTableOptions {
  /** Serve the newer of the bundled file and the nightly job's volume copy. */
  refreshable?: boolean;
  /** For tests: the environment the volume root and the flag are read from. */
  env?: Record<string, string | undefined>;
}

function readBundled(rel: string): { value: unknown; bytes: number } {
  const fp = resolveLazyAssetPath(rel);
  let buf: Buffer;
  try {
    buf = fs.readFileSync(fp);
  } catch (e: unknown) {
    throw new Error(
      `lazyAssets: ${rel} is not readable at ${fp} — it is registered in server/lazyAssets.ts but not ` +
      `staged into dist/datacore/ by script/build.ts (the frozen Dockerfile ships dist/ only): ` +
      (e instanceof Error ? e.message : String(e)),
    );
  }
  try {
    return { value: JSON.parse(buf.toString("utf8")), bytes: buf.length };
  } catch (e: unknown) {
    throw new Error(`lazyAssets: ${rel} at ${fp} is not valid JSON: ${e instanceof Error ? e.message : String(e)}`);
  }
}

export function lazyJsonTable(rel: string, opts: LazyJsonTableOptions = {}): LazyJsonTable {
  const refreshable = opts.refreshable === true;
  const env = () => opts.env ?? process.env;
  let memo: { value: unknown } | null = null;
  let size = 0;
  let origin: LazyTableOrigin | null = null;
  return {
    rel,
    get() {
      if (!memo) {
        const r = readTable(rel, refreshable, env());
        memo = { value: r.value };
        size = r.bytes;
        origin = r.origin;
      }
      return memo.value;
    },
    peek() {
      if (memo) return memo.value;
      const r = readTable(rel, refreshable, env());
      size = r.bytes;
      origin = r.origin;
      return r.value;
    },
    loaded: () => memo !== null,
    bytes: () => size,
    reset() {
      memo = null;
    },
    origin: () => origin,
  };
}

/** Every table the bundle no longer inlines. To add one: (1) swap the
 *  consumer's static import for `LAZY_TABLES.<name>.get()`, (2) add its
 *  cp() line to script/build.ts — lazyAssets.test.ts fails until you do.
 *  SMALL tables (strategic_sites, shadow_zones, entity_map, nuclear
 *  accidents/facilities, radnet, wiki/nhtsa/fleet lists; all <20KB) and the
 *  boot-critical shared registry (layers.json) stay static imports on
 *  purpose: the win here is bytes, and those cost nothing. ofac_sdn_vessels
 *  (194KB) stays inlined too — the shadow-fleet job needs it minutes after
 *  boot, so there is nothing to defer. */
export const LAZY_TABLES = {
  // us_power_plants: boot-adjacent (entity-graph poll, NRC poller) AND
  // /api/data/powerplants + two cross-tie routes — resident soon after boot
  // either way; the saving is the bundle text and the inlined-literal cost.
  usPowerPlants: lazyJsonTable("datacore/powerplants/us_power_plants.json"),
  // entity_spine (3.9MB, the largest): per-hex route + fleet rollup/series.
  entitySpine: lazyJsonTable("datacore/aircraft/entity_spine.json"),
  // CPC degree days: grid-stress poller (when enabled) + inventory stamp.
  cpcDegreeDays: lazyJsonTable("datacore/cpc/degree_days.json", { refreshable: true }),
  // Dashboard-request-only from here down: nothing touches these at boot.
  militaryInstallations: lazyJsonTable("datacore/military_installations.json"),
  quakeHistory: lazyJsonTable("datacore/quake_history.json"),
  nuclearTests: lazyJsonTable("datacore/nuclear_tests.json"),
  boundariesAdmin0: lazyJsonTable("datacore/boundaries/ne_110m_admin0.json"),
  boundariesAdmin1Lines: lazyJsonTable("datacore/boundaries/ne_50m_admin1_lines.json"),
  pfasUcmr5: lazyJsonTable("datacore/pfas/ucmr5_detections.json"),
  cdcCountyRates: lazyJsonTable("datacore/cdc_cancer/county_rates.json"),
  cdcCountiesTopo: lazyJsonTable("datacore/cdc_cancer/counties-10m.topo.json"),
  jodiPrimaryStocks: lazyJsonTable("datacore/jodi/primary_stocks.json", { refreshable: true }),
  railEp724: lazyJsonTable("datacore/rail/ep724_carloads.json", { refreshable: true }),
  // EIA weekly: no server route reads it — only its inventory build stamp
  // (snapshotStreams peek()), so in lazy mode it is parsed once and dropped.
  eiaWeekly: lazyJsonTable("datacore/eia/weekly_series.json", { refreshable: true }),
  signalLadder: lazyJsonTable("datacore/signal_ladder.json"),
} as const;

export type LazyTableName = keyof typeof LAZY_TABLES;

/** The four tables the nightly job refreshes (snapshot_refresh_job.py SOURCES;
 *  server/snapshotRefresh.test.ts pins the two lists against each other). */
export const REFRESHABLE_TABLES: readonly LazyJsonTable[] = [
  LAZY_TABLES.eiaWeekly, LAZY_TABLES.cpcDegreeDays, LAZY_TABLES.railEp724, LAZY_TABLES.jodiPrimaryStocks,
];

export function lazyTableList(): LazyJsonTable[] {
  return Object.values(LAZY_TABLES);
}

// ── sharp / libvips ──────────────────────────────────────────────────────────
let sharpMemo: typeof SharpLib | null = null;

/** sharp (libvips) on first use. A missing or broken native module rejects
 *  with the loader's own error on EVERY call — nothing is cached on failure —
 *  so the caller sees exactly what the static import would have thrown, at
 *  request time instead of at boot. */
export async function getSharp(): Promise<typeof SharpLib> {
  if (sharpMemo) return sharpMemo;
  const mod = (await import("sharp")) as unknown as { default?: typeof SharpLib };
  sharpMemo = (typeof mod.default === "function" ? mod.default : (mod as unknown)) as typeof SharpLib;
  return sharpMemo;
}

export function sharpLoaded(): boolean {
  return sharpMemo !== null;
}

// ── eager mode ───────────────────────────────────────────────────────────────
export interface PreloadReport {
  mode: "lazy" | "eager";
  /** tables loaded (eager) */
  tables: number;
  /** tables whose load threw (eager) — each was logged; its routes will throw the same error */
  failed: number;
  /** bytes of JSON read (eager) */
  bytes: number;
  ms: number;
  sharp: "loaded" | "failed" | "skipped";
}

/** VOLTRADE_LAZY_ASSETS=0 → run every loader now (the pre-A7 boot profile).
 *  Default → nothing happens here and each asset loads on first use. A
 *  table or sharp that fails to load is logged loudly and skipped rather
 *  than aborting the boot: the blast radius stays the same as in lazy mode
 *  (that asset's own requests fail with the same error) instead of turning
 *  a packaging defect into a deploy the health gate rejects. */
export async function preloadLazyAssets(env: NodeJS.ProcessEnv = process.env): Promise<PreloadReport> {
  if (lazyAssetsEnabled(env)) return { mode: "lazy", tables: 0, failed: 0, bytes: 0, ms: 0, sharp: "skipped" };
  const t0 = Date.now();
  let bytes = 0;
  let failed = 0;
  let tables = 0;
  for (const t of lazyTableList()) {
    try {
      t.get();
      tables++;
      bytes += t.bytes();
    } catch (e: unknown) {
      failed++;
      console.error(`[lazyAssets] eager preload: ${e instanceof Error ? e.message : String(e)}`);
    }
  }
  let sharp: PreloadReport["sharp"] = "loaded";
  try {
    await getSharp();
  } catch (e: unknown) {
    sharp = "failed";
    console.error(`[lazyAssets] eager preload: sharp failed to load — FAA chart tiles will error on request: ${e instanceof Error ? e.message : String(e)}`);
  }
  return { mode: "eager", tables, failed, bytes, ms: Date.now() - t0, sharp };
}

/** One line for the boot log. */
export function describePreload(r: PreloadReport): string {
  if (r.mode === "lazy") return `lazy (VOLTRADE_LAZY_ASSETS unset/1): ${lazyTableList().length} datacore tables + sharp load on first use`;
  return `eager (VOLTRADE_LAZY_ASSETS=0): ${r.tables} tables / ${(r.bytes / (1024 * 1024)).toFixed(1)}MB preloaded in ${r.ms}ms` +
    (r.failed ? `, ${r.failed} FAILED (see errors above)` : "") + `, sharp ${r.sharp}`;
}
