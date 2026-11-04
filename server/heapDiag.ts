/**
 * heapDiag.ts — read-only heap composition + cache-size gauges for
 * /api/health (memory audit 2026-11-03, fix A0; docs/memory_audit_2026-11-03.md).
 *
 * WHY: the audit found Node's heap parked at ~1.9-2.1 GB while every probe
 * only reported heapUsed/rss totals — nothing said WHICH structures held it
 * (the DTCC seen-ID set alone is ~0.4-0.5 GB, the raster bloom cache up to
 * 64 MB, the dashboard caches, the per-poller dedup sets…). Every later
 * memory fix (A1-A8) is measured against this baseline, so it ships first
 * and changes NOTHING else: it reads sizes that modules already track.
 *
 * MODEL: modules register a gauge (a cheap closure returning counts/bytes);
 * the health handler reads them all. A gauge that throws is reported as an
 * error entry — the diagnostic can never fail the health response. Every
 * read is O(number of gauges); no gauge may walk or serialize its structure
 * (byte figures are arithmetic estimates, labelled as such).
 *
 * FLAG: VOLTRADE_HEAP_DIAG=0 turns the block off (payload carries
 * {enabled:false}); unset/any other value = on. Read-only, so on by default.
 */
import v8 from "node:v8";

export const HEAP_DIAG_FLAG = "VOLTRADE_HEAP_DIAG";

export interface GaugeValue {
  /** number of entries (Map/Set size, array length, loaded rows…) */
  entries?: number;
  /** arithmetic estimate of retained bytes — never a serialization walk */
  bytes?: number;
  /** the structure's own cap, when it has one */
  cap?: number;
  /** anything else the module wants to say (age, mode, hit counts…) */
  [k: string]: unknown;
}
export type GaugeFn = () => GaugeValue;

const gauges = new Map<string, GaugeFn>();

/** Register (or replace) a named gauge. Names are dotted module.structure. */
export function registerHeapGauge(name: string, fn: GaugeFn): void {
  gauges.set(name, fn);
}

/** A sized structure; one that knows more about itself (an idleTtl.ts map
 *  reports ttl_s / last_used_s / drops) merges that into its reading. */
interface Sized { readonly size: number; stats?: () => GaugeValue }

/** Convenience: a Set's size (+ optional cap, + the structure's own stats). */
export function trackSet(name: string, set: Sized, cap?: number): void {
  registerHeapGauge(name, () => {
    const base: GaugeValue = cap == null ? { entries: set.size } : { entries: set.size, cap };
    return typeof set.stats === "function" ? { ...set.stats(), ...base } : base;
  });
}

/** Convenience: a Map's size (+ optional cap, + the structure's own stats). */
export function trackMap(name: string, map: Sized, cap?: number): void {
  trackSet(name, map, cap);
}

export function heapDiagEnabled(env: NodeJS.ProcessEnv = process.env): boolean {
  return env[HEAP_DIAG_FLAG] !== "0";
}

const MB = 1024 * 1024;
const mb = (n: number): number => Math.round((n / MB) * 10) / 10;

export interface HeapSpace {
  used_mb: number;
  size_mb: number;
  physical_mb: number;
  available_mb: number;
}

export interface HeapComposition {
  spaces: Record<string, HeapSpace>;
  /** process.memoryUsage() split that the summary line hides */
  process: { heap_used_mb: number; heap_total_mb: number; rss_mb: number; external_mb: number; array_buffers_mb: number };
  v8: {
    heap_size_limit_mb: number;
    malloced_mb: number;
    peak_malloced_mb: number;
    native_contexts: number;
    /** >0 is the classic leak signal: a context (e.g. a vm/worker) that was
     *  released but is still reachable */
    detached_contexts: number;
  };
}

/** Per-space V8 heap statistics (old_space is where long-lived structures
 *  live; large_object_space holds big arrays/strings; code_space is JIT).
 *  v8.getHeapSpaceStatistics() does not trigger a GC and costs microseconds. */
export function heapComposition(): HeapComposition {
  const spaces: Record<string, HeapSpace> = {};
  for (const s of v8.getHeapSpaceStatistics()) {
    spaces[s.space_name] = {
      used_mb: mb(s.space_used_size),
      size_mb: mb(s.space_size),
      physical_mb: mb(s.physical_space_size),
      available_mb: mb(s.space_available_size),
    };
  }
  const hs = v8.getHeapStatistics();
  const mu = process.memoryUsage();
  return {
    spaces,
    process: {
      heap_used_mb: mb(mu.heapUsed),
      heap_total_mb: mb(mu.heapTotal),
      rss_mb: mb(mu.rss),
      external_mb: mb(mu.external),
      array_buffers_mb: mb(mu.arrayBuffers),
    },
    v8: {
      heap_size_limit_mb: mb(hs.heap_size_limit),
      malloced_mb: mb(hs.malloced_memory),
      peak_malloced_mb: mb(hs.peak_malloced_memory),
      native_contexts: hs.number_of_native_contexts,
      detached_contexts: hs.number_of_detached_contexts,
    },
  };
}

export type GaugeReading = GaugeValue | { error: string };

/** Every registered gauge, each guarded: one bad closure never hides the rest. */
export function readHeapGauges(): Record<string, GaugeReading> {
  const out: Record<string, GaugeReading> = {};
  for (const [name, fn] of Array.from(gauges.entries()).sort((a, b) => a[0].localeCompare(b[0]))) {
    try {
      out[name] = fn();
    } catch (err: unknown) {
      out[name] = { error: String((err as Error)?.message || err).slice(0, 120) };
    }
  }
  return out;
}

export type HeapDiagnostic =
  | { enabled: false; flag: string }
  | { enabled: true; flag: string; gauge_count: number; composition: HeapComposition; caches: Record<string, GaugeReading> };

/** The block /api/health publishes under checks.memory.detail. */
export function heapDiagnostic(env: NodeJS.ProcessEnv = process.env): HeapDiagnostic {
  if (!heapDiagEnabled(env)) return { enabled: false, flag: HEAP_DIAG_FLAG };
  return {
    enabled: true,
    flag: HEAP_DIAG_FLAG,
    gauge_count: gauges.size,
    composition: heapComposition(),
    caches: readHeapGauges(),
  };
}

export function _resetHeapGaugesForTests(): void {
  gauges.clear();
}
