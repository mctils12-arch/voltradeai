/**
 * memoryCeiling.ts — where the walls are. Pure except for two tiny sysfs
 * reads (injectable), node:test safe.
 *
 * WHY (KNOWN BROKEN #41, the 2026-09-08 crash loop): three sessions watched
 * `/api/health`'s rss climb 500 -> 770-990 MB and then the process vanish,
 * and could not say whether that was V8's `--max-old-space-size` (a
 * "FATAL ERROR: Reached heap limit" — the Node process alone) or the
 * container's cgroup limit (a bare SIGKILL — Node + the Python daemon +
 * every subprocess child, summed). The two need different fixes and look
 * identical from outside, because nothing reported either ceiling:
 * run_with_daemon.sh computes the heap cap from the cgroup limit at boot
 * and prints it once to a stdout nobody in this sandbox can read.
 *
 * This puts both walls, and how close we are to each, in the payload every
 * probe already fetches. The next ramp is classifiable from the outside in
 * one read: `pressure` names the wall being approached.
 *
 * cgroup v2: /sys/fs/cgroup/memory.max ("max" = unlimited), memory.current.
 * cgroup v1: /sys/fs/cgroup/memory/memory.limit_in_bytes (>= 2^60 means
 * unlimited), memory.usage_in_bytes. Anything unreadable -> null, never a
 * throw: this must not be able to fail the health handler.
 */
import fs from "node:fs";
import v8 from "node:v8";

export type ReadFile = (path: string) => string;

export interface CgroupMemory {
  limitBytes: number | null;
  usageBytes: number | null;
  /** "v2" | "v1" | null when neither hierarchy was readable */
  source: "v2" | "v1" | null;
}

export interface MemoryCeiling {
  /** V8's old-space cap for THIS process (what --max-old-space-size set) */
  heapLimitMB: number;
  /** container-wide limit (Node + daemon + children); null = unknown/unlimited */
  cgroupLimitMB: number | null;
  /** container-wide usage right now; null = unreadable */
  cgroupUsageMB: number | null;
  cgroupSource: "v2" | "v1" | null;
  /** cgroupLimitMB - cgroupUsageMB, null when either is unknown */
  cgroupHeadroomMB: number | null;
}

export type MemoryPressure = "ok" | "heap-near-limit" | "cgroup-near-limit" | "unknown";

/** Fraction of a ceiling at which the wall is "near". 0.85 leaves one more
 *  probe interval to see it coming on the observed ~90-130s ramps. */
export const NEAR_LIMIT_FRACTION = 0.85;
/** cgroup v1 reports "unlimited" as a huge number rather than a word. */
const V1_UNLIMITED_FLOOR = 2 ** 60;

const MB = 1048576;

const defaultRead: ReadFile = (p) => fs.readFileSync(p, "utf8");

function parseBytes(raw: string | null): number | null {
  if (raw == null) return null;
  const t = raw.trim();
  if (t === "max" || t === "") return null;
  const n = Number(t);
  if (!Number.isFinite(n) || n < 0) return null;
  if (n >= V1_UNLIMITED_FLOOR) return null;
  return n;
}

function tryRead(readFile: ReadFile, p: string): string | null {
  try { return readFile(p); } catch { return null; }
}

export function readCgroupMemory(readFile: ReadFile = defaultRead): CgroupMemory {
  const v2max = tryRead(readFile, "/sys/fs/cgroup/memory.max");
  const v2cur = tryRead(readFile, "/sys/fs/cgroup/memory.current");
  if (v2max !== null || v2cur !== null) {
    return { limitBytes: parseBytes(v2max), usageBytes: parseBytes(v2cur), source: "v2" };
  }
  const v1max = tryRead(readFile, "/sys/fs/cgroup/memory/memory.limit_in_bytes");
  const v1cur = tryRead(readFile, "/sys/fs/cgroup/memory/memory.usage_in_bytes");
  if (v1max !== null || v1cur !== null) {
    return { limitBytes: parseBytes(v1max), usageBytes: parseBytes(v1cur), source: "v1" };
  }
  return { limitBytes: null, usageBytes: null, source: null };
}

export function memoryCeiling(opts: { readFile?: ReadFile; heapLimitBytes?: number } = {}): MemoryCeiling {
  const heapLimitBytes = opts.heapLimitBytes ?? v8.getHeapStatistics().heap_size_limit;
  const cg = readCgroupMemory(opts.readFile ?? defaultRead);
  const limit = cg.limitBytes == null ? null : Math.round(cg.limitBytes / MB);
  const usage = cg.usageBytes == null ? null : Math.round(cg.usageBytes / MB);
  return {
    heapLimitMB: Math.round(heapLimitBytes / MB),
    cgroupLimitMB: limit,
    cgroupUsageMB: usage,
    cgroupSource: cg.source,
    cgroupHeadroomMB: limit == null || usage == null ? null : limit - usage,
  };
}

/** Which wall, if any, the process is approaching. Heap is checked first:
 *  a heap-near-limit reading with cgroup headroom is a Node-side problem
 *  (a V8 OOM is coming), while cgroup-near-limit with heap headroom means
 *  the container as a whole — daemon + children — is what will be killed. */
export function memoryPressure(heapUsedMB: number, ceiling: MemoryCeiling): MemoryPressure {
  if (ceiling.heapLimitMB > 0 && heapUsedMB >= ceiling.heapLimitMB * NEAR_LIMIT_FRACTION) return "heap-near-limit";
  if (ceiling.cgroupLimitMB == null || ceiling.cgroupUsageMB == null) {
    return ceiling.heapLimitMB > 0 ? "ok" : "unknown";
  }
  if (ceiling.cgroupUsageMB >= ceiling.cgroupLimitMB * NEAR_LIMIT_FRACTION) return "cgroup-near-limit";
  return "ok";
}
