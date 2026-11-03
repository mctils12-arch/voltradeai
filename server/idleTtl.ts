/**
 * idleTtl.ts — idle time-to-live for the dashboard/map caches and the lazy
 * reference datasets (memory audit 2026-11-03, fix A3 — item 1d in
 * docs/memory_audit_2026-11-03.md).
 *
 * WHY: the /data map's caches (archive window reads, world aircraft bodies,
 * bbox snapshots, per-hour-file id Bloom filters) only ever evicted by COUNT,
 * and the reference datasets (OurAirports grid, the GEM registries) were
 * parsed on the first dashboard hit and held for the life of the process.
 * Nothing in that set is needed while nobody is looking at the dashboard, yet
 * it sat in the heap overnight, on every one of the 5-12 containers a day.
 *
 * MODEL — three wrappers, one registry, one unref'd sweep timer:
 *   - idleTtlMap(name): a Map whose entries remember their last access.
 *     Entries unused for the TTL are dropped by a sweep (on access, rate-
 *     limited to once a minute per map, plus the shared timer). Insertion
 *     order, size, keys()/entries() shapes and the callers' own cap-eviction
 *     lines are untouched — the handlers run exactly as before, they just
 *     see a miss where a stale entry used to be.
 *   - lazyDataset(name, loader): a module-level singleton. get() runs the
 *     SAME loader the module used before (byte-identical result, same
 *     sync-ness), hands back the same reference while the dataset stays in
 *     use, and releases it after the TTL unused so the next get() reloads
 *     from disk.
 *   - idleRelease(name, release, isHeld): for a module whose state is several
 *     bindings (airportsIndex's grid + byIdent + count): touch() records use,
 *     the sweep calls release() once the state has been idle for the TTL.
 *
 * FLAG — VOLTRADE_IDLE_TTL_MIN: unset/empty = 12 minutes (DEFAULT ON);
 *   `0` = disabled (nothing ever drops — exactly the pre-A3 behavior);
 *   10..15 honored; anything else is clamped into 10..15 and logged once.
 *   One flag for every A3 structure (the audit treats A3 as one fix).
 *
 * HOT-PATH SAFETY: a drop only removes the holder's reference. A value a
 * handler already pulled onto its stack stays valid (JS keeps it alive), and
 * nothing is ever mutated on drop. Every reload path is the structure's own
 * existing cold path, so the request after a drop costs exactly what the
 * first request after a deploy already costs today — never more (Law V).
 *
 * OBSERVABILITY: every wrapper exposes stats() (entries/loaded, ttl_s,
 * last_used_s, drops, loads); heapDiag's trackMap merges them into the
 * structure's own gauge, and the aggregate `idleTtl` gauge lists every
 * managed structure so a /api/health read shows whether drops are happening.
 */
import { registerHeapGauge, type GaugeValue } from "./heapDiag";

export const IDLE_TTL_FLAG = "VOLTRADE_IDLE_TTL_MIN";
export const IDLE_TTL_DEFAULT_MIN = 12;
export const IDLE_TTL_MIN_MIN = 10;
export const IDLE_TTL_MAX_MIN = 15;
/** On-access sweeps are rate-limited to this per structure, and the shared
 *  timer ticks at it — so an idle entry lives at most TTL + one tick. */
export const IDLE_SWEEP_INTERVAL_MS = 60_000;

export type IdleTtlSource = "default" | "disabled" | "env" | "clamped" | "invalid";

export interface IdleTtlResolution {
  /** 0 = disabled */
  ttlMs: number;
  source: IdleTtlSource;
  raw: string | undefined;
}

/** Pure: the flag's value → TTL in ms. Exported for tests; the process-wide
 *  memoized reading is idleTtlMs(). */
export function resolveIdleTtl(env: NodeJS.ProcessEnv = process.env): IdleTtlResolution {
  const raw = env[IDLE_TTL_FLAG];
  const s = (raw ?? "").trim();
  if (s === "") return { ttlMs: IDLE_TTL_DEFAULT_MIN * 60_000, source: "default", raw };
  const n = Number(s);
  if (!Number.isFinite(n)) return { ttlMs: IDLE_TTL_DEFAULT_MIN * 60_000, source: "invalid", raw };
  if (n === 0) return { ttlMs: 0, source: "disabled", raw };
  const clamped = Math.min(IDLE_TTL_MAX_MIN, Math.max(IDLE_TTL_MIN_MIN, n));
  return { ttlMs: Math.round(clamped * 60_000), source: clamped === n ? "env" : "clamped", raw };
}

let resolved: IdleTtlResolution | null = null;

/** The process-wide TTL (ms), read from the environment once and logged once. */
export function idleTtlMs(): number {
  if (!resolved) {
    resolved = resolveIdleTtl();
    const min = resolved.ttlMs / 60_000;
    if (resolved.source === "clamped" || resolved.source === "invalid") {
      console.warn(`[idle-ttl] ${IDLE_TTL_FLAG}=${JSON.stringify(resolved.raw)} is not 0 or ${IDLE_TTL_MIN_MIN}..${IDLE_TTL_MAX_MIN} — using ${min} min`);
    } else if (resolved.source === "disabled") {
      console.log(`[idle-ttl] disabled (${IDLE_TTL_FLAG}=0): caches and lazy datasets are never dropped`);
    } else {
      console.log(`[idle-ttl] idle TTL ${min} min (${resolved.source})`);
    }
  }
  return resolved.ttlMs;
}

export interface IdleStats extends GaugeValue {
  /** map entries, or 1/0 for a dataset */
  entries: number;
  /** effective TTL in seconds; 0 = disabled */
  ttl_s: number;
  /** seconds since the structure was last used; null = never */
  last_used_s: number | null;
  /** entries (or datasets) dropped by idle sweeps since boot */
  drops: number;
  /** datasets only: loads since boot (1 + reloads after drops) */
  loads?: number;
  /** datasets only */
  loaded?: boolean;
}

interface IdleOpts {
  /** per-structure override (tests); default = the process-wide flag */
  ttlMs?: number;
  /** clock injection (tests) */
  now?: () => number;
}

interface Sweepable {
  readonly name: string;
  sweep(nowMs?: number): number;
  stats(): IdleStats;
}

// ── registry + the one shared sweep timer ────────────────────────────────────
const registry = new Map<string, Sweepable>();
let timer: ReturnType<typeof setInterval> | null = null;

function ensureTimer(): void {
  if (timer || idleTtlMs() <= 0) return;
  timer = setInterval(() => {
    try { sweepIdleTtl(Date.now()); } catch (e: unknown) {
      console.error("[idle-ttl] sweep:", e instanceof Error ? e.message : String(e));
    }
  }, IDLE_SWEEP_INTERVAL_MS);
  timer.unref?.();
}

function register(s: Sweepable): void {
  // replace-by-name (a route module re-registered in tests): the newest
  // instance is the live one; the old one is garbage with its owner.
  registry.set(s.name, s);
  ensureTimer();
}

/** Sweeps every managed structure as of nowMs; returns the total dropped.
 *  Tests pass a far-future time to force the idle drop. */
export function sweepIdleTtl(nowMs: number = Date.now()): number {
  let n = 0;
  for (const s of registry.values()) n += s.sweep(nowMs);
  return n;
}

/** Per-structure stats, keyed by name (sorted). */
export function idleTtlStats(): Record<string, IdleStats> {
  const out: Record<string, IdleStats> = {};
  for (const name of Array.from(registry.keys()).sort()) out[name] = registry.get(name)!.stats();
  return out;
}

/** Tests: stop the shared timer and forget the memoized flag. The registry is
 *  deliberately left alone — the module-level singletons registered at import
 *  (bloom cache, airports grid, GEM datasets) must stay reachable through
 *  sweepIdleTtl(); throwaway test instances unregister via dispose(). */
export function _resetIdleTtlForTests(): void {
  if (timer) clearInterval(timer);
  timer = null;
  resolved = null;
}

// ── shared base: last-use clock, rate-limited sweep, counters ────────────────
abstract class IdleBase implements Sweepable {
  /** ms of the last recorded use; null = never used (0 is a valid clock reading) */
  protected lastUsed: number | null = null;
  protected lastSweep: number;
  protected drops = 0;
  constructor(readonly name: string, protected readonly opts: IdleOpts) {
    this.lastSweep = this.now();
  }
  protected now(): number { return (this.opts.now ?? Date.now)(); }
  /** effective TTL; a per-structure override wins, else the process flag */
  ttlMs(): number { return this.opts.ttlMs ?? idleTtlMs(); }
  protected touchAt(t: number): void { this.lastUsed = t; }
  /** ms idle as of nowMs (never used = idle since the epoch) */
  protected idleFor(nowMs: number): number { return nowMs - (this.lastUsed ?? 0); }
  /** Unregister; subclasses release what they hold. */
  dispose(): void {
    if (registry.get(this.name) === this) registry.delete(this.name);
    this.onDispose();
  }
  protected onDispose(): void { /* nothing held by default */ }
  protected maybeSweep(t: number): void {
    if (t - this.lastSweep >= IDLE_SWEEP_INTERVAL_MS) this.sweep(t);
  }
  /** Drops what has been idle for the TTL as of nowMs; returns the count. 0 TTL = never. */
  sweep(nowMs: number = this.now()): number {
    this.lastSweep = nowMs;
    const ttl = this.ttlMs();
    if (ttl <= 0) return 0;
    const n = this.dropIdle(nowMs, ttl);
    this.drops += n;
    return n;
  }
  protected abstract dropIdle(nowMs: number, ttlMs: number): number;
  protected baseStats(entries: number): IdleStats {
    const t = this.now();
    return {
      entries,
      ttl_s: Math.round(this.ttlMs() / 1000),
      last_used_s: this.lastUsed === null ? null : Math.max(0, Math.round((t - this.lastUsed) / 1000)),
      drops: this.drops,
    };
  }
  abstract stats(): IdleStats;
}

// ── idleTtlMap ──────────────────────────────────────────────────────────────
interface Slot<V> { v: V; used: number }

/** A Map whose entries are dropped after the idle TTL. The surface the
 *  dashboard caches use — size, get, set, has, delete, clear, keys, values,
 *  entries, iteration — keeps Map semantics: insertion order is preserved
 *  (set on an existing key keeps its position; get never reorders), so the
 *  callers' FIFO / oldest-by-timestamp eviction lines behave identically. */
export class IdleTtlMap<K, V> extends IdleBase {
  private readonly m = new Map<K, Slot<V>>();

  constructor(name: string, opts: IdleOpts = {}) {
    super(name, opts);
    register(this);
  }

  get size(): number { return this.m.size; }

  /** Records the access (the entry survives the sweep that may run now). */
  get(key: K): V | undefined {
    const t = this.now();
    const s = this.m.get(key);
    if (s) { s.used = t; this.touchAt(t); }
    this.maybeSweep(t);
    return s?.v;
  }

  /** Read without recording a use (tests / diagnostics). */
  peek(key: K): V | undefined { return this.m.get(key)?.v; }

  has(key: K): boolean { return this.m.has(key); }

  set(key: K, value: V): this {
    const t = this.now();
    const s = this.m.get(key);
    if (s) { s.v = value; s.used = t; }
    else this.m.set(key, { v: value, used: t });
    this.touchAt(t);
    this.maybeSweep(t);
    return this;
  }

  delete(key: K): boolean { return this.m.delete(key); }
  clear(): void { this.m.clear(); }

  /** The underlying Map's own key iterator (insertion order). */
  keys(): MapIterator<K> { return this.m.keys(); }
  *values(): IterableIterator<V> { for (const s of this.m.values()) yield s.v; }
  *entries(): IterableIterator<[K, V]> { for (const [k, s] of this.m) yield [k, s.v]; }
  [Symbol.iterator](): IterableIterator<[K, V]> { return this.entries(); }

  protected dropIdle(nowMs: number, ttlMs: number): number {
    let n = 0;
    for (const [k, s] of this.m) {
      if (nowMs - s.used >= ttlMs) { this.m.delete(k); n++; }
    }
    return n;
  }

  stats(): IdleStats { return this.baseStats(this.m.size); }

  protected onDispose(): void { this.m.clear(); }
}

export function idleTtlMap<K, V>(name: string, opts: IdleOpts = {}): IdleTtlMap<K, V> {
  return new IdleTtlMap<K, V>(name, opts);
}

// ── lazyDataset ─────────────────────────────────────────────────────────────
/** A module-level singleton loaded through `loader` on first use (and again
 *  after every idle release). `null`/`undefined` results are cached like any
 *  other value — a loader that degrades to null on a missing file is not
 *  re-run until the TTL passes, exactly as the modules' `cached` did. */
export class LazyDataset<T> extends IdleBase {
  private value: T | undefined;
  private loaded = false;
  private loads = 0;

  constructor(name: string, private readonly loader: () => T, opts: IdleOpts = {}) {
    super(name, opts);
    register(this);
  }

  get(): T {
    const t = this.now();
    if (!this.loaded) {
      this.value = this.loader();
      this.loaded = true;
      this.loads++;
    }
    this.touchAt(t);
    this.maybeSweep(t);
    return this.value as T;
  }

  isLoaded(): boolean { return this.loaded; }

  /** The held value without loading or recording a use. */
  peek(): T | undefined { return this.value; }

  /** Forget the value (today's `_reset…ForTests` hooks call this). */
  reset(): void { this.value = undefined; this.loaded = false; }

  protected dropIdle(nowMs: number, ttlMs: number): number {
    if (!this.loaded || this.idleFor(nowMs) < ttlMs) return 0;
    console.log(`[idle-ttl] released ${this.name} after ${Math.round(this.idleFor(nowMs) / 60_000)} min idle`);
    this.reset();
    return 1;
  }

  stats(): IdleStats {
    return { ...this.baseStats(this.loaded ? 1 : 0), loaded: this.loaded, loads: this.loads };
  }

  protected onDispose(): void { this.reset(); }
}

export function lazyDataset<T>(name: string, loader: () => T, opts: IdleOpts = {}): LazyDataset<T> {
  return new LazyDataset<T>(name, loader, opts);
}

// ── idleRelease ─────────────────────────────────────────────────────────────
/** For a module whose lazily-loaded state lives in its own bindings: the
 *  module calls touch() on every use; the sweep calls release() once the
 *  state (per isHeld) has been idle for the TTL. */
export class IdleRelease extends IdleBase {
  constructor(name: string, private readonly release: () => void, private readonly isHeld: () => boolean, opts: IdleOpts = {}) {
    super(name, opts);
    register(this);
  }

  touch(): void {
    const t = this.now();
    this.touchAt(t);
    this.maybeSweep(t);
  }

  protected dropIdle(nowMs: number, ttlMs: number): number {
    if (!this.isHeld() || this.idleFor(nowMs) < ttlMs) return 0;
    console.log(`[idle-ttl] released ${this.name} after ${Math.round(this.idleFor(nowMs) / 60_000)} min idle`);
    this.release();
    return 1;
  }

  stats(): IdleStats {
    const held = this.isHeld();
    return { ...this.baseStats(held ? 1 : 0), loaded: held };
  }
}

export function idleRelease(name: string, release: () => void, isHeld: () => boolean, opts: IdleOpts = {}): IdleRelease {
  return new IdleRelease(name, release, isHeld, opts);
}

// ── aggregate gauge: one /api/health entry listing every managed structure ──
registerHeapGauge("idleTtl", () => {
  const ttl = idleTtlMs();
  const structures = idleTtlStats();
  let drops = 0;
  for (const s of Object.values(structures)) drops += s.drops;
  return { enabled: ttl > 0, ttl_min: ttl / 60_000, flag: IDLE_TTL_FLAG, entries: registry.size, drops, structures };
});
