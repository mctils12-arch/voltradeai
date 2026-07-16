// Satellite propagation Web Worker — the off-main-thread SGP4 loop for the
// ORBITAL program (research/orbital_program.md → O1/O2). It owns the GP
// element set and, once per tick (~1-2 Hz), propagates the FULL population
// with the zero-dependency inline SGP4 in ./propagate and posts back a
// transferable Float32Array of render-ready positions. The vertex shader in
// ./satLayer interpolates between ticks for smooth 60fps (parent wiring).
//
// WHY A WORKER: 10k+ SGP4 propagations per tick (~29ms for 10k in V8, O1
// spike) must not block the main thread. The worker keeps the render loop at
// 60fps while positions refresh at a low rate.
//
// HONESTY (charter): REAL POSITIONS ONLY. Since the SDP4 port (O6-5,
// 2026-07-16) deep-space objects (GEO comms, GPS/GLONASS/Galileo MEO,
// Molniya) propagate for real alongside the near-earth population. Objects
// the kernel still refuses (incomplete/decayed elements) are written as a
// SENTINEL and COUNTED (deepSpaceSkipped now counts only deep-space objects
// that FAILED, not a skipped class) — we NEVER fabricate a position. No
// silent drops.
//
// INDEX ALIGNMENT (load-bearing for picking): the output buffer keeps ONE
// slot per input GP record, in input order, INCLUDING skipped objects (their
// slot carries the sentinel). So buffer index i ALWAYS maps to gp[i]. The
// parent keeps a parallel metadata array (noradId, epoch, name…) aligned by
// the same index; CPU nearest-point picking resolves a click to index i and
// straight into that metadata. Compacting/omitting would break this.
//
// TESTABILITY: all the real logic lives in PURE exported functions
// (packPositions, analyzePopulation, the mercator helpers, readSatAt) that
// take injected time and injected propagate/isDeepSpace deps — hermetic under
// `npx tsx --test`, no Worker, no WebGL, no network, no clock. The Worker
// runtime is a thin wrapper (createSatWorker / installWorker) around them.

import type { GpRecord } from './tle.js';
import {
  propagate as realPropagate,
  isDeepSpace as realIsDeepSpace,
  orbitClassFromAltKm,
} from './propagate.js';
import {
  SAT_STRIDE,
  SENTINEL_SKIP,
  CLASS_CODE,
  lonLatToMercator,
  type OrbitClassCode,
} from './satBuffer.js';

// Re-export the buffer-layout contract so the worker's message stride and the
// layer stay provably in sync (single source of truth in ./satBuffer).
export { SAT_STRIDE, SENTINEL_SKIP, CLASS_CODE, lonLatToMercator, mercatorToLonLat, readSatAt } from './satBuffer.js';
export type { OrbitClassCode } from './satBuffer.js';

/** Altitude (km) -> packed orbit-class code (0 LEO / 1 MEO / 2 GEO). */
export function orbitClassCode(altKm: number): OrbitClassCode {
  const c = orbitClassFromAltKm(altKm);
  return c === 'LEO' ? CLASS_CODE.LEO : c === 'MEO' ? CLASS_CODE.MEO : CLASS_CODE.GEO;
}

// ---------------------------------------------------------------------------
// Message protocol (typed)
// ---------------------------------------------------------------------------

/** main -> worker: load/replace the satellite set. */
export interface SatInitMessage {
  type: 'init';
  gp: GpRecord[];
}
/** main -> worker: propagate once to an explicit time and post positions back. */
export interface SatTickMessage {
  type: 'tick';
  timeMs: number;
}
/** main -> worker: begin self-driven propagation at `hz` ticks/sec (default 1). */
export interface SatStartMessage {
  type: 'start';
  hz?: number;
}
/** main -> worker: stop self-driven propagation. */
export interface SatStopMessage {
  type: 'stop';
}
export type SatWorkerInbound =
  | SatInitMessage
  | SatTickMessage
  | SatStartMessage
  | SatStopMessage;

/** worker -> main: acknowledges an init with a population breakdown. */
export interface SatReadyMessage {
  type: 'ready';
  total: number;
  /** objects the near-earth kernel WILL attempt (not deep-space). */
  nearEarth: number;
  /** objects that need SDP4 (skipped, honestly counted). */
  deepSpace: number;
  /** objects missing required orbital elements (cannot propagate). */
  noElements: number;
}
/** worker -> main: a freshly propagated population. `buf` is transferable. */
export interface SatPositionsMessage {
  type: 'positions';
  timeMs: number;
  stride: number;
  /** total slots in the buffer (== gp.length; index-aligned to it). */
  total: number;
  /** slots carrying a REAL propagated position (rendered). */
  shown: number;
  /** slots skipped because the object is deep-space (needs SDP4). */
  deepSpaceSkipped: number;
  /** slots skipped for any other reason (missing elements / decayed / degenerate). */
  invalidSkipped: number;
  /** interleaved Float32Array buffer, SAT_STRIDE floats per object. */
  buf: ArrayBuffer;
}
export type SatWorkerOutbound = SatReadyMessage | SatPositionsMessage;

// ---------------------------------------------------------------------------
// Pure core: counts + packing
// ---------------------------------------------------------------------------

/** Injectable deps so the pure functions are testable without real orbital math. */
export interface PropagationDeps {
  propagate?: (gp: GpRecord, dateMs: number) => { latDeg: number; lonDeg: number; altKm: number } | null;
  isDeepSpace?: (gp: GpRecord) => boolean;
}

export interface PopulationCounts {
  total: number;
  nearEarth: number;
  deepSpace: number;
  noElements: number;
}

function hasElements(gp: GpRecord): boolean {
  return (
    gp.meanMotion != null &&
    gp.ecc != null &&
    gp.inclination != null &&
    gp.raan != null &&
    gp.argp != null &&
    gp.meanAnomaly != null &&
    gp.epoch != null
  );
}

/**
 * Static breakdown of a population WITHOUT propagating: how many are
 * near-earth (kernel will attempt), deep-space (need SDP4), or missing
 * elements. Used for the `ready` acknowledgement. Pure.
 */
export function analyzePopulation(gp: GpRecord[], deps: PropagationDeps = {}): PopulationCounts {
  const isDeepSpace = deps.isDeepSpace ?? realIsDeepSpace;
  let deepSpace = 0;
  let noElements = 0;
  let nearEarth = 0;
  for (const g of gp) {
    if (!hasElements(g)) {
      noElements++;
      continue;
    }
    if (isDeepSpace(g)) deepSpace++;
    else nearEarth++;
  }
  return { total: gp.length, nearEarth, deepSpace, noElements };
}

export interface PackResult {
  buffer: Float32Array;
  shown: number;
  deepSpaceSkipped: number;
  invalidSkipped: number;
  total: number;
}

/**
 * Propagate every GP record to `timeMs` and pack render-ready positions into a
 * fresh Float32Array (SAT_STRIDE floats each, index-aligned to `gp`). Skipped
 * objects (null propagation) get the SENTINEL_SKIP classCode and are counted:
 * deep-space -> deepSpaceSkipped, everything else -> invalidSkipped. Pure and
 * deterministic (time injected; propagate/isDeepSpace injectable). NEVER writes
 * a fabricated position.
 */
export function packPositions(
  gp: GpRecord[],
  timeMs: number,
  deps: PropagationDeps = {},
): PackResult {
  const propagate = deps.propagate ?? realPropagate;
  const isDeepSpace = deps.isDeepSpace ?? realIsDeepSpace;
  const n = gp.length;
  const buffer = new Float32Array(n * SAT_STRIDE);
  let shown = 0;
  let deepSpaceSkipped = 0;
  let invalidSkipped = 0;

  for (let i = 0; i < n; i++) {
    const base = i * SAT_STRIDE;
    const pos = propagate(gp[i], timeMs);
    if (pos) {
      const m = lonLatToMercator(pos.lonDeg, pos.latDeg);
      buffer[base] = m.x;
      buffer[base + 1] = m.y;
      buffer[base + 2] = pos.altKm * 1000; // km -> meters (projectTileFor3D)
      buffer[base + 3] = orbitClassCode(pos.altKm);
      shown++;
    } else {
      // No real position -> sentinel slot, keep index alignment, count honestly.
      buffer[base] = 0;
      buffer[base + 1] = 0;
      buffer[base + 2] = 0;
      buffer[base + 3] = SENTINEL_SKIP;
      if (isDeepSpace(gp[i])) deepSpaceSkipped++;
      else invalidSkipped++;
    }
  }
  return { buffer, shown, deepSpaceSkipped, invalidSkipped, total: n };
}

/** Build a `positions` message from a PackResult (pure; shapes the wire format). */
export function positionsMessage(pack: PackResult, timeMs: number): SatPositionsMessage {
  return {
    type: 'positions',
    timeMs,
    stride: SAT_STRIDE,
    total: pack.total,
    shown: pack.shown,
    deepSpaceSkipped: pack.deepSpaceSkipped,
    invalidSkipped: pack.invalidSkipped,
    buf: pack.buffer.buffer as ArrayBuffer,
  };
}

/** Build a `ready` message from a population (pure). */
export function readyMessage(gp: GpRecord[], deps: PropagationDeps = {}): SatReadyMessage {
  const c = analyzePopulation(gp, deps);
  return { type: 'ready', total: c.total, nearEarth: c.nearEarth, deepSpace: c.deepSpace, noElements: c.noElements };
}

// ---------------------------------------------------------------------------
// Worker runtime (thin, testable via injected post + timers + clock)
// ---------------------------------------------------------------------------

export type PostFn = (msg: SatWorkerOutbound, transfer?: Transferable[]) => void;

export interface SatWorkerRuntimeDeps extends PropagationDeps {
  now?: () => number;
  setIntervalFn?: (cb: () => void, ms: number) => unknown;
  clearIntervalFn?: (handle: unknown) => void;
}

export interface SatWorkerRuntime {
  handle(msg: SatWorkerInbound): void;
  /** current satellite count (test/introspection). */
  size(): number;
  /** stop any self-driven interval (idempotent). */
  dispose(): void;
}

/**
 * Create the worker's message handler with all side-effect deps injected.
 * `installWorker()` wires this to a real DedicatedWorkerGlobalScope; tests wire
 * it to a fake post + fake timers + a fixed clock for full determinism.
 */
export function createSatWorker(post: PostFn, deps: SatWorkerRuntimeDeps = {}): SatWorkerRuntime {
  const now = deps.now ?? (() => Date.now());
  const setIntervalFn = deps.setIntervalFn ?? ((cb, ms) => setInterval(cb, ms));
  const clearIntervalFn = deps.clearIntervalFn ?? ((h) => clearInterval(h as ReturnType<typeof setInterval>));
  const propDeps: PropagationDeps = { propagate: deps.propagate, isDeepSpace: deps.isDeepSpace };

  let gp: GpRecord[] = [];
  let timer: unknown = null;

  function stopTimer(): void {
    if (timer != null) {
      clearIntervalFn(timer);
      timer = null;
    }
  }

  function tick(timeMs: number): void {
    const pack = packPositions(gp, timeMs, propDeps);
    const msg = positionsMessage(pack, timeMs);
    post(msg, [msg.buf]);
  }

  return {
    handle(msg: SatWorkerInbound): void {
      switch (msg.type) {
        case 'init':
          gp = Array.isArray(msg.gp) ? msg.gp : [];
          post(readyMessage(gp, propDeps));
          break;
        case 'tick':
          tick(msg.timeMs);
          break;
        case 'start': {
          stopTimer();
          const hz = msg.hz && msg.hz > 0 ? msg.hz : 1;
          const ms = Math.max(1, Math.round(1000 / hz));
          tick(now()); // immediate first frame
          timer = setIntervalFn(() => tick(now()), ms);
          break;
        }
        case 'stop':
          stopTimer();
          break;
        default:
          // Unknown message: ignore (forward-compat).
          break;
      }
    },
    size(): number {
      return gp.length;
    },
    dispose(): void {
      stopTimer();
    },
  };
}

// ---------------------------------------------------------------------------
// Auto-install when running as a real dedicated worker.
// ---------------------------------------------------------------------------

/** Minimal shape of a dedicated worker scope (avoids a webworker lib dependency). */
interface DedicatedScopeLike {
  postMessage(message: unknown, transfer?: Transferable[]): void;
  onmessage: ((ev: { data: SatWorkerInbound }) => void) | null;
}

/**
 * Wire createSatWorker to a dedicated worker scope. Exported for explicit use;
 * normally the module auto-installs (below) when it detects a worker context.
 */
export function installWorker(scope: DedicatedScopeLike, deps: SatWorkerRuntimeDeps = {}): SatWorkerRuntime {
  const runtime = createSatWorker(
    (msg, transfer) => scope.postMessage(msg, transfer),
    deps,
  );
  scope.onmessage = (ev) => runtime.handle(ev.data);
  return runtime;
}

// Robust worker-context detection that stays false in Node (tests) and on the
// browser MAIN thread: only a real DedicatedWorkerGlobalScope self passes.
{
  const g = globalThis as unknown as {
    DedicatedWorkerGlobalScope?: unknown;
    self?: unknown;
  };
  const Ctor = g.DedicatedWorkerGlobalScope as (new () => unknown) | undefined;
  if (typeof Ctor === 'function' && g.self instanceof Ctor) {
    installWorker(g.self as unknown as DedicatedScopeLike);
  }
}
