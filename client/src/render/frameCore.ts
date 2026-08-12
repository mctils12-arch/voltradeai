// frameCore — THE FRAME LOOP (Rendering & Motion Law, Law I).
//
// One requestAnimationFrame loop per view. Everything animated registers
// with it and is stepped in a deterministic priority order. Map events and
// data ticks may set a TARGET; this loop interpolates toward it. Nothing
// visual is recomputed inside a `zoomend`/`moveend`/`move`/`idle` handler —
// that is the single bug behind moon tile fuzz, satellite pulsing, the
// aircraft curtain glitch and layer flicker on zoom.
//
// Design constraints that shaped this file:
//   * NO DOM AT IMPORT TIME. The module loads under `tsx --test` (node, no
//     window) and in the Vite bundle alike; every host binding is lazy and
//     injectable, so the loop is exercisable headless with a manual clock.
//   * FIXED-STEP SPRINGS. A spring that is critically damped in continuous
//     time overshoots under a variable timestep. Springs are therefore
//     integrated at a fixed SPRING_STEP_MS sub-step, never with the raw
//     frame dt. Callbacks still run once per frame.
//   * SLEEPING SPRINGS COST ZERO. Awake springs live in their own set;
//     a settled spring is not iterated at all.
//
// Companion modules: perfHud.ts (the `?perf=1` HUD — build/read it first,
// you cannot verify any of this without it) and zoomInput.ts (wheel/pinch
// normalization into one log-altitude target this loop springs toward).

/** Frame-loop timing constants. Values are fixed by the Rendering & Motion
 *  Law article in CLAUDE.md — changing one is a rule change, not a tune. */
export const FRAME = {
  TARGET_FPS: 60,
  FRAME_BUDGET_MS: 16.7,
  MAX_DT_MS: 100, // clamp — tab resume delivers a huge dt
  SPRING_STEP_MS: 4, // fixed sub-step for spring integration
  MAX_SUBSTEPS: 8, // prevents spiral of death on slow frames
  EPSILON: 1e-4, // below this, snap to target and sleep
} as const;

/** Spring tunings per interaction axis. Damping is ~2*sqrt(stiffness) in
 *  each case, i.e. critically damped: fast, no overshoot, no ring. */
export const SPRING = {
  ZOOM_STIFFNESS: 120,
  ZOOM_DAMPING: 22, // critically damped: 2*sqrt(120) ≈ 21.9
  /** Quiet period after the last zoom input before the zoom TARGET is
   *  considered final. This is NOT a cutoff on the spring — a critically
   *  damped k=120 spring needs ~840ms to reach EPSILON, so snapping at
   *  120ms would leave ~62% of the travel unspent and read as a hard stop.
   *  It is the gesture-settled signal streaming prefetches from (Law II.4:
   *  "prefetch from the target, not the current position"): 120ms of
   *  wheel silence means the destination has stopped moving even though
   *  the camera is still flying toward it. See zoomInput.isSettled(). */
  ZOOM_SETTLE_MS: 120,
  ROTATE_STIFFNESS: 180,
  ROTATE_DAMPING: 27, // 2*sqrt(180) ≈ 26.8
  PAN_STIFFNESS: 150,
  PAN_DAMPING: 24.5,
} as const;

/** Deterministic per-frame execution order. Input reads the world, sim
 *  advances it, streaming decides what to fetch from the freshly-advanced
 *  target, render draws. Any other order makes streaming prefetch from a
 *  stale target — the exact failure Law II.4 exists to prevent. */
export const PRIORITY = {
  INPUT: 0,
  SIM: 1,
  STREAM: 2,
  RENDER: 3,
} as const;

export type Priority = (typeof PRIORITY)[keyof typeof PRIORITY];

/** Ring length for the frame-time histogram: 240 frames ≈ 4s at 60fps —
 *  long enough for a stable p95, short enough to react within a gesture. */
export const FRAME_SAMPLES = 240;

// ── spring (pure, fixed-step) ────────────────────────────────────────────────

export interface SpringOptions {
  stiffness?: number;
  damping?: number;
  /** Below this distance AND velocity the spring snaps and sleeps. */
  epsilon?: number;
}

/**
 * A damped spring toward a target. Integrated ONLY at a fixed sub-step
 * (see `step`); never hand it a raw frame dt.
 *
 * Semi-implicit (symplectic) Euler:
 *   a = -k*(x - target) - c*v ;  v += a*h ;  x += v*h
 * Stable while h*sqrt(k) << 2 — at k=180 and h=4ms that is 0.054, three
 * orders of margin.
 */
export class Spring {
  value: number;
  target: number;
  velocity = 0;
  asleep = true;

  readonly stiffness: number;
  readonly damping: number;
  readonly epsilon: number;

  /** Set by FrameLoop.addSpring. Waking is pushed to the loop (O(1)) rather
   *  than discovered by scanning every spring each frame — that scan is
   *  exactly the cost "sleeping springs cost zero" forbids. */
  _loop: { _wake(s: Spring): void } | null = null;

  constructor(initial = 0, opts: SpringOptions = {}) {
    this.value = initial;
    this.target = initial;
    this.stiffness = opts.stiffness ?? SPRING.ZOOM_STIFFNESS;
    this.damping = opts.damping ?? SPRING.ZOOM_DAMPING;
    this.epsilon = opts.epsilon ?? FRAME.EPSILON;
  }

  /** Set the destination. Wakes the spring iff the target actually moved —
   *  re-setting the same target on every map event must not resurrect a
   *  settled spring, or "sleeping springs cost zero" is a lie. */
  setTarget(target: number): void {
    if (!Number.isFinite(target)) return;
    if (target === this.target) return;
    this.target = target;
    if (this.asleep) {
      this.asleep = false;
      this._loop?._wake(this);
    }
  }

  /** Teleport: value and target together, asleep. Used on init, on a hard
   *  camera cut, and on WebGL context restore — never for animation. */
  snap(value: number): void {
    if (!Number.isFinite(value)) return;
    this.value = value;
    this.target = value;
    this.velocity = 0;
    this.asleep = true;
  }

  /** Distance still to travel. */
  get distance(): number {
    return this.target - this.value;
  }

  /**
   * One FIXED sub-step. `hSec` is SPRING_STEP_MS/1000 — the loop owns the
   * accumulator that decides how many of these run. Returns true while the
   * spring is still awake after the step.
   */
  step(hSec: number): boolean {
    if (this.asleep) return false;
    const a = -this.stiffness * (this.value - this.target) - this.damping * this.velocity;
    this.velocity += a * hSec;
    this.value += this.velocity * hSec;
    if (Math.abs(this.target - this.value) < this.epsilon && Math.abs(this.velocity) < this.epsilon) {
      this.value = this.target;
      this.velocity = 0;
      this.asleep = true;
      return false;
    }
    return true;
  }
}

/** Frame-rate-independent lerp toward a target: the fraction of the
 *  remaining distance covered per fixed sub-step, expressed as a half-life
 *  so the feel does not change with the step size. */
export function lerpTowards(value: number, target: number, halfLifeMs: number, hMs: number): number {
  if (!(halfLifeMs > 0)) return target;
  const k = Math.pow(0.5, hMs / halfLifeMs);
  const next = target + (value - target) * k;
  return Math.abs(target - next) < FRAME.EPSILON ? target : next;
}

// ── accumulator (pure) ───────────────────────────────────────────────────────

export interface SubstepPlan {
  /** How many fixed sub-steps to run this frame. */
  steps: number;
  /** Accumulator carried into the next frame, in ms. */
  carry: number;
  /** True when MAX_SUBSTEPS clipped the plan and the remainder was dropped. */
  clipped: boolean;
}

/**
 * Decide the fixed sub-steps for one frame.
 *
 * Two guards, both load-bearing:
 *  1. `dtMs` is clamped by the caller to MAX_DT_MS — a tab resume delivers
 *     a multi-second dt and an unclamped accumulator would try to catch up.
 *  2. When the plan hits MAX_SUBSTEPS the carry is DROPPED, not banked.
 *     Banking it means the accumulator only ever grows on a slow device and
 *     every subsequent frame runs the full 8 sub-steps forever — the spiral
 *     of death this cap exists to prevent.
 */
export function planSubsteps(
  accumulatorMs: number,
  dtMs: number,
  stepMs = FRAME.SPRING_STEP_MS,
  maxSteps = FRAME.MAX_SUBSTEPS,
): SubstepPlan {
  const acc = accumulatorMs + Math.max(0, dtMs);
  let steps = Math.floor(acc / stepMs);
  if (steps <= 0) return { steps: 0, carry: acc, clipped: false };
  if (steps > maxSteps) return { steps: maxSteps, carry: 0, clipped: true };
  return { steps, carry: acc - steps * stepMs, clipped: false };
}

// ── frame-time statistics (pure) ─────────────────────────────────────────────

/** Nearest-rank percentile over an ALREADY SORTED ascending array. */
export function percentile(sortedAsc: readonly number[], p: number): number {
  if (!sortedAsc.length) return 0;
  const rank = Math.ceil((p / 100) * sortedAsc.length);
  const i = Math.min(sortedAsc.length - 1, Math.max(0, rank - 1));
  return sortedAsc[i];
}

export interface FrameStats {
  /** Frames executed since the loop started. */
  frames: number;
  /** Samples in the histogram. */
  samples: number;
  p50: number;
  p95: number;
  worst: number;
  /** Frames in the window that exceeded FRAME_BUDGET_MS. */
  overBudget: number;
}

// ── host bindings (injectable, so the loop runs headless) ────────────────────

export interface Scheduler {
  request(cb: (tsMs: number) => void): number;
  cancel(handle: number): void;
}

export interface VisibilitySource {
  isHidden(): boolean;
  /** Subscribe to visibility changes; returns an unsubscribe. */
  subscribe(fn: () => void): () => void;
}

export interface FrameHost {
  now(): number;
  scheduler: Scheduler;
  visibility: VisibilitySource;
}

const NOOP_VISIBILITY: VisibilitySource = {
  isHidden: () => false,
  subscribe: () => () => {},
};

/** Host bound to the browser, or a headless stand-in when there is no DOM.
 *  Resolved lazily (on first start) so importing this module in node is
 *  free of side effects. */
export function defaultHost(): FrameHost {
  const g = globalThis as unknown as {
    performance?: { now(): number };
    requestAnimationFrame?: (cb: (t: number) => void) => number;
    cancelAnimationFrame?: (h: number) => void;
    document?: { hidden: boolean; addEventListener: Function; removeEventListener: Function };
  };
  const perf = g.performance;
  const now = perf && typeof perf.now === "function" ? () => perf.now() : () => Date.now();

  const raf = g.requestAnimationFrame;
  const caf = g.cancelAnimationFrame;
  const scheduler: Scheduler =
    typeof raf === "function" && typeof caf === "function"
      ? { request: (cb) => raf.call(g, cb), cancel: (h) => caf.call(g, h) }
      : {
          // Headless fallback: a fixed-cadence timer. Never used in the
          // browser; keeps node tests and SSR from throwing.
          request: (cb) => setTimeout(() => cb(now()), 1000 / FRAME.TARGET_FPS) as unknown as number,
          cancel: (h) => clearTimeout(h as unknown as ReturnType<typeof setTimeout>),
        };

  const doc = g.document;
  const visibility: VisibilitySource =
    doc && typeof doc.addEventListener === "function"
      ? {
          isHidden: () => !!doc.hidden,
          subscribe: (fn) => {
            doc.addEventListener("visibilitychange", fn);
            return () => doc.removeEventListener("visibilitychange", fn);
          },
        }
      : NOOP_VISIBILITY;

  return { now, scheduler, visibility };
}

// ── the loop ─────────────────────────────────────────────────────────────────

/** `(dtMs, nowMs)` — dt is CLAMPED frame delta, now is the frame's single
 *  sampled timestamp. Never call performance.now() inside a callback: one
 *  clock per frame, or interpolating layers desync from each other. */
export type FrameCallback = (dtMs: number, nowMs: number) => void;

interface Registration {
  fn: FrameCallback;
  priority: Priority;
  /** Monotonic insertion id — ties within a priority resolve by
   *  registration order, so the frame is byte-for-byte deterministic. */
  seq: number;
  label: string;
}

export interface RegisterOptions {
  /** Shown in the `?perf=1` HUD and in error reports. */
  label?: string;
}

export class FrameLoop {
  private host: FrameHost | null = null;
  private readonly hostFactory: () => FrameHost;

  private regs: Registration[] = [];
  private seq = 0;
  private readonly springs = new Set<Spring>();
  private readonly awake = new Set<Spring>();

  private handle: number | null = null;
  private unsubscribeVisibility: (() => void) | null = null;
  private running = false;
  private paused = false;

  private accumulator = 0;
  private lastTs = 0;
  private nowMs = 0;
  private dtMs = 0;
  private frames = 0;

  private readonly samples: number[] = [];
  private sampleHead = 0;

  /** Errors thrown by callbacks are caught so one broken layer cannot stop
   *  the whole view. Counted here and surfaced in the HUD. */
  errorCount = 0;
  lastError: unknown = null;

  constructor(hostFactory: () => FrameHost = defaultHost) {
    this.hostFactory = hostFactory;
  }

  /** The frame's single sampled timestamp, monotonic ms. Real time — NOT
   *  the sum of clamped dts (those diverge across a stall, deliberately:
   *  interpolating layers need wall clock, integrators need clamped dt). */
  now(): number {
    return this.nowMs;
  }

  /** Last frame's delta, clamped to FRAME.MAX_DT_MS. */
  get dt(): number {
    return this.dtMs;
  }

  get frameCount(): number {
    return this.frames;
  }

  get isRunning(): boolean {
    return this.running;
  }

  get isPaused(): boolean {
    return this.paused;
  }

  get registrationCount(): number {
    return this.regs.length;
  }

  /** Springs currently being integrated. Settled springs are not counted
   *  and not touched. */
  get awakeSpringCount(): number {
    return this.awake.size;
  }

  /**
   * Register a per-frame callback. Returns an unregister function — call it
   * in the layer's `dispose()` (Law IV), or the loop keeps the closure and
   * everything it captures alive forever.
   */
  register(fn: FrameCallback, priority: Priority = PRIORITY.RENDER, opts: RegisterOptions = {}): () => void {
    const reg: Registration = { fn, priority, seq: this.seq++, label: opts.label ?? "" };
    this.regs.push(reg);
    this.sortRegistrations();
    this.start();
    let removed = false;
    return () => {
      if (removed) return;
      removed = true;
      const i = this.regs.indexOf(reg);
      if (i >= 0) this.regs.splice(i, 1);
    };
  }

  private sortRegistrations(): void {
    this.regs.sort((a, b) => a.priority - b.priority || a.seq - b.seq);
  }

  /**
   * Hand a spring to the loop for fixed-step integration. Returns a
   * detach function. A spring may be added once; adding twice is a no-op.
   */
  addSpring(spring: Spring): () => void {
    this.springs.add(spring);
    spring._loop = this;
    if (!spring.asleep) this.awake.add(spring);
    this.start();
    return () => {
      this.springs.delete(spring);
      this.awake.delete(spring);
      if (spring._loop === this) spring._loop = null;
    };
  }

  /** Wake hook called by Spring.setTarget. Internal. */
  _wake(spring: Spring): void {
    if (this.springs.has(spring)) this.awake.add(spring);
  }

  /** Convenience: create a spring already attached to this loop. */
  createSpring(initial = 0, opts: SpringOptions = {}): { spring: Spring; detach: () => void } {
    const spring = new Spring(initial, opts);
    return { spring, detach: this.addSpring(spring) };
  }

  start(): void {
    if (this.running) return;
    this.running = true;
    const host = (this.host ??= this.hostFactory());
    this.unsubscribeVisibility = host.visibility.subscribe(() => this.onVisibilityChange());
    if (host.visibility.isHidden()) {
      this.paused = true;
      return;
    }
    this.resume();
  }

  stop(): void {
    this.running = false;
    this.paused = false;
    if (this.handle !== null && this.host) {
      this.host.scheduler.cancel(this.handle);
      this.handle = null;
    }
    this.unsubscribeVisibility?.();
    this.unsubscribeVisibility = null;
  }

  /** Full teardown: stop the loop, drop every registration and spring.
   *  Used by the view's unmount path and by the tests between cases. */
  dispose(): void {
    this.stop();
    this.regs = [];
    for (const s of this.springs) if (s._loop === this) s._loop = null;
    this.springs.clear();
    this.awake.clear();
    this.samples.length = 0;
    this.sampleHead = 0;
    this.accumulator = 0;
    this.frames = 0;
    this.errorCount = 0;
    this.lastError = null;
    this.host = null;
  }

  private onVisibilityChange(): void {
    if (!this.running || !this.host) return;
    if (this.host.visibility.isHidden()) {
      this.paused = true;
      if (this.handle !== null) {
        this.host.scheduler.cancel(this.handle);
        this.handle = null;
      }
      return;
    }
    if (this.paused) this.resume();
  }

  private resume(): void {
    if (!this.host) return;
    this.paused = false;
    // On resume the accumulator is RESET, not caught up. A hidden tab owes
    // the simulation nothing: replaying 30s of springs would fling the
    // camera across the sky on the first visible frame.
    this.accumulator = 0;
    this.lastTs = 0;
    this.schedule();
  }

  private schedule(): void {
    if (!this.host || !this.running || this.paused || this.handle !== null) return;
    this.handle = this.host.scheduler.request((ts) => {
      this.handle = null;
      this.tick(ts);
      this.schedule();
    });
  }

  /**
   * One frame. Exposed for the headless harness (drive it with a manual
   * scheduler); production always reaches it through `schedule()`.
   */
  tick(tsMs: number): void {
    if (!this.host) this.host = this.hostFactory();
    const host = this.host;
    const started = host.now();

    const raw = this.lastTs === 0 ? 0 : tsMs - this.lastTs;
    this.lastTs = tsMs;
    // Clamp BEFORE accumulating (Law I / known failure mode "dt explosion
    // after tab resume"). A negative delta means the caller handed us a
    // non-monotonic timestamp; treat it as zero rather than integrating
    // backwards.
    this.dtMs = Math.min(FRAME.MAX_DT_MS, Math.max(0, raw));
    this.nowMs = tsMs;
    this.frames++;

    this.integrateSprings();

    for (const reg of this.regs) {
      try {
        reg.fn(this.dtMs, this.nowMs);
      } catch (err) {
        this.errorCount++;
        this.lastError = err;
        // One layer throwing must never take the view down. Reported in the
        // HUD and to the console once per occurrence.
        // eslint-disable-next-line no-console
        console.error(`[frameCore] callback${reg.label ? ` '${reg.label}'` : ""} threw`, err);
      }
    }

    this.recordSample(host.now() - started);
  }

  private integrateSprings(): void {
    // Nothing awake costs one Set.size read. The accumulator is zeroed while
    // idle: otherwise it banks real time during a still camera and spends it
    // in one burst on the first frame after the next input.
    if (this.awake.size === 0) {
      this.accumulator = 0;
      return;
    }

    const plan = planSubsteps(this.accumulator, this.dtMs);
    this.accumulator = plan.carry;
    if (plan.steps === 0) return;

    const h = FRAME.SPRING_STEP_MS / 1000;
    for (const spring of this.awake) {
      let alive = true;
      for (let i = 0; i < plan.steps && alive; i++) alive = spring.step(h);
      if (!alive) this.awake.delete(spring);
    }
  }

  private recordSample(ms: number): void {
    if (this.samples.length < FRAME_SAMPLES) {
      this.samples.push(ms);
    } else {
      this.samples[this.sampleHead] = ms;
      this.sampleHead = (this.sampleHead + 1) % FRAME_SAMPLES;
    }
  }

  stats(): FrameStats {
    const sorted = this.samples.slice().sort((a, b) => a - b);
    let overBudget = 0;
    for (const s of sorted) if (s > FRAME.FRAME_BUDGET_MS) overBudget++;
    return {
      frames: this.frames,
      samples: sorted.length,
      p50: percentile(sorted, 50),
      p95: percentile(sorted, 95),
      worst: sorted.length ? sorted[sorted.length - 1] : 0,
      overBudget,
    };
  }

  /** Registered labels in execution order — the HUD renders this so a slow
   *  frame can be attributed to a layer instead of guessed at. */
  registrations(): { label: string; priority: Priority }[] {
    return this.regs.map((r) => ({ label: r.label, priority: r.priority }));
  }
}

// ── the singleton ────────────────────────────────────────────────────────────

let singleton: FrameLoop | null = null;

/** The one loop per view. Constructed on first use, never at import time. */
export function frameCore(): FrameLoop {
  return (singleton ??= new FrameLoop());
}

/** Replace the singleton — tests only, and the view's remount path. */
export function __setFrameCore(loop: FrameLoop | null): void {
  singleton = loop;
}
