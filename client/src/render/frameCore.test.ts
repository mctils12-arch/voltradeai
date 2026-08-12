// frameCore.test.ts — the Frame Law, mechanically.
//
// Every "known failure mode" in the rendering work order that belongs to
// the loop has a case here: dt explosion on tab resume, spring overshoot
// under a variable timestep, the sub-step spiral of death, sleeping springs
// that never truly sleep, and non-deterministic callback order.

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  FRAME,
  FrameLoop,
  PRIORITY,
  SPRING,
  Spring,
  frameCore,
  lerpTowards,
  percentile,
  planSubsteps,
  __setFrameCore,
  type FrameHost,
  type Scheduler,
} from "./frameCore.ts";

// ── a manual host: one clock, one hand-cranked scheduler ────────────────────

interface Harness {
  host: FrameHost;
  /** Advance wall clock by ms and deliver exactly one frame. */
  frame(ms: number): void;
  /** Advance the clock without delivering a frame (a stall). */
  idle(ms: number): void;
  setHidden(hidden: boolean): void;
  clockMs: () => number;
  pendingFrames: () => number;
}

function harness(): Harness {
  let clock = 1000;
  let nextHandle = 1;
  const queue = new Map<number, (ts: number) => void>();
  let hidden = false;
  const subs = new Set<() => void>();

  const scheduler: Scheduler = {
    request(cb) {
      const h = nextHandle++;
      queue.set(h, cb);
      return h;
    },
    cancel(h) {
      queue.delete(h);
    },
  };

  const host: FrameHost = {
    now: () => clock,
    scheduler,
    visibility: {
      isHidden: () => hidden,
      subscribe: (fn) => {
        subs.add(fn);
        return () => subs.delete(fn);
      },
    },
  };

  return {
    host,
    clockMs: () => clock,
    pendingFrames: () => queue.size,
    idle(ms) {
      clock += ms;
    },
    frame(ms) {
      clock += ms;
      const entries = [...queue.entries()];
      queue.clear();
      for (const [, cb] of entries) cb(clock);
    },
    setHidden(h) {
      hidden = h;
      for (const fn of [...subs]) fn();
    },
  };
}

// ── constants are law, not taste ────────────────────────────────────────────

test("FRAME/SPRING constants match the Rendering & Motion Law article", () => {
  assert.equal(FRAME.TARGET_FPS, 60);
  assert.equal(FRAME.FRAME_BUDGET_MS, 16.7);
  assert.equal(FRAME.MAX_DT_MS, 100);
  assert.equal(FRAME.SPRING_STEP_MS, 4);
  assert.equal(FRAME.MAX_SUBSTEPS, 8);
  assert.equal(FRAME.EPSILON, 1e-4);

  assert.equal(SPRING.ZOOM_STIFFNESS, 120);
  assert.equal(SPRING.ZOOM_DAMPING, 22);
  assert.equal(SPRING.ZOOM_SETTLE_MS, 120);
  assert.equal(SPRING.ROTATE_STIFFNESS, 180);
  assert.equal(SPRING.ROTATE_DAMPING, 27);
  assert.equal(SPRING.PAN_STIFFNESS, 150);
  assert.equal(SPRING.PAN_DAMPING, 24.5);

  assert.equal(PRIORITY.INPUT, 0);
  assert.equal(PRIORITY.SIM, 1);
  assert.equal(PRIORITY.STREAM, 2);
  assert.equal(PRIORITY.RENDER, 3);
});

test("every spring tuning is at or above critical damping (no ring, no overshoot)", () => {
  const pairs: [number, number][] = [
    [SPRING.ZOOM_STIFFNESS, SPRING.ZOOM_DAMPING],
    [SPRING.ROTATE_STIFFNESS, SPRING.ROTATE_DAMPING],
    [SPRING.PAN_STIFFNESS, SPRING.PAN_DAMPING],
  ];
  for (const [k, c] of pairs) {
    // Critical damping is c = 2*sqrt(k) for unit mass; below it the spring
    // rings, which reads as the camera bouncing at the end of a zoom.
    assert.ok(c >= 2 * Math.sqrt(k) * 0.995, `damping ${c} under critical for stiffness ${k}`);
  }
});

// ── the accumulator ─────────────────────────────────────────────────────────

test("planSubsteps: normal frame consumes whole steps and carries the remainder", () => {
  const p = planSubsteps(0, 16.7);
  assert.equal(p.steps, 4); // 16.7 / 4 = 4 whole steps
  assert.ok(Math.abs(p.carry - 0.7) < 1e-9);
  assert.equal(p.clipped, false);
});

test("planSubsteps: sub-frame dt runs no steps and banks the time", () => {
  const p = planSubsteps(0, 3);
  assert.equal(p.steps, 0);
  assert.equal(p.carry, 3);
});

test("planSubsteps: caps at MAX_SUBSTEPS and DROPS the remainder", () => {
  // A 2s stall must not run 500 integrations, and must not bank 1.968s of
  // debt that pins every later frame at 8 sub-steps forever.
  const p = planSubsteps(0, 2000);
  assert.equal(p.steps, FRAME.MAX_SUBSTEPS);
  assert.equal(p.carry, 0, "clipped frames must drop the remainder, not bank it");
  assert.equal(p.clipped, true);
});

test("planSubsteps: repeated slow frames do not spiral", () => {
  let acc = 0;
  let total = 0;
  for (let i = 0; i < 200; i++) {
    const p = planSubsteps(acc, FRAME.MAX_DT_MS);
    acc = p.carry;
    total += p.steps;
  }
  assert.equal(total, 200 * FRAME.MAX_SUBSTEPS, "steps must stay capped every frame");
  assert.ok(acc < FRAME.SPRING_STEP_MS * FRAME.MAX_SUBSTEPS, "accumulator must not grow without bound");
});

// ── springs ─────────────────────────────────────────────────────────────────

test("Spring: starts asleep, wakes only on a real target change", () => {
  const s = new Spring(5);
  assert.equal(s.asleep, true);
  s.setTarget(5);
  assert.equal(s.asleep, true, "re-setting the same target must not wake a settled spring");
  s.setTarget(6);
  assert.equal(s.asleep, false);
});

test("Spring: converges to the target and sleeps at EPSILON", () => {
  const s = new Spring(0, { stiffness: SPRING.ZOOM_STIFFNESS, damping: SPRING.ZOOM_DAMPING });
  s.setTarget(10);
  const h = FRAME.SPRING_STEP_MS / 1000;
  let steps = 0;
  while (!s.asleep && steps < 20000) {
    s.step(h);
    steps++;
  }
  assert.equal(s.asleep, true, "spring must reach sleep, not creep forever");
  assert.equal(s.value, 10, "sleeping snaps exactly to target");
  assert.equal(s.velocity, 0);
  // ~840ms of settle at k=120 → ~210 sub-steps. Generous bound; the point
  // is that it terminates in a human-scale time, not 20k steps.
  assert.ok(steps < 600, `settled in ${steps} sub-steps`);
});

test("Spring: critically damped tuning never overshoots the target", () => {
  const s = new Spring(0, { stiffness: SPRING.ZOOM_STIFFNESS, damping: SPRING.ZOOM_DAMPING });
  s.setTarget(1);
  const h = FRAME.SPRING_STEP_MS / 1000;
  let maxValue = 0;
  for (let i = 0; i < 1000 && !s.asleep; i++) {
    s.step(h);
    maxValue = Math.max(maxValue, s.value);
  }
  assert.ok(maxValue <= 1 + 1e-9, `overshot to ${maxValue}`);
});

test("Spring: fixed sub-stepping is what keeps it stable — raw 100ms dt would not be", () => {
  // The failure mode the fixed step exists to prevent, demonstrated: the
  // SAME spring integrated once at a clamped 100ms frame dt diverges.
  const bad = new Spring(0, { stiffness: SPRING.ZOOM_STIFFNESS, damping: SPRING.ZOOM_DAMPING });
  bad.setTarget(1);
  for (let i = 0; i < 20; i++) bad.step(FRAME.MAX_DT_MS / 1000);
  assert.ok(Math.abs(bad.value) > 10, "raw-dt integration must blow up (this is why we sub-step)");

  const good = new Spring(0, { stiffness: SPRING.ZOOM_STIFFNESS, damping: SPRING.ZOOM_DAMPING });
  good.setTarget(1);
  // Same 2s of simulated time, delivered as fixed sub-steps.
  for (let i = 0; i < (20 * FRAME.MAX_DT_MS) / FRAME.SPRING_STEP_MS; i++) good.step(FRAME.SPRING_STEP_MS / 1000);
  assert.ok(Math.abs(good.value - 1) < 1e-6, `fixed-step landed at ${good.value}`);
});

test("Spring: snap teleports and sleeps; non-finite input is ignored", () => {
  const s = new Spring(0);
  s.setTarget(4);
  s.snap(7);
  assert.equal(s.value, 7);
  assert.equal(s.target, 7);
  assert.equal(s.asleep, true);
  s.setTarget(NaN);
  assert.equal(s.target, 7);
  s.snap(Infinity);
  assert.equal(s.value, 7);
});

test("lerpTowards is frame-rate independent (half-life holds across step sizes)", () => {
  // Same simulated time, different step sizes, same result — the property a
  // naive `v += (t-v)*0.1` per frame does NOT have.
  const run = (hMs: number) => {
    let v = 0;
    for (let t = 0; t < 400; t += hMs) v = lerpTowards(v, 1, 100, hMs);
    return v;
  };
  const a = run(4);
  const b = run(16.7);
  assert.ok(Math.abs(a - b) < 0.02, `${a} vs ${b} — lerp must not depend on frame rate`);
  // 400ms at a 100ms half-life ≈ 1 - 2^-4 = 0.9375
  assert.ok(Math.abs(a - 0.9375) < 0.02);
});

// ── the loop ────────────────────────────────────────────────────────────────

test("FrameLoop: callbacks run in priority order, ties by registration order", () => {
  const h = harness();
  const loop = new FrameLoop(() => h.host);
  const order: string[] = [];
  loop.register(() => order.push("render-a"), PRIORITY.RENDER);
  loop.register(() => order.push("input"), PRIORITY.INPUT);
  loop.register(() => order.push("render-b"), PRIORITY.RENDER);
  loop.register(() => order.push("stream"), PRIORITY.STREAM);
  loop.register(() => order.push("sim"), PRIORITY.SIM);

  h.frame(16);
  assert.deepEqual(order, ["input", "sim", "stream", "render-a", "render-b"]);

  order.length = 0;
  h.frame(16);
  assert.deepEqual(order, ["input", "sim", "stream", "render-a", "render-b"], "order must be identical every frame");
  loop.dispose();
});

test("FrameLoop: STREAM runs after SIM so prefetch sees the advanced target", () => {
  // Law II.4 in execution-order form. If STREAM ran first it would select
  // tiles from last frame's target — the "fuzzy then snaps sharp" bug.
  const h = harness();
  const loop = new FrameLoop(() => h.host);
  const { spring } = loop.createSpring(0);
  spring.setTarget(1);
  let seenByStream = -1;
  let seenBySim = -1;
  loop.register(() => {
    seenBySim = spring.value;
  }, PRIORITY.SIM);
  loop.register(() => {
    seenByStream = spring.value;
  }, PRIORITY.STREAM);
  h.frame(16); // dt 0 — nothing integrates on the very first frame
  h.frame(16);
  assert.ok(seenByStream > 0, "stream must observe this frame's integrated value, not zero");
  assert.equal(seenBySim, seenByStream, "sim and stream see the same already-integrated value");
  loop.dispose();
});

test("FrameLoop: the first frame has dt 0 and integrates nothing", () => {
  // Not a wart — there is no previous timestamp to subtract, and inventing
  // one (e.g. assuming 16.7ms) would make the very first frame after every
  // resume advance the sim by a delta that never elapsed.
  const h = harness();
  const loop = new FrameLoop(() => h.host);
  const { spring } = loop.createSpring(0);
  spring.setTarget(1);
  h.frame(16);
  assert.equal(loop.dt, 0);
  assert.equal(spring.value, 0);
  h.frame(16);
  assert.equal(loop.dt, 16);
  assert.ok(spring.value > 0);
  loop.dispose();
});

test("FrameLoop: dt is clamped to MAX_DT_MS on tab resume", () => {
  const h = harness();
  const loop = new FrameLoop(() => h.host);
  const seen: number[] = [];
  loop.register((dt) => seen.push(dt));
  h.frame(16);
  h.frame(5000); // the tab was in the background
  assert.equal(seen[0], 0, "first frame has no previous timestamp — dt 0");
  assert.equal(seen[1], FRAME.MAX_DT_MS, "a 5s gap must arrive clamped");
  loop.dispose();
});

test("FrameLoop: a non-monotonic timestamp never integrates backwards", () => {
  const h = harness();
  const loop = new FrameLoop(() => h.host);
  const seen: number[] = [];
  loop.register((dt) => seen.push(dt));
  h.frame(16);
  loop.tick(0); // pathological: timestamp went backwards
  assert.equal(seen[1], 0);
  loop.dispose();
});

test("FrameLoop: now() is the frame's single sampled clock", () => {
  const h = harness();
  const loop = new FrameLoop(() => h.host);
  const samples: number[] = [];
  loop.register(() => samples.push(loop.now()), PRIORITY.INPUT);
  loop.register(() => samples.push(loop.now()), PRIORITY.RENDER);
  h.frame(16);
  assert.equal(samples[0], samples[1], "every callback in a frame must see the same now()");
  const first = samples[0];
  h.frame(16);
  assert.ok(loop.now() > first, "now() is monotonic across frames");
  loop.dispose();
});

test("FrameLoop: hidden tab pauses, resume resets the accumulator instead of catching up", () => {
  const h = harness();
  const loop = new FrameLoop(() => h.host);
  const { spring } = loop.createSpring(0);
  let frames = 0;
  loop.register(() => frames++);

  h.frame(16);
  assert.equal(frames, 1);

  h.setHidden(true);
  assert.equal(loop.isPaused, true);
  assert.equal(h.pendingFrames(), 0, "no frame may be scheduled while hidden");
  h.idle(30_000);
  h.frame(0);
  assert.equal(frames, 1, "no frames run while hidden");

  spring.setTarget(1);
  h.setHidden(false);
  assert.equal(loop.isPaused, false);
  h.frame(16);
  assert.equal(frames, 2);
  // If the 30s had been banked, one resumed frame would run the full
  // MAX_SUBSTEPS burst and fling the value most of the way to target.
  assert.ok(spring.value < 0.35, `resumed frame advanced too far (${spring.value}) — accumulator was not reset`);
  loop.dispose();
});

test("FrameLoop: a sleeping spring is not integrated and costs nothing", () => {
  const h = harness();
  const loop = new FrameLoop(() => h.host);
  const { spring } = loop.createSpring(3);
  h.frame(16);
  assert.equal(loop.awakeSpringCount, 0);
  assert.equal(spring.value, 3);

  spring.setTarget(4);
  assert.equal(loop.awakeSpringCount, 1, "setTarget must wake the spring in the loop, not just on the spring");
  for (let i = 0; i < 400 && loop.awakeSpringCount > 0; i++) h.frame(16);
  assert.equal(loop.awakeSpringCount, 0, "a settled spring must leave the awake set");
  assert.equal(spring.value, 4);
  loop.dispose();
});

test("FrameLoop: unregister removes the callback and releases the closure", () => {
  const h = harness();
  const loop = new FrameLoop(() => h.host);
  let n = 0;
  const off = loop.register(() => n++);
  h.frame(16);
  assert.equal(n, 1);
  off();
  assert.equal(loop.registrationCount, 0);
  h.frame(16);
  assert.equal(n, 1);
  off(); // idempotent
  assert.equal(loop.registrationCount, 0);
  loop.dispose();
});

test("FrameLoop: a throwing callback is contained, counted, and does not stop the frame", () => {
  const h = harness();
  const loop = new FrameLoop(() => h.host);
  const errs = console.error;
  console.error = () => {};
  try {
    let after = 0;
    loop.register(() => {
      throw new Error("layer blew up");
    }, PRIORITY.SIM);
    loop.register(() => after++, PRIORITY.RENDER);
    h.frame(16);
    h.frame(16);
    assert.equal(after, 2, "one broken layer must not take the view down");
    assert.equal(loop.errorCount, 2);
  } finally {
    console.error = errs;
  }
  loop.dispose();
});

test("FrameLoop: detaching a spring stops it being integrated", () => {
  const h = harness();
  const loop = new FrameLoop(() => h.host);
  const { spring, detach } = loop.createSpring(0);
  spring.setTarget(1);
  h.frame(16); // dt 0
  h.frame(16);
  const moved = spring.value;
  assert.ok(moved > 0);
  detach();
  h.frame(16);
  assert.equal(spring.value, moved, "a detached spring must not advance");
  assert.equal(spring._loop, null);
  loop.dispose();
});

test("FrameLoop: dispose is total teardown (Law IV)", () => {
  const h = harness();
  const loop = new FrameLoop(() => h.host);
  const { spring } = loop.createSpring(0);
  loop.register(() => {});
  h.frame(16);
  loop.dispose();
  assert.equal(loop.registrationCount, 0);
  assert.equal(loop.awakeSpringCount, 0);
  assert.equal(loop.isRunning, false);
  assert.equal(spring._loop, null);
  assert.equal(h.pendingFrames(), 0, "no rAF may survive dispose");
});

test("FrameLoop: starting while already hidden does not schedule a frame", () => {
  const h = harness();
  h.setHidden(true);
  const loop = new FrameLoop(() => h.host);
  loop.register(() => {});
  assert.equal(loop.isPaused, true);
  assert.equal(h.pendingFrames(), 0);
  loop.dispose();
});

// ── statistics ──────────────────────────────────────────────────────────────

test("percentile: nearest-rank on a sorted array", () => {
  const s = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
  assert.equal(percentile(s, 50), 5);
  assert.equal(percentile(s, 95), 10);
  assert.equal(percentile(s, 100), 10);
  assert.equal(percentile(s, 0), 1);
  assert.equal(percentile([], 95), 0);
  assert.equal(percentile([42], 95), 42);
});

test("FrameLoop.stats reports p50/p95 and over-budget frames", () => {
  const h = harness();
  const loop = new FrameLoop(() => h.host);
  // Burn a known amount of host time inside the frame by advancing the
  // harness clock from within the callback.
  let burn = 1;
  loop.register(() => h.idle(burn));
  for (let i = 0; i < 19; i++) h.frame(16);
  burn = 40;
  h.frame(16);
  const st = loop.stats();
  assert.equal(st.samples, 20);
  assert.equal(st.frames, 20);
  assert.ok(st.p50 <= 2, `p50 ${st.p50}`);
  assert.equal(st.worst, 40);
  assert.equal(st.overBudget, 1, "exactly the one 40ms frame is over the 16.7ms budget");
  loop.dispose();
});

// ── the singleton ───────────────────────────────────────────────────────────

test("frameCore() is a lazily-constructed singleton, replaceable for tests", () => {
  __setFrameCore(null);
  const a = frameCore();
  const b = frameCore();
  assert.equal(a, b);
  const h = harness();
  const injected = new FrameLoop(() => h.host);
  __setFrameCore(injected);
  assert.equal(frameCore(), injected);
  injected.dispose();
  __setFrameCore(null);
});

test("importing frameCore touches no DOM (it loads under node with no window)", () => {
  // If this test file ran at all, the import above already proved it. Assert
  // the guard explicitly so a future `document.` at module scope fails here
  // rather than at Vite SSR time.
  assert.equal(typeof (globalThis as { document?: unknown }).document, "undefined");
});
