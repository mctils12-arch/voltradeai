// zoomInput.test.ts — one zoom model, verified.
//
// The properties that matter are the ones a human notices on a phone:
// a notch feels the same at 30,000km and at 100m; a pinch and a wheel move
// the camera by the same amount for the same gesture; the point under your
// finger stays under your finger; and input never assigns a position.

import { test } from "node:test";
import assert from "node:assert/strict";

import { FRAME, SPRING, Spring } from "./frameCore.ts";
import {
  DELTA_MODE,
  LINE_HEIGHT_PX,
  LOG_ALT_MAX,
  LOG_ALT_MIN,
  MAX_WHEEL_DELTA_PX,
  ZOOM,
  ZoomInput,
  altToLog,
  anchorScale,
  anchorScale2,
  centroid,
  clampAltKm,
  clampLogAlt,
  logToAlt,
  pinchToLogDelta,
  spread,
  wheelToLogDelta,
} from "./zoomInput.ts";

test("ZOOM constants match the Rendering & Motion Law article", () => {
  assert.equal(ZOOM.WHEEL_RATE, 1 / 450);
  assert.equal(ZOOM.PINCH_RATE, 1.0);
  assert.equal(ZOOM.ALT_MIN_KM, 0.1);
  assert.equal(ZOOM.ALT_MAX_KM, 30000);
  assert.equal(ZOOM.MOON_RADIUS_KM, 1737.4);
  assert.equal(ZOOM.EARTH_RADIUS_KM, 6371.0);
});

// ── log-altitude space ──────────────────────────────────────────────────────

test("altToLog / logToAlt round-trip across the whole envelope", () => {
  for (const km of [0.1, 1, 10, 100, 1000, 6371, 30000]) {
    assert.ok(Math.abs(logToAlt(altToLog(km)) - km) < 1e-9, `round trip failed at ${km}km`);
  }
});

test("the envelope clamps at both ends, in both spaces", () => {
  assert.equal(clampAltKm(0), ZOOM.ALT_MIN_KM);
  assert.equal(clampAltKm(1e9), ZOOM.ALT_MAX_KM);
  assert.equal(clampAltKm(NaN), ZOOM.ALT_MIN_KM);
  assert.equal(clampLogAlt(-999), LOG_ALT_MIN);
  assert.equal(clampLogAlt(999), LOG_ALT_MAX);
  assert.equal(logToAlt(999), ZOOM.ALT_MAX_KM);
});

test("a wheel notch is the same PROPORTIONAL move at every altitude", () => {
  // This is the entire reason zoom lives in log space. One notch at
  // 30,000km and one notch at 100m must divide the altitude by the same
  // factor — in linear altitude they would differ by 300,000×.
  const step = wheelToLogDelta(-100, DELTA_MODE.PIXEL);
  const ratios = [30000, 1000, 10, 0.2].map((alt) => logToAlt(altToLog(alt) + step) / alt);
  for (const r of ratios) {
    assert.ok(Math.abs(r - ratios[0]) < 1e-9, `ratio drifted: ${ratios.join(", ")}`);
  }
  assert.ok(ratios[0] < 1, "negative deltaY (scroll up) must zoom IN");
});

// ── input normalization ─────────────────────────────────────────────────────

test("wheel: sign convention — scroll down zooms out", () => {
  assert.ok(wheelToLogDelta(100, DELTA_MODE.PIXEL) > 0, "deltaY > 0 raises altitude");
  assert.ok(wheelToLogDelta(-100, DELTA_MODE.PIXEL) < 0);
  assert.equal(wheelToLogDelta(0, DELTA_MODE.PIXEL), 0);
  assert.equal(wheelToLogDelta(NaN, DELTA_MODE.PIXEL), 0);
});

test("wheel: deltaMode LINE and PAGE normalize to pixels", () => {
  // Firefox reports lines, not pixels; without this a Firefox notch moves
  // 16× a Chrome notch.
  assert.equal(wheelToLogDelta(3, DELTA_MODE.LINE), wheelToLogDelta(3 * LINE_HEIGHT_PX, DELTA_MODE.PIXEL));
  assert.equal(wheelToLogDelta(1, DELTA_MODE.PAGE, 900), wheelToLogDelta(MAX_WHEEL_DELTA_PX, DELTA_MODE.PIXEL));
});

test("wheel: a single monster event is capped", () => {
  // A trackpad flick can deliver a 4-digit deltaY in one event; uncapped it
  // jumps the whole envelope in one frame and supersedes every tile in
  // flight at once.
  const capped = wheelToLogDelta(9999, DELTA_MODE.PIXEL);
  assert.equal(capped, MAX_WHEEL_DELTA_PX * ZOOM.WHEEL_RATE);
  assert.equal(wheelToLogDelta(-9999, DELTA_MODE.PIXEL), -MAX_WHEEL_DELTA_PX * ZOOM.WHEEL_RATE);
});

test("pinch: spreading fingers to 2× spread halves the altitude", () => {
  // The wheel/pinch agreement is by construction, not tuning: pinch is
  // -ln(scale), so scale 2 is exactly one halving in log space.
  const d = pinchToLogDelta(2);
  assert.ok(Math.abs(Math.exp(d) - 0.5) < 1e-12);
  assert.ok(Math.abs(Math.exp(pinchToLogDelta(0.5)) - 2) < 1e-12);
  assert.equal(pinchToLogDelta(1), 0);
  assert.equal(pinchToLogDelta(0), 0, "degenerate spread must not produce -Infinity");
  assert.equal(pinchToLogDelta(NaN), 0);
});

test("pinch and wheel produce the SAME unit — they are interchangeable", () => {
  // Find the wheel deltaY equivalent to a pinch, then confirm both land on
  // the same altitude from the same start. If these ever diverge, desktop
  // and phone stop feeling the same, which is the whole point.
  const pinch = pinchToLogDelta(1.5);
  const deltaY = pinch / ZOOM.WHEEL_RATE;
  const wheel = wheelToLogDelta(deltaY, DELTA_MODE.PIXEL);
  assert.ok(Math.abs(wheel - pinch) < 1e-12);
});

// ── anchoring ───────────────────────────────────────────────────────────────

test("anchorScale keeps the anchor point fixed", () => {
  assert.equal(anchorScale(100, 40, 0.5), 70);
  assert.equal(anchorScale(40, 40, 0.5), 40, "the anchor itself never moves");
  assert.equal(anchorScale(100, 40, 1), 100, "ratio 1 is a no-op");
  const p = anchorScale2({ x: 10, y: 20 }, { x: 0, y: 0 }, 0.25);
  assert.deepEqual(p, { x: 2.5, y: 5 });
});

test("anchorScale composes: two half-zooms equal one quarter-zoom", () => {
  const once = anchorScale(100, 40, 0.25);
  const twice = anchorScale(anchorScale(100, 40, 0.5), 40, 0.5);
  assert.ok(Math.abs(once - twice) < 1e-12);
});

test("spread and centroid", () => {
  assert.equal(spread({ x: 0, y: 0 }, { x: 3, y: 4 }), 5);
  assert.deepEqual(centroid({ x: 0, y: 0 }, { x: 10, y: 20 }), { x: 5, y: 10 });
});

// ── the controller ──────────────────────────────────────────────────────────

test("ZoomInput moves the TARGET, never the value (Law I at the input boundary)", () => {
  const zi = new ZoomInput({ initialAltKm: 1000 });
  const before = zi.altKm;
  const change = zi.wheel(-100, DELTA_MODE.PIXEL, { x: 10, y: 10 }, 0);
  assert.ok(change);
  assert.equal(zi.altKm, before, "input must not assign the camera position");
  assert.ok(zi.targetAltKm < before, "input moves the destination");
  assert.equal(zi.spring.asleep, false, "and wakes the spring that flies there");
});

test("ZoomInput: the spring actually arrives at the target under fixed sub-steps", () => {
  const zi = new ZoomInput({ initialAltKm: 1000 });
  // A -450px event is capped to -MAX_WHEEL_DELTA_PX before rate conversion.
  zi.wheel(-450, DELTA_MODE.PIXEL, { x: 0, y: 0 }, 0);
  const target = zi.targetAltKm;
  const expected = 1000 * Math.exp(-MAX_WHEEL_DELTA_PX * ZOOM.WHEEL_RATE);
  assert.ok(Math.abs(target - expected) < 1e-9, `landed at ${target}, expected ${expected}`);

  const h = FRAME.SPRING_STEP_MS / 1000;
  for (let i = 0; i < 2000 && !zi.spring.asleep; i++) zi.spring.step(h);
  assert.equal(zi.spring.asleep, true);
  assert.ok(Math.abs(zi.altKm - target) < 1e-6);
});

test("ZoomInput: an uncapped-size notch divides altitude by exactly e", () => {
  // -450px at WHEEL_RATE 1/450 is -1 in log space. Delivered as two events
  // inside the per-event cap so the cap does not truncate it.
  const zi = new ZoomInput({ initialAltKm: 1000 });
  zi.wheel(-225, DELTA_MODE.PIXEL, { x: 0, y: 0 }, 0);
  zi.wheel(-225, DELTA_MODE.PIXEL, { x: 0, y: 0 }, 1);
  assert.ok(Math.abs(zi.targetAltKm - 1000 / Math.E) < 1e-9, `landed at ${zi.targetAltKm}`);
});

test("ZoomInput: change.ratio is what anchorScale wants", () => {
  const zi = new ZoomInput({ initialAltKm: 1000 });
  const c = zi.wheel(-225, DELTA_MODE.PIXEL, { x: 5, y: 6 }, 0)!;
  const expected = Math.exp(-225 * ZOOM.WHEEL_RATE);
  assert.ok(Math.abs(c.ratio - expected) < 1e-12, "ratio is newTarget/oldTarget");
  assert.deepEqual(c.anchor, { x: 5, y: 6 });
  assert.ok(Math.abs(c.targetAltKm - 1000 * c.ratio) < 1e-9);
});

test("ZoomInput: pinned at an envelope end, further input is a no-op", () => {
  const zi = new ZoomInput({ initialAltKm: ZOOM.ALT_MAX_KM });
  assert.equal(zi.wheel(500, DELTA_MODE.PIXEL, { x: 0, y: 0 }, 0), null, "already at max altitude");
  assert.equal(zi.targetLogAlt, LOG_ALT_MAX);
  const zi2 = new ZoomInput({ initialAltKm: ZOOM.ALT_MIN_KM });
  assert.equal(zi2.wheel(-500, DELTA_MODE.PIXEL, { x: 0, y: 0 }, 0), null);
});

test("ZoomInput: a custom envelope is respected (the Moon view is not the Earth view)", () => {
  const zi = new ZoomInput({ initialAltKm: 500, minAltKm: 1, maxAltKm: 5000 });
  for (let i = 0; i < 200; i++) zi.wheel(240, DELTA_MODE.PIXEL, { x: 0, y: 0 }, i);
  assert.ok(Math.abs(zi.targetAltKm - 5000) < 1e-6);
  for (let i = 0; i < 400; i++) zi.wheel(-240, DELTA_MODE.PIXEL, { x: 0, y: 0 }, i);
  assert.ok(Math.abs(zi.targetAltKm - 1) < 1e-9);
});

test("ZoomInput: onTarget fires exactly once per real change", () => {
  const seen: number[] = [];
  const zi = new ZoomInput({ initialAltKm: 1000, onTarget: (c) => seen.push(c.targetAltKm) });
  zi.wheel(-45, DELTA_MODE.PIXEL, { x: 0, y: 0 }, 0);
  zi.wheel(0, DELTA_MODE.PIXEL, { x: 0, y: 0 }, 1); // no-op
  zi.wheel(-45, DELTA_MODE.PIXEL, { x: 0, y: 0 }, 2);
  assert.equal(seen.length, 2);
  assert.ok(seen[1] < seen[0]);
});

test("ZoomInput: pinch needs two pointers, and tracks the centroid", () => {
  const anchors: { x: number; y: number }[] = [];
  const zi = new ZoomInput({ initialAltKm: 1000, onTarget: (c) => anchors.push(c.anchor) });
  zi.pointerDown(1, 100, 100);
  assert.equal(zi.pointerMove(1, 120, 100, 0), null, "one finger is a pan, not a zoom");
  zi.pointerDown(2, 200, 100);
  assert.equal(zi.pointerCount, 2);
  // Spread from 80 to 160 = scale 2 = one halving.
  const c = zi.pointerMove(2, 280, 100, 1);
  assert.ok(c);
  assert.ok(Math.abs(c!.ratio - 0.5) < 1e-9, `ratio ${c!.ratio}`);
  assert.deepEqual(anchors[anchors.length - 1], { x: 200, y: 100 }, "anchor is the pinch centroid");
});

test("ZoomInput: a move from an unknown pointer is ignored", () => {
  const zi = new ZoomInput({ initialAltKm: 1000 });
  assert.equal(zi.pointerMove(99, 0, 0, 0), null);
  assert.equal(zi.pointerCount, 0);
});

test("ZoomInput: re-grabbing after a pinch does not zoom against the STALE spread", () => {
  // The classic bug: pointerup/pointerdown leave the previous gesture's
  // spread in place, so the first move of the new gesture computes its
  // scale against it and teleports the camera. Here the old gesture ended
  // at spread 200 and the new one starts at spread 50 — the stale-spread
  // bug would read that as a 4× zoom on the first 1px of movement.
  const zi = new ZoomInput({ initialAltKm: 1000 });
  zi.pointerDown(1, 100, 100);
  zi.pointerDown(2, 200, 100);
  zi.pointerMove(2, 300, 100, 0); // spread 100 → 200
  const afterPinch = zi.targetAltKm;

  zi.pointerUp(2);
  zi.pointerDown(3, 150, 100); // new gesture: spread 50
  zi.pointerMove(3, 151, 100, 1); // spread 50 → 51, a 2% zoom

  const jump = Math.abs(Math.log(zi.targetAltKm / afterPinch));
  const legitimate = Math.abs(Math.log(51 / 50));
  assert.ok(
    Math.abs(jump - legitimate) < 1e-9,
    `re-grab moved the target by ${jump} in log space; only ${legitimate} is legitimate`,
  );
});

test("ZoomInput: clearPointers ends any gesture cleanly", () => {
  const zi = new ZoomInput({ initialAltKm: 1000 });
  zi.pointerDown(1, 0, 0);
  zi.pointerDown(2, 10, 0);
  zi.clearPointers();
  assert.equal(zi.pointerCount, 0);
  assert.equal(zi.pointerMove(1, 50, 0, 0), null);
});

test("ZoomInput.isSettled is the prefetch signal, not a spring cutoff", () => {
  // Law II.4: streaming prefetches from the TARGET. isSettled says the
  // target has stopped moving — while the camera is still flying to it.
  const zi = new ZoomInput({ initialAltKm: 1000 });
  zi.wheel(-100, DELTA_MODE.PIXEL, { x: 0, y: 0 }, 1000);
  assert.equal(zi.isSettled(1000), false);
  assert.equal(zi.isSettled(1000 + SPRING.ZOOM_SETTLE_MS - 1), false);
  assert.equal(zi.isSettled(1000 + SPRING.ZOOM_SETTLE_MS), true);
  assert.equal(zi.spring.asleep, false, "settled target, still-flying camera — the point of the split");
  zi.wheel(-100, DELTA_MODE.PIXEL, { x: 0, y: 0 }, 1200);
  assert.equal(zi.isSettled(1200), false, "new input un-settles the target");
});

test("ZoomInput: snapToAltKm teleports value and target together", () => {
  const zi = new ZoomInput({ initialAltKm: 1000 });
  zi.wheel(-100, DELTA_MODE.PIXEL, { x: 0, y: 0 }, 0);
  zi.snapToAltKm(42);
  assert.ok(Math.abs(zi.altKm - 42) < 1e-9);
  assert.ok(Math.abs(zi.targetAltKm - 42) < 1e-9);
  assert.equal(zi.spring.asleep, true);
});

test("ZoomInput: setTargetAltKm shares the wheel path (a fly-to cannot diverge)", () => {
  const zi = new ZoomInput({ initialAltKm: 1000 });
  const c = zi.setTargetAltKm(250, { x: 1, y: 2 }, 0);
  assert.ok(c);
  assert.ok(Math.abs(c!.targetAltKm - 250) < 1e-9);
  assert.ok(Math.abs(c!.ratio - 0.25) < 1e-12);
  assert.ok(Math.abs(zi.altKm - 1000) < 1e-9, "still a target, not an assignment");
});

test("ZoomInput accepts an externally-owned spring (the loop integrates it)", () => {
  const s = new Spring(Math.log(1000), { stiffness: SPRING.ZOOM_STIFFNESS, damping: SPRING.ZOOM_DAMPING });
  const zi = new ZoomInput({ spring: s });
  zi.wheel(-225, DELTA_MODE.PIXEL, { x: 0, y: 0 }, 0);
  assert.equal(zi.spring, s);
  assert.ok(Math.abs(s.target - (Math.log(1000) - 225 * ZOOM.WHEEL_RATE)) < 1e-12);
});
