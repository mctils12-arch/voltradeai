// simClock tests — rate math, continuity, pause, ⟲now, and above all the
// REGRESSION CONTRACT: at 1× real time the clock is the bit-exact identity
// of Date.now (celestial v2 B3). `npx tsx --test`.

import test from "node:test";
import assert from "node:assert/strict";

import {
  SIM_RATES,
  simNowMs,
  simOffsetMs,
  isRealtime,
  resetState,
  withRate,
  simRateLabel,
  fmtSimOffset,
  getSimClock,
  setSimRate,
  resetSimClock,
  subscribeSimClock,
  simNow,
  type SimClockState,
} from "./simClock.js";

const T0 = Date.UTC(2026, 6, 18, 12, 0, 0);

test("realtime identity: a reset clock maps every real instant to itself, bit-exactly", () => {
  const st = resetState(T0);
  assert.equal(isRealtime(st), true);
  // arbitrary instants, including fractional ms and far future — EXACT equality
  for (const t of [T0, T0 + 1, T0 + 1234.567, T0 + 365 * 86_400_000, 0, 9e12]) {
    assert.equal(simNowMs(st, t), t, `identity at ${t}`);
    assert.equal(simOffsetMs(st, t), 0, `zero offset at ${t}`);
  }
});

test("rate math: simulated time advances rate× real time, for every control rate", () => {
  for (const rate of SIM_RATES) {
    const st = withRate(resetState(T0), rate, T0);
    assert.equal(simNowMs(st, T0 + 1000) - simNowMs(st, T0), rate * 1000, `rate ${rate}`);
  }
  // paused: frozen
  const paused = withRate(resetState(T0), 0, T0);
  assert.equal(simNowMs(paused, T0 + 3_600_000), simNowMs(paused, T0), "paused clock is frozen");
});

test("withRate continuity: the simulated instant never jumps across a rate switch", () => {
  let st = resetState(T0);
  let real = T0;
  // 1× → 60× → 3600× → pause → 1 day/s → 1×, dwelling at each
  for (const rate of [60, 3600, 0, 86400, 1]) {
    real += 12_345;
    const before = simNowMs(st, real);
    st = withRate(st, rate, real);
    assert.equal(simNowMs(st, real), before, `continuous at switch to ${rate}×`);
  }
  // after that walk the clock is rate 1 but OFFSET — not realtime, honestly so
  assert.equal(st.rate, 1);
  assert.equal(isRealtime(st), false, "1× with an offset is NOT realtime");
  assert.ok(simOffsetMs(st, real) > 0, "warp left simulated time ahead of real");
});

test("⟲now: reset restores the exact identity state", () => {
  const warped = withRate(withRate(resetState(T0), 3600, T0), 1, T0 + 60_000);
  assert.equal(isRealtime(warped), false);
  const reset = resetState(T0 + 120_000);
  assert.equal(isRealtime(reset), true);
  assert.equal(simNowMs(reset, T0 + 500_000), T0 + 500_000);
});

test("withRate guards: non-finite / negative rates clamp to paused", () => {
  const st = resetState(T0);
  assert.equal(withRate(st, Number.NaN, T0).rate, 0);
  assert.equal(withRate(st, -5, T0).rate, 0);
  assert.equal(withRate(st, Number.POSITIVE_INFINITY, T0).rate, 0);
});

test("simRateLabel: the directive's control labels", () => {
  assert.equal(simRateLabel(1), "1×");
  assert.equal(simRateLabel(60), "60×");
  assert.equal(simRateLabel(3600), "3600×");
  assert.equal(simRateLabel(86400), "1 day/s");
  assert.equal(simRateLabel(0), "paused");
});

test("fmtSimOffset: signed, two most significant units", () => {
  assert.equal(fmtSimOffset(0), "0s");
  assert.equal(fmtSimOffset(400), "0s");
  assert.equal(fmtSimOffset(45_000), "+45s");
  assert.equal(fmtSimOffset(-45_000), "−45s");
  assert.equal(fmtSimOffset(65_000), "+1m 5s");
  assert.equal(fmtSimOffset((3 * 3600 + 12 * 60) * 1000), "+3h 12m");
  assert.equal(fmtSimOffset(-(2 * 86400 + 4 * 3600) * 1000), "−2d 4h");
  assert.equal(fmtSimOffset(86_400_000), "+1d");
});

test("store: subscribe/set/reset lifecycle, and simNow tracks Date.now at realtime", () => {
  // ensure a clean start (other tests may not have touched the store)
  resetSimClock();
  assert.equal(isRealtime(getSimClock()), true);
  // at realtime simNow() IS Date.now() (bit-exact affine identity); the two
  // reads happen at slightly different instants, so bound the skew instead
  // of asserting equality of two separate clock reads
  const a = simNow();
  const b = Date.now();
  assert.ok(Math.abs(b - a) <= 50, "simNow ≡ Date.now at realtime");
  let fired = 0;
  const off = subscribeSimClock(() => { fired++; });
  setSimRate(60);
  assert.equal(fired, 1, "rate change notifies");
  assert.equal(getSimClock().rate, 60);
  assert.equal(isRealtime(getSimClock()), false);
  setSimRate(0);
  assert.equal(fired, 2);
  assert.equal(getSimClock().rate, 0, "paused");
  resetSimClock();
  assert.equal(fired, 3, "reset notifies");
  assert.equal(isRealtime(getSimClock()), true);
  resetSimClock();
  assert.equal(fired, 3, "reset at realtime is a no-op");
  off();
  setSimRate(3600);
  assert.equal(fired, 3, "unsubscribed");
  resetSimClock(); // leave the store clean for any later test file
});
