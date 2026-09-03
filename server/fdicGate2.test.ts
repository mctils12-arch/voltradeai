// fdic_bank_failures GATE 2 (SIGNAL) event-study battery: forward-return
// computation, the bootstrap null test against the base rate, small-n
// floor, event-date dedup, and seeded reproducibility. Synthetic bar
// fixtures only — this module never touches fs/network.
import { test } from "node:test";
import assert from "node:assert/strict";
import { forwardReturn, eventStudy, type Bar } from "./fdicGate2";

function makeFlatBars(n: number, start = "2023-01-02", dailyReturn = 0): Bar[] {
  const bars: Bar[] = [];
  let close = 100;
  const d0 = new Date(start + "T00:00:00Z");
  for (let i = 0; i < n; i++) {
    const d = new Date(d0.getTime() + i * 86400_000);
    bars.push({ date: d.toISOString().slice(0, 10), close });
    close *= 1 + dailyReturn;
  }
  return bars;
}

/** Deterministic but non-constant daily returns (a sawtooth-ish pattern),
 *  so forward-return values actually vary across the eligible base-rate
 *  pool — needed for the seed-reproducibility test below, where
 *  `makeFlatBars`'s pure constant-return series would make every
 *  bootstrap draw identical regardless of seed and trivially defeat the
 *  "different seed can differ" assertion. */
function makeVariedBars(n: number, start = "2023-01-02"): Bar[] {
  const bars: Bar[] = [];
  const d0 = new Date(start + "T00:00:00Z");
  for (let i = 0; i < n; i++) {
    const d = new Date(d0.getTime() + i * 86400_000);
    bars.push({ date: d.toISOString().slice(0, 10), close: 100 + 8 * Math.sin(i * 0.37) + (i % 5) });
  }
  return bars;
}

test("forwardReturn: basic computation, entry = first bar on/after entryDate", () => {
  const bars: Bar[] = [
    { date: "2023-01-02", close: 100 },
    { date: "2023-01-03", close: 110 },
    { date: "2023-01-04", close: 121 },
  ];
  assert.ok(Math.abs(forwardReturn(bars, "2023-01-02", 2)! - 0.21) < 1e-9);
  assert.ok(Math.abs(forwardReturn(bars, "2023-01-01", 2)! - 0.21) < 1e-9,
    "entryDate before the series' first bar still anchors on the first bar");
});

test("forwardReturn: null when not enough forward bar depth, or entryDate after the whole series", () => {
  const bars: Bar[] = [
    { date: "2023-01-02", close: 100 },
    { date: "2023-01-03", close: 110 },
  ];
  assert.equal(forwardReturn(bars, "2023-01-02", 5), null, "horizon runs past the last bar");
  assert.equal(forwardReturn(bars, "2023-06-01", 1), null, "entryDate after every bar in the series");
});

test("forwardReturn: null (never divide-by-zero/Infinity) on a non-positive entry close", () => {
  const bars: Bar[] = [{ date: "2023-01-02", close: 0 }, { date: "2023-01-03", close: 5 }];
  assert.equal(forwardReturn(bars, "2023-01-02", 1), null);
});

test("eventStudy: insufficient_n when fewer than 5 valid events, nulls out the fields a real verdict needs", () => {
  const bars = makeFlatBars(60);
  const v = eventStudy(bars, ["2023-01-02", "2023-01-10", "2023-01-20"], 5, 1);
  assert.equal(v.insufficient_n, true);
  assert.equal(v.n_events_valid, 3);
  assert.equal(v.base_rate_mean_return, null);
  assert.equal(v.bootstrap_ci_95, null);
  assert.equal(v.two_sided_p, null);
  assert.equal(v.signal_detected, false, "insufficient_n must never also claim a detected signal");
});

test("eventStudy: clean null (constant daily return everywhere) -> event mean exactly matches base rate, no signal", () => {
  // Every bar-to-bar return is the SAME fixed 0.1% — event windows and the
  // base-rate pool are drawn from an identical, noiseless series, so the
  // event mean, base-rate mean, and every bootstrap draw must all land on
  // the exact same forward-return value. This is the deterministic
  // (non-random-seed-dependent) proof that the test doesn't manufacture a
  // false positive out of nothing.
  const bars = makeFlatBars(300, "2023-01-02", 0.001);
  const events = ["2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01", "2023-06-01", "2023-07-01"];
  const v = eventStudy(bars, events, 5, 7);
  assert.equal(v.insufficient_n, false);
  assert.ok(v.event_mean_return != null && v.base_rate_mean_return != null);
  assert.ok(Math.abs(v.event_mean_return! - v.base_rate_mean_return!) < 1e-9,
    "a noiseless series must show zero event/base-rate gap");
  assert.equal(v.signal_detected, false);
  // A true null does not guarantee a HIGH p-value (p is uniformly
  // distributed under H0, not concentrated near 1) — it guarantees a
  // p-value that clears the 0.05 significance bar, which is exactly what
  // `signal_detected: false` above already certifies. This assertion only
  // pins that the two agree (no case where one says "not significant" and
  // the other's raw number implies otherwise).
  assert.ok(v.two_sided_p! > 0.05, `expected p above the 0.05 bar to match signal_detected=false, got p=${v.two_sided_p}`);
});

test("eventStudy: a real, isolated post-event jump is detected against a flat base rate", () => {
  // Flat/noiseless series (base rate ~= 0) EXCEPT a fixed +20% jump
  // starting exactly `horizon` bars after each of 6 event dates, spaced
  // far enough apart that no event's own exclusion window overlaps
  // another's jump — an unambiguous, hand-verifiable enrichment case.
  const n = 400;
  const horizon = 5;
  const bars = makeFlatBars(n);
  const eventIdxs = [20, 60, 100, 140, 180, 220];
  for (const idx of eventIdxs) {
    for (let i = idx + horizon; i < n; i++) bars[i] = { ...bars[i], close: bars[i].close * 1.2 };
  }
  const events = eventIdxs.map((i) => bars[i].date);
  const v = eventStudy(bars, events, horizon, 3);
  assert.equal(v.insufficient_n, false);
  assert.equal(v.n_events_valid, 6);
  assert.ok(v.event_mean_return! > 0.15, `expected ~20% event-window jump, got ${v.event_mean_return}`);
  assert.ok(Math.abs(v.base_rate_mean_return!) < 0.05, `expected a near-zero base rate away from the jumps, got ${v.base_rate_mean_return}`);
  assert.equal(v.signal_detected, true);
  assert.equal(v.two_sided_p, 0, "the jump is far outside every one of 2000 bootstrap draws from the flat pool");
});

test("eventStudy: duplicate event dates count once, not twice", () => {
  const bars = makeFlatBars(300, "2023-01-02", 0.0005);
  const events = ["2023-02-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01", "2023-06-01"];
  const v = eventStudy(bars, events, 5, 1);
  assert.equal(v.n_events_total, 5, "the repeated 2023-02-01 must collapse to one event");
});

test("eventStudy: same seed -> identical verdict (reproducible from a logged seed); a different seed can differ", () => {
  const bars = makeVariedBars(300);
  const events = ["2023-02-01", "2023-02-15", "2023-03-01", "2023-03-15", "2023-04-01", "2023-04-15"];
  const a = eventStudy(bars, events, 5, 42);
  const b = eventStudy(bars, events, 5, 42);
  assert.deepEqual(a, b);
  const c = eventStudy(bars, events, 5, 43);
  assert.ok(
    a.bootstrap_ci_95!.lower !== c.bootstrap_ci_95!.lower || a.bootstrap_ci_95!.upper !== c.bootstrap_ci_95!.upper,
    "different seeds should not coincidentally draw the identical bootstrap sample every time",
  );
});
