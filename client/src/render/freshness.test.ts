// freshness.test.ts — Law V, and the reconcile predicate behind the
// OpenWeatherMap root-cause fix.

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  DEFAULT_DEAD_AFTER_MS,
  DEFAULT_STALE_AFTER_MS,
  FreshnessTracker,
  formatDataAge,
  freshnessLevel,
  needsReconcile,
  reportChainFallthrough,
} from "./freshness.ts";

// ── the reconcile predicate ─────────────────────────────────────────────────

test("needsReconcile: a wanted, serving layer with BOTH halves present is left alone", () => {
  assert.equal(
    needsReconcile({ wanted: true, providerOk: true, sourcePresent: true, layerPresent: true }),
    false,
    "re-adding a healthy layer every styledata would thrash the map",
  );
});

test("needsReconcile: a style rebuild that wiped both halves is caught", () => {
  // The actual OpenWeatherMap non-render: setProjection() rebuilds the
  // style, both halves vanish, and nothing re-added them for up to 10
  // minutes because the only retry was a setInterval.
  assert.equal(needsReconcile({ wanted: true, providerOk: true, sourcePresent: false, layerPresent: false }), true);
});

test("needsReconcile: a HALF-built pair is also a broken layer", () => {
  // A source with no layer draws nothing; a layer with no source throws on
  // the next render. The old `if (!map.getSource(...))` guard missed both,
  // because it only ever checked the source.
  assert.equal(needsReconcile({ wanted: true, providerOk: true, sourcePresent: true, layerPresent: false }), true);
  assert.equal(needsReconcile({ wanted: true, providerOk: true, sourcePresent: false, layerPresent: true }), true);
});

test("needsReconcile: never re-adds a layer the user turned off or a dead provider", () => {
  assert.equal(needsReconcile({ wanted: false, providerOk: true, sourcePresent: false, layerPresent: false }), false);
  assert.equal(needsReconcile({ wanted: true, providerOk: false, sourcePresent: false, layerPresent: false }), false);
});

// ── age classification ──────────────────────────────────────────────────────

test("freshnessLevel bands, with `unknown` distinct from `dead`", () => {
  // A layer that has never reported an age has not FAILED — it has failed
  // to instrument itself. Law V: it may not claim to be live either way,
  // but conflating the two hides a wiring bug behind an outage.
  assert.equal(freshnessLevel(null), "unknown");
  assert.equal(freshnessLevel(undefined), "unknown");
  assert.equal(freshnessLevel(NaN), "unknown");
  assert.equal(freshnessLevel(-1), "unknown");
  assert.equal(freshnessLevel(0), "fresh");
  assert.equal(freshnessLevel(DEFAULT_STALE_AFTER_MS - 1), "fresh");
  assert.equal(freshnessLevel(DEFAULT_STALE_AFTER_MS), "stale");
  assert.equal(freshnessLevel(DEFAULT_DEAD_AFTER_MS), "dead");
});

test("formatDataAge is coarse on purpose", () => {
  // Rendering an age to the second implies a precision a 10-minute poll
  // does not have.
  assert.equal(formatDataAge(0), "just now");
  assert.equal(formatDataAge(44_000), "just now");
  assert.equal(formatDataAge(5 * 60_000), "5m ago");
  assert.equal(formatDataAge(90 * 60_000), "1h ago");
  assert.equal(formatDataAge(50 * 3600_000), "2d ago");
  assert.equal(formatDataAge(null), "age unknown");
});

// ── the tracker ─────────────────────────────────────────────────────────────

function clock(start = 1_000_000) {
  const t = { now: start };
  return { t, fn: () => t.now };
}

test("a tracker that has never been fed reports unknown, not fresh", () => {
  const c = clock();
  const f = new FreshnessTracker("x", c.fn);
  assert.equal(f.ageMs, null);
  assert.equal(f.level, "unknown");
  assert.equal(f.describe(), "age unknown", "silence must never imply freshness");
});

test("the tracker ages, and markFresh resets it", () => {
  const c = clock();
  const f = new FreshnessTracker("x", c.fn);
  f.markFresh();
  assert.equal(f.ageMs, 0);
  c.t.now += 5 * 60_000;
  assert.equal(f.ageMs, 5 * 60_000);
  assert.equal(f.describe(), "5m ago");
  f.markFresh();
  assert.equal(f.ageMs, 0);
});

test("a failure does NOT blank the last-known age (Law V renders cached state)", () => {
  // "Provider failover must never block first paint. Render last-known
  // cached state immediately." Dropping the timestamp on failure would
  // make the layer claim it has no data when it has old data.
  const c = clock();
  const f = new FreshnessTracker("x", c.fn);
  f.markFresh();
  c.t.now += 60_000;
  f.markFailure();
  assert.equal(f.ageMs, 60_000, "the last good timestamp must survive a failure");
  assert.equal(f.failures, 1);
  assert.match(f.describe(), /retrying \(1\)/);
  f.markFailure();
  assert.equal(f.failures, 2);
  f.markFresh();
  assert.equal(f.failures, 0, "a success clears the retry counter");
  assert.equal(f.describe(), "just now");
});

test("reset returns the tracker to unknown", () => {
  const c = clock();
  const f = new FreshnessTracker("x", c.fn);
  f.markFresh();
  f.markFailure();
  f.reset();
  assert.equal(f.level, "unknown");
  assert.equal(f.failures, 0);
  assert.equal(f.lastFailureAtMs, null);
});

// ── chain fallthrough ───────────────────────────────────────────────────────

test("reaching the LAST provider in a chain logs loudly; earlier ones stay quiet", () => {
  const err = console.error;
  const lines: string[] = [];
  console.error = (m: string) => lines.push(String(m));
  try {
    const chain = ["primary", "secondary", "backup"];
    assert.equal(reportChainFallthrough("ais", 0, chain), false);
    assert.equal(reportChainFallthrough("ais", 1, chain), false);
    assert.equal(lines.length, 0, "a chain with fallback left is not degraded");

    assert.equal(reportChainFallthrough("ais", 2, chain), true);
    assert.equal(lines.length, 1);
    assert.match(lines[0], /LAST option/);
    assert.match(lines[0], /backup/);
    assert.match(lines[0], /primary -> secondary -> backup/, "the whole chain must be in the log");
  } finally {
    console.error = err;
  }
});

test("reportChainFallthrough handles a single-provider chain and an empty one", () => {
  const err = console.error;
  console.error = () => {};
  try {
    // One provider IS the last one — a chain with no fallback is degraded
    // by construction and should say so.
    assert.equal(reportChainFallthrough("solo", 0, ["only"]), true);
    assert.equal(reportChainFallthrough("none", 0, []), false);
  } finally {
    console.error = err;
  }
});
