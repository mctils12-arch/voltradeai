// foldShadow.ts battery — the shared three-mode vocabulary of memory fix A2b
// (gnssIntegrityFold.ts, entityGraph.ts's managed poll). Pins the one
// property that makes the rollout safe: an unknown flag never selects the
// unproven path, and a mismatch report names WHERE the two folds diverged.
import { test } from "node:test";
import assert from "node:assert/strict";
import { firstDifference, parseFoldMode } from "./foldShadow";

test("parseFoldMode: unset, empty and unknown values all resolve to shadow (never to the new path)", () => {
  assert.equal(parseFoldMode(undefined), "shadow");
  assert.equal(parseFoldMode(null), "shadow");
  assert.equal(parseFoldMode(""), "shadow");
  assert.equal(parseFoldMode("incrementl"), "shadow");
  assert.equal(parseFoldMode("on"), "shadow");
  assert.equal(parseFoldMode("shadow"), "shadow");
});

test("parseFoldMode: full and incremental are recognised case- and whitespace-insensitively", () => {
  assert.equal(parseFoldMode("full"), "full");
  assert.equal(parseFoldMode(" FULL "), "full");
  assert.equal(parseFoldMode("incremental"), "incremental");
  assert.equal(parseFoldMode("Incremental\n"), "incremental");
});

test("firstDifference: deep-strict-equal values (any key order) report null", () => {
  assert.equal(firstDifference({ a: 1, b: [1, { c: "x" }] }, { b: [1, { c: "x" }], a: 1 }), null);
  assert.equal(firstDifference([], []), null);
  assert.equal(firstDifference(null, null), null);
});

test("firstDifference: names the first differing path, down through arrays and nested objects", () => {
  const a = { bands: [{ band: "cruise", candidate_k: 3 }, { band: "mid", candidate_k: 8 }], verdict: "PASS" };
  const b = { bands: [{ band: "cruise", candidate_k: 3 }, { band: "mid", candidate_k: 9 }], verdict: "PASS" };
  assert.equal(firstDifference(a, b), "$.bands[1].candidate_k: 8 vs 9");
});

test("firstDifference: array length and one-side-only keys are themselves the reported difference", () => {
  assert.equal(firstDifference({ days: ["a", "b"] }, { days: ["a"] }), "$.days.length: 2 vs 1");
  assert.equal(firstDifference({ x: 1 }, { x: 1, y: 2 }), "$.y: present on one side only");
  assert.ok(firstDifference(1, "1")!.startsWith("$: 1 vs \"1\""), "type-different primitives are a difference");
});

test("firstDifference: the report is bounded so it always fits an audit row", () => {
  const big = { k: "x".repeat(5000) };
  const d = firstDifference(big, { k: "y".repeat(5000) })!;
  assert.ok(d.length <= 400);
});
