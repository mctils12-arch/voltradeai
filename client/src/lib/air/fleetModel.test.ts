import { test } from "node:test";
import assert from "node:assert/strict";
import {
  prepareFleet, trackFromPoints, spliceOverride, headAt, assignLods, decimateIdx, buildFleetVerts, fleetBudgetForTier,
  eyeDistanceKm, trackScore, pointsPerTrack, headColor,
  FLEET_STRIDE, FLEET_BUDGETS, FLEET_GAP_SEC, FLEET_HOLD_SEC, LOD_FULL, LOD_THIN, LOD_HEAD, LOD_HIDDEN,
  type WindowHexIn, type FleetBudget,
} from "./fleetModel.ts";

const T0 = 1_759_000_000;

function hex(id: string, n: number, every: number, opts: { gaps?: number[]; alt?: number | null; t0?: number } = {}): WindowHexIn {
  const points: Array<[number, number, number, number | null]> = [];
  for (let k = 0; k < n; k++) points.push([(opts.t0 ?? T0) + k * every, 40 + k * 0.01, -100 + k * 0.02, opts.alt === undefined ? 10000 : opts.alt]);
  return { i: id, c: "CS" + id, points, ...(opts.gaps ? { gaps: opts.gaps } : {}) };
}

test("headAt: linear between the two bracketing REAL fixes", () => {
  const [tr] = prepareFleet([hex("aaaaaa", 3, 60)], 60);
  const h = headAt(tr, T0 + 30)!;
  assert.ok(Math.abs(h.lat - 40.005) < 1e-9);
  assert.ok(Math.abs(h.lon - (-99.99)) < 1e-9);
  assert.equal(h.altM, 10000);
  assert.equal(h.nearestFixSec, 30);
  assert.ok(h.hdg > 0 && h.hdg < 90, `heading NE-ish (${h.hdg})`);
  const exact = headAt(tr, T0 + 60)!;
  assert.equal(exact.nearestFixSec, 0);
  assert.ok(Math.abs(exact.lat - 40.01) < 1e-12);
});

test("headAt: nothing before the first fix, beyond the hold, or across an honest gap", () => {
  // server-marked gap before the point at T0 + 3*300
  const [tr] = prepareFleet([hex("aaaaaa", 6, 300, { gaps: [T0 + 900] })], 300);
  const last = T0 + 5 * 300;
  assert.equal(headAt(tr, T0 - 1), null);
  assert.ok(headAt(tr, last), "the last fix itself is drawn");
  const heldEnd = headAt(tr, last + FLEET_HOLD_SEC)!;
  assert.ok(heldEnd, "held at the last fix for one cadence window");
  assert.equal(heldEnd.lat, tr.lat[5], "held, never extrapolated");
  assert.equal(heldEnd.nearestFixSec, FLEET_HOLD_SEC);
  assert.equal(headAt(tr, last + FLEET_HOLD_SEC + 1), null, "beyond the hold: not drawn");
  // gap between fixes at T0+600 and T0+900
  const inGapEarly = headAt(tr, T0 + 600 + 60)!;
  assert.equal(inGapEarly.lat, tr.lat[2], "just after the pre-gap fix: held there, not moved toward the far side");
  assert.equal(headAt(tr, T0 + 750 + 60), null, "deep inside the marked gap: not drawn");
  assert.ok(headAt(tr, T0 + 450), "5-min decimation spacing without a mark is NOT a gap");
  assert.ok(headAt(tr, T0 + 450)!.lat > tr.lat[1], "…and is interpolated, not held");
});

test("prepareFleet gap fallback: payloads without `gaps` use max(10 min, 2 × step)", () => {
  const [a] = prepareFleet([hex("aaaaaa", 4, 900)], 900); // 15-min spacing, 15-min step
  assert.equal(Array.from(a.gapBefore).some((g) => g === 1), false, "spacing = step is not a gap");
  const [b] = prepareFleet([hex("bbbbbb", 4, 700)], 60); // 11.7 min spacing at a 1-min step
  assert.equal(b.gapBefore[1], 1);
  assert.equal(FLEET_GAP_SEC, 600);
});

test("trackFromPoints: full-fidelity override marks raw holes > 10 min", () => {
  const tr = trackFromPoints("aaaaaa", "AAL1", [
    { t: T0, la: 40, lo: -100, al: 9000 },
    { t: T0 + 30, la: 40.01, lo: -100, al: 9000 },
    { t: T0 + 30 + 601, la: 40.2, lo: -100, al: null },
  ])!;
  assert.equal(tr.n, 3);
  assert.deepEqual(Array.from(tr.gapBefore), [0, 0, 1]);
  assert.ok(Number.isNaN(tr.alt[2]));
});

test("spliceOverride: full-fidelity fixes replace the window's only inside their span, and only if they cover the encounter", () => {
  // window track: 5-min fixes T0 .. T0+3600
  const [base] = prepareFleet([hex("aaaaaa", 13, 300)], 300);
  // full-res: 30 s fixes T0+1200 .. T0+1800
  const pts = [];
  for (let s = 1200; s <= 1800; s += 30) pts.push({ t: T0 + s, la: 40.1, lo: -99.9 + s * 1e-5, al: 9500 });
  const ov = trackFromPoints("aaaaaa", "AAL1", pts)!;
  const merged = spliceOverride(base, ov, [T0 + 1500, T0 + 1560])!;
  assert.ok(merged);
  const ts = Array.from(merged.t);
  assert.ok(ts.includes(T0) && ts.includes(T0 + 3600), "window points outside the span kept");
  assert.ok(!ts.includes(T0 + 1500) || merged.lat[ts.indexOf(T0 + 1500)] === 40.1, "inside the span: the full-res fix wins");
  assert.equal(ts.filter((t) => t >= T0 + 1200 && t <= T0 + 1800).length, 21);
  assert.equal(Array.from(merged.gapBefore).some((g) => g === 1), false, "300 s junctions are not gaps");
  // a response for the wrong time never replaces real window data
  const wrong = trackFromPoints("aaaaaa", "AAL1", [{ t: T0 + 90_000, la: 36, lo: -97, al: 500 }, { t: T0 + 90_060, la: 36.1, lo: -97, al: 900 }])!;
  assert.equal(spliceOverride(base, wrong, [T0 + 1500]), null);
  // no window track at all (capped out): the override alone
  assert.ok(spliceOverride(undefined, ov, [T0 + 1500]));
});

test("assignLods: budgets per class, best-scored first, HIDDEN for non-finite scores", () => {
  const budget: FleetBudget = { full: 2, thin: 3, heads: 7, fullSegments: 1e9, thinSegments: 1e9 };
  const scores = new Float64Array([-1, -2, -3, -4, -5, -6, -7, -8, -9, -Infinity]);
  const l = assignLods(scores, null, budget, 0);
  assert.deepEqual(Array.from(l), [LOD_FULL, LOD_FULL, LOD_THIN, LOD_THIN, LOD_THIN, LOD_HEAD, LOD_HEAD, LOD_HIDDEN, LOD_HIDDEN, LOD_HIDDEN]);
});

test("assignLods: hysteresis — a boundary track does not flicker, a clear winner still promotes", () => {
  const budget: FleetBudget = { full: 10, thin: 20, heads: 60, fullSegments: 1e9, thinSegments: 1e9 };
  const n = 60;
  const scores = new Float64Array(n);
  for (let i = 0; i < n; i++) scores[i] = -i;
  const first = assignLods(scores, null, budget);
  assert.equal(Array.from(first).filter((x) => x === LOD_FULL).length, 10);
  // track 9 (last FULL) and track 10 (first THIN) swap ranks by a hair
  const jitter = scores.slice();
  jitter[9] = -10.01; jitter[10] = -9.99;
  const second = assignLods(jitter, first, budget);
  assert.equal(second[9], LOD_FULL, "incumbent inside the keep band stays FULL");
  assert.equal(second[10], LOD_THIN, "challenger outside the entry band stays THIN");
  assert.equal(Array.from(second).filter((x) => x === LOD_FULL).length, 10, "budget never exceeded");
  // swap back and forth 20 times: zero class changes
  let prev = second;
  for (let k = 0; k < 20; k++) {
    const s = scores.slice();
    if (k % 2) { s[9] = -10.01; s[10] = -9.99; }
    const next = assignLods(s, prev, budget);
    assert.deepEqual(Array.from(next), Array.from(prev));
    prev = next;
  }
  // a track that jumps to the top is promoted immediately
  const jump = scores.slice();
  jump[45] = 100;
  const third = assignLods(jump, prev, budget);
  assert.equal(third[45], LOD_FULL);
  assert.equal(Array.from(third).filter((x) => x === LOD_FULL).length, 10);
});

test("assignLods: every class respects its budget on a large random fleet", () => {
  const b = FLEET_BUDGETS.full;
  let seed = 3;
  const rnd = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
  const scores = new Float64Array(5000);
  for (let i = 0; i < scores.length; i++) scores[i] = rnd() < 0.1 ? -Infinity : -rnd() * 1000;
  let prev: Uint8Array | null = null;
  for (let k = 0; k < 5; k++) {
    for (let i = 0; i < scores.length; i++) if (Number.isFinite(scores[i])) scores[i] += (rnd() - 0.5) * 5;
    const l: Uint8Array = assignLods(scores, prev, b);
    const count = (c: number) => Array.from(l).filter((x) => x === c).length;
    assert.ok(count(LOD_FULL) <= b.full);
    assert.ok(count(LOD_FULL) + count(LOD_THIN) <= b.full + b.thin);
    assert.ok(count(LOD_FULL) + count(LOD_THIN) + count(LOD_HEAD) <= b.heads);
    for (let i = 0; i < scores.length; i++) if (!Number.isFinite(scores[i])) assert.equal(l[i], LOD_HIDDEN);
    prev = l;
  }
});

test("device-tier budgets: full > reduced > minimal; unknown tier → reduced", () => {
  assert.equal(fleetBudgetForTier("full"), FLEET_BUDGETS.full);
  assert.equal(fleetBudgetForTier("minimal"), FLEET_BUDGETS.minimal);
  assert.equal(fleetBudgetForTier(undefined), FLEET_BUDGETS.reduced);
  for (const k of ["full", "thin", "heads", "fullSegments", "thinSegments"] as const) {
    assert.ok(FLEET_BUDGETS.full[k] > FLEET_BUDGETS.reduced[k] && FLEET_BUDGETS.reduced[k] > FLEET_BUDGETS.minimal[k], k);
  }
  // heads-only tier exists at every device class (heads > full + thin)
  for (const t of ["full", "reduced", "minimal"] as const) {
    assert.ok(FLEET_BUDGETS[t].heads > FLEET_BUDGETS[t].full + FLEET_BUDGETS[t].thin, t);
  }
});

test("scoring: nearer to the eye scores higher; off-screen always ranks below on-screen", () => {
  const eye = { lat: 40, lon: -100, heightM: 20000 };
  const near = eyeDistanceKm(eye, 40.1, -100, 10000);
  const far = eyeDistanceKm(eye, 41, -100, 10000);
  assert.ok(near < far);
  assert.ok(trackScore(far, true) > trackScore(near, false));
});

test("decimateIdx keeps ends and both sides of every gap, and fits the cap", () => {
  const [tr] = prepareFleet([hex("aaaaaa", 200, 30, { gaps: [T0 + 100 * 30] })], 30);
  const idx = decimateIdx(tr, 20);
  assert.equal(idx[0], 0);
  assert.equal(idx[idx.length - 1], 199);
  assert.ok(idx.includes(99) && idx.includes(100));
  assert.ok(idx.length <= 22);
  assert.deepEqual(decimateIdx(tr, 500).length, 200);
  assert.ok(pointsPerTrack(1000, 10) === 101);
});

test("buildFleetVerts: FULL = trace + curtain + line; THIN = one ribbon; no segment across a gap", () => {
  const [tr] = prepareFleet([hex("aaaaaa", 5, 60, { gaps: [T0 + 180] })], 60);
  const idx = decimateIdx(tr, 100);
  const full = buildFleetVerts(tr, idx, "full", 7, T0, 1);
  // 4 segments, one across the gap → 3 drawn × (trace + curtain + line) × 4 verts
  assert.equal(full.length / FLEET_STRIDE, 3 * 3 * 4);
  const thin = buildFleetVerts(tr, idx, "thin", 7, T0, 1);
  assert.equal(thin.length / FLEET_STRIDE, 3 * 4);
  // every vertex carries its fix time (relative) and its track slot
  for (let v = 0; v < thin.length / FLEET_STRIDE; v++) {
    const t = thin[v * FLEET_STRIDE + 13];
    assert.ok([0, 60, 120, 180, 240].includes(t), `t ${t}`);
    assert.equal(thin[v * FLEET_STRIDE + 14], 7);
  }
  // no altitude → THIN draws the ground trace, FULL draws trace only (no curtain)
  const [flat] = prepareFleet([hex("bbbbbb", 3, 60, { alt: null })], 60);
  assert.equal(buildFleetVerts(flat, [0, 1, 2], "full", 0, T0, 1).length / FLEET_STRIDE, 2 * 4);
  assert.equal(buildFleetVerts(flat, [0, 1, 2], "thin", 0, T0, 1).length / FLEET_STRIDE, 2 * 4);
  assert.equal(headColor(NaN).length, 3);
});
