// signalLadder.test.ts — MAP V2 ROADMAP R6(a) SIGNAL-STRENGTH dashboard.
import { test } from "node:test";
import assert from "node:assert/strict";
import { summarizeLadder, loadSignalLadder, type LadderRoot } from "./signalLadder";

function root(overrides: Partial<LadderRoot>): LadderRoot {
  return {
    id: "x", name: "X", category: "other", status: "raw_only",
    current_gate: 0, last_update_date: "2026-07-01", note: "n", source_ref: "s",
    ...overrides,
  };
}

test("summarizeLadder counts status/category/gate buckets correctly", () => {
  const roots: LadderRoot[] = [
    root({ id: "a", status: "raw_only", current_gate: 0, category: "environmental" }),
    root({ id: "b", status: "gate1_pass", current_gate: 1, category: "macro" }),
    root({ id: "c", status: "gate2_pending", current_gate: 2, category: "macro" }),
    root({ id: "d", status: "killed", current_gate: 1, category: "other" }),
  ];
  const s = summarizeLadder(roots);
  assert.equal(s.total, 4);
  assert.deepEqual(s.by_status, { raw_only: 1, gate1_pass: 1, gate2_pending: 1, killed: 1 });
  assert.deepEqual(s.by_category, { environmental: 1, macro: 2, other: 1 });
  assert.equal(s.killed_count, 1);
  assert.equal(s.raw_only_count, 1);
  assert.equal(s.gate_counts.length, 6);
  assert.equal(s.gate_counts[0].count, 1);
  assert.equal(s.gate_counts[1].count, 2); // gate1_pass (b) + killed-at-gate-1 (d)
  assert.equal(s.gate_counts[2].count, 1);
});

test("summarizeLadder: furthest_gate_reached only counts genuine passes, never a killed root's death-gate or a raw_only root", () => {
  // A root killed AT gate 3 must not make furthest_gate_reached read 3 —
  // dying at a gate is not the same as passing it. Only *_pass statuses count.
  const roots: LadderRoot[] = [
    root({ id: "a", status: "killed", current_gate: 3 }),
    root({ id: "b", status: "raw_only", current_gate: 0 }),
    root({ id: "c", status: "gate2_pass", current_gate: 2 }),
    root({ id: "d", status: "gate1_pass", current_gate: 1 }),
  ];
  const s = summarizeLadder(roots);
  assert.equal(s.furthest_gate_reached, 2);
});

test("summarizeLadder on an empty registry is all zeros, never crashes", () => {
  const s = summarizeLadder([]);
  assert.equal(s.total, 0);
  assert.equal(s.killed_count, 0);
  assert.equal(s.raw_only_count, 0);
  assert.equal(s.furthest_gate_reached, 0);
  assert.deepEqual(s.by_status, {});
  assert.deepEqual(s.by_category, {});
});

test("loadSignalLadder reads the real committed registry: every root has required fields, ids are unique, source_ref is never empty", () => {
  const { roots, summary, compiled, sources } = loadSignalLadder();
  assert.ok(roots.length > 10, "expected a substantial compiled registry");
  assert.ok(compiled.length > 0);
  assert.ok(sources.length > 0);
  const seen = new Set<string>();
  for (const r of roots) {
    assert.ok(!seen.has(r.id), `duplicate root id ${r.id}`);
    seen.add(r.id);
    assert.ok(r.name.length > 0, `${r.id} missing name`);
    assert.ok(r.source_ref.length > 0, `${r.id} missing source_ref — every claim must be re-checkable`);
    assert.ok(r.current_gate >= 0 && r.current_gate <= 5, `${r.id} gate out of range`);
    assert.ok(r.note.length > 0, `${r.id} missing note`);
  }
  assert.equal(summary.total, roots.length);
});

test("loadSignalLadder: raw_only roots always carry current_gate 0 — the ladder doesn't apply to display-only overlays", () => {
  const { roots } = loadSignalLadder();
  for (const r of roots) {
    if (r.status === "raw_only") {
      assert.equal(r.current_gate, 0, `${r.id} is raw_only but has a nonzero gate`);
    }
  }
});
