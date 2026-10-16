import { test } from "node:test";
import assert from "node:assert/strict";
import { Director } from "./director";

test("first state applies at once; a change waits the minimum dwell", () => {
  const d = new Director({ minDwellMs: 8_000, recedeGraceMs: 20_000 });
  assert.equal(d.propose("idle", 0), "idle");
  assert.equal(d.propose("scanning", 1_000), "idle");
  assert.equal(d.propose("scanning", 7_999), "idle");
  assert.equal(d.propose("scanning", 8_000), "scanning");
});

test("an incident takes the layout immediately and recedes only after the grace", () => {
  const d = new Director({ minDwellMs: 8_000, recedeGraceMs: 20_000 });
  d.propose("idle", 0);
  assert.equal(d.propose("incident", 100), "incident");
  assert.equal(d.propose("idle", 9_000), "incident", "dwell met but recede grace not");
  assert.equal(d.propose("idle", 28_999), "incident", "grace counts from when the resolution was first seen (9 s)");
  assert.equal(d.propose("idle", 29_000), "idle");
  assert.equal(d.log.at(-1)?.why, "incident receded");
});

test("a flapping candidate restarts its own clock; hold freezes the director but not the data", () => {
  const d = new Director({ minDwellMs: 1_000, recedeGraceMs: 5_000 });
  d.propose("incident", 0);
  d.propose("idle", 1_000);
  d.propose("candidate", 3_000);      // candidate changed → its clock restarts
  assert.equal(d.propose("candidate", 7_000), "incident");
  assert.equal(d.propose("candidate", 8_000), "candidate");
  assert.equal(d.propose("scanning", 20_000, true), "candidate", "held");
  assert.equal(d.propose("scanning", 21_000), "scanning");
});
