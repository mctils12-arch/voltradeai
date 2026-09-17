import { test } from "node:test";
import assert from "node:assert/strict";
import { sortPortsByActivity, filterPorts, totalAnomalies } from "./portDwellView";

const base = (over: Partial<{ name: string; in_port_now: number; visits_completed: number; anomaly_count: number }> = {}) => ({
  name: "Port", in_port_now: 0, visits_completed: 0, anomaly_count: 0, ...over,
});

test("sortPortsByActivity puts higher in_port_now first regardless of input order", () => {
  const ports = [base({ name: "B", in_port_now: 1 }), base({ name: "A", in_port_now: 5 })];
  const sorted = sortPortsByActivity(ports);
  assert.deepEqual(sorted.map((p) => p.name), ["A", "B"]);
});

test("sortPortsByActivity breaks an in_port_now tie by higher visits_completed", () => {
  const ports = [
    base({ name: "Low", in_port_now: 2, visits_completed: 3 }),
    base({ name: "High", in_port_now: 2, visits_completed: 9 }),
  ];
  const sorted = sortPortsByActivity(ports);
  assert.deepEqual(sorted.map((p) => p.name), ["High", "Low"]);
});

test("sortPortsByActivity breaks a full tie alphabetically by name", () => {
  const ports = [base({ name: "Zebra" }), base({ name: "Alpha" })];
  const sorted = sortPortsByActivity(ports);
  assert.deepEqual(sorted.map((p) => p.name), ["Alpha", "Zebra"]);
});

test("sortPortsByActivity never mutates the input array", () => {
  const ports = [base({ name: "B", in_port_now: 1 }), base({ name: "A", in_port_now: 5 })];
  const original = ports.slice();
  sortPortsByActivity(ports);
  assert.deepEqual(ports, original);
});

test("filterPorts 'active' keeps only ports with vessels in port now", () => {
  const ports = [base({ name: "A", in_port_now: 0 }), base({ name: "B", in_port_now: 2 })];
  assert.deepEqual(filterPorts(ports, "active").map((p) => p.name), ["B"]);
});

test("filterPorts 'anomaly' keeps only ports with a nonzero anomaly count", () => {
  const ports = [base({ name: "A", anomaly_count: 0 }), base({ name: "B", anomaly_count: 1 })];
  assert.deepEqual(filterPorts(ports, "anomaly").map((p) => p.name), ["B"]);
});

test("filterPorts 'all' returns every port unchanged", () => {
  const ports = [base({ name: "A" }), base({ name: "B" })];
  assert.deepEqual(filterPorts(ports, "all"), ports);
});

test("totalAnomalies sums anomaly_count across all ports, zero on empty input", () => {
  assert.equal(totalAnomalies([base({ anomaly_count: 2 }), base({ anomaly_count: 1 })]), 3);
  assert.equal(totalAnomalies([]), 0);
});
