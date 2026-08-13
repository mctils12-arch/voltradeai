import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import os from "os";
import path from "path";
import {
  parseFireballs, groundHeading, mergeEvents, eventKey,
  loadStore, saveStore, METEORS_POLL_MS,
} from "./meteors";

// real CNEOS rows (probed live 2026-08-13) — the wire format, verbatim
const FIXTURE = {
  signature: { version: "1.2", source: "NASA/JPL Fireball Data API" },
  count: "4",
  fields: ["date", "energy", "impact-e", "lat", "lat-dir", "lon", "lon-dir", "alt", "vel", "vx", "vy", "vz"],
  data: [
    ["2026-08-01 17:43:48", "2.9", "0.1", "19.5", "S", "176.2", "E", "45.0", null, null, null, null],
    ["2026-05-21 02:15:51", "3.5", "0.12", "46.6", "N", "133.1", "E", "31.5", "15.8", "14.4", "-6.5", "1.0"],
    ["2026-05-30 18:06:23", "42.1", "1.1", "42.0", "N", "70.5", "W", "32.0", null, null, null, null],
    ["2025-08-19 14:08:48", "65.8", "1.6", "30.9", "N", "131.8", "E", "27.5", "19.6", "-13.6", "10.6", "9.5"],
  ],
};

test("parseFireballs: real wire rows → events; direction only where the vector exists", () => {
  const ev = parseFireballs(FIXTURE);
  assert.equal(ev.length, 4);
  assert.equal(ev[0].date, "2026-08-01 17:43:48", "sorted newest-first");
  const aug = ev[0];
  assert.equal(aug.la, -19.5, "S → negative lat");
  assert.equal(aug.lo, 176.2);
  assert.equal(aug.vel, null, "unpublished speed stays null");
  assert.equal(aug.hdg, null, "no vector → no heading, no invented direction");
  const may = ev.find((e) => e.date.startsWith("2026-05-21"))!;
  assert.ok(may.hdg != null && may.hdg >= 0 && may.hdg < 360);
  const w = ev.find((e) => e.date.startsWith("2026-05-30"))!;
  assert.equal(w.lo, -70.5, "W → negative lon");
  assert.equal(w.imp, 1.1);
});

test("groundHeading: pure-east and pure-north vectors at the equator/prime meridian", () => {
  // at (0,0): east = +Y, north = +Z in the Earth-fixed frame
  assert.ok(Math.abs(groundHeading(0, 1, 0, 0, 0) - 90) < 1e-9, "east");
  assert.ok(Math.abs(groundHeading(0, 0, 1, 0, 0) - 0) < 1e-9, "north");
  assert.ok(Math.abs(groundHeading(0, -1, 0, 0, 0) - 270) < 1e-9, "west");
  // second analytic case: at (0°N, 90°E) local east = −X in the fixed frame
  assert.ok(Math.abs(groundHeading(-1, 0, 0, 0, 90) - 90) < 1e-9, "east at 90E");
  // consistency pin for the real 2026-05-21 vector (46.6N 133.1E) — value
  // produced by this math, pinned so refactors can't silently change it
  const h = groundHeading(14.4, -6.5, 1.0, 46.6, 133.1);
  assert.ok(Math.abs(h - 331.7) < 0.1, `actual ${h.toFixed(1)}`);
});

test("mergeEvents: same event across polls dedupes, revised poll wins, order newest-first", () => {
  const a = parseFireballs(FIXTURE);
  const revised = { ...a[0], imp: 0.2 };
  const merged = mergeEvents(a, [revised]);
  assert.equal(merged.length, 4);
  assert.equal(merged.find((e) => eventKey(e) === eventKey(revised))!.imp, 0.2);
  for (let i = 1; i < merged.length; i++) assert.ok(merged[i - 1].t >= merged[i].t);
});

test("store round-trips through the volume path", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "vt-meteors-"));
  const ev = parseFireballs(FIXTURE);
  saveStore(ev, dir);
  const back = loadStore(dir);
  assert.deepEqual(back, ev);
  assert.deepEqual(loadStore(fs.mkdtempSync(path.join(os.tmpdir(), "vt-meteors-empty-"))), [],
    "absent store → empty, never a throw");
  fs.rmSync(dir, { recursive: true, force: true });
});

test("poll cadence is provider-polite (4×/day for a ~weekly-event feed)", () => {
  assert.equal(METEORS_POLL_MS, 21_600_000);
});
