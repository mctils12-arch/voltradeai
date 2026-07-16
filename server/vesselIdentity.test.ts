// IMO-based vessel identity resolution — hermetic tmpdir tests (GIP BUILD
// QUEUE item 3, research/open_questions.md ~line 4105).
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  isValidImo, recordIdentityObservation, readIdentityRegistry,
  detectImoReuse, imoIdentityStats, clearIdentityCache,
} from "./vesselIdentity";

function tmpBase(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), "vt-vessel-identity-"));
}

test("isValidImo: known real IMO numbers pass checksum validation", () => {
  // 9074729 = Ever Given; 9319466 = MSC Oscar — both real, publicly
  // documented IMO numbers, independently checksum-verifiable by hand:
  // 9*7+0*6+7*5+4*4+7*3+2*2 = 63+0+35+16+21+4 = 139 -> 139 % 10 = 9 (matches).
  assert.equal(isValidImo(9074729), true);
  assert.equal(isValidImo("9074729"), true);
  assert.equal(isValidImo(9319466), true);
});

test("isValidImo: rejects bad checksum, wrong length, zero, and empty", () => {
  assert.equal(isValidImo(9074728), false);   // wrong check digit
  assert.equal(isValidImo(123456), false);    // 6 digits
  assert.equal(isValidImo(12345678), false);  // 8 digits
  assert.equal(isValidImo("0000000"), false); // AIS "no IMO assigned"
  assert.equal(isValidImo(0), false);
  assert.equal(isValidImo(null), false);
  assert.equal(isValidImo(undefined), false);
  assert.equal(isValidImo(""), false);
});

test("recordIdentityObservation: rejects invalid IMO, no registry file created", () => {
  const base = tmpBase();
  clearIdentityCache();
  const wrote = recordIdentityObservation({ mmsi: "111000111", imo: 9074728 }, base);
  assert.equal(wrote, false);
  assert.equal(fs.existsSync(path.join(base, "vessel_identity", "registry.jsonl")), false);
});

test("recordIdentityObservation: records a new (mmsi, imo) pair once, dedupes repeats", () => {
  const base = tmpBase();
  clearIdentityCache();
  const first = recordIdentityObservation(
    { mmsi: "111000111", imo: 9074729, callsign: "H3RC", name: "EVER GIVEN" }, base, 1_000_000);
  assert.equal(first, true);
  const dup = recordIdentityObservation({ mmsi: "111000111", imo: 9074729 }, base, 2_000_000);
  assert.equal(dup, false, "the exact same (mmsi, imo) pair must not be re-appended");
  const registry = readIdentityRegistry(base);
  assert.equal(registry.length, 1);
  assert.equal(registry[0].mmsi, "111000111");
  assert.equal(registry[0].imo, 9074729);
  assert.equal(registry[0].callsign, "H3RC");
  assert.equal(registry[0].name, "EVER GIVEN");
});

test("recordIdentityObservation: dedupe survives a process restart (reads disk on first use)", () => {
  const base = tmpBase();
  clearIdentityCache();
  recordIdentityObservation({ mmsi: "111000111", imo: 9074729 }, base, 1_000_000);
  clearIdentityCache(); // simulate a fresh process — cache must reload from disk, not re-append
  const dup = recordIdentityObservation({ mmsi: "111000111", imo: 9074729 }, base, 2_000_000);
  assert.equal(dup, false);
  assert.equal(readIdentityRegistry(base).length, 1);
});

test("recordIdentityObservation: the SAME mmsi with a DIFFERENT valid imo is a new pair", () => {
  const base = tmpBase();
  clearIdentityCache();
  recordIdentityObservation({ mmsi: "111000111", imo: 9074729 }, base, 1_000_000);
  const second = recordIdentityObservation({ mmsi: "111000111", imo: 9319466 }, base, 2_000_000);
  assert.equal(second, true);
  assert.equal(readIdentityRegistry(base).length, 2);
});

test("detectImoReuse: an IMO under 2+ distinct MMSIs is a reuse candidate; a single-MMSI IMO is not", () => {
  const records = [
    { t: 100, mmsi: "AAA", imo: 9074729, name: "EVER GIVEN" },
    { t: 200, mmsi: "BBB", imo: 9074729, name: "EVERGREEN X" }, // same hull, reflagged/renamed
    { t: 150, mmsi: "CCC", imo: 9319466, name: "MSC OSCAR" },   // never reused — no candidate
  ];
  const out = detectImoReuse(records);
  assert.equal(out.length, 1);
  assert.equal(out[0].imo, 9074729);
  assert.deepEqual(out[0].mmsis, ["AAA", "BBB"], "must be ordered oldest-first by observation time");
  assert.equal(out[0].firstSeen, 100);
  assert.equal(out[0].lastSeen, 200);
  assert.deepEqual(out[0].names.sort(), ["EVER GIVEN", "EVERGREEN X"]);
});

test("detectImoReuse: sorts multiple candidates by most-recently-seen first", () => {
  const records = [
    { t: 100, mmsi: "A1", imo: 9074729 }, { t: 500, mmsi: "A2", imo: 9074729 },
    { t: 100, mmsi: "B1", imo: 9319466 }, { t: 900, mmsi: "B2", imo: 9319466 },
  ];
  const out = detectImoReuse(records);
  assert.equal(out.length, 2);
  assert.equal(out[0].imo, 9319466, "lastSeen=900 sorts before lastSeen=500");
  assert.equal(out[1].imo, 9074729);
});

test("imoIdentityStats: reads the registry fresh from disk and caps examples", () => {
  const base = tmpBase();
  clearIdentityCache();
  recordIdentityObservation({ mmsi: "AAA", imo: 9074729 }, base, 100_000);
  recordIdentityObservation({ mmsi: "BBB", imo: 9074729 }, base, 200_000);
  const stats = imoIdentityStats(base, 1);
  assert.equal(stats.imo_confirmed_identity_changes, 1);
  assert.equal(stats.imo_confirmed_examples.length, 1);
  assert.equal(stats.imo_confirmed_examples[0].imo, 9074729);
});

test("imoIdentityStats: empty registry (no observations yet) reports zero, not an error", () => {
  const base = tmpBase();
  clearIdentityCache();
  const stats = imoIdentityStats(base);
  assert.equal(stats.imo_confirmed_identity_changes, 0);
  assert.deepEqual(stats.imo_confirmed_examples, []);
});
