// operator-resolution battery (BUILD ORDER 4 #1): prefix inference rules,
// trustee detection, and the fleet integration's callsign path.
import { test, beforeEach } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { resolveOperator, isTrusteeRegistrant, ICAO_OPERATORS } from "./operatorResolution";
import { buildFleetSeries, _resetFleetCache } from "./fleetUtilization";
import { _resetAircraftEntityCache } from "./aircraftEntities";

beforeEach(() => { _resetFleetCache(); _resetAircraftEntityCache(); });

test("resolveOperator: majority known prefix wins with agreement; edge cases stay null", () => {
  const r = resolveOperator(["UAL123", "UAL9", "UAL4567"]);
  assert.equal(r!.operator, "UNITED AIRLINES");
  assert.equal(r!.basis, "callsign-prefix");
  assert.equal(r!.agreement, 1);
  // mixed but majority
  const m = resolveOperator(["DAL1", "DAL2", "SKW5"]);
  assert.equal(m!.icao, "DAL");
  assert.equal(m!.agreement, +(2 / 3).toFixed(3));
  // below majority -> null
  assert.equal(resolveOperator(["DAL1", "SKW5"]), null);
  // single sample -> null (never resolve on one observation)
  assert.equal(resolveOperator(["UAL123"]), null);
  // unknown prefix -> null, never guessed
  assert.equal(resolveOperator(["ZZZ1", "ZZZ2"]), null);
  // bizjet-style callsigns (registration as callsign) -> null
  assert.equal(resolveOperator(["N549SC", "N549SC"]), null);
  assert.equal(resolveOperator([null, undefined, ""]), null);
});

test("isTrusteeRegistrant: shell patterns from our own spine composition", () => {
  for (const o of ["BANK OF UTAH TRUSTEE", "UMB BANK NA TRUSTEE", "WILMINGTON TRUST CO TRUSTEE",
                   "TVPX AIRCRAFT SOLUTIONS INC TRUSTEE", "SKYWEST LEASING INC"]) {
    assert.ok(isTrusteeRegistrant(o), o);
  }
  assert.ok(!isTrusteeRegistrant("SOUTHWEST AIRLINES CO"));
  assert.ok(!isTrusteeRegistrant("SPARTAN EDUCATION LLC"));
  assert.ok(!isTrusteeRegistrant(null));
});

test("fleet integration: trustee-registered hex flying airline callsigns aggregates under the OPERATOR", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "opres-"));
  fs.mkdirSync(path.join(base, "aircraft"), { recursive: true });
  const spineFp = path.join(base, "spine.json");
  fs.writeFileSync(spineFp, JSON.stringify({
    entities: {
      abc123: { n_number: "N1TRST", owner: "BANK OF UTAH TRUSTEE", registrant_type: "corporation" },
      ddd444: { n_number: "N2CORP", owner: "ACME JETS INC", registrant_type: "corporation" },
    },
  }));
  const t0 = Math.floor(Date.parse("2026-06-29T12:00:00Z") / 1000);
  fs.writeFileSync(path.join(base, "aircraft", "2026-06-29-12.jsonl"), [
    { t: t0, i: "abc123", c: "UAL100" }, { t: t0 + 600, i: "abc123", c: "UAL100" },
    { t: t0, i: "ddd444", c: "N2CORP" }, { t: t0 + 600, i: "ddd444" },
    { t: t0, i: "ffffff", c: "DLH400" }, { t: t0 + 600, i: "ffffff", c: "DLH401" }, // non-US hex, spine can't match
  ].map((l) => JSON.stringify(l)).join("\n") + "\n");
  const series = await buildFleetSeries(base, spineFp);
  const byOwner = Object.fromEntries(series.map((s) => [s.owner, s]));
  const ual = byOwner["UNITED AIRLINES"];
  assert.ok(ual, "trustee hex resolves to its operator, not the bank");
  assert.equal(ual.resolution.callsign, 1);
  assert.equal(ual.trustee_airframes, 1, "the trustee shadow is counted, not hidden");
  assert.ok(byOwner["LUFTHANSA"], "non-US hex resolves by callsign with no spine entry at all");
  assert.equal(byOwner["ACME JETS INC"].resolution.registrant, 1, "bizjet stays registrant-labeled");
  assert.ok(!byOwner["BANK OF UTAH TRUSTEE"], "the bank is not an operator series");
});

test("prefix table stays a stated subset (no wildcard guessing)", () => {
  assert.ok(Object.keys(ICAO_OPERATORS).length >= 20);
  for (const [k, v] of Object.entries(ICAO_OPERATORS)) {
    assert.match(k, /^[A-Z]{3}$/);
    assert.ok(v.length > 3);
  }
});
