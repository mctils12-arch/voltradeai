import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAirports, nearestAirport, haversineKm, _resetAirports } from "./airportsIndex";
import { splitTrips, endpointAgreesWithField, type ArchivedFix, type NearestAirportFn } from "./aircraftTrips";

// ── the real dataset (bundled, 72k fields) ──────────────────────────────────

test("loadAirports: the bundled catalog loads and finds real fields at their reference points", () => {
  _resetAirports();
  const n = loadAirports();
  assert.ok(n > 60_000, `expected the full catalog, got ${n}`);
  const jfk = nearestAirport(40.6413, -73.7781, 6, ["L", "M", "S", "W"]);
  assert.ok(jfk, "JFK must resolve");
  assert.equal(jfk!.id, "KJFK");
  assert.ok(jfk!.el != null && Math.abs(jfk!.el - 4) < 10, `JFK elevation ~4m, got ${jfk!.el}`);
  // mid-Atlantic: no airport within 6km
  assert.equal(nearestAirport(38.0, -40.0, 6), null);
  // type filter: fixed-wing lookup must not land at a heliport by accident
  const anyType = nearestAirport(40.070985, -74.933689, 2); // Total RF Heliport's own coords
  assert.equal(anyType?.ty, "H");
  const fixedWing = nearestAirport(40.070985, -74.933689, 2, ["L", "M", "S", "W"]);
  assert.ok(!fixedWing || fixedWing.ty !== "H");
});

test("haversineKm sanity: JFK->LGA ~17km", () => {
  const d = haversineKm(40.6413, -73.7781, 40.7769, -73.8740);
  assert.ok(Math.abs(d - 17) < 2, `got ${d}`);
});

// ── endpoint agreement (pure) ───────────────────────────────────────────────

test("endpointAgreesWithField: grounded agrees; altitude within tolerance agrees; cruise over a field does NOT", () => {
  const ap = { el: 190 };
  assert.equal(endpointAgreesWithField({ t: 0, la: 0, lo: 0, g: true } as ArchivedFix, ap), true);
  assert.equal(endpointAgreesWithField({ t: 0, la: 0, lo: 0, al: 400 } as ArchivedFix, ap), true, "within 350m tolerance");
  assert.equal(endpointAgreesWithField({ t: 0, la: 0, lo: 0, al: 3000 } as ArchivedFix, ap), false, "3km over the field is not a landing");
  assert.equal(endpointAgreesWithField({ t: 0, la: 0, lo: 0, al: 3000 } as ArchivedFix, null), false, "no airport = unverified");
  assert.equal(endpointAgreesWithField({ t: 0, la: 0, lo: 0, al: 3000 } as ArchivedFix, { el: null }), true, "unpublished elevation cannot disagree");
});

// ── verified flights end-to-end through splitTrips ──────────────────────────

const fix = (t: number, extra: Partial<ArchivedFix> = {}): ArchivedFix =>
  ({ t, la: 40, lo: -95, al: 9000, ...extra });

test("splitTrips + airports: a ground-to-ground flight between two catalog fields is VERIFIED", () => {
  const nearest: NearestAirportFn = (la) =>
    la < 40.5 ? { id: "KAAA", n: "Field A", dist_km: 1.2, el: 200 }
              : { id: "KBBB", n: "Field B", dist_km: 0.8, el: 350 };
  const fixes = [
    fix(0, { la: 40.0, g: true, al: null }), fix(60, { la: 40.2, al: 5000 }),
    fix(120, { la: 40.4, al: 9000 }), fix(180, { la: 40.8, al: 700 }), fix(240, { la: 40.9, g: true, al: null }),
  ];
  const trips = splitTrips(fixes, undefined, undefined, undefined, nearest);
  assert.equal(trips.length, 1);
  assert.equal(trips[0].quality, "complete");
  assert.equal(trips[0].verified, true);
  assert.equal(trips[0].is_flight, true);
  assert.equal(trips[0].from_airport?.id, "KAAA");
  assert.equal(trips[0].to_airport?.id, "KBBB");
});

test("splitTrips + airports: complete shape but NO catalog field nearby stays unverified (honest)", () => {
  const nearest: NearestAirportFn = () => null;
  const fixes = [
    fix(0, { g: true, al: null }), fix(60, { al: 5000 }), fix(120, { al: 9000 }), fix(180, { g: true, al: null }),
  ];
  const trips = splitTrips(fixes, undefined, undefined, undefined, nearest);
  assert.equal(trips[0].quality, "complete");
  assert.equal(trips[0].verified, false, "no field within radius — private strip or coverage cut; never claimed verified");
  assert.equal(trips[0].is_flight, true);
});

test("splitTrips: taxi logs are is_flight=false — 'that's not a flight'", () => {
  const fixes = [fix(0, { g: true, al: null }), fix(60, { g: true, al: null }), fix(120, { g: true, al: null })];
  const trips = splitTrips(fixes);
  assert.equal(trips[0].is_flight, false);
  assert.equal(trips[0].verified, false);
});
