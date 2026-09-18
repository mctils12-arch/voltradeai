// NHTSA complaints battery (BUILD ORDER 6 #4): curated-seed sanity,
// MM/DD/YYYY normalization, ODI event-identity dedup across fetches and
// restarts, politeness-spaced sweep, summaries-not-archived pin.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  VEHICLES, parseComplaints, normalizeUsDate, fetchVehicleComplaints,
  archiveNewComplaints, refreshComplaints, latestComplaintStats,
  computeVehicleStats, readArchivedComplaints, _resetComplaintsForTests,
} from "./nhtsaComplaints";

const V = { ticker: "TSLA", make: "tesla", model: "model 3", modelYear: 2024 };

// Mirrors the live shape verified 2026-07-06.
const RESULT = (odi: number, filed: string, crash: boolean, fire: boolean) => ({
  odiNumber: odi, manufacturer: "Tesla, Inc.", crash, fire,
  numberOfInjuries: 0, dateComplaintFiled: filed, dateOfIncident: filed,
  components: "VEHICLE SPEED CONTROL", summary: "free text NOT archived",
});

test("watchlist seed: ticker-mapped, recent model years, bounded size", () => {
  assert.ok(VEHICLES.length >= 15 && VEHICLES.length <= 40,
    "one API call per vehicle per cycle — the list must stay bounded");
  for (const v of VEHICLES) {
    assert.ok(v.ticker && v.make && v.model, JSON.stringify(v));
    assert.ok(v.modelYear >= 2023, "complaint VELOCITY needs current product, not history");
  }
});

test("normalizeUsDate + parse: dates normalized, flags land, summaries dropped", () => {
  assert.equal(normalizeUsDate("06/26/2026"), "2026-06-26");
  assert.equal(normalizeUsDate("bad"), null);
  const events = parseComplaints({ count: 2, results: [
    RESULT(11746845, "06/26/2026", false, false),
    RESULT(11746042, "06/23/2026", true, false),
  ] }, V, "2026-07-06");
  assert.equal(events.length, 2);
  assert.equal(events[0].filed, "2026-06-26");
  assert.equal(events[0].ticker, "TSLA");
  assert.equal(events[1].crash, true);
  assert.ok(!("summary" in events[0]),
    "free-text summaries stay out of the archive (bulk-file follow-up owns them)");
  assert.deepEqual(parseComplaints(null, V, "x"), []);
  assert.deepEqual(parseComplaints({ results: [{ odiNumber: "not-a-number" }] }, V, "x"), []);
});

test("fetch: non-200 -> []; url carries encoded make/model + year", async () => {
  const urls: string[] = [];
  const bad = async (url: string) => { urls.push(url); return { ok: false, status: 500, text: async () => "" }; };
  assert.deepEqual(await fetchVehicleComplaints(V, bad as any), []);
  assert.ok(urls[0].includes("make=tesla") && urls[0].includes("model=model%203") && urls[0].includes("modelYear=2024"));
});

test("archive: ODI event-identity dedup within and across fetches", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "nhtsa-"));
  const now = Date.parse("2026-07-06T12:00:00Z");
  const batch = parseComplaints({ results: [RESULT(1, "06/26/2026", false, false), RESULT(2, "06/23/2026", true, false)] }, V, "2026-07-06");
  assert.equal(archiveNewComplaints(batch, base, now), 2);
  assert.equal(archiveNewComplaints(batch, base, now), 0, "same ODIs never re-archive");
  const plus = [...batch, ...parseComplaints({ results: [RESULT(3, "07/01/2026", false, true)] }, V, "2026-07-06")];
  assert.equal(archiveNewComplaints(plus, base, now), 1, "only the new ODI lands");
});

test("refresh sweep: per-vehicle stats cached; spacing=0 keeps the test fast", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "nhtsa-"));
  let calls = 0;
  const ok = async () => {
    calls++;
    return { ok: true, status: 200,
             text: async () => JSON.stringify({ count: 1, results: [RESULT(1000 + calls, "07/01/2026", calls % 2 === 0, false)] }) };
  };
  await refreshComplaints(ok as any, Date.parse("2026-07-06T12:00:00Z"), base, 0);
  assert.equal(calls, VEHICLES.length, "one call per watchlist vehicle");
  const hit = latestComplaintStats();
  assert.ok(hit);
  assert.equal(hit!.stats.length, VEHICLES.length);
  assert.ok(hit!.stats.every((s) => s.total_complaints === 1 && s.newest_filed === "2026-07-01"));
});

test("computeVehicleStats: groups a flat event list back into per-vehicle stats (pure, reused by both the live sweep and the disk backfill)", () => {
  const events = [
    ...parseComplaints({ results: [RESULT(1, "06/26/2026", false, false), RESULT(2, "06/23/2026", true, false)] }, V, "2026-07-06"),
    ...parseComplaints({ results: [RESULT(3, "07/01/2026", false, true)] },
      { ticker: "GM", make: "chevrolet", model: "bolt", modelYear: 2024 }, "2026-07-06"),
  ];
  const stats = computeVehicleStats(events);
  assert.equal(stats.length, 2, "two distinct vehicles in the flat event list");
  const tsla = stats.find((s) => s.ticker === "TSLA")!;
  assert.equal(tsla.total_complaints, 2);
  assert.equal(tsla.crash_count, 1);
  assert.equal(tsla.newest_filed, "2026-06-26");
  const gm = stats.find((s) => s.ticker === "GM")!;
  assert.equal(gm.total_complaints, 1);
  assert.equal(gm.fire_count, 1);
});

test("refreshComplaints: cold cache backfills from disk when the ENTIRE watchlist's live sweep fails — same class of fix as euLoad.ts's/satellites.ts's cold-cache backfill (an NHTSA outage or a fresh boot must not report warming_up over real archived complaint history)", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "nhtsa-coldcache-"));
  _resetComplaintsForTests();
  try {
    // Real archived history from before the simulated restart.
    const prior = parseComplaints({ results: [RESULT(9001, "06/20/2026", true, false)] }, V, "2026-06-20");
    assert.equal(archiveNewComplaints(prior, base, Date.parse("2026-06-20T12:00:00Z")), 1);
    _resetComplaintsForTests(); // simulated restart — in-memory dedup/cache gone
    assert.equal(latestComplaintStats(), null, "cache must still be cold going into this cycle");
    const failing = async () => { throw new Error("network unreachable"); };
    await refreshComplaints(failing as any, Date.parse("2026-07-06T12:00:00Z"), base, 0);
    const hit = latestComplaintStats();
    assert.ok(hit, "cache must be populated, not left null, despite every vehicle's live fetch failing");
    assert.equal(hit!.stats.length, 1, "only TSLA has archived history — never zero-filled for the rest of the watchlist");
    assert.equal(hit!.stats[0].ticker, "TSLA");
    assert.equal(hit!.stats[0].total_complaints, 1);
    assert.equal(hit!.stats[0].crash_count, 1);
    const archived = readArchivedComplaints(base, Date.parse("2026-07-06T12:00:00Z"), 30);
    assert.equal(archived.length, 1, "readArchivedComplaints itself returns the raw archived row the backfill used");
    assert.equal(archived[0].odi, 9001);
  } finally {
    _resetComplaintsForTests();
  }
});
