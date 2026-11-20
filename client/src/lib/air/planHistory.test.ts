// planHistory.test — the flight card's "Plan changes" read, on the shape the
// server answers with (trimmed from a live FAA-filed flight, 2026-11-20: four
// versions, three changes) and on the shapes the card must survive.
import test from "node:test";
import assert from "node:assert/strict";
import { parsePlanHistory, planChangesHeadline, changeWords, originNote, offRouteLine, positionWords, zulu, fetchPlanHistory, type FetchLike } from "./planHistory.ts";

const T = Date.UTC(2026, 10, 20, 21, 0, 0);
const geo = (changed: boolean, nm: number | null) => ({ changed, firstDifferingLeg: null, maxLateralNm: nm, lateralEstimated: false, note: null });
const LIVE = {
  hex: "a12733", callsign: "UAL643", kind: "raw", source: "FILED_FAA", label: "FILED — 4 versions of the FAA SWIM flight plan KSFO→KSNA", honesty: "Flight plan versions for UAL643 from FAA SWIM SFDPS, matched to this aircraft by callsign.",
  plan: { planFirstSeenMs: T - 3600_000, originalAsFiled: false, originalIsFirstSeen: true, originalNote: "version 1 is the plan as we FIRST SAW it", versionsReturned: 4, versionsMissing: 0, ended: false },
  original: { version: 1, departure: "KSFO", arrival: "KSNA", cruiseAltFt: 33000 },
  current: { version: 4, departure: "KSFO", arrival: "KSNA", cruiseAltFt: 29000 },
  versions: [{ version: 1 }, { version: 2 }, { version: 3 }, { version: 4 }],
  amendments: [
    { version: 2, fromVersion: 1, at: T + 7 * 60_000, atBasis: "SWIM_MESSAGE_TIME", detail: "route text changed", kinds: ["ROUTE_TEXT_ONLY"], change: { kinds: ["ROUTE_TEXT_ONLY"], fixesAdded: [], fixesRemoved: [], destination: null, cruiseAltFt: null, directTo: null, geometry: geo(false, 0) }, spansMissingVersions: false, position: null, positionReason: "the nearest ADS-B fix is 1998 s from that time (tolerance 30 s)" },
    { version: 3, fromVersion: 2, at: T + 20 * 60_000, atBasis: "SWIM_MESSAGE_TIME", detail: "cruise 33000→29000 ft", kinds: ["ALTITUDE_CHANGE"], change: { kinds: ["ALTITUDE_CHANGE"], fixesAdded: [], fixesRemoved: [], destination: null, cruiseAltFt: { from: 33000, to: 29000, deltaFt: -4000 }, directTo: null, geometry: geo(false, null) }, spansMissingVersions: false, position: { lat: 36.1, lon: -120.4, altFt: 33000, method: "NEAREST", interpolated: false, fixTimeMs: T + 20 * 60_000 - 9000, gapSec: 9, between: null }, positionReason: null },
    { version: 4, fromVersion: 3, at: T + 28 * 60_000, atBasis: "OUR_RECEIVE_TIME", detail: "route text changed; expanded route changed", kinds: ["REROUTE", "FIXES_REMOVED", "DIRECT_TO"], change: { kinds: ["REROUTE", "FIXES_REMOVED", "DIRECT_TO"], fixesAdded: [], fixesRemoved: ["ZIPRR", "WURLL"], destination: null, cruiseAltFt: null, directTo: { fix: "TCUPS" }, geometry: geo(true, 12.4) }, spansMissingVersions: false, position: null, positionReason: "no fix near that time" },
  ],
  restatements: [],
  offRoute: { assessed: true, reason: null, fixesJudged: 41, episodes: [] },
  generatedAt: T + 30 * 60_000,
};

test("a filed plan that changed: newest change first, each in plain words, the headline quotes the newest", () => {
  const v = parsePlanHistory(LIVE)!;
  assert.equal(v.state, "changes");
  assert.deepEqual([v.versions, v.changes.length, v.departure, v.arrival, v.asFiled], [4, 3, "KSFO", "KSNA", false]);
  assert.deepEqual(v.changes.map((c) => [c.version, c.words, c.atBasis]), [
    [4, "direct to TCUPS · dropped ZIPRR, WURLL", "received"],
    [3, "cruise FL330 → FL290", "message"],
    [2, "route text reworded, same path", "message"],
  ]);
  assert.equal(v.changes[0].lateralNm, 12.4);
  assert.equal(v.changes[1].lateralNm, null, "an unchanged or unmeasured geometry is not a 0 nm move");
  assert.equal(planChangesHeadline(v), "4 plan versions · 3 changes · latest 21:28Z: direct to TCUPS · dropped ZIPRR, WURLL");
  assert.equal(originNote(v), "Version 1 is the plan as we first saw it, not necessarily as filed.");
  assert.equal(positionWords(v.changes[1]), "aircraft position: nearest recorded position");
  assert.equal(positionWords(v.changes[0]), "aircraft position at that moment not known");
  assert.equal(offRouteLine(v, (nm) => `${nm} nm`), "No off-route stretch in the 41 positions we hold for this flight.");
  assert.equal(zulu(T), "21:00Z");
});

test("change words: destination, altitude below the flight levels, shortcut, added fixes, a long list, an unknown kind", () => {
  assert.equal(changeWords(["DESTINATION_CHANGE"], { destination: { from: "KSNA", to: "KLAX" } }, null), "destination KSNA → KLAX");
  assert.equal(changeWords(["ALTITUDE_CHANGE"], { cruiseAltFt: { from: 12000, to: 9000 } }, null), "cruise 12,000 ft → 9,000 ft");
  assert.equal(changeWords(["ALTITUDE_CHANGE"], {}, null), "cruise altitude changed");
  assert.equal(changeWords(["SHORTCUT", "FIXES_REMOVED"], { fixesRemoved: ["A", "B", "C", "D", "E", "F"] }, null), "shortcut · dropped A, B, C, D +2 more");
  assert.equal(changeWords(["REROUTE", "FIXES_ADDED"], { fixesAdded: ["ROKNE"] }, null), "rerouted · added ROKNE");
  assert.equal(changeWords(["DIRECT_TO"], { directTo: { fix: null } }, null), "direct from its position");
  assert.equal(changeWords([], null, "route text changed"), "route text changed");
  assert.equal(changeWords([], null, null), "changed");
});

test("the states the card must tell apart: one version, restatements only, no filed plan, withheld, off-route", () => {
  const single = parsePlanHistory({ ...LIVE, versions: [{ version: 1 }], amendments: [], plan: { ...LIVE.plan, originalAsFiled: true } })!;
  assert.equal(single.state, "single");
  assert.equal(originNote(single), "Version 1 is the plan as filed.");
  const restated = parsePlanHistory({ ...LIVE, versions: [{ version: 1 }, { version: 2 }], amendments: [], restatements: [{ version: 2, fromVersion: 1 }] })!;
  assert.equal(restated.state, "changes");
  assert.equal(planChangesHeadline(restated), "2 plan versions · no change to the route ahead · 1 restatement");
  assert.equal(parsePlanHistory({ source: "NONE", label: "No FAA-filed plan", versions: [], amendments: [] })!.state, "none");
  assert.equal(parsePlanHistory({ source: "NONE", privacy: "LADD", versions: [], amendments: [] })!.state, "withheld");
  const off = parsePlanHistory({ ...LIVE, offRoute: { assessed: true, reason: null, fixesJudged: 60, episodes: [{ startMs: T, endMs: T + 240_000, maxCrossTrackNm: 11.2, classification: "AMENDMENT_FOLLOWED" }, { startMs: T + 600_000, endMs: null, maxCrossTrackNm: 14.6, classification: "ONGOING" }] } })!;
  assert.equal(offRouteLine(off, (nm) => `${nm.toFixed(1)} nm`), "2 off-route stretches, up to 14.6 nm from the route. No cause is given.");
  const unjudged = parsePlanHistory({ ...LIVE, offRoute: { assessed: false, reason: "the path is a great-circle stand-in", fixesJudged: 0, episodes: [] } })!;
  assert.equal(offRouteLine(unjudged, String), "Off-route stretches not judged: the path is a great-circle stand-in.");
  const nofix = parsePlanHistory({ ...LIVE, offRoute: { assessed: true, reason: null, fixesJudged: 0, episodes: [] } })!;
  assert.equal(offRouteLine(nofix, String), "No position of this flight could be judged against its route.");
  const missing = parsePlanHistory({ ...LIVE, plan: { ...LIVE.plan, versionsMissing: 3 } })!;
  assert.ok(planChangesHeadline(missing).startsWith("4 plan versions (+3 not held) · 3 changes"));
});

test("another shape is null (the card shows nothing); a malformed amendment is dropped, not rendered", () => {
  for (const bad of [null, undefined, [], {}, { error: "x" }, "<!doctype html>", { source: "PREDICTED" }]) assert.equal(parsePlanHistory(bad), null);
  const v = parsePlanHistory({ ...LIVE, amendments: [null, { version: 2 }, { version: 2, fromVersion: 1, at: "soon" }, LIVE.amendments[1]] })!;
  assert.equal(v.changes.length, 1);
});

test("fetchPlanHistory: one GET with the callsign, a bad hex or a non-OK answer is null, the signal is passed through", async () => {
  const seen: Array<{ url: string; signal?: AbortSignal }> = [];
  const ok: FetchLike = async (url, init) => { seen.push({ url, signal: init?.signal }); return { ok: true, status: 200, json: async () => LIVE }; };
  const ac = new AbortController();
  const v = await fetchPlanHistory("A12733", " ual643 ", ok, ac.signal);
  assert.equal(v?.changes.length, 3);
  assert.deepEqual(seen.map((s) => s.url), ["/api/data/aircraft/plan-history/a12733?callsign=UAL643"]);
  assert.equal(seen[0].signal, ac.signal);
  assert.equal(await fetchPlanHistory("not-hex", null, ok), null);
  assert.equal(seen.length, 1, "a bad hex is never requested");
  await fetchPlanHistory("a12733", "bad call/sign", ok);
  assert.equal(seen[1].url, "/api/data/aircraft/plan-history/a12733", "a callsign that is not one is left out, never sent");
  assert.equal(await fetchPlanHistory("a12733", null, async () => ({ ok: false, status: 503, json: async () => ({}) })), null);
});
