import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import os from "os";
import path from "path";
import {
  normDate,
  parseTrials,
  diffTrials,
  buildTrialsUrl,
  fetchTrials,
  loadTrialState,
  saveTrialState,
  archiveTrialEvents,
  readArchivedTrialDay,
  listArchivedTrialDates,
  runClinicalTrialsOnce,
  refreshClinicalTrials,
  latestClinicalTrials,
  SuspectFetchError,
  _resetClinicalTrialsForTests,
  type TrialState,
} from "./clinicalTrials";

// Shape verified live against clinicaltrials.gov/api/v2 on 2026-10-19.
interface StudyOver {
  title?: string; status?: string; pcd?: string | null; fp?: string; sponsor?: string; phases?: string[];
}
const study = (nct: string, over: StudyOver = {}) => ({
  protocolSection: {
    identificationModule: { nctId: nct, briefTitle: over.title ?? `Study ${nct}` },
    statusModule: {
      overallStatus: over.status ?? "RECRUITING",
      primaryCompletionDateStruct: over.pcd === null ? undefined : { date: over.pcd ?? "2027-04-01", type: "ESTIMATED" },
      lastUpdatePostDateStruct: { date: "2026-10-01" },
      studyFirstPostDateStruct: { date: over.fp ?? "2024-01-01" },
    },
    sponsorCollaboratorsModule: { leadSponsor: { name: over.sponsor ?? "Acme Bio Inc", class: "INDUSTRY" } },
    conditionsModule: { conditions: ["A", "B", "C", "D"] },
    designModule: { phases: over.phases ?? ["PHASE2"], enrollmentInfo: { count: 120 } },
  },
});
const tmp = () => fs.mkdtempSync(path.join(os.tmpdir(), "ct-"));
const NOW = Date.parse("2026-10-19T12:00:00Z");

test("normDate: month-only is flagged coarse and normalised to the 1st", () => {
  assert.deepEqual(normDate("2027-04"), { iso: "2027-04-01", coarse: true });
  assert.deepEqual(normDate("2027-04-15"), { iso: "2027-04-15", coarse: false });
  assert.equal(normDate("garbage"), null);
  assert.equal(normDate(undefined), null);
});

test("parseTrials: maps fields, caps conditions, skips rows without an NCT id", () => {
  const out = parseTrials({ studies: [study("NCT1", { pcd: "2027-04" , phases: ["PHASE2", "PHASE3"] }), { protocolSection: {} }] });
  assert.equal(out.length, 1);
  assert.equal(out[0].ph, "PHASE2/PHASE3");
  assert.equal(out[0].coarse, true);
  assert.equal(out[0].co, "A; B; C");
  assert.equal(out[0].en, 120);
  assert.equal(parseTrials({}).length, 0);
});

test("diffTrials: bootstrap run emits ONE marker and no fabricated changes", () => {
  const cur = parseTrials({ studies: [study("NCT1"), study("NCT2")] });
  const ev = diffTrials(null, cur, "2026-10-19");
  assert.equal(ev.length, 1);
  assert.equal(ev[0].t, "bootstrap");
  assert.equal(ev[0].tracked, 2);
});

const snap = (studies: ReturnType<typeof study>[]) => {
  const m: Record<string, TrialState> = {};
  for (const s of parseTrials({ studies })) m[s.n] = s;
  return m;
};

test("diffTrials: pcd delay is positive days, pull-forward negative, with key + coarse flag", () => {
  const prev = snap([study("NCT1", { pcd: "2027-01-15" }), study("NCT2", { pcd: "2027-03-01" })]);
  const cur = parseTrials({ studies: [study("NCT1", { pcd: "2027-05-15" }), study("NCT2", { pcd: "2027-02" })] });
  const ev = diffTrials(prev, cur, "2026-10-19").filter((e) => e.t === "pcd_shift");
  const e1 = ev.find((e) => e.nct === "NCT1")!;
  assert.equal(e1.shiftDays, 120);
  assert.equal(e1.prevPcd, "2027-01-15");
  assert.equal(e1.key, "pcd_shift|NCT1|2027-01-15>2027-05-15");
  assert.equal(e1.coarse, false);
  const e2 = ev.find((e) => e.nct === "NCT2")!;
  assert.equal(e2.shiftDays, -28);
  assert.equal(e2.coarse, true);
});

test("diffTrials: status change captured; unchanged trial emits nothing", () => {
  const prev = snap([study("NCT1", { status: "RECRUITING" }), study("NCT2")]);
  const cur = parseTrials({ studies: [study("NCT1", { status: "TERMINATED" }), study("NCT2")] });
  const ev = diffTrials(prev, cur, "2026-10-19");
  assert.equal(ev.length, 1);
  assert.equal(ev[0].t, "status");
  assert.equal(ev[0].prevStatus, "RECRUITING");
  assert.equal(ev[0].status, "TERMINATED");
});

test("diffTrials: new registration vs window-edge entry are labelled differently", () => {
  const prev = snap([study("NCT1")]);
  const cur = parseTrials({ studies: [
    study("NCT1"),
    study("NCT9", { fp: "2026-10-10" }),   // posted 9 days ago -> genuinely new
    study("NCT8", { fp: "2022-05-01" }),   // old trial rolled into the forward window edge
  ] });
  const ev = diffTrials(prev, cur, "2026-10-19");
  assert.equal(ev.find((e) => e.nct === "NCT9")!.t, "new");
  assert.equal(ev.find((e) => e.nct === "NCT8")!.t, "entered_window");
});

test("diffTrials: a trial missing from the new snapshot is left_window (no completion claim)", () => {
  const prev = snap([study("NCT1"), study("NCT2"), study("NCT3"), study("NCT4")]);
  const cur = parseTrials({ studies: [study("NCT1"), study("NCT2"), study("NCT3")] });
  const ev = diffTrials(prev, cur, "2026-10-19");
  assert.deepEqual(ev.map((e) => [e.t, e.nct]), [["left_window", "NCT4"]]);
  assert.equal(ev[0].sponsor, "Acme Bio Inc");
});

// Regression guard for the failure that would poison the archive: a
// truncated fetch must NOT be read as mass departures.
test("diffTrials: truncated snapshot (<70% of previous) throws instead of emitting mass left_window", () => {
  const ids = Array.from({ length: 10 }, (_, i) => `NCT${i}`);
  const prev = snap(ids.map((i) => study(i)));
  const cur = parseTrials({ studies: ids.slice(0, 5).map((i) => study(i)) });
  assert.throws(() => diffTrials(prev, cur, "2026-10-19"), SuspectFetchError);
});

test("buildTrialsUrl: industry Phase 2/3 interventional, date window, page token", () => {
  const u = decodeURIComponent(buildTrialsUrl(NOW, "TOK"));
  assert.match(u, /AREA\[Phase\]\(PHASE2 OR PHASE3\)/);
  assert.match(u, /LeadSponsorClass\]INDUSTRY/);
  assert.match(u, /RANGE\[2026-08-20,2027-10-19\]/);
  assert.match(u, /pageToken=TOK/);
  assert.ok(!decodeURIComponent(buildTrialsUrl(NOW)).includes("pageToken"));
});

const okFetch = (pages: unknown[]) => {
  let i = 0;
  return async () => {
    const body = pages[Math.min(i++, pages.length - 1)];
    return { ok: true, status: 200, text: async () => JSON.stringify(body) };
  };
};

test("fetchTrials: follows nextPageToken across pages", async () => {
  const out = await fetchTrials(okFetch([
    { studies: [study("NCT1")], nextPageToken: "p2" },
    { studies: [study("NCT2")] },
  ]) as any, NOW);
  assert.deepEqual(out.map((t) => t.n), ["NCT1", "NCT2"]);
});

test("fetchTrials: endless pagination throws rather than returning a partial snapshot", async () => {
  await assert.rejects(
    fetchTrials(okFetch([{ studies: [study("NCT1")], nextPageToken: "more" }]) as any, NOW),
    /partial snapshot/,
  );
});

test("fetchTrials: HTTP error throws", async () => {
  await assert.rejects(fetchTrials((async () => ({ ok: false, status: 503, text: async () => "" })) as any, NOW), /503/);
});

test("archive: state round-trips; events dedupe within a day; list/read work", () => {
  const dir = tmp();
  assert.equal(loadTrialState(dir), null);
  const cur = parseTrials({ studies: [study("NCT1")] });
  saveTrialState(cur, dir);
  assert.equal(loadTrialState(dir)!["NCT1"].sp, "Acme Bio Inc");
  const ev = diffTrials(snap([study("NCT1", { pcd: "2027-01-01" })]), parseTrials({ studies: [study("NCT1", { pcd: "2027-06-01" })] }), "2026-10-19");
  assert.equal(archiveTrialEvents(ev, dir, NOW), 1);
  assert.equal(archiveTrialEvents(ev, dir, NOW), 0, "same-day rerun is idempotent");
  assert.deepEqual(listArchivedTrialDates(dir), ["2026-10-19"], "_state.json.gz must not appear as a day");
  assert.equal(readArchivedTrialDay("2026-10-19", dir)[0].shiftDays, 151);
});

test("runClinicalTrialsOnce: day 1 bootstraps, day 2 records only the change, failed day 3 keeps state", async () => {
  const dir = tmp();
  const day1 = [study("NCT1", { pcd: "2027-01-01" }), study("NCT2"), study("NCT3"), study("NCT4")];
  let r = await runClinicalTrialsOnce(okFetch([{ studies: day1 }]) as any, NOW, dir);
  assert.deepEqual(r, { tracked: 4, events: 1 });
  assert.equal(readArchivedTrialDay("2026-10-19", dir)[0].t, "bootstrap");

  const day2 = [study("NCT1", { pcd: "2027-04-01" }), study("NCT2"), study("NCT3"), study("NCT4")];
  r = await runClinicalTrialsOnce(okFetch([{ studies: day2 }]) as any, NOW + 86400_000, dir);
  assert.deepEqual(r, { tracked: 4, events: 1 });
  const ev = readArchivedTrialDay("2026-10-20", dir);
  assert.equal(ev[0].t, "pcd_shift");
  assert.equal(ev[0].shiftDays, 90);

  // failed fetch: throws, state untouched, nothing archived
  await assert.rejects(runClinicalTrialsOnce((async () => ({ ok: false, status: 500, text: async () => "" })) as any, NOW + 2 * 86400_000, dir));
  assert.equal(loadTrialState(dir)!["NCT1"].pcd, "2027-04-01");
  assert.deepEqual(listArchivedTrialDates(dir), ["2026-10-20", "2026-10-19"]);

  // truncated fetch: refused, state untouched
  await assert.rejects(runClinicalTrialsOnce(okFetch([{ studies: [study("NCT1")] }]) as any, NOW + 3 * 86400_000, dir), SuspectFetchError);
  assert.equal(Object.keys(loadTrialState(dir)!).length, 4);
});

test("runClinicalTrialsOnce: empty snapshot is refused (never recorded as 'all trials gone')", async () => {
  await assert.rejects(runClinicalTrialsOnce(okFetch([{ studies: [] }]) as any, NOW, tmp()), /empty snapshot/);
});

test("refreshClinicalTrials: cold boot with a dead upstream restores the cache from disk", async () => {
  const dir = tmp();
  _resetClinicalTrialsForTests();
  await runClinicalTrialsOnce(okFetch([{ studies: [study("NCT1", { pcd: "2027-01-01" }), study("NCT2"), study("NCT3")] }]) as any, NOW, dir);
  await runClinicalTrialsOnce(okFetch([{ studies: [study("NCT1", { pcd: "2027-05-01" }), study("NCT2"), study("NCT3")] }]) as any, NOW + 86400_000, dir);
  _resetClinicalTrialsForTests();
  assert.equal(latestClinicalTrials(), null);
  const origErr = console.error; console.error = () => {};
  try {
    await refreshClinicalTrials((async () => ({ ok: false, status: 502, text: async () => "" })) as any, NOW + 2 * 86400_000, dir);
  } finally { console.error = origErr; }
  const c = latestClinicalTrials()!;
  assert.equal(c.tracked, 3);
  assert.equal(c.events.length, 1);
  assert.equal(c.events[0].shiftDays, 120);
});
