// faaTfrTextCheck.ts — the text reader's gate-1 check, running. The notice
// texts and FAA outlines are REAL (server/__fixtures__/tfr_text_gate1_sample.json
// and __fixtures__/faa_tfr_list, read 2026-11-22); the feed message is the
// production one (notam_fdc_6_9306.xml).
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  FaaTfrTextCheck, judgeOutlines, outlineDistanceNm, toleranceNm, restoreFaaTfrTextCheck, saveFaaTfrTextCheck,
  sharedFaaTfrTextCheck, _resetFaaTfrTextCheckForTests, TEXT_CHECK_MAX_NOTICES, TEXT_CHECK_MAX_NAMED, TEXT_CHECK_TOLERANCE,
  type Ring, type FeedOutline, type ListOutline,
} from "./faaTfrTextCheck";
import { textAreas } from "./swimNotamText";
import { NotamStore, NOTAM_MAX_UNSHAPED_CITATIONS } from "./swimNotam";
import { judgeNotice, outlineDistanceNm as scriptDistance, toleranceNm as scriptTolerance } from "../scripts/tfr_text_gate1";
import { parseFaaList, runFaaTfrPass, sharedFaaTfrList, _resetFaaTfrListForTests, bootFaaTfrList, FAA_TFR_FIRST_TICK_MS, type FaaFetch } from "./faaTfrList";
import { flightRestrictionsAnswer, runFlightRestrictionTextCheck, sharedNotamStore, _resetSwimCaptureForTests } from "./swimCapture";

interface Sample { notice: string; type: string; text: string; faa: Ring[] }
const SAMPLE = (JSON.parse(fs.readFileSync(new URL("./__fixtures__/tfr_text_gate1_sample.json", import.meta.url), "utf8")) as { notices: Sample[] }).notices;
const by = (n: string) => SAMPLE.find((s) => s.notice === n)!;
const T0 = Date.UTC(2026, 10, 23, 18, 0);
const NONE: ReadonlySet<string> = new Set();

const listRow = (s: Sample): ListOutline => ({ citation: `FDC ${s.notice}`, geometry: { rings: s.faa } });
/** the feed's row as swimNotam.ts stores it: the outline our reader made from the notice's words */
const feedRow = (s: Sample): FeedOutline => {
  const t = textAreas(s.text);
  return { citation: `FDC ${s.notice}`, geometry: { kind: "polygon", rings: t.rings, from: "text", unread: t.unread } };
};
const shifted = (r: Ring, dLat: number): Ring => r.map(([la, lo]) => [la + dLat, lo] as [number, number]);

test("one rule, two uses: the script's verdict on every sample notice is judgeOutlines' verdict, and the measure is the same function", () => {
  assert.equal(scriptDistance, outlineDistanceNm);
  assert.equal(scriptTolerance, toleranceNm);
  for (const s of SAMPLE) {
    const t = textAreas(s.text);
    const j = judgeOutlines(s.faa, t.rings, t.unread);
    const n = judgeNotice(s.notice, s.type, s.faa, s.text);
    assert.deepEqual([n.verdict, n.matched, n.worst_matched_nm, n.distances_nm], [j.verdict, j.matched, j.worst_matched_nm, j.distances_nm], s.notice);
  }
  // the verdicts the one-day check recorded for these notices (research/tfr_text_gate1_2026-11-22.json)
  assert.deepEqual(SAMPLE.map((s) => [s.notice, judgeNotice(s.notice, s.type, s.faa, s.text).verdict]), [
    ["6/9306", "agree"], ["6/2478", "agree"], ["6/2520", "agree"], ["6/4116", "agree"], ["6/2481", "agree"], ["6/9071", "not_read"], ["6/9287", "not_read"],
  ]);
});

test("a notice both sources hold is judged against the FAA's outline, once", () => {
  const c = new FaaTfrTextCheck();
  const s = by("6/9306");
  assert.equal(c.observe([listRow(s)], [feedRow(s)], NONE, T0), 1);
  assert.deepEqual(c.get("FDC 6/9306"), { verdict: "agree", at: T0, faa: 1, read: 1, unread: 0, matched: 1, worst_matched_nm: 0 });
  // a later pass holds the same pair: nothing is judged again and the first verdict's time stands
  assert.equal(c.observe([listRow(s)], [feedRow(s)], NONE, T0 + 1_800_000), 0);
  const v = c.view();
  assert.deepEqual([v.since, v.last_judged_at, v.compared, v.agree, v.partial, v.disagree, v.not_read, v.feed_sent_geometry], [T0, T0, 1, 1, 0, 0, 0, 0]);
  assert.equal(v.tolerance, TEXT_CHECK_TOLERANCE);
  assert.deepEqual(v.not_agreeing, []);
});

test("the real sample: all five notices our reader reads agree, the four-area one included, and the largest distance is inside the tolerance", () => {
  const c = new FaaTfrTextCheck();
  const read = SAMPLE.filter((s) => textAreas(s.text).rings.length);
  assert.equal(c.observe(read.map(listRow), read.map(feedRow), NONE, T0), 5);
  const v = c.view();
  assert.deepEqual([v.compared, v.agree, v.partial, v.disagree], [5, 5, 0, 0]);
  assert.deepEqual(v.not_agreeing, []);
  assert.deepEqual([c.get("FDC 6/2481")!.faa, c.get("FDC 6/2481")!.matched], [4, 4]);
  assert.ok(v.worst_agree_nm !== null && v.worst_agree_nm < 0.5, String(v.worst_agree_nm));
});

test("an outline that is not the FAA's is a disagreement, named in the answer; an unread area makes a match partial", () => {
  const c = new FaaTfrTextCheck();
  const s = by("6/9306");
  // one tenth of a degree north is 6 NM: far outside 0.1 NM + 0.4%
  c.observe([listRow(s)], [{ citation: "FDC 6/9306", geometry: { kind: "polygon", rings: [shifted(s.faa[0], 0.1)], from: "text", unread: 0 } }], NONE, T0);
  assert.equal(c.get("FDC 6/9306")!.verdict, "disagree");
  assert.equal(c.get("FDC 6/9306")!.worst_matched_nm, null);
  const r = by("6/2478");
  c.observe([listRow(r)], [{ ...feedRow(r), geometry: { ...feedRow(r).geometry!, unread: 1 } }], NONE, T0 + 1);
  assert.equal(c.get("FDC 6/2478")!.verdict, "partial");
  const v = c.view();
  assert.deepEqual([v.compared, v.agree, v.partial, v.disagree], [2, 0, 1, 1]);
  assert.deepEqual(v.not_agreeing.map((n) => n.citation), ["FDC 6/2478", "FDC 6/9306"], "newest first");
  assert.equal(v.worst_agree_nm, null);
});

test("what is counted apart, and what is not judged at all", () => {
  const c = new FaaTfrTextCheck();
  const a = by("6/9071"), b = by("6/9287"), d = by("6/2520"), e = by("6/4116"), f = by("6/2478");
  const added = c.observe(
    [listRow(a), listRow(b), listRow(d), listRow(e), { citation: "FDC 1/5060", geometry: null }, listRow(f)],
    [
      { citation: "FDC 6/9287", geometry: { kind: "circle" } },                       // drawn as the index circle: the text gave no outline
      { citation: "FDC 6/2520", geometry: { kind: "polygon", rings: d.faa } },        // geometry as the feed sent it: not the reader's
      { citation: "FDC 1/5060", geometry: { kind: "polygon", rings: d.faa, from: "text", unread: 0 } }, // the FAA publishes no outline: no ground truth
      { geometry: { kind: "polygon", rings: e.faa, from: "text", unread: 0 } },       // a feed row with no citation cannot be paired
    ],
    new Set(["FDC 6/9071", "FDC 9/9999"]), T0);                                        // 6/9071 arrived on the feed and could not be drawn
  assert.equal(added, 3);
  assert.deepEqual([c.get("FDC 6/9071")!.verdict, c.get("FDC 6/9287")!.verdict, c.get("FDC 6/2520")!.verdict], ["not_read", "not_read", "feed_sent_geometry"]);
  for (const k of ["FDC 6/4116", "FDC 1/5060", "FDC 6/2478", "FDC 9/9999"]) assert.equal(c.has(k), false, k);
  const v = c.view();
  assert.deepEqual([v.compared, v.not_read, v.feed_sent_geometry], [0, 2, 1]);
  assert.deepEqual(v.not_agreeing.map((n) => n.verdict), ["not_read", "not_read"]);
  // the feed sends 6/2478 later: it is judged then
  assert.equal(c.observe([listRow(f)], [feedRow(f)], NONE, T0 + 60_000), 1);
  assert.equal(c.view().last_judged_at, T0 + 60_000);
  assert.equal(c.view().since, T0);
});

test("the answer names at most the newest non-agreeing notices and the store drops its oldest verdicts at the limit, counted", () => {
  const c = new FaaTfrTextCheck();
  const s = by("6/9306");
  const cite = (i: number) => `FDC ${Math.floor(i / 10_000)}/${String(i % 10_000).padStart(4, "0")}`;
  const n = TEXT_CHECK_MAX_NOTICES + 3;
  for (let i = 0; i < n; i++) c.observe([{ citation: cite(i), geometry: { rings: s.faa } }], [], new Set([cite(i)]), T0 + i);
  assert.equal(c.size(), TEXT_CHECK_MAX_NOTICES);
  assert.equal(c.has(cite(0)), false);
  assert.equal(c.has(cite(3)), true);
  const v = c.view();
  assert.equal(v.dropped, 3);
  assert.equal(v.not_agreeing.length, TEXT_CHECK_MAX_NAMED);
  assert.equal(v.not_agreeing[0].citation, cite(n - 1));
});

test("verdicts survive a restart: saved, read back, malformed entries left out, a verdict already held wins", async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "tfr-text-check-"));
  const file = path.join(dir, "faa_tfr_text_check.json");
  try {
    const c = new FaaTfrTextCheck();
    const read = SAMPLE.filter((s) => textAreas(s.text).rings.length);
    c.observe(read.map(listRow), read.map(feedRow), NONE, T0);
    assert.equal(await saveFaaTfrTextCheck(c, file), true);
    const back = new FaaTfrTextCheck();
    assert.equal(restoreFaaTfrTextCheck(back, file), 5);
    assert.deepEqual(back.view(), c.view());
    // a judged notice is not judged again after the restart
    assert.equal(back.observe(read.map(listRow), read.map(feedRow), NONE, T0 + 5), 0);

    const doc = JSON.parse(fs.readFileSync(file, "utf8"));
    doc.notices.push(["not a citation", doc.notices[0][1]], ["FDC 7/0001", { verdict: "fine", at: T0, faa: 1, read: 1, unread: 0, matched: 1 }], ["FDC 7/0002", null], ["FDC 7/0003", { verdict: "agree", at: -1, faa: 1, read: 1, unread: 0, matched: 1 }]);
    fs.writeFileSync(file, JSON.stringify(doc));
    const held = new FaaTfrTextCheck();
    held.observe([listRow(by("6/9306"))], [{ citation: "FDC 6/9306", geometry: { kind: "circle" } }], NONE, T0 + 9);
    assert.equal(restoreFaaTfrTextCheck(held, file), 4, "the four not already held; the malformed four left out");
    assert.equal(held.get("FDC 6/9306")!.verdict, "not_read");
    assert.equal(held.since, T0);

    for (const bad of ["{", JSON.stringify({ v: 99, notices: [] }), "null"]) {
      fs.writeFileSync(file, bad);
      assert.equal(restoreFaaTfrTextCheck(new FaaTfrTextCheck(), file), 0, bad);
    }
    assert.equal(restoreFaaTfrTextCheck(new FaaTfrTextCheck(), path.join(dir, "absent.json")), 0);
  } finally { fs.rmSync(dir, { recursive: true, force: true }); }
});

test("the feed store remembers which restriction notices it could not draw, by citation, bounded", () => {
  const xml = fs.readFileSync(path.join("server", "__fixtures__", "notam_fdc_6_9306.xml"), "utf8");
  // the same real message with its area wording broken: a restriction notice with no shape
  const noArea = xml.replace(/WI AN AREA DEFINED AS/g, "IN THE AREA OF");
  assert.notEqual(noArea, xml);
  const st = new NotamStore();
  st.ingest(noArea, T0);
  st.ingest(noArea, T0 + 1);
  assert.equal(st.stats().tfrs_without_shape, 2, "messages");
  assert.deepEqual(Array.from(st.unshapedCitations()), ["FDC 6/9306"], "notices");
  const drawn = new NotamStore();
  drawn.ingest(xml, T0);
  assert.equal(drawn.unshapedCitations().size, 0);
  assert.ok(NOTAM_MAX_UNSHAPED_CITATIONS >= 1000);
});

const FX = path.join("server", "__fixtures__", "faa_tfr_list");
const fxFile = (n: string) => fs.readFileSync(path.join(FX, n), "utf8");
const FOUR = JSON.stringify((JSON.parse(fxFile("list.json")) as Array<{ notam_id: string }>).filter((e) => ["6/9306", "6/9276", "6/6925", "1/5060"].includes(e.notam_id)));
const fixtureFetch: FaaFetch = async (url) => {
  if (url.endsWith("exportTfrList")) return { ok: true, status: 200, text: FOUR };
  const m = /detail_(\d_\d{4})\.xml$/.exec(url);
  const f = m ? path.join(FX, `detail_${m[1]}.xml`) : null;
  return f && fs.existsSync(f) ? { ok: true, status: 200, text: fs.readFileSync(f, "utf8") } : { ok: false, status: 404, text: "" };
};

test("end to end on the real production message: the feed's 6/9306, read from its words, is the FAA's own outline; the route says so and still serves one row", async () => {
  _resetSwimCaptureForTests(); _resetFaaTfrListForTests(); _resetFaaTfrTextCheckForTests();
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "tfr-text-check-e2e-"));
  const file = path.join(dir, "faa_tfr_text_check.json");
  const now = Date.UTC(2026, 10, 22, 13, 0);
  const realNow = Date.now;
  Date.now = () => now;
  try {
    assert.equal(parseFaaList(JSON.parse(FOUR)).entries.length, 4);
    const check = sharedFaaTfrTextCheck(file);
    sharedNotamStore().ingest(fs.readFileSync(path.join("server", "__fixtures__", "notam_fdc_6_9306.xml"), "utf8"), now);
    // before the list is read there is nothing to compare with
    assert.equal(await runFlightRestrictionTextCheck(now, { file }), null);
    assert.equal(fs.existsSync(file), false, "nothing judged, nothing written");
    await runFaaTfrPass(sharedFaaTfrList(), { fetch: fixtureFetch, now: () => now, sleep: async () => {} });
    assert.equal(await runFlightRestrictionTextCheck(now, { file }), "text reader checked on 1 more notice (1 of 1 compared agree so far)");
    assert.equal(check.get("FDC 6/9306")!.verdict, "agree");
    assert.ok(check.get("FDC 6/9306")!.worst_matched_nm! < 0.001);
    assert.equal(await runFlightRestrictionTextCheck(now + 1_800_000, { file }), null, "judged once");
    const feedEnv = { SWIM_NOTAM_URL: "u", SWIM_NOTAM_VPN: "v", SWIM_NOTAM_QUEUE: "q", SWIM_NOTAM_USER: "n", SWIM_NOTAM_PASSWORD: "p" };
    const a = flightRestrictionsAnswer(now, feedEnv);
    const tc = a.faa_list.text_check;
    assert.deepEqual([tc.since, tc.compared, tc.agree, tc.partial, tc.disagree, tc.not_read, tc.feed_sent_geometry], [now, 1, 1, 0, 0, 0, 0]);
    // the check changes nothing that is served: one row per notice, the FAA's outline
    assert.deepEqual(a.tfrs.map((t) => [t.citation, (t.geometry as { from?: string }).from]).sort(), [["FDC 6/6925", "faa_list"], ["FDC 6/9306", "faa_list"]]);
    // the verdict is on disk, and a new process reads it back
    _resetFaaTfrTextCheckForTests();
    assert.equal(sharedFaaTfrTextCheck(file).view().agree, 1);
  } finally {
    Date.now = realNow;
    _resetSwimCaptureForTests(); _resetFaaTfrListForTests(); _resetFaaTfrTextCheckForTests();
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test("the list's timer runs the after-pass step after a read that succeeded, puts its words on the job's line, and a throwing step does not fail the read", async (t) => {
  _resetFaaTfrListForTests();
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "tfr-text-check-boot-"));
  t.mock.timers.enable({ apis: ["setTimeout", "setInterval"] });
  const feedEnv = { SWIM_NOTAM_URL: "u", SWIM_NOTAM_VPN: "v", SWIM_NOTAM_QUEUE: "q", SWIM_NOTAM_USER: "n", SWIM_NOTAM_PASSWORD: "p", VOLTRADE_FAA_TFR_ARCHIVE: "off" } as unknown as NodeJS.ProcessEnv;
  let calls = 0;
  const errors: string[] = [];
  const realError = console.error;
  console.error = (...a: unknown[]) => { errors.push(a.join(" ")); };
  try {
    // a fetch with no waiting between detail files is not available through boot: answer the list with one notice
    const one = JSON.stringify((JSON.parse(FOUR) as Array<{ notam_id: string }>).filter((e) => e.notam_id === "6/9306"));
    const fetchOne: FaaFetch = async (url) => (url.endsWith("exportTfrList") ? { ok: true, status: 200, text: one } : fixtureFetch(url));
    assert.equal(bootFaaTfrList(feedEnv, { file: path.join(dir, "list.json"), fetch: fetchOne, archiveBaseDir: dir, afterPass: () => { calls++; throw new Error("boom"); } }), true);
    t.mock.timers.tick(FAA_TFR_FIRST_TICK_MS);
    // the pass writes the keep-file before the step runs: real disk work, whose length is the machine's.
    // Wait by the clock (the timers are mocked, the event loop is not), never by a count of turns:
    // 50 turns was enough on a laptop and not on the CI runner (PR #1894's one red test).
    const until = performance.now() + 15_000;
    while (!calls && performance.now() < until) await new Promise<void>((r) => setImmediate(r));
    assert.equal(calls, 1);
    for (let i = 0; i < 20; i++) await new Promise<void>((r) => setImmediate(r));
    assert.equal(sharedFaaTfrList().listed().length, 1, "the list read stands");
    assert.deepEqual(errors.filter((e) => e.includes("failed") && !e.includes("after-pass")), []);
  } finally {
    console.error = realError;
    t.mock.timers.reset();
    _resetFaaTfrListForTests();
    fs.rmSync(dir, { recursive: true, force: true });
  }
});
