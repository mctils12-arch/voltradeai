// faaTfrList.ts — the FAA's public list of flight restrictions. Every
// fixture under __fixtures__/faa_tfr_list is a REAL file from tfr.faa.gov,
// unedited (read 2026-11-22): the list (48 notices), and four detail files:
//   6/9306  one area, a four-corner outline, daily hours (the notice the text reader was built on)
//   6/9276  four areas with four outlines and different limits
//   6/6925  a standing restriction, "PERM", no dates in the file's own fields
//   1/5060  a listed notice whose file carries no outline at all
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  parseFaaList, parseFaaDetail, faaCoord, FaaTfrListStore, runFaaTfrPass, faaTfrDetailUrl, faaTfrListEnabled,
  restoreFaaTfrList, saveFaaTfrList, FAA_TFR_LIST_URL, FAA_TFR_STALE_MS, FAA_TFR_MAX_LISTED, type FaaFetch, type FaaListEntry,
} from "./faaTfrList";
import { flightRestrictionsAnswer, sharedNotamStore, _resetSwimCaptureForTests } from "./swimCapture";
import { sharedFaaTfrList, _resetFaaTfrListForTests } from "./faaTfrList";
import { judgeListNotice } from "../scripts/tfr_list_gate1";

const FX = path.join("server", "__fixtures__", "faa_tfr_list");
const file = (n: string) => fs.readFileSync(path.join(FX, n), "utf8");
const LIST = JSON.parse(file("list.json"));
const entry = (notice: string): FaaListEntry => parseFaaList(LIST).entries.find((e) => e.notice === notice)!;
const detail = (notice: string) => file(`detail_${notice.replace("/", "_")}.xml`);
const T0 = Date.UTC(2026, 10, 22, 13, 0); // 2026-11-22 13:00Z, the day the fixtures were read

/** a fetch that serves the fixtures; anything else is a 404 */
const fixtureFetch = (o: { listBody?: string; fail?: Set<string>; calls?: string[] } = {}): FaaFetch => async (url) => {
  o.calls?.push(url);
  if (url === FAA_TFR_LIST_URL) return { ok: true, status: 200, text: o.listBody ?? file("list.json") };
  for (const n of ["6/9306", "6/9276", "6/6925", "1/5060"]) {
    if (url === faaTfrDetailUrl(n)) return o.fail?.has(n) ? { ok: false, status: 503, text: "" } : { ok: true, status: 200, text: detail(n) };
  }
  return { ok: false, status: 404, text: "" };
};
const FOUR = JSON.stringify(LIST.filter((x: { notam_id: string }) => ["6/9306", "6/9276", "6/6925", "1/5060"].includes(x.notam_id)));

test("the FAA's list: every notice once, malformed rows counted, a long list cut", () => {
  const p = parseFaaList(LIST);
  assert.equal(p.entries.length, 48);
  assert.equal(p.malformed, 0);
  assert.deepEqual(p.entries[0], { notice: "6/9306", type: "HAZARDS", facility: "ZDC", state: "VA", description: "20NM W LEXINGTON, VA, Sunday, November 22, 2026 through Sunday, November 29, 2026 UTC" });
  const odd = parseFaaList([{ notam_id: "6/9306" }, { notam_id: "6/9306" }, { notam_id: "9306" }, null, { notam_id: 7 }, "x"]);
  assert.equal(odd.entries.length, 1);
  assert.equal(odd.malformed, 4);
  assert.deepEqual(parseFaaList({ not: "a list" }), { entries: [], malformed: 0, cut: 0 });
  const long = parseFaaList(Array.from({ length: FAA_TFR_MAX_LISTED + 7 }, (_, i) => ({ notam_id: `${i % 10}/${String(1000 + i).slice(-4)}` })));
  assert.equal(long.entries.length, FAA_TFR_MAX_LISTED);
  assert.equal(long.cut, 7);
});

test("coordinates as the FAA writes them", () => {
  assert.equal(faaCoord("37.86666667N", "lat"), 37.86666667);
  assert.equal(faaCoord("079.875W", "lon"), -79.875);
  assert.equal(faaCoord("13.5S", "lat"), -13.5);
  assert.equal(faaCoord("144.8E", "lon"), 144.8);
  // a longitude where a latitude belongs, a missing hemisphere, out of range: not a number
  for (const [s, k] of [["079.875W", "lat"], ["37.5", "lat"], ["91N", "lat"], ["181E", "lon"], ["", "lon"], ["N", "lat"]] as const) assert.equal(faaCoord(s, k), null, s);
});

test("FDC 6/9306: the FAA's own four corners, its dates from the notice's effective group, its limits", () => {
  const r = parseFaaDetail(detail("6/9306"), entry("6/9306"))!;
  assert.equal(r.id, "FAA:6/9306");
  assert.equal(r.source, "faa_list");
  assert.equal(r.citation, "FDC 6/9306");
  assert.equal(r.location, "ZDC");
  assert.equal(r.issued, Date.UTC(2026, 10, 21, 23, 27, 1));
  assert.equal(r.start, Date.UTC(2026, 10, 22, 12, 0));
  assert.equal(r.end, Date.UTC(2026, 10, 29, 23, 0));
  assert.deepEqual(r.geometry, { kind: "polygon", from: "faa_list", rings: [[[37.866667, -79.875], [37.85, -79.833333], [37.808333, -79.866667], [37.816667, -79.9], [37.866667, -79.875]]] });
  assert.deepEqual(r.faa, {
    type: "HAZARDS", rule: "91.137(a)(2)", place: "20NM W LEXINGTON, VIRGINIA", areas_without_outline: 0,
    areas: [{ name: "Hazard Area1", lower: { value: 0, uom: "FT", ref: "ALT" }, upper: { value: 4000, uom: "FT", ref: "ALT" }, scheduled: true }],
  });
  assert.match(r.text!, /^!FDC 6\/9306 ZDC/);
});

test("FDC 6/9276: four areas, four outlines, each area's own limits; every outline is closed", () => {
  const r = parseFaaDetail(detail("6/9276"), entry("6/9276"))!;
  assert.equal(r.geometry!.rings.length, 4);
  assert.deepEqual(r.geometry!.rings.map((x) => x.length), [56, 10, 7, 5]);
  for (const ring of r.geometry!.rings) assert.deepEqual(ring[0], ring[ring.length - 1]);
  assert.deepEqual(r.faa.areas.map((a) => [a.name, a.lower?.value, a.lower?.ref, a.upper?.value]), [["Area A", 0, "HEI", 2500], ["Area B", 1700, "ALT", 14000], ["Area D", 2500, "ALT", 8500], ["Area E", 1200, "ALT", 3000]]);
  assert.equal(r.start, Date.UTC(2026, 10, 23, 13, 0));
  assert.equal(r.end, Date.UTC(2026, 10, 26, 7, 0));
});

test("a standing restriction has a start and no end; a listed notice with no outline is a row with no geometry", () => {
  const perm = parseFaaDetail(detail("6/6925"), entry("6/6925"))!;
  assert.equal(perm.start, Date.UTC(2026, 10, 10, 17, 55));
  assert.equal(perm.end, null);
  assert.equal(perm.geometry!.rings.length, 1);
  const none = parseFaaDetail(detail("1/5060"), entry("1/5060"))!;
  assert.equal(none.geometry, null);
  assert.equal(none.citation, "FDC 1/5060");
});

test("a file that is not the notice asked for is not a row; nor is a cut file or an outline with a corner that is not a point", () => {
  assert.equal(parseFaaDetail(detail("6/9306"), entry("6/9276")), null, "another notice's file");
  assert.equal(parseFaaDetail("<html>Service unavailable</html>", entry("6/9306")), null);
  assert.equal(parseFaaDetail("", entry("6/9306")), null);
  // one corner written as an arc centre: the whole outline is left out, never completed by guess
  const arc = detail("6/9306").replace(/(<abdMergedArea>[\s\S]*?<codeType>)GRC(<\/codeType>)/, "$1CWA$2");
  const r = parseFaaDetail(arc, entry("6/9306"))!;
  assert.equal(r.geometry, null);
  assert.equal(r.faa.areas_without_outline, 1);
  assert.equal(r.faa.areas.length, 1);
  // a corner with no hemisphere letter
  const bad = parseFaaDetail(detail("6/9306").replace("<geoLat>37.85N</geoLat>", "<geoLat>37.85</geoLat>"), entry("6/9306"))!;
  assert.equal(bad.geometry, null);
});

test("the outline copied from the detail file is the one the FAA's shape service draws (the gate's own judge)", () => {
  const r = parseFaaDetail(detail("6/9306"), entry("6/9306"))!;
  // the shape service's polygon for 6/9306, 2026-11-22 (research/tfr_text_gate1_2026-11-22.json's source)
  const faa: Array<[number, number]> = [[37.86666667, -79.875], [37.85, -79.83333333], [37.80833333, -79.86666667], [37.81666667, -79.9], [37.86666667, -79.875]];
  assert.equal(judgeListNotice("6/9306", [faa], r).verdict, "agree");
  const moved = faa.map(([la, lo]) => [la + 0.05, lo] as [number, number]);
  assert.equal(judgeListNotice("6/9306", [moved], r).verdict, "disagree", "3 NM north is not the same outline");
  assert.equal(judgeListNotice("6/9306", [faa, moved], r).verdict, "partial");
  assert.equal(judgeListNotice("6/9306", [faa], null).verdict, "not_read");
});

test("a pass: the list, then each unread detail file once; the second pass asks for the list only", async () => {
  const store = new FaaTfrListStore();
  const calls: string[] = [];
  let now = T0;
  const o = { fetch: fixtureFetch({ listBody: FOUR, calls }), now: () => now, sleep: async () => {} };
  const a = await runFaaTfrPass(store, o);
  assert.deepEqual(a, { cause: "ok", listed: 4, details_read: 4, details_failed: 0, changed: true });
  assert.equal(calls.length, 5);
  const cov = store.coverage(now, true, new Set(["FDC 6/9306"]));
  assert.deepEqual({ ...cov, not_drawn: cov.not_drawn.map((x) => x.citation) }, {
    status: "ok", read_at: T0, last_attempt_at: T0, last_cause: "ok", listed: 4, with_outline: 3, without_outline: 1, detail_pending: 0,
    also_in_feed: 1, not_drawn: ["FDC 1/5060"], restored: false,
  });
  now += 30 * 60_000;
  const b = await runFaaTfrPass(store, o);
  assert.equal(b.details_read, 0);
  assert.equal(calls.length, 6, "no detail file is asked for twice");
  assert.equal(store.readAt, now);
});

test("what is served: in force or starting within 6 hours, with an outline; a notice the FAA no longer lists is gone", async () => {
  const store = new FaaTfrListStore();
  await runFaaTfrPass(store, { fetch: fixtureFetch({ listBody: FOUR }), now: () => T0, sleep: async () => {} });
  // 13:00Z on the 22nd: 6/9306 in force since 12:00Z, 6/6925 standing, 6/9276 starts on the 23rd (not yet), 1/5060 has no outline
  assert.deepEqual(store.active(T0).map((r) => r.citation).sort(), ["FDC 6/6925", "FDC 6/9306"]);
  const eve = Date.UTC(2026, 10, 23, 8, 0);
  assert.equal(store.fresh(eve), true);
  assert.ok(store.active(eve).some((r) => r.citation === "FDC 6/9276"), "inside 6 hours of its start");
  assert.deepEqual(Array.from(store.citations(T0)).sort(), ["FDC 6/6925", "FDC 6/9276", "FDC 6/9306"]);
  // past 6/9306's published end it is not served even while listed
  const later = new FaaTfrListStore();
  const dec = Date.UTC(2026, 11, 1, 0, 0);
  await runFaaTfrPass(later, { fetch: fixtureFetch({ listBody: FOUR }), now: () => dec, sleep: async () => {} });
  assert.deepEqual(later.active(dec).map((r) => r.citation), ["FDC 6/6925"]);
  // the FAA drops 6/6925 from its list: the next read drops the row
  const three = JSON.stringify(JSON.parse(FOUR).filter((x: { notam_id: string }) => x.notam_id !== "6/6925"));
  await runFaaTfrPass(store, { fetch: fixtureFetch({ listBody: three }), now: () => T0 + 60_000, sleep: async () => {} });
  assert.deepEqual(store.active(T0 + 60_000).map((r) => r.citation), ["FDC 6/9306"]);
});

test("the FAA cannot be reached: the last read is served with its age for 24 hours, then nothing, and the status says which", async () => {
  const store = new FaaTfrListStore();
  const down: FaaFetch = async () => { const e = new Error("x"); e.name = "TimeoutError"; throw e; };
  const first = await runFaaTfrPass(store, { fetch: down, now: () => T0, sleep: async () => {} });
  assert.equal(first.cause, "timeout");
  assert.equal(store.coverage(T0, true, null).status, "unreachable");
  assert.deepEqual(store.active(T0), []);
  await runFaaTfrPass(store, { fetch: fixtureFetch({ listBody: FOUR }), now: () => T0, sleep: async () => {} });
  const t1 = T0 + 3 * 3600_000;
  assert.equal((await runFaaTfrPass(store, { fetch: async () => ({ ok: false, status: 503, text: "" }), now: () => t1, sleep: async () => {} })).cause, "http_error");
  const c = store.coverage(t1, true, null);
  assert.deepEqual([c.status, c.read_at, c.last_attempt_at, c.last_cause], ["ok", T0, t1, "http_error"]);
  assert.equal(store.active(t1).length, 2, "the last read is still served");
  const t2 = T0 + FAA_TFR_STALE_MS + 1;
  assert.equal(store.coverage(t2, true, null).status, "stale");
  assert.deepEqual(store.active(t2), []);
  assert.equal(store.citations(t2).size, 0);
  assert.equal(store.coverage(t2, false, null).status, "off");
  assert.equal(new FaaTfrListStore().coverage(T0, true, null).status, "never");
});

test("an answer that is not the list never empties the list; a failed detail file is asked for again next pass", async () => {
  const store = new FaaTfrListStore();
  const calls: string[] = [];
  const a = await runFaaTfrPass(store, { fetch: fixtureFetch({ listBody: FOUR, fail: new Set(["6/9276"]), calls }), now: () => T0, sleep: async () => {} });
  assert.equal(a.details_failed, 1);
  assert.equal(store.coverage(T0, true, null).detail_pending, 1);
  for (const body of ["[]", "<html>maintenance</html>", "{\"error\":true}"]) {
    const r = await runFaaTfrPass(store, { fetch: fixtureFetch({ listBody: body }), now: () => T0 + 1, sleep: async () => {} });
    assert.equal(r.cause, "not_a_list", body);
    assert.equal(r.changed, false);
  }
  assert.equal(store.coverage(T0 + 1, true, null).listed, 4);
  const b = await runFaaTfrPass(store, { fetch: fixtureFetch({ listBody: FOUR, calls }), now: () => T0 + 2, sleep: async () => {} });
  assert.equal(b.details_read, 1);
  assert.equal(store.coverage(T0 + 2, true, null).with_outline, 3);
});

test("the read is kept across a restart, and a saved file that is malformed puts nothing back", async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "faa-tfr-"));
  const f = path.join(dir, "faa_tfr_list.json");
  const store = new FaaTfrListStore();
  assert.equal(await saveFaaTfrList(store, f), false, "nothing read, nothing written");
  await runFaaTfrPass(store, { fetch: fixtureFetch({ listBody: FOUR }), now: () => T0, sleep: async () => {} });
  assert.equal(await saveFaaTfrList(store, f), true);
  const back = new FaaTfrListStore();
  assert.equal(restoreFaaTfrList(back, f, T0 + 60_000), 4);
  assert.deepEqual(back.active(T0 + 60_000), store.active(T0 + 60_000));
  assert.equal(back.coverage(T0 + 60_000, true, null).restored, true);
  assert.equal(back.pending().length, 0, "a restart does not ask for the detail files again");
  // a day later the saved rows are known but not served until the list is read again
  assert.deepEqual(back.active(T0 + FAA_TFR_STALE_MS + 1), []);
  for (const body of ["", "{", "[]", JSON.stringify({ v: 99, read_at: T0, entries: [], rows: [] }), JSON.stringify({ v: 1, read_at: T0 + 9e9, entries: JSON.parse(FOUR), rows: [] })]) {
    fs.writeFileSync(f, body);
    assert.equal(restoreFaaTfrList(new FaaTfrListStore(), f, T0), 0, body.slice(0, 20));
  }
  assert.equal(restoreFaaTfrList(new FaaTfrListStore(), path.join(dir, "absent.json"), T0), 0);
  fs.rmSync(dir, { recursive: true, force: true });
});

test("on only where the notice feed is configured, and off by its switch", () => {
  const feed = { SWIM_NOTAM_URL: "u", SWIM_NOTAM_VPN: "v", SWIM_NOTAM_QUEUE: "q", SWIM_NOTAM_USER: "n", SWIM_NOTAM_PASSWORD: "p" };
  assert.equal(faaTfrListEnabled({}), false);
  assert.equal(faaTfrListEnabled(feed), true);
  for (const v of ["off", "0", "false", "no", " OFF "]) assert.equal(faaTfrListEnabled({ ...feed, VOLTRADE_FAA_TFR_LIST: v }), false, v);
});

test("the route's answer: one row per notice; the FAA's outline replaces our reading of the text, never geometry the feed sent", async () => {
  _resetSwimCaptureForTests();
  _resetFaaTfrListForTests();
  const now = Date.UTC(2026, 10, 22, 13, 0);
  const realNow = Date.now;
  Date.now = () => now;
  try {
    // the feed's own 6/9306 (the real production message): an outline read from its text
    sharedNotamStore().ingest(fs.readFileSync(path.join("server", "__fixtures__", "notam_fdc_6_9306.xml"), "utf8"), now);
    const before = flightRestrictionsAnswer(now, {});
    assert.equal(before.tfrs.length, 1);
    assert.equal((before.tfrs[0].geometry as { from?: string }).from, "text");
    assert.deepEqual([before.faa_list.status, before.faa_list.listed, before.faa_list.served], ["off", 0, 0]);
    await runFaaTfrPass(sharedFaaTfrList(), { fetch: fixtureFetch({ listBody: FOUR }), now: () => now, sleep: async () => {} });
    const feedEnv = { SWIM_NOTAM_URL: "u", SWIM_NOTAM_VPN: "v", SWIM_NOTAM_QUEUE: "q", SWIM_NOTAM_USER: "n", SWIM_NOTAM_PASSWORD: "p" };
    const a = flightRestrictionsAnswer(now, feedEnv);
    assert.deepEqual(a.tfrs.map((t) => [t.citation, (t.geometry as { from?: string }).from]).sort(), [["FDC 6/6925", "faa_list"], ["FDC 6/9306", "faa_list"]]);
    assert.deepEqual([a.faa_list.status, a.faa_list.listed, a.faa_list.with_outline, a.faa_list.without_outline, a.faa_list.also_in_feed, a.faa_list.served], ["ok", 4, 3, 1, 1, 2]);
    assert.equal(a.stats.tfrs, 1, "the feed's own count is unchanged");
    // once the list read is stale the feed's row is what is left
    const late = flightRestrictionsAnswer(now + FAA_TFR_STALE_MS + 1, feedEnv);
    assert.deepEqual(late.tfrs.map((t) => (t.geometry as { from?: string }).from), ["text"]);
    assert.equal(late.faa_list.status, "stale");
  } finally {
    Date.now = realNow;
    _resetSwimCaptureForTests();
    _resetFaaTfrListForTests();
  }
});

test("a notice the feed sent WITH airspace geometry keeps the feed's row, and the list's row for it is left out", async () => {
  _resetSwimCaptureForTests();
  _resetFaaTfrListForTests();
  const now = Date.UTC(2026, 10, 22, 13, 0);
  try {
    const sent = {
      id: "FDC:6/9306", series: null, number: "9306", year: null, type: "N", classification: "FDC", location: "KZDC", qcode: null,
      issued: now - 3600_000, start: now - 1800_000, end: now + 86_400_000, minFl: 0, maxFl: 40, center: null, radiusNm: null,
      text: "TEMPORARY FLIGHT RESTRICTIONS 91.137", scenario: null, citation: "FDC 6/9306",
      geometry: { kind: "polygon", rings: [[[37.9, -79.9], [37.8, -79.8], [37.8, -79.9], [37.9, -79.9]]] },
    };
    assert.deepEqual(sharedNotamStore().restore([sent], now), { tfrs: 1, dropped: 0 });
    assert.deepEqual(Array.from(sharedNotamStore().citations()), ["FDC 6/9306"]);
    await runFaaTfrPass(sharedFaaTfrList(), { fetch: fixtureFetch({ listBody: FOUR }), now: () => now, sleep: async () => {} });
    const a = flightRestrictionsAnswer(now, {});
    const rows = a.tfrs.filter((t) => t.citation === "FDC 6/9306");
    assert.equal(rows.length, 1);
    assert.equal(rows[0].id, "FDC:6/9306");
    assert.equal("from" in (rows[0].geometry as object), false);
    assert.equal(a.faa_list.served, 1, "only the standing 6/6925 comes from the list");
  } finally {
    _resetSwimCaptureForTests();
    _resetFaaTfrListForTests();
  }
});
