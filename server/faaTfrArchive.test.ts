// faaTfrArchive.ts — the record of what appeared on and left the FAA's public
// list of flight restrictions. The list and detail files are the REAL FAA
// files of faaTfrList.test.ts (read 2026-11-22), unedited.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import {
  diffFaaListRead, readFaaTfrArchiveState, recordFaaTfrRead, loadFaaTfrArchive, faaTfrListedSince, faaTfrArchiveView, faaTfrArchiveEnabled,
  faaTfrArchiveDir, gzipOldFaaTfrDays, _resetFaaTfrArchiveForTests, FAA_TFR_MASS_LEAVE_MIN, FAA_TFR_ARCHIVE_STATE_VERSION,
  buildFaaTfrHistory, readFaaTfrHistory, faaTfrHistoryDays, FAA_TFR_HISTORY_DEFAULT_DAYS, FAA_TFR_HISTORY_MAX_DAYS, FAA_TFR_HISTORY_MAX_EVENTS, FAA_TFR_HISTORY_MAX_FILE_BYTES,
  type FaaTfrArchiveRow, type FaaTfrArchiveState,
} from "./faaTfrArchive";
import {
  parseFaaList, parseFaaDetail, FaaTfrListStore, runFaaTfrPass, faaTfrDetailUrl, sharedFaaTfrList, _resetFaaTfrListForTests,
  FAA_TFR_LIST_URL, FAA_TFR_MAX_LISTED, type FaaFetch, type FaaListEntry, type FaaTfrRow,
} from "./faaTfrList";
import { flightRestrictionsAnswer, flightRestrictionHistoryAnswer, _resetSwimCaptureForTests } from "./swimCapture";
import { RESPONSE_DATA_SCHEMAS } from "./apiProduct";

const FX = path.join("server", "__fixtures__", "faa_tfr_list");
const file = (n: string) => fs.readFileSync(path.join(FX, n), "utf8");
const LIST = JSON.parse(file("list.json"));
const ALL = parseFaaList(LIST).entries;
const entry = (notice: string): FaaListEntry => ALL.find((e) => e.notice === notice)!;
const NOTICES = ["6/9306", "6/9276", "6/6925", "1/5060"] as const;
const ROWS = new Map<string, FaaTfrRow>(NOTICES.map((n) => [n, parseFaaDetail(file(`detail_${n.replace("/", "_")}.xml`), entry(n))!]));
const rowOf = (n: string) => ROWS.get(n);
const T0 = Date.UTC(2026, 10, 22, 13, 0);
const HALF = 30 * 60_000;
const tmpBase = () => fs.mkdtempSync(path.join(os.tmpdir(), "faa-tfr-archive-"));
const lines = (base: string, day: string): FaaTfrArchiveRow[] =>
  fs.readFileSync(path.join(faaTfrArchiveDir(base), `${day}.jsonl`), "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l));
const plenty = () => 50 * 1024 ** 3;

test("the first read of a recording run: every listed notice is written as already listed, with what the FAA publishes for it", () => {
  const d = diffFaaListRead(null, NOTICES.map(entry), rowOf, T0);
  assert.equal(d.refused, false);
  assert.deepEqual(d.rows.map((r) => r.kind), ["listed", "listed", "listed", "listed", "run"]);
  const first = d.rows[0];
  assert.ok(first.kind === "listed");
  assert.deepEqual([first.seen, first.prev_read, first.backfill, first.notice, first.citation], ["2026-11-22T13:00:00.000Z", null, true, "6/9306", "FDC 6/9306"]);
  assert.deepEqual(first.list, { type: "HAZARDS", facility: "ZDC", state: "VA", description: "20NM W LEXINGTON, VA, Sunday, November 22, 2026 through Sunday, November 29, 2026 UTC" });
  // the row is the served row, unchanged: the FAA's outline and limits are in the record
  assert.deepEqual(first.row, ROWS.get("6/9306"));
  assert.equal(first.row!.geometry!.rings[0].length, 5);
  // a listed notice the FAA publishes no outline for is recorded too, with its text
  const noOutline = d.rows.find((r) => r.kind === "listed" && r.notice === "1/5060");
  assert.ok(noOutline && noOutline.kind === "listed" && noOutline.row && noOutline.row.geometry === null && (noOutline.row.text ?? "").length > 50);
  assert.deepEqual(d.rows[4], { kind: "run", seen: "2026-11-22T13:00:00.000Z", listed: 4, with_outline: 3, without_outline: 1, detail_pending: 0 });
  assert.deepEqual([d.next.v, d.next.started_at, d.next.last_read_at, d.next.last_run_day, d.next.refused_at], [FAA_TFR_ARCHIVE_STATE_VERSION, T0, T0, "2026-11-22", null]);
  assert.deepEqual(d.next.open["6/9306"], { first_seen: T0, backfill: true, detail: true, entry: entry("6/9306"), start: ROWS.get("6/9306")!.start, end: ROWS.get("6/9306")!.end });
});

test("a read that changes nothing writes nothing; the first read of the next UTC day writes that day's run line", () => {
  const a = diffFaaListRead(null, NOTICES.map(entry), rowOf, T0);
  const same = diffFaaListRead(a.next, NOTICES.map(entry), rowOf, T0 + HALF);
  assert.deepEqual(same.rows, []);
  assert.equal(same.next.last_read_at, T0 + HALF);
  assert.deepEqual(same.next.open, a.next.open);
  const nextDay = diffFaaListRead(same.next, NOTICES.map(entry), rowOf, T0 + 12 * 3600_000);
  assert.deepEqual(nextDay.rows, [{ kind: "run", seen: "2026-11-23T01:00:00.000Z", listed: 4, with_outline: 3, without_outline: 1, detail_pending: 0 }]);
  assert.equal(nextDay.next.last_run_day, "2026-11-23");
});

test("a notice that appears is written with the two reads it appeared between; one that leaves, with the two it left between", () => {
  const a = diffFaaListRead(null, ["6/9306", "6/6925"].map(entry), rowOf, T0);
  const b = diffFaaListRead(a.next, ["6/6925", "6/9276"].map(entry), rowOf, T0 + HALF);
  assert.deepEqual(b.rows.map((r) => [r.kind, (r as { notice?: string }).notice]), [["listed", "6/9276"], ["left", "6/9306"]]);
  const appeared = b.rows[0], left = b.rows[1];
  assert.ok(appeared.kind === "listed" && left.kind === "left");
  assert.deepEqual([appeared.seen, appeared.prev_read, appeared.backfill], ["2026-11-22T13:30:00.000Z", "2026-11-22T13:00:00.000Z", false]);
  assert.equal(appeared.row!.geometry!.rings.length, 4, "the four areas of 6/9276");
  const r9306 = ROWS.get("6/9306")!;
  assert.deepEqual(left, {
    kind: "left", seen: "2026-11-22T13:30:00.000Z", last_listed: "2026-11-22T13:00:00.000Z", notice: "6/9306", citation: "FDC 6/9306",
    first_seen: "2026-11-22T13:00:00.000Z", backfill: true, detail: true, list: a.rows[0].kind === "listed" ? a.rows[0].list : null,
    published_start: new Date(r9306.start!).toISOString(), published_end: new Date(r9306.end!).toISOString(),
  });
  // no verdict is written: whether it left before its published end is for the reader to work out
  assert.equal("early" in left, false);
  assert.deepEqual(Object.keys(b.next.open).sort(), ["6/6925", "6/9276"]);
  // a standing restriction has no published end
  const c = diffFaaListRead(b.next, ["6/9276"].map(entry), rowOf, T0 + 2 * HALF);
  const perm = c.rows.find((r) => r.kind === "left");
  assert.ok(perm && perm.kind === "left" && perm.notice === "6/6925" && perm.published_end === null);
});

test("a notice whose detail file was not read: listed with no row, the row follows when it is read; it can leave unread", () => {
  const none = () => undefined;
  const a = diffFaaListRead(null, ["6/6925"].map(entry), rowOf, T0);
  const b = diffFaaListRead(a.next, ["6/6925", "6/9306"].map(entry), (n) => (n === "6/9306" ? undefined : rowOf(n)), T0 + HALF);
  assert.ok(b.rows.length === 1 && b.rows[0].kind === "listed" && b.rows[0].row === null);
  assert.deepEqual(b.next.open["6/9306"], { first_seen: T0 + HALF, backfill: false, detail: false, entry: entry("6/9306"), start: null, end: null });
  const c = diffFaaListRead(b.next, ["6/6925", "6/9306"].map(entry), rowOf, T0 + 2 * HALF);
  assert.deepEqual(c.rows, [{ kind: "detail", seen: "2026-11-22T14:00:00.000Z", notice: "6/9306", citation: "FDC 6/9306", first_seen: "2026-11-22T13:30:00.000Z", row: ROWS.get("6/9306") }]);
  assert.equal(c.next.open["6/9306"].first_seen, T0 + HALF, "since when it was listed does not move");
  assert.equal(c.next.open["6/9306"].detail, true);
  assert.deepEqual(diffFaaListRead(c.next, ["6/6925", "6/9306"].map(entry), rowOf, T0 + 3 * HALF).rows, [], "the detail is written once");
  // left before its detail file was ever read: the list's own words are what the record has
  const d = diffFaaListRead(b.next, ["6/6925"].map(entry), none, T0 + 2 * HALF);
  const left = d.rows[0];
  assert.ok(left.kind === "left" && left.detail === false && left.published_end === null && left.list.description === entry("6/9306").description);
});

test("a read missing more than half of the recorded notices is not recorded the first time, and is recorded if the next read says the same", () => {
  const many = ALL.slice(0, 20);
  const a = diffFaaListRead(null, many, () => undefined, T0);
  assert.ok(Object.keys(a.next.open).length >= FAA_TFR_MASS_LEAVE_MIN);
  const cutAnswer = many.slice(0, 9); // 11 of 20 gone
  const b = diffFaaListRead(a.next, cutAnswer, () => undefined, T0 + HALF);
  assert.equal(b.refused, true);
  assert.deepEqual(b.rows, []);
  assert.deepEqual([b.next.refused_at, b.next.last_read_at], [T0 + HALF, T0], "the last recorded read does not move");
  assert.deepEqual(b.next.open, a.next.open);
  // the full list is back: nothing appeared, nothing left, and the refusal is forgotten
  const back = diffFaaListRead(b.next, many, () => undefined, T0 + 2 * HALF);
  assert.deepEqual([back.refused, back.rows.length, back.next.refused_at], [false, 0, null]);
  // the same short list again: recorded as read, each ending dated from the last RECORDED read
  const again = diffFaaListRead(b.next, cutAnswer, () => undefined, T0 + 2 * HALF);
  assert.equal(again.refused, false);
  assert.equal(again.rows.filter((r) => r.kind === "left").length, 11);
  assert.ok(again.rows.every((r) => r.kind !== "left" || (r.last_listed === "2026-11-22T13:00:00.000Z" && r.seen === "2026-11-22T14:00:00.000Z")));
  // exactly half is not "more than half"; under the minimum the rule does not apply
  assert.equal(diffFaaListRead(a.next, many.slice(0, 10), () => undefined, T0 + HALF).refused, false);
  const few = diffFaaListRead(null, many.slice(0, FAA_TFR_MASS_LEAVE_MIN - 1), () => undefined, T0);
  assert.equal(diffFaaListRead(few.next, [], () => undefined, T0 + HALF).refused, false);
});

test("the state file as read back: a malformed one is not a state", () => {
  const a = diffFaaListRead(null, NOTICES.map(entry), rowOf, T0).next;
  assert.deepEqual(readFaaTfrArchiveState(JSON.parse(JSON.stringify(a))), a);
  const bad: unknown[] = [
    null, "x", [], {}, { ...a, v: 2 }, { ...a, last_read_at: "then" }, { ...a, open: null },
    { ...a, open: { "69306": a.open["6/9306"] } },
    { ...a, open: { "6/9306": { ...a.open["6/9306"], first_seen: null } } },
    { ...a, open: { "6/9306": { ...a.open["6/9306"], entry: { ...a.open["6/9306"].entry, notice: "6/0001" } } } },
  ];
  for (const b of bad) assert.equal(readFaaTfrArchiveState(b), null, JSON.stringify(b)?.slice(0, 60));
});

test("on disk: lines are appended to the day file of our read, the state survives a restart, and nothing is rewritten", async () => {
  _resetFaaTfrArchiveForTests();
  const base = tmpBase();
  const r1 = await recordFaaTfrRead({ entries: ["6/9306", "6/6925"].map(entry), rowOf, now: T0, baseDir: base, freeBytes: plenty });
  assert.deepEqual(r1, { status: "recorded", rows: 3, listed: 2, left: 0, open: 2 });
  assert.deepEqual(lines(base, "2026-11-22").map((r) => r.kind), ["listed", "listed", "run"]);
  assert.deepEqual(faaTfrListedSince("6/9306"), { first_seen: T0, backfill: true });
  assert.equal(faaTfrListedSince("6/9276"), null);
  // "a restart": this process forgets, the next one reads the state back
  _resetFaaTfrArchiveForTests();
  assert.equal(faaTfrListedSince("6/9306"), null);
  assert.equal(await loadFaaTfrArchive(base), 2);
  assert.deepEqual(faaTfrListedSince("6/9306"), { first_seen: T0, backfill: true });
  const before = fs.readFileSync(path.join(faaTfrArchiveDir(base), "2026-11-22.jsonl"), "utf8");
  const r2 = await recordFaaTfrRead({ entries: ["6/6925", "6/9276"].map(entry), rowOf, now: T0 + HALF, baseDir: base, freeBytes: plenty });
  assert.deepEqual(r2, { status: "recorded", rows: 2, listed: 1, left: 1, open: 2 });
  const after = fs.readFileSync(path.join(faaTfrArchiveDir(base), "2026-11-22.jsonl"), "utf8");
  assert.ok(after.startsWith(before), "append-only: the earlier lines are byte for byte what they were");
  assert.deepEqual(lines(base, "2026-11-22").slice(3).map((r) => [r.kind, (r as { backfill?: boolean }).backfill]), [["listed", false], ["left", true]]);
  assert.deepEqual(faaTfrListedSince("6/9276"), { first_seen: T0 + HALF, backfill: false });
  assert.equal(faaTfrListedSince("6/9306"), null, "no longer listed");
  // the next UTC day goes to its own file
  await recordFaaTfrRead({ entries: ["6/6925", "6/9276"].map(entry), rowOf, now: T0 + 12 * 3600_000, baseDir: base, freeBytes: plenty });
  assert.deepEqual(lines(base, "2026-11-23").map((r) => r.kind), ["run"]);
  assert.deepEqual(fs.readdirSync(faaTfrArchiveDir(base)).sort(), ["2026-11-22.jsonl", "2026-11-23.jsonl", "_open.json"]);
  const v = faaTfrArchiveView({});
  assert.deepEqual([v.status, v.started_at, v.last_recorded_read, v.open, v.last?.status], ["recording", T0, T0 + 12 * 3600_000, 2, "recorded"]);
  fs.rmSync(base, { recursive: true, force: true });
});

test("what stops a read from being recorded writes no line and moves no state: low disk, a cut list, an unreadable state file", async () => {
  _resetFaaTfrArchiveForTests();
  const base = tmpBase();
  const dir = faaTfrArchiveDir(base);
  const low = await recordFaaTfrRead({ entries: NOTICES.map(entry), rowOf, now: T0, baseDir: base, freeBytes: () => 1024 ** 3 - 1 });
  assert.equal(low.status, "skipped_low_disk");
  assert.deepEqual(fs.readdirSync(dir), []);
  assert.equal(faaTfrArchiveView({}).status, "paused");
  // free space that cannot be read does not stop the record (globalScopes.volumeAllowsWrite's own rule)
  assert.equal((await recordFaaTfrRead({ entries: NOTICES.map(entry), rowOf, now: T0, baseDir: base, freeBytes: () => null })).status, "recorded");
  const state = fs.readFileSync(path.join(dir, "_open.json"), "utf8");
  const day = fs.readFileSync(path.join(dir, "2026-11-22.jsonl"), "utf8");
  // a list cut at the reader's cap: the notices beyond the cut would read as left
  const cut = await recordFaaTfrRead({ entries: ["6/9306"].map(entry), rowOf, now: T0 + HALF, cut: 3, baseDir: base, freeBytes: plenty });
  assert.equal(cut.status, "skipped_cut_list");
  assert.equal(fs.readFileSync(path.join(dir, "_open.json"), "utf8"), state);
  // a state file that exists and is not a state: recording stops rather than begin again from nothing
  fs.writeFileSync(path.join(dir, "_open.json"), "{ not json");
  const broken = await recordFaaTfrRead({ entries: NOTICES.map(entry), rowOf, now: T0 + HALF, baseDir: base, freeBytes: plenty });
  assert.equal(broken.status, "state_unreadable");
  assert.equal(fs.readFileSync(path.join(dir, "2026-11-22.jsonl"), "utf8"), day, "no second backfill");
  assert.equal(fs.readFileSync(path.join(dir, "_open.json"), "utf8"), "{ not json", "left for a person to read");
  fs.rmSync(base, { recursive: true, force: true });
});

test("day files older than two days are compressed, with the same lines; the state file and newer days are left", async () => {
  const base = tmpBase();
  const dir = faaTfrArchiveDir(base);
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, "2026-11-19.jsonl"), '{"kind":"run"}\n');
  fs.writeFileSync(path.join(dir, "2026-11-21.jsonl"), '{"kind":"run"}\n');
  fs.writeFileSync(path.join(dir, "_open.json"), "{}");
  assert.equal(await gzipOldFaaTfrDays(dir, T0), 1);
  assert.deepEqual(fs.readdirSync(dir).sort(), ["2026-11-19.jsonl.gz", "2026-11-21.jsonl", "_open.json"]);
  assert.equal(zlib.gunzipSync(fs.readFileSync(path.join(dir, "2026-11-19.jsonl.gz"))).toString("utf8"), '{"kind":"run"}\n');
  fs.rmSync(base, { recursive: true, force: true });
});

test("the switch: on by default, off by VOLTRADE_FAA_TFR_ARCHIVE", () => {
  assert.equal(faaTfrArchiveEnabled({}), true);
  for (const v of ["off", "0", "false", "no", " OFF "]) assert.equal(faaTfrArchiveEnabled({ VOLTRADE_FAA_TFR_ARCHIVE: v }), false, v);
  _resetFaaTfrArchiveForTests();
  assert.equal(faaTfrArchiveView({ VOLTRADE_FAA_TFR_ARCHIVE: "off" }).status, "off");
  assert.equal(faaTfrArchiveView({}).status, "not_yet");
});

/** a fetch that serves the fixtures; anything else is a 404 */
const fixtureFetch = (listBody: string): FaaFetch => async (url) => {
  if (url === FAA_TFR_LIST_URL) return { ok: true, status: 200, text: listBody };
  for (const n of NOTICES) if (url === faaTfrDetailUrl(n)) return { ok: true, status: 200, text: file(`detail_${n.replace("/", "_")}.xml`) };
  return { ok: false, status: 404, text: "" };
};
const listOf = (ns: readonly string[]) => JSON.stringify(LIST.filter((x: { notam_id: string }) => ns.includes(x.notam_id)));

test("end to end on the real FAA files: a pass of the list reader is recorded, and the route says since when each listed notice has been in the record", async () => {
  _resetSwimCaptureForTests();
  _resetFaaTfrListForTests();
  _resetFaaTfrArchiveForTests();
  const base = tmpBase();
  const store = sharedFaaTfrList();
  const record = () => recordFaaTfrRead({ entries: store.listed(), rowOf: (n) => store.rowOf(n), now: store.readAt!, cut: store.lastListCut, baseDir: base, freeBytes: plenty });
  try {
    await runFaaTfrPass(store, { fetch: fixtureFetch(listOf(["6/9306", "6/6925", "1/5060"])), now: () => T0, sleep: async () => {} });
    assert.equal(store.lastListCut, 0);
    assert.deepEqual(await record(), { status: "recorded", rows: 4, listed: 3, left: 0, open: 3 });
    await runFaaTfrPass(store, { fetch: fixtureFetch(listOf(["6/9306", "6/6925", "1/5060", "6/9276"])), now: () => T0 + HALF, sleep: async () => {} });
    assert.deepEqual(await record(), { status: "recorded", rows: 1, listed: 1, left: 0, open: 4 });
    const a = flightRestrictionsAnswer(T0 + HALF, {});
    const since = Object.fromEntries(a.tfrs.map((t) => [t.citation, (t as { listed_since?: unknown }).listed_since]));
    // 6/9276 starts later than six hours from the fixtures' day and is not served; 1/5060 has no outline
    assert.deepEqual(since, { "FDC 6/6925": { first_seen: T0, backfill: true }, "FDC 6/9306": { first_seen: T0, backfill: true } });
    assert.deepEqual([a.faa_list.record.status, a.faa_list.record.started_at, a.faa_list.record.last_recorded_read, a.faa_list.record.open], ["recording", T0, T0 + HALF, 4]);
    // the store's own rows are not changed by the route's added field
    assert.equal("listed_since" in (store.rowOf("6/9306") as object), false);
    // a record that does not hold a served notice adds nothing to its row
    _resetFaaTfrArchiveForTests();
    assert.equal(flightRestrictionsAnswer(T0 + HALF, {}).tfrs.some((t) => "listed_since" in t), false);
    // a list answer longer than the reader's cap says so on the store
    const long = JSON.stringify(Array.from({ length: FAA_TFR_MAX_LISTED + 2 }, (_, i) => ({ notam_id: `${i % 10}/${String(1000 + i).slice(-4)}` })));
    const s2 = new FaaTfrListStore();
    await runFaaTfrPass(s2, { fetch: fixtureFetch(long), now: () => T0, sleep: async () => {} });
    assert.equal(s2.lastListCut, 2);
  } finally {
    _resetSwimCaptureForTests();
    _resetFaaTfrListForTests();
    _resetFaaTfrArchiveForTests();
    fs.rmSync(base, { recursive: true, force: true });
  }
});

// ── reading the record back ((y2)) ────────────────────────────────────────
const DAY = 86_400_000;
const isoT = (ms: number) => new Date(ms).toISOString();
const dayOf = (ms: number) => isoT(ms).slice(0, 10);

/** three reads through the real recorder: 3 listed at T0; 6/9276 appears at T0+30min; 6/6925 leaves a day later */
async function recordThreeReads(base: string): Promise<void> {
  const rec = (ns: readonly string[], now: number) => recordFaaTfrRead({ entries: ns.map(entry), rowOf, now, baseDir: base, freeBytes: plenty });
  await rec(["6/9306", "6/6925", "1/5060"], T0);
  await rec(["6/9306", "6/6925", "1/5060", "6/9276"], T0 + HALF);
  await rec(["6/9306", "1/5060", "6/9276"], T0 + DAY);
}

test("history: what appeared and what left, newest first, with the two reads each change lies between and none of the outline or text", async () => {
  _resetFaaTfrArchiveForTests();
  const base = tmpBase();
  try {
    await recordThreeReads(base);
    const h = await readFaaTfrHistory({ days: 3, now: T0 + DAY + HALF, baseDir: base });
    assert.deepEqual([h.days_requested, h.from_day, h.to_day], [3, dayOf(T0 - DAY), dayOf(T0 + DAY)]);
    assert.deepEqual(h.events.map((e) => [e.kind, e.notice, e.seen, e.prev_read]), [
      ["left", "6/6925", isoT(T0 + DAY), isoT(T0 + HALF)],
      ["listed", "6/9276", isoT(T0 + HALF), isoT(T0)],
    ]);
    const [left, listed] = h.events;
    // the leaving line carries since when the record held it and the end the notice itself published
    const r6925 = ROWS.get("6/6925")!;
    assert.deepEqual([left.first_seen, left.first_seen_backfill, left.published_end, left.outline], [isoT(T0), true, r6925.end === null ? null : isoT(r6925.end), null]);
    assert.equal(left.list.type, entry("6/6925").type);
    const r9276 = ROWS.get("6/9276")!;
    assert.deepEqual([listed.first_seen, listed.published_start, listed.published_end, listed.outline], [null, isoT(r9276.start!), isoT(r9276.end!), !!r9276.geometry]);
    assert.equal(listed.citation, "FDC 6/9276");
    // newest day first; the day before the record began has no file and says so
    assert.deepEqual(h.days.map((d) => [d.day, d.recorded, d.listed, d.left, d.already_listed, d.run?.listed ?? null]), [
      [dayOf(T0 + DAY), true, 0, 1, 0, 3],
      [dayOf(T0), true, 1, 0, 3, 3],
      [dayOf(T0 - DAY), false, 0, 0, 0, null],
    ]);
    assert.equal(h.events_cut, 0);
    // bounded payload: no outline, no text, whatever the day files hold
    const body = JSON.stringify(h);
    assert.equal(/rings|geometry|"text"/.test(body), false);
    assert.ok(body.length < 3000, `history of two events is ${body.length} bytes`);
    assert.ok(fs.statSync(path.join(faaTfrArchiveDir(base), `${dayOf(T0)}.jsonl`)).size > 3 * body.length, "the day file is the heavy one");
  } finally { _resetFaaTfrArchiveForTests(); fs.rmSync(base, { recursive: true, force: true }); }
});

test("history: a compressed day reads the same as it did before it was compressed", async () => {
  _resetFaaTfrArchiveForTests();
  const base = tmpBase();
  try {
    await recordThreeReads(base);
    const now = T0 + DAY + HALF;
    const before = await readFaaTfrHistory({ days: 3, now, baseDir: base });
    assert.equal(await gzipOldFaaTfrDays(faaTfrArchiveDir(base), T0 + 4 * DAY), 2);
    assert.deepEqual(fs.readdirSync(faaTfrArchiveDir(base)).filter((f) => !f.startsWith("_")).sort(), [`${dayOf(T0)}.jsonl.gz`, `${dayOf(T0 + DAY)}.jsonl.gz`]);
    _resetFaaTfrArchiveForTests(); // drops the minute's kept answer
    assert.deepEqual(await readFaaTfrHistory({ days: 3, now, baseDir: base }), before);
  } finally { _resetFaaTfrArchiveForTests(); fs.rmSync(base, { recursive: true, force: true }); }
});

test("history: lines repeated after a restart are counted once, the earliest kept; a notice that leaves and is listed again is two changes", () => {
  const L = (kind: string, notice: string, at: number, extra: object = {}) => JSON.stringify({ kind, seen: isoT(at), notice, citation: `FDC ${notice}`, list: { type: "HAZARDS", facility: "ZDC", state: "VA", description: null }, ...extra });
  const h = buildFaaTfrHistory([{ day: dayOf(T0), lines: [
    L("listed", "6/0001", T0, { prev_read: isoT(T0 - HALF), backfill: false, row: null }),
    L("listed", "6/0001", T0 + HALF, { prev_read: isoT(T0), backfill: false, row: null }), // the repeat
    L("detail", "6/0001", T0 + HALF, { first_seen: isoT(T0), row: { start: T0, end: T0 + DAY, geometry: null } }),
    L("left", "6/0001", T0 + 2 * HALF, { last_listed: isoT(T0 + HALF), first_seen: isoT(T0), backfill: false, detail: true, published_start: isoT(T0), published_end: isoT(T0 + DAY) }),
    L("left", "6/0001", T0 + 3 * HALF, { last_listed: isoT(T0 + 2 * HALF), first_seen: isoT(T0), backfill: false, detail: true, published_start: null, published_end: null }), // the repeat
    L("listed", "6/0001", T0 + 4 * HALF, { prev_read: isoT(T0 + 3 * HALF), backfill: false, row: null }),
    "not json", JSON.stringify({ kind: "listed", seen: isoT(T0), notice: "../etc" }), JSON.stringify({ kind: "other", seen: isoT(T0), notice: "6/0002" }), "",
  ] }]);
  assert.deepEqual(h.events.map((e) => [e.kind, e.seen]), [["listed", isoT(T0 + 4 * HALF)], ["left", isoT(T0 + 2 * HALF)], ["listed", isoT(T0)]]);
  // the detail line that followed completes the first appearance: its period, and that the FAA publishes no outline for it
  assert.deepEqual([h.events[2].published_end, h.events[2].outline], [isoT(T0 + DAY), false]);
  // the second appearance has no detail line yet: not known, and said as null
  assert.deepEqual([h.events[0].published_end, h.events[0].outline], [null, null]);
  assert.deepEqual([h.days[0].listed, h.days[0].left, h.days[0].unreadable_lines, h.days[0].run], [2, 1, 3, null]);
});

test("history is bounded: days, events and file size each have a ceiling, and what is left out is said", async () => {
  assert.deepEqual([undefined, "", "0", "-3", "abc", "7", "31", "32", "4000", "12345", "2.5"].map(faaTfrHistoryDays),
    [FAA_TFR_HISTORY_DEFAULT_DAYS, FAA_TFR_HISTORY_DEFAULT_DAYS, FAA_TFR_HISTORY_DEFAULT_DAYS, FAA_TFR_HISTORY_DEFAULT_DAYS, FAA_TFR_HISTORY_DEFAULT_DAYS, 7, 31, FAA_TFR_HISTORY_MAX_DAYS, FAA_TFR_HISTORY_MAX_DAYS, FAA_TFR_HISTORY_DEFAULT_DAYS, FAA_TFR_HISTORY_DEFAULT_DAYS]);
  // events: the oldest are the ones cut
  const many = Array.from({ length: 30 }, (_, i) => JSON.stringify({ kind: "listed", seen: isoT(T0 + i * HALF), prev_read: null, backfill: false, notice: `6/${String(1000 + i)}`, list: {}, row: null }));
  const cut = buildFaaTfrHistory([{ day: dayOf(T0), lines: many }], 10);
  assert.deepEqual([cut.events.length, cut.events_cut, cut.events[0].notice, cut.events[9].notice, cut.days[0].listed], [10, 20, "6/1029", "6/1020", 30]);
  assert.ok(FAA_TFR_HISTORY_MAX_EVENTS <= 500 && FAA_TFR_HISTORY_MAX_DAYS <= 31);
  _resetFaaTfrArchiveForTests();
  const base = tmpBase();
  try {
    const dir = faaTfrArchiveDir(base);
    fs.mkdirSync(dir, { recursive: true });
    // a day file above the ceiling is not read, and its row says why; a request for 400 days reads 31
    fs.writeFileSync(path.join(dir, `${dayOf(T0)}.jsonl`), Buffer.alloc(FAA_TFR_HISTORY_MAX_FILE_BYTES + 1, 0x20));
    const h = await readFaaTfrHistory({ days: 400, now: T0, baseDir: base });
    assert.equal(h.days.length, FAA_TFR_HISTORY_MAX_DAYS);
    assert.deepEqual([h.days[0].day, h.days[0].recorded, h.days[0].skipped, h.events.length], [dayOf(T0), true, "too_large", 0]);
    // no record at all: every day unrecorded, nothing thrown
    const empty = await readFaaTfrHistory({ days: 2, now: T0, baseDir: path.join(base, "nowhere") });
    assert.deepEqual(empty.days.map((d) => d.recorded), [false, false]);
  } finally { _resetFaaTfrArchiveForTests(); fs.rmSync(base, { recursive: true, force: true }); }
});

test("history: a burst of readers inside a minute is one read of the files; a later reader sees what was written since", async () => {
  _resetFaaTfrArchiveForTests();
  const base = tmpBase();
  try {
    await recordThreeReads(base);
    const now = T0 + DAY + HALF;
    const a = readFaaTfrHistory({ days: 3, now, baseDir: base });
    const b = readFaaTfrHistory({ days: 3, now: now + 30_000, baseDir: base });
    assert.equal(a, b, "the same pending answer");
    assert.equal((await a).events.length, 2);
    await recordFaaTfrRead({ entries: ["1/5060", "6/9276"].map(entry), rowOf, now: now + 40_000, baseDir: base, freeBytes: plenty });
    assert.equal((await readFaaTfrHistory({ days: 3, now: now + 50_000, baseDir: base })).events.length, 2);
    const later = await readFaaTfrHistory({ days: 3, now: now + 61_000, baseDir: base });
    assert.deepEqual(later.events.map((e) => [e.kind, e.notice]), [["left", "6/9306"], ["left", "6/6925"], ["listed", "6/9276"]]);
  } finally { _resetFaaTfrArchiveForTests(); fs.rmSync(base, { recursive: true, force: true }); }
});

// The keyed mirror (/api/v1/data/flight-restrictions-history) publishes a
// schema for this answer in the OpenAPI document. Checked here against a
// real answer read from a record on disk, so the published shape cannot
// drift from what the route sends.
function schemaProblems(schema: Record<string, unknown>, value: unknown, at: string): string[] {
  const out: string[] = [];
  const t = schema.type;
  if (t === undefined) return out; // ANY
  if (t === "string") { if (typeof value !== "string") out.push(`${at}: not a string`); else if (Array.isArray(schema.enum) && !schema.enum.includes(value)) out.push(`${at}: "${value}" not in enum`); return out; }
  if (t === "integer") { if (!Number.isInteger(value)) out.push(`${at}: not an integer`); return out; }
  if (t === "boolean") { if (typeof value !== "boolean") out.push(`${at}: not a boolean`); return out; }
  if (t === "array") {
    if (!Array.isArray(value)) return [`${at}: not an array`];
    value.forEach((v, i) => out.push(...schemaProblems((schema.items ?? {}) as Record<string, unknown>, v, `${at}[${i}]`)));
    return out;
  }
  if (t === "object") {
    if (!value || typeof value !== "object" || Array.isArray(value)) return [`${at}: not an object`];
    const o = value as Record<string, unknown>;
    const props = (schema.properties ?? {}) as Record<string, Record<string, unknown>>;
    for (const k of (schema.required ?? []) as string[]) if (!(k in o)) out.push(`${at}.${k}: required and absent`);
    for (const k of Object.keys(o)) {
      if (!(k in props)) out.push(`${at}.${k}: sent but not in the published schema`);
      else out.push(...schemaProblems(props[k], o[k], `${at}.${k}`));
    }
    return out;
  }
  return [`${at}: schema type ${String(t)} not handled by this check`];
}

test("the keyed mirror's published schema is the shape the route sends: every field of a real answer is in it, with its type, and nothing required is absent", async () => {
  _resetFaaTfrArchiveForTests();
  const base = tmpBase();
  try {
    const schema = RESPONSE_DATA_SCHEMAS.voltrade_flight_restrictions_history;
    assert.ok(schema, "the mirror has a published schema");
    await recordThreeReads(base);
    const a = await flightRestrictionHistoryAnswer(T0 + DAY + HALF, "2", {}, base);
    assert.ok(a.events.length > 0 && a.days.some((d) => d.run !== null), "the answer checked holds events and a run line");
    assert.deepEqual(schemaProblems(schema, JSON.parse(JSON.stringify(a)), "data"), []);
    // an empty record (a fresh volume) has the same shape
    _resetFaaTfrArchiveForTests();
    const empty = await flightRestrictionHistoryAnswer(T0, "31", {}, path.join(base, "nowhere"));
    assert.deepEqual([empty.events.length, empty.record.status], [0, "not_yet"]);
    assert.deepEqual(schemaProblems(schema, JSON.parse(JSON.stringify(empty)), "data"), []);
    // the check itself is not vacuous: a field the schema does not name, and a wrong type, are both reported
    const bent = { ...JSON.parse(JSON.stringify(a)), extra: 1, events_cut: "0" };
    assert.deepEqual(schemaProblems(schema, bent, "data").sort(), ["data.events_cut: not an integer", "data.extra: sent but not in the published schema"]);
  } finally { _resetFaaTfrArchiveForTests(); fs.rmSync(base, { recursive: true, force: true }); }
});

test("the history route's answer: the bounded history, the record's own state, its limits, and what a line does and does not say", async () => {
  _resetFaaTfrArchiveForTests();
  const base = tmpBase();
  try {
    await recordThreeReads(base);
    const a = await flightRestrictionHistoryAnswer(T0 + DAY + HALF, "2", {}, base);
    assert.deepEqual([a.at, a.days_requested, a.events.length, a.days.length], [T0 + DAY + HALF, 2, 2, 2]);
    assert.deepEqual(a.limits, { max_days: FAA_TFR_HISTORY_MAX_DAYS, max_events: FAA_TFR_HISTORY_MAX_EVENTS });
    assert.deepEqual([a.record.status, a.record.started_at, a.record.open], ["recording", T0, 3]);
    assert.match(a.note, /whether a notice left before it is the reader's comparison/);
    assert.match(a.note, /not for flight planning/);
    assert.match(a.source, /tfr\.faa\.gov/);
    // a bad ?days falls back to the default, never to "everything"
    assert.equal((await flightRestrictionHistoryAnswer(T0 + DAY + HALF, "all", {}, base)).days_requested, FAA_TFR_HISTORY_DEFAULT_DAYS);
    // switched off: the record on disk is still readable, and the answer says the recording is off
    assert.equal((await flightRestrictionHistoryAnswer(T0 + DAY + HALF, "2", { VOLTRADE_FAA_TFR_ARCHIVE: "off" }, base)).record.status, "off");
  } finally { _resetFaaTfrArchiveForTests(); fs.rmSync(base, { recursive: true, force: true }); }
});
