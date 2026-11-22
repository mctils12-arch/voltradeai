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
  type FaaTfrArchiveRow, type FaaTfrArchiveState,
} from "./faaTfrArchive";
import {
  parseFaaList, parseFaaDetail, FaaTfrListStore, runFaaTfrPass, faaTfrDetailUrl, sharedFaaTfrList, _resetFaaTfrListForTests,
  FAA_TFR_LIST_URL, FAA_TFR_MAX_LISTED, type FaaFetch, type FaaListEntry, type FaaTfrRow,
} from "./faaTfrList";
import { flightRestrictionsAnswer, _resetSwimCaptureForTests } from "./swimCapture";

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
