// restrictionHistory.test — the "recently appeared and left" list under the
// /data map's Flight restrictions row, parsed from the shape GET
// /api/data/swim/notam/tfrs/history answers with (server/swimCapture.ts
// flightRestrictionHistoryAnswer; the list words are the FAA's own, from the
// real list file of 2026-11-22).
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import {
  parseRestrictionHistory, historyHeadWords, historyRow, historyRows, betweenWords, placeOf, HISTORY_FOOT,
  RESTRICTION_HISTORY_DAYS, RESTRICTION_HISTORY_MAX_ROWS, RESTRICTION_HISTORY_REFRESH_MS, RESTRICTION_HISTORY_URL, READ_GAP_MS,
} from "./restrictionHistory.ts";

const NOW = Date.UTC(2026, 10, 24, 9, 10);
const M = 60_000, H = 3600_000, D = 86_400_000;
const iso = (ms: number) => new Date(ms).toISOString();
const day = (ms: number) => iso(ms).slice(0, 10);
const START = Date.UTC(2026, 10, 22, 18, 43);
const LEX = { type: "HAZARDS", facility: "ZDC", state: "VA", description: "20NM W LEXINGTON, VA, Sunday, November 22, 2026 through Sunday, November 29, 2026 UTC" };
const MID = { type: "VIP", facility: "ZDC", state: "VA", description: "Middleburg, VA, Saturday, November 21, 2026 through Sunday, November 22, 2026 Local" };
const LEFT = { kind: "left", seen: iso(NOW - 2 * H), prev_read: iso(NOW - 2 * H - 30 * M), notice: "6/9141", citation: "FDC 6/9141", list: MID,
  first_seen: iso(START), first_seen_backfill: true, published_start: iso(START - D), published_end: iso(NOW - 3 * H), outline: null };
const LISTED = { kind: "listed", seen: iso(NOW - 20 * H), prev_read: iso(NOW - 20 * H - 30 * M), notice: "6/9306", citation: "FDC 6/9306", list: LEX,
  first_seen: null, first_seen_backfill: false, published_start: iso(NOW - 19 * H), published_end: iso(NOW + 5 * D), outline: true };
const dayRow = (ms: number, over: Record<string, unknown> = {}) => ({ day: day(ms), recorded: true, run: { listed: 50, with_outline: 38, without_outline: 12, detail_pending: 0 }, listed: 0, left: 0, already_listed: 0, unreadable_lines: 0, skipped: null, ...over });
const ANSWER = {
  at: NOW, days_requested: 7, from_day: day(NOW - 6 * D), to_day: day(NOW),
  days: [dayRow(NOW, { left: 1 }), dayRow(NOW - D, { listed: 1 }), dayRow(NOW - 2 * D, { already_listed: 50 }),
    ...[3, 4, 5, 6].map((n) => dayRow(NOW - n * D, { recorded: false, run: null }))],
  events: [LEFT, LISTED], events_cut: 0, limits: { max_days: 31, max_events: 500 },
  record: { status: "recording", started_at: START, last_recorded_read: NOW - 10 * M, open: 49, last: { at: NOW - 10 * M, status: "recorded", rows: 0 }, rows_this_process: 3 },
  source: "fixture", note: "fixture",
};

test("the answer is read into the panel's view: the changes newest first, the counts of the whole window, and since when there is a record", () => {
  const v = parseRestrictionHistory(ANSWER)!;
  assert.deepEqual([v.atMs, v.daysRequested, v.appeared, v.left, v.eventsCut, v.startedMs, v.status], [NOW, 7, 1, 1, 0, START, "recording"]);
  // the four days before the record began are not "missing": there was no record to miss them
  assert.deepEqual([v.daysRecorded, v.daysMissing, v.daysUnread], [3, 0, 0]);
  assert.deepEqual(v.events.map((e) => [e.kind, e.citation, e.type, e.place]), [["left", "FDC 6/9141", "VIP", "Middleburg, VA"], ["listed", "FDC 6/9306", "HAZARDS", "20NM W LEXINGTON, VA"]]);
  assert.deepEqual([v.events[0].firstSeenMs, v.events[0].firstSeenBackfill, v.events[0].publishedEndMs], [START, true, NOW - 3 * H]);
  assert.equal(v.events[1].outline, true);
});

test("another shape is not a history; a line that is not a change is left out, never guessed at", () => {
  for (const bad of [null, 3, [], {}, { events: [] }, { days: [] }, { events: {}, days: [] }]) assert.equal(parseRestrictionHistory(bad), null);
  const v = parseRestrictionHistory({ ...ANSWER, record: undefined, events: [LEFT, { ...LEFT, kind: "run" }, { ...LEFT, seen: "soon" }, { ...LEFT, citation: "" }, null, "x"] })!;
  assert.equal(v.events.length, 1);
  assert.deepEqual([v.startedMs, v.status, v.daysMissing], [null, null, 0]);
});

test("the place is the FAA's description up to its dates; a description with no dates is kept, cut at 60 characters", () => {
  assert.equal(placeOf(LEX.description), "20NM W LEXINGTON, VA");
  assert.equal(placeOf("MAYPORT, FL, Monday, November 23, 2026 through Thursday, November 26, 2026 Local"), "MAYPORT, FL");
  assert.equal(placeOf("Nationwide"), "Nationwide");
  assert.equal(placeOf(null), null);
  assert.equal(placeOf("x".repeat(80))!.length, 60);
});

test("the sentence above the rows: the window, the counts, when the record began, and each thing the record cannot say", () => {
  const v = parseRestrictionHistory(ANSWER)!;
  assert.equal(historyHeadWords(v, NOW), "This server's record of the FAA's list, last 7 days: 1 appeared, 1 left. The record began Nov 22 18:43Z; nothing before that is in it.");
  // a record older than the window does not repeat when it began
  const old = parseRestrictionHistory({ ...ANSWER, record: { ...ANSWER.record, started_at: NOW - 40 * D }, days: ANSWER.days.map((d) => ({ ...d, recorded: true })) })!;
  assert.equal(historyHeadWords({ ...old, appeared: 0, left: 0 }, NOW), "This server's record of the FAA's list, last 7 days: none appeared or left.");
  // days after the record began with no file are said; so is a file that could not be read
  const holes = parseRestrictionHistory({ ...ANSWER, record: { ...ANSWER.record, started_at: NOW - 40 * D, status: "paused" }, days: [...ANSWER.days.slice(0, 6), dayRow(NOW - 6 * D, { skipped: "too_large" })] })!;
  assert.deepEqual([holes.daysMissing, holes.daysUnread], [3, 1]);
  assert.match(historyHeadWords(holes, NOW), /3 days of this window were not recorded\. 1 day could not be read\. Recording is paused: changes since the last recorded read are not in it\.$/);
  assert.match(historyHeadWords({ ...v, status: "off" }, NOW), /Recording is switched off: nothing is being added\.$/);
  assert.match(historyHeadWords({ ...v, startedMs: null, daysRecorded: 0, appeared: 0, left: 0 }, NOW), /none appeared or left\. No day of this window has been recorded\.$/);
});

test("when a change happened is the two reads it lies between; a wide gap between them is said, and no read before it is not invented", () => {
  assert.equal(betweenWords(NOW - 30 * M, NOW, NOW), "between 08:40Z and 09:10Z");
  assert.equal(betweenWords(NOW - 20 * H - 30 * M, NOW - 20 * H, NOW), "between Nov 23 12:40Z and 13:10Z");
  // across midnight UTC both days are named
  assert.equal(betweenWords(Date.UTC(2026, 10, 22, 23, 50), Date.UTC(2026, 10, 23, 0, 20), NOW), "between Nov 22 23:50Z and Nov 23 00:20Z");
  assert.equal(betweenWords(NOW - READ_GAP_MS, NOW, NOW), "between 08:25Z and 09:10Z");
  assert.equal(betweenWords(NOW - 3.5 * H, NOW, NOW), "between 05:40Z and 09:10Z (the two reads were 3.5 h apart: the list was not read in between)");
  assert.equal(betweenWords(null, NOW, NOW), "by 09:10Z");
});

test("a row: the FAA's words, when it appeared or left, the end the notice published; never a verdict that it ended early", () => {
  const v = parseRestrictionHistory(ANSWER)!;
  const [left, listed] = v.events.map((e) => historyRow(e, NOW));
  assert.deepEqual([left.mark, left.head], ["−", "FDC 6/9141 · VIP · Middleburg, VA"]);
  assert.equal(left.body, "left the list between 06:40Z and 07:10Z; its published end was 06:10Z; on the list since before the record began");
  assert.deepEqual([listed.mark, listed.head], ["+", "FDC 6/9306 · HAZARDS · 20NM W LEXINGTON, VA"]);
  assert.equal(listed.body, "appeared between Nov 23 12:40Z and 13:10Z; published end Nov 29 09:10Z");
  for (const r of [left, listed]) assert.doesNotMatch(`${r.head} ${r.body}`, /early|cancel|late|overdue/i);
  // what is not known is said as not known
  const bare = historyRow({ ...v.events[0], publishedEndMs: null, firstSeenMs: NOW - 9 * H, firstSeenBackfill: false, place: null }, NOW);
  assert.deepEqual([bare.head, bare.body], ["FDC 6/9141 · VIP · VA", "left the list between 06:40Z and 07:10Z; no published end on record; first seen 00:10Z"]);
  assert.equal(historyRow({ ...v.events[1], outline: false, publishedEndMs: null }, NOW).body, "appeared between Nov 23 12:40Z and 13:10Z; no outline published, not drawn");
  assert.notEqual(left.key, listed.key);
});

test("the panel is bounded: at most 8 rows, and the changes not shown are counted, the server's cut included", () => {
  const many = Array.from({ length: 20 }, (_, i) => ({ ...LISTED, seen: iso(NOW - i * H), prev_read: iso(NOW - i * H - 30 * M), citation: `FDC 6/${1000 + i}`, notice: `6/${1000 + i}` }));
  const v = parseRestrictionHistory({ ...ANSWER, events: many, events_cut: 5 })!;
  const out = historyRows(v, NOW);
  assert.deepEqual([out.rows.length, out.more, out.rows[0].head.slice(0, 10)], [RESTRICTION_HISTORY_MAX_ROWS, 12 + 5, "FDC 6/1000"]);
  assert.equal(new Set(out.rows.map((r) => r.key)).size, RESTRICTION_HISTORY_MAX_ROWS);
  assert.deepEqual(historyRows(parseRestrictionHistory({ ...ANSWER, events: [] })!, NOW), { rows: [], more: 0 });
  assert.ok(RESTRICTION_HISTORY_MAX_ROWS <= 10 && RESTRICTION_HISTORY_DAYS <= 31);
  assert.equal(RESTRICTION_HISTORY_URL, "/api/data/swim/notam/tfrs/history?days=7");
  assert.equal(RESTRICTION_HISTORY_REFRESH_MS, 5 * 60_000);
  assert.match(HISTORY_FOOT, /half an hour apart, in UTC/);
});

test("in the map page: the list sits inside the Flight restrictions row only while the layer is on, its request is aborted on close, and no map event drives it", () => {
  const src = fs.readFileSync(new URL("../../pages/datamap.tsx", import.meta.url), "utf8");
  assert.match(src, /\{l\.id === "flight_restrictions" && on && <RestrictionHistoryPanel \/>\}/);
  const panel = src.slice(src.indexOf("function RestrictionHistoryPanel("), src.indexOf("/** NONSTOP TRACKED PLANES"));
  assert.ok(panel.length > 500);
  assert.match(panel, /fetch\(RESTRICTION_HISTORY_URL, \{ signal: mine\.signal \}\)/);
  assert.match(panel, /pollWhenVisible\(load, RESTRICTION_HISTORY_REFRESH_MS\)/);
  assert.match(panel, /return \(\) => \{ stop = true; ctl\?\.abort\(\); off\(\); \};/);
  assert.doesNotMatch(panel, /\.on\(\s*["']/);
  // a failed read keeps the list it had, and the first wait is a skeleton, not a blank
  assert.match(panel, /The last read failed; this is the answer before it\./);
  assert.match(panel, /vt-hist-skel/);
  assert.equal(src.split("RESTRICTION_HISTORY_URL").length - 1, 2, "one import, one request site");
});
