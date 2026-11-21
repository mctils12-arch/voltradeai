import test from "node:test";
import assert from "node:assert/strict";
import {
  ageLabel, changeLine, companyTitle, conceptLabel, conceptRows, factRows, fmtAmount, fmtPct, periodLabel, periodOrder,
  revisionPct, runLine, type FundamentalsPayload,
} from "./fundamentalsView";

const NOW = Date.parse("2026-11-21T12:00:00Z");
const CONCEPTS = [
  { tag: "NetIncomeLoss", label: "Net income", unit: "USD", kind: "duration" },
  { tag: "Assets", label: "Total assets", unit: "USD", kind: "instant" },
];
const run = (over: Record<string, unknown> = {}) => ({
  day: "2026-11-21", started: "2026-11-21T07:10:00Z", finished: "2026-11-21T07:11:00Z", status: "ok", detail: "",
  frames: { roster: 96, read: 89, not_published: 7, failed: 0 }, rows: { first: 0, revised: 0, dropped: 0, refiled: 0, backfill: 0 }, companies: 7606, ...over,
});

test("a period reads as a person would say it", () => {
  assert.equal(periodLabel("CY2026Q3"), "Q3 2026");
  assert.equal(periodLabel("CY2026Q3I"), "end of Q3 2026");
  assert.equal(periodLabel("CY2025"), "year 2025");
  assert.equal(periodLabel("odd"), "odd");
  assert.ok(periodOrder("CY2026Q1") > periodOrder("CY2025"));
  assert.ok(periodOrder("CY2025") > periodOrder("CY2025Q4"));
  assert.ok(periodOrder("CY2025Q4") > periodOrder("CY2025Q3I"));
});

test("amounts keep their sign and their unit", () => {
  assert.equal(fmtAmount(404548000000, "USD"), "$404.5B");
  assert.equal(fmtAmount(-3531000, "USD"), "-$3.53M");
  assert.equal(fmtAmount(-11828, "USD"), "-$11.83K");
  assert.equal(fmtAmount(950, "USD"), "$950");
  assert.equal(fmtAmount(14532182000, "shares"), "14.53B shares");
  assert.equal(fmtAmount(0, "USD"), "$0");
  assert.equal(fmtAmount(null, "USD"), "—");
  assert.equal(fmtAmount(2.5e12, "USD"), "$2.50T");
});

test("a revision's size is measured against the number first held", () => {
  assert.equal(fmtPct(revisionPct(455000000, 456000000)), "+0.22%");
  assert.equal(fmtPct(revisionPct(-100, -150)), "-50%");
  assert.equal(fmtPct(revisionPct(100, 103.4)), "+3.4%");
  assert.equal(revisionPct(0, 5), null);
  assert.equal(fmtPct(null), "—");
  assert.equal(fmtPct(0), "0.00%");
});

test("age is said in hours, then days", () => {
  assert.equal(ageLabel("2026-11-21T11:30:00Z", NOW), "under an hour ago");
  assert.equal(ageLabel("2026-11-21T07:11:00Z", NOW), "4 h ago");
  assert.equal(ageLabel("2026-11-18T07:11:00Z", NOW), "3 d ago");
  assert.equal(ageLabel(null, NOW), "age unknown");
});

test("the run line states what was read, how old it is, and when a read was not whole", () => {
  assert.equal(runLine({ last_run: null }, NOW), "No read has run yet. The first one runs tonight between 02:00 and 05:00 ET.");
  assert.equal(runLine({ last_run: run() }, NOW), "Last read 2026-11-21 (4 h ago): 89 of 96 tables read, 7 not published by SEC yet. 7,606 companies on record.");
  assert.match(runLine({ last_run: run({ frames: { roster: 96, read: 86, not_published: 7, failed: 3 } }) }, NOW), /3 failed and will be read again tonight/);
  assert.match(runLine({ last_run: run({ status: "skipped_low_disk" }) }, NOW), /did not run: the storage volume/);
  assert.match(runLine({ last_run: run({ status: "refused", frames: { roster: 96, read: 4, not_published: 0, failed: 1 } }) }, NOW), /SEC refused the request\. 4 of 96/);
});

test("the starting copy is called that, never 'new numbers'", () => {
  const first: FundamentalsPayload = { last_run: run({ rows: { first: 281870, revised: 0, dropped: 0, refiled: 0, backfill: 281870 } }) };
  assert.match(changeLine(first), /starting copy: 281,870 numbers already published before recording began/);
  const later: FundamentalsPayload = { last_run: run({ rows: { first: 412, revised: 9, dropped: 1, refiled: 300, backfill: 0 } }) };
  assert.equal(changeLine(later), "That night: 412 new numbers, 9 changed, 1 removed by SEC.");
  assert.equal(changeLine({ last_run: null }), "");
});

test("a concept's line shows its newest well-filled quarter and leaves the starting copy out of 'added'", () => {
  const p: FundamentalsPayload = { concepts: CONCEPTS, coverage: [
    { tag: "NetIncomeLoss", period: "CY2026Q4", state: "read", held: 12, first: 3, revised: 0, dropped: 0, backfill: false },
    { tag: "NetIncomeLoss", period: "CY2026Q3", state: "read", held: 4618, first: 40, revised: 2, dropped: 0, backfill: false },
    { tag: "NetIncomeLoss", period: "CY2025", state: "read", held: 6000, first: 6000, revised: 0, dropped: 0, backfill: true },
    { tag: "Assets", period: "CY2026Q4I", state: "not_published", held: 0, first: 0, revised: 0, dropped: 0, backfill: false },
  ] };
  assert.deepEqual(conceptRows(p), [
    { tag: "NetIncomeLoss", label: "Net income", newestPeriod: "CY2026Q3", companies: 4618, added: 43, revised: 2 },
    { tag: "Assets", label: "Total assets", newestPeriod: null, companies: 0, added: 0, revised: 0 },
  ]);
});

test("a company's numbers are grouped by concept, newest period first", () => {
  const c = { cik: 320193, name: "Apple Inc.", loc: "US-CA", ticker: "AAPL", facts: [
    { tag: "Assets", label: "Total assets", unit: "USD", period: "CY2026Q3I", end: "2026-09-26", val: 404548000000, filing: "f1" },
    { tag: "NetIncomeLoss", label: "Net income", unit: "USD", period: "CY2026Q2", end: "2026-06-27", val: 1, filing: "f2" },
    { tag: "NetIncomeLoss", label: "Net income", unit: "USD", period: "CY2026Q3", end: "2026-09-26", val: 2, filing: "f3" },
  ] };
  assert.deepEqual(factRows(c, CONCEPTS).map((r) => [r.label, r.period, r.amount]), [
    ["Net income", "Q3 2026", "$2"], ["Net income", "Q2 2026", "$1"], ["Total assets", "end of Q3 2026", "$404.5B"],
  ]);
  assert.equal(companyTitle(c), "Apple Inc. (AAPL)");
  assert.equal(companyTitle({ cik: 5, name: null, loc: null, ticker: null, facts: [] }), "CIK 5");
  assert.equal(conceptLabel("Assets", CONCEPTS), "Total assets");
  assert.equal(conceptLabel("Other", CONCEPTS), "Other");
});
