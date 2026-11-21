// SEC XBRL fundamentals record, the pure half: which frames a night reads,
// how SEC's answer is read, and what is written when a frame changes.
import test from "node:test";
import assert from "node:assert/strict";
import {
  FrameShrunkError, QUARTERS_BACK, SHRINK_GUARD_MIN_PREV, XBRL_CONCEPTS, YEARS_BACK,
  diffFrame, dueNow, etDay, etHour, filingUrl, frameFileName, frameKey, frameRoster, frameUrl, parseFrame,
  quarterPeriods, yearPeriods, type FrameFact, type FrameSnapshot,
} from "./xbrlFramesCore";

const NIGHT = "2026-11-21T07:10:00.000Z";
const NI = XBRL_CONCEPTS.find((c) => c.tag === "NetIncomeLoss")!;
const ASSETS = XBRL_CONCEPTS.find((c) => c.tag === "Assets")!;

test("quarters: the quarter in progress and the six before it, newest first, across a year end", () => {
  assert.deepEqual(quarterPeriods(new Date("2026-11-21T07:00:00Z"), "duration"),
    ["CY2026Q4", "CY2026Q3", "CY2026Q2", "CY2026Q1", "CY2025Q4", "CY2025Q3", "CY2025Q2"]);
  assert.deepEqual(quarterPeriods(new Date("2027-01-02T07:00:00Z"), "instant").slice(0, 3),
    ["CY2027Q1I", "CY2026Q4I", "CY2026Q3I"]);
  assert.equal(quarterPeriods(new Date(NIGHT), "instant").length, QUARTERS_BACK + 1);
});

test("years: the last two full calendar years, never the year in progress", () => {
  assert.deepEqual(yearPeriods(new Date(NIGHT)), ["CY2025", "CY2024"]);
  assert.equal(yearPeriods(new Date(NIGHT)).length, YEARS_BACK);
});

test("roster: duration concepts read quarters and years, balances read quarter ends only; every key is unique", () => {
  const roster = frameRoster(new Date(NIGHT));
  const duration = XBRL_CONCEPTS.filter((c) => c.kind === "duration").length;
  const instant = XBRL_CONCEPTS.filter((c) => c.kind === "instant").length;
  assert.equal(roster.length, duration * (QUARTERS_BACK + 1 + YEARS_BACK) + instant * (QUARTERS_BACK + 1));
  assert.equal(new Set(roster.map((r) => r.key)).size, roster.length);
  assert.equal(new Set(roster.map((r) => frameFileName(r.key))).size, roster.length, "no two frames share a snapshot file");
  assert.ok(roster.some((r) => r.key === "us-gaap/NetIncomeLoss/USD/CY2025"));
  assert.ok(!roster.some((r) => r.key === "us-gaap/Assets/USD/CY2025"), "a balance has no full-year frame");
  assert.ok(roster.some((r) => r.key === "dei/EntityCommonStockSharesOutstanding/shares/CY2026Q3I"));
});

test("the address is SEC's documented frames path", () => {
  assert.equal(frameKey(NI, "CY2026Q3"), "us-gaap/NetIncomeLoss/USD/CY2026Q3");
  assert.equal(frameUrl(ASSETS, "CY2026Q3I"), "https://data.sec.gov/api/xbrl/frames/us-gaap/Assets/USD/CY2026Q3I.json");
  assert.equal(frameFileName("us-gaap/Assets/USD/CY2026Q3I"), "us_gaap_Assets_USD_CY2026Q3I");
});

// Two rows as data.sec.gov served them on 2026-11-21 (CY2026Q3I Assets, CY2026Q2 GrossProfit).
const REAL = {
  taxonomy: "us-gaap", tag: "Assets", ccp: "CY2026Q3I", uom: "USD", pts: 2,
  data: [
    { accn: "0001104659-26-111786", cik: 1750, entityName: "AAR CORP", loc: "US-IL", end: "2026-08-31", val: 3426900000 },
    { accn: "0001193125-26-338211", cik: 2098, entityName: "ACME UNITED CORP", loc: "US-CT", start: "2026-04-01", end: "2026-06-30", val: 26688000 },
  ],
};

test("parse: SEC's own rows are read whole, with the company's name kept apart from the fact", () => {
  const p = parseFrame(REAL);
  assert.equal(p.rejected, 0);
  assert.deepEqual(p.facts[0], { cik: 1750, accn: "0001104659-26-111786", end: "2026-08-31", val: 3426900000 });
  assert.equal(p.facts[1].start, "2026-04-01");
  assert.deepEqual(p.entities.get(1750), { name: "AAR CORP", loc: "US-IL" });
});

test("parse: a row that is not a whole fact is counted and left out; a repeated company is kept once", () => {
  const p = parseFrame({ data: [
    REAL.data[0],
    { ...REAL.data[0] },
    { accn: "bad", cik: 5, end: "2026-09-30", val: 1 },
    { accn: "0001104659-26-111786", cik: 6.5, end: "2026-09-30", val: 1 },
    { accn: "0001104659-26-111786", cik: 7, end: "2026-09", val: 1 },
    { accn: "0001104659-26-111786", cik: 8, end: "2026-09-30", val: "12" },
    { accn: "0001104659-26-111786", cik: 9, end: "2026-09-30", val: Number.NaN },
    null,
  ] });
  assert.equal(p.facts.length, 1);
  assert.equal(p.duplicates, 1);
  assert.equal(p.rejected, 6);
});

test("parse: an answer with no data list is an error, not an empty frame", () => {
  assert.throws(() => parseFrame({ message: "Too many requests" }), /no data list/);
  assert.throws(() => parseFrame(null), /no data list/);
});

const F = (cik: number, val: number, accn = "0000000001-26-000001", end = "2026-09-30"): FrameFact => ({ cik, accn, end, val });

test("diff: the first read of a frame writes every fact marked backfill", () => {
  const d = diffFrame("f", null, [F(1, 10), F(2, 20)], NIGHT);
  assert.equal(d.rows.length, 2);
  assert.ok(d.rows.every((r) => r.kind === "first" && r.backfill === true && r.seen === NIGHT));
  assert.deepEqual(d.stats, { held: 2, first: 2, revised: 0, dropped: 0, refiled: 0, backfill: true });
  assert.deepEqual(d.next, { "1": [10, "0000000001-26-000001", "2026-09-30"], "2": [20, "0000000001-26-000001", "2026-09-30"] });
});

test("diff: a frame read before as empty is not a backfill when its first companies arrive", () => {
  const d = diffFrame("f", {}, [F(1, 10)], NIGHT);
  assert.equal(d.rows[0].kind, "first");
  assert.equal(d.rows[0].backfill, undefined, "the night is a real first sighting");
  assert.equal(d.stats.backfill, false);
});

test("diff: an unchanged frame writes nothing", () => {
  const prev: FrameSnapshot = diffFrame("f", null, [F(1, 10), F(2, 20)], NIGHT).next;
  const d = diffFrame("f", prev, [F(1, 10), F(2, 20)], "2026-11-22T07:10:00.000Z");
  assert.equal(d.rows.length, 0);
  assert.deepEqual(d.next, prev);
});

test("diff: the same number named by a later filing is counted and not written (the yearly comparative column)", () => {
  // Measured 2026-11-21: 86% of CY2025Q3 net-income facts carry a 2026 accession number.
  const prev = diffFrame("f", null, [F(1, 10, "0000000001-25-000001")], NIGHT).next;
  const d = diffFrame("f", prev, [F(1, 10, "0000000001-26-000009")], "2026-11-22T07:10:00.000Z");
  assert.equal(d.rows.length, 0);
  assert.equal(d.stats.refiled, 1);
  assert.equal(d.next["1"][1], "0000000001-26-000009", "the snapshot follows SEC so a later revision names the right prior filing");
});

test("diff: a changed value is a revised row carrying what was held and the filing that held it", () => {
  // CIK 811830, net income for the quarter ended September 2025: the 2026 10-Q's comparative
  // column differs from the number first filed 2025-11-03 (session cross-check, 2026-11-21).
  const prev = diffFrame("f", null, [F(811830, 455000000, "0000811830-25-000050")], NIGHT).next;
  const d = diffFrame("f", prev, [F(811830, 456000000, "0000811830-26-000061")], "2026-11-22T07:10:00.000Z");
  assert.deepEqual(d.rows, [{
    seen: "2026-11-22T07:10:00.000Z", frame: "f", cik: 811830, kind: "revised", accn: "0000811830-26-000061", end: "2026-09-30",
    val: 456000000, prev_val: 455000000, prev_accn: "0000811830-25-000050", prev_end: "2026-09-30",
  }]);
  assert.equal(d.rows[0].backfill, undefined);
});

test("diff: a changed period end with the same value is a revision too", () => {
  const prev = diffFrame("f", null, [F(1, 10, "0000000001-26-000001", "2026-09-30")], NIGHT).next;
  const d = diffFrame("f", prev, [F(1, 10, "0000000001-26-000002", "2026-09-27")], NIGHT);
  assert.equal(d.rows[0].kind, "revised");
  assert.equal(d.rows[0].prev_end, "2026-09-30");
});

test("diff: a company that leaves the frame is a dropped row with the value it had", () => {
  const prev = diffFrame("f", null, [F(1, 10), F(2, 20), F(3, 30)], NIGHT).next;
  const d = diffFrame("f", prev, [F(1, 10), F(3, 30)], NIGHT);
  assert.equal(d.rows.length, 1);
  assert.deepEqual({ kind: d.rows[0].kind, cik: d.rows[0].cik, val: d.rows[0].val, prev_val: d.rows[0].prev_val }, { kind: "dropped", cik: 2, val: null, prev_val: 20 });
  assert.equal(d.next["2"], undefined);
});

test("diff: an answer that lost more than half its companies is refused, never recorded as thousands of drops", () => {
  const many = Array.from({ length: SHRINK_GUARD_MIN_PREV + 20 }, (_, i) => F(i + 1, i));
  const prev = diffFrame("f", null, many, NIGHT).next;
  assert.throws(() => diffFrame("f", prev, many.slice(0, 40), NIGHT), FrameShrunkError);
  // a small frame may shrink: three companies to one is two drops
  const small = diffFrame("f", null, [F(1, 1), F(2, 2), F(3, 3)], NIGHT).next;
  assert.equal(diffFrame("f", small, [F(1, 1)], NIGHT).stats.dropped, 2);
  // losing fewer than half is recorded
  assert.equal(diffFrame("f", prev, many.slice(0, 100), NIGHT).stats.dropped, 20);
});

test("the nightly window is read in ET, once per ET day", () => {
  const at = new Date("2026-11-21T07:10:00.000Z"); // 02:10 ET (EST)
  assert.equal(etDay(at), "2026-11-21");
  assert.equal(etHour(at), 2);
  assert.equal(dueNow(at, null, 2, 5), true);
  assert.equal(dueNow(at, "2026-11-20", 2, 5), true);
  assert.equal(dueNow(at, "2026-11-21", 2, 5), false, "already read today");
  assert.equal(dueNow(new Date("2026-11-21T05:10:00.000Z"), null, 2, 5), false, "00:10 ET is before the window");
  assert.equal(dueNow(new Date("2026-11-21T10:00:00.000Z"), null, 2, 5), false, "05:00 ET is after it");
  // daylight time: 06:30Z in July is 02:30 EDT
  assert.equal(etHour(new Date("2026-07-10T06:30:00.000Z")), 2);
});

test("a filing link is SEC's index page for the accession number", () => {
  assert.equal(filingUrl(1750, "0001104659-26-111786"),
    "https://www.sec.gov/Archives/edgar/data/1750/000110465926111786/0001104659-26-111786-index.htm");
});
