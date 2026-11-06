import { test } from "node:test";
import assert from "node:assert/strict";
import { tierFor, TEMPLATES, gridStyle, toggleIn, isPinned, EMPTY_FLAGS, clampRect, initialFloatRect, snapEdges } from "./zoneModel";

test("tiers: big screen first, phone is its own tier", () => {
  assert.equal(tierFor(2560), "xl");
  assert.equal(tierFor(1440), "l");
  assert.equal(tierFor(1280), "l");
  assert.equal(tierFor(1024), "m");
  assert.equal(tierFor(768), "m");
  assert.equal(tierFor(390), "s");
  assert.equal(tierFor(NaN), "l");
});

test("every template row names the same number of columns as its track list", () => {
  for (const [tier, t] of Object.entries(TEMPLATES)) {
    const cols = t.columns.split(/\s+(?![^(]*\))/).length;
    for (const row of t.areas) assert.equal(row.split(" ").length, cols, `${tier}: "${row}"`);
    assert.ok(t.areas.some((r) => r.includes("centre")) && t.areas.some((r) => r.includes("timeline")));
  }
  assert.equal(TEMPLATES.l.scrolls, false, "desktop never scrolls the page");
  assert.equal(TEMPLATES.s.scrolls, true);
  assert.match(gridStyle("l").gridTemplateAreas, /^"kpi kpi kpi" "left centre right" "timeline timeline timeline"$/);
});

test("flags: toggling and the held-implies-pinned rule", () => {
  assert.deepEqual(toggleIn([], "book"), ["book"]);
  assert.deepEqual(toggleIn(["book"], "book"), []);
  assert.equal(isPinned({ ...EMPTY_FLAGS, held: ["book"] }, "book"), true);
  assert.equal(isPinned(EMPTY_FLAGS, "book"), false);
});

test("float rects stay on the page and snap to edges", () => {
  const page = { w: 1440, h: 844 };
  const r = initialFloatRect({ x: 300, y: 80, w: 700, h: 600 }, page);
  assert.ok(r.w <= page.w * 0.78 && r.h <= page.h * 0.78 && r.x >= 0 && r.y >= 0);
  const off = clampRect({ x: 1500, y: 900, w: 400, h: 300 }, page);
  assert.equal(off.x, page.w - 120);
  assert.equal(off.y, page.h - 120);
  assert.deepEqual(snapEdges({ x: 7, y: 836 - 300 + 5, w: 400, h: 300 }, page), { x: 0, y: page.h - 300, w: 400, h: 300 });
});

test("overview (2026-11-06): the view defaults to the overview, its grid rows name the same columns, the chart spans two columns and both views keep the essentials", async () => {
  const { botModeFrom, OVERVIEW_TEMPLATES, overviewGridStyle, MODE_PANELS, PANEL_IDS } = await import("./zoneModel");
  assert.equal(botModeFrom(null), "overview");
  assert.equal(botModeFrom("junk"), "overview");
  assert.equal(botModeFrom("cockpit"), "cockpit");
  for (const [tier, t] of Object.entries(OVERVIEW_TEMPLATES)) {
    const cols = t.columns.split(/\s+(?![^(]*\))/).length;
    for (const row of t.areas) assert.equal(row.split(" ").length, cols, `${tier}: "${row}"`);
    assert.match(t.areas.join(" "), /perf perf/, `${tier}: the chart spans the centre and right columns`);
    assert.match(t.rows, /minmax\(\d{3}px, 1\.3fr\)/, `${tier}: the chart's row keeps a floor`);
  }
  // re-pinned 2026-11-06 ([PRODUCT] the daily line card): a fourth row, the daily line across the centre and right columns
  // (was /^"kpi kpi kpi" "left perf perf" "left why picks"$/)
  assert.match(overviewGridStyle("l").gridTemplateAreas, /^"kpi kpi kpi" "left perf perf" "left why picks" "left daily daily"$/);
  // the daily line's row: a floor and a share both below the chart's, at the same width — the chart stays the largest panel
  for (const [tier, t] of Object.entries(OVERVIEW_TEMPLATES)) {
    const rows = [...t.rows.matchAll(/minmax\((\d+)px, ([\d.]+)fr\)/g)].map((m) => ({ floor: Number(m[1]), fr: Number(m[2]) }));
    assert.equal(rows.length, 3, `${tier}: the chart, why/picks and daily rows each keep a floor`);
    const [chart, , daily] = rows;
    assert.ok(daily.floor < chart.floor && daily.fr < chart.fr, `${tier}: the daily line's row (${daily.floor}px, ${daily.fr}fr) stays below the chart's (${chart.floor}px, ${chart.fr}fr)`);
  }
  for (const id of ["issues", "book", "orders"] as const) {
    assert.ok(MODE_PANELS.overview.includes(id) && MODE_PANELS.cockpit.includes(id), `${id} is in both views`);
  }
  for (const id of [...MODE_PANELS.overview, ...MODE_PANELS.cockpit]) assert.ok(PANEL_IDS.includes(id), `${id} is a panel id`);
  // re-pinned 2026-11-06: + daily (was ["book", "issues", "orders", "perf", "picks", "why"])
  assert.deepEqual([...MODE_PANELS.overview].sort(), ["book", "daily", "issues", "orders", "perf", "picks", "why"]);
  assert.ok(!MODE_PANELS.cockpit.includes("daily"), "the cockpit keeps its daily scoreboard; the daily line is the overview's");
});
