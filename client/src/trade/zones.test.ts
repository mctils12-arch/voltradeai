import { test } from "node:test";
import assert from "node:assert/strict";
import { TRADE_COLUMNS, TRADE_PANEL_IDS, TRADE_TEMPLATES, TRADE_TITLES, panelForTab, templateCoversAll } from "./zones";

test("every Trade panel has a title and a column, and every tier's areas place every column", () => {
  for (const id of TRADE_PANEL_IDS) assert.ok(TRADE_TITLES[id], `${id} has a title`);
  const inColumns = new Set(Object.values(TRADE_COLUMNS).flat());
  for (const id of TRADE_PANEL_IDS) if (id !== "kpi") assert.ok(inColumns.has(id), `${id} sits in a column`);
  for (const [tier, t] of Object.entries(TRADE_TEMPLATES)) {
    assert.deepEqual(templateCoversAll(t, TRADE_COLUMNS), [], `tier ${tier} leaves no panel without a slot`);
    assert.ok(t.areas[0].split(/\s+/).every((a) => a === "kpi"), `tier ${tier} puts the strip on its own first row`);
    const cols = t.columns.split(/\s+(?![^(]*\))/).length;
    for (const row of t.areas) assert.equal(row.split(/\s+/).length, cols, `tier ${tier} row "${row}" matches ${cols} columns`);
  }
  assert.equal(TRADE_TEMPLATES.xl.scrolls, false);
  assert.equal(TRADE_TEMPLATES.s.scrolls, true);
});

test("the old Trade tabs land on one panel each; the analyze view never zooms the hero", () => {
  assert.deepEqual(panelForTab("scanner", "options"), { panel: "screener", zoom: true });
  assert.deepEqual(panelForTab("research", "options"), { panel: "stock", zoom: true });
  assert.deepEqual(panelForTab("planner", "options"), { panel: "plan", zoom: true });
  assert.deepEqual(panelForTab("taxes", "options"), { panel: "plan", zoom: true });
  assert.deepEqual(panelForTab("news", "options"), { panel: "news", zoom: true });
  assert.deepEqual(panelForTab("watchlist", "options"), { panel: "watchlist", zoom: true });
  assert.deepEqual(panelForTab("analyze", "options"), { panel: null, zoom: false });
  assert.deepEqual(panelForTab("analyze", "smart-money"), { panel: "smart", zoom: false });
  assert.deepEqual(panelForTab("analyze", "structure"), { panel: "structure", zoom: false });
  assert.deepEqual(panelForTab("analyze", "etf-builder"), { panel: "etf", zoom: true });
});
