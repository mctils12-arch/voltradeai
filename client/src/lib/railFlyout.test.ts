// railFlyout.test.ts — where a menu opened from the /data map's icon rail goes.
//
// Owner report 2026-11-19 (1440px desktop): "when you click on the mountain,
// it covers the other icon and moves down. Why does it move? It's where I
// have my mouse." The base-map-style menu was an in-flow sibling of its
// button inside a centring inline-flex row: the 91px-tall menu re-centred the
// 44px button 23px lower, over the time-machine button beneath it.
//
// REPAIRS MUST RATCHET: the geometry below is the measured page (rail buttons
// at left 12, tops 12/62/112/162/212, 44px; map page 1440x844, 768x968,
// 390x732), and the source assertions pin the structure that makes the bug
// impossible — a fixed slot, an out-of-flow menu.
// Run: npx tsx --test client/src/lib/railFlyout.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  RAIL_FLYOUT_GAP,
  RAIL_FLYOUT_MARGIN,
  RAIL_FLYOUT_MIN_WIDTH,
  isBackdrop,
  overlapArea,
  placeRailFlyout,
  railFlyoutMaxWidth,
  type Box,
} from "./railFlyout";

const RAIL_TOPS = [12, 62, 112, 162, 212];
const railButton = (top: number): Box => ({ left: 12, top, width: 44, height: 44 });
const STYLE = railButton(162);
/** the other four rail buttons — obstacles for the style menu */
const otherButtons = RAIL_TOPS.filter((t) => t !== STYLE.top).map(railButton);

/** The style menu's size model: two stacked pill rows that wrap. Row 1 is
 *  Natural/Night/Minimal, row 2 the five chart views. Pill widths are chosen
 *  so each row's one-line width equals the measured desktop row (226px and
 *  483px; the menu measured 483x91): a row box adds 8px of padding+border,
 *  pills sit 2px apart, a line is 34px tall, the rows are 6px apart. */
const ROWS = [[72, 62, 80], [85, 118, 116, 80, 68]];
function styleMenu(maxWidth: number): { width: number; height: number } {
  let width = 0, height = 0;
  for (const row of ROWS) {
    let lines = 1, line = 0, widest = 0;
    for (const pill of row) {
      const next = line === 0 ? pill : line + 2 + pill;
      if (next + 8 > maxWidth && line > 0) { lines++; widest = Math.max(widest, line); line = pill; }
      else line = next;
    }
    widest = Math.max(widest, line);
    width = Math.max(width, Math.min(maxWidth, widest + 8));
    height += lines * 34 + (lines - 1) * 2 + 8;
  }
  return { width, height: height + 6 };
}

const boxOf = (p: { left: number; top: number }, maxWidth: number): Box => {
  const s = styleMenu(maxWidth);
  return { left: p.left, top: p.top, width: s.width, height: s.height };
};

test("the size model reproduces the measured desktop menu (one line per row)", () => {
  const s = styleMenu(1370);
  assert.equal(s.width, 483);
  assert.equal(s.height, 90);
});

for (const [label, container] of [
  ["1440 desktop", { width: 1440, height: 844 }],
  ["768 tablet", { width: 768, height: 968 }],
  ["390 phone", { width: 390, height: 732 }],
] as const) {
  test(`${label}: the menu opens right of the rail, level with its button, on the page`, () => {
    const p = placeRailFlyout({ anchor: STYLE, container, measure: styleMenu, obstacles: otherButtons });
    assert.equal(p.mode, "beside");
    assert.equal(p.left, STYLE.left + STYLE.width + RAIL_FLYOUT_GAP);
    assert.equal(p.top, STYLE.top);
    const b = boxOf(p, p.maxWidth);
    assert.ok(b.left + b.width <= container.width - RAIL_FLYOUT_MARGIN, "inside the right edge");
    assert.ok(b.top + b.height <= container.height - RAIL_FLYOUT_MARGIN, "inside the bottom edge");
    for (const btn of RAIL_TOPS.map(railButton)) assert.equal(overlapArea(b, btn), 0, "covers no rail button");
  });
}

test("390 phone: the menu takes the width beside the rail and wraps inside it", () => {
  const p = placeRailFlyout({ anchor: STYLE, container: { width: 390, height: 732 }, measure: styleMenu, obstacles: otherButtons });
  assert.equal(p.maxWidth, 390 - 62 - RAIL_FLYOUT_MARGIN);
  assert.ok(styleMenu(p.maxWidth).width <= p.maxWidth);
  assert.ok(styleMenu(p.maxWidth).height > styleMenu(1370).height, "the chart row wrapped");
});

test("a rail button near the bottom: the menu is pulled up, never off the page", () => {
  const anchor = railButton(700);
  const container = { width: 390, height: 732 };
  const p = placeRailFlyout({ anchor, container, measure: styleMenu });
  const b = boxOf(p, p.maxWidth);
  assert.equal(b.top + b.height, container.height - RAIL_FLYOUT_MARGIN);
  assert.ok(p.top >= RAIL_FLYOUT_MARGIN);
  assert.equal(p.mode, "beside");
});

test("a page shorter than the menu: capped to the page (it scrolls inside)", () => {
  const container = { width: 390, height: 120 };
  const p = placeRailFlyout({ anchor: railButton(12), container, measure: styleMenu });
  assert.equal(p.maxHeight, 120 - 2 * RAIL_FLYOUT_MARGIN);
  assert.equal(p.top, RAIL_FLYOUT_MARGIN);
});

test("768 tablet with the nav cluster and the layers panel in the row: the menu narrows to the free width beside its button", () => {
  // measured 2026-11-19: nav-cluster toolbar from x=292 at the button's row,
  // layers panel from x=448 for the full height
  const toolbar: Box = { left: 292, top: 206, width: 114, height: 22 };
  const layers: Box = { left: 448, top: 0, width: 320, height: 968 };
  const obstacles = [...otherButtons, toolbar, layers];
  const p = placeRailFlyout({ anchor: STYLE, container: { width: 768, height: 968 }, measure: styleMenu, obstacles });
  assert.equal(p.mode, "beside");
  assert.equal(p.top, STYLE.top);
  assert.equal(p.maxWidth, toolbar.left - RAIL_FLYOUT_GAP - 62);
  const b = boxOf(p, p.maxWidth);
  for (const o of obstacles) assert.equal(overlapArea(b, o), 0);
});

test("the time machine open where the menu would go: the menu steps aside, covering nothing", () => {
  // .vt-timescrub-panel: left 64, top 162 — exactly the style menu's spot
  const timescrub: Box = { left: 64, top: 162, width: 300, height: 240 };
  const obstacles = [...otherButtons, timescrub];
  const container = { width: 1440, height: 844 };
  const p = placeRailFlyout({ anchor: STYLE, container, measure: styleMenu, obstacles });
  assert.equal(p.mode, "shifted");
  const b = boxOf(p, p.maxWidth);
  for (const o of obstacles) assert.equal(overlapArea(b, o), 0);
  assert.ok(b.left >= 62 && b.top >= RAIL_FLYOUT_MARGIN);
  assert.ok(b.left + b.width <= container.width - RAIL_FLYOUT_MARGIN && b.top + b.height <= container.height - RAIL_FLYOUT_MARGIN);
  // the smallest move: up beside the rail (96px) rather than 308px to the right
  assert.equal(p.left, 62);
  assert.equal(p.top, timescrub.top - RAIL_FLYOUT_GAP - b.height);
});

test("a full-height panel beside the rail (analyst pane): the menu opens past it, on its button's row", () => {
  const analyst: Box = { left: 64, top: 12, width: 360, height: 820 };
  const obstacles = [...otherButtons, analyst];
  const p = placeRailFlyout({ anchor: STYLE, container: { width: 1440, height: 844 }, measure: styleMenu, obstacles });
  assert.equal(p.mode, "shifted");
  assert.equal(p.top, STYLE.top);
  assert.equal(p.left, analyst.left + analyst.width + RAIL_FLYOUT_GAP);
  assert.equal(overlapArea(boxOf(p, p.maxWidth), analyst), 0);
});

test("nowhere free: it still opens beside its button, flagged as an overlap", () => {
  const wall: Box = { left: 60, top: 0, width: 330, height: 732 };
  const p = placeRailFlyout({ anchor: STYLE, container: { width: 390, height: 732 }, measure: styleMenu, obstacles: [wall] });
  assert.equal(p.mode, "overlap");
  assert.equal(p.left, 62);
  assert.equal(p.top, STYLE.top);
});

test("nowhere free on a crowded tablet: the overlap covers as little as it can", () => {
  // measured 768 with the analyst pane, the layers panel and the legend all
  // open: no free spot fits the menu. At full width it would lie over the
  // analyst pane AND the layers panel; narrowed to stop at the layers panel
  // it covers the pane only.
  const analyst: Box = { left: 64, top: 12, width: 316, height: 388 };
  const layers: Box = { left: 448, top: 0, width: 320, height: 968 };
  const legend: Box = { left: 18, top: 434, width: 272, height: 470 };
  const obstacles = [...otherButtons, analyst, layers, legend];
  const p = placeRailFlyout({ anchor: STYLE, container: { width: 768, height: 968 }, measure: styleMenu, obstacles });
  assert.equal(p.mode, "overlap");
  assert.equal(p.left, 62);
  assert.equal(p.top, STYLE.top);
  const b = boxOf(p, p.maxWidth);
  assert.equal(overlapArea(b, layers), 0, "stops short of the layers panel");
  for (const btn of otherButtons) assert.equal(overlapArea(b, btn), 0, "still covers no rail button");
});

test("a phone bottom sheet is a control to step around; only a page-filling box is a backdrop", () => {
  const page390 = { width: 390, height: 732 };
  // measured 2026-11-19: the time machine's sheet at 390 is 374x414.6 — 54%
  // of the page. Treated as a backdrop (the first cut's 50% line), the menu
  // opened over its top edge.
  assert.equal(isBackdrop({ width: 374, height: 414.6 }, page390), false);
  assert.equal(isBackdrop({ width: 390, height: 732 }, page390), true, "the map canvas");
  assert.equal(isBackdrop({ width: 320, height: 968 }, { width: 768, height: 968 }), false, "the layers panel");
  // and with the sheet counted, the menu (320x152 on the phone, where every
  // pill is a 44px target) clears it: slid up beside the rail
  const sheet: Box = { left: 8, top: 307.4, width: 374, height: 414.6 };
  const phoneMenu = () => ({ width: 320, height: 152 });
  const p = placeRailFlyout({ anchor: STYLE, container: page390, measure: phoneMenu, obstacles: [...otherButtons, sheet] });
  assert.equal(p.mode, "shifted");
  assert.equal(overlapArea({ left: p.left, top: p.top, width: 320, height: 152 }, sheet), 0);
  assert.equal(p.left, 62);
  assert.equal(p.top, 149, "6px above the sheet, 13px above its button's top");
});

test("width floor and touching boxes", () => {
  assert.equal(railFlyoutMaxWidth(STYLE, 100), RAIL_FLYOUT_MIN_WIDTH);
  assert.equal(railFlyoutMaxWidth(STYLE, 1440), 1440 - 62 - RAIL_FLYOUT_MARGIN);
  // boxes that only share an edge do not overlap (menu flush against a control)
  assert.equal(overlapArea({ left: 0, top: 0, width: 10, height: 10 }, { left: 10, top: 0, width: 10, height: 10 }), 0);
  assert.equal(overlapArea({ left: 0, top: 0, width: 10, height: 10 }, { left: 5, top: 5, width: 10, height: 10 }), 25);
});

// ── the structure that made the bug impossible ────────────────────────────
const css = readFileSync(new URL("../index.css", import.meta.url), "utf-8").replace(/\/\*[\s\S]*?\*\//g, "");
const page = readFileSync(new URL("../pages/datamap.tsx", import.meta.url), "utf-8");
const menu = readFileSync(new URL("../components/RailMenu.tsx", import.meta.url), "utf-8");

function block(selector: string): string {
  const m = new RegExp(`(?:^|\\})\\s*${selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\s*\\{([^}]*)\\}`).exec(css);
  assert.ok(m, `${selector} block not found in index.css`);
  return m[1];
}

test("the rail slot is a fixed 44px box and its menu is out of flow", () => {
  const slot = block(".vt-rail-menu");
  assert.match(slot, /position:\s*absolute/);
  assert.match(slot, /width:\s*44px/);
  assert.match(slot, /height:\s*44px/);
  const fly = block(".vt-rail-flyout");
  assert.match(fly, /position:\s*absolute/, "an in-flow menu is what pushed the button");
});

test("the style menu's slot no longer lays its menu out beside the button", () => {
  const sw = block(".vt-preset-switch");
  assert.doesNotMatch(sw, /inline-flex|align-items/, "the centring flex row that moved the button 23px");
  assert.match(sw, /top:\s*162px/, "fourth slot of the rail");
});

test("the style menu is a RailMenu and every rail button is enumerable", () => {
  assert.match(page, /<RailMenu\s+id="style"/);
  assert.doesNotMatch(page, /onMouseEnter=\{\(\) => setPresetOpen\(true\)\}/, "hover-open on the wrapper raced the click on touch (tap opened then closed)");
  for (const k of ["fullscreen", "globe", "analyst", "timescrub"]) {
    assert.match(page, new RegExp(`data-vt-rail-btn="${k}"`), `rail button ${k}`);
  }
  assert.match(menu, /data-vt-rail-btn=\{id\}/);
});

test("the menu closes on Escape and on a press outside, and only a mouse hovers it open", () => {
  assert.match(menu, /e\.key !== "Escape"/);
  assert.match(menu, /addEventListener\("pointerdown", onDown, true\)/);
  assert.match(menu, /e\.pointerType === "mouse"/);
});

test("phone: every option in the style menu is a 44px touch target", () => {
  const i = css.indexOf(".vt-preset-pop .vt-preset-pill { min-height: 44px");
  assert.ok(i > 0, "phone touch-target rule for the menu's pills");
  assert.match(css.slice(css.lastIndexOf("@media", i), i), /max-width:\s*480px/);
});
