// cardLayout.test.ts — the site card's height contract.
//
// This card has now produced TWO live reports in two days, and they pull in
// opposite directions:
//
//   2026-08-11 "the card looks cut off" — the card had `overflow: visible`,
//   so once its always-shown chrome outgrew 60vh the lower rows painted
//   outside the rounded box onto the bare map. Fixed by clipping to 60vh and
//   making the flights list scroll internally.
//
//   2026-08-12 "the log data is smashed up to where you can't read it ...
//   when you click DETAILS it should drop down bigger" — the SAME 60vh cap,
//   applied while DETAILS is EXPANDED, left the details body a ~60px sliver
//   with its own scrollbar. The first fix's cap became the second bug.
//
// Per CLAUDE.md's REPAIRS MUST RATCHET rule, both invariants are pinned here
// so a future change cannot satisfy one by reintroducing the other. These are
// source assertions on index.css rather than a rendered measurement because
// the visual harness does not open an aircraft card, and an unenforced CSS
// invariant is the reason this recurred.

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const cssRaw = readFileSync(new URL("../index.css", import.meta.url), "utf-8");
// Strip comments before parsing: these blocks carry long explanatory comments
// BETWEEN declarations, which breaks any "previous char was a semicolon"
// boundary assumption.
const css = cssRaw.replace(/\/\*[\s\S]*?\*\//g, "");

/** Pull a declaration value out of a specific selector's block. */
function decl(selector: string, prop: string): string | null {
  const i = css.indexOf(selector);
  if (i < 0) return null;
  const open = css.indexOf("{", i);
  const close = css.indexOf("}", open);
  if (open < 0 || close < 0) return null;
  const block = css.slice(open + 1, close);
  const m = new RegExp(`(?:^|[;{\\s])${prop}\\s*:\\s*([^;]+)`).exec(block);
  return m ? m[1].trim() : null;
}

const vhOf = (v: string | null): number | null => {
  if (!v) return null;
  const m = /([\d.]+)\s*d?vh/.exec(v);
  return m ? Number(m[1]) : null;
};

// ── invariant 1: the card is still clipped (the 2026-08-11 fix) ────────────

test("the card clips to its box — nothing may paint outside the rounded card", () => {
  // Regression guard for "the card looks cut off": with overflow visible, the
  // Follow / freshness / DETAILS rows spilled onto the bare map.
  assert.equal(decl(".vt-site-card {", "overflow"), "hidden");
});

test("the flights list still scrolls internally rather than pushing chrome out", () => {
  assert.match(decl(".vt-trips-list {", "overflow-y") ?? "", /auto|scroll/);
  assert.equal(decl(".vt-trips-list {", "min-height"), "0", "a flex child needs min-height:0 to actually scroll");
});

// ── invariant 2: expanding DETAILS actually gives it room (2026-08-12) ─────

test("the height cap is STATE-DEPENDENT — compact closed, tall when expanded", () => {
  // The whole point of the second fix. A single cap for both states is what
  // made the details body unreadable.
  const collapsed = vhOf(decl(".vt-site-card {", "max-height"));
  const expanded = vhOf(decl(".vt-site-card:not(.vt-card-closed) {", "max-height"));
  assert.ok(collapsed != null, ".vt-site-card must declare a max-height");
  assert.ok(expanded != null, "expanded state must declare its own max-height");
  assert.ok(
    expanded! > collapsed!,
    `expanded cap ${expanded}vh must exceed the collapsed ${collapsed}vh — otherwise DETAILS has nowhere to grow`,
  );
});

test("the collapsed card does not fill the screen", () => {
  // "you want the screen not to be filled when it pops up" — a popup that
  // covers the map defeats the map.
  const collapsed = vhOf(decl(".vt-site-card {", "max-height"));
  assert.ok(collapsed != null && collapsed <= 65, `collapsed cap ${collapsed}vh is too tall for a popup`);
});

test("the expanded card still leaves the map visible", () => {
  // Growth has a ceiling too: a full-screen takeover is a different bug.
  const expanded = vhOf(decl(".vt-site-card:not(.vt-card-closed) {", "max-height"));
  assert.ok(expanded != null && expanded <= 90, `expanded cap ${expanded}vh leaves no map visible`);
});

test("the flights list YIELDS space when details are open, instead of competing", () => {
  // The details body and the flights list were both trying to occupy the
  // same slack. With the list pinned at 24vh the body got the remainder,
  // which was almost nothing.
  const base = vhOf(decl(".vt-trips-list {", "max-height"));
  const open = vhOf(decl(".vt-site-card:not(.vt-card-closed) .vt-trips-list {", "max-height"));
  assert.ok(base != null && open != null, "both trips-list caps must be declared");
  assert.ok(open! < base!, `open-state list cap ${open}vh must be under the ${base}vh default`);
});

test("the details body has a floor high enough to read several fact rows", () => {
  // The live symptom was a ~60px scroll sliver. A fact row is ~34px (9.5px
  // label + 12.5px value + margins), so the floor must clear at least three.
  const v = decl(".vt-site-card:not(.vt-card-closed) .vt-card-detbody {", "min-height");
  assert.ok(v, "the expanded details body must declare a min-height");
  const px = Number(/([\d.]+)px/.exec(v!)?.[1] ?? NaN);
  assert.ok(px >= 100, `details body floor ${px}px is still a sliver — needs room for 3+ fact rows`);
});

test("the details body grows into available space when open", () => {
  const flex = decl(".vt-site-card:not(.vt-card-closed) .vt-card-detbody {", "flex");
  assert.match(flex ?? "", /^1\s+1/, "the details body must be the growth region (flex: 1 1 auto) once open");
});

// ── invariant 3: the facts grid is readable at the card's real width ───────

test("the facts grid collapses to one column rather than crushing labels", () => {
  // At the card's 352px width a hard `1fr 1fr` gave each column ~150px, and
  // a label like "COUNTRY (ICAO ALLOC)" wrapped to three lines.
  const cols = decl(".vt-card-facts {", "grid-template-columns");
  assert.ok(cols, ".vt-card-facts must declare grid-template-columns");
  assert.match(cols!, /auto-fit|auto-fill/, "the facts grid must be responsive, not a hard 2-up");
  assert.match(cols!, /minmax\(\s*1[45]\d px?\s*,|minmax\(\s*1[45]\d/, "minmax floor should be ~150px");
});

// ── phone ─────────────────────────────────────────────────────────────────

test("the phone bottom sheet gets the same expanded allowance", () => {
  // The human reviews on a Galaxy S24; the phone path is the primary one.
  const idx = css.indexOf(".vt-site-card:not(.vt-site-card-min):not(.vt-card-closed)");
  assert.ok(idx > 0, "the phone sheet must also raise its cap when details are open");
  const expanded = vhOf(decl(".vt-site-card:not(.vt-site-card-min):not(.vt-card-closed) {", "max-height"));
  assert.ok(expanded != null && expanded > 74, `phone expanded cap ${expanded}dvh must exceed the 74dvh collapsed sheet`);
});
