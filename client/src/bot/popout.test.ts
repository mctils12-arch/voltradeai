import { test } from "node:test";
import assert from "node:assert/strict";
import { paneIdFromHash, paneUrl, paneName, paneFeatures, parsePaneMsg, PANE_MIN, PANE_PREFIX } from "./popout";

test("paneIdFromHash: only #/bot/pane/<known id> names a pane", () => {
  assert.equal(paneIdFromHash("#/bot/pane/chart"), "chart");
  assert.equal(paneIdFromHash("#/bot/pane/kpi"), "kpi");
  assert.equal(paneIdFromHash("#/bot/pane/book?x=1"), "book");
  assert.equal(paneIdFromHash("#/bot/pane/chart/extra"), "chart");
  assert.equal(paneIdFromHash("#/bot"), null);
  assert.equal(paneIdFromHash("#/bot/pane/"), null);
  assert.equal(paneIdFromHash("#/bot/pane/nope"), null);
  assert.equal(paneIdFromHash("#/data/pane/chart"), null);
  assert.equal(paneIdFromHash(""), null);
});

test("paneUrl and paneName: the same app, one named window per panel", () => {
  assert.equal(paneUrl("chart", { origin: "https://voltradeai.com", pathname: "/app" }), `https://voltradeai.com/app${PANE_PREFIX}chart`);
  assert.equal(paneName("book"), "vt-pane-book");
  assert.notEqual(paneName("book"), paneName("chart"));
});

test("paneFeatures: a popup sized from the slot, clamped to the minimum and the screen, placed over the slot", () => {
  const f = paneFeatures({ x: 900, y: 150, w: 480, h: 320 }, { w: 1920, h: 1080 });
  assert.match(f, /^popup=yes,width=552,height=416,left=900,top=150,resizable=yes,scrollbars=yes$/);
  // a tiny slot opens at the minimum
  const small = paneFeatures({ x: 10, y: 10, w: 200, h: 100 }, { w: 1920, h: 1080 });
  assert.match(small, new RegExp(`width=${PANE_MIN.w},height=${PANE_MIN.h},`));
  // a slot near the edge is pulled back inside the screen
  const edge = paneFeatures({ x: 1800, y: 1000, w: 480, h: 320 }, { w: 1920, h: 1080 });
  assert.match(edge, /left=1368,top=664,/);
  // a screen smaller than the minimum caps at the screen, never beyond it
  const tiny = paneFeatures({ x: 0, y: 0, w: 900, h: 700 }, { w: 400, h: 300 });
  assert.match(tiny, /width=480,height=360,left=0,top=0,/);
  // no slot: a default size, centred
  assert.match(paneFeatures(null, { w: 1920, h: 1080 }), /width=760,height=520,left=580,top=280,/);
});

test("parsePaneMsg: accepts our three messages and nothing else", () => {
  assert.deepEqual(parsePaneMsg({ t: "focus", sym: "QQQ", from: "a1" }), { t: "focus", sym: "QQQ", from: "a1" });
  assert.deepEqual(parsePaneMsg({ t: "focus", sym: null, from: "a1" }), { t: "focus", sym: null, from: "a1" });
  assert.deepEqual(parsePaneMsg({ t: "pane-closed", id: "chart", from: "b2" }), { t: "pane-closed", id: "chart", from: "b2" });
  assert.deepEqual(parsePaneMsg({ t: "pane-open", id: "book", from: "b2" }), { t: "pane-open", id: "book", from: "b2" });
  assert.equal(parsePaneMsg({ t: "pane-closed", id: "nope", from: "b2" }), null);
  assert.equal(parsePaneMsg({ t: "focus", sym: 5, from: "a1" }), null);
  assert.equal(parsePaneMsg({ t: "focus", sym: "QQQ" }), null);
  assert.equal(parsePaneMsg("focus"), null);
  assert.equal(parsePaneMsg(null), null);
});
