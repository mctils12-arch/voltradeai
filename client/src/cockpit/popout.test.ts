import { test } from "node:test";
import assert from "node:assert/strict";
import { paneIdFromHash, paneUrl, paneName, paneFeatures, parseChannelMsg, paneLifecycleMsg, windowId, PANE_MIN } from "./popout";

const IDS = ["alpha", "beta"] as const;
const PREFIX = "#/x/pane/";

test("paneIdFromHash: only <prefix><known id> names a pane, whatever the cockpit's prefix and ids", () => {
  assert.equal(paneIdFromHash("#/x/pane/alpha", PREFIX, IDS), "alpha");
  assert.equal(paneIdFromHash("#/x/pane/beta?x=1", PREFIX, IDS), "beta");
  assert.equal(paneIdFromHash("#/x/pane/beta/extra", PREFIX, IDS), "beta");
  assert.equal(paneIdFromHash("#/x/pane/", PREFIX, IDS), null);
  assert.equal(paneIdFromHash("#/x/pane/gamma", PREFIX, IDS), null);
  assert.equal(paneIdFromHash("#/bot/pane/alpha", PREFIX, IDS), null, "another cockpit's prefix");
  assert.equal(paneIdFromHash("", PREFIX, IDS), null);
});

test("paneUrl and paneName: the same app under the cockpit's prefix, one named window per panel per cockpit", () => {
  assert.equal(paneUrl("alpha", PREFIX, { origin: "https://voltradeai.com", pathname: "/app" }), "https://voltradeai.com/app#/x/pane/alpha");
  assert.equal(paneName("alpha", "vt-x-pane"), "vt-x-pane-alpha");
  assert.notEqual(paneName("alpha", "vt-x-pane"), paneName("alpha", "vt-pane"), "two cockpits never share a window name");
});

test("paneFeatures: sized from the slot, clamped to the minimum and the screen", () => {
  assert.match(paneFeatures({ x: 900, y: 150, w: 480, h: 320 }, { w: 1920, h: 1080 }), /^popup=yes,width=552,height=416,left=900,top=150,resizable=yes,scrollbars=yes$/);
  assert.match(paneFeatures({ x: 10, y: 10, w: 200, h: 100 }, { w: 1920, h: 1080 }), new RegExp(`width=${PANE_MIN.w},height=${PANE_MIN.h},`));
  assert.match(paneFeatures(null, { w: 1920, h: 1080 }), /width=760,height=520,left=580,top=280,/);
});

test("parseChannelMsg: a kind and a sender, nothing less; paneLifecycleMsg: pane-closed / pane-open naming one of the cockpit's ids", () => {
  assert.deepEqual(parseChannelMsg({ t: "focus", sym: "QQQ", from: "a1" }), { t: "focus", sym: "QQQ", from: "a1" });
  assert.equal(parseChannelMsg({ t: "focus", sym: "QQQ" }), null, "no sender");
  assert.equal(parseChannelMsg({ from: "a1" }), null, "no kind");
  assert.equal(parseChannelMsg({ t: 5, from: "a1" }), null);
  assert.equal(parseChannelMsg("focus"), null);
  assert.equal(parseChannelMsg(null), null);
  assert.deepEqual(paneLifecycleMsg({ t: "pane-closed", id: "alpha", from: "b2" }, IDS), { t: "pane-closed", id: "alpha", from: "b2" });
  assert.deepEqual(paneLifecycleMsg({ t: "pane-open", id: "beta", from: "b2" }, IDS), { t: "pane-open", id: "beta", from: "b2" });
  assert.equal(paneLifecycleMsg({ t: "pane-closed", id: "gamma", from: "b2" }, IDS), null, "not this cockpit's panel");
  assert.equal(paneLifecycleMsg({ t: "pane-closed", from: "b2" }, IDS), null);
  assert.equal(paneLifecycleMsg({ t: "focus", id: "alpha", from: "b2" }, IDS), null, "the cockpit's own message is not a lifecycle");
});

test("windowId: short, url-safe, different each call", () => {
  const a = windowId(), b = windowId();
  assert.match(a, /^[a-z0-9]{1,8}$/);
  assert.notEqual(a, b);
});
