// The hook's pure parts: the flag transitions its handlers apply, the hold
// ledger, the pop-out gate and the slot geometry. The hook itself needs a DOM
// (BroadcastChannel, window.open, element rects) and is exercised by the
// visual harness; everything that decides state is here, headless.
import { test } from "node:test";
import assert from "node:assert/strict";
import { applyHold, applyPin, applyHoldAll, applyFloat, applyDock, applyZoom, applyPop, applyDockAll, syncHeld, canPopOut, pageSlot, screenSlot, DEFAULT_CHANNEL, DEFAULT_HASH_PREFIX, DEFAULT_WINDOW_PREFIX, WIN_ID, type HeldSnapshot } from "./useCockpitLayout";
import { emptyFlags, type LayoutFlags } from "./zoneModel";

type Id = "a" | "b" | "c";
const E = () => emptyFlags<Id>();

test("hold / pin toggle their lists and touch nothing else; the page-wide hold flips", () => {
  const f1 = applyHold(E(), "a");
  assert.deepEqual(f1, { ...E(), held: ["a"] });
  assert.deepEqual(applyHold(f1, "a"), E());
  const p = applyPin(E(), "b");
  assert.deepEqual(p, { ...E(), pinned: ["b"] });
  assert.deepEqual(applyPin(p, "b"), E());
  assert.deepEqual(applyHoldAll(E()), { ...E(), hold: true });
  assert.deepEqual(applyHoldAll(applyHoldAll(E())), E());
});

test("float is idempotent (the same object back when already floating); dock removes the window and ends the zoom only when it was that panel's", () => {
  const f = applyFloat(E(), "a");
  assert.deepEqual(f.floating, ["a"]);
  assert.equal(applyFloat(f, "a"), f, "no new object, no rect work");
  assert.deepEqual(applyFloat(f, "b").floating, ["a", "b"]);
  const z: LayoutFlags<Id> = { ...E(), floating: ["a", "b"], zoomed: "c" };
  assert.deepEqual(applyDock(z, "a"), { ...E(), floating: ["b"], zoomed: "c" });
  assert.deepEqual(applyDock({ ...E(), zoomed: "c" }, "c"), E());
  assert.deepEqual(applyDock(E(), "a"), E(), "docking what is not open changes nothing");
});

test("zoom toggles the zoomed panel and pulls it out of the floating windows", () => {
  const f: LayoutFlags<Id> = { ...E(), floating: ["a", "b"] };
  const z = applyZoom(f, "a");
  assert.deepEqual(z, { ...E(), floating: ["b"], zoomed: "a" });
  assert.deepEqual(applyZoom(z, "a"), { ...E(), floating: ["b"] });
  assert.deepEqual(applyZoom(z, "c"), { ...E(), floating: ["b"], zoomed: "c" });
});

test("pop (its own browser window) leaves the page's windows and the zoom; Esc docks everything; held, pinned and the page hold survive both", () => {
  const f: LayoutFlags<Id> = { hold: true, floating: ["a", "b"], zoomed: "a", held: ["a"], pinned: ["b"] };
  assert.deepEqual(applyPop(f, "a"), { ...f, floating: ["b"], zoomed: null });
  assert.deepEqual(applyPop(f, "b"), { ...f, floating: ["a"] });
  assert.deepEqual(applyDockAll(f), { ...f, floating: [], zoomed: null });
});

test("syncHeld: a newly held panel is snapshotted once (body, clock, tick); a still-held one keeps its snapshot; a released one is dropped", () => {
  const held = new Map<Id, HeldSnapshot>();
  syncHeld(held, ["a"], (id) => `${id}-1`, 1000, 5);
  assert.deepEqual(held.get("a"), { since: 1000, body: "a-1", ticks: 5 });
  syncHeld(held, ["a", "b"], (id) => `${id}-2`, 2000, 7);
  assert.deepEqual(held.get("a"), { since: 1000, body: "a-1", ticks: 5 }, "the first look stays");
  assert.deepEqual(held.get("b"), { since: 2000, body: "b-2", ticks: 7 });
  syncHeld(held, ["b"], (id) => `${id}-3`, 3000, 9);
  assert.equal(held.has("a"), false);
  assert.deepEqual([...held.keys()], ["b"]);
  syncHeld(held, [], () => null, 4000, 10);
  assert.equal(held.size, 0);
});

test("canPopOut: not on touch, not on the phone tier", () => {
  assert.equal(canPopOut(false, "xl"), true);
  assert.equal(canPopOut(false, "l"), true);
  assert.equal(canPopOut(false, "m"), true);
  assert.equal(canPopOut(false, "s"), false);
  assert.equal(canPopOut(true, "l"), false);
});

test("pageSlot / screenSlot: a slot in page coordinates (the root's offset removed, a default when unmounted) and in screen coordinates (the browser chrome above the viewport added, never subtracted)", () => {
  const box = { left: 300, top: 156, width: 400, height: 240 };
  assert.deepEqual(pageSlot(box, { left: 0, top: 56 }), { x: 300, y: 100, w: 400, h: 240 });
  assert.deepEqual(pageSlot(box, undefined), { x: 300, y: 156, w: 400, h: 240 });
  assert.deepEqual(pageSlot(null, { left: 0, top: 56 }), { x: 100, y: 100, w: 600, h: 400 });
  assert.deepEqual(screenSlot(box, { screenX: 50, screenY: 20, outerHeight: 1000, innerHeight: 900 }), { x: 350, y: 276, w: 400, h: 240 });
  assert.deepEqual(screenSlot({ left: 0, top: 0, width: 10, height: 10 }, { screenX: 0, screenY: 0, outerHeight: 800, innerHeight: 900 }), { x: 0, y: 0, w: 10, h: 10 });
});

test("the defaults keep an unconfigured cockpit apart from the bot's windows and channel", () => {
  assert.notEqual(DEFAULT_CHANNEL, "vt-bot-panes");
  assert.notEqual(DEFAULT_HASH_PREFIX, "#/bot/pane/");
  assert.equal(DEFAULT_WINDOW_PREFIX, "vt-pane");
  assert.match(WIN_ID, /^[a-z0-9]{1,8}$/);
});
