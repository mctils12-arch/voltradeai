// The /data map's scale + zoom readout, without a browser: the camera
// tracker driven by a fake map and a fake frame clock, and the markup the
// component renders (the status bar's old inline JSX, span for span).
// Run: npx tsx --test client/src/components/MapScaleReadout.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import React, { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import {
  MapScaleReadout, SCALE_VIEW_AT_START, readScaleView, trackScaleView,
  type ScaleView,
} from "./MapScaleReadout";
import { scaleReading, zoomLabel } from "../lib/mapScale";
import { getUnits, setUnits, type UnitSystem } from "../lib/units";

// tsx compiles a component's JSX to React.createElement here (the Vite build
// uses the automatic runtime and needs no import), so rendering one needs
// React in scope.
Object.assign(globalThis, { React });

/** a map with a settable camera that records its listeners and can fail */
function fakeMap(zoom: number, lat: number) {
  const listeners = new Map<string, Set<() => void>>();
  const m = {
    zoom, lat,
    reads: 0,
    failReads: false,
    failOff: false,
    getZoom() { m.reads++; if (m.failReads) throw new Error("map removed"); return m.zoom; },
    getCenter() { if (m.failReads) throw new Error("map removed"); return { lng: -96.77, lat: m.lat }; },
    on(type: string, fn: () => void) {
      if (!listeners.has(type)) listeners.set(type, new Set());
      listeners.get(type)!.add(fn);
    },
    off(type: string, fn: () => void) {
      if (m.failOff) throw new Error("map removed");
      listeners.get(type)?.delete(fn);
    },
    /** the map fires an event: every listener runs, as MapLibre does */
    fire(type: string) { for (const fn of Array.from(listeners.get(type) ?? [])) fn(); },
    listening(type: string) { return listeners.get(type)?.size ?? 0; },
    events() { return Array.from(listeners.keys()).filter((t) => listeners.get(t)!.size > 0); },
  };
  return m;
}

/** an animation-frame clock the test advances by hand */
function fakeFrames() {
  let next = 1;
  const queue = new Map<number, () => void>();
  const f = {
    requested: 0,
    cancelled: [] as number[],
    request(cb: () => void) { f.requested++; const h = next++; queue.set(h, cb); return h; },
    cancel(h: number) { f.cancelled.push(h); queue.delete(h); },
    /** one animation frame: every callback queued before it began */
    frame() { const run = Array.from(queue.values()); queue.clear(); for (const cb of run) cb(); },
    waiting() { return queue.size; },
  };
  return f;
}

test("start: the camera is reported once, before any move, and one \"move\" listener is added", () => {
  const map = fakeMap(5.25, 41), frames = fakeFrames();
  const views: ScaleView[] = [];
  trackScaleView(map, (v) => views.push(v), frames);
  assert.deepEqual(views, [{ zoom: 5.25, lat: 41 }]);
  assert.deepEqual(map.events(), ["move"], "the readout listens to \"move\" and nothing else");
  assert.equal(map.listening("move"), 1);
  assert.equal(frames.requested, 0, "an idle map asks for no frame");
});

test("a burst of move events asks for ONE frame; the camera is read in the frame, not in the event", () => {
  const map = fakeMap(4, 30), frames = fakeFrames();
  const views: ScaleView[] = [];
  trackScaleView(map, (v) => views.push(v), frames);
  const readsAfterStart = map.reads;
  for (let i = 1; i <= 40; i++) { map.zoom = 4 + i * 0.05; map.lat = 30 + i * 0.1; map.fire("move"); }
  assert.equal(frames.requested, 1, "40 move events before the frame runs, one frame requested");
  assert.equal(frames.waiting(), 1);
  assert.equal(views.length, 1, "nothing is reported inside the event handler");
  assert.equal(map.reads, readsAfterStart, "the camera is not read inside the event handler");
  frames.frame();
  assert.equal(views.length, 2, "one report for the whole burst");
  assert.deepEqual(views[1], { zoom: 4 + 40 * 0.05, lat: 30 + 40 * 0.1 }, "the frame reads the camera as it is NOW, not as it was at the first event");
  assert.equal(frames.waiting(), 0);
});

test("a continuous move: at most one report per animation frame, every frame's last camera", () => {
  const map = fakeMap(3.6, 37.5), frames = fakeFrames();
  const views: ScaleView[] = [];
  trackScaleView(map, (v) => views.push(v), frames);
  const FRAMES = 120, EVENTS_PER_FRAME = 3;
  for (let f = 1; f <= FRAMES; f++) {
    for (let e = 1; e <= EVENTS_PER_FRAME; e++) { map.zoom = 3.6 + f * 0.01 + e * 0.001; map.fire("move"); }
    frames.frame();
    assert.equal(views.length, 1 + f, `frame ${f}: exactly one report`);
    assert.equal(views[views.length - 1].zoom, 3.6 + f * 0.01 + EVENTS_PER_FRAME * 0.001);
  }
  assert.equal(frames.requested, FRAMES, `${FRAMES * EVENTS_PER_FRAME} move events, ${FRAMES} frames`);
  // the map stops: frames with no move event report nothing and request nothing
  frames.frame(); frames.frame();
  assert.equal(views.length, 1 + FRAMES);
  assert.equal(frames.requested, FRAMES);
});

test("stop: the \"move\" listener is removed and the waiting frame is cancelled", () => {
  const map = fakeMap(6, 10), frames = fakeFrames();
  const views: ScaleView[] = [];
  const stop = trackScaleView(map, (v) => views.push(v), frames);
  map.fire("move");
  assert.equal(frames.waiting(), 1, "a frame is waiting when the readout unmounts");
  stop();
  assert.equal(map.listening("move"), 0, "listener removed");
  assert.deepEqual(frames.cancelled, [1], "the waiting frame was cancelled by its handle");
  assert.equal(frames.waiting(), 0);
  frames.frame();
  map.zoom = 9; map.fire("move"); frames.frame();
  assert.equal(views.length, 1, "nothing is reported after stop");
  assert.equal(frames.requested, 1, "a later move asks for no frame: nothing is listening");
});

test("stop with no frame waiting cancels nothing; a second tracker on the same map is independent", () => {
  const map = fakeMap(6, 10), frames = fakeFrames();
  const a: ScaleView[] = [], b: ScaleView[] = [];
  const stopA = trackScaleView(map, (v) => a.push(v), frames);
  const stopB = trackScaleView(map, (v) => b.push(v), frames);
  stopA();
  assert.deepEqual(frames.cancelled, []);
  assert.equal(map.listening("move"), 1, "only the stopped tracker's listener went");
  map.zoom = 7; map.fire("move"); frames.frame();
  assert.equal(a.length, 1);
  assert.deepEqual(b, [{ zoom: 6, lat: 10 }, { zoom: 7, lat: 10 }]);
  stopB();
  assert.equal(map.listening("move"), 0);
});

test("a map that cannot be read reports nothing, never throws, and tracking resumes when it can", () => {
  const map = fakeMap(8, 20), frames = fakeFrames();
  const views: ScaleView[] = [];
  map.failReads = true;
  assert.equal(readScaleView(map), null);
  const stop = trackScaleView(map, (v) => views.push(v), frames);
  assert.deepEqual(views, [], "no made-up view while the map is unreadable");
  map.fire("move"); frames.frame();
  assert.deepEqual(views, []);
  map.failReads = false;
  map.fire("move");
  assert.equal(frames.requested, 2, "a failed read does not leave the frame latch stuck");
  frames.frame();
  assert.deepEqual(views, [{ zoom: 8, lat: 20 }]);
  assert.deepEqual(readScaleView(map), { zoom: 8, lat: 20 });
  stop();
});

test("stop survives a map that was already removed (off() throws) and still cancels the frame", () => {
  const map = fakeMap(8, 20), frames = fakeFrames();
  const stop = trackScaleView(map, () => {}, frames);
  map.fire("move");
  map.failOff = true;
  assert.doesNotThrow(stop);
  assert.deepEqual(frames.cancelled, [1]);
});

/** the status bar's inline JSX as it was in pages/datamap.tsx before the
 *  readout moved out, written as the markup it produced */
function oldInlineMarkup(zoom: number, lat: number, units: UnitSystem): string {
  const sc = scaleReading(zoom, lat, units === "imperial" ? "imperial" : "metric");
  return `<span class="vt-statusbar-scale" aria-label="scale ${sc.label}">` +
    `<span class="vt-statusbar-bar" style="width:${Math.round(sc.widthPx)}px"></span>${sc.label}</span>` +
    `<span class="vt-statusbar-sep">·</span>` +
    `<span class="vt-statusbar-zoom">${zoomLabel(zoom)}</span>`;
}
const render = (map: ReturnType<typeof fakeMap> | null) => renderToStaticMarkup(createElement(MapScaleReadout, { map }));

test("before the map is ready the readout shows the map's start view, as the page did", () => {
  assert.deepEqual(SCALE_VIEW_AT_START, { zoom: 3.6, lat: 37.5 });
  assert.equal(getUnits(), "imperial", "node has no stored preference: the site default");
  const html = render(null);
  assert.equal(html, oldInlineMarkup(3.6, 37.5, "imperial"));
  // by hand: 156543.034 m/px x cos(37.5 deg) / 2^3.6 = 10,242 m/px; a 90 px
  // target is 573 mi, the nice number below it is 500 mi, and 500 mi is 79 px
  assert.equal(html,
    '<span class="vt-statusbar-scale" aria-label="scale 500 mi"><span class="vt-statusbar-bar" style="width:79px"></span>500 mi</span>' +
    '<span class="vt-statusbar-sep">·</span><span class="vt-statusbar-zoom">z3.6</span>');
});

test("the markup is the old inline JSX at every camera, in both unit systems", () => {
  const cameras: Array<[number, number]> = [[0.8, 0], [3.6, 37.5], [5.25, -41], [9.42, 35], [12, 60.2], [16.7, 40.7128], [19, -84.9], [22, 12]];
  try {
    for (const units of ["imperial", "metric"] as const) {
      setUnits(units);
      for (const [zoom, lat] of cameras) {
        assert.equal(render(fakeMap(zoom, lat)), oldInlineMarkup(zoom, lat, units), `z${zoom} lat ${lat} ${units}`);
      }
    }
    // the unit switch changes the label's unit, not the camera
    setUnits("metric");
    assert.match(render(fakeMap(9.42, 35)), /aria-label="scale \d[\d,]* (m|km)"/);
    setUnits("imperial");
    assert.match(render(fakeMap(9.42, 35)), /aria-label="scale \d[\d,]* (ft|mi)"/);
  } finally {
    setUnits("imperial");
  }
});

test("a readout mounted with a ready map reads the live camera on its first render", () => {
  // the status bar remounts when the space frame hands the viewport back: the
  // first paint must be the camera, not the start view
  const html = render(fakeMap(9.42, 35));
  assert.match(html, /<span class="vt-statusbar-zoom">z9\.4<\/span>/);
  assert.notEqual(html, render(null));
  // an unreadable map falls back to the start view rather than throwing
  const dead = fakeMap(9.42, 35);
  dead.failReads = true;
  assert.equal(render(dead), render(null));
});
