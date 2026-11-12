// SCALE program stability queue item 1 (research/scale_program.md, filed
// 2026-07-31): the camera state behind the status bar's scale + zoom readout
// was DataMapPage state, set from a map "move" listener, so every frame of
// every camera move re-rendered the whole page. The readout is now
// components/MapScaleReadout.tsx, which follows the camera itself.
// Source-inspection style, like datamap.symbols.test.ts: this pins the wiring
// (the page holds no camera state and adds no "move" listener for it, the
// status bar renders the component, the component owns the listener, the
// frame throttle and its cleanup). What the tracker does is tested with a fake
// map in components/MapScaleReadout.test.ts.
// Run: npx tsx --test client/src/pages/datamap.scaleReadout.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const pageSrc = fs.readFileSync(path.join(here, "datamap.tsx"), "utf8");
const readoutPath = path.join(here, "..", "components", "MapScaleReadout.tsx");
// read inside each test: where the component is missing, the page pins still
// report on their own
const readoutSrc = () => fs.readFileSync(readoutPath, "utf8");

// Whole-file checks go through these two, not assert.match: a failed
// assert.match prints the string it searched, and datamap.tsx is 16,000 lines.
const has = (src: string, re: RegExp, msg: string) => assert.ok(re.test(src), msg);
const lacks = (src: string, re: RegExp, msg: string) => assert.ok(!re.test(src), msg);

/** every handler a source registers on the map's "move" event, by name */
function moveHandlers(src: string): { named: string[]; registrations: number } {
  return {
    named: Array.from(src.matchAll(/\.on\(\s*["']move["']\s*,\s*([A-Za-z_$][\w$]*)\s*\)/g), (m) => m[1]).sort(),
    registrations: (src.match(/\.on\(\s*["']move["']/g) ?? []).length,
  };
}

test("RATCHET: DataMapPage holds no camera state for the scale readout", () => {
  lacks(pageSrc, /\bsetScaleView\b/, "setScaleView is back in datamap.tsx: a camera move would re-render the whole page again");
  lacks(pageSrc, /\bscaleView\b/, "the readout's camera state belongs to components/MapScaleReadout.tsx, not the page");
  lacks(pageSrc, /\b(scaleReading|zoomLabel)\s*\(/, "the page must not compute the scale reading itself; the readout component does");
});

test("RATCHET: the page's map \"move\" listeners are the two LOD appliers and nothing else", () => {
  const { named, registrations } = moveHandlers(pageSrc);
  assert.deepEqual(named, ["applyLod", "applyMarkerLod"],
    "datamap.tsx registers a different set of \"move\" handlers. \"move\" fires on every frame of a camera move: " +
    "a handler that sets page state re-renders the whole page per frame (the scale readout did until 2026-11-12). " +
    "Put a per-frame camera reader in its own component (components/MapScaleReadout.tsx is the pattern), " +
    "then list a handler here only if it sets no state per frame.");
  assert.equal(registrations, named.length, "a \"move\" handler is registered inline (not by name), so this ratchet cannot see what it does");
});

test("the status bar renders the readout component, ahead of the unchanged capture date", () => {
  has(pageSrc, /^import \{ MapScaleReadout \} from "@\/components\/MapScaleReadout";$/m, "datamap.tsx does not import the readout component");
  const start = pageSrc.indexOf('className="vt-map-statusbar"');
  assert.ok(start >= 0, "the combined status bar is gone from datamap.tsx");
  const bar = pageSrc.slice(start, start + 900);
  assert.match(bar, /data-testid="imagery-date" role="status"/, "the bar keeps the attributes the visual harness and screen readers key on");
  assert.match(bar, /title="Map scale · zoom level · Esri World Imagery capture date at the view centre \(varies within a view and by zoom\)"/);
  // one element, stable props: the map (null until ready) and nothing that
  // changes per render, or React.memo could not skip the page's re-renders
  assert.match(bar, /<MapScaleReadout map=\{mapReady \? mapRef\.current : null\} \/>/);
  const readoutAt = bar.indexOf("<MapScaleReadout");
  const dateAt = bar.indexOf("vt-statusbar-date");
  assert.ok(dateAt > readoutAt, "the capture date follows the readout inside the same bar");
  assert.match(bar, /\{enabled\.imagery && \(/, "the capture date is still gated on the imagery base");
  assert.match(bar, /\{imageryDate\.label\}/);
  // the spans the readout owns are not also drawn by the page
  lacks(pageSrc, /vt-statusbar-(scale|bar|zoom)\b/, "the scale / zoom spans are rendered twice (page and component)");
});

test("the readout component owns the \"move\" listener, throttled to one animation frame", () => {
  const src = readoutSrc();
  assert.deepEqual(moveHandlers(src), { named: ["onMove"], registrations: 1 });
  assert.deepEqual(Array.from(src.matchAll(/\.on\(\s*["'](\w+)["']/g), (m) => m[1]), ["move"], "the readout listens to \"move\" only");
  // the event handler only asks for a frame; the frame does the read
  has(src, /const onMove = \(\) => \{\s*if \(pending == null\) pending = frames\.request\(sync\);\s*\};/,
    "onMove must do nothing but request one frame while none is waiting");
  has(src, /const sync = \(\) => \{\s*pending = null;\s*const view = readScaleView\(map\);\s*if \(view\) onView\(view\);\s*\};/,
    "the frame callback must clear the latch, read the camera and report it");
  // and the default frame clock is the browser's
  has(src, /frames: FrameScheduler = browserFrames/, "trackScaleView must default to the browser's frame clock");
  has(src, /request: \(callback\) => requestAnimationFrame\(callback\)/, "the browser frame clock must request with requestAnimationFrame");
  has(src, /cancel: \(handle\) => cancelAnimationFrame\(handle\)/, "the browser frame clock must cancel with cancelAnimationFrame");
});

test("the readout removes its listener and cancels its frame on cleanup", () => {
  const src = readoutSrc();
  const from = src.indexOf('map.on("move", onMove);');
  assert.ok(from >= 0, "trackScaleView does not add the \"move\" listener");
  assert.match(src.slice(from, from + 400),
    /^map\.on\("move", onMove\);\s*return \(\) => \{\s*if \(pending != null\) frames\.cancel\(pending\);[\s\S]{0,120}?map\.off\("move", onMove\);/,
    "trackScaleView must return a stop function that cancels the waiting frame and removes the listener it added");
  // the effect hands that stop function to React as its cleanup, keyed on the map
  has(src, /useEffect\(\(\) => \{\s*if \(!map\) return;\s*return trackScaleView\(map, setView\);\s*\}, \[map\]\);/,
    "the component's effect must return trackScaleView's stop function as its cleanup, keyed on the map");
});

test("the readout is a memo component that follows the unit setting itself; the page keeps its own subscription", () => {
  const src = readoutSrc();
  has(src, /export const MapScaleReadout = memo\(function MapScaleReadout\(\{ map \}: MapScaleReadoutProps\) \{/,
    "MapScaleReadout must be wrapped in React.memo, or every page re-render re-renders it");
  has(src, /import \{ getUnits, subscribeUnits \} from "@\/lib\/units";/, "the readout must read the site-wide unit setting from lib/units");
  has(src, /const units = useSyncExternalStore\(subscribeUnits, getUnits, getUnits\);/,
    "a memo component is not re-rendered when the page re-renders on a unit change, so it must subscribe to the unit setting itself");
  has(src, /scaleReading\(view\.zoom, view\.lat, units\)/, "the reading must be computed in the subscribed unit system");
  // moving the readout out must not cost the page its own re-render on a
  // unit change: every other unit-dependent label in its JSX relies on it
  has(pageSrc, /const unitSystem = useSyncExternalStore\(subscribeUnits, getUnits, getUnits\);/,
    "DataMapPage no longer subscribes to the unit setting: its distance / speed / temperature labels would go stale on a switch");
});
