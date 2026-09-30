// Custom-layer keeper (2026-09-30 fullscreen report): the selected plane's
// curtain + gray plan must survive anything that rebuilds the style's layer
// list (GL context restore, style/basemap/chart-view switch), re-added from
// the FRAME LOOP (Law I), in order, never re-adding a deliberately removed
// layer, and waiting out a loading style.
// Run: npx tsx --test client/src/lib/layerKeeper.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { KEEPER_CHECK_MS, restoreMissingLayers, startLayerKeeper, type KeeperMapLike } from './layerKeeper.js';
import { FrameLoop, type FrameHost } from '../render/frameCore.js';

function fakeMap() {
  const order: string[] = [];
  const impls = new Map<string, unknown>();
  let loaded = true;
  let throwNext = false;
  let repaints = 0;
  const m: KeeperMapLike & {
    order: string[]; drop(): void; setLoaded(v: boolean): void; failNextAdd(): void; repaints(): number;
  } = {
    order,
    getLayer: (id) => (order.includes(id) ? { id } : undefined),
    addLayer: (layer, beforeId) => {
      if (throwNext) { throwNext = false; throw new Error('Style is not done loading'); }
      const id = (layer as { id: string }).id;
      impls.set(id, layer);
      const i = beforeId ? order.indexOf(beforeId) : -1;
      if (i >= 0) order.splice(i, 0, id); else order.push(id);
    },
    isStyleLoaded: () => loaded,
    triggerRepaint: () => { repaints++; },
    drop: () => { order.length = 0; }, // what a context restore / setStyle does to custom layers
    setLoaded: (v) => { loaded = v; },
    failNextAdd: () => { throwNext = true; },
    repaints: () => repaints,
  };
  return m;
}

function manualLoop(): FrameLoop {
  const host: FrameHost = {
    now: () => 0,
    scheduler: { request: () => 1, cancel: () => {} },
    visibility: { isHidden: () => false, subscribe: () => () => {} },
  };
  return new FrameLoop(() => host);
}

const track = { id: 'flight-track-3d' };
const plan = { id: 'flight-plan-curtain' };
const beforeOf = (id: string) => (id === 'flight-plan-curtain' ? 'flight-track-3d' : undefined);

test('restoreMissingLayers: re-adds only what is missing; the plan goes back UNDER the live curtain', () => {
  const map = fakeMap();
  const reg = new Map<string, unknown>([['flight-track-3d', track], ['flight-plan-curtain', plan]]);
  map.addLayer(track); map.addLayer(plan, 'flight-track-3d');
  assert.deepEqual(restoreMissingLayers(map, reg, beforeOf), { restored: [], failed: [] }, 'nothing missing -> nothing touched');
  map.drop();
  const r = restoreMissingLayers(map, reg, beforeOf);
  assert.deepEqual(r.restored, ['flight-track-3d', 'flight-plan-curtain']);
  assert.deepEqual(map.order, ['flight-plan-curtain', 'flight-track-3d'], 'gray plan drawn beneath the colored curtain');
});

test('restoreMissingLayers: a throwing addLayer (style mid-load) is reported, not fatal', () => {
  const map = fakeMap();
  const reg = new Map<string, unknown>([['flight-track-3d', track]]);
  map.failNextAdd();
  assert.deepEqual(restoreMissingLayers(map, reg), { restored: [], failed: ['flight-track-3d'] });
  assert.deepEqual(restoreMissingLayers(map, reg).restored, ['flight-track-3d'], 'next check succeeds');
});

test('keeper: re-adds from the frame loop after a drop (fullscreen/context restore/style switch), waits for a loaded style', () => {
  const map = fakeMap();
  const loop = manualLoop();
  const reg = new Map<string, unknown>([['flight-track-3d', track], ['flight-plan-curtain', plan]]);
  map.addLayer(track); map.addLayer(plan, 'flight-track-3d');
  const restored: string[][] = [];
  const stop = startLayerKeeper({ map, registry: reg, beforeOf, loop, onRestored: (ids) => restored.push(ids) });
  assert.equal(loop.registrationCount, 1, 'one frame-loop registration');
  loop.tick(0);
  assert.equal(restored.length, 0, 'present layers are left alone');
  map.drop();
  map.setLoaded(false);
  loop.tick(KEEPER_CHECK_MS + 1);
  loop.tick(2 * (KEEPER_CHECK_MS + 1));
  assert.deepEqual(map.order, [], 'never adds into a loading style');
  map.setLoaded(true);
  let t = 2 * (KEEPER_CHECK_MS + 1);
  for (let i = 0; i < 20 && restored.length === 0; i++) { t += 50; loop.tick(t); }
  assert.deepEqual(restored, [['flight-track-3d', 'flight-plan-curtain']], 'restored within one check interval, onRestored told which');
  assert.deepEqual(map.order, ['flight-plan-curtain', 'flight-track-3d']);
  assert.ok(map.repaints() >= 1, 'repaint requested after a restore');
  stop();
  assert.equal(loop.registrationCount, 0, 'Law IV: stop unregisters');
});

test('keeper: a layer its owner removed on purpose (registry entry deleted first) is never resurrected', () => {
  const map = fakeMap();
  const loop = manualLoop();
  const reg = new Map<string, unknown>([['flight-plan-curtain', plan]]);
  map.addLayer(plan);
  const stop = startLayerKeeper({ map, registry: reg, loop });
  reg.delete('flight-plan-curtain'); // controller stop(): unregister, then removeLayer
  map.drop();
  for (let t = 0; t < 2000; t += 100) loop.tick(t);
  assert.deepEqual(map.order, [], 'deselected plan stays gone');
  stop();
});

test('keeper: checks are throttled to KEEPER_CHECK_MS of frame time (cheap when nothing is missing)', () => {
  let gets = 0;
  const map = fakeMap();
  const base = map.getLayer.bind(map);
  map.getLayer = (id: string) => { gets++; return base(id); };
  const loop = manualLoop();
  map.addLayer(track);
  const stop = startLayerKeeper({ map, registry: new Map([['flight-track-3d', track]]), loop });
  loop.tick(0); // first frame checks immediately (dt 0, acc primed)
  const afterFirst = gets;
  for (let t = 16; t < KEEPER_CHECK_MS - 20; t += 16) loop.tick(t);
  assert.equal(gets, afterFirst, 'no checks inside the interval');
  stop();
});
