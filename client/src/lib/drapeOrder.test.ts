// Drape-order guard — pins the stack-split prevention rule (probe-proven
// 2026-07-20: a custom layer under a draped layer defeats MapLibre's
// terrain RTT cache; 588ms/frame buried vs 180ms floated, same scene).
// Run: npx tsx --test client/src/lib/drapeOrder.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  buriedCustomIds,
  floatBuriedCustomLayers,
  fullLayerOrder,
  installDrapeOrderGuard,
  DRAPED_TYPES,
  type OrderedLayer,
} from './drapeOrder.js';

const L = (id: string, type: string): OrderedLayer => ({ id, type });

test('buriedCustomIds: flags exactly the customs with a draped layer above', () => {
  // the real-world burial that caused the report: custom between raster and line
  assert.deepEqual(
    buriedCustomIds([L('bg', 'background'), L('imagery', 'raster'), L('aircraft-3d', 'custom'),
      L('aircraft-veclines', 'line'), L('sym', 'symbol')]),
    ['aircraft-3d'],
  );
  // custom above all draped layers (only symbols above it) — nothing to do
  assert.deepEqual(
    buriedCustomIds([L('bg', 'background'), L('imagery', 'raster'), L('veclines', 'line'),
      L('sym', 'symbol'), L('aircraft-3d', 'custom')]),
    [],
  );
  // multiple buried customs report in bottom-to-top order
  assert.deepEqual(
    buriedCustomIds([L('a3d', 'custom'), L('fill', 'fill'), L('track', 'custom'), L('hs', 'hillshade')]),
    ['a3d', 'track'],
  );
  // every draped type triggers; non-draped types (symbol/circle/heatmap/
  // fill-extrusion) never do
  for (const t of DRAPED_TYPES) {
    assert.deepEqual(buriedCustomIds([L('c', 'custom'), L('x', t)]), ['c'], `draped ${t}`);
  }
  for (const t of ['symbol', 'circle', 'heatmap', 'fill-extrusion']) {
    assert.deepEqual(buriedCustomIds([L('c', 'custom'), L('x', t)]), [], `non-draped ${t}`);
  }
  assert.deepEqual(buriedCustomIds([]), []);
});

/** minimal fake MapLibre map: style internals + moveLayer-to-top */
function fakeMap(layers: OrderedLayer[]) {
  const state = {
    order: layers.map((l) => l.id),
    layers: Object.fromEntries(layers.map((l) => [l.id, { ...l }])),
    handlers: {} as Record<string, Array<() => void>>,
    moved: [] as string[],
  };
  const map = {
    style: { get _order() { return state.order; }, _layers: state.layers },
    moveLayer(id: string) {
      state.moved.push(id);
      state.order = state.order.filter((x) => x !== id);
      state.order.push(id);
      // real maplibre fires styledata on the NEXT update; emulate the echo
      queueMicrotask(() => (state.handlers.styledata ?? []).forEach((f) => f()));
    },
    on(ev: string, fn: () => void) { (state.handlers[ev] ??= []).push(fn); },
    off(ev: string, fn: () => void) {
      state.handlers[ev] = (state.handlers[ev] ?? []).filter((f) => f !== fn);
    },
    emit(ev: string) { (state.handlers[ev] ?? []).forEach((f) => f()); },
  };
  return { map, state };
}

test('floatBuriedCustomLayers: floats buried customs to top, preserves relative order, idempotent', () => {
  const { map, state } = fakeMap([
    L('bg', 'background'), L('a3d', 'custom'), L('veclines', 'line'),
    L('track', 'custom'), L('hs', 'hillshade'), L('sym', 'symbol'),
  ]);
  const n = floatBuriedCustomLayers(map as any);
  assert.equal(n, 2, 'both buried customs moved');
  assert.deepEqual(state.order, ['bg', 'veclines', 'hs', 'sym', 'a3d', 'track'],
    'draped layers now contiguous at the bottom; customs on top in original relative order');
  assert.equal(floatBuriedCustomLayers(map as any), 0, 'second pass is a no-op');
});

test('fullLayerOrder: reads style internals, degrades to empty when absent', () => {
  const { map } = fakeMap([L('bg', 'background'), L('c', 'custom')]);
  assert.deepEqual(fullLayerOrder(map as any), [L('bg', 'background'), L('c', 'custom')]);
  assert.deepEqual(fullLayerOrder({} as any), [], 'no style → empty, no throw');
});

test('installDrapeOrderGuard: fixes current state, reacts to styledata, terminates (no move storm)', async () => {
  const { map, state } = fakeMap([
    L('bg', 'background'), L('imagery', 'raster'), L('a3d', 'custom'), L('veclines', 'line'),
  ]);
  const off = installDrapeOrderGuard(map as any);
  assert.deepEqual(state.order, ['bg', 'imagery', 'veclines', 'a3d'], 'buried state fixed on install');
  const movesAfterInstall = state.moved.length;
  // let the moveLayer-echoed styledata microtasks drain — must not re-move
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(state.moved.length, movesAfterInstall, 'echoed styledata caused no further moves');

  // a later stream buries it again (new draped layer appended on top)
  state.layers['flood'] = { id: 'flood', type: 'fill' };
  state.order.push('flood');
  (map as any).emit('styledata');
  assert.deepEqual(state.order, ['bg', 'imagery', 'veclines', 'flood', 'a3d'], 're-floated above the new fill');

  off();
  state.layers['more'] = { id: 'more', type: 'raster' };
  state.order.push('more');
  (map as any).emit('styledata');
  assert.equal(state.order[state.order.length - 1], 'more', 'uninstalled guard no longer reorders');
});
