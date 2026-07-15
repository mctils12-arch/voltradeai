// Auto-3D minis — EARTH TWIN O6-2. Pins the selection honesty (catalogued
// only, sentinel-skipped never shown, followed object excluded), the
// nearest-to-centre ranking with antimeridian wrap, and the band constants.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { selectMiniSats, formsFromSatcat, MINI_CAP, MINI_MAX_CAM_KM } from './miniSelect.js';
import { SAT_STRIDE } from './satBuffer.js';

function buf(rows: Array<[number, number, number, number]>): Float32Array {
  const b = new Float32Array(rows.length * SAT_STRIDE);
  rows.forEach((r, i) => b.set(r, i * SAT_STRIDE));
  return b;
}
const VIEW = { minX: 0.4, maxX: 0.6, minY: 0.4, maxY: 0.6, cx: 0.5, cy: 0.5 };

test('selectMiniSats: catalogued-only, in-view, sentinel-skipped never shown, nearest first', () => {
  const positions = buf([
    [0.50, 0.50, 500_000, 0],  // catalogued, dead centre
    [0.55, 0.55, 500_000, 0],  // catalogued, further out
    [0.52, 0.52, 500_000, 0],  // UNCATALOGUED (forms null) → dot, never a form
    [0.51, 0.51, 500_000, -1], // sentinel slot (invalid/filtered) → never shown
    [0.90, 0.90, 500_000, 0],  // outside the viewport
  ]);
  const forms = ['bus', 'cubesat', null, 'bus', 'bus'] as any;
  const minis = selectMiniSats(positions, forms, VIEW);
  assert.deepEqual(minis.map((m) => m.index), [0, 1], 'centre-first, honest exclusions');
  assert.equal(minis[0].form, 'bus');
  assert.equal(minis[0].altMeters, 500_000);
});

test('selectMiniSats: cap bounds the draw count; excludeIndex drops the followed object', () => {
  const rows: Array<[number, number, number, number]> = [];
  for (let i = 0; i < 30; i++) rows.push([0.5 + i * 0.002, 0.5, 400_000, 0]);
  const positions = buf(rows);
  const forms = new Array(30).fill('smallsat');
  assert.equal(selectMiniSats(positions, forms, VIEW).length, MINI_CAP);
  const withoutFirst = selectMiniSats(positions, forms, VIEW, MINI_CAP, 0);
  assert.ok(!withoutFirst.some((m) => m.index === 0), 'followed object keeps its big model instead');
});

test('selectMiniSats: antimeridian wrap — x near 1 is IN a view spanning past 1', () => {
  const positions = buf([[0.98, 0.5, 400_000, 0]]);
  const view = { minX: 0.95, maxX: 1.05, minY: 0.4, maxY: 0.6, cx: 1.0, cy: 0.5 };
  assert.equal(selectMiniSats(positions, ['bus'] as any, view).length, 1);
  // and x just past 0 wraps into the same view
  const positions2 = buf([[0.02, 0.5, 400_000, 0]]);
  assert.equal(selectMiniSats(positions2, ['bus'] as any, view).length, 1, 'x+1 wraps into range');
});

test('formsFromSatcat: index-aligned, null where the catalog has no row', () => {
  const gp = [{ noradId: 1 }, { noradId: 2 }, { noradId: 3 }] as any;
  const byNorad = new Map<number, any>([
    [1, { objectType: 'PAYLOAD', rcsSize: 'LARGE' }],
    [3, { objectType: 'ROCKET BODY', rcsSize: null }],
  ]);
  const forms = formsFromSatcat(gp, byNorad);
  assert.deepEqual(forms, ['bus', null, 'rocket-body']);
});

test('band constants: minis live INSIDE the visible envelope', () => {
  assert.ok(MINI_MAX_CAM_KM > 100, 'band ceiling above the registry LOD floor (100km)');
  assert.ok(MINI_MAX_CAM_KM <= 5000, 'band is a close-zoom feature, not the whole-globe view');
  assert.ok(MINI_CAP >= 4 && MINI_CAP <= 32, 'draw-call budget stays sane');
});
