// Panel layout core — pins the persistence + clamp rules behind the
// movable/lockable/remembered floating panels (human directive 2026-07-20).
// Run: npx tsx --test client/src/lib/panelLayout.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  parseLayout,
  clampPos,
  mergePrefs,
  PANEL_KEEP_X,
  PANEL_KEEP_Y,
  type PanelLayout,
} from './panelLayout.js';

test('parseLayout: round-trips a valid layout', () => {
  const layout: PanelLayout = {
    'site-card': { pos: { left: 40, top: 300 }, locked: true },
    'nav-cluster': { min: true },
  };
  const back = parseLayout(JSON.stringify(layout));
  assert.deepEqual(back, layout);
});

test('parseLayout: corrupt or hostile records degrade to no preference, never throw', () => {
  assert.deepEqual(parseLayout(null), {});
  assert.deepEqual(parseLayout(''), {});
  assert.deepEqual(parseLayout('not json {{{'), {});
  assert.deepEqual(parseLayout('[1,2,3]'), {});
  assert.deepEqual(parseLayout('"str"'), {});
  // wrong shapes inside are dropped field-by-field
  const messy = parseLayout(JSON.stringify({
    a: { pos: { left: 'x', top: 2 }, min: 'yes', locked: true },
    b: { pos: { left: Infinity, top: 5 } },
    c: 7,
    d: { pos: { left: 10, top: 20 }, min: false },
  }));
  assert.deepEqual(messy.a, { locked: true }, 'non-finite/typed-wrong fields dropped');
  assert.deepEqual(messy.b, {}, 'Infinity is not a position');
  assert.equal(messy.c, undefined, 'non-object entry dropped');
  assert.deepEqual(messy.d, { pos: { left: 10, top: 20 }, min: false });
});

test('clampPos: a position saved on a big window stays reachable on a small one', () => {
  // saved far right/bottom on a 2560×1300 container, restored on 900×500
  const p = clampPos({ left: 2400, top: 1200 }, 900, 500);
  assert.equal(p.left, 900 - PANEL_KEEP_X, 'grip pulled back inside horizontally');
  assert.equal(p.top, 500 - PANEL_KEEP_Y, 'grip pulled back inside vertically');
  // never negative, even on absurdly small containers
  assert.deepEqual(clampPos({ left: -50, top: -50 }, 40, 30), { left: 0, top: 0 });
  // an in-bounds position is untouched
  assert.deepEqual(clampPos({ left: 100, top: 120 }, 900, 500), { left: 100, top: 120 });
});

test('mergePrefs: pos/min/locked update independently; undefined deletes (double-click reset)', () => {
  let l: PanelLayout = {};
  l = mergePrefs(l, 'site-card', { pos: { left: 5, top: 6 } });
  l = mergePrefs(l, 'site-card', { locked: true });
  assert.deepEqual(l['site-card'], { pos: { left: 5, top: 6 }, locked: true },
    'lock did not clobber the position');
  l = mergePrefs(l, 'site-card', { min: true });
  l = mergePrefs(l, 'site-card', { pos: undefined });
  assert.deepEqual(l['site-card'], { locked: true, min: true },
    'reset forgot the position and kept lock/min');
  // other panels untouched
  l = mergePrefs(l, 'nav-cluster', { min: false });
  assert.deepEqual(l['site-card'], { locked: true, min: true });
});
