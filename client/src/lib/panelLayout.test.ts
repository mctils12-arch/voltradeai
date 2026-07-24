// Panel layout core — pins the persistence + clamp rules behind the
// movable/lockable/remembered floating panels (human directive 2026-07-20).
// Run: npx tsx --test client/src/lib/panelLayout.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  parseLayout,
  clampPos,
  mergePrefs,
  clampScale,
  PANEL_KEEP_X,
  PANEL_KEEP_Y,
  PANEL_SCALE_MIN,
  PANEL_SCALE_MAX,
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

test('scale: parse validates, merge round-trips, clamp bounds the range', () => {
  // clampScale — non-finite / nonsense means "no preference" (1)
  assert.equal(clampScale(undefined), 1);
  assert.equal(clampScale(NaN), 1);
  assert.equal(clampScale(-2), 1);
  assert.equal(clampScale(0), 1);
  assert.equal(clampScale(1.2), 1.2, 'in-range value passes through');
  assert.equal(clampScale(0.1), PANEL_SCALE_MIN, 'floor');
  assert.equal(clampScale(9), PANEL_SCALE_MAX, 'ceiling');

  // parseLayout — a stored scale survives; hostile values are dropped or clamped
  const back = parseLayout(JSON.stringify({
    'site-card': { scale: 1.3 },
    a: { scale: 'big' },
    b: { scale: Infinity },
    c: { scale: 99 },
  }));
  assert.deepEqual(back['site-card'], { scale: 1.3 });
  assert.deepEqual(back.a, {}, 'string scale dropped');
  assert.deepEqual(back.b, {}, 'Infinity dropped');
  assert.deepEqual(back.c, { scale: PANEL_SCALE_MAX }, 'absurd value clamped, not dropped');

  // mergePrefs — scale updates independently of pos/lock; undefined deletes
  let l: PanelLayout = { 'site-card': { pos: { left: 5, top: 6 }, locked: true } };
  l = mergePrefs(l, 'site-card', { scale: 1.2 });
  assert.deepEqual(l['site-card'], { pos: { left: 5, top: 6 }, locked: true, scale: 1.2 });
  l = mergePrefs(l, 'site-card', { scale: undefined });
  assert.deepEqual(l['site-card'], { pos: { left: 5, top: 6 }, locked: true },
    'back-to-1 forgets the field (record stays minimal)');
});

test('parseLayout: container dims (cw/ch) survive round-trip; absent stays absent (backward compat)', () => {
  const withDims = parseLayout(JSON.stringify({
    p: { pos: { left: 100, top: 50, cw: 1440, ch: 800 } },
  }));
  assert.deepEqual(withDims.p, { pos: { left: 100, top: 50, cw: 1440, ch: 800 } });
  // legacy record (no cw/ch) parses without them
  const legacy = parseLayout(JSON.stringify({ p: { pos: { left: 100, top: 50 } } }));
  assert.deepEqual(legacy.p, { pos: { left: 100, top: 50 } });
  // garbage cw/ch dropped, position kept
  const bad = parseLayout(JSON.stringify({ p: { pos: { left: 100, top: 50, cw: -5, ch: "x" } } }));
  assert.deepEqual(bad.p, { pos: { left: 100, top: 50 } });
});
