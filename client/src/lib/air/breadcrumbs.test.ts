// Session breadcrumbs — trail/curtain continuity for the followed plane.
// Pins the buffer honesty rules: real fixes only, bounded, never reorders
// or smooths the archived history.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  CRUMB_CAP,
  CRUMB_MIN_DT_SEC,
  CRUMB_MERGE_GUARD_SEC,
  ALT_HOLD_MAX_AGE_SEC,
  pushCrumb,
  mergeTrackWithCrumbs,
  type Crumb,
} from './breadcrumbs.js';

const c = (t: number, lo = -97, la = 39, al: number | null = 9000): Crumb => ({ lo, la, al, t });

test('pushCrumb: appends real fixes, immutably, in poll order', () => {
  const a = pushCrumb([], c(1000));
  const b = pushCrumb(a, c(1015, -96.9));
  assert.equal(a.length, 1);
  assert.equal(b.length, 2, 'append');
  assert.equal(b[1].lo, -96.9);
  assert.equal(a.length, 1, 'input buffer never mutated');
});

test('pushCrumb: rejects junk, repeats and out-of-order fixes', () => {
  const base = pushCrumb([], c(1000));
  assert.equal(pushCrumb(base, c(NaN)).length, 1, 'junk time rejected');
  assert.equal(pushCrumb(base, { lo: NaN, la: 39, al: 0, t: 1015 }).length, 1, 'junk position rejected');
  assert.equal(pushCrumb(base, c(1000)).length, 1, 'same-tick repeat rejected');
  assert.equal(pushCrumb(base, c(1000 + CRUMB_MIN_DT_SEC - 1)).length, 1, 'too-frequent fix rejected');
  assert.equal(pushCrumb(base, c(990)).length, 1, 'out-of-order fix rejected');
  assert.equal(pushCrumb(base, c(1000 + CRUMB_MIN_DT_SEC)).length, 2, 'next poll accepted');
});

test('pushCrumb: cap drops the oldest — a day-long card stays bounded', () => {
  let buf: Crumb[] = [];
  for (let i = 0; i < CRUMB_CAP + 10; i++) buf = pushCrumb(buf, c(1000 + i * 15));
  assert.equal(buf.length, CRUMB_CAP);
  assert.equal(buf[0].t, 1000 + 10 * 15, 'oldest dropped, newest kept');
});

test('pushCrumb: null altitude is a legal fix (honest 3D gap downstream)', () => {
  const buf = pushCrumb([], c(1000, -97, 39, null));
  assert.equal(buf.length, 1);
  assert.equal(buf[0].al, null, 'altitude stays null — never invented');
});

test('merge: crumbs newer than the archive extend it; the seam guard drops the overlap', () => {
  const archived = [
    { lo: -98, la: 38, t: 900, al: 8000 },
    { lo: -97.5, la: 38.5, t: 1200, al: 9000 },
  ];
  const crumbs = [
    c(1190),                                   // older than the archive head — dropped
    c(1200 + CRUMB_MERGE_GUARD_SEC),           // inside the seam guard — dropped
    c(1215, -97.4, 38.6),                      // fresh — kept
    c(1230, -97.3, 38.7, null),                // fresh, no alt — altitude HOLD kicks in
  ];
  const merged = mergeTrackWithCrumbs(archived, crumbs);
  assert.equal(merged.length, 4);
  assert.deepEqual(merged.slice(0, 2), archived, 'archived history untouched, in order');
  assert.equal(merged[2].t, 1215);
  // 2026-07-31: one missing altitude field no longer NaN-gaps the live
  // bridge — the display point carries the last real broadcast altitude
  // (here the 1215 crumb's 9000), explicitly flagged as held
  assert.equal(merged[3].al, 9000);
  assert.equal(merged[3].held, true, 'held display altitude is flagged, never passed off as broadcast');
  assert.equal(merged[2].held, undefined, 'real broadcast points carry no held flag');
});

test('merge: ALTITUDE HOLD seeds from the archived end when the first crumb already lacks altitude', () => {
  const archived = [{ lo: -98, la: 38, t: 1200, al: 8000 }];
  const merged = mergeTrackWithCrumbs(archived, [c(1215, -97.9, 38.1, null)]);
  assert.equal(merged.length, 2);
  assert.equal(merged[1].al, 8000, 'archived altitude carries the bridge');
  assert.equal(merged[1].held, true);
});

test('merge: ALTITUDE HOLD expires at the bound — a sustained outage is still an honest gap', () => {
  const archived = [{ lo: -98, la: 38, t: 1000, al: 8000 }];
  const inBound = 1000 + ALT_HOLD_MAX_AGE_SEC;        // exactly at the bound — held
  const outBound = 1000 + ALT_HOLD_MAX_AGE_SEC + 15;  // past it — honest null
  const merged = mergeTrackWithCrumbs(archived, [
    c(inBound, -97.9, 38.1, null),
    c(outBound, -97.8, 38.2, null),
  ]);
  assert.equal(merged[1].al, 8000);
  assert.equal(merged[1].held, true);
  assert.equal(merged[2].al, null, 'the hold is bounded from the last REAL fix, not the last held point');
  assert.equal(merged[2].held, undefined);
});

test('merge: a real broadcast altitude resets the hold source and value', () => {
  const merged = mergeTrackWithCrumbs([], [
    c(1000, -97, 39, 9000),
    c(1015, -96.9, 39, null),   // held at 9000
    c(1030, -96.8, 39, 10000),  // real fix — new source
    c(1045, -96.7, 39, null),   // held at 10000, aged from t=1030
  ]);
  assert.equal(merged[1].al, 9000);
  assert.equal(merged[1].held, true);
  assert.equal(merged[2].al, 10000);
  assert.equal(merged[2].held, undefined);
  assert.equal(merged[3].al, 10000);
  assert.equal(merged[3].held, true);
});

test('merge: ALTITUDE HOLD with no altitude anywhere stays null — nothing is ever invented', () => {
  const archived = [{ lo: -98, la: 38, t: 1200, al: null }];
  const merged = mergeTrackWithCrumbs(archived, [c(1215, -97.9, 38.1, null)]);
  assert.equal(merged[1].al, null, 'no hold source, no fabricated altitude');
  assert.equal(merged[1].held, undefined);
});

test('merge: an untimed archived altitude never seeds the hold (cannot be age-bounded)', () => {
  const archived = [{ lo: -98, la: 38, al: 8000 }]; // no t — treated as old
  const merged = mergeTrackWithCrumbs(archived, [c(1215, -97.9, 38.1, null)]);
  assert.equal(merged[1].al, null);
});

test('merge: no archive yet — the live crumbs alone form the track', () => {
  const merged = mergeTrackWithCrumbs([], [c(1000), c(1015, -96.9)]);
  assert.equal(merged.length, 2, 'live segment is real data in its own right');
});

test('merge: no fresh crumbs returns the archived array untouched (no realloc churn)', () => {
  const archived = [{ lo: -98, la: 38, t: 2000, al: 8000 }];
  const merged = mergeTrackWithCrumbs(archived, [c(1000)]);
  assert.equal(merged, archived as any, 'same reference — cheap for the 30s refresh');
});

test('merge: archived points without timestamps never block the extension', () => {
  const archived = [{ lo: -98, la: 38 }, { lo: -97.5, la: 38.5 }];
  const merged = mergeTrackWithCrumbs(archived, [c(1215)]);
  assert.equal(merged.length, 3, 'untimed archive treated as old — crumbs still extend');
});
