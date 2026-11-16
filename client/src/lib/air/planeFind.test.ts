// Find a plane anywhere (owner report 2026-11-12, N843S): what the page says
// and does for each of the server's four states, the tracked chip's three
// signal states, and the shared signal store.
// Run: npx tsx --test client/src/lib/air/planeFind.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  SIGNAL_MARK, TRACKED_SIGNAL_MAX_AGE_MS, TRACKED_SIGNAL_STALE_MS, _resetTrackedSignals, decidePlaneFind,
  fetchPlaneFind, findPendingNote, fmtLatLon, heardSentence, heardValues, planeSignal, publishTrackedSignals,
  refreshTrackedSignals, subscribeTrackedSignals, trackedChipText, trackedSignalFor,
  type PlaneFindAnswer,
} from './planeFind.js';

const NOW = Date.UTC(2026, 10, 12, 19, 49, 45);
const FIELDS = ['hex', 'lon', 'lat', 'altFt', 'gsKt', 'trk', 'callsign', 'type', 'seenAt', 'cat', 'gnd', 'reg', 'src'];
const heard = { at: NOW - 2_000, alt_ft: 850, gs_kt: 144.3, source: 'mode_s', has_position: false };

test('HEARD: the exact case — heard now at 850 ft / 144 kt, no position yet', () => {
  const ans: PlaneFindAnswer = { state: 'heard', hex: 'ab8c8e', reg: 'N843S', heard, last_seen: null, upstream: 'found' };
  const d = decidePlaneFind('N843S', ans, NOW, 'imperial');
  assert.equal(d.kind, 'heard');
  assert.equal(d.note, "N843S's transponder is being heard now (850 ft, 144 kt) but it has not broadcast a position yet — it will appear as soon as it does");
});

test('HEARD: altitude goes through the units formatter (metric -> metres); knots stay knots', () => {
  const ans: PlaneFindAnswer = { state: 'heard', reg: 'N843S', heard };
  const d = decidePlaneFind('N843S', ans, NOW, 'metric');
  assert.match(d.note, /\(259 m, 144 kt\)/);
  assert.equal(heardValues({ alt_ft: null, gs_kt: null }, 'imperial'), 'altitude not sent, speed not sent', 'a value the transponder did not send is never filled in');
});

test('HEARD: an older reading carries its age; a stale last position is said to be stale', () => {
  const old = heardSentence('N843S', { ...heard, at: NOW - 70_000 }, NOW, null, 'imperial');
  assert.match(old, /transponder was heard 70s ago \(850 ft, 144 kt\) but it has not broadcast a position yet/);
  const stale = heardSentence('N843S', heard, NOW, { at: NOW - 6 * 60_000, lat: 41.7, lon: -87.7, alt_ft: 9000 }, 'imperial');
  assert.match(stale, /is being heard now \(850 ft, 144 kt\) but its last position is 6m old/);
});

test('LIVE: the row decodes into the normal card shape and the note carries the fix age', () => {
  const ans: PlaneFindAnswer = {
    state: 'live', hex: 'ab8c8e', reg: 'N843S', at: NOW, fields: FIELDS,
    rows: [['ab8c8e', -87.8, 41.787, 2825, 202, 148.2, 'N843S', 'FA7X', NOW - 4_000, 'A2', 0, 'N843S', 'adsblol']],
  };
  const d = decidePlaneFind('N843S', ans, NOW, 'imperial');
  assert.equal(d.kind, 'live');
  if (d.kind !== 'live') return;
  assert.equal(d.row.icao24, 'ab8c8e');
  assert.equal(d.row.lat, 41.787);
  assert.equal(d.row.lon, -87.8);
  assert.equal(d.row.registration, 'N843S');
  assert.equal(d.row.seen_pos, 4);
  assert.equal(d.note, 'N843S is outside this view — flying to it (position 4s old)');
});

test('LIVE with no decodable row is never treated as a position', () => {
  const d = decidePlaneFind('N843S', { state: 'live', hex: 'ab8c8e', rows: [] }, NOW, 'imperial');
  assert.equal(d.kind, 'unknown');
});

test('LAST_SEEN: when, where, how high, from which source — and the archive card is offered', () => {
  const ans: PlaneFindAnswer = {
    state: 'last_seen', hex: 'AB8C8E', reg: 'N843S', upstream: 'empty',
    last_seen: { at: NOW - 3 * 3600_000, lat: 41.9221, lon: -72.7071, alt_ft: 174, via: 'tracked' },
  };
  const d = decidePlaneFind('N843S', ans, NOW, 'imperial');
  assert.equal(d.kind, 'last_seen');
  if (d.kind !== 'last_seen') return;
  assert.equal(d.hex, 'ab8c8e', 'the hex the archive card opens with');
  assert.equal(d.reg, 'N843S');
  assert.equal(d.note, 'no signal from N843S right now — last seen 3h ago near 41.92°N 72.71°W at 174 ft (the 24/7 tracked poll); its archived trips are on the card');
});

test('LAST_SEEN when the provider could not be asked: says so instead of claiming "no signal"', () => {
  const ans: PlaneFindAnswer = {
    state: 'last_seen', hex: 'ab8c8e', reg: 'N843S', upstream: 'limited',
    last_seen: { at: NOW - 600_000, lat: -33.95, lon: 151.18, alt_ft: null, via: 'archive' },
  };
  const d = decidePlaneFind('N843S', ans, NOW, 'imperial');
  assert.match(d.note, /^N843S is not in our worldwide feed right now \(the provider lookup is rate-limited just now — try again in a few seconds\) — last seen 10m ago near 33\.95°S 151\.18°E \(our archive\)/);
  assert.doesNotMatch(d.note, /no signal/);
});

test('UNKNOWN: says where we looked; invalid and unreachable are worded as what they are', () => {
  const u = decidePlaneFind('N000ZZ', { state: 'unknown', upstream: 'empty' }, NOW, 'imperial');
  assert.equal(u.kind, 'unknown');
  assert.match(u.note, /^no "N000ZZ" in the current view, our worldwide feed, the 24\/7 tracked list or the provider's lookup right now — it may be on the ground with its transponder off, or outside receiver coverage/);
  assert.match(u.note, /watched planes stay reachable via their archive$/);
  const err = decidePlaneFind('N000ZZ', { state: 'unknown', upstream: 'error' }, NOW, 'imperial');
  assert.match(err.note, /the provider lookup failed just now/);
  const invalid = decidePlaneFind('B738 X', 'invalid', NOW, 'imperial');
  assert.match(invalid.note, /takes a tail number, a callsign or a 6-digit hex; type searches \(e\.g\. B738\) cover the current view only/);
  const down = decidePlaneFind('N843S', null, NOW, 'imperial');
  assert.match(down.note, /the worldwide lookup could not be reached just now — try again/);
  assert.equal(findPendingNote('N843S'), 'no "N843S" in the current view — checking the worldwide feed…');
});

test('a heard / last_seen answer missing its reading falls back to unknown (never a guessed state)', () => {
  assert.equal(decidePlaneFind('N843S', { state: 'heard', heard: null }, NOW, 'imperial').kind, 'unknown');
  assert.equal(decidePlaneFind('N843S', { state: 'last_seen', last_seen: null }, NOW, 'imperial').kind, 'unknown');
});

test('fmtLatLon: hemispheres, two decimals', () => {
  assert.equal(fmtLatLon(41.787, -87.8), '41.79°N 87.80°W');
  assert.equal(fmtLatLon(-33.9461, 151.1772), '33.95°S 151.18°E');
});

test('fetchPlaneFind: the query is URL-encoded; 400 -> invalid; failure -> null; abortable', async () => {
  const seen: string[] = [];
  const ok = (async (url: string, init?: RequestInit) => {
    seen.push(url);
    assert.equal(init?.cache, 'no-store');
    return { ok: true, status: 200, json: async () => ({ state: 'heard', heard }) };
  }) as unknown as typeof fetch;
  const a = await fetchPlaneFind('G-EZTL', { fetchImpl: ok });
  assert.deepEqual(seen, ['/api/data/aircraft/find?q=G-EZTL']);
  assert.equal(a !== null && a !== 'invalid' && a.state, 'heard');
  await fetchPlaneFind('N1 &x=1', { fetchImpl: ok });
  assert.equal(seen[1], '/api/data/aircraft/find?q=N1%20%26x%3D1', 'never spliced raw into the URL');
  const bad = (async () => ({ ok: false, status: 400, json: async () => ({}) })) as unknown as typeof fetch;
  assert.equal(await fetchPlaneFind('B738 X', { fetchImpl: bad }), 'invalid');
  const down = (async () => { throw new Error('network'); }) as unknown as typeof fetch;
  assert.equal(await fetchPlaneFind('N843S', { fetchImpl: down }), null);
  const five = (async () => ({ ok: false, status: 502, json: async () => ({}) })) as unknown as typeof fetch;
  assert.equal(await fetchPlaneFind('N843S', { fetchImpl: five }), null);
  const junk = (async () => ({ ok: true, status: 200, json: async () => ({ nope: 1 }) })) as unknown as typeof fetch;
  assert.equal(await fetchPlaneFind('N843S', { fetchImpl: junk }), null);
});

test('planeSignal: the three honest states of a tracked plane', () => {
  assert.equal(planeSignal({ reg: 'N843S', last_seen: NOW - 30_000 }, NOW), 'live');
  assert.equal(planeSignal({ reg: 'N843S', last_seen: NOW - 3 * 3600_000, heard }, NOW), 'heard', 'heard now, last position hours old');
  assert.equal(planeSignal({ reg: 'N843S', last_seen: NOW - 3 * 3600_000, heard: { ...heard, at: NOW - 10 * 60_000 } }, NOW), 'dark', 'a heard reading older than 2 min is not "now"');
  assert.equal(planeSignal({ reg: 'N843S', last_seen: NOW - 3 * 3600_000, heard: { ...heard, has_position: true, at: NOW - 3 * 3600_000 } }, NOW), 'dark');
  assert.equal(planeSignal({ reg: 'N843S' }, NOW), 'dark');
  assert.deepEqual(SIGNAL_MARK, { live: '●', heard: '◐', dark: '○' });
});

test('trackedChipText: mark + tail + age for each state', () => {
  assert.equal(trackedChipText({ reg: 'N843S', last_seen: NOW - 12_000 }, NOW), '● N843S · 12s ago');
  assert.equal(trackedChipText({ reg: 'N843S', last_seen: NOW - 3 * 3600_000, heard }, NOW), '◐ N843S · heard 2s ago, no position yet');
  assert.equal(trackedChipText({ reg: 'N843S', heard }, NOW), '◐ N843S · heard 2s ago, no position yet', 'heard before any position was ever seen');
  assert.equal(trackedChipText({ reg: 'N843S', last_seen: NOW - 3 * 3600_000 }, NOW), '○ N843S · 3h ago');
  assert.equal(trackedChipText({ reg: 'N843S' }, NOW), '○ N843S · awaiting first signal');
});

test('signal store: only hexes on the tracked list get a mark; subscribers hear every publish; marks expire with their data', () => {
  _resetTrackedSignals();
  let ticks = 0;
  const off = subscribeTrackedSignals(() => { ticks++; });
  publishTrackedSignals([
    { reg: 'N843S', hex: 'AB8C8E', last_seen: NOW - 3 * 3600_000, heard },
    { reg: 'N369CA', hex: 'a42dee', last_seen: NOW - 5_000 },
    { reg: 'N1', hex: null },
  ], NOW);
  assert.equal(ticks, 1);
  assert.equal(trackedSignalFor('ab8c8e', NOW), 'heard');
  assert.equal(trackedSignalFor('A42DEE', NOW), 'live');
  assert.equal(trackedSignalFor('ffffff', NOW), null, 'not tracked -> unknown, no mark');
  assert.equal(trackedSignalFor(null, NOW), null);
  assert.equal(trackedSignalFor('a42dee', NOW + TRACKED_SIGNAL_MAX_AGE_MS + 1), null, 'three missed polls: the list is too old to speak — no mark, never a stale ●');
  publishTrackedSignals([], NOW);
  assert.equal(trackedSignalFor('ab8c8e', NOW), null);
  off();
  publishTrackedSignals([], NOW);
  assert.equal(ticks, 2);
});

test('refreshTrackedSignals: fetches only when nothing was published recently; one request in flight; a failure keeps the old rows', async () => {
  _resetTrackedSignals();
  let t = NOW;
  const urls: string[] = [];
  let fail = false;
  const fetchImpl = (async (url: string) => {
    urls.push(url);
    if (fail) throw new Error('network');
    return { ok: true, status: 200, json: async () => ({ planes: [{ reg: 'N843S', hex: 'ab8c8e', last_seen: t - 3 * 3600_000, heard: { ...heard, at: t - 2_000 } }] }) };
  }) as unknown as typeof fetch;
  const opts = { fetchImpl, nowMs: () => t };
  await Promise.all([refreshTrackedSignals(opts), refreshTrackedSignals(opts)]);
  assert.deepEqual(urls, ['/api/data/aircraft/tracked'], 'two callers, ONE request');
  assert.equal(trackedSignalFor('ab8c8e', t), 'heard');
  t += TRACKED_SIGNAL_STALE_MS - 1;
  await refreshTrackedSignals(opts);
  assert.equal(urls.length, 1, 'published recently (by this or by the tracked panel) -> no request');
  // the tracked panel's own poll counts as a publish: the legend never doubles it
  publishTrackedSignals([{ reg: 'N843S', hex: 'ab8c8e', last_seen: t - 1_000 }], t);
  t += TRACKED_SIGNAL_STALE_MS - 1;
  await refreshTrackedSignals(opts);
  assert.equal(urls.length, 1);
  t += 2;
  fail = true;
  await refreshTrackedSignals(opts);
  assert.equal(urls.length, 2, 'stale -> asked');
  assert.equal(trackedSignalFor('ab8c8e', t), 'live', 'a failed refresh leaves the last rows (they expire by age)');
  assert.equal(trackedSignalFor('ab8c8e', t + TRACKED_SIGNAL_MAX_AGE_MS), null);
});
