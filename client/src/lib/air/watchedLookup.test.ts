// Watched-plane lookup (field bug 2026-09-30, N843S): a plane outside the
// viewport feed must open the LIVE card when the server holds a fresh fix,
// and the archive card must say "last seen …", never a false "not
// currently broadcasting".
// Run: npx tsx --test client/src/lib/air/watchedLookup.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  WATCH_LIVE_MAX_AGE_MS,
  aircraftFreshnessClause,
  decideWatchedOpen,
  decodeLiveRow,
  fetchWatchedSources,
  fmtAgeShort,
  type WatchedRow,
} from './watchedLookup.js';

const NOW = 1_790_799_400_000;
const row = (o: Partial<WatchedRow> = {}): WatchedRow => ({
  icao24: 'ab8c8e', callsign: 'N843S', registration: 'N843S', lon: -83.9, lat: 36.1,
  altitude_m: 13106, on_ground: false, velocity_ms: 240, heading: 190, type: 'FA7X', category: 'A2',
  ...o,
});

test('in the viewport feed: the local row wins (unchanged fast path)', () => {
  const d = decideWatchedOpen({ hex: 'ab8c8e', local: row(), snapshot: null, tracked: null, nowMs: NOW });
  assert.equal(d.kind, 'live');
  assert.equal(d.kind === 'live' && d.source, 'viewport');
});

test('off-viewport, snapshot fix 20 s old -> LIVE card from the snapshot row (the N843S bug)', () => {
  const d = decideWatchedOpen({ hex: 'ab8c8e', local: null, snapshot: row({ seen_pos: 20 }), tracked: null, nowMs: NOW });
  assert.equal(d.kind, 'live');
  if (d.kind !== 'live') return;
  assert.equal(d.source, 'snapshot');
  assert.equal(d.row.callsign, 'N843S');
  assert.equal(d.ageMs, 20_000);
});

test('stale local/global row does not count as live; a fresher snapshot does', () => {
  const d = decideWatchedOpen({
    hex: 'ab8c8e', local: row({ stale: true, seen_pos: 400 }), snapshot: row({ seen_pos: 5 }), tracked: null, nowMs: NOW,
  });
  assert.equal(d.kind === 'live' && d.source, 'snapshot');
});

test('snapshot missing, tracked registry saw it 40 s ago -> live at last_pos, speed/heading unknown (never invented)', () => {
  const d = decideWatchedOpen({
    hex: 'ab8c8e', local: null, snapshot: null,
    tracked: { reg: 'N843S', hex: 'ab8c8e', last_seen: NOW - 40_000, last_pos: { la: 36.098, lo: -83.897, al: 13106 } },
    nowMs: NOW,
  });
  assert.equal(d.kind, 'live');
  if (d.kind !== 'live') return;
  assert.equal(d.source, 'tracked');
  assert.equal(d.row.lat, 36.098);
  assert.equal(d.row.altitude_m, 13106);
  assert.equal(d.row.velocity_ms, null);
  assert.equal(d.row.heading, null);
});

test('nothing fresh -> archive card carrying the newest last-seen time from any source', () => {
  const d = decideWatchedOpen({
    hex: 'ab8c8e', local: null, snapshot: row({ seen_pos: 400 }),
    tracked: { reg: 'N843S', hex: 'ab8c8e', last_seen: NOW - 900_000, last_pos: { la: 1, lo: 2, al: null } },
    nowMs: NOW,
  });
  assert.deepEqual(d, { kind: 'archive', lastSeenMs: NOW - 400_000 });
  const never = decideWatchedOpen({ hex: 'ab8c8e', local: null, snapshot: null, tracked: null, nowMs: NOW });
  assert.deepEqual(never, { kind: 'archive', lastSeenMs: null });
});

test('the 2-minute boundary is inclusive and matches the global feed stale threshold', () => {
  assert.equal(WATCH_LIVE_MAX_AGE_MS, 120_000);
  const at = decideWatchedOpen({ hex: 'x', local: null, snapshot: row({ seen_pos: 120 }), tracked: null, nowMs: NOW });
  assert.equal(at.kind, 'live');
  const past = decideWatchedOpen({ hex: 'x', local: null, snapshot: row({ seen_pos: 121 }), tracked: null, nowMs: NOW });
  assert.equal(past.kind, 'archive');
});

test('subtitle freshness clause derives from the newest fix, not a click-time constant', () => {
  assert.equal(aircraftFreshnessClause(NOW / 1000 - 30, NOW), 'broadcasting now');
  assert.equal(aircraftFreshnessClause(NOW / 1000 - 420, NOW), 'last seen 7m ago');
  assert.equal(aircraftFreshnessClause(null, NOW), 'no recent signal');
  assert.equal(fmtAgeShort(3 * 86_400_000), '3d');
});

test('decodeLiveRow: /live/:hex wire shape via the global adapter; empty -> null', () => {
  const fields = ['hex', 'lon', 'lat', 'altFt', 'gsKt', 'trk', 'callsign', 'type', 'seenAt', 'cat', 'gnd', 'reg', 'src'];
  const r = decodeLiveRow({ at: NOW, fields, rows: [['ab8c8e', -83.9, 36.1, 43000, 466, 190, 'N843S', 'FA7X', NOW - 10_000, 'A2', 0, 'N843S', 'adsblol']] });
  assert.ok(r);
  assert.equal(r?.callsign, 'N843S');
  assert.equal(r?.seen_pos, 10);
  assert.equal(r?.stale, false);
  assert.equal(decodeLiveRow({ at: NOW, fields, rows: [] }), null);
  assert.equal(decodeLiveRow(null), null);
});

test('fetchWatchedSources: both reads in parallel; a failed read degrades to null + failed flag', async () => {
  const urls: string[] = [];
  const ok = (async (u: string) => {
    urls.push(u);
    if (u.startsWith('/api/data/aircraft/live/')) {
      return { ok: true, json: async () => ({ at: NOW, fields: ['hex', 'lon', 'lat', 'seenAt'], rows: [['ab8c8e', -83, 36, NOW]] }) };
    }
    return { ok: true, json: async () => ({ planes: [{ reg: 'N843S', hex: 'AB8C8E', last_seen: NOW }] }) };
  }) as unknown as typeof fetch;
  const a = await fetchWatchedSources('AB8C8E', { fetchImpl: ok });
  assert.deepEqual(urls.sort(), ['/api/data/aircraft/live/ab8c8e', '/api/data/aircraft/tracked']);
  assert.equal(a.snapshot?.icao24, 'ab8c8e');
  assert.equal(a.tracked?.reg, 'N843S');
  assert.equal(a.failed, false);
  const bad = (async () => { throw new Error('offline'); }) as unknown as typeof fetch;
  const b = await fetchWatchedSources('ab8c8e', { fetchImpl: bad });
  assert.deepEqual(b, { snapshot: null, tracked: null, failed: true });
});
