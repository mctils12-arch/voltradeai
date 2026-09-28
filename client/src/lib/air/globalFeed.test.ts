// FLIGHT PROGRAM B1 — the zoomed-out worldwide aircraft feed: mode decision
// (same disc math as the server, with hysteresis), payload adaptation into
// the viewport row shape with honest staleness, bbox padding/wrapping, and
// abortable/deduped fetching. No DOM, no network.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  discsNeeded, wantsGlobalFeed, globalQueryBBox, globalFeedUrl, adaptGlobalPayload, createAircraftFeed,
  MAX_DISCS_PER_REFRESH, DISC_RADIUS_MAX_NM, GLOBAL_EXIT_DISCS, STALE_ROW_MS, GLOBAL_FEED_POLL_MS,
  STALE_BAND_COLORS, type Bounds,
} from './globalFeed.js';
import {
  planDiscs, MAX_DISCS_PER_REFRESH as SERVER_MAX_DISCS, DISC_RADIUS_MAX_NM as SERVER_RADIUS,
} from '../../../../server/aircraftTiling.js';

const B = (south: number, north: number, west: number, east: number): Bounds => ({ south, north, west, east });

test('constants mirror the server tiling module', () => {
  assert.equal(MAX_DISCS_PER_REFRESH, SERVER_MAX_DISCS);
  assert.equal(DISC_RADIUS_MAX_NM, SERVER_RADIUS);
  assert.ok(GLOBAL_EXIT_DISCS < MAX_DISCS_PER_REFRESH, 'hysteresis band exists');
  assert.equal(GLOBAL_FEED_POLL_MS, 20_000);
  assert.equal(STALE_ROW_MS, 120_000);
});

test('discsNeeded == server planDiscs().needed across viewport sizes', () => {
  const cases: [number, number, number, number][] = [
    [39.5, 40.5, -75.5, -74.5], [30, 45, -100, -80], [24, 50, -125, -66], [-10, 10, 0, 40],
    [35, 70, -10, 40], [-45, -10, 110, 155], [50, 60, -5, 5], [10, 60, -130, -60], [-60, 60, -170, 170],
  ];
  for (const [s, n, w, e] of cases) {
    assert.equal(discsNeeded(B(s, n, w, e)), planDiscs(s, n, w, e).needed, `bbox ${s},${n},${w},${e}`);
  }
});

test('mode: wide views go global, zoomed-in views stay on the precise feed, with hysteresis', () => {
  const conus = B(24, 50, -125, -66);           // needs >8 discs
  const region = B(38, 42, -78, -72);           // a few discs
  assert.equal(wantsGlobalFeed(false, conus, 4.2), true);
  assert.equal(wantsGlobalFeed(false, region, 7), false);
  assert.equal(wantsGlobalFeed(false, region, 2), true, 'globe-scale zoom forces global');
  // a view needing exactly 7 discs: stays global once in (7 > exit 6), never enters (7 <= 8)
  let seven: Bounds | null = null;
  for (let w = 2; w < 90 && !seven; w += 0.5) {
    const b = B(40, 43, -80, -80 + w); // one row of discs, w degrees wide
    if (discsNeeded(b) === 7) seven = b;
  }
  assert.ok(seven, 'found a 7-disc view');
  assert.equal(wantsGlobalFeed(false, seven!, 6), false, 'not entered at 7');
  assert.equal(wantsGlobalFeed(true, seven!, 6), true, 'not left at 7 — no flapping at the boundary');
  assert.equal(wantsGlobalFeed(true, B(40, 41, -80, -79), 8), false, 'leaves once it fits');
  assert.equal(wantsGlobalFeed(false, B(-80, 80, -200, 200), 5), true, 'unwrapped world bounds');
});

test('bbox for the server filter: padded outward to whole degrees, wrapped, null when global', () => {
  assert.deepEqual(globalQueryBBox(B(30, 40, -100, -80)), { lamin: 27, lamax: 43, lomin: -105, lomax: -75 });
  assert.equal(globalQueryBBox(B(-80, 80, -180, 180)), null);
  const am = globalQueryBBox(B(-20, -10, 170, 200))!;   // unwrapped east past 180
  assert.ok(am.lomin > am.lomax, 'antimeridian bbox expressed as lomin > lomax');
  assert.equal(globalFeedUrl(B(-80, 80, -180, 180)), '/api/data/aircraft/global');
  assert.match(globalFeedUrl(B(30, 40, -100, -80)), /lamin=27&lamax=43&lomin=-105&lomax=-75$/);
});

const payload = (at: number) => ({
  at,
  fields: ['hex', 'lon', 'lat', 'altFt', 'gsKt', 'trk', 'callsign', 'type', 'seenAt', 'cat', 'gnd', 'reg', 'src'],
  rows: [
    ['a1', -75, 40, 35000, 450, 90, 'UAL1', 'B738', at - 5_000, 'A3', 0, 'N1', 'adsblol'],
    ['a2', 8, 50, null, 0, null, 'DLH2', 'A320', at - 300_000, 'A3', 1, 'D-AB', 'opensky'],
    ['bad', null, 50, 1, 1, 1, '', null, at, null, 0, null, 'adsblol'],
  ],
  coverage: { opensky: { enabled: true } },
});

test('adaptGlobalPayload: exact viewport row shape, metric units, honest staleness by SERVER clock', () => {
  const at = 1_790_000_000_000;
  const d = adaptGlobalPayload(payload(at));
  assert.equal(d.count, 2, 'row without a position dropped');
  assert.equal(d.global, true);
  assert.equal(d.time, `g${at}`, 'namespaced delta cursor');
  const [a, b] = d.aircraft;
  assert.equal(a.icao24, 'a1');
  assert.equal(Math.round(a.altitude_m!), 10668, '35000 ft -> m');
  assert.equal(Math.round(a.velocity_ms! * 10) / 10, 231.5, '450 kt -> m/s (pipeline factor)');
  assert.equal(a.heading, 90);
  assert.equal(a.stale, false);
  assert.equal(a.seen_pos, 5);
  assert.equal(a.provider, 'adsblol');
  assert.equal(b.on_ground, true);
  assert.equal(b.altitude_m, null);
  assert.equal(b.stale, true, '5-minute-old fix is dimmed');
  assert.match(d.coverage_note, /50% of 2 positions updated <2 min/);
  assert.match(d.coverage_note, /remote oceans may be uncovered/);
  assert.match(d.source, /OpenSky/);
  assert.match(d.coverage_note, /includes The OpenSky Network data/, 'attribution surfaces in the layer note');
  assert.doesNotMatch(adaptGlobalPayload({ ...payload(at), coverage: { opensky: { enabled: false } } }).coverage_note, /OpenSky/);
  // decodes by the response's own field list, not assumed order
  const shuffled = adaptGlobalPayload({ at, fields: ['lat', 'lon', 'hex', 'seenAt'], rows: [[10, 20, 'zz', at]] });
  assert.equal(shuffled.aircraft[0].lat, 10);
  assert.equal(shuffled.aircraft[0].lon, 20);
  assert.equal(adaptGlobalPayload(null).count, 0);
});

test('dimmed band colors keep the altitude meaning (3 bands, darker)', () => {
  assert.deepEqual(Object.keys(STALE_BAND_COLORS), ['ground', 'low', 'cruise']);
  for (const c of Object.values(STALE_BAND_COLORS)) assert.match(c, /^#[0-9a-f]{6}$/);
});

test('createAircraftFeed: dedupes same-bbox refetches, aborts superseded requests, dispose aborts', async () => {
  let t = 1_000_000;
  const signals: AbortSignal[] = [];
  const resolvers: (() => void)[] = [];
  const fetchImpl = ((_url: unknown, init?: RequestInit) => {
    signals.push(init!.signal!);
    return new Promise((resolve) => {
      resolvers.push(() => resolve({ ok: true, status: 200, json: async () => payload(t) } as unknown as Response));
    });
  }) as typeof fetch;
  const feed = createAircraftFeed({ fetchImpl, now: () => t });
  const world = B(-80, 80, -180, 180);
  assert.equal(feed.shouldUseGlobal(world, 2), true);
  assert.equal(feed.pollMs(), GLOBAL_FEED_POLL_MS);
  const p1 = feed.fetchGlobal(world);
  const p2 = feed.fetchGlobal(B(20, 50, -130, -60)); // a newer camera supersedes p1
  assert.equal(signals[0].aborted, true, 'superseded request aborted');
  resolvers[1]();
  const r2 = await p2;
  assert.ok('aircraft' in r2 && r2.count === 2);
  resolvers[0]();
  assert.deepEqual(await p1, { unchanged: true }, 'an aborted request never renders');
  t += 1_000;
  assert.deepEqual(await feed.fetchGlobal(B(20, 50, -130, -60)), { unchanged: true }, 'same bbox inside the spacing -> reuse');
  assert.equal(signals.length, 2, 'no request sent for the deduped call');
  assert.equal(feed.shouldUseGlobal(B(40, 41, -80, -79), 9), false, 'zoomed in -> precise feed');
  assert.equal(feed.pollMs(), null);
  feed.shouldUseGlobal(world, 2);
  const p3 = feed.fetchGlobal(world);
  feed.dispose();
  assert.equal(signals[2].aborted, true, 'dispose aborts the in-flight request');
  resolvers[2]();
  assert.deepEqual(await p3, { unchanged: true });
});
