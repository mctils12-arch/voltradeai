// Hermetic tests for the port-dwell click card formatter (user report: port
// markers had no popup at all, so clicks fell through to the Starlink
// coverage fallback). Run: npx tsx --test client/src/lib/portDetail.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { formatPortDetail } from './portDetail.ts';

const FULL = {
  name: 'Port of Los Angeles',
  in_port_now: 12,
  visits_completed: 87,
  unique_vessels: 61,
  dwell_median_h: 5.9,
  dwell_p90_h: 21.4,
  dwell_max_h: 96.2,
  anomaly_count: 2,
  anomaly_examples: JSON.stringify([
    { mmsi: '367001234', name: 'EVER GIVEN JR', dwell_h: 41.0, median_h: 5.9 },
    { mmsi: '367005678', dwell_h: 22.3, median_h: 5.9 },
    { mmsi: '367009999', dwell_h: 19.9, median_h: 5.9 },
  ]),
  window_hours: 168,
  caveat: 'RAW statistics from our own terrestrial-AIS archive.',
};

test('formatPortDetail: full stats — title, window, dwell spread, capped anomalies, verbatim caveat', () => {
  const card = formatPortDetail(FULL);
  assert.equal(card.title, 'Port of Los Angeles');
  assert.equal(card.subtitle, '12 vessels in port now · 7d window');
  assert.match(card.body, /Completed calls: 87 \(61 unique vessels\)/);
  assert.match(card.body, /median 5\.9h · p90 21\.4h · max 96\.2h/);
  assert.match(card.body, /2 anomaly flags/);
  assert.match(card.body, /EVER GIVEN JR: 41h vs 5\.9h median/);
  assert.match(card.body, /MMSI 367005678/);           // example without a name falls back to MMSI
  assert.ok(!card.body.includes('367009999'), 'anomaly examples cap at 2');
  assert.ok(card.body.includes(FULL.caveat), 'server caveat passes through verbatim');
});

test('formatPortDetail: thin history — no median yet, no anomalies, safe defaults', () => {
  const card = formatPortDetail({ name: 'Port of Oakland', in_port_now: 0, visits_completed: 3 });
  assert.equal(card.subtitle, '0 vessels in port now');
  assert.match(card.body, /not enough completed calls yet for a median/);
  assert.match(card.body, /No dwell anomalies flagged/);
});

test('formatPortDetail: malformed anomaly JSON and missing name never throw', () => {
  const card = formatPortDetail({ anomaly_count: 1, anomaly_examples: '{oops' });
  assert.equal(card.title, 'Port');
  assert.match(card.body, /1 anomaly flag/); // count still shown, examples silently dropped
});
