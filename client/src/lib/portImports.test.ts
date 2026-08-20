// Hermetic tests for the /api/data/imports decode helpers.
// Run: npx tsx --test client/src/lib/portImports.test.ts
//
// Fixtures are the live 2026-08-20 shapes (Census FT920): the national
// aggregate under port code "-", a land-border port publishing a real 0 for
// containerized cargo, and a port with no published name (code 1791).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  splitNational, monthsOf, joinMonths, reconcile, sortRows, filterRows,
  NATIONAL_PORT_CODE, type ImportObs,
} from './portImports.ts';

const obs = (o: Partial<ImportObs> & { port: string; month: string }): ImportObs => ({
  port_name: null, gen_val: null, cnt_val: null, cnt_wgt: null, rt: '2026-08-20', ...o,
});

const LIVE: ImportObs[] = [
  obs({ port: '-', port_name: 'TOTAL FOR ALL PORTS', month: '2026-06', gen_val: 317408003903, cnt_val: 83901230609, cnt_wgt: 18173621123 }),
  obs({ port: '-', port_name: 'TOTAL FOR ALL PORTS', month: '2026-05', gen_val: 311856112671, cnt_val: 81842364461, cnt_wgt: 18601934510 }),
  obs({ port: '2704', port_name: 'LOS ANGELES, CA', month: '2026-06', gen_val: 22917449056, cnt_val: 21049766472, cnt_wgt: 3940035515 }),
  obs({ port: '2704', port_name: 'LOS ANGELES, CA', month: '2026-05', gen_val: 20595308379, cnt_val: 19026611264, cnt_wgt: 3665704530 }),
  obs({ port: '2304', port_name: 'LAREDO, TX', month: '2026-06', gen_val: 24201376152, cnt_val: 0, cnt_wgt: 0 }),
  obs({ port: '2304', port_name: 'LAREDO, TX', month: '2026-05', gen_val: 24288603574, cnt_val: 0, cnt_wgt: 0 }),
  // present in the newest month only — a real case, 11 of 358 ports live
  obs({ port: '3013', port_name: 'GREAT FALLS, MT', month: '2026-06', gen_val: 270289178658, cnt_val: 0, cnt_wgt: 0 }),
  // Census publishes no name for this one — must never be given an invented one
  obs({ port: '1791', port_name: null, month: '2026-06', gen_val: 77310711, cnt_val: 0, cnt_wgt: 0 }),
];

test('the national aggregate row is separated out, not left in the port list', () => {
  const { ports, national } = splitNational(LIVE);
  assert.equal(national.length, 2);
  assert.ok(national.every((r) => r.port === NATIONAL_PORT_CODE));
  assert.equal(ports.length, 6);
  assert.ok(!ports.some((r) => r.port === NATIONAL_PORT_CODE));
  // the trap it exists to close: the aggregate outranks every real port
  assert.ok(national[0].gen_val! > Math.max(...ports.map((p) => p.gen_val ?? 0)));
});

test('splitNational tolerates a malformed/empty payload without throwing', () => {
  assert.deepEqual(splitNational(null), { ports: [], national: [] });
  assert.deepEqual(splitNational([{ nope: 1 } as unknown as ImportObs]), { ports: [], national: [] });
});

test('months come back newest first', () => {
  assert.deepEqual(monthsOf(LIVE), ['2026-06', '2026-05']);
  assert.deepEqual(monthsOf([]), []);
});

test('a port with no prior-month row gets a null delta, never 0', () => {
  const { ports } = splitNational(LIVE);
  const rows = joinMonths(ports, '2026-06', '2026-05');
  const gf = rows.find((r) => r.port === '3013')!;
  assert.equal(gf.prev_gen_val, null);
  assert.equal(gf.delta, null);
  const la = rows.find((r) => r.port === '2704')!;
  assert.ok(Math.abs(la.delta! - 0.11276) < 1e-4);
  const laredo = rows.find((r) => r.port === '2304')!;
  assert.ok(laredo.delta! < 0); // real month-over-month decline, sign preserved
});

test('a zero prior value yields null rather than an infinite move', () => {
  const rows = joinMonths(
    [obs({ port: 'X', month: '2026-06', gen_val: 500 }), obs({ port: 'X', month: '2026-05', gen_val: 0 })],
    '2026-06', '2026-05',
  );
  assert.equal(rows[0].prev_gen_val, 0);
  assert.equal(rows[0].delta, null);
});

test('no prior month at all (first month of the window) still produces rows', () => {
  const { ports } = splitNational(LIVE);
  const rows = joinMonths(ports, '2026-05', null);
  assert.equal(rows.length, 2);
  assert.ok(rows.every((r) => r.delta === null && r.prev_gen_val === null));
});

test('a published 0 stays 0 and a missing column stays null — they never merge', () => {
  const { ports } = splitNational(LIVE);
  const rows = joinMonths(ports, '2026-06', '2026-05');
  assert.equal(rows.find((r) => r.port === '2304')!.cnt_val, 0);      // Laredo publishes 0
  const noCols = joinMonths([obs({ port: 'Y', month: '2026-06', gen_val: 1 })], '2026-06', null);
  assert.equal(noCols[0].cnt_val, null);                              // variant lacked the column
});

test('an unnamed port keeps its null name — no name is inferred from the code', () => {
  const { ports } = splitNational(LIVE);
  const row = joinMonths(ports, '2026-06', '2026-05').find((r) => r.port === '1791')!;
  assert.equal(row.port_name, null);
});

test('reconcile reports an exact match when the per-port rows partition the total', () => {
  const nat = obs({ port: '-', month: '2026-07', gen_val: 300 });
  const ports = [obs({ port: 'A', month: '2026-07', gen_val: 100 }), obs({ port: 'B', month: '2026-07', gen_val: 200 })];
  const r = reconcile(ports, [nat], '2026-07');
  assert.equal(r.sum, 300);
  assert.equal(r.published, 300);
  assert.equal(r.diff, 0);
  assert.equal(r.exact, true);
});

test('reconcile reports a real discrepancy instead of hiding it', () => {
  const nat = obs({ port: '-', month: '2026-07', gen_val: 300 });
  const ports = [obs({ port: 'A', month: '2026-07', gen_val: 100 })];
  const r = reconcile(ports, [nat], '2026-07');
  assert.equal(r.diff, -200);
  assert.equal(r.exact, false);
});

test('reconcile with no published total reports null, not a false pass', () => {
  const r = reconcile([obs({ port: 'A', month: '2026-07', gen_val: 100 })], [], '2026-07');
  assert.equal(r.published, null);
  assert.equal(r.diff, null);
  assert.equal(r.exact, false);
});

test('reconcile only counts the requested month', () => {
  const rows = [obs({ port: 'A', month: '2026-07', gen_val: 100 })];
  const nat = [obs({ port: '-', month: '2026-06', gen_val: 999 }), obs({ port: '-', month: '2026-07', gen_val: 100 })];
  assert.equal(reconcile(rows, nat, '2026-07').exact, true);
});

test('numeric sorts are descending with nulls last in every column', () => {
  const rows = joinMonths(
    [
      obs({ port: 'A', month: 'm', gen_val: 5, cnt_wgt: null, cnt_val: 1 }),
      obs({ port: 'B', month: 'm', gen_val: null, cnt_wgt: 9, cnt_val: null }),
      obs({ port: 'C', month: 'm', gen_val: 50, cnt_wgt: 1, cnt_val: 7 }),
    ], 'm', null,
  );
  assert.deepEqual(sortRows(rows, 'gen_val').map((r) => r.port), ['C', 'A', 'B']);
  assert.deepEqual(sortRows(rows, 'cnt_wgt').map((r) => r.port), ['B', 'C', 'A']);
  assert.deepEqual(sortRows(rows, 'cnt_val').map((r) => r.port), ['C', 'A', 'B']);
  // every row's delta is null here — the sort must be stable, not throw
  assert.equal(sortRows(rows, 'delta').length, 3);
});

test('a null never outranks a real DECLINE — the case a null-as-zero sort gets wrong', () => {
  // The only signed column: a port down 50% is a real reading and must rank
  // above a port whose change is unknown. Coercing null to 0 would put the
  // unknown ahead of the decline, inventing a ranking from missing data.
  const rows = joinMonths(
    [
      obs({ port: 'DOWN', month: '2026-06', gen_val: 50 }),
      obs({ port: 'DOWN', month: '2026-05', gen_val: 100 }),
      obs({ port: 'UNKNOWN', month: '2026-06', gen_val: 80 }),
      obs({ port: 'UP', month: '2026-06', gen_val: 120 }),
      obs({ port: 'UP', month: '2026-05', gen_val: 100 }),
    ], '2026-06', '2026-05',
  );
  assert.deepEqual(sortRows(rows, 'delta').map((r) => r.port), ['UP', 'DOWN', 'UNKNOWN']);
});

test('name sort falls back to the code for unnamed ports and does not mutate its input', () => {
  const { ports } = splitNational(LIVE);
  const rows = joinMonths(ports, '2026-06', '2026-05');
  const before = rows.map((r) => r.port);
  const sorted = sortRows(rows, 'port_name');
  assert.deepEqual(rows.map((r) => r.port), before);
  assert.equal(sorted[0].port, '1791'); // digits sort ahead of the letter names
});

test('search matches published name or Schedule D code, case-insensitively', () => {
  const { ports } = splitNational(LIVE);
  const rows = joinMonths(ports, '2026-06', '2026-05');
  assert.deepEqual(filterRows(rows, 'laredo').map((r) => r.port), ['2304']);
  assert.deepEqual(filterRows(rows, '2704').map((r) => r.port), ['2704']);
  assert.equal(filterRows(rows, '  ').length, rows.length);
  assert.equal(filterRows(rows, 'nowhere').length, 0);
});
