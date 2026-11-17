// Provider clock units (2026-11-16). adsb.fi answers with `now` in SECONDS
// (1794880068.001); adsb.lol and airplanes.live answer in milliseconds. The
// viewport chain and the coverage gap-fill passed adsb.fi's value on as
// milliseconds, so every adsb.fi fix was dated January 1970 and the worldwide
// snapshot refused all of them — the fallback provider's rows never reached
// the worldwide feed, with nothing counting the loss.
// Run: npx tsx --test server/aircraftFixBus.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import { providerClockMs, PROVIDER_CLOCK_SECONDS_BELOW, subscribeFixes, type FixBatch } from "./aircraftFixBus";
import { fetchDiscs, planDiscs, type DiscProvider } from "./aircraftTiling";
import { createGapFiller } from "./coverageGapFill";
import { GlobalSnapshot, ALL_REFUSED_MIN_ROWS } from "./globalSnapshot";
import { upstreamBaseMs } from "./trackedPlanes";

const NOW = 1_794_880_068_400; // 2026-11-17T01:47:48Z, the live reading's moment
const FI_NOW_S = 1_794_880_068.001; // adsb.fi's own `now` in that answer

test("providerClockMs: milliseconds pass through, seconds are converted, garbage is null", () => {
  assert.equal(providerClockMs(1_794_880_068_001), 1_794_880_068_001);
  assert.equal(providerClockMs(FI_NOW_S), 1_794_880_068_001);
  assert.equal(providerClockMs(1_794_880_068), 1_794_880_068_000);
  for (const v of [null, undefined, "1794880068", NaN, Infinity, 0, -5, {}]) {
    assert.equal(providerClockMs(v), null, `not a clock: ${String(v)}`);
  }
  // the boundary: 1e11 itself is read as milliseconds (1973), just under it as seconds
  assert.equal(providerClockMs(PROVIDER_CLOCK_SECONDS_BELOW), 1e11);
  assert.equal(providerClockMs(PROVIDER_CLOCK_SECONDS_BELOW - 1), (1e11 - 1) * 1000);
});

test("providerClockMs: our clock is an upper bound only when it is given", () => {
  assert.equal(providerClockMs(NOW + 5_000, NOW), NOW, "a provider clock ahead of ours is held to ours");
  assert.equal(providerClockMs(NOW - 5_000, NOW), NOW - 5_000);
  assert.equal(providerClockMs(NOW + 5_000), NOW + 5_000, "no bound asked, none applied");
  assert.equal(providerClockMs(FI_NOW_S + 60, NOW), NOW, "bounded after the unit conversion");
});

test("upstreamBaseMs keeps its contract (now one rule, shared with every publisher)", () => {
  assert.equal(upstreamBaseMs({ now: NOW - 2_000 }, NOW), NOW - 2_000);
  assert.equal(upstreamBaseMs({ now: FI_NOW_S }, NOW), 1_794_880_068_001);
  assert.equal(upstreamBaseMs({ now: NOW + 9_000 }, NOW), NOW);
  assert.equal(upstreamBaseMs({}, NOW), NOW);
  assert.equal(upstreamBaseMs({ now: 0 }, NOW), NOW);
  assert.equal(upstreamBaseMs(null, NOW), NOW);
});

const FI: DiscProvider = {
  key: "adsbfi", label: "adsb.fi (ADS-B, community)", arr: "aircraft",
  url: (la, lo, r) => `https://opendata.adsb.fi/api/v2/lat/${la.toFixed(3)}/lon/${lo.toFixed(3)}/dist/${r}`,
};
const fiAnswer = (n: number) => ({
  now: FI_NOW_S,
  aircraft: Array.from({ length: n }, (_, i) => ({
    hex: `aca1d${i}`, flight: "ASA913  ", r: "N913AK", t: "B39M", lat: 40.09 + i * 0.01, lon: -79.4,
    alt_baro: 34000, gs: 440, track: 270, seen: 0.1, seen_pos: 0.088,
  })),
});
const okRes = (body: unknown) => ({ ok: true, status: 200, json: async () => body }) as unknown as Response;

test("viewport chain: an adsb.fi answer is published in milliseconds and the worldwide snapshot keeps its rows", async () => {
  const batches: FixBatch[] = [];
  const off = subscribeFixes((b) => { batches.push(b); });
  try {
    const plan = planDiscs(39.5, 40.5, -80, -79);
    const r = await fetchDiscs(plan, {
      providers: [FI], fetchImpl: (async () => okRes(fiAnswer(4))) as unknown as typeof fetch,
      sleep: async () => {}, governor: null,
    });
    assert.equal(r.aircraft.length, 4);
  } finally { off(); }
  assert.equal(batches.length, 1);
  assert.equal(batches[0].provider, "adsbfi");
  assert.equal(batches[0].upstreamNowMs, 1_794_880_068_001, "seconds were read as milliseconds before 2026-11-16");
  const snap = new GlobalSnapshot();
  assert.equal(snap.ingest(batches[0], NOW), 4, "every adsb.fi fix was refused as 56 years old before the fix");
  const row = snap.get("aca1d0");
  assert.ok(row);
  assert.equal(row.src, "adsbfi", "provenance stays the provider that served it");
  assert.equal(row.seenAt, 1_794_880_068_001 - 88, "fix time = the provider's clock minus the row's position age");
  assert.deepEqual(snap.summary(NOW).refused_too_old_by_source, {});
});

test("coverage gap-fill: adsb.fi rows are published in milliseconds too", async () => {
  const batches: FixBatch[] = [];
  const off = subscribeFixes((b) => { batches.push(b); });
  try {
    const filler = createGapFiller({
      enabled: true, now: () => NOW, fetchImpl: (async () => okRes(fiAnswer(3))) as unknown as typeof fetch,
    });
    const out = await filler.fillDisc({ centers: [{ lat: 40.1, lon: -79.4 }], radiusNm: 4 }, [], ["adsb.lol (ADS-B, community)"]);
    assert.equal(out.added, 3);
  } finally { off(); }
  assert.equal(batches.length, 1);
  assert.equal(batches[0].origin, "gapfill");
  assert.equal(batches[0].upstreamNowMs, 1_794_880_068_001);
  assert.equal(new GlobalSnapshot().ingest(batches[0], NOW), 3);
});

test("snapshot: fixes refused as too old are counted per provider, and a wholly refused batch is counted and logged", () => {
  const snap = new GlobalSnapshot();
  const mk = (provider: string, upstreamNowMs: number, n: number): FixBatch => ({
    provider, origin: "viewport", fetchedAt: NOW, upstreamNowMs,
    aircraft: Array.from({ length: n }, (_, i) => ({ icao24: `${provider.slice(0, 3)}${i}`, lat: 1, lon: 1, seen_pos: 1 })) as never,
  });
  const logged: string[] = [];
  const origErr = console.error;
  console.error = (...a: unknown[]) => { logged.push(a.map(String).join(" ")); };
  try {
    // the pre-fix shape: a seconds clock handed over as milliseconds
    assert.equal(snap.ingest(mk("adsbfi", FI_NOW_S, ALL_REFUSED_MIN_ROWS), NOW), 0);
    // a healthy batch with one long-silent row is NOT a suspect clock
    const mixed = mk("adsblol", NOW - 1_000, 3);
    (mixed.aircraft[0] as { seen_pos?: number }).seen_pos = 3_600;
    assert.equal(snap.ingest(mixed, NOW), 2);
    // below the row floor a wholly refused batch is counted but not flagged
    assert.equal(snap.ingest(mk("opensky", NOW - 3_600_000, ALL_REFUSED_MIN_ROWS - 1), NOW), 0);
  } finally { console.error = origErr; }
  const s = snap.summary(NOW);
  assert.deepEqual(s.refused_too_old_by_source, { adsbfi: ALL_REFUSED_MIN_ROWS, adsblol: 1, opensky: ALL_REFUSED_MIN_ROWS - 1 });
  assert.deepEqual(s.all_refused_batches_by_source, { adsbfi: 1 });
  assert.equal(logged.length, 1, "said once, plainly");
  assert.match(logged[0], /every fix of a adsbfi batch \(3 rows, via viewport\).*wrong unit/);
});
