import { test } from "node:test";
import assert from "node:assert/strict";
import {
  findCloseApproaches, findCloseApproachesAsync, confidenceFor, distNm, CLOSE_APPROACH_BASIS, CA_DEFAULTS,
  type CloseApproachTrack, type Fix,
} from "./closeApproach";

const FT = 0.3048; // ft -> m (archive altitudes are meters)
const T0 = 1_759_000_000;

/** straight-line track: start (lat, lon), heading deg, speed kt, alt ft,
 *  fixes every `every` seconds from t0 for n fixes */
function line(hex: string, lat: number, lon: number, hdg: number, kt: number, altFt: number | null,
              t0: number, n: number, every: number, callsign?: string): CloseApproachTrack {
  const pts: Fix[] = [];
  const nmPerSec = kt / 3600;
  for (let k = 0; k < n; k++) {
    const dNm = nmPerSec * every * k;
    const la = lat + (dNm * Math.cos((hdg * Math.PI) / 180)) / 60;
    const lo = lon + (dNm * Math.sin((hdg * Math.PI) / 180)) / (60 * Math.cos((lat * Math.PI) / 180));
    pts.push([t0 + every * k, la, lo, altFt == null ? null : altFt * FT]);
  }
  return { i: hex, c: callsign, points: pts };
}

test("crossing tracks at the same level are detected at the right instant", () => {
  // A flies east, B flies north; both reach (40, -100) at T0 + 600 s.
  const kt = 450;
  const nm600 = (kt / 3600) * 600; // 75 nm
  const A = line("aaaaaa", 40, -100 - nm600 / (60 * Math.cos(40 * Math.PI / 180)), 90, kt, 35000, T0, 41, 30, "AAL1");
  const B = line("bbbbbb", 40 - nm600 / 60, -100, 0, kt, 35000, T0, 41, 30, "UAL2");
  const r = findCloseApproaches([A, B]);
  assert.equal(r.approaches.length, 1);
  const ca = r.approaches[0];
  assert.equal(ca.a, "aaaaaa");
  assert.equal(ca.b, "bbbbbb");
  assert.equal(ca.ca, "AAL1");
  assert.equal(ca.cb, "UAL2");
  assert.ok(Math.abs(ca.t - (T0 + 600) * 1000) <= 3000, `t=${ca.t} expected ≈${(T0 + 600) * 1000}`);
  assert.ok(ca.horizNm < 0.2, `horiz ${ca.horizNm}`);
  assert.ok(ca.vertFt < 1);
  assert.equal(ca.basis, CLOSE_APPROACH_BASIS);
  assert.match(ca.basis, /not an official loss-of-separation report/);
  // 30 s cadence: every instant is within 15 s of a real fix → high
  assert.equal(ca.confidence, "high");
  assert.ok(Math.abs(ca.lat - 40) < 0.01 && Math.abs(ca.lon + 100) < 0.01);
  assert.equal(ca.altAFt, 35000);
});

test("parallel tracks 6 nm apart are never flagged", () => {
  const A = line("aaaaaa", 40, -100, 90, 450, 35000, T0, 60, 30);
  const B = line("bbbbbb", 40 + 6 / 60, -100, 90, 450, 35000, T0, 60, 30);
  const r = findCloseApproaches([A, B]);
  assert.equal(r.approaches.length, 0);
  assert.equal(r.found, 0);
});

test("3 nm apart with 800 ft vertical IS flagged; 1,200 ft is not", () => {
  const A = line("aaaaaa", 40, -100, 90, 450, 35000, T0, 20, 30);
  const B = line("bbbbbb", 40 + 3 / 60, -100, 90, 450, 35800, T0, 20, 30);
  const r = findCloseApproaches([A, B]);
  assert.equal(r.approaches.length, 1);
  assert.ok(Math.abs(r.approaches[0].horizNm - 3) < 0.05, `horiz ${r.approaches[0].horizNm}`);
  assert.ok(Math.abs(r.approaches[0].vertFt - 800) < 2, `vert ${r.approaches[0].vertFt}`);

  const C = line("cccccc", 40 + 3 / 60, -100, 90, 450, 36200, T0, 20, 30);
  assert.equal(findCloseApproaches([A, C]).approaches.length, 0);
});

test("sparse fixes are skipped: no interpolation across a > 180 s gap", () => {
  // Two converging aircraft that DO cross — but each only has fixes every
  // 5 min and the crossing falls 150 s from the nearest fix on either side:
  // interpolating across a 300 s segment is not allowed, so the encounter is
  // honestly unknown and never flagged.
  const kt = 450;
  const nm = (kt / 3600) * 750; // crossing 750 s after the first fix
  const A = line("aaaaaa", 40, -100 - nm / (60 * Math.cos(40 * Math.PI / 180)), 90, kt, 35000, T0, 5, 300);
  const B = line("bbbbbb", 40 - nm / 60, -100, 0, kt, 35000, T0, 5, 300);
  const r = findCloseApproaches([A, B]);
  assert.equal(r.approaches.length, 0);
  // …while the same geometry at 60 s cadence is found (medium/high
  // confidence — the nearest fix is ≤ 30 s away)
  const A2 = line("aaaaaa", 40, -100 - nm / (60 * Math.cos(40 * Math.PI / 180)), 90, kt, 35000, T0, 26, 60);
  const B2 = line("bbbbbb", 40 - nm / 60, -100, 0, kt, 35000, T0, 26, 60);
  const r2 = findCloseApproaches([A2, B2]);
  assert.equal(r2.approaches.length, 1);
  assert.notEqual(r2.approaches[0].confidence, "low");
  assert.ok(Math.abs(r2.approaches[0].t - (T0 + 750) * 1000) <= 3000);
});

test("missing altitude on either aircraft = not evaluated (vertical unknown)", () => {
  const A = line("aaaaaa", 40, -100, 90, 450, 35000, T0, 20, 30);
  const B = line("bbbbbb", 40 + 1 / 60, -100, 90, 450, null, T0, 20, 30);
  assert.equal(findCloseApproaches([A, B]).approaches.length, 0);
});

test("ground / same-airport low-level pairs are excluded", () => {
  // both at 800-1,500 ft MSL, 1 nm apart (pattern traffic / parallel finals)
  const A = line("aaaaaa", 40, -100, 90, 140, 800, T0, 20, 30);
  const B = line("bbbbbb", 40 + 1 / 60, -100, 90, 140, 1500, T0, 20, 30);
  assert.equal(findCloseApproaches([A, B]).approaches.length, 0);
  // at a high-elevation airport (Denver-like, 5,400 ft): 6,000 vs 6,500 ft MSL
  // is pattern altitude there — excluded only when the airport lookup says so
  const C = line("cccccc", 39.86, -104.67, 90, 140, 6000, T0, 20, 30);
  const D = line("dddddd", 39.86 + 1 / 60, -104.67, 90, 140, 6500, T0, 20, 30);
  assert.equal(findCloseApproaches([C, D]).approaches.length, 1, "without an airport lookup: flagged");
  const airportNear = (la: number, lo: number) =>
    Math.abs(la - 39.87) < 0.2 && Math.abs(lo + 104.5) < 0.6 ? { elevFt: 5434 } : null;
  assert.equal(findCloseApproaches([C, D], { airportNear }).approaches.length, 0, "near a 5,434 ft field: excluded");
  // one aircraft well above the low band keeps the pair evaluable
  const E = line("eeeeee", 40 + 1 / 60, -100, 90, 140, 2600, T0, 20, 30);
  const F = line("ffffff", 40, -100, 90, 140, 2000, T0, 20, 30);
  assert.equal(findCloseApproaches([E, F]).approaches.length, 1);
  // tracks with no altitude at all (on-ground rows carry none) never enter
  const G1 = line("111111", 40, -100, 90, 10, null, T0, 20, 30);
  const G2 = line("222222", 40, -100.001, 90, 10, null, T0, 20, 30);
  const r = findCloseApproaches([G1, G2]);
  assert.equal(r.approaches.length, 0);
  assert.equal(r.evaluated_hexes, 0);
});

test("non-ICAO (~) rebroadcast addresses are excluded and counted", () => {
  const A = line("aaaaaa", 40, -100, 90, 450, 35000, T0, 20, 30);
  const echo = line("~aaaaa", 40.001, -100, 90, 450, 35000, T0, 20, 30);
  const r = findCloseApproaches([A, echo]);
  assert.equal(r.approaches.length, 0);
  assert.equal(r.excluded_non_icao, 1);
});

test("confidence tiers follow the worse aircraft's nearest-fix distance", () => {
  assert.equal(confidenceFor(0, 20), "high");
  assert.equal(confidenceFor(21, 5), "medium");
  assert.equal(confidenceFor(60, 60), "medium");
  assert.equal(confidenceFor(61, 0), "low");
  assert.equal(confidenceFor(90, 90), "low");
  // 180 s-spaced fixes: the crossing sits mid-segment (90 s from any fix) → low
  const kt = 300;
  const nm = (kt / 3600) * 540;
  const A = line("aaaaaa", 40, -100 - nm / (60 * Math.cos(40 * Math.PI / 180)), 90, kt, 20000, T0, 7, 180);
  const B = line("bbbbbb", 40 - nm / 60, -100, 0, kt, 20000, T0, 7, 180);
  const r = findCloseApproaches([A, B]);
  assert.equal(r.approaches.length, 1);
  // fixes at 0,180,360,540 s and the crossing is at 540 s — a real fix
  // instant for both aircraft → high, even though the cadence is sparse
  assert.equal(r.approaches[0].confidence, "high");
  // shift the crossing to mid-segment (fixes at 0,180,360,540 but crossing
  // at 450 s): 90 s from the nearest fix on both → low
  const nm2 = (kt / 3600) * 450;
  const A2 = line("aaaaaa", 40, -100 - nm2 / (60 * Math.cos(40 * Math.PI / 180)), 90, kt, 20000, T0, 7, 180);
  const B2 = line("bbbbbb", 40 - nm2 / 60, -100, 0, kt, 20000, T0, 7, 180);
  const r2 = findCloseApproaches([A2, B2]);
  assert.equal(r2.approaches.length, 1);
  assert.equal(r2.approaches[0].confidence, "low");
});

test("two separate encounters of the same pair are reported separately", () => {
  // formation for 5 min, apart for 20 min, formation again for 5 min
  const pts = (off: number, altFt: number): Fix[] => {
    const out: Fix[] = [];
    for (let k = 0; k <= 60; k++) {
      const t = T0 + k * 30;
      const minute = (k * 30) / 60;
      const apart = minute > 5 && minute < 25 ? 20 / 60 : 0; // 20 nm lateral offset
      out.push([t, 40 + off + apart * (off > 0 ? 1 : 0), -100 + k * 0.05, altFt * FT]);
    }
    return out;
  };
  const r = findCloseApproaches([
    { i: "aaaaaa", points: pts(0, 30000) },
    { i: "bbbbbb", points: pts(2 / 60, 30000) },
  ]);
  assert.equal(r.approaches.length, 2, JSON.stringify(r.approaches.map((a) => [a.t, a.horizNm])));
});

test("output is capped, sorted by minimum separation, and reports what it dropped", () => {
  const tracks: CloseApproachTrack[] = [];
  for (let k = 0; k < 8; k++) {
    // pairs at 0.5k nm lateral offsets, well apart from each other
    tracks.push(line(`a${k}0000`, 30 + k, -100, 90, 400, 30000, T0, 10, 30));
    tracks.push(line(`b${k}0000`, 30 + k + (0.5 + 0.5 * k) / 60, -100, 90, 400, 30000, T0, 10, 30));
  }
  const r = findCloseApproaches(tracks, { cap: 3 });
  assert.equal(r.found, 8);
  assert.equal(r.capped, true);
  assert.equal(r.approaches.length, 3);
  for (let k = 1; k < r.approaches.length; k++) {
    assert.ok(r.approaches[k].horizNm >= r.approaches[k - 1].horizNm);
  }
  assert.ok(Math.abs(r.approaches[0].horizNm - 0.5) < 0.02);
});

test("antimeridian: a pair straddling ±180° is found", () => {
  const A = line("aaaaaa", 50, 179.98, 90, 450, 36000, T0, 12, 30);
  const B = line("bbbbbb", 50 + 1 / 60, 179.98, 90, 450, 36000, T0, 12, 30);
  const r = findCloseApproaches([A, B]);
  assert.equal(r.approaches.length, 1);
  assert.ok(Math.abs(r.approaches[0].horizNm - 1) < 0.05);
  assert.ok(Math.abs(Math.abs(r.approaches[0].lon) - 180) < 1);
});

test("grid bucketing equals brute force on a random 500-track fixture", () => {
  let seed = 12345;
  const rnd = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
  const tracks: CloseApproachTrack[] = [];
  for (let k = 0; k < 500; k++) {
    const lat = 38 + rnd() * 3, lon = -102 + rnd() * 4;
    const hdg = rnd() * 360, kt = 250 + rnd() * 250;
    // a few shared flight levels so vertical proximity actually happens
    const alt = [24000, 28000, 31000, 33000, 35000, 36000, 37000, 39000][Math.floor(rnd() * 8)] + Math.round(rnd() * 600);
    const every = [20, 30, 45, 60, 75, 120][Math.floor(rnd() * 6)];
    const t0 = T0 + Math.floor(rnd() * 1800);
    const n = 6 + Math.floor(rnd() * 14);
    const tr = line(`h${k.toString(16).padStart(5, "0")}`, lat, lon, hdg, kt, alt, t0, n, every);
    // sprinkle honest altitude gaps
    if (rnd() < 0.2) tr.points[Math.floor(rnd() * tr.points.length)][3] = null;
    tracks.push(tr);
  }
  const grid = findCloseApproaches(tracks, { cap: 10_000 });
  const brute = findCloseApproaches(tracks, { cap: 10_000, useGrid: false });
  assert.ok(brute.found > 5, `fixture should produce approaches (got ${brute.found})`);
  assert.deepEqual(grid, brute);
  return findCloseApproachesAsync(tracks, { cap: 10_000 }, 1).then((asyncGrid) => {
    assert.deepEqual(asyncGrid, grid, "the event-loop-yielding driver returns the identical result");
  });
});

test("grid stays fast on thousands of tracks (O(n), not O(n²) in hexes)", () => {
  let seed = 99;
  const rnd = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
  const tracks: CloseApproachTrack[] = [];
  for (let k = 0; k < 4000; k++) {
    tracks.push(line(`p${k.toString(16).padStart(5, "0")}`, 25 + rnd() * 24, -125 + rnd() * 58,
      rnd() * 360, 250 + rnd() * 250, 20000 + Math.round(rnd() * 20000), T0 + Math.floor(rnd() * 600), 40, 45));
  }
  const t = Date.now();
  const r = findCloseApproaches(tracks);
  const ms = Date.now() - t;
  assert.ok(r.pieces > 150_000);
  assert.ok(ms < 8000, `took ${ms} ms for ${r.pieces} pieces`);
});

test("defaults are the documented en-route minima", () => {
  assert.equal(CA_DEFAULTS.HORIZ_NM, 5);
  assert.equal(CA_DEFAULTS.VERT_FT, 1000);
  assert.equal(CA_DEFAULTS.FIX_WINDOW_SEC, 90);
  assert.equal(CA_DEFAULTS.CAP, 200);
  assert.ok(Math.abs(distNm(40, -100, 41, -100) - 60.04) < 0.1);
});
