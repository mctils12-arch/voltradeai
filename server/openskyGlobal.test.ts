// FLIGHT PROGRAM B1 — optional OpenSky global snapshot: SI-unit
// normalization, category decode, credit budgeter, token handling, and the
// no-credentials inert path (zero network calls). No live requests.
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  mapOpenSkyStates, CreditBudget, TokenCache, pollOpenSkyOnce, startOpenSkyGlobal,
  openskyConfigured, intervalMsFromEnv, budgetFromEnv,
  OPENSKY_STATES_URL, OPENSKY_TOKEN_URL, OPENSKY_CATEGORY_TO_ADSB, STATES_ALL_CREDIT_COST,
} from "./openskyGlobal";
import type { FixBatch } from "./aircraftFixBus";

// the real shape probed 2026-09-28 (extended=1 → 18 fields)
const LIVE_ROW = ["39de54", "TVF75UB ", "France", 1790627097, 1790627097, 6.2301, 46.4108, 10980.42, false, 219.35, 319.66, -0.33, null, 11430, "1000", false, 0, 0];

test("mapOpenSkyStates: SI units pass through the metric pipeline fields (never double-converted)", () => {
  const [p] = mapOpenSkyStates({ time: 1790627098, states: [LIVE_ROW] });
  assert.equal(p.icao24, "39de54");
  assert.equal(p.callsign, "TVF75UB");
  assert.equal(p.lon, 6.2301);
  assert.equal(p.lat, 46.4108);
  assert.equal(p.altitude_m, 10980, "baro_altitude is already meters");
  assert.equal(p.velocity_ms, 219, "velocity is already m/s");
  assert.equal(p.heading, 319.66);
  assert.equal(p.on_ground, false);
  assert.equal(p.category, null, "0 = no information");
  assert.equal(p.provider, "opensky");
  assert.equal(p.seen_at_ms, 1790627097000, "per-row fix time from time_position");
  assert.equal(p.type, null, "OpenSky states carry no type — never guessed");
});

test("mapOpenSkyStates: ground, geo-altitude fallback, MLAT origin, category decode, junk skipped", () => {
  const rows = [
    ["aaa001", "GND1", "X", 100, 100, 1, 2, 50, true, 3, 90, 0, null, 60, null, false, 0, 4],
    ["aaa002", "GEO1", "X", 100, 100, 1, 2, null, false, 200, 90, 0, null, 9000, null, false, 2, 6],
    ["aaa003", "NOPOS", "X", 100, 100, null, null, 1, false, 1, 1, 0, null, 1, null, false, 0, 0],
    ["aaa004", "NOTIME", "X", null, null, 1, 2, 1, false, 1, 1, 0, null, 1, null, false, 0, 0],
    "garbage",
  ];
  const out = mapOpenSkyStates({ states: rows });
  assert.deepEqual(out.map((p) => p.icao24), ["aaa001", "aaa002"]);
  assert.equal(out[0].altitude_m, null, "on ground -> no altitude (readsb mapping parity)");
  assert.equal(out[0].category, "A3", "4 = Large -> A3");
  assert.equal(out[1].altitude_m, 9000, "geo_altitude fallback when baro is null");
  assert.equal(out[1].category, "A5", "6 = Heavy -> A5");
  assert.equal(out[1].pos_type, "mlat", "position_source 2 = MLAT (ground-computed)");
  assert.equal(out[0].pos_type, null, "ADS-B source has no exact readsb subtype — null, not guessed");
  assert.equal(OPENSKY_CATEGORY_TO_ADSB[8], "A7", "rotorcraft");
  assert.equal(OPENSKY_CATEGORY_TO_ADSB[13], undefined, "reserved");
  assert.deepEqual(mapOpenSkyStates(null), []);
});

const DAY = Date.UTC(2026, 8, 28, 12);

test("CreditBudget: daily ceiling with reserve, whole-day interval, UTC rollover", () => {
  const b = new CreditBudget({ dailyCredits: 4000 });
  assert.equal(b.reserve, 400);
  assert.equal(b.cost, STATES_ALL_CREDIT_COST);
  assert.equal(b.minIntervalMs(), 96_000, "3600 credits / 4 = 900 calls -> one per 96s");
  let calls = 0;
  for (let i = 0; i < 2000 && b.canSpend(DAY); i++) { b.spend(DAY); calls++; }
  assert.equal(calls, 900, "stops BEFORE exhausting the account (reserve kept)");
  assert.equal(b.canSpend(DAY + 1000), false);
  assert.equal(b.canSpend(DAY + 13 * 3600_000), true, "next UTC day resets the counter");
});

test("CreditBudget: X-Rate-Limit-Remaining low -> stop until next UTC day; 429 honors retry-after", () => {
  const b = new CreditBudget({ dailyCredits: 4000 });
  b.noteResponse(DAY, 200, "500", null);
  assert.equal(b.canSpend(DAY), true);
  b.noteResponse(DAY, 200, "7", null);
  assert.equal(b.canSpend(DAY + 60_000), false, "upstream counter nearly out");
  const c = new CreditBudget();
  c.noteResponse(DAY, 429, null, "120");
  assert.equal(c.canSpend(DAY + 119_000), false);
  assert.equal(c.canSpend(DAY + 121_000), true);
  assert.match(String(c.status(DAY).pause_reason), /429/);
});

test("interval + budget from env: default 100s, never faster than the budget allows", () => {
  const b = budgetFromEnv({} as any);
  assert.equal(intervalMsFromEnv({} as any, b), 100_000);
  assert.equal(intervalMsFromEnv({ OPENSKY_INTERVAL_S: "10" } as any, b), 96_000, "floored at the whole-day spacing");
  assert.equal(intervalMsFromEnv({ OPENSKY_INTERVAL_S: "300" } as any, b), 300_000);
  assert.equal(budgetFromEnv({ OPENSKY_DAILY_CREDITS: "8000" } as any).dailyCredits, 8000);
});

test("openskyConfigured requires BOTH credentials; inert handle makes ZERO calls", async () => {
  assert.equal(openskyConfigured({} as any), false);
  assert.equal(openskyConfigured({ OPENSKY_CLIENT_ID: "a" } as any), false);
  assert.equal(openskyConfigured({ OPENSKY_CLIENT_ID: "a", OPENSKY_CLIENT_SECRET: " " } as any), false);
  assert.equal(openskyConfigured({ OPENSKY_CLIENT_ID: "a", OPENSKY_CLIENT_SECRET: "b" } as any), true);
  let calls = 0;
  const h = startOpenSkyGlobal({ env: {} as any, fetchImpl: (async () => { calls++; }) as any });
  await new Promise((r) => setTimeout(r, 10));
  h.stop();
  assert.equal(calls, 0);
  assert.equal((h.status() as any).enabled, false);
});

test("pollOpenSkyOnce: OAuth client-credentials token, bearer states call, budget spent, published as opensky", async () => {
  const calls: { url: string; init?: RequestInit }[] = [];
  const got: FixBatch[] = [];
  const fetchImpl = (async (url: unknown, init?: RequestInit) => {
    calls.push({ url: String(url), init });
    if (String(url) === OPENSKY_TOKEN_URL) return { ok: true, status: 200, json: async () => ({ access_token: "tok1", expires_in: 1800 }) };
    return {
      ok: true, status: 200,
      headers: { get: (k: string) => (k === "x-rate-limit-remaining" ? "3900" : null) },
      json: async () => ({ time: 1790627098, states: [LIVE_ROW] }),
    };
  }) as any;
  const budget = new CreditBudget();
  const tokens = new TokenCache("id1", "sec1", fetchImpl);
  const r = await pollOpenSkyOnce({ fetchImpl, tokens, budget, now: () => DAY, publish: (b) => got.push(b) });
  assert.equal(r.ok, true);
  assert.equal(r.count, 1);
  assert.equal(calls[0].url, OPENSKY_TOKEN_URL);
  assert.equal(calls[0].init?.method, "POST");
  assert.match(String(calls[0].init?.body), /grant_type=client_credentials/);
  assert.match(String(calls[0].init?.body), /client_id=id1/);
  assert.equal(calls[1].url, OPENSKY_STATES_URL);
  assert.equal((calls[1].init?.headers as Record<string, string>).Authorization, "Bearer tok1");
  assert.equal(budget.status(DAY).credits_used_today, 4);
  assert.equal(budget.status(DAY).upstream_remaining, 3900);
  assert.equal(got[0].provider, "opensky");
  assert.equal(got[0].origin, "opensky");
  // the token is cached: a second poll does not re-authenticate
  await pollOpenSkyOnce({ fetchImpl, tokens, budget, now: () => DAY + 100_000, publish: () => {} });
  assert.equal(calls.filter((c) => c.url === OPENSKY_TOKEN_URL).length, 1);
});

test("pollOpenSkyOnce: 401 invalidates the token; exhausted budget skips WITHOUT a request", async () => {
  let tokenCalls = 0;
  const fetchImpl = (async (url: unknown) => {
    if (String(url) === OPENSKY_TOKEN_URL) { tokenCalls++; return { ok: true, status: 200, json: async () => ({ access_token: `t${tokenCalls}`, expires_in: 1800 }) }; }
    return { ok: false, status: 401, headers: { get: () => null } };
  }) as any;
  const tokens = new TokenCache("i", "s", fetchImpl);
  const budget = new CreditBudget();
  const r = await pollOpenSkyOnce({ fetchImpl, tokens, budget, now: () => DAY });
  assert.equal(r.ok, false);
  assert.equal(r.status, 401);
  await pollOpenSkyOnce({ fetchImpl, tokens, budget, now: () => DAY + 1000 });
  assert.equal(tokenCalls, 2, "re-authenticated after the 401");
  const spent = new CreditBudget({ dailyCredits: 8, reserve: 0 });
  spent.spend(DAY); spent.spend(DAY);
  let n = 0;
  const r2 = await pollOpenSkyOnce({ fetchImpl: (async () => { n++; }) as any, tokens, budget: spent, now: () => DAY });
  assert.equal(r2.skipped, "credit budget");
  assert.equal(n, 0);
});
