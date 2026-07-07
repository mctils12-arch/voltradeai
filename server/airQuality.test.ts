// Google Air Quality datacore stream (EDGE-DOCTRINE root). No live network:
// every test drives an injected FetchFn against fixtures mirrored from the
// DOCUMENTED contract (developers.google.com/maps/documentation/air-quality).
// Covers: key gate (awaiting_key = zero calls), awaiting_enable (403
// SERVICE_DISABLED → honest state, no throw, no archive), happy-path extract
// (uaqi/usa_epa/pm25/no2), dedup siteId|dateTime, day-file append + gz-merge,
// free-tier budget guard (rotating subset, never over budget), key-scrub, and
// the required manifest envelope.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import { fileURLToPath } from "node:url";
import {
  airQualityEnabled, airQualityKey, SITES,
  cyclesPerDay, maxSitesPerCycle, selectCycleSites, callBudgetState,
  currentConditionsUrl, currentConditionsBody, heatmapTileUrl, isServiceDisabled,
  parseCurrentConditions, scrubKey,
  archiveAirQuality, gzipOldAqDays, refreshAirQuality, latestAirQuality,
  seedFileInWindow, SEED_WINDOW_DAYS, DAILY_CALL_BUDGET,
} from "./airQuality";

const KEY = "AIzaSyFAKE_TESTKEY_0123456789abcdef";
const here = path.dirname(fileURLToPath(import.meta.url));

// Fixture mirrored from the documented currentConditions:lookup response.
const CURRENT = (dateTime: string, opts: { uaqi?: number; epa?: number; pm25?: number; no2?: number } = {}) => ({
  dateTime,
  regionCode: "us",
  indexes: [
    { code: "uaqi", displayName: "Universal AQI", aqi: opts.uaqi ?? 72, aqiDisplay: String(opts.uaqi ?? 72),
      color: { red: 100, green: 180, blue: 60 }, category: "Good air quality", dominantPollutant: "pm25" },
    { code: "usa_epa", displayName: "AQI (US)", aqi: opts.epa ?? 54, aqiDisplay: String(opts.epa ?? 54),
      category: "Moderate air quality", dominantPollutant: "pm25" },
  ],
  pollutants: [
    { code: "pm25", displayName: "PM2.5", fullName: "Fine particulate matter (<2.5µm)",
      concentration: { value: opts.pm25 ?? 12.7, units: "MICROGRAMS_PER_CUBIC_METER" } },
    { code: "no2", displayName: "NO2", fullName: "Nitrogen dioxide",
      concentration: { value: opts.no2 ?? 21.4, units: "PARTS_PER_BILLION" } },
    { code: "o3", displayName: "O3", fullName: "Ozone", concentration: { value: 40, units: "PARTS_PER_BILLION" } },
  ],
});

// Documented Google API "service disabled" error envelope (HTTP 403).
const DISABLED_BODY = JSON.stringify({
  error: {
    code: 403,
    message: "Air Quality API has not been used in project 123456 before or it is disabled. Enable it by visiting https://console.developers.google.com/apis/api/airquality.googleapis.com/overview then retry.",
    status: "PERMISSION_DENIED",
    details: [{ "@type": "type.googleapis.com/google.rpc.ErrorInfo", reason: "SERVICE_DISABLED",
      domain: "googleapis.com", metadata: { service: "airquality.googleapis.com", consumer: "projects/123456" } }],
  },
});

const okResp = (obj: any) => ({ ok: true, status: 200, text: async () => JSON.stringify(obj) });
const twoSites = [{ id: "site_a", lat: 35.9, lon: -96.7 }, { id: "site_b", lat: 33.7, lon: -118.2 }];

test("key gate: GOOGLE_MAPS_API_KEY activates; keyless = awaiting_key, zero network calls", async () => {
  assert.equal(airQualityEnabled({} as any), false);
  assert.equal(airQualityEnabled({ GOOGLE_MAPS_API_KEY: KEY } as any), true);
  assert.equal(airQualityKey({ GOOGLE_MAPS_API_KEY: KEY } as any), KEY);
  let called = 0;
  const spy = async () => { called++; return okResp(CURRENT("2026-07-07T15:00:00Z")); };
  await refreshAirQuality(spy as any, {} as any, Date.parse("2026-07-07T15:30:00Z"),
    fs.mkdtempSync(path.join(os.tmpdir(), "aq-")), 0, twoSites);
  assert.equal(called, 0, "no key ⇒ never calls the network");
});

test("url + body contract: POST currentConditions, key URL-encoded, LOCAL_AQI + POLLUTANT_CONCENTRATION requested", () => {
  const u = currentConditionsUrl("se cret&x");
  assert.ok(u.startsWith("https://airquality.googleapis.com/v1/currentConditions:lookup?key="));
  assert.ok(u.includes("key=se%20cret%26x"), "key URL-encoded");
  const b = currentConditionsBody(35.9487, -96.7587);
  assert.equal(b.universalAqi, true);
  assert.deepEqual(b.location, { latitude: 35.9487, longitude: -96.7587 });
  assert.ok(b.extraComputations.includes("LOCAL_AQI") && b.extraComputations.includes("POLLUTANT_CONCENTRATION"));
  // heatmap template (later client layer)
  assert.equal(heatmapTileUrl("PM25_INDIGO_PERSIAN", 4, 3, 6, "K"),
    "https://airquality.googleapis.com/v1/mapTypes/PM25_INDIGO_PERSIAN/heatmapTiles/4/3/6?key=K");
});

test("parse: extracts universal + US EPA AQI and pm25/no2 concentrations; no dateTime ⇒ null", () => {
  const r = parseCurrentConditions(CURRENT("2026-07-07T15:00:00Z", { uaqi: 80, epa: 61, pm25: 9.3, no2: 15.2 }), "site_a", "2026-07-07")!;
  assert.equal(r.siteId, "site_a");
  assert.equal(r.dateTime, "2026-07-07T15:00:00Z");
  assert.equal(r.uaqi, 80);
  assert.equal(r.uaqi_dominant, "pm25");
  assert.equal(r.epa_aqi, 61);
  assert.equal(r.epa_category, "Moderate air quality");
  assert.equal(r.pm25, 9.3);
  assert.equal(r.pm25_units, "MICROGRAMS_PER_CUBIC_METER");
  assert.equal(r.no2, 15.2);
  assert.equal(r.no2_units, "PARTS_PER_BILLION");
  assert.ok(r.pollutants.length >= 3, "full pollutant list retained");
  assert.equal(parseCurrentConditions({ indexes: [], pollutants: [] }, "x", "y"), null, "no dateTime ⇒ null, never fabricated");
});

test("awaiting_enable: 403 SERVICE_DISABLED ⇒ honest state, no throw, no archive write", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "aq-"));
  let called = 0;
  const fake = async () => { called++; return { ok: false, status: 403, text: async () => DISABLED_BODY }; };
  await refreshAirQuality(fake as any, { GOOGLE_MAPS_API_KEY: KEY } as any,
    Date.parse("2026-07-07T15:30:00Z"), base, 0, twoSites);
  const hit = latestAirQuality()!;
  assert.ok(hit, "cache set even when disabled");
  assert.equal(hit.awaiting_enable, true, "awaiting_enable surfaced");
  assert.equal(called, 1, "disabled is project-wide — stops after the first call, doesn't burn the budget");
  assert.ok(String(hit.issues.site_a).includes("api_not_enabled"));
  assert.ok(!fs.existsSync(path.join(base, "airquality")), "nothing archived while disabled");
  assert.ok(isServiceDisabled(403, DISABLED_BODY));
  assert.ok(!isServiceDisabled(200, DISABLED_BODY), "only 403 counts");
  assert.ok(!isServiceDisabled(403, "{}"), "a plain 403 without the reason is not treated as disabled");
});

test("happy path: fixture ⇒ archived reading with correct extracted fields; awaiting_enable clears", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "aq-"));
  const fake = async (_u: string, init: any) => {
    // body carries the polled coordinate — return a per-site reading
    const body = JSON.parse(init.body);
    const isA = body.location.latitude === twoSites[0].lat;
    return okResp(CURRENT("2026-07-07T15:00:00Z", isA ? { pm25: 30.1, no2: 44.0 } : { pm25: 5.0, no2: 8.0 }));
  };
  await refreshAirQuality(fake as any, { GOOGLE_MAPS_API_KEY: KEY } as any,
    Date.parse("2026-07-07T15:30:00Z"), base, 0, twoSites);
  const hit = latestAirQuality()!;
  assert.equal(hit.awaiting_enable, false);
  const a = hit.readings.find((r) => r.siteId === "site_a")!;
  assert.equal(a.pm25, 30.1);
  assert.equal(a.no2, 44.0);
  assert.equal(a.epa_aqi, 54);
  const dir = path.join(base, "airquality");
  const lines = fs.readFileSync(path.join(dir, "2026-07-07.jsonl"), "utf8").trim().split("\n").map((l) => JSON.parse(l));
  assert.equal(lines.length, 2, "both sites archived");
  assert.ok(lines.every((r) => r.siteId && r.dateTime === "2026-07-07T15:00:00Z" && r.rt === "2026-07-07"));
});

test("dedup siteId|dateTime across polls: same reading re-served ⇒ archived once", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "aq-"));
  const mk = (siteId: string) => parseCurrentConditions(CURRENT("2026-05-01T09:00:00Z"), siteId, "2026-05-01")!;
  const first = [mk("dd_a"), mk("dd_b")];
  assert.equal(archiveAirQuality(first, base), 2, "two fresh readings");
  assert.equal(archiveAirQuality(first, base), 0, "identical siteId|dateTime never re-archived (next poll)");
  // a genuinely new hour for the same site DOES append
  const nextHour = parseCurrentConditions(CURRENT("2026-05-01T10:00:00Z"), "dd_a", "2026-05-01")!;
  assert.equal(archiveAirQuality([nextHour], base), 1, "new dateTime for same site appends");
  const lines = fs.readFileSync(path.join(base, "airquality", "2026-05-01.jsonl"), "utf8").trim().split("\n");
  assert.equal(lines.length, 3);
});

test("day-file append + gz merge: late reading after gz merges, never overwrites", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "aq-"));
  const dir = path.join(base, "airquality");
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, "2026-04-01.jsonl"), '{"siteId":"z","dateTime":"2026-04-01T00:00:00Z"}\n');
  assert.equal(gzipOldAqDays(base, Date.parse("2026-07-07")), 1, "old day gz'd");
  assert.ok(fs.existsSync(path.join(dir, "2026-04-01.jsonl.gz")) && !fs.existsSync(path.join(dir, "2026-04-01.jsonl")));
  // a late reading appends a fresh plain file after the gz already exists
  fs.writeFileSync(path.join(dir, "2026-04-01.jsonl"), '{"siteId":"z","dateTime":"2026-04-01T01:00:00Z"}\n');
  assert.equal(gzipOldAqDays(base, Date.parse("2026-07-07")), 1);
  const merged = zlib.gunzipSync(fs.readFileSync(path.join(dir, "2026-04-01.jsonl.gz"))).toString();
  assert.equal(merged.trim().split("\n").length, 2, "prior gz rows survive the late-reading re-gz");
  assert.ok(seedFileInWindow("2026-06-20.jsonl", Date.parse("2026-07-07")));
  assert.ok(!seedFileInWindow("2020-01-01.jsonl.gz", Date.parse("2026-07-07")), `seed window bounded to ${SEED_WINDOW_DAYS}d`);
});

test("budget guard: 16 real sites fit; a large list self-limits to a rotating subset, never over budget", async () => {
  // math: 3h poll ⇒ 8 cycles/day; budget 150 ⇒ ≤18 sites/cycle
  assert.equal(cyclesPerDay(), 8);
  assert.equal(maxSitesPerCycle(), 18);
  assert.ok(SITES.length <= 18, `real site list (${SITES.length}) polls fully every cycle`);
  const full = callBudgetState(SITES.length, SITES.length, false);
  assert.equal(full.rotating_subset, false);
  assert.ok(full.projected_calls_per_day <= DAILY_CALL_BUDGET, `${full.projected_calls_per_day} ≤ ${DAILY_CALL_BUDGET}`);

  // synthetic large list ⇒ rotation
  const big = Array.from({ length: 40 }, (_, i) => ({ id: `s${i}`, lat: 0, lon: 0 }));
  const c0 = selectCycleSites(big, 0, 18), c1 = selectCycleSites(big, 1, 18), c2 = selectCycleSites(big, 2, 18);
  assert.equal(c0.batch.length, 18);
  assert.equal(c0.limited, true);
  const covered = new Set([...c0.batch, ...c1.batch, ...c2.batch].map((s) => s.id));
  assert.equal(covered.size, 40, "three cycles visit every site (no starvation)");
  const st = callBudgetState(40, 18, true);
  assert.equal(st.rotating_subset, true);
  assert.ok(st.projected_calls_per_day <= DAILY_CALL_BUDGET, `${st.projected_calls_per_day} ≤ ${DAILY_CALL_BUDGET} even with 40 sites`);

  // end-to-end: refresh over 40 sites issues at most maxSitesPerCycle calls
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "aq-"));
  let calls = 0;
  const fake = async () => { calls++; return okResp(CURRENT("2026-07-07T16:00:00Z")); };
  await refreshAirQuality(fake as any, { GOOGLE_MAPS_API_KEY: KEY } as any,
    Date.parse("2026-07-07T16:30:00Z"), base, 0, big);
  assert.equal(calls, 18, "poller self-limits to the per-cycle cap");
  const hit = latestAirQuality()!;
  assert.equal(hit.budget.rotating_subset, true);
  assert.ok(hit.budget.projected_calls_per_day <= DAILY_CALL_BUDGET);
});

test("key scrub: the API key never lands in an archived record or an issue string", async () => {
  assert.equal(scrubKey(`GET https://x/?key=${KEY}&a=1`, KEY), "GET https://x/?key=REDACTED&a=1");
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "aq-"));
  // a fetch that throws an error echoing the full URL (key in the query)
  const throwing = async (url: string) => { throw new Error(`connect ECONNREFUSED ${url}`); };
  await refreshAirQuality(throwing as any, { GOOGLE_MAPS_API_KEY: KEY } as any,
    Date.parse("2026-07-07T15:30:00Z"), base, 0, twoSites);
  const hit = latestAirQuality()!;
  const issueBlob = JSON.stringify(hit.issues);
  assert.ok(!issueBlob.includes(KEY), "key scrubbed from issue strings");
  assert.ok(issueBlob.includes("REDACTED"), "scrubbed placeholder present");
  // and a real archived reading carries no key
  const good = async (_u: string, init: any) => okResp(CURRENT("2026-07-07T17:00:00Z"));
  await refreshAirQuality(good as any, { GOOGLE_MAPS_API_KEY: KEY } as any,
    Date.parse("2026-07-07T17:30:00Z"), base, 0, twoSites);
  const text = fs.readFileSync(path.join(base, "airquality", "2026-07-07.jsonl"), "utf8");
  assert.ok(!text.includes(KEY), "no archived record contains the key");
});

test("manifest: airquality.json carries the full universal envelope", () => {
  const doc = JSON.parse(fs.readFileSync(path.join(here, "..", "datacore", "manifests", "airquality.json"), "utf8"));
  for (const k of ["stream", "storage", "written_by", "source", "license", "attribution",
    "schema_version", "field_map", "confidence_model", "geo_fields", "entity_key", "started", "cadence"]) {
    assert.ok(k in doc, `manifest missing '${k}'`);
  }
  assert.equal(doc.stream, "airquality");
  assert.ok(Object.keys(doc.field_map).length >= 3);
  assert.match(doc.attribution, /Google/);
});
