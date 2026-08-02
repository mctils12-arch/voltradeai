import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import os from "os";
import path from "path";
import {
  ELEVATED_FEED_URL,
  GVP_WFS_URL,
  parseElevatedVolcanoes,
  buildGvpWfsUrl,
  parseGvpFeatureCollection,
  joinVolcanoAlerts,
  fetchElevatedVolcanoes,
  fetchGvpCoords,
  fetchVolcanoAlerts,
  archiveVolcanoAlerts,
  gzipOldVolcanoDays,
  bootVolcanoesPoll,
} from "./usgsVolcanoes";

// ROOT VALIDATION LADDER gate 1 (DATA) fixture — captured verbatim from a
// live GET of ELEVATED_FEED_URL during this build (2026-08-02), not hand-built.
const LIVE_ELEVATED_SAMPLE = [
  {
    obs_fullname: "Alaska Volcano Observatory", obs_abbr: "avo",
    volcano_name: "Great Sitkin", vnum: "311120", notice_type_cd: "DU",
    notice_identifier: "DOI-USGS-AVO-2026-08-01T19:08:13+00:00",
    sent_utc: "2026-08-01 19:17:44", sent_unixtime: 1785611864,
    color_code: "ORANGE", alert_level: "WATCH",
    notice_url: "https://volcanoes.usgs.gov/hans-public/notice/DOI-USGS-AVO-2026-08-01T19:08:13+00:00",
    notice_data: "https://volcanoes.usgs.gov/hans-public/api/notice/getNotice/DOI-USGS-AVO-2026-08-01T19:08:13+00:00",
  },
  {
    obs_fullname: "Alaska Volcano Observatory", obs_abbr: "avo",
    volcano_name: "Shishaldin", vnum: "311360", notice_type_cd: "DU",
    notice_identifier: "DOI-USGS-AVO-2026-08-01T19:08:13+00:00",
    sent_utc: "2026-08-01 19:17:44", sent_unixtime: 1785611864,
    color_code: "YELLOW", alert_level: "ADVISORY",
    notice_url: "https://volcanoes.usgs.gov/hans-public/notice/DOI-USGS-AVO-2026-08-01T19:08:13+00:00",
    notice_data: "https://volcanoes.usgs.gov/hans-public/api/notice/getNotice/DOI-USGS-AVO-2026-08-01T19:08:13+00:00",
  },
  {
    obs_fullname: "Hawaiian Volcano Observatory", obs_abbr: "hvo",
    volcano_name: "Kilauea", vnum: "332010", notice_type_cd: "DU",
    notice_identifier: "DOI-USGS-HVO-2026-08-01T18:35:33+00:00",
    sent_utc: "2026-08-01 19:07:02", sent_unixtime: 1785611222,
    color_code: "YELLOW", alert_level: "ADVISORY",
    notice_url: "https://volcanoes.usgs.gov/hans-public/notice/DOI-USGS-HVO-2026-08-01T18:35:33+00:00",
    notice_data: "https://volcanoes.usgs.gov/hans-public/api/notice/getNotice/DOI-USGS-HVO-2026-08-01T18:35:33+00:00",
  },
];

// ROOT VALIDATION LADDER gate 1 (DATA) fixture — captured verbatim from a
// live GET of the GVP WFS CQL_FILTER endpoint during this build (2026-08-02).
const LIVE_GVP_SAMPLE = {
  type: "FeatureCollection",
  features: [
    { type: "Feature", geometry: null, properties: { Volcano_Number: 311120, Volcano_Name: "Great Sitkin", Country: "United States", Latitude: 52.076, Longitude: -176.13, Elevation: 1740 } },
    { type: "Feature", geometry: null, properties: { Volcano_Number: 311360, Volcano_Name: "Shishaldin", Country: "United States", Latitude: 54.756, Longitude: -163.97, Elevation: 2857 } },
    { type: "Feature", geometry: null, properties: { Volcano_Number: 332010, Volcano_Name: "Kilauea", Country: "United States", Latitude: 19.421, Longitude: -155.287, Elevation: 1222 } },
  ],
  totalFeatures: 3, numberMatched: 3, numberReturned: 3,
};

test("parseElevatedVolcanoes: maps real USGS feed shape (vnum/alert_level/color_code preserved)", () => {
  const rows = parseElevatedVolcanoes(LIVE_ELEVATED_SAMPLE, "2026-08-02");
  assert.equal(rows.length, 3);
  assert.equal(rows[0].vnum, "311120");
  assert.equal(rows[0].name, "Great Sitkin");
  assert.equal(rows[0].obs, "avo");
  assert.equal(rows[0].alertLevel, "WATCH");
  assert.equal(rows[0].colorCode, "ORANGE");
  assert.equal(rows[0].noticeId, "DOI-USGS-AVO-2026-08-01T19:08:13+00:00");
  assert.equal(rows[0].rt, "2026-08-02");
});

test("parseElevatedVolcanoes: rows with no vnum are dropped; non-array input returns no rows", () => {
  assert.deepEqual(parseElevatedVolcanoes([{ volcano_name: "no vnum" }], "2026-08-02"), []);
  assert.deepEqual(parseElevatedVolcanoes(null, "2026-08-02"), []);
  assert.deepEqual(parseElevatedVolcanoes({}, "2026-08-02"), []);
});

test("buildGvpWfsUrl: builds a CQL_FILTER IN() query scoped to the given vnums, deduped, non-numeric junk dropped", () => {
  const url = buildGvpWfsUrl(["311120", "311360", "311120", "not-a-vnum"]);
  assert.ok(url!.startsWith(GVP_WFS_URL));
  const decoded = decodeURIComponent(url!.replace(/\+/g, " "));
  assert.ok(decoded.includes("Volcano_Number IN (311120,311360)"), decoded);
});

test("buildGvpWfsUrl: returns null for an empty/all-invalid vnum list — never fetches the whole database", () => {
  assert.equal(buildGvpWfsUrl([]), null);
  assert.equal(buildGvpWfsUrl(["garbage", ""]), null);
});

test("parseGvpFeatureCollection: maps real GVP WFS shape, keyed by Volcano_Number", () => {
  const coords = parseGvpFeatureCollection(LIVE_GVP_SAMPLE);
  assert.equal(coords.size, 3);
  const gs = coords.get("311120")!;
  assert.equal(gs.lat, 52.076);
  assert.equal(gs.lon, -176.13);
  assert.equal(gs.elevationM, 1740);
  assert.equal(gs.country, "United States");
});

test("parseGvpFeatureCollection: rows missing lat/lon are skipped, never fabricated as 0,0", () => {
  const coords = parseGvpFeatureCollection({
    features: [{ properties: { Volcano_Number: 1, Country: "X" } }, { properties: {} }],
  });
  assert.equal(coords.size, 0);
});

test("joinVolcanoAlerts: matched vnums get real coordinates, unmatched vnums keep null (kept, not dropped)", () => {
  const elevated = parseElevatedVolcanoes(LIVE_ELEVATED_SAMPLE, "2026-08-02");
  const coords = parseGvpFeatureCollection(LIVE_GVP_SAMPLE);
  const joined = joinVolcanoAlerts(elevated.concat([{ ...elevated[0], vnum: "999999" }]), coords);
  assert.equal(joined[0].lat, 52.076);
  assert.equal(joined[0].lon, -176.13);
  assert.equal(joined[0].elevationM, 1740);
  const unmatched = joined.find((j) => j.vnum === "999999")!;
  assert.equal(unmatched.lat, null);
  assert.equal(unmatched.lon, null);
  assert.equal(unmatched.elevationM, null);
});

test("fetchElevatedVolcanoes: hits ELEVATED_FEED_URL with a UA header and parses the response", async () => {
  let calledUrl = "";
  let calledInit: any = null;
  const fetchImpl = async (url: string, init: any) => {
    calledUrl = url; calledInit = init;
    return { ok: true, status: 200, text: async () => JSON.stringify(LIVE_ELEVATED_SAMPLE) };
  };
  const rows = await fetchElevatedVolcanoes(fetchImpl as any, Date.parse("2026-08-02T00:00:00Z"));
  assert.equal(calledUrl, ELEVATED_FEED_URL);
  assert.ok(calledInit.headers["User-Agent"]);
  assert.equal(rows.length, 3);
});

test("fetchElevatedVolcanoes: throws with the HTTP status on a non-ok response", async () => {
  const fetchImpl = async () => ({ ok: false, status: 503, text: async () => "" });
  await assert.rejects(() => fetchElevatedVolcanoes(fetchImpl as any), /503/);
});

test("fetchGvpCoords: skips the network call entirely for an empty vnum list", async () => {
  let called = false;
  const fetchImpl = async () => { called = true; return { ok: true, status: 200, text: async () => "{}" }; };
  const coords = await fetchGvpCoords([], fetchImpl as any);
  assert.equal(called, false);
  assert.equal(coords.size, 0);
});

test("fetchGvpCoords: hits GVP_WFS_URL and parses the response", async () => {
  let calledUrl = "";
  const fetchImpl = async (url: string) => { calledUrl = url; return { ok: true, status: 200, text: async () => JSON.stringify(LIVE_GVP_SAMPLE) }; };
  const coords = await fetchGvpCoords(["311120"], fetchImpl as any);
  assert.ok(calledUrl.startsWith(GVP_WFS_URL));
  assert.equal(coords.size, 3);
});

test("fetchGvpCoords: throws with the HTTP status on a non-ok response", async () => {
  const fetchImpl = async () => ({ ok: false, status: 500, text: async () => "" });
  await assert.rejects(() => fetchGvpCoords(["1"], fetchImpl as any), /500/);
});

test("fetchVolcanoAlerts: joins elevated + coords end-to-end via two fetch calls", async () => {
  const urls: string[] = [];
  const fetchImpl = async (url: string) => {
    urls.push(url);
    if (url === ELEVATED_FEED_URL) return { ok: true, status: 200, text: async () => JSON.stringify(LIVE_ELEVATED_SAMPLE) };
    return { ok: true, status: 200, text: async () => JSON.stringify(LIVE_GVP_SAMPLE) };
  };
  const alerts = await fetchVolcanoAlerts(fetchImpl as any, Date.parse("2026-08-02T12:00:00Z"));
  assert.equal(urls.length, 2);
  assert.equal(alerts.length, 3);
  assert.equal(alerts.find((a) => a.vnum === "332010")!.lat, 19.421);
});

test("fetchVolcanoAlerts: a vnum already coordinate-cached from a prior call skips a repeat GVP request", async () => {
  const uniqueVnum = "555001";
  const firstElevated = [{ ...LIVE_ELEVATED_SAMPLE[0], vnum: uniqueVnum, notice_identifier: "cache-test-1" }];
  const gvpUrls: string[] = [];
  const fetchImpl1 = async (url: string) => {
    if (url === ELEVATED_FEED_URL) return { ok: true, status: 200, text: async () => JSON.stringify(firstElevated) };
    gvpUrls.push(url);
    return { ok: true, status: 200, text: async () => JSON.stringify({ features: [{ properties: { Volcano_Number: uniqueVnum, Latitude: 1, Longitude: 2, Elevation: 3, Country: "Testland" } }] }) };
  };
  await fetchVolcanoAlerts(fetchImpl1 as any);
  assert.equal(gvpUrls.length, 1);
  // second poll, same vnum, different notice (alert level updated) — coordinate lookup must NOT repeat
  const secondElevated = [{ ...LIVE_ELEVATED_SAMPLE[0], vnum: uniqueVnum, notice_identifier: "cache-test-2", alert_level: "WARNING" }];
  const fetchImpl2 = async (url: string) => {
    if (url === ELEVATED_FEED_URL) return { ok: true, status: 200, text: async () => JSON.stringify(secondElevated) };
    gvpUrls.push(url);
    return { ok: true, status: 200, text: async () => "{}" };
  };
  const alerts2 = await fetchVolcanoAlerts(fetchImpl2 as any);
  assert.equal(gvpUrls.length, 1, "no second GVP request for an already-cached vnum");
  assert.equal(alerts2[0].lat, 1);
  assert.equal(alerts2[0].alertLevel, "WARNING");
});

test("archiveVolcanoAlerts: fresh alerts write a day-file, dedup by vnum+noticeId suppresses re-archiving unchanged rows", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "vt-volcanoes-"));
  const now = Date.parse("2026-08-02T12:00:00Z");
  const elevated = parseElevatedVolcanoes(LIVE_ELEVATED_SAMPLE, "2026-08-02");
  const coords = parseGvpFeatureCollection(LIVE_GVP_SAMPLE);
  const alerts = joinVolcanoAlerts(elevated, coords);
  const n1 = archiveVolcanoAlerts(alerts, dir, now);
  assert.equal(n1, 3);
  const fp = path.join(dir, "volcanoes", "2026-08-02.jsonl");
  assert.ok(fs.existsSync(fp));
  assert.equal(fs.readFileSync(fp, "utf8").trim().split("\n").length, 3);
  const n2 = archiveVolcanoAlerts(alerts, dir, now + 60_000);
  assert.equal(n2, 0);
  assert.equal(fs.readFileSync(fp, "utf8").trim().split("\n").length, 3);
});

test("archiveVolcanoAlerts: a new notice_identifier for a known vnum re-archives it (alert level changed)", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "vt-volcanoes-"));
  const now = Date.parse("2026-08-02T12:00:00Z");
  const base = { vnum: "vt_test_1", name: "Test Peak", obs: "avo", obsFullname: null, alertLevel: "ADVISORY", colorCode: "YELLOW", noticeId: "n1", noticeUrl: null, sentUtc: null, sentUnixtime: 1, lat: 1, lon: 2, elevationM: 3, country: "Testland", rt: "2026-08-02" };
  archiveVolcanoAlerts([base], dir, now);
  const revised = { ...base, noticeId: "n2", alertLevel: "WATCH", colorCode: "ORANGE" };
  const n2 = archiveVolcanoAlerts([revised], dir, now + 60_000);
  assert.equal(n2, 1);
  const fp = path.join(dir, "volcanoes", "2026-08-02.jsonl");
  const lines = fs.readFileSync(fp, "utf8").trim().split("\n").map((l) => JSON.parse(l));
  assert.equal(lines.length, 2); // both the original and the update, append-only
  assert.equal(lines[1].alertLevel, "WATCH");
});

test("gzipOldVolcanoDays: gzips day-files older than 2 days, leaves recent ones alone", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "vt-volcanoes-"));
  const now = Date.parse("2026-08-02T12:00:00Z");
  const vDir = path.join(dir, "volcanoes");
  fs.mkdirSync(vDir, { recursive: true });
  fs.writeFileSync(path.join(vDir, "2026-07-26.jsonl"), '{"vnum":"old"}\n');
  fs.writeFileSync(path.join(vDir, "2026-08-02.jsonl"), '{"vnum":"new"}\n');
  const n = gzipOldVolcanoDays(dir, now);
  assert.equal(n, 1);
  assert.ok(fs.existsSync(path.join(vDir, "2026-07-26.jsonl.gz")));
  assert.ok(!fs.existsSync(path.join(vDir, "2026-07-26.jsonl")));
  assert.ok(fs.existsSync(path.join(vDir, "2026-08-02.jsonl")));
});

test("bootVolcanoesPoll: keyless — starts polling unconditionally, idempotent across repeat calls", () => {
  assert.doesNotThrow(() => {
    bootVolcanoesPoll(3600_000);
    bootVolcanoesPoll(3600_000); // second call is a no-op (module-level `polling` guard)
  });
});
