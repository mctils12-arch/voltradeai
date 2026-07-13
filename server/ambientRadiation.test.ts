// ambientRadiation parser tests — fixtures mirror the real payload shapes
// verified live 2026-07-12 (BfS GeoJSON, HC ESRI GeoJSON, FMI BsWfs XML,
// RadNet CSV incl. the Denver dose-column-empty case). No network.
import { test } from "node:test";
import assert from "node:assert/strict";
import { parseBfs, parseHealthCanada, parseFmi, parseRadnetCsv, loadRadnetCities, friendlyHcName } from "./ambientRadiation";

test("parseBfs: values pass, bad coords / negative values quarantined", () => {
  const s = parseBfs({
    features: [
      { geometry: { coordinates: [13.4, 52.5] }, properties: { kenn: "DEZ1", name: "Berlin", value: 0.076, end_measure: "2026-07-12T19:00:00Z" } },
      { geometry: { coordinates: [0, 0] }, properties: { kenn: "DEZ2", name: "NullIsland", value: 0.1 } },
      { geometry: { coordinates: [10.0, 50.0] }, properties: { kenn: "DEZ3", name: "Bad", value: -1 } },
    ],
  });
  assert.equal(s.length, 1);
  assert.equal(s[0].network, "bfs-de");
  assert.equal(s[0].unit, "uSv/h");
  assert.equal(s[0].value, 0.076);
  assert.equal(s[0].time, "2026-07-12T19:00:00Z");
});

test("parseHealthCanada: nSv/h converts to uSv/h (unit arithmetic only)", () => {
  const s = parseHealthCanada({
    features: [{ geometry: { coordinates: [-75.7, 45.4] }, properties: { OBJECTID: 7, location_name: "Ottawa", latest_value_nSvHr: 45, time: 1783971900000 } }],
  });
  assert.equal(s.length, 1);
  assert.equal(s[0].value, 0.045);
  assert.ok(s[0].time?.startsWith("2026-"));
});

test("parseFmi: keeps only DR_PT10M_avg dose elements", () => {
  const xml = `
  <BsWfs:BsWfsElement><BsWfs:Location><gml:Point><gml:pos>60.17 24.94</gml:pos></gml:Point></BsWfs:Location>
    <BsWfs:Time>2026-07-12T19:30:00Z</BsWfs:Time><BsWfs:ParameterName>DR_PT10M_avg</BsWfs:ParameterName><BsWfs:ParameterValue>0.11</BsWfs:ParameterValue></BsWfs:BsWfsElement>
  <BsWfs:BsWfsElement><BsWfs:Location><gml:Point><gml:pos>60.17 24.94</gml:pos></gml:Point></BsWfs:Location>
    <BsWfs:Time>2026-07-12T19:30:00Z</BsWfs:Time><BsWfs:ParameterName>DRS1_PT10M_avg</BsWfs:ParameterName><BsWfs:ParameterValue>0.09</BsWfs:ParameterValue></BsWfs:BsWfsElement>`;
  const s = parseFmi(xml);
  assert.equal(s.length, 1);
  assert.equal(s[0].value, 0.11);
  assert.equal(s[0].lat, 60.17);
  assert.equal(s[0].lon, 24.94);
});

const RADNET_HEADER = "LOCATION_NAME,SAMPLE COLLECTION TIME,DOSE EQUIVALENT RATE (nSv/h),GAMMA COUNT RATE R02 (CPM),GAMMA COUNT RATE R03 (CPM),STATUS";

test("parseRadnetCsv: dose station serves uSv/h from latest APPROVED row", () => {
  const csv = [RADNET_HEADER,
    "DC: WASHINGTON,07/12/2026 10:00:00,26.00,2000.00,1000.00,APPROVED",
    "DC: WASHINGTON,07/12/2026 12:00:00,27.00,2100.00,1100.00,APPROVED",
    "DC: WASHINGTON,07/12/2026 13:00:00,99.00,9999.00,9999.00,PENDING",
  ].join("\n");
  const s = parseRadnetCsv(csv, "us-DC/WASHINGTON", "Washington, DC", 38.9, -77.0);
  assert.ok(s);
  assert.equal(s!.unit, "uSv/h");
  assert.equal(s!.value, 0.027, "latest APPROVED row wins; PENDING skipped");
  assert.equal(s!.time, "2026-07-12T12:00:00Z");
  assert.equal(s!.approx, true, "RadNet locations are city-approximate and must say so");
});

test("parseRadnetCsv: dose-blank station (Denver case) falls back to CPM window sum", () => {
  const csv = [RADNET_HEADER,
    "CO: DENVER,07/12/2026 19:03:00,,2220.00,1223.00,APPROVED",
  ].join("\n");
  const s = parseRadnetCsv(csv, "us-CO/DENVER", "Denver, CO", 39.7, -104.9);
  assert.ok(s);
  assert.equal(s!.unit, "cpm", "never convert CPM to dose — serve CPM as CPM");
  assert.equal(s!.value, 3443);
});

test("parseRadnetCsv: no APPROVED rows -> null (never serve unvetted rows)", () => {
  const csv = [RADNET_HEADER, "CO: DENVER,07/12/2026 19:03:00,,2220.00,1223.00,PENDING"].join("\n");
  assert.equal(parseRadnetCsv(csv, "x", "x", 39.7, -104.9), null);
});

test("radnet city table loads with valid coordinates for every station", () => {
  const cities = loadRadnetCities();
  assert.ok(cities.length >= 70, `expected ~76 cities, got ${cities.length}`);
  for (const c of cities) {
    assert.ok(Number.isFinite(c.lat) && Number.isFinite(c.lon), c.key);
    assert.ok(Math.abs(c.lat) <= 72 && Math.abs(c.lon) <= 180, c.key);
    assert.ok(c.st.length === 2 && c.city.length >= 3, c.key);
    assert.ok(c.rkm >= 1 && c.rkm <= 40, `${c.key} rkm ${c.rkm} — city-area radius out of sane range`);
  }
});

test("friendlyHcName: coded HC names render readable, real names pass through", () => {
  assert.equal(
    friendlyHcName("7000_15_52.1347_-106.6317", 52.1347, -106.6317),
    "Gamma monitor 52.13\u00b0N 106.63\u00b0W (station 7000-15)");
  assert.equal(friendlyHcName("Ottawa", 45.4, -75.7), "Ottawa", "a future friendly name must pass through");
  assert.equal(friendlyHcName("", 45.4, -75.7), "FPSN station");
});

test("parseHealthCanada: coded names come out readable", () => {
  const s = parseHealthCanada({
    features: [{ geometry: { coordinates: [-106.6317, 52.1347] }, properties: { OBJECTID: 9, location_name: "7000_15_52.1347_-106.6317", latest_value_nSvHr: 41, time: 1783971900000 } }],
  });
  assert.match(s[0].name, /^Gamma monitor 52\.13/);
});
