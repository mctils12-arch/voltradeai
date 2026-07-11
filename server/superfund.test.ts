// superfund battery — EPA NPL sites parse + the data-quality gate in action.
import { test } from "node:test";
import assert from "node:assert/strict";
import { siteFromFeature, parseSites, pageUrl, fetchSuperfund, NPL_STATUS } from "./superfund";

const feat = (props: any) => ({ type: "Feature", geometry: { type: "Point", coordinates: [props.Longitude, props.Latitude] }, properties: props });
const good = {
  Site_Name: "Kellogg-Deering Well Field", Site_Score: 39.92, Site_EPA_ID: "CTD980670814",
  State: "Connecticut", City: "Norwalk", County: "Fairfield", Status: "NPL Site",
  Longitude: -73.43195, Latitude: 41.13055, Listing_Date: "09/21/1984",
};

test("siteFromFeature: valid site parses with all fields", () => {
  const { site, issues } = siteFromFeature(feat(good));
  assert.equal(issues.length, 0);
  assert.equal(site!.name, "Kellogg-Deering Well Field");
  assert.equal(site!.hrs_score, 39.92);
  assert.equal(site!.status, "NPL Site");
  assert.equal(site!.state, "Connecticut");
  assert.ok(Math.abs(site!.lat - 41.13055) < 1e-4);
});

test("data-quality gate: bad records are quarantined, not parsed", () => {
  // missing name
  assert.ok(siteFromFeature(feat({ ...good, Site_Name: "" })).issues.some((i) => i.rule === "required"));
  // null-island coords (failed geocode)
  assert.ok(siteFromFeature(feat({ ...good, Longitude: 0, Latitude: 0 })).issues.some((i) => i.rule === "coordValid"));
  // out-of-range coords
  assert.ok(siteFromFeature(feat({ ...good, Latitude: 999 })).issues.some((i) => i.rule === "coordValid"));
  // HRS score out of 0-100 range
  assert.ok(siteFromFeature(feat({ ...good, Site_Score: 250 })).issues.some((i) => i.rule === "max"));
  assert.ok(siteFromFeature(feat({ ...good, Site_Score: -5 })).issues.some((i) => i.rule === "min"));
  // unknown status
  assert.ok(siteFromFeature(feat({ ...good, Status: "Made Up Status" })).issues.some((i) => i.field === "Status"));
});

test("siteFromFeature: missing HRS score is allowed (null), not a failure", () => {
  const { site, issues } = siteFromFeature(feat({ ...good, Site_Score: null }));
  assert.equal(issues.length, 0);
  assert.equal(site!.hrs_score, null);
});

test("parseSites: splits a FeatureCollection into clean sites + quarantined count", () => {
  const fc = { features: [
    feat(good),
    feat({ ...good, Site_Name: "" }),          // quarantined
    feat({ ...good, Longitude: 0, Latitude: 0 }), // quarantined
    feat({ ...good, Site_Name: "Second Good Site" }),
  ] };
  const { sites, suspect } = parseSites(fc);
  assert.equal(sites.length, 2);
  assert.equal(suspect, 2);
});

test("pageUrl: encodes the geojson query with paging", () => {
  const u = pageUrl(1000);
  assert.match(u, /resultOffset=1000/);
  assert.match(u, /f=geojson/);
  assert.match(u, /FeatureServer\/0\/query/);
});

test("fetchSuperfund: paginates and stops on a short page; validates each page", async () => {
  const p1 = { features: Array.from({ length: 1000 }, (_, i) => feat({ ...good, Site_Name: `S${i}` })) };
  const p2 = { features: [feat({ ...good, Site_Name: "last" }), feat({ ...good, Site_Name: "", Longitude: 0, Latitude: 0 })] };
  let call = 0;
  const spy = async () => { call++; return { ok: true, status: 200, text: async () => JSON.stringify(call === 1 ? p1 : p2) }; };
  const { sites, suspect } = await fetchSuperfund(spy as any);
  assert.equal(call, 2, "second page fetched, then short page stops paging");
  assert.equal(sites.length, 1001);
  assert.equal(suspect, 1, "the null-island row on page 2 is quarantined");
});

test("NPL_STATUS covers the common EPA statuses", () => {
  assert.ok(NPL_STATUS.includes("NPL Site"));
  assert.ok(NPL_STATUS.includes("Deleted NPL Site"));
});
