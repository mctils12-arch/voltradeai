// waterViolators battery — ECHO CWA chronic violators: CSV parse + the
// data-quality gate + the two-step qid->download fetch.
import { test } from "node:test";
import assert from "node:assert/strict";
import { parseCsvLine, violatorFromRow, parseDownloadCsv, fetchWaterViolators, facilitiesUrl, downloadUrl } from "./waterViolators";

test("parseCsvLine: quoted fields, embedded commas + quotes", () => {
  assert.deepEqual(parseCsvLine('"A, B","C""D",plain,'), ["A, B", 'C"D', "plain", ""]);
});

const HEADER = '"CWPName","SourceID","CWPCity","CWPState","FacLat","FacLong","CWPPermitTypeDesc","CWPSNCStatus","CWPQtrsWithNC","CWPFormalEaCnt"';
const goodRow = '"ACME OUTFALL, PLANT 2","TX0012345","VICTORIA","TX",28.891,-96.823,"NPDES Individual Permit","Effluent - Monthly Average Limit",9,2';

test("violatorFromRow: clean row parses fully", () => {
  const rec: Record<string, string> = {};
  parseCsvLine(HEADER).forEach((h, i) => { rec[h] = parseCsvLine(goodRow)[i] ?? ""; });
  const { v, issues } = violatorFromRow(rec);
  assert.equal(issues.length, 0);
  assert.equal(v!.name, "ACME OUTFALL, PLANT 2");
  assert.equal(v!.qtrs, 9);
  assert.equal(v!.snc, "Effluent - Monthly Average Limit");
  assert.equal(v!.actions, 2);
});

test("data-quality gate: bad rows quarantined (coords, name, quarters bounds)", () => {
  const base: Record<string, string> = {};
  parseCsvLine(HEADER).forEach((h, i) => { base[h] = parseCsvLine(goodRow)[i] ?? ""; });
  assert.ok(violatorFromRow({ ...base, FacLat: "0", FacLong: "0" }).issues.some((i) => i.rule === "coordValid"));
  assert.ok(violatorFromRow({ ...base, FacLat: "", FacLong: "" }).issues.some((i) => i.rule === "coordValid"));
  assert.ok(violatorFromRow({ ...base, CWPName: "" }).issues.some((i) => i.rule === "required"));
  assert.ok(violatorFromRow({ ...base, CWPQtrsWithNC: "13" }).issues.some((i) => i.rule === "max"));
  assert.ok(violatorFromRow({ ...base, CWPQtrsWithNC: "-1" }).issues.some((i) => i.rule === "min"));
});

test("parseDownloadCsv: splits clean vs suspect, header-driven", () => {
  const csv = [HEADER, goodRow, '"NO COORDS","X1","C","ST",,,"P","",9,'].join("\n");
  const { violators, suspect } = parseDownloadCsv(csv);
  assert.equal(violators.length, 1);
  assert.equal(suspect, 1);
});

test("fetchWaterViolators: two-step qid -> csv download", async () => {
  const calls: string[] = [];
  const spy = async (url: string) => {
    calls.push(url);
    if (url.includes("get_facilities")) return { ok: true, status: 200, text: async () => JSON.stringify({ Results: { QueryID: "814", QueryRows: "2" } }) };
    return { ok: true, status: 200, text: async () => [HEADER, goodRow].join("\n") };
  };
  const { violators, suspect } = await fetchWaterViolators(spy as any);
  assert.equal(calls.length, 2);
  assert.match(calls[0], /p_qiv=GT8/);
  assert.match(calls[1], /qid=814/);
  assert.equal(violators.length, 1);
  assert.equal(suspect, 0);
});

test("urls: chronic-violator filter + column selection stay pinned", () => {
  assert.match(facilitiesUrl(), /p_act=Y&p_qiv=GT8/);
  assert.match(downloadUrl("9"), /qcolumns=1,2,4,5,24,25,54,98,101,114/);
});
