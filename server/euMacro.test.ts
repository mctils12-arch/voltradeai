// euMacro tests — European macro cluster (census build #7). Fixtures are
// shaped EXACTLY like the live-verified responses (2026-07-07 workup):
// ECB SDMX-CSV with dataflow-varying columns + ISO-week ILM periods,
// Eurostat JSON-stat sparse values + status flags + the silent-empty
// trap, Bundesbank string values + null placeholder rows.
import { test, beforeEach } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import path from "path";
import os from "os";
import {
  parseEcbCsv, parseEurostatJsonStat, parseBbkJson, isoWeekFriday,
  archiveEuObs, refreshEuMacro, latestEuMacro, _resetEuMacroForTests,
  EU_SERIES,
} from "./euMacro";

function tmp(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), "eumacro-"));
}

beforeEach(() => _resetEuMacroForTests());

test("isoWeekFriday: ECB weekly statement anchors (verified: 2026-W26 relates to Friday 2026-06-26)", () => {
  assert.equal(isoWeekFriday(2026, 26), "2026-06-26");
  assert.equal(isoWeekFriday(2026, 1), "2026-01-02");
  assert.equal(isoWeekFriday(2020, 53), "2021-01-01", "ISO week-53 year (2020 has one) spills into January correctly");
});

test("parseEcbCsv: header-name parse (positions vary per dataflow), weekly periods normalized to Friday", () => {
  // EXR-like: TIME_PERIOD/OBS_VALUE buried among other columns
  const daily = "KEY,FREQ,CURRENCY,TIME_PERIOD,OBS_VALUE,OBS_STATUS\n" +
    "EXR.D.USD.EUR.SP00.A,D,USD,2026-07-03,1.1448,A\n" +
    "EXR.D.USD.EUR.SP00.A,D,USD,2026-07-06,1.1415,A\n";
  const d = parseEcbCsv("ECB_EURUSD", daily, "2026-07-07");
  assert.deepEqual(d.map((o) => [o.d, o.v]), [["2026-07-03", 1.1448], ["2026-07-06", 1.1415]]);
  assert.equal(d[0].rt, "2026-07-07");

  // ILM-like: ISO-week TIME_PERIOD, different column count
  const weekly = "KEY,FREQ,REF_AREA,TIME_PERIOD,OBS_VALUE\n" +
    "ILM.W.U2.C.T000000.Z5.Z01,W,U2,2026-W26,6117260\n";
  const w = parseEcbCsv("ECB_BS_TOTAL", weekly, "2026-07-07");
  assert.equal(w[0].d, "2026-06-26", "week normalized to the Friday reference date");
  assert.equal(w[0].wk, "2026-W26", "raw ISO week preserved");
  assert.equal(w[0].v, 6117260);

  // header drift alarms with empty result, never a bad parse
  assert.deepEqual(parseEcbCsv("ECB_EURUSD", "SOMETHING,ELSE\n1,2\n", "rt"), []);
});

test("parseEurostatJsonStat: sparse value{} skipped honestly, status flags carried, silent-empty returns null", () => {
  const json = {
    version: "2.0",
    value: { "0": 97.5, "1": 97.7, "3": 98.3 }, // index 2 sparse (no data yet)
    status: { "3": "i" },
    dimension: { time: { category: { index: { "2026-01": 0, "2026-02": 1, "2026-03": 2, "2026-04": 3 } } } },
  };
  const obs = parseEurostatJsonStat("EU_INDPROD", json, "2026-07-07")!;
  assert.deepEqual(obs.map((o) => o.d), ["2026-01-01", "2026-02-01", "2026-04-01"], "sparse index 2 skipped, never zeroed");
  assert.equal(obs[2].flag, "i", "imputed flag carried into the archive");
  assert.equal(obs[0].flag, undefined);

  // the verified silent-empty 200 (bad dimension value) → null, caller alarms
  assert.equal(parseEurostatJsonStat("EU_INDPROD", { value: {}, dimension: { time: { category: { index: {} } } } }, "rt"), null);
});

test("parseBbkJson: string values parsed, null placeholder rows skipped (weekends), zero-obs 200 yields []", () => {
  const json = {
    data: {
      dataSets: [{ series: { "0:0:0:0:0:0:0:0:0:0:0:0:0:0:0": { observations: { "0": [null, 1], "1": ["2.98", 0, 1], "2": ["3.01", 0, 1] } } } }],
      structure: { dimensions: { observation: [{ values: [{ id: "2026-07-04" }, { id: "2026-07-06" }, { id: "2026-07-07" }] }] } },
    },
  };
  const obs = parseBbkJson("DE_BUND10Y", json, "2026-07-07");
  assert.deepEqual(obs.map((o) => [o.d, o.v]), [["2026-07-06", 2.98], ["2026-07-07", 3.01]], "null day skipped, strings parsed");
  assert.deepEqual(parseBbkJson("DE_BUND10Y", { data: { dataSets: [{ series: {} }], structure: { dimensions: { observation: [{ values: [] }] } } } }, "rt"), []);
});

test("archive: vintage mechanism — value change appends, repeat is dropped, revert re-appends (fredMacro semantics)", () => {
  const base = tmp();
  const NOW = Date.parse("2026-07-07T12:00:00Z");
  assert.equal(archiveEuObs([{ s: "ECB_EURUSD", d: "2026-07-06", v: 1.1415, rt: "2026-07-07" }], base, NOW), 1);
  assert.equal(archiveEuObs([{ s: "ECB_EURUSD", d: "2026-07-06", v: 1.1415, rt: "2026-07-07" }], base, NOW), 0, "same value dropped");
  assert.equal(archiveEuObs([{ s: "ECB_EURUSD", d: "2026-07-06", v: 1.1420, rt: "2026-07-07" }], base, NOW), 1, "revision appends");
  assert.equal(archiveEuObs([{ s: "ECB_EURUSD", d: "2026-07-06", v: 1.1415, rt: "2026-07-07" }], base, NOW), 1, "REVERT is a vintage event");
  const lines = fs.readFileSync(path.join(base, "eumacro", "2026-07-07.jsonl"), "utf8").trim().split("\n");
  assert.equal(lines.length, 3);
});

test("refresh end-to-end: all five series fetched from fixtures, cache built, silent-empty not cached as truth", async () => {
  const base = tmp();
  const fetchImpl = async (url: string) => {
    let body = "";
    if (url.includes("EXR/D.USD")) body = "KEY,TIME_PERIOD,OBS_VALUE\nx,2026-07-06,1.1415\n";
    else if (url.includes("EST/B.EU000")) body = "KEY,TIME_PERIOD,OBS_VALUE\nx,2026-07-03,2.183\n";
    else if (url.includes("ILM/W.U2")) body = "KEY,TIME_PERIOD,OBS_VALUE\nx,2026-W26,6117260\n";
    else if (url.includes("eurostat")) body = JSON.stringify({
      value: { "0": 98.3 }, status: {},
      dimension: { time: { category: { index: { "2026-05": 0 } } } },
    });
    else if (url.includes("bundesbank")) body = JSON.stringify({
      data: {
        dataSets: [{ series: { k: { observations: { "0": ["2.98", 0] } } } }],
        structure: { dimensions: { observation: [{ values: [{ id: "2026-07-06" }] }] } },
      },
    });
    return { ok: true, status: 200, text: async () => body };
  };
  await refreshEuMacro(fetchImpl as any, Date.parse("2026-07-07T12:00:00Z"), base, 0);
  const hit = latestEuMacro()!;
  assert.equal(hit.series.length, EU_SERIES.length);
  const by = Object.fromEntries(hit.series.map((s) => [s.key, s]));
  assert.equal(by.ECB_EURUSD.latest!.v, 1.1415);
  assert.equal(by.ECB_BS_TOTAL.latest!.d, "2026-06-26");
  assert.equal(by.EU_INDPROD.latest!.v, 98.3);
  assert.equal(by.DE_BUND10Y.latest!.v, 2.98);
  assert.ok(fs.existsSync(path.join(base, "eumacro", "2026-07-07.jsonl")), "vintage archive written");
  for (const s of hit.series) assert.ok(s.attribution.length > 0, `${s.key} carries its attribution`);
});

test("refresh survives total transport failure with honest empty snapshots (no fabrication)", async () => {
  const dead = async () => { throw new Error("ECONNRESET"); };
  await refreshEuMacro(dead as any, Date.parse("2026-07-07T12:00:00Z"), tmp(), 0);
  const hit = latestEuMacro()!;
  assert.ok(hit.series.every((s) => s.latest === null && s.history.length === 0));
});
