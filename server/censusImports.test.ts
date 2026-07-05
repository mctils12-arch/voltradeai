// Census imports battery (BUILD ORDER 3 #4, key-gated): header-driven
// parse, variant fallback with readable-error logging, key gating, dedup.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  parseImports, fetchImports, archiveImports, gzipOldImportDays,
  censusEnabled, QUERY_VARIANTS,
} from "./censusImports";

// Census array-of-arrays shape (header row + data rows)
const PAYLOAD = [
  ["PORT", "PORT_NAME", "GEN_VAL_MO", "CNT_VAL_MO", "CNT_WGT_MO", "time"],
  ["2704", "LOS ANGELES, CA", "26000000000", "21000000000", "3900000000", "2026-03"],
  ["1401", "NORFOLK, VA", "5100000000", "4400000000", "", "2026-03"],
  ["", "NO PORT", "1", "1", "1", "2026-03"],           // no port -> dropped
  ["9999", "BAD MONTH", "1", "1", "1", "March 2026"],  // malformed month -> dropped
];

test("parseImports: header-driven, '' stays null, malformed rows dropped", () => {
  const obs = parseImports(PAYLOAD, "2026-07-05");
  assert.equal(obs.length, 2);
  const la = obs[0];
  assert.equal(la.port, "2704");
  assert.equal(la.month, "2026-03");
  assert.equal(la.gen_val, 26000000000);
  assert.equal(la.cnt_wgt, 3900000000);
  assert.equal(obs[1].cnt_wgt, null, "empty containerized weight is null, never zero");
  // shuffled column order must parse identically (header-driven contract)
  const shuffled = PAYLOAD.map((r) => [r[5], r[2], r[0], r[1], r[3], r[4]]);
  assert.deepEqual(parseImports(shuffled, "2026-07-05"), obs);
  assert.deepEqual(parseImports(null, "x"), []);
  assert.deepEqual(parseImports([["PORT_NAME"]], "x"), [], "missing PORT/TIME headers = empty");
});

test("censusEnabled gates on the key; fetch no-ops keyless", async () => {
  assert.equal(censusEnabled({} as any), false);
  assert.equal(censusEnabled({ CENSUS_API_KEY: "x" } as any), true);
  assert.deepEqual(await fetchImports(undefined as any, {} as any), [], "keyless fetch returns empty, never throws");
});

test("fetchImports: first variant 400s -> falls back; error body logged, key never in records", async () => {
  const calls: string[] = [];
  const fake = async (url: string) => {
    calls.push(url);
    if (url.includes("CNT_VAL_MO")) {
      return { ok: false, status: 400, text: async () => "error: unknown variable 'CNT_VAL_MO'" };
    }
    return {
      ok: true, status: 200,
      text: async () => JSON.stringify([["PORT", "PORT_NAME", "GEN_VAL_MO", "time"],
                                        ["2704", "LOS ANGELES, CA", "26000000000", "2026-03"]]),
    };
  };
  const obs = await fetchImports(fake as any, { CENSUS_API_KEY: "sekret" } as any, Date.parse("2026-07-05T12:00:00Z"));
  assert.ok(obs.length >= 1, "fallback variant produced records");
  assert.equal(obs[0].cnt_val, null, "fields the fallback variant lacks stay null");
  assert.ok(calls.every((u) => u.includes("key=sekret")), "key rides the request");
  assert.ok(JSON.stringify(obs).indexOf("sekret") === -1, "key never lands in records");
  assert.equal(QUERY_VARIANTS.length, 2);
});

test("archive: dedup by port|month|values (revisions append as new vintages); gz lifecycle", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "census-"));
  const now = Date.parse("2026-07-05T12:00:00Z");
  const obs = parseImports(PAYLOAD, "2026-07-05");
  assert.equal(archiveImports(obs, base, now), 2);
  assert.equal(archiveImports(obs, base, now), 0, "same values never re-archive");
  const revised = [{ ...obs[0], gen_val: 26500000000 }];
  assert.equal(archiveImports(revised, base, now), 1, "a revised value appends as a new vintage");
  assert.equal(gzipOldImportDays(base, now + 3 * 86400_000), 1);
  const day = path.join(base, "censusimports", "2026-07-05.jsonl");
  assert.ok(!fs.existsSync(day) && fs.existsSync(`${day}.gz`));
});
