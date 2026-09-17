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
  backfillImportsFromArchive, refreshImportCache, latestImports,
  _resetImportsCacheForTests, type ImportObs,
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

// ── cold-cache-no-disk-backfill fix (this module joins the thread) ─────────
// FT920 is monthly and the archive is change-only dedup (keyOf includes the
// observed values), so a port|month observation fetched early in the
// pipeline's life sits undisturbed in a months-old daily file — never
// re-written on later days like the higher-frequency modules in this thread.
// A short lookback would miss it; backfillImportsFromArchive's default
// window is intentionally a year+.

function fakeObs(port: string, month: string, rt: string, gen_val: number): ImportObs {
  return { port, port_name: `PORT ${port}`, month, gen_val, cnt_val: null, cnt_wgt: null, rt };
}

test("backfillImportsFromArchive: reconstructs one row per port|month from across widely separated days, no recency filter on the month itself", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "vt-census-backfill-"));
  const now = Date.parse("2026-09-17T12:00:00Z");
  const cDir = path.join(dir, "censusimports");
  fs.mkdirSync(cDir, { recursive: true });
  // LA archived the day this pipeline's key first went live, 74 days back —
  // never re-written since (no revision), unlike the hourly-changing
  // archives the rest of this thread deals with.
  fs.writeFileSync(
    path.join(cDir, "2026-07-05.jsonl"),
    JSON.stringify(fakeObs("2704", "2026-03", "2026-07-05", 26_000_000_000)) + "\n",
  );
  fs.writeFileSync(
    path.join(cDir, "2026-09-10.jsonl"),
    JSON.stringify(fakeObs("1401", "2026-07", "2026-09-10", 5_100_000_000)) + "\n",
  );
  const backfilled = backfillImportsFromArchive(dir, now);
  const ports = backfilled.map((o) => o.port).sort();
  assert.deepEqual(ports, ["1401", "2704"], "both ports returned across a 74-day gap — no short lookback misses the older one");
});

test("backfillImportsFromArchive: a revised value (new rt, same port|month) wins over the earlier vintage", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "vt-census-revision-"));
  const now = Date.parse("2026-09-17T12:00:00Z");
  const cDir = path.join(dir, "censusimports");
  fs.mkdirSync(cDir, { recursive: true });
  fs.writeFileSync(
    path.join(cDir, "2026-07-05.jsonl"),
    JSON.stringify(fakeObs("2704", "2026-03", "2026-07-05", 26_000_000_000)) + "\n",
  );
  fs.writeFileSync(
    path.join(cDir, "2026-08-01.jsonl"),
    JSON.stringify(fakeObs("2704", "2026-03", "2026-08-01", 26_500_000_000)) + "\n",
  );
  const backfilled = backfillImportsFromArchive(dir, now);
  assert.equal(backfilled.length, 1);
  assert.equal(backfilled[0].gen_val, 26_500_000_000, "the later-rt revision wins, not the first-seen vintage");
});

test("refreshImportCache: cold cache (keyless session — fetchImports returns []) backfills from the on-disk archive instead of latching to an empty result", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-census-coldcache-"));
  const prevDataDir = process.env.DATA_DIR;
  process.env.DATA_DIR = base;
  _resetImportsCacheForTests();
  try {
    const now = Date.now();
    const dir = path.join(base, "datacore_archive", "censusimports");
    fs.mkdirSync(dir, { recursive: true });
    const today = new Date(now).toISOString().slice(0, 10);
    fs.writeFileSync(
      path.join(dir, `${today}.jsonl`),
      JSON.stringify(fakeObs("2704", "2026-06", today, 1_000_000)) + "\n",
    );

    assert.equal(latestImports(), null, "cache must still be cold going into this cycle");
    await refreshImportCache(undefined as any, {} as any, now); // no CENSUS_API_KEY -> fetchImports returns []
    const cached = latestImports();
    assert.ok(cached, "cache must be populated, not left null, on a keyless/empty poll");
    assert.equal(cached!.imports.length, 1);
    assert.equal(cached!.imports[0].port, "2704", "backfilled from the archived observation, not fabricated");
  } finally {
    _resetImportsCacheForTests();
    if (prevDataDir === undefined) delete process.env.DATA_DIR; else process.env.DATA_DIR = prevDataDir;
  }
});

test("refreshImportCache: an already-warm cache is left untouched by a transient empty poll (no regression from this fix)", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-census-warm-"));
  const prevDataDir = process.env.DATA_DIR;
  process.env.DATA_DIR = base;
  _resetImportsCacheForTests();
  try {
    const now = Date.now();
    // Warm the cache with a real prior poll first — only one of the four
    // requested months (recentMonths(now, 4)) actually has data, matching a
    // realistic FT920 release pattern; the other three come back empty.
    const releasedMonth = new Date(now);
    releasedMonth.setUTCMonth(releasedMonth.getUTCMonth() - 1);
    const releasedMonthStr = releasedMonth.toISOString().slice(0, 7);
    const fakeFetch = async (url: string) => ({
      ok: true, status: 200,
      text: async () => url.includes(`time=${releasedMonthStr}`)
        ? JSON.stringify([["PORT", "PORT_NAME", "GEN_VAL_MO", "time"],
                          ["2704", "LOS ANGELES, CA", "26000000000", releasedMonthStr]])
        : "",
    });
    await refreshImportCache(fakeFetch as any, { CENSUS_API_KEY: "k" } as any, now);
    assert.equal(latestImports()!.imports.length, 1, "cache warmed by the first poll");

    await refreshImportCache(undefined as any, {} as any, now + 1000); // keyless -> empty poll
    assert.equal(latestImports()!.imports.length, 1, "a subsequent empty poll must not blank an already-warm cache");
  } finally {
    _resetImportsCacheForTests();
    if (prevDataDir === undefined) delete process.env.DATA_DIR; else process.env.DATA_DIR = prevDataDir;
  }
});
