// lazyAssets tests — [REPAIR A7 2026-11-03] the large datacore JSON tables
// and sharp/libvips load on first use instead of being inlined into
// dist/index.cjs and parsed/held from boot. These pin (1) parity: every lazy
// table parses deep-equal to the static import it replaced, against the real
// committed files; (2) packaging: every registry entry is staged into dist/
// by script/build.ts and nothing still imports it statically (which would
// hold it twice); (3) resolution: working tree, then dist/ under cwd (the
// production image), then the module's own tree (a process whose cwd is not
// the app root), and an honest throw for a file that exists nowhere; (4) the
// VOLTRADE_LAZY_ASSETS flag — default lazy loads nothing, "0" preloads
// everything through the same loaders; (5) snapshotStreams' build stamps
// read the tables WITHOUT retaining them (the whole point for the inventory's
// boot refresh); (6) sharp loads once, on demand, and encodes.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import os from "os";
import path from "path";
import { fileURLToPath } from "node:url";
import {
  LAZY_TABLES, LAZY_ASSETS_ENV, lazyTableList, lazyJsonTable, resolveLazyAssetPath,
  lazyAssetsEnabled, preloadLazyAssets, getSharp, sharpLoaded,
} from "./lazyAssets";
import { snapshotStreams, stamp, _resetSnapshotStreamsForTests } from "./snapshotStreams";

// What esbuild inlined until A7: the static imports. Each lazy table must
// parse to exactly these objects.
import usPowerPlants from "../datacore/powerplants/us_power_plants.json";
import entitySpine from "../datacore/aircraft/entity_spine.json";
import cpcDegreeDays from "../datacore/cpc/degree_days.json";
import militaryInstallations from "../datacore/military_installations.json";
import quakeHistory from "../datacore/quake_history.json";
import nuclearTests from "../datacore/nuclear_tests.json";
import boundariesAdmin0 from "../datacore/boundaries/ne_110m_admin0.json";
import boundariesAdmin1Lines from "../datacore/boundaries/ne_50m_admin1_lines.json";
import pfasUcmr5 from "../datacore/pfas/ucmr5_detections.json";
import cdcCountyRates from "../datacore/cdc_cancer/county_rates.json";
import cdcCountiesTopo from "../datacore/cdc_cancer/counties-10m.topo.json";
import jodiPrimaryStocks from "../datacore/jodi/primary_stocks.json";
import railEp724 from "../datacore/rail/ep724_carloads.json";
import eiaWeekly from "../datacore/eia/weekly_series.json";
import signalLadder from "../datacore/signal_ladder.json";
import unComtrade from "../datacore/un_comtrade/bilateral_trade.json";

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, "..");

const STATIC: Record<keyof typeof LAZY_TABLES, unknown> = {
  usPowerPlants, entitySpine, cpcDegreeDays, militaryInstallations, quakeHistory, nuclearTests,
  boundariesAdmin0, boundariesAdmin1Lines, pfasUcmr5, cdcCountyRates, cdcCountiesTopo,
  jodiPrimaryStocks, railEp724, eiaWeekly, signalLadder,
};

function tmp(prefix: string): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), prefix));
}

test("registry: 15 tables, every one a datacore/ JSON file that exists in the repo", () => {
  const list = lazyTableList();
  assert.equal(list.length, 15);
  assert.equal(new Set(list.map((t) => t.rel)).size, list.length, "no two entries may point at one file");
  for (const t of list) {
    assert.match(t.rel, /^datacore\/.+\.json$/, `${t.rel} must be a repo-relative datacore JSON path`);
    assert.ok(fs.existsSync(path.join(repoRoot, t.rel)), `${t.rel} missing from the repo`);
  }
  assert.deepEqual(Object.keys(STATIC).sort(), Object.keys(LAZY_TABLES).sort(), "this test must import every registry table statically");
});

test("PARITY: every lazy table parses deep-equal to the static import it replaced, and is memoized", () => {
  for (const [name, table] of Object.entries(LAZY_TABLES) as Array<[keyof typeof LAZY_TABLES, (typeof LAZY_TABLES)[keyof typeof LAZY_TABLES]]>) {
    table.reset();
    assert.equal(table.loaded(), false);
    const got = table.get();
    assert.deepStrictEqual(got, STATIC[name], `${name} (${table.rel}) differs from the static import`);
    assert.equal(table.loaded(), true);
    assert.equal(table.bytes(), fs.statSync(path.join(repoRoot, table.rel)).size, `${name}: bytes() must be the file's size`);
    assert.strictEqual(table.get(), got, `${name}: the second get() must return the memoized object`);
    assert.strictEqual(table.peek(), got, `${name}: peek() returns the memoized object once loaded`);
  }
});

test("RATCHET: script/build.ts stages every LAZY_TABLES entry into dist/ (the frozen Dockerfile ships dist/ only)", () => {
  const src = fs.readFileSync(path.join(repoRoot, "script", "build.ts"), "utf8");
  for (const t of lazyTableList()) {
    assert.ok(src.includes(`cp("${t.rel}", "dist/${t.rel}"`),
      `script/build.ts must copy ${t.rel} into dist/ — a lazy table that is not staged throws on its first request in prod`);
  }
});

test("RATCHET: no non-test server module still imports a lazy table statically (it would be inlined AND read from disk)", () => {
  const serverDir = path.join(repoRoot, "server");
  const rels = lazyTableList().map((t) => t.rel);
  const esc = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  for (const f of fs.readdirSync(serverDir)) {
    if (!f.endsWith(".ts") || f.endsWith(".test.ts")) continue;
    const body = fs.readFileSync(path.join(serverDir, f), "utf8");
    for (const rel of rels) {
      assert.ok(!new RegExp(`from\\s+["']\\.\\./${esc(rel)}["']`).test(body),
        `server/${f} statically imports ${rel}; use LAZY_TABLES instead`);
    }
  }
});

test("RATCHET: no non-test server module imports sharp statically — libvips must load on first use", () => {
  const serverDir = path.join(repoRoot, "server");
  for (const f of fs.readdirSync(serverDir)) {
    if (!f.endsWith(".ts") || f.endsWith(".test.ts")) continue;
    const body = fs.readFileSync(path.join(serverDir, f), "utf8");
    for (const line of body.split("\n")) {
      if (/^\s*import\s+type\s/.test(line)) continue; // erased at compile time
      assert.ok(!/^\s*import\b[^;]*from\s+["']sharp["']/.test(line) && !/require\(\s*["']sharp["']\s*\)/.test(line),
        `server/${f} loads sharp at import time: ${line.trim()} — go through lazyAssets.getSharp()`);
    }
  }
});

test("resolution: working tree, then dist/ under cwd (production image layout), then this module's own tree", () => {
  const rel = "datacore/nuclear_tests.json";
  assert.equal(resolveLazyAssetPath(rel, repoRoot), path.join(repoRoot, rel));
  // the image: cwd=/app holds dist/datacore/... and no datacore/ tree
  const image = tmp("lazyassets-image-");
  fs.mkdirSync(path.join(image, "dist", "datacore"), { recursive: true });
  fs.writeFileSync(path.join(image, "dist", rel), "{}");
  assert.equal(resolveLazyAssetPath(rel, image), path.join(image, "dist", rel));
  // a cwd with neither (aircraftEntities.test.ts's chdir ratchet; any process
  // started outside the app root): found next to this module's own tree
  const bare = tmp("lazyassets-bare-");
  assert.equal(resolveLazyAssetPath(rel, bare), path.join(repoRoot, rel));
});

test("a table that exists nowhere resolves to the honest direct path and get() THROWS naming it — never an empty dataset", () => {
  const bare = tmp("lazyassets-none-");
  const nope = lazyJsonTable("datacore/definitely_not_staged.json");
  assert.equal(resolveLazyAssetPath(nope.rel, bare), path.join(bare, nope.rel));
  assert.throws(() => nope.get(), /definitely_not_staged\.json[\s\S]*script\/build\.ts/);
  assert.throws(() => nope.peek(), /definitely_not_staged\.json/);
  assert.equal(nope.loaded(), false);
  assert.equal(nope.bytes(), 0);
});

test("a corrupt staged file throws naming the path (never a silently empty table)", () => {
  const cwd = tmp("lazyassets-corrupt-");
  fs.mkdirSync(path.join(cwd, "dist", "datacore"), { recursive: true });
  fs.writeFileSync(path.join(cwd, "dist", "datacore", "broken.json"), "{ not json");
  const prev = process.cwd();
  process.chdir(cwd);
  try {
    const t = lazyJsonTable("datacore/broken.json");
    assert.throws(() => t.get(), /broken\.json[\s\S]*not valid JSON/);
    assert.equal(t.loaded(), false);
  } finally {
    process.chdir(prev);
  }
});

test("get() still serves after chdir away from the repo (the production bundle's cwd-independence, as aircraftEntities pins for the spine)", () => {
  LAZY_TABLES.nuclearTests.reset();
  const prev = process.cwd();
  process.chdir(os.tmpdir());
  try {
    assert.deepStrictEqual(LAZY_TABLES.nuclearTests.get(), nuclearTests);
  } finally {
    process.chdir(prev);
    LAZY_TABLES.nuclearTests.reset();
  }
});

test("snapshotStreams(): same stamps the static imports gave, computed WITHOUT retaining any lazy table, memoized", () => {
  for (const t of lazyTableList()) t.reset();
  _resetSnapshotStreamsForTests();
  const s = snapshotStreams();
  assert.deepEqual(Object.keys(s).sort(), ["aircraftspine", "cpcdegreedays", "eiaweekly", "jodi", "railep724", "uncomtrade"]);
  assert.deepEqual(s.eiaweekly, stamp("datacore/eia/weekly_series.json", eiaWeekly));
  assert.deepEqual(s.cpcdegreedays, stamp("datacore/cpc/degree_days.json", cpcDegreeDays));
  assert.deepEqual(s.jodi, stamp("datacore/jodi/primary_stocks.json", jodiPrimaryStocks));
  assert.deepEqual(s.railep724, stamp("datacore/rail/ep724_carloads.json", railEp724));
  assert.deepEqual(s.uncomtrade, stamp("datacore/un_comtrade/bilateral_trade.json", unComtrade));
  assert.deepEqual(s.aircraftspine, stamp("datacore/aircraft/entity_spine.json", entitySpine));
  for (const v of Object.values(s)) assert.ok(v.built && !Number.isNaN(Date.parse(v.built)), `${v.artifact}: real embedded build time`);
  for (const t of [LAZY_TABLES.eiaWeekly, LAZY_TABLES.cpcDegreeDays, LAZY_TABLES.jodiPrimaryStocks, LAZY_TABLES.railEp724, LAZY_TABLES.entitySpine]) {
    assert.equal(t.loaded(), false, `${t.rel} must not be pinned in memory by its build stamp`);
  }
  assert.strictEqual(snapshotStreams(), s, "stamps are memoized for the process lifetime");
});

test("stamp(): built, then built_at, else null; non-ISO strings are null", () => {
  assert.deepEqual(stamp("a", { built: "2026-11-02T13:39:27.506732+00:00" }), { artifact: "a", built: "2026-11-02T13:39:27.506732+00:00" });
  assert.deepEqual(stamp("a", { built_at: "2026-11-02T13:40:01Z" }), { artifact: "a", built: "2026-11-02T13:40:01Z" });
  assert.deepEqual(stamp("a", { built: "yesterday" }), { artifact: "a", built: null });
  assert.deepEqual(stamp("a", {}), { artifact: "a", built: null });
});

test("VOLTRADE_LAZY_ASSETS: unset/1 = lazy (loads nothing); 0 = eager preload of every table + sharp through the same loaders", async () => {
  assert.equal(lazyAssetsEnabled({}), true);
  assert.equal(lazyAssetsEnabled({ [LAZY_ASSETS_ENV]: "1" }), true);
  assert.equal(lazyAssetsEnabled({ [LAZY_ASSETS_ENV]: "0" }), false);
  for (const t of lazyTableList()) t.reset();
  const lazy = await preloadLazyAssets({});
  assert.equal(lazy.mode, "lazy");
  assert.equal(lazy.tables, 0);
  assert.ok(lazyTableList().every((t) => !t.loaded()), "lazy mode must load no table at boot");
  const eager = await preloadLazyAssets({ [LAZY_ASSETS_ENV]: "0" });
  assert.equal(eager.mode, "eager");
  assert.equal(eager.tables, lazyTableList().length);
  assert.equal(eager.failed, 0);
  assert.ok(lazyTableList().every((t) => t.loaded()), "eager mode must leave every table memoized");
  assert.equal(eager.bytes, lazyTableList().reduce((s, t) => s + t.bytes(), 0));
  assert.ok(eager.bytes > 15_000_000, `eager preload read ${eager.bytes} bytes — the real tables are ~18MB`);
  assert.equal(eager.sharp, "loaded");
  assert.ok(sharpLoaded());
});

test("getSharp(): one module for every caller, usable for a real encode", async () => {
  const a = await getSharp();
  const b = await getSharp();
  assert.strictEqual(a, b);
  assert.equal(typeof a, "function");
  const png = await a({ create: { width: 2, height: 2, channels: 4, background: { r: 1, g: 2, b: 3, alpha: 1 } } }).png().toBuffer();
  assert.ok(png.length > 0);
  const meta = await a(png).metadata();
  assert.equal(meta.width, 2);
  assert.equal(meta.height, 2);
});
