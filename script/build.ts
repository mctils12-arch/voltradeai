import { build as esbuild } from "esbuild";
import { build as viteBuild } from "vite";
import { rm, readFile, cp } from "fs/promises";

// server deps to bundle to reduce openat(2) syscalls
// which helps cold start times
const allowlist = [
  "@google/generative-ai",
  "axios",
  "cors",
  "date-fns",
  "drizzle-orm",
  "drizzle-zod",
  "express",
  "express-rate-limit",
  "express-session",
  "jsonwebtoken",
  "memorystore",
  "multer",
  "nanoid",
  "nodemailer",
  "openai",
  "passport",
  "passport-local",
  "stripe",
  "uuid",
  "ws",
  "xlsx",
  "zod",
];

async function buildAll() {
  await rm("dist", { recursive: true, force: true });

  console.log("building client...");
  await viteBuild();

  console.log("building server...");
  const pkg = JSON.parse(await readFile("package.json", "utf-8"));
  const allDeps = [
    ...Object.keys(pkg.dependencies || {}),
    ...Object.keys(pkg.devDependencies || {}),
  ];
  const externals = allDeps.filter((dep) => !allowlist.includes(dep));

  await esbuild({
    entryPoints: ["server/index.ts"],
    platform: "node",
    bundle: true,
    format: "cjs",
    outfile: "dist/index.cjs",
    define: {
      "process.env.NODE_ENV": '"production"',
    },
    minify: true,
    external: externals,
    logLevel: "info",
    // [A7 2026-11-03] server/lazyAssets.ts resolves its staged tables from
    // its own directory; under tsx/ESM that is import.meta.url, which the
    // cjs bundle never reaches (it takes the __dirname branch). esbuild
    // still notices the token and would print the empty-import-meta warning
    // on every build — silenced here, deliberately, for that one known site.
    logOverride: { "empty-import-meta": "silent" },
  });

  // [REPAIR 2026-07-07] Runtime datacore files → dist/datacore/. The
  // production image ships dist/ only (frozen Dockerfile — no repo
  // datacore/ tree), so server code that READS these from disk at
  // runtime (streamsInventory manifests, platformStats sentinel2
  // freshness) found nothing on Railway. Statically-imported JSON is
  // unaffected (esbuild inlines it). Keep this list to what the server
  // actually reads at runtime — datacore/ is 254MB (sentinel2 chips);
  // the runtime set is ~184KB. See server/repoFiles.ts for resolution.
  console.log("copying runtime datacore files into dist/ ...");
  await cp("datacore/manifests", "dist/datacore/manifests", { recursive: true });
  await cp("datacore/sentinel2/readings.jsonl", "dist/datacore/sentinel2/readings.jsonl");
  // QC-2 (2026-08-11): airportsIndex reads the OurAirports catalog at runtime
  await cp("datacore/aircraft/airports_min.json", "dist/datacore/aircraft/airports_min.json");
  await cp("datacore/aircraft/nasr_fixes.json", "dist/datacore/aircraft/nasr_fixes.json");
  // [REPAIR 2026-10-03] server/navAirways.ts reads this via repoDataPath(path.join(...))
  // at runtime but was never staged: on prod the airway table was EMPTY since
  // v1.0.1007, so routeShape.airwayExpanded stayed 0 across ~29k plans and
  // placeDirectFixes() silently ignored every airway token (R14 class, 3rd
  // occurrence — the ratchet only matched the string-literal call form).
  await cp("datacore/aircraft/nasr_airways.json", "dist/datacore/aircraft/nasr_airways.json");
  // [PRODUCT 2026-10-14] server/atcFrequencies.ts reads the NASR frequency table at runtime
  await cp("datacore/aircraft/nasr_atc.json", "dist/datacore/aircraft/nasr_atc.json");
  // [PRODUCT 2026-10-19] server/clinicalTrialsTicker.ts loadSponsorIndex reads this via repoDataPath
  await cp("datacore/clinical_trials_sponsor_map.json", "dist/datacore/clinical_trials_sponsor_map.json");
  await cp("datacore/gem/ownership.json.gz", "dist/datacore/gem/ownership.json.gz");
  await cp("datacore/gem/methane_emitters.json.gz", "dist/datacore/gem/methane_emitters.json.gz");
  // [REPAIR 2026-07-20] gemMethaneAssets.ts (server/gemMethaneAssets.ts,
  // shipped v1.0.409 2026-07-19) reads these two via repoDataPath at
  // runtime but was never added here — exactly the R14 defect class this
  // comment block already names. Live-verified 2026-07-20: production's
  // /api/data/methane-plumes served matchedCount:0/assetStats:[] on every
  // request since gate-2(a) shipped (cachedGemAssets() silently degraded
  // to zero assets, so the proximity join had nothing to match against —
  // no error anywhere, just empty results, the exact silent-failure shape
  // this resolver's own docstring warns about).
  await cp("datacore/gem/oil_gas_extraction.json.gz", "dist/datacore/gem/oil_gas_extraction.json.gz");
  await cp("datacore/gem/coal_mine_tracker.json.gz", "dist/datacore/gem/coal_mine_tracker.json.gz");
  // [PRODUCT 2026-07-21] server/gemCoalMineFeatures.ts reads this via
  // repoDataPath at runtime (new /api/data/coal-mine-features route) —
  // added here in the SAME PR this time, per the R14/2026-07-20 lesson
  // (the ratchet test below caught the miss before it ever reached prod).
  await cp("datacore/gem/coal_mine_features.geojson.gz", "dist/datacore/gem/coal_mine_features.geojson.gz");
  // [PRODUCT] server/gemCoalTerminals.ts reads this via repoDataPath at
  // runtime (new /api/data/coal-terminals route) — added in the same PR,
  // same R14/2026-07-20 lesson, caught by the ratchet test below.
  await cp("datacore/gem/coal_terminals.json", "dist/datacore/gem/coal_terminals.json");
  // [PRODUCT] server/gemIronOreMines.ts reads this via repoDataPath at
  // runtime (new /api/data/iron-ore-mines route) — added in the same PR,
  // same R14/2026-07-20 lesson, caught by the ratchet test below.
  await cp("datacore/gem/iron_ore_mines.json", "dist/datacore/gem/iron_ore_mines.json");
  // [PRODUCT] server/gemIronSteelPlants.ts reads this via repoDataPath at
  // runtime (new /api/data/iron-steel-plants route) — added in the same PR,
  // same R14/2026-07-20 lesson, caught by the ratchet test below.
  await cp("datacore/gem/iron_steel_plants.json", "dist/datacore/gem/iron_steel_plants.json");
  // [PRODUCT] server/gemChemicals.ts reads this via repoDataPath at runtime
  // (new /api/data/chemicals route) — added in the same PR, same
  // R14/2026-07-20 lesson, caught by the ratchet test below.
  await cp("datacore/gem/chemicals.json", "dist/datacore/gem/chemicals.json");
  // [PRODUCT] server/gemLngCarriers.ts reads this via repoDataPath at
  // runtime (new /api/data/lng-shipyards route) — added in the same PR,
  // same R14/2026-07-20 lesson, caught by the ratchet test below.
  await cp("datacore/gem/lng_carriers.json", "dist/datacore/gem/lng_carriers.json");
  // [PRODUCT] server/gemSteelRawMaterials.ts reads this via repoDataPath at
  // runtime (new /api/data/steel-raw-materials route) — added in the same
  // PR, same R14/2026-07-20 lesson, caught by the ratchet test below.
  await cp("datacore/gem/steel_raw_materials.json", "dist/datacore/gem/steel_raw_materials.json");
  // [PRODUCT] server/gemIronSteelPlants.ts's loadGemSteelUnits() reads this
  // via repoDataPath at runtime (furnace-unit detail joined onto
  // /api/data/iron-steel-plants) — added in the same PR, same
  // R14/2026-07-20 lesson, caught by the ratchet test below.
  await cp("datacore/gem/steel_units.json", "dist/datacore/gem/steel_units.json");

  // [A7 2026-11-03, memory audit] The LARGE datacore tables are no longer
  // statically imported (esbuild inlined ~18MB of JSON into dist/index.cjs,
  // parsed and held from boot whether or not a request ever needed it).
  // server/lazyAssets.ts reads each from THIS staged copy on first use, so
  // every entry of its LAZY_TABLES registry must be listed here — the frozen
  // Dockerfile ships dist/ only. server/lazyAssets.test.ts fails until a new
  // registry entry gets its cp() line (same R14 lesson as the gem/ block).
  await cp("datacore/powerplants/us_power_plants.json", "dist/datacore/powerplants/us_power_plants.json");
  await cp("datacore/aircraft/entity_spine.json", "dist/datacore/aircraft/entity_spine.json");
  await cp("datacore/cpc/degree_days.json", "dist/datacore/cpc/degree_days.json");
  await cp("datacore/military_installations.json", "dist/datacore/military_installations.json");
  await cp("datacore/quake_history.json", "dist/datacore/quake_history.json");
  await cp("datacore/nuclear_tests.json", "dist/datacore/nuclear_tests.json");
  await cp("datacore/boundaries/ne_110m_admin0.json", "dist/datacore/boundaries/ne_110m_admin0.json");
  await cp("datacore/boundaries/ne_50m_admin1_lines.json", "dist/datacore/boundaries/ne_50m_admin1_lines.json");
  await cp("datacore/pfas/ucmr5_detections.json", "dist/datacore/pfas/ucmr5_detections.json");
  await cp("datacore/cdc_cancer/county_rates.json", "dist/datacore/cdc_cancer/county_rates.json");
  await cp("datacore/cdc_cancer/counties-10m.topo.json", "dist/datacore/cdc_cancer/counties-10m.topo.json");
  await cp("datacore/jodi/primary_stocks.json", "dist/datacore/jodi/primary_stocks.json");
  await cp("datacore/rail/ep724_carloads.json", "dist/datacore/rail/ep724_carloads.json");
  await cp("datacore/eia/weekly_series.json", "dist/datacore/eia/weekly_series.json");
  await cp("datacore/signal_ladder.json", "dist/datacore/signal_ladder.json");
}

buildAll().catch((err) => {
  console.error(err);
  process.exit(1);
});
