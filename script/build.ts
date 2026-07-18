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
  "zod-validation-error",
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
  await cp("datacore/gem/ownership.json.gz", "dist/datacore/gem/ownership.json.gz");
  await cp("datacore/gem/methane_emitters.json.gz", "dist/datacore/gem/methane_emitters.json.gz");
}

buildAll().catch((err) => {
  console.error(err);
  process.exit(1);
});
