// Durability ratchet (audit 2026-07-05): every server module that WRITES to
// the filesystem must be classified here. The Railway volume mounts at
// /data; anything written elsewhere in the container is wiped on every
// redeploy — the volume-attach analysis showed ~5 days of position archive
// lost that way. A NEW module with fs writes that is not classified below
// fails this test: the author must either route paths through
// archiveBaseDir()/the /data convention (DURABLE) or add it to
// TMP_BY_DESIGN with a justification.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const SERVER = path.dirname(fileURLToPath(import.meta.url));

// Modules whose writes are all transient by design (justification required).
const TMP_BY_DESIGN: Record<string, string> = {
  "routes.ts": "CBOE universe cache at /tmp (regenerable); all archive writes route through archiveBaseDir()/DATA_DIR",
  "bot.ts": "/tmp scratch JSON handed to spawned python and removed immediately (IPC); state writes use /data/voltrade with /tmp local fallback",
};

// Known stray, wishlisted for the human (frozen file — see wishlist billing
// amendment): billing SQLite at process.cwd(). MUST shrink, never grow.
const WISHLISTED_STRAYS = new Set(["billing.ts"]);

const WRITE_RE = /fs\.(writeFileSync|appendFileSync|createWriteStream|mkdirSync)\s*\(|new Database\s*\(/;
const DURABLE_MARKERS = [
  "archiveBaseDir(",              // datacore streams
  '"/data',                       // explicit volume path
  "'/data",
  "DATA_DIR",                     // env-integrated /data/voltrade convention
];

test("every fs-writing server module is classified durable, tmp-by-design, or wishlisted", () => {
  const unclassified: string[] = [];
  for (const f of fs.readdirSync(SERVER)) {
    if (!f.endsWith(".ts") || f.endsWith(".test.ts")) continue;
    if (f === "vite.ts") continue; // build tooling, not runtime persistence
    const src = fs.readFileSync(path.join(SERVER, f), "utf8");
    if (!WRITE_RE.test(src)) continue;
    if (TMP_BY_DESIGN[f]) continue;
    if (WISHLISTED_STRAYS.has(f)) continue;
    const durable = DURABLE_MARKERS.some((m) => src.includes(m));
    if (!durable) unclassified.push(f);
  }
  assert.deepEqual(unclassified, [],
    `modules write to disk without a durable-path marker and without a stated ` +
    `classification — route paths through archiveBaseDir()/the /data convention ` +
    `or classify in durability.test.ts with justification: ${unclassified.join(", ")}`);
});

test("archiveBaseDir resolves under /data when the volume exists (the mount contract)", async () => {
  const { archiveBaseDir } = await import("./datacoreArchive");
  const base = archiveBaseDir();
  if (fs.existsSync("/data")) {
    assert.ok(base.startsWith("/data/"), `volume present but archive base is ${base}`);
  } else {
    assert.ok(base.startsWith("/tmp") || process.env.DATA_DIR,
      `no volume: expected /tmp fallback or DATA_DIR override, got ${base}`);
  }
});

test("the wishlisted-stray set never grows silently", () => {
  // billing.ts is the ONLY accepted stray (frozen file; migration proposal
  // filed in research/wishlist.md for the human). If billing.ts is fixed,
  // remove it here so the ratchet tightens.
  assert.deepEqual(Array.from(WISHLISTED_STRAYS), ["billing.ts"]);
  const src = fs.readFileSync(path.join(SERVER, "billing.ts"), "utf8");
  assert.ok(src.includes('path.resolve(process.cwd(), "voltradeai.db")'),
    "billing.ts stray signature changed — if the path was fixed to /data, " +
    "remove billing.ts from WISHLISTED_STRAYS and close the wishlist item");
});
