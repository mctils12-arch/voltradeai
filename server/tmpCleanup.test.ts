// See server/tmpCleanup.ts header for the full KNOWN BROKEN #18 correlation
// this fix targets (EVENTLOOP-LAG stalls landing on the same 600000ms
// cadence as the old synchronous fs.readdirSync/statSync/unlinkSync sweep).
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import fsp from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  cleanupOrphanedTempFiles,
  TMP_CLEANUP_INTERVAL_MS,
  TMP_FILE_MAX_AGE_MS,
  TMP_CLEANUP_AUDIT_THRESHOLD,
} from "./tmpCleanup";

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");

async function withTmpDir(fn: (dir: string) => Promise<void>) {
  const dir = await fsp.mkdtemp(path.join(os.tmpdir(), "voltrade-tmpcleanup-test-"));
  try {
    await fn(dir);
  } finally {
    await fsp.rm(dir, { recursive: true, force: true });
  }
}

async function touch(dir: string, name: string, ageMs: number) {
  const full = path.join(dir, name);
  await fsp.writeFile(full, "{}");
  const past = new Date(Date.now() - ageMs);
  await fsp.utimes(full, past, past);
}

test("cleanupOrphanedTempFiles deletes matching-prefix files older than TMP_FILE_MAX_AGE_MS", async () => {
  await withTmpDir(async (dir) => {
    await touch(dir, "fb_123.json", TMP_FILE_MAX_AGE_MS + 60000);
    await touch(dir, "fill_x_ABC_1.json", TMP_FILE_MAX_AGE_MS + 60000);
    await touch(dir, "opt_order_XYZ_1.json", TMP_FILE_MAX_AGE_MS + 60000);

    const result = await cleanupOrphanedTempFiles(dir);
    assert.equal(result.scanned, 3);
    assert.equal(result.deleted, 3);
    assert.deepEqual(await fsp.readdir(dir), []);
  });
});

test("cleanupOrphanedTempFiles leaves matching-prefix files newer than TMP_FILE_MAX_AGE_MS alone", async () => {
  await withTmpDir(async (dir) => {
    await touch(dir, "fb_recent.json", 1000);
    const result = await cleanupOrphanedTempFiles(dir);
    assert.equal(result.scanned, 1);
    assert.equal(result.deleted, 0);
    assert.deepEqual(await fsp.readdir(dir), ["fb_recent.json"]);
  });
});

test("cleanupOrphanedTempFiles ignores files that don't match any tracked prefix, however old", async () => {
  await withTmpDir(async (dir) => {
    await touch(dir, "unrelated_cache.json", TMP_FILE_MAX_AGE_MS + 60000);
    const result = await cleanupOrphanedTempFiles(dir);
    assert.equal(result.scanned, 0);
    assert.equal(result.deleted, 0);
    assert.deepEqual(await fsp.readdir(dir), ["unrelated_cache.json"]);
  });
});

test("cleanupOrphanedTempFiles never throws when the directory doesn't exist", async () => {
  const result = await cleanupOrphanedTempFiles("/nonexistent/path/for/this/test");
  assert.deepEqual(result, { scanned: 0, deleted: 0 });
});

test("REGRESSION GUARD: tmpCleanup.ts must not use synchronous fs calls (the whole point of this fix)", () => {
  const src = fs.readFileSync(path.join(here, "tmpCleanup.ts"), "utf8");
  assert.ok(!/fs\.(readdirSync|statSync|unlinkSync)/.test(src), "tmpCleanup.ts must use fs/promises, not the Sync fs API — a Sync call here reintroduces KNOWN BROKEN #18's event-loop-blocking hazard");
  assert.ok(/from\s+["']node:fs\/promises["']/.test(src), "must import from node:fs/promises");
});

test("wiring pinned: bot.ts imports cleanupOrphanedTempFiles from ./tmpCleanup instead of reimplementing the sweep inline", () => {
  assert.ok(
    /import\s*\{[^}]*cleanupOrphanedTempFiles[^}]*\}\s*from\s*"\.\/tmpCleanup"/.test(bot),
    "bot.ts must import cleanupOrphanedTempFiles (and siblings) from ./tmpCleanup",
  );
});

test("wiring pinned: bot.ts no longer runs the temp-file sweep with synchronous fs calls", () => {
  const anchorIdx = bot.indexOf("Temp File Cleanup (OOM fix)");
  assert.ok(anchorIdx > 0, "temp file cleanup block not found in bot.ts");
  const block = bot.slice(anchorIdx, anchorIdx + 1500);
  assert.ok(!/fs\.(readdirSync|statSync|unlinkSync)\(/.test(block), "the temp-file cleanup block in bot.ts must not call the Sync fs API directly — that was KNOWN BROKEN #18's suspected root cause");
  assert.ok(block.includes("cleanupOrphanedTempFiles("), "must delegate to cleanupOrphanedTempFiles");
  assert.ok(block.includes("TMP_CLEANUP_INTERVAL_MS"), "must tick at TMP_CLEANUP_INTERVAL_MS");
});

test("wiring pinned: bot.ts audits an anomalously large scan under a distinct TMP-CLEANUP type, gated on TMP_CLEANUP_AUDIT_THRESHOLD", () => {
  const anchorIdx = bot.indexOf("Temp File Cleanup (OOM fix)");
  const block = bot.slice(anchorIdx, anchorIdx + 1500);
  assert.ok(block.includes("TMP_CLEANUP_AUDIT_THRESHOLD"), "must gate the diagnostic audit on TMP_CLEANUP_AUDIT_THRESHOLD");
  assert.ok(block.includes('audit("TMP-CLEANUP"'), "must log an audit entry under a distinct TMP-CLEANUP type so a future session can see whether a real backlog was occurring");
});

test("TMP_CLEANUP_INTERVAL_MS matches the previously-hardcoded 600000ms cadence (unchanged behavior, only the blocking mechanism changed)", () => {
  assert.equal(TMP_CLEANUP_INTERVAL_MS, 600000);
});

test("TMP_FILE_MAX_AGE_MS matches the previously-hardcoded 300000ms staleness threshold (unchanged behavior)", () => {
  assert.equal(TMP_FILE_MAX_AGE_MS, 300000);
});

test("TMP_CLEANUP_AUDIT_THRESHOLD is set well above a single scan's normal temp-file volume", () => {
  assert.ok(TMP_CLEANUP_AUDIT_THRESHOLD >= 100, "threshold should be high enough that a normal cycle never trips it, so an audit entry is itself meaningful evidence");
});
