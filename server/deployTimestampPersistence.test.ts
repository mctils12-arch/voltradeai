// REPAIR (found via live /api/diag/audit this session, PRIORITY-2 integrity
// of learning): DEPLOY_TIMESTAMP gates adjustStrategyWeights()'s post-deploy
// win-rate window (server/bot.ts) — it existed to exclude trades from the
// pre-27-bug-fix/scale-out/HEAT-CAP broken-code era, a ONE-TIME historical
// boundary. But it was computed as `new Date().toISOString()` on every
// process boot, and Railway redeploys on every merge to main — including
// merges with zero trading-logic relevance. Live evidence this session:
// server_version matched the day's latest merge and node_uptime_s showed
// the process had been up only ~2 hours, during which exactly 20 trades
// closed at a 0% win rate — that noisy 2-hour sample was driving real
// strategyWeights shifts (VRP toward 40%, momentum toward 15%), and would
// reset again at the next unrelated deploy. Fix: persist the timestamp
// across restarts (set once, read thereafter), the same pattern
// EQUITY_CURVE_PATH already uses a few lines above it in bot.ts.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");

// Extract and actually execute loadOrInitDeployTimestamp — it's a pure
// function of its `paths` argument plus real fs I/O, no closures over other
// bot.ts module state, so behavioral testing against real temp files is
// cheap and exact (same technique as loadIsOptionSymbol in
// optionSymbolStreamSubscribe.test.ts).
function loadOrInitDeployTimestampFactory(): (paths: string[]) => string {
  const start = bot.indexOf("function loadOrInitDeployTimestamp(paths: string[]): string {");
  assert.ok(start > 0, "loadOrInitDeployTimestamp() not found in bot.ts");
  const end = bot.indexOf("\n}", start);
  const src = bot.slice(start, end + 2)
    .replace(
      "function loadOrInitDeployTimestamp(paths: string[]): string {",
      "function loadOrInitDeployTimestamp(paths) {",
    );
  const factory = new Function("fs", `${src}\nreturn loadOrInitDeployTimestamp;`);
  return factory(fs);
}

function tmpPaths(n: number): string[] {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "deploy-ts-test-"));
  return Array.from({ length: n }, (_, i) => path.join(dir, `deploy_timestamp_${i}.json`));
}

test("first-ever boot mints a timestamp and persists it to every candidate path", () => {
  const loadOrInitDeployTimestamp = loadOrInitDeployTimestampFactory();
  const [p1, p2] = tmpPaths(2);
  const before = Date.now();
  const minted = loadOrInitDeployTimestamp([p1, p2]);
  const after = Date.now();

  assert.ok(!Number.isNaN(Date.parse(minted)), "must return a valid ISO timestamp");
  const mintedMs = Date.parse(minted);
  assert.ok(mintedMs >= before - 1000 && mintedMs <= after + 1000, "minted timestamp must be roughly 'now'");

  assert.ok(fs.existsSync(p1), "the write loop must persist to the first writable path");
  assert.equal(JSON.parse(fs.readFileSync(p1, "utf8")).deployTimestamp, minted);
});

test("a restart reads the persisted timestamp back instead of minting a new one — the core fix", () => {
  const loadOrInitDeployTimestamp = loadOrInitDeployTimestampFactory();
  const [p1, p2] = tmpPaths(2);

  const firstBoot = loadOrInitDeployTimestamp([p1, p2]);

  // Simulate real time passing across a restart (e.g. an unrelated merge
  // redeploying the whole app hours later) — the bug this test guards
  // against was `new Date().toISOString()` recomputing "now" on every boot.
  const laterRealNow = new Date(Date.now() + 6 * 60 * 60 * 1000).toISOString();
  const realDateNow = Date.now;
  Date.now = () => Date.parse(laterRealNow);
  try {
    const secondBoot = loadOrInitDeployTimestamp([p1, p2]);
    assert.equal(secondBoot, firstBoot, "a subsequent boot must reuse the persisted boundary, not reset it to the new 'now'");
  } finally {
    Date.now = realDateNow;
  }
});

test("falls back to the second path when the first is unwritable, and reads from whichever path has the value", () => {
  const loadOrInitDeployTimestamp = loadOrInitDeployTimestampFactory();
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "deploy-ts-test-"));
  const unwritable = path.join(dir, "no-such-parent-dir", "deploy_timestamp.json");
  // Make the parent a file (not a directory) so mkdirSync/writeFileSync for
  // the first path both fail deterministically, forcing the fallback.
  fs.writeFileSync(path.join(dir, "no-such-parent-dir"), "not a directory");
  const fallback = path.join(dir, "deploy_timestamp_fallback.json");

  const minted = loadOrInitDeployTimestamp([unwritable, fallback]);
  assert.ok(fs.existsSync(fallback), "must fall back to the second path when the first path's directory cannot be created");
  assert.equal(JSON.parse(fs.readFileSync(fallback, "utf8")).deployTimestamp, minted);

  // A later boot with the same path order must find the value via the
  // fallback (first path still unwritable) rather than minting a new one.
  const laterRealNow = new Date(Date.now() + 6 * 60 * 60 * 1000).toISOString();
  const realDateNow = Date.now;
  Date.now = () => Date.parse(laterRealNow);
  try {
    const secondBoot = loadOrInitDeployTimestamp([unwritable, fallback]);
    assert.equal(secondBoot, minted, "must read the persisted value from the fallback path rather than reset on the next boot");
  } finally {
    Date.now = realDateNow;
  }
});

test("a corrupt/unreadable persisted file is treated as absent, not a crash", () => {
  const loadOrInitDeployTimestamp = loadOrInitDeployTimestampFactory();
  const [p1, p2] = tmpPaths(2);
  fs.writeFileSync(p1, "{ not valid json");

  assert.doesNotThrow(() => loadOrInitDeployTimestamp([p1, p2]));
});

test("DEPLOY_TIMESTAMP is wired through the persisted loader, not a bare new Date() at boot", () => {
  assert.ok(
    /const DEPLOY_TIMESTAMP = loadOrInitDeployTimestamp\(\[DEPLOY_TIMESTAMP_PATH, DEPLOY_TIMESTAMP_FALLBACK\]\);/.test(bot),
    "DEPLOY_TIMESTAMP must be sourced from the persisted loader so it survives redeploys",
  );
  assert.ok(
    !/^const DEPLOY_TIMESTAMP = new Date\(\)\.toISOString\(\);/m.test(bot),
    "DEPLOY_TIMESTAMP must not be recomputed fresh on every process boot — that resets the post-deploy learning window on every unrelated redeploy",
  );
  assert.ok(
    bot.includes('"/data/voltrade/voltrade_deploy_timestamp.json"'),
    "must persist on the Railway volume, matching the EQUITY_CURVE_PATH convention",
  );
});
