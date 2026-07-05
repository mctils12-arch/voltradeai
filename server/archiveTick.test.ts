import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const here = path.dirname(fileURLToPath(import.meta.url));

// [REPAIR 2026-07-05, audit defect #1] Aircraft + trains archiving was
// request-driven — no visitors meant permanent archive gaps (11 hourly
// gaps each in the first 36h, verified from prod archive stats). The
// eager tick makes the archive independent of traffic. These pins keep
// the tick from being silently removed or de-scoped.
test("routes.ts runs the eager archive tick for aircraft AND trains", () => {
  const src = fs.readFileSync(path.join(here, "routes.ts"), "utf8");
  const at = src.indexOf("ARCHIVE_TICK_REGIONS");
  assert.ok(at > -1, "archive tick regions missing");
  const block = src.slice(at, at + 2000);
  assert.ok(/fetchAircraft\(/.test(block), "tick must archive aircraft");
  assert.ok(/fetchTrains\(\)/.test(block), "tick must archive trains");
  assert.ok(/setInterval\(archiveTick, 10 \* 60_000\)/.test(block), "10-min cadence pinned");
  assert.ok(block.includes("archiveTick();"), "tick must fire once at boot, not wait an interval");
  // rotation must cover multiple strategic regions, not one bbox forever
  const regions = (src.slice(at, src.indexOf("];", at)).match(/lamin:/g) || []).length;
  assert.ok(regions >= 3, `only ${regions} tick regions — coverage rotation de-scoped`);
});
