// 2026-09-07 PRODUCT session: pins the Tier 3 wiring for
// portDwellCapture.ts's in-process weekly-snapshot capture (see that
// module's own header + research/experiments.md this date). Mirrors
// tier3ManipVisibility.test.ts's string-pinning approach for the
// same-file wiring class.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");

function captureBlock(): string {
  const start = bot.indexOf("captureNextPortDwellWeekIfDue(portDwellPorts");
  assert.ok(start > 0, "Tier 3 port-dwell capture call site not found in bot.ts");
  const catchStart = bot.indexOf("} catch (err: unknown) {", start);
  assert.ok(catchStart > start, "port-dwell capture catch block not found");
  const catchEnd = bot.indexOf("\n    }", catchStart);
  assert.ok(catchEnd > catchStart, "port-dwell capture catch block not closed as expected");
  return bot.slice(start - 400, catchEnd);
}

test("wiring pinned: bot.ts imports captureIfDue from portDwellCapture.ts, not a re-derived copy", () => {
  assert.ok(bot.includes('import { captureIfDue as captureNextPortDwellWeekIfDue, loadCapturedSnapshots } from "./portDwellCapture"'),
    "must import the shared capture function and read helper, not reimplement them inline");
});

test("wiring pinned: Tier 3 calls captureIfDue with live ports + the live raw-vessel-retention floor, in-process (no fetch/HTTP)", () => {
  const block = captureBlock();
  assert.ok(block.includes("portsFromSites((datacoreSites as any).sites"), "must derive ports from the live sites registry, same as portdwell_window");
  assert.ok(block.includes('oldestRawHour("vessels")'), "must pass the live raw-retention floor, not a stale/assumed one");
  assert.ok(!block.includes("fetch("), "this must be an in-process call, never an HTTP round trip (that's the whole point of the fix)");
});

test("wiring pinned: captured and skipped-degenerate outcomes are audited (persisted trail), not just console.error", () => {
  const block = captureBlock();
  assert.ok(block.includes('audit("TIER3-PORTDWELL"'), "capture outcomes must reach the persisted audit log");
  assert.ok(block.includes('"captured"') && block.includes('"skipped_degenerate"'), "must branch on both outcomes worth a human/session seeing");
  assert.ok(block.includes("console.error"), "console diagnostics stay too — audit() is additive, matching the repo's own TIER3 error-visibility convention");
});

test("wiring pinned: portdwell_weekly_captured diag probe reads the same capture module's state, read-only", () => {
  const start = bot.indexOf('case "portdwell_weekly_captured"');
  const end = bot.indexOf("default:", start);
  assert.ok(start > 0 && end > start, "portdwell_weekly_captured probe block not found");
  const block = bot.slice(start, end);
  assert.ok(block.includes("loadCapturedSnapshots()"), "must reuse the shared reader, not re-derive a file path inline");
  assert.ok(block.includes("sanitizeDiag"), "portdwell_weekly_captured probe must pass the sanitizer like every other probe");
});
