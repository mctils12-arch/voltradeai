// DAEMON-TIMEOUT-VISIBILITY (2026-07-10): production audit logs showed
// TIER2-ERROR "daemon run_full_scan failed: Daemon timeout" recurring 7x
// across ~90 minutes (18:22-19:57 UTC) with zero diagnostic detail. The
// tier2Intelligence catch block's classification logic (stderr/stdout/
// code/signal + a Node-process memory snapshot) is built for subprocess
// failures; run_full_scan is HEAVY_DAEMON_ONLY (server/bot.ts's pythonCall)
// so a daemon timeout never carries any of those fields, producing a
// content-free "code=? signal=none" line describing the wrong process's
// memory. This pins the fix: daemon-path failures are tagged and routed to
// a distinct branch that probes the daemon's own health (rss/uptime/
// active_dispatches, the new voltrade_daemon.py counter) instead.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");

function tier2ScanTryCatch(): string {
  const enginePathIdx = bot.indexOf('require("path").resolve(process.cwd(), "bot_engine.py")');
  assert.ok(enginePathIdx > 0, "tier2 scan block anchor not found in bot.ts");
  const catchStart = bot.indexOf("} catch (err: any) {", enginePathIdx);
  assert.ok(catchStart > enginePathIdx, "tier2 scan catch block not found");
  // The block runs until the matching close before "Overnight/pre-market research".
  const overnightIdx = bot.indexOf("Overnight/pre-market research", catchStart);
  assert.ok(overnightIdx > catchStart, "overnight-research anchor not found after tier2 catch block");
  return bot.slice(catchStart, overnightIdx);
}

test("wiring pinned: daemon-only failures are tagged so the catch block can distinguish them from subprocess errors", () => {
  const daemonThrowIdx = bot.indexOf('new Error(callResult.result?.error || "scan call failed (no daemon, no subprocess)")');
  assert.ok(daemonThrowIdx > 0, "daemon-failure throw site not found");
  const surroundingBlock = bot.slice(daemonThrowIdx - 400, daemonThrowIdx + 200);
  assert.ok(
    surroundingBlock.includes("(e as any).daemonFailure = true"),
    "the daemon-path Error must be tagged .daemonFailure so the catch block can branch on it (not the raw subprocess-error rethrow path)",
  );
});

test("wiring pinned: the tier2 catch block branches on err.daemonFailure before the subprocess classification logic runs", () => {
  const block = tier2ScanTryCatch();
  const daemonBranchIdx = block.indexOf("if (err?.daemonFailure)");
  assert.ok(daemonBranchIdx > 0, "catch block must check err?.daemonFailure");
  const subprocessClassifyIdx = block.indexOf("Gather everything useful: stderr, stdout tail");
  assert.ok(subprocessClassifyIdx > daemonBranchIdx, "daemon branch must come before the subprocess stderr/stdout classification logic");
});

test("wiring pinned: the daemon branch probes daemon health via pythonRpc, not the Node process's own memoryUsage()", () => {
  const block = tier2ScanTryCatch();
  const daemonBranchStart = block.indexOf("if (err?.daemonFailure)");
  const daemonBranchEnd = block.indexOf("} else {", daemonBranchStart);
  assert.ok(daemonBranchEnd > daemonBranchStart, "daemon branch must be followed by the subprocess else-branch");
  const daemonBranch = block.slice(daemonBranchStart, daemonBranchEnd);
  assert.ok(daemonBranch.includes('await pythonRpc("health"'), "must query the daemon's own health RPC");
  assert.ok(daemonBranch.includes("active_dispatches"), "must surface active_dispatches — the zombie-thread-pileup signal voltrade_daemon.py now exposes");
  assert.ok(!daemonBranch.includes("process.memoryUsage()"), "must not report Node's own memory for a daemon-side failure — that was the misleading part of the original bug");
});

test("BUGFIX 2026-07-11: the daemon branch unwraps pythonRpc's {status,result} envelope before reading .alive/.active_dispatches, not the raw envelope", () => {
  // pythonRpc's raw return is the RPC envelope voltrade_daemon.py's
  // dispatch() produces: {"status": "ok", "result": {"alive": true, ...}}.
  // _health()'s alive/rss_mb/active_dispatches/uptime_seconds fields live
  // under .result, not on the envelope itself. Checking `h.alive` directly
  // is always undefined/falsy for every successful health call, so every
  // TIER2-ERROR daemon-timeout entry misclassifies a healthy daemon as
  // "non-alive" and never surfaces active_dispatches — the exact evidence
  // KNOWN BROKEN #18 needs to confirm or refute the zombie-thread-pileup
  // theory. Live production audit log confirmed this: every occurrence
  // since v1.0.266 shipped logged "non-alive: {\"status\":\"ok\",\"result\":
  // {\"alive\":true,...}" instead of "daemon rss=... active_dispatches=...".
  const block = tier2ScanTryCatch();
  const daemonBranchStart = block.indexOf("if (err?.daemonFailure)");
  const daemonBranchEnd = block.indexOf("} else {", daemonBranchStart);
  const daemonBranch = block.slice(daemonBranchStart, daemonBranchEnd);
  assert.ok(
    /h\.status\s*===\s*["']ok["']\s*\?\s*h\.result/.test(daemonBranch),
    "must unwrap the {status,result} RPC envelope (h.status === 'ok' ? h.result : ...) before reading alive/active_dispatches — reading h.alive directly on the raw envelope is always undefined",
  );
  assert.ok(
    !/\bh\.alive\b/.test(daemonBranch),
    "must not read .alive directly off the raw pythonRpc envelope — that field lives under .result",
  );
  assert.ok(
    !/\bh\.active_dispatches\b/.test(daemonBranch),
    "must not read .active_dispatches directly off the raw pythonRpc envelope — that field lives under .result",
  );
});

test("wiring pinned: the daemon branch still emits a TIER2-ERROR audit entry (same action type, richer detail)", () => {
  const block = tier2ScanTryCatch();
  const daemonBranchStart = block.indexOf("if (err?.daemonFailure)");
  const daemonBranchEnd = block.indexOf("} else {", daemonBranchStart);
  const daemonBranch = block.slice(daemonBranchStart, daemonBranchEnd);
  assert.ok(daemonBranch.includes('audit("TIER2-ERROR"'), "daemon-timeout failures must still land in the persisted audit log under the same action type existing dashboards/scripts already filter on");
});

test("wiring pinned: the subprocess classification branch (SIGKILL/OOM/timeout) is unchanged, not deleted", () => {
  const block = tier2ScanTryCatch();
  assert.ok(block.includes("likely OOM kill or maxBuffer exceeded"), "subprocess OOM classification must survive");
  assert.ok(block.includes("stdout buffer exceeded"), "subprocess maxBuffer classification must survive");
  assert.ok(block.includes("timed out or killed externally"), "subprocess SIGTERM classification must survive");
  assert.ok(block.includes('audit("TIER2-ERROR"'), "subprocess branch must still audit TIER2-ERROR too");
});

test("wiring pinned: the catch block does not early-return — overnight research must still run after either branch", () => {
  // A prior draft of this fix used `return` inside the daemon branch, which
  // would have silently skipped the unrelated overnight-research call that
  // runs after the try/catch on every tier2Intelligence invocation whenever
  // a scan happened to fail via the daemon path specifically. Guard against
  // that regression recurring.
  const enginePathIdx = bot.indexOf('require("path").resolve(process.cwd(), "bot_engine.py")');
  const catchStart = bot.indexOf("} catch (err: any) {", enginePathIdx);
  const overnightIdx = bot.indexOf("Overnight/pre-market research", catchStart);
  const catchBody = bot.slice(catchStart, overnightIdx);
  assert.ok(!/\breturn;/.test(catchBody), "the tier2 scan catch block must not early-return — it would skip the overnight-research call that follows the try/catch");
});
