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

test("KNOWN BROKEN #18 continuation 2026-07-20: the daemon branch surfaces active_dispatch_detail (method + elapsed time per in-flight dispatch), not just the bare count", () => {
  // active_dispatches alone can't distinguish healthy 2x concurrency from a
  // self-perpetuating cascade (an abandoned run_full_scan zombie thread from
  // a PRIOR timed-out cycle still running, competing for the shared
  // alpaca_throttle bucket with a fresh run_full_scan) — three live
  // TIER2-ERROR catches this session all read active_dispatches=2 with no
  // way to tell which. voltrade_daemon.py's _health() now returns
  // active_dispatch_detail; this pins that bot.ts actually reads it.
  const block = tier2ScanTryCatch();
  const daemonBranchStart = block.indexOf("if (err?.daemonFailure)");
  const daemonBranchEnd = block.indexOf("} else {", daemonBranchStart);
  const daemonBranch = block.slice(daemonBranchStart, daemonBranchEnd);
  assert.ok(daemonBranch.includes("active_dispatch_detail"), "must read hr.active_dispatch_detail off the unwrapped health result");
  assert.ok(daemonBranch.includes("d.method") && daemonBranch.includes("d.elapsed_sec"), "must format each active dispatch's method name and elapsed time, not just the count");
});

test("KNOWN BROKEN #18 continuation 2026-07-21: the daemon branch surfaces layer2_prefetch (cache_hit/completed/total/elapsed_sec/budget_exceeded/age), not just active_dispatch_detail", () => {
  // Two prior sessions (v1.0.418, v1.0.454) tried to catch a live
  // TIER2-ERROR and csp_universe.py's Layer 2 prefetch stats in the SAME
  // window by polling /api/diag/timings and missed both times — that
  // endpoint only reflects the last scan_market() call that actually
  // returned, never a call still hung past its own 300s timeout.
  // voltrade_daemon.py's _health() now reads csp_universe's module-level
  // stats live (even mid-hang, from a separate RPC thread); this pins that
  // bot.ts actually surfaces it in the same audit line instead of needing
  // a separate live stakeout.
  const block = tier2ScanTryCatch();
  const daemonBranchStart = block.indexOf("if (err?.daemonFailure)");
  const daemonBranchEnd = block.indexOf("} else {", daemonBranchStart);
  const daemonBranch = block.slice(daemonBranchStart, daemonBranchEnd);
  assert.ok(daemonBranch.includes("layer2_prefetch"), "must read hr.layer2_prefetch off the unwrapped health result");
  assert.ok(daemonBranch.includes("l2.cache_hit") && daemonBranch.includes("l2.budget_exceeded"), "must format cache_hit/budget_exceeded — the exact fields this item's own NEXT STEP asks to correlate against a live TIER2-ERROR");
  assert.ok(daemonBranch.includes("l2.age_sec"), "must surface how stale the reading is, so a future session can tell a fresh mid-hang reading from a leftover prior-scan one");
});

test("KNOWN BROKEN #18 continuation 2026-07-22: the daemon branch reads bot_engine.py's own scan-timings file directly off disk, not just layer2_prefetch", () => {
  // v1.0.468 shipped the deep_score ThreadPoolExecutor shutdown-hazard fix
  // as this item's third named mechanism; a full day of live post-deploy
  // evidence (13 fresh TIER2-ERROR occurrences, all still active_dispatches=2
  // with high layer2_prefetch age) shows the storm continuing unchanged —
  // Layer 2 alone was never sufficient to pinpoint the hang. bot_engine.py's
  // pre-existing TIMING-DISK mechanism (2026-04-23) already persists
  // per-phase progress straight to shared disk, generalizing "which phase"
  // beyond Layer 2 to run_full_scan's entire pipeline — this pins that the
  // daemon-timeout branch reads it directly (same file /api/diag/timings and
  // /api/system/snapshot already read) instead of requiring a live stakeout.
  const block = tier2ScanTryCatch();
  const daemonBranchStart = block.indexOf("if (err?.daemonFailure)");
  const daemonBranchEnd = block.indexOf("} else {", daemonBranchStart);
  const daemonBranch = block.slice(daemonBranchStart, daemonBranchEnd);
  assert.ok(daemonBranch.includes("voltrade_scan_timings.json"), "must read the same TIMING-DISK file bot_engine.py's _scan_market_inner() writes progressively");
  assert.ok(daemonBranch.includes("scanTimingsDetail"), "must build a scan-timings detail string");
  assert.ok(daemonBranch.includes("last_phase_completed"), "must surface last_phase_completed — the exact field that names where a stuck scan last checkpointed");
  assert.ok(/daemonState\s*=[\s\S]*scanTimingsDetail/.test(daemonBranch), "scanTimingsDetail must actually be interpolated into the daemonState message, not computed and discarded");
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
