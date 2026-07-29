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

test("KNOWN BROKEN #18 continuation 2026-07-27: the daemon branch surfaces layer1_stats (cache_hit/universe_size/elapsed_sec/age), not just layer2_prefetch", () => {
  // Live evidence this session: every tier_engine_start-stuck TIER2-ERROR
  // occurrence since v1.0.503's on_phase instrumentation shipped (11/11
  // checked, 2026-07-26 16:33Z through 2026-07-27 13:44Z) read
  // layer2_prefetch cache_hit=true (elapsed=0s) — Layer 2 was never the
  // explanation for THESE hangs. csp_universe._layer1_hard_gates has its
  // own independent 15-min cache and was never instrumented at all; this
  // pins that bot.ts surfaces it in the same audit line so the next
  // occurrence can show whether Layer 1, not Layer 2, is what's hanging.
  const block = tier2ScanTryCatch();
  const daemonBranchStart = block.indexOf("if (err?.daemonFailure)");
  const daemonBranchEnd = block.indexOf("} else {", daemonBranchStart);
  const daemonBranch = block.slice(daemonBranchStart, daemonBranchEnd);
  assert.ok(daemonBranch.includes("layer1_stats"), "must read hr.layer1_stats off the unwrapped health result");
  assert.ok(daemonBranch.includes("l1.cache_hit") && daemonBranch.includes("l1.universe_size"), "must format cache_hit/universe_size — the fields that distinguish a fast cache hit from a live full-universe fetch");
  assert.ok(daemonBranch.includes("l1.age_sec"), "must surface how stale the reading is, same discipline layer2_prefetch's age_sec already holds");
  assert.ok(/daemonState\s*=[\s\S]*layer1Detail/.test(daemonBranch), "layer1Detail must actually be interpolated into the daemonState message, not computed and discarded");
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

test("KNOWN BROKEN #18 continuation 2026-07-27 #2: the daemon branch surfaces the preamble's total duration and slowest phase, not just last_phase+age", () => {
  // Live evidence this session: every busy-hours TIER2-ERROR occurrence
  // today (18:09Z-19:49Z, 6/6) read last_phase=tier_engine_start with a
  // SMALL age (20-40s) — meaning tier1 itself was only running briefly
  // before the outer 300s daemon timeout fired, so the preamble
  // (universe_load..step10b_defensive_floor) must have already consumed
  // the other ~260-280s. bot_engine.py's TIMING-DISK file already persists
  // each preamble phase's own duration_sec progressively (survives the
  // kill) — this pins that bot.ts sums it and names the slowest phase
  // directly in the audit line, so the next occurrence answers "which
  // preamble phase" without a future session needing to catch a live
  // in-progress scan (the two prior stakeouts for tier1/layer1/layer2 both
  // missed their window before landing on file-based reads instead).
  const block = tier2ScanTryCatch();
  const daemonBranchStart = block.indexOf("if (err?.daemonFailure)");
  const daemonBranchEnd = block.indexOf("} else {", daemonBranchStart);
  const daemonBranch = block.slice(daemonBranchStart, daemonBranchEnd);
  assert.ok(daemonBranch.includes("preambleTotal"), "must sum duration_sec across the non-tier_engine phases into a preamble total");
  assert.ok(daemonBranch.includes("slowest"), "must identify the single slowest recorded phase, not just the total");
  assert.ok(
    /startsWith\(["']tier_engine["']\)/.test(daemonBranch),
    "the preamble total must exclude tier_engine_* phases (those are the tier engine itself, not the preamble that precedes it)",
  );
  assert.ok(/scanTimingsDetail\s*=[\s\S]*preambleDetail/.test(daemonBranch), "preambleDetail must actually be interpolated into scanTimingsDetail, not computed and discarded");
});

test("KNOWN BROKEN #18 continuation 2026-07-28: the daemon branch surfaces surface_score_stats (calls/cache_misses/cumulative_elapsed/age), not just layer1_stats/layer2_prefetch", () => {
  // Live evidence this session: two fresh TIER2-ERROR occurrences today
  // (13:36:33Z, 13:42:33Z) both read preamble={... slowest=
  // step6_trade_loop_and_options:233-240s} — the options scanner phase
  // RECURRING as the dominant cost despite v1.0.503's "ROOT CAUSE FOUND +
  // FIXED" claim for that exact phase. Root-caused this session:
  // options_scanner.py's Setup 7 calls vol_surface.get_surface_score() for
  // every high_iv/anchor candidate, and that function's own spot/chain/bars
  // fetches are NOT gated by alpaca_throttle — a cost none of this item's
  // prior sessions' cost models accounted for. This pins that bot.ts
  // surfaces the new accumulator the same way layer1_stats/layer2_prefetch
  // already are, so the next occurrence's audit line shows the call count
  // and cumulative elapsed time directly instead of requiring another
  // session to re-derive this from scratch.
  const block = tier2ScanTryCatch();
  const daemonBranchStart = block.indexOf("if (err?.daemonFailure)");
  const daemonBranchEnd = block.indexOf("} else {", daemonBranchStart);
  const daemonBranch = block.slice(daemonBranchStart, daemonBranchEnd);
  assert.ok(daemonBranch.includes("surface_score_stats"), "must read hr.surface_score_stats off the unwrapped health result");
  assert.ok(daemonBranch.includes("ss.calls") && daemonBranch.includes("ss.cache_misses"), "must format calls/cache_misses — the fields that distinguish scan volume from actual network cost");
  assert.ok(daemonBranch.includes("ss.cumulative_elapsed_sec"), "must surface the cumulative wall-clock cost, the number this whole finding hinges on");
  assert.ok(daemonBranch.includes("ss.age_sec"), "must surface how stale the reading is, same discipline layer1_stats/layer2_prefetch's age_sec already holds");
  assert.ok(/daemonState\s*=[\s\S]*surfaceDetail/.test(daemonBranch), "surfaceDetail must actually be interpolated into the daemonState message, not computed and discarded");
});

test("KNOWN BROKEN #18 continuation 2026-07-29: the daemon branch surfaces skipped_budget, the scan-wide Setup 7 budget-guard's own truncation counter", () => {
  // The 2026-07-28 accumulator above made the cost visible; it did nothing
  // to CAP it. Live TIER2-ERROR occurrences on 2026-07-27/28 (12+ occurrences,
  // essentially every Tier-2 scan during market hours both days) show
  // cumulative_elapsed_sec (312-365s) alone exceeding the 300s daemon RPC
  // timeout, so run_full_scan() times out and returns NOTHING every cycle —
  // not just no options trades, the entire Tier-2 stock+options scan.
  // options_scanner.py's Setup 7 now stops calling get_surface_score() once
  // this scan's cumulative cost trips a budget; skipped_budget is how the
  // audit line shows whether that budget actually tripped and how many
  // candidates it cost, so a truncated scan is never mistaken for a healthy
  // one just because it finally returned.
  const block = tier2ScanTryCatch();
  const daemonBranchStart = block.indexOf("if (err?.daemonFailure)");
  const daemonBranchEnd = block.indexOf("} else {", daemonBranchStart);
  const daemonBranch = block.slice(daemonBranchStart, daemonBranchEnd);
  assert.ok(daemonBranch.includes("ss.skipped_budget"), "must surface skipped_budget — the truncation must never be silent");
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
