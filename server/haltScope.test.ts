// HALT SCOPE (2026-10-27, audit §6.8 N15) — a halt blocks ENTRIES only.
//
// Before this fix every Node halt also stopped every exit: checkPositionOnTick
// and the 60s POS-KILL sync returned on `!state.active || state.killSwitch`
// (the drawdown kill writes BOTH), the Tier 1 interval skipped tier1Reflex
// while killed or circuit-broken (so the circuit breaker also suspended the
// -10% drawdown check itself), and the WS feed would not reconnect while
// killed. These tests pin the pure decision (server/haltScope.ts) as a table
// and pin HOW bot.ts uses it. Both halves fail on the pre-fix code.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { haltAllows, haltMode, haltStateFrom, type HaltAction, type HaltState } from "./haltScope";

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");

function slice(fromMarker: string, toMarker: string): string {
  const start = bot.indexOf(fromMarker);
  assert.ok(start > 0, `marker not found: ${fromMarker}`);
  const end = bot.indexOf(toMarker, start);
  assert.ok(end > start, `end marker not found after start: ${toMarker}`);
  return bot.slice(start, end);
}

// ── the decision table ──────────────────────────────────────────────────────
// {killed, circuitBreaker, paused} x {entry, exit, ddCheck} -> allowed?
// `paused` is a system halt that wrote state.active = false (the weekly-loss
// pause; the drawdown kill writes it too). Owner not stopped in all 8 rows.
const TABLE: Array<[killed: boolean, circuitBreaker: boolean, paused: boolean, entry: boolean, exit: boolean, ddCheck: boolean]> = [
  // killed  breaker paused   entry  exit  ddCheck
  [false,   false,  false,   true,  true, true],
  [false,   false,  true,    false, true, true],
  [false,   true,   false,   false, true, true], // breaker no longer suspends the -10% check
  [false,   true,   true,    false, true, true],
  [true,    false,  false,   false, true, false], // latched: the auto-resume tick owns the dd tick
  [true,    false,  true,    false, true, false], // the real drawdown-kill state (kill + active=false)
  [true,    true,   false,   false, true, false],
  [true,    true,   true,    false, true, false],
];

for (const [killed, circuitBreaker, paused, entry, exit, ddCheck] of TABLE) {
  const s: HaltState = { ownerStopped: false, killed, circuitBreaker, paused };
  const label = `killed=${killed} breaker=${circuitBreaker} paused=${paused}`;
  test(`halt table: ${label} -> entry=${entry} exit=${exit} ddCheck=${ddCheck}`, () => {
    const expected: Record<HaltAction, boolean> = { entry, exit, ddCheck };
    for (const action of ["entry", "exit", "ddCheck"] as HaltAction[]) {
      assert.equal(haltAllows(s, action), expected[action], `${label} action=${action}`);
    }
  });
}

test("an explicit owner STOP turns everything off, in every halt combination", () => {
  for (const killed of [false, true]) for (const circuitBreaker of [false, true]) {
    const s: HaltState = { ownerStopped: true, killed, circuitBreaker, paused: false };
    for (const action of ["entry", "exit", "ddCheck"] as HaltAction[]) {
      assert.equal(haltAllows(s, action), false, `ownerStopped killed=${killed} breaker=${circuitBreaker} ${action}`);
    }
    assert.equal(haltMode(s), "off");
  }
});

test("protective exits are allowed in EVERY halt (the N15 regression)", () => {
  for (const [killed, circuitBreaker, paused] of TABLE) {
    assert.equal(haltAllows({ ownerStopped: false, killed, circuitBreaker, paused }, "exit"), true);
  }
});

test("entries are blocked by every halt flag on its own", () => {
  assert.equal(haltAllows({ ownerStopped: false, killed: true, circuitBreaker: false, paused: false }, "entry"), false);
  assert.equal(haltAllows({ ownerStopped: false, killed: false, circuitBreaker: true, paused: false }, "entry"), false);
  assert.equal(haltAllows({ ownerStopped: false, killed: false, circuitBreaker: false, paused: true }, "entry"), false);
});

test("haltMode: full / exits_only / off", () => {
  assert.equal(haltMode({ ownerStopped: false, killed: false, circuitBreaker: false, paused: false }), "full");
  assert.equal(haltMode({ ownerStopped: false, killed: true, circuitBreaker: false, paused: true }), "exits_only");
  assert.equal(haltMode({ ownerStopped: false, killed: false, circuitBreaker: true, paused: false }), "exits_only");
  assert.equal(haltMode({ ownerStopped: true, killed: false, circuitBreaker: false, paused: false }), "off");
});

// ── raw bot.ts flags -> HaltState ──────────────────────────────────────────
const NOW = Date.UTC(2026, 9, 27, 15, 0, 0);

test("haltStateFrom: the drawdown kill (killSwitch + active=false) is killed AND paused, not owner-stopped", () => {
  const s = haltStateFrom({ ownerStopped: false, active: false, killSwitch: true, circuitBreakerUntil: 0, nowMs: NOW });
  assert.deepEqual(s, { ownerStopped: false, killed: true, circuitBreaker: false, paused: true });
  assert.equal(haltAllows(s, "exit"), true, "the drawdown kill must not stop protective exits");
  assert.equal(haltAllows(s, "entry"), false);
});

test("haltStateFrom: an owner stop is not a system pause", () => {
  const s = haltStateFrom({ ownerStopped: true, active: false, killSwitch: false, circuitBreakerUntil: 0, nowMs: NOW });
  assert.deepEqual(s, { ownerStopped: true, killed: false, circuitBreaker: false, paused: false });
});

test("haltStateFrom: the circuit breaker is open strictly before circuitBreakerUntil", () => {
  const open = haltStateFrom({ ownerStopped: false, active: true, killSwitch: false, circuitBreakerUntil: NOW + 1, nowMs: NOW });
  const expired = haltStateFrom({ ownerStopped: false, active: true, killSwitch: false, circuitBreakerUntil: NOW, nowMs: NOW });
  assert.equal(open.circuitBreaker, true);
  assert.equal(expired.circuitBreaker, false);
  assert.equal(haltMode(expired), "full");
});

// ── bot.ts wiring ───────────────────────────────────────────────────────────
const onTick = slice("async function checkPositionOnTick(", "// Skip stock exit logic for options positions");
const syncInterval = slice("// Sync monitored positions every 60 seconds", "function startStreaming()");
const tier1Interval = slice("// TIER 1: Reflex (every 45 seconds)", "// TIER 2: Intelligence");
const haltedFn = slice("async function tier1HaltedExits(", "async function tier1Reflex()");
const protectiveFn = slice("async function tier1ProtectiveSync()", "async function tier1HaltedExits(");
const tier1Reflex = slice("async function tier1Reflex() {", "  // Once per day, backfill shadow portfolio outcomes.");
const streamFn = slice("function startStreaming()", "function stopStreaming()");

test("bot.ts: one halt-state builder over the raw flags, incl. the owner stop", () => {
  assert.ok(/import \{[^}]*haltAllows[^}]*\} from "\.\/haltScope";/.test(bot));
  const fn = slice("function currentHaltState(): HaltState {", "function startBotActivity(");
  for (const k of ["ownerStopped", "active: state.active", "killSwitch: state.killSwitch", "circuitBreakerUntil: state.circuitBreakerUntil"]) {
    assert.ok(fn.includes(k), `currentHaltState must pass ${k}`);
  }
});

test("bot.ts: checkPositionOnTick (WS stops/scale-outs/trails) is exit-gated, not kill-gated", () => {
  assert.ok(onTick.includes('if (!haltAllows(currentHaltState(), "exit")) return;'));
  assert.ok(!onTick.includes("state.killSwitch"), "the kill latch must not stop WS exits");
  assert.ok(!/!state\.active/.test(onTick), "state.active (which the drawdown kill clears) must not stop WS exits");
});

test("bot.ts: the 60s POS-KILL / TIME-EXIT sync is exit-gated", () => {
  assert.ok(syncInterval.includes('if (!haltAllows(currentHaltState(), "exit")) return;'));
  assert.ok(syncInterval.includes("await syncMonitoredPositions()"));
  assert.ok(!syncInterval.includes("state.killSwitch"));
});

test("bot.ts: Tier 1 runs the protective half while halted, before the auto-resume tick", () => {
  const mode = tier1Interval.indexOf('haltMode(t1Gate) === "exits_only"');
  const halted = tier1Interval.indexOf("await tier1HaltedExits(t1Gate)");
  const resume = tier1Interval.indexOf("if (state.killSwitch) { await killSwitchAutoResumeTick(); return; }");
  const breaker = tier1Interval.indexOf("if (state.circuitBreakerUntil > Date.now()) return;");
  const reflex = tier1Interval.indexOf("await tier1Reflex();");
  assert.ok(mode > 0 && halted > mode, "the exits-only branch must exist");
  assert.ok(resume > halted, "halted exits run even on a latched-kill tick (which then returns)");
  assert.ok(breaker > halted, "the circuit breaker must no longer skip the protective half");
  assert.ok(reflex > breaker, "tier1Reflex (with the morning queue) still never runs while halted");
  assert.ok(tier1Interval.includes("if (t1ExitsOnly) return;"));
  assert.ok(/clock\.is_open\) await tier1HaltedExits/.test(tier1Interval), "halted exits are market-hours-only, like tier1Reflex");
});

test("bot.ts: tier1HaltedExits runs the dd check unless latched, the sync and trade tracking — never an entry path", () => {
  assert.ok(haltedFn.includes('if (haltAllows(gate, "ddCheck"))'));
  assert.ok(haltedFn.indexOf("await checkDrawdownKillSwitch()") > haltedFn.indexOf('haltAllows(gate, "ddCheck")'));
  assert.ok(haltedFn.includes("await tier1ProtectiveSync()"));
  assert.ok(haltedFn.includes("await trackClosedTrades()"));
  for (const entryPath of ["executeMorningQueue", "executeTrades", "replaceIfBetter", "placeOptionsOrder"]) {
    assert.ok(!haltedFn.includes(entryPath), `halted Tier 1 must not reach ${entryPath}`);
  }
  for (const entryPath of ["executeMorningQueue", "executeTrades", "replaceIfBetter"]) {
    assert.ok(!protectiveFn.includes(entryPath), `the shared protective sync must not reach ${entryPath}`);
  }
});

test("bot.ts: tier1Reflex and the halted path share one protective sync", () => {
  assert.ok(protectiveFn.includes("await sweepStaleOrders()"));
  assert.ok(protectiveFn.includes("from bot_engine import manage_positions"));
  assert.ok(protectiveFn.includes("await syncMonitoredPositions()"));
  assert.ok(tier1Reflex.includes("await tier1ProtectiveSync()"));
  assert.ok(tier1Reflex.indexOf("await tier1ProtectiveSync()") < tier1Reflex.indexOf("if (clockT1.is_open"));
});

test("bot.ts: the WS feed reconnects through a halt; stream signals (entries) stay halt-gated", () => {
  assert.ok(streamFn.includes('setTimeout(() => { if (haltAllows(currentHaltState(), "exit")) startStreaming(); }, 10000);'));
  assert.ok(!streamFn.includes("if (state.active && !state.killSwitch) startStreaming()"));
  assert.ok(streamFn.includes('if (!haltAllows(currentHaltState(), "entry")) continue;'));
});

test("bot.ts: the kill trip, its order cancel and the -25% liquidate rule are untouched", () => {
  const dd = slice("async function checkDrawdownKillSwitch(", "// ── Morning Queue Execution");
  assert.ok(/if \(t1dd\.valid && t1dd\.kill && !state\.killSwitch\) \{\s*state\.killSwitch = true;\s*saveKillSwitch\("drawdown-kill \(tier1\)"\);\s*state\.active = false;/.test(dd));
  assert.ok(dd.includes('await alpaca("/v2/orders", { method: "DELETE" })'));
  assert.ok(/VOLTRADE_LIQUIDATE_ON_KILL/.test(dd) && /t1Drawdown <= -25\.0/.test(dd));
});

// ── Tier 2 (options_manager exits) while halted ─────────────────────────────
test("bot.ts: scheduleTier2 runs the options exits-only pass while halted, the full scan only in full mode", () => {
  const sched = slice("async function scheduleTier2() {", "let hitMutexTimeout = false;");
  const mode = sched.indexOf("const t2Mode = haltMode(currentHaltState());");
  const halted = sched.indexOf('if (t2Mode === "exits_only") {');
  const call = sched.indexOf("await tier2HaltedExits();");
  const full = sched.indexOf('if (t2Mode !== "full") {');
  assert.ok(mode > 0 && halted > mode && call > halted && full > call);
  assert.ok(!sched.includes("state.killSwitch"), "the kill latch must not skip the options exits");
  assert.ok(!sched.includes("state.circuitBreakerUntil"), "the circuit breaker must not skip the options exits");
});

test("bot.ts: tier2HaltedExits = options_manager with rolls OFF, regular hours, audited — never the scan", () => {
  const fn = slice("async function tier2HaltedExits(): Promise<void> {", "async function scheduleTier2() {");
  assert.ok(fn.includes('"manage_options_exits"'), "daemon route");
  assert.ok(fn.includes("{ allow_rolls: false }"), "daemon args disable rolls");
  assert.ok(fn.includes("manage_options_positions(allow_rolls=False)"), "subprocess fallback disables rolls");
  assert.ok(fn.indexOf("if (!clock.is_open) return;") < fn.indexOf("pythonCall("), "regular hours only");
  assert.ok(fn.includes("formatOptionsMgmt(r.result)"), "every halted exit is audited like a scan's");
  for (const entryPath of ["run_full_scan", "tier2Intelligence", "executeTrades", "select_contract", "placeOptionsOrder"]) {
    assert.ok(!fn.includes(entryPath), `the halted pass must not reach ${entryPath}`);
  }
  assert.ok(fn.includes("tier2Running = true;") && fn.includes("tier2Running = false;"), "shares the Tier 2 lock");
});

test("voltrade_daemon.py registers manage_options_exits -> options_manager.manage_options_positions", () => {
  const daemon = fs.readFileSync(path.join(here, "..", "voltrade_daemon.py"), "utf8");
  assert.ok(daemon.includes('"manage_options_exits": ("options_manager", "manage_options_positions")'));
});
