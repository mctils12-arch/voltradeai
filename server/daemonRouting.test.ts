// A6 DAEMON ROUTING (2026-11-03, memory audit fix A6). server/daemonRouting.ts
// is the pure half; this pins it, pins bot.ts's wiring of it, and proves the
// identical-output claim across the language boundary: the same
// manage_positions() result reaches bot.ts byte-for-byte through the daemon's
// RPC envelope and through the fresh-interpreter one-liner it replaces.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import {
  TIER1_VIA_DAEMON_ENV, RESEARCH_VIA_DAEMON_ENV, TIER1_SYNC_TIMEOUT_MS, RESEARCH_ANALYZE_TIMEOUT_MS,
  envFlag, tier1ViaDaemon, researchViaDaemon, classifyRpc, rpcMayFallBack, posMonitorSyncLines, viaTransitionLine,
} from "./daemonRouting";

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.join(here, "..");
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
const daemonPy = fs.readFileSync(path.join(repo, "voltrade_daemon.py"), "utf8");
const analyzePy = fs.readFileSync(path.join(repo, "analyze.py"), "utf8");

function slice(start: string, end: string, from = bot): string {
  const a = from.indexOf(start);
  assert.ok(a >= 0, `anchor not found: ${start.slice(0, 60)}`);
  const b = from.indexOf(end, a + start.length);
  assert.ok(b > a, `end anchor not found after start: ${end.slice(0, 60)}`);
  return from.slice(a, b);
}

// ── flags ───────────────────────────────────────────────────────────────────
test("both flags default ON; 0/false/off/no turn a path off; junk keeps the default", () => {
  assert.equal(tier1ViaDaemon({}), true);
  assert.equal(researchViaDaemon({}), true);
  for (const off of ["0", "false", "OFF", " no "]) {
    assert.equal(tier1ViaDaemon({ [TIER1_VIA_DAEMON_ENV]: off }), false, off);
    assert.equal(researchViaDaemon({ [RESEARCH_VIA_DAEMON_ENV]: off }), false, off);
  }
  for (const on of ["1", "true", "On", "yes"]) assert.equal(tier1ViaDaemon({ [TIER1_VIA_DAEMON_ENV]: on }), true, on);
  assert.equal(envFlag({ X: "maybe" }, "X", true), true);
  assert.equal(envFlag({ X: "maybe" }, "X", false), false);
  assert.equal(envFlag({ X: "" }, "X", false), false);
  // the two flags are independent
  assert.equal(researchViaDaemon({ [TIER1_VIA_DAEMON_ENV]: "0" }), true);
});

test("the budgets are exactly what the fresh-interpreter calls always had (Tier 1: 15 s, analyze: 30 s)", () => {
  assert.equal(TIER1_SYNC_TIMEOUT_MS, 15_000);
  assert.equal(RESEARCH_ANALYZE_TIMEOUT_MS, 30_000);
});

// ── classifyRpc: which envelopes may fall back to a fresh interpreter ────────
test("classifyRpc: ok envelopes hand back result; only never-ran outcomes may fall back", () => {
  const ok = classifyRpc({ status: "ok", result: { actions: [] } });
  assert.deepEqual(ok, { kind: "ok", result: { actions: [] } });
  assert.equal(rpcMayFallBack(ok), false);

  // pythonRpc's own: socket down / daemon disabled -> unreachable -> fallback
  for (const m of ["Socket: connect ENOENT /tmp/voltrade_daemon.sock", "Socket: connect ECONNREFUSED", "Daemon disabled via env"]) {
    const o = classifyRpc({ status: "error", error_message: m });
    assert.equal(o.kind, "unreachable", m);
    assert.equal(rpcMayFallBack(o), true, m);
  }
  // the daemon declined before running anything -> rejected -> fallback
  for (const m of [
    "Daemon busy: >8 in-flight requests",
    "Daemon thread creation failed: can't start new thread",
    "Unknown method: manage_positions",
    "Module bot_engine not loaded",
    "Method manage_positions not found in bot_engine",
  ]) {
    const o = classifyRpc({ status: "error", error_message: m });
    assert.equal(o.kind, "rejected", m);
    assert.equal(rpcMayFallBack(o), true, m);
  }
  // the dispatch may still be running -> timeout -> NEVER a second run
  for (const m of ["Daemon timeout", "Request timed out after 15s", "Request timed out after 300s"]) {
    const o = classifyRpc({ status: "error", error_message: m });
    assert.equal(o.kind, "timeout", m);
    assert.equal(rpcMayFallBack(o), false, m);
  }
  // the method ran and raised (dispatch() attaches a traceback) -> error -> no rerun
  const raised = classifyRpc({ status: "error", error_message: "KeyError: 'qty'", traceback: "Traceback (most recent call last) ..." });
  assert.equal(raised.kind, "error");
  assert.equal(rpcMayFallBack(raised), false);
  // a raised exception whose text happens to start like a routing error still carries a traceback -> error
  const lookalike = classifyRpc({ status: "error", error_message: "Unknown method: inside the method body", traceback: "Traceback ..." });
  assert.equal(lookalike.kind, "error");
  // unparseable daemon output -> error (the method ran; the fallback would print the same thing)
  assert.equal(classifyRpc({ status: "error", error_message: "Parse: Unexpected token N" }).kind, "error");
  // malformed envelopes never throw and never fall back
  for (const weird of [null, undefined, 42, "ok", {}, { status: "error" }]) {
    const o = classifyRpc(weird);
    assert.equal(o.kind, "error", JSON.stringify(weird));
    assert.equal(rpcMayFallBack(o), false);
  }
});

// ── the Tier-1 consumer, fed the same JSON through both paths ───────────────
const FIXTURE = {
  actions: [
    { ticker: "NVDA", type: "TRAIL_STOP", price: 181.25, reason: "Phase 4 trail — 3.0× ATR" },
    { ticker: "AMD", type: "SCALE_OUT", qty: 3 },
    { ticker: "GLD", type: "TIME_EXIT" },
  ],
  positions: 7,
  upgrade_candidates: [],
  regime: "NEUTRAL_BULL",
};

test("posMonitorSyncLines reproduces the pre-A6 inline loop over the parsed stdout and the daemon envelope alike", () => {
  const monitored = (t: string) => t === "NVDA" || t === "GLD";
  // (a) what the subprocess path hands bot.ts: JSON.parse(stdout) of python's json.dumps(result)
  const viaSubprocess = JSON.parse(JSON.stringify(FIXTURE));
  // (b) what the daemon path hands bot.ts: the envelope's .result
  const viaDaemon = classifyRpc({ status: "ok", result: JSON.parse(JSON.stringify(FIXTURE)) });
  assert.equal(viaDaemon.kind, "ok");
  const expected = [
    "bot_engine flagged NVDA (TRAIL_STOP) — WS monitor should handle",
    "bot_engine flagged GLD (TIME_EXIT) — WS monitor should handle",
  ];
  assert.deepEqual(posMonitorSyncLines(viaSubprocess, monitored), expected);
  assert.deepEqual(posMonitorSyncLines(viaDaemon.kind === "ok" ? viaDaemon.result : null, monitored), expected);
  // the pre-A6 loop: `for (const action of (mgmt.actions || []))` — missing/odd shapes audit nothing
  assert.deepEqual(posMonitorSyncLines({ actions: [], error: "Could not fetch positions" }, monitored), []);
  assert.deepEqual(posMonitorSyncLines({}, monitored), []);
  assert.deepEqual(posMonitorSyncLines(null, monitored), []);
  assert.deepEqual(posMonitorSyncLines({ actions: [null, { type: "X" }, { ticker: "NVDA" }] }, monitored),
    ["bot_engine flagged NVDA (undefined) — WS monitor should handle"]);
});

test("cross-language: the daemon envelope's result equals what the Tier-1 one-liner prints, for the same manage_positions()", () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "a6-parity-"));
  try {
    // A stub bot_engine that shadows the real one: both paths import `bot_engine`
    // and call manage_positions(); only the transport differs.
    const fixtureJson = JSON.stringify({
      ...FIXTURE,
      actions: [...FIXTURE.actions, { ticker: "TSLA", type: "BREAKEVEN", note: "em dash — and unicode ✓", frac: 0.1 + 0.2, none: null }],
    });
    fs.writeFileSync(path.join(tmp, "bot_engine.py"),
      `import json\n_FIXTURE = json.loads(${JSON.stringify(fixtureJson)})\ndef manage_positions():\n    return _FIXTURE\n`);

    // (a) the exact one-liner bot.ts runs as the fallback, sliced from bot.ts so it cannot drift
    const cmdBlock = slice('const tier1SyncCmd = `python3 -c "', '"`;');
    const program = cmdBlock.slice('const tier1SyncCmd = `python3 -c "'.length);
    assert.ok(program.includes("from bot_engine import manage_positions"), "the fallback is still the bot_engine one-liner");
    const stdout = execFileSync("python3", ["-c", program], { cwd: tmp, encoding: "utf8", env: { ...process.env, PYTHONPATH: tmp } });

    // (b) the daemon's dispatch(), with the stub first on sys.path
    const daemonProgram = [
      "import sys, json",
      `sys.path.insert(0, ${JSON.stringify(repo)})`,
      `sys.path.insert(0, ${JSON.stringify(tmp)})`,
      "import voltrade_daemon",
      'print(json.dumps(voltrade_daemon._dispatcher.dispatch("manage_positions", {})))',
    ].join("\n");
    const envelopeOut = execFileSync("python3", ["-c", daemonProgram], {
      cwd: tmp, encoding: "utf8", stdio: ["ignore", "pipe", "ignore"], env: { ...process.env, PYTHONPATH: tmp },
    });
    const envelope = JSON.parse(envelopeOut.trim());
    assert.equal(envelope.status, "ok");
    const fromSubprocess = JSON.parse(stdout.trim());
    const fromDaemon = classifyRpc(envelope);
    assert.equal(fromDaemon.kind, "ok");
    assert.deepEqual(fromDaemon.kind === "ok" ? fromDaemon.result : null, fromSubprocess, "identical shape through both paths");
    assert.deepEqual(fromSubprocess, JSON.parse(fixtureJson));
    const monitored = (t: string) => t === "TSLA" || t === "AMD";
    assert.deepEqual(
      posMonitorSyncLines(fromDaemon.kind === "ok" ? fromDaemon.result : null, monitored),
      posMonitorSyncLines(fromSubprocess, monitored),
    );
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});

// ── the TIER1-SYNC transition line ──────────────────────────────────────────
test("viaTransitionLine: one line per path change, none per cycle; a daemon boot says nothing", () => {
  assert.equal(viaTransitionLine(null, "daemon"), null);
  assert.equal(viaTransitionLine("daemon", "daemon"), null);
  assert.equal(viaTransitionLine("subprocess", "subprocess", "unreachable: Socket: x"), null);
  assert.match(viaTransitionLine(null, "subprocess", "unreachable: Socket: connect ENOENT")!, /running in a fresh interpreter \(unreachable: Socket: connect ENOENT\)/);
  assert.match(viaTransitionLine("daemon", "subprocess", `${TIER1_VIA_DAEMON_ENV}=0`)!, /fell back to a fresh interpreter \(VOLTRADE_TIER1_VIA_DAEMON=0\)/);
  assert.equal(viaTransitionLine("subprocess", "daemon"), "stop-state sync back on the daemon RPC");
});

// ── bot.ts wiring ───────────────────────────────────────────────────────────
const protectiveFn = slice("async function tier1ProtectiveSync()", "async function tier1HaltedExits(");

test("bot.ts: Tier 1's stop-state sync tries the daemon RPC first, with the 15 s budget, then the unchanged one-liner only when the daemon never ran it", () => {
  assert.ok(/import \{[^}]*classifyRpc[^}]*\} from "\.\/daemonRouting";/s.test(bot), "bot.ts imports the routing helpers");
  const rpcIdx = protectiveFn.indexOf('pythonRpc("manage_positions", {}, TIER1_SYNC_TIMEOUT_MS)');
  const gateIdx = protectiveFn.indexOf("tier1ViaDaemon() ?");
  const fallbackIdx = protectiveFn.indexOf("execPythonSerialized(tier1SyncCmd, { timeout: TIER1_SYNC_TIMEOUT_MS })");
  const noRerunIdx = protectiveFn.indexOf("if (rpc && !rpcMayFallBack(rpc)) throw new Error(");
  assert.ok(gateIdx > 0 && rpcIdx > gateIdx, "the RPC is behind the VOLTRADE_TIER1_VIA_DAEMON flag");
  assert.ok(fallbackIdx > rpcIdx, "the fresh interpreter is the fallback, after the RPC");
  assert.ok(noRerunIdx > rpcIdx && noRerunIdx < fallbackIdx, "a timeout / in-method error throws before any fallback could start a second run");
  assert.ok(protectiveFn.includes("from bot_engine import manage_positions"), "the fallback is still the bot_engine one-liner");
  assert.ok(!protectiveFn.includes('pythonCall("manage_positions"'), "not pythonCall: HEAVY_DAEMON_ONLY's blanket no-fallback is not the rule here");
  assert.ok(!protectiveFn.includes("{ timeout: 15000 }"), "no second copy of the budget — both paths read TIER1_SYNC_TIMEOUT_MS");
  assert.ok(protectiveFn.includes("posMonitorSyncLines(mgmt, (ticker) => Boolean(monitoredPositions[ticker]))"), "the consumer is the tested function");
  assert.ok(protectiveFn.includes('audit("TIER1-SYNC", viaLine)'), "path changes are audited");
  assert.ok(protectiveFn.includes("await syncMonitoredPositions()"), "step 3 unchanged");
  assert.ok(/let tier1SyncVia: SyncVia \| null = null;/.test(bot), "the transition state exists once");
});

test("bot.ts: pythonRpc sends its own deadline as timeout_sec so the daemon stops waiting when the client does", () => {
  const rpcFn = slice("async function pythonRpc(", "// pythonCall: try daemon first");
  assert.ok(rpcFn.includes("JSON.stringify({ method, args, timeout_sec: effectiveTimeout / 1000 })"));
  const handler = slice("    def handle(self):", "class ThreadingUnixServer", daemonPy);
  assert.ok(handler.includes('join_sec = _effective_join_sec(request.get("timeout_sec"))'), "and the daemon's handler honours it");
  assert.ok(handler.includes("t.join(join_sec)"));
  assert.ok(!handler.includes("t.join(REQUEST_TIMEOUT_SEC)"), "the fixed 300 s join is gone from the handler");
});

test("bot.ts: every per-ticker analyze.py spawn now goes through analyzeTickerJson (daemon first, same 30 s)", () => {
  const spawnSites = bot.match(/`python3 "\$\{path\.resolve\(['"]analyze\.py['"]\)\}" "\$\{\w+\}" --mode=scan`/g) ?? [];
  assert.equal(spawnSites.length, 1, "exactly one analyze.py --mode=scan command remains: the fallback inside analyzeTickerJson");
  const helper = slice("async function analyzeTickerJson(ticker: string)", "// Generate signals for a list of tickers");
  assert.ok(helper.includes("researchViaDaemon()"), "behind VOLTRADE_RESEARCH_VIA_DAEMON");
  assert.ok(helper.includes('pythonRpc("analyze_ticker", { symbol: ticker }, RESEARCH_ANALYZE_TIMEOUT_MS)'));
  assert.ok(helper.indexOf("if (!rpcMayFallBack(rpc)) throw new Error(") < helper.indexOf("execPythonSerialized("), "no second run after a timeout / error");
  assert.ok(helper.includes("{ timeout: RESEARCH_ANALYZE_TIMEOUT_MS }"));
  assert.ok(!helper.includes("30000"), "no second copy of the budget");
  const generate = slice("async function generateSignals(tickers: string[])", "// Signals endpoint");
  assert.ok(generate.includes("await analyzeTickerJson(ticker)"));
  assert.ok(!generate.includes("execPythonSerialized("), "generateSignals no longer spawns anything itself");
  const earnings = slice("// 12am-2am: Deep analyze earnings reporters for tomorrow", "// 2am-4am: Prepare pre-market report");
  assert.ok(earnings.includes("await analyzeTickerJson(t)"));
  assert.ok(!earnings.includes("execPythonSerialized("), "the earnings loop no longer spawns anything itself");
});

test("voltrade_daemon.py registers both routes against the real entry points; analyze.py's CLI prints the route's function", () => {
  assert.ok(daemonPy.includes('"manage_positions": ("bot_engine", "manage_positions"),'));
  assert.ok(daemonPy.includes('"analyze_ticker": ("analyze", "analyze_ticker_rpc"),'));
  assert.ok(analyzePy.includes("def analyze_ticker_rpc(symbol):"));
  const mainBlock = analyzePy.slice(analyzePy.indexOf('if __name__ == "__main__":'));
  assert.ok(mainBlock.includes("print(json.dumps(analyze_ticker_rpc(cmd)))"), "the CLI and the daemon route share one function");
});
