// [RULE-REVIEW] 2026-11-12 — the `gate_counterfactuals` diag probe
// (gate_counterfactuals_probe.py): what the tier engine's gates blocked or cut
// and what those dollars would have made. These pins are the wiring (token-gated
// whitelist, daemon first, python3 -c fallback, sanitized, read-only) and the one
// thing only this side can check: the report the Python module really produces
// survives sanitizeDiag unchanged (the sanitizer replaces anything deeper than
// 8 levels with "[depth]", caps arrays at 200 and strings at 2000).
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { DIAG_PROBES, sanitizeDiag } from "./diag";

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.join(here, "..");
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");

function probeBlock(): string {
  const i = bot.indexOf('case "gate_counterfactuals": {');
  assert.ok(i > 0, "gate_counterfactuals probe missing from bot.ts");
  const j = bot.indexOf("\n        case \"", i + 10);
  assert.ok(j > i, "end of the gate_counterfactuals case not found");
  return bot.slice(i, j);
}

test("gate_counterfactuals is whitelisted and served through pythonCall with the subprocess fallback", () => {
  assert.ok((DIAG_PROBES as readonly string[]).includes("gate_counterfactuals"));
  const block = probeBlock();
  assert.match(block, /pythonCall\(\s*"gate_counterfactuals"/);
  assert.match(block, /from gate_counterfactuals_probe import gate_counterfactuals_report/);
  assert.match(block, /sanitizeDiag\(\{\s*probe: "gate_counterfactuals"/);
  const daemon = fs.readFileSync(path.join(repo, "voltrade_daemon.py"), "utf8");
  assert.match(daemon, /"gate_counterfactuals": \("gate_counterfactuals_probe", "gate_counterfactuals_report"\)/);
});

test("the probe is read-only: no broker call, no state, no file write, no query parameter", () => {
  const block = probeBlock();
  assert.doesNotMatch(block, /alpaca\(|fetch\(|writeFile|appendFile|state\.|req\.query|audit\(/);
  // a daemon-only method would lose the fallback; this one must stay light enough for a subprocess
  const heavy = bot.slice(bot.indexOf("const HEAVY_DAEMON_ONLY"), bot.indexOf("]);", bot.indexOf("const HEAVY_DAEMON_ONLY")));
  assert.doesNotMatch(heavy, /gate_counterfactuals/);
});

test("the probe module imports the shadow log reader and nothing heavy (no bot_engine, no tier engine)", () => {
  const py = fs.readFileSync(path.join(repo, "gate_counterfactuals_probe.py"), "utf8");
  const imports = py.split("\n").filter((l) => /^\s*(import|from)\s/.test(l));
  assert.ok(imports.some((l) => /^import shadow_portfolio as sp$/.test(l)), "must read through shadow_portfolio");
  for (const l of imports) assert.doesNotMatch(l, /bot_engine|tiered_strategy|numpy|pandas|requests/);
});

// A small log in the shapes tiered_strategy writes: pre-change master-kill records (no gate
// block), a daily-loss kill's tier-1 / tier-2 / tier-3 rows, a scaled row, and scanner
// candidates for the base rate.
const PROGRAM = `
import json
import gate_counterfactuals_probe as g

def rec(ticker, day, decision, reason, ret, exit_date, gate=None):
    why = "target" if ret >= 2 else ("stop" if ret <= -4 else "timeout")
    r = {"ticker": ticker, "timestamp": day + "T14:00:00+00:00", "score": 0.0, "decision": decision,
         "decision_reason": reason, "entry_price": 50.0, "features": {},
         "outcomes": {"+5d": {"return_pct": ret, "label": 1 if ret >= 2 else 0, "exit_date": exit_date,
                              "exit_reason": why}, "+10d": None, "+20d": None}}
    if gate:
        r["gate"] = gate
    return r

CEIL = "Portfolio invested 96.0% >= 95% (regime=BULL)"
DAILY = "Daily loss limit: -3.5% <= -3%"
blk = lambda tier, action, cut, net: {"v": 1, "tier": tier, "action": action, "regime": "NEUTRAL", "cut_dollars": cut,
                                      "net_dollars": net, "t2": "levered", "t3": "proposed", "also": ["cash_reserve"]}
names = ["XA", "XB", "XC", "XD", "XE", "XF"]
log = [rec(n, "2026-10-05", "rejected_masterkill", CEIL, r, "2026-10-07")
       for n, r in zip(names, [2.0, -4.0, 2.0, 1.0, -4.0, 2.0])]
log += [rec(n, "2026-10-08", "rejected_masterkill", DAILY, r, "2026-10-12", blk(1, "dropped", 8000.0, 4000.0))
        for n, r in zip(names, [2.0, -4.0, 2.0, 1.0, -4.0, 2.0])]
log += [rec("XA", "2026-10-08", "rejected_masterkill", DAILY, 2.0, "2026-10-12", blk(2, "dropped", 8000.0, 4000.0)),
        rec("XQ", "2026-10-08", "rejected_masterkill", DAILY, 1.0, "2026-10-15", blk(3, "dropped", 20000.0, 10000.0)),
        rec("XG", "2026-10-08", "rejected_cash_reserve", "cash reserve leaves room for 88% of the proposed size",
            2.0, "2026-10-09", blk(1, "scaled", 1000.0, 1000.0))]
log += [rec("Z%d" % i, d, "taken", "", r, "2026-10-12")
        for d in ("2026-10-05", "2026-10-08") for i, r in enumerate([2.0, -4.0, 2.0, -4.0, 1.0, 2.0])]
print(json.dumps(g.build_report(log)))
`;

test("the report gate_counterfactuals_probe really produces passes sanitizeDiag unchanged", () => {
  const report = JSON.parse(execFileSync("python3", ["-c", PROGRAM], { cwd: repo, encoding: "utf8" }));
  const wire = { probe: "gate_counterfactuals", via: "daemon", ...report };
  assert.deepEqual(sanitizeDiag(wire), wire);      // nothing cut to "[depth]", truncated or redacted

  // and the numbers a session reads are where the sanitizer can reach them
  const daily = wire.gates.daily_loss;
  assert.deepEqual(daily.groups.map((x: { tier: string; action: string }) => [x.tier, x.action]),
    [["1", "dropped"], ["2", "dropped"], ["3", "dropped"]]);
  const t1 = daily.groups[0]["+5d"];
  assert.equal(t1.labeled, 6);
  assert.equal(t1.win_rate_pct, 50);
  assert.equal(t1.return_pct.mean, -0.167);
  assert.deepEqual(t1.base_taken_same_days, { n: 6, win_rate_pct: 50, mean_return_pct: -0.167, median_return_pct: 1.5 });
  assert.equal(t1.dollars.cut_dollars, 48000);
  assert.equal(t1.dollars.would_have_made_dollars, -80);            // 8000 x (2 - 4 + 2 + 1 - 4 + 2)%
  assert.equal(t1.dollars.prevention_pnl_dollars, 80);
  assert.equal(t1.dollars.net_prevention_pnl_dollars, 40);
  assert.equal(daily.dollars_all_tiers["+5d"].cut_dollars, 76000);
  // the deepest numbers in the report: a group's per-regime cell at one horizon (depth 8, the sanitizer's limit)
  assert.equal(daily.groups[0].by_regime.NEUTRAL["+5d"].prevention_pnl_dollars, 80);
  assert.equal(wire.gates.exposure_ceiling.groups[0].by_regime.BULL["+5d"].win_rate_pct, 50);
  assert.equal(wire.gates.exposure_ceiling.groups[0]["+5d"].dollars.unsized_positions, 6);   // pre-change records: no size
  assert.equal(wire.gates.cash_reserve.groups[0].action, "scaled");
  assert.match(wire.dollar_basis, /^ESTIMATE/);
  for (const name of ["XA", "XQ", "Z0"]) assert.ok(!JSON.stringify(wire).includes(`"${name}"`), "no ticker leaves the probe");
});
