// safetyLimitsWiring.test.ts — how server/bot.ts serves the owner's page the
// safety limits ([PRODUCT] 2026-11-20, UI coverage batch A, third part).
// Source pins, like ownerEvidenceWiring.test.ts: the route is an owner-gated
// GET that changes nothing; its `node` block is the `kill_state` probe's, to
// the character; the engine half is safety_limits_view.safety_limits through
// the daemon with a subprocess fallback naming the same function; and it does
// not serve get_kill_switch_status (the daemon's "risk_status"), whose streak
// and day-loss figures come from two files nothing writes.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const read = (rel: string) => fs.readFileSync(path.join(here, rel), "utf8");
const bot = read("bot.ts");

const a = bot.indexOf("// ── SAFETY LIMITS FOR THE OWNER");
const b = bot.indexOf("// ── OWNER VIEWS OF THE EVIDENCE PROBES");
assert.ok(a > 0 && b > a, "the safety-limits block sits just above the evidence block");
const BLOCK = bot.slice(a, b);
const probeStart = bot.indexOf('case "kill_state": {');
const PROBE = bot.slice(probeStart, bot.indexOf('\n        case "', probeStart + 10));

test("one owner-gated GET; nothing in the block writes, resets, audits or sends an order", () => {
  assert.ok(BLOCK.includes('app.get("/api/bot/safety-limits", requireOwner,'));
  assert.equal((BLOCK.match(/app\.(get|post|put|patch|delete)\(/g) ?? []).length, 1);
  assert.ok(!/app\.(post|put|patch|delete)\(/.test(BLOCK));
  assert.ok(!/method:|writeFileSync|appendFileSync|audit\(|notify\(|reset_kill_state|set_peak_equity|state\.\w+\s*=[^=]/.test(BLOCK), "read-only: no write, no reset, no state assignment");
  // the only broker call is the account read (a GET)
  assert.deepEqual(BLOCK.match(/alpaca\([^)]*\)/g), ['alpaca("/v2/account")']);
});

test("the page's copy is memoised: one computation per 30 s, one in flight", () => {
  assert.ok(BLOCK.includes('await ownerView("limits", 30_000, async () => {'));
  assert.ok(BLOCK.includes('ownerViewError(res, "/api/bot/safety-limits", e)'));
});

test("`node` is the kill_state probe's block, and both measure every guard from the same validated equity", () => {
  const equityRead = "const dd = evaluateDrawdown(acct?.equity, state.equityPeak, state.maxDrawdownPct);";
  const valid = "acctEq = dd.valid ? Number(acct.equity) : null;";
  const fields = [
    "killSwitch: state.killSwitch,",
    "reason: state.killSwitch ? killSwitchMeta.reason : null,",
    "equityPeak: state.equityPeak,",
    "maxDrawdownPct: state.maxDrawdownPct,",
    "equity: acctEq,",
    "drawdownPct: acctEq !== null && state.equityPeak > 0 ? +((acctEq - state.equityPeak) / state.equityPeak * 100).toFixed(2) : null,",
  ];
  for (const src of [BLOCK, PROBE]) {
    assert.ok(src.includes(equityRead) && src.includes(valid));
    for (const f of fields) assert.ok(src.includes(f), `missing: ${f}`);
  }
});

test("the engine half: safety_limits_view through the daemon, the same function in the fallback, equity passed to both", () => {
  assert.ok(BLOCK.includes('"safety_limits",\n          { equity: acctEq },'));
  assert.ok(BLOCK.includes("from safety_limits_view import safety_limits; print(json.dumps(safety_limits(equity=${eqArg}), default=str))"));
  assert.ok(BLOCK.includes('const eqArg = acctEq === null ? "None" : String(acctEq);'));
  assert.ok(read("../voltrade_daemon.py").includes('"safety_limits": ("safety_limits_view", "safety_limits"),'));
  // a failed engine call is said in the body; the loop's own kill is still served
  assert.ok(BLOCK.includes("...(r.success ? { engine: r.result } : { engine_error:"));
});

test("it does not serve the status function whose streak and day-loss figures are never written", () => {
  assert.ok(!/["'`]risk_status["'`]/.test(BLOCK) && !/get_kill_switch_status\(/.test(BLOCK));
  // the finding itself, pinned at the source: the function reads two files and no code in the repository writes them
  const rks = read("../risk_kill_switch.py");
  assert.ok(rks.includes('_state_path("consec_losses.json")') && rks.includes('_state_path("daily_pnl.json")'));
  const view = read("../safety_limits_view.py");
  assert.ok(view.includes("ksp.kill_state_report(") && view.includes("rks.KILLSWITCH_PATH"));
  assert.ok(!/open\([^)]*["']w["']|json\.dump\(|os\.replace|os\.remove|\.write\(/.test(view), "the view writes nothing");
});
