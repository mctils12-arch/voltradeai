// DD-HALT VISIBILITY (2026-09-09): live production audit logs (this
// session's own routine health check) showed TIER2 auditing "Scanned 0
// stocks, 0 trade candidates" every cycle for 100+ minutes straight during
// market hours, with no accompanying explanation anywhere in the audit
// trail. Traced the call graph: bot_engine.py's scan_market() has its own
// portfolio-level drawdown halt (update_equity_peak/is_trading_halted,
// DRAWDOWN_HALT_PCT=18% by default) that returns early with
// {halted, halt_reason, peak_equity, current_equity, dd_pct} whenever it
// fires — bot.ts read result.scanned and result.new_trades off that same
// return value but never result.halted, so a halt firing here produced the
// exact same "Scanned 0 stocks, 0 trade candidates" line as a normal scan
// that simply found nothing. This is the same visibility gap KNOWN BROKEN
// #3 already found and closed for the tier engine's own separate
// master_kill_switch (TIER-KILL) — this pins the same fix for scan_market's
// own DD halt.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");

function tier2ScanBlock(): string {
  const scanLineIdx = bot.indexOf('audit("TIER2", `Scanned ${result.scanned || 0} stocks,');
  assert.ok(scanLineIdx > 0, "tier2 scan-result audit line not found in bot.ts");
  const dataSourceErrorsIdx = bot.indexOf("tier2LastDataSourceErrors", scanLineIdx);
  assert.ok(dataSourceErrorsIdx > scanLineIdx, "tier2LastDataSourceErrors assignment not found after the scan audit line");
  return bot.slice(scanLineIdx, dataSourceErrorsIdx);
}

test("wiring pinned: the tier2 scan result's halted field is checked right after the Scanned-stocks audit line", () => {
  const block = tier2ScanBlock();
  assert.ok(/if\s*\(result\.halted\)/.test(block), "must check result.halted — bot_engine.py's scan_market() sets this on its own DD-halt early return");
});

test("wiring pinned: a halted scan result is audited under a distinct DD-HALT type, not silently folded into TIER2", () => {
  const block = tier2ScanBlock();
  const haltedIdx = block.search(/if\s*\(result\.halted\)/);
  assert.ok(haltedIdx > 0, "result.halted check not found");
  const haltedBranch = block.slice(haltedIdx);
  assert.ok(haltedBranch.includes('audit("DD-HALT"'), "must audit a distinct DD-HALT action type, mirroring the existing TIER-KILL pattern for the tier engine's own separate kill switch");
});

test("wiring pinned: the DD-HALT audit line surfaces the halt reason, current equity, peak equity, and dd_pct — not just the fact that it halted", () => {
  const block = tier2ScanBlock();
  const haltedIdx = block.search(/if\s*\(result\.halted\)/);
  const haltedBranch = block.slice(haltedIdx);
  assert.ok(haltedBranch.includes("result.halt_reason"), "must surface bot_engine.py's own halt_reason string (already includes peak/cur numbers, but the raw fields below let a probe parse them without re-deriving from prose)");
  assert.ok(haltedBranch.includes("result.current_equity"), "must surface the equity value the halt evaluated");
  assert.ok(haltedBranch.includes("result.peak_equity"), "must surface the peak the halt compared against");
  assert.ok(haltedBranch.includes("result.dd_pct"), "must surface the computed drawdown percentage directly, not force a future session to recompute it from equity/peak");
});

test("wiring pinned: the DD-HALT check does not replace or gate the existing Scanned-stocks TIER2 audit line", () => {
  // A prior draft could have made the DD-HALT branch exclusive with (or placed
  // before) the existing TIER2 audit — guard against that regression, since
  // dashboards/scripts already filter on TIER2 appearing every scan cycle.
  const scanLineIdx = bot.indexOf('audit("TIER2", `Scanned ${result.scanned || 0} stocks,');
  const haltedIdx = bot.indexOf("if (result.halted)", scanLineIdx);
  assert.ok(scanLineIdx > 0 && haltedIdx > scanLineIdx, "the TIER2 scan-result audit must still run, unconditionally, before the halted check");
});
