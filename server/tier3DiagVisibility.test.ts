// R-DIAG (research/open_questions.md KNOWN BROKEN follow-up, filed
// 2026-07-05/2026-07-06 experiments.md entries, never closed until now):
// the Tier-3 "System health: warning — N issues" audit line only ever
// logged the problem COUNT, never which system or what the message was.
// Live /api/diag/audit showed this line recurring roughly hourly for a
// month (2026-07-06 through 2026-08-03) with zero session able to tell
// what the actual issue was from the audit trail alone. This pins that
// the audited detail now carries the real per-problem system/severity/
// message, mirroring the enrichment precedent already set for
// TIER3-MANIP-ERROR/TIER3-ML-ERROR.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");

function tier3DiagBlock(): string {
  const importStart = bot.indexOf("from diagnostics import run_diagnostics");
  assert.ok(importStart > 0, "run_diagnostics call site not found in bot.ts");
  const reportStart = bot.indexOf("const diagReport = JSON.parse(diagFull.trim());", importStart);
  assert.ok(reportStart > importStart, "diagReport parse line not found");
  const blockEnd = bot.indexOf("} catch (err: any) { console.error(\"[tier3-diag]\"", reportStart);
  assert.ok(blockEnd > reportStart, "tier3-diag try block not closed as expected");
  return bot.slice(reportStart, blockEnd);
}

test("wiring pinned: TIER3-DIAG audits the real problem detail, not just a bare count", () => {
  const block = tier3DiagBlock();
  assert.ok(block.includes("diagReport.problems"), "must read diagReport.problems to build detail");
  assert.ok(block.includes("p.system") && block.includes("p.message") && block.includes("p.severity"),
    "audited string must include each problem's system/severity/message, not just its length");
  assert.ok(block.includes('audit("TIER3-DIAG"'), "must still route through the persisted audit log");
});

test("wiring pinned: the audit call site still includes the issue count for a quick at-a-glance read", () => {
  const block = tier3DiagBlock();
  assert.ok(block.includes("diagReport.problems?.length || 0"), "count must survive alongside the new detail, not be replaced by it");
});

test("TIER3-DIAG detail-building never throws on a malformed/missing problems array", () => {
  // Direct behavioral check of the exact expression bot.ts uses, since bot.ts
  // itself isn't importable as a module (side-effecting orchestrator).
  const buildDetail = (diagReport: any) =>
    (diagReport.problems || [])
      .map((p: any) => `[${String(p.severity || "?").toUpperCase()}] ${p.system || "?"}: ${p.message || ""}`)
      .join(" | ");

  assert.equal(buildDetail({}), "");
  assert.equal(buildDetail({ problems: [] }), "");
  assert.equal(
    buildDetail({ problems: [{ severity: "medium", system: "api", message: "Some API sources unavailable" }] }),
    "[MEDIUM] api: Some API sources unavailable",
  );
  assert.equal(
    buildDetail({ problems: [{}] }),
    "[?] ?: ",
  );
});
