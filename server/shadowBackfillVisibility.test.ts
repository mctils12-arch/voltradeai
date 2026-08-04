// REPAIR 2026-08-04 (scheduled-routine session): the nightly
// shadow_portfolio.backfill_outcomes() job (bot.ts tier1Reflex, fires
// once/day at 10pm UTC) has logged only to console since inception —
// /api/diag/shadow (built 2026-08-03) showed 12,048 shadow records
// spanning 2026-04-20 through present with ZERO outcomes labeled at any
// of +5d/+10d/+20d, and there was no queryable trail to tell whether the
// job was erroring every night, running with 0 updates every night, or
// silently never firing. This pins that both the success path (parsed
// stats dict) and the failure path (parse error or subprocess rejection)
// now route through the persisted audit() log as SHADOW-BACKFILL /
// SHADOW-BACKFILL-ERROR, mirroring the TIER-KILL / TIER3-DIAG visibility
// precedents (see open_questions.md KNOWN BROKEN #3, #20 and
// tier3DiagVisibility.test.ts).
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");

function shadowBackfillBlock(): string {
  const start = bot.indexOf("Once per day, backfill shadow portfolio outcomes.");
  assert.ok(start > 0, "shadow backfill cron block not found in bot.ts");
  const end = bot.indexOf("if (nowHour === 0) {", start);
  assert.ok(end > start, "shadow backfill block did not end where expected");
  return bot.slice(start, end);
}

test("wiring pinned: shadow backfill success path audits the parsed stats dict", () => {
  const block = shadowBackfillBlock();
  assert.ok(block.includes("JSON.parse(result.stdout.trim())"), "must parse the backfill_outcomes() stdout");
  assert.ok(block.includes('audit("SHADOW-BACKFILL"'), "success path must route through the persisted audit log");
  assert.ok(block.includes("JSON.stringify(stats)"), "audited detail must carry the real stats dict, not a placeholder");
});

test("wiring pinned: shadow backfill failure paths (bad JSON and subprocess rejection) both audit an error", () => {
  const block = shadowBackfillBlock();
  const errorAudits = block.match(/audit\("SHADOW-BACKFILL-ERROR"/g) || [];
  assert.equal(errorAudits.length, 2, "both the JSON.parse catch and the outer .catch() must audit SHADOW-BACKFILL-ERROR");
});

test("wiring pinned: the console-only logging this replaces is still present for local debugging, not removed", () => {
  const block = shadowBackfillBlock();
  assert.ok(block.includes("console.log(\"[SHADOW] Backfill result:\""));
  assert.ok(block.includes("console.error(\"[SHADOW] Backfill failed:\""));
});

test("SHADOW-BACKFILL-ERROR detail-building never throws on a non-Error rejection value", () => {
  // Direct behavioral check of the exact expression bot.ts uses for the
  // outer .catch(), since bot.ts itself isn't importable (side-effecting
  // orchestrator) — mirrors the pattern in tier3DiagVisibility.test.ts.
  const buildDetail = (e: any) => String(e?.message || e).slice(0, 500);

  assert.equal(buildDetail(new Error("boom")), "boom");
  assert.equal(buildDetail("plain string failure"), "plain string failure");
  assert.equal(buildDetail({ code: "ETIMEDOUT" }), "[object Object]");
  assert.equal(buildDetail(undefined), "undefined");
});
