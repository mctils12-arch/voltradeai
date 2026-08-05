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
//
// REPAIR 2026-08-05 (root cause of the zero-labeled-records symptom the
// 2026-08-04 visibility fix above could only observe, not explain): the
// nowHour===22 check lived inside tier1Reflex(), and the TIER 1
// setInterval that invokes tier1Reflex() returns early whenever
// `!clock.is_open`. NYSE regular hours end by 21:00 UTC even in EST
// (20:00 UTC in EDT) — 22:00 UTC is therefore NEVER inside market hours,
// so the whole nowHour===22 branch (and the nowHour===0 reset) was
// unreachable dead code since inception; the job never actually ran, on
// any night, ever. Fixed by extracting the check into its own
// checkShadowBackfill() function driven by an unconditional setInterval
// (no clock.is_open, state.active, or killSwitch gate) alongside the
// EVENTLOOP-LAG interval. The tests below pin both halves of the fix:
// the function is defined outside tier1Reflex's body, and it is wired to
// an interval that does not check clock.is_open first.
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

test("regression: checkShadowBackfill is NOT nested inside tier1Reflex (which is market-hours-gated)", () => {
  const tier1Start = bot.indexOf("async function tier1Reflex()");
  assert.ok(tier1Start > 0, "tier1Reflex not found");
  const tier1End = bot.indexOf("\n  }\n\n  // Once per day, backfill shadow portfolio outcomes.", tier1Start);
  assert.ok(tier1End > tier1Start, "tier1Reflex's closing brace / handoff comment not found where expected");
  const tier1Body = bot.slice(tier1Start, tier1End);
  assert.ok(
    !tier1Body.includes("backfill_outcomes"),
    "the shadow backfill job must not be inlined inside tier1Reflex — tier1Reflex only runs when clock.is_open, " +
    "and 22:00 UTC (the job's fire hour) is never inside NYSE market hours, so it would never run (KNOWN BROKEN #10/#20 root cause)"
  );
  assert.ok(bot.includes("async function checkShadowBackfill()"), "checkShadowBackfill must be its own top-level function");
});

test("regression: checkShadowBackfill is driven by an interval with no clock.is_open / market-hours gate", () => {
  const callSite = bot.indexOf("checkShadowBackfill().catch(");
  assert.ok(callSite > 0, "checkShadowBackfill() must be called from a scheduled interval");
  // The call must be the interval's own callback body, not gated behind an
  // await'd market-clock check the way the TIER 1 setInterval gates tier1Reflex().
  const lineStart = bot.lastIndexOf("\n", callSite);
  const line = bot.slice(lineStart, bot.indexOf("\n", callSite));
  assert.ok(!line.includes("is_open"), "the shadow-backfill interval callback must not be gated on clock.is_open");
  assert.ok(!line.includes("state.killSwitch"), "the shadow-backfill interval callback must not be gated on killSwitch");
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
