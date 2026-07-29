// OPTIONS-SLOT-CAP-CONSISTENCY (2026-07-29): live audit logs showed
// OPTIONS-SLOT-FULL firing for VXUS/KWEB at "(4/3)" while /api/diag/
// positions-detail confirmed the account correctly held 4 CSP options
// positions — within system_config.py's documented MAX_OPTIONS_POSITIONS
// budget of 6 (added 2026-07-11, KNOWN BROKEN #3 fix, explicitly "the
// single source of truth" for total open OPTIONS (CSP) positions, held
// constant across every regime). executeTrades()'s Tier-2-scanner options
// slot check in server/bot.ts carried its own independent, stale local
// constant of 3 — never updated when the 2026-07-11 fix established 6 as
// canonical — so the scanner path silently started blocking legitimate
// options trades a full 2 slots earlier than the rest of the system
// intends. This pins the fix (bot.ts's local constant now matches
// system_config.py's canonical value) and the two files staying in sync.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, "..");
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
const systemConfig = fs.readFileSync(path.join(repoRoot, "system_config.py"), "utf8");

function canonicalMaxOptionsPositions(): number {
  const m = systemConfig.match(/"MAX_OPTIONS_POSITIONS":\s*(\d+)/);
  assert.ok(m, "system_config.py must define MAX_OPTIONS_POSITIONS — it is documented there as the single source of truth");
  return Number(m![1]);
}

function botLocalMaxOptionsPositions(): number {
  const m = bot.match(/const MAX_OPTIONS_POSITIONS = (\d+);/);
  assert.ok(m, "executeTrades()'s local MAX_OPTIONS_POSITIONS constant not found in bot.ts");
  return Number(m![1]);
}

test("bot.ts's Tier-2-scanner options slot cap matches system_config.py's canonical MAX_OPTIONS_POSITIONS", () => {
  const canonical = canonicalMaxOptionsPositions();
  const local = botLocalMaxOptionsPositions();
  assert.equal(
    local, canonical,
    `bot.ts's executeTrades() options slot cap (${local}) has drifted from system_config.py's ` +
    `documented single source of truth (${canonical}) — the scanner-path options check will ` +
    `silently block legitimate trades below the system's real intended cap`,
  );
});

test("wiring pinned: the options slot check compares against the (now-corrected) local constant, not a re-hardcoded literal", () => {
  const checkIdx = bot.indexOf("if (optionsSlotsUsed >= MAX_OPTIONS_POSITIONS)");
  assert.ok(checkIdx > 0, "options slot cap check not found — did the check site move or get renamed?");
  const auditIdx = bot.indexOf('audit("OPTIONS-SLOT-FULL"', checkIdx);
  assert.ok(auditIdx > checkIdx && auditIdx < checkIdx + 200, "OPTIONS-SLOT-FULL audit call must immediately follow the slot-cap check");
});
