// STALE-ORDER-SWEEP FIX 2026-08-22 (KNOWN BROKEN, filed this session):
// `openOrders` — the TrackedOrder[] array sweepStaleOrders() (tier1Reflex,
// every ~45s) reads to cancel DAY limit orders unfilled for
// STALE_ORDER_MINUTES+ minutes — was never populated anywhere in this file,
// for ANY order type. The sweeper was fully dead code since its own
// introduction. Live evidence this session: two CSP DAY limit orders (NU,
// XLP) sat "accepted" for 2+ hours, both counted by
// countOpenOptionsOpeningOrders() into a real 6/6 OPTIONS-SLOT-FULL block on
// TLT/IBIT/CSX for the rest of the afternoon, with no mechanism able to free
// the squatted slots early.
//
// This pins the options-order half of the fix: the SELL_CSP and BUY_PUT tier
// dispatch branches now register the returned order_id with `openOrders` so
// the sweeper can actually act on a resting options entry. The general bug
// (every other order-submission call site in this file has the same gap) is
// filed as its own KNOWN BROKEN entry in research/open_questions.md for a
// dedicated follow-up session — this PR fixes the layer this session found
// live evidence for, per CLAUDE.md's ROOT VALIDATION LADDER discipline
// ("never debug the whole pipeline at once").
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");

function slice(fromMarker: string, toMarker: string): string {
  const start = bot.indexOf(fromMarker);
  assert.ok(start > 0, `marker not found: ${fromMarker}`);
  const end = bot.indexOf(toMarker, start);
  assert.ok(end > start, `end marker not found after start: ${toMarker}`);
  return bot.slice(start, end);
}

test("openOrders is populated somewhere in bot.ts (the sweeper was previously fed by nothing)", () => {
  const pushCalls = bot.match(/openOrders\.push\(/g) || [];
  assert.ok(
    pushCalls.length >= 2,
    "openOrders.push( must appear at least twice (SELL_CSP + BUY_PUT tier-dispatch branches) — zero occurrences means the stale-order sweeper has nothing to ever cancel",
  );
});

test("SELL_CSP tier dispatch registers a successful submission with openOrders", () => {
  const section = slice(
    'if (action.action === "SELL_CSP") {',
    "// ── TIER 3: BUY SPY/QQQ at 2x",
  );
  assert.ok(
    section.includes("openOrders.push("),
    "the SELL_CSP branch must push the returned order onto openOrders so sweepStaleOrders() can cancel it if it never fills",
  );
  assert.ok(
    /if\s*\(r\.order_id\)\s*\{[\s\S]*?openOrders\.push\(/.test(section),
    "the push must be gated on r.order_id actually being present (submit_options_order can return an error with no order_id)",
  );
  assert.ok(
    /orderId:\s*r\.order_id/.test(section) &&
      /ticker:\s*action\.ticker/.test(section) &&
      /placedAt:\s*Date\.now\(\)/.test(section),
    "the tracked entry must carry a real orderId, ticker, and placedAt timestamp — sweepStaleOrders() keys its staleness check off placedAt",
  );
  // the push must happen inside the success branch (submitted/filled), not unconditionally
  const pushIdx = section.indexOf("openOrders.push(");
  const successCheckIdx = section.indexOf('r.status === "submitted" || r.status === "filled"');
  assert.ok(successCheckIdx >= 0 && successCheckIdx < pushIdx, "the push must be inside the submitted/filled success branch");
});

test("BUY_PUT (tail hedge) tier dispatch registers a successful submission with openOrders", () => {
  const section = slice(
    '// ── TIER 4: BUY OTM SPY PUT (tail hedge) ───────────────────────────',
    'audit("TIER-DISPATCH-ERR"',
  );
  assert.ok(
    section.includes("openOrders.push("),
    "the BUY_PUT branch must push the returned order onto openOrders so sweepStaleOrders() can cancel it if it never fills",
  );
  assert.ok(
    /if\s*\(r\.order_id\)\s*\{[\s\S]*?openOrders\.push\(/.test(section),
    "the push must be gated on r.order_id actually being present",
  );
});

test("MAX_OPTIONS_POSITIONS live-slot counting (countOpenOptionsOpeningOrders) is untouched by this fix — it already queries Alpaca directly, not openOrders", () => {
  // Regression guard: this fix must not make the tier dispatcher's slot-cap
  // check depend on the (separately fed, in-process) openOrders array —
  // that check must keep reading live Alpaca state so it stays correct even
  // across a process restart that would reset openOrders to empty.
  const dispatcherSection = slice(
    "TIER DISPATCHER — routes tier_actions to the right execution path",
    "// ── TIER 3: BUY SPY/QQQ at 2x",
  );
  assert.ok(
    /tierOptionsSlotsUsed\s*=\s*countOptionsPositions\(/.test(dispatcherSection),
    "the slot-cap gate must still derive from a live Alpaca positions/orders fetch, not from openOrders",
  );
});
