// entryLatchRecheck.test.ts — Node's own opening orders re-read the kill latch
// right before they are sent (2026-11-12 [RULE-REVIEW], KNOWN BROKEN #51
// residual). Each of these paths checked state.killSwitch at its own start or
// loop top, then awaited Python or the broker (seconds) before its POST, so a
// latch that tripped in between let the entry through. The re-check sits after
// the last await before each POST.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const src = fs.readFileSync(path.join(path.dirname(fileURLToPath(import.meta.url)), "bot.ts"), "utf8");

function fnBody(name: string): string {
  const start = src.indexOf(`async function ${name}(`);
  assert.ok(start > 0, `${name} not found`);
  const next = src.indexOf("\n  async function ", start + 10);
  return src.slice(start, next > 0 ? next : undefined);
}

/** The text between the last latch check before `post` and the POST itself. */
function afterLastCheck(body: string, post: string): string {
  const at = body.indexOf(post);
  assert.ok(at > 0, `POST not found: ${post}`);
  const check = body.lastIndexOf("state.killSwitch", at);
  assert.ok(check > 0, `no latch check before ${post}`);
  return body.slice(check, at);
}

const OPENING_POSTS: Array<[fn: string, post: string]> = [
  ["executeMorningQueue", 'const order = await alpaca("/v2/orders", {'],
  ["tier2Intelligence", 'const t3OrderResult = await alpaca("/v2/orders", {'],
  ["runUpgradeCandidates", 'const upgradeOrderResult = await alpaca("/v2/orders", {'],
  ["executeScaleInTranches", 'const trancheOrder = await alpaca("/v2/orders", {'],
];

for (const [fn, post] of OPENING_POSTS) {
  test(`${fn}: no await between the last latch check and its opening order`, () => {
    const gap = afterLastCheck(fnBody(fn), post);
    assert.doesNotMatch(gap, /\bawait\b/, `${fn}: the latch can trip during an await after the check`);
  });
}

test("the morning queue and the tier dispatcher re-check with their loop-top handling", () => {
  const mq = fnBody("executeMorningQueue");
  assert.equal((mq.match(/if \(state\.killSwitch\) break;/g) || []).length, 2, "loop top + right before the order");
  const t2 = fnBody("tier2Intelligence");
  const t3 = t2.indexOf('const t3OrderResult = await alpaca("/v2/orders", {');
  const block = t2.slice(t2.lastIndexOf("} else if (shares > 0) {", t3), t3);
  assert.match(block, /if \(state\.killSwitch\) \{\s*for \(const rest of result\.tier_actions\.slice\(result\.tier_actions\.indexOf\(action\)\)\) \{\s*noteCspGate\(rest, "rejected_killswitch", "Node kill switch on — dispatcher ended"\);\s*\}\s*break;\s*\}/);
});

test("a rotation's sale stands; its buy is skipped and logged when entries halted after the sale", () => {
  const ru = fnBody("runUpgradeCandidates");
  const wait = ru.indexOf("await new Promise(r => setTimeout(r, 1000));");
  const halted = ru.indexOf('const buyHalted = state.killSwitch || !haltAllows(currentHaltState(), "entry");');
  const post = ru.indexOf('const upgradeOrderResult = await alpaca("/v2/orders", {');
  assert.ok(wait > 0 && wait < halted && halted < post, "checked after the sale and the wait, before the buy");
  assert.match(ru, /if \(upgradeQty > 0 && !buyHalted\) \{/);
  assert.match(ru, /audit\("ROTATE-BUY-HALTED"/);
  assert.match(ru, /buy_qty: buyHalted \? 0 : upgradeQty/, "the rotation log does not claim a buy that was never sent");
  assert.ok(ru.indexOf('alpaca(`/v2/positions/${candidate.ticker}`, { method: "DELETE" })') < halted, "the exit is not gated");
});

test("the scale-in re-check is outside the order's try and uses its loop-top expression", () => {
  const si = fnBody("executeScaleInTranches");
  const checks = si.match(/if \(state\.killSwitch \|\| !haltAllows\(currentHaltState\(\), "entry"\)\) break;/g) || [];
  assert.equal(checks.length, 2, "loop top + right before the order");
  const post = si.indexOf('const trancheOrder = await alpaca("/v2/orders", {');
  const last = si.lastIndexOf('if (state.killSwitch || !haltAllows(currentHaltState(), "entry")) break;', post);
  assert.match(si.slice(last, post), /break;\n\s*try \{\n\s*const trancheParams = getOrderParams\(price\);\n/);
});
