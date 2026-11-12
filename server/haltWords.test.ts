// HALT WORDS (2026-11-06, rule call "kill-toggle scope") — what the owner
// reads about a kill latch and an owner STOP must match what the code does.
//
// The contract is server/haltScope.ts (audit N15, 2026-10-27) plus the human's
// 2026-11-05 decision for the kill latch ("the floor may only reduce exposure
// … may not make any buys"). Until this change the bot page said the reverse:
// the kill "stops all exits" / "no exits of any kind" / the bot "is not …
// managing exits", and STOP "stops keep firing". These tests derive each
// claim in shared/haltWords.ts from haltAllows()'s truth table, pin that every
// surface reads the shared words, and fail on any stale phrase.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { haltAllows, haltMode, haltStateFrom } from "./haltScope";
import * as W from "../shared/haltWords";

const here = path.dirname(fileURLToPath(import.meta.url));
const read = (rel: string) => fs.readFileSync(path.join(here, rel), "utf8");
const bot = read("bot.ts");

// The two states the words describe, built the way bot.ts builds them.
const killed = haltStateFrom({ ownerStopped: false, active: false, killSwitch: true, circuitBreakerUntil: 0, nowMs: 1 });
const stopped = haltStateFrom({ ownerStopped: true, active: false, killSwitch: false, circuitBreakerUntil: 0, nowMs: 1 });

const KILL_TEXTS: Record<string, string> = {
  KILL_CONSEQUENCE: W.KILL_CONSEQUENCE, KILL_BANNER_TEXT: W.KILL_BANNER_TEXT, KILL_CONFIRM: W.KILL_CONFIRM,
  KILL_LINE_GLOSS: W.KILL_LINE_GLOSS, KILL_ON_AUDIT: W.KILL_ON_AUDIT, KILL_ON_NOTIFY: W.KILL_ON_NOTIFY,
  KILL_ON_EMAIL: W.KILL_ON_EMAIL, DRAWDOWN_KILL_SUFFIX: W.DRAWDOWN_KILL_SUFFIX, KILL_PRETRADE_REASON: W.KILL_PRETRADE_REASON,
};

test("kill latch: the code blocks entries and keeps exits running — every kill text says exactly that", () => {
  assert.equal(haltAllows(killed, "entry"), false);
  assert.equal(haltAllows(killed, "exit"), true);
  assert.equal(haltMode(killed), "exits_only");
  for (const [name, text] of Object.entries(KILL_TEXTS)) {
    assert.match(text, /new (entries|positions)/i, `${name} must say new entries stop: ${text}`);
    assert.match(text, /protective exits keep running|option exits and the floor's reduce-only pass keep running/i, `${name} must say exits keep running: ${text}`);
    assert.doesNotMatch(text, /no exits|stops all exits|not (scanning, trading, or )?managing exits|all trading (halted|stopped)/i, `${name} still claims exits stop: ${text}`);
  }
});

test("kill latch: the floor claim ('may only reduce') matches the code — tier2HaltedExits runs the reduce-only pass while killed", () => {
  for (const t of [W.KILL_CONSEQUENCE, W.KILL_BANNER_TEXT, W.KILL_CONFIRM, W.KILL_ON_EMAIL]) assert.match(t, /floor/i);
  const halted = bot.slice(bot.indexOf("async function tier2HaltedExits()"), bot.indexOf("async function killSwitchFloorReduce()"));
  assert.ok(halted.length > 0);
  assert.ok(halted.includes("if (state.killSwitch) await killSwitchFloorReduce();"));
  assert.ok(bot.includes("from bot_engine import floor_reduce_only_pass"));
});

test("owner STOP: the code freezes every path, exits included — the STOP text says so", () => {
  assert.equal(haltAllows(stopped, "entry"), false);
  assert.equal(haltAllows(stopped, "exit"), false);
  assert.equal(haltMode(stopped), "off");
  assert.match(W.STOP_CONFIRM, /freezes everything/);
  assert.match(W.STOP_CONFIRM, /no exits, stops included/);
  assert.doesNotMatch(W.STOP_CONFIRM, /stops keep firing/);
  assert.match(W.KILL_ON_EMAIL, /\/api\/bot\/stop freezes everything/);
});

test("every bot-page surface reads the shared words", () => {
  const derive = read("../client/src/bot/derive.ts");
  const panels = read("../client/src/bot/panels.tsx");
  const page = read("../client/src/bot/BotPage.tsx");
  const glossary = read("../client/src/bot/glossary.ts");
  assert.match(derive, /import \{ KILL_CONSEQUENCE \} from "\.\.\/\.\.\/\.\.\/shared\/haltWords";/);
  assert.match(derive, /consequence: KILL_CONSEQUENCE, clears:/);
  assert.match(panels, /stop: STOP_CONFIRM,\s*kill: KILL_CONFIRM,/);
  assert.match(page, /<span>\{KILL_BANNER_TEXT\}<\/span>/);
  assert.match(glossary, /The kill line \(−20 % from the peak\) is \$\{KILL_LINE_GLOSS\}\./);
});

test("the server's kill lines read the shared words (owner toggle, both drawdown-kill sites, pre-trade check)", () => {
  const route = bot.slice(bot.indexOf('app.post("/api/bot/kill"'), bot.indexOf('app.post("/api/bot/circuit-breaker/reset"'));
  assert.ok(route.includes('sendEmailAlert("Kill Switch Activated", KILL_ON_EMAIL);'));
  assert.ok(route.includes('audit("KILL SWITCH ON", KILL_ON_AUDIT);'));
  assert.ok(route.includes('notify("alert", KILL_ON_NOTIFY);'));
  // both DRAWDOWN-KILL sites (account route + tier1 check) keep the parsed prefix and add the suffix
  const sites = bot.split("\n").filter((l) => l.includes("const msg = `MAX DRAWDOWN KILL SWITCH: Equity $"));
  assert.equal(sites.length, 2);
  for (const l of sites) assert.ok(l.trimEnd().endsWith("${DRAWDOWN_KILL_SUFFIX}`;"), l);
  assert.equal(bot.split("sendEmailAlert(DRAWDOWN_KILL_EMAIL_SUBJECT, msg);").length - 1, 2);
  assert.ok(bot.includes("return { allowed: false, reason: KILL_PRETRADE_REASON };"));
});

test("no stale kill wording is left on the bot page or in bot.ts", () => {
  const stale = /All trading (halted|stopped)|stops all exits|no exits of any kind|not scanning, trading, or managing exits|stops keep firing|Trading Stopped|stops trading automatically/;
  const dir = path.join(here, "../client/src/bot");
  for (const f of fs.readdirSync(dir).filter((n) => /\.tsx?$/.test(n) && !n.includes(".test."))) {
    assert.doesNotMatch(fs.readFileSync(path.join(dir, f), "utf8"), stale, `client/src/bot/${f}`);
  }
  assert.doesNotMatch(bot, stale, "server/bot.ts");
});
