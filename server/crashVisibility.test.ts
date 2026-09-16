// crashVisibility.ts — a dying process leaves a durable reason.
//
// KNOWN BROKEN #41: the 2026-09-08 restart loop left nothing in the audit
// log because the server had no unhandledRejection / uncaughtException
// handlers. These pin: rejections are recorded and survived, exceptions are
// recorded and then exit(1), neither handler can throw, and bot.ts installs
// them at boot and reports the counts on /api/health.
import { test } from "node:test";
import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  installCrashHandlers, describeReason, crashStats, _resetCrashStats,
  FATAL_REJECTION, FATAL_EXCEPTION, EXIT_GRACE_MS,
} from "./crashVisibility";

const HERE = path.dirname(fileURLToPath(import.meta.url));

function harness() {
  const proc = new EventEmitter();
  const audits: Array<[string, string]> = [];
  const exits: number[] = [];
  const scheduled: Array<{ fn: () => void; ms: number }> = [];
  _resetCrashStats();
  const ok = installCrashHandlers(proc as any, {
    audit: (a, d) => { audits.push([a, d]); },
    exit: (c) => { exits.push(c); },
    schedule: (fn, ms) => { scheduled.push({ fn, ms }); },
    now: () => new Date("2026-09-16T03:00:00Z"),
  });
  return { proc, audits, exits, scheduled, ok };
}

test("an unhandled rejection is audited with its stack and the process is NOT exited", () => {
  const h = harness();
  assert.equal(h.ok, true);
  const err = new Error("ENOENT: archive day file missing");
  h.proc.emit("unhandledRejection", err, Promise.resolve());
  assert.equal(h.audits.length, 1);
  assert.equal(h.audits[0][0], FATAL_REJECTION);
  assert.match(h.audits[0][1], /ENOENT: archive day file missing/);
  assert.match(h.audits[0][1], /crashVisibility\.test\.ts|@/, "stack frames are included");
  assert.deepEqual(h.exits, []);
  assert.deepEqual(h.scheduled, []);
  const s = crashStats();
  assert.equal(s.unhandledRejections, 1);
  assert.equal(s.lastRejection?.at, "2026-09-16T03:00:00.000Z");
  assert.match(s.lastRejection!.detail, /ENOENT/);
});

test("an uncaught exception is audited FIRST, then exit(1) is scheduled after the grace", () => {
  const h = harness();
  h.proc.emit("uncaughtException", new TypeError("Cannot read properties of undefined (reading 'rows')"), "uncaughtException");
  assert.equal(h.audits.length, 1);
  assert.equal(h.audits[0][0], FATAL_EXCEPTION);
  assert.match(h.audits[0][1], /TypeError: Cannot read properties of undefined/);
  assert.match(h.audits[0][1], /exiting 1/);
  assert.equal(h.scheduled.length, 1);
  assert.equal(h.scheduled[0].ms, EXIT_GRACE_MS);
  assert.deepEqual(h.exits, [], "exit is deferred so the audit write and stdout flush land first");
  h.scheduled[0].fn();
  assert.deepEqual(h.exits, [1]);
  assert.equal(crashStats().uncaughtExceptions, 1);
});

test("non-Error rejection reasons are still described, never dropped", () => {
  assert.equal(describeReason("plain string"), "plain string");
  assert.equal(describeReason(undefined), "undefined (rejected with no reason)");
  assert.equal(describeReason({ code: 42, why: "x" }), '{"code":42,"why":"x"}');
  const circular: Record<string, unknown> = {}; circular.self = circular;
  assert.equal(describeReason(circular), "[object Object]");
  const long = describeReason("x".repeat(5000));
  assert.ok(long.length < 1600 && long.endsWith("…"), "bounded");
});

test("the handlers survive an audit sink that throws (a crash handler must never crash)", () => {
  const proc = new EventEmitter();
  const exits: number[] = [];
  _resetCrashStats();
  installCrashHandlers(proc as any, {
    audit: () => { throw new Error("sqlite is locked"); },
    exit: (c) => { exits.push(c); },
    schedule: (fn) => fn(),
  });
  assert.doesNotThrow(() => proc.emit("unhandledRejection", new Error("x"), Promise.resolve()));
  assert.doesNotThrow(() => proc.emit("uncaughtException", new Error("y"), "uncaughtException"));
  assert.deepEqual(exits, [1], "the exception path still exits even when auditing failed");
  assert.equal(crashStats().unhandledRejections, 1);
  assert.equal(crashStats().handlerFaults, 2, "each throwing audit call is counted as a handler fault, not hidden");
});

test("installing twice on the same process object is a no-op (no double logging)", () => {
  const proc = new EventEmitter();
  const audits: string[] = [];
  _resetCrashStats();
  assert.equal(installCrashHandlers(proc as any, { audit: (a) => { audits.push(a); }, exit: () => {}, schedule: (fn) => fn() }), true);
  assert.equal(installCrashHandlers(proc as any, { audit: (a) => { audits.push(a); }, exit: () => {}, schedule: (fn) => fn() }), false);
  proc.emit("unhandledRejection", new Error("once"), Promise.resolve());
  assert.deepEqual(audits, [FATAL_REJECTION]);
});

test("counts accumulate and crashStats() returns copies", () => {
  const h = harness();
  h.proc.emit("unhandledRejection", new Error("a"), Promise.resolve());
  h.proc.emit("unhandledRejection", "b", Promise.resolve());
  const s1 = crashStats();
  assert.equal(s1.unhandledRejections, 2);
  assert.equal(s1.lastRejection?.detail, "b");
  s1.lastRejection!.detail = "mutated";
  assert.equal(crashStats().lastRejection?.detail, "b");
});

// ── source ratchets on bot.ts ───────────────────────────────────────────────
test("bot.ts installs the handlers at boot, right after the STARTUP audit line", () => {
  const bot = fs.readFileSync(path.join(HERE, "bot.ts"), "utf8");
  const startup = bot.indexOf('audit("STARTUP", `Server boot');
  assert.ok(startup > 0, "STARTUP audit line not found");
  const after = bot.slice(startup, startup + 2500);
  assert.match(after, /installCrashHandlers\(process, \{ audit \}\)/, "handlers must be installed with the real audit sink, once, at boot");
});

test("bot.ts's /api/health reports crash counts so a restart loop names its reason", () => {
  const bot = fs.readFileSync(path.join(HERE, "bot.ts"), "utf8");
  const start = bot.indexOf('app.get("/api/health"');
  const end = bot.indexOf("app.get(", start + 10);
  const handler = bot.slice(start, end);
  assert.match(handler, /checks\.checks\.process = crashStats\(\)/);
});
