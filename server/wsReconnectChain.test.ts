// RECONNECT CHAIN (2026-11-05). Live: after two owner toggles, the position-
// monitor stream logged 200 STREAM rows in 11 minutes and Alpaca answered
// 406 "connection limit exceeded": every socket's close handler scheduled a
// reconnect, including a socket startStreaming() had just replaced, so each
// extra start began a chain of restarts that never ended. This harness slices
// startStreaming()/stopStreaming() out of bot.ts, transpiles them with esbuild
// and RUNS them against a fake WebSocket and a fake clock.
// Run: npx tsx --test server/wsReconnectChain.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { transformSync } from "esbuild";

const here = path.dirname(fileURLToPath(import.meta.url));
// WS_CHAIN_BOT_TS points the harness at another copy of bot.ts (e.g. the pre-fix one, to watch it fail)
const BOT_PATH = process.env.WS_CHAIN_BOT_TS || path.join(here, "bot.ts");
const bot = fs.readFileSync(BOT_PATH, "utf8");

function slice(from: string, to: string): string {
  const i = bot.indexOf(from);
  assert.ok(i >= 0, `missing: ${from}`);
  const j = bot.indexOf(to, i);
  assert.ok(j > i, `missing end: ${to}`);
  return bot.slice(i, j);
}

class FakeWs {
  static all: FakeWs[] = [];
  static pendingCloses: FakeWs[] = [];
  handlers: Record<string, (...a: unknown[]) => void> = {};
  sent: string[] = [];
  closed = false;
  constructor(public url: string) { FakeWs.all.push(this); }
  on(ev: string, fn: (...a: unknown[]) => void) { this.handlers[ev] = fn; }
  send(m: string) { this.sent.push(m); }
  close() { if (!this.closed) { this.closed = true; FakeWs.pendingCloses.push(this); } }
  emit(ev: string, ...a: unknown[]) { this.handlers[ev]?.(...a); }
  subscribe() { this.emit("open"); this.emit("message", Buffer.from(JSON.stringify([{ T: "subscription", bars: ["SPY", "SSO"] }]))); }
}

function harness() {
  FakeWs.all = []; FakeWs.pendingCloses = [];
  let now = 0, seq = 0;
  const timers = new Map<number, { at: number; fn: () => void }>();
  const audits: Array<[string, string]> = [];
  const body = slice("  function startStreaming() {", "\n  // PYTHON DAEMON LIVENESS");
  const js = transformSync(
    `let streamWs = null; let streamConnected = false; let streamReconnectTimer = null;\n${body}\n` +
    `return { startStreaming, stopStreaming, current: () => streamWs, connected: () => streamConnected };`,
    { loader: "ts", format: "cjs" },
  ).code;
  const api = new Function(
    "WebSocket", "audit", "haltAllows", "currentHaltState", "setTimeout", "clearTimeout",
    "ALPACA_KEY", "ALPACA_SECRET", "STREAM_TICKERS", "syncMonitoredPositions", "monitoredPositions",
    "checkPositionOnTick", "streamVolHistory", "streamPriceHistory", "streamLastSignal", "state", js,
  )(
    FakeWs, (t: string, m: string) => audits.push([t, m]), () => true, () => ({}),
    (fn: () => void, ms: number) => { const id = ++seq; timers.set(id, { at: now + ms, fn }); return id; },
    (id: number) => { timers.delete(id); },
    "k", "s", ["SPY"], async () => {}, {}, async () => {}, {}, {}, {}, {},
  ) as { startStreaming: () => void; stopStreaming: () => void; current: () => FakeWs | null; connected: () => boolean };
  const flushCloses = () => { while (FakeWs.pendingCloses.length) FakeWs.pendingCloses.shift()!.emit("close"); };
  const advance = (ms: number) => {
    const end = now + ms;
    for (;;) {
      flushCloses();
      const due = [...timers.entries()].filter(([, t]) => t.at <= end).sort((a, b) => a[1].at - b[1].at)[0];
      if (!due) break;
      now = due[1].at; timers.delete(due[0]); due[1].fn();
      // a fresh socket subscribes like Alpaca would once the slot is free
      const cur = api.current(); if (cur && !cur.closed && !api.connected()) cur.subscribe();
    }
    now = end; flushCloses();
  };
  return { api, audits, advance, timers, flushCloses };
}

test("a second start (owner toggle) replaces the socket once: no reconnect chain", () => {
  const h = harness();
  h.api.startStreaming(); FakeWs.all[0].subscribe();
  h.api.startStreaming();                     // the toggle: socket A replaced by B
  FakeWs.all[1].subscribe();
  h.advance(120_000);                          // two minutes of timers
  assert.equal(FakeWs.all.length, 2, `sockets opened: ${FakeWs.all.length} (a chain keeps restarting the stream)`);
  assert.equal(h.api.current(), FakeWs.all[1]);
  assert.equal(FakeWs.all[1].closed, false, "the healthy socket is never closed by a stale reconnect");
  assert.equal(h.audits.filter(([t]) => t === "STREAM-DISCONNECT").length, 0, "a deliberate replacement is not a disconnect");
  assert.equal(h.timers.size, 0, "nothing is left scheduled");
});

test("a real drop reconnects exactly once, after 10 s", () => {
  const h = harness();
  h.api.startStreaming(); FakeWs.all[0].subscribe();
  FakeWs.all[0].closed = true; FakeWs.all[0].emit("close");     // the network drops it
  assert.equal(h.timers.size, 1);
  h.advance(9_000);
  assert.equal(FakeWs.all.length, 1, "not before 10 s");
  h.advance(60_000);
  assert.equal(FakeWs.all.length, 2, "one reconnect, and the new socket stays up");
  assert.deepEqual(h.audits.filter(([t]) => t === "STREAM-DISCONNECT").map(([, m]) => m), ["Feed disconnected — reconnecting in 10s"]);
});

test("a start while a reconnect is pending cancels it: still one socket", () => {
  const h = harness();
  h.api.startStreaming(); FakeWs.all[0].subscribe();
  FakeWs.all[0].closed = true; FakeWs.all[0].emit("close");
  h.api.startStreaming(); FakeWs.all[1].subscribe();            // e.g. a boot path or toggle inside the 10 s
  h.advance(120_000);
  assert.equal(FakeWs.all.length, 2);
  assert.equal(h.timers.size, 0);
});

test("stopStreaming stops for good: its close schedules nothing", () => {
  const h = harness();
  h.api.startStreaming(); FakeWs.all[0].subscribe();
  h.api.stopStreaming();
  h.advance(120_000);
  assert.equal(FakeWs.all.length, 1);
  assert.equal(h.api.current(), null);
  assert.equal(h.timers.size, 0);
});

test("a replaced socket's late frames and errors are ignored; the error listener stays attached", () => {
  const h = harness();
  h.api.startStreaming(); const a = FakeWs.all[0]; a.subscribe();
  h.api.startStreaming();
  a.emit("error", new Error("WebSocket was closed before the connection was established"));
  assert.equal(h.audits.filter(([t]) => t === "STREAM-ERROR").length, 0, "a replaced socket's error is noise");
  assert.ok(typeof a.handlers.error === "function", "an 'error' event with no listener would exit the process");
  FakeWs.all[1].emit("error", new Error("boom"));
  assert.equal(h.audits.filter(([t]) => t === "STREAM-ERROR").length, 1, "the current socket's errors are audited");
});
