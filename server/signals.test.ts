/**
 * signals.test.ts — the signal attribution reader (2026-11-06): one cached
 * body behind GET /api/bot/signals and the diag probe "signals".
 */
import test from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "node:url";
import { makeSignalsReader, SIGNALS_CMD, SIGNALS_CACHE_MS, SIGNALS_EXEC_OPTS, type PythonCallResult } from "./signals";
import { DIAG_PROBES } from "./diag";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const ok = (body: Record<string, unknown>, via = "daemon"): PythonCallResult => ({ success: true, result: body, via });

test("serves the Python body and caches it for SIGNALS_CACHE_MS", async () => {
  let t = 1_000_000;
  let calls = 0;
  const read = makeSignalsReader(async () => { calls++; return ok({ headline: `run ${calls}`, signals: [{ id: "volume" }] }); }, () => t);
  const a = await read();
  assert.equal(a.headline, "run 1");
  assert.equal(a.via, "daemon");
  assert.equal(a.cache_age_s, 0);
  t += SIGNALS_CACHE_MS - 1;
  const b = await read();
  assert.equal(b.headline, "run 1");
  assert.equal(calls, 1);
  assert.equal(b.cache_age_s, Math.round((SIGNALS_CACHE_MS - 1) / 1000));
  t += 2;
  assert.equal((await read()).headline, "run 2");
  assert.equal(calls, 2);
});

test("concurrent readers share one Python call", async () => {
  let calls = 0;
  let release!: () => void;
  const gate = new Promise<void>((r) => { release = r; });
  const read = makeSignalsReader(async () => { calls++; await gate; return ok({ headline: "x" }); });
  const p = Promise.all([read(), read(), read()]);
  release();
  const out = await p;
  assert.equal(calls, 1);
  assert.ok(out.every((o) => o.headline === "x"));
});

test("a failed refresh keeps the last good body, marks it stale and keeps its true age", async () => {
  let t = 0;
  let fail = false;
  const read = makeSignalsReader(async () => (fail ? { success: false, result: { error: "boom" }, via: "none" } : ok({ headline: "good" })), () => t);
  await read();
  fail = true;
  t += SIGNALS_CACHE_MS + 5_000;
  const s = await read();
  assert.equal(s.headline, "good");
  assert.equal(s.stale, true);
  assert.equal(s.refresh_error, "boom");
  assert.equal(s.cache_age_s, Math.round((SIGNALS_CACHE_MS + 5_000) / 1000), "age counts from when the body was computed");
});

test("no body ever: an error payload, never a fabricated table; a thrown call is caught", async () => {
  const read = makeSignalsReader(async () => { throw new Error("socket closed"); });
  const r = await read();
  assert.equal(r.error, "socket closed");
  assert.deepEqual(r.signals, []);
  const read2 = makeSignalsReader(async () => ok({ error: "shadow log unreadable" }));
  assert.equal((await read2()).error, "shadow log unreadable");
});

test("command, options and wiring: owner route + diag probe share the reader; daemon method registered", () => {
  assert.match(SIGNALS_CMD, /from signal_attribution import get_signal_attribution/);
  assert.ok(SIGNALS_EXEC_OPTS.timeout <= 85_000, "inside the 90 s Python mutex lane");
  const bot = fs.readFileSync(path.join(ROOT, "server/bot.ts"), "utf8");
  assert.ok(bot.includes('app.get("/api/bot/signals", requireOwner,'), "owner-only route");
  assert.ok(bot.includes('pythonCall("signal_attribution", {}, SIGNALS_CMD,'), "daemon first, subprocess fallback");
  assert.ok(bot.includes('sanitizeDiag({ probe: "signals", ...(await readSignals()) })'), "diag probe serves the same body, sanitized");
  assert.ok((DIAG_PROBES as readonly string[]).includes("signals"));
  const daemon = fs.readFileSync(path.join(ROOT, "voltrade_daemon.py"), "utf8");
  assert.ok(daemon.includes('"signal_attribution": ("signal_attribution", "get_signal_attribution")'));
  assert.ok(fs.existsSync(path.join(ROOT, "signal_attribution.py")));
});
