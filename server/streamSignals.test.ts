// Tier-0 stream signals restored as a MEASURED signal (2026-11-06): the rule,
// the regular-session within-day baseline, the log, and the bot.ts wiring.
// Run: npx tsx --test server/streamSignals.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  STREAM_SIGNAL, appendStreamSignal, detectStreamSignal, etSession, isRegularSessionMinute, pushRegularBar,
  streamSignalLogEnabled, streamSignalLogPath, type StreamHistory,
} from "./streamSignals";

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");

test("bar times map to the ET day and minute across the DST change", () => {
  assert.deepEqual(etSession("2026-10-28T13:31:00Z"), { day: "2026-10-28", minute: 9 * 60 + 31 }, "EDT");
  assert.deepEqual(etSession("2026-11-05T14:31:00Z"), { day: "2026-11-05", minute: 9 * 60 + 31 }, "EST");
  assert.deepEqual(etSession("2026-11-06T03:30:00Z"), { day: "2026-11-05", minute: 22 * 60 + 30 }, "late evening is the previous ET day");
  assert.equal(etSession("not a time"), null);
  assert.equal(etSession(""), null);
});

test("only 09:30-16:00 ET bars count", () => {
  assert.equal(isRegularSessionMinute(9 * 60 + 29), false);
  assert.equal(isRegularSessionMinute(9 * 60 + 30), true);
  assert.equal(isRegularSessionMinute(15 * 60 + 59), true);
  assert.equal(isRegularSessionMinute(16 * 60), false);
});

test("the history is within the day and capped", () => {
  let h: StreamHistory | undefined;
  for (let i = 0; i < 25; i++) h = pushRegularBar(h, "2026-11-05", 100 + i, 50 + i);
  assert.equal(h!.vols.length, STREAM_SIGNAL.HISTORY);
  assert.equal(h!.vols[0], 105, "oldest bars drop first");
  h = pushRegularBar(h, "2026-11-06", 7, 8);
  assert.deepEqual(h, { day: "2026-11-06", vols: [7], closes: [8] }, "a new ET day starts empty");
});

test("the rule itself is unchanged: 2.5x volume, 0.8% bar, 0.3% five-bar momentum, five bars", () => {
  const day = "2026-11-05";
  let h: StreamHistory | undefined;
  for (const c of [100, 100.1, 100.2, 100.3]) h = pushRegularBar(h, day, 100_000, c);
  assert.equal(detectStreamSignal(h!, 100.3, 100.3), null, "four bars: not enough history");
  h = pushRegularBar(h, day, 300_000, 101.4);          // 3x volume, +1.1% bar, +1.4% vs 4 bars ago
  const s = detectStreamSignal(h, 100.3, 101.4)!;
  assert.ok(s, "fires");
  assert.equal(s.direction, "bullish");
  assert.equal(s.volRatio, 3);
  assert.ok(Math.abs(s.barPct - 1.0967) < 1e-3 && Math.abs(s.mom5Pct - 1.4) < 1e-9);
  const quiet = pushRegularBar({ day, vols: [100_000, 100_000, 100_000, 100_000], closes: [100, 100.1, 100.2, 100.3] }, day, 240_000, 101.4);
  assert.equal(detectStreamSignal(quiet, 100.3, 101.4), null, "2.4x is below 2.5x");
  const small = pushRegularBar({ day, vols: [100_000, 100_000, 100_000, 100_000], closes: [100, 100.1, 100.2, 100.3] }, day, 300_000, 101.0);
  assert.equal(detectStreamSignal(small, 100.3, 101.0), null, "a 0.70% bar is below 0.8%");
});

test("the live 09:31 artifact: the opening auction against thin pre-market bars no longer fires", () => {
  // what bot.ts did until 2026-11-06: every bar, pre-market included, in one 20-bar list
  const old = { vols: [] as number[], closes: [] as number[] };
  const bars: Array<[string, number, number, number]> = [];   // [t, open, close, volume]
  for (let m = 0; m < 19; m++) bars.push([new Date(Date.UTC(2026, 10, 4, 14, 12 + m)).toISOString(), 250, 250.1, 1_000]);
  bars.push(["2026-11-04T14:30:00Z", 250.1, 252.6, 849_000]);   // 09:30 ET opening bar, +1.0%
  let oldFired = false, newFired = false;
  let h: StreamHistory | undefined;
  for (const [t, o, c, v] of bars) {
    old.vols.push(v); old.closes.push(c);
    if (old.vols.length >= 5) {
      const s = detectStreamSignal({ day: "x", vols: old.vols.slice(-20), closes: old.closes.slice(-20) }, o, c);
      if (s) oldFired = true;
    }
    const sess = etSession(t)!;
    if (!isRegularSessionMinute(sess.minute)) continue;
    h = pushRegularBar(h, sess.day, v, c);
    if (detectStreamSignal(h, o, c)) newFired = true;
  }
  assert.equal(oldFired, true, "premise: the old baseline flagged the open (849x)");
  assert.equal(newFired, false, "the within-day regular-session baseline cannot fire before the fifth bar");
});

test("the log: one JSON line per flag, rotated above its size cap, failures reported", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "stream-sig-"));
  const file = path.join(dir, "s.jsonl");
  assert.equal(appendStreamSignal({ ticker: "AMD", price: 250.1 }, file, 1_000), true);
  assert.equal(appendStreamSignal({ ticker: "NVDA", price: 190 }, file, 1_000), true);
  assert.deepEqual(fs.readFileSync(file, "utf8").trim().split("\n").map((l) => JSON.parse(l).ticker), ["AMD", "NVDA"]);
  fs.writeFileSync(file, "x".repeat(2_000));
  assert.equal(appendStreamSignal({ ticker: "AAPL" }, file, 1_000), true);
  assert.equal(fs.readFileSync(`${file}.1`, "utf8").length, 2_000, "the full file is kept as .1");
  assert.deepEqual(fs.readFileSync(file, "utf8").trim().split("\n").map((l) => JSON.parse(l).ticker), ["AAPL"]);
  const errors: unknown[] = [];
  const orig = console.error;
  console.error = (...a: unknown[]) => { errors.push(a); };
  try {
    assert.equal(appendStreamSignal({ ticker: "X" }, dir, 1_000), false, "a directory is not a log file (and is never renamed)");
    assert.equal(appendStreamSignal({ ticker: "Y" }, path.join(file, "child.jsonl"), 1_000), false, "a parent that is a file");
  } finally { console.error = orig; }
  assert.ok(fs.statSync(dir).isDirectory() && !fs.existsSync(`${dir}.1`), "the directory was left alone");
  assert.equal(errors.length, 2, "each failure is logged, not swallowed");
  assert.equal(streamSignalLogPath("/data/voltrade"), "/data/voltrade/voltrade_stream_signals.jsonl");
});

test("one off switch: VOLTRADE_STREAM_SIGNAL_LOG", () => {
  assert.equal(streamSignalLogEnabled({}), true, "on by default");
  for (const v of ["off", "0", "false", "none", "NO", " Off "]) assert.equal(streamSignalLogEnabled({ VOLTRADE_STREAM_SIGNAL_LOG: v }), false, v);
  assert.equal(streamSignalLogEnabled({ VOLTRADE_STREAM_SIGNAL_LOG: "on" }), true);
});

test("bot.ts: flags are measured and never queued for a trade, and the untrue 'processing first' row is gone", () => {
  assert.ok(!bot.includes("streamSignalQueue"), "the queue the scan cleared unread is gone");
  assert.ok(!bot.includes('audit("STREAM-PRIORITY"') && !bot.includes("queued for fast scan"));
  const h = bot.slice(bot.indexOf("// ── Tier-0 stream signal (2026-11-06)"), bot.indexOf('ws.on("error"'));
  for (const piece of ["etSession(String(item.t", "isRegularSessionMinute(sess.minute)", "pushRegularBar(streamHistory[ticker]",
    "detectStreamSignal(hist, open_, close)", "STREAM_SIGNAL.COOLDOWN_MS", "streamSignalLogEnabled() && appendStreamSignal(",
    "not traded"]) {
    assert.ok(h.includes(piece), piece);
  }
  assert.ok(h.indexOf("checkPositionOnTick") < 0, "the signal block comes after the position monitor, which still sees every bar");
  const monitor = bot.slice(bot.indexOf("// 1-minute bar received"), bot.indexOf("// ── Tier-0 stream signal (2026-11-06)"));
  assert.ok(monitor.includes("checkPositionOnTick(ticker, close)"), "exits still run on every bar, any session");
});
