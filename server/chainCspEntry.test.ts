// chainCspEntry — the CSP the bot would have sold, per underlying per archived
// day (research test 2 gate 3, [RULE-REVIEW] 2026-10-26).
// Run: npx tsx --test server/chainCspEntry.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "path";
import zlib from "zlib";
import { fileURLToPath } from "node:url";
import {
  cspEntryFor, cspEntryForDay, cspEntryForDays, derivedDir, isLiquid, impliedSpot,
  CSP_SIM_TARGET_DELTA, CSP_SIM_DTE_MIN, CSP_SIM_DTE_MAX, CSP_SIM_MIN_OI, CSP_SIM_MAX_SPREAD_PCT,
  CSP_SIM_MIN_MID, CSP_SIM_MIN_QUOTE_SIZE,
} from "./chainCspEntry";
import type { ChainRecord } from "./optionsChainArchive";

const here = path.dirname(fileURLToPath(import.meta.url));
const py = (f: string) => fs.readFileSync(path.join(here, "..", f), "utf8");
const num = (src: string, re: RegExp): number => {
  const m = re.exec(src);
  assert.ok(m, `pattern not found: ${re}`);
  return Number(m[1]);
};

const DAY = "2026-10-01";
const DAY_MS = Date.parse(`${DAY}T00:00:00Z`);
const occ = (u: string, dte: number, type: "C" | "P", strike: number) => {
  const d = new Date(DAY_MS + dte * 86_400_000);
  const yy = String(d.getUTCFullYear()).slice(2), mm = String(d.getUTCMonth() + 1).padStart(2, "0"), dd = String(d.getUTCDate()).padStart(2, "0");
  return `${u}${yy}${mm}${dd}${type}${String(Math.round(strike * 1000)).padStart(8, "0")}`;
};
const rec = (u: string, dte: number, type: "C" | "P", strike: number, delta: number | null, extra: Partial<ChainRecord> = {}): ChainRecord =>
  ({ s: occ(u, dte, type, strike), u, t: DAY_MS + 20 * 3_600_000, bid: 1.0, ask: 1.08, bsz: 20, asz: 20, iv: 0.45, delta, oi: 500, feed: "indicative", ...extra });

test("the selection constants are the live Python values, read from the source (never a drifting copy)", () => {
  const ex = py("options_execution.py"), mg = py("options_manager.py"), ts = py("tiered_strategy.py");
  // Each pattern is read from the function that owns the rule — other
  // selectors carry their own target_delta / floors.
  const fn = (src: string, name: string) => {
    const at = src.indexOf(`def ${name}(`);
    assert.ok(at >= 0, `${name} not found`);
    const next = src.indexOf("\ndef ", at + 1);
    return src.slice(at, next < 0 ? undefined : next);
  };
  const sellPut = fn(ex, "_select_sell_put"), liquid = fn(ex, "_is_liquid"), fetchChain = fn(ex, "_fetch_option_chain");
  assert.equal(CSP_SIM_MIN_OI, num(ex, /^MIN_OPEN_INTEREST\s*=\s*(\d+)/m));
  assert.equal(CSP_SIM_MAX_SPREAD_PCT, num(ex, /^MAX_SPREAD_PCT\s*=\s*([\d.]+)/m));
  assert.equal(CSP_SIM_TARGET_DELTA, num(sellPut, /target_delta\s*=\s*(-[\d.]+)/));
  assert.equal(CSP_SIM_DTE_MAX, num(fetchChain, /max_exp\s*=\s*\(now \+ timedelta\(days=(\d+)\)\)/));
  assert.equal(CSP_SIM_DTE_MIN, Math.max(num(ts, /^T1_TARGET_DTE_MIN\s*=\s*(\d+)/m), num(mg, /^DTE_EXIT_THRESHOLD\s*=\s*(\d+)/m) + 1),
    "CSP_ENTRY_MIN_DTE = max(T1_TARGET_DTE_MIN, DTE_EXIT_THRESHOLD + 1)");
  assert.equal(CSP_SIM_MIN_MID, num(liquid, /if mid < ([\d.]+):/));
  assert.equal(CSP_SIM_MIN_QUOTE_SIZE, num(liquid, /bid_size >= (\d+) and ask_size >= \d+/));
});

test("isLiquid mirrors _is_liquid minus the volume branch: quote present, spread ≤ 10% of mid, mid ≥ 0.10, OI ≥ 200 or both sizes ≥ 10", () => {
  const ok = { bid: 1.0, ask: 1.08, oi: 500, bsz: 1, asz: 1 };
  assert.equal(isLiquid(ok), true);
  assert.equal(isLiquid({ ...ok, bid: 0 }), false, "no bid");
  assert.equal(isLiquid({ ...ok, ask: null }), false, "no ask");
  assert.equal(isLiquid({ ...ok, ask: 1.12 }), false, "spread 11.3% of mid");
  assert.equal(isLiquid({ bid: 0.07, ask: 0.08, oi: 500, bsz: 1, asz: 1 }), false, "penny option");
  assert.equal(isLiquid({ ...ok, oi: 150, bsz: 12, asz: 12 }), true, "thin OI but both sizes ≥ 10");
  assert.equal(isLiquid({ ...ok, oi: 150, bsz: 5, asz: 12 }), false, "thin OI and one thin size");
  assert.equal(isLiquid({ ...ok, oi: null, bsz: null, asz: null }), false);
});

test("cspEntryFor: the liquid OTM put nearest −0.30 delta inside the DTE window; calls, short/long expiries, illiquid and ITM rows never win", () => {
  const u = "AAA";
  const recs: ChainRecord[] = [
    rec(u, 35, "P", 90, -0.15),
    rec(u, 35, "P", 95, -0.28, { bid: 1.5, ask: 1.6, iv: 0.5, oi: 800 }),
    rec(u, 35, "P", 97, -0.33),
    rec(u, 35, "P", 102, -0.52),                             // ITM — excluded from the pick, used by the spot proxy
    rec(u, 35, "P", 96, -0.30, { bid: 1.0, ask: 1.3 }),      // exactly −0.30 but a 26% spread — illiquid
    rec(u, 35, "C", 100, 0.5),                               // a call
    rec(u, 10, "P", 95, -0.30),                              // inside the exit zone
    rec(u, 55, "P", 95, -0.30),                              // beyond the fetch ceiling
  ];
  const row = cspEntryFor(DAY, u, recs, DAY_MS);
  assert.equal(row.reason, "ok");
  assert.equal(row.occ, occ(u, 35, "P", 95));
  assert.equal(row.strike, 95);
  assert.equal(row.dte, 35);
  assert.equal(row.exp, new Date(DAY_MS + 35 * 86_400_000).toISOString().slice(0, 10));
  assert.deepEqual([row.bid, row.ask, row.mid], [1.5, 1.6, 1.55]);
  assert.equal(row.spreadPct, 0.0645);
  assert.deepEqual([row.iv, row.delta, row.oi], [0.5, -0.28, 800]);
  assert.equal(row.nPuts, 5, "puts inside the window: 90, 95, 97, 102 and the illiquid 96");
  assert.equal(row.nLiquid, 4);
  // |delta| crosses 0.5 between the 97 (0.33) and 102 (0.52) strikes
  assert.equal(row.spotImplied, 101.47);
  assert.equal(row.feed, "indicative");
});

test("cspEntryFor: ties on delta distance go to the higher open interest", () => {
  const u = "BBB";
  const recs = [rec(u, 40, "P", 50, -0.28, { oi: 300 }), rec(u, 40, "P", 51, -0.28, { oi: 900 })];
  assert.equal(cspEntryFor(DAY, u, recs, DAY_MS).strike, 51);
});

test("cspEntryFor: every failure names its reason and keeps the counts", () => {
  const u = "CCC";
  assert.equal(cspEntryFor(DAY, u, [rec(u, 35, "C", 100, 0.5)], DAY_MS).reason, "no_puts");
  const noLiquid = cspEntryFor(DAY, u, [rec(u, 35, "P", 95, -0.3, { bid: 1.0, ask: 1.5 })], DAY_MS);
  assert.deepEqual([noLiquid.reason, noLiquid.nPuts, noLiquid.nLiquid], ["no_liquid", 1, 0]);
  const noDelta = cspEntryFor(DAY, u, [rec(u, 35, "P", 95, null), rec(u, 35, "P", 94, 0)], DAY_MS);
  assert.deepEqual([noDelta.reason, noDelta.nPuts, noDelta.nLiquid], ["no_delta", 2, 2]);
  assert.equal(cspEntryFor(DAY, u, [rec(u, 35, "P", 120, -0.8)], DAY_MS).reason, "no_otm");
  assert.equal(cspEntryFor(DAY, u, [rec(u, 35, "P", 95, 0.3)], DAY_MS).reason, "no_delta", "a put with positive delta is a bad row");
  assert.equal(cspEntryFor(DAY, u, [], DAY_MS).occ, null);
});

test("impliedSpot: exact 0.5 wins; interpolated between the bracketing strikes; null without a bracket", () => {
  const p = (strike: number, delta: number) => ({ r: rec("D", 35, "P", strike, delta), exp: 0, dte: 35, strike });
  assert.equal(impliedSpot([p(90, -0.2), p(100, -0.5), p(110, -0.7)]), 100);
  assert.equal(impliedSpot([p(90, -0.3), p(100, -0.6)]), 96.67);
  assert.equal(impliedSpot([p(90, -0.2), p(95, -0.3)]), null, "all OTM — no bracket");
  assert.equal(impliedSpot([p(90, -0.6), p(95, -0.7)]), null, "all ITM — no bracket");
  assert.equal(impliedSpot([]), null);
});

test("cspEntryForDay: one row per underlying, sorted", () => {
  const rows = cspEntryForDay([rec("ZZZ", 35, "P", 95, -0.3), rec("AAA", 35, "P", 95, -0.3), rec("MMM", 35, "C", 95, 0.3)], DAY);
  assert.deepEqual(rows.map((r) => [r.u, r.reason]), [["AAA", "ok"], ["MMM", "no_puts"], ["ZZZ", "ok"]]);
});

test("archive reading + derived cache: gz and plain day files, write-once for finished days, never for today, pending beyond maxCompute", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "chaincsp-"));
  const dir = path.join(base, "optionchains");
  fs.mkdirSync(dir, { recursive: true });
  const lines = (rs: ChainRecord[]) => rs.map((r) => JSON.stringify(r)).join("\n") + "\n";
  fs.writeFileSync(path.join(dir, "2026-10-01.jsonl.gz"), zlib.gzipSync(lines([rec("AAA", 35, "P", 95, -0.3)])));
  fs.writeFileSync(path.join(dir, "2026-10-02.jsonl"), lines([rec("AAA", 35, "P", 95, -0.3)]) + "not json\n");
  fs.writeFileSync(path.join(dir, "2026-10-03.jsonl"), lines([rec("AAA", 35, "P", 95, -0.3)]));
  const now = Date.parse("2026-10-03T18:00:00Z"); // 10-03 is "today": never cached
  const a = await cspEntryForDays(["2026-10-01", "2026-10-02", "2026-10-03"], base, now, 2);
  assert.deepEqual([a.computed, a.cached, a.pending], [["2026-10-01", "2026-10-02"], [], ["2026-10-03"]]);
  assert.equal(a.rows.length, 2);
  assert.ok(fs.existsSync(path.join(derivedDir(base), "2026-10-01.json")));
  const b = await cspEntryForDays(["2026-10-01", "2026-10-02", "2026-10-03"], base, now, 2);
  assert.deepEqual([b.computed, b.cached, b.pending], [["2026-10-03"], ["2026-10-01", "2026-10-02"], []]);
  assert.equal(b.rows.length, 3);
  assert.ok(!fs.existsSync(path.join(derivedDir(base), "2026-10-03.json")), "today is never cached");
  // write-once: a changed source day does not change the derived row
  fs.writeFileSync(path.join(dir, "2026-10-02.jsonl"), lines([rec("AAA", 35, "P", 90, -0.3)]));
  const c = await cspEntryForDays(["2026-10-02"], base, now);
  assert.equal(c.rows[0].strike, 95);
  assert.equal(derivedDir(base).endsWith(path.join("optionchains", "derived", "csp_entry")), true, "its own cache dir, beside iv30");
});

test("the probe is whitelisted and wired with the iv30 query shape", () => {
  const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
  const diag = fs.readFileSync(path.join(here, "diag.ts"), "utf8");
  assert.ok(diag.includes('"chain_csp_entry"'));
  const at = bot.indexOf('case "chain_csp_entry": {');
  assert.ok(at > 0);
  const body = bot.slice(at, bot.indexOf("default:", at));
  for (const s of ["req.query.from", "req.query.to", "req.query.u", "req.query.max", "cspEntryForDays(", 'feed: "indicative"', "rows.map((x) => sanitizeDiag(x))"]) {
    assert.ok(body.includes(s), `probe body must include ${s}`);
  }
  assert.equal(CSP_SIM_DTE_MIN < CSP_SIM_DTE_MAX, true);
});
