// chainVs — Cremers–Weinbaum IV spread from the chain archive (Edge P3 shadow
// stream, [PIPELINE] 2026-10-30). Run: npx tsx --test server/chainVs.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "path";
import { fileURLToPath } from "url";
import { vsFor, vsForDay, vsGate1Summary, vsForDays, derivedDir, VS_KILL_SD } from "./chainVs";
import type { ChainRecord } from "./optionsChainArchive";

const here = path.dirname(fileURLToPath(import.meta.url));
const DAY = "2026-10-01";
const DAY_MS = Date.parse(`${DAY}T00:00:00Z`);
const occ = (u: string, dte: number, type: "C" | "P", strike: number) => {
  const d = new Date(DAY_MS + dte * 86_400_000);
  const yy = String(d.getUTCFullYear()).slice(2), mm = String(d.getUTCMonth() + 1).padStart(2, "0"), dd = String(d.getUTCDate()).padStart(2, "0");
  return `${u}${yy}${mm}${dd}${type}${String(Math.round(strike * 1000)).padStart(8, "0")}`;
};
const rec = (u: string, dte: number, type: "C" | "P", strike: number, delta: number | null, iv: number | null): ChainRecord =>
  ({ s: occ(u, dte, type, strike), u, t: DAY_MS + 20 * 3_600_000, bid: 1, ask: 1.1, bsz: 1, asz: 1, iv, delta, oi: null, feed: "indicative" });
/** a call+put pair at one strike: call iv = put iv + spread */
const pair = (u: string, dte: number, strike: number, cd: number, putIv: number, spread: number): ChainRecord[] =>
  [rec(u, dte, "C", strike, cd, putIv + spread), rec(u, dte, "P", strike, cd - 1, putIv)];

test("vsFor: mean of (call iv − put iv) over same-strike same-expiry pairs inside the delta band and DTE window", () => {
  const recs = [
    ...pair("AAA", 30, 100, 0.50, 0.40, 0.02),
    ...pair("AAA", 30, 105, 0.40, 0.42, 0.04),
    ...pair("AAA", 45, 100, 0.55, 0.38, -0.01),
    ...pair("AAA", 30, 120, 0.10, 0.60, 0.50),   // call delta out of band: not a pair
    ...pair("AAA", 5, 100, 0.50, 0.40, 0.30),    // DTE under the window
    rec("AAA", 30, "C", 110, 0.30, 0.45),        // call with no put at that strike
  ];
  const r = vsFor(DAY, "AAA", recs, DAY_MS);
  assert.equal(r.nPairs, 3);
  assert.equal(r.vs, 0.0167); // (0.02 + 0.04 − 0.01) / 3
});

test("vsFor: fewer than 3 pairs, missing/zero iv, bad-delta rows and duplicates never produce a guess", () => {
  const two = [...pair("BBB", 30, 100, 0.5, 0.4, 0.02), ...pair("BBB", 30, 105, 0.45, 0.4, 0.02)];
  assert.equal(vsFor(DAY, "BBB", two, DAY_MS).vs, null);
  assert.equal(vsFor(DAY, "BBB", two, DAY_MS).nPairs, 2);
  const bad = [
    ...pair("CCC", 30, 100, 0.5, 0.4, 0.02), ...pair("CCC", 30, 101, 0.5, 0.4, 0.02), ...pair("CCC", 30, 102, 0.5, 0.4, 0.02),
    rec("CCC", 30, "C", 103, 0.5, null), rec("CCC", 30, "P", 103, -0.5, 0.4),   // no call iv
    rec("CCC", 30, "C", 104, -0.5, 0.4), rec("CCC", 30, "P", 104, -0.5, 0.4),   // call with negative delta
    rec("CCC", 30, "C", 105, 0.5, 0), rec("CCC", 30, "P", 105, -0.5, 0.4),      // zero iv
    ...pair("CCC", 30, 100, 0.5, 0.4, 0.02),                                    // duplicate (archive retry)
  ];
  const r = vsFor(DAY, "CCC", bad, DAY_MS);
  assert.equal(r.nPairs, 3, "duplicates collapse; bad rows drop");
  assert.equal(r.vs, 0.02);
  assert.equal(vsFor(DAY, "CCC", [], DAY_MS).vs, null);
});

test("vsFor: parity-imposed data gives exactly 0 (the gate-1 failure mode is visible, not hidden)", () => {
  const recs = [...pair("DDD", 30, 100, 0.5, 0.4, 0), ...pair("DDD", 30, 105, 0.45, 0.4, 0), ...pair("DDD", 30, 95, 0.55, 0.4, 0)];
  assert.equal(vsFor(DAY, "DDD", recs, DAY_MS).vs, 0);
});

test("vsFor: descriptive atmIv and smirk (OTM put iv minus ATM call iv); null when a side is absent", () => {
  const recs = [
    ...pair("EEE", 30, 100, 0.50, 0.40, 0.02),                                // ATM call iv 0.42
    rec("EEE", 30, "P", 90, -0.25, 0.50),                                     // OTM put in the smirk band
    ...pair("EEE", 30, 95, 0.60, 0.40, 0.00), ...pair("EEE", 30, 96, 0.60, 0.40, 0.00),
  ];
  const r = vsFor(DAY, "EEE", recs, DAY_MS);
  assert.equal(r.atmIv, 0.42);
  assert.equal(r.smirk, 0.08);
  const noAtm = vsFor(DAY, "EEE", [rec("EEE", 30, "P", 90, -0.25, 0.5)], DAY_MS);
  assert.deepEqual([noAtm.atmIv, noAtm.smirk], [null, null]);
});

test("vsForDay: one row per underlying, sorted, uppercase", () => {
  const rows = vsForDay([...pair("zzz", 30, 100, 0.5, 0.4, 0.01), ...pair("AAA", 30, 100, 0.5, 0.4, 0.01)], DAY);
  assert.deepEqual(rows.map((r) => r.u), ["AAA", "ZZZ"]);
});

test("vsGate1Summary: cross-sectional sd per day; a flat day trips the pre-registered kill share; thin days are unscored", () => {
  const mk = (day: string, vals: number[]) => vals.map((vs, i) => ({ day, u: `U${i}`, vs, nPairs: 5, atmIv: 0.3, smirk: null }));
  const flat = Array.from({ length: 12 }, () => 0);
  const wide = Array.from({ length: 12 }, (_, i) => (i - 6) * 0.01);
  const s = vsGate1Summary([...mk("2026-10-01", flat), ...mk("2026-10-02", flat), ...mk("2026-10-03", wide), ...mk("2026-10-04", [0.1, 0.2])]);
  assert.equal(s.days, 4);
  assert.equal(s.daysScored, 3, "the 2-name day is not scored");
  assert.equal(s.daysUnderKillSd, 2);
  assert.equal(s.datasetKill, false, "2 of 3 = 67% < 80%");
  assert.ok(s.perDay[2].sd! > VS_KILL_SD);
  const allFlat = vsGate1Summary([...mk("2026-10-01", flat), ...mk("2026-10-02", flat)]);
  assert.equal(allFlat.datasetKill, true);
  assert.equal(allFlat.medianCrossSectionSd, 0);
  assert.equal(vsGate1Summary([]).datasetKill, false, "no scored day is never a kill");
});

test("derived cache: write-once for finished days, never for today, pending beyond maxCompute, own dir", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "chainvs-"));
  const dir = path.join(base, "optionchains");
  fs.mkdirSync(dir, { recursive: true });
  const lines = (rs: ChainRecord[]) => rs.map((r) => JSON.stringify(r)).join("\n") + "\n";
  const body = (spread: number) => lines([...pair("AAA", 30, 100, 0.5, 0.4, spread), ...pair("AAA", 30, 101, 0.5, 0.4, spread), ...pair("AAA", 30, 102, 0.5, 0.4, spread)]);
  for (const d of ["2026-10-01", "2026-10-02", "2026-10-03"]) fs.writeFileSync(path.join(dir, `${d}.jsonl`), body(0.02));
  const now = Date.parse("2026-10-03T18:00:00Z");
  const a = await vsForDays(["2026-10-01", "2026-10-02", "2026-10-03"], base, now, 2);
  assert.deepEqual([a.computed, a.cached, a.pending], [["2026-10-01", "2026-10-02"], [], ["2026-10-03"]]);
  assert.ok(fs.existsSync(path.join(derivedDir(base), "2026-10-01.json")));
  const b = await vsForDays(["2026-10-01", "2026-10-02", "2026-10-03"], base, now, 2);
  assert.deepEqual([b.computed, b.cached], [["2026-10-03"], ["2026-10-01", "2026-10-02"]]);
  assert.ok(!fs.existsSync(path.join(derivedDir(base), "2026-10-03.json")), "today is never cached");
  fs.writeFileSync(path.join(dir, "2026-10-02.jsonl"), body(0.09));
  const c = await vsForDays(["2026-10-02"], base, now);
  assert.equal(c.rows[0].vs, 0.02, "write-once: a changed source does not change the derived row");
  assert.equal(derivedDir(base).endsWith(path.join("optionchains", "derived", "vs")), true);
});

test("the probe is whitelisted and wired with the iv30 query shape plus the gate-1 summary", () => {
  const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
  const diag = fs.readFileSync(path.join(here, "diag.ts"), "utf8");
  assert.ok(diag.includes('"chain_vs"'));
  const at = bot.indexOf('case "chain_vs": {');
  assert.ok(at > 0);
  const body = bot.slice(at, bot.indexOf("case \"", at + 10));
  for (const s of ["req.query.from", "req.query.to", "req.query.u", "req.query.max", "vsForDays(", "vsGate1Summary(", 'feed: "indicative"', "rows.map((x) => sanitizeDiag(x))"]) {
    assert.ok(body.includes(s), `probe body must include ${s}`);
  }
});
