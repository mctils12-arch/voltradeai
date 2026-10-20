// chainIv — ATM IV30 from the chain archive (research test 2 step 1,
// [RULE-REVIEW] 2026-10-20). Run: npx tsx --test server/chainIv.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "path";
import zlib from "zlib";
import { atmByExpiry, iv30From, atmIv30ForDay, listChainDays, readChainDay, iv30ForDays, derivedDir, IV30_NEAREST_DTE } from "./chainIv";
import type { ChainRecord } from "./optionsChainArchive";

const DAY = "2026-10-01";
const DAY_MS = Date.parse(`${DAY}T00:00:00Z`);
// OCC for an expiry `dte` days after DAY
const occ = (u: string, dte: number, type: "C" | "P", strike: number) => {
  const d = new Date(DAY_MS + dte * 86_400_000);
  const yy = String(d.getUTCFullYear()).slice(2), mm = String(d.getUTCMonth() + 1).padStart(2, "0"), dd = String(d.getUTCDate()).padStart(2, "0");
  return `${u}${yy}${mm}${dd}${type}${String(Math.round(strike * 1000)).padStart(8, "0")}`;
};
const rec = (u: string, dte: number, type: "C" | "P", strike: number, delta: number | null, iv: number | null): ChainRecord =>
  ({ s: occ(u, dte, type, strike), u, t: DAY_MS + 20 * 3_600_000, bid: 1, ask: 1.1, bsz: 1, asz: 1, iv, delta, oi: 10, feed: "indicative" });

test("atmByExpiry: the call nearest +0.5 and the put nearest −0.5 inside the band, averaged; out-of-band, bad and empty rows ignored; duplicates harmless", () => {
  const recs = [
    rec("AAA", 20, "C", 100, 0.52, 0.42), rec("AAA", 20, "C", 105, 0.40, 0.50), rec("AAA", 20, "P", 100, -0.48, 0.38),
    rec("AAA", 20, "P", 90, -0.20, 0.60),      // out of band
    rec("AAA", 20, "C", 100, 0.52, 0.42),      // duplicate (archive retry)
    rec("AAA", 20, "C", 95, 0.62, null),       // no IV
    rec("AAA", 20, "P", 100, 0.45, 0.99),      // a put with a positive delta: bad row
    rec("AAA", 3, "C", 100, 0.50, 0.90),       // inside IV30_MIN_DTE: ignored
    rec("AAA", 48, "C", 100, 0.51, 0.30),      // second expiry, one side only
  ];
  const ex = atmByExpiry(recs, DAY_MS);
  assert.deepEqual([...ex.keys()].sort((a, b) => a - b), [20, 48]);
  const e20 = ex.get(20)!;
  assert.equal(e20.call, 0.42);
  assert.equal(e20.put, 0.38);
  assert.equal(e20.iv, 0.4);
  assert.equal(e20.n, 4, "in-band contracts counted (incl. the duplicate)");
  assert.equal(ex.get(48)!.iv, 0.30);
  assert.equal(ex.get(48)!.put, null);
});

test("iv30From: total-variance interpolation between the bracketing expiries; exact 30; nearest fallback inside 20–45; null otherwise", () => {
  const ex = (pts: Array<[number, number]>) => new Map(pts.map(([dte, iv]) => [dte, { dte, iv, n: 2, call: iv, put: iv }]));
  // V_lo = 0.16*20 = 3.2, V_hi = 0.09*48 = 4.32, V30 = 3.2 + 1.12*(10/28) = 3.6 → sqrt(3.6/30)
  const r = iv30From(ex([[20, 0.40], [48, 0.30]]));
  assert.deepEqual(r, { iv30: 0.3464, method: "interp", dteLo: 20, dteHi: 48 });
  assert.deepEqual(iv30From(ex([[30, 0.33]])), { iv30: 0.33, method: "interp", dteLo: 30, dteHi: 30 });
  assert.deepEqual(iv30From(ex([[25, 0.5]])), { iv30: 0.5, method: "nearest", dteLo: 25, dteHi: 25 });
  assert.deepEqual(iv30From(ex([[45, 0.5]])).method, "nearest");
  assert.deepEqual(iv30From(ex([[IV30_NEAREST_DTE[1] + 1, 0.5]])), { iv30: null, method: null, dteLo: null, dteHi: null });
  assert.deepEqual(iv30From(ex([[10, 0.5]])).iv30, null);
  assert.equal(iv30From(new Map()).iv30, null);
  // three expiries: the bracket is the nearest pair around 30, not the extremes
  assert.deepEqual(iv30From(ex([[10, 0.9], [27, 0.40], [34, 0.40], [90, 0.1]])), { iv30: 0.4, method: "interp", dteLo: 27, dteHi: 34 });
});

test("atmIv30ForDay: one row per underlying, sorted, with counts", () => {
  const recs = [rec("BBB", 20, "C", 50, 0.5, 0.6), rec("AAA", 20, "C", 100, 0.5, 0.4), rec("AAA", 48, "P", 100, -0.5, 0.3), rec("CCC", 90, "C", 10, 0.5, 0.2)];
  const rows = atmIv30ForDay(recs, DAY);
  assert.deepEqual(rows.map((r) => r.u), ["AAA", "BBB", "CCC"]);
  assert.equal(rows[0].iv30, 0.3464);
  assert.equal(rows[0].n, 2);
  assert.equal(rows[0].expiries, 2);
  assert.deepEqual([rows[1].iv30, rows[1].method], [0.6, "nearest"]);
  assert.deepEqual([rows[2].iv30, rows[2].method, rows[2].expiries], [null, null, 1]);
  assert.equal(rows[0].day, DAY);
});

test("archive reading + derived cache: gz and plain day files, write-once for finished days, never for today, pending beyond maxCompute", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "chainiv-"));
  const dir = path.join(base, "optionchains");
  fs.mkdirSync(dir, { recursive: true });
  const lines = (rs: ChainRecord[]) => rs.map((r) => JSON.stringify(r)).join("\n") + "\n";
  fs.writeFileSync(path.join(dir, "2026-10-01.jsonl.gz"), zlib.gzipSync(lines([rec("AAA", 20, "C", 100, 0.5, 0.4), rec("AAA", 48, "P", 100, -0.5, 0.3)])));
  fs.writeFileSync(path.join(dir, "2026-10-02.jsonl"), lines([rec("AAA", 30, "C", 100, 0.5, 0.5)]) + "not json\n");
  fs.writeFileSync(path.join(dir, "2026-10-03.jsonl"), lines([rec("AAA", 30, "C", 100, 0.5, 0.7)]));
  fs.writeFileSync(path.join(dir, "notes.txt"), "x");
  assert.deepEqual(listChainDays(base), ["2026-10-01", "2026-10-02", "2026-10-03"]);
  assert.equal((await readChainDay("2026-10-01", base)).length, 2);
  assert.equal((await readChainDay("2026-10-02", base)).length, 1, "a bad line is skipped");
  assert.deepEqual(await readChainDay("2026-09-30", base), []);

  const now = Date.parse("2026-10-03T18:00:00Z"); // 10-03 is "today": never cached
  const a = await iv30ForDays(["2026-10-01", "2026-10-02", "2026-10-03"], base, now, 2);
  assert.deepEqual([a.computed, a.cached, a.pending], [["2026-10-01", "2026-10-02"], [], ["2026-10-03"]]);
  assert.equal(a.rows.length, 2);
  assert.ok(fs.existsSync(path.join(derivedDir(base), "2026-10-01.json")));
  assert.ok(fs.existsSync(path.join(derivedDir(base), "2026-10-02.json")));
  const b = await iv30ForDays(["2026-10-01", "2026-10-02", "2026-10-03"], base, now, 2);
  assert.deepEqual([b.computed, b.cached, b.pending], [["2026-10-03"], ["2026-10-01", "2026-10-02"], []]);
  assert.equal(b.rows.length, 3);
  assert.ok(!fs.existsSync(path.join(derivedDir(base), "2026-10-03.json")), "today is never cached");
  // write-once: a changed source day does not change the derived row
  fs.writeFileSync(path.join(dir, "2026-10-02.jsonl"), lines([rec("AAA", 30, "C", 100, 0.5, 0.9)]));
  const c = await iv30ForDays(["2026-10-02"], base, now);
  assert.equal(c.rows[0].iv30, 0.5);
  assert.equal(fs.statSync(path.join(dir, "2026-10-01.jsonl.gz")).size > 0, true, "source files untouched");
  assert.deepEqual(listChainDays(path.join(base, "nowhere")), []);
});
