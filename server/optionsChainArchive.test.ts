// Options-chain daily archiver — pure-helper tests + wiring pins.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "path";
import { fileURLToPath } from "node:url";
import {
  underlyingsFor, parseOcc, withinFilters, compactContract,
  archiveChainRecords, shouldRunNow, snapshotUrl, fetchChainForUnderlying,
  CHAIN_MAX_UNDERLYINGS, CHAIN_MAX_DTE_DAYS,
} from "./optionsChainArchive";

const here = path.dirname(fileURLToPath(import.meta.url));
const NOW = Date.UTC(2026, 6, 8, 20, 30); // Wed 2026-07-08 16:30 ET

test("universe: positions first, then cache candidates with their spot, capped, deduped", () => {
  const cache = { candidates: [["AAPL", 210.5, 1e6, 2.1e8], ["F", 12.1, 5e6, 6e7], ["AAPL", 210.5, 1, 1]] };
  const u = underlyingsFor(cache, ["f", "SPY"]);
  assert.deepEqual(u.map((x) => x.sym), ["F", "SPY", "AAPL"]);
  assert.equal(u.find((x) => x.sym === "AAPL")?.spot, 210.5, "spot must ride along from the cache tuple");
  assert.equal(u.find((x) => x.sym === "SPY")?.spot, null, "position-only underlyings have no cached spot");
  const big = { candidates: Array.from({ length: 500 }, (_, i) => [`T${i}`, 10, 1, 1]) };
  assert.equal(underlyingsFor(big).length, CHAIN_MAX_UNDERLYINGS, "hard cap");
});

test("OCC parse + filters: DTE and strike band enforced; malformed rejected", () => {
  const p = parseOcc("AAPL260717C00200000");
  assert.ok(p && p.type === "C" && p.strike === 200);
  assert.equal(parseOcc("NOT_AN_OCC"), null);
  // exp 2026-07-17 = 9 days out from NOW, strike 200 vs spot 210 → in band
  assert.equal(withinFilters("AAPL260717C00200000", 210, NOW), true);
  // strike 300 vs spot 210 → outside ±20%
  assert.equal(withinFilters("AAPL260717C00300000", 210, NOW), false);
  // expiry beyond 60 days → out
  assert.equal(withinFilters("AAPL270115C00200000", 210, NOW), false);
  assert.ok(CHAIN_MAX_DTE_DAYS === 60);
  // unknown spot (position underlying) → band skipped, DTE still enforced
  assert.equal(withinFilters("AAPL260717C00300000", null, NOW), true);
});

test("compact record: quote required, indicative label NEVER dropped, numbers null-safe", () => {
  const rec = compactContract("F260717P00011000", "F", {
    latestQuote: { bp: 0.55, ap: 0.6, bs: 12, as: 40 },
    impliedVolatility: 0.31, greeks: { delta: -0.22 }, openInterest: 5321,
  }, NOW);
  assert.ok(rec);
  assert.equal(rec!.feed, "indicative", "the honesty label travels with every record");
  assert.equal(rec!.bid, 0.55);
  assert.equal(rec!.delta, -0.22);
  assert.equal(compactContract("X", "X", { greeks: {} }, NOW), null, "no quote = no record");
});

test("day-file append + once-per-ET-day scheduling", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "chains-"));
  const n = archiveChainRecords([
    compactContract("F260717P00011000", "F", { latestQuote: { bp: 0.5, ap: 0.6 } }, NOW)!,
  ], dir, NOW);
  assert.equal(n, 1);
  const fp = path.join(dir, "optionchains", "2026-07-08.jsonl");
  assert.ok(fs.existsSync(fp));
  assert.ok(fs.readFileSync(fp, "utf8").includes('"feed":"indicative"'));
  // scheduling: Wed 16:30 ET runs once, not twice; Saturday never
  const w = shouldRunNow(null, NOW);
  assert.equal(w.run, true);
  assert.equal(shouldRunNow(w.etDay, NOW).run, false, "already ran today");
  assert.equal(shouldRunNow(null, Date.UTC(2026, 6, 11, 20, 30)).run, false, "Saturday");
  assert.equal(shouldRunNow(null, Date.UTC(2026, 6, 8, 17, 0)).run, false, "13:00 ET — market still open");
  fs.rmSync(dir, { recursive: true, force: true });
});

test("fetch: paginates, filters, and surfaces HTTP errors", async () => {
  const pages: Record<string, any> = {
    first: { snapshots: { "F260717P00011000": { latestQuote: { bp: 0.5, ap: 0.6 } }, "F270115P00011000": { latestQuote: { bp: 1, ap: 1.1 } } }, next_page_token: "t2" },
    second: { snapshots: { "F260724P00012000": { latestQuote: { bp: 0.7, ap: 0.8 } } } },
  };
  const fetched: string[] = [];
  const fakeFetch = async (url: string) => {
    fetched.push(url);
    return { ok: true, status: 200, json: async () => (url.includes("page_token") ? pages.second : pages.first) };
  };
  const recs = await fetchChainForUnderlying("F", 12, {} as any, fakeFetch as any, NOW);
  assert.equal(fetched.length, 2, "must follow next_page_token");
  assert.deepEqual(recs.map((r) => r.s).sort(), ["F260717P00011000", "F260724P00012000"], "2027 expiry filtered out");
  assert.ok(snapshotUrl("F").includes("feed=indicative"), "indicative feed pinned in the URL");
  await assert.rejects(
    () => fetchChainForUnderlying("F", 12, {} as any, (async () => ({ ok: false, status: 403, json: async () => ({}) })) as any, NOW),
    /403/,
  );
});

test("wiring pinned: manifest exists with the indicative label; routes boots the archiver", () => {
  const manifest = JSON.parse(fs.readFileSync(path.join(here, "..", "datacore", "manifests", "optionchains.json"), "utf8"));
  assert.equal(manifest.stream, "optionchains");
  assert.ok(/indicative/i.test(manifest.license), "manifest must carry the NOT-NBBO honesty label");
  const routes = fs.readFileSync(path.join(here, "routes.ts"), "utf8");
  assert.ok(routes.includes("bootChainArchive"), "routes.ts must boot the daily chain archiver");
});

// ── [REPAIR 2026-07-05, audit defect #4] day-loss + holiday + gzip ──────────
import { parseMarketHolidays, shouldClaimDay, gzipOldChainDays } from "./optionsChainArchive";
import zlibT from "zlib";

test("parseMarketHolidays reads the real market_calendar.py (frozen source of truth, parsed not duplicated)", () => {
  const py = fs.readFileSync(path.join(path.dirname(new URL(import.meta.url).pathname), "..", "market_calendar.py"), "utf8");
  const h = parseMarketHolidays(py);
  assert.ok(h.has("2026-07-03"), "July 3 (the holiday the archiver wrongly ran on) must parse");
  assert.ok(h.has("2026-12-25") && h.has("2026-01-01"));
  assert.ok(h.size >= 10, `expected >=10 NYSE closures, got ${h.size}`);
});

test("shouldRunNow skips market holidays", () => {
  // Fri 2026-07-03 17:00 ET = 21:00 UTC — after the close, weekday, but a full closure
  const holidayEvening = Date.parse("2026-07-03T21:00:00Z");
  const h = new Set(["2026-07-03"]);
  assert.equal(shouldRunNow(null, holidayEvening, h).run, false, "holiday must not run");
  assert.equal(shouldRunNow(null, holidayEvening, new Set()).run, true, "same instant runs when not a holiday");
});

test("shouldClaimDay: crash-safety semantics — total failure never claims, partial success does", () => {
  assert.equal(shouldClaimDay({ underlyings: 120, contracts: 50000, failures: 3 }), true, "partial failure claims");
  assert.equal(shouldClaimDay({ underlyings: 120, contracts: 0, failures: 120 }), false, "total failure retries");
  assert.equal(shouldClaimDay({ underlyings: 0, contracts: 0, failures: 0 }), true, "empty universe is not a failure");
});

test("gzipOldChainDays compresses 2-day-old files (the manifest's promised lifecycle)", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "vtchain-"));
  const sub = path.join(dir, "optionchains");
  fs.mkdirSync(sub, { recursive: true });
  fs.writeFileSync(path.join(sub, "2026-07-01.jsonl"), '{"s":"X"}\n');
  fs.writeFileSync(path.join(sub, "2026-07-05.jsonl"), '{"s":"Y"}\n');
  const n = gzipOldChainDays(dir, Date.parse("2026-07-05T12:00:00Z"));
  assert.equal(n, 1);
  assert.ok(fs.existsSync(path.join(sub, "2026-07-01.jsonl.gz")), "old day gzipped");
  assert.ok(fs.existsSync(path.join(sub, "2026-07-05.jsonl")), "recent day untouched");
  assert.equal(zlibT.gunzipSync(fs.readFileSync(path.join(sub, "2026-07-01.jsonl.gz"))).toString(), '{"s":"X"}\n');
  fs.rmSync(dir, { recursive: true, force: true });
});
