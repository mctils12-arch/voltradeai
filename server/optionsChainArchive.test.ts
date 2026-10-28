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
  CHAIN_MAX_UNDERLYINGS, CHAIN_MAX_DTE_DAYS, CHAIN_ALWAYS,
} from "./optionsChainArchive";

const here = path.dirname(fileURLToPath(import.meta.url));
const NOW = Date.UTC(2026, 6, 8, 20, 30); // Wed 2026-07-08 16:30 ET

test("universe: the index chains first, then positions, then cache candidates with their spot, capped, deduped", () => {
  const cache = { candidates: [["AAPL", 210.5, 1e6, 2.1e8], ["F", 12.1, 5e6, 6e7], ["AAPL", 210.5, 1, 1]] };
  const u = underlyingsFor(cache, ["f", "SPY"]);
  assert.deepEqual(u.map((x) => x.sym), ["SPY", "QQQ", "IWM", "F", "AAPL"]);
  assert.equal(u.find((x) => x.sym === "AAPL")?.spot, 210.5, "spot must ride along from the cache tuple");
  assert.equal(u.find((x) => x.sym === "SPY")?.spot, null, "pinned/position-only underlyings have no cached spot");
  const big = { candidates: Array.from({ length: 500 }, (_, i) => [`T${i}`, 10, 1, 1]) };
  const capped = underlyingsFor(big);
  assert.equal(capped.length, CHAIN_MAX_UNDERLYINGS, "hard cap");
  assert.deepEqual(capped.slice(0, 3).map((x) => x.sym), [...CHAIN_ALWAYS], "the index names survive a full cache — the 2026-10-20 gap");
});

test("universe: an index name that the cache also carries keeps its pinned slot but takes the cache's spot", () => {
  const cache = { candidates: [...Array.from({ length: 200 }, (_, i) => [`T${i}`, 10, 1, 1]), ["SPY", 598.2, 1e6, 1e9]] };
  const u = underlyingsFor(cache);
  assert.equal(u[0].sym, "SPY");
  assert.equal(u[0].spot, 598.2, "the cache's spot replaces the pinned null so the strike band applies");
  assert.equal(u.length, CHAIN_MAX_UNDERLYINGS);
  assert.equal(new Set(u.map((x) => x.sym)).size, u.length, "no duplicates");
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

// ── [REPAIR 2026-10-27, audit §2.4 defect 1] day label + window + idempotency ──
import {
  etMinutesFromParts, etClock, captureTradingDay, isDayCaptured,
  chainDayFile, chainStagingFile, chainCapturedFile,
  CHAIN_RUN_AFTER_ET_MINUTES, CHAIN_RUN_BEFORE_ET_MINUTES,
} from "./optionsChainArchive";

// Tue 2026-10-27 (EDT, UTC-4)
const ET = (h: number, m: number) => Date.UTC(2026, 9, 27, h + 4, m);
const NO_HOLIDAYS = new Set<string>();

test("ET clock: an h24 runtime's '24' hour is midnight, not minute 1440 (the Node 20 misparse)", () => {
  // exactly what Node 20's hour12:false formatter produced for 2026-10-21T04:46Z
  const node20Midnight = [
    { type: "weekday", value: "Wed" }, { type: "literal", value: " " },
    { type: "hour", value: "24" }, { type: "literal", value: ":" }, { type: "minute", value: "46" },
  ];
  assert.equal(etMinutesFromParts(node20Midnight), 46);
  const c = etClock(Date.UTC(2026, 9, 21, 4, 46));
  assert.equal(c.mins, 46, "00:46 ET");
  assert.equal(c.etDay, "2026-10-21");
  assert.equal(c.dow, "Wed");
});

test("window: runs only inside 16:30–18:30 ET on a trading day — never at midnight, never late evening", () => {
  assert.equal(CHAIN_RUN_AFTER_ET_MINUTES, 16 * 60 + 30);
  assert.equal(CHAIN_RUN_BEFORE_ET_MINUTES, 18 * 60 + 30);
  assert.equal(shouldRunNow(null, Date.UTC(2026, 9, 21, 4, 46), NO_HOLIDAYS).run, false, "00:46 ET — the prod misfire");
  assert.equal(shouldRunNow(null, ET(16, 20), NO_HOLIDAYS).run, false, "16:20 ET — index options still trading");
  const ok = shouldRunNow(null, ET(16, 45), NO_HOLIDAYS);
  assert.deepEqual(ok, { run: true, etDay: "2026-10-27" });
  assert.equal(shouldRunNow(null, ET(18, 29), NO_HOLIDAYS).run, true, "catch-up inside the window");
  assert.equal(shouldRunNow(null, ET(20, 30), NO_HOLIDAYS).run, false, "20:30 ET = 00:30Z tomorrow — outside the window");
});

test("day label is the ET trading date, never the UTC date", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "chains-et-"));
  const late = Date.UTC(2026, 9, 28, 0, 30); // 20:30 ET Tue 2026-10-27
  archiveChainRecords([compactContract("F261120P00011000", "F", { latestQuote: { bp: 0.5, ap: 0.6 } }, late)!], dir, late);
  assert.ok(fs.existsSync(path.join(dir, "optionchains", "2026-10-27.jsonl")), "labelled with the ET session");
  assert.ok(!fs.existsSync(path.join(dir, "optionchains", "2026-10-28.jsonl")), "not the UTC date");
  fs.rmSync(dir, { recursive: true, force: true });
});

function chainFake(calls: string[]) {
  return async (url: string) => {
    calls.push(url);
    const u = /snapshots\/([A-Z]+)\?/.exec(url)?.[1] || "X";
    return { ok: true, status: 200, json: async () => ({
      snapshots: { [`${u}261120P00100000`]: { latestQuote: { bp: 1, ap: 1.1 } } },
    }) };
  };
}
const lines = (fp: string) => fs.readFileSync(fp, "utf8").trim().split("\n").filter(Boolean);

test("idempotent per ET trading date: a second capture (restart / extra tick) is skipped, not appended", async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "chains-idem-"));
  const calls: string[] = [];
  const opts = { baseDir: dir, sleepMs: 0, cacheJson: null };
  const a = await captureTradingDay({} as any, chainFake(calls) as any, [], ET(16, 35), opts);
  assert.equal(a.status, "published");
  assert.equal(a.day, "2026-10-27");
  const fp = chainDayFile("2026-10-27", dir);
  assert.equal(lines(fp).length, 3, "SPY/QQQ/IWM one contract each");
  assert.ok(fs.existsSync(chainCapturedFile("2026-10-27", dir)), "sidecar marks the day");
  assert.ok(!fs.existsSync(chainStagingFile("2026-10-27", dir)), "staging consumed");
  const n = calls.length;
  const b = await captureTradingDay({} as any, chainFake(calls) as any, [], ET(16, 40), opts);
  assert.equal(b.status, "skipped");
  assert.equal(calls.length, n, "no API calls on a captured day");
  assert.equal(lines(fp).length, 3, "no duplicate rows");
  // crash between publish and sidecar: the published file's post-close t repairs the sidecar
  fs.unlinkSync(chainCapturedFile("2026-10-27", dir));
  assert.equal(isDayCaptured("2026-10-27", dir), true);
  assert.ok(fs.existsSync(chainCapturedFile("2026-10-27", dir)));
  fs.rmSync(dir, { recursive: true, force: true });
});

test("restart mid-capture: a leftover partial stage is discarded, the day is written once", async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "chains-restart-"));
  const staged = chainStagingFile("2026-10-27", dir);
  fs.mkdirSync(path.dirname(staged), { recursive: true });
  fs.writeFileSync(staged, '{"s":"SPY261120P00100000","u":"SPY","t":1}\n{"s":"QQ'); // crashed mid-write
  const r = await captureTradingDay({} as any, chainFake([]) as any, [], ET(16, 50), { baseDir: dir, sleepMs: 0, cacheJson: null });
  assert.equal(r.status, "published");
  const rows = lines(chainDayFile("2026-10-27", dir)).map((l) => JSON.parse(l));
  assert.equal(rows.length, 3);
  assert.ok(rows.every((x) => x.t === ET(16, 50)), "only the fresh capture");
  fs.rmSync(dir, { recursive: true, force: true });
});

test("a pre-fix midnight file under today's label does not block the real capture and is kept aside", async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "chains-legacy-"));
  const fp = chainDayFile("2026-10-27", dir);
  fs.mkdirSync(path.dirname(fp), { recursive: true });
  const legacyT = ET(0, 40); // 00:40 ET — previous session's close
  fs.writeFileSync(fp, JSON.stringify({ s: "SPY261120P00100000", u: "SPY", t: legacyT, feed: "indicative" }) + "\n");
  assert.equal(isDayCaptured("2026-10-27", dir), false, "midnight capture is not a post-close capture for the day");
  const r = await captureTradingDay({} as any, chainFake([]) as any, [], ET(16, 31), { baseDir: dir, sleepMs: 0, cacheJson: null });
  assert.equal(r.status, "published");
  const rows = lines(fp).map((l) => JSON.parse(l));
  assert.ok(rows.every((x) => x.t === ET(16, 31)), "the day file holds only the post-close capture");
  const aside = path.join(dir, "optionchains", "_superseded", "2026-10-27.jsonl");
  assert.ok(fs.existsSync(aside), "legacy rows moved, never deleted");
  assert.equal(JSON.parse(lines(aside)[0]).t, legacyT);
  fs.rmSync(dir, { recursive: true, force: true });
});

test("total failure stages nothing and claims nothing (retry next tick)", async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "chains-fail-"));
  const bad = async () => ({ ok: false, status: 500, json: async () => ({}) });
  const r = await captureTradingDay({} as any, bad as any, [], ET(16, 31), { baseDir: dir, sleepMs: 0, cacheJson: null });
  assert.equal(r.status, "unclaimed");
  assert.equal(isDayCaptured("2026-10-27", dir), false);
  assert.ok(!fs.existsSync(chainDayFile("2026-10-27", dir)));
  fs.rmSync(dir, { recursive: true, force: true });
});

// ── [REPAIR 2026-10-27, audit §2.4 defect 2] full index chains (pagination) ──
import { fetchChainPages, chainQueryWindow, CHAIN_MAX_PAGES } from "./optionsChainArchive";

/** A paged chain like SPY's: `n` pages in OCC order (nearest expiry first),
 *  one in-scope contract per page, each a later expiry than the last. */
function pagedChain(n: number, calls: string[], repeatToken = false) {
  return async (url: string) => {
    calls.push(url);
    const m = /page_token=p(\d+)/.exec(url);
    const page = m ? Number(m[1]) : 0;
    const exp = new Date(Date.UTC(2026, 9, 28 + page * 4)); // +4 days per page, all ≤ 60 DTE for 12 pages
    const yymmdd = exp.toISOString().slice(2, 10).replace(/-/g, "");
    const occ = `SPY${yymmdd}P00590000`;
    const next = repeatToken ? "p1" : page + 1 < n ? `p${page + 1}` : undefined;
    return { ok: true, status: 200, json: async () => ({ snapshots: { [occ]: { latestQuote: { bp: 1, ap: 1.1 } } }, ...(next ? { next_page_token: next } : {}) }) };
  };
}

test("pagination: a chain longer than the old 5-page cap is fetched whole (SPY lost every expiry past ~39 DTE)", async () => {
  const calls: string[] = [];
  const got = await fetchChainPages("SPY", 600, {} as any, pagedChain(12, calls) as any, ET(16, 35));
  assert.equal(calls.length, 12, "follows next_page_token to exhaustion");
  assert.equal(got.pages, 12);
  assert.equal(got.records.length, 12, "every page's contracts kept — the far expiries included");
  assert.equal(got.truncated, false);
  assert.ok(CHAIN_MAX_PAGES >= 20, "safety cap well above a ≤60-DTE index chain");
});

test("pagination: the documented scope is asked of the server (≤60 DTE, ±20% band when spot is known)", async () => {
  const calls: string[] = [];
  await fetchChainPages("SPY", 600, {} as any, pagedChain(2, calls) as any, ET(16, 35));
  assert.ok(calls[0].includes("expiration_date_gte=2026-10-27"), "0DTE kept: from the ET capture date");
  assert.ok(calls[0].includes("expiration_date_lte=2026-12-26"), "capture date + 60 calendar days");
  assert.ok(calls[0].includes("strike_price_gte=480.00") && calls[0].includes("strike_price_lte=720.00"));
  assert.ok(calls[1].includes("page_token=p1") && calls[1].includes("expiration_date_lte="), "bounds ride on every page");
  assert.deepEqual(chainQueryWindow(ET(16, 35), null), { expGte: "2026-10-27", expLte: "2026-12-26" }, "unknown spot: no strike bound");
});

test("pagination: hitting the safety cap or a repeated token is flagged truncated, never silent", async () => {
  const capped = await fetchChainPages("SPY", 600, {} as any, pagedChain(12, []) as any, ET(16, 35), 3);
  assert.equal(capped.pages, 3);
  assert.equal(capped.truncated, true);
  const calls: string[] = [];
  const looped = await fetchChainPages("SPY", 600, {} as any, pagedChain(99, calls, true) as any, ET(16, 35));
  assert.equal(looped.truncated, true);
  assert.ok(calls.length <= 3, "a repeated page token cannot loop the capture");
});

test("pagination: the run summary and the day sidecar name truncated underlyings", async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "chains-trunc-"));
  // every underlying serves an endless chain → each one truncates at the cap
  const endless = async (url: string) => {
    const m = /page_token=p(\d+)/.exec(url);
    const page = m ? Number(m[1]) : 0;
    const u = /snapshots\/([A-Z]+)\?/.exec(url)?.[1] || "X";
    return { ok: true, status: 200, json: async () => ({ snapshots: { [`${u}261120P00100000`]: { latestQuote: { bp: 1, ap: 1.1 } } }, next_page_token: `p${page + 1}` }) };
  };
  const r = await captureTradingDay({} as any, endless as any, [], ET(16, 35), { baseDir: dir, sleepMs: 0, cacheJson: null });
  assert.equal(r.status, "published");
  assert.deepEqual(r.truncated, ["SPY", "QQQ", "IWM"]);
  const side = JSON.parse(fs.readFileSync(chainCapturedFile("2026-10-27", dir), "utf8"));
  assert.deepEqual(side.truncated, ["SPY", "QQQ", "IWM"]);
  assert.equal(side.scope.max_dte_days, 60);
  fs.rmSync(dir, { recursive: true, force: true });
});
