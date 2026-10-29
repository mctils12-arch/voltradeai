// pricePanel — nightly survivorship-reduced 2016+ daily price panel from
// Alpaca free bars (2026-10-28). Run: npx tsx --test server/pricePanel.test.ts
// Mock fetch + a temp dir: no network, no real keys, no real volume.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import { fileURLToPath } from "node:url";
import {
  pricePanelFlag, pricePanelEnabled, inBuildWindow, parseAssets, parseCorporateActions, buildUniverse,
  encodeSegment, decodeSegment, scaleFor, splitSegments, linesFor, runPanelSession, pricePanelStatus,
  PRICE_PANEL_SUBDIR, BACKOFF_429_MS, type Row, type UniverseEntry, type PanelLine,
} from "./pricePanel";
import { JOB_REGISTRY } from "./jobHeartbeat";
import type { FetchLike } from "./optionsEntitlements";
import { sanitizeDiag, DIAG_PROBES } from "./diag";

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.join(here, "..");
const ENV = { ALPACA_KEY: "TESTALPACAKEYID0001", ALPACA_SECRET: "TESTALPACASECRET0002" };
// 2026-10-28T01:00Z = 21:00 ET on 2026-10-27: inside the build window.
const T0 = Date.parse("2026-10-28T01:00:00Z");

const bdays = (from: string, to: string): string[] => {
  const out: string[] = [];
  for (let t = Date.parse(`${from}T00:00:00Z`); t <= Date.parse(`${to}T00:00:00Z`); t += 86_400_000) {
    const wd = new Date(t).getUTCDay();
    if (wd !== 0 && wd !== 6) out.push(new Date(t).toISOString().slice(0, 10));
  }
  return out;
};
const series = (days: string[], base: number) => days.map((d, i) => ({ d, c: Number((base + i * 0.01).toFixed(4)) }));

// What Alpaca would return per (symbol, mapping).
const DEFAULT: Record<string, Array<{ d: string; c: number }>> = {
  AAA: series(bdays("2026-06-01", "2026-10-27"), 50),
  S: series(bdays("2019-01-02", "2020-03-31"), 5),          // default mapping -> OLD company (live finding)
  TWTR: series(bdays("2022-06-01", "2023-01-10"), 40),      // runs past the 2022-10-27 event
  MON: series(bdays("2018-01-02", "2018-06-06"), 100),
  ZZZ: series(bdays("2017-01-03", "2017-05-01"), 0.0042),   // sub-cent: needs a finer scale
};
const RAW: Record<string, Array<{ d: string; c: number }>> = {
  S: [...series(bdays("2019-01-02", "2020-03-31"), 5), ...series(bdays("2021-06-30", "2026-10-27"), 30)],
};

interface Seen { url: string; method: string; headers: Record<string, string> }

function api(opts: { seen?: Seen[]; first429?: boolean } = {}): FetchLike {
  let sent429 = false;
  return async (url, init) => {
    opts.seen?.push({ url, method: init.method, headers: init.headers });
    const u = new URL(url);
    const ok = (body: unknown) => ({ status: 200, json: async () => body });
    if (opts.first429 && !sent429) { sent429 = true; return { status: 429, json: async () => ({ message: "too many requests" }) }; }
    if (u.pathname === "/v2/assets") {
      return u.searchParams.get("status") === "active"
        ? ok([{ id: "a-aaa", symbol: "AAA", exchange: "NASDAQ" }, { id: "a-s1", symbol: "S", exchange: "NYSE" },
          { id: "a-bbb", symbol: "BBB", exchange: "OTC" }, { id: "a-bad", symbol: "BAD", exchange: "NYSE" }])
        : ok([{ id: "i-twtr", symbol: "TWTR", exchange: "NYSE" }, { id: "i-zzz", symbol: "ZZZ", exchange: "NASDAQ" }]);
    }
    if (u.pathname === "/v1beta1/corporate-actions") {
      const y = u.searchParams.get("start")!.slice(0, 4);
      const ca: Record<string, unknown[]> = {};
      if (y === "2017") ca.name_changes = [{ old_symbol: "YHOO", new_symbol: "AABA", process_date: "2017-06-16" }];
      if (y === "2018") ca.cash_mergers = [{ acquiree_symbol: "MON", acquirer_symbol: "BAYRY", effective_date: "2018-06-07" }];
      if (y === "2022") ca.cash_mergers = [{ acquiree_symbol: "TWTR", acquirer_symbol: "", effective_date: "2022-10-27" }];
      return ok({ corporate_actions: ca, next_page_token: null });
    }
    if (u.pathname === "/v2/stocks/bars") {
      const syms = String(u.searchParams.get("symbols")).split(",");
      if (syms.includes("BAD")) return { status: 422, json: async () => ({ message: "invalid symbol: BAD" }) };
      const src = u.searchParams.get("asof") === "-" ? { ...DEFAULT, ...RAW } : DEFAULT;
      const rawAdj = u.searchParams.get("adjustment") === "raw";
      const page2 = u.searchParams.get("page_token") === "p2";
      const bars: Record<string, unknown[]> = {};
      for (const s of syms) {
        const rows = src[s];
        if (!rows) continue;
        const half = Math.ceil(rows.length / 2);
        // raw = 2x adjusted before 2020 (a 2:1 split), equal after.
        bars[s] = (page2 ? rows.slice(half) : rows.slice(0, half)).map((r) => {
          const c = rawAdj && r.d < "2020-01-01" ? r.c * 2 : r.c;
          return { t: `${r.d}T04:00:00Z`, o: c, h: c * 1.01, l: c * 0.99, c, v: 1000.4, n: 5, vw: c };
        });
      }
      return ok({ bars, next_page_token: page2 ? null : "p2" });
    }
    return { status: 404, json: async () => ({ message: "no route" }) };
  };
}

function tmpRoot(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), "pp-test-"));
}

function clockDeps(start = T0) {
  let t = start;
  const slept: number[] = [];
  return { now: () => t, sleep: async (ms: number) => { slept.push(ms); t += ms; }, slept, advance: (ms: number) => { t += ms; } };
}

function readLines(root: string): PanelLine[] {
  const cur = JSON.parse(fs.readFileSync(path.join(root, "current.json"), "utf8"));
  const dir = path.join(root, cur.build_id, "bars");
  return fs.readdirSync(dir).sort().flatMap((f) =>
    zlib.gunzipSync(fs.readFileSync(path.join(dir, f))).toString("utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l) as PanelLine));
}

test("flag: env present wins (0/off/empty disable), absent = ON; registry enabledWhen agrees", () => {
  const job = JOB_REGISTRY.find((j) => j.id === "price_panel_build");
  assert.ok(job && job.kind === "daily" && job.source === "server/pricePanel.ts");
  for (const [v, on] of [[undefined, true], ["1", true], ["on", true], ["0", false], ["off", false], ["", false], ["false", false]] as const) {
    const env = { ...ENV, ...(v === undefined ? {} : { VOLTRADE_PRICE_PANEL: v }) };
    assert.equal(pricePanelFlag(env).on, on, `value ${v}`);
    assert.equal(pricePanelEnabled(env), on);
    assert.equal(job!.enabledWhen!(env), on, `registry disagrees on ${v}`);
  }
  assert.equal(pricePanelFlag({}).source, "default");
  assert.equal(pricePanelEnabled({ VOLTRADE_PRICE_PANEL: "1" }), false, "no keys = disabled");
});

test("window: 20:15-04:00 ET only (Tier 2 scans own 04:00-20:00 ET)", () => {
  assert.equal(inBuildWindow(Date.parse("2026-10-28T01:00:00Z")), true);   // 21:00 ET
  assert.equal(inBuildWindow(Date.parse("2026-10-28T07:30:00Z")), true);   // 03:30 ET
  assert.equal(inBuildWindow(Date.parse("2026-10-28T08:30:00Z")), false);  // 04:30 ET
  assert.equal(inBuildWindow(Date.parse("2026-10-27T16:00:00Z")), false);  // 12:00 ET
  assert.equal(inBuildWindow(Date.parse("2026-10-28T00:05:00Z")), false);  // 20:05 ET, before due
});

test("universe: OTC excluded, corp-action-only names added, reuse and events attached", () => {
  const assets = [
    ...parseAssets([{ id: "x1", symbol: "AAA", exchange: "NASDAQ" }, { id: "x2", symbol: "S", exchange: "NYSE" }, { id: "x3", symbol: "OTCQ", exchange: "OTC" }, { symbol: "bad sym" }], "active"),
    ...parseAssets([{ id: "y1", symbol: "S", exchange: "NYSE" }, { id: "y2", symbol: "OLD", exchange: "NYSE" }], "inactive"),
  ];
  const { events, next } = parseCorporateActions({
    corporate_actions: {
      stock_mergers: [{ acquiree_symbol: "EMC", acquirer_symbol: "DVMT", effective_date: "2016-09-07" }],
      worthless_removals: [{ symbol: "OLD", process_date: "2019-02-01" }],
      name_changes: [{ old_symbol: "PRE", new_symbol: "NEW", process_date: "2014-01-01" }],
    }, next_page_token: "n",
  });
  assert.equal(next, "n");
  const { entries, excluded } = buildUniverse(assets, events);
  const by = Object.fromEntries(entries.map((e) => [e.symbol, e]));
  assert.equal(excluded, 1);
  assert.deepEqual(entries.map((e) => e.symbol), ["AAA", "EMC", "OLD", "S"], "sorted; pre-2016 PRE dropped");
  assert.equal(by.S.reuse_suspect, true);
  assert.deepEqual(by.S.ids, ["x2", "y1"]);
  assert.equal(by.EMC.status, "corp_action");
  assert.deepEqual(by.OLD.event, { kind: "worthless_removal", date: "2019-02-01" });
  assert.equal(by.AAA.reuse_suspect, false);
});

test("encoding: lossless to the scale, sub-cent series get a finer scale, raw-close factor runs", () => {
  const rows: Row[] = [
    { d: "2019-12-30", o: 10.5, h: 11, l: 10, c: 10.75, v: 100, rc: 21.5 },
    { d: "2019-12-31", o: 10.75, h: 10.9, l: 10.6, c: 10.8, v: 200.6, rc: 21.6 },
    { d: "2020-01-02", o: 10.8, h: 11.2, l: 10.7, c: 11.1, v: 300, rc: 11.1 },
    { d: "2020-01-03", o: 11.1, h: 11.3, l: 11, c: 11.2, v: 0, rc: null },
  ];
  const e = encodeSegment(rows);
  assert.equal(e.k, 4);
  assert.deepEqual(e.dd, [0, 1, 2, 1]);
  assert.deepEqual(e.f, [[0, 2], [2, 1]], "one run per adjustment regime");
  const back = decodeSegment(e);
  rows.forEach((r, i) => {
    for (const k of ["o", "h", "l", "c"] as const) assert.ok(Math.abs(back[i][k] - r[k]) < 1e-9, `${k}@${i}`);
    assert.equal(back[i].d, r.d);
    assert.equal(back[i].v, Math.round(r.v));
  });
  assert.ok(Math.abs((back[1].rc ?? 0) - 21.6) < 1e-9);
  assert.ok(Math.abs((back[3].rc ?? 0) - 11.2) < 1e-9, "a missing raw close inherits the last factor");
  const tiny: Row[] = [{ d: "2017-01-03", o: 0.0042, h: 0.0043, l: 0.0041, c: 0.00423, v: 1, rc: null }];
  assert.equal(scaleFor(tiny), 7);
  assert.ok(Math.abs(decodeSegment(encodeSegment(tiny))[0].c - 0.00423) < 1e-12);
});

test("segments split at > 30-day gaps; clipping after a dated event; flags", () => {
  const r = (d: string): Row => ({ d, o: 1, h: 1, l: 1, c: 1, v: 1, rc: 1 });
  assert.equal(splitSegments([r("2020-03-30"), r("2020-03-31"), r("2021-06-30")]).length, 2);
  assert.equal(splitSegments([r("2020-03-01"), r("2020-03-31")]).length, 1);
  const twtr: UniverseEntry = { symbol: "TWTR", status: "inactive", ids: ["i"], event: { kind: "merger", date: "2022-10-27" }, reuse_suspect: false };
  const res = linesFor(twtr, "default", [r("2022-10-26"), r("2022-10-27"), r("2022-10-28"), r("2022-11-01")]);
  assert.equal(res.clipped, 2);
  assert.equal(res.lines[0].seg_end, "2022-10-27");
  assert.ok(res.lines[0].flags.includes("clipped_after_event"));
  const live: UniverseEntry = { symbol: "AAA", status: "active", ids: ["a"], event: { kind: "merger", date: "2022-10-27" }, reuse_suspect: false };
  assert.equal(linesFor(live, "default", [r("2022-10-26"), r("2022-11-01")]).clipped, 0, "never clip a live, non-reused company");
  const zzz: UniverseEntry = { symbol: "ZZZ", status: "inactive", ids: [], event: null, reuse_suspect: false };
  assert.ok(linesFor(zzz, "default", [r("2017-01-03")]).lines[0].flags.includes("inactive_no_event_date"));
});

test("full session: complete panel incl. delisted names, reuse split, clip, bad ticker skipped, pointer swapped", async () => {
  const root = tmpRoot();
  const c = clockDeps();
  const seen: Seen[] = [];
  const out = await runPanelSession({ env: ENV, fetch: api({ seen }), root, now: c.now, sleep: c.sleep, freeBytes: () => 50 * 2 ** 30, inWindow: () => true });
  assert.equal(out.status, "complete", out.detail);
  const cur = JSON.parse(fs.readFileSync(path.join(root, "current.json"), "utf8"));
  assert.equal(cur.build_id, "b20261028");
  assert.ok(!fs.existsSync(path.join(root, "building.json")));
  const m = JSON.parse(fs.readFileSync(path.join(root, cur.build_id, "manifest.json"), "utf8"));
  assert.equal(m.status, "complete");
  // T0 is 21:00 ET on 10-27: as_of is ET date minus one (10-26). The old expectation, 10-27, was the
  // UTC-yesterday value Alpaca refuses as "recent SIP" (the 2026-10-29 live 403).
  assert.equal(m.as_of, "2026-10-26");
  assert.equal(m.source, "alpaca");
  assert.deepEqual(m.universe, {
    total: 7, active: 3, inactive: 2, corp_action_only: 2, excluded_otc: 1, reuse_suspect: 0, with_event: 3,
    corp_actions_http: 200, corp_action_events: 3,
  });
  assert.equal(m.counts.skipped_symbols, 1, "BAD rejected alone -> skipped, batch not wedged");
  assert.equal(m.counts.stale_active, 1, "S (active) mapped to the old company -> refetched unmapped");
  assert.equal(m.counts.symbols_with_bars, 5);
  assert.equal(m.counts.delisted_with_bars, 3, "TWTR, MON, ZZZ");
  assert.equal(m.counts.earliest_bar, "2017-01-03");
  assert.ok(m.caveats.some((x: string) => x.includes("2010-2015")));
  const lines = readLines(root);
  const get = (s: string, mode: string) => lines.filter((l) => l.s === s && l.mode === mode);
  assert.equal(get("TWTR", "default")[0].seg_end, "2022-10-27", "clipped at the merger");
  assert.equal(m.counts.clipped_bars, DEFAULT.TWTR.filter((r) => r.d > "2022-10-27").length);
  assert.equal(get("MON", "default")[0].st, "corp_action");
  const sRaw = get("S", "raw");
  assert.equal(sRaw.length, 2, "unmapped S splits into Sprint and the new company");
  assert.equal(sRaw[1].seg_start, "2021-06-30");
  assert.ok(get("S", "default")[0].flags.includes("stale_active"));
  const aaa = decodeSegment(get("AAA", "default")[0]);
  assert.equal(aaa.length, DEFAULT.AAA.length, "pages merged per symbol");
  assert.equal(aaa[aaa.length - 1].d, "2026-10-27");
  const mon = decodeSegment(get("MON", "default")[0]);
  assert.ok(Math.abs((mon[0].rc ?? 0) - mon[0].c * 2) < 1e-6, "raw close recovered from the factor run");
  // Requests: GET only, keys in headers only, never an order path, paced.
  for (const s of seen) {
    assert.equal(s.method, "GET");
    assert.ok(!s.url.includes(ENV.ALPACA_KEY) && !s.url.includes(ENV.ALPACA_SECRET));
    assert.equal(s.headers["APCA-API-KEY-ID"], ENV.ALPACA_KEY);
    assert.ok(!s.url.includes("/orders"));
  }
  assert.ok(c.slept.every((ms) => ms <= 600), "pacing only, no back-off on a clean run");
  fs.rmSync(root, { recursive: true, force: true });
});

test("resumable: a window close checkpoints; the next session finishes with the same panel", async () => {
  const one = tmpRoot(), two = tmpRoot();
  const c1 = clockDeps();
  await runPanelSession({ env: ENV, fetch: api(), root: one, now: c1.now, sleep: c1.sleep, freeBytes: () => null, inWindow: () => true, batchSymbols: 2 });
  const c2 = clockDeps();
  let calls = 0;
  const paused = await runPanelSession({ env: ENV, fetch: api(), root: two, now: c2.now, sleep: c2.sleep, freeBytes: () => null, inWindow: () => ++calls <= 30, batchSymbols: 2 });
  assert.equal(paused.status, "paused_window");
  const building = JSON.parse(fs.readFileSync(path.join(two, "building.json"), "utf8"));
  const m = JSON.parse(fs.readFileSync(path.join(two, building.build_id, "manifest.json"), "utf8"));
  assert.equal(m.status, "building");
  assert.ok(m.progress.next_batch > 0 && m.progress.next_batch < m.progress.batches_total, `checkpoint ${m.progress.next_batch}/${m.progress.batches_total}`);
  const done = await runPanelSession({ env: ENV, fetch: api(), root: two, now: c2.now, sleep: c2.sleep, freeBytes: () => null, inWindow: () => true, batchSymbols: 2 });
  assert.equal(done.status, "complete", done.detail);
  const strip = (ls: PanelLine[]) => ls.map((l) => JSON.stringify(l)).sort();
  assert.deepEqual(strip(readLines(two)), strip(readLines(one)));
  const m2 = JSON.parse(fs.readFileSync(path.join(two, building.build_id, "manifest.json"), "utf8"));
  assert.equal(m2.progress.sessions, 2);
  fs.rmSync(one, { recursive: true, force: true }); fs.rmSync(two, { recursive: true, force: true });
});

test("guards: disk floor blocks before any request; 429 backs off 60 s; fresh build is not redone; disabled flag", async () => {
  const root = tmpRoot();
  const c = clockDeps();
  const seen: Seen[] = [];
  const blocked = await runPanelSession({ env: ENV, fetch: api({ seen }), root, now: c.now, sleep: c.sleep, freeBytes: () => 2 ** 30, inWindow: () => true });
  assert.equal(blocked.status, "blocked");
  assert.match(blocked.detail, /disk/);
  assert.equal(seen.length, 0);
  const ok = await runPanelSession({ env: ENV, fetch: api({ seen, first429: true }), root, now: c.now, sleep: c.sleep, freeBytes: () => null, inWindow: () => true });
  assert.equal(ok.status, "complete", ok.detail);
  assert.ok(c.slept.includes(BACKOFF_429_MS));
  const n = seen.length;
  const again = await runPanelSession({ env: ENV, fetch: api({ seen }), root, now: c.now, sleep: c.sleep, freeBytes: () => null, inWindow: () => true });
  assert.equal(again.status, "fresh");
  assert.equal(seen.length, n);
  const off = await runPanelSession({ env: { ...ENV, VOLTRADE_PRICE_PANEL: "off" }, fetch: api({ seen }), root, inWindow: () => true });
  assert.equal(off.status, "disabled");
  fs.rmSync(root, { recursive: true, force: true });
});

test("status probe: manifests only (no bars, no keys), sanitizer no-op, wired in diag + bot.ts", async () => {
  const root = tmpRoot();
  const c = clockDeps();
  await runPanelSession({ env: ENV, fetch: api(), root, now: c.now, sleep: c.sleep, freeBytes: () => null, inWindow: () => true });
  const st = pricePanelStatus(ENV, root);
  const cur = st.current as Record<string, unknown>;
  assert.equal(cur.status, "complete");
  assert.equal(st.building, null);
  const flat = JSON.stringify(st);
  for (const leak of [ENV.ALPACA_KEY, ENV.ALPACA_SECRET, "\"dd\"", "https://"]) assert.ok(!flat.includes(leak), `status leaked ${leak}`);
  assert.deepEqual(sanitizeDiag(st), st);
  assert.ok((DIAG_PROBES as readonly string[]).includes("price_panel"));
  const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
  const i = bot.indexOf('case "price_panel":');
  assert.ok(i > 0 && bot.slice(i, i + 400).includes("sanitizeDiag(pricePanelStatus())"));
  fs.rmSync(root, { recursive: true, force: true });
});

test("wiring: booted from routes.ts, never builds on boot, GET-only source, storage_config names the same dir", () => {
  const routes = fs.readFileSync(path.join(here, "routes.ts"), "utf8");
  assert.ok(routes.includes("bootPricePanel();"));
  const src = fs.readFileSync(path.join(here, "pricePanel.ts"), "utf8");
  const boot = src.slice(src.indexOf("export function bootPricePanel"));
  assert.ok(!/^\s*(void\s+)?tick\(\);/m.test(boot), "a boot must not run a tick immediately");
  assert.ok(boot.includes("setInterval(") && boot.includes(".unref"));
  assert.ok(!/method:\s*"(POST|PUT|PATCH|DELETE)"/.test(src));
  assert.ok(!src.includes("/orders"));
  const sc = fs.readFileSync(path.join(repo, "storage_config.py"), "utf8");
  assert.ok(sc.includes(`PRICE_PANEL_DIR = os.path.join(DATA_DIR, "${PRICE_PANEL_SUBDIR}")`));
});

test("panelAsOf: ET date minus one, never the UTC day before (403 recent-SIP regression, 2026-10-29)", async () => {
  const { panelAsOf } = await import("./pricePanel");
  // 00:59Z on 10-29 is 20:59 ET on 10-28 (the failing live run): as_of must be 10-27, not 10-28.
  assert.equal(panelAsOf(Date.parse("2026-10-29T00:59:25Z")), "2026-10-27");
  // 05:00Z on 10-29 is 01:00 ET on 10-29: as_of 10-28.
  assert.equal(panelAsOf(Date.parse("2026-10-29T05:00:00Z")), "2026-10-28");
  // 12:00Z mid-day: unchanged from the old rule.
  assert.equal(panelAsOf(Date.parse("2026-10-28T12:00:00Z")), "2026-10-27");
});
