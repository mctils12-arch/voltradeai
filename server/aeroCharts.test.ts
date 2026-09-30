// aeroCharts.test.ts — FAA chart tile read-through cache: URL/key building,
// edition keying, the 56-day cycle, and every cache path (tmp hit/miss,
// R2 hit/miss, upstream 404 -> empty, upstream error -> 502 not cached).

import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import os from "os";
import path from "path";

import {
  AERO_CHARTS, AERO_EMPTY_PNG, AERO_TMP_ENTRY_FLOOR_BYTES, aeroCacheControl, aeroCacheKey, aeroClientTileTemplate,
  aeroUpstreamTileUrl, addDays, createAeroChartService, createTmpLru, editionInfo, faaCycleStart,
  isAeroChartId, parseEditionFromSubject, prefetchTiles, sniffImageType, validTile, handleAeroTile,
} from "./aeroCharts";
import type { R2Client } from "./r2Client";

const JPEG = Buffer.from([0xff, 0xd8, 0xff, 0xe0, 1, 2, 3, 4]);
const PNG = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 9, 9]);
const NOW = Date.parse("2026-09-30T12:00:00Z");

function tmpDir(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), "aero-test-"));
}

interface FakeFetchLog { urls: string[] }
function fakeFetch(route: (url: string) => { status: number; body?: Buffer | string; json?: unknown }, log: FakeFetchLog) {
  return (async (input: string | URL | Request) => {
    const url = String(input);
    log.urls.push(url);
    const r = route(url);
    const body = r.json !== undefined ? JSON.stringify(r.json) : r.body ?? "";
    return new Response(typeof body === "string" ? body : new Uint8Array(body), { status: r.status });
  }) as typeof fetch;
}

const META = (subject: string, minLOD = 8, maxLOD = 12) => ({ status: 200, json: { minLOD, maxLOD, documentInfo: { subject } } });

function fakeR2(): R2Client & { store: Map<string, Buffer>; gets: number; puts: number } {
  const store = new Map<string, Buffer>();
  const r: R2Client & { store: Map<string, Buffer>; gets: number; puts: number } = {
    configured: true, store, gets: 0, puts: 0,
    async putObject(key, body) { r.puts++; store.set(key, Buffer.from(body)); return { ok: true }; },
    async getObject(key) {
      r.gets++;
      const b = store.get(key);
      return b ? { ok: true, status: 200, body: b } : { ok: false, status: 404 };
    },
    async getObjectToFile() { return { ok: false }; },
    async headObject(key) { return { ok: true, exists: store.has(key) }; },
    async listObjects() { return { ok: true, objects: [], prefixes: [], truncated: false, pages: 0 }; },
    async deleteObjects() { return { ok: true, requested: 0, errors: [] }; },
    async deleteObject() { return { ok: true }; },
  };
  return r;
}

const flush = () => new Promise<void>((r) => setImmediate(r));

// ── pure ─────────────────────────────────────────────────────────────────────

test("upstream URL uses ArcGIS {z}/{y}/{x} order; cache key is edition-scoped; client template pins the edition", () => {
  assert.equal(aeroUpstreamTileUrl("sectional", 9, 117, 210),
    "https://tiles.arcgis.com/tiles/ssFJjBXIUyZDrSYZ/arcgis/rest/services/VFR_Sectional/MapServer/tile/9/210/117");
  assert.equal(aeroUpstreamTileUrl("ifrlow", 8, 1, 2).includes("/IFR_AreaLow/"), true);
  assert.equal(aeroCacheKey("tac", "2026-07-09", 10, 234, 421), "aero/tac/2026-07-09/10/421/234");
  assert.notEqual(aeroCacheKey("tac", "2026-07-09", 10, 1, 1), aeroCacheKey("tac", "2026-09-03", 10, 1, 1));
  assert.equal(aeroClientTileTemplate("ifrhigh", "2026-05-14"), "/tiles/aero/ifrhigh/{z}/{x}/{y}?e=2026-05-14");
  assert.equal(aeroClientTileTemplate("ifrhigh", null), "/tiles/aero/ifrhigh/{z}/{x}/{y}");
  assert.ok(isAeroChartId("sectional") && !isAeroChartId("../etc"));
  assert.match(AERO_CHARTS.ifrlow.label, /Enroute \+ Area/, "IFR_AreaLow is named for what it actually carries");
});

test("tile coordinate validation", () => {
  assert.ok(validTile(0, 0, 0));
  assert.ok(validTile(9, 511, 511));
  assert.ok(!validTile(9, 512, 0));
  assert.ok(!validTile(3, -1, 0));
  assert.ok(!validTile(2.5, 0, 0));
  assert.ok(!validTile(23, 0, 0));
});

test("image sniffing: JPEG, PNG, and junk (an HTML error page is never cached as a tile)", () => {
  assert.equal(sniffImageType(JPEG), "image/jpeg");
  assert.equal(sniffImageType(PNG), "image/png");
  assert.equal(sniffImageType(AERO_EMPTY_PNG), "image/png");
  assert.equal(sniffImageType(Buffer.from("<html>")), null);
});

test("edition parsing from the service's documentInfo.subject", () => {
  assert.equal(parseEditionFromSubject("Updated with the latest charts on 07-09-2026."), "2026-07-09");
  assert.equal(parseEditionFromSubject("Updated with the latest charts on 05-14-2026"), "2026-05-14");
  assert.equal(parseEditionFromSubject("updated 5/14/2026"), "2026-05-14");
  assert.equal(parseEditionFromSubject("Updated 02-31-2026"), null);
  assert.equal(parseEditionFromSubject(""), null);
  assert.equal(parseEditionFromSubject(null), null);
});

test("FAA 56-day cycle: known editions land on cycle boundaries; expiry and 'behind current cycle' are honest", () => {
  // the service's own reported editions are cycle starts
  assert.equal(faaCycleStart(Date.parse("2026-05-14T00:00:00Z")), "2026-05-14");
  assert.equal(faaCycleStart(Date.parse("2026-07-09T08:00:00Z")), "2026-07-09");
  assert.equal(faaCycleStart(Date.parse("2026-07-08T23:59:00Z")), "2026-05-14");
  assert.equal(faaCycleStart(NOW), "2026-09-03");
  assert.equal(addDays("2026-07-09", 56), "2026-09-03");
  const e = editionInfo("2026-07-09", NOW);
  assert.deepEqual({ ...e }, {
    edition: "2026-07-09", source: "service-metadata", effective: "2026-07-09", expires: "2026-09-03",
    expired: true, currentFaaCycle: "2026-09-03", behindCurrentCycle: true,
  });
  const cur = editionInfo("2026-09-03", NOW);
  assert.equal(cur.expired, false);
  assert.equal(cur.behindCurrentCycle, false);
  const unk = editionInfo(null, NOW);
  assert.equal(unk.source, "unverified");
  assert.equal(unk.effective, null, "no effective date is claimed without service metadata");
  assert.match(unk.edition, /^unverified-2026-09-03$/);
});

test("cache-control: immutable only when the request pins the current edition", () => {
  assert.match(aeroCacheControl("2026-07-09", "2026-07-09"), /immutable/);
  assert.doesNotMatch(aeroCacheControl("2026-05-14", "2026-07-09"), /immutable/);
  assert.doesNotMatch(aeroCacheControl(undefined, "2026-07-09"), /immutable/);
});

test("prefetch enumeration: CONUS, coarse levels first, capped at z9, empty when minzoom > 9", () => {
  const t = prefetchTiles({ minzoom: 8, maxzoom: 12 });
  assert.equal(t[0].z, 8);
  assert.equal(Math.max(...t.map((q) => q.z)), 9);
  for (let i = 1; i < t.length; i++) assert.ok(t[i].z >= t[i - 1].z);
  // Kansas (-98, 38.5) z8 = 58/98 must be inside the box
  assert.ok(t.some((q) => q.z === 8 && q.x === 58 && q.y === 98));
  assert.equal(prefetchTiles({ minzoom: 10, maxzoom: 12 }).length, 0, "TAC starts at z10 — nothing to pre-bake");
  // bounded: whole CONUS sectional z8-9 is a few thousand tiles, not millions
  assert.ok(t.length > 1000 && t.length < 8000, String(t.length));
});

test("tmp LRU: byte cap evicts least-recently-used; empty markers still count", () => {
  const dir = tmpDir();
  const lru = createTmpLru(dir, 3 * AERO_TMP_ENTRY_FLOOR_BYTES);
  lru.put("a", Buffer.alloc(0));
  lru.put("b", Buffer.alloc(10));
  lru.put("c", Buffer.alloc(10));
  assert.ok(lru.get("a"), "touch a -> most recent");
  lru.put("d", Buffer.alloc(10));
  assert.equal(lru.get("b"), null, "b was least recently used");
  assert.ok(lru.get("a") && lru.get("c") && lru.get("d"));
  assert.equal(lru.stats().entries, 3);
  assert.equal(lru.stats().evictions, 1);
  assert.ok(lru.stats().bytes <= lru.stats().maxBytes);
});

// ── service: cache paths with a fake fetch ────────────────────────────────────

test("no-R2 path: miss -> upstream -> tmp; second request is a tmp hit with zero upstream calls", async () => {
  const log: FakeFetchLog = { urls: [] };
  const svc = createAeroChartService({
    now: () => NOW, env: {}, tmpDir: tmpDir(),
    r2: { ...fakeR2(), configured: false },
    fetchImpl: fakeFetch((u) => (u.includes("?f=json") ? META("Updated with the latest charts on 07-09-2026.") : { status: 200, body: JPEG }), log),
  });
  const a = await svc.getTile("sectional", 9, 117, 210);
  assert.equal(a.outcome.kind, "tile");
  assert.equal(a.outcome.kind === "tile" && a.outcome.from, "upstream");
  assert.equal(a.edition.edition, "2026-07-09");
  const tileCalls = () => log.urls.filter((u) => u.includes("/tile/")).length;
  assert.equal(tileCalls(), 1);
  const b = await svc.getTile("sectional", 9, 117, 210);
  assert.equal(b.outcome.kind === "tile" && b.outcome.from, "tmp");
  assert.equal(tileCalls(), 1, "cache hit never touches upstream");
  const st = await svc.status();
  assert.equal(st.cache.mode, "tmp-lru");
  assert.equal(st.cache.tmpHits, 1);
  assert.equal(st.charts.find((c) => c.id === "sectional")!.misses, 1);
  assert.equal(st.r2Config.configured, false);
  assert.equal(st.notForNavigation, true);
});

test("upstream 404 = empty (transparent), cached as a 0-byte marker; upstream 500 = error, NOT cached", async () => {
  const log: FakeFetchLog = { urls: [] };
  let tileStatus = 404;
  const svc = createAeroChartService({
    now: () => NOW, env: {}, tmpDir: tmpDir(), r2: { ...fakeR2(), configured: false },
    fetchImpl: fakeFetch((u) => (u.includes("?f=json") ? META("on 07-09-2026") : { status: tileStatus, body: "<html>not found</html>" }), log),
  });
  const a = await svc.getTile("sectional", 9, 1, 1);
  assert.deepEqual(a.outcome, { kind: "empty", from: "upstream" });
  const b = await svc.getTile("sectional", 9, 1, 1);
  assert.deepEqual(b.outcome, { kind: "empty", from: "tmp" }, "the 404 is remembered for the edition");
  tileStatus = 500;
  const c = await svc.getTile("sectional", 9, 2, 2);
  assert.equal(c.outcome.kind, "error");
  const before = log.urls.length;
  const d = await svc.getTile("sectional", 9, 2, 2);
  assert.equal(d.outcome.kind, "error");
  assert.equal(log.urls.length, before + 1, "an error is retried upstream next time, never cached");
  const st = await svc.status();
  assert.equal(st.cache.upstream404, 1);
  assert.equal(st.cache.upstreamErrors, 2);
});

test("an HTML 200 from upstream is an error, never cached as a tile", async () => {
  const log: FakeFetchLog = { urls: [] };
  const svc = createAeroChartService({
    now: () => NOW, env: {}, tmpDir: tmpDir(), r2: { ...fakeR2(), configured: false },
    fetchImpl: fakeFetch((u) => (u.includes("?f=json") ? META("on 07-09-2026") : { status: 200, body: "<html>" }), log),
  });
  assert.equal((await svc.getTile("sectional", 9, 3, 3)).outcome.kind, "error");
});

test("out-of-LOD requests are empty without an upstream call (LOD band read from metadata)", async () => {
  const log: FakeFetchLog = { urls: [] };
  const svc = createAeroChartService({
    now: () => NOW, env: {}, tmpDir: tmpDir(), r2: { ...fakeR2(), configured: false },
    fetchImpl: fakeFetch((u) => (u.includes("?f=json") ? META("on 05-14-2026", 5, 9) : { status: 200, body: JPEG }), log),
  });
  const r = await svc.getTile("ifrhigh", 11, 400, 800);
  assert.deepEqual(r.outcome, { kind: "empty", from: "out-of-range" });
  assert.equal(log.urls.filter((u) => u.includes("/tile/")).length, 0);
});

test("R2 path: miss -> upstream -> PUT under the edition key; next request is an R2 hit", async () => {
  const log: FakeFetchLog = { urls: [] };
  const r2 = fakeR2();
  const svc = createAeroChartService({
    now: () => NOW, env: {}, r2,
    fetchImpl: fakeFetch((u) => (u.includes("?f=json") ? META("on 07-09-2026") : { status: 200, body: PNG }), log),
  });
  const a = await svc.getTile("tac", 10, 234, 421);
  assert.equal(a.outcome.kind === "tile" && a.outcome.contentType, "image/png");
  await flush();
  assert.ok(r2.store.has("aero/tac/2026-07-09/10/421/234"));
  const b = await svc.getTile("tac", 10, 234, 421);
  assert.equal(b.outcome.kind === "tile" && b.outcome.from, "r2");
  assert.equal(log.urls.filter((u) => u.includes("/tile/")).length, 1);
  assert.equal((await svc.status()).cache.mode, "r2");
});

test("edition keying: a new service edition reads a fresh key space (old tiles never served for the new cycle)", async () => {
  const log: FakeFetchLog = { urls: [] };
  const r2 = fakeR2();
  let t = NOW;
  let subject = "on 07-09-2026";
  const svc = createAeroChartService({
    now: () => t, env: {}, r2,
    fetchImpl: fakeFetch((u) => (u.includes("?f=json") ? META(subject) : { status: 200, body: JPEG }), log),
  });
  await svc.getTile("sectional", 8, 58, 98);
  await flush();
  subject = "on 09-03-2026";
  t = NOW + 7 * 3600_000; // metadata TTL elapsed
  await svc.getTile("sectional", 8, 58, 98); // triggers the background refresh
  await flush(); await flush();
  const r = await svc.getTile("sectional", 8, 58, 98);
  assert.equal(r.edition.edition, "2026-09-03");
  await flush();
  assert.ok(r2.store.has("aero/sectional/2026-07-09/8/98/58"));
  assert.ok(r2.store.has("aero/sectional/2026-09-03/8/98/58"));
});

test("concurrent requests for one tile share ONE upstream fetch", async () => {
  const log: FakeFetchLog = { urls: [] };
  const svc = createAeroChartService({
    now: () => NOW, env: {}, tmpDir: tmpDir(), r2: { ...fakeR2(), configured: false },
    fetchImpl: fakeFetch((u) => (u.includes("?f=json") ? META("on 07-09-2026") : { status: 200, body: JPEG }), log),
  });
  await svc.edition("sectional");
  await Promise.all([1, 2, 3, 4].map(() => svc.getTile("sectional", 9, 5, 5)));
  assert.equal(log.urls.filter((u) => u.includes("/tile/")).length, 1);
});

test("metadata unreachable: edition is 'unverified', never a claimed effective date; retried after minutes", async () => {
  const log: FakeFetchLog = { urls: [] };
  let t = NOW;
  let up = false;
  const svc = createAeroChartService({
    now: () => t, env: {}, tmpDir: tmpDir(), r2: { ...fakeR2(), configured: false },
    fetchImpl: fakeFetch((u) => (u.includes("?f=json") ? (up ? META("on 07-09-2026") : { status: 503 }) : { status: 200, body: JPEG }), log),
  });
  const e = await svc.edition("ifrlow");
  assert.equal(e.source, "unverified");
  assert.equal(e.effective, null);
  up = true;
  t += 6 * 60_000;
  await svc.edition("ifrlow");
  await flush(); await flush();
  assert.equal((await svc.edition("ifrlow")).edition, "2026-07-09");
});

test("prefetch: R2 only; bakes CONUS tiles for the edition, writes a done-marker, skips on the next run", async () => {
  const log: FakeFetchLog = { urls: [] };
  const r2 = fakeR2();
  const svc = createAeroChartService({
    now: () => NOW, env: {}, r2, sleep: async () => {},
    fetchImpl: fakeFetch((u) => (u.includes("?f=json")
      ? (u.includes("IFR_High") ? META("on 05-14-2026", 5, 5) : META("on 07-09-2026", 13, 14))
      : { status: 200, body: JPEG }), log),
  });
  await svc.runPrefetch();
  await flush();
  const expected = prefetchTiles({ minzoom: 5, maxzoom: 5 }).length;
  assert.equal(log.urls.filter((u) => u.includes("/tile/")).length, expected);
  assert.ok(r2.store.has("aero/ifrhigh/2026-05-14/_prefetch_done"));
  const before = log.urls.filter((u) => u.includes("/tile/")).length;
  await svc.runPrefetch();
  assert.equal(log.urls.filter((u) => u.includes("/tile/")).length, before, "done-marker short-circuits");

  const noR2 = createAeroChartService({ now: () => NOW, env: {}, tmpDir: tmpDir(), r2: { ...fakeR2(), configured: false },
    fetchImpl: fakeFetch(() => ({ status: 200, body: JPEG }), { urls: [] }) });
  await noR2.runPrefetch();
  assert.match((await noR2.status()).prefetch.note, /R2 not configured/);
});

test("HTTP handler: empty -> 200 transparent PNG; error -> 502 no-store; tile -> bytes + edition header", async () => {
  let tileStatus = 200;
  const svc = createAeroChartService({
    now: () => NOW, env: {}, tmpDir: tmpDir(), r2: { ...fakeR2(), configured: false },
    fetchImpl: fakeFetch((u) => (u.includes("?f=json") ? META("on 07-09-2026") : { status: tileStatus, body: JPEG }), { urls: [] }),
  });
  function call(params: Record<string, string>, query: Record<string, string> = {}) {
    const res = {
      code: 0, headers: {} as Record<string, string>, body: undefined as unknown, headersSent: false,
      setHeader(k: string, v: string) { this.headers[k.toLowerCase()] = v; },
      status(c: number) { this.code = c; return this; },
      json(b: unknown) { this.body = b; return this; },
      end(b?: unknown) { this.body = b; return this; },
    };
    const req = { params, query };
    return handleAeroTile(svc, req as never, res as never).then(() => res);
  }
  const ok = await call({ chart: "sectional", z: "9", x: "10", y: "10" }, { e: "2026-07-09" });
  assert.equal(ok.code, 200);
  assert.equal(ok.headers["content-type"], "image/jpeg");
  assert.equal(ok.headers["x-aero-edition"], "2026-07-09");
  assert.match(ok.headers["cache-control"], /immutable/);
  const oor = await call({ chart: "sectional", z: "3", x: "1", y: "1" });
  assert.equal(oor.code, 200);
  assert.equal(oor.headers["content-type"], "image/png");
  assert.deepEqual(oor.body, AERO_EMPTY_PNG);
  tileStatus = 503;
  const bad = await call({ chart: "sectional", z: "9", x: "11", y: "11" });
  assert.equal(bad.code, 502);
  assert.equal(bad.headers["cache-control"], "no-store");
  assert.equal((await call({ chart: "nope", z: "9", x: "1", y: "1" })).code, 404);
  assert.equal((await call({ chart: "sectional", z: "9", x: "9999", y: "1" })).code, 400);
});
