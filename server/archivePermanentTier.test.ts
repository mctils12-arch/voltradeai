import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import os from "os";
import path from "path";
import zlib from "zlib";
import crypto from "crypto";
import {
  createPermanentTier, permanentTierFlag, permanentKey, gzipLineCount, loadPermanentManifest, listPermanentLocalFiles,
  PERMANENT_STREAMS, PERMANENT_LOCAL_DAYS, PERMANENT_MANIFEST_NAME, permanentLocalDays,
} from "./archivePermanentTier";
import { createArchiveOffloadService } from "./archiveOffload";
import { createR2Client, r2ConfigFromEnv } from "./r2Client";
import { readArchiveDay, setColdDayFileSource, RAW_RETENTION_DAYS } from "./datacoreArchive";
import { DAYS_CAP, snapshotWindow } from "./queryEngine";
import { WINDOW_DAYS as SITE_TIMELINE_DAYS } from "./siteTimeline";
import {
  REQUIRED_COLUMNS, parseHeader, parseDtccLine, archiveNewRows, archiveNewRowsAsync, dtccOwedDayFiles,
  DTCC_ARCHIVE_BACKFILL_LOOKBACK_DAYS, _resetDtccForTests, type DtccSwapRow,
} from "./dtccSwaps";

const NOW = Date.parse("2026-11-06T15:00:00Z");
const DAY = 86_400_000;
const ENV = {
  R2_ACCOUNT_ID: "0123456789abcdef0123456789abcdef",
  R2_ACCESS_KEY_ID: "AKID", R2_SECRET_ACCESS_KEY: "TOPSECRETKEY", R2_ARCHIVE_BUCKET: "voltrade-archive-private",
};
const CFG = r2ConfigFromEnv(ENV as NodeJS.ProcessEnv)!;
const sha = (b: Buffer) => crypto.createHash("sha256").update(b).digest("hex");

function tmp(prefix = "vt-perm-"): string { return fs.mkdtempSync(path.join(os.tmpdir(), prefix)); }
function dateAgo(n: number): string { return new Date(Date.parse("2026-11-06T00:00:00Z") - n * DAY).toISOString().slice(0, 10); }

function rollupRows(date: string, ids = ["244010352", "636019825"]): string[] {
  return ids.map((i, k) => JSON.stringify({ i, d: date, n: 10 + k, t0: 1, t1: 2, bbox: [40, -100, 41, -99], pl: [[40, -100], [41, -99]] }));
}

/** a `<stream>/<date>.jsonl.gz` rollup in the archive's own shape, idle 2 h by default */
function writeDay(base: string, stream: string, date: string, lines = rollupRows(date), mtimeMs = NOW - 2 * 3600_000): string {
  const dir = path.join(base, stream);
  fs.mkdirSync(dir, { recursive: true });
  const fp = path.join(dir, `${date}.jsonl.gz`);
  fs.writeFileSync(fp, zlib.gzipSync(lines.join("\n") + "\n"));
  fs.utimesSync(fp, mtimeMs / 1000, mtimeMs / 1000);
  return fp;
}

/** In-memory R2 behind a fake fetch — the REAL r2Client talks to it. */
function fakeR2() {
  const store = new Map<string, Buffer>();
  const log: Array<{ method: string; key: string; payloadSha?: string }> = [];
  const ctl = {
    down: false, failPutKeys: new Set<string>(), truncatePut: new Set<string>(),
    corruptPut: new Set<string>(), failGetKeys: new Set<string>(),
  };
  const fetchImpl = (async (url: string | URL | Request, init: RequestInit = {}) => {
    const u = new URL(String(url));
    const key = decodeURIComponent(u.pathname.split("/").slice(2).join("/"));
    const method = init.method || "GET";
    const headers = (init.headers || {}) as Record<string, string>;
    log.push({ method, key, payloadSha: headers["x-amz-content-sha256"] });
    if (ctl.down) throw new Error("ECONNREFUSED");
    if (method === "PUT") {
      if (ctl.failPutKeys.has(key)) return new Response("<Error><Code>InternalError</Code></Error>", { status: 500 });
      const b = Buffer.from(init.body as Uint8Array);
      // R2 checks the signed payload hash against the body it received
      if (headers["x-amz-content-sha256"] !== sha(b)) return new Response("<Error><Code>XAmzContentSHA256Mismatch</Code></Error>", { status: 400 });
      let stored = b;
      if (ctl.truncatePut.has(key)) stored = b.subarray(0, 3);
      if (ctl.corruptPut.has(key)) { stored = Buffer.from(b); stored[stored.length - 1] ^= 0xff; } // same length, other bytes
      store.set(key, stored);
      return new Response(null, { status: 200, headers: { etag: '"e"' } });
    }
    if (method === "HEAD") {
      const b = store.get(key);
      if (!b) return new Response(null, { status: 404 });
      return new Response(null, { status: 200, headers: { "content-length": String(b.length) } });
    }
    if (method === "GET" && key) {
      if (ctl.failGetKeys.has(key)) return new Response("<Error><Code>InternalError</Code></Error>", { status: 500 });
      const b = store.get(key);
      return b ? new Response(b, { status: 200 }) : new Response("<Error><Code>NoSuchKey</Code></Error>", { status: 404 });
    }
    if (method === "GET") {
      const prefix = u.searchParams.get("prefix") || "";
      const keys = Array.from(store.keys()).filter((k) => k.startsWith(prefix)).sort();
      const xml = `<ListBucketResult><IsTruncated>false</IsTruncated>${keys.map((k) =>
        `<Contents><Key>${k}</Key><Size>${store.get(k)!.length}</Size></Contents>`).join("")}</ListBucketResult>`;
      return new Response(xml, { status: 200 });
    }
    if (method === "POST" && u.search === "?delete") {
      const body = Buffer.from(init.body as Uint8Array).toString();
      for (const m of Array.from(body.matchAll(/<Key>([^<]*)<\/Key>/g))) store.delete(m[1]);
      return new Response("<DeleteResult></DeleteResult>", { status: 200 });
    }
    return new Response(null, { status: 400 });
  }) as typeof fetch;
  const client = createR2Client(CFG, { fetchImpl, sleep: async () => {}, maxRetries: 0 });
  return { store, log, ctl, client };
}

function tierFor(base: string, r2: ReturnType<typeof fakeR2> | null, o: { now?: () => number; env?: Record<string, string>; errors?: string[]; logs?: string[] } = {}) {
  return createPermanentTier({
    base,
    client: r2 ? r2.client : createR2Client(null),
    env: (o.env || {}) as NodeJS.ProcessEnv,
    nowMs: o.now ?? (() => NOW),
    cacheDir: path.join(base, "..", path.basename(base) + "-pcache"),
    log: (m) => { o.logs?.push(m); },
    logError: (m) => { o.errors?.push(m); },
  });
}

const key = (stream: string, date: string) => permanentKey(stream, `${date}.jsonl.gz`);

// ── pure pieces ──────────────────────────────────────────────────────────────
test("flag: default ON; a present env value of 0/off/false/no/empty turns it off", () => {
  assert.equal(permanentTierFlag({} as NodeJS.ProcessEnv).on, true);
  assert.equal(permanentTierFlag({} as NodeJS.ProcessEnv).source, "default");
  for (const v of ["0", "off", "false", "no", "", " OFF "]) assert.equal(permanentTierFlag({ ARCHIVE_PERMANENT_TIER: v } as NodeJS.ProcessEnv).on, false, v);
  assert.equal(permanentTierFlag({ ARCHIVE_PERMANENT_TIER: "1" } as NodeJS.ProcessEnv).on, true);
});

test("keys and scope: permanent/<stream>/<file>; the three _tracks streams + occvolume, only <date>.jsonl.gz", () => {
  // pin updated 2026-11-06 (volume-space-4): occvolume joined with a 30-day local window;
  // (volume-space-5): fires + buoys with 31 days; 2026-11-25 ((aa4)): dtccswaps with 14 days
  assert.deepEqual([...PERMANENT_STREAMS], ["vessels_tracks", "aircraft_tracks", "trains_tracks", "occvolume", "fires", "buoys", "dtccswaps"]);
  assert.equal(permanentLocalDays("dtccswaps"), 14);
  assert.equal(permanentLocalDays("vessels_tracks"), PERMANENT_LOCAL_DAYS);
  assert.equal(permanentLocalDays("occvolume"), 30);
  assert.equal(permanentKey("vessels_tracks", "2026-07-03.jsonl.gz"), "permanent/vessels_tracks/2026-07-03.jsonl.gz");
  const base = tmp();
  writeDay(base, "vessels_tracks", dateAgo(9));
  fs.writeFileSync(path.join(base, "vessels_tracks", "notes.txt"), "x");
  fs.writeFileSync(path.join(base, "vessels_tracks", `${dateAgo(9)}.jsonl`), "{}\n");  // plain: never this tier's
  writeDay(base, "vessels", dateAgo(9));                                               // raw stream: not in scope
  assert.deepEqual(listPermanentLocalFiles(base).map((f) => `${f.stream}/${f.file}`), [`vessels_tracks/${dateAgo(9)}.jsonl.gz`]);
});

test("gzipLineCount: counts records with or without a trailing newline, skips blank lines, rejects a broken gzip", async () => {
  assert.deepEqual(await gzipLineCount(zlib.gzipSync("a\nb\n")), { ok: true, lines: 2, error: null });
  assert.deepEqual(await gzipLineCount(zlib.gzipSync("a\n\nb")), { ok: true, lines: 2, error: null });
  assert.deepEqual(await gzipLineCount(zlib.gzipSync("")), { ok: true, lines: 0, error: null });
  const big = Array.from({ length: 50_000 }, (_, i) => `{"i":${i}}`).join("\n") + "\n";
  assert.equal((await gzipLineCount(zlib.gzipSync(big))).lines, 50_000);
  const gz = zlib.gzipSync("a\nb\n");
  const truncated = await gzipLineCount(gz.subarray(0, gz.length - 6));
  assert.equal(truncated.ok, false, "a truncated gzip is not readable end to end");
  assert.equal((await gzipLineCount(Buffer.from("not gzip"))).ok, false);
});

// ── offload ──────────────────────────────────────────────────────────────────
test("offload: closed day files 2+ days old and idle 1 h go up byte-for-byte, SHA-256 signed, HEAD-verified, recorded; local untouched; idempotent", async () => {
  const base = tmp();
  const r2 = fakeR2();
  const fp = writeDay(base, "vessels_tracks", dateAgo(3));
  writeDay(base, "aircraft_tracks", dateAgo(1));                                  // too recent
  writeDay(base, "trains_tracks", dateAgo(5), rollupRows(dateAgo(5)), NOW - 10 * 60_000); // written 10 min ago
  const tier = tierFor(base, r2);
  const s = await tier.tick();
  assert.equal(s.uploaded, 1);
  assert.deepEqual(Array.from(r2.store.keys()), [key("vessels_tracks", dateAgo(3))]);
  const local = fs.readFileSync(fp);
  assert.deepEqual(r2.store.get(key("vessels_tracks", dateAgo(3))), local, "byte-for-byte");
  const put = r2.log.find((x) => x.method === "PUT")!;
  assert.equal(put.payloadSha, sha(local), "the signed payload hash is the local file's SHA-256");
  const methods = r2.log.filter((x) => x.key === key("vessels_tracks", dateAgo(3))).map((x) => x.method);
  assert.deepEqual(methods, ["HEAD", "PUT", "HEAD"], "never-overwrite check, upload, size verify");
  const e = loadPermanentManifest(base).entries[key("vessels_tracks", dateAgo(3))];
  assert.equal(e.sha256, sha(local));
  assert.equal(e.bytes, local.length);
  assert.equal(e.lines, 2);
  assert.equal(e.src, "uploaded");
  assert.equal(e.evictedAt, null);
  assert.ok(fs.existsSync(fp), "3 days old: inside the local window, kept");
  const puts = r2.log.filter((x) => x.method === "PUT").length;
  const s2 = await tier.tick();
  assert.equal(s2.uploaded, 0);
  assert.equal(r2.log.filter((x) => x.method === "PUT").length, puts, "idempotent");
});

// ── eviction ─────────────────────────────────────────────────────────────────
test("evict: the local copy goes only when its date is more than 7 days old AND the R2 object reads back byte-for-byte", async () => {
  const base = tmp();
  const r2 = fakeR2();
  const keep = writeDay(base, "vessels_tracks", dateAgo(PERMANENT_LOCAL_DAYS));
  const go = writeDay(base, "aircraft_tracks", dateAgo(PERMANENT_LOCAL_DAYS + 1));
  const goBytes = fs.readFileSync(go);
  const logs: string[] = [];
  const tier = tierFor(base, r2, { logs });
  const s = await tier.tick();
  assert.equal(s.uploaded, 2);
  assert.equal(s.evicted, 1);
  assert.equal(s.bytesEvicted, goBytes.length);
  assert.ok(fs.existsSync(keep), "day 7: kept");
  assert.ok(!fs.existsSync(go), "day 8: evicted");
  assert.deepEqual(r2.store.get(key("aircraft_tracks", dateAgo(8))), goBytes, "R2 holds the exact bytes");
  const k = key("aircraft_tracks", dateAgo(8));
  assert.deepEqual(r2.log.filter((x) => x.key === k).map((x) => x.method), ["HEAD", "PUT", "HEAD", "GET"], "the read-back GET precedes the delete");
  const e = loadPermanentManifest(base).entries[k];
  assert.equal(e.evictedAt, NOW);
  assert.match(logs.join("\n"), /evicted local aircraft_tracks\/.* sha256 match/);
  assert.deepEqual(tier.coldSource.heldOnlyInR2(), [
    { stream: "aircraft_tracks", file: `${dateAgo(8)}.jsonl.gz`, bytes: goBytes.length, lines: 2 },
  ], "only the evicted file is held only in R2");
});

test("byte-safety: a failed or unverifiable upload records nothing and deletes nothing; the next good tick finishes the job", async () => {
  const base = tmp();
  const r2 = fakeR2();
  const a = writeDay(base, "vessels_tracks", dateAgo(30));
  const b = writeDay(base, "vessels_tracks", dateAgo(29));
  r2.ctl.failPutKeys.add(key("vessels_tracks", dateAgo(30)));
  r2.ctl.truncatePut.add(key("vessels_tracks", dateAgo(29)));
  const errors: string[] = [];
  const tier = tierFor(base, r2, { errors });
  const s = await tier.tick();
  assert.equal(s.uploaded, 0);
  assert.equal(s.uploadFailed, 2);
  assert.equal(s.evicted, 0);
  assert.ok(fs.existsSync(a) && fs.existsSync(b), "nothing deleted");
  assert.deepEqual(Object.keys(loadPermanentManifest(base).entries), [], "nothing recorded");
  assert.match(errors.join("\n"), /verify permanent\/vessels_tracks\/.*R2 size 3 != uploaded/);
  // the truncated object is still in R2 (3 bytes) — the next tick must NOT
  // adopt it (bytes differ) and must not overwrite it either: a conflict
  r2.ctl.failPutKeys.clear();
  r2.ctl.truncatePut.clear();
  const s2 = await tier.tick();
  assert.equal(s2.uploaded, 1, "the failed PUT retries and lands");
  assert.equal(s2.evicted, 1);
  assert.ok(!fs.existsSync(a));
  assert.ok(fs.existsSync(b), "the key holding different bytes is never overwritten nor evicted");
  assert.equal(r2.store.get(key("vessels_tracks", dateAgo(29)))!.length, 3);
  const st = tier.status();
  assert.deepEqual(st.conflicts.map((c) => c.key), [key("vessels_tracks", dateAgo(29))]);
});

test("byte-safety: an R2 copy that reads back different (same size) or cannot be read deletes nothing", async () => {
  const base = tmp();
  const r2 = fakeR2();
  const bad = writeDay(base, "vessels_tracks", dateAgo(20));
  const flaky = writeDay(base, "vessels_tracks", dateAgo(19));
  r2.ctl.corruptPut.add(key("vessels_tracks", dateAgo(20)));      // stored bytes differ, length equal: HEAD passes
  r2.ctl.failGetKeys.add(key("vessels_tracks", dateAgo(19)));     // read-back GET fails
  const tier = tierFor(base, r2);
  const s = await tier.tick();
  assert.equal(s.uploaded, 2, "both passed the size check");
  assert.equal(s.evicted, 0);
  assert.ok(fs.existsSync(bad) && fs.existsSync(flaky), "nothing deleted");
  assert.deepEqual(tier.status().conflicts.map((c) => c.key), [key("vessels_tracks", dateAgo(20))], "a mismatch is a conflict");
  assert.equal(s.evictFailed, 1, "a failed read is retried, not a conflict");
  r2.ctl.failGetKeys.clear();
  const s2 = await tier.tick();
  assert.equal(s2.evicted, 1);
  assert.ok(!fs.existsSync(flaky));
  assert.ok(fs.existsSync(bad), "the corrupted copy never licenses a delete");
});

test("byte-safety: R2 down — nothing uploaded, nothing deleted, the tick stops after 3 failures", async () => {
  const base = tmp();
  const r2 = fakeR2();
  const fps = [30, 29, 28, 27, 26].map((n) => writeDay(base, "trains_tracks", dateAgo(n)));
  r2.ctl.down = true;
  const s = await tierFor(base, r2).tick();
  assert.equal(s.uploaded, 0);
  assert.equal(s.evicted, 0);
  assert.equal(s.uploadFailed, 3);
  assert.ok(fps.every((f) => fs.existsSync(f)));
  assert.match(s.errors.join("\n"), /offload paused this tick after 3 consecutive failures/);
});

// ── never overwrite ──────────────────────────────────────────────────────────
test("never overwrite: an existing object with other bytes is a conflict (no PUT); identical bytes are adopted (no PUT) and then evicted", async () => {
  const base = tmp();
  const r2 = fakeR2();
  const tier = tierFor(base, r2);
  await tier.tick(); // the daily reconcile runs on an empty bucket, so the next tick takes the HEAD path
  // objects appear AFTER the listing (e.g. another instance during a deploy overlap)
  const same = writeDay(base, "vessels_tracks", dateAgo(15));
  const diff = writeDay(base, "vessels_tracks", dateAgo(14));
  r2.store.set(key("vessels_tracks", dateAgo(15)), fs.readFileSync(same));
  const foreign = zlib.gzipSync("someone else's bytes\n");
  r2.store.set(key("vessels_tracks", dateAgo(14)), foreign);
  const s = await tier.tick();
  assert.equal(s.reconciled, null, "no listing this tick");
  assert.equal(r2.log.filter((x) => x.method === "PUT").length, 0, "no PUT at all");
  assert.equal(s.adopted, 1);
  assert.equal(s.evicted, 1);
  assert.ok(!fs.existsSync(same));
  assert.ok(fs.existsSync(diff));
  assert.deepEqual(r2.store.get(key("vessels_tracks", dateAgo(14))), foreign, "the foreign object is untouched");
  assert.equal(tier.status().conflicts.length, 1);
  // a conflict is sticky: later ticks neither upload nor evict it
  await tier.tick();
  assert.ok(fs.existsSync(diff));
  assert.equal(r2.log.filter((x) => x.method === "PUT").length, 0);
});

test("never overwrite (listing path): the daily listing adopts unknown objects unverified; a read-back then verifies the identical one and flags the other", async () => {
  const base = tmp();
  const r2 = fakeR2();
  const same = writeDay(base, "aircraft_tracks", dateAgo(15));
  const diff = writeDay(base, "aircraft_tracks", dateAgo(14));
  r2.store.set(key("aircraft_tracks", dateAgo(15)), fs.readFileSync(same));
  r2.store.set(key("aircraft_tracks", dateAgo(14)), zlib.gzipSync("other\n"));
  const tier = tierFor(base, r2);
  const s = await tier.tick();
  assert.deepEqual(s.reconciled, { adopted: 2, missing: 0, dropped: 0 });
  assert.equal(r2.log.filter((x) => x.method === "PUT").length, 0);
  assert.equal(s.evicted, 1, "the identical one, after its read-back");
  assert.ok(!fs.existsSync(same));
  assert.ok(fs.existsSync(diff));
  assert.match(tier.status().conflicts[0].reason, /adopted R2 object differs/);
});

test("a local file that changes after upload, or re-appears different after eviction, never replaces the permanent copy", async () => {
  const base = tmp();
  const r2 = fakeR2();
  let now = NOW;
  const tier = tierFor(base, r2, { now: () => now });
  // (a) uploaded at age 3, then rewritten (e.g. a partial re-rollup)
  const fp = writeDay(base, "vessels_tracks", dateAgo(3));
  const original = fs.readFileSync(fp);
  await tier.tick();
  writeDay(base, "vessels_tracks", dateAgo(3), rollupRows(dateAgo(3), ["only-one"]));
  await tier.tick();
  assert.deepEqual(r2.store.get(key("vessels_tracks", dateAgo(3))), original, "R2 keeps the original");
  now = NOW + 10 * DAY; // long past the local window
  await tier.tick();
  assert.ok(fs.existsSync(fp), "the changed file is never evicted");
  assert.match(tier.status().conflicts.map((c) => c.reason).join(";"), /changed after it was offloaded/);
  // (b) evicted, then the same day re-appears: identical bytes -> evicted again; different -> conflict
  now = NOW;
  const base2 = tmp();
  const r2b = fakeR2();
  const t2 = tierFor(base2, r2b, { now: () => now });
  const fp2 = writeDay(base2, "aircraft_tracks", dateAgo(9));
  const bytes2 = fs.readFileSync(fp2);
  await t2.tick();
  assert.ok(!fs.existsSync(fp2));
  fs.writeFileSync(fp2, bytes2);
  fs.utimesSync(fp2, (NOW - 2 * 3600_000) / 1000, (NOW - 2 * 3600_000) / 1000);
  const s = await t2.tick();
  assert.equal(s.evicted, 1, "identical bytes: evicted again without a PUT");
  assert.equal(r2b.log.filter((x) => x.method === "PUT").length, 1);
  writeDay(base2, "aircraft_tracks", dateAgo(9), rollupRows(dateAgo(9), ["other"]));
  await t2.tick();
  assert.ok(fs.existsSync(fp2), "different bytes stay local");
  assert.deepEqual(r2b.store.get(key("aircraft_tracks", dateAgo(9))), bytes2, "R2 untouched");
  assert.match(t2.status().conflicts.map((c) => c.reason).join(";"), /re-appeared after eviction/);
});

test("a local file whose gzip is unreadable is never offloaded or evicted", async () => {
  const base = tmp();
  const r2 = fakeR2();
  const fp = writeDay(base, "trains_tracks", dateAgo(40));
  const gz = fs.readFileSync(fp);
  fs.writeFileSync(fp, gz.subarray(0, gz.length - 6));
  fs.utimesSync(fp, (NOW - 2 * 3600_000) / 1000, (NOW - 2 * 3600_000) / 1000);
  const tier = tierFor(base, r2);
  const s = await tier.tick();
  assert.equal(s.uploaded, 0);
  assert.equal(r2.store.size, 0);
  assert.ok(fs.existsSync(fp));
  assert.match(tier.status().conflicts[0].reason, /gzip is unreadable/);
});

test("flag off: no uploads, no evictions — but days already evicted stay readable", async () => {
  const base = tmp();
  const r2 = fakeR2();
  writeDay(base, "vessels_tracks", dateAgo(12));
  const on = tierFor(base, r2);
  await on.tick();
  assert.ok(!fs.existsSync(path.join(base, "vessels_tracks", `${dateAgo(12)}.jsonl.gz`)));
  const fresh = writeDay(base, "vessels_tracks", dateAgo(11));
  const off = tierFor(base, r2, { env: { ARCHIVE_PERMANENT_TIER: "0" } });
  const calls = r2.log.length;
  const s = await off.tick();
  assert.equal(s.enabled, false);
  assert.equal(r2.log.length, calls, "no R2 traffic at all from the tick");
  assert.ok(fs.existsSync(fresh));
  setColdDayFileSource(off.coldSource);
  try {
    const r = await readArchiveDay("vessels_tracks", dateAgo(12), base, 100);
    assert.equal(r!.rows.length, 2, "the evicted day still reads from R2");
  } finally { setColdDayFileSource(null); }
});

// ── read-through ─────────────────────────────────────────────────────────────
test("read-through: /api/diag/archive's reader serves an evicted day from R2 — same rows, real file name, SHA-checked, cached", async () => {
  const base = tmp();
  const r2 = fakeR2();
  const d = dateAgo(10);
  writeDay(base, "vessels_tracks", d, rollupRows(d, ["a", "b", "c"]));
  const before = await readArchiveDay("vessels_tracks", d, base, 100);
  const tier = tierFor(base, r2);
  await tier.tick();
  assert.ok(!fs.existsSync(path.join(base, "vessels_tracks", `${d}.jsonl.gz`)), "evicted");
  const localOnly = await readArchiveDay("vessels_tracks", d, base, 100);
  assert.equal(localOnly!.rows.length, 0, "without the cold source the day is simply absent (pre-existing behavior)");
  setColdDayFileSource(tier.coldSource);
  try {
    const after = await readArchiveDay("vessels_tracks", d, base, 100);
    assert.deepEqual(after!.rows, before!.rows, "identical rows");
    assert.deepEqual(after!.files, [`${d}.jsonl.gz`], "the original basename");
    const gets = r2.log.filter((x) => x.method === "GET" && x.key).length;
    await readArchiveDay("vessels_tracks", d, base, 100);
    assert.equal(r2.log.filter((x) => x.method === "GET" && x.key).length, gets, "second read is a cache hit");
    assert.equal(tier.status().readThroughCache.hits, 1);
    // bbox filter still applies to cold rows
    const none = await readArchiveDay("vessels_tracks", d, base, 100, () => false);
    assert.equal(none!.rows.length, 0);
    // a day never offloaded and not local: no R2 call, no rows
    const g0 = r2.log.length;
    const missing = await readArchiveDay("vessels_tracks", dateAgo(60), base, 100);
    assert.equal(missing!.rows.length, 0);
    assert.equal(r2.log.length, g0, "no manifest entry, no request");
    // a non-permanent stream never touches R2
    fs.mkdirSync(path.join(base, "fires"), { recursive: true });
    await readArchiveDay("fires", d, base, 100);
    assert.equal(r2.log.length, g0);
  } finally { setColdDayFileSource(null); }
});

test("read-through: bytes that fail the recorded SHA-256 are never served", async () => {
  const base = tmp();
  const r2 = fakeR2();
  const d = dateAgo(10);
  writeDay(base, "aircraft_tracks", d);
  const errors: string[] = [];
  const tier = tierFor(base, r2, { errors });
  await tier.tick();
  const k = key("aircraft_tracks", d);
  const b = Buffer.from(r2.store.get(k)!);
  b[b.length - 1] ^= 0xff;
  r2.store.set(k, b); // R2 copy damaged after eviction
  setColdDayFileSource(tier.coldSource);
  try {
    const r = await readArchiveDay("aircraft_tracks", d, base, 100);
    assert.equal(r!.rows.length, 0);
    assert.equal(tier.status().readThroughCache.errors, 1);
    assert.match(errors.join("\n"), /sha256 .* != recorded .* not served/);
  } finally { setColdDayFileSource(null); }
});

// ── reconcile ────────────────────────────────────────────────────────────────
test("reconcile: a lost manifest re-adopts listed objects (evicted days stay readable); an evicted object gone from R2 is MISSING, never dropped", async () => {
  const base = tmp();
  const r2 = fakeR2();
  let now = NOW;
  writeDay(base, "vessels_tracks", dateAgo(10));
  writeDay(base, "vessels_tracks", dateAgo(9));
  const t1 = tierFor(base, r2, { now: () => now });
  await t1.tick();
  assert.equal(listPermanentLocalFiles(base).length, 0, "both evicted");
  // the manifest is lost
  fs.rmSync(path.join(base, PERMANENT_MANIFEST_NAME));
  const t2 = tierFor(base, r2, { now: () => now });
  const s = await t2.tick();
  assert.deepEqual(s.reconciled, { adopted: 2, missing: 0, dropped: 0 });
  setColdDayFileSource(t2.coldSource);
  try {
    const r = await readArchiveDay("vessels_tracks", dateAgo(10), base, 100);
    assert.equal(r!.rows.length, 2, "readable again after the re-adopt");
  } finally { setColdDayFileSource(null); }
  // an evicted object disappears from R2 (e.g. a bucket rule): loud, kept, not served
  r2.store.delete(key("vessels_tracks", dateAgo(9)));
  now += 24 * 3600_000;
  const errors: string[] = [];
  const t3 = tierFor(base, r2, { now: () => now, errors });
  const s3 = await t3.tick();
  assert.equal(s3.reconciled!.missing, 1);
  assert.deepEqual(t3.status().missing, [key("vessels_tracks", dateAgo(9))]);
  assert.ok(loadPermanentManifest(base).entries[key("vessels_tracks", dateAgo(9))], "the entry is kept as evidence");
  assert.deepEqual(t3.coldSource.heldOnlyInR2().map((f) => f.file), [`${dateAgo(10)}.jsonl.gz`], "a MISSING object is not reported as held");
  assert.match(errors.join("\n"), /MISSING permanent\/vessels_tracks\//);
  // a later listing that FAILS cannot clear the alarm: the key stays MISSING
  r2.ctl.down = true;
  now += 24 * 3600_000;
  const t4 = tierFor(base, r2, { now: () => now, errors: [] });
  await t4.tick();
  assert.deepEqual(t4.status().missing, [key("vessels_tracks", dateAgo(9))], "kept through a failed listing");
});

test("read-through serves every recorded entry even after its stream leaves the tier's stream list", async () => {
  const base = tmp();
  const r2 = fakeR2();
  writeDay(base, "vessels_tracks", dateAgo(12));
  await tierFor(base, r2).tick();
  // a later config without vessels_tracks: no new offloads, but the evicted day still reads
  const narrowed = createPermanentTier({
    base, client: r2.client, env: {} as NodeJS.ProcessEnv, nowMs: () => NOW, streams: ["aircraft_tracks"],
    cacheDir: path.join(base, "..", path.basename(base) + "-pcache2"), log: () => {}, logError: () => {},
  });
  setColdDayFileSource(narrowed.coldSource);
  try {
    const r = await readArchiveDay("vessels_tracks", dateAgo(12), base, 100);
    assert.equal(r!.rows.length, 2);
  } finally { setColdDayFileSource(null); }
});

test("not configured: nothing happens and nothing is deleted", async () => {
  const base = tmp();
  const fp = writeDay(base, "vessels_tracks", dateAgo(30));
  const s = await tierFor(base, null).tick();
  assert.equal(s.uploaded + s.evicted, 0);
  assert.ok(fs.existsSync(fp));
  assert.equal(fs.existsSync(path.join(base, PERMANENT_MANIFEST_NAME)), false);
});

// ── integration with the rolling tier ────────────────────────────────────────
test("integration: the offload service runs the tier each tick and reports it; the rolling window's R2 delete never touches permanent/", async () => {
  const base = tmp();
  const r2 = fakeR2();
  writeDay(base, "vessels_tracks", dateAgo(40));
  // a rolling-tier object past the 30-day replay window (the daily job deletes it)
  r2.store.set(`archive/aircraft/${dateAgo(45)}/00.jsonl.gz`, zlib.gzipSync("x\n"));
  const svc = createArchiveOffloadService({
    base, client: r2.client, env: ENV as NodeJS.ProcessEnv, nowMs: () => NOW,
    freeBytes: () => 50 * 1024 ** 3, beforeEvictAircraft: async () => {},
    cacheDir: path.join(base, "..", path.basename(base) + "-cache"),
    log: () => {}, logError: () => {},
  });
  const s = (await svc.runTick())!;
  assert.equal(s.permanent!.uploaded, 1);
  assert.equal(s.permanent!.evicted, 1);
  assert.equal(r2.store.has(`archive/aircraft/${dateAgo(45)}/00.jsonl.gz`), false, "rolling delete ran");
  assert.equal(r2.store.has(key("vessels_tracks", dateAgo(40))), true, "permanent object kept");
  await svc.runTick({ forceDaily: true });
  assert.equal(r2.store.has(key("vessels_tracks", dateAgo(40))), true, "still kept after another daily run");
  const st = await svc.status();
  assert.equal(st.permanentTier.evictedFiles, 1);
  assert.equal(st.permanentTier.enabled, true);
  assert.match(st.notes[0], /permanent\/ prefix/);
  assert.ok(st.permanentTier.r2Bytes > 0);
  assert.ok(st.estimatedMonthlyCost.atCurrentGrowth.storedBytes >= st.permanentTier.r2Bytes, "the cost estimate counts the never-expiring objects");
});

test("occvolume keeps 30 days local (its readers need the newest 5 trading days); older report days go after a read-back, counted as held", async () => {
  const base = tmp();
  const r2 = fakeR2();
  const keep = writeDay(base, "occvolume", dateAgo(30), ['{"qty":1}', '{"qty":2}', '{"qty":3}']);
  const go = writeDay(base, "occvolume", dateAgo(31), ['{"qty":4}']);
  fs.writeFileSync(path.join(base, "occvolume", "backfill_done.json"), "{}"); // the backfill marker is never touched
  const tier = tierFor(base, r2);
  const s = await tier.tick();
  assert.equal(s.uploaded, 2);
  assert.equal(s.evicted, 1);
  assert.ok(fs.existsSync(keep), "day 30: kept");
  assert.ok(!fs.existsSync(go), "day 31: evicted");
  assert.ok(fs.existsSync(path.join(base, "occvolume", "backfill_done.json")));
  assert.equal(r2.store.has("permanent/occvolume/backfill_done.json"), false);
  assert.deepEqual(tier.coldSource.heldOnlyInR2(), [{ stream: "occvolume", file: `${dateAgo(31)}.jsonl.gz`, bytes: r2.store.get(key("occvolume", dateAgo(31)))!.length, lines: 1 }]);
  assert.equal(tier.status().byStream.occvolume.localDays, 30);
});

test("fires and buoys keep 31 days local; older days go after a read-back", async () => {
  const base = tmp();
  const r2 = fakeR2();
  const fKeep = writeDay(base, "fires", dateAgo(31), ['{"id":"a"}']);
  const fGo = writeDay(base, "fires", dateAgo(32), ['{"id":"b"}']);
  const bKeep = writeDay(base, "buoys", dateAgo(31), ['{"station":"41001"}']);
  const bGo = writeDay(base, "buoys", dateAgo(40), ['{"station":"41002"}']);
  fs.writeFileSync(path.join(base, "buoys", `${dateAgo(0)}.jsonl`), '{"station":"x"}\n'); // today's plain file: never this tier's
  const s = await tierFor(base, r2).tick();
  assert.equal(s.uploaded, 4);
  assert.equal(s.evicted, 2);
  assert.ok(fs.existsSync(fKeep) && fs.existsSync(bKeep), "day 31 kept");
  assert.ok(!fs.existsSync(fGo) && !fs.existsSync(bGo), "older evicted");
  assert.ok(fs.existsSync(path.join(base, "buoys", `${dateAgo(0)}.jsonl`)));
});

test("RATCHET: every stream's local window covers the longest in-process look-back of its readers", () => {
  // queryEngine (analyst console: fires/nwsalerts/usgswater event layers and snapshots)
  // reads at most RAW_RETENTION_DAYS back; widening it past fires' window must fail here
  assert.ok(DAYS_CAP <= RAW_RETENTION_DAYS);
  const snap = snapshotWindow(NOW);
  const oldestSnapshotAge = Math.ceil((NOW - snap.minMs) / DAY);
  assert.ok(permanentLocalDays("fires") > Math.max(DAYS_CAP, oldestSnapshotAge), "fires: queryEngine's reach stays local");
  assert.ok(permanentLocalDays("fires") > SITE_TIMELINE_DAYS, "fires: siteTimeline's 7-day window stays local");
  assert.ok(permanentLocalDays("fires") > 7, "fires: readFireHistory default (7 d) stays local");
  assert.ok(permanentLocalDays("buoys") > 2, "buoys: readBuoyHistory / seedSeen (2 d) stay local");
  assert.ok(permanentLocalDays("occvolume") > 7, "occvolume: refreshOcc's 5 trading days stay local");
  for (const st of ["vessels_tracks", "aircraft_tracks", "trains_tracks"]) assert.ok(permanentLocalDays(st) >= 1);
  // dtccswaps: readArchivedDtccRows (the cold-boot backfill of the live cache) reads local
  // files of ages 0 .. lookback-1; a file leaves only when its age is ABOVE the local window
  assert.ok(permanentLocalDays("dtccswaps") >= DTCC_ARCHIVE_BACKFILL_LOOKBACK_DAYS, "dtccswaps: the cold-boot backfill's look-back stays local");
});

// ── dtccswaps (2026-11-25, open_questions (aa4)) ─────────────────────────────
function dtccRows(ids: readonly string[]): DtccSwapRow[] {
  const cols = [...REQUIRED_COLUMNS];
  const idx = parseHeader(cols.map((c) => `"${c}"`).join(","));
  return ids.map((d) => {
    const byName: Record<string, string> = {
      "Dissemination Identifier": d, "Original Dissemination Identifier": d, "Action type": "NEWT",
      "Event timestamp": "2026-11-05T12:33:36Z", "Effective Date": "2026-11-05", "Asset Class": "EQ",
      "Notional amount-Leg 1": "28,000,000", "Notional currency-Leg 1": "USD", "Underlier ID-Leg 1": "US0378331005",
      "Underlier ID source-Leg 1": "ISIN", "UPI Underlier Name": "APPLE INC",
    };
    return parseDtccLine(cols.map((c) => `"${byName[c] ?? ""}"`).join(","), idx, cols.length)!;
  });
}

test("dtccswaps end to end: day files older than 14 days leave after a read-back, the directory's bookkeeping never does; a lost dedup index is then rebuilt whole from the tier and no evicted row is archived again", async () => {
  const base = tmp();
  const r2 = fakeR2();
  const dir = path.join(base, "dtccswaps");
  const idOf = (day: number, i: number) => `61${String(day).padStart(3, "0")}${String(i).padStart(12, "0")}01`;
  const ages = [20, 15, 14, 3];
  const idsOfAge = (age: number) => Array.from({ length: 25 }, (_, i) => idOf(age, i));
  const prev = process.env.VOLTRADE_DTCC_IDS_STORE;
  process.env.VOLTRADE_DTCC_IDS_STORE = "memory";
  _resetDtccForTests();
  const cErr = console.error, cLog = console.log;
  console.error = () => {}; console.log = () => {};
  try {
    // written by the real archiver, then idle 2 h like any closed day file
    for (const age of ages) {
      assert.equal(archiveNewRows(dtccRows(idsOfAge(age)), dateAgo(age), base).newRows, 25);
      const fp = path.join(dir, `${dateAgo(age)}.jsonl.gz`);
      fs.utimesSync(fp, (NOW - 2 * 3600_000) / 1000, (NOW - 2 * 3600_000) / 1000);
    }
    fs.writeFileSync(path.join(dir, "cusip_ticker_cache.json"), "{}");
    fs.writeFileSync(path.join(dir, "_seen_ids.db"), "derived");
    const indexBefore = fs.readFileSync(path.join(dir, "_seen_ids_index.jsonl.gz"));
    const evictedBytes = [20, 15].map((a) => fs.readFileSync(path.join(dir, `${dateAgo(a)}.jsonl.gz`)));

    const tier = tierFor(base, r2);
    const s = await tier.tick();
    assert.equal(s.uploaded, 4, "every closed day file is copied");
    assert.equal(s.evicted, 2, "only the two older than the 14-day window leave");
    assert.deepEqual(s.errors, []);
    assert.ok(!fs.existsSync(path.join(dir, `${dateAgo(20)}.jsonl.gz`)) && !fs.existsSync(path.join(dir, `${dateAgo(15)}.jsonl.gz`)));
    assert.ok(fs.existsSync(path.join(dir, `${dateAgo(14)}.jsonl.gz`)), "day 14 stays: the cold-boot backfill may read it");
    assert.ok(fs.existsSync(path.join(dir, `${dateAgo(3)}.jsonl.gz`)));
    assert.deepEqual(r2.store.get(key("dtccswaps", dateAgo(20))), evictedBytes[0], "byte for byte");
    assert.deepEqual(r2.store.get(key("dtccswaps", dateAgo(15))), evictedBytes[1]);
    // the index, the dedup DB and the CUSIP cache are not day files: untouched, never uploaded
    assert.deepEqual(fs.readFileSync(path.join(dir, "_seen_ids_index.jsonl.gz")), indexBefore);
    assert.ok(fs.existsSync(path.join(dir, "_seen_ids.db")) && fs.existsSync(path.join(dir, "cusip_ticker_cache.json")));
    assert.deepEqual(Array.from(r2.store.keys()).filter((k) => k.startsWith("permanent/dtccswaps/")).sort(),
      ages.map((a) => key("dtccswaps", dateAgo(a))).sort());
    assert.equal(tier.status().byStream.dtccswaps.localDays, 14);
    assert.deepEqual(tier.coldSource.heldOnlyInR2().map((f) => `${f.stream}/${f.file}:${f.lines}`).sort(),
      [20, 15].map((a) => `dtccswaps/${dateAgo(a)}.jsonl.gz:25`).sort(), "the archive's counters still count the evicted rows");

    // with the index intact nothing changes for the archiver: an evicted day's rows are known
    _resetDtccForTests();
    assert.deepEqual(await archiveNewRowsAsync(dtccRows(idsOfAge(20).slice(0, 5)), dateAgo(0), base), { newRows: 0, seenTotal: 100 });
    assert.deepEqual(dtccOwedDayFiles(base), []);

    // the index is lost. The rebuild scans two local days, records the two evicted ones as
    // owed, and the same poll reads them back from the tier (SHA-256 checked against the manifest).
    fs.rmSync(path.join(dir, "_seen_ids_index.jsonl.gz"));
    _resetDtccForTests();
    setColdDayFileSource(tier.coldSource);
    const getsBefore = r2.log.filter((x) => x.method === "GET" && x.key.startsWith("permanent/dtccswaps/")).length;
    const served = [...idsOfAge(20).slice(0, 5), ...idsOfAge(15).slice(0, 5), ...idsOfAge(3).slice(0, 5), idOf(0, 1)];
    const r = await archiveNewRowsAsync(dtccRows(served), dateAgo(0), base);
    assert.deepEqual(r, { newRows: 1, seenTotal: 101 }, "one row never archived; none of an evicted day written again");
    assert.deepEqual(dtccOwedDayFiles(base), []);
    assert.equal(r2.log.filter((x) => x.method === "GET" && x.key.startsWith("permanent/dtccswaps/")).length - getsBefore, 2, "one read per evicted day");
    const lines = zlib.gunzipSync(fs.readFileSync(path.join(dir, "_seen_ids_index.jsonl.gz"))).toString("utf8").split("\n").filter(Boolean);
    assert.equal(lines.length, 101);
    assert.equal(new Set(lines).size, 101, "every ID once");

    // the tier down at the moment the index is lost: the poll still archives, the days stay owed
    fs.rmSync(path.join(dir, "_seen_ids_index.jsonl.gz"));
    _resetDtccForTests();
    const down = tierFor(base, r2); // a fresh process: empty read-through cache
    setColdDayFileSource(down.coldSource);
    r2.ctl.down = true;
    const d2 = await archiveNewRowsAsync(dtccRows([idOf(0, 2)]), dateAgo(0), base);
    assert.equal(d2.newRows, 1, "an unarchived day does not refill: the row is written");
    assert.deepEqual(dtccOwedDayFiles(base), [20, 15].map((a) => `${dateAgo(a)}.jsonl.gz`).sort());
    r2.ctl.down = false;
    const d3 = await archiveNewRowsAsync(dtccRows([idOf(0, 2), ...idsOfAge(20).slice(0, 3)]), dateAgo(0), base);
    assert.deepEqual(d3, { newRows: 0, seenTotal: 102 });
    assert.deepEqual(dtccOwedDayFiles(base), []);
  } finally {
    console.error = cErr; console.log = cLog;
    setColdDayFileSource(null);
    _resetDtccForTests();
    if (prev === undefined) delete process.env.VOLTRADE_DTCC_IDS_STORE; else process.env.VOLTRADE_DTCC_IDS_STORE = prev;
  }
});
