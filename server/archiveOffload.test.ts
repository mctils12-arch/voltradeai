import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import os from "os";
import path from "path";
import zlib from "zlib";
import {
  createArchiveOffloadService, registerArchiveOffloadRoutes, createColdHourSource,
  retentionConfig, effectiveRetentionDays, ageDays, isDateExpired, oldestReplayableDate,
  hourKey, eventsKey, dateOfKey, estimateR2MonthlyCost, loadManifest, isOffloaded, listLocalHourUnits,
  HOT_TIER_FLOOR_DAYS, HOT_TIER_TARGET_FREE_BYTES, MANIFEST_NAME, OFFLOAD_MAX_UNITS_PER_TICK,
  type OffloadManifest,
} from "./archiveOffload";
import { createR2Client, r2ConfigFromEnv } from "./r2Client";
import { readWindow, setColdHourSource } from "./aircraftWindow";
import { rollupDayAsync, rollupOldDaysAsync, RAW_RETENTION_DAYS } from "./datacoreArchive";

// Fixed clock in the PAST (readWindow treats hours after the real clock as
// "future, not a gap", so fixtures must sit before real time).
const NOW = Date.parse("2026-09-20T12:00:00Z");
const DAY = 86_400_000;
const ENV = {
  R2_ACCOUNT_ID: "0123456789abcdef0123456789abcdef",
  R2_ACCESS_KEY_ID: "AKID", R2_SECRET_ACCESS_KEY: "TOPSECRETKEY", R2_ARCHIVE_BUCKET: "voltrade-archive-private",
};
const CFG = r2ConfigFromEnv(ENV as any)!;

function tmp(prefix = "vt-offload-"): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), prefix));
}

/** UTC date string `n` days before NOW's date */
function dateAgo(n: number): string {
  return new Date(Date.parse("2026-09-20T00:00:00Z") - n * DAY).toISOString().slice(0, 10);
}

function hourSecOf(date: string, hh: number): number {
  return Date.parse(`${date}T${String(hh).padStart(2, "0")}:00:00Z`) / 1000;
}

function rows(date: string, hh: number, hexes = ["aaaaaa"]): string[] {
  const t0 = hourSecOf(date, hh);
  return hexes.map((i, k) => JSON.stringify({ t: t0 + 60 + k, i, la: 40 + k * 0.01, lo: -100, al: 9000 }));
}

/** write an hour file in the archive's own shape (.jsonl.gz by default) */
function writeHour(base: string, kind: string, date: string, hh: number, lines: string[], gz = true): string {
  const dir = path.join(base, kind);
  fs.mkdirSync(dir, { recursive: true });
  const fp = path.join(dir, `${date}-${String(hh).padStart(2, "0")}.jsonl${gz ? ".gz" : ""}`);
  const raw = lines.join("\n") + "\n";
  fs.writeFileSync(fp, gz ? zlib.gzipSync(raw) : raw);
  return fp;
}

/** In-memory S3/R2 behind a fake fetch — the REAL r2Client talks to it, so
 *  signing/URL/XML paths are exercised end to end. */
function fakeR2() {
  const store = new Map<string, Buffer>();
  const log: Array<{ method: string; key: string }> = [];
  const ctl = { failPutKeys: new Set<string>(), truncatePut: new Set<string>(), down: false };
  const fetchImpl = (async (url: any, init: any = {}) => {
    const u = new URL(String(url));
    const parts = u.pathname.split("/").slice(2); // ["", bucket, ...key]
    const key = decodeURIComponent(parts.join("/"));
    const method = init.method || "GET";
    log.push({ method, key });
    if (ctl.down) throw new Error("ECONNREFUSED");
    if (method === "PUT") {
      if (ctl.failPutKeys.has(key)) return new Response("<Error><Code>InternalError</Code></Error>", { status: 400 });
      const b = Buffer.from(init.body);
      store.set(key, ctl.truncatePut.has(key) ? b.subarray(0, 3) : b);
      return new Response(null, { status: 200, headers: { etag: '"e"' } });
    }
    if (method === "HEAD") {
      const b = store.get(key);
      if (!b) return new Response(null, { status: 404 });
      return new Response(null, { status: 200, headers: { "content-length": String(b.length) } });
    }
    if (method === "GET" && key) {
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
      const body = Buffer.from(init.body).toString();
      for (const m of Array.from(body.matchAll(/<Key>([^<]*)<\/Key>/g))) store.delete(m[1]);
      return new Response("<DeleteResult></DeleteResult>", { status: 200 });
    }
    return new Response(null, { status: 400 });
  }) as typeof fetch;
  const client = createR2Client(CFG, { fetchImpl, sleep: async () => {}, maxRetries: 0 });
  return { store, log, ctl, client };
}

function svcFor(base: string, r2: ReturnType<typeof fakeR2> | null, o: {
  now?: () => number; free?: () => number | null; env?: Record<string, string>;
  folds?: Array<{ cutoffMs: number }>; maxUnits?: number; errors?: string[];
} = {}) {
  return createArchiveOffloadService({
    base,
    client: r2 ? r2.client : createR2Client(null),
    env: { ...(r2 ? ENV : {}), ...(o.env || {}) } as any,
    nowMs: o.now ?? (() => NOW),
    freeBytes: o.free ?? (() => 50 * 1024 ** 3),
    beforeEvictAircraft: async (_b, cutoffMs) => { o.folds?.push({ cutoffMs }); },
    maxUnitsPerTick: o.maxUnits,
    cacheDir: path.join(base, "..", path.basename(base) + "-cache"),
    log: () => {},
    logError: (m) => { o.errors?.push(m); },
  });
}

// ── retention math: the rolling boundary (UTC) ───────────────────────────────
test("rolling boundary: exactly day 30 kept, day 31 expired, computed in UTC", () => {
  const now = Date.parse("2026-09-28T12:00:00Z");
  assert.equal(oldestReplayableDate(now, 30), "2026-08-29");
  assert.equal(ageDays("2026-08-29", now), 30);
  assert.equal(isDateExpired("2026-08-29", now, 30), false, "day 30 kept");
  assert.equal(isDateExpired("2026-08-28", now, 30), true, "day 31 deleted");
  // the whole UTC day keeps the same boundary...
  assert.equal(oldestReplayableDate(Date.parse("2026-09-28T00:00:00Z"), 30), "2026-08-29");
  assert.equal(oldestReplayableDate(Date.parse("2026-09-28T23:59:59.999Z"), 30), "2026-08-29");
  // ...and it rolls at UTC midnight, not at a local-time midnight: 20:00 in
  // UTC-7 on the 28th is already the 29th in UTC
  assert.equal(oldestReplayableDate(Date.parse("2026-09-28T20:00:00-07:00"), 30), "2026-08-30");
  assert.equal(isDateExpired("2026-08-29", Date.parse("2026-09-29T00:00:00Z"), 30), true);
});

test("retention config: REPLAY_RETENTION_DAYS default 30, LOCAL_HOT_DAYS default min(replay, 30), floors enforced", () => {
  assert.deepEqual(retentionConfig({} as any), { replayDays: 30, replayClamped: false, localHotDays: 30, localHotClamped: false });
  const long = retentionConfig({ REPLAY_RETENTION_DAYS: "45" } as any);
  assert.equal(long.replayDays, 45);
  assert.equal(long.localHotDays, RAW_RETENTION_DAYS, "local never keeps raw longer than RAW_RETENTION_DAYS");
  const short = retentionConfig({ REPLAY_RETENTION_DAYS: "10" } as any);
  assert.equal(short.replayDays, HOT_TIER_FLOOR_DAYS, "clamped to the GNSS-signal floor");
  assert.equal(short.replayClamped, true);
  assert.equal(retentionConfig({ LOCAL_HOT_DAYS: "7" } as any).localHotDays, HOT_TIER_FLOOR_DAYS);
  assert.equal(retentionConfig({ LOCAL_HOT_DAYS: "25" } as any).localHotDays, 25);
  assert.equal(retentionConfig({ REPLAY_RETENTION_DAYS: "abc" } as any).replayDays, 30);
  // without R2 the local rule stays exactly RAW_RETENTION_DAYS whatever the env says
  assert.equal(effectiveRetentionDays(false, retentionConfig({ REPLAY_RETENTION_DAYS: "60" } as any)), RAW_RETENTION_DAYS);
  assert.equal(effectiveRetentionDays(true, retentionConfig({ REPLAY_RETENTION_DAYS: "60" } as any)), 60);
  assert.equal(RAW_RETENTION_DAYS, 30, "the local constant is untouched");
});

test("R2 key layout: archive/<kind>/<YYYY-MM-DD>/<HH>.jsonl.gz", () => {
  assert.equal(hourKey("aircraft", "2026-09-20-13"), "archive/aircraft/2026-09-20/13.jsonl.gz");
  assert.equal(hourKey("vessels", "2026-01-02-00"), "archive/vessels/2026-01-02/00.jsonl.gz");
  assert.equal(eventsKey("2026-09-20", "2026-09-20"), "archive/flight_events/2026-09-20/2026-09-20.jsonl.gz");
  assert.equal(dateOfKey("archive/aircraft/2026-09-20/13.jsonl.gz"), "2026-09-20");
  assert.equal(dateOfKey("tiles/power.pmtiles"), null);
});

// ── offload ──────────────────────────────────────────────────────────────────
test("offload: only COMPLETED hours older than 2 days upload; gz byte-identical, plain gzipped; verified + manifested; idempotent", async () => {
  const base = tmp();
  const r2 = fakeR2();
  const gzFp = writeHour(base, "aircraft", dateAgo(5), 10, rows(dateAgo(5), 10));
  writeHour(base, "vessels", dateAgo(5), 11, rows(dateAgo(5), 11, ["244010352"]), false); // plain (compress missed it)
  writeHour(base, "aircraft", dateAgo(1), 10, rows(dateAgo(1), 10));                     // too recent
  writeHour(base, "aircraft", dateAgo(0), 11, rows(dateAgo(0), 11));                     // current day
  writeHour(base, "trains", dateAgo(5), 10, rows(dateAgo(5), 10, ["FI-1"]));              // not an offload kind
  const svc = svcFor(base, r2);
  const s = (await svc.runTick())!;
  assert.equal(s.uploaded, 2);
  assert.deepEqual(Array.from(r2.store.keys()).sort(), [
    `archive/aircraft/${dateAgo(5)}/10.jsonl.gz`,
    `archive/vessels/${dateAgo(5)}/11.jsonl.gz`,
  ]);
  assert.deepEqual(r2.store.get(`archive/aircraft/${dateAgo(5)}/10.jsonl.gz`), fs.readFileSync(gzFp), "gz uploads as-is");
  assert.equal(zlib.gunzipSync(r2.store.get(`archive/vessels/${dateAgo(5)}/11.jsonl.gz`)!).toString(),
    rows(dateAgo(5), 11, ["244010352"]).join("\n") + "\n", "plain is gzipped, content preserved");
  // each PUT is followed by a HEAD verify
  const methods = r2.log.filter((x) => x.method === "PUT" || x.method === "HEAD").map((x) => x.method);
  assert.deepEqual(methods, ["PUT", "HEAD", "PUT", "HEAD"]);
  const m = loadManifest(base);
  const e = m.entries[`archive/aircraft/${dateAgo(5)}/10.jsonl.gz`];
  assert.equal(e.src, "gz");
  assert.equal(e.bytes, fs.statSync(gzFp).size);
  assert.equal(e.srcBytes, fs.statSync(gzFp).size);
  // local files are NOT touched by offload
  assert.ok(fs.existsSync(gzFp));
  // second tick: nothing new
  const puts = r2.log.filter((x) => x.method === "PUT").length;
  const s2 = (await svc.runTick())!;
  assert.equal(s2.uploaded, 0);
  assert.equal(r2.log.filter((x) => x.method === "PUT").length, puts, "idempotent: no re-upload");
});

test("offload is bounded per tick, oldest first, and resumes where it stopped", async () => {
  const base = tmp();
  const r2 = fakeR2();
  for (let h = 0; h < 5; h++) writeHour(base, "aircraft", dateAgo(4), h, rows(dateAgo(4), h));
  const svc = svcFor(base, r2, { maxUnits: 3 });
  const a = (await svc.runTick())!;
  assert.equal(a.uploaded, 3);
  assert.equal(a.pending, 2);
  assert.deepEqual(Array.from(r2.store.keys()).sort(), [0, 1, 2].map((h) => `archive/aircraft/${dateAgo(4)}/0${h}.jsonl.gz`));
  const b = (await svc.runTick())!;
  assert.equal(b.uploaded, 2);
  assert.equal(r2.store.size, 5);
  assert.equal(OFFLOAD_MAX_UNITS_PER_TICK, 48);
});

test("offload: a changed local file re-uploads; a both-flavors hour is skipped until it settles", async () => {
  const base = tmp();
  const r2 = fakeR2();
  const fp = writeHour(base, "aircraft", dateAgo(6), 3, rows(dateAgo(6), 3));
  writeHour(base, "aircraft", dateAgo(6), 4, rows(dateAgo(6), 4));
  writeHour(base, "aircraft", dateAgo(6), 4, rows(dateAgo(6), 4, ["bbbbbb"]), false); // .jsonl beside .jsonl.gz
  const svc = svcFor(base, r2);
  const s = (await svc.runTick())!;
  assert.equal(s.uploaded, 1);
  assert.equal(s.skippedBothFlavors, 1);
  // a late backfill grew the hour: sizes differ -> re-upload
  fs.writeFileSync(fp, zlib.gzipSync(rows(dateAgo(6), 3, ["aaaaaa", "cccccc"]).join("\n") + "\n"));
  const s2 = (await svc.runTick())!;
  assert.equal(s2.uploaded, 1);
  assert.deepEqual(r2.store.get(`archive/aircraft/${dateAgo(6)}/03.jsonl.gz`), fs.readFileSync(fp));
});

test("offload: a HEAD size mismatch is NOT verified — never licenses a local delete, error surfaced, retried", async () => {
  const base = tmp();
  const r2 = fakeR2();
  const d = dateAgo(25); // old enough to be evictable if it WERE verified
  const fp = writeHour(base, "aircraft", d, 1, rows(d, 1));
  const localBytes = fs.readFileSync(fp);
  const key = `archive/aircraft/${d}/01.jsonl.gz`;
  r2.ctl.truncatePut.add(key); // R2 keeps a truncated copy: HEAD and LIST both see 3 bytes
  const errors: string[] = [];
  const svc = svcFor(base, r2, { errors, free: () => 0 }); // max pressure
  const s = (await svc.runTick())!;
  assert.equal(s.uploaded, 0);
  assert.equal(s.uploadFailed, 1);
  assert.match(errors.join("\n"), /verify archive\/aircraft\/.*R2 size 3 != uploaded/);
  // the daily listing may adopt the (bad) object, but a size-mismatched
  // adoption never counts as verified
  const unit = listLocalHourUnits(base, "aircraft")[0];
  assert.equal(isOffloaded(unit, svc.getManifest()), false);
  assert.equal(s.evicted.length, 0, "unverified -> local copy kept even under pressure");
  assert.ok(fs.existsSync(fp));
  assert.match((await svc.status()).lastError, /verify/);
  // R2 heals -> next tick re-uploads, verifies, and only THEN may evict
  r2.ctl.truncatePut.clear();
  const s2 = (await svc.runTick())!;
  assert.equal(s2.uploaded, 1);
  assert.deepEqual(s2.evicted.map((e) => e.date), [d]);
  assert.deepEqual(r2.store.get(key), localBytes, "R2 now holds the exact local bytes");
});

// ── R2 not configured: behavior exactly as today ─────────────────────────────
test("R2 not configured: no network, no manifest, no early eviction even under pressure; only flight_events roll at 30 days", async () => {
  const base = tmp();
  const hourFp = writeHour(base, "aircraft", dateAgo(25), 5, rows(dateAgo(25), 5));
  const evDir = path.join(base, "flight_events");
  fs.mkdirSync(evDir, { recursive: true });
  fs.writeFileSync(path.join(evDir, `${dateAgo(30)}.jsonl`), "{}\n");
  fs.writeFileSync(path.join(evDir, `${dateAgo(31)}.jsonl`), "{}\n");
  let fetched = 0;
  const svc = createArchiveOffloadService({
    base, env: {} as any, client: createR2Client(r2ConfigFromEnv({} as any), { fetchImpl: (async () => { fetched++; return new Response(null); }) as any }),
    nowMs: () => NOW, freeBytes: () => 1, log: () => {}, logError: () => {}, cacheDir: tmp("vt-cache-"),
  });
  const s = (await svc.runTick())!;
  assert.equal(s.configured, false);
  assert.equal(fetched, 0);
  assert.ok(fs.existsSync(hourFp), "hour file untouched (RAW_RETENTION_DAYS rollup still owns it)");
  assert.equal(fs.existsSync(path.join(base, MANIFEST_NAME)), false);
  assert.deepEqual(fs.readdirSync(evDir), [`${dateAgo(30)}.jsonl`], "day 30 kept, day 31 rolled off");
  const st = await svc.status();
  assert.equal(st.configured, false);
  assert.equal(st.retentionDays, 30);
  assert.match(st.retentionSource, /RAW_RETENTION_DAYS/);
});

// ── hot-tier eviction: delete local ONLY after a verified upload ─────────────
test("eviction under pressure: a verified day is folded, rolled up, then deleted; an unverified day is kept and blocks younger days", async () => {
  const base = tmp();
  const r2 = fakeR2();
  const d27 = dateAgo(27), d26 = dateAgo(26), d25 = dateAgo(25);
  writeHour(base, "aircraft", d27, 0, rows(d27, 0));
  writeHour(base, "aircraft", d27, 1, rows(d27, 1, ["aaaaaa", "bbbbbb"]));
  writeHour(base, "aircraft", d26, 0, rows(d26, 0));
  writeHour(base, "aircraft", d25, 0, rows(d25, 0));
  // d26's upload fails -> d26 unverified
  r2.ctl.failPutKeys.add(`archive/aircraft/${d26}/00.jsonl.gz`);
  const folds: Array<{ cutoffMs: number }> = [];
  let free = HOT_TIER_TARGET_FREE_BYTES - 1; // under pressure
  const svc = svcFor(base, r2, { folds, free: () => free });
  const s = (await svc.runTick())!;
  assert.deepEqual(s.evicted, [{ kind: "aircraft", date: d27, files: 2, reason: "pressure" }]);
  // fold hook ran BEFORE the delete with a cutoff selecting exactly d27
  assert.equal(folds.length, 1);
  assert.equal(folds[0].cutoffMs, Date.parse(`${d26}T00:00:00Z`) - 1000);
  // the permanent per-day rollup survived the early delete
  const summary = zlib.gunzipSync(fs.readFileSync(path.join(base, "aircraft_tracks", `${d27}.jsonl.gz`))).toString();
  assert.match(summary, /"i":"aaaaaa"/);
  assert.match(summary, /"i":"bbbbbb"/);
  const left = fs.readdirSync(path.join(base, "aircraft")).sort();
  assert.deepEqual(left, [`${d26}-00.jsonl.gz`, `${d25}-00.jsonl.gz`].sort(),
    "unverified d26 kept; verified d25 kept too (contiguity: never evict past a day that must stay)");

  // R2 recovers -> d26 verifies next tick -> now d26 and d25 can go
  r2.ctl.failPutKeys.clear();
  const s2 = (await svc.runTick())!;
  assert.deepEqual(s2.evicted.map((e) => e.date), [d26, d25]);
  assert.equal(fs.readdirSync(path.join(base, "aircraft")).length, 0);

  // pressure relieved -> nothing more is evicted
  free = HOT_TIER_TARGET_FREE_BYTES * 4;
  writeHour(base, "aircraft", dateAgo(24), 0, rows(dateAgo(24), 0));
  const s3 = (await svc.runTick())!;
  assert.equal(s3.evicted.length, 0);
  assert.ok(fs.existsSync(path.join(base, "aircraft", `${dateAgo(24)}-00.jsonl.gz`)));
});

test("eviction floor: a verified day younger than HOT_TIER_FLOOR_DAYS is never evicted, even under pressure", async () => {
  const base = tmp();
  const r2 = fakeR2();
  const young = dateAgo(HOT_TIER_FLOOR_DAYS - 1);
  const edge = dateAgo(HOT_TIER_FLOOR_DAYS);
  writeHour(base, "vessels", edge, 0, rows(edge, 0, ["1"]));
  writeHour(base, "vessels", young, 0, rows(young, 0, ["1"]));
  const svc = svcFor(base, r2, { free: () => 0 });
  const s = (await svc.runTick())!;
  assert.equal(s.uploaded, 2, "both are offloaded (verified)");
  assert.deepEqual(s.evicted.map((e) => e.date), [edge], "day 22 may go; day 21 never");
  assert.ok(fs.existsSync(path.join(base, "vessels", `${young}-00.jsonl.gz`)));
  assert.ok(fs.existsSync(path.join(base, "vessels_tracks", `${edge}.jsonl.gz`)), "vessel day rollup written before delete");
});

test("eviction by age: LOCAL_HOT_DAYS=24 evicts verified days older than 24 without pressure", async () => {
  const base = tmp();
  const r2 = fakeR2();
  for (const n of [26, 25, 24]) writeHour(base, "aircraft", dateAgo(n), 0, rows(dateAgo(n), 0));
  const svc = svcFor(base, r2, { env: { LOCAL_HOT_DAYS: "24" } });
  const s = (await svc.runTick())!;
  assert.deepEqual(s.evicted.map((e) => [e.date, e.reason]), [[dateAgo(26), "age"], [dateAgo(25), "age"]]);
  assert.ok(fs.existsSync(path.join(base, "aircraft", `${dateAgo(24)}-00.jsonl.gz`)));
});

test("eviction leaves days past RAW_RETENTION_DAYS to the existing rollup (no double ownership)", async () => {
  const base = tmp();
  const r2 = fakeR2();
  const old = dateAgo(RAW_RETENTION_DAYS + 1); // rollupOldDaysAsync's territory (and past the replay window)
  writeHour(base, "aircraft", old, 0, rows(old, 0));
  const svc = svcFor(base, r2, { free: () => 0 });
  const s = (await svc.runTick())!;
  assert.equal(s.uploaded, 0, "expired hours are never uploaded");
  assert.equal(s.evicted.length, 0);
  assert.ok(fs.existsSync(path.join(base, "aircraft", `${old}-00.jsonl.gz`)));
  // the existing rollup still handles it exactly as before
  assert.equal(await rollupOldDaysAsync(base, NOW), 1);
});

// ── daily rolling delete on R2 ───────────────────────────────────────────────
test("R2 rolling delete: day 30 kept, day 31+ deleted (UTC), flight_events included, unknown in-window objects adopted", async () => {
  const base = tmp();
  const r2 = fakeR2();
  const put = (k: string) => r2.store.set(k, Buffer.from("x".repeat(10)));
  put(`archive/aircraft/${dateAgo(30)}/23.jsonl.gz`);
  put(`archive/aircraft/${dateAgo(31)}/00.jsonl.gz`);
  put(`archive/aircraft/${dateAgo(45)}/05.jsonl.gz`);
  put(`archive/vessels/${dateAgo(31)}/12.jsonl.gz`);
  put(`archive/flight_events/${dateAgo(31)}/${dateAgo(31)}.jsonl.gz`);
  put(`archive/flight_events/${dateAgo(2)}/${dateAgo(2)}.jsonl.gz`);
  // a stale manifest row whose object is gone
  const m: OffloadManifest = { version: 1, lastDailyRunAt: null, entries: {
    [`archive/vessels/${dateAgo(3)}/01.jsonl.gz`]: { kind: "vessels", date: dateAgo(3), bytes: 5, src: "gz", srcBytes: 5, at: NOW },
  } };
  fs.writeFileSync(path.join(base, MANIFEST_NAME), JSON.stringify(m));
  const svc = svcFor(base, r2);
  const s = (await svc.runTick({ forceDaily: true }))!;
  assert.deepEqual(Array.from(r2.store.keys()).sort(), [
    `archive/aircraft/${dateAgo(30)}/23.jsonl.gz`,
    `archive/flight_events/${dateAgo(2)}/${dateAgo(2)}.jsonl.gz`,
  ]);
  assert.equal(s.daily!.r2Deleted, 4);
  assert.equal(s.daily!.adopted, 2, "in-window objects a lost manifest forgot are adopted");
  const mm = svc.getManifest();
  assert.ok(mm.entries[`archive/aircraft/${dateAgo(30)}/23.jsonl.gz`]);
  assert.equal(mm.entries[`archive/aircraft/${dateAgo(31)}/00.jsonl.gz`], undefined);
  assert.equal(mm.entries[`archive/vessels/${dateAgo(3)}/01.jsonl.gz`], undefined, "entry R2 no longer has is dropped");
  assert.equal(mm.lastDailyRunAt, NOW);
  const posts = r2.log.filter((x) => x.method === "POST").length;
  assert.ok(posts >= 1, "deletes go through DeleteObjects");
  // not due again within the day
  await svc.runTick();
  assert.equal(r2.log.filter((x) => x.method === "POST").length, posts);
});

test("R2 rolling delete honors REPLAY_RETENTION_DAYS (45): day 45 kept, day 46 deleted", async () => {
  const base = tmp();
  const r2 = fakeR2();
  r2.store.set(`archive/aircraft/${dateAgo(45)}/00.jsonl.gz`, Buffer.from("a"));
  r2.store.set(`archive/aircraft/${dateAgo(46)}/00.jsonl.gz`, Buffer.from("b"));
  const svc = svcFor(base, r2, { env: { REPLAY_RETENTION_DAYS: "45" } });
  await svc.runTick({ forceDaily: true });
  assert.deepEqual(Array.from(r2.store.keys()), [`archive/aircraft/${dateAgo(45)}/00.jsonl.gz`]);
  assert.equal((await svc.status()).oldestReplayableDate, dateAgo(45));
});

test("flight_events: daily files offload like hours and roll off locally past the window", async () => {
  const base = tmp();
  const r2 = fakeR2();
  const ev = path.join(base, "flight_events");
  fs.mkdirSync(ev, { recursive: true });
  fs.writeFileSync(path.join(ev, `${dateAgo(4)}.jsonl`), '{"type":"DEVIATION_START"}\n');
  fs.writeFileSync(path.join(ev, `${dateAgo(1)}.jsonl`), '{"type":"REPLANNED"}\n'); // day not complete + 2d yet
  fs.writeFileSync(path.join(ev, `${dateAgo(31)}.jsonl.gz`), zlib.gzipSync("{}\n"));
  fs.writeFileSync(path.join(ev, "state.json"), "{}"); // not a day file: ignored
  const svc = svcFor(base, r2);
  const s = (await svc.runTick())!;
  assert.equal(s.uploaded, 1);
  assert.equal(zlib.gunzipSync(r2.store.get(`archive/flight_events/${dateAgo(4)}/${dateAgo(4)}.jsonl.gz`)!).toString(),
    '{"type":"DEVIATION_START"}\n');
  assert.equal(s.localEventsExpired, 1);
  assert.deepEqual(fs.readdirSync(ev).sort(), [`${dateAgo(1)}.jsonl`, `${dateAgo(4)}.jsonl`, "state.json"].sort());
});

test("no flight_events directory is fine (agent C's dir may not exist yet)", async () => {
  const base = tmp();
  const svc = svcFor(base, fakeR2());
  const s = (await svc.runTick())!;
  assert.equal(s.errors.length, 0);
});

// ── cold reader ─────────────────────────────────────────────────────────────
test("cold source: expired / available / absent; fetch caches; LRU byte cap evicts the least recently used", async () => {
  const r2 = fakeR2();
  const h1 = hourSecOf(dateAgo(3), 1), h2 = hourSecOf(dateAgo(3), 2), h3 = hourSecOf(dateAgo(3), 3);
  const manifest: OffloadManifest = { version: 1, lastDailyRunAt: null, entries: {} };
  for (const h of [h1, h2, h3]) {
    const k = `archive/aircraft/${dateAgo(3)}/0${new Date(h * 1000).getUTCHours()}.jsonl.gz`;
    r2.store.set(k, zlib.gzipSync("x".repeat(2000)));
    manifest.entries[k] = { kind: "aircraft", date: dateAgo(3), bytes: r2.store.get(k)!.length, src: "gz", srcBytes: 1, at: NOW };
  }
  const size = r2.store.get(`archive/aircraft/${dateAgo(3)}/01.jsonl.gz`)!.length;
  const cacheDir = tmp("vt-cold-");
  const src = createColdHourSource({
    client: r2.client, getManifest: () => manifest, retentionDays: () => 30, nowMs: () => NOW,
    cacheDir, cacheMaxBytes: size * 2 + 1,
  });
  assert.equal(src.locate("aircraft", h1), "available");
  assert.equal(src.locate("aircraft", hourSecOf(dateAgo(3), 9)), "absent");
  assert.equal(src.locate("aircraft", hourSecOf(dateAgo(31), 0)), "expired");
  assert.equal(src.locate("aircraft", hourSecOf(dateAgo(30), 23)), "absent", "day 30 is inside the window (not expired), just not offloaded");
  const a = await src.fetch("aircraft", h1);
  assert.ok(a && fs.existsSync(a.fp));
  await src.fetch("aircraft", h1); // cache hit, refreshes recency
  await src.fetch("aircraft", h2);
  await src.fetch("aircraft", h3); // over cap -> evicts LRU (h1 used before h2)
  const st = src.stats();
  assert.equal(st.fetches, 3);
  assert.equal(st.hits, 1);
  assert.equal(st.files, 2);
  assert.ok(st.bytes <= size * 2 + 1);
  await new Promise((r) => setTimeout(r, 20));
  assert.equal(fs.existsSync(a!.fp), false, "LRU file removed from /tmp cache");
  // failure -> null, counted
  r2.store.clear();
  src.clear();
  assert.equal(await src.fetch("aircraft", h1), null);
  assert.equal(src.stats().errors, 1);
});

test("readWindow + cold tier: evicted hours replay from R2 through the same parse path; expiry and caps are honest", async () => {
  const base = tmp();
  const r2 = fakeR2();
  const d = dateAgo(25);
  writeHour(base, "aircraft", d, 10, rows(d, 10, ["abc123", "def456"]));
  writeHour(base, "aircraft", d, 11, rows(d, 11, ["abc123"]));
  const svc = svcFor(base, r2, { free: () => 0 });
  const s = (await svc.runTick())!;
  assert.equal(s.evicted.length, 1, "day evicted locally after verified upload");
  assert.equal(fs.readdirSync(path.join(base, "aircraft")).length, 0);

  const bbox = { w: -110, s: 35, e: -90, n: 45 };
  const from = hourSecOf(d, 10), to = hourSecOf(d, 12);
  const r = await readWindow({ bbox, fromSec: from, toSec: to, zoom: 10, baseDir: base, coldSource: svc.coldSource });
  assert.deepEqual(r.coldTier, { hoursFromR2: 2, hoursMissing: 0, hoursExpired: 0, hoursDeferred: 0 });
  assert.equal(r.hexes_seen, 2);
  assert.equal(r.hexes.find((h) => h.i === "abc123")!.points.length, 2);
  assert.equal(r.coverage.complete, true);

  // local-only (no cold source): the same window is an honest gap
  const local = await readWindow({ bbox, fromSec: from, toSec: to, zoom: 10, baseDir: base, coldSource: null });
  assert.equal(local.hexes_seen, 0);
  assert.equal(local.coldTier!.hoursMissing, 2);

  // per-request cap: 1 cold hour allowed -> newest hour served, older deferred
  const capped = await readWindow({ bbox, fromSec: from, toSec: to, zoom: 10, baseDir: base,
    coldSource: { ...svc.coldSource, locate: svc.coldSource.locate, fetch: svc.coldSource.fetch, maxHoursPerRequest: 1 } });
  assert.equal(capped.coldTier!.hoursFromR2, 1);
  assert.equal(capped.coldTier!.hoursDeferred, 1);
  assert.equal(capped.coverage.complete, false);

  // expired hours are refused, never fetched
  const old = dateAgo(35);
  const getsBefore = r2.log.filter((x) => x.method === "GET").length;
  const ex = await readWindow({ bbox, fromSec: hourSecOf(old, 0), toSec: hourSecOf(old, 3), zoom: 10, baseDir: base, coldSource: svc.coldSource });
  assert.equal(ex.coldTier!.hoursExpired, 3);
  assert.match(ex.note || "", /older than the rolling replay window/);
  assert.equal(r2.log.filter((x) => x.method === "GET").length, getsBefore);
});

test("readWindow: an aborted request stops scanning", async () => {
  const base = tmp();
  writeHour(base, "aircraft", dateAgo(2), 0, rows(dateAgo(2), 0));
  const ac = new AbortController();
  ac.abort();
  const r = await readWindow({ bbox: { w: -110, s: 35, e: -90, n: 45 }, fromSec: hourSecOf(dateAgo(2), 0),
    toSec: hourSecOf(dateAgo(2), 1), zoom: 10, baseDir: base, coldSource: null, signal: ac.signal });
  assert.equal(r.coverage.files_scanned, 0);
  assert.equal(r.coverage.complete, false);
});

// ── day rollup (datacoreArchive addition) ────────────────────────────────────
test("rollupDayAsync writes the same summary rollupOldDaysAsync would, without deleting; null on a corrupt file", async () => {
  const a = tmp(), b = tmp();
  const d = dateAgo(40);
  for (const base of [a, b]) {
    writeHour(base, "aircraft", d, 0, rows(d, 0, ["aaaaaa", "bbbbbb"]));
    writeHour(base, "aircraft", d, 1, rows(d, 1, ["aaaaaa"]), false);
  }
  const read = await rollupDayAsync("aircraft", d, a);
  assert.deepEqual(read, [`${d}-00.jsonl.gz`, `${d}-01.jsonl`]);
  assert.equal(fs.readdirSync(path.join(a, "aircraft")).length, 2, "raw files kept");
  assert.equal(await rollupOldDaysAsync(b, NOW), 1);
  assert.equal(
    zlib.gunzipSync(fs.readFileSync(path.join(a, "aircraft_tracks", `${d}.jsonl.gz`))).toString(),
    zlib.gunzipSync(fs.readFileSync(path.join(b, "aircraft_tracks", `${d}.jsonl.gz`))).toString(),
  );
  const c = tmp();
  fs.mkdirSync(path.join(c, "aircraft"), { recursive: true });
  fs.writeFileSync(path.join(c, "aircraft", `${d}-02.jsonl.gz`), Buffer.from([0x1f, 0x8b, 0x08, 0x00, 1, 2, 3]));
  assert.equal(await rollupDayAsync("aircraft", d, c), null);
  assert.deepEqual(await rollupDayAsync("aircraft", "2020-01-01", c), []);
});

// ── status endpoint + cost model ─────────────────────────────────────────────
test("GET /api/data/archive/offload-status: tiers, retention, estimates — and never a secret", async () => {
  const base = tmp();
  const r2 = fakeR2();
  for (let n = 3; n <= 7; n++) for (let h = 0; h < 2; h++) writeHour(base, "aircraft", dateAgo(n), h, rows(dateAgo(n), h));
  const handlers: Record<string, (req: any, res: any) => any> = {};
  const app: any = { get: (p: string, h: any) => { handlers[p] = h; } };
  const svc = registerArchiveOffloadRoutes(app, {
    startTimers: false, base, client: r2.client, env: ENV as any, nowMs: () => NOW,
    freeBytes: () => 5 * 1024 ** 3, log: () => {}, logError: () => {}, cacheDir: tmp("vt-cache-"),
    beforeEvictAircraft: async () => {},
  });
  try {
    await svc.runTick();
    svc._resetStatusCache();
    let body: any = null;
    await handlers["/api/data/archive/offload-status"]({}, { json: (j: any) => { body = j; }, status: () => ({ json: () => {} }) });
    assert.equal(body.configured, true);
    assert.equal(body.retentionDays, 30);
    assert.equal(body.oldestReplayableDate, dateAgo(30));
    assert.equal(body.hoursOffloaded, 10);
    assert.ok(body.bytesOffloaded > 0);
    assert.equal(body.tiers.r2.bytes, body.bytesOffloaded);
    assert.ok(body.tiers.local.bytes > 0);
    assert.equal(body.tiers.local.byKind.aircraft.files, 10);
    assert.ok(body.lastRun);
    assert.equal(body.lastError, null);
    assert.ok(body.estimatedMonthlyBytes > 0);
    assert.equal(typeof body.estimatedMonthlyCost.atCurrentGrowth.totalUsd, "number");
    assert.match(body.estimateBasis, /R2 ledger/);
    const s = JSON.stringify(body);
    for (const secret of [ENV.R2_SECRET_ACCESS_KEY, ENV.R2_ACCESS_KEY_ID + "/", ENV.R2_ACCOUNT_ID, ENV.R2_ARCHIVE_BUCKET]) {
      assert.equal(s.includes(secret), false, `status must not leak ${secret}`);
    }
  } finally {
    setColdHourSource(null);
  }
});

test("cost model: rolling 30 days plateaus inside the free tier at today's growth; 10x costs cents", () => {
  // measured 2026-09-28 (live /api/data/archive/stats): aircraft 367.6MB/639
  // hour files, vessels 1417.9MB/628 hour files -> ~68.0 MB/day gz combined
  const perDay = (367_580_397 / 639 + 1_417_944_795 / 628) * 24;
  const steady = perDay * 31;
  const ops = { classAOps: 49 * 30 + 180, classBOps: 49 * 30 + 14_400 };
  const now1 = estimateR2MonthlyCost({ storedBytes: steady, ...ops });
  assert.equal(now1.billableGb, 0);
  assert.equal(now1.totalUsd, 0);
  const x10 = estimateR2MonthlyCost({ storedBytes: steady * 10, ...ops });
  assert.ok(x10.billableGb > 10 && x10.billableGb < 13, `billable ${x10.billableGb}`);
  assert.ok(x10.totalUsd > 0.1 && x10.totalUsd < 0.25, `10x total ${x10.totalUsd}`);
  // aircraft-only 10x (the global sweep) stays free
  const air10 = estimateR2MonthlyCost({ storedBytes: ((367_580_397 / 639) * 24 * 10 + (1_417_944_795 / 628) * 24) * 31, ...ops });
  assert.equal(air10.totalUsd, 0);
});
