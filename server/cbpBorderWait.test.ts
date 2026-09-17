// CBP border-wait battery (BUILD ORDER 5 #5): live-shape parse with
// locale-independent keying, absent-lane honesty, change-only dedup.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  parseBorderWaits, fetchBorderWaits, archiveBorderWaits,
  gzipOldBorderWaitDays, refreshBorderWaits, latestBorderWaits,
  backfillBorderWaitsFromArchive, _resetBorderWaitCacheForTests,
} from "./cbpBorderWait";

// Trimmed from the LIVE response captured 2026-07-05 — including the
// Spanish localization our probe egress received (status strings are
// archived verbatim; parsing never matches on them).
const REC = {
  port_number: "230401", border: "Frontera mexicana", port_name: "Laredo",
  crossing_name: "Bridge I", date: "7/5/2026", time: "15:07:51", port_status: "Abierto",
  commercial_vehicle_lanes: {
    standard_lanes: { update_time: "En 2:00 pm CDT", operational_status: "demora", delay_minutes: "45", lanes_open: "3" },
    FAST_lanes: { update_time: "", operational_status: "N/A", delay_minutes: "", lanes_open: "" },
  },
  passenger_vehicle_lanes: {
    standard_lanes: { update_time: "", operational_status: "Carriles cerrados", delay_minutes: "", lanes_open: "" },
  },
};

test("parseBorderWaits: locale-independent fields parsed, localized strings verbatim", () => {
  const obs = parseBorderWaits([REC], "2026-07-05T20:10:00Z");
  assert.equal(obs.length, 3, "three lane classes flattened");
  const cv = obs.find((o) => o.lane === "commercial_standard")!;
  assert.equal(cv.delay_min, 45);
  assert.equal(cv.lanes_open, 3);
  assert.equal(cv.status, "demora", "status archived verbatim, never translated");
  assert.equal(cv.port_number, "230401");
  const fast = obs.find((o) => o.lane === "commercial_FAST")!;
  assert.equal(fast.delay_min, null, "empty delay is null, never zero");
  assert.deepEqual(parseBorderWaits(null, "x"), []);
  assert.deepEqual(parseBorderWaits([{ no_port: true }], "x"), []);
});

test("absent lane blocks contribute nothing (many crossings have no commercial lanes)", () => {
  const pedOnly = { port_number: "1", port_name: "X", crossing_name: "Y", border: "B",
                    passenger_vehicle_lanes: { standard_lanes: { operational_status: "open", delay_minutes: "5", lanes_open: "2" } } };
  const obs = parseBorderWaits([pedOnly], "x");
  assert.equal(obs.length, 1);
  assert.equal(obs[0].lane, "passenger_standard");
});

test("archive: change-only dedup — identical snapshot re-poll writes nothing; a delay change appends", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "bwt-"));
  const now = Date.parse("2026-07-05T20:10:00Z");
  assert.equal(archiveBorderWaits(parseBorderWaits([REC], "2026-07-05T20:10:00Z"), base, now), 3);
  assert.equal(archiveBorderWaits(parseBorderWaits([REC], "2026-07-05T21:10:00Z"), base, now), 0,
    "unchanged waits re-polled an hour later never re-archive");
  const worse = JSON.parse(JSON.stringify(REC));
  worse.commercial_vehicle_lanes.standard_lanes.delay_minutes = "90";
  assert.equal(archiveBorderWaits(parseBorderWaits([worse], "2026-07-05T22:10:00Z"), base, now), 1,
    "only the changed lane appends");
  assert.equal(gzipOldBorderWaitDays(base, now + 3 * 86400_000), 1);
});

test("refresh: transport error keeps the last snapshot", async () => {
  const ok = async () => ({ ok: true, status: 200, text: async () => JSON.stringify([REC]) });
  await refreshBorderWaits(ok as any, Date.parse("2026-07-05T20:10:00Z"));
  const hit = latestBorderWaits();
  assert.equal(hit!.obs.length, 3);
  const err = async () => ({ ok: false, status: 503, text: async () => "" });
  await refreshBorderWaits(err as any);
  assert.equal(latestBorderWaits(), hit);
  assert.equal(await fetchBorderWaits(err as any), null);
});

// Cold-cache-no-disk-backfill thread (research/open_questions.md — cbpBorderWait.ts
// was one of the 15 remaining vulnerable modules from the 2026-09-15 module audit).
// The archive is change-only dedup, not a full snapshot per file, so backfill
// must reconstruct "latest known value per (port_number, crossing_name, lane)"
// rather than replay every historical change as if it were current.

test("backfillBorderWaitsFromArchive: keeps only the latest observation per (port_number, crossing_name, lane) identity across the lookback window", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "vtbwt-backfill-"));
  const dir = path.join(root, "cbpborderwait");
  fs.mkdirSync(dir, { recursive: true });
  const now = Date.parse("2026-09-10T12:00:00Z");
  const day0 = new Date(now).toISOString().slice(0, 10);
  const day1 = new Date(now - 86400_000).toISOString().slice(0, 10);
  const older = { ...parseBorderWaits([REC], "2026-09-09T10:00:00Z")[0] };
  const newer = { ...parseBorderWaits([REC], "2026-09-10T11:00:00Z")[0], delay_min: 90 };
  const otherLane = parseBorderWaits([REC], "2026-09-10T09:00:00Z")[1]; // FAST lane, distinct identity
  fs.writeFileSync(path.join(dir, `${day1}.jsonl`), JSON.stringify(older) + "\n");
  fs.writeFileSync(path.join(dir, `${day0}.jsonl`), [newer, otherLane].map((o) => JSON.stringify(o)).join("\n") + "\n");
  const out = backfillBorderWaitsFromArchive(root, now, 3);
  assert.equal(out.length, 2, "two distinct identities (commercial_standard + commercial_FAST)");
  const std = out.find((o) => o.lane === "commercial_standard")!;
  assert.equal(std.delay_min, 90, "the newer-rt observation wins, not the older one");
  assert.ok(out.find((o) => o.lane === "commercial_FAST"), "the second identity is preserved");
});

test("backfillBorderWaitsFromArchive: respects the days window", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "vtbwt-backfill-window-"));
  const dir = path.join(root, "cbpborderwait");
  fs.mkdirSync(dir, { recursive: true });
  const now = Date.parse("2026-09-10T12:00:00Z");
  const tooOld = new Date(now - 10 * 86400_000).toISOString().slice(0, 10);
  fs.writeFileSync(path.join(dir, `${tooOld}.jsonl`), JSON.stringify(parseBorderWaits([REC], "x")[0]) + "\n");
  assert.deepEqual(backfillBorderWaitsFromArchive(root, now, 3), [], "a day 10 days back is outside the default 3-day window");
});

test("refreshBorderWaits: cold cache backfills from the on-disk archive when the live poll throws", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vtbwt-coldcache-"));
  const prevDataDir = process.env.DATA_DIR;
  process.env.DATA_DIR = base;
  _resetBorderWaitCacheForTests();
  try {
    const dir = path.join(base, "datacore_archive", "cbpborderwait");
    fs.mkdirSync(dir, { recursive: true });
    const today = new Date().toISOString().slice(0, 10);
    fs.writeFileSync(path.join(dir, `${today}.jsonl`), JSON.stringify(parseBorderWaits([REC], "2026-09-10T10:00:00Z")[0]) + "\n");
    assert.equal(latestBorderWaits(), null, "cache must still be cold going into this cycle");
    const throwingFetch = (async () => { throw new Error("bwt.cbp.gov unreachable"); }) as any;
    await refreshBorderWaits(throwingFetch);
    const cached = latestBorderWaits();
    assert.ok(cached, "cache must be populated, not left null, despite the live poll throwing");
    assert.equal(cached!.obs.length, 1);
    assert.equal(cached!.obs[0].lane, "commercial_standard", "backfilled from the archived observation, not fabricated");
  } finally {
    _resetBorderWaitCacheForTests();
    if (prevDataDir === undefined) delete process.env.DATA_DIR; else process.env.DATA_DIR = prevDataDir;
  }
});

test("refreshBorderWaits: an empty-but-non-throwing live poll also backfills from disk when the cache is cold", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vtbwt-coldcache-empty-"));
  const prevDataDir = process.env.DATA_DIR;
  process.env.DATA_DIR = base;
  _resetBorderWaitCacheForTests();
  try {
    const dir = path.join(base, "datacore_archive", "cbpborderwait");
    fs.mkdirSync(dir, { recursive: true });
    const today = new Date().toISOString().slice(0, 10);
    fs.writeFileSync(path.join(dir, `${today}.jsonl`), JSON.stringify(parseBorderWaits([REC], "2026-09-10T10:00:00Z")[0]) + "\n");
    const emptyFetch = async () => ({ ok: true, status: 200, text: async () => JSON.stringify([]) });
    await refreshBorderWaits(emptyFetch as any);
    const cached = latestBorderWaits();
    assert.ok(cached, "cache must be populated from disk, not left null, on an empty-but-successful poll");
    assert.equal(cached!.obs.length, 1);
  } finally {
    _resetBorderWaitCacheForTests();
    if (prevDataDir === undefined) delete process.env.DATA_DIR; else process.env.DATA_DIR = prevDataDir;
  }
});

test("refreshBorderWaits: a transient empty poll never overwrites an already-good cache with a stale archive read", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vtbwt-warmcache-"));
  const prevDataDir = process.env.DATA_DIR;
  process.env.DATA_DIR = base;
  _resetBorderWaitCacheForTests();
  try {
    const goodFetch = async () => ({ ok: true, status: 200, text: async () => JSON.stringify([REC]) });
    await refreshBorderWaits(goodFetch as any, Date.parse("2026-09-10T10:00:00Z"));
    assert.equal(latestBorderWaits()!.obs.length, 3, "warm the cache first");

    const emptyFetch = async () => ({ ok: true, status: 200, text: async () => JSON.stringify([]) });
    await refreshBorderWaits(emptyFetch as any);
    assert.equal(latestBorderWaits()!.obs.length, 3, "an empty poll must not blank out an already-warm cache");
  } finally {
    _resetBorderWaitCacheForTests();
    if (prevDataDir === undefined) delete process.env.DATA_DIR; else process.env.DATA_DIR = prevDataDir;
  }
});
