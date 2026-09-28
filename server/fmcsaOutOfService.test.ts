// FMCSA Out-of-Service orders battery (EDGE DOCTRINE #1, 2026-09-28):
// live-shape parse, change-only dedup (including a rescission update),
// cold-cache-no-disk-backfill.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  parseOosOrders, fetchOosOrders, archiveOosOrders,
  gzipOldOosDays, refreshOosOrders, latestOosOrders,
  backfillOosOrdersFromArchive, _resetOosCacheForTests,
} from "./fmcsaOutOfService";

// Trimmed from the LIVE response captured 2026-09-28.
const REC = {
  dot_number: "4539029", legal_name: "NO NAME TRUCKING & REPAIR LLC",
  oos_date: "2026-09-26", oos_reason: "New Entrant Revoked - Refusal of Audit/No Contact",
  status: "ACTIVE", rescind_date: null,
};

test("parseOosOrders: typed rows, malformed/missing-key rows dropped", () => {
  const obs = parseOosOrders([REC], "2026-09-28T11:00:00Z");
  assert.equal(obs.length, 1);
  assert.equal(obs[0].dot_number, "4539029");
  assert.equal(obs[0].oos_date, "2026-09-26");
  assert.equal(obs[0].status, "ACTIVE");
  assert.equal(obs[0].rescind_date, null);
  assert.deepEqual(parseOosOrders(null, "x"), []);
  assert.deepEqual(parseOosOrders([{ legal_name: "no dot number" }], "x"), []);
  assert.deepEqual(parseOosOrders([{ dot_number: "1", legal_name: "no oos_date" }], "x"), []);
});

test("parseOosOrders: rescind_date present is carried through, truncated to a date", () => {
  const rescinded = { ...REC, status: "INACTIVE", rescind_date: "2026-09-27T00:00:00.000" };
  const obs = parseOosOrders([rescinded], "x");
  assert.equal(obs[0].rescind_date, "2026-09-27");
  assert.equal(obs[0].status, "INACTIVE");
});

test("archive: change-only dedup — identical re-poll writes nothing; a rescission (status/rescind_date change) appends", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "oos-"));
  const now = Date.parse("2026-09-28T11:00:00Z");
  assert.equal(archiveOosOrders(parseOosOrders([REC], "2026-09-28T11:00:00Z"), base, now), 1);
  assert.equal(archiveOosOrders(parseOosOrders([REC], "2026-09-28T17:00:00Z"), base, now), 0,
    "an unchanged order re-polled the same day never re-archives");
  const rescinded = { ...REC, status: "INACTIVE", rescind_date: "2026-09-29" };
  assert.equal(archiveOosOrders(parseOosOrders([rescinded], "2026-09-29T11:00:00Z"), base, now), 1,
    "a later rescission is a real state change and appends");
  assert.equal(gzipOldOosDays(base, now + 3 * 86400_000), 1);
});

test("refresh: transport error keeps the last snapshot", async () => {
  const ok = async () => ({ ok: true, status: 200, text: async () => JSON.stringify([REC]) });
  await refreshOosOrders(ok as any, Date.parse("2026-09-28T11:00:00Z"));
  const hit = latestOosOrders();
  assert.equal(hit!.obs.length, 1);
  const err = async () => ({ ok: false, status: 503, text: async () => "" });
  await refreshOosOrders(err as any);
  assert.equal(latestOosOrders(), hit);
  assert.equal(await fetchOosOrders(err as any), null);
});

// Cold-cache-no-disk-backfill thread (research/open_questions.md). The
// archive is change-only dedup, not a full snapshot per file, so backfill
// must reconstruct "latest known state per dot_number+oos_date identity"
// rather than replay every historical change as if it were current.

test("backfillOosOrdersFromArchive: keeps only the latest observation per dot_number+oos_date identity across the lookback window", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "oos-backfill-"));
  const dir = path.join(root, "fmcsaoos");
  fs.mkdirSync(dir, { recursive: true });
  const now = Date.parse("2026-09-28T12:00:00Z");
  const day0 = new Date(now).toISOString().slice(0, 10);
  const day1 = new Date(now - 86400_000).toISOString().slice(0, 10);
  const older = { ...parseOosOrders([REC], "2026-09-27T10:00:00Z")[0] };
  const newer = { ...parseOosOrders([REC], "2026-09-28T11:00:00Z")[0], status: "INACTIVE", rescind_date: "2026-09-28" };
  const other = parseOosOrders([{ ...REC, dot_number: "999", oos_date: "2026-09-20" }], "2026-09-28T09:00:00Z")[0];
  fs.writeFileSync(path.join(dir, `${day1}.jsonl`), JSON.stringify(older) + "\n");
  fs.writeFileSync(path.join(dir, `${day0}.jsonl`), [newer, other].map((o) => JSON.stringify(o)).join("\n") + "\n");
  const out = backfillOosOrdersFromArchive(root, now, 3);
  assert.equal(out.length, 2, "two distinct dot_number+oos_date identities");
  const rec = out.find((o) => o.dot_number === "4539029")!;
  assert.equal(rec.status, "INACTIVE", "the newer-rt observation wins, not the older one");
  assert.ok(out.find((o) => o.dot_number === "999"), "the second identity is preserved");
});

test("backfillOosOrdersFromArchive: respects the days window", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "oos-backfill-window-"));
  const dir = path.join(root, "fmcsaoos");
  fs.mkdirSync(dir, { recursive: true });
  const now = Date.parse("2026-09-28T12:00:00Z");
  const tooOld = new Date(now - 60 * 86400_000).toISOString().slice(0, 10);
  fs.writeFileSync(path.join(dir, `${tooOld}.jsonl`), JSON.stringify(parseOosOrders([REC], "x")[0]) + "\n");
  assert.deepEqual(backfillOosOrdersFromArchive(root, now, 3), [], "a day 60 days back is outside a 3-day window");
});

test("refreshOosOrders: cold cache backfills from the on-disk archive when the live poll throws", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "oos-coldcache-"));
  const prevDataDir = process.env.DATA_DIR;
  process.env.DATA_DIR = base;
  _resetOosCacheForTests();
  try {
    const dir = path.join(base, "datacore_archive", "fmcsaoos");
    fs.mkdirSync(dir, { recursive: true });
    const today = new Date().toISOString().slice(0, 10);
    fs.writeFileSync(path.join(dir, `${today}.jsonl`), JSON.stringify(parseOosOrders([REC], "2026-09-28T10:00:00Z")[0]) + "\n");
    assert.equal(latestOosOrders(), null, "cache must still be cold going into this cycle");
    const throwingFetch = (async () => { throw new Error("data.transportation.gov unreachable"); }) as any;
    await refreshOosOrders(throwingFetch);
    const cached = latestOosOrders();
    assert.ok(cached, "cache must be populated, not left null, despite the live poll throwing");
    assert.equal(cached!.obs.length, 1);
    assert.equal(cached!.obs[0].dot_number, "4539029", "backfilled from the archived observation, not fabricated");
  } finally {
    _resetOosCacheForTests();
    if (prevDataDir === undefined) delete process.env.DATA_DIR; else process.env.DATA_DIR = prevDataDir;
  }
});

test("refreshOosOrders: an empty-but-non-throwing live poll also backfills from disk when the cache is cold", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "oos-coldcache-empty-"));
  const prevDataDir = process.env.DATA_DIR;
  process.env.DATA_DIR = base;
  _resetOosCacheForTests();
  try {
    const dir = path.join(base, "datacore_archive", "fmcsaoos");
    fs.mkdirSync(dir, { recursive: true });
    const today = new Date().toISOString().slice(0, 10);
    fs.writeFileSync(path.join(dir, `${today}.jsonl`), JSON.stringify(parseOosOrders([REC], "2026-09-28T10:00:00Z")[0]) + "\n");
    const emptyFetch = async () => ({ ok: true, status: 200, text: async () => JSON.stringify([]) });
    await refreshOosOrders(emptyFetch as any);
    const cached = latestOosOrders();
    assert.ok(cached, "cache must be populated from disk, not left null, on an empty-but-successful poll");
    assert.equal(cached!.obs.length, 1);
  } finally {
    _resetOosCacheForTests();
    if (prevDataDir === undefined) delete process.env.DATA_DIR; else process.env.DATA_DIR = prevDataDir;
  }
});

test("refreshOosOrders: a transient empty poll never overwrites an already-good cache with a stale archive read", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "oos-warmcache-"));
  const prevDataDir = process.env.DATA_DIR;
  process.env.DATA_DIR = base;
  _resetOosCacheForTests();
  try {
    const goodFetch = async () => ({ ok: true, status: 200, text: async () => JSON.stringify([REC]) });
    await refreshOosOrders(goodFetch as any, Date.parse("2026-09-28T10:00:00Z"));
    assert.equal(latestOosOrders()!.obs.length, 1, "warm the cache first");

    const emptyFetch = async () => ({ ok: true, status: 200, text: async () => JSON.stringify([]) });
    await refreshOosOrders(emptyFetch as any);
    assert.equal(latestOosOrders()!.obs.length, 1, "an empty poll must not blank out an already-warm cache");
  } finally {
    _resetOosCacheForTests();
    if (prevDataDir === undefined) delete process.env.DATA_DIR; else process.env.DATA_DIR = prevDataDir;
  }
});
