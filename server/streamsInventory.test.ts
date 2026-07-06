// streamsInventory tests — Phase 4 aggregate + the Phase 5
// INVENTORY-COVERAGE RATCHET: every manifested stream must appear in the
// inventory the /data Streams tab renders. The aggregator enumerates the
// manifest dir at runtime, so coverage is mechanical — these tests pin
// that property so a future rewrite cannot regress to a hardcoded list.
import { test, beforeEach } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import path from "path";
import os from "os";
import zlib from "zlib";
import {
  buildStreamsInventory,
  getStreamsInventoryCached,
  _resetStreamsInventoryForTests,
  MANIFEST_DIR_DEFAULT,
} from "./streamsInventory";

function tmp(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), "sinv-"));
}

function writeManifest(dir: string, name: string, cadence: string): void {
  fs.writeFileSync(path.join(dir, `${name}.json`), JSON.stringify({
    stream: name, storage: "s", written_by: "w", source: "src-" + name,
    license: "lic", attribution: "attr-" + name, schema_version: 1,
    field_map: { a: "1", b: "2", c: "3" }, confidence_model: "hypo-" + name,
    geo_fields: [], entity_key: "k", started: "2026-07-01", cadence,
  }));
}

beforeEach(() => _resetStreamsInventoryForTests());

test("joins manifest metadata with archive scan; plain + gz peeks; missing dir is no-data (never hidden)", async () => {
  const mdir = tmp(); const base = tmp();
  writeManifest(mdir, "alpha", "daily source, 4h poll");
  writeManifest(mdir, "beta", "weekly");
  writeManifest(mdir, "gamma", "daily"); // no archive dir on purpose

  const NOW = Date.parse("2026-07-06T12:00:00Z");
  // alpha: plain jsonl, fresh (2h old)
  fs.mkdirSync(path.join(base, "alpha"));
  const af = path.join(base, "alpha", "2026-07-06.jsonl");
  fs.writeFileSync(af, JSON.stringify({ d: "2026-07-05", v: 1 }) + "\n" + JSON.stringify({ d: "2026-07-06", v: 2 }) + "\n");
  fs.utimesSync(af, new Date(NOW - 2 * 3.6e6), new Date(NOW - 2 * 3.6e6));
  // beta: gz file, 5 days old (within weekly cadence)
  fs.mkdirSync(path.join(base, "beta"));
  const bf = path.join(base, "beta", "2026-07-01.jsonl.gz");
  fs.writeFileSync(bf, zlib.gzipSync(JSON.stringify({ week: "2026-06-28", rows: 10 }) + "\n"));
  fs.utimesSync(bf, new Date(NOW - 5 * 24 * 3.6e6), new Date(NOW - 5 * 24 * 3.6e6));

  const inv = await buildStreamsInventory(base, mdir, NOW);
  assert.equal(inv.count, 3);
  const by = Object.fromEntries(inv.streams.map((s) => [s.stream, s]));

  assert.equal(by.alpha.health, "live");
  assert.equal(by.alpha.files, 1);
  assert.equal(by.alpha.age_hours, 2);
  assert.equal(by.alpha.records_newest_file, 2);
  assert.ok(by.alpha.latest_peek!.includes('"v":2'), "peek must be the LAST record");
  assert.equal(by.alpha.source, "src-alpha");
  assert.equal(by.alpha.hypothesis, "hypo-alpha");
  assert.equal(by.alpha.started, "2026-07-01");

  assert.equal(by.beta.health, "recent", "5d-old weekly stream is healthy, not stale");
  assert.ok(by.beta.latest_peek!.includes('"rows":10'), "gz peek must decompress");

  assert.equal(by.gamma.health, "no-data");
  assert.equal(by.gamma.files, 0);
  assert.equal(by.gamma.latest_peek, null);
  assert.ok(by.gamma.health_note.includes("key-gated") || by.gamma.health_note.includes("warming"));
});

test("health derivation is cadence-aware and age is always reported raw", async () => {
  const mdir = tmp(); const base = tmp();
  writeManifest(mdir, "hourly", "live positions, poll every 5 minutes");
  writeManifest(mdir, "monthly", "monthly release");
  const NOW = Date.parse("2026-07-06T12:00:00Z");
  for (const [name, ageDays] of [["hourly", 3], ["monthly", 20]] as const) {
    fs.mkdirSync(path.join(base, name));
    const f = path.join(base, name, "x.jsonl");
    fs.writeFileSync(f, JSON.stringify({ ok: 1 }) + "\n");
    const t = new Date(NOW - ageDays * 24 * 3.6e6);
    fs.utimesSync(f, t, t);
  }
  const inv = await buildStreamsInventory(base, mdir, NOW);
  const by = Object.fromEntries(inv.streams.map((s) => [s.stream, s]));
  assert.equal(by.hourly.health, "stale", "3d-quiet minute-cadence stream is stale");
  assert.equal(by.hourly.age_hours, 72);
  assert.equal(by.monthly.health, "recent", "20d-quiet monthly stream is fine");
});

test("peek honesty: oversized file skipped with note; garbage tail flagged, never fabricated", async () => {
  const mdir = tmp(); const base = tmp();
  writeManifest(mdir, "big", "daily");
  writeManifest(mdir, "torn", "daily");
  const NOW = Date.parse("2026-07-06T12:00:00Z");
  fs.mkdirSync(path.join(base, "big"));
  const bigf = path.join(base, "big", "big.jsonl");
  const fd = fs.openSync(bigf, "w");
  fs.ftruncateSync(fd, 9 * 1024 * 1024); // sparse 9MB > 8MB cap
  fs.closeSync(fd);
  fs.utimesSync(bigf, new Date(NOW), new Date(NOW));
  fs.mkdirSync(path.join(base, "torn"));
  fs.writeFileSync(path.join(base, "torn", "t.jsonl"), '{"ok":1}\n{"truncated_mid_wri');

  const inv = await buildStreamsInventory(base, mdir, NOW);
  const by = Object.fromEntries(inv.streams.map((s) => [s.stream, s]));
  assert.equal(by.big.latest_peek, null);
  assert.equal(by.big.records_newest_file, null, "no fabricated count when the file was not read");
  assert.ok(by.big.peek_note!.includes("peek skipped"), by.big.peek_note!);
  assert.equal(by.torn.latest_peek, null);
  assert.equal(by.torn.records_newest_file, 2, "line count still honest even when tail is torn");
  assert.ok(by.torn.peek_note!.includes("not valid JSON"));
});

test("INVENTORY-COVERAGE RATCHET: every real manifest appears in the inventory, even with an empty archive", async () => {
  const emptyBase = tmp();
  const manifestFiles = fs.readdirSync(MANIFEST_DIR_DEFAULT).filter((f) => f.endsWith(".json"));
  const inv = await buildStreamsInventory(emptyBase, MANIFEST_DIR_DEFAULT, Date.parse("2026-07-06T12:00:00Z"));
  assert.equal(inv.count, manifestFiles.length,
    "inventory must enumerate the manifest dir — a stream missing here is a stream invisible on /data");
  const names = new Set(inv.streams.map((s) => s.stream));
  for (const f of manifestFiles) {
    const n = f.replace(/\.json$/, "");
    assert.ok(names.has(n), `manifested stream '${n}' missing from inventory`);
    const row = inv.streams.find((s) => s.stream === n)!;
    assert.equal(row.health, "no-data", `empty archive must yield an honest no-data row for '${n}'`);
  }
  assert.ok(manifestFiles.length >= 41, `manifest census shrank? found ${manifestFiles.length}`);
});

test("cache starts null (route reports warming) and reset helper clears state", () => {
  assert.equal(getStreamsInventoryCached(), null);
});
