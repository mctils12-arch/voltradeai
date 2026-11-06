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
  defaultVolumeFiles,
} from "./streamsInventory";
import { SQUEEZE_SUMMARY_FILE } from "./squeezeGex";

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
  const inv = await buildStreamsInventory(emptyBase, MANIFEST_DIR_DEFAULT, Date.parse("2026-07-06T12:00:00Z"), {}, {}); // {}, {} = no bundled snapshots, no volume files: this test is about the volume archive alone
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

// [REPAIR 2026-10-11] a done-marker / index sentinel sorts AFTER date-named
// files ("b" > "2"), so the old "last name" read reported its ancient mtime
// as the stream's age — finrashortvol/griddemand/occvolume read "stale
// ~2300h" while holding two-day-old data. Pin: sentinels never set age.
test("sentinel files (backfill_done.json, _index) never masquerade as the newest data file", async () => {
  const mdir = tmp(); const base = tmp();
  writeManifest(mdir, "marked", "daily source, 4h poll");
  const NOW = Date.parse("2026-10-11T12:00:00Z");
  fs.mkdirSync(path.join(base, "marked"));
  const data = path.join(base, "marked", "2026-10-09.jsonl");
  fs.writeFileSync(data, JSON.stringify({ d: "2026-10-09" }) + "\n");
  fs.utimesSync(data, new Date(NOW - 3.6e6), new Date(NOW - 3.6e6));
  for (const sentinel of ["backfill_done.json", "_seen_ids_index.jsonl.gz"]) {
    const fp = path.join(base, "marked", sentinel);
    fs.writeFileSync(fp, "{}");
    const old = new Date(NOW - 97 * 24 * 3.6e6);
    fs.utimesSync(fp, old, old);
  }
  const inv = await buildStreamsInventory(base, mdir, NOW);
  const m = inv.streams.find((x) => x.stream === "marked")!;
  assert.equal(m.newest_file, "2026-10-09.jsonl");
  assert.equal(m.age_hours, 1);
  assert.equal(m.health, "live");
  assert.equal(m.files, 3, "sentinels still count toward files/bytes");
});

test("bundled git-versioned snapshot streams age from their embedded build time, not 'no-data / warming up' (2026-10-12)", async () => {
  const mdir = tmp(); const base = tmp();
  writeManifest(mdir, "snapweekly", "weekly");
  writeManifest(mdir, "snapfresh", "weekly");
  writeManifest(mdir, "snapnostamp", "weekly");
  writeManifest(mdir, "nosnap", "weekly"); // no dir, no snapshot → stays no-data
  const NOW = Date.parse("2026-10-12T00:00:00Z");
  const inv = await buildStreamsInventory(base, mdir, NOW, {
    snapweekly: { artifact: "datacore/x/a.json", built: "2026-07-05T10:00:00Z" },
    snapfresh: { artifact: "datacore/x/b.json", built: "2026-10-09T00:00:00Z" },
    snapnostamp: { artifact: "datacore/x/c.json", built: null },
  });
  const by = Object.fromEntries(inv.streams.map((s) => [s.stream, s]));
  assert.equal(by.snapweekly.health, "stale"); // 99d vs weekly threshold: served but stale, said so
  assert.equal(by.snapweekly.newest_file, "datacore/x/a.json");
  assert.equal(by.snapweekly.files, 1);
  assert.match(by.snapweekly.health_note, /bundled snapshot built 2026-07-05/);
  assert.equal(by.snapfresh.health, "recent"); // 72h within weekly cadence
  assert.equal(by.snapnostamp.health, "no-data");
  assert.equal(by.nosnap.health, "no-data");
  assert.equal(by.nosnap.files, 0);
});

test("snapshotStreams() stamps are real ISO times for every registered stream", async () => {
  const { snapshotStreams } = await import("./snapshotStreams");
  for (const [k, v] of Object.entries(snapshotStreams())) {
    assert.ok(v.built && !Number.isNaN(Date.parse(v.built)), `${k} has no embedded build time`);
    assert.ok(fs.existsSync(path.resolve(process.cwd(), v.artifact)), `${k} artifact path ${v.artifact} missing`);
  }
});

test("volume-resident keyed file (cot) ages by last successful write, not a no-data row", async () => {
  const base = tmp();
  const mdir = path.join(tmp(), "m");
  fs.mkdirSync(mdir, { recursive: true });
  for (const n of ["volweekly", "volmissing"]) {
    fs.writeFileSync(path.join(mdir, `${n}.json`), JSON.stringify({ stream: n, cadence: "weekly", source: "x" }));
  }
  const f = path.join(tmp(), "cot_archive.json");
  fs.writeFileSync(f, '{"ES":[{"report_date":"2026-10-06"}]}');
  const mt = Date.parse("2026-10-09T12:00:00Z");
  fs.utimesSync(f, mt / 1000, mt / 1000);
  const now = Date.parse("2026-10-12T12:00:00Z"); // 72h after the write
  const inv = await buildStreamsInventory(base, mdir, now, {}, {
    volweekly: f, volmissing: path.join(base, "does_not_exist.json"),
  });
  const by = Object.fromEntries(inv.streams.map((s) => [s.stream, s]));
  assert.equal(by.volweekly.health, "recent");
  assert.equal(by.volweekly.age_hours, 72);
  assert.equal(by.volweekly.files, 1);
  assert.equal(by.volweekly.newest_file, "cot_archive.json");
  assert.match(by.volweekly.health_note, /last successful write/);
  assert.equal(by.volmissing.health, "no-data"); // absent file stays honest
  assert.equal(by.volmissing.files, 0);
});

test("rare-event source with a fast poll is not stale after a quiet stretch (fdicfailures regression)", async () => {
  const mdir = tmp(); const base = tmp();
  writeManifest(mdir, "evt", "event-driven source (failures as they occur); 12h poll makes quiet stretches dedup no-ops");
  const NOW = Date.parse("2026-11-02T12:00:00Z");
  fs.mkdirSync(path.join(base, "evt"));
  const f = path.join(base, "evt", "x.jsonl");
  fs.writeFileSync(f, JSON.stringify({ ok: 1 }) + "\n");
  const t = new Date(NOW - 68 * 24 * 3.6e6);
  fs.utimesSync(f, t, t);
  const inv = await buildStreamsInventory(base, mdir, NOW);
  assert.equal(inv.streams[0].health, "recent");
  assert.equal(inv.streams[0].age_hours, 68 * 24, "raw age still reported");
});

test("peek:false in a manifest withholds every record, even from a surviving archive dir; squeeze_gex ages from its derived summary (2026-11-06)", async () => {
  // Before: /api/data/streams served the squeeze_gex archive's newest raw row
  // (date, SPX price, DIX, GEX) as latest_peek — SqueezeMetrics' values on a
  // public route. Their manifest now sets "peek": false.
  const real = JSON.parse(fs.readFileSync(path.join(MANIFEST_DIR_DEFAULT, "squeeze_gex.json"), "utf8"));
  assert.equal(real.peek, false, "the real squeeze_gex manifest withholds the peek");
  const base = tmp();
  const mdir = path.join(tmp(), "m");
  fs.mkdirSync(mdir, { recursive: true });
  fs.writeFileSync(path.join(mdir, "squeeze_gex.json"), JSON.stringify({
    stream: "squeeze_gex", cadence: "daily source, published after the close; 6h poll", source: "x",
    peek: false, peek_reason: "SqueezeMetrics' terms: no redistribution",
  }));
  const now = Date.parse("2026-11-06T12:00:00Z");
  // an old archive dir that survived (say the boot cleanup failed): still no record shown
  fs.mkdirSync(path.join(base, "squeeze_gex"));
  fs.writeFileSync(path.join(base, "squeeze_gex", "2026.jsonl"),
    JSON.stringify({ date: "2026-11-05", price: 7353.73, dix: 0.37856964877480676, gex: 1844749552.5200007 }) + "\n");
  let inv = await buildStreamsInventory(base, mdir, now, {}, {});
  let row = inv.streams.find((x) => x.stream === "squeeze_gex")!;
  assert.equal(row.latest_peek, null);
  assert.match(row.peek_note || "", /withheld by the manifest: SqueezeMetrics' terms/);
  assert.ok(!JSON.stringify(inv).includes("1844749552"), "no raw value anywhere in the inventory");
  // after the cleanup: no dir; the stream's volume file is the derived summary
  fs.rmSync(path.join(base, "squeeze_gex"), { recursive: true });
  const f = path.join(tmp(), SQUEEZE_SUMMARY_FILE);
  fs.writeFileSync(f, JSON.stringify({ v: 1, asOf: "2026-11-05", rows: 3903, rank252: 0.0916 }));
  fs.utimesSync(f, (now - 7 * 3.6e6) / 1000, (now - 7 * 3.6e6) / 1000);
  inv = await buildStreamsInventory(base, mdir, now, {}, { squeeze_gex: f });
  row = inv.streams.find((x) => x.stream === "squeeze_gex")!;
  assert.equal(row.health, "live", "a stream read every 6h is live, not 'no-data / warming up'");
  assert.equal(row.age_hours, 7);
  assert.equal(row.newest_file, SQUEEZE_SUMMARY_FILE);
  assert.equal(row.latest_peek, null);
  const prev = process.env.DATA_DIR;
  process.env.DATA_DIR = "/srv/v";
  try {
    assert.equal(defaultVolumeFiles().squeeze_gex, path.join("/srv/v", SQUEEZE_SUMMARY_FILE));
    assert.equal(defaultVolumeFiles().cot, path.join("/srv/v", "voltrade_cot_archive.json"));
  } finally {
    if (prev === undefined) delete process.env.DATA_DIR; else process.env.DATA_DIR = prev;
  }
});
