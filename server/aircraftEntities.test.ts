// aircraft entity spine v1 battery (BUILD ORDER 2 #1, 2026-07-05):
// archive fold semantics, gz-aware enumeration, cache staleness contract,
// and graceful degradation before the spine artifact exists.
import { test, beforeEach } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import {
  foldLine, distinctArchivedAircraft, archivedAircraftCached,
  loadEntitySpine, entityForHex, _resetAircraftEntityCache,
  type HexSummary,
} from "./aircraftEntities";

beforeEach(() => _resetAircraftEntityCache());

function tmpArchive(): string {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "spine-"));
  fs.mkdirSync(path.join(base, "aircraft"), { recursive: true });
  return base;
}

const line = (o: any) => JSON.stringify(o);

test("foldLine: first/last seen, counts, callsign cap, latest type wins", () => {
  const acc = new Map<string, HexSummary>();
  foldLine(acc, line({ t: 100, i: "A1433A", c: "SWA1" }));
  foldLine(acc, line({ t: 50, i: "a1433a", c: "SWA2", ty: "B737" }));
  foldLine(acc, line({ t: 200, i: "a1433a", ty: "B38M" }));
  for (let k = 0; k < 10; k++) foldLine(acc, line({ t: 300 + k, i: "a1433a", c: `X${k}` }));
  const s = acc.get("a1433a")!;
  assert.equal(acc.size, 1, "hex case-folds to one entity");
  assert.equal(s.n, 13);
  assert.equal(s.f, 50);
  assert.equal(s.l, 309);
  assert.equal(s.c!.length, 5, "callsigns capped");
  assert.equal(s.ty, "B38M", "type from the LATEST record, not the last line read");
  // garbage never throws, never counts
  foldLine(acc, "not json");
  foldLine(acc, line({ t: "x", i: "a1433a" }));
  foldLine(acc, line({ t: 1 }));
  assert.equal(acc.get("a1433a")!.n, 13);
});

test("distinctArchivedAircraft reads plain and gzipped day files", async () => {
  const base = tmpArchive();
  fs.writeFileSync(path.join(base, "aircraft", "2026-07-01-00.jsonl"),
    line({ t: 10, i: "abc123", c: "N1" }) + "\n" + line({ t: 20, i: "def456" }) + "\n");
  fs.writeFileSync(path.join(base, "aircraft", "2026-06-30-00.jsonl.gz"),
    zlib.gzipSync(line({ t: 5, i: "abc123", ty: "C172" }) + "\n"));
  fs.writeFileSync(path.join(base, "aircraft", "ignored.txt"), "nope\n");
  const hexes = await distinctArchivedAircraft(base);
  assert.equal(hexes.length, 2);
  const abc = hexes.find((h) => h.i === "abc123")!;
  assert.equal(abc.n, 2, "gz + plain both counted");
  assert.equal(abc.f, 5, "first seen crosses file boundary");
  assert.equal(abc.l, 10);
  // most points first (join-priority ordering for the build script)
  assert.equal(hexes[0].i, "abc123");
});

// REPAIR (found 2026-07-22, same root cause + fix as datacoreArchive.ts's
// streamJsonlLines): readline.Interface re-emits a piped-in stream's error
// on ITSELF too, independent of the stream.on("error", ...) guard here —
// unlistened, a truncated/corrupt .gz crashed the WHOLE PROCESS. See
// datacoreArchive.test.ts for the full writeup + minimal repro.
test("distinctArchivedAircraft resolves (never crashes the process) on a truncated/corrupt gzip file", async () => {
  const base = tmpArchive();
  const good = zlib.gzipSync(line({ t: 1, i: "abc123" }) + "\n");
  fs.writeFileSync(path.join(base, "aircraft", "2026-07-01-00.jsonl.gz"), good.subarray(0, good.length - 4));
  // The point is that the scan RESOLVES at all — if the missing rl.on("error",
  // ...) guard regresses, this either hangs or throws an unhandled 'error'
  // event that kills the whole node:test process. Whatever lines the
  // decompressor got through before the truncation error is incidental
  // (gzip streams incrementally; a short file may still yield a record).
  await distinctArchivedAircraft(base);
});

test("distinctArchivedAircraft: missing dir is empty, never throws", async () => {
  assert.deepEqual(await distinctArchivedAircraft("/nonexistent/base"), []);
});

test("archivedAircraftCached: scans once per TTL window", async () => {
  const base = tmpArchive();
  fs.writeFileSync(path.join(base, "aircraft", "2026-07-01-00.jsonl"), line({ t: 1, i: "aaa111" }) + "\n");
  const first = await archivedAircraftCached(base);
  assert.equal(first.hexes.length, 1);
  // new data arrives; within TTL the cached view is served (stale-ok contract)
  fs.writeFileSync(path.join(base, "aircraft", "2026-07-01-01.jsonl"), line({ t: 2, i: "bbb222" }) + "\n");
  const second = await archivedAircraftCached(base);
  assert.equal(second.hexes.length, 1, "TTL window serves the cached scan");
  assert.equal(second.as_of, first.as_of);
});

test("RATCHET [REPAIR 2026-07-05]: spine serves WITHOUT datacore/ on disk (runtime Docker image never ships it)", () => {
  // The break: loadEntitySpine read datacore/aircraft/entity_spine.json from
  // cwd at runtime — fine locally and in CI (repo = cwd), null FOREVER on
  // Railway (frozen Dockerfile copies dist/ but not datacore/). The fix
  // bundles the artifact via static import. chdir simulates the image.
  _resetAircraftEntityCache();
  const prev = process.cwd();
  process.chdir(os.tmpdir());
  try {
    const s = loadEntitySpine();
    assert.ok(s && Object.keys(s).length > 10_000, "bundled spine must serve with no datacore/ under cwd");
    assert.equal(entityForHex("a6faee")!.owner, "SPARTAN EDUCATION LLC");
  } finally {
    process.chdir(prev);
    _resetAircraftEntityCache();
  }
});

test("spine: missing artifact degrades to null, real artifact serves per-hex", () => {
  assert.equal(loadEntitySpine("/nonexistent/spine.json"), null);
  assert.equal(entityForHex("a1433a", "/nonexistent/spine.json"), null);
  _resetAircraftEntityCache();
  const fp = path.join(os.tmpdir(), `spine-${process.pid}.json`);
  fs.writeFileSync(fp, JSON.stringify({
    entities: {
      a1433a: {
        n_number: "N1801U", owner: "SOUTHWEST AIRLINES CO", registrant_type: "corporation",
        mfr: "BOEING", model: "737-8", year_mfr: 2022, cert_issue: "2022-07-12",
        expiration: "2029-07-31",
        evidence: { source: "FAA Releasable Aircraft DB (US government work, public domain)", retrieved: "2026-07-05", match: "exact Mode S hex" },
      },
    },
  }));
  const e = entityForHex("A1433A", fp);
  assert.ok(e, "case-insensitive lookup");
  assert.equal(e!.owner, "SOUTHWEST AIRLINES CO");
  assert.equal(e!.evidence.match, "exact Mode S hex");
  assert.equal(entityForHex("deadbe", fp), null, "unknown hex is null, never guessed");
  fs.unlinkSync(fp);
});
