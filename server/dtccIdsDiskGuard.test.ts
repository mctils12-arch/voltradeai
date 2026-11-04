/**
 * VOLUME GUARD for the A1 dedup DB (2026-11-04). On its first live boot the
 * 178 MB `_seen_ids.db` moved the Railway volume's free space from ~1,146 MB to
 * 967.9 MB, under globalScopes' MIN_FREE_BYTES (1 GiB) — the line below which
 * the global-scope aircraft archiver skips writes. The DB is derived data, so
 * it must never spend the archives' headroom. These tests pin: removal under
 * the guard, refusal to create without margin, hysteresis, fail-open on an
 * unreadable reading, recovery, and that the archive output is identical to
 * memory mode whenever the guard refuses.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import {
  REQUIRED_COLUMNS, parseHeader, parseDtccLine, archiveNewRows, _resetDtccForTests,
  dtccIdsStats, _setDtccFreeBytesReaderForTests, DTCC_IDS_DISK_MARGIN_BYTES, DTCC_IDS_EST_BYTES_PER_ID,
} from "./dtccSwaps";
import { MIN_FREE_BYTES } from "./globalScopes";

const IDS_ENV = "VOLTRADE_DTCC_IDS_STORE";
const GIB = 2 ** 30;
const HEADER_COLS = [...REQUIRED_COLUMNS, "Some Future Column"];
const HEADER_LINE = HEADER_COLS.map((c) => `"${c}"`).join(",");

function row(diss: string, id: string, src: string, name: string, notional = "28,000,000", ccy = "USD") {
  // same builder as dtccSwaps.test.ts
  const byName: Record<string, string> = {
    "Dissemination Identifier": diss,
    "Original Dissemination Identifier": diss,
    "Action type": "NEWT",
    "Event timestamp": "2026-08-21T12:33:36Z",
    "Effective Date": "2026-08-21",
    "Asset Class": "EQ",
    "Notional amount-Leg 1": notional,
    "Notional currency-Leg 1": ccy,
    "Underlier ID-Leg 1": id,
    "Underlier ID source-Leg 1": src,
    "UPI Underlier Name": name,
    "Some Future Column": "x",
  };
  return HEADER_COLS.map((c) => `"${byName[c]}"`).join(",");
}

function fourRows() {
  const idx = parseHeader(HEADER_LINE);
  const n = HEADER_COLS.length;
  return [
    parseDtccLine(row("1", "US0378331005", "ISIN", "APPLE INC", "1,000,000"), idx, n),
    parseDtccLine(row("2", "US88160R1014", "ISIN", "TESLA INC", "50,000,000"), idx, n),
    parseDtccLine(row("3", "037833100", "CUSIP", "APPLE INC", ""), idx, n),
    parseDtccLine(row("4", "88160R101", "CUSIP", "TESLA INC", "10,000,000"), idx, n),
  ].filter((r): r is NonNullable<typeof r> => r !== null);
}

function seededDir(tag: string): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), `dtcc-guard-${tag}-`));
  fs.mkdirSync(path.join(dir, "dtccswaps"), { recursive: true });
  fs.writeFileSync(path.join(dir, "dtccswaps", "_seen_ids_index.jsonl.gz"), zlib.gzipSync("1\n2\n"));
  return dir;
}
const dbPath = (dir: string) => path.join(dir, "dtccswaps", "_seen_ids.db");

function withGuard<T>(mode: "dual" | "sqlite" | "memory", free: () => number | null, fn: () => T): { out: T; errors: string[] } {
  const prev = process.env[IDS_ENV];
  process.env[IDS_ENV] = mode;
  _resetDtccForTests();
  _setDtccFreeBytesReaderForTests(free);
  const errors: string[] = [];
  const orig = console.error;
  console.error = (...a: unknown[]) => { errors.push(a.map(String).join(" ")); };
  try {
    return { out: fn(), errors };
  } finally {
    console.error = orig;
    _setDtccFreeBytesReaderForTests(null);
    _resetDtccForTests();
    if (prev === undefined) delete process.env[IDS_ENV]; else process.env[IDS_ENV] = prev;
  }
}

test("the guard line is the archivers' own: globalScopes MIN_FREE_BYTES (1 GiB) + a 256 MB creation margin", () => {
  assert.equal(MIN_FREE_BYTES, GIB);
  assert.equal(DTCC_IDS_DISK_MARGIN_BYTES, 256 * 2 ** 20);
  assert.ok(DTCC_IDS_EST_BYTES_PER_ID >= 25, "estimate must not undercount the measured 24.7 B/id");
});

test("an existing DB is REMOVED when free < the guard, the call runs the memory path, and the archive output is identical to memory mode", () => {
  // build a DB with plenty of room first
  const dir = seededDir("remove");
  withGuard("dual", () => 64 * GIB, () => archiveNewRows(fourRows(), "2026-11-03", dir));
  assert.ok(fs.existsSync(dbPath(dir)), "dual mode created the DB");
  const dayBefore = fs.readFileSync(path.join(dir, "dtccswaps", "2026-11-03.jsonl.gz"));
  // now the volume is short: 967.9 MB free (the live 2026-11-04 reading)
  const { out, errors } = withGuard("dual", () => 967_929_856, () => {
    const r = archiveNewRows(fourRows(), "2026-11-03", dir);
    return { r, st: dtccIdsStats() };
  });
  assert.ok(!fs.existsSync(dbPath(dir)), "the derived DB is removed");
  assert.ok(!fs.existsSync(dbPath(dir) + "-wal") && !fs.existsSync(dbPath(dir) + "-shm"), "with its WAL/SHM");
  assert.deepEqual(out.r, { newRows: 0, seenTotal: 4 }, "memory path: same verdict as before (nothing new)");
  assert.equal(out.st.diskGuard.active, true);
  assert.equal(out.st.dbCount, null, "no DB open");
  assert.equal(out.st.memoryCount, 4, "the Set serves the decision");
  assert.equal(errors.filter((e) => e.includes("DTCC-IDS-DISK-GUARD:")).length, 1, "said once, loudly");
  assert.ok(fs.readFileSync(path.join(dir, "dtccswaps", "2026-11-03.jsonl.gz")).equals(dayBefore), "archive untouched");
  fs.rmSync(dir, { recursive: true, force: true });
});

test("no DB is CREATED when free − estimated size < guard + margin; sqlite mode then runs the memory path (never stops the archive)", () => {
  const dir = seededDir("create");
  const { out, errors } = withGuard("sqlite", () => 1_146_126_336, () => ({ r: archiveNewRows(fourRows(), "2026-11-03", dir), st: dtccIdsStats() }));
  assert.ok(!fs.existsSync(dbPath(dir)), "nothing created");
  assert.deepEqual(out.r, { newRows: 2, seenTotal: 4 }, "ids 3 and 4 archived exactly as memory mode would");
  assert.equal(out.st.diskGuard.active, true);
  assert.equal(out.st.memoryCount, 4, "sqlite mode fell back to the Set");
  assert.ok(errors.some((e) => e.includes("DTCC-IDS-DISK-GUARD:") && e.includes("would leave")));
  fs.rmSync(dir, { recursive: true, force: true });
});

test("hysteresis: an existing DB is KEPT while free ≥ the guard even inside the creation margin (no create/remove flapping)", () => {
  const dir = seededDir("hyst");
  withGuard("dual", () => 64 * GIB, () => archiveNewRows(fourRows(), "2026-11-03", dir));
  const { out } = withGuard("dual", () => GIB + 10 * 2 ** 20, () => ({ r: archiveNewRows(fourRows(), "2026-11-03", dir), st: dtccIdsStats() }));
  assert.ok(fs.existsSync(dbPath(dir)), "kept: free is above the guard");
  assert.equal(out.st.diskGuard.active, false);
  assert.equal(out.st.dbCount, 4);
  fs.rmSync(dir, { recursive: true, force: true });
});

test("an unreadable free-space reading fails OPEN (same posture as volumeAllowsWrite): the DB is used", () => {
  const dir = seededDir("open");
  const { out } = withGuard("dual", () => null, () => ({ r: archiveNewRows(fourRows(), "2026-11-03", dir), st: dtccIdsStats() }));
  assert.ok(fs.existsSync(dbPath(dir)));
  assert.equal(out.st.diskGuard.active, false);
  assert.equal(out.st.mismatches, 0);
  fs.rmSync(dir, { recursive: true, force: true });
});

test("recovery: when the volume has room again the guard clears (DTCC-IDS-DISK-GUARD-CLEAR) and dual mode rebuilds the DB from the index", () => {
  const dir = seededDir("recover");
  let free = 900 * 2 ** 20;
  const { out, errors } = withGuard("dual", () => free, () => {
    const r1 = archiveNewRows(fourRows(), "2026-11-03", dir);
    const s1 = dtccIdsStats();
    free = 8 * GIB;
    const r2 = archiveNewRows(fourRows(), "2026-11-03", dir);
    return { r1, s1, r2, s2: dtccIdsStats() };
  });
  assert.equal(out.s1.diskGuard.active, true);
  assert.equal(out.s2.diskGuard.active, false);
  assert.ok(fs.existsSync(dbPath(dir)), "rebuilt");
  assert.equal(out.s2.dbCount, 4, "backfilled from the compact index, which the memory path kept appending");
  assert.equal(out.s2.mismatches, 0);
  assert.deepEqual(out.r2, { newRows: 0, seenTotal: 4 });
  assert.ok(errors.some((e) => e.includes("DTCC-IDS-DISK-GUARD-CLEAR")));
  fs.rmSync(dir, { recursive: true, force: true });
});

test("memory mode never consults the guard (no DB, no lines)", () => {
  const dir = seededDir("mem");
  const { out, errors } = withGuard("memory", () => 1, () => ({ r: archiveNewRows(fourRows(), "2026-11-03", dir), st: dtccIdsStats() }));
  assert.deepEqual(out.r, { newRows: 2, seenTotal: 4 });
  assert.equal(out.st.diskGuard.active, false);
  assert.ok(!errors.some((e) => e.includes("DTCC-IDS-DISK-GUARD")));
  fs.rmSync(dir, { recursive: true, force: true });
});
