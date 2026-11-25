// dtccEdgeDays.test.ts — [PIPELINE] 2026-11-25, open_questions.md (ap4): the
// vendor moves a dated file to cold storage two years after it is written (a
// HEAD still answers 200, a GET is refused), so one vendor day becomes
// unreadable every day. Each poll reads the days at that edge, oldest first,
// before the catch-up. The files that old are STREAMED zip members (sizes in a
// trailing data descriptor), which the reader refused; it now reads them.
// Run: npx tsx --test server/dtccEdgeDays.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import { Readable } from "node:stream";
import {
  REQUIRED_COLUMNS, refreshDtccSwaps, latestDtcc, setDtccIdsAuditSink, _resetDtccForTests,
  streamDtccZip, tryParseLocalHeader, parseStreamLocalHeader,
  readDtccSourceDays, dtccEdgeCandidates, dtccEdgeEnabled, dtccSourceDaysSummary,
  DTCC_EDGE_OLDEST_DAYS, DTCC_EDGE_SPAN_DAYS, DTCC_EDGE_ENV, DTCC_CATCHUP_ENV, type DtccSourceDays,
} from "./dtccSwaps";

const IDS_ENV = "VOLTRADE_DTCC_IDS_STORE";
const HEADER_COLS = [...REQUIRED_COLUMNS];
const HEADER_LINE = HEADER_COLS.map((c) => `"${c}"`).join(",");

function csvRow(diss: string): string {
  const byName: Record<string, string> = {
    "Dissemination Identifier": diss, "Original Dissemination Identifier": diss, "Action type": "NEWT",
    "Event timestamp": "2024-11-25T12:33:36Z", "Effective Date": "2024-11-25", "Asset Class": "EQ",
    "Notional amount-Leg 1": "1,000,000", "Notional currency-Leg 1": "USD",
    "Underlier ID-Leg 1": "US0378331005", "Underlier ID source-Leg 1": "ISIN", "UPI Underlier Name": "APPLE INC",
  };
  return HEADER_COLS.map((c) => `"${byName[c] ?? ""}"`).join(",");
}

/** A single-member zip. `streamed`: general-purpose flag bit 3, zero sizes in
 *  the local header, the real ones in a data descriptor after the data — the
 *  form of the vendor's files from mid-2025 back. A central-directory stand-in
 *  follows either way, so the reader must stop at the member's own end. */
function buildZip(ids: string[], opts: { streamed?: boolean; method?: number; cutCompressedTo?: number } = {}): Buffer {
  const nameBuf = Buffer.from("SEC_CUMULATIVE_EQUITIES.csv", "utf8");
  const dataBuf = Buffer.from([HEADER_LINE, ...ids.map(csvRow)].join("\n") + "\n", "utf8");
  const compressed = zlib.deflateRawSync(dataBuf);
  const header = Buffer.alloc(30);
  header.writeUInt32LE(0x04034b50, 0);
  header.writeUInt16LE(20, 4);
  header.writeUInt16LE(opts.streamed ? 0x0008 : 0, 6);
  header.writeUInt16LE(opts.method ?? 8, 8);
  header.writeUInt32LE(opts.streamed ? 0 : compressed.length, 18);
  header.writeUInt32LE(opts.streamed ? 0 : dataBuf.length, 22);
  header.writeUInt16LE(nameBuf.length, 26);
  if (opts.cutCompressedTo !== undefined) return Buffer.concat([header, nameBuf, compressed.subarray(0, opts.cutCompressedTo)]);
  const descriptor = Buffer.alloc(16);
  descriptor.writeUInt32LE(0x08074b50, 0);
  descriptor.writeUInt32LE(compressed.length, 8);
  descriptor.writeUInt32LE(dataBuf.length, 12);
  return Buffer.concat([header, nameBuf, compressed, ...(opts.streamed ? [descriptor] : []), Buffer.from("PK\u0001\u0002CENTRAL-DIRECTORY-STAND-IN".repeat(50))]);
}

function body(buf: Buffer, chunk = 4096) {
  const parts: Buffer[] = [];
  for (let i = 0; i < buf.length; i += chunk) parts.push(buf.subarray(i, i + chunk));
  return Readable.toWeb ? Readable.toWeb(Readable.from(parts)) : Readable.from(parts);
}

function vendor(files: Record<string, Buffer>) {
  const asked: string[] = [];
  const fetchImpl = async (url: string) => {
    const m = /EQUITIES_(\d{4})_(\d{2})_(\d{2})\.zip/.exec(url);
    const day = m ? `${m[1]}-${m[2]}-${m[3]}` : "";
    asked.push(day);
    const zip = files[day];
    if (!zip) return { ok: false, status: 403, body: undefined as any };
    return { ok: true, status: 200, body: body(zip) };
  };
  return { fetchImpl: fetchImpl as any, asked };
}

async function withEnv<T>(vars: Record<string, string | undefined>, fn: () => Promise<T>): Promise<T> {
  const before: Record<string, string | undefined> = {};
  for (const k of Object.keys(vars)) { before[k] = process.env[k]; if (vars[k] === undefined) delete process.env[k]; else process.env[k] = vars[k]; }
  try { return await fn(); } finally {
    for (const k of Object.keys(vars)) { if (before[k] === undefined) delete process.env[k]; else process.env[k] = before[k]; }
  }
}

function freshDir(): string {
  _resetDtccForTests();
  return fs.mkdtempSync(path.join(os.tmpdir(), "dtcc-edge-"));
}

function dayFileIds(dir: string, day: string): string[] {
  const fp = path.join(dir, "dtccswaps", `${day}.jsonl.gz`);
  if (!fs.existsSync(fp)) return [];
  return zlib.gunzipSync(fs.readFileSync(fp)).toString("utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l).disseminationId);
}

const at = (iso: string) => Date.parse(iso);

test("the reader reads a streamed member to its own end: every line, nothing from the descriptor or the directory behind it", async () => {
  const ids = Array.from({ length: 3000 }, (_, i) => `s${i}`);
  const zip = buildZip(ids, { streamed: true });
  assert.throws(() => tryParseLocalHeader(zip), /streamed \(data-descriptor\) member not supported/, "the size-first parser still refuses it");
  const h = parseStreamLocalHeader(zip)!;
  assert.equal(h.streamed, true);
  assert.equal(h.method, 8);
  const lines: string[] = [];
  const r = await streamDtccZip("https://x/y.zip", (line) => lines.push(line), (async () => ({ ok: true, status: 200, body: body(zip, 1500) })) as any);
  assert.deepEqual([r.ok, r.status, r.lines], [true, 200, 3001]);
  assert.equal(lines[0], HEADER_LINE);
  assert.equal(lines[3000], csvRow("s2999"));
  assert.ok(lines.every((l) => !l.includes("CENTRAL-DIRECTORY")), "no byte after the member is read as data");
});

test("a member that is not streamed is read exactly as before, by its stated size", async () => {
  const zip = buildZip(["a", "b"]);
  assert.equal(parseStreamLocalHeader(zip)!.streamed, false);
  assert.deepEqual(parseStreamLocalHeader(zip)!.compressedSize, tryParseLocalHeader(zip)!.compressedSize);
  const lines: string[] = [];
  const r = await streamDtccZip("https://x/y.zip", (line) => lines.push(line), (async () => ({ ok: true, status: 200, body: body(zip) })) as any);
  assert.equal(r.lines, 3);
  assert.equal(lines[2], csvRow("b"));
});

test("a streamed member cut off mid-body is an error, never a short file; a streamed STORED member is refused", async () => {
  const ids = Array.from({ length: 3000 }, (_, i) => `s${i}`);
  const cut = buildZip(ids, { streamed: true, cutCompressedTo: 900 });
  await assert.rejects(
    streamDtccZip("https://x/y.zip", () => {}, (async () => ({ ok: true, status: 200, body: body(cut) })) as any),
    /unexpected end of file/,
  );
  assert.throws(() => parseStreamLocalHeader(buildZip(["a"], { streamed: true, method: 0 })), /streamed member with compression method 0 not supported/);
});

test("edge candidates: the three vendor days about to leave the readable window, oldest first; a day that did not answer is returned again at once", () => {
  assert.equal(DTCC_EDGE_OLDEST_DAYS, 731);
  assert.equal(DTCC_EDGE_SPAN_DAYS, 3);
  // measured on UTC day 2026-11-25: 2024_11_24 downloaded, 2024_11_23 was in cold storage
  const now = at("2026-11-25T13:35:00Z");
  assert.deepEqual(dtccEdgeCandidates({}, now), ["2024-11-24", "2024-11-25", "2024-11-26"]);
  const t = "2026-11-25T13:35:00.000Z";
  const days: DtccSourceDays = {
    "2024-11-24": { firstAt: t, lastAt: t, via: "edge", usRows: 2, newRows: 2, reads: 1 },
    "2024-11-25": { firstAt: t, lastAt: t, via: "edge", usRows: 0, newRows: 0, reads: 1, unavailable: 503 },
  };
  assert.deepEqual(dtccEdgeCandidates(days, now + 60_000), ["2024-11-25", "2024-11-26"], "no week's wait at the edge");
  assert.deepEqual(dtccEdgeCandidates(days, at("2026-11-26T00:10:00Z")), ["2024-11-25", "2024-11-26", "2024-11-27"], "the window moves a day at 00:00Z");
  for (const off of ["0", "off", "false"]) assert.equal(dtccEdgeEnabled({ [DTCC_EDGE_ENV]: off }), false);
  assert.equal(dtccEdgeEnabled({}), true);
});

test("THE EDGE: a poll reads the days about to become unreadable before the catch-up, archives them, and asks again for one that did not answer", async () => {
  await withEnv({ [IDS_ENV]: "memory", [DTCC_EDGE_ENV]: undefined, [DTCC_CATCHUP_ENV]: undefined }, async () => {
    const dir = freshDir();
    const audits: string[] = [];
    setDtccIdsAuditSink((type, message) => audits.push(`${type}: ${message}`));
    try {
      const files: Record<string, Buffer> = {
        "2026-11-24": buildZip(["n1", "n2"]),
        "2026-11-23": buildZip(["m1"]),
        "2024-11-24": buildZip(["e1"], { streamed: true }),
        "2024-11-25": buildZip(["e2", "e3", "e4"], { streamed: true }),
      };
      const v = vendor(files);
      await refreshDtccSwaps(v.fetchImpl, at("2026-11-25T13:35:00Z"), dir);
      assert.deepEqual(v.asked, [
        "2026-11-25", "2026-11-24",                 // the poll's walk to the newest file
        "2024-11-24", "2024-11-25", "2024-11-26",   // the edge, oldest first
        "2026-11-23", "2026-11-22",                 // then the catch-up
      ]);
      const hit = latestDtcc()!;
      assert.equal(hit.sourceDate, "2026-11-24");
      assert.equal(hit.newRows, 2, "the poll's own count is unchanged by what follows it");
      assert.equal(hit.totalArchived, 7);
      assert.deepEqual(dayFileIds(dir, "2026-11-25").sort(), ["e1", "e2", "e3", "e4", "m1", "n1", "n2"]);

      const days = readDtccSourceDays(dir);
      assert.deepEqual([days["2024-11-24"].via, days["2024-11-24"].newRows], ["edge", 1]);
      assert.deepEqual([days["2024-11-25"].via, days["2024-11-25"].newRows], ["edge", 3]);
      assert.equal(days["2024-11-26"].unavailable, 403);
      assert.equal(days["2024-11-26"].via, "edge");
      assert.deepEqual(audits.filter((a) => a.startsWith("DTCC-EDGE-DAY-UNREAD")).length, 1);
      assert.match(audits.find((a) => a.startsWith("DTCC-EDGE-DAY-UNREAD"))!, /vendor file dated 2024-11-26 did not answer \(HTTP 403\)/);
      assert.equal(audits.filter((a) => a.startsWith("DTCC-SOURCE-DAY-REFILLED") && /2024-/.test(a)).length, 0, "an edge day is not a refill: it was never due before");

      const s = dtccSourceDaysSummary(dir);
      assert.deepEqual(s.edge, { enabled: true, days: 2, rows: 4, oldest: "2024-11-24", newest: "2024-11-25" });
      assert.equal(s.through, "2026-11-24");
      assert.equal(s.expected, 96, "the edge days are older than the floor and are not in the since-the-floor count");
      assert.equal(s.read, 2);

      // the next poll, six hours on: the two edge days held are not fetched again; the third is
      files["2024-11-26"] = buildZip(["e5"], { streamed: true });
      const v2 = vendor(files);
      await refreshDtccSwaps(v2.fetchImpl, at("2026-11-25T19:35:00Z"), dir);
      assert.deepEqual(v2.asked.filter((d) => d.startsWith("2024-")), ["2024-11-26"]);
      assert.equal(readDtccSourceDays(dir)["2024-11-26"].unavailable, undefined);
      assert.equal(dtccSourceDaysSummary(dir).edge.days, 3);
      assert.equal(latestDtcc()!.totalArchived, 8);
    } finally {
      setDtccIdsAuditSink(null);
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });
});

test("edge off switch: no edge file is asked for; the catch-up runs as before", async () => {
  await withEnv({ [IDS_ENV]: "memory", [DTCC_EDGE_ENV]: "0", [DTCC_CATCHUP_ENV]: undefined }, async () => {
    const dir = freshDir();
    try {
      const v = vendor({ "2026-11-24": buildZip(["n1"]), "2024-11-24": buildZip(["e1"], { streamed: true }) });
      await refreshDtccSwaps(v.fetchImpl, at("2026-11-25T13:35:00Z"), dir);
      assert.deepEqual(v.asked, ["2026-11-25", "2026-11-24", "2026-11-23", "2026-11-22"]);
      assert.deepEqual(dtccSourceDaysSummary(dir).edge, { enabled: false, days: 0, rows: 0, oldest: null, newest: null });
    } finally { fs.rmSync(dir, { recursive: true, force: true }); }
  });
});
