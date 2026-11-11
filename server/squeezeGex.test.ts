// Edge P1 squeeze_gex stream: parse, registered rank definition, and — since
// 2026-11-06 ([REPAIR], SqueezeMetrics' terms) — NOTHING stored: the file is
// summarized in memory, only the derived summary is written, the old archive
// directory is deleted once at boot, the status probe serves no raw value.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import { fileURLToPath } from "node:url";
import {
  parseSqueezeCsv, trailingPctRank, summarizeSqueeze, squeezeReport, refreshSqueezeGex,
  readSqueezeSummary, removeSqueezeArchive, runSqueezeArchiveCleanup, squeezeSummaryPath,
  _resetSqueezeForTests, RANK_WINDOW, SQUEEZE_SUMMARY_FILE, SQUEEZE_SOURCE, type SqueezeSummary,
} from "./squeezeGex";

const here = path.dirname(fileURLToPath(import.meta.url));

// verbatim row shape from the live 2026-10-29 probe
const CSV = [
  "date,price,dix,gex",
  "2026-10-27,7722.6,0.4549768975156808,5515099082.650005",
  "2026-10-28,7743.98,0.46105040243405043,5605140552.230006",
  "garbage,1,2,3",
  "2026-10-29,7750,0.4,notanumber",
].join("\n") + "\n";

const tmp = () => fs.mkdtempSync(path.join(os.tmpdir(), "sqgex-"));

/** Synthetic DIX.csv (integers only, so the TS and Python generators print the
 *  same text): n weekday rows ending 2026-11-05, gex_i = ((i*7919) % 1009) * 1e6,
 *  the last row's gex chosen per scenario. test_dealer_gamma.py builds the same. */
function fixtureCsv(lastGex: number, n = 300, last = "2026-11-05"): string {
  const days: string[] = [];
  const d = new Date(last + "T00:00:00Z");
  while (days.length < n) {
    const wd = d.getUTCDay();
    if (wd !== 0 && wd !== 6) days.push(d.toISOString().slice(0, 10));
    d.setUTCDate(d.getUTCDate() - 1);
  }
  days.reverse();
  const lines = days.map((dt, i) => `${dt},${5000 + i},0.4,${i === n - 1 ? lastGex : ((i * 7919) % 1009) * 1000000}`);
  return ["date,price,dix,gex", ...lines].join("\n") + "\n";
}
const SCENARIOS = { low: [50000000, 300], normal: [900000000, 300], negative: [-5000000, 300], thin: [50000000, 200] } as const;
const FETCHED = "2026-11-06T05:00:00.000Z";

/** The OLD archive path's answers on these fixtures — parseSqueezeCsv ->
 *  archiveSqueezeRows -> squeezeReport, run on main a017a2be before the
 *  archive was removed (2026-11-06). The in-memory summary must match them. */
const OLD_ARCHIVE_REPORT = {
  low: { rows: 300, first_date: "2025-09-12", latest_date: "2026-11-05", latest_gex_negative: false, latest_rank: 0.049800796812749, rank_rows: 49 },
  normal: { rows: 300, first_date: "2025-09-12", latest_date: "2026-11-05", latest_gex_negative: false, latest_rank: 0.8924302788844621, rank_rows: 49 },
  negative: { rows: 300, first_date: "2025-09-12", latest_date: "2026-11-05", latest_gex_negative: true, latest_rank: 0, rank_rows: 49 },
  thin: { rows: 200, first_date: "2026-01-30", latest_date: "2026-11-05", latest_gex_negative: false, latest_rank: null, rank_rows: 0 },
} as const;

test("parseSqueezeCsv: verbatim shape, bad lines skipped, sorted", () => {
  const rows = parseSqueezeCsv(CSV);
  assert.deepEqual(rows.map((r) => r.date), ["2026-10-27", "2026-10-28"]);
  assert.equal(rows[1].gex, 5605140552.230006);
  assert.deepEqual(parseSqueezeCsv("nope\n1,2"), []);
});

test("trailingPctRank: burn-in null, extremes 0/1, ties half, uses only past rows", () => {
  const v = [5, 1, 2, 3, 4];
  assert.deepEqual(trailingPctRank(v, 5), [null, null, null, null, 0.75]);
  assert.equal(trailingPctRank([1, 2, 3, 4, 5], 5)[4], 1);
  assert.equal(trailingPctRank([5, 4, 3, 2, 1], 5)[4], 0);
  assert.equal(trailingPctRank([2, 2, 2, 2, 2], 5)[4], 0.5);
  // appending a future value never changes an earlier rank
  assert.equal(trailingPctRank([...v, 100], 5)[4], trailingPctRank(v, 5)[4]);
  assert.equal(RANK_WINDOW, 252);
});

test("in-memory summary == the old archive path's report on the same fixtures (rank, dates, counts, sign)", () => {
  assert.equal(fixtureCsv(50000000).split("\n")[1], "2025-09-12,5000,0.4,0");
  assert.equal(fixtureCsv(50000000).trim().split("\n").pop(), "2026-11-05,5299,0.4,50000000");
  for (const [name, [lastGex, n]] of Object.entries(SCENARIOS)) {
    const s = summarizeSqueeze(parseSqueezeCsv(fixtureCsv(lastGex, n)), FETCHED);
    const old = OLD_ARCHIVE_REPORT[name as keyof typeof OLD_ARCHIVE_REPORT];
    assert.deepEqual(
      { rows: s.rows, first_date: s.firstDate, latest_date: s.asOf, latest_gex_negative: s.gexNegative, latest_rank: s.rank252, rank_rows: s.rankRows },
      old, name);
    // ... and equal to the full-history rank the archive path computed
    const all = trailingPctRank(parseSqueezeCsv(fixtureCsv(lastGex, n)).map((r) => r.gex));
    assert.equal(s.rank252, all[all.length - 1], `${name}: last-252 rank == full-history rank`);
  }
  // the exact file dealer_gamma.py reads for the "low" fixture (test_dealer_gamma.py pins the same object)
  assert.deepEqual(summarizeSqueeze(parseSqueezeCsv(fixtureCsv(50000000)), FETCHED), {
    v: 1, source: SQUEEZE_SOURCE, fetchedAt: FETCHED, asOf: "2026-11-05", firstDate: "2025-09-12",
    rows: 300, rank252: 12.5 / 251, rankRows: 49, gexNegative: false,
  } satisfies SqueezeSummary);
  assert.equal(summarizeSqueeze([], FETCHED).asOf, null);
});

test("a poll writes the derived summary and NOTHING else: no archive dir, no row, no raw value", async () => {
  const root = tmp();
  const prev = process.env.DATA_DIR;
  process.env.DATA_DIR = root;
  _resetSqueezeForTests();
  try {
    const csv = fixtureCsv(50000000);
    const s = await refreshSqueezeGex(async () => ({ ok: true, status: 200, text: async () => csv }), Date.parse(FETCHED));
    assert.ok(s && s.rank252 === 12.5 / 251);
    const written = (fs.readdirSync(root, { recursive: true }) as string[]).sort();
    assert.deepEqual(written, [SQUEEZE_SUMMARY_FILE], "the only file under DATA_DIR: no datacore_archive/squeeze_gex, no temp file");
    assert.equal(squeezeSummaryPath(), path.join(root, SQUEEZE_SUMMARY_FILE));
    const onDisk = JSON.parse(fs.readFileSync(path.join(root, SQUEEZE_SUMMARY_FILE), "utf8"));
    assert.deepEqual(Object.keys(onDisk).sort(), ["asOf", "fetchedAt", "firstDate", "gexNegative", "rank252", "rankRows", "rows", "source", "v"]);
    const raw = new Set(parseSqueezeCsv(csv).flatMap((r) => [r.price, r.dix, r.gex]));
    for (const [k, v] of Object.entries(onDisk)) {
      if (typeof v === "number") assert.ok(!raw.has(v), `${k}=${v} is not a SqueezeMetrics value`);
    }
    assert.deepEqual(readSqueezeSummary(path.join(root, SQUEEZE_SUMMARY_FILE)), s);
  } finally {
    _resetSqueezeForTests();
    if (prev === undefined) delete process.env.DATA_DIR; else process.env.DATA_DIR = prev;
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("a failed poll writes nothing and flags failure; the last summary stands; the status never carries a raw value", async () => {
  const root = tmp();
  const fp = path.join(root, SQUEEZE_SUMMARY_FILE);
  _resetSqueezeForTests();
  try {
    const ok = async () => ({ ok: true, status: 200, text: async () => CSV });
    assert.equal((await refreshSqueezeGex(ok, Date.parse("2026-10-29T12:00:00Z"), fp))?.rows, 2);
    const good = fs.readFileSync(fp);
    assert.equal(squeezeReport(Date.parse("2026-10-29T12:00:00Z"), fp).last_refresh_ok, true);
    const bad = async () => ({ ok: false, status: 503, text: async () => "" });
    assert.equal(await refreshSqueezeGex(bad, Date.now(), fp), null);
    let r = squeezeReport(Date.parse("2026-10-29T12:00:00Z"), fp);
    assert.equal(r.last_refresh_ok, false);
    assert.equal(r.last_error, "HTTP 503");
    const empty = async () => ({ ok: true, status: 200, text: async () => "<html>" });
    assert.equal(await refreshSqueezeGex(empty, Date.now(), fp), null);
    assert.ok(fs.readFileSync(fp).equals(good), "the last good summary is untouched");
    assert.deepEqual(fs.readdirSync(root), [SQUEEZE_SUMMARY_FILE]);
    r = squeezeReport(Date.parse("2026-10-29T12:00:00Z"), fp);
    assert.equal(r.stored, false);
    assert.equal(r.rows, 2);
    assert.equal(r.latest_date, "2026-10-28");
    assert.equal(r.latest_rank, null, "2 rows: burn-in");
    assert.equal(r.latest_age_days, 1);
    assert.ok(!/"(gex|dix|price)"\s*:/.test(JSON.stringify(r)), "no raw value in the status");
    // a restart: no in-memory summary, the persisted one is reported
    _resetSqueezeForTests();
    assert.equal(squeezeReport(Date.parse("2026-10-29T12:00:00Z"), fp).latest_date, "2026-10-28");
    assert.equal(squeezeReport(Date.now(), path.join(root, "absent.json")).rows, 0);
  } finally {
    _resetSqueezeForTests();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("boot cleanup deletes the old squeeze_gex archive and NOTHING else, once, with an audit row", () => {
  const base = tmp();
  const outside = tmp();
  try {
    const sq = path.join(base, "squeeze_gex");
    fs.mkdirSync(sq);
    fs.writeFileSync(path.join(sq, "2026.jsonl"), '{"date":"2026-11-05","price":1,"dix":0.4,"gex":2,"fetched_at":"T"}\n');
    fs.writeFileSync(path.join(sq, "2025.jsonl.gz"), zlib.gzipSync('{"date":"2025-12-31","price":1,"dix":0.4,"gex":1,"fetched_at":"T"}\n'));
    const keep: Record<string, Buffer> = {
      "dtccswaps/2026-11-05.jsonl.gz": zlib.gzipSync('{"disseminationId":"1"}\n'),
      "optionchains/oi/2026-11-05.jsonl": Buffer.from('{"s":"SPY"}\n'),
      "squeeze_gex_other/2026.jsonl": Buffer.from("{}\n"),
      ".platform_stats_cache.json": Buffer.from("{}"),
    };
    for (const [rel, body] of Object.entries(keep)) {
      fs.mkdirSync(path.dirname(path.join(base, rel)), { recursive: true });
      fs.writeFileSync(path.join(base, rel), body);
    }
    const audit: Array<[string, string]> = [];
    const r = runSqueezeArchiveCleanup((t, m) => audit.push([t, m]), base, Date.parse("2026-11-06T21:30:00Z"));
    assert.equal(r?.removed, true);
    assert.equal(r?.files, 2);
    assert.ok((r?.bytes ?? 0) > 0);
    assert.equal(fs.existsSync(sq), false, "the archive is gone");
    for (const [rel, body] of Object.entries(keep)) {
      assert.ok(fs.readFileSync(path.join(base, rel)).equals(body), `${rel} untouched`);
    }
    assert.deepEqual(audit.map(([t]) => t), ["SQUEEZE-ARCHIVE-REMOVED"]);
    assert.match(audit[0][1], /2 file\(s\)/);
    assert.ok(audit[0][1].length <= 500);
    assert.equal(squeezeReport().archive_cleanup?.removed, true, "the status shows this boot's cleanup");
    // every later boot: nothing there, nothing logged
    assert.equal(runSqueezeArchiveCleanup((t, m) => audit.push([t, m]), base), null);
    assert.equal(audit.length, 1);

    // a symlink in its place is removed as a link; the target is never touched
    fs.writeFileSync(path.join(outside, "2026.jsonl"), "{}\n");
    fs.symlinkSync(outside, sq);
    assert.equal(removeSqueezeArchive(base)?.removed, true);
    assert.equal(fs.existsSync(sq), false);
    assert.ok(fs.existsSync(path.join(outside, "2026.jsonl")), "the symlink target survives");

    // an unreadable location reports an error row (the next boot retries), never a throw
    const fileBase = path.join(base, ".platform_stats_cache.json"); // a FILE: <file>/squeeze_gex is ENOTDIR
    const errAudit: Array<[string, string]> = [];
    const e = runSqueezeArchiveCleanup((t, m) => errAudit.push([t, m]), fileBase);
    assert.equal(e?.removed, false);
    assert.deepEqual(errAudit.map(([t]) => t), ["SQUEEZE-ARCHIVE-ERROR"]);
  } finally {
    _resetSqueezeForTests();
    fs.rmSync(base, { recursive: true, force: true });
    fs.rmSync(outside, { recursive: true, force: true });
  }
});

test("wiring: routes.ts passes the audit sink to the boot poll; the module has no archive writer left", () => {
  const routes = fs.readFileSync(path.join(here, "routes.ts"), "utf8");
  const i = routes.indexOf("bootSqueezeGexPoll({");
  assert.ok(i > 0, "bootSqueezeGexPoll called with options");
  assert.match(routes.slice(i, i + 600), /audit:\s*\(type, message\)/);
  assert.match(routes.slice(i, i + 600), /INSERT INTO audit_log/);
  const src = fs.readFileSync(path.join(here, "squeezeGex.ts"), "utf8");
  assert.ok(!/appendFileSync|createWriteStream|zlib/.test(src), "no append / stream / gzip writer: nothing accumulates");
  assert.equal((src.match(/writeFileSync\(/g) || []).length, 1, "one write: the summary's temp file (then renamed)");
});
