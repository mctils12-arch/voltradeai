// SEC XBRL fundamentals record, the reading and writing half: one night
// against a stand-in for data.sec.gov, the files it leaves, and the schedule.
import test from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import os from "os";
import path from "path";
import zlib from "zlib";
import { fileURLToPath } from "url";
import {
  XBRL_MAX_ATTEMPTS_PER_DAY, XbrlRefusedError, fetchFrame, gzipOldDays, loadRunFile, loadSnapshot, resolveCompanyQuery,
  runXbrlNight, xbrlCompanyView, xbrlDir, xbrlFramesEnabled, xbrlFramesStatus, xbrlNightDue, xbrlPublicView, xbrlTick,
  type XbrlFetchFn, type XbrlFetchResponse, type XbrlNightDeps,
} from "./xbrlFrames";
import { XBRL_CONCEPTS, frameRoster, frameUrl, type FrameRef } from "./xbrlFramesCore";
import { sanitizeDiag } from "./diag";
import { JOB_REGISTRY } from "./jobHeartbeat";

const here = path.dirname(fileURLToPath(import.meta.url));
const NI = XBRL_CONCEPTS.find((c) => c.tag === "NetIncomeLoss")!;
const ASSETS = XBRL_CONCEPTS.find((c) => c.tag === "Assets")!;
const ref = (c: typeof NI, period: string): FrameRef => ({ concept: c, period, key: `${c.taxonomy}/${c.tag}/${c.unit}/${period}`, url: frameUrl(c, period) });
const ROSTER = [ref(NI, "CY2026Q3"), ref(ASSETS, "CY2026Q3I"), ref(NI, "CY2026Q4")];

type Row = { accn: string; cik: number; entityName: string; loc: string; end: string; val: number; start?: string };
const row = (cik: number, val: number, accn = "0000000001-26-000001", name = `CO ${cik}`): Row => ({ accn, cik, entityName: name, loc: "US-NY", end: "2026-09-30", val });
const answer = (status: number, body: unknown): XbrlFetchResponse => ({ ok: status >= 200 && status < 300, status, text: async () => (typeof body === "string" ? body : JSON.stringify(body)) });

/** A stand-in SEC: url -> rows (served as a frame), a status number, or absent (404). */
function sec(frames: Record<string, Row[] | number>, calls: string[] = []): XbrlFetchFn {
  return async (url) => {
    calls.push(url);
    const hit = frames[url];
    if (hit === undefined) return answer(404, "not found");
    if (typeof hit === "number") return answer(hit, "refused");
    return answer(200, { data: hit });
  };
}

function deps(fetch: XbrlFetchFn, at: string, over: Partial<XbrlNightDeps> = {}): XbrlNightDeps {
  return { fetch, now: () => new Date(at), pauseMs: 0, retryPauseMs: 0, timeoutsMs: [1000, 1000, 1000], freeBytes: () => 5e9, tickers: async () => new Map([["2", "TWO"]]), roster: () => ROSTER, ...over };
}
const tmp = () => fs.mkdtempSync(path.join(os.tmpdir(), "xbrl-frames-"));
const lines = (dir: string, day: string) => fs.readFileSync(path.join(xbrlDir(dir), `${day}.jsonl`), "utf8").trim().split("\n").map((l) => JSON.parse(l));

const N1 = "2026-11-21T07:10:00.000Z";
const N2 = "2026-11-22T07:10:00.000Z";

test("the first night writes every fact as a backfill row, a snapshot per frame, and the night's own line", async () => {
  const base = tmp();
  const s = await runXbrlNight(deps(sec({ [ROSTER[0].url]: [row(1, 10), row(2, 20)], [ROSTER[1].url]: [row(1, 100)] }), N1, { baseDir: base }));
  assert.equal(s.status, "ok");
  assert.deepEqual(s.frames, { roster: 3, read: 2, not_published: 1, failed: 0 });
  assert.deepEqual(s.rows, { first: 3, revised: 0, dropped: 0, refiled: 0, backfill: 3 });
  const out = lines(base, "2026-11-21");
  assert.equal(out.length, 4);
  assert.ok(out.slice(0, 3).every((r) => r.kind === "first" && r.backfill === true && r.seen === N1));
  assert.equal(out[3].kind, "run");
  assert.deepEqual(loadSnapshot(xbrlDir(base), ROSTER[0].key), { "1": [10, "0000000001-26-000001", "2026-09-30"], "2": [20, "0000000001-26-000001", "2026-09-30"] });
  assert.deepEqual(loadSnapshot(xbrlDir(base), ROSTER[2].key), {}, "a frame SEC has not published yet is held as empty");
  assert.equal(s.companies, 2);
});

test("a night with no change writes one line: the night itself", async () => {
  const base = tmp();
  const frames = { [ROSTER[0].url]: [row(1, 10)], [ROSTER[1].url]: [row(1, 100)] };
  await runXbrlNight(deps(sec(frames), N1, { baseDir: base }));
  const s = await runXbrlNight(deps(sec(frames), N2, { baseDir: base }));
  assert.equal(s.rows.first + s.rows.revised + s.rows.dropped, 0);
  const out = lines(base, "2026-11-22");
  assert.equal(out.length, 1);
  assert.equal(out[0].kind, "run");
  assert.equal(loadRunFile(xbrlDir(base)).nights, 2);
});

test("the second night records a new company as a real first sighting and a changed number as a revision with its company named", async () => {
  const base = tmp();
  await runXbrlNight(deps(sec({ [ROSTER[0].url]: [row(2, 20, "0000000002-25-000007")], [ROSTER[1].url]: [row(1, 100)] }), N1, { baseDir: base }));
  const s = await runXbrlNight(deps(sec({
    [ROSTER[0].url]: [row(2, 21, "0000000002-26-000009", "TWO INC"), row(3, 30)],
    [ROSTER[1].url]: [row(1, 100, "0000000001-26-000044")],
    [ROSTER[2].url]: [row(9, 5)],
  }), N2, { baseDir: base }));
  assert.deepEqual(s.rows, { first: 2, revised: 1, dropped: 0, refiled: 1, backfill: 0 });
  const out = lines(base, "2026-11-22");
  const first3 = out.find((r) => r.cik === 3);
  assert.equal(first3.backfill, undefined);
  const early = out.find((r) => r.cik === 9);
  assert.equal(early.backfill, undefined, "a frame first read as empty: its first company is a first sighting");
  const view = xbrlPublicView(base);
  assert.equal(view.kind, "raw");
  assert.equal(view.warming_up, false);
  assert.deepEqual(view.revisions, [{
    seen: N2, tag: "NetIncomeLoss", period: "CY2026Q3", cik: 2, name: "TWO INC", ticker: "TWO", end: "2026-09-30", val: 21, prev_val: 20, unit: "USD",
    filing: "https://www.sec.gov/Archives/edgar/data/2/000000000226000009/0000000002-26-000009-index.htm",
    prev_filing: "https://www.sec.gov/Archives/edgar/data/2/000000000225000007/0000000002-25-000007-index.htm",
  }]);
  assert.equal(view.coverage.length, 3);
});

test("one failed frame does not stop the night, and its snapshot is left as it was", async () => {
  const base = tmp();
  await runXbrlNight(deps(sec({ [ROSTER[0].url]: [row(1, 10)], [ROSTER[1].url]: [row(1, 100)] }), N1, { baseDir: base }));
  const calls: string[] = [];
  const s = await runXbrlNight(deps(sec({ [ROSTER[0].url]: 503, [ROSTER[1].url]: [row(1, 101)] }, calls), N2, { baseDir: base }));
  assert.equal(s.status, "partial");
  assert.equal(s.frames.failed, 1);
  assert.equal(calls.filter((u) => u === ROSTER[0].url).length, 3, "a 5xx is asked three times");
  assert.equal(s.rows.revised, 1, "the frame after the failed one was still read");
  assert.deepEqual(loadSnapshot(xbrlDir(base), ROSTER[0].key), { "1": [10, "0000000001-26-000001", "2026-09-30"] });
});

test("a 404 for a frame that held companies is a failure, never a wave of dropped rows", async () => {
  const base = tmp();
  await runXbrlNight(deps(sec({ [ROSTER[0].url]: [row(1, 10)], [ROSTER[1].url]: [row(1, 100)] }), N1, { baseDir: base }));
  const s = await runXbrlNight(deps(sec({ [ROSTER[1].url]: [row(1, 100)] }), N2, { baseDir: base }));
  assert.equal(s.rows.dropped, 0);
  assert.match(s.per_frame[0].error ?? "", /404 for a frame that held 1/);
});

test("SEC refusing the request stops the night at once and the day is not marked complete", async () => {
  const base = tmp();
  const calls: string[] = [];
  const s = await runXbrlNight(deps(sec({ [ROSTER[0].url]: 429, [ROSTER[1].url]: [row(1, 100)] }, calls), N1, { baseDir: base }));
  assert.equal(s.status, "refused");
  assert.deepEqual(calls, [ROSTER[0].url], "asked once, and nothing after it");
  assert.equal(loadRunFile(xbrlDir(base)).completed_day, null);
  await assert.rejects(fetchFrame(sec({ u: 403 }), "u"), XbrlRefusedError);
});

test("a request that throws is asked again; a 404 is 'not published', not an error", async () => {
  let n = 0;
  const flaky: XbrlFetchFn = async () => { n++; if (n === 1) throw new Error("The operation was aborted due to timeout"); return answer(200, { data: [] }); };
  assert.deepEqual(await fetchFrame(flaky, "u", { pauseMs: 0 }), { data: [] });
  assert.equal(n, 2);
  assert.equal(await fetchFrame(sec({}), "u"), null);
});

test("under the volume's free-space floor nothing is fetched or written", async () => {
  const base = tmp();
  const calls: string[] = [];
  const s = await runXbrlNight(deps(sec({}, calls), N1, { baseDir: base, freeBytes: () => 500 * 1_048_576 }));
  assert.equal(s.status, "skipped_low_disk");
  assert.equal(calls.length, 0);
  assert.ok(!fs.existsSync(path.join(xbrlDir(base), "2026-11-21.jsonl")));
  assert.match(s.detail, /500 MB free/);
});

test("a snapshot that cannot be read fails its frame instead of writing the frame again as a backfill", async () => {
  const base = tmp();
  await runXbrlNight(deps(sec({ [ROSTER[0].url]: [row(1, 10)], [ROSTER[1].url]: [row(1, 100)] }), N1, { baseDir: base }));
  const snap = fs.readdirSync(xbrlDir(base)).find((f) => f.startsWith("_snap_") && f.includes("NetIncomeLoss_USD_CY2026Q3"))!;
  fs.writeFileSync(path.join(xbrlDir(base), snap), "not gzip");
  const s = await runXbrlNight(deps(sec({ [ROSTER[0].url]: [row(1, 10)], [ROSTER[1].url]: [row(1, 100)] }), N2, { baseDir: base }));
  assert.equal(s.frames.failed, 1);
  assert.equal(s.rows.first, 0);
});

test("the company view reads what the last night holds for one company, with a link to each filing", async () => {
  const base = tmp();
  const roster = () => frameRoster(new Date(N1)).filter((r) => r.key === "us-gaap/NetIncomeLoss/USD/CY2026Q3" || r.key === "us-gaap/Assets/USD/CY2026Q3I");
  const r = roster();
  await runXbrlNight(deps(sec({ [r[0].url]: [row(320193, 27466000000, "0000320193-26-000111", "Apple Inc.")], [r[1].url]: [row(320193, 404548000000, "0000320193-26-000111", "Apple Inc."), row(5, 1)] }), N1, { baseDir: base, roster }));
  const c = await xbrlCompanyView(320193, base, new Date(N1), "AAPL");
  assert.equal(c.name, "Apple Inc.");
  assert.equal(c.ticker, "AAPL");
  assert.deepEqual(c.facts.map((f) => [f.label, f.period, f.val]), [["Net income", "CY2026Q3", 27466000000], ["Total assets", "CY2026Q3I", 404548000000]]);
  assert.match(c.facts[0].filing, /edgar\/data\/320193\/000032019326000111\//);
  assert.equal((await xbrlCompanyView(777, base, new Date(N1))).facts.length, 0, "a company the record does not hold is an empty answer");
});

test("the company question takes a ticker or a CIK and nothing else", async () => {
  const t = async () => new Map([["320193", "AAPL"], ["2", "BRK-B"]]);
  assert.deepEqual(await resolveCompanyQuery("aapl", t), { cik: 320193, ticker: "AAPL" });
  assert.deepEqual(await resolveCompanyQuery("320193", t), { cik: 320193, ticker: "AAPL" });
  assert.deepEqual(await resolveCompanyQuery("12345", t), { cik: 12345, ticker: null });
  assert.deepEqual(await resolveCompanyQuery("BRK-B", t), { cik: 2, ticker: "BRK-B" });
  for (const bad of ["", "ZZZZ", "../etc", "0", "A B", "x".repeat(40)]) assert.equal(await resolveCompanyQuery(bad, t), null, bad);
});

test("schedule: one read per ET day inside 02:00-05:00, a second try after an interrupted one, never a third", () => {
  const at = new Date("2026-11-21T07:10:00.000Z"); // 02:10 ET
  assert.equal(xbrlNightDue(at, { attempt_day: null, attempts: 0, completed_day: null }), true);
  assert.equal(xbrlNightDue(at, { attempt_day: "2026-11-21", attempts: 1, completed_day: null }), true);
  assert.equal(xbrlNightDue(at, { attempt_day: "2026-11-21", attempts: XBRL_MAX_ATTEMPTS_PER_DAY, completed_day: null }), false);
  assert.equal(xbrlNightDue(at, { attempt_day: "2026-11-21", attempts: 1, completed_day: "2026-11-21" }), false);
  assert.equal(xbrlNightDue(new Date("2026-11-21T15:00:00.000Z"), { attempt_day: null, attempts: 0, completed_day: null }), false, "10:00 ET is outside the window");
});

test("the tick runs the night once, marks the ET day, and stays off when switched off", async () => {
  const base = tmp();
  const calls: string[] = [];
  const d = deps(sec({ [ROSTER[0].url]: [row(1, 10)] }, calls), N1, { baseDir: base });
  assert.equal(await xbrlTick(d, { VOLTRADE_XBRL_FRAMES: "off", DATA_DIR: base }), null);
  assert.equal(calls.length, 0);
  const s = await xbrlTick(d, { DATA_DIR: base });
  assert.equal(s?.status, "ok");
  const run = loadRunFile(xbrlDir(base));
  assert.deepEqual([run.attempt_day, run.attempts, run.completed_day], ["2026-11-21", 1, "2026-11-21"]);
  const before = calls.length;
  assert.equal(await xbrlTick(d, { DATA_DIR: base }), null, "already read today");
  assert.equal(calls.length, before);
  assert.equal(xbrlFramesEnabled({}), true);
});

test("old day files are compressed and still read back whole", async () => {
  const dir = tmp();
  fs.writeFileSync(path.join(dir, "2026-11-10.jsonl"), '{"a":1}\n{"a":2}\n');
  fs.writeFileSync(path.join(dir, "2026-11-21.jsonl"), '{"a":3}\n');
  assert.equal(await gzipOldDays(dir, Date.parse(N1)), 1);
  assert.equal(zlib.gunzipSync(fs.readFileSync(path.join(dir, "2026-11-10.jsonl.gz"))).toString("utf8"), '{"a":1}\n{"a":2}\n');
  assert.ok(!fs.existsSync(path.join(dir, "2026-11-10.jsonl")));
  assert.ok(fs.existsSync(path.join(dir, "2026-11-21.jsonl")), "the newest day stays plain");
});

test("wiring: the job is registered, the probe survives the diag scrub, the routes are mounted, and no trading module reads this", () => {
  assert.ok(JOB_REGISTRY.some((j) => j.id === "xbrl_frames_nightly" && j.source === "server/xbrlFrames.ts"));
  const base = tmp();
  const status = sanitizeDiag(xbrlFramesStatus(base, {})) as Record<string, unknown>;
  assert.equal(status.probe, "xbrl_frames");
  assert.equal(status.enabled, true);
  const routes = fs.readFileSync(path.join(here, "routes.ts"), "utf8");
  assert.match(routes, /registerXbrlFramesRoutes\(app\)/);
  const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
  assert.match(bot, /case "xbrl_frames":/);
  for (const f of ["bot_engine.py", "system_config.py", "ml_model_v2.py"]) {
    assert.ok(!/xbrl/i.test(fs.readFileSync(path.join(here, "..", f), "utf8")), `${f} does not read the fundamentals record`);
  }
  const manifest = JSON.parse(fs.readFileSync(path.join(here, "..", "datacore", "manifests", "xbrlframes.json"), "utf8"));
  assert.equal(manifest.stream, "xbrlframes");
});
