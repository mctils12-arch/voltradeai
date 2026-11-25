// snapshotRefresh — the nightly job that runs snapshot_refresh_job.py (repo root) and
// the reader rule in lazyAssets.ts that serves the newer of the volume copy and the
// bundled one (research/open_questions.md (ah4), 2026-11-25).
// Run: npx tsx --test server/snapshotRefresh.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  snapshotRefreshCmd, inSnapshotRefreshWindow, decideSnapshotRefresh, parseSnapshotRefreshSummary, runSnapshotRefresh,
  describeSource, adoptRefreshedSnapshots, snapshotRefreshLastFinishedAt,
  SNAPSHOT_REFRESH_JOB, SNAPSHOT_REFRESH_TIMEOUT_MS, SNAPSHOT_REFRESH_MIN_FREE_BYTES, SNAPSHOT_REFRESH_SOURCES,
  SNAPSHOT_REFRESH_WINDOW_START_MIN, SNAPSHOT_REFRESH_WINDOW_END_MIN,
} from "./snapshotRefresh";
import {
  lazyJsonTable, pickSnapshotOrigin, snapshotBuiltMs, snapshotRefreshEnabled, snapshotRefreshRoot,
  LAZY_TABLES, REFRESHABLE_TABLES, SNAPSHOT_REFRESH_SUBDIR,
} from "./lazyAssets";
import { snapshotStreams, resetSnapshotStreams } from "./snapshotStreams";
import { WALKFORWARD_ENV_CAPS, type WalkforwardExec } from "./walkforward";
import { JOB_REGISTRY, MAX_DETAIL_CHARS } from "./jobHeartbeat";

const here = path.dirname(fileURLToPath(import.meta.url));
// 2026-11-25 06:20Z = 01:20 ET (EST), a Wednesday
const IN = Date.parse("2026-11-25T06:20:00Z");
const base = { enabled: true, lastFinishedAt: null as string | null, lastAttemptDay: "", freeBytes: 5e9 as number | null };

// The Python's own last stdout line, two runs from a session machine on 2026-11-25:
// the first before the spreadsheet readers were installed, the second after.
const REAL_ERROR = JSON.stringify({ schema: "snapshot_refresh/1", status: "error", code_version: "unknown", finished_at: "2026-11-25T02:39:00+00:00", refreshed: 2, sources: {
  eia: { status: "error", reason: "RuntimeError: every series failed — refusing to write an empty artifact" },
  cpc: { status: "refreshed", built: "2026-11-25T02:38:52+00:00", latest: "2026-11-22", n_series: 96, bytes: 990038 },
  rail: { status: "error", reason: "ModuleNotFoundError: No module named 'openpyxl'" },
  jodi: { status: "refreshed", built: "2026-11-25T02:39:00+00:00", latest: "2026-09", n_series: 349, bytes: 1386124 },
}, elapsed_s: 14.3, peak_rss_mb: 53.5 });
const REAL_OK = JSON.stringify({ schema: "snapshot_refresh/1", status: "ok", code_version: "unknown", finished_at: "2026-11-25T02:39:25+00:00", refreshed: 2, sources: {
  eia: { status: "refreshed", built: "2026-11-25T02:39:07+00:00", latest: "2026-11-13", n_series: 5, bytes: 232746 },
  cpc: { status: "fresh", origin: "refreshed", built: "2026-11-25T02:38:52+00:00", reason: "served copy is 0.0 h old (under 20 h)" },
  rail: { status: "refreshed", built: "2026-11-25T02:39:25+00:00", latest: "2026-11-18", n_series: 255, bytes: 868119 },
  jodi: { status: "fresh", origin: "refreshed", built: "2026-11-25T02:39:00+00:00", reason: "served copy is 0.0 h old (under 144 h)" },
}, elapsed_s: 17.5, peak_rss_mb: 58.4 });

test("registry: a daily job due when the window opens, its grace covering the window and the budget; one flag for job and readers", () => {
  const job = JOB_REGISTRY.find((j) => j.id === SNAPSHOT_REFRESH_JOB);
  assert.ok(job && job.kind === "daily" && job.source === "server/snapshotRefresh.ts");
  assert.ok(job!.maxRunMs >= SNAPSHOT_REFRESH_TIMEOUT_MS, "registry maxRunMs must cover the job budget");
  if (job!.kind === "daily") {
    assert.equal(job!.tz, "America/New_York");
    assert.equal(job!.dueMin, SNAPSHOT_REFRESH_WINDOW_START_MIN);
    assert.equal(job!.tradingDaysOnly, false, "CPC publishes every day");
    assert.ok(job!.graceMs >= (SNAPSHOT_REFRESH_WINDOW_END_MIN - SNAPSHOT_REFRESH_WINDOW_START_MIN) * 60_000 + SNAPSHOT_REFRESH_TIMEOUT_MS);
  }
  for (const [v, on] of [[undefined, true], ["1", true], ["0", false], ["off", false], ["", false]] as const) {
    const env = v === undefined ? {} : { VOLTRADE_SNAPSHOT_REFRESH: v };
    assert.equal(snapshotRefreshEnabled(env), on);
    assert.equal(job!.enabledWhen!(env), on);
  }
});

test("window is 01:00-03:00 ET on any day, in winter and summer, and a run started at its end is over before the 04:00 ET retrain", () => {
  assert.equal(inSnapshotRefreshWindow(IN), true);
  assert.equal(inSnapshotRefreshWindow(Date.parse("2026-11-22T07:59:00Z")), true, "Sunday 02:59 ET");
  assert.equal(inSnapshotRefreshWindow(Date.parse("2026-11-25T05:59:00Z")), false);
  assert.equal(inSnapshotRefreshWindow(Date.parse("2026-11-25T08:00:00Z")), false);
  assert.equal(inSnapshotRefreshWindow(Date.parse("2026-07-06T05:20:00Z")), true, "01:20 EDT");
  assert.ok(SNAPSHOT_REFRESH_WINDOW_END_MIN * 60_000 + SNAPSHOT_REFRESH_TIMEOUT_MS <= 4 * 60 * 60_000);
});

test("decide: one attempt per ET day, also across a restart and after an error; skip on low disk", () => {
  assert.equal(decideSnapshotRefresh(IN, { ...base, enabled: false }).action, "idle");
  assert.equal(decideSnapshotRefresh(Date.parse("2026-11-25T15:00:00Z"), base).action, "idle");
  const run = decideSnapshotRefresh(IN, base);
  assert.equal(run.action, "run");
  assert.equal((run as { day: string }).day, "2026-11-25");
  assert.equal(decideSnapshotRefresh(IN, { ...base, lastAttemptDay: "2026-11-25" }).action, "idle");
  assert.equal(decideSnapshotRefresh(IN, { ...base, lastFinishedAt: "2026-11-25T06:05:00+00:00" }).action, "idle", "a new container after the day's run");
  assert.equal(decideSnapshotRefresh(IN, { ...base, lastFinishedAt: "2026-11-24T06:05:00+00:00" }).action, "run", "yesterday's run");
  // 23:30 ET the evening before is the previous ET day
  assert.equal(decideSnapshotRefresh(IN, { ...base, lastFinishedAt: "2026-11-25T04:30:00+00:00" }).action, "run");
  const skip = decideSnapshotRefresh(IN, { ...base, freeBytes: SNAPSHOT_REFRESH_MIN_FREE_BYTES - 1 });
  assert.equal(skip.action, "skip");
  assert.equal(decideSnapshotRefresh(IN, { ...base, freeBytes: null }).action, "run", "an unreadable volume size does not stop the job");
});

test("the command names a file at the repo root (the image ships root *.py, never scripts/)", () => {
  const cmd = snapshotRefreshCmd("1.0.1550");
  assert.match(cmd, /^python3 snapshot_refresh_job\.py --run --nice 19 --max-rss-mb \d+ --code-version 1\.0\.1550$/);
  assert.ok(fs.existsSync(path.join(here, "..", "snapshot_refresh_job.py")));
  assert.match(snapshotRefreshCmd("1.0; rm -rf /"), /--code-version unknown$/);
});

test("summary: a run with failed sources is an error that names them FIRST and still counts what was refreshed", () => {
  const out = parseSnapshotRefreshSummary("noise\n" + REAL_ERROR + "\n");
  assert.equal(out.status, "error");
  assert.equal(out.refreshed, 2);
  assert.ok(out.detail.startsWith("eia error: RuntimeError: every series failed"), out.detail);
  assert.ok(out.detail.indexOf("rail error: ModuleNotFoundError") < out.detail.indexOf("cpc refreshed to 2026-11-22"), out.detail);
  // both failures are inside what the job row keeps
  assert.ok(out.detail.indexOf("rail error") + "rail error: ModuleNotFoundError".length < MAX_DETAIL_CHARS, out.detail);
});

test("summary: a clean run says what each source did", () => {
  const out = parseSnapshotRefreshSummary(REAL_OK);
  assert.equal(out.status, "ok");
  assert.equal(out.refreshed, 2);
  assert.equal(out.detail, "eia refreshed to 2026-11-13; cpc fresh; rail refreshed to 2026-11-18; jodi fresh (17.5s, peak 58.4 MB)");
  assert.ok(out.detail.length <= MAX_DETAIL_CHARS);
});

test("summary: garbage, a bare error and a summary with no sources are errors, never ok", () => {
  assert.equal(parseSnapshotRefreshSummary("Traceback (most recent call last)").status, "error");
  const bare = parseSnapshotRefreshSummary(JSON.stringify({ status: "error", reason: "OSError: read-only volume" }));
  assert.deepEqual(bare, { status: "error", refreshed: 0, detail: "OSError: read-only volume" });
  assert.equal(parseSnapshotRefreshSummary(JSON.stringify({ status: "ok", sources: {} })).status, "error");
  assert.equal(describeSource("rail", { status: "rejected", reason: "250 series against 255 served" }), "rail rejected: 250 series against 255 served");
});

test("run: exit 1 with a full summary on stdout is that summary; a kill and a crash are reported as such", async () => {
  let seen: { cmd: string; opts: Record<string, unknown> } | null = null;
  const okExec: WalkforwardExec = async (cmd, opts) => { seen = { cmd, opts: opts as unknown as Record<string, unknown> }; return { stdout: REAL_OK, stderr: "" }; };
  assert.equal((await runSnapshotRefresh({ A: "1" } as NodeJS.ProcessEnv, okExec)).status, "ok");
  assert.match(seen!.cmd, /^python3 snapshot_refresh_job\.py /);
  assert.equal(seen!.opts.timeout, SNAPSHOT_REFRESH_TIMEOUT_MS);
  for (const [k, v] of Object.entries(WALKFORWARD_ENV_CAPS)) assert.equal((seen!.opts.env as Record<string, string>)[k], v);
  const exit1: WalkforwardExec = async () => { throw Object.assign(new Error("Command failed"), { code: 1, stdout: REAL_ERROR, stderr: "" }); };
  const partial = await runSnapshotRefresh({} as NodeJS.ProcessEnv, exit1);
  assert.equal(partial.status, "error");
  assert.equal(partial.refreshed, 2, "the two sources that did refresh must still be adopted");
  const killed: WalkforwardExec = async () => { throw Object.assign(new Error("killed"), { killed: true, signal: "SIGKILL" }); };
  assert.match((await runSnapshotRefresh({} as NodeJS.ProcessEnv, killed)).detail, /20-minute budget/);
  const crash: WalkforwardExec = async () => { throw Object.assign(new Error("x"), { code: 2, stderr: "python3: can't open file 'snapshot_refresh_job.py'" }); };
  const c = await runSnapshotRefresh({} as NodeJS.ProcessEnv, crash);
  assert.equal(c.status, "error");
  assert.match(c.detail, /code=2 .*can't open file/);
});

test("the Python's source list, this module's and the readers' four tables are the same four", () => {
  const py = fs.readFileSync(path.join(here, "..", "snapshot_refresh_job.py"), "utf8");
  const rows = [...py.matchAll(/\{"key": "(\w+)", "module": "\w+", "rel": "([^"]+)"/g)].map((m) => [m[1], m[2]]);
  assert.deepEqual(rows.map((r) => r[0]), [...SNAPSHOT_REFRESH_SOURCES]);
  assert.deepEqual(rows.map((r) => r[1]).sort(), REFRESHABLE_TABLES.map((t) => t.rel).sort());
  assert.match(py, new RegExp(`OUT_SUBDIR = "${SNAPSHOT_REFRESH_SUBDIR}"`));
  // no other table reads the volume
  const refreshable = new Set(REFRESHABLE_TABLES);
  assert.equal(Object.values(LAZY_TABLES).filter((t) => refreshable.has(t)).length, 4);
});

// ── the reader rule ──────────────────────────────────────────────────────────
const REL = "datacore/eia/weekly_series.json";
function volume(): { env: Record<string, string>; file: string } {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "snaprefresh-"));
  const env = { DATA_DIR: dir };
  const file = path.join(snapshotRefreshRoot(env), REL);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  return { env, file };
}
const bundled = JSON.parse(fs.readFileSync(path.join(here, "..", REL), "utf8")) as { built: string };

test("pick: the refreshed copy only when both carry a build time and its own is strictly later", () => {
  assert.equal(pickSnapshotOrigin(1000, 2000), "refreshed");
  assert.equal(pickSnapshotOrigin(2000, 2000), "bundled");
  assert.equal(pickSnapshotOrigin(3000, 2000), "bundled", "a deploy that bundles a newer file wins back");
  assert.equal(pickSnapshotOrigin(1000, NaN), "bundled");
  assert.equal(pickSnapshotOrigin(NaN, 2000), "bundled");
  assert.equal(snapshotBuiltMs({ built_at: "2026-11-02T13:40:01.200686+00:00" }), Date.parse("2026-11-02T13:40:01.200686+00:00"), "JODI's field");
  assert.ok(Number.isNaN(snapshotBuiltMs({})) && Number.isNaN(snapshotBuiltMs(null)));
});

test("reader: no volume copy serves the bundled file, deep-equal to it", () => {
  const { env } = volume();
  const t = lazyJsonTable(REL, { refreshable: true, env });
  assert.equal(t.origin(), null);
  assert.deepEqual(t.get(), bundled);
  assert.equal(t.origin(), "bundled");
});

test("reader: a newer volume copy is served after reset(); an older one, a broken one and a stamp-less one are not", () => {
  const { env, file } = volume();
  const t = lazyJsonTable(REL, { refreshable: true, env });
  t.get();
  const newer = { ...bundled, built: new Date(Date.parse(bundled.built) + 86_400_000).toISOString(), marker: "volume" };
  fs.writeFileSync(file, JSON.stringify(newer));
  assert.equal(t.origin(), "bundled", "a loaded table is not re-read by itself");
  t.reset();
  assert.equal((t.get() as { marker?: string }).marker, "volume");
  assert.equal(t.origin(), "refreshed");
  assert.equal(t.bytes(), fs.statSync(file).size);

  fs.writeFileSync(file, JSON.stringify({ ...newer, built: new Date(Date.parse(bundled.built) - 1000).toISOString() }));
  t.reset();
  assert.deepEqual(t.get(), bundled);
  assert.equal(t.origin(), "bundled");

  const errs: unknown[] = [];
  const orig = console.error;
  console.error = (...a: unknown[]) => { errs.push(a.join(" ")); };
  try {
    fs.writeFileSync(file, "{ truncated");
    t.reset();
    assert.deepEqual(t.peek(), bundled);
    assert.equal(t.origin(), "bundled");
  } finally { console.error = orig; }
  assert.ok(errs.some((e) => String(e).includes("not valid JSON")), "a broken copy is said, not swallowed");

  fs.writeFileSync(file, JSON.stringify({ series: {} }));
  t.reset();
  assert.deepEqual(t.get(), bundled);
});

test("reader: the flag off ignores a newer volume copy (the rollback), and a table not marked refreshable never reads the volume", () => {
  const { env, file } = volume();
  fs.writeFileSync(file, JSON.stringify({ ...bundled, built: "2099-01-01T00:00:00+00:00", marker: "volume" }));
  const off = lazyJsonTable(REL, { refreshable: true, env: { ...env, VOLTRADE_SNAPSHOT_REFRESH: "0" } });
  assert.deepEqual(off.get(), bundled);
  assert.equal(off.origin(), "bundled");
  const plain = lazyJsonTable(REL, { env });
  assert.deepEqual(plain.get(), bundled);
  const on = lazyJsonTable(REL, { refreshable: true, env });
  assert.equal((on.get() as { marker?: string }).marker, "volume");
});

test("adopt: the four tables are dropped and the Streams inventory's stamps are taken again", () => {
  for (const t of REFRESHABLE_TABLES) t.get();
  const before = snapshotStreams();
  adoptRefreshedSnapshots();
  for (const t of REFRESHABLE_TABLES) assert.equal(t.loaded(), false);
  const after = snapshotStreams();
  assert.notEqual(after, before, "a fresh stamp object, not the memo");
  assert.deepEqual(Object.keys(after), Object.keys(before));
  resetSnapshotStreams();
});

test("last_run.json: its finished_at is what a restarted container reads; absent is null", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "snaprefresh-last-"));
  assert.equal(snapshotRefreshLastFinishedAt(dir), null);
  fs.writeFileSync(path.join(dir, "last_run.json"), REAL_OK);
  assert.equal(snapshotRefreshLastFinishedAt(dir), "2026-11-25T02:39:25+00:00");
});
