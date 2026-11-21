/**
 * xbrlFrames.ts — the nightly SEC XBRL fundamentals record (RAW).
 *
 * Once a night (02:00-05:00 ET) this reads SEC's XBRL "frames" for a fixed
 * list of twelve reporting concepts over the last seven calendar quarters
 * (and, for income-statement concepts, the last two full years), compares
 * each frame with what the last read held, and appends what is new or
 * changed to an append-only day file. The rules for what is a row are in
 * server/xbrlFramesCore.ts, with the measurement that motivates them.
 *
 * ON DISK, under <archive>/xbrlframes/:
 *   YYYY-MM-DD.jsonl[.gz]      the record: one line per first sighting,
 *                              revision or drop, and one `run` line per
 *                              night (so a night with no change is a line
 *                              that says so, not a missing file);
 *   _snap_<frame>.json.gz      what the last read held for one frame;
 *   _entities.json.gz          CIK -> [name, state/country] as SEC printed it;
 *   _run.json                  the last night's summary and the newest
 *                              revisions (what the page and the probe read).
 *
 * COST, measured 2026-11-21 from a session: 72 frames were 7.1 MB on the
 * wire and 241,402 facts; a night is about 100 requests, one at a time with
 * a pause, against SEC's limit of 10 a second. The first night writes the
 * whole backfill (about 35 MB before compression, about 6 MB after two
 * days); later nights write only changes. Nothing is held in memory between
 * runs: each frame is read, compared and released.
 *
 * NOT IN THE TRADING PATH. No score, sizing or order reads this. The job
 * runs at night, yields between frames, and stops when the volume is under
 * its free-space floor.
 */
import fs from "fs";
import path from "path";
import zlib from "zlib";
import { pipeline } from "stream/promises";
import type { Express, Request, Response } from "express";
import { archiveBaseDir } from "./datacoreArchive";
import { MIN_FREE_BYTES, readFreeBytes, volumeAllowsWrite } from "./globalScopes";
import { beat } from "./jobHeartbeat";
import { getCikTickerMap } from "./sec8kEarnings";
import {
  FrameShrunkError, XBRL_CONCEPTS, diffFrame, dueNow, etDay, filingUrl, frameFileName, frameRoster, parseFrame,
  type DiffStats, type FrameEntity, type FrameRef, type FrameSnapshot, type XbrlRow,
} from "./xbrlFramesCore";

export const XBRL_STREAM = "xbrlframes";
export const XBRL_WINDOW_START_HOUR_ET = 2;
export const XBRL_WINDOW_END_HOUR_ET = 5;
/** A night interrupted by a restart is tried once more inside the window. */
export const XBRL_MAX_ATTEMPTS_PER_DAY = 2;
export const XBRL_FRAME_PAUSE_MS = 300;
/** Failed frames a night may have and still count as a good run of the job. */
export const XBRL_MAX_FAILED_FRAMES_OK = 5;
export const XBRL_ATTEMPT_TIMEOUTS_MS: readonly number[] = [20_000, 30_000, 45_000];
export const XBRL_RECENT_REVISIONS = 100;
const SEC_UA = { "User-Agent": "voltradeai-datacore/1.0 (research@voltradeai.com)", "Accept-Encoding": "gzip" };
const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

export function xbrlFramesEnabled(env: NodeJS.ProcessEnv = process.env): boolean {
  const raw = env.VOLTRADE_XBRL_FRAMES;
  const v = (raw === undefined ? "1" : raw).trim().toLowerCase();
  return v === "1" || v === "true" || v === "on" || v === "yes";
}

export function xbrlDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "xbrlframes");
}

// ── files ──────────────────────────────────────────────────────────────────

function readGzJson<T>(fp: string): T | null {
  let buf: Buffer;
  try { buf = fs.readFileSync(fp); } catch (e: unknown) {
    if ((e as NodeJS.ErrnoException)?.code === "ENOENT") return null;
    throw e;
  }
  return JSON.parse(zlib.gunzipSync(buf).toString("utf8")) as T;
}

function writeAtomic(fp: string, data: Buffer | string): void {
  fs.writeFileSync(fp + ".tmp", data);
  fs.renameSync(fp + ".tmp", fp); // a crash never leaves half a file
}

function snapPath(dir: string, key: string): string {
  return path.join(dir, `_snap_${frameFileName(key)}.json.gz`);
}

/** What the last read held for a frame, or null when it was never read. A
 *  snapshot that cannot be read is an error: treating it as "never read"
 *  would write the whole frame again as a backfill. */
export function loadSnapshot(dir: string, key: string): FrameSnapshot | null {
  return readGzJson<FrameSnapshot>(snapPath(dir, key));
}

export function saveSnapshot(dir: string, key: string, snap: FrameSnapshot): void {
  writeAtomic(snapPath(dir, key), zlib.gzipSync(JSON.stringify(snap)));
}

type EntityTable = Record<string, [string, string | null]>;

function loadEntities(dir: string): EntityTable {
  return readGzJson<EntityTable>(path.join(dir, "_entities.json.gz")) ?? {};
}

/** Compresses day files older than two days, as a stream: the first night's
 *  file is about 58 MB, too large to compress in one blocking call. */
export async function gzipOldDays(dir: string, nowMs: number): Promise<number> {
  let n = 0;
  for (const f of fs.readdirSync(dir)) {
    if (f.startsWith("_") || !f.endsWith(".jsonl")) continue;
    if (nowMs - Date.parse(f.slice(0, 10)) < 2 * 86_400_000) continue;
    const fp = path.join(dir, f);
    await pipeline(fs.createReadStream(fp), zlib.createGzip(), fs.createWriteStream(`${fp}.gz.tmp`));
    fs.renameSync(`${fp}.gz.tmp`, `${fp}.gz`);
    fs.unlinkSync(fp);
    n++;
  }
  return n;
}

// ── the request ────────────────────────────────────────────────────────────

export interface XbrlFetchResponse { ok: boolean; status: number; text(): Promise<string> }
export type XbrlFetchFn = (url: string, init: { headers: Record<string, string>; signal: AbortSignal }) => Promise<XbrlFetchResponse>;

/** SEC refused the request (its fair-access limit, or the address). Asking
 *  again changes neither, so the night stops. */
export class XbrlRefusedError extends Error {
  constructor(public readonly status: number, url: string) {
    super(`xbrl frames: HTTP ${status} from ${url}`);
    this.name = "XbrlRefusedError";
  }
}

/** One frame's answer as parsed JSON, or null when SEC has no such frame
 *  yet (HTTP 404: no company has reported into the period). A request that
 *  threw or answered 5xx is asked again, each attempt with a longer limit. */
export async function fetchFrame(fetchImpl: XbrlFetchFn, url: string, opts: { timeoutsMs?: readonly number[]; pauseMs?: number } = {}): Promise<unknown | null> {
  const timeouts = opts.timeoutsMs ?? XBRL_ATTEMPT_TIMEOUTS_MS;
  let last: unknown = new Error("xbrl frames: no attempt made");
  for (let i = 0; i < timeouts.length; i++) {
    if (i > 0) await sleep(opts.pauseMs ?? 500);
    try {
      const res = await fetchImpl(url, { headers: SEC_UA, signal: AbortSignal.timeout(timeouts[i]) });
      if (res.status === 404) return null;
      if (res.status >= 500) { last = new Error(`xbrl frames: HTTP ${res.status} from ${url}`); continue; }
      if (!res.ok) throw new XbrlRefusedError(res.status, url);
      return JSON.parse(await res.text()) as unknown;
    } catch (e: unknown) {
      if (e instanceof XbrlRefusedError) throw e;
      last = e;
    }
  }
  throw last instanceof Error ? last : new Error(String(last));
}

// ── one night ──────────────────────────────────────────────────────────────

export interface FrameRunRow extends DiffStats { frame: string; tag: string; period: string; state: "read" | "not_published" | "failed"; error?: string; rejected?: number }

export interface XbrlRevision extends XbrlRow { name: string | null; ticker: string | null }

export interface XbrlRunSummary {
  day: string;
  started: string;
  finished: string | null;
  status: "ok" | "partial" | "refused" | "skipped_low_disk" | "running";
  detail: string;
  frames: { roster: number; read: number; not_published: number; failed: number };
  rows: { first: number; revised: number; dropped: number; refiled: number; backfill: number };
  companies: number;
  per_frame: FrameRunRow[];
}

interface RunFile { attempt_day: string | null; attempts: number; completed_day: string | null; last: XbrlRunSummary | null; recent_revisions: XbrlRevision[]; nights: number }

function runPath(dir: string): string { return path.join(dir, "_run.json"); }

export function loadRunFile(dir: string): RunFile {
  const empty: RunFile = { attempt_day: null, attempts: 0, completed_day: null, last: null, recent_revisions: [], nights: 0 };
  let text: string;
  try { text = fs.readFileSync(runPath(dir), "utf8"); } catch (e: unknown) {
    if ((e as NodeJS.ErrnoException)?.code === "ENOENT") return empty;
    throw e;
  }
  return { ...empty, ...(JSON.parse(text) as Partial<RunFile>) };
}

export interface XbrlNightDeps {
  fetch: XbrlFetchFn;
  baseDir?: string;
  now?: () => Date;
  pauseMs?: number;
  timeoutsMs?: readonly number[];
  retryPauseMs?: number;
  freeBytes?: (dir: string) => number | null;
  tickers?: () => Promise<Map<string, string>>;
  roster?: (now: Date) => FrameRef[];
}

/** Reads every frame on the roster once and records the differences. */
export async function runXbrlNight(deps: XbrlNightDeps): Promise<XbrlRunSummary> {
  const now = deps.now ?? (() => new Date());
  const startedAt = now();
  const seen = startedAt.toISOString();
  const day = seen.slice(0, 10);
  const dir = xbrlDir(deps.baseDir);
  fs.mkdirSync(dir, { recursive: true });
  const roster = (deps.roster ?? frameRoster)(startedAt);
  const summary: XbrlRunSummary = {
    day, started: seen, finished: null, status: "running", detail: "",
    frames: { roster: roster.length, read: 0, not_published: 0, failed: 0 },
    rows: { first: 0, revised: 0, dropped: 0, refiled: 0, backfill: 0 },
    companies: 0, per_frame: [],
  };
  const runFile = loadRunFile(dir);

  const free = (deps.freeBytes ?? readFreeBytes)(dir);
  if (!volumeAllowsWrite(free, MIN_FREE_BYTES)) {
    summary.status = "skipped_low_disk";
    summary.finished = now().toISOString();
    summary.detail = `not read: ${Math.round((free ?? 0) / 1_048_576)} MB free on the volume, under the ${Math.round(MIN_FREE_BYTES / 1_048_576)} MB floor`;
    writeAtomic(runPath(dir), JSON.stringify({ ...runFile, last: summary }));
    return summary;
  }

  const entities = loadEntities(dir);
  const dayFile = path.join(dir, `${day}.jsonl`);
  const revised: XbrlRow[] = [];
  let refused: XbrlRefusedError | null = null;

  for (let i = 0; i < roster.length; i++) {
    const ref = roster[i];
    if (i > 0) await sleep(deps.pauseMs ?? XBRL_FRAME_PAUSE_MS);
    const line: FrameRunRow = { frame: ref.key, tag: ref.concept.tag, period: ref.period, state: "read", held: 0, first: 0, revised: 0, dropped: 0, refiled: 0, backfill: false };
    try {
      const prev = loadSnapshot(dir, ref.key);
      const body = await fetchFrame(deps.fetch, ref.url, { timeoutsMs: deps.timeoutsMs, pauseMs: deps.retryPauseMs });
      if (body === null) {
        if (prev && Object.keys(prev).length > 0) throw new Error(`xbrl frame ${ref.key}: HTTP 404 for a frame that held ${Object.keys(prev).length} companies`);
        if (!prev) saveSnapshot(dir, ref.key, {}); // so its first companies are real first sightings
        line.state = "not_published";
        summary.frames.not_published++;
      } else {
        const parsed = parseFrame(body);
        const diff = diffFrame(ref.key, prev, parsed.facts, seen);
        if (diff.rows.length) fs.appendFileSync(dayFile, diff.rows.map((r) => JSON.stringify(r)).join("\n") + "\n");
        saveSnapshot(dir, ref.key, diff.next); // after the rows: a crash between the two repeats rows, never loses them
        for (const [cik, e] of parsed.entities) entities[String(cik)] = [e.name, e.loc];
        Object.assign(line, diff.stats);
        if (parsed.rejected) line.rejected = parsed.rejected;
        summary.frames.read++;
        summary.rows.first += diff.stats.first;
        summary.rows.revised += diff.stats.revised;
        summary.rows.dropped += diff.stats.dropped;
        summary.rows.refiled += diff.stats.refiled;
        if (diff.stats.backfill) summary.rows.backfill += diff.stats.first;
        for (const r of diff.rows) if (r.kind === "revised") revised.push(r);
      }
    } catch (e: unknown) {
      line.state = "failed";
      line.error = e instanceof Error ? e.message : String(e);
      summary.frames.failed++;
      console.error(`[datacore] xbrlframes ${ref.key}: ${line.error}${e instanceof FrameShrunkError ? " (shrink guard)" : ""}`);
      if (e instanceof XbrlRefusedError) refused = e;
    }
    summary.per_frame.push(line);
    if (refused) break;
  }

  summary.companies = Object.keys(entities).length;
  summary.finished = now().toISOString();
  summary.status = refused ? "refused" : summary.frames.failed > 0 ? "partial" : "ok";
  const written = summary.rows.first + summary.rows.revised + summary.rows.dropped;
  summary.detail = `${summary.frames.read} of ${summary.frames.roster} frames read`
    + (summary.frames.not_published ? `, ${summary.frames.not_published} not published yet` : "")
    + (summary.frames.failed ? `, ${summary.frames.failed} failed` : "")
    + `; ${written} rows (${summary.rows.first} first, ${summary.rows.revised} revised, ${summary.rows.dropped} dropped)`
    + (summary.rows.backfill ? `, ${summary.rows.backfill} of the first rows are the backfill` : "")
    + (refused ? `; stopped: SEC answered HTTP ${refused.status}` : "");

  // The night's own line: a night with no change is still on the record.
  const runRow = { seen, kind: "run", status: summary.status, frames: summary.frames, rows: summary.rows, finished: summary.finished };
  fs.appendFileSync(dayFile, JSON.stringify(runRow) + "\n");
  writeAtomic(path.join(dir, "_entities.json.gz"), zlib.gzipSync(JSON.stringify(entities)));

  let tickers = new Map<string, string>();
  if (revised.length) tickers = await (deps.tickers ?? (() => getCikTickerMap()))();
  const newRevisions: XbrlRevision[] = revised.map((r) => ({ ...r, name: entities[String(r.cik)]?.[0] ?? null, ticker: tickers.get(String(r.cik)) ?? null }));
  const completed = summary.status === "ok" || summary.status === "partial";
  const out: RunFile = {
    ...loadRunFile(dir),
    completed_day: completed ? etDay(startedAt) : runFile.completed_day,
    last: summary,
    recent_revisions: [...newRevisions.reverse(), ...runFile.recent_revisions].slice(0, XBRL_RECENT_REVISIONS),
    nights: runFile.nights + (completed ? 1 : 0),
  };
  writeAtomic(runPath(dir), JSON.stringify(out));
  await gzipOldDays(dir, startedAt.getTime());
  return summary;
}

// ── what the page and the probe read ───────────────────────────────────────

const SOURCE = "SEC EDGAR XBRL frames API (data.sec.gov/api/xbrl/frames)";
const ATTRIBUTION = "U.S. Securities and Exchange Commission, EDGAR";
const NOTE = "Numbers are the values companies tagged in their own 10-Q and 10-K filings, as SEC's frames API served them on our nightly read. "
  + "\"Seen\" is the night our read first held a value, to one day; it is not SEC's acceptance time. "
  + "Rows from the first read of a frame are a backfill: their date is when recording began. "
  + "A revision is a later filing reporting a different number for the same period. Raw record; no score and no forecast.";

export function xbrlFramesStatus(baseDir?: string, env: NodeJS.ProcessEnv = process.env): Record<string, unknown> {
  const dir = xbrlDir(baseDir);
  const run = loadRunFile(dir);
  return {
    probe: "xbrl_frames",
    enabled: xbrlFramesEnabled(env),
    rule: `one read a night, ${XBRL_WINDOW_START_HOUR_ET}:00-${XBRL_WINDOW_END_HOUR_ET}:00 ET; ${XBRL_CONCEPTS.length} concepts; a frame that lost more than half its companies is refused`,
    attempt_day: run.attempt_day, attempts: run.attempts, completed_day: run.completed_day, nights: run.nights,
    last: run.last,
    recent_revisions: run.recent_revisions.length,
    free_bytes: readFreeBytes(fs.existsSync(dir) ? dir : path.dirname(dir)),
  };
}

export interface XbrlPublicView {
  kind: "raw"; source: string; attribution: string; license: string; note: string;
  warming_up: boolean;
  concepts: Array<{ tag: string; label: string; unit: string; kind: string }>;
  last_run: null | { day: string; started: string; finished: string | null; status: string; detail: string; frames: XbrlRunSummary["frames"]; rows: XbrlRunSummary["rows"]; companies: number };
  nights: number;
  coverage: Array<{ tag: string; period: string; state: string; held: number; first: number; revised: number; dropped: number; backfill: boolean }>;
  revisions: Array<{ seen: string; tag: string; period: string; cik: number; name: string | null; ticker: string | null; end: string; val: number | null; prev_val: number | null; unit: string; filing: string; prev_filing: string | null }>;
}

export function xbrlPublicView(baseDir?: string): XbrlPublicView {
  const run = loadRunFile(xbrlDir(baseDir));
  const unitOf = new Map(XBRL_CONCEPTS.map((c) => [c.tag, c.unit] as const));
  const last = run.last;
  return {
    kind: "raw", source: SOURCE, attribution: ATTRIBUTION, license: "US government work, public domain", note: NOTE,
    warming_up: !last || last.status === "running",
    concepts: XBRL_CONCEPTS.map((c) => ({ tag: c.tag, label: c.label, unit: c.unit, kind: c.kind })),
    last_run: last ? { day: last.day, started: last.started, finished: last.finished, status: last.status, detail: last.detail, frames: last.frames, rows: last.rows, companies: last.companies } : null,
    nights: run.nights,
    coverage: (last?.per_frame ?? []).map((f) => ({ tag: f.tag, period: f.period, state: f.state, held: f.held, first: f.first, revised: f.revised, dropped: f.dropped, backfill: f.backfill })),
    revisions: run.recent_revisions.map((r) => {
      const [, tag, , period] = r.frame.split("/");
      return {
        seen: r.seen, tag, period, cik: r.cik, name: r.name, ticker: r.ticker, end: r.end, val: r.val, prev_val: r.prev_val ?? null,
        unit: unitOf.get(tag) ?? "USD", filing: filingUrl(r.cik, r.accn), prev_filing: r.prev_accn ? filingUrl(r.cik, r.prev_accn) : null,
      };
    }),
  };
}

export interface XbrlCompanyView {
  kind: "raw"; source: string; attribution: string; note: string;
  cik: number; name: string | null; loc: string | null; ticker: string | null;
  facts: Array<{ tag: string; label: string; unit: string; period: string; end: string; val: number; filing: string }>;
}

/** What the last read holds for one company, across every frame. Reads one
 *  snapshot per turn of the event loop. */
export async function xbrlCompanyView(cik: number, baseDir?: string, now: Date = new Date(), ticker: string | null = null): Promise<XbrlCompanyView> {
  const dir = xbrlDir(baseDir);
  const id = String(cik);
  const facts: XbrlCompanyView["facts"] = [];
  for (const ref of frameRoster(now)) {
    let buf: Buffer;
    try { buf = await fs.promises.readFile(snapPath(dir, ref.key)); } catch (e: unknown) {
      if ((e as NodeJS.ErrnoException)?.code === "ENOENT") continue;
      throw e;
    }
    const held = (JSON.parse(zlib.gunzipSync(buf).toString("utf8")) as FrameSnapshot)[id];
    if (held) facts.push({ tag: ref.concept.tag, label: ref.concept.label, unit: ref.concept.unit, period: ref.period, end: held[2], val: held[0], filing: filingUrl(cik, held[1]) });
  }
  const entity: FrameEntity | null = (() => { const e = loadEntities(dir)[id]; return e ? { name: e[0], loc: e[1] } : null; })();
  return { kind: "raw", source: SOURCE, attribution: ATTRIBUTION, note: NOTE, cik, name: entity?.name ?? null, loc: entity?.loc ?? null, ticker, facts };
}

/** `?q=` on the company route: a CIK (digits) or a ticker. */
export async function resolveCompanyQuery(q: string, tickers: () => Promise<Map<string, string>> = () => getCikTickerMap()): Promise<{ cik: number; ticker: string | null } | null> {
  const s = q.trim().toUpperCase();
  if (!s || s.length > 12) return null;
  const map = await tickers();
  if (/^\d{1,10}$/.test(s)) {
    const cik = Number(s);
    return cik > 0 ? { cik, ticker: map.get(String(cik)) ?? null } : null;
  }
  if (!/^[A-Z][A-Z0-9.\-]{0,9}$/.test(s)) return null;
  for (const [cik, t] of map) if (t.toUpperCase() === s) return { cik: Number(cik), ticker: t };
  return null;
}

// ── schedule ───────────────────────────────────────────────────────────────

let booted = false;
let timer: ReturnType<typeof setInterval> | null = null;
const TICK_MS = 20 * 60_000;

/** Whether a tick at `at` should start the night, from the persisted
 *  markers: inside the window, not completed for this ET day, and tried
 *  fewer than twice. */
export function xbrlNightDue(at: Date, run: Pick<RunFile, "attempt_day" | "attempts" | "completed_day">): boolean {
  if (!dueNow(at, run.completed_day, XBRL_WINDOW_START_HOUR_ET, XBRL_WINDOW_END_HOUR_ET)) return false;
  return !(run.attempt_day === etDay(at) && run.attempts >= XBRL_MAX_ATTEMPTS_PER_DAY);
}

export async function xbrlTick(deps: XbrlNightDeps, env: NodeJS.ProcessEnv = process.env): Promise<XbrlRunSummary | null> {
  if (!xbrlFramesEnabled(env)) return null;
  const at = (deps.now ?? (() => new Date()))();
  const dir = xbrlDir(deps.baseDir);
  const run = loadRunFile(dir);
  if (!xbrlNightDue(at, run)) return null;
  const today = etDay(at);
  fs.mkdirSync(dir, { recursive: true });
  writeAtomic(runPath(dir), JSON.stringify({ ...run, attempt_day: today, attempts: run.attempt_day === today ? run.attempts + 1 : 1 }));
  beat("xbrl_frames_nightly", "start");
  try {
    const summary = await runXbrlNight(deps);
    // A few failed frames are read again the next night and are named in the detail; more than that is an error.
    const bad = summary.status === "refused" || summary.frames.failed > XBRL_MAX_FAILED_FRAMES_OK;
    beat("xbrl_frames_nightly", "finish", summary.status === "skipped_low_disk" ? "skipped" : bad ? "error" : "ok", summary.detail);
    return summary;
  } catch (e: unknown) {
    beat("xbrl_frames_nightly", "finish", "error", e instanceof Error ? e.message : String(e));
    throw e;
  }
}

/** 20-minute tick (unref'd; a boot never starts a read by itself). */
export function bootXbrlFrames(env: NodeJS.ProcessEnv = process.env): void {
  if (booted) return;
  booted = true;
  let running = false;
  timer = setInterval(() => {
    if (running) return;
    running = true;
    xbrlTick({ fetch: fetch as unknown as XbrlFetchFn }, env)
      .catch((e: unknown) => { console.error("[datacore] xbrlframes night:", e instanceof Error ? e.message : e); })
      .finally(() => { running = false; });
  }, TICK_MS);
  timer.unref?.();
}

/** Test-only: stop the tick and allow a fresh boot. */
export function _resetXbrlFramesForTests(): void {
  if (timer) clearInterval(timer);
  timer = null;
  booted = false;
}

export function registerXbrlFramesRoutes(app: Express, env: NodeJS.ProcessEnv = process.env): void {
  app.get("/api/data/xbrl-fundamentals", (_req: Request, res: Response) => {
    try { res.json(xbrlPublicView()); } catch (e: unknown) {
      console.error("[datacore] xbrlframes view:", e instanceof Error ? e.message : e);
      res.status(500).json({ error: "could not read the fundamentals record" });
    }
  });
  app.get("/api/data/xbrl-fundamentals/company", async (req: Request, res: Response) => {
    try {
      const q = typeof req.query.q === "string" ? req.query.q : "";
      const hit = await resolveCompanyQuery(q);
      if (!hit) return res.status(404).json({ error: "no company for that ticker or CIK" });
      res.json(await xbrlCompanyView(hit.cik, undefined, new Date(), hit.ticker));
    } catch (e: unknown) {
      console.error("[datacore] xbrlframes company:", e instanceof Error ? e.message : e);
      res.status(500).json({ error: "could not read the fundamentals record" });
    }
  });
  bootXbrlFrames(env);
}
