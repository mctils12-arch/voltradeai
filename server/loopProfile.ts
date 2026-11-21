// LOOP PROFILE — what was running while the event loop was blocked
// ([REPAIR] 2026-11-21, measurement only).
//
// WHY: server/eventLoopLag.ts has written an EVENTLOOP-LAG audit row for
// every stall of 500 ms or more since 2026-07-12, and a row says only how
// long the loop was blocked. Read 2026-11-21 (a Saturday, no Tier-2 scans):
// 70 of the last 200 audit rows were stalls of 0.5 to 3.5 s, most of them
// 5 or 10 minutes apart at a phase that restarts with each boot. Two earlier
// causes were found by reading code and guessing (the SQLite write theory,
// refuted; the DTCC index load, right). This module replaces the guess with
// a reading: V8's own sampling CPU profiler, run on demand for a bounded
// time, and a summary that keeps only what was on the stack while the loop
// was blocked.
//
// WHAT IT DOES: nothing until asked. GET /api/diag/loop_profile?start=1
// (diag token) starts one run: the profiler samples the main thread every
// `sample_ms`, and a 200 ms tick of this module's own records every stretch
// in which the loop could not run it. At the end of the run the profile is
// reduced to (a) each blocked stretch with the frames sampled inside it,
// (b) the same over all blocked stretches, by frame on the stack and by the
// frame on top, and (c) the whole run's busiest frames. The production
// bundle is minified, so every frame also carries the first characters of
// its function's text read from the running file: property names and string
// literals survive minification and can be searched in the source.
//
// COST while a run is on: the sampler's interrupt every `sample_ms`, and two
// blocks of the loop this module causes itself, one when the profiler starts
// (V8 lists the compiled code) and one when it stops (the profile is built):
// both are reported in the result as `own_start_ms` / `own_stop_ms`. One run
// at a time, 15 minutes at most, the last result held in memory until the
// next run or a restart. Nothing here is read by a trading path.
//
// SWITCH: VOLTRADE_LOOP_PROFILE=off refuses every start (no deploy needed).

import inspector from "node:inspector";
import fs from "node:fs";
import { fileURLToPath } from "node:url";

export const LOOP_PROFILE_FLAG = "VOLTRADE_LOOP_PROFILE";
export const LOOP_PROFILE_DEFAULT_S = 660; // two 5-minute periods and a margin
export const LOOP_PROFILE_MIN_S = 2;
export const LOOP_PROFILE_MAX_S = 900;
export const LOOP_PROFILE_DEFAULT_SAMPLE_MS = 10;
export const LOOP_PROFILE_MIN_SAMPLE_MS = 2;
export const LOOP_PROFILE_MAX_SAMPLE_MS = 100;
/** This module's own tick while a run is on. */
export const LOOP_PROFILE_TICK_MS = 200;
/** A tick this late is a blocked stretch (eventLoopLag.ts alerts at 500 ms on a 2 s tick). */
export const LOOP_PROFILE_STALL_MS = 400;
/** Blocked stretches listed one by one (the longest); all of them are in the totals. */
export const LOOP_PROFILE_MAX_STALLS = 40;
export const LOOP_PROFILE_TOP = 15;
export const LOOP_PROFILE_SNIPPET_CHARS = 220;
/** Slack on each side of a stretch when matching samples to it: the profile's
 *  clock is mapped to wall time through one reading taken at the start. */
export const LOOP_PROFILE_WINDOW_SLACK_MS = 30;

export interface CallFrame { functionName: string; url: string; lineNumber: number; columnNumber: number }
export interface ProfileNode { id: number; callFrame: CallFrame; children?: number[] }
export interface CpuProfile { nodes: ProfileNode[]; startTime: number; endTime: number; samples?: number[]; timeDeltas?: number[] }
/** A stretch in which the loop could not run a tick: wall-clock ms. */
export interface BlockedStretch { from: number; to: number; blockedMs: number }

export interface FrameRow { ms: number; fn: string; at: string; text?: string }
export interface StallRow {
  ended: string; blocked_ms: number; sampled_busy_ms: number; gc_ms: number;
  /** the frames on the stack for the longest time inside this stretch, outermost first */
  on_stack: FrameRow[];
  on_top: FrameRow[];
}
export interface LoopProfileSummary {
  run_ms: number; samples: number; busy_ms: number; idle_ms: number; gc_ms: number;
  stalls: { count: number; blocked_ms: number; longest_ms: number; listed: number };
  /** over every blocked stretch: time each frame was anywhere on the stack */
  stalls_on_stack: FrameRow[];
  /** over every blocked stretch: time each frame was the one running */
  stalls_on_top: FrameRow[];
  stall_list: StallRow[];
  /** the whole run, blocked or not: the frames running for the longest time */
  run_on_top: FrameRow[];
}

const IDLE = new Set(["(idle)", "(program)"]);
const GC = "(garbage collector)";
const round = (n: number): number => Math.round(n);

function frameAt(f: CallFrame): string {
  const file = (f.url || "").split(/[\\/]/).pop() || "(native)";
  return f.url ? `${file}:${f.lineNumber + 1}:${f.columnNumber + 1}` : file;
}
function frameKey(f: CallFrame): string { return `${f.functionName || "(anonymous)"} ${frameAt(f)}`; }

/** Clamp a query value into [lo, hi]; anything unreadable is the default. */
export function clampInt(raw: unknown, dflt: number, lo: number, hi: number): number {
  const n = parseInt(String(raw ?? ""), 10);
  if (!Number.isFinite(n)) return dflt;
  return Math.min(hi, Math.max(lo, n));
}

export function loopProfileEnabled(env: NodeJS.ProcessEnv = process.env): boolean {
  return !/^(off|0|false|no)$/i.test(String(env[LOOP_PROFILE_FLAG] ?? "").trim());
}

/** Reduce a V8 CPU profile to what ran inside the blocked stretches.
 *  `wallStartMs` is the wall-clock time of `profile.startTime`; a sample's
 *  wall time is that plus its offset from `profile.startTime`. A sample owns
 *  the time up to the next sample. Pure: no clock, no file. */
export function summarizeProfile(
  profile: CpuProfile, wallStartMs: number, stretches: BlockedStretch[],
  textOf: (f: CallFrame) => string | undefined = () => undefined,
): LoopProfileSummary {
  const byId = new Map<number, ProfileNode>();
  const parent = new Map<number, number>();
  for (const n of profile.nodes) byId.set(n.id, n);
  for (const n of profile.nodes) for (const c of n.children ?? []) parent.set(c, n.id);
  const samples = profile.samples ?? [];
  const deltas = profile.timeDeltas ?? [];
  // start time of each sample in µs on the profile's clock
  const at: number[] = new Array(samples.length);
  let t = profile.startTime;
  for (let i = 0; i < samples.length; i++) { t += deltas[i] ?? 0; at[i] = t; }
  const toWall = (us: number): number => wallStartMs + (us - profile.startTime) / 1000;

  const frames = new Map<string, CallFrame>();
  const stackOf = (id: number): string[] => { // outermost first, "(root)" left out
    const out: string[] = [];
    for (let cur: number | undefined = id; cur !== undefined; cur = parent.get(cur)) {
      const n = byId.get(cur);
      if (!n) break;
      if (n.callFrame.functionName === "(root)") continue;
      const k = frameKey(n.callFrame);
      frames.set(k, n.callFrame);
      out.push(k);
    }
    return out.reverse();
  };
  const row = (k: string, ms: number): FrameRow => {
    const f = frames.get(k) as CallFrame;
    const text = textOf(f);
    return { ms: round(ms), fn: f.functionName || "(anonymous)", at: frameAt(f), ...(text ? { text } : {}) };
  };
  const top = (m: Map<string, number>, n: number): FrameRow[] =>
    Array.from(m.entries()).sort((a, b) => b[1] - a[1]).slice(0, n).filter(([, ms]) => ms >= 1).map(([k, ms]) => row(k, ms));
  const add = (m: Map<string, number>, k: string, ms: number): void => { m.set(k, (m.get(k) ?? 0) + ms); };

  const sorted = stretches.slice().sort((a, b) => a.from - b.from);
  const perStall = sorted.map(() => ({ busy: 0, gc: 0, stack: new Map<string, number>(), topm: new Map<string, number>() }));
  const allStack = new Map<string, number>();
  const allTop = new Map<string, number>();
  const runTop = new Map<string, number>();
  let busy = 0, idle = 0, gc = 0, si = 0;

  for (let i = 0; i < samples.length; i++) {
    const endUs = i + 1 < samples.length ? at[i + 1] : Math.max(profile.endTime, at[i]);
    const ms = (endUs - at[i]) / 1000;
    if (ms <= 0) continue;
    const node = byId.get(samples[i]);
    if (!node) continue;
    const name = node.callFrame.functionName;
    if (IDLE.has(name)) { idle += ms; continue; }
    busy += ms;
    if (name === GC) gc += ms;
    const stack = stackOf(samples[i]);
    const leaf = stack[stack.length - 1];
    if (leaf) add(runTop, leaf, ms);
    const w = toWall(at[i]);
    while (si < sorted.length && sorted[si].to + LOOP_PROFILE_WINDOW_SLACK_MS < w) si++;
    const s = sorted[si];
    if (!s || w < s.from - LOOP_PROFILE_WINDOW_SLACK_MS) continue;
    const p = perStall[si];
    p.busy += ms;
    if (name === GC) p.gc += ms;
    for (const k of new Set(stack)) { add(p.stack, k, ms); add(allStack, k, ms); }
    if (leaf) { add(p.topm, leaf, ms); add(allTop, leaf, ms); }
  }

  const listed = sorted.map((s, i) => ({ s, p: perStall[i] }))
    .sort((a, b) => b.s.blockedMs - a.s.blockedMs).slice(0, LOOP_PROFILE_MAX_STALLS)
    .sort((a, b) => a.s.to - b.s.to);
  return {
    run_ms: round((profile.endTime - profile.startTime) / 1000),
    samples: samples.length, busy_ms: round(busy), idle_ms: round(idle), gc_ms: round(gc),
    stalls: {
      count: sorted.length,
      blocked_ms: round(sorted.reduce((a, s) => a + s.blockedMs, 0)),
      longest_ms: round(sorted.reduce((a, s) => Math.max(a, s.blockedMs), 0)),
      listed: listed.length,
    },
    stalls_on_stack: top(allStack, LOOP_PROFILE_TOP),
    stalls_on_top: top(allTop, LOOP_PROFILE_TOP),
    stall_list: listed.map(({ s, p }) => ({
      ended: new Date(s.to).toISOString(), blocked_ms: round(s.blockedMs),
      sampled_busy_ms: round(p.busy), gc_ms: round(p.gc),
      // outermost first: a frame on the stack for longer sits further out
      on_stack: top(p.stack, 8), on_top: top(p.topm, 4),
    })),
    run_on_top: top(runTop, LOOP_PROFILE_TOP),
  };
}

/** The first characters of a function's text, from the file V8 names for it.
 *  Files are read once per summary; anything unreadable gives no text. */
export function sourceTextReader(maxChars = LOOP_PROFILE_SNIPPET_CHARS): (f: CallFrame) => string | undefined {
  const lineStarts = new Map<string, { text: string; starts: number[] } | null>();
  return (f) => {
    if (!f.url || f.lineNumber < 0) return undefined;
    let file = f.url;
    try { if (file.startsWith("file:")) file = fileURLToPath(file); } catch { return undefined; }
    if (!file.startsWith("/") || /node_modules/.test(file)) return undefined;
    let e = lineStarts.get(file);
    if (e === undefined) {
      try {
        const text = fs.readFileSync(file, "utf8");
        const starts = [0];
        for (let i = text.indexOf("\n"); i !== -1; i = text.indexOf("\n", i + 1)) starts.push(i + 1);
        e = { text, starts };
      } catch { e = null; }
      lineStarts.set(file, e);
    }
    if (!e || f.lineNumber >= e.starts.length) return undefined;
    const from = e.starts[f.lineNumber] + Math.max(0, f.columnNumber);
    return e.text.slice(from, from + maxChars).replace(/\s+/g, " ").trim() || undefined;
  };
}

// ── the run ────────────────────────────────────────────────────────────────
export interface LoopProfileState {
  state: "idle" | "running" | "done" | "error";
  enabled: boolean;
  started_at: string | null; ends_at: string | null; finished_at: string | null;
  seconds: number | null; sample_ms: number | null;
  stalls_so_far: number;
  /** ms the loop was blocked by starting and by stopping the profiler */
  own_start_ms: number | null; own_stop_ms: number | null;
  error: string | null;
  result: LoopProfileSummary | null;
}

let cur: LoopProfileState = {
  state: "idle", enabled: true, started_at: null, ends_at: null, finished_at: null, seconds: null, sample_ms: null,
  stalls_so_far: 0, own_start_ms: null, own_stop_ms: null, error: null, result: null,
};
let stretches: BlockedStretch[] = [];

export function loopProfileStatus(env: NodeJS.ProcessEnv = process.env): LoopProfileState {
  return { ...cur, enabled: loopProfileEnabled(env), stalls_so_far: cur.state === "running" ? stretches.length : cur.stalls_so_far };
}

function post<T>(session: inspector.Session, method: string, params?: object): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    (session.post as (m: string, p: object | undefined, cb: (e: Error | null, r?: unknown) => void) => void)(
      method, params, (e, r) => (e ? reject(e) : resolve(r as T)));
  });
}

/** Start one run. Returns at once; the result is on loopProfileStatus() when
 *  the run ends. `done` resolves then (tests await it). */
export function startLoopProfile(
  opts: { seconds?: unknown; sampleMs?: unknown } = {}, env: NodeJS.ProcessEnv = process.env,
): { started: boolean; reason: string | null; done: Promise<void> } {
  if (!loopProfileEnabled(env)) return { started: false, reason: `${LOOP_PROFILE_FLAG} is off`, done: Promise.resolve() };
  if (cur.state === "running") return { started: false, reason: `a run is on until ${cur.ends_at}`, done: Promise.resolve() };
  const seconds = clampInt(opts.seconds, LOOP_PROFILE_DEFAULT_S, LOOP_PROFILE_MIN_S, LOOP_PROFILE_MAX_S);
  const sampleMs = clampInt(opts.sampleMs, LOOP_PROFILE_DEFAULT_SAMPLE_MS, LOOP_PROFILE_MIN_SAMPLE_MS, LOOP_PROFILE_MAX_SAMPLE_MS);
  const t0 = Date.now();
  cur = {
    state: "running", enabled: true, started_at: new Date(t0).toISOString(), ends_at: new Date(t0 + seconds * 1000).toISOString(),
    finished_at: null, seconds, sample_ms: sampleMs, stalls_so_far: 0, own_start_ms: null, own_stop_ms: null, error: null, result: null,
  };
  stretches = [];
  const done = runProfile(seconds, sampleMs).catch((e: unknown) => {
    cur = { ...cur, state: "error", finished_at: new Date().toISOString(), error: e instanceof Error ? e.message : String(e), stalls_so_far: stretches.length };
  });
  return { started: true, reason: null, done };
}

async function runProfile(seconds: number, sampleMs: number): Promise<void> {
  const session = new inspector.Session();
  session.connect();
  let tick: NodeJS.Timeout | null = null;
  try {
    await post(session, "Profiler.enable");
    await post(session, "Profiler.setSamplingInterval", { interval: sampleMs * 1000 });
    const beforeStart = Date.now();
    await post(session, "Profiler.start");
    const wallStart = Date.now();
    cur.own_start_ms = wallStart - beforeStart;
    let last = wallStart;
    tick = setInterval(() => {
      const now = Date.now();
      const late = now - last - LOOP_PROFILE_TICK_MS;
      if (late >= LOOP_PROFILE_STALL_MS) stretches.push({ from: last, to: now, blockedMs: late });
      last = now;
    }, LOOP_PROFILE_TICK_MS);
    await new Promise<void>((resolve) => { setTimeout(resolve, seconds * 1000); });
    clearInterval(tick); tick = null;
    const beforeStop = Date.now();
    const { profile } = await post<{ profile: CpuProfile }>(session, "Profiler.stop");
    const ownStop = Date.now() - beforeStop;
    const mine = stretches.slice();
    // The profile's clock is tied to wall time at its END: V8 stamps endTime as
    // the stop is taken, a few ms after `beforeStop`, while its startTime falls
    // somewhere inside the start call (own_start_ms wide: 222 ms on a local
    // boot of the bundle).
    const profileWallStart = beforeStop - (profile.endTime - profile.startTime) / 1000;
    const result = summarizeProfile(profile, profileWallStart, mine, sourceTextReader());
    cur = { ...cur, state: "done", finished_at: new Date().toISOString(), own_stop_ms: ownStop, stalls_so_far: mine.length, result };
  } finally {
    if (tick) clearInterval(tick);
    // Never throws into the run: a failed close is logged and the result stands.
    const warn = (e: unknown): void => console.warn("[loop-profile] closing the profiler session:", e instanceof Error ? e.message : e);
    await post(session, "Profiler.disable").catch(warn);
    try { session.disconnect(); } catch (e) { warn(e); }
  }
}
