// Ratchet for the 2026-10-14 stale-tab repair (lib/tabResume.ts). Human
// reports: 2026-07-22 "I leave the page… come back and the data is stale";
// 2026-10-14 "leave the tab up overnight and the adsb gets stale and the
// other stuff". Fake time + fake DOM, no real timers.
// Run: npx tsx --test client/src/lib/tabResume.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import {
  createTabResume, pollWhenVisible, RESUME_HEARTBEAT_MS, RESUME_SLEEP_GAP_MS,
  type ResumeEvent, type ResumeListener,
} from "./tabResume.ts";
import { createAircraftFeed } from "./air/globalFeed.ts";

function fakeTarget() {
  const fns = new Map<string, Set<(e: Event) => void>>();
  return {
    addEventListener(t: string, fn: (e: Event) => void) { (fns.get(t) ?? fns.set(t, new Set()).get(t)!).add(fn); },
    removeEventListener(t: string, fn: (e: Event) => void) { fns.get(t)?.delete(fn); },
    fire(t: string, e: object = {}) { for (const fn of Array.from(fns.get(t) ?? [])) fn(e as Event); },
    count(t: string) { return fns.get(t)?.size ?? 0; },
  };
}

function rig(startHidden = false) {
  let t = 1_000_000;
  const doc = Object.assign(fakeTarget(), { hidden: startHidden });
  const win = fakeTarget();
  const intervals = new Map<number, () => void>();
  let nextId = 1;
  const tr = createTabResume({
    doc, win, now: () => t,
    setInterval: (fn) => { const id = nextId++; intervals.set(id, fn); return id; },
    clearInterval: (h) => { intervals.delete(h as number); },
  });
  const got: ResumeEvent[] = [];
  tr.subscribe((ev) => got.push(ev));
  return {
    tr, doc, win, got, intervals,
    advance(ms: number) { t += ms; },
    beat() { for (const fn of intervals.values()) fn(); },
    now: () => t,
  };
}

test("visible: coming back to the tab emits the time away", () => {
  const r = rig();
  r.doc.hidden = true; r.doc.fire("visibilitychange");
  r.advance(8 * 3600_000); // overnight
  r.doc.hidden = false; r.doc.fire("visibilitychange");
  assert.deepEqual(r.got, [{ reason: "visible", awayMs: 8 * 3600_000 }]);
});

test("wake: a heartbeat gap with the tab in front (computer slept) emits", () => {
  const r = rig();
  r.advance(RESUME_HEARTBEAT_MS); r.beat();
  assert.equal(r.got.length, 0, "a normal beat is not a resume");
  r.advance(6 * 3600_000); r.beat();
  assert.equal(r.got.length, 1);
  assert.equal(r.got[0].reason, "wake");
  assert.equal(r.got[0].awayMs, 6 * 3600_000);
  r.advance(RESUME_HEARTBEAT_MS); r.beat();
  assert.equal(r.got.length, 1, "fires once per sleep");
});

test("wake: throttled timers in a HIDDEN tab never count as a wake", () => {
  const r = rig(true);
  r.advance(RESUME_SLEEP_GAP_MS * 5); r.beat();
  assert.equal(r.got.length, 0);
});

test("online after offline emits; bfcache restore emits; plain pageshow does not", () => {
  const r = rig();
  r.win.fire("offline"); r.advance(90_000); r.win.fire("online");
  assert.deepEqual(r.got[0], { reason: "online", awayMs: 90_000 });
  r.win.fire("pageshow", { persisted: false });
  assert.equal(r.got.length, 1);
  r.win.fire("pageshow", { persisted: true });
  assert.equal(r.got[1].reason, "pageshow");
});

test("a throwing listener never starves the others, is logged, and dispose detaches everything", () => {
  const r = rig();
  r.tr.subscribe(() => { throw new Error("bad listener"); });
  const logged: unknown[][] = [];
  const realError = console.error;
  console.error = (...args: unknown[]) => { logged.push(args); };
  try {
    r.doc.hidden = true; r.doc.fire("visibilitychange");
    r.doc.hidden = false; r.doc.fire("visibilitychange");
  } finally { console.error = realError; }
  assert.equal(r.got.length, 1);
  assert.equal(logged.length, 1, "the failure is loud, not swallowed");
  r.tr.dispose();
  assert.equal(r.doc.count("visibilitychange"), 0);
  for (const t of ["online", "offline", "pageshow"]) assert.equal(r.win.count(t), 0, t);
  assert.equal(r.intervals.size, 0);
});

function pollRig(opts: { whileHidden?: boolean } = {}) {
  let t = 0;
  let hidden = false;
  let tick: (() => void) | null = null;
  let resume: ResumeListener | null = null;
  let loads = 0;
  const stop = pollWhenVisible(() => { loads++; }, 300_000, {
    ...opts,
    now: () => t,
    isHidden: () => hidden,
    setInterval: (fn) => { tick = fn; return 1; },
    clearInterval: () => { tick = null; },
    subscribe: (fn) => { resume = fn; return () => { resume = null; }; },
  });
  return {
    stop, get loads() { return loads; },
    set hidden(v: boolean) { hidden = v; },
    advance(ms: number) { t += ms; },
    tick() { tick?.(); },
    resume(awayMs = 0) { resume?.({ reason: "visible", awayMs }); },
    get armed() { return tick !== null && resume !== null; },
  };
}

test("pollWhenVisible: hidden ticks skip; return after a full interval reloads at once", () => {
  const p = pollRig();
  p.advance(300_000); p.tick();
  assert.equal(p.loads, 1, "visible tick loads");
  p.hidden = true;
  p.advance(300_000); p.tick();
  p.advance(300_000); p.tick();
  assert.equal(p.loads, 1, "hidden ticks skip");
  p.hidden = false; p.resume(600_000);
  assert.equal(p.loads, 2, "came back after 10 min -> immediate refresh, not up to 5 more min");
});

test("pollWhenVisible: a quick tab flip costs no request", () => {
  const p = pollRig();
  p.advance(20_000); p.resume(20_000);
  assert.equal(p.loads, 0);
});

test("pollWhenVisible: whileHidden keeps ticking hidden; stop disarms both", () => {
  const p = pollRig({ whileHidden: true });
  p.hidden = true; p.advance(300_000); p.tick();
  assert.equal(p.loads, 1);
  assert.equal(p.armed, true);
  p.stop();
  assert.equal(p.armed, false);
});

test("global aircraft feed: a request hung on a dead connection times out as an ERROR", async () => {
  const timers: Array<() => void> = [];
  const hung = ((_u: string, init?: { signal?: AbortSignal }) => new Promise((_res, rej) => {
    init?.signal?.addEventListener("abort", () => rej(Object.assign(new Error("aborted"), { name: "AbortError" })));
  })) as unknown as typeof fetch;
  const feed = createAircraftFeed({
    fetchImpl: hung, now: () => 1, timeoutMs: 30_000,
    setTimeout: (fn) => { timers.push(fn); return timers.length; },
    clearTimeout: () => {},
  });
  feed.shouldUseGlobal({ north: 80, south: -80, east: 180, west: -180 }, 1);
  const p = feed.fetchGlobal({ north: 80, south: -80, east: 180, west: -180 });
  assert.equal(timers.length, 1, "the request is time-boxed");
  timers[0]();
  await assert.rejects(p, /timed out/);
});

test("datamap.tsx: every feed poller goes through the shared resume signal", () => {
  const here = dirname(fileURLToPath(import.meta.url));
  const src = readFileSync(join(here, "..", "pages", "datamap.tsx"), "utf8");
  assert.doesNotMatch(src, /window\.setInterval\(\(\) => \{ if \(!document\.hidden\) load\(\); \}/,
    "hidden-gated poller without return refresh: use pollWhenVisible");
  assert.doesNotMatch(src, /window\.setInterval\(load, /, "bare poller: use pollWhenVisible");
  assert.doesNotMatch(src, /addEventListener\("visibilitychange"/,
    "visibilitychange alone misses sleep/wake and reconnect: use onTabResume");
  assert.ok((src.match(/pollWhenVisible\(load, /g) || []).length >= 34);
  assert.match(src, /signal: timeoutSignal\(LIVE_FETCH_TIMEOUT_MS\)/, "live feed request is time-boxed");
});
