// server/loopProfile.ts: the on-demand reading of what blocks the event loop.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import {
  summarizeProfile, sourceTextReader, clampInt, loopProfileEnabled, startLoopProfile, loopProfileStatus,
  LOOP_PROFILE_DEFAULT_S, LOOP_PROFILE_MAX_S, LOOP_PROFILE_MIN_S, LOOP_PROFILE_MAX_STALLS, LOOP_PROFILE_STALL_MS,
  type CpuProfile, type ProfileNode,
} from "./loopProfile";

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");

const node = (id: number, functionName: string, children: number[] = [], line = id, url = "file:///app/dist/index.cjs"): ProfileNode =>
  ({ id, callFrame: { functionName, url: functionName.startsWith("(") ? "" : url, lineNumber: line, columnNumber: 4 }, children });

/** root -> idle | gc | timerTick -> saveAll -> stringify ; root -> onMessage */
function tree(): ProfileNode[] {
  return [
    node(1, "(root)", [2, 3, 4, 7]), node(2, "(idle)"), node(3, "(garbage collector)"),
    node(4, "timerTick", [5]), node(5, "saveAll", [6]), node(6, "stringify"),
    node(7, "onMessage"),
  ];
}
/** samples every 10 ms from wall 1,000,000: `plan` is [nodeId, count] runs */
function profile(plan: Array<[number, number]>): CpuProfile {
  const samples: number[] = [];
  for (const [id, n] of plan) for (let i = 0; i < n; i++) samples.push(id);
  return { nodes: tree(), startTime: 5_000_000, endTime: 5_000_000 + samples.length * 10_000 + 10_000,
    samples, timeDeltas: samples.map(() => 10_000) };
}
const WALL = 1_000_000;

test("a blocked stretch is charged to the frames sampled inside it, and to nothing sampled outside", () => {
  // 0-1 s idle, 1-3 s inside stringify under the timer, 3-4 s idle, 4-4.2 s a message handler
  const p = profile([[2, 100], [6, 200], [2, 100], [7, 20]]);
  const s = summarizeProfile(p, WALL, [{ from: WALL + 1000, to: WALL + 3010, blockedMs: 1810 }]);
  assert.equal(s.stalls.count, 1);
  assert.equal(s.stalls.longest_ms, 1810);
  assert.deepEqual(s.stalls_on_top.map((r) => r.fn), ["stringify"]);
  assert.deepEqual(s.stalls_on_stack.map((r) => r.fn).sort(), ["saveAll", "stringify", "timerTick"]);
  for (const r of s.stalls_on_stack) assert.ok(r.ms >= 1990 && r.ms <= 2010, `${r.fn} ${r.ms}`);
  assert.ok(!s.stalls_on_stack.some((r) => r.fn === "onMessage"), "a handler outside the stretch is not charged to it");
  assert.equal(s.stall_list.length, 1);
  assert.equal(s.stall_list[0].ended, new Date(WALL + 3010).toISOString());
  assert.ok(s.stall_list[0].sampled_busy_ms >= 1990 && s.stall_list[0].sampled_busy_ms <= 2010);
  assert.equal(s.stall_list[0].gc_ms, 0);
  // the whole run still names the handler
  assert.ok(s.run_on_top.some((r) => r.fn === "onMessage" && r.ms === 200));
  assert.equal(s.idle_ms, 2000);
  assert.equal(s.busy_ms, 2200);
});

test("idle samples are never busy time; garbage collection is counted and named", () => {
  const p = profile([[2, 50], [3, 80], [2, 50]]);
  const s = summarizeProfile(p, WALL, [{ from: WALL + 500, to: WALL + 1300, blockedMs: 600 }]);
  assert.equal(s.gc_ms, 800);
  assert.equal(s.stall_list[0].gc_ms, 800);
  assert.deepEqual(s.stalls_on_top.map((r) => r.fn), ["(garbage collector)"]);
  assert.equal(s.stalls_on_top[0].at, "(native)");
  assert.ok(!s.run_on_top.some((r) => r.fn === "(idle)"));
});

test("a frame that recurses is charged once per sample, not once per level", () => {
  const nodes: ProfileNode[] = [node(1, "(root)", [2]), node(2, "walk", [3], 40), node(3, "walk", [4], 40), node(4, "walk", [], 40)];
  const p: CpuProfile = { nodes, startTime: 0, endTime: 1_010_000, samples: new Array(100).fill(4), timeDeltas: new Array(100).fill(10_000) };
  const s = summarizeProfile(p, WALL, [{ from: WALL, to: WALL + 1010, blockedMs: 800 }]);
  assert.equal(s.stalls_on_stack.length, 1);
  assert.ok(s.stalls_on_stack[0].ms <= 1010, `charged ${s.stalls_on_stack[0].ms} ms for 1 s of samples`);
});

test("no blocked stretch: the totals are empty and the run is still described", () => {
  const s = summarizeProfile(profile([[2, 100], [7, 10]]), WALL, []);
  assert.deepEqual(s.stalls, { count: 0, blocked_ms: 0, longest_ms: 0, listed: 0 });
  assert.deepEqual(s.stalls_on_stack, []);
  assert.deepEqual(s.stall_list, []);
  assert.equal(s.run_on_top[0].fn, "onMessage");
});

test("more stretches than the list holds: the longest are listed in time order, all are in the totals", () => {
  const n = LOOP_PROFILE_MAX_STALLS + 5;
  const plan: Array<[number, number]> = [];
  const st = [];
  for (let i = 0; i < n; i++) {
    plan.push([2, 100], [6, 50 + i]);
    const from = WALL + i * 1500 + (i * (i - 1) / 2) * 10 + 1000;
    st.push({ from, to: from + (50 + i) * 10, blockedMs: 500 + i });
  }
  const s = summarizeProfile(profile(plan), WALL, st);
  assert.equal(s.stalls.count, n);
  assert.equal(s.stall_list.length, LOOP_PROFILE_MAX_STALLS);
  assert.equal(s.stall_list[0].blocked_ms, 505, "the five shortest are the ones left out");
  const ends = s.stall_list.map((r) => r.ended);
  assert.deepEqual(ends, ends.slice().sort());
  assert.equal(s.stalls.blocked_ms, st.reduce((a, x) => a + x.blockedMs, 0));
});

test("a frame carries the first characters of its function's text from the file V8 names", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "loopprofile-"));
  const file = path.join(dir, "bundle.cjs");
  fs.writeFileSync(file, 'var a=1;\nvar b=2;function q(){return saveSnapshot("swim_plan_history",x)}\n');
  const read = sourceTextReader(40);
  const text = read({ functionName: "q", url: pathToFileURL(file).href, lineNumber: 1, columnNumber: 8 });
  assert.equal(text, 'function q(){return saveSnapshot("swim_p');
  assert.equal(read({ functionName: "q", url: file, lineNumber: 1, columnNumber: 8 }), text, "a plain path reads the same");
  assert.equal(read({ functionName: "x", url: "", lineNumber: 0, columnNumber: 0 }), undefined);
  assert.equal(read({ functionName: "x", url: "node:internal/timers", lineNumber: 3, columnNumber: 0 }), undefined);
  assert.equal(read({ functionName: "x", url: pathToFileURL(path.join(dir, "missing.cjs")).href, lineNumber: 0, columnNumber: 0 }), undefined);
  assert.equal(read({ functionName: "x", url: pathToFileURL(file).href, lineNumber: 99, columnNumber: 0 }), undefined);
  assert.equal(read({ functionName: "x", url: "file:///app/node_modules/ws/lib/x.js", lineNumber: 0, columnNumber: 0 }), undefined);
  const s = summarizeProfile(
    { nodes: [node(1, "(root)", [2]), node(2, "q", [], 1, pathToFileURL(file).href)], startTime: 0, endTime: 20_000, samples: [2], timeDeltas: [10_000] },
    WALL, [{ from: WALL, to: WALL + 20, blockedMs: 500 }], (f) => read({ ...f, columnNumber: 8 }));
  assert.equal(s.stalls_on_top[0].text, text);
});

test("query values are clamped; the switch refuses a start", () => {
  assert.equal(clampInt(undefined, LOOP_PROFILE_DEFAULT_S, LOOP_PROFILE_MIN_S, LOOP_PROFILE_MAX_S), LOOP_PROFILE_DEFAULT_S);
  assert.equal(clampInt("abc", 660, 2, 900), 660);
  assert.equal(clampInt("99999", 660, 2, 900), 900);
  assert.equal(clampInt("-4", 660, 2, 900), 2);
  assert.equal(clampInt("30.9", 660, 2, 900), 30);
  assert.equal(loopProfileEnabled({}), true);
  for (const v of ["off", "0", "false", "OFF", " no "]) assert.equal(loopProfileEnabled({ VOLTRADE_LOOP_PROFILE: v }), false, v);
  const r = startLoopProfile({ seconds: 2 }, { VOLTRADE_LOOP_PROFILE: "off" });
  assert.equal(r.started, false);
  assert.match(String(r.reason), /VOLTRADE_LOOP_PROFILE is off/);
  assert.equal(loopProfileStatus({ VOLTRADE_LOOP_PROFILE: "off" }).enabled, false);
});

// The real profiler on a real block: the function that held the loop is named.
function burnTheLoopForTheTest(ms: number): number {
  const end = Date.now() + ms;
  let x = 0;
  while (Date.now() < end) x += Math.sqrt(x + 1);
  return x;
}

test("a real run: the function that blocked the loop is on the stack of the blocked stretch; a second start is refused meanwhile", async () => {
  const r = startLoopProfile({ seconds: 3, sampleMs: 5 }, {});
  assert.equal(r.started, true, String(r.reason));
  const again = startLoopProfile({ seconds: 3 }, {});
  assert.equal(again.started, false);
  assert.match(String(again.reason), /a run is on until/);
  assert.equal(loopProfileStatus({}).state, "running");
  await new Promise((res) => setTimeout(res, 700));
  burnTheLoopForTheTest(LOOP_PROFILE_STALL_MS + 800);
  await r.done;
  const st = loopProfileStatus({});
  assert.equal(st.state, "done", String(st.error));
  assert.equal(st.seconds, 3);
  assert.equal(st.sample_ms, 5);
  assert.equal(typeof st.own_start_ms, "number");
  assert.equal(typeof st.own_stop_ms, "number");
  const res = st.result;
  assert.ok(res, "a finished run carries its result");
  assert.ok(res.stalls.count >= 1, "the block was seen by the tick");
  assert.ok(res.stalls.longest_ms >= 600, `longest ${res.stalls.longest_ms}`);
  const named = res.stalls_on_stack.find((f) => f.fn === "burnTheLoopForTheTest");
  assert.ok(named, `on stack: ${res.stalls_on_stack.map((f) => f.fn).join(", ")}`);
  assert.ok(named.ms >= 500, `charged ${named.ms} ms`);
  assert.match(named.at, /loopProfile\.test\.ts:\d+:\d+$/);
  assert.ok(res.stall_list.some((s) => s.on_stack.some((f) => f.fn === "burnTheLoopForTheTest")));
  // and a new run may start once this one is done
  const next = startLoopProfile({ seconds: 2 }, {});
  assert.equal(next.started, true);
  await next.done;
  assert.equal(loopProfileStatus({}).state, "done");
});

test("the diag route serves the reading behind the token, starts only when asked, and audits a start", () => {
  const a = bot.indexOf('case "loop_profile": {');
  assert.ok(a > 0, "the probe exists");
  const block = bot.slice(a, bot.indexOf('case "xbrl_frames": {', a));
  assert.ok(bot.lastIndexOf("checkDiagToken(process.env, provided)", a) > bot.lastIndexOf('app.get("/api/diag/:probe"', a), "inside the token-gated dispatcher");
  assert.match(block, /String\(req\.query\.start \|\| ""\) === "1"/);
  assert.match(block, /startLoopProfile\(\{ seconds: req\.query\.seconds, sampleMs: req\.query\.sample_ms \}\)/);
  assert.match(block, /if \(r\.started\) audit\("LOOP-PROFILE"/);
  assert.equal((block.match(/sanitizeDiag\(/g) || []).length, 2, "both answers go through the scrub");
  assert.equal((bot.match(/startLoopProfile\(/g) || []).length, 1, "nothing starts a run but the probe");
});
