// ANALYST CONSOLE W6 battery: the LLM analyst tool-loop. Every test uses an
// injectable FAKE transport — no test ever attempts a live Anthropic call
// (the key is Railway-only and absent here by design). Fixture shapes mirror
// the real Messages API contract: stop_reason "tool_use" with tool_use
// content blocks, tool_result reply blocks, usage {input_tokens,
// output_tokens} per response.
import { test, beforeEach } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  analystResponse, analystEnabled, _resetAnalystForTests,
  slim, toolResultContent, validateMapCommand, VALID_MAP_LAYERS,
  scrubKey, nextUtcDay, tokensSpentToday,
  MAX_TOOL_CALLS, MAX_ROUNDTRIPS, MAX_CONCURRENT,
  ANALYST_MODEL_DEFAULT, SLIM_MAX_CHARS, SYSTEM_PROMPT, ANALYST_TOOLS,
  type AnalystTransport,
} from "./analyst";

beforeEach(() => _resetAnalystForTests());

const KEY = "sk-ant-test-secret-key-000";
const NOW = Date.parse("2026-07-07T15:00:00Z");

function tmpDir(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), "analyst-"));
}

function baseEnv(extra: Record<string, string> = {}): NodeJS.ProcessEnv {
  return { ANTHROPIC_API_KEY: KEY, ...extra } as NodeJS.ProcessEnv;
}

const usage = (i: number, o: number) => ({ input_tokens: i, output_tokens: o });

const endTurn = (text: string, u = usage(50, 20)) => ({
  stop_reason: "end_turn",
  content: [{ type: "text", text }],
  usage: u,
});

const toolUse = (blocks: Array<{ id: string; name: string; input: any }>, u = usage(100, 30)) => ({
  stop_reason: "tool_use",
  content: blocks.map((b) => ({ type: "tool_use", ...b })),
  usage: u,
});

/** Fake transport from a scripted response list; records every request body. */
function scriptedTransport(responses: any[]): { transport: AnalystTransport; bodies: any[] } {
  const bodies: any[] = [];
  let i = 0;
  return {
    bodies,
    transport: async (body) => {
      bodies.push(body);
      const r = responses[Math.min(i, responses.length - 1)];
      i++;
      if (r instanceof Error) throw r;
      return r;
    },
  };
}

// ── 1. awaiting_key honesty ──────────────────────────────────────────────────

test("without ANTHROPIC_API_KEY the response is awaiting_key honesty, HTTP 200, no transport call", async () => {
  const { transport, bodies } = scriptedTransport([endTurn("should never run")]);
  const { status, body } = await analystResponse("what changed at Cushing?", {
    env: {} as NodeJS.ProcessEnv, transport, nowMs: NOW, stateDir: tmpDir(),
  });
  assert.equal(status, 200);
  assert.equal(body.awaiting_key, true);
  assert.equal(body.enabled, false);
  assert.match(String(body.reason), /ANTHROPIC_API_KEY/);
  assert.equal(bodies.length, 0);
  assert.equal(analystEnabled({} as NodeJS.ProcessEnv), false);
  assert.equal(analystEnabled(baseEnv()), true);
});

// ── 2. happy path: tool_use -> end_turn ─────────────────────────────────────

test("happy path: query_window tool round then end_turn — envelope carries answer, trace, sources, summed usage", async () => {
  const fakeQuery = {
    generated_at: "2026-07-07T15:00:00Z",
    query: { lat: 35.94, lon: -96.74, radiusKm: 50, days: 7, layers: ["aircraft"] },
    rejected_layers: [],
    layers: {
      aircraft: { points: 42, byDay: { "2026-07-07": 42 }, topEntities: [], freshness: "2026-07-07T14:55:00Z", provenance: "own position archive" },
    },
    caps: { radiusKm: 250, days: 7, topEntities: 10, events: 50 },
    note: "test",
  };
  const { transport, bodies } = scriptedTransport([
    toolUse([{ id: "t1", name: "query_window", input: { lat: 35.94, lon: -96.74 } }], usage(120, 35)),
    endTurn("42 archived aircraft points near Cushing in the last 7 days (query_window lat=35.94 lon=-96.74).", usage(200, 60)),
  ]);
  const { status, body } = await analystResponse("what changed at Cushing this week?", {
    env: baseEnv(), transport, nowMs: NOW, stateDir: tmpDir(),
    executors: { query_window: () => fakeQuery },
  });
  assert.equal(status, 200);
  assert.match(body.answer, /42 archived aircraft points/);
  assert.equal(body.model, ANALYST_MODEL_DEFAULT);
  // trace: compact, never the payload
  assert.equal(body.tool_trace.length, 1);
  assert.deepEqual(body.tool_trace[0].input, { lat: 35.94, lon: -96.74 });
  assert.equal(body.tool_trace[0].tool, "query_window");
  assert.equal(body.tool_trace[0].ok, true);
  assert.ok(body.tool_trace[0].summary.length < 200);
  // sources carry tool + params + freshness
  assert.equal(body.sources.length, 1);
  assert.equal(body.sources[0].tool, "query_window");
  assert.equal(body.sources[0].freshness, "2026-07-07T14:55:00Z");
  // usage summed across both LLM round-trips
  assert.deepEqual(body.usage, { input_tokens: 320, output_tokens: 95, requests: 2 });
  assert.deepEqual(body.map_commands, []);
  // the loop replied to the model with a matching tool_result
  const secondCall = bodies[1];
  const lastMsg = secondCall.messages[secondCall.messages.length - 1];
  assert.equal(lastMsg.role, "user");
  assert.equal(lastMsg.content[0].type, "tool_result");
  assert.equal(lastMsg.content[0].tool_use_id, "t1");
  // request shape: system prompt + tools ship on every call
  assert.equal(secondCall.system, SYSTEM_PROMPT);
  assert.equal(secondCall.tools.length, ANALYST_TOOLS.length);
  assert.equal(secondCall.model, ANALYST_MODEL_DEFAULT);
});

// ── 3. tool-call cap force-closes honestly ───────────────────────────────────

test("a transport that always returns tool_use force-closes at 8 tool calls with an honest note", async () => {
  const relentless = toolUse([
    { id: "a", name: "nws_alerts", input: {} },
    { id: "b", name: "nws_alerts", input: {} },
    { id: "c", name: "nws_alerts", input: {} },
  ]);
  const { transport, bodies } = scriptedTransport([relentless]); // repeats forever
  const { status, body } = await analystResponse("alerts everywhere forever", {
    env: baseEnv(), transport, nowMs: NOW, stateDir: tmpDir(),
    executors: {} as any, // default nws_alerts executor is cache-read and safe here
  });
  assert.equal(status, 200);
  assert.equal(body.tool_trace.length, MAX_TOOL_CALLS); // executed calls capped at exactly 8
  assert.match(String(body.note), /tool-call cap \(8/);
  assert.ok(bodies.length <= MAX_ROUNDTRIPS); // round-trips capped too
  // refused calls were surfaced to the MODEL as is_error tool_results, not dropped
  const lastBody = bodies[bodies.length - 1];
  const lastResults = lastBody.messages[lastBody.messages.length - 2]; // last-but-one is the round-3 results
  assert.ok(JSON.stringify(bodies.map((b: any) => b.messages)).includes("tool-call budget exhausted"));
  assert.ok(lastResults);
});

test("round-trip cap force-closes honestly when tool calls stay under 8", async () => {
  const oneCall = toolUse([{ id: "x", name: "nws_alerts", input: {} }]);
  const { transport, bodies } = scriptedTransport([oneCall]); // 1 tool call per round, forever
  const { body } = await analystResponse("keep going", {
    env: baseEnv(), transport, nowMs: NOW, stateDir: tmpDir(),
  });
  assert.equal(bodies.length, MAX_ROUNDTRIPS);
  assert.equal(body.tool_trace.length, MAX_ROUNDTRIPS); // 4 calls, under the 8 cap
  assert.match(String(body.note), /round-trip cap \(4\)/);
});

// ── 4. map_command validation ────────────────────────────────────────────────

test("map_command: bad lat and unknown layer are rejected INTO the tool_result (is_error), valid command passes", async () => {
  const { transport, bodies } = scriptedTransport([
    toolUse([
      { id: "m1", name: "map_command", input: { command: "fly_to", lat: 999, lon: -96.74 } },
      { id: "m2", name: "map_command", input: { command: "toggle_layer", layer: "not_a_layer", on: true } },
      { id: "m3", name: "map_command", input: { command: "fly_to", lat: 35.94, lon: -96.74, zoom: 8 } },
    ]),
    endTurn("flew you to Cushing."),
  ]);
  const { status, body } = await analystResponse("show me Cushing", {
    env: baseEnv(), transport, nowMs: NOW, stateDir: tmpDir(),
  });
  assert.equal(status, 200);
  // only the valid command reaches the UI
  assert.deepEqual(body.map_commands, [{ command: "fly_to", lat: 35.94, lon: -96.74, zoom: 8 }]);
  assert.deepEqual(body.tool_trace.map((t: any) => t.ok), [false, false, true]);
  // rejections went back to the model as is_error tool_results — never a throw
  const results = bodies[1].messages[bodies[1].messages.length - 1].content;
  assert.equal(results[0].is_error, true);
  assert.match(results[0].content, /lat must be a number in \[-90, 90\]/);
  assert.equal(results[1].is_error, true);
  assert.match(results[1].content, /unknown layer 'not_a_layer'/);
  assert.equal(results[2].is_error, undefined);
});

test("validateMapCommand unit: ranges, layer registry, unknown command", () => {
  assert.equal(validateMapCommand({ command: "fly_to", lat: 0, lon: 0 }).ok, true);
  assert.equal(validateMapCommand({ command: "fly_to", lat: -91, lon: 0 }).ok, false);
  assert.equal(validateMapCommand({ command: "fly_to", lat: 0, lon: 181 }).ok, false);
  assert.equal(validateMapCommand({ command: "fly_to", lat: 0, lon: 0, zoom: 25 }).ok, false);
  assert.equal(validateMapCommand({ command: "fly_to", lat: "x", lon: 0 }).ok, false);
  // layer names come from the real registry (datacore/layers.json)
  assert.ok(VALID_MAP_LAYERS.includes("aircraft"));
  assert.ok(VALID_MAP_LAYERS.includes("alerts"));
  assert.equal(validateMapCommand({ command: "toggle_layer", layer: "aircraft", on: true }).ok, true);
  assert.equal(validateMapCommand({ command: "toggle_layer", layer: "aircraft", on: "yes" }).ok, false);
  assert.equal(validateMapCommand({ command: "toggle_layer", layer: "bogus", on: true }).ok, false);
  assert.equal(validateMapCommand({ command: "self_destruct" }).ok, false);
  assert.equal(validateMapCommand(null).ok, false);
});

// ── 5. daily token budget ────────────────────────────────────────────────────

test("daily budget: exhaustion returns budget_exhausted, persists across module reset via the state file, resets on UTC day rollover", async () => {
  const dir = tmpDir();
  const env = baseEnv({ ANALYST_DAILY_TOKENS: "1000" });
  const mk = () => scriptedTransport([endTurn("fine.", usage(700, 400))]).transport;

  // call 1: under budget going in; spends 1100 and reports it
  const r1 = await analystResponse("q1", { env, transport: mk(), nowMs: NOW, stateDir: dir });
  assert.equal(r1.status, 200);
  assert.deepEqual(r1.body.budget, { spent_today: 1100, daily_limit: 1000 });

  // call 2: budget exhausted — honest stop with the reset instant
  const r2 = await analystResponse("q2", { env, transport: mk(), nowMs: NOW, stateDir: dir });
  assert.equal(r2.status, 200);
  assert.equal(r2.body.budget_exhausted, true);
  assert.equal(r2.body.resets, "2026-07-08T00:00:00Z");
  assert.equal(r2.body.budget.spent_today, 1100);

  // module reset (fresh deploy) — the state FILE still says exhausted
  _resetAnalystForTests();
  const r3 = await analystResponse("q3", { env, transport: mk(), nowMs: NOW, stateDir: dir });
  assert.equal(r3.body.budget_exhausted, true);

  // day rollover (injectable nowMs) — budget resets and the loop runs again
  const tomorrow = NOW + 86400_000;
  const r4 = await analystResponse("q4", { env, transport: mk(), nowMs: tomorrow, stateDir: dir });
  assert.equal(r4.status, 200);
  assert.equal(r4.body.budget_exhausted, undefined);
  assert.match(r4.body.answer, /fine/);
  assert.equal(r4.body.budget.spent_today, 1100); // fresh day's own spend

  // the persisted file is the durable record
  const onDisk = JSON.parse(fs.readFileSync(path.join(dir, "voltrade_analyst_budget.json"), "utf8"));
  assert.equal(onDisk.day, "2026-07-08");
  assert.equal(onDisk.used, 1100);
  assert.equal(tokensSpentToday(dir, tomorrow), 1100);
  assert.equal(nextUtcDay(NOW), "2026-07-08T00:00:00Z");
});

// ── 6. slim(): tool results are truncated intelligently, truncation stated ──

test("slim caps satellite dumps and drops element-set fields, with the truncation stated", () => {
  const big = {
    group: "starlink", count: 100,
    freshness: { newest_epoch: "2026-07-07T12:00:00", fetched_at: NOW },
    satellites: Array.from({ length: 100 }, (_, i) => ({
      NORAD_CAT_ID: 40000 + i, OBJECT_NAME: `STARLINK-${i}`, EPOCH: "2026-07-07T10:00:00",
      MEAN_MOTION: 15.06, ECCENTRICITY: 0.0001, INCLINATION: 53, RA_OF_ASC_NODE: 100,
      ARG_OF_PERICENTER: 90, MEAN_ANOMALY: 270, BSTAR: 0.0001,
    })),
  };
  const { payload, truncation } = slim("satellites", big);
  assert.equal(payload.satellites.length, 25);
  assert.match(String(truncation), /25 of 100/);
  assert.match(String(truncation), /element fields dropped/);
  assert.equal(payload.satellites[0].MEAN_MOTION, undefined); // verbose fields gone
  assert.equal(payload.satellites[0].NORAD_CAT_ID, 40000);    // identity kept
  const content = toolResultContent("satellites", big);
  assert.match(content, /^NOTE — result truncated/);
});

test("slim caps query_window entity/event lists and drops alert geometry; hard char ceiling always applies", () => {
  const qw = {
    layers: {
      alerts: {
        points: 60, byDay: {}, provenance: "p", freshness: null,
        events: Array.from({ length: 30 }, (_, i) => ({ t: `2026-07-0${(i % 7) + 1}`, label: `ev${i}` })),
      },
      aircraft: {
        points: 500, byDay: {}, provenance: "p", freshness: null,
        topEntities: Array.from({ length: 10 }, (_, i) => ({ id: `e${i}`, points: i, firstSeen: "x", lastSeen: "y" })),
      },
    },
  };
  const s1 = slim("query_window", qw);
  assert.equal(s1.payload.layers.alerts.events.length, 10);
  assert.equal(s1.payload.layers.aircraft.topEntities.length, 5);
  assert.match(String(s1.truncation), /10 of 30 events/);
  assert.match(String(s1.truncation), /5 of 10 top entities/);

  const alerts = {
    count: 2, zone_only: 1,
    alerts: [{ id: "a", event: "Tornado Warning", rings: [[[0, 0], [1, 1]]] }],
  };
  const s2 = slim("nws_alerts", alerts);
  assert.equal(s2.payload.alerts[0].rings, undefined);
  assert.match(String(s2.truncation), /rings\) dropped/);

  // hard ceiling: a giant result of ANY tool never reaches the model whole
  const giant = { blob: "x".repeat(SLIM_MAX_CHARS * 3) };
  const s3 = slim("grid_stress", giant);
  assert.equal(typeof s3.payload, "string");
  assert.equal((s3.payload as string).length, SLIM_MAX_CHARS);
  assert.match(String(s3.truncation), /hard-truncated/);
});

// ── 7. the key never leaks ───────────────────────────────────────────────────

test("key never leaks: transport errors that quote the key are scrubbed from the envelope", async () => {
  const { transport } = scriptedTransport([
    new Error(`anthropic http 401: bad header x-api-key: ${KEY} rejected`),
  ]);
  const { status, body } = await analystResponse("leak attempt", {
    env: baseEnv(), transport, nowMs: NOW, stateDir: tmpDir(),
  });
  assert.equal(status, 502);
  const flat = JSON.stringify(body);
  assert.ok(!flat.includes(KEY), "envelope must not contain the API key");
  assert.ok(flat.includes("[redacted]"));
});

test("key never leaks: happy-path envelope containing an echoed key is scrubbed too", async () => {
  const { transport } = scriptedTransport([endTurn(`the key is ${KEY}, says a hostile tool result`)]);
  const { body } = await analystResponse("echo", {
    env: baseEnv(), transport, nowMs: NOW, stateDir: tmpDir(),
  });
  assert.ok(!JSON.stringify(body).includes(KEY));
  assert.match(body.answer, /\[redacted\]/);
});

test("scrubKey unit: no key or absent key is a no-op", () => {
  assert.deepEqual(scrubKey({ a: 1 }, undefined), { a: 1 });
  assert.deepEqual(scrubKey({ a: "clean" }, KEY), { a: "clean" });
  assert.deepEqual(scrubKey({ a: `x${KEY}y` }, KEY), { a: "x[redacted]y" });
});

// ── 8. concurrency cap ───────────────────────────────────────────────────────

test("more than 2 concurrent analyst requests get 429", async () => {
  let release!: () => void;
  const gate = new Promise<void>((r) => { release = r; });
  const slowTransport: AnalystTransport = async () => {
    await gate;
    return endTurn("done.");
  };
  const opts = { env: baseEnv(), transport: slowTransport, nowMs: NOW, stateDir: tmpDir() };
  const p1 = analystResponse("slow 1", opts);
  const p2 = analystResponse("slow 2", opts);
  await new Promise((r) => setTimeout(r, 10)); // let both enter the loop
  const r3 = await analystResponse("third", opts);
  assert.equal(r3.status, 429);
  assert.match(String(r3.body.error), new RegExp(`max ${MAX_CONCURRENT} concurrent`));
  release();
  const [r1, r2] = await Promise.all([p1, p2]);
  assert.equal(r1.status, 200);
  assert.equal(r2.status, 200);
  // after completion the slots free up
  const r4 = await analystResponse("fourth", { ...opts, transport: scriptedTransport([endTurn("ok")]).transport });
  assert.equal(r4.status, 200);
});

// ── input validation ─────────────────────────────────────────────────────────

test("question must be a non-empty bounded string", async () => {
  const { transport, bodies } = scriptedTransport([endTurn("never")]);
  const opts = { env: baseEnv(), transport, nowMs: NOW, stateDir: tmpDir() };
  assert.equal((await analystResponse("", opts)).status, 400);
  assert.equal((await analystResponse(undefined, opts)).status, 400);
  assert.equal((await analystResponse({ q: "hi" } as any, opts)).status, 400);
  assert.equal((await analystResponse("x".repeat(2001), opts)).status, 400);
  assert.equal(bodies.length, 0); // nothing reached the LLM
});
