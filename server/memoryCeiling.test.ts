// memoryCeiling.ts — the two memory walls, readable from outside.
//
// KNOWN BROKEN #41: three sessions could not tell a V8 heap OOM from a
// container cgroup kill because neither ceiling was ever reported. These pin
// that both are, that unreadable/unlimited never throws, and that `pressure`
// names the right wall.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { memoryCeiling, memoryPressure, readCgroupMemory, NEAR_LIMIT_FRACTION } from "./memoryCeiling";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const GB = 1073741824;

function fakeFs(files: Record<string, string>) {
  return (p: string) => {
    if (p in files) return files[p];
    throw Object.assign(new Error(`ENOENT: ${p}`), { code: "ENOENT" });
  };
}

test("cgroup v2: limit and usage are read and converted to MB", () => {
  const c = memoryCeiling({
    readFile: fakeFs({ "/sys/fs/cgroup/memory.max": `${8 * GB}\n`, "/sys/fs/cgroup/memory.current": `${1.5 * GB}\n` }),
    heapLimitBytes: 4.8 * GB,
  });
  assert.equal(c.cgroupSource, "v2");
  assert.equal(c.cgroupLimitMB, 8192);
  assert.equal(c.cgroupUsageMB, 1536);
  assert.equal(c.cgroupHeadroomMB, 8192 - 1536);
  assert.equal(c.heapLimitMB, Math.round(4.8 * 1024));
});

test("cgroup v2 'max' means unlimited: limit null, headroom null, usage still reported", () => {
  const c = memoryCeiling({
    readFile: fakeFs({ "/sys/fs/cgroup/memory.max": "max\n", "/sys/fs/cgroup/memory.current": `${GB}` }),
    heapLimitBytes: GB,
  });
  assert.equal(c.cgroupLimitMB, null);
  assert.equal(c.cgroupUsageMB, 1024);
  assert.equal(c.cgroupHeadroomMB, null);
});

test("cgroup v1 fallback, and its 2^60-style 'unlimited' sentinel reads as null", () => {
  const cg = readCgroupMemory(fakeFs({
    "/sys/fs/cgroup/memory/memory.limit_in_bytes": "9223372036854771712\n",
    "/sys/fs/cgroup/memory/memory.usage_in_bytes": `${2 * GB}\n`,
  }));
  assert.equal(cg.source, "v1");
  assert.equal(cg.limitBytes, null);
  assert.equal(cg.usageBytes, 2 * GB);
  const real = readCgroupMemory(fakeFs({
    "/sys/fs/cgroup/memory/memory.limit_in_bytes": `${4 * GB}`,
    "/sys/fs/cgroup/memory/memory.usage_in_bytes": `${GB}`,
  }));
  assert.equal(real.limitBytes, 4 * GB);
});

test("no cgroup files at all (a dev laptop, macOS): nulls, never a throw", () => {
  const c = memoryCeiling({ readFile: fakeFs({}), heapLimitBytes: GB });
  assert.equal(c.cgroupSource, null);
  assert.equal(c.cgroupLimitMB, null);
  assert.equal(c.cgroupUsageMB, null);
  assert.equal(c.cgroupHeadroomMB, null);
  assert.equal(c.heapLimitMB, 1024);
});

test("garbage in the sysfs file is null, not NaN", () => {
  const cg = readCgroupMemory(fakeFs({ "/sys/fs/cgroup/memory.max": "not-a-number", "/sys/fs/cgroup/memory.current": "-5" }));
  assert.equal(cg.limitBytes, null);
  assert.equal(cg.usageBytes, null);
});

test("the real reader on this machine returns finite numbers or nulls and never throws", () => {
  const c = memoryCeiling();
  assert.ok(c.heapLimitMB > 0, "V8 always has a heap limit");
  for (const k of ["cgroupLimitMB", "cgroupUsageMB", "cgroupHeadroomMB"] as const) {
    assert.ok(c[k] === null || Number.isFinite(c[k]), `${k} must be null or finite`);
  }
});

test("pressure names the wall: heap first, then cgroup, ok with headroom on both", () => {
  const roomy = { heapLimitMB: 4900, cgroupLimitMB: 8192, cgroupUsageMB: 1500, cgroupSource: "v2" as const, cgroupHeadroomMB: 6692 };
  assert.equal(memoryPressure(900, roomy), "ok");
  // THE SEPT-8 SHAPE under a 1228MB heap cap (60% of a 2GB cgroup): heap at
  // 1050MB is the wall being hit — a V8 OOM, a Node-side fix.
  const smallHeap = { heapLimitMB: 1228, cgroupLimitMB: 2048, cgroupUsageMB: 1300, cgroupSource: "v2" as const, cgroupHeadroomMB: 748 };
  assert.equal(memoryPressure(1050, smallHeap), "heap-near-limit");
  // Same container, heap fine, but daemon + children have the cgroup at 95%:
  // a SIGKILL is coming and it is NOT a Node heap problem.
  const fullContainer = { heapLimitMB: 1228, cgroupLimitMB: 2048, cgroupUsageMB: 1950, cgroupSource: "v2" as const, cgroupHeadroomMB: 98 };
  assert.equal(memoryPressure(600, fullContainer), "cgroup-near-limit");
  // exactly at the fraction counts as near (>=), just under does not
  assert.equal(memoryPressure(Math.ceil(1228 * NEAR_LIMIT_FRACTION), smallHeap), "heap-near-limit");
  assert.equal(memoryPressure(Math.floor(1228 * NEAR_LIMIT_FRACTION) - 1, { ...smallHeap, cgroupUsageMB: 100 }), "ok");
});

test("pressure is 'unknown' only when there is nothing to compare against", () => {
  assert.equal(memoryPressure(100, { heapLimitMB: 0, cgroupLimitMB: null, cgroupUsageMB: null, cgroupSource: null, cgroupHeadroomMB: null }), "unknown");
  // heap known, cgroup unknown: still a verdict on the heap
  assert.equal(memoryPressure(100, { heapLimitMB: 1024, cgroupLimitMB: null, cgroupUsageMB: null, cgroupSource: null, cgroupHeadroomMB: null }), "ok");
});

// ── source ratchet: the health payload carries the ceilings ─────────────────
test("bot.ts's /api/health memory block reports both walls and the pressure verdict", () => {
  const bot = fs.readFileSync(path.join(HERE, "bot.ts"), "utf8");
  const start = bot.indexOf('app.get("/api/health"');
  const end = bot.indexOf("app.get(", start + 10);
  const handler = bot.slice(start, end);
  assert.match(handler, /memoryCeiling\(\)/, "the handler must read the ceilings");
  assert.match(handler, /memoryPressure\(/, "the handler must publish the pressure verdict");
  for (const key of ["heapLimitMB", "cgroupLimitMB", "cgroupUsageMB", "cgroupHeadroomMB", "pressure"]) {
    assert.match(handler, new RegExp(`\\b${key}\\b`), `memory block must expose ${key}`);
  }
});
