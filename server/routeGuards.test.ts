import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { raceDeadline, slotExpired, makeSlot, ROUTE_DEADLINE_MS, INFLIGHT_MAX_AGE_MS } from "./routeGuards";

const here = path.dirname(fileURLToPath(import.meta.url));
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

test("raceDeadline passes through fast resolution and rejection", async () => {
  assert.equal(await raceDeadline(Promise.resolve(42), 1000), 42);
  await assert.rejects(raceDeadline(Promise.reject(new Error("boom")), 1000), /boom/);
});

test("raceDeadline rejects when the promise is STUCK — the production outage shape", async () => {
  const stuck = new Promise(() => {}); // never settles
  await assert.rejects(raceDeadline(stuck, 50, "trains"), /trains: upstream still pending after 50ms/);
});

test("slotExpired: fresh slots live, stale slots are abandoned", () => {
  const now = 1_000_000;
  assert.equal(slotExpired(null, now), true);
  assert.equal(slotExpired({ p: Promise.resolve(), at: now - 1000 }, now), false);
  assert.equal(slotExpired({ p: Promise.resolve(), at: now - INFLIGHT_MAX_AGE_MS - 1 }, now), true);
});

test("makeSlot cleanup is identity-guarded: a late-settling orphan never clobbers its replacement", async () => {
  let holder: any = null;
  const clear = (self: any) => { if (holder === self) holder = null; };

  // orphan: stuck fetch that will settle LATE
  let releaseOrphan!: () => void;
  const orphan = makeSlot(() => new Promise<void>((r) => { releaseOrphan = r; }), 0, clear);
  holder = orphan;

  // replacement arrives after the orphan is presumed stuck
  const fresh = makeSlot(() => sleep(5), 100, clear);
  holder = fresh;

  releaseOrphan();          // orphan settles late…
  await sleep(1);
  assert.equal(holder, fresh, "orphan settling must NOT clear the replacement");

  await sleep(10);          // fresh settles → clears itself
  assert.equal(holder, null);
});

test("makeSlot absorbs rejections — no unhandled rejection from the slot chain", async () => {
  let unhandled = 0;
  const onUR = () => { unhandled++; };
  process.on("unhandledRejection", onUR);
  try {
    let holder: any = null;
    const slot = makeSlot(() => Promise.reject(new Error("upstream down")), 0, (s) => { if (holder === s) holder = null; });
    holder = slot;
    await assert.rejects(raceDeadline(slot.p, 100), /upstream down/);
    await sleep(20);
    assert.equal(unhandled, 0, "slot chain leaked an unhandled rejection");
    assert.equal(holder, null, "rejected slot must still clear itself");
  } finally {
    process.off("unhandledRejection", onUR);
  }
});

test("constants sane: route deadline under typical proxy timeouts; expiry over worst legitimate chain-step budget", () => {
  assert.ok(ROUTE_DEADLINE_MS <= 20_000);
  assert.ok(INFLIGHT_MAX_AGE_MS > 2 * ROUTE_DEADLINE_MS);
});

// ── wiring pins: both vulnerable routes actually use the guards, and the
// Digitraffic fetch sends the now-required gzip header ──────────────────────

test("routes.ts: trains + aircraft use raceDeadline/slot expiry; Digitraffic gets Accept-Encoding gzip", () => {
  const routes = fs.readFileSync(path.join(here, "routes.ts"), "utf8");
  assert.ok(routes.includes('from "./routeGuards"'), "guards imported");
  const trainsBlock = routes.slice(routes.indexOf("fetchTrains"), routes.indexOf('app.get("/api/data/trains"') + 2000);
  assert.ok(trainsBlock.includes("raceDeadline("), "trains route missing hard deadline");
  assert.ok(trainsBlock.includes("slotExpired("), "trains route missing slot expiry");
  const airBlock = routes.slice(routes.indexOf('app.get("/api/data/aircraft"'), routes.indexOf('app.get("/api/data/aircraft"') + 3000);
  assert.ok(airBlock.includes("raceDeadline("), "aircraft route missing hard deadline");
  assert.ok(airBlock.includes("slotExpired("), "aircraft route missing slot expiry");
  assert.ok(/rata\.digitraffic\.fi[\s\S]{0,400}Accept-Encoding/.test(routes) || /"Accept-Encoding": "gzip"[\s\S]{0,400}rata\.digitraffic\.fi/.test(routes),
    "Digitraffic fetch must send Accept-Encoding: gzip (406 required since 2026-07)");
});
