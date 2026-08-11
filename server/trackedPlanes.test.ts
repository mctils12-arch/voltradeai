import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import {
  normalizeReg, loadRegistry, saveRegistry, addTracked, removeTracked,
  batchUrl, pollTrackedOnce, registryPath, TRACKED_CAP,
  type TrackedRegistry,
} from "./trackedPlanes";

test("normalizeReg: real tails pass, junk rejected", () => {
  assert.equal(normalizeReg("n843s"), "N843S");
  assert.equal(normalizeReg(" G-EZTL "), "G-EZTL");
  assert.equal(normalizeReg("VH-OQA"), "VH-OQA");
  assert.equal(normalizeReg(""), null);
  assert.equal(normalizeReg("-BAD"), null);
  assert.equal(normalizeReg("WAY-TOO-LONG-REG"), null);
  assert.equal(normalizeReg(null), null);
});

test("loadRegistry seeds the human's named plane when no file exists", () => {
  const base = mkdtempSync(path.join(tmpdir(), "tracked-"));
  const r = loadRegistry(base);
  assert.equal(r.planes.length, 1);
  assert.equal(r.planes[0].reg, "N843S");
  saveRegistry(r, base);
  assert.ok(existsSync(registryPath(base)));
  const again = loadRegistry(base);
  assert.equal(again.planes[0].reg, "N843S");
  rmSync(base, { recursive: true, force: true });
});

test("addTracked: idempotent, validated, capped at TRACKED_CAP", () => {
  let r: TrackedRegistry = { planes: [] };
  r = addTracked(r, "n843s")!;
  assert.equal(r.planes.length, 1);
  assert.equal(addTracked(r, "N843S"), r, "duplicate add is a no-op, same object");
  assert.equal(addTracked(r, "!!bad!!"), null, "invalid tail refused");
  for (let i = 0; i < TRACKED_CAP + 5; i++) {
    const next = addTracked(r, `N${100 + i}AB`);
    if (next) r = next;
  }
  assert.equal(r.planes.length, TRACKED_CAP, "cap holds — provider politeness bound");
});

test("removeTracked round-trips", () => {
  let r = addTracked({ planes: [] }, "N843S")!;
  r = removeTracked(r, "n843s");
  assert.equal(r.planes.length, 0);
});

test("batchUrl: ONE request for the whole registry", () => {
  assert.equal(batchUrl(["N843S", "G-EZTL"]), "https://api.adsb.lol/v2/reg/N843S,G-EZTL");
});

test("pollTrackedOnce: live fix updates last_seen/hex/pos; dark planes keep aging honestly", async () => {
  const reg: TrackedRegistry = {
    planes: [
      { reg: "N843S", added_at: 1, last_seen: 500 },
      { reg: "N999XX", added_at: 1, last_seen: 500 }, // no signal this cycle
    ],
  };
  // real adsb.lol /v2/reg shape (probe 2026-08-08: the actual N843S response)
  const payload = {
    ac: [{ hex: "ab8c8e", flight: "N843S   ", r: "N843S", t: "FA7X",
           alt_baro: 18175, gs: 410.3, track: 92.1, category: "A2",
           lat: 35.308273, lon: -80.357094 }],
    msg: "No error",
  };
  const fetchImpl = (async () => ({ ok: true, json: async () => payload })) as unknown as typeof fetch;
  const out = await pollTrackedOnce(reg, fetchImpl, {}, 1_000_000);
  assert.equal(out.points.length, 1);
  assert.equal((out.points[0] as any).registration, "N843S", "T1's honest field name flows here too");
  const tracked = out.registry.planes.find((p) => p.reg === "N843S")!;
  assert.equal(tracked.last_seen, 1_000_000);
  assert.equal(tracked.hex, "ab8c8e");
  assert.equal(tracked.last_pos?.al, Math.round(18175 * 0.3048));
  const dark = out.registry.planes.find((p) => p.reg === "N999XX")!;
  assert.equal(dark.last_seen, 500, "no signal -> last_seen untouched, never fabricated");
});

test("pollTrackedOnce: upstream error throws (caller treats a cycle as a missed sample)", async () => {
  const fetchImpl = (async () => ({ ok: false, status: 503 })) as unknown as typeof fetch;
  await assert.rejects(
    () => pollTrackedOnce({ planes: [{ reg: "N843S", added_at: 1 }] }, fetchImpl, {}, 1),
    /503/,
  );
});
