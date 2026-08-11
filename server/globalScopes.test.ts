import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, readdirSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import {
  scopeUrl, volumeAllowsWrite, pollGlobalScopesOnce, GLOBAL_SCOPES, MIN_FREE_BYTES,
} from "./globalScopes";

test("the three free global scopes, one URL each", () => {
  assert.deepEqual(GLOBAL_SCOPES.map(scopeUrl), [
    "https://api.adsb.lol/v2/mil",
    "https://api.adsb.lol/v2/ladd",
    "https://api.adsb.lol/v2/pia",
  ]);
});

test("volumeAllowsWrite: measured-low blocks, healthy allows, unreadable fails OPEN", () => {
  assert.equal(volumeAllowsWrite(MIN_FREE_BYTES - 1), false, "below headroom -> paused (bot outranks archive)");
  assert.equal(volumeAllowsWrite(MIN_FREE_BYTES), true);
  assert.equal(volumeAllowsWrite(null), true, "stat error must not silently kill archiving");
});

test("pollGlobalScopesOnce: fetches all scopes, dedupes overlap, archives to real hour files", async () => {
  const base = mkdtempSync(path.join(tmpdir(), "gscope-"));
  const mk = (hex: string, lat: number) => ({
    hex, flight: "TEST01  ", r: "N1TEST", t: "C130",
    alt_baro: 20000, gs: 300, track: 90, category: "A5", lat, lon: 10,
  });
  const payloads: Record<string, any> = {
    mil: { ac: [mk("ae0001", 30), mk("ae0002", 31)] },
    ladd: { ac: [mk("ae0001", 30), mk("a11111", 40)] }, // ae0001 overlaps mil
    pia: { ac: [] },
  };
  const fetchImpl = (async (url: string) => ({
    ok: true,
    json: async () => payloads[String(url).split("/").pop()!],
  })) as unknown as typeof fetch;
  const st = await pollGlobalScopesOnce(fetchImpl, {}, [], base);
  assert.equal(st.per_scope.mil, 2);
  assert.equal(st.per_scope.ladd, 2);
  assert.equal(st.per_scope.pia, 0);
  assert.equal(st.skipped_low_disk, false);
  assert.equal(st.archived, 3, "overlap deduped: 2 + 2 + 0 -> 3 unique aircraft");
  const files = readdirSync(path.join(base, "aircraft"));
  assert.equal(files.length, 1);
  const lines = readFileSync(path.join(base, "aircraft", files[0]), "utf-8").trim().split("\n");
  assert.equal(lines.length, 3);
  assert.equal(JSON.parse(lines[0]).rg, "N1TEST", "registration flows into global-scope lines too");
  rmSync(base, { recursive: true, force: true });
});

test("pollGlobalScopesOnce: a failed scope reads -1 and never sinks the others", async () => {
  const base = mkdtempSync(path.join(tmpdir(), "gscope2-"));
  const fetchImpl = (async (url: string) => {
    if (String(url).endsWith("/mil")) throw new Error("boom");
    if (String(url).endsWith("/ladd")) return { ok: false, status: 503 };
    return { ok: true, json: async () => ({ ac: [{ hex: "a22222", r: "N2TEST", lat: 1, lon: 2, alt_baro: 100, category: "A1" }] }) };
  }) as unknown as typeof fetch;
  const st = await pollGlobalScopesOnce(fetchImpl, {}, [], base);
  assert.equal(st.per_scope.mil, -1);
  assert.equal(st.per_scope.ladd, -1);
  assert.equal(st.per_scope.pia, 1);
  assert.equal(st.archived, 1);
  rmSync(base, { recursive: true, force: true });
});
