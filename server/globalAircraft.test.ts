// FLIGHT PROGRAM B1 — GET /api/data/aircraft/global end-to-end on a real
// Express app: contract shape, honest coverage block, bbox/lawful/since
// filters, archive recording for sweep/OpenSky origins only (viewport and
// scope fixes archive themselves), and the ONE-line routes.ts wiring.
// Sweep disabled + no OpenSky creds → zero upstream calls.
import { test } from "node:test";
import assert from "node:assert/strict";
import express from "express";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type { AddressInfo } from "node:net";
import {
  registerGlobalAircraftRoutes, parseBBoxQuery, sweepArchiveAllowed,
  GLOBAL_HONESTY, SWEEP_ARCHIVE_MIN_FREE_BYTES,
} from "./globalAircraft";
import { MIN_FREE_BYTES } from "./globalScopes";
import { publishFixes } from "./aircraftFixBus";
import { ROW_FIELDS } from "./globalSnapshot";
import type { AircraftPoint } from "./datacoreArchive";

type Row = (string | number | null)[];
interface GlobalResp {
  at: number; count: number; full: boolean; scope: string; fields: string[]; rows: Row[]; honesty: string;
  coverage: {
    discs_in_plan: number; note: string;
    sweep: { enabled: boolean }; opensky: { enabled: boolean };
    sources: { key: string; commercial_ok: boolean }[];
    archive: {
      enabled: boolean; paused: boolean; min_free_bytes: number; lines_archived_total: number;
      fixes_offered_total: number; fixes_skipped_low_disk_total: number; fixes_skipped_disabled_total: number;
    };
    [k: string]: unknown;
  };
}
const getJson = async (url: string): Promise<GlobalResp> => (await (await fetch(url)).json()) as GlobalResp;
const hexes = (d: GlobalResp): string[] => d.rows.map((r) => String(r[0])).sort();

const here = path.dirname(fileURLToPath(import.meta.url));

type Fix = AircraftPoint & { seen_at_ms?: number | null };
const ac = (hex: string, lat: number, lon: number, o: Partial<Fix> = {}): Fix => ({
  icao24: hex, callsign: `C${hex}`, lat, lon, altitude_m: 10000, on_ground: false,
  velocity_ms: 230, heading: 45, type: "B738", category: "A3", registration: `N${hex}`, seen_pos: 1, ...o,
});

const GiB = 1_073_741_824;

async function withApp(
  fn: (base: string, h: ReturnType<typeof registerGlobalAircraftRoutes>, archiveDir: string) => Promise<void>,
  opts: { free?: number | null; env?: Record<string, string> } = {},
) {
  const archiveDir = fs.mkdtempSync(path.join(os.tmpdir(), "vt-global-"));
  let calls = 0;
  const app = express();
  const h = registerGlobalAircraftRoutes(app, {
    env: { GLOBAL_SWEEP_ENABLED: "0", ...(opts.env || {}) } as NodeJS.ProcessEnv,
    fetchImpl: (async () => { calls++; throw new Error("no network in tests"); }) as unknown as typeof fetch,
    sites: [], baseDir: archiveDir,
    readFree: () => (opts.free === undefined ? 50 * GiB : opts.free),
  });
  const srv = app.listen(0);
  try {
    const port = (srv.address() as AddressInfo).port;
    await fn(`http://127.0.0.1:${port}`, h, archiveDir);
  } finally {
    h.stop();
    srv.close();
  }
  assert.equal(calls, 0, "sweep off + no OpenSky creds -> zero upstream calls");
}

test("contract: { at, count, fields, rows, coverage, honesty } with positional rows", async () => {
  await withApp(async (base) => {
    const now = Date.now();
    publishFixes({ provider: "adsblol", origin: "sweep-type", aircraft: [ac("aa0001", 40, -75), ac("aa0002", 51, 0)], fetchedAt: now, upstreamNowMs: now });
    publishFixes({ provider: "opensky", origin: "opensky", aircraft: [{ ...ac("bb0001", -33, 151), type: null, seen_at_ms: now - 5000 }], fetchedAt: now });
    const r = await fetch(`${base}/api/data/aircraft/global`);
    assert.equal(r.status, 200);
    assert.match(r.headers.get("cache-control") || "", /max-age=10/);
    assert.ok(r.headers.get("etag"), "weak ETag for cheap revalidation");
    const d = (await r.json()) as GlobalResp;
    assert.deepEqual(d.fields, [...ROW_FIELDS]);
    assert.equal(d.count, 3);
    assert.equal(d.rows.length, 3);
    assert.ok(Math.abs(d.at - now) < 5000);
    const byHex = Object.fromEntries(d.rows.map((row) => [String(row[0]), Object.fromEntries(d.fields.map((f, i) => [f, row[i]]))]));
    assert.equal(byHex.aa0001.src, "adsblol");
    assert.equal(byHex.aa0001.altFt, 32808, "10000 m in feet");
    assert.equal(byHex.aa0001.gsKt, 447);
    assert.equal(byHex.bb0001.src, "opensky");
    assert.equal(byHex.bb0001.seenAt, now - 5000);
    // honest coverage block
    const c = d.coverage;
    for (const k of ["discs_in_plan", "discs_refreshed_last_cycle", "oldest_disc_age_s", "type_lane", "sources", "snapshot", "note", "opensky", "sweep"]) {
      assert.ok(k in c, `coverage.${k} missing`);
    }
    assert.ok(c.discs_in_plan > 300);
    assert.match(c.note, /remote oceans/);
    assert.equal(c.sweep.enabled, false);
    assert.equal(c.opensky.enabled, false);
    const adsb = c.sources.find((s) => s.key === "adsblol")!;
    assert.equal(adsb.commercial_ok, true);
    assert.equal(c.sources.find((s) => s.key === "opensky")!.commercial_ok, false);
    assert.equal(d.honesty, GLOBAL_HONESTY);
    assert.match(d.honesty, /Not every plane on Earth/);
  });
});

test("filters: bbox (also registers viewer interest), lawful=1 (ODbL subset), since", async () => {
  await withApp(async (base, h) => {
    const now = Date.now();
    publishFixes({ provider: "adsblol", origin: "viewport", aircraft: [ac("cc0001", 40, -75)], fetchedAt: now, disc: { lat: 40, lon: -75, radiusNm: 50 } });
    publishFixes({ provider: "airplaneslive", origin: "viewport", aircraft: [ac("cc0002", 41, -74)], fetchedAt: now });
    publishFixes({ provider: "adsblol", origin: "scopes", aircraft: [ac("cc0003", 50, 8, { seen_pos: 30 })], fetchedAt: now });
    const us = await getJson(`${base}/api/data/aircraft/global?lamin=30&lamax=45&lomin=-80&lomax=-70`);
    assert.equal(us.scope, "bbox");
    assert.deepEqual(hexes(us), ["cc0001", "cc0002"]);
    const lawful = await getJson(`${base}/api/data/aircraft/global?lawful=1`);
    assert.deepEqual(hexes(lawful), ["cc0001", "cc0003"], "airplanes.live row excluded");
    const since = await getJson(`${base}/api/data/aircraft/global?since=${now - 10_000}`);
    assert.equal(since.full, false);
    assert.deepEqual(hexes(since), ["cc0001", "cc0002"], "the 30s-old scope fix is before `since`");
    assert.equal(h.snapshot.size, 3, "viewport + scope fixes reach the snapshot through the bus (no refetch)");
  });
});

test("recording: sweep/OpenSky fixes archived (with provenance); viewport/scope fixes are NOT re-archived here", async () => {
  await withApp(async (_base, h, dir) => {
    const now = Date.now();
    h.onBatch({ provider: "adsblol", origin: "sweep-disc", aircraft: [ac("dd0001", 10, 10)], fetchedAt: now });
    h.onBatch({ provider: "opensky", origin: "opensky", aircraft: [{ ...ac("dd0002", 11, 11), provider: "opensky" }], fetchedAt: now });
    h.onBatch({ provider: "adsblol", origin: "viewport", aircraft: [ac("dd0003", 12, 12)], fetchedAt: now });
    h.onBatch({ provider: "adsblol", origin: "scopes", aircraft: [ac("dd0004", 13, 13)], fetchedAt: now });
    const files = fs.readdirSync(path.join(dir, "aircraft"));
    const lines = files.flatMap((f) => fs.readFileSync(path.join(dir, "aircraft", f), "utf8").split("\n").filter(Boolean)).map((l) => JSON.parse(l));
    assert.deepEqual(lines.map((l) => l.i).sort(), ["dd0001", "dd0002"]);
    assert.equal(lines.find((l) => l.i === "dd0002").pv, "opensky", "provenance persisted in the archive");
  });
});

test("stricter sweep gate: 2 GiB floor (2x the shared 1 GiB), fails CLOSED when unreadable", () => {
  assert.equal(SWEEP_ARCHIVE_MIN_FREE_BYTES, 2 * MIN_FREE_BYTES);
  assert.equal(sweepArchiveAllowed(1.24 * GiB), false, "today's measured prod headroom -> sweep archive paused");
  assert.equal(sweepArchiveAllowed(2 * GiB), true);
  assert.equal(sweepArchiveAllowed(null), false, "unreadable -> skip (the shared guard fails open; this one must not)");
});

test("low disk: sweep fixes still served LIVE but not archived; counters say so", async () => {
  await withApp(async (base, h, dir) => {
    const now = Date.now();
    h.onBatch({ provider: "adsblol", origin: "sweep-type", aircraft: [ac("ee0001", 10, 10), ac("ee0002", 11, 11)], fetchedAt: now });
    assert.equal(fs.existsSync(path.join(dir, "aircraft")), false, "nothing written under the 2 GiB floor");
    const d = await getJson(`${base}/api/data/aircraft/global`);
    assert.equal(d.count, 2, "live snapshot unaffected");
    const a = d.coverage.archive;
    assert.equal(a.paused, true);
    assert.equal(a.fixes_offered_total, 2);
    assert.equal(a.fixes_skipped_low_disk_total, 2);
    assert.equal(a.lines_archived_total, 0);
    assert.equal(a.min_free_bytes, 2 * GiB);
  }, { free: 1.24 * GiB });
  await withApp(async (_base, h, dir) => {
    h.onBatch({ provider: "adsblol", origin: "sweep-disc", aircraft: [ac("ee0003", 10, 10)], fetchedAt: Date.now() });
    assert.equal(fs.existsSync(path.join(dir, "aircraft")), false, "unreadable free space -> fail closed");
  }, { free: null });
});

test("GLOBAL_SWEEP_ARCHIVE=0: live snapshot on, archive writes off, counted as disabled", async () => {
  await withApp(async (base, h, dir) => {
    h.onBatch({ provider: "adsblol", origin: "sweep-type", aircraft: [ac("ff0001", 10, 10)], fetchedAt: Date.now() });
    assert.equal(fs.existsSync(path.join(dir, "aircraft")), false);
    const d = await getJson(`${base}/api/data/aircraft/global`);
    assert.equal(d.count, 1);
    assert.equal(d.coverage.archive.enabled, false);
    assert.equal(d.coverage.archive.fixes_skipped_disabled_total, 1);
  }, { env: { GLOBAL_SWEEP_ARCHIVE: "0" } });
});

test("parseBBoxQuery: all four params or nothing; whole-world = no filter", () => {
  assert.equal(parseBBoxQuery({ lamin: "1", lamax: "2", lomin: "3" }), null);
  assert.equal(parseBBoxQuery({ lamin: "-85", lamax: "85", lomin: "-180", lomax: "180" }), null);
  assert.deepEqual(parseBBoxQuery({ lamin: "45", lamax: "30", lomin: "-80", lomax: "-70" }), { lamin: 30, lamax: 45, lomin: -80, lomax: -70 });
  assert.deepEqual(parseBBoxQuery({ lamin: "-20", lamax: "-10", lomin: "170", lomax: "-170" }),
    { lamin: -20, lamax: -10, lomin: 170, lomax: -170 }, "antimeridian bbox kept as lomin > lomax");
});

test("routes.ts wiring: exactly one registration line + import", () => {
  const src = fs.readFileSync(path.join(here, "routes.ts"), "utf8");
  assert.equal(src.split("registerGlobalAircraftRoutes(app)").length - 1, 1);
  assert.ok(src.includes('import { registerGlobalAircraftRoutes } from "./globalAircraft"'));
  // the OpenSky removal pins on routes.ts stay true (OpenSky lives in its own gated module)
  assert.ok(!src.includes("OPENSKY_CLIENT_ID"));
});
