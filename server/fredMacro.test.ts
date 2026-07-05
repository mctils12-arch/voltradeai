import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import os from "os";
import path from "path";
import { fileURLToPath } from "url";
import {
  FRED_SERIES,
  FRED_ATTRIBUTION,
  parseObservations,
  archiveFredObs,
  _resetFredArchiveState,
  buildMacroPayload,
  refreshFredCache,
  latestFredSeries,
  fredEnabled,
  bootFredPoll,
} from "./fredMacro";

const here = path.dirname(fileURLToPath(import.meta.url));

// Documented FRED response shape (fred/series/observations, file_type=json):
// values are STRINGS and missing values are "." — both quirks pinned here.
// Gate-1 value accuracy is verified against the live fredgraph.csv ground
// truth post-deploy (the API key lives only on Railway), and recorded in
// research/experiments.md.
const OBS_JSON = {
  realtime_start: "2026-07-05", realtime_end: "2026-07-05",
  observation_start: "2026-06-25", observation_end: "9999-12-31",
  units: "lin", count: 4,
  observations: [
    { realtime_start: "2026-07-05", realtime_end: "2026-07-05", date: "2026-06-29", value: "4.38" },
    { realtime_start: "2026-07-05", realtime_end: "2026-07-05", date: "2026-06-30", value: "4.44" },
    { realtime_start: "2026-07-05", realtime_end: "2026-07-05", date: "2026-07-03", value: "." },
    { realtime_start: "2026-07-05", realtime_end: "2026-07-05", date: "2026-07-01", value: "4.48" },
  ],
};

test("parseObservations parses string values, skips '.' missing, stamps rt", () => {
  const obs = parseObservations("DGS10", OBS_JSON, "2026-07-05");
  assert.equal(obs.length, 3, "'.' row must be skipped");
  assert.deepEqual(obs[0], { s: "DGS10", d: "2026-06-29", v: 4.38, rt: "2026-07-05" });
  assert.deepEqual(obs[2], { s: "DGS10", d: "2026-07-01", v: 4.48, rt: "2026-07-05" });
});

test("series table: unique ids, valid licenses, third-party series pinned restricted", () => {
  const ids = FRED_SERIES.map((s) => s.id);
  assert.equal(new Set(ids).size, ids.length, "duplicate series id");
  for (const s of FRED_SERIES) assert.ok(["public", "restricted"].includes(s.license), s.id);
  // The licensing decision this module encodes: third-party copyrighted
  // series never reach a product surface. Pin the exact set.
  const restricted = FRED_SERIES.filter((s) => s.license === "restricted").map((s) => s.id).sort();
  assert.deepEqual(restricted, ["BAMLH0A0HYM2", "UMCSENT", "VIXCLS"]);
  assert.ok(FRED_SERIES.length >= 25, "regime feed needs breadth");
});

test("archive: dedup by (s,d,v); a REVISION appends as a new vintage row", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "vtfred-"));
  const t = Date.parse("2026-07-05T12:00:00Z");
  const first = [{ s: "DGS10", d: "2026-07-01", v: 4.48, rt: "2026-07-05" }];
  assert.equal(archiveFredObs(first, dir, t), 1);
  assert.equal(archiveFredObs(first, dir, t), 0, "same (s,d,v) never archives twice");
  // FRED revises 2026-07-01 from 4.48 to 4.5 — the vintage record keeps BOTH
  const revised = [{ s: "DGS10", d: "2026-07-01", v: 4.5, rt: "2026-07-06" }];
  assert.equal(archiveFredObs(revised, dir, t + 86400_000), 1, "revision must append");
  const rows = fs.readFileSync(path.join(dir, "fredmacro", "2026-07-05.jsonl"), "utf8").trim().split("\n")
    .concat(fs.readFileSync(path.join(dir, "fredmacro", "2026-07-06.jsonl"), "utf8").trim().split("\n"))
    .map((l) => JSON.parse(l));
  assert.deepEqual(rows.map((r) => r.v), [4.48, 4.5]);
  fs.rmSync(dir, { recursive: true, force: true });
});

test("refreshFredCache end-to-end with mocked API; public payload excludes restricted series", async () => {
  const requested: string[] = [];
  const mock = async (url: string) => {
    requested.push(url);
    const id = new URL(url).searchParams.get("series_id");
    return { ok: true, status: 200, text: async () => JSON.stringify({
      observations: [{ date: "2026-07-01", value: id === "VIXCLS" ? "17.2" : "4.48" }],
    }) };
  };
  await refreshFredCache({ FRED_API_KEY: "test-key" } as any, mock as any, 0, Date.parse("2026-07-05T12:00:00Z"));
  const hit = latestFredSeries();
  assert.ok(hit, "cache populated");
  assert.equal(hit!.series.length, FRED_SERIES.length, "every series snapshotted");
  assert.equal(requested.length, FRED_SERIES.length, "one request per series");
  assert.ok(requested.every((u) => u.includes("api.stlouisfed.org")), "only FRED hit");
  const vix = hit!.series.find((s) => s.id === "VIXCLS");
  assert.equal(vix?.latest?.v, 17.2, "restricted series still cached internally");
  const payload = buildMacroPayload(hit)!;
  assert.equal(payload.attribution, FRED_ATTRIBUTION);
  const ids = payload.series.map((s: any) => s.id);
  assert.ok(!ids.includes("VIXCLS") && !ids.includes("BAMLH0A0HYM2") && !ids.includes("UMCSENT"),
    "restricted series must NEVER appear in the public payload");
  assert.ok(ids.includes("DGS10") && payload.series.find((s: any) => s.id === "DGS10").latest.v === 4.48);
  assert.ok(payload.series.every((s: any) => s.license === undefined), "license field stripped from payload");
});

test("key gating: disabled without FRED_API_KEY; bootFredPoll no-ops", () => {
  assert.equal(fredEnabled({} as any), false);
  assert.equal(fredEnabled({ FRED_API_KEY: "x" } as any), true);
  bootFredPoll({} as any); // must not throw or start anything without a key
});

test("routes.ts boots the FRED poll and registers /api/data/macro; manifest exists", () => {
  const routes = fs.readFileSync(path.join(here, "routes.ts"), "utf8");
  assert.ok(routes.includes("bootFredPoll()"), "FRED poll must boot eagerly");
  assert.ok(routes.includes('"/api/data/macro"'), "macro route registered");
  const manifest = JSON.parse(fs.readFileSync(path.join(here, "..", "datacore", "manifests", "fredmacro.json"), "utf8"));
  assert.equal(manifest.stream, "fredmacro");
  assert.ok(String(manifest.attribution).includes("FRED"), "FRED attribution required by license");
  assert.ok(String(manifest.license).toLowerCase().includes("restricted"), "restricted-series rule stated");
  assert.ok(String(manifest.confidence_model).includes("rt"), "point-in-time vintage rule stated");
});

// ── [REPAIR 2026-07-05, audit defect #6] latest-value vintage dedup ─────────

test("archive: re-poll of a 120-day window after restart appends NOTHING (the duplicate-bloat bug)", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "vtfred6-"));
  const t0 = Date.parse("2026-07-05T12:00:00Z");
  const window = Array.from({ length: 10 }, (_, i) => ({
    s: "DGS10", d: `2026-06-${String(i + 1).padStart(2, "0")}`, v: 4 + i / 100, rt: "2026-07-05",
  }));
  assert.equal(archiveFredObs(window, dir, t0), 10);
  // simulate a REAL restart 10 days later: in-memory state cleared, the
  // 130-day disk seed must cover the whole fetch window so re-polling
  // identical values appends zero rows (the old 3-day seed re-appended
  // ~120d x 31 series as duplicates)
  _resetFredArchiveState();
  assert.equal(archiveFredObs(window.map(o => ({ ...o, rt: "2026-07-15" })), dir, t0 + 10 * 86400_000), 0,
    "restart + identical re-poll must not bloat the vintage record");
  fs.rmSync(dir, { recursive: true, force: true });
});

test("archive: a revision that REVERTS to a prior value is a new vintage row, not a silent drop", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "vtfred6b-"));
  const t0 = Date.parse("2026-07-05T12:00:00Z");
  const ob = (v: number, rt: string) => [{ s: "ICSA", d: "2026-06-27", v, rt }];
  assert.equal(archiveFredObs(ob(215000, "2026-07-05"), dir, t0), 1);
  assert.equal(archiveFredObs(ob(217000, "2026-07-06"), dir, t0 + 86400_000), 1, "revision appends");
  assert.equal(archiveFredObs(ob(215000, "2026-07-07"), dir, t0 + 2 * 86400_000), 1,
    "REVERT to the original value is a vintage transition — must append");
  assert.equal(archiveFredObs(ob(215000, "2026-07-08"), dir, t0 + 3 * 86400_000), 0,
    "unchanged value never re-appends");
  fs.rmSync(dir, { recursive: true, force: true });
});
