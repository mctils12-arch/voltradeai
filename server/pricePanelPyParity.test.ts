// Decoder parity: price_panel.py (the Python reader every long-term test
// stands on) must decode exactly what server/pricePanel.ts encodes.
// Run: npx tsx --test server/pricePanelPyParity.test.ts
//
// The committed fixture (server/__fixtures__/price_panel/) is produced by the
// TS encoder from inputs.json; test_price_panel.py decodes it in the Python-
// only CI job against expected_decoded.json (the TS decoder's output). This
// file (a) proves the committed shard is still what the current encoder
// writes, (b) runs the Python decoder on it and on a seeded fuzz panel and
// compares every number exactly.
// Regenerate after an encoder change: WRITE_PRICE_PANEL_FIXTURE=1 npx tsx --test server/pricePanelPyParity.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { decodeSegment, linesFor, type PanelLine, type Row, type UniverseEntry } from "./pricePanel";

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.join(here, "..");
const FIX = path.join(here, "__fixtures__", "price_panel");

interface FixtureCase { u: UniverseEntry; mode: "default" | "raw"; rows: Row[] }
interface FixtureInputs { build_id: string; as_of: string; cases: FixtureCase[] }
interface Decoded { s: string; mode: string; seg: number; rows: Array<{ d: string; o: number; h: number; l: number; c: number; v: number; rc: number | null }> }

function encodeAll(cases: FixtureCase[]): PanelLine[] {
  return cases.flatMap((k) => linesFor(k.u, k.mode, k.rows).lines);
}

function decodeAll(lines: PanelLine[]): Decoded[] {
  return lines.map((l) => ({ s: l.s, mode: l.mode, seg: l.seg, rows: decodeSegment(l) }));
}

function writePanel(root: string, buildId: string, asOf: string, lines: PanelLine[]): void {
  const dir = path.join(root, buildId, "bars");
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, "b00000.jsonl.gz"), zlib.gzipSync(Buffer.from(lines.map((l) => JSON.stringify(l)).join("\n") + "\n")));
  fs.writeFileSync(path.join(root, buildId, "manifest.json"), JSON.stringify({ version: 1, build_id: buildId, status: "complete", as_of: asOf, adjustment: "all", start: "2016-01-01" }));
  fs.writeFileSync(path.join(root, "current.json"), JSON.stringify({ build_id: buildId, completed_at: `${asOf}T23:00:00.000Z` }));
}

function pythonDump(root: string): Decoded[] {
  const out = execFileSync("python3", [path.join(repo, "price_panel.py"), "--dump", root], { cwd: repo, encoding: "utf8", maxBuffer: 64 * 1024 * 1024 });
  return out.split("\n").filter(Boolean).map((l) => JSON.parse(l) as Decoded);
}

const inputs = JSON.parse(fs.readFileSync(path.join(FIX, "inputs.json"), "utf8")) as FixtureInputs;

if (process.env.WRITE_PRICE_PANEL_FIXTURE === "1") {
  const lines = encodeAll(inputs.cases);
  writePanel(FIX, inputs.build_id, inputs.as_of, lines);
  fs.writeFileSync(path.join(FIX, "expected_decoded.json"), JSON.stringify(decodeAll(lines)));
}

test("committed fixture shard is exactly what the current TS encoder writes from inputs.json", () => {
  const shard = path.join(FIX, inputs.build_id, "bars", "b00000.jsonl.gz");
  const committed = zlib.gunzipSync(fs.readFileSync(shard)).toString("utf8").split("\n").filter(Boolean);
  const now = encodeAll(inputs.cases).map((l) => JSON.stringify(l));
  assert.deepEqual(committed, now, "encoder drifted: regenerate the fixture (see header) and re-check price_panel.py");
  const expected = JSON.parse(fs.readFileSync(path.join(FIX, "expected_decoded.json"), "utf8")) as Decoded[];
  assert.deepEqual(expected, decodeAll(encodeAll(inputs.cases)));
});

test("fixture covers the cases the reader must survive", () => {
  const lines = encodeAll(inputs.cases);
  const flags = new Set(lines.flatMap((l) => l.flags));
  for (const f of ["clipped_after_event", "inactive_no_event_date", "reuse_suspect"]) assert.ok(flags.has(f), `no ${f} line`);
  assert.ok(lines.some((l) => l.mode === "raw"), "a raw (asof=-) line");
  assert.ok(lines.some((l) => l.seg > 0), "a second date segment");
  assert.ok(lines.some((l) => l.k >= 6), "a sub-cent price scale");
  assert.ok(lines.some((l) => l.f.length > 1), "a raw-close factor change (split)");
  assert.ok(lines.some((l) => l.f.length === 0), "a line with no raw close");
});

test("price_panel.py decodes the committed fixture number for number", () => {
  assert.deepEqual(pythonDump(FIX), decodeAll(encodeAll(inputs.cases)));
});

test("price_panel.py decodes a seeded fuzz panel number for number (prices 1e-4..5e4, gaps, factor runs)", () => {
  let seed = 20261029;
  const rnd = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648; };
  const cases: FixtureCase[] = [];
  for (let s = 0; s < 40; s++) {
    const base = 10 ** (rnd() * 8.7 - 4);
    const n = 5 + Math.floor(rnd() * 120);
    let t = Date.parse("2016-01-04T00:00:00Z") + Math.floor(rnd() * 2000) * 86_400_000;
    let p = base;
    const rows: Row[] = [];
    let ratio = 1 + rnd() * 3;
    for (let i = 0; i < n; i++) {
      t += (rnd() < 0.03 ? 40 : 1 + Math.floor(rnd() * 3)) * 86_400_000;
      p *= Math.exp((rnd() - 0.5) * 0.1);
      if (rnd() < 0.05) ratio = 1 + rnd() * 3;
      const c = Number(p.toPrecision(7));
      rows.push({ d: new Date(t).toISOString().slice(0, 10), o: c * (1 + (rnd() - 0.5) * 0.02), h: c * (1 + rnd() * 0.03), l: c * (1 - rnd() * 0.03), c, v: Math.floor(rnd() * 5e7), rc: rnd() < 0.1 ? null : c * ratio });
    }
    cases.push({ u: { symbol: `F${s}`, status: s % 3 ? "active" : "inactive", ids: [`id${s}`], event: null, reuse_suspect: false }, mode: "default", rows });
  }
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "pp-parity-"));
  try {
    const lines = encodeAll(cases);
    writePanel(root, "b20261030", "2026-10-29", lines);
    assert.deepEqual(pythonDump(root), decodeAll(lines));
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});
