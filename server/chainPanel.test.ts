// Edge P3 fixed chain panel — ranking, write-once freeze, archive ordering.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { rankPanel, ensurePanel, loadPanel, panelPath, PANEL_SIZE } from "./chainPanel";
import { underlyingsFor, CHAIN_ALWAYS, CHAIN_MAX_UNDERLYINGS } from "./optionsChainArchive";

const cache = (n: number, dv = (i: number) => 1e9 - i) => ({
  candidates: Array.from({ length: n }, (_, i) => [`T${String.fromCharCode(65 + (i % 26))}${String.fromCharCode(65 + Math.floor(i / 26) % 26)}`, 10, 1, dv(i)]),
});
const tmp = () => fs.mkdtempSync(path.join(os.tmpdir(), "panel-"));

test("rankPanel: top dollar volume, deduped, junk dropped, capped", () => {
  const c = { candidates: [["LOW", 1, 1, 5], ["HI", 1, 1, 50], ["HI", 1, 1, 999], ["BRK.B", 1, 1, 9e9], ["BAD", 1, 1, 0], ["MID", 1, 1, 20]] };
  assert.deepEqual(rankPanel(c), ["HI", "MID", "LOW"]);
  assert.equal(rankPanel(cache(500)).length, PANEL_SIZE);
});

test("ensurePanel: thin cache freezes nothing; full cache freezes once and never re-ranks", () => {
  const d = tmp();
  assert.equal(ensurePanel(cache(100), "2026-11-02", d), null);
  assert.equal(fs.existsSync(panelPath(d)), false);
  const first = ensurePanel(cache(400), "2026-11-02", d)!;
  assert.equal(first.names.length, PANEL_SIZE);
  assert.equal(first.formed, "2026-11-02");
  const reversed = ensurePanel(cache(400, (i) => i + 1), "2026-11-09", d)!;
  assert.deepEqual(reversed.names, first.names, "a later, different ranking must not replace the frozen panel");
  assert.equal(loadPanel(d)!.formed, "2026-11-02");
});

test("underlyingsFor: panel archived right after the index names, rotation keeps its slots", () => {
  const panel = Array.from({ length: 300 }, (_, i) => `P${i}`);
  const big = { candidates: Array.from({ length: 500 }, (_, i) => [`R${i}`, 10, 1, 1]) };
  const u = underlyingsFor(big, [], CHAIN_MAX_UNDERLYINGS, panel);
  assert.deepEqual(u.slice(0, 3).map((x) => x.sym), [...CHAIN_ALWAYS]);
  assert.deepEqual(u.slice(3, 303).map((x) => x.sym), panel);
  assert.equal(u.length, CHAIN_MAX_UNDERLYINGS + panel.length);
  assert.ok(u.some((x) => x.sym === "R0"), "rotating candidates still archived");
  assert.equal(underlyingsFor(big).length, CHAIN_MAX_UNDERLYINGS, "no panel → unchanged behaviour");
});
