// PHONE TEXT FLOOR (2026-11-26): the stylesheet's contract, read from index.css itself.
// At 640 px and under, a named set of small labels is raised to 11 px (DESIGN.md's legible
// floor at 390 px). This pins: the block exists, no later rule undoes it,
// every class it names still has a rule of its own (so the list holds no dead name), and each
// of those rules is under 11 px at desktop (so the block is not raising something already fine).
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const css = readFileSync(fileURLToPath(new URL("../index.css", import.meta.url)), "utf8");
const START = "/* ── PHONE TEXT FLOOR (2026-11-26";
const at = css.indexOf(START);
const block = at >= 0 ? css.slice(at) : "";
const m = /@media \(max-width: 640px\) \{\s*([^{}]+)\{\s*font-size: 11px;\s*\}\s*\}/.exec(block);
const classes = m ? m[1].split(",").map((s) => s.trim()).filter(Boolean) : [];
const before = at >= 0 ? css.slice(0, at) : css;
const after = m ? block.slice(block.indexOf(m[0]) + m[0].length) : "";

/** font sizes (px) of every rule in `text` whose selector list names `cls` exactly; rem is 16 px */
function sizesIn(text: string, cls: string): number[] {
  const out: number[] = [];
  const esc = cls.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const re = new RegExp(`(^|[\\s,}])${esc}(?![\\w-])[^{}]*\\{([^{}]*)\\}`, "g");
  for (let r = re.exec(text); r; r = re.exec(text)) {
    const f = /font-size:\s*([\d.]+)(px|rem)/.exec(r[2]);
    if (f) out.push(Number(f[1]) * (f[2] === "rem" ? 16 : 1));
  }
  return out;
}

test("phone text floor: the block is present, at phone width only, and sets 11 px", () => {
  assert.ok(at > 0, "the PHONE TEXT FLOOR block is in index.css");
  assert.ok(m, "it is one @media (max-width: 640px) rule that sets font-size: 11px");
  assert.equal(classes.length, 18);
  assert.ok(classes.every((c) => /^\.[a-z][\w-]*$/.test(c)), `plain class selectors only: ${classes.join(" ")}`);
  assert.equal(new Set(classes).size, classes.length, "no class twice");
});

test("phone text floor: no later rule puts one of its classes back under 11 px", () => {
  // other work appends to index.css; that is fine unless it re-sets one of these sizes after the block
  for (const c of classes) {
    for (const px of sizesIn(after, c)) assert.ok(px >= 11, `${c} is set to ${px}px after the floor block`);
  }
});

test("phone text floor: every class it names has a rule of its own, and that rule is under 11 px", () => {
  for (const c of classes) {
    // the Signals source line takes its size from its row: it has a rule, with no size of its own
    if (c === ".vt-ladder-row-src") { assert.ok(before.includes(".vt-ladder-row-src {"), c); continue; }
    const sizes = sizesIn(before, c);
    assert.ok(sizes.length > 0, `${c} has no font-size rule of its own before the block: a dead name in the list`);
    assert.ok(Math.min(...sizes) < 11, `${c} is never under 11 px (${sizes.join(", ")}): it does not need the floor`);
  }
});

test("phone text floor: the size reader tells an exact class from a longer one", () => {
  const sample = ".a-chip { font-size: 10px; } .a-chip-on { font-size: 9px; } .x, .a-chip { font-size: 0.68rem; }";
  assert.deepEqual(sizesIn(sample, ".a-chip"), [10, 10.88]);
  assert.deepEqual(sizesIn(sample, ".a-chip-on"), [9]);
  assert.deepEqual(sizesIn(sample, ".missing"), []);
});
