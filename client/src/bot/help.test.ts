import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { GLOSSARY, GLOSSARY_KEYS, glossary } from "./glossary";
import { placeCard, CARD_W, CARD_MARGIN, CARD_GAP } from "./help";

test("every glossary entry is a term with one plain line, short enough for the card", () => {
  assert.ok(GLOSSARY_KEYS.length >= 50);
  for (const k of GLOSSARY_KEYS) {
    const e = glossary(k);
    assert.ok(e.term.length > 0 && e.term.length <= 40, `${k}: term`);
    assert.ok(e.text.length >= 40 && e.text.length <= 340, `${k}: text ${e.text.length}`);
    assert.ok(!/\bTODO\b|lorem/i.test(e.text), `${k}: placeholder`);
  }
  assert.equal(new Set(GLOSSARY_KEYS).size, GLOSSARY_KEYS.length);
});

test("every <Help k=…> the panels use names a glossary entry (a dangling key would render a crash, not a card)", () => {
  const dir = join(process.cwd(), "client", "src", "bot");
  const used = new Set<string>();
  const walk = (d: string) => {
    for (const f of readdirSync(d, { withFileTypes: true })) {
      if (f.isDirectory()) { walk(join(d, f.name)); continue; }
      if (!/\.tsx?$/.test(f.name) || /\.test\./.test(f.name)) continue;
      const src = readFileSync(join(d, f.name), "utf8");
      for (const m of src.matchAll(/<Help\s+k="([a-zA-Z0-9_]+)"/g)) used.add(m[1]);
      // a panel's help prop names an entry too — literal or chosen in an expression
      for (const m of src.matchAll(/\bhelp="([a-zA-Z0-9_]+)"/g)) used.add(m[1]);
      for (const expr of src.matchAll(/\bhelp=\{([^}]*)\}/g)) for (const m of expr[1].matchAll(/[?:]\s*"([a-zA-Z0-9_]+)"/g)) used.add(m[1]); // the ternary's branches, not its conditions
      // and EVERY value of a map typed Record<string, GlossaryKey> (the chart legend)
      for (const map of src.matchAll(/GlossaryKey>\s*=\s*\{([^}]*)\}/g)) for (const m of map[1].matchAll(/:\s*"([a-zA-Z0-9_]+)"/g)) used.add(m[1]);
    }
  };
  walk(dir);
  assert.ok(used.size >= 30, `only ${used.size} help keys found in the panels`);
  for (const k of used) assert.ok(k in GLOSSARY, `glossary has no entry for "${k}"`);
});

test("placeCard: below the anchor by default, above when the bottom would leave the window, clamped horizontally, never over the anchor", () => {
  const win = { w: 1440, h: 900 };
  const a = { left: 300, top: 100, width: 14, height: 14 };
  const p = placeCard(a, 80, win);
  assert.equal(p.above, false);
  assert.equal(p.top, 100 + 14 + CARD_GAP);
  assert.equal(p.left, 300);
  const low = placeCard({ ...a, top: 860 }, 80, win);
  assert.equal(low.above, true);
  assert.equal(low.top, 860 - CARD_GAP - 80);
  assert.ok(low.top + 80 <= 860);
  const right = placeCard({ ...a, left: 1430 }, 80, win);
  assert.equal(right.left, win.w - CARD_W - CARD_MARGIN);
  const leftEdge = placeCard({ ...a, left: 2 }, 80, win);
  assert.equal(leftEdge.left, CARD_MARGIN);
  // a tiny window: neither side fits — the card sits at the bottom margin, still inside
  const tiny = placeCard({ left: 10, top: 40, width: 14, height: 14 }, 300, { w: 390, h: 320 });
  assert.ok(tiny.top >= CARD_MARGIN && tiny.top + 300 <= 320);
});
