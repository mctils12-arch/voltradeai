// Book tape, second line (2026-11-02, human-asked: "we have an option up 300%,
// what is the buy price and the total in the position, I want to see that on
// the UI"). Pins that every Book row carries entry · cost · value, that an
// option's cost uses the 100-share multiplier, and that the row wraps so the
// tape's columns keep their widths at every tier.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";

const here = path.dirname(new URL(import.meta.url).pathname);
const panels = readFileSync(path.resolve(here, "./panels.tsx"), "utf8");
const css = readFileSync(path.resolve(here, "../index.css"), "utf8");
const book = panels.slice(panels.indexOf("export function BookBody"), panels.indexOf("export function OrdersBody"));

test("every Book row shows the entry price, the cost basis and the current value", () => {
  assert.match(book, /const entry = num\(p\.entryPrice\); const mv = num\(p\.marketValue\); const q = num\(p\.qty\);/);
  assert.match(book, /const basis = entry !== null && q !== null \? entry \* Math\.abs\(q\) \* \(isOpt\(p\) \? 100 : 1\) : null;/, "an option's cost is price x 100 x contracts");
  assert.match(book, /<span className="sub">entry \{entry === null \? "·" : `\$\$\{entry\.toFixed\(2\)\}`\}\{isOpt\(p\) \? "\/ct" : ""\} · cost \{basis === null/, "entry per contract, then cost");
  assert.match(book, /· value \{mv === null \? "·" : `\$\$\{Math\.round\(mv\)\.toLocaleString\("en-US"\)\}`\}<\/span>/, "then the position's market value");
});

test("the row wraps so the second line sits under the columns", () => {
  assert.match(book, /className="vt-bot-trow vt-bot-trow--book" role="button"/);
  assert.match(css, /\.vt-bot-trow--book \{ flex-wrap: wrap;/);
  assert.match(css, /\.vt-bot-trow \.sub \{ flex: 1 0 100%;/);
});
