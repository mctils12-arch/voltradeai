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
  assert.match(book, /<span className="sub"><i>entry<\/i>\{entry === null \? "·" : `\$\$\{entry\.toFixed\(2\)\}`\}\{isOpt\(p\) \? "\/ct" : ""\} · <i>cost<\/i>\{basis === null/, "entry per contract, then cost, labels dimmed");
  assert.match(book, /· <i>value<\/i>\{mv === null \? "·" : `\$\$\{Math\.round\(mv\)\.toLocaleString\("en-US"\)\}`\}<\/span>/, "then the position's market value");
});

test("the Book panel gets the legibility pass without breaking the tape contract", () => {
  assert.match(css, /section\.vt-bot-panel\[data-panel="book"\] \.vt-bot-trow \{ font-size: 13px;/, "13 px rows inside the Book only");
  assert.match(css, /section\.vt-bot-panel\[data-panel="book"\] \.vt-bot-trow \.sub \{ font-size: 11\.5px; color: var\(--text-secondary\);/, "the second line reads in the secondary color");
  assert.match(css, /@container \(max-width: 379px\) \{ section\.vt-bot-panel\[data-panel="book"\] \.vt-bot-trow \.pl \{ width: 60px; \} \}/, "narrow Books keep the row inside the panel");
});

test("the row wraps so the second line sits under the columns", () => {
  assert.match(book, /className="vt-bot-trow vt-bot-trow--book" role="button"/);
  assert.match(css, /\.vt-bot-trow--book \{ flex-wrap: wrap;/);
  assert.match(css, /\.vt-bot-trow \.sub \{ flex: 1 0 100%;/);
});

// 2026-11-02, second question: "how do I know what type of option I have — did I
// sell a put or buy, was it a single option or a strategy … and when would you
// sell the position or a portion, is that shown". Lines 3–4 of every Book row.
test("every option row says what it is, and every row says when the engine exits it — from the shared engine mirrors, never retyped", () => {
  assert.match(book, /optionManagement\(p\.optionMeta\?\.strategy\) === "group"/, "sibling legs of one strategy are counted as the manager groups them");
  assert.match(panels, /import \{ optionExitPlan, optionManagement \} from "\.\.\/\.\.\/\.\.\/shared\/positionLevels";/);
  assert.match(panels, /import \{ optionWhatLine, optionExitSegments, stockExitSegments, type PlanSegment \} from "\.\/exitPlan";/);
  assert.match(book, /\{isOpt\(p\) \? <span className="sub sub--plan sub--what">\{segs\(optionWhatLine\(\{ qty: q \?\? 0, optionType: p\.optionMeta\?\.optionType, daysToExpiry: p\.optionMeta\?\.daysToExpiry, strategy: p\.optionMeta\?\.strategy, legsInGroup: legCount\.get\(groupKey\(p\)\) \?\? 1 \}\)\)\}<\/span> : null\}/, "line 3 on option rows only");
  assert.match(book, /<span className="sub sub--plan sub--exit"><i>exit<\/i>\{segs\(isOpt\(p\) \? optionExitSegments\(optionExitPlan\(\{ qty: q \?\? 0, entryPrice: entry, currentPrice: last, daysToExpiry: p\.optionMeta\?\.daysToExpiry \?\? null, strategy: p\.optionMeta\?\.strategy, legsInGroup: legCount\.get\(groupKey\(p\)\) \?\? 1 \}\)\) : stockExitSegments\(p\.levels\)\)\}<\/span>/, "line 4 on every row: the option plan or the share ladder");
  assert.match(book, /<b className="hit">\{sg\.text\}<\/b>/, "a rule the current reading already satisfies is marked");
  assert.match(book, /\{name === "options" \? <Help k="bookOptionExit" \/> : <Help k="bookStockExit" \/>\}/, "each group label explains its lines");
  assert.match(css, /\.vt-bot-trow \.sub--plan \{ white-space: normal; overflow: visible; text-overflow: clip; \}/, "the plan lines wrap — a rule is never clipped");
  assert.match(css, /\.vt-bot-trow \.sub b\.hit \{ font-weight: 600; color: var\(--accent-orange\); \}/);
});
