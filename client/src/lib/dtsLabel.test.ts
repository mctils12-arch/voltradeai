import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dtsCategoryLabel } from "./dtsLabel.ts";

test("a published category is shown as published", () => {
  assert.equal(dtsCategoryLabel("Withheld Income and Employment Taxes"), "Withheld Income and Employment Taxes");
  assert.equal(dtsCategoryLabel("  Public Debt Cash Issues (Table IIIB) "), "Public Debt Cash Issues (Table IIIB)");
});

test("FiscalData's literal \"null\" on roll-up rows is a dash, never the word", () => {
  for (const v of ["null", "NULL", " null ", "", "   ", null, undefined]) {
    assert.equal(dtsCategoryLabel(v), "—", JSON.stringify(v));
  }
});

test("a category that merely contains the letters is left alone", () => {
  assert.equal(dtsCategoryLabel("Nullification Fund"), "Nullification Fund");
});

test("SOURCE WIRING: the page words the category through dtsCategoryLabel", () => {
  const src = readFileSync(new URL("../pages/treasuryDts.tsx", import.meta.url), "utf8");
  assert.match(src, /\{dtsCategoryLabel\(l\.category\)\}/);
  assert.doesNotMatch(src, /className="vt-filings-ticker">\{l\.category\}</, "the raw field is printed again");
});
