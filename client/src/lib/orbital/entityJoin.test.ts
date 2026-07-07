// entityJoin.test.ts — ORBITAL workstream e-entity. Hermetic: everything is
// bundled via imports (operators.json is a static seed) — no network, no fs.
//
// The load-bearing property under test is the HONESTY RULE: resolveOperator
// must map real SATCAT operator strings correctly AND return null (never a
// wrong company/ticker) for anything it hasn't seeded. A false map is a
// worse failure than an honest "unmapped".
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  normalizeOperator,
  resolveOperator,
  orbitalFootprintByTicker,
  groupOrbitalFootprint,
  OPERATORS,
  type JoinedSat,
} from "./entityJoin.ts";
import operatorsFile from "../../../../datacore/orbital/operators.json" with { type: "json" };

// ---------------------------------------------------------------------------
// normalizeOperator — the conservative canonicalizer
// ---------------------------------------------------------------------------
test("normalizeOperator strips parentheticals, punctuation, case, and legal-form suffixes", () => {
  assert.equal(normalizeOperator("Planet Labs, Inc."), "PLANET LABS");
  assert.equal(normalizeOperator("Space Exploration Technologies (SpaceX)"), "SPACE EXPLORATION TECHNOLOGIES");
  assert.equal(normalizeOperator("iridium communications"), "IRIDIUM COMMUNICATIONS");
  assert.equal(normalizeOperator("  Telesat   Canada  "), "TELESAT CANADA");
  // descriptor words are NOT stripped (they disambiguate distinct companies)
  assert.equal(normalizeOperator("Viasat Inc"), "VIASAT");
  assert.equal(normalizeOperator(""), "");
  assert.equal(normalizeOperator(null), "");
  assert.equal(normalizeOperator(undefined), "");
});

// ---------------------------------------------------------------------------
// resolveOperator — real-string HITS
// ---------------------------------------------------------------------------
test("resolveOperator maps canonical SATCAT operator strings to the right company", () => {
  const spacex = resolveOperator("SPACE EXPLORATION TECHNOLOGIES");
  assert.ok(spacex, "SpaceX must resolve");
  assert.match(spacex!.company, /SpaceX/);
  assert.equal(spacex!.category, "launch-provider");

  const iridium = resolveOperator("IRIDIUM");
  assert.ok(iridium);
  assert.equal(iridium!.ticker, "IRDM");
  assert.equal(iridium!.category, "comms");

  const planet = resolveOperator("PLANET LABS INC");
  assert.ok(planet);
  assert.equal(planet!.ticker, "PL");
  assert.equal(planet!.category, "earth-observation");
  assert.equal(planet!.is_public, true);
});

// ---------------------------------------------------------------------------
// resolveOperator — HONEST NULLS (no false match on an unseeded operator)
// ---------------------------------------------------------------------------
test("resolveOperator returns null for unmapped/obscure operators — never a wrong guess", () => {
  assert.equal(resolveOperator("KEPLER COMMUNICATIONS INC"), null);
  assert.equal(resolveOperator("SWARM TECHNOLOGIES"), null);
  assert.equal(resolveOperator("GEO-INFORMATICS AND SPACE TECHNOLOGY DEVELOPMENT AGENCY"), null);
  assert.equal(resolveOperator("ACME MICROSAT LTD"), null);
  assert.equal(resolveOperator(""), null);
  assert.equal(resolveOperator(null), null);
  // a partial token of a real entry must NOT false-match a different company
  assert.equal(resolveOperator("SATELLITE"), null);
  assert.equal(resolveOperator("COMMUNICATIONS"), null);
});

// ---------------------------------------------------------------------------
// resolveOperator — aliases + case-insensitivity
// ---------------------------------------------------------------------------
test("resolveOperator is case-insensitive and follows curated aliases", () => {
  // case variants of the same string
  assert.equal(resolveOperator("space exploration technologies")!.company, resolveOperator("SPACE EXPLORATION TECHNOLOGIES")!.company);
  // short alias -> same private parent
  assert.match(resolveOperator("SpaceX")!.company, /SpaceX/);
  assert.match(resolveOperator("Starlink")!.company, /SpaceX/);

  // M&A aliases fold into the acquirer
  assert.equal(resolveOperator("Inmarsat")!.ticker, "VSAT", "Inmarsat now rolls up to Viasat");
  assert.equal(resolveOperator("OneWeb")!.ticker, "ETL", "OneWeb merged into Eutelsat");
  assert.equal(resolveOperator("Hughes Network Systems")!.ticker, "ECHO", "Hughes is EchoStar");
  assert.equal(resolveOperator("O3b")!.ticker, "SESG", "O3b is SES");
});

// ---------------------------------------------------------------------------
// STALE-SYMBOL regression guard (this build's live correction)
// ---------------------------------------------------------------------------
test("EchoStar resolves to the current ECHO ticker, not the stale SATS", () => {
  const echo = resolveOperator("ECHOSTAR");
  assert.ok(echo);
  assert.equal(echo!.ticker, "ECHO");
  assert.notEqual(echo!.ticker, "SATS");
});

// ---------------------------------------------------------------------------
// mapped-but-private vs unmapped: both null ticker, DIFFERENT meaning
// ---------------------------------------------------------------------------
test("private/gov operators are mapped (is_public=false) and distinct from unmapped null", () => {
  const spacex = resolveOperator("SpaceX");
  assert.ok(spacex, "SpaceX is a known operator (mapped)");
  assert.equal(spacex!.ticker, null, "…but has no public equity");
  assert.equal(spacex!.is_public, false);

  const gov = resolveOperator("NASA");
  assert.ok(gov);
  assert.equal(gov!.category, "gov");
  assert.equal(gov!.ticker, null);

  // vs a genuinely unmapped operator -> the resolve itself is null
  assert.equal(resolveOperator("KEPLER COMMUNICATIONS INC"), null);
});

// ---------------------------------------------------------------------------
// footprint grouping
// ---------------------------------------------------------------------------
test("orbitalFootprintByTicker groups a mixed satellite set by owning ticker", () => {
  const sats: JoinedSat[] = [
    { norad: 1, operator: "Iridium" },                       // -> IRDM
    { norad: 2, operator: "IRIDIUM SATELLITE LLC" },         // -> IRDM (alias)
    { norad: 3, operator: "Planet Labs Inc" },               // -> PL
    { norad: 4, operator: "SPACE EXPLORATION TECHNOLOGIES" },// -> null (private)
    { norad: 5, operator: "Totally Unknown Operator" },      // -> null (unmapped)
    { norad: 6, ticker: "irdm" },                            // explicit lowercase ticker -> IRDM
  ];

  const irdm = orbitalFootprintByTicker(sats, "IRDM");
  assert.deepEqual(irdm.map((s) => s.norad).sort(), [1, 2, 6]);

  const pl = orbitalFootprintByTicker(sats, "pl"); // case-insensitive query
  assert.deepEqual(pl.map((s) => s.norad), [3]);

  // private + unmapped both excluded — never mis-attributed
  assert.equal(orbitalFootprintByTicker(sats, "SPACEX").length, 0);
  assert.equal(orbitalFootprintByTicker(sats, "").length, 0);
});

test("groupOrbitalFootprint buckets everything, private+unmapped under null (no silent drops)", () => {
  const sats: JoinedSat[] = [
    { operator: "Iridium" },
    { operator: "Iridium Communications" },
    { operator: "Planet" },
    { operator: "SpaceX" },          // private -> null bucket
    { operator: "Nonexistent Corp" },// unmapped -> null bucket
  ];
  const grouped = groupOrbitalFootprint(sats);
  assert.equal(grouped.get("IRDM")!.length, 2);
  assert.equal(grouped.get("PL")!.length, 1);
  assert.equal(grouped.get(null)!.length, 2);
  // every input accounted for
  const total = Array.from(grouped.values()).reduce((n, arr) => n + arr.length, 0);
  assert.equal(total, sats.length);
});

// ---------------------------------------------------------------------------
// DATA INTEGRITY — the seed's own honesty invariants
// ---------------------------------------------------------------------------
test("operators.json coverage counts match the actual entries (no drift)", () => {
  const ops = operatorsFile.operators;
  assert.equal(ops.length, operatorsFile.coverage.operators);
  const withTicker = ops.filter((o) => o.ticker != null);
  const nullTicker = ops.filter((o) => o.ticker == null);
  assert.equal(withTicker.length, operatorsFile.coverage.with_public_ticker);
  assert.equal(nullTicker.length, operatorsFile.coverage.null_ticker);
  const priv = nullTicker.filter((o) => o.category !== "gov");
  const gov = nullTicker.filter((o) => o.category === "gov");
  assert.equal(priv.length, operatorsFile.coverage.null_breakdown.private);
  assert.equal(gov.length, operatorsFile.coverage.null_breakdown.government);
});

test("every seeded ticker is a clean uppercase symbol; gov/private are null (no guessed symbols)", () => {
  const validCat = new Set(operatorsFile.category_values);
  for (const o of OPERATORS) {
    assert.ok(o.company && o.operator_name_variants.length > 0, `entry needs company + variants: ${o.company}`);
    assert.ok(validCat.has(o.category), `bad category on ${o.company}: ${o.category}`);
    if (o.ticker != null) {
      assert.match(o.ticker, /^[A-Z][A-Z.]{0,7}$/, `ticker looks malformed: ${o.company} -> ${o.ticker}`);
      assert.ok(o.exchange, `a public ticker must carry an exchange: ${o.company}`);
    } else {
      assert.equal(o.exchange, null, `null ticker must have null exchange: ${o.company}`);
    }
    // government owners never carry a ticker (honesty by construction)
    if (o.category === "gov") assert.equal(o.ticker, null, `gov owner must be ticker-null: ${o.company}`);
  }
});
