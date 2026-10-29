import { test } from "node:test";
import assert from "node:assert/strict";
import { signedPct, frac, ciText, icText, windowText, groupByKind, filterComponents, headlineArm, type BrainComponent, type SelectionArm } from "./brain";

const comp = (id: string, kind: BrainComponent["kind"], status: BrainComponent["status"], flag: BrainComponent["agreement"]["flag"] = "insufficient"): BrainComponent => ({
  id, name: id, kind, group: "g", does: "does a thing", inventory_ref: null, status, status_source: "config", detail: "",
  short_term: { measured: false, what: "—", source: "s", as_of: null, sign: "insufficient", sign_note: "" },
  long_term: { measured: false, items: [], sign: "insufficient", sign_note: "" }, agreement: { flag, note: "" },
});

test("formatting: signed percents, fractions, CIs, IC; missing values never print a number", () => {
  assert.equal(signedPct(0.364), "+0.36%");
  assert.equal(signedPct(-1.2, 1), "-1.2%");
  assert.equal(signedPct(undefined), "—");
  assert.equal(frac(0.513, 1), "51.3%");
  assert.equal(ciText([-0.9, 0.3]), "95% CI -0.90% … +0.30%");
  assert.equal(ciText(null), "no CI");
  assert.equal(ciText([-0.01, 0.05], ""), "95% CI -0.01 … 0.05");
  assert.equal(icText(-0.0912), "-0.091");
  assert.equal(windowText({ first: "2026-06-01", last: "2026-10-28" }), "2026-06-01 → 2026-10-28");
  assert.equal(windowText(null), "no dated rows");
});

test("grouping and filters: sleeves first; disagree filter; status filter", () => {
  const cs = [comp("a", "input", "shadow"), comp("b", "sleeve", "live", "disagree"), comp("c", "rule", "off")];
  assert.deepEqual(groupByKind(cs).map((g) => g.kind), ["sleeve", "rule", "input"]);
  assert.deepEqual(filterComponents(cs, "disagree").map((c) => c.id), ["b"]);
  assert.deepEqual(filterComponents(cs, "off").map((c) => c.id), ["c"]);
  assert.equal(filterComponents(cs, "all").length, 3);
});

test("headlineArm: a SELECTED arm wins; else the larger arm; none -> null", () => {
  const arm = (a: SelectionArm["arm"], status: string, n: number): SelectionArm => ({ arm: a, label: a, status, reason: null, n_rows: n, candidates: 1, variants_tried: 1, selected: [], weights: null, steps: [], holdout_selected: null, holdout_live_score: null, holdout_vs_live_score: null, holdout_window: null });
  assert.equal(headlineArm([arm("features", "NOTHING_SELECTED", 900), arm("breakdown", "SELECTED", 300)])?.arm, "breakdown");
  assert.equal(headlineArm([arm("features", "INSUFFICIENT", 100), arm("breakdown", "INSUFFICIENT", 300)])?.arm, "breakdown");
  assert.equal(headlineArm([]), null);
});
