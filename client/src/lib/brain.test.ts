import { test } from "node:test";
import assert from "node:assert/strict";
import { signedPct, frac, ciText, icText, windowText, groupByKind, filterComponents, headlineArm, brainKpis, type BrainComponent, type SelectionArm, type BrainReport, type BrainStatus, type AgreementFlag } from "./brain";

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

test("brainKpis: six tiles — the five filters with the real counts, then the evidence age", () => {
  const comp = (status: BrainStatus, flag: AgreementFlag): BrainComponent => ({ id: status + flag, name: "x", kind: "input", group: "g", does: "", inventory_ref: null, status, status_source: "config", detail: "",
    short_term: { measured: false, what: "—", source: "", as_of: null, sign: "insufficient", sign_note: "" }, long_term: { measured: false, items: [], sign: "insufficient", sign_note: "" }, agreement: { flag, note: "" } });
  const rep: BrainReport = { generated_at: "2026-10-29T12:00:00Z", evidence_generated_at: "2026-10-29T11:56:00Z", evidence_via: "daemon", evidence_error: null, explainer: "", rule: "",
    status_counts: { live: 1, shadow: 2, off: 1 }, switches: { config_source: "python", entry_shadow: { stocks: true, options: true, value: "stocks,options", source: "config" }, regime: {} },
    short_term: { window: { first: null, last: null }, records: 19874, spy: null, primary_outcome: "", note: "" }, long_term: { sources: [], gaps: [] },
    components: [comp("live", "agree"), comp("shadow", "disagree"), comp("shadow", "insufficient"), comp("off", "insufficient")],
    combinations: { short_term: { available: false, reason: null, as_of: null, source: "", family: null, summary: null, arms: [], ablation: null }, long_term: [] } };
  const k = brainKpis(rep, Date.parse("2026-10-29T12:00:00Z"));
  assert.deepEqual(k.map((t) => t.key), ["all", "live", "shadow", "off", "disagree", "evidence"]);
  assert.deepEqual(k.map((t) => t.value), ["4", "1", "2", "1", "1", "4m"]);
  assert.equal(k[4].tone, "r");
  assert.equal(k[5].line, "via daemon · 19,874 shadow records");
  const noEv = brainKpis({ ...rep, evidence_generated_at: null, components: rep.components.filter((c) => c.agreement.flag !== "disagree") }, 0);
  assert.equal(noEv[5].value, "—"); assert.equal(noEv[4].value, "0"); assert.equal(noEv[4].tone, undefined);
});
