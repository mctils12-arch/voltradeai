// signalProducts.test.ts — the Signals PRODUCT catalog (#/data/validated-signals).
// Hermetic (fixture roots) plus two REQUIRED pins against the REAL ladder:
// every validated root is wired, and every wiring row points at a real
// validated root with a real license mark. Runs via `npm run test:node`.
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  SIGNAL_PRODUCTS, buildSignalProducts, gnssReading, wikiReading,
  GATE_DEFINITIONS, IN_VALIDATION_NOTE_CHARS,
} from "./signalProducts";
import { loadSignalLadder, isValidatedSignal, type LadderRootView } from "./signalLadder";
import { LICENSE_MARKS } from "./apiProduct";
import type { GnssIntegritySignalSummary } from "./gnssIntegritySignal";
import type { WikiAttentionSignalSummary } from "./wikiAttentionSignal";

function root(overrides: Partial<LadderRootView>): LadderRootView {
  return {
    id: "x", name: "X", category: "other", status: "raw_only",
    current_gate: 0, last_update_date: "2026-07-01", note: "n", source_ref: "s",
    ...overrides,
  };
}

const GNSS: GnssIntegritySignalSummary = {
  kind: "signal", root_id: "gnss_integrity_adsb", generated_at: "2026-11-10T20:00:00.000Z",
  gate: { current_gate: 2, status: "gate2_pass" }, verdict: "PASS",
  bands: [
    { band: "cruise", candidate_k: 15, candidate_n: 414, control_rate: 0.00147, expected_under_null: 0.61, p_value: 1e-6, elevated: true, expected_to_elevate: true },
    { band: "mid", candidate_k: 12, candidate_n: 277, control_rate: 0.00272, expected_under_null: 0.75, p_value: 1e-6, elevated: true, expected_to_elevate: true },
    { band: "low", candidate_k: 7, candidate_n: 903, control_rate: 0.02214, expected_under_null: 20, p_value: 0.46, elevated: false, expected_to_elevate: false },
    { band: "ground", candidate_k: 0, candidate_n: 19, control_rate: 0.01, expected_under_null: 0.2, p_value: 1, elevated: false, expected_to_elevate: false },
  ],
  region: { candidate_bbox: [53, 60, 17, 24], candidate_label: "c", control_bbox: [35, 55, -80, 10], control_label: "k" },
  freshness: {
    writer_live_since: "2026-08-11",
    candidate: { days_read: ["2026-11-01", "2026-11-02", "2026-11-03"], days_missing: [], rows_scanned: 10, truncated: false, days_from_daily_archive: [] },
    control: { days_read: ["2026-11-01", "2026-11-02", "2026-11-03"], days_missing: [], rows_scanned: 10, truncated: false, days_from_daily_archive: [] },
  },
  methodology_note: "m", caveats: ["c"], license: { source: "s", note: "n" },
};

const WIKI: WikiAttentionSignalSummary = {
  kind: "signal", root_id: "wikimedia_pageviews_attention", generated_at: "2026-11-10T21:00:00.000Z",
  gate: { current_gate: 2, status: "gate2_pass", channel: "trading_volume_elevation" },
  z_threshold: 2, trailing_window_days: 90, min_baseline_days: 20,
  tickers: [
    { ticker: "GME", article: "GameStop", cap_tier: "small_mid", latest_date: "2026-11-10", current_views: 1, baseline_mean: 1, baseline_days: 60, baseline_complete: false, z_score: 3.1, spike: true },
    { ticker: "PLTR", article: "Palantir", cap_tier: "small_mid", latest_date: "2026-11-10", current_views: 1, baseline_mean: 1, baseline_days: 60, baseline_complete: false, z_score: 0.2, spike: false },
    { ticker: "ACHR", article: "Archer_Aviation", cap_tier: "small_mid", latest_date: null, current_views: null, baseline_mean: null, baseline_days: 0, baseline_complete: false, z_score: null, spike: false },
  ],
  spike_count: 1,
  validated_effect: { study_date: "2026-09-04", bonferroni_alpha: 0.005, small_mid: [], mega: [] },
  methodology_note: "m", caveats: ["c"], license: { source: "s", note: "n" },
};

test("gnssReading: warming cache -> warming_up; summary -> verdict + elevated/expected bands + day count", () => {
  assert.deepEqual(gnssReading(null), { state: "warming_up", generated_at: null, headline: "first archive scan in progress" });
  const r = gnssReading(GNSS);
  assert.equal(r.state, "live");
  assert.equal(r.generated_at, "2026-11-10T20:00:00.000Z");
  assert.equal(r.headline, "PASS — 2/2 expected bands elevated (cruise, mid) over 3 archive days");
});

test("wikiReading: null -> unavailable (never fabricated); summary -> spikes of scored tickers + unscored count", () => {
  assert.equal(wikiReading(null).state, "unavailable");
  const r = wikiReading(WIKI);
  assert.equal(r.state, "live");
  assert.equal(r.headline, "1 of 2 scored tickers spiking (z ≥ 2) · 1 without enough baseline");
  const allScored = wikiReading({ ...WIKI, tickers: WIKI.tickers.slice(0, 2) });
  assert.equal(allScored.headline, "1 of 2 scored tickers spiking (z ≥ 2)");
});

test("buildSignalProducts: only gate>=2 passes become products; gate2_pending -> in_validation; others dropped", () => {
  const roots: LadderRootView[] = [
    root({ id: "gnss_integrity_adsb", name: "GNSS", status: "gate2_pass", current_gate: 2, last_update_date: "2026-09-26" }),
    root({ id: "wikimedia_pageviews_attention", name: "Wiki", status: "gate2_pass", current_gate: 2, last_update_date: "2026-09-28" }),
    root({ id: "pend", name: "Pending", status: "gate2_pending", current_gate: 2, last_update_date: "2026-10-02", note: "short note" }),
    root({ id: "data_only", status: "gate1_pass", current_gate: 1 }),
    root({ id: "dead", status: "killed", current_gate: 2 }),
    root({ id: "raw", status: "raw_only", current_gate: 0 }),
    root({ id: "failed2", status: "gate2_fail", current_gate: 2 }),
  ];
  const p = buildSignalProducts(roots, "2026-11-10", {
    gnss_integrity_adsb: gnssReading(GNSS),
    wikimedia_pageviews_attention: wikiReading(WIKI),
  }, Date.parse("2026-11-11T00:00:00Z"));
  assert.equal(p.generated_at, "2026-11-11T00:00:00.000Z");
  assert.equal(p.ladder_compiled, "2026-11-10");
  assert.deepEqual(p.products.map((x) => x.root_id), ["wikimedia_pageviews_attention", "gnss_integrity_adsb"], "same gate -> newer ladder update first");
  assert.deepEqual(p.unwired, []);
  assert.deepEqual(p.in_validation.map((x) => x.id), ["pend"]);
  for (const row of p.products) {
    assert.equal(row.tradeable, false);
    assert.equal(row.live.state, "live");
    assert.ok(["ok", "share-alike", "conditional"].includes(row.license.resell), "license resolved from LICENSE_MARKS");
    assert.ok(row.license.attribution.length > 0);
    assert.match(row.v1_path, /^\/api\/v1\/data\//);
    assert.match(row.data_route, /^\/api\/data\//);
    assert.match(row.detail_route, /^#\/data\//);
  }
  assert.equal(p.gate_definitions, GATE_DEFINITIONS);
  assert.match(p.honesty, /gate 3/);
});

test("buildSignalProducts: a validated root without wiring is listed in `unwired`, never silently dropped; missing live -> unavailable", () => {
  const roots: LadderRootView[] = [
    root({ id: "brand_new_signal", status: "gate2_pass", current_gate: 2 }),
    root({ id: "gnss_integrity_adsb", status: "gate2_pass", current_gate: 2 }),
  ];
  const p = buildSignalProducts(roots, "c", {});
  assert.deepEqual(p.unwired, ["brand_new_signal"]);
  assert.equal(p.products.length, 1);
  assert.equal(p.products[0].live.state, "unavailable");
  assert.equal(p.products[0].live.generated_at, null);
});

test("buildSignalProducts: a gate3+ pass is still a product (not filtered to gate 2 exactly)", () => {
  const p = buildSignalProducts([root({ id: "gnss_integrity_adsb", status: "gate3_pass", current_gate: 3 })], "c", {});
  assert.equal(p.products.length, 1);
  assert.equal(p.products[0].current_gate, 3);
});

test("in_validation notes are truncated at a word boundary with an ellipsis", () => {
  const long = Array.from({ length: 80 }, (_, i) => `word${i}`).join(" ");
  const p = buildSignalProducts([root({ id: "pend", status: "gate2_pending", current_gate: 2, note: long })], "c", {});
  const note = p.in_validation[0].note;
  assert.ok(note.length <= IN_VALIDATION_NOTE_CHARS + 1, `truncated (${note.length})`);
  assert.ok(note.endsWith("…"));
  assert.doesNotMatch(note, /word\d+word/, "cut on a word boundary");
});

// ── REQUIRED pins against the REAL ladder + REAL license marks ─────────────
test("REQUIRED: every validated root in datacore/signal_ladder.json has a product wiring row (unwired == [])", () => {
  const ladder = loadSignalLadder();
  const p = buildSignalProducts(ladder.roots, ladder.compiled, {});
  assert.deepEqual(p.unwired, [], "a root passed gate 2 without its Signals product surface — add a SIGNAL_PRODUCTS row in server/signalProducts.ts (and its page) before merging");
  assert.ok(p.products.length >= 2, "the two 2026-09 gate-2 passes are products");
});

test("REQUIRED: every SIGNAL_PRODUCTS row names a real validated root, a real LICENSE_MARKS key, and the ladder's own detail_route", () => {
  const ladder = loadSignalLadder();
  const byId = new Map(ladder.roots.map((r) => [r.id, r]));
  const ids = new Set<string>();
  for (const s of SIGNAL_PRODUCTS) {
    assert.ok(!ids.has(s.root_id), `duplicate wiring for ${s.root_id}`);
    ids.add(s.root_id);
    const r = byId.get(s.root_id);
    assert.ok(r, `${s.root_id} is not in the ladder`);
    assert.ok(isValidatedSignal(r!.status, r!.current_gate), `${s.root_id} is wired as a product but the ladder says ${r!.status}`);
    assert.ok(LICENSE_MARKS[s.license_key], `${s.root_id}: LICENSE_MARKS["${s.license_key}"] missing`);
    if (r!.detail_route) assert.equal(s.detail_route, r!.detail_route, `${s.root_id}: detail_route drifted from the ladder`);
    for (const k of ["measures", "validated_channel", "cadence", "product"] as const) {
      assert.ok(s[k].trim().length > 0, `${s.root_id}.${k} empty`);
    }
  }
});
