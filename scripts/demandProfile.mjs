// DEVICE ENVELOPE D1 — demand-ledger pure helpers (scale_program.md,
// human directive 2026-08-07: "see for the entire system what the demands
// are"). The harness's --demand mode measures each layer's cost in
// isolation at multiple emulated tiers; these helpers turn raw samples
// into ledger cells and rank them. Pure module so node:test can pin the
// contract without a browser.

/** One measured sample window. All fields optional-null: a metric the
 *  browser can't provide (e.g. performance.memory outside Chrome) is
 *  recorded null, never fabricated as 0. */
export function demandCell(layerId, tierKey, baseline, sample, extra = {}) {
  const d = (a, b) => (a == null || b == null ? null : Math.round((b - a) * 10) / 10);
  return {
    layer: layerId,
    tier: tierKey,
    // frame cost is ABSOLUTE for the window (medians don't subtract
    // meaningfully — baseline median is reported alongside instead)
    medianMs: sample.medianMs ?? null,
    p95Ms: sample.p95Ms ?? null,
    baselineMedianMs: baseline.medianMs ?? null,
    baselineP95Ms: baseline.p95Ms ?? null,
    heapDeltaMB: d(baseline.heapMB, sample.heapMB),
    netDeltaKB: d(baseline.netKB, sample.netKB),
    longTasks: sample.longTasks ?? null,
    renderedFeatures: sample.renderedFeatures ?? null,
    ...extra,
  };
}

/** Rank: most demanding first. Frame cost over baseline dominates; heap
 *  breaks ties. Null-metric cells sink (unmeasured ≠ cheap). */
export function rankLedger(cells) {
  const score = (c) => {
    if (c.medianMs == null || c.baselineMedianMs == null) return -Infinity;
    return (c.medianMs - c.baselineMedianMs) + (c.heapDeltaMB ?? 0) * 0.5;
  };
  return [...cells].sort((a, b) => score(b) - score(a));
}

/** The ledger file shape — version-stamped so profiles diff across
 *  releases. `caps` records every bound the run applied (layer list,
 *  sample seconds, tiers) so a smaller-than-expected ledger is visibly
 *  bounded, never silently truncated. */
export function buildLedger(version, cells, caps) {
  return {
    schema: "vt-demand-profile/1",
    version,
    cells: rankLedger(cells),
    caps,
    note: "fixture-data render/memory demand at emulated tiers (SwiftShader); " +
      "regression-diff tool, not an on-device budget — DESIGN.md owns budgets. " +
      "null metric = not measurable in this browser, never assumed 0.",
  };
}
