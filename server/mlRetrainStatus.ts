// REPAIR 2026-10-16 (bot cockpit audit, S4 "retrain trigger" row): Tier 3
// audited every retrain whose status did not start with "error:"/"failed"
// as `TIER3 "ML retrain complete — status: X"`. ml_model_v2._train_model_impl
// also returns {"status": "insufficient_data"} and {"status": "skipped"},
// ml_retrain_safe.py returns {"status": "error"} (no colon) and
// {"status": "fatal"} — none of which produced a model, all of which were
// audited as complete, so "retraining silently stopped firing" was
// undetectable from the audit log. The only status that means a model was
// written is "trained" (ml_model_v2.py: every success return carries it).
export const RETRAIN_SUCCESS_STATUSES: ReadonlySet<string> = new Set(["trained"]);

export function retrainStatusIsSuccess(status: unknown): boolean {
  return RETRAIN_SUCCESS_STATUSES.has(String(status ?? "").trim());
}

/** Extra detail for a non-success retrain result, in the order a reader
 *  needs it: the wrapper's `error`, then `samples` (insufficient_data's
 *  only payload), then which steps completed before it died. */
export interface RetrainResultLike { status?: unknown; error?: unknown; samples?: unknown; steps_completed?: unknown }

export function retrainFailureDetail(result: RetrainResultLike | null | undefined): string {
  const parts: string[] = [];
  const r: RetrainResultLike = result && typeof result === "object" ? result : {};
  if (r.error) parts.push(String(r.error).slice(0, 200));
  if (r.samples !== undefined && r.samples !== null) parts.push(`samples ${String(r.samples)}`);
  if (Array.isArray(r.steps_completed) && r.steps_completed.length) parts.push(`steps ${r.steps_completed.map(String).join(">")}`);
  return parts.join(" · ");
}

// [RULE-REVIEW] 2026-11-06 (v2 training honesty, ml_model_v2._fit_v2_honest):
// a "trained" result now carries `honest_cv` — the out-of-sample record of
// the dated, purged walk-forward — and `training_set.shadow` (labeled scan
// rows vs first-seen candidate-days kept). The TIER3 line used to print the
// accuracy alone, which reads as skill whenever it sits above 50% even when
// always predicting the majority class scores higher. This renders the bar
// next to each number. "" when the result has no honest_cv (switch off, or a
// result from before the change), so the line is then exactly what it was.
function finiteNum(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function asRecord(v: unknown): Record<string, unknown> | null {
  return v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : null;
}

export function retrainHonestDetail(result: unknown): string {
  const r = asRecord(result);
  const hc = asRecord(r?.honest_cv);
  const pooled = asRecord(hc?.pooled);
  if (!hc || !pooled || pooled.status !== "OK") return "";
  const auc = finiteNum(pooled.auc);
  const acc = finiteNum(pooled.accuracy);
  const maj = finiteNum(pooled.majority_accuracy);
  const ll = finiteNum(pooled.log_loss);
  const bll = finiteNum(pooled.base_rate_log_loss);
  const ci = Array.isArray(hc.auc_ci95_date_block) ? hc.auc_ci95_date_block.map(finiteNum) : [];
  const parts: string[] = [];
  const lo = ci.length === 2 ? ci[0] : null;
  const hi = ci.length === 2 ? ci[1] : null;
  if (auc !== null) {
    const ciTxt = lo !== null && hi !== null ? `95% CI ${lo.toFixed(3)}–${hi.toFixed(3)}; ` : "";
    parts.push(`out-of-sample AUC ${auc.toFixed(3)} (${ciTxt}0.5 = no skill)`);
  }
  if (acc !== null && maj !== null) parts.push(`pooled accuracy ${(acc * 100).toFixed(1)}% vs ${(maj * 100).toFixed(1)}% always-majority`);
  if (ll !== null && bll !== null) parts.push(`log-loss ${ll.toFixed(3)} vs ${bll.toFixed(3)} always-base-rate`);
  let out = parts.join(", ");
  if (out) out += hc.skill === true ? " — skill" : " — no skill";
  const sh = asRecord(asRecord(r?.training_set)?.shadow);
  const labeled = finiteNum(sh?.rows_labeled);
  const used = finiteNum(sh?.rows_used);
  if (labeled !== null && used !== null) out += `${out ? "; " : ""}shadow ${labeled} scan rows → ${used} candidate-days`;
  return out;
}
