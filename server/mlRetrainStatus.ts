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
export function retrainFailureDetail(result: any): string {
  const parts: string[] = [];
  if (result?.error) parts.push(String(result.error).slice(0, 200));
  if (result?.samples !== undefined && result?.samples !== null) parts.push(`samples ${result.samples}`);
  if (Array.isArray(result?.steps_completed) && result.steps_completed.length) parts.push(`steps ${result.steps_completed.join(">")}`);
  return parts.join(" · ");
}
