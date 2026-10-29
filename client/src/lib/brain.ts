// Brain page (2026-10-29) — the shape of GET /api/bot/brain (server/brain.ts
// brainBody) and the page's pure helpers. Kept out of the page so they are tested.

export type BrainKind = "input" | "rule" | "model" | "sleeve";
export type BrainStatus = "live" | "shadow" | "off";
export type Sign = "positive" | "negative" | "null" | "insufficient";
export type AgreementFlag = "agree" | "disagree" | "insufficient";
type CI = [number, number] | null | undefined;

export interface ShortTerm {
  measured: boolean; reason?: string; what: string; metric?: "mean_return" | "ic" | "forecast";
  n?: number; n_days?: number; mean_pct?: number; hit_rate?: number; ci95?: CI;
  vs_spy?: { n: number; mean_excess_pct?: number | null; ci95?: CI; beat_rate?: number | null } | null;
  ic?: number; q_bh?: number; verdict?: string; verdict_reason?: string; bucket?: string; outcome?: string;
  horizons?: Record<string, { n: number; mean_pct?: number; hit_rate?: number; ci95?: CI; mean_excess_pct?: number | null }>;
  window?: { first: string | null; last: string | null }; source: string; as_of: string | null; sign: Sign; sign_note: string;
}
export interface LongItem {
  id: string; title: string; source_file: string; date: string; horizon: string; finding: string; compared_to: string;
  sign: Sign; variants_tried?: number; live?: boolean;
}
export interface LongTerm { measured: boolean; reason?: string; items: LongItem[]; sign: Sign; sign_note: string }
export interface BrainComponent {
  id: string; name: string; kind: BrainKind; group: string; does: string; inventory_ref: string | null; ref_file?: string;
  status: BrainStatus; status_source: "config" | "env" | "code"; detail: string;
  short_term: ShortTerm; long_term: LongTerm; agreement: { flag: AgreementFlag; note: string };
}
export interface SelectionStep { candidate: string; select_ic: number; gain: number; se: number | null; p_bonferroni: number; tried: number; accepted: boolean }
export interface IcLite { n?: number; n_days?: number; ic?: number; ci95?: CI }
export interface SelectionArm {
  arm: "features" | "breakdown"; label: string; status: string; reason: string | null; n_rows: number | null; candidates: number | null;
  variants_tried: number; selected: string[]; weights: Record<string, number> | null; steps: SelectionStep[];
  holdout_selected: IcLite | null; holdout_live_score: IcLite | null; holdout_vs_live_score: { ic_gain: number; se: number | null } | null;
  holdout_window: { first: string | null; last: string | null } | null;
}
export interface AblationRow { name: string; removed: string[]; n: number | null; ic: number | null; ci95: CI; ic_change_vs_baseline: number | null; ic_change_se: number | null; q5_minus_q1_pct: number | null; status: string | null }
export interface LongCombination { id: string; title: string; source_file: string; date: string; horizon: string; variants_tried: number; discount: string; columns: string[]; rows: string[][] }
export interface BrainReport {
  generated_at: string; evidence_generated_at: string | null; evidence_via: string | null; evidence_error: string | null; evidence_cache_age_s?: number | null;
  explainer: string; rule: string; status_counts: Record<BrainStatus, number>;
  switches: { config_source: "python" | "mirror"; entry_shadow: { stocks: boolean; options: boolean; value: string; source: string }; regime: Record<string, { value: string; source: string }> };
  short_term: { window: { first: string | null; last: string | null }; records: number | null; spy: { ok: boolean; error?: string | null; first_day?: string | null; last_day?: string | null; method?: string } | null; primary_outcome: string; note: string };
  long_term: { sources: Array<{ file: string; date: string }>; gaps: string[] };
  components: BrainComponent[];
  combinations: {
    short_term: { available: boolean; reason: string | null; as_of: string | null; source: string; family: string | null; summary: Record<string, string[]> | null; arms: SelectionArm[];
      ablation: { status: string | null; reason: string | null; n_rows: number | null; baseline_inverted: boolean | null; un_inverts: boolean | null; improvement_significant: boolean | null; rows: AblationRow[] } | null };
    long_term: LongCombination[];
  };
}

export const BRAIN_KIND_LABEL: Record<BrainKind, string> = { input: "Inputs", rule: "Rules", model: "Models", sleeve: "Sleeves" };
export const KIND_ORDER: BrainKind[] = ["sleeve", "model", "rule", "input"];
export const BRAIN_STATUS_LABEL: Record<BrainStatus, string> = { live: "Live", shadow: "Shadow", off: "Off" };
export const STATUS_HINT: Record<BrainStatus, string> = {
  live: "reaches a real order today",
  shadow: "computed and logged; nothing it touches is ordered",
  off: "not computed, or computed and never read",
};
export const AGREE_LABEL: Record<AgreementFlag, string> = { agree: "Agree", disagree: "Disagree", insufficient: "Not enough data" };
export const SIGN_LABEL: Record<Sign, string> = { positive: "helps", negative: "hurts", null: "no effect", insufficient: "no verdict" };

const num = (x: unknown): x is number => typeof x === "number" && Number.isFinite(x);

/** Signed percent with 2 decimals: "+0.36%". Values are already in percent units. */
export function signedPct(x: number | null | undefined, digits = 2): string {
  if (!num(x)) return "—";
  return `${x > 0 ? "+" : ""}${x.toFixed(digits)}%`;
}
/** A 0..1 fraction as a percent. */
export function frac(x: number | null | undefined, digits = 0): string {
  return num(x) ? `${(x * 100).toFixed(digits)}%` : "—";
}
export function ciText(ci: CI, unit: "%" | "" = "%", digits = 2): string {
  if (!ci || !num(ci[0]) || !num(ci[1])) return "no CI";
  const f = (v: number) => (unit === "%" ? signedPct(v, digits) : v.toFixed(digits));
  return `95% CI ${f(ci[0])} … ${f(ci[1])}`;
}
export function icText(ic: number | null | undefined): string {
  return num(ic) ? `${ic > 0 ? "+" : ""}${ic.toFixed(3)}` : "—";
}
export function windowText(w: { first: string | null; last: string | null } | null | undefined): string {
  if (!w || (!w.first && !w.last)) return "no dated rows";
  return w.first === w.last ? String(w.first) : `${w.first ?? "?"} → ${w.last ?? "?"}`;
}

/** Group components by kind in page order, keeping registry order inside a group. */
export function groupByKind(cs: BrainComponent[]): Array<{ kind: BrainKind; items: BrainComponent[] }> {
  return KIND_ORDER.map((kind) => ({ kind, items: cs.filter((c) => c.kind === kind) })).filter((g) => g.items.length > 0);
}

export type StatusFilter = "all" | BrainStatus | "disagree";
export function filterComponents(cs: BrainComponent[], f: StatusFilter): BrainComponent[] {
  if (f === "all") return cs;
  if (f === "disagree") return cs.filter((c) => c.agreement.flag === "disagree");
  return cs.filter((c) => c.status === f);
}

/** The arm worth headlining: SELECTED beats anything else; else the arm with more rows. */
export function headlineArm(arms: SelectionArm[]): SelectionArm | null {
  if (!arms.length) return null;
  return arms.find((a) => a.status === "SELECTED") ?? [...arms].sort((a, b) => (b.n_rows ?? 0) - (a.n_rows ?? 0))[0];
}
