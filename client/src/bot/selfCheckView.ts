// Bot page — the bot's own self-check and its alerts ([PRODUCT] 2026-11-20,
// UI coverage batch A, second part). Two owner routes that existed and no page
// read: GET /api/bot/diagnostics (diagnostics.py run_diagnostics + the
// settings get_auto_fix_params derives from it: the size multiplier and the
// score bar the loop applies) and GET /api/bot/notifications (the in-memory
// list the loop writes on a trade, an exit, a halt or an alarm).
//
// Pure; pinned by selfCheckView.test.ts. Read tolerantly like whyView.ts: a
// body of another shape is a designed sentence, never a blank or a guess.
import { fmtAge } from "./format";

const fin = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const str = (v: unknown): string | null => (typeof v === "string" && v.trim() ? v.trim() : null);
const obj = (v: unknown): Record<string, unknown> | null => (v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : null);
const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;

// ── the self-check ──────────────────────────────────────────────────────────
export interface CheckItem { system: string; message: string; severity: "high" | "medium" | "low" | null; fix: string | null }
export interface SelfCheckView {
  ok: boolean; error: string | null;
  /** the report's own word: healthy / degraded / critical …; null when it carries none */
  status: string | null; at: string | null;
  problems: CheckItem[]; warnings: CheckItem[];
  /** housekeeping the check did while it ran (old cache files removed) */
  housekeeping: string[]; recommendations: string[];
  /** what the check's findings set, when the loop applies them: null = the report did not say */
  sizeMultiplier: number | null; scoreBar: number | null; pause: boolean; retrain: boolean;
}
/** what a fix code does, in words; an unknown code is shown with its underscores as spaces */
export const FIX_WORDS: Readonly<Record<string, string>> = {
  reduce_position_size: "new positions are sized smaller",
  force_retrain: "the model is retrained at the next hourly pass",
  force_retrain_and_reduce: "the model is retrained and new positions are sized smaller",
  increase_score_threshold: "only higher-scoring picks are taken",
};
export const SYSTEM_WORDS: Readonly<Record<string, string>> = { cache: "Data freshness", ml_model: "Model", disk: "Disk", api: "Data sources", memory: "Memory" };
const sev = (v: unknown): CheckItem["severity"] => { const s = (str(v) ?? "").toLowerCase(); return s === "high" || s === "medium" || s === "low" ? s : null; };
function items(v: unknown): CheckItem[] {
  if (!Array.isArray(v)) return [];
  const out: CheckItem[] = [];
  for (const x of v) {
    const o = obj(x);
    const message = o ? str(o.message) : str(x);
    if (!message) continue;
    out.push({ system: (o && str(o.system)) ?? "other", message, severity: o ? sev(o.severity) : null, fix: o ? str(o.auto_fix) : null });
  }
  return out;
}
const lines = (v: unknown): string[] => (Array.isArray(v) ? v.map((x) => str(x) ?? str(obj(x)?.message) ?? str(obj(x)?.action)).filter((x): x is string => !!x) : []);
/** GET /api/bot/diagnostics → the view; null when the body is not the report's shape. */
export function parseSelfCheck(raw: unknown): SelfCheckView | null {
  const o = obj(raw);
  if (!o) return null;
  const err = str(o.error);
  const rep = obj(o.report), fix = obj(o.auto_fix);
  if (!rep && !fix) return err ? { ok: false, error: err, status: null, at: null, problems: [], warnings: [], housekeeping: [], recommendations: [], sizeMultiplier: null, scoreBar: null, pause: false, retrain: false } : null;
  return {
    ok: true, error: null, status: str(rep?.overall_status), at: str(rep?.timestamp),
    problems: items(rep?.problems), warnings: items(rep?.warnings), housekeeping: lines(rep?.auto_fixes), recommendations: lines(rep?.recommendations),
    sizeMultiplier: fin(fix?.position_size_multiplier) ? fix.position_size_multiplier : null, scoreBar: fin(fix?.min_score_threshold) ? fix.min_score_threshold : null,
    pause: fix?.should_pause === true, retrain: fix?.force_retrain === true,
  };
}
/** the settings line: what the findings change, or that they change nothing */
export function selfCheckSettings(v: SelfCheckView): string {
  const parts: string[] = [];
  if (v.sizeMultiplier !== null) parts.push(v.sizeMultiplier < 1 ? `new positions at ×${v.sizeMultiplier.toFixed(2)} of normal size` : "normal position size");
  if (v.scoreBar !== null) parts.push(`score bar ${Math.round(v.scoreBar)}`);
  if (v.retrain) parts.push("model retrain asked for");
  if (v.pause) parts.push("it asks the loop to PAUSE");
  return parts.join(" · ");
}
/** The block's one line: the count of problems and warnings and what they set. */
export function selfCheckSentence(v: SelfCheckView): { text: string; tone: "ok" | "warn" | "bad" } {
  if (!v.ok) return { text: `Self-check did not run${v.error ? `: ${v.error.slice(0, 140)}` : ""}`, tone: "warn" };
  const set = selfCheckSettings(v);
  const tail = set ? ` · ${set}` : "";
  if (!v.problems.length && !v.warnings.length) return { text: `Self-check clear${tail}`, tone: "ok" };
  const counts = [v.problems.length ? plural(v.problems.length, "problem") : null, v.warnings.length ? plural(v.warnings.length, "warning") : null].filter(Boolean).join(", ");
  return { text: `Self-check: ${counts}${tail}`, tone: v.pause || v.problems.some((p) => p.severity === "high") ? "bad" : v.problems.length ? "warn" : "ok" };
}
/** "Data freshness — … → new positions are sized smaller" */
export function checkItemWords(i: CheckItem): { lead: string; text: string; fix: string | null } {
  return { lead: SYSTEM_WORDS[i.system] ?? i.system.replace(/_/g, " "), text: i.message, fix: i.fix ? FIX_WORDS[i.fix] ?? i.fix.replace(/_/g, " ") : null };
}
/** When the loop's applied score bar differs from the check's: the loop applies the check on its own cadence. */
export function appliedNote(v: SelfCheckView, appliedScoreBar: number | null): string | null {
  if (v.scoreBar === null || appliedScoreBar === null || Math.round(v.scoreBar) === Math.round(appliedScoreBar)) return null;
  return `The loop is using a score bar of ${Math.round(appliedScoreBar)} until it next applies the check.`;
}

// ── alerts ──────────────────────────────────────────────────────────────────
export type AlertKind = "trade" | "exit" | "alert" | "critical" | "system" | "upgrade" | "other";
export interface AlertView { key: string; time: string; kind: AlertKind; message: string; read: boolean }
const KINDS: readonly AlertKind[] = ["trade", "exit", "alert", "critical", "system", "upgrade"];
export const ALERT_KIND_WORDS: Readonly<Record<AlertKind, string>> = { trade: "trade", exit: "exit", alert: "alert", critical: "critical", system: "system", upgrade: "rotation", other: "note" };
/** GET /api/bot/notifications → newest first; null when the body is not a list. */
export function parseAlerts(raw: unknown): AlertView[] | null {
  if (!Array.isArray(raw)) return null;
  const out: AlertView[] = [];
  raw.forEach((x, i) => {
    const o = obj(x);
    const time = o ? str(o.time) : null, message = o ? str(o.message) : null;
    if (!o || !time || !message || !Number.isFinite(Date.parse(time))) return;
    const k = (str(o.type) ?? "").toLowerCase() as AlertKind;
    out.push({ key: `${time}|${i}`, time, kind: KINDS.includes(k) ? k : "other", message, read: o.read === true });
  });
  return out.sort((a, b) => Date.parse(b.time) - Date.parse(a.time));
}
export const alertTone = (k: AlertKind): "bad" | "warn" | "none" => (k === "critical" ? "bad" : k === "alert" ? "warn" : "none");
/** The block's one line: how many are unread, and the newest one. */
export function alertsSentence(rows: readonly AlertView[], nowMs: number): string {
  if (!rows.length) return "No alerts since the server last restarted.";
  const unread = rows.filter((r) => !r.read).length;
  const newest = rows[0];
  return `${plural(rows.length, "alert")}${unread ? ` · ${unread} new` : ""} · newest ${fmtAge(nowMs - Date.parse(newest.time))} ago: ${newest.message.length > 110 ? `${newest.message.slice(0, 109)}…` : newest.message}`;
}
