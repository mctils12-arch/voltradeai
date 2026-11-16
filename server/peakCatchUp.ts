/**
 * peakCatchUp.ts — the ONE-TIME catch-up that aligns the Python engine's
 * drawdown peak with Node's re-baselined peak (2026-11-11, KNOWN BROKEN #49;
 * the human's pick of research/wishlist.md option (a)).
 *
 * WHY: the human-directed paper re-baseline (2026-09-28) lowered Node's
 * equityPeak ($110.7K -> $99,962) and bot_engine's DD-halt peak, but the
 * SHARED Python peak (voltrade_peak_equity.json) had only max-ratchet writers,
 * so risk_kill_switch's -20% kill kept measuring from the old high
 * ($111,736.69). It latched on 2026-11-06 15:32Z at equity $89,370 — a -10.6%
 * drawdown from the peak Node and the human's 2026-11-05 -20% decision use —
 * and has refused every new entry since. paper_resume_sync now lowers the
 * shared peak on every future re-baseline; this module runs the catch-up for
 * the 09-28 one, whose event already fired: once per volume (a marker file),
 * paper only, never while Node's own kill is latched (the auto-resume path
 * owns re-baselines then).
 *
 * WHAT RUNS: paper_resume_sync.catch_up_shared_peak(target_peak = Node's
 * equityPeak, equity) through the daemon (python3 -c fallback). It writes the
 * shared peak DOWN through tiered_strategy.rebaseline_peak_equity and clears a
 * "Portfolio DD" kill through risk_kill_switch's own reset_kill_state() — never
 * a manual kill. risk_kill_switch.py is not edited: its thresholds and its
 * mechanism stay; only the peak it is handed changes.
 *
 * OFF SWITCH: VOLTRADE_PEAK_CATCHUP=0 (present env var wins; absent = on).
 */
import fs from "fs";
import path from "path";

export const PEAK_CATCHUP_MARKER = "voltrade_shared_peak_catchup.json";
/** first attempt after boot (the daemon is up and Tier 1 has read the account) */
export const PEAK_CATCHUP_DELAY_MS = 3 * 60_000;
export const PEAK_CATCHUP_RETRY_MS = 30 * 60_000;
export const PEAK_CATCHUP_MAX_ATTEMPTS = 4;

export function peakCatchUpFlag(env: Record<string, string | undefined>): { value: string; source: "env" | "default"; on: boolean } {
  const raw = env.VOLTRADE_PEAK_CATCHUP;
  const value = raw === undefined ? "1" : String(raw);
  const v = value.trim().toLowerCase();
  return { value, source: raw === undefined ? "default" : "env", on: v === "1" || v === "true" || v === "on" || v === "yes" };
}

export function peakCatchUpMarkerPath(env: Record<string, string | undefined> = process.env): string {
  return path.join(env.DATA_DIR || (fs.existsSync("/data") ? "/data/voltrade" : "/tmp"), PEAK_CATCHUP_MARKER);
}

export function readPeakCatchUpMarker(fp: string): Record<string, unknown> | null {
  try {
    const v = JSON.parse(fs.readFileSync(fp, "utf8"));
    return v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : null;
  } catch (e) {
    if ((e as NodeJS.ErrnoException)?.code !== "ENOENT") console.error("[peak-catch-up] unreadable marker", fp, e instanceof Error ? e.message : e);
    return null;
  }
}

/** Atomic (tmp + rename): a crash never leaves a half marker. */
export function writePeakCatchUpMarker(fp: string, body: Record<string, unknown>): void {
  fs.mkdirSync(path.dirname(fp), { recursive: true });
  const tmp = `${fp}.${process.pid}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(body) + "\n");
  fs.renameSync(tmp, fp);
}

export interface PeakCatchUpInput {
  enabled: boolean;
  paper: boolean;
  marker: boolean;
  nodeKillSwitch: boolean;
  nodePeak: number;
  equity: number | null;
}

/** Pure: whether this boot runs the catch-up. `retry` false = nothing to
 *  retry later (done, switched off, or not applicable). */
export function decidePeakCatchUp(s: PeakCatchUpInput): { run: boolean; retry: boolean; reason: string } {
  if (!s.enabled) return { run: false, retry: false, reason: "VOLTRADE_PEAK_CATCHUP off" };
  if (s.marker) return { run: false, retry: false, reason: "already done (marker present)" };
  if (!s.paper) return { run: false, retry: false, reason: "not the paper account: the catch-up is paper-only" };
  if (s.nodeKillSwitch) return { run: false, retry: false, reason: "Node's own kill switch is latched: the auto-resume path owns re-baselines" };
  if (!(Number.isFinite(s.nodePeak) && s.nodePeak > 0)) return { run: false, retry: true, reason: "Node's equityPeak is not loaded yet" };
  if (s.equity === null || !(Number.isFinite(s.equity) && s.equity > 0)) return { run: false, retry: true, reason: "no account equity read" };
  if (s.equity > s.nodePeak) return { run: false, retry: false, reason: `equity ${s.equity.toFixed(2)} is above Node's peak ${s.nodePeak.toFixed(2)}: nothing to align against` };
  return { run: true, retry: true, reason: `align the shared Python peak with Node's ${s.nodePeak.toFixed(2)}` };
}

const money = (x: number): string => {
  if (!Number.isFinite(x) || x <= 0 || x > 1e12) throw new Error(`not a credible amount: ${x}`);
  return x.toFixed(2);
};

/** python3 -c fallback for the daemon's paper_resume_catch_up route. */
export function peakCatchUpCmd(targetPeak: number, equity: number): string {
  return `python3 -c "import json; from paper_resume_sync import catch_up_shared_peak; print(json.dumps(catch_up_shared_peak(target_peak=${money(targetPeak)}, equity=${money(equity)})))"`;
}

const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const obj = (v: unknown): Record<string, unknown> => (v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : {});

/** One audit line from catch_up_shared_peak's report; ok only when the
 *  report says the peak settled and no Portfolio-DD kill is left. */
export function summarizePeakCatchUp(success: boolean, result: unknown, via: string): { ok: boolean; line: string } {
  if (!success) return { ok: false, line: `python call failed (via ${via}): ${JSON.stringify(result).slice(0, 200)}` };
  const r = obj(result);
  const sp = obj(r.shared_peak);
  const ks = obj(r.risk_kill_switch);
  const lv = obj(r.levels);
  const notes = Array.isArray(r.notes) ? r.notes.filter((n) => typeof n === "string").join("; ") : "";
  const before = num(sp.before), after = num(sp.after), kill = num(lv.kill_equity), dd = num(lv.drawdown_pct);
  const ok = r.ok === true;
  const line =
    `shared Python peak ${before?.toFixed(2) ?? "?"} -> ${after?.toFixed(2) ?? "?"}` +
    ` (${sp.changed === true ? "written down" : "unchanged"}); engine kill ${ks.killed === true ? (ks.reset === true ? "was latched, RESET" : "latched, NOT reset") : "not latched"}` +
    `${ks.kill_reason ? ` (${String(ks.kill_reason)})` : ""}; -20% now trips at ${kill?.toFixed(2) ?? "?"}, drawdown ${dd?.toFixed(2) ?? "?"}%` +
    ` [via ${via}]${notes ? ` — ${notes}` : ""}`;
  return { ok, line: ok ? line : `NOT DONE: ${line}` };
}
