/**
 * lazyPrices.ts — the weekly job that runs lazy_prices_run.py (ladder gate 2
 * for the Lazy Prices root: the pre-registered 10-K similarity sort over the
 * 2016+ price panel) and the read-only status behind the `lazy_prices` diag
 * probe. Research plumbing: it schedules and reports; nothing it produces is
 * read by any trading path or surfaced on /data.
 *
 * WHEN: Sundays 16:00-18:00 ET (start window) — after edgeP2's 10:00-16:00
 * window closes and the 2h budget ends before the 20:15 panel-build window, so
 * no two panel readers overlap. At most one attempt per ISO week; re-runs only
 * for a new panel build, a new code version, or a result older than 27 days.
 * HOW: same long-job lane as edgeP2 (detached process group, thread caps, nice
 * 19, peak-RSS budget enforced by the Python). Result files are KB-sized under
 * DATA_DIR/lazy_prices; the re-fetchable SEC pairs cache lives in container tmp.
 * Derived statistics only — no ticker, CIK or price (Alpaca data licence).
 * FLAG: VOLTRADE_LAZY_PRICES, default ON; present env var wins.
 */
import fs from "fs";
import os from "os";
import path from "path";
import { beat, zonedParts } from "./jobHeartbeat";
import { execGroupKill } from "./pythonExec";
import { pricePanelRoot } from "./pricePanel";
import {
  WALKFORWARD_ENV_CAPS, currentPanelBuild, isoWeekKey, type RunOutcome, type WalkforwardExec,
} from "./walkforward";
import { version as pkgVersion } from "../package.json";

export const LAZY_PRICES_SUBDIR = "lazy_prices";
export const LAZY_PRICES_TIMEOUT_MS = 2 * 60 * 60_000;
export const LAZY_PRICES_TICK_MS = 30 * 60_000;
export const LAZY_PRICES_MAX_RSS_MB = 1500;
export const LAZY_PRICES_STALE_DAYS = 27;
export const LAZY_PRICES_WINDOW_START_MIN = 16 * 60;
export const LAZY_PRICES_WINDOW_END_MIN = 18 * 60;
export const LAZY_PRICES_MIN_FREE_BYTES = 64_000_000;
const ET = "America/New_York";
const DAY_MS = 86_400_000;

export function lazyPricesFlag(env: Record<string, string | undefined>): { value: string; source: "env" | "default"; on: boolean } {
  const raw = env.VOLTRADE_LAZY_PRICES;
  const value = raw === undefined ? "1" : String(raw);
  const v = value.trim().toLowerCase();
  return { value, source: raw === undefined ? "default" : "env", on: v === "1" || v === "true" || v === "on" || v === "yes" };
}

export function lazyPricesRoot(env: Record<string, string | undefined> = process.env): string {
  return path.join(env.DATA_DIR || (fs.existsSync("/data") ? "/data/voltrade" : "/tmp"), LAZY_PRICES_SUBDIR);
}

/** Container-local (non-volume) SEC pairs cache; env override must be a plain path. */
export function lazyPricesPairsDir(env: Record<string, string | undefined> = process.env): string {
  const o = env.VOLTRADE_LAZY_PRICES_PAIRS_DIR;
  return o && /^[0-9A-Za-z_./-]{1,200}$/.test(o) ? o : path.join(os.tmpdir(), "voltrade_lazy_prices_pairs");
}

export function lazyPricesCmd(codeVersion: string = pkgVersion, env: Record<string, string | undefined> = process.env): string {
  const v = /^[0-9A-Za-z.+-]{1,40}$/.test(codeVersion) ? codeVersion : "unknown";
  const dir = lazyPricesPairsDir(env);
  const d = /^[0-9A-Za-z_./-]{1,200}$/.test(dir) ? dir : "/tmp/voltrade_lazy_prices_pairs";
  return `python3 lazy_prices_run.py --run --nice 19 --max-rss-mb ${LAZY_PRICES_MAX_RSS_MB} --code-version ${v} --pairs-dir ${d}`;
}

/** Sunday, 16:00-18:00 ET. */
export function inLazyPricesWindow(nowMs: number): boolean {
  const p = zonedParts(nowMs, ET);
  return p.weekday === "Sun" && p.minutes >= LAZY_PRICES_WINDOW_START_MIN && p.minutes < LAZY_PRICES_WINDOW_END_MIN;
}

export interface LazyPricesLatest { build_id: string | null; code_version: string | null; generated_at: string | null }

export type LazyPricesDecision =
  | { action: "idle"; reason: string }
  | { action: "skip" | "run"; reason: string; week: string };

/** Pure: what this tick should do. */
export function decideLazyPrices(nowMs: number, s: {
  enabled: boolean; panelBuildId: string | null; latest: LazyPricesLatest | null; lastAttemptWeek: string; codeVersion: string; freeBytes: number | null;
}): LazyPricesDecision {
  if (!s.enabled) return { action: "idle", reason: "VOLTRADE_LAZY_PRICES off" };
  if (!inLazyPricesWindow(nowMs)) return { action: "idle", reason: "outside Sunday 16:00-18:00 ET" };
  const week = isoWeekKey(zonedParts(nowMs, ET).date);
  if (s.lastAttemptWeek === week) return { action: "idle", reason: `already attempted ${week}` };
  if (!s.panelBuildId) return { action: "skip", week, reason: "no complete price-panel build yet (current.json absent)" };
  if (s.freeBytes !== null && s.freeBytes < LAZY_PRICES_MIN_FREE_BYTES) {
    return { action: "skip", week, reason: `only ${Math.floor(s.freeBytes / 1e6)} MB free on the volume (< ${LAZY_PRICES_MIN_FREE_BYTES / 1e6} MB)` };
  }
  const l = s.latest;
  if (l && l.build_id === s.panelBuildId && l.code_version === s.codeVersion && l.generated_at) {
    const age = nowMs - Date.parse(l.generated_at);
    if (Number.isFinite(age) && age < LAZY_PRICES_STALE_DAYS * DAY_MS) {
      return { action: "skip", week, reason: `fresh: build ${l.build_id}, code ${l.code_version}, ${Math.floor(age / DAY_MS)}d old` };
    }
  }
  return { action: "run", week, reason: l ? `build ${l.build_id ?? "?"}/code ${l.code_version ?? "?"} -> ${s.panelBuildId}/${s.codeVersion}` : "no result yet" };
}

function readJson(file: string): Record<string, unknown> | null {
  try {
    const v = JSON.parse(fs.readFileSync(file, "utf8"));
    return v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : null;
  } catch (e) {
    if ((e as NodeJS.ErrnoException)?.code !== "ENOENT") console.error("[lazy_prices] unreadable", file, e instanceof Error ? e.message : e);
    return null;
  }
}
const str = (v: unknown): string | null => (typeof v === "string" ? v : null);
const obj = (v: unknown): Record<string, unknown> => (v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : {});

export function lazyPricesLatest(root: string): LazyPricesLatest | null {
  const l = readJson(path.join(root, "latest.json"));
  if (!l) return null;
  return { build_id: str(obj(l.panel).build_id), code_version: str(l.code_version), generated_at: str(l.generated_at) };
}

/** Parse lazy_prices_run.py's one-line JSON summary (its last stdout line). */
export function parseLazyPricesSummary(stdout: string): RunOutcome {
  const last = stdout.trim().split("\n").pop() ?? "";
  let j: Record<string, unknown>;
  try { j = obj(JSON.parse(last)); } catch (e) {
    return { status: "error", detail: `unparseable summary: ${last.slice(0, 120)} (${e instanceof Error ? e.message : String(e)})` };
  }
  const st = str(j.status);
  if (st === "ok") return { status: "ok", detail: `verdict ${str(j.verdict)} build ${str(j.build_id)}: ${String(j.n_obs)} observations, ${String(j.elapsed_s)}s, peak ${String(j.peak_rss_mb)} MB` };
  if (st === "skipped" || st === "insufficient") return { status: st, detail: str(j.reason) ?? st };
  return { status: "error", detail: str(j.reason) ?? `status ${String(st)}` };
}

/** Run lazy_prices_run.py once (process-group kill on timeout). Never throws. */
export async function runLazyPrices(env: NodeJS.ProcessEnv, exec: WalkforwardExec = execGroupKill): Promise<RunOutcome> {
  try {
    const { stdout } = await exec(lazyPricesCmd(pkgVersion, env), {
      timeout: LAZY_PRICES_TIMEOUT_MS, killSignal: "SIGKILL", env: { ...env, ...WALKFORWARD_ENV_CAPS }, maxBuffer: 1024 * 1024, encoding: "utf8",
    });
    return parseLazyPricesSummary(stdout);
  } catch (e) {
    const err = e as { killed?: boolean; signal?: string | null; code?: unknown; stdout?: string; stderr?: string; message?: string };
    if (err.killed) return { status: "error", detail: `exceeded the ${Math.round(LAZY_PRICES_TIMEOUT_MS / 60_000)}-minute budget — process group killed` };
    const fromStdout = err.stdout ? parseLazyPricesSummary(err.stdout) : null;
    if (fromStdout && fromStdout.status === "error" && !fromStdout.detail.startsWith("unparseable")) return fromStdout;
    const tail = String(err.stderr || err.message || "").trim().slice(-160);
    return { status: "error", detail: `code=${String(err.code ?? "?")} signal=${String(err.signal ?? "none")}: ${tail}` };
  }
}

/** Read-only status for the "lazy_prices" diag probe: verdict and headline blocks only. */
export function lazyPricesStatus(env: Record<string, string | undefined> = process.env, root = lazyPricesRoot(env), panelRoot = pricePanelRoot(env)): Record<string, unknown> {
  const l = readJson(path.join(root, "latest.json"));
  const flag = lazyPricesFlag(env);
  return {
    probe: "lazy_prices",
    flag: { value: flag.value, source: flag.source, on: flag.on },
    schedule: `Sunday ${LAZY_PRICES_WINDOW_START_MIN / 60}:00-${LAZY_PRICES_WINDOW_END_MIN / 60}:00 ET, once per ISO week, budget ${LAZY_PRICES_TIMEOUT_MS / 60_000} min, ${LAZY_PRICES_MAX_RSS_MB} MB`,
    panel_build: currentPanelBuild(panelRoot),
    code_version: pkgVersion,
    latest: l ? {
      schema: str(l.schema), generated_at: str(l.generated_at), code_version: str(l.code_version), panel: obj(l.panel),
      verdict: obj(l.verdict), variants: l.variants ?? null, n_obs: l.n_obs ?? null, drops: obj(l.drops),
      delisted_exits: l.delisted_exits ?? null, summary: obj(l.summary), join: obj(l.join), caveat: str(l.caveat),
      rows_scored: l.rows_scored ?? null, elapsed_s: l.elapsed_s ?? null,
    } : null,
    last_run: readJson(path.join(root, "last_run.json")),
  };
}

let booted = false;

/** Weekly tick (unref'd; first tick one interval after boot). One run at a time. */
export function bootLazyPrices(env: NodeJS.ProcessEnv = process.env): void {
  if (booted) return;
  booted = true;
  let running = false;
  let lastAttemptWeek = "";
  const tick = async () => {
    if (running) return;
    const root = lazyPricesRoot(env);
    let free: number | null = null;
    try { const s = fs.statfsSync(root.startsWith("/data") ? "/data" : path.dirname(root)); free = Number(s.bavail) * Number(s.bsize); } catch { free = null; }
    const d = decideLazyPrices(Date.now(), {
      enabled: lazyPricesFlag(env).on, panelBuildId: currentPanelBuild(pricePanelRoot(env)),
      latest: lazyPricesLatest(root), lastAttemptWeek, codeVersion: pkgVersion, freeBytes: free,
    });
    if (d.action === "idle") return;
    lastAttemptWeek = d.week;
    if (d.action === "skip") {
      beat("lazy_prices_weekly", "finish", "skipped", d.reason);
      return;
    }
    running = true;
    beat("lazy_prices_weekly", "start");
    try {
      const out = await runLazyPrices(env);
      beat("lazy_prices_weekly", "finish", out.status === "error" ? "error" : out.status === "ok" ? "ok" : "skipped", `${out.status}: ${out.detail}`);
    } catch (e) {
      beat("lazy_prices_weekly", "finish", "error", e instanceof Error ? e.message : String(e));
    } finally {
      running = false;
    }
  };
  setInterval(() => { void tick(); }, LAZY_PRICES_TICK_MS).unref?.();
}
