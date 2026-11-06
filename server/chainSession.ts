// chainSession.ts — READ-TIME SESSION MAPPING for the options chain archive
// (rule call "options-archive day-shift relabel 07-06 → 10-27", decided
// 2026-11-06 under the human's "fix all of this").
//
// THE SHIFT: before the 2026-10-28 fix (server/optionsChainArchive.ts header,
// audit 2026-10-27 §2.4 defect 1) the capture fired at ~00:40 ET and the file
// was labelled with the UTC date of the capture. Every day-file labelled
// 2026-07-06 … 2026-10-27 therefore holds the PREVIOUS session's closing
// quotes (verified on prod files: first records at 00:01–00:51 ET on the label
// date; 10-28 onward at 16:30–16:34 ET). The fix moved the pre-fix 10-28 file
// (session 10-27) to `_superseded/`. Readers that key on the label (chainIv,
// chainCspEntry, chainVs: DTE, the IV30 date, the CSP entry day, the VS day)
// were one session late on those days.
//
// THE FIX — read time only. No archive file is renamed, rewritten or deleted:
//   * a day-file with a `_captured/<label>.json` sidecar is a post-fix capture:
//     its session IS its label;
//   * a day-file WITHOUT a sidecar whose first record was captured on the
//     label's own UTC date was labelled by the pre-fix UTC rule, so its
//     session is the session whose quotes a capture at that instant holds
//     (sessionOfCapture: before that ET day's open, or on a non-trading day,
//     the previous NYSE trading day; otherwise that ET day);
//   * a `_superseded/` file is mapped by the same capture rule (the 10-27
//     session lives only there);
//   * anything else (no `t`, or a label that is not its capture's UTC date)
//     keeps its label — unknown provenance is never re-dated.
// When several files claim one session, a post-fix file wins; otherwise the
// latest capture wins and the others are reported as `shadowed`.
// Derived readers cache under `derived/session/<kind>/` (a new namespace): the
// old write-once caches under `derived/<kind>/` keep the shifted values as
// they were written, untouched, as the record of what earlier studies read.

import fs from "fs";
import path from "path";
import zlib from "zlib";
import { chainDir, chainCapturedFile, etClock, marketHolidays } from "./optionsChainArchive";

const DAY_RE = /^(\d{4}-\d{2}-\d{2})\.jsonl(?:\.gz)?$/;
const SUPERSEDED_RE = /^(\d{4}-\d{2}-\d{2})\.jsonl(?:\.gz)?(?:\.\d+)?$/;
/** Regular-session open, ET minutes. A capture before it holds the previous session. */
export const SESSION_OPEN_ET_MINUTES = 9 * 60 + 30;
/** Bytes read from the head of a day-file to find its first record. */
const HEAD_BYTES = 64 * 1024;

const isWeekend = (day: string): boolean => {
  const d = new Date(`${day}T00:00:00Z`).getUTCDay();
  return d === 0 || d === 6;
};
const addDays = (day: string, n: number): string =>
  new Date(Date.parse(`${day}T00:00:00Z`) + n * 86_400_000).toISOString().slice(0, 10);

export function isTradingDay(day: string, holidays: ReadonlySet<string> = marketHolidays()): boolean {
  return !isWeekend(day) && !holidays.has(day);
}

/** The latest NYSE trading day strictly before `day` (weekends + holidays skipped). */
export function previousTradingDay(day: string, holidays: ReadonlySet<string> = marketHolidays()): string {
  let d = addDays(day, -1);
  for (let i = 0; i < 14 && !isTradingDay(d, holidays); i++) d = addDays(d, -1);
  return d;
}

/** The trading session whose quotes a capture taken at `tMs` holds: that ET
 *  day once its session has opened, else the previous trading day. */
export function sessionOfCapture(tMs: number, holidays: ReadonlySet<string> = marketHolidays()): string | null {
  if (!Number.isFinite(tMs)) return null;
  const { mins, etDay } = etClock(tMs);
  if (!/^\d{4}-\d{2}-\d{2}$/.test(etDay) || !Number.isFinite(mins)) return null;
  if (isTradingDay(etDay, holidays) && mins >= SESSION_OPEN_ET_MINUTES) return etDay;
  return previousTradingDay(etDay, holidays);
}

/** First record's `t` of a .jsonl or .jsonl.gz day-file (null when absent or unreadable). */
export function headRecordT(fp: string): number | null {
  let fd: number | null = null;
  try {
    fd = fs.openSync(fp, "r");
    const buf = Buffer.alloc(HEAD_BYTES);
    const n = fs.readSync(fd, buf, 0, buf.length, 0);
    const head = /\.gz(?:\.\d+)?$/.test(fp)
      ? zlib.gunzipSync(buf.subarray(0, n), { finishFlush: zlib.constants.Z_SYNC_FLUSH })
      : buf.subarray(0, n);
    const line = head.toString("utf8").split("\n")[0];
    const t = Number(JSON.parse(line)?.t);
    return Number.isFinite(t) ? t : null;
  } catch {
    return null; // unreadable / partial first line: no capture time, so the label is kept
  } finally {
    if (fd !== null) fs.closeSync(fd);
  }
}

export type SessionBasis = "captured" | "prefix_capture" | "superseded" | "label";
export interface SessionSource { session: string; label: string; files: string[]; basis: SessionBasis; t: number | null }
export interface SessionIndex {
  /** session → the files read for it */
  sessions: Map<string, SessionSource>;
  /** sources that lost a same-session tie (reported, never read) */
  shadowed: SessionSource[];
}

const rank = (s: SessionSource) => (s.basis === "captured" ? 2 : 1);

/** Map every day-file (and every `_superseded/` file) to the session it holds. */
export function buildSessionIndex(baseDir?: string, holidays: ReadonlySet<string> = marketHolidays()): SessionIndex {
  const dir = chainDir(baseDir);
  const sources: SessionSource[] = [];
  let names: string[] = [];
  try { names = fs.readdirSync(dir); } catch { names = []; }
  const byLabel = new Map<string, string[]>();
  for (const f of names) {
    const m = DAY_RE.exec(f);
    if (m) (byLabel.get(m[1]) ?? byLabel.set(m[1], []).get(m[1])!).push(path.join(dir, f));
  }
  for (const [label, files] of byLabel) {
    files.sort();
    if (fs.existsSync(chainCapturedFile(label, baseDir))) {
      sources.push({ session: label, label, files, basis: "captured", t: null });
      continue;
    }
    const t = headRecordT(files[0]);
    const s = t !== null && new Date(t).toISOString().slice(0, 10) === label ? sessionOfCapture(t, holidays) : null;
    sources.push(s ? { session: s, label, files, basis: "prefix_capture", t } : { session: label, label, files, basis: "label", t });
  }
  let sup: string[] = [];
  try { sup = fs.readdirSync(path.join(dir, "_superseded")); } catch { sup = []; }
  for (const f of sup.sort()) {
    const m = SUPERSEDED_RE.exec(f);
    if (!m) continue;
    const fp = path.join(dir, "_superseded", f);
    const t = headRecordT(fp);
    const s = t !== null && new Date(t).toISOString().slice(0, 10) === m[1] ? sessionOfCapture(t, holidays) : null;
    if (s) sources.push({ session: s, label: m[1], files: [fp], basis: "superseded", t });
  }
  const sessions = new Map<string, SessionSource>();
  const shadowed: SessionSource[] = [];
  for (const src of sources) {
    const have = sessions.get(src.session);
    if (!have) { sessions.set(src.session, src); continue; }
    const better = rank(src) > rank(have) || (rank(src) === rank(have) && (src.t ?? -1) > (have.t ?? -1));
    if (better) { shadowed.push(have); sessions.set(src.session, src); } else shadowed.push(src);
  }
  return { sessions, shadowed };
}

// Memoized per base dir on the directory listing (names only: an existing
// file's head never changes; a new day adds a name).
const memo = new Map<string, { sig: string; index: SessionIndex }>();
function listingSig(baseDir?: string): string {
  const dir = chainDir(baseDir);
  const ls = (d: string) => { try { return fs.readdirSync(d).sort().join("|"); } catch { return ""; } };
  return `${ls(dir)}#${ls(path.join(dir, "_captured"))}#${ls(path.join(dir, "_superseded"))}`;
}
export function sessionIndex(baseDir?: string): SessionIndex {
  const key = chainDir(baseDir);
  const sig = listingSig(baseDir);
  const hit = memo.get(key);
  if (hit && hit.sig === sig) return hit.index;
  const index = buildSessionIndex(baseDir);
  memo.set(key, { sig, index });
  return index;
}

/** A derived reader's write-once cache, keyed by SESSION: `derived/session/<kind>/`.
 *  The old label-keyed caches (`derived/<kind>/`) are left as they were written. */
export function sessionDerivedDir(kind: "iv30" | "csp_entry" | "vs", baseDir?: string): string {
  return path.join(chainDir(baseDir), "derived", "session", kind);
}

/** Re-dated labels (label ≠ session) — the read-time correction, for reporting. */
export function sessionSummary(baseDir?: string): { sessions: number; redated: number; shadowed: number; first: string | null; last: string | null } {
  const ix = sessionIndex(baseDir);
  const days = [...ix.sessions.keys()].sort();
  let redated = 0;
  for (const s of ix.sessions.values()) if (s.session !== s.label) redated++;
  return { sessions: days.length, redated, shadowed: ix.shadowed.length, first: days[0] ?? null, last: days[days.length - 1] ?? null };
}
