// faaTfrTextCheck.ts — LADDER GATE 1 (DATA) for the text reader, RUNNING
// ([PIPELINE] 2026-11-23; research/open_questions.md 2026-11-22 (x4)).
//
// QUESTION, the same one scripts/tfr_text_gate1.ts asked once, on one day's
// list: when swimNotamText.ts turns a notice's words into an outline, is it
// the outline the FAA itself publishes for that notice?
//
// WHY IT CAN RUN ON THE SERVER WITH NO NEW REQUEST. A US restriction notice
// can reach this process twice: on the notice feed (swimNotam.ts), where its
// area is in the words only and our reader draws it, and on the FAA's public
// list (faaTfrList.ts), whose detail file carries the FAA's OWN outline. The
// route serves the FAA's outline and leaves the feed's row out. Before this
// module that second reading was dropped unmeasured. Here each such notice is
// judged ONCE, by the measure and the tolerance the one-day check fixed
// before its first run, and the verdict is kept across restarts.
//
// WHAT IS JUDGED. A notice the FAA lists WITH an outline and the feed has
// also sent:
//   agree / partial / disagree   the feed row's outline was read from its text (geometry.from "text")
//   not_read                     the feed sent the notice and our reader made no outline of it
//   feed_sent_geometry           the feed's row carried airspace geometry as sent: not the reader's work, counted apart
// A notice is judged once: a notice number never changes its text (a changed
// restriction is a new notice with a new number).
//
// WHAT THIS CANNOT KNOW. Only notices the feed sent while this process (or
// one before it, through the feed's keep-file) was listening AND the FAA
// lists with an outline; a notice the FAA publishes no outline for has no
// ground truth here. `not_read` is counted from the notices this process saw
// arrive without a shape, which a restart forgets: the not-read count is a
// floor. The FAA's outline here is the detail file's (abdMergedArea); the
// one-day check used the FAA's shape service, and the two were compared on
// 2026-11-22 (scripts/tfr_list_gate1.ts: 34 of 35 notices the same, the
// other differing by area count).
//
// A measurement of a RAW overlay's drawing. It changes nothing that is
// served or drawn. No trading path reads this module.
import fs from "fs";
import path from "path";
import { archiveBaseDir } from "./datacoreArchive";

export type Ring = Array<[number, number]>; // [lat, lon]
const NM_PER_DEG = 60;

/** distance in NM from p to the segment a-b, on a local flat map around p (shapes here are tens of NM across) */
function segDistNm(p: [number, number], a: [number, number], b: [number, number]): number {
  const k = Math.cos((p[0] * Math.PI) / 180);
  const ax = (a[1] - p[1]) * k * NM_PER_DEG, ay = (a[0] - p[0]) * NM_PER_DEG;
  const bx = (b[1] - p[1]) * k * NM_PER_DEG, by = (b[0] - p[0]) * NM_PER_DEG;
  const dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy;
  const t = l2 > 0 ? Math.max(0, Math.min(1, -(ax * dx + ay * dy) / l2)) : 0;
  return Math.hypot(ax + t * dx, ay + t * dy);
}
function oneWayNm(from: Ring, to: Ring): number {
  let worst = 0;
  for (const p of from) {
    let best = Infinity;
    for (let i = 0; i + 1 < to.length; i++) best = Math.min(best, segDistNm(p, to[i], to[i + 1]));
    worst = Math.max(worst, best);
  }
  return worst;
}
/** the larger one-way corner-to-outline distance between two closed outlines, NM */
export function outlineDistanceNm(a: Ring, b: Ring): number {
  return Math.max(oneWayNm(a, b), oneWayNm(b, a));
}
export function halfDiagonalNm(r: Ring): number {
  let la0 = 90, la1 = -90, lo0 = 180, lo1 = -180;
  for (const [la, lo] of r) { la0 = Math.min(la0, la); la1 = Math.max(la1, la); lo0 = Math.min(lo0, lo); lo1 = Math.max(lo1, lo); }
  const k = Math.cos((((la0 + la1) / 2) * Math.PI) / 180);
  return Math.hypot((la1 - la0) * NM_PER_DEG, (lo1 - lo0) * k * NM_PER_DEG) / 2;
}
/** fixed before the one-day check's first run (2026-11-22): 0.1 NM plus the chord sag of the FAA's own 36-sided circles */
export const TEXT_CHECK_TOLERANCE = "0.1 NM + 0.4% of the FAA outline's half-diagonal";
export const toleranceNm = (faa: Ring) => 0.1 + 0.004 * halfDiagonalNm(faa);

export type OutlineVerdict = "agree" | "partial" | "not_read" | "disagree";
export interface OutlineJudgement {
  verdict: OutlineVerdict;
  /** FAA outlines that have one of ours within tolerance */
  matched: number;
  /** the largest distance among the matched outlines, NM; null when none matched */
  worst_matched_nm: number | null;
  /** per FAA outline, the distance to the nearest of ours, NM; -1 = we have none */
  distances_nm: number[];
}
/** The one rule, shared with scripts/tfr_text_gate1.ts: the FAA's outlines against ours.
 *    agree     every FAA outline has one of ours within tolerance, and no area of the text was left unread
 *    partial   at least one FAA outline matched, and another did not or an area was left unread
 *    not_read  we produced no outline
 *    disagree  we produced outlines and none is within tolerance */
export function judgeOutlines(faa: Ring[], ours: Ring[], unread: number): OutlineJudgement {
  const distances = faa.map((f) => (ours.length ? Math.min(...ours.map((r) => outlineDistanceNm(f, r))) : Infinity));
  const ok = faa.map((f, i) => distances[i] <= toleranceNm(f));
  const matched = ok.filter(Boolean).length;
  const verdict: OutlineVerdict = !ours.length ? "not_read"
    : matched === faa.length && unread === 0 ? "agree"
    : matched > 0 ? "partial" : "disagree";
  const md = distances.filter((_, i) => ok[i]);
  return {
    verdict, matched,
    worst_matched_nm: md.length ? Math.round(Math.max(...md) * 1000) / 1000 : null,
    distances_nm: distances.map((d) => (Number.isFinite(d) ? Math.round(d * 1000) / 1000 : -1)),
  };
}

// ── the running form ────────────────────────────────────────────────────────
export const TEXT_CHECK_KEEP_VERSION = 1;
/** notices kept; beyond it the oldest verdicts are dropped and the drop is counted */
export const TEXT_CHECK_MAX_NOTICES = 5_000;
/** notices named in the route's answer that did not agree */
export const TEXT_CHECK_MAX_NAMED = 20;

export type CheckVerdict = OutlineVerdict | "feed_sent_geometry";
export interface CheckedNotice {
  verdict: CheckVerdict;
  /** when this server judged it */
  at: number;
  /** outlines the FAA publishes for the notice */
  faa: number;
  /** outlines our reader made from the feed's text; areas it could not follow */
  read: number; unread: number;
  matched: number;
  worst_matched_nm: number | null;
}
/** the feed's row for a notice, as far as this check needs it */
export interface FeedOutline { citation?: string; geometry: { kind: string; rings?: Ring[]; from?: string; unread?: number } | null }
/** the FAA list's row for a notice, as far as this check needs it */
export interface ListOutline { citation: string; geometry: { rings: Ring[] } | null }

export interface TextCheckView {
  /** when the first notice was judged; null = none yet */
  since: number | null;
  last_judged_at: number | null;
  /** notices compared outline against outline: agree + partial + disagree */
  compared: number;
  agree: number; partial: number; disagree: number;
  /** the feed sent the notice, the FAA publishes an outline for it, and our reader made none (a floor: see the module header) */
  not_read: number;
  /** the feed's own row carried geometry as sent; nothing of the reader's to judge */
  feed_sent_geometry: number;
  /** the largest distance among agreeing notices, NM */
  worst_agree_nm: number | null;
  tolerance: string;
  /** the newest notices that did not agree, at most TEXT_CHECK_MAX_NAMED */
  not_agreeing: Array<{ citation: string; verdict: CheckVerdict; at: number; faa: number; read: number; unread: number; matched: number }>;
  /** verdicts dropped at the size limit since the keep-file began */
  dropped: number;
}

interface KeepDoc { v: number; since: number | null; dropped: number; notices: Array<[string, CheckedNotice]> }
const VERDICTS: readonly CheckVerdict[] = ["agree", "partial", "not_read", "disagree", "feed_sent_geometry"];
const count = (v: unknown) => (typeof v === "number" && Number.isFinite(v) && v >= 0 ? Math.floor(v) : null);

export class FaaTfrTextCheck {
  private readonly notices = new Map<string, CheckedNotice>();
  since: number | null = null;
  dropped = 0;

  has(citation: string): boolean { return this.notices.has(citation); }
  get(citation: string): CheckedNotice | undefined { return this.notices.get(citation); }
  size(): number { return this.notices.size; }

  /** Judge every notice both sources hold that has no verdict yet. `unshaped` = citations of restriction notices
   *  the feed sent that could not be given a shape. Returns the number newly judged. Never throws on a bad ring:
   *  a row whose outline cannot be measured is left for the next pass, never guessed. */
  observe(list: readonly ListOutline[], feed: readonly FeedOutline[], unshaped: ReadonlySet<string>, now: number): number {
    const feedBy = new Map<string, FeedOutline>();
    for (const f of feed) if (f.citation) feedBy.set(f.citation, f);
    let added = 0;
    for (const l of list) {
      const faa = l.geometry?.rings ?? [];
      if (!faa.length || this.notices.has(l.citation)) continue;
      const f = feedBy.get(l.citation);
      const g = f?.geometry ?? null;
      let n: CheckedNotice | null = null;
      if (g && g.kind === "polygon" && g.from === "text" && Array.isArray(g.rings)) {
        const unread = count(g.unread) ?? 0;
        const j = judgeOutlines(faa, g.rings, unread);
        if (j.distances_nm.some((d) => Number.isNaN(d))) continue;
        n = { verdict: j.verdict, at: now, faa: faa.length, read: g.rings.length, unread, matched: j.matched, worst_matched_nm: j.worst_matched_nm };
      } else if (g && g.kind === "polygon") {
        n = { verdict: "feed_sent_geometry", at: now, faa: faa.length, read: 0, unread: 0, matched: 0, worst_matched_nm: null };
      } else if (g ? g.kind === "circle" : unshaped.has(l.citation)) {
        // an index circle is drawn only when the text gave no outline
        n = { verdict: "not_read", at: now, faa: faa.length, read: 0, unread: 0, matched: 0, worst_matched_nm: null };
      }
      if (!n) continue;
      this.notices.set(l.citation, n);
      this.since ??= now;
      added++;
    }
    while (this.notices.size > TEXT_CHECK_MAX_NOTICES) {
      // insertion order is judging order: the first key is the oldest verdict
      this.notices.delete(this.notices.keys().next().value as string);
      this.dropped++;
    }
    return added;
  }

  view(): TextCheckView {
    const c: Record<CheckVerdict, number> = { agree: 0, partial: 0, not_read: 0, disagree: 0, feed_sent_geometry: 0 };
    let worst: number | null = null, last: number | null = null;
    const named: TextCheckView["not_agreeing"] = [];
    for (const [citation, n] of Array.from(this.notices)) {
      c[n.verdict]++;
      last = Math.max(last ?? 0, n.at);
      if (n.verdict === "agree" && n.worst_matched_nm !== null) worst = Math.max(worst ?? 0, n.worst_matched_nm);
      if (n.verdict === "partial" || n.verdict === "disagree" || n.verdict === "not_read") named.push({ citation, verdict: n.verdict, at: n.at, faa: n.faa, read: n.read, unread: n.unread, matched: n.matched });
    }
    return {
      since: this.since, last_judged_at: last,
      compared: c.agree + c.partial + c.disagree, agree: c.agree, partial: c.partial, disagree: c.disagree,
      not_read: c.not_read, feed_sent_geometry: c.feed_sent_geometry,
      worst_agree_nm: worst, tolerance: TEXT_CHECK_TOLERANCE,
      not_agreeing: named.sort((a, b) => b.at - a.at).slice(0, TEXT_CHECK_MAX_NAMED), dropped: this.dropped,
    };
  }

  snapshot(): KeepDoc { return { v: TEXT_CHECK_KEEP_VERSION, since: this.since, dropped: this.dropped, notices: Array.from(this.notices) }; }
  /** Put saved verdicts back (boot). A verdict already held wins; malformed entries are left out. Returns those put back. */
  restore(raw: unknown): number {
    const d = raw as KeepDoc | null;
    if (!d || d.v !== TEXT_CHECK_KEEP_VERSION || !Array.isArray(d.notices)) return 0;
    let n = 0;
    for (const e of d.notices) {
      if (!Array.isArray(e) || typeof e[0] !== "string" || !/^FDC \d\/\d{4}$/.test(e[0]) || this.notices.has(e[0])) continue;
      const o = e[1] as CheckedNotice | null;
      const at = count(o?.at), faa = count(o?.faa), read = count(o?.read), unread = count(o?.unread), matched = count(o?.matched);
      if (!o || !VERDICTS.includes(o.verdict) || at === null || faa === null || read === null || unread === null || matched === null) continue;
      if (this.notices.size >= TEXT_CHECK_MAX_NOTICES) break;
      const w = typeof o.worst_matched_nm === "number" && Number.isFinite(o.worst_matched_nm) && o.worst_matched_nm >= 0 ? o.worst_matched_nm : null;
      this.notices.set(e[0], { verdict: o.verdict, at, faa, read, unread, matched, worst_matched_nm: w });
      n++;
    }
    if (n) {
      const s = count(d.since);
      const oldest = Math.min(...Array.from(this.notices.values()).map((x) => x.at));
      this.since = Math.min(this.since ?? Infinity, s !== null && s > 0 ? s : oldest);
      this.dropped += count(d.dropped) ?? 0;
    }
    return n;
  }
}

// ── the one process-wide check and its keep-file ────────────────────────────
let shared: FaaTfrTextCheck | null = null;
/** beside the list's own keep-file; verdicts only (no text, no outline), about 150 bytes a notice */
export function faaTfrTextCheckFile(baseDir?: string): string { return path.join(baseDir ?? path.dirname(archiveBaseDir()), "faa_tfr_text_check.json"); }

export function restoreFaaTfrTextCheck(check: FaaTfrTextCheck, file = faaTfrTextCheckFile()): number {
  if (!fs.existsSync(file)) return 0;
  try {
    return check.restore(JSON.parse(fs.readFileSync(file, "utf8")));
  } catch (e) {
    console.error(`[faaTfrTextCheck] saved verdicts not read back: ${(e as NodeJS.ErrnoException)?.code ?? "unparseable"}`);
    return 0;
  }
}
export async function saveFaaTfrTextCheck(check: FaaTfrTextCheck, file = faaTfrTextCheckFile()): Promise<boolean> {
  try {
    const tmp = `${file}.${process.pid}.tmp`;
    await fs.promises.mkdir(path.dirname(file), { recursive: true });
    await fs.promises.writeFile(tmp, JSON.stringify(check.snapshot()));
    await fs.promises.rename(tmp, file);
    return true;
  } catch (e) {
    console.error(`[faaTfrTextCheck] save failed: ${(e as NodeJS.ErrnoException)?.code ?? "error"}`);
    return false;
  }
}
/** The process's check; the saved verdicts are put back the first time it is asked for. */
export function sharedFaaTfrTextCheck(file?: string): FaaTfrTextCheck {
  if (!shared) { shared = new FaaTfrTextCheck(); restoreFaaTfrTextCheck(shared, file); }
  return shared;
}
export function _resetFaaTfrTextCheckForTests(): void { shared = null; }
