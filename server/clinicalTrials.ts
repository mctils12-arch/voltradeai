/**
 * clinicalTrials.ts — ClinicalTrials.gov catalyst-drift archiver (EDGE
 * DOCTRINE axis (a), BUILD-FIRST rule 2: ACCUMULATION substitutes for
 * purchase). Industry-sponsored Phase 2/3 trials carry a sponsor-set
 * PRIMARY COMPLETION DATE — the earliest honest public bound on when
 * topline data can land. The registry shows only the CURRENT value; the
 * history of how that date MOVED (slips, terminations, withdrawals) is
 * something paid biotech-catalyst vendors sell and the API does not
 * serve. We record it ourselves: poll a rolling window, diff against the
 * last persisted snapshot, archive only the CHANGES.
 *
 * HYPOTHESIS (gate 2, NOT attempted — needs >= 52 weeks of accumulation;
 * pre-registered in research/open_questions.md): a sponsor pushing a
 * primary completion date out >= 90 days is a leading indicator of
 * enrollment/operational trouble, and small-cap sponsors (where one
 * trial is the company) are capacity-constrained territory for large
 * funds. Nothing here is a trading signal; this is a RAW archive.
 *
 * HONESTY: dates are what the SPONSOR self-reports (type ESTIMATED until
 * completion); the registry's own update lag is unmeasured, so `rt` (the
 * day WE saw a change) is the only knowledge-date we claim. Month-only
 * dates ("2027-04") are normalised to the 1st and flagged `coarse`.
 * A short/failed page fetch can never be mistaken for mass trial
 * departures: the diff refuses to run (and state is not overwritten)
 * when the result is < 70% of the previous snapshot.
 *
 * Boundary: no imports from trading logic. Keyless public API
 * (clinicaltrials.gov/api/v2), US-government data.
 */

import fs from "fs";
import path from "path";
import zlib from "zlib";
import { archiveBaseDir } from "./datacoreArchive";

export interface TrialState {
  n: string;               // NCT id
  sp: string | null;       // lead sponsor name
  ph: string | null;       // phases joined "PHASE2/PHASE3"
  st: string | null;       // overall status
  pcd: string | null;      // primary completion date (normalised YYYY-MM-DD)
  coarse: boolean;         // pcd was month-only in the registry
  pt: string | null;       // pcd type: ACTUAL | ESTIMATED
  up: string | null;       // last update posted
  fp: string | null;       // first posted
  en: number | null;       // enrollment count
  ti: string | null;       // brief title (120)
  co: string | null;       // first 3 conditions
}

export interface TrialEvent {
  t: "bootstrap" | "new" | "entered_window" | "pcd_shift" | "status" | "left_window";
  key: string;
  nct: string | null;
  sponsor: string | null;
  phase: string | null;
  status: string | null;
  prevStatus: string | null;
  pcd: string | null;
  prevPcd: string | null;
  shiftDays: number | null; // + = delayed, - = pulled forward
  coarse: boolean;
  title: string | null;
  cond: string | null;
  enroll: number | null;
  tracked: number | null;   // bootstrap only: snapshot size
  rt: string;               // as-seen UTC date
}

const DAY = 86400_000;
/** A diff against a snapshot this much bigger than the new result is a
 *  truncated/failed fetch, not 30% of trials vanishing overnight. */
export const MIN_RETAINED_FRACTION = 0.7;
/** Window: recently-completed (results pending) through ~1y forward. */
export const WINDOW_BACK_DAYS = 60;
export const WINDOW_FWD_DAYS = 365;
const MAX_PAGES = 12;
const PAGE_SIZE = 1000;

// ── parse ───────────────────────────────────────────────────────────────────

/** "2027-04" -> {iso:"2027-04-01", coarse:true}; "2027-04-15" -> coarse:false. */
export function normDate(s: string | null | undefined): { iso: string; coarse: boolean } | null {
  const m = String(s || "").match(/^(\d{4})-(\d{2})(?:-(\d{2}))?$/);
  if (!m) return null;
  return { iso: `${m[1]}-${m[2]}-${m[3] || "01"}`, coarse: !m[3] };
}

/** The slice of the ClinicalTrials.gov v2 response we read (verified live
 *  2026-10-19); every field optional because the registry omits freely. */
export interface CtStudy {
  protocolSection?: {
    identificationModule?: { nctId?: string; briefTitle?: string };
    statusModule?: {
      overallStatus?: string;
      primaryCompletionDateStruct?: { date?: string; type?: string };
      lastUpdatePostDateStruct?: { date?: string };
      studyFirstPostDateStruct?: { date?: string };
    };
    sponsorCollaboratorsModule?: { leadSponsor?: { name?: string; class?: string } };
    conditionsModule?: { conditions?: string[] };
    designModule?: { phases?: string[]; enrollmentInfo?: { count?: number } };
  };
}
export interface CtResponse { studies?: CtStudy[]; nextPageToken?: string }

export function parseTrials(json: CtResponse | null | undefined): TrialState[] {
  const out: TrialState[] = [];
  for (const s of json?.studies || []) {
    const p = s?.protocolSection;
    const n = p?.identificationModule?.nctId;
    if (!n) continue;
    const pcd = normDate(p?.statusModule?.primaryCompletionDateStruct?.date);
    out.push({
      n: String(n),
      sp: p?.sponsorCollaboratorsModule?.leadSponsor?.name ?? null,
      ph: (p?.designModule?.phases || []).join("/") || null,
      st: p?.statusModule?.overallStatus ?? null,
      pcd: pcd?.iso ?? null,
      coarse: pcd?.coarse ?? false,
      pt: p?.statusModule?.primaryCompletionDateStruct?.type ?? null,
      up: p?.statusModule?.lastUpdatePostDateStruct?.date ?? null,
      fp: p?.statusModule?.studyFirstPostDateStruct?.date ?? null,
      en: typeof p?.designModule?.enrollmentInfo?.count === "number" ? p.designModule.enrollmentInfo.count : null,
      ti: (p?.identificationModule?.briefTitle || "").slice(0, 120) || null,
      co: (p?.conditionsModule?.conditions || []).slice(0, 3).join("; ") || null,
    });
  }
  return out;
}

// ── diff ────────────────────────────────────────────────────────────────────

function daysBetween(a: string, b: string): number {
  return Math.round((Date.parse(b + "T00:00:00Z") - Date.parse(a + "T00:00:00Z")) / DAY);
}

function ev(t: TrialEvent["t"], key: string, cur: TrialState | null, prev: TrialState | null, rt: string): TrialEvent {
  const ref = cur || prev;
  return {
    t, key,
    nct: ref?.n ?? null, sponsor: ref?.sp ?? null, phase: ref?.ph ?? null,
    status: cur?.st ?? null, prevStatus: prev?.st ?? null,
    pcd: cur?.pcd ?? null, prevPcd: prev?.pcd ?? null,
    shiftDays: null,
    coarse: !!(cur?.coarse || prev?.coarse),
    title: ref?.ti ?? null, cond: ref?.co ?? null, enroll: ref?.en ?? null,
    tracked: null, rt,
  };
}

export class SuspectFetchError extends Error {}

/** Diffs the new snapshot against the persisted one. prev === null is the
 *  bootstrap run: ONE bootstrap marker, no change events (we never saw
 *  the "before", so claiming changes would be fabrication). Throws
 *  SuspectFetchError when the result shrank below MIN_RETAINED_FRACTION. */
export function diffTrials(prev: Record<string, TrialState> | null, cur: TrialState[], rt: string): TrialEvent[] {
  if (prev === null) {
    const b = ev("bootstrap", `bootstrap|${rt}`, null, null, rt);
    b.tracked = cur.length;
    return [b];
  }
  const prevN = Object.keys(prev).length;
  if (prevN > 0 && cur.length < prevN * MIN_RETAINED_FRACTION) {
    throw new SuspectFetchError(`clinicaltrials: ${cur.length} trials vs ${prevN} in last snapshot — refusing to diff a likely-truncated fetch`);
  }
  const out: TrialEvent[] = [];
  const seen = new Set<string>();
  const fresh = new Date(Date.parse(rt + "T00:00:00Z") - 14 * DAY).toISOString().slice(0, 10);
  for (const c of cur) {
    seen.add(c.n);
    const p = prev[c.n];
    if (!p) {
      // Genuinely new registration vs. an old trial whose date rolled into
      // the forward edge of the window — different phenomena, separate labels.
      const isNew = !!c.fp && c.fp >= fresh;
      out.push(ev(isNew ? "new" : "entered_window", `${isNew ? "new" : "entered_window"}|${c.n}|${rt}`, c, null, rt));
      continue;
    }
    if (p.pcd && c.pcd && p.pcd !== c.pcd) {
      const e = ev("pcd_shift", `pcd_shift|${c.n}|${p.pcd}>${c.pcd}`, c, p, rt);
      e.shiftDays = daysBetween(p.pcd, c.pcd);
      out.push(e);
    }
    if (p.st !== c.st) out.push(ev("status", `status|${c.n}|${p.st}>${c.st}`, c, p, rt));
  }
  for (const nct of Object.keys(prev)) {
    if (!seen.has(nct)) out.push(ev("left_window", `left_window|${nct}|${rt}`, null, prev[nct], rt));
  }
  return out;
}

// ── fetch (injectable) ──────────────────────────────────────────────────────

export type FetchFn = (url: string, init?: RequestInit) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;
const UA = { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)" };
const FIELDS = [
  "NCTId", "BriefTitle", "LeadSponsorName", "Phase", "OverallStatus", "PrimaryCompletionDate",
  "LastUpdatePostDate", "StudyFirstPostDate", "EnrollmentCount", "Condition",
].join(",");

export function buildTrialsUrl(nowMs: number, pageToken?: string): string {
  const iso = (ms: number) => new Date(ms).toISOString().slice(0, 10);
  const adv = `AREA[Phase](PHASE2 OR PHASE3) AND AREA[StudyType]INTERVENTIONAL AND AREA[LeadSponsorClass]INDUSTRY` +
    ` AND AREA[PrimaryCompletionDate]RANGE[${iso(nowMs - WINDOW_BACK_DAYS * DAY)},${iso(nowMs + WINDOW_FWD_DAYS * DAY)}]`;
  return "https://clinicaltrials.gov/api/v2/studies?filter.advanced=" + encodeURIComponent(adv) +
    `&pageSize=${PAGE_SIZE}&fields=${FIELDS}` + (pageToken ? `&pageToken=${encodeURIComponent(pageToken)}` : "");
}

/** Pages through the whole window. THROWS if pages remain after MAX_PAGES —
 *  a partial snapshot must never reach the diff. */
export async function fetchTrials(fetchImpl: FetchFn = fetch as unknown as FetchFn, nowMs: number = Date.now()): Promise<TrialState[]> {
  const all: TrialState[] = [];
  let token: string | undefined;
  for (let i = 0; i < MAX_PAGES; i++) {
    const r = await fetchImpl(buildTrialsUrl(nowMs, token), { headers: UA, signal: AbortSignal.timeout(30000) });
    if (!r.ok) throw new Error(`clinicaltrials page ${i} -> ${r.status}`);
    const json = JSON.parse(await r.text()) as CtResponse;
    all.push(...parseTrials(json));
    token = json?.nextPageToken || undefined;
    if (!token) return all;
  }
  throw new Error(`clinicaltrials: more than ${MAX_PAGES} pages — refusing a partial snapshot`);
}

// ── archive ────────────────────────────────────────────────────────────────

function ctDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "clinical_trials");
}

const STATE_FILE = "_state.json.gz"; // "_" prefix: skipped by newestDataFile / day listings

export function loadTrialState(baseDir?: string): Record<string, TrialState> | null {
  try {
    return JSON.parse(zlib.gunzipSync(fs.readFileSync(path.join(ctDir(baseDir), STATE_FILE))).toString("utf8"));
  } catch { return null; }
}

export function saveTrialState(cur: TrialState[], baseDir?: string): void {
  const dir = ctDir(baseDir);
  fs.mkdirSync(dir, { recursive: true });
  const map: Record<string, TrialState> = {};
  for (const c of cur) map[c.n] = c;
  const fp = path.join(dir, STATE_FILE);
  fs.writeFileSync(fp + ".tmp", zlib.gzipSync(JSON.stringify(map)));
  fs.renameSync(fp + ".tmp", fp); // atomic: a crash never leaves a half state
}

function readDay(dir: string, day: string): TrialEvent[] {
  for (const fp of [path.join(dir, `${day}.jsonl`), path.join(dir, `${day}.jsonl.gz`)]) {
    let text: string;
    try {
      text = fp.endsWith(".gz") ? zlib.gunzipSync(fs.readFileSync(fp)).toString("utf8") : fs.readFileSync(fp, "utf8");
    } catch { continue; }
    const out: TrialEvent[] = [];
    for (const line of text.split("\n")) {
      if (!line) continue;
      try { out.push(JSON.parse(line)); } catch { continue; }
    }
    return out;
  }
  return [];
}

export function readArchivedTrialDay(iso: string, baseDir?: string): TrialEvent[] {
  return readDay(ctDir(baseDir), iso);
}

export function listArchivedTrialDates(baseDir?: string, limit = 90): string[] {
  let files: string[];
  try { files = fs.readdirSync(ctDir(baseDir)); } catch { return []; }
  return files
    .map((f) => f.match(/^(\d{4}-\d{2}-\d{2})\.jsonl(\.gz)?$/))
    .filter((m): m is RegExpMatchArray => !!m)
    .map((m) => m[1])
    .sort().reverse().slice(0, limit);
}

/** Appends events not already in today's file (idempotent re-run). */
export function archiveTrialEvents(events: TrialEvent[], baseDir?: string, nowMs?: number): number {
  if (!events.length) return 0;
  const dir = ctDir(baseDir);
  const day = new Date(nowMs ?? Date.now()).toISOString().slice(0, 10);
  const have = new Set(readDay(dir, day).map((e) => e.key));
  const fresh = events.filter((e) => e.key && !have.has(e.key));
  if (!fresh.length) return 0;
  fs.mkdirSync(dir, { recursive: true });
  fs.appendFileSync(path.join(dir, `${day}.jsonl`), fresh.map((e) => JSON.stringify(e)).join("\n") + "\n");
  return fresh.length;
}

export function gzipOldTrialDays(baseDir?: string, nowMs?: number): number {
  const dir = ctDir(baseDir);
  const now = nowMs ?? Date.now();
  let n = 0;
  try {
    for (const f of fs.readdirSync(dir)) {
      if (!/^\d{4}-\d{2}-\d{2}\.jsonl$/.test(f)) continue;
      if (now - Date.parse(f.slice(0, 10)) < 2 * DAY) continue;
      const fp = path.join(dir, f);
      fs.writeFileSync(`${fp}.gz`, zlib.gzipSync(fs.readFileSync(fp)));
      fs.unlinkSync(fp);
      n++;
    }
  } catch (e) {
    console.error("[datacore] clinical trials gzip:", e instanceof Error ? e.message : e);
  }
  return n;
}

// ── run once + cache + poll ─────────────────────────────────────────────────

/** One full cycle: fetch -> diff -> archive events -> persist state.
 *  Order matters: events first, state second. A crash between them
 *  re-emits the same events next cycle, which archiveTrialEvents'
 *  same-day key check absorbs; the reverse order would LOSE them. */
export async function runClinicalTrialsOnce(
  fetchImpl: FetchFn = fetch as unknown as FetchFn, nowMs: number = Date.now(), baseDir?: string,
): Promise<{ tracked: number; events: number }> {
  const rt = new Date(nowMs).toISOString().slice(0, 10);
  const cur = await fetchTrials(fetchImpl, nowMs);
  if (!cur.length) throw new Error("clinicaltrials: empty snapshot — refusing to record it");
  const events = diffTrials(loadTrialState(baseDir), cur, rt);
  const written = archiveTrialEvents(events, baseDir, nowMs);
  saveTrialState(cur, baseDir);
  try { gzipOldTrialDays(baseDir, nowMs); } catch (e) {
    console.error("[datacore] clinical trials gzip:", e instanceof Error ? e.message : e);
  }
  return { tracked: cur.length, events: written };
}

let cache: { at: number; tracked: number; events: TrialEvent[] } | null = null;
let polling = false;

export function latestClinicalTrials(): { at: number; tracked: number; events: TrialEvent[] } | null {
  return cache;
}

export function _resetClinicalTrialsForTests(): void {
  cache = null;
  polling = false;
}

/** Recent (last 14 archived days) change events, newest first. */
export function recentTrialEvents(baseDir?: string, nowMs?: number): TrialEvent[] {
  const cutoff = new Date((nowMs ?? Date.now()) - 14 * DAY).toISOString().slice(0, 10);
  const out: TrialEvent[] = [];
  for (const d of listArchivedTrialDates(baseDir, 30)) {
    if (d < cutoff) break;
    out.push(...readArchivedTrialDay(d, baseDir).filter((e) => e.t !== "bootstrap"));
  }
  return out;
}

export async function refreshClinicalTrials(fetchImpl: FetchFn = fetch as unknown as FetchFn, nowMs?: number, baseDir?: string): Promise<void> {
  try {
    const r = await runClinicalTrialsOnce(fetchImpl, nowMs, baseDir);
    cache = { at: Date.now(), tracked: r.tracked, events: recentTrialEvents(baseDir, nowMs) };
  } catch (e) {
    // Loud, never silent (Law V spirit): a failed cycle keeps the old
    // snapshot and old cache; the next cycle retries from the same state.
    console.error("[datacore] clinical trials:", e instanceof Error ? e.message : e);
    if (!cache) {
      const events = recentTrialEvents(baseDir, nowMs);
      const st = loadTrialState(baseDir);
      if (st) cache = { at: Date.now(), tracked: Object.keys(st).length, events };
    }
  }
}

/** 12h poll — the registry refreshes daily; one cycle is ~2-3 requests. */
export function bootClinicalTrialsPoll(intervalMs = 12 * 60 * 60_000): void {
  if (polling) return;
  polling = true;
  refreshClinicalTrials();
  setInterval(() => { refreshClinicalTrials(); }, intervalMs).unref?.();
}
