/**
 * fdaEvents.ts — FDA binary-event pipeline (approvals + advisory-committee
 * meeting dates). Stream #5 of the DATA STREAM EXPANSION build order
 * (hypothesis + ladder filed in research/open_questions.md BEFORE the
 * build; endpoints live-verified 2026-07-05 — see the research annex).
 *
 * HYPOTHESIS (gate 2, not attempted): binary-event timing for biotech
 * options — IV ramps into scheduled FDA catalysts; a theta-side input,
 * not directional. HONESTY: PDUFA target dates are NOT freely available
 * (FDA is barred from confirming pending applications, 21 CFR 314.430;
 * aggregator calendars are ToS-protected scrapes we do not touch). The
 * free sources that preserve the hypothesis:
 *  - openFDA drugsfda: approvals as they publish (backward-looking
 *    confirmation; keyless, 240/min + 1,000/day per IP).
 *  - Federal Register API: FDA advisory-committee meeting NOTICES —
 *    forward-looking binary catalysts (committee + sponsor + BLA + drug
 *    in the title), legally required publications, US-gov public domain.
 * Follow-on (filed, separate): mine company-disclosed PDUFA dates from
 * our own 8-K archive as labeled estimates.
 *
 * Boundary: no imports from trading logic. Keyless; optional FDA_API_KEY
 * raises the openFDA daily cap if ever needed (not required).
 */

import fs from "fs";
import path from "path";
import zlib from "zlib";
import { archiveBaseDir } from "./datacoreArchive";

export interface FdaEvent {
  t: "approval" | "adcom";
  /** dedup key: approvals appl|sub|date; adcom = FR document_number */
  key: string;
  appl: string | null;      // application number (approvals)
  date: string | null;      // event date YYYY-MM-DD (approval status date; adcom meeting date when parsed)
  sponsor: string | null;
  brand: string | null;     // brand name(s) (approvals)
  committee: string | null; // committee name (adcom)
  title: string | null;     // FR notice title (adcom — carries sponsor/BLA/drug)
  url: string | null;       // FR html_url (adcom evidence link)
  pub: string | null;       // FR publication date (adcom — when the public could know)
  dateConf: "parsed" | "unparsed" | null; // adcom meeting-date parse confidence
  rt: string;               // as-seen UTC date
}

// ── openFDA approvals ───────────────────────────────────────────────────────

function yyyymmddToIso(s: string | null | undefined): string | null {
  const m = String(s || "").match(/^(\d{4})(\d{2})(\d{2})$/);
  return m ? `${m[1]}-${m[2]}-${m[3]}` : null;
}

/** Parses openFDA drugsfda results into approval events — one per approved
 *  (AP) submission INSIDE the window. The search matches whole
 *  APPLICATIONS having any in-window AP submission, but each application
 *  carries its full submission history — without the window filter a
 *  30-day poll emitted 1,619 "approvals" including supplements from years
 *  back (caught in the live E2E 2026-07-05). */
export function parseApprovals(json: any, rt: string, sinceIso?: string): FdaEvent[] {
  const out: FdaEvent[] = [];
  for (const app of json?.results || []) {
    const appl = app?.application_number || null;
    const sponsor = app?.sponsor_name || null;
    const brand = (app?.products || []).map((p: any) => p?.brand_name).filter(Boolean).join("; ") || null;
    for (const sub of app?.submissions || []) {
      if (sub?.submission_status !== "AP") continue;
      const date = yyyymmddToIso(sub?.submission_status_date);
      if (!date || !appl) continue;
      if (sinceIso && date < sinceIso) continue;
      out.push({
        t: "approval",
        key: `${appl}|${sub?.submission_number ?? ""}|${date}`,
        appl, date, sponsor, brand,
        committee: null, title: null, url: null, pub: null, dateConf: null,
        rt,
      });
    }
  }
  return out;
}

// ── Federal Register advisory-committee notices ─────────────────────────────

const MONTHS: Record<string, string> = {
  january: "01", february: "02", march: "03", april: "04", may: "05", june: "06",
  july: "07", august: "08", september: "09", october: "10", november: "11", december: "12",
};

/** Extracts the FIRST "Month D, YYYY" date from notice text — AdCom notices
 *  state the meeting date up front ("The meeting will be held on ..."). A
 *  miss archives dateConf:"unparsed" with date:null — labeled, never
 *  guessed. */
export function parseMeetingDate(text: string | null | undefined): string | null {
  const m = String(text || "").match(/\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),\s+(\d{4})/i);
  if (!m) return null;
  const dd = m[2].padStart(2, "0");
  return `${m[3]}-${MONTHS[m[1].toLowerCase()]}-${dd}`;
}

export function parseAdcomNotices(json: any, rt: string): FdaEvent[] {
  const out: FdaEvent[] = [];
  for (const doc of json?.results || []) {
    if (!doc?.document_number) continue;
    const title = doc?.title || "";
    // committee name: the "... Advisory Committee" phrase in the title
    const cm = title.match(/([A-Z][A-Za-z,&\- ]*Advisory Committee)/);
    const date = parseMeetingDate(doc?.abstract) || parseMeetingDate(title);
    out.push({
      t: "adcom",
      key: String(doc.document_number),
      appl: (title.match(/\b(BLA|NDA|ANDA)\s*(\d{5,6})/) || []).slice(1).join("") || null,
      date,
      sponsor: (title.match(/From\s+([^;,]+?)(?:,? Inc\.?| LLC| Corp\.?| Ltd\.?|;|$)/i) || [])[1]?.trim() || null,
      brand: null,
      committee: cm ? cm[1] : null,
      title: title.slice(0, 300) || null,
      url: doc?.html_url || null,
      pub: doc?.publication_date || null,
      dateConf: date ? "parsed" : "unparsed",
      rt,
    });
  }
  return out;
}

// ── Fetch (injectable) ──────────────────────────────────────────────────────

type FetchFn = (url: string, init?: any) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;
const UA = { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)" };

async function getJson(fetchImpl: FetchFn, url: string): Promise<any> {
  const r = await fetchImpl(url, { headers: UA, signal: AbortSignal.timeout(20000) as any });
  if (!r.ok) throw new Error(`${url} -> ${r.status}`);
  return JSON.parse(await r.text());
}

export async function fetchFdaEvents(
  fetchImpl: FetchFn = fetch as any, nowMs?: number,
): Promise<FdaEvent[]> {
  const now = nowMs ?? Date.now();
  const rt = new Date(now).toISOString().slice(0, 10);
  const out: FdaEvent[] = [];
  const d30 = (ms: number) => new Date(ms).toISOString().slice(0, 10).replace(/-/g, "");
  // approvals in the last 30 days (window overlaps across polls; dedup absorbs)
  try {
    const url = "https://api.fda.gov/drug/drugsfda.json?search=" + encodeURIComponent(
      `submissions.submission_status_date:[${d30(now - 30 * 86400_000)}+TO+${d30(now)}]+AND+submissions.submission_status:AP`,
    ).replace(/%2B/g, "+") + "&limit=99";
    const sinceIso = new Date(now - 30 * 86400_000).toISOString().slice(0, 10);
    out.push(...parseApprovals(await getJson(fetchImpl, url), rt, sinceIso));
  } catch (e: any) {
    console.error("[datacore] fda approvals:", e?.message || e);
  }
  // AdCom meeting notices, newest first
  try {
    const url = "https://www.federalregister.gov/api/v1/documents.json" +
      "?conditions[agencies][]=food-and-drug-administration" +
      "&conditions[term]=" + encodeURIComponent('"advisory committee" meeting') +
      "&conditions[type][]=NOTICE&order=newest&per_page=40" +
      "&fields[]=title&fields[]=publication_date&fields[]=html_url&fields[]=abstract&fields[]=document_number";
    out.push(...parseAdcomNotices(await getJson(fetchImpl, url), rt));
  } catch (e: any) {
    console.error("[datacore] fda adcom:", e?.message || e);
  }
  return out;
}

// ── Archive ────────────────────────────────────────────────────────────────

const archivedKeys = new Set<string>();
let seeded = false;

function fdaDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "fda");
}

function seedSeen(dir: string, nowMs: number): void {
  for (let i = 0; i < 35; i++) { // approvals window is 30d — seed the whole overlap
    const day = new Date(nowMs - i * 86400_000).toISOString().slice(0, 10);
    for (const fp of [path.join(dir, `${day}.jsonl`), path.join(dir, `${day}.jsonl.gz`)]) {
      let text: string | null = null;
      try {
        text = fp.endsWith(".gz")
          ? zlib.gunzipSync(fs.readFileSync(fp)).toString("utf8")
          : fs.readFileSync(fp, "utf8");
      } catch { continue; }
      for (const line of text.split("\n")) {
        if (!line) continue;
        try { archivedKeys.add(JSON.parse(line).key); } catch {}
      }
    }
  }
}

export function archiveFdaEvents(events: FdaEvent[], baseDir?: string, nowMs?: number): number {
  const dir = fdaDir(baseDir);
  const now = nowMs ?? Date.now();
  if (!seeded) { seedSeen(dir, now); seeded = true; }
  const fresh = events.filter((e) => e.key && !archivedKeys.has(e.key));
  if (!fresh.length) return 0;
  try {
    fs.mkdirSync(dir, { recursive: true });
    const fp = path.join(dir, `${new Date(now).toISOString().slice(0, 10)}.jsonl`);
    fs.appendFileSync(fp, fresh.map((e) => JSON.stringify(e)).join("\n") + "\n");
    fresh.forEach((e) => archivedKeys.add(e.key));
    if (archivedKeys.size > 100_000) archivedKeys.clear();
    return fresh.length;
  } catch (e: any) {
    console.error("[datacore] fda archive:", e?.message || e);
    return 0;
  }
}

export function gzipOldFdaDays(baseDir?: string, nowMs?: number): number {
  const dir = fdaDir(baseDir);
  const now = nowMs ?? Date.now();
  let n = 0;
  try {
    for (const f of fs.readdirSync(dir)) {
      if (!f.endsWith(".jsonl")) continue;
      if (now - Date.parse(f.slice(0, 10)) < 2 * 86400_000) continue;
      const fp = path.join(dir, f);
      fs.writeFileSync(`${fp}.gz`, zlib.gzipSync(fs.readFileSync(fp)));
      fs.unlinkSync(fp);
      n++;
    }
  } catch {}
  return n;
}

// ── Cache + poll ────────────────────────────────────────────────────────────

let cache: { at: number; events: FdaEvent[] } | null = null;
let polling = false;

export function latestFdaEvents(): { at: number; events: FdaEvent[] } | null {
  return cache;
}

export async function refreshFdaCache(fetchImpl: FetchFn = fetch as any, nowMs?: number): Promise<void> {
  try {
    const events = await fetchFdaEvents(fetchImpl, nowMs);
    if (events.length || !cache) cache = { at: Date.now(), events };
    try { archiveFdaEvents(events, undefined, nowMs); } catch {}
    try { gzipOldFdaDays(undefined, nowMs); } catch {}
  } catch (e: any) {
    console.error("[datacore] fda refresh:", e?.message || e);
  }
}

/** 6h poll — both sources publish on daily-or-slower cycles; two requests
 *  per cycle sits far under the keyless 240/min + 1,000/day openFDA caps. */
export function bootFdaPoll(intervalMs = 6 * 60 * 60_000): void {
  if (polling) return;
  polling = true;
  refreshFdaCache();
  setInterval(() => { refreshFdaCache(); }, intervalMs).unref?.();
}
