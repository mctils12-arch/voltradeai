/**
 * edgar13f.ts — SEC EDGAR 13F-HR (institutional holdings) pipeline.
 * Stream #2 of the DATA STREAM EXPANSION build order (directive 2026-07-05;
 * hypothesis + ladder path filed in research/open_questions.md BEFORE this
 * first pull, per Reasoning Standard #10).
 *
 * HYPOTHESIS (gate 2, not attempted here): new-position clustering by
 * capacity-constrained managers in small caps precedes 60-90d
 * outperformance. The 45-day filing lag is modeled honestly — holdings are
 * STALE when public; the signal, if any, is crowd formation, not news.
 *
 * ROOT VALIDATION LADDER gate 1 (DATA) status: parser built against two
 * real, live-fetched 13F-HR filings (2026-07-04) and every extracted field
 * hand-checked against the filed XML — see edgar13f.test.ts. This module
 * surfaces the feed as RAW as-filed records only, no predictive claim.
 *
 * FOCUSED-MANAGER CAP (explicit, never silent): filings with more than
 * FOCUSED_MAX_HOLDINGS positions are archived as summary-only records
 * (holdings table omitted, holdingsOmitted=true). This is the hypothesis,
 * not a shortcut — mega-manager tables are index-hugging noise for a
 * cluster signal that targets capacity-constrained names (EDGE DOCTRINE
 * #2: fish where whales can't), and they would dominate the archive's
 * bytes. The cap is stated in the manifest and in every summary record.
 *
 * Boundary: no imports from trading logic — pure fetch + parse + archive,
 * same shape as edgarForm4.ts. No API key; SEC fair-access UA + spacing.
 */

import fs from "fs";
import path from "path";
import zlib from "zlib";
import { archiveBaseDir } from "./datacoreArchive";
import { parseFilingFeed } from "./edgarForm4";

export const FOCUSED_MAX_HOLDINGS = 250;

export interface Holding13F {
  issuer: string | null;
  titleOfClass: string | null;
  cusip: string | null;
  /** As-filed value in USD (full dollars since the 2023-01 rule change —
   *  older filings used thousands; this pipeline only sees current ones). */
  value: number | null;
  shares: number | null;
  sharesType: string | null;       // SH | PRN
  putCall: string | null;          // "Put" | "Call" | null (equity)
  discretion: string | null;       // SOLE | SHARED | OTHER
}

export interface Parsed13FPrimary {
  managerCik: string | null;
  managerName: string | null;
  periodOfReport: string | null;   // normalized to YYYY-MM-DD
  submissionType: string | null;   // 13F-HR | 13F-HR/A
  entryTotal: number | null;       // summaryPage tableEntryTotal
  valueTotal: number | null;       // summaryPage tableValueTotal (USD)
  otherManagersCount: number | null;
}

export interface Filing13F extends Parsed13FPrimary {
  accession: string;
  filedAt: string | null;
  cik: string;
  indexUrl: string;
  /** Full as-filed table for focused managers; null when entryTotal exceeds
   *  FOCUSED_MAX_HOLDINGS (summary-only record, holdingsOmitted=true). */
  holdings: Holding13F[] | null;
  holdingsOmitted: boolean;
}

// ── Namespace-tolerant XML helpers ──────────────────────────────────────────
// 13F documents prefix every tag (`<ns1:infoTable>`), unlike Form 4's bare
// ownership schema — the regexes accept an optional `prefix:` so the same
// parser handles both prefixed and unprefixed documents.

function nsTag(block: string, tag: string): string | null {
  const m = block.match(
    new RegExp(`<(?:[A-Za-z0-9_]+:)?${tag}[^>]*>([\\s\\S]*?)<\\/(?:[A-Za-z0-9_]+:)?${tag}>`, "i"),
  );
  if (!m) return null;
  const v = m[1].trim();
  return v.length ? v : null;
}

function nsBlocks(xml: string, tag: string): string[] {
  const out: string[] = [];
  const re = new RegExp(`<(?:[A-Za-z0-9_]+:)?${tag}[^>]*>([\\s\\S]*?)<\\/(?:[A-Za-z0-9_]+:)?${tag}>`, "gi");
  let m: RegExpExecArray | null;
  while ((m = re.exec(xml))) out.push(m[1]);
  return out;
}

function toNum(s: string | null): number | null {
  if (s == null) return null;
  const n = parseFloat(s.replace(/,/g, ""));
  return Number.isFinite(n) ? n : null;
}

function unescapeXml(s: string | null): string | null {
  if (s == null) return null;
  return s
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;|&apos;/g, "'");
}

/** EDGAR 13F dates are MM-DD-YYYY; normalize to ISO so records sort. */
export function normalize13FDate(s: string | null): string | null {
  if (!s) return null;
  const m = s.trim().match(/^(\d{2})-(\d{2})-(\d{4})$/);
  if (m) return `${m[3]}-${m[1]}-${m[2]}`;
  return /^\d{4}-\d{2}-\d{2}$/.test(s.trim()) ? s.trim() : s.trim();
}

/** Parses primary_doc.xml (<edgarSubmission>, thirteenffiler schema). */
export function parse13FPrimaryDoc(xml: string): Parsed13FPrimary {
  const filerBlock = nsTag(xml, "filer") || "";
  const managerBlock = nsTag(xml, "filingManager") || "";
  const summaryBlock = nsTag(xml, "summaryPage") || "";
  return {
    managerCik: nsTag(filerBlock, "cik"),
    managerName: unescapeXml(nsTag(managerBlock, "name")),
    periodOfReport: normalize13FDate(nsTag(xml, "periodOfReport")),
    submissionType: nsTag(xml, "submissionType"),
    entryTotal: toNum(nsTag(summaryBlock, "tableEntryTotal")),
    valueTotal: toNum(nsTag(summaryBlock, "tableValueTotal")),
    otherManagersCount: toNum(nsTag(summaryBlock, "otherIncludedManagersCount")),
  };
}

/** Parses the information-table XML (<informationTable> of <infoTable>s). */
export function parse13FInfoTable(xml: string, maxEntries = FOCUSED_MAX_HOLDINGS): Holding13F[] {
  const out: Holding13F[] = [];
  for (const block of nsBlocks(xml, "infoTable")) {
    if (out.length >= maxEntries) break;
    const shrsBlock = nsTag(block, "shrsOrPrnAmt") || "";
    out.push({
      issuer: unescapeXml(nsTag(block, "nameOfIssuer")),
      titleOfClass: unescapeXml(nsTag(block, "titleOfClass")),
      cusip: nsTag(block, "cusip"),
      value: toNum(nsTag(block, "value")),
      shares: toNum(nsTag(shrsBlock, "sshPrnamt")),
      sharesType: nsTag(shrsBlock, "sshPrnamtType"),
      putCall: nsTag(block, "putCall"),
      discretion: nsTag(block, "investmentDiscretion"),
    });
  }
  return out;
}

/** Picks {primary, info} XML names from a filing's index.json directory.
 *  primary_doc.xml is the EDGAR-standard name; the information table is
 *  whatever other .xml the filer uploaded (custom names are the norm —
 *  "2026qtr2submissionapr2026.xml", "13F_ATMOS_6_30_26.xml"). */
export function pick13FDocNames(indexJson: any): { primary: string | null; info: string | null } {
  const items: any[] = indexJson?.directory?.item || [];
  const xmls = items.map((it) => String(it.name || "")).filter((n) => /\.xml$/i.test(n));
  const primary = xmls.find((n) => /primary_doc\.xml$/i.test(n)) || null;
  const others = xmls.filter((n) => n !== primary);
  const info = others.find((n) => /info/i.test(n)) || others[0] || null;
  return { primary, info };
}

// ── Live fetch (injectable for tests; SEC fair-access spacing) ─────────────

const SEC_UA = { "User-Agent": "voltradeai-datacore/1.0 (research@voltradeai.com)" };
type FetchFn = (url: string, init?: any) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;

async function fetchText(fetchImpl: FetchFn, url: string): Promise<string> {
  const r = await fetchImpl(url, { headers: SEC_UA, signal: AbortSignal.timeout(15000) as any });
  if (!r.ok) throw new Error(`${url} -> ${r.status}`);
  return r.text();
}

/** Fetches the N most recent 13F-HR filings, parsed. Three requests per
 *  filing (index.json, primary doc, info table) with spacing — well under
 *  the 10 req/s fair-access ceiling. */
export async function fetchLatest13FFilings(
  limit: number,
  fetchImpl: FetchFn = fetch as any,
  delayMs = 200,
): Promise<Filing13F[]> {
  const feedUrl = `https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=13F-HR&company=&dateb=&owner=include&count=${Math.min(100, limit * 2)}&output=atom`;
  const feedXml = await fetchText(fetchImpl, feedUrl);
  const entries = parseFilingFeed(feedXml).slice(0, limit);
  const out: Filing13F[] = [];
  for (const entry of entries) {
    try {
      const base = entry.indexUrl.replace(/[^/]+$/, "");
      const indexJson = JSON.parse(await fetchText(fetchImpl, `${base}index.json`));
      const { primary, info } = pick13FDocNames(indexJson);
      if (!primary) continue;
      const head = parse13FPrimaryDoc(await fetchText(fetchImpl, `${base}${primary}`));
      let omit = (head.entryTotal ?? 0) > FOCUSED_MAX_HOLDINGS;
      let holdings: Holding13F[] | null = null;
      if (!omit && info) {
        holdings = parse13FInfoTable(await fetchText(fetchImpl, `${base}${info}`));
        // [REPAIR 2026-07-05, audit defect #10a] null entryTotal + a parse
        // that HITS the cap means the table may have been truncated — the
        // never-silent cap doctrine requires marking it, not shipping 250
        // rows that look complete. Treated as over-cap: summary-only.
        if (head.entryTotal == null && holdings.length >= FOCUSED_MAX_HOLDINGS) {
          omit = true;
          holdings = null;
        }
      }
      out.push({
        ...head,
        accession: entry.accession,
        filedAt: entry.filedAt,
        cik: entry.cik || head.managerCik || "",
        indexUrl: entry.indexUrl,
        holdings,
        holdingsOmitted: omit,
      });
    } catch {
      // one malformed filing never kills the batch
    }
    if (delayMs > 0) await new Promise((res) => setTimeout(res, delayMs));
  }
  return out;
}

// ── Archive (COLLECT-EVERYTHING: append-only JSONL day-files, dedup by
// accession, old days gzipped — identical lifecycle to <archive>/filings/) ──

const archivedAccessions = new Set<string>();
let seeded = false;

function filings13fDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "filings13f");
}

function seedSeen(dir: string, nowMs: number): void {
  for (const dayMs of [nowMs, nowMs - 86400_000]) {
    const fp = path.join(dir, `${new Date(dayMs).toISOString().slice(0, 10)}.jsonl`);
    try {
      for (const line of fs.readFileSync(fp, "utf8").split("\n")) {
        if (!line) continue;
        try { archivedAccessions.add(JSON.parse(line).accession); } catch {}
      }
    } catch {}
  }
}

export function archive13FFilings(filings: Filing13F[], baseDir?: string, nowMs?: number): number {
  const dir = filings13fDir(baseDir);
  const now = nowMs ?? Date.now();
  if (!seeded) { seedSeen(dir, now); seeded = true; }
  const fresh = filings.filter((f) => f.accession && !archivedAccessions.has(f.accession));
  if (!fresh.length) return 0;
  try {
    fs.mkdirSync(dir, { recursive: true });
    const fp = path.join(dir, `${new Date(now).toISOString().slice(0, 10)}.jsonl`);
    fs.appendFileSync(fp, fresh.map((f) => JSON.stringify(f)).join("\n") + "\n");
    fresh.forEach((f) => archivedAccessions.add(f.accession));
    if (archivedAccessions.size > 50_000) archivedAccessions.clear(); // bound memory
    return fresh.length;
  } catch (e: any) {
    console.error("[datacore] filings13f archive:", e?.message || e);
    return 0;
  }
}

export function gzipOld13FDays(baseDir?: string, nowMs?: number): number {
  const dir = filings13fDir(baseDir);
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

/** Reads archived 13F filings for the last N days (newest first), with
 *  holdings tables stripped to `topHoldings` rows by value — the full
 *  table stays on disk; API payloads stay bounded. */
export function read13FHistory(
  days = 30, baseDir?: string, nowMs?: number, maxFilings = 500, topHoldings = 25,
): Filing13F[] {
  const dir = filings13fDir(baseDir);
  const now = nowMs ?? Date.now();
  const out: Filing13F[] = [];
  const seen = new Set<string>();
  for (let d = 0; d < days && out.length < maxFilings; d++) {
    const day = new Date(now - d * 86400_000).toISOString().slice(0, 10);
    for (const fp of [path.join(dir, `${day}.jsonl`), path.join(dir, `${day}.jsonl.gz`)]) {
      let text: string | null = null;
      try {
        text = fp.endsWith(".gz")
          ? zlib.gunzipSync(fs.readFileSync(fp)).toString("utf8")
          : fs.readFileSync(fp, "utf8");
      } catch { continue; }
      for (const line of text.split("\n")) {
        if (!line) continue;
        try {
          const f: Filing13F = JSON.parse(line);
          if (!f.accession || seen.has(f.accession)) continue;
          seen.add(f.accession);
          out.push(trimHoldings(f, topHoldings));
        } catch {}
      }
    }
  }
  out.sort((a, b) => String(b.filedAt || "").localeCompare(String(a.filedAt || "")));
  return out.slice(0, maxFilings);
}

export function trimHoldings(f: Filing13F, top = 25): Filing13F {
  if (!f.holdings || f.holdings.length <= top) return f;
  const sorted = [...f.holdings].sort((a, b) => (b.value ?? 0) - (a.value ?? 0));
  return { ...f, holdings: sorted.slice(0, top) };
}

// ── In-memory cache + poll loop (eager boot, KNOWN BROKEN #9 lesson) ───────

let cache: { at: number; filings: Filing13F[] } | null = null;
let polling = false;

export function latest13FFilings(): { at: number; filings: Filing13F[] } | null {
  return cache;
}

export async function refresh13FCache(limit = 40): Promise<void> {
  try {
    const filings = await fetchLatest13FFilings(limit);
    if (filings.length > 0 || !cache) cache = { at: Date.now(), filings };
    try { archive13FFilings(filings); } catch {}
    try { gzipOld13FDays(); } catch {}
  } catch (e: any) {
    console.error("[datacore] edgar13f refresh:", e?.message || e);
  }
}

/** 15-min poll: outside 13F season the feed is a trickle; at the quarterly
 *  deadline burst (Feb/May/Aug/Nov 14) 40 filings x 96 polls/day keeps up
 *  with the small-manager tail this pipeline targets. */
export function boot13FPoll(intervalMs = 15 * 60_000): void {
  if (polling) return;
  polling = true;
  refresh13FCache();
  setInterval(() => { refresh13FCache(); }, intervalMs).unref?.();
}
