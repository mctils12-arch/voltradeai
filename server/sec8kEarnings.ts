/**
 * sec8kEarnings.ts — SEC 8-K Item 2.02 (Results of Operations and Financial
 * Condition) earnings-language pipeline. The lawful transcript substitute
 * named in research/open_questions.md NEW DATA ROOTS #1 (build-order
 * rationale: EDGAR history already exists, so gate 2 is testable immediately,
 * not time-blocked like the jobs/patents roots).
 *
 * LICENSING (verified 2026-07-04, see wishlist.md NEW DATA ROOTS #1): EDGAR
 * is public-record, free, "no restrictions on public domain use," fair-access
 * policy asks only for a descriptive User-Agent and a sane request rate —
 * same terms edgarForm4.ts already relies on. Motley Fool / Seeking Alpha /
 * FMP transcript sources are PROHIBITED or paid-restricted and are
 * deliberately NOT used here; this module only ever reads the issuer's own
 * as-filed Exhibit 99 press release, never a third-party transcript.
 *
 * ROOT VALIDATION LADDER gate 1 (DATA) status: PASSED for the extraction
 * path — the feed parser, item-code filter, exhibit picker, and HTML-to-text
 * converter were each built against real, live-fetched 8-K filings and hand
 * checked against the actual rendered exhibit text (see
 * sec8kEarnings.test.ts for the two fixtures and what was verified). Gate 2
 * (does guidance/results language predict forward returns better than a
 * size-matched random-entry base rate) is NOT attempted here — this module
 * and its /api/data/earnings-language route surface the extracted text as a
 * RAW-DATA overlay only (as-filed display, no predictive claim), per
 * datacore/README.md's RAW-vs-SIGNAL boundary rule.
 *
 * HONEST GAP (carried from the wishlist entry): Q&A sessions are almost
 * never filed as an exhibit — this pipeline gets prepared remarks/guidance
 * language only, not the full call. Filings with no EX-99-class exhibit
 * (results announced only in the 8-K body, or via a linked press release
 * not filed as an exhibit) are skipped, not fabricated.
 *
 * Boundary: no imports from trading logic — pure fetch + parse + archive,
 * same shape as edgarForm4.ts (discrete dated filings, not continuous
 * tracks). Deliberately does not import edgarForm4.ts — each datacore
 * pipeline stays a self-contained module per existing convention (nasaFirms
 * vs vesselStream also don't share internals).
 */

import fs from "fs";
import path from "path";
import zlib from "zlib";
import { archiveBaseDir } from "./datacoreArchive";
import { resolveCacheItems } from "./cacheBackfill";
import { trackSet } from "./heapDiag";

// ── Minimal dependency-free Atom-feed tag extraction (mirrors edgarForm4.ts's
// approach: SEC's feed schema is flat and predictable, a full XML parser is
// unnecessary weight for it). ───────────────────────────────────────────────

function tagContent(block: string, tag: string): string | null {
  const m = block.match(new RegExp(`<${tag}[^>]*>([\\s\\S]*?)<\\/${tag}>`, "i"));
  if (!m) return null;
  const v = m[1].trim();
  return v.length ? v : null;
}

function allBlocks(xml: string, tag: string): string[] {
  const out: string[] = [];
  const re = new RegExp(`<${tag}[^>]*>([\\s\\S]*?)<\\/${tag}>`, "gi");
  let m: RegExpExecArray | null;
  while ((m = re.exec(xml))) out.push(m[1]);
  return out;
}

export interface Filing8KEntry {
  accession: string;
  cik: string;
  companyName: string;
  filedAt: string | null;
  /** EDGAR getcurrent entry <updated> timestamp — the lookahead-free
   *  "public could know it" time the manifest documents
   *  ([REPAIR 2026-07-05, audit defect #5]: the field was documented but
   *  never stored; filedAt is only a date). */
  acceptanceDatetime: string | null;
  itemCodes: string[];
  indexUrl: string;
}

/** Parses the EDGAR "getcurrent" Atom feed for form type 8-K. Unlike Form 4's
 *  feed (one entry per filer/issuer pair needing dedup), 8-K "getcurrent"
 *  emits one entry per filing already, keyed by accession. The item-code
 *  list SEC embeds in the summary (e.g. "Item 2.02: Results of Operations
 *  and Financial Condition") is read directly off this feed — no need to
 *  open the filing itself just to know which items it covers. */
export function parse8KFeed(atomXml: string): Filing8KEntry[] {
  const out: Filing8KEntry[] = [];
  for (const entryBlock of allBlocks(atomXml, "entry")) {
    const title = (tagContent(entryBlock, "title") || "").trim();
    const linkMatch = entryBlock.match(/<link[^>]*href="([^"]+)"/i);
    const indexUrl = linkMatch ? linkMatch[1] : null;
    const summary = tagContent(entryBlock, "summary") || "";
    const accMatch = summary.match(/AccNo:&lt;\/b&gt;\s*([0-9-]+)/i) || summary.match(/AccNo:<\/b>\s*([0-9-]+)/i);
    const filedMatch = summary.match(/Filed:&lt;\/b&gt;\s*([0-9-]+)/i) || summary.match(/Filed:<\/b>\s*([0-9-]+)/i);
    const nameMatch = title.match(/^8-K\s*-\s*(.+?)\s*\((\d+)\)/i);
    if (!indexUrl || !accMatch || !nameMatch) continue;
    const itemCodes = Array.from(new Set(Array.from(summary.matchAll(/Item\s+(\d+\.\d+)\s*:/gi)).map((m) => m[1])));
    out.push({
      accession: accMatch[1],
      cik: nameMatch[2],
      companyName: nameMatch[1],
      filedAt: filedMatch ? filedMatch[1] : null,
      acceptanceDatetime: tagContent(entryBlock, "updated"),
      itemCodes,
      indexUrl,
    });
  }
  return out;
}

/** Item 2.02 = "Results of Operations and Financial Condition" — the item
 *  code that gates a filing into this earnings-language pipeline. */
export function hasItem202(entry: Filing8KEntry): boolean {
  return entry.itemCodes.includes("2.02");
}

/** Picks the first EX-99-class exhibit out of a filing's index.htm listing
 *  (row-content matching, not fixed column offsets — the index table's
 *  column order has been observed to vary between filer agents, e.g.
 *  "EX-99" vs "EXHIBIT 99.1" in the description column, both landing in an
 *  "EX-99*" Type column). Returns the path relative to sec.gov, or null if
 *  the filing has no such exhibit (results announced only in the 8-K body —
 *  the HONEST GAP noted above; such filings are skipped, not guessed at). */
export function pickExhibit99Href(indexHtml: string): string | null {
  const rows = indexHtml.match(/<tr[^>]*>[\s\S]*?<\/tr>/gi) || [];
  for (const row of rows) {
    if (!/EX-?\s*99/i.test(row)) continue;
    const hrefMatch = row.match(/href="([^"]+\.htm[l]?)"/i);
    if (hrefMatch) return hrefMatch[1];
  }
  return null;
}

// ── Dependency-free HTML -> plain text (exhibit press releases are simple
// filer-agent-generated HTML, not general web pages — script/style stripping
// plus block-tag-to-newline plus a common-entity decode table covers them). ─

const NAMED_ENTITIES: Record<string, string> = {
  amp: "&", lt: "<", gt: ">", quot: '"', apos: "'", nbsp: " ",
  mdash: "—", ndash: "–", lsquo: "‘", rsquo: "’",
  ldquo: "“", rdquo: "”", hellip: "…", copy: "©", reg: "®",
};

function decodeEntities(s: string): string {
  return s.replace(/&(#x?[0-9a-fA-F]+|[a-zA-Z]+);/g, (m, ent) => {
    if (ent[0] === "#") {
      const isHex = ent[1] === "x" || ent[1] === "X";
      const code = parseInt(ent.slice(isHex ? 2 : 1), isHex ? 16 : 10);
      return Number.isFinite(code) ? String.fromCodePoint(code) : m;
    }
    return NAMED_ENTITIES[ent.toLowerCase()] ?? m;
  });
}

export function htmlToText(html: string): string {
  let t = html
    .replace(/<script[\s\S]*?<\/script>/gi, " ")
    .replace(/<style[\s\S]*?<\/style>/gi, " ")
    .replace(/<(br|\/p|\/div|\/tr|\/li|\/h[1-6])[^>]*>/gi, "\n")
    .replace(/<[^>]+>/g, " ");
  t = decodeEntities(t).replace(/\u00A0/g, " "); // numeric &#160; decodes to a literal NBSP, not a regular space
  t = t.replace(/[ \t]+/g, " ").replace(/[ \t]*\n[ \t]*/g, "\n").replace(/\n{3,}/g, "\n\n");
  return t.trim();
}

export const MAX_EXHIBIT_TEXT_LEN = 30_000;

export interface Earnings8K {
  accession: string;
  cik: string;
  companyName: string;
  filedAt: string | null;
  acceptanceDatetime: string | null;
  /** exact CIK match against SEC company_tickers.json; null = unlisted
   *  filer or map miss — never guessed */
  ticker: string | null;
  itemCodes: string[];
  indexUrl: string;
  exhibitUrl: string;
  text: string;
  textLength: number;
  truncated: boolean;
  /** JOINT filings only: the OTHER registrants that filed this same accession
   *  (a parent and its subsidiaries file one 8-K — DTE Energy + DTE Electric,
   *  Southern Co + five utilities, Sempra + SoCalGas + SDG&E…). EDGAR's
   *  getcurrent feed lists one entry per registrant, so the row carries them
   *  here instead of becoming one row per registrant. The row's own
   *  cik/companyName/ticker are the PRIMARY registrant: the first feed entry
   *  with a resolved ticker, else the first entry. Absent (not `[]`) for a
   *  single-registrant filing. Rows archived before v1.0.1352 were written
   *  one line per registrant; the readers fold them into this shape. */
  coRegistrants?: Earnings8KRegistrant[];
}

export interface Earnings8KRegistrant {
  cik: string;
  companyName: string;
  ticker: string | null;
}

/** Folds one-row-per-registrant copies of a JOINT filing into one row per
 *  accession (first-appearance order kept). Primary = the first row with a
 *  resolved ticker, else the first row; every other distinct CIK becomes a
 *  co-registrant (and co-registrants the rows already carry are kept).
 *  Repeated lines of the SAME CIK (a restart re-seed gap) collapse to one
 *  without inventing a co-registrant. Single rows are returned untouched —
 *  no `coRegistrants` key is added. Used by the fetch layer (before any
 *  exhibit is fetched) and by both archive readers (legacy rows). */
export function mergeCoRegistrants<T extends Earnings8K>(rows: T[]): T[] {
  const groups = new Map<string, T[]>();
  for (const r of rows) {
    const g = groups.get(r.accession);
    if (g) g.push(r); else groups.set(r.accession, [r]);
  }
  const out: T[] = [];
  for (const g of groups.values()) {
    if (g.length === 1) { out.push(g[0]); continue; }
    const primary = g.find((r) => r.ticker) ?? g[0];
    const seenCik = new Set<string>([primary.cik]);
    const co: Earnings8KRegistrant[] = [];
    const add = (x: Earnings8KRegistrant) => {
      if (seenCik.has(x.cik)) return;
      seenCik.add(x.cik);
      co.push({ cik: x.cik, companyName: x.companyName, ticker: x.ticker ?? null });
    };
    for (const r of g) {
      if (r !== primary) add(r);
      for (const x of r.coRegistrants || []) add(x);
    }
    const merged: T = { ...primary };
    if (co.length) merged.coRegistrants = co; else delete merged.coRegistrants;
    out.push(merged);
  }
  return out;
}

// ── fetch layer (injectable for tests, same FetchFn shape as edgarForm4.ts /
// nasaFirms.ts) ─────────────────────────────────────────────────────────────
type FetchFn = (url: string, init?: any) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;
const SEC_UA = { "User-Agent": "voltradeai-datacore/1.0 (research@voltradeai.com)" };

async function fetchText(fetchImpl: FetchFn, url: string): Promise<string> {
  const r = await fetchImpl(url, { headers: SEC_UA, signal: AbortSignal.timeout(12000) as any });
  if (!r.ok) throw new Error(`${url} -> ${r.status}`);
  return r.text();
}

function toExhibitUrl(indexUrl: string, href: string): string {
  if (/^https?:\/\//i.test(href)) return href;
  return `https://www.sec.gov${href.startsWith("/") ? "" : "/"}${href}`;
}

/** CIK -> ticker via SEC company_tickers.json — EXACT numeric-CIK match
 *  (no name normalization involved), null for unlisted filers. 24h cache;
 *  a fetch failure degrades to an empty map (tickers null), never a
 *  batch failure. [REPAIR 2026-07-05, audit defect #5]: the manifest's
 *  ticker entity key was documented but never stored.
 *  [REPAIR 2026-07-11, found while building the gate-2 forward-return test
 *  for this stream's hypothesis]: SEC lists MULTIPLE rows per CIK when an
 *  issuer has more than one listed security class — e.g. CIK 797468
 *  (Occidental Petroleum) has both "OXY" (common) and "OXY-WT" (warrants).
 *  The old loop did `map.set` unconditionally for every row, so whichever
 *  row appeared LAST in the feed silently won — verified live 2026-07-11
 *  that this mapped Occidental's 8-K to "OXY-WT", not "OXY", corrupting
 *  the ticker field this pipeline publishes on /api/data/earnings-language
 *  (a customer-facing RAW-DATA field) and would have corrupted any forward-
 *  return signal test that fetched prices for the wrong instrument (a
 *  warrant, not the common stock). Fix: never let a suffixed ticker
 *  (contains "-", e.g. -WT/-WS/-PR/-U/-RT) overwrite an already-resolved
 *  plain one for the same CIK — order-independent, so feed ordering
 *  changes can't reintroduce this. */
let cikTickerCache: { at: number; map: Map<string, string> } | null = null;
export async function getCikTickerMap(fetchImpl: FetchFn = fetch as any): Promise<Map<string, string>> {
  if (cikTickerCache && Date.now() - cikTickerCache.at < 24 * 3600_000) return cikTickerCache.map;
  const map = new Map<string, string>();
  try {
    const txt = await fetchText(fetchImpl, "https://www.sec.gov/files/company_tickers.json");
    for (const row of Object.values(JSON.parse(txt)) as any[]) {
      if (row?.cik_str == null || !row?.ticker) continue;
      const cik = String(row.cik_str);
      const existing = map.get(cik);
      if (existing !== undefined && !existing.includes("-")) continue; // keep the already-resolved primary class
      if (existing !== undefined && existing.includes("-") && !String(row.ticker).includes("-")) {
        map.set(cik, row.ticker); // upgrade a suffixed placeholder to the primary class
        continue;
      }
      if (existing === undefined) map.set(cik, row.ticker);
    }
  } catch {}
  // never cache a failed/empty fetch — the next call retries instead of
  // serving 24h of null tickers
  if (map.size > 0) cikTickerCache = { at: Date.now(), map };
  return map;
}

/** Test-only: clears the 24h in-memory cache so each test controls its own
 *  fixture (mirrors satellites.ts's resetSatellitesForTests pattern). */
export function resetCikTickerCacheForTests(): void {
  cikTickerCache = null;
}

/** Fetches recent 8-Ks tagged Item 2.02, resolves each one's Exhibit 99
 *  press release, and returns the extracted text. Sequential with a delay
 *  between filings (SEC fair-access: stay well under 10 req/s even though
 *  each matched filing needs 2 requests — the index page then the
 *  exhibit). Filings with no EX-99-class exhibit are skipped (not returned
 *  with empty text) so every returned record has real language to show. */
export async function fetchLatestEarnings8Ks(
  limit = 15,
  fetchImpl: FetchFn = fetch as any,
  delayMs = 200,
): Promise<Earnings8K[]> {
  const feedUrl = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&company=&dateb=&owner=include&count=100&output=atom";
  const feedXml = await fetchText(fetchImpl, feedUrl);
  // One FILING per accession: a joint 8-K (parent + subsidiaries) is one
  // accession listed once per registrant in the feed. Group first so `limit`
  // counts filings, the exhibit is fetched once, and the archive gets one
  // row carrying its co-registrants (see mergeCoRegistrants). Found on
  // production 2026-11-11: 77 accessions / 114 extra rows across 07..11,
  // every one a parent+subsidiary pair — not the deploy overlap first blamed.
  const byAccession = new Map<string, Filing8KEntry[]>();
  for (const e of parse8KFeed(feedXml).filter(hasItem202)) {
    const g = byAccession.get(e.accession);
    if (g) g.push(e); else byAccession.set(e.accession, [e]);
  }
  const candidates = Array.from(byAccession.values()).slice(0, limit);
  const cikTickers = await getCikTickerMap(fetchImpl);
  const tickerOf = (e: Filing8KEntry) => cikTickers.get(String(parseInt(e.cik, 10))) ?? null;
  const out: Earnings8K[] = [];
  for (const group of candidates) {
    const entry = group.find((e) => tickerOf(e) !== null) ?? group[0];
    const coRegistrants = group.filter((e) => e !== entry && e.cik !== entry.cik)
      .map((e) => ({ cik: e.cik, companyName: e.companyName, ticker: tickerOf(e) }));
    try {
      const indexHtml = await fetchText(fetchImpl, entry.indexUrl);
      const exhibitHref = pickExhibit99Href(indexHtml);
      if (!exhibitHref) continue;
      const exhibitUrl = toExhibitUrl(entry.indexUrl, exhibitHref);
      const exhibitHtml = await fetchText(fetchImpl, exhibitUrl);
      const fullText = htmlToText(exhibitHtml);
      const truncated = fullText.length > MAX_EXHIBIT_TEXT_LEN;
      const text = truncated ? fullText.slice(0, MAX_EXHIBIT_TEXT_LEN) : fullText;
      out.push({
        accession: entry.accession,
        cik: entry.cik,
        companyName: entry.companyName,
        filedAt: entry.filedAt,
        acceptanceDatetime: entry.acceptanceDatetime,
        ticker: tickerOf(entry),
        itemCodes: entry.itemCodes,
        indexUrl: entry.indexUrl,
        exhibitUrl,
        text,
        textLength: fullText.length,
        truncated,
        ...(coRegistrants.length ? { coRegistrants } : {}),
      });
    } catch {
      // one bad filing (missing exhibit, transient 4xx) never kills the batch
    }
    if (delayMs > 0) await new Promise((res) => setTimeout(res, delayMs));
  }
  return out;
}

// ── Archive (COLLECT-EVERYTHING doctrine: EDGAR's full-text search only
// covers filings from 2001 forward and doesn't give us the exhibit text
// itself pre-parsed, so accumulating our own extracted-text archive from
// here forward is the free substitute for a paid transcript history
// product — BUILD-FIRST rule #2). Append-only JSONL per UTC day under
// <archive>/earnings8k/, deduped by accession, gzipped after 2 days —
// identical shape to edgarForm4.ts's filings archive. ───────────────────────

const archivedAccessions = new Set<string>();
trackSet("sec8kEarnings.archivedAccessions", archivedAccessions, 50_000); // memory audit 2026-11-03 A0
let seeded = false;

function earnings8kDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "earnings8k");
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

export function archiveEarnings8Ks(filings: Earnings8K[], baseDir?: string, nowMs?: number): number {
  const dir = earnings8kDir(baseDir);
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
    console.error("[datacore] earnings8k archive:", e?.message || e);
    return 0;
  }
}

export function gzipOldEarnings8kDays(baseDir?: string, nowMs?: number): number {
  const dir = earnings8kDir(baseDir);
  const now = nowMs ?? Date.now();
  let n = 0;
  try {
    for (const f of fs.readdirSync(dir)) {
      if (!f.endsWith(".jsonl")) continue;
      const day = f.slice(0, 10);
      if (now - Date.parse(day) < 2 * 86400_000) continue;
      const fp = path.join(dir, f);
      fs.writeFileSync(`${fp}.gz`, zlib.gzipSync(fs.readFileSync(fp)));
      fs.unlinkSync(fp);
      n++;
    }
  } catch {}
  return n;
}

/** Reads archived earnings-language filings for the last N days (newest
 *  first), transparently handling gzipped days — mirrors
 *  edgarForm4.ts's readFilingHistory. */
export function readEarnings8kHistory(days = 30, baseDir?: string, nowMs?: number, maxFilings = 500): Earnings8K[] {
  const dir = earnings8kDir(baseDir);
  const now = nowMs ?? Date.now();
  let out: Earnings8K[] = [];
  for (let d = 0; d < days && out.length < maxFilings; d++) {
    const day = new Date(now - d * 86400_000).toISOString().slice(0, 10);
    const dayRows: Earnings8K[] = [];
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
          const f = JSON.parse(line);
          if (f.accession) dayRows.push(f);
        } catch {}
      }
    }
    // one row per accession across everything read so far: a joint filing's
    // per-registrant lines fold into one row (mergeCoRegistrants), a repeated
    // line of the same filing collapses, first appearance wins
    if (dayRows.length) out = mergeCoRegistrants(out.concat(dayRows));
  }
  out.sort((a, b) => String(b.filedAt || "").localeCompare(String(a.filedAt || "")));
  return out.slice(0, maxFilings);
}

// ── Windowed archive reader (2026-11-11). `readEarnings8kHistory` above
// answers "the newest N filings in the last D days" — right for the cold-
// cache backfill, wrong for a HISTORY consumer: in the Q3 2026 season the
// archive took in 88-157 filings per UTC day, so `?days=90` returned 500
// filings covering 4 trading days (11-05..11-10) while echoing `days: 90`
// back with no truncation marker — a product endpoint (and its keyed v1
// mirror) silently misreporting its own window, and the pre-registered QoQ
// gate-2 test could no longer be fed from a session. This reader is the
// honest version: an explicit [from, to] window of ARCHIVE days, whole days
// included atomically (a day's file is appended as a unit, so a day is
// either fully in the page or not at all), an explicit `truncated` flag,
// and a `next_to` cursor so a consumer can walk a whole season in pages
// with neither gaps nor duplicates. The cap is soft by at most ONE day: if
// the newest day alone exceeds it, that day is still returned whole (the
// page must always make progress). `includeText=false` strips the ~23 KB
// average exhibit text so a season of metadata fits one call. ───────────────

/** Oldest archive day — the poll loop has appended every extracted filing
 *  since this date (COLLECT-EVERYTHING). */
export const EARNINGS8K_ARCHIVE_START = "2026-07-04";
/** Default and maximum filings per page (a 500-filing page with text is
 *  ~12 MB — the pre-existing cap, now reported instead of hidden). */
export const EARNINGS8K_HISTORY_MAX_FILINGS = 500;
/** Legacy `?days=` lookback cap (unchanged from the route's original bound;
 *  deeper reads use from/to + the cursor). */
export const EARNINGS8K_HISTORY_MAX_DAYS = 90;
/** Widest [from, to] span scanned in one request — bounds the day loop even
 *  on a sparse archive (ENOENT per missing day is cheap, but not free). */
export const EARNINGS8K_HISTORY_MAX_SPAN_DAYS = 400;

export interface Earnings8kHistoryWindow {
  /** inclusive UTC archive day, YYYY-MM-DD */
  from: string;
  /** inclusive UTC archive day, YYYY-MM-DD */
  to: string;
  /** page cap (soft by at most one day, see above) */
  maxFilings: number;
  /** false strips `text` from every filing (textLength/truncated/exhibitUrl stay) */
  includeText: boolean;
  /** the legacy `?days=` value when the window came from it (echoed for compat) */
  days: number | null;
}

export interface Earnings8kHistoryPage {
  filings: Array<Earnings8K | Omit<Earnings8K, "text">>;
  window: { from: string; to: string };
  /** archive days read IN FULL into this page (newest = `to`). NOTE: a
   *  filing's `filedAt` can be one day earlier than its archive day (filed
   *  late in the US evening, archived on the next UTC day). */
  covered: { from: string; to: string };
  /** true when older days inside `window` were NOT read because the page cap
   *  would have been exceeded — continue with `?from=<window.from>&to=<next_to>` */
  truncated: boolean;
  next_to: string | null;
  days_scanned: number;
  max_filings: number;
  text_omitted: boolean;
  /** archive lines in the covered days that did not parse as a filing (a
   *  torn line from a kill mid-append) — reported, never silently dropped */
  malformed_lines: number;
}

const DAY_RE = /^\d{4}-\d{2}-\d{2}$/;
const utcDay = (ms: number): string => new Date(ms).toISOString().slice(0, 10);
const dayMs = (day: string): number => Date.parse(`${day}T00:00:00Z`);
const isValidDay = (s: unknown): s is string =>
  typeof s === "string" && DAY_RE.test(s) && !Number.isNaN(dayMs(s)) && utcDay(dayMs(s)) === s;

/** Turns a route's query object into a validated window. Legacy `?days=N`
 *  (1..90, default 30) still works and means [today-(N-1), today]; `from`/
 *  `to` (YYYY-MM-DD UTC archive days) take precedence over `days`; `to` is
 *  clamped to today, `from` to the archive start and to `to`; the span is
 *  bounded to EARNINGS8K_HISTORY_MAX_SPAN_DAYS by raising `from`. `limit`
 *  1..500 (default 500). `text=0|false|no` or `fields=meta` omits text.
 *  Never throws on bad input — a malformed value falls back to its default,
 *  and the effective window is echoed in the response so nothing is silent. */
export function parseEarnings8kHistoryQuery(query: Record<string, unknown>, nowMs = Date.now()): Earnings8kHistoryWindow {
  const today = utcDay(nowMs);
  const str = (k: string): string | undefined => {
    const v = query[k];
    return typeof v === "string" ? v.trim() : Array.isArray(v) && typeof v[0] === "string" ? String(v[0]).trim() : undefined;
  };
  const toRaw = str("to");
  const fromRaw = str("from");
  const daysRaw = str("days");
  let days: number | null = null;
  if (!isValidDay(fromRaw) && !isValidDay(toRaw)) {
    const parsed = parseInt(daysRaw ?? "", 10);
    days = Math.min(EARNINGS8K_HISTORY_MAX_DAYS, Math.max(1, Number.isFinite(parsed) ? parsed : 30));
  }
  let to = isValidDay(toRaw) ? toRaw : today;
  if (dayMs(to) > dayMs(today)) to = today;
  let from: string;
  if (isValidDay(fromRaw)) from = fromRaw;
  else if (days !== null) from = utcDay(dayMs(to) - (days - 1) * 86400_000);
  else from = utcDay(dayMs(to) - 29 * 86400_000);
  if (dayMs(from) < dayMs(EARNINGS8K_ARCHIVE_START)) from = EARNINGS8K_ARCHIVE_START;
  if (dayMs(from) > dayMs(to)) from = to;
  const spanDays = Math.round((dayMs(to) - dayMs(from)) / 86400_000) + 1;
  if (spanDays > EARNINGS8K_HISTORY_MAX_SPAN_DAYS) from = utcDay(dayMs(to) - (EARNINGS8K_HISTORY_MAX_SPAN_DAYS - 1) * 86400_000);
  const limitParsed = parseInt(str("limit") ?? "", 10);
  const maxFilings = Math.min(EARNINGS8K_HISTORY_MAX_FILINGS, Math.max(1, Number.isFinite(limitParsed) ? limitParsed : EARNINGS8K_HISTORY_MAX_FILINGS));
  const text = (str("text") ?? "").toLowerCase();
  const includeText = !(text === "0" || text === "false" || text === "no" || (str("fields") ?? "").toLowerCase() === "meta");
  return { from, to, maxFilings, includeText, days };
}

function readArchiveDay(dir: string, day: string): { rows: Earnings8K[]; malformed: number } {
  const rows: Earnings8K[] = [];
  let malformed = 0;
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
        const f = JSON.parse(line);
        if (f && f.accession) rows.push(f); else malformed++;
      } catch { malformed++; } // a torn line (kill mid-append) is counted, never silently dropped
    }
  }
  return { rows, malformed };
}

/** Drops the exhibit text; everything that says what the text WAS
 *  (textLength, truncated, exhibitUrl) stays so a consumer can fetch it. */
export function omitEarnings8kText(f: Earnings8K): Omit<Earnings8K, "text"> {
  const { text: _text, ...rest } = f;
  return rest;
}

/** Reads the archive over an explicit [from, to] window of UTC archive days,
 *  newest day first, whole days at a time, stopping BEFORE the day that
 *  would push the page over `maxFilings` (except the first day, which is
 *  always included whole). Deduped by accession, sorted newest-first by
 *  filedAt. See the section comment above for why this exists beside
 *  readEarnings8kHistory. */
export function readEarnings8kHistoryWindow(win: Earnings8kHistoryWindow, baseDir?: string): Earnings8kHistoryPage {
  const dir = earnings8kDir(baseDir);
  const out: Earnings8K[] = [];
  const seen = new Set<string>();
  let truncated = false;
  let nextTo: string | null = null;
  let coveredFrom = win.to;
  let daysScanned = 0;
  let malformed = 0;
  const fromMs = dayMs(win.from);
  for (let ms = dayMs(win.to); ms >= fromMs; ms -= 86400_000) {
    const day = utcDay(ms);
    const read = readArchiveDay(dir, day);
    // One row per accession. Rows archived before v1.0.1352 hold a JOINT
    // filing once per registrant (parent + subsidiaries, same accession,
    // different CIK — 77 accessions / 114 extra lines on production, all of
    // them utility/holding-company pairs on earnings-season days; the
    // v1.0.1351 "deploy overlap" diagnosis was wrong): mergeCoRegistrants
    // folds them into the primary registrant's row with `coRegistrants`, so
    // a customer filtering on the parent's ticker still finds the filing —
    // v1.0.1351's first-seen dedup kept whichever subsidiary the feed listed
    // first (Southern Power Co, ticker null, over Southern Co "SO"). Rows
    // whose accession an earlier (newer) day already served are dropped, as
    // before. The page cap is measured on these merged rows.
    const rows = mergeCoRegistrants(read.rows).filter((f) => !seen.has(f.accession));
    if (out.length > 0 && out.length + rows.length > win.maxFilings) {
      truncated = true;
      nextTo = day;
      break;
    }
    for (const f of rows) { seen.add(f.accession); out.push(f); }
    malformed += read.malformed;
    coveredFrom = day;
    daysScanned++;
  }
  out.sort((a, b) => String(b.filedAt || "").localeCompare(String(a.filedAt || "")));
  return {
    filings: win.includeText ? out : out.map(omitEarnings8kText),
    window: { from: win.from, to: win.to },
    covered: { from: coveredFrom, to: win.to },
    truncated,
    next_to: nextTo,
    days_scanned: daysScanned,
    max_filings: win.maxFilings,
    text_omitted: !win.includeText,
    malformed_lines: malformed,
  };
}

/** The query string a consumer sends to continue a truncated page — or null
 *  when the page already reached `window.from`. */
export function earnings8kHistoryNextQuery(page: Earnings8kHistoryPage): string | null {
  if (!page.truncated || !page.next_to) return null;
  const parts = [`from=${page.window.from}`, `to=${page.next_to}`];
  if (page.max_filings !== EARNINGS8K_HISTORY_MAX_FILINGS) parts.push(`limit=${page.max_filings}`);
  if (page.text_omitted) parts.push("text=0");
  return `?${parts.join("&")}`;
}

// ── In-memory cache + poll loop (mirrors edgarForm4.ts's boot pattern) ──────

let cache: { at: number; filings: Earnings8K[] } | null = null;
let polling = false;

export function latestEarnings8Ks(): { at: number; filings: Earnings8K[] } | null {
  return cache;
}

/** Reconstructs a cache-shaped filings list from the on-disk earnings8k
 *  archive — used to backfill a cold cache when a boot's live poll throws
 *  (SEC EDGAR transient outage / feed-fetch failure) or comes back with
 *  zero qualifying filings before any cache exists. Same
 *  cold-cache-no-disk-backfill fix already shipped for edgarForm4.ts's
 *  identically-shaped `backfillForm4FromArchive` — this closes the same gap
 *  for the earnings-language root, found by a 2026-09-15 audit of the
 *  ~30 datacore modules that audit hadn't yet checked. */
export function backfillEarnings8kFromArchive(baseDir?: string, nowMs?: number, days = 5, limit = 15): Earnings8K[] {
  return readEarnings8kHistory(days, baseDir, nowMs, limit);
}

export async function refreshEarnings8kCache(limit = 15): Promise<void> {
  try {
    const filings = await fetchLatestEarnings8Ks(limit);
    const next = resolveCacheItems(cache !== null, filings, () => backfillEarnings8kFromArchive());
    if (next) cache = { at: Date.now(), filings: next };
    try { archiveEarnings8Ks(filings); } catch {}
    try { gzipOldEarnings8kDays(); } catch {}
  } catch (e: any) {
    console.error("[datacore] earnings8k refresh:", e?.message || e);
    const next = resolveCacheItems(cache !== null, [], () => backfillEarnings8kFromArchive());
    if (next) cache = { at: Date.now(), filings: next };
  }
}

/** Starts the poll loop once at boot (no lazy-first-request gap, per the
 *  KNOWN BROKEN #9 lesson). No API key required. */
export function bootEarnings8kPoll(intervalMs = 15 * 60_000): void {
  if (polling) return;
  polling = true;
  refreshEarnings8kCache();
  setInterval(() => { refreshEarnings8kCache(); }, intervalMs).unref?.();
}

/** Test-only reset — `cache`/`polling` are module-singleton state (same
 *  problem edgarForm4.ts's `_resetForm4CacheForTests` solves for its own
 *  module), so a test exercising the cold-cache backfill path must be able
 *  to force `cache` back to null rather than rely on file execution order. */
export function _resetEarnings8kCacheForTests(): void {
  cache = null;
  polling = false;
}
