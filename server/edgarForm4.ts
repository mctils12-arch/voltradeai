/**
 * edgarForm4.ts — SEC EDGAR Form 4 (insider transaction) pipeline
 * (SPINOUT-READY DATA LAYER; EDGE DOCTRINE #1 "BUILD DATA, DON'T BUY IT" —
 * SEC EDGAR real-time Form 4 was a standing example in CLAUDE.md; this ships
 * it as the first datacore pipeline beyond the aircraft/vessel feeds).
 *
 * ROOT VALIDATION LADDER gate 1 (DATA — verified against an external truth
 * source) status: PASSED. The parser below was built against two real,
 * live-fetched Form 4 filings (one derivative RSU grant, one multi-owner
 * non-derivative open-market sale) and every extracted field was hand-checked
 * against the filed XML — see edgarForm4.test.ts. Gate 2 (SIGNAL — does
 * insider-buy clustering predict forward returns better than base rate) is
 * NOT yet attempted; this module and its /api/data/insider route surface the
 * feed as a RAW-DATA overlay only (as-filed display, no predictive claim),
 * per datacore/README.md's RAW-vs-SIGNAL boundary rule. See
 * research/open_questions.md for the gate-2 hypothesis.
 *
 * Boundary: no imports from trading logic (bot_engine.py, system_config.py,
 * strategies/, server/bot.ts) — pure fetch + parse + in-memory cache, same
 * shape as vesselStream.ts / providerCompliance.ts.
 *
 * No API key required — SEC EDGAR asks only for a descriptive User-Agent
 * (fair-access policy: https://www.sec.gov/os/webmaster-faq#developers).
 */

export interface Form4Owner {
  cik: string;
  name: string;
  isDirector: boolean;
  isOfficer: boolean;
  isTenPercentOwner: boolean;
  officerTitle: string | null;
}

export type Form4TransactionKind =
  | "open_market_buy"     // code P — the informative "insider buy" case
  | "open_market_sale"    // code S
  | "award_grant"         // code A — RSU/option grant, not a discretionary buy
  | "option_exercise"     // code M
  | "gift"                // code G
  | "tax_withholding"     // code F
  | "other";

export interface Form4Transaction {
  table: "derivative" | "nonDerivative";
  securityTitle: string | null;
  transactionDate: string | null;
  transactionCode: string | null;
  kind: Form4TransactionKind;
  shares: number | null;
  pricePerShare: number | null;
  acquiredDisposedCode: "A" | "D" | null;
  sharesOwnedAfter: number | null;
}

export interface ParsedForm4 {
  issuerCik: string | null;
  issuerName: string | null;
  issuerTradingSymbol: string | null;
  periodOfReport: string | null;
  owners: Form4Owner[];
  transactions: Form4Transaction[];
}

export interface Form4Filing extends ParsedForm4 {
  accession: string;
  filedAt: string | null;
  cik: string;
  indexUrl: string;
}

// ── Minimal dependency-free XML tag extraction ──────────────────────────────
// SEC's ownership-document schema is flat and predictable; a full XML parser
// is unnecessary weight for it. Every lookup scopes to the enclosing block
// text first so repeated tag names (e.g. <value>) never leak across siblings.

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

// Nested `<x><value>1</value></x>`-style fields — SEC wraps nearly every
// leaf scalar in its own tag plus a <value> (and often a <footnoteId/>).
function nestedValue(block: string, tag: string): string | null {
  const inner = tagContent(block, tag);
  if (inner == null) return null;
  const v = tagContent(inner, "value");
  return v ?? (inner.trim() || null);
}

function toNum(s: string | null): number | null {
  if (s == null) return null;
  const n = parseFloat(s);
  return Number.isFinite(n) ? n : null;
}

function toBool01(s: string | null): boolean {
  return s === "1" || s?.toLowerCase() === "true";
}

export function classifyTransactionCode(code: string | null): Form4TransactionKind {
  switch (code) {
    case "P": return "open_market_buy";
    case "S": return "open_market_sale";
    case "A": return "award_grant";
    case "M": return "option_exercise";
    case "G": return "gift";
    case "F": return "tax_withholding";
    default: return "other";
  }
}

function parseOwner(block: string): Form4Owner {
  const idBlock = tagContent(block, "reportingOwnerId") || "";
  const relBlock = tagContent(block, "reportingOwnerRelationship") || "";
  return {
    cik: (tagContent(idBlock, "rptOwnerCik") || "").trim(),
    name: (tagContent(idBlock, "rptOwnerName") || "").trim(),
    isDirector: toBool01(tagContent(relBlock, "isDirector")),
    isOfficer: toBool01(tagContent(relBlock, "isOfficer")),
    isTenPercentOwner: toBool01(tagContent(relBlock, "isTenPercentOwner")),
    officerTitle: tagContent(relBlock, "officerTitle"),
  };
}

function parseTransaction(block: string, table: "derivative" | "nonDerivative"): Form4Transaction {
  const codingBlock = tagContent(block, "transactionCoding") || "";
  const amountsBlock = tagContent(block, "transactionAmounts") || "";
  const postBlock = tagContent(block, "postTransactionAmounts") || "";
  const code = tagContent(codingBlock, "transactionCode");
  const acq = nestedValue(amountsBlock, "transactionAcquiredDisposedCode");
  return {
    table,
    securityTitle: nestedValue(block, "securityTitle"),
    transactionDate: nestedValue(block, "transactionDate"),
    transactionCode: code,
    kind: classifyTransactionCode(code),
    shares: toNum(nestedValue(amountsBlock, "transactionShares")),
    pricePerShare: toNum(nestedValue(amountsBlock, "transactionPricePerShare")),
    acquiredDisposedCode: acq === "A" || acq === "D" ? acq : null,
    sharesOwnedAfter: toNum(nestedValue(postBlock, "sharesOwnedFollowingTransaction")),
  };
}

/** Parses one <ownershipDocument> XML body (a single Form 4 filing). */
export function parseForm4Xml(xml: string): ParsedForm4 {
  const issuerBlock = tagContent(xml, "issuer") || "";
  const owners = allBlocks(xml, "reportingOwner").map(parseOwner);
  const nonDerivBlock = tagContent(xml, "nonDerivativeTable");
  const derivBlock = tagContent(xml, "derivativeTable");
  const transactions: Form4Transaction[] = [
    ...(nonDerivBlock ? allBlocks(nonDerivBlock, "nonDerivativeTransaction").map((b) => parseTransaction(b, "nonDerivative")) : []),
    ...(derivBlock ? allBlocks(derivBlock, "derivativeTransaction").map((b) => parseTransaction(b, "derivative")) : []),
  ];
  return {
    issuerCik: tagContent(issuerBlock, "issuerCik"),
    issuerName: tagContent(issuerBlock, "issuerName"),
    issuerTradingSymbol: tagContent(issuerBlock, "issuerTradingSymbol"),
    periodOfReport: tagContent(xml, "periodOfReport"),
    owners,
    transactions,
  };
}

// ── Live filing feed (fetch layer; injectable for tests) ───────────────────

export interface FilingIndexEntry {
  accession: string;
  cik: string;
  title: string;
  filedAt: string | null;
  indexUrl: string;
}

const SEC_UA = { "User-Agent": "voltradeai-datacore/1.0 (research@voltradeai.com)" };
type FetchFn = (url: string, init?: any) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;

/** Parses the EDGAR "getcurrent" Atom feed for form type 4, deduped by
 *  accession number — the feed lists one entry per filer/issuer pair on the
 *  SAME accession, which would otherwise double-count every filing. */
export function parseFilingFeed(atomXml: string): FilingIndexEntry[] {
  const seen = new Set<string>();
  const out: FilingIndexEntry[] = [];
  for (const entryBlock of allBlocks(atomXml, "entry")) {
    const title = (tagContent(entryBlock, "title") || "").trim();
    const linkMatch = entryBlock.match(/<link[^>]*href="([^"]+)"/i);
    const indexUrl = linkMatch ? linkMatch[1] : null;
    const summary = tagContent(entryBlock, "summary") || "";
    const accMatch = summary.match(/AccNo:&lt;\/b&gt;\s*([0-9-]+)/i) || summary.match(/AccNo:<\/b>\s*([0-9-]+)/i);
    const filedMatch = summary.match(/Filed:&lt;\/b&gt;\s*([0-9-]+)/i) || summary.match(/Filed:<\/b>\s*([0-9-]+)/i);
    const cikMatch = title.match(/\((\d{10}|\d+)\)/);
    if (!indexUrl || !accMatch) continue;
    const accession = accMatch[1];
    if (seen.has(accession)) continue;
    seen.add(accession);
    out.push({
      accession,
      cik: cikMatch ? cikMatch[1] : "",
      title,
      filedAt: filedMatch ? filedMatch[1] : null,
      indexUrl,
    });
  }
  return out;
}

/** Picks the ownership XML document out of a filing's directory listing. */
export function pickOwnershipXmlName(indexJson: any): string | null {
  const items: any[] = indexJson?.directory?.item || [];
  const xmlItems = items.filter((it) => /\.xml$/i.test(it.name || ""));
  if (xmlItems.length === 0) return null;
  const preferred = xmlItems.find((it) => /ownership/i.test(it.name));
  return (preferred || xmlItems[0]).name;
}

async function fetchText(fetchImpl: FetchFn, url: string): Promise<string> {
  const r = await fetchImpl(url, { headers: SEC_UA, signal: AbortSignal.timeout(12000) as any });
  if (!r.ok) throw new Error(`${url} -> ${r.status}`);
  return r.text();
}

/** Fetches the N most recent Form 4 filings, parsed. Sequential with a small
 *  delay between filings (SEC fair-access: keep well under 10 req/s even
 *  though each filing needs 2 requests — index.json then the XML). */
export async function fetchLatestForm4Filings(
  limit: number,
  fetchImpl: FetchFn = fetch as any,
  delayMs = 150,
): Promise<Form4Filing[]> {
  const feedUrl = `https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=4&company=&dateb=&owner=include&count=${Math.min(100, limit * 3)}&output=atom`;
  const feedXml = await fetchText(fetchImpl, feedUrl);
  const entries = parseFilingFeed(feedXml).slice(0, limit);
  const out: Form4Filing[] = [];
  for (const entry of entries) {
    try {
      const base = entry.indexUrl.replace(/[^/]+$/, "");
      const indexJson = JSON.parse(await fetchText(fetchImpl, `${base}index.json`));
      const xmlName = pickOwnershipXmlName(indexJson);
      if (!xmlName) continue;
      const xml = await fetchText(fetchImpl, `${base}${xmlName}`);
      const parsed = parseForm4Xml(xml);
      out.push({ ...parsed, accession: entry.accession, filedAt: entry.filedAt, cik: entry.cik, indexUrl: entry.indexUrl });
    } catch {
      // one bad filing (malformed doc, transient 4xx) never kills the batch
    }
    if (delayMs > 0) await new Promise((res) => setTimeout(res, delayMs));
  }
  return out;
}

// ── Filings archive (COLLECT-EVERYTHING doctrine, human directive
// 2026-07-04: accumulate history, never just display latest). Append-only
// JSONL per UTC day under <archive>/filings/, deduped by accession; days
// older than 2 are gzipped in place. Same volume as the position archive
// so archiveStats/volume-watch cover it. ──────────────────────────────────
import fs from "fs";
import path from "path";
import zlib from "zlib";
import { archiveBaseDir } from "./datacoreArchive";

const archivedAccessions = new Set<string>();

function filingsDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "filings");
}

/** Loads accession ids already archived today/yesterday so restarts don't
 *  duplicate rows (append-only files are never rewritten). */
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

let seeded = false;
export function archiveFilings(filings: Form4Filing[], baseDir?: string, nowMs?: number): number {
  const dir = filingsDir(baseDir);
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
    console.error("[datacore] filings archive:", e?.message || e);
    return 0;
  }
}

export function gzipOldFilingDays(baseDir?: string, nowMs?: number): number {
  const dir = filingsDir(baseDir);
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

/** Reads archived filings for the last N days (newest first), transparently
 *  handling gzipped days. Powers the /data/filings history view. */
export function readFilingHistory(days = 30, baseDir?: string, nowMs?: number, maxFilings = 2000): Form4Filing[] {
  const dir = filingsDir(baseDir);
  const now = nowMs ?? Date.now();
  const out: Form4Filing[] = [];
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
          const f = JSON.parse(line);
          if (f.accession && !seen.has(f.accession)) { seen.add(f.accession); out.push(f); }
        } catch {}
      }
    }
  }
  out.sort((a, b) => String(b.filedAt || "").localeCompare(String(a.filedAt || "")));
  return out.slice(0, maxFilings);
}

// ── In-memory cache + poll loop (mirrors vesselStream.ts's boot pattern) ───

let cache: { at: number; filings: Form4Filing[] } | null = null;
let polling = false;

export function latestForm4Filings(): { at: number; filings: Form4Filing[] } | null {
  return cache;
}

export async function refreshForm4Cache(limit = 25): Promise<void> {
  try {
    const filings = await fetchLatestForm4Filings(limit);
    if (filings.length > 0 || !cache) cache = { at: Date.now(), filings };
    try { archiveFilings(filings); } catch {}
    try { gzipOldFilingDays(); } catch {}
  } catch (e: any) {
    console.error("[datacore] edgarForm4 refresh:", e?.message || e);
  }
}

/** Starts the poll loop once at boot (no lazy-first-request gap, per the
 *  KNOWN BROKEN #9 lesson: vessels went cold until someone opened the map). */
export function bootForm4Poll(intervalMs = 15 * 60_000): void {
  if (polling) return;
  polling = true;
  refreshForm4Cache();
  setInterval(() => { refreshForm4Cache(); }, intervalMs).unref?.();
}
