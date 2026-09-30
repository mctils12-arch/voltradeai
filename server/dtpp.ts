// FAA d-TPP (Digital Terminal Procedures Publication) — the chart half of
// "plate on the map": the cycle metafile (which PDF is which chart at which
// airport), chart <-> CIFP procedure matching, and a bounded PDF cache.
//
// Source (FAA, public domain, 28-day AIRAC cycle, same idents as CIFP):
//   https://aeronav.faa.gov/d-tpp/<cycle>/xml_data/d-tpp_Metafile.xml  (~16 MB)
//   https://aeronav.faa.gov/d-tpp/<cycle>/<pdf_name>
//
// CACHE POLICY: PDFs go to R2 (key plates/<cycle>/<pdf>) when R2 is
// configured, else to a bounded LRU directory under the container's /tmp —
// never the nearly-full /data volume. Only PDF names that the current
// metafile lists for the requested airport are ever fetched (no open proxy).

import fs from "fs";
import os from "os";
import path from "path";
import { cycleAt, type AiracCycle, type ProcKind, parseApproachIdent } from "./cifp";
import { createR2Client, r2ConfigFromEnv, type R2Client } from "./r2Client";

export const dtppMetafileUrl = (cycle: string) => `https://aeronav.faa.gov/d-tpp/${cycle}/xml_data/d-tpp_Metafile.xml`;
export const dtppPdfUrl = (cycle: string, pdf: string) => `https://aeronav.faa.gov/d-tpp/${cycle}/${pdf}`;
const UA = { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)" };

export interface DtppChart {
  code: string;      // IAP | STR | DP | APD | MIN | HOT | ODP | DAU | LAH
  name: string;
  pdf: string;
  amdt: string | null;
  amdtDate: string | null;
  /** "A" added / "C" changed / "D" deleted this cycle */
  userAction: string | null;
}
export interface DtppAirport { icao: string | null; faa: string; name: string; charts: DtppChart[] }
export interface DtppIndex {
  cycle: string;
  from: string | null;
  to: string | null;
  airports: Map<string, DtppAirport>;
}

const xmlUnescape = (s: string) => s.replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"').replace(/&apos;/g, "'").replace(/&amp;/g, "&");
const attrOf = (tag: string, name: string) => {
  const m = new RegExp(`\\b${name}="([^"]*)"`).exec(tag);
  return m ? xmlUnescape(m[1]) : null;
};
const child = (rec: string, name: string) => {
  const m = new RegExp(`<${name}>([\\s\\S]*?)</${name}>`).exec(rec);
  const v = m ? xmlUnescape(m[1].trim()) : "";
  return v || null;
};

/** Parse the metafile. Airports are keyed by BOTH their ICAO ident (when
 *  published) and their FAA ident, so /KAUS and /AUS resolve alike. */
export function parseDtppMetafile(xml: string): DtppIndex {
  const root = /<digital_tpp\b[^>]*>/.exec(xml)?.[0] ?? "";
  const idx: DtppIndex = {
    cycle: attrOf(root, "cycle") ?? "",
    from: attrOf(root, "from_edate"),
    to: attrOf(root, "to_edate"),
    airports: new Map(),
  };
  let pos = 0;
  for (;;) {
    const a = xml.indexOf("<airport_name", pos);
    if (a < 0) break;
    const tagEnd = xml.indexOf(">", a);
    const end = xml.indexOf("</airport_name>", tagEnd);
    if (tagEnd < 0 || end < 0) break;
    const tag = xml.slice(a, tagEnd + 1);
    const body = xml.slice(tagEnd + 1, end);
    pos = end + 15;
    const faa = attrOf(tag, "apt_ident");
    if (!faa) continue;
    const icao = attrOf(tag, "icao_ident");
    const charts: DtppChart[] = [];
    let rp = 0;
    for (;;) {
      const rs = body.indexOf("<record>", rp);
      if (rs < 0) break;
      const re = body.indexOf("</record>", rs);
      if (re < 0) break;
      const rec = body.slice(rs, re);
      rp = re + 9;
      const pdf = child(rec, "pdf_name");
      const code = child(rec, "chart_code");
      const name = child(rec, "chart_name");
      if (!pdf || !code || !name) continue;
      charts.push({ code, name, pdf, amdt: child(rec, "amdtnum"), amdtDate: child(rec, "amdtdate"), userAction: child(rec, "useraction") });
    }
    const ap: DtppAirport = { icao: icao || null, faa, name: attrOf(tag, "ID") ?? faa, charts };
    idx.airports.set(faa.toUpperCase(), ap);
    if (icao) idx.airports.set(icao.toUpperCase(), ap);
  }
  return idx;
}

// ── chart <-> procedure matching ────────────────────────────────────────────

const NUMBER_WORDS: Record<string, string> = {
  ZERO: "0", ONE: "1", TWO: "2", THREE: "3", FOUR: "4", FIVE: "5", SIX: "6", SEVEN: "7", EIGHT: "8", NINE: "9",
};

/** "AUSTIN SEVEN" -> { base: "AUSTIN", digit: "7" }; "DXEEE THREE (RNAV)" -> DXEEE/3 */
export function parseRouteChartName(name: string): { base: string; digit: string | null; cont: boolean } {
  const cont = /,\s*CONT\.\s*\d+/.test(name);
  const clean = name.replace(/,\s*CONT\.\s*\d+/, "").replace(/\(.*?\)/g, " ").trim();
  const words = clean.split(/\s+/);
  let digit: string | null = null;
  const baseWords: string[] = [];
  for (const w of words) {
    if (NUMBER_WORDS[w] != null && digit == null && baseWords.length) digit = NUMBER_WORDS[w];
    else if (/^\d$/.test(w) && digit == null && baseWords.length) digit = w;
    else if (digit == null) baseWords.push(w);
  }
  return { base: baseWords.join(" "), digit, cont };
}

/**
 * Chart(s) for a CIFP SID/STAR ident ("AUS7", "BLEWE5", "CWK8"): chart kind
 * must match (DP / STR), the version digit must match, and the chart's base
 * name must equal the ident's letters, start with them (AUS -> AUSTIN), or
 * be the NAME of the navaid those letters identify (CWK -> CENTEX).
 * Continuation pages ("…, CONT.1") ride along as extra pages.
 */
export function matchRouteCharts(kind: "SID" | "STAR", id: string, charts: DtppChart[], navaidName: (ident: string) => string | null): DtppChart[] {
  const m = /^([A-Z]+)(\d)$/.exec(id);
  if (!m) return [];
  const [, letters, digit] = m;
  const code = kind === "SID" ? "DP" : "STR";
  const nav = navaidName(letters)?.toUpperCase() ?? null;
  const hits = charts.filter((c) => {
    if (c.code !== code) return false;
    const p = parseRouteChartName(c.name);
    if (p.digit !== digit) return false;
    const base = p.base.toUpperCase();
    const first = base.split(" ")[0];
    return first === letters || (letters.length >= 3 && first.startsWith(letters)) || (!!nav && (base === nav || first === nav.split(" ")[0]));
  });
  return hits.sort((a, b) => Number(parseRouteChartName(a.name).cont) - Number(parseRouteChartName(b.name).cont));
}

const TYPE_KEYWORDS: Record<string, RegExp> = {
  I: /\bILS\b/, L: /\bLOC\b/, B: /\bBC\b/, R: /RNAV \(GPS\)|\bGPS\b/, H: /RNAV \(RNP\)/, V: /\bVOR\b/, D: /\bVOR\/DME\b|\bVOR\b/,
  S: /\bVOR\b/, N: /\bNDB\b/, Q: /\bNDB\b/, P: /\bGPS\b/, X: /\bLDA\b/, U: /\bSDF\b/, G: /\bIGS\b/, J: /\bGLS\b/, T: /\bTACAN\b/,
};

/** Best IAP chart for a CIFP approach ident ("I18L", "R18LY", "H36RZ").
 *  Runway and variant letter must match; the type keyword must appear.
 *  Special-authorization variants ("(SA CAT I)", "(CAT II - III)") lose to
 *  the plain chart. */
export function matchApproachChart(id: string, charts: DtppChart[]): DtppChart | null {
  const a = parseApproachIdent(id);
  const kw = TYPE_KEYWORDS[a.typeCode];
  let best: { c: DtppChart; score: number } | null = null;
  for (const c of charts) {
    if (c.code !== "IAP") continue;
    const n = c.name.toUpperCase();
    if (kw && !kw.test(n)) continue;
    if (a.typeCode === "V" && /VOR\/DME/.test(n) && !/\bVOR\b(?!\/)/.test(n.replace("VOR/DME", ""))) continue;
    const rwy = /RWY\s+(\d{2}[LRC]?)/.exec(n)?.[1] ?? null;
    if (a.runway && rwy !== a.runway.replace(/B$/, "")) continue;
    if (!a.runway && rwy) continue;
    const variant = /\s([W-Z])\s+RWY/.exec(n)?.[1] ?? /-([A-Z])$/.exec(n)?.[1] ?? null;
    if ((variant ?? null) !== (a.variant ?? null)) continue;
    let score = 100 - n.length / 10;
    if (/\bCAT\b/.test(n)) score -= 50;
    if (/,\s*CONT\./.test(n)) score -= 80;
    if (!best || score > best.score) best = { c, score };
  }
  return best?.c ?? null;
}

export function chartsForProcedure(kind: ProcKind, id: string, charts: DtppChart[], navaidName: (ident: string) => string | null): DtppChart[] {
  if (kind === "IAP") { const c = matchApproachChart(id, charts); return c ? [c] : []; }
  return matchRouteCharts(kind, id, charts, navaidName);
}

// ── metafile store ──────────────────────────────────────────────────────────

export interface DtppStoreDeps { fetchImpl?: typeof fetch; now?: () => number; timeoutMs?: number }
export const DTPP_FAILURE_BACKOFF_MS = 10 * 60_000;

export class DtppStore {
  private idx: DtppIndex | null = null;
  private idxCycle: AiracCycle | null = null;
  private loading: Promise<DtppIndex> | null = null;
  private lastError: { at: number; message: string } | null = null;
  private readonly fetchImpl: typeof fetch;
  private readonly now: () => number;
  private readonly timeoutMs: number;

  constructor(deps: DtppStoreDeps = {}) {
    this.fetchImpl = deps.fetchImpl ?? ((...a: Parameters<typeof fetch>) => fetch(...a));
    this.now = deps.now ?? (() => Date.now());
    this.timeoutMs = deps.timeoutMs ?? 90_000;
  }

  status() {
    return { loaded: !!this.idx, cycle: this.idxCycle?.ident ?? null, from: this.idx?.from ?? null, to: this.idx?.to ?? null, lastError: this.lastError };
  }

  async get(): Promise<{ idx: DtppIndex; cycle: AiracCycle }> {
    const want = cycleAt(this.now());
    if (this.idx && this.idxCycle?.ident === want.ident) return { idx: this.idx, cycle: this.idxCycle };
    const backingOff = this.lastError && this.now() - this.lastError.at < DTPP_FAILURE_BACKOFF_MS;
    if (backingOff && this.idx && this.idxCycle) return { idx: this.idx, cycle: this.idxCycle };
    if (backingOff) throw new Error(`d-TPP metafile unavailable: ${this.lastError!.message}`);
    if (!this.loading) this.loading = this.load(want).finally(() => { this.loading = null; });
    const idx = await this.loading;
    return { idx, cycle: this.idxCycle! };
  }

  private async load(want: AiracCycle): Promise<DtppIndex> {
    const errors: string[] = [];
    for (const c of [want, cycleAt(want.effectiveMs, -1)]) {
      const ac = new AbortController();
      const timer = setTimeout(() => ac.abort(), this.timeoutMs);
      try {
        const r = await this.fetchImpl(dtppMetafileUrl(c.ident), { signal: ac.signal, headers: UA });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const idx = parseDtppMetafile(await r.text());
        if (!idx.airports.size) throw new Error("metafile parsed to zero airports");
        this.idx = idx; this.idxCycle = c; this.lastError = null;
        return idx;
      } catch (e: unknown) {
        errors.push(`${c.ident}: ${e instanceof Error ? e.message : String(e)}`);
      } finally {
        clearTimeout(timer);
      }
    }
    this.lastError = { at: this.now(), message: errors.join("; ") };
    console.error(`[dtpp] metafile load failed — ${this.lastError.message}`);
    if (this.idx) return this.idx;
    throw new Error(`d-TPP metafile unavailable: ${this.lastError.message}`);
  }

  /** test hook */
  _setIndex(idx: DtppIndex, cycle: AiracCycle): void { this.idx = idx; this.idxCycle = cycle; }
}

// ── PDF cache (R2 when configured, else bounded /tmp LRU) ───────────────────

export const PLATE_PDF_RE = /^[A-Z0-9_]{2,40}\.PDF$/i;
export const PLATE_MAX_BYTES = 8 * 1024 * 1024;
export const PLATE_TMP_MAX_BYTES = 256 * 1024 * 1024;
export const PLATE_TMP_MAX_FILES = 1500;

export interface PlateCacheDeps {
  fetchImpl?: typeof fetch;
  r2?: R2Client | null;
  dir?: string;
  timeoutMs?: number;
  maxBytes?: number;
  maxFiles?: number;
}
export interface PlateFetch { body: Buffer; source: "r2" | "tmp" | "faa" }

export class PlateCache {
  private readonly fetchImpl: typeof fetch;
  private readonly r2: R2Client | null;
  readonly dir: string;
  private readonly timeoutMs: number;
  private readonly maxBytes: number;
  private readonly maxFiles: number;
  private inflight = new Map<string, Promise<PlateFetch>>();
  readonly counters = { r2Hits: 0, tmpHits: 0, upstream: 0, upstreamErrors: 0, r2PutErrors: 0, evicted: 0 };

  constructor(deps: PlateCacheDeps = {}) {
    this.fetchImpl = deps.fetchImpl ?? ((...a: Parameters<typeof fetch>) => fetch(...a));
    const r2 = deps.r2 !== undefined ? deps.r2 : createR2Client(r2ConfigFromEnv());
    this.r2 = r2 && r2.configured ? r2 : null;
    this.dir = deps.dir ?? path.join(os.tmpdir(), "voltrade_plates");
    this.timeoutMs = deps.timeoutMs ?? 30_000;
    this.maxBytes = deps.maxBytes ?? PLATE_TMP_MAX_BYTES;
    this.maxFiles = deps.maxFiles ?? PLATE_TMP_MAX_FILES;
  }

  get backend(): "r2" | "tmp" { return this.r2 ? "r2" : "tmp"; }
  static key(cycle: string, pdf: string): string { return `plates/${cycle}/${pdf.toUpperCase()}`; }

  get(cycle: string, pdfRaw: string): Promise<PlateFetch> {
    const pdf = pdfRaw.toUpperCase();
    if (!PLATE_PDF_RE.test(pdf) || !/^\d{4}$/.test(cycle)) return Promise.reject(new Error("bad plate name"));
    const k = PlateCache.key(cycle, pdf);
    const hit = this.inflight.get(k);
    if (hit) return hit;
    const p = this.load(cycle, pdf, k).finally(() => { this.inflight.delete(k); });
    this.inflight.set(k, p);
    return p;
  }

  private tmpPath(cycle: string, pdf: string): string { return path.join(this.dir, `${cycle}_${pdf}`); }

  private async load(cycle: string, pdf: string, key: string): Promise<PlateFetch> {
    if (this.r2) {
      const r = await this.r2.getObject(key, { maxBytes: PLATE_MAX_BYTES });
      if (r.ok && r.body && isPdf(r.body)) { this.counters.r2Hits++; return { body: r.body, source: "r2" }; }
    } else {
      const f = this.tmpPath(cycle, pdf);
      try {
        const body = await fs.promises.readFile(f);
        if (isPdf(body)) {
          this.counters.tmpHits++;
          const t = new Date();
          fs.promises.utimes(f, t, t).catch((e: unknown) => console.warn("[plates] touch:", e instanceof Error ? e.message : e));
          return { body, source: "tmp" };
        }
      } catch (e: unknown) {
        if ((e as NodeJS.ErrnoException)?.code !== "ENOENT") console.warn("[plates] tmp read:", e instanceof Error ? e.message : e);
      }
    }
    const body = await this.fetchUpstream(cycle, pdf);
    if (this.r2) {
      const put = await this.r2.putObject(key, body, "application/pdf");
      if (!put.ok) { this.counters.r2PutErrors++; console.warn(`[plates] R2 put ${key}: ${put.error}`); }
    } else {
      await this.writeTmp(cycle, pdf, body);
    }
    return { body, source: "faa" };
  }

  private async fetchUpstream(cycle: string, pdf: string): Promise<Buffer> {
    const ac = new AbortController();
    const timer = setTimeout(() => ac.abort(), this.timeoutMs);
    try {
      this.counters.upstream++;
      const r = await this.fetchImpl(dtppPdfUrl(cycle, pdf), { signal: ac.signal, headers: UA });
      if (!r.ok) throw new Error(`FAA d-TPP HTTP ${r.status} for ${pdf}`);
      const body = Buffer.from(await r.arrayBuffer());
      if (body.length > PLATE_MAX_BYTES) throw new Error(`plate ${pdf} too large (${body.length} bytes)`);
      if (!isPdf(body)) throw new Error(`FAA returned a non-PDF body for ${pdf}`);
      return body;
    } catch (e: unknown) {
      this.counters.upstreamErrors++;
      throw e;
    } finally {
      clearTimeout(timer);
    }
  }

  private async writeTmp(cycle: string, pdf: string, body: Buffer): Promise<void> {
    try {
      await fs.promises.mkdir(this.dir, { recursive: true });
      const f = this.tmpPath(cycle, pdf);
      await fs.promises.writeFile(`${f}.part`, body);
      await fs.promises.rename(`${f}.part`, f);
      await this.evict();
    } catch (e: unknown) {
      console.warn("[plates] tmp write:", e instanceof Error ? e.message : e); // cache miss next time; the response is unaffected
    }
  }

  /** LRU by mtime: over the byte or file cap, the least recently used go. */
  async evict(): Promise<number> {
    const names = await fs.promises.readdir(this.dir);
    const files: Array<{ f: string; size: number; t: number }> = [];
    for (const n of names) {
      if (n.endsWith(".part")) continue;
      const f = path.join(this.dir, n);
      const st = await fs.promises.stat(f);
      files.push({ f, size: st.size, t: st.mtimeMs });
    }
    files.sort((a, b) => a.t - b.t);
    let total = files.reduce((s, x) => s + x.size, 0);
    let count = files.length;
    let removed = 0;
    for (const x of files) {
      if (total <= this.maxBytes && count <= this.maxFiles) break;
      await fs.promises.rm(x.f, { force: true });
      total -= x.size; count--; removed++;
    }
    this.counters.evicted += removed;
    return removed;
  }
}

export const isPdf = (b: Buffer) => b.length > 5 && b.toString("latin1", 0, 5) === "%PDF-";
