/**
 * archiveLedger.ts — "does this archive already hold this filing?", answered
 * from the archive's own day files instead of from process memory.
 *
 * WHY THIS EXISTS (2026-11-16, research/open_questions.md "the Form 4, 13F
 * and 8-K archives re-append filings they already hold"): the three EDGAR
 * archive writers (edgarForm4.ts, edgar13f.ts, sec8kEarnings.ts) each kept a
 * module-level Set of accessions, rebuilt after a restart from exactly two
 * day files (today's and yesterday's UTC) and cleared outright at 50,000.
 * EDGAR's "current" feed returns its newest N entries whatever their age, so
 * after every restart a filing first archived two or more days earlier and
 * still on the feed's page was "fresh" again and appended under today.
 * Measured live through the public history routes: 13F 2026-09-18..11-16,
 * 776 repeat rows of 8,446 served (9.2%); Form 4 2026-10-27..11-16, 52 of
 * 6,658; 13F archive day 2026-11-08 (a Sunday) was 40 rows, all 40 already
 * in the 2026-11-06 day file.
 *
 * THE RULE: a candidate is fresh only if NO archive day file from its own
 * `filedAt` day through today holds its accession. A filing cannot be
 * archived before it is filed, and `filedAt` is SEC's Eastern date while the
 * archive day is the UTC date of the append (never earlier), so that window
 * is exactly where an earlier copy can live.
 *
 * HOW IT STAYS CHEAP AND BOUNDED:
 *   · day files are read lazily, accession-only, each at most once per
 *     process (a past day's file is never appended to again);
 *   · TODAY's file is the exception: its size is checked before every
 *     filter, and a size this process did not produce (another container
 *     appending during a deploy overlap) triggers one re-read;
 *   · the window is capped at `lookbackDays`. The cap is an I/O bound, NOT an
 *     age policy: a candidate filed before the cap is checked against the
 *     whole capped window, counted (`beyond_lookback`) and, if not found,
 *     still archived — refusing very old feed entries is a separate, filed
 *     question (open_questions.md 2026-11-16, the 24 releases from
 *     2020/2021);
 *   · day sets older than the window are dropped. Dropping can never cause a
 *     repeat: the disk is the truth and a dropped day is simply read again
 *     if a candidate ever needs it. There is no `clear()`.
 *
 * APPEND-ONLY: nothing here writes or rewrites an archive file. Repeats
 * already on disk stay there; the history readers keep their own dedupe.
 *
 * Boundary: fs + zlib only. No imports from trading logic, no network.
 */
import fs from "fs";
import path from "path";
import zlib from "zlib";

/** Days back from today a candidate is checked against. An I/O bound (see
 *  header), chosen to cover the oldest entry ever measured on a feed page in
 *  normal operation (13F, Labor Day 2026: 9 days) plus the longest outage on
 *  record (2026-09-10..16: 6 days) with room to spare. */
export const LEDGER_LOOKBACK_DAYS = 30;

const DAY_MS = 86400_000;
const DAY_RE = /^\d{4}-\d{2}-\d{2}$/;
/** First `"accession":"…"` in a line. A JSON string value can never contain
 *  an unescaped `"`, so this cannot match inside a text field; a value with
 *  an escape in it falls back to JSON.parse. */
const ACCESSION_RE = /"accession":"([^"\\]*)"/;

export function utcDay(ms: number): string {
  return new Date(ms).toISOString().slice(0, 10);
}

/** The accession of one archived JSONL line, or null (blank/corrupt line). */
export function accessionOfLine(line: string): string | null {
  if (!line) return null;
  const m = ACCESSION_RE.exec(line);
  if (m) return m[1] || null;
  try {
    const a = JSON.parse(line)?.accession;
    return typeof a === "string" && a ? a : null;
  } catch {
    return null;
  }
}

/** Every accession in one archive day, from the plain file, the gzipped
 *  file, or both (a crash between "write .gz" and "unlink .jsonl" leaves
 *  both). A missing or unreadable file contributes nothing. */
export function readDayAccessions(dir: string, day: string): Set<string> {
  const out = new Set<string>();
  for (const name of [`${day}.jsonl`, `${day}.jsonl.gz`]) {
    let text: string;
    try {
      const buf = fs.readFileSync(path.join(dir, name));
      text = name.endsWith(".gz") ? zlib.gunzipSync(buf).toString("utf8") : buf.toString("utf8");
    } catch {
      continue;
    }
    for (const line of text.split("\n")) {
      const a = accessionOfLine(line);
      if (a) out.add(a);
    }
  }
  return out;
}

/** A type alias, not an interface, so it fits heapDiag's open GaugeValue. */
export type LedgerStats = {
  /** accessions held in memory across the loaded days */
  entries: number;
  /** day files' accession sets currently held in memory */
  days_loaded: number;
  /** day-file reads since boot (lazy loads + today re-reads) */
  day_reads: number;
  /** today's file re-read because its size was not this process's doing */
  today_rereads: number;
  /** candidates refused because an archive day already holds them */
  repeats_refused: number;
  /** candidates filed before the lookback window (checked against the whole
   *  window; archived if not found there) */
  beyond_lookback: number;
};

interface DirState {
  days: Map<string, Set<string>>;
  /** the UTC day `todaySize` describes */
  today: string | null;
  /** today's plain-file size after this process last read or appended it */
  todaySize: number;
}

export interface LedgerRow {
  accession: string;
  filedAt?: string | null;
}

/** One per archive writer. Keyed by directory inside so tests (and any
 *  future second archive of the same kind) never share state across dirs. */
export class AccessionLedger {
  private dirs = new Map<string, DirState>();
  private counters = { day_reads: 0, today_rereads: 0, repeats_refused: 0, beyond_lookback: 0 };

  constructor(private readonly lookbackDays: number = LEDGER_LOOKBACK_DAYS) {}

  /** Accessions held in memory across all loaded days (heapDiag gauge). */
  get size(): number {
    let n = 0;
    for (const st of this.dirs.values()) for (const s of st.days.values()) n += s.size;
    return n;
  }

  stats(): LedgerStats {
    let days = 0;
    for (const st of this.dirs.values()) days += st.days.size;
    return { entries: this.size, days_loaded: days, ...this.counters };
  }

  private state(dir: string): DirState {
    let st = this.dirs.get(dir);
    if (!st) {
      st = { days: new Map(), today: null, todaySize: -1 };
      this.dirs.set(dir, st);
    }
    return st;
  }

  private fileSize(dir: string, day: string): number {
    try { return fs.statSync(path.join(dir, `${day}.jsonl`)).size; } catch { return 0; }
  }

  private loadDay(dir: string, st: DirState, day: string): Set<string> {
    let set = st.days.get(day);
    if (!set) {
      set = readDayAccessions(dir, day);
      st.days.set(day, set);
      this.counters.day_reads++;
    }
    return set;
  }

  /** Makes today's set match today's file, and drops days outside the window. */
  private sync(dir: string, st: DirState, today: string, firstDay: string): void {
    const size = this.fileSize(dir, today);
    if (st.today !== today) {
      st.today = today;
      st.todaySize = size;
      st.days.delete(today); // lazy-load it fresh below
    } else if (size !== st.todaySize) {
      st.days.delete(today);
      st.todaySize = size;
      this.counters.today_rereads++;
    }
    for (const d of Array.from(st.days.keys())) {
      if (d < firstDay || d > today) st.days.delete(d);
    }
  }

  /**
   * The rows of `rows` this archive does not hold yet, in input order, one
   * per accession. Reads only the day files a candidate's own `filedAt`
   * makes relevant.
   */
  filterFresh<T extends LedgerRow>(dir: string, rows: T[], nowMs: number): T[] {
    const st = this.state(dir);
    const today = utcDay(nowMs);
    const firstDay = utcDay(nowMs - this.lookbackDays * DAY_MS);
    this.sync(dir, st, today, firstDay);
    const fresh: T[] = [];
    const inBatch = new Set<string>();
    for (const row of rows) {
      const acc = row?.accession;
      if (!acc || inBatch.has(acc)) continue;
      inBatch.add(acc);
      const filed = typeof row.filedAt === "string" ? row.filedAt.slice(0, 10) : "";
      let start: string;
      if (!DAY_RE.test(filed) || Number.isNaN(Date.parse(filed))) {
        start = firstDay;                    // no usable date: check the whole window
      } else if (filed < firstDay) {
        start = firstDay;
        this.counters.beyond_lookback++;
      } else if (filed > today) {
        start = today;                       // dated ahead of our clock: today is all that can hold it
      } else {
        start = filed;
      }
      let held = false;
      // newest first: a repeat is most often in the last day or two
      for (let ms = Date.parse(today); !held; ms -= DAY_MS) {
        const day = utcDay(ms);
        if (day < start) break;
        held = this.loadDay(dir, st, day).has(acc);
      }
      if (held) this.counters.repeats_refused++;
      else fresh.push(row);
    }
    return fresh;
  }

  /** Call after `bytes` bytes holding `rows` were appended to
   *  `<dir>/<utcDay(nowMs)>.jsonl`. If the file did not grow by exactly that
   *  much, someone else appended too: the next filter re-reads the day. */
  recordAppended(dir: string, rows: LedgerRow[], nowMs: number, bytes: number): void {
    const st = this.state(dir);
    const today = utcDay(nowMs);
    if (st.today !== today) return; // filterFresh always runs first; nothing to keep in step otherwise
    const set = this.loadDay(dir, st, today);
    for (const r of rows) if (r?.accession) set.add(r.accession);
    st.todaySize = this.fileSize(dir, today) === st.todaySize + bytes ? st.todaySize + bytes : -1;
  }

  _resetForTests(): void {
    this.dirs.clear();
    this.counters = { day_reads: 0, today_rereads: 0, repeats_refused: 0, beyond_lookback: 0 };
  }
}
