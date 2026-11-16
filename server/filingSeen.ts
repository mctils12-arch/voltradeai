/**
 * filingSeen.ts — "does the archive already hold this filing?" for the three
 * EDGAR archive writers (edgarForm4.ts <archive>/filings/, edgar13f.ts
 * <archive>/filings13f/, sec8kEarnings.ts <archive>/earnings8k/).
 *
 * WHY (research/open_questions.md 2026-11-16, measured live): each writer
 * rebuilt its seen set after a restart from exactly two day files, today's
 * and yesterday's. EDGAR's "current" feed returns its newest N entries
 * whatever their age, so every restart (every deploy) re-appended whatever
 * was still on the feed's page and had first been archived two or more days
 * before: 776 of 8,446 13F rows over 60 days (9.2%), 52 of 6,658 Form 4 rows
 * over 21 days. 13F archive day 2026-11-08, a Sunday, is 40 rows that are all
 * already in 2026-11-06. The sets were also clear()ed at 50,000, which opened
 * the same door without a restart.
 *
 * RULE: a candidate is fresh only if no archive day file from the day before
 * its own `filedAt` through today holds its accession. The answer is read
 * from the day files themselves, so it is the same after a restart as before
 * one. Nothing here forgets a day that can still be on the feed's page:
 * memory is bounded by dropping day sets older than the lookback window,
 * never by clearing.
 *
 * LOOKBACK (FILING_SEEN_LOOKBACK_DAYS = 21): the longest a filing was
 * measured staying on a feed page is 9 days (13F, first archived 2026-08-29,
 * re-appended 2026-09-07 over Labor Day). 21 is twice that plus margin. A
 * candidate filed before the window, or with no readable filed date, is
 * checked against the whole window and counted (`before_lookback`); it is
 * still archived when the window does not hold it. Refusing very old feed
 * entries is a separate, open rule (open_questions.md 2026-11-16, the 24
 * releases filed in 2020/2021) and is NOT decided here.
 *
 * COST: day files are read lazily, each at most once per process, as a byte
 * scan for the writer's own `"accession":"…"` key (no JSON.parse, no string
 * copy of the file). A boot whose feed entries were all filed in the last
 * few days reads only those few days. Reads are synchronous like the
 * writers; the largest measured day file class (13F deadline week, ~900
 * filings) scans in well under a second.
 *
 * DIRECTION OF ERROR: an unreadable day file is counted and treated as
 * holding nothing, so the worst case is a repeat row (what happened before),
 * never a filing that is not archived.
 *
 * APPEND-ONLY: no day file is rewritten. Repeats already on disk stay.
 *
 * Boundary: fs + zlib only; no trading logic (SPINOUT-READY DATA LAYER).
 */
import fs from "fs";
import path from "path";
import zlib from "zlib";

export const FILING_SEEN_LOOKBACK_DAYS = 21;

const DAY_MS = 86400_000;
const DAY_RE = /^\d{4}-\d{2}-\d{2}/;
const KEY = Buffer.from('"accession":"');
const QUOTE = 0x22;
/** EDGAR accessions are 20 characters; anything far longer is not one. */
const MAX_ACCESSION_BYTES = 64;

const dayOf = (ms: number): string => new Date(ms).toISOString().slice(0, 10);

/** Adds every `"accession":"<id>"` value in a JSONL buffer to `into`.
 *  The writers serialize their rows with JSON.stringify, so the key always
 *  has this exact byte form; the same text inside a string VALUE is escaped
 *  (`\"accession\":\"`) and cannot match. Row types hold `accession` at the
 *  top level only (filingSeen.test.ts pins that on each writer's row). */
export function accessionsInBuffer(buf: Buffer, into: Set<string>): void {
  let i = 0;
  for (;;) {
    i = buf.indexOf(KEY, i);
    if (i === -1) return;
    const start = i + KEY.length;
    const end = buf.indexOf(QUOTE, start);
    if (end === -1) return;
    if (end > start && end - start <= MAX_ACCESSION_BYTES) into.add(buf.toString("latin1", start, end));
    i = end + 1;
  }
}

export interface FilingSeenPoll {
  /** rows offered by the poll that carry an accession */
  candidates: number;
  /** rows the archive did not hold: these are appended */
  fresh: number;
  /** held by today's or yesterday's day file (the old two-day check saw these) */
  held_recent: number;
  /** held only by a day file two or more days old (the old check re-appended
   *  these after a restart) */
  held_older_day: number;
  /** filed before the lookback window, or no readable filed date: checked
   *  against the whole window */
  before_lookback: number;
}

export interface FilingSeenStats {
  [k: string]: unknown;
  lookback_days: number;
  day_sets: number;
  entries: number;
  unreadable_day_files: number;
  last_poll: FilingSeenPoll | null;
}

export interface FilingSeen {
  /** The subset of `filings` no archive day in the window holds. Rows without
   *  an accession are dropped, as the writers always did. */
  selectFresh<T extends { accession: string; filedAt?: string | null }>(dir: string, filings: T[], nowMs: number): T[];
  /** Record rows this process has just appended under `nowMs`'s UTC day. */
  noteAppended(dir: string, accessions: Iterable<string>, nowMs: number): void;
  /** accessions held in memory (the /api/health cache gauge's `entries`) */
  readonly size: number;
  stats(): FilingSeenStats;
  _resetForTests(): void;
}

/** One instance per archive. It is `Sized` with `stats()`, so each writer
 *  keeps the /api/health cache gauge it already had through
 *  `trackSet(name, seen)` (entries = accessions held in memory, plus the last
 *  poll's counts). */
export function createFilingSeen(lookbackDays = FILING_SEEN_LOOKBACK_DAYS): FilingSeen {
  // dir -> day -> accessions in that day's file(s). Production has one dir.
  const byDir = new Map<string, Map<string, Set<string>>>();
  let unreadable = 0;
  let lastPoll: FilingSeenPoll | null = null;

  function loadDay(dir: string, day: string): Set<string> {
    const set = new Set<string>();
    // .jsonl and .jsonl.gz can both exist for a moment (gzip writes the .gz,
    // then unlinks the .jsonl): read whichever are there.
    for (const fp of [path.join(dir, `${day}.jsonl`), path.join(dir, `${day}.jsonl.gz`)]) {
      let raw: Buffer;
      try {
        raw = fs.readFileSync(fp);
      } catch (e) {
        if ((e as NodeJS.ErrnoException)?.code !== "ENOENT") {
          unreadable++;
          console.error(`[datacore] filingSeen: cannot read ${fp}: ${(e as Error)?.message || e}`);
        }
        continue;
      }
      try {
        accessionsInBuffer(fp.endsWith(".gz") ? zlib.gunzipSync(raw) : raw, set);
      } catch (e) {
        unreadable++;
        console.error(`[datacore] filingSeen: cannot decode ${fp}: ${(e as Error)?.message || e}`);
      }
    }
    return set;
  }

  function daysOf(dir: string): Map<string, Set<string>> {
    let days = byDir.get(dir);
    if (!days) { days = new Map(); byDir.set(dir, days); }
    return days;
  }

  function dayHolds(dir: string, day: string, accession: string): boolean {
    const days = daysOf(dir);
    let set = days.get(day);
    if (!set) { set = loadDay(dir, day); days.set(day, set); }
    return set.has(accession);
  }

  function evict(dir: string, earliestDay: string): void {
    const days = byDir.get(dir);
    if (!days) return;
    for (const day of Array.from(days.keys())) if (day < earliestDay) days.delete(day);
  }

  function selectFresh<T extends { accession: string; filedAt?: string | null }>(dir: string, filings: T[], nowMs: number): T[] {
    const today = dayOf(nowMs);
    const yesterday = dayOf(nowMs - DAY_MS);
    const earliestMs = nowMs - (lookbackDays - 1) * DAY_MS;
    const earliest = dayOf(earliestMs);
    evict(dir, earliest);
    const poll: FilingSeenPoll = { candidates: 0, fresh: 0, held_recent: 0, held_older_day: 0, before_lookback: 0 };
    const fresh: T[] = [];
    for (const f of filings) {
      if (!f || !f.accession) continue;
      poll.candidates++;
      // The archive day is a UTC date and `filedAt` an Eastern one, so a row
      // is never archived under a day before its filed date; one extra day
      // back costs one cached lookup and covers a clock or feed oddity.
      const filed = typeof f.filedAt === "string" && DAY_RE.test(f.filedAt) ? Date.parse(`${f.filedAt.slice(0, 10)}T00:00:00Z`) : NaN;
      let fromMs = Number.isFinite(filed) ? filed - DAY_MS : NaN;
      if (!Number.isFinite(fromMs) || fromMs < earliestMs) {
        if (!Number.isFinite(filed) || dayOf(filed) < earliest) poll.before_lookback++;
        fromMs = earliestMs;
      }
      if (fromMs > nowMs) fromMs = nowMs;
      // Newest day first: the common case (still on today's page) ends on the
      // first lookup.
      let heldDay: string | null = null;
      const fromDay = dayOf(fromMs);
      for (let ms = nowMs; ; ms -= DAY_MS) {
        const day = dayOf(ms);
        if (day < fromDay) break;
        if (dayHolds(dir, day, f.accession)) { heldDay = day; break; }
      }
      if (heldDay === null) { fresh.push(f); poll.fresh++; }
      else if (heldDay === today || heldDay === yesterday) poll.held_recent++;
      else poll.held_older_day++;
    }
    lastPoll = poll;
    return fresh;
  }

  function noteAppended(dir: string, accessions: Iterable<string>, nowMs: number): void {
    const day = dayOf(nowMs);
    const days = daysOf(dir);
    let set = days.get(day);
    // Not loaded yet: load it now (the append has already happened, so the
    // file holds these rows) rather than caching a partial set for the day.
    if (!set) { set = loadDay(dir, day); days.set(day, set); }
    for (const a of accessions) if (a) set.add(a);
  }

  function stats(): FilingSeenStats {
    let daySets = 0;
    let entries = 0;
    byDir.forEach((days) => { daySets += days.size; days.forEach((s) => { entries += s.size; }); });
    return { lookback_days: lookbackDays, day_sets: daySets, entries, unreadable_day_files: unreadable, last_poll: lastPoll };
  }

  return {
    selectFresh,
    noteAppended,
    stats,
    get size() { return stats().entries; },
    _resetForTests() { byDir.clear(); unreadable = 0; lastPoll = null; },
  };
}
