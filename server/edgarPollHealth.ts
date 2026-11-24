/**
 * EDGAR poll health: what each of the three EDGAR pollers (Form 4, 13F, 8-K
 * earnings) did on its last polls, and when its archive last took a row.
 *
 * WHY (research/open_questions.md 2026-11-16, "the Form 4 archive recorded
 * nothing ... and nothing alarmed", item (2) THE SILENCE): on Monday
 * 2026-11-16 every Form 4 poll threw (EDGAR answered HTTP 400), each refresh
 * caught the error, logged one console line and refilled the cache from the
 * archive. /data looked populated, the stream registry said "live", and the
 * archive took no row for 11.6 hours while four sessions ran health checks.
 * A poller that fails must be countable from outside the container.
 *
 * Two kinds of evidence, because a deploy restarts the process many times a
 * day:
 *   - counters since boot (in memory): polls ok / failed, consecutive
 *     failures, the last error;
 *   - the archive's own last write (on disk): the newest mtime among the
 *     stream's plain `.jsonl` day files. A `.jsonl.gz` is written by the
 *     compactor, not by a poll, so it is not counted.
 *
 * Read-only for everything but its own counters. No ticker, name, CIK or
 * accession leaves this module; an error string is the poller's own message
 * (an sec.gov URL and a status code), truncated.
 *
 * The verdict (is a business day silent?) is taken by
 * scripts/session_health_check.py `check_edgar_archives`, which has the NYSE
 * calendar; this module serves facts.
 */
import fs from "fs";
import path from "path";
import { archiveBaseDir } from "./datacoreArchive";

export const EDGAR_POLL_STREAMS = ["filings", "filings13f", "earnings8k"] as const;
export type EdgarPollStream = (typeof EDGAR_POLL_STREAMS)[number];

const ERROR_MAX_CHARS = 240;

export interface EdgarPollCounters {
  polls_ok: number;
  polls_failed: number;
  /** failed polls since the last one that succeeded (0 after a success) */
  consecutive_failures: number;
  last_ok_at: string | null;
  last_failure_at: string | null;
  /** the poller's own message for the latest failure; kept after a later success */
  last_error: string | null;
  /** rows appended to the archive by polls since boot */
  rows_archived: number;
  /** feed requests that timed out, dropped or answered 5xx (edgarFeedFetch.ts);
   *  each was followed by another attempt unless it was the poll's last */
  feed_attempts_failed: number;
  /** polls whose feed page arrived on a second or third attempt: polls the
   *  single-request poller would have lost */
  polls_saved_by_retry: number;
}

function emptyCounters(): EdgarPollCounters {
  return {
    polls_ok: 0, polls_failed: 0, consecutive_failures: 0,
    last_ok_at: null, last_failure_at: null, last_error: null, rows_archived: 0,
    feed_attempts_failed: 0, polls_saved_by_retry: 0,
  };
}

/** What a stream's latest poll did with the filings it chose off its feed
 *  page. Counts and one error in words that name no filing.
 *
 *  [PIPELINE 2026-11-24, open_questions.md (ai3)] Four EDGAR repairs in nine
 *  days each began with a session finding out by hand what a poller had caught
 *  and dropped. Form 4 kept these counts from the fourth; 13F and 8-K caught
 *  every per-filing error in an empty `catch` and counted the poll ok. All
 *  three now report the same counts here, and the same rule fails a poll that
 *  fetched documents and read none (`pollUnreadError`). */
export interface EdgarPollDocuments {
  at: string;
  /** filings the poll chose off the feed page and asked EDGAR for */
  candidates: number;
  /** read and handed to the archive writer (which appends only the new ones) */
  returned: number;
  /** fetched, and the filing has nothing this stream keeps (Form 4: no issuer;
   *  13F: no primary document; 8-K: no Exhibit 99) */
  skipped: number;
  /** a request for the filing's listing or document failed */
  failed: number;
  /** the first failed request as "<what was asked> -> <status>"; null when none failed */
  first_error: string | null;
}

/** A failed document request as "<stage> -> 404" (or the stage and the error's
 *  text with every URL and accession removed). `stage` is the poller's own word
 *  for what it asked ("index.json", "information table", "exhibit"), never the
 *  file name: a 13F information table and an 8-K exhibit are named by the filer. */
export function describeStageError(stage: string, e: unknown): string {
  const msg = String((e as { message?: unknown } | null)?.message ?? e);
  const m = msg.match(/ -> (\d{3})$/);
  if (m) return `${stage} -> ${m[1]}`;
  const text = msg.replace(/https?:\/\/\S+/g, "<url>").replace(/\d{10}-\d{2}-\d{6}/g, "<accession>");
  return `${stage}: ${text}`.slice(0, 120);
}

/** The error a poll is counted failed with when it asked for documents and
 *  read none; null for every other poll. An empty page, a page read in part
 *  and a page whose filings were all skipped are good polls. */
export function pollUnreadError(what: string, docs: EdgarPollDocuments | null): Error | null {
  if (!docs || docs.returned > 0 || docs.failed === 0) return null;
  return new Error(`0 of ${docs.failed} ${what} could be read: ${docs.first_error ?? "no error recorded"}`);
}

const counters = new Map<EdgarPollStream, EdgarPollCounters>();
const documents = new Map<EdgarPollStream, EdgarPollDocuments>();
let bootedAtMs = Date.now();

function of(stream: EdgarPollStream): EdgarPollCounters {
  let c = counters.get(stream);
  if (!c) { c = emptyCounters(); counters.set(stream, c); }
  return c;
}

/** What the poll just finished did with its filings (replaces the previous poll's). */
export function notePollDocuments(stream: EdgarPollStream, docs: EdgarPollDocuments): void {
  documents.set(stream, { ...docs });
}

export function pollDocuments(stream: EdgarPollStream): EdgarPollDocuments | null {
  return documents.get(stream) ?? null;
}

/** A poll that read its feed. `rowsArchived` is what the archive writer returned. */
export function notePollOk(stream: EdgarPollStream, rowsArchived = 0, nowMs: number = Date.now()): void {
  const c = of(stream);
  c.polls_ok++;
  c.consecutive_failures = 0;
  c.last_ok_at = new Date(nowMs).toISOString();
  if (Number.isFinite(rowsArchived) && rowsArchived > 0) c.rows_archived += rowsArchived;
}

/** A poll whose feed request threw. Never throws itself. */
export function notePollFailed(stream: EdgarPollStream, err: unknown, nowMs: number = Date.now()): void {
  const c = of(stream);
  c.polls_failed++;
  c.consecutive_failures++;
  c.last_failure_at = new Date(nowMs).toISOString();
  const msg = err instanceof Error ? err.message : String(err ?? "unknown error");
  c.last_error = msg.length > ERROR_MAX_CHARS ? `${msg.slice(0, ERROR_MAX_CHARS)}…` : msg;
}

/** One feed request of a poll failed in a way that is retried. Does not
 *  count as a failed poll: `notePollFailed` does that if every attempt fails. */
export function noteFeedAttemptFailed(stream: EdgarPollStream, _err?: unknown): void {
  of(stream).feed_attempts_failed++;
}

/** A poll's feed page arrived after at least one failed attempt. */
export function noteFeedRecovered(stream: EdgarPollStream): void {
  of(stream).polls_saved_by_retry++;
}

/** Newest mtime among the stream's plain `<day>.jsonl` files, or null when the
 *  directory is missing or holds none (nothing appended in about two days:
 *  older days are gzipped in place). */
export function lastArchiveWriteMs(stream: EdgarPollStream, baseDir?: string): number | null {
  const dir = path.join(baseDir || archiveBaseDir(), stream);
  let names: string[];
  try {
    names = fs.readdirSync(dir);
  } catch (e) {
    if ((e as NodeJS.ErrnoException)?.code !== "ENOENT") {
      console.error("[datacore] edgarPollHealth: unreadable", dir, e instanceof Error ? e.message : e);
    }
    return null;
  }
  let newest: number | null = null;
  for (const name of names) {
    if (!/^\d{4}-\d{2}-\d{2}\.jsonl$/.test(name)) continue;
    try {
      const m = fs.statSync(path.join(dir, name)).mtimeMs;
      if (newest === null || m > newest) newest = m;
    } catch (e) {
      // ENOENT: the compactor replaced the file between readdir and stat (not a write)
      if ((e as NodeJS.ErrnoException)?.code !== "ENOENT") {
        console.error("[datacore] edgarPollHealth: stat failed", name, e instanceof Error ? e.message : e);
      }
    }
  }
  return newest;
}

export interface EdgarPollHealthRow extends EdgarPollCounters {
  stream: EdgarPollStream;
  last_archive_write_at: string | null;
  /** the latest poll's document counts; null before the first poll finishes its page */
  documents: EdgarPollDocuments | null;
}

/** Token-gated diag probe `edgar_poll`. */
export function edgarPollHealthStatus(baseDir?: string, nowMs: number = Date.now()): Record<string, unknown> {
  const streams: EdgarPollHealthRow[] = EDGAR_POLL_STREAMS.map((stream) => {
    const w = lastArchiveWriteMs(stream, baseDir);
    return {
      stream, ...of(stream),
      last_archive_write_at: w === null ? null : new Date(w).toISOString(),
      documents: pollDocuments(stream),
    };
  });
  return {
    probe: "edgar_poll",
    at: new Date(nowMs).toISOString(),
    booted_at: new Date(bootedAtMs).toISOString(),
    uptime_s: Math.max(0, Math.round((nowMs - bootedAtMs) / 1000)),
    poll_interval_min: 15,
    note: "Counters are since boot. feed_attempts_failed counts single feed requests that timed out or answered 5xx; polls_saved_by_retry counts polls that got their page on a later attempt. last_archive_write_at is read from the archive's day files and survives a restart; null means no plain day file (nothing appended in about two days). documents is the latest poll alone: filings it asked EDGAR for (candidates), read (returned), fetched with nothing to keep (skipped), not readable (failed) and the first failure as what was asked and the status. A poll with failed above 0 and returned 0 is counted in polls_failed.",
    streams,
  };
}

export function _resetEdgarPollHealthForTests(nowMs: number = Date.now()): void {
  counters.clear();
  documents.clear();
  bootedAtMs = nowMs;
}
