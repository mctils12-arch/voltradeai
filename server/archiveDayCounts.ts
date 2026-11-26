/**
 * archiveDayCounts.ts — counts of what an archive day file holds, for the
 * three archives a late row could overwrite (KNOWN BROKEN #62).
 *
 * WHY (2026-11-26, [PIPELINE], open_questions.md (av2)). Until #62's repair a
 * late row for an already-compressed day replaced every row the archive held
 * for that day, in `gridgeneration`, `griddemand` and `wikiattention`. The
 * repair stops new loss. It cannot say how many days were already lost,
 * because nothing a session can call read a day file's rows. EIA serves its
 * hourly history back to 2019 and Wikimedia its page views back to 2015, so
 * a lost day CAN be read again, but only once it is counted. This module is
 * the count. It reads; it never writes, repairs or re-requests.
 *
 * WHAT A DAY'S COUNT IS. Each row belongs to a GROUP (the balancing authority
 * for the two EIA archives, the ticker for page views), a SLOT inside the day
 * (the observed hour; page views have one row a day and no slot) and a SERIES
 * (fuel type; demand or forecast). The key group|slot|series is the same
 * identity the archive modules dedup on. Per day: rows, distinct keys, rows
 * that repeat a key, rows whose value is null ((av4)), groups, and how many
 * groups hold fewer slots than a whole day has.
 *
 * WHAT COUNTS AS A SUSPECTED OVERWRITE, FIXED BEFORE ANY PRODUCTION DAY WAS
 * READ. Only a compressed day can have been overwritten. Two signatures:
 *   A. (av2)'s, as registered: the day holds fewer groups than the calendar
 *      day before it AND the calendar day after it.
 *   B. added here because two damaged days in a row hide each other from A:
 *      the day holds fewer than half the median row count of the compressed
 *      days around it (LOCAL_MEDIAN_SPAN held days each side), so a year in
 *      which the source itself carried fewer series is not flagged against a
 *      later year.
 * A day carrying either is listed with which. A suspect is a day to read
 * again from the source, not a proven loss: a source that really published
 * less that day carries the same signature, and the re-read decides.
 *
 * Read-only and paced: one file at a time, the event loop yielded between
 * files, at most MAX_WINDOW_DAYS days a request with per-day rows, or the whole archive
 * (?scan=all, at most MAX_SCAN_DAYS) returning suspects and totals only. Counts, days and group codes
 * only (public balancing-authority codes and tickers); no value is returned.
 */
import fs from "fs";
import path from "path";
import zlib from "zlib";
import { promisify } from "util";
import { archiveBaseDir } from "./datacoreArchive";
import { gzipDayFileStats } from "./gzipDayFile";

const gunzip = promisify(zlib.gunzip);

type Row = Record<string, unknown>;

export interface StreamSpec {
  /** archive directory name */
  stream: string;
  group: (r: Row) => string | null;
  /** null: one row a group a day */
  slot: ((r: Row) => string | null) | null;
  series: (r: Row) => string;
  /** the field whose null is counted */
  value: string;
  /** slots a whole day has for one group (null: no slots) */
  slotsPerDay: number | null;
  /** what the three words mean for this stream, returned with every answer */
  reads: string;
}

const str = (v: unknown): string | null => (typeof v === "string" && v ? v : null);

/** The archives #62 exposed. Keys mirror each module's own dedup key
 *  (gridGeneration.ts obsKey, gridDemand.ts obsKey, wikiAttention.ts keyOf);
 *  server/archiveDayCounts.test.ts holds them to the modules' real rows. */
export const COUNT_STREAMS: readonly StreamSpec[] = [
  {
    stream: "gridgeneration",
    group: (r) => str(r.respondent),
    slot: (r) => str(r.period),
    series: (r) => String(r.fueltype ?? ""),
    value: "mwh",
    slotsPerDay: 24,
    reads: "group = balancing authority, slot = observed UTC hour, series = fuel type",
  },
  {
    stream: "griddemand",
    group: (r) => str(r.respondent),
    slot: (r) => str(r.period),
    series: (r) => String(r.type || "D"),
    value: "mwh",
    slotsPerDay: 24,
    reads: "group = balancing authority, slot = observed UTC hour, series = D (demand) or DF (day-ahead forecast)",
  },
  {
    stream: "wikiattention",
    group: (r) => str(r.ticker),
    slot: null,
    series: () => "",
    value: "views",
    slotsPerDay: null,
    reads: "group = ticker, one row a ticker a day",
  },
];

export const MAX_WINDOW_DAYS = 60;
export const DEFAULT_WINDOW_DAYS = 30;
/** ?scan=all reads at most this many of the newest held days (11 years). */
export const MAX_SCAN_DAYS = 4000;
/** Signature B compares a day with this many held days each side of it. */
export const LOCAL_MEDIAN_SPAN = 15;
/** A compressed day file larger than this is not read (reported unreadable). */
export const MAX_GZ_BYTES = 64 * 1024 * 1024;
/** Per-group rows returned for one day (the diag sanitizer cuts arrays at 200). */
export const MAX_GROUP_ROWS = 200;

const DAY_FILE = /^(\d{4}-\d{2}-\d{2})\.jsonl(\.gz)?$/;
const DAY = /^\d{4}-\d{2}-\d{2}$/;

export function specFor(stream: unknown): StreamSpec | null {
  return COUNT_STREAMS.find((s) => s.stream === stream) ?? null;
}

export interface GroupCount { group: string; rows: number; slots: number; series: number; null_values: number }

export interface DayCount {
  day: string;
  /** which of the day's two files exist; both means late rows wait beside a compressed day */
  plain: boolean;
  gz: boolean;
  /** why a file of this day could not be read, else null */
  unreadable: string | null;
  rows: number;
  bad_lines: number;
  keys: number;
  repeated_rows: number;
  null_values: number;
  groups: number;
  /** distinct slots over all groups (0 when the stream has no slots) */
  slots: number;
  /** groups holding fewer slots than a whole day has (0 when the stream has no slots) */
  groups_short: number;
  slots_min: number | null;
  slots_max: number | null;
}

interface Folded { count: DayCount; byGroup: GroupCount[] }

/** Folds the text of a day's file(s) into counts. Pure. */
export function foldDay(spec: StreamSpec, day: string, texts: string[], files: { plain: boolean; gz: boolean; unreadable: string | null }): Folded {
  const keys = new Set<string>();
  const slots = new Set<string>();
  const groups = new Map<string, { rows: number; slots: Set<string>; series: Set<string>; nulls: number }>();
  let rows = 0, bad = 0, repeated = 0, nulls = 0;
  for (const text of texts) {
    for (const line of text.split("\n")) {
      if (!line) continue;
      let r: Row;
      try {
        const parsed: unknown = JSON.parse(line);
        if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) { bad++; continue; }
        r = parsed as Row;
      } catch {
        bad++;
        continue;
      }
      const g = spec.group(r);
      const s = spec.slot ? spec.slot(r) : "";
      if (g === null || s === null) { bad++; continue; }
      rows++;
      const ser = spec.series(r);
      const key = `${g}|${s}|${ser}`;
      if (keys.has(key)) repeated++; else keys.add(key);
      const isNull = r[spec.value] === null || r[spec.value] === undefined;
      if (isNull) nulls++;
      let acc = groups.get(g);
      if (!acc) { acc = { rows: 0, slots: new Set(), series: new Set(), nulls: 0 }; groups.set(g, acc); }
      acc.rows++;
      if (spec.slot) { acc.slots.add(s); slots.add(s); }
      acc.series.add(ser);
      if (isNull) acc.nulls++;
    }
  }
  const byGroup: GroupCount[] = Array.from(groups.entries())
    .map(([group, a]) => ({ group, rows: a.rows, slots: a.slots.size, series: a.series.size, null_values: a.nulls }))
    .sort((a, b) => a.group.localeCompare(b.group));
  const per = spec.slotsPerDay;
  const slotCounts = byGroup.map((g) => g.slots);
  return {
    count: {
      day, ...files, rows, bad_lines: bad, keys: keys.size, repeated_rows: repeated, null_values: nulls,
      groups: byGroup.length,
      slots: slots.size,
      groups_short: per === null ? 0 : byGroup.filter((g) => g.slots < per).length,
      slots_min: per === null || !slotCounts.length ? null : Math.min(...slotCounts),
      slots_max: per === null || !slotCounts.length ? null : Math.max(...slotCounts),
    },
    byGroup,
  };
}

async function readDay(spec: StreamSpec, dir: string, day: string): Promise<Folded> {
  const texts: string[] = [];
  const files = { plain: false, gz: false, unreadable: null as string | null };
  const plain = path.join(dir, `${day}.jsonl`);
  const gz = `${plain}.gz`;
  // the compressed file first: it is the older part of the day
  try {
    const st = await fs.promises.stat(gz);
    files.gz = true;
    if (st.size > MAX_GZ_BYTES) files.unreadable = `the .gz is over ${MAX_GZ_BYTES} bytes and was not read`;
    else texts.push((await gunzip(await fs.promises.readFile(gz))).toString("utf8"));
  } catch (e: unknown) {
    if ((e as NodeJS.ErrnoException)?.code !== "ENOENT") files.unreadable = `the .gz does not read: ${e instanceof Error ? e.message : String(e)}`;
  }
  try {
    texts.push(await fs.promises.readFile(plain, "utf8"));
    files.plain = true;
  } catch (e: unknown) {
    if ((e as NodeJS.ErrnoException)?.code !== "ENOENT") files.unreadable = `the plain file does not read: ${e instanceof Error ? e.message : String(e)}`;
  }
  return foldDay(spec, day, texts, files);
}

const median = (xs: number[]): number | null => {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

const dayPlus = (day: string, n: number): string =>
  new Date(Date.parse(`${day}T00:00:00Z`) + n * 86400_000).toISOString().slice(0, 10);

export interface Suspect { day: string; rows: number; groups: number; fewer_groups_than_both_neighbours: boolean; under_half_the_median_rows: boolean }

/** The two signatures of the header, over one window of day counts. Pure. */
export function suspects(days: DayCount[]): { median_rows: number | null; suspects: Suspect[] } {
  const byDay = new Map(days.map((d) => [d.day, d]));
  const usable = (d: DayCount) => d.gz && !d.unreadable;
  const out: Suspect[] = [];
  for (let i = 0; i < days.length; i++) {
    const d = days[i];
    if (!usable(d)) continue;
    const med = median(days.slice(Math.max(0, i - LOCAL_MEDIAN_SPAN), i + LOCAL_MEDIAN_SPAN + 1).filter(usable).map((x) => x.rows));
    const prev = byDay.get(dayPlus(d.day, -1));
    const next = byDay.get(dayPlus(d.day, 1));
    const a = Boolean(prev && next && !prev.unreadable && !next.unreadable && d.groups < prev.groups && d.groups < next.groups);
    const b = med !== null && d.rows < med / 2;
    if (a || b) out.push({ day: d.day, rows: d.rows, groups: d.groups, fewer_groups_than_both_neighbours: a, under_half_the_median_rows: b });
  }
  return { median_rows: median(days.filter(usable).map((d) => d.rows)), suspects: out };
}

function listDays(dir: string): string[] {
  let names: string[];
  try {
    names = fs.readdirSync(dir);
  } catch (e: unknown) {
    if ((e as NodeJS.ErrnoException)?.code === "ENOENT") return [];
    throw e;
  }
  const days = new Set<string>();
  for (const f of names) {
    const m = DAY_FILE.exec(f);
    if (m) days.add(m[1]);
  }
  return Array.from(days).sort();
}

const HOW_TO_READ =
  "A suspect is a compressed day that either holds fewer groups than the day before and the day after, or holds under half " +
  "the median rows of the compressed days around it. It is a day to read again from the source, not a proven loss. " +
  "?scan=all reads the whole archive and returns suspects and totals only; ?to=<day>&days=N walks it with per-day rows.";

export interface CountQuery { stream?: unknown; day?: unknown; to?: unknown; days?: unknown; scan?: unknown }

/**
 * The `archive_day_counts` diag probe.
 *   no stream            the streams it can count, with each one's held days
 *   ?stream=             the newest `days` days (default 30, at most 60) up to `to`
 *   ?stream=&scan=all    every held day: suspects and totals, no per-day rows
 *   ?stream=&day=        one day, with its groups
 */
export async function archiveDayCounts(q: CountQuery = {}, baseDir?: string): Promise<Record<string, unknown>> {
  const base = baseDir || archiveBaseDir();
  const merges = gzipDayFileStats();
  if (q.stream === undefined || q.stream === null || q.stream === "") {
    return {
      probe: "archive_day_counts",
      compress_since_boot: merges,
      streams: COUNT_STREAMS.map((s) => {
        const days = listDays(path.join(base, s.stream));
        return { stream: s.stream, reads: s.reads, days_held: days.length, first_day: days[0] ?? null, last_day: days[days.length - 1] ?? null };
      }),
      usage: "?stream=<name> for the newest days; &to=YYYY-MM-DD&days=N (at most 60) for an older window; &scan=all for the whole archive's suspects; &day=YYYY-MM-DD for one day's groups",
    };
  }
  const spec = specFor(q.stream);
  if (!spec) return { probe: "archive_day_counts", error: `unknown stream; one of: ${COUNT_STREAMS.map((s) => s.stream).join(", ")}` };
  const dir = path.join(base, spec.stream);
  const held = listDays(dir);
  const head = {
    probe: "archive_day_counts", stream: spec.stream, reads: spec.reads, slots_per_day: spec.slotsPerDay,
    days_held: held.length, first_day: held[0] ?? null, last_day: held[held.length - 1] ?? null,
    compress_since_boot: merges,
  };
  if (q.day !== undefined && q.day !== "") {
    if (typeof q.day !== "string" || !DAY.test(q.day)) return { ...head, error: "day must be YYYY-MM-DD" };
    if (!held.includes(q.day)) return { ...head, day: q.day, held: false };
    const f = await readDay(spec, dir, q.day);
    return { ...head, held: true, ...f.count, groups_listed: Math.min(f.byGroup.length, MAX_GROUP_ROWS), by_group: f.byGroup.slice(0, MAX_GROUP_ROWS) };
  }
  if (q.to !== undefined && q.to !== "" && (typeof q.to !== "string" || !DAY.test(q.to))) return { ...head, error: "to must be YYYY-MM-DD" };
  const n = Math.max(1, Math.min(MAX_WINDOW_DAYS, Math.floor(Number(q.days)) || DEFAULT_WINDOW_DAYS));
  const all = q.scan === "all";
  const upTo = typeof q.to === "string" && q.to ? held.filter((d) => d <= (q.to as string)) : held;
  const window = upTo.slice(-(all ? MAX_SCAN_DAYS : n));
  const days: DayCount[] = [];
  for (const day of window) {
    days.push((await readDay(spec, dir, day)).count);
    await new Promise<void>((r) => setImmediate(r));
  }
  let missing = 0;
  if (window.length > 1) {
    const span = Math.round((Date.parse(window[window.length - 1]) - Date.parse(window[0])) / 86400_000) + 1;
    missing = span - window.length;
  }
  const sus = suspects(days);
  return {
    ...head,
    window: {
      first: window[0] ?? null, last: window[window.length - 1] ?? null, days: window.length,
      calendar_days_with_no_file: missing,
      compressed: days.filter((d) => d.gz).length,
      late_rows_waiting_beside_a_compressed_day: days.filter((d) => d.gz && d.plain).length,
      unreadable: days.filter((d) => d.unreadable).length,
      compressed_days_with_a_group_short_of_a_whole_day: days.filter((d) => d.gz && d.groups_short > 0).length,
      rows: days.reduce((a, d) => a + d.rows, 0),
      repeated_rows: days.reduce((a, d) => a + d.repeated_rows, 0),
      null_values: days.reduce((a, d) => a + d.null_values, 0),
      bad_lines: days.reduce((a, d) => a + d.bad_lines, 0),
      median_rows_of_compressed_days: sus.median_rows,
      suspect_days: sus.suspects.length,
    },
    suspects_listed: Math.min(sus.suspects.length, MAX_GROUP_ROWS),
    suspects: sus.suspects.slice(0, MAX_GROUP_ROWS),
    how_to_read: HOW_TO_READ,
    ...(all ? { scan: "all" } : { by_day: days }),
  };
}
