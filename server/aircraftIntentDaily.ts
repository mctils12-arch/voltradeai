// aircraftIntentDaily.ts — permanent daily rollup of the archived aircraft
// intent fields (AIRCRAFT INTENT B, 2026-10-14).
//
// WHY: the generic rollup (datacoreArchive rollupOldDaysAsync) keeps only
// position/time polylines, and raw hour files leave local disk at the hot-
// tier floor and leave R2 after RAW_RETENTION_DAYS. Without a fold, the
// intent history the E products need would vanish after 30 days — the same
// bug shape gnssIntegrityDaily.ts and fleetUtilization.ts fixed for their
// fields. This folds each COMPLETE UTC day (once, idempotent) into one small
// file `aircraft_intent_daily/<day>.json.gz` that is never deleted.
//
// WHAT IT HOLDS — ingredients, not products (the products E2-E4 define
// their own claims and honesty labels on top):
//   grid: per UTC hour x 2-degree cell x altitude band, airborne samples only:
//     - wind (E3): count + summed u/v components (kt, direction the air moves
//       TO), so the mean is a proper vector mean
//     - outside air temperature (E3): count, sum, sum of squares (degC)
//     - altimeter setting nav_qnh below FL180 (E4): count, sum (hPa)
//     - alt_geom - alt_baro (E4 pressure anomaly): count, sum, sum of sq (ft)
//     - baro_rate (E2 ingredient — dispersion at ARCHIVE cadence, which is
//       NOT a turbulence measurement by itself): count, sum, sum of sq (fpm)
//   coverage: per field, the share of airborne intent lines carrying it, and
//   the number of aircraft that sent mode bits — sample size travels with
//   every number; nothing is extrapolated to cells without samples.
// E1 (ATC stress) and E5 (airline ops) need the C/D detectors (vectoring,
// go-arounds, flags); their rollups ship with those detectors.
//
// Reads are streamed (datacoreArchive.streamJsonlLines yields between
// chunks) and only lines carrying an `x` block are parsed, so the fold never
// stalls the shared event loop. Best-effort: never throws.

import fs from "fs";
import path from "path";
import zlib from "zlib";
import { archiveBaseDir, streamJsonlLines } from "./datacoreArchive";
import { IntentArchiveDecoder, INTENT_FIELDS, type IntentField, type IntentValues } from "./aircraftIntent";

export const INTENT_DAILY_DIR = "aircraft_intent_daily";
export const INTENT_DAILY_SCHEMA = 1;
export const INTENT_GRID_DEG = 2;
/** band i covers [BANDS[i], BANDS[i+1]) ft; the last band is open-ended */
export const INTENT_BANDS_FT = [0, 5_000, 10_000, 15_000, 20_000, 25_000, 30_000, 35_000, 40_000] as const;
/** nav_qnh counts toward the pressure field only below this (US transition
 *  ceiling; above it crews set standard 1013.25) */
export const QNH_MAX_ALT_FT = 18_000;
/** a day is folded only once its last hour file can no longer grow */
export const FOLD_AFTER_DAY_END_MS = 2 * 3600_000;
/** The first UTC day whose hour files can carry `x` lines (schema v4 went
 *  live 2026-10-14 ~22:40Z, v1.0.1076). Older days hold no intent by
 *  construction: folding them only re-reads ~12 MB/day of gz to write an
 *  empty rollup (seen live the first hour: 2026-09-23/24 folded empty). */
export const INTENT_ARCHIVE_SINCE_DAY = "2026-10-14";
/** bound the work one call does (older days catch up on later ticks) */
export const MAX_DAYS_PER_FOLD = 2;

export const GRID_COLS = [
  "h", "la", "lo", "b", "n",
  "wn", "wu", "wv",
  "tn", "ts", "tss",
  "qn", "qs",
  "gn", "gs", "gss",
  "rn", "rs", "rss",
] as const;

const HOUR_FILE_RE = /^(\d{4}-\d{2}-\d{2})-(\d{2})\.jsonl(\.gz)?$/;
const DEG = Math.PI / 180;
const r1 = (x: number) => Math.round(x * 10) / 10;

export function bandOf(altFt: number): number {
  let b = 0;
  for (let i = 1; i < INTENT_BANDS_FT.length; i++) if (altFt >= INTENT_BANDS_FT[i]) b = i;
  return b;
}

export interface IntentDaily {
  schema: number;
  day: string;
  built_at: number;
  hours_present: number;
  lines: number;
  intent_lines: number;
  corrupt_lines: number;
  airborne_intent_lines: number;
  aircraft: number;
  mode_bit_aircraft: number;
  /** % of airborne intent lines carrying each field */
  field_pct: Record<IntentField, number>;
  grid: { res_deg: number; bands_ft: readonly number[]; cols: readonly string[]; rows: number[][] };
  note: string;
}

/** Pure accumulator: feed each hour file's rows in file order (one fresh
 *  decoder per file — the archive's keyframe-per-hour contract). */
export class IntentDayAccumulator {
  private readonly cells = new Map<string, number[]>();
  private readonly fieldN = new Array(INTENT_FIELDS.length).fill(0);
  private readonly aircraftSet = new Set<string>();
  private readonly modeSet = new Set<string>();
  private readonly hours = new Set<number>();
  lines = 0;
  intentLines = 0;
  airborne = 0;
  /** unparseable lines that carried an x block (skipped, never guessed) */
  corruptLines = 0;

  /** rows of ONE hour file, in file order */
  addHour(hour: number, rows: Iterable<Record<string, unknown>>): void {
    this.hours.add(hour);
    const dec = new IntentArchiveDecoder();
    for (const row of rows) {
      this.lines++;
      const v = dec.feed(row);
      if (v) this.addSample(hour, row, v);
    }
  }

  /** count a line that was never parsed (no `x`), keeping `lines` honest */
  countBareLines(n: number): void { this.lines += n; }

  addSample(hour: number, row: Record<string, unknown>, v: IntentValues): void {
    this.intentLines++;
    const hex = String(row.i);
    this.aircraftSet.add(hex);
    if (v.nav_modes !== null) this.modeSet.add(hex);
    const la = row.la, lo = row.lo;
    if (row.g || typeof la !== "number" || typeof lo !== "number") return; // airborne only
    const altFt = v.alt_baro ?? v.alt_geom ?? (typeof row.al === "number" ? row.al / 0.3048 : null);
    if (altFt == null) return;
    this.airborne++;
    const vr = v as unknown as Record<string, unknown>;
    for (let i = 0; i < INTENT_FIELDS.length; i++) if (vr[INTENT_FIELDS[i]] !== null) this.fieldN[i]++;
    const cla = Math.floor(la / INTENT_GRID_DEG) * INTENT_GRID_DEG;
    const clo = Math.floor(lo / INTENT_GRID_DEG) * INTENT_GRID_DEG;
    const b = bandOf(altFt);
    const key = `${hour}|${cla}|${clo}|${b}`;
    let c = this.cells.get(key);
    if (!c) { c = [hour, cla, clo, b, ...new Array(GRID_COLS.length - 4).fill(0)]; this.cells.set(key, c); }
    c[4]++; // n
    if (v.wd !== null && v.ws !== null) {
      c[5]++;
      c[6] += -v.ws * Math.sin(v.wd * DEG); // wd = direction the wind blows FROM
      c[7] += -v.ws * Math.cos(v.wd * DEG);
    }
    if (v.oat !== null) { c[8]++; c[9] += v.oat; c[10] += v.oat * v.oat; }
    if (v.nav_qnh !== null && altFt < QNH_MAX_ALT_FT) { c[11]++; c[12] += v.nav_qnh; }
    if (v.alt_geom !== null && v.alt_baro !== null) {
      const d = v.alt_geom - v.alt_baro;
      c[13]++; c[14] += d; c[15] += d * d;
    }
    if (v.baro_rate !== null) { c[16]++; c[17] += v.baro_rate; c[18] += v.baro_rate * v.baro_rate; }
  }

  result(day: string, builtAt: number): IntentDaily {
    const rows = Array.from(this.cells.values())
      .map((c) => c.map((x, i) => (i < 5 ? x : r1(x))))
      .sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2] || a[3] - b[3]);
    const pct = (n: number) => (this.airborne ? Math.round((n / this.airborne) * 1000) / 10 : 0);
    return {
      schema: INTENT_DAILY_SCHEMA, day, built_at: builtAt,
      hours_present: this.hours.size, lines: this.lines, intent_lines: this.intentLines,
      corrupt_lines: this.corruptLines,
      airborne_intent_lines: this.airborne, aircraft: this.aircraftSet.size,
      mode_bit_aircraft: this.modeSet.size,
      field_pct: Object.fromEntries(INTENT_FIELDS.map((f, i) => [f, pct(this.fieldN[i])])) as Record<IntentField, number>,
      grid: { res_deg: INTENT_GRID_DEG, bands_ft: INTENT_BANDS_FT, cols: GRID_COLS, rows },
      note: "Ingredients for the E products, from archived aircraft intent lines (archive cadence, airborne "
        + "only). Every cell carries its own sample counts; cells without samples are absent, never "
        + "filled. wu/wv sum the wind's TO-direction components (kt); baro_rate dispersion at archive "
        + "cadence is not a turbulence measurement by itself.",
    };
  }
}

export function intentDailyPath(base: string, day: string): string {
  return path.join(base, INTENT_DAILY_DIR, `${day}.json.gz`);
}

/** Fold every complete UTC day (ended >= FOLD_AFTER_DAY_END_MS ago, and not
 *  before INTENT_ARCHIVE_SINCE_DAY) whose rollup does not exist yet, oldest
 *  first, up to MAX_DAYS_PER_FOLD per call. Days with no local hour files
 *  (already evicted) are skipped. Never throws. */
export async function foldIntentDays(
  base = archiveBaseDir(), nowMs = Date.now(), maxDays = MAX_DAYS_PER_FOLD,
): Promise<{ daysFolded: string[]; bytes: number }> {
  const out = { daysFolded: [] as string[], bytes: 0 };
  const dir = path.join(base, "aircraft");
  let names: string[] = [];
  try { names = fs.readdirSync(dir); } catch { return out; }
  // hour -> file; when compression is mid-flight both exist and the plain
  // file is the complete one (the .gz may be partial)
  const byDay = new Map<string, Map<number, string>>();
  for (const f of names) {
    const m = HOUR_FILE_RE.exec(f);
    if (!m || m[1] < INTENT_ARCHIVE_SINCE_DAY) continue;
    const dayEnd = Date.parse(`${m[1]}T00:00:00Z`) + 86_400_000;
    if (!Number.isFinite(dayEnd) || nowMs - dayEnd < FOLD_AFTER_DAY_END_MS) continue;
    if (fs.existsSync(intentDailyPath(base, m[1]))) continue;
    const hours = byDay.get(m[1]) ?? new Map<number, string>();
    const h = Number(m[2]);
    if (!hours.has(h) || !m[3]) hours.set(h, f);
    byDay.set(m[1], hours);
  }
  for (const day of Array.from(byDay.keys()).sort().slice(0, maxDays)) {
    try {
      const acc = new IntentDayAccumulator();
      const hours = Array.from(byDay.get(day)!.entries()).sort((a, b) => a[0] - b[0]);
      for (const [h, f] of hours) {
        const rows: Record<string, unknown>[] = [];
        let bare = 0;
        await streamJsonlLines(path.join(dir, f), f.endsWith(".gz"), (line) => {
          if (!line.includes('"x":')) { bare++; return; } // cheap skip: no intent block
          try { rows.push(JSON.parse(line)); } catch { acc.corruptLines++; } // skipped like every reader, but counted
        });
        acc.countBareLines(bare);
        acc.addHour(h, rows);
      }
      const body = zlib.gzipSync(JSON.stringify(acc.result(day, nowMs)));
      const fp = intentDailyPath(base, day);
      fs.mkdirSync(path.dirname(fp), { recursive: true });
      fs.writeFileSync(fp + ".tmp", body);
      fs.renameSync(fp + ".tmp", fp);
      out.daysFolded.push(day);
      out.bytes += body.length;
    } catch (e) {
      console.error("[intent-daily] fold", day, e instanceof Error ? e.message : String(e));
    }
  }
  return out;
}

export function loadIntentDaily(day: string, base = archiveBaseDir()): IntentDaily | null {
  try { return JSON.parse(zlib.gunzipSync(fs.readFileSync(intentDailyPath(base, day))).toString()) as IntentDaily; }
  catch { return null; }
}

/** Days on disk + their total size (coverage/health reporting). */
export function intentDailyInventory(base = archiveBaseDir()): { days: number; latest: string | null; bytes: number } {
  try {
    const dir = path.join(base, INTENT_DAILY_DIR);
    const files = fs.readdirSync(dir).filter((f) => /^\d{4}-\d{2}-\d{2}\.json\.gz$/.test(f)).sort();
    let bytes = 0;
    for (const f of files) bytes += fs.statSync(path.join(dir, f), { throwIfNoEntry: false })?.size ?? 0; // a raced delete counts 0
    return { days: files.length, latest: files.length ? files[files.length - 1].slice(0, 10) : null, bytes };
  } catch {
    return { days: 0, latest: null, bytes: 0 };
  }
}
