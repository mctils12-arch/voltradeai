/**
 * gzipDayFile.ts — compress one archive file without ever replacing a
 * compressed file that already exists.
 *
 * WHY (2026-11-26, [REPAIR], a root-cause session: the same fault was fixed
 * for the aircraft hour files on 2026-11-16 in datacoreArchive.ts and was
 * still live in every other archive). Some forty archive modules each carried
 * their own copy of
 *
 *     fs.writeFileSync(`${fp}.gz`, zlib.gzipSync(fs.readFileSync(fp)));
 *     fs.unlinkSync(fp);
 *
 * A day file is named for the day of the DATA in many of them (the observed
 * hour, the filing date, the report date). When a row for an old day arrives
 * after that day was compressed, the module appends it to a NEW plain file
 * beside the existing `.gz`, and the lines above then write the new file's
 * gzip OVER the existing one: every row archived for that day is replaced by
 * the late rows. The in-memory dedup set still holds the lost rows' keys, so
 * they are never written again. Reproduced on gridGeneration.ts: a respondent
 * publishing four late hours removed the four hours already held for the
 * eleven other respondents on that day.
 *
 * NOW, in one place:
 *   - an existing `.gz` is MERGED: its lines first, then the late ones;
 *   - the result is written to `<name>.gz.tmp` and renamed into place, so an
 *     interrupted compression loses only its own temp file;
 *   - a merge that cannot be done safely (the existing `.gz` does not gunzip,
 *     or is larger than MERGE_MAX_GZ_BYTES) leaves BOTH files where they are
 *     and says so on stderr. Every reader of these archives already scans
 *     `.jsonl` and `.jsonl.gz`.
 * server/gzipDayFile.test.ts holds the rule and fails the build if the raw
 * overwrite line reappears in any server module.
 *
 * Standalone on purpose (fs, path, zlib only): the archive modules import it,
 * so it must not import any of them.
 */
import fs from "fs";
import path from "path";
import zlib from "zlib";

/** An existing .gz larger than this is not merged in memory (both files stay). */
export const MERGE_MAX_GZ_BYTES = 64 * 1024 * 1024;

export type GzipDayFileResult = "compressed" | "merged" | "left_beside";

const counts = { compressed: 0, merged: 0, left_beside: 0 };
/** Since boot: files compressed, late files merged into an existing .gz, and
 *  late files left beside one. */
export function gzipDayFileStats(): { compressed: number; merged: number; left_beside: number } {
  return { ...counts };
}

/** Existing lines first, then the late ones; a missing final newline never
 *  joins two lines. */
export function mergedBody(existing: Buffer, late: Buffer): Buffer {
  const needNl = existing.length > 0 && existing[existing.length - 1] !== 0x0a;
  return needNl ? Buffer.concat([existing, Buffer.from("\n"), late]) : Buffer.concat([existing, late]);
}

function leftBeside(fp: string, why: string): GzipDayFileResult {
  counts.left_beside++;
  console.error(`[archive] ${path.basename(path.dirname(fp))}/${path.basename(fp)} left beside its existing .gz (${why}); both are still read`);
  return "left_beside";
}

/**
 * Compresses `fp` to `fp + ".gz"` and removes `fp`. Throws what the old lines
 * threw (an unreadable source, a full disk); every caller already catches.
 */
export function gzipDayFile(fp: string, mergeMaxGzBytes = MERGE_MAX_GZ_BYTES): GzipDayFileResult {
  const gzPath = `${fp}.gz`;
  const tmp = `${gzPath}.tmp`;
  let body = fs.readFileSync(fp);
  const late = fs.existsSync(gzPath);
  if (late) {
    if (fs.statSync(gzPath).size > mergeMaxGzBytes) return leftBeside(fp, "the existing .gz is over the merge size limit");
    let existing: Buffer;
    try {
      existing = zlib.gunzipSync(fs.readFileSync(gzPath));
    } catch (e: unknown) {
      return leftBeside(fp, `the existing .gz does not gunzip: ${e instanceof Error ? e.message : String(e)}`);
    }
    body = mergedBody(existing, body);
  }
  try {
    fs.writeFileSync(tmp, zlib.gzipSync(body));
    fs.renameSync(tmp, gzPath);
  } catch (e: unknown) {
    fs.rmSync(tmp, { force: true });
    throw e;
  }
  fs.unlinkSync(fp);
  if (late) counts.merged++; else counts.compressed++;
  return late ? "merged" : "compressed";
}
