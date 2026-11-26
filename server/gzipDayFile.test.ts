// gzipDayFile battery. [REPAIR 2026-11-26]: forty archive modules compressed a
// day file by writing its gzip OVER whatever `.gz` already stood there, so a
// late row for an already-compressed day replaced every row archived for that
// day. The helper merges instead; this file holds the rule, reproduces the
// loss on the two EIA-930 archives, and fails if the raw line comes back.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import { gzipDayFile, gzipDayFileStats, mergedBody } from "./gzipDayFile";
import { archiveGeneration, gzipOldGenerationDays, _resetGridGenerationForTests } from "./gridGeneration";
import { archiveDemand, gzipOldDemandDays, _resetGridDemandForTests } from "./gridDemand";

const tmpDir = (tag: string) => fs.mkdtempSync(path.join(os.tmpdir(), `gzday-${tag}-`));
const gunzipLines = (fp: string) => zlib.gunzipSync(fs.readFileSync(fp)).toString("utf8").split("\n").filter(Boolean);

test("no existing .gz: the file is compressed and removed, byte for byte what the old lines wrote", () => {
  const dir = tmpDir("plain");
  const fp = path.join(dir, "2026-11-01.jsonl");
  fs.writeFileSync(fp, '{"a":1}\n{"a":2}\n');
  assert.equal(gzipDayFile(fp), "compressed");
  assert.equal(fs.existsSync(fp), false);
  assert.equal(zlib.gunzipSync(fs.readFileSync(`${fp}.gz`)).toString("utf8"), '{"a":1}\n{"a":2}\n');
  assert.deepEqual(fs.readdirSync(dir), ["2026-11-01.jsonl.gz"], "no temp file is left");
});

test("an existing .gz is merged, its lines first; nothing already archived is lost", () => {
  const dir = tmpDir("merge");
  const fp = path.join(dir, "2026-11-01.jsonl");
  fs.writeFileSync(`${fp}.gz`, zlib.gzipSync('{"a":1}\n{"a":2}\n'));
  fs.writeFileSync(fp, '{"a":3}\n');
  const before = gzipDayFileStats().merged;
  assert.equal(gzipDayFile(fp), "merged");
  assert.deepEqual(gunzipLines(`${fp}.gz`), ['{"a":1}', '{"a":2}', '{"a":3}']);
  assert.equal(fs.existsSync(fp), false);
  assert.equal(gzipDayFileStats().merged, before + 1);
});

test("mergedBody: an existing body without a final newline never joins two lines", () => {
  assert.equal(mergedBody(Buffer.from('{"a":1}'), Buffer.from('{"a":2}\n')).toString(), '{"a":1}\n{"a":2}\n');
  assert.equal(mergedBody(Buffer.from('{"a":1}\n'), Buffer.from('{"a":2}\n')).toString(), '{"a":1}\n{"a":2}\n');
  assert.equal(mergedBody(Buffer.alloc(0), Buffer.from('{"a":2}\n')).toString(), '{"a":2}\n');
});

test("a merge that cannot be done safely leaves BOTH files exactly as they were", () => {
  const dir = tmpDir("unsafe");
  const fp = path.join(dir, "2026-11-01.jsonl");
  fs.writeFileSync(`${fp}.gz`, "this is not gzip");
  fs.writeFileSync(fp, '{"a":3}\n');
  assert.equal(gzipDayFile(fp), "left_beside");
  assert.equal(fs.readFileSync(`${fp}.gz`, "utf8"), "this is not gzip");
  assert.equal(fs.readFileSync(fp, "utf8"), '{"a":3}\n');

  const big = path.join(dir, "2026-11-02.jsonl");
  const bigGz = zlib.gzipSync('{"a":1}\n');
  fs.writeFileSync(`${big}.gz`, bigGz);
  fs.writeFileSync(big, '{"a":3}\n');
  assert.equal(gzipDayFile(big, bigGz.length - 1), "left_beside", "over the merge size limit");
  assert.deepEqual(gunzipLines(`${big}.gz`), ['{"a":1}']);
  assert.equal(fs.existsSync(big), true);
});

test("a source that cannot be read throws, as the old lines did, and touches nothing", () => {
  const dir = tmpDir("missing");
  const fp = path.join(dir, "2026-11-01.jsonl");
  fs.writeFileSync(`${fp}.gz`, zlib.gzipSync('{"a":1}\n'));
  assert.throws(() => gzipDayFile(fp));
  assert.deepEqual(gunzipLines(`${fp}.gz`), ['{"a":1}']);
});

// ── The loss, reproduced on the two archives where it was found ─────────────
// Day files are named for the OBSERVED day. Day D is compressed three days
// later. A respondent that publishes hours of day D after that (EIA's ERCO
// series was 60 hours behind on 2026-11-26) gets a new plain file for day D.

const DAY = "2026-11-01";
const NOW = Date.parse("2026-11-10T00:00:00Z");

test("generation: late hours for a compressed day are added to it; the other respondents' rows of that day survive", () => {
  _resetGridGenerationForTests();
  const base = tmpDir("gen");
  const dir = path.join(base, "gridgeneration");
  const row = (respondent: string, hh: string) => ({ period: `${DAY}T${hh}`, respondent, fueltype: "NG", mwh: 10, rt: DAY });
  const early = ["PJM", "MISO", "CISO"].flatMap((r) => ["00", "01", "02", "03"].map((h) => row(r, h)));
  assert.equal(archiveGeneration(early, base), 12);
  assert.equal(gzipOldGenerationDays(base, NOW), 1);
  assert.equal(gunzipLines(path.join(dir, `${DAY}.jsonl.gz`)).length, 12);

  const lateRows = ["00", "01", "02", "03"].map((h) => row("ERCO", h));
  assert.equal(archiveGeneration(lateRows, base), 4);
  assert.equal(gzipOldGenerationDays(base, NOW), 1);
  const held = gunzipLines(path.join(dir, `${DAY}.jsonl.gz`)).map((l) => JSON.parse(l));
  assert.equal(held.length, 16, "before the repair this file held 4 rows: ERCO's, and nobody else's");
  for (const r of ["PJM", "MISO", "CISO", "ERCO"]) assert.equal(held.filter((o) => o.respondent === r).length, 4, r);
  assert.deepEqual(fs.readdirSync(dir), [`${DAY}.jsonl.gz`]);
});

test("demand: the same late arrival keeps the day whole", () => {
  _resetGridDemandForTests();
  const base = tmpDir("dem");
  const dir = path.join(base, "griddemand");
  const row = (respondent: string, hh: string) => ({ period: `${DAY}T${hh}`, respondent, type: "D" as const, mwh: 100, rt: DAY });
  assert.equal(archiveDemand(["PJM", "MISO"].flatMap((r) => ["00", "01"].map((h) => row(r, h))), base), 4);
  assert.equal(gzipOldDemandDays(base, NOW), 1);
  assert.equal(archiveDemand([row("NW", "00"), row("NW", "01")], base), 2);
  assert.equal(gzipOldDemandDays(base, NOW), 1);
  const held = gunzipLines(path.join(dir, `${DAY}.jsonl.gz`)).map((l) => JSON.parse(l));
  assert.equal(held.length, 6);
  assert.equal(held.filter((o) => o.respondent === "PJM").length, 2);
});

// ── The generator of the break: forty copies of one line ────────────────────

test("no server module writes a file's gzip straight onto `<file>.gz`", () => {
  const serverDir = path.dirname(new URL(import.meta.url).pathname);
  // `writeFileSync(<x>.gz, gzipSync(readFileSync(<x>)))` in its spellings: the
  // target is the source's own name plus ".gz", and nothing is read first.
  const raw = /writeFileSync\(\s*(`\$\{(\w+)\}\.gz`|(\w+)\s*\+\s*"\.gz")\s*,\s*zlib\.gzipSync\(\s*fs\.readFileSync\(\s*(\w+)\s*\)\s*\)\s*\)/;
  const offenders: string[] = [];
  let scanned = 0;
  for (const f of fs.readdirSync(serverDir)) {
    if (!f.endsWith(".ts") || f.endsWith(".test.ts")) continue;
    scanned++;
    fs.readFileSync(path.join(serverDir, f), "utf8").split("\n").forEach((line, i) => {
      if (line.trimStart().startsWith("*") || line.trimStart().startsWith("//")) return;
      if (raw.test(line)) offenders.push(`${f}:${i + 1}`);
    });
  }
  assert.ok(scanned > 300, `scanned ${scanned} server modules`);
  assert.deepEqual(offenders, [], "use gzipDayFile(fp) from ./gzipDayFile: it merges into an existing .gz");
  assert.ok(raw.test("fs.writeFileSync(`${fp}.gz`, zlib.gzipSync(fs.readFileSync(fp)));"), "the pattern still matches the line it forbids");
});
