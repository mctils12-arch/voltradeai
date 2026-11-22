/**
 * tfr_list_gate1.ts — LADDER GATE 1 (DATA) for the flight restrictions read
 * from the FAA's public list (server/faaTfrList.ts).
 *
 * TWO QUESTIONS.
 *  A. OUTLINE: when our reader copies a notice's outline out of the FAA's
 *     detail file (abdMergedArea), is it the outline the FAA's own shape
 *     service draws for that notice? Two FAA publications of one shape, read
 *     by two different pieces of our code: a disagreement is our parser's
 *     fault (corner order, hemisphere, a dropped area).
 *  B. COVERAGE: of the notices the FAA lists, how many does the reader hold
 *     with an outline, how many does the FAA publish no outline for, and how
 *     many could not be read? With `--served URL`, the same count is made on
 *     what a running server answers (GET /api/data/swim/notam/tfrs), so the
 *     check is end to end.
 *
 * GROUND TRUTH, all public, no key:
 *   list    https://tfr.faa.gov/tfrapi/exportTfrList
 *   detail  https://tfr.faa.gov/download/detail_<y>_<nnnn>.xml
 *   shape   https://tfr.faa.gov/geoserver/TFR/ows (V_TFR_LOC)
 *
 * MEASURE and TOLERANCE for A: those of scripts/tfr_text_gate1.ts, unchanged
 * (larger one-way corner-to-outline distance; 0.1 NM plus 0.4% of the FAA
 * shape's half-diagonal). Fixed before the first run.
 *
 * VERDICT A per notice with a shape-service polygon:
 *   agree     every shape-service polygon has an outline of ours within tolerance, and the counts are equal
 *   partial   some polygon matched, another did not, or the counts differ
 *   not_read  the detail file gave us no outline
 *   disagree  we hold outlines and none is within tolerance
 * VERDICT B: PASS when no detail file is unread and every listed notice
 *   either is held with an outline or is one the FAA's file has no outline for.
 *
 * Usage: npx tsx scripts/tfr_list_gate1.ts [--cache DIR] [--out FILE] [--served URL]
 * Touches no runtime state. Prints the JSON result.
 */
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { parseFaaList, parseFaaDetail, faaTfrDetailUrl, FAA_TFR_LIST_URL, type FaaTfrRow } from "../server/faaTfrList";
import { outlineDistanceNm, toleranceNm, shapesByNotice, type Ring } from "./tfr_text_gate1";

export type ListVerdict = "agree" | "partial" | "not_read" | "disagree";
export interface ListNoticeResult { notice: string; faa_shapes: number; ours: number; matched: number; verdict: ListVerdict; worst_matched_nm: number | null }

/** Compare one notice: the shape service's polygons against the outlines copied from its detail file. */
export function judgeListNotice(notice: string, shapes: Ring[], row: FaaTfrRow | null): ListNoticeResult {
  const ours: Ring[] = row?.geometry?.rings ?? [];
  const d = shapes.map((f) => (ours.length ? Math.min(...ours.map((r) => outlineDistanceNm(f, r))) : Infinity));
  const ok = shapes.map((f, i) => d[i] <= toleranceNm(f));
  const matched = ok.filter(Boolean).length;
  const verdict: ListVerdict = !ours.length ? "not_read"
    : matched === shapes.length && ours.length === shapes.length ? "agree"
    : matched > 0 ? "partial" : "disagree";
  const md = d.filter((_, i) => ok[i]);
  return { notice, faa_shapes: shapes.length, ours: ours.length, matched, verdict, worst_matched_nm: md.length ? Math.round(Math.max(...md) * 1000) / 1000 : null };
}

async function cached(dir: string | null, name: string, url: string): Promise<string> {
  const file = dir ? path.join(dir, name) : null;
  if (file && fs.existsSync(file) && fs.statSync(file).size > 200) return fs.readFileSync(file, "utf8");
  const res = await fetch(url, { signal: AbortSignal.timeout(40_000) });
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const body = await res.text();
  if (file) { fs.mkdirSync(dir!, { recursive: true }); fs.writeFileSync(file, body); }
  return body;
}

async function main(): Promise<void> {
  const arg = (n: string) => { const i = process.argv.indexOf(n); return i > 0 ? process.argv[i + 1] ?? null : null; };
  const dir = arg("--cache"), outFile = arg("--out"), served = arg("--served");
  const { entries, malformed } = parseFaaList(JSON.parse((await cached(dir, "list.json", FAA_TFR_LIST_URL)).replace(/^﻿/, "")));
  const shapes = shapesByNotice(JSON.parse((await cached(dir, "wfs.json",
    "https://tfr.faa.gov/geoserver/TFR/ows?service=WFS&version=1.1.0&request=GetFeature&typeName=TFR:V_TFR_LOC&maxFeatures=1000&outputFormat=application/json&srsname=EPSG:4326")).replace(/^﻿/, "")));
  const rows = new Map<string, FaaTfrRow>();
  const unread: string[] = [];
  for (const e of entries) {
    const name = `detail_${e.notice.replace("/", "_")}.xml`;
    const row = await cached(dir, name, faaTfrDetailUrl(e.notice)).then((x) => parseFaaDetail(x, e), (err: unknown) => {
      console.error(`${e.notice}: detail file not read (${String(err)})`);
      return null;
    });
    if (row) rows.set(e.notice, row); else unread.push(e.notice);
  }
  const judged = entries.filter((e) => (shapes.get(e.notice) ?? []).length).map((e) => judgeListNotice(e.notice, shapes.get(e.notice)!, rows.get(e.notice) ?? null));
  const count = (v: ListVerdict) => judged.filter((r) => r.verdict === v).length;
  const withOutline = entries.filter((e) => rows.get(e.notice)?.geometry).map((e) => e.notice);
  const noOutline = entries.filter((e) => { const r = rows.get(e.notice); return r && !r.geometry; }).map((e) => e.notice);
  // a notice the shape service draws and the detail file has no outline for would be a hole in the reader
  const shapeButNoOutline = noOutline.filter((n) => (shapes.get(n) ?? []).length);
  let servedBlock: Record<string, unknown> | null = null;
  if (served) {
    const body = (await (await fetch(served, { signal: AbortSignal.timeout(40_000) })).json()) as { tfrs?: Array<{ citation?: string; source?: string; start?: number | null; end?: number | null }>; faa_list?: unknown; at?: number };
    const now = body.at ?? Date.now();
    const have = new Set((body.tfrs ?? []).filter((t) => t.source === "faa_list").map((t) => String(t.citation ?? "").replace(/^FDC /, "")));
    // what the route should be showing: an outline, in force or starting within 6 hours
    const due = withOutline.filter((n) => { const r = rows.get(n)!; return (r.end === null || r.end >= now) && (r.start === null || r.start <= now + 6 * 3600_000); });
    servedBlock = { url: served, due_now: due.length, served_of_due: due.filter((n) => have.has(n)).length, missing: due.filter((n) => !have.has(n)), served_not_due: Array.from(have).filter((n) => !due.includes(n)), faa_list: body.faa_list ?? null };
  }
  const result = {
    at: new Date().toISOString(),
    faa_listed: entries.length, list_rows_malformed: malformed,
    outline: {
      compared: judged.length, agree: count("agree"), partial: count("partial"), not_read: count("not_read"), disagree: count("disagree"),
      shape_service_polygons: judged.reduce((s, r) => s + r.faa_shapes, 0), matched: judged.reduce((s, r) => s + r.matched, 0),
      worst_matched_nm: judged.reduce((m, r) => Math.max(m, r.worst_matched_nm ?? 0), 0),
      tolerance: "0.1 NM + 0.4% of the shape-service polygon's half-diagonal",
    },
    coverage: {
      with_outline: withOutline.length, no_outline_in_faa_file: noOutline, detail_unread: unread,
      shape_service_has_one_we_lack: shapeButNoOutline,
      verdict: unread.length === 0 && shapeButNoOutline.length === 0 ? "PASS" : "FAIL",
    },
    served: servedBlock,
    notices: judged,
  };
  const json = JSON.stringify(result, null, 1);
  if (outFile) fs.writeFileSync(outFile, json + "\n");
  console.log(json);
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) {
  main().catch((e) => { console.error(String(e)); process.exit(1); });
}
