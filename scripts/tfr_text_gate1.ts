/**
 * tfr_text_gate1.ts — LADDER GATE 1 (DATA) for the outline read from a
 * flight-restriction notice's text (server/swimNotamText.ts).
 *
 * QUESTION: when our reader turns a notice's words into an outline, is it
 * the outline the FAA itself publishes for that notice?
 *
 * GROUND TRUTH, all public, no key:
 *   list    https://tfr.faa.gov/tfrapi/exportTfrList            every restriction the FAA lists
 *   text    https://tfr.faa.gov/download/detail_<y>_<nnnn>.xml  txtDescrUSNS = the notice as issued
 *   shape   https://tfr.faa.gov/geoserver/TFR/ows (V_TFR_LOC)   the FAA's own polygon(s) per notice
 * The text in the detail file is the same string the FAA notice feed sends
 * (compared on FDC 6/9306, 2026-11-22: identical). The shape service is
 * the FAA's drawing of that text, made by its own system, not by ours.
 *
 * MEASURE: for each FAA polygon, the nearest of our outlines, by the
 * larger of the two one-way corner-to-outline distances (every FAA corner
 * to our outline, every corner of ours to the FAA's), in nautical miles.
 * TOLERANCE, fixed before the first run: 0.1 NM plus 0.4% of the shape's
 * half-diagonal. The second term is the chord sag of the FAA's own
 * 36-sided circles (1 - cos 5 deg = 0.38%): a 30 NM circle drawn with 36
 * sides is 0.11 NM inside the true circle at each side's middle.
 *
 * VERDICT per notice the FAA publishes a shape for:
 *   agree      every FAA polygon has an outline of ours within tolerance, and we read no area we could not finish
 *   partial    at least one FAA polygon matched, and another did not or an area was left unread
 *   not_read   we produced no outline (the text does not use the wording we read, or the list is cut)
 *   disagree   we produced outlines and none is within tolerance
 * A notice with no FAA shape is reported apart (nothing to compare to).
 *
 * Usage:  npx tsx scripts/tfr_text_gate1.ts [--cache DIR] [--out FILE]
 *   --cache DIR  read list.json / wfs.json / detail_*.xml from DIR when present, save them there when fetched
 *   --out FILE   also write the JSON result to FILE
 * Touches no runtime state. Prints the JSON result.
 */
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { textAreas } from "../server/swimNotamText";

// the measure, the tolerance and the verdict rule live in server/faaTfrTextCheck.ts, where the same check
// runs on production for every notice the feed and the FAA's list both hold (2026-11-23): one rule, two uses
import { judgeOutlines, outlineDistanceNm, halfDiagonalNm, toleranceNm, type Ring } from "../server/faaTfrTextCheck";
export { outlineDistanceNm, halfDiagonalNm, toleranceNm, type Ring };

export type Verdict = "agree" | "partial" | "not_read" | "disagree";
export interface NoticeResult {
  notice: string; type: string | null; faa_shapes: number; areas_in_text: number; read: number; unread: number;
  verdict: Verdict; matched: number; worst_matched_nm: number | null; distances_nm: number[];
}
/** Compare one notice: the FAA's polygons against the outlines read from its text. */
export function judgeNotice(notice: string, type: string | null, faa: Ring[], text: string): NoticeResult {
  const t = textAreas(text);
  const j = judgeOutlines(faa, t.rings, t.unread);
  return {
    notice, type, faa_shapes: faa.length, areas_in_text: t.areasInText, read: t.rings.length, unread: t.unread, verdict: j.verdict, matched: j.matched,
    worst_matched_nm: j.worst_matched_nm, distances_nm: j.distances_nm,
  };
}

const unescapeXml = (s: string) => s.replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"').replace(/&apos;/g, "'").replace(/&amp;/g, "&");
/** the notice as issued, from an FAA detail file */
export function detailText(xml: string): string | null {
  const m = /<txtDescrUSNS>([\s\S]*?)<\/txtDescrUSNS>/.exec(xml);
  return m ? unescapeXml(m[1]) : null;
}
/** FAA shape service answer -> polygons by notice number ("6/9306") */
export function shapesByNotice(wfs: unknown): Map<string, Ring[]> {
  const out = new Map<string, Ring[]>();
  const feats = (wfs as { features?: unknown[] } | null)?.features;
  for (const f of Array.isArray(feats) ? feats : []) {
    const o = f as { properties?: { NOTAM_KEY?: unknown }; geometry?: { type?: unknown; coordinates?: unknown } };
    const key = typeof o.properties?.NOTAM_KEY === "string" ? o.properties.NOTAM_KEY.split("-")[0] : null;
    const g = o.geometry;
    if (!key || !g) continue;
    const polys = g.type === "Polygon" ? [g.coordinates] : g.type === "MultiPolygon" ? (g.coordinates as unknown[]) : [];
    for (const p of polys) {
      const outer = Array.isArray(p) ? (p[0] as unknown) : null;
      if (!Array.isArray(outer) || outer.length < 4) continue;
      const ring = (outer as number[][]).map((c) => [c[1], c[0]] as [number, number]);
      out.set(key, [...(out.get(key) ?? []), ring]);
    }
  }
  return out;
}

async function cached(dir: string | null, name: string, url: string): Promise<string> {
  const file = dir ? path.join(dir, name) : null;
  if (file && fs.existsSync(file) && fs.statSync(file).size > 200) return fs.readFileSync(file, "utf8").replace(/^﻿/, "");
  const res = await fetch(url, { signal: AbortSignal.timeout(40_000) });
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const body = (await res.text()).replace(/^﻿/, "");
  if (file) { fs.mkdirSync(dir!, { recursive: true }); fs.writeFileSync(file, body); }
  return body;
}

async function main(): Promise<void> {
  const arg = (n: string) => { const i = process.argv.indexOf(n); return i > 0 ? process.argv[i + 1] ?? null : null; };
  const dir = arg("--cache"), outFile = arg("--out");
  const list = JSON.parse(await cached(dir, "list.json", "https://tfr.faa.gov/tfrapi/exportTfrList")) as Array<{ notam_id: string; type?: string }>;
  const shapes = shapesByNotice(JSON.parse(await cached(dir, "wfs.json",
    "https://tfr.faa.gov/geoserver/TFR/ows?service=WFS&version=1.1.0&request=GetFeature&typeName=TFR:V_TFR_LOC&maxFeatures=1000&outputFormat=application/json&srsname=EPSG:4326")));
  const rows: NoticeResult[] = [];
  const noFaaShape: Array<{ notice: string; type: string | null; areas_in_text: number; read: number }> = [];
  const failed: string[] = [];
  for (const it of list) {
    const name = `detail_${it.notam_id.replace("/", "_")}.xml`;
    const text = await cached(dir, name, `https://tfr.faa.gov/download/${name}`).then(detailText, (e: unknown) => {
      console.error(`${it.notam_id}: detail file not read (${String(e)})`);
      return null;
    });
    if (text == null) { failed.push(it.notam_id); continue; }
    const faa = shapes.get(it.notam_id) ?? [];
    if (!faa.length) { const t = textAreas(text); noFaaShape.push({ notice: it.notam_id, type: it.type ?? null, areas_in_text: t.areasInText, read: t.rings.length }); continue; }
    rows.push(judgeNotice(it.notam_id, it.type ?? null, faa, text));
  }
  const count = (v: Verdict) => rows.filter((r) => r.verdict === v).length;
  const read = rows.filter((r) => r.read > 0);
  const result = {
    at: new Date().toISOString(),
    faa_listed: list.length, text_unavailable: failed, with_faa_shape: rows.length, without_faa_shape: noFaaShape.length,
    agree: count("agree"), partial: count("partial"), not_read: count("not_read"), disagree: count("disagree"),
    read_share: rows.length ? Math.round((read.length / rows.length) * 1000) / 1000 : null,
    agree_share_of_read: read.length ? Math.round((count("agree") / read.length) * 1000) / 1000 : null,
    faa_polygons: rows.reduce((s, r) => s + r.faa_shapes, 0),
    faa_polygons_matched: rows.reduce((s, r) => s + r.matched, 0),
    tolerance: "0.1 NM + 0.4% of the FAA shape's half-diagonal",
    notices: rows, no_faa_shape: noFaaShape,
  };
  const json = JSON.stringify(result, null, 1);
  if (outFile) fs.writeFileSync(outFile, json + "\n");
  console.log(json);
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) {
  main().catch((e) => { console.error(String(e)); process.exit(1); });
}
