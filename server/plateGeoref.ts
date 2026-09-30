// PLATE GEOREFERENCE — "plate on the map" (ForeFlight look) for FAA d-TPP
// instrument procedure charts, built from the chart's OWN vector text.
//
// FAA terminal procedure PDFs are generated vector documents: every fix name
// on the plan view (DOFFS, JEDYE, HOOKK, "RRTOO INT", …) is a real text
// object in a standard Type1 font (Times-Roman / WinAnsi), so its position on
// the page is exact. We parse the page's content streams (a small, bounded
// PDF reader — no new dependency: zlib is Node's), match those labels to the
// same fixes' CIFP coordinates, and fit a similarity transform (scale +
// rotation + translation) page -> ground with RANSAC outlier rejection.
//
// HONESTY RULES
//  * A plate is marked `georeferenced: true` only when >= GEOREF_MIN_POINTS
//    distinct fixes agree with a residual RMS under GEOREF_MAX_RMS_NM. The
//    residual is REPORTED, never hidden. Labels sit beside their symbols, so
//    the residual includes label offset — it is an upper bound on the fit
//    error, not a claim of sub-label precision.
//  * Charts that are NOT TO SCALE (most STAR/DP plan views, insets) fail the
//    fit by construction (no consistent scale) and stay side-viewer only.
//  * Many FAA PDFs also embed an OGC GeoPDF viewport (/VP /Measure /GPTS).
//    When present it is parsed and compared with the text fit (reported as
//    `embeddedAgreementNm`) — an independent second opinion, never a
//    substitute for the text-fit gate.
//
// The plan-view crop box is found from the page's own long horizontal /
// vertical ruled lines: the smallest ruled rectangle that contains the
// matched fix labels (the plan view is always boxed; the profile and
// minimums panels sit in separate boxes below it).

import zlib from "zlib";

// ── bounded PDF reader ──────────────────────────────────────────────────────

export const PDF_MAX_BYTES = 12 * 1024 * 1024;
export const PDF_MAX_OBJECTS = 20_000;
export const PDF_MAX_STREAM_BYTES = 8 * 1024 * 1024;
export const CONTENT_MAX_TOKENS = 3_000_000;

export interface PdfObject { num: number; dict: string; stream: Buffer | null }

function inflateStream(dict: string, raw: Buffer): Buffer | null {
  if (!/\/Filter\s*(\[\s*)?\/FlateDecode/.test(dict)) {
    return /\/Filter/.test(dict) ? null : raw; // unsupported filter -> no guess
  }
  try {
    return zlib.inflateSync(raw, { maxOutputLength: PDF_MAX_STREAM_BYTES });
  } catch (e: unknown) {
    // some producers leave trailing bytes after the zlib stream
    try {
      return zlib.inflateSync(raw, { finishFlush: zlib.constants.Z_SYNC_FLUSH, maxOutputLength: PDF_MAX_STREAM_BYTES });
    } catch (e2: unknown) {
      void e; void e2;
      return null;
    }
  }
}

/** Scan every `N G obj … endobj` (plus objects packed in /ObjStm streams). */
export function parsePdfObjects(buf: Buffer): Map<number, PdfObject> {
  if (buf.length > PDF_MAX_BYTES) throw new Error(`pdf too large (${buf.length} bytes)`);
  const s = buf.toString("latin1");
  const out = new Map<number, PdfObject>();
  const re = /(\d+)\s+(\d+)\s+obj\b/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(s)) && out.size < PDF_MAX_OBJECTS) {
    const num = Number(m[1]);
    const start = m.index + m[0].length;
    const end = s.indexOf("endobj", start);
    if (end < 0) break;
    const body = s.slice(start, end);
    const si = body.search(/\bstream\r?\n/);
    let dict = body;
    let stream: Buffer | null = null;
    if (si >= 0) {
      dict = body.slice(0, si);
      const nl = body.indexOf("\n", si) + 1;
      let se = body.lastIndexOf("endstream");
      if (se < nl) se = body.length;
      let raw = buf.subarray(start + nl, start + se);
      // strip the EOL before `endstream`
      while (raw.length && (raw[raw.length - 1] === 0x0a || raw[raw.length - 1] === 0x0d)) raw = raw.subarray(0, raw.length - 1);
      stream = inflateStream(dict, raw);
    }
    out.set(num, { num, dict, stream });
    re.lastIndex = end;
  }
  // unpack object streams (PDF 1.5+)
  for (const o of Array.from(out.values())) {
    if (!o.stream || !/\/Type\s*\/ObjStm/.test(o.dict)) continue;
    const n = Number(/\/N\s+(\d+)/.exec(o.dict)?.[1] ?? 0);
    const first = Number(/\/First\s+(\d+)/.exec(o.dict)?.[1] ?? 0);
    const txt = o.stream.toString("latin1");
    const hdr = txt.slice(0, first).trim().split(/\s+/).map(Number);
    for (let i = 0; i < n && i * 2 + 1 < hdr.length; i++) {
      const on = hdr[i * 2], off = hdr[i * 2 + 1];
      const next = i + 1 < n ? hdr[(i + 1) * 2 + 1] : txt.length - first;
      if (!out.has(on)) out.set(on, { num: on, dict: txt.slice(first + off, first + next), stream: null });
    }
  }
  return out;
}

const refsIn = (s: string): number[] => Array.from(s.matchAll(/(\d+)\s+\d+\s+R/g), (x) => Number(x[1]));

/** Resolve a `/Key value` that may be inline or an indirect reference. */
function dictValue(dict: string, key: string, objs: Map<number, PdfObject>): string | null {
  const i = dict.search(new RegExp(`/${key}(?![A-Za-z0-9])`));
  if (i < 0) return null;
  const rest = dict.slice(i + key.length + 1).trimStart();
  const ref = /^(\d+)\s+\d+\s+R/.exec(rest);
  if (ref) return objs.get(Number(ref[1]))?.dict ?? null;
  return rest;
}

/** Take one balanced PDF value (array / dict / scalar) from the head of s. */
function takeValue(s: string): string {
  s = s.trimStart();
  const open = s.startsWith("<<") ? "<<" : s.startsWith("[") ? "[" : null;
  if (!open) return (/^[^\s/\]>]*|^\/[^\s/\]>[(<]*/.exec(s) || [""])[0];
  const close = open === "<<" ? ">>" : "]";
  let depth = 0;
  for (let i = 0; i < s.length; i++) {
    if (s.startsWith(open, i)) { depth++; i += open.length - 1; } else if (s.startsWith(close, i)) {
      depth--; i += close.length - 1;
      if (depth === 0) return s.slice(0, i + 1);
    } else if (s[i] === "(") { // skip literal strings (may hold brackets)
      let d = 0;
      for (; i < s.length; i++) {
        if (s[i] === "\\") { i++; continue; }
        if (s[i] === "(") d++;
        else if (s[i] === ")" && --d === 0) break;
      }
    }
  }
  return s;
}

const nums = (s: string | null | undefined): number[] =>
  s ? Array.from(s.matchAll(/-?\d*\.?\d+(?:[eE][-+]?\d+)?/g), (x) => Number(x[0])) : [];

export interface PdfPage {
  width: number;
  height: number;
  mediaBox: [number, number, number, number];
  rotate: number;
  contents: Buffer[];
  /** embedded OGC GeoPDF viewport(s): bbox in page points + corner lat/lons */
  geo: EmbeddedGeo[];
}
export interface EmbeddedGeo {
  bbox: [number, number, number, number];
  /** LPTS: fractions of bbox, pairs (x,y) */
  lpts: number[];
  /** GPTS: lat,lon pairs */
  gpts: number[];
  wkt: string | null;
}

export function firstPage(objs: Map<number, PdfObject>): PdfPage | null {
  const page = Array.from(objs.values()).find((o) => /\/Type\s*\/Page(?![s\w])/.test(o.dict));
  if (!page) return null;
  const mb = nums(takeValue(dictValue(page.dict, "MediaBox", objs) ?? "[0 0 612 792]"));
  const mediaBox: [number, number, number, number] = mb.length >= 4 ? [mb[0], mb[1], mb[2], mb[3]] : [0, 0, 612, 792];
  const rotate = Number(/\/Rotate\s+(-?\d+)/.exec(page.dict)?.[1] ?? 0);
  const cRaw = takeValue(page.dict.slice(page.dict.search(/\/Contents/) + 9));
  const contents: Buffer[] = [];
  for (const r of refsIn(cRaw)) {
    const o = objs.get(r);
    if (o?.stream) contents.push(o.stream);
    else if (o && /^\s*\[/.test(o.dict)) for (const rr of refsIn(o.dict)) { const oo = objs.get(rr); if (oo?.stream) contents.push(oo.stream); }
  }
  const geo: EmbeddedGeo[] = [];
  const vpi = page.dict.search(/\/VP\s*\[/);
  if (vpi >= 0) {
    const vp = takeValue(page.dict.slice(vpi + 3));
    for (const block of vp.split(/\/BBox/).slice(1)) {
      const bbox = nums(takeValue(block));
      const gpts = nums(takeValue(block.slice(block.indexOf("/GPTS") + 5)));
      const lpts = nums(takeValue(block.slice(block.indexOf("/LPTS") + 5)));
      const wkt = /\/WKT\s*\(((?:\\.|[^\\)])*)\)/.exec(block)?.[1]?.replace(/\\([()\\])/g, "$1") ?? null;
      if (bbox.length === 4 && gpts.length >= 6 && lpts.length === gpts.length && block.includes("/GPTS")) {
        geo.push({ bbox: [bbox[0], bbox[1], bbox[2], bbox[3]], gpts, lpts, wkt });
      }
    }
  }
  return {
    width: mediaBox[2] - mediaBox[0], height: mediaBox[3] - mediaBox[1],
    mediaBox, rotate, contents, geo,
  };
}

// ── content-stream interpreter (text positions + ruled lines) ───────────────

export interface TextItem { text: string; x: number; y: number; /** page units per character (approx) */ charW: number; size: number; angle: number }
export interface Segment { x1: number; y1: number; x2: number; y2: number }

type M6 = [number, number, number, number, number, number];
const mul = (a: M6, b: M6): M6 => [
  a[0] * b[0] + a[1] * b[2], a[0] * b[1] + a[1] * b[3],
  a[2] * b[0] + a[3] * b[2], a[2] * b[1] + a[3] * b[3],
  a[4] * b[0] + a[5] * b[2] + b[4], a[4] * b[1] + a[5] * b[3] + b[5],
];
const apply = (m: M6, x: number, y: number): [number, number] => [m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]];

const WINANSI_EXTRA: Record<number, string> = { 0xb0: "°", 0x96: "-", 0x97: "-", 0x92: "'" };

function decodeLiteral(raw: string): string {
  let out = "";
  for (let i = 0; i < raw.length; i++) {
    const c = raw[i];
    if (c !== "\\") { out += c; continue; }
    const n = raw[++i];
    if (n === undefined) break;
    if (n === "n") out += "\n"; else if (n === "r") out += "\r"; else if (n === "t") out += "\t";
    else if (n === "b") out += "\b"; else if (n === "f") out += "\f";
    else if (/[0-7]/.test(n)) {
      let oct = n;
      while (oct.length < 3 && /[0-7]/.test(raw[i + 1] ?? "")) oct += raw[++i];
      out += String.fromCharCode(parseInt(oct, 8));
    } else if (n === "\r" || n === "\n") { if (n === "\r" && raw[i + 1] === "\n") i++; } else out += n;
  }
  return Array.from(out, (ch) => WINANSI_EXTRA[ch.charCodeAt(0)] ?? ch).join("");
}

type Tok = { k: "num"; v: number } | { k: "str"; v: string } | { k: "name"; v: string } | { k: "op"; v: string } | { k: "arr"; v: Tok[] } | { k: "dict" };

/** Tokenize one content stream. Inline images (BI … ID … EI) are skipped. */
export function tokenize(src: string, maxTokens = CONTENT_MAX_TOKENS): Tok[] {
  const out: Tok[] = [];
  const stack: Tok[][] = [];
  let cur = out;
  let i = 0;
  const n = src.length;
  let count = 0;
  const WS = /\s/;
  const DELIM = /[\s()<>[\]{}/%]/;
  while (i < n && count++ < maxTokens) {
    const c = src[i];
    if (WS.test(c)) { i++; continue; }
    if (c === "%") { while (i < n && src[i] !== "\n" && src[i] !== "\r") i++; continue; }
    if (c === "(") {
      let d = 0; const st = i + 1;
      for (; i < n; i++) {
        if (src[i] === "\\") { i++; continue; }
        if (src[i] === "(") d++;
        else if (src[i] === ")" && --d === 0) break;
      }
      cur.push({ k: "str", v: decodeLiteral(src.slice(st, i)) }); i++; continue;
    }
    if (c === "<" && src[i + 1] === "<") { // inline dict (BDC properties): skip balanced
      let d = 0;
      for (; i < n; i++) {
        if (src.startsWith("<<", i)) { d++; i++; } else if (src.startsWith(">>", i)) { d--; i++; if (d === 0) { i++; break; } }
      }
      cur.push({ k: "dict" }); continue;
    }
    if (c === "<") {
      const e = src.indexOf(">", i);
      const hex = src.slice(i + 1, e < 0 ? n : e).replace(/\s/g, "");
      let s = "";
      for (let h = 0; h < hex.length; h += 2) s += String.fromCharCode(parseInt((hex.slice(h, h + 2) + "0").slice(0, 2), 16));
      cur.push({ k: "str", v: s }); i = e < 0 ? n : e + 1; continue;
    }
    if (c === "[") { stack.push(cur); const a: Tok[] = []; cur.push({ k: "arr", v: a }); cur = a; i++; continue; }
    if (c === "]") { cur = stack.pop() ?? out; i++; continue; }
    if (c === "/") {
      let j = i + 1; while (j < n && !DELIM.test(src[j])) j++;
      cur.push({ k: "name", v: src.slice(i + 1, j) }); i = j; continue;
    }
    let j = i; while (j < n && !DELIM.test(src[j])) j++;
    if (j === i) { i++; continue; }
    const w = src.slice(i, j);
    i = j;
    if (/^[-+]?(\d+\.?\d*|\.\d+)$/.test(w)) { cur.push({ k: "num", v: Number(w) }); continue; }
    if (w === "ID") { // inline image data: skip to EI
      const e = src.indexOf("EI", i);
      i = e < 0 ? n : e + 2;
      continue;
    }
    cur.push({ k: "op", v: w });
  }
  return out;
}

/** a small painted shape (fix symbol candidate): bbox centre + size, page units */
export interface Mark { x: number; y: number; w: number; h: number; filled: boolean }
export interface PageContent { texts: TextItem[]; segments: Segment[]; marks: Mark[] }

export const MARK_MIN_PT = 1.2;
export const MARK_MAX_PT = 16;

/** Average advance of a Times-Roman capital (1000-unit em): ~0.66. FAA fix
 *  labels are all-caps; this estimates the label's centre, which only feeds
 *  a fit whose residual is reported. */
export const CAP_ADVANCE_EM = 0.66;
export const MIN_RULE_LENGTH = 60;

export function interpretContent(streams: Buffer[]): PageContent {
  const texts: TextItem[] = [];
  const segments: Segment[] = [];
  const marks: Mark[] = [];
  let sub: Array<[number, number]> = [];
  const subs: Array<Array<[number, number]>> = [];
  let ctm: M6 = [1, 0, 0, 1, 0, 0];
  const gs: M6[] = [];
  let tm: M6 = [1, 0, 0, 1, 0, 0];
  let tlm: M6 = [1, 0, 0, 1, 0, 0];
  let size = 1, leading = 0, hscale = 1;
  let path: Segment[] = [];
  let cx = 0, cy = 0, sx = 0, sy = 0;
  const operands: Tok[] = [];
  const num = (k: number) => { const t = operands[operands.length - k]; return t && t.k === "num" ? t.v : 0; };
  const show = (s: string) => {
    if (!s) return;
    const m = mul([size * hscale, 0, 0, size, 0, 0], mul(tm, ctm));
    const [x, y] = apply(m, 0, 0);
    const ax = Math.hypot(m[0], m[1]);
    texts.push({ text: s, x, y, charW: ax * CAP_ADVANCE_EM, size: Math.hypot(m[2], m[3]), angle: Math.atan2(m[1], m[0]) * 180 / Math.PI });
    // advance (approximate): keeps successive shows on one line in order
    tm = mul([1, 0, 0, 1, s.length * CAP_ADVANCE_EM * size * hscale, 0], tm);
  };
  for (const buf of streams) {
    const toks = tokenize(buf.toString("latin1"));
    for (const t of toks) {
      if (t.k !== "op") { operands.push(t); continue; }
      switch (t.v) {
        case "q": gs.push(ctm); break;
        case "Q": ctm = gs.pop() ?? [1, 0, 0, 1, 0, 0]; break;
        case "cm": ctm = mul([num(6), num(5), num(4), num(3), num(2), num(1)], ctm); break;
        case "BT": tm = [1, 0, 0, 1, 0, 0]; tlm = tm; break;
        case "Tf": size = num(1); break;
        case "Tz": hscale = num(1) / 100; break;
        case "TL": leading = num(1); break;
        case "Tm": tm = [num(6), num(5), num(4), num(3), num(2), num(1)]; tlm = tm; break;
        case "Td": tlm = mul([1, 0, 0, 1, num(2), num(1)], tlm); tm = tlm; break;
        case "TD": leading = -num(1); tlm = mul([1, 0, 0, 1, num(2), num(1)], tlm); tm = tlm; break;
        case "T*": tlm = mul([1, 0, 0, 1, 0, -leading], tlm); tm = tlm; break;
        case "Tj": case "'": case "\"": {
          if (t.v !== "Tj") { tlm = mul([1, 0, 0, 1, 0, -leading], tlm); tm = tlm; }
          const s = operands[operands.length - 1];
          if (s && s.k === "str") show(s.v);
          break;
        }
        case "TJ": {
          const a = operands[operands.length - 1];
          if (a && a.k === "arr") {
            let acc = "";
            for (const e of a.v) {
              if (e.k === "str") acc += e.v;
              else if (e.k === "num" && e.v < -200) acc += " "; // a kerning gap wide enough to be a space
            }
            show(acc);
          }
          break;
        }
        case "m": cx = sx = num(2); cy = sy = num(1); if (sub.length) subs.push(sub); sub = [apply(ctm, cx, cy)]; break;
        case "l": { const x = num(2), y = num(1); path.push({ x1: cx, y1: cy, x2: x, y2: y }); cx = x; cy = y; sub.push(apply(ctm, x, y)); break; }
        case "c": cx = num(2); cy = num(1); sub.push(apply(ctm, num(6), num(5)), apply(ctm, num(4), num(3)), apply(ctm, cx, cy)); break;
        case "v": case "y": cx = num(2); cy = num(1); sub.push(apply(ctm, num(4), num(3)), apply(ctm, cx, cy)); break;
        case "h": path.push({ x1: cx, y1: cy, x2: sx, y2: sy }); cx = sx; cy = sy; break;
        case "re": {
          const x = num(4), y = num(3), w = num(2), h = num(1);
          path.push({ x1: x, y1: y, x2: x + w, y2: y }, { x1: x + w, y1: y, x2: x + w, y2: y + h },
            { x1: x + w, y1: y + h, x2: x, y2: y + h }, { x1: x, y1: y + h, x2: x, y2: y });
          cx = sx = x; cy = sy = y;
          if (sub.length) subs.push(sub);
          sub = [apply(ctm, x, y), apply(ctm, x + w, y + h)];
          break;
        }
        case "S": case "s": case "B": case "B*": case "b": case "b*": case "f": case "F": case "f*": {
          if (sub.length) subs.push(sub);
          const filled = t.v !== "S" && t.v !== "s";
          for (const sp of subs) {
            let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
            for (const [px, py] of sp) { x0 = Math.min(x0, px); y0 = Math.min(y0, py); x1 = Math.max(x1, px); y1 = Math.max(y1, py); }
            const w = x1 - x0, h = y1 - y0;
            if (sp.length >= 3 && Math.max(w, h) <= MARK_MAX_PT && Math.min(w, h) >= MARK_MIN_PT) {
              marks.push({ x: (x0 + x1) / 2, y: (y0 + y1) / 2, w, h, filled });
            }
          }
          sub = []; subs.length = 0;
          for (const p of path) {
            const [x1, y1] = apply(ctm, p.x1, p.y1);
            const [x2, y2] = apply(ctm, p.x2, p.y2);
            const axis = Math.abs(x1 - x2) < 0.5 || Math.abs(y1 - y2) < 0.5;
            if (axis && Math.hypot(x2 - x1, y2 - y1) >= MIN_RULE_LENGTH) segments.push({ x1, y1, x2, y2 });
          }
          path = [];
          break;
        }
        case "n": path = []; sub = []; subs.length = 0; break;
        default: break;
      }
      operands.length = 0;
    }
  }
  return { texts, segments, marks };
}

// ── geodesy helpers (local equirectangular about a centre; the plan view is
//    ~30 nm across, where this is within ~0.1 nm of a conformal projection) ──

export interface LL { lat: number; lon: number }
const D2R = Math.PI / 180;
export function toLocalNm(p: LL, c: LL): [number, number] {
  return [(p.lon - c.lon) * 60 * Math.cos(c.lat * D2R), (p.lat - c.lat) * 60];
}
export function fromLocalNm(x: number, y: number, c: LL): LL {
  return { lat: c.lat + y / 60, lon: c.lon + x / (60 * Math.cos(c.lat * D2R)) };
}

// ── similarity fit ──────────────────────────────────────────────────────────

/** q ≈ s·R(θ)·p + t (no reflection). a = s·cosθ, b = s·sinθ. */
export interface Similarity { a: number; b: number; tx: number; ty: number }
export const simApply = (T: Similarity, x: number, y: number): [number, number] =>
  [T.a * x - T.b * y + T.tx, T.b * x + T.a * y + T.ty];
export const simScale = (T: Similarity) => Math.hypot(T.a, T.b);
export const simRotationDeg = (T: Similarity) => Math.atan2(T.b, T.a) * 180 / Math.PI;
export function simInverse(T: Similarity): Similarity {
  const d = T.a * T.a + T.b * T.b;
  const a = T.a / d, b = -T.b / d;
  return { a, b, tx: -(a * T.tx - b * T.ty), ty: -(b * T.tx + a * T.ty) };
}

/** Least-squares similarity from >= 2 correspondences (closed form). */
export function fitSimilarity(src: Array<[number, number]>, dst: Array<[number, number]>): Similarity | null {
  const n = src.length;
  if (n < 2 || dst.length !== n) return null;
  let mx = 0, my = 0, nx = 0, ny = 0;
  for (let i = 0; i < n; i++) { mx += src[i][0]; my += src[i][1]; nx += dst[i][0]; ny += dst[i][1]; }
  mx /= n; my /= n; nx /= n; ny /= n;
  let sxx = 0, sab = 0, sba = 0;
  for (let i = 0; i < n; i++) {
    const px = src[i][0] - mx, py = src[i][1] - my, qx = dst[i][0] - nx, qy = dst[i][1] - ny;
    sxx += px * px + py * py;
    sab += px * qx + py * qy;
    sba += px * qy - py * qx;
  }
  if (sxx <= 0) return null;
  const a = sab / sxx, b = sba / sxx;
  return { a, b, tx: nx - (a * mx - b * my), ty: ny - (b * mx + a * my) };
}

export interface ControlCandidate { fix: string; page: [number, number]; ground: [number, number] }
export interface FitResult {
  T: Similarity;
  inliers: ControlCandidate[];
  residualsNm: number[];
  rmsNm: number;
  rejected: number;
}

export const GEOREF_MIN_POINTS = 3;
/** blind RANSAC (no embedded GeoPDF to seed the association) needs one more
 *  agreeing fix: with ~100+ label/symbol candidates, a 3-point coincidence at
 *  sub-mile residual is possible (seen on KAUS ILS 18L in development) */
export const GEOREF_MIN_POINTS_UNSEEDED = 4;
export const GEOREF_MAX_RMS_NM = 0.5;
/** a candidate within this many nm of the model is an inlier. Control
 *  points are fix SYMBOL centres (not labels), so a true match sits well
 *  inside this; the gate itself is GEOREF_MAX_RMS_NM on the survivors. */
export const RANSAC_INLIER_NM = 0.6;
/** plausible plan-view scale window, nm per page point (1 in = 72 pt):
 *  ~1 nm/in (large-scale insets) to ~40 nm/in (enroute-ish) */
export const SCALE_MIN_NM_PER_PT = 1 / 72;
export const SCALE_MAX_NM_PER_PT = 40 / 72;
/** FAA plan views are drawn north-up; a hypothesis rotated beyond this is
 *  a coincidence of labels, not the chart */
export const MAX_ROTATION_DEG = 3;
/** RANSAC work bound: candidate pairs are drawn from at most this many */
export const RANSAC_MAX_CANDIDATES = 200;

/**
 * RANSAC over label candidates (several labels may carry one fix name —
 * plan view and profile both print "RRTOO INT"): every pair of candidates
 * with distinct fixes proposes a similarity; the one with the most distinct-
 * fix inliers (ties -> lowest RMS) wins, then it is refit by least squares on
 * its inliers (one candidate per fix: the closest) and iterated to a fixed
 * point.
 *
 * `seed`: a prior hypothesis (the chart's embedded GeoPDF, when present)
 * used INSTEAD of pair enumeration — it only chooses which symbol answers
 * for which fix; the returned transform is still the least-squares fit to
 * those symbols, and its residual is what gets gated.
 */
export function robustFit(cands: ControlCandidate[], seed: Similarity | null = null): FitResult | null {
  const byFix = new Set(cands.map((c) => c.fix));
  if (byFix.size < 2) return null;
  const plausible = (T: Similarity) => {
    const s = simScale(T);
    return s >= SCALE_MIN_NM_PER_PT && s <= SCALE_MAX_NM_PER_PT && Math.abs(simRotationDeg(T)) <= MAX_ROTATION_DEG;
  };
  const inliersOf = (T: Similarity) => {
    const best = new Map<string, { c: ControlCandidate; r: number }>();
    for (const c of cands) {
      const [x, y] = simApply(T, c.page[0], c.page[1]);
      const r = Math.hypot(x - c.ground[0], y - c.ground[1]);
      if (r > RANSAC_INLIER_NM) continue;
      const b = best.get(c.fix);
      if (!b || r < b.r) best.set(c.fix, { c, r });
    }
    return Array.from(best.values());
  };
  let bestSet: Array<{ c: ControlCandidate; r: number }> = [];
  let bestRms = Infinity;
  const lim = seed ? 0 : Math.min(cands.length, RANSAC_MAX_CANDIDATES); // bounded work
  if (seed) bestSet = inliersOf(seed);
  for (let i = 0; i < lim; i++) {
    for (let j = i + 1; j < lim; j++) {
      if (cands[i].fix === cands[j].fix) continue;
      const T = fitSimilarity([cands[i].page, cands[j].page], [cands[i].ground, cands[j].ground]);
      if (!T || !plausible(T)) continue;
      const set = inliersOf(T);
      const rms = Math.sqrt(set.reduce((s, x) => s + x.r * x.r, 0) / Math.max(1, set.length));
      if (set.length > bestSet.length || (set.length === bestSet.length && rms < bestRms)) { bestSet = set; bestRms = rms; }
    }
  }
  if (bestSet.length < 2) return null;
  let T: Similarity | null = null;
  let set = bestSet;
  for (let iter = 0; iter < 5; iter++) {
    T = fitSimilarity(set.map((x) => x.c.page), set.map((x) => x.c.ground));
    if (!T) return null;
    const next = inliersOf(T);
    if (next.length < 2) break;
    const same = next.length === set.length && next.every((x, k) => x.c === set[k].c);
    set = next;
    if (same) break;
  }
  if (!T) return null;
  const residualsNm = set.map((x) => {
    const [px, py] = simApply(T!, x.c.page[0], x.c.page[1]);
    return Math.hypot(px - x.c.ground[0], py - x.c.ground[1]);
  });
  const rmsNm = Math.sqrt(residualsNm.reduce((s, r) => s + r * r, 0) / residualsNm.length);
  return { T, inliers: set.map((x) => x.c), residualsNm, rmsNm, rejected: cands.length - set.length };
}

// ── labels, symbols, and label -> symbol candidates ─────────────────────────

export interface FixLabel { fix: string; x: number; y: number }

/** Every occurrence of a known fix ident as a whole word in a text item,
 *  located at the word's estimated centre. Frequency lines of navaid boxes
 *  ("112.8  CWK") still count — the box is a pointer to the region. */
export function fixLabels(texts: TextItem[], fixes: Map<string, LL>): FixLabel[] {
  const out: FixLabel[] = [];
  for (const t of texts) {
    if (Math.abs(t.angle) > 3) continue; // rotated course/radial annotations
    const re = /[A-Z0-9-]{2,6}/g;
    let m: RegExpExecArray | null;
    while ((m = re.exec(t.text))) {
      const w = m[0];
      if (!fixes.has(w)) continue;
      out.push({ fix: w, x: t.x + (m.index + w.length / 2) * t.charW, y: t.y + t.size * 0.35 });
    }
  }
  return out;
}

export interface Symbol { x: number; y: number; w: number; h: number; parts: number }
/** a chart symbol is small: waypoint stars, triangles, navaid roses */
export const SYMBOL_MAX_PT = 18;
export const SYMBOL_MIN_PT = 2.5;
export const SYMBOL_JOIN_GAP_PT = 3.0;

/** Merge touching small shapes into symbols (FAA symbols are drawn in
 *  pieces: a waypoint star is two filled halves, a VORTAC three lobes plus a
 *  centre). Bounded: O(n²) over <= MARKS_MAX marks. */
export const MARKS_MAX = 6000;
/** Greedy, size-bounded grouping: largest shapes seed symbols; a shape joins
 *  a symbol it touches only while the union stays symbol-sized — so a
 *  symbol never chains into the short line pieces around it. */
export function clusterSymbols(marks: Mark[]): Symbol[] {
  const ms = marks.slice(0, MARKS_MAX).map((m) => ({ x0: m.x - m.w / 2, y0: m.y - m.h / 2, x1: m.x + m.w / 2, y1: m.y + m.h / 2 }))
    .sort((a, b) => Math.max(b.x1 - b.x0, b.y1 - b.y0) - Math.max(a.x1 - a.x0, a.y1 - a.y0));
  const cl: Array<{ x0: number; y0: number; x1: number; y1: number; parts: number }> = [];
  const g = SYMBOL_JOIN_GAP_PT;
  for (const m of ms) {
    let joined = false;
    for (const c of cl) {
      if (m.x0 > c.x1 + g || m.x1 < c.x0 - g || m.y0 > c.y1 + g || m.y1 < c.y0 - g) continue;
      const u = { x0: Math.min(c.x0, m.x0), y0: Math.min(c.y0, m.y0), x1: Math.max(c.x1, m.x1), y1: Math.max(c.y1, m.y1) };
      if (Math.max(u.x1 - u.x0, u.y1 - u.y0) > SYMBOL_MAX_PT) continue;
      Object.assign(c, u); c.parts++; joined = true; break;
    }
    if (!joined) cl.push({ ...m, parts: 1 });
  }
  return cl
    .filter((c) => Math.max(c.x1 - c.x0, c.y1 - c.y0) >= SYMBOL_MIN_PT)
    .map((c) => ({ x: (c.x0 + c.x1) / 2, y: (c.y0 + c.y1) / 2, w: c.x1 - c.x0, h: c.y1 - c.y0, parts: c.parts }));
}

/** symbols considered per label: the nearest few within this radius */
export const LABEL_SYMBOL_RADIUS_PT = 70;
export const LABEL_SYMBOL_K = 6;
/** with a GeoPDF seed the association is checked against a prior, so a
 *  crowded label may nominate more symbols without inviting coincidences */
export const LABEL_SYMBOL_K_SEEDED = 16;

/** Control-point candidates: for each fix label, its nearest symbols. The
 *  label only nominates; the SYMBOL centre is the control point (labels sit
 *  ~0.2-0.5 in beside their fix — up to several nm at plate scale). */
export function symbolCandidates(labels: FixLabel[], symbols: Symbol[], fixes: Map<string, LL>, center: LL, k = LABEL_SYMBOL_K): ControlCandidate[] {
  const out: ControlCandidate[] = [];
  const seen = new Set<string>();
  for (const l of labels) {
    const f = fixes.get(l.fix);
    if (!f) continue;
    const near = symbols
      .map((s) => ({ s, d: Math.hypot(s.x - l.x, s.y - l.y) }))
      .filter((x) => x.d <= LABEL_SYMBOL_RADIUS_PT)
      .sort((a, b) => a.d - b.d)
      .slice(0, k);
    for (const { s } of near) {
      const key = `${l.fix}|${s.x.toFixed(1)}|${s.y.toFixed(1)}`;
      if (seen.has(key)) continue;
      seen.add(key);
      out.push({ fix: l.fix, page: [s.x, s.y], ground: toLocalNm(f, center) });
    }
  }
  return out;
}

// ── plan-view crop box ──────────────────────────────────────────────────────

export interface Box { x0: number; y0: number; x1: number; y1: number }

/** Smallest ruled rectangle (from long axis-aligned lines) containing every
 *  point; falls back to the points' hull padded by 15% when no enclosing
 *  rules exist. */
export function planViewBox(segments: Segment[], pts: Array<[number, number]>, page: { width: number; height: number }): Box {
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const horiz = segments.filter((s) => Math.abs(s.y1 - s.y2) < 0.5).map((s) => ({ y: s.y1, xa: Math.min(s.x1, s.x2), xb: Math.max(s.x1, s.x2) }));
  const vert = segments.filter((s) => Math.abs(s.x1 - s.x2) < 0.5).map((s) => ({ x: s.x1, ya: Math.min(s.y1, s.y2), yb: Math.max(s.y1, s.y2) }));
  const spans = (h: { xa: number; xb: number }) => h.xa <= minX + 1 && h.xb >= maxX - 1;
  const top = horiz.filter((h) => h.y > maxY && spans(h)).sort((a, b) => a.y - b.y)[0];
  const bottom = horiz.filter((h) => h.y < minY && spans(h)).sort((a, b) => b.y - a.y)[0];
  const vspans = (v: { ya: number; yb: number }) => v.ya <= minY + 1 && v.yb >= maxY - 1;
  const left = vert.filter((v) => v.x < minX && vspans(v)).sort((a, b) => b.x - a.x)[0];
  const right = vert.filter((v) => v.x > maxX && vspans(v)).sort((a, b) => a.x - b.x)[0];
  const padX = (maxX - minX) * 0.15 + 10, padY = (maxY - minY) * 0.15 + 10;
  return {
    x0: left ? left.x : Math.max(0, minX - padX),
    x1: right ? right.x : Math.min(page.width, maxX + padX),
    y0: bottom ? bottom.y : Math.max(0, minY - padY),
    y1: top ? top.y : Math.min(page.height, maxY + padY),
  };
}

// ── embedded GeoPDF (second opinion) ────────────────────────────────────────

/** Lambert Conformal Conic (2SP, ellipsoidal — Snyder 1987 §15) built from
 *  the GeoPDF's own WKT (spheroid + parallels + origin read from the file,
 *  never assumed). Returns null when the WKT is not an LCC we can read. */
export function lccFromWkt(wkt: string | null): ((p: LL) => [number, number]) | null {
  if (!wkt || !/Lambert_Conformal_Conic/i.test(wkt)) return null;
  const sph = /SPHEROID\[\s*"[^"]*"\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)/i.exec(wkt);
  const par = (name: string) => {
    const m = new RegExp(`PARAMETER\\[\\s*"${name}"\\s*,\\s*([-\\d.]+)`, "i").exec(wkt);
    return m ? Number(m[1]) : null;
  };
  const lat0 = par("Latitude_Of_Origin"), lon0 = par("Central_Meridian"), p1 = par("Standard_Parallel_1"), p2 = par("Standard_Parallel_2");
  if (!sph || lat0 == null || lon0 == null || p1 == null || p2 == null) return null;
  const a = Number(sph[1]), invF = Number(sph[2]);
  const f = invF > 0 ? 1 / invF : 0;
  const e = Math.sqrt(2 * f - f * f);
  const m = (phi: number) => Math.cos(phi) / Math.sqrt(1 - e * e * Math.sin(phi) ** 2);
  const t = (phi: number) => Math.tan(Math.PI / 4 - phi / 2) / Math.pow((1 - e * Math.sin(phi)) / (1 + e * Math.sin(phi)), e / 2);
  const f1 = p1 * D2R, f2 = p2 * D2R, f0 = lat0 * D2R;
  const n = Math.abs(p1 - p2) < 1e-9 ? Math.sin(f1) : (Math.log(m(f1)) - Math.log(m(f2))) / (Math.log(t(f1)) - Math.log(t(f2)));
  const F = m(f1) / (n * Math.pow(t(f1), n));
  const rho0 = a * F * Math.pow(t(f0), n);
  return (p: LL) => {
    const rho = a * F * Math.pow(t(p.lat * D2R), n);
    const th = n * (p.lon - lon0) * D2R;
    return [rho * Math.sin(th), rho0 - rho * Math.cos(th)];
  };
}

export interface GeoModel {
  toPage(p: LL): [number, number];
  /** "lcc" = the file's own projection; "latlon-affine" = corner fit */
  kind: "lcc" | "latlon-affine";
  /** worst corner misfit of the corner fit, page points */
  cornerResidualPt: number;
}

/** Page <- (lat,lon) from the embedded GPTS/LPTS corners: corners are
 *  projected with the file's LCC (when readable) and an affine projected ->
 *  page is fitted over them (least squares). */
export function embeddedGeoModel(g: EmbeddedGeo): GeoModel | null {
  const n = g.gpts.length / 2;
  if (n < 3) return null;
  const [bx0, by0, bx1, by1] = g.bbox;
  const lcc = lccFromWkt(g.wkt);
  const proj = lcc ?? ((p: LL): [number, number] => [p.lon, p.lat]);
  const Q: Array<[number, number]> = [], X: number[] = [], Y: number[] = [];
  for (let i = 0; i < n; i++) {
    Q.push(proj({ lat: g.gpts[i * 2], lon: g.gpts[i * 2 + 1] }));
    X.push(bx0 + g.lpts[i * 2] * (bx1 - bx0));
    Y.push(by0 + g.lpts[i * 2 + 1] * (by1 - by0));
  }
  const cx = lsq3(Q, X), cy = lsq3(Q, Y);
  if (!cx || !cy) return null;
  const toPage = (p: LL): [number, number] => {
    const [u, v] = proj(p);
    return [cx[0] * u + cx[1] * v + cx[2], cy[0] * u + cy[1] * v + cy[2]];
  };
  let res = 0;
  for (let i = 0; i < n; i++) {
    const [x, y] = toPage({ lat: g.gpts[i * 2], lon: g.gpts[i * 2 + 1] });
    res = Math.max(res, Math.hypot(x - X[i], y - Y[i]));
  }
  return { toPage, kind: lcc ? "lcc" : "latlon-affine", cornerResidualPt: res };
}

/** least squares v ≈ a·x + b·y + c */
function lsq3(P: Array<[number, number]>, v: number[]): [number, number, number] | null {
  let sxx = 0, sxy = 0, sx = 0, syy = 0, sy = 0, n = 0, sxv = 0, syv = 0, sv = 0;
  P.forEach(([x, y], i) => { sxx += x * x; sxy += x * y; sx += x; syy += y * y; sy += y; n++; sxv += x * v[i]; syv += y * v[i]; sv += v[i]; });
  const A = [[sxx, sxy, sx], [sxy, syy, sy], [sx, sy, n]];
  const B = [sxv, syv, sv];
  const det = (m: number[][]) => m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  const D = det(A);
  if (Math.abs(D) < 1e-12) return null;
  const col = (k: number) => A.map((row, r) => row.map((val, c) => (c === k ? B[r] : val)));
  return [det(col(0)) / D, det(col(1)) / D, det(col(2)) / D];
}

// ── the whole pipeline ─────────────────────────────────────────────────────

export interface GeorefResult {
  georeferenced: boolean;
  reason: string;
  controlPoints: Array<{ fix: string; pageX: number; pageY: number; lat: number; lon: number; residualNm: number }>;
  rmsNm: number | null;
  maxResidualNm: number | null;
  rejectedCandidates: number;
  scaleNmPerInch: number | null;
  rotationDeg: number | null;
  page: { width: number; height: number; rotate: number };
  /** plan-view crop in PDF points (origin bottom-left) */
  planView: Box | null;
  /** ground corners of the crop, [lon, lat] in MapLibre image order:
   *  top-left, top-right, bottom-right, bottom-left */
  corners: Array<[number, number]> | null;
  embeddedGeoPdf: boolean;
  /** worst distance (nm) between our fit and the embedded GeoPDF over the procedure's fixes */
  embeddedAgreementNm: number | null;
  textItems: number;
  /** how symbols were associated with fixes (the fit itself is always the
   *  least-squares similarity over the associated symbols) */
  method: "geopdf-seeded" | "ransac" | null;
}

/** two independent georeferences further apart than this = neither trusted */
export const EMBEDDED_CONFLICT_NM = 1.0;

export function georeferencePlate(pdf: Buffer, fixes: Map<string, LL>, center: LL): GeorefResult {
  const base: GeorefResult = {
    georeferenced: false, reason: "", controlPoints: [], rmsNm: null, maxResidualNm: null, rejectedCandidates: 0,
    scaleNmPerInch: null, rotationDeg: null, page: { width: 0, height: 0, rotate: 0 }, planView: null, corners: null,
    embeddedGeoPdf: false, embeddedAgreementNm: null, textItems: 0, method: null,
  };
  let objs: Map<number, PdfObject>;
  try { objs = parsePdfObjects(pdf); } catch (e: unknown) {
    return { ...base, reason: `the PDF could not be read (${e instanceof Error ? e.message : String(e)})` };
  }
  const page = firstPage(objs);
  if (!page) return { ...base, reason: "no page found in the PDF" };
  base.page = { width: page.width, height: page.height, rotate: page.rotate };
  base.embeddedGeoPdf = page.geo.length > 0;
  if (page.rotate % 360 !== 0) return { ...base, reason: `page is rotated ${page.rotate}° (not handled — side viewer only)` };
  const { texts, segments, marks } = interpretContent(page.contents);
  base.textItems = texts.length;
  if (!texts.length) return { ...base, reason: "the PDF carries no extractable text (outlined glyphs) — side viewer only" };
  const labels = fixLabels(texts, fixes);
  const distinct = new Set(labels.map((c) => c.fix)).size;
  if (distinct < GEOREF_MIN_POINTS) {
    return { ...base, reason: `only ${distinct} procedure fix label(s) found on the chart (need ${GEOREF_MIN_POINTS})` };
  }
  const eg = page.geo.length ? embeddedGeoModel(page.geo[0]) : null;
  const cands = symbolCandidates(labels, clusterSymbols(marks), fixes, center, eg ? LABEL_SYMBOL_K_SEEDED : LABEL_SYMBOL_K);
  // the embedded GeoPDF (when present) seeds the symbol association; without
  // it, blind RANSAC must find one more agreeing fix to rule out coincidence
  let seed: Similarity | null = null;
  if (eg) {
    const pts = Array.from(fixes.values());
    seed = fitSimilarity(pts.map((f) => eg.toPage(f)), pts.map((f) => toLocalNm(f, center)));
  }
  const minPoints = seed ? GEOREF_MIN_POINTS : GEOREF_MIN_POINTS_UNSEEDED;
  base.method = seed ? "geopdf-seeded" : "ransac";
  const fit = robustFit(cands, seed);
  if (!fit) return { ...base, reason: "no consistent scale/orientation among the labelled fix symbols (chart is likely NOT TO SCALE)" };
  const T = fit.T;
  const cps = fit.inliers.map((c, i) => {
    const g = fromLocalNm(c.ground[0], c.ground[1], center);
    return { fix: c.fix, pageX: round2(c.page[0]), pageY: round2(c.page[1]), lat: g.lat, lon: g.lon, residualNm: round3(fit.residualsNm[i]) };
  });
  const box = planViewBox(segments, fit.inliers.map((c) => c.page), page);
  const toLL = (x: number, y: number) => { const [gx, gy] = simApply(T, x, y); return fromLocalNm(gx, gy, center); };
  const corners: Array<[number, number]> = [
    toLL(box.x0, box.y1), toLL(box.x1, box.y1), toLL(box.x1, box.y0), toLL(box.x0, box.y0),
  ].map((p) => [round6(p.lon), round6(p.lat)]);
  let agreement: number | null = null;
  if (eg) {
    // worst distance, over the procedure fixes drawn inside the plan view,
    // between where the embedded GeoPDF and our symbol fit put them (nm)
    const invT = simInverse(T);
    const inBox = (x: number, y: number) => x >= box.x0 && x <= box.x1 && y >= box.y0 && y <= box.y1;
    const probe = Array.from(fixes.values()).filter((f) => { const [x, y] = eg.toPage(f); return inBox(x, y); });
    agreement = 0;
    for (const f of probe.length ? probe : fit.inliers.map((c) => fromLocalNm(c.ground[0], c.ground[1], center))) {
      const [ex, ey] = eg.toPage(f);
      const [gx, gy] = toLocalNm(f, center);
      const [fx, fy] = simApply(invT, gx, gy);
      agreement = Math.max(agreement, Math.hypot(ex - fx, ey - fy) * simScale(T));
    }
    agreement = round3(agreement);
  }
  const conflict = agreement != null && agreement > EMBEDDED_CONFLICT_NM;
  const ok = fit.inliers.length >= minPoints && fit.rmsNm < GEOREF_MAX_RMS_NM && !conflict;
  return {
    ...base,
    georeferenced: ok,
    reason: ok
      ? `${fit.inliers.length} fix symbols fit a north-up similarity at RMS ${fit.rmsNm.toFixed(2)} nm` +
        (agreement != null ? `; the chart's embedded GeoPDF agrees within ${agreement.toFixed(2)} nm` : "")
      : fit.inliers.length < minPoints
        ? `only ${fit.inliers.length} consistent fix symbols (need ${minPoints})`
        : conflict
          ? `symbol fit and the chart's embedded GeoPDF disagree by ${(agreement ?? 0).toFixed(2)} nm (> ${EMBEDDED_CONFLICT_NM} nm)`
          : `fit residual RMS ${fit.rmsNm.toFixed(2)} nm exceeds ${GEOREF_MAX_RMS_NM} nm`,
    controlPoints: cps,
    rmsNm: round3(fit.rmsNm),
    maxResidualNm: round3(Math.max(...fit.residualsNm)),
    rejectedCandidates: fit.rejected,
    scaleNmPerInch: round3(simScale(T) * 72),
    rotationDeg: round3(simRotationDeg(T)),
    planView: { x0: round2(box.x0), y0: round2(box.y0), x1: round2(box.x1), y1: round2(box.y1) },
    corners: ok ? corners : null,
    embeddedAgreementNm: agreement,
  };
}

const round2 = (x: number) => Math.round(x * 100) / 100;
const round3 = (x: number) => Math.round(x * 1000) / 1000;
const round6 = (x: number) => Math.round(x * 1e6) / 1e6;
