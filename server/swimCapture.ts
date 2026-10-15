// swimCapture.ts — capture-first consumers for the SCDS products that have
// no parser yet: TFMS (flow programs, reroutes, advisories) and NOTAM (FNS
// digital NOTAMs, incl. TFR geometry). Ops item 1, human-directed 2026-10-14.
//
// WHY CAPTURE FIRST: the SFDPS consumer shipped with FIXM element names
// assumed offline and its first live deploy processed 0 of 39,472 messages
// (experiments.md 2026-09-28: the body sat in a section nobody read). The
// exact TFMS/NOTAM schemas reaching OUR queues are not knowable from here,
// so the parsers that feed the deviation classifier are written against
// REAL messages, not guesses. This module:
//   - connects each configured product through swimConnector.ts (the same
//     transport, acking, backoff, redaction and per-hour counters as SFDPS);
//   - profiles every message by a content-free SHAPE: the first opening
//     tags (local names) plus type-like attribute values — what kinds of
//     messages arrive and how often;
//   - keeps the 2 latest raw samples per shape (bounded), readable only
//     behind the diag token (the products are NDRB pre-approved for public
//     release, but TFMS flight data can include LADD/PIA flights — samples
//     are never served publicly).
// STDDS and ITWS stay configured-only (directive): STDDS is surface/terminal
// radar volume with no consumer need yet; ITWS lacks its queue settings.

import type { Express } from "express";
import { startSwimProduct, swimProductStatus, swimProductsEnvStatus, SWIM_PRODUCTS, type SwimConnectorHandle } from "./swimConnector";
import { checkDiagToken, diagEnabled } from "./diag";
import { TfmsStore } from "./swimTfms";
import { NotamStore, TFR_TEXT_RE } from "./swimNotam";

// TFMS parser (written against the captured samples): every TFMS payload is
// profiled AND parsed into programs / airport configs / per-flight
// amendments. One process-wide store; the deviation classifier reads it.
let tfmsStore: TfmsStore | null = null;
export function sharedTfmsStore(): TfmsStore { return (tfmsStore ??= new TfmsStore()); }
let notamStore: NotamStore | null = null;
export function sharedNotamStore(): NotamStore { return (notamStore ??= new NotamStore()); }

export const CAPTURE_PRODUCTS = ["TFMS", "NOTAM"] as const;
export type CaptureProduct = typeof CAPTURE_PRODUCTS[number];
export const CAPTURE_MAX_SHAPES = 60;
export const CAPTURE_SAMPLES_PER_SHAPE = 2;
export const CAPTURE_SAMPLE_MAX_BYTES = 96 * 1024;
/** opening tags that make a shape (deep enough to reach a message type) */
export const SHAPE_TAGS = 6;

const OPEN_TAG_RE = /<([A-Za-z_][\w.-]*:)?([A-Za-z_][\w.-]*)((?:\s+[^\s=/>]+\s*=\s*(?:"[^"]*"|'[^']*'))*)\s*\/?>/g;
// type-like attributes carry the message kind (msgType="...", xsi:type="...")
const TYPE_ATTR_RE = /(?:^|\s)(?:[\w.-]+:)?(msgType|messageType|type|eventType|scenario|fiType)\s*=\s*(?:"([^"]{1,48})"|'([^']{1,48})')/gi;

/** Content-free structural signature: the first SHAPE_TAGS opening tags
 *  (local names) with type-like attribute values, e.g.
 *  `tfmDataService>fiOutput>fiMessage[msgType=ADVISORY]>…`. Never element
 *  text, never ids, never other attribute values. */
export function xmlShape(payload: string, tags = SHAPE_TAGS): string {
  if (typeof payload !== "string" || !payload) return "(empty)";
  const head = payload.slice(0, 4096);
  const parts: string[] = [];
  OPEN_TAG_RE.lastIndex = 0;
  let m: RegExpExecArray | null;
  while (parts.length < tags && (m = OPEN_TAG_RE.exec(head))) {
    const types: string[] = [];
    TYPE_ATTR_RE.lastIndex = 0;
    let a: RegExpExecArray | null;
    while ((a = TYPE_ATTR_RE.exec(m[3] || ""))) {
      const v = (a[2] ?? a[3] ?? "").replace(/^[\w.-]+:/, ""); // xsi:type="ns:Foo" -> Foo
      types.push(`${a[1]}=${v}`);
    }
    parts.push(types.length ? `${m[2]}[${types.join(",")}]` : m[2]);
  }
  return parts.length ? parts.join(">") : "(not xml)";
}

interface ShapeState {
  count: number;
  firstAt: number;
  lastAt: number;
  maxBytes: number;
  samples: Array<{ at: number; bytes: number; truncated: boolean; payload: string }>;
}

export class SwimCapture {
  private readonly shapes = new Map<string, ShapeState>();
  /** messages whose shape arrived after CAPTURE_MAX_SHAPES were taken */
  overflow = 0;
  total = 0;

  constructor(readonly product: string) {}

  /** `bucket` files the message under its own label (e.g. "TFR"), so a rare
   *  kind keeps samples of its own instead of sharing a common shape's two */
  observe(payload: string, now: number, bucket?: string): void {
    this.total++;
    const sig = bucket ? `${bucket}:${xmlShape(payload)}` : xmlShape(payload);
    let st = this.shapes.get(sig);
    if (!st) {
      if (this.shapes.size >= CAPTURE_MAX_SHAPES) { this.overflow++; return; }
      st = { count: 0, firstAt: now, lastAt: now, maxBytes: 0, samples: [] };
      this.shapes.set(sig, st);
    }
    st.count++;
    st.lastAt = now;
    const bytes = payload.length;
    st.maxBytes = Math.max(st.maxBytes, bytes);
    st.samples.push({
      at: now, bytes, truncated: bytes > CAPTURE_SAMPLE_MAX_BYTES,
      payload: bytes > CAPTURE_SAMPLE_MAX_BYTES ? payload.slice(0, CAPTURE_SAMPLE_MAX_BYTES) : payload,
    });
    if (st.samples.length > CAPTURE_SAMPLES_PER_SHAPE) st.samples.shift();
  }

  /** public, content-free */
  shapesView(top = 30) {
    return {
      total: this.total,
      distinct: this.shapes.size,
      overflow: this.overflow,
      shapes: Array.from(this.shapes.entries())
        .sort((a, b) => b[1].count - a[1].count)
        .slice(0, top)
        .map(([sig, s]) => ({ sig, count: s.count, firstAt: s.firstAt, lastAt: s.lastAt, maxBytes: s.maxBytes })),
    };
  }

  /** raw samples (diag-token only) — one shape, or every shape */
  samplesView(sig?: string) {
    const out: Array<{ sig: string; at: number; bytes: number; truncated: boolean; payload: string }> = [];
    for (const [k, s] of Array.from(this.shapes.entries())) {
      if (sig && k !== sig) continue;
      for (const x of s.samples) out.push({ sig: k, ...x });
    }
    return out;
  }
}

const captures = new Map<string, SwimCapture>();
export function swimCapture(product: string): SwimCapture | null { return captures.get(product) ?? null; }

/** Start the capture consumers (each a no-op unless its env is complete). */
export async function startSwimCapture(o: {
  products?: readonly string[];
  env?: NodeJS.ProcessEnv;
  importer?: (name: string) => Promise<unknown>;
  log?: (m: string) => void;
  now?: () => number;
} = {}): Promise<SwimConnectorHandle[]> {
  const handles: SwimConnectorHandle[] = [];
  for (const product of o.products ?? CAPTURE_PRODUCTS) {
    const cap = captures.get(product) ?? new SwimCapture(product);
    captures.set(product, cap);
    handles.push(await startSwimProduct({
      product, envPrefix: `SWIM_${product}`, env: o.env, importer: o.importer, log: o.log, now: o.now,
      onPayload: product === "TFMS"
        ? (payload, now) => { cap.observe(payload, now); sharedTfmsStore().ingest(payload, now); }
        : product === "NOTAM"
          ? (payload, now) => {
            cap.observe(payload, now, TFR_TEXT_RE.test(payload) ? "TFR" : undefined);
            sharedNotamStore().ingest(payload, now);
          }
          : (payload, now) => cap.observe(payload, now),
    }));
  }
  return handles;
}

/** test seam */
export function _resetSwimCaptureForTests(): void { captures.clear(); tfmsStore = null; notamStore = null; }

/** GET /api/data/swim/products (public: every SCDS product's env readiness
 *  by NAME, transport status, per-hour cadence and the content-free shape
 *  profile) and GET /api/data/swim/samples (diag-token: raw samples to write
 *  the parsers against). */
export function registerSwimCaptureRoutes(app: Express, env: NodeJS.ProcessEnv = process.env): void {
  app.get("/api/data/swim/products", (_req, res) => {
    const envStatus = swimProductsEnvStatus(env);
    const products: Record<string, unknown> = {};
    for (const p of SWIM_PRODUCTS) {
      const s = swimProductStatus(p);
      const cap = captures.get(p);
      products[p] = {
        env: envStatus[p],
        consumer: p === "SFDPS" ? "flight plans (swimSfdps.ts)" : cap ? "capture (shape profile + samples)" : "configured-only",
        connected: s.connected, messagesReceived: s.messagesReceived, messagesProcessed: s.messagesProcessed,
        processErrors: s.processErrors, droppedOverflow: s.droppedOverflow, reconnects: s.reconnects,
        stallRestarts: s.stallRestarts, lastMessageAt: s.lastMessageAt, lastError: s.lastError, hourly: s.hourly,
        shapes: cap ? cap.shapesView() : null,
        ...(p === "TFMS" ? { parsed: sharedTfmsStore().stats() } : {}),
        ...(p === "NOTAM" ? { parsed: sharedNotamStore().stats() } : {}),
      };
    }
    res.set("Cache-Control", "no-store");
    res.json({
      at: Date.now(), products,
      note: "Env readiness is reported by variable NAME only. Shapes are content-free (tag names + type "
        + "attributes); raw samples are diag-token only. STDDS and ITWS are configured-only by design.",
    });
  });

  // active flow programs (GDP / ground stop / AFP / restrictions / ATCSCC
  // advisories / flow-constrained areas) + live airport runway configs and
  // rates. Per-flight amendments are NOT served here (the classifier reads
  // them in-process; flight-level data stays behind the LADD gate).
  app.get("/api/data/swim/tfms/active", (_req, res) => {
    const st = sharedTfmsStore();
    const now = Date.now();
    st.evict(now);
    res.set("Cache-Control", "public, max-age=30");
    res.json({
      at: now, ...st.active(now), stats: st.stats(),
      source: "FAA SWIM TFMS (SCDS, NDRB pre-approved for public release; not for operational use)",
    });
  });

  // active TFRs (FAA FNS digital NOTAMs; FDC TFR rules or Q-code QRT):
  // exact polygon when the NOTAM carries airspace geometry, else its Q-line
  // circle labeled approximate
  app.get("/api/data/swim/notam/tfrs", (_req, res) => {
    const st = sharedNotamStore();
    const now = Date.now();
    st.evict(now);
    res.set("Cache-Control", "public, max-age=60");
    res.json({
      at: now, tfrs: st.active(now), stats: st.stats(),
      source: "FAA SWIM FNS NOTAM (SCDS, NDRB pre-approved for public release; not for operational use)",
      note: "geometry.kind 'circle' with approximate:true is the NOTAM Q-line circle, which ENCLOSES the area; only 'polygon' is the restriction's own shape.",
    });
  });

  app.get("/api/data/swim/samples", (req, res) => {
    if (!diagEnabled(env)) return res.status(404).end();
    const token = String((req.query as Record<string, unknown>).token ?? "");
    if (!checkDiagToken(env, token)) return res.status(404).end();
    const product = String((req.query as Record<string, unknown>).product ?? "").toUpperCase();
    const cap = captures.get(product);
    if (!cap) return res.status(400).json({ error: `product must be one of ${CAPTURE_PRODUCTS.join(", ")}` });
    const sigQ = (req.query as Record<string, unknown>).sig;
    res.set("Cache-Control", "no-store");
    res.json({ product, samples: cap.samplesView(typeof sigQ === "string" && sigQ ? sigQ : undefined) });
  });
}
