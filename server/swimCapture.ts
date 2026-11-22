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
import { NotamStore, TFR_TEXT_RE, type TfrRecord } from "./swimNotam";
import { trackMap } from "./heapDiag";
import { createSwimKeeper, type SwimKeeper, type SwimKeepView } from "./swimStoreKeep";
import { sharedFaaTfrList, faaTfrListEnabled } from "./faaTfrList";
import { faaTfrListedSince, faaTfrArchiveView } from "./faaTfrArchive";

// TFMS parser (written against the captured samples): every TFMS payload is
// profiled AND parsed into programs / airport configs / per-flight
// amendments. One process-wide store; the deviation classifier reads it.
let tfmsStore: TfmsStore | null = null;
export function sharedTfmsStore(): TfmsStore { return (tfmsStore ??= new TfmsStore()); }
let notamStore: NotamStore | null = null;
export function sharedNotamStore(): NotamStore { return (notamStore ??= new NotamStore()); }

// The rows of both stores that are still in force are kept across a restart
// (swimStoreKeep.ts; open_questions 2026-11-20 (t6)). One keeper per process,
// started only where a feed is configured, so a process with no FAA feed (a
// test, a local run) neither reads nor writes the file.
let keeper: SwimKeeper | null = null;
export function startSwimStoreKeep(o: { env?: NodeJS.ProcessEnv; file?: string; intervalMs?: number; log?: (m: string) => void; now?: () => number } = {}): SwimKeeper | null {
  if (keeper) return keeper;
  const status = swimProductsEnvStatus(o.env ?? process.env);
  if (!status.TFMS?.configured && !status.NOTAM?.configured) return null;
  keeper = createSwimKeeper(sharedTfmsStore(), sharedNotamStore(), { file: o.file, intervalMs: o.intervalMs, log: o.log, now: o.now });
  keeper.start();
  return keeper;
}
/** What the public routes say about the keeping; null when no keeper runs
 *  (no feed configured), so a page keeps its "restarts with the server" words. */
export function swimKeepView(): SwimKeepView | null { return keeper ? keeper.view() : null; }

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
trackMap("swimCapture.captures", captures); // memory audit 2026-11-03 A0
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
export function _resetSwimCaptureForTests(): void { captures.clear(); tfmsStore = null; notamStore = null; keeper?.stop(); keeper = null; }

/** The keeping as one route's answer states it: whether the list survives a
 *  restart, what this process put back at boot (this route's rows only) and
 *  when the file was last written. Counts and times, no text. null = the
 *  list is NOT kept (no keeper, or switched off). */
export function keptFor(which: "tfms" | "notam", v: SwimKeepView | null = swimKeepView()) {
  if (!v || !v.enabled) return null;
  const r = v.restored;
  return {
    saved_every_s: v.interval_s,
    last_saved_at: v.last_saved_at,
    last_save: v.last_save,
    restored: r ? {
      at: r.at, saved_at: r.saved_at, cause: r.cause,
      ...(which === "tfms" ? { programs: r.programs, airport_configs: r.configs } : { tfrs: r.tfrs }),
    } : null,
  };
}

/** GET /api/data/swim/products (public: every SCDS product's env readiness
 *  by NAME, transport status, per-hour cadence and the content-free shape
 *  profile) and GET /api/data/swim/samples (diag-token: raw samples to write
 *  the parsers against). */
/** What GET /api/data/swim/notam/tfrs answers: the notice feed's rows and the
 *  rows read from the FAA's public list (faaTfrList.ts), each labelled with
 *  where it came from. ONE ROW PER NOTICE: when both hold an FDC notice, the
 *  FAA's published outline is served and the feed's row is left out, unless
 *  the feed's row carries airspace geometry as sent (then the feed's stays).
 *  `faa_list` is the coverage check against the FAA's list at its last read. */
export function flightRestrictionsAnswer(now: number, env: NodeJS.ProcessEnv = process.env) {
  const st = sharedNotamStore();
  st.evict(now);
  const list = sharedFaaTfrList();
  const feedRows = st.active(now);
  const sentGeometry = (t: TfrRecord) => t.geometry?.kind === "polygon" && !("from" in t.geometry);
  const keptFeed = new Set(feedRows.filter(sentGeometry).map((t) => t.citation).filter((c): c is string => !!c));
  const listCites = list.citations(now);
  const fromFeed = feedRows.filter((t) => !t.citation || !listCites.has(t.citation) || sentGeometry(t));
  // since when our record of the list has held each listed notice (faaTfrArchive.ts); absent = not in the record
  const fromList = list.active(now).filter((r) => !keptFeed.has(r.citation)).map((r) => {
    const since = faaTfrListedSince(r.number);
    return since ? { ...r, listed_since: since } : r;
  });
  const feed = swimProductStatus("NOTAM");
  return {
    at: now, tfrs: [...fromFeed, ...fromList], stats: st.stats(), kept: keptFor("notam"),
    // how old the list is (RENDERING & MOTION LAW V): whether the notice
    // feed is connected and when its newest message arrived (any notice,
    // not only a flight restriction)
    feed: { connected: feed.connected, last_message_at: feed.lastMessageAt },
    faa_list: { ...list.coverage(now, faaTfrListEnabled(env), st.citations()), served: fromList.length, record: faaTfrArchiveView(env) },
    source: "FAA SWIM FNS NOTAM (SCDS, NDRB pre-approved for public release; not for operational use); rows with source 'faa_list' are from the FAA's public list at tfr.faa.gov",
    note: "A row with source 'faa_list' was read from the FAA's public list (tfr.faa.gov), not from the notice feed: its geometry (from:'faa_list') is the FAA's own published outline, its limits are in faa.areas, and `faa_list` says when that list was read and how many listed notices have no published outline. `listed_since` on such a row is our own record of the list (first_seen = our read that first held the notice; backfill true = it was already listed when recording began), and `faa_list.record` is that record's state. Every other row is from the notice feed: geometry.kind 'circle' with approximate:true is the NOTAM Q-line circle, which ENCLOSES the area. 'polygon' is the restriction's own shape as sent, unless it carries from:'text': then the outline was read from the notice's words (the notice sent no geometry), circles and arcs drawn as 5-degree straight sides, and `unread` counts areas the text defines that could not be followed.",
  };
}

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
      at: now, ...st.active(now), stats: st.stats(), kept: keptFor("tfms"),
      source: "FAA SWIM TFMS (SCDS, NDRB pre-approved for public release; not for operational use)",
    });
  });

  // active TFRs (FAA FNS digital NOTAMs; FDC TFR rules or Q-code QRT):
  // exact polygon when the NOTAM carries airspace geometry, else its Q-line
  // circle labeled approximate
  app.get("/api/data/swim/notam/tfrs", (_req, res) => {
    res.set("Cache-Control", "public, max-age=60");
    res.json(flightRestrictionsAnswer(Date.now(), env));
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
