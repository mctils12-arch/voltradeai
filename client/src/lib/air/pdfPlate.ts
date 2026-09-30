// PLATE RENDERER — FAA d-TPP PDFs rendered in the browser with pdf.js,
// lazily loaded from cdnjs ONLY when a user opens a plate (zero cost
// otherwise).
//
// WHY CLIENT-SIDE (decision recorded 2026-09-30): the server has no PDF
// rasteriser — no poppler/mupdf in the frozen Dockerfile image and no pure-JS
// renderer among our npm deps (and no new deps allowed). pdf.js is Mozilla's
// reference renderer; a DOCUMENT viewer is not a raster TILE layer, so Law
// II's "tiles from our CDN" does not apply — and the PDF itself still comes
// from OUR server (/api/data/plates/…, proxied + cached), never from FAA at
// runtime. The script is pinned to one version with SRI; the worker is
// fetched with the same integrity check and started from a blob: URL (CSP
// worker-src allows 'self' blob:, not third-party origins).
//
// Every render is abortable (AbortSignal -> loadingTask.destroy()) and
// bounded (maxPx on the longest side).

import { cropPixelRect, overlayScale } from "./procedures.js";

export const PDFJS_VERSION = "3.11.174";
const CDN = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${PDFJS_VERSION}`;
export const PDFJS_SRC = `${CDN}/pdf.min.js`;
export const PDFJS_WORKER_SRC = `${CDN}/pdf.worker.min.js`;
// sha384 of the exact cdnjs files (computed 2026-09-30); a changed file is refused
export const PDFJS_SRI = "sha384-/1qUCSGwTur9vjf/z9lmu/eCUYbpOTgSjmpbMQZ1/CtX2v/WcAIKqRv+U1DUCG6e";
export const PDFJS_WORKER_SRI = "sha384-SnzOobpRMLXZ52iJvZm/C0fYw0OQemTXzTjIsdsfMcrCtCEe9qgzxTd3RSklO5x2";

interface PdfViewport { width: number; height: number }
interface PdfPage {
  getViewport(o: { scale: number }): PdfViewport;
  render(o: { canvasContext: CanvasRenderingContext2D; viewport: PdfViewport; transform?: number[] }): { promise: Promise<void>; cancel(): void };
  view: number[];
}
interface PdfDoc { getPage(n: number): Promise<PdfPage>; destroy(): Promise<void> }
interface PdfLoadingTask { promise: Promise<PdfDoc>; destroy(): Promise<void> }
interface PdfJs {
  getDocument(src: { url: string; withCredentials?: boolean } | { data: ArrayBuffer }): PdfLoadingTask;
  GlobalWorkerOptions: { workerSrc: string };
}

let loading: Promise<PdfJs> | null = null;

/** Load pdf.js once (script tag with SRI) and point it at a blob: worker. */
export function loadPdfJs(): Promise<PdfJs> {
  if (loading) return loading;
  loading = (async () => {
    const w = window as unknown as { pdfjsLib?: PdfJs };
    if (!w.pdfjsLib) {
      await new Promise<void>((resolve, reject) => {
        const s = document.createElement("script");
        s.src = PDFJS_SRC;
        s.integrity = PDFJS_SRI;
        s.crossOrigin = "anonymous";
        s.async = true;
        s.onload = () => resolve();
        s.onerror = () => reject(new Error("pdf.js failed to load"));
        document.head.appendChild(s);
      });
    }
    const lib = w.pdfjsLib;
    if (!lib) throw new Error("pdf.js loaded but did not register");
    const r = await fetch(PDFJS_WORKER_SRC, { integrity: PDFJS_WORKER_SRI, mode: "cors" });
    if (!r.ok) throw new Error(`pdf.js worker HTTP ${r.status}`);
    lib.GlobalWorkerOptions.workerSrc = URL.createObjectURL(new Blob([await r.text()], { type: "text/javascript" }));
    return lib;
  })();
  // a failed load may be retried on the next open
  loading.catch(() => { loading = null; });
  return loading;
}

export interface RenderOpts {
  /** crop to this box (PDF points, origin bottom-left) — the plan view */
  crop?: { x0: number; y0: number; x1: number; y1: number } | null;
  /** longest side of the output, px */
  maxPx: number;
  signal?: AbortSignal;
}

const aborted = (s?: AbortSignal) => { if (s?.aborted) throw new DOMException("aborted", "AbortError"); };

/** Render page 1 of the plate (optionally cropped) to a canvas. */
export async function renderPlate(url: string, o: RenderOpts): Promise<HTMLCanvasElement> {
  const lib = await loadPdfJs();
  aborted(o.signal);
  const task = lib.getDocument({ url, withCredentials: false });
  const onAbort = () => { void task.destroy(); };
  o.signal?.addEventListener("abort", onAbort, { once: true });
  try {
    const doc = await task.promise;
    aborted(o.signal);
    const page = await doc.getPage(1);
    const vb = page.view; // [x0, y0, x1, y1] in points
    const pageW = vb[2] - vb[0], pageH = vb[3] - vb[1];
    const crop = o.crop ?? { x0: 0, y0: 0, x1: pageW, y1: pageH };
    const scale = overlayScale(crop, o.maxPx);
    const viewport = page.getViewport({ scale });
    const rect = cropPixelRect(crop, pageH, scale);
    const canvas = document.createElement("canvas");
    canvas.width = rect.w;
    canvas.height = rect.h;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("no 2d context");
    const job = page.render({ canvasContext: ctx, viewport, transform: [1, 0, 0, 1, -rect.x, -rect.y] });
    const cancel = () => job.cancel();
    o.signal?.addEventListener("abort", cancel, { once: true });
    try { await job.promise; } finally { o.signal?.removeEventListener("abort", cancel); }
    aborted(o.signal);
    await doc.destroy();
    return canvas;
  } finally {
    o.signal?.removeEventListener("abort", onAbort);
  }
}
