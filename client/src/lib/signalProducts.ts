// signalProducts.ts — pure helpers for the Signals product hub
// (client/src/pages/validatedSignals.tsx). No fetch, no DOM: the relative
// freshness stamp every card shows (PREMIUM EXPERIENCE STANDARD (c): every
// number carries its age) and the license-posture label, both unit-tested.

export type Resell = "ok" | "share-alike" | "conditional" | "unknown";

/** Relative age of an ISO stamp, or "no reading yet" for null. Never throws on bad input. */
export function relativeAge(iso: string | null | undefined, now: number = Date.now()): string {
  if (!iso) return "no reading yet";
  const t = Date.parse(iso);
  if (!Number.isFinite(t)) return "unknown age";
  const sec = Math.max(0, Math.round((now - t) / 1000));
  if (sec < 60) return "just now";
  const min = Math.round(sec / 60);
  if (min < 60) return `${min}m ago`;
  const h = sec / 3600;
  if (h < 48) return `${h.toFixed(h < 10 ? 1 : 0)}h ago`;
  return `${Math.round(h / 24)}d ago`;
}

export interface Posture { label: string; cls: "ok" | "share" | "cond" | "unknown"; blurb: string }

/** Customer-facing wording for LICENSE_MARKS.resell — same three values the /developers license table uses. */
export function licensePosture(resell: Resell | string): Posture {
  switch (resell) {
    case "ok": return { label: "freely resellable", cls: "ok", blurb: "public-domain or CC0 lineage — redistribution allowed with attribution" };
    case "share-alike": return { label: "share-alike", cls: "share", blurb: "ODbL lineage — derived products must carry the same license" };
    case "conditional": return { label: "conditional", cls: "cond", blurb: "informational-use source terms — redistribution needs separate review" };
    default: return { label: "license unresolved", cls: "unknown", blurb: "no license mark on record — do not redistribute" };
  }
}

/** The one-line GET a customer pastes; `base` defaults to the current origin in the page. */
export function curlFor(base: string, v1Path: string): string {
  return `curl -H "x-api-key: $KEY" ${base.replace(/\/$/, "")}${v1Path}`;
}
