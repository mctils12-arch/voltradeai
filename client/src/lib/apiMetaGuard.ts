// Shape guard for /api/v1/meta responses (PLATFORM INTEGRATION P2 hardening).
//
// WHY: /developers renders straight off the live meta document — by design it
// never hardcodes a response. The cost of that honesty is that whatever the
// fetch yields BECOMES the page: a JSON-shaped error body (our own routes
// return `{error: ...}` JSON on 4xx/5xx) or a version-skewed older meta
// (pre-P2 servers had no `limits` field) used to flow into setMeta unchecked,
// and the first `meta.limits[tier].perMinute` crashed the whole public page
// to the error boundary. Same bug class as the satellite layer's unchecked
// CelesTrak 403 (fixed v1.0.252): a non-ok response parsed as data.
//
// Pure TS, zero React/DOM deps — unit-testable under `tsx --test`.

export interface MetaEndpoint { path: string; params: string; desc: string; preview?: string }
export interface ApiMetaDoc {
  version: string;
  auth: string;
  endpoints: MetaEndpoint[];
  coming_gated: string[];
  limits: Record<string, { perMinute: number; perDay: number }>;
  license_marks: Record<string, { license: string; attribution: string; resell: string }>;
  disclaimer: string;
}

/** True when `lim` is a usable per-tier limit entry. */
function isLimit(lim: unknown): lim is { perMinute: number; perDay: number } {
  if (lim == null || typeof lim !== "object") return false;
  const o = lim as Record<string, unknown>;
  return typeof o.perMinute === "number" && Number.isFinite(o.perMinute) &&
         typeof o.perDay === "number" && Number.isFinite(o.perDay);
}

/**
 * Validate a fetched /api/v1/meta body. Returns the typed doc when it has the
 * structure the page renders (endpoints array + at least one valid tier limit
 * + license_marks object), null for anything else — error bodies, older
 * skewed meta versions, or garbage. Null means "show the designed
 * 'reference unavailable' state", never a crash.
 *
 * `ok` is the response's res.ok — a non-2xx is rejected up front so a
 * JSON-shaped error body can't masquerade as documentation.
 */
export function parseApiMetaDoc(ok: boolean, body: unknown): ApiMetaDoc | null {
  if (!ok) return null;
  if (body == null || typeof body !== "object" || Array.isArray(body)) return null;
  const o = body as Record<string, unknown>;
  if (!Array.isArray(o.endpoints)) return null;
  if (o.limits == null || typeof o.limits !== "object" || Array.isArray(o.limits)) return null;
  if (!Object.values(o.limits as object).some(isLimit)) return null;
  if (o.license_marks == null || typeof o.license_marks !== "object" || Array.isArray(o.license_marks)) return null;
  return body as ApiMetaDoc;
}

/** Per-tier limit lookup that tolerates a missing tier (renders a fallback
 *  instead of throwing — a future tier rename must not take the page down). */
export function tierLimit(
  meta: ApiMetaDoc,
  tier: string,
): { perMinute: number; perDay: number } | null {
  const lim = meta.limits[tier];
  return isLimit(lim) ? lim : null;
}
